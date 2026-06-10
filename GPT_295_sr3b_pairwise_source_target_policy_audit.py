# -*- coding: utf-8 -*-
"""
SR-3B: Pairwise Source-Target StructuralResolution Policy Audit

Purpose
-------
Train and evaluate StructuralResolution on the corrected object:

    (source_case, target_case) -> pair_action_label / pair_utility_label

Then aggregate pair-level predictions back to target-level reuse policy.

Inputs
------
    sr3a_outputs/sr3a_pairwise_source_target_dataset.csv
    sr1b1_outputs/sr1b1_topk_vim_features.csv
    sr1c_outputs/sr1c_cv_results.csv        optional baseline comparison

Outputs
-------
    sr3b_outputs/sr3b_pair_cv_results.csv
    sr3b_outputs/sr3b_pair_predictions.csv
    sr3b_outputs/sr3b_target_policy_results.csv
    sr3b_outputs/sr3b_target_predictions.csv
    sr3b_outputs/sr3b_policy_utility.csv
    sr3b_outputs/sr3b_summary.json

Core idea
---------
Pairwise model learns:
    source-target contrast -> useful / neutral / harmful
    source-target contrast -> direct / observe / reject

Target-level policy is selected by aggregating scores across source buckets:
    direct_score  = strongest useful/direct evidence from source pairs
    reject_score  = strongest harmful/reject evidence from source pairs
    observe_score = uncertainty / neutral evidence

Main evaluation
---------------
Target-level predictions are compared against:
    gain_oracle_action
    best_policy_action

Key split
---------
    group_by_target_reuse_group
    group_by_target_reuse_group_non_source

This is the first true pairwise SR audit.
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


warnings.filterwarnings("ignore")


BASE_DIR = Path(r"C:\Users\ZH\Desktop\AGI\python_script")

IN_PAIRWISE = BASE_DIR / "sr3a_outputs" / "sr3a_pairwise_source_target_dataset.csv"
IN_TARGETS = BASE_DIR / "sr1b1_outputs" / "sr1b1_topk_vim_features.csv"
IN_SR1C = BASE_DIR / "sr1c_outputs" / "sr1c_cv_results.csv"

OUT_DIR = BASE_DIR / "sr3b_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PAIR_CV = OUT_DIR / "sr3b_pair_cv_results.csv"
OUT_PAIR_PREDS = OUT_DIR / "sr3b_pair_predictions.csv"
OUT_TARGET_POLICY = OUT_DIR / "sr3b_target_policy_results.csv"
OUT_TARGET_PREDS = OUT_DIR / "sr3b_target_predictions.csv"
OUT_UTILITY = OUT_DIR / "sr3b_policy_utility.csv"
OUT_SUMMARY = OUT_DIR / "sr3b_summary.json"


SOURCE_GROUP = "G0_STRICT_SOURCE"

PAIR_TARGETS = [
    "pair_action_label",
    "pair_utility_label",
]

TARGET_EVALS = [
    "gain_oracle_action",
    "best_policy_action",
]

SPLITS = [
    "stratified_pair",
    "group_by_target_concept",
    "group_by_target_task_family",
    "group_by_target_reuse_group",
    "group_by_target_reuse_group_non_source",
]

MODELS = [
    "dummy_most_frequent",
    "logistic_l2",
    "rf_small",
    "extra_trees",
]

PAIR_FEATURE_SETS = [
    "meta_only",
    "geometry_only",
    "geometry_plus_meta",
]


def read_csv(path: Path, required=True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"[LOAD] {path} shape={df.shape}")
    return df


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def get_feature_columns(df: pd.DataFrame, feature_set: str) -> Tuple[List[str], List[str]]:
    meta_cols = [
        "source_bucket",
        "source_reuse_group",
        "target_reuse_group",
        "source_task_family",
        "target_task_family",
        "source_operator",
        "target_operator",
        "source_concept",
        "target_concept",
        "source_gain_oracle_action",
        "target_gain_oracle_action",
        "same_reuse_group",
        "same_task_family",
        "same_operator",
        "same_concept",
        "source_boundary_strength",
        "target_boundary_strength",
        "boundary_strength_delta_target_minus_source",
        "boundary_strength_abs_delta",
    ]

    geometry_cols = [
        c for c in df.columns
        if c.startswith("vec_")
        or c.endswith("_l2")
        or c.endswith("_l1_mean")
        or c.endswith("_cos")
        or c.endswith("_delta_mean")
        or c.endswith("_abs_delta_max")
        or c in [
            "target_norm",
            "source_norm",
            "norm_ratio_target_over_source",
        ]
    ]

    categorical = []
    numeric = []

    if feature_set == "meta_only":
        categorical = [
            "source_bucket",
            "source_reuse_group",
            "target_reuse_group",
            "source_task_family",
            "target_task_family",
            "source_operator",
            "target_operator",
            "source_gain_oracle_action",
        ]
        numeric = [
            "same_reuse_group",
            "same_task_family",
            "same_operator",
            "same_concept",
            "source_boundary_strength",
            "target_boundary_strength",
            "boundary_strength_delta_target_minus_source",
            "boundary_strength_abs_delta",
        ]

    elif feature_set == "geometry_only":
        numeric = geometry_cols

    elif feature_set == "geometry_plus_meta":
        categorical = [
            "source_bucket",
            "source_reuse_group",
            "target_reuse_group",
            "source_task_family",
            "target_task_family",
            "source_operator",
            "target_operator",
            "source_gain_oracle_action",
        ]
        numeric = list(dict.fromkeys([
            "same_reuse_group",
            "same_task_family",
            "same_operator",
            "same_concept",
            "source_boundary_strength",
            "target_boundary_strength",
            "boundary_strength_delta_target_minus_source",
            "boundary_strength_abs_delta",
        ] + geometry_cols))

    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    categorical = [c for c in categorical if c in df.columns]
    numeric = [c for c in numeric if c in df.columns]

    return numeric, categorical


def build_model(df: pd.DataFrame, feature_set: str, model_name: str):
    numeric, categorical = get_feature_columns(df, feature_set)

    transformers = []

    if numeric:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, numeric))

    if categorical:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", make_ohe()),
        ])
        transformers.append(("cat", cat_pipe, categorical))

    if not transformers:
        raise ValueError(f"No features for {feature_set}")

    pre = ColumnTransformer(transformers=transformers, remainder="drop")

    if model_name == "dummy_most_frequent":
        clf = DummyClassifier(strategy="most_frequent")

    elif model_name == "logistic_l2":
        clf = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
        )

    elif model_name == "rf_small":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=7,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )

    elif model_name == "extra_trees":
        clf = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=7,
            min_samples_leaf=3,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([("pre", pre), ("clf", clf)])


def get_pair_splits(pair_df: pd.DataFrame, target: str, split_name: str):
    y = pair_df[target].astype(str).values

    if split_name == "stratified_pair":
        vc = pd.Series(y).value_counts()
        if vc.min() < 2:
            return []
        n_splits = int(min(5, vc.min()))
        n_splits = max(2, n_splits)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return list(splitter.split(pair_df, y))

    if split_name == "group_by_target_concept":
        groups = pair_df["target_concept"].astype(str).values
        if len(np.unique(groups)) < 2:
            return []
        splitter = GroupKFold(n_splits=int(min(5, len(np.unique(groups)))))
        return list(splitter.split(pair_df, y, groups))

    if split_name == "group_by_target_task_family":
        groups = pair_df["target_task_family"].astype(str).values
        if len(np.unique(groups)) < 2:
            return []
        splitter = GroupKFold(n_splits=int(min(5, len(np.unique(groups)))))
        return list(splitter.split(pair_df, y, groups))

    if split_name == "group_by_target_reuse_group":
        groups = pair_df["target_reuse_group"].astype(str).values
        if len(np.unique(groups)) < 2:
            return []
        splitter = GroupKFold(n_splits=int(min(5, len(np.unique(groups)))))
        return list(splitter.split(pair_df, y, groups))

    if split_name == "group_by_target_reuse_group_non_source":
        splits = []
        all_idx = np.arange(len(pair_df))
        for group in sorted(pair_df["target_reuse_group"].astype(str).unique()):
            if group == SOURCE_GROUP:
                continue
            te = np.where(pair_df["target_reuse_group"].astype(str).values == group)[0]
            tr = np.setdiff1d(all_idx, te)
            if len(te) > 0 and len(tr) > 0:
                splits.append((tr, te))
        return splits

    raise ValueError(f"Unknown split: {split_name}")


def score_positive_class(model, X, positive_label: str):
    """
    Return score for positive_label if possible.
    Works with predict_proba; fallback to binary prediction equality.
    """
    clf = model.named_steps["clf"]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes = list(clf.classes_)
        if positive_label in classes:
            return proba[:, classes.index(positive_label)]
        return np.zeros(len(X))

    pred = model.predict(X)
    return (pred.astype(str) == str(positive_label)).astype(float)


def run_pair_cv(pair_df: pd.DataFrame, target: str, feature_set: str, model_name: str, split_name: str):
    splits = get_pair_splits(pair_df, target, split_name)
    if not splits:
        return None, None, None

    y = pair_df[target].astype(str).values
    pred = np.empty(len(pair_df), dtype=object)
    pred[:] = None

    pair_pred_parts = []
    target_score_parts = []

    for fold, (tr, te) in enumerate(splits):
        model = build_model(pair_df, feature_set, model_name)
        model.fit(pair_df.iloc[tr], y[tr])
        pred[te] = model.predict(pair_df.iloc[te])

        te_df = pair_df.iloc[te].copy()
        te_df["fold"] = fold
        te_df["pair_target"] = target
        te_df["feature_set"] = feature_set
        te_df["model"] = model_name
        te_df["split"] = split_name
        te_df["y_true"] = y[te]
        te_df["y_pred"] = pred[te]

        # Scores for aggregation.
        if target == "pair_action_label":
            for action in ["direct_reuse", "observe", "reject"]:
                te_df[f"score_{action}"] = score_positive_class(model, te_df, action)
        elif target == "pair_utility_label":
            for lab in ["useful", "neutral", "harmful"]:
                te_df[f"score_{lab}"] = score_positive_class(model, te_df, lab)

            # Map utility labels to action-like scores.
            te_df["score_direct_reuse"] = te_df["score_useful"]
            te_df["score_observe"] = te_df["score_neutral"]
            te_df["score_reject"] = te_df["score_harmful"]

        pair_pred_parts.append(te_df[[
            "pair_id",
            "source_task_id",
            "target_task_id",
            "source_bucket",
            "target_reuse_group",
            "fold",
            "pair_target",
            "feature_set",
            "model",
            "split",
            "y_true",
            "y_pred",
            "score_direct_reuse",
            "score_observe",
            "score_reject",
        ]])

        # Aggregate target-level action from pair scores.
        agg_rows = []
        for tid, g in te_df.groupby("target_task_id"):
            row = {
                "target_task_id": tid,
                "fold": fold,
                "pair_target": target,
                "feature_set": feature_set,
                "model": model_name,
                "split": split_name,
            }

            # Bucket-specific useful/direct evidence.
            for action in ["direct_reuse", "observe", "reject"]:
                s = pd.to_numeric(g[f"score_{action}"], errors="coerce")
                row[f"{action}_score_max"] = float(s.max())
                row[f"{action}_score_mean"] = float(s.mean())
                row[f"{action}_score_top3_mean"] = float(s.sort_values(ascending=False).head(3).mean())

            # Direct source anchor evidence is critical.
            direct_g = g[g["source_bucket"].astype(str).isin(["direct_source", "same_operator_source"])]
            if len(direct_g):
                row["direct_source_direct_score_max"] = float(pd.to_numeric(direct_g["score_direct_reuse"], errors="coerce").max())
                row["direct_source_reject_score_max"] = float(pd.to_numeric(direct_g["score_reject"], errors="coerce").max())
            else:
                row["direct_source_direct_score_max"] = np.nan
                row["direct_source_reject_score_max"] = np.nan

            # Final simple aggregation rule:
            # direct if useful/direct evidence wins; reject if harmful/reject wins;
            # observe if neutral/uncertain wins.
            action_scores = {
                "direct_reuse": max(row["direct_reuse_score_max"], row["direct_source_direct_score_max"]),
                "observe": row["observe_score_top3_mean"],
                "reject": max(row["reject_score_max"], row["direct_source_reject_score_max"]),
            }

            # Uncertainty bonus for observe if direct and reject are close.
            margin_dr = abs(action_scores["direct_reuse"] - action_scores["reject"])
            action_scores["observe"] = action_scores["observe"] + max(0.0, 0.15 - margin_dr)

            pred_action = max(action_scores, key=action_scores.get)
            row["pred_target_action"] = pred_action
            row["selected_action_score"] = float(action_scores[pred_action])
            row["direct_reuse_score_final"] = float(action_scores["direct_reuse"])
            row["observe_score_final"] = float(action_scores["observe"])
            row["reject_score_final"] = float(action_scores["reject"])
            agg_rows.append(row)

        target_score_parts.append(pd.DataFrame(agg_rows))

    mask = pd.Series(pred).notna().values

    pair_metrics = {
        "eval_level": "pair",
        "target": target,
        "feature_set": feature_set,
        "model": model_name,
        "split": split_name,
        "n": int(mask.sum()),
        "accuracy": float(accuracy_score(y[mask], pred[mask])),
        "macro_f1": float(f1_score(y[mask], pred[mask], average="macro")),
    }

    pair_preds = pd.concat(pair_pred_parts, ignore_index=True) if pair_pred_parts else pd.DataFrame()
    target_scores = pd.concat(target_score_parts, ignore_index=True) if target_score_parts else pd.DataFrame()

    return pair_metrics, pair_preds, target_scores


def evaluate_target_policies(target_scores: pd.DataFrame, target_df: pd.DataFrame):
    rows = []
    pred_rows = []

    if target_scores.empty:
        return pd.DataFrame(rows), pd.DataFrame(pred_rows)

    target_df = target_df.copy()
    target_df["task_id"] = target_df["task_id"].astype(str)

    merged = target_scores.merge(
        target_df,
        left_on="target_task_id",
        right_on="task_id",
        how="left",
    )

    for eval_target in ["gain_oracle_action", "best_policy_action"]:
        if eval_target not in merged.columns:
            continue

        for keys, g in merged.groupby(["pair_target", "feature_set", "model", "split"], dropna=False):
            y_true = g[eval_target].astype(str).values
            y_pred = g["pred_target_action"].astype(str).values

            rows.append({
                "eval_level": "target_policy",
                "target": eval_target,
                "pair_training_target": keys[0],
                "feature_set": keys[1],
                "model": keys[2],
                "split": keys[3],
                "n": int(len(g)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            })

            pr = g[[
                "target_task_id",
                "pair_target",
                "feature_set",
                "model",
                "split",
                "pred_target_action",
                "direct_reuse_score_final",
                "observe_score_final",
                "reject_score_final",
                "correct_vs_no",
                "best_policy_gain",
            ]].copy()
            pr["eval_target"] = eval_target
            pr["y_true"] = y_true
            pr["y_pred"] = y_pred
            pr["is_correct"] = (pr["y_true"] == pr["y_pred"]).astype(int)
            pred_rows.append(pr)

    return pd.DataFrame(rows), pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()


def action_utility(row, action_value):
    action = str(action_value)
    if action == "direct_reuse":
        return row.get("correct_vs_no", np.nan)
    if action in ["observe", "reject", "no_reuse"]:
        return 0.0
    return np.nan


def summarize_policy_utility(target_pred: pd.DataFrame):
    rows = []
    if target_pred.empty:
        return pd.DataFrame(rows)

    target_pred = target_pred.copy()
    target_pred["pred_gain_vs_no"] = target_pred.apply(
        lambda r: action_utility(r, r["y_pred"]),
        axis=1,
    )
    target_pred["true_gain_vs_no"] = target_pred.apply(
        lambda r: action_utility(r, r["y_true"]),
        axis=1,
    )

    for keys, g in target_pred.groupby(["eval_target", "pair_target", "feature_set", "model", "split"], dropna=False):
        pred_mean = pd.to_numeric(g["pred_gain_vs_no"], errors="coerce").mean()
        true_mean = pd.to_numeric(g["true_gain_vs_no"], errors="coerce").mean()
        rows.append({
            "target": keys[0],
            "pair_training_target": keys[1],
            "feature_set": keys[2],
            "model": keys[3],
            "split": keys[4],
            "n": int(len(g)),
            "mean_pred_gain_vs_no": float(pred_mean) if pd.notna(pred_mean) else None,
            "mean_true_gain_vs_no": float(true_mean) if pd.notna(true_mean) else None,
            "gain_gap_pred_minus_true": float(pred_mean - true_mean) if pd.notna(pred_mean) and pd.notna(true_mean) else None,
        })

    return pd.DataFrame(rows)


def main():
    pair_df = read_csv(IN_PAIRWISE)
    target_df = read_csv(IN_TARGETS)

    # Normalize IDs.
    pair_df["target_task_id"] = pair_df["target_task_id"].astype(str)
    pair_df["source_task_id"] = pair_df["source_task_id"].astype(str)
    target_df["task_id"] = target_df["task_id"].astype(str)

    # Basic cleanup.
    for c in pair_df.columns:
        if c.startswith("same_") or c.endswith("_strength") or c.endswith("_delta") or c.startswith("vec_") or c.endswith("_l2") or c.endswith("_cos") or c.endswith("_mean") or c.endswith("_max"):
            pair_df[c] = pd.to_numeric(pair_df[c], errors="coerce")

    result_rows = []
    pair_pred_parts = []
    target_score_parts = []

    for pair_target in PAIR_TARGETS:
        for feature_set in PAIR_FEATURE_SETS:
            for model in MODELS:
                for split in SPLITS:
                    try:
                        metrics, pair_preds, target_scores = run_pair_cv(pair_df, pair_target, feature_set, model, split)
                        if metrics is None:
                            continue

                        result_rows.append(metrics)
                        if pair_preds is not None and not pair_preds.empty:
                            pair_pred_parts.append(pair_preds)
                        if target_scores is not None and not target_scores.empty:
                            target_score_parts.append(target_scores)

                        print(
                            f"[PAIR CV] target={pair_target:18s} fs={feature_set:18s} "
                            f"model={model:20s} split={split:35s} "
                            f"acc={metrics['accuracy']:.3f} f1={metrics['macro_f1']:.3f}"
                        )

                    except Exception as e:
                        result_rows.append({
                            "eval_level": "pair",
                            "target": pair_target,
                            "feature_set": feature_set,
                            "model": model,
                            "split": split,
                            "n": 0,
                            "accuracy": np.nan,
                            "macro_f1": np.nan,
                            "error": str(e),
                        })
                        print(f"[WARN] Failed {pair_target} {feature_set} {model} {split}: {e}")

    pair_results = pd.DataFrame(result_rows)
    pair_preds = pd.concat(pair_pred_parts, ignore_index=True) if pair_pred_parts else pd.DataFrame()
    target_scores = pd.concat(target_score_parts, ignore_index=True) if target_score_parts else pd.DataFrame()

    target_results, target_preds = evaluate_target_policies(target_scores, target_df)
    utility = summarize_policy_utility(target_preds)

    all_results = pd.concat([pair_results, target_results], ignore_index=True)

    all_results.to_csv(OUT_PAIR_CV, index=False, encoding="utf-8-sig")
    pair_preds.to_csv(OUT_PAIR_PREDS, index=False, encoding="utf-8-sig")
    target_results.to_csv(OUT_TARGET_POLICY, index=False, encoding="utf-8-sig")
    target_preds.to_csv(OUT_TARGET_PREDS, index=False, encoding="utf-8-sig")
    utility.to_csv(OUT_UTILITY, index=False, encoding="utf-8-sig")

    # Baseline comparison with SR-1C tabular.
    sr1c = read_csv(IN_SR1C, required=False)
    comparison = {}

    if not sr1c.empty:
        clean = sr1c.dropna(subset=["macro_f1"]).copy()
        tabular_sets = {
            "boundary_only",
            "reuse_group_only",
            "boundary_plus_group",
            "symbolic_no_concept",
            "symbolic_with_concept",
            "surface_text_only",
            "all_tabular_text",
        }

        target_clean = target_results.dropna(subset=["macro_f1"]).copy()

        for eval_target in ["gain_oracle_action", "best_policy_action"]:
            comparison[eval_target] = {}
            for split in ["group_by_target_reuse_group", "group_by_target_reuse_group_non_source"]:
                # Map SR-3B split to SR-1C baseline group_by_reuse_group.
                base = clean[
                    (clean["target"] == eval_target)
                    & (clean["split"] == "group_by_reuse_group")
                    & (clean["feature_set"].isin(tabular_sets))
                ]
                cur = target_clean[
                    (target_clean["target"] == eval_target)
                    & (target_clean["split"] == split)
                ]

                best_base = base.sort_values("macro_f1", ascending=False).head(1).to_dict(orient="records")
                best_cur = cur.sort_values("macro_f1", ascending=False).head(1).to_dict(orient="records")

                b = best_base[0]["macro_f1"] if best_base else np.nan
                c = best_cur[0]["macro_f1"] if best_cur else np.nan

                comparison[eval_target][split] = {
                    "best_sr1c_tabular_group_by_reuse_group": best_base[0] if best_base else None,
                    "best_sr3b_pairwise_target_policy": best_cur[0] if best_cur else None,
                    "delta_pairwise_minus_tabular_macro_f1": (
                        float(c - b) if pd.notna(c) and pd.notna(b) else None
                    ),
                }

    best_results = (
        all_results.dropna(subset=["macro_f1"])
        .sort_values(["eval_level", "target", "split", "macro_f1", "accuracy"], ascending=[True, True, True, False, False])
        .groupby(["eval_level", "target", "split"], as_index=False)
        .head(1)
    )

    gain_delta_non_source = (
        comparison
        .get("gain_oracle_action", {})
        .get("group_by_target_reuse_group_non_source", {})
        .get("delta_pairwise_minus_tabular_macro_f1", None)
    )
    gain_delta_strict = (
        comparison
        .get("gain_oracle_action", {})
        .get("group_by_target_reuse_group", {})
        .get("delta_pairwise_minus_tabular_macro_f1", None)
    )

    if gain_delta_non_source is not None and gain_delta_non_source >= 0.10:
        verdict = "PASS_STRONG_PAIRWISE_SR_IMPROVES_NON_SOURCE_CROSS_REUSE"
    elif gain_delta_non_source is not None and gain_delta_non_source >= 0.03:
        verdict = "PASS_LITE_PAIRWISE_SR_IMPROVES_NON_SOURCE_CROSS_REUSE"
    elif gain_delta_strict is not None and gain_delta_strict >= 0.03:
        verdict = "MIXED_PAIRWISE_SR_ONLY_STRICT_SPLIT_SIGNAL"
    elif gain_delta_non_source is not None and gain_delta_non_source > -0.03:
        verdict = "MIXED_PAIRWISE_SR_PARITY_WITH_TABULAR"
    else:
        verdict = "FAIL_PAIRWISE_SR_NOT_ENOUGH_YET"

    summary = {
        "input_pairwise": str(IN_PAIRWISE),
        "input_targets": str(IN_TARGETS),
        "n_pair_rows": int(len(pair_df)),
        "n_target_rows": int(len(target_df)),
        "pair_targets": PAIR_TARGETS,
        "target_evals": TARGET_EVALS,
        "feature_sets": PAIR_FEATURE_SETS,
        "models": MODELS,
        "splits": SPLITS,
        "pair_label_counts": {
            t: pair_df[t].astype(str).value_counts(dropna=False).to_dict()
            for t in PAIR_TARGETS
            if t in pair_df.columns
        },
        "best_results": best_results.to_dict(orient="records"),
        "sr1c_tabular_comparison": comparison,
        "verdict": verdict,
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[SAVED]")
    print(f"  all cv results : {OUT_PAIR_CV}")
    print(f"  pair preds     : {OUT_PAIR_PREDS}")
    print(f"  target results : {OUT_TARGET_POLICY}")
    print(f"  target preds   : {OUT_TARGET_PREDS}")
    print(f"  utility        : {OUT_UTILITY}")
    print(f"  summary        : {OUT_SUMMARY}")
    print("\n[VERDICT]")
    print(verdict)


if __name__ == "__main__":
    main()
