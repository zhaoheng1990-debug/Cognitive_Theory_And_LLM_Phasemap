# -*- coding: utf-8 -*-
"""
SR-3C: Learned Pair-to-Target Aggregator Audit

Purpose
-------
SR-3B showed a split result:

    Pair-level utility/action prediction can be strong,
    especially pair_utility_label under target reuse-group splits.

    But fixed hand-written aggregation from pair scores to target policy failed.

Therefore SR-3C learns the aggregation step:

    pair model:
        (source_case, target_case) -> pair utility/action scores

    learned aggregator:
        aggregated pair-score statistics -> target reuse policy

This tests whether the bottleneck is aggregation rather than pairwise SR signal.

Inputs
------
    sr3a_outputs/sr3a_pairwise_source_target_dataset.csv
    sr1b1_outputs/sr1b1_topk_vim_features.csv
    sr1c_outputs/sr1c_cv_results.csv

Outputs
-------
    sr3c_outputs/sr3c_target_cv_results.csv
    sr3c_outputs/sr3c_target_predictions.csv
    sr3c_outputs/sr3c_aggregated_features.csv
    sr3c_outputs/sr3c_policy_utility.csv
    sr3c_outputs/sr3c_summary.json

Main criterion
--------------
Under group_by_target_reuse_group_non_source:
    learned pair-to-target aggregation should improve over SR-1C tabular baseline,
    or at least substantially improve over SR-3B fixed aggregation.
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


BASE_DIR = Path(r"C:\Users\ZH\Desktop\AGI\python_script")

IN_PAIRWISE = BASE_DIR / "sr3a_outputs" / "sr3a_pairwise_source_target_dataset.csv"
IN_TARGETS = BASE_DIR / "sr1b1_outputs" / "sr1b1_topk_vim_features.csv"
IN_SR1C = BASE_DIR / "sr1c_outputs" / "sr1c_cv_results.csv"
IN_SR3B = BASE_DIR / "sr3b_outputs" / "sr3b_target_policy_results.csv"

OUT_DIR = BASE_DIR / "sr3c_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_TARGET_CV = OUT_DIR / "sr3c_target_cv_results.csv"
OUT_TARGET_PREDS = OUT_DIR / "sr3c_target_predictions.csv"
OUT_AGG_FEATURES = OUT_DIR / "sr3c_aggregated_features.csv"
OUT_UTILITY = OUT_DIR / "sr3c_policy_utility.csv"
OUT_SUMMARY = OUT_DIR / "sr3c_summary.json"


SOURCE_GROUP = "G0_STRICT_SOURCE"

PAIR_FEATURE_SETS = [
    "meta_only",
    "geometry_only",
    "geometry_plus_meta",
]

PAIR_TRAINING_TARGETS = [
    "pair_utility_label",
    "pair_action_label",
]

TARGET_EVALS = [
    "gain_oracle_action",
    "best_policy_action",
]

SPLITS = [
    "stratified_target",
    "group_by_target_concept",
    "group_by_target_task_family",
    "group_by_target_reuse_group",
    "group_by_target_reuse_group_non_source",
]

PAIR_MODELS = [
    "pair_logistic",
    "pair_rf",
    "pair_extra_trees",
]

AGG_MODELS = [
    "dummy_most_frequent",
    "agg_logistic",
    "agg_rf",
    "agg_extra_trees",
    "agg_gbdt",
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


def get_pair_feature_columns(df: pd.DataFrame, feature_set: str) -> Tuple[List[str], List[str]]:
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
        categorical = []
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
        raise ValueError(f"Unknown pair feature set: {feature_set}")

    categorical = [c for c in categorical if c in df.columns]
    numeric = [c for c in numeric if c in df.columns]
    return numeric, categorical


def build_pair_model(df: pd.DataFrame, feature_set: str, model_name: str):
    numeric, categorical = get_pair_feature_columns(df, feature_set)

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
        raise ValueError(f"No pair features selected: {feature_set}")

    pre = ColumnTransformer(transformers=transformers, remainder="drop")

    if model_name == "pair_logistic":
        clf = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
        )

    elif model_name == "pair_rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=7,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )

    elif model_name == "pair_extra_trees":
        clf = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=7,
            min_samples_leaf=3,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )

    else:
        raise ValueError(f"Unknown pair model: {model_name}")

    return Pipeline([("pre", pre), ("clf", clf)])


def build_agg_model(feature_df: pd.DataFrame, model_name: str):
    categorical = [
        "target_reuse_group",
        "target_task_family",
        "target_operator",
        "target_concept",
    ]
    categorical = [c for c in categorical if c in feature_df.columns]

    numeric = [
        c for c in feature_df.columns
        if c not in categorical
        and c not in [
            "target_task_id",
            "gain_oracle_action",
            "best_policy_action",
            "best_policy_id",
        ]
    ]

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

    pre = ColumnTransformer(transformers=transformers, remainder="drop")

    if model_name == "dummy_most_frequent":
        clf = DummyClassifier(strategy="most_frequent")

    elif model_name == "agg_logistic":
        clf = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
        )

    elif model_name == "agg_rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=3,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )

    elif model_name == "agg_extra_trees":
        clf = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )

    elif model_name == "agg_gbdt":
        clf = GradientBoostingClassifier(
            n_estimators=120,
            max_depth=2,
            learning_rate=0.05,
            random_state=42,
        )

    else:
        raise ValueError(f"Unknown agg model: {model_name}")

    return Pipeline([("pre", pre), ("clf", clf)])


def get_target_splits(target_df: pd.DataFrame, eval_target: str, split: str):
    y = target_df[eval_target].astype(str).values

    if split == "stratified_target":
        vc = pd.Series(y).value_counts()
        if vc.min() < 2:
            return []
        n_splits = int(min(5, vc.min()))
        n_splits = max(2, n_splits)
        sp = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return list(sp.split(target_df, y))

    if split == "group_by_target_concept":
        groups = target_df["target_concept"].astype(str).values
        if len(np.unique(groups)) < 2:
            return []
        sp = GroupKFold(n_splits=int(min(5, len(np.unique(groups)))))
        return list(sp.split(target_df, y, groups))

    if split == "group_by_target_task_family":
        groups = target_df["target_task_family"].astype(str).values
        if len(np.unique(groups)) < 2:
            return []
        sp = GroupKFold(n_splits=int(min(5, len(np.unique(groups)))))
        return list(sp.split(target_df, y, groups))

    if split == "group_by_target_reuse_group":
        groups = target_df["target_reuse_group"].astype(str).values
        if len(np.unique(groups)) < 2:
            return []
        sp = GroupKFold(n_splits=int(min(5, len(np.unique(groups)))))
        return list(sp.split(target_df, y, groups))

    if split == "group_by_target_reuse_group_non_source":
        all_idx = np.arange(len(target_df))
        splits = []
        for group in sorted(target_df["target_reuse_group"].astype(str).unique()):
            if group == SOURCE_GROUP:
                continue
            te = np.where(target_df["target_reuse_group"].astype(str).values == group)[0]
            tr = np.setdiff1d(all_idx, te)
            if len(te) > 0 and len(tr) > 0:
                splits.append((tr, te))
        return splits

    raise ValueError(f"Unknown split: {split}")


def add_pair_scores(model, pair_df: pd.DataFrame, pair_target: str) -> pd.DataFrame:
    out = pair_df.copy()
    pred = model.predict(pair_df)
    out["pair_pred"] = pred.astype(str)

    clf = model.named_steps["clf"]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(pair_df)
        classes = list(clf.classes_)
        for cls in classes:
            out[f"score_{pair_target}_{cls}"] = proba[:, classes.index(cls)]
    else:
        for cls in sorted(pd.Series(pred).astype(str).unique()):
            out[f"score_{pair_target}_{cls}"] = (pred.astype(str) == cls).astype(float)

    return out


def aggregate_pair_scores(scored_pairs: pd.DataFrame, pair_target: str) -> pd.DataFrame:
    """
    Aggregate pair-level scores into one target-level evidence row per target.

    v2 fix:
    The previous version only created pair_pred_rate_<class> columns for classes
    actually predicted inside a fold. This made train/test aggregated schemas
    differ, causing ColumnTransformer transform errors such as:

        ValueError: columns are missing: {'pair_pred_rate_useful'}

    This version always emits a stable schema for all expected score classes.
    Missing classes are filled with 0.
    """
    rows = []

    if pair_target == "pair_utility_label":
        expected_labels = ["harmful", "neutral", "useful"]
    elif pair_target == "pair_action_label":
        expected_labels = ["direct_reuse", "observe", "reject"]
    else:
        expected_labels = sorted(scored_pairs[pair_target].astype(str).unique()) if pair_target in scored_pairs.columns else []

    expected_score_cols = [f"score_{pair_target}_{lab}" for lab in expected_labels]

    # Ensure score columns exist even if the model fold never saw/predicted a class.
    for col in expected_score_cols:
        if col not in scored_pairs.columns:
            scored_pairs[col] = 0.0

    for tid, g in scored_pairs.groupby("target_task_id"):
        row = {
            "target_task_id": str(tid),
            "target_reuse_group": str(g["target_reuse_group"].iloc[0]),
            "target_task_family": str(g["target_task_family"].iloc[0]),
            "target_operator": str(g["target_operator"].iloc[0]),
            "target_concept": str(g["target_concept"].iloc[0]),
            "gain_oracle_action": str(g["target_gain_oracle_action"].iloc[0]),
            "best_policy_action": str(g["target_best_policy_action"].iloc[0]),
            "best_policy_id": str(g["target_best_policy_id"].iloc[0]),
            "n_pairs": int(len(g)),
        }

        for col in expected_score_cols:
            s = pd.to_numeric(g[col], errors="coerce").fillna(0.0)
            base = col.replace(f"score_{pair_target}_", "")
            row[f"{base}_score_mean"] = float(s.mean())
            row[f"{base}_score_max"] = float(s.max())
            row[f"{base}_score_min"] = float(s.min())
            row[f"{base}_score_std"] = float(s.std(ddof=0))
            row[f"{base}_score_top3_mean"] = float(s.sort_values(ascending=False).head(3).mean())

            for bucket_name in ["direct_source", "observe_source", "reject_source", "same_operator_source", "broad_source"]:
                gb = g[g["source_bucket"].astype(str) == bucket_name]
                if len(gb):
                    sb = pd.to_numeric(gb[col], errors="coerce").fillna(0.0)
                    row[f"{bucket_name}__{base}_score_mean"] = float(sb.mean())
                    row[f"{bucket_name}__{base}_score_max"] = float(sb.max())
                    row[f"{bucket_name}__{base}_score_top3_mean"] = float(sb.sort_values(ascending=False).head(3).mean())
                else:
                    row[f"{bucket_name}__{base}_score_mean"] = 0.0
                    row[f"{bucket_name}__{base}_score_max"] = 0.0
                    row[f"{bucket_name}__{base}_score_top3_mean"] = 0.0

        # Stable prediction distribution schema.
        pred_series = g["pair_pred"].astype(str)
        pred_dist = pred_series.value_counts(normalize=True)
        for cls in expected_labels:
            row[f"pair_pred_rate_{cls}"] = float(pred_dist.get(cls, 0.0))

        rows.append(row)

    out = pd.DataFrame(rows)

    # Final defensive schema fill.
    for cls in expected_labels:
        c = f"pair_pred_rate_{cls}"
        if c not in out.columns:
            out[c] = 0.0

    for col in expected_score_cols:
        base = col.replace(f"score_{pair_target}_", "")
        base_cols = [
            f"{base}_score_mean",
            f"{base}_score_max",
            f"{base}_score_min",
            f"{base}_score_std",
            f"{base}_score_top3_mean",
        ]
        for c in base_cols:
            if c not in out.columns:
                out[c] = 0.0
        for bucket_name in ["direct_source", "observe_source", "reject_source", "same_operator_source", "broad_source"]:
            for suffix in ["mean", "max", "top3_mean"]:
                c = f"{bucket_name}__{base}_score_{suffix}"
                if c not in out.columns:
                    out[c] = 0.0

    return out

def action_utility(row, action):
    action = str(action)
    if action == "direct_reuse":
        return row.get("correct_vs_no", np.nan)
    if action in ["observe", "reject", "no_reuse"]:
        return 0.0
    return np.nan


def prepare_pair_df(pair_df: pd.DataFrame) -> pd.DataFrame:
    for c in pair_df.columns:
        if (
            c.startswith("same_")
            or c.endswith("_strength")
            or c.endswith("_delta")
            or c.startswith("vec_")
            or c.endswith("_l2")
            or c.endswith("_cos")
            or c.endswith("_mean")
            or c.endswith("_max")
            or c.endswith("_ratio_target_over_source")
        ):
            pair_df[c] = pd.to_numeric(pair_df[c], errors="coerce")
    return pair_df


def prepare_target_df(targets: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "task_id",
        "reuse_group",
        "task_family",
        "target_operator",
        "concept",
        "gain_oracle_action",
        "best_policy_action",
        "best_policy_id",
        "correct_vs_no",
        "best_policy_gain",
    ]
    t = targets[[c for c in keep if c in targets.columns]].copy()
    t = t.rename(columns={
        "task_id": "target_task_id",
        "reuse_group": "target_reuse_group",
        "task_family": "target_task_family",
        "target_operator": "target_operator",
        "concept": "target_concept",
    })
    t["target_task_id"] = t["target_task_id"].astype(str)
    return t


def main():
    pair_df = read_csv(IN_PAIRWISE)
    targets_raw = read_csv(IN_TARGETS)
    sr1c = read_csv(IN_SR1C, required=False)
    sr3b = read_csv(IN_SR3B, required=False)

    pair_df = prepare_pair_df(pair_df)
    target_df = prepare_target_df(targets_raw)

    pair_df["target_task_id"] = pair_df["target_task_id"].astype(str)

    result_rows = []
    pred_rows = []
    agg_feature_parts = []
    utility_rows = []

    for pair_target in PAIR_TRAINING_TARGETS:
        for pair_feature_set in PAIR_FEATURE_SETS:
            for pair_model_name in PAIR_MODELS:
                for split in SPLITS:
                    # Target-level split controls both pair model and aggregator.
                    target_splits = get_target_splits(target_df, "gain_oracle_action", split)
                    if not target_splits:
                        continue

                    for agg_model_name in AGG_MODELS:
                        y_pred_all = {}
                        y_true_all = {}

                        for fold, (target_tr, target_te) in enumerate(target_splits):
                            train_tids = set(target_df.iloc[target_tr]["target_task_id"].astype(str))
                            test_tids = set(target_df.iloc[target_te]["target_task_id"].astype(str))

                            pair_train = pair_df[pair_df["target_task_id"].astype(str).isin(train_tids)].copy()
                            pair_test = pair_df[pair_df["target_task_id"].astype(str).isin(test_tids)].copy()

                            # Train pair model.
                            pair_model = build_pair_model(pair_train, pair_feature_set, pair_model_name)
                            y_pair = pair_train[pair_target].astype(str).values
                            pair_model.fit(pair_train, y_pair)

                            # Score train and test pairs.
                            scored_train = add_pair_scores(pair_model, pair_train, pair_target)
                            scored_test = add_pair_scores(pair_model, pair_test, pair_target)

                            agg_train = aggregate_pair_scores(scored_train, pair_target)
                            agg_test = aggregate_pair_scores(scored_test, pair_target)

                            # Add real target utility columns from target_df.
                            agg_train = agg_train.merge(target_df, on=[
                                "target_task_id",
                                "target_reuse_group",
                                "target_task_family",
                                "target_operator",
                                "target_concept",
                                "gain_oracle_action",
                                "best_policy_action",
                                "best_policy_id",
                            ], how="left")

                            agg_test = agg_test.merge(target_df, on=[
                                "target_task_id",
                                "target_reuse_group",
                                "target_task_family",
                                "target_operator",
                                "target_concept",
                                "gain_oracle_action",
                                "best_policy_action",
                                "best_policy_id",
                            ], how="left")

                            # Store aggregated features for audit.
                            audit_train = agg_train.copy()
                            audit_train["fold"] = fold
                            audit_train["fold_role"] = "train"
                            audit_train["pair_training_target"] = pair_target
                            audit_train["pair_feature_set"] = pair_feature_set
                            audit_train["pair_model"] = pair_model_name
                            audit_train["split"] = split
                            agg_feature_parts.append(audit_train)

                            audit_test = agg_test.copy()
                            audit_test["fold"] = fold
                            audit_test["fold_role"] = "test"
                            audit_test["pair_training_target"] = pair_target
                            audit_test["pair_feature_set"] = pair_feature_set
                            audit_test["pair_model"] = pair_model_name
                            audit_test["split"] = split
                            agg_feature_parts.append(audit_test)

                            for eval_target in TARGET_EVALS:
                                agg_model = build_agg_model(agg_train, agg_model_name)
                                y_train = agg_train[eval_target].astype(str).values
                                y_test = agg_test[eval_target].astype(str).values
                                agg_model.fit(agg_train, y_train)
                                # Align schemas defensively: folds can still differ because some
                                # bucket/class combinations may be absent after aggregation.
                                for col in agg_train.columns:
                                    if col not in agg_test.columns:
                                        agg_test[col] = 0.0
                                for col in agg_test.columns:
                                    if col not in agg_train.columns:
                                        agg_train[col] = 0.0
                                agg_test = agg_test[agg_train.columns]

                                pred = agg_model.predict(agg_test)

                                for tid, yt, yp in zip(agg_test["target_task_id"].astype(str), y_test, pred):
                                    key = (eval_target, tid)
                                    y_pred_all[key] = str(yp)
                                    y_true_all[key] = str(yt)

                        # Evaluate after all folds.
                        for eval_target in TARGET_EVALS:
                            keys = [k for k in y_true_all.keys() if k[0] == eval_target]
                            if not keys:
                                continue
                            yt = np.array([y_true_all[k] for k in keys])
                            yp = np.array([y_pred_all[k] for k in keys])
                            tids = [k[1] for k in keys]

                            acc = accuracy_score(yt, yp)
                            mf1 = f1_score(yt, yp, average="macro")

                            result_rows.append({
                                "eval_target": eval_target,
                                "pair_training_target": pair_target,
                                "pair_feature_set": pair_feature_set,
                                "pair_model": pair_model_name,
                                "agg_model": agg_model_name,
                                "split": split,
                                "n": int(len(yt)),
                                "accuracy": float(acc),
                                "macro_f1": float(mf1),
                            })

                            pred_df = pd.DataFrame({
                                "target_task_id": tids,
                                "eval_target": eval_target,
                                "pair_training_target": pair_target,
                                "pair_feature_set": pair_feature_set,
                                "pair_model": pair_model_name,
                                "agg_model": agg_model_name,
                                "split": split,
                                "y_true": yt,
                                "y_pred": yp,
                            })

                            pred_df = pred_df.merge(target_df[[
                                "target_task_id",
                                "correct_vs_no",
                                "best_policy_gain",
                            ]], on="target_task_id", how="left")
                            pred_df["is_correct"] = (pred_df["y_true"] == pred_df["y_pred"]).astype(int)
                            pred_rows.append(pred_df)

                        print(
                            f"[SR3C] pair_target={pair_target:18s} pair_fs={pair_feature_set:18s} "
                            f"pair_model={pair_model_name:16s} agg={agg_model_name:20s} split={split}"
                        )

    results = pd.DataFrame(result_rows)
    preds = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    agg_features = pd.concat(agg_feature_parts, ignore_index=True) if agg_feature_parts else pd.DataFrame()

    # Utility.
    if not preds.empty:
        u = preds.copy()
        u["pred_gain_vs_no"] = u.apply(lambda r: action_utility(r, r["y_pred"]), axis=1)
        u["true_gain_vs_no"] = u.apply(lambda r: action_utility(r, r["y_true"]), axis=1)

        for keys, g in u.groupby(["eval_target", "pair_training_target", "pair_feature_set", "pair_model", "agg_model", "split"], dropna=False):
            pred_mean = pd.to_numeric(g["pred_gain_vs_no"], errors="coerce").mean()
            true_mean = pd.to_numeric(g["true_gain_vs_no"], errors="coerce").mean()
            utility_rows.append({
                "eval_target": keys[0],
                "pair_training_target": keys[1],
                "pair_feature_set": keys[2],
                "pair_model": keys[3],
                "agg_model": keys[4],
                "split": keys[5],
                "n": int(len(g)),
                "mean_pred_gain_vs_no": float(pred_mean) if pd.notna(pred_mean) else None,
                "mean_true_gain_vs_no": float(true_mean) if pd.notna(true_mean) else None,
                "gain_gap_pred_minus_true": float(pred_mean - true_mean) if pd.notna(pred_mean) and pd.notna(true_mean) else None,
            })

    utility = pd.DataFrame(utility_rows)

    results.to_csv(OUT_TARGET_CV, index=False, encoding="utf-8-sig")
    preds.to_csv(OUT_TARGET_PREDS, index=False, encoding="utf-8-sig")
    agg_features.to_csv(OUT_AGG_FEATURES, index=False, encoding="utf-8-sig")
    utility.to_csv(OUT_UTILITY, index=False, encoding="utf-8-sig")

    # Baseline comparisons.
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

        for eval_target in TARGET_EVALS:
            comparison[eval_target] = {}
            for split in ["group_by_target_reuse_group", "group_by_target_reuse_group_non_source"]:
                sr1c_split = "group_by_reuse_group"
                base = clean[
                    (clean["target"] == eval_target)
                    & (clean["split"] == sr1c_split)
                    & (clean["feature_set"].isin(tabular_sets))
                ]
                cur = results[
                    (results["eval_target"] == eval_target)
                    & (results["split"] == split)
                ]

                best_base = base.sort_values("macro_f1", ascending=False).head(1).to_dict(orient="records")
                best_cur = cur.sort_values("macro_f1", ascending=False).head(1).to_dict(orient="records")

                b = best_base[0]["macro_f1"] if best_base else np.nan
                c = best_cur[0]["macro_f1"] if best_cur else np.nan

                comparison[eval_target][split] = {
                    "best_sr1c_tabular_group_by_reuse_group": best_base[0] if best_base else None,
                    "best_sr3c_learned_aggregator": best_cur[0] if best_cur else None,
                    "delta_sr3c_minus_tabular_macro_f1": (
                        float(c - b) if pd.notna(c) and pd.notna(b) else None
                    ),
                }

    # Compare to SR-3B if available.
    sr3b_comparison = {}
    if not sr3b.empty:
        for eval_target in TARGET_EVALS:
            sr3b_comparison[eval_target] = {}
            for split in ["group_by_target_reuse_group", "group_by_target_reuse_group_non_source"]:
                b = sr3b[
                    (sr3b["target"] == eval_target)
                    & (sr3b["split"] == split)
                ]
                c = results[
                    (results["eval_target"] == eval_target)
                    & (results["split"] == split)
                ]
                best_b = b.sort_values("macro_f1", ascending=False).head(1).to_dict(orient="records")
                best_c = c.sort_values("macro_f1", ascending=False).head(1).to_dict(orient="records")

                bv = best_b[0]["macro_f1"] if best_b else np.nan
                cv = best_c[0]["macro_f1"] if best_c else np.nan

                sr3b_comparison[eval_target][split] = {
                    "best_sr3b_fixed_aggregation": best_b[0] if best_b else None,
                    "best_sr3c_learned_aggregation": best_c[0] if best_c else None,
                    "delta_sr3c_minus_sr3b_macro_f1": (
                        float(cv - bv) if pd.notna(cv) and pd.notna(bv) else None
                    ),
                }

    best_results = (
        results.dropna(subset=["macro_f1"])
        .sort_values(["eval_target", "split", "macro_f1", "accuracy"], ascending=[True, True, False, False])
        .groupby(["eval_target", "split"], as_index=False)
        .head(1)
        if not results.empty else pd.DataFrame()
    )

    gain_delta_non_source = (
        comparison
        .get("gain_oracle_action", {})
        .get("group_by_target_reuse_group_non_source", {})
        .get("delta_sr3c_minus_tabular_macro_f1", None)
    )

    gain_delta_strict = (
        comparison
        .get("gain_oracle_action", {})
        .get("group_by_target_reuse_group", {})
        .get("delta_sr3c_minus_tabular_macro_f1", None)
    )

    gain_delta_vs_sr3b_non_source = (
        sr3b_comparison
        .get("gain_oracle_action", {})
        .get("group_by_target_reuse_group_non_source", {})
        .get("delta_sr3c_minus_sr3b_macro_f1", None)
    )

    if gain_delta_non_source is not None and gain_delta_non_source >= 0.10:
        verdict = "PASS_STRONG_LEARNED_PAIR_AGGREGATOR_BEATS_TABULAR"
    elif gain_delta_non_source is not None and gain_delta_non_source >= 0.03:
        verdict = "PASS_LITE_LEARNED_PAIR_AGGREGATOR_BEATS_TABULAR"
    elif gain_delta_vs_sr3b_non_source is not None and gain_delta_vs_sr3b_non_source >= 0.10:
        verdict = "PASS_INTERNAL_LEARNED_AGGREGATOR_FIXES_SR3B_BUT_NOT_TABULAR"
    elif gain_delta_non_source is not None and gain_delta_non_source > -0.03:
        verdict = "MIXED_LEARNED_AGGREGATOR_PARITY_WITH_TABULAR"
    else:
        verdict = "FAIL_LEARNED_PAIR_AGGREGATOR_NOT_ENOUGH"

    summary = {
        "input_pairwise": str(IN_PAIRWISE),
        "input_targets": str(IN_TARGETS),
        "n_pair_rows": int(len(pair_df)),
        "n_target_rows": int(len(target_df)),
        "pair_training_targets": PAIR_TRAINING_TARGETS,
        "target_evals": TARGET_EVALS,
        "pair_feature_sets": PAIR_FEATURE_SETS,
        "pair_models": PAIR_MODELS,
        "agg_models": AGG_MODELS,
        "splits": SPLITS,
        "best_results": best_results.to_dict(orient="records"),
        "sr1c_tabular_comparison": comparison,
        "sr3b_fixed_aggregation_comparison": sr3b_comparison,
        "verdict": verdict,
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[SAVED]")
    print(f"  target cv       : {OUT_TARGET_CV}")
    print(f"  target preds    : {OUT_TARGET_PREDS}")
    print(f"  agg features    : {OUT_AGG_FEATURES}")
    print(f"  utility         : {OUT_UTILITY}")
    print(f"  summary         : {OUT_SUMMARY}")
    print("\n[VERDICT]")
    print(verdict)


if __name__ == "__main__":
    main()
