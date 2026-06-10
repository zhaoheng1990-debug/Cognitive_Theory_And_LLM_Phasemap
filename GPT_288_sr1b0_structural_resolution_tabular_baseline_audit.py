# -*- coding: utf-8 -*-
"""
SR-1B.0: StructuralResolution Tabular Baseline Audit v2

Fixes
-----
v2 fixes a pandas merge bug in summarize_policy_utility():

    KeyError: 'target'

Cause:
    The enriched label dataframe may already contain a column named 'target',
    so merging prediction rows with columns ['target', ...] can silently create
    target_x / target_y and remove the plain 'target' column expected by groupby.

Fix:
    Rename prediction columns before merge:
        target      -> pred_target_name
        y_true      -> pred_true
        y_pred      -> pred_label

Inputs
------
    C:\\Users\\ZH\\Desktop\\AGI\\python_script\\sr1a_outputs\\sr1a1_enriched_structural_resolution_labels.csv

Outputs
-------
    sr1b0_outputs/sr1b0_cv_results.csv
    sr1b0_outputs/sr1b0_predictions.csv
    sr1b0_outputs/sr1b0_policy_utility.csv
    sr1b0_outputs/sr1b0_summary.json
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings("ignore")


BASE_DIR = Path(r"C:\Users\ZH\Desktop\AGI\python_script")
IN_ENRICHED = BASE_DIR / "sr1a_outputs" / "sr1a1_enriched_structural_resolution_labels.csv"

OUT_DIR = BASE_DIR / "sr1b0_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CV = OUT_DIR / "sr1b0_cv_results.csv"
OUT_PREDS = OUT_DIR / "sr1b0_predictions.csv"
OUT_UTILITY = OUT_DIR / "sr1b0_policy_utility.csv"
OUT_SUMMARY = OUT_DIR / "sr1b0_summary.json"


def read_input() -> pd.DataFrame:
    if not IN_ENRICHED.exists():
        raise FileNotFoundError(f"Required input not found: {IN_ENRICHED}")
    df = pd.read_csv(IN_ENRICHED)
    print(f"[LOAD] {IN_ENRICHED}")
    print(f"       shape={df.shape}")
    print(f"       columns={list(df.columns)}")
    return df


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipeline(feature_set: str, model_name: str):
    numeric_features = []
    categorical_features = []
    text_features = []

    if feature_set == "boundary_only":
        numeric_features = ["boundary_strength"]

    elif feature_set == "reuse_group_only":
        categorical_features = ["reuse_group"]

    elif feature_set == "boundary_plus_group":
        numeric_features = ["boundary_strength"]
        categorical_features = ["reuse_group"]

    elif feature_set == "symbolic_no_concept":
        numeric_features = ["boundary_strength"]
        categorical_features = ["reuse_group", "task_family", "target_operator"]

    elif feature_set == "symbolic_with_concept":
        numeric_features = ["boundary_strength"]
        categorical_features = ["reuse_group", "task_family", "target_operator", "concept"]

    elif feature_set == "surface_text_only":
        text_features = ["request"]

    elif feature_set == "all_tabular_text":
        numeric_features = ["boundary_strength"]
        categorical_features = ["reuse_group", "task_family", "target_operator", "concept"]
        text_features = ["request"]

    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    transformers = []

    if numeric_features:
        transformers.append(("num", StandardScaler(), numeric_features))

    if categorical_features:
        transformers.append(("cat", make_ohe(), categorical_features))

    if text_features:
        transformers.append((
            "text",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=1,
                max_features=2000,
            ),
            text_features[0],
        ))

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

    elif model_name == "tree_depth3":
        clf = DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced",
        )

    elif model_name == "rf_small":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=3,
            random_state=42,
            class_weight="balanced_subsample",
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return Pipeline([("pre", pre), ("clf", clf)])


def get_splits(df: pd.DataFrame, target: str, split_name: str):
    y = df[target].astype(str).values

    if split_name == "stratified5":
        vc = pd.Series(y).value_counts()
        if vc.min() < 2:
            return []
        n_splits = int(min(5, vc.min()))
        n_splits = max(2, n_splits)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return list(splitter.split(df, y))

    if split_name == "group_by_concept":
        groups = df["concept"].astype(str).values
        if len(np.unique(groups)) < 2:
            return []
        n_splits = int(min(5, len(np.unique(groups))))
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(df, y, groups))

    if split_name == "group_by_task_family":
        groups = df["task_family"].astype(str).values
        if len(np.unique(groups)) < 2:
            return []
        n_splits = int(min(5, len(np.unique(groups))))
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(df, y, groups))

    if split_name == "group_by_reuse_group":
        groups = df["reuse_group"].astype(str).values
        if len(np.unique(groups)) < 2:
            return []
        n_splits = int(min(5, len(np.unique(groups))))
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(df, y, groups))

    raise ValueError(f"Unknown split_name: {split_name}")


def cv_predict(df: pd.DataFrame, target: str, feature_set: str, model_name: str, split_name: str):
    splits = get_splits(df, target, split_name)
    if not splits:
        return None, None

    y = df[target].astype(str).values
    pred = np.empty(len(df), dtype=object)
    pred[:] = None

    for fold, (tr, te) in enumerate(splits):
        pipe = build_pipeline(feature_set, model_name)
        pipe.fit(df.iloc[tr], y[tr])
        pred[te] = pipe.predict(df.iloc[te])

    mask = pd.Series(pred).notna().values
    if mask.sum() == 0:
        return None, None

    metrics = {
        "target": target,
        "feature_set": feature_set,
        "model": model_name,
        "split": split_name,
        "n": int(mask.sum()),
        "accuracy": float(accuracy_score(y[mask], pred[mask])),
        "macro_f1": float(f1_score(y[mask], pred[mask], average="macro")),
    }

    pred_df = pd.DataFrame({
        "task_id": df["task_id"].astype(str),
        "target": target,
        "feature_set": feature_set,
        "model": model_name,
        "split": split_name,
        "y_true": y,
        "y_pred": pred,
        "is_correct": (y == pred).astype(int),
    })

    return metrics, pred_df


def action_utility(row, action_value):
    action = str(action_value)
    if action == "direct_reuse":
        return row.get("correct_vs_no", np.nan)
    if action in ["observe", "reject", "no_reuse"]:
        return 0.0
    return np.nan


def policy_id_utility(row, policy_id_value):
    pid = str(policy_id_value)
    c = f"{pid}__policy_gain"
    if c in row.index:
        return row[c]
    return np.nan


def summarize_policy_utility(df: pd.DataFrame, preds: pd.DataFrame):
    rows = []

    if preds.empty:
        return pd.DataFrame(rows)

    # Keep only columns needed from predictions, and rename them to avoid merge collisions.
    pred_keep = preds[[
        "task_id",
        "target",
        "feature_set",
        "model",
        "split",
        "y_true",
        "y_pred",
    ]].copy()

    pred_keep = pred_keep.rename(columns={
        "target": "pred_target_name",
        "y_true": "pred_true",
        "y_pred": "pred_label",
    })

    # Action targets.
    action_targets = ["gain_oracle_action", "best_policy_action"]
    for target_name in action_targets:
        sub = pred_keep[pred_keep["pred_target_name"] == target_name].copy()
        if sub.empty:
            continue

        merged = df.merge(sub, on="task_id", how="inner", suffixes=("", "_pred"))

        merged["pred_action_gain_vs_no"] = merged.apply(
            lambda r: action_utility(r, r["pred_label"]),
            axis=1,
        )
        merged["true_action_gain_vs_no"] = merged.apply(
            lambda r: action_utility(r, r["pred_true"]),
            axis=1,
        )

        for keys, g in merged.groupby(["pred_target_name", "feature_set", "model", "split"]):
            pred_mean = pd.to_numeric(g["pred_action_gain_vs_no"], errors="coerce").mean()
            true_mean = pd.to_numeric(g["true_action_gain_vs_no"], errors="coerce").mean()
            rows.append({
                "target": keys[0],
                "feature_set": keys[1],
                "model": keys[2],
                "split": keys[3],
                "n": int(len(g)),
                "mean_pred_action_gain_vs_no": float(pred_mean) if pd.notna(pred_mean) else None,
                "mean_true_action_gain_vs_no": float(true_mean) if pd.notna(true_mean) else None,
                "gain_gap_pred_minus_true": float(pred_mean - true_mean) if pd.notna(pred_mean) and pd.notna(true_mean) else None,
            })

    # Policy-id target.
    sub = pred_keep[pred_keep["pred_target_name"] == "best_policy_id"].copy()
    if not sub.empty:
        merged = df.merge(sub, on="task_id", how="inner", suffixes=("", "_pred"))
        merged["pred_policy_gain"] = merged.apply(
            lambda r: policy_id_utility(r, r["pred_label"]),
            axis=1,
        )
        merged["true_policy_gain"] = merged["best_policy_gain"]

        for keys, g in merged.groupby(["pred_target_name", "feature_set", "model", "split"]):
            pred_mean = pd.to_numeric(g["pred_policy_gain"], errors="coerce").mean()
            true_mean = pd.to_numeric(g["true_policy_gain"], errors="coerce").mean()
            rows.append({
                "target": keys[0],
                "feature_set": keys[1],
                "model": keys[2],
                "split": keys[3],
                "n": int(len(g)),
                "mean_pred_policy_gain": float(pred_mean) if pd.notna(pred_mean) else None,
                "mean_true_policy_gain": float(true_mean) if pd.notna(true_mean) else None,
                "gain_gap_pred_minus_true": float(pred_mean - true_mean) if pd.notna(pred_mean) and pd.notna(true_mean) else None,
            })

    return pd.DataFrame(rows)


def main():
    df = read_input()

    required = [
        "task_id", "reuse_group", "task_family", "target_operator", "concept",
        "request", "boundary_strength", "gain_oracle_action",
        "best_policy_action", "best_policy_id", "correct_vs_no",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for c in ["reuse_group", "task_family", "target_operator", "concept", "request"]:
        df[c] = df[c].fillna("NA").astype(str)

    df["boundary_strength"] = pd.to_numeric(df["boundary_strength"], errors="coerce").fillna(-1)

    targets = ["gain_oracle_action", "best_policy_action", "best_policy_id"]

    feature_sets = [
        "boundary_only",
        "reuse_group_only",
        "boundary_plus_group",
        "symbolic_no_concept",
        "symbolic_with_concept",
        "surface_text_only",
        "all_tabular_text",
    ]

    models = [
        "dummy_most_frequent",
        "logistic_l2",
        "tree_depth3",
        "rf_small",
    ]

    splits = [
        "stratified5",
        "group_by_concept",
        "group_by_task_family",
        "group_by_reuse_group",
    ]

    result_rows = []
    pred_parts = []

    for target in targets:
        for feature_set in feature_sets:
            for model_name in models:
                for split_name in splits:
                    try:
                        metrics, pred_df = cv_predict(df, target, feature_set, model_name, split_name)
                        if metrics is None:
                            continue
                        result_rows.append(metrics)
                        pred_parts.append(pred_df)
                        print(
                            f"[CV] target={target:20s} feature={feature_set:22s} "
                            f"model={model_name:20s} split={split_name:20s} "
                            f"acc={metrics['accuracy']:.3f} f1={metrics['macro_f1']:.3f}"
                        )
                    except Exception as e:
                        result_rows.append({
                            "target": target,
                            "feature_set": feature_set,
                            "model": model_name,
                            "split": split_name,
                            "n": 0,
                            "accuracy": np.nan,
                            "macro_f1": np.nan,
                            "error": str(e),
                        })
                        print(f"[WARN] Failed: {target} {feature_set} {model_name} {split_name}: {e}")

    results = pd.DataFrame(result_rows)
    preds = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()

    utility = summarize_policy_utility(df, preds) if not preds.empty else pd.DataFrame()

    results.to_csv(OUT_CV, index=False, encoding="utf-8-sig")
    preds.to_csv(OUT_PREDS, index=False, encoding="utf-8-sig")
    utility.to_csv(OUT_UTILITY, index=False, encoding="utf-8-sig")

    clean_results = results.dropna(subset=["macro_f1"]).copy()

    if not clean_results.empty:
        best_by_target_split = (
            clean_results
            .sort_values(["target", "split", "macro_f1", "accuracy"], ascending=[True, True, False, False])
            .groupby(["target", "split"], as_index=False)
            .head(1)
        )
    else:
        best_by_target_split = pd.DataFrame()

    utility_top = []
    if not utility.empty:
        sort_cols = []
        if "mean_pred_action_gain_vs_no" in utility.columns:
            sort_cols.append("mean_pred_action_gain_vs_no")
        if "mean_pred_policy_gain" in utility.columns:
            sort_cols.append("mean_pred_policy_gain")

        if sort_cols:
            utility_top = (
                utility.sort_values(sort_cols, ascending=[False] * len(sort_cols))
                .head(20)
                .to_dict(orient="records")
            )
        else:
            utility_top = utility.head(20).to_dict(orient="records")

    summary = {
        "input": str(IN_ENRICHED),
        "n_rows": int(len(df)),
        "targets": targets,
        "feature_sets": feature_sets,
        "models": models,
        "splits": splits,
        "target_counts": {
            t: df[t].astype(str).value_counts(dropna=False).to_dict()
            for t in targets
        },
        "best_by_target_split": best_by_target_split.to_dict(orient="records"),
        "utility_top": utility_top,
    }

    strat = results[
        (results["split"] == "stratified5")
        & (results["target"] == "gain_oracle_action")
        & results["macro_f1"].notna()
    ]
    group_rg = results[
        (results["split"] == "group_by_reuse_group")
        & (results["target"] == "gain_oracle_action")
        & results["macro_f1"].notna()
    ]

    best_strat = float(strat["macro_f1"].max()) if not strat.empty else np.nan
    best_group_rg = float(group_rg["macro_f1"].max()) if not group_rg.empty else np.nan

    summary["diagnostic"] = {
        "best_stratified_gain_oracle_macro_f1": best_strat,
        "best_group_by_reuse_group_gain_oracle_macro_f1": best_group_rg,
    }

    if best_strat >= 0.80 and (np.isnan(best_group_rg) or best_group_rg < 0.55):
        verdict = "TABULAR_BASELINE_STRONG_IN_DISTRIBUTION_WEAK_OUT_OF_REUSE_GROUP"
    elif best_strat >= 0.70 and best_group_rg >= 0.55:
        verdict = "TABULAR_BASELINE_HAS_GENERALIZATION_SIGNAL"
    elif best_strat >= 0.55:
        verdict = "TABULAR_BASELINE_PARTIAL"
    else:
        verdict = "TABULAR_BASELINE_WEAK"

    summary["verdict"] = verdict

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[SAVED]")
    print(f"  cv results : {OUT_CV}")
    print(f"  predictions: {OUT_PREDS}")
    print(f"  utility    : {OUT_UTILITY}")
    print(f"  summary    : {OUT_SUMMARY}")

    print("\n[VERDICT]")
    print(verdict)


if __name__ == "__main__":
    main()
