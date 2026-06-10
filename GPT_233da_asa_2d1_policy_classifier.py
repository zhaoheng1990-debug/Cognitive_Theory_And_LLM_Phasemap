# -*- coding: utf-8 -*-
"""
DA-ASA-2D.1 Policy Classifier Audit

Purpose
-------
Train and audit a policy-class classifier from DA-ASA-2D.0g feature table.

Recommended command on Windows:
  cd /d C:\\Users\\ZH\\Desktop\\AGI\\python_script
  python da_asa_2d1_policy_classifier.py ^
    --feature-file C:\\Users\\ZH\\Desktop\\AGI\\python_script\\da_asa_2d0g_outputs\\da_asa2d_policy_feature_numeric.csv

Outputs
-------
  da_asa_2d1_outputs/da_asa2d1_cv_summary.csv
  da_asa_2d1_outputs/da_asa2d1_feature_set_report.csv
  da_asa_2d1_outputs/da_asa2d1_verdict.json
  da_asa_2d1_outputs/da_asa2d1_predictions.csv
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


POLICY_ORDER = [
    "NO_INTERVENTION",
    "CORE_CLOSURE_UPDATE",
    "OVERRIDE_THREE_STAGE",
    "EQUAL_EVIDENCE_ORDER",
    "RESERVED_HALLUCINATION_CONTROLLER",
]

LEAKAGE_EXACT = {
    "policy_id",
}

# These are direct rule-label columns from condition -> policy. They must not be used.
LEAKAGE_PREFIXES = (
    "is_core_closure_target",
    "is_override_target",
    "is_equal_evidence_target",
    "is_hallucination_like",
    "is_source_claim",
    "is_protected_no_intervention",
)

LEAKAGE_SUBSTRINGS = (
    "row_index",
    "sample_index",
    "_id",      # numeric ids often encode row/graph identity; keep graph_id only for grouping, not feature
    "token_id",
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    df = pd.read_csv(path)
    if "policy_class" not in df.columns:
        raise RuntimeError("policy_class column not found. Please use DA-ASA-2D.0g numeric/table output.")
    if "graph_id" not in df.columns:
        print("[warn] graph_id not found; GroupKFold will be skipped.")
    return df


def is_numeric_feature(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and pd.api.types.is_numeric_dtype(df[col])


def is_leakage_col(col: str) -> bool:
    if col in LEAKAGE_EXACT:
        return True
    if any(col.startswith(p) for p in LEAKAGE_PREFIXES):
        return True
    # allow policy_outcome_*; block only obvious ids/indices
    low = col.lower()
    if any(s in low for s in LEAKAGE_SUBSTRINGS):
        # Do not block condition-independent scientific features containing "signed".
        if "signed" in low:
            return False
        return True
    return False


def clean_numeric_cols(df: pd.DataFrame, cols: List[str], min_nonnull_frac: float, allow_sparse: bool = False) -> List[str]:
    out = []
    for c in cols:
        if c not in df.columns:
            continue
        if not is_numeric_feature(df, c):
            continue
        if is_leakage_col(c):
            continue
        nonnull = float(df[c].notna().mean())
        if nonnull <= 0:
            continue
        if (not allow_sparse) and nonnull < min_nonnull_frac:
            continue
        # Drop constants among observed values.
        vals = df[c].dropna().values
        if len(vals) > 0 and np.nanstd(vals) == 0:
            continue
        out.append(c)
    # stable order, no duplicates
    return list(dict.fromkeys(out))


def get_feature_sets(df: pd.DataFrame, min_nonnull_frac: float) -> Dict[str, List[str]]:
    baseline_candidates = [
        "baseline_R_final",
        "baseline_pred_clean",
        "R_final",
        "baseline_margin",
        "boundary_distance_proxy",
        "baseline_trajectory_baseline_R_final",
        "baseline_trajectory_baseline_pred_clean",
        "baseline_trajectory_R_final",
        "baseline_trajectory_baseline_margin",
        "baseline_trajectory_boundary_distance_proxy",
    ]
    policy_candidates = [c for c in df.columns if c.startswith("policy_outcome_")]
    dsta_candidates = [c for c in df.columns if c.startswith("dsta_signed_attractor_")]
    direction_candidates = [c for c in df.columns if c.startswith("direction_spectrum_")]
    topk_candidates = [c for c in df.columns if c.startswith("topk_") or c.startswith("vim_")]

    baseline = clean_numeric_cols(df, baseline_candidates, min_nonnull_frac)
    policy = clean_numeric_cols(df, policy_candidates, min_nonnull_frac)
    dsta_dense = clean_numeric_cols(df, dsta_candidates, min_nonnull_frac)
    dsta_sparse = clean_numeric_cols(df, dsta_candidates, 0.0, allow_sparse=True)
    direction = clean_numeric_cols(df, direction_candidates, min_nonnull_frac)
    topk = clean_numeric_cols(df, topk_candidates, min_nonnull_frac)

    sets = {
        "baseline_only": baseline,
        "baseline_plus_policy_outcome": baseline + policy,
        "baseline_plus_dsta_dense": baseline + dsta_dense,
        "baseline_policy_dsta_dense": baseline + policy + dsta_dense,
        # Diagnostic only: sparse DSTA may partly encode merge/coverage artifacts.
        "baseline_policy_dsta_sparse_diagnostic": baseline + policy + dsta_sparse,
    }
    if direction:
        sets["baseline_policy_dsta_direction"] = baseline + policy + dsta_dense + direction
    if topk:
        sets["baseline_policy_dsta_topk"] = baseline + policy + dsta_dense + topk

    return {k: list(dict.fromkeys(v)) for k, v in sets.items() if len(v) > 0}


def make_models(random_state: int):
    return {
        "logreg_balanced": Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]),
        "rf_balanced": Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=random_state,
            )),
        ]),
    }


def score_fold(y_true, y_pred) -> Dict[str, float]:
    labels = sorted(pd.Series(y_true).unique().tolist())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def run_cv(
    df: pd.DataFrame,
    y: pd.Series,
    cols: List[str],
    model,
    cv_name: str,
    splits,
    feature_set_name: str,
    model_name: str,
) -> Tuple[List[Dict], pd.DataFrame, Dict]:
    rows = []
    pred_rows = []
    all_labels = sorted(y.unique().tolist())
    cm_total = np.zeros((len(all_labels), len(all_labels)), dtype=int)
    label_to_idx = {lab: i for i, lab in enumerate(all_labels)}

    X = df[cols]
    for fold, (tr, te) in enumerate(splits):
        # Skip invalid folds that lack more than one class in training.
        if y.iloc[tr].nunique() < 2:
            continue
        m = clone(model)
        m.fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict(X.iloc[te])
        sc = score_fold(y.iloc[te], pred)
        row = {
            "feature_set": feature_set_name,
            "model": model_name,
            "cv": cv_name,
            "fold": fold,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "n_features": int(len(cols)),
            **sc,
        }
        rows.append(row)
        cm = confusion_matrix(y.iloc[te], pred, labels=all_labels)
        cm_total += cm
        for idx, p in zip(te, pred):
            pred_rows.append({
                "row_index": int(idx),
                "sample_id": df.iloc[idx].get("sample_id", idx),
                "graph_id": df.iloc[idx].get("graph_id", ""),
                "condition": df.iloc[idx].get("condition", ""),
                "true_policy": y.iloc[idx],
                "pred_policy": p,
                "feature_set": feature_set_name,
                "model": model_name,
                "cv": cv_name,
                "fold": fold,
            })

    cm_dict = {
        "labels": all_labels,
        "matrix": cm_total.tolist(),
    }
    return rows, pd.DataFrame(pred_rows), cm_dict


def summarize_cv(rows: List[Dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    group_cols = ["feature_set", "model", "cv"]
    metrics = ["accuracy", "macro_f1", "balanced_accuracy"]
    parts = []
    for keys, sub in df.groupby(group_cols):
        rec = dict(zip(group_cols, keys))
        rec["n_folds"] = int(len(sub))
        rec["n_features"] = int(sub["n_features"].iloc[0])
        for m in metrics:
            rec[m + "_mean"] = float(sub[m].mean())
            rec[m + "_std"] = float(sub[m].std(ddof=0))
        parts.append(rec)
    return pd.DataFrame(parts).sort_values(["cv", "macro_f1_mean"], ascending=[True, False])


def feature_report(df: pd.DataFrame, feature_sets: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    for name, cols in feature_sets.items():
        for c in cols:
            rows.append({
                "feature_set": name,
                "feature": c,
                "nonnull_frac": float(df[c].notna().mean()) if c in df else 0.0,
                "n_unique_observed": int(df[c].dropna().nunique()) if c in df else 0,
            })
    return pd.DataFrame(rows)


def make_verdict(summary: pd.DataFrame, feature_sets: Dict[str, List[str]], df: pd.DataFrame) -> Dict:
    verdict = {
        "experiment": "DA-ASA-2D.1 Policy Classifier",
        "n_rows": int(len(df)),
        "policy_counts": df["policy_class"].value_counts().to_dict(),
        "feature_sets": {k: len(v) for k, v in feature_sets.items()},
        "notes": [],
    }
    if summary.empty:
        verdict["verdict"] = "FAIL_NO_VALID_CV"
        return verdict

    group = summary[summary["cv"] == "group"]
    if group.empty:
        best = summary.sort_values("macro_f1_mean", ascending=False).iloc[0]
    else:
        best = group.sort_values("macro_f1_mean", ascending=False).iloc[0]
    verdict["best_group_or_available"] = best.to_dict()

    # Compare baseline vs DSTA sparse diagnostic if present.
    def get_score(fs):
        sub = group[(group["feature_set"] == fs) & (group["model"] == best["model"])]
        if len(sub) == 0:
            sub = summary[(summary["feature_set"] == fs) & (summary["model"] == best["model"])]
        if len(sub) == 0:
            return None
        return float(sub["macro_f1_mean"].iloc[0])

    b = get_score("baseline_only")
    bp = get_score("baseline_plus_policy_outcome")
    bd = get_score("baseline_policy_dsta_dense")
    bs = get_score("baseline_policy_dsta_sparse_diagnostic")

    verdict["comparisons"] = {
        "baseline_macro_f1": b,
        "baseline_policy_macro_f1": bp,
        "baseline_policy_dsta_dense_macro_f1": bd,
        "baseline_policy_dsta_sparse_diagnostic_macro_f1": bs,
    }

    if bs is not None and b is not None and bs > b + 0.05:
        verdict["verdict"] = "PASS_LITE_DSTA_DIAGNOSTIC"
        verdict["notes"].append("Sparse DSTA diagnostic improves over baseline, but check missingness/merge artifacts before claiming DSTA causality.")
    elif bp is not None and b is not None and bp > b + 0.05:
        verdict["verdict"] = "PASS_LITE_POLICY_OUTCOME"
        verdict["notes"].append("Policy outcome features improve over baseline; DSTA contribution not established.")
    else:
        verdict["verdict"] = "WEAK_OR_FAIL_CURRENT_FEATURES"
        verdict["notes"].append("Current dense feature groups do not clearly improve policy-class prediction over baseline.")

    dsta_cols = [c for c in df.columns if c.startswith("dsta_signed_attractor_") and is_numeric_feature(df, c)]
    if dsta_cols:
        max_cov = max(float(df[c].notna().mean()) for c in dsta_cols)
        mean_cov = float(np.mean([df[c].notna().mean() for c in dsta_cols]))
        verdict["dsta_coverage"] = {"max_nonnull_frac": max_cov, "mean_nonnull_frac": mean_cov}
        if max_cov < 0.2:
            verdict["notes"].append("DSTA feature coverage is sparse; consider rebuilding 2D.0 with condition-level DSTA aggregation broadcast to all samples.")
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-file", required=True, help="DA-ASA-2D.0g numeric feature CSV")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--min-nonnull-frac", type=float, default=0.20)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()

    feature_file = Path(args.feature_file)
    out_dir = Path(args.out_dir) if args.out_dir else feature_file.parent.parent / "da_asa_2d1_outputs"
    ensure_dir(out_dir)

    df = load_table(feature_file)
    y = df["policy_class"].astype(str)
    feature_sets = get_feature_sets(df, args.min_nonnull_frac)

    print("[info] rows=", len(df))
    print("[info] policy counts:")
    print(y.value_counts().to_string())
    print("[info] feature sets:")
    for k, v in feature_sets.items():
        print(f"  {k}: {len(v)} features")

    models = make_models(args.random_state)
    all_rows = []
    all_preds = []
    cms = {}

    # CV splitters
    n_splits = min(args.n_splits, int(y.value_counts().min()))
    if n_splits < 2:
        n_splits = 2
    strat = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.random_state)

    group_splits = None
    if "graph_id" in df.columns and df["graph_id"].nunique() >= 2:
        n_groups = df["graph_id"].nunique()
        g_splits = min(args.n_splits, n_groups)
        group_splits = GroupKFold(n_splits=g_splits)

    for fs_name, cols in feature_sets.items():
        for model_name, model in models.items():
            rows, preds, cm = run_cv(
                df, y, cols, model,
                "stratified", strat.split(df[cols], y), fs_name, model_name,
            )
            all_rows.extend(rows)
            all_preds.append(preds)
            cms[f"{fs_name}__{model_name}__stratified"] = cm

            if group_splits is not None:
                rows, preds, cm = run_cv(
                    df, y, cols, model,
                    "group", group_splits.split(df[cols], y, df["graph_id"]), fs_name, model_name,
                )
                all_rows.extend(rows)
                all_preds.append(preds)
                cms[f"{fs_name}__{model_name}__group"] = cm

    cv_fold_df = pd.DataFrame(all_rows)
    cv_summary = summarize_cv(all_rows)
    feat_report = feature_report(df, feature_sets)
    pred_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    verdict = make_verdict(cv_summary, feature_sets, df)

    cv_fold_df.to_csv(out_dir / "da_asa2d1_cv_folds.csv", index=False, encoding="utf-8-sig")
    cv_summary.to_csv(out_dir / "da_asa2d1_cv_summary.csv", index=False, encoding="utf-8-sig")
    feat_report.to_csv(out_dir / "da_asa2d1_feature_set_report.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(out_dir / "da_asa2d1_predictions.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "da_asa2d1_confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump(cms, f, ensure_ascii=False, indent=2)
    with open(out_dir / "da_asa2d1_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)

    print("[done] outputs:", out_dir)
    if not cv_summary.empty:
        print(cv_summary.to_string(index=False))
    print("[verdict]", verdict.get("verdict"))


if __name__ == "__main__":
    main()
