# -*- coding: utf-8 -*-
"""
DA-ASA-2D.1B Leakage / Generalization Audit

Purpose:
  Re-audit 2D.1 after dense condition-level DSTA aggregation.
  The previous 2D.1 may be inflated because many DSTA features are broadcast by condition,
  while policy_class is itself deterministically derived from condition.

This script compares:
  1) baseline_only
  2) strict_graph_dsta_only: graph-paired DSTA columns only, avoiding condition-level broadcast summaries
  3) baseline_plus_strict_graph_dsta
  4) condition_broadcast_dsta_only: likely condition-level diagnostic/leakage-prone features
  5) all_dsta_no_policy_outcome

CV modes:
  - StratifiedKFold rows
  - GroupKFold by graph_id
  - LeaveOneConditionOut diagnostic (marked infeasible when held-out class absent from train)

Outputs:
  da_asa2d1b_cv_summary.csv
  da_asa2d1b_cv_folds.csv
  da_asa2d1b_feature_set_report.csv
  da_asa2d1b_loco_summary.csv
  da_asa2d1b_verdict.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LABEL_ORDER = [
    "CORE_CLOSURE_UPDATE",
    "EQUAL_EVIDENCE_ORDER",
    "NO_INTERVENTION",
    "OVERRIDE_THREE_STAGE",
]

ALWAYS_EXCLUDE_SUBSTRINGS = [
    "policy_id", "policy_class", "condition", "sample_id", "graph_id", "prompt_id", "id",
    "is_core_closure_target", "is_override_target", "is_equal_evidence_target",
    "is_hallucination_like", "is_source_claim", "is_protected_no_intervention",
    "policy_outcome",  # excluded except in explicit policy-outcome comparisons; avoids direct outcome leakage
]

STRICT_GRAPH_DSTA_COLS = [
    "dsta_signed_attractor_center_signed_align_vs_clean",
    "dsta_signed_attractor_center_reversal_score",
    "dsta_signed_attractor_center_rotation_abs_deg",
    "dsta_signed_attractor_center_signed_angle_deg",
    "dsta_signed_attractor_delta_center_norm",
]

BASELINE_HINTS = [
    "baseline_R_final", "baseline_pred_clean", "R_final", "baseline_margin", "boundary_distance_proxy"
]


def safe_mkdir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def infer_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    def allowed(c: str) -> bool:
        return not any(s in c for s in ALWAYS_EXCLUDE_SUBSTRINGS)

    baseline = [c for c in numeric_cols if allowed(c) and any(h == c or c.endswith(h) for h in BASELINE_HINTS)]
    # Deduplicate but preserve order
    baseline = list(dict.fromkeys(baseline))

    strict_graph_dsta = [c for c in STRICT_GRAPH_DSTA_COLS if c in numeric_cols and allowed(c)]

    all_dsta = [c for c in numeric_cols if allowed(c) and c.startswith("dsta_signed_attractor_")]
    condition_broadcast = [
        c for c in all_dsta
        if c not in strict_graph_dsta
        and (
            c.endswith("_mean") or "critical_20_22" in c or "commit_23_26" in c
            or "d_to_" in c or c in [
                "dsta_signed_attractor_distance",
                "dsta_signed_attractor_d_x_a",
                "dsta_signed_attractor_d_x_b",
                "dsta_signed_attractor_d_a_b",
                "dsta_signed_attractor_between_excess",
                "dsta_signed_attractor_relative_position_from_a",
                "dsta_signed_attractor_proj_projection_t",
                "dsta_signed_attractor_proj_axis_length",
                "dsta_signed_attractor_proj_perpendicular_residual",
                "dsta_signed_attractor_proj_normalized_perp",
            ]
        )
    ]

    feature_sets = {
        "baseline_only": baseline,
        "strict_graph_dsta_only": strict_graph_dsta,
        "baseline_plus_strict_graph_dsta": baseline + strict_graph_dsta,
        "condition_broadcast_dsta_only": condition_broadcast,
        "all_dsta_no_policy_outcome": all_dsta,
        "baseline_plus_all_dsta_no_policy_outcome": baseline + all_dsta,
    }
    return {k: list(dict.fromkeys(v)) for k, v in feature_sets.items() if len(v) > 0}


def make_models(random_state: int):
    return {
        "logreg_balanced": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="lbfgs",
                C=1.0,
                random_state=random_state,
            )),
        ]),
        "rf_balanced": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=4,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            )),
        ]),
    }


def eval_cv(df, features, y, splitter, cv_name, model_name, model):
    rows = []
    pred_records = []
    X = df[features]
    for fold, (tr, te) in enumerate(splitter, start=1):
        y_train = y.iloc[tr]
        y_test = y.iloc[te]
        model.fit(X.iloc[tr], y_train)
        pred = pd.Series(model.predict(X.iloc[te]), index=y_test.index)
        rows.append({
            "fold": fold,
            "cv": cv_name,
            "model": model_name,
            "accuracy": accuracy_score(y_test, pred),
            "macro_f1": f1_score(y_test, pred, average="macro", labels=LABEL_ORDER, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "n_test": len(te),
        })
        for idx, yt, yp in zip(y_test.index, y_test, pred):
            pred_records.append({
                "row_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                "fold": fold,
                "cv": cv_name,
                "model": model_name,
                "y_true": yt,
                "y_pred": yp,
            })
    return rows, pred_records


def summarize(rows: List[dict], keys: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    agg = df.groupby(keys).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"), balanced_accuracy_std=("balanced_accuracy", "std"),
        n_folds=("fold", "count"),
        n_test_total=("n_test", "sum"),
    ).reset_index()
    return agg


def leave_one_condition_out(df, features, y, model_name, model) -> pd.DataFrame:
    X = df[features]
    records = []
    for cond in sorted(df["condition"].astype(str).unique()):
        te_mask = df["condition"].astype(str).eq(cond).values
        tr_mask = ~te_mask
        y_train = y.loc[tr_mask]
        y_test = y.loc[te_mask]
        held_label = sorted(y_test.unique().tolist())
        feasible = all(lbl in set(y_train.unique()) for lbl in held_label)
        rec = {
            "condition": cond,
            "model": model_name,
            "n_test": int(te_mask.sum()),
            "heldout_labels": ",".join(held_label),
            "feasible_class_seen_in_train": bool(feasible),
        }
        if feasible:
            model.fit(X.loc[tr_mask], y_train)
            pred = model.predict(X.loc[te_mask])
            rec.update({
                "accuracy": accuracy_score(y_test, pred),
                "macro_f1": f1_score(y_test, pred, average="macro", labels=LABEL_ORDER, zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            })
        else:
            rec.update({"accuracy": np.nan, "macro_f1": np.nan, "balanced_accuracy": np.nan})
        records.append(rec)
    return pd.DataFrame(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-file", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()

    feature_path = Path(args.feature_file)
    out_dir = args.out_dir or str(feature_path.parent.parent / "da_asa_2d1b_outputs")
    safe_mkdir(out_dir)

    df = pd.read_csv(feature_path)
    if "policy_class" not in df.columns:
        raise RuntimeError("policy_class column missing from feature file")
    if "graph_id" not in df.columns:
        raise RuntimeError("graph_id column missing from feature file")
    if "condition" not in df.columns:
        raise RuntimeError("condition column missing from feature file")

    y = df["policy_class"].astype(str)
    feature_sets = infer_feature_sets(df)
    report_rows = []
    for fs, cols in feature_sets.items():
        for c in cols:
            report_rows.append({
                "feature_set": fs,
                "feature": c,
                "nonnull_frac": float(df[c].notna().mean()),
                "n_unique": int(df[c].nunique(dropna=True)),
            })
    pd.DataFrame(report_rows).to_csv(Path(out_dir)/"da_asa2d1b_feature_set_report.csv", index=False)

    models = make_models(args.random_state)
    all_fold_rows = []
    all_pred_rows = []
    loco_all = []

    for fs_name, cols in feature_sets.items():
        for model_name, model in models.items():
            skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
            splitter = skf.split(df[cols], y)
            rows, preds = eval_cv(df, cols, y, splitter, "stratified", model_name, model)
            for r in rows: r["feature_set"] = fs_name; r["n_features"] = len(cols)
            for p in preds: p["feature_set"] = fs_name
            all_fold_rows.extend(rows); all_pred_rows.extend(preds)

            gkf = GroupKFold(n_splits=args.n_splits)
            splitter = gkf.split(df[cols], y, groups=df["graph_id"].astype(str))
            rows, preds = eval_cv(df, cols, y, splitter, "group_graph", model_name, model)
            for r in rows: r["feature_set"] = fs_name; r["n_features"] = len(cols)
            for p in preds: p["feature_set"] = fs_name
            all_fold_rows.extend(rows); all_pred_rows.extend(preds)

            loco = leave_one_condition_out(df, cols, y, model_name, model)
            loco.insert(0, "feature_set", fs_name)
            loco.insert(1, "n_features", len(cols))
            loco_all.append(loco)

    folds_df = pd.DataFrame(all_fold_rows)
    folds_df.to_csv(Path(out_dir)/"da_asa2d1b_cv_folds.csv", index=False)
    pd.DataFrame(all_pred_rows).to_csv(Path(out_dir)/"da_asa2d1b_predictions.csv", index=False)
    summary = summarize(folds_df, ["feature_set", "model", "cv", "n_features"])
    summary.to_csv(Path(out_dir)/"da_asa2d1b_cv_summary.csv", index=False)

    loco_df = pd.concat(loco_all, ignore_index=True) if loco_all else pd.DataFrame()
    loco_df.to_csv(Path(out_dir)/"da_asa2d1b_loco_summary.csv", index=False)

    # Best graph-cv row, but interpret carefully.
    graph_sum = summary[summary["cv"].eq("group_graph")].copy()
    best = graph_sum.sort_values(["macro_f1_mean", "balanced_accuracy_mean", "accuracy_mean"], ascending=False).head(1)
    best_rec = best.to_dict("records")[0] if not best.empty else {}

    # Leakage warnings.
    baseline = graph_sum[graph_sum["feature_set"].eq("baseline_only")]["macro_f1_mean"].max() if not graph_sum.empty else np.nan
    strict = graph_sum[graph_sum["feature_set"].eq("baseline_plus_strict_graph_dsta")]["macro_f1_mean"].max() if not graph_sum.empty else np.nan
    broadcast = graph_sum[graph_sum["feature_set"].eq("condition_broadcast_dsta_only")]["macro_f1_mean"].max() if not graph_sum.empty else np.nan

    impossible_loco = loco_df[loco_df["feasible_class_seen_in_train"].eq(False)] if not loco_df.empty else pd.DataFrame()

    verdict = {
        "experiment": "DA-ASA-2D.1B Leakage / Generalization Audit",
        "n_rows": int(len(df)),
        "policy_counts": y.value_counts().to_dict(),
        "feature_sets": {k: len(v) for k, v in feature_sets.items()},
        "best_group_graph": best_rec,
        "comparisons_group_macro_f1": {
            "baseline_only": None if pd.isna(baseline) else float(baseline),
            "baseline_plus_strict_graph_dsta": None if pd.isna(strict) else float(strict),
            "condition_broadcast_dsta_only": None if pd.isna(broadcast) else float(broadcast),
        },
        "leave_one_condition_out_infeasible_conditions": impossible_loco[["feature_set", "model", "condition", "heldout_labels"]].drop_duplicates().to_dict("records") if not impossible_loco.empty else [],
        "interpretation": [
            "If condition_broadcast_dsta_only is near-perfect while strict_graph_dsta is not, the previous 2D.1 result is condition-level diagnostic rather than sample-level generalization.",
            "Leave-one-condition-out is partly infeasible because OVERRIDE_THREE_STAGE and EQUAL_EVIDENCE_ORDER each currently have only one condition; more heldout families are required for true unseen-policy generalization.",
        ],
    }
    with open(Path(out_dir)/"da_asa2d1b_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)

    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    print(f"\n[done] outputs: {out_dir}")

if __name__ == "__main__":
    main()
