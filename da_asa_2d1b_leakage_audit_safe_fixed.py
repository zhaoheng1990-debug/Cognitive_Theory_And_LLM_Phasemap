# -*- coding: utf-8 -*-
r"""
DA-ASA-2D.1B Leakage / Generalization Audit - SAFE VERSION

Purpose:
  Audit whether the DA-ASA-2D.1 perfect policy classifier is driven by
  condition-level broadcast DSTA fingerprints or by graph/sample-level DSTA signals.

Key fixes vs previous version:
  1. Skip any feature set with 0 usable features instead of crashing.
  2. Skip CV folds where train has <2 classes.
  3. In Leave-One-Condition-Out, explicitly mark held-out classes absent from train.
  4. Exclude direct label/condition indicator columns by default.
  5. Produce diagnostics for feature-set sizes and non-null coverage.

Run:
  python da_asa_2d1b_leakage_audit_safe.py ^
    --feature-file C:\Users\ZH\Desktop\AGI\python_script\da_asa_2d0h5_outputs\da_asa2d_policy_feature_numeric.csv ^
    --out-dir C:\Users\ZH\Desktop\AGI\python_script\da_asa_2d1b_safe_outputs
"""

import argparse
import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

LABEL_COL = "policy_class"

DIRECT_LEAK_PATTERNS = [
    r"^policy_id$",
    r"^is_core_closure_target$",
    r"^is_override_target$",
    r"^is_equal_evidence_target$",
    r"^is_hallucination_like$",
    r"^is_source_claim$",
    r"^is_protected_no_intervention$",
]

ID_OR_META_PATTERNS = [
    r"(^|_)row_index$", r"(^|_)id$", r"sample_id", r"graph_id", r"prompt_id",
    r"source_file", r"policy_class", r"condition", r"mechanism", r"risk_regime",
]

BROADCAST_DSTA_HINTS = [
    "_mean", "critical_", "commit_", "d_to_", "distance", "proj_", "between_",
    "relative_", "n_layers", "n_graph", "_std", "n_rows_agg",
]

GRAPH_DSTA_HINTS = [
    "center_signed_align_vs_clean", "center_reversal_score", "center_rotation_abs_deg",
    "center_signed_angle_deg", "delta_center_norm", "transport_signed_align_vs_clean",
    "transport_rotation_abs_deg", "transport_delta_norm", "delta_align_to_update",
    "delta_align_to_override",
]


def safe_mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def is_numeric_series(s):
    return pd.api.types.is_numeric_dtype(s)


def match_any(name, patterns):
    return any(re.search(p, name) for p in patterns)


def clean_feature_cols(df):
    numeric = [c for c in df.columns if is_numeric_series(df[c])]
    keep = []
    for c in numeric:
        if match_any(c, DIRECT_LEAK_PATTERNS):
            continue
        if match_any(c, ID_OR_META_PATTERNS):
            continue
        keep.append(c)
    return keep


def nonnull_filter(df, cols, min_nonnull_frac=0.01):
    out = []
    for c in cols:
        if c not in df.columns:
            continue
        frac = float(df[c].notna().mean())
        if frac >= min_nonnull_frac:
            out.append(c)
    return out


def build_feature_sets(df, min_nonnull_frac):
    all_num = clean_feature_cols(df)
    all_num = nonnull_filter(df, all_num, min_nonnull_frac)

    baseline = [c for c in all_num if (
        c.startswith("baseline_") or c in ["R_final", "baseline_margin", "boundary_distance_proxy", "baseline_R_final"]
    ) and not c.startswith("dsta_") and not c.startswith("policy_outcome") and not c.startswith("topk_vim")]

    policy = [c for c in all_num if c.startswith("policy_outcome")]
    dsta = [c for c in all_num if c.startswith("dsta_signed_attractor")]
    topk = [c for c in all_num if c.startswith("topk_vim")]

    # Strict graph-level DSTA: prefer graph/sample aligned raw signed metrics, not condition summaries.
    strict_graph_dsta = []
    for c in dsta:
        if any(h in c for h in BROADCAST_DSTA_HINTS):
            continue
        if any(h in c for h in GRAPH_DSTA_HINTS) or c.endswith("_layer") or c.endswith("_topk"):
            strict_graph_dsta.append(c)

    condition_broadcast_dsta = [c for c in dsta if c not in strict_graph_dsta]

    # If strict set is unexpectedly empty, keep it empty and let evaluator skip it.
    fs = {
        "baseline_only": baseline,
        "strict_graph_dsta_only": strict_graph_dsta,
        "baseline_plus_strict_graph_dsta": sorted(set(baseline + strict_graph_dsta)),
        "condition_broadcast_dsta_only": condition_broadcast_dsta,
        "baseline_plus_condition_broadcast_dsta": sorted(set(baseline + condition_broadcast_dsta)),
        "all_dsta_no_policy_outcome": dsta,
        "baseline_plus_all_dsta_no_policy_outcome": sorted(set(baseline + dsta)),
        "baseline_plus_topk": sorted(set(baseline + topk)),
        "baseline_plus_policy_outcome": sorted(set(baseline + policy)),
        # Diagnostic only: likely leakage because policy outcome is derived from known policies.
        "baseline_policy_dsta_diagnostic": sorted(set(baseline + policy + dsta)),
    }
    return {k: v for k, v in fs.items()}


def make_models(random_state):
    return {
        "logreg_balanced": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs")),
        ]),
        "rf_balanced": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
            )),
        ]),
    }


def eval_splits(df, cols, splits, split_name, model_name, model):
    y = df[LABEL_COL].astype(str).values
    if len(cols) == 0:
        return [], [], {"skipped": True, "reason": "zero_features"}

    rows = []
    pred_rows = []
    labels_all = sorted(pd.unique(y).tolist())

    X = df[cols].copy()
    for fold, (tr, te) in enumerate(splits):
        y_train = y[tr]
        y_test = y[te]
        train_classes = sorted(pd.unique(y_train).tolist())
        test_classes = sorted(pd.unique(y_test).tolist())
        missing_test_classes = sorted(set(test_classes) - set(train_classes))

        if len(train_classes) < 2:
            rows.append({
                "fold": fold, "split": split_name, "model": model_name,
                "n_train": len(tr), "n_test": len(te), "n_features": len(cols),
                "skipped": True, "reason": "train_has_less_than_two_classes",
                "train_classes": json.dumps(train_classes, ensure_ascii=False),
                "test_classes": json.dumps(test_classes, ensure_ascii=False),
                "missing_test_classes": json.dumps(missing_test_classes, ensure_ascii=False),
            })
            continue

        model.fit(X.iloc[tr], y_train)
        pred = model.predict(X.iloc[te])

        rows.append({
            "fold": fold,
            "split": split_name,
            "model": model_name,
            "n_train": len(tr),
            "n_test": len(te),
            "n_features": len(cols),
            "skipped": False,
            "accuracy": accuracy_score(y_test, pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "macro_f1": f1_score(y_test, pred, average="macro", labels=labels_all, zero_division=0),
            "weighted_f1": f1_score(y_test, pred, average="weighted", zero_division=0),
            "train_classes": json.dumps(train_classes, ensure_ascii=False),
            "test_classes": json.dumps(test_classes, ensure_ascii=False),
            "missing_test_classes": json.dumps(missing_test_classes, ensure_ascii=False),
        })
        idxs = df.index.values[te]
        for idx, yt, yp in zip(idxs, y_test, pred):
            pred_rows.append({
                "row_index": int(idx),
                "split": split_name,
                "model": model_name,
                "fold": fold,
                "true": yt,
                "pred": yp,
                "condition": str(df.loc[idx, "condition"]) if "condition" in df.columns else "",
                "graph_id": str(df.loc[idx, "graph_id"]) if "graph_id" in df.columns else "",
            })
    meta = {"skipped": False, "reason": ""}
    return rows, pred_rows, meta


def summarize(cv_rows):
    if not cv_rows:
        return pd.DataFrame()
    df = pd.DataFrame(cv_rows)
    valid = df[df.get("skipped", False) == False].copy()
    if valid.empty:
        return pd.DataFrame()
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    group_cols = ["feature_set", "split", "model"]
    out = []
    for keys, sub in valid.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row["n_folds_valid"] = len(sub)
        row["n_features"] = int(sub["n_features"].iloc[0])
        for m in metrics:
            row[m + "_mean"] = float(sub[m].mean())
            row[m + "_std"] = float(sub[m].std(ddof=0))
        out.append(row)
    return pd.DataFrame(out).sort_values(["macro_f1_mean", "balanced_accuracy_mean"], ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-file", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--min-nonnull-frac", type=float, default=0.01)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()

    feature_file = Path(args.feature_file)
    out_dir = Path(args.out_dir) if args.out_dir else feature_file.parent / "da_asa_2d1b_safe_outputs"
    safe_mkdir(out_dir)

    df = pd.read_csv(feature_file)
    # Drop duplicate columns defensively.
    df = df.loc[:, ~df.columns.duplicated()].copy()
    if LABEL_COL not in df.columns:
        raise RuntimeError(f"Missing required target column: {LABEL_COL}")
    if "condition" not in df.columns:
        raise RuntimeError("Missing condition column; cannot run leakage audit.")

    # Feature sets.
    feature_sets = build_feature_sets(df, args.min_nonnull_frac)
    fs_report = []
    for name, cols in feature_sets.items():
        nonnull_mean = float(df[cols].notna().mean().mean()) if cols else 0.0
        fs_report.append({"feature_set": name, "n_features": len(cols), "mean_nonnull_frac": nonnull_mean, "features": cols})
    pd.DataFrame([{k:v for k,v in r.items() if k != "features"} for r in fs_report]).to_csv(out_dir / "da_asa2d1b_feature_set_report.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "da_asa2d1b_feature_sets.json", "w", encoding="utf-8") as f:
        json.dump(fs_report, f, ensure_ascii=False, indent=2)

    y = df[LABEL_COL].astype(str).values
    models = make_models(args.random_state)

    # Splitters.
    splitters = []
    min_class_count = pd.Series(y).value_counts().min()
    n_splits = max(2, min(args.n_splits, int(min_class_count)))
    splitters.append(("stratified", list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.random_state).split(df, y))))

    if "graph_id" in df.columns and df["graph_id"].nunique() >= 2:
        g_n = max(2, min(args.n_splits, int(df["graph_id"].nunique())))
        splitters.append(("group_graph", list(GroupKFold(n_splits=g_n).split(df, y, groups=df["graph_id"]))))

    # Leave-one-condition-out. Keep as diagnostics; some held-out labels absent in train are expected.
    loco_splits = []
    for cond in sorted(df["condition"].astype(str).unique()):
        te = np.where(df["condition"].astype(str).values == cond)[0]
        tr = np.where(df["condition"].astype(str).values != cond)[0]
        if len(te) and len(tr):
            loco_splits.append((tr, te))
    splitters.append(("leave_one_condition", loco_splits))

    all_rows = []
    all_preds = []
    skipped_feature_sets = []
    for fs_name, cols in feature_sets.items():
        if len(cols) == 0:
            skipped_feature_sets.append({"feature_set": fs_name, "reason": "zero_features"})
            continue
        for split_name, splits in splitters:
            for model_name, model in models.items():
                rows, preds, meta = eval_splits(df, cols, splits, split_name, model_name, model)
                for r in rows:
                    r["feature_set"] = fs_name
                for p in preds:
                    p["feature_set"] = fs_name
                all_rows.extend(rows)
                all_preds.extend(preds)

    folds_df = pd.DataFrame(all_rows)
    preds_df = pd.DataFrame(all_preds)
    summary_df = summarize(all_rows)

    folds_df.to_csv(out_dir / "da_asa2d1b_cv_folds.csv", index=False, encoding="utf-8-sig")
    preds_df.to_csv(out_dir / "da_asa2d1b_predictions.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "da_asa2d1b_cv_summary.csv", index=False, encoding="utf-8-sig")

    best = summary_df.iloc[0].to_dict() if not summary_df.empty else {}

    # Leakage interpretation helpers.
    def best_macro(feature_set, split="group_graph"):
        if summary_df.empty:
            return None
        sub = summary_df[(summary_df["feature_set"] == feature_set) & (summary_df["split"] == split)]
        if sub.empty:
            return None
        return float(sub["macro_f1_mean"].max())

    comparison = {
        "baseline_only_group_macro_f1": best_macro("baseline_only"),
        "strict_graph_dsta_only_group_macro_f1": best_macro("strict_graph_dsta_only"),
        "condition_broadcast_dsta_only_group_macro_f1": best_macro("condition_broadcast_dsta_only"),
        "all_dsta_no_policy_outcome_group_macro_f1": best_macro("all_dsta_no_policy_outcome"),
        "baseline_plus_all_dsta_no_policy_outcome_group_macro_f1": best_macro("baseline_plus_all_dsta_no_policy_outcome"),
        "baseline_plus_policy_outcome_group_macro_f1": best_macro("baseline_plus_policy_outcome"),
    }

    # Detect absent target classes in LOCO.
    loco = folds_df[folds_df.get("split", "") == "leave_one_condition"] if not folds_df.empty else pd.DataFrame()
    absent_loco = []
    if not loco.empty and "missing_test_classes" in loco.columns:
        for _, row in loco.iterrows():
            try:
                miss = json.loads(row.get("missing_test_classes", "[]"))
            except Exception:
                miss = []
            if miss:
                absent_loco.append({
                    "feature_set": row.get("feature_set"),
                    "model": row.get("model"),
                    "fold": int(row.get("fold", -1)),
                    "missing_test_classes": miss,
                })

    verdict = "UNKNOWN"
    b = comparison.get("baseline_only_group_macro_f1")
    sg = comparison.get("strict_graph_dsta_only_group_macro_f1")
    cb = comparison.get("condition_broadcast_dsta_only_group_macro_f1")
    ad = comparison.get("all_dsta_no_policy_outcome_group_macro_f1")
    if cb is not None and sg is not None:
        if cb >= 0.95 and sg < 0.70:
            verdict = "CONDITION_BROADCAST_DIAGNOSTIC_LIKELY"
        elif sg >= 0.80:
            verdict = "GRAPH_LEVEL_DSTA_SIGNAL_STRONG"
        elif ad is not None and b is not None and ad > b + 0.15:
            verdict = "DSTA_SIGNAL_MIXED_BUT_POSITIVE"
        else:
            verdict = "NO_CLEAR_DSTA_GENERALIZATION"

    out = {
        "experiment": "DA-ASA-2D.1B Leakage Audit SAFE",
        "feature_file": str(feature_file),
        "n_rows": int(len(df)),
        "policy_counts": df[LABEL_COL].value_counts().to_dict(),
        "condition_counts": df["condition"].astype(str).value_counts().to_dict(),
        "feature_sets": {k: len(v) for k, v in feature_sets.items()},
        "skipped_feature_sets": skipped_feature_sets,
        "best": best,
        "comparison": comparison,
        "leave_one_condition_absent_class_warnings_count": len(absent_loco),
        "leave_one_condition_absent_class_warnings_preview": absent_loco[:20],
        "verdict": verdict,
        "outputs": {
            "cv_summary": str(out_dir / "da_asa2d1b_cv_summary.csv"),
            "cv_folds": str(out_dir / "da_asa2d1b_cv_folds.csv"),
            "predictions": str(out_dir / "da_asa2d1b_predictions.csv"),
            "feature_set_report": str(out_dir / "da_asa2d1b_feature_set_report.csv"),
            "feature_sets_json": str(out_dir / "da_asa2d1b_feature_sets.json"),
        }
    }
    with open(out_dir / "da_asa2d1b_verdict.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
