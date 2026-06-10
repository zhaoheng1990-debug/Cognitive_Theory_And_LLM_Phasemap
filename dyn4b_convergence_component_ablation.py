# -*- coding: utf-8 -*-
r"""
DYN-4B: Convergence Dynamics Component Ablation Audit

Purpose
-------
DYN-4 showed that no-answer MultiProjection / TopK + DYN4 FullDynamics strongly predicts
GenError and commitment magnitude.

DYN-4B asks:

    Which part of DYN4 is carrying the gain?

Components:
  - TopK topology
  - Raw dynamics
  - Boundary residence / boundary stabilization
  - Late stabilization
  - Velocity / curvature
  - Order precursor
  - Order residual
  - Full no-answer multiprojection

Input:
  C:\Users\ZH\Desktop\AGI\outputs\dyn4_outputs\dyn4_features_augmented.csv

Outputs:
  C:\Users\ZH\Desktop\AGI\outputs\dyn4b_outputs\dyn4b_component_summary.csv
  C:\Users\ZH\Desktop\AGI\outputs\dyn4b_outputs\dyn4b_incremental_summary.json
  C:\Users\ZH\Desktop\AGI\outputs\dyn4b_outputs\dyn4b_verdict.json
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    r2_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_SEED = 42
N_SPLITS = 5

INPUT_FEATURES = Path(r"C:\Users\ZH\Desktop\AGI\outputs\dyn4_outputs\dyn4_features_augmented.csv")
OUT_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\dyn4b_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def numeric_cols(df, cols):
    return [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def cols_by_prefix(df, prefixes):
    out = []
    for c in df.columns:
        for p in prefixes:
            if c.startswith(p):
                out.append(c)
                break
    return numeric_cols(df, out)


def cols_containing(df, patterns):
    out = []
    for c in df.columns:
        lc = c.lower()
        if any(p.lower() in lc for p in patterns):
            out.append(c)
    return numeric_cols(df, out)


def unique_keep_order(xs):
    return list(dict.fromkeys(xs))


def cv_indices(df, y=None, group_col="graph_id", stratify=False):
    n = len(df)
    groups = df[group_col].values if group_col in df.columns else None
    if groups is not None and len(np.unique(groups)) >= N_SPLITS:
        cv = GroupKFold(n_splits=N_SPLITS)
        y_dummy = np.zeros(n) if y is None else y
        return list(cv.split(np.zeros(n), y_dummy, groups=groups))

    if stratify and y is not None:
        vals, counts = np.unique(y, return_counts=True)
        if len(vals) >= 2 and counts.min() >= N_SPLITS:
            cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
            return list(cv.split(np.zeros(n), y))

    cv = KFold(n_splits=min(N_SPLITS, n), shuffle=True, random_state=RANDOM_SEED)
    return list(cv.split(np.zeros(n)))


def make_binary_classifier():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_SEED,
        )),
    ])


def make_multiclass_classifier():
    base = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_SEED,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", OneVsRestClassifier(base)),
    ])


def make_regressor():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=1.0)),
    ])


def eval_binary(df, cols, target="gen_error", group_col="graph_id"):
    cols = numeric_cols(df, cols)
    if target not in df.columns or not cols:
        return {"valid": False, "reason": "missing_target_or_features"}
    d = df.dropna(subset=[target]).copy()
    if len(d) < 10:
        return {"valid": False, "reason": "too_few_rows", "n": len(d)}
    y = d[target].astype(int).values
    if len(np.unique(y)) < 2:
        return {"valid": False, "reason": "single_class", "n": len(d)}
    X = d[cols].replace([np.inf, -np.inf], np.nan).values

    prob = np.zeros(len(d), dtype=float)
    pred = np.zeros(len(d), dtype=int)
    used = 0
    for train_idx, test_idx in cv_indices(d, y, group_col=group_col, stratify=True):
        if len(np.unique(y[train_idx])) < 2:
            continue
        clf = make_binary_classifier()
        clf.fit(X[train_idx], y[train_idx])
        p = clf.predict_proba(X[test_idx])[:, 1]
        prob[test_idx] = p
        pred[test_idx] = (p >= 0.5).astype(int)
        used += 1

    if used == 0:
        return {"valid": False, "reason": "no_valid_folds", "n": len(d)}
    try:
        auc = roc_auc_score(y, prob)
    except Exception:
        auc = np.nan
    return {
        "valid": True,
        "n": int(len(d)),
        "n_features": int(len(cols)),
        "auc": safe_float(auc),
        "acc": safe_float(accuracy_score(y, pred)),
        "f1": safe_float(f1_score(y, pred, zero_division=0)),
        "bal_acc": safe_float(balanced_accuracy_score(y, pred)),
    }


def eval_multiclass(df, cols, target="submechanism", group_col="graph_id"):
    cols = numeric_cols(df, cols)
    if target not in df.columns or not cols:
        return {"valid": False, "reason": "missing_target_or_features"}
    d = df.dropna(subset=[target]).copy()
    if len(d) < 10:
        return {"valid": False, "reason": "too_few_rows", "n": len(d)}
    y = d[target].astype(str).values
    if len(np.unique(y)) < 2:
        return {"valid": False, "reason": "single_class", "n": len(d)}
    X = d[cols].replace([np.inf, -np.inf], np.nan).values

    pred = np.array([""] * len(d), dtype=object)
    used = 0
    for train_idx, test_idx in cv_indices(d, y, group_col=group_col):
        if len(np.unique(y[train_idx])) < 2:
            continue
        clf = make_multiclass_classifier()
        clf.fit(X[train_idx], y[train_idx])
        pred[test_idx] = clf.predict(X[test_idx])
        used += 1

    if used == 0:
        return {"valid": False, "reason": "no_valid_folds", "n": len(d)}

    return {
        "valid": True,
        "n": int(len(d)),
        "n_features": int(len(cols)),
        "n_classes": int(len(np.unique(y))),
        "acc": safe_float(accuracy_score(y, pred)),
        "macro_f1": safe_float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": safe_float(f1_score(y, pred, average="weighted", zero_division=0)),
        "bal_acc": safe_float(balanced_accuracy_score(y, pred)),
    }


def eval_regression(df, cols, target="commit_abs_final_margin", group_col="graph_id"):
    cols = numeric_cols(df, cols)
    if target not in df.columns or not cols:
        return {"valid": False, "reason": "missing_target_or_features"}
    d = df.dropna(subset=[target]).copy()
    if len(d) < 10:
        return {"valid": False, "reason": "too_few_rows", "n": len(d)}
    y = d[target].astype(float).values
    X = d[cols].replace([np.inf, -np.inf], np.nan).values

    pred = np.zeros(len(d), dtype=float)
    used = 0
    for train_idx, test_idx in cv_indices(d, None, group_col=group_col):
        reg = make_regressor()
        reg.fit(X[train_idx], y[train_idx])
        pred[test_idx] = reg.predict(X[test_idx])
        used += 1
    if used == 0:
        return {"valid": False, "reason": "no_valid_folds", "n": len(d)}

    try:
        corr = np.corrcoef(y, pred)[0, 1]
    except Exception:
        corr = np.nan

    return {
        "valid": True,
        "n": int(len(d)),
        "n_features": int(len(cols)),
        "r2": safe_float(r2_score(y, pred)),
        "corr": safe_float(corr),
    }


def mutual_info_top(df, cols, target="gen_error", top_n=30):
    cols = numeric_cols(df, cols)
    if not cols or target not in df.columns:
        return pd.DataFrame()
    d = df.dropna(subset=[target]).copy()
    y = d[target].astype(int).values
    if len(np.unique(y)) < 2:
        return pd.DataFrame()
    X = d[cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0).values
    try:
        mi = mutual_info_classif(X, y, random_state=RANDOM_SEED)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame({"feature": cols, "mutual_info": mi}).sort_values("mutual_info", ascending=False).head(top_n)


def build_groups(df):
    topk = cols_by_prefix(df, ["topk_"])

    answer = numeric_cols(df, [
        "ans_margin_L20", "ans_margin_L21", "ans_margin_L22",
        "ans_margin_L23", "ans_margin_L24", "ans_margin_L25",
        "ans_margin_main_mean", "ans_margin_main_slope", "ans_margin_main_area",
        "ans_margin_main_std", "ans_margin_main_minabs", "ans_margin_first_flip",
        "final_margin_C_minus_E",
        "commit_abs_final_margin", "commit_high_abs_margin",
    ])

    raw_dyn = [c for c in cols_by_prefix(df, ["dyn_", "dyn2_"]) if "__" not in c]
    dyn4_boundary_late = cols_by_prefix(df, ["dyn4_"])
    order_precursor = cols_by_prefix(df, ["order_precursor_"])
    order_residual = cols_containing(df, ["__opresid"])
    residual_dyn = cols_containing(df, ["__resid"])

    # Component definitions
    boundary_residence = numeric_cols(df, [
        "dyn4_boundary_residence_q25",
        "dyn4_boundary_residence_q50",
        "dyn4_abs_margin_area",
        "dyn4_margin_area",
    ])

    late_stabilization = numeric_cols(df, [
        "dyn4_late_minus_early_abs",
        "dyn4_late_over_early_abs",
        "dyn4_abs_margin_slope",
        "dyn4_final_proxy_abs_L25",
        "dyn4_start_abs_L20",
    ])

    velocity_curvature = unique_keep_order(
        cols_containing(df, [
            "velocity", "accel", "curvature", "dmargin", "second", "sign_changes"
        ])
    )

    dyn4_core = unique_keep_order(dyn4_boundary_late + order_precursor)
    dyn4_full = unique_keep_order(raw_dyn + dyn4_boundary_late + order_precursor + order_residual)
    no_answer_mpr = unique_keep_order(topk + dyn4_full)
    with_answer_mpr = unique_keep_order(topk + dyn4_full + answer)

    groups = {
        "TopK": topk,
        "AnswerReadout": answer,
        "RawDynamics": raw_dyn,
        "BoundaryResidence": boundary_residence,
        "LateStabilization": late_stabilization,
        "VelocityCurvature": velocity_curvature,
        "OrderPrecursor": order_precursor,
        "OrderResidual": order_residual,
        "ResidualDynamics": residual_dyn,
        "DYN4_BoundaryLate": dyn4_boundary_late,
        "DYN4_Core": dyn4_core,
        "DYN4_FullDynamics": dyn4_full,
        "TopK_plus_BoundaryResidence": unique_keep_order(topk + boundary_residence),
        "TopK_plus_LateStabilization": unique_keep_order(topk + late_stabilization),
        "TopK_plus_VelocityCurvature": unique_keep_order(topk + velocity_curvature),
        "TopK_plus_OrderPrecursor": unique_keep_order(topk + order_precursor),
        "TopK_plus_OrderResidual": unique_keep_order(topk + order_residual),
        "TopK_plus_DYN4_BoundaryLate": unique_keep_order(topk + dyn4_boundary_late),
        "TopK_plus_DYN4_Core": unique_keep_order(topk + dyn4_core),
        "TopK_plus_DYN4_FullDynamics": unique_keep_order(topk + dyn4_full),
        "MultiProjection_DYN4_NoAnswer": no_answer_mpr,
        "MultiProjection_DYN4_WithAnswer": with_answer_mpr,
    }

    return {k: numeric_cols(df, v) for k, v in groups.items() if len(numeric_cols(df, v)) > 0}


def main():
    if not INPUT_FEATURES.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_FEATURES}")

    df = pd.read_csv(INPUT_FEATURES).replace([np.inf, -np.inf], np.nan)
    groups = build_groups(df)

    rows = []
    for name, cols in groups.items():
        gen = eval_binary(df, cols, target="gen_error", group_col="graph_id")
        mech = eval_multiclass(df, cols, target="mechanism", group_col="graph_id")
        subm = eval_multiclass(df, cols, target="submechanism", group_col="graph_id")
        commit = eval_regression(df, cols, target="commit_abs_final_margin", group_col="graph_id")

        rows.append({
            "feature_group": name,
            "n_features": len(cols),
            "gen_auc": gen.get("auc", np.nan),
            "gen_f1": gen.get("f1", np.nan),
            "gen_acc": gen.get("acc", np.nan),
            "mechanism_macro_f1": mech.get("macro_f1", np.nan),
            "submechanism_macro_f1": subm.get("macro_f1", np.nan),
            "commit_abs_r2": commit.get("r2", np.nan),
            "commit_abs_corr": commit.get("corr", np.nan),
            "valid_gen": gen.get("valid", False),
            "valid_mech": mech.get("valid", False),
            "valid_submech": subm.get("valid", False),
            "valid_commit": commit.get("valid", False),
        })

    summary = pd.DataFrame(rows).sort_values(
        ["gen_auc", "commit_abs_r2", "submechanism_macro_f1"],
        ascending=False,
        na_position="last",
    )

    def metric(group, key):
        r = summary[summary["feature_group"] == group]
        if len(r) == 0:
            return np.nan
        return safe_float(r.iloc[0].get(key, np.nan))

    inc = {
        "topk_gen_auc": metric("TopK", "gen_auc"),
        "answer_gen_auc": metric("AnswerReadout", "gen_auc"),
        "raw_dyn_gen_auc": metric("RawDynamics", "gen_auc"),
        "boundary_residence_gen_auc": metric("BoundaryResidence", "gen_auc"),
        "late_stabilization_gen_auc": metric("LateStabilization", "gen_auc"),
        "velocity_curvature_gen_auc": metric("VelocityCurvature", "gen_auc"),
        "order_precursor_gen_auc": metric("OrderPrecursor", "gen_auc"),
        "order_residual_gen_auc": metric("OrderResidual", "gen_auc"),
        "dyn4_full_gen_auc": metric("DYN4_FullDynamics", "gen_auc"),
        "topk_plus_boundary_gen_auc": metric("TopK_plus_BoundaryResidence", "gen_auc"),
        "topk_plus_late_gen_auc": metric("TopK_plus_LateStabilization", "gen_auc"),
        "topk_plus_velocity_gen_auc": metric("TopK_plus_VelocityCurvature", "gen_auc"),
        "topk_plus_order_precursor_gen_auc": metric("TopK_plus_OrderPrecursor", "gen_auc"),
        "topk_plus_order_residual_gen_auc": metric("TopK_plus_OrderResidual", "gen_auc"),
        "topk_plus_dyn4_full_gen_auc": metric("TopK_plus_DYN4_FullDynamics", "gen_auc"),
        "mpr_no_answer_gen_auc": metric("MultiProjection_DYN4_NoAnswer", "gen_auc"),
        "mpr_with_answer_gen_auc": metric("MultiProjection_DYN4_WithAnswer", "gen_auc"),

        "topk_commit_r2": metric("TopK", "commit_abs_r2"),
        "answer_commit_r2": metric("AnswerReadout", "commit_abs_r2"),
        "raw_dyn_commit_r2": metric("RawDynamics", "commit_abs_r2"),
        "boundary_residence_commit_r2": metric("BoundaryResidence", "commit_abs_r2"),
        "late_stabilization_commit_r2": metric("LateStabilization", "commit_abs_r2"),
        "velocity_curvature_commit_r2": metric("VelocityCurvature", "commit_abs_r2"),
        "order_precursor_commit_r2": metric("OrderPrecursor", "commit_abs_r2"),
        "order_residual_commit_r2": metric("OrderResidual", "commit_abs_r2"),
        "dyn4_full_commit_r2": metric("DYN4_FullDynamics", "commit_abs_r2"),
        "topk_plus_dyn4_full_commit_r2": metric("TopK_plus_DYN4_FullDynamics", "commit_abs_r2"),
        "mpr_no_answer_commit_r2": metric("MultiProjection_DYN4_NoAnswer", "commit_abs_r2"),
        "mpr_with_answer_commit_r2": metric("MultiProjection_DYN4_WithAnswer", "commit_abs_r2"),
    }

    # Component contribution table relative to TopK
    comp_groups = [
        "TopK_plus_BoundaryResidence",
        "TopK_plus_LateStabilization",
        "TopK_plus_VelocityCurvature",
        "TopK_plus_OrderPrecursor",
        "TopK_plus_OrderResidual",
        "TopK_plus_DYN4_BoundaryLate",
        "TopK_plus_DYN4_Core",
        "TopK_plus_DYN4_FullDynamics",
        "MultiProjection_DYN4_NoAnswer",
    ]

    comp_rows = []
    topk_auc = inc["topk_gen_auc"]
    topk_r2 = inc["topk_commit_r2"]
    for g in comp_groups:
        comp_rows.append({
            "feature_group": g,
            "gen_auc": metric(g, "gen_auc"),
            "gen_auc_gain_over_topk": safe_float(metric(g, "gen_auc") - topk_auc),
            "commit_abs_r2": metric(g, "commit_abs_r2"),
            "commit_r2_gain_over_topk": safe_float(metric(g, "commit_abs_r2") - topk_r2),
            "submechanism_macro_f1": metric(g, "submechanism_macro_f1"),
        })
    comp_df = pd.DataFrame(comp_rows).sort_values("gen_auc_gain_over_topk", ascending=False, na_position="last")

    # MI diagnostics
    mi_all = mutual_info_top(df, groups.get("MultiProjection_DYN4_NoAnswer", []), target="gen_error", top_n=40)
    mi_dyn4 = mutual_info_top(df, groups.get("DYN4_FullDynamics", []), target="gen_error", top_n=40)

    # Verdict
    reasons = []
    verdict = "UNDETERMINED"
    eps = 0.01

    topk_auc = inc["topk_gen_auc"]
    noans_auc = inc["mpr_no_answer_gen_auc"]
    withans_auc = inc["mpr_with_answer_gen_auc"]
    topk_dyn_auc = inc["topk_plus_dyn4_full_gen_auc"]
    noans_commit = inc["mpr_no_answer_commit_r2"]
    topk_dyn_commit = inc["topk_plus_dyn4_full_commit_r2"]

    if not np.isfinite(topk_auc):
        verdict = "FAIL_EVAL_INVALID"
        reasons.append("TopK baseline not evaluable.")
    elif np.isfinite(noans_auc) and noans_auc > topk_auc + 0.05 and np.isfinite(noans_commit) and noans_commit > 0.80:
        verdict = "PASS_DYN4B_NOANSWER_CONVERGENCE_DECOMPOSED"
        reasons.append("No-answer DYN4 MultiProjection strongly exceeds TopK and predicts commitment magnitude.")
    elif np.isfinite(topk_dyn_auc) and topk_dyn_auc > topk_auc + 0.05 and np.isfinite(topk_dyn_commit) and topk_dyn_commit > 0.70:
        verdict = "PASS_DYN4B_TOPOLOGY_PLUS_DYNAMICS"
        reasons.append("TopK + DYN4 FullDynamics strongly exceeds TopK and predicts commitment magnitude.")
    elif np.isfinite(noans_auc) and noans_auc > topk_auc + eps:
        verdict = "PARTIAL_DYN4B_NOANSWER_GEN_GAIN"
        reasons.append("No-answer dynamics improves GenError, but commitment magnitude is not strong.")
    elif np.isfinite(withans_auc) and withans_auc > topk_auc + eps:
        verdict = "PARTIAL_DYN4B_WITH_ANSWER_ONLY"
        reasons.append("Only answer-including model improves over TopK.")
    else:
        verdict = "FAIL_DYN4B_NO_COMPONENT_GAIN"
        reasons.append("No DYN4 component improves over TopK.")

    verdict_obj = {
        "verdict": verdict,
        "reasons": reasons,
        "metrics": inc,
        "feature_set_sizes": {k: len(v) for k, v in groups.items()},
        "n_rows": int(len(df)),
        "n_graphs": int(df["graph_id"].nunique()) if "graph_id" in df.columns else None,
        "input_features": str(INPUT_FEATURES),
        "outputs": {
            "component_summary": str(OUT_DIR / "dyn4b_component_summary.csv"),
            "incremental_summary": str(OUT_DIR / "dyn4b_incremental_summary.json"),
            "mutual_info_all": str(OUT_DIR / "dyn4b_mutual_info_noanswer.csv"),
        },
        "top_rows": summary.head(15).to_dict(orient="records"),
        "component_rows": comp_df.to_dict(orient="records"),
    }

    summary.to_csv(OUT_DIR / "dyn4b_model_summary.csv", index=False, encoding="utf-8-sig")
    comp_df.to_csv(OUT_DIR / "dyn4b_component_summary.csv", index=False, encoding="utf-8-sig")
    mi_all.to_csv(OUT_DIR / "dyn4b_mutual_info_noanswer.csv", index=False, encoding="utf-8-sig")
    mi_dyn4.to_csv(OUT_DIR / "dyn4b_mutual_info_dyn4.csv", index=False, encoding="utf-8-sig")
    with open(OUT_DIR / "dyn4b_incremental_summary.json", "w", encoding="utf-8") as f:
        json.dump(inc, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / "dyn4b_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict_obj, f, ensure_ascii=False, indent=2)

    print(json.dumps(verdict_obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
