# -*- coding: utf-8 -*-
"""
DYN-3: Residual Convergence Dynamics Audit

Purpose
-------
DYN-2 showed:
  - MultiProjection improves GenError / commitment prediction.
  - Lightweight Dynamics-only is weaker than TopK for GenError.
  - Dynamics predicts commitment magnitude to a meaningful degree, but is not yet the full commitment variable.

DYN-3 asks a sharper question:

    Does a dynamics projection retain independent information after removing
    TopK topology and answer-margin/commitment readout?

This is NOT a model-forward script. It reuses:

    r"C:\Users\ZH\Desktop\AGI\outputs\dyn2_outputs\dyn2_features_augmented.csv"

and produces:

    r"C:\Users\ZH\Desktop\AGI\outputs\dyn3_outputs\dyn3_model_summary.csv"
    r"C:\Users\ZH\Desktop\AGI\outputs\dyn3_outputs\dyn3_incremental_summary.csv
    C:\Users\ZH\Desktop\AGI\outputs\dyn3_outputs\dyn3_verdict.json

Interpretation
--------------
If residual dynamics improves GenError / commitment beyond TopK and answer readout,
then D_dyn has an independent projection.

If only TopK+Dynamics works, but residualized dynamics is weak, current dynamics proxy
is still mostly entangled with topology / answer readout and needs true O_cont / Hshape / order precursor extraction.
"""

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    r2_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

INPUT_FEATURES = Path(r"C:\Users\ZH\Desktop\AGI\outputs\dyn2_outputs\dyn2_features_augmented.csv")
OUT_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\dyn3_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_SPLITS = 5


# -----------------------------
# Utility
# -----------------------------

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def numeric_cols(df, cols):
    out = []
    for c in cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def cols_by_prefix(df, prefixes):
    cols = []
    for c in df.columns:
        for p in prefixes:
            if c.startswith(p):
                cols.append(c)
                break
    return numeric_cols(df, cols)


def build_feature_groups(df):
    topk = cols_by_prefix(df, ["topk_"])

    # Answer margin/readout features. These are deliberately separated as readout controls.
    answer_margin = numeric_cols(df, [
        "ans_margin_L20", "ans_margin_L21", "ans_margin_L22",
        "ans_margin_L23", "ans_margin_L24", "ans_margin_L25",
        "ans_margin_main_mean", "ans_margin_main_slope", "ans_margin_main_area",
        "ans_margin_main_std", "ans_margin_main_minabs", "ans_margin_first_flip",
        "final_margin_C_minus_E",
        "commit_abs_final_margin", "commit_high_abs_margin",
    ])

    # Lightweight dynamics proxy from previous scripts.
    dyn_base = cols_by_prefix(df, ["dyn_"])
    dyn2 = cols_by_prefix(df, ["dyn2_"])
    commit_abs_layers = cols_by_prefix(df, ["commit_abs_ans_margin_"])
    dynamics = list(dict.fromkeys(dyn_base + dyn2 + commit_abs_layers))

    # No-final dynamics excludes final readout and raw final commitment if present.
    dynamics_no_final = [
        c for c in dynamics
        if c not in {"commit_abs_final_margin", "commit_high_abs_margin", "final_margin_C_minus_E"}
        and not c.startswith("final_")
    ]

    # Boundary / commitment dynamics subset.
    boundary_dyn = numeric_cols(df, [
        "dyn_boundary_minabs",
        "dyn2_min_abs_20_25",
        "dyn2_mean_abs_20_25",
        "dyn2_std_abs_20_25",
        "dyn2_start_abs_L20",
        "dyn2_final_proxy_abs_L25",
        "dyn2_commitment_velocity_L20_to_L25",
        "dyn2_commitment_gain_late_minus_early",
    ])

    velocity_dyn = numeric_cols(df, [
        "dyn_dmargin_mean",
        "dyn_dmargin_area",
        "dyn_dmargin_std",
        "dyn_abs_dmargin_mean",
        "dyn_second_abs_mean",
        "dyn_curvature_proxy",
        "dyn2_velocity_mean",
        "dyn2_velocity_abs_mean",
        "dyn2_velocity_std",
        "dyn2_accel_abs_mean",
        "dyn2_accel_std",
        "dyn2_sign_changes",
    ])

    topk_plus_dyn = list(dict.fromkeys(topk + dynamics_no_final))
    multiproj = list(dict.fromkeys(topk + dynamics_no_final + answer_margin))

    groups = {
        "TopK": topk,
        "AnswerReadout": answer_margin,
        "DynamicsNoFinal": dynamics_no_final,
        "BoundaryDynamics": boundary_dyn,
        "VelocityCurvatureDynamics": velocity_dyn,
        "TopK_plus_Dynamics": topk_plus_dyn,
        "MultiProjection": multiproj,
    }

    return {k: v for k, v in groups.items() if len(v) > 0}


def make_classifier():
    # OneVsRest keeps behavior stable across sklearn versions and supports binary/multiclass.
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


def make_binary_classifier():
    clf = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_SEED,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


def make_regressor():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=1.0)),
    ])


def cv_indices(df, y, group_col=None):
    groups = df[group_col].values if group_col and group_col in df.columns else None
    n = len(df)
    if groups is not None and len(np.unique(groups)) >= N_SPLITS:
        cv = GroupKFold(n_splits=N_SPLITS)
        return list(cv.split(np.zeros(n), y, groups=groups))
    # fallback stratified for classification
    vals, counts = np.unique(y, return_counts=True)
    if len(vals) >= 2 and counts.min() >= N_SPLITS:
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        return list(cv.split(np.zeros(n), y))
    # fallback simple KFold-like split
    idx = np.arange(n)
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(idx)
    folds = np.array_split(idx, min(N_SPLITS, n))
    splits = []
    for test_idx in folds:
        train_idx = np.setdiff1d(idx, test_idx)
        splits.append((train_idx, test_idx))
    return splits


def eval_binary(df, feature_cols, target="gen_error", group_col="graph_id"):
    feature_cols = numeric_cols(df, feature_cols)
    if len(feature_cols) == 0 or target not in df.columns:
        return {"valid": False, "reason": "missing_features_or_target"}

    d = df.dropna(subset=[target]).copy()
    y = d[target].astype(int).values
    if len(np.unique(y)) < 2:
        return {"valid": False, "reason": "single_class_target", "n": len(d)}

    X = d[feature_cols].replace([np.inf, -np.inf], np.nan).values
    preds = np.zeros(len(d), dtype=float)
    pred_labels = np.zeros(len(d), dtype=int)

    used = 0
    for train_idx, test_idx in cv_indices(d, y, group_col=group_col):
        if len(np.unique(y[train_idx])) < 2:
            continue
        clf = make_binary_classifier()
        clf.fit(X[train_idx], y[train_idx])
        if hasattr(clf[-1], "predict_proba"):
            prob = clf.predict_proba(X[test_idx])[:, 1]
        else:
            prob = clf.decision_function(X[test_idx])
        preds[test_idx] = prob
        pred_labels[test_idx] = (prob >= 0.5).astype(int)
        used += 1

    if used == 0:
        return {"valid": False, "reason": "no_valid_folds", "n": len(d)}

    try:
        auc = roc_auc_score(y, preds)
    except Exception:
        auc = np.nan

    return {
        "valid": True,
        "n": int(len(d)),
        "n_features": int(len(feature_cols)),
        "auc": safe_float(auc),
        "acc": safe_float(accuracy_score(y, pred_labels)),
        "f1": safe_float(f1_score(y, pred_labels, zero_division=0)),
        "bal_acc": safe_float(balanced_accuracy_score(y, pred_labels)),
    }


def eval_multiclass(df, feature_cols, target="mechanism", group_col="graph_id"):
    feature_cols = numeric_cols(df, feature_cols)
    if len(feature_cols) == 0 or target not in df.columns:
        return {"valid": False, "reason": "missing_features_or_target"}

    d = df.dropna(subset=[target]).copy()
    y = d[target].astype(str).values
    if len(np.unique(y)) < 2:
        return {"valid": False, "reason": "single_class_target", "n": len(d)}

    X = d[feature_cols].replace([np.inf, -np.inf], np.nan).values
    pred_labels = np.array([""] * len(d), dtype=object)

    used = 0
    for train_idx, test_idx in cv_indices(d, y, group_col=group_col):
        if len(np.unique(y[train_idx])) < 2:
            continue
        clf = make_classifier()
        clf.fit(X[train_idx], y[train_idx])
        pred_labels[test_idx] = clf.predict(X[test_idx])
        used += 1

    if used == 0:
        return {"valid": False, "reason": "no_valid_folds", "n": len(d)}

    return {
        "valid": True,
        "n": int(len(d)),
        "n_features": int(len(feature_cols)),
        "n_classes": int(len(np.unique(y))),
        "acc": safe_float(accuracy_score(y, pred_labels)),
        "macro_f1": safe_float(f1_score(y, pred_labels, average="macro", zero_division=0)),
        "weighted_f1": safe_float(f1_score(y, pred_labels, average="weighted", zero_division=0)),
        "bal_acc": safe_float(balanced_accuracy_score(y, pred_labels)),
    }


def eval_regression(df, feature_cols, target="commit_abs_final_margin", group_col="graph_id"):
    feature_cols = numeric_cols(df, feature_cols)
    if len(feature_cols) == 0 or target not in df.columns:
        return {"valid": False, "reason": "missing_features_or_target"}

    d = df.dropna(subset=[target]).copy()
    y = d[target].astype(float).values
    if len(d) < 10:
        return {"valid": False, "reason": "too_few_rows", "n": len(d)}

    X = d[feature_cols].replace([np.inf, -np.inf], np.nan).values
    preds = np.zeros(len(d), dtype=float)

    used = 0
    for train_idx, test_idx in cv_indices(d, np.zeros(len(d)), group_col=group_col):
        reg = make_regressor()
        reg.fit(X[train_idx], y[train_idx])
        preds[test_idx] = reg.predict(X[test_idx])
        used += 1

    if used == 0:
        return {"valid": False, "reason": "no_valid_folds", "n": len(d)}

    try:
        corr = np.corrcoef(y, preds)[0, 1]
    except Exception:
        corr = np.nan

    return {
        "valid": True,
        "n": int(len(d)),
        "n_features": int(len(feature_cols)),
        "r2": safe_float(r2_score(y, preds)),
        "corr": safe_float(corr),
    }


def residualize_against(df, cols, against_cols, group_col="graph_id"):
    """Cross-fitted residualization: for each feature in cols, regress it on against_cols in train fold."""
    cols = numeric_cols(df, cols)
    against_cols = numeric_cols(df, against_cols)
    if not cols or not against_cols:
        return pd.DataFrame(index=df.index)

    d = df.copy()
    X_against = d[against_cols].replace([np.inf, -np.inf], np.nan).values
    out = pd.DataFrame(index=df.index)

    # initialize with NaN
    for c in cols:
        out[c + "__resid"] = np.nan

    y_dummy = np.zeros(len(d))
    splits = cv_indices(d, y_dummy, group_col=group_col)

    for c in cols:
        y = d[c].astype(float).replace([np.inf, -np.inf], np.nan).values
        for train_idx, test_idx in splits:
            valid_train = ~np.isnan(y[train_idx])
            if valid_train.sum() < 5:
                continue
            reg = make_regressor()
            reg.fit(X_against[train_idx][valid_train], y[train_idx][valid_train])
            pred = reg.predict(X_against[test_idx])
            out.iloc[test_idx, out.columns.get_loc(c + "__resid")] = y[test_idx] - pred

    return out


def mutual_info_table(df, feature_cols, target="gen_error", top_n=25):
    feature_cols = numeric_cols(df, feature_cols)
    if target not in df.columns or len(feature_cols) == 0:
        return pd.DataFrame()

    d = df.dropna(subset=[target]).copy()
    X = d[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    y = d[target].astype(int).values
    if len(np.unique(y)) < 2:
        return pd.DataFrame()

    try:
        mi = mutual_info_classif(X.values, y, random_state=RANDOM_SEED)
    except Exception:
        return pd.DataFrame()

    tab = pd.DataFrame({"feature": feature_cols, "mutual_info": mi})
    return tab.sort_values("mutual_info", ascending=False).head(top_n)


def main():
    if not INPUT_FEATURES.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_FEATURES}")

    df = pd.read_csv(INPUT_FEATURES)
    df = df.replace([np.inf, -np.inf], np.nan)

    groups = build_feature_groups(df)

    # Core residualized dynamics groups
    topk = groups.get("TopK", [])
    answer = groups.get("AnswerReadout", [])
    dyn = groups.get("DynamicsNoFinal", [])
    boundary_dyn = groups.get("BoundaryDynamics", [])
    vel_dyn = groups.get("VelocityCurvatureDynamics", [])

    resid_dyn_vs_topk_answer = residualize_against(df, dyn, topk + answer, group_col="graph_id")
    resid_boundary_vs_topk_answer = residualize_against(df, boundary_dyn, topk + answer, group_col="graph_id")
    resid_vel_vs_topk_answer = residualize_against(df, vel_dyn, topk + answer, group_col="graph_id")

    for new_df in [resid_dyn_vs_topk_answer, resid_boundary_vs_topk_answer, resid_vel_vs_topk_answer]:
        for c in new_df.columns:
            df[c] = new_df[c]

    groups["ResidualDynamics_vs_TopKAnswer"] = list(resid_dyn_vs_topk_answer.columns)
    groups["ResidualBoundaryDynamics_vs_TopKAnswer"] = list(resid_boundary_vs_topk_answer.columns)
    groups["ResidualVelocityDynamics_vs_TopKAnswer"] = list(resid_vel_vs_topk_answer.columns)

    groups["TopK_plus_ResidualDynamics"] = list(dict.fromkeys(topk + groups["ResidualDynamics_vs_TopKAnswer"]))
    groups["TopK_plus_BoundaryResidual"] = list(dict.fromkeys(topk + groups["ResidualBoundaryDynamics_vs_TopKAnswer"]))

    # Evaluate model matrix
    rows = []
    for name, cols in groups.items():
        gen = eval_binary(df, cols, target="gen_error", group_col="graph_id")
        mech = eval_multiclass(df, cols, target="mechanism", group_col="graph_id")
        subm = eval_multiclass(df, cols, target="submechanism", group_col="graph_id")
        reg = eval_regression(df, cols, target="commit_abs_final_margin", group_col="graph_id")

        row = {
            "feature_group": name,
            "n_features": len(numeric_cols(df, cols)),
            "gen_auc": gen.get("auc", np.nan),
            "gen_f1": gen.get("f1", np.nan),
            "gen_acc": gen.get("acc", np.nan),
            "mechanism_macro_f1": mech.get("macro_f1", np.nan),
            "submechanism_macro_f1": subm.get("macro_f1", np.nan),
            "commit_abs_r2": reg.get("r2", np.nan),
            "commit_abs_corr": reg.get("corr", np.nan),
            "valid_gen": gen.get("valid", False),
            "valid_mech": mech.get("valid", False),
            "valid_submech": subm.get("valid", False),
            "valid_commit": reg.get("valid", False),
        }
        rows.append(row)

    model_summary = pd.DataFrame(rows).sort_values(
        ["gen_auc", "commit_abs_r2", "submechanism_macro_f1"],
        ascending=False,
        na_position="last",
    )

    # Incremental summary
    def get_row(name):
        r = model_summary[model_summary["feature_group"] == name]
        return r.iloc[0].to_dict() if len(r) else {}

    topk_row = get_row("TopK")
    dyn_row = get_row("DynamicsNoFinal")
    ans_row = get_row("AnswerReadout")
    multi_row = get_row("MultiProjection")
    residual_dyn_row = get_row("ResidualDynamics_vs_TopKAnswer")
    topk_resid_row = get_row("TopK_plus_ResidualDynamics")
    boundary_resid_row = get_row("ResidualBoundaryDynamics_vs_TopKAnswer")

    inc = {
        "topk_gen_auc": topk_row.get("gen_auc", np.nan),
        "dynamics_gen_auc": dyn_row.get("gen_auc", np.nan),
        "answer_gen_auc": ans_row.get("gen_auc", np.nan),
        "multiprojection_gen_auc": multi_row.get("gen_auc", np.nan),
        "residual_dynamics_gen_auc": residual_dyn_row.get("gen_auc", np.nan),
        "topk_plus_residual_dynamics_gen_auc": topk_resid_row.get("gen_auc", np.nan),
        "residual_boundary_gen_auc": boundary_resid_row.get("gen_auc", np.nan),
        "topk_commit_r2": topk_row.get("commit_abs_r2", np.nan),
        "dynamics_commit_r2": dyn_row.get("commit_abs_r2", np.nan),
        "answer_commit_r2": ans_row.get("commit_abs_r2", np.nan),
        "multiprojection_commit_r2": multi_row.get("commit_abs_r2", np.nan),
        "residual_dynamics_commit_r2": residual_dyn_row.get("commit_abs_r2", np.nan),
        "topk_plus_residual_dynamics_commit_r2": topk_resid_row.get("commit_abs_r2", np.nan),
    }

    # Top mutual information features for diagnostic
    mi_all = mutual_info_table(df, groups.get("MultiProjection", []), target="gen_error", top_n=30)
    mi_dyn = mutual_info_table(df, dyn, target="gen_error", top_n=30)
    mi_resid = mutual_info_table(df, groups.get("ResidualDynamics_vs_TopKAnswer", []), target="gen_error", top_n=30)

    # Verdict
    eps = 0.01
    topk_auc = inc.get("topk_gen_auc", np.nan)
    dyn_auc = inc.get("dynamics_gen_auc", np.nan)
    resid_auc = inc.get("residual_dynamics_gen_auc", np.nan)
    multi_auc = inc.get("multiprojection_gen_auc", np.nan)
    topk_resid_auc = inc.get("topk_plus_residual_dynamics_gen_auc", np.nan)
    dyn_commit_r2 = inc.get("dynamics_commit_r2", np.nan)
    resid_commit_r2 = inc.get("residual_dynamics_commit_r2", np.nan)
    multi_commit_r2 = inc.get("multiprojection_commit_r2", np.nan)

    reasons = []
    verdict = "UNDETERMINED"

    if not np.isfinite(topk_auc) or not np.isfinite(dyn_auc):
        verdict = "FAIL_EVAL_INVALID"
        reasons.append("Could not evaluate TopK or Dynamics.")
    else:
        residual_independent = np.isfinite(resid_auc) and resid_auc > max(0.5, topk_auc + eps)
        topk_resid_improves = np.isfinite(topk_resid_auc) and topk_resid_auc > topk_auc + eps
        multi_improves = np.isfinite(multi_auc) and multi_auc > topk_auc + eps
        dyn_commit_meaningful = np.isfinite(dyn_commit_r2) and dyn_commit_r2 > 0.3
        resid_commit_meaningful = np.isfinite(resid_commit_r2) and resid_commit_r2 > 0.1

        if residual_independent and resid_commit_meaningful:
            verdict = "PASS_RESIDUAL_DYNAMICS_INDEPENDENT"
            reasons.append("Residual dynamics remains predictive after removing TopK and answer readout.")
        elif topk_resid_improves and dyn_commit_meaningful:
            verdict = "PASS_TOPOLOGY_PLUS_RESIDUAL_DYNAMICS"
            reasons.append("TopK plus residualized dynamics improves over TopK, and dynamics predicts commitment magnitude.")
        elif multi_improves and dyn_commit_meaningful:
            verdict = "PARTIAL_MPR_COMMITMENT_DYNAMICS_WEAK"
            reasons.append("MultiProjection improves over TopK, and raw dynamics predicts commitment magnitude, but residual dynamics is weak.")
        elif multi_improves:
            verdict = "PARTIAL_MPR_GAIN_ONLY"
            reasons.append("MultiProjection improves over TopK, but dynamics commitment signal is weak.")
        else:
            verdict = "FAIL_DYNAMICS_NO_INDEPENDENT_GAIN"
            reasons.append("Dynamics does not show independent GenError or commitment gain beyond TopK / answer readout.")

    verdict_obj = {
        "verdict": verdict,
        "reasons": reasons,
        "metrics": inc,
        "feature_set_sizes": {k: len(v) for k, v in groups.items()},
        "n_rows": int(len(df)),
        "n_graphs": int(df["graph_id"].nunique()) if "graph_id" in df.columns else None,
        "input_features": str(INPUT_FEATURES),
        "outputs": {
            "model_summary": str(OUT_DIR / "dyn3_model_summary.csv"),
            "incremental_summary": str(OUT_DIR / "dyn3_incremental_summary.json"),
            "mutual_info_all": str(OUT_DIR / "dyn3_mutual_info_all.csv"),
            "features_augmented": str(OUT_DIR / "dyn3_features_augmented.csv"),
        },
        "top_rows": model_summary.head(12).to_dict(orient="records"),
    }

    df.to_csv(OUT_DIR / "dyn3_features_augmented.csv", index=False, encoding="utf-8-sig")
    model_summary.to_csv(OUT_DIR / "dyn3_model_summary.csv", index=False, encoding="utf-8-sig")
    with open(OUT_DIR / "dyn3_incremental_summary.json", "w", encoding="utf-8") as f:
        json.dump(inc, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / "dyn3_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict_obj, f, ensure_ascii=False, indent=2)

    mi_all.to_csv(OUT_DIR / "dyn3_mutual_info_all.csv", index=False, encoding="utf-8-sig")
    mi_dyn.to_csv(OUT_DIR / "dyn3_mutual_info_dynamics.csv", index=False, encoding="utf-8-sig")
    mi_resid.to_csv(OUT_DIR / "dyn3_mutual_info_residual_dynamics.csv", index=False, encoding="utf-8-sig")

    print(json.dumps(verdict_obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
