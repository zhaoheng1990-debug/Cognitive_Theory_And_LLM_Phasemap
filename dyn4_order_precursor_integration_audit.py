# -*- coding: utf-8 -*-
r"""
DYN-4: Order-Precursor / Boundary-Stabilization Integration Audit

Purpose
-------
DYN-3 showed:
  - TopK + residualized dynamics improves GenError over TopK.
  - Residualized dynamics does NOT explain commitment magnitude after removing TopK + answer readout.
  - Therefore current lightweight dynamics is not the full commitment variable.

DYN-4 asks:

  Can we construct stronger convergence-dynamics proxies from existing features?

Specifically:
  1. OrderPrecursorProxy:
     PCA components of dynamics features residualized against TopK + AnswerReadout.
     This aims to capture hidden order-parameter precursor-like structure.

  2. BoundaryResidenceProxy:
     How long the trajectory stays near the decision boundary in L20-L25.

  3. LateStabilizationProxy:
     Whether the trajectory stabilizes or accelerates toward commitment in L23-L25.

  4. ResidualErrorModulation:
     Residualized dynamics components that improve GenError beyond TopK.

This script reuses:
  C:\Users\ZH\Desktop\AGI\outputs\dyn3_outputs\dyn3_features_augmented.csv

Outputs:
  C:\Users\ZH\Desktop\AGI\outputs\dyn4_outputs\dyn4_features_augmented.csv
  C:\Users\ZH\Desktop\AGI\outputs\dyn4_model_summary.csv
  C:\Users\ZH\Desktop\AGI\outputs\dyn4_incremental_summary.json
  C:\Users\ZH\Desktop\AGI\outputs\dyn4_same_answer_summary.json
  C:\Users\ZH\Desktop\AGI\outputs\dyn4_verdict.json
"""

import json
import math
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
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_SEED = 42
N_SPLITS = 5

INPUT_FEATURES = Path(r"C:\Users\ZH\Desktop\AGI\outputs\dyn3_outputs\dyn3_features_augmented.csv")
OUT_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\dyn4_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Utilities
# -----------------------------

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


def area_unit(xs):
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    if arr.size == 1:
        return float(arr[0])
    return float(np.sum((arr[:-1] + arr[1:]) * 0.5))


def slope_unit(xs):
    arr = np.asarray(xs, dtype=np.float64)
    if arr.size < 2 or np.any(~np.isfinite(arr)):
        return np.nan
    x = np.arange(arr.size, dtype=np.float64)
    try:
        return float(np.polyfit(x, arr, 1)[0])
    except Exception:
        return np.nan


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
    for train_idx, test_idx in cv_indices(d, y, group_col=group_col, stratify=False):
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


def crossfit_residualize(df, cols, against_cols, group_col="graph_id", suffix="__resid"):
    cols = numeric_cols(df, cols)
    against_cols = numeric_cols(df, against_cols)
    out = pd.DataFrame(index=df.index)
    if not cols or not against_cols:
        return out

    A = df[against_cols].replace([np.inf, -np.inf], np.nan).values
    splits = cv_indices(df, None, group_col=group_col)

    for c in cols:
        y = df[c].astype(float).replace([np.inf, -np.inf], np.nan).values
        res = np.full(len(df), np.nan, dtype=float)
        for train_idx, test_idx in splits:
            valid_train = np.isfinite(y[train_idx])
            if valid_train.sum() < 5:
                continue
            reg = make_regressor()
            reg.fit(A[train_idx][valid_train], y[train_idx][valid_train])
            pred = reg.predict(A[test_idx])
            res[test_idx] = y[test_idx] - pred
        out[c + suffix] = res
    return out


def add_layer_margin_features(df):
    # Try to locate ans_margin_L20...L25.
    margin_cols = [f"ans_margin_L{i}" for i in range(20, 26) if f"ans_margin_L{i}" in df.columns]
    if len(margin_cols) >= 2:
        arr = df[margin_cols].astype(float).replace([np.inf, -np.inf], np.nan).values
        abs_arr = np.abs(arr)

        df["dyn4_boundary_residence_q25"] = np.nan
        df["dyn4_boundary_residence_q50"] = np.nan

        # global thresholds on abs margin
        flat = abs_arr[np.isfinite(abs_arr)]
        if flat.size:
            q25 = np.nanquantile(flat, 0.25)
            q50 = np.nanquantile(flat, 0.50)
            df["dyn4_boundary_residence_q25"] = np.nanmean(abs_arr <= q25, axis=1)
            df["dyn4_boundary_residence_q50"] = np.nanmean(abs_arr <= q50, axis=1)

        df["dyn4_abs_margin_area"] = [area_unit(row) for row in abs_arr]
        df["dyn4_abs_margin_slope"] = [slope_unit(row) for row in abs_arr]
        df["dyn4_margin_area"] = [area_unit(row) for row in arr]
        df["dyn4_margin_slope"] = [slope_unit(row) for row in arr]

        # early vs late absolute stabilization
        if len(margin_cols) >= 6:
            early = np.nanmean(abs_arr[:, :3], axis=1)
            late = np.nanmean(abs_arr[:, 3:], axis=1)
            df["dyn4_late_minus_early_abs"] = late - early
            df["dyn4_late_over_early_abs"] = late / (early + 1e-6)
            df["dyn4_final_proxy_abs_L25"] = abs_arr[:, -1]
            df["dyn4_start_abs_L20"] = abs_arr[:, 0]

        # sign changes
        signs = np.sign(arr)
        sign_changes = []
        for row in signs:
            r = row[np.isfinite(row)]
            if r.size < 2:
                sign_changes.append(np.nan)
            else:
                sign_changes.append(float(np.sum(r[:-1] * r[1:] < 0)))
        df["dyn4_margin_sign_changes"] = sign_changes

    return df


def add_pca_order_precursor(df, dyn_cols, against_cols, n_comp=5):
    dyn_cols = numeric_cols(df, dyn_cols)
    against_cols = numeric_cols(df, against_cols)

    # Residualize dynamics against TopK + AnswerReadout first.
    resid_df = crossfit_residualize(df, dyn_cols, against_cols, group_col="graph_id", suffix="__opresid")
    for c in resid_df.columns:
        df[c] = resid_df[c]

    resid_cols = list(resid_df.columns)
    if not resid_cols:
        return df, []

    X = df[resid_cols].replace([np.inf, -np.inf], np.nan)
    # Fill median before PCA.
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0).values

    # Standardize.
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    n_comp_eff = min(n_comp, Xs.shape[1], Xs.shape[0])
    pca = PCA(n_components=n_comp_eff, random_state=RANDOM_SEED)
    Z = pca.fit_transform(Xs)

    pc_cols = []
    for i in range(n_comp_eff):
        name = f"order_precursor_pc{i+1}"
        df[name] = Z[:, i]
        pc_cols.append(name)

    # Add simple norms/energy of precursor coordinates.
    if pc_cols:
        mat = df[pc_cols].values
        df["order_precursor_norm"] = np.linalg.norm(mat, axis=1)
        df["order_precursor_pc1_abs"] = np.abs(df[pc_cols[0]].values)
        pc_cols += ["order_precursor_norm", "order_precursor_pc1_abs"]

    return df, pc_cols + resid_cols


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

    dyn = cols_by_prefix(df, ["dyn_", "dyn2_"])
    # Avoid including residual cols created later by name collision.
    dyn = [c for c in dyn if "__" not in c]
    dyn4 = cols_by_prefix(df, ["dyn4_"])
    order = cols_by_prefix(df, ["order_precursor_"])
    opresid = cols_containing(df, ["__opresid"])

    non_topk = unique_keep_order(dyn4 + order + opresid)
    dyn4_core = unique_keep_order(dyn4 + order)
    full_dynamics = unique_keep_order(dyn + dyn4 + order + opresid)

    groups = {
        "TopK": topk,
        "AnswerReadout": answer,
        "RawDynamics": dyn,
        "DYN4_BoundaryLate": dyn4,
        "OrderPrecursor": order,
        "OrderResidual": opresid,
        "DYN4_Core": dyn4_core,
        "DYN4_FullDynamics": full_dynamics,
        "TopK_plus_DYN4_Core": unique_keep_order(topk + dyn4_core),
        "TopK_plus_OrderPrecursor": unique_keep_order(topk + order),
        "TopK_plus_OrderResidual": unique_keep_order(topk + opresid),
        "TopK_plus_DYN4_FullDynamics": unique_keep_order(topk + full_dynamics),
        "MultiProjection_DYN4_NoAnswer": unique_keep_order(topk + non_topk),
        "MultiProjection_DYN4_WithAnswer": unique_keep_order(topk + non_topk + answer),
    }
    return {k: numeric_cols(df, v) for k, v in groups.items() if len(numeric_cols(df, v)) > 0}


def same_answer_split(df, groups, target="submechanism"):
    if "answer_pred" not in df.columns or target not in df.columns:
        return {"valid": False, "reason": "missing_answer_pred_or_target"}

    details = []
    for answer, sub in df.groupby("answer_pred"):
        if len(sub) < 80 or sub[target].nunique() < 2:
            continue
        for name, cols in groups.items():
            res = eval_multiclass(sub, cols, target=target, group_col="graph_id")
            if res.get("valid"):
                details.append({
                    "answer": answer,
                    "feature_group": name,
                    "target": target,
                    "n": int(len(sub)),
                    "macro_f1": res.get("macro_f1", np.nan),
                    "bal_acc": res.get("bal_acc", np.nan),
                    "acc": res.get("acc", np.nan),
                })

    det = pd.DataFrame(details)
    if det.empty:
        return {"valid": False, "reason": "no_valid_same_answer_groups"}

    agg = (
        det.groupby("feature_group")
        .agg(
            same_answer_macro_f1_mean=("macro_f1", "mean"),
            same_answer_bal_acc_mean=("bal_acc", "mean"),
            same_answer_acc_mean=("acc", "mean"),
            n_answers=("answer", "nunique"),
        )
        .reset_index()
        .sort_values("same_answer_macro_f1_mean", ascending=False)
    )
    return {
        "valid": True,
        "aggregate": agg.to_dict(orient="records"),
        "details": det.to_dict(orient="records"),
    }


def main():
    if not INPUT_FEATURES.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_FEATURES}")

    df = pd.read_csv(INPUT_FEATURES)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Build initial groups to know TopK/Answer/Dynamics.
    temp_groups = {
        "topk": cols_by_prefix(df, ["topk_"]),
        "answer": numeric_cols(df, [
            "ans_margin_L20", "ans_margin_L21", "ans_margin_L22",
            "ans_margin_L23", "ans_margin_L24", "ans_margin_L25",
            "ans_margin_main_mean", "ans_margin_main_slope", "ans_margin_main_area",
            "ans_margin_main_std", "ans_margin_main_minabs", "ans_margin_first_flip",
            "final_margin_C_minus_E",
            "commit_abs_final_margin", "commit_high_abs_margin",
        ]),
        "dyn": [c for c in cols_by_prefix(df, ["dyn_", "dyn2_"]) if "__" not in c],
    }

    # Add DYN-4 engineered features.
    df = add_layer_margin_features(df)

    # Add residual PCA order precursor.
    df, order_cols = add_pca_order_precursor(
        df,
        dyn_cols=temp_groups["dyn"],
        against_cols=temp_groups["topk"] + temp_groups["answer"],
        n_comp=5,
    )

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

    model_summary = pd.DataFrame(rows).sort_values(
        ["gen_auc", "commit_abs_r2", "submechanism_macro_f1"],
        ascending=False,
        na_position="last",
    )

    same_ans = same_answer_split(df, groups, target="submechanism")

    def get_metric(group, metric):
        r = model_summary[model_summary["feature_group"] == group]
        if len(r) == 0:
            return np.nan
        return safe_float(r.iloc[0].get(metric, np.nan))

    metrics = {
        "topk_gen_auc": get_metric("TopK", "gen_auc"),
        "raw_dynamics_gen_auc": get_metric("RawDynamics", "gen_auc"),
        "dyn4_core_gen_auc": get_metric("DYN4_Core", "gen_auc"),
        "order_precursor_gen_auc": get_metric("OrderPrecursor", "gen_auc"),
        "order_residual_gen_auc": get_metric("OrderResidual", "gen_auc"),
        "topk_plus_order_gen_auc": get_metric("TopK_plus_OrderPrecursor", "gen_auc"),
        "topk_plus_order_residual_gen_auc": get_metric("TopK_plus_OrderResidual", "gen_auc"),
        "topk_plus_dyn4_full_gen_auc": get_metric("TopK_plus_DYN4_FullDynamics", "gen_auc"),
        "mpr_no_answer_gen_auc": get_metric("MultiProjection_DYN4_NoAnswer", "gen_auc"),
        "mpr_with_answer_gen_auc": get_metric("MultiProjection_DYN4_WithAnswer", "gen_auc"),
        "answer_gen_auc": get_metric("AnswerReadout", "gen_auc"),

        "topk_commit_r2": get_metric("TopK", "commit_abs_r2"),
        "raw_dynamics_commit_r2": get_metric("RawDynamics", "commit_abs_r2"),
        "dyn4_core_commit_r2": get_metric("DYN4_Core", "commit_abs_r2"),
        "order_precursor_commit_r2": get_metric("OrderPrecursor", "commit_abs_r2"),
        "order_residual_commit_r2": get_metric("OrderResidual", "commit_abs_r2"),
        "mpr_no_answer_commit_r2": get_metric("MultiProjection_DYN4_NoAnswer", "commit_abs_r2"),
        "mpr_with_answer_commit_r2": get_metric("MultiProjection_DYN4_WithAnswer", "commit_abs_r2"),
        "answer_commit_r2": get_metric("AnswerReadout", "commit_abs_r2"),
    }

    eps = 0.01
    reasons = []
    verdict = "UNDETERMINED"

    topk_auc = metrics["topk_gen_auc"]
    order_auc = metrics["order_precursor_gen_auc"]
    order_res_auc = metrics["order_residual_gen_auc"]
    topk_order_res_auc = metrics["topk_plus_order_residual_gen_auc"]
    mpr_no_ans_auc = metrics["mpr_no_answer_gen_auc"]
    mpr_ans_auc = metrics["mpr_with_answer_gen_auc"]
    dyn4_commit = metrics["dyn4_core_commit_r2"]
    order_commit = metrics["order_precursor_commit_r2"]
    order_res_commit = metrics["order_residual_commit_r2"]
    answer_commit = metrics["answer_commit_r2"]

    if not np.isfinite(topk_auc):
        verdict = "FAIL_EVAL_INVALID"
        reasons.append("TopK baseline could not be evaluated.")
    else:
        order_independent = np.isfinite(order_res_auc) and order_res_auc > topk_auc + eps
        topk_order_gain = np.isfinite(topk_order_res_auc) and topk_order_res_auc > topk_auc + eps
        mpr_no_answer_gain = np.isfinite(mpr_no_ans_auc) and mpr_no_ans_auc > topk_auc + eps
        order_commit_meaningful = np.isfinite(order_commit) and order_commit > 0.30
        order_res_commit_meaningful = np.isfinite(order_res_commit) and order_res_commit > 0.10

        if order_independent and order_res_commit_meaningful:
            verdict = "PASS_ORDER_RESIDUAL_DYNAMICS_INDEPENDENT"
            reasons.append("Order residual dynamics independently exceeds TopK and predicts commitment.")
        elif topk_order_gain and order_commit_meaningful:
            verdict = "PASS_TOPOLOGY_PLUS_ORDER_PRECURSOR"
            reasons.append("TopK plus order-precursor dynamics improves over TopK, and order precursor predicts commitment.")
        elif mpr_no_answer_gain and (order_commit_meaningful or np.isfinite(dyn4_commit) and dyn4_commit > 0.30):
            verdict = "PASS_MPR_NOANSWER_DYNAMICS_GAIN"
            reasons.append("No-answer MultiProjection improves over TopK with meaningful dynamics/commitment signal.")
        elif mpr_no_answer_gain:
            verdict = "PARTIAL_MPR_NOANSWER_GEN_GAIN_DYNAMICS_WEAK"
            reasons.append("No-answer MultiProjection improves GenError, but dynamics commitment signal remains weak.")
        elif np.isfinite(mpr_ans_auc) and mpr_ans_auc > topk_auc + eps:
            verdict = "PARTIAL_MPR_WITH_ANSWER_ONLY"
            reasons.append("Only answer-including MultiProjection improves over TopK.")
        else:
            verdict = "FAIL_NO_DYN4_GAIN"
            reasons.append("DYN-4 proxies do not improve over TopK.")

    verdict_obj = {
        "verdict": verdict,
        "reasons": reasons,
        "metrics": metrics,
        "feature_set_sizes": {k: len(v) for k, v in groups.items()},
        "n_rows": int(len(df)),
        "n_graphs": int(df["graph_id"].nunique()) if "graph_id" in df.columns else None,
        "input_features": str(INPUT_FEATURES),
        "outputs": {
            "model_summary": str(OUT_DIR / "dyn4_model_summary.csv"),
            "same_answer_summary": str(OUT_DIR / "dyn4_same_answer_summary.json"),
            "features_augmented": str(OUT_DIR / "dyn4_features_augmented.csv"),
        },
        "top_rows": model_summary.head(15).to_dict(orient="records"),
    }

    df.to_csv(OUT_DIR / "dyn4_features_augmented.csv", index=False, encoding="utf-8-sig")
    model_summary.to_csv(OUT_DIR / "dyn4_model_summary.csv", index=False, encoding="utf-8-sig")
    with open(OUT_DIR / "dyn4_same_answer_summary.json", "w", encoding="utf-8") as f:
        json.dump(same_ans, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / "dyn4_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict_obj, f, ensure_ascii=False, indent=2)

    print(json.dumps(verdict_obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
