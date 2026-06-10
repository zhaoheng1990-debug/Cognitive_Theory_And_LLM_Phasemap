# -*- coding: utf-8 -*-
r"""
CFT-4 / PSA-2: Constraint Field -> Effective Cbit Audit

Purpose
-------
CFT-3C established cross-model support for a no-answer constraint-field proxy:

    F_c_hat = TopK_aligned + Dynamics_aligned

PSA-1 established that possibility space and Cbit gain are measurable at least
in an answer-candidate projection:

    H_c(l), N_eff(l)=2^H_c(l), ΔCbit = H_before - H_after

CFT-4 / PSA-2 connects the two:

    Does no-answer constraint-field quality predict effective Cbit compression?

Main hypothesis
---------------
Raw Cbit gain is not enough. Wrong closure trajectories may also compress quickly.
The useful quantity is:

    Cbit_eff = ΔH * Q_c

where Q_c is a structural-quality proxy.

This script tests:
  1. F_c_hat -> raw ΔCbit
  2. F_c_hat -> effective ΔCbit
  3. correct vs wrong samples:
       Cbit_eff_correct > Cbit_eff_wrong
  4. wrong-closure / error samples:
       raw Cbit can be high while Cbit_eff is lower

Inputs
------
Default uses Qwen only because PSA-1 was run only on Qwen:

  CFT3B features:
    C:\Users\ZH\Desktop\AGI\outputs\cft3b_outputs\qwen\cft3b_qwen_features.csv

  PSA1 features:
    C:\Users\ZH\Desktop\AGI\outputs\psa1_outputs\qwen\psa1_qwen_features.csv

Outputs
-------
  C:\Users\ZH\Desktop\AGI\outputs\cft4_psa2_outputs\
    cft4_psa2_joined_features.csv
    cft4_psa2_model_summary.csv
    cft4_psa2_group_summary.csv
    cft4_psa2_verdict.json

Important caveat
----------------
CFT3B and PSA1 datasets are not guaranteed to be identical.
This script joins on:
  condition + graph_id + surface_id
when possible.

If join coverage is low, it falls back to condition-level aggregation.

Verdicts
--------
PASS_CFT4_PSA2_CONSTRAINT_TO_EFFECTIVE_CBIT:
  Constraint-field proxy predicts effective Cbit, and correct samples have
  higher Cbit_eff than wrong samples.

PARTIAL_CFT4_PSA2_CONSTRAINT_TO_CBIT:
  Constraint-field proxy predicts raw Cbit but effective-quality separation is weak.

FAIL_CFT4_PSA2:
  No reliable relation between constraint proxy and Cbit measurements.
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

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_SEED = 42
N_SPLITS = 4

CFT_FEATURES = Path(r"C:\Users\ZH\Desktop\AGI\outputs\cft3b_outputs\qwen\cft3b_qwen_features.csv")
PSA_FEATURES = Path(r"C:\Users\ZH\Desktop\AGI\outputs\psa1_outputs\qwen\psa1_qwen_features.csv")

OUT_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\cft4_psa2_outputs")
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


def cv_indices(df, y=None, group_col="graph_id", stratify=False):
    n = len(df)
    groups = df[group_col].values if group_col in df.columns else None

    if groups is not None and len(np.unique(groups)) >= min(N_SPLITS, n):
        ns = min(N_SPLITS, len(np.unique(groups)))
        cv = GroupKFold(n_splits=ns)
        y_dummy = np.zeros(n) if y is None else y
        return list(cv.split(np.zeros(n), y_dummy, groups=groups))

    if stratify and y is not None:
        vals, counts = np.unique(y, return_counts=True)
        ns = min(N_SPLITS, int(counts.min())) if len(counts) else 2
        if len(vals) >= 2 and ns >= 2:
            cv = StratifiedKFold(n_splits=ns, shuffle=True, random_state=RANDOM_SEED)
            return list(cv.split(np.zeros(n), y))

    ns = min(N_SPLITS, n)
    cv = KFold(n_splits=ns, shuffle=True, random_state=RANDOM_SEED)
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


def eval_binary(df, cols, target, group_col="graph_id"):
    cols = numeric_cols(df, cols)
    if target not in df.columns or not cols:
        return {"valid": False, "reason": "missing_target_or_features"}
    d = df.dropna(subset=[target]).copy()
    if len(d) < 20:
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


def eval_regression(df, cols, target, group_col="graph_id"):
    cols = numeric_cols(df, cols)
    if target not in df.columns or not cols:
        return {"valid": False, "reason": "missing_target_or_features"}

    d = df.dropna(subset=[target]).copy()
    if len(d) < 20:
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


def oof_regression_pred(df, cols, target, group_col="graph_id"):
    cols = numeric_cols(df, cols)
    d = df.dropna(subset=[target]).copy()
    out = pd.Series(index=d.index, dtype=float)
    if not cols or len(d) < 20:
        out[:] = np.nan
        return out.reindex(df.index)

    y = d[target].astype(float).values
    X = d[cols].replace([np.inf, -np.inf], np.nan).values

    for train_idx, test_idx in cv_indices(d, None, group_col=group_col):
        reg = make_regressor()
        reg.fit(X[train_idx], y[train_idx])
        out.iloc[test_idx] = reg.predict(X[test_idx])

    return out.reindex(df.index)


def oof_binary_prob(df, cols, target, group_col="graph_id"):
    cols = numeric_cols(df, cols)
    d = df.dropna(subset=[target]).copy()
    out = pd.Series(index=d.index, dtype=float)
    if not cols or len(d) < 20:
        out[:] = np.nan
        return out.reindex(df.index)

    y = d[target].astype(int).values
    if len(np.unique(y)) < 2:
        out[:] = np.nan
        return out.reindex(df.index)

    X = d[cols].replace([np.inf, -np.inf], np.nan).values
    for train_idx, test_idx in cv_indices(d, y, group_col=group_col, stratify=True):
        if len(np.unique(y[train_idx])) < 2:
            out.iloc[test_idx] = np.nan
            continue
        clf = make_binary_classifier()
        clf.fit(X[train_idx], y[train_idx])
        out.iloc[test_idx] = clf.predict_proba(X[test_idx])[:, 1]

    return out.reindex(df.index)


# -----------------------------
# Feature groups
# -----------------------------

def build_cft_groups(df):
    topk_topology = cols_by_prefix(df, ["topk_topology_"])
    topk_mech = cols_by_prefix(df, ["topk_mechanism_seed_"])
    topk_down = cols_by_prefix(df, ["topk_downstream_seed_"])
    topk_decision = cols_by_prefix(df, ["topk_decision_"])
    topk_all = unique_keep_order(topk_topology + topk_mech + topk_down + topk_decision)

    answer = numeric_cols(df, [
        "score_C", "score_E", "score_OTHER",
        "final_margin_C_minus_E",
        "commit_abs_final_margin",
        "commit_high_abs_margin",
        "binding_priority_proxy_CminusE",
        "binding_priority_proxy_OTHERmax",
    ]) + cols_by_prefix(df, ["ans_margin_"])

    dynamics = unique_keep_order(
        cols_by_prefix(df, ["dyn_", "dyn4_"]) +
        cols_by_prefix(df, ["commit_abs_"]) +
        cols_by_prefix(df, ["other_gap_"]) +
        cols_by_prefix(df, ["margin_sign_changes_"])
    )
    dynamics = [c for c in dynamics if c not in answer]

    cft_no_answer = unique_keep_order(topk_all + dynamics)

    return {
        "TopK_aligned": numeric_cols(df, topk_all),
        "Dynamics_aligned": numeric_cols(df, dynamics),
        "CFT_NoAnswer": numeric_cols(df, cft_no_answer),
        "AnswerReadout": numeric_cols(df, answer),
    }


# -----------------------------
# Join and target construction
# -----------------------------

def add_join_keys(df):
    d = df.copy()

    for col in ["condition", "graph_id", "surface_id"]:
        if col not in d.columns:
            d[col] = "NA"

    # Normalize types.
    d["condition"] = d["condition"].astype(str)
    d["graph_id"] = d["graph_id"].astype(str)
    d["surface_id"] = d["surface_id"].astype(str)

    d["join_key"] = d["condition"] + "||" + d["graph_id"] + "||" + d["surface_id"]
    return d


def load_joined():
    if not CFT_FEATURES.exists():
        raise FileNotFoundError(f"CFT features not found: {CFT_FEATURES}")
    if not PSA_FEATURES.exists():
        raise FileNotFoundError(f"PSA features not found: {PSA_FEATURES}")

    cft = pd.read_csv(CFT_FEATURES).replace([np.inf, -np.inf], np.nan)
    psa = pd.read_csv(PSA_FEATURES).replace([np.inf, -np.inf], np.nan)

    cft = add_join_keys(cft)
    psa = add_join_keys(psa)

    # Prefix PSA columns except join/meta columns.
    meta_cols = {"join_key", "condition", "graph_id", "surface_id", "row_id", "mechanism", "gold"}
    psa_ren = {}
    for c in psa.columns:
        if c not in meta_cols:
            psa_ren[c] = "psa_" + c
    psa = psa.rename(columns=psa_ren)

    # Direct join.
    joined = cft.merge(
        psa,
        on=["join_key", "condition", "graph_id", "surface_id"],
        how="inner",
        suffixes=("", "_psa_meta"),
    )

    join_mode = "row_level"
    coverage = len(joined) / max(1, min(len(cft), len(psa)))

    # Fallback: condition-level mean PSA targets merged to CFT rows.
    if len(joined) < 30 or coverage < 0.20:
        join_mode = "condition_level"
        psa_num = [c for c in psa.columns if c.startswith("psa_") and pd.api.types.is_numeric_dtype(psa[c])]
        psa_agg = psa.groupby("condition")[psa_num].mean().reset_index()
        joined = cft.merge(psa_agg, on="condition", how="left")
        coverage = len(joined) / max(1, len(cft))

    return joined, {
        "join_mode": join_mode,
        "n_cft": int(len(cft)),
        "n_psa": int(len(psa)),
        "n_joined": int(len(joined)),
        "coverage": safe_float(coverage),
    }


def construct_targets(df):
    d = df.copy()

    # PSA target columns with fallback.
    # PSA script prefixes all non-meta as psa_...
    target_map = {
        "raw_cbit_init_to_basin": [
            "psa_Cbit_answer_init_to_basin",
            "psa_mean_Cbit_init_to_basin",
        ],
        "raw_cbit_init_to_final": [
            "psa_Cbit_answer_init_to_final",
            "psa_mean_Cbit_init_to_final",
        ],
        "psa_init_H": [
            "psa_init_answer_H_mean",
            "psa_answer_H_bits_L0",
        ],
        "psa_basin_H": [
            "psa_basin_answer_H_mean",
        ],
        "psa_final_H": [
            "psa_final_answer_H_mean",
        ],
    }

    def first_existing(cols):
        for c in cols:
            if c in d.columns:
                return c
        return None

    for new_col, candidates in target_map.items():
        src = first_existing(candidates)
        if src:
            d[new_col] = d[src]
        else:
            d[new_col] = np.nan

    # Gen error/correctness.
    if "gen_error" not in d.columns:
        if "gold" in d.columns and "answer_pred" in d.columns:
            d["gen_error"] = (d["answer_pred"].astype(str) != d["gold"].astype(str)).astype(int)
        elif "token_correct" in d.columns:
            d["gen_error"] = 1 - d["token_correct"].astype(int)
        elif "psa_token_correct" in d.columns:
            d["gen_error"] = 1 - d["psa_token_correct"].astype(int)
        else:
            d["gen_error"] = np.nan

    d["is_correct"] = 1 - d["gen_error"].astype(float)

    # Structural quality proxy Q_c.
    # Start with correctness; add soft components if available.
    q = d["is_correct"].copy()

    # Mechanism prior: stable/clean and correctly handled closure/source tasks are valued.
    if "mechanism" in d.columns:
        # Do not overfit; small bonus for correct non-random structure.
        mech = d["mechanism"].astype(str)
        q = q + 0.10 * mech.str.contains("stable|closure|source|competition", case=False, regex=True).astype(float)

    if "condition" in d.columns:
        cond = d["condition"].astype(str)
        # Penalize random_broken if model is wrong; reward correct OTHER if available through is_correct already.
        q = q - 0.10 * ((cond.str.contains("random_broken", case=False)) & (d["gen_error"].astype(float) > 0)).astype(float)

    # Normalize to [0,1].
    q = q.clip(lower=0.0, upper=1.0)
    d["Q_correctness_structural"] = q

    # Effective Cbit.
    d["Cbit_eff_basin"] = d["raw_cbit_init_to_basin"] * d["Q_correctness_structural"]
    d["Cbit_eff_final"] = d["raw_cbit_init_to_final"] * d["Q_correctness_structural"]

    # Raw can be positive but ineffective.
    d["raw_high_eff_low_basin"] = (
        (d["raw_cbit_init_to_basin"] > d["raw_cbit_init_to_basin"].median(skipna=True))
        & (d["Cbit_eff_basin"] < d["Cbit_eff_basin"].median(skipna=True))
    ).astype(int)

    return d


# -----------------------------
# Main audit
# -----------------------------

def main():
    joined, join_info = load_joined()
    df = construct_targets(joined)

    groups = build_cft_groups(df)

    # Create OOF estimates from constraint proxy.
    for gname, cols in groups.items():
        if not cols:
            continue
        for target in ["raw_cbit_init_to_basin", "Cbit_eff_basin", "raw_cbit_init_to_final", "Cbit_eff_final"]:
            if target in df.columns:
                df[f"pred_{gname}_{target}"] = oof_regression_pred(df, cols, target, group_col="graph_id")
        if "gen_error" in df.columns:
            df[f"pred_{gname}_gen_error"] = oof_binary_prob(df, cols, "gen_error", group_col="graph_id")

    rows = []
    for gname, cols in groups.items():
        if not cols:
            continue

        for target in [
            "raw_cbit_init_to_basin",
            "Cbit_eff_basin",
            "raw_cbit_init_to_final",
            "Cbit_eff_final",
            "Q_correctness_structural",
        ]:
            res = eval_regression(df, cols, target, group_col="graph_id")
            rows.append({
                "feature_group": gname,
                "target": target,
                "task": "regression",
                "n_features": len(cols),
                "r2": res.get("r2", np.nan),
                "corr": res.get("corr", np.nan),
                "auc": np.nan,
                "f1": np.nan,
                "acc": np.nan,
                "valid": res.get("valid", False),
                "n": res.get("n", np.nan),
            })

        for target in ["gen_error", "raw_high_eff_low_basin"]:
            res = eval_binary(df, cols, target, group_col="graph_id")
            rows.append({
                "feature_group": gname,
                "target": target,
                "task": "binary",
                "n_features": len(cols),
                "r2": np.nan,
                "corr": np.nan,
                "auc": res.get("auc", np.nan),
                "f1": res.get("f1", np.nan),
                "acc": res.get("acc", np.nan),
                "valid": res.get("valid", False),
                "n": res.get("n", np.nan),
            })

    model_summary = pd.DataFrame(rows)

    # Group summaries: correct vs wrong, mechanism/condition.
    group_rows = []
    if "gen_error" in df.columns:
        for val, sub in df.groupby("gen_error"):
            group_rows.append({
                "group_type": "gen_error",
                "group": int(val) if pd.notna(val) else "NA",
                "n": int(len(sub)),
                "raw_cbit_basin_mean": safe_float(sub["raw_cbit_init_to_basin"].mean()),
                "raw_cbit_final_mean": safe_float(sub["raw_cbit_init_to_final"].mean()),
                "cbit_eff_basin_mean": safe_float(sub["Cbit_eff_basin"].mean()),
                "cbit_eff_final_mean": safe_float(sub["Cbit_eff_final"].mean()),
                "Q_mean": safe_float(sub["Q_correctness_structural"].mean()),
            })

    for col in ["condition", "mechanism"]:
        if col in df.columns:
            for val, sub in df.groupby(col):
                group_rows.append({
                    "group_type": col,
                    "group": str(val),
                    "n": int(len(sub)),
                    "raw_cbit_basin_mean": safe_float(sub["raw_cbit_init_to_basin"].mean()),
                    "raw_cbit_final_mean": safe_float(sub["raw_cbit_init_to_final"].mean()),
                    "cbit_eff_basin_mean": safe_float(sub["Cbit_eff_basin"].mean()),
                    "cbit_eff_final_mean": safe_float(sub["Cbit_eff_final"].mean()),
                    "Q_mean": safe_float(sub["Q_correctness_structural"].mean()),
                })

    group_summary = pd.DataFrame(group_rows)

    def get_metric(group, target, key):
        r = model_summary[(model_summary["feature_group"] == group) & (model_summary["target"] == target)]
        if len(r) == 0:
            return np.nan
        return safe_float(r.iloc[0].get(key, np.nan))

    metrics = {
        "join_mode": join_info["join_mode"],
        "join_coverage": join_info["coverage"],
        "n_joined": join_info["n_joined"],

        "topk_raw_basin_corr": get_metric("TopK_aligned", "raw_cbit_init_to_basin", "corr"),
        "cft_raw_basin_corr": get_metric("CFT_NoAnswer", "raw_cbit_init_to_basin", "corr"),
        "dyn_raw_basin_corr": get_metric("Dynamics_aligned", "raw_cbit_init_to_basin", "corr"),

        "topk_eff_basin_corr": get_metric("TopK_aligned", "Cbit_eff_basin", "corr"),
        "cft_eff_basin_corr": get_metric("CFT_NoAnswer", "Cbit_eff_basin", "corr"),
        "dyn_eff_basin_corr": get_metric("Dynamics_aligned", "Cbit_eff_basin", "corr"),

        "topk_eff_basin_r2": get_metric("TopK_aligned", "Cbit_eff_basin", "r2"),
        "cft_eff_basin_r2": get_metric("CFT_NoAnswer", "Cbit_eff_basin", "r2"),
        "dyn_eff_basin_r2": get_metric("Dynamics_aligned", "Cbit_eff_basin", "r2"),

        "cft_gen_error_auc": get_metric("CFT_NoAnswer", "gen_error", "auc"),
        "topk_gen_error_auc": get_metric("TopK_aligned", "gen_error", "auc"),
        "answer_gen_error_auc": get_metric("AnswerReadout", "gen_error", "auc"),

        "cft_raw_high_eff_low_auc": get_metric("CFT_NoAnswer", "raw_high_eff_low_basin", "auc"),
    }

    # Correct/wrong Cbit differences.
    ge_groups = group_summary[group_summary["group_type"] == "gen_error"].copy()
    correct = ge_groups[ge_groups["group"].astype(str) == "0"]
    wrong = ge_groups[ge_groups["group"].astype(str) == "1"]

    if len(correct) and len(wrong):
        metrics["correct_raw_basin_mean"] = safe_float(correct.iloc[0]["raw_cbit_basin_mean"])
        metrics["wrong_raw_basin_mean"] = safe_float(wrong.iloc[0]["raw_cbit_basin_mean"])
        metrics["correct_eff_basin_mean"] = safe_float(correct.iloc[0]["cbit_eff_basin_mean"])
        metrics["wrong_eff_basin_mean"] = safe_float(wrong.iloc[0]["cbit_eff_basin_mean"])
        metrics["eff_correct_minus_wrong"] = metrics["correct_eff_basin_mean"] - metrics["wrong_eff_basin_mean"]
        metrics["raw_correct_minus_wrong"] = metrics["correct_raw_basin_mean"] - metrics["wrong_raw_basin_mean"]
    else:
        metrics["correct_raw_basin_mean"] = np.nan
        metrics["wrong_raw_basin_mean"] = np.nan
        metrics["correct_eff_basin_mean"] = np.nan
        metrics["wrong_eff_basin_mean"] = np.nan
        metrics["eff_correct_minus_wrong"] = np.nan
        metrics["raw_correct_minus_wrong"] = np.nan

    # Verdict.
    reasons = []
    verdict = "UNDETERMINED"

    cft_eff_corr = metrics["cft_eff_basin_corr"]
    cft_eff_r2 = metrics["cft_eff_basin_r2"]
    cft_raw_corr = metrics["cft_raw_basin_corr"]
    eff_gap = metrics["eff_correct_minus_wrong"]
    raw_gap = metrics["raw_correct_minus_wrong"]
    cft_vs_topk_eff_gain = metrics["cft_eff_basin_corr"] - metrics["topk_eff_basin_corr"]

    if np.isfinite(cft_eff_corr) and cft_eff_corr > 0.30 and np.isfinite(eff_gap) and eff_gap > 0:
        verdict = "PASS_CFT4_PSA2_CONSTRAINT_TO_EFFECTIVE_CBIT"
        reasons.append("CFT no-answer proxy predicts effective Cbit, and correct samples have higher Cbit_eff.")
    elif np.isfinite(cft_raw_corr) and cft_raw_corr > 0.30:
        verdict = "PARTIAL_CFT4_PSA2_CONSTRAINT_TO_RAW_CBIT"
        reasons.append("CFT no-answer proxy predicts raw Cbit, but effective-quality separation is weak.")
    elif np.isfinite(eff_gap) and eff_gap > 0:
        verdict = "PARTIAL_CFT4_PSA2_EFFECTIVE_CBIT_SEPARATES_CORRECTNESS"
        reasons.append("Effective Cbit separates correct/wrong samples, but CFT prediction is weak.")
    else:
        verdict = "FAIL_CFT4_PSA2_NO_STABLE_LINK"

    if np.isfinite(cft_vs_topk_eff_gain) and cft_vs_topk_eff_gain > 0:
        reasons.append("CFT no-answer proxy improves effective-Cbit prediction over TopK.")
    if np.isfinite(raw_gap) and np.isfinite(eff_gap) and raw_gap <= eff_gap:
        reasons.append("Effective Cbit improves correctness separation relative to raw Cbit.")

    verdict_obj = {
        "verdict": verdict,
        "reasons": reasons,
        "metrics": metrics,
        "join_info": join_info,
        "feature_set_sizes": {k: len(v) for k, v in groups.items()},
        "outputs": {
            "joined_features": str(OUT_DIR / "cft4_psa2_joined_features.csv"),
            "model_summary": str(OUT_DIR / "cft4_psa2_model_summary.csv"),
            "group_summary": str(OUT_DIR / "cft4_psa2_group_summary.csv"),
        },
        "top_rows": model_summary.sort_values(["target", "corr", "r2"], ascending=[True, False, False]).head(30).to_dict(orient="records"),
    }

    df.to_csv(OUT_DIR / "cft4_psa2_joined_features.csv", index=False, encoding="utf-8-sig")
    model_summary.to_csv(OUT_DIR / "cft4_psa2_model_summary.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(OUT_DIR / "cft4_psa2_group_summary.csv", index=False, encoding="utf-8-sig")
    with open(OUT_DIR / "cft4_psa2_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict_obj, f, ensure_ascii=False, indent=2)

    print(json.dumps(verdict_obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
