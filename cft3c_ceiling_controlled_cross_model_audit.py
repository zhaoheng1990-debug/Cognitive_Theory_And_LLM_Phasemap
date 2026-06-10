# -*- coding: utf-8 -*-
r"""
CFT-3C: Ceiling-Controlled Cross-Model Constraint Field Audit

Purpose
-------
CFT-3B returned:

  Qwen  : PASS-Strong
  Gemma : PASS Gen Gain
  Llama : script FAIL, but micro-positive and very high TopK baseline

Hypothesis
----------
The small Llama gain is not a theoretical failure, but a ceiling effect:

    Gain(CFT3B_NoAnswer over TopK_aligned)
    ∝
    1 - TopK_baseline_strength

In other words:
  - When TopK alone is already near ceiling, CFT3B has little GenError AUC headroom.
  - Dynamics may still strongly improve commitment R2 and submechanism F1.
  - The correct evaluation should stratify examples by TopK confidence / difficulty.

Inputs
------
This script reuses CFT-3B feature CSV files:

  C:\Users\ZH\Desktop\AGI\outputs\cft3b_outputs\qwen\cft3b_qwen_features.csv
  C:\Users\ZH\Desktop\AGI\outputs\cft3b_outputs\llama\cft3b_llama_features.csv
  C:\Users\ZH\Desktop\AGI\outputs\cft3b_outputs\gemma\cft3b_gemma_features.csv

Outputs
-------
  C:\Users\ZH\Desktop\AGI\outputs\cft3c_outputs\cft3c_cross_model_ceiling_summary.csv
  C:\Users\ZH\Desktop\AGI\outputs\cft3c_outputs\cft3c_bucket_summary.csv
  C:\Users\ZH\Desktop\AGI\outputs\cft3c_outputs\cft3c_verdict.json

Method
------
For each model:
  1. Build feature groups matching CFT-3B:
     - TopK_all_aligned
     - Dynamics_aligned
     - CFT3C_NoAnswer = TopK_all_aligned + Dynamics_aligned
     - AnswerReadout

  2. Cross-fit TopK model to obtain out-of-fold TopK probabilities.
     These probabilities define baseline difficulty / ceiling:
       easy   = TopK confident
       medium = intermediate
       hard   = TopK uncertain / misranked

  3. Within each bucket, evaluate:
       TopK
       Dynamics
       CFT3C_NoAnswer
       AnswerReadout

  4. Test whether:
       gain_hard > gain_medium > gain_easy
     or at least:
       gain increases as TopK baseline decreases.

Verdicts
--------
PASS_CFT3C_CEILING_EFFECT:
  Cross-model pattern supports ceiling effect.

PARTIAL_CFT3C_CEILING_EFFECT:
  At least Llama or two models support ceiling effect.

FAIL_CFT3C_NO_CEILING_EFFECT:
  No evidence that TopK ceiling explains mixed-positive CFT-3B.
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

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_SEED = 42
N_SPLITS = 4

INPUTS = {
    "qwen": Path(r"C:\Users\ZH\Desktop\AGI\outputs\cft3b_outputs\qwen\cft3b_qwen_features.csv"),
    "llama": Path(r"C:\Users\ZH\Desktop\AGI\outputs\cft3b_outputs\llama\cft3b_llama_features.csv"),
    "gemma": Path(r"C:\Users\ZH\Desktop\AGI\outputs\cft3b_outputs\gemma\cft3b_gemma_features.csv"),
}

OUT_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\cft3c_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Utility
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


def eval_binary(df, cols, target="gen_error", group_col="graph_id"):
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


def eval_multiclass(df, cols, target="submechanism", group_col="graph_id"):
    cols = numeric_cols(df, cols)
    if target not in df.columns or not cols:
        return {"valid": False, "reason": "missing_target_or_features"}

    d = df.dropna(subset=[target]).copy()
    if len(d) < 20:
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


def oof_binary_probs(df, cols, target="gen_error", group_col="graph_id"):
    cols = numeric_cols(df, cols)
    d = df.dropna(subset=[target]).copy()
    out = pd.Series(index=d.index, dtype=float)
    y = d[target].astype(int).values
    if len(np.unique(y)) < 2 or not cols:
        out[:] = np.nan
        return out

    X = d[cols].replace([np.inf, -np.inf], np.nan).values
    for train_idx, test_idx in cv_indices(d, y, group_col=group_col, stratify=True):
        if len(np.unique(y[train_idx])) < 2:
            out.iloc[test_idx] = np.nan
            continue
        clf = make_binary_classifier()
        clf.fit(X[train_idx], y[train_idx])
        out.iloc[test_idx] = clf.predict_proba(X[test_idx])[:, 1]

    return out.reindex(df.index)


def build_feature_groups(df):
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
    cft_with_answer = unique_keep_order(cft_no_answer + answer)

    return {
        "TopK_all_aligned": numeric_cols(df, topk_all),
        "Dynamics_aligned": numeric_cols(df, dynamics),
        "CFT3C_NoAnswer": numeric_cols(df, cft_no_answer),
        "AnswerReadout": numeric_cols(df, answer),
        "CFT3C_WithAnswer": numeric_cols(df, cft_with_answer),
    }


def bucket_by_topk_oof(df, topk_prob_col="topk_oof_prob"):
    d = df.copy()
    p = d[topk_prob_col].astype(float)

    # confidence = distance from 0.5
    d["topk_oof_confidence"] = np.abs(p - 0.5)
    d["topk_oof_pred"] = (p >= 0.5).astype(int)
    d["topk_oof_correct"] = (d["topk_oof_pred"] == d["gen_error"].astype(int)).astype(int)

    # Difficulty:
    # hard   = low confidence or topk wrong
    # easy   = high confidence and topk correct
    # medium = rest
    valid_conf = d["topk_oof_confidence"].replace([np.inf, -np.inf], np.nan)
    q33 = valid_conf.quantile(0.33)
    q67 = valid_conf.quantile(0.67)

    def assign(row):
        if not np.isfinite(row["topk_oof_confidence"]):
            return "unknown"
        if row["topk_oof_correct"] == 0:
            return "hard_topk_wrong"
        if row["topk_oof_confidence"] <= q33:
            return "hard_low_conf"
        if row["topk_oof_confidence"] >= q67:
            return "easy_high_conf"
        return "medium"

    d["ceiling_bucket"] = d.apply(assign, axis=1)

    # Also a simpler tertile-only bucket, useful if TopK is too good.
    labels = ["low_conf_hard", "mid_conf", "high_conf_easy"]
    try:
        d["confidence_tertile"] = pd.qcut(
            d["topk_oof_confidence"].rank(method="first"),
            q=3,
            labels=labels,
        ).astype(str)
    except Exception:
        d["confidence_tertile"] = "all"

    return d


def eval_all_groups(df, groups, target="gen_error"):
    rows = []
    for name, cols in groups.items():
        gen = eval_binary(df, cols, target=target, group_col="graph_id")
        subm = eval_multiclass(df, cols, target="submechanism", group_col="graph_id")
        commit = eval_regression(df, cols, target="commit_abs_final_margin", group_col="graph_id")
        rows.append({
            "feature_group": name,
            "n_features": len(cols),
            "gen_auc": gen.get("auc", np.nan),
            "gen_acc": gen.get("acc", np.nan),
            "gen_f1": gen.get("f1", np.nan),
            "submechanism_macro_f1": subm.get("macro_f1", np.nan),
            "commit_abs_r2": commit.get("r2", np.nan),
            "commit_abs_corr": commit.get("corr", np.nan),
            "valid_gen": gen.get("valid", False),
            "valid_submech": subm.get("valid", False),
            "valid_commit": commit.get("valid", False),
            "n": gen.get("n", len(df)),
        })
    return pd.DataFrame(rows)


def metric(summary, group, key):
    r = summary[summary["feature_group"] == group]
    if len(r) == 0:
        return np.nan
    return safe_float(r.iloc[0].get(key, np.nan))


def run_one_model(model_key, input_path):
    info = {
        "model_key": model_key,
        "input_path": str(input_path),
        "exists": input_path.exists(),
        "status": "pending",
    }

    if not input_path.exists():
        info.update({"status": "missing_input", "error": f"Input not found: {input_path}"})
        return info, pd.DataFrame(), pd.DataFrame()

    df = pd.read_csv(input_path).replace([np.inf, -np.inf], np.nan)
    if "gen_error" not in df.columns:
        if "gold" in df.columns and "answer_pred" in df.columns:
            df["gen_error"] = (df["answer_pred"].astype(str) != df["gold"].astype(str)).astype(int)
        elif "token_correct" in df.columns:
            df["gen_error"] = 1 - df["token_correct"].astype(int)
        else:
            info.update({"status": "missing_gen_error", "error": "No gen_error/gold/answer_pred found"})
            return info, pd.DataFrame(), pd.DataFrame()

    groups = build_feature_groups(df)

    topk_cols = groups.get("TopK_all_aligned", [])
    if not topk_cols:
        info.update({"status": "missing_topk_features", "error": "No aligned TopK features found"})
        return info, pd.DataFrame(), pd.DataFrame()

    df["topk_oof_prob"] = oof_binary_probs(df, topk_cols, target="gen_error", group_col="graph_id")
    df = bucket_by_topk_oof(df, "topk_oof_prob")

    # Overall summary
    overall = eval_all_groups(df, groups)
    overall["model_key"] = model_key
    overall["bucket_type"] = "overall"
    overall["bucket"] = "all"

    bucket_rows = []
    bucket_types = ["ceiling_bucket", "confidence_tertile"]
    for bt in bucket_types:
        for bucket, sub in df.groupby(bt):
            if len(sub) < 30:
                continue
            if sub["gen_error"].nunique() < 2:
                # Can still evaluate submechanism/commit, but skip binary AUC may be invalid.
                pass
            summ = eval_all_groups(sub, groups)
            summ["model_key"] = model_key
            summ["bucket_type"] = bt
            summ["bucket"] = str(bucket)
            summ["bucket_n"] = int(len(sub))
            summ["bucket_gen_error_rate"] = safe_float(sub["gen_error"].mean())
            summ["bucket_topk_oof_acc"] = safe_float(sub["topk_oof_correct"].mean())
            summ["bucket_topk_oof_conf"] = safe_float(sub["topk_oof_confidence"].mean())
            bucket_rows.append(summ)

    bucket_summary = pd.concat(bucket_rows, ignore_index=True) if bucket_rows else pd.DataFrame()

    # Extract row-level gains for buckets.
    def row_for(group, summ):
        r = summ[summ["feature_group"] == group]
        return r.iloc[0].to_dict() if len(r) else {}

    overall_topk = row_for("TopK_all_aligned", overall)
    overall_cft = row_for("CFT3C_NoAnswer", overall)
    overall_dyn = row_for("Dynamics_aligned", overall)
    overall_ans = row_for("AnswerReadout", overall)

    overall_metrics = {
        "topk_auc": overall_topk.get("gen_auc", np.nan),
        "cft_auc": overall_cft.get("gen_auc", np.nan),
        "dyn_auc": overall_dyn.get("gen_auc", np.nan),
        "answer_auc": overall_ans.get("gen_auc", np.nan),
        "cft_gain_auc": safe_float(overall_cft.get("gen_auc", np.nan) - overall_topk.get("gen_auc", np.nan)),
        "topk_sub_f1": overall_topk.get("submechanism_macro_f1", np.nan),
        "cft_sub_f1": overall_cft.get("submechanism_macro_f1", np.nan),
        "cft_gain_sub_f1": safe_float(overall_cft.get("submechanism_macro_f1", np.nan) - overall_topk.get("submechanism_macro_f1", np.nan)),
        "topk_commit_r2": overall_topk.get("commit_abs_r2", np.nan),
        "cft_commit_r2": overall_cft.get("commit_abs_r2", np.nan),
        "cft_gain_commit_r2": safe_float(overall_cft.get("commit_abs_r2", np.nan) - overall_topk.get("commit_abs_r2", np.nan)),
    }

    # Bucket gain table.
    gain_rows = []
    if not bucket_summary.empty:
        for (bt, bucket), sub_summ in bucket_summary.groupby(["bucket_type", "bucket"]):
            topk = row_for("TopK_all_aligned", sub_summ)
            cft = row_for("CFT3C_NoAnswer", sub_summ)
            dyn = row_for("Dynamics_aligned", sub_summ)
            ans = row_for("AnswerReadout", sub_summ)
            if not topk or not cft:
                continue
            gain_rows.append({
                "model_key": model_key,
                "bucket_type": bt,
                "bucket": bucket,
                "n": int(topk.get("bucket_n", topk.get("n", 0))),
                "gen_error_rate": topk.get("bucket_gen_error_rate", np.nan),
                "topk_oof_acc": topk.get("bucket_topk_oof_acc", np.nan),
                "topk_oof_conf": topk.get("bucket_topk_oof_conf", np.nan),
                "topk_auc": topk.get("gen_auc", np.nan),
                "cft_auc": cft.get("gen_auc", np.nan),
                "dyn_auc": dyn.get("gen_auc", np.nan),
                "answer_auc": ans.get("gen_auc", np.nan),
                "cft_gain_auc": safe_float(cft.get("gen_auc", np.nan) - topk.get("gen_auc", np.nan)),
                "topk_sub_f1": topk.get("submechanism_macro_f1", np.nan),
                "cft_sub_f1": cft.get("submechanism_macro_f1", np.nan),
                "cft_gain_sub_f1": safe_float(cft.get("submechanism_macro_f1", np.nan) - topk.get("submechanism_macro_f1", np.nan)),
                "topk_commit_r2": topk.get("commit_abs_r2", np.nan),
                "cft_commit_r2": cft.get("commit_abs_r2", np.nan),
                "cft_gain_commit_r2": safe_float(cft.get("commit_abs_r2", np.nan) - topk.get("commit_abs_r2", np.nan)),
            })

    gain_df = pd.DataFrame(gain_rows)

    # Ceiling effect tests.
    ceiling_evidence = {}
    if not gain_df.empty:
        tert = gain_df[gain_df["bucket_type"] == "confidence_tertile"].copy()
        # Lower confidence = harder. We expect gain in low_conf_hard >= high_conf_easy.
        hard = tert[tert["bucket"] == "low_conf_hard"]
        easy = tert[tert["bucket"] == "high_conf_easy"]
        mid = tert[tert["bucket"] == "mid_conf"]

        hard_gain = safe_float(hard["cft_gain_auc"].iloc[0]) if len(hard) else np.nan
        mid_gain = safe_float(mid["cft_gain_auc"].iloc[0]) if len(mid) else np.nan
        easy_gain = safe_float(easy["cft_gain_auc"].iloc[0]) if len(easy) else np.nan

        ceiling_evidence.update({
            "hard_gain_auc": hard_gain,
            "mid_gain_auc": mid_gain,
            "easy_gain_auc": easy_gain,
            "hard_gt_easy": bool(np.isfinite(hard_gain) and np.isfinite(easy_gain) and hard_gain > easy_gain),
            "hard_gt_mid": bool(np.isfinite(hard_gain) and np.isfinite(mid_gain) and hard_gain > mid_gain),
        })

        # Correlation across all buckets between topk_auc and cft_gain_auc.
        valid = gain_df[["topk_auc", "cft_gain_auc"]].dropna()
        if len(valid) >= 3:
            try:
                ceiling_evidence["corr_topk_auc_vs_gain"] = safe_float(np.corrcoef(valid["topk_auc"], valid["cft_gain_auc"])[0, 1])
            except Exception:
                ceiling_evidence["corr_topk_auc_vs_gain"] = np.nan
        else:
            ceiling_evidence["corr_topk_auc_vs_gain"] = np.nan

    # Model verdict.
    reasons = []
    verdict = "UNDETERMINED"

    overall_gain = overall_metrics["cft_gain_auc"]
    commit_gain = overall_metrics["cft_gain_commit_r2"]
    sub_gain = overall_metrics["cft_gain_sub_f1"]
    hard_gt_easy = ceiling_evidence.get("hard_gt_easy", False)
    hard_gain = ceiling_evidence.get("hard_gain_auc", np.nan)
    corr_ceiling = ceiling_evidence.get("corr_topk_auc_vs_gain", np.nan)

    if np.isfinite(overall_gain) and overall_gain > 0.03:
        verdict = "PASS_CFT3C_MODEL_DIRECT_GAIN"
        reasons.append("Overall CFT3C no-answer GenError gain over TopK is substantial.")
    elif np.isfinite(overall_gain) and overall_gain > 0 and (hard_gt_easy or (np.isfinite(corr_ceiling) and corr_ceiling < -0.3)):
        verdict = "PASS_CFT3C_MODEL_CEILING_EFFECT"
        reasons.append("Overall gain is small but bucket analysis supports TopK ceiling effect.")
    elif np.isfinite(overall_gain) and overall_gain > 0 and (np.isfinite(commit_gain) and commit_gain > 0.2 or np.isfinite(sub_gain) and sub_gain > 0.01):
        verdict = "PARTIAL_CFT3C_MODEL_MICRO_POSITIVE_WITH_STRUCTURE_GAIN"
        reasons.append("Overall GenError gain is micro-positive, but structure/commitment gains remain positive.")
    elif np.isfinite(overall_gain) and overall_gain > 0:
        verdict = "PARTIAL_CFT3C_MODEL_MICRO_POSITIVE"
        reasons.append("Overall GenError gain is micro-positive but ceiling evidence is weak.")
    else:
        verdict = "FAIL_CFT3C_MODEL_NO_GAIN"
        reasons.append("CFT3C does not improve over TopK and no ceiling evidence was detected.")

    info.update({
        "status": "done",
        "verdict": verdict,
        "reasons": reasons,
        "overall_metrics": overall_metrics,
        "ceiling_evidence": ceiling_evidence,
        "n_rows": int(len(df)),
        "feature_set_sizes": {k: len(v) for k, v in groups.items()},
    })

    return info, overall, gain_df


def main():
    all_infos = []
    all_overall = []
    all_gains = []

    for model_key, path in INPUTS.items():
        print("=" * 80)
        print(f"[CFT-3C] {model_key}: {path}")
        info, overall, gains = run_one_model(model_key, path)
        all_infos.append(info)
        if not overall.empty:
            all_overall.append(overall)
        if not gains.empty:
            all_gains.append(gains)

        print(json.dumps({
            "model_key": info.get("model_key"),
            "status": info.get("status"),
            "verdict": info.get("verdict"),
            "overall_metrics": info.get("overall_metrics"),
            "ceiling_evidence": info.get("ceiling_evidence"),
            "error": info.get("error"),
        }, ensure_ascii=False, indent=2))

    overall_df = pd.concat(all_overall, ignore_index=True) if all_overall else pd.DataFrame()
    gains_df = pd.concat(all_gains, ignore_index=True) if all_gains else pd.DataFrame()

    overall_df.to_csv(OUT_DIR / "cft3c_model_group_summary.csv", index=False, encoding="utf-8-sig")
    gains_df.to_csv(OUT_DIR / "cft3c_bucket_summary.csv", index=False, encoding="utf-8-sig")

    # Cross-model summary from infos.
    rows = []
    for info in all_infos:
        m = info.get("overall_metrics", {}) or {}
        ce = info.get("ceiling_evidence", {}) or {}
        rows.append({
            "model_key": info.get("model_key"),
            "status": info.get("status"),
            "verdict": info.get("verdict"),
            "topk_auc": m.get("topk_auc"),
            "cft_auc": m.get("cft_auc"),
            "cft_gain_auc": m.get("cft_gain_auc"),
            "topk_sub_f1": m.get("topk_sub_f1"),
            "cft_sub_f1": m.get("cft_sub_f1"),
            "cft_gain_sub_f1": m.get("cft_gain_sub_f1"),
            "topk_commit_r2": m.get("topk_commit_r2"),
            "cft_commit_r2": m.get("cft_commit_r2"),
            "cft_gain_commit_r2": m.get("cft_gain_commit_r2"),
            "hard_gain_auc": ce.get("hard_gain_auc"),
            "mid_gain_auc": ce.get("mid_gain_auc"),
            "easy_gain_auc": ce.get("easy_gain_auc"),
            "hard_gt_easy": ce.get("hard_gt_easy"),
            "corr_topk_auc_vs_gain": ce.get("corr_topk_auc_vs_gain"),
            "n_rows": info.get("n_rows"),
            "error": info.get("error"),
        })

    cross_df = pd.DataFrame(rows)
    cross_df.to_csv(OUT_DIR / "cft3c_cross_model_ceiling_summary.csv", index=False, encoding="utf-8-sig")

    done = [x for x in all_infos if x.get("status") == "done"]
    pass_count = sum(str(x.get("verdict", "")).startswith("PASS") for x in done)
    partial_count = sum(str(x.get("verdict", "")).startswith("PARTIAL") for x in done)
    ceiling_count = sum("CEILING_EFFECT" in str(x.get("verdict", "")) for x in done)

    # Global verdict.
    if len(done) == 0:
        global_verdict = "FAIL_CFT3C_NO_MODELS"
    elif pass_count == len(done):
        global_verdict = "PASS_CFT3C_CROSS_MODEL"
    elif pass_count + partial_count == len(done) and ceiling_count >= 1:
        global_verdict = "PASS_CFT3C_CEILING_EXPLAINED_MIXED"
    elif pass_count + partial_count == len(done):
        global_verdict = "PARTIAL_CFT3C_ALL_NONNEGATIVE"
    elif pass_count >= 1:
        global_verdict = "PARTIAL_CFT3C_MIXED"
    else:
        global_verdict = "FAIL_CFT3C_NO_SUPPORT"

    global_obj = {
        "verdict": global_verdict,
        "n_models_done": len(done),
        "n_models_pass": pass_count,
        "n_models_partial": partial_count,
        "n_models_ceiling": ceiling_count,
        "models": all_infos,
        "outputs": {
            "cross_summary": str(OUT_DIR / "cft3c_cross_model_ceiling_summary.csv"),
            "bucket_summary": str(OUT_DIR / "cft3c_bucket_summary.csv"),
            "model_group_summary": str(OUT_DIR / "cft3c_model_group_summary.csv"),
        },
    }

    with open(OUT_DIR / "cft3c_verdict.json", "w", encoding="utf-8") as f:
        json.dump(global_obj, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(json.dumps(global_obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
