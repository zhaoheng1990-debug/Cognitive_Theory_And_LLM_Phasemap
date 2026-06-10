# -*- coding: utf-8 -*-
r"""
CFT-2: Constraint Field Proxy with DYN4 No-Answer Convergence

Purpose
-------
CFT-1 failed to beat answer margin robustly because the original constraint-field proxy
was contaminated by answer-basin readout and used weak dynamics proxies.

Recent chain:
  TSA-1  : TopK preserves coarse mechanism topology.
  TCD-2  : TopK loses submechanism / binding direction.
  MPR-1B : H(S|X)>0, same final answer still contains internal structure.
  DYN-4B : TopK + DYN4 FullDynamics, without AnswerReadout, strongly predicts GenError and commitment.

CFT-2 now rebuilds the constraint-field proxy:

  F_c^(2) = [
      TopK topology,
      Binding/Priority / submechanism structure,
      DYN4 velocity-curvature,
      DYN4 boundary-late / late-stabilization,
      DYN4 residual / order-residual,
  ]

Crucially:
  - Main proxy excludes AnswerReadout.
  - AnswerReadout is included only as a baseline / upper contamination control.
  - Main tasks include GenError, Mechanism, Submechanism, Same-answer split.
  - WrongClosure-vs-RandomBroken is included when labels exist.

Input:
  C:\Users\ZH\Desktop\AGI\outputs\dyn4_outputs\dyn4_features_augmented.csv

Outputs:
  C:\Users\ZH\Desktop\AGI\outputs\cft2_outputs\cft2_model_summary.csv
  C:\Users\ZH\Desktop\AGI\outputs\cft2_outputs\cft2_same_answer_summary.json
  C:\Users\ZH\Desktop\AGI\outputs\cft2_outputs\cft2_wrong_closure_random_summary.json
  C:\Users\ZH\Desktop\AGI\outputs\cft2_outputs\cft2_verdict.json
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
N_SPLITS = 5

INPUT_FEATURES = Path(r"C:\Users\ZH\Desktop\AGI\outputs\dyn4_outputs\dyn4_features_augmented.csv")
OUT_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\cft2_outputs")
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


def eval_multiclass(df, cols, target="mechanism", group_col="graph_id"):
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


def crossfit_residualize(df, cols, against_cols, group_col="graph_id", suffix="__orth"):
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


def build_base_groups(df):
    topk = cols_by_prefix(df, ["topk_"])

    answer = numeric_cols(df, [
        "ans_margin_L20", "ans_margin_L21", "ans_margin_L22",
        "ans_margin_L23", "ans_margin_L24", "ans_margin_L25",
        "ans_margin_main_mean", "ans_margin_main_slope", "ans_margin_main_area",
        "ans_margin_main_std", "ans_margin_main_minabs", "ans_margin_first_flip",
        "final_margin_C_minus_E",
        "commit_abs_final_margin", "commit_high_abs_margin",
    ])

    # DYN4 categories
    raw_dyn = [c for c in cols_by_prefix(df, ["dyn_", "dyn2_"]) if "__" not in c]
    dyn4 = cols_by_prefix(df, ["dyn4_"])
    order_precursor = cols_by_prefix(df, ["order_precursor_"])
    order_residual = cols_containing(df, ["__opresid"])

    boundary_late = unique_keep_order(
        numeric_cols(df, [
            "dyn4_boundary_residence_q25",
            "dyn4_boundary_residence_q50",
            "dyn4_abs_margin_area",
            "dyn4_margin_area",
            "dyn4_late_minus_early_abs",
            "dyn4_late_over_early_abs",
            "dyn4_abs_margin_slope",
            "dyn4_final_proxy_abs_L25",
            "dyn4_start_abs_L20",
        ])
    )

    velocity_curvature = unique_keep_order(
        cols_containing(df, ["velocity", "accel", "curvature", "dmargin", "second", "sign_changes"])
    )

    # Binding / priority proxies are non-TopK and non-answer, but include direction-sensitive dynamics/probes.
    # In this synthetic setup, we approximate binding/priority with order residual / order precursor / raw dynamics.
    binding_priority = unique_keep_order(order_residual + order_precursor + velocity_curvature)

    dyn4_full = unique_keep_order(raw_dyn + dyn4 + order_precursor + order_residual)
    cft2_no_answer = unique_keep_order(topk + binding_priority + dyn4_full)
    cft2_with_answer = unique_keep_order(cft2_no_answer + answer)

    groups = {
        "TopK": topk,
        "AnswerReadout": answer,
        "RawDynamics": raw_dyn,
        "DYN4_BoundaryLate": boundary_late,
        "DYN4_VelocityCurvature": velocity_curvature,
        "DYN4_OrderPrecursor": order_precursor,
        "DYN4_OrderResidual": order_residual,
        "BindingPriorityProxy": binding_priority,
        "DYN4_FullDynamics": dyn4_full,
        "CFT2_NoAnswer": cft2_no_answer,
        "CFT2_WithAnswer": cft2_with_answer,
        "TopK_plus_BindingPriority": unique_keep_order(topk + binding_priority),
        "TopK_plus_DYN4": unique_keep_order(topk + dyn4_full),
        "BindingPriority_plus_DYN4": unique_keep_order(binding_priority + dyn4_full),
    }
    return {k: numeric_cols(df, v) for k, v in groups.items() if len(numeric_cols(df, v)) > 0}


def add_orthogonal_groups(df, groups):
    answer = groups.get("AnswerReadout", [])
    # Residualize key groups against answer readout only.
    for g in ["TopK", "DYN4_FullDynamics", "BindingPriorityProxy", "CFT2_NoAnswer"]:
        cols = groups.get(g, [])
        if cols and answer:
            orth = crossfit_residualize(df, cols, answer, group_col="graph_id", suffix=f"__{g}_ansorth")
            for c in orth.columns:
                df[c] = orth[c]
            groups[g + "__answer_orthogonal"] = list(orth.columns)
    return df, groups


def same_answer_split(df, groups, target="submechanism"):
    if "answer_pred" not in df.columns or target not in df.columns:
        return {"valid": False, "reason": "missing_answer_pred_or_target"}
    details = []
    for answer, sub in df.groupby("answer_pred"):
        if len(sub) < 80 or sub[target].nunique() < 2:
            continue
        for name, cols in groups.items():
            if name.endswith("WithAnswer"):
                continue
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


def wrong_closure_random_audit(df, groups):
    # Flexible label matching.
    if "submechanism" not in df.columns:
        return {"valid": False, "reason": "missing_submechanism"}

    sm = df["submechanism"].astype(str)
    mech = df["mechanism"].astype(str) if "mechanism" in df.columns else pd.Series([""] * len(df))

    wrong_mask = sm.str.contains("closure", case=False, na=False) & (
        sm.str.contains("_E", case=False, na=False) | sm.str.contains("wrong", case=False, na=False)
    )
    random_mask = sm.str.contains("random", case=False, na=False) | mech.str.contains("random", case=False, na=False)

    d = df[wrong_mask | random_mask].copy()
    if len(d) < 50 or wrong_mask.sum() == 0 or random_mask.sum() == 0:
        return {"valid": False, "reason": "insufficient_wrong_closure_or_random", "n": int(len(d))}

    d["wrong_closure_vs_random"] = wrong_mask[wrong_mask | random_mask].astype(int).values

    rows = []
    for name, cols in groups.items():
        if "AnswerReadout" in name:
            continue
        res = eval_binary(d, cols, target="wrong_closure_vs_random", group_col="graph_id")
        if res.get("valid"):
            rows.append({
                "feature_group": name,
                "auc": res.get("auc", np.nan),
                "acc": res.get("acc", np.nan),
                "f1": res.get("f1", np.nan),
                "n": res.get("n", len(d)),
            })

    if not rows:
        return {"valid": False, "reason": "no_valid_models", "n": int(len(d))}
    out = pd.DataFrame(rows).sort_values("auc", ascending=False)
    return {"valid": True, "summary": out.to_dict(orient="records"), "n": int(len(d))}


def main():
    if not INPUT_FEATURES.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_FEATURES}")

    df = pd.read_csv(INPUT_FEATURES).replace([np.inf, -np.inf], np.nan)

    groups = build_base_groups(df)
    df, groups = add_orthogonal_groups(df, groups)

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
        ["gen_auc", "submechanism_macro_f1", "commit_abs_r2"],
        ascending=False,
        na_position="last",
    )

    same_ans = same_answer_split(df, groups, target="submechanism")
    wrong_random = wrong_closure_random_audit(df, groups)

    def metric(group, key):
        r = model_summary[model_summary["feature_group"] == group]
        if len(r) == 0:
            return np.nan
        return safe_float(r.iloc[0].get(key, np.nan))

    metrics = {
        "topk_gen_auc": metric("TopK", "gen_auc"),
        "answer_gen_auc": metric("AnswerReadout", "gen_auc"),
        "cft2_noanswer_gen_auc": metric("CFT2_NoAnswer", "gen_auc"),
        "cft2_withanswer_gen_auc": metric("CFT2_WithAnswer", "gen_auc"),
        "cft2_noanswer_orth_gen_auc": metric("CFT2_NoAnswer__answer_orthogonal", "gen_auc"),
        "topk_submechanism_macro_f1": metric("TopK", "submechanism_macro_f1"),
        "cft2_noanswer_submechanism_macro_f1": metric("CFT2_NoAnswer", "submechanism_macro_f1"),
        "cft2_noanswer_orth_submechanism_macro_f1": metric("CFT2_NoAnswer__answer_orthogonal", "submechanism_macro_f1"),
        "topk_commit_r2": metric("TopK", "commit_abs_r2"),
        "answer_commit_r2": metric("AnswerReadout", "commit_abs_r2"),
        "cft2_noanswer_commit_r2": metric("CFT2_NoAnswer", "commit_abs_r2"),
        "cft2_withanswer_commit_r2": metric("CFT2_WithAnswer", "commit_abs_r2"),
    }

    # Verdict
    reasons = []
    verdict = "UNDETERMINED"
    eps = 0.01

    topk_auc = metrics["topk_gen_auc"]
    cft_auc = metrics["cft2_noanswer_gen_auc"]
    cft_orth_auc = metrics["cft2_noanswer_orth_gen_auc"]
    topk_sub = metrics["topk_submechanism_macro_f1"]
    cft_sub = metrics["cft2_noanswer_submechanism_macro_f1"]
    cft_sub_orth = metrics["cft2_noanswer_orth_submechanism_macro_f1"]
    cft_commit = metrics["cft2_noanswer_commit_r2"]

    same_answer_pass = False
    if same_ans.get("valid"):
        agg = pd.DataFrame(same_ans.get("aggregate", []))
        r = agg[agg["feature_group"] == "CFT2_NoAnswer"]
        if len(r) and safe_float(r.iloc[0]["same_answer_macro_f1_mean"]) > 0.90:
            same_answer_pass = True

    if not np.isfinite(topk_auc) or not np.isfinite(cft_auc):
        verdict = "FAIL_EVAL_INVALID"
        reasons.append("TopK or CFT2_NoAnswer not evaluable.")
    elif cft_auc > topk_auc + 0.05 and np.isfinite(cft_commit) and cft_commit > 0.80 and cft_sub > topk_sub + eps:
        verdict = "PASS_CFT2_NOANSWER_CONSTRAINT_FIELD_PROXY"
        reasons.append("CFT2 no-answer proxy strongly improves GenError, commitment, and submechanism over TopK.")
        if same_answer_pass:
            reasons.append("Same-answer structure remains recoverable with CFT2 no-answer proxy.")
    elif cft_auc > topk_auc + 0.03 and cft_sub > topk_sub + eps:
        verdict = "PASS_CFT2_STRUCTURE_GAIN_COMMIT_PARTIAL"
        reasons.append("CFT2 improves GenError and submechanism, but commitment signal is partial.")
    elif cft_auc > topk_auc + eps:
        verdict = "PARTIAL_CFT2_GEN_GAIN_ONLY"
        reasons.append("CFT2 improves GenError but structure/commitment criteria are weak.")
    else:
        verdict = "FAIL_CFT2_NO_GAIN_OVER_TOPK"
        reasons.append("CFT2 no-answer proxy does not improve over TopK.")

    verdict_obj = {
        "verdict": verdict,
        "reasons": reasons,
        "metrics": metrics,
        "same_answer_valid": same_ans.get("valid", False),
        "wrong_closure_random_valid": wrong_random.get("valid", False),
        "feature_set_sizes": {k: len(v) for k, v in groups.items()},
        "n_rows": int(len(df)),
        "n_graphs": int(df["graph_id"].nunique()) if "graph_id" in df.columns else None,
        "input_features": str(INPUT_FEATURES),
        "outputs": {
            "model_summary": str(OUT_DIR / "cft2_model_summary.csv"),
            "same_answer_summary": str(OUT_DIR / "cft2_same_answer_summary.json"),
            "wrong_closure_random_summary": str(OUT_DIR / "cft2_wrong_closure_random_summary.json"),
            "verdict": str(OUT_DIR / "cft2_verdict.json"),
        },
        "top_rows": model_summary.head(15).to_dict(orient="records"),
    }

    model_summary.to_csv(OUT_DIR / "cft2_model_summary.csv", index=False, encoding="utf-8-sig")
    with open(OUT_DIR / "cft2_same_answer_summary.json", "w", encoding="utf-8") as f:
        json.dump(same_ans, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / "cft2_wrong_closure_random_summary.json", "w", encoding="utf-8") as f:
        json.dump(wrong_random, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / "cft2_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict_obj, f, ensure_ascii=False, indent=2)

    print(json.dumps(verdict_obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
