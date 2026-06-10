# -*- coding: utf-8 -*-
r"""
MPR-1B: Same-Answer Internal Structure Audit

Purpose
-------
This script verifies the projection-residual claim:

    H(S | X) > 0

where X is the final answer class and S is the internal structural state
(coarse mechanism / submechanism).  It reuses the feature file from TCD-1 / MPR-1
and evaluates whether internal mechanism/submechanism can still be recovered
*within samples that share the same final answer*.

Default input:
    C:\Users\ZH\Desktop\AGI\outputs\tcd1_outputs\tcd1_features.csv

Default output:
    C:\Users\ZH\Desktop\AGI\outputs\mpr1b_outputs

No command-line args are required.
"""

import os
import json
import math
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix

warnings.filterwarnings("ignore")

# =========================
# Hard-coded local paths
# =========================
INPUT_FEATURES = r"C:\Users\ZH\Desktop\AGI\outputs\tcd1_outputs\tcd1_features.csv"
OUT_DIR = r"C:\Users\ZH\Desktop\AGI\outputs\mpr1b_outputs"

RANDOM_SEED = 42
N_SPLITS = 5
MIN_CLASS_COUNT = 8
MIN_ROWS_PER_ANSWER = 40

META_COLS = {
    "row_id", "graph_id", "pair_id", "surface_id", "condition", "mechanism", "submechanism",
    "target", "gen_answer", "gen_error", "prompt", "clean_pair_id", "answer_class",
    "cond_a", "cond_b", "mech_a", "mech_b", "submech_a", "submech_b"
}

# =========================
# Utility
# =========================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def infer_answer_class(df: pd.DataFrame) -> pd.Series:
    """Infer final answer group for same-answer split."""
    # Preferred if available.
    for col in ["answer_class", "gen_answer", "prediction", "generated_answer", "gen_label"]:
        if col in df.columns:
            s = df[col].astype(str).str.strip()
            # normalize variants
            def norm(v: str) -> str:
                v_up = v.upper()
                if v_up in {"C", "CLEAN", "C_ANSWER", "TARGET_C"}:
                    return "C"
                if v_up in {"E", "ERROR", "CONFLICT", "TARGET_E"}:
                    return "E"
                if v_up in {"OTHER", "UNKNOWN", "NA", "NAN", "NONE", ""}:
                    return "OTHER"
                if "C" == v_up[:1] and "E" not in v_up[:1]:
                    return "C"
                if "E" == v_up[:1]:
                    return "E"
                return "OTHER"
            return s.map(norm)

    # Fall back to gen_error if present: assumes error=>E, non-error=>C.
    if "gen_error" in df.columns:
        return df["gen_error"].apply(lambda x: "E" if safe_float(x) >= 0.5 else "C")

    # Fall back to answer margin, if columns exist.
    margin_cols = [c for c in df.columns if "answer_margin" in c.lower() or "margin" in c.lower()]
    if margin_cols:
        col = margin_cols[0]
        return df[col].apply(lambda x: "C" if safe_float(x) >= 0 else "E")

    return pd.Series(["ALL"] * len(df), index=df.index)


def feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    numeric_cols = []
    for c in df.columns:
        if c in META_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)

    def pick(preds):
        out = []
        for c in numeric_cols:
            lc = c.lower()
            if any(p in lc for p in preds):
                out.append(c)
        return sorted(set(out))

    groups = {}
    groups["TopK"] = pick(["topk", "vim", "center", "spread", "entropy", "jaccard"])
    groups["Dynamics"] = pick(["delta", "slope", "area", "curv", "db_", "boundary", "hshape", "residual", "velocity", "accel", "margin_diff"])
    groups["AnswerMargin"] = pick(["answer", "margin", "rankgap", "support", "logit"])
    groups["NonTopK"] = [c for c in numeric_cols if c not in groups["TopK"]]
    groups["MultiProjection"] = numeric_cols

    # Answer-orthogonal variants are computed later with residualization.
    return {k: v for k, v in groups.items() if len(v) > 0}


def residualize_against_answer(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Residualize cols against the strongest available answer/margin columns."""
    ans_cols = []
    for c in df.columns:
        lc = c.lower()
        if c in cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and any(p in lc for p in ["answer", "margin", "rankgap", "support", "logit"]):
            ans_cols.append(c)
    ans_cols = sorted(set(ans_cols))[:20]

    out = df.copy()
    if not ans_cols:
        return out, cols

    A = out[ans_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    A = np.column_stack([np.ones(len(out)), A])
    new_cols = []
    for c in cols:
        y = out[c].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
        try:
            beta = np.linalg.lstsq(A, y, rcond=None)[0]
            resid = y - A.dot(beta)
        except Exception:
            resid = y
        nc = c + "__ans_orth"
        out[nc] = resid
        new_cols.append(nc)
    return out, new_cols


def make_classifier() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, solver="lbfgs", class_weight="balanced")),
    ])


def enough_classes(y: np.ndarray) -> bool:
    vals, counts = np.unique(y, return_counts=True)
    return len(vals) >= 2 and counts.min() >= MIN_CLASS_COUNT


def eval_group_cv(df: pd.DataFrame, cols: List[str], target: str, group_col: str = "graph_id") -> Optional[Dict]:
    sub = df.dropna(subset=[target]).copy()
    if len(sub) < 50:
        return None
    y_raw = sub[target].astype(str).values
    if not enough_classes(y_raw):
        return None
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    X = sub[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    groups = sub[group_col].astype(str).values if group_col in sub.columns else np.arange(len(sub))
    uniq_groups = np.unique(groups)
    n_splits = min(N_SPLITS, len(uniq_groups))
    if n_splits < 2:
        return None

    preds = np.full_like(y, fill_value=-1)
    used = 0
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups=groups):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_classifier()
        try:
            clf.fit(X[tr], y[tr])
            preds[te] = clf.predict(X[te])
            used += 1
        except Exception:
            continue
    mask = preds >= 0
    if mask.sum() < 20 or used == 0:
        return None
    labels_present = np.unique(y[mask])
    return {
        "n": int(mask.sum()),
        "classes": list(le.classes_),
        "n_classes": int(len(le.classes_)),
        "cv_used": int(used),
        "acc": float(accuracy_score(y[mask], preds[mask])),
        "balanced_acc": float(balanced_accuracy_score(y[mask], preds[mask])),
        "macro_f1": float(f1_score(y[mask], preds[mask], average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y[mask], preds[mask], average="weighted", zero_division=0)),
    }


def eval_within_answer(df: pd.DataFrame, cols: List[str], target: str, answer_value: str, group_col: str = "graph_id") -> Optional[Dict]:
    sub = df[df["answer_class"] == answer_value].copy()
    if len(sub) < MIN_ROWS_PER_ANSWER:
        return None
    res = eval_group_cv(sub, cols, target=target, group_col=group_col)
    if res is None:
        return None
    res["answer"] = answer_value
    return res


def summarize_same_answer(df: pd.DataFrame, groups: Dict[str, List[str]], target: str) -> Tuple[pd.DataFrame, Dict]:
    rows = []
    details = []
    answers = sorted(df["answer_class"].astype(str).unique())
    for gname, cols in groups.items():
        for ans in answers:
            res = eval_within_answer(df, cols, target, ans)
            if res is None:
                continue
            row = {"feature_group": gname, "target": target, **res}
            rows.append(row)
            details.append(row)
    out_df = pd.DataFrame(rows)
    summary = {}
    if not out_df.empty:
        agg = out_df.groupby("feature_group").agg(
            same_answer_macro_f1_mean=("macro_f1", "mean"),
            same_answer_bal_acc_mean=("balanced_acc", "mean"),
            same_answer_acc_mean=("acc", "mean"),
            n_answers=("answer", "nunique"),
        ).reset_index().sort_values("same_answer_macro_f1_mean", ascending=False)
        summary["aggregate"] = agg.to_dict(orient="records")
    else:
        summary["aggregate"] = []
    summary["details"] = details
    return out_df, summary


def build_verdict(summary: Dict) -> Dict:
    agg = summary.get("aggregate", [])
    if not agg:
        return {"verdict": "FAIL_NO_SAME_ANSWER_RESULTS", "reasons": ["No answer group had enough class diversity for same-answer split."]}
    # Create dict for common groups
    by_group = {r["feature_group"]: r for r in agg}
    top = agg[0]
    mp = by_group.get("MultiProjection")
    topk = by_group.get("TopK")
    non = by_group.get("NonTopK")
    dyn = by_group.get("Dynamics")

    reasons = []
    verdict = "PARTIAL_SAME_ANSWER_STRUCTURE"

    if top["same_answer_macro_f1_mean"] >= 0.50:
        reasons.append("Same-answer internal structure is recoverable above weak threshold, supporting H(S|X)>0.")
        verdict = "PASS_SAME_ANSWER_STRUCTURE"
    if mp and topk and mp["same_answer_macro_f1_mean"] > topk["same_answer_macro_f1_mean"] + 0.03:
        reasons.append("MultiProjection improves same-answer structure recovery over TopK-only.")
        verdict = "PASS_SAME_ANSWER_MPR_GAIN"
    if topk and topk["same_answer_macro_f1_mean"] >= 0.50:
        reasons.append("TopK alone retains nontrivial internal structure even within same final answer.")
    if top["same_answer_macro_f1_mean"] < 0.35:
        verdict = "FAIL_WEAK_PROJECTION_RESIDUAL"
        reasons.append("Same-answer structure recovery is weak; H(S|X)>0 not supported by this feature set.")

    return {
        "verdict": verdict,
        "reasons": reasons,
        "best_group": top,
        "multiprojection": mp,
        "topk": topk,
        "non_topk": non,
        "dynamics": dyn,
    }


def main():
    ensure_dir(OUT_DIR)
    if not os.path.exists(INPUT_FEATURES):
        verdict = {"verdict": "FAIL_INPUT_NOT_FOUND", "input": INPUT_FEATURES}
        with open(os.path.join(OUT_DIR, "mpr1b_verdict.json"), "w", encoding="utf-8") as f:
            json.dump(verdict, f, ensure_ascii=False, indent=2)
        print(json.dumps(verdict, ensure_ascii=False, indent=2))
        return

    df = pd.read_csv(INPUT_FEATURES, encoding="utf-8-sig")
    if df.empty:
        verdict = {"verdict": "FAIL_EMPTY_FEATURES", "input": INPUT_FEATURES}
        with open(os.path.join(OUT_DIR, "mpr1b_verdict.json"), "w", encoding="utf-8") as f:
            json.dump(verdict, f, ensure_ascii=False, indent=2)
        print(json.dumps(verdict, ensure_ascii=False, indent=2))
        return

    df["answer_class"] = infer_answer_class(df)

    groups = feature_groups(df)
    # Keep compact high-level groups.
    compact = {}
    if "TopK" in groups:
        compact["TopK"] = groups["TopK"]
    if "Dynamics" in groups:
        compact["Dynamics"] = groups["Dynamics"]
    if "NonTopK" in groups:
        compact["NonTopK"] = groups["NonTopK"]
    if "MultiProjection" in groups:
        compact["MultiProjection"] = groups["MultiProjection"]

    # Add answer-orthogonal variants for TopK and MultiProjection.
    df_aug = df.copy()
    if "TopK" in compact:
        df_aug, topk_orth = residualize_against_answer(df_aug, compact["TopK"])
        compact["TopK__answer_orthogonal"] = topk_orth
    if "MultiProjection" in compact:
        df_aug, mp_orth = residualize_against_answer(df_aug, compact["MultiProjection"])
        compact["MultiProjection__answer_orthogonal"] = mp_orth

    # Same-answer audits for mechanism and submechanism.
    all_rows = []
    all_summaries = {}
    for target in ["mechanism", "submechanism"]:
        if target not in df_aug.columns:
            continue
        detail_df, summary = summarize_same_answer(df_aug, compact, target=target)
        all_summaries[target] = summary
        if not detail_df.empty:
            all_rows.append(detail_df)

    if all_rows:
        detail_all = pd.concat(all_rows, ignore_index=True)
    else:
        detail_all = pd.DataFrame()

    detail_path = os.path.join(OUT_DIR, "mpr1b_same_answer_detail.csv")
    detail_all.to_csv(detail_path, index=False, encoding="utf-8-sig")

    summary_path = os.path.join(OUT_DIR, "mpr1b_same_answer_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    # Verdict mainly on submechanism if available, else mechanism.
    verdict_target = "submechanism" if "submechanism" in all_summaries else "mechanism"
    verdict = build_verdict(all_summaries.get(verdict_target, {}))
    verdict.update({
        "target_used_for_verdict": verdict_target,
        "n_rows": int(len(df_aug)),
        "answer_counts": df_aug["answer_class"].value_counts().to_dict(),
        "feature_set_sizes": {k: len(v) for k, v in compact.items()},
        "input_features": INPUT_FEATURES,
    })

    verdict_path = os.path.join(OUT_DIR, "mpr1b_verdict.json")
    with open(verdict_path, "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)

    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    print("\nSaved:")
    print(" -", detail_path)
    print(" -", summary_path)
    print(" -", verdict_path)


if __name__ == "__main__":
    main()
