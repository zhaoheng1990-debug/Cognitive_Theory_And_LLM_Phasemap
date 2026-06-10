# -*- coding: utf-8 -*-
"""
DYN-2: Convergence Commitment Audit

Purpose
-------
Reuses TCD-1 features to audit which dynamics variables actually predict final
commitment / GenError, after the MPR series showed:

    TopK      -> relation topology / coarse mechanism
    MPR       -> submechanism / binding-priority
    Dynamics  -> still under-specified for final commitment

Core questions:
1. Are trajectory-shape / commitment-dynamics variables stronger than TopK for GenError?
2. Do derived commitment variables improve over raw answer margin trajectory?
3. Does answer-orthogonal dynamics still carry GenError / commitment signal?
4. Which layer segment (L20-22 vs L23-25) carries the useful commitment signal?

Default paths are written for the user's local Windows setup.
"""

import os
import json
import math
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, r2_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
TCD1_OUT_DIR = r"C:\Users\ZH\Desktop\AGI\outputs\tcd1_outputs"
OUT_DIR = r"C:\Users\ZH\Desktop\AGI\outputs\dyn2_outputs"
FEATURES_PATH = os.path.join(TCD1_OUT_DIR, "tcd1_features.csv")
RANDOM_SEED = 42
N_SPLITS = 5
EPS = 1e-9


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_json_dump(obj, path: str) -> None:
    def convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return v if math.isfinite(v) else None
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, float):
            return o if math.isfinite(o) else None
        return str(o)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=convert)


def numeric_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().sum() > 0:
            out.append(c)
    return out


def fill_X(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    X = df[cols].copy().replace([np.inf, -np.inf], np.nan)
    for c in cols:
        med = X[c].median(skipna=True)
        if not np.isfinite(med):
            med = 0.0
        X[c] = X[c].fillna(med)
    return X.values.astype(np.float64)


def make_clf():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, solver="lbfgs", class_weight="balanced", random_state=RANDOM_SEED),
    )


def make_reg():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=RANDOM_SEED))


def cv_splits(df: pd.DataFrame, y=None, group_col: str = "graph_id"):
    groups = df[group_col].astype(str).values if group_col in df.columns else np.arange(len(df))
    n_groups = len(np.unique(groups))
    if n_groups >= 2:
        n_splits = min(N_SPLITS, n_groups)
        return GroupKFold(n_splits=n_splits).split(np.zeros(len(df)), y, groups)
    if y is not None:
        counts = np.bincount(np.asarray(y, dtype=int)) if len(np.unique(y)) > 1 else [len(y)]
        n_splits = max(2, min(N_SPLITS, min(counts)))
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED).split(np.zeros(len(df)), y)
    return GroupKFold(n_splits=2).split(np.zeros(len(df)), None, np.arange(len(df)))


def eval_binary(df: pd.DataFrame, cols: List[str], target: str = "gen_error", group_col: str = "graph_id") -> Dict:
    if target not in df.columns or not cols:
        return {"valid": False, "reason": "missing_target_or_features"}
    y = df[target].astype(int).values
    if len(np.unique(y)) < 2:
        return {"valid": False, "reason": "single_class"}
    X = fill_X(df, cols)
    prob = np.full(len(df), np.nan)
    pred = np.full(len(df), -1)
    for tr, te in cv_splits(df, y, group_col):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_clf()
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        prob[te] = p
        pred[te] = (p >= 0.5).astype(int)
    mask = np.isfinite(prob) & (pred >= 0)
    if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
        return {"valid": False, "reason": "no_valid_folds"}
    return {
        "valid": True,
        "auc": float(roc_auc_score(y[mask], prob[mask])),
        "acc": float(accuracy_score(y[mask], pred[mask])),
        "bal_acc": float(balanced_accuracy_score(y[mask], pred[mask])),
        "f1": float(f1_score(y[mask], pred[mask], zero_division=0)),
        "n": int(mask.sum()),
    }


def eval_multiclass(df: pd.DataFrame, cols: List[str], target: str, group_col: str = "graph_id") -> Dict:
    if target not in df.columns or not cols:
        return {"valid": False, "reason": "missing_target_or_features"}
    y_raw = df[target].astype(str).values
    if len(np.unique(y_raw)) < 2:
        return {"valid": False, "reason": "single_class"}
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    X = fill_X(df, cols)
    pred = np.full(len(df), -1)
    for tr, te in cv_splits(df, y, group_col):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_clf()
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    mask = pred >= 0
    if mask.sum() == 0:
        return {"valid": False, "reason": "no_valid_folds"}
    return {
        "valid": True,
        "acc": float(accuracy_score(y[mask], pred[mask])),
        "bal_acc": float(balanced_accuracy_score(y[mask], pred[mask])),
        "macro_f1": float(f1_score(y[mask], pred[mask], average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y[mask], pred[mask], average="weighted", zero_division=0)),
        "n": int(mask.sum()),
        "classes": list(le.classes_),
    }


def eval_regression(df: pd.DataFrame, cols: List[str], target: str, group_col: str = "graph_id") -> Dict:
    if target not in df.columns or not cols:
        return {"valid": False, "reason": "missing_target_or_features"}
    y = df[target].astype(float).values
    X = fill_X(df, cols)
    pred = np.full(len(df), np.nan)
    for tr, te in cv_splits(df, None, group_col):
        reg = make_reg()
        reg.fit(X[tr], y[tr])
        pred[te] = reg.predict(X[te])
    mask = np.isfinite(pred)
    if mask.sum() < 10:
        return {"valid": False, "reason": "no_valid_folds"}
    corr = np.corrcoef(y[mask], pred[mask])[0, 1] if np.std(pred[mask]) > EPS and np.std(y[mask]) > EPS else np.nan
    return {
        "valid": True,
        "r2": float(r2_score(y[mask], pred[mask])),
        "corr": float(corr) if np.isfinite(corr) else None,
        "mae": float(np.mean(np.abs(y[mask] - pred[mask]))),
        "n": int(mask.sum()),
    }


def residualize_against(df: pd.DataFrame, cols: List[str], controls: List[str]) -> pd.DataFrame:
    """Return a copy with feature columns residualized against controls."""
    out = df.copy()
    cols = numeric_cols(out, cols)
    controls = numeric_cols(out, controls)
    if not cols or not controls:
        return out
    C = fill_X(out, controls)
    C = np.column_stack([np.ones(len(out)), C])
    for c in cols:
        y = out[c].astype(float).replace([np.inf, -np.inf], np.nan)
        med = y.median(skipna=True)
        if not np.isfinite(med):
            med = 0.0
        yv = y.fillna(med).values.astype(np.float64)
        try:
            beta, *_ = np.linalg.lstsq(C, yv, rcond=None)
            resid = yv - C @ beta
            out[c] = resid
        except Exception:
            pass
    return out


def add_dyn2_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    layers = [20, 21, 22, 23, 24, 25]
    mcols = [f"ans_margin_L{l}" for l in layers if f"ans_margin_L{l}" in out.columns]
    if len(mcols) >= 2:
        M = out[mcols].values.astype(np.float64)
        for i, c in enumerate(mcols):
            out[f"commit_abs_{c}"] = np.abs(M[:, i])
        early_idx = [i for i, c in enumerate(mcols) if any(s in c for s in ["20", "21", "22"])]
        late_idx = [i for i, c in enumerate(mcols) if any(s in c for s in ["23", "24", "25"])]
        if early_idx and late_idx:
            early = M[:, early_idx]
            late = M[:, late_idx]
            out["dyn2_abs_early_mean"] = np.mean(np.abs(early), axis=1)
            out["dyn2_abs_late_mean"] = np.mean(np.abs(late), axis=1)
            out["dyn2_commitment_gain_late_minus_early"] = out["dyn2_abs_late_mean"] - out["dyn2_abs_early_mean"]
            out["dyn2_signed_late_mean"] = np.mean(late, axis=1)
            out["dyn2_signed_early_mean"] = np.mean(early, axis=1)
            out["dyn2_signed_shift_late_minus_early"] = out["dyn2_signed_late_mean"] - out["dyn2_signed_early_mean"]
        dM = np.diff(M, axis=1)
        out["dyn2_velocity_mean"] = np.mean(dM, axis=1)
        out["dyn2_velocity_abs_mean"] = np.mean(np.abs(dM), axis=1)
        out["dyn2_velocity_std"] = np.std(dM, axis=1)
        if dM.shape[1] >= 2:
            ddM = np.diff(dM, axis=1)
            out["dyn2_accel_abs_mean"] = np.mean(np.abs(ddM), axis=1)
            out["dyn2_accel_std"] = np.std(ddM, axis=1)
        signs = np.sign(M)
        out["dyn2_sign_changes"] = np.sum(signs[:, 1:] != signs[:, :-1], axis=1)
        out["dyn2_final_proxy_abs_L25"] = np.abs(M[:, -1])
        out["dyn2_start_abs_L20"] = np.abs(M[:, 0])
        out["dyn2_commitment_velocity_L20_to_L25"] = np.abs(M[:, -1]) - np.abs(M[:, 0])
        out["dyn2_min_abs_20_25"] = np.min(np.abs(M), axis=1)
        out["dyn2_mean_abs_20_25"] = np.mean(np.abs(M), axis=1)
        out["dyn2_std_abs_20_25"] = np.std(np.abs(M), axis=1)
        # margin area with local trapezoid, no np.trapz dependency
        out["dyn2_signed_area_20_25"] = np.sum((M[:, :-1] + M[:, 1:]) * 0.5, axis=1)
        out["dyn2_abs_area_20_25"] = np.sum((np.abs(M[:, :-1]) + np.abs(M[:, 1:])) * 0.5, axis=1)

    # Commitment confidence target proxies
    if "final_margin_C_minus_E" in out.columns:
        out["commit_abs_final_margin"] = np.abs(out["final_margin_C_minus_E"].astype(float))
        med = out["commit_abs_final_margin"].median()
        out["commit_high_abs_margin"] = (out["commit_abs_final_margin"] >= med).astype(int)
    if "pred_answer" in out.columns:
        # In this synthetic setup Red/Blue are the two continuations. Keep as label target.
        out["pred_answer_binary"] = (out["pred_answer"].astype(str).str.lower().str.contains("blue")).astype(int)
    return out


def infer_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    non_feature = {
        "row_id", "prompt", "answer", "gen", "gen_label", "mechanism", "mechanism_group", "submechanism",
        "condition", "graph_id", "surface_id", "pair_id", "topic", "relation", "target", "clean_answer",
        "conflict_answer", "label_C", "label_E", "is_error", "gen_error", "pred_answer", "pred_answer_binary",
        "commit_high_abs_margin",
    }
    all_num = [c for c in df.columns if c not in non_feature and pd.api.types.is_numeric_dtype(df[c])]
    def has(c, keys):
        cl = c.lower()
        return any(k in cl for k in keys)

    topk = [c for c in all_num if has(c, ["topk", "vim", "jaccard", "center", "spread", "entropy", "gap", "maxlogit"])]
    answer = [c for c in all_num if has(c, ["ans_margin", "final_margin", "answer_margin", "rank", "logit"])]
    dyn_raw = [c for c in all_num if has(c, ["dyn_", "dyn2_", "dmargin", "curvature", "boundary", "hshape", "delta", "db", "traj", "residual", "commit_"])]
    # remove topk from dynamics; keep answer-derived dynamics as a separate commitment family
    dyn = [c for c in dyn_raw if c not in set(topk)]
    dyn_no_direct_final = [c for c in dyn if "final_margin" not in c.lower() and "commit_abs_final" not in c.lower()]
    binding = [c for c in all_num if has(c, ["closure", "logic", "source", "trusted", "priority", "binding", "rule", "chain", "consistency", "probe", "contrad"])]
    binding = [c for c in binding if c not in set(topk) and c not in set(answer)]

    groups = {
        "TopK": topk,
        "AnswerMargin": answer,
        "DynamicsRaw": dyn,
        "DynamicsNoFinal": dyn_no_direct_final,
        "BindingPriority": binding,
        "TopK_plus_Dynamics": sorted(set(topk + dyn_no_direct_final)),
        "Binding_plus_Dynamics": sorted(set(binding + dyn_no_direct_final)),
        "MultiProjection": sorted(set(topk + binding + dyn_no_direct_final)),
        "CommitmentDerivedOnly": [c for c in dyn_no_direct_final if c.startswith("dyn2_") or c.startswith("commit_abs_ans_margin")],
    }
    return {k: numeric_cols(df, v) for k, v in groups.items() if len(numeric_cols(df, v)) > 0}


def evaluate_all(df: pd.DataFrame, groups: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    for name, cols in groups.items():
        ge = eval_binary(df, cols, "gen_error") if "gen_error" in df.columns else {"valid": False}
        ans = eval_binary(df, cols, "pred_answer_binary") if "pred_answer_binary" in df.columns else {"valid": False}
        ch = eval_binary(df, cols, "commit_high_abs_margin") if "commit_high_abs_margin" in df.columns else {"valid": False}
        mech = eval_multiclass(df, cols, "mechanism") if "mechanism" in df.columns else {"valid": False}
        sub = eval_multiclass(df, cols, "submechanism") if "submechanism" in df.columns else {"valid": False}
        reg = eval_regression(df, cols, "commit_abs_final_margin") if "commit_abs_final_margin" in df.columns else {"valid": False}
        rows.append({
            "feature_group": name,
            "n_features": len(cols),
            "gen_auc": ge.get("auc"),
            "gen_f1": ge.get("f1"),
            "answer_auc": ans.get("auc"),
            "answer_f1": ans.get("f1"),
            "commit_high_auc": ch.get("auc"),
            "commit_high_f1": ch.get("f1"),
            "commit_abs_r2": reg.get("r2"),
            "commit_abs_corr": reg.get("corr"),
            "mechanism_macro_f1": mech.get("macro_f1"),
            "submechanism_macro_f1": sub.get("macro_f1"),
            "valid_gen": ge.get("valid", False),
            "valid_commit_reg": reg.get("valid", False),
        })
    return pd.DataFrame(rows)


def answer_orthogonal_evaluation(df: pd.DataFrame, groups: Dict[str, List[str]]) -> pd.DataFrame:
    controls = [c for c in ["final_margin_C_minus_E", "ans_margin_main_mean", "ans_margin_L25", "ans_margin_L24"] if c in df.columns]
    if not controls:
        return pd.DataFrame([])
    rows = []
    for name, cols in groups.items():
        if name == "AnswerMargin":
            continue
        rdf = residualize_against(df, cols, controls)
        ge = eval_binary(rdf, cols, "gen_error") if "gen_error" in rdf.columns else {"valid": False}
        sub = eval_multiclass(rdf, cols, "submechanism") if "submechanism" in rdf.columns else {"valid": False}
        mech = eval_multiclass(rdf, cols, "mechanism") if "mechanism" in rdf.columns else {"valid": False}
        rows.append({
            "feature_group": name + "__answer_orthogonal",
            "n_features": len(cols),
            "gen_auc": ge.get("auc"),
            "gen_f1": ge.get("f1"),
            "mechanism_macro_f1": mech.get("macro_f1"),
            "submechanism_macro_f1": sub.get("macro_f1"),
        })
    return pd.DataFrame(rows)


def segment_ablation(df: pd.DataFrame) -> pd.DataFrame:
    """Compare early/boundary/late commitment trajectory segments if layer columns exist."""
    segs = {
        "L20_22_margin": [c for c in ["ans_margin_L20", "ans_margin_L21", "ans_margin_L22"] if c in df.columns],
        "L23_25_margin": [c for c in ["ans_margin_L23", "ans_margin_L24", "ans_margin_L25"] if c in df.columns],
        "L20_25_margin": [c for c in [f"ans_margin_L{i}" for i in range(20, 26)] if c in df.columns],
        "Dyn2_commitment": [c for c in df.columns if c.startswith("dyn2_")],
    }
    rows = []
    for name, cols in segs.items():
        cols = numeric_cols(df, cols)
        if not cols:
            continue
        ge = eval_binary(df, cols, "gen_error")
        ch = eval_binary(df, cols, "commit_high_abs_margin") if "commit_high_abs_margin" in df.columns else {"valid": False}
        reg = eval_regression(df, cols, "commit_abs_final_margin") if "commit_abs_final_margin" in df.columns else {"valid": False}
        rows.append({
            "segment": name,
            "n_features": len(cols),
            "gen_auc": ge.get("auc"),
            "gen_f1": ge.get("f1"),
            "commit_high_auc": ch.get("auc"),
            "commit_abs_r2": reg.get("r2"),
            "commit_abs_corr": reg.get("corr"),
        })
    return pd.DataFrame(rows)


def make_verdict(model_df: pd.DataFrame, orth_df: pd.DataFrame, seg_df: pd.DataFrame) -> Dict:
    def get(group, metric):
        row = model_df[model_df["feature_group"] == group]
        if row.empty:
            return None
        v = row.iloc[0].get(metric)
        return float(v) if pd.notna(v) else None
    topk_gen = get("TopK", "gen_auc")
    dyn_gen = get("DynamicsNoFinal", "gen_auc") or get("DynamicsRaw", "gen_auc")
    mp_gen = get("MultiProjection", "gen_auc")
    ans_gen = get("AnswerMargin", "gen_auc")
    commit_r2_dyn = get("DynamicsNoFinal", "commit_abs_r2") or get("DynamicsRaw", "commit_abs_r2")
    commit_r2_mp = get("MultiProjection", "commit_abs_r2")

    reasons = []
    verdict = "PARTIAL_DYNAMICS_SIGNAL"
    if dyn_gen is not None and topk_gen is not None and dyn_gen > topk_gen + 0.02:
        reasons.append("DynamicsNoFinal predicts GenError stronger than TopK.")
    elif dyn_gen is not None and topk_gen is not None:
        reasons.append("DynamicsNoFinal does not exceed TopK for GenError; commitment dynamics still needs stronger observables.")
    if mp_gen is not None and topk_gen is not None and mp_gen > topk_gen + 0.02:
        reasons.append("MultiProjection improves GenError over TopK.")
    if commit_r2_dyn is not None and commit_r2_dyn > 0.3:
        reasons.append("Dynamics features predict final commitment magnitude with meaningful R2.")
    if commit_r2_mp is not None and commit_r2_dyn is not None and commit_r2_mp > commit_r2_dyn + 0.05:
        reasons.append("MultiProjection improves commitment magnitude prediction over dynamics alone.")

    # verdict ladder
    if dyn_gen is not None and topk_gen is not None and dyn_gen > topk_gen + 0.05 and commit_r2_dyn is not None and commit_r2_dyn > 0.5:
        verdict = "PASS_DYNAMICS_COMMITMENT_STRONG"
    elif mp_gen is not None and topk_gen is not None and mp_gen > topk_gen + 0.02:
        verdict = "PASS_MPR_IMPROVES_COMMITMENT_BUT_DYNAMICS_WEAK"
    elif commit_r2_dyn is not None and commit_r2_dyn > 0.3:
        verdict = "PASS_DYNAMICS_COMMITMENT_MAGNITUDE_ONLY"
    else:
        verdict = "PARTIAL_TOPOLOGY_DOMINANT_DYNAMICS_WEAK"

    return {
        "verdict": verdict,
        "reasons": reasons,
        "metrics": {
            "topk_gen_auc": topk_gen,
            "dynamics_gen_auc": dyn_gen,
            "multiprojection_gen_auc": mp_gen,
            "answer_margin_gen_auc": ans_gen,
            "dynamics_commit_abs_r2": commit_r2_dyn,
            "multiprojection_commit_abs_r2": commit_r2_mp,
        },
    }


def main():
    ensure_dir(OUT_DIR)
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Missing features file: {FEATURES_PATH}")
    df = pd.read_csv(FEATURES_PATH)
    if df.empty:
        verdict = {"verdict": "FAIL_EMPTY_FEATURES", "input_features": FEATURES_PATH}
        safe_json_dump(verdict, os.path.join(OUT_DIR, "dyn2_verdict.json"))
        return
    df = add_dyn2_features(df)
    groups = infer_groups(df)
    if not groups:
        verdict = {"verdict": "FAIL_NO_FEATURE_GROUPS", "columns": list(df.columns)}
        safe_json_dump(verdict, os.path.join(OUT_DIR, "dyn2_verdict.json"))
        return

    model_df = evaluate_all(df, groups)
    orth_df = answer_orthogonal_evaluation(df, groups)
    seg_df = segment_ablation(df)

    if not orth_df.empty:
        full_model_df = pd.concat([model_df, orth_df], ignore_index=True)
    else:
        full_model_df = model_df
    sort_cols = [c for c in ["gen_auc", "commit_abs_r2", "submechanism_macro_f1"] if c in full_model_df.columns]
    if sort_cols:
        full_model_df = full_model_df.sort_values(sort_cols, ascending=[False] * len(sort_cols))

    model_path = os.path.join(OUT_DIR, "dyn2_model_summary.csv")
    seg_path = os.path.join(OUT_DIR, "dyn2_segment_ablation.csv")
    feat_path = os.path.join(OUT_DIR, "dyn2_features_augmented.csv")
    full_model_df.to_csv(model_path, index=False, encoding="utf-8-sig")
    seg_df.to_csv(seg_path, index=False, encoding="utf-8-sig")
    df.to_csv(feat_path, index=False, encoding="utf-8-sig")

    verdict = make_verdict(model_df, orth_df, seg_df)
    verdict.update({
        "n_rows": int(len(df)),
        "n_graphs": int(df["graph_id"].nunique()) if "graph_id" in df.columns else None,
        "feature_set_sizes": {k: len(v) for k, v in groups.items()},
        "input_features": FEATURES_PATH,
        "outputs": {
            "model_summary": model_path,
            "segment_ablation": seg_path,
            "features_augmented": feat_path,
        },
    })
    # Include compact best rows
    if not full_model_df.empty:
        verdict["top_rows"] = full_model_df.head(10).to_dict(orient="records")
    if not seg_df.empty:
        verdict["segment_rows"] = seg_df.to_dict(orient="records")

    safe_json_dump(verdict, os.path.join(OUT_DIR, "dyn2_verdict.json"))
    print(json.dumps(verdict, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
