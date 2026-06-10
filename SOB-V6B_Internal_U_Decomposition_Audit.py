# -*- coding: utf-8 -*-
r"""
SOB-V6B: Internal U Decomposition Audit

This script reuses SOB-V6 internal_u_table outputs and decomposes U_internal into:

    B_bind     = task probe - current probe
    R_route    = USE/PARTIAL/ANALOGY route probes - REJECT route probe
    B_boundary = REJECT route probe vs use-route average

Why reuse V6 outputs?
---------------------
SOB-V6 already completed the expensive model forward pass. V6B is a measurement-model
audit, not a new forward experiment. It should test whether specialized internal
components explain specialized targets better than one large AllInternal feature bag.

Input
-----
C:\Users\ZH\Desktop\AGI\outputs\SOB-V6\<model>\SOB-V6_<model>_internal_u_table.csv

Output
------
C:\Users\ZH\Desktop\AGI\outputs\SOB-V6B

No CLI args.
"""

import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, r2_score, accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

SCRIPT_VERSION = "SOB-V6B_REUSE_V6_OUTPUTS_DECOMPOSITION_v1"
EXPERIMENT_ID = "SOB-V6B"
IN_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\SOB-V6")
OUT_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\SOB-V6B")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS = ["qwen", "llama", "gemma"]
N_SPLITS = 5
RANDOM_SEED = 42


def now_time():
    return time.strftime("%Y-%m-%d %H:%M:%S")


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
    return [c for c in df.columns if any(c.startswith(p) for p in prefixes) and pd.api.types.is_numeric_dtype(df[c])]


def cv_splits(df, y=None, group_col="object_key", stratify=False):
    n = len(df)
    if n < 2:
        return []
    if group_col in df.columns and df[group_col].nunique() >= min(N_SPLITS, n):
        ns = min(N_SPLITS, df[group_col].nunique())
        cv = GroupKFold(n_splits=ns)
        return list(cv.split(np.zeros(n), y if y is not None else np.zeros(n), groups=df[group_col].values))
    if stratify and y is not None and len(np.unique(y)) > 1:
        vals, counts = np.unique(y, return_counts=True)
        ns = min(N_SPLITS, int(counts.min()))
        if ns >= 2:
            cv = StratifiedKFold(n_splits=ns, shuffle=True, random_state=RANDOM_SEED)
            return list(cv.split(np.zeros(n), y))
    ns = min(N_SPLITS, n)
    cv = KFold(n_splits=ns, shuffle=True, random_state=RANDOM_SEED)
    return list(cv.split(np.zeros(n)))


def reg_cv(df, cols, target):
    cols = numeric_cols(df, cols)
    d = df.dropna(subset=[target]).copy() if target in df.columns else pd.DataFrame()
    if len(d) < 12 or not cols:
        return {"r2": np.nan, "corr": np.nan, "n": len(d), "n_features": len(cols)}
    X = d[cols].replace([np.inf, -np.inf], np.nan).values
    y = d[target].astype(float).values
    pred = np.zeros(len(d), dtype=float)
    used = 0
    for tr, te in cv_splits(d, y, group_col="object_key", stratify=False):
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=10.0)),
        ])
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
        used += 1
    if used == 0:
        return {"r2": np.nan, "corr": np.nan, "n": len(d), "n_features": len(cols)}
    corr = float(np.corrcoef(pred, y)[0, 1]) if np.std(pred) > 1e-12 and np.std(y) > 1e-12 else np.nan
    return {"r2": safe_float(r2_score(y, pred)), "corr": corr, "n": len(d), "n_features": len(cols)}


def clf_cv(df, cols, target):
    cols = numeric_cols(df, cols)
    d = df.dropna(subset=[target]).copy() if target in df.columns else pd.DataFrame()
    if len(d) < 12 or not cols:
        return {"auc": np.nan, "acc": np.nan, "f1": np.nan, "n": len(d), "n_features": len(cols)}
    y = d[target].astype(int).values
    if len(np.unique(y)) < 2:
        return {"auc": np.nan, "acc": np.nan, "f1": np.nan, "n": len(d), "n_features": len(cols)}
    X = d[cols].replace([np.inf, -np.inf], np.nan).values
    prob = np.zeros(len(d), dtype=float)
    pred = np.zeros(len(d), dtype=int)
    used = 0
    for tr, te in cv_splits(d, y, group_col="object_key", stratify=True):
        if len(np.unique(y[tr])) < 2:
            continue
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
        ])
        model.fit(X[tr], y[tr])
        prob[te] = model.predict_proba(X[te])[:, 1]
        pred[te] = (prob[te] >= 0.5).astype(int)
        used += 1
    if used == 0:
        return {"auc": np.nan, "acc": np.nan, "f1": np.nan, "n": len(d), "n_features": len(cols)}
    try:
        auc = float(roc_auc_score(y, prob))
    except Exception:
        auc = np.nan
    return {"auc": auc, "acc": safe_float(accuracy_score(y, pred)), "f1": safe_float(f1_score(y, pred)), "n": len(d), "n_features": len(cols)}


def ensure_binary_labels(df):
    out = df.copy()
    if "use_binary" not in out.columns and "corrected_use_score" in out.columns:
        out["use_binary"] = (out["corrected_use_score"] >= 0.45).astype(int)
    if "correct_nonuse_binary" not in out.columns and "correct_nonuse_Cbit" in out.columns:
        out["correct_nonuse_binary"] = (out["correct_nonuse_Cbit"] >= 0.45).astype(int)
    if "overuse_binary" not in out.columns and "overuse_penalty" in out.columns:
        out["overuse_binary"] = (out["overuse_penalty"] >= 0.20).astype(int)
    if "future_success" not in out.columns and {"Cbit_future_proxy", "relevance_alignment"}.issubset(out.columns):
        out["future_success"] = ((out["Cbit_future_proxy"] >= 0.48) & (out["relevance_alignment"] >= 0.45)).astype(int)
    return out


def build_component_columns(df):
    # SOB-V6 internal table already contains these groups in most versions:
    # cur_, task_, bind_delta_, route_*, contrast_*.
    current = cols_by_prefix(df, ["cur_"])
    task = cols_by_prefix(df, ["task_"])
    bind = cols_by_prefix(df, ["bind_delta_", "bind_"])
    route_raw = cols_by_prefix(df, ["route_"])
    route_contrast = cols_by_prefix(df, ["contrast_", "route_USE_vs_REJECT_", "route_PARTIAL_USE_vs_REJECT_", "route_ANALOGY_USE_vs_REJECT_"])

    # Construct boundary columns if possible from existing route features.
    # This does not destroy original table; it only adds derived REJECT-vs-USEAVG features.
    route_suffixes = []
    for c in route_raw:
        if c.startswith("route_REJECT_USE_"):
            suffix = c.replace("route_REJECT_USE_", "")
            use_candidates = [f"route_{r}_{suffix}" for r in ["USE", "PARTIAL_USE", "ANALOGY_USE"]]
            if all(u in df.columns for u in use_candidates):
                route_suffixes.append(suffix)
    for suffix in route_suffixes:
        col = f"boundary_REJECT_vs_USEAVG_{suffix}"
        if col not in df.columns:
            df[col] = df[f"route_REJECT_USE_{suffix}"] - df[[f"route_USE_{suffix}", f"route_PARTIAL_USE_{suffix}", f"route_ANALOGY_USE_{suffix}"]].mean(axis=1)

    boundary = cols_by_prefix(df, ["boundary_REJECT_vs_USEAVG_", "route_REJECT_USE_"])

    meta = [c for c in ["state_rank", "low_state", "relevance_weight"] if c in df.columns]

    return {
        "CurrentOnly": current + meta,
        "TaskProbeOnly": task + meta,
        "B_bind": bind + meta,
        "R_route": route_contrast + meta,
        "B_boundary": boundary + meta,
        "U_decomposed": bind + route_contrast + boundary + meta,
        "AllRouteRaw": route_raw + meta,
        "AllInternalComparable": current + task + bind + route_contrast + boundary + meta,
    }, df


def evaluate_components(df):
    df = ensure_binary_labels(df)
    groups, df = build_component_columns(df)
    rows = []

    regression_targets = [
        "Cbit_future_proxy",
        "useful_use_Cbit",
        "correct_nonuse_Cbit",
        "overuse_penalty",
        "TQ_future_proxy",
        "AQ_future_proxy",
    ]
    classification_targets = [
        "use_binary",
        "correct_nonuse_binary",
        "overuse_binary",
        "future_success",
    ]

    scopes = [("all", df)]
    if "relevance" in df.columns:
        for rel in ["high", "medium", "low"]:
            scopes.append((f"rel_{rel}", df[df["relevance"] == rel].copy()))
    if "low_state" in df.columns:
        scopes.append(("low_states", df[df["low_state"] == 1].copy()))
        for rel in ["high", "medium", "low"]:
            scopes.append((f"low_states_rel_{rel}", df[(df["low_state"] == 1) & (df["relevance"] == rel)].copy()))

    for scope_name, sub in scopes:
        for group_name, cols in groups.items():
            row = {"scope": scope_name, "feature_group": group_name, "n": int(len(sub)), "n_features": int(len(numeric_cols(sub, cols)))}
            for target in regression_targets:
                m = reg_cv(sub, cols, target)
                row[f"{target}_r2"] = m["r2"]
                row[f"{target}_corr"] = m["corr"]
            for target in classification_targets:
                m = clf_cv(sub, cols, target)
                row[f"{target}_auc"] = m["auc"]
                row[f"{target}_acc"] = m["acc"]
                row[f"{target}_f1"] = m["f1"]
            rows.append(row)
    return pd.DataFrame(rows), df


def get_metric(summary, scope, group, metric):
    r = summary[(summary["scope"] == scope) & (summary["feature_group"] == group)]
    if len(r) == 0 or metric not in r.columns:
        return np.nan
    return safe_float(r[metric].iloc[0])


def verdict_from_summary(summary):
    bind_cbit_all = get_metric(summary, "all", "B_bind", "Cbit_future_proxy_corr")
    bind_cbit_high = get_metric(summary, "rel_high", "B_bind", "Cbit_future_proxy_corr")
    bind_cbit_medium = get_metric(summary, "rel_medium", "B_bind", "Cbit_future_proxy_corr")

    route_use_all = get_metric(summary, "all", "R_route", "useful_use_Cbit_corr")
    route_use_high = get_metric(summary, "rel_high", "R_route", "useful_use_Cbit_corr")
    route_use_medium = get_metric(summary, "rel_medium", "R_route", "useful_use_Cbit_corr")
    route_auc_all = get_metric(summary, "all", "R_route", "use_binary_auc")

    boundary_nonuse_auc_low = get_metric(summary, "rel_low", "B_boundary", "correct_nonuse_binary_auc")
    boundary_overuse_auc_low = get_metric(summary, "rel_low", "B_boundary", "overuse_binary_auc")
    boundary_nonuse_corr_low = get_metric(summary, "rel_low", "B_boundary", "correct_nonuse_Cbit_corr")
    boundary_overuse_corr_low = get_metric(summary, "rel_low", "B_boundary", "overuse_penalty_corr")

    triplet_use_auc = get_metric(summary, "all", "U_decomposed", "use_binary_auc")
    triplet_nonuse_auc_low = get_metric(summary, "rel_low", "U_decomposed", "correct_nonuse_binary_auc")
    triplet_cbit_corr = get_metric(summary, "all", "U_decomposed", "Cbit_future_proxy_corr")
    triplet_success_auc = get_metric(summary, "all", "U_decomposed", "future_success_auc")

    binding_component_support = bool(
        (np.isfinite(bind_cbit_all) and bind_cbit_all > 0.35) or
        (np.isfinite(bind_cbit_high) and bind_cbit_high > 0.35) or
        (np.isfinite(bind_cbit_medium) and bind_cbit_medium > 0.35)
    )
    routing_component_support = bool(
        (np.isfinite(route_use_all) and route_use_all > 0.35) or
        (np.isfinite(route_use_high) and route_use_high > 0.35) or
        (np.isfinite(route_use_medium) and route_use_medium > 0.35) or
        (np.isfinite(route_auc_all) and route_auc_all > 0.70)
    )
    boundary_component_support = bool(
        (np.isfinite(boundary_nonuse_auc_low) and boundary_nonuse_auc_low > 0.70) or
        (np.isfinite(boundary_overuse_auc_low) and boundary_overuse_auc_low > 0.70) or
        (np.isfinite(boundary_nonuse_corr_low) and boundary_nonuse_corr_low > 0.35) or
        (np.isfinite(boundary_overuse_corr_low) and abs(boundary_overuse_corr_low) > 0.35)
    )
    triplet_support = bool(
        (np.isfinite(triplet_use_auc) and triplet_use_auc > 0.70) and
        (np.isfinite(triplet_nonuse_auc_low) and triplet_nonuse_auc_low > 0.70)
    )
    cbit_triplet_support = bool(np.isfinite(triplet_cbit_corr) and triplet_cbit_corr > 0.35)
    success_support = bool(np.isfinite(triplet_success_auc) and triplet_success_auc > 0.70)

    component_count = sum([binding_component_support, routing_component_support, boundary_component_support])
    if component_count == 3 and triplet_support:
        verdict = "PASS_STRONG_INTERNAL_U_DECOMPOSITION"
    elif component_count >= 2 and (triplet_support or cbit_triplet_support or success_support):
        verdict = "PASS_INTERNAL_U_DECOMPOSITION"
    elif component_count >= 2 or triplet_support or cbit_triplet_support or success_support:
        verdict = "PARTIAL_INTERNAL_U_DECOMPOSITION"
    else:
        verdict = "FAIL_INTERNAL_U_DECOMPOSITION"

    return {
        "verdict": verdict,
        "bind_cbit_corr_all": bind_cbit_all,
        "bind_cbit_corr_high": bind_cbit_high,
        "bind_cbit_corr_medium": bind_cbit_medium,
        "route_useful_use_corr_all": route_use_all,
        "route_useful_use_corr_high": route_use_high,
        "route_useful_use_corr_medium": route_use_medium,
        "route_use_binary_auc_all": route_auc_all,
        "boundary_low_correct_nonuse_auc": boundary_nonuse_auc_low,
        "boundary_low_overuse_auc": boundary_overuse_auc_low,
        "boundary_low_correct_nonuse_corr": boundary_nonuse_corr_low,
        "boundary_low_overuse_corr": boundary_overuse_corr_low,
        "triplet_use_auc": triplet_use_auc,
        "triplet_low_correct_nonuse_auc": triplet_nonuse_auc_low,
        "triplet_cbit_corr": triplet_cbit_corr,
        "triplet_success_auc": triplet_success_auc,
        "binding_component_support": binding_component_support,
        "routing_component_support": routing_component_support,
        "boundary_component_support": boundary_component_support,
        "triplet_support": triplet_support,
        "cbit_triplet_support": cbit_triplet_support,
        "success_support": success_support,
    }


def run_one_model(model_key):
    model_out = OUT_DIR / model_key
    model_out.mkdir(parents=True, exist_ok=True)
    in_path = IN_DIR / model_key / f"SOB-V6_{model_key}_internal_u_table.csv"
    info = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "model_key": model_key,
        "input_path": str(in_path),
        "exists": in_path.exists(),
        "status": "pending",
        "started_at": now_time(),
    }
    if not in_path.exists():
        info.update({"status": "missing_input", "error": f"Missing V6 internal table: {in_path}"})
        return info
    try:
        df = pd.read_csv(in_path)
        summary, df2 = evaluate_components(df)
        verdict = verdict_from_summary(summary)
        df2.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_u_decomposition_table.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_u_decomposition_eval_summary.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([{**{"model_key": model_key}, **verdict}]).to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_verdict_summary.csv", index=False, encoding="utf-8-sig")
        info.update({
            "status": "done",
            "finished_at": now_time(),
            "n_rows": int(len(df2)),
            "n_columns": int(len(df2.columns)),
            "verdict": verdict["verdict"],
            "metrics": verdict,
            "outputs": {
                "u_decomposition_table": str(model_out / f"{EXPERIMENT_ID}_{model_key}_u_decomposition_table.csv"),
                "eval_summary": str(model_out / f"{EXPERIMENT_ID}_{model_key}_u_decomposition_eval_summary.csv"),
                "verdict_summary": str(model_out / f"{EXPERIMENT_ID}_{model_key}_verdict_summary.csv"),
            },
        })
        with open(model_out / f"{EXPERIMENT_ID}_{model_key}_verdict.json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
    except Exception as e:
        info.update({"status": "failed", "finished_at": now_time(), "error_type": type(e).__name__, "error": str(e)})
    return info


def cross_model_summary(model_infos):
    rows = []
    for info in model_infos:
        m = info.get("metrics", {}) or {}
        rows.append({
            "model_key": info.get("model_key"),
            "status": info.get("status"),
            "verdict": info.get("verdict"),
            "bind_cbit_corr_all": m.get("bind_cbit_corr_all"),
            "bind_cbit_corr_high": m.get("bind_cbit_corr_high"),
            "bind_cbit_corr_medium": m.get("bind_cbit_corr_medium"),
            "route_useful_use_corr_all": m.get("route_useful_use_corr_all"),
            "route_useful_use_corr_high": m.get("route_useful_use_corr_high"),
            "route_useful_use_corr_medium": m.get("route_useful_use_corr_medium"),
            "route_use_binary_auc_all": m.get("route_use_binary_auc_all"),
            "boundary_low_correct_nonuse_auc": m.get("boundary_low_correct_nonuse_auc"),
            "boundary_low_overuse_auc": m.get("boundary_low_overuse_auc"),
            "boundary_low_correct_nonuse_corr": m.get("boundary_low_correct_nonuse_corr"),
            "boundary_low_overuse_corr": m.get("boundary_low_overuse_corr"),
            "triplet_use_auc": m.get("triplet_use_auc"),
            "triplet_low_correct_nonuse_auc": m.get("triplet_low_correct_nonuse_auc"),
            "triplet_cbit_corr": m.get("triplet_cbit_corr"),
            "triplet_success_auc": m.get("triplet_success_auc"),
            "binding_component_support": m.get("binding_component_support"),
            "routing_component_support": m.get("routing_component_support"),
            "boundary_component_support": m.get("boundary_component_support"),
            "triplet_support": m.get("triplet_support"),
            "cbit_triplet_support": m.get("cbit_triplet_support"),
            "success_support": m.get("success_support"),
            "n_rows": info.get("n_rows"),
            "error": info.get("error"),
        })
    return pd.DataFrame(rows)


def global_verdict(cross):
    done = cross[cross["status"] == "done"].copy()
    if len(done) == 0:
        return "FAIL_SOBV6B_NO_MODELS"
    verdicts = done["verdict"].astype(str).tolist()
    pass_strong = sum(v.startswith("PASS_STRONG") for v in verdicts)
    pass_any = sum(v.startswith("PASS") for v in verdicts)
    partial = sum(v.startswith("PARTIAL") for v in verdicts)
    if pass_strong == len(done):
        return "PASS_STRONG_SOBV6B_CROSS_MODEL"
    if pass_any == len(done):
        return "PASS_SOBV6B_CROSS_MODEL"
    if pass_any + partial == len(done):
        return "PARTIAL_SOBV6B_CROSS_MODEL"
    if pass_any >= 2:
        return "MIXED_POSITIVE_SOBV6B"
    if pass_any >= 1:
        return "MIXED_SOBV6B"
    return "FAIL_SOBV6B"


def main():
    print(f"script_version={SCRIPT_VERSION}", flush=True)
    print(f"IN_DIR={IN_DIR}", flush=True)
    print(f"OUT_DIR={OUT_DIR}", flush=True)
    model_infos = []
    for model_key in MODELS:
        print("=" * 80)
        print(f"Running {model_key}", flush=True)
        info = run_one_model(model_key)
        model_infos.append(info)
        print(json.dumps({"model_key": model_key, "status": info.get("status"), "verdict": info.get("verdict"), "error": info.get("error")}, ensure_ascii=False, indent=2), flush=True)
    cross = cross_model_summary(model_infos)
    cross_path = OUT_DIR / f"{EXPERIMENT_ID}_cross_model_summary.csv"
    cross.to_csv(cross_path, index=False, encoding="utf-8-sig")
    gv = global_verdict(cross)
    global_obj = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "verdict": gv,
        "n_models": len(model_infos),
        "n_models_done": int((cross["status"] == "done").sum()) if "status" in cross.columns else 0,
        "model_infos": model_infos,
        "outputs": {"cross_model_summary": str(cross_path), "out_dir": str(OUT_DIR)},
        "interpretation": {
            "B_bind": "TaskProbe - CurrentProbe; object-task binding component.",
            "R_route": "USE/PARTIAL/ANALOGY route contrast against REJECT; useful-use routing component.",
            "B_boundary": "REJECT route versus use-route average; correct-nonuse / overuse boundary component.",
            "U_decomposed": "B_bind + R_route + B_boundary; specialized internal U measurement model.",
        },
    }
    with open(OUT_DIR / f"{EXPERIMENT_ID}_verdict.json", "w", encoding="utf-8") as f:
        json.dump(global_obj, f, ensure_ascii=False, indent=2)
    print("=" * 80)
    print(json.dumps(global_obj, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
