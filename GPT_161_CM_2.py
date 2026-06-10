# -*- coding: utf-8 -*-
"""
CM-2: Model-Specific Critical Band Audit

Purpose
-------
CM-1 showed that cross-model PC1 dominance is strong, but absolute tau layers and
phase ordering are model-dependent. CM-2 treats tau as a *model-specific critical
band*, not a fixed layer threshold.

Input
-----
Reads CM-1 outputs from:
    cm1_outputs/<model>_deltaR_dataset.csv
    cm1_outputs/<model>_layer_R_dataset.csv

Output
------
Writes:
    cm2_outputs/cm2_model_summary.csv
    cm2_outputs/cm2_overall_summary.json
    cm2_outputs/<model>_critical_band_summary.json
    cm2_outputs/<model>_deltaU_band_dataset.csv
    cm2_outputs/<model>_layer_boundary_profile.csv

Run
---
    python cm2_model_specific_critical_band_audit.py

Interpretation
--------------
CM-2 PASS-Structural if:
  1) PC1 remains dominant in each model.
  2) A model-specific [U-, U+] can enrich critical samples.
  3) Layerwise boundary-distance profile has a localized high critical-vs-rest signal.

This script does not assume that Qwen/Llama/Gemma share the same absolute tau layer.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score


MODEL_KEYS = ["qwen", "llama", "gemma"]
PHASE_ORDER = ["positive", "critical", "negative"]


def eta_squared(values: np.ndarray, groups: List[str]) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    grand = np.mean(values)
    ss_total = np.sum((values - grand) ** 2)
    if ss_total <= 1e-12:
        return 0.0
    ss_between = 0.0
    for g in sorted(set(groups)):
        idx = np.array([x == g for x in groups])
        if idx.sum() == 0:
            continue
        ss_between += idx.sum() * (np.mean(values[idx]) - grand) ** 2
    return float(ss_between / ss_total)


def safe_auc(y_true, score):
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y, np.asarray(score, dtype=float)))
    except Exception:
        return np.nan


def binary_metrics(y_true, pred):
    y = np.asarray(y_true).astype(int)
    p = np.asarray(pred).astype(int)
    return {
        "accuracy": float(accuracy_score(y, p)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "pred_pos_rate": float(np.mean(p)),
    }


def get_dR_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if c.startswith("dR_L"):
            try:
                int(c.replace("dR_L", ""))
                cols.append(c)
            except Exception:
                pass
    return sorted(cols, key=lambda x: int(x.replace("dR_L", "")))


def get_R_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if c.startswith("R_L"):
            try:
                int(c.replace("R_L", ""))
                cols.append(c)
            except Exception:
                pass
    return sorted(cols, key=lambda x: int(x.replace("R_L", "")))


def infer_decision_cols(df_delta: pd.DataFrame, model_key: str) -> List[str]:
    # Prefer the same decision windows used by CM-1.
    all_cols = get_dR_cols(df_delta)
    layers = [int(c.replace("dR_L", "")) for c in all_cols]
    L = max(layers) + 1
    if model_key == "qwen":
        lo, hi = 20, 25
    elif model_key == "llama":
        lo, hi = 11, 14
    elif model_key == "gemma":
        lo, hi = 19, 23
    else:
        lo, hi = int(round(20/28 * L)), int(round(25/28 * L))
    return [f"dR_L{l}" for l in range(lo, hi + 1) if f"dR_L{l}" in df_delta.columns]


def orient_pc1(df_delta: pd.DataFrame, decision_cols: List[str]) -> Tuple[np.ndarray, PCA, float]:
    X = df_delta[decision_cols].values.astype(np.float32)
    pca = PCA(n_components=min(3, X.shape[1]))
    pcs = pca.fit_transform(X)
    pc1 = pcs[:, 0]
    phases = np.asarray(df_delta["phase"].tolist())
    mean_pos = float(np.mean(pc1[phases == "positive"])) if np.any(phases == "positive") else 0.0
    mean_neg = float(np.mean(pc1[phases == "negative"])) if np.any(phases == "negative") else 0.0
    sign = 1.0 if mean_pos >= mean_neg else -1.0
    return pc1 * sign, pca, sign


def fit_critical_band(delta_u: np.ndarray, phase: np.ndarray) -> Dict:
    """Fit model-specific critical band by critical quantiles and grid-search variants."""
    ycrit = (phase == "critical").astype(int)
    crit_vals = delta_u[ycrit == 1]
    if len(crit_vals) < 3:
        return {"error": "not enough critical samples"}

    # Baseline quantile band.
    q10, q90 = np.quantile(crit_vals, [0.10, 0.90])
    q25, q75 = np.quantile(crit_vals, [0.25, 0.75])

    def eval_band(lo, hi):
        pred = ((delta_u >= lo) & (delta_u <= hi)).astype(int)
        mets = binary_metrics(ycrit, pred)
        # Band score: prefer high F1, but penalize trivial huge bands by precision.
        score = 0.65 * mets["f1"] + 0.35 * mets["precision"]
        return score, mets

    # Grid search over critical quantiles and slight expansions.
    qs = np.linspace(0.02, 0.98, 49)
    crit_quantiles = np.quantile(crit_vals, qs)
    best = None
    for i, lo in enumerate(crit_quantiles):
        for j, hi in enumerate(crit_quantiles):
            if hi <= lo:
                continue
            score, mets = eval_band(lo, hi)
            if best is None or score > best["score"]:
                best = {"lo": float(lo), "hi": float(hi), "score": float(score), **mets}

    # Continuous distance to band; lower means more critical-like.
    lo, hi = best["lo"], best["hi"]
    dist = np.where(delta_u < lo, lo - delta_u, np.where(delta_u > hi, delta_u - hi, 0.0))
    auc_dist_low = safe_auc(ycrit, -dist)

    by_phase = {}
    for ph in PHASE_ORDER:
        idx = phase == ph
        if idx.sum() == 0:
            continue
        in_band = ((delta_u[idx] >= lo) & (delta_u[idx] <= hi)).astype(float)
        by_phase[ph] = {
            "n": int(idx.sum()),
            "DeltaU_mean": float(np.mean(delta_u[idx])),
            "DeltaU_std": float(np.std(delta_u[idx])),
            "in_band_frac": float(np.mean(in_band)),
            "dist_to_band_mean": float(np.mean(dist[idx])),
        }

    return {
        "critical_quantile_band_q10_q90": [float(q10), float(q90)],
        "critical_iqr_band_q25_q75": [float(q25), float(q75)],
        "optimized_band": best,
        "auc_critical_by_negative_distance_to_band": auc_dist_low,
        "by_phase": by_phase,
        "critical_enrichment_over_positive": float(by_phase.get("critical", {}).get("in_band_frac", np.nan) - by_phase.get("positive", {}).get("in_band_frac", np.nan)),
        "critical_enrichment_over_negative": float(by_phase.get("critical", {}).get("in_band_frac", np.nan) - by_phase.get("negative", {}).get("in_band_frac", np.nan)),
    }


def layer_boundary_profile(df_R: pd.DataFrame) -> pd.DataFrame:
    rows = []
    R_cols = get_R_cols(df_R)
    phase = np.asarray(df_R["phase"].tolist())
    ycrit = (phase == "critical").astype(int)
    L = len(R_cols)
    for c in R_cols:
        l = int(c.replace("R_L", ""))
        R = df_R[c].values.astype(float)
        DB = np.abs(R)
        rec = {
            "layer": l,
            "s_norm": l / max(1, L),
            "eta2_R_phase": eta_squared(R, phase.tolist()),
            "eta2_DB_phase": eta_squared(DB, phase.tolist()),
            "auc_critical_low_DB": safe_auc(ycrit, -DB),
            "DB_mean_all": float(np.mean(DB)),
        }
        for ph in PHASE_ORDER:
            idx = phase == ph
            if idx.sum() == 0:
                continue
            rec[f"R_mean_{ph}"] = float(np.mean(R[idx]))
            rec[f"DB_mean_{ph}"] = float(np.mean(DB[idx]))
        rows.append(rec)
    return pd.DataFrame(rows)


def audit_one(model_key: str, cm1_dir: Path, outdir: Path) -> Dict:
    delta_path = cm1_dir / f"{model_key}_deltaR_dataset.csv"
    R_path = cm1_dir / f"{model_key}_layer_R_dataset.csv"
    if not delta_path.exists() or not R_path.exists():
        return {"model_key": model_key, "error": f"missing {delta_path} or {R_path}"}

    df_delta = pd.read_csv(delta_path)
    df_R = pd.read_csv(R_path)
    decision_cols = infer_decision_cols(df_delta, model_key)
    delta_u, pca, sign = orient_pc1(df_delta, decision_cols)
    df_delta["DeltaU_CM2"] = delta_u

    phase = np.asarray(df_delta["phase"].tolist())
    phase_means = {ph: float(np.mean(delta_u[phase == ph])) for ph in sorted(set(phase))}

    band = fit_critical_band(delta_u, phase)

    # Layerwise boundary-distance profile.
    df_prof = layer_boundary_profile(df_R)
    best_layer_auc = int(df_prof.loc[df_prof["auc_critical_low_DB"].idxmax(), "layer"])
    best_layer_etaDB = int(df_prof.loc[df_prof["eta2_DB_phase"].idxmax(), "layer"])

    df_delta.to_csv(outdir / f"{model_key}_deltaU_band_dataset.csv", index=False, encoding="utf-8-sig")
    df_prof.to_csv(outdir / f"{model_key}_layer_boundary_profile.csv", index=False, encoding="utf-8-sig")

    summary = {
        "model_key": model_key,
        "decision_cols": decision_cols,
        "pc_variance": pca.explained_variance_ratio_.tolist(),
        "pc1_variance": float(pca.explained_variance_ratio_[0]),
        "phase_means_DeltaU_CM2": phase_means,
        "critical_band": band,
        "best_layer_critical_low_DB_auc": best_layer_auc,
        "best_layer_eta2_DB_phase": best_layer_etaDB,
        "best_auc_critical_low_DB": float(df_prof["auc_critical_low_DB"].max()),
        "best_eta2_DB_phase": float(df_prof["eta2_DB_phase"].max()),
    }

    # Compact PASS flags.
    cb = band.get("by_phase", {})
    c_frac = cb.get("critical", {}).get("in_band_frac", np.nan)
    p_frac = cb.get("positive", {}).get("in_band_frac", np.nan)
    n_frac = cb.get("negative", {}).get("in_band_frac", np.nan)
    summary["pass_pc1_dominant_0p80"] = bool(summary["pc1_variance"] >= 0.80)
    summary["pass_critical_band_enriched"] = bool(np.isfinite(c_frac) and c_frac > max(p_frac, n_frac))
    summary["pass_boundary_profile_signal"] = bool(summary["best_auc_critical_low_DB"] >= 0.65 or summary["best_eta2_DB_phase"] >= 0.25)

    with open(outdir / f"{model_key}_critical_band_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cm1_dir", type=str, default="cm1_outputs")
    parser.add_argument("--outdir", type=str, default="cm2_outputs")
    parser.add_argument("--models", nargs="*", default=MODEL_KEYS)
    args = parser.parse_args()

    cm1_dir = Path(args.cm1_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for m in args.models:
        print(f"\n==== CM-2 auditing {m} ====")
        s = audit_one(m, cm1_dir, outdir)
        summaries.append(s)
        print(json.dumps({k: v for k, v in s.items() if k not in ["critical_band"]}, ensure_ascii=False, indent=2))

    # Flatten a few key fields for CSV.
    flat = []
    for s in summaries:
        row = {
            "model_key": s.get("model_key"),
            "pc1_variance": s.get("pc1_variance"),
            "best_layer_critical_low_DB_auc": s.get("best_layer_critical_low_DB_auc"),
            "best_layer_eta2_DB_phase": s.get("best_layer_eta2_DB_phase"),
            "best_auc_critical_low_DB": s.get("best_auc_critical_low_DB"),
            "best_eta2_DB_phase": s.get("best_eta2_DB_phase"),
            "pass_pc1_dominant_0p80": s.get("pass_pc1_dominant_0p80"),
            "pass_critical_band_enriched": s.get("pass_critical_band_enriched"),
            "pass_boundary_profile_signal": s.get("pass_boundary_profile_signal"),
        }
        band = s.get("critical_band", {})
        opt = band.get("optimized_band", {}) if isinstance(band, dict) else {}
        row.update({
            "band_lo": opt.get("lo"),
            "band_hi": opt.get("hi"),
            "band_f1": opt.get("f1"),
            "band_precision": opt.get("precision"),
            "band_recall": opt.get("recall"),
            "auc_critical_by_neg_dist_to_band": band.get("auc_critical_by_negative_distance_to_band") if isinstance(band, dict) else None,
            "critical_in_band_frac": band.get("by_phase", {}).get("critical", {}).get("in_band_frac") if isinstance(band, dict) else None,
            "positive_in_band_frac": band.get("by_phase", {}).get("positive", {}).get("in_band_frac") if isinstance(band, dict) else None,
            "negative_in_band_frac": band.get("by_phase", {}).get("negative", {}).get("in_band_frac") if isinstance(band, dict) else None,
        })
        flat.append(row)

    pd.DataFrame(flat).to_csv(outdir / "cm2_model_summary.csv", index=False, encoding="utf-8-sig")
    with open(outdir / "cm2_overall_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print("\n==== CM-2 complete ====")
    print(f"Outputs saved to: {outdir.resolve()}")
    print(pd.DataFrame(flat))


if __name__ == "__main__":
    main()
