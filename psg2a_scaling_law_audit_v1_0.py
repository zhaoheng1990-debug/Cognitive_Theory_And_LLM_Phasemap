# -*- coding: utf-8 -*-
"""
PSG-2A Scaling Law Audit
Projection-Selection Geometry Boundary Audit

Goal
----
Map how TopK centroid readback strength scales with:
    d       : dimension
    V       : vocabulary / candidate pool size
    k       : selected top-k size
    k/V     : selection fraction
    n       : number of query vectors

Pure synthetic experiment. No model, no tokenizer.

Core estimator
--------------
Given query H_i and random candidate matrix R:

    scores_i = H_i R^T
    S_i = TopK(scores_i, k)
    Z_i = mean(R[S_i])

Then audit:
    1. pairwise geometry preservation:
        corr(dist(H_i,H_j), dist(Z_i,Z_j))

    2. direct readback:
        self_cos = cos(Z_i, H_i)
        off_cos  = average_j!=i cos(Z_i, H_j)
        margin   = self_cos - off_cos

    3. estimator coefficient:
        alpha_i = <Z_i, H_i> / ||H_i||^2
        residual_ratio = ||Z_i - alpha_i H_i|| / ||Z_i||

Expected:
    Z_i ≈ alpha(d,V,k) * H_i + residual

Run
---
python psg2a_scaling_law_audit_v1_0.py

Outputs
-------
C:\\Users\\ZH\\Desktop\\AGI\\outputs\\psg2a_scaling_outputs
"""

import time
import random
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import pairwise_distances


# =========================
# 0. Config
# =========================

EXPERIMENT_ID = "PSG-2A_Scaling_Law_Audit_v1_0"

OUT_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\psg2a_scaling_outputs")

SEED = 42

# Keep practical. V=65536, d=1536, n=64 is still manageable on CPU,
# but runtime depends on machine. This grid is medium-small.
N_QUERIES_LIST = [16, 32, 64]
D_LIST = [128, 256, 512, 1024, 1536]
VOCAB_LIST = [4096, 8192, 16384, 32768, 65536]

# Use both absolute k and fraction-like coverage.
K_LIST = [64, 128, 256, 512, 1000, 2000, 5000, 10000]

N_REPEATS = 3

# Main geometry. Add anisotropic only as light boundary check.
GEOMETRY_MODES = ["sphere_iid", "gaussian_iid"]

# Selection comparison. TopK is target; randomK is control.
SELECTION_MODES = ["topk", "randomk", "bottomk"]


# =========================
# 1. Utilities
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def l2_normalize(x, axis=-1, eps=1e-9):
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(denom, eps)


def cosine_distance_matrix(x):
    return pairwise_distances(np.asarray(x, dtype=np.float32), metric="cosine")


def upper_tri_values(mat):
    idx = np.triu_indices(mat.shape[0], k=1)
    return mat[idx]


def safe_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4:
        return np.nan, np.nan
    if np.std(x[mask]) < 1e-12 or np.std(y[mask]) < 1e-12:
        return np.nan, np.nan
    return float(pearsonr(x[mask], y[mask]).statistic), float(spearmanr(x[mask], y[mask]).statistic)


def make_H_R(n, V, d, mode, rng):
    if mode == "sphere_iid":
        H = rng.standard_normal((n, d)).astype(np.float32)
        R = rng.standard_normal((V, d)).astype(np.float32)
        H = l2_normalize(H)
        R = l2_normalize(R)
        return H, R

    if mode == "gaussian_iid":
        H = rng.standard_normal((n, d)).astype(np.float32)
        R = rng.standard_normal((V, d)).astype(np.float32)
        # Normalize H only; R remains Gaussian.
        # This tests whether raw Gaussian norm noise changes scaling.
        H = l2_normalize(H)
        return H, R

    raise ValueError(f"Unknown geometry mode: {mode}")


def select_indices(scores, k, selection, rng):
    n, V = scores.shape
    k = min(k, V)

    if selection == "topk":
        idx = np.argpartition(-scores, kth=k-1, axis=1)[:, :k]
        selected_scores = np.take_along_axis(scores, idx, axis=1)
        order = np.argsort(-selected_scores, axis=1)
        return np.take_along_axis(idx, order, axis=1)

    if selection == "bottomk":
        idx = np.argpartition(scores, kth=k-1, axis=1)[:, :k]
        selected_scores = np.take_along_axis(scores, idx, axis=1)
        order = np.argsort(selected_scores, axis=1)
        return np.take_along_axis(idx, order, axis=1)

    if selection == "randomk":
        return np.stack([rng.choice(V, size=k, replace=False) for _ in range(n)], axis=0)

    raise ValueError(f"Unknown selection: {selection}")


def estimator_metrics(H, Z):
    Hn = l2_normalize(H)
    Zn = l2_normalize(Z)

    sim = Zn @ Hn.T
    self_cos = np.diag(sim)
    off_mask = ~np.eye(sim.shape[0], dtype=bool)
    off_cos = sim[off_mask]

    # H is normalized in all modes.
    dot = np.sum(Z * Hn, axis=1)
    hnorm2 = np.sum(Hn * Hn, axis=1)
    alpha = dot / np.maximum(hnorm2, 1e-9)

    Z_proj = alpha[:, None] * Hn
    residual = Z - Z_proj
    residual_norm = np.linalg.norm(residual, axis=1)
    z_norm = np.linalg.norm(Z, axis=1)
    residual_ratio = residual_norm / np.maximum(z_norm, 1e-9)

    return {
        "self_cos_mean": float(np.mean(self_cos)),
        "self_cos_std": float(np.std(self_cos)),
        "off_cos_mean": float(np.mean(off_cos)),
        "self_vs_off_margin": float(np.mean(self_cos) - np.mean(off_cos)),
        "alpha_mean": float(np.mean(alpha)),
        "alpha_std": float(np.std(alpha)),
        "z_norm_mean": float(np.mean(z_norm)),
        "z_norm_std": float(np.std(z_norm)),
        "residual_ratio_mean": float(np.mean(residual_ratio)),
        "residual_ratio_std": float(np.std(residual_ratio)),
    }


def run_one(n, V, d, k, mode, selection, repeat_id):
    seed = SEED + 1000003 * repeat_id + 1009 * n + 17 * V + 13 * d + k + hash((mode, selection)) % 10007
    rng = np.random.default_rng(seed)

    t0 = time.time()
    H, R = make_H_R(n, V, d, mode, rng)

    scores = H @ R.T
    idx = select_indices(scores, k, selection, rng)
    Z = R[idx].mean(axis=1)

    D_H = cosine_distance_matrix(H)
    D_Z = cosine_distance_matrix(Z)

    pear, spear = safe_corr(upper_tri_values(D_H), upper_tri_values(D_Z))
    metrics = estimator_metrics(H, Z)

    out = {
        "experiment_id": EXPERIMENT_ID,
        "n_queries": n,
        "V": V,
        "d": d,
        "k": k,
        "k_over_V": k / V,
        "geometry_mode": mode,
        "selection": selection,
        "repeat_id": repeat_id,
        "topo_pearson": pear,
        "topo_spearman": spear,
        "abs_topo_spearman": abs(spear) if np.isfinite(spear) else np.nan,
        "elapsed_sec": time.time() - t0,
    }
    out.update(metrics)
    return out


def build_verdict(summary):
    lines = []
    lines.append("=" * 80)
    lines.append("PSG-2A Scaling Law Audit Verdict")
    lines.append("=" * 80)

    if summary.empty:
        lines.append("NO_VALID_ROWS")
        return "\n".join(lines)

    def mean_query(expr, col="topo_spearman"):
        sub = summary.query(expr)
        if sub.empty:
            return np.nan
        return float(sub[col].mean())

    topk = mean_query("selection == 'topk'")
    randomk = mean_query("selection == 'randomk'")
    bottomk = mean_query("selection == 'bottomk'")
    topk_margin = mean_query("selection == 'topk'", "self_vs_off_margin")
    topk_resid = mean_query("selection == 'topk'", "residual_ratio_mean")

    lines.append("\nMain means:")
    lines.append(f"topk_topo_spearman_mean={topk:.4f}")
    lines.append(f"bottomk_topo_spearman_mean={bottomk:.4f}")
    lines.append(f"randomk_topo_spearman_mean={randomk:.4f}")
    lines.append(f"topk_self_vs_off_margin_mean={topk_margin:.4f}")
    lines.append(f"topk_residual_ratio_mean={topk_resid:.4f}")

    lines.append("\nTop 20 conditions:")
    top = summary.sort_values("topo_spearman", ascending=False).head(20)
    for _, r in top.iterrows():
        lines.append(
            f"mode={r['geometry_mode']} n={int(r['n_queries'])} V={int(r['V'])} "
            f"d={int(r['d'])} k={int(r['k'])} k/V={r['k_over_V']:.4f} "
            f"select={r['selection']} spearman={r['topo_spearman']:.4f} "
            f"selfcos={r['self_cos_mean']:.4f} alpha={r['alpha_mean']:.4f} "
            f"resid={r['residual_ratio_mean']:.4f}"
        )

    lines.append("\nScaling summaries:")
    for col in ["d", "V", "k", "k_over_V", "n_queries"]:
        if col == "k_over_V":
            tmp = summary[summary["selection"] == "topk"].copy()
            # bin k/V for concise display
            tmp["k_over_V_bin"] = pd.cut(tmp["k_over_V"], bins=[0, .005, .02, .08, .25, 1.0])
            view = tmp.groupby("k_over_V_bin")["topo_spearman"].mean().reset_index()
            lines.append("\n[topk by k_over_V_bin]")
            for _, rr in view.iterrows():
                lines.append(f"{rr['k_over_V_bin']}: mean={rr['topo_spearman']:.4f}")
        else:
            view = summary[summary["selection"] == "topk"].groupby(col)["topo_spearman"].mean().reset_index()
            lines.append(f"\n[topk by {col}]")
            for _, rr in view.iterrows():
                lines.append(f"{rr[col]}: mean={rr['topo_spearman']:.4f}")

    lines.append("\nInterpretation:")
    if np.isfinite(topk) and np.isfinite(randomk):
        if topk > randomk + 0.20:
            lines.append("PASS: TopK selection strongly exceeds randomK across scaling grid.")
        else:
            lines.append("WEAK/FAIL: TopK does not strongly exceed randomK across scaling grid.")

    if np.isfinite(bottomk) and np.isfinite(topk) and abs(bottomk - topk) < 0.05:
        lines.append("SUPPORT: BottomK preserves geometry similarly to TopK, consistent with Z≈-alpha H.")
    else:
        lines.append("MIXED: BottomK differs from TopK; inspect sign and distribution effects.")

    if np.isfinite(topk_resid):
        if topk_resid < 0.40:
            lines.append("PASS-ESTIMATOR: TopK centroid is close to a rank-1 query-direction estimator.")
        elif topk_resid < 0.75:
            lines.append("WEAK-ESTIMATOR: TopK centroid has query-direction component but substantial residual.")
        else:
            lines.append("PAIRWISE-ONLY: Geometry may be preserved without tight direct estimator form.")

    lines.append("\nSuggested next:")
    lines.append("Use psg2a_scaling_summary.csv to fit lightweight formula alpha/self_cos vs d,V,k.")
    lines.append("Proceed to PSG-2B only if direct estimator coefficient/residual needs finer audit.")
    lines.append("=" * 80)
    return "\n".join(lines)


def main():
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    total = 0
    for n in N_QUERIES_LIST:
        for d in D_LIST:
            for V in VOCAB_LIST:
                for k in K_LIST:
                    if k < V:
                        total += len(GEOMETRY_MODES) * len(SELECTION_MODES) * N_REPEATS

    done = 0
    start = time.time()

    for n in N_QUERIES_LIST:
        for d in D_LIST:
            for V in VOCAB_LIST:
                for k in K_LIST:
                    if k >= V:
                        continue
                    for mode in GEOMETRY_MODES:
                        for selection in SELECTION_MODES:
                            for rep in range(N_REPEATS):
                                try:
                                    rows.append(run_one(n, V, d, k, mode, selection, rep))
                                except Exception as e:
                                    rows.append({
                                        "experiment_id": EXPERIMENT_ID,
                                        "n_queries": n,
                                        "V": V,
                                        "d": d,
                                        "k": k,
                                        "k_over_V": k / V,
                                        "geometry_mode": mode,
                                        "selection": selection,
                                        "repeat_id": rep,
                                        "error": repr(e),
                                    })
                                done += 1
                                if done % 50 == 0:
                                    print(f"[INFO] progress {done}/{total}; rows={len(rows)}; elapsed={time.time() - start:.1f}s")

    raw = pd.DataFrame(rows)
    raw_path = OUT_DIR / "psg2a_scaling_raw.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")

    ok = raw[raw.get("topo_spearman", pd.Series(index=raw.index)).notna()].copy()
    group_cols = ["n_queries", "V", "d", "k", "k_over_V", "geometry_mode", "selection"]
    numeric_cols = [
        "topo_pearson", "topo_spearman", "abs_topo_spearman",
        "self_cos_mean", "self_cos_std", "off_cos_mean", "self_vs_off_margin",
        "alpha_mean", "alpha_std", "z_norm_mean", "z_norm_std",
        "residual_ratio_mean", "residual_ratio_std", "elapsed_sec"
    ]
    summary = ok.groupby(group_cols, dropna=False)[numeric_cols].mean().reset_index()
    summary["n_repeats"] = ok.groupby(group_cols, dropna=False).size().values

    summary_path = OUT_DIR / "psg2a_scaling_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    for col in ["selection", "geometry_mode", "d", "V", "k", "n_queries"]:
        view = summary.groupby(col)["topo_spearman"].agg(["mean", "median", "max", "std", "count"]).reset_index()
        view = view.sort_values("mean", ascending=False)
        view.to_csv(OUT_DIR / f"psg2a_factor_view_{col}.csv", index=False, encoding="utf-8-sig")

    # k/V binned view
    tmp = summary.copy()
    tmp["k_over_V_bin"] = pd.cut(tmp["k_over_V"], bins=[0, .002, .005, .01, .02, .05, .10, .25, 1.0])
    kv_view = tmp.groupby(["selection", "k_over_V_bin"])["topo_spearman"].agg(["mean", "median", "max", "std", "count"]).reset_index()
    kv_view.to_csv(OUT_DIR / "psg2a_factor_view_k_over_V_bin.csv", index=False, encoding="utf-8-sig")

    top = summary.sort_values("topo_spearman", ascending=False).head(100)
    top.to_csv(OUT_DIR / "psg2a_top100_conditions.csv", index=False, encoding="utf-8-sig")

    verdict = build_verdict(summary)
    verdict_path = OUT_DIR / "psg2a_verdict.txt"
    verdict_path.write_text(verdict, encoding="utf-8")

    print("\n" + verdict)
    print(f"\n[OK] raw: {raw_path}")
    print(f"[OK] summary: {summary_path}")
    print(f"[OK] verdict: {verdict_path}")


if __name__ == "__main__":
    main()
