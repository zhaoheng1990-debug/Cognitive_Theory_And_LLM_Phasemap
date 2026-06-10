# -*- coding: utf-8 -*-
"""
PSG-2B Conditional Mean Estimator Audit

Goal
----
Validate the mathematical mechanism behind PSG:

For isotropic random vectors R and a unit query h,

    z = mean{ R_j : h·R_j is in TopK }

should satisfy:

    z ≈ alpha * h + epsilon

where alpha is approximately the conditional mean of the projection variable
given it lies above the TopK threshold.

For Gaussian R:
    s = h·R ~ N(0,1) if R is raw Gaussian and h is unit.
    alpha_theory ≈ E[s | s >= t] = phi(t) / (1-Phi(t))
    where t = Phi^{-1}(1 - k/V).

For sphere-normalized R:
    s = h·R has variance approx 1/d.
    sqrt(d)*s ≈ N(0,1).
    alpha_theory ≈ (1/sqrt(d)) * phi(t) / (1-Phi(t)),
    t = Phi^{-1}(1 - k/V).

This experiment checks:
1. empirical alpha vs theoretical alpha
2. residual ratio
3. self cosine
4. pairwise geometry readback

Run
---
python psg2b_conditional_mean_estimator_audit_v1_0.py
"""

import math
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import norm, pearsonr, spearmanr
from sklearn.metrics import pairwise_distances


EXPERIMENT_ID = "PSG-2B_Conditional_Mean_Estimator_Audit_v1_0"

OUT_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\psg2b_conditional_mean_outputs")

SEED = 42

# Focused grid. No need to reproduce PSG-2A full scaling.
N_QUERIES_LIST = [32, 64]
D_LIST = [128, 256, 512, 1024, 1536]
VOCAB_LIST = [8192, 32768, 65536]
K_LIST = [128, 512, 2000, 5000, 10000]
N_REPEATS = 5

GEOMETRY_MODES = ["gaussian_iid", "sphere_iid"]
SELECTION_MODES = ["topk", "bottomk"]


def set_seed(seed):
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
    H = rng.standard_normal((n, d)).astype(np.float32)
    H = l2_normalize(H)

    if mode == "gaussian_iid":
        R = rng.standard_normal((V, d)).astype(np.float32)
        return H, R

    if mode == "sphere_iid":
        R = rng.standard_normal((V, d)).astype(np.float32)
        R = l2_normalize(R)
        return H, R

    raise ValueError(mode)


def conditional_alpha_theory(d, V, k, mode, selection):
    """
    Computes approximate conditional mean of h·R under top/bottom k.

    q = k/V. For TopK, threshold is upper tail quantile 1-q.
    For BottomK, alpha is negative symmetric.
    """
    q = k / V
    q = min(max(q, 1e-8), 1 - 1e-8)

    # upper tail threshold.
    t = norm.ppf(1.0 - q)
    # lambda(t)=phi(t)/Q(t)
    lam = norm.pdf(t) / max(q, 1e-12)

    if mode == "gaussian_iid":
        alpha = lam
    elif mode == "sphere_iid":
        alpha = lam / math.sqrt(d)
    else:
        raise ValueError(mode)

    if selection == "bottomk":
        alpha = -alpha

    return float(alpha), float(t), float(q)


def select_indices(scores, k, selection):
    n, V = scores.shape
    k = min(k, V)
    if selection == "topk":
        idx = np.argpartition(-scores, kth=k-1, axis=1)[:, :k]
        selected_scores = np.take_along_axis(scores, idx, axis=1)
        order = np.argsort(-selected_scores, axis=1)
        return np.take_along_axis(idx, order, axis=1), np.take_along_axis(selected_scores, order, axis=1)
    if selection == "bottomk":
        idx = np.argpartition(scores, kth=k-1, axis=1)[:, :k]
        selected_scores = np.take_along_axis(scores, idx, axis=1)
        order = np.argsort(selected_scores, axis=1)
        return np.take_along_axis(idx, order, axis=1), np.take_along_axis(selected_scores, order, axis=1)
    raise ValueError(selection)


def run_one(n, V, d, k, mode, selection, repeat_id):
    seed = SEED + 99991 * repeat_id + 1009 * n + 17 * V + 13 * d + k + hash((mode, selection)) % 10007
    rng = np.random.default_rng(seed)

    t0 = time.time()
    H, R = make_H_R(n, V, d, mode, rng)

    scores = H @ R.T
    idx, selected_scores = select_indices(scores, k, selection)
    Z = R[idx].mean(axis=1)

    Hn = l2_normalize(H)
    Zn = l2_normalize(Z)

    # Empirical alpha: projection of Z onto H.
    alpha_emp = np.sum(Z * Hn, axis=1) / np.maximum(np.sum(Hn * Hn, axis=1), 1e-9)
    alpha_emp_mean = float(np.mean(alpha_emp))
    alpha_emp_std = float(np.std(alpha_emp))

    alpha_th, threshold_z, q = conditional_alpha_theory(d, V, k, mode, selection)

    alpha_abs_err = abs(alpha_emp_mean - alpha_th)
    alpha_rel_err = alpha_abs_err / max(abs(alpha_th), 1e-9)

    Z_proj = alpha_emp[:, None] * Hn
    residual = Z - Z_proj
    z_norm = np.linalg.norm(Z, axis=1)
    residual_ratio = np.linalg.norm(residual, axis=1) / np.maximum(z_norm, 1e-9)

    sim = Zn @ Hn.T
    self_cos = np.diag(sim)
    off_mask = ~np.eye(n, dtype=bool)
    off_cos = sim[off_mask]

    D_H = cosine_distance_matrix(H)
    D_Z = cosine_distance_matrix(Z)
    pear, spear = safe_corr(upper_tri_values(D_H), upper_tri_values(D_Z))

    # Projection distribution diagnostics.
    empirical_threshold = float(np.mean(selected_scores[:, -1])) if selection == "topk" else float(np.mean(selected_scores[:, -1]))
    selected_score_mean = float(np.mean(selected_scores))
    selected_score_std = float(np.std(selected_scores))

    return {
        "experiment_id": EXPERIMENT_ID,
        "n_queries": n,
        "V": V,
        "d": d,
        "k": k,
        "k_over_V": k / V,
        "geometry_mode": mode,
        "selection": selection,
        "repeat_id": repeat_id,

        "alpha_emp_mean": alpha_emp_mean,
        "alpha_emp_std": alpha_emp_std,
        "alpha_theory": alpha_th,
        "alpha_abs_err": alpha_abs_err,
        "alpha_rel_err": alpha_rel_err,
        "threshold_z_theory": threshold_z,
        "tail_q": q,

        "z_norm_mean": float(np.mean(z_norm)),
        "z_norm_std": float(np.std(z_norm)),
        "residual_ratio_mean": float(np.mean(residual_ratio)),
        "residual_ratio_std": float(np.std(residual_ratio)),

        "self_cos_mean": float(np.mean(self_cos)),
        "self_cos_std": float(np.std(self_cos)),
        "off_cos_mean": float(np.mean(off_cos)),
        "self_vs_off_margin": float(np.mean(self_cos) - np.mean(off_cos)),

        "topo_pearson": pear,
        "topo_spearman": spear,

        "selected_score_mean": selected_score_mean,
        "selected_score_std": selected_score_std,
        "empirical_edge_score_mean": empirical_threshold,

        "elapsed_sec": time.time() - t0,
    }


def build_verdict(summary):
    lines = []
    lines.append("=" * 80)
    lines.append("PSG-2B Conditional Mean Estimator Audit Verdict")
    lines.append("=" * 80)

    if summary.empty:
        lines.append("NO_VALID_ROWS")
        return "\n".join(lines)

    def mean_query(expr, col):
        sub = summary.query(expr)
        if sub.empty:
            return np.nan
        return float(sub[col].mean())

    topk_spear = mean_query("selection == 'topk'", "topo_spearman")
    bottomk_spear = mean_query("selection == 'bottomk'", "topo_spearman")
    alpha_rel = mean_query("selection == 'topk'", "alpha_rel_err")
    alpha_abs = mean_query("selection == 'topk'", "alpha_abs_err")
    resid = mean_query("selection == 'topk'", "residual_ratio_mean")
    selfcos = mean_query("selection == 'topk'", "self_cos_mean")

    lines.append("\nMain means, TopK:")
    lines.append(f"topk_topo_spearman_mean={topk_spear:.4f}")
    lines.append(f"bottomk_topo_spearman_mean={bottomk_spear:.4f}")
    lines.append(f"topk_alpha_abs_err_mean={alpha_abs:.4f}")
    lines.append(f"topk_alpha_rel_err_mean={alpha_rel:.4f}")
    lines.append(f"topk_residual_ratio_mean={resid:.4f}")
    lines.append(f"topk_self_cos_mean={selfcos:.4f}")

    lines.append("\nTop 20 by alpha relative accuracy among TopK:")
    top_alpha = summary[summary["selection"] == "topk"].sort_values("alpha_rel_err", ascending=True).head(20)
    for _, r in top_alpha.iterrows():
        lines.append(
            f"mode={r['geometry_mode']} n={int(r['n_queries'])} V={int(r['V'])} "
            f"d={int(r['d'])} k={int(r['k'])} q={r['k_over_V']:.4f} "
            f"alpha_emp={r['alpha_emp_mean']:.4f} alpha_th={r['alpha_theory']:.4f} "
            f"relerr={r['alpha_rel_err']:.4f} resid={r['residual_ratio_mean']:.4f} "
            f"spear={r['topo_spearman']:.4f}"
        )

    lines.append("\nTop 20 by geometry readback among TopK:")
    top_geo = summary[summary["selection"] == "topk"].sort_values("topo_spearman", ascending=False).head(20)
    for _, r in top_geo.iterrows():
        lines.append(
            f"mode={r['geometry_mode']} n={int(r['n_queries'])} V={int(r['V'])} "
            f"d={int(r['d'])} k={int(r['k'])} q={r['k_over_V']:.4f} "
            f"spear={r['topo_spearman']:.4f} selfcos={r['self_cos_mean']:.4f} "
            f"alpha_emp={r['alpha_emp_mean']:.4f} alpha_th={r['alpha_theory']:.4f} "
            f"resid={r['residual_ratio_mean']:.4f}"
        )

    lines.append("\nFactor means for TopK:")
    for col in ["geometry_mode", "d", "V", "k"]:
        view = summary[summary["selection"] == "topk"].groupby(col)[
            ["topo_spearman", "alpha_rel_err", "residual_ratio_mean", "self_cos_mean"]
        ].mean().reset_index()
        lines.append(f"\n[{col}]")
        for _, rr in view.iterrows():
            lines.append(
                f"{rr[col]}: spear={rr['topo_spearman']:.4f}, "
                f"relerr={rr['alpha_rel_err']:.4f}, resid={rr['residual_ratio_mean']:.4f}, "
                f"selfcos={rr['self_cos_mean']:.4f}"
            )

    lines.append("\nInterpretation:")
    if np.isfinite(alpha_rel):
        if alpha_rel < 0.10:
            lines.append("PASS-ALPHA: Conditional-mean alpha theory matches empirical alpha closely.")
        elif alpha_rel < 0.25:
            lines.append("WEAK-PASS-ALPHA: Conditional-mean alpha theory approximately matches empirical alpha.")
        else:
            lines.append("WEAK-ALPHA: Simple Gaussian tail alpha is only a rough approximation across this grid.")

    if np.isfinite(topk_spear) and topk_spear > 0.70:
        lines.append("PASS-GEOMETRY: TopK centroid strongly preserves query geometry.")
    elif np.isfinite(topk_spear) and topk_spear > 0.40:
        lines.append("WEAK-PASS-GEOMETRY: TopK centroid moderately preserves query geometry.")
    else:
        lines.append("FAIL-GEOMETRY: TopK centroid does not consistently preserve query geometry.")

    if np.isfinite(resid):
        if resid < 0.30:
            lines.append("PASS-RANK1: Residual is small; rank-1 estimator form is strong.")
        elif resid < 0.60:
            lines.append("WEAK-RANK1: Residual is moderate; estimator is noisy but usable.")
        else:
            lines.append("PAIRWISE-READBACK: Residual is large; use pairwise geometry readback framing.")

    if np.isfinite(bottomk_spear) and np.isfinite(topk_spear) and abs(bottomk_spear - topk_spear) < 0.05:
        lines.append("SUPPORT-SYMMETRY: BottomK and TopK preserve geometry similarly with opposite alpha signs.")

    lines.append("\nSuggested theory sentence:")
    lines.append(
        "Conditioning isotropic candidate vectors on extreme projection onto a query shifts their conditional mean along the query direction; "
        "the TopK centroid is therefore a noisy conditional-mean estimator of the query, producing pairwise geometry readback."
    )
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
    raw_path = OUT_DIR / "psg2b_conditional_mean_raw.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")

    ok = raw[raw.get("topo_spearman", pd.Series(index=raw.index)).notna()].copy()
    group_cols = ["n_queries", "V", "d", "k", "k_over_V", "geometry_mode", "selection"]
    numeric_cols = [
        "alpha_emp_mean", "alpha_emp_std", "alpha_theory",
        "alpha_abs_err", "alpha_rel_err", "threshold_z_theory", "tail_q",
        "z_norm_mean", "z_norm_std", "residual_ratio_mean", "residual_ratio_std",
        "self_cos_mean", "self_cos_std", "off_cos_mean", "self_vs_off_margin",
        "topo_pearson", "topo_spearman",
        "selected_score_mean", "selected_score_std", "empirical_edge_score_mean",
        "elapsed_sec",
    ]
    summary = ok.groupby(group_cols, dropna=False)[numeric_cols].mean().reset_index()
    summary["n_repeats"] = ok.groupby(group_cols, dropna=False).size().values

    summary_path = OUT_DIR / "psg2b_conditional_mean_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    for col in ["selection", "geometry_mode", "d", "V", "k", "n_queries"]:
        view = summary.groupby(col)[
            ["topo_spearman", "alpha_rel_err", "residual_ratio_mean", "self_cos_mean"]
        ].mean().reset_index()
        view = view.sort_values("topo_spearman", ascending=False)
        view.to_csv(OUT_DIR / f"psg2b_factor_view_{col}.csv", index=False, encoding="utf-8-sig")

    tmp = summary.copy()
    tmp["k_over_V_bin"] = pd.cut(tmp["k_over_V"], bins=[0, .002, .005, .01, .02, .05, .10, .25, 1.0])
    kv = tmp.groupby(["selection", "k_over_V_bin"])[
        ["topo_spearman", "alpha_rel_err", "residual_ratio_mean", "self_cos_mean"]
    ].mean().reset_index()
    kv.to_csv(OUT_DIR / "psg2b_factor_view_k_over_V_bin.csv", index=False, encoding="utf-8-sig")

    summary.sort_values("topo_spearman", ascending=False).head(100).to_csv(
        OUT_DIR / "psg2b_top100_geometry.csv", index=False, encoding="utf-8-sig"
    )
    summary[summary["selection"] == "topk"].sort_values("alpha_rel_err", ascending=True).head(100).to_csv(
        OUT_DIR / "psg2b_top100_alpha_fit.csv", index=False, encoding="utf-8-sig"
    )

    verdict = build_verdict(summary)
    verdict_path = OUT_DIR / "psg2b_verdict.txt"
    verdict_path.write_text(verdict, encoding="utf-8")

    print("\n" + verdict)
    print(f"\n[OK] raw: {raw_path}")
    print(f"[OK] summary: {summary_path}")
    print(f"[OK] verdict: {verdict_path}")


if __name__ == "__main__":
    main()
