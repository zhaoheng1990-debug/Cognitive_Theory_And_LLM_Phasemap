# -*- coding: utf-8 -*-
"""
PSG-1B Gaussian High-Dimensional Selection Null

Goal
----
Validate whether the pipeline

    H_i R^T -> TopK_i -> mean(R_TopK_i)

can recover hidden/query geometry in a purely random high-dimensional setting.

This script does NOT load any LLM, tokenizer, or W matrix.
It is intentionally small and validation-oriented.

Core question
-------------
If H and R are independent Gaussian matrices, does TopK-selected centroid Z_i
preserve pairwise geometry of H_i?

If yes, PSG-1A's surprising result is explained as a general
projection-selection-aggregation geometry, not necessarily learned semantic W.

Default output
--------------
C:\\Users\\ZH\\Desktop\\AGI\\outputs\\psg1b_gaussian_null_outputs

Run
---
python psg1b_gaussian_null_v1_0.py
"""

import math
import json
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

EXPERIMENT_ID = "PSG-1B_Gaussian_HighDim_Selection_Null_v1_0"

OUT_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\psg1b_gaussian_null_outputs")

SEED = 42

# Keep scale small. This is enough to validate the mechanism.
N_QUERIES_LIST = [16, 32, 64]
D_LIST = [256, 768, 1536]
VOCAB_LIST = [8192, 32768]
K_LIST = [256, 1000, 5000]

# Number of independent random seeds per condition.
N_REPEATS = 3

# Main modes.
SELECTION_MODES = ["topk", "randomk", "bottomk"]

AGGREGATIONS = [
    "center",
    "normalized_center",
    "rank_weighted_center",
    "spread_scalar",
]

# Score variants.
# gaussian_iid: H and R independently standard normal, row-normalized.
# anisotropic: shared diagonal covariance, tests anisotropy effect.
# norm_hetero: R has heterogeneous row norms, tests norm confound.
SCORE_GEOMETRY_MODES = ["gaussian_iid", "anisotropic", "norm_hetero"]


# =========================
# 1. Utils
# =========================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def l2_normalize(x, axis=-1, eps=1e-9):
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(denom, eps)


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


def cosine_distance_matrix(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return pairwise_distances(x, metric="cosine")


def euclidean_distance_matrix(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return pairwise_distances(x, metric="euclidean")


def make_H_R(n_queries, vocab, d, mode, rng):
    if mode == "gaussian_iid":
        H = rng.standard_normal((n_queries, d)).astype(np.float32)
        R = rng.standard_normal((vocab, d)).astype(np.float32)
        H = l2_normalize(H)
        R = l2_normalize(R)
        return H, R

    if mode == "anisotropic":
        # Shared diagonal covariance creates a global anisotropic geometry.
        decay = np.linspace(1.0, 0.15, d, dtype=np.float32)
        H = rng.standard_normal((n_queries, d)).astype(np.float32) * decay[None, :]
        R = rng.standard_normal((vocab, d)).astype(np.float32) * decay[None, :]
        H = l2_normalize(H)
        R = l2_normalize(R)
        return H, R

    if mode == "norm_hetero":
        H = rng.standard_normal((n_queries, d)).astype(np.float32)
        R = rng.standard_normal((vocab, d)).astype(np.float32)
        H = l2_normalize(H)
        R = l2_normalize(R)
        # Log-normal row norm heterogeneity.
        scales = rng.lognormal(mean=0.0, sigma=0.6, size=(vocab, 1)).astype(np.float32)
        R = l2_normalize(R) * scales
        return H, R

    raise ValueError(f"Unknown mode: {mode}")


def select_indices(scores, k, mode, rng):
    n, v = scores.shape
    k = min(k, v)

    if mode == "topk":
        return np.argpartition(-scores, kth=k-1, axis=1)[:, :k]

    if mode == "bottomk":
        return np.argpartition(scores, kth=k-1, axis=1)[:, :k]

    if mode == "randomk":
        arr = []
        for _ in range(n):
            arr.append(rng.choice(v, size=k, replace=False))
        return np.stack(arr, axis=0)

    raise ValueError(f"Unknown selection mode: {mode}")


def sort_selected_by_score(idx, scores):
    selected_scores = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-selected_scores, axis=1)
    idx_sorted = np.take_along_axis(idx, order, axis=1)
    scores_sorted = np.take_along_axis(selected_scores, order, axis=1)
    return idx_sorted, scores_sorted


def aggregate(R, idx, selected_scores, mode, rng):
    X = R[idx]  # [n,k,d]
    n, k, d = X.shape

    if mode == "center":
        return X.mean(axis=1)

    if mode == "normalized_center":
        return l2_normalize(X.mean(axis=1))

    if mode == "rank_weighted_center":
        weights = np.linspace(1.0, 0.1, k, dtype=np.float32)
        weights = weights / weights.sum()
        return (X * weights[None, :, None]).sum(axis=1)

    if mode == "spread_scalar":
        c = X.mean(axis=1, keepdims=True)
        spread = np.linalg.norm(X - c, axis=2).mean(axis=1, keepdims=True)
        return spread.astype(np.float32)

    raise ValueError(f"Unknown aggregation: {mode}")


def mean_jaccard(idx):
    vals = []
    for i in range(idx.shape[0]):
        si = set(idx[i].tolist())
        for j in range(i + 1, idx.shape[0]):
            sj = set(idx[j].tolist())
            vals.append(len(si & sj) / max(1, len(si | sj)))
    return float(np.mean(vals)) if vals else np.nan


def run_one_condition(n_queries, vocab, d, k, mode, selection, aggregation, repeat_id):
    seed = SEED + 100000 * repeat_id + 1000 * n_queries + 10 * d + k + hash((mode, selection, aggregation)) % 997
    rng = np.random.default_rng(seed)

    H, R = make_H_R(n_queries, vocab, d, mode, rng)

    t0 = time.time()

    # Scores [n,vocab]. For configured sizes this is small enough.
    scores = H @ R.T

    idx = select_indices(scores, k, selection, rng)

    # Sort top/bottom/random selected indices by score for rank-weighted aggregation.
    idx_sorted, selected_scores = sort_selected_by_score(idx, scores)

    if selection == "bottomk":
        # bottomk sorted descending would put less bottom-like first.
        # For consistency of rank_weighted_center with selected extreme, sort ascending.
        s = np.take_along_axis(scores, idx, axis=1)
        order = np.argsort(s, axis=1)
        idx_sorted = np.take_along_axis(idx, order, axis=1)
        selected_scores = np.take_along_axis(s, order, axis=1)

    Z = aggregate(R, idx_sorted, selected_scores, aggregation, rng)

    D_H = cosine_distance_matrix(H)
    if aggregation == "spread_scalar":
        D_Z = euclidean_distance_matrix(Z)
    else:
        D_Z = cosine_distance_matrix(Z)

    pear, spear = safe_corr(upper_tri_values(D_H), upper_tri_values(D_Z))

    # Direct vector alignment diagnostic: Z should be close to H for topk in Gaussian null.
    if Z.ndim == 2 and Z.shape[1] == H.shape[1]:
        zn = l2_normalize(Z)
        hn = l2_normalize(H)
        self_cos = np.sum(zn * hn, axis=1)
        # off diagonal mean
        off = zn @ hn.T
        mask = ~np.eye(n_queries, dtype=bool)
        off_mean = float(off[mask].mean()) if mask.sum() else np.nan
        self_cos_mean = float(self_cos.mean())
        self_vs_off_margin = self_cos_mean - off_mean
        z_norm_mean = float(np.linalg.norm(Z, axis=1).mean())
    else:
        self_cos_mean = np.nan
        off_mean = np.nan
        self_vs_off_margin = np.nan
        z_norm_mean = np.nan

    elapsed = time.time() - t0

    return {
        "experiment_id": EXPERIMENT_ID,
        "n_queries": n_queries,
        "vocab": vocab,
        "d": d,
        "k": k,
        "geometry_mode": mode,
        "selection": selection,
        "aggregation": aggregation,
        "repeat_id": repeat_id,
        "topo_pearson": pear,
        "topo_spearman": spear,
        "abs_topo_spearman": abs(spear) if np.isfinite(spear) else np.nan,
        "self_cos_mean": self_cos_mean,
        "off_cos_mean": off_mean,
        "self_vs_off_margin": self_vs_off_margin,
        "z_norm_mean": z_norm_mean,
        "mean_selected_jaccard": mean_jaccard(idx_sorted),
        "elapsed_sec": elapsed,
    }


def build_verdict(summary):
    lines = []
    lines.append("=" * 80)
    lines.append("PSG-1B Gaussian High-Dimensional Selection Null Verdict")
    lines.append("=" * 80)

    if summary.empty:
        lines.append("NO_VALID_ROWS")
        return "\n".join(lines)

    def mean_query(expr):
        sub = summary.query(expr)
        if sub.empty:
            return np.nan
        return float(sub["topo_spearman"].mean())

    topk = mean_query("selection == 'topk'")
    randomk = mean_query("selection == 'randomk'")
    bottomk = mean_query("selection == 'bottomk'")
    center = mean_query("aggregation == 'center'")
    spread = mean_query("aggregation == 'spread_scalar'")
    iid_topk_center = mean_query("geometry_mode == 'gaussian_iid' and selection == 'topk' and aggregation == 'center'")
    iid_random_center = mean_query("geometry_mode == 'gaussian_iid' and selection == 'randomk' and aggregation == 'center'")

    lines.append("\nMain means:")
    lines.append(f"topk_mean={topk:.4f}")
    lines.append(f"randomk_mean={randomk:.4f}")
    lines.append(f"bottomk_mean={bottomk:.4f}")
    lines.append(f"center_mean={center:.4f}")
    lines.append(f"spread_scalar_mean={spread:.4f}")
    lines.append(f"gaussian_iid_topk_center_mean={iid_topk_center:.4f}")
    lines.append(f"gaussian_iid_randomk_center_mean={iid_random_center:.4f}")

    lines.append("\nTop 20 conditions:")
    top = summary.sort_values("topo_spearman", ascending=False).head(20)
    for _, r in top.iterrows():
        lines.append(
            f"mode={r['geometry_mode']} n={int(r['n_queries'])} vocab={int(r['vocab'])} "
            f"d={int(r['d'])} k={int(r['k'])} select={r['selection']} agg={r['aggregation']} "
            f"spearman={r['topo_spearman']:.4f} margin={r['self_vs_off_margin']:.4f}"
        )

    lines.append("\nInterpretation:")
    if np.isfinite(iid_topk_center) and iid_topk_center > 0.70:
        lines.append("PASS-STRONG: Pure Gaussian TopK centroid strongly preserves query geometry.")
    elif np.isfinite(iid_topk_center) and iid_topk_center > 0.40:
        lines.append("PASS: Pure Gaussian TopK centroid moderately preserves query geometry.")
    else:
        lines.append("WEAK/FAIL: Pure Gaussian TopK centroid did not strongly preserve query geometry at this scale.")

    if np.isfinite(topk) and np.isfinite(randomk):
        if topk > randomk + 0.20:
            lines.append("PASS: TopK selection strongly exceeds random-K.")
        elif topk > randomk + 0.05:
            lines.append("WEAK-PASS: TopK exceeds random-K.")
        else:
            lines.append("FAIL: TopK does not clearly exceed random-K.")

    if np.isfinite(center) and np.isfinite(spread):
        if center > spread + 0.10:
            lines.append("SUPPORT: Centroid aggregation is the main readback mechanism, not spread-only.")
        else:
            lines.append("MIXED: Spread-only statistics may carry nontrivial geometry.")

    lines.append("\nSuggested next step:")
    lines.append("If PASS-STRONG, update PSG theory: TopK centroid is a high-dimensional estimator of query direction.")
    lines.append("Then run a tiny PSG-1C anisotropy/whitening audit only if Paper1 needs real-W residual decomposition.")
    lines.append("=" * 80)
    return "\n".join(lines)


def main():
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    total = (
        len(N_QUERIES_LIST)
        * len(D_LIST)
        * len(VOCAB_LIST)
        * len(K_LIST)
        * len(SCORE_GEOMETRY_MODES)
        * len(SELECTION_MODES)
        * len(AGGREGATIONS)
        * N_REPEATS
    )
    done = 0
    start = time.time()

    for n in N_QUERIES_LIST:
        for d in D_LIST:
            for vocab in VOCAB_LIST:
                for k in K_LIST:
                    if k >= vocab:
                        continue
                    for mode in SCORE_GEOMETRY_MODES:
                        for selection in SELECTION_MODES:
                            for aggregation in AGGREGATIONS:
                                for rep in range(N_REPEATS):
                                    try:
                                        row = run_one_condition(
                                            n_queries=n,
                                            vocab=vocab,
                                            d=d,
                                            k=k,
                                            mode=mode,
                                            selection=selection,
                                            aggregation=aggregation,
                                            repeat_id=rep,
                                        )
                                        rows.append(row)
                                    except Exception as e:
                                        rows.append({
                                            "experiment_id": EXPERIMENT_ID,
                                            "n_queries": n,
                                            "vocab": vocab,
                                            "d": d,
                                            "k": k,
                                            "geometry_mode": mode,
                                            "selection": selection,
                                            "aggregation": aggregation,
                                            "repeat_id": rep,
                                            "error": repr(e),
                                        })
                                    done += 1
                                    if done % 25 == 0:
                                        elapsed = time.time() - start
                                        print(f"[INFO] progress {done}/{total}; rows={len(rows)}; elapsed={elapsed:.1f}s")

    raw = pd.DataFrame(rows)
    raw_path = OUT_DIR / "psg1b_gaussian_null_raw.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")

    ok = raw[raw.get("topo_spearman", pd.Series(index=raw.index)).notna()].copy()
    group_cols = ["n_queries", "vocab", "d", "k", "geometry_mode", "selection", "aggregation"]
    numeric_cols = [
        "topo_pearson", "topo_spearman", "abs_topo_spearman",
        "self_cos_mean", "off_cos_mean", "self_vs_off_margin",
        "z_norm_mean", "mean_selected_jaccard", "elapsed_sec"
    ]
    summary = ok.groupby(group_cols, dropna=False)[numeric_cols].mean().reset_index()
    summary["n_repeats"] = ok.groupby(group_cols, dropna=False).size().values

    summary_path = OUT_DIR / "psg1b_gaussian_null_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    for col in ["geometry_mode", "selection", "aggregation", "d", "vocab", "k", "n_queries"]:
        view = summary.groupby(col)["topo_spearman"].agg(["mean", "median", "max", "std", "count"]).reset_index()
        view = view.sort_values("mean", ascending=False)
        view.to_csv(OUT_DIR / f"psg1b_factor_view_{col}.csv", index=False, encoding="utf-8-sig")

    top = summary.sort_values("topo_spearman", ascending=False).head(100)
    top.to_csv(OUT_DIR / "psg1b_top100_conditions.csv", index=False, encoding="utf-8-sig")

    verdict = build_verdict(summary)
    verdict_path = OUT_DIR / "psg1b_verdict.txt"
    verdict_path.write_text(verdict, encoding="utf-8")

    print("\n" + verdict)
    print(f"\n[OK] raw: {raw_path}")
    print(f"[OK] summary: {summary_path}")
    print(f"[OK] verdict: {verdict_path}")


if __name__ == "__main__":
    main()
