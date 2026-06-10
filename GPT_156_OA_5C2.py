# -*- coding: utf-8 -*-
"""
OA-5C.2: High-Dimensional Tangent + Fiber Linearity Audit
=========================================================

Hypothesis
----------
The shallow TopK neighborhood is not a low-dimensional tangent plane.

Instead:

    TopK(H) ≈ high-dimensional local tangent neighborhood

and this high-dimensional neighborhood is decomposed into multiple
"explanation-direction fibers":

    T_x M ≈ union / direct-sum of E_i

Expected pattern:
    1. Global neighborhood dimension is high.
    2. Same task/fiber is more locally linear than mixed tasks.
    3. Same-task reconstruction by its own fiber PCA is better than
       reconstruction by mismatched task fibers.
    4. Cross-fiber geometry shows nonlinear / anisotropic separation.

Inputs
------
Uses OA-5C.1 data:

    C:\\Users\\ZH\\Desktop\\AGI\\python_script\\oa5c1_outputs\\oa5c1_topk_center_records.csv

Expected columns:
    prompt_id, topic, task, paraphrase_id, layer_index, k,
    center_0 ... center_1535

Outputs
-------
    oa5c2_outputs/oa5c2_global_dimension_summary.csv
    oa5c2_outputs/oa5c2_fiber_dimension_summary.csv
    oa5c2_outputs/oa5c2_fiber_projection_summary.csv
    oa5c2_outputs/oa5c2_fiber_distance_summary.csv
    oa5c2_outputs/oa5c2_verdict_summary.csv
"""

import gc
import json
import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

# ============================================================
# PATH CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\ZH\Desktop\AGI\python_script")
INPUT_FILE = BASE_DIR / "oa5c1_outputs" / "oa5c1_topk_center_records.csv"

OUTPUT_DIR = BASE_DIR / "oa5c2_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_GLOBAL_DIM = OUTPUT_DIR / "oa5c2_global_dimension_summary.csv"
OUT_FIBER_DIM = OUTPUT_DIR / "oa5c2_fiber_dimension_summary.csv"
OUT_PROJECTION = OUTPUT_DIR / "oa5c2_fiber_projection_summary.csv"
OUT_DISTANCE = OUTPUT_DIR / "oa5c2_fiber_distance_summary.csv"
OUT_VERDICT = OUTPUT_DIR / "oa5c2_verdict_summary.csv"

# ============================================================
# CONFIG
# ============================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

SHALLOW_GROUPS = {
    "L0": [0],
    "L0_2": [0, 1, 2],
    "L0_6": list(range(0, 7)),
}

PCA_DIMS = [2, 3, 5, 8, 12, 16, 24, 32, 48, 64]
FIBER_PCA_DIMS = [2, 3, 4, 5]

# A fiber is topic+task by default.
# A task fiber ignores topic, useful for cross-topic task direction.
FIBER_MODES = ["topic_task", "task_only"]

# ============================================================
# HELPERS
# ============================================================

def load_data():
    print("=" * 80)
    print("OA-5C.2 High-Dimensional Tangent + Fiber Linearity Audit")
    print("=" * 80)
    print("INPUT_FILE:", INPUT_FILE)
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    center_cols = [c for c in df.columns if c.startswith("center_")]
    if len(center_cols) < 8:
        raise ValueError(
            f"Cannot find center columns. Found {len(center_cols)} columns starting with center_."
        )

    # Sort center columns numerically
    center_cols = sorted(center_cols, key=lambda x: int(x.split("_")[1]))

    required = ["prompt_id", "topic", "task", "paraphrase_id", "layer_index", "k"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column {c}. Available: {list(df.columns)[:50]}")

    print("Rows:", len(df))
    print("Center dim:", len(center_cols))
    print("Layers:", sorted(df["layer_index"].unique().tolist()))
    print("K:", sorted(df["k"].unique().tolist()))
    print("Prompts:", df["prompt_id"].nunique())

    return df, center_cols


def aggregate_vectors(df, center_cols, layers, k):
    """
    Aggregate center vectors across selected layers by prompt_id.
    Returns meta dataframe and X matrix.
    """
    sub = df[(df["layer_index"].isin(layers)) & (df["k"] == k)].copy()
    if sub.empty:
        return None, None

    rows = []
    X = []

    for pid, g in sub.groupby("prompt_id"):
        xv = g[center_cols].to_numpy(dtype=np.float32).mean(axis=0)
        first = g.iloc[0]
        rows.append({
            "prompt_id": str(pid),
            "topic": str(first["topic"]),
            "task": str(first["task"]),
            "paraphrase_id": str(first["paraphrase_id"]),
        })
        X.append(xv)

    meta = pd.DataFrame(rows)
    X = np.stack(X).astype(np.float32)
    return meta, X


def standardize(X):
    return StandardScaler(with_mean=True, with_std=True).fit_transform(X)


def pca_stats(X, dims):
    """
    X should be standardized or raw; this function standardizes internally.
    """
    Xz = standardize(X)
    nmax = min(Xz.shape[0] - 1, Xz.shape[1])
    if nmax < 1:
        return {}

    pca = PCA(n_components=nmax, random_state=RANDOM_SEED)
    pca.fit(Xz)

    evr = pca.explained_variance_ratio_
    csum = np.cumsum(evr)

    out = {
        "n_samples": int(Xz.shape[0]),
        "ambient_dim": int(Xz.shape[1]),
        "pc1": float(csum[0]) if len(csum) >= 1 else np.nan,
        "pc1_pc2": float(csum[1]) if len(csum) >= 2 else np.nan,
        "pc1_pc2_pc3": float(csum[2]) if len(csum) >= 3 else np.nan,
        "effective_dim_80": int(np.searchsorted(csum, 0.80) + 1) if len(csum) else np.nan,
        "effective_dim_90": int(np.searchsorted(csum, 0.90) + 1) if len(csum) else np.nan,
        "effective_dim_95": int(np.searchsorted(csum, 0.95) + 1) if len(csum) else np.nan,
        "participation_ratio": float((np.sum(evr) ** 2) / (np.sum(evr ** 2) + 1e-12)),
    }

    for d in dims:
        if d <= len(csum):
            out[f"evr_{d}"] = float(csum[d - 1])
        else:
            out[f"evr_{d}"] = np.nan

    return out


def pca_basis(X, dim):
    Xz = standardize(X)
    ncomp = min(dim, Xz.shape[0] - 1, Xz.shape[1])
    if ncomp < 1:
        return None, None, None
    scaler = StandardScaler(with_mean=True, with_std=True).fit(X)
    Xs = scaler.transform(X)
    pca = PCA(n_components=ncomp, random_state=RANDOM_SEED).fit(Xs)
    return scaler, pca.components_, pca.mean_


def reconstruction_error_to_basis(X_train, X_test, dim):
    """
    Fit PCA on X_train, reconstruct X_test.
    Return relative reconstruction error.
    """
    if len(X_train) <= dim or len(X_test) == 0:
        return np.nan

    scaler = StandardScaler(with_mean=True, with_std=True).fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)

    ncomp = min(dim, Xtr.shape[0] - 1, Xtr.shape[1])
    if ncomp < 1:
        return np.nan

    pca = PCA(n_components=ncomp, random_state=RANDOM_SEED).fit(Xtr)
    Z = pca.transform(Xte)
    Xhat = pca.inverse_transform(Z)

    err = np.mean(np.sum((Xte - Xhat) ** 2, axis=1))
    denom = np.mean(np.sum((Xte - Xtr.mean(axis=0, keepdims=True)) ** 2, axis=1)) + 1e-12
    return float(err / denom)


def make_fiber_label(meta, mode):
    if mode == "topic_task":
        return meta["topic"].astype(str) + "::" + meta["task"].astype(str)
    if mode == "task_only":
        return meta["task"].astype(str)
    raise ValueError(mode)


def pair_distance_categories(meta, X):
    Xz = standardize(X)
    D = pairwise_distances(Xz, metric="euclidean")

    vals = {
        "same_topic_same_task": [],
        "diff_topic_same_task": [],
        "same_topic_diff_task": [],
        "diff_topic_diff_task": [],
    }

    n = len(meta)
    topics = meta["topic"].astype(str).values
    tasks = meta["task"].astype(str).values

    for i in range(n):
        for j in range(i + 1, n):
            same_topic = topics[i] == topics[j]
            same_task = tasks[i] == tasks[j]
            d = float(D[i, j])
            if same_topic and same_task:
                vals["same_topic_same_task"].append(d)
            elif (not same_topic) and same_task:
                vals["diff_topic_same_task"].append(d)
            elif same_topic and (not same_task):
                vals["same_topic_diff_task"].append(d)
            else:
                vals["diff_topic_diff_task"].append(d)

    out = {}
    for k, v in vals.items():
        out[f"D_{k}"] = float(np.mean(v)) if v else np.nan

    out["lift_diff_topic_same_task_over_same_topic_same_task"] = (
        out["D_diff_topic_same_task"] / (out["D_same_topic_same_task"] + 1e-12)
        if np.isfinite(out["D_same_topic_same_task"]) else np.nan
    )
    out["lift_same_topic_diff_task_over_same_topic_same_task"] = (
        out["D_same_topic_diff_task"] / (out["D_same_topic_same_task"] + 1e-12)
        if np.isfinite(out["D_same_topic_same_task"]) else np.nan
    )
    return out


def knn_fiber_purity(meta, X, label_col, n_neighbors=8):
    Xz = standardize(X)
    n = len(Xz)
    if n <= 2:
        return np.nan, np.nan

    nn = min(n_neighbors + 1, n)
    nbrs = NearestNeighbors(n_neighbors=nn, metric="euclidean").fit(Xz)
    dist, ind = nbrs.kneighbors(Xz)

    labels = meta[label_col].astype(str).values
    same_edges = 0
    total_edges = 0

    for i in range(n):
        for j in ind[i]:
            if i == j:
                continue
            total_edges += 1
            if labels[i] == labels[j]:
                same_edges += 1

    purity = same_edges / max(total_edges, 1)
    freqs = meta[label_col].value_counts(normalize=True).values
    random_purity = float(np.sum(freqs ** 2))
    lift = purity / (random_purity + 1e-12)
    return float(purity), float(lift)


# ============================================================
# ANALYSES
# ============================================================

def global_dimension_analysis(df, center_cols):
    rows = []

    for k in sorted(df["k"].unique()):
        for group_name, layers in SHALLOW_GROUPS.items():
            meta, X = aggregate_vectors(df, center_cols, layers, k)
            if meta is None:
                continue

            st = pca_stats(X, PCA_DIMS)
            dist_st = pair_distance_categories(meta, X)

            # task-only and topic-task purity
            tmp = meta.copy()
            tmp["topic_task"] = tmp["topic"] + "::" + tmp["task"]
            p_task, lift_task = knn_fiber_purity(tmp, X, "task", n_neighbors=8)
            p_tt, lift_tt = knn_fiber_purity(tmp, X, "topic_task", n_neighbors=8)

            rows.append({
                "k": int(k),
                "layer_group": group_name,
                **st,
                **dist_st,
                "knn_task_purity": p_task,
                "knn_task_purity_lift": lift_task,
                "knn_topic_task_purity": p_tt,
                "knn_topic_task_purity_lift": lift_tt,
            })

    return pd.DataFrame(rows)


def fiber_dimension_analysis(df, center_cols):
    rows = []

    for k in sorted(df["k"].unique()):
        for group_name, layers in SHALLOW_GROUPS.items():
            meta, X = aggregate_vectors(df, center_cols, layers, k)
            if meta is None:
                continue

            for mode in FIBER_MODES:
                meta2 = meta.copy()
                meta2["fiber"] = make_fiber_label(meta2, mode)

                for fiber, g in meta2.groupby("fiber"):
                    inds = g.index.values
                    if len(inds) < 4:
                        continue
                    Xg = X[inds]
                    st = pca_stats(Xg, PCA_DIMS)

                    rows.append({
                        "k": int(k),
                        "layer_group": group_name,
                        "fiber_mode": mode,
                        "fiber": fiber,
                        "n_fiber_samples": len(inds),
                        "topic_count": int(g["topic"].nunique()),
                        "task_count": int(g["task"].nunique()),
                        **st,
                    })

    return pd.DataFrame(rows)


def fiber_projection_analysis(df, center_cols):
    """
    For each fiber, fit PCA basis on that fiber.
    Compare reconstruction:
      own fiber samples -> own PCA
      same task but different topic -> own PCA
      same topic but different task -> own PCA
      different task/topic -> own PCA

    If explanation-direction fibers exist:
      own reconstruction error should be much lower.
      task_only fiber basis should generalize across topics better than across tasks.
    """
    rows = []

    for k in sorted(df["k"].unique()):
        for group_name, layers in SHALLOW_GROUPS.items():
            meta, X = aggregate_vectors(df, center_cols, layers, k)
            if meta is None:
                continue

            for dim in FIBER_PCA_DIMS:
                # topic_task fibers
                meta_tt = meta.copy()
                meta_tt["fiber"] = make_fiber_label(meta_tt, "topic_task")

                for fiber, g in meta_tt.groupby("fiber"):
                    own_idx = g.index.values
                    if len(own_idx) <= dim:
                        continue

                    own_topic = g["topic"].iloc[0]
                    own_task = g["task"].iloc[0]

                    same_task_diff_topic_idx = meta_tt[
                        (meta_tt["task"] == own_task) & (meta_tt["topic"] != own_topic)
                    ].index.values
                    same_topic_diff_task_idx = meta_tt[
                        (meta_tt["topic"] == own_topic) & (meta_tt["task"] != own_task)
                    ].index.values
                    diff_both_idx = meta_tt[
                        (meta_tt["topic"] != own_topic) & (meta_tt["task"] != own_task)
                    ].index.values

                    err_own = reconstruction_error_to_basis(X[own_idx], X[own_idx], dim)
                    err_same_task_diff_topic = reconstruction_error_to_basis(
                        X[own_idx], X[same_task_diff_topic_idx], dim
                    )
                    err_same_topic_diff_task = reconstruction_error_to_basis(
                        X[own_idx], X[same_topic_diff_task_idx], dim
                    )
                    err_diff_both = reconstruction_error_to_basis(
                        X[own_idx], X[diff_both_idx], dim
                    )

                    rows.append({
                        "k": int(k),
                        "layer_group": group_name,
                        "basis_mode": "topic_task",
                        "basis_fiber": fiber,
                        "dim": int(dim),
                        "n_own": int(len(own_idx)),
                        "own_topic": own_topic,
                        "own_task": own_task,
                        "err_own": err_own,
                        "err_same_task_diff_topic": err_same_task_diff_topic,
                        "err_same_topic_diff_task": err_same_topic_diff_task,
                        "err_diff_both": err_diff_both,
                        "ratio_same_task_diff_topic_over_own": err_same_task_diff_topic / (err_own + 1e-12) if np.isfinite(err_own) else np.nan,
                        "ratio_same_topic_diff_task_over_own": err_same_topic_diff_task / (err_own + 1e-12) if np.isfinite(err_own) else np.nan,
                        "ratio_diff_both_over_own": err_diff_both / (err_own + 1e-12) if np.isfinite(err_own) else np.nan,
                    })

                # task_only fibers
                meta_task = meta.copy()
                meta_task["fiber"] = make_fiber_label(meta_task, "task_only")

                for task, g in meta_task.groupby("fiber"):
                    own_idx = g.index.values
                    if len(own_idx) <= dim:
                        continue

                    diff_task_idx = meta_task[meta_task["task"] != task].index.values
                    err_own = reconstruction_error_to_basis(X[own_idx], X[own_idx], dim)
                    err_diff_task = reconstruction_error_to_basis(X[own_idx], X[diff_task_idx], dim)

                    rows.append({
                        "k": int(k),
                        "layer_group": group_name,
                        "basis_mode": "task_only",
                        "basis_fiber": task,
                        "dim": int(dim),
                        "n_own": int(len(own_idx)),
                        "own_topic": "ALL",
                        "own_task": task,
                        "err_own": err_own,
                        "err_same_task_diff_topic": np.nan,
                        "err_same_topic_diff_task": np.nan,
                        "err_diff_both": err_diff_task,
                        "ratio_same_task_diff_topic_over_own": np.nan,
                        "ratio_same_topic_diff_task_over_own": np.nan,
                        "ratio_diff_both_over_own": err_diff_task / (err_own + 1e-12) if np.isfinite(err_own) else np.nan,
                    })

    return pd.DataFrame(rows)


def distance_analysis(df, center_cols):
    rows = []

    for k in sorted(df["k"].unique()):
        for group_name, layers in SHALLOW_GROUPS.items():
            meta, X = aggregate_vectors(df, center_cols, layers, k)
            if meta is None:
                continue

            st = pair_distance_categories(meta, X)
            rows.append({
                "k": int(k),
                "layer_group": group_name,
                **st,
            })

    return pd.DataFrame(rows)


def verdict_summary(global_df, fiber_dim_df, proj_df, dist_df):
    rows = []

    for k in sorted(global_df["k"].unique()):
        for group_name in sorted(global_df["layer_group"].unique()):
            gd = global_df[(global_df["k"] == k) & (global_df["layer_group"] == group_name)]
            if gd.empty:
                continue
            gd = gd.iloc[0]

            fd = fiber_dim_df[
                (fiber_dim_df["k"] == k)
                & (fiber_dim_df["layer_group"] == group_name)
                & (fiber_dim_df["fiber_mode"] == "topic_task")
            ]

            ft = fiber_dim_df[
                (fiber_dim_df["k"] == k)
                & (fiber_dim_df["layer_group"] == group_name)
                & (fiber_dim_df["fiber_mode"] == "task_only")
            ]

            pr = proj_df[
                (proj_df["k"] == k)
                & (proj_df["layer_group"] == group_name)
                & (proj_df["basis_mode"] == "topic_task")
                & (proj_df["dim"] == 3)
            ]

            pr_task = proj_df[
                (proj_df["k"] == k)
                & (proj_df["layer_group"] == group_name)
                & (proj_df["basis_mode"] == "task_only")
                & (proj_df["dim"] == 3)
            ]

            global_dim90 = float(gd["effective_dim_90"])
            global_pc3 = float(gd["pc1_pc2_pc3"])
            task_lift = float(gd["lift_diff_topic_same_task_over_same_topic_same_task"])
            same_topic_diff_task_lift = float(gd["lift_same_topic_diff_task_over_same_topic_same_task"])
            knn_task_lift = float(gd["knn_task_purity_lift"])
            knn_tt_lift = float(gd["knn_topic_task_purity_lift"])

            mean_fiber_dim90 = float(fd["effective_dim_90"].mean()) if len(fd) else np.nan
            mean_fiber_pc3 = float(fd["pc1_pc2_pc3"].mean()) if len(fd) else np.nan
            mean_task_fiber_dim90 = float(ft["effective_dim_90"].mean()) if len(ft) else np.nan
            mean_task_fiber_pc3 = float(ft["pc1_pc2_pc3"].mean()) if len(ft) else np.nan

            own_err = float(pr["err_own"].mean()) if len(pr) else np.nan
            same_task_ratio = float(pr["ratio_same_task_diff_topic_over_own"].mean()) if len(pr) else np.nan
            same_topic_diff_task_ratio = float(pr["ratio_same_topic_diff_task_over_own"].mean()) if len(pr) else np.nan
            diff_both_ratio = float(pr["ratio_diff_both_over_own"].mean()) if len(pr) else np.nan

            task_own_err = float(pr_task["err_own"].mean()) if len(pr_task) else np.nan
            task_diff_ratio = float(pr_task["ratio_diff_both_over_own"].mean()) if len(pr_task) else np.nan

            # Evidence flags
            high_global_dim = bool(global_dim90 > 8 or global_pc3 < 0.70)
            fiber_more_linear = bool(
                np.isfinite(mean_fiber_dim90) and mean_fiber_dim90 < global_dim90
            )
            same_task_generalizes = bool(
                np.isfinite(same_task_ratio)
                and np.isfinite(same_topic_diff_task_ratio)
                and same_task_ratio < same_topic_diff_task_ratio
            )
            task_basis_separates = bool(np.isfinite(task_diff_ratio) and task_diff_ratio > 1.10)
            local_task_continuity = bool(task_lift > 1.03 and knn_task_lift > 1.05)

            score = sum([
                high_global_dim,
                fiber_more_linear,
                same_task_generalizes,
                task_basis_separates,
                local_task_continuity,
            ])

            if score >= 4:
                verdict = "PASS-Strong: high-dimensional tangent with fiber-local linearity"
            elif score == 3:
                verdict = "PASS-Moderate: partial fiber tangent structure"
            elif score == 2:
                verdict = "PASS-Lite/Mixed"
            else:
                verdict = "FAIL/Mixed"

            rows.append({
                "k": int(k),
                "layer_group": group_name,
                "global_effective_dim90": global_dim90,
                "global_pc1pc2pc3": global_pc3,
                "mean_topic_task_fiber_dim90": mean_fiber_dim90,
                "mean_topic_task_fiber_pc1pc2pc3": mean_fiber_pc3,
                "mean_task_only_fiber_dim90": mean_task_fiber_dim90,
                "mean_task_only_fiber_pc1pc2pc3": mean_task_fiber_pc3,
                "D_lift_diff_topic_same_task_over_same_topic_same_task": task_lift,
                "D_lift_same_topic_diff_task_over_same_topic_same_task": same_topic_diff_task_lift,
                "knn_task_lift": knn_task_lift,
                "knn_topic_task_lift": knn_tt_lift,
                "projection_err_own_dim3": own_err,
                "projection_ratio_same_task_diff_topic_over_own_dim3": same_task_ratio,
                "projection_ratio_same_topic_diff_task_over_own_dim3": same_topic_diff_task_ratio,
                "projection_ratio_diff_both_over_own_dim3": diff_both_ratio,
                "task_basis_err_own_dim3": task_own_err,
                "task_basis_ratio_diff_task_over_own_dim3": task_diff_ratio,
                "high_global_dim": high_global_dim,
                "fiber_more_linear_than_global": fiber_more_linear,
                "same_task_generalizes_better_than_same_topic_diff_task": same_task_generalizes,
                "task_basis_separates_diff_task": task_basis_separates,
                "local_task_continuity": local_task_continuity,
                "score": int(score),
                "verdict": verdict,
            })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    df, center_cols = load_data()

    print("\nRunning global dimension analysis...")
    global_df = global_dimension_analysis(df, center_cols)
    global_df.to_csv(OUT_GLOBAL_DIM, index=False, encoding="utf-8-sig")
    print("Saved:", OUT_GLOBAL_DIM)

    print("\nRunning fiber dimension analysis...")
    fiber_df = fiber_dimension_analysis(df, center_cols)
    fiber_df.to_csv(OUT_FIBER_DIM, index=False, encoding="utf-8-sig")
    print("Saved:", OUT_FIBER_DIM)

    print("\nRunning fiber projection analysis...")
    proj_df = fiber_projection_analysis(df, center_cols)
    proj_df.to_csv(OUT_PROJECTION, index=False, encoding="utf-8-sig")
    print("Saved:", OUT_PROJECTION)

    print("\nRunning fiber distance analysis...")
    dist_df = distance_analysis(df, center_cols)
    dist_df.to_csv(OUT_DISTANCE, index=False, encoding="utf-8-sig")
    print("Saved:", OUT_DISTANCE)

    print("\nBuilding verdict...")
    verdict_df = verdict_summary(global_df, fiber_df, proj_df, dist_df)
    verdict_df.to_csv(OUT_VERDICT, index=False, encoding="utf-8-sig")
    print("Saved:", OUT_VERDICT)

    print("\n=== OA-5C.2 VERDICT PREVIEW ===")
    print(verdict_df.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
