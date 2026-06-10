# -*- coding: utf-8 -*-
"""
OA-5C.2b: Fiber Coupling Audit
==============================

Purpose
-------
OA-5C.2 supported:

    TopK(H) ≈ high-dimensional tangent neighborhood
    neighborhood = fiber structure + possible fiber coupling

OA-5C.2b tests the coupling part:

    T_x M ≈ sum_i E_i + sum_{i<j} C_ij

Core tests
----------
1. Fiber overlap matrix:
   Measure PCA subspace overlap between task fibers.

2. Additive reconstruction gain:
   Compare reconstruction error using:
       E_i
       E_i + E_j
   If E_i+E_j improves reconstruction of samples near i/j, there is coupling/complementarity.

3. Residual coupling:
   Reconstruct x by E_i, get residual r_i.
   Test whether E_j explains r_i.

4. Coupling graph:
   Build task-fiber coupling graph and summarize strongest pairs.

Input
-----
C:\\Users\\ZH\\Desktop\\AGI\\python_script\\oa5c1_outputs\\oa5c1_topk_center_records.csv

Output
------
oa5c2b_outputs/
    oa5c2b_fiber_overlap_matrix.csv
    oa5c2b_additive_reconstruction_gain.csv
    oa5c2b_residual_coupling_summary.csv
    oa5c2b_coupling_graph_edges.csv
    oa5c2b_verdict_summary.csv
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

warnings.filterwarnings("ignore")

# ============================================================
# PATH CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\ZH\Desktop\AGI\python_script")
INPUT_FILE = BASE_DIR / "oa5c1_outputs" / "oa5c1_topk_center_records.csv"

OUTPUT_DIR = BASE_DIR / "oa5c2b_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_OVERLAP = OUTPUT_DIR / "oa5c2b_fiber_overlap_matrix.csv"
OUT_GAIN = OUTPUT_DIR / "oa5c2b_additive_reconstruction_gain.csv"
OUT_RESID = OUTPUT_DIR / "oa5c2b_residual_coupling_summary.csv"
OUT_GRAPH = OUTPUT_DIR / "oa5c2b_coupling_graph_edges.csv"
OUT_VERDICT = OUTPUT_DIR / "oa5c2b_verdict_summary.csv"

# ============================================================
# CONFIG
# ============================================================

SHALLOW_GROUPS = {
    "L0": [0],
    "L0_2": [0, 1, 2],
    "L0_6": list(range(0, 7)),
}

# Use task fibers as primary explanation directions.
# topic_task fibers are too small for reliable subspace overlap.
FIBER_MODE = "task"

PCA_DIM = 3
ALT_PCA_DIMS = [2, 3, 5]

# A pair is considered coupled if subspace overlap is above this
# and residual/additive tests agree.
OVERLAP_THRESHOLD = 0.20
GAIN_THRESHOLD = 0.05
RESIDUAL_THRESHOLD = 0.05

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================
# HELPERS
# ============================================================

def load_data():
    print("=" * 80)
    print("OA-5C.2b Fiber Coupling Audit")
    print("=" * 80)
    print("INPUT_FILE:", INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    center_cols = [c for c in df.columns if c.startswith("center_")]
    center_cols = sorted(center_cols, key=lambda x: int(x.split("_")[1]))

    if len(center_cols) < 8:
        raise ValueError("No center_* columns found.")

    required = ["prompt_id", "topic", "task", "paraphrase_id", "layer_index", "k"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column {c}")

    print("Rows:", len(df))
    print("Center dim:", len(center_cols))
    print("Prompts:", df["prompt_id"].nunique())
    print("Tasks:", sorted(df["task"].unique().tolist()))
    print("K:", sorted(df["k"].unique().tolist()))
    print("Layers:", sorted(df["layer_index"].unique().tolist()))

    return df, center_cols


def aggregate_vectors(df, center_cols, layers, k):
    sub = df[(df["layer_index"].isin(layers)) & (df["k"] == k)].copy()
    if sub.empty:
        return None, None

    rows = []
    X = []

    for pid, g in sub.groupby("prompt_id"):
        x = g[center_cols].to_numpy(dtype=np.float32).mean(axis=0)
        first = g.iloc[0]
        rows.append({
            "prompt_id": str(pid),
            "topic": str(first["topic"]),
            "task": str(first["task"]),
            "paraphrase_id": str(first["paraphrase_id"]),
        })
        X.append(x)

    meta = pd.DataFrame(rows)
    X = np.stack(X).astype(np.float32)
    return meta, X


def zscore(X):
    return StandardScaler(with_mean=True, with_std=True).fit_transform(X)


def fit_pca_basis(X, dim):
    scaler = StandardScaler(with_mean=True, with_std=True).fit(X)
    Xs = scaler.transform(X)
    ncomp = min(dim, Xs.shape[0] - 1, Xs.shape[1])
    if ncomp < 1:
        return None
    pca = PCA(n_components=ncomp, random_state=RANDOM_SEED).fit(Xs)
    return {
        "scaler": scaler,
        "pca": pca,
        "basis": pca.components_.T,  # [D, r], orthonormal in standardized space
        "mean": pca.mean_,
        "dim": ncomp,
    }


def project_reconstruct(model, X):
    scaler = model["scaler"]
    pca = model["pca"]
    Xs = scaler.transform(X)
    Z = pca.transform(Xs)
    Xhat = pca.inverse_transform(Z)
    return Xs, Xhat


def recon_error(model, X):
    if model is None or len(X) == 0:
        return np.nan
    Xs, Xhat = project_reconstruct(model, X)
    err = np.mean(np.sum((Xs - Xhat) ** 2, axis=1))
    denom = np.mean(np.sum((Xs - Xs.mean(axis=0, keepdims=True)) ** 2, axis=1)) + 1e-12
    return float(err / denom)


def combined_basis_recon_error(model_i, model_j, X_ref_i, X):
    """
    Combine PCA bases from two fibers in a shared standardized coordinate
    using scaler fitted on X_ref_i for comparability.

    Simpler operational version:
      standardize by model_i.scaler,
      take basis_i plus transformed basis_j approximately.
    Since PCA bases live in each fiber's standardized coordinate, exact basis
    combination is not fully canonical. For robustness, we instead fit PCA on
    union of the two fiber samples and evaluate X.
    """
    # This function is left as a conceptual placeholder.
    raise NotImplementedError


def union_recon_error(X_train_a, X_train_b, X_test, dim_each=3):
    """
    Fit PCA on union of fiber A and B with n_components <= 2*dim_each.
    This approximates E_i + E_j.
    """
    X_train = np.vstack([X_train_a, X_train_b])
    dim = min(2 * dim_each, X_train.shape[0] - 1, X_train.shape[1])
    if dim < 1:
        return np.nan

    scaler = StandardScaler(with_mean=True, with_std=True).fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)

    pca = PCA(n_components=dim, random_state=RANDOM_SEED).fit(Xtr)
    Z = pca.transform(Xte)
    Xhat = pca.inverse_transform(Z)

    err = np.mean(np.sum((Xte - Xhat) ** 2, axis=1))
    denom = np.mean(np.sum((Xte - Xtr.mean(axis=0, keepdims=True)) ** 2, axis=1)) + 1e-12
    return float(err / denom)


def principal_overlap(model_a, model_b):
    """
    Return normalized subspace overlap between two PCA bases.

    For orthonormal bases A [D,r], B [D,s]:

        overlap = ||A^T B||_F^2 / min(r,s)

    Range approximately [0,1].
    """
    if model_a is None or model_b is None:
        return np.nan
    A = model_a["basis"]
    B = model_b["basis"]
    r = min(A.shape[1], B.shape[1])
    if r < 1:
        return np.nan
    M = A.T @ B
    return float(np.sum(M ** 2) / r)


def residual_explained_by(model_i, model_j, X):
    """
    Fit E_i, compute residual in i-standardized space.
    Then ask how much of residual energy is explainable by E_j-like directions.

    Because E_j has its own scaler, we use a practical approximation:
      compare err_i vs err_union(i,j) on X.
      residual_gain = err_i - err_union
    """
    err_i = recon_error(model_i, X)
    # union error must be calculated externally with train samples.
    return err_i


def task_groups(meta, X):
    groups = {}
    for task, g in meta.groupby("task"):
        inds = g.index.values
        groups[str(task)] = {
            "idx": inds,
            "X": X[inds],
            "n": len(inds),
        }
    return groups


# ============================================================
# ANALYSIS
# ============================================================

def analyze_overlap(df, center_cols):
    rows = []

    for k in sorted(df["k"].unique()):
        for group_name, layers in SHALLOW_GROUPS.items():
            meta, X = aggregate_vectors(df, center_cols, layers, k)
            if meta is None:
                continue

            groups = task_groups(meta, X)

            for dim in ALT_PCA_DIMS:
                models = {}
                for task, obj in groups.items():
                    if obj["n"] > dim:
                        models[task] = fit_pca_basis(obj["X"], dim)
                    else:
                        models[task] = None

                tasks = sorted(groups.keys())
                for i, ti in enumerate(tasks):
                    for tj in tasks[i+1:]:
                        ov = principal_overlap(models[ti], models[tj])

                        rows.append({
                            "k": int(k),
                            "layer_group": group_name,
                            "dim": int(dim),
                            "fiber_i": ti,
                            "fiber_j": tj,
                            "n_i": int(groups[ti]["n"]),
                            "n_j": int(groups[tj]["n"]),
                            "subspace_overlap": ov,
                        })

    return pd.DataFrame(rows)


def analyze_additive_gain(df, center_cols):
    rows = []

    for k in sorted(df["k"].unique()):
        for group_name, layers in SHALLOW_GROUPS.items():
            meta, X = aggregate_vectors(df, center_cols, layers, k)
            if meta is None:
                continue

            groups = task_groups(meta, X)
            tasks = sorted(groups.keys())

            for dim in ALT_PCA_DIMS:
                models = {}
                for task, obj in groups.items():
                    models[task] = fit_pca_basis(obj["X"], dim) if obj["n"] > dim else None

                for i, ti in enumerate(tasks):
                    Xi = groups[ti]["X"]
                    mi = models[ti]
                    if mi is None:
                        continue

                    err_i_on_i = recon_error(mi, Xi)

                    for tj in tasks:
                        if ti == tj:
                            continue
                        Xj = groups[tj]["X"]
                        mj = models[tj]
                        if mj is None:
                            continue

                        # Reconstruction of Xi by own basis, other basis, and union basis
                        err_j_on_i = recon_error(mj, Xi)
                        err_union_on_i = union_recon_error(Xi, Xj, Xi, dim_each=dim)

                        additive_gain_vs_own = err_i_on_i - err_union_on_i
                        additive_gain_frac = additive_gain_vs_own / (err_i_on_i + 1e-12)

                        # Also test pair neighborhood: Xi+Xj reconstructed by union vs individual average.
                        Xpair = np.vstack([Xi, Xj])
                        err_i_on_pair = recon_error(mi, Xpair)
                        err_j_on_pair = recon_error(mj, Xpair)
                        err_individual_avg = 0.5 * (err_i_on_pair + err_j_on_pair)
                        err_union_on_pair = union_recon_error(Xi, Xj, Xpair, dim_each=dim)
                        pair_gain = err_individual_avg - err_union_on_pair
                        pair_gain_frac = pair_gain / (err_individual_avg + 1e-12)

                        rows.append({
                            "k": int(k),
                            "layer_group": group_name,
                            "dim": int(dim),
                            "target_fiber": ti,
                            "added_fiber": tj,
                            "n_target": int(groups[ti]["n"]),
                            "n_added": int(groups[tj]["n"]),
                            "err_own_on_target": err_i_on_i,
                            "err_added_on_target": err_j_on_i,
                            "err_union_on_target": err_union_on_i,
                            "additive_gain_vs_own": additive_gain_vs_own,
                            "additive_gain_frac": additive_gain_frac,
                            "err_individual_avg_on_pair": err_individual_avg,
                            "err_union_on_pair": err_union_on_pair,
                            "pair_gain": pair_gain,
                            "pair_gain_frac": pair_gain_frac,
                        })

    return pd.DataFrame(rows)


def analyze_residual_coupling(df, center_cols):
    """
    Approximate residual coupling by:
      If E_i reconstructs Xi with err_i,
      and E_i+E_j reconstructs Xi with lower err,
      then residual of i contains directions captured by j/union.
    """
    gain_df = analyze_additive_gain(df, center_cols)
    if gain_df.empty:
        return gain_df

    out = gain_df.copy()
    out["residual_coupling_strength"] = out["additive_gain_frac"]
    out["residual_coupling_positive"] = out["residual_coupling_strength"] > RESIDUAL_THRESHOLD
    return out[[
        "k", "layer_group", "dim",
        "target_fiber", "added_fiber",
        "err_own_on_target", "err_union_on_target",
        "residual_coupling_strength",
        "residual_coupling_positive",
    ]]


def build_graph_edges(overlap_df, gain_df, resid_df):
    if overlap_df.empty or gain_df.empty:
        return pd.DataFrame()

    # Use default PCA_DIM for graph
    ov = overlap_df[overlap_df["dim"] == PCA_DIM].copy()
    gn = gain_df[gain_df["dim"] == PCA_DIM].copy()
    rs = resid_df[resid_df["dim"] == PCA_DIM].copy()

    # Make directed gain/residual, undirected overlap.
    rows = []

    for _, r in gn.iterrows():
        k = r["k"]
        lg = r["layer_group"]
        ti = r["target_fiber"]
        tj = r["added_fiber"]

        # overlap lookup undirected
        o1 = ov[
            (ov["k"] == k)
            & (ov["layer_group"] == lg)
            & (
                ((ov["fiber_i"] == ti) & (ov["fiber_j"] == tj))
                | ((ov["fiber_i"] == tj) & (ov["fiber_j"] == ti))
            )
        ]
        overlap = float(o1["subspace_overlap"].iloc[0]) if len(o1) else np.nan

        res = rs[
            (rs["k"] == k)
            & (rs["layer_group"] == lg)
            & (rs["target_fiber"] == ti)
            & (rs["added_fiber"] == tj)
        ]
        residual_strength = float(res["residual_coupling_strength"].iloc[0]) if len(res) else np.nan

        # Composite coupling score:
        # overlap captures shared directions;
        # pair_gain captures complementarity;
        # residual_strength captures whether j explains i residual.
        vals = []
        if np.isfinite(overlap):
            vals.append(max(0.0, overlap))
        if np.isfinite(r["pair_gain_frac"]):
            vals.append(max(0.0, float(r["pair_gain_frac"])))
        if np.isfinite(residual_strength):
            vals.append(max(0.0, residual_strength))

        score = float(np.mean(vals)) if vals else np.nan

        rows.append({
            "k": int(k),
            "layer_group": lg,
            "source_fiber": tj,
            "target_fiber": ti,
            "subspace_overlap": overlap,
            "pair_gain_frac": float(r["pair_gain_frac"]),
            "residual_coupling_strength": residual_strength,
            "coupling_score": score,
            "coupled_by_overlap": bool(np.isfinite(overlap) and overlap > OVERLAP_THRESHOLD),
            "coupled_by_gain": bool(np.isfinite(r["pair_gain_frac"]) and r["pair_gain_frac"] > GAIN_THRESHOLD),
            "coupled_by_residual": bool(np.isfinite(residual_strength) and residual_strength > RESIDUAL_THRESHOLD),
        })

    return pd.DataFrame(rows)


def verdict_summary(overlap_df, gain_df, resid_df, graph_df):
    rows = []

    if graph_df.empty:
        return pd.DataFrame()

    for (k, lg), g in graph_df.groupby(["k", "layer_group"]):
        ov = overlap_df[(overlap_df["k"] == k) & (overlap_df["layer_group"] == lg) & (overlap_df["dim"] == PCA_DIM)]
        gn = gain_df[(gain_df["k"] == k) & (gain_df["layer_group"] == lg) & (gain_df["dim"] == PCA_DIM)]
        rs = resid_df[(resid_df["k"] == k) & (resid_df["layer_group"] == lg) & (resid_df["dim"] == PCA_DIM)]

        mean_overlap = float(ov["subspace_overlap"].mean()) if len(ov) else np.nan
        max_overlap = float(ov["subspace_overlap"].max()) if len(ov) else np.nan
        overlap_frac = float((ov["subspace_overlap"] > OVERLAP_THRESHOLD).mean()) if len(ov) else np.nan

        mean_pair_gain = float(gn["pair_gain_frac"].mean()) if len(gn) else np.nan
        max_pair_gain = float(gn["pair_gain_frac"].max()) if len(gn) else np.nan
        gain_frac = float((gn["pair_gain_frac"] > GAIN_THRESHOLD).mean()) if len(gn) else np.nan

        mean_resid = float(rs["residual_coupling_strength"].mean()) if len(rs) else np.nan
        max_resid = float(rs["residual_coupling_strength"].max()) if len(rs) else np.nan
        resid_frac = float((rs["residual_coupling_strength"] > RESIDUAL_THRESHOLD).mean()) if len(rs) else np.nan

        mean_score = float(g["coupling_score"].mean())
        max_score = float(g["coupling_score"].max())

        n_edges = int(len(g))
        strong_edges = int((g["coupling_score"] > 0.15).sum())
        very_strong_edges = int((g["coupling_score"] > 0.25).sum())

        # Verdict logic
        # We want evidence of non-zero coupling, not necessarily all pairs strongly coupled.
        if (overlap_frac > 0.25 and (gain_frac > 0.25 or resid_frac > 0.25)) or very_strong_edges >= 2:
            verdict = "PASS-Strong: fiber coupling field detected"
        elif (overlap_frac > 0.10 or gain_frac > 0.10 or resid_frac > 0.10) and strong_edges >= 1:
            verdict = "PASS-Moderate: partial fiber coupling"
        elif mean_score > 0.05:
            verdict = "PASS-Lite/Mixed: weak coupling signal"
        else:
            verdict = "FAIL/Mixed"

        rows.append({
            "k": int(k),
            "layer_group": lg,
            "dim": PCA_DIM,
            "mean_subspace_overlap": mean_overlap,
            "max_subspace_overlap": max_overlap,
            "overlap_edge_frac": overlap_frac,
            "mean_pair_gain_frac": mean_pair_gain,
            "max_pair_gain_frac": max_pair_gain,
            "gain_edge_frac": gain_frac,
            "mean_residual_coupling": mean_resid,
            "max_residual_coupling": max_resid,
            "residual_edge_frac": resid_frac,
            "mean_coupling_score": mean_score,
            "max_coupling_score": max_score,
            "n_directed_edges": n_edges,
            "n_strong_edges_score_gt_0p15": strong_edges,
            "n_very_strong_edges_score_gt_0p25": very_strong_edges,
            "verdict": verdict,
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    df, center_cols = load_data()

    print("\nRunning fiber overlap analysis...")
    overlap_df = analyze_overlap(df, center_cols)
    overlap_df.to_csv(OUT_OVERLAP, index=False, encoding="utf-8-sig")
    print("Saved:", OUT_OVERLAP)

    print("\nRunning additive reconstruction gain analysis...")
    gain_df = analyze_additive_gain(df, center_cols)
    gain_df.to_csv(OUT_GAIN, index=False, encoding="utf-8-sig")
    print("Saved:", OUT_GAIN)

    print("\nRunning residual coupling analysis...")
    resid_df = analyze_residual_coupling(df, center_cols)
    resid_df.to_csv(OUT_RESID, index=False, encoding="utf-8-sig")
    print("Saved:", OUT_RESID)

    print("\nBuilding coupling graph...")
    graph_df = build_graph_edges(overlap_df, gain_df, resid_df)
    graph_df.to_csv(OUT_GRAPH, index=False, encoding="utf-8-sig")
    print("Saved:", OUT_GRAPH)

    print("\nBuilding verdict...")
    verdict_df = verdict_summary(overlap_df, gain_df, resid_df, graph_df)
    verdict_df.to_csv(OUT_VERDICT, index=False, encoding="utf-8-sig")
    print("Saved:", OUT_VERDICT)

    print("\n=== OA-5C.2b VERDICT PREVIEW ===")
    if len(verdict_df):
        print(verdict_df.to_string(index=False))
    else:
        print("No verdict rows generated.")

    print("\nTop coupling edges preview:")
    if len(graph_df):
        preview = graph_df.sort_values("coupling_score", ascending=False).head(20)
        print(preview.to_string(index=False))
    else:
        print("No graph edges.")

    print("\nDone.")


if __name__ == "__main__":
    main()
