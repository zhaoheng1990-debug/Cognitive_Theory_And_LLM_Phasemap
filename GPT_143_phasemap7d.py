# ============================================================
# PhaseMap-7D: Operator Manifold Audit
#
# Goal:
#   Audit whether continuous operator coordinates
#       O_l^cont = (op_pc1,...,op_pc10)
#   form a low-dimensional manifold rather than ten unrelated
#   engineering features.
#
# Fixed workflow:
#   Put this script in the same directory as:
#       phasemap6b1_correction_dataset.csv
#   Then run:
#       python phasemap7d_operator_manifold_audit.py
#
# Outputs:
#   phasemap7d_outputs/
#       phasemap7d_summary.json
#       phasemap7d_pca_spectrum.csv
#       phasemap7d_reconstruction_summary.csv
#       phasemap7d_intrinsic_dimension.csv
#       phasemap7d_neighbor_purity.csv
#       phasemap7d_layer_trajectory.csv
#       phasemap7d_embedding_2d.csv
#       phasemap7d_centroids_by_layer.csv
#       phasemap7d_centroids_by_operator.csv
#       phasemap7d_centroids_by_phase_condition.csv
# ============================================================

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.manifold import trustworthiness

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = r"C:\Windows\System32\phasemap6b1_outputs\phasemap6b1_correction_dataset.csv"
OUTPUT_DIR = r"phasemap7d_outputs"

RANDOM_SEED = 42
MAX_TRUST_SAMPLE = 3000
MAX_ID_SAMPLE = 5000
MAX_PURITY_SAMPLE = 5000
KNN_KS = [5, 10, 20, 50]
RECON_DIMS = list(range(1, 11))

OP_COLS = [f"op_pc{i}" for i in range(1, 11)]
META_COLS = [
    "sample_id", "graph_id", "condition", "phase", "layer", "transition",
    "heuristic_operator", "operator_cluster",
]
STATE_COLS = [
    "R_l", "boundary_dist_l", "spread_l", "rank_gap_l",
    "R_prev_delta", "rank_gap_prev_delta", "center_backtrack_init",
    "center_backtrack_prevprev", "R_velocity_reversal", "rank_gap_reversal",
]

# ============================================================
# UTILS
# ============================================================

def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    a = a[mask]
    b = b[mask]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def reconstruction_r2(X_scaled, n_dim):
    pca = PCA(n_components=n_dim, random_state=RANDOM_SEED)
    Z = pca.fit_transform(X_scaled)
    X_hat = pca.inverse_transform(Z)
    ss_res = np.sum((X_scaled - X_hat) ** 2)
    ss_tot = np.sum((X_scaled - X_scaled.mean(axis=0, keepdims=True)) ** 2)
    return float(1.0 - ss_res / ss_tot)


def participation_ratio(evals):
    evals = np.asarray(evals, dtype=float)
    s1 = np.sum(evals)
    s2 = np.sum(evals ** 2)
    if s2 <= 1e-12:
        return np.nan
    return float((s1 ** 2) / s2)


def two_nn_id(X, max_n=MAX_ID_SAMPLE, seed=RANDOM_SEED):
    # Facco et al. TWO-NN intrinsic dimension estimator.
    rng = np.random.default_rng(seed)
    n = len(X)
    if n > max_n:
        idx = rng.choice(n, size=max_n, replace=False)
        Xs = X[idx]
    else:
        Xs = X
    nn = NearestNeighbors(n_neighbors=3, metric="euclidean")
    nn.fit(Xs)
    dist, _ = nn.kneighbors(Xs)
    # dist[:,0] is self; use nearest and second-nearest non-self.
    r1 = np.maximum(dist[:, 1], 1e-12)
    r2 = np.maximum(dist[:, 2], 1e-12)
    mu = r2 / r1
    mu = mu[np.isfinite(mu) & (mu > 1.0)]
    if len(mu) < 100:
        return np.nan, len(mu)
    # MLE: d = 1 / mean(log(mu))
    d_hat = 1.0 / np.mean(np.log(mu))
    return float(d_hat), int(len(mu))


def mle_id_knn(X, k=10, max_n=MAX_ID_SAMPLE, seed=RANDOM_SEED):
    # Levina-Bickel local MLE intrinsic dimension.
    rng = np.random.default_rng(seed)
    n = len(X)
    if n > max_n:
        idx = rng.choice(n, size=max_n, replace=False)
        Xs = X[idx]
    else:
        Xs = X
    k_eff = min(k + 1, len(Xs))
    if k_eff < 4:
        return np.nan, len(Xs)
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(Xs)
    dist, _ = nn.kneighbors(Xs)
    # exclude self distance
    T = np.maximum(dist[:, 1:], 1e-12)
    Tk = T[:, -1]
    logs = np.log(Tk[:, None] / T[:, :-1])
    denom = np.mean(np.sum(logs, axis=1) / max(k_eff - 2, 1))
    if denom <= 1e-12 or not np.isfinite(denom):
        return np.nan, len(Xs)
    return float(1.0 / denom), int(len(Xs))


def neighbor_purity(df, X, labels, k, max_n=MAX_PURITY_SAMPLE, seed=RANDOM_SEED):
    labels = np.asarray(labels)
    valid_idx = np.where(pd.notna(labels))[0]
    if len(valid_idx) < k + 2:
        return np.nan, int(len(valid_idx))
    rng = np.random.default_rng(seed + k)
    if len(valid_idx) > max_n:
        valid_idx = rng.choice(valid_idx, size=max_n, replace=False)
    Xv = X[valid_idx]
    lv = labels[valid_idx]
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(Xv)), metric="euclidean")
    nn.fit(Xv)
    _, ind = nn.kneighbors(Xv)
    neigh = ind[:, 1:]
    same = np.mean(lv[neigh] == lv[:, None], axis=1)
    return float(np.mean(same)), int(len(Xv))


def centroid_table(df, emb_cols, group_cols, min_n=5):
    cols = group_cols + emb_cols + OP_COLS
    tmp = df[cols].dropna(subset=emb_cols)
    g = tmp.groupby(group_cols, dropna=False)
    out = g[emb_cols + OP_COLS].mean().reset_index()
    out["n"] = g.size().values
    out = out[out["n"] >= min_n].copy()
    return out

# ============================================================
# MAIN
# ============================================================

def main():
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    if not Path(INPUT_CSV).exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_CSV}. Put this script in the same directory as phasemap6b1_correction_dataset.csv."
        )

    df = pd.read_csv(INPUT_CSV)
    missing = [c for c in OP_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing operator coordinate columns: {missing}")

    # Clean operator matrix.
    work = df.copy()
    for c in OP_COLS:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    mask = work[OP_COLS].notna().all(axis=1)
    work = work.loc[mask].reset_index(drop=True)

    X_raw = work[OP_COLS].to_numpy(dtype=float)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # PCA full spectrum.
    pca_full = PCA(n_components=len(OP_COLS), random_state=RANDOM_SEED)
    Z_full = pca_full.fit_transform(X)
    eig = pca_full.explained_variance_
    var = pca_full.explained_variance_ratio_
    cum = np.cumsum(var)
    pr = participation_ratio(eig)

    pca_rows = []
    for i, (v, cv, ev) in enumerate(zip(var, cum, eig), start=1):
        row = {
            "component": i,
            "explained_variance_ratio": float(v),
            "cumulative_variance_ratio": float(cv),
            "eigenvalue": float(ev),
        }
        for j, col in enumerate(OP_COLS):
            row[f"loading_{col}"] = float(pca_full.components_[i-1, j])
        pca_rows.append(row)
    pca_df = pd.DataFrame(pca_rows)
    pca_df.to_csv(outdir / "phasemap7d_pca_spectrum.csv", index=False, encoding="utf-8-sig")

    # Reconstruction by dimensions.
    rec_rows = []
    for d in RECON_DIMS:
        r2 = reconstruction_r2(X, d)
        rec_rows.append({"n_dim": d, "reconstruction_r2": r2, "residual_fraction": 1 - r2})
    rec_df = pd.DataFrame(rec_rows)
    rec_df.to_csv(outdir / "phasemap7d_reconstruction_summary.csv", index=False, encoding="utf-8-sig")

    # Intrinsic dimension estimates.
    id_rows = []
    d_two, n_two = two_nn_id(X)
    id_rows.append({"method": "TWO_NN", "k": np.nan, "intrinsic_dim": d_two, "n_used": n_two})
    for k in [5, 10, 20, 30]:
        d_mle, n_mle = mle_id_knn(X, k=k)
        id_rows.append({"method": "Levina_Bickel_MLE", "k": k, "intrinsic_dim": d_mle, "n_used": n_mle})
    id_rows.append({"method": "PCA_participation_ratio", "k": np.nan, "intrinsic_dim": pr, "n_used": len(X)})
    id_df = pd.DataFrame(id_rows)
    id_df.to_csv(outdir / "phasemap7d_intrinsic_dimension.csv", index=False, encoding="utf-8-sig")

    # 2D/3D embeddings from PCA for reliable reproducibility.
    emb = work[META_COLS + OP_COLS].copy()
    for i in range(1, 6):
        emb[f"op_m{i}"] = Z_full[:, i-1]
    # Add state if available for downstream plotting.
    for c in STATE_COLS:
        if c in work.columns:
            emb[c] = work[c]
    emb.to_csv(outdir / "phasemap7d_embedding_2d.csv", index=False, encoding="utf-8-sig")

    # Trustworthiness of low-dimensional PCA embeddings.
    rng = np.random.default_rng(RANDOM_SEED)
    if len(X) > MAX_TRUST_SAMPLE:
        idx = rng.choice(len(X), size=MAX_TRUST_SAMPLE, replace=False)
    else:
        idx = np.arange(len(X))
    trust_rows = []
    for d in [1, 2, 3, 4, 5]:
        Zd = Z_full[:, :d]
        t = trustworthiness(X[idx], Zd[idx], n_neighbors=10, metric="euclidean")
        trust_rows.append({"embedding": f"PCA_{d}D", "trustworthiness_k10": float(t), "n_used": int(len(idx))})
    trust_df = pd.DataFrame(trust_rows)
    trust_df.to_csv(outdir / "phasemap7d_trustworthiness.csv", index=False, encoding="utf-8-sig")

    # Neighbor purity: if manifold meaningful, local neighborhoods should enrich phase/operator/condition/layer segment.
    purity_rows = []
    label_cols = [
        "heuristic_operator", "operator_cluster", "phase", "condition", "layer"
    ]
    if "layer" in work.columns:
        work["layer_band"] = pd.cut(
            work["layer"],
            bins=[-np.inf, 6, 19, 22, 26, np.inf],
            labels=["L0_6", "L7_19", "L20_22", "L23_26", "L27_plus"],
        )
        label_cols.append("layer_band")

    for label in label_cols:
        if label not in work.columns:
            continue
        for k in KNN_KS:
            p_raw, n_raw = neighbor_purity(work, X, work[label].to_numpy(), k)
            p_2d, n_2d = neighbor_purity(work, Z_full[:, :2], work[label].to_numpy(), k)
            p_3d, n_3d = neighbor_purity(work, Z_full[:, :3], work[label].to_numpy(), k)
            # Baseline: class prior purity = sum p_c^2.
            vc = pd.Series(work[label]).dropna().value_counts(normalize=True)
            baseline = float(np.sum(vc.values ** 2)) if len(vc) else np.nan
            purity_rows.append({
                "label": label,
                "k": k,
                "baseline_prior_purity": baseline,
                "purity_raw10D": p_raw,
                "purity_pca2D": p_2d,
                "purity_pca3D": p_3d,
                "lift_raw10D": p_raw - baseline if pd.notna(p_raw) else np.nan,
                "lift_pca2D": p_2d - baseline if pd.notna(p_2d) else np.nan,
                "lift_pca3D": p_3d - baseline if pd.notna(p_3d) else np.nan,
                "n_used": n_raw,
            })
    purity_df = pd.DataFrame(purity_rows)
    purity_df.to_csv(outdir / "phasemap7d_neighbor_purity.csv", index=False, encoding="utf-8-sig")

    # Layer trajectory in operator manifold: centroid by layer and finite differences.
    layer_df = None
    if "layer" in work.columns:
        layer_cent = centroid_table(emb, [f"op_m{i}" for i in range(1, 6)], ["layer"], min_n=1)
        layer_cent = layer_cent.sort_values("layer")
        coords = layer_cent[["op_m1", "op_m2", "op_m3", "op_m4", "op_m5"]].to_numpy(float)
        step = np.full(len(layer_cent), np.nan)
        accel = np.full(len(layer_cent), np.nan)
        if len(coords) >= 2:
            step[1:] = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        if len(coords) >= 3:
            accel[2:] = np.linalg.norm(coords[2:] - 2*coords[1:-1] + coords[:-2], axis=1)
        layer_cent["centroid_step_norm"] = step
        layer_cent["centroid_accel_norm"] = accel
        layer_cent.to_csv(outdir / "phasemap7d_layer_trajectory.csv", index=False, encoding="utf-8-sig")
        layer_df = layer_cent

    # Centroids by important groups.
    if "heuristic_operator" in emb.columns:
        centroid_table(emb, [f"op_m{i}" for i in range(1, 6)], ["heuristic_operator"], min_n=5).to_csv(
            outdir / "phasemap7d_centroids_by_operator.csv", index=False, encoding="utf-8-sig"
        )
    if "operator_cluster" in emb.columns:
        centroid_table(emb, [f"op_m{i}" for i in range(1, 6)], ["operator_cluster"], min_n=5).to_csv(
            outdir / "phasemap7d_centroids_by_operator_cluster.csv", index=False, encoding="utf-8-sig"
        )
    group_pc = [c for c in ["phase", "condition"] if c in emb.columns]
    if group_pc:
        centroid_table(emb, [f"op_m{i}" for i in range(1, 6)], group_pc, min_n=5).to_csv(
            outdir / "phasemap7d_centroids_by_phase_condition.csv", index=False, encoding="utf-8-sig"
        )
    if "layer" in emb.columns:
        centroid_table(emb, [f"op_m{i}" for i in range(1, 6)], ["layer"], min_n=1).to_csv(
            outdir / "phasemap7d_centroids_by_layer.csv", index=False, encoding="utf-8-sig"
        )

    # Correlation with layer and state variables.
    corr_rows = []
    check_cols = ["layer"] + [c for c in STATE_COLS if c in work.columns]
    for pc_i in range(1, 6):
        pc = Z_full[:, pc_i-1]
        for c in check_cols:
            corr_rows.append({
                "op_manifold_coord": f"op_m{pc_i}",
                "variable": c,
                "corr": safe_corr(pc, work[c]),
            })
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(outdir / "phasemap7d_manifold_correlation_audit.csv", index=False, encoding="utf-8-sig")

    # Main interpretation readout.
    dims_80 = int(np.searchsorted(cum, 0.80) + 1)
    dims_90 = int(np.searchsorted(cum, 0.90) + 1)
    dims_95 = int(np.searchsorted(cum, 0.95) + 1)
    dims_99 = int(np.searchsorted(cum, 0.99) + 1)

    def rec_at(d):
        return float(rec_df.loc[rec_df["n_dim"] == d, "reconstruction_r2"].iloc[0])

    # Heuristic verdict.
    manifold_pass = bool((dims_90 <= 3) or (pr <= 3.5 and rec_at(3) >= 0.85))
    strong_pass = bool((dims_90 <= 2 and rec_at(2) >= 0.90) or (dims_95 <= 3 and rec_at(3) >= 0.95))

    summary = {
        "experiment": "PhaseMap-7D Operator Manifold Audit",
        "input": str(Path(INPUT_CSV).resolve()),
        "outdir": OUTPUT_DIR,
        "n_rows_total": int(len(df)),
        "n_rows_used": int(len(work)),
        "operator_cols": OP_COLS,
        "main_readout": {
            "pc1_var": float(var[0]),
            "pc2_var": float(var[1]),
            "pc3_var": float(var[2]),
            "pc1_pc2_var": float(cum[1]),
            "pc1_pc2_pc3_var": float(cum[2]),
            "dims_for_80pct": dims_80,
            "dims_for_90pct": dims_90,
            "dims_for_95pct": dims_95,
            "dims_for_99pct": dims_99,
            "pca_participation_ratio": pr,
            "two_nn_intrinsic_dim": d_two,
            "lb_mle_k10_intrinsic_dim": float(id_df[(id_df.method=="Levina_Bickel_MLE") & (id_df.k==10)]["intrinsic_dim"].iloc[0]),
            "reconstruction_r2_1d": rec_at(1),
            "reconstruction_r2_2d": rec_at(2),
            "reconstruction_r2_3d": rec_at(3),
            "trustworthiness_pca2d_k10": float(trust_df[trust_df.embedding=="PCA_2D"]["trustworthiness_k10"].iloc[0]),
            "trustworthiness_pca3d_k10": float(trust_df[trust_df.embedding=="PCA_3D"]["trustworthiness_k10"].iloc[0]),
            "manifold_pass": manifold_pass,
            "manifold_strong_pass": strong_pass,
        },
        "interpretation_rules": {
            "low_dimensionality": "dims_for_90pct <= 3 or participation_ratio <= 3.5 supports low-dimensional operator manifold.",
            "local_geometry": "PCA2D/PCA3D trustworthiness near 1 means low-dimensional embedding preserves local neighborhoods.",
            "semantic_organization": "neighbor purity lift for phase/operator/condition means local manifold neighborhoods align with meaningful dynamics labels.",
            "layer_clock_warning": "high correlation of manifold coordinates with layer would indicate layer-clock contamination rather than pure operator geometry.",
            "caveat": "This audits geometry of op_pc coordinates already extracted from transitions; it does not prove op_pc is pre-transition observable. That was the 7C caveat.",
        },
        "outputs": {
            "pca_spectrum": str(outdir / "phasemap7d_pca_spectrum.csv"),
            "reconstruction_summary": str(outdir / "phasemap7d_reconstruction_summary.csv"),
            "intrinsic_dimension": str(outdir / "phasemap7d_intrinsic_dimension.csv"),
            "trustworthiness": str(outdir / "phasemap7d_trustworthiness.csv"),
            "neighbor_purity": str(outdir / "phasemap7d_neighbor_purity.csv"),
            "embedding_2d": str(outdir / "phasemap7d_embedding_2d.csv"),
            "layer_trajectory": str(outdir / "phasemap7d_layer_trajectory.csv"),
            "manifold_correlation_audit": str(outdir / "phasemap7d_manifold_correlation_audit.csv"),
        },
    }

    with open(outdir / "phasemap7d_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== PhaseMap-7D Operator Manifold Audit ===")
    print(json.dumps(summary["main_readout"], ensure_ascii=False, indent=2))
    print(f"\nSaved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
