# ============================================================
# PhaseMap-6B
# TopK Operator Re-Identification
#
# Goal:
#   Re-identify layerwise operators as finite morphism families
#   over TopK neighborhoods.
#
# Operator object:
#   N_k(H_l) -> N_k(H_{l+1})
#
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

SAVE_DIR = Path(r"C:\Windows\System32\phasemap5a_outputs")
OUT_DIR = Path("./phasemap6b_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SAVE_DIR / "phasemap5a_dataset.csv")
topk_df = pd.read_csv(SAVE_DIR / "phasemap5a_topk_features.csv")
r_df = pd.read_csv(SAVE_DIR / "phasemap5a_r_features.csv")

TRACK_LAYERS = list(range(7, 20))
DECISION_LAYERS = list(range(20, 26))
TOPK_LIST = [50, 100, 200, 500, 1000]

# ============================================================
# NOTE:
# 5A topk_features are aggregated over L7-L19.
# For true operator re-identification, we need per-layer features.
#
# If current 5A did not save per-layer TopK centers/spreads,
# this script builds a first-pass operator dataset from available
# trajectory summary features + R trajectory.
#
# Better 6B.full should rerun hidden extraction and save per-layer
# TopK operator transitions.
# ============================================================

full = pd.read_csv(
    SAVE_DIR / "phasemap5a_dataset.csv"
)

# ============================================================
# 1. Construct first-pass operator descriptors
# ============================================================

rows = []

for idx, row in full.iterrows():
    base = {
        "sample_id": idx,
        "graph_id": row["graph_id"],
        "condition": row["condition"],
        "phase": row["phase"],
    }

    # R-space transition descriptors over L20-L25
    R = np.array([row[f"R{l}"] for l in DECISION_LAYERS], dtype=float)
    v = np.diff(R)
    a = np.diff(v)

    base.update({
        "R_mean": float(R.mean()),
        "R_final": float(R[-1]),
        "R_slope": float(np.polyfit(np.arange(len(R)), R, 1)[0]),
        "R_range": float(R.max() - R.min()),
        "R_abs_min": float(np.min(np.abs(R))),
        "R_cross_zero": int(np.any(np.sign(R[:-1]) != np.sign(R[1:]))),
        "R_v_mean": float(v.mean()),
        "R_v_std": float(v.std()),
        "R_v_sign_changes": int(np.sum(np.sign(v[:-1]) != np.sign(v[1:]))),
        "R_a_mean": float(a.mean()) if len(a) else 0.0,
        "R_a_std": float(a.std()) if len(a) else 0.0,
    })

    # TopK trajectory summary descriptors from 5A
    for k in TOPK_LIST:
        for name in [
            f"k{k}_spread_mean",
            f"k{k}_spread_std",
            f"k{k}_spread_slope",
            f"k{k}_jacc_mean",
            f"k{k}_jacc_min",
            f"k{k}_center_step_mean",
            f"k{k}_center_step_max",
            f"k{k}_rank_gap_mean",
            f"k{k}_rank_gap_slope",
            f"k{k}_traj_speed_mean",
            f"k{k}_traj_speed_std",
            f"k{k}_traj_speed_max",
            f"k{k}_traj_accel_mean",
            f"k{k}_traj_accel_max",
        ]:
            if name in row:
                base[name] = row[name]

    rows.append(base)

op_df = pd.DataFrame(rows)

meta_cols = ["sample_id", "graph_id", "condition", "phase"]

# Only use numeric operator descriptors actually constructed in op_df
feature_cols = [
    c for c in op_df.columns
    if c not in meta_cols
    and pd.api.types.is_numeric_dtype(op_df[c])
]
X = op_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
Xs = StandardScaler().fit_transform(X)

# ============================================================
# 2. PCA for operator geometry
# ============================================================

pca = PCA(n_components=min(10, Xs.shape[1]))
Xp = pca.fit_transform(Xs)

for i in range(Xp.shape[1]):
    op_df[f"op_pc{i+1}"] = Xp[:, i]

pca_info = [
    {
        "pc": i + 1,
        "explained_variance": float(v),
    }
    for i, v in enumerate(pca.explained_variance_ratio_)
]

# ============================================================
# 3. Cluster operator families
# ============================================================

cluster_results = []

for K in range(2, 11):
    km = KMeans(n_clusters=K, random_state=42, n_init=30)
    labels = km.fit_predict(Xs)

    sil = silhouette_score(Xs, labels)

    cluster_results.append({
        "K": K,
        "silhouette": float(sil),
        "cluster_sizes": {
            str(i): int(np.sum(labels == i))
            for i in range(K)
        },
    })

best = max(cluster_results, key=lambda x: x["silhouette"])
BEST_K = best["K"]

km = KMeans(n_clusters=BEST_K, random_state=42, n_init=50)
op_df["operator_cluster"] = km.fit_predict(Xs)

# ============================================================
# 4. Cluster interpretation
# ============================================================

cluster_summary = []

for c in sorted(op_df["operator_cluster"].unique()):
    sub = op_df[op_df["operator_cluster"] == c]

    item = {
        "operator_cluster": int(c),
        "n": int(len(sub)),
        "condition_counts": sub["condition"].value_counts().to_dict(),
        "phase_counts": sub["phase"].value_counts().to_dict(),
    }

    for feat in [
        "R_slope",
        "R_range",
        "R_abs_min",
        "R_cross_zero",
        "R_v_std",
        "R_v_sign_changes",
        "k500_spread_slope",
        "k500_jacc_mean",
        "k500_jacc_min",
        "k500_center_step_mean",
        "k500_center_step_max",
        "k500_rank_gap_slope",
        "k500_traj_speed_mean",
        "k500_traj_accel_mean",
    ]:
        if feat in sub.columns:
            item[f"{feat}_mean"] = float(sub[feat].mean())
            item[f"{feat}_std"] = float(sub[feat].std())

    cluster_summary.append(item)

cluster_summary_df = pd.DataFrame(cluster_summary)

# ============================================================
# 5. Heuristic operator labels
# ============================================================

def label_operator(row):
    """
    First-pass heuristic labels.
    These are not final; they help interpret clusters.
    """

    jacc = row.get("k500_jacc_mean", 0)
    spread_slope = row.get("k500_spread_slope", 0)
    speed = row.get("k500_traj_speed_mean", 0)
    accel = row.get("k500_traj_accel_mean", 0)
    r_cross = row.get("R_cross_zero", 0)
    r_v_sign = row.get("R_v_sign_changes", 0)
    r_abs_min = row.get("R_abs_min", 999)
    r_slope = row.get("R_slope", 0)

    if r_abs_min < 0.2 or r_cross:
        return "Boundary_Crossing"

    if r_v_sign >= 2:
        return "Oscillatory_Rotation"

    if spread_slope < -0.01:
        return "Contraction"

    if spread_slope > 0.01:
        return "Expansion"

    if jacc < 0.25 and speed > 0.1:
        return "Transport_Jump"

    if accel > 0.1:
        return "Branching_or_Reorientation"

    if abs(r_slope) > 0.5:
        return "Commitment_Drift"

    return "Smooth_Transport"

op_df["heuristic_operator"] = op_df.apply(label_operator, axis=1)

heuristic_summary = op_df.groupby(
    ["operator_cluster", "heuristic_operator"]
).size().reset_index(name="count")

# ============================================================
# 6. Layer-regime proxy profile
# ============================================================
# Since 5A features are aggregated over L7-L19, this is a sample-level
# operator-family profile, not true layerwise transition profile.
# Full 6B should save per-transition rows:
#   sample_id, l, features(l->l+1), cluster.
# ============================================================

phase_profile = op_df.groupby(
    ["phase", "operator_cluster"]
).size().reset_index(name="count")

condition_profile = op_df.groupby(
    ["condition", "operator_cluster"]
).size().reset_index(name="count")

# ============================================================
# SAVE
# ============================================================

op_df.to_csv(OUT_DIR / "phasemap6b_operator_dataset.csv", index=False)
cluster_summary_df.to_csv(OUT_DIR / "phasemap6b_operator_cluster_summary.csv", index=False)
pd.DataFrame(cluster_results).to_csv(OUT_DIR / "phasemap6b_k_selection.csv", index=False)
heuristic_summary.to_csv(OUT_DIR / "phasemap6b_heuristic_operator_summary.csv", index=False)
phase_profile.to_csv(OUT_DIR / "phasemap6b_phase_operator_profile.csv", index=False)
condition_profile.to_csv(OUT_DIR / "phasemap6b_condition_operator_profile.csv", index=False)

summary = {
    "n_samples": int(len(op_df)),
    "best_K": int(BEST_K),
    "best_silhouette": float(best["silhouette"]),
    "pca": pca_info,
    "cluster_results": cluster_results,
    "cluster_summary": cluster_summary,
}

with open(OUT_DIR / "phasemap6b_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("Done.")
print("Saved to:", OUT_DIR)
print("Best K:", BEST_K)
print("Best silhouette:", best["silhouette"])
print(cluster_summary_df)
print("\nHeuristic summary:")
print(heuristic_summary)