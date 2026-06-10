# -*- coding: utf-8 -*-
"""
OA-5C: Neighborhood Continuity & Curvature Audit
================================================

Purpose
-------
OA-5A showed shallow TopK neighborhoods contain task-structure signals.
OA-5Bv2 rejected the over-strong "low-dimensional linear tangent plane" version.

OA-5C now tests a weaker and more geometric hypothesis:

    TopK(H) ≈ local neighborhood N_epsilon(x)

not necessarily a low-dimensional linear tangent plane.

Core questions
--------------
1. Continuity:
   Do small semantic/surface perturbations produce smaller TopK neighborhood changes
   than task/entity changes?

2. Curvature / nonlinearity:
   Is the local neighborhood better described as a curved neighborhood than as a
   flat local plane?
   Operationally:
      - low-dimensional PCA plane should not fully explain the neighborhood;
      - geodesic-like graph distances and Euclidean distances should diverge;
      - local distance preservation should be better within same task than across tasks.

Inputs
------
Reads OA-5A outputs from fixed path:

    C:\\Users\\ZH\\Desktop\\AGI\\python_script\\tangent_audit_oa5a_outputs\\oa5a_topk_records.csv
    C:\\Users\\ZH\\Desktop\\AGI\\python_script\\tangent_audit_oa5a_outputs\\oa5a_prompt_dataset.csv

Outputs
-------
    oa5c_outputs/oa5c_continuity_summary.csv
    oa5c_outputs/oa5c_curvature_summary.csv
    oa5c_outputs/oa5c_neighborhood_graph_summary.csv
    oa5c_outputs/oa5c_verdict_summary.csv

Expected columns in oa5a_topk_records.csv
------------------------------------------
The script is intentionally robust. It tries to infer columns.

Required semantic columns, likely from OA-5A:
    prompt_id or id
    topic
    task
    paraphrase / surface / variant
    layer
    k
    center_* columns OR embedding/center serialized column
    topk token ids column if available

If center vector columns are unavailable, it will fall back to TopK identity
Jaccard distances only where possible.
"""

import ast
import json
import math
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# PATH CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\ZH\Desktop\AGI\python_script\tangent_audit_oa5a_outputs")
INPUT_DIR = BASE_DIR
OUTPUT_DIR = BASE_DIR / "oa5c_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOPK_FILE = INPUT_DIR / "oa5a_topk_records.csv"
DATASET_FILE = INPUT_DIR / "oa5a_prompt_dataset.csv"

# ============================================================
# CONFIG
# ============================================================

SHALLOW_LAYER_MAX = 6
K_VALUES = None          # None = auto all k values
LAYER_GROUPS = {
    "L0": [0],
    "L0_2": [0, 1, 2],
    "L0_6": list(range(0, 7)),
}

# PCA dimensions for "flatness" comparison
PCA_DIMS = [2, 3, 5, 8, 12]

# kNN graph parameters for graph/geodesic proxy
GRAPH_N_NEIGHBORS = 8

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================
# HELPERS
# ============================================================

def infer_col(df, candidates, required=True):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(f"Cannot infer column from candidates: {candidates}\nAvailable: {list(df.columns)[:80]}")
    return None


def parse_vector_cell(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.asarray(x, dtype=np.float32)
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        if s.startswith("[") or s.startswith("("):
            v = ast.literal_eval(s)
            return np.asarray(v, dtype=np.float32)
        # comma/space separated
        if "," in s:
            return np.asarray([float(t) for t in s.split(",")], dtype=np.float32)
        return np.asarray([float(t) for t in s.split()], dtype=np.float32)
    except Exception:
        return None


def parse_id_set(x):
    if isinstance(x, (list, tuple, set, np.ndarray)):
        return set(map(str, list(x)))
    if pd.isna(x):
        return set()
    s = str(x).strip()
    if not s:
        return set()
    try:
        if s.startswith("[") or s.startswith("(") or s.startswith("{"):
            obj = ast.literal_eval(s)
            return set(map(str, list(obj)))
    except Exception:
        pass
    # common separators
    for sep in ["|", ",", " "]:
        if sep in s:
            return set([t for t in s.split(sep) if t != ""])
    return {s}


def find_center_matrix(df):
    """
    Return (X, center_cols or source_desc).
    Tries:
      1. columns with prefix center_
      2. embedding dims c0,c1...
      3. serialized center column
    """
    # Numeric center_* columns
    center_cols = [c for c in df.columns if c.lower().startswith(("center_", "c_", "ck_", "vec_", "z_"))]
    numeric_center_cols = []
    for c in center_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_center_cols.append(c)

    # Avoid metadata false positives if too few
    if len(numeric_center_cols) >= 8:
        X = df[numeric_center_cols].to_numpy(dtype=np.float32)
        return X, numeric_center_cols, "wide_center_columns"

    # common dim columns: d0, d1...
    dim_cols = []
    for c in df.columns:
        cl = c.lower()
        if (cl.startswith("dim") or cl.startswith("emb") or cl.startswith("x")):
            suffix = ''.join([ch for ch in cl if ch.isdigit()])
            if suffix and pd.api.types.is_numeric_dtype(df[c]):
                dim_cols.append(c)
    if len(dim_cols) >= 8:
        # sort by trailing number
        def key(c):
            digs = ''.join([ch for ch in c if ch.isdigit()])
            return int(digs) if digs else 10**9
        dim_cols = sorted(dim_cols, key=key)
        X = df[dim_cols].to_numpy(dtype=np.float32)
        return X, dim_cols, "dim_columns"

    # serialized vector columns
    candidates = [
        "center", "center_vec", "center_vector", "Ck", "C_k",
        "topk_center", "embedding_center", "vec", "vector"
    ]
    for col in candidates:
        if col in df.columns:
            parsed = [parse_vector_cell(v) for v in df[col].values]
            if all(v is not None for v in parsed):
                lens = [len(v) for v in parsed]
                if len(set(lens)) == 1 and lens[0] >= 8:
                    X = np.stack(parsed).astype(np.float32)
                    return X, [col], "serialized_center"

    return None, None, "none"


def find_topk_col(df):
    candidates = [
        "topk_ids", "topk_tokens", "topk_token_ids", "token_ids",
        "ids", "topk", "topk_set", "S_k", "Sk"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def cosine_dist_matrix(X):
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    Xn = X / norms
    sim = Xn @ Xn.T
    return np.clip(1.0 - sim, 0.0, 2.0)


def euclidean_dist_matrix(X):
    return pairwise_distances(X, metric="euclidean")


def jaccard_distance_sets(sets):
    n = len(sets)
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        si = sets[i]
        for j in range(i + 1, n):
            sj = sets[j]
            union = len(si | sj)
            inter = len(si & sj)
            d = 1.0 if union == 0 else 1.0 - inter / union
            D[i, j] = D[j, i] = d
    return D


def pair_categories(meta):
    """
    Generate pair categories:
      same_prompt: same prompt id across layers? not used mostly.
      same_task: same topic + same task, different surface
      same_topic_diff_task
      diff_topic_same_task
      diff_topic_diff_task
    """
    n = len(meta)
    cats = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            same_topic = meta.iloc[i]["topic"] == meta.iloc[j]["topic"]
            same_task = meta.iloc[i]["task"] == meta.iloc[j]["task"]
            same_pid = meta.iloc[i]["prompt_id"] == meta.iloc[j]["prompt_id"]

            if same_pid:
                cats["same_prompt"].append((i, j))
            elif same_topic and same_task:
                cats["same_task"].append((i, j))
            elif same_topic and not same_task:
                cats["same_topic_diff_task"].append((i, j))
            elif (not same_topic) and same_task:
                cats["diff_topic_same_task"].append((i, j))
            else:
                cats["diff_topic_diff_task"].append((i, j))
    return cats


def mean_for_pairs(D, pairs):
    if not pairs:
        return np.nan
    return float(np.mean([D[i, j] for i, j in pairs]))


def graph_shortest_paths_knn(X, n_neighbors=8):
    """
    Lightweight all-pairs geodesic proxy using kNN graph + Floyd-Warshall.
    n is small in OA-5A, so dense Floyd is OK.
    """
    X = np.asarray(X, dtype=np.float32)
    n = len(X)
    if n <= 2:
        return np.zeros((n, n), dtype=np.float32)

    nn = min(n_neighbors + 1, n)
    nbrs = NearestNeighbors(n_neighbors=nn, metric="euclidean").fit(X)
    dist, ind = nbrs.kneighbors(X)

    G = np.full((n, n), np.inf, dtype=np.float32)
    np.fill_diagonal(G, 0.0)
    for i in range(n):
        for d, j in zip(dist[i], ind[i]):
            if i != j:
                G[i, j] = min(G[i, j], d)
                G[j, i] = min(G[j, i], d)

    # Floyd-Warshall
    for k in range(n):
        G = np.minimum(G, G[:, [k]] + G[[k], :])

    # Handle disconnected, replace inf with max finite * 2
    finite = G[np.isfinite(G)]
    if len(finite) > 0:
        maxf = float(np.max(finite))
        G[~np.isfinite(G)] = maxf * 2.0
    else:
        G[~np.isfinite(G)] = 0.0
    return G


def safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    a = a[mask]
    b = b[mask]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def flatten_upper(D):
    iu = np.triu_indices_from(D, k=1)
    return D[iu]


def pca_reconstruction_error(X, n_components):
    X = np.asarray(X, dtype=np.float32)
    if len(X) <= n_components + 1:
        return np.nan, np.nan
    Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    pca = PCA(n_components=min(n_components, Xs.shape[0]-1, Xs.shape[1]), random_state=RANDOM_SEED)
    Z = pca.fit_transform(Xs)
    Xhat = pca.inverse_transform(Z)
    err = np.mean(np.sum((Xs - Xhat) ** 2, axis=1))
    total = np.mean(np.sum((Xs - Xs.mean(axis=0, keepdims=True)) ** 2, axis=1)) + 1e-12
    rel = err / total
    evr = float(np.sum(pca.explained_variance_ratio_))
    return float(rel), evr


# ============================================================
# LOAD
# ============================================================

def load_data():
    print("=" * 72)
    print("OA-5C Neighborhood Continuity & Curvature Audit")
    print("=" * 72)
    print("TOPK_FILE:", TOPK_FILE)
    print("DATASET_FILE:", DATASET_FILE)

    if not TOPK_FILE.exists():
        raise FileNotFoundError(f"Missing {TOPK_FILE}")
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Missing {DATASET_FILE}")

    topk = pd.read_csv(TOPK_FILE)
    dataset = pd.read_csv(DATASET_FILE)

    prompt_col_topk = infer_col(topk, ["prompt_id", "id", "sample_id", "idx"])
    layer_col = infer_col(topk, ["layer", "layer_id", "layer_index", "l"])
    k_col = infer_col(topk, ["k", "topk_k", "K"])

    topk = topk.rename(columns={
        prompt_col_topk: "prompt_id",
        layer_col: "layer",
        k_col: "k",
    })

    # Prefer metadata already present in topk_records
    if "topic" in topk.columns and "task" in topk.columns:
        merged = topk.copy()
        if "surface" not in merged.columns:
            if "paraphrase_id" in merged.columns:
                merged["surface"] = merged["paraphrase_id"]
            elif "paraphrase" in merged.columns:
                merged["surface"] = merged["paraphrase"]
            else:
                merged["surface"] = ""
    else:
        prompt_col_data = infer_col(dataset, ["prompt_id", "id", "sample_id", "idx"])
        topic_col = infer_col(dataset, ["topic", "entity", "subject"])
        task_col = infer_col(dataset, ["task", "task_type", "query_type"])
        surface_col = infer_col(
            dataset,
            ["surface", "paraphrase", "paraphrase_id", "variant", "template"],
            required=False
        )

        keep_cols = [prompt_col_data, topic_col, task_col]
        if surface_col:
            keep_cols.append(surface_col)

        dataset_small = dataset[keep_cols].copy()
        rename = {
            prompt_col_data: "prompt_id",
            topic_col: "topic",
            task_col: "task",
        }
        if surface_col:
            rename[surface_col] = "surface"

        dataset_small = dataset_small.rename(columns=rename)
        merged = topk.merge(dataset_small, on="prompt_id", how="left")

    if merged["topic"].isna().any() or merged["task"].isna().any():
        print("WARNING: Some rows have missing topic/task.")

    X, center_cols, center_source = find_center_matrix(merged)
    topk_col = find_topk_col(merged)

    print("Rows:", len(merged))
    print("Columns:", list(merged.columns))
    print("Center source:", center_source)
    print("TopK identity column:", topk_col)

    return merged, X, center_cols, center_source, topk_col

# ============================================================
# ANALYSES
# ============================================================

def aggregate_layer_group(df, X, layer_group, k):
    """
    Aggregate rows across selected shallow layers by prompt:
      - center vectors: mean over layers
      - topk sets: union over layers
    """
    sub = df[(df["layer"].isin(layer_group)) & (df["k"] == k)].copy()
    if sub.empty:
        return None, None, None

    sub["_row_index"] = sub.index

    meta_rows = []
    X_rows = []

    if X is not None:
        for pid, g in sub.groupby("prompt_id"):
            inds = g["_row_index"].values
            xv = X[inds].mean(axis=0)
            first = g.iloc[0]
            meta_rows.append({
                "prompt_id": pid,
                "topic": first["topic"],
                "task": first["task"],
                "surface": first["surface"] if "surface" in g.columns else "",
            })
            X_rows.append(xv)
        meta = pd.DataFrame(meta_rows)
        Xagg = np.stack(X_rows).astype(np.float32)
    else:
        meta = sub.drop_duplicates("prompt_id")[["prompt_id", "topic", "task"]].copy()
        if "surface" in sub.columns:
            meta["surface"] = sub.drop_duplicates("prompt_id")["surface"].values
        Xagg = None

    return sub, meta, Xagg


def aggregate_sets(df, layer_group, k, topk_col):
    sub = df[(df["layer"].isin(layer_group)) & (df["k"] == k)].copy()
    if sub.empty or topk_col is None:
        return None, None

    rows = []
    sets = []
    for pid, g in sub.groupby("prompt_id"):
        union = set()
        for v in g[topk_col].values:
            union |= parse_id_set(v)
        first = g.iloc[0]
        rows.append({
            "prompt_id": pid,
            "topic": first["topic"],
            "task": first["task"],
            "surface": first["surface"] if "surface" in g.columns else "",
        })
        sets.append(union)
    return pd.DataFrame(rows), sets


def continuity_analysis(df, X, topk_col):
    rows = []

    k_values = sorted(df["k"].dropna().unique().tolist()) if K_VALUES is None else K_VALUES

    for k in k_values:
        for lg_name, layers in LAYER_GROUPS.items():
            # center-vector continuity
            sub, meta, Xagg = aggregate_layer_group(df, X, layers, k)
            if meta is None or len(meta) < 5:
                continue

            cats = pair_categories(meta)

            if Xagg is not None:
                Dcos = cosine_dist_matrix(Xagg)
                Deuc = euclidean_dist_matrix(StandardScaler().fit_transform(Xagg))

                for metric_name, D in [("cosine_center", Dcos), ("euclidean_center_z", Deuc)]:
                    same_task = mean_for_pairs(D, cats["same_task"])
                    diff_entity_same_task = mean_for_pairs(D, cats["diff_topic_same_task"])
                    same_topic_diff_task = mean_for_pairs(D, cats["same_topic_diff_task"])
                    diff_all = mean_for_pairs(D, cats["diff_topic_diff_task"])

                    rows.append({
                        "k": k,
                        "layer_group": lg_name,
                        "metric": metric_name,
                        "D_same_task": same_task,
                        "D_diff_entity_same_task": diff_entity_same_task,
                        "D_same_topic_diff_task": same_topic_diff_task,
                        "D_diff_topic_diff_task": diff_all,
                        "continuity_order_task_vs_entity": bool(
                            np.isfinite(same_task) and np.isfinite(diff_entity_same_task)
                            and same_task < diff_entity_same_task
                        ),
                        "continuity_order_task_vs_same_topic_diff_task": bool(
                            np.isfinite(same_task) and np.isfinite(same_topic_diff_task)
                            and same_task < same_topic_diff_task
                        ),
                        "task_lift_vs_diff_entity_same_task": float(diff_entity_same_task / (same_task + 1e-12)) if np.isfinite(same_task) else np.nan,
                        "task_lift_vs_same_topic_diff_task": float(same_topic_diff_task / (same_task + 1e-12)) if np.isfinite(same_task) else np.nan,
                        "n_prompts": len(meta),
                    })

            # topk identity continuity
            if topk_col is not None:
                meta_s, sets = aggregate_sets(df, layers, k, topk_col)
                if meta_s is not None and len(meta_s) >= 5:
                    cats_s = pair_categories(meta_s)
                    Dj = jaccard_distance_sets(sets)
                    same_task = mean_for_pairs(Dj, cats_s["same_task"])
                    diff_entity_same_task = mean_for_pairs(Dj, cats_s["diff_topic_same_task"])
                    same_topic_diff_task = mean_for_pairs(Dj, cats_s["same_topic_diff_task"])
                    diff_all = mean_for_pairs(Dj, cats_s["diff_topic_diff_task"])

                    rows.append({
                        "k": k,
                        "layer_group": lg_name,
                        "metric": "jaccard_topk_union",
                        "D_same_task": same_task,
                        "D_diff_entity_same_task": diff_entity_same_task,
                        "D_same_topic_diff_task": same_topic_diff_task,
                        "D_diff_topic_diff_task": diff_all,
                        "continuity_order_task_vs_entity": bool(same_task < diff_entity_same_task),
                        "continuity_order_task_vs_same_topic_diff_task": bool(same_task < same_topic_diff_task),
                        "task_lift_vs_diff_entity_same_task": float(diff_entity_same_task / (same_task + 1e-12)),
                        "task_lift_vs_same_topic_diff_task": float(same_topic_diff_task / (same_task + 1e-12)),
                        "n_prompts": len(meta_s),
                    })

    return pd.DataFrame(rows)


def curvature_analysis(df, X):
    if X is None:
        return pd.DataFrame()

    rows = []
    k_values = sorted(df["k"].dropna().unique().tolist()) if K_VALUES is None else K_VALUES

    for k in k_values:
        for lg_name, layers in LAYER_GROUPS.items():
            sub, meta, Xagg = aggregate_layer_group(df, X, layers, k)
            if meta is None or Xagg is None or len(meta) < 10:
                continue

            Xz = StandardScaler().fit_transform(Xagg)
            De = euclidean_dist_matrix(Xz)
            Dg = graph_shortest_paths_knn(Xz, n_neighbors=GRAPH_N_NEIGHBORS)

            # Graph/geodesic vs Euclidean: if curved/nonlinear, geodesic proxy departs from Euclidean
            eu = flatten_upper(De)
            ge = flatten_upper(Dg)
            ge_eu_corr = safe_corr(eu, ge)
            ratio = ge / (eu + 1e-9)
            finite_ratio = ratio[np.isfinite(ratio)]
            geodesic_stretch_mean = float(np.mean(finite_ratio))
            geodesic_stretch_p90 = float(np.quantile(finite_ratio, 0.90))

            # PCA flatness: if low-dim flat, small dim reconstruction error low and explained variance high
            pca_records = {}
            for dim in PCA_DIMS:
                rel_err, evr = pca_reconstruction_error(Xagg, dim)
                pca_records[f"pca{dim}_rel_recon_error"] = rel_err
                pca_records[f"pca{dim}_evr"] = evr

            # Within each same topic+task group, estimate intrinsic dim 90
            group_dims = []
            group_pc2 = []
            group_pc3 = []
            for (topic, task), gmeta in meta.groupby(["topic", "task"]):
                inds = gmeta.index.values
                if len(inds) < 4:
                    continue
                Xg = Xz[inds]
                ncomp = min(len(inds)-1, Xg.shape[1])
                if ncomp < 2:
                    continue
                pca = PCA(n_components=ncomp, random_state=RANDOM_SEED).fit(Xg)
                csum = np.cumsum(pca.explained_variance_ratio_)
                dim90 = int(np.searchsorted(csum, 0.90) + 1)
                group_dims.append(dim90)
                group_pc2.append(float(csum[min(1, len(csum)-1)]))
                group_pc3.append(float(csum[min(2, len(csum)-1)]))

            rows.append({
                "k": k,
                "layer_group": lg_name,
                "n_prompts": len(meta),
                "geodesic_euclidean_corr": ge_eu_corr,
                "geodesic_stretch_mean": geodesic_stretch_mean,
                "geodesic_stretch_p90": geodesic_stretch_p90,
                "mean_group_effective_dim90": float(np.mean(group_dims)) if group_dims else np.nan,
                "mean_group_pc1pc2": float(np.mean(group_pc2)) if group_pc2 else np.nan,
                "mean_group_pc1pc2pc3": float(np.mean(group_pc3)) if group_pc3 else np.nan,
                **pca_records,
            })

    return pd.DataFrame(rows)


def graph_neighborhood_analysis(df, X):
    if X is None:
        return pd.DataFrame()

    rows = []
    k_values = sorted(df["k"].dropna().unique().tolist()) if K_VALUES is None else K_VALUES

    for k in k_values:
        for lg_name, layers in LAYER_GROUPS.items():
            sub, meta, Xagg = aggregate_layer_group(df, X, layers, k)
            if meta is None or Xagg is None or len(meta) < 10:
                continue

            Xz = StandardScaler().fit_transform(Xagg)

            n_neighbors = min(GRAPH_N_NEIGHBORS + 1, len(Xz))
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(Xz)
            dist, ind = nbrs.kneighbors(Xz)

            same_task_edges = 0
            same_topic_edges = 0
            total_edges = 0
            same_task_random_expected = 0
            same_topic_random_expected = 0

            tasks = meta["task"].astype(str).values
            topics = meta["topic"].astype(str).values

            for i in range(len(meta)):
                neigh = [j for j in ind[i] if j != i]
                for j in neigh:
                    total_edges += 1
                    if tasks[i] == tasks[j]:
                        same_task_edges += 1
                    if topics[i] == topics[j]:
                        same_topic_edges += 1

                same_task_random_expected += np.mean(tasks != tasks[i]) * 0  # placeholder

            # random baselines by label frequencies
            task_counts = meta["task"].value_counts(normalize=True)
            topic_counts = meta["topic"].value_counts(normalize=True)
            rand_same_task = float(np.sum(task_counts.values ** 2))
            rand_same_topic = float(np.sum(topic_counts.values ** 2))

            same_task_rate = same_task_edges / max(total_edges, 1)
            same_topic_rate = same_topic_edges / max(total_edges, 1)

            rows.append({
                "k": k,
                "layer_group": lg_name,
                "n_prompts": len(meta),
                "n_edges": total_edges,
                "knn_same_task_rate": same_task_rate,
                "knn_same_task_random": rand_same_task,
                "knn_same_task_lift": same_task_rate / (rand_same_task + 1e-12),
                "knn_same_topic_rate": same_topic_rate,
                "knn_same_topic_random": rand_same_topic,
                "knn_same_topic_lift": same_topic_rate / (rand_same_topic + 1e-12),
            })

    return pd.DataFrame(rows)


def verdict(continuity_df, curvature_df, graph_df):
    rows = []

    if not continuity_df.empty:
        for (k, lg), g in continuity_df.groupby(["k", "layer_group"]):
            center_g = g[g["metric"].isin(["cosine_center", "euclidean_center_z"])]
            jac_g = g[g["metric"] == "jaccard_topk_union"]

            center_task_order_rate = float(center_g["continuity_order_task_vs_entity"].mean()) if len(center_g) else np.nan
            center_same_topic_sep_rate = float(center_g["continuity_order_task_vs_same_topic_diff_task"].mean()) if len(center_g) else np.nan
            center_lift = float(center_g["task_lift_vs_diff_entity_same_task"].mean()) if len(center_g) else np.nan

            jac_task_order = float(jac_g["continuity_order_task_vs_entity"].mean()) if len(jac_g) else np.nan
            jac_lift = float(jac_g["task_lift_vs_diff_entity_same_task"].mean()) if len(jac_g) else np.nan

            csub = curvature_df[(curvature_df["k"] == k) & (curvature_df["layer_group"] == lg)] if not curvature_df.empty else pd.DataFrame()
            gsub = graph_df[(graph_df["k"] == k) & (graph_df["layer_group"] == lg)] if not graph_df.empty else pd.DataFrame()

            dim90 = float(csub["mean_group_effective_dim90"].iloc[0]) if len(csub) and "mean_group_effective_dim90" in csub else np.nan
            pc3 = float(csub["mean_group_pc1pc2pc3"].iloc[0]) if len(csub) and "mean_group_pc1pc2pc3" in csub else np.nan
            pca3_err = float(csub["pca3_rel_recon_error"].iloc[0]) if len(csub) and "pca3_rel_recon_error" in csub else np.nan
            stretch = float(csub["geodesic_stretch_mean"].iloc[0]) if len(csub) and "geodesic_stretch_mean" in csub else np.nan
            knn_task_lift = float(gsub["knn_same_task_lift"].iloc[0]) if len(gsub) and "knn_same_task_lift" in gsub else np.nan

            # Interpret:
            # continuity strong if own task is closer with lift > 1.03.
            # nonlinear/high-dim if dim90 > 3 or pca3 does not explain >90%.
            continuity_pass = (
                (np.isfinite(center_lift) and center_lift > 1.03 and center_task_order_rate >= 0.5)
                or (np.isfinite(jac_lift) and jac_lift > 1.03)
            )
            highdim_or_curved = (
                (np.isfinite(dim90) and dim90 > 3.0)
                or (np.isfinite(pc3) and pc3 < 0.90)
                or (np.isfinite(pca3_err) and pca3_err > 0.10)
            )
            graph_local_task = np.isfinite(knn_task_lift) and knn_task_lift > 1.10

            if continuity_pass and highdim_or_curved and graph_local_task:
                v = "PASS-Strong: continuous high-dimensional neighborhood"
            elif continuity_pass and highdim_or_curved:
                v = "PASS-Moderate: continuity + non-lowdim structure"
            elif continuity_pass:
                v = "PASS-Lite: continuity only"
            else:
                v = "FAIL/Mixed"

            rows.append({
                "k": k,
                "layer_group": lg,
                "center_task_order_rate": center_task_order_rate,
                "center_same_topic_separation_rate": center_same_topic_sep_rate,
                "center_lift_diff_entity_same_task_over_same_task": center_lift,
                "jaccard_task_order_rate": jac_task_order,
                "jaccard_lift_diff_entity_same_task_over_same_task": jac_lift,
                "mean_group_effective_dim90": dim90,
                "mean_group_pc1pc2pc3": pc3,
                "pca3_rel_recon_error_global": pca3_err,
                "geodesic_stretch_mean": stretch,
                "knn_same_task_lift": knn_task_lift,
                "continuity_pass": continuity_pass,
                "highdim_or_curved": highdim_or_curved,
                "graph_local_task": graph_local_task,
                "verdict": v,
            })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    df, X, center_cols, center_source, topk_col = load_data()

    # shallow only by default
    df = df[df["layer"] <= SHALLOW_LAYER_MAX].copy()

    print("\nRunning continuity analysis...")
    continuity_df = continuity_analysis(df, X, topk_col)
    continuity_file = OUTPUT_DIR / "oa5c_continuity_summary.csv"
    continuity_df.to_csv(continuity_file, index=False, encoding="utf-8-sig")
    print("Saved:", continuity_file)

    print("\nRunning curvature analysis...")
    curvature_df = curvature_analysis(df, X)
    curvature_file = OUTPUT_DIR / "oa5c_curvature_summary.csv"
    curvature_df.to_csv(curvature_file, index=False, encoding="utf-8-sig")
    print("Saved:", curvature_file)

    print("\nRunning neighborhood graph analysis...")
    graph_df = graph_neighborhood_analysis(df, X)
    graph_file = OUTPUT_DIR / "oa5c_neighborhood_graph_summary.csv"
    graph_df.to_csv(graph_file, index=False, encoding="utf-8-sig")
    print("Saved:", graph_file)

    print("\nBuilding verdict...")
    verdict_df = verdict(continuity_df, curvature_df, graph_df)
    verdict_file = OUTPUT_DIR / "oa5c_verdict_summary.csv"
    verdict_df.to_csv(verdict_file, index=False, encoding="utf-8-sig")
    print("Saved:", verdict_file)

    print("\n=== OA-5C VERDICT PREVIEW ===")
    if len(verdict_df):
        print(verdict_df.to_string(index=False))
    else:
        print("No verdict rows generated. Check input columns.")

    print("\nDone.")


if __name__ == "__main__":
    main()
