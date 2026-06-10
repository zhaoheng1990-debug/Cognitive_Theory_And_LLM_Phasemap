
# -*- coding: utf-8 -*-
"""
PSG-1A Decomposition Audit
Projection-Selection-Aggregation Geometry decomposition experiment.

Purpose
-------
This script audits why the pipeline

    H_l R^T -> TopK -> Aggregate(realization vectors)

can recover hidden-state neighborhood geometry even when R is disrupted or random.

It decomposes:
1. score_source:
   - real_W
   - row_permuted_W
   - gaussian_R
   - orthogonal_rotated_W

2. realization_source:
   - real_W
   - score_R
   - gaussian_Z
   - whitened_W

3. selection:
   - topk
   - bottomk
   - midk
   - randomk

4. aggregation:
   - center
   - normalized_center
   - spread_scalar
   - center_plus_spread
   - rank_weighted_center
   - score_weighted_center
   - random_signed_center

Main output
-----------
C:\\Users\\ZH\\Desktop\\AGI\\outputs\\psg1a_outputs\\psg1a_decomposition_summary.csv
C:\\Users\\ZH\\Desktop\\AGI\\outputs\\psg1a_outputs\\psg1a_verdict.txt

Default model
-------------
D:\\model\\models--Qwen--Qwen2.5-1.5B-Instruct\\main

Edit MODEL_PATH below if needed.

Recommended command
-------------------
python psg1a_decomposition_audit.py
"""

import os
import math
import json
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA

from transformers import AutoModelForCausalLM, AutoTokenizer


# =========================
# 0. Hardcoded configuration
# =========================

EXPERIMENT_ID = "PSG-1A_Decomposition_Audit_v1_1_tokenizer_patch"

MODEL_PATH = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct"
# If your actual path is different, edit the above line.
# Common alternatives:
# MODEL_PATH = r"D:\model\Qwen2.5-1.5B-Instruct"
# MODEL_PATH = r"D:\model\Llama-3.2-1B-Instruct"
# MODEL_PATH = r"D:\model\gemma-2-2b-it"

OUT_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\psg1a_outputs")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SEED = 42

# Keep this moderate for first run.
MAX_PROMPTS = 96
MAX_NEW_TOKENS = 1

# Qwen 1.5B usually has 28 layers. The script auto clips invalid layers.
LAYER_IDS = [3, 6, 9, 12, 15, 18, 20, 22, 24, 26]

# The PSG phenomenon was strong at large k. 5000 is the key setting.
K_LIST = [500, 1000, 3000, 5000]

# Number of random repeats for randomk and random_signed_center.
N_RANDOM_REPEATS = 3

# To keep runtime bounded, use a focused factorial first.
SCORE_SOURCES = [
    "real_W",
    "row_permuted_W",
    "gaussian_R",
    "orthogonal_rotated_W",
]

REALIZATION_SOURCES = [
    "real_W",
    "score_R",
    "gaussian_Z",
    "whitened_W",
]

SELECTION_MODES = [
    "topk",
    "bottomk",
    "midk",
    "randomk",
]

AGGREGATIONS = [
    "center",
    "normalized_center",
    "spread_scalar",
    "center_plus_spread",
    "rank_weighted_center",
    "score_weighted_center",
    "random_signed_center",
]

# If GPU memory is tight, set this True. It keeps W on CPU and moves only needed tensors.
CPU_W_STORAGE = True


# =========================
# 1. Prompt set
# =========================

def build_prompts():
    """
    Built-in prompt set for first-pass PSG audit.
    The goal is not answer evaluation, but hidden-neighborhood geometry across diverse prompts.
    """
    concepts = [
        "Paris", "Newton", "Einstein", "photosynthesis", "democracy", "inflation",
        "semiconductor", "gradient descent", "black hole", "evolution", "Shakespeare",
        "supply chain", "quantum mechanics", "climate change", "neural network", "volcano",
        "antibiotics", "electric field", "Renaissance", "machine translation", "compiler",
        "Bayesian inference", "game theory", "protein folding"
    ]
    operators = [
        "define the core idea of",
        "explain the causal mechanism behind",
        "compare two competing interpretations of",
        "give a simple example of",
    ]
    prompts = []
    for c in concepts:
        for op in operators:
            prompts.append(f"{op} {c}. Answer in one concise paragraph.")
    return prompts[:MAX_PROMPTS]


# =========================
# 2. Utility functions
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4:
        return np.nan, np.nan
    if np.std(x[mask]) < 1e-12 or np.std(y[mask]) < 1e-12:
        return np.nan, np.nan
    pr = pearsonr(x[mask], y[mask]).statistic
    sr = spearmanr(x[mask], y[mask]).statistic
    return float(pr), float(sr)


def upper_tri_values(mat):
    mat = np.asarray(mat)
    idx = np.triu_indices(mat.shape[0], k=1)
    return mat[idx]


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


def l2_normalize_np(x, axis=-1, eps=1e-9):
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(denom, eps)


def get_output_embedding_weight(model):
    emb = model.get_output_embeddings()
    if emb is None:
        raise RuntimeError("Model has no output embedding / lm_head.")
    W = emb.weight.detach()
    return W


def whiten_rows_np(W_np, n_components=None, eps=1e-5):
    """
    Row-wise whitening in feature dimension:
    W_centered @ V @ diag(1/sqrt(eig)).
    For d around 1536 this is OK for first-pass.
    """
    W = W_np.astype(np.float32)
    mean = W.mean(axis=0, keepdims=True)
    X = W - mean
    # covariance d x d
    cov = (X.T @ X) / max(1, X.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, eps)
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    Xw = X @ inv_sqrt
    return Xw.astype(np.float32)


def make_orthogonal_matrix(d, seed=42):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d)).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    return Q.astype(np.float32)


def compute_midk_indices(scores, k):
    """
    Select k items around median score.
    scores: torch tensor [n, vocab]
    """
    n, v = scores.shape
    mid = v // 2
    # sort full vocab can be expensive but acceptable for n <= 100.
    idx = torch.argsort(scores, dim=1, descending=True)
    start = max(0, mid - k // 2)
    end = min(v, start + k)
    return idx[:, start:end]


def select_indices(scores, k, mode, rng):
    """
    scores: torch [n, vocab] float32/float16
    returns indices [n, k]
    """
    n, v = scores.shape
    k = min(k, v)

    if mode == "topk":
        return torch.topk(scores, k=k, dim=1, largest=True).indices
    if mode == "bottomk":
        return torch.topk(scores, k=k, dim=1, largest=False).indices
    if mode == "midk":
        return compute_midk_indices(scores, k)
    if mode == "randomk":
        arr = []
        for _ in range(n):
            arr.append(rng.choice(v, size=k, replace=False))
        return torch.tensor(np.stack(arr), dtype=torch.long, device=scores.device)

    raise ValueError(f"Unknown selection mode: {mode}")


def gather_rows(matrix_np, indices_np):
    """
    matrix_np: [vocab, d]
    indices_np: [n, k]
    returns [n, k, d]
    """
    return matrix_np[indices_np]


def aggregate_selected(selected_vecs, selected_scores_np, agg_mode, rng):
    """
    selected_vecs: [n, k, d] np.float32
    selected_scores_np: [n, k] or None
    output features [n, p]
    """
    X = selected_vecs.astype(np.float32)
    n, k, d = X.shape

    if agg_mode == "center":
        return X.mean(axis=1)

    if agg_mode == "normalized_center":
        c = X.mean(axis=1)
        return l2_normalize_np(c)

    if agg_mode == "spread_scalar":
        c = X.mean(axis=1, keepdims=True)
        spread = np.linalg.norm(X - c, axis=2).mean(axis=1, keepdims=True)
        return spread.astype(np.float32)

    if agg_mode == "center_plus_spread":
        c = X.mean(axis=1)
        spread = np.linalg.norm(X - c[:, None, :], axis=2).mean(axis=1, keepdims=True)
        cn = np.linalg.norm(c, axis=1, keepdims=True)
        return np.concatenate([c, spread, cn], axis=1).astype(np.float32)

    if agg_mode == "rank_weighted_center":
        # rank weights: highest selected rank receives largest weight
        weights = np.linspace(1.0, 0.1, k, dtype=np.float32)
        weights = weights / weights.sum()
        return (X * weights[None, :, None]).sum(axis=1)

    if agg_mode == "score_weighted_center":
        if selected_scores_np is None:
            return X.mean(axis=1)
        S = selected_scores_np.astype(np.float32)
        S = S - S.max(axis=1, keepdims=True)
        # stable softmax; temperature avoids domination
        temp = max(1e-6, float(np.std(S)) + 1e-6)
        W = np.exp(S / temp)
        W = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-9)
        return (X * W[:, :, None]).sum(axis=1)

    if agg_mode == "random_signed_center":
        signs = rng.choice([-1.0, 1.0], size=(n, k)).astype(np.float32)
        return (X * signs[:, :, None]).mean(axis=1)

    raise ValueError(f"Unknown aggregation mode: {agg_mode}")


def make_distance_for_features(features, agg_mode):
    if agg_mode in ["spread_scalar"]:
        return euclidean_distance_matrix(features)
    return cosine_distance_matrix(features)


# =========================
# 3. Hidden state extraction
# =========================

def load_model_and_tokenizer():
    print(f"[INFO] Loading model from: {MODEL_PATH}")

    # v1.1 patch:
    # Some local Qwen/Gemma/Llama tokenizer caches fail when transformers tries to
    # instantiate the fast tokenizer backend without sentencepiece/tiktoken.
    # We therefore prefer the slow tokenizer first, then fall back to fast only if needed.
    tokenizer = None
    tokenizer_errors = []

    for use_fast in [False, True]:
        try:
            print(f"[INFO] Loading tokenizer with use_fast={use_fast}")
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATH,
                local_files_only=True,
                trust_remote_code=True,
                use_fast=use_fast
            )
            break
        except Exception as e:
            tokenizer_errors.append(f"use_fast={use_fast}: {repr(e)}")
            tokenizer = None

    if tokenizer is None:
        msg = "\n".join(tokenizer_errors)
        raise RuntimeError(
            "Failed to load tokenizer with both slow and fast modes.\n"
            "If this is Qwen, try: pip install tiktoken sentencepiece protobuf\n"
            "Tokenizer errors:\n" + msg
        )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        device_map=None
    ).to(DEVICE)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


@torch.no_grad()
def extract_hidden_states(model, tokenizer, prompts):
    """
    Extract final-token hidden state for every layer.
    Returns:
      hidden_by_layer: dict layer_id -> np [n, d]
    """
    print(f"[INFO] Extracting hidden states for {len(prompts)} prompts on {DEVICE}")
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    ).to(DEVICE)

    out = model(
        **encoded,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True
    )

    attention_mask = encoded["attention_mask"]
    last_idx = attention_mask.sum(dim=1) - 1

    hidden_states = out.hidden_states
    n_layers_total = len(hidden_states) - 1  # excluding embeddings
    valid_layers = [l for l in LAYER_IDS if 0 <= l <= n_layers_total]
    print(f"[INFO] Model hidden layers: {n_layers_total}; using layers: {valid_layers}")

    hidden_by_layer = {}
    batch_idx = torch.arange(len(prompts), device=DEVICE)
    for l in valid_layers:
        H = hidden_states[l][batch_idx, last_idx, :].detach().float().cpu().numpy()
        hidden_by_layer[l] = H.astype(np.float32)

    return hidden_by_layer, valid_layers


# =========================
# 4. Score and realization matrices
# =========================

def prepare_matrices(model):
    """
    Creates score readouts and realization matrices.

    Returns:
      matrices: dict name -> np [vocab, d]
    """
    W_t = get_output_embedding_weight(model).detach().float().cpu()
    W_np = W_t.numpy().astype(np.float32)
    vocab, d = W_np.shape
    print(f"[INFO] W shape: vocab={vocab}, d={d}")

    rng = np.random.default_rng(SEED)

    matrices = {}
    matrices["real_W"] = W_np

    perm = rng.permutation(vocab)
    matrices["row_permuted_W"] = W_np[perm].copy()

    # Gaussian matched to W row norm scale.
    row_norm_mean = np.linalg.norm(W_np, axis=1).mean()
    gaussian = rng.standard_normal((vocab, d)).astype(np.float32)
    gaussian = l2_normalize_np(gaussian) * row_norm_mean
    matrices["gaussian_R"] = gaussian.astype(np.float32)
    matrices["gaussian_Z"] = gaussian.copy()

    # Orthogonal rotation of W in feature space.
    print("[INFO] Building orthogonal rotation matrix...")
    Q = make_orthogonal_matrix(d, SEED + 17)
    matrices["orthogonal_rotated_W"] = (W_np @ Q).astype(np.float32)

    print("[INFO] Building whitened W realization matrix...")
    try:
        matrices["whitened_W"] = whiten_rows_np(W_np)
    except Exception as e:
        print(f"[WARN] Whitening failed: {e}; fallback to row-normalized centered W.")
        X = W_np - W_np.mean(axis=0, keepdims=True)
        matrices["whitened_W"] = l2_normalize_np(X).astype(np.float32)

    return matrices


# =========================
# 5. Core PSG audit
# =========================

def run_condition_for_layer(
    H_np,
    layer_id,
    score_source,
    realization_source,
    selection_mode,
    aggregation,
    k,
    matrices,
    repeat_id=0
):
    rng = np.random.default_rng(SEED + 1000 * layer_id + 97 * repeat_id + k)

    R_score_np = matrices[score_source]

    if realization_source == "score_R":
        R_realize_np = R_score_np
    else:
        R_realize_np = matrices[realization_source]

    # Hidden geometry.
    D_H = cosine_distance_matrix(H_np)
    y_h = upper_tri_values(D_H)

    # Compute scores.
    H_t = torch.tensor(H_np, dtype=torch.float32, device=DEVICE)
    R_t = torch.tensor(R_score_np, dtype=torch.float32, device=DEVICE)
    scores = H_t @ R_t.T

    # Select indices.
    idx_t = select_indices(scores, k, selection_mode, rng)
    idx_np = idx_t.detach().cpu().numpy()

    # Gather selected scores for weighted aggregation.
    selected_scores = None
    if aggregation == "score_weighted_center":
        selected_scores = torch.gather(scores, 1, idx_t).detach().float().cpu().numpy()

    # Realization vectors and aggregation.
    selected_vecs = gather_rows(R_realize_np, idx_np)
    features = aggregate_selected(selected_vecs, selected_scores, aggregation, rng)

    # Feature geometry.
    D_Z = make_distance_for_features(features, aggregation)
    y_z = upper_tri_values(D_Z)

    pear, spear = safe_corr(y_h, y_z)

    # Additional diagnostics.
    avg_selected_norm = float(np.linalg.norm(selected_vecs, axis=2).mean())
    center_norm = float(np.linalg.norm(features, axis=1).mean()) if features.ndim == 2 else float("nan")

    # TopK overlap diagnostics. Expensive but n is small.
    overlaps = []
    for i in range(idx_np.shape[0]):
        si = set(idx_np[i].tolist())
        for j in range(i + 1, idx_np.shape[0]):
            sj = set(idx_np[j].tolist())
            inter = len(si.intersection(sj))
            union = len(si.union(sj))
            overlaps.append(inter / union if union > 0 else 0.0)
    mean_jaccard = float(np.mean(overlaps)) if overlaps else np.nan

    return {
        "experiment_id": EXPERIMENT_ID,
        "layer": layer_id,
        "k": k,
        "score_source": score_source,
        "realization_source": realization_source,
        "selection": selection_mode,
        "aggregation": aggregation,
        "repeat_id": repeat_id,
        "topo_pearson": pear,
        "topo_spearman": spear,
        "abs_topo_spearman": abs(spear) if np.isfinite(spear) else np.nan,
        "feature_dim": int(features.shape[1]) if features.ndim == 2 else 1,
        "avg_selected_norm": avg_selected_norm,
        "feature_center_norm": center_norm,
        "mean_selected_jaccard": mean_jaccard,
        "n_samples": int(H_np.shape[0]),
    }


def run_audit():
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    prompts = build_prompts()
    pd.DataFrame({"prompt": prompts}).to_csv(OUT_DIR / "psg1a_prompts.csv", index=False, encoding="utf-8-sig")

    model, tokenizer = load_model_and_tokenizer()
    hidden_by_layer, valid_layers = extract_hidden_states(model, tokenizer, prompts)
    matrices = prepare_matrices(model)

    rows = []
    start = time.time()

    total = (
        len(valid_layers)
        * len(K_LIST)
        * len(SCORE_SOURCES)
        * len(REALIZATION_SOURCES)
        * len(SELECTION_MODES)
        * len(AGGREGATIONS)
    )
    done = 0

    for layer_id in valid_layers:
        H_np = hidden_by_layer[layer_id]

        # Save hidden geometry summary.
        D_H = cosine_distance_matrix(H_np)
        pd.DataFrame(D_H).to_csv(OUT_DIR / f"hidden_distance_layer_{layer_id}.csv", index=False)

        for k in K_LIST:
            for score_source in SCORE_SOURCES:
                for realization_source in REALIZATION_SOURCES:
                    for selection_mode in SELECTION_MODES:
                        repeats = N_RANDOM_REPEATS if selection_mode == "randomk" else 1
                        for aggregation in AGGREGATIONS:
                            agg_repeats = N_RANDOM_REPEATS if aggregation == "random_signed_center" else 1
                            combined_repeats = max(repeats, agg_repeats)

                            for rep in range(combined_repeats):
                                try:
                                    row = run_condition_for_layer(
                                        H_np=H_np,
                                        layer_id=layer_id,
                                        score_source=score_source,
                                        realization_source=realization_source,
                                        selection_mode=selection_mode,
                                        aggregation=aggregation,
                                        k=k,
                                        matrices=matrices,
                                        repeat_id=rep,
                                    )
                                    rows.append(row)
                                except RuntimeError as e:
                                    if "out of memory" in str(e).lower():
                                        print("[WARN] CUDA OOM; clearing cache and recording failure.")
                                        torch.cuda.empty_cache()
                                    rows.append({
                                        "experiment_id": EXPERIMENT_ID,
                                        "layer": layer_id,
                                        "k": k,
                                        "score_source": score_source,
                                        "realization_source": realization_source,
                                        "selection": selection_mode,
                                        "aggregation": aggregation,
                                        "repeat_id": rep,
                                        "error": repr(e),
                                    })
                                except Exception as e:
                                    rows.append({
                                        "experiment_id": EXPERIMENT_ID,
                                        "layer": layer_id,
                                        "k": k,
                                        "score_source": score_source,
                                        "realization_source": realization_source,
                                        "selection": selection_mode,
                                        "aggregation": aggregation,
                                        "repeat_id": rep,
                                        "error": repr(e),
                                    })

                        done += 1
                        if done % 50 == 0:
                            elapsed = time.time() - start
                            print(f"[INFO] Progress unit {done}/{total}; rows={len(rows)}; elapsed={elapsed:.1f}s")

    df = pd.DataFrame(rows)
    raw_path = OUT_DIR / "psg1a_decomposition_raw.csv"
    df.to_csv(raw_path, index=False, encoding="utf-8-sig")

    # Aggregate repeated stochastic settings.
    group_cols = [
        "layer", "k", "score_source", "realization_source", "selection", "aggregation"
    ]
    numeric_cols = [
        "topo_pearson", "topo_spearman", "abs_topo_spearman",
        "feature_dim", "avg_selected_norm", "feature_center_norm",
        "mean_selected_jaccard", "n_samples"
    ]
    ok = df[~df.get("topo_spearman", pd.Series(index=df.index)).isna()].copy()
    summary = ok.groupby(group_cols, dropna=False)[numeric_cols].mean().reset_index()
    summary["n_repeats"] = ok.groupby(group_cols, dropna=False).size().values

    summary_path = OUT_DIR / "psg1a_decomposition_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # Pivot summaries for audit.
    best = summary.sort_values("topo_spearman", ascending=False).head(50)
    best.to_csv(OUT_DIR / "psg1a_top50_conditions.csv", index=False, encoding="utf-8-sig")

    # Decomposition views.
    views = {}
    for col in ["score_source", "realization_source", "selection", "aggregation", "k", "layer"]:
        v = summary.groupby(col)["topo_spearman"].agg(["mean", "median", "max", "std", "count"]).reset_index()
        v = v.sort_values("mean", ascending=False)
        views[col] = v
        v.to_csv(OUT_DIR / f"psg1a_factor_view_{col}.csv", index=False, encoding="utf-8-sig")

    verdict = build_verdict(summary, views)
    with open(OUT_DIR / "psg1a_verdict.txt", "w", encoding="utf-8") as f:
        f.write(verdict)

    print("\n" + verdict)
    print(f"\n[OK] Raw: {raw_path}")
    print(f"[OK] Summary: {summary_path}")
    print(f"[OK] Top50: {OUT_DIR / 'psg1a_top50_conditions.csv'}")
    print(f"[OK] Verdict: {OUT_DIR / 'psg1a_verdict.txt'}")


def build_verdict(summary, views):
    lines = []
    lines.append("=" * 80)
    lines.append("PSG-1A Decomposition Audit Verdict")
    lines.append("=" * 80)

    if summary.empty:
        lines.append("NO_VALID_ROWS: no successful condition was produced.")
        return "\n".join(lines)

    top = summary.sort_values("topo_spearman", ascending=False).head(10)
    lines.append("\nTop 10 conditions by topo_spearman:")
    for _, r in top.iterrows():
        lines.append(
            f"layer={int(r['layer'])} k={int(r['k'])} "
            f"score={r['score_source']} realize={r['realization_source']} "
            f"select={r['selection']} agg={r['aggregation']} "
            f"spearman={r['topo_spearman']:.4f}"
        )

    lines.append("\nFactor means:")
    for col, v in views.items():
        lines.append(f"\n[{col}]")
        for _, r in v.head(10).iterrows():
            lines.append(
                f"{r[col]}: mean={r['mean']:.4f}, median={r['median']:.4f}, "
                f"max={r['max']:.4f}, n={int(r['count'])}"
            )

    # Main interpretation heuristics.
    def mean_for(query):
        sub = summary.query(query)
        if sub.empty:
            return np.nan
        return float(sub["topo_spearman"].mean())

    topk_mean = mean_for("selection == 'topk'")
    randomk_mean = mean_for("selection == 'randomk'")
    bottomk_mean = mean_for("selection == 'bottomk'")
    real_score_mean = mean_for("score_source == 'real_W'")
    gauss_score_mean = mean_for("score_source == 'gaussian_R'")
    rowperm_score_mean = mean_for("score_source == 'row_permuted_W'")
    real_realize_mean = mean_for("realization_source == 'real_W'")
    gauss_realize_mean = mean_for("realization_source == 'gaussian_Z'")
    center_mean = mean_for("aggregation == 'center'")
    spread_mean = mean_for("aggregation == 'spread_scalar'")

    lines.append("\nMain contrasts:")
    lines.append(f"topk_mean={topk_mean:.4f}")
    lines.append(f"randomk_mean={randomk_mean:.4f}")
    lines.append(f"bottomk_mean={bottomk_mean:.4f}")
    lines.append(f"real_score_mean={real_score_mean:.4f}")
    lines.append(f"gaussian_score_mean={gauss_score_mean:.4f}")
    lines.append(f"rowperm_score_mean={rowperm_score_mean:.4f}")
    lines.append(f"real_realization_mean={real_realize_mean:.4f}")
    lines.append(f"gaussian_realization_mean={gauss_realize_mean:.4f}")
    lines.append(f"center_mean={center_mean:.4f}")
    lines.append(f"spread_scalar_mean={spread_mean:.4f}")

    lines.append("\nInterpretation rules:")
    if np.isfinite(topk_mean) and np.isfinite(randomk_mean):
        if topk_mean > randomk_mean + 0.10:
            lines.append("PASS: TopK selection is a major contributor over random-K.")
        elif topk_mean > randomk_mean + 0.03:
            lines.append("WEAK_PASS: TopK selection improves over random-K, but margin is modest.")
        else:
            lines.append("FAIL_OR_NULL: TopK does not clearly beat random-K; aggregation/realization may dominate.")

    if np.isfinite(gauss_score_mean) and np.isfinite(real_score_mean):
        if gauss_score_mean >= real_score_mean - 0.03:
            lines.append("SUPPORT: Random/Gaussian score readout can recover comparable geometry.")
        else:
            lines.append("SUPPORT_REAL_W: Real W score source is materially stronger than Gaussian readout.")

    if np.isfinite(real_realize_mean) and np.isfinite(gauss_realize_mean):
        if real_realize_mean > gauss_realize_mean + 0.05:
            lines.append("SUPPORT: Real W as realization space adds recovery power beyond Gaussian realization.")
        elif gauss_realize_mean >= real_realize_mean - 0.03:
            lines.append("SUPPORT: Realization source need not be semantic W; high-dimensional realization geometry suffices.")
        else:
            lines.append("MIXED: Realization source effect is present but small.")

    if np.isfinite(center_mean) and np.isfinite(spread_mean):
        if center_mean > spread_mean + 0.05:
            lines.append("SUPPORT: Centroid/center aggregation is more important than spread-only statistics.")
        else:
            lines.append("MIXED: Spread/covariance-like statistics may carry substantial geometry.")

    lines.append("\nSuggested next step:")
    lines.append("Run PSG-1B Gaussian Null if gaussian_R + gaussian_Z + topk + center remains high.")
    lines.append("Run PSG-1C Whitening/Anisotropy if real_W realization is consistently weaker or stronger than gaussian_Z.")
    lines.append("=" * 80)
    return "\n".join(lines)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    run_audit()
