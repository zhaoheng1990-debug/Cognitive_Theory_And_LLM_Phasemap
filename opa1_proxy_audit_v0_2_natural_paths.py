# -*- coding: utf-8 -*-
"""
OPA-1 Ontology Proxy Audit v0.2

Purpose:
  Run OPA-1A Proxy Rank Audit + OPA-1B PSG-Residual Audit.

Inputs:
  TXT natural prompt files, or CSV files.
  TXT: one prompt per non-empty line.
  CSV: at least prompt_id,prompt; recommended clean_answer,conflict_answer,phase_label,split_group.

Outputs:
  output_dir/<model_key>/
    hidden_last_token_layers.npy
    opa1_layer_readback_geometry.csv
    opa1_proxy_rank_deltaU.csv
    opa1_proxy_rank_phase.csv
    opa1_features_*.csv / npy
    opa1_summary.json

Notes:
  - TopK/VIM features are treated as PSG-corrected readback observables.
  - The script does not claim TopK/VIM proves learned-W semantic causality.
  - MemoryUnit / ASA control tables are reserved as second-pass plug-ins.
"""

import os
import json
import math
import random
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder


# =========================
# 0. USER CONFIG
# =========================

@dataclass
class ModelCfg:
    key: str
    path: str
    pre_layers: List[int]
    precursor_layers: List[int]
    decision_layers: List[int]
    obs_layers: List[int]
    use_chat_template: bool = False


# Paths aligned to the uploaded CM-5C feature export script.
# Model paths are copied from GPT_170_CM_5C_feature_export_v0_4.py.
# Output path uses the same outputs root, but writes to opa1_outputs to avoid overwriting CM-5C.
PROMPT_INPUT_PATHS = [
    r"C:\Users\ZH\Desktop\AGI\data\natural_wiki_texts_220.txt",
]
OUTPUT_DIR = r"C:\Users\ZH\Desktop\AGI\outputs\opa1_outputs"

# For natural wiki text, do not wrap the text with chat template by default.
USE_CHAT_TEMPLATE_FOR_NATURAL_TEXT = False

MODEL_CONFIGS = [
    ModelCfg(
        key="qwen",
        path=r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main",
        pre_layers=list(range(7, 20)),
        precursor_layers=[17, 18, 19],
        decision_layers=[20, 21, 22, 23, 24, 25],
        obs_layers=[13, 14, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    ),
    ModelCfg(
        key="llama",
        path=r"D:\model\Llama-3.2-1B-Instruct",
        pre_layers=list(range(4, 11)),
        precursor_layers=[9, 10, 11],
        decision_layers=[11, 12, 13, 14, 15],
        obs_layers=[4, 5, 8, 9, 10, 11, 12, 13, 14, 15],
    ),
    ModelCfg(
        key="gemma",
        path=r"D:\model\gemma-2-2b-it",
        pre_layers=list(range(7, 18)),
        precursor_layers=[15, 16, 17],
        decision_layers=[18, 19, 20, 21, 22, 23],
        obs_layers=[7, 8, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    ),
]

RUN_MODEL_KEYS = ["qwen"]  # smoke test first; use ["qwen", "llama", "gemma"] after validation

TOPK = 50
RANDOM_SEED = 42
MAX_PROMPTS = None  # set an integer such as 50 for smoke test
DTYPE = "auto"  # auto / float16 / bfloat16 / float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOCAL_FILES_ONLY = True

# For large vocabularies, gaussian null can be expensive but usually OK for 1B-2B models.
RUN_GAUSSIAN_NULL = True
RUN_ROWPERM_NULL = True
RUN_RANDOMK_NULL = True

# =========================
# 1. UTILITIES
# =========================


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_pearson(x, y) -> float:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    x = x[mask]
    y = y[mask]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def rankdata_simple(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    return ranks


def safe_spearman(x, y) -> float:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    return safe_pearson(rankdata_simple(x[mask]), rankdata_simple(y[mask]))


def cosine_distance_vector(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norms
    sim = Xn @ Xn.T
    dist = 1.0 - sim
    iu = np.triu_indices(X.shape[0], k=1)
    return dist[iu]


def pairwise_geometry_corr(X_ref: np.ndarray, X_proxy: np.ndarray) -> Dict[str, float]:
    if X_ref.shape[0] < 4 or X_proxy.shape[0] < 4:
        return {"pearson": np.nan, "spearman": np.nan}
    d_ref = cosine_distance_vector(X_ref)
    d_proxy = cosine_distance_vector(X_proxy)
    return {
        "pearson": safe_pearson(d_ref, d_proxy),
        "spearman": safe_spearman(d_ref, d_proxy),
    }


def pca_reduce_for_model(X: np.ndarray, max_dim: int = 64) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X)
    n, d = X.shape
    if d <= max_dim or n < 5:
        return X
    k = min(max_dim, n - 2, d)
    if k < 2:
        return X
    return PCA(n_components=k, random_state=RANDOM_SEED).fit_transform(X)


def get_weight_matrix(model) -> torch.Tensor:
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        return model.lm_head.weight.detach()
    return model.get_input_embeddings().weight.detach()


def resolve_dtype(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    return "auto"


def format_prompt(tokenizer, prompt: str, use_chat_template: bool) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = [{"role": "user", "content": prompt}]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return prompt
    return prompt


def first_token_id(tokenizer, text: str) -> Optional[int]:
    if not isinstance(text, str) or not text.strip():
        return None
    ids = tokenizer.encode(text.strip(), add_special_tokens=False)
    if not ids:
        return None
    return int(ids[0])


# =========================
# 2. EXTRACTION
# =========================


def load_model_and_tokenizer(cfg: ModelCfg):
    dtype = resolve_dtype(DTYPE)
    tokenizer = AutoTokenizer.from_pretrained(cfg.path, local_files_only=LOCAL_FILES_ONLY, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.path,
        local_files_only=LOCAL_FILES_ONLY,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    if DEVICE != "cuda":
        model.to(DEVICE)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def extract_last_token_hidden(model, tokenizer, prompts: List[str], use_chat_template: bool) -> np.ndarray:
    all_layers = []
    n_layers = None
    for idx, prompt in enumerate(prompts):
        text = format_prompt(tokenizer, prompt, use_chat_template)
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        # hs includes embedding state + transformer layers. We drop embedding state so layer 0 = first block output.
        layer_vecs = []
        for h in hs[1:]:
            layer_vecs.append(h[0, -1, :].detach().float().cpu().numpy())
        arr = np.stack(layer_vecs, axis=0)
        if n_layers is None:
            n_layers = arr.shape[0]
        all_layers.append(arr)
        if (idx + 1) % 10 == 0:
            print(f"  extracted {idx+1}/{len(prompts)} prompts")
    return np.stack(all_layers, axis=0)  # [N, L, D]


# =========================
# 3. FEATURE BUILDERS
# =========================


def valid_layers(layers: List[int], n_layers: int) -> List[int]:
    return [l for l in layers if 0 <= l < n_layers]


def mean_layer_feature(H: np.ndarray, layers: List[int]) -> np.ndarray:
    layers = valid_layers(layers, H.shape[1])
    if not layers:
        return np.zeros((H.shape[0], 1), dtype=np.float32)
    return H[:, layers, :].mean(axis=1)


def answer_logit_trajectory(H: np.ndarray, W: np.ndarray, clean_ids: List[Optional[int]], conflict_ids: List[Optional[int]], layers: List[int]) -> Optional[np.ndarray]:
    layers = valid_layers(layers, H.shape[1])
    if not layers:
        return None
    if any(x is None for x in clean_ids) or any(x is None for x in conflict_ids):
        return None
    N = H.shape[0]
    R = np.zeros((N, len(layers)), dtype=np.float32)
    W_np = W.astype(np.float32)
    for j, l in enumerate(layers):
        h = H[:, l, :].astype(np.float32)
        clean_vecs = W_np[np.array(clean_ids)]
        conflict_vecs = W_np[np.array(conflict_ids)]
        R[:, j] = np.sum(h * clean_vecs, axis=1) - np.sum(h * conflict_vecs, axis=1)
    return R


def compute_deltaU_from_R(R_decision: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    X = np.nan_to_num(R_decision.astype(np.float32))
    X = StandardScaler().fit_transform(X)
    if X.shape[0] < 4 or X.shape[1] < 2:
        return np.zeros(X.shape[0], dtype=np.float32), {"pc1_var": np.nan}
    pca = PCA(n_components=1, random_state=RANDOM_SEED)
    u = pca.fit_transform(X).reshape(-1)
    return u.astype(np.float32), {"pc1_var": float(pca.explained_variance_ratio_[0])}


def topk_indices_for_matrix(H_layer: np.ndarray, Wmat: np.ndarray, k: int) -> np.ndarray:
    logits = H_layer.astype(np.float32) @ Wmat.astype(np.float32).T
    k = min(k, logits.shape[1])
    idx = np.argpartition(-logits, kth=k-1, axis=1)[:, :k]
    # sort within topk for deterministic weighted features
    row = np.arange(idx.shape[0])[:, None]
    vals = logits[row, idx]
    order = np.argsort(-vals, axis=1)
    return idx[row, order]


def centroid_from_topk(Wmat: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return Wmat[idx].mean(axis=1).astype(np.float32)


def randomK_centroid(Wmat: np.ndarray, n: int, k: int, rng: np.random.RandomState) -> np.ndarray:
    V = Wmat.shape[0]
    idx = rng.randint(0, V, size=(n, min(k, V)))
    return centroid_from_topk(Wmat, idx)


def make_gaussian_R_like(W: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    # Matched row norms, global covariance approximated by per-dim std for speed.
    W = W.astype(np.float32)
    V, D = W.shape
    mu = W.mean(axis=0, keepdims=True)
    std = W.std(axis=0, keepdims=True) + 1e-6
    R = rng.normal(size=(V, D)).astype(np.float32) * std + mu
    target_norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-6
    R_norms = np.linalg.norm(R, axis=1, keepdims=True) + 1e-6
    R = R / R_norms * target_norms
    return R.astype(np.float32)


def jaccard_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = []
    for x, y in zip(a, b):
        sx = set(map(int, x))
        sy = set(map(int, y))
        inter = len(sx & sy)
        union = len(sx | sy)
        out.append(inter / union if union else 0.0)
    return np.array(out, dtype=np.float32)


def build_topk_flow_features(H: np.ndarray, W: np.ndarray, layers: List[int], kind: str, rng: np.random.RandomState, k: int) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    layers = valid_layers(layers, H.shape[1])
    if len(layers) < 2:
        return np.zeros((H.shape[0], 1), dtype=np.float32), []

    W_use = W.astype(np.float32)
    if kind == "rowperm":
        perm = rng.permutation(W_use.shape[0])
        W_use = W_use[perm]
    elif kind == "gaussian":
        W_use = make_gaussian_R_like(W_use, rng)

    centroids = []
    topk_sets = []
    geom_rows = []
    for l in layers:
        H_l = H[:, l, :].astype(np.float32)
        if kind == "randomK":
            idx = rng.randint(0, W_use.shape[0], size=(H.shape[0], min(k, W_use.shape[0])))
        else:
            idx = topk_indices_for_matrix(H_l, W_use, k)
        c = centroid_from_topk(W_use, idx)
        centroids.append(c)
        topk_sets.append(idx)
        corr = pairwise_geometry_corr(H_l, c)
        corr.update({"kind": kind, "layer": int(l)})
        geom_rows.append(corr)

    flow_feats = []
    for t in range(len(layers) - 1):
        c0 = centroids[t]
        c1 = centroids[t + 1]
        cossim = np.sum(c0 * c1, axis=1) / ((np.linalg.norm(c0, axis=1) + 1e-8) * (np.linalg.norm(c1, axis=1) + 1e-8))
        jac = jaccard_rows(topk_sets[t], topk_sets[t + 1])
        spread0 = np.linalg.norm(c0, axis=1)
        spread1 = np.linalg.norm(c1, axis=1)
        flow_feats.append(np.stack([1 - cossim, 1 - jac, spread1 - spread0], axis=1))
    return np.concatenate(flow_feats, axis=1).astype(np.float32), geom_rows


# =========================
# 4. EVALUATION
# =========================


def make_cv(n: int, groups: Optional[np.ndarray] = None, n_splits: int = 5):
    n_splits = max(2, min(n_splits, n))
    if groups is not None and len(set(groups)) >= n_splits:
        return GroupKFold(n_splits=n_splits).split(np.arange(n), groups=groups)
    return KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED).split(np.arange(n))


def cv_regression(X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None) -> Dict[str, float]:
    X = np.nan_to_num(np.asarray(X, dtype=np.float32))
    y = np.nan_to_num(np.asarray(y, dtype=np.float32).reshape(-1))
    if len(y) < 8 or np.std(y) < 1e-9:
        return {"r2": np.nan, "corr": np.nan, "mae": np.nan, "n": len(y)}
    Xr = pca_reduce_for_model(X, 64)
    preds = np.zeros_like(y, dtype=np.float32)
    for train_idx, test_idx in make_cv(len(y), groups):
        pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
        pipe.fit(Xr[train_idx], y[train_idx])
        preds[test_idx] = pipe.predict(Xr[test_idx]).astype(np.float32)
    return {
        "r2": float(r2_score(y, preds)),
        "corr": safe_pearson(y, preds),
        "mae": float(mean_absolute_error(y, preds)),
        "n": int(len(y)),
    }


def cv_classification(X: np.ndarray, labels: np.ndarray, groups: Optional[np.ndarray] = None) -> Dict[str, float]:
    X = np.nan_to_num(np.asarray(X, dtype=np.float32))
    labels = np.asarray(labels)
    mask = pd.Series(labels).notna().values
    X = X[mask]
    labels = labels[mask]
    if len(labels) < 8 or len(set(labels)) < 2:
        return {"acc": np.nan, "macro_f1": np.nan, "n": int(len(labels)), "n_classes": int(len(set(labels)))}
    le = LabelEncoder()
    y = le.fit_transform(labels.astype(str))
    groups2 = groups[mask] if groups is not None else None
    Xr = pca_reduce_for_model(X, 64)
    preds = np.zeros_like(y)
    for train_idx, test_idx in make_cv(len(y), groups2):
        clf = Pipeline([("scaler", StandardScaler()), ("ridgeclf", RidgeClassifier(alpha=10.0))])
        clf.fit(Xr[train_idx], y[train_idx])
        preds[test_idx] = clf.predict(Xr[test_idx])
    return {
        "acc": float(accuracy_score(y, preds)),
        "macro_f1": float(f1_score(y, preds, average="macro")),
        "n": int(len(y)),
        "n_classes": int(len(set(y))),
    }


def evaluate_proxy_rank(feature_map: Dict[str, np.ndarray], y_deltaU: Optional[np.ndarray], phase: Optional[np.ndarray], groups: Optional[np.ndarray]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    reg_rows = []
    cls_rows = []
    for name, X in feature_map.items():
        if y_deltaU is not None:
            r = cv_regression(X, y_deltaU, groups)
            r["proxy"] = name
            reg_rows.append(r)
        if phase is not None:
            c = cv_classification(X, phase, groups)
            c["proxy"] = name
            cls_rows.append(c)
    reg_df = pd.DataFrame(reg_rows).sort_values("corr", ascending=False) if reg_rows else pd.DataFrame()
    cls_df = pd.DataFrame(cls_rows).sort_values("macro_f1", ascending=False) if cls_rows else pd.DataFrame()
    return reg_df, cls_df


# =========================
# 5. MAIN PER MODEL
# =========================


def run_one_model(cfg: ModelCfg, df: pd.DataFrame):
    print(f"\n=== OPA-1 running model: {cfg.key} ===")
    out_dir = os.path.join(OUTPUT_DIR, cfg.key)
    ensure_dir(out_dir)

    model, tokenizer = load_model_and_tokenizer(cfg)
    prompts = df["prompt"].astype(str).tolist()
    H = extract_last_token_hidden(model, tokenizer, prompts, cfg.use_chat_template)
    np.save(os.path.join(out_dir, "hidden_last_token_layers.npy"), H)

    W_t = get_weight_matrix(model).float().cpu()
    W = W_t.numpy().astype(np.float32)

    n, n_layers, d = H.shape
    print(f"  hidden shape = {H.shape}, vocab W = {W.shape}")

    # Token IDs for clean/conflict answers.
    clean_ids = None
    conflict_ids = None
    R_decision = None
    deltaU = None
    deltaU_info = {}
    if "clean_answer" in df.columns and "conflict_answer" in df.columns:
        clean_ids = [first_token_id(tokenizer, x) for x in df["clean_answer"].tolist()]
        conflict_ids = [first_token_id(tokenizer, x) for x in df["conflict_answer"].tolist()]
        if not any(x is None for x in clean_ids) and not any(x is None for x in conflict_ids):
            R_decision = answer_logit_trajectory(H, W, clean_ids, conflict_ids, cfg.decision_layers)
            if R_decision is not None:
                deltaU, deltaU_info = compute_deltaU_from_R(R_decision)
                pd.DataFrame(R_decision, columns=[f"R_L{l}" for l in valid_layers(cfg.decision_layers, n_layers)]).to_csv(os.path.join(out_dir, "opa1_R_decision.csv"), index=False)
                pd.DataFrame({"prompt_id": df["prompt_id"].values if "prompt_id" in df.columns else np.arange(n), "deltaU_pc1": deltaU}).to_csv(os.path.join(out_dir, "opa1_deltaU_target.csv"), index=False)
        else:
            warnings.warn("Some clean/conflict answers could not be tokenized. DeltaR/H_shape disabled.")

    # Core feature map.
    feature_map: Dict[str, np.ndarray] = {}
    feature_map["hidden_pre_mean"] = mean_layer_feature(H, cfg.pre_layers)
    feature_map["hidden_precursor_mean"] = mean_layer_feature(H, cfg.precursor_layers)
    feature_map["hidden_decision_mean"] = mean_layer_feature(H, cfg.decision_layers)

    if R_decision is not None:
        feature_map["h_shape_decision_R"] = R_decision
        R_pre = answer_logit_trajectory(H, W, clean_ids, conflict_ids, cfg.pre_layers)
        if R_pre is not None and R_pre.shape[1] >= 2:
            feature_map["o_adv_pre_dR"] = np.diff(R_pre, axis=1).astype(np.float32)
            pd.DataFrame(R_pre, columns=[f"R_L{l}" for l in valid_layers(cfg.pre_layers, n_layers)]).to_csv(os.path.join(out_dir, "opa1_R_pre.csv"), index=False)

    # PSG-corrected TopK/VIM readback features.
    rng = np.random.RandomState(RANDOM_SEED)
    geom_rows_all = []
    obs_layers = valid_layers(cfg.obs_layers, n_layers)

    # Direct hidden baseline: pairwise hidden geometry compared with itself.
    # This is not a readback proxy; it is an upper-bound baseline for geometry preservation.
    for l in obs_layers:
        geom_rows_all.append({"kind": "directH", "layer": int(l), "pearson": 1.0, "spearman": 1.0})

    topk_real, geom_rows = build_topk_flow_features(H, W, obs_layers, "realW", rng, TOPK)
    feature_map["topk_realW_flow"] = topk_real
    geom_rows_all.extend(geom_rows)

    if RUN_ROWPERM_NULL:
        topk_rowperm, geom_rows = build_topk_flow_features(H, W, obs_layers, "rowperm", rng, TOPK)
        feature_map["topk_rowpermW_flow"] = topk_rowperm
        geom_rows_all.extend(geom_rows)

    if RUN_GAUSSIAN_NULL:
        topk_gauss, geom_rows = build_topk_flow_features(H, W, obs_layers, "gaussian", rng, TOPK)
        feature_map["topk_gaussianR_flow"] = topk_gauss
        geom_rows_all.extend(geom_rows)

    if RUN_RANDOMK_NULL:
        topk_rand, geom_rows = build_topk_flow_features(H, W, obs_layers, "randomK", rng, TOPK)
        feature_map["randomK_flow"] = topk_rand
        geom_rows_all.extend(geom_rows)

    # Save features.
    for name, X in feature_map.items():
        np.save(os.path.join(out_dir, f"features_{name}.npy"), np.asarray(X, dtype=np.float32))

    geom_df = pd.DataFrame(geom_rows_all)
    if not geom_df.empty:
        geom_df.to_csv(os.path.join(out_dir, "opa1_layer_readback_geometry.csv"), index=False)

    phase = df["phase_label"].values if "phase_label" in df.columns else None
    groups = df["split_group"].astype(str).values if "split_group" in df.columns else None

    if deltaU is None:
        print("  No clean_answer/conflict_answer columns detected. DeltaU/H_shape rank is disabled; PSG/readback geometry audit will still run.")
    reg_df, cls_df = evaluate_proxy_rank(feature_map, deltaU, phase, groups)
    if not reg_df.empty:
        reg_df.to_csv(os.path.join(out_dir, "opa1_proxy_rank_deltaU.csv"), index=False)
    if not cls_df.empty:
        cls_df.to_csv(os.path.join(out_dir, "opa1_proxy_rank_phase.csv"), index=False)

    # PSG residual summary.
    psg_summary = {}
    if not geom_df.empty:
        for kind in geom_df["kind"].unique():
            sub = geom_df[geom_df["kind"] == kind]
            psg_summary[f"{kind}_mean_pearson_hidden_geometry"] = float(sub["pearson"].mean())
            psg_summary[f"{kind}_mean_spearman_hidden_geometry"] = float(sub["spearman"].mean())
        if "realW" in geom_df["kind"].unique() and "rowperm" in geom_df["kind"].unique():
            psg_summary["real_minus_rowperm_pearson"] = psg_summary.get("realW_mean_pearson_hidden_geometry", np.nan) - psg_summary.get("rowperm_mean_pearson_hidden_geometry", np.nan)
        if "realW" in geom_df["kind"].unique() and "gaussian" in geom_df["kind"].unique():
            psg_summary["real_minus_gaussian_pearson"] = psg_summary.get("realW_mean_pearson_hidden_geometry", np.nan) - psg_summary.get("gaussian_mean_pearson_hidden_geometry", np.nan)
        if "realW" in geom_df["kind"].unique() and "randomK" in geom_df["kind"].unique():
            psg_summary["real_minus_randomK_pearson"] = psg_summary.get("realW_mean_pearson_hidden_geometry", np.nan) - psg_summary.get("randomK_mean_pearson_hidden_geometry", np.nan)

    summary = {
        "model": asdict(cfg),
        "n_prompts": int(n),
        "hidden_shape": list(H.shape),
        "deltaU_info": deltaU_info,
        "psg_summary": psg_summary,
        "proxy_rank_deltaU_top": reg_df.head(10).to_dict(orient="records") if not reg_df.empty else [],
        "proxy_rank_phase_top": cls_df.head(10).to_dict(orient="records") if not cls_df.empty else [],
        "interpretation_guardrail": "TopK/VIM results must be interpreted as PSG-corrected readback observables unless realW shows stable residual over rowperm/gaussian/directH controls.",
    }
    with open(os.path.join(out_dir, "opa1_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Free memory.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  done: {out_dir}")


# =========================
# 6. INPUT LOADER + ENTRY
# =========================


def load_prompt_inputs(paths: List[str]) -> pd.DataFrame:
    rows = []
    seen = set()
    for path_str in paths:
        path = os.path.expanduser(path_str)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prompt input not found: {path}")
        base = os.path.splitext(os.path.basename(path))[0]
        ext = os.path.splitext(path)[1].lower()

        if ext == ".csv":
            sub = pd.read_csv(path)
            if "prompt" not in sub.columns:
                # fallback for old text-column naming
                if "text" in sub.columns:
                    sub = sub.rename(columns={"text": "prompt"})
                else:
                    raise ValueError(f"CSV must contain prompt or text column: {path}")
            if "prompt_id" not in sub.columns:
                sub["prompt_id"] = [f"{base}_{i:05d}" for i in range(len(sub))]
            sub["source_file"] = base
            for _, r in sub.iterrows():
                prompt = str(r["prompt"]).strip()
                if not prompt or prompt in seen:
                    continue
                seen.add(prompt)
                item = r.to_dict()
                item["prompt"] = prompt
                rows.append(item)
        else:
            with open(path, "r", encoding="utf-8") as f:
                lines = [x.strip() for x in f.readlines()]
            local_idx = 0
            for line in lines:
                if not line or line in seen:
                    continue
                seen.add(line)
                rows.append({
                    "prompt_id": f"{base}_{local_idx:05d}",
                    "prompt": line,
                    "source_file": base,
                    "split_group": base,
                })
                local_idx += 1

    if not rows:
        raise RuntimeError("No prompts loaded from PROMPT_INPUT_PATHS.")

    df = pd.DataFrame(rows)
    if MAX_PROMPTS is not None:
        df = df.head(MAX_PROMPTS).copy()
    return df


def main():
    set_seed(RANDOM_SEED)
    ensure_dir(OUTPUT_DIR)

    df = load_prompt_inputs(PROMPT_INPUT_PATHS)
    if "prompt" not in df.columns:
        raise ValueError("Input data must contain a 'prompt' column after loading.")
    if "prompt_id" not in df.columns:
        df["prompt_id"] = [f"p{i:05d}" for i in range(len(df))]

    df.to_csv(os.path.join(OUTPUT_DIR, "opa1_input_used.csv"), index=False, encoding="utf-8-sig")
    print(f"Loaded {len(df)} prompts from {len(PROMPT_INPUT_PATHS)} input file(s).")
    print("Output directory:", OUTPUT_DIR)

    cfg_map = {c.key: c for c in MODEL_CONFIGS}
    for key in RUN_MODEL_KEYS:
        if key not in cfg_map:
            print(f"Skip unknown model key: {key}")
            continue
        try:
            run_one_model(cfg_map[key], df)
        except Exception as e:
            err_path = os.path.join(OUTPUT_DIR, f"ERROR_{key}.txt")
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(repr(e))
            print(f"ERROR in {key}: {e}")
            raise

    print("\nOPA-1 v0.2 finished.")


if __name__ == "__main__":
    main()
