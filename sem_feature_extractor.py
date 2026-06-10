# -*- coding: utf-8 -*-
"""
SEM Feature Extractor
=====================

Default target
--------------
Input:
  sem5a2_outputs/sem5a2_dataset.csv

Output:
  sem5a2_outputs/sem5a2_features.csv

This script extracts the feature format expected by SEM-5A / 5A.1 / 5A.2:

For each prompt and each layer L0-L25:
  k100_L{l}_spread
  k100_L{l}_entropy
  k100_L{l}_gap
  k100_L{l}_meanlogit
  k100_L{l}_center_norm
  k100_L{l}_center_shift_cosdist
  k100_L{l}_center_shift_l2
  k100_L{l}_jaccard_prev
  k100_L{l}_jaccard_dist_prev

  same for k500

Then global PCA over TopK center vectors:
  k100_L{l}_centerPC1 ... centerPC16
  k500_L{l}_centerPC1 ... centerPC16

Important
---------
- This uses hidden_states at the final prompt token position.
- It projects each layer hidden state through lm_head to get vocabulary logits.
- It reads only prompts; no generation is performed.
- It is designed for Qwen2.5-1.5B-Instruct by default.
- If your model path differs, edit MODEL_PATH below.

Run
---
  python sem_feature_extractor.py

Expected local model path example:
  D:\\model\\models--Qwen--Qwen2.5-1.5B-Instruct\\main

If this path is wrong on your machine, replace MODEL_PATH.
"""

from __future__ import annotations

import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# Hardcoded config
# =============================================================================

MODEL_PATH = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

INPUT_CSV = Path("sem5a2_outputs/sem5a2_dataset.csv")
OUTPUT_CSV = Path("sem5a2_outputs/sem5a2_features.csv")
OUTPUT_DIR = Path("sem5a2_outputs")

# For old SEM-3A / SEM-5A runs, you can switch to:
# INPUT_CSV = Path("sem3a_outputs/sem3a_dataset.csv")
# OUTPUT_CSV = Path("sem3a_outputs/sem3a_features.csv")
# OUTPUT_DIR = Path("sem3a_outputs")

PROMPT_COL = "prompt"
ID_COL = "row_id"

MAX_LAYER = 25
TOPKS = (100, 500)
CENTER_PCA_DIM = 16

BATCH_SIZE = 4
MAX_LENGTH = 256

DTYPE = torch.float16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# Save extra metadata/debug files.
SAVE_AUDIT_JSON = True


# =============================================================================
# Utility
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def cosine_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an < eps or bn < eps:
        return 1.0
    return float(1.0 - np.dot(a, b) / (an * bn))


def softmax_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    # logits: [B, K]
    p = torch.softmax(logits.float(), dim=-1)
    h = -(p * torch.log(torch.clamp(p, min=1e-12))).sum(dim=-1)
    return h / math.log(logits.shape[-1])


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 and len(b) == 0:
        return 1.0
    sa = set(map(int, a.tolist()))
    sb = set(map(int, b.tolist()))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def get_lm_head_weight(model) -> torch.Tensor:
    """
    Returns W with shape [vocab, hidden].
    For tied embeddings, model.get_output_embeddings().weight works.
    """
    out_emb = model.get_output_embeddings()
    if out_emb is None or not hasattr(out_emb, "weight"):
        raise RuntimeError("Could not access model output embedding / lm_head weight.")
    return out_emb.weight


def last_nonpad_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    # attention_mask: [B, T]
    return attention_mask.long().sum(dim=1) - 1


def format_prompt(tokenizer, prompt: str) -> str:
    """
    For instruct models, chat template is often useful.
    But for internal geometry audits, adding chat template can add system/instruction
    tokens. Default: raw prompt.

    If you want chat template, uncomment below.
    """
    # try:
    #     return tokenizer.apply_chat_template(
    #         [{"role": "user", "content": prompt}],
    #         tokenize=False,
    #         add_generation_prompt=True,
    #     )
    # except Exception:
    #     return prompt
    return str(prompt)


# =============================================================================
# Core extraction
# =============================================================================

@torch.inference_mode()
def extract_batch_features(
    prompts: List[str],
    tokenizer,
    model,
    W: torch.Tensor,
    max_layer: int,
    topks: Tuple[int, ...],
) -> Tuple[List[Dict], Dict[int, Dict[int, List[np.ndarray]]]]:
    """
    Returns:
      rows: list of per-prompt feature dicts without centerPC columns
      centers_by_k_layer: {k: {layer: [center_vec_per_prompt]}}
    """
    texts = [format_prompt(tokenizer, p) for p in prompts]
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    last_idx = last_nonpad_indices(enc["attention_mask"])

    outputs = model(
        **enc,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    hidden_states = outputs.hidden_states  # tuple: embeddings + layers
    n_available = len(hidden_states) - 1
    Lmax = min(max_layer, n_available)

    B = enc["input_ids"].shape[0]
    rows = [dict() for _ in range(B)]
    centers_by_k_layer = {k: {l: [] for l in range(Lmax + 1)} for k in topks}

    prev_top_ids = {k: [None for _ in range(B)] for k in topks}
    prev_centers = {k: [None for _ in range(B)] for k in topks}

    W_float = W.float()

    for l in range(Lmax + 1):
        # hidden_states[0] is embedding output; hidden_states[l+1] after layer l.
        h_all = hidden_states[l + 1] if l + 1 < len(hidden_states) else hidden_states[-1]
        h = h_all[torch.arange(B, device=DEVICE), last_idx]  # [B, D]

        logits = h.float() @ W_float.T  # [B, V]

        for k in topks:
            top_vals, top_ids = torch.topk(logits, k=k, dim=-1)
            top_emb = W_float[top_ids]  # [B, K, D]
            center = top_emb.mean(dim=1)  # [B, D]
            center_norm = torch.linalg.norm(center, dim=-1)
            emb_norm = torch.nn.functional.normalize(top_emb, dim=-1)
            center_normed = torch.nn.functional.normalize(center, dim=-1)
            cos_to_center = (emb_norm * center_normed[:, None, :]).sum(dim=-1)
            spread = (1.0 - cos_to_center).mean(dim=-1)

            entropy = softmax_entropy_from_logits(top_vals)
            gap = top_vals[:, 0] - top_vals[:, -1]
            meanlogit = top_vals.mean(dim=-1)

            top_ids_np = top_ids.detach().cpu().numpy()
            center_np = center.detach().cpu().numpy()
            spread_np = spread.detach().cpu().numpy()
            entropy_np = entropy.detach().cpu().numpy()
            gap_np = gap.detach().cpu().numpy()
            meanlogit_np = meanlogit.detach().cpu().numpy()
            center_norm_np = center_norm.detach().cpu().numpy()

            for i in range(B):
                prefix = f"k{k}_L{l}"
                rows[i][f"{prefix}_spread"] = float(spread_np[i])
                rows[i][f"{prefix}_entropy"] = float(entropy_np[i])
                rows[i][f"{prefix}_gap"] = float(gap_np[i])
                rows[i][f"{prefix}_meanlogit"] = float(meanlogit_np[i])
                rows[i][f"{prefix}_center_norm"] = float(center_norm_np[i])

                if prev_centers[k][i] is None:
                    shift_cosdist = 0.0
                    shift_l2 = 0.0
                    jac_prev = 1.0
                    jac_dist_prev = 0.0
                else:
                    shift_cosdist = cosine_distance(center_np[i], prev_centers[k][i])
                    shift_l2 = float(np.linalg.norm(center_np[i] - prev_centers[k][i]))
                    jac_prev = jaccard(top_ids_np[i], prev_top_ids[k][i])
                    jac_dist_prev = 1.0 - jac_prev

                rows[i][f"{prefix}_center_shift_cosdist"] = float(shift_cosdist)
                rows[i][f"{prefix}_center_shift_l2"] = float(shift_l2)
                rows[i][f"{prefix}_jaccard_prev"] = float(jac_prev)
                rows[i][f"{prefix}_jaccard_dist_prev"] = float(jac_dist_prev)

                prev_centers[k][i] = center_np[i]
                prev_top_ids[k][i] = top_ids_np[i]
                centers_by_k_layer[k][l].append(center_np[i])

        # Reduce temporary memory.
        del logits

    return rows, centers_by_k_layer


def fit_center_pcas(
    all_centers: Dict[int, Dict[int, List[np.ndarray]]],
    center_pca_dim: int,
) -> Dict[int, PCA]:
    """
    Fit one PCA per top-k over all layers and prompts.
    This keeps PC axes comparable across layers for a given k.
    """
    pcas = {}
    for k, layer_dict in all_centers.items():
        mats = []
        for l, centers in layer_dict.items():
            if centers:
                mats.append(np.vstack(centers))
        if not mats:
            continue
        X = np.vstack(mats)
        dim = min(center_pca_dim, X.shape[0], X.shape[1])
        pca = PCA(n_components=dim, random_state=RANDOM_SEED)
        pca.fit(X)
        pcas[k] = pca
    return pcas


def add_center_pca_columns(
    features: pd.DataFrame,
    centers_per_row: Dict[int, Dict[int, List[np.ndarray]]],
    pcas: Dict[int, PCA],
    row_order: List[int],
    max_layer: int,
    center_pca_dim: int,
) -> pd.DataFrame:
    """
    centers_per_row[k][l] is a list aligned to row_order.
    """
    for k, pca in pcas.items():
        actual_dim = pca.n_components_
        for l in range(max_layer + 1):
            centers = centers_per_row.get(k, {}).get(l, [])
            if not centers:
                continue
            X = np.vstack(centers)
            Z = pca.transform(X)

            # Map row_order -> PCA values.
            for j in range(actual_dim):
                col = f"k{k}_L{l}_centerPC{j+1}"
                features[col] = Z[:, j].astype(np.float32)

            # Pad missing PCs if actual_dim < requested dim.
            for j in range(actual_dim, center_pca_dim):
                col = f"k{k}_L{l}_centerPC{j+1}"
                features[col] = 0.0

    return features


def extract_features() -> None:
    ensure_dir(OUTPUT_DIR)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV.resolve()}")

    df = pd.read_csv(INPUT_CSV)
    if PROMPT_COL not in df.columns:
        raise ValueError(f"Missing prompt column: {PROMPT_COL}")

    if ID_COL not in df.columns:
        df[ID_COL] = np.arange(len(df))

    print("=== SEM FEATURE EXTRACTOR ===")
    print(f"Input:  {INPUT_CSV}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Rows:   {len(df)}")
    print(f"Device: {DEVICE}")
    print(f"Model:  {MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE if DEVICE == "cuda" else torch.float32,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    model.eval()
    if DEVICE != "cuda":
        model.to(DEVICE)

    W = get_lm_head_weight(model)
    W = W.to(DEVICE)

    # Determine actual max layer.
    n_layers = getattr(model.config, "num_hidden_layers", None)
    if n_layers is None:
        n_layers = MAX_LAYER + 1
    actual_max_layer = min(MAX_LAYER, int(n_layers) - 1)
    print(f"Using layers L0-L{actual_max_layer}")

    all_rows = []
    all_centers = {k: {l: [] for l in range(actual_max_layer + 1)} for k in TOPKS}
    row_order = []

    prompts = df[PROMPT_COL].fillna("").astype(str).tolist()
    ids = df[ID_COL].tolist()

    start_time = time.time()

    for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Extracting"):
        end = min(start + BATCH_SIZE, len(df))
        batch_prompts = prompts[start:end]
        batch_rows, batch_centers = extract_batch_features(
            batch_prompts,
            tokenizer,
            model,
            W,
            actual_max_layer,
            TOPKS,
        )

        for i, feat in enumerate(batch_rows):
            rid = ids[start + i]
            feat[ID_COL] = rid
            all_rows.append(feat)
            row_order.append(rid)

        for k in TOPKS:
            for l in range(actual_max_layer + 1):
                all_centers[k][l].extend(batch_centers[k][l])

    features = pd.DataFrame(all_rows)

    # Ensure row_id first.
    ordered_cols = [ID_COL] + [c for c in features.columns if c != ID_COL]
    features = features[ordered_cols]

    print("Fitting center PCA...")
    pcas = fit_center_pcas(all_centers, CENTER_PCA_DIM)
    features = add_center_pca_columns(
        features,
        all_centers,
        pcas,
        row_order,
        actual_max_layer,
        CENTER_PCA_DIM,
    )

    # Merge metadata-like fields? No: SEM audit scripts merge dataset+features by row_id.
    features.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    elapsed = time.time() - start_time
    print(f"Saved features: {OUTPUT_CSV}")
    print(f"Feature shape: {features.shape}")
    print(f"Elapsed: {elapsed:.1f}s")

    if SAVE_AUDIT_JSON:
        audit = {
            "input_csv": str(INPUT_CSV),
            "output_csv": str(OUTPUT_CSV),
            "model_path": MODEL_PATH,
            "device": DEVICE,
            "dtype": str(DTYPE),
            "n_rows": int(len(df)),
            "feature_shape": list(features.shape),
            "max_layer_requested": MAX_LAYER,
            "max_layer_used": actual_max_layer,
            "topks": list(TOPKS),
            "center_pca_dim": CENTER_PCA_DIM,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "pca_explained_variance": {
                str(k): pcas[k].explained_variance_ratio_.tolist()
                for k in pcas
            },
        }
        (OUTPUT_DIR / "sem_feature_extractor_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved audit: {OUTPUT_DIR / 'sem_feature_extractor_audit.json'}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        extract_features()
