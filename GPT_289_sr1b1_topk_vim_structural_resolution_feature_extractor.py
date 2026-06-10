# -*- coding: utf-8 -*-
"""
SR-1B.1: TopK/VIM Structural Resolution Feature Extractor

Purpose
-------
Extract model-internal TopK/VIM features for StructuralResolution estimation.

This script reads:
    sr1a_outputs/sr1a1_enriched_structural_resolution_labels.csv

and writes:
    sr1b1_outputs/sr1b1_topk_vim_features.csv
    sr1b1_outputs/sr1b1_topk_vim_feature_manifest.json
    sr1b1_outputs/sr1b1_summary.json

Theory
------
SR-1B.0 showed tabular metadata is strong in-distribution but weak when
reuse_group is held out. Therefore SR-1B.1 extracts model-internal features:

    Prompt -> TopK(H_l W^T) -> VIM neighborhood geometry

The aim is to later test whether TopK/VIM features improve cross-boundary
StructuralResolution beyond metadata leakage.

Notes
-----
- Hardcoded for local Windows execution.
- No command-line arguments required.
- Uses Qwen2.5-1.5B-Instruct by default.
- Extracts shallow/init, mid, and boundary-window VIM features.
"""

import os
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# =========================
# Hardcoded paths
# =========================

BASE_DIR = Path(r"C:\Users\ZH\Desktop\AGI\python_script")

MODEL_PATH = Path(r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main")

INPUT_CSV = BASE_DIR / "sr1a_outputs" / "sr1a1_enriched_structural_resolution_labels.csv"

OUT_DIR = BASE_DIR / "sr1b1_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FEATURES = OUT_DIR / "sr1b1_topk_vim_features.csv"
OUT_MANIFEST = OUT_DIR / "sr1b1_topk_vim_feature_manifest.json"
OUT_SUMMARY = OUT_DIR / "sr1b1_summary.json"


# =========================
# Feature config
# =========================

MAX_ROWS = None

MAX_LENGTH = 512

BATCH_SIZE = 2

TOPK_LIST = [50, 100, 500]

# hidden_states index convention:
# hidden_states[0] = embedding output
# hidden_states[1] = after transformer block 0
# ...
# Qwen2.5-1.5B has 28 blocks, so hidden_states length is usually 29.
INIT_LAYERS = list(range(0, 7))
MID_LAYERS = [7, 10, 13, 16, 19]
BOUNDARY_LAYERS = [20, 21, 22, 23, 24, 25]

WINDOWS = {
    "init_0_6": INIT_LAYERS,
    "mid_sparse_7_19": MID_LAYERS,
    "boundary_20_25": BOUNDARY_LAYERS,
    "init_plus_mid": INIT_LAYERS + MID_LAYERS,
    "mid_plus_boundary": MID_LAYERS + BOUNDARY_LAYERS,
}

SEED = 42


# =========================
# Utilities
# =========================

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_float(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, torch.Tensor):
            x = x.detach().float().cpu().item()
        return float(x)
    except Exception:
        return np.nan


def entropy_from_logits(logits_1d: torch.Tensor) -> float:
    p = torch.softmax(logits_1d.float(), dim=-1)
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum()
    return safe_float(ent)


def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def jaccard_ids(a, b) -> float:
    sa = set([int(x) for x in a])
    sb = set([int(x) for x in b])
    if not sa and not sb:
        return np.nan
    return float(len(sa & sb) / max(1, len(sa | sb)))


def slope_of_values(values: List[float]) -> float:
    vals = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(vals)
    if mask.sum() < 2:
        return np.nan
    x = np.arange(len(vals), dtype=np.float64)[mask]
    y = vals[mask]
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom <= 1e-12:
        return np.nan
    return float(np.dot(x, y - y.mean()) / denom)


def summarize_series(prefix: str, values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(arr)
    if mask.sum() == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_slope": np.nan,
            f"{prefix}_first": np.nan,
            f"{prefix}_last": np.nan,
            f"{prefix}_delta_last_first": np.nan,
        }
    good = arr[mask]
    first = arr[np.where(mask)[0][0]]
    last = arr[np.where(mask)[0][-1]]
    return {
        f"{prefix}_mean": float(np.mean(good)),
        f"{prefix}_std": float(np.std(good)),
        f"{prefix}_min": float(np.min(good)),
        f"{prefix}_max": float(np.max(good)),
        f"{prefix}_slope": slope_of_values(values),
        f"{prefix}_first": float(first),
        f"{prefix}_last": float(last),
        f"{prefix}_delta_last_first": float(last - first),
    }


def token_text_preview(tokenizer, ids, max_items=20):
    out = []
    for x in list(ids)[:max_items]:
        try:
            out.append(tokenizer.decode([int(x)]))
        except Exception:
            out.append(str(int(x)))
    return out


# =========================
# Model loading
# =========================

def load_model_and_tokenizer():
    print(f"[MODEL] loading from: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=None,
    )

    model.to(device)
    model.eval()

    out_emb = model.get_output_embeddings()
    if out_emb is None:
        raise RuntimeError("model.get_output_embeddings() returned None")

    W = out_emb.weight.detach()
    W = W.to(device)

    # Use normalized W for center/spread cosine calculations.
    W_norm = torch.nn.functional.normalize(W.float(), dim=-1)

    print(f"[MODEL] device={device}, dtype={dtype}")
    print(f"[MODEL] vocab={W.shape[0]}, hidden={W.shape[1]}")

    return model, tokenizer, W, W_norm, device


# =========================
# Feature extraction
# =========================

@torch.no_grad()
def forward_batch(model, tokenizer, prompts: List[str], device: str):
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True, use_cache=False)

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    last_positions = attention_mask.sum(dim=1) - 1
    hidden_states = out.hidden_states

    return input_ids, attention_mask, last_positions, hidden_states


def extract_one_prompt_features(
    row_meta: Dict,
    hidden_states,
    sample_idx: int,
    last_pos: int,
    W: torch.Tensor,
    W_norm: torch.Tensor,
    tokenizer,
) -> Dict[str, float]:
    feats = {}

    task_id = row_meta.get("task_id", "")
    feats["task_id"] = task_id

    n_layers_available = len(hidden_states)
    selected_layers = sorted(set(INIT_LAYERS + MID_LAYERS + BOUNDARY_LAYERS))
    selected_layers = [l for l in selected_layers if l < n_layers_available]

    # Store per-k, per-layer neighborhood objects for delta metrics.
    cache = {}

    for layer_idx in selected_layers:
        h = hidden_states[layer_idx][sample_idx, last_pos, :]
        h_float = h.float()

        logits = torch.matmul(h_float, W.float().T)

        for k in TOPK_LIST:
            k_eff = min(k, logits.numel())
            vals, ids = torch.topk(logits, k=k_eff, dim=-1)

            ids_cpu = ids.detach().cpu().numpy().astype(np.int64)
            vals_float = vals.float()

            tok_vec = W_norm[ids].float()
            center = tok_vec.mean(dim=0)
            center_norm = torch.linalg.norm(center).clamp_min(1e-12)
            center_unit = center / center_norm

            cos_to_center = torch.matmul(tok_vec, center_unit)
            spread = (1.0 - cos_to_center).mean()

            prefix = f"k{k}_L{layer_idx}"

            feats[f"{prefix}_top_logit"] = safe_float(vals_float[0])
            feats[f"{prefix}_mean_logit"] = safe_float(vals_float.mean())
            feats[f"{prefix}_std_logit"] = safe_float(vals_float.std(unbiased=False))
            feats[f"{prefix}_min_logit"] = safe_float(vals_float.min())
            feats[f"{prefix}_logit_range"] = safe_float(vals_float.max() - vals_float.min())
            feats[f"{prefix}_entropy"] = entropy_from_logits(vals_float)
            feats[f"{prefix}_center_norm"] = safe_float(center_norm)
            feats[f"{prefix}_spread"] = safe_float(spread)
            feats[f"{prefix}_top_id"] = int(ids_cpu[0])

            cache[(k, layer_idx)] = {
                "ids": ids_cpu,
                "center": center.detach().cpu().numpy().astype(np.float32),
                "entropy": feats[f"{prefix}_entropy"],
                "spread": feats[f"{prefix}_spread"],
                "mean_logit": feats[f"{prefix}_mean_logit"],
                "std_logit": feats[f"{prefix}_std_logit"],
            }

    # Consecutive layer transition features.
    for k in TOPK_LIST:
        layers_k = [l for l in selected_layers if (k, l) in cache]
        for a, b in zip(layers_k[:-1], layers_k[1:]):
            ca = cache[(k, a)]
            cb = cache[(k, b)]
            prefix = f"k{k}_T{a}_{b}"
            feats[f"{prefix}_jaccard"] = jaccard_ids(ca["ids"], cb["ids"])
            feats[f"{prefix}_center_cos"] = cosine_np(ca["center"], cb["center"])
            feats[f"{prefix}_entropy_delta"] = cb["entropy"] - ca["entropy"]
            feats[f"{prefix}_spread_delta"] = cb["spread"] - ca["spread"]
            feats[f"{prefix}_mean_logit_delta"] = cb["mean_logit"] - ca["mean_logit"]

    # Window summaries.
    for k in TOPK_LIST:
        for wname, layers in WINDOWS.items():
            layers = [l for l in layers if (k, l) in cache]
            if not layers:
                continue

            entropy_vals = [cache[(k, l)]["entropy"] for l in layers]
            spread_vals = [cache[(k, l)]["spread"] for l in layers]
            mean_logit_vals = [cache[(k, l)]["mean_logit"] for l in layers]
            std_logit_vals = [cache[(k, l)]["std_logit"] for l in layers]

            feats.update(summarize_series(f"k{k}_{wname}_entropy", entropy_vals))
            feats.update(summarize_series(f"k{k}_{wname}_spread", spread_vals))
            feats.update(summarize_series(f"k{k}_{wname}_mean_logit", mean_logit_vals))
            feats.update(summarize_series(f"k{k}_{wname}_std_logit", std_logit_vals))

            centers = [cache[(k, l)]["center"] for l in layers]

            if len(centers) >= 2:
                first_last_cos = cosine_np(centers[0], centers[-1])
                step_cos = [
                    cosine_np(centers[i], centers[i + 1])
                    for i in range(len(centers) - 1)
                ]
                step_dist = [
                    float(np.linalg.norm(centers[i + 1] - centers[i]))
                    for i in range(len(centers) - 1)
                ]

                feats[f"k{k}_{wname}_center_first_last_cos"] = first_last_cos
                feats[f"k{k}_{wname}_center_path_len"] = float(np.nansum(step_dist))
                feats[f"k{k}_{wname}_center_step_cos_mean"] = float(np.nanmean(step_cos))
                feats[f"k{k}_{wname}_center_step_dist_mean"] = float(np.nanmean(step_dist))
                feats[f"k{k}_{wname}_center_step_dist_max"] = float(np.nanmax(step_dist))

                ids_first = cache[(k, layers[0])]["ids"]
                ids_last = cache[(k, layers[-1])]["ids"]
                feats[f"k{k}_{wname}_ids_first_last_jaccard"] = jaccard_ids(ids_first, ids_last)

            else:
                feats[f"k{k}_{wname}_center_first_last_cos"] = np.nan
                feats[f"k{k}_{wname}_center_path_len"] = np.nan
                feats[f"k{k}_{wname}_center_step_cos_mean"] = np.nan
                feats[f"k{k}_{wname}_center_step_dist_mean"] = np.nan
                feats[f"k{k}_{wname}_center_step_dist_max"] = np.nan
                feats[f"k{k}_{wname}_ids_first_last_jaccard"] = np.nan

    return feats


def main():
    set_seed(SEED)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"INPUT_CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    if MAX_ROWS is not None:
        df = df.head(MAX_ROWS).copy()

    if "request" not in df.columns:
        raise ValueError("Input file must contain a request column.")

    df["request"] = df["request"].fillna("").astype(str)
    df["task_id"] = df["task_id"].astype(str)

    print(f"[INPUT] rows={len(df)}")
    print(f"[INPUT] columns={list(df.columns)}")

    model, tokenizer, W, W_norm, device = load_model_and_tokenizer()

    all_features = []
    start = time.time()

    prompts = df["request"].tolist()

    for start_idx in range(0, len(df), BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, len(df))
        batch_prompts = prompts[start_idx:end_idx]

        print(f"[BATCH] {start_idx}..{end_idx - 1}")

        input_ids, attention_mask, last_positions, hidden_states = forward_batch(
            model,
            tokenizer,
            batch_prompts,
            device,
        )

        for local_i, global_i in enumerate(range(start_idx, end_idx)):
            row = df.iloc[global_i].to_dict()
            last_pos = int(last_positions[local_i].detach().cpu().item())

            feats = extract_one_prompt_features(
                row_meta=row,
                hidden_states=hidden_states,
                sample_idx=local_i,
                last_pos=last_pos,
                W=W,
                W_norm=W_norm,
                tokenizer=tokenizer,
            )

            # Keep target/meta columns for downstream training.
            meta_cols = [
                "task_id",
                "reuse_group",
                "task_family",
                "target_operator",
                "concept",
                "boundary_strength",
                "sr_policy_v1_action",
                "gain_oracle_action",
                "best_policy_action",
                "best_policy_id",
                "correct_vs_no",
                "correct_vs_wrong",
                "correct_vs_random",
                "best_policy_gain",
                "best_policy_vs_no",
                "best_policy_vs_wrong",
                "best_policy_vs_random",
            ]

            for c in meta_cols:
                if c in row and c != "task_id":
                    feats[c] = row[c]

            all_features.append(feats)

    feat_df = pd.DataFrame(all_features)

    # Put meta columns first.
    meta_first = [
        "task_id",
        "reuse_group",
        "task_family",
        "target_operator",
        "concept",
        "boundary_strength",
        "sr_policy_v1_action",
        "gain_oracle_action",
        "best_policy_action",
        "best_policy_id",
        "correct_vs_no",
        "correct_vs_wrong",
        "correct_vs_random",
        "best_policy_gain",
        "best_policy_vs_no",
        "best_policy_vs_wrong",
        "best_policy_vs_random",
    ]
    cols = [c for c in meta_first if c in feat_df.columns] + [
        c for c in feat_df.columns if c not in meta_first
    ]
    feat_df = feat_df[cols]

    feat_df.to_csv(OUT_FEATURES, index=False, encoding="utf-8-sig")

    feature_cols = [c for c in feat_df.columns if c not in meta_first]

    manifest = {
        "input_csv": str(INPUT_CSV),
        "model_path": str(MODEL_PATH),
        "output_features": str(OUT_FEATURES),
        "n_rows": int(len(feat_df)),
        "n_feature_cols": int(len(feature_cols)),
        "topk_list": TOPK_LIST,
        "init_layers": INIT_LAYERS,
        "mid_layers": MID_LAYERS,
        "boundary_layers": BOUNDARY_LAYERS,
        "windows": WINDOWS,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "feature_cols": feature_cols,
    }

    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    summary = {
        "verdict": "SR1B1_FEATURES_EXTRACTED",
        "n_rows": int(len(feat_df)),
        "n_feature_cols": int(len(feature_cols)),
        "elapsed_sec": float(time.time() - start),
        "output_features": str(OUT_FEATURES),
        "output_manifest": str(OUT_MANIFEST),
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[SAVED]")
    print(f"  features : {OUT_FEATURES}")
    print(f"  manifest : {OUT_MANIFEST}")
    print(f"  summary  : {OUT_SUMMARY}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
