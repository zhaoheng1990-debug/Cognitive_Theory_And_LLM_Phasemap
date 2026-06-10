# -*- coding: utf-8 -*-
"""
OA-5C.1: TopK Center Geometry Data Collection
=============================================

Goal
----
Collect real geometric data for OA-5C curvature / intrinsic-dimension audit.

This script re-runs the OA-5A prompt set through the model and exports, for
shallow layers L0-L6:

    1. TopK token ids
    2. TopK center vector C_k(H_l)
    3. TopK spread r_k
    4. TopK logit statistics

Why
---
Previous OA-5C could only use topk_ids / Jaccard continuity.
It could not run curvature analysis because center vectors were absent.

This script creates:

    oa5c1_topk_center_records.csv
    oa5c1_metadata_summary.csv

Then OA-5C.2 can run real geometry audit:
    - intrinsic dimension
    - graph geodesic vs Euclidean
    - local PCA curvature / flatness
    - continuity in center-vector space

Fixed local paths
-----------------
Edit MODEL_PATH if needed.
The input prompt dataset is expected at:

    C:\\Users\\ZH\\Desktop\\AGI\\python_script\\oa5a_prompt_dataset.csv

Outputs go to:

    C:\\Users\\ZH\\Desktop\\AGI\\python_script\\oa5c1_outputs
"""

import os
import gc
import ast
import json
import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

# ============================================================
# PATH CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\ZH\Desktop\AGI\python_script\tangent_audit_oa5a_outputs")

# IMPORTANT: edit this if your snapshot folder is different.
MODEL_PATH = Path(
    r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"
)

PROMPT_FILE = BASE_DIR / "oa5a_prompt_dataset.csv"

OUTPUT_DIR = BASE_DIR / "oa5c1_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_RECORDS = OUTPUT_DIR / "oa5c1_topk_center_records.csv"
OUT_META = OUTPUT_DIR / "oa5c1_metadata_summary.csv"

# ============================================================
# CONFIG
# ============================================================

RANDOM_SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

MAX_LEN = 256
BATCH_SIZE = 4

TRACK_LAYERS = list(range(0, 27))
K_LIST = [100, 500, 1000]

# To reduce CSV size, center vector can be stored as serialized JSON list.
# If you prefer wide columns center_0...center_d, set True.
SAVE_WIDE_CENTER_COLUMNS = True

# If wide columns are too large, set this True and wide False.
SAVE_CENTER_AS_JSON = False

# ============================================================
# SEED
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)

# ============================================================
# COLUMN INFERENCE
# ============================================================

def infer_col(df, candidates, required=True):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(
            f"Cannot infer column from {candidates}. Available columns: {list(df.columns)}"
        )
    return None

# ============================================================
# LOAD PROMPTS
# ============================================================

def load_prompt_dataset():
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"Missing prompt dataset: {PROMPT_FILE}")

    df = pd.read_csv(PROMPT_FILE)

    prompt_id_col = infer_col(df, ["prompt_id", "id", "sample_id", "idx"])
    topic_col = infer_col(df, ["topic", "entity", "subject"])
    task_col = infer_col(df, ["task", "task_type", "query_type"])
    prompt_col = infer_col(df, ["prompt", "text", "input", "query"])
    paraphrase_col = infer_col(
        df,
        ["paraphrase_id", "paraphrase", "surface", "variant", "template"],
        required=False,
    )

    out = df.copy()
    out = out.rename(columns={
        prompt_id_col: "prompt_id",
        topic_col: "topic",
        task_col: "task",
        prompt_col: "prompt",
    })

    if paraphrase_col and paraphrase_col != "paraphrase_id":
        out = out.rename(columns={paraphrase_col: "paraphrase_id"})
    elif "paraphrase_id" not in out.columns:
        out["paraphrase_id"] = np.arange(len(out))

    keep = ["prompt_id", "topic", "task", "paraphrase_id", "prompt"]
    out = out[keep].copy()

    out["prompt_id"] = out["prompt_id"].astype(str)
    out["topic"] = out["topic"].astype(str)
    out["task"] = out["task"].astype(str)
    out["paraphrase_id"] = out["paraphrase_id"].astype(str)
    out["prompt"] = out["prompt"].astype(str)

    return out

# ============================================================
# MODEL
# ============================================================

def load_model():
    print("=" * 72)
    print("OA-5C.1 TopK Center Geometry Data Collection")
    print("=" * 72)
    print("MODEL_PATH:", MODEL_PATH)
    print("PROMPT_FILE:", PROMPT_FILE)
    print("OUTPUT_DIR:", OUTPUT_DIR)
    print("DEVICE:", DEVICE)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "MODEL_PATH does not exist. Edit MODEL_PATH at top of script:\n"
            f"{MODEL_PATH}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None,
        local_files_only=True,
        trust_remote_code=True,
    )

    if DEVICE == "cpu":
        model.to(DEVICE)

    model.eval()

    num_layers = len(model.model.layers)
    if max(TRACK_LAYERS) >= num_layers:
        raise RuntimeError(
            f"TRACK_LAYERS includes {max(TRACK_LAYERS)}, but model has {num_layers} layers."
        )

    print("Num layers:", num_layers)
    print("LM head:", tuple(model.lm_head.weight.shape))

    return tokenizer, model

# ============================================================
# GEOMETRY HELPERS
# ============================================================

def get_last_positions(attention_mask):
    # left padding: last real token is always final position
    return torch.full(
        (attention_mask.shape[0],),
        attention_mask.shape[1] - 1,
        dtype=torch.long,
        device=attention_mask.device,
    )

def normalize_rows(x, eps=1e-9):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)

def compute_topk_geometry(hidden_np, W_np, k):
    """
    hidden_np: [B, D]
    W_np: [V, D]
    return:
      topk_ids: [B, k]
      center: [B, D]
      spread: [B]
      logit_mean/std/gap: [B]
    """
    # logits [B,V]
    logits = hidden_np @ W_np.T

    # topk via argpartition then sort within topk
    topk_unsorted = np.argpartition(-logits, kth=k-1, axis=1)[:, :k]
    topk_logits_unsorted = np.take_along_axis(logits, topk_unsorted, axis=1)
    order = np.argsort(-topk_logits_unsorted, axis=1)

    topk_ids = np.take_along_axis(topk_unsorted, order, axis=1)
    topk_logits = np.take_along_axis(logits, topk_ids, axis=1)

    # center
    W_topk = W_np[topk_ids]        # [B,k,D]
    center = W_topk.mean(axis=1)   # [B,D]

    # spread = mean 1-cos(w_i, center)
    Wn = W_topk / (np.linalg.norm(W_topk, axis=2, keepdims=True) + 1e-9)
    Cn = center / (np.linalg.norm(center, axis=1, keepdims=True) + 1e-9)
    cos = np.sum(Wn * Cn[:, None, :], axis=2)
    spread = np.mean(1.0 - cos, axis=1)

    logit_mean = topk_logits.mean(axis=1)
    logit_std = topk_logits.std(axis=1)
    logit_gap = topk_logits[:, 0] - topk_logits[:, -1]

    return (
        topk_ids.astype(np.int32),
        center.astype(np.float32),
        spread.astype(np.float32),
        logit_mean.astype(np.float32),
        logit_std.astype(np.float32),
        logit_gap.astype(np.float32),
    )

# ============================================================
# MAIN EXTRACTION
# ============================================================

def collect():
    prompts = load_prompt_dataset()
    tokenizer, model = load_model()

    W_np = model.lm_head.weight.detach().float().cpu().numpy().astype(np.float32)
    hidden_dim = W_np.shape[1]

    rows = []
    total = len(prompts)

    print("Prompt count:", total)
    print("Hidden dim:", hidden_dim)
    print("K_LIST:", K_LIST)
    print("TRACK_LAYERS:", TRACK_LAYERS)

    with torch.no_grad():
        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            batch_df = prompts.iloc[start:end].copy()
            texts = batch_df["prompt"].tolist()

            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
            ).to(DEVICE)

            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )

            # outputs.hidden_states[0] = embedding output
            # outputs.hidden_states[1] = after layer 0
            # We use transformer layer L as hidden_states[L+1]
            hidden_states = outputs.hidden_states
            last_idx = get_last_positions(inputs["attention_mask"])

            for layer in TRACK_LAYERS:
                hs = hidden_states[layer + 1]
                picked = hs[
                    torch.arange(hs.shape[0], device=hs.device),
                    last_idx,
                    :
                ].detach().float().cpu().numpy().astype(np.float32)

                for k in K_LIST:
                    (
                        topk_ids,
                        center,
                        spread,
                        logit_mean,
                        logit_std,
                        logit_gap,
                    ) = compute_topk_geometry(picked, W_np, k)

                    for bi, (_, meta) in enumerate(batch_df.iterrows()):
                        row = {
                            "prompt_id": str(meta["prompt_id"]),
                            "topic": str(meta["topic"]),
                            "task": str(meta["task"]),
                            "paraphrase_id": str(meta["paraphrase_id"]),
                            "prompt": str(meta["prompt"]),
                            "layer_index": int(layer),
                            "layer_name": f"L{layer}",
                            "k": int(k),
                            "topk_ids": json.dumps(topk_ids[bi].tolist(), ensure_ascii=False),
                            "spread": float(spread[bi]),
                            "logit_mean": float(logit_mean[bi]),
                            "logit_std": float(logit_std[bi]),
                            "logit_gap": float(logit_gap[bi]),
                        }

                        if SAVE_WIDE_CENTER_COLUMNS:
                            for d in range(hidden_dim):
                                row[f"center_{d}"] = float(center[bi, d])

                        if SAVE_CENTER_AS_JSON:
                            row["center_vec"] = json.dumps(
                                center[bi].round(7).tolist(),
                                ensure_ascii=False,
                            )

                        rows.append(row)

            if start % (BATCH_SIZE * 5) == 0:
                print(f"Processed {end}/{total}")

            del outputs, hidden_states, inputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_RECORDS, index=False, encoding="utf-8-sig")

    meta_rows = [{
        "n_prompts": int(total),
        "n_rows": int(len(out_df)),
        "hidden_dim": int(hidden_dim),
        "track_layers": json.dumps(TRACK_LAYERS),
        "k_list": json.dumps(K_LIST),
        "model_path": str(MODEL_PATH),
        "prompt_file": str(PROMPT_FILE),
        "save_wide_center_columns": bool(SAVE_WIDE_CENTER_COLUMNS),
        "save_center_as_json": bool(SAVE_CENTER_AS_JSON),
        "output_records": str(OUT_RECORDS),
    }]
    pd.DataFrame(meta_rows).to_csv(OUT_META, index=False, encoding="utf-8-sig")

    print("\nSaved:")
    print(OUT_RECORDS)
    print(OUT_META)
    print("\nRows:", len(out_df))
    print("Columns:", len(out_df.columns))
    print("Done.")

# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    collect()
