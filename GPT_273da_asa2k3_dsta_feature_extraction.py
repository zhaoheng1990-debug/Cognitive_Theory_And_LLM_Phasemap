# -*- coding: utf-8 -*-
r"""
DA-ASA-2K.3 Expanded DenseDSTA Feature Extraction

Purpose
-------
Extract signed / transport / attractor-style DSTA features for the expanded
DA-ASA-2K.1 / 2K.2 sample table.

Input:
    C:\Users\ZH\Desktop\AGI\python_script\da_asa2k2_baseline_outputs\da_asa2k2_baseline_scores.csv

Output:
    C:\Users\ZH\Desktop\AGI\python_script\da_asa2k3_dsta_outputs\

Notes
-----
This script does not generate policy_results or LPF labels. It only creates
expanded DenseDSTA features for downstream 2K.4 / 2K.5 stages.
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# =========================
# Config
# =========================

ROOT = Path(r"C:\Users\ZH\Desktop\AGI\python_script")
MODEL_PATH = Path(r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main")

INPUT_CSV = ROOT / "da_asa2k2_baseline_outputs" / "da_asa2k2_baseline_scores.csv"
BASELINE_DIAG_JSON = ROOT / "da_asa2k2_baseline_outputs" / "da_asa2k2_diagnostics.json"

OUT_DIR = ROOT / "da_asa2k3_dsta_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 768
BATCH_SIZE = 2
DTYPE = torch.float16
RANDOM_SEED = 20260606

# Layer windows are Qwen2.5-1.5B style. If the loaded model has a different
# number of layers, windows are clipped automatically.
WINDOWS = {
    "shallow_0_6": (0, 6),
    "middle_7_19": (7, 19),
    "critical_20_22": (20, 22),
    "commit_23_26": (23, 26),
    "late_23_last": (23, 10_000),
}


# =========================
# Utility
# =========================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_name() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_json(path: Path) -> Dict[str, Any]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    low = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def get_candidate_token_ids(tokenizer, text: str) -> List[int]:
    variants = [
        text,
        " " + text,
        "\n" + text,
        text + "\n",
        " " + text + "\n",
    ]
    ids = []
    for v in variants:
        enc = tokenizer.encode(v, add_special_tokens=False)
        if enc:
            ids.append(int(enc[0]))
    return sorted(set(ids))


def load_c_e_ids(tokenizer) -> Tuple[List[int], List[int]]:
    diag = load_json(BASELINE_DIAG_JSON)
    c_ids = diag.get("c_token_ids")
    e_ids = diag.get("e_token_ids")
    if isinstance(c_ids, list) and isinstance(e_ids, list) and len(c_ids) and len(e_ids):
        return [int(x) for x in c_ids], [int(x) for x in e_ids]
    return get_candidate_token_ids(tokenizer, "C"), get_candidate_token_ids(tokenizer, "E")


def safe_cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.cosine_similarity(a.float(), b.float(), dim=-1, eps=eps)


def window_slice(n_layers: int, start: int, end: int) -> List[int]:
    # hidden states include embedding state at index 0, transformer layers start at 1.
    # Here we refer to transformer layer index 0..n_layers-1.
    end = min(end, n_layers - 1)
    start = max(0, min(start, n_layers - 1))
    if end < start:
        return []
    return list(range(start, end + 1))


def np_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.nanmean(x))


def np_std(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.nanstd(x))


def np_min(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.nanmin(x))


def np_max(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.nanmax(x))


def np_slope(x: np.ndarray) -> float:
    if x.size <= 1:
        return 0.0 if x.size == 1 else float("nan")
    t = np.arange(len(x), dtype=np.float32)
    try:
        return float(np.polyfit(t, x.astype(np.float32), 1)[0])
    except Exception:
        return float("nan")


def entropy_binary_from_margin(margin: np.ndarray) -> np.ndarray:
    p = 1.0 / (1.0 + np.exp(-np.clip(margin, -30, 30)))
    q = 1.0 - p
    return -(p * np.log(p + 1e-12) + q * np.log(q + 1e-12))


def add_window_features(prefix: str, arr: np.ndarray, out: Dict[str, Any]) -> None:
    out[f"{prefix}_mean"] = np_mean(arr)
    out[f"{prefix}_std"] = np_std(arr)
    out[f"{prefix}_min"] = np_min(arr)
    out[f"{prefix}_max"] = np_max(arr)
    out[f"{prefix}_slope"] = np_slope(arr)
    out[f"{prefix}_abs_mean"] = np_mean(np.abs(arr))


# =========================
# Model scoring
# =========================

@torch.no_grad()
def extract_batch(
    model,
    tokenizer,
    prompts: List[str],
    c_ids: List[int],
    e_ids: List[int],
    c_dir: torch.Tensor,
    e_dir: torch.Tensor,
    ce_dir: torch.Tensor,
    device: str,
) -> Tuple[List[Dict[str, Any]], List[pd.DataFrame]]:
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    attn = enc["attention_mask"]
    last_idx = attn.sum(dim=1) - 1

    outputs = model(**enc, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states  # embedding + layer outputs
    n_total_states = len(hidden_states)
    n_layers = n_total_states - 1
    batch_n = len(prompts)

    # Use transformer layer outputs only: state index 1..n_layers
    h_layers = []
    for li in range(n_layers):
        hs = hidden_states[li + 1]
        h_last = hs[torch.arange(batch_n, device=device), last_idx, :]
        h_layers.append(h_last)
    H = torch.stack(h_layers, dim=1)  # [B, L, D]

    # lm_head scores for C/E token candidates per layer without full-vocab logits.
    lm_w = model.get_output_embeddings().weight.detach()
    c_w = lm_w[torch.tensor(c_ids, device=device)]
    e_w = lm_w[torch.tensor(e_ids, device=device)]

    c_logits_all = torch.einsum("bld,kd->blk", H.float(), c_w.float()).amax(dim=-1)
    e_logits_all = torch.einsum("bld,kd->blk", H.float(), e_w.float()).amax(dim=-1)
    margin = c_logits_all - e_logits_all
    p_c = torch.sigmoid(margin)

    # Center alignments
    center_align_ce = safe_cos(H, ce_dir.view(1, 1, -1))
    center_align_c = safe_cos(H, c_dir.view(1, 1, -1))
    center_align_e = safe_cos(H, e_dir.view(1, 1, -1))

    # Layer transport
    if n_layers >= 2:
        dH = H[:, 1:, :] - H[:, :-1, :]
        delta_norm = dH.float().norm(dim=-1)
        transport_align_ce = safe_cos(dH, ce_dir.view(1, 1, -1))
        transport_align_c = safe_cos(dH, c_dir.view(1, 1, -1))
        transport_align_e = safe_cos(dH, e_dir.view(1, 1, -1))
        if n_layers >= 3:
            cos_dd = safe_cos(dH[:, 1:, :], dH[:, :-1, :]).clamp(-1, 1)
            rotation_deg = torch.rad2deg(torch.acos(cos_dd))
            # pad to length L-1 by adding nan at first transition
            nan_col = torch.full((batch_n, 1), float("nan"), device=device)
            rotation_padded = torch.cat([nan_col, rotation_deg], dim=1)
        else:
            rotation_padded = torch.full_like(delta_norm, float("nan"))
    else:
        dH = None
        delta_norm = torch.full((batch_n, 0), float("nan"), device=device)
        transport_align_ce = torch.full((batch_n, 0), float("nan"), device=device)
        transport_align_c = torch.full((batch_n, 0), float("nan"), device=device)
        transport_align_e = torch.full((batch_n, 0), float("nan"), device=device)
        rotation_padded = torch.full((batch_n, 0), float("nan"), device=device)

    sample_rows: List[Dict[str, Any]] = []
    layer_frames: List[pd.DataFrame] = []

    for bi in range(batch_n):
        m = margin[bi].detach().float().cpu().numpy()
        pc = p_c[bi].detach().float().cpu().numpy()
        cl = c_logits_all[bi].detach().float().cpu().numpy()
        el = e_logits_all[bi].detach().float().cpu().numpy()
        ca = center_align_ce[bi].detach().float().cpu().numpy()
        cac = center_align_c[bi].detach().float().cpu().numpy()
        cae = center_align_e[bi].detach().float().cpu().numpy()

        dn = delta_norm[bi].detach().float().cpu().numpy()
        ta = transport_align_ce[bi].detach().float().cpu().numpy()
        tac = transport_align_c[bi].detach().float().cpu().numpy()
        tae = transport_align_e[bi].detach().float().cpu().numpy()
        rot = rotation_padded[bi].detach().float().cpu().numpy()

        ent = entropy_binary_from_margin(m)

        row: Dict[str, Any] = {}
        row["dsta_signed_attractor_n_layers"] = n_layers
        row["dsta_signed_attractor_final_margin_C_minus_E"] = float(m[-1])
        row["dsta_signed_attractor_final_prob_C"] = float(pc[-1])
        row["dsta_signed_attractor_margin_slope_all"] = np_slope(m)
        row["dsta_signed_attractor_margin_abs_mean_all"] = np_mean(np.abs(m))
        row["dsta_signed_attractor_entropy_mean_all"] = np_mean(ent)
        row["dsta_signed_attractor_center_signed_align_mean"] = np_mean(ca)
        row["dsta_signed_attractor_center_signed_align_std"] = np_std(ca)
        row["dsta_signed_attractor_center_align_to_C_mean"] = np_mean(cac)
        row["dsta_signed_attractor_center_align_to_E_mean"] = np_mean(cae)

        row["dsta_signed_attractor_transport_delta_norm_mean"] = np_mean(dn)
        row["dsta_signed_attractor_transport_delta_norm_max"] = np_max(dn)
        row["dsta_signed_attractor_transport_signed_align_mean"] = np_mean(ta)
        row["dsta_signed_attractor_transport_align_to_C_mean"] = np_mean(tac)
        row["dsta_signed_attractor_transport_align_to_E_mean"] = np_mean(tae)
        row["dsta_signed_attractor_transport_rotation_abs_deg_mean"] = np_mean(np.abs(rot))
        row["dsta_signed_attractor_transport_rotation_abs_deg_max"] = np_max(np.abs(rot))

        # Window features
        for wname, (s, e) in WINDOWS.items():
            idxs = window_slice(n_layers, s, e)
            if not idxs:
                continue
            arr_m = m[idxs]
            arr_pc = pc[idxs]
            arr_ca = ca[idxs]
            arr_ent = ent[idxs]
            add_window_features(f"dsta_signed_attractor_{wname}__margin", arr_m, row)
            add_window_features(f"dsta_signed_attractor_{wname}__prob_C", arr_pc, row)
            add_window_features(f"dsta_signed_attractor_{wname}__center_signed_align", arr_ca, row)
            add_window_features(f"dsta_signed_attractor_{wname}__entropy", arr_ent, row)

            # Transport windows are between layer t-1 -> t. Use matching transition indices.
            tidxs = [i - 1 for i in idxs if i - 1 >= 0 and i - 1 < len(dn)]
            if tidxs:
                add_window_features(f"dsta_signed_attractor_{wname}__transport_delta_norm", dn[tidxs], row)
                add_window_features(f"dsta_signed_attractor_{wname}__transport_signed_align", ta[tidxs], row)
                add_window_features(f"dsta_signed_attractor_{wname}__transport_rotation_abs_deg", np.abs(rot[tidxs]), row)

        # Cross-window contrasts
        def margin_mean_window(w: str) -> float:
            idxs = window_slice(n_layers, *WINDOWS[w])
            return np_mean(m[idxs]) if idxs else float("nan")

        mid = margin_mean_window("middle_7_19")
        crit = margin_mean_window("critical_20_22")
        commit = margin_mean_window("commit_23_26")
        row["dsta_signed_attractor_delta_margin_critical_minus_middle"] = crit - mid
        row["dsta_signed_attractor_delta_margin_commit_minus_critical"] = commit - crit
        row["dsta_signed_attractor_delta_margin_commit_minus_middle"] = commit - mid

        # Layer-long table
        lf = pd.DataFrame({
            "layer": np.arange(n_layers, dtype=int),
            "dsta_layer_C_logit": cl,
            "dsta_layer_E_logit": el,
            "dsta_layer_margin_C_minus_E": m,
            "dsta_layer_prob_C": pc,
            "dsta_layer_entropy_CE": ent,
            "dsta_layer_center_signed_align": ca,
            "dsta_layer_center_align_to_C": cac,
            "dsta_layer_center_align_to_E": cae,
        })
        # Transition arrays have length n_layers-1. align them to target layer 1..n_layers-1
        lf["dsta_layer_transport_delta_norm"] = np.nan
        lf["dsta_layer_transport_signed_align"] = np.nan
        lf["dsta_layer_transport_align_to_C"] = np.nan
        lf["dsta_layer_transport_align_to_E"] = np.nan
        lf["dsta_layer_transport_rotation_abs_deg"] = np.nan
        if len(dn):
            lf.loc[1:, "dsta_layer_transport_delta_norm"] = dn
            lf.loc[1:, "dsta_layer_transport_signed_align"] = ta
            lf.loc[1:, "dsta_layer_transport_align_to_C"] = tac
            lf.loc[1:, "dsta_layer_transport_align_to_E"] = tae
            lf.loc[1:, "dsta_layer_transport_rotation_abs_deg"] = np.abs(rot)

        sample_rows.append(row)
        layer_frames.append(lf)

    return sample_rows, layer_frames


# =========================
# Main
# =========================

def main() -> None:
    set_seed(RANDOM_SEED)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model path not found: {MODEL_PATH}")

    df = pd.read_csv(INPUT_CSV)
    required = ["graph_id", "condition", "prompt"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing required column {c}. Columns={list(df.columns)}")

    device = device_name()
    print(f"[INFO] device={device}")
    print(f"[INFO] loading tokenizer/model from {MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=DTYPE if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        local_files_only=True,
    )
    if device != "cuda":
        model.to(device)
    model.eval()

    c_ids, e_ids = load_c_e_ids(tokenizer)
    print(f"[INFO] C token ids={c_ids}")
    print(f"[INFO] E token ids={e_ids}")

    emb = model.get_input_embeddings().weight.detach()
    c_vec = emb[torch.tensor(c_ids, device=emb.device)].float().mean(dim=0)
    e_vec = emb[torch.tensor(e_ids, device=emb.device)].float().mean(dim=0)
    ce_vec = c_vec - e_vec
    c_vec = F.normalize(c_vec, dim=0)
    e_vec = F.normalize(e_vec, dim=0)
    ce_vec = F.normalize(ce_vec, dim=0)

    c_vec = c_vec.to(model.device if hasattr(model, "device") else device)
    e_vec = e_vec.to(c_vec.device)
    ce_vec = ce_vec.to(c_vec.device)

    prompts = df["prompt"].astype(str).tolist()
    sample_feature_rows: List[Dict[str, Any]] = []
    layer_metric_frames: List[pd.DataFrame] = []

    for start in range(0, len(df), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(df))
        print(f"[BATCH] {start}:{end} / {len(df)}")
        batch_prompts = prompts[start:end]
        rows, lframes = extract_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=batch_prompts,
            c_ids=c_ids,
            e_ids=e_ids,
            c_dir=c_vec,
            e_dir=e_vec,
            ce_dir=ce_vec,
            device=device,
        )

        meta_cols = [
            "graph_id", "condition_family", "condition", "variant_id",
            "gold_label", "expected_policy_family", "mechanism_template",
            "baseline_C_logit", "baseline_E_logit", "baseline_margin_C_minus_E",
            "baseline_prob_C_over_CE", "baseline_prob_E_over_CE",
            "baseline_pred_label", "baseline_correct", "baseline_prob_gold",
            "baseline_gold_margin",
        ]
        meta_cols = [c for c in meta_cols if c in df.columns]

        for i, r in enumerate(rows):
            meta = df.iloc[start + i][meta_cols].to_dict()
            sample_feature_rows.append({**meta, **r})

        for i, lf in enumerate(lframes):
            meta = df.iloc[start + i][[c for c in meta_cols if c not in {
                "baseline_C_logit", "baseline_E_logit", "baseline_margin_C_minus_E",
                "baseline_prob_C_over_CE", "baseline_prob_E_over_CE",
                "baseline_pred_label", "baseline_correct", "baseline_prob_gold",
                "baseline_gold_margin",
            }]].to_dict()
            for k, v in meta.items():
                lf[k] = v
            layer_metric_frames.append(lf)

    feat = pd.DataFrame(sample_feature_rows)
    layer_long = pd.concat(layer_metric_frames, ignore_index=True) if layer_metric_frames else pd.DataFrame()

    # Family / condition summaries for quick audit
    numeric_feature_cols = [
        c for c in feat.select_dtypes(include=[np.number]).columns
        if c not in ["variant_id"]
    ]

    cond_summary = (
        feat.groupby(["condition_family", "condition"], as_index=False)
        .agg(
            n=("graph_id", "count"),
            mean_final_margin=("dsta_signed_attractor_final_margin_C_minus_E", "mean"),
            mean_final_prob_C=("dsta_signed_attractor_final_prob_C", "mean"),
            mean_transport_delta_norm=("dsta_signed_attractor_transport_delta_norm_mean", "mean"),
            mean_transport_align=("dsta_signed_attractor_transport_signed_align_mean", "mean"),
            mean_critical_margin=("dsta_signed_attractor_critical_20_22__margin_mean", "mean"),
            mean_commit_margin=("dsta_signed_attractor_commit_23_26__margin_mean", "mean"),
        )
    )

    fam_summary = (
        feat.groupby(["condition_family"], as_index=False)
        .agg(
            n=("graph_id", "count"),
            n_conditions=("condition", "nunique"),
            mean_final_margin=("dsta_signed_attractor_final_margin_C_minus_E", "mean"),
            mean_final_prob_C=("dsta_signed_attractor_final_prob_C", "mean"),
            mean_transport_delta_norm=("dsta_signed_attractor_transport_delta_norm_mean", "mean"),
            mean_transport_align=("dsta_signed_attractor_transport_signed_align_mean", "mean"),
        )
    )

    out_feat = OUT_DIR / "da_asa2k3_dsta_sample_features.csv"
    out_layer = OUT_DIR / "da_asa2k3_signed_layer_metrics_long.csv"
    out_cond = OUT_DIR / "da_asa2k3_condition_summary.csv"
    out_fam = OUT_DIR / "da_asa2k3_family_summary.csv"
    out_diag = OUT_DIR / "da_asa2k3_diagnostics.json"
    out_seed = OUT_DIR / "da_asa2k3_next_stage_seed.md"

    feat.to_csv(out_feat, index=False, encoding="utf-8-sig")
    layer_long.to_csv(out_layer, index=False, encoding="utf-8-sig")
    cond_summary.to_csv(out_cond, index=False, encoding="utf-8-sig")
    fam_summary.to_csv(out_fam, index=False, encoding="utf-8-sig")

    diagnostics = {
        "stage": "DA-ASA-2K.3",
        "purpose": "Expanded DenseDSTA signed/transport/attractor feature extraction",
        "verdict": "DSTA_FEATURE_EXTRACTION_COMPLETE",
        "root": str(ROOT),
        "input_csv": str(INPUT_CSV),
        "output_dir": str(OUT_DIR),
        "model_path": str(MODEL_PATH),
        "n_rows": int(len(feat)),
        "n_conditions": int(feat["condition"].nunique()),
        "n_condition_families": int(feat["condition_family"].nunique()) if "condition_family" in feat.columns else None,
        "n_numeric_feature_cols": int(len(numeric_feature_cols)),
        "c_token_ids": c_ids,
        "e_token_ids": e_ids,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "outputs": {
            "dsta_sample_features": str(out_feat),
            "signed_layer_metrics_long": str(out_layer),
            "condition_summary": str(out_cond),
            "family_summary": str(out_fam),
            "next_stage_seed": str(out_seed),
        },
        "next_stage": "DA-ASA-2K.4 policy_results table generation or LPF assignment pipeline",
    }
    with open(out_diag, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    seed = f"""# DA-ASA-2K.4 Next Stage Seed

## Current status

DA-ASA-2K.3 completed expanded DenseDSTA feature extraction.

Verdict:

```text
DSTA_FEATURE_EXTRACTION_COMPLETE
```

## Paths

Root:

```text
{ROOT}
```

Baseline score table:

```text
{INPUT_CSV}
```

DenseDSTA feature table:

```text
{out_feat}
```

Layer-long DSTA metrics:

```text
{out_layer}
```

## Next required stage

Run policy_results table generation across the DA-ASA action set, or if using existing 2J topology,
assign LPF labels and run DenseDSTA -> LPF CV.

Recommended sequence:

```text
1. policy_results table on expanded samples
2. effect vectors
3. LPF clustering or assignment using 2J topology
4. DenseDSTA -> TransferableLPF CV
```

Primary keys:

```text
graph_id + condition
```
"""
    out_seed.write_text(seed, encoding="utf-8")

    print("\n" + "=" * 80)
    print("DA-ASA-2K.3 DenseDSTA feature extraction complete")
    print("=" * 80)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
