# -*- coding: utf-8 -*-
r"""
DA-ASA-2K.2 Baseline C/E Logit Scoring

Purpose
-------
First real-run step after DA-ASA-2K.1 expanded condition-family sample table.
Reads expanded C/E prompts and scores the model's next-token logits for C/E.

Input
-----
Default:
  C:\Users\ZH\Desktop\AGI\python_script\da_asa2k1_outputs\da_asa2k1_expanded_samples_lite.csv

Output
------
  da_asa2k2_baseline_outputs/da_asa2k2_baseline_scores.csv
  da_asa2k2_baseline_outputs/da_asa2k2_condition_summary.csv
  da_asa2k2_baseline_outputs/da_asa2k2_family_summary.csv
  da_asa2k2_baseline_outputs/da_asa2k2_diagnostics.json
  da_asa2k2_baseline_outputs/da_asa2k2_next_stage_seed.md

Notes
-----
- This script does not run DSTA yet. It only produces baseline C/E scores.
- Paths are intentionally hardcoded for the user's Windows environment.
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# =========================
# Config
# =========================

ROOT = Path(r"C:\Users\ZH\Desktop\AGI\python_script")
INPUT_CSV = ROOT / "da_asa2k1_outputs" / "da_asa2k1_expanded_samples_lite.csv"
OUT_DIR = ROOT / "da_asa2k2_baseline_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

BATCH_SIZE = 4
MAX_LENGTH = 768
RANDOM_SEED = 20260606
DTYPE = "auto"  # use "float16" if needed
DEVICE_MAP = "auto"

# We score several token spellings because tokenizer may encode C/E with leading-space tokens.
C_TOKEN_STRINGS = ["C", " C", "\nC"]
E_TOKEN_STRINGS = ["E", " E", "\nE"]

REQUIRED_COLS = [
    "graph_id",
    "condition_family",
    "condition",
    "variant_id",
    "prompt",
    "C_answer",
    "E_answer",
    "gold_label",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def first_token_ids(tokenizer, strings: List[str]) -> List[int]:
    ids = []
    for s in strings:
        enc = tokenizer(s, add_special_tokens=False)["input_ids"]
        if not enc:
            continue
        ids.append(int(enc[0]))
    return sorted(set(ids))


def logsumexp(values: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(values, dim=-1)


@torch.no_grad()
def score_batch(model, tokenizer, prompts: List[str], c_ids: List[int], e_ids: List[int]) -> Dict[str, np.ndarray]:
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model(**enc)
    logits = out.logits

    # Next token after last non-pad token for each sample.
    attention = enc["attention_mask"]
    last_idx = attention.sum(dim=1) - 1
    batch_idx = torch.arange(logits.shape[0], device=logits.device)
    next_logits = logits[batch_idx, last_idx, :]

    c_logits_all = next_logits[:, c_ids]
    e_logits_all = next_logits[:, e_ids]
    c_logit = logsumexp(c_logits_all)
    e_logit = logsumexp(e_logits_all)
    margin = c_logit - e_logit
    pred = torch.where(margin >= 0, torch.tensor(1, device=margin.device), torch.tensor(0, device=margin.device))

    prob_c = torch.softmax(torch.stack([c_logit, e_logit], dim=1), dim=1)[:, 0]
    prob_e = 1.0 - prob_c

    return {
        "baseline_C_logit": c_logit.detach().float().cpu().numpy(),
        "baseline_E_logit": e_logit.detach().float().cpu().numpy(),
        "baseline_margin_C_minus_E": margin.detach().float().cpu().numpy(),
        "baseline_prob_C_over_CE": prob_c.detach().float().cpu().numpy(),
        "baseline_prob_E_over_CE": prob_e.detach().float().cpu().numpy(),
        "baseline_pred_label": np.where(pred.detach().cpu().numpy() == 1, "C", "E"),
    }


def load_model_and_tokenizer():
    print(f"[LOAD] tokenizer/model from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = DTYPE
    if DTYPE == "float16":
        torch_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=DEVICE_MAP,
    )
    model.eval()
    return model, tokenizer


def summarize(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update({
            "n": int(len(sub)),
            "baseline_accuracy": float((sub["baseline_pred_label"] == sub["gold_label"]).mean()),
            "mean_margin_C_minus_E": float(sub["baseline_margin_C_minus_E"].mean()),
            "mean_abs_margin": float(sub["baseline_margin_C_minus_E"].abs().mean()),
            "mean_prob_gold": float(sub["baseline_prob_gold"].mean()),
            "gold_C_count": int((sub["gold_label"] == "C").sum()),
            "gold_E_count": int((sub["gold_label"] == "E").sum()),
            "pred_C_count": int((sub["baseline_pred_label"] == "C").sum()),
            "pred_E_count": int((sub["baseline_pred_label"] == "E").sum()),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def main() -> None:
    set_seed(RANDOM_SEED)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Input CSV missing required columns: {missing}; columns={list(df.columns)}")

    df = df.copy()
    df["gold_label"] = df["gold_label"].astype(str).str.strip().str.upper()
    df["prompt"] = df["prompt"].astype(str)

    model, tokenizer = load_model_and_tokenizer()

    c_ids = first_token_ids(tokenizer, C_TOKEN_STRINGS)
    e_ids = first_token_ids(tokenizer, E_TOKEN_STRINGS)
    if not c_ids or not e_ids:
        raise ValueError(f"Could not resolve C/E token ids. c_ids={c_ids}, e_ids={e_ids}")
    print(f"[TOKENS] C ids={c_ids}, E ids={e_ids}")

    scored_parts = []
    n = len(df)
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        prompts = df.iloc[start:end]["prompt"].tolist()
        scores = score_batch(model, tokenizer, prompts, c_ids, e_ids)
        part = df.iloc[start:end].copy()
        for k, v in scores.items():
            part[k] = v
        scored_parts.append(part)
        if start == 0 or end == n or (start // BATCH_SIZE) % 10 == 0:
            print(f"[SCORE] {end}/{n}")

    scored = pd.concat(scored_parts, ignore_index=True)
    scored["baseline_correct"] = scored["baseline_pred_label"] == scored["gold_label"]
    scored["baseline_prob_gold"] = np.where(
        scored["gold_label"] == "C",
        scored["baseline_prob_C_over_CE"],
        scored["baseline_prob_E_over_CE"],
    )
    scored["baseline_gold_margin"] = np.where(
        scored["gold_label"] == "C",
        scored["baseline_margin_C_minus_E"],
        -scored["baseline_margin_C_minus_E"],
    )

    cond_summary = summarize(scored, ["condition_family", "condition"])
    fam_summary = summarize(scored, ["condition_family"])

    out_scores = OUT_DIR / "da_asa2k2_baseline_scores.csv"
    out_cond = OUT_DIR / "da_asa2k2_condition_summary.csv"
    out_fam = OUT_DIR / "da_asa2k2_family_summary.csv"
    out_diag = OUT_DIR / "da_asa2k2_diagnostics.json"
    out_seed = OUT_DIR / "da_asa2k2_next_stage_seed.md"

    scored.to_csv(out_scores, index=False, encoding="utf-8-sig")
    cond_summary.to_csv(out_cond, index=False, encoding="utf-8-sig")
    fam_summary.to_csv(out_fam, index=False, encoding="utf-8-sig")

    diagnostics = {
        "stage": "DA-ASA-2K.2",
        "purpose": "Baseline C/E logit scoring for expanded condition-family samples",
        "verdict": "BASELINE_SCORING_COMPLETE",
        "root": str(ROOT),
        "input_csv": str(INPUT_CSV),
        "output_dir": str(OUT_DIR),
        "model_path": MODEL_PATH,
        "n_rows": int(len(scored)),
        "n_conditions": int(scored["condition"].nunique()),
        "n_condition_families": int(scored["condition_family"].nunique()),
        "gold_label_counts": scored["gold_label"].value_counts().to_dict(),
        "baseline_pred_counts": scored["baseline_pred_label"].value_counts().to_dict(),
        "baseline_accuracy": float(scored["baseline_correct"].mean()),
        "mean_gold_margin": float(scored["baseline_gold_margin"].mean()),
        "mean_prob_gold": float(scored["baseline_prob_gold"].mean()),
        "c_token_ids": c_ids,
        "e_token_ids": e_ids,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "outputs": {
            "baseline_scores": str(out_scores),
            "condition_summary": str(out_cond),
            "family_summary": str(out_fam),
            "next_stage_seed": str(out_seed),
        },
        "next_stage": "DA-ASA-2K.3 DSTA signed/transport/attractor feature extraction on expanded samples",
    }
    with open(out_diag, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    seed = f"""# DA-ASA-2K.3 Next Stage Seed\n\n## Current status\n\nDA-ASA-2K.2 completed baseline C/E logit scoring for expanded condition-family samples.\n\nVerdict:\n\n```text\nBASELINE_SCORING_COMPLETE\n```\n\n## Paths\n\nRoot:\n\n```text\n{ROOT}\n```\n\nBaseline score table:\n\n```text\n{out_scores}\n```\n\nCondition summary:\n\n```text\n{out_cond}\n```\n\nFamily summary:\n\n```text\n{out_fam}\n```\n\n## Next required stage\n\nRun DSTA signed / transport / attractor feature extraction on the same expanded samples.\n\nInputs for 2K.3 should include:\n\n```text\n{out_scores}\n```\n\nUse graph_id + condition as sample keys.\n\n## Goal\n\nProduce expanded DenseDSTA features that can later be joined with LPF labels for:\n\n```text\nDenseDSTA -> TransferableLPF CV\n```\n"""
    with open(out_seed, "w", encoding="utf-8") as f:
        f.write(seed)

    print("\n" + "=" * 80)
    print("DA-ASA-2K.2 BASELINE SCORING COMPLETE")
    print("=" * 80)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
