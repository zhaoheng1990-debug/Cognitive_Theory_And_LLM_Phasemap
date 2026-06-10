# -*- coding: utf-8 -*-
r"""
SEM-6H.2 Live Model-Forward MemoryUnit Write Replay

Purpose
-------
Read:
    C:\Users\ZH\Desktop\AGI\python_script\sem6h2_outputs\sem6h2_replay_plan.csv

Run real Qwen forward passes with a lightweight latent MemoryUnit write intervention:
    target prompt hidden at write layers L15-L19
    -> interpolated toward candidate memory hidden vectors

Then output:
    sem6h2_live_results.csv

The output can be fed back to:
    GPT_sem6h2_noleak_live_write_replay.py --live_results sem6h2_live_results.csv

Important
---------
This script is a live-forward replay bridge. Because sem6h2_replay_plan.csv contains
concept/operator metadata rather than stored full MemoryUnit tensors, the script builds
latent candidate memory vectors by forwarding a synthetic MemoryUnit prompt:

    "MemoryUnit prior: concept=X, operator=Y ..."

Then it injects those vectors into the target task prompt.

This is therefore a real model-forward latent write proxy, not an offline oracle-utility
reuse. It is designed to test whether 6H.1b's selected candidates cause live trajectory
changes under a consistent intervention protocol.

Default paths are hardcoded for your Windows setup.

Run
---
conda activate dhrf_4080s

python GPT_sem6h2_live_forward_write.py

Faster smoke test:
python GPT_sem6h2_live_forward_write.py --max_rows 200 --max_null_reps 2

Fuller run:
python GPT_sem6h2_live_forward_write.py --max_rows 0 --max_null_reps 50

Then evaluate:
python GPT_sem6h2_noleak_live_write_replay.py ^
  --candidates sem6h1b_outputs\sem6h1b_scored_candidates.csv ^
  --live_results sem6h2_outputs\sem6h2_live_results.csv ^
  --out_dir sem6h2_outputs
"""

import argparse
import gc
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================
# Hardcoded local configuration
# =============================

DEFAULT_MODEL_PATH = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

DEFAULT_ROOT = r"C:\Users\ZH\Desktop\AGI\python_script"
DEFAULT_SEM6H2_DIR = rf"{DEFAULT_ROOT}\sem6h2_outputs"

DEFAULT_REPLAY_PLAN = rf"{DEFAULT_SEM6H2_DIR}\sem6h2_replay_plan.csv"
DEFAULT_OUT_CSV = rf"{DEFAULT_SEM6H2_DIR}\sem6h2_live_results.csv"
DEFAULT_PROGRESS_JSON = rf"{DEFAULT_SEM6H2_DIR}\sem6h2_live_progress.json"

DEFAULT_WRITE_LAYERS = "15,16,17,18,19"
DEFAULT_ALPHAS = "0.1,0.2,0.3"

SEED = 20260606


# =============================
# Operator prompt templates
# =============================

OPERATOR_TEMPLATES = {
    "Definition": "What is {concept}? Give a concise definition.",
    "MechanismExplanation": "Explain the mechanism behind {concept}.",
    "CausalExplanation": "Explain what causes {concept} and what effects it produces.",
    "RelationMapping": "Map the key relationships involving {concept}.",
    "PropertyDescription": "Describe the key properties of {concept}.",
    "Comparison": "Compare {concept} with closely related concepts.",
    "CounterExample": "Find a counterexample or limitation related to {concept}.",
    "InvariantSearch": "Find the invariant structure behind {concept}.",
    "ClosureCheck": "Check whether the explanation of {concept} is internally closed and consistent.",
    "PolicySelection": "Select the best strategy for reasoning about {concept}.",
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_csv_list(s: str, cast=float):
    out = []
    for x in str(s).split(","):
        x = x.strip()
        if not x:
            continue
        out.append(cast(x))
    return out


def operator_description(op: str) -> str:
    desc = {
        "Definition": "define the concept and state its core meaning",
        "MechanismExplanation": "explain internal mechanism and process",
        "CausalExplanation": "explain causes, effects, and causal chain",
        "RelationMapping": "map relations between concepts, entities, and roles",
        "PropertyDescription": "describe attributes, properties, and characteristic features",
        "Comparison": "compare similarities and differences",
        "CounterExample": "test the claim through counterexamples",
        "InvariantSearch": "search for invariant structure across cases",
        "ClosureCheck": "check relation closure and internal consistency",
        "PolicySelection": "select an appropriate reasoning/control policy",
    }
    return desc.get(str(op), f"use the reasoning operator named {op}")


def build_task_request(concept: str, operator: str) -> str:
    tmpl = OPERATOR_TEMPLATES.get(str(operator))
    if tmpl:
        return tmpl.format(concept=concept)
    # Generic fallback: keep the operator as task cue.
    return f"Reason about {concept} using the operator {operator}."


def build_classifier_prompt(
    concept: str,
    target_operator: str,
    operator_to_letter: Dict[str, str],
) -> str:
    request = build_task_request(concept, target_operator)
    options = "\n".join(
        f"{letter}. {op}: {operator_description(op)}"
        for op, letter in operator_to_letter.items()
    )
    return (
        "You are classifying the reasoning operator required by a request.\n"
        "Choose exactly one option letter.\n\n"
        f"Options:\n{options}\n\n"
        f"Request:\n{request}\n\n"
        "Answer with one option letter only:"
    )


def build_memory_prompt(candidate_concept: str, candidate_operator: str) -> str:
    return (
        "MemoryUnit prior.\n"
        f"Concept identity: {candidate_concept}\n"
        f"Operator prior: {candidate_operator}\n"
        f"Transferable trajectory prior: {operator_description(candidate_operator)}.\n"
        "Use this as a latent memory vector, not as a final answer."
    )


def apply_chat(tokenizer, user_text: str) -> str:
    # Qwen instruct models generally benefit from chat template.
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": user_text}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return user_text + "\nAssistant:"


def first_token_id_for_label(tokenizer, letter: str) -> int:
    # Try a few forms; use the shortest tokenization.
    variants = [letter, " " + letter, "\n" + letter]
    best = None
    for v in variants:
        ids = tokenizer(v, add_special_tokens=False).input_ids
        if ids:
            if best is None or len(ids) < len(best):
                best = ids
    if not best:
        raise ValueError(f"Could not tokenize label {letter!r}")
    return int(best[0])


def get_layers(model):
    # Qwen/Llama/Gemma-like CausalLMs.
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("Could not locate transformer layers on this model.")


@torch.no_grad()
def get_memory_vectors(
    model,
    tokenizer,
    text: str,
    write_layers: List[int],
    device: str,
    max_len: int,
) -> Dict[int, torch.Tensor]:
    prompt = apply_chat(tokenizer, text)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    ).to(device)

    outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    # hidden_states[0] = embedding output, hidden_states[layer+1] = after layer.
    vecs = {}
    for layer in write_layers:
        idx = layer + 1
        if idx >= len(outputs.hidden_states):
            raise ValueError(
                f"Requested layer {layer}, but model returned only {len(outputs.hidden_states)-1} layers."
            )
        vecs[layer] = outputs.hidden_states[idx][0, -1, :].detach().to(device)
    return vecs


def register_write_hooks(model, memory_vecs: Dict[int, torch.Tensor], alpha: float):
    layers = get_layers(model)
    handles = []

    def make_hook(layer_idx: int):
        def hook(module, inputs, output):
            if layer_idx not in memory_vecs:
                return output

            mem = memory_vecs[layer_idx]
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None

            # hidden shape: [batch, seq, dim]
            h = hidden.clone()
            mem2 = mem.to(device=h.device, dtype=h.dtype).view(1, 1, -1)
            h[:, -1:, :] = (1.0 - alpha) * h[:, -1:, :] + alpha * mem2

            if rest is None:
                return h
            return (h,) + rest
        return hook

    for layer_idx in memory_vecs.keys():
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"Layer index {layer_idx} out of range 0..{len(layers)-1}")
        handles.append(layers[layer_idx].register_forward_hook(make_hook(layer_idx)))
    return handles


@torch.no_grad()
def score_prompt_letters(
    model,
    tokenizer,
    prompt_text: str,
    letter_to_token_id: Dict[str, int],
    device: str,
    max_len: int,
    memory_vecs: Optional[Dict[int, torch.Tensor]] = None,
    alpha: float = 0.0,
) -> Dict[str, float]:
    prompt = apply_chat(tokenizer, prompt_text)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    ).to(device)

    handles = []
    try:
        if memory_vecs is not None and alpha > 0:
            handles = register_write_hooks(model, memory_vecs, alpha)
        outputs = model(**inputs, use_cache=False)
        logits = outputs.logits[0, -1, :].float()
        logp = torch.log_softmax(logits, dim=-1)
        return {letter: float(logp[token_id].detach().cpu()) for letter, token_id in letter_to_token_id.items()}
    finally:
        for h in handles:
            h.remove()
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def compute_margin_and_rank(scores: Dict[str, float], correct_letter: str) -> Tuple[float, int, float]:
    correct = scores[correct_letter]
    others = [v for k, v in scores.items() if k != correct_letter]
    best_other = max(others) if others else float("-inf")
    margin = correct - best_other

    sorted_letters = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    rank = sorted_letters.index(correct_letter) + 1
    best_score = scores[sorted_letters[0]]
    return float(margin), int(rank), float(correct - best_score)


def downsample_null_reps(df: pd.DataFrame, max_null_reps: int) -> pd.DataFrame:
    if max_null_reps <= 0:
        return df

    keep_parts = []
    for fam, sub in df.groupby("_selection_family", sort=False):
        if fam not in {"random_pool", "shuffle_learned_Q"}:
            keep_parts.append(sub)
            continue

        # _selection values like random_pool_003; keep first N per family.
        sel_values = sorted(sub["_selection"].dropna().unique().tolist())
        keep_sels = set(sel_values[:max_null_reps])
        keep_parts.append(sub[sub["_selection"].isin(keep_sels)])

    return pd.concat(keep_parts, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--replay_plan", default=DEFAULT_REPLAY_PLAN)
    parser.add_argument("--out_csv", default=DEFAULT_OUT_CSV)
    parser.add_argument("--progress_json", default=DEFAULT_PROGRESS_JSON)

    parser.add_argument("--write_layers", default=DEFAULT_WRITE_LAYERS)
    parser.add_argument("--alphas", default=DEFAULT_ALPHAS)
    parser.add_argument("--max_len", type=int, default=384)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)

    parser.add_argument("--max_rows", type=int, default=0, help="0 means all rows after filtering.")
    parser.add_argument("--max_null_reps", type=int, default=10, help="Use 50 for full null run. 0 keeps all.")
    parser.add_argument("--selection_families", default="", help="Comma-separated filter, empty means all.")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--flush_every", type=int, default=20)
    args = parser.parse_args()

    set_seed(SEED)

    replay_path = Path(args.replay_plan)
    if not replay_path.exists():
        raise FileNotFoundError(f"Replay plan not found: {replay_path}")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = Path(args.progress_json)
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    write_layers = [int(x) for x in parse_csv_list(args.write_layers, int)]
    alphas = [float(x) for x in parse_csv_list(args.alphas, float)]

    plan = pd.read_csv(replay_path)
    required = [
        "_selection", "_selection_family", "_group_id",
        "target_concept", "target_operator",
        "candidate_concept", "candidate_operator",
        "candidate_family", "pool", "candidate_i",
    ]
    missing = [c for c in required if c not in plan.columns]
    if missing:
        raise ValueError(f"Replay plan missing required columns: {missing}\nColumns={list(plan.columns)}")

    plan = downsample_null_reps(plan, args.max_null_reps)

    if args.selection_families.strip():
        keep = {x.strip() for x in args.selection_families.split(",") if x.strip()}
        plan = plan[plan["_selection_family"].isin(keep)].copy()

    if args.max_rows and args.max_rows > 0:
        plan = plan.head(args.max_rows).copy()

    # Stable row ids for resume.
    if "_live_row_id" not in plan.columns:
        plan["_live_row_id"] = np.arange(len(plan), dtype=int)

    done_ids = set()
    existing_rows = []
    if args.resume and out_path.exists():
        try:
            old = pd.read_csv(out_path)
            if "_live_row_id" in old.columns:
                done_ids = set(pd.to_numeric(old["_live_row_id"], errors="coerce").dropna().astype(int).tolist())
                existing_rows = old.to_dict("records")
                print(f"[resume] loaded {len(done_ids)} completed rows from {out_path}")
        except Exception as e:
            print(f"[resume] could not read existing output: {e}")

    # Operator label mapping.
    operator_list = sorted(set(plan["target_operator"].astype(str).tolist()) | set(plan["candidate_operator"].astype(str).tolist()))
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if len(operator_list) > len(letters):
        raise ValueError(f"Too many operators for letter labels: {len(operator_list)}")
    operator_to_letter = {op: letters[i] for i, op in enumerate(operator_list)}
    letter_to_operator = {v: k for k, v in operator_to_letter.items()}

    print("Operators:")
    for op, letter in operator_to_letter.items():
        print(f"  {letter}: {op}")

    # Load model.
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; using CPU.")
        device = "cpu"

    dtype = torch.float16 if args.dtype == "float16" and device == "cuda" else torch.float32

    print(f"Loading tokenizer/model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if device != "cuda":
        model.to(device)
    model.eval()

    letter_to_token_id = {
        letter: first_token_id_for_label(tokenizer, letter)
        for letter in operator_to_letter.values()
    }

    # Caches.
    memory_cache: Dict[Tuple[str, str], Dict[int, torch.Tensor]] = {}
    baseline_cache: Dict[Tuple[str, str], Tuple[Dict[str, float], float, int, float]] = {}

    results = list(existing_rows)

    rows = plan.to_dict("records")
    total = len(rows)
    print(f"Running SEM-6H.2 live forward rows: {total}; already done: {len(done_ids)}")
    print(f"Write layers: {write_layers}; alphas: {alphas}")

    for idx, row in enumerate(rows):
        live_id = int(row["_live_row_id"])
        if live_id in done_ids:
            continue

        target_concept = str(row["target_concept"])
        target_operator = str(row["target_operator"])
        candidate_concept = str(row["candidate_concept"])
        candidate_operator = str(row["candidate_operator"])

        correct_letter = operator_to_letter[target_operator]
        target_key = (target_concept, target_operator)
        memory_key = (candidate_concept, candidate_operator)

        # Baseline prompt score.
        if target_key not in baseline_cache:
            target_prompt = build_classifier_prompt(target_concept, target_operator, operator_to_letter)
            base_scores = score_prompt_letters(
                model, tokenizer, target_prompt, letter_to_token_id,
                device=device, max_len=args.max_len,
            )
            base_margin, base_rank, base_logprob_gap_to_top = compute_margin_and_rank(base_scores, correct_letter)
            baseline_cache[target_key] = (base_scores, base_margin, base_rank, base_logprob_gap_to_top)
        else:
            base_scores, base_margin, base_rank, base_logprob_gap_to_top = baseline_cache[target_key]

        # Candidate memory vector.
        if memory_key not in memory_cache:
            mem_prompt = build_memory_prompt(candidate_concept, candidate_operator)
            memory_cache[memory_key] = get_memory_vectors(
                model, tokenizer, mem_prompt, write_layers,
                device=device, max_len=args.max_len,
            )
        mem_vecs = memory_cache[memory_key]

        target_prompt = build_classifier_prompt(target_concept, target_operator, operator_to_letter)

        out = dict(row)
        out["correct_letter"] = correct_letter
        out["operator_options_json"] = json.dumps(operator_to_letter, ensure_ascii=False)
        out["base_margin"] = base_margin
        out["base_rank"] = base_rank
        out["base_logprob_gap_to_top"] = base_logprob_gap_to_top

        for alpha in alphas:
            live_scores = score_prompt_letters(
                model, tokenizer, target_prompt, letter_to_token_id,
                device=device, max_len=args.max_len,
                memory_vecs=mem_vecs,
                alpha=alpha,
            )
            live_margin, live_rank, live_logprob_gap_to_top = compute_margin_and_rank(live_scores, correct_letter)

            tag = str(alpha).replace(".", "_")
            out[f"live_margin_alpha_{tag}"] = live_margin
            out[f"live_margin_gain_alpha_{tag}"] = live_margin - base_margin
            out[f"live_rank_alpha_{tag}"] = live_rank
            out[f"live_rank_improvement_alpha_{tag}"] = base_rank - live_rank
            out[f"live_logprob_gap_to_top_alpha_{tag}"] = live_logprob_gap_to_top
            out[f"live_logprob_gap_gain_alpha_{tag}"] = live_logprob_gap_to_top - base_logprob_gap_to_top

            # Generic aliases picked up by evaluator.
            out[f"live_gain_alpha_{alpha}"] = live_margin - base_margin
            out[f"live_rank_improvement_alpha_{alpha}"] = base_rank - live_rank

        # Main convenience metric: alpha 0.3 if present else last alpha.
        main_alpha = 0.3 if 0.3 in alphas else alphas[-1]
        main_tag = str(main_alpha).replace(".", "_")
        out["live_margin_gain"] = out[f"live_margin_gain_alpha_{main_tag}"]
        out["live_rank_improvement"] = out[f"live_rank_improvement_alpha_{main_tag}"]
        out["trajectory_improvement"] = out["live_margin_gain"]

        results.append(out)

        if (len(results) % args.flush_every == 0) or (idx == total - 1):
            pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")
            progress = {
                "completed_rows": len(results),
                "total_planned_rows_after_filter": total,
                "output": str(out_path),
                "last_live_row_id": live_id,
                "write_layers": write_layers,
                "alphas": alphas,
                "note": "This is live model-forward latent MemoryUnit write replay.",
            }
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
            print(f"[progress] {len(results)}/{total} rows written -> {out_path}")

        # Light cleanup.
        if idx % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[DONE] SEM-6H.2 live results written to:\n{out_path}")
    print("\nNext evaluation command:")
    print(
        "python GPT_sem6h2_noleak_live_write_replay.py ^\n"
        "  --candidates sem6h1b_outputs\\sem6h1b_scored_candidates.csv ^\n"
        f"  --live_results {out_path} ^\n"
        "  --out_dir sem6h2_outputs"
    )


if __name__ == "__main__":
    main()
