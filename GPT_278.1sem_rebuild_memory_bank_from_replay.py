# -*- coding: utf-8 -*-
r"""
SEM-6H.2b MemoryUnit Bank Rebuilder from Replay Plan

Purpose
-------
Your inventory shows no saved full hidden-dim MemoryUnit bank. This script rebuilds
a hidden-dim MemoryUnit bank by re-forwarding canonical/paraphrased prompts for each
family found in:

    sem6h2_outputs\sem6h2_replay_plan.csv

It exports:

    sem5c_outputs\sem5c_memory_bank.npz

with:
    family_ids
    mu_shape   [n_family, hidden_dim]   mean over L7-L19 and paraphrases
    mu_commit  [n_family, hidden_dim]   mean over L23-L25 and paraphrases

Then SEM-6H.2b can run true hidden-dim tensor injection.

Important
---------
This is not the original SEM-5C frozen bank if you did not save it. It is a
reconstructed hidden-dim MemoryUnit bank from the same family metadata:
    family = target/candidate concept + operator

Compared with the failed 6H.2 synthetic proxy, this is stronger because:
  1. It builds one reusable family-level center per family;
  2. It uses multiple paraphrases per family, not a single memory prompt;
  3. It separately averages shape layers L7-L19 and commit layers L23-L25;
  4. It exports full hidden_size vectors, not PCA/utility features.

Run
---
conda activate dhrf_4080s
cd C:\Users\ZH\Desktop\AGI\python_script

python GPT_sem_rebuild_memory_bank_from_replay.py

Smoke:
python GPT_sem_rebuild_memory_bank_from_replay.py --max_families 10

Then inspect:
python GPT_sem6h2b_true_memoryunit_live_write.py --inspect_only

Then run:
python GPT_sem6h2b_true_memoryunit_live_write.py --max_rows 120 --max_null_reps 2
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_ROOT = r"C:\Users\ZH\Desktop\AGI\python_script"
DEFAULT_MODEL_PATH = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"
DEFAULT_REPLAY_PLAN = rf"{DEFAULT_ROOT}\sem6h2_outputs\sem6h2_replay_plan.csv"
DEFAULT_OUT_NPZ = rf"{DEFAULT_ROOT}\sem5c_outputs\sem5c_memory_bank.npz"
DEFAULT_REPORT = rf"{DEFAULT_ROOT}\sem5c_outputs\sem5c_memory_bank_rebuild_report.json"

SHAPE_LAYERS = list(range(7, 20))
COMMIT_LAYERS = [23, 24, 25]
SEED = 20260606


OPERATOR_TEMPLATES = {
    "Definition": [
        "What is {concept}? Give a concise definition.",
        "Define {concept} in one clear paragraph.",
        "Explain the core meaning of {concept}.",
        "State what {concept} refers to and why it matters.",
    ],
    "MechanismExplanation": [
        "Explain the mechanism behind {concept}.",
        "Describe how {concept} works internally.",
        "What process or mechanism produces {concept}?",
        "Break down the mechanism of {concept} step by step.",
    ],
    "CausalExplanation": [
        "Explain what causes {concept} and what effects it produces.",
        "Describe the causal chain involving {concept}.",
        "What are the causes and consequences of {concept}?",
        "Explain {concept} from a cause-and-effect perspective.",
    ],
    "RelationMapping": [
        "Map the key relationships involving {concept}.",
        "Describe how {concept} relates to nearby concepts.",
        "List the important relations around {concept}.",
        "Explain the relation structure connected to {concept}.",
    ],
    "PropertyDescription": [
        "Describe the key properties of {concept}.",
        "What are the main attributes of {concept}?",
        "Summarize the characteristic features of {concept}.",
        "Describe {concept} by its properties and traits.",
    ],
    "Comparison": [
        "Compare {concept} with closely related concepts.",
        "How is {concept} similar to and different from related concepts?",
        "Compare and contrast {concept} with nearby alternatives.",
        "Explain {concept} by comparison.",
    ],
    "CounterExample": [
        "Find a counterexample or limitation related to {concept}.",
        "Test {concept} by looking for a counterexample.",
        "What case would challenge a claim about {concept}?",
        "Identify a limitation or counterexample for {concept}.",
    ],
    "InvariantSearch": [
        "Find the invariant structure behind {concept}.",
        "What remains stable across variations of {concept}?",
        "Search for the invariant pattern in {concept}.",
        "Identify the structure preserved by {concept}.",
    ],
    "ClosureCheck": [
        "Check whether the explanation of {concept} is internally closed and consistent.",
        "Audit the closure and consistency of {concept}.",
        "Verify whether the relation chain for {concept} closes correctly.",
        "Check for closure breaks in reasoning about {concept}.",
    ],
    "PolicySelection": [
        "Select the best strategy for reasoning about {concept}.",
        "Choose an appropriate reasoning policy for {concept}.",
        "Decide which operator should be applied to {concept}.",
        "Select the control strategy for analyzing {concept}.",
    ],
}


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize(s) -> str:
    return str(s).strip()


def apply_chat(tokenizer, user_text: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return user_text + "\nAssistant:"


def templates_for_operator(operator: str, concept: str, n_paraphrases: int) -> List[str]:
    base = OPERATOR_TEMPLATES.get(operator)
    if base is None:
        base = [
            "Reason about {concept} using the operator {operator}.",
            "Apply {operator} to analyze {concept}.",
            "Use {operator} as the reasoning policy for {concept}.",
            "Analyze {concept} through {operator}.",
        ]
        texts = [x.format(concept=concept, operator=operator) for x in base]
    else:
        texts = [x.format(concept=concept) for x in base]

    # Add a structured carrier that tends to stabilize family identity.
    texts.append(
        "SeedFamily latent carrier.\n"
        f"Concept: {concept}\n"
        f"Operator: {operator}\n"
        "Task: form the internal trajectory geometry for this concept-operator family."
    )

    return texts[:max(1, n_paraphrases)]


def collect_families(plan: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # Candidate families are the write targets for 6H.2b, so they must be included.
    for _, r in plan.iterrows():
        if "candidate_family" in plan.columns:
            rows.append({
                "family": sanitize(r["candidate_family"]),
                "concept": sanitize(r.get("candidate_concept", "")),
                "operator": sanitize(r.get("candidate_operator", "")),
                "source": "candidate",
            })
        if "_group_id" in plan.columns:
            rows.append({
                "family": sanitize(r["_group_id"]),
                "concept": sanitize(r.get("target_concept", "")),
                "operator": sanitize(r.get("target_operator", "")),
                "source": "target",
            })

    df = pd.DataFrame(rows).dropna()
    df = df[(df["family"] != "") & (df["concept"] != "") & (df["operator"] != "")]
    # Deduplicate by family; prefer candidate metadata, then target.
    df["_source_rank"] = df["source"].map({"candidate": 0, "target": 1}).fillna(9)
    df = df.sort_values(["family", "_source_rank"]).drop_duplicates("family", keep="first")
    return df[["family", "concept", "operator", "source"]].reset_index(drop=True)


@torch.no_grad()
def hidden_for_prompt(model, tokenizer, text: str, device: str, max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    prompt = apply_chat(tokenizer, text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    hs = outputs.hidden_states
    shape_vecs = []
    for l in SHAPE_LAYERS:
        idx = l + 1
        if idx >= len(hs):
            raise ValueError(f"Model has only {len(hs)-1} layers; requested L{l}.")
        shape_vecs.append(hs[idx][0, -1, :].detach().float().cpu().numpy())

    commit_vecs = []
    for l in COMMIT_LAYERS:
        idx = l + 1
        if idx >= len(hs):
            raise ValueError(f"Model has only {len(hs)-1} layers; requested L{l}.")
        commit_vecs.append(hs[idx][0, -1, :].detach().float().cpu().numpy())

    del inputs, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.mean(np.stack(shape_vecs), axis=0), np.mean(np.stack(commit_vecs), axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--replay_plan", default=DEFAULT_REPLAY_PLAN)
    ap.add_argument("--out_npz", default=DEFAULT_OUT_NPZ)
    ap.add_argument("--report", default=DEFAULT_REPORT)
    ap.add_argument("--n_paraphrases", type=int, default=5)
    ap.add_argument("--max_families", type=int, default=0, help="0 means all families.")
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--local_files_only", action="store_true", default=True)
    ap.add_argument("--trust_remote_code", action="store_true", default=True)
    args = ap.parse_args()

    set_seed(SEED)

    replay_path = Path(args.replay_plan)
    if not replay_path.exists():
        raise FileNotFoundError(f"Replay plan not found: {replay_path}")

    plan = pd.read_csv(replay_path)
    families_df = collect_families(plan)
    if families_df.empty:
        raise ValueError("Could not extract family/concept/operator triples from replay plan.")

    if args.max_families and args.max_families > 0:
        families_df = families_df.head(args.max_families).copy()

    print(f"Families to rebuild: {len(families_df)}")
    print(families_df.head(10).to_string(index=False))

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; using CPU.")
        device = "cpu"

    torch_dtype = torch.float16 if args.dtype == "float16" and device == "cuda" else torch.float32

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
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if device != "cuda":
        model.to(device)
    model.eval()

    hidden_size = int(model.config.hidden_size)
    print(f"hidden_size={hidden_size}")

    family_ids = []
    mu_shape = []
    mu_commit = []
    report_rows = []

    for i, row in families_df.iterrows():
        fam = row["family"]
        concept = row["concept"]
        operator = row["operator"]

        prompts = templates_for_operator(operator, concept, args.n_paraphrases)
        shape_list = []
        commit_list = []

        for p in prompts:
            s, c = hidden_for_prompt(model, tokenizer, p, device=device, max_len=args.max_len)
            shape_list.append(s)
            commit_list.append(c)

        shape = np.mean(np.stack(shape_list), axis=0).astype(np.float32)
        commit = np.mean(np.stack(commit_list), axis=0).astype(np.float32)

        if shape.shape[-1] != hidden_size or commit.shape[-1] != hidden_size:
            raise ValueError(f"Hidden dim mismatch for {fam}: {shape.shape}, {commit.shape}, expected {hidden_size}")

        family_ids.append(fam)
        mu_shape.append(shape)
        mu_commit.append(commit)

        report_rows.append({
            "family": fam,
            "concept": concept,
            "operator": operator,
            "n_prompts": len(prompts),
            "shape_norm": float(np.linalg.norm(shape)),
            "commit_norm": float(np.linalg.norm(commit)),
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(families_df):
            print(f"[progress] {i+1}/{len(families_df)} families")

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    mu_shape = np.stack(mu_shape).astype(np.float32)
    mu_commit = np.stack(mu_commit).astype(np.float32)

    np.savez_compressed(
        out_npz,
        family_ids=np.array(family_ids, dtype=object),
        mu_shape=mu_shape,
        mu_commit=mu_commit,
        shape_layers=np.array(SHAPE_LAYERS, dtype=np.int32),
        commit_layers=np.array(COMMIT_LAYERS, dtype=np.int32),
        source_replay_plan=str(replay_path),
        n_paraphrases=np.array([args.n_paraphrases], dtype=np.int32),
        note="Reconstructed hidden-dim MemoryUnit bank from replay plan family metadata.",
    )

    report = {
        "out_npz": str(out_npz),
        "n_family": len(family_ids),
        "hidden_size": hidden_size,
        "mu_shape_shape": list(mu_shape.shape),
        "mu_commit_shape": list(mu_commit.shape),
        "shape_layers": SHAPE_LAYERS,
        "commit_layers": COMMIT_LAYERS,
        "source_replay_plan": str(replay_path),
        "n_paraphrases": args.n_paraphrases,
        "families_preview": report_rows[:20],
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n[EXPORT DONE]")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nNext:")
    print("python GPT_sem6h2b_true_memoryunit_live_write.py --inspect_only")
    print("python GPT_sem6h2b_true_memoryunit_live_write.py --max_rows 120 --max_null_reps 2")


if __name__ == "__main__":
    main()
