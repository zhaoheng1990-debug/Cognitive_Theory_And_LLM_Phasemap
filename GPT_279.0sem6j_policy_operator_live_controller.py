# -*- coding: utf-8 -*-
r"""
SEM-6J Live Controller: Policy / Operator Write

Purpose
-------
SEM-6J plan has converted Q-selected candidates into write routes:

    write_operator_prior
    trigger_correction_operator
    write_address_gate
    write_residual_mode
    reject_as_null

This live controller executes route-specific ASA-style interventions.

Important distinction from SEM-6H.2/6H.2b
-----------------------------------------
This is NOT direct MemoryUnit vector interpolation:

    H <- (1-alpha)H + alpha*mu_F

Instead it uses route-specific deltas:

    operator prior:
        H(target concept, candidate operator) - H(target concept, target operator)

    correction operator:
        H(target concept, candidate operator) - H(target concept, target operator)

    address gate:
        H(candidate concept, target operator) - H(target concept, target operator)

    residual mode:
        H(candidate concept, candidate operator) - H(target concept, target operator)

    reject:
        no intervention

These are small directional/gated updates, closer to ASA operator/precursor control
than to raw hidden vector replacement.

Default paths
-------------
Root:
    C:\Users\ZH\Desktop\AGI\python_script

Plan:
    sem6j_outputs\sem6j_policy_operator_write_plan.csv

Output:
    sem6j_outputs\sem6j_live_results.csv

Model:
    D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main

Run
---
conda activate dhrf_4080s
cd C:\Users\ZH\Desktop\AGI\python_script

Smoke:
python GPT_sem6j_policy_operator_live_controller.py --max_rows 200 --max_null_reps 2

Evaluate:
python GPT_sem6j_policy_operator_write_audit.py ^
  --live_results sem6j_outputs\sem6j_live_results.csv

Full-ish:
python GPT_sem6j_policy_operator_live_controller.py --max_rows 0 --max_null_reps 10
"""

import argparse
import gc
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_ROOT = r"C:\Users\ZH\Desktop\AGI\python_script"
DEFAULT_MODEL_PATH = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"
DEFAULT_PLAN = rf"{DEFAULT_ROOT}\sem6j_outputs\sem6j_policy_operator_write_plan.csv"
DEFAULT_OUT = rf"{DEFAULT_ROOT}\sem6j_outputs\sem6j_live_results.csv"
DEFAULT_PROGRESS = rf"{DEFAULT_ROOT}\sem6j_outputs\sem6j_live_progress.json"

SEED = 20260606

OPERATOR_LAYERS = list(range(15, 20))
CORRECTION_LAYERS = list(range(15, 20))
COMMIT_LAYERS = [23, 24, 25]
RESIDUAL_LAYERS = list(range(7, 20))


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
    # Names seen in current project that may appear in plan:
    "Planning": "Plan a reasoning strategy for {concept}.",
    "RiskAudit": "Audit the risks, failure modes, and caveats for {concept}.",
}


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def parse_float(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("Could not locate transformer layers.")


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
        "Planning": "construct a plan or staged reasoning procedure",
        "RiskAudit": "audit risk, uncertainty, and possible failure modes",
    }
    return desc.get(str(op), f"use the reasoning operator named {op}")


def build_task_request(concept: str, operator: str) -> str:
    tmpl = OPERATOR_TEMPLATES.get(str(operator))
    if tmpl:
        return tmpl.format(concept=concept)
    return f"Reason about {concept} using the operator {operator}."


def build_operator_prompt(concept: str, operator: str) -> str:
    return (
        "Latent operator carrier.\n"
        f"Concept: {concept}\n"
        f"Operator: {operator}\n"
        f"Instruction: {build_task_request(concept, operator)}\n"
        "Form the internal trajectory for this concept-operator pair."
    )


def build_classifier_prompt(concept: str, target_operator: str, operator_to_letter: Dict[str, str]) -> str:
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


def apply_chat(tokenizer, user_text: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return user_text + "\nAssistant:"


def first_token_id_for_label(tokenizer, letter: str) -> int:
    variants = [letter, " " + letter, "\n" + letter]
    best = None
    for v in variants:
        ids = tokenizer(v, add_special_tokens=False).input_ids
        if ids and (best is None or len(ids) < len(best)):
            best = ids
    if best is None:
        raise ValueError(f"Could not tokenize label {letter!r}")
    return int(best[0])


@torch.no_grad()
def hidden_vectors_by_layer(model, tokenizer, text: str, layers: List[int], device: str, max_len: int) -> Dict[int, torch.Tensor]:
    prompt = apply_chat(tokenizer, text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    out = {}
    for l in layers:
        idx = l + 1
        if idx >= len(outputs.hidden_states):
            raise ValueError(f"Layer L{l} requested, model returned {len(outputs.hidden_states)-1} layers.")
        out[l] = outputs.hidden_states[idx][0, -1, :].detach().to(device)
    del inputs, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def make_delta(
    plus: Dict[int, torch.Tensor],
    minus: Dict[int, torch.Tensor],
    layers: List[int],
    scale: float,
    normalize: bool = True,
) -> Dict[int, torch.Tensor]:
    out = {}
    for l in layers:
        if l not in plus or l not in minus:
            continue
        d = plus[l] - minus[l]
        if normalize:
            # Match the norm to minus vector scale but keep delta small via scale.
            dn = torch.norm(d) + 1e-8
            base_n = torch.norm(minus[l]) + 1e-8
            d = d / dn * base_n
        out[l] = d * float(scale)
    return out


def register_delta_hooks(model, deltas: Dict[int, torch.Tensor]):
    layers = get_layers(model)
    handles = []

    def make_hook(layer_idx: int):
        def hook(module, inputs, output):
            if layer_idx not in deltas:
                return output
            d = deltas[layer_idx]
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None
            h = hidden.clone()
            d2 = d.to(device=h.device, dtype=h.dtype).view(1, 1, -1)
            if d2.shape[-1] != h.shape[-1]:
                raise ValueError(f"Delta dim {d2.shape[-1]} != hidden dim {h.shape[-1]}")
            h[:, -1:, :] = h[:, -1:, :] + d2
            if rest is None:
                return h
            return (h,) + rest
        return hook

    for l in deltas.keys():
        if l < 0 or l >= len(layers):
            raise ValueError(f"Layer index {l} out of range 0..{len(layers)-1}")
        handles.append(layers[l].register_forward_hook(make_hook(l)))
    return handles


@torch.no_grad()
def score_prompt_letters(
    model,
    tokenizer,
    prompt_text: str,
    letter_to_token_id: Dict[str, int],
    device: str,
    max_len: int,
    deltas: Optional[Dict[int, torch.Tensor]] = None,
) -> Dict[str, float]:
    prompt = apply_chat(tokenizer, prompt_text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(device)

    handles = []
    try:
        if deltas:
            handles = register_delta_hooks(model, deltas)
        outputs = model(**inputs, use_cache=False)
        logits = outputs.logits[0, -1, :].float()
        logp = torch.log_softmax(logits, dim=-1)
        return {letter: float(logp[tok].detach().cpu()) for letter, tok in letter_to_token_id.items()}
    finally:
        for h in handles:
            h.remove()
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def compute_margin_rank(scores: Dict[str, float], correct_letter: str) -> Tuple[float, int, float]:
    correct = scores[correct_letter]
    others = [v for k, v in scores.items() if k != correct_letter]
    best_other = max(others) if others else float("-inf")
    margin = correct - best_other
    order = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    rank = order.index(correct_letter) + 1
    gap = correct - scores[order[0]]
    return float(margin), int(rank), float(gap)


def downsample_null_reps(df: pd.DataFrame, max_null_reps: int) -> pd.DataFrame:
    if max_null_reps <= 0:
        return df
    parts = []
    for fam, sub in df.groupby("_selection_family", sort=False):
        if fam not in {"random_pool", "shuffle_learned_Q"}:
            parts.append(sub)
        else:
            sels = sorted(sub["_selection"].dropna().unique().tolist())
            keep = set(sels[:max_null_reps])
            parts.append(sub[sub["_selection"].isin(keep)])
    return pd.concat(parts, ignore_index=True)


def route_layers(route: str) -> List[int]:
    if route == "write_operator_prior":
        return OPERATOR_LAYERS
    if route == "trigger_correction_operator":
        return CORRECTION_LAYERS
    if route == "write_address_gate":
        return COMMIT_LAYERS
    if route == "write_residual_mode":
        return RESIDUAL_LAYERS
    return []


def build_route_delta(
    row: pd.Series,
    cache: Dict[Tuple[str, str, Tuple[int, ...]], Dict[int, torch.Tensor]],
    model,
    tokenizer,
    device: str,
    max_len: int,
    delta_scale: float,
) -> Dict[int, torch.Tensor]:
    route = str(row.get("write_route", ""))
    if route == "reject_as_null":
        return {}

    target_concept = str(row["target_concept"])
    target_operator = str(row["target_operator"])
    candidate_concept = str(row["candidate_concept"])
    candidate_operator = str(row["candidate_operator"])

    alpha_operator = parse_float(row.get("alpha_operator", 0.0), 0.0)
    beta_precursor = parse_float(row.get("beta_precursor", 0.0), 0.0)
    stage_scale = max(alpha_operator, beta_precursor, 0.0) * float(delta_scale)

    if stage_scale <= 0:
        return {}

    layers = route_layers(route)
    layer_key = tuple(layers)

    def get_vecs(concept: str, operator: str, layers_: List[int]) -> Dict[int, torch.Tensor]:
        key = (concept, operator, tuple(layers_))
        if key not in cache:
            cache[key] = hidden_vectors_by_layer(
                model,
                tokenizer,
                build_operator_prompt(concept, operator),
                layers_,
                device=device,
                max_len=max_len,
            )
        return cache[key]

    base = get_vecs(target_concept, target_operator, layers)

    if route in {"write_operator_prior", "trigger_correction_operator"}:
        plus = get_vecs(target_concept, candidate_operator, layers)
        return make_delta(plus, base, layers, stage_scale, normalize=True)

    if route == "write_address_gate":
        plus = get_vecs(candidate_concept, target_operator, layers)
        return make_delta(plus, base, layers, stage_scale, normalize=True)

    if route == "write_residual_mode":
        plus = get_vecs(candidate_concept, candidate_operator, layers)
        return make_delta(plus, base, layers, stage_scale, normalize=True)

    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--plan", default=DEFAULT_PLAN)
    ap.add_argument("--out_csv", default=DEFAULT_OUT)
    ap.add_argument("--progress_json", default=DEFAULT_PROGRESS)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--max_null_reps", type=int, default=10)
    ap.add_argument("--selection_families", default="")
    ap.add_argument("--delta_scale", type=float, default=0.25, help="Global shrink factor for route deltas.")
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--local_files_only", action="store_true", default=True)
    ap.add_argument("--trust_remote_code", action="store_true", default=True)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--flush_every", type=int, default=20)
    args = ap.parse_args()

    set_seed(SEED)

    plan_path = Path(args.plan)
    if not plan_path.exists():
        raise FileNotFoundError(f"SEM-6J plan not found: {plan_path}")

    plan = pd.read_csv(plan_path)
    required = [
        "_selection", "_selection_family", "target_concept", "target_operator",
        "candidate_concept", "candidate_operator", "write_route", "alpha_operator", "beta_precursor"
    ]
    missing = [c for c in required if c not in plan.columns]
    if missing:
        raise ValueError(f"Plan missing columns: {missing}")

    if "_group_id" not in plan.columns:
        if "family" in plan.columns:
            plan["_group_id"] = plan["family"]
        else:
            plan["_group_id"] = np.arange(len(plan))

    plan = downsample_null_reps(plan, args.max_null_reps)

    if args.selection_families.strip():
        keep = {x.strip() for x in args.selection_families.split(",") if x.strip()}
        plan = plan[plan["_selection_family"].isin(keep)].copy()

    if args.max_rows and args.max_rows > 0:
        plan = plan.head(args.max_rows).copy()

    if "_live_row_id" not in plan.columns:
        plan["_live_row_id"] = np.arange(len(plan), dtype=int)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = Path(args.progress_json)
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume.
    done_ids = set()
    results = []
    if args.resume and out_path.exists():
        old = pd.read_csv(out_path)
        if "_live_row_id" in old.columns:
            done_ids = set(pd.to_numeric(old["_live_row_id"], errors="coerce").dropna().astype(int).tolist())
            results = old.to_dict("records")
            print(f"[resume] loaded {len(done_ids)} completed rows from {out_path}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable; using CPU.")
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

    operator_list = sorted(set(plan["target_operator"].astype(str).tolist()) | set(plan["candidate_operator"].astype(str).tolist()))
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if len(operator_list) > len(letters):
        raise ValueError(f"Too many operators: {len(operator_list)}")
    operator_to_letter = {op: letters[i] for i, op in enumerate(operator_list)}
    letter_to_token_id = {letter: first_token_id_for_label(tokenizer, letter) for letter in operator_to_letter.values()}

    print("Operators:")
    for op, letter in operator_to_letter.items():
        print(f"  {letter}: {op}")

    baseline_cache: Dict[Tuple[str, str], Tuple[Dict[str, float], float, int, float]] = {}
    vector_cache: Dict[Tuple[str, str, Tuple[int, ...]], Dict[int, torch.Tensor]] = {}

    rows = plan.to_dict("records")
    total = len(rows)
    print(f"Running SEM-6J live controller rows={total}, done={len(done_ids)}, delta_scale={args.delta_scale}")

    for idx, row in enumerate(rows):
        live_id = int(row["_live_row_id"])
        if live_id in done_ids:
            continue

        target_concept = str(row["target_concept"])
        target_operator = str(row["target_operator"])
        correct_letter = operator_to_letter[target_operator]
        target_key = (target_concept, target_operator)

        if target_key not in baseline_cache:
            prompt = build_classifier_prompt(target_concept, target_operator, operator_to_letter)
            base_scores = score_prompt_letters(
                model, tokenizer, prompt, letter_to_token_id,
                device=device, max_len=args.max_len,
                deltas=None,
            )
            base_margin, base_rank, base_gap = compute_margin_rank(base_scores, correct_letter)
            baseline_cache[target_key] = (base_scores, base_margin, base_rank, base_gap)
        else:
            base_scores, base_margin, base_rank, base_gap = baseline_cache[target_key]

        deltas = build_route_delta(
            pd.Series(row),
            vector_cache,
            model,
            tokenizer,
            device=device,
            max_len=args.max_len,
            delta_scale=args.delta_scale,
        )

        prompt = build_classifier_prompt(target_concept, target_operator, operator_to_letter)
        live_scores = score_prompt_letters(
            model, tokenizer, prompt, letter_to_token_id,
            device=device, max_len=args.max_len,
            deltas=deltas,
        )
        live_margin, live_rank, live_gap = compute_margin_rank(live_scores, correct_letter)

        out = dict(row)
        out["delta_scale"] = args.delta_scale
        out["correct_letter"] = correct_letter
        out["operator_options_json"] = json.dumps(operator_to_letter, ensure_ascii=False)
        out["base_margin"] = base_margin
        out["base_rank"] = base_rank
        out["base_logprob_gap_to_top"] = base_gap
        out["live_margin"] = live_margin
        out["live_rank"] = live_rank
        out["live_logprob_gap_to_top"] = live_gap
        out["live_margin_gain"] = live_margin - base_margin
        out["live_gain"] = live_margin - base_margin
        out["trajectory_improvement"] = live_margin - base_margin
        out["live_rank_improvement"] = base_rank - live_rank
        out["live_logprob_gap_gain"] = live_gap - base_gap
        out["n_delta_layers"] = len(deltas)
        out["delta_layers"] = ",".join(str(k) for k in sorted(deltas.keys()))

        results.append(out)

        if (len(results) % args.flush_every == 0) or (idx == total - 1):
            pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")
            progress = {
                "completed_rows": len(results),
                "total_planned_rows_after_filter": total,
                "output": str(out_path),
                "last_live_row_id": live_id,
                "delta_scale": args.delta_scale,
                "note": "SEM-6J ASA-style policy/operator live controller.",
            }
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
            print(f"[progress] {len(results)}/{total} rows -> {out_path}")

        if idx % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[DONE] SEM-6J live results written to:\n{out_path}")
    print("\nEvaluate:")
    print("python GPT_sem6j_policy_operator_write_audit.py ^")
    print(f"  --live_results {out_path}")


if __name__ == "__main__":
    main()
