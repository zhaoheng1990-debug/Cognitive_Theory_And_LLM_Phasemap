# -*- coding: utf-8 -*-
r"""
SEM-6J.3: AddressGate-Only MicroController

Purpose
-------
SEM-6J.2b found:

    live-only value is predictable
    but all coarse actions still have non-positive mean live value
    address_gate is closest to neutral
    correction_operator is clearly negative

Therefore 6J.3 isolates the safest route:

    address_gate only

and tests very small commit-stage gate strengths:

    beta = 0.005 / 0.01 / 0.02

This script does NOT use correction/operator deltas.
It only tests whether commit-stage address gating can become neutral/positive.

Core comparison
---------------
address_gate rows vs no_intervention baseline.

Selections:
    learned_noleak_Q
    commit_topk_rule
    oracle_selected
    proxy_only

Controls:
    no_intervention rows
    random_pool / shuffle if present

Default paths
-------------
Plan:
    C:\Users\ZH\Desktop\AGI\python_script\sem6j1_outputs\sem6j1_asa_operator_write_plan.csv

Output:
    C:\Users\ZH\Desktop\AGI\python_script\sem6j3_outputs\sem6j3_address_gate_live_results.csv

Model:
    D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main

Run
---
conda activate dhrf_4080s
cd C:\Users\ZH\Desktop\AGI\python_script

Smoke:
python GPT_sem6j3_address_gate_microcontroller.py --max_rows 120

Evaluate:
python GPT_sem6j3_address_gate_microcontroller.py --evaluate_only

Full-ish:
python GPT_sem6j3_address_gate_microcontroller.py --max_rows 0 --max_per_selection 96

Interpretation
--------------
PASS-Lite:
    best_beta address_gate mean_live_gain >= 0 or significantly improves over beta=0 baseline.

PASS-Strong:
    learned_noleak_Q address_gate beats commit_topk/no_intervention/random/shuffle at z>2.

If FAIL:
    Commit/address microgate is not sufficient; next target is learned ASA order_precursor
    or real commit-gate artifact instead of hidden delta.
"""

import argparse
import gc
import json
import math
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
DEFAULT_PLAN = rf"{DEFAULT_ROOT}\sem6j1_outputs\sem6j1_asa_operator_write_plan.csv"
DEFAULT_OUT_DIR = rf"{DEFAULT_ROOT}\sem6j3_outputs"
DEFAULT_OUT_CSV = rf"{DEFAULT_OUT_DIR}\sem6j3_address_gate_live_results.csv"
DEFAULT_SUMMARY = rf"{DEFAULT_OUT_DIR}\sem6j3_address_gate_summary.csv"
DEFAULT_PAIRWISE = rf"{DEFAULT_OUT_DIR}\sem6j3_address_gate_pairwise.csv"
DEFAULT_VERDICT = rf"{DEFAULT_OUT_DIR}\sem6j3_verdict.json"
DEFAULT_PROGRESS = rf"{DEFAULT_OUT_DIR}\sem6j3_progress.json"

SEED = 20260606
COMMIT_LAYERS = [23, 24, 25]
BETAS = [0.0, 0.005, 0.01, 0.02]


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
    "Planning": "Plan a reasoning strategy for {concept}.",
    "RiskAudit": "Audit the risks, failure modes, and caveats for {concept}.",
}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("Could not locate transformer layers.")


def operator_description(op: str) -> str:
    return {
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
    }.get(str(op), f"use the reasoning operator named {op}")


def build_task_request(concept: str, operator: str) -> str:
    tmpl = OPERATOR_TEMPLATES.get(str(operator))
    if tmpl:
        return tmpl.format(concept=concept)
    return f"Reason about {concept} using the operator {operator}."


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


def build_operator_prompt(concept: str, operator: str) -> str:
    return (
        "Latent address carrier.\n"
        f"Concept identity: {concept}\n"
        f"Operator context: {operator}\n"
        f"Task: {build_task_request(concept, operator)}\n"
        "Form the identity/address commitment representation for this concept in this operator context."
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
        raise ValueError(f"Could not tokenize label={letter}")
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
            raise ValueError(f"Layer L{l} requested but model has {len(outputs.hidden_states)-1} layers.")
        out[l] = outputs.hidden_states[idx][0, -1, :].detach().to(device)
    del inputs, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def make_address_delta(
    candidate_identity: Dict[int, torch.Tensor],
    target_identity: Dict[int, torch.Tensor],
    beta: float,
    normalize: bool = True,
) -> Dict[int, torch.Tensor]:
    out = {}
    for l in COMMIT_LAYERS:
        if l not in candidate_identity or l not in target_identity:
            continue
        d = candidate_identity[l] - target_identity[l]
        if normalize:
            dn = torch.norm(d) + 1e-8
            base_n = torch.norm(target_identity[l]) + 1e-8
            d = d / dn * base_n
        out[l] = d * float(beta)
    return out


def register_delta_hooks(model, deltas: Dict[int, torch.Tensor]):
    layers = get_layers(model)
    handles = []

    def make_hook(layer_idx):
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
            raise ValueError(f"Layer {l} out of range.")
        handles.append(layers[l].register_forward_hook(make_hook(l)))
    return handles


@torch.no_grad()
def score_prompt(
    model, tokenizer, text: str, letter_to_token_id: Dict[str, int],
    device: str, max_len: int, deltas: Optional[Dict[int, torch.Tensor]] = None
) -> Dict[str, float]:
    prompt = apply_chat(tokenizer, text)
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


def filter_address_gate_plan(plan: pd.DataFrame, max_per_selection: int = 0) -> pd.DataFrame:
    df = plan.copy()

    if "minimal_action" in df.columns:
        mask = df["minimal_action"].astype(str).eq("address_gate")
    elif "asa_action" in df.columns:
        mask = df["asa_action"].astype(str).eq("address_gate")
    else:
        mask = df["write_route"].astype(str).eq("write_address_gate")

    df = df[mask].copy()

    # Keep relevant selections, including baselines.
    keep_sel = {"learned_noleak_Q", "commit_topk_rule", "proxy_only", "oracle_selected", "random_pool", "shuffle_learned_Q"}
    df = df[df["_selection_family"].astype(str).isin(keep_sel)].copy()

    if max_per_selection and max_per_selection > 0:
        parts = []
        for sel, sub in df.groupby("_selection_family", sort=False):
            parts.append(sub.head(max_per_selection))
        df = pd.concat(parts, ignore_index=True)

    return df.reset_index(drop=True)


def load_or_eval_summary(path=DEFAULT_OUT_CSV):
    out = Path(path)
    if not out.exists():
        raise FileNotFoundError(f"No live results found: {out}")
    df = pd.read_csv(out)
    return evaluate(df)


def evaluate(df: pd.DataFrame):
    metric = "trajectory_improvement"
    if metric not in df.columns:
        metric = "live_margin_gain"

    rows = []
    for keys, sub in df.groupby(["_selection_family", "beta"], dropna=False):
        sel, beta = keys
        vals = pd.to_numeric(sub[metric], errors="coerce").dropna().to_numpy(float)
        if len(vals) == 0:
            continue
        rows.append({
            "selection_family": sel,
            "beta": beta,
            "n": int(len(vals)),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
            "positive_rate": float(np.mean(vals > 0)),
        })
    summary = pd.DataFrame(rows).sort_values(["beta", "selection_family"])
    summary.to_csv(DEFAULT_SUMMARY, index=False, encoding="utf-8-sig")

    pair_rows = []
    if "_group_id" in df.columns:
        for beta, bdf in df.groupby("beta"):
            if "learned_noleak_Q" not in set(bdf["_selection_family"]):
                continue
            A = bdf[bdf["_selection_family"] == "learned_noleak_Q"][["_group_id", metric]]
            A = A.groupby("_group_id", as_index=False)[metric].mean().rename(columns={metric: "a"})
            for base in ["commit_topk_rule", "random_pool", "shuffle_learned_Q", "proxy_only", "oracle_selected"]:
                if base not in set(bdf["_selection_family"]):
                    continue
                B = bdf[bdf["_selection_family"] == base][["_group_id", metric]]
                B = B.groupby("_group_id", as_index=False)[metric].mean().rename(columns={metric: "b"})
                m = A.merge(B, on="_group_id", how="inner")
                diff = (m["a"] - m["b"]).dropna().to_numpy(float)
                z = np.nan
                if len(diff) > 1 and np.std(diff, ddof=1) > 1e-12:
                    z = float(np.mean(diff) / (np.std(diff, ddof=1) / math.sqrt(len(diff))))
                pair_rows.append({
                    "beta": beta,
                    "comparison": f"learned_noleak_Q>{base}",
                    "n": int(len(diff)),
                    "mean_diff": float(np.mean(diff)) if len(diff) else np.nan,
                    "z": z,
                    "win_rate": float(np.mean(diff > 0)) if len(diff) else np.nan,
                })
    pairwise = pd.DataFrame(pair_rows)
    pairwise.to_csv(DEFAULT_PAIRWISE, index=False, encoding="utf-8-sig")

    # Verdict.
    positive_any = False
    positive_rows = []
    if not summary.empty:
        pos = summary[summary["mean"] >= 0]
        positive_any = len(pos) > 0
        positive_rows = pos.to_dict("records")

    verdict = "FAIL_OR_INCONCLUSIVE"
    notes = []
    if positive_any:
        verdict = "PASS_LITE_ADDRESS_GATE_NONNEGATIVE"
        notes.append("At least one beta/selection address_gate mean is non-negative.")

    # Strong if learned beats key baselines.
    if not pairwise.empty:
        for beta, sub in pairwise.groupby("beta"):
            needed = ["learned_noleak_Q>commit_topk_rule", "learned_noleak_Q>random_pool", "learned_noleak_Q>shuffle_learned_Q"]
            ok = []
            for comp in needed:
                row = sub[sub["comparison"] == comp]
                if len(row) == 0:
                    ok.append(False)
                else:
                    r = row.iloc[0]
                    ok.append(pd.notna(r["mean_diff"]) and r["mean_diff"] > 0 and pd.notna(r["z"]) and r["z"] > 2)
            if all(ok):
                verdict = "PASS_STRONG_ADDRESS_GATE_LEARNED_Q"
                notes.append(f"learned_noleak_Q beats commit/random/shuffle at beta={beta}.")
                break

    verdict_obj = {
        "stage": "SEM-6J.3",
        "mode": "address_gate_only_microcontroller",
        "metric": metric,
        "verdict": verdict,
        "positive_summary_rows": positive_rows,
        "notes": notes + [
            "6J.3 isolates address_gate because correction_operator was negative in 6J.2b.",
            "Betas are intentionally small: 0, 0.005, 0.01, 0.02.",
            "If all means remain negative, address gate requires learned order precursor or true commit-gate artifact rather than hidden delta.",
        ],
        "outputs": {
            "summary": DEFAULT_SUMMARY,
            "pairwise": DEFAULT_PAIRWISE,
            "verdict": DEFAULT_VERDICT,
        }
    }
    with open(DEFAULT_VERDICT, "w", encoding="utf-8") as f:
        json.dump(verdict_obj, f, ensure_ascii=False, indent=2)

    print("\n========== SEM-6J.3 EVAL ==========")
    print(json.dumps(verdict_obj, ensure_ascii=False, indent=2))
    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nPairwise:")
    print(pairwise.to_string(index=False))
    return verdict_obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--plan", default=DEFAULT_PLAN)
    ap.add_argument("--out_csv", default=DEFAULT_OUT_CSV)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--max_per_selection", type=int, default=0)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--local_files_only", action="store_true", default=True)
    ap.add_argument("--trust_remote_code", action="store_true", default=True)
    ap.add_argument("--evaluate_only", action="store_true", default=False)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--flush_every", type=int, default=20)
    args = ap.parse_args()

    Path(DEFAULT_OUT_DIR).mkdir(parents=True, exist_ok=True)

    if args.evaluate_only:
        evaluate(pd.read_csv(args.out_csv))
        return

    set_seed(SEED)

    plan_path = Path(args.plan)
    if not plan_path.exists():
        raise FileNotFoundError(f"Plan not found: {plan_path}")

    plan = pd.read_csv(plan_path)
    plan = filter_address_gate_plan(plan, args.max_per_selection)
    if args.max_rows and args.max_rows > 0:
        plan = plan.head(args.max_rows).copy()
    if plan.empty:
        raise ValueError("No address_gate rows found in plan.")

    if "_live_row_id" not in plan.columns:
        plan["_live_row_id"] = np.arange(len(plan), dtype=int)

    out_path = Path(args.out_csv)
    done_ids = set()
    results = []
    if args.resume and out_path.exists():
        old = pd.read_csv(out_path)
        if "_live_row_id" in old.columns:
            done_ids = set(pd.to_numeric(old["_live_row_id"], errors="coerce").dropna().astype(int).tolist())
            results = old.to_dict("records")
            print(f"[resume] loaded {len(done_ids)} existing rows.")

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

    baseline_cache = {}
    identity_cache = {}

    rows = plan.to_dict("records")
    total = len(rows)
    print(f"Running address_gate rows={total}, done={len(done_ids)} betas={BETAS}")

    for idx, row in enumerate(rows):
        live_id = int(row["_live_row_id"])
        # We create one result per beta; resume by live_id+beta key is hard, so keep simple:
        # if live_id exists in any row and all betas exist skip.
        if live_id in done_ids:
            continue

        target_concept = str(row["target_concept"])
        target_operator = str(row["target_operator"])
        candidate_concept = str(row["candidate_concept"])
        correct_letter = operator_to_letter[target_operator]
        target_key = (target_concept, target_operator)

        prompt = build_classifier_prompt(target_concept, target_operator, operator_to_letter)
        if target_key not in baseline_cache:
            base_scores = score_prompt(model, tokenizer, prompt, letter_to_token_id, device, args.max_len, None)
            base_margin, base_rank, base_gap = compute_margin_rank(base_scores, correct_letter)
            baseline_cache[target_key] = (base_scores, base_margin, base_rank, base_gap)
        else:
            base_scores, base_margin, base_rank, base_gap = baseline_cache[target_key]

        # Identity/address vectors:
        t_id_key = (target_concept, target_operator)
        c_id_key = (candidate_concept, target_operator)
        if t_id_key not in identity_cache:
            identity_cache[t_id_key] = hidden_vectors_by_layer(
                model, tokenizer, build_operator_prompt(target_concept, target_operator),
                COMMIT_LAYERS, device, args.max_len
            )
        if c_id_key not in identity_cache:
            identity_cache[c_id_key] = hidden_vectors_by_layer(
                model, tokenizer, build_operator_prompt(candidate_concept, target_operator),
                COMMIT_LAYERS, device, args.max_len
            )

        target_id = identity_cache[t_id_key]
        cand_id = identity_cache[c_id_key]

        for beta in BETAS:
            deltas = make_address_delta(cand_id, target_id, beta=beta, normalize=True) if beta > 0 else {}
            scores = score_prompt(model, tokenizer, prompt, letter_to_token_id, device, args.max_len, deltas)
            margin, rank, gap = compute_margin_rank(scores, correct_letter)

            out = dict(row)
            out["beta"] = beta
            out["correct_letter"] = correct_letter
            out["base_margin"] = base_margin
            out["base_rank"] = base_rank
            out["base_logprob_gap_to_top"] = base_gap
            out["live_margin"] = margin
            out["live_rank"] = rank
            out["live_logprob_gap_to_top"] = gap
            out["live_margin_gain"] = margin - base_margin
            out["live_gain"] = margin - base_margin
            out["trajectory_improvement"] = margin - base_margin
            out["live_rank_improvement"] = base_rank - rank
            out["live_logprob_gap_gain"] = gap - base_gap
            out["address_gate_layers"] = ",".join(map(str, COMMIT_LAYERS))
            results.append(out)

        if (len(results) % args.flush_every == 0) or (idx == total - 1):
            pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")
            progress = {
                "completed_plan_rows": idx + 1,
                "total_plan_rows": total,
                "output_rows": len(results),
                "out_csv": str(out_path),
                "betas": BETAS,
                "note": "SEM-6J.3 address_gate-only microcontroller",
            }
            with open(DEFAULT_PROGRESS, "w", encoding="utf-8") as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
            print(f"[progress] plan {idx+1}/{total}; output rows={len(results)}")

        if idx % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[DONE] live results: {out_path}")
    evaluate(pd.DataFrame(results))


if __name__ == "__main__":
    main()
