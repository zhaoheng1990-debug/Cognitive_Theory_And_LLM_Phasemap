# -*- coding: utf-8 -*-
r"""
SEM-6J.3b: AddressGate Candidate Expansion

Purpose
-------
SEM-6J.3 showed weak positive address_gate signal, but learned_noleak_Q address-gate
sample size was too small (n≈6 in the evaluated subset).

SEM-6J.3b expands address_gate candidates from the full scored candidate pool:

    sem6h1b_outputs\sem6h1b_scored_candidates.csv

For each target family, select top-m commit_topk/address_gate candidates by:

    learned_noleak_Q
    oracle_utility
    proxy_utility
    random

Then run address_gate-only live controller at beta=0.02 by default.

This directly tests:

    learned_Q_address_topm > random_address_topm

and whether address_gate primitive is genuinely useful when candidate coverage is
expanded.

Default paths
-------------
Candidates:
    C:\Users\ZH\Desktop\AGI\python_script\sem6h1b_outputs\sem6h1b_scored_candidates.csv

Output plan:
    C:\Users\ZH\Desktop\AGI\python_script\sem6j3b_outputs\sem6j3b_address_gate_expanded_plan.csv

Live results:
    C:\Users\ZH\Desktop\AGI\python_script\sem6j3b_outputs\sem6j3b_address_gate_live_results.csv

Run
---
conda activate dhrf_4080s
cd C:\Users\ZH\Desktop\AGI\python_script

# Build expanded plan only:
python GPT_sem6j3b_address_gate_candidate_expansion.py --plan_only

# Smoke live:
python GPT_sem6j3b_address_gate_candidate_expansion.py --max_families 10 --top_m 3

# Full live:
python GPT_sem6j3b_address_gate_candidate_expansion.py --top_m 3

# Evaluate existing live results:
python GPT_sem6j3b_address_gate_candidate_expansion.py --evaluate_only

Notes
-----
This is still a hidden-delta address gate. If expanded learned-Q address candidates
fail, the next move is to learn a true commit-gate / order-precursor artifact rather
than using hidden identity deltas.
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
DEFAULT_CANDIDATES = rf"{DEFAULT_ROOT}\sem6h1b_outputs\sem6h1b_scored_candidates.csv"
DEFAULT_OUT_DIR = rf"{DEFAULT_ROOT}\sem6j3b_outputs"
DEFAULT_PLAN = rf"{DEFAULT_OUT_DIR}\sem6j3b_address_gate_expanded_plan.csv"
DEFAULT_LIVE = rf"{DEFAULT_OUT_DIR}\sem6j3b_address_gate_live_results.csv"
DEFAULT_SUMMARY = rf"{DEFAULT_OUT_DIR}\sem6j3b_address_gate_summary.csv"
DEFAULT_PAIRWISE = rf"{DEFAULT_OUT_DIR}\sem6j3b_address_gate_pairwise.csv"
DEFAULT_VERDICT = rf"{DEFAULT_OUT_DIR}\sem6j3b_verdict.json"
DEFAULT_PROGRESS = rf"{DEFAULT_OUT_DIR}\sem6j3b_progress.json"

SEED = 20260606
COMMIT_LAYERS = [23, 24, 25]

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


def find_col(df: pd.DataFrame, candidates: List[str], required=True):
    nmap = {norm(c): c for c in df.columns}
    for c in candidates:
        if norm(c) in nmap:
            return nmap[norm(c)]
    for c in candidates:
        nc = norm(c)
        for k, v in nmap.items():
            if nc in k or k in nc:
                return v
    if required:
        raise ValueError(f"Missing col among {candidates}. Columns={list(df.columns)}")
    return None


def detect_cols(df: pd.DataFrame):
    return {
        "family": find_col(df, ["family", "target_family", "query_id", "row_id"]),
        "pool": find_col(df, ["pool", "candidate_type", "candidate_source"], required=False),
        "candidate_i": find_col(df, ["candidate_i", "candidate_id", "cand_id"], required=False),
        "target_concept": find_col(df, ["target_concept", "concept"]),
        "target_operator": find_col(df, ["target_operator", "operator"]),
        "candidate_family": find_col(df, ["candidate_family", "cand_family"]),
        "candidate_concept": find_col(df, ["candidate_concept", "cand_concept"]),
        "candidate_operator": find_col(df, ["candidate_operator", "cand_operator"]),
        "q": find_col(df, ["_learned_noleak_q", "learned_noleak_q", "q_score"]),
        "oracle": find_col(df, ["oracle_utility", "utility", "true_utility"], required=False),
        "proxy": find_col(df, ["proxy_utility", "_proxy_score", "proxy_score"], required=False),
    }


def add_flags(df: pd.DataFrame, pool_col: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    if pool_col and pool_col in out.columns:
        s = out[pool_col].astype(str).str.lower()
    else:
        s = pd.Series([""] * len(out), index=out.index)
    out["_is_commit_topk"] = (
        out["_is_commit_topk"].astype(bool)
        if "_is_commit_topk" in out.columns
        else s.str.contains("commit|topk|top_k|neighbor", regex=True)
    )
    return out


def build_expanded_plan(candidates_path: Path, top_m: int, max_families: int = 0) -> pd.DataFrame:
    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidates not found: {candidates_path}")
    df = pd.read_csv(candidates_path)
    cols = detect_cols(df)
    df = add_flags(df, cols["pool"])

    # Normalize names.
    rename = {
        cols["family"]: "family",
        cols["target_concept"]: "target_concept",
        cols["target_operator"]: "target_operator",
        cols["candidate_family"]: "candidate_family",
        cols["candidate_concept"]: "candidate_concept",
        cols["candidate_operator"]: "candidate_operator",
        cols["q"]: "_learned_noleak_q",
    }
    if cols["pool"]: rename[cols["pool"]] = "pool"
    if cols["candidate_i"]: rename[cols["candidate_i"]] = "candidate_i"
    if cols["oracle"]: rename[cols["oracle"]] = "oracle_utility"
    if cols["proxy"]: rename[cols["proxy"]] = "proxy_utility"
    df = df.rename(columns={k: v for k, v in rename.items() if k != v})

    if "candidate_i" not in df.columns:
        df["candidate_i"] = np.arange(len(df))

    # Address-gate candidates = commit_topk neighbor rows.
    addr = df[df["_is_commit_topk"].fillna(False).astype(bool)].copy()
    if addr.empty:
        # fallback pool string
        addr = df[df["pool"].astype(str).str.lower().str.contains("commit|topk|neighbor", regex=True)].copy()
    if addr.empty:
        raise ValueError("No commit_topk/address_gate candidate rows found.")

    families = sorted(addr["family"].astype(str).unique().tolist())
    if max_families and max_families > 0:
        families = families[:max_families]
        addr = addr[addr["family"].astype(str).isin(families)].copy()

    rng = np.random.default_rng(SEED)
    rows = []

    def add_rows(sub: pd.DataFrame, selection: str, score_col: Optional[str], ascending: bool = False):
        if score_col is not None and score_col in sub.columns:
            tmp = sub.copy()
            tmp[score_col] = pd.to_numeric(tmp[score_col], errors="coerce")
            tmp = tmp.sort_values(score_col, ascending=ascending, na_position="last").head(top_m)
        else:
            tmp = sub.sample(n=min(top_m, len(sub)), replace=False, random_state=SEED)
        for rank, (_, r) in enumerate(tmp.iterrows(), start=1):
            row = r.to_dict()
            row["_selection_family"] = selection
            row["_selection"] = f"{selection}_top{rank}"
            row["_group_id"] = r["family"]
            row["minimal_action"] = "address_gate"
            row["asa_action"] = "address_gate"
            row["write_route"] = "write_address_gate"
            row["relation_type"] = "commit_topk_neighbor"
            row["asa_stage"] = "commit_L23_L25"
            row["asa_action_family"] = "addressing"
            row["asa_alpha"] = 0.0
            row["asa_beta"] = np.nan
            rows.append(row)

    for fam, sub in addr.groupby("family", sort=False):
        add_rows(sub, "learned_Q_address_topm", "_learned_noleak_q", ascending=False)
        if "oracle_utility" in sub.columns:
            add_rows(sub, "oracle_address_topm", "oracle_utility", ascending=False)
        if "proxy_utility" in sub.columns:
            add_rows(sub, "proxy_address_topm", "proxy_utility", ascending=False)
        # random
        add_rows(sub, "random_address_topm", None)

    plan = pd.DataFrame(rows).reset_index(drop=True)
    plan["_live_row_id"] = np.arange(len(plan), dtype=int)
    plan["sem6j3b_hypothesis"] = (
        "Expanded address-gate candidates test whether learned-Q top-m commit neighbors "
        "beat random address candidates at small beta."
    )
    return plan


# ==== live controller reused from 6J.3 ====

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
    options = "\n".join(f"{letter}. {op}: {operator_description(op)}" for op, letter in operator_to_letter.items())
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
        return tokenizer.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True)
    return user_text + "\nAssistant:"


def first_token_id_for_label(tokenizer, letter: str) -> int:
    variants = [letter, " " + letter, "\n" + letter]
    best = None
    for v in variants:
        ids = tokenizer(v, add_special_tokens=False).input_ids
        if ids and (best is None or len(ids) < len(best)):
            best = ids
    if best is None:
        raise ValueError(f"Could not tokenize label {letter}")
    return int(best[0])


@torch.no_grad()
def hidden_vectors_by_layer(model, tokenizer, text: str, layers: List[int], device: str, max_len: int) -> Dict[int, torch.Tensor]:
    prompt = apply_chat(tokenizer, text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    out = {}
    for l in layers:
        idx = l + 1
        out[l] = outputs.hidden_states[idx][0, -1, :].detach().to(device)
    del inputs, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def make_address_delta(cand, target, beta: float, normalize=True):
    out = {}
    for l in COMMIT_LAYERS:
        d = cand[l] - target[l]
        if normalize:
            dn = torch.norm(d) + 1e-8
            base_n = torch.norm(target[l]) + 1e-8
            d = d / dn * base_n
        out[l] = d * float(beta)
    return out


def register_hooks(model, deltas):
    layers = get_layers(model)
    handles = []
    def make_hook(l):
        def hook(module, inputs, output):
            if l not in deltas:
                return output
            if isinstance(output, tuple):
                hidden = output[0]; rest = output[1:]
            else:
                hidden = output; rest = None
            h = hidden.clone()
            d = deltas[l].to(device=h.device, dtype=h.dtype).view(1,1,-1)
            h[:, -1:, :] = h[:, -1:, :] + d
            if rest is None: return h
            return (h,) + rest
        return hook
    for l in deltas.keys():
        handles.append(layers[l].register_forward_hook(make_hook(l)))
    return handles


@torch.no_grad()
def score_prompt(model, tokenizer, text, letter_to_token_id, device, max_len, deltas=None):
    prompt = apply_chat(tokenizer, text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    handles = []
    try:
        if deltas:
            handles = register_hooks(model, deltas)
        outputs = model(**inputs, use_cache=False)
        logits = outputs.logits[0, -1, :].float()
        logp = torch.log_softmax(logits, dim=-1)
        return {letter: float(logp[tok].detach().cpu()) for letter, tok in letter_to_token_id.items()}
    finally:
        for h in handles: h.remove()
        del inputs
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def compute_margin_rank(scores, correct_letter):
    correct = scores[correct_letter]
    best_other = max(v for k,v in scores.items() if k != correct_letter)
    margin = correct - best_other
    order = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    rank = order.index(correct_letter) + 1
    gap = correct - scores[order[0]]
    return float(margin), int(rank), float(gap)


def run_live(plan: pd.DataFrame, args):
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable; using CPU.")
        device = "cpu"
    dtype = torch.float16 if args.dtype == "float16" and device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=args.local_files_only, trust_remote_code=args.trust_remote_code)
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

    operator_list = sorted(set(plan["target_operator"].astype(str).tolist()) | set(plan["candidate_operator"].astype(str).tolist()))
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    operator_to_letter = {op: letters[i] for i, op in enumerate(operator_list)}
    letter_to_token_id = {letter: first_token_id_for_label(tokenizer, letter) for letter in operator_to_letter.values()}

    baseline_cache = {}
    identity_cache = {}
    results = []

    rows = plan.to_dict("records")
    print(f"Running SEM-6J.3b rows={len(rows)} beta={args.beta}")

    for idx, row in enumerate(rows):
        target_concept = str(row["target_concept"])
        target_operator = str(row["target_operator"])
        candidate_concept = str(row["candidate_concept"])
        correct_letter = operator_to_letter[target_operator]

        target_key = (target_concept, target_operator)
        prompt = build_classifier_prompt(target_concept, target_operator, operator_to_letter)

        if target_key not in baseline_cache:
            base_scores = score_prompt(model, tokenizer, prompt, letter_to_token_id, device, args.max_len, None)
            base_margin, base_rank, base_gap = compute_margin_rank(base_scores, correct_letter)
            baseline_cache[target_key] = (base_margin, base_rank, base_gap)
        else:
            base_margin, base_rank, base_gap = baseline_cache[target_key]

        t_id_key = (target_concept, target_operator)
        c_id_key = (candidate_concept, target_operator)
        if t_id_key not in identity_cache:
            identity_cache[t_id_key] = hidden_vectors_by_layer(model, tokenizer, build_operator_prompt(target_concept, target_operator), COMMIT_LAYERS, device, args.max_len)
        if c_id_key not in identity_cache:
            identity_cache[c_id_key] = hidden_vectors_by_layer(model, tokenizer, build_operator_prompt(candidate_concept, target_operator), COMMIT_LAYERS, device, args.max_len)

        deltas = make_address_delta(identity_cache[c_id_key], identity_cache[t_id_key], beta=args.beta, normalize=True)
        live_scores = score_prompt(model, tokenizer, prompt, letter_to_token_id, device, args.max_len, deltas)
        margin, rank, gap = compute_margin_rank(live_scores, correct_letter)

        out = dict(row)
        out["beta"] = args.beta
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
        results.append(out)

        if (idx+1) % args.flush_every == 0 or idx == len(rows)-1:
            pd.DataFrame(results).to_csv(args.live_out, index=False, encoding="utf-8-sig")
            progress = {"completed_rows": idx+1, "total_rows": len(rows), "out": args.live_out, "beta": args.beta}
            with open(DEFAULT_OUT_DIR + r"\sem6j3b_progress.json", "w", encoding="utf-8") as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
            print(f"[progress] {idx+1}/{len(rows)}")

        if idx % 50 == 0:
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    return pd.DataFrame(results)


def evaluate(df: pd.DataFrame):
    metric = "trajectory_improvement"
    summary_rows = []
    for sel, sub in df.groupby("_selection_family"):
        vals = pd.to_numeric(sub[metric], errors="coerce").dropna().to_numpy(float)
        summary_rows.append({
            "selection_family": sel,
            "n": int(len(vals)),
            "mean": float(np.mean(vals)) if len(vals) else np.nan,
            "median": float(np.median(vals)) if len(vals) else np.nan,
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
            "positive_rate": float(np.mean(vals > 0)) if len(vals) else np.nan,
        })
    summary = pd.DataFrame(summary_rows).sort_values("mean", ascending=False)
    summary.to_csv(DEFAULT_SUMMARY, index=False, encoding="utf-8-sig")

    pair_rows = []
    base = "learned_Q_address_topm"
    if "_group_id" in df.columns and base in set(df["_selection_family"]):
        A = df[df["_selection_family"] == base][["_group_id", metric]]
        A = A.groupby("_group_id", as_index=False)[metric].mean().rename(columns={metric: "a"})
        for b in ["random_address_topm", "proxy_address_topm", "oracle_address_topm"]:
            if b not in set(df["_selection_family"]): continue
            B = df[df["_selection_family"] == b][["_group_id", metric]]
            B = B.groupby("_group_id", as_index=False)[metric].mean().rename(columns={metric: "b"})
            m = A.merge(B, on="_group_id")
            diff = (m["a"] - m["b"]).dropna().to_numpy(float)
            z = np.nan
            if len(diff) > 1 and np.std(diff, ddof=1) > 1e-12:
                z = float(np.mean(diff)/(np.std(diff, ddof=1)/math.sqrt(len(diff))))
            pair_rows.append({
                "comparison": f"{base}>{b}",
                "n": int(len(diff)),
                "mean_diff": float(np.mean(diff)) if len(diff) else np.nan,
                "z": z,
                "win_rate": float(np.mean(diff > 0)) if len(diff) else np.nan,
            })
    pair = pd.DataFrame(pair_rows)
    pair.to_csv(DEFAULT_PAIRWISE, index=False, encoding="utf-8-sig")

    verdict = "FAIL_OR_INCONCLUSIVE"
    notes = []
    pos = summary[summary["mean"] >= 0] if not summary.empty else pd.DataFrame()
    if len(pos):
        verdict = "PASS_LITE_ADDRESS_EXPANSION_NONNEGATIVE"
        notes.append("At least one expanded address selection has non-negative mean.")
    if not pair.empty:
        row = pair[pair["comparison"] == "learned_Q_address_topm>random_address_topm"]
        if len(row):
            r = row.iloc[0]
            if pd.notna(r["mean_diff"]) and r["mean_diff"] > 0 and pd.notna(r["z"]) and r["z"] > 2:
                verdict = "PASS_STRONG_LEARNED_Q_ADDRESS_GT_RANDOM"
                notes.append("learned_Q_address_topm beats random_address_topm at z>2.")

    verdict_obj = {
        "stage": "SEM-6J.3b",
        "mode": "address_gate_candidate_expansion",
        "metric": metric,
        "verdict": verdict,
        "notes": notes + [
            "This expands address-gate candidate coverage from replay-selected rows to top-m candidates per family.",
            "If learned_Q does not beat random, address-gate candidate selection still insufficient."
        ],
        "outputs": {"summary": DEFAULT_SUMMARY, "pairwise": DEFAULT_PAIRWISE, "verdict": DEFAULT_VERDICT}
    }
    with open(DEFAULT_VERDICT, "w", encoding="utf-8") as f:
        json.dump(verdict_obj, f, ensure_ascii=False, indent=2)

    print("\n========== SEM-6J.3b EVAL ==========")
    print(json.dumps(verdict_obj, ensure_ascii=False, indent=2))
    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nPairwise:")
    print(pair.to_string(index=False))
    return verdict_obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    ap.add_argument("--plan_out", default=DEFAULT_PLAN)
    ap.add_argument("--live_out", default=DEFAULT_LIVE)
    ap.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--top_m", type=int, default=3)
    ap.add_argument("--max_families", type=int, default=0)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--local_files_only", action="store_true", default=True)
    ap.add_argument("--trust_remote_code", action="store_true", default=True)
    ap.add_argument("--plan_only", action="store_true", default=False)
    ap.add_argument("--evaluate_only", action="store_true", default=False)
    ap.add_argument("--flush_every", type=int, default=20)
    args = ap.parse_args()

    Path(DEFAULT_OUT_DIR).mkdir(parents=True, exist_ok=True)

    if args.evaluate_only:
        df = pd.read_csv(args.live_out)
        evaluate(df)
        return

    set_seed(SEED)

    plan = build_expanded_plan(Path(args.candidates), top_m=args.top_m, max_families=args.max_families)
    plan.to_csv(args.plan_out, index=False, encoding="utf-8-sig")
    print(f"[plan] wrote {len(plan)} rows -> {args.plan_out}")
    print(plan["_selection_family"].value_counts().to_string())

    if args.plan_only:
        return

    live = run_live(plan, args)
    live.to_csv(args.live_out, index=False, encoding="utf-8-sig")
    evaluate(live)


if __name__ == "__main__":
    main()
