# -*- coding: utf-8 -*-
r"""
SEM-6L.5B: True CanonicalOperator Live Replay

Purpose
-------
SEM-6L.5 produced an important boundary result:

    CanonicalOperatorDraft exists,
    but final OperatorMemory write is not closed by offline replay.

SEM-6L.5B is the true model-forward replay step.

It reads:
    sem6l5_outputs\sem6l5_live_replay_plan.csv
    sem6l4_outputs\sem6l4_operator_library_draft.json

and evaluates four replay conditions:

    canonical_operator
    no_operator
    wrong_operator
    random_operator

by converting CanonicalOperator cards into explicit routing-policy prompt prefixes.

This is deliberately implemented as prompt-level operator replay, not hidden injection,
because SEM-6L theory says the long-term write target is ExpansionMap / OperatorRoutingMap,
not a hidden vector.

Model-forward metric
--------------------
For each task row, build a forced-choice operator classification prompt.

The target operator is the correct answer.
We score next-token logprob of the correct option letter.

Replay value:
    live_margin_gain = margin(condition) - margin(no_operator_baseline)

Where:
    margin = logp(correct_letter) - max_other_logp

Main tests
----------
canonical_operator > no_operator
canonical_operator > wrong_operator
canonical_operator > random_operator

Outputs
-------
sem6l5b_outputs\
    sem6l5b_true_live_results.csv
    sem6l5b_operator_vs_controls.csv
    sem6l5b_operator_summary.csv
    sem6l5b_verdict.json

Run
---
conda activate dhrf_4080s
cd C:\Users\ZH\Desktop\AGI\python_script

Smoke:
python GPT_sem6l5b_true_canonical_operator_live_replay.py --max_rows 120

Full:
python GPT_sem6l5b_true_canonical_operator_live_replay.py

Evaluate existing:
python GPT_sem6l5b_true_canonical_operator_live_replay.py --evaluate_only

Notes
-----
This is a conservative proof-of-routing replay. It does not inject activations.
If this passes, it supports writing these CanonicalOperators into W'_cognitive /
OperatorMemory as prompt/routing policies. A later 6L.6 can test hidden/agentic
controller integration.
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
DEFAULT_PLAN = rf"{DEFAULT_ROOT}\sem6l5_outputs\sem6l5_live_replay_plan.csv"
DEFAULT_LIBRARY = rf"{DEFAULT_ROOT}\sem6l4_outputs\sem6l4_operator_library_draft.json"
DEFAULT_OUT_DIR = rf"{DEFAULT_ROOT}\sem6l5b_outputs"
DEFAULT_OUT_CSV = rf"{DEFAULT_OUT_DIR}\sem6l5b_true_live_results.csv"
DEFAULT_VERDICT = rf"{DEFAULT_OUT_DIR}\sem6l5b_verdict.json"

SEED = 20260606

def json_safe(obj):
    """Recursively convert numpy/pandas scalar objects to JSON-safe Python objects."""
    import math
    import numpy as _np
    import pandas as _pd

    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (_np.bool_,)):
        return bool(obj)
    if isinstance(obj, (_np.ndarray,)):
        return [json_safe(x) for x in obj.tolist()]
    if isinstance(obj, (_pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(json_safe(k)): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(x) for x in obj]
    try:
        if _pd.isna(obj):
            return None
    except Exception:
        pass
    return str(obj)


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


def load_json(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_chat(tokenizer, text: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return text + "\nAssistant:"


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


def first_token_id_for_label(tokenizer, letter: str) -> int:
    variants = [letter, " " + letter, "\n" + letter]
    best = None
    for v in variants:
        ids = tokenizer(v, add_special_tokens=False).input_ids
        if ids and (best is None or len(ids) < len(best)):
            best = ids
    if best is None:
        raise ValueError(f"Could not tokenize option label {letter}")
    return int(best[0])


def load_operator_cards(library_path: str) -> Dict[str, dict]:
    lib = load_json(library_path)
    cards = {}
    for op in lib.get("operators", []):
        op_id = op.get("operator_id")
        if op_id:
            cards[op_id] = op
    if not cards:
        raise ValueError("No operators found in operator library draft.")
    return cards


def policy_text(operator_id: str, cards: Dict[str, dict], condition: str, wrong_operator_id: str = "") -> str:
    """
    Prompt-level implementation of ExpansionMap / OperatorRoutingMap update.
    It should alter unfolding policy, not answer token.
    """
    if condition == "no_operator":
        return ""

    if condition in {"wrong_operator", "random_operator"}:
        operator_id = wrong_operator_id or operator_id

    card = cards.get(operator_id, {})
    intent = card.get("intent", "")
    boundary = card.get("applicability_boundary", "")

    if operator_id == "ConstraintInsertionOperator":
        body = (
            "Before solving, insert a constraint-checking step. "
            "Identify boundary conditions, risk factors, and failure modes. "
            "Do not jump directly to a conclusion; first audit whether the requested reasoning "
            "needs an added constraint or boundary check."
        )
    elif operator_id == "CorrectionBranchOperator":
        body = (
            "Before using the requested reasoning operator, add a grounding branch. "
            "First restate the core definition and clarify the target concept, then continue with the requested operator. "
            "Use the grounding branch only to stabilize the unfolding, not to replace the task."
        )
    elif operator_id == "TrajectoryClassExpansionOperator":
        body = (
            "Allow an additional trajectory class when the default unfolding may be incomplete. "
            "Compare mechanism, causal, relation, and property trajectories, then select the one that best matches the task."
        )
    elif operator_id == "AddressPriorIncreaseOperator":
        body = (
            "Increase address and identity priority. "
            "First lock onto the correct concept/relation anchor and preserve that anchor during reasoning. "
            "Avoid drifting to a nearby but wrong concept."
        )
    else:
        body = f"Apply the canonical operator {operator_id}: {intent}"

    return (
        "Canonical operator routing policy:\n"
        f"OperatorID: {operator_id}\n"
        f"Intent: {intent}\n"
        f"Boundary: {boundary}\n"
        f"Policy: {body}\n"
        "Use this policy to choose the correct reasoning operator for the request.\n"
    )


def build_classifier_prompt(
    concept: str,
    target_operator: str,
    operator_to_letter: Dict[str, str],
    route_policy_text: str = "",
) -> str:
    request = build_task_request(concept, target_operator)
    options = "\n".join(
        f"{letter}. {op}: {operator_description(op)}"
        for op, letter in operator_to_letter.items()
    )
    prefix = route_policy_text.strip()
    if prefix:
        prefix = prefix + "\n\n"
    return (
        prefix +
        "You are classifying the reasoning operator required by a request.\n"
        "Choose exactly one option letter. Do not explain.\n\n"
        f"Options:\n{options}\n\n"
        f"Request:\n{request}\n\n"
        "Answer with one option letter only:"
    )


@torch.no_grad()
def score_next_letter(model, tokenizer, text: str, letter_to_token_id: Dict[str, int], device: str, max_len: int) -> Dict[str, float]:
    prompt = apply_chat(tokenizer, text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    outputs = model(**inputs, use_cache=False)
    logits = outputs.logits[0, -1, :].float()
    logp = torch.log_softmax(logits, dim=-1)
    scores = {letter: float(logp[token_id].detach().cpu()) for letter, token_id in letter_to_token_id.items()}
    del inputs, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores


def margin_and_rank(scores: Dict[str, float], correct_letter: str) -> Tuple[float, int]:
    correct = scores[correct_letter]
    others = [v for k, v in scores.items() if k != correct_letter]
    best_other = max(others) if others else float("-inf")
    margin = correct - best_other
    order = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    rank = order.index(correct_letter) + 1
    return float(margin), int(rank)


def normalize_plan(plan: pd.DataFrame, cards: Dict[str, dict]) -> pd.DataFrame:
    df = plan.copy()
    required = ["canonical_operator_id", "replay_condition", "target_concept", "target_operator"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Plan missing required column: {c}")

    for c in ["wrong_operator_id", "candidate_concept", "candidate_operator", "_group_id", "source_experiment", "_selection_family"]:
        if c not in df.columns:
            df[c] = ""

    # Keep only conditions and known operators.
    df = df[df["replay_condition"].isin(["canonical_operator", "no_operator", "wrong_operator", "random_operator"])].copy()
    df = df[df["canonical_operator_id"].isin(cards.keys())].copy()

    if "replay_root_id" not in df.columns:
        # Recover root from replay_plan_id if available. canonical id in previous script was op__i, controls op__i__...
        if "replay_plan_id" in df.columns:
            df["replay_root_id"] = (
                df["replay_plan_id"].astype(str)
                .str.replace(r"__(no|random|wrong_.*)$", "", regex=True)
            )
        else:
            df["replay_root_id"] = (
                df["canonical_operator_id"].astype(str) + "__" +
                df.groupby(["canonical_operator_id", "target_concept", "target_operator"]).cumcount().astype(str)
            )
    return df.reset_index(drop=True)


def run_live(args):
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    cards = load_operator_cards(args.library)
    plan = pd.read_csv(args.plan)
    plan = normalize_plan(plan, cards)

    if args.max_rows and args.max_rows > 0:
        # Keep balanced by operator / condition where possible.
        parts = []
        per_group = max(1, args.max_rows // max(1, plan[["canonical_operator_id", "replay_condition"]].drop_duplicates().shape[0]))
        for _, sub in plan.groupby(["canonical_operator_id", "replay_condition"], sort=False):
            parts.append(sub.head(per_group))
        plan = pd.concat(parts, ignore_index=True).head(args.max_rows)

    # Operator option list from all target/candidate operators in plan.
    op_set = set(plan["target_operator"].astype(str).tolist())
    if "candidate_operator" in plan.columns:
        op_set |= set([x for x in plan["candidate_operator"].astype(str).tolist() if x and x != "nan"])
    # Add known common operators to stabilize label list.
    op_set |= set(OPERATOR_TEMPLATES.keys())
    operators = sorted([x for x in op_set if x and x != "NA" and x != "nan"])
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if len(operators) > len(letters):
        raise ValueError(f"Too many operators for A-Z labels: {len(operators)}")
    operator_to_letter = {op: letters[i] for i, op in enumerate(operators)}

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable; using CPU.")
        device = "cpu"
    dtype = torch.float16 if args.dtype == "float16" and device == "cuda" else torch.float32

    print(f"Loading tokenizer/model: {args.model_path}")
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

    letter_to_token_id = {letter: first_token_id_for_label(tokenizer, letter) for letter in operator_to_letter.values()}

    # Resume.
    out_path = Path(args.out_csv)
    results = []
    done_ids = set()
    if args.resume and out_path.exists():
        old = pd.read_csv(out_path)
        if "_sem6l5b_row_id" in old.columns:
            done_ids = set(pd.to_numeric(old["_sem6l5b_row_id"], errors="coerce").dropna().astype(int).tolist())
            results = old.to_dict("records")
            print(f"[resume] loaded {len(done_ids)} rows")

    if "_sem6l5b_row_id" not in plan.columns:
        plan["_sem6l5b_row_id"] = np.arange(len(plan), dtype=int)

    base_cache = {}
    rows = plan.to_dict("records")
    print(f"Running SEM-6L.5B live rows={len(rows)}")

    for idx, row in enumerate(rows):
        row_id = int(row["_sem6l5b_row_id"])
        if row_id in done_ids:
            continue

        concept = str(row["target_concept"])
        target_operator = str(row["target_operator"])
        correct_letter = operator_to_letter[target_operator]
        op_id = str(row["canonical_operator_id"])
        condition = str(row["replay_condition"])
        wrong_id = str(row.get("wrong_operator_id", ""))

        # Baseline no-policy margin for the same task.
        base_key = (concept, target_operator)
        if base_key not in base_cache:
            base_prompt = build_classifier_prompt(concept, target_operator, operator_to_letter, "")
            base_scores = score_next_letter(model, tokenizer, base_prompt, letter_to_token_id, device, args.max_len)
            base_margin, base_rank = margin_and_rank(base_scores, correct_letter)
            base_cache[base_key] = (base_margin, base_rank, base_scores)
        else:
            base_margin, base_rank, base_scores = base_cache[base_key]

        route_policy = policy_text(op_id, cards, condition, wrong_id)
        prompt = build_classifier_prompt(concept, target_operator, operator_to_letter, route_policy)
        scores = score_next_letter(model, tokenizer, prompt, letter_to_token_id, device, args.max_len)
        margin, rank = margin_and_rank(scores, correct_letter)

        out = dict(row)
        out["correct_letter"] = correct_letter
        out["base_margin_no_policy"] = base_margin
        out["base_rank_no_policy"] = base_rank
        out["live_margin"] = margin
        out["live_rank"] = rank
        out["live_margin_gain"] = margin - base_margin
        out["live_rank_improvement"] = base_rank - rank
        out["correct_logp"] = scores[correct_letter]
        out["best_logp"] = max(scores.values())
        out["operator_option_count"] = len(operators)
        results.append(out)

        if (len(results) % args.flush_every == 0) or (idx == len(rows) - 1):
            pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"[progress] {idx+1}/{len(rows)} results={len(results)}")

        if idx % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    res = pd.DataFrame(results)
    res.to_csv(out_path, index=False, encoding="utf-8-sig")
    evaluate_results(res, args.out_dir)


def evaluate_results(df: pd.DataFrame, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        raise ValueError("No results to evaluate.")

    metric = "live_margin_gain"

    # Summary by operator/condition.
    summary_rows = []
    for keys, sub in df.groupby(["canonical_operator_id", "replay_condition"], dropna=False):
        op_id, cond = keys
        vals = pd.to_numeric(sub[metric], errors="coerce").dropna().to_numpy(float)
        ranks = pd.to_numeric(sub["live_rank_improvement"], errors="coerce").dropna().to_numpy(float)
        if len(vals) == 0:
            continue
        summary_rows.append({
            "canonical_operator_id": op_id,
            "replay_condition": cond,
            "n": int(len(vals)),
            "mean_margin_gain": float(np.mean(vals)),
            "median_margin_gain": float(np.median(vals)),
            "positive_rate": float(np.mean(vals > 0)),
            "mean_rank_improvement": float(np.mean(ranks)) if len(ranks) else np.nan,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "sem6l5b_operator_summary.csv", index=False, encoding="utf-8-sig")

    # Pairwise comparisons root-matched.
    comp_rows = []
    for op_id, sub in df.groupby("canonical_operator_id", dropna=False):
        canon = sub[sub["replay_condition"] == "canonical_operator"].copy()
        if canon.empty:
            continue
        canon = canon[["replay_root_id", metric]].rename(columns={metric: "canon"})
        for ctrl_name in ["no_operator", "wrong_operator", "random_operator"]:
            ctrl = sub[sub["replay_condition"] == ctrl_name].copy()
            if ctrl.empty:
                continue
            ctrl = ctrl.groupby("replay_root_id", as_index=False)[metric].mean().rename(columns={metric: "control"})
            m = canon.merge(ctrl, on="replay_root_id", how="inner")
            diff = (m["canon"] - m["control"]).dropna().to_numpy(float)
            z = np.nan
            if len(diff) > 1 and np.std(diff, ddof=1) > 1e-12:
                z = float(np.mean(diff) / (np.std(diff, ddof=1) / math.sqrt(len(diff))))
            comp_rows.append({
                "canonical_operator_id": op_id,
                "comparison": f"canonical>{ctrl_name}",
                "n": int(len(diff)),
                "mean_canonical": float(m["canon"].mean()) if len(m) else np.nan,
                "mean_control": float(m["control"].mean()) if len(m) else np.nan,
                "mean_diff": float(np.mean(diff)) if len(diff) else np.nan,
                "z": z,
                "win_rate": float(np.mean(diff > 0)) if len(diff) else np.nan,
            })
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(out_dir / "sem6l5b_operator_vs_controls.csv", index=False, encoding="utf-8-sig")

    # Eligibility.
    eligible = []
    pending = []
    for op_id in sorted(df["canonical_operator_id"].astype(str).unique()):
        sub_comp = comp[comp["canonical_operator_id"] == op_id]
        checks = []
        for ctrl in ["no_operator", "wrong_operator", "random_operator"]:
            row = sub_comp[sub_comp["comparison"] == f"canonical>{ctrl}"]
            if len(row) == 0:
                checks.append(False)
                continue
            r = row.iloc[0]
            checks.append(pd.notna(r["mean_diff"]) and r["mean_diff"] > 0 and pd.notna(r["win_rate"]) and r["win_rate"] >= 0.50)
        op_summary = summary[
            (summary["canonical_operator_id"] == op_id) &
            (summary["replay_condition"] == "canonical_operator")
        ]
        mean_gain = float(op_summary.iloc[0]["mean_margin_gain"]) if len(op_summary) else np.nan
        positive_rate = float(op_summary.iloc[0]["positive_rate"]) if len(op_summary) else np.nan
        ok = all(checks) and pd.notna(mean_gain) and mean_gain > 0 and pd.notna(positive_rate) and positive_rate >= 0.50
        card = {
            "operator_id": op_id,
            "mean_gain": mean_gain,
            "positive_rate": positive_rate,
            "control_checks": checks,
            "comparisons": sub_comp.to_dict("records"),
            "status": "eligible_for_operator_memory" if ok else "remain_draft_pending",
        }
        if ok:
            eligible.append(card)
        else:
            pending.append(card)

    if len(eligible) >= 2:
        verdict_str = "PASS_STRONG_TRUE_LIVE_OPERATOR_MEMORY"
    elif len(eligible) == 1:
        verdict_str = "PASS_LITE_ONE_TRUE_LIVE_OPERATOR"
    else:
        verdict_str = "FAIL_OR_INCONCLUSIVE"

    verdict = {
        "stage": "SEM-6L.5B",
        "mode": "true_canonical_operator_live_replay",
        "verdict": verdict_str,
        "n_operator_memory_eligible": len(eligible),
        "n_pending": len(pending),
        "eligible_operator_ids": [x["operator_id"] for x in eligible],
        "notes": [
            "This is a true model-forward replay using prompt-level OperatorRoutingMap policies.",
            "It tests canonical/no/wrong/random conditions directly on next-token operator-choice margins.",
            "If PASS, eligible operators can enter W'_cognitive / OperatorMemory as routing policies, not hidden vector writes.",
        ],
        "outputs": {
            "true_live_results": str(out_dir / "sem6l5b_true_live_results.csv"),
            "operator_summary": str(out_dir / "sem6l5b_operator_summary.csv"),
            "operator_vs_controls": str(out_dir / "sem6l5b_operator_vs_controls.csv"),
            "verdict": str(out_dir / "sem6l5b_verdict.json"),
        },
        "eligible": eligible,
        "pending": pending,
    }
    with open(out_dir / "sem6l5b_verdict.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(verdict), f, ensure_ascii=False, indent=2)

    print("\n========== SEM-6L.5B EVAL ==========")
    print(json.dumps(json_safe({k: v for k, v in verdict.items() if k not in {"eligible", "pending"}}), ensure_ascii=False, indent=2))
    print("\nOperator summary:")
    print(summary.to_string(index=False))
    print("\nOperator vs controls:")
    print(comp.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--plan", default=DEFAULT_PLAN)
    ap.add_argument("--library", default=DEFAULT_LIBRARY)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--out_csv", default=DEFAULT_OUT_CSV)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--local_files_only", action="store_true", default=True)
    ap.add_argument("--trust_remote_code", action="store_true", default=True)
    ap.add_argument("--flush_every", type=int, default=20)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--evaluate_only", action="store_true", default=False)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.evaluate_only:
        res = pd.read_csv(args.out_csv)
        evaluate_results(res, args.out_dir)
        return

    set_seed(SEED)
    run_live(args)


if __name__ == "__main__":
    main()
