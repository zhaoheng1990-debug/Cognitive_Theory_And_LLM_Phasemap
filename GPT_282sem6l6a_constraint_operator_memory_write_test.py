# -*- coding: utf-8 -*-
r"""
SEM-6L.6A: ConstraintInsertionOperator OperatorMemory Write Test

Purpose
-------
SEM-6L.5B produced PASS-Lite:

    ConstraintInsertionOperator = eligible_for_operator_memory

This script tests the next claim:

    OperatorMemory write works when the eligible operator is stored as a reusable
    memory object and retrieved/applied on new held-out tasks.

Key shift from 6L.5B
--------------------
6L.5B:
    Does the operator card work as a one-shot prompt-level routing policy?

6L.6A:
    If the operator is written into OperatorMemory, can it be retrieved and
    applied on new tasks in the right problem class?

This is still prompt-level OperatorMemory, not hidden-state injection.

Test conditions
---------------
For each held-out task:
    1. no_memory
    2. correct_operator_memory = ConstraintInsertionOperator
    3. wrong_operator_memory = one of other canonical operators
    4. random_operator_memory

Metrics
-------
Forced-choice operator classification:
    target option = RiskAudit / ClosureCheck / PolicySelection depending on task

Score:
    margin = logp(correct_letter) - max_other_logp
    memory_gain = margin(condition) - margin(no_memory)

PASS-Lite:
    correct memory beats no_memory and wrong/random in mean margin.

PASS-Strong:
    correct memory beats all controls with win_rate >= 0.60 and z > 2,
    no major non-target damage.

Outputs
-------
sem6l6a_outputs/
    sem6l6a_heldout_tasks.csv
    sem6l6a_true_live_results.csv
    sem6l6a_memory_vs_controls.csv
    sem6l6a_task_family_summary.csv
    sem6l6a_verdict.json

Run
---
conda activate dhrf_4080s
cd C:\Users\ZH\Desktop\AGI\python_script

Smoke:
python GPT_sem6l6a_constraint_operator_memory_write_test.py --max_tasks 40

Full:
python GPT_sem6l6a_constraint_operator_memory_write_test.py

Evaluate only:
python GPT_sem6l6a_constraint_operator_memory_write_test.py --evaluate_only

Theory interpretation
---------------------
If PASS:
    ConstraintInsertionOperator can be written into W'_cognitive / OperatorMemory
    as the first validated long-term reflexive operator.

If FAIL:
    ConstraintInsertionOperator is a live replay policy but not yet a stable
    retrieved memory object; need better addressing / retrieval / boundary policy.
"""

import argparse
import gc
import json
import math
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
DEFAULT_LIBRARY = rf"{DEFAULT_ROOT}\sem6l4_outputs\sem6l4_operator_library_draft.json"
DEFAULT_L5B_VERDICT = rf"{DEFAULT_ROOT}\sem6l5b_outputs\sem6l5b_verdict.json"
DEFAULT_OUT_DIR = rf"{DEFAULT_ROOT}\sem6l6a_outputs"
DEFAULT_OUT_CSV = rf"{DEFAULT_OUT_DIR}\sem6l6a_true_live_results.csv"

SEED = 20260606

CANONICAL_OPERATOR = "ConstraintInsertionOperator"

OPERATOR_OPTIONS = [
    "Definition",
    "MechanismExplanation",
    "CausalExplanation",
    "RelationMapping",
    "PropertyDescription",
    "Comparison",
    "CounterExample",
    "InvariantSearch",
    "ClosureCheck",
    "PolicySelection",
    "Planning",
    "RiskAudit",
]

OPERATOR_TO_LABEL = {
    op: chr(ord("A") + i) for i, op in enumerate(OPERATOR_OPTIONS)
}

OPERATOR_DESCRIPTIONS = {
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

WRONG_OPERATORS = [
    "CorrectionBranchOperator",
    "TrajectoryClassExpansionOperator",
    "AddressPriorIncreaseOperator",
]


def json_safe(obj):
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


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def load_json(path: str) -> dict:
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


def first_token_id_for_label(tokenizer, letter: str) -> int:
    variants = [letter, " " + letter, "\n" + letter]
    best = None
    for v in variants:
        ids = tokenizer(v, add_special_tokens=False).input_ids
        if ids and (best is None or len(ids) < len(best)):
            best = ids
    if best is None:
        raise ValueError(f"Cannot tokenize letter {letter}")
    return int(best[0])


def load_operator_cards(library_path: str) -> Dict[str, dict]:
    lib = load_json(library_path)
    cards = {op.get("operator_id"): op for op in lib.get("operators", []) if op.get("operator_id")}
    if CANONICAL_OPERATOR not in cards:
        # Build a fallback card from 6L.5B theory if library draft is old/missing this operator.
        cards[CANONICAL_OPERATOR] = {
            "operator_id": CANONICAL_OPERATOR,
            "source_label": "add_constraint",
            "intent": "Add or strengthen an expansion constraint / boundary-check before unfolding this problem class.",
            "applicability_boundary": "Use for RiskAudit / closure-like boundary detection; avoid treating it as answer steering.",
        }
    for op in WRONG_OPERATORS:
        if op not in cards:
            cards[op] = {
                "operator_id": op,
                "intent": f"Fallback placeholder for {op}",
                "applicability_boundary": "Used only as wrong/random control.",
            }
    return cards


def build_operator_memory_entry(cards: Dict[str, dict]) -> dict:
    card = cards[CANONICAL_OPERATOR]
    return {
        "memory_id": "OperatorMemory::ConstraintInsertionOperator::v6L6A",
        "operator_id": CANONICAL_OPERATOR,
        "status": "eligible_from_6L5B",
        "source": "SEM-6L.5B",
        "intent": card.get("intent", "Insert constraint / boundary audit before unfolding."),
        "applicability_boundary": card.get("applicability_boundary", "RiskAudit / closure-like / boundary-check tasks."),
        "retrieval_cues": [
            "risk",
            "failure mode",
            "boundary condition",
            "caveat",
            "uncertainty",
            "closure",
            "consistency",
            "constraint",
            "audit",
            "safety",
            "exception",
            "override",
            "source ambiguity",
        ],
        "routing_policy": (
            "Before solving, insert a constraint-checking step. "
            "Identify boundary conditions, risk factors, uncertainty, hidden assumptions, and failure modes. "
            "If the task contains closure, exception, override, or ambiguous evidence, audit whether the relation chain remains valid. "
            "Do not answer by momentum; first decide whether a constraint or boundary check is required."
        ),
    }


def operator_memory_prefix(memory_entry: dict, condition: str, cards: Dict[str, dict], wrong_operator_id: str = "") -> str:
    if condition == "no_memory":
        return ""

    if condition == "correct_operator_memory":
        op_id = memory_entry["operator_id"]
        return (
            "Retrieved OperatorMemory:\n"
            f"MemoryID: {memory_entry['memory_id']}\n"
            f"OperatorID: {op_id}\n"
            f"Intent: {memory_entry['intent']}\n"
            f"Applicability: {memory_entry['applicability_boundary']}\n"
            f"RoutingPolicy: {memory_entry['routing_policy']}\n"
            "Use this memory only to choose the reasoning operator. Do not directly answer the task.\n"
        )

    if condition in {"wrong_operator_memory", "random_operator_memory"}:
        op_id = wrong_operator_id or random.choice(WRONG_OPERATORS)
        card = cards.get(op_id, {})
        if op_id == "CorrectionBranchOperator":
            policy = (
                "Before solving, add a grounding/correction branch: restate the definition, then use that branch before the main task."
            )
        elif op_id == "TrajectoryClassExpansionOperator":
            policy = (
                "Before solving, expand to multiple trajectory classes: mechanism, causal, relation, comparison, then choose one."
            )
        elif op_id == "AddressPriorIncreaseOperator":
            policy = (
                "Before solving, lock onto the identity/relation anchor and avoid drifting to nearby concepts."
            )
        else:
            policy = f"Apply operator {op_id}."
        return (
            "Retrieved OperatorMemory:\n"
            f"OperatorID: {op_id}\n"
            f"Intent: {card.get('intent', '')}\n"
            f"Applicability: {card.get('applicability_boundary', '')}\n"
            f"RoutingPolicy: {policy}\n"
            "Use this memory only to choose the reasoning operator. Do not directly answer the task.\n"
        )

    return ""


def build_heldout_tasks() -> pd.DataFrame:
    """
    Designed to be held-out relative to SEM-6L.5B operator-card derivation.
    We intentionally generate new concepts/surfaces/problem classes.
    """
    concepts = [
        "antibiotic resistance",
        "financial leverage",
        "autonomous vehicle deployment",
        "climate intervention",
        "cryptographic key rotation",
        "medical triage",
        "supply-chain dependency",
        "nuclear reactor cooling",
        "algorithmic hiring",
        "bridge fatigue inspection",
        "wildfire evacuation planning",
        "privacy-preserving analytics",
        "drug interaction screening",
        "air traffic rerouting",
        "pandemic travel policy",
        "data-center power failover",
        "food contamination tracing",
        "spacecraft docking",
        "water reservoir management",
        "AI agent tool use",
    ]

    task_templates = [
        {
            "family": "risk_audit",
            "target_operator": "RiskAudit",
            "template": "Audit the risks, caveats, and failure modes of {concept}.",
        },
        {
            "family": "boundary_check",
            "target_operator": "RiskAudit",
            "template": "Before recommending a decision about {concept}, identify the boundary conditions and hidden assumptions.",
        },
        {
            "family": "closure_check",
            "target_operator": "ClosureCheck",
            "template": "Check whether the reasoning chain about {concept} remains internally consistent under possible exceptions.",
        },
        {
            "family": "exception_override",
            "target_operator": "ClosureCheck",
            "template": "A general rule about {concept} appears valid, but an exception or override may apply. Which reasoning operator should be used first?",
        },
        {
            "family": "policy_selection_boundary",
            "target_operator": "PolicySelection",
            "template": "Select the safest reasoning strategy for {concept} when evidence is incomplete and consequences are high.",
        },
    ]

    non_target_templates = [
        {
            "family": "non_target_definition",
            "target_operator": "Definition",
            "template": "What is {concept}? Choose the operator needed to define it.",
        },
        {
            "family": "non_target_comparison",
            "target_operator": "Comparison",
            "template": "Compare {concept} with a related concept. Choose the required reasoning operator.",
        },
        {
            "family": "non_target_property",
            "target_operator": "PropertyDescription",
            "template": "Describe the key properties of {concept}. Choose the required reasoning operator.",
        },
        {
            "family": "non_target_planning",
            "target_operator": "Planning",
            "template": "Plan a staged approach for studying {concept}. Choose the required reasoning operator.",
        },
    ]

    rows = []
    row_id = 0
    for concept in concepts:
        for spec in task_templates + non_target_templates:
            rows.append({
                "task_id": f"6L6A_{row_id:05d}",
                "concept": concept,
                "task_family": spec["family"],
                "target_operator": spec["target_operator"],
                "request": spec["template"].format(concept=concept),
                "is_target_class": int(spec["family"] in {
                    "risk_audit", "boundary_check", "closure_check", "exception_override", "policy_selection_boundary"
                }),
            })
            row_id += 1
    return pd.DataFrame(rows)


def build_classifier_prompt(request: str, prefix: str = "") -> str:
    options = "\n".join(
        f"{OPERATOR_TO_LABEL[op]}. {op}: {OPERATOR_DESCRIPTIONS[op]}"
        for op in OPERATOR_OPTIONS
    )
    pre = prefix.strip()
    if pre:
        pre = pre + "\n\n"
    return (
        pre +
        "You are classifying which reasoning operator should be used first for the request.\n"
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
    margin = correct - max(others)
    order = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    return float(margin), int(order.index(correct_letter) + 1)


def build_replay_plan(tasks: pd.DataFrame, cards: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    for _, r in tasks.iterrows():
        for cond in ["no_memory", "correct_operator_memory", "wrong_operator_memory", "random_operator_memory"]:
            row = r.to_dict()
            row["memory_condition"] = cond
            if cond == "wrong_operator_memory":
                # deterministic hard wrong: use non-constraint operator
                row["wrong_operator_id"] = WRONG_OPERATORS[hash(row["task_id"]) % len(WRONG_OPERATORS)]
            elif cond == "random_operator_memory":
                row["wrong_operator_id"] = random.choice(WRONG_OPERATORS)
            else:
                row["wrong_operator_id"] = ""
            row["replay_root_id"] = row["task_id"]
            row["replay_id"] = f"{row['task_id']}::{cond}"
            rows.append(row)
    return pd.DataFrame(rows)


def run_live(args):
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    cards = load_operator_cards(args.library)
    memory_entry = build_operator_memory_entry(cards)

    tasks = build_heldout_tasks()
    if args.max_tasks and args.max_tasks > 0:
        tasks = tasks.head(args.max_tasks)
    tasks.to_csv(Path(args.out_dir) / "sem6l6a_heldout_tasks.csv", index=False, encoding="utf-8-sig")

    plan = build_replay_plan(tasks, cards)
    plan_path = Path(args.out_dir) / "sem6l6a_replay_plan.csv"
    plan.to_csv(plan_path, index=False, encoding="utf-8-sig")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable; using CPU.")
        device = "cpu"
    dtype = torch.float16 if args.dtype == "float16" and device == "cuda" else torch.float32

    print(f"Loading model: {args.model_path}")
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
        OPERATOR_TO_LABEL[op]: first_token_id_for_label(tokenizer, OPERATOR_TO_LABEL[op])
        for op in OPERATOR_OPTIONS
    }

    results = []
    out_path = Path(args.out_csv)
    done = set()
    if args.resume and out_path.exists():
        old = pd.read_csv(out_path)
        if "replay_id" in old.columns:
            done = set(old["replay_id"].astype(str).tolist())
            results = old.to_dict("records")
            print(f"[resume] loaded {len(done)} done rows")

    base_cache = {}
    rows = plan.to_dict("records")
    print(f"Running SEM-6L.6A live rows={len(rows)} tasks={len(tasks)}")

    for i, row in enumerate(rows):
        replay_id = str(row["replay_id"])
        if replay_id in done:
            continue

        target_operator = str(row["target_operator"])
        correct_letter = OPERATOR_TO_LABEL[target_operator]
        request = str(row["request"])
        cond = str(row["memory_condition"])
        wrong_id = str(row.get("wrong_operator_id", ""))

        # no_memory baseline for same request
        if row["task_id"] not in base_cache:
            base_prompt = build_classifier_prompt(request, "")
            base_scores = score_next_letter(model, tokenizer, base_prompt, letter_to_token_id, device, args.max_len)
            base_margin, base_rank = margin_and_rank(base_scores, correct_letter)
            base_cache[row["task_id"]] = (base_margin, base_rank)
        else:
            base_margin, base_rank = base_cache[row["task_id"]]

        prefix = operator_memory_prefix(memory_entry, cond, cards, wrong_id)
        prompt = build_classifier_prompt(request, prefix)
        scores = score_next_letter(model, tokenizer, prompt, letter_to_token_id, device, args.max_len)
        margin, rank = margin_and_rank(scores, correct_letter)

        out = dict(row)
        out["correct_letter"] = correct_letter
        out["base_margin_no_memory"] = base_margin
        out["base_rank_no_memory"] = base_rank
        out["live_margin"] = margin
        out["live_rank"] = rank
        out["live_margin_gain"] = margin - base_margin
        out["live_rank_improvement"] = base_rank - rank
        out["correct_logp"] = scores[correct_letter]
        out["best_logp"] = max(scores.values())
        results.append(out)

        if (len(results) % args.flush_every == 0) or (i == len(rows) - 1):
            pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"[progress] {i+1}/{len(rows)} results={len(results)}")

        if i % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    res = pd.DataFrame(results)
    res.to_csv(out_path, index=False, encoding="utf-8-sig")
    evaluate_results(res, args.out_dir)


def compare_conditions(df: pd.DataFrame) -> pd.DataFrame:
    metric = "live_margin_gain"
    rows = []
    canon = df[df["memory_condition"] == "correct_operator_memory"][["replay_root_id", metric]].rename(columns={metric: "canonical"})
    for ctrl in ["no_memory", "wrong_operator_memory", "random_operator_memory"]:
        c = df[df["memory_condition"] == ctrl].groupby("replay_root_id", as_index=False)[metric].mean().rename(columns={metric: "control"})
        m = canon.merge(c, on="replay_root_id", how="inner")
        diff = (m["canonical"] - m["control"]).dropna().to_numpy(float)
        z = np.nan
        if len(diff) > 1 and np.std(diff, ddof=1) > 1e-12:
            z = float(np.mean(diff) / (np.std(diff, ddof=1) / math.sqrt(len(diff))))
        rows.append({
            "comparison": f"correct_memory>{ctrl}",
            "n": int(len(diff)),
            "mean_canonical": float(m["canonical"].mean()) if len(m) else np.nan,
            "mean_control": float(m["control"].mean()) if len(m) else np.nan,
            "mean_diff": float(np.mean(diff)) if len(diff) else np.nan,
            "z": z,
            "win_rate": float(np.mean(diff > 0)) if len(diff) else np.nan,
        })
    return pd.DataFrame(rows)


def evaluate_results(df: pd.DataFrame, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for keys, sub in df.groupby(["task_family", "memory_condition"], dropna=False):
        fam, cond = keys
        vals = pd.to_numeric(sub["live_margin_gain"], errors="coerce").dropna().to_numpy(float)
        ranks = pd.to_numeric(sub["live_rank_improvement"], errors="coerce").dropna().to_numpy(float)
        if len(vals) == 0:
            continue
        summary_rows.append({
            "task_family": fam,
            "memory_condition": cond,
            "n": int(len(vals)),
            "mean_margin_gain": float(np.mean(vals)),
            "median_margin_gain": float(np.median(vals)),
            "positive_rate": float(np.mean(vals > 0)),
            "mean_rank_improvement": float(np.mean(ranks)) if len(ranks) else np.nan,
        })
    fam_summary = pd.DataFrame(summary_rows)
    fam_summary.to_csv(out_dir / "sem6l6a_task_family_summary.csv", index=False, encoding="utf-8-sig")

    overall_rows = []
    for cond, sub in df.groupby("memory_condition", dropna=False):
        vals = pd.to_numeric(sub["live_margin_gain"], errors="coerce").dropna().to_numpy(float)
        overall_rows.append({
            "memory_condition": cond,
            "n": int(len(vals)),
            "mean_margin_gain": float(np.mean(vals)),
            "median_margin_gain": float(np.median(vals)),
            "positive_rate": float(np.mean(vals > 0)),
        })
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(out_dir / "sem6l6a_operator_memory_summary.csv", index=False, encoding="utf-8-sig")

    comp = compare_conditions(df)
    comp.to_csv(out_dir / "sem6l6a_memory_vs_controls.csv", index=False, encoding="utf-8-sig")

    canon = overall[overall["memory_condition"] == "correct_operator_memory"]
    canon_mean = float(canon.iloc[0]["mean_margin_gain"]) if len(canon) else np.nan
    canon_pos = float(canon.iloc[0]["positive_rate"]) if len(canon) else np.nan

    pass_checks = []
    for ctrl in ["no_memory", "wrong_operator_memory", "random_operator_memory"]:
        r = comp[comp["comparison"] == f"correct_memory>{ctrl}"]
        if len(r):
            rr = r.iloc[0]
            pass_checks.append(
                pd.notna(rr["mean_diff"]) and rr["mean_diff"] > 0 and
                pd.notna(rr["win_rate"]) and rr["win_rate"] >= 0.50
            )
        else:
            pass_checks.append(False)

    # target vs non-target summaries
    target_df = df[df["is_target_class"] == 1]
    nontarget_df = df[df["is_target_class"] == 0]
    target_comp = compare_conditions(target_df) if not target_df.empty else pd.DataFrame()
    nontarget_comp = compare_conditions(nontarget_df) if not nontarget_df.empty else pd.DataFrame()
    target_comp.to_csv(out_dir / "sem6l6a_target_memory_vs_controls.csv", index=False, encoding="utf-8-sig")
    nontarget_comp.to_csv(out_dir / "sem6l6a_nontarget_memory_vs_controls.csv", index=False, encoding="utf-8-sig")

    # safety: correct memory should not heavily improve/worsen non-target operator choice incorrectly.
    nt_canon = nontarget_df[nontarget_df["memory_condition"] == "correct_operator_memory"]
    nt_mean = float(nt_canon["live_margin_gain"].mean()) if len(nt_canon) else np.nan
    nt_damage = bool(pd.notna(nt_mean) and nt_mean < -0.25)

    lite = bool(all(pass_checks) and pd.notna(canon_mean) and canon_mean > 0 and pd.notna(canon_pos) and canon_pos >= 0.50 and not nt_damage)
    strong = bool(lite and all(
        (len(r) and float(r.iloc[0]["z"]) > 2 and float(r.iloc[0]["win_rate"]) >= 0.60)
        for _, r in [(ctrl, comp[comp["comparison"] == f"correct_memory>{ctrl}"]) for ctrl in ["no_memory", "wrong_operator_memory", "random_operator_memory"]]
    ))

    if strong:
        verdict_str = "PASS_STRONG_OPERATORMEMORY_WRITE"
    elif lite:
        verdict_str = "PASS_LITE_OPERATORMEMORY_WRITE"
    else:
        verdict_str = "FAIL_OR_INCONCLUSIVE"

    verdict = {
        "stage": "SEM-6L.6A",
        "mode": "constraint_insertion_operator_memory_write_test",
        "verdict": verdict_str,
        "operator_id": CANONICAL_OPERATOR,
        "mean_gain_correct_memory": canon_mean,
        "positive_rate_correct_memory": canon_pos,
        "control_checks": pass_checks,
        "nontarget_mean_gain_correct_memory": nt_mean,
        "nontarget_damage_flag": nt_damage,
        "notes": [
            "6L.6A tests whether ConstraintInsertionOperator works as a retrieved OperatorMemory object.",
            "PASS means it can enter W'_cognitive / OperatorMemory as a prompt-level routing memory.",
            "This still does not test hidden-state implementation; it validates external cognitive OperatorMemory write.",
        ],
        "outputs": {
            "heldout_tasks": str(out_dir / "sem6l6a_heldout_tasks.csv"),
            "true_live_results": str(out_dir / "sem6l6a_true_live_results.csv"),
            "task_family_summary": str(out_dir / "sem6l6a_task_family_summary.csv"),
            "operator_memory_summary": str(out_dir / "sem6l6a_operator_memory_summary.csv"),
            "memory_vs_controls": str(out_dir / "sem6l6a_memory_vs_controls.csv"),
            "target_memory_vs_controls": str(out_dir / "sem6l6a_target_memory_vs_controls.csv"),
            "nontarget_memory_vs_controls": str(out_dir / "sem6l6a_nontarget_memory_vs_controls.csv"),
            "verdict": str(out_dir / "sem6l6a_verdict.json"),
        },
    }

    with open(out_dir / "sem6l6a_verdict.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(verdict), f, ensure_ascii=False, indent=2)

    print("\n========== SEM-6L.6A EVAL ==========")
    print(json.dumps(json_safe(verdict), ensure_ascii=False, indent=2))
    print("\nOverall summary:")
    print(overall.to_string(index=False))
    print("\nMemory vs controls:")
    print(comp.to_string(index=False))
    print("\nTask family summary:")
    print(fam_summary.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--library", default=DEFAULT_LIBRARY)
    ap.add_argument("--sem6l5b_verdict", default=DEFAULT_L5B_VERDICT)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--out_csv", default=DEFAULT_OUT_CSV)
    ap.add_argument("--max_tasks", type=int, default=0)
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
        df = pd.read_csv(args.out_csv)
        evaluate_results(df, args.out_dir)
        return

    # Optional sanity check: 6L.5B should include ConstraintInsertionOperator as eligible.
    p = Path(args.sem6l5b_verdict)
    if p.exists():
        v = load_json(str(p))
        elig = set(v.get("eligible_operator_ids", []))
        if CANONICAL_OPERATOR not in elig:
            print(f"[warn] {CANONICAL_OPERATOR} not listed as eligible in 6L.5B verdict. Continuing anyway.")
    else:
        print("[warn] 6L.5B verdict not found; continuing based on operator library.")

    run_live(args)


if __name__ == "__main__":
    main()
