# -*- coding: utf-8 -*-
r"""
SEM-6L.7: Reuse Boundary Gradient Audit

Purpose
-------
SEM-6L.6B showed:

    ConstraintInsertionOperator + low-complexity gate
    -> PASS-Lite ConditionalOperatorMemory

The next theoretical claim:

    Reflexive generalization is not global.
    It is boundary-conditioned reuse along a gradient:

        strict isomorphism
        -> local manifold geometry near-isomorphism
        -> dynamical context similarity
        -> surface similarity but structural mismatch
        -> unrelated tasks

If the theory is correct:

    reuse_gain should decline as boundary constraints weaken.

This script tests the gradient by creating task groups with graded similarity to the
validated source region:

    Source operator:
        ConstraintInsertionOperator

    Validated source region:
        closure_check / exception_override / risk_audit

Groups
------
G0_STRICT_SOURCE:
    Same task families as 6L.6B source region.
G1_LOCAL_GEOMETRY:
    Different surfaces, but same local geometry: consistency, exception, rule override, risk boundary.
G2_DYNAMIC_CONTEXT:
    Similar operator dynamics: ambiguity, safety, failure-mode, but weaker closure structure.
G3_SURFACE_SIM_STRUCT_DIFF:
    Surface words like "risk/boundary/check" appear, but true required operator is not ConstraintInsertion.
G4_UNRELATED:
    Definition/property/comparison/planning tasks.

Replay conditions
-----------------
1. no_memory
2. correct_memory = ConstraintInsertionOperator
3. wrong_memory
4. random_memory

Metric
------
forced-choice next-token margin of target operator letter:

    margin = logp(correct_letter) - max_other_logp

reuse_gain = margin(correct_memory) - margin(no_memory)

Expected result
---------------
ReuseGain:
    G0 > G1 > G2 > G3 >= G4

Strict monotonicity is not required in v1.
A trend is enough:

    corr(boundary_strength, reuse_gain) > 0

Outputs
-------
sem6l7_outputs/
    sem6l7_reuse_gradient_tasks.csv
    sem6l7_replay_plan.csv
    sem6l7_true_live_results.csv
    sem6l7_gradient_summary.csv
    sem6l7_operator_vs_controls.csv
    sem6l7_verdict.json

Run
---
conda activate dhrf_4080s
cd C:\Users\ZH\Desktop\AGI\python_script

Smoke:
python GPT_sem6l7_reuse_boundary_gradient_audit.py --max_tasks 50

Full:
python GPT_sem6l7_reuse_boundary_gradient_audit.py

Evaluate existing:
python GPT_sem6l7_reuse_boundary_gradient_audit.py --evaluate_only

Theory
------
If PASS:
    Reflexive operator generalization is supported as:
        local-geometry / boundary-conditioned reuse,
    not global transfer.

If FAIL:
    Current prompt-level implementation may not measure geometry reuse;
    next step should use latent/TopK geometry gates rather than task-text gates.
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
DEFAULT_OUT_DIR = rf"{DEFAULT_ROOT}\sem6l7_outputs"
DEFAULT_OUT_CSV = rf"{DEFAULT_OUT_DIR}\sem6l7_true_live_results.csv"

SEED = 20260606
CANONICAL_OPERATOR = "ConstraintInsertionOperator"

WRONG_OPERATORS = [
    "CorrectionBranchOperator",
    "TrajectoryClassExpansionOperator",
    "AddressPriorIncreaseOperator",
]

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
        return {}
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
        raise ValueError(f"Cannot tokenize option label {letter}")
    return int(best[0])


def load_operator_cards(library_path: str) -> Dict[str, dict]:
    lib = load_json(library_path)
    cards = {op.get("operator_id"): op for op in lib.get("operators", []) if op.get("operator_id")}
    cards.setdefault(CANONICAL_OPERATOR, {
        "operator_id": CANONICAL_OPERATOR,
        "intent": "Add or strengthen an expansion constraint / boundary-check before unfolding this problem class.",
        "applicability_boundary": "RiskAudit / closure-like boundary detection; avoid treating it as answer steering.",
    })
    for op in WRONG_OPERATORS:
        cards.setdefault(op, {
            "operator_id": op,
            "intent": f"Fallback wrong-control operator: {op}",
            "applicability_boundary": "Used only as wrong/random control.",
        })
    return cards


def build_constraint_memory(cards: Dict[str, dict]) -> dict:
    card = cards.get(CANONICAL_OPERATOR, {})
    return {
        "memory_id": "OperatorMemory::ConstraintInsertionOperator::6L7",
        "operator_id": CANONICAL_OPERATOR,
        "intent": card.get("intent", "Insert constraint / boundary audit before unfolding."),
        "applicability_boundary": card.get("applicability_boundary", "closure / exception / risk local geometry."),
        "routing_policy": (
            "Before selecting a reasoning operator, check whether the task contains a relation-closure issue, "
            "an exception or override, a hidden boundary condition, a failure-mode audit, or a risk/caveat assessment. "
            "If so, insert a constraint-checking step before ordinary reasoning. "
            "Do not use this memory for plain definition, property listing, ordinary planning, or comparison."
        )
    }


def memory_prefix(memory: dict, condition: str, cards: Dict[str, dict], wrong_operator_id: str = "") -> str:
    if condition == "no_memory":
        return ""

    if condition == "correct_memory":
        return (
            "Retrieved ConditionalOperatorMemory:\n"
            f"MemoryID: {memory['memory_id']}\n"
            f"OperatorID: {memory['operator_id']}\n"
            f"Intent: {memory['intent']}\n"
            f"Boundary: {memory['applicability_boundary']}\n"
            f"RoutingPolicy: {memory['routing_policy']}\n"
            "Use this memory only to select the first reasoning operator. Do not directly answer the request.\n"
        )

    op_id = wrong_operator_id or random.choice(WRONG_OPERATORS)
    card = cards.get(op_id, {})
    if op_id == "CorrectionBranchOperator":
        policy = "First add a grounding/correction branch, then continue with the requested task."
    elif op_id == "TrajectoryClassExpansionOperator":
        policy = "Expand into multiple possible trajectory classes before choosing a path."
    elif op_id == "AddressPriorIncreaseOperator":
        policy = "Lock onto identity and relation anchors before reasoning."
    else:
        policy = f"Apply {op_id}."

    return (
        "Retrieved ConditionalOperatorMemory:\n"
        f"OperatorID: {op_id}\n"
        f"Intent: {card.get('intent', '')}\n"
        f"Boundary: {card.get('applicability_boundary', '')}\n"
        f"RoutingPolicy: {policy}\n"
        "Use this memory only to select the first reasoning operator. Do not directly answer the request.\n"
    )


def build_reuse_gradient_tasks() -> pd.DataFrame:
    """
    Construct graded boundary groups.
    boundary_strength:
      4 = strict/source isomorphism
      3 = local geometry near-isomorphism
      2 = dynamical context similarity
      1 = surface-similar but structurally different
      0 = unrelated/non-target
    """
    base_concepts = [
        "autonomous vehicle deployment",
        "medical triage",
        "cryptographic key rotation",
        "supply-chain dependency",
        "nuclear reactor cooling",
        "AI agent tool use",
        "financial leverage",
        "bridge fatigue inspection",
        "data-center failover",
        "food contamination tracing",
        "pandemic travel policy",
        "water reservoir management",
    ]

    groups = []

    # G0 strict source: exactly source families from 6L.6B best gate.
    groups += [
        ("G0_STRICT_SOURCE", 4, "closure_check", "ClosureCheck",
         "Check whether the reasoning chain about {concept} remains internally consistent under possible exceptions."),
        ("G0_STRICT_SOURCE", 4, "exception_override", "ClosureCheck",
         "A general rule about {concept} appears valid, but an exception or override may apply. Which reasoning operator should be used first?"),
        ("G0_STRICT_SOURCE", 4, "risk_audit", "RiskAudit",
         "Audit the risks, caveats, and failure modes of {concept}."),
    ]

    # G1 local geometry: surface different but same closure/risk/constraint geometry.
    groups += [
        ("G1_LOCAL_GEOMETRY", 3, "rule_consistency_audit", "ClosureCheck",
         "A policy for {concept} has a standard rule and a special-case clause. Check which reasoning operator should verify consistency before proceeding."),
        ("G1_LOCAL_GEOMETRY", 3, "assumption_boundary_audit", "RiskAudit",
         "The conclusion about {concept} may depend on hidden assumptions. Choose the operator that audits constraints and boundary conditions first."),
        ("G1_LOCAL_GEOMETRY", 3, "override_validity_check", "ClosureCheck",
         "A newer instruction appears to override an older relation in {concept}. Select the operator that checks whether the closure remains valid."),
    ]

    # G2 dynamic context similarity: similar uncertainty/failure-mode dynamics, weaker closure geometry.
    groups += [
        ("G2_DYNAMIC_CONTEXT", 2, "failure_mode_strategy", "RiskAudit",
         "Before making a recommendation about {concept}, identify plausible failure modes and uncertainty sources."),
        ("G2_DYNAMIC_CONTEXT", 2, "high_stakes_strategy", "PolicySelection",
         "Select the safest reasoning strategy for {concept} when evidence is incomplete and consequences are high."),
        ("G2_DYNAMIC_CONTEXT", 2, "counterfactual_stress", "CounterExample",
         "Stress-test the proposal about {concept} by considering what could make it fail."),
    ]

    # G3 surface similar but structural mismatch: words overlap, but correct operator is not constraint insertion.
    groups += [
        ("G3_SURFACE_SIM_STRUCT_DIFF", 1, "risk_definition", "Definition",
         "Define what risk means in the context of {concept}."),
        ("G3_SURFACE_SIM_STRUCT_DIFF", 1, "boundary_property", "PropertyDescription",
         "Describe the boundary-related properties of {concept} without auditing a decision."),
        ("G3_SURFACE_SIM_STRUCT_DIFF", 1, "safety_comparison", "Comparison",
         "Compare safety considerations in {concept} with a related system."),
        ("G3_SURFACE_SIM_STRUCT_DIFF", 1, "constraint_planning", "Planning",
         "Plan a staged study of constraints in {concept}."),
    ]

    # G4 unrelated/non-target.
    groups += [
        ("G4_UNRELATED", 0, "plain_definition", "Definition",
         "What is {concept}? Choose the operator needed to define it."),
        ("G4_UNRELATED", 0, "plain_property", "PropertyDescription",
         "Describe the key properties of {concept}. Choose the required reasoning operator."),
        ("G4_UNRELATED", 0, "plain_comparison", "Comparison",
         "Compare {concept} with a related concept. Choose the required reasoning operator."),
        ("G4_UNRELATED", 0, "plain_planning", "Planning",
         "Plan a staged approach for studying {concept}. Choose the required reasoning operator."),
    ]

    rows = []
    idx = 0
    for concept in base_concepts:
        for group, strength, family, target_op, template in groups:
            rows.append({
                "task_id": f"6L7_{idx:05d}",
                "reuse_group": group,
                "boundary_strength": strength,
                "task_family": family,
                "target_operator": target_op,
                "concept": concept,
                "request": template.format(concept=concept),
                "expected_transfer": int(strength >= 2),
            })
            idx += 1

    return pd.DataFrame(rows)


def build_replay_plan(tasks: pd.DataFrame, cards: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    for _, r in tasks.iterrows():
        for cond in ["no_memory", "correct_memory", "wrong_memory", "random_memory"]:
            row = r.to_dict()
            row["memory_condition"] = cond
            if cond == "wrong_memory":
                row["wrong_operator_id"] = WRONG_OPERATORS[hash(row["task_id"]) % len(WRONG_OPERATORS)]
            elif cond == "random_memory":
                row["wrong_operator_id"] = random.choice(WRONG_OPERATORS)
            else:
                row["wrong_operator_id"] = ""
            row["replay_root_id"] = row["task_id"]
            row["replay_id"] = f"{row['task_id']}::{cond}"
            rows.append(row)
    return pd.DataFrame(rows)


def build_classifier_prompt(request: str, prefix: str = "") -> str:
    options = "\n".join(
        f"{OPERATOR_TO_LABEL[op]}. {op}: {OPERATOR_DESCRIPTIONS[op]}"
        for op in OPERATOR_OPTIONS
    )
    pre = prefix.strip()
    if pre:
        pre += "\n\n"
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


def run_live(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    cards = load_operator_cards(args.library)
    memory = build_constraint_memory(cards)

    tasks = build_reuse_gradient_tasks()
    if args.max_tasks and args.max_tasks > 0:
        tasks = tasks.head(args.max_tasks)
    tasks.to_csv(out_dir / "sem6l7_reuse_gradient_tasks.csv", index=False, encoding="utf-8-sig")

    plan = build_replay_plan(tasks, cards)
    plan.to_csv(out_dir / "sem6l7_replay_plan.csv", index=False, encoding="utf-8-sig")

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

    out_path = Path(args.out_csv)
    results = []
    done = set()
    if args.resume and out_path.exists():
        old = pd.read_csv(out_path)
        if "replay_id" in old.columns:
            done = set(old["replay_id"].astype(str).tolist())
            results = old.to_dict("records")
            print(f"[resume] loaded {len(done)} done rows")

    base_cache = {}
    rows = plan.to_dict("records")
    print(f"Running SEM-6L.7 live rows={len(rows)} tasks={len(tasks)}")

    for i, row in enumerate(rows):
        replay_id = str(row["replay_id"])
        if replay_id in done:
            continue

        target_op = str(row["target_operator"])
        correct_letter = OPERATOR_TO_LABEL[target_op]
        request = str(row["request"])
        cond = str(row["memory_condition"])
        wrong_id = str(row.get("wrong_operator_id", ""))

        # Baseline no-memory for same root.
        if row["task_id"] not in base_cache:
            base_prompt = build_classifier_prompt(request, "")
            base_scores = score_next_letter(model, tokenizer, base_prompt, letter_to_token_id, device, args.max_len)
            base_margin, base_rank = margin_and_rank(base_scores, correct_letter)
            base_cache[row["task_id"]] = (base_margin, base_rank)
        else:
            base_margin, base_rank = base_cache[row["task_id"]]

        prefix = memory_prefix(memory, cond, cards, wrong_id)
        prompt = build_classifier_prompt(request, prefix)
        scores = score_next_letter(model, tokenizer, prompt, letter_to_token_id, device, args.max_len)
        margin, rank = margin_and_rank(scores, correct_letter)

        out = dict(row)
        out["correct_letter"] = correct_letter
        out["base_margin_no_memory"] = base_margin
        out["base_rank_no_memory"] = base_rank
        out["live_margin"] = margin
        out["live_rank"] = rank
        out["reuse_gain"] = margin - base_margin
        out["rank_improvement"] = base_rank - rank
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


def pivot_results(df: pd.DataFrame) -> pd.DataFrame:
    idx_cols = [
        "replay_root_id", "task_id", "reuse_group", "boundary_strength",
        "task_family", "target_operator", "concept", "request", "expected_transfer"
    ]
    idx_cols = [c for c in idx_cols if c in df.columns]
    piv = df.pivot_table(
        index="replay_root_id",
        columns="memory_condition",
        values="reuse_gain",
        aggfunc="mean"
    ).reset_index()
    meta = df.drop_duplicates("replay_root_id")[idx_cols]
    out = meta.merge(piv, on="replay_root_id", how="left")
    for c in ["no_memory", "correct_memory", "wrong_memory", "random_memory"]:
        if c not in out.columns:
            out[c] = np.nan
    out["correct_vs_no"] = out["correct_memory"] - out["no_memory"]
    out["correct_vs_wrong"] = out["correct_memory"] - out["wrong_memory"]
    out["correct_vs_random"] = out["correct_memory"] - out["random_memory"]
    return out


def group_summary(piv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, sub in piv.groupby("reuse_group", dropna=False):
        strength = float(sub["boundary_strength"].iloc[0]) if len(sub) else np.nan
        vals = pd.to_numeric(sub["correct_memory"], errors="coerce").dropna().to_numpy(float)
        d_no = pd.to_numeric(sub["correct_vs_no"], errors="coerce").dropna().to_numpy(float)
        d_wrong = pd.to_numeric(sub["correct_vs_wrong"], errors="coerce").dropna().to_numpy(float)
        d_random = pd.to_numeric(sub["correct_vs_random"], errors="coerce").dropna().to_numpy(float)
        rows.append({
            "reuse_group": group,
            "boundary_strength": strength,
            "n": int(len(sub)),
            "mean_reuse_gain": float(np.mean(vals)) if len(vals) else np.nan,
            "median_reuse_gain": float(np.median(vals)) if len(vals) else np.nan,
            "positive_rate": float(np.mean(vals > 0)) if len(vals) else np.nan,
            "mean_diff_vs_no": float(np.mean(d_no)) if len(d_no) else np.nan,
            "win_rate_vs_no": float(np.mean(d_no > 0)) if len(d_no) else np.nan,
            "mean_diff_vs_wrong": float(np.mean(d_wrong)) if len(d_wrong) else np.nan,
            "win_rate_vs_wrong": float(np.mean(d_wrong > 0)) if len(d_wrong) else np.nan,
            "mean_diff_vs_random": float(np.mean(d_random)) if len(d_random) else np.nan,
            "win_rate_vs_random": float(np.mean(d_random > 0)) if len(d_random) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("boundary_strength", ascending=False)


def operator_vs_controls(piv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comp_col, name in [
        ("correct_vs_no", "correct>no_memory"),
        ("correct_vs_wrong", "correct>wrong_memory"),
        ("correct_vs_random", "correct>random_memory"),
    ]:
        vals = pd.to_numeric(piv[comp_col], errors="coerce").dropna().to_numpy(float)
        if len(vals) > 1 and np.std(vals, ddof=1) > 1e-12:
            z = float(np.mean(vals) / (np.std(vals, ddof=1) / math.sqrt(len(vals))))
        else:
            z = np.nan
        rows.append({
            "comparison": name,
            "n": int(len(vals)),
            "mean_diff": float(np.mean(vals)) if len(vals) else np.nan,
            "median_diff": float(np.median(vals)) if len(vals) else np.nan,
            "win_rate": float(np.mean(vals > 0)) if len(vals) else np.nan,
            "z": z,
        })
    return pd.DataFrame(rows)


def evaluate_results(df: pd.DataFrame, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    piv = pivot_results(df)
    summary = group_summary(piv)
    comp = operator_vs_controls(piv)

    piv.to_csv(out_dir / "sem6l7_pivot_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "sem6l7_gradient_summary.csv", index=False, encoding="utf-8-sig")
    comp.to_csv(out_dir / "sem6l7_operator_vs_controls.csv", index=False, encoding="utf-8-sig")

    # Trend: boundary strength vs mean reuse gain by task and group.
    valid = piv.dropna(subset=["boundary_strength", "correct_memory"]).copy()
    if len(valid) > 2 and valid["boundary_strength"].std() > 1e-12 and valid["correct_memory"].std() > 1e-12:
        corr_task = float(np.corrcoef(valid["boundary_strength"], valid["correct_memory"])[0, 1])
    else:
        corr_task = np.nan

    if len(summary) > 2 and summary["boundary_strength"].std() > 1e-12 and summary["mean_reuse_gain"].std() > 1e-12:
        corr_group = float(np.corrcoef(summary["boundary_strength"], summary["mean_reuse_gain"])[0, 1])
    else:
        corr_group = np.nan

    # Monotonic-ish check: sorted by strength, high groups should be higher than low groups.
    s = summary.sort_values("boundary_strength", ascending=False)
    high = s[s["boundary_strength"] >= 3]["mean_reuse_gain"].mean()
    mid = s[s["boundary_strength"] == 2]["mean_reuse_gain"].mean()
    low = s[s["boundary_strength"] <= 1]["mean_reuse_gain"].mean()

    monotonic_soft = bool(pd.notna(high) and pd.notna(mid) and pd.notna(low) and high > mid > low)
    high_positive = bool(pd.notna(high) and high > 0)
    low_nonpositive = bool(pd.notna(low) and low <= 0)

    pass_lite = bool(pd.notna(corr_group) and corr_group > 0.40 and high_positive and low_nonpositive)
    pass_strong = bool(pass_lite and monotonic_soft and pd.notna(corr_task) and corr_task > 0.25)

    if pass_strong:
        verdict_str = "PASS_STRONG_REUSE_BOUNDARY_GRADIENT"
    elif pass_lite:
        verdict_str = "PASS_LITE_REUSE_BOUNDARY_GRADIENT"
    else:
        verdict_str = "FAIL_OR_INCONCLUSIVE"

    verdict = {
        "stage": "SEM-6L.7",
        "mode": "reuse_boundary_gradient_audit",
        "verdict": verdict_str,
        "corr_boundary_strength_vs_task_reuse_gain": corr_task,
        "corr_boundary_strength_vs_group_mean_gain": corr_group,
        "high_strength_mean_gain": float(high) if pd.notna(high) else None,
        "mid_strength_mean_gain": float(mid) if pd.notna(mid) else None,
        "low_strength_mean_gain": float(low) if pd.notna(low) else None,
        "monotonic_soft": monotonic_soft,
        "notes": [
            "Tests whether ConstraintInsertionOperator reuse gain declines as local-geometry / boundary similarity weakens.",
            "A pass supports reflexive generalization as boundary-conditioned local reuse rather than global transfer.",
            "If fail, prompt-level policy may be insufficient; next step should use latent/TopK geometry similarity gates."
        ],
        "outputs": {
            "tasks": str(out_dir / "sem6l7_reuse_gradient_tasks.csv"),
            "replay_plan": str(out_dir / "sem6l7_replay_plan.csv"),
            "true_live_results": str(out_dir / "sem6l7_true_live_results.csv"),
            "pivot_results": str(out_dir / "sem6l7_pivot_results.csv"),
            "gradient_summary": str(out_dir / "sem6l7_gradient_summary.csv"),
            "operator_vs_controls": str(out_dir / "sem6l7_operator_vs_controls.csv"),
            "verdict": str(out_dir / "sem6l7_verdict.json"),
        }
    }

    with open(out_dir / "sem6l7_verdict.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(verdict), f, ensure_ascii=False, indent=2)

    print("\n========== SEM-6L.7 ==========")
    print(json.dumps(json_safe(verdict), ensure_ascii=False, indent=2))
    print("\nGradient summary:")
    print(summary.to_string(index=False))
    print("\nOperator vs controls:")
    print(comp.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--library", default=DEFAULT_LIBRARY)
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

    run_live(args)


if __name__ == "__main__":
    main()
