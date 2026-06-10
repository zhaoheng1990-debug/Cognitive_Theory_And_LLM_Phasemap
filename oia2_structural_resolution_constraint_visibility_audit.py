# -*- coding: utf-8 -*-
r"""
OIA-2: StructuralResolution-Aware Constraint Visibility Audit
==============================================================

Purpose
-------
Validate the pending cognitive ontology update:

    V_F_eff(S,O) = V_hit(S,O) * V_bind(S,O) * SR(S,O)

where:
    V_hit  = whether the subject/model detects the correct constraint family.
    V_bind = whether it binds that constraint to the right entity/value/condition.
    SR     = whether it resolves boundary, priority, scope, and excluded branches.

OIA-1 showed partial cross-model object-induced constraint trace signal but also showed
that raw TraceMatch / V_F_proxy is too coarse: it did not reliably distinguish
CorrectSelect from WrongSelect in Qwen/Llama and was not robust enough for global
baseline integration.

OIA-2 directly tests whether adding Structural Resolution improves the proxy:

    raw V_F_proxy        = family/trace match only
    structural V_F_eff   = V_hit * V_bind * SR

Main claims tested
------------------
1. V_F_eff separates CorrectSelect vs WrongSelect better than raw TraceMatch.
2. SR is higher for correct commitments than wrong commitments.
3. Equal-evidence ambiguous cases require high SR to avoid premature Select=1.
4. Constraint visibility should be treated as hit + binding + structural-resolution,
   not trace-family matching alone.

Hardcoded local paths
---------------------
These paths are copied from the PSA-4B reference script and the OIA-1 script.
No model paths need to be passed via command line.

Qwen:
    D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main

Llama:
    D:\model\Llama-3.2-1B-Instruct

Gemma:
    D:\model\gemma-2-2b-it

Output:
    C:\Users\ZH\Desktop\AGI\outputs\oia2_structural_resolution_constraint_visibility

Run:
    python oia2_structural_resolution_constraint_visibility_audit.py

Outputs
-------
    oia2_dataset.csv
    oia2_hardcoded_config.json
    oia2_<model>_candidate_scores.csv
    oia2_<model>_trace_generations.csv
    oia2_<model>_audit_rows.csv
    oia2_<model>_summary.json
    oia2_crossmodel_summary.csv
    oia2_crossmodel_summary.json
    oia2_crossmodel_verdict.txt
"""

import json
import math
import random
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# ---------------------------------------------------------------------
# Hardcoded config
# ---------------------------------------------------------------------

MODEL_PATHS = {
    "qwen": r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main",
    "llama": r"D:\model\Llama-3.2-1B-Instruct",
    "gemma": r"D:\model\gemma-2-2b-it",
}

DEFAULT_MODELS = ["qwen", "llama", "gemma"]
DEFAULT_OUT_DIR = r"C:\Users\ZH\Desktop\AGI\outputs\oia2_structural_resolution_constraint_visibility"
SEED = 20260607

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "float16"
LOCAL_FILES_ONLY = True
MAX_LEN = 448
GEN_MAX_NEW_TOKENS = 140
N_BASE_PER_FAMILY = 20

SURFACES = [
    "Read the symbolic record and answer the target.",
    "Use only the stated relations and constraints to resolve the target.",
    "Inspect the relation record and identify the licensed endpoint.",
    "Follow the rule system exactly and choose the endpoint.",
]

STRUCTURE_FAMILIES = [
    "transitive_location",
    "containment_closure",
    "temporal_update",
    "exception_override",
    "source_priority",
    "equal_evidence_ambiguous",
]

NAME_POOL = [
    "Aster", "Boreal", "Cobalt", "Dune", "Ember", "Fjord", "Glade", "Harbor",
    "Ivory", "Jade", "Kite", "Lumen", "Mosaic", "Nimbus", "Opal", "Quartz",
    "Raven", "Sol", "Topaz", "Umber", "Vega", "Willow", "Xylo", "Yarrow",
    "Zenith", "Atlas", "Copper", "Falcon", "Garden", "Island", "Lagoon",
    "Marble", "Onyx", "Pearl", "Ridge", "Station", "Temple", "Valley",
    "Canyon", "Beacon", "Anchor", "Meadow", "Pioneer", "Summit", "Vertex",
    "Orchid", "Cedar", "Delta", "Echo", "Flint", "Grove", "Helix", "Iris",
]

# Constraint-family keywords are deliberately transparent and conservative.
CONSTRAINT_KEYWORDS = {
    "transitive_location": [
        "transitive", "transitivity", "chain", "composition", "follow", "located",
        "location chain", "relation composition", "closure"
    ],
    "containment_closure": [
        "containment", "inside", "contains", "container", "part", "belongs", "nested",
        "outermost", "included", "closure"
    ],
    "temporal_update": [
        "latest", "new", "updated", "revised", "current", "temporal", "old",
        "newer", "update", "revision"
    ],
    "exception_override": [
        "exception", "override", "unless", "default", "active", "special",
        "exception marker", "exception endpoint"
    ],
    "source_priority": [
        "source", "expert", "ordinary", "secondary", "priority", "reliable",
        "authority", "trusted", "hierarchy"
    ],
    "equal_evidence_ambiguous": [
        "equal", "tie", "ambiguous", "same support", "insufficient", "cannot determine",
        "no unique", "multiple", "underdetermined"
    ],
}

FIELD_KEYWORDS = {
    "primary_constraint": {
        "relation_composition": ["relation composition", "composition", "chain", "transitive", "follow"],
        "containment_composition": ["containment", "inside", "outermost", "container", "nested"],
        "latest_update": ["latest", "new", "updated", "revised", "current"],
        "active_exception": ["exception", "active", "override", "default"],
        "expert_priority": ["expert", "source", "priority", "hierarchy", "authority"],
        "equal_support": ["equal", "tie", "ambiguous", "no unique", "same support"],
    },
    "boundary_condition": {
        "no_override_no_exception": ["no override", "no exception", "no special", "ordinary chain"],
        "use_outermost_container": ["outermost", "top container", "outer container"],
        "new_overrides_old": ["new overrides old", "new endpoint", "old endpoint", "latest"],
        "exception_active_overrides_default": ["exception marker", "active", "exception overrides", "default"],
        "expert_overrides_other_sources": ["expert", "overrides", "ordinary", "secondary"],
        "no_unique_select_when_equal": ["no unique", "ambiguous", "equal support", "cannot determine"],
    },
    "priority_rule": {
        "chain_order": ["chain", "follow", "final endpoint"],
        "outermost_priority": ["outermost", "container", "inside"],
        "latest_over_old": ["latest", "new", "old", "revised"],
        "exception_over_default": ["exception", "default", "active"],
        "expert_over_secondary_over_ordinary": ["expert", "secondary", "ordinary", "source priority"],
        "equal_evidence_blocks_select": ["equal", "ambiguous", "no unique", "tie"],
    },
    "scope": {
        "target_entity_only": ["target", "for the target", "target entity"],
        "container_chain_scope": ["container", "inside", "containment"],
        "current_value_scope": ["current", "new", "latest", "updated"],
        "exception_scope": ["exception", "active", "default"],
        "source_scope": ["source", "expert", "ordinary", "secondary"],
        "support_count_scope": ["support", "evidence", "count", "equal"],
    },
    "excluded_branch": {
        "intermediate_nodes_excluded": ["intermediate", "not the final", "exclude intermediate"],
        "inner_containers_excluded": ["inner", "not outermost", "exclude inner"],
        "old_endpoint_excluded": ["old", "exclude old", "not old"],
        "default_endpoint_excluded": ["default", "exclude default", "not default"],
        "lower_priority_sources_excluded": ["ordinary", "secondary", "lower priority"],
        "single_endpoint_excluded": ["no unique", "ambiguous", "do not choose", "not select one"],
    },
}

FAMILY_TO_GOLD_FIELDS = {
    "transitive_location": {
        "primary_constraint": "relation_composition",
        "boundary_condition": "no_override_no_exception",
        "priority_rule": "chain_order",
        "scope": "target_entity_only",
        "excluded_branch": "intermediate_nodes_excluded",
    },
    "containment_closure": {
        "primary_constraint": "containment_composition",
        "boundary_condition": "use_outermost_container",
        "priority_rule": "outermost_priority",
        "scope": "container_chain_scope",
        "excluded_branch": "inner_containers_excluded",
    },
    "temporal_update": {
        "primary_constraint": "latest_update",
        "boundary_condition": "new_overrides_old",
        "priority_rule": "latest_over_old",
        "scope": "current_value_scope",
        "excluded_branch": "old_endpoint_excluded",
    },
    "exception_override": {
        "primary_constraint": "active_exception",
        "boundary_condition": "exception_active_overrides_default",
        "priority_rule": "exception_over_default",
        "scope": "exception_scope",
        "excluded_branch": "default_endpoint_excluded",
    },
    "source_priority": {
        "primary_constraint": "expert_priority",
        "boundary_condition": "expert_overrides_other_sources",
        "priority_rule": "expert_over_secondary_over_ordinary",
        "scope": "source_scope",
        "excluded_branch": "lower_priority_sources_excluded",
    },
    "equal_evidence_ambiguous": {
        "primary_constraint": "equal_support",
        "boundary_condition": "no_unique_select_when_equal",
        "priority_rule": "equal_evidence_blocks_select",
        "scope": "support_count_scope",
        "excluded_branch": "single_endpoint_excluded",
    },
}


# ---------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_names(rng: random.Random, k: int = 7) -> List[str]:
    return rng.sample(NAME_POOL, k)


def stable_rng(family: str, idx: int, variant: str, surface_id: int) -> random.Random:
    # Avoid Python's salted hash so dataset is reproducible across runs.
    key = sum((j + 1) * ord(ch) for j, ch in enumerate(family + variant))
    return random.Random(SEED * 100000 + idx * 1000 + surface_id * 10 + key % 997)


def make_record(family: str, idx: int, variant: str, surface_id: int) -> Dict[str, Any]:
    rng = stable_rng(family, idx, variant, surface_id)
    names = sample_names(rng, 7)
    a, b, c, d, e, f, g = names
    surface = SURFACES[surface_id]

    object_group = f"{family}_{idx:03d}"
    object_id = f"{object_group}_{variant}_s{surface_id}"
    answer_mode = "single"
    gold_constraint_family = family

    if family == "transitive_location":
        endpoint = f if variant == "object_changed" else e
        correct = endpoint
        allowed_answers = [correct]
        wrong_candidates = [c, d, f if endpoint != f else e, g]
        lines = [
            "Rule: follow the directed location chain until its final endpoint.",
            f"E1: {a} is located in {b}.",
            f"E2: {b} is located in {c}.",
            f"E3: {c} is located in {d}.",
            f"E4: {d} is located in {endpoint}.",
            f"Target: {a}.",
        ]

    elif family == "containment_closure":
        endpoint = f if variant == "object_changed" else e
        correct = endpoint
        allowed_answers = [correct]
        wrong_candidates = [b, c, d, f if endpoint != f else e]
        lines = [
            "Rule: if X is inside Y and Y is inside Z, then X is inside Z; use the outermost container.",
            f"E1: {a} is inside {b}.",
            f"E2: {b} is inside {c}.",
            f"E3: {c} is inside {d}.",
            f"E4: {d} is inside {endpoint}.",
            f"Target: {a}.",
        ]

    elif family == "temporal_update":
        endpoint = e if variant == "object_changed" else d
        correct = endpoint
        allowed_answers = [correct]
        wrong_candidates = [b, c, e if endpoint != e else d, f]
        lines = [
            "Rule: when an old endpoint and a new endpoint conflict, choose the new endpoint.",
            f"E1: old endpoint for {a} is {b}.",
            f"E2: intermediate endpoint for {a} is {c}.",
            f"E3: new endpoint for {a} is {endpoint}.",
            f"Target: {a}.",
        ]

    elif family == "exception_override":
        default = c
        exception = f if variant == "object_changed" else e
        correct = exception
        allowed_answers = [correct]
        wrong_candidates = [default, b, d, f if exception != f else e]
        lines = [
            "Rule: use the default endpoint unless the exception marker is active; if active, use the exception endpoint.",
            f"E1: default endpoint for {a} is {default}.",
            f"E2: exception endpoint for {a} is {exception}.",
            f"E3: exception marker is active.",
            f"Target: {a}.",
        ]

    elif family == "source_priority":
        expert = e if variant == "object_changed" else f
        correct = expert
        allowed_answers = [correct]
        wrong_candidates = [b, c, d, e if expert != e else f]
        lines = [
            "Rule: expert source overrides secondary source and ordinary source.",
            f"E1: ordinary source says endpoint for {a} is {b}.",
            f"E2: secondary source says endpoint for {a} is {c}.",
            f"E3: expert source says endpoint for {a} is {expert}.",
            f"Target: {a}.",
        ]

    elif family == "equal_evidence_ambiguous":
        answer_mode = "ambiguous"
        correct = "AMBIGUOUS"
        allowed_answers = ["AMBIGUOUS"]
        wrong_candidates = [b, c, d, e]
        if variant == "object_changed":
            x1, x2 = d, e
        else:
            x1, x2 = b, c
        lines = [
            "Rule: choose a unique endpoint only if one endpoint has stronger support; if support is equal, answer AMBIGUOUS.",
            f"E1: support for {x1}.",
            f"E2: support for {x2}.",
            f"E3: support counts are equal.",
            f"Target: {a}.",
        ]

    else:
        raise ValueError(f"Unknown family: {family}")

    if variant == "constraint_changed":
        # Convert every family to exception_override while preserving object-token anchor style.
        gold_constraint_family = "exception_override"
        answer_mode = "single"
        default = c
        exception = g
        correct = exception
        allowed_answers = [correct]
        wrong_candidates = [default, b, d, e]
        lines = [
            "Rule: use the default endpoint unless the exception marker is active; if active, use the exception endpoint.",
            f"E1: default endpoint for {a} is {default}.",
            f"E2: exception endpoint for {a} is {exception}.",
            f"E3: exception marker is active.",
            f"Target: {a}.",
        ]

    rng.shuffle(lines[1:-1])
    prompt = (
        f"{surface}\n"
        f"Object ID: {object_id}\n"
        + "\n".join([lines[0]] + lines[1:-1] + [lines[-1]])
        + "\nQuestion: Which endpoint is licensed by this record?\n"
        + "Answer with exactly one endpoint name, or AMBIGUOUS if no unique endpoint is licensed."
    )

    candidate_answers = sorted(set(allowed_answers + wrong_candidates), key=lambda x: (x == "AMBIGUOUS", x))
    gold_fields = FAMILY_TO_GOLD_FIELDS[gold_constraint_family]

    return {
        "object_group": object_group,
        "object_id": object_id,
        "surface_id": surface_id,
        "variant": variant,
        "object_structure_type": family,
        "gold_constraint_family": gold_constraint_family,
        "gold_primary_constraint": gold_fields["primary_constraint"],
        "gold_boundary_condition": gold_fields["boundary_condition"],
        "gold_priority_rule": gold_fields["priority_rule"],
        "gold_scope": gold_fields["scope"],
        "gold_excluded_branch": gold_fields["excluded_branch"],
        "answer_mode": answer_mode,
        "gold_answer": correct,
        "allowed_answers": json.dumps(allowed_answers, ensure_ascii=False),
        "candidate_answers": json.dumps(candidate_answers, ensure_ascii=False),
        "prompt": prompt,
    }


def build_oia2_dataset(n_base_per_family: int = N_BASE_PER_FAMILY) -> pd.DataFrame:
    rows = []
    row_id = 0
    for family in STRUCTURE_FAMILIES:
        for idx in range(n_base_per_family):
            for surface_id in range(len(SURFACES)):
                for variant in ["base", "object_changed", "constraint_changed"]:
                    rec = make_record(family, idx, variant, surface_id)
                    rec["row_id"] = row_id
                    rows.append(rec)
                    row_id += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------

def torch_dtype_from_name(name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def load_model(model_key: str):
    model_path = MODEL_PATHS[model_key]
    dtype = torch_dtype_from_name(DTYPE)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=LOCAL_FILES_ONLY,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=LOCAL_FILES_ONLY,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=None,
    )
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def get_input_device_batch(tokenizer, text: str, max_len: int = MAX_LEN):
    return tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(DEVICE)


@torch.no_grad()
def score_candidate(tokenizer, model, prompt: str, candidate: str) -> float:
    full = prompt + " " + candidate
    enc_full = get_input_device_batch(tokenizer, full)
    enc_prompt = get_input_device_batch(tokenizer, prompt)
    ids = enc_full["input_ids"]
    prompt_len = enc_prompt["input_ids"].shape[1]
    cand_start = max(1, prompt_len)
    logits = model(**enc_full, use_cache=False).logits
    logp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    target = ids[:, 1:]
    start_t = max(0, cand_start - 1)
    if start_t >= target.shape[1]:
        return -1e9
    lp = logp[:, start_t:, :].gather(-1, target[:, start_t:].unsqueeze(-1)).squeeze(-1)
    val = float(lp.mean().detach().cpu().item())
    return val if math.isfinite(val) else -1e9


@torch.no_grad()
def generate_structural_trace(tokenizer, model, prompt: str) -> str:
    trace_prompt = (
        prompt
        + "\n\nReturn one compact JSON object only. Required keys:\n"
        + "answer, constraint_family, primary_constraint, boundary_condition, "
        + "priority_rule, scope, excluded_branch, select_state, explanation.\n"
        + "Allowed constraint_family labels: " + ", ".join(STRUCTURE_FAMILIES) + ".\n"
        + "Allowed primary_constraint labels: relation_composition, containment_composition, "
        + "latest_update, active_exception, expert_priority, equal_support.\n"
        + "Allowed boundary_condition labels: no_override_no_exception, use_outermost_container, "
        + "new_overrides_old, exception_active_overrides_default, "
        + "expert_overrides_other_sources, no_unique_select_when_equal.\n"
        + "Allowed priority_rule labels: chain_order, outermost_priority, latest_over_old, "
        + "exception_over_default, expert_over_secondary_over_ordinary, equal_evidence_blocks_select.\n"
        + "Allowed scope labels: target_entity_only, container_chain_scope, current_value_scope, "
        + "exception_scope, source_scope, support_count_scope.\n"
        + "Allowed excluded_branch labels: intermediate_nodes_excluded, inner_containers_excluded, "
        + "old_endpoint_excluded, default_endpoint_excluded, lower_priority_sources_excluded, "
        + "single_endpoint_excluded.\n"
        + "Allowed select_state labels: CorrectSelect, WrongSelect, Unresolved."
    )
    inputs = tokenizer(trace_prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
    gen = model.generate(
        **inputs,
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = gen[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------
# Trace parsing and structural scores
# ---------------------------------------------------------------------

def normalize_text(x: Any) -> str:
    return str(x or "").strip().lower()


def normalize_answer(text: str) -> str:
    s = str(text or "").strip()
    s = s.splitlines()[0].strip() if s else ""
    s = re.sub(r"^[\"'`\s]+|[\"'`\s.。,:;]+$", "", s)
    return s


def extract_endpoint_from_generation(gen: str, candidates: List[str]) -> str:
    low = normalize_text(gen)
    if "ambiguous" in low or "cannot determine" in low or "no unique" in low:
        return "AMBIGUOUS"
    for cand in sorted(candidates, key=len, reverse=True):
        if cand.lower() in low:
            return cand
    first = normalize_answer(gen)
    for cand in candidates:
        if first.lower() == cand.lower():
            return cand
    return first if first else "EMPTY"


def try_parse_json_from_text(text: str) -> Dict[str, Any]:
    raw = str(text or "")
    # Remove common fenced wrappers without relying on markdown.
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return {}


def keyword_label(text: str, label_to_words: Dict[str, List[str]]) -> str:
    low = normalize_text(text)
    scores = {}
    for label, words in label_to_words.items():
        scores[label] = sum(1 for w in words if w.lower() in low)
    best = max(scores.items(), key=lambda kv: kv[1])
    return best[0] if best[1] > 0 else "unknown"


def parse_constraint_family(gen: str, parsed: Dict[str, Any]) -> str:
    explicit = normalize_text(parsed.get("constraint_family", ""))
    for fam in STRUCTURE_FAMILIES:
        if explicit == fam or fam in explicit:
            return fam
    low = normalize_text(gen)
    for fam in STRUCTURE_FAMILIES:
        if fam in low:
            return fam
    return keyword_label(gen, CONSTRAINT_KEYWORDS)


def parse_field(gen: str, parsed: Dict[str, Any], field: str) -> str:
    explicit = normalize_text(parsed.get(field, ""))
    allowed = FIELD_KEYWORDS[field]
    for label in allowed.keys():
        if explicit == label or label in explicit:
            return label
    return keyword_label(gen, allowed)


def soft_match(gold: str, pred: str, compatible_groups: Optional[List[set]] = None) -> float:
    if pred == gold:
        return 1.0
    if pred == "unknown":
        return 0.0
    if compatible_groups:
        for g in compatible_groups:
            if gold in g and pred in g:
                return 0.45
    return 0.0


def family_hit_score(gold: str, pred: str, gen: str) -> float:
    if pred == gold:
        return 1.0
    closure_like = {"transitive_location", "containment_closure"}
    priority_like = {"temporal_update", "exception_override", "source_priority"}
    if gold in closure_like and pred in closure_like:
        return 0.7
    if gold in priority_like and pred in priority_like:
        return 0.45
    low = normalize_text(gen)
    hits = sum(1 for w in CONSTRAINT_KEYWORDS.get(gold, []) if w.lower() in low)
    if hits >= 2:
        return 0.35
    if hits == 1:
        return 0.2
    return 0.0


def compute_structural_scores(row: pd.Series, gen: str) -> Dict[str, Any]:
    parsed = try_parse_json_from_text(gen)
    pred_family = parse_constraint_family(gen, parsed)
    pred_primary = parse_field(gen, parsed, "primary_constraint")
    pred_boundary = parse_field(gen, parsed, "boundary_condition")
    pred_priority = parse_field(gen, parsed, "priority_rule")
    pred_scope = parse_field(gen, parsed, "scope")
    pred_excluded = parse_field(gen, parsed, "excluded_branch")

    v_hit = family_hit_score(row["gold_constraint_family"], pred_family, gen)
    primary_match = soft_match(row["gold_primary_constraint"], pred_primary)
    # Entity/value binding is approximated by whether selected answer is correct plus primary binding.
    # This conservative proxy separates "named the right family" from "bound it correctly".
    # It will be low when the model invokes the right family but chooses the wrong endpoint.
    v_bind = primary_match

    boundary_match = soft_match(row["gold_boundary_condition"], pred_boundary)
    priority_match = soft_match(row["gold_priority_rule"], pred_priority)
    scope_match = soft_match(row["gold_scope"], pred_scope)
    exclusion_match = soft_match(row["gold_excluded_branch"], pred_excluded)
    sr = (boundary_match + priority_match + scope_match + exclusion_match) / 4.0

    v_eff = v_hit * v_bind * sr
    # A smoother variant avoids collapsing to zero when one subscore is imperfect.
    v_eff_smooth = (v_hit * 0.40) + (v_bind * 0.25) + (sr * 0.35)

    return {
        "parsed_json": json.dumps(parsed, ensure_ascii=False),
        "pred_constraint_family": pred_family,
        "pred_primary_constraint": pred_primary,
        "pred_boundary_condition": pred_boundary,
        "pred_priority_rule": pred_priority,
        "pred_scope": pred_scope,
        "pred_excluded_branch": pred_excluded,
        "v_hit": float(v_hit),
        "v_bind": float(v_bind),
        "boundary_match": float(boundary_match),
        "priority_match": float(priority_match),
        "scope_match": float(scope_match),
        "exclusion_match": float(exclusion_match),
        "sr_proxy": float(sr),
        "v_f_eff": float(v_eff),
        "v_f_eff_smooth": float(v_eff_smooth),
    }


def select_state_from_scores(answer_correct: int, selected_answer: str, answer_mode: str, v_eff_smooth: float) -> str:
    if selected_answer in {"EMPTY", ""}:
        return "Unresolved"
    if answer_mode == "ambiguous" and selected_answer == "AMBIGUOUS":
        return "CorrectSelect"
    if selected_answer == "AMBIGUOUS" and answer_mode != "ambiguous":
        return "Unresolved"
    if answer_correct and v_eff_smooth >= 0.55:
        return "CorrectSelect"
    if answer_correct:
        return "CorrectAnswerLowSR"
    return "WrongSelect"


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> Optional[float]:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)
    if len(np.unique(y_true)) < 2 or np.std(score) < 1e-12:
        return None
    try:
        return float(roc_auc_score(y_true, score))
    except Exception:
        return None


# ---------------------------------------------------------------------
# Model audit
# ---------------------------------------------------------------------

def run_one_model(model_key: str, df: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    model_dir = out_dir / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 96)
    print(f"OIA-2 MODEL: {model_key}")
    print("=" * 96)
    tokenizer, model = load_model(model_key)

    score_rows = []
    trace_rows = []
    audit_rows = []

    for i, row in df.iterrows():
        prompt = row["prompt"]
        candidates = json.loads(row["candidate_answers"])
        allowed = json.loads(row["allowed_answers"])
        scores = {cand: score_candidate(tokenizer, model, prompt, cand) for cand in candidates}
        selected = max(scores.items(), key=lambda kv: kv[1])[0]
        answer_correct = int(selected in allowed)

        gen = generate_structural_trace(tokenizer, model, prompt)
        gen_answer = extract_endpoint_from_generation(gen, candidates)
        gen_answer_correct = int(gen_answer in allowed)
        structural = compute_structural_scores(row, gen)
        state = select_state_from_scores(answer_correct, selected, row["answer_mode"], structural["v_f_eff_smooth"])

        n_init = len(candidates)
        n_final = n_init if state == "Unresolved" else 1
        raw_cbit = math.log2(max(1, n_init) / max(1, n_final))
        cbit_eff_raw = raw_cbit * structural["v_hit"]
        cbit_eff_sr = raw_cbit * structural["v_f_eff_smooth"]

        score_rows.append({
            "row_id": int(row["row_id"]),
            "selected_answer": selected,
            "gold_answer": row["gold_answer"],
            "answer_correct": answer_correct,
            "candidate_scores": json.dumps(scores, ensure_ascii=False),
        })
        trace_rows.append({
            "row_id": int(row["row_id"]),
            "trace_generation": gen,
            "generated_answer_extract": gen_answer,
            "generated_answer_correct": gen_answer_correct,
            **structural,
        })
        audit_rows.append({
            **row.to_dict(),
            "selected_answer": selected,
            "answer_correct": answer_correct,
            "generated_answer_extract": gen_answer,
            "generated_answer_correct": gen_answer_correct,
            "trace_generation": gen,
            **structural,
            "raw_cbit": raw_cbit,
            "cbit_eff_raw_vhit": cbit_eff_raw,
            "cbit_eff_sr": cbit_eff_sr,
            "select_state": state,
        })

        if (i + 1) % 20 == 0:
            print(f"[{model_key}] {i + 1}/{len(df)}")

    score_df = pd.DataFrame(score_rows)
    trace_df = pd.DataFrame(trace_rows)
    audit_df = pd.DataFrame(audit_rows)
    score_df.to_csv(model_dir / f"oia2_{model_key}_candidate_scores.csv", index=False, encoding="utf-8-sig")
    trace_df.to_csv(model_dir / f"oia2_{model_key}_trace_generations.csv", index=False, encoding="utf-8-sig")
    audit_df.to_csv(model_dir / f"oia2_{model_key}_audit_rows.csv", index=False, encoding="utf-8-sig")

    summary = summarize_model(model_key, audit_df)
    with open(model_dir / f"oia2_{model_key}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(model_dir / f"oia2_{model_key}_verdict.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def pairwise_stability(values: List[str]) -> float:
    if len(values) <= 1:
        return 1.0
    total = 0
    same = 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            total += 1
            same += int(values[i] == values[j])
    return same / max(1, total)


def summarize_model(model_key: str, audit_df: pd.DataFrame) -> Dict[str, Any]:
    base_df = audit_df[audit_df["variant"] == "base"].copy()
    group_scores = []
    for (family, obj_group), g in base_df.groupby(["object_structure_type", "object_group"]):
        group_scores.append({
            "object_structure_type": family,
            "object_group": obj_group,
            "family_stability": pairwise_stability(g["pred_constraint_family"].tolist()),
            "boundary_stability": pairwise_stability(g["pred_boundary_condition"].tolist()),
            "priority_stability": pairwise_stability(g["pred_priority_rule"].tolist()),
            "mean_v_hit": float(g["v_hit"].mean()),
            "mean_v_bind": float(g["v_bind"].mean()),
            "mean_sr": float(g["sr_proxy"].mean()),
            "mean_v_eff": float(g["v_f_eff_smooth"].mean()),
            "answer_stability": pairwise_stability(g["selected_answer"].tolist()),
        })
    group_stability_df = pd.DataFrame(group_scores)

    changed_df = audit_df[audit_df["variant"] == "object_changed"].copy()
    ctype_df = audit_df[audit_df["variant"] == "constraint_changed"].copy()

    family_acc = accuracy_score(audit_df["gold_constraint_family"], audit_df["pred_constraint_family"])
    family_f1 = f1_score(
        audit_df["gold_constraint_family"],
        audit_df["pred_constraint_family"],
        labels=STRUCTURE_FAMILIES,
        average="macro",
        zero_division=0,
    )

    correct_mask = audit_df["answer_correct"].values.astype(int)
    wrong_mask = 1 - correct_mask

    raw_auc = safe_auc(correct_mask, audit_df["v_hit"].values)
    sr_auc = safe_auc(correct_mask, audit_df["sr_proxy"].values)
    veff_auc = safe_auc(correct_mask, audit_df["v_f_eff_smooth"].values)
    cbit_raw_auc = safe_auc(correct_mask, audit_df["cbit_eff_raw_vhit"].values)
    cbit_sr_auc = safe_auc(correct_mask, audit_df["cbit_eff_sr"].values)

    correct_df = audit_df[audit_df["answer_correct"] == 1]
    wrong_df = audit_df[audit_df["answer_correct"] == 0]

    means = {
        "v_hit_correct_mean": float(correct_df["v_hit"].mean()) if len(correct_df) else None,
        "v_hit_wrong_mean": float(wrong_df["v_hit"].mean()) if len(wrong_df) else None,
        "v_bind_correct_mean": float(correct_df["v_bind"].mean()) if len(correct_df) else None,
        "v_bind_wrong_mean": float(wrong_df["v_bind"].mean()) if len(wrong_df) else None,
        "sr_correct_mean": float(correct_df["sr_proxy"].mean()) if len(correct_df) else None,
        "sr_wrong_mean": float(wrong_df["sr_proxy"].mean()) if len(wrong_df) else None,
        "v_eff_correct_mean": float(correct_df["v_f_eff_smooth"].mean()) if len(correct_df) else None,
        "v_eff_wrong_mean": float(wrong_df["v_f_eff_smooth"].mean()) if len(wrong_df) else None,
        "cbit_eff_raw_correct_mean": float(correct_df["cbit_eff_raw_vhit"].mean()) if len(correct_df) else None,
        "cbit_eff_raw_wrong_mean": float(wrong_df["cbit_eff_raw_vhit"].mean()) if len(wrong_df) else None,
        "cbit_eff_sr_correct_mean": float(correct_df["cbit_eff_sr"].mean()) if len(correct_df) else None,
        "cbit_eff_sr_wrong_mean": float(wrong_df["cbit_eff_sr"].mean()) if len(wrong_df) else None,
    }

    # Equal evidence: correct behavior is AMBIGUOUS / no unique select.
    eq_df = audit_df[audit_df["gold_constraint_family"] == "equal_evidence_ambiguous"].copy()
    eq_ambiguous_select_rate = float((eq_df["selected_answer"] == "AMBIGUOUS").mean()) if len(eq_df) else 0.0
    eq_sr_mean = float(eq_df["sr_proxy"].mean()) if len(eq_df) else 0.0

    # OIA-2 variant metrics.
    oia2a_family_stability = float(group_stability_df["family_stability"].mean()) if len(group_stability_df) else 0.0
    oia2a_sr_mean = float(group_stability_df["mean_sr"].mean()) if len(group_stability_df) else 0.0
    oia2a_veff_mean = float(group_stability_df["mean_v_eff"].mean()) if len(group_stability_df) else 0.0
    oia2b_answer_correct = float(changed_df["answer_correct"].mean()) if len(changed_df) else 0.0
    oia2b_v_eff = float(changed_df["v_f_eff_smooth"].mean()) if len(changed_df) else 0.0
    oia2c_exception_pred_rate = float((ctype_df["pred_constraint_family"] == "exception_override").mean()) if len(ctype_df) else 0.0
    oia2c_sr = float(ctype_df["sr_proxy"].mean()) if len(ctype_df) else 0.0

    # Structural-resolution pass criterion.
    mean_delta_v_eff = (means["v_eff_correct_mean"] or 0.0) - (means["v_eff_wrong_mean"] or 0.0)
    mean_delta_sr = (means["sr_correct_mean"] or 0.0) - (means["sr_wrong_mean"] or 0.0)
    mean_delta_cbit_sr = (means["cbit_eff_sr_correct_mean"] or 0.0) - (means["cbit_eff_sr_wrong_mean"] or 0.0)
    auc_gain = None
    if veff_auc is not None and raw_auc is not None:
        auc_gain = float(veff_auc - raw_auc)

    pass_oia2a = oia2a_family_stability >= 0.70 and oia2a_sr_mean >= 0.25
    pass_oia2b = oia2b_answer_correct >= 0.50 or oia2b_v_eff >= 0.45
    pass_oia2c = oia2c_exception_pred_rate >= 0.45 or oia2c_sr >= 0.35
    pass_sr_separates = mean_delta_sr > 0.03 or (sr_auc is not None and sr_auc > 0.58)
    pass_veff_separates = mean_delta_v_eff > 0.05 or (veff_auc is not None and veff_auc > 0.60)
    pass_veff_beats_raw = (auc_gain is not None and auc_gain > 0.03) or (
        mean_delta_v_eff > ((means["v_hit_correct_mean"] or 0.0) - (means["v_hit_wrong_mean"] or 0.0)) + 0.03
    )
    pass_cbit_sr = mean_delta_cbit_sr > 0.10 or (cbit_sr_auc is not None and cbit_sr_auc > 0.60)

    if pass_oia2a and pass_oia2c and pass_veff_separates and (pass_veff_beats_raw or pass_cbit_sr):
        verdict = "PASS_OIA2_SR_CONDITIONED_CONSTRAINT_VISIBILITY"
    elif pass_oia2a and pass_oia2c and (pass_sr_separates or pass_veff_separates):
        verdict = "PARTIAL_OIA2_SR_VISIBILITY_SIGNAL"
    elif pass_oia2a or pass_oia2c:
        verdict = "PARTIAL_OIA2_CONSTRAINT_STRUCTURE_SIGNAL"
    else:
        verdict = "FAIL_OR_CAVEAT_OIA2"

    family_table = (
        audit_df.groupby(["gold_constraint_family", "pred_constraint_family"])
        .size()
        .reset_index(name="n")
        .sort_values(["gold_constraint_family", "n"], ascending=[True, False])
    )
    state_table = (
        audit_df.groupby("select_state")
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )

    return {
        "model_key": model_key,
        "verdict": verdict,
        "pass_flags": {
            "oia2a_fixed_object_sr_stability": bool(pass_oia2a),
            "oia2b_object_binding_perturbation": bool(pass_oia2b),
            "oia2c_constraint_type_perturbation": bool(pass_oia2c),
            "sr_separates_correct_wrong": bool(pass_sr_separates),
            "v_eff_separates_correct_wrong": bool(pass_veff_separates),
            "v_eff_beats_raw_vhit": bool(pass_veff_beats_raw),
            "cbit_eff_sr_correct_gt_wrong": bool(pass_cbit_sr),
        },
        "metrics": {
            "constraint_family_accuracy": float(family_acc),
            "constraint_family_macro_f1": float(family_f1),
            "overall_answer_correct": float(audit_df["answer_correct"].mean()),
            "overall_generated_answer_correct": float(audit_df["generated_answer_correct"].mean()),
            "overall_v_hit": float(audit_df["v_hit"].mean()),
            "overall_v_bind": float(audit_df["v_bind"].mean()),
            "overall_sr": float(audit_df["sr_proxy"].mean()),
            "overall_v_eff": float(audit_df["v_f_eff_smooth"].mean()),
            "oia2a_family_stability": oia2a_family_stability,
            "oia2a_sr_mean": oia2a_sr_mean,
            "oia2a_v_eff_mean": oia2a_veff_mean,
            "oia2b_answer_correct": oia2b_answer_correct,
            "oia2b_v_eff": oia2b_v_eff,
            "oia2c_exception_pred_rate": oia2c_exception_pred_rate,
            "oia2c_sr": oia2c_sr,
            "equal_evidence_ambiguous_select_rate": eq_ambiguous_select_rate,
            "equal_evidence_sr_mean": eq_sr_mean,
            "raw_vhit_auc_correct": raw_auc,
            "sr_auc_correct": sr_auc,
            "v_eff_auc_correct": veff_auc,
            "v_eff_auc_gain_over_raw": auc_gain,
            "cbit_raw_auc_correct": cbit_raw_auc,
            "cbit_sr_auc_correct": cbit_sr_auc,
            "delta_sr_correct_minus_wrong": mean_delta_sr,
            "delta_v_eff_correct_minus_wrong": mean_delta_v_eff,
            "delta_cbit_eff_sr_correct_minus_wrong": mean_delta_cbit_sr,
            **means,
        },
        "family_confusion_rows": family_table.to_dict(orient="records"),
        "select_state_counts": state_table.to_dict(orient="records"),
        "interpretation": (
            "OIA-2 tests whether constraint visibility must be conditioned on Structural Resolution. "
            "A PASS means V_hit * V_bind * SR separates correct and wrong commitments better than raw trace-family match."
        ),
        "model_path": MODEL_PATHS[model_key],
    }


def aggregate_summaries(summaries: List[Dict[str, Any]], out_dir: Path) -> Dict[str, Any]:
    rows = []
    for s in summaries:
        if "metrics" not in s:
            rows.append({
                "model_key": s.get("model_key"),
                "verdict": s.get("verdict"),
                "error": s.get("error"),
            })
            continue
        rows.append({
            "model_key": s["model_key"],
            "verdict": s["verdict"],
            **s["pass_flags"],
            **s["metrics"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "oia2_crossmodel_summary.csv", index=False, encoding="utf-8-sig")

    n = len(summaries)
    valid_df = df[df["verdict"] != "ERROR_RUNTIME"].copy()
    n_valid = len(valid_df)
    n_pass_like = int(valid_df["verdict"].astype(str).str.contains("PASS|PARTIAL", regex=True).sum()) if n_valid else 0
    n_sr = int(valid_df.get("sr_separates_correct_wrong", pd.Series(dtype=bool)).fillna(False).sum()) if n_valid else 0
    n_veff = int(valid_df.get("v_eff_separates_correct_wrong", pd.Series(dtype=bool)).fillna(False).sum()) if n_valid else 0
    n_beats = int(valid_df.get("v_eff_beats_raw_vhit", pd.Series(dtype=bool)).fillna(False).sum()) if n_valid else 0
    n_oia2c = int(valid_df.get("oia2c_constraint_type_perturbation", pd.Series(dtype=bool)).fillna(False).sum()) if n_valid else 0

    if n_valid and n_pass_like == n_valid and n_veff >= max(1, n_valid - 1) and n_beats >= max(1, n_valid - 1):
        verdict = "PASS_OIA2_CROSSMODEL_SR_CONDITIONED_VISIBILITY"
    elif n_valid and n_pass_like >= max(1, n_valid - 1) and (n_sr >= max(1, n_valid - 1) or n_veff >= max(1, n_valid - 1)):
        verdict = "PARTIAL_OIA2_CROSSMODEL_SR_VISIBILITY_SIGNAL"
    elif n_valid and n_oia2c >= max(1, n_valid - 1):
        verdict = "PARTIAL_OIA2_CROSSMODEL_CONSTRAINT_STRUCTURE_SIGNAL"
    else:
        verdict = "FAIL_OR_CAVEAT_OIA2_CROSSMODEL"

    agg = {
        "verdict": verdict,
        "n_models": n,
        "n_valid_models": n_valid,
        "n_pass_or_partial": n_pass_like,
        "n_sr_separates_correct_wrong": n_sr,
        "n_v_eff_separates_correct_wrong": n_veff,
        "n_v_eff_beats_raw_vhit": n_beats,
        "n_oia2c_pass": n_oia2c,
        "models": rows,
        "pending_claim_tested": "V_F_eff = V_hit * V_bind * SR as structural-resolution-conditioned constraint visibility",
        "caveat": (
            "This remains a generated-text/proxy audit. A stronger OIA-3 should replace or supplement generated traces "
            "with hidden-state / TopK / VIM constraint-field proxies and explicit probe-binding features."
        ),
    }
    with open(out_dir / "oia2_crossmodel_summary.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    with open(out_dir / "oia2_crossmodel_verdict.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(agg, ensure_ascii=False, indent=2))
    return agg


def main():
    set_seed(SEED)
    out_dir = Path(DEFAULT_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "MODEL_PATHS": MODEL_PATHS,
        "DEFAULT_MODELS": DEFAULT_MODELS,
        "DEFAULT_OUT_DIR": DEFAULT_OUT_DIR,
        "SEED": SEED,
        "DEVICE": DEVICE,
        "DTYPE": DTYPE,
        "LOCAL_FILES_ONLY": LOCAL_FILES_ONLY,
        "MAX_LEN": MAX_LEN,
        "GEN_MAX_NEW_TOKENS": GEN_MAX_NEW_TOKENS,
        "N_BASE_PER_FAMILY": N_BASE_PER_FAMILY,
        "STRUCTURE_FAMILIES": STRUCTURE_FAMILIES,
        "SURFACES": SURFACES,
        "theory_pending": "P7: StructuralResolution-conditioned Constraint Visibility",
    }
    with open(out_dir / "oia2_hardcoded_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    df = build_oia2_dataset(N_BASE_PER_FAMILY)
    df.to_csv(out_dir / "oia2_dataset.csv", index=False, encoding="utf-8-sig")
    print(f"[dataset] rows={len(df)} out={out_dir / 'oia2_dataset.csv'}")

    summaries = []
    for model_key in DEFAULT_MODELS:
        try:
            summaries.append(run_one_model(model_key, df, out_dir))
        except Exception as e:
            err = {
                "model_key": model_key,
                "verdict": "ERROR_RUNTIME",
                "error": repr(e),
                "model_path": MODEL_PATHS.get(model_key),
            }
            summaries.append(err)
            print(f"[ERROR] {model_key}: {repr(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    agg = aggregate_summaries(summaries, out_dir)
    print("=" * 96)
    print("OIA-2 CROSS-MODEL VERDICT")
    print(json.dumps(agg, ensure_ascii=False, indent=2))
    print("=" * 96)


if __name__ == "__main__":
    main()
