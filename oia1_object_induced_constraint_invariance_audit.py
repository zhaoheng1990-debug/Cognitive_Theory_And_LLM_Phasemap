# -*- coding: utf-8 -*-
"""
OIA-1: Object-Induced Constraint Invariance Audit
=================================================

Purpose
-------
Validate the pending cognitive ontology claim:

    Inv(O) -> Inv(F_O)

Operational version:
    Fixed object structure O + varied surface forms
        -> stable constraint trace and stable Select=1.
    Changed object binding / changed constraint type
        -> systematic change in trace, answer, or ambiguity state.

This script is intentionally self-contained and uses hardcoded local model paths
copied from the PSA-4B reference script provided in this conversation.

Core outputs
------------
    oia1_dataset.csv
    oia1_<model>_candidate_scores.csv
    oia1_<model>_trace_generations.csv
    oia1_<model>_audit_rows.csv
    oia1_<model>_summary.json
    oia1_crossmodel_summary.csv
    oia1_crossmodel_summary.json

Interpretation
--------------
OIA-1A:
    Same object_id + same object_structure_type + different surface_id.
    Measures trace and answer stability under surface perturbation.

OIA-1B:
    Same template but changed object binding.
    Measures whether the selected endpoint follows O.

OIA-1C:
    Changed constraint type.
    Measures whether predicted constraint family changes with the object
    constraint structure.

No basin terminology is used.
"""

import json
import math
import random
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# ---------------------------------------------------------------------
# Hardcoded config copied from the provided PSA-4B reference script.
# ---------------------------------------------------------------------

MODEL_PATHS = {
    "qwen": r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main",
    "llama": r"D:\model\Llama-3.2-1B-Instruct",
    "gemma": r"D:\model\gemma-2-2b-it",
}

DEFAULT_MODELS = ["qwen", "llama", "gemma"]
DEFAULT_OUT_DIR = r"C:\Users\ZH\Desktop\AGI\outputs\oia1_object_induced_constraint_invariance"
SEED = 20260607

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "float16"
LOCAL_FILES_ONLY = True
MAX_LEN = 384
GEN_MAX_NEW_TOKENS = 80
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

CONSTRAINT_KEYWORDS = {
    "transitive_location": [
        "transitive", "transitivity", "chain", "composition", "follow", "located",
        "relation composition", "closure"
    ],
    "containment_closure": [
        "containment", "inside", "contains", "part", "belongs", "closure",
        "nested", "included"
    ],
    "temporal_update": [
        "latest", "new", "updated", "revised", "current", "temporal", "old",
        "newer", "update"
    ],
    "exception_override": [
        "exception", "override", "unless", "default", "active", "special",
        "priority", "exception marker"
    ],
    "source_priority": [
        "source", "expert", "ordinary", "priority", "reliable", "authority",
        "trusted"
    ],
    "equal_evidence_ambiguous": [
        "equal", "tie", "ambiguous", "same support", "insufficient", "cannot determine",
        "multiple", "underdetermined"
    ],
}

NAME_POOL = [
    "Aster", "Boreal", "Cobalt", "Dune", "Ember", "Fjord", "Glade", "Harbor",
    "Ivory", "Jade", "Kite", "Lumen", "Mosaic", "Nimbus", "Opal", "Quartz",
    "Raven", "Sol", "Topaz", "Umber", "Vega", "Willow", "Xylo", "Yarrow",
    "Zenith", "Atlas", "Copper", "Falcon", "Garden", "Island", "Lagoon",
    "Marble", "Onyx", "Pearl", "Ridge", "Station", "Temple", "Valley",
    "Canyon", "Beacon", "Anchor", "Meadow", "Pioneer", "Summit", "Vertex",
    "Orchid", "Cedar", "Delta", "Echo", "Flint", "Grove", "Helix", "Iris",
]


# ---------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_names(rng: random.Random, k: int = 7) -> List[str]:
    return rng.sample(NAME_POOL, k)


def make_record(family: str, idx: int, variant: str, surface_id: int) -> Dict[str, Any]:
    rng = random.Random(SEED * 100000 + idx * 1000 + surface_id * 10 + hash(family + variant) % 997)
    names = sample_names(rng, 7)
    a, b, c, d, e, f, g = names
    surface = SURFACES[surface_id]

    object_group = f"{family}_{idx:03d}"
    object_id = f"{object_group}_{variant}_s{surface_id}"

    answer_mode = "single"
    correct = ""
    allowed_answers: List[str] = []
    wrong_candidates: List[str] = []
    gold_constraint_family = family

    if family == "transitive_location":
        if variant == "base":
            endpoint = e
        elif variant == "object_changed":
            endpoint = f
        else:
            endpoint = e
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
        if variant == "base":
            endpoint = e
        elif variant == "object_changed":
            endpoint = f
        else:
            endpoint = e
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
        if variant == "base":
            endpoint = d
        elif variant == "object_changed":
            endpoint = e
        else:
            endpoint = d
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
        if variant == "base":
            default = c
            exception = e
        elif variant == "object_changed":
            default = c
            exception = f
        else:
            default = c
            exception = e
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
        if variant == "base":
            expert = f
        elif variant == "object_changed":
            expert = e
        else:
            expert = f
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
        # The correct behavior is no unique endpoint.  Candidate scoring will treat
        # AMBIGUOUS as the selected class, but generation may output text.
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

    # Constraint-type perturbation for a selected subset:
    if variant == "constraint_changed":
        # Convert any family into an exception override instance while keeping
        # object token anchor similar.  This tests whether trace follows structure type.
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

    return {
        "object_group": object_group,
        "object_id": object_id,
        "surface_id": surface_id,
        "variant": variant,
        "object_structure_type": family,
        "gold_constraint_family": gold_constraint_family,
        "answer_mode": answer_mode,
        "gold_answer": correct,
        "allowed_answers": json.dumps(allowed_answers, ensure_ascii=False),
        "candidate_answers": json.dumps(candidate_answers, ensure_ascii=False),
        "prompt": prompt,
    }


def build_oia_dataset(n_base_per_family: int = N_BASE_PER_FAMILY) -> pd.DataFrame:
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
    df = pd.DataFrame(rows)
    return df


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
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    ).to(DEVICE)


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
    if not math.isfinite(val):
        return -1e9
    return val


def normalize_answer(text: str) -> str:
    s = str(text).strip()
    s = s.splitlines()[0].strip() if s else ""
    s = re.sub(r"^[\"'`\s]+|[\"'`\s.。,:;]+$", "", s)
    return s


def extract_endpoint_from_generation(gen: str, candidates: List[str]) -> str:
    low = gen.lower()
    if "ambiguous" in low or "cannot determine" in low or "no unique" in low:
        return "AMBIGUOUS"
    # Prefer exact candidate mentions.
    for cand in sorted(candidates, key=len, reverse=True):
        if cand.lower() in low:
            return cand
    first = normalize_answer(gen)
    for cand in candidates:
        if first.lower() == cand.lower():
            return cand
    return first if first else "EMPTY"


@torch.no_grad()
def generate_trace(tokenizer, model, prompt: str) -> str:
    trace_prompt = (
        prompt
        + "\n\nNow give a compact JSON object with keys "
        + '"answer", "constraint_family", and "constraint_trace". '
        + "Use one of these constraint_family labels exactly: "
        + ", ".join(STRUCTURE_FAMILIES)
        + "."
    )
    inputs = tokenizer(
        trace_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
    ).to(DEVICE)

    gen = model.generate(
        **inputs,
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = gen[0, inputs["input_ids"].shape[1]:]
    out = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return out.strip()


# ---------------------------------------------------------------------
# Trace parsing and scoring
# ---------------------------------------------------------------------

def predict_constraint_family_from_text(text: str) -> str:
    low = text.lower()
    scores = {}
    for fam, words in CONSTRAINT_KEYWORDS.items():
        scores[fam] = sum(1 for w in words if w.lower() in low)
    best = max(scores.items(), key=lambda kv: kv[1])
    if best[1] <= 0:
        return "unknown"
    return best[0]


def parse_generation_family(gen: str) -> str:
    # First look for explicit known labels.
    low = gen.lower()
    for fam in STRUCTURE_FAMILIES:
        if fam.lower() in low:
            return fam
    return predict_constraint_family_from_text(gen)


def trace_match_score(gold: str, pred: str, gen: str) -> float:
    if pred == gold:
        return 1.0
    # Some partial credit for shared high-level families.
    closure_like = {"transitive_location", "containment_closure"}
    priority_like = {"temporal_update", "exception_override", "source_priority"}
    if gold in closure_like and pred in closure_like:
        return 0.7
    if gold in priority_like and pred in priority_like:
        return 0.45
    # If generated trace contains gold keywords, weak credit.
    kw = CONSTRAINT_KEYWORDS.get(gold, [])
    low = gen.lower()
    hits = sum(1 for w in kw if w.lower() in low)
    if hits >= 2:
        return 0.35
    if hits == 1:
        return 0.2
    return 0.0


def select_state(answer_correct: int, selected_answer: str, answer_mode: str, v_f: float) -> str:
    if selected_answer in {"EMPTY", ""}:
        return "Unresolved"
    if answer_mode == "ambiguous" and selected_answer == "AMBIGUOUS":
        return "CorrectSelect"
    if selected_answer == "AMBIGUOUS" and answer_mode != "ambiguous":
        return "Unresolved"
    if answer_correct and v_f >= 0.6:
        return "CorrectSelect"
    if answer_correct:
        return "CorrectAnswerLowTrace"
    return "WrongSelect"


# ---------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------

def run_one_model(model_key: str, df: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    model_dir = out_dir / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print(f"OIA-1 MODEL: {model_key}")
    print("=" * 88)

    tokenizer, model = load_model(model_key)

    score_rows = []
    trace_rows = []
    audit_rows = []

    for i, row in df.iterrows():
        prompt = row["prompt"]
        candidates = json.loads(row["candidate_answers"])

        scores = {}
        for cand in candidates:
            scores[cand] = score_candidate(tokenizer, model, prompt, cand)

        selected = max(scores.items(), key=lambda kv: kv[1])[0]
        gold = row["gold_answer"]
        allowed = json.loads(row["allowed_answers"])
        answer_correct = int(selected in allowed)

        gen = generate_trace(tokenizer, model, prompt)
        gen_family = parse_generation_family(gen)
        gen_answer = extract_endpoint_from_generation(gen, candidates)
        gen_answer_correct = int(gen_answer in allowed)
        v_f = trace_match_score(row["gold_constraint_family"], gen_family, gen)

        # Candidate-selected answer is more controlled than free generation.
        state = select_state(answer_correct, selected, row["answer_mode"], v_f)

        score_rows.append({
            "row_id": int(row["row_id"]),
            "selected_answer": selected,
            "gold_answer": gold,
            "answer_correct": answer_correct,
            "candidate_scores": json.dumps(scores, ensure_ascii=False),
        })
        trace_rows.append({
            "row_id": int(row["row_id"]),
            "trace_generation": gen,
            "generated_answer_extract": gen_answer,
            "generated_answer_correct": gen_answer_correct,
            "pred_constraint_family": gen_family,
            "gold_constraint_family": row["gold_constraint_family"],
            "trace_match_score": v_f,
        })
        audit_rows.append({
            **row.to_dict(),
            "selected_answer": selected,
            "answer_correct": answer_correct,
            "generated_answer_extract": gen_answer,
            "generated_answer_correct": gen_answer_correct,
            "pred_constraint_family": gen_family,
            "trace_generation": gen,
            "trace_match_score": v_f,
            "visible_constraint_score": v_f,
            "select_state": state,
        })

        if (i + 1) % 20 == 0:
            print(f"[{model_key}] {i + 1}/{len(df)}")

    score_df = pd.DataFrame(score_rows)
    trace_df = pd.DataFrame(trace_rows)
    audit_df = pd.DataFrame(audit_rows)

    score_df.to_csv(model_dir / f"oia1_{model_key}_candidate_scores.csv", index=False, encoding="utf-8-sig")
    trace_df.to_csv(model_dir / f"oia1_{model_key}_trace_generations.csv", index=False, encoding="utf-8-sig")
    audit_df.to_csv(model_dir / f"oia1_{model_key}_audit_rows.csv", index=False, encoding="utf-8-sig")

    summary = summarize_model(model_key, audit_df)
    with open(model_dir / f"oia1_{model_key}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(model_dir / f"oia1_{model_key}_verdict.txt", "w", encoding="utf-8") as f:
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
    # OIA-1A: fixed O base variants across surfaces
    base_df = audit_df[audit_df["variant"] == "base"].copy()
    group_scores = []
    for (family, idx_group), g in base_df.groupby(["object_structure_type", "object_group"]):
        group_scores.append({
            "object_structure_type": family,
            "object_group": idx_group,
            "answer_stability": pairwise_stability(g["selected_answer"].tolist()),
            "trace_family_stability": pairwise_stability(g["pred_constraint_family"].tolist()),
            "mean_trace_match": float(g["trace_match_score"].mean()),
            "mean_answer_correct": float(g["answer_correct"].mean()),
        })
    group_stability_df = pd.DataFrame(group_scores)

    # OIA-1B: object_changed should keep family but change selected endpoint with new object.
    changed_df = audit_df[audit_df["variant"] == "object_changed"].copy()

    # OIA-1C: constraint_changed should predict exception_override.
    ctype_df = audit_df[audit_df["variant"] == "constraint_changed"].copy()

    all_y_true = audit_df["gold_constraint_family"].astype(str).tolist()
    all_y_pred = audit_df["pred_constraint_family"].astype(str).tolist()
    labels = STRUCTURE_FAMILIES + ["unknown"]
    family_acc = accuracy_score(all_y_true, all_y_pred)
    family_f1 = f1_score(all_y_true, all_y_pred, labels=STRUCTURE_FAMILIES, average="macro", zero_division=0)

    oia1a_trace_stability = float(group_stability_df["trace_family_stability"].mean()) if len(group_stability_df) else 0.0
    oia1a_answer_stability = float(group_stability_df["answer_stability"].mean()) if len(group_stability_df) else 0.0
    oia1a_vf = float(group_stability_df["mean_trace_match"].mean()) if len(group_stability_df) else 0.0

    oia1b_answer_correct = float(changed_df["answer_correct"].mean()) if len(changed_df) else 0.0
    oia1b_trace_match = float(changed_df["trace_match_score"].mean()) if len(changed_df) else 0.0

    oia1c_trace_match = float(ctype_df["trace_match_score"].mean()) if len(ctype_df) else 0.0
    oia1c_exception_pred_rate = float((ctype_df["pred_constraint_family"] == "exception_override").mean()) if len(ctype_df) else 0.0

    correct_select = audit_df[audit_df["select_state"].isin(["CorrectSelect", "CorrectAnswerLowTrace"])]
    wrong_select = audit_df[audit_df["select_state"] == "WrongSelect"]
    unresolved = audit_df[audit_df["select_state"] == "Unresolved"]

    vf_correct = float(correct_select["visible_constraint_score"].mean()) if len(correct_select) else None
    vf_wrong = float(wrong_select["visible_constraint_score"].mean()) if len(wrong_select) else None
    vf_unresolved = float(unresolved["visible_constraint_score"].mean()) if len(unresolved) else None

    # Cbit proxy under controlled candidate set:
    # initial candidate count includes all candidates; final count is 1 for definite select,
    # all candidates for unresolved ambiguity if model says ambiguous incorrectly.
    def cbit_proxy(row):
        n_init = len(json.loads(row["candidate_answers"]))
        if row["select_state"] == "Unresolved":
            n_final = max(2, n_init)
        else:
            n_final = 1
        raw = math.log2(max(1, n_init) / max(1, n_final))
        return raw * float(row["visible_constraint_score"])

    audit_df = audit_df.copy()
    audit_df["cbit_eff_proxy"] = audit_df.apply(cbit_proxy, axis=1)
    mean_cbit_eff_correct = float(audit_df[audit_df["answer_correct"] == 1]["cbit_eff_proxy"].mean())
    mean_cbit_eff_wrong = float(audit_df[audit_df["answer_correct"] == 0]["cbit_eff_proxy"].mean())

    pass_oia1a = oia1a_trace_stability >= 0.70 and oia1a_vf >= 0.45
    pass_oia1b = oia1b_answer_correct >= 0.55 and oia1b_trace_match >= 0.40
    pass_oia1c = oia1c_exception_pred_rate >= 0.45 or oia1c_trace_match >= 0.45
    pass_vf = (
        vf_correct is not None
        and vf_wrong is not None
        and vf_correct > vf_wrong + 0.10
    )
    pass_cbit = mean_cbit_eff_correct > mean_cbit_eff_wrong + 0.20

    if pass_oia1a and pass_oia1b and pass_oia1c and (pass_vf or pass_cbit):
        verdict = "PASS_OIA1_OBJECT_CONSTRAINT_INVARIANCE"
    elif pass_oia1a and (pass_oia1b or pass_oia1c):
        verdict = "PARTIAL_OIA1_OBJECT_CONSTRAINT_SIGNAL"
    else:
        verdict = "FAIL_OR_CAVEAT_OIA1"

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
            "oia1a_fixed_object_surface_trace_stability": bool(pass_oia1a),
            "oia1b_object_binding_perturbation": bool(pass_oia1b),
            "oia1c_constraint_type_perturbation": bool(pass_oia1c),
            "vf_proxy_correct_gt_wrong": bool(pass_vf),
            "cbit_eff_correct_gt_wrong": bool(pass_cbit),
        },
        "metrics": {
            "constraint_family_accuracy": float(family_acc),
            "constraint_family_macro_f1": float(family_f1),
            "overall_answer_correct": float(audit_df["answer_correct"].mean()),
            "overall_generated_answer_correct": float(audit_df["generated_answer_correct"].mean()),
            "overall_trace_match_score": float(audit_df["trace_match_score"].mean()),
            "oia1a_trace_family_stability": oia1a_trace_stability,
            "oia1a_answer_stability": oia1a_answer_stability,
            "oia1a_visible_constraint_score": oia1a_vf,
            "oia1b_answer_correct": oia1b_answer_correct,
            "oia1b_trace_match_score": oia1b_trace_match,
            "oia1c_trace_match_score": oia1c_trace_match,
            "oia1c_exception_pred_rate": oia1c_exception_pred_rate,
            "vf_correct_select_mean": vf_correct,
            "vf_wrong_select_mean": vf_wrong,
            "vf_unresolved_mean": vf_unresolved,
            "mean_cbit_eff_correct": mean_cbit_eff_correct,
            "mean_cbit_eff_wrong": mean_cbit_eff_wrong,
        },
        "family_confusion_rows": family_table.to_dict(orient="records"),
        "select_state_counts": state_table.to_dict(orient="records"),
        "interpretation": (
            "OIA-1 validates the pending claim that fixed objects induce stable observable "
            "constraint traces under surface perturbation, while changed object bindings or "
            "constraint-type perturbations systematically alter trace, selection, or ambiguity state."
        ),
        "model_path": MODEL_PATHS[model_key],
    }


def aggregate_summaries(summaries: List[Dict[str, Any]], out_dir: Path):
    rows = []
    for s in summaries:
        m = s["metrics"]
        rows.append({
            "model_key": s["model_key"],
            "verdict": s["verdict"],
            **s["pass_flags"],
            **m,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "oia1_crossmodel_summary.csv", index=False, encoding="utf-8-sig")

    n = len(summaries)
    n_pass_like = int(df["verdict"].str.contains("PASS|PARTIAL", regex=True).sum())
    n_oia1a = int(df["oia1a_fixed_object_surface_trace_stability"].sum())
    n_oia1b = int(df["oia1b_object_binding_perturbation"].sum())
    n_oia1c = int(df["oia1c_constraint_type_perturbation"].sum())

    if n_pass_like == n and n_oia1a >= max(1, n - 1) and n_oia1b >= max(1, n - 1) and n_oia1c >= max(1, n - 1):
        verdict = "PASS_OIA1_CROSSMODEL_OBJECT_CONSTRAINT_INVARIANCE"
    elif n_pass_like >= max(1, n - 1) and n_oia1a >= max(1, n - 1):
        verdict = "PARTIAL_OIA1_CROSSMODEL_CONSTRAINT_INVARIANCE_SIGNAL"
    else:
        verdict = "FAIL_OR_CAVEAT_OIA1_CROSSMODEL"

    agg = {
        "verdict": verdict,
        "n_models": n,
        "n_pass_or_partial": n_pass_like,
        "n_oia1a_pass": n_oia1a,
        "n_oia1b_pass": n_oia1b,
        "n_oia1c_pass": n_oia1c,
        "models": rows,
        "pending_claim_tested": "Inv(O) -> Inv(F_O) as observable constraint-trace invariance",
        "caveat": (
            "This is a first synthetic-text proxy audit. A stronger version should replace generated "
            "constraint traces with hidden-state / TopK / VIM constraint-field proxies."
        ),
    }
    with open(out_dir / "oia1_crossmodel_summary.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    with open(out_dir / "oia1_crossmodel_verdict.txt", "w", encoding="utf-8") as f:
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
    }
    with open(out_dir / "oia1_hardcoded_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    df = build_oia_dataset(N_BASE_PER_FAMILY)
    df.to_csv(out_dir / "oia1_dataset.csv", index=False, encoding="utf-8-sig")
    print(f"[dataset] rows={len(df)} out={out_dir / 'oia1_dataset.csv'}")

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
    print("=" * 88)
    print("OIA-1 CROSS-MODEL VERDICT")
    print(json.dumps(agg, ensure_ascii=False, indent=2))
    print("=" * 88)


if __name__ == "__main__":
    main()
