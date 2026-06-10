
# ============================================================
# ASA-9B.3 Driver: Probe-Binding Cross-Task Policy Audit
#
# Purpose:
#   ASA-9B-lite showed:
#       temporal_update      PASS-Strong
#       exception_override   FAIL
#
#   Diagnosis:
#       coarse mechanism labels {stable, competition, closure}
#       are insufficient for exception/override transfer.
#
#   ASA-9B.3 adds override-aware submechanism labels while reusing the
#   ASA-8D engine:
#
#       stable
#       competition
#       canonical_closure
#       temporal_update
#       rule_override
#       exception_binding
#
#   The engine itself still expects integer mechanism_id and a string
#   mechanism. We monkey-patch the global MECH_TO_ID / ID_TO_MECH inside
#   the ASA-8D module so the mechanism classifier becomes 6-way.
#
# Main test:
#   Leave-One-Task-Family-Out, focused on:
#       exception_override
#       temporal_update
#
# Default first-pass:
#   N_SHUFFLES = 20
#   N_GRAPHS_PER_FAMILY = 48
#
# If exception_override improves:
#   rerun with N_SHUFFLES = 20 or 50 and N_GRAPHS_PER_FAMILY = 48.
#
# Required:
#   Put this file in the same folder as:
#       asa8d_matched_shuffle_closed_loop_validation_PATCHED.py
#   or:
#       GPT_194_ASA_8D.py
#
# Outputs:
#   asa9b3_outputs/
#     asa9b3_cross_task_summary.csv
#     asa9b3_cross_task_summary.json
#     asa9b3_final_verdict.json
#     heldout_exception_override/
#     heldout_temporal_update/
# ============================================================

import os
import sys
import json
import copy
import random
import traceback
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# USER SETTINGS
# ============================================================

ROOT_SAVE_DIR = Path("./asa9b3_probe_binding_outputs")

MODEL_KEY = "qwen"
MODEL_PATH = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

OPERATOR_LAYERS = tuple(range(15, 20))
PRECURSOR_LAYERS = (17, 18, 19)
DECISION_LAYERS = tuple(range(20, 26))

N_SHUFFLES = 20
BATCH_SIZE = 4
N_GRAPHS_PER_FAMILY = 48

ASA8D_CANDIDATE_FILES = [
    "asa8d_matched_shuffle_closed_loop_validation_PATCHED.py",
    "GPT_194_ASA_8D.py",
    "asa8d_matched_shuffle_closed_loop_validation.py",
]

# Full family set. LOTFO can choose a subset of heldouts.
TASK_FAMILIES = [
    "relation_chain",
    "rule_application",
    "source_priority",
    "temporal_update",
    "exception_override",
]

TASK_FAMILIES_TO_HOLDOUT = [
    "exception_override",
    "temporal_update",
]

# Six-way override-aware mechanism space.
MECH_TO_ID_6 = {
    "stable": 0,
    "competition": 1,
    "canonical_closure": 2,
    "temporal_update": 3,
    "rule_override": 4,
    "exception_binding": 5,
}
ID_TO_MECH_6 = {v: k for k, v in MECH_TO_ID_6.items()}

# Internal probe-binding features: hidden-state readouts of marker directions via lm_head.
PROBE_BINDING_MARKERS = {
    "general": ["general", "rule", "default"],
    "override": ["override", "overridden"],
    "exception": ["exception", "special"],
    "old": ["old", "previous", "earlier"],
    "new": ["new", "updated", "current", "now"],
    "source": ["source", "registry", "record", "verified"],
    "direct": ["direct", "conflicting"],
    "irrelevant": ["irrelevant", "unrelated"],
    "temporal": ["today", "yesterday", "latest", "current"],
}

PROBE_BINDING_FEATURE_COLS = [
    "probe_general", "probe_override", "probe_exception", "probe_old",
    "probe_new", "probe_source", "probe_direct", "probe_irrelevant",
    "probe_temporal", "probe_rule_pressure", "probe_override_pressure",
    "probe_exception_pressure", "probe_update_pressure", "probe_source_pressure",
    "probe_conflict_answer_logit", "probe_clean_answer_logit",
    "probe_answer_margin", "probe_override_x_conflict",
    "probe_exception_x_conflict", "probe_source_x_conflict",
    "probe_update_x_conflict",
]

_ASA9B3_MARKER_VECS = None

def _mean_vec_for_words(tokenizer, W_np, words):
    vecs = []
    for word in words:
        ids = tokenizer(" " + word, add_special_tokens=False)["input_ids"]
        ids = [i for i in ids if 0 <= i < W_np.shape[0]]
        if ids:
            vecs.append(W_np[ids].mean(axis=0))
    if not vecs:
        return None
    v = np.stack(vecs, axis=0).mean(axis=0).astype(np.float32)
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)

def _build_marker_vectors(tokenizer, W_np):
    out = {name: _mean_vec_for_words(tokenizer, W_np, words) for name, words in PROBE_BINDING_MARKERS.items()}
    out["_W_np"] = W_np
    return out

def _normed_dot(h, v):
    if v is None:
        return 0.0
    h = h.astype(np.float32)
    return float(np.dot(h / (np.linalg.norm(h) + 1e-8), v))

def ensure_six_mech_columns(df):
    out = df.copy()
    for name in MECH_TO_ID_6.keys():
        p_col = f"p_mech_{name}"
        pred_col = f"pred_mech_{name}"
        if p_col not in out.columns:
            out[p_col] = 0.0
        if pred_col not in out.columns:
            out[pred_col] = 0
    return out

def attach_probe_binding_features(feature_df, base_df, h_by_layer):
    global _ASA9B3_MARKER_VECS
    marker_vecs = _ASA9B3_MARKER_VECS or {}
    W_np = marker_vecs.get("_W_np", None)
    base_reset = base_df.reset_index(drop=True)
    rows = []

    for _, fr in feature_df.iterrows():
        idx = int(fr["sample_index"])
        l = int(fr["layer"])
        h = h_by_layer[l][idx].astype(np.float32)

        vals = {}
        for name in PROBE_BINDING_MARKERS.keys():
            vals[f"probe_{name}"] = _normed_dot(h, marker_vecs.get(name))

        if W_np is not None:
            clean_id = int(base_reset.iloc[idx]["clean_token_id"])
            conflict_id = int(base_reset.iloc[idx]["conflict_token_id"])
            if 0 <= clean_id < W_np.shape[0] and 0 <= conflict_id < W_np.shape[0]:
                clean_v = W_np[clean_id].astype(np.float32)
                conflict_v = W_np[conflict_id].astype(np.float32)
                clean_v = clean_v / (np.linalg.norm(clean_v) + 1e-8)
                conflict_v = conflict_v / (np.linalg.norm(conflict_v) + 1e-8)
                clean_logit = _normed_dot(h, clean_v)
                conflict_logit = _normed_dot(h, conflict_v)
            else:
                clean_logit = 0.0
                conflict_logit = 0.0
        else:
            clean_logit = 0.0
            conflict_logit = 0.0

        general = vals.get("probe_general", 0.0)
        override = vals.get("probe_override", 0.0)
        exception = vals.get("probe_exception", 0.0)
        old = vals.get("probe_old", 0.0)
        new = vals.get("probe_new", 0.0)
        source = vals.get("probe_source", 0.0)
        temporal = vals.get("probe_temporal", 0.0)

        vals.update({
            "probe_rule_pressure": general,
            "probe_override_pressure": override - general,
            "probe_exception_pressure": exception - general,
            "probe_update_pressure": new - old,
            "probe_source_pressure": source + 0.5 * temporal,
            "probe_conflict_answer_logit": conflict_logit,
            "probe_clean_answer_logit": clean_logit,
            "probe_answer_margin": clean_logit - conflict_logit,
            "probe_override_x_conflict": override * conflict_logit,
            "probe_exception_x_conflict": exception * conflict_logit,
            "probe_source_x_conflict": source * conflict_logit,
            "probe_update_x_conflict": (new - old) * conflict_logit,
        })
        rows.append(vals)

    probe_df = pd.DataFrame(rows).reset_index(drop=True)
    out = feature_df.reset_index(drop=True).copy()
    for col in PROBE_BINDING_FEATURE_COLS:
        out[col] = probe_df[col].values if col in probe_df.columns else 0.0
    return out

# ============================================================
# JSON SANITIZER
# ============================================================

def to_jsonable(obj):
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        if np.isnan(x) or np.isinf(x):
            return None
        return x
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)

# ============================================================
# IMPORT ASA-8D ENGINE
# ============================================================

def find_asa8d_file():
    here = Path(__file__).resolve().parent
    for name in ASA8D_CANDIDATE_FILES:
        p = here / name
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find ASA-8D engine. Put one of these files in the same folder:\n"
        + "\n".join(ASA8D_CANDIDATE_FILES)
    )

def import_module_from_file(path):
    spec = importlib.util.spec_from_file_location("asa8d_engine", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["asa8d_engine"] = mod
    spec.loader.exec_module(mod)
    return mod

# ============================================================
# DATASET
# ============================================================

LABEL_CANDIDATES = [
    "Red", "Blue", "Green", "Yellow", "North", "South", "East", "West",
    "Copper", "Silver", "Gold", "Iron", "Circle", "Square", "Triangle", "Star",
    "River", "Mountain", "Forest", "Ocean", "Sun", "Moon", "Cloud", "Stone",
    "Alpha", "Beta", "Gamma", "Delta", "Apple", "Orange", "Lemon", "Pear",
]

ENTITIES = [
    ("Ava", "Bela", "Cora"), ("Darin", "Elo", "Faye"),
    ("Galen", "Hera", "Ivo"), ("Juno", "Kira", "Lio"),
    ("Mira", "Nero", "Orin"), ("Pia", "Quin", "Rhea"),
    ("Sola", "Taro", "Una"), ("Vera", "Wen", "Xio"),
    ("Yara", "Zeno", "Nia"), ("Orla", "Pavel", "Rin"),
    ("Nora", "Silas", "Tess"), ("Uma", "Vito", "Willa"),
    ("Xena", "Yuri", "Zara"), ("Iris", "Kai", "Lena"),
    ("Omar", "Priya", "Quill"), ("Ravi", "Sara", "Theo"),
    ("Ari", "Bryn", "Cyra"), ("Dax", "Eira", "Finn"),
    ("Gia", "Hale", "Iris2"), ("Jace", "Kora", "Lux"),
    ("Milo", "Naya", "Oren"), ("Pax", "Quora", "Rumi"),
]

REL_WORDS = [
    ("belongs to", "is located at"),
    ("is assigned to", "maps to"),
    ("is part of", "points to"),
    ("is grouped under", "has label"),
    ("routes through", "ends at"),
    ("is linked with", "resolves to"),
]

# Engine requires a condition == "clean" anchor.
# We keep that invariant.
CONDITIONS = [
    "clean",
    "irrelevant",
    "competition_balanced",
    "direct_conflict",
    "closure_update",
    "temporal_update",
    "rule_override",
    "exception_binding",
]

def continuation_ids(tokenizer, text):
    return tokenizer(" " + text, add_special_tokens=False)["input_ids"]

def select_single_token_labels(tokenizer, save_dir):
    rows, chosen = [], []
    for lab in LABEL_CANDIDATES:
        ids = continuation_ids(tokenizer, lab)
        rows.append({"label": lab, "ids": str(ids), "len": len(ids), "single": int(len(ids) == 1)})
        if len(ids) == 1:
            chosen.append(lab)
    pd.DataFrame(rows).to_csv(Path(save_dir) / "asa9b3_label_tokenization_audit.csv", index=False, encoding="utf-8-sig")
    if len(chosen) < 12:
        raise RuntimeError(f"Need >=12 single-token labels; got {chosen}")
    return chosen[:12]

def mechanism_for_condition(task_family, condition):
    if condition in ("clean", "irrelevant"):
        return "stable"
    if condition in ("competition_balanced", "direct_conflict"):
        return "competition"

    if condition == "closure_update":
        # For non-temporal / non-exception families this is canonical closure rewrite.
        if task_family == "temporal_update":
            return "temporal_update"
        if task_family == "exception_override":
            return "exception_binding"
        return "canonical_closure"

    if condition == "temporal_update":
        return "temporal_update"

    if condition == "rule_override":
        return "rule_override"

    if condition == "exception_binding":
        return "exception_binding"

    raise ValueError((task_family, condition))

def make_prompt(task_family, condition, a, b, d, clean_label, conflict_label, aux_label, relation1, relation2):
    # 1. Relation chain family
    if task_family == "relation_chain":
        if condition == "clean":
            return "\n".join([
                "You are given a small relation graph.",
                f"Fact 1: {a} {relation1} {b}.",
                f"Fact 2: {b} {relation2} {clean_label}.",
                f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "irrelevant":
            return "\n".join([
                "You are given a small relation graph.",
                f"Fact 1: {a} {relation1} {b}.",
                f"Fact 2: {b} {relation2} {clean_label}.",
                f"Irrelevant fact: {d} is associated with {aux_label}.",
                f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "competition_balanced":
            return "\n".join([
                "You are given a small relation graph.",
                f"Fact 1: {a} {relation1} {b}.",
                f"Fact 2: {b} {relation2} {clean_label}.",
                f"Competing fact: {a} is also associated with {conflict_label}.",
                f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "direct_conflict":
            return "\n".join([
                "You are given a small relation graph.",
                f"Fact 1: {a} {relation1} {b}.",
                f"Fact 2: {b} {relation2} {clean_label}.",
                f"Direct conflicting fact: {a} {relation2} {conflict_label}.",
                f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition in ("closure_update", "temporal_update"):
            return "\n".join([
                "You are given a small relation graph.",
                f"Fact 1: {a} {relation1} {b}.",
                f"Old record: {b} {relation2} {clean_label}.",
                f"Updated record: {b} {relation2} {conflict_label}, not {clean_label}.",
                f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "rule_override":
            return "\n".join([
                "You are given a rule system.",
                f"General rule: items that {relation1} {b} receive label {clean_label}.",
                f"Override rule: in this case, items that {relation1} {b} receive label {conflict_label}.",
                f"Fact: {a} {relation1} {b}.",
                f"Question: Which label applies to {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "exception_binding":
            return "\n".join([
                "You are given a rule system.",
                f"General rule: items connected to {b} use label {clean_label}.",
                f"Exception: {a} is a special case and uses label {conflict_label}.",
                f"Fact: {a} is connected to {b}.",
                f"Question: Which label applies to {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])

    # 2. Rule application family
    if task_family == "rule_application":
        if condition == "clean":
            return "\n".join([
                "Use the rule system below.",
                f"Rule: every item in group {b} receives label {clean_label}.",
                f"Fact: {a} is in group {b}.",
                f"Question: Which label applies to {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "irrelevant":
            return "\n".join([
                "Use the rule system below.",
                f"Rule: every item in group {b} receives label {clean_label}.",
                f"Fact: {a} is in group {b}.",
                f"Irrelevant rule: items in group {d} receive label {aux_label}.",
                f"Question: Which label applies to {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "competition_balanced":
            return "\n".join([
                "Use the rule system below.",
                f"Rule A: every item in group {b} receives label {clean_label}.",
                f"Rule B: {a} may receive label {conflict_label}.",
                f"Fact: {a} is in group {b}.",
                f"Question: Which label applies to {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "direct_conflict":
            return "\n".join([
                "Use the rule system below.",
                f"General rule: every item in group {b} receives label {clean_label}.",
                f"Direct statement: {a} receives label {conflict_label}.",
                f"Fact: {a} is in group {b}.",
                f"Question: Which label applies to {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition in ("closure_update", "temporal_update"):
            return "\n".join([
                "Use the rule system below.",
                f"Old rule: every item in group {b} receives label {clean_label}.",
                f"New rule: every item in group {b} receives label {conflict_label}.",
                f"Fact: {a} is in group {b}.",
                f"Question: Which label applies to {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "rule_override":
            return "\n".join([
                "Use the rule system below.",
                f"General rule: every item in group {b} receives label {clean_label}.",
                f"Override rule: for this task, group {b} receives label {conflict_label}.",
                f"Fact: {a} is in group {b}.",
                f"Question: Which label applies to {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "exception_binding":
            return "\n".join([
                "Use the rule system below.",
                f"General rule: every item in group {b} receives label {clean_label}.",
                f"Exception list: {a} receives label {conflict_label}.",
                f"Fact: {a} is in group {b}.",
                f"Question: Which label applies to {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])

    # 3. Source priority family
    if task_family == "source_priority":
        if condition == "clean":
            return "\n".join([
                "Resolve the label using the trusted registry.",
                f"Trusted registry: {a} has label {clean_label}.",
                f"Question: Which label is correct for {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "irrelevant":
            return "\n".join([
                "Resolve the label using the trusted registry.",
                f"Trusted registry: {a} has label {clean_label}.",
                f"Unrelated note: {d} has label {aux_label}.",
                f"Question: Which label is correct for {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "competition_balanced":
            return "\n".join([
                "Resolve the label using the records.",
                f"Record 1 says: {a} has label {clean_label}.",
                f"Record 2 says: {a} has label {conflict_label}.",
                f"Question: Which label is correct for {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "direct_conflict":
            return "\n".join([
                "Resolve the label using the records.",
                f"Main record: {a} has label {clean_label}.",
                f"Conflicting memo: {a} has label {conflict_label}.",
                f"Question: Which label is correct for {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition in ("closure_update", "temporal_update"):
            return "\n".join([
                "Resolve the label using source priority.",
                f"Old registry: {a} had label {clean_label}.",
                f"Verified registry update: {a} now has label {conflict_label}.",
                f"Question: Which label is correct for {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "rule_override":
            return "\n".join([
                "Resolve the label using source priority.",
                f"Default registry: {a} has label {clean_label}.",
                f"Override registry: for this task, {a} has label {conflict_label}.",
                f"Instruction: use the override registry.",
                f"Question: Which label is correct for {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "exception_binding":
            return "\n".join([
                "Resolve the label using source priority.",
                f"Registry rule: items like {a} usually have label {clean_label}.",
                f"Exception entry: {a} specifically has label {conflict_label}.",
                f"Question: Which label is correct for {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])

    # 4. Temporal update family
    if task_family == "temporal_update":
        if condition == "clean":
            return "\n".join([
                "Use the latest temporal fact.",
                f"At time T1, {a} was assigned label {clean_label}.",
                f"At time T2, the assignment remains {clean_label}.",
                f"Question: What is the current label of {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "irrelevant":
            return "\n".join([
                "Use the latest temporal fact.",
                f"At time T1, {a} was assigned label {clean_label}.",
                f"At time T2, the assignment remains {clean_label}.",
                f"At time T2, {d} was assigned {aux_label}.",
                f"Question: What is the current label of {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "competition_balanced":
            return "\n".join([
                "Use the temporal notes.",
                f"One note says {a} currently has label {clean_label}.",
                f"Another note says {a} currently has label {conflict_label}.",
                f"Question: What is the current label of {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "direct_conflict":
            return "\n".join([
                "Use the temporal notes.",
                f"Earlier today, {a} was assigned label {clean_label}.",
                f"A direct later note says {a} has label {conflict_label}.",
                f"Question: What is the current label of {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition in ("closure_update", "temporal_update"):
            return "\n".join([
                "Use the latest temporal update.",
                f"Yesterday, {a} had label {clean_label}.",
                f"Today, {a} was updated to label {conflict_label}.",
                f"Question: What is the current label of {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "rule_override":
            return "\n".join([
                "Use the latest temporal override.",
                f"Old standing rule: {a} has label {clean_label}.",
                f"Temporary override for today: {a} has label {conflict_label}.",
                f"Question: What is the current label of {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "exception_binding":
            return "\n".join([
                "Use the latest temporal exception list.",
                f"General current rule: items like {a} have label {clean_label}.",
                f"Current exception: {a} has label {conflict_label}.",
                f"Question: What is the current label of {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])

    # 5. Exception override family
    if task_family == "exception_override":
        if condition == "clean":
            return "\n".join([
                "Use the rule and exception list.",
                f"General rule: items linked to {b} receive label {clean_label}.",
                f"Fact: {a} is linked to {b}.",
                f"No exception applies to {a}.",
                f"Question: Which label applies to {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "irrelevant":
            return "\n".join([
                "Use the rule and exception list.",
                f"General rule: items linked to {b} receive label {clean_label}.",
                f"Fact: {a} is linked to {b}.",
                f"Exception: {d} receives label {aux_label}.",
                f"Question: Which label applies to {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "competition_balanced":
            return "\n".join([
                "Use the rule and exception list.",
                f"General rule: items linked to {b} receive label {clean_label}.",
                f"Possible exception: {a} may receive label {conflict_label}.",
                f"Fact: {a} is linked to {b}.",
                f"Question: Which label applies to {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "direct_conflict":
            return "\n".join([
                "Use the rule and exception list.",
                f"General rule: items linked to {b} receive label {clean_label}.",
                f"Direct exception candidate: {a} receives label {conflict_label}.",
                f"Fact: {a} is linked to {b}.",
                f"Question: Which label applies to {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition in ("closure_update", "exception_binding"):
            return "\n".join([
                "Use the rule and exception list.",
                f"General rule: items linked to {b} receive label {clean_label}.",
                f"Exception: {a} is a special case and receives label {conflict_label}.",
                f"Fact: {a} is linked to {b}.",
                f"Question: Which label applies to {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "temporal_update":
            return "\n".join([
                "Use the rule and exception list.",
                f"Old exception list: {a} had label {clean_label}.",
                f"Updated exception list: {a} now has label {conflict_label}.",
                f"Question: Which label applies to {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "rule_override":
            return "\n".join([
                "Use the rule and exception list.",
                f"General rule: items linked to {b} receive label {clean_label}.",
                f"Override rule: all items linked to {b} receive label {conflict_label} in this case.",
                f"Fact: {a} is linked to {b}.",
                f"Question: Which label applies to {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])

    raise ValueError(f"Unknown task_family={task_family}, condition={condition}")

def build_override_aware_dataset(tokenizer, cfg, save_dir):
    labels = select_single_token_labels(tokenizer, save_dir)
    rows = []
    graph_id = 0
    n_labels = len(labels)

    for task_family in TASK_FAMILIES:
        for local_g in range(N_GRAPHS_PER_FAMILY):
            a, b, d = ENTITIES[local_g % len(ENTITIES)]
            r1, r2 = REL_WORDS[local_g % len(REL_WORDS)]

            clean_label = labels[(2 * graph_id) % n_labels]
            conflict_label = labels[(2 * graph_id + 1) % n_labels]
            aux_label = labels[(2 * graph_id + 2) % n_labels]

            clean_id = continuation_ids(tokenizer, clean_label)[0]
            conflict_id = continuation_ids(tokenizer, conflict_label)[0]

            for cond in CONDITIONS:
                submech = mechanism_for_condition(task_family, cond)

                # IMPORTANT:
                # Keep `mechanism` as coarse labels for the ASA-8D engine:
                #   stable / competition / closure
                # The engine uses this string for:
                #   1) stable/closure operator fitting
                #   2) target/nontarget specificity summaries
                #
                # Use `mechanism_id` for the override-aware 6-way mechanism classifier.
                if submech == "stable":
                    coarse_mech = "stable"
                elif submech == "competition":
                    coarse_mech = "competition"
                else:
                    coarse_mech = "closure"

                text = make_prompt(task_family, cond, a, b, d, clean_label, conflict_label, aux_label, r1, r2)
                rows.append({
                    "prompt_id": f"{task_family}_g{local_g:03d}_{cond}",
                    "graph_id": int(graph_id),
                    "task_family": task_family,
                    "local_graph_id": int(local_g),
                    "condition": cond,
                    "mechanism": coarse_mech,
                    "submechanism": submech,
                    "mechanism_id": MECH_TO_ID_6[submech],
                    "clean_label": clean_label,
                    "conflict_label": conflict_label,
                    "clean_token_id": clean_id,
                    "conflict_token_id": conflict_id,
                    "text": text,
                })
            graph_id += 1

    df = pd.DataFrame(rows)
    df.to_csv(Path(save_dir) / "asa9b3_dataset.csv", index=False, encoding="utf-8-sig")
    return df

# ============================================================
# ENGINE PATCHING
# ============================================================

def patch_engine_for_six_mechanisms(engine):
    engine.MECH_TO_ID = dict(MECH_TO_ID_6)
    engine.ID_TO_MECH = dict(ID_TO_MECH_6)

    pred_mech_cols = [
        "p_mech_stable", "p_mech_competition", "p_mech_canonical_closure",
        "p_mech_temporal_update", "p_mech_rule_override", "p_mech_exception_binding",
        "pred_mech_stable", "pred_mech_competition", "pred_mech_canonical_closure",
        "pred_mech_temporal_update", "pred_mech_rule_override", "pred_mech_exception_binding",
    ]

    engine.POLICY_PRED_MECH_COLS = list(engine.MECH_FEATURE_COLS) + pred_mech_cols + list(PROBE_BINDING_FEATURE_COLS)
    engine.build_dataset = build_override_aware_dataset

    _orig_extract_baseline = engine.extract_baseline

    def _extract_baseline_capture_markers(model, tokenizer, df, cfg, save_dir):
        global _ASA9B3_MARKER_VECS
        W_np = engine.get_lm_head_weight(model).detach().float().cpu().numpy().astype(np.float32)
        _ASA9B3_MARKER_VECS = _build_marker_vectors(tokenizer, W_np)
        return _orig_extract_baseline(model, tokenizer, df, cfg, save_dir)

    engine.extract_baseline = _extract_baseline_capture_markers

    _orig_base_state_features = engine.base_state_features_for_samples

    def _base_state_features_with_probe_binding(base_df, h_by_layer, probes, vel_stats, sample_idx, cfg):
        out = _orig_base_state_features(base_df, h_by_layer, probes, vel_stats, sample_idx, cfg)
        return attach_probe_binding_features(out, base_df, h_by_layer)

    engine.base_state_features_for_samples = _base_state_features_with_probe_binding

    _orig_attach_predicted_mechanism = engine.attach_predicted_mechanism

    def _attach_predicted_mechanism_sixway(features, clf):
        out = _orig_attach_predicted_mechanism(features, clf)
        out = ensure_six_mech_columns(out)
        for col in PROBE_BINDING_FEATURE_COLS:
            if col not in out.columns:
                out[col] = 0.0
        return out

    engine.attach_predicted_mechanism = _attach_predicted_mechanism_sixway

def patch_config(engine, heldout_family):
    cfg = copy.deepcopy(engine.CFG)

    cfg.model_key = MODEL_KEY
    cfg.model_path = MODEL_PATH
    cfg.save_dir = str(ROOT_SAVE_DIR / f"heldout_{heldout_family}")

    cfg.operator_layers = OPERATOR_LAYERS
    cfg.precursor_layers = PRECURSOR_LAYERS
    cfg.decision_layers = DECISION_LAYERS

    cfg.n_shuffles = N_SHUFFLES
    cfg.batch_size = BATCH_SIZE

    cfg.n_graphs = N_GRAPHS_PER_FAMILY * len(TASK_FAMILIES)

    # Keep compact first-pass grid.
    cfg.policy_actions = (
        (0.0, 0.0),
        (0.1, 0.1),
        (0.3, 0.3),
        (0.3, 0.0),
    )

    return cfg

def make_lotfo_split_func(heldout_family):
    def lotfo_split(df, cfg, save_dir):
        test_mask = df["task_family"].values == heldout_family
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]

        split_df = pd.DataFrame({
            "prompt_id": df["prompt_id"],
            "graph_id": df["graph_id"],
            "task_family": df["task_family"],
            "condition": df["condition"],
            "mechanism": df["mechanism"],
            "submechanism": df["submechanism"] if "submechanism" in df.columns else "",
            "mechanism_id": df["mechanism_id"],
            "split": np.where(test_mask, "test", "train"),
        })
        split_df.to_csv(Path(save_dir) / "asa9b3_lotfo_split.csv", index=False, encoding="utf-8-sig")
        return train_idx, test_idx, split_df
    return lotfo_split

def read_outputs(model_dir, heldout_family):
    summary_path = model_dir / "asa8d_summary.json"
    spec_path = model_dir / "asa8d_specificity_summary.csv"

    row = {
        "heldout_task_family": heldout_family,
        "status": "missing_outputs",
        "summary_path": str(summary_path),
    }

    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        row.update({
            "status": "ok",
            "verdict": summary.get("verdict"),
            "real_predmech_specificity": summary.get("main_metrics", {}).get("real_predmech_specificity"),
            "fixed_combo_specificity": summary.get("main_metrics", {}).get("fixed_combo_specificity"),
            "gain_over_fixed": summary.get("main_metrics", {}).get("gain_over_fixed"),
            "z_vs_shuffle_both": summary.get("main_metrics", {}).get("z_vs_shuffle_both"),
            "z_vs_shuffle_mech_policy": summary.get("main_metrics", {}).get("z_vs_shuffle_mech_policy"),
            "z_vs_shuffle_policy_label": summary.get("main_metrics", {}).get("z_vs_shuffle_policy_label"),
            "exceeds_shuffle_both_q95": summary.get("main_metrics", {}).get("exceeds_shuffle_both_q95"),
            "deltaU_pca_explained_variance": summary.get("deltaU_pca_explained_variance"),
            "n_policy_train_prompts": summary.get("n_policy_train_prompts"),
            "n_test_prompts": summary.get("n_test_prompts"),
            "real_mechanism_train_accuracy": summary.get("real_mechanism_audit", {}).get("train_accuracy"),
            "real_mechanism_train_macro_f1": summary.get("real_mechanism_audit", {}).get("train_macro_f1"),
            "real_policy_train_accuracy": summary.get("real_policy_audit", {}).get("train_accuracy"),
            "real_policy_train_macro_f1": summary.get("real_policy_audit", {}).get("train_macro_f1"),
        })

    if spec_path.exists():
        try:
            spec = pd.read_csv(spec_path)
            for ctrl in ["real_predmech_policy", "same_combo_fixed", "operator_only_fixed", "precursor_only_fixed", "random_gain_combo", "answer"]:
                s = spec[spec["control"] == ctrl]
                if len(s):
                    row[f"{ctrl}_specificity"] = float(s.iloc[0]["specificity"])
                    row[f"{ctrl}_target_abs_shift"] = float(s.iloc[0]["target_abs_shift"])
                    row[f"{ctrl}_nontarget_abs_shift"] = float(s.iloc[0]["nontarget_abs_shift"])
                    row[f"{ctrl}_target_hit_rate"] = float(s.iloc[0]["target_hit_rate"])
        except Exception as e:
            row["specificity_parse_error"] = repr(e)

    return row

# ============================================================
# MAIN
# ============================================================

def main():
    ROOT_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    engine_file = find_asa8d_file()
    print(f"[ASA-9B.3] Using ASA-8D engine: {engine_file}")
    engine = import_module_from_file(engine_file)

    patch_engine_for_six_mechanisms(engine)

    all_rows = []

    for heldout in TASK_FAMILIES_TO_HOLDOUT:
        print("\n" + "=" * 88)
        print(f"[ASA-9B.3] Override-aware LOTFO heldout = {heldout}")
        print("=" * 88)

        cfg = patch_config(engine, heldout)
        save_dir = Path(cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        engine.make_split = make_lotfo_split_func(heldout)

        with open(save_dir / "asa9b3_run_config.json", "w", encoding="utf-8") as f:
            json.dump({
                "model_key": MODEL_KEY,
                "model_path": MODEL_PATH,
                "heldout_task_family": heldout,
                "task_families": TASK_FAMILIES,
                "task_families_to_holdout": TASK_FAMILIES_TO_HOLDOUT,
                "mechanism_space": MECH_TO_ID_6,
                "n_graphs_per_family": N_GRAPHS_PER_FAMILY,
                "n_shuffles": N_SHUFFLES,
                "operator_layers": list(OPERATOR_LAYERS),
                "precursor_layers": list(PRECURSOR_LAYERS),
                "decision_layers": list(DECISION_LAYERS),
                "engine_file": str(engine_file),
            }, f, indent=2, ensure_ascii=False, default=to_jsonable)

        try:
            engine.main(cfg)
            row = read_outputs(save_dir, heldout)
            all_rows.append(row)
        except Exception as e:
            print(f"[ASA-9B.3][ERROR] heldout={heldout}: {repr(e)}")
            traceback.print_exc()
            row = {
                "heldout_task_family": heldout,
                "status": "error",
                "error": repr(e),
            }
            all_rows.append(row)
            with open(save_dir / "asa9b3_error.json", "w", encoding="utf-8") as f:
                json.dump(row, f, indent=2, ensure_ascii=False, default=to_jsonable)

    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(ROOT_SAVE_DIR / "asa9b3_cross_task_summary.csv", index=False, encoding="utf-8-sig")

    with open(ROOT_SAVE_DIR / "asa9b3_cross_task_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False, default=to_jsonable)

    ok_rows = [r for r in all_rows if r.get("status") == "ok"]
    n_pass_strong = sum(r.get("verdict") == "PASS_STRONG_VALIDATED_CLOSED_LOOP" for r in ok_rows)
    n_pass_lite = sum("PASS_LITE" in str(r.get("verdict", "")) for r in ok_rows)
    n_gain = sum("GAIN_OVER_FIXED" in str(r.get("verdict", "")) for r in ok_rows)

    final = {
        "audit": "ASA-9B.3 Probe-Binding Cross-Task Policy Audit",
        "model": MODEL_KEY,
        "task_families": TASK_FAMILIES,
        "task_families_to_holdout": TASK_FAMILIES_TO_HOLDOUT,
        "mechanism_space": MECH_TO_ID_6,
        "n_completed": len(ok_rows),
        "n_total": len(TASK_FAMILIES_TO_HOLDOUT),
        "n_pass_strong": int(n_pass_strong),
        "n_pass_lite": int(n_pass_lite),
        "n_gain_over_fixed_but_null_weak": int(n_gain),
        "n_shuffles": N_SHUFFLES,
        "summary_csv": "asa9b3_cross_task_summary.csv",
        "interpretation_rules": {
            "exception_improves": "Internal probe-binding readouts repair the remaining exception/override Lite bottleneck.",
            "temporal_remains_pass": "Temporal update remains transferable under finer mechanism space.",
            "still_fails_exception": "Need stronger relation-specific binding probes or explicit source-target-value probes.",
        },
    }

    with open(ROOT_SAVE_DIR / "asa9b3_final_verdict.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False, default=to_jsonable)

    print("\n[ASA-9B.3] Complete.")
    print(json.dumps(final, indent=2, ensure_ascii=False, default=to_jsonable))

if __name__ == "__main__":
    main()
