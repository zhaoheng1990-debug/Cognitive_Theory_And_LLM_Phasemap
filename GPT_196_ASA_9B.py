
# ============================================================
# ASA-9B Driver: Cross-Task Generalization Audit
# Predicted-Mechanism Closed-Loop Trajectory Control
#
# This driver reuses the already validated ASA-8D engine and replaces
# only the dataset builder + split function.
#
# Goal:
#   ASA-9A showed cross-model matched-null validation.
#   ASA-9B asks whether the validated closed-loop control generalizes
#   across TASK FAMILIES, not merely across graph instances.
#
# Design:
#   Leave-One-Task-Family-Out (LOTFO)
#
#   For each held-out task family:
#       train policy / mechanism classifier / operators on all other families
#       test on the held-out family
#       run ASA-8D-style matched nulls:
#           real_predmech_policy
#           shuffle_mech_policy
#           shuffle_policy_label
#           shuffle_both
#           same_combo_fixed
#
# Default model:
#   Qwen only.
#
# Why Qwen first?
#   ASA-9A already confirmed the phenomenon across Qwen/Llama/Gemma.
#   ASA-9B should first isolate task generalization on the most stable
#   model before multiplying compute by 3 models.
#
# Required:
#   Put this file in the same folder as:
#       asa8d_matched_shuffle_closed_loop_validation_PATCHED.py
#   or:
#       GPT_194_ASA_8D.py
#
# Outputs:
#   asa9b_outputs/
#     asa9b_cross_task_summary.csv
#     asa9b_cross_task_summary.json
#     asa9b_final_verdict.json
#     heldout_<task_family>/asa8d_summary.json
#     heldout_<task_family>/asa8d_specificity_summary.csv
#     heldout_<task_family>/asa8d_shuffle_validation.csv
#
# Interpretation:
#   PASS_STRONG on all held-out task families:
#       cross-task validated closed-loop trajectory control.
#   Mixed:
#       task-family-specific control; inspect weak family and add better
#       mechanism features / prompt initialization features.
# ============================================================

import os
import sys
import json
import copy
import random
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# USER SETTINGS
# ============================================================

ROOT_SAVE_DIR = Path("./asa9b_lite_outputs")

MODEL_KEY = "qwen"
MODEL_PATH = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

# Qwen windows from ASA-8D / ASA-9A.
OPERATOR_LAYERS = tuple(range(15, 20))
PRECURSOR_LAYERS = (17, 18, 19)
DECISION_LAYERS = tuple(range(20, 26))

# Keep first pass manageable.
N_SHUFFLES = 5
BATCH_SIZE = 4

# Number of base graphs per task family.
# Total prompts = N_GRAPHS_PER_FAMILY * task_families * conditions_per_family
N_GRAPHS_PER_FAMILY = 24

ASA8D_CANDIDATE_FILES = [
    "asa8d_matched_shuffle_closed_loop_validation_PATCHED.py",
    "GPT_194_ASA_8D.py",
    "asa8d_matched_shuffle_closed_loop_validation.py",
]

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

MECH_TO_ID = {"stable": 0, "competition": 1, "closure": 2}

TASK_FAMILIES = [
    "relation_chain",
    "rule_application",
    "source_priority",
    "temporal_update",
    "exception_override",
]

# First-pass heldout list. Keep this short for fast validation.
# Recommended hardest tests:
#   exception_override: structural rule/exception transfer
#   temporal_update: update/overwrite transfer
TASK_FAMILIES_TO_HOLDOUT = [
    "exception_override",
    "temporal_update",
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
    pd.DataFrame(rows).to_csv(save_dir / "asa9b_label_tokenization_audit.csv", index=False, encoding="utf-8-sig")
    if len(chosen) < 12:
        raise RuntimeError(f"Need >=12 single-token labels; got {chosen}")
    return chosen[:12]

def make_prompt(task_family, condition, a, b, d, clean_label, conflict_label, aux_label, relation1, relation2):
    """
    Returns prompt text. Mechanism is assigned outside.
    All prompts ask for exactly one of clean_label / conflict_label.
    """

    if task_family == "relation_chain":
        if condition == "stable_clean":
            return "\n".join([
                "You are given a small relation graph.",
                f"Fact 1: {a} {relation1} {b}.",
                f"Fact 2: {b} {relation2} {clean_label}.",
                f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "stable_irrelevant":
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
        if condition == "competition_direct":
            return "\n".join([
                "You are given a small relation graph.",
                f"Fact 1: {a} {relation1} {b}.",
                f"Fact 2: {b} {relation2} {clean_label}.",
                f"Direct conflicting fact: {a} {relation2} {conflict_label}.",
                f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "closure_update":
            return "\n".join([
                "You are given a small relation graph.",
                f"Fact 1: {a} {relation1} {b}.",
                f"Old record: {b} {relation2} {clean_label}.",
                f"Updated record: {b} {relation2} {conflict_label}, not {clean_label}.",
                f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])

    if task_family == "rule_application":
        if condition == "stable_clean":
            return "\n".join([
                "Use the rule system below.",
                f"Rule: every item in group {b} receives label {clean_label}.",
                f"Fact: {a} is in group {b}.",
                f"Question: Which label applies to {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "stable_irrelevant":
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
        if condition == "competition_direct":
            return "\n".join([
                "Use the rule system below.",
                f"General rule: every item in group {b} receives label {clean_label}.",
                f"Direct statement: {a} receives label {conflict_label}.",
                f"Fact: {a} is in group {b}.",
                f"Question: Which label applies to {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "closure_update":
            return "\n".join([
                "Use the rule system below.",
                f"Old rule: every item in group {b} receives label {clean_label}.",
                f"New rule: every item in group {b} receives label {conflict_label}.",
                f"Fact: {a} is in group {b}.",
                f"Question: Which label applies to {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])

    if task_family == "source_priority":
        if condition == "stable_clean":
            return "\n".join([
                "Resolve the label using the trusted registry.",
                f"Trusted registry: {a} has label {clean_label}.",
                f"Question: Which label is correct for {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "stable_irrelevant":
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
        if condition == "competition_direct":
            return "\n".join([
                "Resolve the label using the records.",
                f"Main record: {a} has label {clean_label}.",
                f"Conflicting memo: {a} has label {conflict_label}.",
                f"Question: Which label is correct for {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "closure_update":
            return "\n".join([
                "Resolve the label using source priority.",
                f"Old registry: {a} had label {clean_label}.",
                f"Verified registry update: {a} now has label {conflict_label}.",
                f"Question: Which label is correct for {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])

    if task_family == "temporal_update":
        if condition == "stable_clean":
            return "\n".join([
                "Use the latest temporal fact.",
                f"At time T1, {a} was assigned label {clean_label}.",
                f"At time T2, the assignment remains {clean_label}.",
                f"Question: What is the current label of {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "stable_irrelevant":
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
        if condition == "competition_direct":
            return "\n".join([
                "Use the temporal notes.",
                f"Earlier today, {a} was assigned label {clean_label}.",
                f"A direct later note says {a} has label {conflict_label}.",
                f"Question: What is the current label of {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "closure_update":
            return "\n".join([
                "Use the latest temporal update.",
                f"Yesterday, {a} had label {clean_label}.",
                f"Today, {a} was updated to label {conflict_label}.",
                f"Question: What is the current label of {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])

    if task_family == "exception_override":
        if condition == "stable_clean":
            return "\n".join([
                "Use the rule and exception list.",
                f"General rule: items linked to {b} receive label {clean_label}.",
                f"Fact: {a} is linked to {b}.",
                f"No exception applies to {a}.",
                f"Question: Which label applies to {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "stable_irrelevant":
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
        if condition == "competition_direct":
            return "\n".join([
                "Use the rule and exception list.",
                f"General rule: items linked to {b} receive label {clean_label}.",
                f"Direct exception candidate: {a} receives label {conflict_label}.",
                f"Fact: {a} is linked to {b}.",
                f"Question: Which label applies to {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])
        if condition == "closure_update":
            return "\n".join([
                "Use the rule and exception list.",
                f"General rule: items linked to {b} receive label {clean_label}.",
                f"Exception: {a} is a special case and receives label {conflict_label}.",
                f"Fact: {a} is linked to {b}.",
                f"Question: Which label applies to {a}: {clean_label} or {conflict_label}?",
                "Answer with exactly one word:",
            ])

    raise ValueError(f"Unknown task_family={task_family}, condition={condition}")

# IMPORTANT:
# ASA-8D engine expects condition == "clean" to build clean anchors.
# Therefore the stable anchor must be named exactly "clean".
CONDITION_TO_MECH = {
    "clean": "stable",
    "irrelevant": "stable",
    "competition_balanced": "competition",
    "direct_conflict": "competition",
    "closure_update": "closure",
}

CONDITIONS = list(CONDITION_TO_MECH.keys())

def normalize_condition_for_prompt(condition):
    """Map ASA-8D-compatible condition names to internal prompt templates."""
    return {
        "clean": "stable_clean",
        "irrelevant": "stable_irrelevant",
        "competition_balanced": "competition_balanced",
        "direct_conflict": "competition_direct",
        "closure_update": "closure_update",
    }[condition]

def build_cross_task_dataset(tokenizer, cfg, save_dir):
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
                mech = CONDITION_TO_MECH[cond]
                text = make_prompt(task_family, normalize_condition_for_prompt(cond), a, b, d, clean_label, conflict_label, aux_label, r1, r2)
                rows.append({
                    "prompt_id": f"{task_family}_g{local_g:03d}_{cond}",
                    "graph_id": int(graph_id),
                    "task_family": task_family,
                    "local_graph_id": int(local_g),
                    "condition": cond,
                    "mechanism": mech,
                    "mechanism_id": MECH_TO_ID[mech],
                    "clean_label": clean_label,
                    "conflict_label": conflict_label,
                    "clean_token_id": clean_id,
                    "conflict_token_id": conflict_id,
                    "text": text,
                })
            graph_id += 1

    df = pd.DataFrame(rows)
    df.to_csv(save_dir / "asa9b_dataset.csv", index=False, encoding="utf-8-sig")
    return df

# ============================================================
# DRIVER PATCHING
# ============================================================

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

    # Engine still uses this in config; dataset builder ignores it.
    cfg.n_graphs = N_GRAPHS_PER_FAMILY * len(TASK_FAMILIES)

    # Fast first-pass policy grid: no-control, weak, fixed best combo, operator-only.
    # This greatly reduces the silent policy-train dose grid.
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
            "split": np.where(test_mask, "test", "train"),
        })
        split_df.to_csv(Path(save_dir) / "asa9b_lotfo_split.csv", index=False, encoding="utf-8-sig")
        return train_idx, test_idx, split_df
    return lotfo_split

def read_outputs(model_dir, heldout_family):
    summary_path = model_dir / "asa8d_summary.json"
    verdict_path = model_dir / "asa8d_verdict.csv"
    spec_path = model_dir / "asa8d_specificity_summary.csv"
    shuffle_path = model_dir / "asa8d_shuffle_validation.csv"

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
    print(f"[ASA-9B] Using ASA-8D engine: {engine_file}")
    engine = import_module_from_file(engine_file)

    # Monkey-patch dataset builder globally.
    engine.build_dataset = build_cross_task_dataset

    all_rows = []

    for heldout in TASK_FAMILIES_TO_HOLDOUT:
        print("\n" + "=" * 88)
        print(f"[ASA-9B] Leave-One-Task-Family-Out: heldout = {heldout}")
        print("=" * 88)

        cfg = patch_config(engine, heldout)
        save_dir = Path(cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Monkey-patch split function for this heldout family.
        engine.make_split = make_lotfo_split_func(heldout)

        with open(save_dir / "asa9b_run_config.json", "w", encoding="utf-8") as f:
            json.dump({
                "model_key": MODEL_KEY,
                "model_path": MODEL_PATH,
                "heldout_task_family": heldout,
                "task_families": TASK_FAMILIES,
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
            import traceback
            traceback.print_exc()
            print(f"[ASA-9B][ERROR] heldout={heldout}: {repr(e)}")
            row = {
                "heldout_task_family": heldout,
                "status": "error",
                "error": repr(e),
            }
            all_rows.append(row)
            with open(save_dir / "asa9b_error.json", "w", encoding="utf-8") as f:
                json.dump(row, f, indent=2, ensure_ascii=False, default=to_jsonable)

    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(ROOT_SAVE_DIR / "asa9b_cross_task_summary.csv", index=False, encoding="utf-8-sig")

    with open(ROOT_SAVE_DIR / "asa9b_cross_task_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False, default=to_jsonable)

    ok_rows = [r for r in all_rows if r.get("status") == "ok"]
    n_pass_strong = sum(r.get("verdict") == "PASS_STRONG_VALIDATED_CLOSED_LOOP" for r in ok_rows)
    n_pass_lite = sum("PASS_LITE" in str(r.get("verdict", "")) for r in ok_rows)
    n_gain = sum("GAIN_OVER_FIXED" in str(r.get("verdict", "")) for r in ok_rows)

    final = {
        "audit": "ASA-9B Cross-Task Generalization Audit",
        "model": MODEL_KEY,
        "task_families": TASK_FAMILIES,
        "n_completed": len(ok_rows),
        "n_total": len(TASK_FAMILIES_TO_HOLDOUT),
        "n_pass_strong": int(n_pass_strong),
        "n_pass_lite": int(n_pass_lite),
        "n_gain_over_fixed_but_null_weak": int(n_gain),
        "n_shuffles": N_SHUFFLES,
        "summary_csv": "asa9b_cross_task_summary.csv",
        "interpretation_rules": {
            "all_pass_strong": "Cross-task validated predicted-mechanism closed-loop trajectory control.",
            "mixed_pass": "Closed-loop control transfers only to some task families; inspect weak heldout family.",
            "mostly_fail": "ASA-8D control is task-family-specific under current features/policy.",
        },
    }

    with open(ROOT_SAVE_DIR / "asa9b_final_verdict.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False, default=to_jsonable)

    print("\n[ASA-9B] Complete.")
    print(json.dumps(final, indent=2, ensure_ascii=False, default=to_jsonable))

if __name__ == "__main__":
    main()
