# -*- coding: utf-8 -*-
"""
DA-ASA-2K.1 Expanded Condition Family RealRun Builder

Purpose
-------
Convert DA-ASA-2K candidate prompt templates into concrete controlled C/E samples.

This script does NOT run the model. It builds the expanded sample table needed for the
next real DA-ASA pipeline stages:
  1) baseline scores
  2) DSTA signed / transport / attractor features
  3) policy_results table
  4) effect vectors
  5) LPF clustering / label assignment
  6) DenseDSTA -> LPF CV

Input expected near this script or under ROOT:
  da_asa2k_outputs/da_asa2k_candidate_prompts.csv
or:
  da_asa2k_candidate_prompts.csv

Outputs:
  da_asa2k1_outputs/da_asa2k1_expanded_samples.csv
  da_asa2k1_outputs/da_asa2k1_expanded_samples_lite.csv
  da_asa2k1_outputs/da_asa2k1_condition_summary.csv
  da_asa2k1_outputs/da_asa2k1_family_summary.csv
  da_asa2k1_outputs/da_asa2k1_split_plan.csv
  da_asa2k1_outputs/da_asa2k1_diagnostics.json
  da_asa2k1_outputs/da_asa2k1_next_pipeline_seed.md
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "da_asa2k1_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_PROMPT_NAMES = [
    "da_asa2k_candidate_prompts.csv",
    str(Path("da_asa2k_outputs") / "da_asa2k_candidate_prompts.csv"),
]

TARGET_CONDITIONS_PER_FAMILY = 8
TARGET_GRAPHS_PER_CONDITION = 12


def find_file(names):
    for name in names:
        p = ROOT / name
        if p.exists():
            return p
    for name in names:
        hits = list(ROOT.rglob(Path(name).name))
        if hits:
            # Prefer paths containing da_asa2k_outputs if available.
            hits = sorted(hits, key=lambda x: ("da_asa2k_outputs" not in str(x), len(str(x))))
            return hits[0]
    return None


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())


def side_for(row) -> str:
    # Alternate correct side to avoid side-label shortcut.
    v = int(row.get("variant_id", 1))
    return "C" if v % 2 == 1 else "E"


def swap_by_side(correct: str, incorrect: str, gold_side: str) -> Tuple[str, str]:
    if gold_side == "C":
        return correct, incorrect
    return incorrect, correct


def stable_case(condition: str, v: int) -> Tuple[str, str, str, str]:
    obj = f"mira-{v:02d}"
    color = ["blue", "green", "orange", "silver", "violet", "amber"][v % 6]
    wrong = ["red", "black", "yellow", "white", "brown", "pink"][v % 6]
    if "numeric" in condition:
        val = 20 + v
        return (
            f"In the reference card, the calibrated value of {obj} is exactly {val} units. No later sentence changes this value.",
            f"{obj} has calibrated value {val} units.",
            f"{obj} has calibrated value {val + 3} units.",
            "protect_no_intervention",
        )
    if "definition" in condition:
        return (
            f"Local glossary: in this task, the term {obj} means a small mapping token used for testing. The glossary is stable and contains no exception.",
            f"{obj} means a small mapping token used for testing.",
            f"{obj} means a weather instrument.",
            "protect_no_intervention",
        )
    if "minor_noise" in condition:
        return (
            f"Fact card: {obj} is assigned the color {color}. The following unrelated sentence about rain is noise and does not change the assignment.",
            f"{obj} is assigned {color}.",
            f"{obj} is assigned {wrong}.",
            "protect_no_intervention",
        )
    return (
        f"Context: The stable record says {obj} is assigned {color}. The record is final, direct, and has no override.",
        f"{obj} is assigned {color}.",
        f"{obj} is assigned {wrong}.",
        "protect_no_intervention",
    )


def competition_case(condition: str, v: int) -> Tuple[str, str, str, str]:
    item = f"naru-{v:02d}"
    a = ["north", "east", "left", "upper", "alpha", "primary"][v % 6]
    b = ["south", "west", "right", "lower", "beta", "secondary"][v % 6]
    if "late_disambiguation" in condition:
        ctx = (
            f"Two notes discuss {item}. Early note: {item} maps to {b}. Later disambiguation note, marked FINAL: {item} maps to {a}. "
            f"Use the FINAL note when notes conflict."
        )
        return ctx, f"{item} maps to {a}.", f"{item} maps to {b}.", "late_disambiguation"
    if "alias" in condition:
        ctx = (
            f"Alias table: {item} and alias-{v:02d} look similar, but only {item} maps to {a}. alias-{v:02d} maps to {b}. "
            f"The question asks about {item}, not the alias."
        )
        return ctx, f"{item} maps to {a}.", f"{item} maps to {b}.", "alias_conflict_resolution"
    if "category" in condition:
        ctx = (
            f"Category rule: objects with tag K are in group {a}; objects with tag M are in group {b}. {item} has tag K."
        )
        return ctx, f"{item} belongs to group {a}.", f"{item} belongs to group {b}.", "category_boundary_resolution"
    ctx = (
        f"Evidence A says {item} maps to {a}. Evidence B says {item} maps to {b}. "
        f"The source policy says Evidence A has priority for this question."
    )
    return ctx, f"{item} maps to {a}.", f"{item} maps to {b}.", "competition_resolution"


def closure_case(condition: str, v: int) -> Tuple[str, str, str, str]:
    key = f"tavo-{v:02d}"
    old = ["red", "square", "level-1", "cold", "outer", "small"][v % 6]
    new = ["blue", "circle", "level-2", "warm", "inner", "large"][v % 6]
    if "temporal" in condition:
        ctx = (
            f"Archive entry: {key} was {old} on Monday. Update entry: {key} became {new} on Tuesday. "
            f"The question asks for the latest state."
        )
        return ctx, f"{key} is {new} now.", f"{key} is {old} now.", "temporal_update"
    if "authority" in condition:
        ctx = (
            f"Draft note says {key} is {old}. Verified authority note says {key} is {new}. "
            f"Use the verified authority note when draft and authority disagree."
        )
        return ctx, f"{key} is {new}.", f"{key} is {old}.", "authority_update"
    if "multi_hop" in condition:
        mid = f"bridge-{v:02d}"
        ctx = (
            f"Rule 1: {key} points to {mid}. Rule 2: {mid} points to {new}. "
            f"Old shortcut saying {key} points to {old} is marked obsolete."
        )
        return ctx, f"{key} ultimately points to {new}.", f"{key} ultimately points to {old}.", "multi_hop_closure"
    if "definition" in condition:
        ctx = (
            f"Definition update: within this task, {key} now means {new}. The older meaning {old} is deprecated."
        )
        return ctx, f"{key} means {new}.", f"{key} means {old}.", "definition_update"
    ctx = (
        f"Base statement: {key} was {old}. Correction: for this question, replace {old} with {new}. "
        f"Apply the correction before answering."
    )
    return ctx, f"{key} is {new}.", f"{key} is {old}.", "closure_update"


def override_case(condition: str, v: int) -> Tuple[str, str, str, str]:
    key = f"vexa-{v:02d}"
    base = ["open", "valid", "included", "north", "group A", "enabled"][v % 6]
    exc = ["closed", "invalid", "excluded", "south", "group B", "disabled"][v % 6]
    if "negation" in condition:
        ctx = (
            f"Default rule: {key} is {base}. Explicit negation: {key} is not {base}; it is {exc}. "
            f"Explicit negation overrides the default."
        )
        return ctx, f"{key} is {exc}.", f"{key} is {base}.", "negation_override"
    if "nested" in condition:
        ctx = (
            f"Rule A: all test items are {base}. Rule B: items tagged rare are {exc}. Rule C: {key} is tagged rare. "
            f"Use the most specific applicable rule."
        )
        return ctx, f"{key} is {exc}.", f"{key} is {base}.", "nested_exception"
    if "priority" in condition:
        ctx = (
            f"Low-priority rule says {key} is {base}. High-priority rule says {key} is {exc}. "
            f"High-priority rule wins."
        )
        return ctx, f"{key} is {exc}.", f"{key} is {base}.", "priority_override"
    if "local_dictionary" in condition:
        ctx = (
            f"Global dictionary maps {key} to {base}. Local dictionary for this prompt maps {key} to {exc}. "
            f"Use the local dictionary."
        )
        return ctx, f"{key} maps to {exc}.", f"{key} maps to {base}.", "local_override"
    ctx = (
        f"General rule says {key} is {base}. Exception clause says this exact item, {key}, is {exc}. "
        f"Exceptions override general rules."
    )
    return ctx, f"{key} is {exc}.", f"{key} is {base}.", "exception_override"


def hallucination_case(condition: str, v: int) -> Tuple[str, str, str, str]:
    ent = f"zorel-{v:02d}"
    false_claim = ["invented medal", "hidden archive", "secret bridge", "third moon", "phantom award", "nonexistent treaty"][v % 6]
    if "false_premise" in condition:
        ctx = (
            f"The question assumes {ent} received a {false_claim}. The context provides no evidence that this event happened. "
            f"Do not accept unsupported premises."
        )
    elif "unsupported_citation" in condition:
        ctx = (
            f"A citation-like tag claims {ent} has a {false_claim}, but the tag is marked unverified and no source text supports it."
        )
    elif "unknown_entity" in condition:
        ctx = (
            f"{ent} is introduced only as a placeholder name. No attributes are provided."
        )
    elif "impossible_relation" in condition:
        ctx = (
            f"The proposed relation says {ent} is both before and after itself in the same strict ordering. This relation is inconsistent."
        )
    else:
        ctx = (
            f"Sparse context: {ent} is mentioned, but no verified property is given. Avoid adding unsupported details."
        )
    return ctx, "The requested claim is unsupported by the context.", f"{ent} definitely has the {false_claim}.", "unsupported_or_refuse"


def build_case(row) -> Dict[str, str]:
    fam = str(row["condition_family"])
    cond = str(row["condition"])
    v = int(row["variant_id"])
    if fam == "stable_protect":
        ctx, correct, incorrect, mechanism = stable_case(cond, v)
        expected_family = "PROTECT_OR_NOOP"
    elif fam == "competition":
        ctx, correct, incorrect, mechanism = competition_case(cond, v)
        expected_family = "RESOLVE_COMPETITION"
    elif fam == "closure_update":
        ctx, correct, incorrect, mechanism = closure_case(cond, v)
        expected_family = "APPLY_CLOSURE_UPDATE"
    elif fam == "override_exception":
        ctx, correct, incorrect, mechanism = override_case(cond, v)
        expected_family = "APPLY_OVERRIDE_EXCEPTION"
    elif fam == "hallucination_like":
        ctx, correct, incorrect, mechanism = hallucination_case(cond, v)
        expected_family = "SUPPRESS_UNSUPPORTED_CLOSURE"
    else:
        ctx, correct, incorrect, mechanism = stable_case(cond, v)
        expected_family = "UNKNOWN"

    gold = side_for(row)
    c_ans, e_ans = swap_by_side(correct, incorrect, gold)
    prompt = (
        "You are solving a controlled C/E diagnostic item.\n"
        "Use only the context below. Do not use outside knowledge.\n\n"
        f"Condition family: {fam}\n"
        f"Condition: {cond}\n"
        f"Context: {ctx}\n\n"
        f"Option C: {c_ans}\n"
        f"Option E: {e_ans}\n\n"
        "Answer with exactly one letter: C or E."
    )
    return {
        "prompt": prompt,
        "C_answer": c_ans,
        "E_answer": e_ans,
        "gold_label": gold,
        "gold_answer": correct,
        "distractor_answer": incorrect,
        "mechanism_template": mechanism,
        "expected_policy_family": expected_family,
    }


def main():
    in_path = find_file(CANDIDATE_PROMPT_NAMES)
    if in_path is None:
        raise FileNotFoundError(
            "Could not find da_asa2k_candidate_prompts.csv. Put it next to this script or under da_asa2k_outputs/."
        )
    cand = pd.read_csv(in_path)
    required = {"graph_id", "condition_family", "condition", "variant_id"}
    missing = sorted(required - set(cand.columns))
    if missing:
        raise KeyError(f"candidate prompts missing columns: {missing}; columns={list(cand.columns)}")

    rows = []
    for _, r in cand.iterrows():
        d = r.to_dict()
        case = build_case(r)
        d.update(case)
        # Keep original template as provenance, but use concrete prompt as active prompt.
        d["template_prompt_from_2k"] = r.get("prompt", "")
        d["source_stage"] = "DA-ASA-2K.1"
        d["is_concrete_sample"] = True
        rows.append(d)

    df = pd.DataFrame(rows)

    # Stable ordering and useful columns first.
    first_cols = [
        "graph_id", "condition_family", "condition", "variant_id",
        "prompt", "C_answer", "E_answer", "gold_label", "gold_answer", "distractor_answer",
        "expected_policy_family", "mechanism_template", "is_concrete_sample", "source_stage",
    ]
    other_cols = [c for c in df.columns if c not in first_cols]
    df = df[first_cols + other_cols]

    # Validation.
    per_cond = (
        df.groupby(["condition_family", "condition"], as_index=False)
        .agg(
            n_rows=("graph_id", "count"),
            n_unique_graphs=("graph_id", "nunique"),
            gold_C=("gold_label", lambda x: int((x == "C").sum())),
            gold_E=("gold_label", lambda x: int((x == "E").sum())),
            expected_policy_family=("expected_policy_family", "first"),
        )
    )
    per_family = (
        df.groupby("condition_family", as_index=False)
        .agg(
            n_conditions=("condition", "nunique"),
            n_rows=("graph_id", "count"),
            n_unique_graphs=("graph_id", "nunique"),
            gold_C=("gold_label", lambda x: int((x == "C").sum())),
            gold_E=("gold_label", lambda x: int((x == "E").sum())),
        )
    )

    split_rows = []
    for fam, sub in df.groupby("condition_family"):
        conds = sorted(sub["condition"].unique().tolist())
        for i, cond in enumerate(conds):
            split_rows.append({
                "split_type": "leave_one_condition",
                "holdout_family": fam,
                "holdout_condition": cond,
                "n_test_rows": int((df["condition"] == cond).sum()),
                "n_train_rows": int((df["condition"] != cond).sum()),
            })
        split_rows.append({
            "split_type": "leave_one_family",
            "holdout_family": fam,
            "holdout_condition": "*",
            "n_test_rows": int((df["condition_family"] == fam).sum()),
            "n_train_rows": int((df["condition_family"] != fam).sum()),
        })
    split_plan = pd.DataFrame(split_rows)

    # Lite table for baseline/DSTA scripts.
    lite_cols = [
        "graph_id", "condition_family", "condition", "variant_id", "prompt",
        "C_answer", "E_answer", "gold_label", "expected_policy_family", "mechanism_template",
    ]
    lite = df[lite_cols].copy()

    out_full = OUT_DIR / "da_asa2k1_expanded_samples.csv"
    out_lite = OUT_DIR / "da_asa2k1_expanded_samples_lite.csv"
    out_cond = OUT_DIR / "da_asa2k1_condition_summary.csv"
    out_fam = OUT_DIR / "da_asa2k1_family_summary.csv"
    out_split = OUT_DIR / "da_asa2k1_split_plan.csv"
    out_diag = OUT_DIR / "da_asa2k1_diagnostics.json"
    out_seed = OUT_DIR / "da_asa2k1_next_pipeline_seed.md"

    df.to_csv(out_full, index=False, encoding="utf-8-sig")
    lite.to_csv(out_lite, index=False, encoding="utf-8-sig")
    per_cond.to_csv(out_cond, index=False, encoding="utf-8-sig")
    per_family.to_csv(out_fam, index=False, encoding="utf-8-sig")
    split_plan.to_csv(out_split, index=False, encoding="utf-8-sig")

    diagnostics = {
        "stage": "DA-ASA-2K.1",
        "purpose": "Build concrete expanded condition-family C/E samples for TransferableLPF real run",
        "verdict": "EXPANDED_SAMPLE_TABLE_READY",
        "root": str(ROOT),
        "output_dir": str(OUT_DIR),
        "input_candidate_prompts": str(in_path),
        "n_rows": int(len(df)),
        "n_conditions": int(df["condition"].nunique()),
        "n_condition_families": int(df["condition_family"].nunique()),
        "graphs_per_condition_min": int(per_cond["n_rows"].min()),
        "graphs_per_condition_max": int(per_cond["n_rows"].max()),
        "target_conditions_per_family": TARGET_CONDITIONS_PER_FAMILY,
        "target_graphs_per_condition": TARGET_GRAPHS_PER_CONDITION,
        "family_condition_counts": per_family.set_index("condition_family")["n_conditions"].to_dict(),
        "family_row_counts": per_family.set_index("condition_family")["n_rows"].to_dict(),
        "gold_label_counts": df["gold_label"].value_counts().to_dict(),
        "outputs": {
            "expanded_samples": str(out_full),
            "expanded_samples_lite": str(out_lite),
            "condition_summary": str(out_cond),
            "family_summary": str(out_fam),
            "split_plan": str(out_split),
            "next_pipeline_seed": str(out_seed),
        },
        "next_stage": "Run baseline scoring, DSTA feature extraction, policy_results, effect vectors, LPF assignment, then DenseDSTA->LPF CV.",
    }
    out_diag.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")

    seed = f"""# DA-ASA-2K.1 Next Pipeline Seed\n\n## Current status\n\nDA-ASA-2K.1 generated a concrete expanded condition-family C/E sample table.\n\nVerdict:\n\n```text\nEXPANDED_SAMPLE_TABLE_READY\n```\n\n## Paths\n\nRoot:\n\n```text\n{ROOT}\n```\n\nOutput directory:\n\n```text\n{OUT_DIR}\n```\n\nPrimary expanded sample table:\n\n```text\n{out_lite}\n```\n\nFull provenance table:\n\n```text\n{out_full}\n```\n\nSplit plan:\n\n```text\n{out_split}\n```\n\n## Dataset summary\n\n```json\n{json.dumps({k:v for k,v in diagnostics.items() if k in ['n_rows','n_conditions','n_condition_families','family_condition_counts','family_row_counts','gold_label_counts']}, ensure_ascii=False, indent=2)}\n```\n\n## Next real-run sequence\n\nUse `da_asa2k1_expanded_samples_lite.csv` as the prompt input table.\n\nRequired downstream stages:\n\n```text\n1. Baseline C/E logit scoring\n2. DSTA signed / transport / attractor feature extraction\n3. Policy_results table generation across DA-ASA action set\n4. Policy effect vectors\n5. LPF clustering or assignment using 2J topology\n6. DenseDSTA -> LPF CV\n```\n\n## Main criterion\n\nCompare against 2J.1 baselines:\n\n```text\nleave-one-condition MacroF1 baseline: 0.5397\nleave-one-family MacroF1 baseline:    0.3824\n```\n\nA strong 2K.2 result requires both metrics to improve after condition-family expansion.\n"""
    out_seed.write_text(seed, encoding="utf-8")

    print("=" * 80)
    print("DA-ASA-2K.1 Expanded Condition Family RealRun Builder")
    print("=" * 80)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print("\n[OUT]", OUT_DIR)


if __name__ == "__main__":
    main()
