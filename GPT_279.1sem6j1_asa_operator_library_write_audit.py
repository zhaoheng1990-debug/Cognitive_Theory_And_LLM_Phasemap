# -*- coding: utf-8 -*-
r"""
SEM-6J.1: ASA-Operator Library Write Audit

Purpose
-------
SEM-6J smoke showed:
    naive prompt-delta operator write = FAIL/INCONCLUSIVE

This script moves SEM-6J.1 to the correct target:
    MemoryUnit write = Policy selects learned/operator-library action

It does NOT pretend that prompt-delta is a learned operator.
Instead it builds a route -> ASA-style operator action plan.

If a real ASA/DA-ASA operator library is available locally, it can be linked later.
If not, this script produces a controller-ready plan with explicit action labels:

    no_intervention
    stable_operator
    update_operator
    correction_operator
    suppress_override_operator
    order_precursor
    boundary_push
    address_gate
    residual_enrichment

The next live controller should implement these actions using learned ASA operators,
not raw hidden vector interpolation.

Inputs
------
sem6j_outputs\sem6j_policy_operator_write_plan.csv

Outputs
-------
sem6j1_outputs\sem6j1_asa_operator_write_plan.csv
sem6j1_outputs\sem6j1_asa_operator_summary.csv
sem6j1_outputs\sem6j1_verdict.json
sem6j1_outputs\sem6j1_operator_library_template.json

Run
---
python GPT_sem6j1_asa_operator_library_write_audit.py

Then inspect:
sem6j1_outputs\sem6j1_asa_operator_write_plan.csv

Theory
------
6J.1 freezes the negative result:
    ReflexiveWrite != hidden-vector interpolation
    ReflexiveWrite != prompt-delta additive steering

and tests:
    ReflexiveWrite = Q-selected policy/operator update
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


DEFAULT_ROOT = r"C:\Users\ZH\Desktop\AGI\python_script"
DEFAULT_6J_PLAN = rf"{DEFAULT_ROOT}\sem6j_outputs\sem6j_policy_operator_write_plan.csv"
DEFAULT_OUT_DIR = rf"{DEFAULT_ROOT}\sem6j1_outputs"
DEFAULT_OUT_PLAN = rf"{DEFAULT_OUT_DIR}\sem6j1_asa_operator_write_plan.csv"
DEFAULT_OUT_SUMMARY = rf"{DEFAULT_OUT_DIR}\sem6j1_asa_operator_summary.csv"
DEFAULT_OUT_VERDICT = rf"{DEFAULT_OUT_DIR}\sem6j1_verdict.json"
DEFAULT_LIBRARY_TEMPLATE = rf"{DEFAULT_OUT_DIR}\sem6j1_operator_library_template.json"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def infer_action(row: pd.Series) -> Tuple[str, str, str, float, float, str]:
    """
    Return:
        asa_action, action_stage, action_family, alpha, beta, rationale
    """
    route = str(row.get("write_route", ""))
    relation = str(row.get("relation_type", ""))
    target_op = str(row.get("target_operator", ""))
    cand_op = str(row.get("candidate_operator", ""))
    sel = str(row.get("_selection_family", ""))

    q = row.get("_learned_noleak_q", np.nan)
    oracle = row.get("oracle_utility", np.nan)

    def is_pos(x):
        try:
            return float(x) > 0
        except Exception:
            return False

    pos = is_pos(q) or is_pos(oracle)

    # Route-level conversion to learned operator actions.
    if route == "reject_as_null":
        return (
            "no_intervention",
            "none",
            "null",
            0.0,
            0.0,
            "null/random/artifact route; do not write",
        )

    if route == "write_address_gate":
        return (
            "address_gate",
            "commit_L23_L25",
            "addressing",
            0.0,
            0.05 if pos else 0.02,
            "commit-topk candidate should gate retrieval/address, not alter shape",
        )

    if route == "write_residual_mode":
        return (
            "residual_enrichment",
            "shape_L7_L19",
            "residual",
            0.02 if pos else 0.01,
            0.0,
            "same-family candidate should enrich residual modes with very weak action",
        )

    if route == "write_operator_prior":
        # Same operator other concept: transfer operator prior.
        # If candidate operator equals target operator, use stable_operator/target operator prior.
        return (
            "operator_prior_transfer",
            "operator_L15_L19",
            "operator_transfer",
            0.10 if pos else 0.05,
            0.02,
            "same-operator cross-concept candidate should update OperatorPrior_F via learned operator library",
        )

    if route == "trigger_correction_operator":
        # Same concept different operator: correction/action selection.
        # Try to infer operator action from candidate operator semantics.
        cop = norm(cand_op)
        top = norm(target_op)

        if "risk" in cop or "audit" in cop:
            action = "risk_audit_operator"
        elif "planning" in cop or "plan" in cop:
            action = "planning_operator"
        elif "counter" in cop:
            action = "counterexample_operator"
        elif "invariant" in cop:
            action = "invariant_search_operator"
        elif "closure" in cop:
            action = "closure_check_operator"
        elif "relation" in cop:
            action = "relation_mapping_operator"
        elif "mechanism" in cop:
            action = "mechanism_operator"
        elif "comparison" in cop:
            action = "comparison_operator"
        else:
            action = "correction_operator"

        return (
            action,
            "operator_L15_L19_plus_commit_L23_L25",
            "correction",
            0.08 if pos else 0.03,
            0.05 if pos else 0.02,
            "same-concept different-operator candidate should trigger correction/operator selection, not hidden replacement",
        )

    return (
        "policy_gate",
        "policy_L15_L19",
        "fallback",
        0.03 if pos else 0.0,
        0.02 if pos else 0.0,
        "fallback policy gate",
    )


def build_operator_library_template(path: Path):
    """
    This is intentionally a template: real learned ASA operators must be linked by
    actual local artifact paths in future.
    """
    library = {
        "version": "SEM-6J.1-operator-library-template",
        "status": "template_requires_learned_ASA_operator_artifacts",
        "note": (
            "Do not treat this JSON as learned operators. It defines action names, "
            "expected windows and required artifact slots for the next live controller."
        ),
        "actions": {
            "no_intervention": {
                "type": "null",
                "layers": [],
                "required_artifact": None,
            },
            "address_gate": {
                "type": "commit_gate",
                "layers": [23, 24, 25],
                "required_artifact": "commit/address gate or order precursor direction",
            },
            "residual_enrichment": {
                "type": "weak_residual",
                "layers": list(range(7, 20)),
                "required_artifact": "family residual-mode operator, not raw mu vector",
            },
            "operator_prior_transfer": {
                "type": "learned_operator_prior",
                "layers": list(range(15, 20)),
                "required_artifact": "ASA-style learned operator A_operator(H_l)",
            },
            "correction_operator": {
                "type": "learned_correction",
                "layers": list(range(15, 20)),
                "required_artifact": "Correction/Stable/Update operator from ASA/DA-ASA",
            },
            "risk_audit_operator": {
                "type": "learned_operator",
                "layers": list(range(15, 20)),
                "required_artifact": "RiskAudit operator",
            },
            "planning_operator": {
                "type": "learned_operator",
                "layers": list(range(15, 20)),
                "required_artifact": "Planning operator",
            },
            "counterexample_operator": {
                "type": "learned_operator",
                "layers": list(range(15, 20)),
                "required_artifact": "CounterExample operator",
            },
            "invariant_search_operator": {
                "type": "learned_operator",
                "layers": list(range(15, 20)),
                "required_artifact": "InvariantSearch operator",
            },
            "closure_check_operator": {
                "type": "learned_operator",
                "layers": list(range(15, 20)),
                "required_artifact": "ClosureCheck / closure repair operator",
            },
            "relation_mapping_operator": {
                "type": "learned_operator",
                "layers": list(range(15, 20)),
                "required_artifact": "RelationMapping operator",
            },
            "mechanism_operator": {
                "type": "learned_operator",
                "layers": list(range(15, 20)),
                "required_artifact": "MechanismExplanation operator",
            },
            "comparison_operator": {
                "type": "learned_operator",
                "layers": list(range(15, 20)),
                "required_artifact": "Comparison operator",
            },
            "policy_gate": {
                "type": "fallback_policy_gate",
                "layers": list(range(15, 20)),
                "required_artifact": "learned policy gate",
            },
            "order_precursor": {
                "type": "precursor",
                "layers": [17, 18, 19],
                "required_artifact": "ASA order precursor direction",
            },
            "boundary_push": {
                "type": "commit_correction",
                "layers": [23, 24, 25],
                "required_artifact": "DA-ASA boundary/order push direction",
            },
            "suppress_override_operator": {
                "type": "override_suppression",
                "layers": [18, 19],
                "required_artifact": "DA-ASA suppress_override operator",
            },
            "update_operator": {
                "type": "update_attractor",
                "layers": [15, 16, 17],
                "required_artifact": "DA-ASA to_update operator",
            },
            "stable_operator": {
                "type": "stable_attractor",
                "layers": list(range(15, 20)),
                "required_artifact": "ASA stable operator A_S(H_l)",
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(library, f, ensure_ascii=False, indent=2)
    return library


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=DEFAULT_6J_PLAN)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--out_plan", default=DEFAULT_OUT_PLAN)
    ap.add_argument("--out_summary", default=DEFAULT_OUT_SUMMARY)
    ap.add_argument("--out_verdict", default=DEFAULT_OUT_VERDICT)
    ap.add_argument("--library_template", default=DEFAULT_LIBRARY_TEMPLATE)
    args = ap.parse_args()

    plan_path = Path(args.plan)
    if not plan_path.exists():
        raise FileNotFoundError(f"SEM-6J plan not found: {plan_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(plan_path)
    required = [
        "_selection_family", "write_route", "relation_type",
        "target_operator", "candidate_operator"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Plan missing columns: {missing}. Columns={list(df.columns)}")

    actions = df.apply(infer_action, axis=1, result_type="expand")
    actions.columns = [
        "asa_action", "asa_stage", "asa_action_family",
        "asa_alpha", "asa_beta", "asa_rationale"
    ]
    out = pd.concat([df, actions], axis=1)

    out["sem6j1_write_hypothesis"] = (
        "MemoryUnit write should update Policy_F / OperatorPrior_F via learned ASA-style "
        "operator actions, not hidden-vector or prompt-delta interpolation."
    )

    out_plan = Path(args.out_plan)
    out_plan.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_plan, index=False, encoding="utf-8-sig")

    summary_rows = []
    for keys, sub in out.groupby(["_selection_family", "write_route", "asa_action", "asa_action_family"], dropna=False):
        sel, route, action, fam = keys
        row = {
            "selection_family": sel,
            "write_route": route,
            "asa_action": action,
            "asa_action_family": fam,
            "n": int(len(sub)),
            "mean_asa_alpha": float(pd.to_numeric(sub["asa_alpha"], errors="coerce").mean()),
            "mean_asa_beta": float(pd.to_numeric(sub["asa_beta"], errors="coerce").mean()),
        }
        if "oracle_utility" in sub.columns:
            row["mean_oracle_utility"] = float(pd.to_numeric(sub["oracle_utility"], errors="coerce").mean())
        if "_learned_noleak_q" in sub.columns:
            row["mean_q_score"] = float(pd.to_numeric(sub["_learned_noleak_q"], errors="coerce").mean())
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(["selection_family", "write_route", "asa_action"])
    summary.to_csv(args.out_summary, index=False, encoding="utf-8-sig")

    library = build_operator_library_template(Path(args.library_template))

    verdict = {
        "stage": "SEM-6J.1",
        "mode": "asa_operator_library_write_plan",
        "verdict": "PLAN_CREATED_NEEDS_LEARNED_ASA_OPERATOR_ARTIFACTS",
        "n_rows": int(len(out)),
        "out_plan": str(out_plan),
        "out_summary": str(args.out_summary),
        "operator_library_template": str(args.library_template),
        "selection_families": sorted(out["_selection_family"].dropna().unique().tolist()),
        "asa_actions": sorted(out["asa_action"].dropna().unique().tolist()),
        "notes": [
            "SEM-6J smoke test showed naive prompt-delta operator write is not sufficient.",
            "SEM-6J.1 converts routes into learned ASA-style operator action names.",
            "Next step is to link real ASA/DA-ASA operator artifacts or train a small operator library on current SEM families.",
            "Do not interpret this plan as a live PASS; it is a controller plan."
        ],
    }
    with open(args.out_verdict, "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)

    print("\n========== SEM-6J.1 ==========")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    print("\nSummary head:")
    print(summary.head(40).to_string(index=False))
    print(f"\nWrote: {out_plan}")


if __name__ == "__main__":
    main()
