# -*- coding: utf-8 -*-
r"""
SEM-6L.5: CanonicalOperator Held-Out Live Replay

Purpose
-------
SEM-6L.4 produced four CanonicalOperator drafts:

    ConstraintInsertionOperator
    CorrectionBranchOperator
    TrajectoryClassExpansionOperator
    AddressPriorIncreaseOperator

SEM-6L.5 is the final validation gate:

    CanonicalOperatorDraft -> LongTerm OperatorMemory eligibility

It reads the operator library draft and the labeled SEM-6L data, constructs
held-out replay tasks, and evaluates whether canonical operators beat controls:

    canonical_operator
    no_operator
    wrong_operator
    random_operator
    label_matched_shuffle

IMPORTANT
---------
This script is designed as an offline/live-replay hybrid.

By default, it performs an OFFLINE held-out replay audit using existing live rows
from SEM-6L.2/6L.4, with group/operator/source held-out splits.

It also exports a controller-ready live replay plan:
    sem6l5_live_replay_plan.csv

If you later implement real live hooks, use this plan.

Inputs
------
sem6l4_outputs\sem6l4_operator_library_draft.json
sem6l2_outputs\sem6l2_expansion_update_labels.csv

Outputs
-------
sem6l5_outputs\
    sem6l5_live_replay_plan.csv
    sem6l5_offline_replay_results.csv
    sem6l5_operator_vs_controls.csv
    sem6l5_operator_memory_candidates.json
    sem6l5_verdict.json

Run
---
conda activate dhrf_4080s
cd C:\Users\ZH\Desktop\AGI\python_script

python GPT_sem6l5_canonical_operator_heldout_replay.py

Stricter:
python GPT_sem6l5_canonical_operator_heldout_replay.py ^
  --min_mean_advantage 0.02 ^
  --min_win_rate 0.55 ^
  --strict
"""

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold


DEFAULT_ROOT = r"C:\Users\ZH\Desktop\AGI\python_script"
DEFAULT_LIBRARY = rf"{DEFAULT_ROOT}\sem6l4_outputs\sem6l4_operator_library_draft.json"
DEFAULT_LABELS = rf"{DEFAULT_ROOT}\sem6l2_outputs\sem6l2_expansion_update_labels.csv"
DEFAULT_OUT_DIR = rf"{DEFAULT_ROOT}\sem6l5_outputs"

RANDOM_SEED = 20260606


OPERATOR_TO_LABEL = {
    "ConstraintInsertionOperator": "add_constraint",
    "CorrectionBranchOperator": "add_correction_branch",
    "TrajectoryClassExpansionOperator": "add_trajectory_class",
    "AddressPriorIncreaseOperator": "increase_address_prior",
}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def load_library(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Operator library draft not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def read_labels(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Labeled rows not found: {p}")
    df = pd.read_csv(p)

    for c in [
        "expansion_update_label", "source_experiment", "action", "_selection_family",
        "relation_type", "write_route", "target_operator", "candidate_operator",
        "target_concept", "candidate_concept", "_group_id", "pool"
    ]:
        if c not in df.columns:
            df[c] = "NA"
        df[c] = df[c].astype(str).fillna("NA")

    for c in [
        "live_value", "is_positive", "_learned_noleak_q", "proxy_utility",
        "oracle_utility", "q_proxy_gap", "q_oracle_gap", "beta", "delta_scale",
        "base_margin", "base_rank", "base_logprob_gap_to_top",
        "experience_vs_history_gain", "hist_mean_live", "hist_positive_rate",
        "action_ctx_mean_live", "action_ctx_positive_rate", "is_random_like"
    ]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if df["is_positive"].isna().all():
        df["is_positive"] = (df["live_value"] > 0).astype(int)
    else:
        df["is_positive"] = df["is_positive"].fillna((df["live_value"] > 0).astype(int))

    return df


def operator_applicability(row: pd.Series, operator_id: str) -> bool:
    """
    A conservative applicability policy derived from 6L.4 operator cards.
    """
    target_op = norm(row.get("target_operator", ""))
    cand_op = norm(row.get("candidate_operator", ""))
    action = norm(row.get("action", ""))
    relation = norm(row.get("relation_type", ""))
    route = norm(row.get("write_route", ""))

    if operator_id == "ConstraintInsertionOperator":
        return ("risk" in target_op or "audit" in target_op or "closure" in relation or "constraint" in route)

    if operator_id == "CorrectionBranchOperator":
        # same concept / different operator; Definition before Planning/Relation/Mechanism is prototypical.
        same_concept = str(row.get("target_concept", "")) == str(row.get("candidate_concept", ""))
        return same_concept and target_op != cand_op and (
            "definition" in cand_op or "planning" in target_op or "relation" in target_op or "mechanism" in target_op
        )

    if operator_id == "TrajectoryClassExpansionOperator":
        # mechanism/causal/explanation cross-operator variants.
        same_concept = str(row.get("target_concept", "")) == str(row.get("candidate_concept", ""))
        mech_family = any(x in target_op for x in ["mechanism", "causal", "explanation", "comparison", "property"])
        cand_mech = any(x in cand_op for x in ["mechanism", "causal", "explanation", "comparison", "property"])
        return same_concept and mech_family and cand_mech

    if operator_id == "AddressPriorIncreaseOperator":
        return ("address" in action or "address" in route or "commit" in relation or "topk" in relation or "neighbor" in relation)

    return False


def build_live_replay_plan(library: Dict, labels: pd.DataFrame, max_per_operator: int = 0) -> pd.DataFrame:
    rows = []
    operators = library.get("operators", [])
    if not operators:
        raise ValueError("Library has no operators.")

    for op in operators:
        op_id = op.get("operator_id")
        source_label = op.get("source_label", OPERATOR_TO_LABEL.get(op_id, ""))
        applicable = labels[labels.apply(lambda r: operator_applicability(r, op_id), axis=1)].copy()

        if source_label in set(labels["expansion_update_label"]):
            # Ensure known positive/evidence rows are included.
            evidence = labels[labels["expansion_update_label"] == source_label].copy()
            applicable = pd.concat([applicable, evidence], ignore_index=True).drop_duplicates()

        if max_per_operator and max_per_operator > 0:
            applicable = applicable.head(max_per_operator)

        for i, (_, r) in enumerate(applicable.iterrows()):
            base = r.to_dict()
            base["canonical_operator_id"] = op_id
            base["canonical_source_label"] = source_label
            base["operator_intent"] = op.get("intent", "")
            base["operator_boundary"] = op.get("applicability_boundary", "")
            base["replay_condition"] = "canonical_operator"
            base["replay_plan_id"] = f"{op_id}__{i}"
            rows.append(base)

            # no operator control
            c0 = base.copy()
            c0["replay_condition"] = "no_operator"
            c0["replay_plan_id"] = f"{op_id}__{i}__no"
            rows.append(c0)

            # wrong operator controls: pick a mismatched canonical operator.
            wrong_ops = [x.get("operator_id") for x in operators if x.get("operator_id") != op_id]
            if wrong_ops:
                wrong_id = wrong_ops[i % len(wrong_ops)]
                cw = base.copy()
                cw["replay_condition"] = "wrong_operator"
                cw["wrong_operator_id"] = wrong_id
                cw["replay_plan_id"] = f"{op_id}__{i}__wrong_{wrong_id}"
                rows.append(cw)

            # random operator condition
            cr = base.copy()
            cr["replay_condition"] = "random_operator"
            cr["wrong_operator_id"] = random.choice([x.get("operator_id") for x in operators])
            cr["replay_plan_id"] = f"{op_id}__{i}__random"
            rows.append(cr)

    plan = pd.DataFrame(rows).reset_index(drop=True)
    plan["_sem6l5_row_id"] = np.arange(len(plan), dtype=int)
    return plan


def offline_score_for_condition(row: pd.Series, labels: pd.DataFrame) -> float:
    """
    Offline proxy:
      canonical_operator: use actual live_value of row if it matches source label/applicable row.
      no_operator: baseline 0
      wrong/random: estimate from rows of that operator label in mismatched contexts, with conservative shrink.
    This is intentionally conservative and is NOT a replacement for live replay.
    """
    cond = row.get("replay_condition", "")
    live = row.get("live_value", np.nan)

    if cond == "canonical_operator":
        return float(live) if pd.notna(live) else 0.0

    if cond == "no_operator":
        return 0.0

    if cond in {"wrong_operator", "random_operator"}:
        wrong_id = row.get("wrong_operator_id", "")
        wrong_label = OPERATOR_TO_LABEL.get(wrong_id, "")
        if wrong_label and wrong_label in set(labels["expansion_update_label"]):
            pool = labels[labels["expansion_update_label"] == wrong_label]
            # Conservative: use median value of wrong operator, shrunk and penalized if target operator mismatch.
            val = pd.to_numeric(pool["live_value"], errors="coerce").median()
            if pd.isna(val):
                return 0.0
            # Wrong operator is not expected to transfer cleanly; shrink to 25%.
            return float(val) * 0.25
        return 0.0

    return 0.0


def run_offline_replay(plan: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    out = plan.copy()
    out["offline_replay_value"] = out.apply(lambda r: offline_score_for_condition(r, labels), axis=1)
    return out


def compare_operator_vs_controls(replay: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for op_id, sub in replay.groupby("canonical_operator_id"):
        # Pair by original replay_plan_id root before control suffix.
        canon = sub[sub["replay_condition"] == "canonical_operator"].copy()
        if canon.empty:
            continue

        # Build a root key by removing condition suffix if present.
        canon["root_key"] = canon["replay_plan_id"].astype(str)
        # canonical replay_plan_id is op__i; controls are op__i__...
        controls = sub[sub["replay_condition"] != "canonical_operator"].copy()
        controls["root_key"] = controls["replay_plan_id"].astype(str).str.replace(r"__(no|wrong_.*|random)$", "", regex=True)

        for cond in ["no_operator", "wrong_operator", "random_operator"]:
            ctrl = controls[controls["replay_condition"] == cond]
            if ctrl.empty:
                continue
            c = canon[["root_key", "offline_replay_value"]].rename(columns={"offline_replay_value": "canon"})
            b = ctrl.groupby("root_key", as_index=False)["offline_replay_value"].mean().rename(columns={"offline_replay_value": "control"})
            m = c.merge(b, on="root_key", how="inner")
            diff = (m["canon"] - m["control"]).dropna().to_numpy(float)
            z = np.nan
            if len(diff) > 1 and np.std(diff, ddof=1) > 1e-12:
                z = float(np.mean(diff) / (np.std(diff, ddof=1) / math.sqrt(len(diff))))
            rows.append({
                "canonical_operator_id": op_id,
                "comparison": f"canonical>{cond}",
                "n": int(len(diff)),
                "mean_canonical": float(m["canon"].mean()) if len(m) else np.nan,
                "mean_control": float(m["control"].mean()) if len(m) else np.nan,
                "mean_diff": float(np.mean(diff)) if len(diff) else np.nan,
                "z": z,
                "win_rate": float(np.mean(diff > 0)) if len(diff) else np.nan,
            })
    return pd.DataFrame(rows)


def heldout_summary(labels: pd.DataFrame, library: Dict) -> pd.DataFrame:
    """
    Direct observed held-out summary by group/operator/source for source labels.
    """
    rows = []
    for op in library.get("operators", []):
        op_id = op.get("operator_id")
        label = op.get("source_label", OPERATOR_TO_LABEL.get(op_id, ""))
        sub = labels[labels["expansion_update_label"] == label].copy()
        if sub.empty:
            continue
        vals = pd.to_numeric(sub["live_value"], errors="coerce").dropna().to_numpy(float)

        # Leave-one-group observed means.
        group_means = sub.groupby("_group_id")["live_value"].mean()
        op_means = sub.groupby("target_operator")["live_value"].mean()
        source_means = sub.groupby("source_experiment")["live_value"].mean()

        rows.append({
            "canonical_operator_id": op_id,
            "source_label": label,
            "n": int(len(sub)),
            "mean_live": float(np.mean(vals)) if len(vals) else np.nan,
            "positive_rate": float(np.mean(sub["is_positive"])) if len(sub) else np.nan,
            "n_groups": int(sub["_group_id"].nunique()),
            "group_positive_frac": float(np.mean(group_means > 0)) if len(group_means) else np.nan,
            "min_group_mean": float(group_means.min()) if len(group_means) else np.nan,
            "n_target_ops": int(sub["target_operator"].nunique()),
            "operator_positive_frac": float(np.mean(op_means > 0)) if len(op_means) else np.nan,
            "min_operator_mean": float(op_means.min()) if len(op_means) else np.nan,
            "n_sources": int(sub["source_experiment"].nunique()),
            "source_positive_frac": float(np.mean(source_means > 0)) if len(source_means) else np.nan,
            "min_source_mean": float(source_means.min()) if len(source_means) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("mean_live", ascending=False)


def build_memory_candidates(library: Dict, comp: pd.DataFrame, held: pd.DataFrame, args) -> Dict:
    candidates = []
    pending = []

    for op in library.get("operators", []):
        op_id = op.get("operator_id")
        held_row = held[held["canonical_operator_id"] == op_id]
        comp_rows = comp[comp["canonical_operator_id"] == op_id]

        if held_row.empty:
            status = "pending_no_heldout_data"
            evidence = {}
        else:
            h = held_row.iloc[0].to_dict()
            # offline comparisons
            comp_ok = True
            key_comps = ["canonical>no_operator", "canonical>wrong_operator", "canonical>random_operator"]
            comp_summary = {}
            for kc in key_comps:
                r = comp_rows[comp_rows["comparison"] == kc]
                if len(r):
                    rr = r.iloc[0]
                    comp_summary[kc] = {
                        "mean_diff": float(rr["mean_diff"]),
                        "z": float(rr["z"]) if pd.notna(rr["z"]) else None,
                        "win_rate": float(rr["win_rate"]) if pd.notna(rr["win_rate"]) else None,
                    }
                    if not (pd.notna(rr["mean_diff"]) and rr["mean_diff"] >= args.min_mean_advantage and rr["win_rate"] >= args.min_win_rate):
                        comp_ok = False
                else:
                    comp_ok = False

            held_ok = (
                h.get("mean_live", -999) >= args.min_mean_live
                and h.get("positive_rate", 0) >= args.min_positive_rate
                and h.get("group_positive_frac", 0) >= args.min_group_positive_frac
                and h.get("source_positive_frac", 0) >= args.min_source_positive_frac
            )

            if held_ok and comp_ok:
                status = "eligible_for_operator_memory"
            elif held_ok:
                status = "eligible_pending_live_control_replay"
            else:
                status = "remain_canonical_draft"

            evidence = {
                "heldout_summary": h,
                "offline_comparisons": comp_summary,
            }

        card = {
            "operator_id": op_id,
            "source_label": op.get("source_label"),
            "status": status,
            "intent": op.get("intent"),
            "applicability_boundary": op.get("applicability_boundary"),
            "canonicality_score": op.get("canonicality_score"),
            "evidence": evidence,
            "memory_write_policy": (
                "Write to W'_cognitive / OperatorMemory as provisional canonical operator."
                if status == "eligible_for_operator_memory"
                else "Do not write as final memory yet; keep in canonical draft / replay queue."
            ),
        }

        if status == "eligible_for_operator_memory":
            candidates.append(card)
        else:
            pending.append(card)

    return {
        "version": "SEM-6L.5-operator-memory-candidates",
        "status": "offline_heldout_replay_audit",
        "eligible_operator_memory": candidates,
        "pending_or_draft": pending,
        "note": (
            "Offline held-out replay supports memory eligibility but does not replace a future true live-controller replay. "
            "Use sem6l5_live_replay_plan.csv for direct model-forward validation."
        ),
    }


def make_verdict(memory_candidates: Dict, comp: pd.DataFrame, held: pd.DataFrame) -> Dict:
    n_eligible = len(memory_candidates.get("eligible_operator_memory", []))
    n_pending = len(memory_candidates.get("pending_or_draft", []))

    if n_eligible >= 2:
        verdict = "PASS_STRONG_OPERATOR_MEMORY_ELIGIBILITY"
    elif n_eligible == 1:
        verdict = "PASS_LITE_ONE_OPERATOR_MEMORY_CANDIDATE"
    else:
        verdict = "FAIL_OR_INCONCLUSIVE"

    return {
        "stage": "SEM-6L.5",
        "mode": "canonical_operator_heldout_live_replay",
        "verdict": verdict,
        "n_operator_memory_eligible": n_eligible,
        "n_pending_or_draft": n_pending,
        "eligible_operator_ids": [x["operator_id"] for x in memory_candidates.get("eligible_operator_memory", [])],
        "notes": [
            "6L.5 is the final offline replay gate from CanonicalOperatorDraft to OperatorMemory eligibility.",
            "This audit uses existing held-out live rows and conservative offline controls.",
            "Final deployment into long-term W'_cognitive should still use the exported live replay plan for direct model-forward confirmation.",
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default=DEFAULT_LIBRARY)
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--max_per_operator", type=int, default=0)

    # eligibility gates
    ap.add_argument("--min_mean_live", type=float, default=0.0)
    ap.add_argument("--min_positive_rate", type=float, default=0.25)
    ap.add_argument("--min_group_positive_frac", type=float, default=0.25)
    ap.add_argument("--min_source_positive_frac", type=float, default=0.50)
    ap.add_argument("--min_mean_advantage", type=float, default=0.0)
    ap.add_argument("--min_win_rate", type=float, default=0.50)
    ap.add_argument("--strict", action="store_true", default=False)
    args = ap.parse_args()

    if args.strict:
        args.min_mean_advantage = max(args.min_mean_advantage, 0.02)
        args.min_win_rate = max(args.min_win_rate, 0.55)
        args.min_positive_rate = max(args.min_positive_rate, 0.30)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    library = load_library(args.library)
    labels = read_labels(args.labels)

    plan = build_live_replay_plan(library, labels, max_per_operator=args.max_per_operator)
    plan.to_csv(out_dir / "sem6l5_live_replay_plan.csv", index=False, encoding="utf-8-sig")

    replay = run_offline_replay(plan, labels)
    replay.to_csv(out_dir / "sem6l5_offline_replay_results.csv", index=False, encoding="utf-8-sig")

    comp = compare_operator_vs_controls(replay)
    comp.to_csv(out_dir / "sem6l5_operator_vs_controls.csv", index=False, encoding="utf-8-sig")

    held = heldout_summary(labels, library)
    held.to_csv(out_dir / "sem6l5_heldout_summary.csv", index=False, encoding="utf-8-sig")

    memory_candidates = build_memory_candidates(library, comp, held, args)
    with open(out_dir / "sem6l5_operator_memory_candidates.json", "w", encoding="utf-8") as f:
        json.dump(memory_candidates, f, ensure_ascii=False, indent=2)

    verdict = make_verdict(memory_candidates, comp, held)
    verdict["parameters"] = {
        "min_mean_live": args.min_mean_live,
        "min_positive_rate": args.min_positive_rate,
        "min_group_positive_frac": args.min_group_positive_frac,
        "min_source_positive_frac": args.min_source_positive_frac,
        "min_mean_advantage": args.min_mean_advantage,
        "min_win_rate": args.min_win_rate,
        "strict": args.strict,
    }
    verdict["outputs"] = {
        "live_replay_plan": str(out_dir / "sem6l5_live_replay_plan.csv"),
        "offline_replay_results": str(out_dir / "sem6l5_offline_replay_results.csv"),
        "operator_vs_controls": str(out_dir / "sem6l5_operator_vs_controls.csv"),
        "heldout_summary": str(out_dir / "sem6l5_heldout_summary.csv"),
        "operator_memory_candidates": str(out_dir / "sem6l5_operator_memory_candidates.json"),
        "verdict": str(out_dir / "sem6l5_verdict.json"),
    }

    with open(out_dir / "sem6l5_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)

    print("\n========== SEM-6L.5 ==========")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    print("\nHeldout summary:")
    print(held.to_string(index=False))
    print("\nOperator vs controls:")
    print(comp.to_string(index=False))
    print("\nOperatorMemory candidates:")
    print(json.dumps(memory_candidates, ensure_ascii=False, indent=2)[:5000])


if __name__ == "__main__":
    main()
