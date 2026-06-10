# -*- coding: utf-8 -*-
r"""
SEM-6J: Policy / Operator Write Audit

Purpose
-------
SEM-6H.1b showed:
    no-leak Q_theta selection = PASS

SEM-6H.2 / 6H.2b showed:
    direct latent / hidden interpolation write = FAIL/INCONCLUSIVE

Therefore SEM-6J changes the write target:

    not:    H <- (1-alpha)H + alpha*mu_F
    but:    Experience -> Policy_F / OperatorPrior_F update

This script tests whether the selected candidate from Q_theta can be converted into
a policy/operator write rule that improves trajectory-level live metrics.

Two modes
---------
A) plan mode:
    Build SEM-6J policy/operator write plan from:
        sem6h1b_scored_candidates.csv
        sem6h2_replay_plan.csv
    Output:
        sem6j_policy_operator_write_plan.csv

B) live-evaluation mode:
    If you provide live result CSV from a SEM-6J live controller script, it evaluates
    learned_noleak_Q policy/operator write vs commit_topk / same_family / random / shuffle.

This first script does NOT directly hook the model. It constructs the policy/operator
write plan and evaluation harness. The live controller can then implement ASA-style
operator/precursor interventions using this plan.

Why plan first?
---------------
6H.2b proved that write-operator form is the bottleneck. SEM-6J should not immediately
hardcode a new intervention. It first creates an explicit policy/operator routing table:

    candidate pool relation
    target operator
    candidate operator
    route:
        use_candidate_operator
        keep_target_operator
        correction_operator
        reject_as_artifact
        operator_transfer
    stage:
        shape_window / commit_window / policy_window
    write_strength:
        alpha_operator / beta_precursor / gate

Default paths
-------------
C:\Users\ZH\Desktop\AGI\python_script\sem6h1b_outputs\sem6h1b_scored_candidates.csv
C:\Users\ZH\Desktop\AGI\python_script\sem6h2_outputs\sem6h2_replay_plan.csv
C:\Users\ZH\Desktop\AGI\python_script\sem6j_outputs\sem6j_policy_operator_write_plan.csv

Run
---
python GPT_sem6j_policy_operator_write_audit.py

If live results exist:
python GPT_sem6j_policy_operator_write_audit.py ^
  --live_results sem6j_outputs\sem6j_live_results.csv
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_ROOT = r"C:\Users\ZH\Desktop\AGI\python_script"
DEFAULT_CANDIDATES = rf"{DEFAULT_ROOT}\sem6h1b_outputs\sem6h1b_scored_candidates.csv"
DEFAULT_REPLAY_PLAN = rf"{DEFAULT_ROOT}\sem6h2_outputs\sem6h2_replay_plan.csv"
DEFAULT_OUT_DIR = rf"{DEFAULT_ROOT}\sem6j_outputs"
DEFAULT_PLAN_OUT = rf"{DEFAULT_OUT_DIR}\sem6j_policy_operator_write_plan.csv"
DEFAULT_SUMMARY_OUT = rf"{DEFAULT_OUT_DIR}\sem6j_policy_operator_summary.csv"
DEFAULT_VERDICT_OUT = rf"{DEFAULT_OUT_DIR}\sem6j_verdict.json"


RANDOM_SEED = 20260606


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def detect_cols(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    nmap = {norm(c): c for c in df.columns}

    def pick(cands, required=True):
        for c in cands:
            if norm(c) in nmap:
                return nmap[norm(c)]
        for c in cands:
            nc = norm(c)
            for k, v in nmap.items():
                if nc in k or k in nc:
                    return v
        if required:
            raise ValueError(f"Missing column among {cands}. Columns={list(df.columns)}")
        return None

    return {
        "family": pick(["family", "_group_id", "target_family", "query_id"]),
        "pool": pick(["pool", "candidate_type", "candidate_source"], required=False),
        "target_concept": pick(["target_concept", "concept"], required=False),
        "target_operator": pick(["target_operator", "operator"], required=False),
        "candidate_family": pick(["candidate_family", "cand_family"], required=False),
        "candidate_concept": pick(["candidate_concept", "cand_concept"], required=False),
        "candidate_operator": pick(["candidate_operator", "cand_operator"], required=False),
        "candidate_id": pick(["candidate_i", "candidate_id", "cand_id"], required=False),
        "q_score": pick(["_learned_noleak_q", "learned_noleak_q", "q_score", "proxy_score"], required=False),
        "oracle_utility": pick(["oracle_utility", "utility", "true_utility"], required=False),
        "proxy_utility": pick(["proxy_utility", "_proxy_score"], required=False),
        "selection_family": pick(["_selection_family", "selection_family", "_selection"], required=False),
        "selection": pick(["_selection", "selection"], required=False),
    }


def add_flags(df: pd.DataFrame, pool_col: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    if pool_col and pool_col in out.columns:
        s = out[pool_col].astype(str).str.lower()
    else:
        s = pd.Series([""] * len(out), index=out.index)

    out["_is_same_family"] = (
        out["_is_same_family"].astype(bool)
        if "_is_same_family" in out.columns
        else s.str.contains("same_family|same-family|same family", regex=True)
    )
    out["_is_commit_topk"] = (
        out["_is_commit_topk"].astype(bool)
        if "_is_commit_topk" in out.columns
        else s.str.contains("commit|topk|top_k|neighbor", regex=True)
    )
    out["_is_random"] = (
        out["_is_random"].astype(bool)
        if "_is_random" in out.columns
        else s.str.contains("random|rand", regex=True)
    )
    return out


def infer_relation(row: pd.Series) -> str:
    pool = str(row.get("pool", "")).lower()
    if "same_concept_other_operator" in pool:
        return "same_concept_other_operator"
    if "same_operator_other_concept" in pool:
        return "same_operator_other_concept"
    if "commit" in pool or "topk" in pool or "neighbor" in pool:
        return "commit_topk_neighbor"
    if "same_family" in pool:
        return "same_family"
    if "random" in pool:
        return "random_other_family"

    tgt_c = str(row.get("target_concept", ""))
    cand_c = str(row.get("candidate_concept", ""))
    tgt_o = str(row.get("target_operator", ""))
    cand_o = str(row.get("candidate_operator", ""))
    if tgt_c == cand_c and tgt_o == cand_o:
        return "same_family"
    if tgt_c == cand_c and tgt_o != cand_o:
        return "same_concept_other_operator"
    if tgt_c != cand_c and tgt_o == cand_o:
        return "same_operator_other_concept"
    return "cross_concept_cross_operator"


def route_policy(row: pd.Series) -> Tuple[str, str, float, float, str]:
    """
    Returns:
        route, stage, alpha_operator, beta_precursor, rationale

    This is a first-principles SEM-6J routing rule:
      - same_operator_other_concept -> operator transfer prior
      - same_concept_other_operator -> correction / alternative operator prior
      - commit_topk -> identity/address prior, weak operator write
      - same_family -> residual / variant enrichment
      - random -> reject/null
    """
    relation = row["relation_type"]
    tgt_o = str(row.get("target_operator", ""))
    cand_o = str(row.get("candidate_operator", ""))
    oracle = row.get("oracle_utility", np.nan)
    q = row.get("_learned_noleak_q", row.get("q_score", np.nan))

    # Utility sign if available.
    util_pos = False
    try:
        util_pos = float(oracle) > 0
    except Exception:
        try:
            util_pos = float(q) > 0
        except Exception:
            util_pos = False

    if relation == "same_operator_other_concept":
        return (
            "write_operator_prior",
            "operator_window_L15_L19",
            0.20 if util_pos else 0.10,
            0.05,
            "same operator across concept suggests transferable OperatorPrior_F",
        )

    if relation == "same_concept_other_operator":
        return (
            "trigger_correction_operator",
            "policy_window_L15_L19_plus_commit_L23_L25",
            0.15 if util_pos else 0.05,
            0.10 if util_pos else 0.05,
            "same concept but different operator suggests correction/alternative policy routing",
        )

    if relation == "commit_topk_neighbor":
        return (
            "write_address_gate",
            "commit_window_L23_L25",
            0.00,
            0.05,
            "commit-neighbor candidate is useful as identity/address gate, not hidden vector write",
        )

    if relation == "same_family":
        return (
            "write_residual_mode",
            "shape_window_L7_L19",
            0.05 if util_pos else 0.02,
            0.00,
            "same family candidate enriches residual/variant coverage",
        )

    if relation == "random_other_family":
        return (
            "reject_as_null",
            "none",
            0.00,
            0.00,
            "random pool is null/control",
        )

    return (
        "route_by_q_gate",
        "policy_window_L15_L19",
        0.05 if util_pos else 0.00,
        0.05 if util_pos else 0.00,
        "fallback q-gated route",
    )


def build_plan(candidates_path: Path, replay_path: Path) -> pd.DataFrame:
    if replay_path.exists():
        df = pd.read_csv(replay_path)
    elif candidates_path.exists():
        # fallback: use scored candidates and select later
        df = pd.read_csv(candidates_path)
    else:
        raise FileNotFoundError("Neither replay plan nor candidates found.")

    cols = detect_cols(df)
    df = add_flags(df, cols.get("pool"))

    # Normalize column names needed downstream.
    rename = {}
    for key, col in cols.items():
        if col and col in df.columns:
            target_name = {
                "family": "family",
                "pool": "pool",
                "target_concept": "target_concept",
                "target_operator": "target_operator",
                "candidate_family": "candidate_family",
                "candidate_concept": "candidate_concept",
                "candidate_operator": "candidate_operator",
                "candidate_id": "candidate_i",
                "q_score": "_learned_noleak_q",
                "oracle_utility": "oracle_utility",
                "proxy_utility": "proxy_utility",
                "selection_family": "_selection_family",
                "selection": "_selection",
            }.get(key)
            if target_name and col != target_name and target_name not in df.columns:
                rename[col] = target_name
    df = df.rename(columns=rename)

    if "_selection_family" not in df.columns:
        df["_selection_family"] = "candidate_pool"
    if "_selection" not in df.columns:
        df["_selection"] = df["_selection_family"]

    # Ensure only selected candidate rows if replay plan includes them.
    # If raw candidates are passed, keep all but mark.
    needed = ["target_concept", "target_operator", "candidate_concept", "candidate_operator", "candidate_family"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot build SEM-6J plan, missing columns={missing}. Columns={list(df.columns)}")

    df["relation_type"] = df.apply(infer_relation, axis=1)
    routes = df.apply(route_policy, axis=1, result_type="expand")
    routes.columns = ["write_route", "write_stage", "alpha_operator", "beta_precursor", "route_rationale"]
    out = pd.concat([df, routes], axis=1)

    # Explicit operator write fields.
    out["operator_write_target"] = out.apply(
        lambda r: r["candidate_operator"]
        if r["write_route"] in {"write_operator_prior", "trigger_correction_operator"}
        else r["target_operator"],
        axis=1,
    )

    out["policy_action"] = out.apply(
        lambda r: f"a{r['alpha_operator']:.2f}_b{r['beta_precursor']:.2f}_{r['write_route']}",
        axis=1,
    )

    out["sem6j_hypothesis"] = (
        "Policy/Operator write should convert Q-selected experience into "
        "operator-prior / correction / address-gate updates rather than direct hidden interpolation."
    )

    return out


def summarize_plan(plan: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["_selection_family", "write_route", "relation_type"]
    for keys, sub in plan.groupby(group_cols, dropna=False):
        sel, route, rel = keys
        row = {
            "selection_family": sel,
            "write_route": route,
            "relation_type": rel,
            "n": len(sub),
            "mean_alpha_operator": float(pd.to_numeric(sub["alpha_operator"], errors="coerce").mean()),
            "mean_beta_precursor": float(pd.to_numeric(sub["beta_precursor"], errors="coerce").mean()),
        }
        if "oracle_utility" in sub.columns:
            row["mean_oracle_utility"] = float(pd.to_numeric(sub["oracle_utility"], errors="coerce").mean())
        if "_learned_noleak_q" in sub.columns:
            row["mean_q_score"] = float(pd.to_numeric(sub["_learned_noleak_q"], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["selection_family", "write_route", "relation_type"])


def find_live_metrics(df: pd.DataFrame) -> List[str]:
    metrics = []
    for c in df.columns:
        nc = norm(c)
        if any(x in nc for x in ["specificity", "live_gain", "trajectory_improvement", "live_margin_gain", "delta_u_gain", "rank_improvement"]):
            if pd.api.types.is_numeric_dtype(df[c]):
                metrics.append(c)
    return metrics


def evaluate_live(live_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    df = pd.read_csv(live_path)
    if "_selection_family" not in df.columns:
        raise ValueError("Live results must contain _selection_family.")
    if "write_route" not in df.columns:
        # It may be only keyed by replay plan rows. Still try selection-level summary.
        df["write_route"] = "unknown"

    metrics = find_live_metrics(df)
    if not metrics:
        raise ValueError("No live metric columns found in SEM-6J live results.")

    summary_rows = []
    for metric in metrics:
        for (sel, route), sub in df.groupby(["_selection_family", "write_route"], dropna=False):
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna().to_numpy()
            summary_rows.append({
                "metric": metric,
                "selection_family": sel,
                "write_route": route,
                "n": int(len(vals)),
                "mean": float(np.mean(vals)) if len(vals) else np.nan,
                "median": float(np.median(vals)) if len(vals) else np.nan,
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                "positive_rate": float(np.mean(vals > 0)) if len(vals) else np.nan,
            })
    summary = pd.DataFrame(summary_rows)

    pair_rows = []
    baselines = ["commit_topk_rule", "same_family", "random_pool", "shuffle_learned_Q"]
    for metric in metrics:
        if "learned_noleak_Q" not in set(df["_selection_family"]):
            continue
        A = df[df["_selection_family"] == "learned_noleak_Q"][["_group_id", metric]].copy()
        A = A.groupby("_group_id", as_index=False)[metric].mean().rename(columns={metric: "a"})
        for b in baselines:
            if b not in set(df["_selection_family"]):
                continue
            B = df[df["_selection_family"] == b][["_group_id", metric]].copy()
            B = B.groupby("_group_id", as_index=False)[metric].mean().rename(columns={metric: "b"})
            m = A.merge(B, on="_group_id")
            diff = (m["a"] - m["b"]).dropna().to_numpy()
            z = np.nan
            if len(diff) > 1 and np.std(diff, ddof=1) > 1e-12:
                z = float(np.mean(diff) / (np.std(diff, ddof=1) / math.sqrt(len(diff))))
            pair_rows.append({
                "metric": metric,
                "comparison": f"learned_noleak_Q>{b}",
                "n": int(len(diff)),
                "mean_diff": float(np.mean(diff)) if len(diff) else np.nan,
                "z": z,
                "win_rate": float(np.mean(diff > 0)) if len(diff) else np.nan,
            })
    pairwise = pd.DataFrame(pair_rows)

    verdict = {
        "stage": "SEM-6J",
        "mode": "live_result_evaluation",
        "metrics": metrics,
        "verdict": "FAIL_OR_INCONCLUSIVE",
        "notes": [
            "PASS requires learned_noleak_Q policy/operator write to beat commit_topk, same_family, random, and shuffle on live trajectory metrics.",
            "This evaluator is generic; final interpretation should inspect route-specific effects.",
        ],
    }

    # Conservative pass if any trajectory-like metric passes all required comparisons.
    for metric in metrics:
        sub = pairwise[pairwise["metric"] == metric]
        ok = {}
        for b in baselines:
            comp = f"learned_noleak_Q>{b}"
            row = sub[sub["comparison"] == comp]
            if len(row) == 0:
                ok[b] = False
            else:
                r = row.iloc[0]
                ok[b] = bool(pd.notna(r["mean_diff"]) and r["mean_diff"] > 0 and pd.notna(r["z"]) and r["z"] > 2.0)
        if all(ok.values()):
            verdict["verdict"] = "PASS_STRONG_POLICY_OPERATOR_WRITE"
            verdict["passing_metric"] = metric
            verdict["comparisons"] = ok
            break

    return summary, pairwise, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    ap.add_argument("--replay_plan", default=DEFAULT_REPLAY_PLAN)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--plan_out", default=DEFAULT_PLAN_OUT)
    ap.add_argument("--summary_out", default=DEFAULT_SUMMARY_OUT)
    ap.add_argument("--verdict_out", default=DEFAULT_VERDICT_OUT)
    ap.add_argument("--live_results", default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = build_plan(Path(args.candidates), Path(args.replay_plan))
    plan_path = Path(args.plan_out)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(plan_path, index=False, encoding="utf-8-sig")

    plan_summary = summarize_plan(plan)
    plan_summary.to_csv(args.summary_out, index=False, encoding="utf-8-sig")

    verdict = {
        "stage": "SEM-6J",
        "mode": "policy_operator_write_plan",
        "verdict": "PLAN_CREATED_NEEDS_SEM6J_LIVE_CONTROLLER",
        "plan_out": str(plan_path),
        "summary_out": str(args.summary_out),
        "n_rows": int(len(plan)),
        "selection_families": sorted(plan["_selection_family"].dropna().unique().tolist()),
        "write_routes": sorted(plan["write_route"].dropna().unique().tolist()),
        "notes": [
            "SEM-6J changes write target from hidden vector interpolation to Policy_F / OperatorPrior_F routing.",
            "Use sem6j_policy_operator_write_plan.csv as input to an ASA-style live controller.",
            "Direct hidden interpolation failed in SEM-6H.2/6H.2b; do not interpret this plan as another vector-write test.",
        ],
    }

    if args.live_results.strip():
        summary, pairwise, live_verdict = evaluate_live(Path(args.live_results))
        summary.to_csv(out_dir / "sem6j_live_summary.csv", index=False, encoding="utf-8-sig")
        pairwise.to_csv(out_dir / "sem6j_pairwise_live.csv", index=False, encoding="utf-8-sig")
        verdict = live_verdict

    with open(args.verdict_out, "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)

    print("\n========== SEM-6J ==========")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    print("\nPlan summary:")
    print(plan_summary.head(30).to_string(index=False))
    print(f"\nWrote plan: {plan_path}")


if __name__ == "__main__":
    main()
