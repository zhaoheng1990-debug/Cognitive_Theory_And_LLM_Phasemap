# -*- coding: utf-8 -*-
"""
SEM-6H.2: No-Leak Live Write Replay Plan + Evaluation

Purpose
-------
Move from SEM-6H.1b statistical candidate selection to SEM-6H.2 live write replay.

This script has two modes:

Mode A: replay-plan mode
    Uses sem6h1b_scored_candidates.csv to select candidates per target family
    under learned_noleak_Q / proxy / commit-topk / same-family / random / null rules.
    It writes a replay plan CSV for the true model-forward live write script.

Mode B: live-result evaluation mode
    If live write result columns are already present in the candidate table, or if an
    external sem6g/sem6h2 live result CSV is provided, it evaluates:
        learned_noleak_Q_live_write
        proxy_only_live_write
        commit_topk_live_write
        same_family_live_write
        random_pool_live_write
        shuffle_Q_live_write
        permuted_utility_Q_live_write, if available

Why this split?
---------------
SEM-6H.1b proved clean no-leak Q selection. SEM-6H.2 must test whether selected
candidates produce actual trajectory improvement after MemoryUnit write. That requires
live model replay unless the live metrics have already been computed.

Inputs
------
Default:
    sem6h1b_scored_candidates.csv

Optional:
    --live_results sem6g_outputs/sem6g_live_write_results.csv
    or a CSV containing live metrics keyed by:
        family, candidate_family, pool, candidate_i

Expected key columns in candidate table:
    family
    target_concept
    target_operator
    pool
    candidate_i
    candidate_family
    candidate_concept
    candidate_operator
    oracle_utility
    proxy_utility
    _learned_noleak_q

Outputs
-------
sem6h2_outputs/
    sem6h2_replay_plan.csv
    sem6h2_selection_rows.csv
    sem6h2_live_summary.csv       if live metrics are available
    sem6h2_pairwise_live.csv       if live metrics are available
    sem6h2_verdict.json
    sem6h2_live_forward_todo.json  if live metrics are missing

Run
---
python GPT_sem6h2_noleak_live_write_replay.py

With explicit inputs:
python GPT_sem6h2_noleak_live_write_replay.py ^
  --candidates sem6h1b_scored_candidates.csv ^
  --out_dir sem6h2_outputs

If you have live replay results:
python GPT_sem6h2_noleak_live_write_replay.py ^
  --candidates sem6h1b_scored_candidates.csv ^
  --live_results sem6g_outputs/sem6g_live_write_results.csv ^
  --out_dir sem6h2_outputs
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


RANDOM_SEED = 20260606

DEFAULT_CANDIDATES = [
    "sem6h1b_scored_candidates.csv",
    r"sem6h1b_outputs\sem6h1b_scored_candidates.csv",
    r"sem6h_outputs\sem6h1_scored_candidates.csv",
    r"sem6e_outputs\sem6e_candidate_utilities.csv",
]
DEFAULT_OUT_DIR = "sem6h2_outputs"

# If live write metrics are in the candidate table or live_results table,
# we look for columns matching these patterns.
LIVE_METRIC_PATTERNS = [
    "live_gain",
    "live_utility",
    "live_rank_improvement",
    "rank_improvement_live",
    "write_gain",
    "write_utility",
    "delta_after_write",
    "trajectory_improvement",
]

ALPHA_PATTERNS = [
    r"alpha[_\-]?0?\.?1",
    r"a0?\.?1",
    r"alpha[_\-]?0?\.?2",
    r"a0?\.?2",
    r"alpha[_\-]?0?\.?3",
    r"a0?\.?3",
]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def find_existing(paths: List[str]) -> Optional[Path]:
    for p in paths:
        q = Path(p)
        if q.exists():
            return q
    return None


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
            raise ValueError(f"Missing required column among {cands}. Available={list(df.columns)}")
        return None

    return {
        "group": pick(["family", "query_id", "target_family", "row_id"]),
        "pool": pick(["pool", "candidate_type", "candidate_source"], required=False),
        "candidate_id": pick(["candidate_i", "candidate_id", "cand_id"], required=False),
        "candidate_family": pick(["candidate_family", "cand_family"], required=False),
        "candidate_concept": pick(["candidate_concept", "cand_concept"], required=False),
        "candidate_operator": pick(["candidate_operator", "cand_operator"], required=False),
        "target_concept": pick(["target_concept", "concept"], required=False),
        "target_operator": pick(["target_operator", "operator"], required=False),
        "oracle_utility": pick(["oracle_utility", "utility", "true_utility"], required=False),
        "proxy_score": pick(["proxy_utility", "_proxy_score", "proxy_score"], required=False),
        "learned_score": pick(["_learned_noleak_q", "learned_noleak_q", "q_noleak", "_real_q_score"], required=False),
    }


def ensure_flags(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> pd.DataFrame:
    out = df.copy()
    pool = cols.get("pool")
    if pool and pool in out.columns:
        s = out[pool].astype(str).str.lower()
    else:
        s = pd.Series([""] * len(out), index=out.index)

    if "_is_same_family" not in out.columns:
        out["_is_same_family"] = s.str.contains("same_family|same-family|same family", regex=True)
    else:
        out["_is_same_family"] = out["_is_same_family"].astype(bool)

    if "_is_commit_topk" not in out.columns:
        out["_is_commit_topk"] = s.str.contains("commit|topk|top_k|neighbor", regex=True)
    else:
        out["_is_commit_topk"] = out["_is_commit_topk"].astype(bool)

    if "_is_random" not in out.columns:
        out["_is_random"] = s.str.contains("random|rand", regex=True)
    else:
        out["_is_random"] = out["_is_random"].astype(bool)

    return out


def candidate_key_cols(cols: Dict[str, Optional[str]]) -> List[str]:
    keys = []
    for k in ["group", "pool", "candidate_id", "candidate_family", "candidate_concept", "candidate_operator"]:
        c = cols.get(k)
        if c:
            keys.append(c)
    # Dedup
    return list(dict.fromkeys(keys))


def pick_by_score(sub: pd.DataFrame, score_col: str) -> int:
    vals = pd.to_numeric(sub[score_col], errors="coerce").to_numpy(dtype=float)
    if np.all(~np.isfinite(vals)):
        return int(sub.index[0])
    vals = np.where(np.isfinite(vals), vals, -np.inf)
    return int(sub.index[int(np.argmax(vals))])


def pick_random(sub: pd.DataFrame, rng: np.random.Generator) -> int:
    return int(rng.choice(sub.index.to_numpy()))


def pick_mask_then_score(sub: pd.DataFrame, mask_col: str, score_col: str, rng: np.random.Generator) -> int:
    if mask_col in sub.columns:
        m = sub[sub[mask_col].fillna(False).astype(bool)]
        if len(m) > 0:
            return pick_by_score(m, score_col)
    return pick_random(sub, rng)


def make_selection_rows(
    df: pd.DataFrame,
    cols: Dict[str, Optional[str]],
    n_random_repl: int = 20,
    n_shuffle_repl: int = 20,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    group_col = cols["group"]

    learned_col = cols.get("learned_score")
    proxy_col = cols.get("proxy_score")
    oracle_col = cols.get("oracle_utility")

    if learned_col is None:
        raise ValueError("Need learned no-leak Q score column, e.g. _learned_noleak_q.")
    if proxy_col is None:
        # Fallback to learned for commit/same tie-breaking.
        proxy_col = learned_col

    selectors = []

    selectors.append(("learned_noleak_Q", lambda sub: pick_by_score(sub, learned_col)))
    selectors.append(("proxy_only", lambda sub: pick_by_score(sub, proxy_col)))
    selectors.append(("commit_topk_rule", lambda sub: pick_mask_then_score(sub, "_is_commit_topk", proxy_col, rng)))
    selectors.append(("same_family", lambda sub: pick_mask_then_score(sub, "_is_same_family", proxy_col, rng)))

    if oracle_col:
        selectors.append(("oracle_selected", lambda sub: pick_by_score(sub, oracle_col)))

    rows = []
    for sel_name, picker in selectors:
        for gid, sub in df.groupby(group_col, sort=False):
            idx = picker(sub)
            row = df.loc[idx].to_dict()
            row["_selection"] = sel_name
            row["_group_id"] = gid
            row["_selected_index"] = int(idx)
            rows.append(row)

    # Random reps.
    for r in range(n_random_repl):
        for gid, sub in df.groupby(group_col, sort=False):
            idx = pick_random(sub, rng)
            row = df.loc[idx].to_dict()
            row["_selection"] = f"random_pool_{r:03d}"
            row["_selection_family"] = "random_pool"
            row["_group_id"] = gid
            row["_selected_index"] = int(idx)
            rows.append(row)

    # Shuffle learned score reps.
    for r in range(n_shuffle_repl):
        tmp_score = df[learned_col].to_numpy(dtype=float).copy()
        rng.shuffle(tmp_score)
        tmp = df.copy()
        tmp["_shuffle_learned_score"] = tmp_score
        for gid, sub in tmp.groupby(group_col, sort=False):
            idx = pick_by_score(sub, "_shuffle_learned_score")
            row = df.loc[idx].to_dict()
            row["_selection"] = f"shuffle_learned_Q_{r:03d}"
            row["_selection_family"] = "shuffle_learned_Q"
            row["_group_id"] = gid
            row["_selected_index"] = int(idx)
            rows.append(row)

    out = pd.DataFrame(rows)
    if "_selection_family" not in out.columns:
        out["_selection_family"] = out["_selection"]
    out["_selection_family"] = out["_selection_family"].fillna(out["_selection"])
    return out


def find_live_metric_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        nc = norm(c)
        if any(p in nc for p in LIVE_METRIC_PATTERNS):
            cols.append(c)
            continue
        # alpha-specific live columns with gain/utility/improvement.
        if any(re.search(p, nc) for p in ALPHA_PATTERNS) and any(
            x in nc for x in ["gain", "utility", "improvement", "rank", "delta"]
        ):
            cols.append(c)
    return list(dict.fromkeys(cols))


def merge_live_results(sel: pd.DataFrame, live: pd.DataFrame, cols: Dict[str, Optional[str]]) -> pd.DataFrame:
    # Attempt a robust merge on available candidate keys.
    key_candidates = candidate_key_cols(cols)
    keys = [k for k in key_candidates if k in sel.columns and k in live.columns]
    if not keys:
        raise ValueError(
            "Could not find shared merge keys between selection rows and live results. "
            f"Candidate keys={key_candidates}. Live columns={list(live.columns)}"
        )

    # Deduplicate live rows to avoid cartesian explosion.
    live2 = live.drop_duplicates(subset=keys).copy()
    merged = sel.merge(live2, on=keys, how="left", suffixes=("", "_live"))
    return merged


def summarize_live(sel: pd.DataFrame, metric_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    pair_rows = []
    selections = sorted(sel["_selection_family"].dropna().unique())

    for metric in metric_cols:
        for s, sub in sel.groupby("_selection_family"):
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna().to_numpy(dtype=float)
            summary_rows.append({
                "metric": metric,
                "selection": s,
                "n": int(len(vals)),
                "mean": float(np.mean(vals)) if len(vals) else np.nan,
                "median": float(np.median(vals)) if len(vals) else np.nan,
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                "positive_rate": float(np.mean(vals > 0)) if len(vals) else np.nan,
            })

        # Pairwise vs key baselines by group.
        for baseline in ["commit_topk_rule", "same_family", "random_pool", "shuffle_learned_Q", "proxy_only", "oracle_selected"]:
            if "learned_noleak_Q" not in selections or baseline not in selections:
                continue

            A = sel[sel["_selection_family"] == "learned_noleak_Q"][["_group_id", metric]].rename(columns={metric: "a"})
            B = sel[sel["_selection_family"] == baseline][["_group_id", metric]].rename(columns={metric: "b"})

            # For repeated random/shuffle, average per group first.
            A = A.groupby("_group_id", as_index=False)["a"].mean()
            B = B.groupby("_group_id", as_index=False)["b"].mean()

            m = A.merge(B, on="_group_id", how="inner")
            diff = (pd.to_numeric(m["a"], errors="coerce") - pd.to_numeric(m["b"], errors="coerce")).dropna().to_numpy(dtype=float)
            z = np.nan
            if len(diff) > 1 and np.std(diff, ddof=1) > 1e-12:
                z = float(np.mean(diff) / (np.std(diff, ddof=1) / math.sqrt(len(diff))))
            pair_rows.append({
                "metric": metric,
                "comparison": f"learned_noleak_Q>{baseline}",
                "n": int(len(diff)),
                "mean_diff": float(np.mean(diff)) if len(diff) else np.nan,
                "z": z,
                "win_rate": float(np.mean(diff > 0)) if len(diff) else np.nan,
            })

    return pd.DataFrame(summary_rows), pd.DataFrame(pair_rows)


def make_verdict(
    selection_rows: pd.DataFrame,
    live_metric_cols: List[str],
    live_summary: Optional[pd.DataFrame],
    pairwise: Optional[pd.DataFrame],
) -> Dict:
    out = {
        "stage": "SEM-6H.2",
        "mode": "replay_plan_only" if not live_metric_cols else "live_result_evaluation",
        "live_metric_columns": live_metric_cols,
        "verdict": "REPLAY_PLAN_CREATED_NEEDS_MODEL_FORWARD",
        "notes": [],
    }

    if not live_metric_cols:
        out["notes"].append(
            "No live write metric columns were found. Use sem6h2_replay_plan.csv as input to the local model-forward live write script."
        )
        out["notes"].append(
            "SEM-6H.2 PASS requires actual post-write trajectory metrics, not oracle utility from SEM-6E."
        )
        return out

    verdict_by_metric = {}
    for metric in live_metric_cols:
        sub = pairwise[pairwise["metric"] == metric] if pairwise is not None else pd.DataFrame()
        tests = {row["comparison"]: row for _, row in sub.iterrows()}

        def ok(comp, z_min=2.0, diff_positive=True):
            if comp not in tests:
                return False
            row = tests[comp]
            return (
                pd.notna(row["mean_diff"])
                and (row["mean_diff"] > 0 if diff_positive else True)
                and pd.notna(row["z"])
                and row["z"] > z_min
            )

        pass_lite = (
            ok("learned_noleak_Q>same_family", z_min=2.0)
            and ok("learned_noleak_Q>random_pool", z_min=2.0)
        )
        pass_strong = (
            pass_lite
            and ok("learned_noleak_Q>shuffle_learned_Q", z_min=2.0)
        )
        pass_milestone = (
            pass_strong
            and ok("learned_noleak_Q>commit_topk_rule", z_min=2.0)
        )

        if pass_milestone:
            v = "PASS_MILESTONE_LIVE_WRITE_GT_COMMIT_AND_MATCHED_NULL"
        elif pass_strong:
            v = "PASS_STRONG_LIVE_WRITE_MATCHED_NULL"
        elif pass_lite:
            v = "PASS_LITE_LIVE_WRITE_BASIC_BASELINES"
        else:
            v = "FAIL_OR_INCONCLUSIVE"

        verdict_by_metric[metric] = {
            "verdict": v,
            "pass_lite": pass_lite,
            "pass_strong": pass_strong,
            "pass_milestone": pass_milestone,
        }

    # Global verdict = best metric verdict, but conservative note.
    priorities = [
        "PASS_MILESTONE_LIVE_WRITE_GT_COMMIT_AND_MATCHED_NULL",
        "PASS_STRONG_LIVE_WRITE_MATCHED_NULL",
        "PASS_LITE_LIVE_WRITE_BASIC_BASELINES",
    ]
    best = "FAIL_OR_INCONCLUSIVE"
    for p in priorities:
        if any(v["verdict"] == p for v in verdict_by_metric.values()):
            best = p
            break

    out["verdict"] = best
    out["verdict_by_metric"] = verdict_by_metric
    out["notes"].append(
        "A robust SEM-6H.2 PASS should be based on trajectory metrics such as rank/gain/margin after live MemoryUnit write."
    )
    out["notes"].append(
        "If only oracle_utility is available, that is SEM-6H.1b selection evidence, not SEM-6H.2 live write evidence."
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--live_results", default=None)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--n_random_repl", type=int, default=50)
    ap.add_argument("--n_shuffle_repl", type=int, default=50)
    args = ap.parse_args()

    cand_path = Path(args.candidates) if args.candidates else find_existing(DEFAULT_CANDIDATES)
    if cand_path is None or not cand_path.exists():
        raise FileNotFoundError("Candidate table not found. Provide --candidates.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cand = pd.read_csv(cand_path)
    cols = detect_cols(cand)
    cand = ensure_flags(cand, cols)

    selection_rows = make_selection_rows(
        cand, cols,
        n_random_repl=args.n_random_repl,
        n_shuffle_repl=args.n_shuffle_repl,
    )

    # Replay plan = selected candidates to run in true live write.
    plan_cols = [
        "_selection", "_selection_family", "_group_id", "_selected_index",
        cols.get("group"), cols.get("target_concept"), cols.get("target_operator"),
        cols.get("pool"), cols.get("candidate_id"), cols.get("candidate_family"),
        cols.get("candidate_concept"), cols.get("candidate_operator"),
        cols.get("learned_score"), cols.get("proxy_score"),
        cols.get("oracle_utility"),
        "_is_same_family", "_is_commit_topk", "_is_random",
    ]
    plan_cols = [c for c in plan_cols if c and c in selection_rows.columns]
    replay_plan = selection_rows[plan_cols].copy()
    replay_plan.to_csv(out_dir / "sem6h2_replay_plan.csv", index=False, encoding="utf-8-sig")
    selection_rows.to_csv(out_dir / "sem6h2_selection_rows.csv", index=False, encoding="utf-8-sig")

    live_metric_cols = find_live_metric_cols(selection_rows)
    live_merged = selection_rows.copy()

    if args.live_results:
        live_path = Path(args.live_results)
        if not live_path.exists():
            raise FileNotFoundError(f"Live results not found: {live_path}")
        live = pd.read_csv(live_path)
        live_merged = merge_live_results(selection_rows, live, cols)
        live_metric_cols = find_live_metric_cols(live_merged)

    if live_metric_cols:
        live_merged.to_csv(out_dir / "sem6h2_merged_live_rows.csv", index=False, encoding="utf-8-sig")
        summary, pairwise = summarize_live(live_merged, live_metric_cols)
        summary.to_csv(out_dir / "sem6h2_live_summary.csv", index=False, encoding="utf-8-sig")
        pairwise.to_csv(out_dir / "sem6h2_pairwise_live.csv", index=False, encoding="utf-8-sig")
        verdict = make_verdict(live_merged, live_metric_cols, summary, pairwise)
    else:
        todo = {
            "why": "No live post-write trajectory metric columns were found.",
            "next_action": "Use sem6h2_replay_plan.csv to run local model-forward MemoryUnit write replay.",
            "required_live_metrics_examples": [
                "live_gain_alpha_0.1",
                "live_gain_alpha_0.2",
                "live_gain_alpha_0.3",
                "live_rank_improvement",
                "live_margin_gain",
                "live_deltaU_gain",
                "trajectory_improvement",
            ],
            "required_comparisons": [
                "learned_noleak_Q > commit_topk_rule",
                "learned_noleak_Q > same_family",
                "learned_noleak_Q > random_pool",
                "learned_noleak_Q > shuffle_learned_Q",
                "oracle_selected >= learned_noleak_Q",
            ],
        }
        with open(out_dir / "sem6h2_live_forward_todo.json", "w", encoding="utf-8") as f:
            json.dump(todo, f, ensure_ascii=False, indent=2)
        verdict = make_verdict(selection_rows, [], None, None)

    verdict["candidate_input"] = str(cand_path)
    verdict["n_candidate_rows"] = int(len(cand))
    verdict["n_groups"] = int(cand[cols["group"]].nunique())
    verdict["n_replay_rows"] = int(len(replay_plan))
    verdict["selection_families"] = sorted(selection_rows["_selection_family"].dropna().unique().tolist())
    verdict["detected_columns"] = cols

    with open(out_dir / "sem6h2_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)

    print("\n========== SEM-6H.2 ==========")
    print(f"Candidate input: {cand_path}")
    print(f"Replay plan: {out_dir / 'sem6h2_replay_plan.csv'}")
    print(f"Mode: {verdict['mode']}")
    print(f"Verdict: {verdict['verdict']}")
    if live_metric_cols:
        print("\nLive metric columns:")
        for c in live_metric_cols:
            print("  ", c)
        print("\nLive summary:")
        print(summary.to_string(index=False))
        print("\nPairwise:")
        print(pairwise.to_string(index=False))
    else:
        print("No live metric columns found. Generated replay plan for local forward.")
        print(f"Todo: {out_dir / 'sem6h2_live_forward_todo.json'}")


if __name__ == "__main__":
    main()
