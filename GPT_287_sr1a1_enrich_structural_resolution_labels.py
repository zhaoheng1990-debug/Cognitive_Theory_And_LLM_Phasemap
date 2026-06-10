# -*- coding: utf-8 -*-
"""
SR-1A.1: Enrich StructuralResolution Labels with 6L.8 Gain / Policy Evidence

Purpose
-------
SR-1A produced clean reuse labels:
    direct_reuse / observe / reject

But if sem6l7_pivot_results.csv does not contain gain columns, the primary
label file has no true gain fields. This script repairs that by reading:

    sr1a_outputs/sr1a_structural_resolution_labels.csv
    sr1a_outputs/sr1a_aux_sem6l8_policy_results_normalized.csv

and merging task-level gain evidence from SEM-6L.8 back into a 204-row
enriched label file.

Outputs
-------
    sr1a_outputs/sr1a1_enriched_structural_resolution_labels.csv
    sr1a_outputs/sr1a1_policy_summary_by_policy.csv
    sr1a_outputs/sr1a1_group_summary.csv
    sr1a_outputs/sr1a1_summary.json

Theory
------
SR-1A defines the structural-resolution policy target.
SR-1A.1 attaches outcome evidence so later SR-1C can learn:

    observed state -> predicted SR -> reuse action -> expected utility
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(r"C:\Users\ZH\Desktop\AGI\python_script")
OUT_DIR = BASE_DIR / "sr1a_outputs"

IN_LABELS = OUT_DIR / "sr1a_structural_resolution_labels.csv"
IN_AUX = OUT_DIR / "sr1a_aux_sem6l8_policy_results_normalized.csv"

OUT_ENRICHED = OUT_DIR / "sr1a1_enriched_structural_resolution_labels.csv"
OUT_POLICY_SUMMARY = OUT_DIR / "sr1a1_policy_summary_by_policy.csv"
OUT_GROUP_SUMMARY = OUT_DIR / "sr1a1_group_summary.csv"
OUT_SUMMARY = OUT_DIR / "sr1a1_summary.json"


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    df = pd.read_csv(path)
    print(f"[LOAD] {path}")
    print(f"       shape={df.shape}")
    return df


def num(s):
    return pd.to_numeric(s, errors="coerce")


def choose_action_from_gain(correct_vs_no, correct_vs_wrong, correct_vs_random):
    """
    Gain-derived oracle action.

    direct_reuse:
        correct memory improves over no-memory and is not worse than wrong/random.
    observe:
        correct memory is not clearly useful but not catastrophic.
    reject:
        correct memory is harmful or dominated by wrong/random controls.
    """
    vals = [correct_vs_no, correct_vs_wrong, correct_vs_random]
    if any(pd.isna(v) for v in vals):
        return "unknown"

    if correct_vs_no > 0 and correct_vs_wrong > 0 and correct_vs_random > 0:
        return "direct_reuse"

    if correct_vs_no >= -0.25 and correct_vs_wrong >= -0.25 and correct_vs_random >= -0.25:
        return "observe"

    return "reject"


def main():
    labels = read_required(IN_LABELS)
    aux = read_required(IN_AUX)

    required_aux_cols = [
        "task_id",
        "reuse_group",
        "boundary_strength",
        "task_family",
        "target_operator",
        "correct_memory",
        "no_memory",
        "random_memory",
        "wrong_memory",
        "correct_vs_no",
        "correct_vs_wrong",
        "correct_vs_random",
        "policy_id",
        "reuse_action",
        "policy_gain",
        "wrong_control_gain",
        "random_control_gain",
        "no_control_gain",
        "policy_vs_no",
        "policy_vs_wrong",
        "policy_vs_random",
        "called_direct",
        "observed_only",
        "rejected",
    ]
    missing = [c for c in required_aux_cols if c not in aux.columns]
    if missing:
        raise ValueError(f"Missing required columns in aux file: {missing}")

    # SEM-6L.8 contains one row per task per policy. The memory-control
    # columns are repeated across policies, so take first after sorting.
    task_gain_cols = [
        "task_id",
        "correct_memory",
        "no_memory",
        "random_memory",
        "wrong_memory",
        "correct_vs_no",
        "correct_vs_wrong",
        "correct_vs_random",
    ]

    task_gain = (
        aux.sort_values(["task_id", "policy_id"])
        .groupby("task_id", as_index=False)[task_gain_cols[1:]]
        .first()
    )

    for c in task_gain_cols[1:]:
        task_gain[c] = num(task_gain[c])

    task_gain["gain_oracle_action"] = task_gain.apply(
        lambda r: choose_action_from_gain(
            r["correct_vs_no"],
            r["correct_vs_wrong"],
            r["correct_vs_random"],
        ),
        axis=1,
    )

    task_gain["is_gain_direct_region"] = (task_gain["gain_oracle_action"] == "direct_reuse").astype(int)
    task_gain["is_gain_observe_region"] = (task_gain["gain_oracle_action"] == "observe").astype(int)
    task_gain["is_gain_reject_region"] = (task_gain["gain_oracle_action"] == "reject").astype(int)

    # Per-task best and worst policies.
    aux2 = aux.copy()
    for c in [
        "policy_gain",
        "policy_vs_no",
        "policy_vs_wrong",
        "policy_vs_random",
        "called_direct",
        "observed_only",
        "rejected",
    ]:
        aux2[c] = num(aux2[c])

    aux2["safety_adjusted_policy_score"] = (
        aux2["policy_vs_no"]
        + 0.25 * aux2["policy_vs_wrong"]
        + 0.25 * aux2["policy_vs_random"]
    )

    best_idx = aux2.groupby("task_id")["safety_adjusted_policy_score"].idxmax()
    best_policy = aux2.loc[best_idx, [
        "task_id",
        "policy_id",
        "reuse_action",
        "policy_gain",
        "policy_vs_no",
        "policy_vs_wrong",
        "policy_vs_random",
        "safety_adjusted_policy_score",
        "called_direct",
        "observed_only",
        "rejected",
    ]].copy()

    best_policy = best_policy.rename(columns={
        "policy_id": "best_policy_id",
        "reuse_action": "best_policy_action",
        "policy_gain": "best_policy_gain",
        "policy_vs_no": "best_policy_vs_no",
        "policy_vs_wrong": "best_policy_vs_wrong",
        "policy_vs_random": "best_policy_vs_random",
        "safety_adjusted_policy_score": "best_safety_adjusted_policy_score",
        "called_direct": "best_called_direct",
        "observed_only": "best_observed_only",
        "rejected": "best_rejected",
    })

    # Extract canonical policy rows.
    keep_policies = [
        "sr_policy_v1",
        "sr_policy_v2",
        "strict_only",
        "always_reuse",
        "surface_reuse",
        "dynamic_context_reuse",
        "random_policy",
        "no_reuse",
    ]

    wide_parts = []
    for pid in keep_policies:
        sub = aux2[aux2["policy_id"] == pid].copy()
        if sub.empty:
            continue
        cols = [
            "task_id",
            "reuse_action",
            "policy_gain",
            "policy_vs_no",
            "policy_vs_wrong",
            "policy_vs_random",
            "called_direct",
            "observed_only",
            "rejected",
        ]
        sub = sub[cols].copy()
        rename = {c: f"{pid}__{c}" for c in cols if c != "task_id"}
        sub = sub.rename(columns=rename)
        wide_parts.append(sub)

    policy_wide = None
    for part in wide_parts:
        if policy_wide is None:
            policy_wide = part
        else:
            policy_wide = policy_wide.merge(part, on="task_id", how="outer")

    if policy_wide is None:
        policy_wide = pd.DataFrame({"task_id": labels["task_id"].astype(str)})

    enriched = labels.copy()
    enriched["task_id"] = enriched["task_id"].astype(str)

    enriched = enriched.merge(task_gain, on="task_id", how="left", suffixes=("", "_from6l8"))
    enriched = enriched.merge(best_policy, on="task_id", how="left")
    enriched = enriched.merge(policy_wide, on="task_id", how="left")

    # Compare label policy vs gain-derived oracle.
    enriched["sr_policy_matches_gain_oracle"] = (
        enriched["sr_policy_v1_action"].astype(str) == enriched["gain_oracle_action"].astype(str)
    ).astype(int)

    enriched["sr_policy_matches_best_policy_action"] = (
        enriched["sr_policy_v1_action"].astype(str) == enriched["best_policy_action"].astype(str)
    ).astype(int)

    # Group summaries.
    group_summary = (
        enriched.groupby(["reuse_group", "boundary_strength"], dropna=False)
        .agg(
            n=("task_id", "count"),
            mean_correct_vs_no=("correct_vs_no", "mean"),
            mean_correct_vs_wrong=("correct_vs_wrong", "mean"),
            mean_correct_vs_random=("correct_vs_random", "mean"),
            sr_match_gain_oracle=("sr_policy_matches_gain_oracle", "mean"),
            sr_match_best_policy_action=("sr_policy_matches_best_policy_action", "mean"),
            best_policy_score=("best_safety_adjusted_policy_score", "mean"),
        )
        .reset_index()
        .sort_values(["boundary_strength", "reuse_group"], ascending=[False, True])
    )

    policy_summary = (
        aux2.groupby("policy_id", dropna=False)
        .agg(
            n=("task_id", "count"),
            mean_policy_gain=("policy_gain", "mean"),
            mean_policy_vs_no=("policy_vs_no", "mean"),
            mean_policy_vs_wrong=("policy_vs_wrong", "mean"),
            mean_policy_vs_random=("policy_vs_random", "mean"),
            mean_safety_adjusted_score=("safety_adjusted_policy_score", "mean"),
            direct_rate=("called_direct", "mean"),
            observe_rate=("observed_only", "mean"),
            reject_rate=("rejected", "mean"),
        )
        .reset_index()
        .sort_values("mean_safety_adjusted_score", ascending=False)
    )

    summary = {
        "input_files": {
            "labels": str(IN_LABELS),
            "aux": str(IN_AUX),
        },
        "n_labels": int(len(labels)),
        "n_aux_rows": int(len(aux)),
        "n_enriched_rows": int(len(enriched)),
        "unique_tasks_in_aux": int(aux["task_id"].nunique()),
        "task_rows_per_policy": aux.groupby("task_id").size().value_counts().to_dict(),
        "gain_oracle_action_counts": enriched["gain_oracle_action"].value_counts(dropna=False).to_dict(),
        "sr_policy_v1_action_counts": enriched["sr_policy_v1_action"].value_counts(dropna=False).to_dict(),
        "best_policy_id_counts": enriched["best_policy_id"].value_counts(dropna=False).to_dict(),
        "best_policy_action_counts": enriched["best_policy_action"].value_counts(dropna=False).to_dict(),
        "sr_policy_matches_gain_oracle_rate": float(enriched["sr_policy_matches_gain_oracle"].mean()),
        "sr_policy_matches_best_policy_action_rate": float(enriched["sr_policy_matches_best_policy_action"].mean()),
        "policy_summary_top": policy_summary.head(8).to_dict(orient="records"),
    }

    has_all_rows = len(enriched) == len(labels) == aux["task_id"].nunique()
    has_gain = enriched["correct_vs_no"].notna().all()
    has_policy = enriched["best_policy_id"].notna().all()

    if has_all_rows and has_gain and has_policy:
        summary["verdict"] = "PASS_SR1A1_GAIN_ENRICHED_LABELS_READY"
    else:
        summary["verdict"] = "NEEDS_AUDIT_SR1A1_INCOMPLETE_MERGE"

    enriched.to_csv(OUT_ENRICHED, index=False, encoding="utf-8-sig")
    policy_summary.to_csv(OUT_POLICY_SUMMARY, index=False, encoding="utf-8-sig")
    group_summary.to_csv(OUT_GROUP_SUMMARY, index=False, encoding="utf-8-sig")

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[SAVED]")
    print(f"  enriched labels : {OUT_ENRICHED}")
    print(f"  policy summary  : {OUT_POLICY_SUMMARY}")
    print(f"  group summary   : {OUT_GROUP_SUMMARY}")
    print(f"  summary         : {OUT_SUMMARY}")

    print("\n[SUMMARY]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
