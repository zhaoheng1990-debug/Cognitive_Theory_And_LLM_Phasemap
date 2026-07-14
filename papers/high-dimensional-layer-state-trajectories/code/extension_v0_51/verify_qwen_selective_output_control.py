#!/usr/bin/env python
"""Integrity checks for the frozen Qwen boundary-control result pack."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
EXECUTOR = ROOT / "local_transition_executor.py"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    split = pd.read_csv(OUT / "frozen_group_split.csv")
    train = split[split["split"] == "train"]
    test = split[split["split"] == "test"]
    train_ids, test_ids = set(train["prompt_id"].astype(str)), set(test["prompt_id"].astype(str))
    train_graphs, test_graphs = set(train["graph_id"]), set(test["graph_id"])
    grid = pd.read_csv(OUT / "train_action_grid.csv")
    labels = pd.read_csv(OUT / "train_boundary_policy_labels.csv")
    heldout = pd.read_csv(OUT / "heldout_control_rows.csv")
    top1 = pd.read_csv(OUT / "vocab_top1_rows.csv")
    actions = pd.read_csv(OUT / "actions_policy_boundary.csv")
    verdict = json.loads((OUT / "verdict.json").read_text(encoding="utf-8"))
    pca_audit = json.loads((OUT / "deltaU_trainfit_audit.json").read_text(encoding="utf-8"))

    checks = {
        "train_test_prompt_disjoint": train_ids.isdisjoint(test_ids),
        "train_test_graph_disjoint": train_graphs.isdisjoint(test_graphs),
        "train_prompt_count_670": len(train_ids) == 670,
        "test_prompt_count_290": len(test_ids) == 290,
        "test_graph_count_29": len(test_graphs) == 29,
        "grid_train_only": set(grid["prompt_id"].astype(str)) == train_ids,
        "policy_labels_train_only": set(labels["prompt_id"].astype(str)) == train_ids,
        "heldout_rows_test_only": set(heldout["prompt_id"].astype(str)) == test_ids,
        "top1_rows_test_only": set(top1["prompt_id"].astype(str)) == test_ids,
        "actions_test_only": set(actions["prompt_id"].astype(str)) == test_ids,
        "five_actions_per_test_prompt": actions.groupby("prompt_id").size().eq(5).all(),
        "alpha_bounded_at_one": float(actions["alpha"].max()) <= 1.0,
        "heldout_outputs_finite": heldout[["DeltaU", "final_margin", "margin_shift"]].notna().all().all(),
        "deltaU_pca_train_only": pca_audit["heldout_used_for_pca_fit"] is False,
        "deltaU_pca_fit_count_670": int(pca_audit["n_fit_prompts"]) == 670,
    }

    primary = top1[(top1["control"] == "policy_boundary") & (top1["mechanism"] == "closure")]
    strict = primary[primary["strict_conflict_to_clean"]]
    checks["strict_top1_flip_count_matches_verdict"] = len(strict) == int(verdict["primary_strict_conflict_to_clean_top1"])
    checks["strict_flips_are_vocab_rank_one"] = strict["clean_rank"].eq(1).all()
    checks["strict_flips_move_conflict_to_clean"] = (
        strict["baseline_top1_class"].eq("conflict").all() and strict["top1_class"].eq("clean").all()
    )
    guarded_other = top1[(top1["control"] == "policy_boundary_guarded") & (top1["mechanism"] != "closure")]
    checks["guarded_nonclosure_top1_unchanged"] = (~guarded_other["top1_changed"]).all()

    failed = [name for name, passed in checks.items() if not bool(passed)]
    report = {
        "verdict": "PASS" if not failed else "FAIL",
        "checks": {name: bool(value) for name, value in checks.items()},
        "failed": failed,
        "source_sha256": {
            "local_transition_executor": sha256(EXECUTOR),
            "runner": sha256(ROOT / "run_qwen_selective_output_control.py"),
            "protocol": sha256(ROOT / "README.md"),
        },
    }
    with open(OUT / "verification.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
