#!/usr/bin/env python
"""Run condition-stratified, safety-matched action-assignment nulls."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FORMAL = ROOT / "output"
OUT = ROOT / "safety_matched_null_v0_1" / "output"
RUNNER_PATH = ROOT / "run_qwen_selective_output_control.py"
N_NULLS = 50
SEED_BASE = 1042


def load_runner():
    spec = importlib.util.spec_from_file_location("qwen_output_control_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prompt_action_blocks(actions):
    blocks = {}
    for prompt_id, group in actions.groupby("prompt_id", sort=False):
        blocks[str(prompt_id)] = group.sort_values("layer").reset_index(drop=True)
    return blocks


def permute_blocks(real_actions, closure_meta, seed):
    rng = np.random.default_rng(seed)
    blocks = prompt_action_blocks(real_actions)
    out_blocks, mapping = [], []

    for condition, meta in closure_meta.groupby("condition", sort=True):
        open_ids = meta.loc[meta["gate_open"], "prompt_id"].astype(str).tolist()
        closed_ids = meta.loc[~meta["gate_open"], "prompt_id"].astype(str).tolist()
        donor_ids = open_ids.copy()
        while True:
            rng.shuffle(donor_ids)
            if len(open_ids) <= 1 or all(a != b for a, b in zip(open_ids, donor_ids)):
                break

        for recipient, donor in zip(open_ids, donor_ids):
            block = blocks[donor].copy()
            block["prompt_id"] = recipient
            block["null_seed"] = seed
            block["donor_prompt_id"] = donor
            block["recipient_prompt_id"] = recipient
            block["condition"] = condition
            out_blocks.append(block)
            mapping.append(
                {
                    "null_seed": seed,
                    "condition": condition,
                    "recipient_prompt_id": recipient,
                    "donor_prompt_id": donor,
                    "identity": recipient == donor,
                }
            )

        for prompt_id in closed_ids:
            block = blocks[prompt_id].copy()
            block["null_seed"] = seed
            block["donor_prompt_id"] = prompt_id
            block["recipient_prompt_id"] = prompt_id
            block["condition"] = condition
            out_blocks.append(block)
            mapping.append(
                {
                    "null_seed": seed,
                    "condition": condition,
                    "recipient_prompt_id": prompt_id,
                    "donor_prompt_id": prompt_id,
                    "identity": True,
                }
            )

    actions = pd.concat(out_blocks, ignore_index=True)
    return actions, pd.DataFrame(mapping)


def assert_matched(real_actions, null_actions, closure_meta):
    real = real_actions.merge(closure_meta[["prompt_id", "condition"]], on="prompt_id", how="inner")
    null = null_actions.copy()
    columns = ["condition", "layer", "alpha", "beta"]
    real_counts = real.groupby(columns).size().sort_index()
    null_counts = null.groupby(columns).size().sort_index()
    if not real_counts.equals(null_counts):
        raise AssertionError("Null action multiset differs from the real action multiset")
    if set(null["prompt_id"].astype(str)) != set(closure_meta["prompt_id"].astype(str)):
        raise AssertionError("Null prompt set changed")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runner = load_runner()
    mod = runner.load_executor(runner.EXECUTOR_SOURCE)
    cfg = mod.Config()
    cfg.save_dir = str(OUT / "runtime")
    cfg.n_graphs = 96
    cfg.batch_size = 4
    cfg.n_shuffles = 0
    cfg.include_answer_control = False
    mod.CFG = cfg
    mod.make_combo_hook_factory = runner.norm_tracing_hook_factory(mod)
    mod.set_seed(cfg.seed)

    print("[LOAD] reconstruct frozen executor", flush=True)
    model, tokenizer = mod.load_model_and_tokenizer(cfg)
    runtime = OUT / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    df = mod.build_dataset(tokenizer, cfg, runtime)
    base_df, hidden, _ = mod.extract_baseline(model, tokenizer, df, cfg, runtime)
    train_idx, test_idx, split = mod.make_split(base_df, cfg, runtime)
    base_df, _ = runner.refit_delta_u_on_train(mod, base_df, train_idx, cfg, OUT)
    stable_ops = mod.fit_stable_operators(base_df, hidden, train_idx, cfg, runtime)
    precursor_dirs, _, _ = mod.fit_precursor_and_probes(base_df, hidden, train_idx, cfg, runtime)

    test_df = base_df.iloc[test_idx].reset_index(drop=True)
    closure_df = test_df[test_df["mechanism"] == "closure"].reset_index(drop=True)
    real_actions = pd.read_csv(FORMAL / "actions_policy_boundary_guarded.csv")
    closure_actions = real_actions[real_actions["prompt_id"].astype(str).isin(closure_df["prompt_id"].astype(str))].copy()
    prompt_gate = (
        closure_actions.groupby("prompt_id", as_index=False)
        .agg(gate_open=("gate_open", "first"))
    )
    prompt_gate["gate_open"] = prompt_gate["gate_open"].astype(str).str.lower().eq("true")
    closure_meta = closure_df[["prompt_id", "condition", "graph_id"]].merge(prompt_gate, on="prompt_id", how="left")

    real_top1 = pd.read_csv(FORMAL / "vocab_top1_policy_boundary_guarded.csv")
    real_top1 = real_top1[real_top1["mechanism"] == "closure"].copy()
    real_count = int((real_top1["top1_class"] == "clean").sum())
    if real_count != 17:
        raise AssertionError(f"Expected frozen guarded count 17, found {real_count}")

    summaries, all_rows, all_maps = [], [], []
    for null_id in range(N_NULLS):
        seed = SEED_BASE + null_id
        control = f"shuffle_matched_{null_id:03d}"
        print(f"[NULL {null_id + 1}/{N_NULLS}] seed={seed}", flush=True)
        null_actions, mapping = permute_blocks(closure_actions, closure_meta, seed)
        assert_matched(closure_actions, null_actions, closure_meta)
        mapping["null_id"] = null_id
        all_maps.append(mapping)

        result = runner.evaluate_vocab_top1(
            mod,
            model,
            tokenizer,
            closure_df,
            cfg,
            stable_ops,
            precursor_dirs,
            control,
            null_actions,
            OUT,
        )
        result["null_id"] = null_id
        result["null_seed"] = seed
        result["strict_conflict_to_clean"] = result["top1_class"] == "clean"
        all_rows.append(result)
        by_condition = result.groupby("condition")["strict_conflict_to_clean"].sum().to_dict()
        summaries.append(
            {
                "null_id": null_id,
                "seed": seed,
                "crossings": int(result["strict_conflict_to_clean"].sum()),
                "mean_margin": float(result["clean_minus_conflict_margin"].mean()),
                "closure_override_crossings": int(by_condition.get("closure_override", 0)),
                "closure_update_crossings": int(by_condition.get("closure_update", 0)),
                "exception_override_crossings": int(by_condition.get("exception_override", 0)),
            }
        )

    summary = pd.DataFrame(summaries)
    rows = pd.concat(all_rows, ignore_index=True)
    maps = pd.concat(all_maps, ignore_index=True)
    summary.to_csv(OUT / "matched_null_summary.csv", index=False)
    rows.to_csv(OUT / "matched_null_rows.csv", index=False)
    maps.to_csv(OUT / "matched_null_permutation_map.csv", index=False)

    exceed = int((summary["crossings"] >= real_count).sum())
    empirical_p = (1 + exceed) / (N_NULLS + 1)
    q95 = float(summary["crossings"].quantile(0.95, interpolation="higher"))
    verdict = {
        "verdict": "PASS_ASSIGNMENT_SPECIFICITY" if real_count > q95 and empirical_p <= 0.05 else "NO_ASSIGNMENT_SPECIFICITY",
        "real_guarded_crossings": real_count,
        "n_nulls": N_NULLS,
        "null_mean_crossings": float(summary["crossings"].mean()),
        "null_sd_crossings": float(summary["crossings"].std(ddof=1)),
        "null_min_crossings": int(summary["crossings"].min()),
        "null_max_crossings": int(summary["crossings"].max()),
        "null_q95_crossings": q95,
        "nulls_at_or_above_real": exceed,
        "empirical_one_sided_p": empirical_p,
        "safety_match": {
            "same_gate_open_set": True,
            "same_condition_layer_action_multiset": True,
            "same_nonclosure_zero_action_set": True,
        },
        "claim_boundary": (
            "Policy-to-trajectory assignment specificity only; no correctness, open-ended generation, "
            "deployment-control, universal-dynamics or hidden-ontology claim."
        ),
    }
    with open(OUT / "verdict.json", "w", encoding="utf-8") as handle:
        json.dump(verdict, handle, indent=2)
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
