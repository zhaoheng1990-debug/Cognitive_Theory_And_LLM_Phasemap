"""Safety-matched action-assignment nulls for cross-checkpoint boundary control."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
EXECUTOR_SOURCE = ROOT / "local_transition_executor.py"
CONTROLLER_SOURCE = ROOT / "boundary_controller.py"
FORMAL_ROOT = ROOT / "inputs" / "crossmodel_output_boundary_control"
OUTPUT_ROOT = ROOT / "crossmodel_safety_matched_null"

MODEL_WINDOWS = {
    "llama": {
        "operator_layers": (8, 9, 10, 11),
        "precursor_layers": (10, 11, 12),
        "decision_layers": (12, 13, 14, 15),
    },
    "gemma": {
        "operator_layers": (13, 14, 15, 16, 17),
        "precursor_layers": (15, 16, 17),
        "decision_layers": (18, 19, 20, 21, 22, 23),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_WINDOWS), default=["llama", "gemma"])
    parser.add_argument("--formal-root", type=Path, default=FORMAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--n-nulls", type=int, default=50)
    parser.add_argument("--seed-base", type=int, default=3042)
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prompt_action_blocks(actions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        str(prompt_id): group.sort_values("layer").reset_index(drop=True)
        for prompt_id, group in actions.groupby("prompt_id", sort=False)
    }


def permute_blocks(
    real_actions: pd.DataFrame,
    closure_meta: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    blocks = prompt_action_blocks(real_actions)
    out_blocks: list[pd.DataFrame] = []
    mapping: list[dict] = []

    for condition, meta in closure_meta.groupby("condition", sort=True):
        open_ids = meta.loc[meta["gate_open"], "prompt_id"].astype(str).tolist()
        closed_ids = meta.loc[~meta["gate_open"], "prompt_id"].astype(str).tolist()
        donor_ids = open_ids.copy()
        if len(open_ids) > 1:
            for _ in range(10_000):
                rng.shuffle(donor_ids)
                if all(recipient != donor for recipient, donor in zip(open_ids, donor_ids)):
                    break
            else:
                raise RuntimeError(f"Could not construct derangement for {condition}")

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
                    "gate_open": True,
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
                    "gate_open": False,
                }
            )

    return pd.concat(out_blocks, ignore_index=True), pd.DataFrame(mapping)


def assert_matched(
    real_actions: pd.DataFrame,
    null_actions: pd.DataFrame,
    closure_meta: pd.DataFrame,
) -> None:
    real = real_actions.merge(
        closure_meta[["prompt_id", "condition"]], on="prompt_id", how="inner"
    )
    columns = ["condition", "layer", "alpha", "beta"]
    real_counts = real.groupby(columns).size().sort_index()
    null_counts = null_actions.groupby(columns).size().sort_index()
    if not real_counts.equals(null_counts):
        raise AssertionError("Null action multiset differs from the real action multiset")
    if set(null_actions["prompt_id"].astype(str)) != set(closure_meta["prompt_id"].astype(str)):
        raise AssertionError("Null prompt set changed")


def strict_crossings(controlled: pd.DataFrame, baseline: pd.DataFrame) -> pd.Series:
    base = baseline[["prompt_id", "top1_class"]].rename(columns={"top1_class": "baseline_top1_class"})
    merged = controlled.merge(base, on="prompt_id", how="left", validate="one_to_one")
    return merged["baseline_top1_class"].eq("conflict") & merged["top1_class"].eq("clean")


def run_model(
    model_key: str,
    formal_root: Path,
    output_root: Path,
    n_nulls: int,
    seed_base: int,
) -> dict:
    formal = formal_root / model_key
    out = output_root / model_key
    out.mkdir(parents=True, exist_ok=True)
    controller = load_module(f"safety_matched_controller_{model_key}", CONTROLLER_SOURCE)
    asa = controller.load_executor(EXECUTOR_SOURCE)
    cfg = asa.Config()
    cfg.model_key = model_key
    cfg.model_path = "auto"
    cfg.save_dir = str(out / "runtime")
    cfg.n_graphs = 96
    cfg.batch_size = 4
    cfg.n_shuffles = 0
    cfg.include_answer_control = False
    for field, value in MODEL_WINDOWS[model_key].items():
        setattr(cfg, field, value)
    asa.CFG = cfg
    asa.make_combo_hook_factory = controller.norm_tracing_hook_factory(asa)
    asa.set_seed(cfg.seed)

    print(f"[{model_key}] reconstruct frozen executor", flush=True)
    model, tokenizer = asa.load_model_and_tokenizer(cfg)
    runtime = out / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    dataset = asa.build_dataset(tokenizer, cfg, runtime)
    base, hidden, _ = asa.extract_baseline(model, tokenizer, dataset, cfg, runtime)
    train_index, test_index, split = asa.make_split(base, cfg, runtime)
    split.to_csv(out / "frozen_group_split.csv", index=False)
    base, _ = controller.refit_delta_u_on_train(asa, base, train_index, cfg, out)
    stable_operators = asa.fit_stable_operators(base, hidden, train_index, cfg, runtime)
    precursor_directions, _, _ = asa.fit_precursor_and_probes(
        base, hidden, train_index, cfg, runtime
    )

    test = base.iloc[test_index].reset_index(drop=True)
    closure = test[test["mechanism"].eq("closure")].reset_index(drop=True)
    actions = pd.read_csv(formal / "actions_policy_boundary_guarded.csv")
    closure_actions = actions[
        actions["prompt_id"].astype(str).isin(closure["prompt_id"].astype(str))
    ].copy()
    prompt_gate = closure_actions.groupby("prompt_id", as_index=False).agg(
        gate_open=("gate_open", "first")
    )
    prompt_gate["gate_open"] = prompt_gate["gate_open"].astype(str).str.lower().eq("true")
    closure_meta = closure[["prompt_id", "condition", "graph_id"]].merge(
        prompt_gate, on="prompt_id", how="left", validate="one_to_one"
    )

    baseline = pd.read_csv(formal / "vocab_top1_none.csv")
    baseline = baseline[baseline["mechanism"].eq("closure")].copy()
    real = pd.read_csv(formal / "vocab_top1_policy_boundary_guarded.csv")
    real = real[real["mechanism"].eq("closure")].copy()
    real_strict = strict_crossings(real, baseline)
    real_count = int(real_strict.sum())

    summaries: list[dict] = []
    maps: list[pd.DataFrame] = []
    rows: list[pd.DataFrame] = []
    for null_id in range(n_nulls):
        seed = seed_base + null_id
        control = f"shuffle_matched_{null_id:03d}"
        print(f"[{model_key}] null {null_id + 1}/{n_nulls} seed={seed}", flush=True)
        null_actions, mapping = permute_blocks(closure_actions, closure_meta, seed)
        assert_matched(closure_actions, null_actions, closure_meta)
        mapping["null_id"] = null_id
        maps.append(mapping)

        cache = out / f"vocab_top1_{control}.csv"
        if cache.exists():
            result = pd.read_csv(cache)
        else:
            result = controller.evaluate_vocab_top1(
                asa,
                model,
                tokenizer,
                closure,
                cfg,
                stable_operators,
                precursor_directions,
                control,
                null_actions,
                out,
            )
        result = result.copy()
        result["null_id"] = null_id
        result["null_seed"] = seed
        result["strict_conflict_to_clean"] = strict_crossings(result, baseline).to_numpy()
        rows.append(result)
        by_condition = result.groupby("condition")["strict_conflict_to_clean"].sum().to_dict()
        summaries.append(
            {
                "model": model_key,
                "null_id": null_id,
                "seed": seed,
                "strict_crossings": int(result["strict_conflict_to_clean"].sum()),
                "controlled_top1_clean": int(result["top1_class"].eq("clean").sum()),
                "mean_margin": float(result["clean_minus_conflict_margin"].mean()),
                "closure_override_crossings": int(by_condition.get("closure_override", 0)),
                "closure_update_crossings": int(by_condition.get("closure_update", 0)),
                "exception_override_crossings": int(by_condition.get("exception_override", 0)),
            }
        )

    summary = pd.DataFrame(summaries)
    pd.concat(rows, ignore_index=True).to_csv(out / "matched_null_rows.csv", index=False)
    pd.concat(maps, ignore_index=True).to_csv(out / "matched_null_permutation_map.csv", index=False)
    summary.to_csv(out / "matched_null_summary.csv", index=False)

    exceed = int(summary["strict_crossings"].ge(real_count).sum())
    empirical_p = (1 + exceed) / (n_nulls + 1)
    q95 = float(summary["strict_crossings"].quantile(0.95, interpolation="higher"))
    verdict = {
        "model": model_key,
        "verdict": (
            "PASS_ASSIGNMENT_SPECIFICITY"
            if real_count > q95 and empirical_p <= 0.05
            else "NO_ASSIGNMENT_SPECIFICITY"
        ),
        "real_guarded_strict_crossings": real_count,
        "n_closure": int(len(closure)),
        "n_nulls": n_nulls,
        "null_mean_strict_crossings": float(summary["strict_crossings"].mean()),
        "null_sd_strict_crossings": float(summary["strict_crossings"].std(ddof=1)),
        "null_min_strict_crossings": int(summary["strict_crossings"].min()),
        "null_max_strict_crossings": int(summary["strict_crossings"].max()),
        "null_q95_strict_crossings": q95,
        "nulls_at_or_above_real": exceed,
        "empirical_one_sided_p": empirical_p,
        "safety_match": {
            "same_gate_open_set": True,
            "same_condition_layer_action_multiset": True,
            "same_nonclosure_zero_action_set": True,
        },
        "claim_boundary": (
            "Trajectory-to-action assignment specificity only; no correctness, open-ended generation, "
            "deployment-control, universal-dynamics or hidden-ontology claim."
        ),
    }
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2), flush=True)
    del model, tokenizer, hidden, stable_operators, precursor_directions
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return verdict


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    verdicts = [
        run_model(
            model_key,
            args.formal_root,
            args.output_root,
            args.n_nulls,
            args.seed_base,
        )
        for model_key in args.models
    ]
    pd.DataFrame(verdicts).to_json(
        args.output_root / "crossmodel_matched_null_verdicts.json",
        orient="records",
        indent=2,
    )


if __name__ == "__main__":
    main()
