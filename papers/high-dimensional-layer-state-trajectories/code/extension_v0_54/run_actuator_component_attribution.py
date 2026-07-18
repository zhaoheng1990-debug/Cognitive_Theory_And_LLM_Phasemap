"""Predeclared component ablations for confidence-gated output-boundary control."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
PAPER_ROOT = HERE.parents[1]
EXECUTOR_SOURCE = CODE_ROOT / "extension_v0_52" / "local_transition_executor.py"
CONTROLLER_SOURCE = CODE_ROOT / "extension_v0_52" / "boundary_controller.py"
BOUNDARY_SOURCE = PAPER_ROOT / "source_data" / "source_data" / "extension_v0_52" / "output_boundary_control"

MODEL_WINDOWS = {
    "qwen": {
        "operator_layers": (15, 16, 17, 18, 19),
        "precursor_layers": (17, 18, 19),
        "decision_layers": (20, 21, 22, 23, 24, 25),
    },
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

FULL_ACTIONS = (
    (0.0, 0.0),
    (0.3, 0.3),
    (0.6, 0.3),
    (0.6, 0.6),
    (0.9, 0.3),
    (0.9, 0.6),
    (0.9, 0.9),
    (1.0, 0.6),
    (1.0, 0.9),
    (1.0, 1.2),
)

EXISTING_OUTPUT = {
    "qwen": BOUNDARY_SOURCE / "qwen",
    "llama": BOUNDARY_SOURCE / "llama",
    "gemma": BOUNDARY_SOURCE / "gemma",
}

CONTROL_ORDER = [
    "none",
    "policy_boundary_guarded",
    "policy_gate_operator_only",
    "policy_gate_precursor_only",
    "policy_gate_uniform_max",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODEL_WINDOWS), default=list(MODEL_WINDOWS)
    )
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs")
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def gate_mask(actions: pd.DataFrame) -> pd.Series:
    if actions["gate_open"].dtype == bool:
        return actions["gate_open"].copy()
    return actions["gate_open"].astype(str).str.lower().eq("true")


def make_ablation_actions(guarded: pd.DataFrame) -> dict[str, pd.DataFrame]:
    guarded = guarded.copy()
    open_mask = gate_mask(guarded)

    operator_only = guarded.copy()
    operator_only["beta"] = 0.0
    operator_only["action_label"] = np.where(
        open_mask,
        "gated_real_alpha_beta0",
        "gated_zero",
    )

    precursor_only = guarded.copy()
    precursor_only["alpha"] = 0.0
    precursor_only["action_label"] = np.where(
        open_mask,
        "gated_alpha0_real_beta",
        "gated_zero",
    )

    uniform_max = guarded.copy()
    uniform_max.loc[:, ["alpha", "beta"]] = 0.0
    uniform_max.loc[open_mask, "alpha"] = 1.0
    uniform_max.loc[open_mask, "beta"] = 1.2
    uniform_max["action_label"] = np.where(
        open_mask,
        "gated_uniform_a1.0_b1.2",
        "gated_zero",
    )

    return {
        "policy_boundary_guarded": guarded,
        "policy_gate_operator_only": operator_only,
        "policy_gate_precursor_only": precursor_only,
        "policy_gate_uniform_max": uniform_max,
    }


def evaluate_vocab_top1_with_trace(
    executor,
    model,
    tokenizer,
    test_df: pd.DataFrame,
    cfg,
    stable_ops,
    precursor_dirs,
    control: str,
    actions: pd.DataFrame | None,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_path = out_dir / f"vocab_top1_{control}.csv"
    trace_path = out_dir / f"trace_{control}.csv"
    if output_path.exists() and len(pd.read_csv(output_path)) == len(test_df):
        result = pd.read_csv(output_path)
        trace = pd.read_csv(trace_path) if trace_path.exists() and trace_path.stat().st_size else pd.DataFrame()
        print(f"[{cfg.model_key}] reuse {control}", flush=True)
        return result, trace

    print(f"[{cfg.model_key}] run {control}", flush=True)
    weight = executor.get_lm_head_weight(model).detach().float().to(model.device)
    action_lookup = executor.make_action_lookup(actions)
    layer_modules = executor.get_layers(model)
    hook_layers = sorted(set(cfg.operator_layers) | set(cfg.precursor_layers))
    trace_buffer: list[dict] = []
    rows: list[dict] = []

    texts = test_df["text"].tolist()
    clean_ids_all = test_df["clean_token_id"].to_numpy(np.int64)
    conflict_ids_all = test_df["conflict_token_id"].to_numpy(np.int64)
    prompt_ids_all = test_df["prompt_id"].astype(str).tolist()

    with torch.no_grad():
        for start in range(0, len(test_df), cfg.batch_size):
            end = min(len(test_df), start + cfg.batch_size)
            batch = test_df.iloc[start:end].reset_index(drop=True)
            inputs = tokenizer(
                texts[start:end],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.max_len,
            ).to(model.device)
            clean_ids = torch.tensor(
                clean_ids_all[start:end], dtype=torch.long, device=model.device
            )
            conflict_ids = torch.tensor(
                conflict_ids_all[start:end], dtype=torch.long, device=model.device
            )
            prompt_ids = prompt_ids_all[start:end]

            handles = []
            if control != "none":
                factory = executor.make_combo_hook_factory(
                    cfg=cfg,
                    stable_ops=stable_ops,
                    precursor_dirs=precursor_dirs,
                    control_mode=control,
                    W_device=weight,
                    action_lookup=action_lookup,
                    trace_buffer=trace_buffer,
                    clean_ids_batch=clean_ids,
                    conflict_ids_batch=conflict_ids,
                    prompt_ids_batch=prompt_ids,
                )
                for layer in hook_layers:
                    if layer < len(layer_modules):
                        handles.append(layer_modules[layer].register_forward_hook(factory(layer)))

            outputs = model(**inputs, use_cache=False)
            for handle in handles:
                handle.remove()

            pos = executor.last_positions(inputs["attention_mask"])
            idx = torch.arange(end - start, device=model.device)
            logits = outputs.logits[idx, pos, :].detach().float()
            top_ids = logits.argmax(dim=1)
            clean_logits = logits[idx, clean_ids]
            conflict_logits = logits[idx, conflict_ids]
            clean_ranks = (logits > clean_logits[:, None]).sum(dim=1) + 1
            conflict_ranks = (logits > conflict_logits[:, None]).sum(dim=1) + 1

            for bi, (_, row) in enumerate(batch.iterrows()):
                top_id = int(top_ids[bi].cpu())
                clean_id = int(clean_ids[bi].cpu())
                conflict_id = int(conflict_ids[bi].cpu())
                top_class = (
                    "clean"
                    if top_id == clean_id
                    else "conflict"
                    if top_id == conflict_id
                    else "other"
                )
                rows.append(
                    {
                        "prompt_id": row["prompt_id"],
                        "graph_id": int(row["graph_id"]),
                        "condition": row["condition"],
                        "mechanism": row["mechanism"],
                        "control": control,
                        "clean_token_id": clean_id,
                        "conflict_token_id": conflict_id,
                        "top1_token_id": top_id,
                        "top1_token": tokenizer.decode([top_id]),
                        "top1_class": top_class,
                        "clean_rank": int(clean_ranks[bi].cpu()),
                        "conflict_rank": int(conflict_ranks[bi].cpu()),
                        "clean_minus_conflict_margin": float(
                            (clean_logits[bi] - conflict_logits[bi]).cpu()
                        ),
                    }
                )

            del outputs, logits, inputs

    result = pd.DataFrame(rows)
    trace = pd.DataFrame(trace_buffer)
    result.to_csv(output_path, index=False)
    trace.to_csv(trace_path, index=False)
    return result, trace


def summarize_model(
    model_key: str,
    results: dict[str, pd.DataFrame],
    traces: dict[str, pd.DataFrame],
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = results["none"][
        ["prompt_id", "top1_class", "top1_token_id", "clean_minus_conflict_margin"]
    ].rename(
        columns={
            "top1_class": "baseline_top1_class",
            "top1_token_id": "baseline_top1_token_id",
            "clean_minus_conflict_margin": "baseline_margin",
        }
    )

    summary_rows = []
    prompt_rows = []
    merged_by_control = {}
    for control in CONTROL_ORDER:
        merged = results[control].merge(baseline, on="prompt_id", how="left")
        merged["strict_conflict_to_clean"] = (
            merged["baseline_top1_class"].eq("conflict")
            & merged["top1_class"].eq("clean")
        )
        merged["top1_changed"] = ~merged["top1_token_id"].eq(
            merged["baseline_top1_token_id"]
        )
        merged["margin_shift"] = (
            merged["clean_minus_conflict_margin"] - merged["baseline_margin"]
        )
        merged["model"] = model_key
        merged_by_control[control] = merged
        prompt_rows.append(merged)

        closure = merged[merged["mechanism"].eq("closure")]
        nonclosure = merged[~merged["mechanism"].eq("closure")]
        trace = traces.get(control, pd.DataFrame())
        if len(trace):
            weights = trace["batch_size"].to_numpy(float)
            mean_norm = float(np.average(trace["mean_norm_ratio"], weights=weights))
            max_norm = float(trace["max_norm_ratio"].max())
            all_finite = bool(
                trace["all_finite"].astype(str).str.lower().eq("true").all()
            )
        else:
            mean_norm, max_norm, all_finite = 1.0, 1.0, True

        summary_rows.append(
            {
                "model": model_key,
                "control": control,
                "closure_n": int(len(closure)),
                "strict_conflict_to_clean": int(
                    closure["strict_conflict_to_clean"].sum()
                ),
                "closure_mean_margin_shift": float(closure["margin_shift"].mean()),
                "nonclosure_n": int(len(nonclosure)),
                "nonclosure_top1_changes": int(nonclosure["top1_changed"].sum()),
                "nonclosure_top1_change_rate": float(nonclosure["top1_changed"].mean()),
                "mean_norm_ratio": mean_norm,
                "max_norm_ratio": max_norm,
                "all_finite": all_finite,
            }
        )

    reference = merged_by_control["policy_boundary_guarded"].set_index("prompt_id")
    contrast_rows = []
    rng = np.random.default_rng(20260718)
    for control in CONTROL_ORDER[2:]:
        comparator = merged_by_control[control].set_index("prompt_id")
        paired = reference[["graph_id", "mechanism", "strict_conflict_to_clean"]].join(
            comparator[["strict_conflict_to_clean"]],
            lsuffix="_reference",
            rsuffix="_control",
        )
        closure = paired[paired["mechanism"].eq("closure")]
        ref_only = int(
            (
                closure["strict_conflict_to_clean_reference"]
                & ~closure["strict_conflict_to_clean_control"]
            ).sum()
        )
        control_only = int(
            (
                ~closure["strict_conflict_to_clean_reference"]
                & closure["strict_conflict_to_clean_control"]
            ).sum()
        )
        discordant = ref_only + control_only
        p_value = (
            float(binomtest(ref_only, discordant, 0.5, alternative="two-sided").pvalue)
            if discordant
            else 1.0
        )

        graph_ids = closure["graph_id"].unique()
        boot = np.empty(2_000, dtype=float)
        for draw in range(len(boot)):
            sampled = rng.choice(graph_ids, size=len(graph_ids), replace=True)
            sample = pd.concat(
                [closure[closure["graph_id"].eq(graph)] for graph in sampled],
                ignore_index=True,
            )
            boot[draw] = float(
                sample["strict_conflict_to_clean_reference"].mean()
                - sample["strict_conflict_to_clean_control"].mean()
            )
        ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
        contrast_rows.append(
            {
                "model": model_key,
                "reference": "policy_boundary_guarded",
                "control": control,
                "reference_only_flips": ref_only,
                "control_only_flips": control_only,
                "paired_binomial_p": p_value,
                "flip_rate_difference": float(
                    closure["strict_conflict_to_clean_reference"].mean()
                    - closure["strict_conflict_to_clean_control"].mean()
                ),
                "graph_bootstrap_ci95_low": float(ci_low),
                "graph_bootstrap_ci95_high": float(ci_high),
            }
        )

    summary = pd.DataFrame(summary_rows)
    contrasts = pd.DataFrame(contrast_rows)
    summary.to_csv(out_dir / "actuator_component_summary.csv", index=False)
    contrasts.to_csv(out_dir / "actuator_component_contrasts.csv", index=False)
    pd.concat(prompt_rows, ignore_index=True).to_csv(
        out_dir / "actuator_component_prompt_rows.csv", index=False
    )
    return summary, contrasts


def run_model(model_key: str, output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = output_root / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = EXISTING_OUTPUT[model_key]

    controller = load_module(f"boundary_controller_attribution_{model_key}", CONTROLLER_SOURCE)
    executor = controller.load_executor(EXECUTOR_SOURCE)
    cfg = executor.Config()
    cfg.model_key = model_key
    cfg.model_path = "auto"
    cfg.save_dir = str(out_dir / "local_transition_runtime")
    cfg.n_graphs = 96
    cfg.batch_size = 4
    cfg.n_shuffles = 0
    cfg.include_answer_control = False
    cfg.policy_actions = FULL_ACTIONS
    for field, value in MODEL_WINDOWS[model_key].items():
        setattr(cfg, field, value)

    executor.CFG = cfg
    executor.make_combo_hook_factory = controller.norm_tracing_hook_factory(executor)
    executor.set_seed(cfg.seed)

    config = {
        "audit": "output_boundary_actuator_attribution_v0_1",
        "model": model_key,
        "executor_source": str(EXECUTOR_SOURCE),
        "executor_sha256": sha256(EXECUTOR_SOURCE),
        "controller_source": str(CONTROLLER_SOURCE),
        "controller_sha256": sha256(CONTROLLER_SOURCE),
        "guarded_action_source": str(source_dir / "actions_policy_boundary_guarded.csv"),
        "guarded_action_sha256": sha256(
            source_dir / "actions_policy_boundary_guarded.csv"
        ),
        "frozen_split_sha256": sha256(source_dir / "frozen_group_split.csv"),
        "config": asdict(cfg),
        "controls": CONTROL_ORDER,
        "heldout_tuning": False,
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print(f"[{model_key}] load model", flush=True)
    model, tokenizer = executor.load_model_and_tokenizer(cfg)
    runtime = out_dir / "local_transition_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    dataset = executor.build_dataset(tokenizer, cfg, runtime)
    base, hidden, _ = executor.extract_baseline(model, tokenizer, dataset, cfg, runtime)
    train_index, test_index, split = executor.make_split(base, cfg, runtime)

    existing_split = pd.read_csv(source_dir / "frozen_group_split.csv")
    split_compare = split[["prompt_id", "split"]].sort_values("prompt_id").reset_index(drop=True)
    existing_compare = (
        existing_split[["prompt_id", "split"]]
        .sort_values("prompt_id")
        .reset_index(drop=True)
    )
    if not split_compare.equals(existing_compare):
        raise RuntimeError(f"{model_key}: regenerated graph split differs from frozen split")
    split.to_csv(out_dir / "frozen_group_split_reproduced.csv", index=False)

    base, _ = controller.refit_delta_u_on_train(
        executor, base, train_index, cfg, out_dir
    )
    stable_ops = executor.fit_stable_operators(base, hidden, train_index, cfg, runtime)
    precursor_dirs, _, _ = executor.fit_precursor_and_probes(
        base, hidden, train_index, cfg, runtime
    )
    test = base.iloc[test_index].reset_index(drop=True)

    guarded = pd.read_csv(source_dir / "actions_policy_boundary_guarded.csv")
    expected_pairs = {
        (str(prompt), int(layer))
        for prompt in test["prompt_id"]
        for layer in cfg.operator_layers
    }
    observed_pairs = set(
        zip(guarded["prompt_id"].astype(str), guarded["layer"].astype(int))
    )
    if expected_pairs != observed_pairs:
        raise RuntimeError(f"{model_key}: guarded action rows do not match held-out prompts")

    action_tables = make_ablation_actions(guarded)
    for name, table in action_tables.items():
        table.to_csv(out_dir / f"actions_{name}.csv", index=False)

    results: dict[str, pd.DataFrame] = {}
    traces: dict[str, pd.DataFrame] = {}
    for control in CONTROL_ORDER:
        actions = None if control == "none" else action_tables[control]
        result, trace = evaluate_vocab_top1_with_trace(
            executor,
            model,
            tokenizer,
            test,
            cfg,
            stable_ops,
            precursor_dirs,
            control,
            actions,
            out_dir,
        )
        results[control] = result
        traces[control] = trace

    summary, contrasts = summarize_model(model_key, results, traces, out_dir)

    existing_none = pd.read_csv(source_dir / "vocab_top1_none.csv").sort_values("prompt_id")
    rerun_none = results["none"].sort_values("prompt_id")
    existing_guarded = pd.read_csv(
        source_dir / "vocab_top1_policy_boundary_guarded.csv"
    ).sort_values("prompt_id")
    rerun_guarded = results["policy_boundary_guarded"].sort_values("prompt_id")
    reproduction = {
        "none_top1_exact": bool(
            np.array_equal(
                existing_none["top1_token_id"].to_numpy(),
                rerun_none["top1_token_id"].to_numpy(),
            )
        ),
        "guarded_top1_exact": bool(
            np.array_equal(
                existing_guarded["top1_token_id"].to_numpy(),
                rerun_guarded["top1_token_id"].to_numpy(),
            )
        ),
        "none_margin_max_abs_delta": float(
            np.max(
                np.abs(
                    existing_none["clean_minus_conflict_margin"].to_numpy()
                    - rerun_none["clean_minus_conflict_margin"].to_numpy()
                )
            )
        ),
        "guarded_margin_max_abs_delta": float(
            np.max(
                np.abs(
                    existing_guarded["clean_minus_conflict_margin"].to_numpy()
                    - rerun_guarded["clean_minus_conflict_margin"].to_numpy()
                )
            )
        ),
    }
    (out_dir / "reproduction_check.json").write_text(
        json.dumps(reproduction, indent=2), encoding="utf-8"
    )
    if not reproduction["none_top1_exact"] or not reproduction["guarded_top1_exact"]:
        raise RuntimeError(f"{model_key}: positive or negative reference did not reproduce")

    del model, tokenizer, base, hidden, stable_ops, precursor_dirs
    gc.collect()
    torch.cuda.empty_cache()
    return summary, contrasts


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    contrasts = []
    for model_key in args.models:
        summary, contrast = run_model(model_key, args.output_root)
        summaries.append(summary)
        contrasts.append(contrast)
    pd.concat(summaries, ignore_index=True).to_csv(
        args.output_root / "crossmodel_actuator_component_summary.csv", index=False
    )
    pd.concat(contrasts, ignore_index=True).to_csv(
        args.output_root / "crossmodel_actuator_component_contrasts.csv", index=False
    )


if __name__ == "__main__":
    main()
