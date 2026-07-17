"""Audit ordered trajectory geometry after removing chord self-inclusion.

The original chord-alignment statistic compares every step with a chord that
contains that same step. This script retains that statistic as a reference and
adds future-only and leave-one-step-out coordinates, plus two matched random
walk nulls. It writes prompt-level values so every aggregate can be rebuilt.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-8
MODEL_KEYS = ("qwen", "llama", "gemma")
DEFAULT_SOURCE_ROOT = Path(
    os.environ.get("LLM_DYNAMICS_SOURCE_ROOT", "source_hidden_states")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--random-root", type=Path, default=Path("crossmodel_random_init_geometry_v0_2")
    )
    parser.add_argument("--trained-hidden-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--subset-manifest",
        type=Path,
        default=Path("direct_hidden_geometry_v0_1/controlled_subset_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("architecture_semantic_geometry_v0_1/phase1_debiased_geometry"),
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_KEYS, default=list(MODEL_KEYS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1701, 1702, 1703])
    parser.add_argument("--donor-nulls", type=int, default=200)
    parser.add_argument("--isotropic-nulls", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026071701)
    return parser.parse_args()


def normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, EPS)


def safe_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return numerator / np.maximum(denominator, EPS)


def prompt_metrics_from_steps(steps: np.ndarray) -> dict[str, np.ndarray]:
    """Return one value per prompt for each geometry coordinate."""
    n_prompts, n_steps, _ = steps.shape
    chord = np.sum(steps, axis=1)
    global_alignment = safe_cosine(steps, chord[:, None, :]).mean(axis=1)

    if n_steps > 1:
        reverse_cumulative = np.cumsum(steps[:, ::-1, :], axis=1)[:, ::-1, :]
        future = reverse_cumulative[:, 1:, :]
        future_alignment = safe_cosine(steps[:, :-1, :], future).mean(axis=1)
        lag1 = safe_cosine(steps[:, :-1, :], steps[:, 1:, :]).mean(axis=1)
    else:
        future_alignment = np.full(n_prompts, np.nan)
        lag1 = np.full(n_prompts, np.nan)

    other_steps = chord[:, None, :] - steps
    leave_one_out = safe_cosine(steps, other_steps).mean(axis=1)

    if n_steps > 2:
        lag2 = safe_cosine(steps[:, :-2, :], steps[:, 2:, :]).mean(axis=1)
    else:
        lag2 = np.full(n_prompts, np.nan)

    path_length = np.linalg.norm(steps, axis=-1).sum(axis=1)
    chord_length = np.linalg.norm(chord, axis=-1)
    path_to_chord = path_length / np.maximum(chord_length, EPS)
    return {
        "global_chord_alignment": global_alignment,
        "future_chord_alignment": future_alignment,
        "leave_one_out_alignment": leave_one_out,
        "lag1_step_autocorrelation": lag1,
        "lag2_step_autocorrelation": lag2,
        "path_to_chord_ratio": path_to_chord,
    }


def steps_from_states(states: np.ndarray, representation: str) -> np.ndarray:
    values = np.asarray(states, dtype=np.float32)
    if representation == "unit_hidden_state":
        values = normalize(values)
    elif representation != "raw_residual_state":
        raise ValueError(f"Unknown representation: {representation}")
    return np.diff(values, axis=1)


def layer_conditioned_donor_steps(
    steps: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Keep recipient norms and layer distributions, break cross-layer identity."""
    n_prompts, n_steps, _ = steps.shape
    directions = normalize(steps)
    norms = np.linalg.norm(steps, axis=-1, keepdims=True)
    null = np.empty_like(steps)
    for layer in range(n_steps):
        donor_order = rng.permutation(n_prompts)
        null[:, layer, :] = directions[donor_order, layer, :] * norms[:, layer, :]
    return null


def isotropic_matched_norm_steps(
    steps: np.ndarray, rng: np.random.Generator, simulation_dim: int = 256
) -> np.ndarray:
    """Use a smaller isotropic space; high-dimensional cosine baselines are stable."""
    shape = (steps.shape[0], steps.shape[1], min(simulation_dim, steps.shape[2]))
    directions = normalize(rng.standard_normal(shape).astype(np.float32))
    norms = np.linalg.norm(steps, axis=-1, keepdims=True)
    return directions * norms


def bootstrap_ci(
    values: np.ndarray, rng: np.random.Generator, n_bootstrap: int
) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return np.nan, np.nan
    indices = rng.integers(0, clean.size, size=(n_bootstrap, clean.size))
    means = clean[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def load_conditions(
    model: str,
    seeds: list[int],
    random_root: Path,
    trained_hidden_root: Path,
    source_rows: np.ndarray,
) -> list[tuple[str, str, np.ndarray]]:
    trained_path = trained_hidden_root / f"{model}_hidden_last_token_layers.npy"
    trained_full = np.load(trained_path, mmap_mode="r")
    trained = np.asarray(trained_full[source_rows], dtype=np.float32)
    conditions: list[tuple[str, str, np.ndarray]] = [
        ("trained_checkpoint", "trained", trained)
    ]
    for seed in seeds:
        path = random_root / model / f"random_seed_{seed}_hidden_float16.npy"
        states = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
        if states.shape != trained.shape:
            raise ValueError(f"{path} has shape {states.shape}; expected {trained.shape}")
        conditions.append(("random_initialization", str(seed), states))
    return conditions


def null_tail_p(real: float, null: np.ndarray, higher_is_ordered: bool) -> float:
    if higher_is_ordered:
        extreme = np.sum(null >= real)
    else:
        extreme = np.sum(null <= real)
    return float((extreme + 1) / (null.size + 1))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.subset_manifest, encoding="utf-8-sig")
    source_rows = manifest["source_row"].to_numpy(dtype=int)
    prompt_ids = manifest["prompt_id"].astype(str).tolist()

    prompt_rows: list[dict] = []
    null_rows: list[dict] = []
    summary_rows: list[dict] = []
    master_rng = np.random.default_rng(args.seed)
    representations = ("unit_hidden_state", "raw_residual_state")

    for model_index, model in enumerate(args.models):
        conditions = load_conditions(
            model, args.seeds, args.random_root, args.trained_hidden_root, source_rows
        )
        for condition_index, (state_type, init_seed, states) in enumerate(conditions):
            for representation_index, representation in enumerate(representations):
                local_seed = (
                    args.seed
                    + model_index * 100_000
                    + condition_index * 10_000
                    + representation_index * 1_000
                )
                rng = np.random.default_rng(local_seed)
                steps = steps_from_states(states, representation)
                real = prompt_metrics_from_steps(steps)
                for prompt_index, prompt_id in enumerate(prompt_ids):
                    row = {
                        "model": model,
                        "state_type": state_type,
                        "init_seed": init_seed,
                        "representation": representation,
                        "prompt_id": prompt_id,
                    }
                    row.update({name: float(values[prompt_index]) for name, values in real.items()})
                    prompt_rows.append(row)

                condition_nulls: dict[str, list[float]] = {name: [] for name in real}
                for null_type, repeats in (
                    ("layer_conditioned_donor_walk", args.donor_nulls),
                    ("isotropic_matched_norm_walk", args.isotropic_nulls),
                ):
                    for null_index in range(repeats):
                        if null_type == "layer_conditioned_donor_walk":
                            null_steps = layer_conditioned_donor_steps(steps, rng)
                        else:
                            null_steps = isotropic_matched_norm_steps(steps, rng)
                        metrics = prompt_metrics_from_steps(null_steps)
                        row = {
                            "model": model,
                            "state_type": state_type,
                            "init_seed": init_seed,
                            "representation": representation,
                            "null_type": null_type,
                            "null_index": null_index,
                        }
                        for name, values in metrics.items():
                            value = float(np.nanmean(values))
                            row[name] = value
                            if null_type == "layer_conditioned_donor_walk":
                                condition_nulls[name].append(value)
                        null_rows.append(row)

                summary = {
                    "model": model,
                    "state_type": state_type,
                    "init_seed": init_seed,
                    "representation": representation,
                    "n_prompts": states.shape[0],
                    "n_layers": states.shape[1],
                    "hidden_dim": states.shape[2],
                }
                for name, values in real.items():
                    real_mean = float(np.nanmean(values))
                    ci_low, ci_high = bootstrap_ci(values, master_rng, args.bootstrap)
                    null = np.asarray(condition_nulls[name], dtype=float)
                    higher_is_ordered = name != "path_to_chord_ratio"
                    summary[f"real_{name}"] = real_mean
                    summary[f"real_{name}_ci_low"] = ci_low
                    summary[f"real_{name}_ci_high"] = ci_high
                    summary[f"donor_null_{name}_mean"] = float(np.mean(null))
                    summary[f"donor_null_{name}_q05"] = float(np.quantile(null, 0.05))
                    summary[f"donor_null_{name}_q95"] = float(np.quantile(null, 0.95))
                    summary[f"donor_null_{name}_p"] = null_tail_p(
                        real_mean, null, higher_is_ordered
                    )
                    null_sd = float(np.std(null, ddof=1))
                    signed = real_mean - float(np.mean(null))
                    if not higher_is_ordered:
                        signed *= -1.0
                    summary[f"donor_null_{name}_z_ordered"] = signed / max(null_sd, EPS)
                summary["passes_future_chord_gate"] = bool(
                    summary["donor_null_future_chord_alignment_p"] <= 0.05
                    and summary["real_future_chord_alignment"]
                    > summary["donor_null_future_chord_alignment_q95"]
                )
                summary["passes_leave_one_out_gate"] = bool(
                    summary["donor_null_leave_one_out_alignment_p"] <= 0.05
                    and summary["real_leave_one_out_alignment"]
                    > summary["donor_null_leave_one_out_alignment_q95"]
                )
                summary["passes_joint_debiased_gate"] = bool(
                    summary["passes_future_chord_gate"]
                    and summary["passes_leave_one_out_gate"]
                )
                summary_rows.append(summary)
                print(
                    f"[{model} {state_type} {init_seed} {representation}] "
                    f"future={summary['real_future_chord_alignment']:.4f}, "
                    f"loo={summary['real_leave_one_out_alignment']:.4f}, "
                    f"gate={summary['passes_joint_debiased_gate']}",
                    flush=True,
                )

    prompt_frame = pd.DataFrame(prompt_rows)
    null_frame = pd.DataFrame(null_rows)
    summary_frame = pd.DataFrame(summary_rows)
    prompt_frame.to_csv(
        args.output_dir / "phase1_real_prompt_metrics.csv", index=False, encoding="utf-8-sig"
    )
    null_frame.to_csv(
        args.output_dir / "phase1_null_condition_means.csv", index=False, encoding="utf-8-sig"
    )
    summary_frame.to_csv(
        args.output_dir / "phase1_condition_summary.csv", index=False, encoding="utf-8-sig"
    )

    gate_rows = summary_frame[
        summary_frame["state_type"].eq("random_initialization")
        & summary_frame["representation"].eq("unit_hidden_state")
    ].copy()
    seed_passes = (
        gate_rows.groupby("model")["passes_joint_debiased_gate"].sum().astype(int).to_dict()
    )
    architecture_passes = {model: count >= 2 for model, count in seed_passes.items()}
    trained_rows = summary_frame[
        summary_frame["state_type"].eq("trained_checkpoint")
        & summary_frame["representation"].eq("unit_hidden_state")
    ]
    gate = {
        "phase": 1,
        "primary_representation": "unit_hidden_state",
        "primary_null": "layer_conditioned_donor_walk",
        "random_seed_joint_gate_counts": seed_passes,
        "architecture_passes_debiased_order_gate": architecture_passes,
        "all_architectures_pass": bool(all(architecture_passes.values())),
        "trained_checkpoint_joint_gate": {
            row["model"]: bool(row["passes_joint_debiased_gate"])
            for _, row in trained_rows.iterrows()
        },
        "decision_rule": (
            "random initialization retains debiased order only when future-chord and "
            "leave-one-step-out metrics exceed the donor-null 95th percentile in at least "
            "two of three seeds"
        ),
    }
    (args.output_dir / "phase1_gate_decision.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    config = {
        "script": str(Path(__file__).resolve()),
        "subset_manifest": str(args.subset_manifest.resolve()),
        "trained_hidden_sources": {
            key: str(args.trained_hidden_root / f"{key}_hidden_last_token_layers.npy")
            for key in args.models
        },
        "random_root": str(args.random_root.resolve()),
        "models": args.models,
        "random_initialization_seeds": args.seeds,
        "donor_null_repeats": args.donor_nulls,
        "isotropic_null_repeats": args.isotropic_nulls,
        "bootstrap_repeats": args.bootstrap,
        "seed": args.seed,
        "claim_boundary": (
            "debiased ordered-geometry audit; no semantic, learned-organization or universal-law claim"
        ),
    }
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
