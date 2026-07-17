"""Falsify non-trivial continuity with lag-1 and block-matched nulls.

The primary statistic is prompt identity continuation across distant depth
segments. Strong nulls preserve recipient step norms and either the empirical
one-step cosine distribution or intact four-step fragments.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-8
MODELS = ("qwen", "llama", "gemma")
TASKS = ("relation_graph", "lexical_category", "arithmetic_addition", "arc_challenge")
TRAINED_ROOT = Path(os.environ.get("LLM_DYNAMICS_SOURCE_ROOT", "inputs/trained_relation_states"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/continuity_falsification"),
    )
    parser.add_argument("--null-repeats", type=int, default=199)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--max-prompts", type=int, default=96)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026071709)
    parser.add_argument("--skip-random", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), EPS)


def steps_from_states(states: np.ndarray) -> np.ndarray:
    unit_states = normalize(np.asarray(states, dtype=np.float32))
    return np.diff(unit_states, axis=1)


def derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    if n < 2:
        raise ValueError("At least two prompts are required for donor nulls")
    identity = np.arange(n)
    for _ in range(100):
        donors = rng.permutation(n)
        if np.all(donors != identity):
            return donors
    # A random cyclic shift is a deterministic fixed-point-free fallback.
    return np.roll(identity, int(rng.integers(1, n)))


def stratified_derangement(
    strata: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    strata = np.asarray(strata).astype(str)
    donors = np.empty(len(strata), dtype=int)
    for value in np.unique(strata):
        indices = np.flatnonzero(strata == value)
        if len(indices) < 2:
            raise ValueError(f"Donor stratum {value!r} has fewer than two prompts")
        donors[indices] = indices[derangement(len(indices), rng)]
    return donors


def identity_advantage(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return same-prompt minus other-prompt cosine for every prompt."""
    left_unit = normalize(left)
    right_unit = normalize(right)
    diagonal = np.sum(left_unit * right_unit, axis=1)
    right_sum = np.sum(right_unit, axis=0, keepdims=True)
    other_mean = np.sum(left_unit * (right_sum - right_unit), axis=1) / (len(left) - 1)
    return diagonal - other_mean


def biased_linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    """Standard biased CKA, retained only for estimator-bias diagnostics."""
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    gram_left = left @ left.T
    gram_right = right @ right.T
    numerator = float(np.sum(gram_left * gram_right, dtype=np.float64))
    denominator = np.sqrt(
        float(np.sum(gram_left * gram_left, dtype=np.float64))
        * float(np.sum(gram_right * gram_right, dtype=np.float64))
    )
    return numerator / max(denominator, EPS)


def unbiased_hsic(gram_left: np.ndarray, gram_right: np.ndarray) -> float:
    """U-statistic HSIC estimator with Gram diagonals removed."""
    left = np.asarray(gram_left, dtype=np.float64).copy()
    right = np.asarray(gram_right, dtype=np.float64).copy()
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] != left.shape[1]:
        raise ValueError("HSIC requires two square Gram matrices of equal shape")
    n = left.shape[0]
    if n < 4:
        raise ValueError("Unbiased HSIC requires at least four prompts")
    np.fill_diagonal(left, 0.0)
    np.fill_diagonal(right, 0.0)
    term1 = np.sum(left * right, dtype=np.float64)
    term2 = left.sum() * right.sum() / ((n - 1) * (n - 2))
    term3 = 2.0 * np.sum(left.sum(axis=1) * right.sum(axis=1)) / (n - 2)
    return float((term1 + term2 - term3) / (n * (n - 3)))


def unbiased_linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    """Linear CKA normalized from unbiased HSIC estimates."""
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    gram_left = left @ left.T
    gram_right = right @ right.T
    numerator = unbiased_hsic(gram_left, gram_right)
    left_scale = unbiased_hsic(gram_left, gram_left)
    right_scale = unbiased_hsic(gram_right, gram_right)
    denominator = np.sqrt(max(left_scale, EPS) * max(right_scale, EPS))
    return numerator / max(denominator, EPS)


def cka_metrics(steps: np.ndarray) -> dict[str, float]:
    directions = normalize(steps)
    n_steps = directions.shape[1]
    split = n_steps // 2
    quarter = max(1, n_steps // 4)
    return {
        "cross_half_unbiased_cka": unbiased_linear_cka(
            directions[:, :split].sum(axis=1), directions[:, split:].sum(axis=1)
        ),
        "quarter_unbiased_cka": unbiased_linear_cka(
            directions[:, :quarter].sum(axis=1), directions[:, -quarter:].sum(axis=1)
        ),
    }


def continuity_metrics(steps: np.ndarray) -> dict[str, np.ndarray]:
    directions = normalize(steps)
    n_steps = directions.shape[1]
    split = n_steps // 2
    quarter = max(1, n_steps // 4)
    first_half = directions[:, :split].sum(axis=1)
    second_half = directions[:, split:].sum(axis=1)
    first_quarter = directions[:, :quarter].sum(axis=1)
    last_quarter = directions[:, -quarter:].sum(axis=1)
    lag1 = np.sum(directions[:, :-1] * directions[:, 1:], axis=-1).mean(axis=1)
    lag2 = np.sum(directions[:, :-2] * directions[:, 2:], axis=-1).mean(axis=1)
    return {
        "cross_half_identity_advantage": identity_advantage(first_half, second_half),
        "quarter_identity_advantage": identity_advantage(first_quarter, last_quarter),
        "cross_half_self_cosine": np.sum(
            normalize(first_half) * normalize(second_half), axis=1
        ),
        "lag1_cosine": lag1,
        "lag2_cosine": lag2,
    }


def independent_layer_donor(
    steps: np.ndarray, strata: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    directions = normalize(steps)
    norms = np.linalg.norm(steps, axis=-1, keepdims=True)
    result = np.empty_like(steps)
    for layer in range(steps.shape[1]):
        donors = stratified_derangement(strata, rng)
        result[:, layer] = directions[donors, layer] * norms[:, layer]
    return result


def lag1_matched_markov_donor(
    steps: np.ndarray, strata: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Preserve each layer's empirical lag-1 cosine distribution exactly."""
    directions = normalize(steps)
    norms = np.linalg.norm(steps, axis=-1, keepdims=True)
    n_prompts, n_steps, _ = directions.shape
    null_directions = np.empty_like(directions)
    start_donors = stratified_derangement(strata, rng)
    null_directions[:, 0] = directions[start_donors, 0]
    for layer in range(n_steps - 1):
        donors = stratified_derangement(strata, rng)
        donor_rho = np.sum(
            directions[donors, layer] * directions[donors, layer + 1], axis=1
        )
        donor_rho = np.clip(donor_rho, -0.999999, 0.999999)
        previous = null_directions[:, layer]
        proposal = directions[donors, layer + 1]
        projection = np.sum(proposal * previous, axis=1, keepdims=True)
        orthogonal = proposal - projection * previous
        orthogonal_norm = np.linalg.norm(orthogonal, axis=1, keepdims=True)
        weak = orthogonal_norm[:, 0] < 1e-6
        if np.any(weak):
            fallback = np.roll(proposal, 1, axis=1)
            fallback -= np.sum(fallback * previous, axis=1, keepdims=True) * previous
            orthogonal[weak] = fallback[weak]
            orthogonal_norm = np.linalg.norm(orthogonal, axis=1, keepdims=True)
        orthogonal /= np.maximum(orthogonal_norm, EPS)
        rho = donor_rho[:, None]
        null_directions[:, layer + 1] = (
            rho * previous + np.sqrt(np.maximum(1.0 - rho * rho, 0.0)) * orthogonal
        )
    return null_directions * norms


def contiguous_block_donor(
    steps: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
    block_size: int,
) -> np.ndarray:
    directions = normalize(steps)
    norms = np.linalg.norm(steps, axis=-1, keepdims=True)
    result = np.empty_like(steps)
    for start in range(0, steps.shape[1], block_size):
        stop = min(start + block_size, steps.shape[1])
        donors = stratified_derangement(strata, rng)
        result[:, start:stop] = directions[donors, start:stop] * norms[:, start:stop]
    return result


def empirical_p(real: float, null: np.ndarray) -> float:
    return float((np.sum(null >= real) + 1) / (len(null) + 1))


def bootstrap_ci(
    values: np.ndarray, rng: np.random.Generator, repeats: int
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    draws = rng.integers(0, len(values), size=(repeats, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def task_sources(root: Path) -> dict[str, dict]:
    return {
        "relation_graph": {
            "manifest": root / "direct_hidden_geometry_v0_1/controlled_subset_manifest.csv",
            "hidden": {model: TRAINED_ROOT / f"{model}_hidden_last_token_layers.npy" for model in MODELS},
            "source_row": "source_row",
        },
        "lexical_category": {
            "manifest": root / "inputs/lexical_task/lexical_task_manifest.csv",
            "hidden": {
                model: root / f"inputs/lexical_task/{model}/hidden_last_token_layers_float16.npy"
                for model in MODELS
            },
        },
        "arithmetic_addition": {
            "manifest": root / "inputs/addition_task/arithmetic_task_manifest.csv",
            "hidden": {
                model: root / f"inputs/addition_task/{model}/hidden_last_token_layers_float16.npy"
                for model in MODELS
            },
        },
        "arc_challenge": {
            "manifest": root / "arc_multichoice_geometry_v0_1/arc_challenge_manifest.csv",
            "hidden": {
                model: root / f"arc_multichoice_geometry_v0_1/{model}/hidden_last_token_layers_float16.npy"
                for model in MODELS
            },
        },
    }


def selected_indices(task: str, manifest: pd.DataFrame, max_prompts: int) -> np.ndarray:
    if task in {"lexical_category", "arithmetic_addition"}:
        problems = manifest["problem_id"].drop_duplicates().iloc[: max_prompts // 6]
        return np.flatnonzero(manifest["problem_id"].isin(problems).to_numpy())[:max_prompts]
    return np.arange(min(max_prompts, len(manifest)), dtype=int)


def trained_conditions(root: Path, max_prompts: int):
    for task, source in task_sources(root).items():
        manifest = pd.read_csv(source["manifest"], encoding="utf-8-sig")
        selected = selected_indices(task, manifest, max_prompts)
        hidden_rows = (
            manifest.iloc[selected][source["source_row"]].to_numpy(dtype=int)
            if "source_row" in source
            else selected
        )
        stratum_column = "condition" if task != "arc_challenge" else "answer_key"
        strata = manifest.iloc[selected][stratum_column].astype(str).to_numpy()
        for model in MODELS:
            states = np.asarray(np.load(source["hidden"][model], mmap_mode="r")[hidden_rows])
            yield task, model, "trained_checkpoint", "trained", states, strata, stratum_column


def random_conditions(root: Path, max_prompts: int):
    random_root = root / "crossmodel_random_init_geometry_v0_2"
    manifest = pd.read_csv(
        root / "direct_hidden_geometry_v0_1/controlled_subset_manifest.csv",
        encoding="utf-8-sig",
    ).iloc[:max_prompts]
    strata = manifest["condition"].astype(str).to_numpy()
    for model in MODELS:
        for seed in (1701, 1702, 1703):
            path = random_root / model / f"random_seed_{seed}_hidden_float16.npy"
            states = np.asarray(np.load(path, mmap_mode="r")[:max_prompts])
            yield (
                "relation_graph",
                model,
                "random_initialization",
                str(seed),
                states,
                strata,
                "condition",
            )


def analyze_condition(
    steps: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
    null_repeats: int,
    bootstrap: int,
    block_size: int,
) -> tuple[list[dict], list[dict], dict]:
    real = continuity_metrics(steps)
    real_cka = cka_metrics(steps)
    prompt_rows = []
    for prompt_index in range(len(steps)):
        row = {"prompt_index": prompt_index}
        row.update({name: float(values[prompt_index]) for name, values in real.items()})
        prompt_rows.append(row)

    null_types = {
        "independent_layer_donor": lambda: independent_layer_donor(steps, strata, rng),
        "lag1_matched_markov_donor": lambda: lag1_matched_markov_donor(
            steps, strata, rng
        ),
        f"contiguous_block_donor_k{block_size}": lambda: contiguous_block_donor(
            steps, strata, rng, block_size
        ),
    }
    null_values = {
        null_type: {metric: [] for metric in real} for null_type in null_types
    }
    null_cka_values = {
        null_type: {metric: [] for metric in real_cka} for null_type in null_types
    }
    null_rows = []
    for null_index in range(null_repeats):
        for null_type, constructor in null_types.items():
            null_steps = constructor()
            metrics = continuity_metrics(null_steps)
            cka = cka_metrics(null_steps)
            row = {"null_type": null_type, "null_index": null_index}
            for name, values in metrics.items():
                value = float(np.mean(values))
                null_values[null_type][name].append(value)
                row[name] = value
            for name, value in cka.items():
                null_cka_values[null_type][name].append(value)
                row[name] = value
            null_rows.append(row)

    summary: dict[str, object] = {
        "n_prompts": len(steps),
        "n_steps": steps.shape[1],
        "hidden_dim": steps.shape[2],
    }
    for name, values in real.items():
        real_mean = float(np.mean(values))
        ci_low, ci_high = bootstrap_ci(values, rng, bootstrap)
        summary[f"real_{name}"] = real_mean
        summary[f"real_{name}_ci_low"] = ci_low
        summary[f"real_{name}_ci_high"] = ci_high
        for null_type in null_types:
            null = np.asarray(null_values[null_type][name], dtype=float)
            prefix = f"{null_type}_{name}"
            summary[f"{prefix}_mean"] = float(null.mean())
            summary[f"{prefix}_q95"] = float(np.quantile(null, 0.95))
            summary[f"{prefix}_p"] = empirical_p(real_mean, null)

    for name, real_value in real_cka.items():
        summary[f"real_{name}"] = real_value
        for null_type in null_types:
            null = np.asarray(null_cka_values[null_type][name], dtype=float)
            prefix = f"{null_type}_{name}"
            summary[f"{prefix}_mean"] = float(null.mean())
            summary[f"{prefix}_q95"] = float(np.quantile(null, 0.95))
            summary[f"{prefix}_p"] = empirical_p(real_value, null)

    lag1_null = float(
        summary["lag1_matched_markov_donor_lag1_cosine_mean"]
    )
    lag1_real = float(summary["real_lag1_cosine"])
    summary["lag1_fidelity_absolute_error"] = abs(lag1_real - lag1_null)
    summary["lag1_null_fidelity_pass"] = bool(
        summary["lag1_fidelity_absolute_error"] <= 1e-5
    )
    strong_nulls = (
        "lag1_matched_markov_donor",
        f"contiguous_block_donor_k{block_size}",
    )
    primary_metrics = (
        "cross_half_unbiased_cka",
        "quarter_unbiased_cka",
    )
    comparisons = []
    for null_type in strong_nulls:
        for metric in primary_metrics:
            comparisons.append(
                float(summary[f"{null_type}_{metric}_p"]) <= 0.05
                and float(summary[f"real_{metric}"])
                > float(summary[f"{null_type}_{metric}_q95"])
            )
    summary["passes_strong_multistep_gate"] = bool(
        summary["lag1_null_fidelity_pass"] and all(comparisons)
    )
    return prompt_rows, null_rows, summary


def run_self_test() -> None:
    rng = np.random.default_rng(41)
    prompts, steps, dim = 24, 12, 64
    base = normalize(rng.standard_normal((prompts, dim)).astype(np.float32))
    coherent = np.empty((prompts, steps, dim), dtype=np.float32)
    for layer in range(steps):
        noise = normalize(rng.standard_normal((prompts, dim)).astype(np.float32))
        coherent[:, layer] = normalize(0.85 * base + 0.15 * noise)
    coherent *= rng.uniform(0.5, 1.5, size=(prompts, steps, 1))
    strata = np.repeat(np.arange(4), prompts // 4)
    null = lag1_matched_markov_donor(coherent, strata, rng)
    real_metrics = continuity_metrics(coherent)
    null_metrics = continuity_metrics(null)
    lag1_error = abs(
        float(real_metrics["lag1_cosine"].mean())
        - float(null_metrics["lag1_cosine"].mean())
    )
    if lag1_error > 1e-5:
        raise AssertionError(f"lag-1 preservation failed: {lag1_error}")
    if not (
        real_metrics["cross_half_identity_advantage"].mean()
        > null_metrics["cross_half_identity_advantage"].mean()
    ):
        raise AssertionError("coherent synthetic path did not exceed the Markov null")
    if not cka_metrics(coherent)["cross_half_unbiased_cka"] > cka_metrics(null)[
        "cross_half_unbiased_cka"
    ]:
        raise AssertionError("coherent synthetic CKA did not exceed the Markov null")
    independent_left = rng.standard_normal((96, 512)).astype(np.float32)
    independent_right = rng.standard_normal((96, 512)).astype(np.float32)
    biased = biased_linear_cka(independent_left, independent_right)
    unbiased = unbiased_linear_cka(independent_left, independent_right)
    if not biased > 0.7:
        raise AssertionError("synthetic high-dimensional biased CKA was unexpectedly small")
    if not abs(unbiased) < 0.1:
        raise AssertionError(f"unbiased CKA retained high-dimensional bias: {unbiased}")
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    root = Path.cwd()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    conditions = list(trained_conditions(root, args.max_prompts))
    if not args.skip_random:
        conditions.extend(random_conditions(root, args.max_prompts))

    prompt_frames = []
    null_frames = []
    summaries = []
    for condition_index, (
        task,
        model,
        state_type,
        init_seed,
        states,
        strata,
        stratum_column,
    ) in enumerate(conditions):
        rng = np.random.default_rng(args.seed + condition_index * 100_000)
        steps = steps_from_states(states)
        prompt_rows, null_rows, summary = analyze_condition(
            steps, strata, rng, args.null_repeats, args.bootstrap, args.block_size
        )
        keys = {
            "task": task,
            "model": model,
            "state_type": state_type,
            "init_seed": init_seed,
            "donor_stratum": stratum_column,
        }
        prompt_frame = pd.DataFrame(prompt_rows).assign(**keys)
        null_frame = pd.DataFrame(null_rows).assign(**keys)
        prompt_frames.append(prompt_frame)
        null_frames.append(null_frame)
        summary.update(keys)
        summaries.append(summary)
        print(
            f"[{task} {model} {state_type} {init_seed}] "
            f"half_uCKA={summary['real_cross_half_unbiased_cka']:.4f} "
            f"quarter_uCKA={summary['real_quarter_unbiased_cka']:.4f} "
            f"gate={summary['passes_strong_multistep_gate']}",
            flush=True,
        )

    prompt_frame = pd.concat(prompt_frames, ignore_index=True)
    null_frame = pd.concat(null_frames, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    prompt_frame.to_csv(
        args.output_dir / "continuity_prompt_metrics.csv", index=False, encoding="utf-8-sig"
    )
    null_frame.to_csv(
        args.output_dir / "continuity_strong_nulls.csv", index=False, encoding="utf-8-sig"
    )
    summary_frame.to_csv(
        args.output_dir / "continuity_condition_summary.csv", index=False, encoding="utf-8-sig"
    )

    trained = summary_frame[summary_frame["state_type"].eq("trained_checkpoint")]
    trained_model_counts = (
        trained.groupby("model")["passes_strong_multistep_gate"].sum().astype(int).to_dict()
    )
    trained_task_counts = (
        trained.groupby("task")["passes_strong_multistep_gate"].sum().astype(int).to_dict()
    )
    trained_support = bool(
        trained["passes_strong_multistep_gate"].sum() >= 9
        and all(count >= 3 for count in trained_model_counts.values())
        and all(count >= 2 for count in trained_task_counts.values())
    )
    random_frame = summary_frame[summary_frame["state_type"].eq("random_initialization")]
    random_counts = (
        random_frame.groupby("model")["passes_strong_multistep_gate"].sum().astype(int).to_dict()
        if len(random_frame)
        else {}
    )
    random_support = bool(
        len(random_counts) == 3 and all(count >= 2 for count in random_counts.values())
    )
    if trained_support and random_support:
        interpretation = "architecture-compatible non-trivial multi-step coherence"
    elif trained_support and not random_support:
        interpretation = "training-dependent multi-step coherence"
    elif not trained_support and not random_support:
        interpretation = "only local residual/Markov continuity is supported"
    else:
        interpretation = "random-only pattern requires replication before interpretation"
    gate = {
        "window": "continuity_falsification",
        "trained_pass_count": int(trained["passes_strong_multistep_gate"].sum()),
        "trained_total": int(len(trained)),
        "trained_model_pass_counts": trained_model_counts,
        "trained_task_pass_counts": trained_task_counts,
        "trained_cross_task_support": trained_support,
        "random_seed_pass_counts": random_counts,
        "random_architecture_support": random_support,
        "all_lag1_nulls_valid": bool(summary_frame["lag1_null_fidelity_pass"].all()),
        "interpretation": interpretation,
        "claim_boundary": (
            "prompt-specific multi-step continuation only; no endpoint, semantic, "
            "geodesic, correctness or universal-law claim"
        ),
    }
    (args.output_dir / "continuity_gate_decision.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    config = {
        "script": str(Path(__file__).resolve()),
        "null_repeats": args.null_repeats,
        "bootstrap_repeats": args.bootstrap,
        "max_prompts": args.max_prompts,
        "block_size": args.block_size,
        "seed": args.seed,
        "protocol": str(
            (root / "outputs/continuity_falsification_protocol_v0_1.md").resolve()
        ),
    }
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
