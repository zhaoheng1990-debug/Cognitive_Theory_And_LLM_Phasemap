"""Test whether long-range CKA is explained by static prompt fingerprints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_relational_continuity import (
    EPS,
    cka_metrics,
    empirical_p,
    normalize,
    random_conditions,
    steps_from_states,
    stratified_derangement,
    trained_conditions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/prompt_fingerprint_followup"),
    )
    parser.add_argument("--null-repeats", type=int, default=199)
    parser.add_argument("--max-prompts", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2026071710)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def prompt_fingerprint(steps: np.ndarray) -> np.ndarray:
    return normalize(normalize(steps).sum(axis=1))


def fingerprint_explained_fraction(steps: np.ndarray) -> float:
    directions = normalize(steps)
    fingerprint = prompt_fingerprint(steps)
    coefficients = np.sum(directions * fingerprint[:, None, :], axis=-1)
    return float(np.mean(coefficients * coefficients))


def fingerprint_preserving_donor(
    steps: np.ndarray, strata: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    directions = normalize(steps)
    norms = np.linalg.norm(steps, axis=-1, keepdims=True)
    fingerprint = prompt_fingerprint(steps)
    coefficients = np.sum(directions * fingerprint[:, None, :], axis=-1)
    coefficients = np.clip(coefficients, -0.999999, 0.999999)
    result = np.empty_like(directions)
    for layer in range(directions.shape[1]):
        donors = stratified_derangement(strata, rng)
        proposal = directions[donors, layer]
        projection = np.sum(proposal * fingerprint, axis=1, keepdims=True)
        orthogonal = proposal - projection * fingerprint
        orthogonal_norm = np.linalg.norm(orthogonal, axis=1, keepdims=True)
        weak = orthogonal_norm[:, 0] < 1e-6
        if np.any(weak):
            fallback = np.roll(proposal, 1, axis=1)
            fallback -= (
                np.sum(fallback * fingerprint, axis=1, keepdims=True) * fingerprint
            )
            orthogonal[weak] = fallback[weak]
            orthogonal_norm = np.linalg.norm(orthogonal, axis=1, keepdims=True)
        orthogonal /= np.maximum(orthogonal_norm, EPS)
        coefficient = coefficients[:, layer, None]
        result[:, layer] = (
            coefficient * fingerprint
            + np.sqrt(np.maximum(1.0 - coefficient * coefficient, 0.0)) * orthogonal
        )
    return result * norms


def analyze_condition(
    steps: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
    repeats: int,
) -> tuple[list[dict], dict]:
    real = cka_metrics(steps)
    null_values = {name: [] for name in real}
    rows = []
    for null_index in range(repeats):
        null_steps = fingerprint_preserving_donor(steps, strata, rng)
        metrics = cka_metrics(null_steps)
        row = {"null_index": null_index}
        for name, value in metrics.items():
            null_values[name].append(value)
            row[name] = value
        rows.append(row)
    summary: dict[str, object] = {
        "fingerprint_explained_fraction": fingerprint_explained_fraction(steps)
    }
    comparisons = []
    for name, real_value in real.items():
        null = np.asarray(null_values[name], dtype=float)
        summary[f"real_{name}"] = real_value
        summary[f"fingerprint_null_{name}_mean"] = float(null.mean())
        summary[f"fingerprint_null_{name}_q95"] = float(np.quantile(null, 0.95))
        summary[f"fingerprint_null_{name}_p"] = empirical_p(real_value, null)
        comparisons.append(
            summary[f"fingerprint_null_{name}_p"] <= 0.05
            and real_value > summary[f"fingerprint_null_{name}_q95"]
        )
    summary["passes_beyond_fingerprint_gate"] = bool(all(comparisons))
    return rows, summary


def residual_gain_diagnostic(root: Path) -> pd.DataFrame:
    source = root / "inputs/execution_states/qwen"
    records = []
    for path in sorted(source.glob("*_hidden_float16.npy")):
        if "alpha_0_native" in path.name:
            continue
        states = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
        steps = steps_from_states(states)
        metrics = cka_metrics(steps)
        name = path.stem
        state_type = (
            "random_initialization" if name.startswith("random_initialization") else "trained_checkpoint"
        )
        order = next(
            value for value in ("fixed_permutation", "reverse", "native") if value in name
        )
        alpha_text = name.split("_alpha_", 1)[1].split(f"_{order}", 1)[0]
        alpha = float(alpha_text.replace("p", "."))
        records.append(
            {
                "state_type": state_type,
                "residual_alpha": alpha,
                "executed_order": order,
                "cross_half_unbiased_cka": metrics["cross_half_unbiased_cka"],
                "quarter_unbiased_cka": metrics["quarter_unbiased_cka"],
                "fingerprint_explained_fraction": fingerprint_explained_fraction(steps),
                "source": str(path),
            }
        )
    return pd.DataFrame(records)


def run_self_test() -> None:
    rng = np.random.default_rng(53)
    prompts, layers, dim = 24, 12, 64
    strata = np.repeat(np.arange(4), prompts // 4)
    fingerprints = normalize(rng.standard_normal((prompts, dim)).astype(np.float32))
    steps = np.empty((prompts, layers, dim), dtype=np.float32)
    for layer in range(layers):
        noise = normalize(rng.standard_normal((prompts, dim)).astype(np.float32))
        steps[:, layer] = normalize(0.8 * fingerprints + 0.2 * noise)
    null = fingerprint_preserving_donor(steps, strata, rng)
    real_coefficients = np.sum(
        normalize(steps) * prompt_fingerprint(steps)[:, None, :], axis=-1
    )
    null_coefficients = np.sum(
        normalize(null) * prompt_fingerprint(steps)[:, None, :], axis=-1
    )
    if not np.allclose(real_coefficients, null_coefficients, atol=1e-5):
        raise AssertionError("fingerprint projection schedule was not preserved")
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    root = Path.cwd()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    conditions = list(trained_conditions(root, args.max_prompts))
    conditions.extend(random_conditions(root, args.max_prompts))
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
        rows, summary = analyze_condition(
            steps_from_states(states), strata, rng, args.null_repeats
        )
        keys = {
            "task": task,
            "model": model,
            "state_type": state_type,
            "init_seed": init_seed,
            "donor_stratum": stratum_column,
        }
        null_frames.append(pd.DataFrame(rows).assign(**keys))
        summary.update(keys)
        summaries.append(summary)
        print(
            f"[{task} {model} {state_type} {init_seed}] "
            f"fingerprint={summary['fingerprint_explained_fraction']:.4f} "
            f"gate={summary['passes_beyond_fingerprint_gate']}",
            flush=True,
        )

    null_frame = pd.concat(null_frames, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    null_frame.to_csv(
        args.output_dir / "fingerprint_null_distributions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_frame.to_csv(
        args.output_dir / "fingerprint_condition_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gain_frame = residual_gain_diagnostic(root)
    gain_frame.to_csv(
        args.output_dir / "residual_gain_cka_diagnostic.csv",
        index=False,
        encoding="utf-8-sig",
    )

    random_frame = summary_frame[summary_frame["state_type"].eq("random_initialization")]
    random_counts = (
        random_frame.groupby("model")["passes_beyond_fingerprint_gate"].sum().astype(int).to_dict()
    )
    random_support = bool(all(count >= 2 for count in random_counts.values()))
    trained = summary_frame[summary_frame["state_type"].eq("trained_checkpoint")]
    gate = {
        "window": "prompt_fingerprint_followup",
        "random_seed_pass_counts": random_counts,
        "random_support_beyond_fingerprint": random_support,
        "trained_pass_count": int(trained["passes_beyond_fingerprint_gate"].sum()),
        "trained_total": int(len(trained)),
        "interpretation": (
            "multi-step dynamics beyond static prompt fingerprint"
            if random_support
            else "random CKA is bounded by prompt-fingerprint carriage"
        ),
        "claim_boundary": (
            "fingerprint-null audit of update geometry; no semantic, endpoint, "
            "geodesic or functional claim"
        ),
    }
    (args.output_dir / "fingerprint_gate_decision.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    config = {
        "script": str(Path(__file__).resolve()),
        "null_repeats": args.null_repeats,
        "max_prompts": args.max_prompts,
        "seed": args.seed,
        "protocol": str(
            (root / "outputs/prompt_fingerprint_followup_protocol_v0_1.md").resolve()
        ),
    }
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
