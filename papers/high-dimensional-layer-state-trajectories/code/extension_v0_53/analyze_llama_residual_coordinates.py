"""Test whether the Llama raw-residual effect survives coordinate correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
ORIGINAL_TASKS = {
    "relation_graph",
    "lexical_category",
    "arithmetic_addition",
    "arc_challenge",
}
INDEPENDENT_TASKS = {"lexical_disjoint", "addition_disjoint", "arithmetic_mixed"}
REPRESENTATIONS = {
    "cached_mixed_raw": ("cached_mixed_states_float16.npy", False),
    "reextracted_mixed_raw": ("reextracted_mixed_states_float16.npy", False),
    "cached_truncated_raw": ("cached_mixed_states_float16.npy", True),
    "consistent_block_raw": ("consistent_block_states_float16.npy", False),
    "common_rmsnorm": ("common_rmsnorm_states_float16.npy", False),
}
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "inputs/llama_residual_states",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/llama_residual_coordinates",
    )
    parser.add_argument("--donor-nulls", type=int, default=199)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=2026071721)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def normalize(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), EPS)


def safe_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(left * right, axis=-1) / np.maximum(
        np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1), EPS
    )


def metrics(states: np.ndarray) -> dict[str, np.ndarray]:
    steps = np.diff(states.astype(np.float32), axis=1)
    chord = steps.sum(axis=1)
    reverse = np.cumsum(steps[:, ::-1], axis=1)[:, ::-1]
    future = safe_cosine(steps[:, :-1], reverse[:, 1:]).mean(axis=1)
    leave = safe_cosine(steps, chord[:, None] - steps).mean(axis=1)
    return {"future_alignment": future, "leave_one_out_alignment": leave}


def donor_states_metrics(states: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    steps = np.diff(states.astype(np.float32), axis=1)
    directions = normalize(steps)
    norms = np.linalg.norm(steps, axis=-1, keepdims=True)
    donor = np.empty_like(steps)
    for layer in range(steps.shape[1]):
        order = rng.permutation(len(steps))
        donor[:, layer] = directions[order, layer] * norms[:, layer]
    chord = donor.sum(axis=1)
    reverse = np.cumsum(donor[:, ::-1], axis=1)[:, ::-1]
    return {
        "future_alignment": safe_cosine(donor[:, :-1], reverse[:, 1:]).mean(axis=1),
        "leave_one_out_alignment": safe_cosine(donor, chord[:, None] - donor).mean(axis=1),
    }


def donor_null_means(
    states: np.ndarray,
    rng: np.random.Generator,
    repeats: int,
    device: str,
    chunk_size: int = 16,
) -> dict[str, np.ndarray]:
    """Vectorize the same layer-conditioned recipient-norm donor null."""
    steps = np.diff(states.astype(np.float32), axis=1)
    directions = normalize(steps)
    norms = np.linalg.norm(steps, axis=-1, keepdims=True)
    n_prompts, n_steps, _ = steps.shape
    permutations = np.empty((repeats, n_steps, n_prompts), dtype=np.int64)
    for null_index in range(repeats):
        for layer in range(n_steps):
            permutations[null_index, layer] = rng.permutation(n_prompts)
    if device == "cpu":
        output = {
            "future_alignment": np.empty(repeats, dtype=float),
            "leave_one_out_alignment": np.empty(repeats, dtype=float),
        }
        for null_index in range(repeats):
            donor = np.empty_like(steps)
            for layer in range(n_steps):
                donor[:, layer] = (
                    directions[permutations[null_index, layer], layer]
                    * norms[:, layer]
                )
            chord = donor.sum(axis=1)
            reverse = np.cumsum(donor[:, ::-1], axis=1)[:, ::-1]
            output["future_alignment"][null_index] = float(
                safe_cosine(donor[:, :-1], reverse[:, 1:]).mean()
            )
            output["leave_one_out_alignment"][null_index] = float(
                safe_cosine(donor, chord[:, None] - donor).mean()
            )
        return output

    direction_tensor = torch.as_tensor(directions, device=device)
    norm_tensor = torch.as_tensor(norms, device=device)
    by_layer = direction_tensor.permute(1, 0, 2)
    layer_index = torch.arange(n_steps, device=device)[None, :, None]
    future_values = []
    leave_values = []
    with torch.no_grad():
        for start in range(0, repeats, chunk_size):
            end = min(repeats, start + chunk_size)
            permutation = torch.as_tensor(permutations[start:end], device=device)
            expanded_layers = layer_index.expand(permutation.shape)
            donor = by_layer[expanded_layers, permutation].permute(0, 2, 1, 3)
            donor = donor * norm_tensor[None]
            chord = donor.sum(dim=2)
            reverse = donor.flip(2).cumsum(2).flip(2)

            left = donor[:, :, :-1]
            right = reverse[:, :, 1:]
            future = (left * right).sum(-1) / torch.clamp(
                torch.linalg.vector_norm(left, dim=-1)
                * torch.linalg.vector_norm(right, dim=-1),
                min=EPS,
            )
            other = chord[:, :, None] - donor
            leave = (donor * other).sum(-1) / torch.clamp(
                torch.linalg.vector_norm(donor, dim=-1)
                * torch.linalg.vector_norm(other, dim=-1),
                min=EPS,
            )
            future_values.append(future.mean(dim=(1, 2)).cpu().numpy())
            leave_values.append(leave.mean(dim=(1, 2)).cpu().numpy())
            del donor, chord, reverse, left, right, future, other, leave
    return {
        "future_alignment": np.concatenate(future_values).astype(float),
        "leave_one_out_alignment": np.concatenate(leave_values).astype(float),
    }


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, repeats: int):
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    draws = values[indices].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    audit = pd.read_csv(args.input_dir / "extraction_coordinate_audit.csv")
    summary_rows = []
    prompt_rows = []
    null_rows = []
    task_names = audit["task"].tolist()
    condition_index = 0
    for task in task_names:
        manifest = pd.read_csv(args.input_dir / task / "prompt_manifest.csv", encoding="utf-8-sig")
        for representation, (filename, truncate) in REPRESENTATIONS.items():
            states = np.asarray(np.load(args.input_dir / task / filename, mmap_mode="r"), dtype=np.float32)
            if truncate:
                states = states[:, :-1]
            rng = np.random.default_rng(args.seed + condition_index * 10_000)
            real = metrics(states)
            donor = donor_null_means(states, rng, args.donor_nulls, device)
            for null_index in range(args.donor_nulls):
                row = {
                    "task": task,
                    "representation": representation,
                    "null_index": null_index,
                }
                for name, values in donor.items():
                    row[name] = donor[name][null_index]
                null_rows.append(row)
            summary = {
                "task": task,
                "task_scope": "original" if task in ORIGINAL_TASKS else "independent",
                "representation": representation,
                "n_prompts": len(states),
                "n_states": states.shape[1],
                "hidden_dim": states.shape[2],
            }
            for name, values in real.items():
                observed = float(values.mean())
                low, high = bootstrap_mean(values, rng, args.bootstrap)
                null = donor[name]
                p = float((np.sum(null >= observed) + 1) / (len(null) + 1))
                q95 = float(np.quantile(null, 0.95))
                summary[f"real_{name}"] = observed
                summary[f"real_{name}_ci95_low"] = low
                summary[f"real_{name}_ci95_high"] = high
                summary[f"donor_{name}_mean"] = float(null.mean())
                summary[f"donor_{name}_q95"] = q95
                summary[f"donor_{name}_p"] = p
                for prompt_id, value in zip(manifest["prompt_id"], values):
                    prompt_rows.append(
                        {
                            "task": task,
                            "representation": representation,
                            "prompt_id": prompt_id,
                            "metric": name,
                            "value": float(value),
                        }
                    )
            summary["passes_joint_direction_gate"] = bool(
                summary["real_future_alignment"] > summary["donor_future_alignment_q95"]
                and summary["donor_future_alignment_p"] <= 0.05
                and summary["real_leave_one_out_alignment"]
                > summary["donor_leave_one_out_alignment_q95"]
                and summary["donor_leave_one_out_alignment_p"] <= 0.05
            )
            summary_rows.append(summary)
            print(
                f"[{task} {representation}] pass={summary['passes_joint_direction_gate']}",
                flush=True,
            )
            condition_index += 1

    summary = pd.DataFrame(summary_rows)
    original = summary[summary["task_scope"] == "original"]
    independent = summary[summary["task_scope"] == "independent"]

    def pass_count(frame: pd.DataFrame, representation: str) -> int:
        values = frame[frame["representation"] == representation]["passes_joint_direction_gate"]
        return int(values.astype(str).str.lower().eq("true").sum())

    original_counts = {
        representation: pass_count(original, representation) for representation in REPRESENTATIONS
    }
    independent_counts = {
        representation: pass_count(independent, representation) for representation in REPRESENTATIONS
    }
    original_audit = audit[audit["task"].isin(ORIGINAL_TASKS)]
    cache_reproduction = bool(
        (audit["mean_flat_cosine_cached_vs_reextracted"] >= 0.999).all()
    )
    endpoint_jump = bool(
        (original_audit["cached_final_to_penultimate_norm_ratio_median"] > 3).all()
    )
    corrected_representations = (
        "cached_truncated_raw",
        "consistent_block_raw",
        "common_rmsnorm",
    )
    coordinate_mismatch = bool(
        cache_reproduction
        and endpoint_jump
        and original_counts["reextracted_mixed_raw"] >= 3
        and all(original_counts[name] <= 1 for name in corrected_representations)
        and all(independent_counts[name] <= 1 for name in corrected_representations)
    )
    gate = {
        "window": "llama_raw_residual_coordinate_audit",
        "cache_reproduction_pass": cache_reproduction,
        "mixed_endpoint_norm_jump_pass": endpoint_jump,
        "original_task_pass_counts": original_counts,
        "independent_task_pass_counts": independent_counts,
        "coordinate_mismatch_explanation_pass": coordinate_mismatch,
        "interpretation": (
            "raw Llama anomaly is explained by a mixed final-RMSNorm endpoint"
            if coordinate_mismatch
            else "raw Llama anomaly survives or remains unresolved"
        ),
        "claim_boundary": (
            "coordinate-consistency audit of one Llama checkpoint; not a semantic, "
            "geodesic or architecture-wide claim"
        ),
    }
    summary.to_csv(args.output_dir / "llama_residual_coordinate_summary.csv", index=False)
    pd.DataFrame(prompt_rows).to_csv(
        args.output_dir / "llama_residual_prompt_metrics.csv", index=False
    )
    pd.DataFrame(null_rows).to_csv(
        args.output_dir / "llama_residual_donor_nulls.csv", index=False
    )
    audit.to_csv(args.output_dir / "extraction_coordinate_audit.csv", index=False)
    (args.output_dir / "llama_residual_coordinate_gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    config = {
        "input_dir": str(args.input_dir.resolve()),
        "donor_nulls": args.donor_nulls,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "device": device,
        "protocol": str(
            (ROOT / "outputs/llama_raw_residual_protocol_v0_1.md").resolve()
        ),
    }
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
