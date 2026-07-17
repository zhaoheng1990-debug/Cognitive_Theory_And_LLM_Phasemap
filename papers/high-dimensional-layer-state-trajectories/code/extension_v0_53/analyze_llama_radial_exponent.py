"""Dose the Llama alignment statistic with radial norm exponent gamma."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from analyze_llama_residual_coordinates import (
    ORIGINAL_TASKS,
    bootstrap_mean,
    donor_null_means,
    metrics,
    normalize,
)


ROOT = Path(__file__).resolve().parent
GAMMAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)


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
        default=ROOT / "outputs/llama_radial_exponent",
    )
    parser.add_argument("--donor-nulls", type=int, default=199)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=2026071723)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def exponent_states(states: np.ndarray, gamma: float) -> np.ndarray:
    values = states.astype(np.float32)
    directions = normalize(values)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    relative = norms / np.maximum(norms[:, :1], 1e-8)
    return directions * np.power(relative, gamma)


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
    condition_index = 0
    for task in audit["task"]:
        task_dir = args.input_dir / task
        prompt_ids = pd.read_csv(task_dir / "prompt_manifest.csv", encoding="utf-8-sig")[
            "prompt_id"
        ]
        base = np.asarray(
            np.load(task_dir / "consistent_block_states_float16.npy", mmap_mode="r"),
            dtype=np.float32,
        )
        for gamma in GAMMAS:
            states = exponent_states(base, gamma)
            rng = np.random.default_rng(args.seed + condition_index * 10_000)
            real = metrics(states)
            donor = donor_null_means(states, rng, args.donor_nulls, device)
            summary = {
                "task": task,
                "task_scope": "original" if task in ORIGINAL_TASKS else "independent",
                "gamma": gamma,
                "n_prompts": len(states),
            }
            for name, values in real.items():
                observed = float(values.mean())
                low, high = bootstrap_mean(values, rng, args.bootstrap)
                null = donor[name]
                null_mean = float(null.mean())
                summary[f"real_{name}"] = observed
                summary[f"real_{name}_ci95_low"] = low
                summary[f"real_{name}_ci95_high"] = high
                summary[f"donor_{name}_mean"] = null_mean
                summary[f"donor_{name}_q95"] = float(np.quantile(null, 0.95))
                summary[f"donor_{name}_p"] = float(
                    (np.sum(null >= observed) + 1) / (len(null) + 1)
                )
                summary[f"effect_{name}"] = observed - null_mean
                for prompt_id, value in zip(prompt_ids, values):
                    prompt_rows.append(
                        {
                            "task": task,
                            "gamma": gamma,
                            "prompt_id": prompt_id,
                            "metric": name,
                            "value": float(value),
                        }
                    )
                for null_index, value in enumerate(null):
                    null_rows.append(
                        {
                            "task": task,
                            "gamma": gamma,
                            "metric": name,
                            "null_index": null_index,
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
            print(f"[{task} gamma={gamma:.2f}] pass={summary['passes_joint_direction_gate']}", flush=True)
            condition_index += 1

    summary = pd.DataFrame(summary_rows)
    dose_rows = []
    for task, frame in summary.groupby("task", sort=False):
        row = {"task": task, "task_scope": frame["task_scope"].iloc[0]}
        for metric in ("future_alignment", "leave_one_out_alignment"):
            row[f"rho_gamma_effect_{metric}"] = float(
                spearmanr(frame["gamma"], frame[f"effect_{metric}"]).statistic
            )
        dose_rows.append(row)
    dose = pd.DataFrame(dose_rows)
    monotonic_count = int(
        (
            (dose["rho_gamma_effect_future_alignment"] >= 0.8)
            & (dose["rho_gamma_effect_leave_one_out_alignment"] >= 0.8)
        ).sum()
    )
    gamma_zero_passes = int(
        summary[summary["gamma"] == 0]["passes_joint_direction_gate"]
        .astype(str)
        .str.lower()
        .eq("true")
        .sum()
    )
    gamma_high_passes = int(
        summary[summary["gamma"] == 1.25]["passes_joint_direction_gate"]
        .astype(str)
        .str.lower()
        .eq("true")
        .sum()
    )
    radial_dose = bool(
        monotonic_count >= 6 and gamma_zero_passes <= 1 and gamma_high_passes >= 6
    )
    gate = {
        "window": "llama_radial_exponent",
        "monotonic_task_count": monotonic_count,
        "gamma_zero_joint_pass_count": gamma_zero_passes,
        "gamma_1_25_joint_pass_count": gamma_high_passes,
        "radial_exponent_dose_response_support": radial_dose,
        "interpretation": (
            "the raw alignment statistic is controlled by radial norm scaling"
            if radial_dose
            else "radial exponent dose response not closed"
        ),
        "claim_boundary": (
            "diagnostic coordinate transformation for one Llama checkpoint; not a functional, "
            "semantic, geodesic or architecture-wide law"
        ),
    }
    summary.to_csv(args.output_dir / "llama_radial_exponent_summary.csv", index=False)
    dose.to_csv(args.output_dir / "llama_radial_exponent_monotonicity.csv", index=False)
    pd.DataFrame(prompt_rows).to_csv(
        args.output_dir / "llama_radial_exponent_prompt_metrics.csv", index=False
    )
    pd.DataFrame(null_rows).to_csv(
        args.output_dir / "llama_radial_exponent_donor_nulls.csv", index=False
    )
    (args.output_dir / "llama_radial_exponent_gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    config = {
        "input_dir": str(args.input_dir.resolve()),
        "gammas": list(GAMMAS),
        "donor_nulls": args.donor_nulls,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "device": device,
        "protocol": str(
            (ROOT / "outputs/llama_radial_exponent_protocol_v0_1.md").resolve()
        ),
    }
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
