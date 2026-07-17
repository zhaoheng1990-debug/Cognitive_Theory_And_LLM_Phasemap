"""Separate Llama raw trajectory direction from radial norm schedules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analyze_llama_residual_coordinates import (
    EPS,
    INDEPENDENT_TASKS,
    ORIGINAL_TASKS,
    bootstrap_mean,
    donor_null_means,
    metrics,
    normalize,
)


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "inputs/llama_residual_states",
    )
    parser.add_argument(
        "--endpoint-results",
        type=Path,
        default=ROOT / "outputs/llama_residual_coordinates",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/llama_radial_scale",
    )
    parser.add_argument("--donor-nulls", type=int, default=199)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=2026071722)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def surrogate(states: np.ndarray, representation: str) -> np.ndarray:
    values = states.astype(np.float32)
    unit = normalize(values)
    if representation == "direction_only":
        return unit
    if representation == "norm_only_fixed_direction":
        norms = np.linalg.norm(values, axis=-1, keepdims=True)
        return norms * unit[:, :1]
    raise ValueError(representation)


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
    radial_rows = []
    condition_index = 0
    for task in audit["task"]:
        task_dir = args.input_dir / task
        manifest = pd.read_csv(task_dir / "prompt_manifest.csv", encoding="utf-8-sig")
        consistent = np.asarray(
            np.load(task_dir / "consistent_block_states_float16.npy", mmap_mode="r"),
            dtype=np.float32,
        )
        raw_metric = metrics(consistent)
        norms = np.linalg.norm(consistent, axis=-1)
        radial_fraction = (np.diff(norms, axis=1) > 0).mean(axis=1)
        for index, prompt_id in enumerate(manifest["prompt_id"]):
            radial_rows.append(
                {
                    "task": task,
                    "prompt_id": prompt_id,
                    "positive_norm_step_fraction": float(radial_fraction[index]),
                    "final_to_initial_norm_ratio": float(
                        norms[index, -1] / max(norms[index, 0], EPS)
                    ),
                    "raw_future_alignment": float(raw_metric["future_alignment"][index]),
                    "raw_leave_one_out_alignment": float(
                        raw_metric["leave_one_out_alignment"][index]
                    ),
                }
            )
        for representation in ("direction_only", "norm_only_fixed_direction"):
            states = surrogate(consistent, representation)
            rng = np.random.default_rng(args.seed + condition_index * 10_000)
            real = metrics(states)
            donor = donor_null_means(states, rng, args.donor_nulls, device)
            for null_index in range(args.donor_nulls):
                null_rows.append(
                    {
                        "task": task,
                        "representation": representation,
                        "null_index": null_index,
                        "future_alignment": donor["future_alignment"][null_index],
                        "leave_one_out_alignment": donor["leave_one_out_alignment"][null_index],
                    }
                )
            summary = {
                "task": task,
                "task_scope": "original" if task in ORIGINAL_TASKS else "independent",
                "representation": representation,
                "n_prompts": len(states),
                "n_states": states.shape[1],
            }
            for name, values in real.items():
                observed = float(values.mean())
                low, high = bootstrap_mean(values, rng, args.bootstrap)
                null = donor[name]
                summary[f"real_{name}"] = observed
                summary[f"real_{name}_ci95_low"] = low
                summary[f"real_{name}_ci95_high"] = high
                summary[f"donor_{name}_mean"] = float(null.mean())
                summary[f"donor_{name}_q95"] = float(np.quantile(null, 0.95))
                summary[f"donor_{name}_p"] = float(
                    (np.sum(null >= observed) + 1) / (len(null) + 1)
                )
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
            print(f"[{task} {representation}] pass={summary['passes_joint_direction_gate']}", flush=True)
            condition_index += 1

    summary = pd.DataFrame(summary_rows)
    endpoint = pd.read_csv(args.endpoint_results / "llama_residual_coordinate_summary.csv")

    def count(frame: pd.DataFrame, representation: str, scope: str) -> int:
        values = frame[
            (frame["representation"] == representation) & (frame["task_scope"] == scope)
        ]["passes_joint_direction_gate"]
        return int(values.astype(str).str.lower().eq("true").sum())

    counts = {}
    for representation in (
        "consistent_block_raw",
        "common_rmsnorm",
        "direction_only",
        "norm_only_fixed_direction",
    ):
        frame = endpoint if representation in {"consistent_block_raw", "common_rmsnorm"} else summary
        counts[representation] = {
            scope: count(frame, representation, scope) for scope in ("original", "independent")
        }
    radial_support = bool(
        counts["consistent_block_raw"]["original"] >= 3
        and counts["consistent_block_raw"]["independent"] >= 2
        and counts["common_rmsnorm"]["original"] <= 1
        and counts["common_rmsnorm"]["independent"] <= 1
        and counts["direction_only"]["original"] <= 1
        and counts["direction_only"]["independent"] <= 1
        and counts["norm_only_fixed_direction"]["original"] >= 3
        and counts["norm_only_fixed_direction"]["independent"] >= 2
    )
    gate = {
        "window": "llama_radial_scale_followup",
        "pass_counts": counts,
        "radial_scale_sufficiency_support": radial_support,
        "interpretation": (
            "radial norm schedules are sufficient for the raw-state direction statistic"
            if radial_support
            else "radial-scale explanation not closed"
        ),
        "claim_boundary": (
            "metric explanation for one Llama checkpoint; norm scale may still carry function, "
            "and no semantic, geodesic or architecture-wide claim follows"
        ),
    }
    summary.to_csv(args.output_dir / "llama_radial_surrogate_summary.csv", index=False)
    pd.DataFrame(prompt_rows).to_csv(
        args.output_dir / "llama_radial_surrogate_prompt_metrics.csv", index=False
    )
    pd.DataFrame(null_rows).to_csv(
        args.output_dir / "llama_radial_surrogate_donor_nulls.csv", index=False
    )
    pd.DataFrame(radial_rows).to_csv(
        args.output_dir / "llama_norm_schedule_prompt_diagnostics.csv", index=False
    )
    (args.output_dir / "llama_radial_scale_gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    config = {
        "input_dir": str(args.input_dir.resolve()),
        "endpoint_results": str(args.endpoint_results.resolve()),
        "donor_nulls": args.donor_nulls,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "device": device,
        "protocol": str(
            (ROOT / "outputs/llama_radial_scale_followup_protocol_v0_1.md").resolve()
        ),
    }
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
