"""Confirm random-initialization relational geometry on three new tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_prompt_fingerprint import analyze_condition as analyze_fingerprint
from analyze_relational_continuity import analyze_condition as analyze_strong
from analyze_relational_continuity import steps_from_states


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state-root",
        type=Path,
        default=Path("inputs/random_cross_task_states"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/architecture_continuity"),
    )
    parser.add_argument("--null-repeats", type=int, default=199)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026071711)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extraction = pd.read_csv(
        args.state_root / "extraction_manifest.csv", encoding="utf-8-sig"
    )
    expected = {
        (task, model, seed)
        for task in ("lexical_category", "arithmetic_addition", "arc_challenge")
        for model in ("qwen", "llama", "gemma")
        for seed in (1801, 1802)
    }
    observed = {
        (row.task, row.model, int(row.seed)) for row in extraction.itertuples()
    }
    if observed != expected:
        raise RuntimeError(f"Extraction grid mismatch: missing={expected-observed}, extra={observed-expected}")

    strong_null_frames = []
    fingerprint_null_frames = []
    prompt_frames = []
    summaries = []
    for condition_index, row in enumerate(extraction.itertuples(index=False)):
        manifest = pd.read_csv(
            args.state_root / row.task / "selected_manifest.csv", encoding="utf-8-sig"
        )
        strata = manifest["donor_stratum"].astype(str).to_numpy()
        states = np.asarray(np.load(row.path, mmap_mode="r"), dtype=np.float32)
        if len(states) != len(strata) or not np.isfinite(states).all():
            raise RuntimeError(f"Invalid states or manifest alignment: {row.path}")
        steps = steps_from_states(states)
        base_seed = args.seed + condition_index * 100_000
        prompt_rows, strong_null_rows, strong_summary = analyze_strong(
            steps,
            strata,
            np.random.default_rng(base_seed),
            args.null_repeats,
            args.bootstrap,
            args.block_size,
        )
        fingerprint_rows, fingerprint_summary = analyze_fingerprint(
            steps,
            strata,
            np.random.default_rng(base_seed + 50_000),
            args.null_repeats,
        )
        keys = {"task": row.task, "model": row.model, "init_seed": int(row.seed)}
        prompt_frames.append(pd.DataFrame(prompt_rows).assign(**keys))
        strong_null_frames.append(pd.DataFrame(strong_null_rows).assign(**keys))
        fingerprint_null_frames.append(pd.DataFrame(fingerprint_rows).assign(**keys))
        summary = dict(strong_summary)
        for key, value in fingerprint_summary.items():
            if key.startswith("real_") and key in summary:
                continue
            summary[key] = value
        summary.update(keys)
        summary["passes_all_three_nulls"] = bool(
            summary["passes_strong_multistep_gate"]
            and summary["passes_beyond_fingerprint_gate"]
        )
        summaries.append(summary)
        print(
            f"[{row.task} {row.model} seed {row.seed}] "
            f"strong={summary['passes_strong_multistep_gate']} "
            f"fingerprint={summary['passes_beyond_fingerprint_gate']} "
            f"joint={summary['passes_all_three_nulls']}",
            flush=True,
        )

    prompt_frame = pd.concat(prompt_frames, ignore_index=True)
    strong_null_frame = pd.concat(strong_null_frames, ignore_index=True)
    fingerprint_null_frame = pd.concat(fingerprint_null_frames, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    prompt_frame.to_csv(
        args.output_dir / "random_cross_task_prompt_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    strong_null_frame.to_csv(
        args.output_dir / "random_cross_task_strong_nulls.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fingerprint_null_frame.to_csv(
        args.output_dir / "random_cross_task_fingerprint_nulls.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_frame.to_csv(
        args.output_dir / "random_cross_task_condition_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pair_pass = (
        summary_frame.groupby(["task", "model"])["passes_all_three_nulls"]
        .agg(lambda values: bool(values.all() and len(values) == 2))
        .rename("both_seeds_pass")
        .reset_index()
    )
    pair_pass.to_csv(
        args.output_dir / "random_cross_task_seed_pair_gate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    architecture_counts = (
        pair_pass.groupby("model")["both_seeds_pass"].sum().astype(int).to_dict()
    )
    task_counts = pair_pass.groupby("task")["both_seeds_pass"].sum().astype(int).to_dict()
    architecture_gate = bool(all(count >= 2 for count in architecture_counts.values()))
    task_gate = bool(all(count >= 2 for count in task_counts.values()))
    gate = {
        "window": "random_cross_task_replication",
        "condition_pass_count": int(summary_frame["passes_all_three_nulls"].sum()),
        "condition_total": int(len(summary_frame)),
        "architecture_double_seed_task_counts": architecture_counts,
        "task_double_seed_architecture_counts": task_counts,
        "architecture_gate": architecture_gate,
        "task_gate": task_gate,
        "cross_task_replication_pass": bool(architecture_gate and task_gate),
        "all_lag1_nulls_valid": bool(summary_frame["lag1_null_fidelity_pass"].all()),
        "interpretation": (
            "architecture-compatible preservation of prompt relational geometry across depth"
            if architecture_gate and task_gate
            else "random relation-graph effect did not replicate across task families"
        ),
        "claim_boundary": (
            "relational geometry preservation only; not semantic organization, endpoint "
            "direction, native-order privilege, task function or a geodesic"
        ),
    }
    (args.output_dir / "random_cross_task_gate_decision.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    config = {
        "script": str(Path(__file__).resolve()),
        "state_root": str(args.state_root.resolve()),
        "null_repeats": args.null_repeats,
        "bootstrap_repeats": args.bootstrap,
        "block_size": args.block_size,
        "seed": args.seed,
        "protocol": str(
            (Path.cwd() / "outputs/random_cross_task_protocol_v0_1.md").resolve()
        ),
    }
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
