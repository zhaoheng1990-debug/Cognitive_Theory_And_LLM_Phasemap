"""Independently verify the Llama endpoint and radial-scale audit chain."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ORIGINAL = {"relation_graph", "lexical_category", "arithmetic_addition", "arc_challenge"}
CORRECTED = ("cached_truncated_raw", "consistent_block_raw", "common_rmsnorm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("states_dir", type=Path)
    parser.add_argument("endpoint_dir", type=Path)
    parser.add_argument("radial_dir", type=Path)
    parser.add_argument("exponent_dir", type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def truth(values: pd.Series) -> np.ndarray:
    return values.astype(str).str.lower().eq("true").to_numpy(bool)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reconstructed_summary(
    prompt: pd.DataFrame,
    nulls: pd.DataFrame,
    group_keys: list[str],
) -> tuple[pd.DataFrame, float]:
    rows = []
    for key, frame in prompt.groupby(group_keys, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        group_values = dict(zip(group_keys, key_tuple))
        row = dict(group_values)
        for metric, metric_frame in frame.groupby("metric"):
            observed = float(metric_frame["value"].mean())
            selected = nulls
            for column, value in group_values.items():
                selected = selected[selected[column] == value]
            if "metric" in selected.columns:
                values = selected[selected["metric"] == metric]["value"].to_numpy(float)
            else:
                values = selected[metric].to_numpy(float)
            row[f"real_{metric}"] = observed
            row[f"donor_{metric}_mean"] = float(values.mean())
            row[f"donor_{metric}_q95"] = float(np.quantile(values, 0.95))
            row[f"donor_{metric}_p"] = float(
                (np.sum(values >= observed) + 1) / (len(values) + 1)
            )
        row["passes_recomputed"] = bool(
            row["real_future_alignment"] > row["donor_future_alignment_q95"]
            and row["donor_future_alignment_p"] <= 0.05
            and row["real_leave_one_out_alignment"]
            > row["donor_leave_one_out_alignment_q95"]
            and row["donor_leave_one_out_alignment_p"] <= 0.05
        )
        rows.append(row)
    return pd.DataFrame(rows), 0.0


def compare_summary(stored: pd.DataFrame, rebuilt: pd.DataFrame, keys: list[str]) -> tuple[float, bool]:
    merged = stored.merge(rebuilt, on=keys, validate="one_to_one")
    errors = []
    for metric in ("future_alignment", "leave_one_out_alignment"):
        for field in ("real", "donor"):
            if field == "real":
                left = f"real_{metric}"
                right = left
            else:
                continue
        errors.append(
            np.max(
                np.abs(
                    merged[f"real_{metric}_x"].to_numpy(float)
                    - merged[f"real_{metric}_y"].to_numpy(float)
                )
            )
        )
        for suffix in ("mean", "q95", "p"):
            column = f"donor_{metric}_{suffix}"
            errors.append(
                np.max(
                    np.abs(
                        merged[f"{column}_x"].to_numpy(float)
                        - merged[f"{column}_y"].to_numpy(float)
                    )
                )
            )
    gate_match = bool(
        np.array_equal(
            truth(merged["passes_joint_direction_gate"]),
            merged["passes_recomputed"].to_numpy(bool),
        )
    )
    return float(max(errors)), gate_match


def count_passes(frame: pd.DataFrame, representation: str, scope: str) -> int:
    selected = frame[
        (frame["representation"] == representation) & (frame["task_scope"] == scope)
    ]
    return int(truth(selected["passes_joint_direction_gate"]).sum())


def main() -> None:
    args = parse_args()
    states_dir = args.states_dir.resolve()
    endpoint_dir = args.endpoint_dir.resolve()
    radial_dir = args.radial_dir.resolve()
    exponent_dir = args.exponent_dir.resolve()

    audit = pd.read_csv(states_dir / "extraction_coordinate_audit.csv")
    task_array_checks = []
    array_hashes = {}
    for _, row in audit.iterrows():
        task = row["task"]
        root = states_dir / task
        arrays = {
            "cached": root / "cached_mixed_states_float16.npy",
            "mixed": root / "reextracted_mixed_states_float16.npy",
            "consistent": root / "consistent_block_states_float16.npy",
            "common": root / "common_rmsnorm_states_float16.npy",
        }
        values = {name: np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32) for name, path in arrays.items()}
        cached_flat = values["cached"].reshape(len(values["cached"]), -1).astype(np.float64)
        mixed_flat = values["mixed"].reshape(len(values["mixed"]), -1).astype(np.float64)
        cosine = np.sum(cached_flat * mixed_flat, axis=1) / np.maximum(
            np.linalg.norm(cached_flat, axis=1) * np.linalg.norm(mixed_flat, axis=1), 1e-12
        )
        norms = {name: np.linalg.norm(value, axis=-1) for name, value in values.items()}
        task_array_checks.append(
            {
                "task": task,
                "shape_pass": all(value.shape == (96, 16, 2048) for value in values.values()),
                "mean_cosine_error": abs(
                    float(cosine.mean()) - row["mean_flat_cosine_cached_vs_reextracted"]
                ),
                "first_15_mixed_vs_hook_max_abs": float(
                    np.max(np.abs(values["mixed"][:, :-1] - values["consistent"][:, :-1]))
                ),
                "mixed_final_vs_common_final_max_abs": float(
                    np.max(np.abs(values["mixed"][:, -1] - values["common"][:, -1]))
                ),
                "cached_norm_ratio_error": abs(
                    float(np.median(norms["cached"][:, -1] / norms["cached"][:, -2]))
                    - row["cached_final_to_penultimate_norm_ratio_median"]
                ),
                "consistent_norm_ratio_error": abs(
                    float(np.median(norms["consistent"][:, -1] / norms["consistent"][:, -2]))
                    - row["consistent_final_to_penultimate_norm_ratio_median"]
                ),
            }
        )
        for name, path in arrays.items():
            array_hashes[f"{task}/{name}"] = sha256(path)
    array_checks = pd.DataFrame(task_array_checks)
    array_pass = bool(
        array_checks["shape_pass"].all()
        and (array_checks["mean_cosine_error"] <= 1e-12).all()
        and (array_checks["first_15_mixed_vs_hook_max_abs"] == 0).all()
        and (array_checks["mixed_final_vs_common_final_max_abs"] <= 0.005).all()
        and (array_checks["cached_norm_ratio_error"] <= 1e-6).all()
        and (array_checks["consistent_norm_ratio_error"] <= 1e-6).all()
    )

    endpoint_summary = pd.read_csv(endpoint_dir / "llama_residual_coordinate_summary.csv")
    endpoint_prompt = pd.read_csv(endpoint_dir / "llama_residual_prompt_metrics.csv")
    endpoint_null = pd.read_csv(endpoint_dir / "llama_residual_donor_nulls.csv")
    endpoint_rebuilt, _ = reconstructed_summary(
        endpoint_prompt, endpoint_null, ["task", "representation"]
    )
    endpoint_error, endpoint_gate_rows_match = compare_summary(
        endpoint_summary, endpoint_rebuilt, ["task", "representation"]
    )
    endpoint_gate = json.loads(
        (endpoint_dir / "llama_residual_coordinate_gate.json").read_text(encoding="utf-8")
    )
    endpoint_counts = {
        representation: {
            scope: count_passes(endpoint_summary, representation, scope)
            for scope in ("original", "independent")
        }
        for representation in endpoint_summary["representation"].unique()
    }
    cache_reproduction = bool(
        (audit["mean_flat_cosine_cached_vs_reextracted"] >= 0.999).all()
    )
    endpoint_jump = bool(
        (
            audit[audit["task"].isin(ORIGINAL)][
                "cached_final_to_penultimate_norm_ratio_median"
            ]
            > 3
        ).all()
    )
    coordinate_mismatch = bool(
        cache_reproduction
        and endpoint_jump
        and endpoint_counts["reextracted_mixed_raw"]["original"] >= 3
        and all(endpoint_counts[name]["original"] <= 1 for name in CORRECTED)
        and all(endpoint_counts[name]["independent"] <= 1 for name in CORRECTED)
    )
    endpoint_gate_match = bool(
        endpoint_gate_rows_match
        and endpoint_gate["cache_reproduction_pass"] == cache_reproduction
        and endpoint_gate["mixed_endpoint_norm_jump_pass"] == endpoint_jump
        and endpoint_gate["coordinate_mismatch_explanation_pass"] == coordinate_mismatch
    )

    radial_summary = pd.read_csv(radial_dir / "llama_radial_surrogate_summary.csv")
    radial_prompt = pd.read_csv(radial_dir / "llama_radial_surrogate_prompt_metrics.csv")
    radial_null = pd.read_csv(radial_dir / "llama_radial_surrogate_donor_nulls.csv")
    radial_rebuilt, _ = reconstructed_summary(
        radial_prompt, radial_null, ["task", "representation"]
    )
    radial_error, radial_rows_match = compare_summary(
        radial_summary, radial_rebuilt, ["task", "representation"]
    )
    radial_gate = json.loads(
        (radial_dir / "llama_radial_scale_gate.json").read_text(encoding="utf-8")
    )
    radial_counts = {}
    for representation in (
        "consistent_block_raw",
        "common_rmsnorm",
        "direction_only",
        "norm_only_fixed_direction",
    ):
        frame = endpoint_summary if representation in {"consistent_block_raw", "common_rmsnorm"} else radial_summary
        radial_counts[representation] = {
            scope: count_passes(frame, representation, scope) for scope in ("original", "independent")
        }
    radial_support = bool(
        radial_counts["consistent_block_raw"]["original"] >= 3
        and radial_counts["consistent_block_raw"]["independent"] >= 2
        and radial_counts["common_rmsnorm"]["original"] <= 1
        and radial_counts["common_rmsnorm"]["independent"] <= 1
        and radial_counts["direction_only"]["original"] <= 1
        and radial_counts["direction_only"]["independent"] <= 1
        and radial_counts["norm_only_fixed_direction"]["original"] >= 3
        and radial_counts["norm_only_fixed_direction"]["independent"] >= 2
    )
    radial_gate_match = bool(
        radial_rows_match
        and radial_gate["pass_counts"] == radial_counts
        and radial_gate["radial_scale_sufficiency_support"] == radial_support
    )

    exponent_summary = pd.read_csv(exponent_dir / "llama_radial_exponent_summary.csv")
    exponent_prompt = pd.read_csv(exponent_dir / "llama_radial_exponent_prompt_metrics.csv")
    exponent_null = pd.read_csv(exponent_dir / "llama_radial_exponent_donor_nulls.csv")
    exponent_rebuilt, _ = reconstructed_summary(
        exponent_prompt, exponent_null, ["task", "gamma"]
    )
    exponent_error, exponent_rows_match = compare_summary(
        exponent_summary, exponent_rebuilt, ["task", "gamma"]
    )
    dose_rows = []
    for task, frame in exponent_summary.groupby("task", sort=False):
        dose_rows.append(
            {
                "task": task,
                "future": float(
                    spearmanr(frame["gamma"], frame["effect_future_alignment"]).statistic
                ),
                "leave": float(
                    spearmanr(
                        frame["gamma"], frame["effect_leave_one_out_alignment"]
                    ).statistic
                ),
            }
        )
    dose = pd.DataFrame(dose_rows)
    monotonic_count = int(((dose["future"] >= 0.8) & (dose["leave"] >= 0.8)).sum())
    gamma_zero = int(
        truth(exponent_summary[exponent_summary["gamma"] == 0]["passes_joint_direction_gate"]).sum()
    )
    gamma_high = int(
        truth(
            exponent_summary[exponent_summary["gamma"] == 1.25][
                "passes_joint_direction_gate"
            ]
        ).sum()
    )
    exponent_support = bool(monotonic_count >= 6 and gamma_zero <= 1 and gamma_high >= 6)
    exponent_gate = json.loads(
        (exponent_dir / "llama_radial_exponent_gate.json").read_text(encoding="utf-8")
    )
    exponent_gate_match = bool(
        exponent_rows_match
        and exponent_gate["monotonic_task_count"] == monotonic_count
        and exponent_gate["gamma_zero_joint_pass_count"] == gamma_zero
        and exponent_gate["gamma_1_25_joint_pass_count"] == gamma_high
        and exponent_gate["radial_exponent_dose_response_support"] == exponent_support
    )

    counts = {
        "tasks": len(audit),
        "endpoint_summary": len(endpoint_summary),
        "endpoint_prompt": len(endpoint_prompt),
        "endpoint_null": len(endpoint_null),
        "radial_summary": len(radial_summary),
        "radial_prompt": len(radial_prompt),
        "radial_null": len(radial_null),
        "exponent_summary": len(exponent_summary),
        "exponent_prompt": len(exponent_prompt),
        "exponent_null": len(exponent_null),
    }
    expected = {
        "tasks": 7,
        "endpoint_summary": 35,
        "endpoint_prompt": 6720,
        "endpoint_null": 6965,
        "radial_summary": 14,
        "radial_prompt": 2688,
        "radial_null": 2786,
        "exponent_summary": 42,
        "exponent_prompt": 8064,
        "exponent_null": 16716,
    }
    passed = bool(
        counts == expected
        and array_pass
        and endpoint_error <= 1e-6
        and radial_error <= 1e-6
        and exponent_error <= 1e-6
        and endpoint_gate_match
        and radial_gate_match
        and exponent_gate_match
    )
    key_files = {
        "extraction_audit": states_dir / "extraction_coordinate_audit.csv",
        "endpoint_summary": endpoint_dir / "llama_residual_coordinate_summary.csv",
        "endpoint_null": endpoint_dir / "llama_residual_donor_nulls.csv",
        "endpoint_gate": endpoint_dir / "llama_residual_coordinate_gate.json",
        "radial_summary": radial_dir / "llama_radial_surrogate_summary.csv",
        "radial_null": radial_dir / "llama_radial_surrogate_donor_nulls.csv",
        "radial_gate": radial_dir / "llama_radial_scale_gate.json",
        "exponent_summary": exponent_dir / "llama_radial_exponent_summary.csv",
        "exponent_null": exponent_dir / "llama_radial_exponent_donor_nulls.csv",
        "exponent_gate": exponent_dir / "llama_radial_exponent_gate.json",
    }
    report = {
        "verification": "PASS" if passed else "FAIL",
        "row_counts": counts,
        "expected_counts": expected,
        "array_checks_pass": array_pass,
        "max_summary_errors": {
            "endpoint": endpoint_error,
            "radial": radial_error,
            "exponent": exponent_error,
        },
        "gate_matches": {
            "endpoint": endpoint_gate_match,
            "radial_strict": radial_gate_match,
            "radial_exponent": exponent_gate_match,
        },
        "recomputed_decisions": {
            "endpoint_only_coordinate_mismatch": coordinate_mismatch,
            "strict_radial_scope_gate": radial_support,
            "radial_exponent_dose_response": exponent_support,
        },
        "sha256": {name: sha256(path) for name, path in key_files.items()},
        "array_sha256": array_hashes,
    }
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
