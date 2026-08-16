#!/usr/bin/env python3
"""Read-only verification of the V1R4 reported endpoints from Source Data.

This script does not run model inference. It recomputes the released endpoint
checks from the processed, row-level Source Data used for the V1R4 Article.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


EXPECTED = {
    "pythia_all_trained_accuracy": 0.8490566,
    "pythia_all_initial_accuracy": 0.4716981,
    "olmo_all_trained_accuracy": 0.9895833,
    "olmo_all_initial_accuracy": 0.5208333,
    "olmo_standardized_gain": 2.606071,
    "qwen_target_crossings": 17,
    "llama_target_crossings": 13,
    "gemma_target_crossings": 9,
    "qwen_non_target_changes": 0,
    "llama_non_target_changes": 2,
    "gemma_non_target_changes": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing the V1R4 Source Data tree.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path. The parent directory is created if needed.",
    )
    return parser.parse_args()


def one_file(root: Path, token: str) -> Path:
    matches = sorted(root.rglob(f"*{token}*"))
    matches = [path for path in matches if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one file containing {token!r}; found {len(matches)}")
    return matches[0]


def rows(root: Path, token: str) -> list[dict[str, str]]:
    with one_file(root, token).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def record(name: str, observed: object, expected: object, passed: bool) -> dict[str, object]:
    return {
        "check": name,
        "observed": observed,
        "expected": expected,
        "status": "PASS" if passed else "FAIL",
    }


def close(observed: float, expected: float, tolerance: float = 1e-6) -> bool:
    return abs(observed - expected) <= tolerance


def by_value(table: list[dict[str, str]], column: str, value: str) -> dict[str, str]:
    matches = [row for row in table if row[column] == value]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one row where {column}={value!r}; found {len(matches)}")
    return matches[0]


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    authoritative = data_root / "authoritative"
    if not authoritative.is_dir():
        raise RuntimeError(f"Missing authoritative Source Data directory: {authoritative}")

    pythia = rows(authoritative, "pythia_summary__component_swap_summary.csv")
    pythia_gate_path = one_file(authoritative, "pythia_gate__gate_decision.json")
    pythia_gate = json.loads(pythia_gate_path.read_text(encoding="utf-8"))
    olmo_accuracy = rows(authoritative, "olmo_accuracy__13_pair_accuracy.csv")
    olmo_effects = json.loads(one_file(authoritative, "olmo_effects__17_factorial_effects.json").read_text(encoding="utf-8"))
    initialization = rows(authoritative, "initialization__crossmodel_initialization_geometry.csv")
    geometry = rows(authoritative, "task_geometry__task_geometry_generalization.csv")
    endpoints = rows(authoritative, "crossmodel_endpoints__08_functional_endpoints.csv")
    qwen_execution = rows(authoritative, "qwen_execution__extended_data7b_order_function_dissociation.csv")
    boundary = rows(authoritative, "boundary__crossmodel_output_boundary.csv")
    window = rows(authoritative, "window__functional_window_confirmation_summary_v0_1.csv")
    path_memory = rows(authoritative, "path_memory__trajectory_incremental_condition_gates.csv")
    mixed = rows(authoritative, "mixed_transfer__primary_summary.csv")
    direction = rows(authoritative, "direction__direction_null_crossmodel_summary.csv")

    checks: list[dict[str, object]] = []
    pythia_1111 = float(by_value(pythia, "condition", "1111")["candidate_pair_accuracy"])
    pythia_0000 = float(by_value(pythia, "condition", "0000")["candidate_pair_accuracy"])
    checks += [
        record("Pythia all-trained candidate-pair accuracy", pythia_1111, EXPECTED["pythia_all_trained_accuracy"], close(pythia_1111, EXPECTED["pythia_all_trained_accuracy"])),
        record("Pythia all-initial candidate-pair accuracy", pythia_0000, EXPECTED["pythia_all_initial_accuracy"], close(pythia_0000, EXPECTED["pythia_all_initial_accuracy"])),
        record("Pythia strict superadditive support", bool(pythia_gate["strict_superadditive_support"]), True, bool(pythia_gate["strict_superadditive_support"])),
    ]

    olmo_111 = float(by_value(olmo_accuracy, "condition", "111")["candidate_pair_accuracy"])
    olmo_000 = float(by_value(olmo_accuracy, "condition", "000")["candidate_pair_accuracy"])
    olmo_gain = float(olmo_effects["standardized_all_trained_gain"]["mean"])
    checks += [
        record("OLMo all-trained candidate-pair accuracy", olmo_111, EXPECTED["olmo_all_trained_accuracy"], close(olmo_111, EXPECTED["olmo_all_trained_accuracy"])),
        record("OLMo all-initial candidate-pair accuracy", olmo_000, EXPECTED["olmo_all_initial_accuracy"], close(olmo_000, EXPECTED["olmo_all_initial_accuracy"])),
        record("OLMo all-trained standardized-margin gain", olmo_gain, EXPECTED["olmo_standardized_gain"], close(olmo_gain, EXPECTED["olmo_standardized_gain"], 1e-5)),
    ]

    random_scaffold = all(float(row["random_alignment_gain_mean"]) > float(row["trained_alignment_gain"]) for row in initialization)
    checks.append(record("All three random initializations exceed matched trained alignment", random_scaffold, True, random_scaffold))
    geometry_pass = all(row["real_exceeds_all_50_shuffles"].lower() == "true" for row in geometry)
    checks.append(record("All released task-model geometry rows exceed 50 shuffles", sum(row["real_exceeds_all_50_shuffles"].lower() == "true" for row in geometry), len(geometry), geometry_pass))

    endpoint_sum = sum(float(row["full_vocabulary_top1_agreement"]) for row in endpoints)
    qwen_non_native_sum = sum(float(row["full_vocab_top1_agreement"]) for row in qwen_execution if row["executed_order"] != "native")
    checks += [
        record("Llama and Gemma non-native full-vocabulary agreement", endpoint_sum, 0, endpoint_sum == 0),
        record("Qwen non-native full-vocabulary agreement", qwen_non_native_sum, 0, qwen_non_native_sum == 0),
    ]

    for model in ("qwen", "llama", "gemma"):
        row = by_value(boundary, "model", model)
        target = int(row["guarded_strict_conflict_to_clean_top1"])
        non_target = int(row["guarded_nonclosure_top1_changes"])
        checks.append(record(f"{model} held-out target crossings", target, EXPECTED[f"{model}_target_crossings"], target == EXPECTED[f"{model}_target_crossings"]))
        checks.append(record(f"{model} non-target changes", non_target, EXPECTED[f"{model}_non_target_changes"], non_target == EXPECTED[f"{model}_non_target_changes"]))

    recurrence = sum(row["role_recurrence"].lower() == "true" for row in window)
    path_gate = sum(row["passes_incremental_ordered_path_gate"].lower() == "true" for row in path_memory)
    mixed_crossings = sum(int(row["strict_crossings"]) for row in mixed)
    checks += [
        record("Functional-window common-role recurrence", recurrence, 0, recurrence == 0),
        record("Ordered-prefix general-path gate", path_gate, 0, path_gate == 0),
        record("Direction-null seeds retained", len(direction), 153, len(direction) == 153),
        record("Mixed-arithmetic strict crossings", mixed_crossings, 2, mixed_crossings == 2),
    ]

    report = {
        "release": "V1R4 / reproducibility release v0.60",
        "verification_level": "Processed Source Data endpoint reproduction; no model inference was run.",
        "source_data_root": "provided through --data-root; local path omitted for portability",
        "checks": checks,
        "summary": {
            "checks": len(checks),
            "passed": sum(check["status"] == "PASS" for check in checks),
            "failed": sum(check["status"] == "FAIL" for check in checks),
        },
    }
    output = args.output or Path("v1r4_source_data_verification.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"VERIFICATION_ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
