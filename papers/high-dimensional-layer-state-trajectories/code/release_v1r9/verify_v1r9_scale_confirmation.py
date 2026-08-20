#!/usr/bin/env python3
"""Read-only verification of the V1R9 paired 12B geometry confirmation.

The verifier recomputes summary statistics from processed prompt-level and
endpoint-preserving-null tables. It neither loads a model nor distributes raw
hidden states.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


EXPECTED = {
    "Qwen2.5-1.5B-Instruct": {
        "real": 0.07924080640077591,
        "null_mean": 0.0418033479526639,
        "gain": 0.03743745844811201,
        "standardized": 5.631641158502814,
    },
    "Gemma-3-12B-it-QAT-Q4_0": {
        "real": 0.08640187233686447,
        "null_mean": 0.024027262460440398,
        "gain": 0.06237460987642407,
        "standardized": 3.3923245622911318,
    },
}


def close(observed: float, expected: float, tolerance: float = 1e-10) -> bool:
    return abs(observed - expected) <= tolerance


def check(name: str, observed: object, expected: object, passed: bool) -> dict[str, object]:
    return {"check": name, "observed": observed, "expected": expected, "status": "PASS" if passed else "FAIL"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    geometry_root = data_root / "extension_v0_56" / "newgraph_geometry"
    package_geometry_root = data_root.parent / "extension_v0_56" / "newgraph_geometry"
    if (package_geometry_root / "newgraph_geometry_summary.csv").is_file():
        # Prefer the complete shared extension included in the review package.
        geometry_root = package_geometry_root
    summary_path = geometry_root / "newgraph_geometry_summary.csv"
    manifest_path = geometry_root / "newgraph_prompt_manifest.csv"
    metadata_path = geometry_root / "run_metadata.json"
    if not all(path.is_file() for path in (summary_path, manifest_path, metadata_path)):
        raise RuntimeError("V1R9 paired-geometry Source Data files are incomplete")

    summary = {row["model"]: row for row in csv_rows(summary_path)}
    checks: list[dict[str, object]] = []
    checks.append(check("Exactly two paired checkpoints", sorted(summary), sorted(EXPECTED), sorted(summary) == sorted(EXPECTED)))
    prompt_rows = csv_rows(manifest_path)
    checks.append(check("Fixed paired prompt manifest rows", len(prompt_rows), 240, len(prompt_rows) == 240))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checks.append(check("Declared shared endpoint shuffles", metadata["n_shared_endpoint_shuffles"], 50, metadata["n_shared_endpoint_shuffles"] == 50))

    for model, expected in EXPECTED.items():
        row = summary[model]
        null_path = geometry_root / f"{model}_shared_endpoint_shuffle_null.csv"
        null_rows = csv_rows(null_path)
        null_values = [float(item["chord_alignment"]) for item in null_rows]
        observed = {
            "real": float(row["real_chord_alignment"]),
            "null_mean": float(row["null_chord_alignment_mean"]),
            "gain": float(row["real_minus_null_chord_alignment"]),
            "standardized": float(row["null_standardized_chord_alignment"]),
            "p": float(row["empirical_p_chord_alignment"]),
        }
        for key in ("real", "null_mean", "gain", "standardized"):
            checks.append(check(f"{model} {key}", observed[key], expected[key], close(observed[key], expected[key])))
        calculated_mean = sum(null_values) / len(null_values)
        calculated_p = (1 + sum(value >= observed["real"] for value in null_values)) / (1 + len(null_values))
        checks.extend([
            check(f"{model} null rows", len(null_rows), 50, len(null_rows) == 50),
            check(f"{model} null mean from rows", calculated_mean, observed["null_mean"], close(calculated_mean, observed["null_mean"])),
            check(f"{model} all nulls lower than real", sum(value < observed["real"] for value in null_values), 50, all(value < observed["real"] for value in null_values)),
            check(f"{model} empirical P from rows", calculated_p, 1 / 51, close(calculated_p, 1 / 51)),
        ])

    report = {
        "release": "V1R9 scale confirmation",
        "verification_level": "Processed Source Data recomputation; no model inference was run.",
        "scope": "Checkpoint-specific paired larger-model geometry confirmation; not a pure within-family scale law.",
        "checks": checks,
        "summary": {"checks": len(checks), "passed": sum(item["status"] == "PASS" for item in checks), "failed": sum(item["status"] == "FAIL" for item in checks)},
    }
    output = args.output or Path("v1r9_scale_confirmation_verification.json")
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
