#!/usr/bin/env python3
"""Recompute the eight processed-data verification families for release v2.0.0."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
DATA = ROOT / "source_data" / "source_data"
REPORTS = ROOT / "verification" / "v2_0"


def task(label: str, *parts: str) -> tuple[str, list[str]]:
    return label, [sys.executable, *parts]


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    tasks = [
        task(
            "broad_endpoints",
            str(CODE / "release_v1r9" / "verify_v1r4_source_data.py"),
            "--data-root", str(DATA / "v1r9_final"),
            "--output", str(REPORTS / "broad_endpoints.json"),
        ),
        task(
            "paired_larger_model_geometry",
            str(CODE / "release_v1r9" / "verify_v1r9_scale_confirmation.py"),
            "--data-root", str(DATA / "v1r9_final"),
            "--output", str(REPORTS / "paired_larger_model_geometry.json"),
        ),
        task(
            "actuator_margin_bridge",
            str(CODE / "extension_v0_53" / "verify_actuator_coordinate_margin.py"),
            str(DATA / "extension_v0_53" / "actuator_coordinate_margin"),
            "--matched-verdict", str(DATA / "extension_v0_53" / "actuator_coordinate_margin" / "safety_matched_verdict.json"),
            "--report", str(REPORTS / "actuator_margin_bridge.json"),
        ),
        task(
            "incremental_path_history",
            str(CODE / "extension_v0_53" / "verify_incremental_path_history.py"),
            str(DATA / "extension_v0_53" / "incremental_path_history"),
            "--expected-conditions", "18",
            "--report", str(REPORTS / "incremental_path_history.json"),
        ),
        task("mixed_arithmetic_transfer", str(CODE / "extension_v0_55" / "verify_extension_v0_55.py")),
        task(
            "predecision_coordinate_audit",
            str(CODE / "extension_v0_56" / "verify_predecision_coordinate_audit.py"),
            str(DATA / "extension_v0_56" / "coordinate_audit"),
            "--report", str(REPORTS / "predecision_coordinate_audit.json"),
        ),
        task(
            "empirical_direction_null",
            str(CODE / "extension_v0_56" / "verify_empirical_direction_null.py"),
            str(DATA / "extension_v0_56" / "direction_null"),
            "--report", str(REPORTS / "empirical_direction_null.json"),
        ),
        task(
            "new_graph_geometry",
            str(CODE / "extension_v0_56" / "verify_newgraph_geometry_confirmation.py"),
            "--output-root", str(DATA / "extension_v0_56" / "newgraph_geometry"),
            "--report", str(REPORTS / "new_graph_geometry.json"),
        ),
    ]
    failed: list[str] = []
    for label, command in tasks:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        (REPORTS / f"{label}.stdout.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
        print(f"{'PASS' if result.returncode == 0 else 'FAIL'} {label}")
        if result.returncode != 0:
            failed.append(label)
    if failed:
        print("FAILED: " + ", ".join(failed), file=sys.stderr)
        return 1
    print(f"PASS {len(tasks)}/{len(tasks)} verification tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
