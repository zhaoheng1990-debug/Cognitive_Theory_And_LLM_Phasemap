#!/usr/bin/env python
"""Validate core reproducibility metadata tables in the staging package."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_TABLES = {
    "model_checkpoints_v0_2.csv": 3,
    "hidden_state_extraction_table_v0_2.csv": 3,
    "random_seeds_and_splits_v0_2.csv": 6,
    "null_control_table_v0_2.csv": 6,
    "classifier_spec_table_v0_2.csv": 6,
    "metric_definition_table_v0_5.csv": 11,
    "gv_runner_provenance_v0_1.csv": 3,
    "source_data_manifest_v0_1.csv": 1,
}


def parse_simple_yaml(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, result)]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = value
    return result


def read_csv_rows(path: Path) -> tuple[list[str], int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), len(rows)


def resolve_repo_root(config: dict[str, Any], config_path: Path, repo_root_arg: str | None) -> Path:
    if repo_root_arg:
        return Path(repo_root_arg).expanduser().resolve()
    configured = str(config.get("project_root", "."))
    if configured in {"", "."}:
        return config_path.resolve().parents[2]
    root = Path(configured).expanduser()
    if not root.is_absolute():
        root = config_path.parent / root
    return root.resolve()


def display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root)).replace("\\", "/")
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="code/config/example_paths.yaml", help="Path to the portable path config.")
    parser.add_argument("--repo-root", default=None, help="Override repository root.")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    config = parse_simple_yaml(config_path)
    repo_root = resolve_repo_root(config, config_path, args.repo_root)
    data_cfg = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    metadata_dir = repo_root / str(data_cfg.get("metadata_dir", "metadata"))
    output_dir = repo_root / str(data_cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "repo_root": ".",
        "metadata_dir": display_path(metadata_dir, repo_root),
        "tables": [],
        "missing": [],
        "warnings": [],
    }

    for file_name, min_rows in REQUIRED_TABLES.items():
        path = metadata_dir / file_name
        if not path.exists():
            report["missing"].append(display_path(path, repo_root))
            continue
        columns, row_count = read_csv_rows(path)
        status = "ok" if row_count >= min_rows and columns else "warning"
        if status != "ok":
            report["warnings"].append(f"{file_name}: expected at least {min_rows} rows and a header, observed {row_count}.")
        report["tables"].append({
            "file": file_name,
            "rows": row_count,
            "columns": columns,
            "min_expected_rows": min_rows,
            "status": status,
        })

    report_path = output_dir / "metric_table_check_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")
    print(f"Tables checked: {len(report['tables'])}")
    print(f"Missing tables: {len(report['missing'])}")
    print(f"Warnings: {len(report['warnings'])}")
    return 1 if report["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
