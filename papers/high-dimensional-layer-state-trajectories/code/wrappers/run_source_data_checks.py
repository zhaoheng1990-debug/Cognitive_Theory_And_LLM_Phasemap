#!/usr/bin/env python
"""Smoke-check staged Source Data files without using local workstation paths.

This checker is intentionally small and dependency-free. It validates the
repository-staging package, not the historical raw analysis scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_SOURCE_FILES = [
    "figure_contracts.csv",
    "figure_detail_audit_v0_6.csv",
    "export_qa_summary.csv",
    "figure_freeze_manifest_v0_4.csv",
]


def parse_simple_yaml(path: Path) -> dict[str, Any]:
    """Parse the limited key/value YAML used by code/config/example_paths.yaml."""
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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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
    parser.add_argument("--strict", action="store_true", help="Return non-zero if optional QA files are missing.")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    config = parse_simple_yaml(config_path)
    repo_root = resolve_repo_root(config, config_path, args.repo_root)

    data_cfg = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    figures_cfg = config.get("figures", {}) if isinstance(config.get("figures"), dict) else {}
    source_data_dir = repo_root / str(data_cfg.get("source_data_dir", "data/source_data"))
    metadata_dir = repo_root / str(data_cfg.get("metadata_dir", "metadata"))
    output_dir = repo_root / str(data_cfg.get("output_dir", "outputs"))
    manifest_path = repo_root / str(figures_cfg.get("source_data_manifest", "metadata/source_data_manifest_v0_1.csv"))
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "repo_root": ".",
        "source_data_dir": display_path(source_data_dir, repo_root),
        "metadata_dir": display_path(metadata_dir, repo_root),
        "manifest_path": display_path(manifest_path, repo_root),
        "checks": [],
        "missing": [],
        "warnings": [],
    }

    if not source_data_dir.exists():
        report["missing"].append(display_path(source_data_dir, repo_root))
    if not metadata_dir.exists():
        report["missing"].append(display_path(metadata_dir, repo_root))
    if not manifest_path.exists():
        report["missing"].append(display_path(manifest_path, repo_root))

    source_files = sorted(p.name for p in source_data_dir.glob("*.csv")) if source_data_dir.exists() else []
    report["source_csv_count"] = len(source_files)
    for file_name in REQUIRED_SOURCE_FILES:
        path = source_data_dir / file_name
        if path.exists():
            rows = read_csv_rows(path)
            report["checks"].append({"file": file_name, "rows": len(rows), "status": "ok"})
        else:
            report["missing"].append(display_path(path, repo_root))

    if manifest_path.exists():
        rows = read_csv_rows(manifest_path)
        report["manifest_rows"] = len(rows)
        manifest_file_keys = [k for k in (rows[0].keys() if rows else []) if "file" in k.lower() or "path" in k.lower()]
        if not manifest_file_keys:
            report["warnings"].append("No obvious file/path column found in source_data_manifest_v0_1.csv.")
        else:
            checked = 0
            for row in rows:
                for key in manifest_file_keys:
                    value = (row.get(key) or "").strip()
                    if value and value.endswith(".csv"):
                        candidate = source_data_dir / Path(value).name
                        checked += 1
                        if not candidate.exists():
                            report["missing"].append(display_path(candidate, repo_root))
            report["manifest_file_references_checked"] = checked

    report_path = output_dir / "source_data_check_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")
    print(f"Source CSV files: {report.get('source_csv_count', 0)}")
    print(f"Missing items: {len(report['missing'])}")
    print(f"Warnings: {len(report['warnings'])}")

    if report["missing"] and (args.strict or not source_data_dir.exists() or not manifest_path.exists()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
