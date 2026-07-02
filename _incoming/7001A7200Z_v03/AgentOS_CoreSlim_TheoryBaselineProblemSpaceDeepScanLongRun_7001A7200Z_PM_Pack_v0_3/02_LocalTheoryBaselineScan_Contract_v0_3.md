# Local Theory Baseline Scan Contract v0.3

## Required local path

```text
C:\Users\ZH\Desktop\AGI\理论基线
```

## Path handling

The path is a local Windows path available in the user's/Codex execution environment.

Do not replace it with `/mnt/data`.
Do not use the chat sandbox as a substitute.
Do not infer success if the directory is missing.

## If present

Produce:

```text
source_inventory_7001A7200Z.csv
source_hash_inventory_7001A7200Z.csv
theory_baseline_scan_coverage_report_7001A7200Z.md
```

## If missing

Return:

```text
BLOCKED_LOCAL_THEORY_BASELINE_PATH_MISSING
```

with:

```text
missing_path_report_7001A7200Z.md
```

and stop before closure decisions.

## Source exclusion

Exclude:

```text
large raw logs
binary files
zip archives
image/video files
temporary folders
cache folders
```

unless directly referenced by a baseline document.

## Source inclusion priority

1. Latest theory baseline files.
2. Problem space baseline files.
3. Summary / convergence reports.
4. Experiment seed files with pending routes.
5. Closure reports with remaining gaps.
6. AgentOS baseline files that explicitly reference theory baseline pending items.
