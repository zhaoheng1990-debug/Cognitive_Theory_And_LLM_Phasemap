# Portable wrapper status

This folder contains the first portable wrapper/checker layer for the repository-staging package.

The scripts here do not rewrite or replace the historical scripts in `../analysis_scripts_raw/` or `../figure_scripts/`. Those files are preserved as provenance. The detected raw Python workstation paths have now been parameterized, while this wrapper layer provides repository-root-relative checks for the staged Source Data, metadata and figure-source package.

## Current wrappers

```text
run_source_data_checks.py     checks Source Data manifest, staged figure-source files and selected QA tables
run_metric_table_checks.py    checks required reproducibility metadata tables and expected row counts
run_figure_source_data_checks.R checks R package availability and staged figure Source Data row/column counts
```

## Example

Run from the repository root:

```bash
python code/wrappers/run_source_data_checks.py --config code/config/example_paths.yaml
python code/wrappers/run_metric_table_checks.py --config code/config/example_paths.yaml
Rscript code/wrappers/run_figure_source_data_checks.R --config code/config/example_paths.yaml
```

Both scripts write small JSON reports to the configured output directory, which defaults to `outputs/`.

## Boundary

Passing these wrappers means the staged Source Data and metadata package is internally inspectable. The R wrapper also verifies the R packages needed for figure-source inspection and produces a small smoke-preview image. It does not mean the raw model-generation scripts or the full R figure-generation workflow have been fully reproduced from raw model outputs. Remaining repository-release decisions are tracked in `../../metadata/public_release_readiness_audit_v1_0.md`, `../../metadata/submission_metadata_gap_audit_v2_0.md`, `../../metadata/source_data_repository_plan_v1_6.md` and the release metadata template.
