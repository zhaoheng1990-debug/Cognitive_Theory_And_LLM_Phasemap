# Figures 1-4 Source Data Staging

This staging area contains read-only copies of candidate source files for the first four preprint figures. The AGI research corpus remains unchanged.

## Selection decision

- Figures 1-3 use the scope-clean, PSG-corrected final readback-flow package as their canonical candidate source.
- Figure 4 uses the complete CM-4D window scans and direction-isomorphism outputs.
- Earlier semantic-language figure packages are retained outside this staging area for historical audit only.

## Validation

- 22 total staged artifacts: 21 CSV files and 1 JSON summary.
- 11,813 CSV rows.
- No duplicated rows.
- No required-schema failures.
- No tested range failures.
- Missing cells occur in null/ablation statistics that are undefined for specific controls; plotting code must filter and report them rather than impute them.

## Scope-safe use

- Figures 1-3 may use only readback, transition and robustness terminology.
- Figure 4 may display topology, mechanism and overall functional-window scores.
- Figure 4 distinguishes CM-4D functional-geometry recurrence from CM-4C direction-identity tests; the two are not interchangeable `isomorphism` measures.
- CM-4D columns whose names contain `deltaU` are staged for provenance but cannot appear in Figure 4 or Results 3.3 because \(\Delta U\) has not yet been introduced. They are reserved for Figures 5-6.

## Files

- `manifest_figures_01_04.csv`: source paths, timestamps, hashes and dimensions.
- `staging_summary.json`: copy summary.
- `data_profile.csv`: row, column, duplicate and missing-value profile.
- `schema_audit.csv`: required-column audit.
- `range_audit.csv`: selected metric-range audit.
- `audit_summary.csv`: aggregate audit result.
- `data/`: staged source files.
