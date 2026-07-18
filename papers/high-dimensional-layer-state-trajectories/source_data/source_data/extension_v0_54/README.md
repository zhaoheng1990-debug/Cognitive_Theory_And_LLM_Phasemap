# v0.54 Source Data extension

This directory preserves the processed evidence for the final methodology-closure pass.

- `predictive_transition_boundary/`: included transition table, out-of-fold predictions, grouped folds and audit.
- `functional_window_confirmation/`: frozen protocol, untouched prompt construction, per-checkpoint scans/nulls and the negative joint confirmation result.
- `actuator_attribution/`: cross-checkpoint component summaries and prompt-level output-boundary comparisons with internal development labels removed.
- `label_crosswalk/`: explicit separation of coordinate-audit and intervention label systems.
- `figure_source_data/`: sanitized source tables for all submission figures and Extended Data figures.

`manifest.csv` records size and SHA-256 for every file in this extension. Large hidden-state arrays and model weights are not redistributed.
