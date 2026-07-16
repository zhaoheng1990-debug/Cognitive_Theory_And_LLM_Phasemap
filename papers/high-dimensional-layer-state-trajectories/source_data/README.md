# Source Data for high-dimensional layer-state trajectory dynamics

Version: v0.1-predeposit

This archive contains processed, panel-level Source Data supporting the manuscript **High-dimensional layer-state trajectories reveal the internal dynamics of transformer inference**.

## Contents

- `source_data/`: CSV and JSON files underlying main and supplementary figures, together with panel-level manifests and QA reports.
- `metadata/source_data_manifest_v0_1.csv`: global figure-to-file map.
- `metadata/metric_definition_table_v0_5.csv`: definitions and interpretation boundaries for reported metrics.
- `metadata/prompt_table_v0_1.csv` and `prompt_condition_summary_v0_1.csv`: controlled prompt metadata.
- `metadata/model_checkpoints_v0_2.csv`: checkpoint identifiers and extraction metadata.
- `metadata/random_seeds_and_splits_v0_2.csv`: random seeds and grouped split definitions.
- `metadata/null_control_table_v0_2.csv`: null-control construction and purpose.
- `metadata/classifier_spec_table_v0_2.csv`: classifier labels and implementation summary.
- `metadata/hidden_state_extraction_table_v0_2.csv`: hidden-state extraction scope.
- `metadata/supplementary_methods_v0_2_reproducibility_merged.md`: extended reproducibility notes.

## Scope

The archive provides the processed values needed to inspect the manuscript figures and reported statistical summaries. It does not redistribute Qwen, Llama or Gemma model weights. Raw hidden-state arrays are not included; they are derivative intermediate outputs of third-party checkpoints. Checkpoint revisions, prompts and extraction metadata are supplied to define the regeneration route.

## File formats

Tabular data use CSV. Structured summaries use JSON. Documentation uses Markdown or plain text. Missing-value and field semantics are documented in the panel manifests and metric table.

## Integrity

`MANIFEST_SHA256.csv` lists the SHA-256 digest and byte size of every file in this archive except the manifest itself.

## Citation

The final Zenodo DOI and creator list must be inserted after a DOI is reserved. Do not cite this predeposit directory as a published dataset.

## Licence

No reuse licence has yet been granted. See `LICENSE_DECISION_REQUIRED.md`. The current recommendation is CC BY 4.0 for processed Source Data and documentation, subject to author confirmation.

## v0.51 extension
Row-level evidence and plotted inputs for the independent task, random-initialization, boundary-bridge and selective output-control analyses are under `source_data/extension_v0_51/`. See its `README.md` and checksum manifest.

## v0.52 extension
The task-generalization, predictive-forecasting, cross-architecture initialization and cross-checkpoint output-control evidence is under `source_data/extension_v0_52/`. See its README and checksum manifest.
