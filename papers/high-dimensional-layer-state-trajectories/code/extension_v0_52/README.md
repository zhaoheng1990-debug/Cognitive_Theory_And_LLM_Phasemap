# v0.52 reproducibility extension

This directory contains portable entry points for the three experimental closures added in preprint v0.52: task-family and four-choice generalization, held-out partial-trajectory forecasting, and cross-checkpoint initialization and output-boundary controls.

## Entry points

```bash
python run_arc_four_choice_geometry.py --output-dir output/arc
python analyze_partial_trajectory_forecasting.py --trajectory-root inputs/relation_trajectory_tables
python run_crossmodel_random_initialization_geometry.py --subset-manifest inputs/controlled_subset_manifest.csv
python run_crossmodel_output_boundary_control.py --include-existing-qwen
python run_crossmodel_safety_matched_nulls.py
Rscript draw_extended_data_three_closures.R
```

The processed row-level outputs needed to audit the reported values are in `source_data/source_data/extension_v0_52/`. Full model reruns require the cited checkpoints, regenerated hidden-state intermediates and sufficient local GPU memory. Model weights and large derivative hidden-state arrays are not redistributed.

## Claim boundaries

The repeated result across lexical, addition and ARC tasks is layer-order geometry; task-specific scalar coordinates transfer less consistently. Forecasting is a held-out prediction of one declared final candidate margin, not an autonomous state equation. Random-initialization controls identify an architecture-compatible order scaffold, not learned reasoning. Confidence-gated interventions establish bounded reachability of a declared emitted-token candidate boundary in the three tested checkpoints; they do not establish answer correctness, open-ended generation control, deployment safety or fine-grained trajectory-to-dose specificity.
