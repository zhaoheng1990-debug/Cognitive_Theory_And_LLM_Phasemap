# Experiment closure source data v0.1

This directory contains manuscript-facing summary tables generated from frozen experiment outputs.

- `task_geometry_generalization.csv`: standard four-choice ARC-Challenge plus controlled lexical and addition task geometry.
- `task_coordinate_transfer.csv`: held-out task-specific one-dimensional coordinate tests.
- `heldout_predictive_forecasting.csv`: graph-held-out final candidate-margin forecasts and independent within-prompt order nulls.
- `crossmodel_initialization_geometry.csv`: trained versus three random initializations per architecture.
- `crossmodel_output_boundary.csv`: full-vocabulary output-boundary results under the frozen confidence gate.
- `crossmodel_assignment_matched_nulls.csv`: 50 condition-, gate- and action-multiset-matched reassignments per checkpoint.
- `experiment_closure_verdict.json`: machine-readable claim and boundary audit.

The evidence supports geometry transfer, a bounded predictive foothold, cross-architecture initialization controls and cross-model executor reachability. It does not support a universal numerical coordinate, a closed dynamical law, output correctness, deployment control or trajectory-specific action assignment.
