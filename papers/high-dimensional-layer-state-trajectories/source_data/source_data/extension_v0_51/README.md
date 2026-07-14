# Source-data extension v0.51

This release adds source data for the independent task, random-initialization, held-out boundary-bridge and selective output-control analyses introduced in preprint v0.51.

## Dataset-to-location map

| Analysis | Public location | Unit of observation |
|---|---|---|
| Lexical-category validation | `lexical_category_validation/` | prompt-condition-checkpoint row, plus endpoint-preserving and label-shuffle nulls |
| Random-initialization geometry | `random_initialization_geometry/` | matched prompt and initialization seed |
| Displacement-boundary bridge | `displacement_boundary_bridge/` | held-out prompt-control pair |
| Qwen selective output control | `qwen_selective_output_control/` | held-out prompt-control pair and full-vocabulary top-1 outcome |
| Safety-matched reassignment | `qwen_selective_output_control/safety_matched_action_null/` | null reassignment and closure prompt |
| Figure inputs | `figure_inputs/` | plotted aggregate or row-level table |

The archive does not contain model weights or large hidden-state arrays. Those arrays are regenerable intermediate products; public scripts record the extraction and analysis procedure. All prompts are synthetic controlled-task prompts.

## Interpretation boundaries

The lexical task supports transfer of layer-order geometry more strongly than transfer of the task-constructed coordinate. The random-initialization control identifies an architecture-compatible scaffold in one Qwen architecture, not a transformer-wide law. The Qwen intervention crosses a declared candidate boundary for a subset of held-out closure prompts while preserving all 203 held-out non-closure top-1 outputs under the confidence gate. It does not establish general correctness, open-ended generation control or a deployment guarantee. The safety-matched null supports selective gating and a bounded actuator family but not fine-grained trajectory-specific dosing.
