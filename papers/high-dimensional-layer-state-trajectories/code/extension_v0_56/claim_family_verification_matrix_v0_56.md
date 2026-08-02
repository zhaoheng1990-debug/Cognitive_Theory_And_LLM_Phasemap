# Claim-family verification matrix

This matrix is the public verification index. It groups claims by evidence family rather than by individual plot panel.

| Evidence family | Supported ceiling | Source Data | Verification entry point | Expected decision |
|---|---|---|---|---|
| Coordinate audit | Direct hidden coordinates and complete predecision history can retain more mechanism-label information than a recent matched readback; `DeltaU_pre` is not superior. | `source_data/extension_v0_56/coordinate_audit/` | `verify_predecision_coordinate_audit.py` | All integrity checks pass; the distinct-`DeltaU` gate is false in all checkpoints. |
| Native order geometry | In a frozen 24-new-graph set, Qwen-1.5B and Gemma-3-12B-it-QAT-Q4_0 exceed endpoint-preserving order nulls in chord alignment. | `source_data/extension_v0_56/newgraph_geometry/` | `verify_newgraph_geometry_confirmation.py` | All recomputed metrics and both primary geometry outcomes pass. |
| Actuator direction null | A norm/layer/gate/action matched random-direction null does not support checkpoint-general direction specificity. | `source_data/extension_v0_56/direction_null/` | `verify_empirical_direction_null.py` | Published structured endpoint is reproduced; no model passes the family-adjusted direction-specificity criterion. |
| Relation-graph output control | The gated local-transition actuator produces bounded candidate-boundary crossings in the declared relation-graph assay. | `source_data/remaining_figures/` | Original reproducibility routes and retained action tables | Do not infer direction specificity, correctness, open-ended generation or deployment control. |
| Cross-task actuator boundary | The mixed-arithmetic test gives partial protocol transfer, not confirmed cross-task output control. | `source_data/extension_v0_55/` | `../extension_v0_55/verify_extension_v0_55.py` | The frozen two-checkpoint positive-direction gate fails. |
| Transition forecast | Relation-graph prefix forecasts are bounded; cross-task prefix gains do not establish a general prediction law. | `source_data/extension_v0_54/predictive_transition_boundary/` | `../extension_v0_54/verify_extension_v0_54.py` | Verify released target-excluded summaries; use boundary wording in the manuscript. |

All scripts are read-only verifiers. The relevant protocol, source-table hashes and version receipts must accompany any reconstructed result. Model weights, raw intermediate state arrays and vendor runtimes are excluded from this Git repository but are identified by hash and regeneration instructions in the Source Data metadata.
