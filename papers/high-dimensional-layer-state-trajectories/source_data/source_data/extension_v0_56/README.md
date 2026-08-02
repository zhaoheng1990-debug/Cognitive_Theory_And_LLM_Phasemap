# v0.56 Source Data extension

This directory retains the processed, prompt-level evidence for the three v0.56 closure audits. `manifest.csv` lists every released file with its byte size and SHA-256 digest.

## Contents

- `coordinate_audit/`: five-fold graph-held-out predictions, fold scores, planned contrasts, decision gates, extraction manifests and an independent verification receipt for the predecision `DeltaU` competition.
- `direction_null/`: the structured actuator endpoint, all 50 empirical-subspace random-direction outcomes per checkpoint, cross-model verdicts and an independent verification receipt.
- `newgraph_geometry/`: the frozen 24-group, 240-prompt manifest; raw prompt text; checkpoint-level prompt metrics; 50 shared endpoint-preserving shuffle summaries; run metadata and an independent verification receipt.

The extension deliberately includes the negative/mixed outcomes: `DeltaU_pre` does not exceed a matched recent logit-margin baseline, and empirical-subspace direction specificity does not survive three-checkpoint family correction. These are boundaries of the manuscript claims, not failed uploads.

## Excluded regenerable intermediates

Model weights, vendor runtimes and large native hidden-state arrays are excluded from the Git mirror. The new-graph geometry raw state arrays were used to compute the released prompt-level metrics and are identified by model hash, capture source revision, extraction metadata and prompt hashes in `newgraph_geometry/run_metadata.json`. The corresponding native capture source is in `code/extension_v0_56/native_capture/`.

## Verification

Use the three read-only scripts in `code/extension_v0_56/` as described in that directory's README. The coordinate and direction scripts recompute their released summaries from retained rows. The geometry script verifies released prompt metrics, null summaries and integrity metadata in the public mirror; full raw-state recomputation requires the separately archived intermediate arrays. The claim-family matrix and multiplicity register define what a pass does and does not support.
