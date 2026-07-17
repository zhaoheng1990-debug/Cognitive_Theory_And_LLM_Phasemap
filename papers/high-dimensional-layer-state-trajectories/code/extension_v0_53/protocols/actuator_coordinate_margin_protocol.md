# Control-chain mediation audit

## Question

Does the Qwen held-out boundary intervention support a bounded chain from the
applied local-transition actuator, through displacement of the train-fitted
internal coordinate, to clean-minus-conflict margin displacement and boundary
crossing?

This is a mechanistic-chain audit, not a formal natural-indirect-effect
analysis. The post-intervention coordinate may share unmeasured causes with the
output margin, and no sequential-ignorability claim is made.

## Frozen tests

1. **Held-out displacement.** In the 87 held-out closure prompts under the
   confidence-gated policy, graph-bootstrap 95% confidence-interval lower
   bounds for mean `DeltaU_shift` and mean `margin_shift` must both exceed zero.
2. **Incremental mediator test.** Five-fold GroupKFold by graph predicts held-out
   `margin_shift`. A fixed ridge model containing baseline margin, baseline
   DeltaU, condition, action summaries and gate diagnostics is compared with the
   same model plus post-intervention `DeltaU_shift`. The observed out-of-fold
   R-squared gain must be positive and exceed the 95th percentile of 999
   graph-block permutations that exchange the complete three-condition
   mediator profile between graphs.
3. **Executor dose-response.** In the train-only ten-action calibration grid,
   prompt-level Spearman associations must be positive for dose-to-DeltaU,
   dose-to-margin and DeltaU-to-margin. Their graph-bootstrap lower bounds must
   exceed zero and their within-prompt dose-permutation P values must be at most
   0.05. This test characterizes the executor; it is not held-out efficacy
   evidence.
4. **Boundary correspondence.** Candidate-margin crossings and strict
   full-vocabulary conflict-to-clean top-1 flips must agree for every held-out
   closure prompt.
5. **Assignment-specificity negative control.** The previously frozen
   safety-matched action reassignment remains decisive. If the real assignment
   does not exceed its null, no personalized policy-to-trajectory assignment
   claim is allowed even when tests 1-4 pass.

## Decision

Tests 1-4 jointly support a bounded actuator-coordinate-margin chain. Test 5 is
reported independently and bounds the actor claim. No result implies answer
correctness, open-ended generation control, deployment control, a universal
dynamical equation or a hidden confidence-gated control ontology.
