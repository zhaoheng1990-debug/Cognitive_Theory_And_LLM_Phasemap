# Held-out learned-metric protocol v0.1

## Question

Raw unit-state paths are strongly ordered even at random initialization. Does
training add task organization that is hidden from that architecture-compatible
metric but visible in a task-conditioned coordinate?

## Frozen tests

The same 53 lexical base-cloze prompts and seven checkpoints from one
`Pythia-160m` training run are reused. No new prompt is selected.

### Model-native candidate coordinate

For prompt `p` and layer `l`, define the normalized output-head contrast

`c[p,l] = cos(h[p,l], w_clean[p]) - cos(h[p,l], w_conflict[p])`.

This is a diagnostic coordinate supplied by the checkpoint's own output head.
It is not treated as the hidden process itself. Native layer order is compared
with 999 endpoint-preserving interior-layer shuffles using the mean prompt-wise
Spearman correlation between layer depth and `c`.

### Held-out category metric

- Split clean answer words, not prompt rows, into three deterministic folds.
- In each fold, fit a 16-dimensional PCA followed by fixed-regularization
  multinomial logistic regression on final-layer states from the other answer
  words.
- Apply the fitted map to every layer of held-out prompts. The task evidence is
  the true-category logit minus the mean alternative-category logit.
- Report final-layer macro-F1 and native-order evidence accumulation.
- Run both a within-checkpoint fit and a final-checkpoint coordinate transferred
  to all earlier checkpoints.
- For step 0 and the final checkpoint, compare against 199 category-label
  permutations. Layer-order evidence uses the same 999 endpoint-preserving
  shuffles as the model-native coordinate.

## Falsification-first gates

1. The final checkpoint's candidate-coordinate order statistic must exceed the
   99th percentile of its endpoint-preserving shuffle null.
2. Its real-minus-null order gain must exceed step 0 by at least 0.10.
3. Candidate-coordinate order gain must have Spearman correlation at least 0.60
   with pair accuracy across the seven checkpoints.
4. The final checkpoint's within-checkpoint held-out category macro-F1 must
   exceed its label-permutation 95th percentile and step 0 by at least 0.10.
5. In the final-coordinate transfer test, final-checkpoint macro-F1 must exceed
   step 0 by at least 0.10 and exceed its label-permutation 95th percentile.

The candidate-coordinate and category-metric gates are reported separately. A
failure of the category gate is retained even if candidate evidence progresses.

## Claim boundary

The category word is explicit in this bounded cloze prompt, so the category
metric tests held-out lexical organization, not multi-step reasoning. The
candidate coordinate uses vocabulary weights and remains a diagnostic readback.
Neither test establishes a unique physical metric, a universal semantic
geometry, or an exact geodesic. The purpose is to distinguish
architecture-compatible continuity from training-associated task coordinates.
