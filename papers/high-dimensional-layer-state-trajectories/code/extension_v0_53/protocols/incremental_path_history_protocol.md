# Incremental prediction from ordered trajectory prefixes

## Falsification target

```text
The ordered trajectory prefix contains predictive information about the final
candidate boundary beyond the current state and unordered prefix statistics.
```

The endpoint is the model's final clean-versus-conflict candidate margin sign.
This remains a diagnostic vocabulary-readback endpoint, not answer correctness.

## Horizons and held-out unit

Lexical-category and arithmetic-addition tasks are tested at 25%, 50% and 75%
of model depth for Qwen, Llama and Gemma. Five-fold GroupKFold holds out complete
`problem_id` blocks.

## Nested feature models

```text
prompt condition/mechanism only;
current candidate margin;
current high-dimensional hidden state (fold-fitted PCA + classifier);
unordered prefix summaries with the current margin retained;
ordered candidate-margin prefix.
```

All low-dimensional models use the same standardized class-balanced logistic
classifier. The hidden-state baseline uses fold-fitted standardized PCA with
32 components before the same classifier. No target-layer or held-out-fold
information enters feature fitting.

## Order null

For every prompt independently, earlier prefix layers are permuted while the
current-layer margin is kept fixed. Thus the null preserves the current state
and the complete multiset of earlier margins, removing only their order.

## Inference and gate

Balanced-accuracy differences are tested by swapping paired model predictions
at the held-out problem-group level. The order null uses 199 repeated
permutations. Benjamini-Hochberg correction is applied over all 18
task-model-horizon comparisons for each contrast family.

A condition supports incremental ordered-path prediction only when ordered
prefix performance is greater than all of:

```text
current margin;
current hidden state;
unordered prefix summary;
95th percentile of the order-null distribution;
```

and all four one-sided tests have `q <= 0.05`.

Broad support requires at least 12 of 18 conditions. Bounded support requires
at least 4 of 18 conditions spanning both tasks and at least two checkpoints.
Anything weaker rejects a general path-memory claim while preserving the
previous bounded early-readback forecast result.
