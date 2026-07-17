# Layer-order geometry versus behaviour

## Falsification target

```text
Greater trajectory straightness is sufficient evidence of a more faithful
functional computation.
```

The existing trained-Qwen residual execution cache is used. Native order,
reverse order and one fixed layer permutation share the same prompts, weights,
residual gain (`alpha=1`) and output head.

## Reconstruction audit

Saved final hidden states are mapped through the checkpoint output head. On an
eight-prompt audit subset, reconstructed native logits must reproduce direct
forward full-vocabulary top-1 tokens exactly and candidate margins with maximum
absolute error below `0.05`. Failure invalidates the cached-state behaviour
analysis and triggers a direct rerun.

## Geometry endpoint

The preregistered geometry contrast is the paired reverse-minus-native change
in unit-state future-chord alignment. It passes when the paired bootstrap 95%
CI is above zero and the sign-flip `P <= 0.05`.

## Functional endpoints

Each reordered condition is compared with native execution on the same prompt:

```text
full-vocabulary top-1 agreement;
clean-versus-conflict candidate-choice agreement;
Jensen-Shannon divergence of the full next-token distribution;
rank of the native top-1 token under the reordered distribution.
```

Strong functional disruption requires at least two of:

```text
full-vocabulary top-1 agreement <= 0.50;
candidate-choice agreement <= 0.75;
bootstrap lower 95% bound of mean JSD > 0.05.
```

Task-defined accuracy is secondary and restricted to unambiguous conditions:
`stable` and `stable_shift` target the clean candidate; `closure` targets the
conflict candidate. Competition and source-ambiguity conditions are not forced
into correctness labels.

## Decision

Straightness-function dissociation passes only when reverse order passes the
positive geometry contrast and the strong functional-disruption gate. The
fixed permutation is a replication contrast. Passing does not prove that every
reordering reduces task accuracy; it establishes that greater straightness is
not sufficient for native functional fidelity.
