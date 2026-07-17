# Llama raw-residual endpoint consistency audit

## Falsification target

The earlier secondary analysis found future-chord and leave-one-out donor-null
effects in four of four Llama tasks only in unnormalized states. The extraction
used `output.hidden_states[layer + 1]`. In Hugging Face Llama, intermediate
entries are decoder-block residual outputs, whereas the final entry is after
the model's final RMSNorm. The apparent trajectory can therefore join two
coordinate systems at its endpoint.

## Extraction audit

For the same prompts, decoder-layer forward hooks capture all 16 block outputs
before final RMSNorm. The run also stores:

- the standard mixed `hidden_states[1:]` sequence;
- the consistent pre-final-norm block-output sequence;
- every block output transformed by the same final RMSNorm;
- the matching rows from the original cache.

The four original 96-prompt conditions are relation graph, lexical category,
addition and ARC-Challenge. Three distribution checks use non-overlapping
lexical and addition rows plus the independent mixed-operation arithmetic set.

## Frozen tests

For each sequence, raw steps are evaluated with future-only and
leave-one-step-out alignment against 199 layer-conditioned donor walks. A
condition passes only when both real means exceed their null 95th percentiles
and both empirical one-sided P values are at most 0.05.

The coordinate-mismatch explanation passes when:

1. re-extracted mixed states reproduce their cached counterparts with mean
   per-prompt flattened cosine at least 0.999;
2. the cached final-to-penultimate norm ratio exceeds 3 in each original task;
3. the mixed sequence passes at least three of four original tasks;
4. each of consistent raw block outputs, common-RMSNorm outputs and cached
   sequences with the mixed endpoint removed passes at most one of four
   original tasks;
5. no corrected representation passes more than one of the three independent
   distribution checks.

This audit concerns coordinate consistency only. Passing or failing does not
establish a semantic manifold, geodesic flow or architecture-wide law.
