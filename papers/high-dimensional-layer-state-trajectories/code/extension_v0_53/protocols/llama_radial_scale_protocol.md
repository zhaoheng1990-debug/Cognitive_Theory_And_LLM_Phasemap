# Llama radial-scale follow-up

The endpoint-consistency pilot falsified the hypothesis that the Llama raw
effect is caused only by the mixed final-RMSNorm endpoint: consistent decoder
block outputs retained the joint direction gate in several tasks. This
sequential follow-up tests radial scale drift.

For each consistently extracted trajectory `h_l = r_l u_l`, two surrogates are
constructed:

- **direction only:** `u_l`, removing every layerwise norm;
- **norm only:** `r_l u_0`, retaining the prompt's scalar norm schedule while
  fixing its direction to the first block state.

Both use the same 199 layer-conditioned recipient-norm donor null and joint
future/leave-one-out gate as the endpoint audit. Radial-scale sufficiency is
supported when consistent raw outputs pass at least three of four original and
two of three independent conditions, direction-only and common-RMSNorm outputs
pass at most one in each scope, and norm-only surrogates pass at least three of
four original and two of three independent conditions.

Passing means that scalar norm growth is sufficient for this statistic and
that raw alignment cannot be interpreted as endpoint-directed semantic flow.
It does not imply that radial scale is functionally irrelevant.
