# Llama radial-exponent dose audit

The binary radial follow-up showed direction-only 0/7 and norm-only 7/7, but
its deliberately strict scope gate did not close because consistent raw states
passed only two of four original tasks and all three independent checks. This
third sequential audit tests whether radial scale continuously controls the
statistic rather than merely producing an extreme surrogate.

For consistent block states `h_l = r_l u_l`, construct
`h_l(gamma) = (r_l / r_0)^gamma u_l` at gamma values 0, 0.25, 0.5, 0.75, 1 and
1.25. Promptwise constant factors do not affect cosine step metrics, so gamma 1
is equivalent to raw states and gamma 0 is direction only.

At every gamma, future-only and leave-one-out alignment use 199
layer-conditioned recipient-norm donor walks. The radial-dose explanation
passes when:

1. at least six of seven tasks have Spearman rho at least 0.8 between gamma and
   the real-minus-donor-mean effect for both metrics;
2. gamma 0 passes the joint donor gate in at most one task;
3. gamma 1.25 passes the joint donor gate in at least six tasks.

This identifies sensitivity of the chosen statistic to radial scaling. It does
not show that model function is radial, that norm is causally irrelevant or
that any semantic/geodesic law exists.
