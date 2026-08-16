# V1R9 Figure 2 scale-confirmation addendum

Figure 2b,c in the V1R9 Article adds a paired, larger-checkpoint confirmation
that is not present in V1R4. The test used 24 disjoint relation-graph groups
and 10 fixed conditions per group (240 prompts) in Qwen2.5-1.5B-Instruct and
Gemma-3-12B-it-QAT-Q4_0. All native blocks were retained and no functional
window was selected. The matched null has 50 shared endpoint-preserving
permutations of intermediate block order.

The processed prompt-level records, null samples, manifest, run metadata and
raw-state recomputation record are in `extension_v0_56/newgraph_geometry/`.
For Gemma-3-12B, native chord alignment was 0.0864, the null mean was 0.0240,
the gain was 0.0624 and the null-standardized effect was 3.39. All 50 null
values were lower, yielding the discrete empirical value P=1/51.

This is a checkpoint-specific confirmation of the order-null signature. It
does not identify a pure within-family scale law because the paired systems
differ in model family, quantization and capture runtime.
