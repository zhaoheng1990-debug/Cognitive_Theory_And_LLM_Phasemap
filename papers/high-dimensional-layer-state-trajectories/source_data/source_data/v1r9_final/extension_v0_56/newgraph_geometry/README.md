# V1R9 paired 12B new-graph geometry confirmation

This directory contains the processed evidence used to add the paired 12B confirmation to V1R9 Fig. 2b,c and the corresponding main-text result.

## Frozen protocol

- 24 disjoint relation-graph groups x 10 fixed prompt conditions = 240 prompts.
- The same prompt manifest was evaluated in Qwen2.5-1.5B-Instruct and Gemma-3-12B-it-QAT-Q4_0.
- All native blocks were retained; no functional-window scan or selection was performed.
- The null uses 50 shared endpoint-preserving permutations of intermediate block order.

## Included files

- `newgraph_geometry_summary.csv`: reported model-level statistics.
- `*_prompt_geometry.csv`: one row per prompt-level geometry observation.
- `*_shared_endpoint_shuffle_null.csv`: the 50 endpoint-preserving null values for each checkpoint.
- `newgraph_prompt_manifest.csv`: the fixed 240-prompt manifest.
- `run_metadata.json`: checkpoint, extraction and protocol metadata.
- `independent_verification.json` and `verification_rerun_20260816.json`: validation records; the latter is a fresh raw-state recomputation.

## Retained raw intermediates

The raw all-block state arrays and 240 Gemma capture pairs remain unchanged in
the local experiment archive (`extension_v0_56/newgraph_geometry_formal_final`).

They are intentionally referenced rather than duplicated in this manuscript-stage directory. The independent rerun verified finite raw arrays, expected shapes, the fixed manifest, all prompt metrics, all 50-null records, and all 240 capture pairs.

## Claim boundary

This is a paired, checkpoint-specific larger-model confirmation of the endpoint-preserving order-null signature. It is not a within-family scale law, a quantization-invariance result, or a general claim about all Transformer architectures.
