# Gemma-3-12B Scale Confirmation Closure V1R9

## Window initialization

TheoryBaseline: V1R8 separability-principle manuscript

MethodologyKernel: v1.1

This window inherits MethodologyKernel v1.1. If any experimental conclusion conflicts with this kernel, the conflict must be explicitly stated and converted into a theory revision, downgrade, or caveat.

## Objective

Test whether the native-order, endpoint-preserving-null geometry contrast recurs in a larger local checkpoint under a frozen protocol. This is not a function-window scan, a controller-transfer study, a pure within-family scaling analysis or a general Transformer claim.

## Object before proxy

Native ordered layer-state sequence -> normalized all-layer state path -> chord alignment relative to 50 endpoint-preserving intermediate-layer permutations.

## Evidence

- Checkpoint: Gemma-3-12B-it QAT Q4_0 GGUF; 48 transformer blocks, 3,840 hidden units.
- Prompt set: 24 disjoint new graph groups x 10 frozen conditions = 240 raw prompt strings.
- Null: 50 shared endpoint-preserving permutations; first and final block states fixed, intermediate blocks permuted.
- Primary result: real alignment 0.08640; null mean 0.02403; gain 0.06237; null-standardized effect 3.392; all 50 null values lower; empirical P = 1/51.
- Paired prompt-level Qwen2.5-1.5B control: gain 0.03744; null-standardized effect 5.632; empirical P = 1/51.
- Independent verifier rerun on 2026-08-16: PASS in raw-state recomputation mode. It rechecked prompt-manifest identity, raw tensor shapes and finiteness, 50 null draws, per-prompt metrics, all summary statistics and 240 Gemma capture pairs.

## Status and claim ceiling

**Status: ACCEPT_WITH_SCOPE.**

Evidence coordinate: Internal Project Evidence with raw-state recomputation and retained capture artifacts.

Accepted statement: the tested native-order geometry contrast recurs in this later-generation 12B Gemma checkpoint on a disjoint prompt set.

Not established: a pure scale law, a same-family size effect, quantization invariance, architectural homology, learned reasoning in the geometry, execution-function replication at 12B, or intervention transfer at 12B.

## Writeback

V1R9 makes the confirmation visible in Fig. 2b,c and in the architecture-scaffold Results paragraph. V1R8 remains unchanged. Before public or journal release, archive the raw prompt metrics, null table, run metadata, capture-source revision and independent-verification report alongside the V1R9 source-data manifest.
