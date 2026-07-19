# Frozen mixed-arithmetic actuator-transfer protocol

## Question and empirical object

This audit asks whether the already specified local-transition intervention procedure transfers from relation graphs to an independently constructed mixed-arithmetic candidate task after training-only refitting. The empirical process is the ordered last-token layer-state trajectory. The endpoints are the arithmetic-consistent-versus-distractor candidate margin and the full-vocabulary top-1 token before and after intervention. This is not zero-shot transfer of numerical operators or of the relation-graph candidate coordinate.

## Frozen design

- 64 unique addition, subtraction and multiplication problems, each with six prompt conditions.
- Group split fixed at 44 training and 20 held-out problems with seed 20260714; all conditions from one problem remain together.
- Qwen, Llama and Gemma layer windows, operator parameterization, ridge penalty, ten-action library, classifier hyperparameters, confidence threshold 0.50, normalized-entropy threshold 0.90 and 5% non-target ceiling inherited without held-out retuning.
- Task-specific coordinates, transition maps, precursor directions and action labels fitted only on training problems.
- Strict eligibility: a held-out correction prompt whose unperturbed top-1 is the distractor candidate.
- Strict crossing: the intervened top-1 becomes the arithmetic-consistent candidate.
- Non-target damage: any top-1 change among stable and competition prompts.
- Fifty address-destruction nulls per checkpoint preserve the open-gate count and exact multiset of layerwise action doses while reassigning complete action bundles across prompts.

Confirmation required at least two powered checkpoints with positive median eligible margin shift and one or more strict crossings, at least two checkpoints below the collateral ceiling, pooled specificity above the 95th percentile of the pooled address null and a non-zero pooled crossing count. A checkpoint with fewer than five eligible prompts was predeclared underpowered.

## Interface and reproduction record

The primary assay uses a raw prompt ending immediately before a requested one-word continuation. Every prompt-plus-candidate tokenization had to contain the complete prompt token sequence as an exact prefix, and the first appended token had to match the audited single leading-space token. An initial non-contextual-token run was invalidated and excluded. A chat-template pilot was excluded because Llama emitted free text before either candidate, leaving the declared candidate-token endpoint outside that interface. These pilots diagnosed endpoint validity and did not select actions.

Full-data extraction versus held-out replay was allowed a maximum FP16 candidate-margin discrepancy of 0.125, with any sign disagreement confined to that near-zero band. An independent repeat with identical held-out batching had to reproduce every top-1 token and pair sign exactly. All checkpoints passed. No scientific endpoint or decision threshold changed.

## Frozen result and boundary

Pooled strict crossings were 2/45 and non-target changes were 1/240. Pooled specificity was 0.04028, above the address-null 95th percentile of -0.01111 (plus-one one-sided empirical P = 1/51). Qwen was the only powered checkpoint with positive median displacement and a strict crossing. Llama crossed once but had a negative median displacement; Gemma had only four eligible prompts and no primary crossing. The two-checkpoint directional gate failed. The supported conclusion is partial protocol-level transfer, not confirmed cross-task output control, answer-correctness control, open-ended generation control or deployment control.
