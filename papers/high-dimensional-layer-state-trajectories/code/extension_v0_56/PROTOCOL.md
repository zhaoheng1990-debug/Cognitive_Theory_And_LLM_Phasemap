# v0.56 pre-registered closure audits

## Window initialization

TheoryBaseline: manuscript v0.32 bounded process-object claims

MethodologyKernel: v1.1

This window inherits MethodologyKernel v1.1.
If any experimental conclusion conflicts with this kernel, the conflict must be explicitly stated and converted into a theory revision, downgrade, or caveat.

Inherited objects:
- Ordered, high-dimensional layer-state trajectories are the tested empirical process object.
- Vocabulary readbacks are diagnostics, not the process object.
- `DeltaU` is a task-constructed bounded observable, not an energy, universal order parameter, or general state equation.
- The local-transition actuator has demonstrated bounded relation-graph candidate-boundary control.

Preserved caveats:
- Functional-window locations are exploratory and their joint prospective recurrence failed.
- Ordered-prefix analyses have not established a general predictive dynamics law.
- Cross-task output control is not confirmed; the frozen mixed-arithmetic two-model gate failed.
- Gemma 3 12B QAT Q4_0 is a quantized, later-generation checkpoint, not a pure same-architecture scale ablation.

## Objective and work order

This work package evaluates, in order:

1. A predecision coordinate competition audit: `DeltaU_pre` and an ordered prefix against matched inexpensive output-logit readbacks and a direct hidden-state baseline.
2. A norm-, layer-, and gate-matched empirical-subspace random-direction actuator null.
3. A paired new-graph geometry confirmation using Qwen2.5-1.5B and Gemma-3-12B-it QAT Q4_0, conditional on successful layer-state extraction from the GGUF runtime.
4. A multiplicity-and-evidentiary-family register, a claim-family verification matrix, and a human numerical-review record.

No functional-window scan, intervention policy search, prompt outcome selection, or metric selection is permitted after this protocol is frozen.

## Audit A: predecision coordinate competition

Object-before-proxy declaration:

```text
Prompt-conditioned ordered layer-state trajectory
  -> predecision candidate-readback history and cutoff hidden state
  -> group-held-out mechanism-regime macro-F1
```

### Inputs and split

- The controlled 96-graph relation-graph task is regenerated from the frozen public v0.52 task constructor with its three predefined regimes: stable, competition, and closure.
- Each checkpoint uses its native tokenizer and a fixed five-fold `GroupKFold` over whole `graph_id` groups.
- The target is the prompt-defined mechanism regime. It is not final answer correctness, final candidate choice, or a label derived from `DeltaU`.
- All fitting, scaling, PCA, and classifier parameters are fit within each training fold only.

### Frozen layer cutoffs

The cutoff is the midpoint of the previously frozen decision-layer sequence, strictly before its final layer:

| Checkpoint | Decision layers | Predecision cutoff | Three-layer readback |
|---|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 20-25 | 22 | 20-22 |
| Llama-3.2-1B-Instruct | 12-15 | 13 | 12-13 (two available predecision layers) |
| Gemma-2-2B-it | 18-23 | 20 | 18-20 |

These are frozen analysis locations, not confirmed functional stages.

### Feature sets

At cutoff `k`, `R_l` is the clean-minus-conflict output-logit margin at layer `l`, and `dR_l` is that margin minus the matched clean-prompt margin for the same graph.

1. `raw_current`: `R_k`.
2. `raw_recent`: all available `R_l` from the listed three-layer readback.
3. `matched_current`: `dR_k`.
4. `matched_recent`: all available `dR_l` from the listed three-layer readback.
5. `DeltaU_pre`: the first principal component of the complete training-fold `dR_l` history from block 0 through `k`; its sign is oriented using training-fold stable versus closure means only.
6. `ordered_prefix`: the complete training-fold standardized `dR_l` history from block 0 through `k`.
7. `cutoff_hidden`: the native cutoff hidden state, with a training-fold standardized 32-component PCA before the classifier.

All low-dimensional feature sets use the same class-balanced multinomial logistic regression. `cutoff_hidden` uses the same classifier after the fold-fitted PCA. No final layer, post-cutoff state, final token, correctness label, held-out fold statistic, or policy output enters a feature.

### Primary contrasts and inference

The primary contrasts are `DeltaU_pre - matched_recent` and `ordered_prefix - matched_recent`, evaluated separately in the three checkpoints. `cutoff_hidden - matched_recent` is a contextual secondary contrast. Raw-margin contrasts are descriptive fairness checks.

For each contrast, the effect is the out-of-fold macro-F1 difference. One-sided paired group-sign-flip tests use 100,000 fixed-seed repetitions over graph-level macro-F1 contributions. Benjamini-Hochberg correction applies across the nine primary and secondary checkpoint-by-contrast tests in this coordinate-audit family.

A checkpoint supports a distinct predecision ordered-readback contribution only when both primary effects are positive and have `q <= 0.05`. A null or negative result narrows the claim; it does not trigger a new cutoff, classifier, or feature search.

## Audit B: empirical-subspace random-direction actuator null

Object-before-proxy declaration:

```text
Mechanism-conditioned local transition perturbation
  -> held-out gate-open, layerwise state displacement
  -> target candidate-boundary crossings and non-target top-1 changes
```

### Inputs and preservation rules

- The published relation-graph group split, checkpoint-specific operator layers, precursor layers, confidence gate, policy action table, and held-out prompts are reused exactly.
- The trained local operators and precursor directions are refit from the original training groups only, as in the released controller.
- Each random seed preserves the same open gates, action weights, intervention layers, per-component displacement norms, batch schedule, tokenizer, and full-vocabulary endpoint as the structured actuator.
- Random vectors are sampled once per layer from the training-group empirical update subspace, then projected orthogonally to the structured component direction and normalized. Isotropic full-dimensional noise is prohibited.

### Controls and decision

For every checkpoint, the structured gated actuator is compared with 50 frozen random-direction seeds. The primary endpoint is specificity:

```text
strict conflict-to-clean crossings among closure prompts
minus
full-vocabulary top-1 changes among non-closure prompts.
```

Secondary endpoints are target margin shift, target crossing count, non-target change count, and recorded component/final displacement-norm ratios. The structured actuator supports direction specificity only if its specificity exceeds the random-null 95th percentile and no non-finite or norm-integrity failure occurs. This test does not establish answer correctness, open-ended generation control, or exact dose specificity.

## Audit C: paired larger-checkpoint geometry confirmation

Object-before-proxy declaration:

```text
Native ordered layer-state sequence
  -> normalized all-layer state path
  -> endpoint-preserving order-null chord alignment
```

### Qualification gate

Gemma-3-12B-it QAT Q4_0 is eligible only after a separate extraction probe exports last-token states from every transformer block, reproduces a deterministic repeat, records model hash and quantization, and verifies a usable final output readback. A final-state-only GGUF interface is a qualification failure, not a geometry result.

### Confirmatory design

- Generate 24 new relation-graph groups with disjoint entity names, relation phrases, and candidate labels relative to the original controlled set; use the fixed conditions `clean`, `permuted`, `redundant`, `irrelevant`, `weak_distractor`, `competition_balanced`, `direct_conflict`, `closure_update`, `closure_override`, and `exception_override` per group (240 prompts, not 288).
- Use the identical new prompt manifest for Qwen2.5-1.5B and Gemma-3-12B-it QAT Q4_0.
- Analyze all native layers with no window selection.
- For each checkpoint, compare real chord alignment with 50 endpoint-preserving permutations shared across prompts. The checkpoint-level mean is the inferential unit.
- The primary conclusion is a checkpoint-specific order-null result. It is not a pure scaling law, a quantization-invariant measurement, or a universal Transformer result.

## Documentation and publication rules

- A multiplicity register separates confirmatory, exploratory, and boundary/falsification families; no global study-wide adjusted P value will be claimed.
- A reproduction matrix maps every main claim family to source tables, a verifier, expected outputs, and version/hash controls.
- The statement `H.Z. performed a final item-by-item review ...` may be added only after H.Z. personally completes and signs the generated review checklist.
- New figures or tables will be merged into existing display items where needed; no additional Extended Data display item is assumed available.
- All raw prompt-level outputs, configurations, checksums, and negative results are retained before manuscript language is changed.

## Closure record

Each audit will report `ACCEPT`, `REVISED`, `PENDING`, or `FAIL` with its evidence coordinate, claim boundary, source-data paths, verifier, and required manuscript/repository changes. No baseline object update is forced by a positive numerical result.
