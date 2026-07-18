# Output-boundary actuator attribution protocol v0.1

Frozen before component-ablation execution on 18 July 2026.

## Question

The existing held-out experiment shows that a confidence gate combined with a bounded local transition action family can cross the declared conflict-to-clean output boundary. Safety-matched action reassignment does not support prompt-specific dose assignment. This audit separates the remaining contributions of the operator component, the precursor component and a uniform within-gate action.

## Frozen data and fits

- Checkpoints: Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct and Gemma-2-2B-it.
- Split: the existing seed-42 split of 67 training and 29 held-out relation graphs.
- Held-out set: 290 prompts per checkpoint, including 87 closure and 203 non-closure prompts.
- Windows, training-fitted candidate coordinate, local transition operators, precursor directions and gate-open set are inherited without refitting to held-out outcomes.
- Gate thresholds remain mean closure support at least 0.50 and normalized entropy at most 0.90.
- No threshold, layer, dose or control is selected from the component-ablation outcomes.

## Controls

1. `policy_boundary_guarded`: the existing gated policy; positive reference.
2. `policy_gate_operator_only`: retain each gated policy alpha and set beta to zero.
3. `policy_gate_precursor_only`: retain each gated policy beta and set alpha to zero.
4. `policy_gate_uniform_max`: retain the same gate-open set, set every open action to alpha=1.0 and beta=1.2 at every intervention layer, and leave gate-closed actions at zero.
5. `none`: no intervention; negative reference.

The component ablations preserve the original prompt and layer assignment of the retained dose. The uniform control is the maximum combination already present in the declared action library; it is not a newly optimized dose.

## Endpoints

- Primary: strict full-vocabulary top-1 continuation change from the declared conflict token at baseline to the declared clean token under control, among the 87 closure prompts.
- Damage guard: any full-vocabulary top-1 change among the 203 non-closure prompts.
- Internal bridge: paired clean-minus-conflict margin shift and candidate crossing.
- Numerical guard: finite intervened states and intervention-layer state-norm ratios.

## Interpretation rules

- A component is necessary at the tested operating point only if removing it materially reduces strict crossings without a compensating reduction that fully explains the result through gate coverage or dose.
- Similar or greater performance from `policy_gate_uniform_max` supports gate-plus-family sufficiency and further rejects fine-grained dose assignment.
- Failure of either ablation does not establish that the retained component is uniquely mechanistic; the two components can interact nonlinearly.
- All conclusions remain bounded to the declared candidate boundary. They do not imply answer correctness, safety, open-ended generation control or deployment readiness.
