# Output-boundary actuator attribution audit v0.1

**Status:** PASS

At the declared operating point, output-boundary leverage is carried primarily by the confidence-gated local transition operator. The precursor component alone is insufficient, and neither the uniform-action control nor safety-matched reassignment supports fine-grained prompt-specific dose assignment.

This is a component attribution within one bounded actuator and one controlled candidate boundary. It is not answer-correctness improvement, open-ended generation control, safety, deployment control or a universal mechanism decomposition.

## Pooled counts

- gated_policy_flips: 39
- gated_policy_damage: 2
- operator_only_flips: 39
- operator_only_damage: 2
- precursor_only_flips: 1
- precursor_only_damage: 0
- uniform_max_flips: 47
- uniform_max_damage: 3
- ungated_policy_flips: 41
- ungated_policy_damage: 11

## Checks

- PASS: protocol_precedes_all_control_outputs
- PASS: all_reference_outputs_reproduced_exactly
- PASS: five_controls_per_model
- PASS: complete_prompt_rows
- PASS: frozen_group_counts
- PASS: all_intervened_states_finite
- PASS: norm_guard_below_1.10
- PASS: reference_counts_match_frozen_report
- PASS: operator_only_preserves_pooled_crossing_count
- PASS: precursor_only_is_not_sufficient
- PASS: uniform_max_is_not_inferior_in_crossing_count
- PASS: matched_reassignment_rejects_fine_assignment_specificity
