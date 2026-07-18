# Transition-predictive bilinear closure audit v0.1

**Implementation closed:** PASS

- State-only macro R2: 0.822154
- Best target-excluded two-stage macro R2: 0.861589
- Increment: 0.039436 (group-bootstrap 95% CI 0.036806 to 0.042617)
- Realized-operator gain retained: 25.0947%
- Margin above the pre-existing PASS-Lite retention threshold: 0.0947%

## Interpretation boundary

The corrected two-stage branch supports a finite held-out incremental prediction claim. It does not establish a universal dynamical law, confirmatory generalization, or predictive equivalence to the realized O_l^cont coordinate.

## Checks

- PASS: input_identity - sha256=5C5E3C3686145CA889E0F25B7654677E680F9B765B77FAB5A6A6265253FF88FB; rows=14592
- PASS: all_two_stage_branches_executed - rows=8; additive=4; bilinear=4
- PASS: historical_failure_reproduced - historical_error_rows=4
- PASS: unaffected_branches_invariant - max_abs_summary_delta=0.000e+00
- PASS: complete_grouped_folds - fold_rows=40; models=8; group_col=sample_id
- PASS: group_leakage_excluded - train_test_group_overlap_by_fold=[0, 0, 0, 0, 0]; unique_groups=768
- PASS: target_layer_columns_excluded_from_G - forbidden_features=[]
- PASS: bilinear_oof_predictions_complete - prediction_columns=16; rows=14592
- PASS: bilinear_gain_consistent_across_G_models - gains={"G1_state_only": 0.027903512120246887, "G2_state_plus_history": 0.03179293870925903, "G3_state_plus_context": 0.02951228618621826, "G4_state_history_context": 0.028905019164085388}
- PASS: positive_group_bootstrap_increment - delta_R2=0.039436; 95%_CI=[0.036806, 0.042617]; Pr(delta>0)=1.0000
- PASS: pre_existing_pass_lite_rule_met - retention=0.250947; margin_above_0.25=0.000947; delta_R2=0.039436
