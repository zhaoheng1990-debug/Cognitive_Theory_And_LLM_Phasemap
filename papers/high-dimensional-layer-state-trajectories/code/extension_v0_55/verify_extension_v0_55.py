from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "source_data" / "extension_v0_55" / "mixed_arithmetic_actuator_transfer"
if not DATA.exists():
    DATA = HERE.parents[1] / "source_data" / "source_data" / "extension_v0_55" / "mixed_arithmetic_actuator_transfer"


def metrics(frame):
    eligible = frame[frame.mechanism.eq("closure") & frame.baseline_top1_class.eq("distractor")].copy()
    non_target = frame[~frame.mechanism.eq("closure")].copy()
    crossing = eligible.intervened_top1_class.eq("consistent")
    changed = ~non_target.intervened_top1_token_id.eq(non_target.baseline_top1_token_id)
    margin_shift = (
        eligible.intervened_consistent_minus_distractor_margin
        - eligible.baseline_consistent_minus_distractor_margin
    )
    coordinate_shift = eligible.intervened_task_coordinate - eligible.baseline_task_coordinate
    eligible_n = len(eligible)
    non_target_n = len(non_target)
    crossing_n = int(crossing.sum())
    changed_n = int(changed.sum())
    crossing_rate = crossing_n / eligible_n
    collateral_rate = changed_n / non_target_n
    return {
        "eligible_n": eligible_n,
        "strict_crossings": crossing_n,
        "target_crossing_rate": crossing_rate,
        "eligible_median_margin_shift": float(margin_shift.median()),
        "eligible_mean_margin_shift": float(margin_shift.mean()),
        "eligible_median_task_coordinate_shift": float(coordinate_shift.median()),
        "non_target_n": non_target_n,
        "non_target_top1_changes": changed_n,
        "collateral_rate": collateral_rate,
        "specificity": crossing_rate - collateral_rate,
    }


def compare(recomputed, reported, label):
    integer = ("eligible_n", "strict_crossings", "non_target_n", "non_target_top1_changes")
    floating = (
        "target_crossing_rate", "eligible_median_margin_shift", "eligible_mean_margin_shift",
        "eligible_median_task_coordinate_shift", "collateral_rate", "specificity",
    )
    for key in integer:
        assert int(recomputed[key]) == int(reported[key]), f"{label} {key}"
    for key in floating:
        assert np.isclose(float(recomputed[key]), float(reported[key]), atol=1e-10, rtol=1e-10), f"{label} {key}"


primary_pairs = pd.read_csv(DATA / "primary_prompt_pairs.csv")
primary_actions = pd.read_csv(DATA / "primary_actions.csv")
null_pairs = pd.read_csv(DATA / "address_null_prompt_pairs.csv")
null_actions = pd.read_csv(DATA / "address_null_actions.csv")
reported_primary = pd.read_csv(DATA / "primary_summary.csv")
reported_null = pd.read_csv(DATA / "address_null_summary.csv")
reproduction = pd.read_csv(DATA / "reproduction_checks.csv")
decision = json.loads((DATA / "decision.json").read_text())

primary_rows = []
null_rows = []
for model in ("qwen", "llama", "gemma"):
    observed = metrics(primary_pairs[primary_pairs.model.eq(model)])
    compare(observed, reported_primary[reported_primary.model.eq(model)].iloc[0], f"primary {model}")
    primary_rows.append({"model": model, **observed})

    signature = primary_actions[primary_actions.model.eq(model)][["alpha", "beta"]].sort_values(
        ["alpha", "beta"], kind="stable"
    ).reset_index(drop=True)
    for null_id in range(50):
        frame = null_pairs[null_pairs.model.eq(model) & null_pairs.null_id.eq(null_id)]
        observed_null = metrics(frame)
        reported = reported_null[reported_null.model.eq(model) & reported_null.null_id.eq(null_id)].iloc[0]
        compare(observed_null, reported, f"null {model} {null_id}")
        null_signature = null_actions[
            null_actions.model.eq(model) & null_actions.null_id.eq(null_id)
        ][["alpha", "beta"]].sort_values(["alpha", "beta"], kind="stable").reset_index(drop=True)
        assert signature.equals(null_signature), f"dose multiset {model} {null_id}"
        null_rows.append({"model": model, "null_id": null_id, **observed_null})

for row in reproduction.itertuples(index=False):
    assert row.prompt_ids_exact and row.repeat_top1_exact and row.repeat_pair_sign_exact
    assert row.all_single_token_candidates and row.all_context_prefixes_exact and row.all_first_appended_tokens_match
    assert row.baseline_margin_max_abs_delta <= row.fp16_margin_tolerance
    assert row.repeat_margin_max_abs_delta <= row.fp16_margin_tolerance
    assert row.extraction_pair_sign_disagreements_outside_tolerance_band == 0

primary = pd.DataFrame(primary_rows)
nulls = pd.DataFrame(null_rows)
eligible = int(primary.eligible_n.sum())
crossings = int(primary.strict_crossings.sum())
non_target = int(primary.non_target_n.sum())
collateral = int(primary.non_target_top1_changes.sum())
specificity = crossings / eligible - collateral / non_target
pooled_null = pd.Series(
    {
        int(null_id): (
            group.strict_crossings.sum() / group.eligible_n.sum()
            - group.non_target_top1_changes.sum() / group.non_target_n.sum()
        )
        for null_id, group in nulls.groupby("null_id")
    }
)
q95 = float(pooled_null.quantile(0.95))
empirical_p = float((1 + pooled_null.ge(specificity).sum()) / 51)
powered_positive = int((primary.eligible_n.ge(5) & primary.eligible_median_margin_shift.gt(0) & primary.strict_crossings.gt(0)).sum())

assert (eligible, crossings, non_target, collateral) == (45, 2, 240, 1)
assert np.isclose(specificity, decision["specificity_pooled"])
assert np.isclose(q95, decision["address_null_specificity_q95"])
assert np.isclose(empirical_p, decision["address_null_empirical_p_one_sided"])
assert powered_positive == 1
assert decision["status"] == "partial_protocol_transfer"
print("PASS mixed-arithmetic transfer: 2/45 crossings, 1/240 non-target changes, P=1/51; confirmation gate failed")
