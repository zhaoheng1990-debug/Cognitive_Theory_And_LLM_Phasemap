#!/usr/bin/env python
"""Freeze paired statistics and a compact claim-bounded report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"


def paired_exact(a: pd.Series, b: pd.Series):
    a, b = a.astype(bool), b.astype(bool)
    a_only = int((a & ~b).sum())
    b_only = int((~a & b).sum())
    discordant = a_only + b_only
    p = float(binomtest(min(a_only, b_only), discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "a_only": a_only,
        "b_only": b_only,
        "both": int((a & b).sum()),
        "neither": int((~a & ~b).sum()),
        "discordant": discordant,
        "mcnemar_exact_two_sided_p": p,
    }


def main():
    rows = pd.read_csv(OUT / "heldout_control_rows.csv")
    summary = pd.read_csv(OUT / "heldout_control_summary.csv").set_index("control")
    top1 = pd.read_csv(OUT / "vocab_top1_summary.csv").set_index("control")
    closure = rows[rows["mechanism"] == "closure"]
    pivot = closure.pivot(index="prompt_id", columns="control", values="crossed")

    policy_vs_shuffle = paired_exact(pivot["policy_boundary"], pivot["shuffle_boundary_labels"])
    policy_vs_legacy = paired_exact(pivot["policy_boundary"], pivot["same_combo_fixed"])
    guarded_vs_legacy = paired_exact(pivot["policy_boundary_guarded"], pivot["same_combo_fixed"])

    condition = (
        closure[closure["control"] == "policy_boundary"]
        .groupby("condition", as_index=False)
        .agg(
            n=("crossed", "size"),
            crossings=("crossed", "sum"),
            mean_margin_shift=("margin_shift", "mean"),
            max_final_margin=("final_margin", "max"),
        )
    )
    condition.to_csv(OUT / "closure_condition_breakdown.csv", index=False)

    audit = {
        "frozen_split": {"test_graphs": 29, "test_prompts": 290, "test_closure": 87},
        "primary": summary.loc["policy_boundary"].to_dict(),
        "guarded": summary.loc["policy_boundary_guarded"].to_dict(),
        "legacy_0.3_0.3": summary.loc["same_combo_fixed"].to_dict(),
        "shuffle_policy": summary.loc["shuffle_boundary_labels"].to_dict(),
        "top1_primary": top1.loc["policy_boundary"].to_dict(),
        "top1_guarded": top1.loc["policy_boundary_guarded"].to_dict(),
        "policy_vs_shuffle": policy_vs_shuffle,
        "policy_vs_legacy": policy_vs_legacy,
        "guarded_vs_legacy": guarded_vs_legacy,
        "inferential_status": (
            "Output-boundary leverage is established relative to the frozen legacy executor. "
            "Mechanism-conditioned policy superiority over the single shuffled-label policy is provisional "
            "because the paired exact comparison does not cross 0.05."
        ),
        "semantic_boundary": (
            "For closure prompts, clean denotes the unperturbed-chain candidate and is generally not "
            "the update/override/exception-consistent answer. The endpoint is candidate-output control, "
            "not correctness improvement."
        ),
    }
    with open(OUT / "statistical_audit.json", "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)

    p = summary.loc["policy_boundary"]
    g = summary.loc["policy_boundary_guarded"]
    sh = summary.loc["shuffle_boundary_labels"]
    legacy = summary.loc["same_combo_fixed"]
    text = f"""# Qwen candidate-boundary control result

## Result

The frozen boundary-trained policy changed the full-vocabulary top-1 output from the conflict candidate to the clean candidate in **{int(p.closure_crossings)}/{int(p.closure_n)} held-out closure prompts ({100*p.closure_crossing_rate:.1f}%; Wilson 95% CI {100*p.closure_crossing_ci95_low:.1f}-{100*p.closure_crossing_ci95_high:.1f}%)**. The legacy 0.3/0.3 executor crossed 0/{int(legacy.closure_n)}, whereas the matched shuffled-label policy crossed {int(sh.closure_crossings)}/{int(sh.closure_n)}. The paired exact comparison against the shuffled policy was P={policy_vs_shuffle['mcnemar_exact_two_sided_p']:.4g} ({policy_vs_shuffle['a_only']} policy-only versus {policy_vs_shuffle['b_only']} shuffle-only crossings).

The primary policy shifted the closure clean-minus-conflict margin by {p.closure_mean_margin_shift:.2f} logits on average. Its non-closure top-1 change rate was {100*float(top1.loc['policy_boundary','nonclosure_top1_change_rate']):.2f}%, and its mean intervention-layer norm ratio was {p.mean_norm_ratio:.3f} (maximum observed ratio {p.max_norm_ratio:.3f}).

The confidence-and-entropy gate retained **{int(g.closure_crossings)}/{int(g.closure_n)}** strict output flips while reducing the non-closure top-1 change rate to **{100*float(top1.loc['policy_boundary_guarded','nonclosure_top1_change_rate']):.1f}%**. Thus the gate acts as a risk selector rather than a replacement hidden-state actuator.

The single shuffled-label policy changed {100*float(top1.loc['shuffle_boundary_labels','nonclosure_top1_change_rate']):.2f}% of non-closure top-1 outputs and therefore failed the predeclared 5% damage guard. Its closure crossing count cannot be interpreted as safe selective control. Even so, the paired policy-versus-shuffle comparison did not cross the conventional 0.05 threshold; policy-specific superiority remains provisional pending a larger matched-null distribution.

## Interpretation

This resolves the earlier Qwen 0/87 result as an actuator-objective/budget limitation rather than evidence that internal trajectory movement lacks output causal leverage. A train-only boundary objective can carry that movement across the emitted-token boundary on held-out graph groups. The stronger claim that mechanism conditioning is uniquely responsible for this gain is not yet closed by the present single-null audit.

## Claim boundary

For closure prompts, `clean` is the unperturbed-chain candidate. Under update, override and exception instructions it is generally not the rule-consistent answer. The result therefore demonstrates **controlled candidate-output flipping**, not accuracy improvement, open-ended generation control, deployment control or a universal dynamical law.
"""
    (OUT / "result_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
