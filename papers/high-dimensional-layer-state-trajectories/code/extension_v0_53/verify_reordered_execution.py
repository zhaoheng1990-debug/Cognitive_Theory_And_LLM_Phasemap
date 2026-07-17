"""Independent checks for the layer-order geometry/behaviour window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("outputs/execution_order_confirmatory")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", nargs="?", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_dir
    prompt = pd.read_csv(root / "layer_order_behavior_prompt_metrics.csv", encoding="utf-8-sig")
    summary = pd.read_csv(root / "layer_order_behavior_summary.csv", encoding="utf-8-sig")
    geometry = pd.read_csv(root / "layer_order_geometry_contrasts.csv", encoding="utf-8-sig")
    gate = json.loads((root / "layer_order_behavior_gate.json").read_text(encoding="utf-8"))
    errors = []
    if len(prompt) != 144:
        errors.append(f"expected 144 prompt rows, found {len(prompt)}")
    if set(prompt["executed_order"]) != {"native", "reverse", "fixed_permutation"}:
        errors.append("prompt condition grid mismatch")
    if len(summary) != 3 or len(geometry) != 2:
        errors.append("summary or geometry row count mismatch")
    recalculated = []
    for row in summary.itertuples(index=False):
        subset = prompt[prompt["executed_order"].eq(row.executed_order)]
        top1 = float(subset["top1_matches_native"].mean())
        candidate = float(subset["candidate_choice_matches_native"].mean())
        js = float(subset["js_divergence_from_native"].mean())
        components = int(top1 <= 0.50) + int(candidate <= 0.75) + int(
            row.mean_js_divergence_ci_low > 0.05
        )
        disruption = components >= 2
        if not np.isclose(top1, row.full_vocab_top1_agreement):
            errors.append(f"{row.executed_order}: top1 summary mismatch")
        if not np.isclose(candidate, row.candidate_choice_agreement):
            errors.append(f"{row.executed_order}: candidate summary mismatch")
        if not np.isclose(js, row.mean_js_divergence):
            errors.append(f"{row.executed_order}: JSD summary mismatch")
        if disruption != bool(row.passes_strong_functional_disruption):
            errors.append(f"{row.executed_order}: disruption gate mismatch")
        recalculated.append(
            {
                "executed_order": row.executed_order,
                "top1_agreement": top1,
                "candidate_agreement": candidate,
                "mean_js": js,
                "disruption_components": components,
                "disruption_gate": disruption,
            }
        )
    geometry_index = geometry.set_index("executed_order")
    summary_index = summary.set_index("executed_order")
    reverse = bool(
        geometry_index.loc["reverse", "passes_positive_geometry_contrast"]
        and summary_index.loc["reverse", "passes_strong_functional_disruption"]
    )
    fixed = bool(
        geometry_index.loc["fixed_permutation", "passes_positive_geometry_contrast"]
        and summary_index.loc["fixed_permutation", "passes_strong_functional_disruption"]
    )
    if reverse != bool(gate["reverse_dissociation_pass"]):
        errors.append("reverse final gate mismatch")
    if fixed != bool(gate["fixed_permutation_replication_pass"]):
        errors.append("fixed-permutation final gate mismatch")
    result = {
        "status": "PASS" if not errors and reverse and fixed else "FAIL",
        "errors": errors,
        "prompt_rows": len(prompt),
        "recalculated": recalculated,
        "reverse_dissociation_pass": reverse,
        "fixed_permutation_replication_pass": fixed,
        "reconstruction_audit": gate["reconstruction_audit"],
    }
    report = args.report or (root / "independent_verification.json")
    report.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
