"""Independently reconstruct the frozen control-chain decision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--matched-verdict",
        type=Path,
        default=Path("inputs/safety_matched_nulls/verdict.json"),
    )
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def truth(values: pd.Series) -> np.ndarray:
    return values.astype(str).str.lower().eq("true").to_numpy(bool)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = args.output_dir.resolve()
    files = {
        "displacement": "heldout_displacement.csv",
        "displacement_bootstrap": "heldout_displacement_bootstrap.csv",
        "mediator_predictions": "heldout_mediator_predictions.csv",
        "mediator_nulls": "mediator_graph_block_nulls.csv",
        "mediator_summary": "heldout_mediator_summary.csv",
        "dose_prompt": "train_grid_prompt_dose_correlations.csv",
        "dose_summary": "train_grid_dose_response_summary.csv",
        "dose_bootstrap": "train_grid_dose_response_bootstrap.csv",
        "dose_nulls": "train_grid_dose_response_nulls.csv",
        "boundary": "heldout_boundary_correspondence.csv",
        "gate": "control_mediation_gate.json",
        "config": "analysis_config.json",
    }
    paths = {name: root / filename for name, filename in files.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")

    displacement = pd.read_csv(paths["displacement"])
    displacement_boot = pd.read_csv(paths["displacement_bootstrap"])
    predictions = pd.read_csv(paths["mediator_predictions"])
    mediator_nulls = pd.read_csv(paths["mediator_nulls"])
    mediator_summary = pd.read_csv(paths["mediator_summary"]).iloc[0]
    dose_prompt = pd.read_csv(paths["dose_prompt"])
    dose_summary = pd.read_csv(paths["dose_summary"])
    dose_boot = pd.read_csv(paths["dose_bootstrap"])
    dose_nulls = pd.read_csv(paths["dose_nulls"])
    boundary = pd.read_csv(paths["boundary"])
    gate = json.loads(paths["gate"].read_text(encoding="utf-8"))
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    matched = json.loads(args.matched_verdict.read_text(encoding="utf-8"))

    expected_counts = {
        "displacement": 2,
        "displacement_bootstrap": 2 * int(config["bootstrap"]),
        "mediator_predictions": 87,
        "mediator_nulls": int(config["mediator_nulls"]),
        "mediator_summary": 1,
        "dose_prompt": 201 * 3,
        "dose_summary": 3,
        "dose_bootstrap": 3 * int(config["bootstrap"]),
        "dose_nulls": 3 * int(config["dose_nulls"]),
        "boundary": 87,
    }
    counts = {
        "displacement": len(displacement),
        "displacement_bootstrap": len(displacement_boot),
        "mediator_predictions": len(predictions),
        "mediator_nulls": len(mediator_nulls),
        "mediator_summary": 1,
        "dose_prompt": len(dose_prompt),
        "dose_summary": len(dose_summary),
        "dose_bootstrap": len(dose_boot),
        "dose_nulls": len(dose_nulls),
        "boundary": len(boundary),
    }
    counts_pass = counts == expected_counts

    errors: dict[str, float] = {}
    displacement_rebuilt = []
    for metric, row in displacement.set_index("metric").iterrows():
        if metric == "DeltaU_shift":
            observed = float(predictions["DeltaU_shift"].mean())
        elif metric == "margin_shift":
            observed = float(predictions["margin_shift"].mean())
        else:
            raise ValueError(f"Unknown displacement metric: {metric}")
        draws = displacement_boot[displacement_boot["metric"] == metric]["value"].to_numpy(float)
        low, high = np.quantile(draws, [0.025, 0.975])
        displacement_rebuilt.append((observed, float(low), float(high)))
        errors[f"displacement_{metric}"] = float(
            max(abs(observed - row["mean"]), abs(low - row["ci95_low"]), abs(high - row["ci95_high"]))
        )
    displacement_pass = bool(all(low > 0 for _, low, _ in displacement_rebuilt))

    target = predictions["margin_shift"].to_numpy(float)
    base_prediction = predictions["base_margin_shift_oof"].to_numpy(float)
    full_prediction = predictions["mediator_margin_shift_oof"].to_numpy(float)
    base_r2 = float(r2_score(target, base_prediction))
    full_r2 = float(r2_score(target, full_prediction))
    delta_r2 = full_r2 - base_r2
    null_values = mediator_nulls["delta_r2"].to_numpy(float)
    null_q95 = float(np.quantile(null_values, 0.95))
    mediator_p = float((np.sum(null_values >= delta_r2) + 1) / (len(null_values) + 1))
    crossed = truth(predictions["crossed"])
    mediator_values = {
        "base_r2": base_r2,
        "mediator_r2": full_r2,
        "delta_r2": delta_r2,
        "mediator_null_q95": null_q95,
        "empirical_p": mediator_p,
        "base_mae": float(mean_absolute_error(target, base_prediction)),
        "mediator_mae": float(mean_absolute_error(target, full_prediction)),
        "n_crossed": int(crossed.sum()),
        "base_crossing_auc": float(
            roc_auc_score(crossed, predictions["base_final_margin_oof"].to_numpy(float))
        ),
        "mediator_crossing_auc": float(
            roc_auc_score(crossed, predictions["mediator_final_margin_oof"].to_numpy(float))
        ),
    }
    errors["mediator_summary"] = float(
        max(abs(float(mediator_summary[key]) - value) for key, value in mediator_values.items())
    )
    mediator_pass = bool(delta_r2 > 0 and delta_r2 > null_q95 and mediator_p <= 0.05)

    dose_errors = []
    dose_passes = []
    for contrast, row in dose_summary.set_index("contrast").iterrows():
        observed = float(
            dose_prompt[dose_prompt["contrast"] == contrast]["rho"].median()
        )
        boot_values = dose_boot[dose_boot["contrast"] == contrast][
            "median_prompt_rho"
        ].to_numpy(float)
        null = dose_nulls[dose_nulls["contrast"] == contrast][
            "median_prompt_rho"
        ].to_numpy(float)
        low, high = np.quantile(boot_values, [0.025, 0.975])
        q95 = float(np.quantile(null, 0.95))
        p = float((np.sum(null >= observed) + 1) / (len(null) + 1))
        dose_errors.extend(
            [
                abs(observed - row["median_prompt_rho"]),
                abs(low - row["graph_bootstrap_ci95_low"]),
                abs(high - row["graph_bootstrap_ci95_high"]),
                abs(q95 - row["null_q95"]),
                abs(p - row["empirical_p"]),
            ]
        )
        dose_passes.append(observed > 0 and low > 0 and p <= 0.05)
    errors["dose_summary"] = float(max(dose_errors))
    dose_pass = bool(all(dose_passes))

    candidate_crossed = truth(boundary["candidate_crossed_recomputed"])
    strict_top1 = truth(boundary["strict_top1_flip"])
    stored_crossed = truth(boundary["crossed"])
    final_recomputed = (
        (boundary["baseline_margin"].to_numpy(float) < 0)
        & (boundary["final_margin"].to_numpy(float) >= 0)
    )
    boundary_pass = bool(
        np.array_equal(candidate_crossed, strict_top1)
        and np.array_equal(candidate_crossed, stored_crossed)
        and np.array_equal(candidate_crossed, final_recomputed)
    )
    chain_pass = bool(displacement_pass and mediator_pass and dose_pass and boundary_pass)
    assignment_specificity = bool(
        matched["empirical_one_sided_p"] <= 0.05
        and matched["real_guarded_crossings"] > matched["null_q95_crossings"]
    )
    gate_match = bool(
        gate["heldout_displacement_pass"] == displacement_pass
        and gate["incremental_mediator_pass"] == mediator_pass
        and gate["train_grid_dose_response_pass"] == dose_pass
        and gate["boundary_correspondence_pass"] == boundary_pass
        and gate["bounded_control_chain_support"] == chain_pass
        and gate["personalized_assignment_specificity"] == assignment_specificity
    )

    finite_pass = bool(
        all(np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all() for frame in (
            displacement,
            displacement_boot,
            predictions,
            mediator_nulls,
            dose_prompt,
            dose_summary,
            dose_boot,
            dose_nulls,
            boundary,
        ))
    )
    tolerance = 1e-12
    passed = bool(
        counts_pass
        and finite_pass
        and max(errors.values()) <= tolerance
        and gate_match
    )
    report = {
        "verification": "PASS" if passed else "FAIL",
        "row_counts": counts,
        "expected_counts": expected_counts,
        "finite_values": finite_pass,
        "max_reconstruction_errors": errors,
        "recomputed_gates": {
            "heldout_displacement_pass": displacement_pass,
            "incremental_mediator_pass": mediator_pass,
            "train_grid_dose_response_pass": dose_pass,
            "boundary_correspondence_pass": boundary_pass,
            "bounded_control_chain_support": chain_pass,
            "personalized_assignment_specificity": assignment_specificity,
        },
        "gate_match": gate_match,
        "sha256": {name: sha256(path) for name, path in paths.items()},
    }
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
