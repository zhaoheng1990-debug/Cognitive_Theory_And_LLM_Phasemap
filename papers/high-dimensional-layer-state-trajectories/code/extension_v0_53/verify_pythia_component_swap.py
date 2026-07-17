"""Independent structural and numerical audit of the Pythia component swap."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


COMPONENTS = ("E", "B", "N", "W")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir", type=Path,
        default=Path("outputs/pythia_component_compatibility"),
    )
    parser.add_argument(
        "--prior-outcomes", type=Path,
        default=Path(
            "inputs/pythia_training_states/"
            "phase4_prompt_outcomes.csv"
        ),
    )
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2026071712)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for start in range(0, n_boot, 1000):
        end = min(start + 1000, n_boot)
        index = rng.integers(0, len(values), size=(end - start, len(values)))
        means[start:end] = values[index].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def main() -> None:
    args = parse_args()
    outcome_path = args.result_dir / "component_swap_prompt_outcomes.csv"
    summary_path = args.result_dir / "component_swap_summary.csv"
    effects_path = args.result_dir / "orthogonal_factorial_effects.csv"
    decision_path = args.result_dir / "gate_decision.json"
    metadata_path = args.result_dir / "run_metadata.json"
    outcomes = pd.read_csv(outcome_path, dtype={"condition": str})
    outcomes["condition"] = outcomes["condition"].str.zfill(4)
    reported = pd.read_csv(summary_path, dtype={"condition": str})
    reported["condition"] = reported["condition"].str.zfill(4)
    effects = pd.read_csv(effects_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_codes = ["".join(map(str, bits)) for bits in itertools.product((0, 1), repeat=4)]

    checks: dict[str, bool] = {}
    checks["row_count_16x53"] = len(outcomes) == 16 * 53
    checks["all_conditions_present"] = sorted(outcomes["condition"].unique()) == expected_codes
    checks["each_condition_has_53_prompts"] = bool(
        (outcomes.groupby("condition").size() == 53).all()
    )
    checks["condition_prompt_keys_unique"] = not outcomes.duplicated(
        ["condition", "prompt_index"]
    ).any()
    checks["prompt_sets_identical"] = outcomes.groupby("condition")["prompt_index"].apply(
        lambda values: tuple(sorted(values))
    ).nunique() == 1
    bit_match = True
    for code, frame in outcomes.groupby("condition"):
        for component, character in zip(COMPONENTS, code):
            bit_match &= bool((frame[component] == int(character)).all())
    checks["condition_bits_match_columns"] = bit_match

    recalculated = outcomes.groupby("condition").agg(
        candidate_pair_accuracy=("pair_correct", "mean"),
        candidate_margin_mean=("candidate_margin", "mean"),
    ).reset_index()
    merged = recalculated.merge(
        reported[["condition", "candidate_pair_accuracy", "candidate_margin_mean"]],
        on="condition", suffixes=("_recalculated", "_reported"), validate="one_to_one",
    )
    accuracy_error = float(np.max(np.abs(
        merged["candidate_pair_accuracy_recalculated"]
        - merged["candidate_pair_accuracy_reported"]
    )))
    margin_error = float(np.max(np.abs(
        merged["candidate_margin_mean_recalculated"]
        - merged["candidate_margin_mean_reported"]
    )))
    checks["summary_accuracy_exact"] = accuracy_error < 1e-12
    checks["summary_margin_exact"] = margin_error < 1e-12

    prior = pd.read_csv(args.prior_outcomes)
    anchor_comparisons = {}
    for code, revision in (("0000", "step0"), ("1111", "step143000")):
        current = outcomes[outcomes["condition"].eq(code)].copy()
        old = prior[prior["revision"].eq(revision)].copy()
        current["problem_id"] = current["problem_id"].astype(str)
        old["problem_id"] = old["problem_id"].astype(str)
        joined = current.merge(old, on="problem_id", suffixes=("_new", "_old"), validate="one_to_one")
        anchor_comparisons[code] = {
            "n": len(joined),
            "pair_decision_agreement": float(np.mean(
                joined["pair_correct_new"] == joined["pair_correct_old"]
            )),
            "mean_margin_difference": float(np.mean(
                joined["candidate_margin_new"] - joined["candidate_margin_old"]
            )),
            "max_absolute_margin_difference": float(np.max(np.abs(
                joined["candidate_margin_new"] - joined["candidate_margin_old"]
            ))),
        }
    checks["anchor_pair_decisions_reproduce"] = all(
        item["n"] == 53 and item["pair_decision_agreement"] == 1.0
        for item in anchor_comparisons.values()
    )

    pivot_accuracy = outcomes.pivot(
        index="prompt_index", columns="condition", values="pair_correct"
    ).astype(float)
    pivot_margin = outcomes.pivot(
        index="prompt_index", columns="condition", values="candidate_margin"
    ).astype(float)
    accuracy = pivot_accuracy.mean(axis=0).to_dict()
    single_codes = ("1000", "0100", "0010", "0001")
    best_no_blocks = max(accuracy[code] for code in expected_codes if code[1] == "0")
    best_no_output = max(accuracy[code] for code in expected_codes if code[3] == "0")
    superadditive = (
        pivot_margin["1111"] - pivot_margin["0000"]
        - sum(
            (pivot_margin[code] - pivot_margin["0000"] for code in single_codes),
            start=pd.Series(0.0, index=pivot_margin.index),
        )
    ).to_numpy(float)
    super_ci = bootstrap_mean_ci(superadditive, args.bootstrap, args.seed)
    independent_gates = {
        "training_function": bool(
            accuracy["1111"] - accuracy["0000"] >= 0.20 and accuracy["1111"] >= 0.75
        ),
        "no_single_component_sufficient": bool(
            max(accuracy[code] for code in single_codes) <= accuracy["1111"] - 0.10
        ),
        "trained_blocks_necessary": bool(best_no_blocks <= accuracy["1111"] - 0.10),
        "trained_output_head_necessary": bool(best_no_output <= accuracy["1111"] - 0.10),
        "superadditive_compatibility": bool(super_ci[0] > 0),
    }
    checks["independent_gates_match_report"] = all(
        independent_gates[key] == decision["gates"][key] for key in independent_gates
    )

    design_sign = np.array(
        [[int(character) * 2 - 1 for character in code] for code in expected_codes]
    )
    reconstruction_errors = {}
    for metric, pivot in (("pair_correct", pivot_accuracy), ("candidate_margin", pivot_margin)):
        condition_means = pivot[expected_codes].mean(axis=0).to_numpy(float)
        reconstruction = np.full(16, condition_means.mean())
        metric_effects = effects[effects["metric"].eq(metric)]
        for _, row in metric_effects.iterrows():
            indices = [COMPONENTS.index(component) for component in str(row["effect"]).split(":")]
            reconstruction += float(row["orthogonal_beta"]) * np.prod(
                design_sign[:, indices], axis=1
            )
        reconstruction_errors[metric] = float(np.max(np.abs(reconstruction - condition_means)))
    checks["orthogonal_effects_reconstruct_means"] = all(
        value < 1e-10 for value in reconstruction_errors.values()
    )

    fingerprints = metadata["fingerprints"]
    early_revision = metadata["early_revision"]
    trained_revision = metadata["trained_revision"]
    checks["all_component_fingerprints_differ_across_training"] = all(
        fingerprints[early_revision][component]["sha256"]
        != fingerprints[trained_revision][component]["sha256"]
        for component in COMPONENTS
    )
    checks["input_output_fingerprints_distinct_within_checkpoint"] = all(
        fingerprints[revision]["E"]["sha256"] != fingerprints[revision]["W"]["sha256"]
        for revision in (early_revision, trained_revision)
    )

    output = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "max_summary_accuracy_error": accuracy_error,
        "max_summary_margin_error": margin_error,
        "anchor_comparisons": anchor_comparisons,
        "independent_gates": independent_gates,
        "independent_superadditive_margin_mean": float(superadditive.mean()),
        "independent_superadditive_margin_ci": list(super_ci),
        "factorial_reconstruction_max_errors": reconstruction_errors,
        "source_hashes": {
            path.name: sha256(path)
            for path in (outcome_path, summary_path, effects_path, decision_path, metadata_path)
        },
    }
    if args.report is not None:
        args.report.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    if output["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
