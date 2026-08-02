"""Read-only verification for the v0.56 predecision coordinate audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


MODELS = ("qwen", "llama", "gemma")
FEATURES = (
    "raw_current",
    "raw_recent",
    "matched_current",
    "matched_recent",
    "DeltaU_pre",
    "ordered_prefix",
    "cutoff_hidden",
)
CONTRASTS = (
    ("DeltaU_pre", "matched_recent"),
    ("ordered_prefix", "matched_recent"),
    ("cutoff_hidden", "matched_recent"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def bh_adjust(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def sign_flip(reference, candidate, target, groups, repeats, seed):
    values = []
    for graph_id in np.unique(groups):
        mask = groups == graph_id
        ref = f1_score(target[mask], reference[mask], labels=(0, 1, 2), average="macro", zero_division=0)
        cand = f1_score(target[mask], candidate[mask], labels=(0, 1, 2), average="macro", zero_division=0)
        values.append(float(cand - ref))
    values = np.asarray(values, dtype=float)
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    exceed, completed = 0, 0
    while completed < repeats:
        chunk = min(10000, repeats - completed)
        null = (rng.choice(np.array((-1.0, 1.0)), size=(chunk, len(values))) * values[None, :]).mean(axis=1)
        exceed += int(np.sum(null >= observed))
        completed += chunk
    return observed, float((exceed + 1) / (repeats + 1))


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    config = json.loads((root / "analysis_config.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(root / "coordinate_oof_predictions.csv", encoding="utf-8-sig")
    performance = pd.read_csv(root / "coordinate_feature_performance.csv", encoding="utf-8-sig")
    comparisons = pd.read_csv(root / "coordinate_primary_contrasts.csv", encoding="utf-8-sig")
    gates = pd.read_csv(root / "coordinate_gates.csv", encoding="utf-8-sig")
    manifests = json.loads((root / "model_extraction_manifests.json").read_text(encoding="utf-8"))

    checks = {}
    expected_models = tuple(config["models"])
    checks["models"] = tuple(sorted(predictions["model"].unique())) == tuple(sorted(expected_models))
    checks["no_final_layer"] = not bool(config["final_layer_used"])
    checks["no_final_output_target"] = not bool(config["final_output_target_used"])
    checks["manifests"] = all(not bool(entry["final_layer_used"]) and int(entry["n_prompts"]) > 0 for entry in manifests)
    checks["prediction_columns"] = all(f"prediction_{name}" in predictions.columns for name in FEATURES)
    checks["row_count"] = len(predictions) == sum(int(entry["n_prompts"]) for entry in manifests)

    rebuilt_performance = []
    rebuilt_comparisons = []
    for model_index, model_key in enumerate(expected_models):
        frame = predictions[predictions["model"].eq(model_key)].copy()
        target = frame["target"].to_numpy(int)
        groups = frame["graph_id"].to_numpy(int)
        for feature in FEATURES:
            pred = frame[f"prediction_{feature}"].to_numpy(int)
            rebuilt_performance.append({
                "model": model_key,
                "feature": feature,
                "oof_macro_f1_recomputed": float(f1_score(target, pred, labels=(0, 1, 2), average="macro", zero_division=0)),
            })
        for contrast_index, (candidate, reference) in enumerate(CONTRASTS):
            difference, p = sign_flip(
                frame[f"prediction_{reference}"].to_numpy(int),
                frame[f"prediction_{candidate}"].to_numpy(int),
                target,
                groups,
                int(config["sign_flips"]),
                int(config["seed"]) + 10000 * (model_index + 1) + contrast_index,
            )
            rebuilt_comparisons.append({
                "model": model_key,
                "candidate": candidate,
                "reference": reference,
                "difference_recomputed": difference,
                "p_recomputed": p,
            })
    performance_check = performance.merge(pd.DataFrame(rebuilt_performance), on=["model", "feature"], validate="one_to_one")
    comparison_check = comparisons.merge(pd.DataFrame(rebuilt_comparisons), on=["model", "candidate", "reference"], validate="one_to_one")
    q_recomputed = bh_adjust(comparison_check["p_recomputed"].to_numpy(float))
    checks["performance"] = bool(np.max(np.abs(performance_check["oof_macro_f1"] - performance_check["oof_macro_f1_recomputed"])) <= 1e-12)
    checks["differences"] = bool(np.max(np.abs(comparison_check["macro_f1_difference"] - comparison_check["difference_recomputed"])) <= 1e-12)
    checks["p_values"] = bool(np.max(np.abs(comparison_check["p"] - comparison_check["p_recomputed"])) <= 1e-12)
    checks["q_values"] = bool(np.max(np.abs(comparison_check["q"] - q_recomputed)) <= 1e-12)
    recomputed_gates = []
    for model_key, group in comparison_check.assign(q_recomputed=q_recomputed).groupby("model", sort=True):
        primary = group[group["candidate"].isin(("DeltaU_pre", "ordered_prefix"))]
        recomputed_gates.append({
            "model": model_key,
            "supports_recomputed": bool(len(primary) == 2 and (primary["difference_recomputed"] > 0).all() and (primary["q_recomputed"] <= 0.05).all()),
        })
    gate_check = gates.merge(pd.DataFrame(recomputed_gates), on="model", validate="one_to_one")
    checks["gates"] = bool(np.array_equal(gate_check["supports_distinct_predecision_ordered_readback"].to_numpy(bool), gate_check["supports_recomputed"].to_numpy(bool)))
    report = {
        "verifier": Path(__file__).name,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "claim_boundary": "The audit does not establish DeltaU_pre superiority over matched recent readbacks; it does not use final-layer or final-output targets.",
    }
    if args.report is not None:
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
