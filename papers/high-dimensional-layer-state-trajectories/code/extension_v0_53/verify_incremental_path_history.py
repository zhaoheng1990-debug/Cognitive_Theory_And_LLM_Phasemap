"""Independently verify frozen trajectory-incremental prediction outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


FEATURES = (
    "prompt_only",
    "current_margin",
    "current_hidden",
    "unordered_prefix",
    "ordered_prefix",
)
BASELINES = ("current_margin", "current_hidden", "unordered_prefix")
KEYS = ["task", "model", "horizon"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-conditions", type=int, required=True)
    parser.add_argument("--expected-prompts-per-condition", type=int, default=384)
    parser.add_argument("--expected-order-nulls", type=int, default=199)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bh_adjust(values: pd.Series) -> pd.Series:
    p = values.to_numpy(float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return pd.Series(result, index=values.index)


def main() -> None:
    args = parse_args()
    root = args.output_dir.resolve()
    names = {
        "predictions": "trajectory_feature_model_predictions.csv",
        "performance": "trajectory_feature_model_performance.csv",
        "comparisons": "trajectory_incremental_comparisons.csv",
        "nulls": "trajectory_order_nulls.csv",
        "gates": "trajectory_incremental_condition_gates.csv",
        "decision": "trajectory_incremental_gate.json",
        "config": "analysis_config.json",
    }
    paths = {key: root / value for key, value in names.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing verification inputs: {missing}")

    predictions = pd.read_csv(paths["predictions"], encoding="utf-8-sig")
    performance = pd.read_csv(paths["performance"], encoding="utf-8-sig")
    comparisons = pd.read_csv(paths["comparisons"], encoding="utf-8-sig")
    nulls = pd.read_csv(paths["nulls"], encoding="utf-8-sig")
    gates = pd.read_csv(paths["gates"], encoding="utf-8-sig")
    decision = json.loads(paths["decision"].read_text(encoding="utf-8"))

    expected_prediction_rows = (
        args.expected_conditions * args.expected_prompts_per_condition
    )
    expected_performance_rows = args.expected_conditions * len(FEATURES)
    expected_comparison_rows = args.expected_conditions * 4
    expected_null_rows = args.expected_conditions * args.expected_order_nulls
    row_counts = {
        "predictions": len(predictions),
        "performance": len(performance),
        "comparisons": len(comparisons),
        "nulls": len(nulls),
        "gates": len(gates),
    }
    expected_counts = {
        "predictions": expected_prediction_rows,
        "performance": expected_performance_rows,
        "comparisons": expected_comparison_rows,
        "nulls": expected_null_rows,
        "gates": args.expected_conditions,
    }
    counts_pass = row_counts == expected_counts

    probability_columns = [f"{name}_probability" for name in FEATURES]
    finite_pass = bool(
        np.isfinite(predictions[probability_columns].to_numpy(float)).all()
        and np.isfinite(performance[["balanced_accuracy", "roc_auc"]].to_numpy(float)).all()
        and np.isfinite(nulls[["balanced_accuracy"]].to_numpy(float)).all()
    )

    recomputed_performance = []
    recomputed_comparisons = []
    for key, frame in predictions.groupby(KEYS, sort=True):
        target = frame["target"].to_numpy(int)
        scores = {}
        for feature in FEATURES:
            probability = frame[f"{feature}_probability"].to_numpy(float)
            score = float(balanced_accuracy_score(target, probability >= 0.5))
            scores[feature] = score
            recomputed_performance.append(
                dict(zip(KEYS, key))
                | {"feature_model": feature, "balanced_accuracy_recomputed": score}
            )
        for baseline in BASELINES:
            recomputed_comparisons.append(
                dict(zip(KEYS, key))
                | {
                    "contrast": f"ordered_vs_{baseline}",
                    "difference_recomputed": scores["ordered_prefix"] - scores[baseline],
                }
            )

        null_frame = nulls
        for column, value in zip(KEYS, key):
            null_frame = null_frame[null_frame[column] == value]
        null_values = null_frame["balanced_accuracy"].to_numpy(float)
        ordered = scores["ordered_prefix"]
        recomputed_comparisons.append(
            dict(zip(KEYS, key))
            | {
                "contrast": "ordered_vs_order_shuffle",
                "difference_recomputed": ordered - float(null_values.mean()),
                "p_recomputed": float(
                    (np.sum(null_values >= ordered) + 1) / (len(null_values) + 1)
                ),
                "order_null_mean_recomputed": float(null_values.mean()),
                "order_null_q95_recomputed": float(np.quantile(null_values, 0.95)),
            }
        )

    performance_check = performance.merge(
        pd.DataFrame(recomputed_performance), on=KEYS + ["feature_model"], validate="one_to_one"
    )
    performance_error = float(
        np.max(
            np.abs(
                performance_check["balanced_accuracy"].to_numpy(float)
                - performance_check["balanced_accuracy_recomputed"].to_numpy(float)
            )
        )
    )

    comparison_check = comparisons.merge(
        pd.DataFrame(recomputed_comparisons), on=KEYS + ["contrast"], validate="one_to_one"
    )
    difference_error = float(
        np.max(
            np.abs(
                comparison_check["balanced_accuracy_difference"].to_numpy(float)
                - comparison_check["difference_recomputed"].to_numpy(float)
            )
        )
    )
    shuffle = comparison_check[comparison_check["contrast"] == "ordered_vs_order_shuffle"]
    shuffle_p_error = float(
        np.max(np.abs(shuffle["p"].to_numpy(float) - shuffle["p_recomputed"].to_numpy(float)))
    )
    shuffle_q95_error = float(
        np.max(
            np.abs(
                shuffle["order_null_q95"].to_numpy(float)
                - shuffle["order_null_q95_recomputed"].to_numpy(float)
            )
        )
    )

    q_recomputed = comparisons.groupby("contrast", group_keys=False)["p"].apply(bh_adjust)
    q_error = float(np.max(np.abs(comparisons["q"].to_numpy(float) - q_recomputed.to_numpy(float))))
    rebuilt_comparisons = comparisons.copy()
    rebuilt_comparisons["q_recomputed"] = q_recomputed
    rebuilt_gates = []
    required = {
        "ordered_vs_current_margin",
        "ordered_vs_current_hidden",
        "ordered_vs_unordered_prefix",
        "ordered_vs_order_shuffle",
    }
    for key, frame in rebuilt_comparisons.groupby(KEYS, sort=True):
        indexed = frame.set_index("contrast")
        passed = bool(
            set(indexed.index) == required
            and all(
                indexed.loc[name, "balanced_accuracy_difference"] > 0
                and indexed.loc[name, "q_recomputed"] <= 0.05
                for name in required
            )
        )
        rebuilt_gates.append(dict(zip(KEYS, key)) | {"gate_recomputed": passed})
    gate_check = gates.merge(pd.DataFrame(rebuilt_gates), on=KEYS, validate="one_to_one")
    stored_gate = gate_check["passes_incremental_ordered_path_gate"].astype(str).str.lower() == "true"
    gate_match = bool(np.array_equal(stored_gate.to_numpy(), gate_check["gate_recomputed"].to_numpy(bool)))
    pass_count = int(gate_check["gate_recomputed"].sum())
    decision_match = bool(
        decision["condition_pass_count"] == pass_count
        and decision["condition_total"] == args.expected_conditions
    )

    tolerance = 1e-12
    passed = bool(
        counts_pass
        and finite_pass
        and performance_error <= tolerance
        and difference_error <= tolerance
        and shuffle_p_error <= tolerance
        and shuffle_q95_error <= tolerance
        and q_error <= tolerance
        and gate_match
        and decision_match
    )
    report = {
        "verification": "PASS" if passed else "FAIL",
        "output_dir": str(root),
        "row_counts": row_counts,
        "expected_counts": expected_counts,
        "finite_values": finite_pass,
        "max_errors": {
            "balanced_accuracy": performance_error,
            "contrast_difference": difference_error,
            "shuffle_p": shuffle_p_error,
            "shuffle_q95": shuffle_q95_error,
            "bh_q": q_error,
        },
        "gate_match": gate_match,
        "decision_match": decision_match,
        "recomputed_pass_count": pass_count,
        "sha256": {key: sha256(path) for key, path in paths.items()},
    }
    output = root / "independent_verification.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
