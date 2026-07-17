"""Test whether ordered margin history predicts beyond the current state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


MODELS = ("qwen", "llama", "gemma")
TASKS = {
    "lexical_category": (
        Path("inputs/lexical_task"),
        "lexical_prompt_outcomes.csv",
    ),
    "arithmetic_addition": (
        Path("inputs/addition_task"),
        "arithmetic_prompt_outcomes.csv",
    ),
    "arithmetic_mixed": (
        Path("inputs/mixed_arithmetic_task"),
        "arithmetic_prompt_outcomes.csv",
    ),
}
HORIZONS = (0.25, 0.50, 0.75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/incremental_path_history"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(TASKS),
        default=["lexical_category", "arithmetic_addition"],
    )
    parser.add_argument("--order-nulls", type=int, default=199)
    parser.add_argument("--group-swaps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=2026071713)
    return parser.parse_args()


def cutoff_for(n_layers: int, fraction: float) -> int:
    return max(2, min(n_layers - 1, int(np.floor(n_layers * fraction))))


def prefix_features(margins: np.ndarray, cutoff: int) -> dict[str, np.ndarray]:
    prefix = np.asarray(margins[:, :cutoff], dtype=np.float32)
    earlier = prefix[:, :-1]
    current = prefix[:, -1:]
    unordered = np.c_[
        current,
        earlier.mean(axis=1),
        earlier.std(axis=1),
        earlier.min(axis=1),
        earlier.max(axis=1),
        np.quantile(earlier, 0.25, axis=1),
        np.quantile(earlier, 0.75, axis=1),
    ]
    return {
        "current_margin": current,
        "unordered_prefix": unordered.astype(np.float32),
        "ordered_prefix": prefix,
    }


def classifier(kind: str, seed: int):
    logistic = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="liblinear",
        C=1.0,
        random_state=seed,
    )
    if kind == "hidden":
        return make_pipeline(
            StandardScaler(),
            PCA(n_components=32, svd_solver="randomized", random_state=seed),
            logistic,
        )
    return make_pipeline(StandardScaler(), logistic)


def grouped_predictions(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    folds: int,
    kind: str,
    seed: int,
) -> np.ndarray:
    probability = np.full(len(target), np.nan, dtype=float)
    splitter = GroupKFold(n_splits=folds)
    for fold, (train, test) in enumerate(splitter.split(features, target, groups)):
        model = classifier(kind, seed + fold)
        model.fit(features[train], target[train])
        probability[test] = model.predict_proba(features[test])[:, 1]
    if not np.isfinite(probability).all():
        raise RuntimeError("Cross-validated predictions contain non-finite values")
    return probability


def balanced_weights(target: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=int)
    counts = np.bincount(target, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Both target classes are required")
    return np.where(target == 1, 0.5 / counts[1], 0.5 / counts[0])


def paired_group_swap_p(
    target: np.ndarray,
    probability_a: np.ndarray,
    probability_b: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
    repeats: int,
) -> tuple[float, float]:
    weights = balanced_weights(target)
    correct_a = (probability_a >= 0.5) == target
    correct_b = (probability_b >= 0.5) == target
    contribution = weights * (correct_a.astype(float) - correct_b.astype(float))
    unique_groups = np.unique(groups)
    group_contribution = np.array(
        [contribution[groups == group].sum() for group in unique_groups], dtype=float
    )
    observed = float(group_contribution.sum())
    exceed = 0
    completed = 0
    chunk = 10000
    while completed < repeats:
        count = min(chunk, repeats - completed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(count, len(unique_groups)))
        values = signs @ group_contribution
        exceed += int(np.sum(values >= observed))
        completed += count
    return observed, float((exceed + 1) / (repeats + 1))


def bh_adjust(values: pd.Series) -> pd.Series:
    p = values.to_numpy(dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return pd.Series(result, index=values.index)


def shuffled_prefix(prefix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shuffled = np.array(prefix, copy=True)
    for row in range(len(shuffled)):
        shuffled[row, :-1] = shuffled[row, rng.permutation(prefix.shape[1] - 1)]
    return shuffled


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    performance_rows = []
    prediction_rows = []
    comparison_rows = []
    order_null_rows = []
    condition_index = 0
    for task in args.tasks:
        root, outcome_name = TASKS[task]
        for model in MODELS:
            margins = np.asarray(
                np.load(root / model / "candidate_margin_layers.npy", mmap_mode="r"),
                dtype=np.float32,
            )
            hidden = np.load(root / model / "hidden_last_token_layers_float16.npy", mmap_mode="r")
            outcomes = pd.read_csv(root / model / outcome_name, encoding="utf-8-sig")
            target = (outcomes["final_margin"].to_numpy(float) > 0).astype(int)
            groups = outcomes["problem_id"].to_numpy(int)
            prompt_features = pd.get_dummies(
                outcomes[["condition", "mechanism"]].astype(str), dtype=float
            ).to_numpy(np.float32)
            for horizon in HORIZONS:
                cutoff = cutoff_for(margins.shape[1], horizon)
                features = prefix_features(margins, cutoff)
                features["prompt_only"] = prompt_features
                features["current_hidden"] = np.asarray(hidden[:, cutoff - 1], dtype=np.float32)
                probabilities = {}
                for feature_name in (
                    "prompt_only",
                    "current_margin",
                    "current_hidden",
                    "unordered_prefix",
                    "ordered_prefix",
                ):
                    kind = "hidden" if feature_name == "current_hidden" else "low_dimensional"
                    probability = grouped_predictions(
                        features[feature_name],
                        target,
                        groups,
                        args.folds,
                        kind,
                        args.seed + condition_index * 100 + len(probabilities),
                    )
                    probabilities[feature_name] = probability
                    balanced = float(
                        balanced_accuracy_score(target, probability >= 0.5)
                    )
                    auc = float(roc_auc_score(target, probability))
                    performance_rows.append(
                        {
                            "task": task,
                            "model": model,
                            "horizon": horizon,
                            "cutoff_layers": cutoff,
                            "feature_model": feature_name,
                            "balanced_accuracy": balanced,
                            "roc_auc": auc,
                        }
                    )
                for row_index, outcome in outcomes.iterrows():
                    record = {
                        "task": task,
                        "model": model,
                        "horizon": horizon,
                        "prompt_id": outcome["prompt_id"],
                        "problem_id": int(outcome["problem_id"]),
                        "target": int(target[row_index]),
                    }
                    record.update(
                        {
                            f"{name}_probability": float(value[row_index])
                            for name, value in probabilities.items()
                        }
                    )
                    prediction_rows.append(record)

                rng = np.random.default_rng(args.seed + condition_index * 100_000)
                ordered = probabilities["ordered_prefix"]
                for baseline in ("current_margin", "current_hidden", "unordered_prefix"):
                    difference, p = paired_group_swap_p(
                        target,
                        ordered,
                        probabilities[baseline],
                        groups,
                        rng,
                        args.group_swaps,
                    )
                    comparison_rows.append(
                        {
                            "task": task,
                            "model": model,
                            "horizon": horizon,
                            "contrast": f"ordered_vs_{baseline}",
                            "balanced_accuracy_difference": difference,
                            "p": p,
                        }
                    )

                order_null_values = []
                prefix = features["ordered_prefix"]
                for null_index in range(args.order_nulls):
                    null_probability = grouped_predictions(
                        shuffled_prefix(prefix, rng),
                        target,
                        groups,
                        args.folds,
                        "low_dimensional",
                        args.seed + condition_index * 1000 + null_index,
                    )
                    value = float(
                        balanced_accuracy_score(target, null_probability >= 0.5)
                    )
                    order_null_values.append(value)
                    order_null_rows.append(
                        {
                            "task": task,
                            "model": model,
                            "horizon": horizon,
                            "null_index": null_index,
                            "balanced_accuracy": value,
                        }
                    )
                ordered_ba = next(
                    row["balanced_accuracy"]
                    for row in reversed(performance_rows)
                    if row["task"] == task
                    and row["model"] == model
                    and row["horizon"] == horizon
                    and row["feature_model"] == "ordered_prefix"
                )
                null_array = np.asarray(order_null_values, dtype=float)
                comparison_rows.append(
                    {
                        "task": task,
                        "model": model,
                        "horizon": horizon,
                        "contrast": "ordered_vs_order_shuffle",
                        "balanced_accuracy_difference": float(
                            ordered_ba - null_array.mean()
                        ),
                        "p": float(
                            (np.sum(null_array >= ordered_ba) + 1)
                            / (len(null_array) + 1)
                        ),
                        "order_null_mean": float(null_array.mean()),
                        "order_null_q95": float(np.quantile(null_array, 0.95)),
                    }
                )
                print(
                    f"[{task} {model} {horizon:.2f}] ordered_BA={ordered_ba:.3f} "
                    f"current={next(row['balanced_accuracy'] for row in reversed(performance_rows) if row['task']==task and row['model']==model and row['horizon']==horizon and row['feature_model']=='current_margin'):.3f}",
                    flush=True,
                )
                condition_index += 1

    performance = pd.DataFrame(performance_rows)
    predictions = pd.DataFrame(prediction_rows)
    comparisons = pd.DataFrame(comparison_rows)
    order_nulls = pd.DataFrame(order_null_rows)
    comparisons["q"] = comparisons.groupby("contrast", group_keys=False)["p"].apply(
        bh_adjust
    )
    gate_rows = []
    for (task, model, horizon), group in comparisons.groupby(
        ["task", "model", "horizon"]
    ):
        indexed = group.set_index("contrast")
        required = (
            "ordered_vs_current_margin",
            "ordered_vs_current_hidden",
            "ordered_vs_unordered_prefix",
            "ordered_vs_order_shuffle",
        )
        passed = bool(
            all(
                indexed.loc[name, "balanced_accuracy_difference"] > 0
                and indexed.loc[name, "q"] <= 0.05
                for name in required
            )
        )
        gate_rows.append(
            {
                "task": task,
                "model": model,
                "horizon": horizon,
                "passes_incremental_ordered_path_gate": passed,
            }
        )
    gate_frame = pd.DataFrame(gate_rows)
    pass_count = int(gate_frame["passes_incremental_ordered_path_gate"].sum())
    passing = gate_frame[gate_frame["passes_incremental_ordered_path_gate"]]
    if args.tasks == ["arithmetic_mixed"]:
        bounded = bool(pass_count >= 4 and passing["model"].nunique() >= 2)
    else:
        bounded = bool(
            pass_count >= 4
            and passing["task"].nunique() == len(args.tasks)
            and passing["model"].nunique() >= 2
        )
    broad = bool(pass_count >= 12)
    gate = {
        "window": "trajectory_incremental_prediction",
        "condition_pass_count": pass_count,
        "condition_total": len(gate_frame),
        "broad_path_memory_support": broad,
        "bounded_path_memory_support": bounded,
        "interpretation": (
            "broad incremental ordered-path prediction"
            if broad
            else "bounded incremental ordered-path prediction"
            if bounded
            else "no general incremental path-memory support beyond current state"
        ),
        "claim_boundary": (
            "held-out prediction of a final candidate readback boundary; not answer "
            "correctness, hidden-state closure or a universal dynamical equation"
        ),
    }
    performance.to_csv(
        args.output_dir / "trajectory_feature_model_performance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        args.output_dir / "trajectory_feature_model_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparisons.to_csv(
        args.output_dir / "trajectory_incremental_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )
    order_nulls.to_csv(
        args.output_dir / "trajectory_order_nulls.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gate_frame.to_csv(
        args.output_dir / "trajectory_incremental_condition_gates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.output_dir / "trajectory_incremental_gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    config = {
        "script": str(Path(__file__).resolve()),
        "tasks": args.tasks,
        "models": list(MODELS),
        "horizons": list(HORIZONS),
        "folds": args.folds,
        "order_nulls": args.order_nulls,
        "group_swaps": args.group_swaps,
        "seed": args.seed,
        "protocol": str(
            (Path.cwd() / "outputs/incremental_path_history_protocol_v0_1.md").resolve()
        ),
    }
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
