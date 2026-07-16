"""Prospective held-out forecasting from partially observed candidate-margin trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODELS = ("qwen", "llama", "gemma")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectory-root",
        type=Path,
        default=Path("inputs/relation_trajectory_tables"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("partial_trajectory_forecasting")
    )
    parser.add_argument("--shuffles", type=int, default=50)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def slope(values: np.ndarray) -> np.ndarray:
    if values.shape[1] < 2:
        return np.zeros(values.shape[0], dtype=float)
    x = np.arange(values.shape[1], dtype=float)
    x = x - x.mean()
    return (values @ x) / np.sum(x * x)


def reversals(values: np.ndarray) -> np.ndarray:
    if values.shape[1] < 3:
        return np.zeros(values.shape[0], dtype=float)
    velocity = np.diff(values, axis=1)
    return np.sum(velocity[:, 1:] * velocity[:, :-1] < 0, axis=1).astype(float)


def history_features(
    data: pd.DataFrame,
    observed_layers: list[int],
    include_delta: bool,
    permutation: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    r = data[[f"R_L{layer}" for layer in observed_layers]].to_numpy(dtype=float)
    d = data[[f"dR_L{layer}" for layer in observed_layers]].to_numpy(dtype=float)
    if permutation is not None and len(observed_layers) > 1:
        expected = list(range(len(observed_layers) - 1))
        if permutation.ndim != 2 or permutation.shape != (len(data), len(expected)):
            raise ValueError("History permutation must have one pre-current permutation per row")
        if any(sorted(row.tolist()) != expected for row in permutation):
            raise ValueError("Each history permutation must cover all pre-current positions")
        row_index = np.arange(len(data))[:, None]
        r = np.column_stack([r[row_index, permutation], r[:, -1]])
        d = np.column_stack([d[row_index, permutation], d[:, -1]])

    def summarize(values: np.ndarray, prefix: str) -> tuple[list[np.ndarray], list[str]]:
        lag = values[:, -2] if values.shape[1] > 1 else values[:, -1]
        velocity = values[:, -1] - lag
        variation = (
            np.sum(np.abs(np.diff(values, axis=1)), axis=1)
            if values.shape[1] > 1
            else np.zeros(values.shape[0], dtype=float)
        )
        arrays = [
            values[:, -1],
            lag,
            velocity,
            np.mean(values, axis=1),
            slope(values),
            variation,
            reversals(values),
        ]
        names = [
            f"{prefix}_current",
            f"{prefix}_lag",
            f"{prefix}_velocity",
            f"{prefix}_mean",
            f"{prefix}_slope",
            f"{prefix}_variation",
            f"{prefix}_reversals",
        ]
        return arrays, names

    arrays, names = summarize(r, "R")
    if include_delta:
        delta_arrays, delta_names = summarize(d, "dR")
        arrays.extend(delta_arrays)
        names.extend(delta_names)
    return np.column_stack(arrays), names


def current_features(data: pd.DataFrame, cutoff: int, include_delta: bool) -> tuple[np.ndarray, list[str]]:
    names = [f"R_L{cutoff}"]
    if include_delta:
        names.append(f"dR_L{cutoff}")
    return data[names].to_numpy(dtype=float), names


def fit_regression(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
    model.fit(x_train, y_train)
    return model.predict(x_test)


def fit_classifier(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    if np.unique(y_train).size < 2:
        return np.full(len(x_test), float(np.mean(y_train)), dtype=float)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def safe_correlation(function, observed: np.ndarray, predicted: np.ndarray) -> float:
    if len(observed) < 3 or np.std(observed) < 1e-12 or np.std(predicted) < 1e-12:
        return np.nan
    return float(function(observed, predicted).statistic)


def metrics(observed: np.ndarray, predicted: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    boundary = (observed >= 0).astype(int)
    predicted_boundary = (predicted >= 0).astype(int)
    if np.unique(boundary).size < 2:
        auroc = average_precision = np.nan
    else:
        auroc = float(roc_auc_score(boundary, probability))
        average_precision = float(average_precision_score(boundary, probability))
    return {
        "r2": float(r2_score(observed, predicted)),
        "mae": float(mean_absolute_error(observed, predicted)),
        "pearson_r": safe_correlation(pearsonr, observed, predicted),
        "spearman_rho": safe_correlation(spearmanr, observed, predicted),
        "boundary_auroc": auroc,
        "boundary_average_precision": average_precision,
        "boundary_brier": float(brier_score_loss(boundary, probability)),
        "boundary_accuracy": float(accuracy_score(boundary, predicted_boundary)),
    }


def bootstrap_intervals(
    rows: pd.DataFrame, metric_names: list[str], n_boot: int, seed: int
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    group_values = rows["graph_id"].to_numpy()
    groups = np.unique(group_values)
    group_indices = {group: np.flatnonzero(group_values == group) for group in groups}
    observed = rows["observed_final_margin"].to_numpy(float)
    predicted = rows["predicted_final_margin"].to_numpy(float)
    probability = rows["predicted_boundary_probability"].to_numpy(float)
    collected = {name: [] for name in metric_names}
    for _ in range(n_boot):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        index = np.concatenate([group_indices[group] for group in sampled])
        summary = metrics(
            observed[index],
            predicted[index],
            probability[index],
        )
        for name in metric_names:
            collected[name].append(summary[name])
    intervals = {}
    for name, values in collected.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        intervals[name] = (
            (float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975)))
            if finite.size
            else (np.nan, np.nan)
        )
    return intervals


def load_model_data(root: Path, model_key: str) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    model_dir = root / model_key
    data = pd.read_csv(model_dir / "local_transition_baseline_features.csv", encoding="utf-8-sig")
    split = pd.read_csv(model_dir / "local_transition_split.csv", encoding="utf-8-sig")
    config = json.loads((model_dir / "local_transition_config.json").read_text(encoding="utf-8"))
    layers = [int(layer) for layer in config["decision_layers"]]
    merged = data.merge(
        split[["prompt_id", "split"]], on="prompt_id", how="left", validate="one_to_one"
    )
    if merged["split"].isna().any():
        raise RuntimeError(f"{model_key}: missing split labels")
    return merged[merged["split"].eq("train")].copy(), merged[merged["split"].eq("test")].copy(), layers


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    prediction_rows = []
    null_rows = []

    for model_offset, model_key in enumerate(MODELS):
        train, test, decision_layers = load_model_data(args.trajectory_root, model_key)
        y_train = train["R_final_margin"].to_numpy(dtype=float)
        y_test = test["R_final_margin"].to_numpy(dtype=float)
        train_boundary = (y_train >= 0).astype(int)

        for cutoff_index, cutoff in enumerate(decision_layers):
            observed_layers = decision_layers[: cutoff_index + 1]
            feature_sets = {
                "current_R": (
                    *current_features(train, cutoff, include_delta=False),
                    *current_features(test, cutoff, include_delta=False),
                ),
                "ordered_R_history": (
                    *history_features(train, observed_layers, include_delta=False),
                    *history_features(test, observed_layers, include_delta=False),
                ),
                "ordered_R_dR_history": (
                    *history_features(train, observed_layers, include_delta=True),
                    *history_features(test, observed_layers, include_delta=True),
                ),
            }
            for feature_set, packed in feature_sets.items():
                x_train, train_names, x_test, test_names = packed
                if train_names != test_names:
                    raise RuntimeError("Train/test feature mismatch")
                prediction = fit_regression(x_train, y_train, x_test)
                probability = fit_classifier(x_train, train_boundary, x_test)
                score = metrics(y_test, prediction, probability)
                row_frame = test[["prompt_id", "graph_id", "condition", "mechanism"]].copy()
                row_frame["model"] = model_key
                row_frame["cutoff_layer"] = cutoff
                row_frame["n_observed_layers"] = len(observed_layers)
                row_frame["feature_set"] = feature_set
                row_frame["observed_final_margin"] = y_test
                row_frame["predicted_final_margin"] = prediction
                row_frame["predicted_boundary_probability"] = probability
                prediction_rows.append(row_frame)
                intervals = bootstrap_intervals(
                    row_frame,
                    list(score),
                    args.bootstrap,
                    args.seed + model_offset * 100 + cutoff_index * 10 + len(summary_rows),
                )
                summary = {
                    "model": model_key,
                    "cutoff_layer": cutoff,
                    "n_observed_layers": len(observed_layers),
                    "observed_layers": ";".join(map(str, observed_layers)),
                    "feature_set": feature_set,
                    "n_train": len(train),
                    "n_test": len(test),
                    "n_test_graphs": int(test["graph_id"].nunique()),
                    "n_features": x_train.shape[1],
                    **score,
                }
                for metric_name, (low, high) in intervals.items():
                    summary[f"{metric_name}_ci_low"] = low
                    summary[f"{metric_name}_ci_high"] = high
                summary_rows.append(summary)

            if len(observed_layers) >= 3:
                rng = np.random.default_rng(args.seed + model_offset * 1000 + cutoff_index)
                for shuffle_id in range(args.shuffles):
                    train_permutation = np.vstack(
                        [rng.permutation(len(observed_layers) - 1) for _ in range(len(train))]
                    )
                    test_permutation = np.vstack(
                        [rng.permutation(len(observed_layers) - 1) for _ in range(len(test))]
                    )
                    x_train, _ = history_features(
                        train, observed_layers, include_delta=False, permutation=train_permutation
                    )
                    x_test, _ = history_features(
                        test, observed_layers, include_delta=False, permutation=test_permutation
                    )
                    prediction = fit_regression(x_train, y_train, x_test)
                    probability = fit_classifier(x_train, train_boundary, x_test)
                    score = metrics(y_test, prediction, probability)
                    null_rows.append(
                        {
                            "model": model_key,
                            "cutoff_layer": cutoff,
                            "n_observed_layers": len(observed_layers),
                            "shuffle_id": shuffle_id,
                            "permutation": "independent_within_prompt_precurrent",
                            **score,
                        }
                    )

    summary = pd.DataFrame(summary_rows)
    nulls = pd.DataFrame(null_rows)
    if len(nulls):
        history_mask = summary["feature_set"].eq("ordered_R_history")
        summary["order_null_r2_mean"] = np.nan
        summary["order_null_r2_sd"] = np.nan
        summary["order_null_r2_q95"] = np.nan
        summary["real_r2_vs_order_null_z"] = np.nan
        summary["real_r2_exceeds_order_null_q95"] = False
        for index in summary[history_mask].index:
            row = summary.loc[index]
            matched = nulls[
                nulls["model"].eq(row["model"])
                & nulls["cutoff_layer"].eq(row["cutoff_layer"])
            ]
            if matched.empty:
                continue
            mean = float(matched["r2"].mean())
            sd = float(matched["r2"].std(ddof=1))
            q95 = float(matched["r2"].quantile(0.95))
            summary.loc[index, "order_null_r2_mean"] = mean
            summary.loc[index, "order_null_r2_sd"] = sd
            summary.loc[index, "order_null_r2_q95"] = q95
            summary.loc[index, "real_r2_vs_order_null_z"] = (
                (float(row["r2"]) - mean) / sd if sd > 0 else np.nan
            )
            summary.loc[index, "real_r2_exceeds_order_null_q95"] = bool(float(row["r2"]) > q95)

    summary.to_csv(
        args.output_dir / "early_trajectory_forecasting_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        args.output_dir / "early_trajectory_forecasting_rows.csv", index=False, encoding="utf-8-sig"
    )
    nulls.to_csv(
        args.output_dir / "early_trajectory_order_nulls.csv", index=False, encoding="utf-8-sig"
    )
    config = {
        "analysis": "prospective final candidate-boundary forecasting from partial decision-window history",
        "asa_root": str(args.trajectory_root.resolve()),
        "models": list(MODELS),
        "split": "frozen 67-train/29-test relation-graph groups",
        "target": "final clean-minus-conflict output-logit margin",
        "predictor_information_boundary": "candidate-margin observations at the cutoff layer and earlier only",
        "regression": "StandardScaler plus Ridge(alpha=10)",
        "classification": "StandardScaler plus balanced LogisticRegression(C=1)",
        "order_null": "independent within-prompt permutation of all pre-current positions; current cutoff retained",
        "n_order_shuffles": args.shuffles,
        "bootstrap_replicates": args.bootstrap,
        "seed": args.seed,
        "claim_boundary": "reduced-order held-out forecast; not a closed-form or universal state equation",
    }
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
