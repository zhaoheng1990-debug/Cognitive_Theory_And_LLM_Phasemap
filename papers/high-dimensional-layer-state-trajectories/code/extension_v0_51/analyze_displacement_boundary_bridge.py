"""Cross-fitted bridge from held-out DeltaU shifts to candidate-boundary crossing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold


MODELS = ("qwen", "llama", "gemma")
STRATA = ("closure", "nonclosure")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paired-input",
        type=Path,
        default=Path("intervention_behavior_coupling_v0_1/paired_intervention_behavior_rows.csv"),
    )
    parser.add_argument(
        "--controller-root", type=Path, default=Path("inputs/internal_control_outputs")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("displacement_boundary_bridge"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def add_design_columns(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    stratum = np.where(df["mechanism"].eq("closure"), "closure", "nonclosure")
    cells = pd.Series(df["model"].astype(str) + "__" + stratum, index=df.index)
    names = [f"{model}__{stratum_name}" for model in MODELS for stratum_name in STRATA]
    columns = []
    arrays = []
    delta_u = df["delta_DeltaU"].to_numpy(dtype=float)
    for name in names:
        indicator = cells.eq(name).to_numpy(dtype=float)
        arrays.extend((indicator, indicator * delta_u))
        columns.extend((f"intercept_{name}", f"slope_{name}"))
    return np.column_stack(arrays), columns


def grouped_crossfit_shift(df: pd.DataFrame, folds: int) -> np.ndarray:
    groups = df["graph_id"].to_numpy()
    splitter = GroupKFold(n_splits=min(folds, np.unique(groups).size))
    predictions = np.full(len(df), np.nan, dtype=float)
    for train_index, test_index in splitter.split(df, groups=groups):
        x_train, _ = add_design_columns(df.iloc[train_index])
        x_test, _ = add_design_columns(df.iloc[test_index])
        model = Ridge(alpha=1e-6, fit_intercept=False)
        model.fit(x_train, df.iloc[train_index]["delta_margin"].to_numpy(dtype=float))
        predictions[test_index] = model.predict(x_test)
    if not np.isfinite(predictions).all():
        raise RuntimeError("Non-finite cross-fitted margin-shift predictions")
    return predictions


def nested_boundary_probabilities(df: pd.DataFrame, folds: int) -> np.ndarray:
    groups = df["graph_id"].to_numpy()
    splitter = GroupKFold(n_splits=min(folds, np.unique(groups).size))
    probabilities = np.full(len(df), np.nan, dtype=float)
    for train_index, test_index in splitter.split(df, groups=groups):
        train = df.iloc[train_index].copy().reset_index(drop=True)
        test = df.iloc[test_index].copy()
        inner_prediction = grouped_crossfit_shift(train, max(2, folds - 1))
        inner_score = train["margin_none"].to_numpy(dtype=float) + inner_prediction
        train_at_risk = train["margin_none"].to_numpy(dtype=float) < 0
        y_train = train["crossed_boundary"].to_numpy(dtype=int)
        if np.unique(y_train[train_at_risk]).size < 2:
            continue
        calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
        calibrator.fit(inner_score[train_at_risk, None], y_train[train_at_risk])

        x_train, _ = add_design_columns(train)
        x_test, _ = add_design_columns(test)
        shift_model = Ridge(alpha=1e-6, fit_intercept=False)
        shift_model.fit(x_train, train["delta_margin"].to_numpy(dtype=float))
        test_score = test["margin_none"].to_numpy(dtype=float) + shift_model.predict(x_test)
        probabilities[test_index] = calibrator.predict_proba(test_score[:, None])[:, 1]
    return probabilities


def safe_metric(function, y: np.ndarray, score: np.ndarray) -> float:
    keep = np.isfinite(score)
    if keep.sum() == 0 or np.unique(y[keep]).size < 2:
        return np.nan
    return float(function(y[keep], score[keep]))


def binary_summary(df: pd.DataFrame) -> dict[str, float | int]:
    y = df["crossed_boundary"].to_numpy(dtype=int)
    probability = df["crossing_probability_oof"].to_numpy(dtype=float)
    predicted_margin = df["predicted_margin_policy_oof"].to_numpy(dtype=float)
    predicted = predicted_margin >= 0
    keep = np.isfinite(probability)
    tp = int(np.sum(predicted & (y == 1)))
    tn = int(np.sum((~predicted) & (y == 0)))
    fp = int(np.sum(predicted & (y == 0)))
    fn = int(np.sum((~predicted) & (y == 1)))
    return {
        "n_at_risk": int(len(df)),
        "n_crossings": int(y.sum()),
        "crossing_rate": float(y.mean()) if len(y) else np.nan,
        "auroc": safe_metric(roc_auc_score, y, probability),
        "average_precision": safe_metric(average_precision_score, y, probability),
        "brier": float(brier_score_loss(y[keep], probability[keep])) if keep.any() else np.nan,
        "zero_threshold_accuracy": float(np.mean(predicted == y)) if len(y) else np.nan,
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
    }


def continuous_summary(df: pd.DataFrame) -> dict[str, float | int]:
    observed = df["delta_margin"].to_numpy(dtype=float)
    predicted = df["predicted_delta_margin_oof"].to_numpy(dtype=float)
    return {
        "n_prompts": int(len(df)),
        "n_graphs": int(df["graph_id"].nunique()),
        "r2": float(r2_score(observed, predicted)),
        "mae": float(mean_absolute_error(observed, predicted)),
        "pearson_r": float(pearsonr(observed, predicted).statistic),
        "spearman_rho": float(spearmanr(observed, predicted).statistic),
    }


def bootstrap_intervals(
    df: pd.DataFrame, summary_function, metric_names: list[str], n_boot: int, seed: int
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    groups = np.unique(df["graph_id"].to_numpy())
    values = {name: [] for name in metric_names}
    for _ in range(n_boot):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        pieces = [df[df["graph_id"].eq(group)] for group in sampled]
        boot = pd.concat(pieces, ignore_index=True)
        summary = summary_function(boot)
        for name in metric_names:
            values[name].append(summary[name])
    intervals = {}
    for name, samples in values.items():
        finite = np.asarray(samples, dtype=float)
        finite = finite[np.isfinite(finite)]
        intervals[name] = (
            (float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975)))
            if finite.size
            else (np.nan, np.nan)
        )
    return intervals


def verify_heldout_source(df: pd.DataFrame, controller_root: Path) -> pd.DataFrame:
    audit_rows = []
    for model in MODELS:
        split = pd.read_csv(controller_root / model / "local_transition_split.csv", encoding="utf-8-sig")
        test_prompts = set(split.loc[split["split"].eq("test"), "prompt_id"].astype(str))
        observed = set(df.loc[df["model"].eq(model), "prompt_id"].astype(str))
        audit_rows.append(
            {
                "model": model,
                "n_observed_prompts": len(observed),
                "n_declared_test_prompts": len(test_prompts),
                "all_observed_are_declared_test": observed.issubset(test_prompts),
                "observed_equals_declared_test": observed == test_prompts,
            }
        )
    audit = pd.DataFrame(audit_rows)
    if not audit["observed_equals_declared_test"].all():
        raise RuntimeError("Paired intervention rows do not exactly match declared held-out prompts")
    return audit


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.paired_input, encoding="utf-8-sig")
    required = {
        "model",
        "prompt_id",
        "graph_id",
        "mechanism",
        "margin_none",
        "margin_policy",
        "delta_margin",
        "delta_DeltaU",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    source_audit = verify_heldout_source(data, args.controller_root)
    data["crossed_boundary"] = (
        data["margin_none"].lt(0) & data["margin_policy"].ge(0)
    ).astype(int)
    data["predicted_delta_margin_oof"] = grouped_crossfit_shift(data, args.folds)
    data["predicted_margin_policy_oof"] = (
        data["margin_none"] + data["predicted_delta_margin_oof"]
    )
    data["crossing_probability_oof"] = nested_boundary_probabilities(data, args.folds)

    continuous_rows = []
    boundary_rows = []
    subsets = [
        ("pooled_all", data),
        ("pooled_closure", data[data["mechanism"].eq("closure")]),
    ]
    for model in MODELS:
        model_rows = data[data["model"].eq(model)]
        subsets.extend(
            [
                (f"{model}_all", model_rows),
                (f"{model}_closure", model_rows[model_rows["mechanism"].eq("closure")]),
            ]
        )
    for index, (label, subset) in enumerate(subsets):
        continuous = continuous_summary(subset)
        intervals = bootstrap_intervals(
            subset,
            continuous_summary,
            ["r2", "mae", "pearson_r", "spearman_rho"],
            args.bootstrap,
            args.seed + index,
        )
        row = {"scope": label, **continuous}
        for metric, (low, high) in intervals.items():
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        continuous_rows.append(row)

        at_risk = subset[subset["margin_none"].lt(0)].copy()
        boundary = binary_summary(at_risk)
        binary_intervals = bootstrap_intervals(
            at_risk,
            binary_summary,
            ["crossing_rate", "auroc", "average_precision", "brier", "zero_threshold_accuracy"],
            args.bootstrap,
            args.seed + 100 + index,
        )
        boundary_row = {"scope": label, **boundary}
        for metric, (low, high) in binary_intervals.items():
            boundary_row[f"{metric}_ci_low"] = low
            boundary_row[f"{metric}_ci_high"] = high
        boundary_rows.append(boundary_row)

    calibration_input = data[data["margin_none"].lt(0) & data["crossing_probability_oof"].notna()].copy()
    calibration_input["probability_bin"] = pd.qcut(
        calibration_input["crossing_probability_oof"], q=5, duplicates="drop"
    )
    calibration = (
        calibration_input.groupby("probability_bin", observed=True)
        .agg(
            n=("prompt_id", "size"),
            mean_predicted_probability=("crossing_probability_oof", "mean"),
            observed_crossing_rate=("crossed_boundary", "mean"),
            mean_predicted_post_margin=("predicted_margin_policy_oof", "mean"),
        )
        .reset_index()
    )
    calibration["probability_bin"] = calibration["probability_bin"].astype(str)

    data.to_csv(args.output_dir / "heldout_boundary_crossing_rows.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(continuous_rows).to_csv(
        args.output_dir / "heldout_margin_shift_prediction_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(boundary_rows).to_csv(
        args.output_dir / "heldout_boundary_crossing_summary.csv", index=False, encoding="utf-8-sig"
    )
    calibration.to_csv(args.output_dir / "heldout_boundary_calibration_bins.csv", index=False, encoding="utf-8-sig")
    source_audit.to_csv(args.output_dir / "heldout_source_split_audit.csv", index=False, encoding="utf-8-sig")

    config = {
        "analysis": "held-out DeltaU-shift to candidate-boundary bridge",
        "paired_input": str(args.paired_input.resolve()),
        "controller_root": str(args.controller_root.resolve()),
        "folds": args.folds,
        "bootstrap_replicates": args.bootstrap,
        "seed": args.seed,
        "split_unit": "graph_id shared across checkpoints",
        "continuous_model": "model-by-regime affine Ridge map: delta_margin from delta_DeltaU",
        "boundary_score": "baseline margin plus out-of-fold predicted margin shift",
        "boundary_population": "prompts with baseline clean-minus-conflict margin below zero",
        "claim_boundary": "held-out calibration of candidate crossing, not open-ended correctness control",
    }
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
