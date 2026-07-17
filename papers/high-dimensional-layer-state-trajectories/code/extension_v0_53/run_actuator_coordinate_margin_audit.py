"""Audit the actuator-to-coordinate-to-margin control chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("inputs/qwen_boundary_control"),
    )
    parser.add_argument(
        "--matched-null-dir",
        type=Path,
        default=Path("inputs/safety_matched_nulls"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/actuator_coordinate_margin"),
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--mediator-nulls", type=int, default=999)
    parser.add_argument("--dose-nulls", type=int, default=999)
    parser.add_argument("--seed", type=int, default=2026071717)
    return parser.parse_args()


def truth(values: pd.Series) -> np.ndarray:
    return values.astype(str).str.lower().eq("true").to_numpy(bool)


def graph_bootstrap_stat(
    frame: pd.DataFrame,
    value: str,
    repeats: int,
    rng: np.random.Generator,
    statistic: str = "mean",
) -> tuple[float, float, float, np.ndarray]:
    graph_ids = frame["graph_id"].drop_duplicates().to_numpy()

    def calculate(sample: pd.DataFrame) -> float:
        values = sample[value].to_numpy(float)
        return float(np.mean(values) if statistic == "mean" else np.median(values))

    observed = calculate(frame)
    draws = np.empty(repeats, dtype=float)
    grouped = {graph: frame[frame["graph_id"] == graph] for graph in graph_ids}
    for index in range(repeats):
        selected = rng.choice(graph_ids, size=len(graph_ids), replace=True)
        sample = pd.concat([grouped[graph] for graph in selected], ignore_index=True)
        draws[index] = calculate(sample)
    low, high = np.quantile(draws, [0.025, 0.975])
    return observed, float(low), float(high), draws


def design_matrix(frame: pd.DataFrame, include_mediator: bool) -> np.ndarray:
    numeric = frame[
        [
            "baseline_margin",
            "baseline_DeltaU",
            "mean_alpha",
            "mean_beta",
            "mean_dose",
            "mean_p_closure",
            "mean_entropy",
        ]
    ].to_numpy(float)
    condition = pd.get_dummies(frame["condition"], dtype=float).to_numpy(float)
    pieces = [numeric, condition]
    if include_mediator:
        pieces.append(frame[["DeltaU_shift"]].to_numpy(float))
    return np.column_stack(pieces)


def grouped_ridge_predictions(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    prediction = np.full(len(target), np.nan, dtype=float)
    splitter = GroupKFold(n_splits=5)
    for train, test in splitter.split(features, target, groups):
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(features[train], target[train])
        prediction[test] = model.predict(features[test])
    if not np.isfinite(prediction).all():
        raise RuntimeError("Non-finite grouped ridge predictions")
    return prediction


def permuted_graph_profiles(
    frame: pd.DataFrame, rng: np.random.Generator
) -> np.ndarray:
    pivot = frame.pivot(index="graph_id", columns="condition", values="DeltaU_shift")
    if pivot.isna().any().any():
        raise ValueError("Incomplete graph-by-condition mediator profile")
    graph_ids = pivot.index.to_numpy()
    permuted = pivot.loc[rng.permutation(graph_ids)]
    permuted.index = graph_ids
    permuted.index.name = "graph_id"
    permuted.columns.name = "condition"
    lookup = permuted.stack().rename("permuted").reset_index()
    merged = frame[["graph_id", "condition"]].merge(
        lookup, on=["graph_id", "condition"], how="left", validate="many_to_one"
    )
    return merged["permuted"].to_numpy(float)


def prompt_spearman(frame: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    rows = []
    for prompt_id, group in frame.groupby("prompt_id", sort=False):
        rho = float(spearmanr(group[x].to_numpy(float), group[y].to_numpy(float)).statistic)
        rows.append(
            {
                "prompt_id": prompt_id,
                "graph_id": int(group["graph_id"].iloc[0]),
                "condition": group["condition"].iloc[0],
                "contrast": f"{x}_to_{y}",
                "rho": rho,
            }
        )
    return pd.DataFrame(rows)


def dose_permutation_null(
    frame: pd.DataFrame,
    y: str,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    groups = [group for _, group in frame.groupby("prompt_id", sort=False)]
    observed_values = [
        spearmanr(group["dose"].to_numpy(float), group[y].to_numpy(float)).statistic
        for group in groups
    ]
    observed = float(np.nanmedian(observed_values))
    null = np.empty(repeats, dtype=float)
    for index in range(repeats):
        values = []
        for group in groups:
            shuffled = rng.permutation(group["dose"].to_numpy(float))
            values.append(spearmanr(shuffled, group[y].to_numpy(float)).statistic)
        null[index] = float(np.nanmedian(values))
    return observed, null


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    heldout = pd.read_csv(args.input_dir / "heldout_control_rows.csv", encoding="utf-8-sig")
    actions = pd.read_csv(
        args.input_dir / "actions_policy_boundary_guarded.csv", encoding="utf-8-sig"
    )
    top1 = pd.read_csv(args.input_dir / "vocab_top1_rows.csv", encoding="utf-8-sig")
    grid = pd.read_csv(args.input_dir / "train_action_grid_merged.csv", encoding="utf-8-sig")
    matched = json.loads(
        (args.matched_null_dir / "verdict.json").read_text(encoding="utf-8")
    )

    action_summary = actions.groupby("prompt_id", as_index=False).agg(
        mean_alpha=("alpha", "mean"),
        mean_beta=("beta", "mean"),
        mean_p_closure=("mean_p_closure", "first"),
        mean_entropy=("mean_entropy", "first"),
        gate_open=("gate_open", "first"),
    )
    dose = actions.assign(
        layer_dose=actions["alpha"].astype(float) ** 2
        + actions["beta"].astype(float) ** 2
    ).groupby("prompt_id", as_index=False)["layer_dose"].mean()
    action_summary = action_summary.merge(dose, on="prompt_id", validate="one_to_one").rename(
        columns={"layer_dose": "mean_dose"}
    )
    closure = heldout[
        (heldout["control"] == "policy_boundary_guarded")
        & (heldout["mechanism"] == "closure")
    ].copy()
    closure = closure.merge(action_summary, on="prompt_id", validate="one_to_one")
    if len(closure) != 87 or closure["graph_id"].nunique() != 29:
        raise ValueError("Unexpected held-out closure sample")

    displacement_rows = []
    displacement_bootstrap_rows = []
    for value in ("DeltaU_shift", "margin_shift"):
        observed, low, high, draws = graph_bootstrap_stat(
            closure, value, args.bootstrap, rng, statistic="mean"
        )
        displacement_rows.append(
            {"metric": value, "mean": observed, "ci95_low": low, "ci95_high": high}
        )
        displacement_bootstrap_rows.extend(
            {"metric": value, "bootstrap_index": index, "value": draw}
            for index, draw in enumerate(draws)
        )
    displacement = pd.DataFrame(displacement_rows)
    displacement_pass = bool((displacement["ci95_low"] > 0).all())

    target = closure["margin_shift"].to_numpy(float)
    groups = closure["graph_id"].to_numpy(int)
    base_features = design_matrix(closure, include_mediator=False)
    full_features = design_matrix(closure, include_mediator=True)
    base_prediction = grouped_ridge_predictions(base_features, target, groups)
    full_prediction = grouped_ridge_predictions(full_features, target, groups)
    base_r2 = float(r2_score(target, base_prediction))
    full_r2 = float(r2_score(target, full_prediction))
    delta_r2 = full_r2 - base_r2

    mediator_nulls = np.empty(args.mediator_nulls, dtype=float)
    for index in range(args.mediator_nulls):
        null_frame = closure.copy()
        null_frame["DeltaU_shift"] = permuted_graph_profiles(closure, rng)
        null_prediction = grouped_ridge_predictions(
            design_matrix(null_frame, include_mediator=True), target, groups
        )
        mediator_nulls[index] = float(r2_score(target, null_prediction) - base_r2)
    mediator_p = float(
        (np.sum(mediator_nulls >= delta_r2) + 1) / (len(mediator_nulls) + 1)
    )
    mediator_q95 = float(np.quantile(mediator_nulls, 0.95))
    mediator_pass = bool(delta_r2 > 0 and delta_r2 > mediator_q95 and mediator_p <= 0.05)

    prediction_rows = closure[
        ["prompt_id", "graph_id", "condition", "baseline_margin", "DeltaU_shift", "margin_shift", "crossed"]
    ].copy()
    prediction_rows["base_margin_shift_oof"] = base_prediction
    prediction_rows["mediator_margin_shift_oof"] = full_prediction
    prediction_rows["base_final_margin_oof"] = closure["baseline_margin"].to_numpy(float) + base_prediction
    prediction_rows["mediator_final_margin_oof"] = closure["baseline_margin"].to_numpy(float) + full_prediction
    crossed = truth(closure["crossed"])
    crossing_metrics = {
        "n_crossed": int(crossed.sum()),
        "base_crossing_auc": float(roc_auc_score(crossed, prediction_rows["base_final_margin_oof"])),
        "mediator_crossing_auc": float(
            roc_auc_score(crossed, prediction_rows["mediator_final_margin_oof"])
        ),
    }

    closure_grid = grid[grid["mechanism"] == "closure"].copy()
    zero = closure_grid[closure_grid["action_label"] == "a0.0_b0.0"][
        ["prompt_id", "DeltaU"]
    ].rename(columns={"DeltaU": "zero_DeltaU"})
    closure_grid = closure_grid.merge(zero, on="prompt_id", validate="many_to_one")
    closure_grid["DeltaU_shift_grid"] = (
        closure_grid["DeltaU"].to_numpy(float)
        - closure_grid["zero_DeltaU"].to_numpy(float)
    )
    dose_frames = [
        prompt_spearman(closure_grid, "dose", "DeltaU_shift_grid"),
        prompt_spearman(closure_grid, "dose", "margin_shift"),
        prompt_spearman(closure_grid, "DeltaU_shift_grid", "margin_shift"),
    ]
    dose_prompt = pd.concat(dose_frames, ignore_index=True)
    dose_rows = []
    dose_bootstrap_rows = []
    dose_null_rows = []
    for contrast, frame in dose_prompt.groupby("contrast", sort=False):
        observed, low, high, bootstrap_draws = graph_bootstrap_stat(
            frame.rename(columns={"rho": "value"}),
            "value",
            args.bootstrap,
            rng,
            statistic="median",
        )
        if contrast == "dose_to_DeltaU_shift_grid":
            _, null = dose_permutation_null(
                closure_grid, "DeltaU_shift_grid", args.dose_nulls, rng
            )
        elif contrast == "dose_to_margin_shift":
            _, null = dose_permutation_null(
                closure_grid, "margin_shift", args.dose_nulls, rng
            )
        else:
            groups_grid = [group for _, group in closure_grid.groupby("prompt_id", sort=False)]
            null = np.empty(args.dose_nulls, dtype=float)
            for index in range(args.dose_nulls):
                values = []
                for group in groups_grid:
                    shifted = rng.permutation(group["DeltaU_shift_grid"].to_numpy(float))
                    values.append(
                        spearmanr(shifted, group["margin_shift"].to_numpy(float)).statistic
                    )
                null[index] = float(np.nanmedian(values))
        p = float((np.sum(null >= observed) + 1) / (len(null) + 1))
        dose_rows.append(
            {
                "contrast": contrast,
                "median_prompt_rho": observed,
                "graph_bootstrap_ci95_low": low,
                "graph_bootstrap_ci95_high": high,
                "null_q95": float(np.quantile(null, 0.95)),
                "empirical_p": p,
            }
        )
        dose_bootstrap_rows.extend(
            {"contrast": contrast, "bootstrap_index": index, "median_prompt_rho": draw}
            for index, draw in enumerate(bootstrap_draws)
        )
        dose_null_rows.extend(
            {"contrast": contrast, "null_index": index, "median_prompt_rho": draw}
            for index, draw in enumerate(null)
        )
    dose_summary = pd.DataFrame(dose_rows)
    dose_pass = bool(
        (
            (dose_summary["median_prompt_rho"] > 0)
            & (dose_summary["graph_bootstrap_ci95_low"] > 0)
            & (dose_summary["empirical_p"] <= 0.05)
        ).all()
    )

    top1_closure = top1[
        (top1["control"] == "policy_boundary_guarded")
        & (top1["mechanism"] == "closure")
    ][["prompt_id", "strict_conflict_to_clean", "top1_class"]]
    boundary = closure[["prompt_id", "baseline_margin", "final_margin", "margin_shift", "crossed"]].merge(
        top1_closure, on="prompt_id", validate="one_to_one"
    )
    boundary["candidate_crossed_recomputed"] = (
        (boundary["baseline_margin"].to_numpy(float) < 0)
        & (boundary["final_margin"].to_numpy(float) >= 0)
    )
    boundary["strict_top1_flip"] = truth(boundary["strict_conflict_to_clean"])
    boundary["boundary_headroom"] = (
        boundary["margin_shift"].to_numpy(float)
        + boundary["baseline_margin"].to_numpy(float)
    )
    boundary_pass = bool(
        np.array_equal(
            boundary["candidate_crossed_recomputed"].to_numpy(bool),
            boundary["strict_top1_flip"].to_numpy(bool),
        )
        and np.array_equal(
            truth(boundary["crossed"]), boundary["candidate_crossed_recomputed"].to_numpy(bool)
        )
    )

    assignment_specificity = bool(
        matched["empirical_one_sided_p"] <= 0.05
        and matched["real_guarded_crossings"] > matched["null_q95_crossings"]
    )
    chain_pass = bool(
        displacement_pass and mediator_pass and dose_pass and boundary_pass
    )
    gate = {
        "window": "control_mediation",
        "heldout_displacement_pass": displacement_pass,
        "incremental_mediator_pass": mediator_pass,
        "train_grid_dose_response_pass": dose_pass,
        "boundary_correspondence_pass": boundary_pass,
        "bounded_control_chain_support": chain_pass,
        "personalized_assignment_specificity": assignment_specificity,
        "matched_assignment_empirical_p": matched["empirical_one_sided_p"],
        "interpretation": (
            "bounded actuator-coordinate-margin chain; personalized assignment not supported"
            if chain_pass and not assignment_specificity
            else "bounded actuator-coordinate-margin chain with assignment specificity"
            if chain_pass
            else "control chain not closed"
        ),
        "claim_boundary": (
            "Qwen controlled relation-graph candidate boundary only; mechanistic-chain audit, "
            "not a formal natural indirect effect, answer correctness, open-ended generation, "
            "deployment control, universal dynamics or a hidden confidence-gated control ontology"
        ),
    }

    displacement.to_csv(args.output_dir / "heldout_displacement.csv", index=False)
    pd.DataFrame(displacement_bootstrap_rows).to_csv(
        args.output_dir / "heldout_displacement_bootstrap.csv", index=False
    )
    prediction_rows.to_csv(args.output_dir / "heldout_mediator_predictions.csv", index=False)
    pd.DataFrame(
        {
            "null_index": np.arange(args.mediator_nulls),
            "delta_r2": mediator_nulls,
        }
    ).to_csv(args.output_dir / "mediator_graph_block_nulls.csv", index=False)
    pd.DataFrame(
        [
            {
                "base_r2": base_r2,
                "mediator_r2": full_r2,
                "delta_r2": delta_r2,
                "mediator_null_q95": mediator_q95,
                "empirical_p": mediator_p,
                "base_mae": float(mean_absolute_error(target, base_prediction)),
                "mediator_mae": float(mean_absolute_error(target, full_prediction)),
                **crossing_metrics,
            }
        ]
    ).to_csv(args.output_dir / "heldout_mediator_summary.csv", index=False)
    dose_prompt.to_csv(args.output_dir / "train_grid_prompt_dose_correlations.csv", index=False)
    dose_summary.to_csv(args.output_dir / "train_grid_dose_response_summary.csv", index=False)
    pd.DataFrame(dose_bootstrap_rows).to_csv(
        args.output_dir / "train_grid_dose_response_bootstrap.csv", index=False
    )
    pd.DataFrame(dose_null_rows).to_csv(
        args.output_dir / "train_grid_dose_response_nulls.csv", index=False
    )
    boundary.to_csv(args.output_dir / "heldout_boundary_correspondence.csv", index=False)
    (args.output_dir / "control_mediation_gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    config = {
        "input_dir": str(args.input_dir.resolve()),
        "matched_null_dir": str(args.matched_null_dir.resolve()),
        "bootstrap": args.bootstrap,
        "mediator_nulls": args.mediator_nulls,
        "dose_nulls": args.dose_nulls,
        "seed": args.seed,
        "ridge_alpha": 1.0,
        "group_folds": 5,
        "protocol": str(
            (Path.cwd() / "outputs/control_mediation_protocol_v0_1.md").resolve()
        ),
    }
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
