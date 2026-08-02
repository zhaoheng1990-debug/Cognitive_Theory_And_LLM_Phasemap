"""Read-only verifier for the v0.56 empirical-direction actuator null."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (model, control, seed), frame in rows.groupby(["model", "control", "random_seed"], dropna=False, sort=True):
        closure = frame[frame["mechanism"].eq("closure")]
        nonclosure = frame[~frame["mechanism"].eq("closure")]
        crossing = int(closure["strict_conflict_to_clean"].sum())
        changed = int(nonclosure["nonclosure_top1_changed"].sum())
        records.append({
            "model": model,
            "control": control,
            "random_seed": seed,
            "closure_n": int(len(closure)),
            "strict_conflict_to_clean": crossing,
            "closure_crossing_rate": crossing / len(closure),
            "nonclosure_n": int(len(nonclosure)),
            "nonclosure_top1_changes": changed,
            "nonclosure_change_rate": changed / len(nonclosure),
            "specificity_count": crossing - changed,
            "specificity_rate": crossing / len(closure) - changed / len(nonclosure),
            "closure_mean_margin_shift": float(closure["margin_shift"].mean()),
        })
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    summary = pd.read_csv(root / "direction_null_crossmodel_summary.csv")
    traces = pd.read_csv(root / "direction_null_crossmodel_traces.csv")
    verdicts = json.loads((root / "direction_null_crossmodel_verdicts.json").read_text(encoding="utf-8"))
    per_model = []
    checks = {}
    for verdict in verdicts:
        model = verdict["model"]
        rows = pd.read_csv(root / model / "direction_null_prompt_rows.csv")
        rebuilt = summarize(rows)
        stored = summary[summary["model"].eq(model)].copy()
        joined = stored.merge(rebuilt, on=["model", "control", "random_seed"], suffixes=("", "_recomputed"), validate="one_to_one")
        fields = ("closure_n", "strict_conflict_to_clean", "nonclosure_n", "nonclosure_top1_changes", "specificity_count")
        count_pass = all(np.array_equal(joined[field].to_numpy(), joined[f"{field}_recomputed"].to_numpy()) for field in fields)
        numeric_fields = ("closure_crossing_rate", "nonclosure_change_rate", "specificity_rate", "closure_mean_margin_shift")
        numeric_pass = all(np.max(np.abs(joined[field].to_numpy(float) - joined[f"{field}_recomputed"].to_numpy(float))) <= 1e-12 for field in numeric_fields)
        structured = rebuilt[rebuilt["control"].eq("structured_guarded")].iloc[0]
        null = rebuilt[rebuilt["control"].str.startswith("random_direction_")]
        empirical_p = float((int(np.sum(null["specificity_count"].to_numpy(int) >= int(structured["specificity_count"]))) + 1) / (len(null) + 1))
        q95 = float(np.quantile(null["specificity_count"], 0.95))
        verdict_pass = (
            bool(verdict["published_structured_top1_reproduced"])
            and int(verdict["null_count"]) == 50
            and int(verdict["structured_specificity_count"]) == int(structured["specificity_count"])
            and abs(float(verdict["empirical_p"]) - empirical_p) <= 1e-12
            and abs(float(verdict["null_specificity_count_q95"]) - q95) <= 1e-12
            and bool(verdict["structured_exceeds_null_q95"]) == bool(int(structured["specificity_count"]) > q95)
        )
        trace = traces[traces["model"].eq(model)]
        random_trace = trace[trace["control"].str.startswith("random_direction_")]
        norm_pass = bool(
            len(random_trace) > 0
            and np.isfinite(random_trace[["mean_operator_component_norm", "mean_random_operator_component_norm", "mean_precursor_component_norm", "mean_random_precursor_component_norm", "mean_structured_total_ratio", "mean_random_total_ratio"]].to_numpy(float)).all()
            and (random_trace["operator_component_norm_abs_error"].to_numpy(float) == 0.0).all()
            and (random_trace["precursor_component_norm_abs_error"].to_numpy(float) == 0.0).all()
            and random_trace["all_finite"].astype(bool).all()
        )
        checks[model] = bool(count_pass and numeric_pass and verdict_pass and norm_pass)
        per_model.append({"model": model, "count_pass": count_pass, "numeric_pass": numeric_pass, "verdict_pass": verdict_pass, "norm_pass": norm_pass, "empirical_p_recomputed": empirical_p, "q95_recomputed": q95})
    report = {
        "verifier": Path(__file__).name,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "models": per_model,
        "claim_boundary": "Only Gemma passed this empirical-subspace direction-null test; Qwen and Llama did not establish direction specificity under the frozen specificity endpoint.",
    }
    if args.report is not None:
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
