# -*- coding: utf-8 -*-
"""
DA-ASA-2D.3 Policy Replay Robustness Audit

Purpose
-------
Audit whether the DA-ASA-2D.2 PASS result is robust rather than a table artifact.

Inputs expected in current folder or subfolders:
  - da_asa2d2_v2_selected_predictions.csv
  - da_asa2d2_v2_policy_outcomes_aggregated.csv
  - da_asa2d2_v2_replay_rows.csv              optional, used for cross-check
  - da_asa2d2_v2_metric_verdicts.csv          optional, used for cross-check
  - da_asa2d2_v2_replay_diagnostics.json      optional, used for metadata

This script does NOT run model forward passes.
It uses graph_id + condition + action_class table replay only.

Outputs:
  da_asa2d3_robustness_outputs/
    da_asa2d3_base_metric_summary.csv
    da_asa2d3_equivalence_audit.csv
    da_asa2d3_condition_contribution.csv
    da_asa2d3_family_contribution.csv
    da_asa2d3_leave_one_condition_out.csv
    da_asa2d3_bootstrap_ci.csv
    da_asa2d3_noise_stress.csv
    da_asa2d3_dropout_stress.csv
    da_asa2d3_robustness_diagnostics.json

Notes
-----
- Index-like columns are ignored as outcome metrics.
- Default primary metric is pred_clean if present, then R_final.
- Higher is better for pred_clean and R_final.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================
# Config
# =========================

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "da_asa2d3_robustness_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_CANDIDATES = [
    "da_asa2d2_v2_selected_predictions.csv",
    "da_asa2d2_selected_predictions.csv",
]
POLICY_CANDIDATES = [
    "da_asa2d2_v2_policy_outcomes_aggregated.csv",
    "da_asa2d2_policy_outcomes_aggregated.csv",
]
REPLAY_CANDIDATES = [
    "da_asa2d2_v2_replay_rows.csv",
    "da_asa2d2_replay_rows.csv",
]
VERDICT_CANDIDATES = [
    "da_asa2d2_v2_metric_verdicts.csv",
]
DIAG_CANDIDATES = [
    "da_asa2d2_v2_replay_diagnostics.json",
    "da_asa2d2_replay_diagnostics.json",
]

KEY_COLS = ["graph_id", "condition"]
ACTION_COL = "action_class"
PRED_ACTION_COL = "pred_action"
ORACLE_ACTION_COL = "oracle_action"

FIXED_ACTIONS_DEFAULT = [
    "NO_INTERVENTION",
    "CORE_CLOSURE_UPDATE",
    "OVERRIDE_THREE_STAGE",
    "EQUAL_EVIDENCE_ORDER",
]

INDEX_LIKE_COLS = {
    "Unnamed: 0",
    "idx",
    "index",
    "level_0",
    "original_row",
    "row_index",
    "sample_id",
    "source_row",
}

PREFERRED_METRICS = ["pred_clean", "R_final"]
HIGHER_IS_BETTER = {
    "pred_clean": True,
    "R_final": True,
    "clean_hit_rate": True,
    "clean_rate": True,
    "target_clean_rate": True,
    "correct_hit_rate": True,
    "correct_rate": True,
    "success_rate": True,
    "Gen_C": True,
    "gen_clean": True,
    "gen_correct": True,
    "score": True,
    "utility": True,
    "safety_adjusted_gain": True,
    "error_rate": False,
    "Gen_E": False,
    "gen_error": False,
    "wrong_rate": False,
    "loss": False,
}

RNG_SEED = 20260605
BOOT_N = 2000
STRESS_N = 1000
NOISE_RATES = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
DROPOUT_RATES = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


# =========================
# IO helpers
# =========================

def find_file(candidates: List[str], root: Path = ROOT, required: bool = True) -> Optional[Path]:
    for name in candidates:
        direct = root / name
        if direct.exists():
            return direct
        hits = list(root.rglob(name))
        if hits:
            return hits[0]
    if required:
        raise FileNotFoundError(f"None of these files found: {candidates}")
    return None


def read_csv(candidates: List[str], required: bool = True) -> Optional[pd.DataFrame]:
    p = find_file(candidates, required=required)
    if p is None:
        return None
    print(f"[LOAD] {p}")
    return pd.read_csv(p)


def read_json(candidates: List[str], required: bool = False) -> Optional[dict]:
    p = find_file(candidates, required=required)
    if p is None:
        return None
    print(f"[LOAD] {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_string_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


# =========================
# Metric helpers
# =========================

def get_metric_cols(policy: pd.DataFrame) -> List[str]:
    numeric_cols = policy.select_dtypes(include=[np.number]).columns.tolist()
    metric_cols = []
    for c in numeric_cols:
        if c in INDEX_LIKE_COLS:
            continue
        if c in KEY_COLS or c == ACTION_COL:
            continue
        metric_cols.append(c)
    ordered = [c for c in PREFERRED_METRICS if c in metric_cols]
    ordered += [c for c in metric_cols if c not in ordered]
    return ordered


def higher_is_better(metric: str) -> bool:
    return HIGHER_IS_BETTER.get(metric, True)


def better_delta(value_a: float, value_b: float, metric: str) -> float:
    """Positive means A better than B."""
    if higher_is_better(metric):
        return value_a - value_b
    return value_b - value_a


def family_from_condition(cond: str) -> str:
    s = str(cond).lower()
    if s.startswith("stable"):
        return "stable"
    if s.startswith("closure"):
        return "closure"
    if s.startswith("competition"):
        return "competition"
    if "hallucination" in s:
        return "hallucination_like"
    if "non" in s and "target" in s:
        return "non_target"
    return "other"


# =========================
# Replay core
# =========================

def replay_actions(
    selected: pd.DataFrame,
    policy: pd.DataFrame,
    action_values: pd.Series,
    strategy: str,
    metric_cols: List[str],
) -> pd.DataFrame:
    left = selected[KEY_COLS].copy()
    left["strategy"] = strategy
    left[ACTION_COL] = action_values.astype(str).values
    out = left.merge(policy[KEY_COLS + [ACTION_COL] + metric_cols], on=KEY_COLS + [ACTION_COL], how="left", validate="many_to_one")
    out["_matched"] = out[metric_cols].notna().any(axis=1)
    return out


def build_replay_table(selected: pd.DataFrame, policy: pd.DataFrame, fixed_actions: List[str], metric_cols: List[str]) -> pd.DataFrame:
    frames = []
    frames.append(replay_actions(selected, policy, selected[PRED_ACTION_COL], "predicted_policy", metric_cols))
    frames.append(replay_actions(selected, policy, selected[ORACLE_ACTION_COL], "oracle_policy", metric_cols))
    for act in fixed_actions:
        frames.append(replay_actions(selected, policy, pd.Series([act] * len(selected), index=selected.index), f"fixed__{act}", metric_cols))
    return pd.concat(frames, ignore_index=True, sort=False)


def summarize_replay(replay: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    rows = []
    for strategy, sub in replay.groupby("strategy", sort=True):
        matched = int(sub["_matched"].sum())
        missing = int((~sub["_matched"]).sum())
        for metric in metric_cols:
            rows.append({
                "strategy": strategy,
                "metric": metric,
                "n": int(len(sub)),
                "matched": matched,
                "missing": missing,
                "missing_action_rate": 1.0 - float(sub["_matched"].mean()),
                "metric_mean": float(pd.to_numeric(sub[metric], errors="coerce").mean()),
                "metric_std": float(pd.to_numeric(sub[metric], errors="coerce").std(ddof=1)),
                "higher_is_better": higher_is_better(metric),
            })
    return pd.DataFrame(rows)


def metric_strategy_mean(replay: pd.DataFrame, metric: str, strategy: str) -> float:
    sub = replay[replay["strategy"] == strategy]
    return float(pd.to_numeric(sub[metric], errors="coerce").mean())


def fixed_strategies(replay: pd.DataFrame) -> List[str]:
    return sorted([s for s in replay["strategy"].unique() if str(s).startswith("fixed__")])


def best_fixed_strategy(replay: pd.DataFrame, metric: str) -> Tuple[str, float]:
    rows = []
    for s in fixed_strategies(replay):
        rows.append((s, metric_strategy_mean(replay, metric, s)))
    if not rows:
        return "", float("nan")
    if higher_is_better(metric):
        return max(rows, key=lambda x: x[1])
    return min(rows, key=lambda x: x[1])


def base_metric_summary(replay: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    rows = []
    for metric in metric_cols:
        pred = metric_strategy_mean(replay, metric, "predicted_policy")
        oracle = metric_strategy_mean(replay, metric, "oracle_policy")
        no = metric_strategy_mean(replay, metric, "fixed__NO_INTERVENTION") if "fixed__NO_INTERVENTION" in replay["strategy"].unique() else float("nan")
        best_fixed, best_fixed_val = best_fixed_strategy(replay, metric)
        fixed_vals = [metric_strategy_mean(replay, metric, s) for s in fixed_strategies(replay)]
        fixed_avg = float(np.nanmean(fixed_vals)) if fixed_vals else float("nan")
        rows.append({
            "metric": metric,
            "higher_is_better": higher_is_better(metric),
            "predicted": pred,
            "oracle": oracle,
            "no_intervention": no,
            "best_fixed_strategy": best_fixed,
            "best_fixed": best_fixed_val,
            "fixed_avg": fixed_avg,
            "pred_vs_no": better_delta(pred, no, metric),
            "pred_vs_best_fixed": better_delta(pred, best_fixed_val, metric),
            "pred_vs_fixed_avg": better_delta(pred, fixed_avg, metric),
            "pred_gap_vs_oracle": better_delta(oracle, pred, metric),
        })
    return pd.DataFrame(rows)


# =========================
# Audits
# =========================

def equivalence_audit(selected: pd.DataFrame, policy: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    mis = selected[selected[PRED_ACTION_COL] != selected[ORACLE_ACTION_COL]].copy()
    if mis.empty:
        return pd.DataFrame(columns=KEY_COLS + ["condition_family", PRED_ACTION_COL, ORACLE_ACTION_COL])
    pred_rows = replay_actions(mis, policy, mis[PRED_ACTION_COL], "pred", metric_cols)
    oracle_rows = replay_actions(mis, policy, mis[ORACLE_ACTION_COL], "oracle", metric_cols)
    keep_cols = KEY_COLS + [ACTION_COL, "_matched"] + metric_cols
    merged = pred_rows[keep_cols].merge(
        oracle_rows[keep_cols],
        on=KEY_COLS,
        suffixes=("_pred", "_oracle"),
        how="outer",
    )
    merged = merged.merge(mis[KEY_COLS + [PRED_ACTION_COL, ORACLE_ACTION_COL]], on=KEY_COLS, how="left")
    merged["condition_family"] = merged["condition"].map(family_from_condition)
    for metric in metric_cols:
        merged[f"{metric}_pred_minus_oracle_raw"] = merged[f"{metric}_pred"] - merged[f"{metric}_oracle"]
        merged[f"{metric}_pred_vs_oracle_better_delta"] = [
            better_delta(a, b, metric) for a, b in zip(merged[f"{metric}_pred"], merged[f"{metric}_oracle"])
        ]
    return merged


def condition_contribution(replay: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    rows = []
    for metric in metric_cols:
        best_fixed, _ = best_fixed_strategy(replay, metric)
        for cond, sub in replay.groupby("condition", sort=True):
            pred = metric_strategy_mean(sub, metric, "predicted_policy")
            oracle = metric_strategy_mean(sub, metric, "oracle_policy")
            no = metric_strategy_mean(sub, metric, "fixed__NO_INTERVENTION") if "fixed__NO_INTERVENTION" in sub["strategy"].unique() else float("nan")
            best_fixed_val = metric_strategy_mean(sub, metric, best_fixed) if best_fixed else float("nan")
            rows.append({
                "condition": cond,
                "condition_family": family_from_condition(cond),
                "metric": metric,
                "n_keys": int(sub[KEY_COLS].drop_duplicates().shape[0]),
                "predicted": pred,
                "oracle": oracle,
                "no_intervention": no,
                "best_fixed_strategy_global": best_fixed,
                "best_fixed_global_value": best_fixed_val,
                "pred_vs_no": better_delta(pred, no, metric),
                "pred_vs_best_fixed": better_delta(pred, best_fixed_val, metric),
                "pred_gap_vs_oracle": better_delta(oracle, pred, metric),
            })
    return pd.DataFrame(rows)


def family_contribution(cond_df: pd.DataFrame) -> pd.DataFrame:
    if cond_df.empty:
        return cond_df
    numeric_cols = [
        "n_keys", "predicted", "oracle", "no_intervention", "best_fixed_global_value",
        "pred_vs_no", "pred_vs_best_fixed", "pred_gap_vs_oracle",
    ]
    out = (
        cond_df.groupby(["condition_family", "metric"], as_index=False)[numeric_cols]
        .mean(numeric_only=True)
    )
    return out


def leave_one_condition_out(replay: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    rows = []
    conditions = sorted(replay["condition"].unique())
    for metric in metric_cols:
        for cond in conditions:
            sub = replay[replay["condition"] != cond]
            base = base_metric_summary(sub, [metric]).iloc[0].to_dict()
            base["left_out_condition"] = cond
            base["left_out_family"] = family_from_condition(cond)
            rows.append(base)
    return pd.DataFrame(rows)


def bootstrap_ci(replay: pd.DataFrame, metric_cols: List[str], n_boot: int = BOOT_N, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    keys = replay[KEY_COLS].drop_duplicates().reset_index(drop=True)
    key_tuples = list(map(tuple, keys[KEY_COLS].values.tolist()))
    key_index = pd.MultiIndex.from_frame(replay[KEY_COLS])
    rows = []
    for metric in metric_cols:
        best_fixed, _ = best_fixed_strategy(replay, metric)
        boot_pred_vs_no = []
        boot_pred_vs_best = []
        boot_gap_oracle = []
        boot_pred = []
        boot_oracle = []
        boot_no = []
        boot_best = []
        for _ in range(n_boot):
            sample_idx = rng.integers(0, len(key_tuples), size=len(key_tuples))
            sample_keys = [key_tuples[i] for i in sample_idx]
            # Build sampled frame by concatenating matching rows. Small table, simple method is fine.
            parts = []
            for g, c in sample_keys:
                parts.append(replay[(replay["graph_id"] == g) & (replay["condition"] == c)])
            sub = pd.concat(parts, ignore_index=True)
            pred = metric_strategy_mean(sub, metric, "predicted_policy")
            oracle = metric_strategy_mean(sub, metric, "oracle_policy")
            no = metric_strategy_mean(sub, metric, "fixed__NO_INTERVENTION") if "fixed__NO_INTERVENTION" in sub["strategy"].unique() else float("nan")
            best = metric_strategy_mean(sub, metric, best_fixed) if best_fixed else float("nan")
            boot_pred.append(pred)
            boot_oracle.append(oracle)
            boot_no.append(no)
            boot_best.append(best)
            boot_pred_vs_no.append(better_delta(pred, no, metric))
            boot_pred_vs_best.append(better_delta(pred, best, metric))
            boot_gap_oracle.append(better_delta(oracle, pred, metric))
        for name, vals in [
            ("predicted", boot_pred),
            ("oracle", boot_oracle),
            ("no_intervention", boot_no),
            ("best_fixed", boot_best),
            ("pred_vs_no", boot_pred_vs_no),
            ("pred_vs_best_fixed", boot_pred_vs_best),
            ("pred_gap_vs_oracle", boot_gap_oracle),
        ]:
            arr = np.array(vals, dtype=float)
            rows.append({
                "metric": metric,
                "quantity": name,
                "mean": float(np.nanmean(arr)),
                "std": float(np.nanstd(arr, ddof=1)),
                "q025": float(np.nanquantile(arr, 0.025)),
                "q050": float(np.nanquantile(arr, 0.500)),
                "q975": float(np.nanquantile(arr, 0.975)),
                "positive_rate": float(np.nanmean(arr > 0)) if name.startswith("pred_vs") else float("nan"),
                "best_fixed_strategy": best_fixed,
            })
    return pd.DataFrame(rows)


def evaluate_action_vector(selected: pd.DataFrame, policy: pd.DataFrame, metric_cols: List[str], action_vec: pd.Series) -> Dict[str, float]:
    replay = replay_actions(selected, policy, action_vec, "stress_policy", metric_cols)
    return {m: float(pd.to_numeric(replay[m], errors="coerce").mean()) for m in metric_cols}


def noise_stress(selected: pd.DataFrame, policy: pd.DataFrame, metric_cols: List[str], candidate_actions: List[str]) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED + 17)
    base_actions = selected[PRED_ACTION_COL].astype(str).reset_index(drop=True)
    selected0 = selected.reset_index(drop=True)
    rows = []
    # Baselines for comparison
    base_replay = build_replay_table(selected, policy, [a for a in candidate_actions if a != ""], metric_cols)
    base_summary = base_metric_summary(base_replay, metric_cols)
    baseline_by_metric = {r["metric"]: r for _, r in base_summary.iterrows()}
    for rate in NOISE_RATES:
        for it in range(STRESS_N):
            actions = base_actions.copy()
            flip_mask = rng.random(len(actions)) < rate
            for idx in np.where(flip_mask)[0]:
                choices = [a for a in candidate_actions if a != actions.iloc[idx]]
                if choices:
                    actions.iloc[idx] = rng.choice(choices)
            vals = evaluate_action_vector(selected0, policy, metric_cols, actions)
            actual_flip_rate = float((actions != base_actions).mean())
            for metric in metric_cols:
                pred_val = vals[metric]
                no = float(baseline_by_metric[metric]["no_intervention"])
                best = float(baseline_by_metric[metric]["best_fixed"])
                oracle = float(baseline_by_metric[metric]["oracle"])
                rows.append({
                    "stress_type": "random_action_flip",
                    "target_flip_rate": rate,
                    "iteration": it,
                    "actual_flip_rate": actual_flip_rate,
                    "metric": metric,
                    "metric_value": pred_val,
                    "delta_vs_no_intervention": better_delta(pred_val, no, metric),
                    "delta_vs_best_fixed": better_delta(pred_val, best, metric),
                    "gap_vs_oracle": better_delta(oracle, pred_val, metric),
                })
    raw = pd.DataFrame(rows)
    return aggregate_stress(raw, ["stress_type", "target_flip_rate", "metric"])


def dropout_stress(selected: pd.DataFrame, policy: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED + 23)
    base_actions = selected[PRED_ACTION_COL].astype(str).reset_index(drop=True)
    selected0 = selected.reset_index(drop=True)
    rows = []
    base_replay = build_replay_table(selected, policy, FIXED_ACTIONS_DEFAULT, metric_cols)
    base_summary = base_metric_summary(base_replay, metric_cols)
    baseline_by_metric = {r["metric"]: r for _, r in base_summary.iterrows()}
    for rate in DROPOUT_RATES:
        for it in range(STRESS_N):
            actions = base_actions.copy()
            mask = rng.random(len(actions)) < rate
            actions.loc[mask] = "NO_INTERVENTION"
            vals = evaluate_action_vector(selected0, policy, metric_cols, actions)
            actual_dropout_rate = float(mask.mean())
            for metric in metric_cols:
                pred_val = vals[metric]
                no = float(baseline_by_metric[metric]["no_intervention"])
                best = float(baseline_by_metric[metric]["best_fixed"])
                oracle = float(baseline_by_metric[metric]["oracle"])
                rows.append({
                    "stress_type": "drop_to_no_intervention",
                    "target_dropout_rate": rate,
                    "iteration": it,
                    "actual_dropout_rate": actual_dropout_rate,
                    "metric": metric,
                    "metric_value": pred_val,
                    "delta_vs_no_intervention": better_delta(pred_val, no, metric),
                    "delta_vs_best_fixed": better_delta(pred_val, best, metric),
                    "gap_vs_oracle": better_delta(oracle, pred_val, metric),
                })
    raw = pd.DataFrame(rows)
    return aggregate_stress(raw, ["stress_type", "target_dropout_rate", "metric"])


def aggregate_stress(raw: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in raw.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for q in ["metric_value", "delta_vs_no_intervention", "delta_vs_best_fixed", "gap_vs_oracle"]:
            arr = sub[q].astype(float).to_numpy()
            row[f"{q}_mean"] = float(np.nanmean(arr))
            row[f"{q}_std"] = float(np.nanstd(arr, ddof=1))
            row[f"{q}_q025"] = float(np.nanquantile(arr, 0.025))
            row[f"{q}_q500"] = float(np.nanquantile(arr, 0.500))
            row[f"{q}_q975"] = float(np.nanquantile(arr, 0.975))
        row["pass_vs_no_rate"] = float(np.nanmean(sub["delta_vs_no_intervention"].to_numpy() > 0))
        row["pass_vs_best_fixed_rate"] = float(np.nanmean(sub["delta_vs_best_fixed"].to_numpy() > 0))
        rows.append(row)
    return pd.DataFrame(rows)


def derive_overall_verdict(base: pd.DataFrame, boot: pd.DataFrame, noise: pd.DataFrame, metric: str = "pred_clean") -> Dict[str, object]:
    metric_base = base[base["metric"] == metric].iloc[0].to_dict()
    boot_no = boot[(boot["metric"] == metric) & (boot["quantity"] == "pred_vs_no")]
    boot_best = boot[(boot["metric"] == metric) & (boot["quantity"] == "pred_vs_best_fixed")]
    ci_no_low = float(boot_no["q025"].iloc[0]) if len(boot_no) else float("nan")
    ci_best_low = float(boot_best["q025"].iloc[0]) if len(boot_best) else float("nan")
    # noise threshold: largest flip rate with q025 delta_vs_no > 0
    nsub = noise[(noise["metric"] == metric) & (noise["stress_type"] == "random_action_flip")].copy()
    robust_rates_no = nsub[nsub["delta_vs_no_intervention_q025"] > 0]["target_flip_rate"].tolist() if len(nsub) else []
    robust_rates_best = nsub[nsub["delta_vs_best_fixed_q025"] > 0]["target_flip_rate"].tolist() if len(nsub) else []
    noise_threshold_no = max(robust_rates_no) if robust_rates_no else 0.0
    noise_threshold_best = max(robust_rates_best) if robust_rates_best else 0.0

    if metric_base["pred_gap_vs_oracle"] <= 1e-9 and ci_no_low > 0 and ci_best_low > 0 and noise_threshold_no >= 0.10:
        verdict = "PASS_STRONG_ROBUST_REPLAY"
    elif metric_base["pred_gap_vs_oracle"] <= 1e-9 and ci_no_low > 0 and metric_base["pred_vs_best_fixed"] > 0:
        verdict = "PASS_ROBUST_VS_NO_INTERVENTION_BEST_FIXED_MARGIN_SMALL"
    elif ci_no_low > 0:
        verdict = "PASS_LITE_ROBUST_VS_NO_INTERVENTION"
    else:
        verdict = "WEAK_OR_FAIL_ROBUSTNESS"
    return {
        "primary_metric": metric,
        "verdict": verdict,
        "base_pred_vs_no": metric_base["pred_vs_no"],
        "base_pred_vs_best_fixed": metric_base["pred_vs_best_fixed"],
        "base_pred_gap_vs_oracle": metric_base["pred_gap_vs_oracle"],
        "bootstrap_pred_vs_no_q025": ci_no_low,
        "bootstrap_pred_vs_best_fixed_q025": ci_best_low,
        "noise_flip_threshold_vs_no_q025_positive": noise_threshold_no,
        "noise_flip_threshold_vs_best_fixed_q025_positive": noise_threshold_best,
    }


# =========================
# Main
# =========================

def main() -> None:
    selected = read_csv(SELECTED_CANDIDATES, required=True)
    policy = read_csv(POLICY_CANDIDATES, required=True)
    replay_existing = read_csv(REPLAY_CANDIDATES, required=False)
    metric_verdicts = read_csv(VERDICT_CANDIDATES, required=False)
    prev_diag = read_json(DIAG_CANDIDATES, required=False)

    selected = normalize_string_cols(selected, KEY_COLS + [PRED_ACTION_COL, ORACLE_ACTION_COL])
    policy = normalize_string_cols(policy, KEY_COLS + [ACTION_COL])

    metric_cols = get_metric_cols(policy)
    if not metric_cols:
        raise RuntimeError("No valid outcome metrics found. Check policy outcomes table.")
    print(f"[INFO] metric_cols={metric_cols}")

    # Candidate action universe includes fixed actions and all observed selected/oracle actions.
    candidate_actions = sorted(set(FIXED_ACTIONS_DEFAULT) | set(policy[ACTION_COL].dropna().astype(str).unique()))
    fixed_actions = [a for a in FIXED_ACTIONS_DEFAULT if a in candidate_actions]
    missing_fixed = [a for a in FIXED_ACTIONS_DEFAULT if a not in candidate_actions]
    if missing_fixed:
        print(f"[WARN] fixed actions missing from policy table: {missing_fixed}")

    replay = build_replay_table(selected, policy, fixed_actions, metric_cols)
    summary_all = summarize_replay(replay, metric_cols)
    base = base_metric_summary(replay, metric_cols)
    equiv = equivalence_audit(selected, policy, metric_cols)
    cond = condition_contribution(replay, metric_cols)
    fam = family_contribution(cond)
    loco = leave_one_condition_out(replay, metric_cols)
    boot = bootstrap_ci(replay, metric_cols, n_boot=BOOT_N, seed=RNG_SEED)
    noise = noise_stress(selected, policy, metric_cols, candidate_actions=fixed_actions)
    dropout = dropout_stress(selected, policy, metric_cols)

    primary_metric = "pred_clean" if "pred_clean" in metric_cols else metric_cols[0]
    overall = derive_overall_verdict(base, boot, noise, metric=primary_metric)

    pred_acc = float((selected[PRED_ACTION_COL] == selected[ORACLE_ACTION_COL]).mean())
    n_mis = int((selected[PRED_ACTION_COL] != selected[ORACLE_ACTION_COL]).sum())

    diagnostics = {
        "stage": "DA-ASA-2D.3",
        "purpose": "Policy replay robustness / stress audit",
        "overall_verdict": overall,
        "selected_rows": int(len(selected)),
        "selected_prediction_accuracy": pred_acc,
        "selected_misclassified_rows": n_mis,
        "policy_rows": int(len(policy)),
        "replay_rows": int(len(replay)),
        "metric_cols": metric_cols,
        "fixed_actions_used": fixed_actions,
        "candidate_actions_for_noise_stress": fixed_actions,
        "bootstrap_n": BOOT_N,
        "stress_n": STRESS_N,
        "noise_rates": NOISE_RATES,
        "dropout_rates": DROPOUT_RATES,
        "previous_diagnostics_loaded": prev_diag is not None,
        "previous_metric_verdicts_loaded": metric_verdicts is not None,
        "existing_replay_loaded": replay_existing is not None,
    }

    replay.to_csv(OUT_DIR / "da_asa2d3_replay_rebuilt.csv", index=False, encoding="utf-8-sig")
    summary_all.to_csv(OUT_DIR / "da_asa2d3_replay_summary_all_metrics.csv", index=False, encoding="utf-8-sig")
    base.to_csv(OUT_DIR / "da_asa2d3_base_metric_summary.csv", index=False, encoding="utf-8-sig")
    equiv.to_csv(OUT_DIR / "da_asa2d3_equivalence_audit.csv", index=False, encoding="utf-8-sig")
    cond.to_csv(OUT_DIR / "da_asa2d3_condition_contribution.csv", index=False, encoding="utf-8-sig")
    fam.to_csv(OUT_DIR / "da_asa2d3_family_contribution.csv", index=False, encoding="utf-8-sig")
    loco.to_csv(OUT_DIR / "da_asa2d3_leave_one_condition_out.csv", index=False, encoding="utf-8-sig")
    boot.to_csv(OUT_DIR / "da_asa2d3_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    noise.to_csv(OUT_DIR / "da_asa2d3_noise_stress.csv", index=False, encoding="utf-8-sig")
    dropout.to_csv(OUT_DIR / "da_asa2d3_dropout_stress.csv", index=False, encoding="utf-8-sig")
    with open(OUT_DIR / "da_asa2d3_robustness_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 88)
    print("DA-ASA-2D.3 POLICY REPLAY ROBUSTNESS AUDIT")
    print("=" * 88)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print("\n[BASE METRIC SUMMARY]")
    print(base.to_string(index=False))
    print(f"\n[OUT] {OUT_DIR}")


if __name__ == "__main__":
    main()
