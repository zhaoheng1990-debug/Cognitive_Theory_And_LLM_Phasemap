# -*- coding: utf-8 -*-
"""
DA-ASA-2D.2 Final Table Replay v2

Fixes v1 issue:
- Never uses row_index / index-like columns as outcome metric.
- Reports both pred_clean and R_final when present.
- Keeps replay key fixed as: graph_id + condition + action_class.
- Does not use sample_id / row_index for matching.

Run:
    python da_asa2d2_final_table_replay_v2.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "da_asa2d2_final_replay_v2_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_FILE_CANDIDATES = [
    "da_asa2d1b_predictions.csv",
    "da_asa2d1b_group_predictions.csv",
    "da_asa2d1_predictions.csv",
    "da_asa2d2_selected_predictions.csv",
]

POLICY_FILE_CANDIDATES = [
    "da_asa2a_policy_results.csv",
    "da_asa2b_policy_results.csv",
    "da_asa2c_policy_results.csv",
    "da_asa2d2_policy_outcomes_aggregated.csv",
]

PRED_FILTER = {
    "feature_set": "strict_graph_dsta_only",
    "split": "group_graph",
    "model": "logreg_balanced",
}

KEY_COLS = ["graph_id", "condition"]

FIXED_ACTIONS = [
    "NO_INTERVENTION",
    "CORE_CLOSURE_UPDATE",
    "OVERRIDE_THREE_STAGE",
    "EQUAL_EVIDENCE_ORDER",
]

OUTCOME_PRIORITY_HIGHER_BETTER = [
    "pred_clean",
    "clean_hit_rate",
    "clean_rate",
    "target_clean_rate",
    "correct_hit_rate",
    "correct_rate",
    "Gen_C",
    "gen_clean",
    "gen_correct",
    "success_rate",
    "safety_adjusted_gain",
    "R_final",
    "final_R",
    "margin",
    "score",
    "utility",
]

OUTCOME_PRIORITY_LOWER_BETTER = [
    "error_rate",
    "Gen_E",
    "gen_error",
    "wrong_rate",
    "loss",
]

INDEX_LIKE_COLS = {
    "row_index", "index", "idx", "Unnamed: 0", "level_0",
    "sample_id", "source_row", "original_row",
}


def find_file(name: str, root: Path = ROOT) -> Optional[Path]:
    p = root / name
    if p.exists():
        return p
    hits = list(root.rglob(name))
    return hits[0] if hits else None


def load_first_existing(candidates: List[str]) -> pd.DataFrame:
    for name in candidates:
        p = find_file(name)
        if p is not None:
            print(f"[LOAD] {name} -> {p}")
            return pd.read_csv(p)
    raise FileNotFoundError(f"None found: {candidates}")


def load_many_existing(candidates: List[str]) -> pd.DataFrame:
    dfs = []
    for name in candidates:
        p = find_file(name)
        if p is None:
            print(f"[MISS] {name}")
            continue
        print(f"[LOAD] {name} -> {p}")
        df = pd.read_csv(p)
        df["_source_file"] = name
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"None found: {candidates}")
    return pd.concat(dfs, ignore_index=True, sort=False)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def standardize_key_cols(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = normalize_columns(df)
    graph_col = find_first_col(df, ["graph_id", "graph", "gid"])
    cond_col = find_first_col(df, ["condition", "cond", "condition_name"])
    if graph_col is None or cond_col is None:
        raise KeyError(f"[{name}] missing graph_id/condition columns: {list(df.columns)}")
    if graph_col != "graph_id":
        df = df.rename(columns={graph_col: "graph_id"})
    if cond_col != "condition":
        df = df.rename(columns={cond_col: "condition"})
    df["graph_id"] = df["graph_id"].astype(str).str.strip()
    df["condition"] = df["condition"].astype(str).str.strip()
    return df


def canonical_action(x) -> str:
    if pd.isna(x):
        return "MISSING_ACTION"
    s = str(x).strip()
    u = s.upper()

    aliases = {
        "NONE": "NO_INTERVENTION",
        "NOOP": "NO_INTERVENTION",
        "NO_OP": "NO_INTERVENTION",
        "NO INTERVENTION": "NO_INTERVENTION",
        "NO_INTERVENTION": "NO_INTERVENTION",
        "A0.0_B0.0": "NO_INTERVENTION",
        "A0_B0": "NO_INTERVENTION",

        "CORE": "CORE_CLOSURE_UPDATE",
        "CORE_CLOSURE": "CORE_CLOSURE_UPDATE",
        "CORE_UPDATE": "CORE_CLOSURE_UPDATE",
        "CLOSURE_UPDATE": "CORE_CLOSURE_UPDATE",
        "CORE_CLOSURE_UPDATE": "CORE_CLOSURE_UPDATE",

        "OVERRIDE": "OVERRIDE_THREE_STAGE",
        "OVERRIDE_3_STAGE": "OVERRIDE_THREE_STAGE",
        "THREE_STAGE_OVERRIDE": "OVERRIDE_THREE_STAGE",
        "OVERRIDE_THREE_STAGE": "OVERRIDE_THREE_STAGE",

        "EQUAL": "EQUAL_EVIDENCE_ORDER",
        "EQUAL_EVIDENCE": "EQUAL_EVIDENCE_ORDER",
        "EQUAL_ORDER": "EQUAL_EVIDENCE_ORDER",
        "EQUAL_EVIDENCE_ORDER": "EQUAL_EVIDENCE_ORDER",
    }
    if u in aliases:
        return aliases[u]

    if "NO" in u and "INTERVENTION" in u:
        return "NO_INTERVENTION"
    if "CORE" in u and ("CLOSURE" in u or "UPDATE" in u):
        return "CORE_CLOSURE_UPDATE"
    if "OVERRIDE" in u and ("THREE" in u or "3" in u or "STAGE" in u):
        return "OVERRIDE_THREE_STAGE"
    if "EQUAL" in u and ("EVIDENCE" in u or "ORDER" in u):
        return "EQUAL_EVIDENCE_ORDER"

    return u


def infer_prediction_cols(pred: pd.DataFrame) -> Tuple[str, str]:
    pred_col = find_first_col(pred, [
        "pred", "y_pred", "pred_label", "pred_policy", "pred_policy_class",
        "predicted_policy", "predicted_policy_class", "policy_pred", "pred_action",
    ])
    true_col = find_first_col(pred, [
        "true", "y_true", "label", "true_label", "policy_class",
        "true_policy", "true_policy_class", "oracle_policy", "target_policy", "oracle_action",
    ])
    if pred_col is None or true_col is None:
        raise KeyError(f"Cannot infer pred/true columns: {list(pred.columns)}")
    return pred_col, true_col


def infer_policy_action_col(policy: pd.DataFrame) -> str:
    action_col = find_first_col(policy, [
        "action_class", "canonical_action_class", "policy_class",
        "policy_action", "action", "selected_action", "policy",
    ])
    if action_col is None:
        raise KeyError(f"Cannot infer action column: {list(policy.columns)}")
    return action_col


def valid_outcome_columns(df: pd.DataFrame) -> List[Tuple[str, bool]]:
    out = []
    for c in OUTCOME_PRIORITY_HIGHER_BETTER:
        found = find_first_col(df, [c])
        if found is not None and found not in INDEX_LIKE_COLS and pd.api.types.is_numeric_dtype(df[found]):
            out.append((found, True))
    for c in OUTCOME_PRIORITY_LOWER_BETTER:
        found = find_first_col(df, [c])
        if found is not None and found not in INDEX_LIKE_COLS and pd.api.types.is_numeric_dtype(df[found]):
            out.append((found, False))

    seen = set(x[0] for x in out)
    for c in df.select_dtypes(include=[np.number]).columns:
        if c in seen or c in INDEX_LIKE_COLS or c.startswith("_"):
            continue
        if c in KEY_COLS:
            continue
        # Keep fallback numeric columns, but only after preferred outcome columns.
        out.append((c, True))
    return out


def replay_strategy(selected: pd.DataFrame, policy: pd.DataFrame, strategy: str, actions: pd.Series) -> pd.DataFrame:
    left = selected[KEY_COLS].copy()
    left["strategy"] = strategy
    left["action_class"] = actions.values
    merged = left.merge(policy, on=KEY_COLS + ["action_class"], how="left", validate="many_to_one")
    metric_cols = [c for c in policy.columns if c not in KEY_COLS + ["action_class"]]
    merged["_matched"] = merged[metric_cols].notna().any(axis=1)
    return merged


def summarize_metric(replay: pd.DataFrame, metric: str, higher_is_better: bool) -> pd.DataFrame:
    rows = []
    for strategy, sub in replay.groupby("strategy"):
        x = pd.to_numeric(sub[metric], errors="coerce")
        rows.append({
            "strategy": strategy,
            "n": int(len(sub)),
            "matched": int(sub["_matched"].sum()),
            "missing": int((~sub["_matched"]).sum()),
            "missing_action_rate": 1.0 - float(sub["_matched"].mean()),
            "metric": metric,
            "higher_is_better": higher_is_better,
            "metric_mean": float(x.mean()) if x.notna().any() else np.nan,
            "metric_std": float(x.std(ddof=1)) if x.notna().sum() > 1 else np.nan,
        })
    summary = pd.DataFrame(rows)

    def val(name: str) -> float:
        v = summary.loc[summary["strategy"] == name, "metric_mean"]
        return float(v.iloc[0]) if len(v) else np.nan

    pred = val("predicted_policy")
    oracle = val("oracle_policy")
    no = val("fixed__NO_INTERVENTION")

    if higher_is_better:
        summary["delta_vs_no_intervention"] = summary["metric_mean"] - no
        summary["gap_vs_oracle"] = oracle - summary["metric_mean"]
    else:
        summary["delta_vs_no_intervention"] = no - summary["metric_mean"]
        summary["gap_vs_oracle"] = summary["metric_mean"] - oracle

    return summary


def verdict_from_summary(summary: pd.DataFrame) -> dict:
    pred = summary[summary["strategy"] == "predicted_policy"].iloc[0]
    oracle = summary[summary["strategy"] == "oracle_policy"].iloc[0]
    fixed = summary[summary["strategy"].str.startswith("fixed__")].copy()

    pred_missing = float(pred["missing_action_rate"])
    oracle_missing = float(oracle["missing_action_rate"])
    fixed_missing = float(fixed["missing_action_rate"].max())

    pred_delta = float(pred["delta_vs_no_intervention"])
    pred_gap = float(pred["gap_vs_oracle"])
    pred_mean = float(pred["metric_mean"])
    oracle_mean = float(oracle["metric_mean"])
    fixed_avg = float(fixed["metric_mean"].mean())

    if bool(pred["higher_is_better"]):
        fixed_best = float(fixed["metric_mean"].max())
        pred_vs_fixed_avg = pred_mean - fixed_avg
        pred_vs_fixed_best = pred_mean - fixed_best
    else:
        fixed_best = float(fixed["metric_mean"].min())
        pred_vs_fixed_avg = fixed_avg - pred_mean
        pred_vs_fixed_best = fixed_best - pred_mean

    if pred_missing > 0 or oracle_missing > 0:
        verdict = "FAIL_REPLAY_MISSING_PRED_OR_ORACLE"
    elif fixed_missing > 0:
        verdict = "FAIL_REPLAY_MISSING_FIXED_POLICY"
    elif pred_delta > 0 and abs(pred_gap) <= abs(oracle_mean) * 0.10 + 1e-9:
        verdict = "PASS_PREDICTED_REPLAY_CLOSE_TO_ORACLE"
    elif pred_delta > 0:
        verdict = "PASS_PREDICTED_REPLAY_BEATS_NO_INTERVENTION"
    elif pred_vs_fixed_avg > 0:
        verdict = "PARTIAL_PASS_BEATS_FIXED_AVERAGE_BUT_NOT_NO_INTERVENTION"
    else:
        verdict = "FAIL_PREDICTED_REPLAY_NO_GAIN"

    return {
        "metric": str(pred["metric"]),
        "higher_is_better": bool(pred["higher_is_better"]),
        "verdict": verdict,
        "predicted_missing_action_rate": pred_missing,
        "oracle_missing_action_rate": oracle_missing,
        "fixed_missing_action_rate_max": fixed_missing,
        "predicted_metric": pred_mean,
        "oracle_metric": oracle_mean,
        "no_intervention_metric": float(summary.loc[summary["strategy"] == "fixed__NO_INTERVENTION", "metric_mean"].iloc[0]),
        "predicted_delta_vs_no_intervention": pred_delta,
        "predicted_gap_vs_oracle": pred_gap,
        "fixed_best_metric": fixed_best,
        "fixed_avg_metric": fixed_avg,
        "predicted_vs_fixed_best": pred_vs_fixed_best,
        "predicted_vs_fixed_avg": pred_vs_fixed_avg,
    }


def main() -> None:
    pred = standardize_key_cols(load_first_existing(PRED_FILE_CANDIDATES), "predictions")
    policy_raw = standardize_key_cols(load_many_existing(POLICY_FILE_CANDIDATES), "policy_results")

    selected = pred.copy()
    for col, val in PRED_FILTER.items():
        if col in selected.columns:
            before = len(selected)
            selected = selected[selected[col].astype(str) == str(val)].copy()
            print(f"[FILTER] {col}={val}: {before} -> {len(selected)}")
        else:
            print(f"[WARN] prediction file has no {col}; skip")

    if len(selected) == 0:
        raise ValueError("Prediction selection is empty.")

    pred_col, true_col = infer_prediction_cols(selected)
    selected["pred_action"] = selected[pred_col].map(canonical_action)
    selected["oracle_action"] = selected[true_col].map(canonical_action)
    selected = selected[KEY_COLS + ["pred_action", "oracle_action"] + [
        c for c in ["feature_set", "split", "model"] if c in selected.columns
    ]].drop_duplicates(KEY_COLS)

    prediction_accuracy = float((selected["pred_action"] == selected["oracle_action"]).mean())

    action_col = infer_policy_action_col(policy_raw)
    policy_raw["action_class"] = policy_raw[action_col].map(canonical_action)
    policy_raw = policy_raw[policy_raw["action_class"] != "MISSING_ACTION"].copy()

    numeric_cols = [
        c for c in policy_raw.select_dtypes(include=[np.number]).columns
        if c not in KEY_COLS + ["action_class"] and c != action_col
    ]
    if not numeric_cols:
        raise KeyError("No numeric outcome columns found.")

    policy = policy_raw.groupby(KEY_COLS + ["action_class"], as_index=False).agg({c: "mean" for c in numeric_cols})

    replay_parts = [
        replay_strategy(selected, policy, "predicted_policy", selected["pred_action"]),
        replay_strategy(selected, policy, "oracle_policy", selected["oracle_action"]),
    ]
    for action in FIXED_ACTIONS:
        replay_parts.append(
            replay_strategy(selected, policy, f"fixed__{action}", pd.Series([action] * len(selected), index=selected.index))
        )
    replay = pd.concat(replay_parts, ignore_index=True, sort=False)

    outcome_cols = valid_outcome_columns(replay)
    if not outcome_cols:
        raise KeyError(
            "No valid outcome metric found. "
            "Refusing to use row_index/index-like columns as outcome."
        )

    all_summaries = []
    metric_verdicts = []
    for metric, higher in outcome_cols:
        s = summarize_metric(replay, metric, higher)
        all_summaries.append(s)
        metric_verdicts.append(verdict_from_summary(s))

    summary_all = pd.concat(all_summaries, ignore_index=True, sort=False)
    verdicts = pd.DataFrame(metric_verdicts)

    primary = verdicts.iloc[0].to_dict()
    diagnostics = {
        "primary_verdict": primary,
        "all_metric_verdicts": metric_verdicts,
        "selected_prediction_rows": int(len(selected)),
        "selected_prediction_accuracy": prediction_accuracy,
        "policy_raw_rows": int(len(policy_raw)),
        "policy_aggregated_rows": int(len(policy)),
        "key_cols": KEY_COLS,
        "action_key": "action_class",
        "prediction_filter": PRED_FILTER,
        "fixed_actions": FIXED_ACTIONS,
        "ignored_index_like_cols": sorted(INDEX_LIKE_COLS),
        "outcome_columns_used": [{"metric": m, "higher_is_better": h} for m, h in outcome_cols],
    }

    selected.to_csv(OUT_DIR / "da_asa2d2_v2_selected_predictions.csv", index=False, encoding="utf-8-sig")
    policy.to_csv(OUT_DIR / "da_asa2d2_v2_policy_outcomes_aggregated.csv", index=False, encoding="utf-8-sig")
    replay.to_csv(OUT_DIR / "da_asa2d2_v2_replay_rows.csv", index=False, encoding="utf-8-sig")
    summary_all.to_csv(OUT_DIR / "da_asa2d2_v2_replay_summary_all_metrics.csv", index=False, encoding="utf-8-sig")
    verdicts.to_csv(OUT_DIR / "da_asa2d2_v2_metric_verdicts.csv", index=False, encoding="utf-8-sig")

    with open(OUT_DIR / "da_asa2d2_v2_replay_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("DA-ASA-2D.2 FINAL TABLE REPLAY v2")
    print("=" * 80)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print("\n[METRIC VERDICTS]")
    print(verdicts.to_string(index=False))
    print(f"\n[OUT] {OUT_DIR}")


if __name__ == "__main__":
    main()
