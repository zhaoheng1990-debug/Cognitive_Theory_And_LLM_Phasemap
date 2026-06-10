# -*- coding: utf-8 -*-
"""
DA-ASA-2E: Policy Equivalence Compression Audit

Purpose
-------
2D.3 showed that raw PolicyClass accuracy can understate replay utility:
some predicted actions differ from oracle actions but are outcome-equivalent.

This audit builds condition-specific OutcomeEquivalenceClass labels from
existing table-replay outcomes, then evaluates whether predicted policy is
oracle-equivalent after compression.

No model forward pass. No hidden-state computation. Table replay only.

Default inputs searched under the script directory:
  - da_asa2d2_v2_policy_outcomes_aggregated.csv
  - da_asa2d2_v2_selected_predictions.csv
  - da_asa2d3_replay_rebuilt.csv              optional
  - da_asa2d3_base_metric_summary.csv         optional

Outputs:
  da_asa2e_outputs/
    da_asa2e_diagnostics.json
    da_asa2e_action_outcome_signatures.csv
    da_asa2e_equivalence_classes_long.csv
    da_asa2e_condition_compression_summary.csv
    da_asa2e_prediction_outcome_equivalence.csv
    da_asa2e_replay_by_outcome_class.csv
    da_asa2e_metric_summary.csv
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "da_asa2e_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POLICY_OUTCOMES_CANDIDATES = [
    "da_asa2d2_v2_policy_outcomes_aggregated.csv",
    "da_asa2d2_policy_outcomes_aggregated.csv",
]
SELECTED_PRED_CANDIDATES = [
    "da_asa2d2_v2_selected_predictions.csv",
    "da_asa2d2_selected_predictions.csv",
]
REPLAY_CANDIDATES = [
    "da_asa2d3_replay_rebuilt.csv",
    "da_asa2d2_v2_replay_rows.csv",
]

KEY_COLS = ["graph_id", "condition"]
ACTION_COL = "action_class"
METRIC_COLS_PREF = ["pred_clean", "R_final"]

# Equivalence tolerance. pred_clean is usually exact 0/1 or mean; R_final may have small float noise.
METRIC_ROUND_DECIMALS = {
    "pred_clean": 8,
    "R_final": 6,
}

FIXED_ACTIONS = [
    "NO_INTERVENTION",
    "CORE_CLOSURE_UPDATE",
    "OVERRIDE_THREE_STAGE",
    "EQUAL_EVIDENCE_ORDER",
]


def find_file(candidates: Iterable[str], required: bool = True) -> Optional[Path]:
    for name in candidates:
        p = ROOT / name
        if p.exists():
            return p
        hits = list(ROOT.rglob(name))
        if hits:
            return hits[0]
    if required:
        raise FileNotFoundError(f"Could not find any of: {list(candidates)} under {ROOT}")
    return None


def read_csv(candidates: Iterable[str], required: bool = True) -> Optional[pd.DataFrame]:
    p = find_file(candidates, required=required)
    if p is None:
        return None
    print(f"[LOAD] {p}")
    return pd.read_csv(p)


def normalize_str_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


def infer_condition_family(condition: str) -> str:
    s = str(condition).lower()
    if s.startswith("stable") or s in {"clean", "baseline"}:
        return "stable"
    if "closure" in s or "override" in s or "exception" in s or "update" in s or "temporal" in s or "authority" in s:
        return "closure"
    if "competition" in s or "equal" in s or "source" in s or "ambiguous" in s or "branch" in s or "direct" in s:
        return "competition"
    if "hallucination" in s:
        return "hallucination_like"
    if "weak" in s or "distractor" in s or "irrelevant" in s or "redundant" in s:
        return "other"
    return "other"


def metric_cols_available(df: pd.DataFrame) -> List[str]:
    cols = [c for c in METRIC_COLS_PREF if c in df.columns]
    if not cols:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        cols = [c for c in numeric if c not in {"row_index"}]
    if not cols:
        raise ValueError("No usable metric columns found in policy outcomes.")
    return cols


def add_signature_columns(policy: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    df = policy.copy()
    sig_parts = []
    for c in metric_cols:
        rounded = pd.to_numeric(df[c], errors="coerce").round(METRIC_ROUND_DECIMALS.get(c, 6))
        sig_col = f"sig__{c}"
        df[sig_col] = rounded
        sig_parts.append(sig_col)
    df["outcome_signature"] = df[sig_parts].astype(str).agg("|".join, axis=1)
    return df


def build_condition_action_signatures(policy: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    """Represent each action by its full graph-level outcome vector inside a condition."""
    sig_policy = add_signature_columns(policy, metric_cols)

    rows = []
    for (condition, action), sub in sig_policy.groupby(["condition", ACTION_COL], sort=True):
        sub = sub.sort_values("graph_id")
        graph_outcomes = [f"{g}:{sig}" for g, sig in zip(sub["graph_id"].astype(str), sub["outcome_signature"].astype(str))]
        row = {
            "condition": condition,
            "condition_family": infer_condition_family(condition),
            ACTION_COL: action,
            "n_graphs": int(sub["graph_id"].nunique()),
            "vector_signature": "||".join(graph_outcomes),
        }
        for c in metric_cols:
            vals = pd.to_numeric(sub[c], errors="coerce")
            row[f"mean_{c}"] = float(vals.mean()) if vals.notna().any() else np.nan
            row[f"std_{c}"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def assign_equivalence_classes(action_sig: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    """Within each condition, actions with identical full graph-outcome vector share a class."""
    out_rows = []
    for condition, sub in action_sig.groupby("condition", sort=True):
        sub = sub.copy().sort_values(["vector_signature", ACTION_COL])
        sig_to_class: Dict[str, str] = {}
        class_index = 0
        for sig in sub["vector_signature"].tolist():
            if sig not in sig_to_class:
                sig_to_class[sig] = f"{condition}__OE{class_index:02d}"
                class_index += 1
        sub["outcome_class"] = sub["vector_signature"].map(sig_to_class)
        out_rows.append(sub)
    out = pd.concat(out_rows, ignore_index=True)

    # Representative is chosen by priority: fixed actions in a stable human-readable order, then lexical.
    priority = {a: i for i, a in enumerate(FIXED_ACTIONS)}
    rep_map = {}
    for cls, sub in out.groupby("outcome_class"):
        ss = sub.copy()
        ss["_prio"] = ss[ACTION_COL].map(priority).fillna(999).astype(int)
        ss = ss.sort_values(["_prio", ACTION_COL])
        rep_map[cls] = ss[ACTION_COL].iloc[0]
    out["outcome_class_representative"] = out["outcome_class"].map(rep_map)
    return out


def summarize_compression(eq_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, sub in eq_long.groupby("condition", sort=True):
        n_actions = int(sub[ACTION_COL].nunique())
        n_classes = int(sub["outcome_class"].nunique())
        class_sizes = sub.groupby("outcome_class")[ACTION_COL].nunique().sort_values(ascending=False).tolist()
        rows.append({
            "condition": condition,
            "condition_family": infer_condition_family(condition),
            "n_actions": n_actions,
            "n_outcome_classes": n_classes,
            "compression_ratio_actions_per_class": n_actions / max(n_classes, 1),
            "max_class_size": int(max(class_sizes)) if class_sizes else 0,
            "class_sizes": ";".join(map(str, class_sizes)),
        })
    return pd.DataFrame(rows)


def map_predictions_to_equiv(selected: pd.DataFrame, eq_long: pd.DataFrame) -> pd.DataFrame:
    selected = selected.copy()
    selected = normalize_str_cols(selected, KEY_COLS + ["pred_action", "oracle_action"])

    mapper = eq_long[["condition", ACTION_COL, "outcome_class", "outcome_class_representative"]].drop_duplicates()

    pred_map = mapper.rename(columns={
        ACTION_COL: "pred_action",
        "outcome_class": "pred_outcome_class",
        "outcome_class_representative": "pred_outcome_class_representative",
    })
    oracle_map = mapper.rename(columns={
        ACTION_COL: "oracle_action",
        "outcome_class": "oracle_outcome_class",
        "outcome_class_representative": "oracle_outcome_class_representative",
    })

    out = selected.merge(pred_map, on=["condition", "pred_action"], how="left")
    out = out.merge(oracle_map, on=["condition", "oracle_action"], how="left")
    out["raw_action_correct"] = out["pred_action"] == out["oracle_action"]
    out["outcome_class_correct"] = out["pred_outcome_class"] == out["oracle_outcome_class"]
    out["condition_family"] = out["condition"].map(infer_condition_family)
    out["was_raw_misclassified_but_outcome_equivalent"] = (~out["raw_action_correct"]) & out["outcome_class_correct"]
    return out


def replay_by_outcome_class(policy: pd.DataFrame, pred_equiv: pd.DataFrame, eq_long: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    """Replay using the representative action of the predicted/oracle outcome class."""
    eq_min = eq_long[["condition", "outcome_class", "outcome_class_representative"]].drop_duplicates()

    rows = []
    for strategy, class_col in [
        ("predicted_outcome_class", "pred_outcome_class"),
        ("oracle_outcome_class", "oracle_outcome_class"),
    ]:
        tmp = pred_equiv[KEY_COLS + [class_col]].copy().rename(columns={class_col: "outcome_class"})
        tmp = tmp.merge(eq_min, on=["condition", "outcome_class"], how="left")
        tmp = tmp.rename(columns={"outcome_class_representative": ACTION_COL})
        tmp["strategy"] = strategy
        rows.append(tmp)

    for action in FIXED_ACTIONS:
        tmp = pred_equiv[KEY_COLS].copy()
        tmp[ACTION_COL] = action
        tmp["strategy"] = f"fixed__{action}"
        rows.append(tmp)

    left = pd.concat(rows, ignore_index=True)
    replay = left.merge(policy[KEY_COLS + [ACTION_COL] + metric_cols], on=KEY_COLS + [ACTION_COL], how="left", validate="many_to_one")
    replay["_matched"] = replay[metric_cols].notna().any(axis=1)
    return replay


def summarize_metrics(replay: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    rows = []
    for metric in metric_cols:
        for strategy, sub in replay.groupby("strategy", sort=True):
            vals = pd.to_numeric(sub[metric], errors="coerce")
            rows.append({
                "metric": metric,
                "strategy": strategy,
                "n": int(len(sub)),
                "matched": int(sub["_matched"].sum()),
                "missing": int((~sub["_matched"]).sum()),
                "mean": float(vals.mean()) if vals.notna().any() else np.nan,
                "std": float(vals.std(ddof=1)) if vals.notna().sum() > 1 else np.nan,
            })
    summary = pd.DataFrame(rows)

    # Attach deltas for each metric.
    out = []
    for metric, sub in summary.groupby("metric", sort=False):
        sub = sub.copy()
        def get(strategy: str) -> float:
            s = sub.loc[sub["strategy"] == strategy, "mean"]
            return float(s.iloc[0]) if len(s) else np.nan
        pred = get("predicted_outcome_class")
        oracle = get("oracle_outcome_class")
        no = get("fixed__NO_INTERVENTION")
        fixed = sub[sub["strategy"].str.startswith("fixed__", na=False)]
        best_fixed = float(fixed["mean"].max()) if len(fixed) else np.nan
        sub["predicted_outcome_class_mean"] = pred
        sub["oracle_outcome_class_mean"] = oracle
        sub["no_intervention_mean"] = no
        sub["best_fixed_mean"] = best_fixed
        sub["pred_vs_no"] = pred - no
        sub["pred_vs_best_fixed"] = pred - best_fixed
        sub["pred_gap_vs_oracle"] = oracle - pred
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    policy = read_csv(POLICY_OUTCOMES_CANDIDATES, required=True)
    selected = read_csv(SELECTED_PRED_CANDIDATES, required=True)
    _ = read_csv(REPLAY_CANDIDATES, required=False)  # loaded only to make path visible in log; not required.

    policy = normalize_str_cols(policy, KEY_COLS + [ACTION_COL])
    selected = normalize_str_cols(selected, KEY_COLS + ["pred_action", "oracle_action"])

    metric_cols = metric_cols_available(policy)
    print(f"[INFO] metric_cols={metric_cols}")

    action_sig = build_condition_action_signatures(policy, metric_cols)
    eq_long = assign_equivalence_classes(action_sig, metric_cols)
    cond_summary = summarize_compression(eq_long)
    pred_equiv = map_predictions_to_equiv(selected, eq_long)
    replay = replay_by_outcome_class(policy, pred_equiv, eq_long, metric_cols)
    metric_summary = summarize_metrics(replay, metric_cols)

    raw_acc = float(pred_equiv["raw_action_correct"].mean())
    outcome_acc = float(pred_equiv["outcome_class_correct"].mean())
    rescued = int(pred_equiv["was_raw_misclassified_but_outcome_equivalent"].sum())
    raw_wrong = int((~pred_equiv["raw_action_correct"]).sum())

    # Verdict rules.
    pred_clean_row = metric_summary[(metric_summary["metric"] == "pred_clean") & (metric_summary["strategy"] == "predicted_outcome_class")]
    if len(pred_clean_row):
        pred_vs_no = float(pred_clean_row["pred_vs_no"].iloc[0])
        pred_vs_best = float(pred_clean_row["pred_vs_best_fixed"].iloc[0])
        gap = float(pred_clean_row["pred_gap_vs_oracle"].iloc[0])
    else:
        pred_vs_no = pred_vs_best = gap = np.nan

    if outcome_acc == 1.0 and raw_acc < outcome_acc and pred_vs_no > 0 and gap == 0:
        verdict = "PASS_OUTCOME_EQUIVALENCE_COMPRESSION_EXPLAINS_REPLAY"
    elif outcome_acc > raw_acc and pred_vs_no > 0:
        verdict = "PASS_LITE_OUTCOME_EQUIVALENCE_IMPROVES_POLICY_LABELS"
    else:
        verdict = "CHECK_EQUIVALENCE_COMPRESSION"

    diagnostics = {
        "stage": "DA-ASA-2E",
        "purpose": "Policy outcome equivalence compression",
        "verdict": verdict,
        "n_selected_rows": int(len(pred_equiv)),
        "raw_action_accuracy": raw_acc,
        "outcome_class_accuracy": outcome_acc,
        "raw_misclassified_rows": raw_wrong,
        "rescued_by_outcome_equivalence_rows": rescued,
        "rescued_fraction_of_raw_errors": rescued / raw_wrong if raw_wrong else 0.0,
        "n_conditions": int(eq_long["condition"].nunique()),
        "n_actions_total_unique": int(eq_long[ACTION_COL].nunique()),
        "mean_actions_per_condition": float(cond_summary["n_actions"].mean()),
        "mean_outcome_classes_per_condition": float(cond_summary["n_outcome_classes"].mean()),
        "mean_compression_ratio_actions_per_class": float(cond_summary["compression_ratio_actions_per_class"].mean()),
        "metric_cols": metric_cols,
        "pred_clean_pred_vs_no": pred_vs_no,
        "pred_clean_pred_vs_best_fixed": pred_vs_best,
        "pred_clean_gap_vs_oracle": gap,
    }

    action_sig.to_csv(OUT_DIR / "da_asa2e_action_outcome_signatures.csv", index=False, encoding="utf-8-sig")
    eq_long.to_csv(OUT_DIR / "da_asa2e_equivalence_classes_long.csv", index=False, encoding="utf-8-sig")
    cond_summary.to_csv(OUT_DIR / "da_asa2e_condition_compression_summary.csv", index=False, encoding="utf-8-sig")
    pred_equiv.to_csv(OUT_DIR / "da_asa2e_prediction_outcome_equivalence.csv", index=False, encoding="utf-8-sig")
    replay.to_csv(OUT_DIR / "da_asa2e_replay_by_outcome_class.csv", index=False, encoding="utf-8-sig")
    metric_summary.to_csv(OUT_DIR / "da_asa2e_metric_summary.csv", index=False, encoding="utf-8-sig")
    with open(OUT_DIR / "da_asa2e_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("DA-ASA-2E POLICY EQUIVALENCE COMPRESSION")
    print("=" * 80)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print("\n[CONDITION COMPRESSION SUMMARY]")
    print(cond_summary.sort_values(["condition_family", "condition"]).to_string(index=False))
    print(f"\n[OUT] {OUT_DIR}")


if __name__ == "__main__":
    main()
