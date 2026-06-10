#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DA-ASA-2D.2 Predicted Policy Control Replay

This script performs replay-by-table: it does not rerun the language model.
It uses existing DA-ASA policy_results tables and CV predictions from 2D.1B.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

POLICY_TO_ID = {
    "NO_INTERVENTION": 0,
    "CORE_CLOSURE_UPDATE": 1,
    "OVERRIDE_THREE_STAGE": 2,
    "EQUAL_EVIDENCE_ORDER": 3,
    "RESERVED_HALLUCINATION_CONTROLLER": 4,
}
ID_TO_POLICY = {v: k for k, v in POLICY_TO_ID.items()}

CONDITION_TO_POLICY = {
    "stable": "NO_INTERVENTION",
    "stable_clean": "NO_INTERVENTION",
    "stable_clean_basic": "NO_INTERVENTION",
    "stable_redundant": "NO_INTERVENTION",
    "stable_weak_note": "NO_INTERVENTION",
    "weak_distractor": "NO_INTERVENTION",
    "competition_direct": "NO_INTERVENTION",
    "competition_source_claim": "NO_INTERVENTION",
    "source_claim": "NO_INTERVENTION",
    "hallucination_like": "NO_INTERVENTION",
    "non_target": "NO_INTERVENTION",
    "nontarget": "NO_INTERVENTION",
    "control": "NO_INTERVENTION",
    "closure_update": "NO_INTERVENTION",
    "closure_temporal": "NO_INTERVENTION",
    "closure_authority": "NO_INTERVENTION",
    "closure_exception": "CORE_CLOSURE_UPDATE",
    "closure_negation": "CORE_CLOSURE_UPDATE",
    "closure_override": "OVERRIDE_THREE_STAGE",
    "competition_equal_evidence": "EQUAL_EVIDENCE_ORDER",
    "equal_evidence": "EQUAL_EVIDENCE_ORDER",
}

PREFERRED_PREDICTION_FILTERS = [
    {"feature_set": "strict_graph_dsta_only", "split": "group_graph", "model": "logreg_balanced"},
    {"feature_set": "all_dsta_no_policy_outcome", "split": "group_graph", "model": "logreg_balanced"},
    {"feature_set": "baseline_plus_all_dsta_no_policy_outcome", "split": "group_graph", "model": "logreg_balanced"},
    {"feature_set": "condition_broadcast_dsta_only", "split": "group_graph", "model": "logreg_balanced"},
]


def read_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def ensure_sample_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "sample_id" not in df.columns:
        if "id" in df.columns:
            df["sample_id"] = df["id"].astype(str)
        elif "row_index" in df.columns:
            df["sample_id"] = df["row_index"].astype(str)
        elif "prompt_id" in df.columns:
            df["sample_id"] = df["prompt_id"].astype(str)
        else:
            df["sample_id"] = np.arange(len(df)).astype(str)
    else:
        df["sample_id"] = df["sample_id"].astype(str)
    if "condition" in df.columns:
        df["condition"] = df["condition"].astype(str)
    return df


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def normalize_policy_label(x) -> str:
    if pd.isna(x):
        return "UNKNOWN"
    if isinstance(x, (int, np.integer)):
        return ID_TO_POLICY.get(int(x), "UNKNOWN")
    if isinstance(x, float) and x.is_integer():
        return ID_TO_POLICY.get(int(x), "UNKNOWN")
    s = str(x).strip()
    if s in POLICY_TO_ID:
        return s
    # Sometimes labels are stored as ids in strings
    if re.fullmatch(r"\d+", s):
        return ID_TO_POLICY.get(int(s), "UNKNOWN")
    su = s.upper()
    for name in POLICY_TO_ID:
        if name in su:
            return name
    return s


def assign_true_policy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "policy_class" not in df.columns:
        if "condition" not in df.columns:
            raise RuntimeError("Need condition or policy_class in feature/prediction table.")
        df["policy_class"] = df["condition"].astype(str).str.lower().map(CONDITION_TO_POLICY).fillna("NO_INTERVENTION")
    df["policy_class"] = df["policy_class"].map(normalize_policy_label)
    df["policy_id"] = df["policy_class"].map(POLICY_TO_ID).fillna(-1).astype(int)
    return df


def select_prediction_rows(pred: pd.DataFrame, feature_set: Optional[str], split: Optional[str], model: Optional[str]) -> Tuple[pd.DataFrame, Dict]:
    pred = pred.copy()
    original_rows = len(pred)
    if feature_set and "feature_set" in pred.columns:
        pred = pred[pred["feature_set"].astype(str) == feature_set]
    if split and "split" in pred.columns:
        pred = pred[pred["split"].astype(str) == split]
    elif split and "cv" in pred.columns:
        pred = pred[pred["cv"].astype(str) == split]
    if model and "model" in pred.columns:
        pred = pred[pred["model"].astype(str) == model]

    auto_used = False
    if len(pred) == 0:
        pred = read_csv(args.predictions_file) if False else None  # placeholder for static analyzers
        raise RuntimeError("No prediction rows matched explicit feature_set/split/model filters.")

    meta = {
        "initial_prediction_rows": original_rows,
        "selected_rows": len(pred),
        "explicit_feature_set": feature_set,
        "explicit_split": split,
        "explicit_model": model,
        "auto_used": auto_used,
    }
    return pred.copy(), meta


def auto_select_prediction_rows(all_pred: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    pred = all_pred.copy()
    for filt in PREFERRED_PREDICTION_FILTERS:
        tmp = pred.copy()
        ok = True
        for col, val in filt.items():
            if col not in tmp.columns:
                ok = False
                break
            tmp = tmp[tmp[col].astype(str) == val]
        if ok and len(tmp) > 0:
            return tmp.copy(), {
                "initial_prediction_rows": len(all_pred),
                "selected_rows": len(tmp),
                "auto_filter": filt,
                "auto_used": True,
            }
    # Fallback: largest group by feature_set/split/model if available
    group_cols = [c for c in ["feature_set", "split", "cv", "model"] if c in pred.columns]
    if group_cols:
        sizes = pred.groupby(group_cols, dropna=False).size().reset_index(name="n").sort_values("n", ascending=False)
        row = sizes.iloc[0]
        tmp = pred.copy()
        filt = {}
        for col in group_cols:
            val = row[col]
            filt[col] = str(val)
            tmp = tmp[tmp[col].astype(str) == str(val)]
        return tmp.copy(), {
            "initial_prediction_rows": len(all_pred),
            "selected_rows": len(tmp),
            "auto_filter": filt,
            "auto_used": True,
        }
    return pred.copy(), {
        "initial_prediction_rows": len(all_pred),
        "selected_rows": len(pred),
        "auto_filter": {},
        "auto_used": True,
    }


def infer_pred_columns(pred: pd.DataFrame) -> Tuple[str, str]:
    true_col = find_col(pred, [
        "y_true", "true_policy_class", "policy_class_true", "policy_class", "target", "label", "y",
    ])
    pred_col = find_col(pred, [
        "y_pred", "pred_policy_class", "policy_class_pred", "pred_class", "prediction", "pred", "yhat",
    ])
    if true_col is None:
        if "condition" in pred.columns:
            pred = assign_true_policy(pred)
            true_col = "policy_class"
        else:
            raise RuntimeError("Could not infer true label column in predictions file.")
    if pred_col is None:
        raise RuntimeError("Could not infer predicted label column in predictions file. Expected y_pred/pred_policy_class/pred/etc.")
    return true_col, pred_col


def action_class_from_text(policy, actions) -> str:
    text = f"{policy if not pd.isna(policy) else ''} {actions if not pd.isna(actions) else ''}".lower()
    text = text.replace("-", "_")
    if text.strip() in {"", "nan"}:
        return "UNKNOWN"
    if any(tok in text for tok in ["no_intervention", "no intervention", "none", "[]", "null", "control"]):
        # Be careful: a policy name containing control can still have actions. If it also has action terms, action terms win below.
        if not any(tok in text for tok in ["to_update", "update", "suppress_override", "order_precursor", "boundary_push"]):
            return "NO_INTERVENTION"
    if "suppress_override" in text or "three_stage" in text or "override_three" in text:
        return "OVERRIDE_THREE_STAGE"
    if "order_precursor" in text or "boundary_push" in text or "equal" in text:
        # Equal-evidence policy has update + order, but no suppress_override.
        if "suppress_override" not in text:
            return "EQUAL_EVIDENCE_ORDER"
    if "to_update" in text or "update" in text:
        return "CORE_CLOSURE_UPDATE"
    if any(tok in text for tok in ["stable", "noop", "no_intervention"]):
        return "NO_INTERVENTION"
    return "UNKNOWN"


def prepare_policy_results(paths: List[str]) -> Tuple[pd.DataFrame, Dict]:
    frames = []
    report = {"loaded": [], "failed": []}
    for p in paths:
        try:
            df = read_csv(p)
            df = ensure_sample_id(df)
            if "condition" not in df.columns:
                report["failed"].append({"path": p, "reason": "no condition column"})
                continue
            if "policy" not in df.columns:
                df["policy"] = Path(p).stem
            if "applied_actions" not in df.columns:
                df["applied_actions"] = ""
            df["action_class"] = [action_class_from_text(pol, act) for pol, act in zip(df["policy"], df["applied_actions"])]
            # Standardize outcome columns
            if "R_final" not in df.columns:
                if "baseline_R_final" in df.columns:
                    df["R_final"] = df["baseline_R_final"]
                elif "mean_R_final" in df.columns:
                    df["R_final"] = df["mean_R_final"]
            if "pred_clean" not in df.columns:
                if "baseline_pred_clean" in df.columns:
                    df["pred_clean"] = df["baseline_pred_clean"]
                elif "clean_rate" in df.columns:
                    df["pred_clean"] = df["clean_rate"]
            if "R_final" not in df.columns or "pred_clean" not in df.columns:
                report["failed"].append({"path": p, "reason": "no R_final/pred_clean outcome columns"})
                continue
            df["source_policy_file"] = str(p)
            keep_cols = [c for c in [
                "sample_id", "graph_id", "condition", "policy", "applied_actions", "action_class", "R_final", "pred_clean", "source_policy_file"
            ] if c in df.columns]
            frames.append(df[keep_cols].copy())
            report["loaded"].append({"path": p, "rows": int(len(df)), "action_counts": df["action_class"].value_counts().to_dict()})
        except Exception as e:
            report["failed"].append({"path": p, "error": repr(e)})
    if not frames:
        raise RuntimeError("No usable policy_results files were loaded.")
    allres = pd.concat(frames, ignore_index=True)
    allres["pred_clean"] = pd.to_numeric(allres["pred_clean"], errors="coerce")
    allres["R_final"] = pd.to_numeric(allres["R_final"], errors="coerce")
    # Aggregate to avoid one-to-many expansion / cherry-picking.
    group_keys = ["sample_id", "condition", "action_class"]
    agg = allres.groupby(group_keys, dropna=False).agg(
        replay_R_final=("R_final", "mean"),
        replay_pred_clean=("pred_clean", "mean"),
        n_policy_rows=("R_final", "size"),
        policies=("policy", lambda x: " | ".join(sorted(set(map(str, x)))[:8])),
        source_policy_files=("source_policy_file", lambda x: " | ".join(sorted(set(map(str, x)))[:8])),
    ).reset_index()
    report["aggregated_rows"] = int(len(agg))
    report["aggregated_action_counts"] = agg["action_class"].value_counts().to_dict()
    return agg, report


def prepare_baseline(path: str) -> pd.DataFrame:
    base = read_csv(path)
    base = ensure_sample_id(base)
    if "condition" not in base.columns:
        raise RuntimeError("baseline file must contain condition column")
    if "baseline_R_final" not in base.columns:
        if "R_final" in base.columns:
            base["baseline_R_final"] = base["R_final"]
        else:
            base["baseline_R_final"] = np.nan
    if "baseline_pred_clean" not in base.columns:
        if "pred_clean" in base.columns:
            base["baseline_pred_clean"] = base["pred_clean"]
        else:
            base["baseline_pred_clean"] = np.nan
    base = assign_true_policy(base)
    keep = [c for c in ["sample_id", "graph_id", "condition", "baseline_R_final", "baseline_pred_clean", "policy_class", "policy_id"] if c in base.columns]
    return base[keep].drop_duplicates(["sample_id", "condition"])


def build_replay_table(pred: pd.DataFrame, baseline: pd.DataFrame, outcomes: pd.DataFrame, true_col: str, pred_col: str) -> pd.DataFrame:
    pred = ensure_sample_id(pred)
    if "condition" not in pred.columns:
        # try to add condition from baseline by sample_id
        pred = pred.merge(baseline[["sample_id", "condition"]].drop_duplicates("sample_id"), on="sample_id", how="left")
    pred["true_policy_class"] = pred[true_col].map(normalize_policy_label)
    pred["pred_policy_class"] = pred[pred_col].map(normalize_policy_label)
    # If true column was numeric but not mapped, derive from condition
    bad_true = ~pred["true_policy_class"].isin(POLICY_TO_ID)
    if bad_true.any() and "condition" in pred.columns:
        derived = pred.loc[bad_true, "condition"].astype(str).str.lower().map(CONDITION_TO_POLICY).fillna("NO_INTERVENTION")
        pred.loc[bad_true, "true_policy_class"] = derived

    keys = ["sample_id", "condition"]
    pred = pred.merge(baseline.drop(columns=["policy_class", "policy_id"], errors="ignore"), on=keys, how="left", suffixes=("", "_base"))

    modes = {
        "predicted_policy": pred["pred_policy_class"],
        "oracle_policy": pred["true_policy_class"],
        "no_intervention": pd.Series(["NO_INTERVENTION"] * len(pred), index=pred.index),
        "fixed_core_update": pd.Series(["CORE_CLOSURE_UPDATE"] * len(pred), index=pred.index),
        "fixed_override_three_stage": pd.Series(["OVERRIDE_THREE_STAGE"] * len(pred), index=pred.index),
        "fixed_equal_order": pd.Series(["EQUAL_EVIDENCE_ORDER"] * len(pred), index=pred.index),
    }

    rows = []
    outcome_key = outcomes.set_index(["sample_id", "condition", "action_class"])
    for mode, action_series in modes.items():
        tmp = pred.copy()
        tmp["replay_mode"] = mode
        tmp["selected_action_class"] = action_series.map(normalize_policy_label).values
        r_vals, c_vals, n_vals, pol_vals, src_vals, missing = [], [], [], [], [], []
        for _, row in tmp.iterrows():
            key = (str(row["sample_id"]), str(row["condition"]), str(row["selected_action_class"]))
            if key in outcome_key.index:
                out = outcome_key.loc[key]
                if isinstance(out, pd.DataFrame):
                    out = out.iloc[0]
                r_vals.append(out.get("replay_R_final", np.nan))
                c_vals.append(out.get("replay_pred_clean", np.nan))
                n_vals.append(out.get("n_policy_rows", np.nan))
                pol_vals.append(out.get("policies", ""))
                src_vals.append(out.get("source_policy_files", ""))
                missing.append(False)
            else:
                r_vals.append(row.get("baseline_R_final", np.nan))
                c_vals.append(row.get("baseline_pred_clean", np.nan))
                n_vals.append(0)
                pol_vals.append("FALLBACK_BASELINE")
                src_vals.append("")
                missing.append(True)
        tmp["replay_R_final"] = r_vals
        tmp["replay_pred_clean"] = c_vals
        tmp["n_policy_rows"] = n_vals
        tmp["matched_policies"] = pol_vals
        tmp["matched_source_files"] = src_vals
        tmp["missing_action_outcome"] = missing
        tmp["baseline_clean"] = pd.to_numeric(tmp["baseline_pred_clean"], errors="coerce")
        tmp["baseline_R_final"] = pd.to_numeric(tmp["baseline_R_final"], errors="coerce")
        tmp["clean_gain"] = tmp["replay_pred_clean"] - tmp["baseline_clean"]
        tmp["R_gain"] = tmp["replay_R_final"] - tmp["baseline_R_final"]
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def summarize(replay: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    replay = replay.copy()
    replay["is_target"] = replay["true_policy_class"] != "NO_INTERVENTION"
    replay["is_protected"] = replay["true_policy_class"] == "NO_INTERVENTION"

    def safe_mean(x):
        return float(pd.to_numeric(x, errors="coerce").mean())

    rows = []
    for mode, g in replay.groupby("replay_mode"):
        target = g[g["is_target"]]
        prot = g[g["is_protected"]]
        rows.append({
            "replay_mode": mode,
            "n": int(len(g)),
            "missing_action_rate": safe_mean(g["missing_action_outcome"].astype(float)),
            "clean_rate": safe_mean(g["replay_pred_clean"]),
            "baseline_clean_rate": safe_mean(g["baseline_clean"]),
            "clean_rate_gain": safe_mean(g["clean_gain"]),
            "mean_R_final": safe_mean(g["replay_R_final"]),
            "mean_R_gain": safe_mean(g["R_gain"]),
            "target_clean_rate": safe_mean(target["replay_pred_clean"]) if len(target) else np.nan,
            "target_clean_gain": safe_mean(target["clean_gain"]) if len(target) else np.nan,
            "protected_clean_rate": safe_mean(prot["replay_pred_clean"]) if len(prot) else np.nan,
            "protected_clean_gain": safe_mean(prot["clean_gain"]) if len(prot) else np.nan,
            "protected_harm": max(0.0, -safe_mean(prot["clean_gain"])) if len(prot) else np.nan,
            "safety_adjusted_target_gain": (safe_mean(target["clean_gain"]) if len(target) else 0.0) - (max(0.0, -safe_mean(prot["clean_gain"])) if len(prot) else 0.0),
        })
    summary = pd.DataFrame(rows).sort_values("replay_mode")

    cond = replay.groupby(["replay_mode", "condition", "true_policy_class"], dropna=False).agg(
        n=("sample_id", "size"),
        clean_rate=("replay_pred_clean", "mean"),
        baseline_clean_rate=("baseline_clean", "mean"),
        clean_rate_gain=("clean_gain", "mean"),
        mean_R_final=("replay_R_final", "mean"),
        mean_R_gain=("R_gain", "mean"),
        missing_action_rate=("missing_action_outcome", "mean"),
    ).reset_index()
    return summary, cond


def make_verdict(summary: pd.DataFrame, pred_accuracy: Optional[float], meta: Dict, policy_report: Dict) -> Dict:
    def row(mode):
        sub = summary[summary["replay_mode"] == mode]
        return sub.iloc[0].to_dict() if len(sub) else {}

    pred = row("predicted_policy")
    oracle = row("oracle_policy")
    noint = row("no_intervention")
    fixed_modes = ["fixed_core_update", "fixed_override_three_stage", "fixed_equal_order"]
    fixed_best = None
    for m in fixed_modes:
        r = row(m)
        if not r:
            continue
        if fixed_best is None or r.get("safety_adjusted_target_gain", -999) > fixed_best.get("safety_adjusted_target_gain", -999):
            fixed_best = r

    pred_safety = pred.get("safety_adjusted_target_gain", np.nan)
    oracle_safety = oracle.get("safety_adjusted_target_gain", np.nan)
    fixed_safety = fixed_best.get("safety_adjusted_target_gain", np.nan) if fixed_best else np.nan
    noint_safety = noint.get("safety_adjusted_target_gain", np.nan)

    # Conservative verdict based on table replay, not actual fresh model run.
    if math.isnan(pred_safety):
        verdict = "FAIL_NO_PREDICTED_REPLAY"
    elif pred.get("missing_action_rate", 1.0) > 0.5:
        verdict = "FAIL_INSUFFICIENT_ACTION_COVERAGE"
    elif pred_safety >= max(fixed_safety, noint_safety) - 1e-9 and pred_safety >= 0 and pred.get("protected_harm", 1.0) <= 0.05:
        if not math.isnan(oracle_safety) and abs(pred_safety - oracle_safety) <= 0.05:
            verdict = "PASS_STRONG_TABLE_REPLAY"
        else:
            verdict = "PASS_LITE_TABLE_REPLAY"
    elif pred_safety >= noint_safety and pred.get("protected_harm", 1.0) <= 0.1:
        verdict = "PASS_WEAK_BEATS_NO_INTERVENTION"
    else:
        verdict = "FAIL_PREDICTED_POLICY_REPLAY"

    return {
        "experiment": "DA-ASA-2D.2 Predicted Policy Control Replay",
        "method": "table_replay_using_existing_policy_results_not_model_rerun",
        "prediction_selection": meta,
        "prediction_accuracy_if_available": pred_accuracy,
        "policy_results_report": policy_report,
        "predicted_policy": pred,
        "oracle_policy": oracle,
        "no_intervention": noint,
        "best_fixed_policy": fixed_best,
        "verdict": verdict,
        "caveats": [
            "This is replay-by-table from already generated policy_results; it does not rerun the model with predicted interventions.",
            "If policy_results lack a selected action for a sample, the script falls back to baseline and records missing_action_outcome.",
            "For full 2D.2, follow with true model replay using predicted PolicyClass to choose interventions online.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="DA-ASA-2D.2 predicted policy control replay by existing policy_results tables.")
    parser.add_argument("--predictions-file", required=True, help="2D.1B predictions CSV")
    parser.add_argument("--baseline-file", required=True, help="DA-ASA baseline_scores CSV, e.g. da_asa2c_baseline_scores.csv")
    parser.add_argument("--policy-results-file", action="append", required=True, help="One or more DA-ASA policy_results CSV files")
    parser.add_argument("--feature-set", default=None, help="Optional prediction feature_set filter")
    parser.add_argument("--split", default=None, help="Optional prediction split/cv filter, e.g. group_graph")
    parser.add_argument("--model", default=None, help="Optional prediction model filter, e.g. logreg_balanced")
    parser.add_argument("--out-dir", default="da_asa_2d2_outputs")
    args_ns = parser.parse_args()

    out_dir = Path(args_ns.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_pred = read_csv(args_ns.predictions_file)
    all_pred = ensure_sample_id(all_pred)
    if args_ns.feature_set or args_ns.split or args_ns.model:
        pred, meta = select_prediction_rows(all_pred, args_ns.feature_set, args_ns.split, args_ns.model)
    else:
        pred, meta = auto_select_prediction_rows(all_pred)
    true_col, pred_col = infer_pred_columns(pred)
    pred["_true_norm"] = pred[true_col].map(normalize_policy_label)
    pred["_pred_norm"] = pred[pred_col].map(normalize_policy_label)
    pred_accuracy = float((pred["_true_norm"] == pred["_pred_norm"]).mean()) if len(pred) else None
    meta.update({"true_col": true_col, "pred_col": pred_col, "selected_prediction_accuracy": pred_accuracy})

    baseline = prepare_baseline(args_ns.baseline_file)
    outcomes, policy_report = prepare_policy_results(args_ns.policy_results_file)
    replay = build_replay_table(pred, baseline, outcomes, true_col, pred_col)
    summary, cond_summary = summarize(replay)
    verdict = make_verdict(summary, pred_accuracy, meta, policy_report)

    replay.to_csv(out_dir / "da_asa2d2_sample_replay.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "da_asa2d2_replay_summary.csv", index=False, encoding="utf-8-sig")
    cond_summary.to_csv(out_dir / "da_asa2d2_condition_summary.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "da_asa2d2_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)

    print("[OK] DA-ASA-2D.2 table replay complete")
    print(f"[out] {out_dir}")
    print(f"[prediction rows] {len(pred)}")
    print(f"[prediction accuracy] {pred_accuracy}")
    print(f"[verdict] {verdict['verdict']}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
