# -*- coding: utf-8 -*-
r"""
SEM-6J.2b: No-Fallback Live-Only Action Value Audit

Purpose
-------
SEM-6J.2 gave strong action/value metrics, but the target source was:

    live_metric:trajectory_improvement_fallback_oracle

This means the value model may mix true live trajectory metrics with oracle utility.
SEM-6J.2b removes that ambiguity.

This script ONLY uses rows with real live metrics from SEM-6J live controller.
No oracle fallback is allowed.

Core questions
--------------
1. Which minimal action has positive real live trajectory value?
2. Does learned_noleak_Q route to actions with higher live value than baselines?
3. Is correction_operator genuinely negative in live trajectory, as smoke suggested?
4. Should 6J.3 live controller focus on address_gate / operator_prior_transfer first?

Inputs
------
Required:
    sem6j1_outputs\sem6j1_asa_operator_write_plan.csv
    sem6j_outputs\sem6j_live_results.csv

Optional:
    sem6j2_outputs\sem6j2_predicted_policy.csv

Outputs
-------
sem6j2b_outputs\
    sem6j2b_live_only_table.csv
    sem6j2b_action_value_summary.csv
    sem6j2b_selection_pairwise.csv
    sem6j2b_action_value_model_report.json
    sem6j2b_verdict.json

Run
---
conda activate dhrf_4080s
cd C:\Users\ZH\Desktop\AGI\python_script

python GPT_sem6j2b_live_only_action_value_audit.py

If your live file has another name:
python GPT_sem6j2b_live_only_action_value_audit.py ^
  --live_results sem6j_outputs\sem6j_live_results.csv
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_ROOT = r"C:\Users\ZH\Desktop\AGI\python_script"
DEFAULT_PLAN = rf"{DEFAULT_ROOT}\sem6j1_outputs\sem6j1_asa_operator_write_plan.csv"
DEFAULT_LIVE = rf"{DEFAULT_ROOT}\sem6j_outputs\sem6j_live_results.csv"
DEFAULT_OUT_DIR = rf"{DEFAULT_ROOT}\sem6j2b_outputs"

RANDOM_SEED = 20260606


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def coerce_minimal_action(action: str, route: str = "") -> str:
    action = str(action)
    route = str(route)
    if action in {"address_gate", "operator_prior_transfer", "no_intervention"}:
        return action
    if action in {
        "correction_operator", "planning_operator", "risk_audit_operator",
        "counterexample_operator", "invariant_search_operator", "closure_check_operator",
        "relation_mapping_operator", "mechanism_operator", "comparison_operator",
        "policy_gate", "stable_operator", "update_operator", "boundary_push",
        "suppress_override_operator", "order_precursor"
    }:
        return "correction_operator"
    if route == "write_address_gate":
        return "address_gate"
    if route == "write_operator_prior":
        return "operator_prior_transfer"
    if route == "reject_as_null":
        return "no_intervention"
    if route in {"trigger_correction_operator", "write_residual_mode"}:
        return "correction_operator"
    return "no_intervention"


def find_live_metric(df: pd.DataFrame, preferred: str = "") -> str:
    if preferred and preferred in df.columns:
        return preferred

    candidates = [
        "trajectory_improvement",
        "live_margin_gain",
        "live_gain",
        "live_logprob_gap_gain",
        "live_rank_improvement",
    ]
    for c in candidates:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            return c

    # generic scan
    for c in df.columns:
        nc = norm(c)
        if any(x in nc for x in ["trajectory_improvement", "live_margin_gain", "live_gain"]):
            if pd.api.types.is_numeric_dtype(df[c]):
                return c
    raise ValueError(f"No live metric found. Columns={list(df.columns)}")


def robust_merge_plan_live(plan: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer _live_row_id if available. Otherwise merge on selection + group + candidate fields.
    """
    if "_live_row_id" in plan.columns and "_live_row_id" in live.columns:
        # Important: live rows may already contain plan columns; no merge needed if all action cols present.
        merged = live.copy()
        # Bring missing plan action columns.
        needed = ["minimal_action", "asa_action", "write_route", "relation_type", "asa_alpha", "asa_beta"]
        missing = [c for c in needed if c not in merged.columns and c in plan.columns]
        if missing:
            pcols = ["_live_row_id"] + missing
            merged = merged.merge(plan[pcols].drop_duplicates("_live_row_id"), on="_live_row_id", how="left")
        return merged

    key_candidates = [
        "_selection", "_selection_family", "_group_id", "candidate_family", "candidate_i",
        "target_operator", "candidate_operator", "target_concept", "candidate_concept"
    ]
    keys = [k for k in key_candidates if k in plan.columns and k in live.columns]
    if not keys:
        # live may already include all plan columns
        return live.copy()
    p = plan.drop_duplicates(subset=keys)
    return live.merge(p, on=keys, how="left", suffixes=("", "_plan"))


def add_minimal_action(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "minimal_action" not in out.columns:
        out["minimal_action"] = out.apply(
            lambda r: coerce_minimal_action(r.get("asa_action", ""), r.get("write_route", "")),
            axis=1
        )
    else:
        out["minimal_action"] = out.apply(
            lambda r: coerce_minimal_action(r.get("minimal_action", ""), r.get("write_route", "")),
            axis=1
        )
    return out


def summarize_actions(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for action, sub in df.groupby("minimal_action", dropna=False):
        vals = pd.to_numeric(sub[metric], errors="coerce").dropna().to_numpy(float)
        if len(vals) == 0:
            continue
        rows.append({
            "minimal_action": action,
            "n": int(len(vals)),
            "mean_live_value": float(np.mean(vals)),
            "median_live_value": float(np.median(vals)),
            "std_live_value": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
            "positive_rate": float(np.mean(vals > 0)),
            "q25": float(np.percentile(vals, 25)),
            "q75": float(np.percentile(vals, 75)),
        })
    return pd.DataFrame(rows).sort_values("mean_live_value", ascending=False)


def summarize_action_by_selection(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(["_selection_family", "minimal_action"], dropna=False):
        sel, action = keys
        vals = pd.to_numeric(sub[metric], errors="coerce").dropna().to_numpy(float)
        if len(vals) == 0:
            continue
        rows.append({
            "selection_family": sel,
            "minimal_action": action,
            "n": int(len(vals)),
            "mean_live_value": float(np.mean(vals)),
            "positive_rate": float(np.mean(vals > 0)),
        })
    return pd.DataFrame(rows).sort_values(["selection_family", "mean_live_value"], ascending=[True, False])


def pairwise_selection(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    if "_selection_family" not in df.columns:
        return pd.DataFrame()
    if "_group_id" not in df.columns:
        # fallback to family if available
        if "family" in df.columns:
            df = df.copy()
            df["_group_id"] = df["family"]
        else:
            return pd.DataFrame()

    base = "learned_noleak_Q"
    baselines = ["commit_topk_rule", "same_family", "random_pool", "shuffle_learned_Q", "proxy_only", "oracle_selected"]
    if base not in set(df["_selection_family"].astype(str)):
        return pd.DataFrame()

    A = df[df["_selection_family"] == base][["_group_id", metric]].copy()
    A = A.groupby("_group_id", as_index=False)[metric].mean().rename(columns={metric: "a"})

    for b in baselines:
        if b not in set(df["_selection_family"].astype(str)):
            continue
        B = df[df["_selection_family"] == b][["_group_id", metric]].copy()
        B = B.groupby("_group_id", as_index=False)[metric].mean().rename(columns={metric: "b"})
        m = A.merge(B, on="_group_id", how="inner")
        diff = (pd.to_numeric(m["a"], errors="coerce") - pd.to_numeric(m["b"], errors="coerce")).dropna().to_numpy(float)
        z = np.nan
        if len(diff) > 1 and np.std(diff, ddof=1) > 1e-12:
            z = float(np.mean(diff) / (np.std(diff, ddof=1) / math.sqrt(len(diff))))
        rows.append({
            "comparison": f"{base}>{b}",
            "n": int(len(diff)),
            "mean_diff": float(np.mean(diff)) if len(diff) else np.nan,
            "z": z,
            "win_rate": float(np.mean(diff > 0)) if len(diff) else np.nan,
        })
    return pd.DataFrame(rows)


def feature_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric = []
    for c in [
        "_learned_noleak_q", "proxy_utility", "oracle_utility",
        "asa_alpha", "asa_beta", "alpha_operator", "beta_precursor",
        "live_rank_improvement", "n_delta_layers", "delta_scale",
    ]:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)

    # Avoid using the target metric itself.
    categorical = []
    for c in [
        "_selection_family", "write_route", "relation_type", "target_operator",
        "candidate_operator", "asa_action", "asa_action_family", "minimal_action"
    ]:
        if c in df.columns:
            categorical.append(c)
    return numeric, categorical


def preprocess(numeric: List[str], categorical: List[str]):
    parts = []
    if numeric:
        parts.append(("num", StandardScaler(), numeric))
    if categorical:
        try:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
        parts.append(("cat", enc, categorical))
    return ColumnTransformer(parts, remainder="drop")


def cv_value_model(df: pd.DataFrame, metric: str) -> Dict:
    work = df.dropna(subset=[metric]).copy()
    if len(work) < 20:
        return {"n": int(len(work)), "cv_corr": np.nan, "cv_r2": np.nan, "cv_rmse": np.nan, "note": "too few rows"}

    numeric, categorical = feature_cols(work)
    # Do not leak live target.
    numeric = [c for c in numeric if c != metric and c not in {"live_gain", "live_margin_gain", "trajectory_improvement"}]

    X = work[numeric + categorical].copy()
    for c in numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median() if X[c].notna().any() else 0.0)
    for c in categorical:
        X[c] = X[c].astype(str).fillna("NA")
    y = pd.to_numeric(work[metric], errors="coerce").to_numpy(float)

    groups = None
    for c in ["_group_id", "family", "candidate_family"]:
        if c in work.columns:
            groups = work[c].astype(str).to_numpy()
            break

    if groups is not None and len(np.unique(groups)) >= 2:
        k = min(5, len(np.unique(groups)))
        splits = list(GroupKFold(n_splits=k).split(X, y, groups))
    else:
        k = min(5, len(work))
        splits = list(KFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED).split(X, y))

    pred = np.full(len(work), np.nan)
    for tr, te in splits:
        reg = Pipeline([
            ("pre", preprocess(numeric, categorical)),
            ("reg", ExtraTreesRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ))
        ])
        reg.fit(X.iloc[tr], y[tr])
        pred[te] = reg.predict(X.iloc[te])

    mask = np.isfinite(pred)
    corr = np.nan
    r2 = np.nan
    rmse = np.nan
    if mask.sum() > 1 and np.std(pred[mask]) > 1e-12 and np.std(y[mask]) > 1e-12:
        corr = float(np.corrcoef(y[mask], pred[mask])[0, 1])
        r2 = float(r2_score(y[mask], pred[mask]))
        rmse = float(math.sqrt(mean_squared_error(y[mask], pred[mask])))

    return {
        "n": int(len(work)),
        "metric": metric,
        "features": {"numeric": numeric, "categorical": categorical},
        "cv_corr": corr,
        "cv_r2": r2,
        "cv_rmse": rmse,
    }


def make_verdict(action_summary: pd.DataFrame, pairwise: pd.DataFrame, model_report: Dict, metric: str) -> Dict:
    verdict = "FAIL_OR_INCONCLUSIVE"
    notes = []

    # Check if any action has positive mean.
    positive_actions = []
    if not action_summary.empty:
        positive_actions = action_summary[action_summary["mean_live_value"] > 0]["minimal_action"].tolist()

    if positive_actions:
        verdict = "PASS_LITE_ACTION_HAS_POSITIVE_LIVE_VALUE"
        notes.append(f"Actions with positive mean live value: {positive_actions}")
    else:
        notes.append("No minimal action has positive mean live value.")

    corr = model_report.get("cv_corr", np.nan)
    if np.isfinite(corr) and corr > 0.30:
        if verdict.startswith("PASS_LITE"):
            verdict = "PASS_MIXED_POSITIVE_ACTION_AND_VALUE_MODEL"
        else:
            verdict = "PASS_LITE_VALUE_MODEL_ONLY"
        notes.append(f"Live-only value model has positive CV corr={corr:.3f}.")

    # Strong only if learned_Q beats baselines and some action positive.
    if not pairwise.empty:
        needed = ["learned_noleak_Q>commit_topk_rule", "learned_noleak_Q>random_pool", "learned_noleak_Q>shuffle_learned_Q"]
        ok = []
        for comp in needed:
            row = pairwise[pairwise["comparison"] == comp]
            if len(row):
                r = row.iloc[0]
                ok.append(pd.notna(r["mean_diff"]) and r["mean_diff"] > 0 and pd.notna(r["z"]) and r["z"] > 2)
            else:
                ok.append(False)
        if all(ok) and positive_actions:
            verdict = "PASS_STRONG_LIVE_ONLY_ACTION_VALUE"

    return {
        "stage": "SEM-6J.2b",
        "mode": "live_only_no_fallback_action_value_audit",
        "metric": metric,
        "verdict": verdict,
        "positive_actions": positive_actions,
        "value_model": model_report,
        "notes": notes + [
            "No oracle fallback is used in SEM-6J.2b.",
            "If correction_operator is negative, split it before SEM-6J.3.",
            "If only value_model passes but action means remain negative, route/action choice is predictable but not yet useful for live control.",
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=DEFAULT_PLAN)
    ap.add_argument("--live_results", default=DEFAULT_LIVE)
    ap.add_argument("--metric", default="")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan_path = Path(args.plan)
    live_path = Path(args.live_results)
    if not plan_path.exists():
        raise FileNotFoundError(f"Plan not found: {plan_path}")
    if not live_path.exists():
        raise FileNotFoundError(f"Live results not found: {live_path}")

    plan = pd.read_csv(plan_path)
    live = pd.read_csv(live_path)
    merged = robust_merge_plan_live(plan, live)
    merged = add_minimal_action(merged)

    metric = find_live_metric(merged, args.metric)
    merged[metric] = pd.to_numeric(merged[metric], errors="coerce")

    # Strict: drop missing metric. No fallback.
    table = merged.dropna(subset=[metric]).copy()

    table_path = out_dir / "sem6j2b_live_only_table.csv"
    table.to_csv(table_path, index=False, encoding="utf-8-sig")

    action_summary = summarize_actions(table, metric)
    action_summary.to_csv(out_dir / "sem6j2b_action_value_summary.csv", index=False, encoding="utf-8-sig")

    action_selection_summary = summarize_action_by_selection(table, metric)
    action_selection_summary.to_csv(out_dir / "sem6j2b_action_by_selection_summary.csv", index=False, encoding="utf-8-sig")

    pair = pairwise_selection(table, metric)
    pair.to_csv(out_dir / "sem6j2b_selection_pairwise.csv", index=False, encoding="utf-8-sig")

    model_report = cv_value_model(table, metric)
    with open(out_dir / "sem6j2b_action_value_model_report.json", "w", encoding="utf-8") as f:
        json.dump(model_report, f, ensure_ascii=False, indent=2)

    verdict = make_verdict(action_summary, pair, model_report, metric)
    verdict["n_live_rows"] = int(len(table))
    verdict["live_results"] = str(live_path)
    verdict["plan"] = str(plan_path)
    verdict["outputs"] = {
        "live_only_table": str(table_path),
        "action_summary": str(out_dir / "sem6j2b_action_value_summary.csv"),
        "action_by_selection_summary": str(out_dir / "sem6j2b_action_by_selection_summary.csv"),
        "pairwise": str(out_dir / "sem6j2b_selection_pairwise.csv"),
        "value_model_report": str(out_dir / "sem6j2b_action_value_model_report.json"),
    }

    with open(out_dir / "sem6j2b_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)

    print("\n========== SEM-6J.2b ==========")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    print("\nAction summary:")
    print(action_summary.to_string(index=False))
    print("\nSelection pairwise:")
    print(pair.to_string(index=False))


if __name__ == "__main__":
    main()
