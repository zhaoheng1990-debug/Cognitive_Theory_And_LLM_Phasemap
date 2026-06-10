# -*- coding: utf-8 -*-
r"""
SEM-6K: Action-Conditioned Q Audit

Purpose
-------
SEM-6H.1b:
    Q(E) candidate selection = PASS

SEM-6H.2 / 6H.2b / 6J:
    direct write implementations = FAIL/INCONCLUSIVE

SEM-6J.2b:
    live-only action value is predictable:
        corr ≈ 0.663, R2 ≈ 0.421
    but no coarse action has positive mean live value

SEM-6J.3b:
    address_gate primitive is weakly viable / near-neutral
    but learned-Q address candidate selection does not beat random

Conclusion:
    The next object is not Q(E), but Q(E, a).

This script builds an action-conditioned Q dataset from all available live-only
SEM-6 write experiments:

    6J live controller
    6J.3 address_gate-only
    6J.3b expanded address_gate

It trains Q_theta(E, a) -> live trajectory improvement.

Outputs
-------
sem6k_outputs/
    sem6k_action_conditioned_dataset.csv
    sem6k_action_value_summary.csv
    sem6k_model_report.json
    sem6k_policy_recommendations.csv
    sem6k_counterfactual_action_table.csv
    sem6k_verdict.json

Run
---
conda activate dhrf_4080s
cd C:\Users\ZH\Desktop\AGI\python_script

python GPT_sem6k_action_conditioned_q_audit.py

Optional:
python GPT_sem6k_action_conditioned_q_audit.py ^
  --include_6j sem6j_outputs\sem6j_live_results.csv ^
  --include_6j3 sem6j3_outputs\sem6j3_address_gate_live_results.csv ^
  --include_6j3b sem6j3b_outputs\sem6j3b_address_gate_live_results.csv

Interpretation
--------------
PASS-Lite:
    action-conditioned value model beats action-agnostic baseline by corr gain > 0.05

PASS-Strong:
    action-conditioned Q has group-CV corr > 0.30 and recommends an action policy
    whose observed/estimated value is better than current learned-Q selection.

This is still not the final live controller. It is the Q(E,a) learning step before
SEM-6K.1 live validation.
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
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_ROOT = r"C:\Users\ZH\Desktop\AGI\python_script"
DEFAULT_OUT_DIR = rf"{DEFAULT_ROOT}\sem6k_outputs"

DEFAULT_6J = rf"{DEFAULT_ROOT}\sem6j_outputs\sem6j_live_results.csv"
DEFAULT_6J3 = rf"{DEFAULT_ROOT}\sem6j3_outputs\sem6j3_address_gate_live_results.csv"
DEFAULT_6J3B = rf"{DEFAULT_ROOT}\sem6j3b_outputs\sem6j3b_address_gate_live_results.csv"

RANDOM_SEED = 20260606


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def find_metric(df: pd.DataFrame, preferred: str = "") -> Optional[str]:
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
    for c in df.columns:
        nc = norm(c)
        if any(x in nc for x in ["trajectory_improvement", "live_margin_gain", "live_gain"]):
            if pd.api.types.is_numeric_dtype(df[c]):
                return c
    return None


def coerce_action(row: pd.Series, source: str) -> str:
    # Explicit fields first.
    for c in ["minimal_action", "asa_action", "write_route"]:
        if c in row.index and pd.notna(row[c]):
            v = str(row[c])
            if v in {"address_gate", "operator_prior_transfer", "no_intervention"}:
                return v
            if v == "write_address_gate":
                return "address_gate"
            if v == "write_operator_prior":
                return "operator_prior_transfer"
            if v == "reject_as_null":
                return "no_intervention"
            if v in {"trigger_correction_operator", "write_residual_mode"}:
                return "correction_operator"
            if "operator" in v and v not in {"operator_prior_transfer"}:
                return "correction_operator"
    if source == "6j3" or source == "6j3b":
        return "address_gate"
    return "unknown_action"


def load_live_file(path: str, source: str, metric_name: str = "") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()

    df = pd.read_csv(p)
    metric = find_metric(df, metric_name)
    if metric is None:
        print(f"[warn] no live metric in {p}")
        return pd.DataFrame()

    out = df.copy()
    out["source_experiment"] = source
    out["target_value"] = pd.to_numeric(out[metric], errors="coerce")
    out["target_metric"] = metric
    out = out.dropna(subset=["target_value"]).copy()

    out["action"] = out.apply(lambda r: coerce_action(r, source), axis=1)

    # Normalize some expected fields.
    if "_selection_family" not in out.columns:
        out["_selection_family"] = source
    if "_group_id" not in out.columns:
        if "family" in out.columns:
            out["_group_id"] = out["family"]
        else:
            out["_group_id"] = np.arange(len(out)).astype(str)

    if "beta" not in out.columns:
        out["beta"] = np.nan
    if "delta_scale" not in out.columns:
        out["delta_scale"] = np.nan

    return out


def build_dataset(files: Dict[str, str], metric: str = "") -> pd.DataFrame:
    parts = []
    for src, path in files.items():
        part = load_live_file(path, src, metric)
        if not part.empty:
            parts.append(part)
            print(f"[load] {src}: {len(part)} rows from {path}")
        else:
            print(f"[skip] {src}: no usable rows from {path}")
    if not parts:
        raise FileNotFoundError("No usable live result files found for SEM-6K.")
    df = pd.concat(parts, ignore_index=True)

    # Derive relation/action features.
    if "relation_type" not in df.columns:
        df["relation_type"] = "unknown_relation"
    if "write_route" not in df.columns:
        df["write_route"] = df["action"]
    if "target_operator" not in df.columns:
        df["target_operator"] = "unknown_target_operator"
    if "candidate_operator" not in df.columns:
        df["candidate_operator"] = "unknown_candidate_operator"
    if "target_concept" not in df.columns:
        df["target_concept"] = "unknown_target_concept"
    if "candidate_concept" not in df.columns:
        df["candidate_concept"] = "unknown_candidate_concept"

    # Numeric cleanups.
    numeric_maybe = [
        "_learned_noleak_q", "proxy_utility", "oracle_utility",
        "asa_alpha", "asa_beta", "alpha_operator", "beta_precursor",
        "beta", "delta_scale", "live_rank_improvement", "n_delta_layers",
        "base_margin", "base_rank", "base_logprob_gap_to_top",
    ]
    for c in numeric_maybe:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

    df["q_proxy_gap"] = df["_learned_noleak_q"] - df["proxy_utility"]
    df["q_oracle_gap"] = df["_learned_noleak_q"] - df["oracle_utility"]
    df["is_address_gate"] = (df["action"] == "address_gate").astype(int)
    df["is_correction"] = (df["action"] == "correction_operator").astype(int)
    df["is_operator_prior"] = (df["action"] == "operator_prior_transfer").astype(int)
    df["is_no_intervention"] = (df["action"] == "no_intervention").astype(int)
    return df


def feature_cols(df: pd.DataFrame, action_conditioned: bool = True) -> Tuple[List[str], List[str]]:
    numeric = [
        "_learned_noleak_q", "proxy_utility", "oracle_utility",
        "asa_alpha", "asa_beta", "alpha_operator", "beta_precursor",
        "beta", "delta_scale", "live_rank_improvement", "n_delta_layers",
        "base_margin", "base_rank", "base_logprob_gap_to_top",
        "q_proxy_gap", "q_oracle_gap",
    ]

    # Do not include rank improvement if it could be target-like? It may be another live metric.
    # Keep it optional but remove from default if target is trajectory_improvement in later filter.
    numeric = [c for c in numeric if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    categorical = [
        "source_experiment", "_selection_family", "relation_type", "write_route",
        "target_operator", "candidate_operator", "target_concept", "candidate_concept",
    ]
    if action_conditioned:
        categorical.append("action")
        numeric += [c for c in ["is_address_gate", "is_correction", "is_operator_prior", "is_no_intervention"] if c in df.columns]

    categorical = [c for c in categorical if c in df.columns]
    # Avoid using raw live target columns.
    numeric = [c for c in numeric if c not in {"target_value", "trajectory_improvement", "live_margin_gain", "live_gain"}]
    return list(dict.fromkeys(numeric)), list(dict.fromkeys(categorical))


def make_X(df: pd.DataFrame, numeric: List[str], categorical: List[str]) -> pd.DataFrame:
    X = df[numeric + categorical].copy()
    for c in numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median() if X[c].notna().any() else 0.0)
    for c in categorical:
        X[c] = X[c].astype(str).fillna("NA")
    return X


def preprocessor(numeric: List[str], categorical: List[str]):
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


def cv_regression(df: pd.DataFrame, action_conditioned: bool) -> Tuple[np.ndarray, Dict]:
    work = df.dropna(subset=["target_value"]).copy()
    numeric, categorical = feature_cols(work, action_conditioned=action_conditioned)

    X = make_X(work, numeric, categorical)
    y = pd.to_numeric(work["target_value"], errors="coerce").to_numpy(float)

    groups = None
    for c in ["_group_id", "target_concept", "candidate_concept"]:
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
            ("pre", preprocessor(numeric, categorical)),
            ("reg", ExtraTreesRegressor(
                n_estimators=500,
                min_samples_leaf=2,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ))
        ])
        reg.fit(X.iloc[tr], y[tr])
        pred[te] = reg.predict(X.iloc[te])

    mask = np.isfinite(pred)
    if mask.sum() > 1 and np.std(pred[mask]) > 1e-12 and np.std(y[mask]) > 1e-12:
        corr = float(np.corrcoef(y[mask], pred[mask])[0, 1])
        r2 = float(r2_score(y[mask], pred[mask]))
        rmse = float(math.sqrt(mean_squared_error(y[mask], pred[mask])))
    else:
        corr, r2, rmse = np.nan, np.nan, np.nan

    report = {
        "n": int(len(work)),
        "action_conditioned": action_conditioned,
        "features": {"numeric": numeric, "categorical": categorical},
        "cv_corr": corr,
        "cv_r2": r2,
        "cv_rmse": rmse,
    }
    return pred, report


def summarize_actions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["source_experiment", "action"]
    for keys, sub in df.groupby(group_cols, dropna=False):
        src, action = keys
        vals = pd.to_numeric(sub["target_value"], errors="coerce").dropna().to_numpy(float)
        if len(vals) == 0:
            continue
        rows.append({
            "source_experiment": src,
            "action": action,
            "n": int(len(vals)),
            "mean_value": float(np.mean(vals)),
            "median_value": float(np.median(vals)),
            "std_value": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
            "positive_rate": float(np.mean(vals > 0)),
        })
    return pd.DataFrame(rows).sort_values(["source_experiment", "mean_value"], ascending=[True, False])


def counterfactual_action_table(df: pd.DataFrame, model_report: Dict, out_dir: Path) -> pd.DataFrame:
    """
    Approximate counterfactual table by observed action means per target_operator/relation.
    This is not generated by model predictions for unobserved actions yet; it is a safe
    empirical recommendation table.
    """
    keys = ["target_operator", "relation_type", "action"]
    rows = []
    for k, sub in df.groupby(keys, dropna=False):
        target_op, rel, action = k
        vals = pd.to_numeric(sub["target_value"], errors="coerce").dropna().to_numpy(float)
        if len(vals) < 2:
            continue
        rows.append({
            "target_operator": target_op,
            "relation_type": rel,
            "recommended_action": action,
            "n": int(len(vals)),
            "mean_value": float(np.mean(vals)),
            "positive_rate": float(np.mean(vals > 0)),
        })
    tab = pd.DataFrame(rows)
    if tab.empty:
        return tab
    # keep best action per target/relation
    tab = tab.sort_values(["target_operator", "relation_type", "mean_value"], ascending=[True, True, False])
    best = tab.groupby(["target_operator", "relation_type"], as_index=False).head(1)
    best = best.sort_values("mean_value", ascending=False)
    best.to_csv(out_dir / "sem6k_policy_recommendations.csv", index=False, encoding="utf-8-sig")
    tab.to_csv(out_dir / "sem6k_counterfactual_action_table.csv", index=False, encoding="utf-8-sig")
    return best


def make_verdict(df: pd.DataFrame, ag_report: Dict, ac_report: Dict, action_summary: pd.DataFrame) -> Dict:
    ag_corr = ag_report.get("cv_corr", np.nan)
    ac_corr = ac_report.get("cv_corr", np.nan)
    corr_gain = ac_corr - ag_corr if np.isfinite(ac_corr) and np.isfinite(ag_corr) else np.nan

    any_positive = False
    positive_actions = []
    if not action_summary.empty:
        pos = action_summary[action_summary["mean_value"] > 0]
        any_positive = len(pos) > 0
        positive_actions = pos[["source_experiment", "action", "n", "mean_value", "positive_rate"]].to_dict("records")

    pass_lite = bool(np.isfinite(ac_corr) and ac_corr > 0.10 and (not np.isfinite(ag_corr) or corr_gain > 0.02))
    pass_strong = bool(np.isfinite(ac_corr) and ac_corr > 0.30 and np.isfinite(corr_gain) and corr_gain > 0.05)

    if pass_strong:
        verdict = "PASS_STRONG_ACTION_CONDITIONED_Q"
    elif pass_lite:
        verdict = "PASS_LITE_ACTION_CONDITIONED_Q"
    else:
        verdict = "FAIL_OR_INCONCLUSIVE"

    return {
        "stage": "SEM-6K",
        "mode": "action_conditioned_q_audit",
        "verdict": verdict,
        "pass_lite": pass_lite,
        "pass_strong": pass_strong,
        "n_rows": int(len(df)),
        "action_agnostic_cv_corr": ag_corr,
        "action_conditioned_cv_corr": ac_corr,
        "corr_gain": corr_gain,
        "action_agnostic_cv_r2": ag_report.get("cv_r2", np.nan),
        "action_conditioned_cv_r2": ac_report.get("cv_r2", np.nan),
        "positive_actions": positive_actions,
        "notes": [
            "SEM-6K upgrades Q(E) to Q(E,a).",
            "PASS means action identity improves live value prediction, not that live controller is already validated.",
            "If no action has positive mean, 6K can still pass as a value-model result but needs a new controller primitive.",
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include_6j", default=DEFAULT_6J)
    ap.add_argument("--include_6j3", default=DEFAULT_6J3)
    ap.add_argument("--include_6j3b", default=DEFAULT_6J3B)
    ap.add_argument("--metric", default="")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "6J": args.include_6j,
        "6J3": args.include_6j3,
        "6J3b": args.include_6j3b,
    }
    df = build_dataset(files, args.metric)
    df.to_csv(out_dir / "sem6k_action_conditioned_dataset.csv", index=False, encoding="utf-8-sig")

    action_summary = summarize_actions(df)
    action_summary.to_csv(out_dir / "sem6k_action_value_summary.csv", index=False, encoding="utf-8-sig")

    pred_ag, report_ag = cv_regression(df, action_conditioned=False)
    pred_ac, report_ac = cv_regression(df, action_conditioned=True)

    # Save predictions aligned with dropped rows? cv_regression currently drops none except target.
    pred_df = df.dropna(subset=["target_value"]).copy()
    pred_df["pred_action_agnostic"] = pred_ag
    pred_df["pred_action_conditioned"] = pred_ac
    pred_df.to_csv(out_dir / "sem6k_predictions.csv", index=False, encoding="utf-8-sig")

    best = counterfactual_action_table(df, report_ac, out_dir)

    report = {
        "action_agnostic": report_ag,
        "action_conditioned": report_ac,
        "action_summary": action_summary.to_dict("records"),
    }
    with open(out_dir / "sem6k_model_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    verdict = make_verdict(df, report_ag, report_ac, action_summary)
    verdict["outputs"] = {
        "dataset": str(out_dir / "sem6k_action_conditioned_dataset.csv"),
        "action_summary": str(out_dir / "sem6k_action_value_summary.csv"),
        "model_report": str(out_dir / "sem6k_model_report.json"),
        "predictions": str(out_dir / "sem6k_predictions.csv"),
        "policy_recommendations": str(out_dir / "sem6k_policy_recommendations.csv"),
        "counterfactual_action_table": str(out_dir / "sem6k_counterfactual_action_table.csv"),
    }

    with open(out_dir / "sem6k_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)

    print("\n========== SEM-6K ==========")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    print("\nAction summary:")
    print(action_summary.to_string(index=False))
    if best is not None and not best.empty:
        print("\nTop policy recommendations:")
        print(best.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
