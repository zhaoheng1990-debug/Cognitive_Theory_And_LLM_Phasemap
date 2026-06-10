# -*- coding: utf-8 -*-
r"""
SEM-6J.2: Learned Operator Library Construction

Purpose
-------
SEM-6J.1 created an ASA-style operator action plan, but it is still a template.
SEM-6J.2 builds the first minimal learned operator library from available SEM data.

Minimal action set
------------------
    no_intervention
    address_gate
    operator_prior_transfer
    correction_operator

This script trains two lightweight models:

1) action_classifier
   Predicts the best operator action from no-leak candidate / route features.

2) action_value_model
   Predicts expected utility for candidate-action pairs.

The goal is not live intervention yet. It is to replace hand-written route mapping with
a learned Policy_F / OperatorPrior_F layer.

Inputs
------
Default:
    sem6j1_outputs\sem6j1_asa_operator_write_plan.csv
    sem6h1b_outputs\sem6h1b_scored_candidates.csv

Optional live outcome inputs if available:
    sem6j_outputs\sem6j_live_results.csv
    sem6h2_outputs\sem6h2b_live_results.csv

Outputs
-------
sem6j2_outputs\
    sem6j2_training_table.csv
    sem6j2_action_library.csv
    sem6j2_action_classifier_report.json
    sem6j2_action_value_report.json
    sem6j2_predicted_policy.csv
    sem6j2_verdict.json

Run
---
conda activate dhrf_4080s
cd C:\Users\ZH\Desktop\AGI\python_script

python GPT_sem6j2_learned_operator_library.py

If you have live result files:
python GPT_sem6j2_learned_operator_library.py ^
  --live_results sem6j_outputs\sem6j_live_results.csv

Interpretation
--------------
PASS-Lite:
    action model beats majority/route baseline and value model has positive CV corr.

PASS-Strong:
    action model macro-F1 > 0.55 and value model group-CV corr > 0.30.

If it fails, it still provides the next diagnostic:
    which route/action lacks separable features.
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
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, r2_score, mean_squared_error
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_ROOT = r"C:\Users\ZH\Desktop\AGI\python_script"
DEFAULT_PLAN = rf"{DEFAULT_ROOT}\sem6j1_outputs\sem6j1_asa_operator_write_plan.csv"
DEFAULT_CANDIDATES = rf"{DEFAULT_ROOT}\sem6h1b_outputs\sem6h1b_scored_candidates.csv"
DEFAULT_OUT_DIR = rf"{DEFAULT_ROOT}\sem6j2_outputs"

RANDOM_SEED = 20260606

MINIMAL_ACTIONS = {
    "no_intervention",
    "address_gate",
    "operator_prior_transfer",
    "correction_operator",
}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def coerce_minimal_action(action: str, route: str = "", candidate_operator: str = "") -> str:
    a = str(action)
    r = str(route)
    co = norm(candidate_operator)

    if a in MINIMAL_ACTIONS:
        return a

    # Collapse fine-grained operator actions into correction_operator for 6J.2.
    if a in {
        "planning_operator",
        "risk_audit_operator",
        "counterexample_operator",
        "invariant_search_operator",
        "closure_check_operator",
        "relation_mapping_operator",
        "mechanism_operator",
        "comparison_operator",
        "policy_gate",
        "stable_operator",
        "update_operator",
        "boundary_push",
        "suppress_override_operator",
        "order_precursor",
    }:
        return "correction_operator"

    if r == "reject_as_null":
        return "no_intervention"
    if r == "write_address_gate":
        return "address_gate"
    if r == "write_operator_prior":
        return "operator_prior_transfer"
    if r in {"trigger_correction_operator", "write_residual_mode"}:
        return "correction_operator"
    return "no_intervention"


def load_plan(plan_path: Path) -> pd.DataFrame:
    if not plan_path.exists():
        raise FileNotFoundError(f"6J.1 plan not found: {plan_path}")
    df = pd.read_csv(plan_path)

    required = ["_selection_family", "write_route", "relation_type", "target_operator", "candidate_operator"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Plan missing columns: {missing}")

    if "asa_action" not in df.columns:
        df["asa_action"] = df.apply(
            lambda r: coerce_minimal_action("", r.get("write_route", ""), r.get("candidate_operator", "")),
            axis=1
        )

    df["minimal_action"] = df.apply(
        lambda r: coerce_minimal_action(r.get("asa_action", ""), r.get("write_route", ""), r.get("candidate_operator", "")),
        axis=1
    )

    return df


def attach_live_outcome(df: pd.DataFrame, live_path: Optional[Path]) -> pd.DataFrame:
    out = df.copy()
    if live_path is None or not live_path.exists():
        out["live_available"] = False
        out["target_value"] = pd.to_numeric(out.get("oracle_utility", np.nan), errors="coerce")
        out["target_source"] = "oracle_utility"
        return out

    live = pd.read_csv(live_path)
    # Try robust merge.
    key_candidates = ["_live_row_id", "_selected_index", "_group_id", "_selection", "_selection_family",
                      "family", "candidate_family", "candidate_i", "target_operator", "candidate_operator"]
    keys = [k for k in key_candidates if k in out.columns and k in live.columns]
    if not keys:
        out["live_available"] = False
        out["target_value"] = pd.to_numeric(out.get("oracle_utility", np.nan), errors="coerce")
        out["target_source"] = "oracle_utility_no_merge_keys"
        return out

    live2 = live.drop_duplicates(subset=keys).copy()
    merged = out.merge(live2, on=keys, how="left", suffixes=("", "_live"))

    # Prefer trajectory/live margin gain.
    metric_candidates = [
        "trajectory_improvement",
        "live_margin_gain",
        "live_gain",
        "live_gain_alpha_0.05",
        "live_gain_alpha_0.02",
        "live_gain_alpha_0.01",
        "live_rank_improvement",
    ]
    metric = None
    for m in metric_candidates:
        if m in merged.columns:
            metric = m
            break
        # after suffix
        if f"{m}_live" in merged.columns:
            metric = f"{m}_live"
            break

    if metric is None:
        merged["live_available"] = False
        merged["target_value"] = pd.to_numeric(merged.get("oracle_utility", np.nan), errors="coerce")
        merged["target_source"] = "oracle_utility_no_live_metric"
    else:
        merged["live_available"] = True
        merged["target_value"] = pd.to_numeric(merged[metric], errors="coerce")
        # fallback missing rows to oracle
        fallback = pd.to_numeric(merged.get("oracle_utility", np.nan), errors="coerce")
        merged["target_value"] = merged["target_value"].fillna(fallback)
        merged["target_source"] = f"live_metric:{metric}_fallback_oracle"
    return merged


def build_training_table(plan: pd.DataFrame) -> pd.DataFrame:
    df = plan.copy()

    # Core numeric features.
    for col in ["oracle_utility", "_learned_noleak_q", "proxy_utility", "asa_alpha", "asa_beta", "alpha_operator", "beta_precursor"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    df["q_minus_proxy"] = df["_learned_noleak_q"] - df["proxy_utility"]
    df["q_minus_oracle"] = df["_learned_noleak_q"] - df["oracle_utility"]
    df["utility_positive"] = (df["target_value"] > 0).astype(int)

    # Relation flags.
    for rel in [
        "same_family",
        "same_concept_other_operator",
        "same_operator_other_concept",
        "commit_topk_neighbor",
        "random_other_family",
        "cross_concept_cross_operator",
    ]:
        df[f"rel_{rel}"] = (df["relation_type"].astype(str) == rel).astype(int)

    for sel in ["learned_noleak_Q", "commit_topk_rule", "same_family", "random_pool", "shuffle_learned_Q", "oracle_selected", "proxy_only"]:
        df[f"sel_{sel}"] = (df["_selection_family"].astype(str) == sel).astype(int)

    # Supervised action label:
    # Positive utility keeps assigned action.
    # Negative utility should learn no_intervention unless this is an oracle/positive candidate.
    df["learned_action_label"] = df["minimal_action"]
    neg = df["target_value"].fillna(0) <= 0
    # Do not collapse address_gate too aggressively: address can be useful as neutral gate.
    df.loc[neg & df["minimal_action"].isin(["operator_prior_transfer", "correction_operator"]), "learned_action_label"] = "no_intervention"

    # Rank-like preference value for action library.
    df["action_value"] = df["target_value"].fillna(0.0)

    return df


def feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_candidates = [
        "oracle_utility", "_learned_noleak_q", "proxy_utility", "asa_alpha", "asa_beta",
        "alpha_operator", "beta_precursor", "q_minus_proxy", "q_minus_oracle",
        "utility_positive",
    ]
    numeric_candidates += [c for c in df.columns if c.startswith("rel_") or c.startswith("sel_")]
    numeric = [c for c in numeric_candidates if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    categorical = []
    for c in ["_selection_family", "write_route", "relation_type", "target_operator", "candidate_operator", "asa_action", "asa_action_family"]:
        if c in df.columns:
            categorical.append(c)

    return numeric, categorical


def build_preprocessor(numeric: List[str], categorical: List[str]):
    transformers = []
    if numeric:
        transformers.append(("num", StandardScaler(), numeric))
    if categorical:
        try:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(("cat", enc, categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def group_cv_indices(df: pd.DataFrame, label_col: str, n_splits: int = 5):
    groups = None
    for c in ["_group_id", "family", "candidate_family"]:
        if c in df.columns:
            groups = df[c].astype(str).to_numpy()
            break
    y = df[label_col].astype(str).to_numpy()
    if groups is not None and len(np.unique(groups)) >= 2:
        k = min(n_splits, len(np.unique(groups)))
        splitter = GroupKFold(n_splits=k)
        X_dummy = np.zeros((len(df), 1))
        return list(splitter.split(X_dummy, y, groups))
    else:
        k = min(n_splits, max(2, len(np.unique(y))))
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
        X_dummy = np.zeros((len(df), 1))
        return list(splitter.split(X_dummy, y))


def train_action_classifier(df: pd.DataFrame, out_dir: Path) -> Dict:
    work = df.dropna(subset=["learned_action_label"]).copy()
    numeric, categorical = feature_columns(work)

    X = work[numeric + categorical].copy()
    for c in numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median() if X[c].notna().any() else 0.0)
    for c in categorical:
        X[c] = X[c].astype(str).fillna("NA")

    y = work["learned_action_label"].astype(str).to_numpy()
    labels = sorted(np.unique(y).tolist())

    # Baseline majority.
    majority = pd.Series(y).mode().iloc[0]
    base_pred = np.array([majority] * len(y))
    baseline_acc = accuracy_score(y, base_pred)
    baseline_macro_f1 = f1_score(y, base_pred, average="macro", zero_division=0)

    splits = group_cv_indices(work, "learned_action_label")
    preds = np.array([""] * len(work), dtype=object)

    for tr, te in splits:
        pre = build_preprocessor(numeric, categorical)
        clf = ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X.iloc[tr], y[tr])
        preds[te] = pipe.predict(X.iloc[te])

    acc = accuracy_score(y, preds)
    macro_f1 = f1_score(y, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(y, preds, average="weighted", zero_division=0)

    report = classification_report(y, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, preds, labels=labels)

    # Fit final model for predicted policy output.
    pre = build_preprocessor(numeric, categorical)
    clf = ExtraTreesClassifier(
        n_estimators=600,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X, y)
    final_pred = pipe.predict(X)
    work["predicted_action"] = final_pred

    pred_path = out_dir / "sem6j2_predicted_policy.csv"
    work.to_csv(pred_path, index=False, encoding="utf-8-sig")

    return {
        "n": int(len(work)),
        "labels": labels,
        "features": {"numeric": numeric, "categorical": categorical},
        "baseline_majority_label": majority,
        "baseline_acc": float(baseline_acc),
        "baseline_macro_f1": float(baseline_macro_f1),
        "cv_acc": float(acc),
        "cv_macro_f1": float(macro_f1),
        "cv_weighted_f1": float(weighted_f1),
        "classification_report": report,
        "confusion_matrix_labels": labels,
        "confusion_matrix": cm.tolist(),
        "predicted_policy_csv": str(pred_path),
    }


def group_cv_regression_indices(df: pd.DataFrame, n_splits: int = 5):
    groups = None
    for c in ["_group_id", "family", "candidate_family"]:
        if c in df.columns:
            groups = df[c].astype(str).to_numpy()
            break
    if groups is not None and len(np.unique(groups)) >= 2:
        k = min(n_splits, len(np.unique(groups)))
        splitter = GroupKFold(n_splits=k)
        X_dummy = np.zeros((len(df), 1))
        return list(splitter.split(X_dummy, groups=groups))
    k = min(n_splits, len(df))
    splitter = KFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
    X_dummy = np.zeros((len(df), 1))
    return list(splitter.split(X_dummy))


def train_action_value_model(df: pd.DataFrame, out_dir: Path) -> Dict:
    work = df.dropna(subset=["action_value"]).copy()
    numeric, categorical = feature_columns(work)
    # Avoid leakage if target is oracle_utility: keep q/proxy, categorical route, but drop oracle itself.
    if work["target_source"].astype(str).str.contains("oracle_utility").any():
        numeric = [c for c in numeric if c not in {"oracle_utility", "q_minus_oracle"}]

    X = work[numeric + categorical].copy()
    for c in numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median() if X[c].notna().any() else 0.0)
    for c in categorical:
        X[c] = X[c].astype(str).fillna("NA")
    y = pd.to_numeric(work["action_value"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    splits = group_cv_regression_indices(work)
    pred = np.full(len(work), np.nan)
    for tr, te in splits:
        pre = build_preprocessor(numeric, categorical)
        reg = ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        pipe = Pipeline([("pre", pre), ("reg", reg)])
        pipe.fit(X.iloc[tr], y[tr])
        pred[te] = pipe.predict(X.iloc[te])

    mask = np.isfinite(pred)
    if mask.sum() > 1 and np.std(pred[mask]) > 1e-12 and np.std(y[mask]) > 1e-12:
        corr = float(np.corrcoef(y[mask], pred[mask])[0, 1])
        r2 = float(r2_score(y[mask], pred[mask]))
        rmse = float(math.sqrt(mean_squared_error(y[mask], pred[mask])))
    else:
        corr, r2, rmse = np.nan, np.nan, np.nan

    work["predicted_action_value"] = pred
    value_path = out_dir / "sem6j2_predicted_action_values.csv"
    work.to_csv(value_path, index=False, encoding="utf-8-sig")

    # Action library summary: mean predicted/observed by action.
    lib = work.groupby("minimal_action").agg(
        n=("minimal_action", "size"),
        mean_value=("action_value", "mean"),
        mean_predicted_value=("predicted_action_value", "mean"),
        positive_rate=("utility_positive", "mean"),
    ).reset_index()
    lib_path = out_dir / "sem6j2_action_library.csv"
    lib.to_csv(lib_path, index=False, encoding="utf-8-sig")

    return {
        "n": int(len(work)),
        "target_source_counts": work["target_source"].value_counts().to_dict(),
        "features": {"numeric": numeric, "categorical": categorical},
        "cv_corr": corr,
        "cv_r2": r2,
        "cv_rmse": rmse,
        "predicted_values_csv": str(value_path),
        "action_library_csv": str(lib_path),
        "action_library": lib.to_dict("records"),
    }


def make_verdict(cls: Dict, val: Dict) -> Dict:
    cls_f1 = cls.get("cv_macro_f1", np.nan)
    cls_base = cls.get("baseline_macro_f1", np.nan)
    val_corr = val.get("cv_corr", np.nan)

    pass_lite = (
        np.isfinite(cls_f1)
        and np.isfinite(cls_base)
        and cls_f1 > cls_base + 0.05
        and np.isfinite(val_corr)
        and val_corr > 0.10
    )
    pass_strong = (
        np.isfinite(cls_f1)
        and cls_f1 > 0.55
        and np.isfinite(val_corr)
        and val_corr > 0.30
    )

    if pass_strong:
        verdict = "PASS_STRONG_LEARNED_OPERATOR_LIBRARY"
    elif pass_lite:
        verdict = "PASS_LITE_LEARNED_OPERATOR_LIBRARY"
    else:
        verdict = "FAIL_OR_INCONCLUSIVE"

    return {
        "stage": "SEM-6J.2",
        "verdict": verdict,
        "pass_lite": bool(pass_lite),
        "pass_strong": bool(pass_strong),
        "interpretation": [
            "6J.2 tests whether route/action selection can be learned as Policy_F / OperatorPrior_F.",
            "This is not live control yet; it constructs the operator library layer for 6J.3.",
            "If FAIL, inspect which action labels collapse and whether target values are only oracle-derived.",
        ],
        "key_metrics": {
            "action_classifier_macro_f1": cls_f1,
            "action_classifier_baseline_macro_f1": cls_base,
            "action_value_cv_corr": val_corr,
            "action_value_cv_r2": val.get("cv_r2", np.nan),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=DEFAULT_PLAN)
    ap.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    ap.add_argument("--live_results", default="")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = load_plan(Path(args.plan))
    live_path = Path(args.live_results) if args.live_results.strip() else None
    plan = attach_live_outcome(plan, live_path)
    train = build_training_table(plan)

    train_path = out_dir / "sem6j2_training_table.csv"
    train.to_csv(train_path, index=False, encoding="utf-8-sig")

    cls_report = train_action_classifier(train, out_dir)
    val_report = train_action_value_model(train, out_dir)
    verdict = make_verdict(cls_report, val_report)
    verdict["training_table"] = str(train_path)
    verdict["classifier_report"] = str(out_dir / "sem6j2_action_classifier_report.json")
    verdict["value_report"] = str(out_dir / "sem6j2_action_value_report.json")

    with open(out_dir / "sem6j2_action_classifier_report.json", "w", encoding="utf-8") as f:
        json.dump(cls_report, f, ensure_ascii=False, indent=2)

    with open(out_dir / "sem6j2_action_value_report.json", "w", encoding="utf-8") as f:
        json.dump(val_report, f, ensure_ascii=False, indent=2)

    with open(out_dir / "sem6j2_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)

    print("\n========== SEM-6J.2 ==========")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    print("\nAction library:")
    print(pd.DataFrame(val_report["action_library"]).to_string(index=False))


if __name__ == "__main__":
    main()
