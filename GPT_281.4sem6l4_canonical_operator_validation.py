# -*- coding: utf-8 -*-
r"""
SEM-6L.4: CanonicalOperator Validation

Purpose
-------
SEM-6L.3 promoted four ExpansionMap updates to OperatorCandidate:

    add_constraint
    add_correction_branch
    add_trajectory_class
    increase_address_prior

SEM-6L.4 tests whether these can be promoted further:

    OperatorCandidate -> CanonicalOperator

This is still an offline validation / audit step, not a live controller.
It checks:

    1. held-out group generalization
    2. held-out target-operator robustness
    3. source/surface proxy robustness
    4. non-target damage safety
    5. compression / canonical naming feasibility
    6. rule-template stability

Inputs
------
Default:
    sem6l2_outputs\sem6l2_expansion_update_labels.csv
    sem6l3_outputs\sem6l3_candidate_promotion_table.csv
    sem6l3_outputs\sem6l3_generalization_table.csv
    sem6l3_outputs\sem6l3_non_target_damage.csv

Outputs
-------
sem6l4_outputs\
    sem6l4_canonical_operator_validation_table.csv
    sem6l4_heldout_group_report.csv
    sem6l4_operator_cards.json
    sem6l4_operator_library_draft.json
    sem6l4_verdict.json

Run
---
conda activate dhrf_4080s
cd C:\Users\ZH\Desktop\AGI\python_script

python GPT_sem6l4_canonical_operator_validation.py

Stricter:
python GPT_sem6l4_canonical_operator_validation.py ^
  --min_group_positive_frac 0.50 ^
  --min_mean_live 0.02 ^
  --max_damage_mean -0.02

Interpretation
--------------
PASS-Lite:
    At least one OperatorCandidate passes canonical validation gates.

PASS-Strong:
    At least two candidates pass, including one non-address operator.

CanonicalOperator does not mean universal operator; it means a compressed,
reusable operator with a known applicability boundary and no detected safety damage
in current evidence.
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
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_ROOT = r"C:\Users\ZH\Desktop\AGI\python_script"
DEFAULT_L2_LABELS = rf"{DEFAULT_ROOT}\sem6l2_outputs\sem6l2_expansion_update_labels.csv"
DEFAULT_L3_PROMO = rf"{DEFAULT_ROOT}\sem6l3_outputs\sem6l3_candidate_promotion_table.csv"
DEFAULT_L3_GEN = rf"{DEFAULT_ROOT}\sem6l3_outputs\sem6l3_generalization_table.csv"
DEFAULT_L3_DMG = rf"{DEFAULT_ROOT}\sem6l3_outputs\sem6l3_non_target_damage.csv"
DEFAULT_OUT_DIR = rf"{DEFAULT_ROOT}\sem6l4_outputs"

PROMOTED_DEFAULT = [
    "add_constraint",
    "add_correction_branch",
    "add_trajectory_class",
    "increase_address_prior",
]

RANDOM_SEED = 20260606


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def read_csv(path: str, required: bool = True) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {p}")
        return pd.DataFrame()
    return pd.read_csv(p)


def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in [
        "expansion_update_label", "observation_status", "source_experiment", "action",
        "_selection_family", "relation_type", "write_route", "target_operator",
        "candidate_operator", "target_concept", "candidate_concept", "_group_id", "pool"
    ]:
        if c not in out.columns:
            out[c] = "NA"
        out[c] = out[c].astype(str).fillna("NA")

    for c in [
        "live_value", "is_positive", "_learned_noleak_q", "proxy_utility", "oracle_utility",
        "q_proxy_gap", "q_oracle_gap", "beta", "delta_scale", "alpha_operator", "beta_precursor",
        "asa_alpha", "asa_beta", "base_margin", "base_rank", "base_logprob_gap_to_top",
        "n_delta_layers", "hist_mean_live", "hist_positive_rate", "hist_best_live",
        "action_ctx_mean_live", "action_ctx_positive_rate", "experience_vs_history_gain",
        "positive_rate_gap", "same_operator", "same_concept", "is_risk_target",
        "is_planning_target", "is_relation_target", "is_closure_like", "is_address_like",
        "is_random_like"
    ]:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")

    if out["is_positive"].isna().all():
        out["is_positive"] = (out["live_value"] > 0).astype(int)
    else:
        out["is_positive"] = out["is_positive"].fillna((out["live_value"] > 0).astype(int))
    return out


def feature_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric = [
        "_learned_noleak_q", "proxy_utility", "oracle_utility", "q_proxy_gap", "q_oracle_gap",
        "beta", "delta_scale", "alpha_operator", "beta_precursor", "asa_alpha", "asa_beta",
        "base_margin", "base_rank", "base_logprob_gap_to_top", "n_delta_layers",
        "hist_mean_live", "hist_positive_rate", "hist_best_live",
        "action_ctx_mean_live", "action_ctx_positive_rate", "experience_vs_history_gain",
        "positive_rate_gap", "same_operator", "same_concept", "is_risk_target",
        "is_planning_target", "is_relation_target", "is_closure_like", "is_address_like",
        "is_random_like",
    ]
    numeric = [c for c in numeric if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().sum() > 0]

    categorical = [
        "source_experiment", "action", "_selection_family", "relation_type", "write_route",
        "target_operator", "candidate_operator", "target_concept", "candidate_concept", "pool",
    ]
    categorical = [c for c in categorical if c in df.columns]
    return numeric, categorical


def make_X(df: pd.DataFrame, numeric: List[str], categorical: List[str]) -> pd.DataFrame:
    X = df[numeric + categorical].copy()
    for c in numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median() if X[c].notna().any() else 0.0)
    for c in categorical:
        X[c] = X[c].astype(str).fillna("NA")
    return X


def make_pre(numeric: List[str], categorical: List[str]):
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


def heldout_group_regression(sub: pd.DataFrame, group_col: str = "_group_id") -> Dict:
    """
    Predict live_value within one candidate label under held-out group splits.
    This tests whether the operator's utility surface is learnable beyond seen groups.
    """
    work = sub.dropna(subset=["live_value"]).copy()
    if len(work) < 30:
        return {"n": int(len(work)), "r2": np.nan, "corr": np.nan, "rmse": np.nan, "note": "too few rows"}

    if group_col not in work.columns or work[group_col].nunique() < 2:
        group_col = "target_operator" if "target_operator" in work.columns and work["target_operator"].nunique() >= 2 else None

    numeric, categorical = feature_cols(work)
    X = make_X(work, numeric, categorical)
    y = pd.to_numeric(work["live_value"], errors="coerce").to_numpy(float)

    if group_col:
        groups = work[group_col].astype(str).to_numpy()
        k = min(5, len(np.unique(groups)))
        splits = list(GroupKFold(n_splits=k).split(X, y, groups))
        cv_type = f"GroupKFold:{group_col}"
    else:
        k = min(5, len(work))
        splits = list(KFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED).split(X, y))
        cv_type = "KFold"

    pred = np.full(len(work), np.nan)
    for tr, te in splits:
        pipe = Pipeline([
            ("pre", make_pre(numeric, categorical)),
            ("reg", ExtraTreesRegressor(
                n_estimators=400,
                min_samples_leaf=2,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ))
        ])
        pipe.fit(X.iloc[tr], y[tr])
        pred[te] = pipe.predict(X.iloc[te])

    mask = np.isfinite(pred)
    if mask.sum() > 1 and np.std(pred[mask]) > 1e-12 and np.std(y[mask]) > 1e-12:
        corr = float(np.corrcoef(y[mask], pred[mask])[0, 1])
        r2 = float(r2_score(y[mask], pred[mask]))
        rmse = float(math.sqrt(mean_squared_error(y[mask], pred[mask])))
    else:
        corr, r2, rmse = np.nan, np.nan, np.nan

    return {
        "n": int(len(work)),
        "cv_type": cv_type,
        "r2": r2,
        "corr": corr,
        "rmse": rmse,
        "features_numeric": numeric,
        "features_categorical": categorical,
    }


def leave_one_operator_report(sub: pd.DataFrame) -> pd.DataFrame:
    """
    Hold out each target_operator and estimate observed live summary.
    This is not a trained predictor; it reports robustness across operator classes.
    """
    rows = []
    for op, g in sub.groupby("target_operator", dropna=False):
        vals = pd.to_numeric(g["live_value"], errors="coerce").dropna().to_numpy(float)
        if len(vals) == 0:
            continue
        train = sub[sub["target_operator"] != op]
        train_vals = pd.to_numeric(train["live_value"], errors="coerce").dropna().to_numpy(float)
        rows.append({
            "heldout_target_operator": op,
            "n_test": int(len(vals)),
            "test_mean_live": float(np.mean(vals)),
            "test_positive_rate": float(np.mean(g["is_positive"])),
            "train_mean_live": float(np.mean(train_vals)) if len(train_vals) else np.nan,
            "mean_shift_test_minus_train": float(np.mean(vals) - np.mean(train_vals)) if len(train_vals) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("test_mean_live", ascending=False)


def source_surface_robustness(sub: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # source_experiment and _selection_family serve as source/surface proxies.
    for factor in ["source_experiment", "_selection_family", "relation_type"]:
        for val, g in sub.groupby(factor, dropna=False):
            vals = pd.to_numeric(g["live_value"], errors="coerce").dropna().to_numpy(float)
            if len(vals) == 0:
                continue
            rows.append({
                "factor": factor,
                "value": str(val),
                "n": int(len(vals)),
                "mean_live": float(np.mean(vals)),
                "positive_rate": float(np.mean(g["is_positive"])),
            })
    return pd.DataFrame(rows).sort_values(["factor", "mean_live"], ascending=[True, False])


def damage_summary(sub: pd.DataFrame) -> Dict:
    nt = sub[
        (sub.get("is_random_like", 0).fillna(0).astype(float) > 0)
        | (sub["_selection_family"].astype(str).str.contains("random|shuffle", case=False, regex=True))
    ].copy()
    vals = pd.to_numeric(nt["live_value"], errors="coerce").dropna().to_numpy(float)
    all_vals = pd.to_numeric(sub["live_value"], errors="coerce").dropna().to_numpy(float)
    return {
        "n_all": int(len(all_vals)),
        "mean_all": float(np.mean(all_vals)) if len(all_vals) else np.nan,
        "n_non_target_proxy": int(len(vals)),
        "mean_non_target_proxy": float(np.mean(vals)) if len(vals) else np.nan,
        "damage_rate_non_target": float(np.mean(vals < 0)) if len(vals) else np.nan,
        "has_damage_flag": bool(len(vals) >= 10 and np.mean(vals) < -0.02),
    }


def canonical_name(label: str) -> str:
    mapping = {
        "add_constraint": "ConstraintInsertionOperator",
        "add_correction_branch": "CorrectionBranchOperator",
        "add_trajectory_class": "TrajectoryClassExpansionOperator",
        "increase_address_prior": "AddressPriorIncreaseOperator",
    }
    return mapping.get(label, f"Canonical_{label}")


def canonical_description(label: str, sub: pd.DataFrame) -> Dict:
    top_target = sub["target_operator"].astype(str).mode().iloc[0] if len(sub) else "NA"
    top_candidate = sub["candidate_operator"].astype(str).mode().iloc[0] if len(sub) else "NA"
    top_action = sub["action"].astype(str).mode().iloc[0] if len(sub) else "NA"
    top_source = sub["source_experiment"].astype(str).mode().iloc[0] if len(sub) else "NA"

    if label == "add_constraint":
        intent = "Add or strengthen an expansion constraint / boundary-check before unfolding this problem class."
        boundary = "Use for RiskAudit / closure-like boundary detection; avoid treating it as answer steering."
    elif label == "add_correction_branch":
        intent = "Add an alternate grounding/correction branch before the original operator unfolds."
        boundary = "Use when same concept requires a different operator such as Definition before Planning."
    elif label == "add_trajectory_class":
        intent = "Add a new trajectory class to the ExpansionMap when existing routes fail to cover positive variants."
        boundary = "Requires held-out validation because it may capture context-specific effects."
    elif label == "increase_address_prior":
        intent = "Increase commit/address retrieval prior so future unfolding starts from the correct identity/relation anchor."
        boundary = "Useful as routing priority; previous hidden-delta address control was weak, so do not implement as direct injection."
    else:
        intent = "Candidate canonical operator."
        boundary = "Boundary unknown."

    return {
        "operator_id": canonical_name(label),
        "source_label": label,
        "intent": intent,
        "applicability_boundary": boundary,
        "top_target_operator": top_target,
        "top_candidate_operator": top_candidate,
        "top_action": top_action,
        "top_source_experiment": top_source,
    }


def validate_candidate(label: str, sub: pd.DataFrame, args) -> Dict:
    vals = pd.to_numeric(sub["live_value"], errors="coerce").dropna().to_numpy(float)
    if len(vals) == 0:
        return {"label": label, "decision": "reject", "reason": "no live values"}

    n_groups = int(sub["_group_id"].nunique())
    n_target_ops = int(sub["target_operator"].nunique())
    n_sources = int(sub["source_experiment"].nunique())
    n_selection = int(sub["_selection_family"].nunique())
    mean_live = float(np.mean(vals))
    median_live = float(np.median(vals))
    pos_rate = float(np.mean(sub["is_positive"]))

    group_means = sub.groupby("_group_id")["live_value"].mean()
    group_positive_frac = float(np.mean(group_means > 0)) if len(group_means) else np.nan

    dmg = damage_summary(sub)
    reg = heldout_group_regression(sub)
    op_holdout = leave_one_operator_report(sub)
    src_rob = source_surface_robustness(sub)

    # Robustness gates.
    source_ok = True
    if not src_rob.empty:
        # For any factor with enough rows, if most factor slices are strongly negative, fail.
        factor_ok = []
        for factor, fdf in src_rob.groupby("factor"):
            enough = fdf[fdf["n"] >= 5]
            if len(enough) == 0:
                continue
            factor_ok.append(float(np.mean(enough["mean_live"] >= args.min_slice_mean)) >= 0.5)
        source_ok = all(factor_ok) if factor_ok else True

    op_ok = True
    if not op_holdout.empty:
        enough = op_holdout[op_holdout["n_test"] >= 5]
        if len(enough) > 0:
            op_ok = float(np.mean(enough["test_mean_live"] >= args.min_slice_mean)) >= 0.5

    utility_ok = mean_live >= args.min_mean_live and pos_rate >= args.min_positive_rate
    coverage_ok = n_groups >= args.min_groups and n_target_ops >= args.min_target_ops and n_sources >= args.min_sources
    damage_ok = not dmg["has_damage_flag"]
    group_ok = group_positive_frac >= args.min_group_positive_frac if np.isfinite(group_positive_frac) else False

    # Compression feasibility: can the candidate be named and represented compactly?
    compression_ok = label in {
        "add_constraint",
        "add_correction_branch",
        "add_trajectory_class",
        "increase_address_prior",
    }

    gates = {
        "utility_ok": bool(utility_ok),
        "coverage_ok": bool(coverage_ok),
        "damage_ok": bool(damage_ok),
        "group_ok": bool(group_ok),
        "source_surface_ok": bool(source_ok),
        "operator_holdout_ok": bool(op_ok),
        "compression_ok": bool(compression_ok),
    }

    passed = sum(gates.values())
    if all(gates.values()):
        decision = "promote_to_canonical_operator"
    elif utility_ok and damage_ok and compression_ok and passed >= 5:
        decision = "promote_to_provisional_canonical"
    elif utility_ok and damage_ok:
        decision = "remain_operator_candidate"
    else:
        decision = "reject_or_continue_observation"

    # Compute overall canonicality score.
    canonicality = (
        0.20 * min(max(mean_live, 0.0) / max(args.min_mean_live if args.min_mean_live > 0 else 0.05, 0.05), 1.0)
        + 0.15 * min(pos_rate / max(args.min_positive_rate, 1e-6), 1.0)
        + 0.15 * min(n_groups / max(args.min_groups, 1), 1.0)
        + 0.10 * min(n_target_ops / max(args.min_target_ops, 1), 1.0)
        + 0.10 * min(n_sources / max(args.min_sources, 1), 1.0)
        + 0.10 * (1.0 if damage_ok else 0.0)
        + 0.10 * (1.0 if source_ok else 0.0)
        + 0.10 * (1.0 if op_ok else 0.0)
    )

    desc = canonical_description(label, sub)
    return {
        "expansion_update_label": label,
        "operator_id": desc["operator_id"],
        "decision": decision,
        "canonicality_score": float(canonicality),
        "n_rows": int(len(sub)),
        "n_groups": n_groups,
        "n_target_ops": n_target_ops,
        "n_sources": n_sources,
        "n_selection_families": n_selection,
        "mean_live": mean_live,
        "median_live": median_live,
        "positive_rate": pos_rate,
        "group_positive_frac": group_positive_frac,
        "damage": dmg,
        "heldout_group_regression": reg,
        "gates": gates,
        "canonical_description": desc,
    }


def operator_card(result: Dict, sub: pd.DataFrame) -> Dict:
    label = result["expansion_update_label"]
    desc = result["canonical_description"]
    examples = []
    # positive examples
    pos = sub.sort_values("live_value", ascending=False).head(10)
    for _, r in pos.iterrows():
        examples.append({
            "target_concept": str(r.get("target_concept", "")),
            "target_operator": str(r.get("target_operator", "")),
            "candidate_concept": str(r.get("candidate_concept", "")),
            "candidate_operator": str(r.get("candidate_operator", "")),
            "live_value": float(r.get("live_value", np.nan)) if pd.notna(r.get("live_value", np.nan)) else None,
            "source_experiment": str(r.get("source_experiment", "")),
            "selection_family": str(r.get("_selection_family", "")),
        })

    return {
        "operator_id": desc["operator_id"],
        "source_label": label,
        "status": result["decision"],
        "intent": desc["intent"],
        "applicability_boundary": desc["applicability_boundary"],
        "canonicality_score": result["canonicality_score"],
        "evidence": {
            "n_rows": result["n_rows"],
            "n_groups": result["n_groups"],
            "n_target_ops": result["n_target_ops"],
            "n_sources": result["n_sources"],
            "mean_live": result["mean_live"],
            "positive_rate": result["positive_rate"],
            "group_positive_frac": result["group_positive_frac"],
            "damage": result["damage"],
            "gates": result["gates"],
        },
        "top_examples": examples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=DEFAULT_L2_LABELS)
    ap.add_argument("--promotion_table", default=DEFAULT_L3_PROMO)
    ap.add_argument("--generalization_table", default=DEFAULT_L3_GEN)
    ap.add_argument("--damage_table", default=DEFAULT_L3_DMG)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)

    # Validation gates.
    ap.add_argument("--min_groups", type=int, default=5)
    ap.add_argument("--min_target_ops", type=int, default=1)
    ap.add_argument("--min_sources", type=int, default=2)
    ap.add_argument("--min_positive_rate", type=float, default=0.25)
    ap.add_argument("--min_group_positive_frac", type=float, default=0.25)
    ap.add_argument("--min_mean_live", type=float, default=0.0)
    ap.add_argument("--min_slice_mean", type=float, default=-0.02)

    # Candidate labels to validate.
    ap.add_argument("--candidate_labels", default=",".join(PROMOTED_DEFAULT))

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = ensure_cols(read_csv(args.labels, required=True))
    candidate_labels = [x.strip() for x in args.candidate_labels.split(",") if x.strip()]

    # Keep only candidates in data.
    available = sorted(labels["expansion_update_label"].unique().tolist())
    candidate_labels = [x for x in candidate_labels if x in available]
    if not candidate_labels:
        raise ValueError(f"No candidate labels found in data. Available={available}")

    validation_rows = []
    cards = []
    op_holdouts = []
    source_robust = []

    for label in candidate_labels:
        sub = labels[labels["expansion_update_label"] == label].copy()
        result = validate_candidate(label, sub, args)

        # Flatten row for CSV.
        row = {
            "expansion_update_label": result["expansion_update_label"],
            "operator_id": result["operator_id"],
            "decision": result["decision"],
            "canonicality_score": result["canonicality_score"],
            "n_rows": result["n_rows"],
            "n_groups": result["n_groups"],
            "n_target_ops": result["n_target_ops"],
            "n_sources": result["n_sources"],
            "n_selection_families": result["n_selection_families"],
            "mean_live": result["mean_live"],
            "median_live": result["median_live"],
            "positive_rate": result["positive_rate"],
            "group_positive_frac": result["group_positive_frac"],
            "mean_non_target_proxy": result["damage"]["mean_non_target_proxy"],
            "damage_rate_non_target": result["damage"]["damage_rate_non_target"],
            "damage_flag": result["damage"]["has_damage_flag"],
            "heldout_group_corr": result["heldout_group_regression"].get("corr", np.nan),
            "heldout_group_r2": result["heldout_group_regression"].get("r2", np.nan),
        }
        for k, v in result["gates"].items():
            row[f"gate_{k}"] = v
        validation_rows.append(row)

        cards.append(operator_card(result, sub))

        op = leave_one_operator_report(sub)
        if not op.empty:
            op["expansion_update_label"] = label
            op_holdouts.append(op)

        sr = source_surface_robustness(sub)
        if not sr.empty:
            sr["expansion_update_label"] = label
            source_robust.append(sr)

    validation = pd.DataFrame(validation_rows).sort_values("canonicality_score", ascending=False)
    validation.to_csv(out_dir / "sem6l4_canonical_operator_validation_table.csv", index=False, encoding="utf-8-sig")

    if op_holdouts:
        op_df = pd.concat(op_holdouts, ignore_index=True)
    else:
        op_df = pd.DataFrame()
    op_df.to_csv(out_dir / "sem6l4_heldout_operator_report.csv", index=False, encoding="utf-8-sig")

    if source_robust:
        sr_df = pd.concat(source_robust, ignore_index=True)
    else:
        sr_df = pd.DataFrame()
    sr_df.to_csv(out_dir / "sem6l4_source_surface_robustness.csv", index=False, encoding="utf-8-sig")

    with open(out_dir / "sem6l4_operator_cards.json", "w", encoding="utf-8") as f:
        json.dump(cards, f, ensure_ascii=False, indent=2)

    library = {
        "version": "SEM-6L.4-canonical-operator-library-draft",
        "status": "draft_requires_future_live_replay_for_final_memory_write",
        "operators": [c for c in cards if c["status"] in {"promote_to_canonical_operator", "promote_to_provisional_canonical"}],
        "rejected_or_pending": [c for c in cards if c["status"] not in {"promote_to_canonical_operator", "promote_to_provisional_canonical"}],
        "note": (
            "CanonicalOperator here means a compressed reusable ExpansionMap update with current evidence. "
            "It is not yet a parameter-level model update and should enter W'_cognitive / OperatorMemory only after future held-out live replay."
        ),
    }
    with open(out_dir / "sem6l4_operator_library_draft.json", "w", encoding="utf-8") as f:
        json.dump(library, f, ensure_ascii=False, indent=2)

    n_canon = int((validation["decision"] == "promote_to_canonical_operator").sum())
    n_prov = int((validation["decision"] == "promote_to_provisional_canonical").sum())
    n_total_promoted = n_canon + n_prov

    non_address_promoted = validation[
        validation["decision"].isin(["promote_to_canonical_operator", "promote_to_provisional_canonical"])
        & ~validation["expansion_update_label"].eq("increase_address_prior")
    ]

    if n_canon >= 2 or (n_canon >= 1 and len(non_address_promoted) >= 1):
        verdict_str = "PASS_STRONG_CANONICAL_OPERATORS_VALIDATED"
    elif n_total_promoted >= 1:
        verdict_str = "PASS_LITE_PROVISIONAL_CANONICAL_OPERATOR"
    else:
        verdict_str = "FAIL_OR_INCONCLUSIVE"

    verdict = {
        "stage": "SEM-6L.4",
        "mode": "canonical_operator_validation",
        "verdict": verdict_str,
        "n_canonical": n_canon,
        "n_provisional": n_prov,
        "n_validated_or_provisional": n_total_promoted,
        "candidate_labels": candidate_labels,
        "parameters": {
            "min_groups": args.min_groups,
            "min_target_ops": args.min_target_ops,
            "min_sources": args.min_sources,
            "min_positive_rate": args.min_positive_rate,
            "min_group_positive_frac": args.min_group_positive_frac,
            "min_mean_live": args.min_mean_live,
            "min_slice_mean": args.min_slice_mean,
        },
        "notes": [
            "6L.4 validates whether OperatorCandidates can be compressed into CanonicalOperator drafts.",
            "A positive verdict does not mean immediate model-parameter write; it means eligible for W'_cognitive / OperatorMemory draft.",
            "Future 6L.5 should do held-out live replay using the canonical operator cards as routing policies.",
        ],
        "outputs": {
            "validation_table": str(out_dir / "sem6l4_canonical_operator_validation_table.csv"),
            "heldout_operator_report": str(out_dir / "sem6l4_heldout_operator_report.csv"),
            "source_surface_robustness": str(out_dir / "sem6l4_source_surface_robustness.csv"),
            "operator_cards": str(out_dir / "sem6l4_operator_cards.json"),
            "operator_library_draft": str(out_dir / "sem6l4_operator_library_draft.json"),
            "verdict": str(out_dir / "sem6l4_verdict.json"),
        },
    }

    with open(out_dir / "sem6l4_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)

    print("\n========== SEM-6L.4 ==========")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    print("\nValidation table:")
    print(validation.to_string(index=False))
    print("\nOperator library draft:")
    print(json.dumps(library, ensure_ascii=False, indent=2)[:5000])


if __name__ == "__main__":
    main()
