# -*- coding: utf-8 -*-
r"""
SEM-6L.2: Experience–History Contrast / ExpansionMap Update Audit

Purpose
-------
SEM-6L.1 found strong structure in positive live deltas:

    PASS_STRONG_POSITIVE_PROTOTYPES_FOUND
    positive classifier AUC ≈ 0.922

But the theoretical object has now shifted:

    MemorySeed = anchor
    ExpansionMap = how the seed unfolds
    ReflexiveUpdate acts mainly on ExpansionMap

So SEM-6L.2 does NOT search for a better hidden write vector.
It constructs an Experience–History Contrast table and assigns candidate
ExpansionMap update labels:

    increase_address_prior
    update_operator_prior
    add_correction_branch
    add_constraint
    add_relation_edge
    add_trajectory_class
    split_context
    keep_existing_route
    reject_as_artifact

Then it trains/audits whether these routing-update labels are predictable from
contrast features, and which labels are eligible for an observation period.

Inputs
------
Default:
    sem6l1_outputs\sem6l1_all_live_rows.csv
    sem6l1_outputs\sem6l1_positive_enrichment.csv
    sem6l1_outputs\sem6l1_numeric_contrasts.csv
    sem6l1_outputs\sem6l1_positive_cluster_summary.csv
    sem6l1_outputs\sem6l1_prototype_rules.csv

Outputs
-------
sem6l2_outputs\
    sem6l2_experience_history_contrast_table.csv
    sem6l2_expansion_update_labels.csv
    sem6l2_update_label_summary.csv
    sem6l2_observation_candidates.csv
    sem6l2_update_classifier_report.json
    sem6l2_verdict.json

Run
---
conda activate dhrf_4080s
cd C:\Users\ZH\Desktop\AGI\python_script

python GPT_sem6l2_experience_history_contrast_audit.py

Optional stronger positive threshold:
python GPT_sem6l2_experience_history_contrast_audit.py --positive_threshold 0.01

Interpretation
--------------
PASS-Lite:
    update labels are not degenerate and produce observation candidates.

PASS-Strong:
    update labels are predictable under group CV and multiple candidate patches
    pass observation-period entry criteria.

This is NOT a live controller. It is the routing-map update construction step.
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
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_ROOT = r"C:\Users\ZH\Desktop\AGI\python_script"
DEFAULT_L1_DIR = rf"{DEFAULT_ROOT}\sem6l1_outputs"
DEFAULT_ALL_ROWS = rf"{DEFAULT_L1_DIR}\sem6l1_all_live_rows.csv"
DEFAULT_POS_ENRICH = rf"{DEFAULT_L1_DIR}\sem6l1_positive_enrichment.csv"
DEFAULT_NUM_CONTRASTS = rf"{DEFAULT_L1_DIR}\sem6l1_numeric_contrasts.csv"
DEFAULT_CLUSTER_SUMMARY = rf"{DEFAULT_L1_DIR}\sem6l1_positive_cluster_summary.csv"
DEFAULT_RULES = rf"{DEFAULT_L1_DIR}\sem6l1_prototype_rules.csv"

DEFAULT_OUT_DIR = rf"{DEFAULT_ROOT}\sem6l2_outputs"

RANDOM_SEED = 20260606


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def safe_str(x) -> str:
    if pd.isna(x):
        return "NA"
    return str(x)


def load_csv(path: str, required: bool = False) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {p}")
        return pd.DataFrame()
    return pd.read_csv(p)


def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in [
        "source_experiment", "action", "_selection_family", "relation_type", "write_route",
        "target_operator", "candidate_operator", "target_concept", "candidate_concept", "pool",
        "_group_id"
    ]:
        if c not in out.columns:
            out[c] = "NA"
        out[c] = out[c].astype(str).fillna("NA")

    # Numeric defaults.
    for c in [
        "live_value", "_learned_noleak_q", "proxy_utility", "oracle_utility",
        "q_proxy_gap", "q_oracle_gap", "beta", "delta_scale", "alpha_operator",
        "beta_precursor", "asa_alpha", "asa_beta", "base_margin", "base_rank",
        "base_logprob_gap_to_top", "live_rank_improvement", "n_delta_layers"
    ]:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")

    if "is_positive" not in out.columns:
        out["is_positive"] = (out["live_value"] > 0).astype(int)
    return out


def family_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Historical memory proxy: per group/family statistics.
    This approximates M_F historical baseline from all live rows belonging to the same group.
    """
    group_col = "_group_id" if "_group_id" in df.columns else "target_concept"
    stats = df.groupby(group_col).agg(
        hist_n=("live_value", "size"),
        hist_mean_live=("live_value", "mean"),
        hist_median_live=("live_value", "median"),
        hist_positive_rate=("is_positive", "mean"),
        hist_best_live=("live_value", "max"),
        hist_worst_live=("live_value", "min"),
    ).reset_index().rename(columns={group_col: "_group_id"})
    return stats


def action_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.groupby(["action", "target_operator", "candidate_operator"], dropna=False).agg(
        action_ctx_n=("live_value", "size"),
        action_ctx_mean_live=("live_value", "mean"),
        action_ctx_positive_rate=("is_positive", "mean"),
    ).reset_index()
    return stats


def build_contrast_table(all_rows: pd.DataFrame, positive_threshold: float) -> pd.DataFrame:
    df = ensure_cols(all_rows)
    df["is_positive"] = (df["live_value"] > positive_threshold).astype(int)

    fs = family_stats(df)
    df = df.merge(fs, on="_group_id", how="left")

    acs = action_stats(df)
    df = df.merge(acs, on=["action", "target_operator", "candidate_operator"], how="left")

    # Contrast features: new experience vs historical memory proxy.
    df["experience_vs_history_gain"] = df["live_value"] - df["hist_mean_live"]
    df["experience_vs_best_gap"] = df["live_value"] - df["hist_best_live"]
    df["experience_vs_worst_gap"] = df["live_value"] - df["hist_worst_live"]
    df["positive_rate_gap"] = df["is_positive"] - df["hist_positive_rate"]

    # Concept/operator contrast flags.
    df["same_operator"] = (df["target_operator"] == df["candidate_operator"]).astype(int)
    df["same_concept"] = (df["target_concept"] == df["candidate_concept"]).astype(int)
    df["operator_mismatch"] = 1 - df["same_operator"]
    df["concept_mismatch"] = 1 - df["same_concept"]

    # Operator semantics.
    to = df["target_operator"].astype(str).map(norm)
    co = df["candidate_operator"].astype(str).map(norm)
    relation = df["relation_type"].astype(str).map(norm)
    action = df["action"].astype(str).map(norm)
    source = df["source_experiment"].astype(str).map(norm)
    sel = df["_selection_family"].astype(str).map(norm)

    df["is_risk_target"] = to.str.contains("risk|audit").astype(int)
    df["is_planning_target"] = to.str.contains("planning|plan").astype(int)
    df["is_relation_target"] = to.str.contains("relation").astype(int)
    df["is_closure_like"] = (
        relation.str.contains("closure") |
        df["write_route"].astype(str).map(norm).str.contains("closure|correction") |
        co.str.contains("closure|risk|planning|override|exception")
    ).astype(int)
    df["is_address_like"] = (
        action.str.contains("address_gate") |
        df["write_route"].astype(str).map(norm).str.contains("address")
    ).astype(int)
    df["is_random_like"] = (
        sel.str.contains("random") |
        relation.str.contains("random") |
        df["pool"].astype(str).map(norm).str.contains("random")
    ).astype(int)
    return df


def assign_expansion_update_label(row: pd.Series, positive_threshold: float) -> str:
    """
    Main theoretical labels:
      - increase_address_prior
      - update_operator_prior
      - add_correction_branch
      - add_constraint
      - add_relation_edge
      - add_trajectory_class
      - split_context
      - keep_existing_route
      - reject_as_artifact

    The labels are not final truth; they are candidate routing-map updates for observation period.
    """
    live = row.get("live_value", np.nan)
    positive = bool(pd.notna(live) and float(live) > positive_threshold)

    if not positive:
        # zero/no-intervention can mean keep route; strongly negative / random means reject.
        if safe_str(row.get("action", "")).lower() == "no_intervention" or abs(float(live) if pd.notna(live) else 0.0) < 1e-12:
            return "keep_existing_route"
        return "reject_as_artifact"

    action = norm(row.get("action", ""))
    relation = norm(row.get("relation_type", ""))
    route = norm(row.get("write_route", ""))
    target_op = norm(row.get("target_operator", ""))
    cand_op = norm(row.get("candidate_operator", ""))
    source = norm(row.get("source_experiment", ""))
    selection = norm(row.get("_selection_family", ""))

    # Strong semantic special cases from 6L.1.
    if "risk" in target_op or "audit" in target_op:
        return "add_constraint"

    if "planning" in target_op and "definition" in cand_op:
        return "add_correction_branch"

    if "relation" in target_op and ("address" in action or "address" in route):
        return "increase_address_prior"

    # General structural mapping.
    if "address_gate" in action or "address" in route or "commit_topk" in relation:
        return "increase_address_prior"

    if "same_operator_other_concept" in relation:
        return "update_operator_prior"

    if "same_concept_other_operator" in relation:
        # same concept, different operator means a new unfolding branch.
        if any(x in cand_op for x in ["risk", "closure", "planning", "counter", "invariant", "definition"]):
            return "add_correction_branch"
        return "add_trajectory_class"

    if "closure" in relation or "override" in cand_op or "exception" in cand_op:
        return "add_constraint"

    if "same_family" in relation:
        return "add_relation_edge"

    if "random" in relation or "random" in selection:
        # Positive random rows are not trusted as canonical; they are context splits until replicated.
        return "split_context"

    return "add_trajectory_class"


def add_update_labels(df: pd.DataFrame, positive_threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["expansion_update_label"] = out.apply(lambda r: assign_expansion_update_label(r, positive_threshold), axis=1)

    # Memory level / observation period status.
    # Criteria are intentionally conservative.
    summary = out.groupby("expansion_update_label").agg(
        n=("live_value", "size"),
        mean_live=("live_value", "mean"),
        positive_rate=("is_positive", "mean"),
        n_groups=("_group_id", "nunique"),
        n_target_ops=("target_operator", "nunique"),
    ).reset_index()

    status_map = {}
    for _, r in summary.iterrows():
        label = r["expansion_update_label"]
        n = int(r["n"])
        mean_live = float(r["mean_live"])
        pr = float(r["positive_rate"])
        ng = int(r["n_groups"])
        nt = int(r["n_target_ops"])

        if label in {"reject_as_artifact"}:
            status = "rejected"
        elif label in {"keep_existing_route"}:
            status = "working_patch"
        elif n >= 50 and pr > 0.30 and ng >= 5 and nt >= 2:
            status = "operator_candidate"
        elif n >= 10 and pr > 0.20:
            status = "observation_period"
        else:
            status = "working_patch"
        status_map[label] = status

    out["observation_status"] = out["expansion_update_label"].map(status_map).fillna("working_patch")
    return out


def feature_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric = [
        "_learned_noleak_q", "proxy_utility", "oracle_utility", "q_proxy_gap", "q_oracle_gap",
        "beta", "delta_scale", "alpha_operator", "beta_precursor", "asa_alpha", "asa_beta",
        "base_margin", "base_rank", "base_logprob_gap_to_top", "n_delta_layers",
        "hist_n", "hist_mean_live", "hist_positive_rate", "hist_best_live", "hist_worst_live",
        "action_ctx_n", "action_ctx_mean_live", "action_ctx_positive_rate",
        "experience_vs_history_gain", "experience_vs_best_gap", "experience_vs_worst_gap",
        "positive_rate_gap", "same_operator", "same_concept", "operator_mismatch", "concept_mismatch",
        "is_risk_target", "is_planning_target", "is_relation_target", "is_closure_like", "is_address_like", "is_random_like",
    ]
    numeric = [c for c in numeric if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().sum() > 0]

    categorical = [
        "source_experiment", "action", "_selection_family", "relation_type", "write_route",
        "target_operator", "candidate_operator", "target_concept", "candidate_concept", "pool"
    ]
    categorical = [c for c in categorical if c in df.columns]
    return numeric, categorical


def train_update_classifier(df: pd.DataFrame, out_dir: Path) -> Dict:
    work = df.copy()
    if work["expansion_update_label"].nunique() < 2:
        return {"error": "Only one update label."}

    numeric, categorical = feature_cols(work)
    X = work[numeric + categorical].copy()
    for c in numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median() if X[c].notna().any() else 0.0)
    for c in categorical:
        X[c] = X[c].astype(str).fillna("NA")

    y = work["expansion_update_label"].astype(str).to_numpy()
    labels = sorted(np.unique(y).tolist())

    # Group split.
    groups = None
    for c in ["_group_id", "target_concept", "candidate_concept"]:
        if c in work.columns:
            groups = work[c].astype(str).to_numpy()
            break

    if groups is not None and len(np.unique(groups)) >= 2:
        k = min(5, len(np.unique(groups)))
        splits = list(GroupKFold(n_splits=k).split(X, y, groups))
        cv_type = "GroupKFold"
    else:
        # robust stratified fallback
        min_count = pd.Series(y).value_counts().min()
        k = min(5, int(min_count)) if min_count >= 2 else 2
        splits = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED).split(X, y))
        cv_type = "StratifiedKFold"

    pred = np.array([""] * len(work), dtype=object)
    for tr, te in splits:
        try:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
        pre = ColumnTransformer(
            [("num", StandardScaler(), numeric), ("cat", enc, categorical)],
            remainder="drop",
        )
        clf = ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X.iloc[tr], y[tr])
        pred[te] = pipe.predict(X.iloc[te])

    acc = accuracy_score(y, pred)
    macro_f1 = f1_score(y, pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y, pred, average="weighted", zero_division=0)
    report = classification_report(y, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred, labels=labels)

    pred_df = work.copy()
    pred_df["pred_expansion_update_label"] = pred
    pred_df.to_csv(out_dir / "sem6l2_update_classifier_predictions.csv", index=False, encoding="utf-8-sig")

    # Majority baseline.
    majority = pd.Series(y).mode().iloc[0]
    base_pred = np.array([majority] * len(y))
    base_macro_f1 = f1_score(y, base_pred, average="macro", zero_division=0)

    return {
        "n": int(len(work)),
        "cv_type": cv_type,
        "labels": labels,
        "features": {"numeric": numeric, "categorical": categorical},
        "baseline_majority_label": majority,
        "baseline_macro_f1": float(base_macro_f1),
        "cv_acc": float(acc),
        "cv_macro_f1": float(macro_f1),
        "cv_weighted_f1": float(weighted_f1),
        "classification_report": report,
        "confusion_matrix_labels": labels,
        "confusion_matrix": cm.tolist(),
    }


def summarize_labels(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, sub in df.groupby("expansion_update_label", dropna=False):
        vals = pd.to_numeric(sub["live_value"], errors="coerce").dropna().to_numpy(float)
        rows.append({
            "expansion_update_label": label,
            "observation_status": sub["observation_status"].astype(str).mode().iloc[0],
            "n": int(len(sub)),
            "n_groups": int(sub["_group_id"].nunique()),
            "n_target_ops": int(sub["target_operator"].nunique()),
            "mean_live_value": float(np.mean(vals)) if len(vals) else np.nan,
            "median_live_value": float(np.median(vals)) if len(vals) else np.nan,
            "positive_rate": float(np.mean(sub["is_positive"])) if len(sub) else np.nan,
            "top_source": sub["source_experiment"].astype(str).mode().iloc[0],
            "top_action": sub["action"].astype(str).mode().iloc[0],
            "top_target_operator": sub["target_operator"].astype(str).mode().iloc[0],
            "top_candidate_operator": sub["candidate_operator"].astype(str).mode().iloc[0],
        })
    return pd.DataFrame(rows).sort_values(["observation_status", "mean_live_value"], ascending=[True, False])


def build_observation_candidates(df: pd.DataFrame, label_summary: pd.DataFrame) -> pd.DataFrame:
    keep_status = {"operator_candidate", "observation_period"}
    candidates = label_summary[label_summary["observation_status"].isin(keep_status)].copy()
    if candidates.empty:
        # fallback: top non-rejected labels by positive rate / mean
        candidates = label_summary[~label_summary["expansion_update_label"].isin(["reject_as_artifact"])].copy()
        candidates = candidates.sort_values(["positive_rate", "mean_live_value"], ascending=False).head(5)

    # Add proposed evaluation plan.
    plans = []
    for _, r in candidates.iterrows():
        label = r["expansion_update_label"]
        if label == "increase_address_prior":
            plan = "Observation: same operator/RelationMapping/commit-neighbor tasks; test address-prior replay across new concepts."
        elif label == "add_correction_branch":
            plan = "Observation: same-concept different-operator cases; test Definition/Planning/Risk correction branch generalization."
        elif label == "add_constraint":
            plan = "Observation: RiskAudit/closure-like cases; test whether constraint insertion improves boundary/risk tasks without non-target damage."
        elif label == "update_operator_prior":
            plan = "Observation: same-operator cross-concept cases; test operator prior transfer to held-out concepts."
        elif label == "add_relation_edge":
            plan = "Observation: same-family variants; test whether relation-edge addition improves unseen surface variants."
        elif label == "add_trajectory_class":
            plan = "Observation: new trajectory type; test held-out surface and concept split before promoting."
        elif label == "split_context":
            plan = "Observation: potential context-specific exception; do not generalize until split variable is identified."
        else:
            plan = "Observation: keep as working patch; gather more evidence."
        rr = r.to_dict()
        rr["observation_plan"] = plan
        plans.append(rr)
    return pd.DataFrame(plans)


def make_verdict(df: pd.DataFrame, summary: pd.DataFrame, clf: Dict, obs: pd.DataFrame) -> Dict:
    n_candidate = int((summary["observation_status"] == "operator_candidate").sum()) if not summary.empty else 0
    n_observation = int((summary["observation_status"] == "observation_period").sum()) if not summary.empty else 0
    macro_f1 = clf.get("cv_macro_f1", np.nan) if isinstance(clf, dict) else np.nan
    base_f1 = clf.get("baseline_macro_f1", np.nan) if isinstance(clf, dict) else np.nan

    pass_lite = bool(n_candidate + n_observation >= 2 and not summary.empty)
    pass_strong = bool(pass_lite and np.isfinite(macro_f1) and np.isfinite(base_f1) and macro_f1 > base_f1 + 0.15 and macro_f1 > 0.45)

    if pass_strong:
        verdict = "PASS_STRONG_EXPANSION_UPDATE_LABELS"
    elif pass_lite:
        verdict = "PASS_LITE_OBSERVATION_CANDIDATES_FOUND"
    else:
        verdict = "FAIL_OR_INCONCLUSIVE"

    return {
        "stage": "SEM-6L.2",
        "mode": "experience_history_contrast_expansion_update_audit",
        "verdict": verdict,
        "pass_lite": pass_lite,
        "pass_strong": pass_strong,
        "n_rows": int(len(df)),
        "n_update_labels": int(summary["expansion_update_label"].nunique()) if not summary.empty else 0,
        "n_operator_candidates": n_candidate,
        "n_observation_period": n_observation,
        "classifier_macro_f1": macro_f1,
        "classifier_baseline_macro_f1": base_f1,
        "notes": [
            "6L.2 constructs ExpansionMap update labels from Experience–History contrast.",
            "This is not a live controller; it creates routing patches and observation-period candidates.",
            "Promotion to OperatorMemory requires 6L.3 cross-case generalization replay.",
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all_rows", default=DEFAULT_ALL_ROWS)
    ap.add_argument("--positive_enrichment", default=DEFAULT_POS_ENRICH)
    ap.add_argument("--numeric_contrasts", default=DEFAULT_NUM_CONTRASTS)
    ap.add_argument("--cluster_summary", default=DEFAULT_CLUSTER_SUMMARY)
    ap.add_argument("--prototype_rules", default=DEFAULT_RULES)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--positive_threshold", type=float, default=0.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = load_csv(args.all_rows, required=True)
    all_rows = ensure_cols(all_rows)
    contrast = build_contrast_table(all_rows, args.positive_threshold)
    labeled = add_update_labels(contrast, args.positive_threshold)

    contrast_path = out_dir / "sem6l2_experience_history_contrast_table.csv"
    labels_path = out_dir / "sem6l2_expansion_update_labels.csv"
    contrast.to_csv(contrast_path, index=False, encoding="utf-8-sig")
    labeled.to_csv(labels_path, index=False, encoding="utf-8-sig")

    summary = summarize_labels(labeled)
    summary_path = out_dir / "sem6l2_update_label_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    obs = build_observation_candidates(labeled, summary)
    obs_path = out_dir / "sem6l2_observation_candidates.csv"
    obs.to_csv(obs_path, index=False, encoding="utf-8-sig")

    clf = train_update_classifier(labeled, out_dir)
    clf_path = out_dir / "sem6l2_update_classifier_report.json"
    with open(clf_path, "w", encoding="utf-8") as f:
        json.dump(clf, f, ensure_ascii=False, indent=2)

    verdict = make_verdict(labeled, summary, clf, obs)
    verdict["inputs"] = {
        "all_rows": args.all_rows,
        "positive_enrichment": args.positive_enrichment,
        "numeric_contrasts": args.numeric_contrasts,
        "cluster_summary": args.cluster_summary,
        "prototype_rules": args.prototype_rules,
    }
    verdict["outputs"] = {
        "contrast_table": str(contrast_path),
        "labeled_rows": str(labels_path),
        "label_summary": str(summary_path),
        "observation_candidates": str(obs_path),
        "classifier_report": str(clf_path),
        "classifier_predictions": str(out_dir / "sem6l2_update_classifier_predictions.csv"),
        "verdict": str(out_dir / "sem6l2_verdict.json"),
    }

    with open(out_dir / "sem6l2_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)

    print("\n========== SEM-6L.2 ==========")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    print("\nUpdate label summary:")
    print(summary.to_string(index=False))
    print("\nObservation candidates:")
    print(obs.to_string(index=False))


if __name__ == "__main__":
    main()
