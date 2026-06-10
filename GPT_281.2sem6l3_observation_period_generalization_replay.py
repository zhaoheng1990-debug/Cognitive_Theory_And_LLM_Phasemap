# -*- coding: utf-8 -*-
r"""
SEM-6L.3: Observation-Period Generalization Replay

Purpose
-------
SEM-6L.2 produced ExpansionMap update labels and observation candidates:

    RoutingPatch / ExpansionUpdate
    -> observation_period / operator_candidate

SEM-6L.3 tests whether these candidates generalize enough to be promoted:

    RoutingPatch
    -> OperatorCandidate
    -> CanonicalOperator (future step)

This is an offline generalization audit, not a model-forward controller.

It checks:
    1. within-class utility
    2. cross-group generalization
    3. cross-surface/source robustness
    4. cross-concept/operator coverage
    5. non-target damage
    6. compression / canonicalization feasibility

Inputs
------
Default:
    sem6l2_outputs\sem6l2_expansion_update_labels.csv
    sem6l2_outputs\sem6l2_observation_candidates.csv

Outputs
-------
sem6l3_outputs\
    sem6l3_generalization_table.csv
    sem6l3_candidate_promotion_table.csv
    sem6l3_label_group_replay.csv
    sem6l3_non_target_damage.csv
    sem6l3_verdict.json

Run
---
conda activate dhrf_4080s
cd C:\Users\ZH\Desktop\AGI\python_script

python GPT_sem6l3_observation_period_generalization_replay.py

Optional:
python GPT_sem6l3_observation_period_generalization_replay.py ^
  --min_groups 5 ^
  --min_positive_rate 0.25 ^
  --min_mean_live 0.0
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
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_ROOT = r"C:\Users\ZH\Desktop\AGI\python_script"
DEFAULT_L2_DIR = rf"{DEFAULT_ROOT}\sem6l2_outputs"
DEFAULT_LABELS = rf"{DEFAULT_L2_DIR}\sem6l2_expansion_update_labels.csv"
DEFAULT_CANDIDATES = rf"{DEFAULT_L2_DIR}\sem6l2_observation_candidates.csv"
DEFAULT_OUT_DIR = rf"{DEFAULT_ROOT}\sem6l3_outputs"

RANDOM_SEED = 20260606


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def read_csv_required(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p}")
    return pd.read_csv(p)


def ensure_base_cols(df: pd.DataFrame) -> pd.DataFrame:
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


def group_generalization(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each ExpansionUpdateLabel, compute per-group stats.
    Group = _group_id, a proxy for family / memory anchor.
    """
    rows = []
    for label, sub in df.groupby("expansion_update_label", dropna=False):
        group_stats = []
        for gid, g in sub.groupby("_group_id", dropna=False):
            vals = pd.to_numeric(g["live_value"], errors="coerce").dropna().to_numpy(float)
            if len(vals) == 0:
                continue
            group_stats.append({
                "group": gid,
                "n": len(vals),
                "mean": float(np.mean(vals)),
                "positive_rate": float(np.mean(g["is_positive"])),
            })
        if not group_stats:
            continue
        gs = pd.DataFrame(group_stats)
        rows.append({
            "expansion_update_label": label,
            "n_rows": int(len(sub)),
            "n_groups": int(len(gs)),
            "mean_live_value": float(pd.to_numeric(sub["live_value"], errors="coerce").mean()),
            "median_live_value": float(pd.to_numeric(sub["live_value"], errors="coerce").median()),
            "positive_rate": float(pd.to_numeric(sub["is_positive"], errors="coerce").mean()),
            "group_mean_of_means": float(gs["mean"].mean()),
            "group_std_of_means": float(gs["mean"].std(ddof=1)) if len(gs) > 1 else np.nan,
            "group_positive_frac": float(np.mean(gs["mean"] > 0)),
            "group_positive_rate_mean": float(gs["positive_rate"].mean()),
            "n_target_concepts": int(sub["target_concept"].nunique()),
            "n_candidate_concepts": int(sub["candidate_concept"].nunique()),
            "n_target_ops": int(sub["target_operator"].nunique()),
            "n_candidate_ops": int(sub["candidate_operator"].nunique()),
            "n_sources": int(sub["source_experiment"].nunique()),
            "n_surfaces_proxy": int(sub["_selection_family"].nunique()),
        })
    return pd.DataFrame(rows).sort_values("mean_live_value", ascending=False)


def label_group_replay(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["expansion_update_label", "_group_id", "target_operator", "candidate_operator"]
    for vals, sub in df.groupby(keys, dropna=False):
        label, gid, top, cop = vals
        live = pd.to_numeric(sub["live_value"], errors="coerce").dropna().to_numpy(float)
        if len(live) == 0:
            continue
        rows.append({
            "expansion_update_label": label,
            "_group_id": gid,
            "target_operator": top,
            "candidate_operator": cop,
            "n": int(len(live)),
            "mean_live_value": float(np.mean(live)),
            "positive_rate": float(np.mean(sub["is_positive"])),
            "top_source": sub["source_experiment"].astype(str).mode().iloc[0],
            "top_action": sub["action"].astype(str).mode().iloc[0],
            "top_selection": sub["_selection_family"].astype(str).mode().iloc[0],
        })
    return pd.DataFrame(rows).sort_values(["expansion_update_label", "mean_live_value"], ascending=[True, False])


def non_target_damage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate damage:
      - reject_as_artifact / keep_existing_route are controls
      - negative mean among non-target labels / random-like contexts indicates damage
    """
    rows = []
    control_labels = {"reject_as_artifact", "keep_existing_route"}
    for label, sub in df.groupby("expansion_update_label", dropna=False):
        # Non-target proxies: random-like, source_experiment with non-address/opposite? Conservative.
        nt = sub[
            (sub.get("is_random_like", 0).fillna(0).astype(float) > 0)
            | (sub["_selection_family"].astype(str).str.contains("random|shuffle", case=False, regex=True))
        ].copy()
        vals = pd.to_numeric(nt["live_value"], errors="coerce").dropna().to_numpy(float)
        all_vals = pd.to_numeric(sub["live_value"], errors="coerce").dropna().to_numpy(float)

        rows.append({
            "expansion_update_label": label,
            "n_all": int(len(all_vals)),
            "mean_all": float(np.mean(all_vals)) if len(all_vals) else np.nan,
            "n_non_target_proxy": int(len(vals)),
            "mean_non_target_proxy": float(np.mean(vals)) if len(vals) else np.nan,
            "damage_rate_non_target": float(np.mean(vals < 0)) if len(vals) else np.nan,
            "has_damage_flag": bool(len(vals) >= 10 and np.mean(vals) < -0.02),
            "is_control_label": label in control_labels,
        })
    return pd.DataFrame(rows).sort_values("mean_non_target_proxy", ascending=True)


def feature_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric = [
        "_learned_noleak_q", "proxy_utility", "oracle_utility", "q_proxy_gap", "q_oracle_gap",
        "beta", "delta_scale", "alpha_operator", "beta_precursor", "asa_alpha", "asa_beta",
        "base_margin", "base_rank", "base_logprob_gap_to_top", "n_delta_layers",
        "hist_mean_live", "hist_positive_rate", "hist_best_live",
        "action_ctx_mean_live", "action_ctx_positive_rate",
        "experience_vs_history_gain", "positive_rate_gap", "same_operator", "same_concept",
        "is_risk_target", "is_planning_target", "is_relation_target", "is_closure_like",
        "is_address_like", "is_random_like",
    ]
    numeric = [c for c in numeric if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().sum() > 0]
    categorical = [
        "expansion_update_label", "source_experiment", "action", "_selection_family",
        "relation_type", "write_route", "target_operator", "candidate_operator",
        "target_concept", "candidate_concept", "pool",
    ]
    categorical = [c for c in categorical if c in df.columns]
    return numeric, categorical


def cv_positive_within_label(df: pd.DataFrame, label: str) -> Dict:
    sub = df[df["expansion_update_label"] == label].copy()
    if len(sub) < 30 or sub["is_positive"].nunique() < 2:
        return {
            "expansion_update_label": label,
            "n": int(len(sub)),
            "cv_auc": np.nan,
            "note": "too few rows or one class"
        }

    numeric, categorical = feature_cols(sub)
    X = sub[numeric + categorical].copy()
    for c in numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median() if X[c].notna().any() else 0.0)
    for c in categorical:
        X[c] = X[c].astype(str).fillna("NA")
    y = sub["is_positive"].astype(int).to_numpy()

    groups = sub["_group_id"].astype(str).to_numpy()
    if len(np.unique(groups)) >= 2:
        k = min(5, len(np.unique(groups)))
        splits = list(GroupKFold(n_splits=k).split(X, y, groups))
    else:
        min_count = np.bincount(y).min()
        if min_count < 2:
            return {"expansion_update_label": label, "n": int(len(sub)), "cv_auc": np.nan, "note": "class too small"}
        k = min(5, min_count)
        splits = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED).split(X, y))

    prob = np.full(len(sub), np.nan)
    for tr, te in splits:
        try:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
        pre = ColumnTransformer([("num", StandardScaler(), numeric), ("cat", enc, categorical)], remainder="drop")
        clf = ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X.iloc[tr], y[tr])
        prob[te] = pipe.predict_proba(X.iloc[te])[:, 1]

    mask = np.isfinite(prob)
    auc = roc_auc_score(y[mask], prob[mask]) if len(np.unique(y[mask])) == 2 else np.nan
    return {
        "expansion_update_label": label,
        "n": int(len(sub)),
        "positive_rate": float(np.mean(y)),
        "cv_auc": float(auc) if np.isfinite(auc) else np.nan,
        "features_numeric": numeric,
        "features_categorical": categorical,
    }


def build_promotion_table(gen: pd.DataFrame, damage: pd.DataFrame, df: pd.DataFrame, args) -> pd.DataFrame:
    rows = []
    dmg_map = damage.set_index("expansion_update_label").to_dict("index") if not damage.empty else {}

    for _, r in gen.iterrows():
        label = r["expansion_update_label"]
        dmg = dmg_map.get(label, {})
        mean_live = float(r["mean_live_value"])
        pos_rate = float(r["positive_rate"])
        n_groups = int(r["n_groups"])
        n_rows = int(r["n_rows"])
        n_ops = int(r["n_target_ops"])
        n_sources = int(r["n_sources"])
        group_pos_frac = float(r["group_positive_frac"])
        damage_flag = bool(dmg.get("has_damage_flag", False))

        # Generalization score: simple interpretable composite.
        utility_score = max(0.0, mean_live)
        generalization_score = (
            0.35 * min(n_groups / max(args.min_groups, 1), 1.0)
            + 0.25 * min(n_ops / 2.0, 1.0)
            + 0.20 * min(n_sources / 2.0, 1.0)
            + 0.20 * group_pos_frac
        )
        stability_score = min(pos_rate / max(args.min_positive_rate, 1e-6), 1.0)
        damage_penalty = 0.5 if damage_flag else 0.0
        promotion_score = generalization_score + stability_score + utility_score - damage_penalty

        if damage_flag:
            decision = "reject_or_boundary_needed"
        elif n_groups >= args.min_groups and pos_rate >= args.min_positive_rate and mean_live >= args.min_mean_live and group_pos_frac >= 0.25:
            decision = "promote_to_operator_candidate"
        elif n_groups >= max(2, args.min_groups // 2) and pos_rate >= max(0.15, args.min_positive_rate * 0.7):
            decision = "continue_observation"
        else:
            decision = "working_patch_only"

        rows.append({
            "expansion_update_label": label,
            "decision": decision,
            "promotion_score": float(promotion_score),
            "generalization_score": float(generalization_score),
            "stability_score": float(stability_score),
            "utility_score": float(utility_score),
            "damage_penalty": float(damage_penalty),
            "n_rows": n_rows,
            "n_groups": n_groups,
            "n_target_ops": n_ops,
            "n_sources": n_sources,
            "mean_live_value": mean_live,
            "positive_rate": pos_rate,
            "group_positive_frac": group_pos_frac,
            "mean_non_target_proxy": dmg.get("mean_non_target_proxy", np.nan),
            "damage_rate_non_target": dmg.get("damage_rate_non_target", np.nan),
            "top_source": df[df["expansion_update_label"] == label]["source_experiment"].astype(str).mode().iloc[0],
            "top_action": df[df["expansion_update_label"] == label]["action"].astype(str).mode().iloc[0],
            "top_target_operator": df[df["expansion_update_label"] == label]["target_operator"].astype(str).mode().iloc[0],
        })

    return pd.DataFrame(rows).sort_values("promotion_score", ascending=False)


def make_verdict(promo: pd.DataFrame, gen: pd.DataFrame, within_reports: List[Dict]) -> Dict:
    n_promote = int((promo["decision"] == "promote_to_operator_candidate").sum()) if not promo.empty else 0
    n_continue = int((promo["decision"] == "continue_observation").sum()) if not promo.empty else 0

    aucs = [r.get("cv_auc", np.nan) for r in within_reports]
    valid_aucs = [a for a in aucs if np.isfinite(a)]
    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else np.nan

    if n_promote >= 1:
        verdict = "PASS_STRONG_OPERATOR_CANDIDATE_PROMOTION"
    elif n_continue >= 2:
        verdict = "PASS_LITE_OBSERVATION_CONTINUES"
    else:
        verdict = "FAIL_OR_INCONCLUSIVE"

    return {
        "stage": "SEM-6L.3",
        "mode": "observation_period_generalization_replay",
        "verdict": verdict,
        "n_promoted_operator_candidates": n_promote,
        "n_continue_observation": n_continue,
        "mean_within_label_positive_auc": mean_auc,
        "notes": [
            "6L.3 audits whether ExpansionMap updates generalize enough to enter OperatorCandidate status.",
            "Promotion does not yet mean CanonicalOperator; canonicalization requires 6L.4 held-out replay / non-target safety validation.",
            "Decisions use utility, group coverage, operator coverage, source/surface proxy robustness, and non-target damage proxy.",
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--min_groups", type=int, default=5)
    ap.add_argument("--min_positive_rate", type=float, default=0.25)
    ap.add_argument("--min_mean_live", type=float, default=0.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = ensure_base_cols(read_csv_required(args.labels))
    # Optional candidates file for context; if missing just proceed.
    cand_path = Path(args.candidates)
    candidates = pd.read_csv(cand_path) if cand_path.exists() else pd.DataFrame()

    gen = group_generalization(labels)
    gen.to_csv(out_dir / "sem6l3_generalization_table.csv", index=False, encoding="utf-8-sig")

    lgr = label_group_replay(labels)
    lgr.to_csv(out_dir / "sem6l3_label_group_replay.csv", index=False, encoding="utf-8-sig")

    dmg = non_target_damage(labels)
    dmg.to_csv(out_dir / "sem6l3_non_target_damage.csv", index=False, encoding="utf-8-sig")

    within_reports = []
    for label in sorted(labels["expansion_update_label"].unique()):
        within_reports.append(cv_positive_within_label(labels, label))
    with open(out_dir / "sem6l3_within_label_positive_reports.json", "w", encoding="utf-8") as f:
        json.dump(within_reports, f, ensure_ascii=False, indent=2)

    promo = build_promotion_table(gen, dmg, labels, args)
    promo.to_csv(out_dir / "sem6l3_candidate_promotion_table.csv", index=False, encoding="utf-8-sig")

    verdict = make_verdict(promo, gen, within_reports)
    verdict["parameters"] = {
        "min_groups": args.min_groups,
        "min_positive_rate": args.min_positive_rate,
        "min_mean_live": args.min_mean_live,
    }
    verdict["outputs"] = {
        "generalization_table": str(out_dir / "sem6l3_generalization_table.csv"),
        "candidate_promotion_table": str(out_dir / "sem6l3_candidate_promotion_table.csv"),
        "label_group_replay": str(out_dir / "sem6l3_label_group_replay.csv"),
        "non_target_damage": str(out_dir / "sem6l3_non_target_damage.csv"),
        "within_label_reports": str(out_dir / "sem6l3_within_label_positive_reports.json"),
        "verdict": str(out_dir / "sem6l3_verdict.json"),
    }

    with open(out_dir / "sem6l3_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)

    print("\n========== SEM-6L.3 ==========")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    print("\nCandidate promotion table:")
    print(promo.to_string(index=False))
    print("\nGeneralization table:")
    print(gen.to_string(index=False))


if __name__ == "__main__":
    main()
