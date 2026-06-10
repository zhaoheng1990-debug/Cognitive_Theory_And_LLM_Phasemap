# -*- coding: utf-8 -*-
"""
DA-ASA-2J: OutcomeTopology-Derived LatentPolicyFactorization

Purpose
-------
Continue DA-ASA after 2E/2F/2G/2H/2I.

This script is intentionally table-level:
- It does NOT load Qwen/Llama/Gemma.
- It reads existing DA-ASA outputs.
- It constructs policy effect vectors from outcome topology.
- It clusters functionally equivalent / similar interventions into latent policy factors.
- It trains DSTA features -> latent factors.
- It performs a simple table replay audit.

Run
---
Put this file under:
    C:\\Users\\ZH\\Desktop\\AGI\\python_script

Then run:
    python GPT_257_da_asa2j_outcome_topology_latent_factorization.py

Outputs
-------
    C:\\Users\\ZH\\Desktop\\AGI\\python_script\\da_asa2j_outputs
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ============================================================
# PATHS: hard-coded for local Windows setup
# ============================================================

ROOT = Path(r"C:\Users\ZH\Desktop\AGI\python_script")

DIR_2E = ROOT / "da_asa2e_outputs"
DIR_2F = ROOT / "da_asa2f_outputs"
DIR_2G = ROOT / "da_asa2g_outputs"
DIR_2H = ROOT / "da_asa2h_outputs"
DIR_2I = ROOT / "da_asa2i_outputs"

OUT_DIR = ROOT / "da_asa2j_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Utility functions
# ============================================================

def read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def read_json_if_exists(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def norm_col(s: str) -> str:
    return str(s).strip().lower().replace("-", "_").replace(" ", "_")


def find_col(df: pd.DataFrame, candidates: List[str], required: bool = False) -> Optional[str]:
    cmap = {norm_col(c): c for c in df.columns}
    for c in candidates:
        if norm_col(c) in cmap:
            return cmap[norm_col(c)]
    # relaxed substring
    for cand in candidates:
        nc = norm_col(cand)
        for k, v in cmap.items():
            if nc in k or k in nc:
                return v
    if required:
        raise ValueError(f"Cannot find required column among {candidates}. Existing columns: {list(df.columns)}")
    return None


def numeric_cols(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    exclude = set(exclude or [])
    out = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def safe_mean(x) -> float:
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    return float(x.mean()) if x.notna().any() else float("nan")


def infer_condition_family(x: Any) -> str:
    s = str(x).lower()
    if "closure_override" in s or "override" in s:
        return "closure_override"
    if "closure_exception" in s or "exception" in s:
        return "closure_exception"
    if "closure_negation" in s or "negation" in s:
        return "closure_negation"
    if "equal" in s and "evidence" in s:
        return "competition_equal_evidence"
    if "source" in s or "claim" in s:
        return "competition_source_claim"
    if "hallucination" in s:
        return "hallucination_like"
    if "stable" in s or "clean" in s:
        return "stable_or_clean"
    if "non" in s and "target" in s:
        return "non_target"
    return s


def action_is_noop(action: Any) -> bool:
    s = str(action).lower()
    noop_markers = [
        "a0.0_b0.0", "a0_b0", "no_intervention", "none", "noop", "no_op",
        "identity", "control_none", "no intervention"
    ]
    return any(m in s for m in noop_markers)


def action_is_fixed_candidate(action: Any) -> bool:
    s = str(action).lower()
    return ("a0.3_b0.3" in s) or ("fixed" in s) or ("same_combo" in s)


# ============================================================
# Load DA-ASA files
# ============================================================

def load_inputs() -> Dict[str, Any]:
    files = {}

    files["2e_action_sig"] = read_csv_if_exists(DIR_2E / "da_asa2e_action_outcome_signatures.csv")
    files["2e_equiv_long"] = read_csv_if_exists(DIR_2E / "da_asa2e_equivalence_classes_long.csv")
    files["2e_cond_comp"] = read_csv_if_exists(DIR_2E / "da_asa2e_condition_compression_summary.csv")
    files["2e_pred_equiv"] = read_csv_if_exists(DIR_2E / "da_asa2e_prediction_outcome_equivalence.csv")
    files["2e_replay_by_outcome"] = read_csv_if_exists(DIR_2E / "da_asa2e_replay_by_outcome_class.csv")
    files["2e_metric_summary"] = read_csv_if_exists(DIR_2E / "da_asa2e_metric_summary.csv")
    files["2e_diag"] = read_json_if_exists(DIR_2E / "da_asa2e_diagnostics.json")

    files["2f_feat"] = read_csv_if_exists(DIR_2F / "da_asa2f_fallback_feature_table.csv")
    files["2f_feature_sets"] = read_json_if_exists(DIR_2F / "da_asa2f_feature_sets.json")
    files["2f_pred"] = read_csv_if_exists(DIR_2F / "da_asa2f_predictions_long.csv")
    files["2f_cv"] = read_csv_if_exists(DIR_2F / "da_asa2f_cv_summary.csv")

    files["2g_pred"] = read_csv_if_exists(DIR_2G / "da_asa2g_predictions_long.csv")
    files["2g_best"] = read_csv_if_exists(DIR_2G / "da_asa2g_best_by_mode_target.csv")
    files["2g_holdout"] = read_csv_if_exists(DIR_2G / "da_asa2g_holdout_summary.csv")

    files["2h_factor_pred"] = read_csv_if_exists(DIR_2H / "da_asa2h_factor_predictions_long.csv")
    files["2h_recomposed"] = read_csv_if_exists(DIR_2H / "da_asa2h_recomposed_policy_predictions.csv")
    files["2h_summary"] = read_csv_if_exists(DIR_2H / "da_asa2h_recomposed_policy_summary.csv")

    files["2i_factor_pred"] = read_csv_if_exists(DIR_2I / "da_asa2i_factor_predictions_long.csv")
    files["2i_hier_pred"] = read_csv_if_exists(DIR_2I / "da_asa2i_hierarchical_policy_predictions.csv")
    files["2i_hier_summary"] = read_csv_if_exists(DIR_2I / "da_asa2i_hierarchical_policy_summary.csv")

    present = {k: (v is not None) for k, v in files.items()}
    save_json(present, OUT_DIR / "da_asa2j_input_file_presence.json")
    return files


# ============================================================
# Step 1: Construct policy effect vector
# ============================================================

def choose_policy_table(files: Dict[str, Any]) -> pd.DataFrame:
    """
    Prefer 2E action outcome signature. If too aggregated/empty, fall back to replay_by_outcome.
    """
    candidates = [
        ("2e_action_sig", files.get("2e_action_sig")),
        ("2e_replay_by_outcome", files.get("2e_replay_by_outcome")),
        ("2e_pred_equiv", files.get("2e_pred_equiv")),
    ]
    for name, df in candidates:
        if df is not None and len(df) > 0:
            out = df.copy()
            out["_source_table"] = name
            return out
    raise FileNotFoundError(
        "No usable 2E table found. Expected one of: "
        "da_asa2e_action_outcome_signatures.csv, "
        "da_asa2e_replay_by_outcome_class.csv, "
        "da_asa2e_prediction_outcome_equivalence.csv"
    )


def construct_effect_vectors(policy_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = policy_df.copy()

    condition_col = find_col(df, [
        "condition", "condition_name", "row_type", "target_condition", "family", "condition_family"
    ])
    action_col = find_col(df, [
        "action", "action_class", "policy_class", "pred_policy", "oracle_policy", "selected_action",
        "policy_action", "action_label"
    ])
    outcome_col = find_col(df, [
        "outcome_class", "policy_outcome_class", "outcome", "outcome_label", "equivalence_class"
    ])

    pred_col = find_col(df, ["pred_clean", "clean_rate", "target_hit_rate", "hit_rate", "gen_clean"])
    r_col = find_col(df, ["R_final", "final_R", "r_final", "R", "margin_final", "clean_margin"])
    damage_col = find_col(df, [
        "non_target_damage", "damage", "safety_penalty", "non_target_penalty",
        "negative_side_effect", "side_effect"
    ])

    # If no condition/action columns are detected, synthesize stable identifiers.
    if condition_col is None:
        df["condition"] = "all_conditions"
        condition_col = "condition"
    if action_col is None:
        # try representative action/outcome
        rep = find_col(df, ["outcome_class_representative", "representative", "class_representative"])
        if rep is not None:
            action_col = rep
        else:
            df["action_class"] = [f"row_action_{i}" for i in range(len(df))]
            action_col = "action_class"

    df["_condition"] = df[condition_col].astype(str)
    df["_condition_family"] = df["_condition"].map(infer_condition_family)
    df["_action"] = df[action_col].astype(str)
    if outcome_col is not None:
        df["_outcome_class"] = df[outcome_col].astype(str)
    else:
        df["_outcome_class"] = df["_action"]

    # Aggregate duplicated rows by condition/action/outcome, keeping numeric means.
    group_cols = ["_condition", "_condition_family", "_action", "_outcome_class"]
    num = numeric_cols(df)
    if not num:
        raise ValueError(f"No numeric columns found in policy table. Columns: {list(df.columns)}")

    agg = df.groupby(group_cols, dropna=False)[num].mean().reset_index()

    # Establish primary metrics.
    if pred_col is not None and pred_col in agg.columns:
        agg["_pred_clean"] = pd.to_numeric(agg[pred_col], errors="coerce")
    else:
        # fallback: use best available numeric column containing clean/hit/pred
        maybe = [c for c in num if any(t in norm_col(c) for t in ["clean", "hit", "pred"])]
        agg["_pred_clean"] = pd.to_numeric(agg[maybe[0]], errors="coerce") if maybe else 0.0

    if r_col is not None and r_col in agg.columns:
        agg["_R_final"] = pd.to_numeric(agg[r_col], errors="coerce")
    else:
        maybe = [c for c in num if any(t in norm_col(c) for t in ["r_final", "final", "margin", "r_"])]
        agg["_R_final"] = pd.to_numeric(agg[maybe[0]], errors="coerce") if maybe else agg["_pred_clean"]

    if damage_col is not None and damage_col in agg.columns:
        agg["_safety_penalty"] = pd.to_numeric(agg[damage_col], errors="coerce").fillna(0.0)
    else:
        agg["_safety_penalty"] = 0.0

    # Condition-wise baseline/noop and best.
    rows = []
    for cond, sub in agg.groupby("_condition", dropna=False):
        sub = sub.copy()

        noop = sub[sub["_action"].map(action_is_noop)]
        if len(noop) == 0:
            # fallback: lowest action norm / first row
            noop = sub.iloc[[0]]

        no_pred = float(noop["_pred_clean"].mean())
        no_r = float(noop["_R_final"].mean())

        # Oracle score is lexicographic: pred_clean primary, R_final secondary, safety negative.
        score = (
            sub["_pred_clean"].fillna(-1e9)
            + 0.05 * sub["_R_final"].fillna(0.0)
            - 0.10 * sub["_safety_penalty"].fillna(0.0)
        )
        best_idx = score.idxmax()
        best_pred = float(sub.loc[best_idx, "_pred_clean"])
        best_r = float(sub.loc[best_idx, "_R_final"])
        best_action = str(sub.loc[best_idx, "_action"])

        sub["_delta_pred_vs_noop"] = sub["_pred_clean"] - no_pred
        sub["_delta_R_vs_noop"] = sub["_R_final"] - no_r
        sub["_gap_pred_vs_oracle"] = best_pred - sub["_pred_clean"]
        sub["_gap_R_vs_oracle"] = best_r - sub["_R_final"]
        sub["_is_oracle_action"] = (sub["_action"].astype(str) == best_action).astype(int)

        sub["_functional_score"] = (
            sub["_delta_pred_vs_noop"].fillna(0.0)
            + 0.05 * sub["_delta_R_vs_noop"].fillna(0.0)
            - 0.10 * sub["_safety_penalty"].fillna(0.0)
        )
        sub["_rank_within_condition"] = sub["_functional_score"].rank(ascending=False, method="dense").astype(int)

        # best fixed candidate if available
        fixed = sub[sub["_action"].map(action_is_fixed_candidate)]
        if len(fixed) > 0:
            fixed_score = float(fixed["_functional_score"].max())
        else:
            fixed_score = float(sub["_functional_score"].median())
        sub["_delta_vs_best_fixed_proxy"] = sub["_functional_score"] - fixed_score

        rows.append(sub)

    eff = pd.concat(rows, axis=0, ignore_index=True)

    effect_cols = [
        "_pred_clean",
        "_R_final",
        "_safety_penalty",
        "_delta_pred_vs_noop",
        "_delta_R_vs_noop",
        "_gap_pred_vs_oracle",
        "_gap_R_vs_oracle",
        "_delta_vs_best_fixed_proxy",
        "_rank_within_condition",
        "_functional_score",
    ]

    # Add selected original numeric columns that are not redundant.
    for c in num:
        if c not in effect_cols and c in eff.columns:
            nc = norm_col(c)
            if any(t in nc for t in ["pred", "clean", "r", "margin", "damage", "penalty", "gain", "delta", "gap"]):
                effect_cols.append(c)

    effect_cols = [c for c in effect_cols if c in eff.columns]
    meta = {
        "condition_col": condition_col,
        "action_col": action_col,
        "outcome_col": outcome_col,
        "pred_col": pred_col,
        "r_col": r_col,
        "damage_col": damage_col,
        "effect_cols": effect_cols,
        "n_effect_rows": int(len(eff)),
        "n_conditions": int(eff["_condition"].nunique()),
        "n_actions": int(eff["_action"].nunique()),
        "n_outcome_classes": int(eff["_outcome_class"].nunique()),
    }
    return eff, meta


# ============================================================
# Step 2: Cluster effect vectors into latent policy factors
# ============================================================

def cluster_effect_vectors(eff: pd.DataFrame, effect_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    X = eff[effect_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = SimpleImputer(strategy="median").fit_transform(X)
    Xs = StandardScaler().fit_transform(X)

    n = len(eff)
    max_k = min(8, max(2, n - 1))
    rows = []
    label_store: Dict[str, np.ndarray] = {}

    for k in range(2, max_k + 1):
        if k >= n:
            continue

        km = KMeans(n_clusters=k, random_state=42, n_init=50)
        labels = km.fit_predict(Xs)
        sil = float(silhouette_score(Xs, labels)) if len(set(labels)) > 1 else float("nan")

        rows.append({
            "method": "kmeans",
            "k": k,
            "silhouette": sil,
        })
        label_store[f"kmeans_{k}"] = labels

        try:
            ag = AgglomerativeClustering(n_clusters=k, linkage="ward")
            labels_ag = ag.fit_predict(Xs)
            sil_ag = float(silhouette_score(Xs, labels_ag)) if len(set(labels_ag)) > 1 else float("nan")
            ari = float(adjusted_rand_score(labels, labels_ag))
            nmi = float(normalized_mutual_info_score(labels, labels_ag))
            rows.append({
                "method": "agglomerative_ward",
                "k": k,
                "silhouette": sil_ag,
                "ari_vs_kmeans": ari,
                "nmi_vs_kmeans": nmi,
            })
            label_store[f"agg_{k}"] = labels_ag
        except Exception as e:
            rows.append({
                "method": "agglomerative_ward",
                "k": k,
                "silhouette": float("nan"),
                "error": str(e),
            })

    cluster_eval = pd.DataFrame(rows)

    # Choose best by silhouette, prefer kmeans for stability and downstream classifier.
    km_eval = cluster_eval[cluster_eval["method"] == "kmeans"].dropna(subset=["silhouette"]).copy()
    if len(km_eval) == 0:
        best_key = "kmeans_2"
        best_k = 2
        best_method = "kmeans"
    else:
        best = km_eval.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]
        best_k = int(best["k"])
        best_method = str(best["method"])
        best_key = f"kmeans_{best_k}"

    labels = label_store[best_key]
    out = eff.copy()
    out["latent_policy_factor"] = [f"LPF_{int(x)}" for x in labels]
    out["latent_policy_factor_id"] = labels.astype(int)

    # PCA coordinates for visualization / audit.
    ncomp = min(5, Xs.shape[1], Xs.shape[0])
    if ncomp >= 2:
        pca = PCA(n_components=ncomp, random_state=42)
        Z = pca.fit_transform(Xs)
        for i in range(Z.shape[1]):
            out[f"effect_pc{i+1}"] = Z[:, i]
        pca_var = pca.explained_variance_ratio_.tolist()
    else:
        pca_var = []

    # Cluster summary
    summary_rows = []
    for fac, sub in out.groupby("latent_policy_factor"):
        row = {
            "latent_policy_factor": fac,
            "n": int(len(sub)),
            "n_conditions": int(sub["_condition"].nunique()),
            "n_actions": int(sub["_action"].nunique()),
            "mean_pred_clean": safe_mean(sub["_pred_clean"]),
            "mean_R_final": safe_mean(sub["_R_final"]),
            "mean_delta_pred_vs_noop": safe_mean(sub["_delta_pred_vs_noop"]),
            "mean_delta_R_vs_noop": safe_mean(sub["_delta_R_vs_noop"]),
            "mean_gap_pred_vs_oracle": safe_mean(sub["_gap_pred_vs_oracle"]),
            "mean_safety_penalty": safe_mean(sub["_safety_penalty"]),
            "mean_functional_score": safe_mean(sub["_functional_score"]),
            "oracle_action_rate": safe_mean(sub["_is_oracle_action"]),
            "top_actions": "; ".join(sub["_action"].value_counts().head(5).index.astype(str).tolist()),
            "top_conditions": "; ".join(sub["_condition"].value_counts().head(5).index.astype(str).tolist()),
            "top_outcomes": "; ".join(sub["_outcome_class"].value_counts().head(5).index.astype(str).tolist()),
        }
        summary_rows.append(row)
    cluster_summary = pd.DataFrame(summary_rows).sort_values("mean_functional_score", ascending=False)

    meta = {
        "best_method": best_method,
        "best_k": int(best_k),
        "best_key": best_key,
        "pca_explained_variance_ratio": pca_var,
    }
    return out, cluster_eval, cluster_summary, meta


# ============================================================
# Step 3/4: Build DSTA -> latent factor table
# ============================================================

def build_feature_label_table(files: Dict[str, Any], lpf_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    feat = files.get("2f_feat")
    if feat is None or len(feat) == 0:
        raise FileNotFoundError(
            "Missing 2F fallback feature table: "
            f"{DIR_2F / 'da_asa2f_fallback_feature_table.csv'}"
        )

    f = feat.copy()

    # Resolve feature table metadata columns.
    f_condition = find_col(f, ["condition", "condition_name", "row_type", "target_condition", "family", "condition_family"])
    f_graph = find_col(f, ["graph_id", "graph", "row_id", "idx", "sample_id"])
    f_family = find_col(f, ["condition_family", "family", "mechanism_family"])

    if f_condition is None:
        # fallback: if row_type or family missing, use all condition.
        f["_condition"] = "all_conditions"
    else:
        f["_condition"] = f[f_condition].astype(str)

    if f_family is None:
        f["_condition_family"] = f["_condition"].map(infer_condition_family)
    else:
        f["_condition_family"] = f[f_family].astype(str).map(infer_condition_family)

    if f_graph is None:
        f["_graph_id"] = np.arange(len(f))
    else:
        f["_graph_id"] = f[f_graph].astype(str)

    # Oracle latent factor per condition: choose best functional score action.
    # If lpf has multiple rows per condition/action, it has already been aggregated.
    idx = lpf_df.groupby("_condition")["_functional_score"].idxmax()
    cond_label = lpf_df.loc[idx, [
        "_condition", "_condition_family", "_action", "_outcome_class",
        "latent_policy_factor", "latent_policy_factor_id",
        "_functional_score", "_pred_clean", "_R_final",
    ]].copy()
    cond_label = cond_label.rename(columns={
        "_action": "oracle_action_by_effect",
        "_outcome_class": "oracle_outcome_by_effect",
        "_functional_score": "oracle_functional_score",
        "_pred_clean": "oracle_pred_clean",
        "_R_final": "oracle_R_final",
    })

    # First try condition merge.
    merged = f.merge(cond_label, on="_condition", how="left", suffixes=("", "_label"))

    # If too many missing, try condition_family merge as fallback.
    miss_rate = float(merged["latent_policy_factor"].isna().mean())
    fallback_used = False
    if miss_rate > 0.5:
        fam_idx = lpf_df.groupby("_condition_family")["_functional_score"].idxmax()
        fam_label = lpf_df.loc[fam_idx, [
            "_condition_family", "_action", "_outcome_class", "latent_policy_factor",
            "latent_policy_factor_id", "_functional_score", "_pred_clean", "_R_final"
        ]].copy()
        fam_label = fam_label.rename(columns={
            "_action": "oracle_action_by_effect",
            "_outcome_class": "oracle_outcome_by_effect",
            "_functional_score": "oracle_functional_score",
            "_pred_clean": "oracle_pred_clean",
            "_R_final": "oracle_R_final",
        })
        merged = f.merge(fam_label, on="_condition_family", how="left")
        fallback_used = True

    # Drop unlabeled rows only for classifier.
    labeled = merged[merged["latent_policy_factor"].notna()].copy()

    # Determine feature columns.
    # Prefer feature_sets json if available; otherwise numeric columns excluding labels/metas.
    feature_sets = files.get("2f_feature_sets") or {}
    candidates = []
    def flatten_feature_sets(obj):
        out = []
        if isinstance(obj, dict):
            for v in obj.values():
                out.extend(flatten_feature_sets(v))
        elif isinstance(obj, list):
            out.extend([x for x in obj if isinstance(x, str)])
        return out

    candidates = flatten_feature_sets(feature_sets)
    candidates = [c for c in candidates if c in labeled.columns and pd.api.types.is_numeric_dtype(labeled[c])]

    exclude = set([
        "latent_policy_factor_id", "oracle_functional_score",
        "oracle_pred_clean", "oracle_R_final"
    ])
    exclude.update([c for c in labeled.columns if c.startswith("_")])
    exclude.update(["latent_policy_factor"])

    if len(candidates) >= 3:
        feat_cols = list(dict.fromkeys(candidates))
    else:
        feat_cols = numeric_cols(labeled, exclude=list(exclude))

    # Remove almost constant or label leakage columns.
    leakage_terms = ["policy", "outcome_class", "latent", "oracle", "pred_policy", "target_label"]
    clean_feat_cols = []
    for c in feat_cols:
        nc = norm_col(c)
        if any(t in nc for t in leakage_terms):
            continue
        if labeled[c].nunique(dropna=True) <= 1:
            continue
        clean_feat_cols.append(c)

    meta = {
        "feature_condition_col": f_condition,
        "feature_graph_col": f_graph,
        "feature_family_col": f_family,
        "fallback_family_merge_used": fallback_used,
        "initial_merge_missing_rate": miss_rate,
        "n_feature_rows": int(len(f)),
        "n_labeled_rows": int(len(labeled)),
        "n_feature_cols": int(len(clean_feat_cols)),
        "feature_cols": clean_feat_cols,
    }
    return labeled, meta


def make_classifier(kind: str = "logreg") -> BaseEstimator:
    if kind == "logreg":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
        ])
    if kind == "rf":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=400,
                random_state=42,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
            )),
        ])
    if kind == "gb":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", GradientBoostingClassifier(random_state=42)),
        ])
    raise ValueError(kind)


def evaluate_classifier(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "latent_policy_factor",
    group_col: str = "_graph_id",
    family_col: str = "_condition_family",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = df[feature_cols].values
    y = df[target_col].astype(str).values
    labels_all = sorted(pd.unique(y).tolist())

    rows = []
    pred_rows = []

    def fit_predict(train_idx, test_idx, mode: str, heldout: str, model_kind: str):
        y_train = y[train_idx]
        y_test = y[test_idx]
        X_train = X[train_idx]
        X_test = X[test_idx]

        if len(np.unique(y_train)) < 2:
            model = DummyClassifier(strategy="most_frequent")
        else:
            model = make_classifier(model_kind)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        bal = balanced_accuracy_score(y_test, y_pred)
        macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        absent = sorted(set(y_test) - set(y_train))
        row = {
            "mode": mode,
            "heldout": heldout,
            "model": model_kind,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_train_classes": int(len(set(y_train))),
            "n_test_classes": int(len(set(y_test))),
            "absent_test_label_count": int(len(absent)),
            "absent_test_labels": ";".join(absent),
            "accuracy": float(acc),
            "balanced_accuracy": float(bal),
            "macro_f1": float(macro),
            "weighted_f1": float(weighted),
        }

        temp = df.iloc[test_idx][["_condition", "_condition_family", "_graph_id", target_col]].copy()
        temp["mode"] = mode
        temp["heldout"] = heldout
        temp["model"] = model_kind
        temp["pred_latent_policy_factor"] = y_pred
        pred_rows.append(temp)
        rows.append(row)

    # Graph-heldout / GroupKFold
    groups = df[group_col].astype(str).values if group_col in df.columns else np.arange(len(df)).astype(str)
    n_groups = len(set(groups))
    n_splits = min(5, n_groups)
    if n_splits >= 2 and len(df) >= n_splits:
        gkf = GroupKFold(n_splits=n_splits)
        for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
            for kind in ["logreg", "rf", "gb"]:
                fit_predict(tr, te, "graph_groupkfold", f"fold_{fold}", kind)

    # Stratified fallback
    min_count = pd.Series(y).value_counts().min()
    if min_count >= 2 and len(set(y)) >= 2:
        n_splits = int(min(5, min_count))
        if n_splits >= 2:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            for fold, (tr, te) in enumerate(skf.split(X, y)):
                for kind in ["logreg", "rf", "gb"]:
                    fit_predict(tr, te, "stratified_kfold", f"fold_{fold}", kind)

    # Leave-one-condition
    if "_condition" in df.columns:
        for cond, idxs in df.groupby("_condition").groups.items():
            te = np.array(list(idxs), dtype=int)
            tr = np.array([i for i in range(len(df)) if i not in set(te)], dtype=int)
            if len(tr) > 0 and len(te) > 0:
                for kind in ["logreg", "rf", "gb"]:
                    fit_predict(tr, te, "leave_one_condition", str(cond), kind)

    # Leave-one-family
    if family_col in df.columns:
        for fam, idxs in df.groupby(family_col).groups.items():
            te = np.array(list(idxs), dtype=int)
            tr = np.array([i for i in range(len(df)) if i not in set(te)], dtype=int)
            if len(tr) > 0 and len(te) > 0:
                for kind in ["logreg", "rf", "gb"]:
                    fit_predict(tr, te, "leave_one_family", str(fam), kind)

    metrics = pd.DataFrame(rows)
    preds = pd.concat(pred_rows, axis=0, ignore_index=True) if pred_rows else pd.DataFrame()
    return metrics, preds


# ============================================================
# Step 5: LatentFactor -> policy replay
# ============================================================

def build_factor_to_action_map(lpf_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each latent factor, choose representative action by average functional score.
    """
    rows = []
    for fac, sub in lpf_df.groupby("latent_policy_factor"):
        action_score = (
            sub.groupby("_action")
            .agg(
                n=("_action", "size"),
                mean_functional_score=("_functional_score", "mean"),
                mean_pred_clean=("_pred_clean", "mean"),
                mean_R_final=("_R_final", "mean"),
                mean_safety_penalty=("_safety_penalty", "mean"),
            )
            .reset_index()
        )
        action_score["_map_score"] = (
            action_score["mean_functional_score"].fillna(0.0)
            + 0.02 * action_score["mean_pred_clean"].fillna(0.0)
            - 0.05 * action_score["mean_safety_penalty"].fillna(0.0)
        )
        best = action_score.sort_values("_map_score", ascending=False).iloc[0]
        rows.append({
            "latent_policy_factor": fac,
            "selected_action_for_factor": best["_action"],
            "factor_action_n": int(best["n"]),
            "factor_action_mean_score": float(best["mean_functional_score"]),
            "factor_action_mean_pred_clean": float(best["mean_pred_clean"]),
            "factor_action_mean_R_final": float(best["mean_R_final"]),
        })
    return pd.DataFrame(rows)


def table_replay_from_predictions(
    lpf_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    factor_action_map: pd.DataFrame,
    mode_filter: str = "leave_one_family",
    model_filter: str = "logreg",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Approximate table replay:
    predicted latent factor -> representative action -> effect metrics within same condition if available.
    If exact condition/action not available, fall back to global action average.
    """
    if pred_df is None or len(pred_df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    preds = pred_df[(pred_df["mode"] == mode_filter) & (pred_df["model"] == model_filter)].copy()
    if len(preds) == 0:
        preds = pred_df[pred_df["model"] == model_filter].copy()
    if len(preds) == 0:
        preds = pred_df.copy()

    preds = preds.merge(
        factor_action_map,
        left_on="pred_latent_policy_factor",
        right_on="latent_policy_factor",
        how="left"
    )

    # Exact condition/action effect lookup
    effect_lookup = lpf_df[[
        "_condition", "_action", "_pred_clean", "_R_final", "_functional_score",
        "_delta_pred_vs_noop", "_delta_R_vs_noop", "_safety_penalty",
        "latent_policy_factor"
    ]].copy()

    replay = preds.merge(
        effect_lookup,
        left_on=["_condition", "selected_action_for_factor"],
        right_on=["_condition", "_action"],
        how="left",
        suffixes=("", "_effect")
    )

    # Global action fallback
    global_action = (
        lpf_df.groupby("_action")
        .agg(
            global_pred_clean=("_pred_clean", "mean"),
            global_R_final=("_R_final", "mean"),
            global_functional_score=("_functional_score", "mean"),
            global_delta_pred_vs_noop=("_delta_pred_vs_noop", "mean"),
            global_delta_R_vs_noop=("_delta_R_vs_noop", "mean"),
            global_safety_penalty=("_safety_penalty", "mean"),
        )
        .reset_index()
    )

    replay = replay.merge(
        global_action,
        left_on="selected_action_for_factor",
        right_on="_action",
        how="left",
        suffixes=("", "_global")
    )

    for dst, glob in [
        ("_pred_clean", "global_pred_clean"),
        ("_R_final", "global_R_final"),
        ("_functional_score", "global_functional_score"),
        ("_delta_pred_vs_noop", "global_delta_pred_vs_noop"),
        ("_delta_R_vs_noop", "global_delta_R_vs_noop"),
        ("_safety_penalty", "global_safety_penalty"),
    ]:
        replay[dst] = pd.to_numeric(replay[dst], errors="coerce")
        replay[glob] = pd.to_numeric(replay[glob], errors="coerce")
        replay[dst] = replay[dst].fillna(replay[glob])

    # Baselines from lpf table
    oracle = (
        lpf_df.loc[lpf_df.groupby("_condition")["_functional_score"].idxmax()]
        [["_condition", "_action", "_pred_clean", "_R_final", "_functional_score"]]
        .rename(columns={
            "_action": "oracle_action",
            "_pred_clean": "oracle_pred_clean",
            "_R_final": "oracle_R_final",
            "_functional_score": "oracle_functional_score",
        })
    )
    noop = (
        lpf_df[lpf_df["_action"].map(action_is_noop)]
        .groupby("_condition")
        .agg(
            noop_pred_clean=("_pred_clean", "mean"),
            noop_R_final=("_R_final", "mean"),
            noop_functional_score=("_functional_score", "mean"),
        )
        .reset_index()
    )
    if len(noop) == 0:
        # fallback first action per condition
        noop = (
            lpf_df.sort_values("_rank_within_condition")
            .groupby("_condition")
            .head(1)
            .groupby("_condition")
            .agg(
                noop_pred_clean=("_pred_clean", "mean"),
                noop_R_final=("_R_final", "mean"),
                noop_functional_score=("_functional_score", "mean"),
            )
            .reset_index()
        )

    fixed = (
        lpf_df[lpf_df["_action"].map(action_is_fixed_candidate)]
        .groupby("_condition")
        .agg(
            fixed_pred_clean=("_pred_clean", "mean"),
            fixed_R_final=("_R_final", "mean"),
            fixed_functional_score=("_functional_score", "mean"),
        )
        .reset_index()
    )
    if len(fixed) == 0:
        fixed = (
            lpf_df.groupby("_condition")
            .agg(
                fixed_pred_clean=("_pred_clean", "median"),
                fixed_R_final=("_R_final", "median"),
                fixed_functional_score=("_functional_score", "median"),
            )
            .reset_index()
        )

    replay = replay.merge(oracle, on="_condition", how="left")
    replay = replay.merge(noop, on="_condition", how="left")
    replay = replay.merge(fixed, on="_condition", how="left")

    replay["pred_vs_noop_score"] = replay["_functional_score"] - replay["noop_functional_score"]
    replay["pred_vs_fixed_score"] = replay["_functional_score"] - replay["fixed_functional_score"]
    replay["gap_vs_oracle_score"] = replay["oracle_functional_score"] - replay["_functional_score"]
    replay["exact_action_is_oracle"] = (replay["selected_action_for_factor"].astype(str) == replay["oracle_action"].astype(str)).astype(int)

    summary = []
    for keys, sub in replay.groupby(["mode", "model"]):
        mode, model = keys
        summary.append({
            "mode": mode,
            "model": model,
            "n": int(len(sub)),
            "mean_pred_clean": safe_mean(sub["_pred_clean"]),
            "mean_R_final": safe_mean(sub["_R_final"]),
            "mean_functional_score": safe_mean(sub["_functional_score"]),
            "mean_noop_score": safe_mean(sub["noop_functional_score"]),
            "mean_fixed_score": safe_mean(sub["fixed_functional_score"]),
            "mean_oracle_score": safe_mean(sub["oracle_functional_score"]),
            "gain_vs_noop": safe_mean(sub["pred_vs_noop_score"]),
            "gain_vs_fixed": safe_mean(sub["pred_vs_fixed_score"]),
            "gap_vs_oracle": safe_mean(sub["gap_vs_oracle_score"]),
            "exact_action_oracle_rate": safe_mean(sub["exact_action_is_oracle"]),
        })
    return replay, pd.DataFrame(summary)


# ============================================================
# Verdict helper
# ============================================================

def make_verdict(cluster_eval: pd.DataFrame, clf_metrics: pd.DataFrame, replay_summary: pd.DataFrame) -> Dict[str, Any]:
    verdict = {
        "cluster_status": "UNKNOWN",
        "classifier_status": "UNKNOWN",
        "replay_status": "UNKNOWN",
        "overall": "UNKNOWN",
    }

    if len(cluster_eval) > 0 and "silhouette" in cluster_eval.columns:
        best_sil = float(pd.to_numeric(cluster_eval["silhouette"], errors="coerce").max())
        verdict["best_silhouette"] = best_sil
        if best_sil >= 0.25:
            verdict["cluster_status"] = "PASS-Strong"
        elif best_sil >= 0.10:
            verdict["cluster_status"] = "PASS-Lite"
        else:
            verdict["cluster_status"] = "WEAK"

    if len(clf_metrics) > 0:
        # Focus on leave-one-family first, then graph_groupkfold.
        focus = clf_metrics[clf_metrics["mode"].isin(["leave_one_family", "leave_one_condition", "graph_groupkfold"])].copy()
        if len(focus) == 0:
            focus = clf_metrics.copy()
        best_macro = float(pd.to_numeric(focus["macro_f1"], errors="coerce").max())
        best_bal = float(pd.to_numeric(focus["balanced_accuracy"], errors="coerce").max())
        verdict["best_classifier_macro_f1"] = best_macro
        verdict["best_classifier_balanced_acc"] = best_bal
        if best_macro >= 0.70:
            verdict["classifier_status"] = "PASS-Strong"
        elif best_macro >= 0.45:
            verdict["classifier_status"] = "PASS-Lite"
        else:
            verdict["classifier_status"] = "WEAK/FAIL"

    if len(replay_summary) > 0:
        gain_fixed = float(pd.to_numeric(replay_summary["gain_vs_fixed"], errors="coerce").max())
        gain_noop = float(pd.to_numeric(replay_summary["gain_vs_noop"], errors="coerce").max())
        gap_oracle = float(pd.to_numeric(replay_summary["gap_vs_oracle"], errors="coerce").min())
        verdict["best_replay_gain_vs_fixed"] = gain_fixed
        verdict["best_replay_gain_vs_noop"] = gain_noop
        verdict["best_replay_gap_vs_oracle"] = gap_oracle
        if gain_fixed > 0 and gain_noop > 0:
            verdict["replay_status"] = "PASS"
        elif gain_noop > 0:
            verdict["replay_status"] = "PARTIAL"
        else:
            verdict["replay_status"] = "FAIL/WEAK"

    statuses = [verdict["cluster_status"], verdict["classifier_status"], verdict["replay_status"]]
    if all(("PASS" in s or s == "PARTIAL") for s in statuses):
        verdict["overall"] = "PASS-Lite-or-Better"
    if verdict["cluster_status"] == "PASS-Strong" and verdict["classifier_status"] == "PASS-Strong" and verdict["replay_status"] == "PASS":
        verdict["overall"] = "PASS-Strong"
    if "FAIL" in verdict["classifier_status"] or "FAIL" in verdict["replay_status"]:
        verdict["overall"] = "PARTIAL/FAIL-Bottleneck"

    return verdict


# ============================================================
# Main
# ============================================================

def main() -> None:
    warnings.filterwarnings("ignore")

    print("=" * 80)
    print("DA-ASA-2J: OutcomeTopology-Derived LatentPolicyFactorization")
    print("=" * 80)
    print(f"ROOT    = {ROOT}")
    print(f"OUT_DIR = {OUT_DIR}")

    files = load_inputs()

    policy_df = choose_policy_table(files)
    policy_df.to_csv(OUT_DIR / "da_asa2j_raw_policy_table_used.csv", index=False)

    eff, effect_meta = construct_effect_vectors(policy_df)
    eff.to_csv(OUT_DIR / "da_asa2j_policy_effect_vectors.csv", index=False)
    save_json(effect_meta, OUT_DIR / "da_asa2j_effect_vector_builder.json")
    print("\n[1] PolicyEffectVector built")
    print(json.dumps(effect_meta, indent=2, ensure_ascii=False))

    lpf, cluster_eval, cluster_summary, cluster_meta = cluster_effect_vectors(eff, effect_meta["effect_cols"])
    lpf.to_csv(OUT_DIR / "da_asa2j_latent_policy_factor_table.csv", index=False)
    cluster_eval.to_csv(OUT_DIR / "da_asa2j_cluster_eval.csv", index=False)
    cluster_summary.to_csv(OUT_DIR / "da_asa2j_latent_factor_summary.csv", index=False)
    save_json(cluster_meta, OUT_DIR / "da_asa2j_cluster_builder.json")
    print("\n[2] Latent policy factors discovered")
    print(json.dumps(cluster_meta, indent=2, ensure_ascii=False))
    print(cluster_summary.head(20).to_string(index=False))

    try:
        train_df, feature_meta = build_feature_label_table(files, lpf)
        train_df.to_csv(OUT_DIR / "da_asa2j_dsta_latent_factor_training_table.csv", index=False)
        save_json(feature_meta, OUT_DIR / "da_asa2j_feature_label_builder.json")
        print("\n[3] Feature-label table built")
        print(json.dumps({k: v for k, v in feature_meta.items() if k != "feature_cols"}, indent=2, ensure_ascii=False))
        print(f"Feature cols: {len(feature_meta['feature_cols'])}")

        if len(train_df) >= 5 and len(set(train_df["latent_policy_factor"])) >= 2 and len(feature_meta["feature_cols"]) >= 2:
            clf_metrics, clf_preds = evaluate_classifier(
                train_df,
                feature_meta["feature_cols"],
                target_col="latent_policy_factor",
                group_col="_graph_id",
                family_col="_condition_family",
            )
        else:
            clf_metrics = pd.DataFrame()
            clf_preds = pd.DataFrame()
            print("[WARN] Not enough labeled rows/classes/features for classifier evaluation.")

        clf_metrics.to_csv(OUT_DIR / "da_asa2j_dsta_to_latent_factor_metrics.csv", index=False)
        clf_preds.to_csv(OUT_DIR / "da_asa2j_dsta_to_latent_factor_predictions.csv", index=False)

        print("\n[4] DSTA -> latent factor evaluation")
        if len(clf_metrics) > 0:
            print(
                clf_metrics.sort_values(["macro_f1", "balanced_accuracy"], ascending=False)
                .head(20)
                .to_string(index=False)
            )
        else:
            print("No classifier metrics generated.")

        factor_action_map = build_factor_to_action_map(lpf)
        factor_action_map.to_csv(OUT_DIR / "da_asa2j_latent_factor_to_action_map.csv", index=False)

        replay, replay_summary = table_replay_from_predictions(
            lpf, clf_preds, factor_action_map, mode_filter="leave_one_family", model_filter="logreg"
        )
        replay.to_csv(OUT_DIR / "da_asa2j_latent_factor_policy_replay_predictions.csv", index=False)
        replay_summary.to_csv(OUT_DIR / "da_asa2j_latent_factor_policy_replay_summary.csv", index=False)

    except Exception as e:
        print("\n[ERROR] DSTA -> latent factor / replay stage failed:")
        print(str(e))
        save_json({"error": str(e)}, OUT_DIR / "da_asa2j_downstream_error.json")
        clf_metrics = pd.DataFrame()
        replay_summary = pd.DataFrame()

    verdict = make_verdict(cluster_eval, clf_metrics, replay_summary)
    save_json(verdict, OUT_DIR / "da_asa2j_diagnostics.json")

    print("\n[5] Verdict")
    print(json.dumps(verdict, indent=2, ensure_ascii=False))

    print("\nSaved outputs:")
    for p in sorted(OUT_DIR.glob("da_asa2j_*")):
        print(" ", p)

    print("\nInterpretation guide:")
    print("  PASS-Strong if:")
    print("    1. cluster_eval best silhouette >= 0.25")
    print("    2. leave-one-family / leave-one-condition macro-F1 >= 0.70")
    print("    3. replay gain_vs_fixed > 0 and gain_vs_noop > 0")
    print("  PASS-Lite if:")
    print("    latent factors are weakly clustered but DSTA predicts them better than hand factors,")
    print("    or replay improves no-op while still below fixed/oracle.")
    print("  FAIL-Bottleneck if:")
    print("    outcome topology clusters exist but DSTA cannot predict them under LOCO/LOFO,")
    print("    meaning latent factors are condition-specific rather than transferable.")
    print("  Key comparison:")
    print("    Compare this 2J replay summary against 2G flat unseen policy and 2I hierarchical rule.")
    print("    If 2J improves LOFO replay, then HumanPolicyFactor != LatentOutcomePolicyFactor is resolved productively.")


if __name__ == "__main__":
    main()
