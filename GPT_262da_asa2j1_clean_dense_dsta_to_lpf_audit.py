# -*- coding: utf-8 -*-
"""
DA-ASA-2J.1
Clean DenseDSTA -> LatentPolicyFactor Audit

Purpose
-------
After DA-ASA-2J produced strong outcome-topology-derived LatentPolicyFactors (LPF),
this script audits whether CLEAN dense DSTA features can predict LPF labels.

Frozen from 2J:
  PolicyResults -> EffectVector -> LatentPolicyFactor -> LPF policy replay  = PASS-Strong

Open question for 2J.1:
  DenseDSTA -> LatentPolicyFactor

This script avoids:
  - fold columns
  - prediction columns from previous classifiers
  - policy outcome / replay leakage
  - raw action/outcome table columns
  - condition label one-hot style columns where possible

It prefers:
  - dsta_signed_attractor / transport / rotation / delta / attractor-distance features
  - graph-level or graph+condition level features
  - rows keyed by graph_id + condition

Run:
  python da_asa2j1_clean_dense_dsta_to_lpf_audit_v4.py
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "da_asa2j1_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KEY_COLS = ["graph_id", "condition"]

# Inputs produced by 2J
LPF_TABLE_CANDIDATES = [
    "da_asa2j_latent_policy_factor_table.csv",
    "da_asa2j_dsta_latent_factor_training_table.csv",
]
LPF_ACTION_MAP_CANDIDATES = [
    "da_asa2j_latent_factor_to_action_map.csv",
]
RAW_POLICY_CANDIDATES = [
    "da_asa2j_raw_policy_table_used.csv",
]

# Dense DSTA / 2F fallback table candidates
FEATURE_TABLE_CANDIDATES = [
    # Canonical names if manually created earlier
    "da_asa2d0h5_feature_table.csv",
    "da_asa2d0h5_dense_feature_table.csv",
    "da_asa2d0h5_final_feature_table.csv",
    "da_asa2d_feature_table_h5.csv",
    "da_asa2d_feature_table_dense.csv",
    "da_asa2d_feature_table.csv",
    "da_asa2d1b_feature_table.csv",
    "da_asa2d1b_training_table.csv",
    "da_asa2d1b_input_table.csv",
    # 2F fallback output
    "da_asa2f_fallback_feature_table.csv",
]

DSTA_INCLUDE_PATTERNS = [
    "dsta",
    "signed_attractor",
    "attractor",
    "transport",
    "rotation",
    "delta_align",
    "delta_norm",
    "center_signed",
    "center_rotation",
    "d_to_stable",
    "d_to_update",
    "d_to_override",
    "d_to_exception",
    "proj_",
    "relative_position",
    "between_excess",
]

# Strong exclusions to prevent leakage and previous prediction artifacts.
LEAK_EXCLUDE_PATTERNS = [
    "policy_outcome",
    "mean_delta_r",
    "pred_clean",
    "r_final",
    "functional_score",
    "oracle",
    "best_fixed",
    "noop",
    "replay",
    "outcome",
    "action_class",
    "raw_policy",
    "outcome_representative",
    "policy_class",
    "predicted",
    "prediction",
    "y_pred",
    "y_true",
    "true_label",
    "fold",
    "cluster",
    "latent",
    "lpf",
    "label",
    "target",
    "is_hallucination",
    "is_source_claim",
    "is_protected",
    "is_core_closure",
    "is_equal_evidence",
    "is_override",
    "condition_family",
]

BASELINE_INCLUDE_PATTERNS = [
    "baseline",
    "boundary_distance",
    "c_logit",
    "e_logit",
]

RANDOM_SEED = 42
MAX_FEATURES_PER_SET = 300


def find_file(name: str, root: Path = ROOT) -> Optional[Path]:
    direct = root / name
    if direct.exists():
        return direct
    hits = list(root.rglob(name))
    if hits:
        # Prefer files in recent/expected output dirs
        hits = sorted(hits, key=lambda p: (len(p.parts), str(p)))
        return hits[0]
    return None


def load_first_existing(candidates: List[str], required: bool = True) -> Optional[pd.DataFrame]:
    for name in candidates:
        p = find_file(name)
        if p is not None:
            print(f"[LOAD] {name} -> {p}")
            return pd.read_csv(p)
    if required:
        raise FileNotFoundError(f"None of these files found: {candidates}")
    return None


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()
    return df




def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return first matching column by exact or case-insensitive name."""
    cols = list(df.columns)
    for cand in candidates:
        if cand in df.columns:
            return cand
    lower = {str(c).lower(): c for c in cols}
    for cand in candidates:
        hit = lower.get(str(cand).lower())
        if hit is not None:
            return hit
    return None

def standardize_keys(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = normalize_cols(df)
    lower = {c.lower(): c for c in df.columns}
    g = lower.get("graph_id") or lower.get("graph") or lower.get("gid")
    c = lower.get("condition") or lower.get("cond") or lower.get("condition_name")
    if g is None or c is None:
        raise KeyError(f"[{name}] cannot find graph_id/condition. Columns={list(df.columns)}")
    if g != "graph_id":
        df = df.rename(columns={g: "graph_id"})
    if c != "condition":
        df = df.rename(columns={c: "condition"})
    df["graph_id"] = df["graph_id"].astype(str).str.strip()
    df["condition"] = df["condition"].astype(str).str.strip()
    return df


def safe_to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    return out


def build_lpf_labels() -> pd.DataFrame:
    """
    Prefer 2J dsta training table if it already has graph_id/condition + latent factor.
    Otherwise map condition + selected action to LPF using latent_policy_factor_table.
    """
    training = load_first_existing(["da_asa2j_dsta_latent_factor_training_table.csv"], required=False)
    if training is not None:
        training = standardize_keys(training, "2J training labels")
        lpf_col = infer_lpf_col(training)
        out = training[KEY_COLS + [lpf_col]].copy()
        out = out.rename(columns={lpf_col: "latent_policy_factor"})
        out = out.dropna(subset=["latent_policy_factor"]).drop_duplicates(KEY_COLS)
        if len(out) > 0:
            print(f"[LABEL] using 2J training table labels: rows={len(out)}")
            return out

    lpf_table = load_first_existing(LPF_TABLE_CANDIDATES, required=True)
    lpf_table = normalize_cols(lpf_table)
    lpf_col = infer_lpf_col(lpf_table)

    # If LPF table already graph-level, use directly.
    if "graph_id" in lpf_table.columns and "condition" in lpf_table.columns:
        lpf_table = standardize_keys(lpf_table, "LPF table")
        out = lpf_table[KEY_COLS + [lpf_col]].copy()
        out = out.rename(columns={lpf_col: "latent_policy_factor"})
        return out.dropna().drop_duplicates(KEY_COLS)

    # Fallback: need an action-level or condition-level mapping.
    raise ValueError(
        "Could not build graph-level LPF labels. Expected da_asa2j_dsta_latent_factor_training_table.csv "
        "with graph_id, condition and latent factor columns."
    )


def infer_lpf_col(df: pd.DataFrame) -> str:
    candidates = [
        "latent_policy_factor",
        "lpf",
        "latent_factor",
        "policy_factor",
        "cluster_label",
        "cluster",
        "best_cluster",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # Fuzzy
    for c in df.columns:
        lc = c.lower()
        if ("latent" in lc and "factor" in lc) or lc in ["lpf", "cluster"]:
            return c
    raise KeyError(f"Cannot infer LPF column. Columns={list(df.columns)}")


def load_feature_table_or_discover() -> Tuple[pd.DataFrame, Dict]:
    for name in FEATURE_TABLE_CANDIDATES:
        p = find_file(name)
        if p is not None:
            df = pd.read_csv(p)
            df = standardize_keys(df, name)
            print(f"[FEATURE] using feature table {p}, rows={len(df)}, cols={df.shape[1]}")
            return df, {"mode": "existing_feature_table", "path": str(p)}

    # Discover CSVs with graph_id + condition and DSTA-like numeric columns.
    dfs = []
    used_files = []
    skipped = {"no_keys": 0, "no_dsta_cols": 0, "read_error": 0, "excluded_file": 0}
    exclude_file_patterns = [
        "policy_results", "replay", "2e_", "2f_predictions", "2g_", "2h_", "2i_", "2j_",
        "confusion", "summary", "diagnostics", "metrics", "action_map", "raw_policy"
    ]

    for p in ROOT.rglob("*.csv"):
        rel = str(p.relative_to(ROOT)).lower()
        if any(x in rel for x in exclude_file_patterns):
            skipped["excluded_file"] += 1
            continue
        try:
            df = pd.read_csv(p)
            df = standardize_keys(df, str(p))
        except Exception:
            skipped["no_keys"] += 1
            continue

        numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c not in KEY_COLS]
        dsta_cols = [c for c in numeric if is_dsta_col(c) and not is_leak_col(c)]
        if not dsta_cols:
            skipped["no_dsta_cols"] += 1
            continue

        sub = df[KEY_COLS + dsta_cols].copy()
        prefix = re.sub(r"[^A-Za-z0-9]+", "_", str(p.relative_to(ROOT)))[:80]
        rename = {c: f"{prefix}__{c}" for c in dsta_cols}
        sub = sub.rename(columns=rename)
        sub = sub.groupby(KEY_COLS, as_index=False).mean(numeric_only=True)
        dfs.append(sub)
        used_files.append(str(p))

    if not dfs:
        raise FileNotFoundError("Could not find any DSTA-like CSV feature sources.")

    out = dfs[0]
    for fr in dfs[1:]:
        overlap = [c for c in fr.columns if c in out.columns and c not in KEY_COLS]
        if overlap:
            fr = fr.drop(columns=overlap)
        out = out.merge(fr, on=KEY_COLS, how="outer", validate="one_to_one")
        out = out.loc[:, ~pd.Index(out.columns).duplicated()].copy()

    info = {
        "mode": "discovered_dsta_csvs",
        "used_files_count": len(used_files),
        "used_files": used_files,
        "skipped": skipped,
        "rows": int(len(out)),
        "cols": int(out.shape[1]),
    }
    print(f"[FEATURE] discovered rows={len(out)} cols={out.shape[1]} files={len(used_files)}")
    return out, info


def is_dsta_col(c: str) -> bool:
    lc = c.lower()
    return any(p in lc for p in DSTA_INCLUDE_PATTERNS)


def is_baseline_col(c: str) -> bool:
    lc = c.lower()
    return any(p in lc for p in BASELINE_INCLUDE_PATTERNS)


def is_leak_col(c: str) -> bool:
    lc = c.lower()
    return any(p in lc for p in LEAK_EXCLUDE_PATTERNS)


def build_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in KEY_COLS]

    clean_dsta = [c for c in numeric_cols if is_dsta_col(c) and not is_leak_col(c)]
    strict_transport = [
        c for c in clean_dsta
        if any(x in c.lower() for x in [
            "transport", "delta_align", "delta_norm", "rotation", "signed_align_vs_clean",
            "d_to_stable", "d_to_update", "d_to_override", "d_to_exception"
        ])
    ]
    no_distance = [c for c in strict_transport if "distance" not in c.lower()]
    baseline = [c for c in numeric_cols if is_baseline_col(c) and not is_leak_col(c)]

    # Defensive cap by variance: remove zero-variance and cap size.
    def clean_list(cols: List[str]) -> List[str]:
        cols = list(dict.fromkeys(cols))
        good = []
        for c in cols:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() >= 3 and float(s.fillna(s.mean()).var()) > 1e-12:
                good.append(c)
        if len(good) > MAX_FEATURES_PER_SET:
            # keep most complete + high variance
            scores = []
            for c in good:
                s = pd.to_numeric(df[c], errors="coerce")
                scores.append((s.notna().mean(), float(s.fillna(s.mean()).var()), c))
            scores = sorted(scores, reverse=True)
            good = [c for _, _, c in scores[:MAX_FEATURES_PER_SET]]
        return good

    sets = {
        "clean_dense_dsta": clean_list(clean_dsta),
        "strict_transport_dsta": clean_list(strict_transport),
        "strict_transport_no_distance": clean_list(no_distance),
        "baseline_only_sanity": clean_list(baseline),
    }

    # Add mixed for sanity but not as primary.
    sets["baseline_plus_clean_dsta"] = clean_list(sets["baseline_only_sanity"] + sets["clean_dense_dsta"])

    return {k: v for k, v in sets.items() if len(v) > 0}


def infer_groups(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    groups = {
        "group_graph": df["graph_id"].astype(str).values,
        "leave_one_condition": df["condition"].astype(str).values,
    }
    # Coarse family heuristic
    def fam(cond: str) -> str:
        s = str(cond).lower()
        if "closure" in s:
            return "closure"
        if "competition" in s:
            return "competition"
        if "hallucination" in s:
            return "hallucination_like"
        if "stable" in s:
            return "stable"
        return "other"
    groups["leave_one_family"] = df["condition"].map(fam).astype(str).values
    return groups


def evaluate_cv(df: pd.DataFrame, features: List[str], target_col: str, group_values: np.ndarray, mode: str, feature_set_name: str) -> Tuple[dict, pd.DataFrame]:
    X = safe_to_numeric_df(df[features])
    y = df[target_col].astype(str).values
    groups = np.asarray(group_values)

    preds = []
    fold_rows = []

    logo = LeaveOneGroupOut()
    for fold, (tr, te) in enumerate(logo.split(X, y, groups)):
        y_train = y[tr]
        y_test = y[te]
        test_group = str(groups[te][0])

        absent = sorted(set(y_test) - set(y_train))
        absent_frac = len([v for v in y_test if v in absent]) / max(1, len(y_test))

        if len(set(y_train)) < 2:
            y_pred = np.array([pd.Series(y_train).mode().iloc[0] if len(y_train) else "MISSING"] * len(te))
            model_name = "constant_train"
        else:
            clf = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=RANDOM_SEED,
                )),
            ])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X.iloc[tr], y_train)
            y_pred = clf.predict(X.iloc[te])
            model_name = "logreg_balanced_lbfgs"

        acc = accuracy_score(y_test, y_pred)
        macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        try:
            bal = balanced_accuracy_score(y_test, y_pred)
        except Exception:
            bal = np.nan

        fold_rows.append({
            "mode": mode,
            "feature_set": feature_set_name,
            "target": target_col,
            "fold": fold,
            "heldout_group": test_group,
            "n_test": int(len(te)),
            "n_train": int(len(tr)),
            "model": model_name,
            "accuracy": float(acc),
            "macro_f1": float(macro),
            "weighted_f1": float(weighted),
            "balanced_accuracy": float(bal) if not np.isnan(bal) else np.nan,
            "absent_true_labels": "|".join(absent),
            "absent_true_label_frac": float(absent_frac),
        })

        pp = df.iloc[te][KEY_COLS].copy()
        pp["mode"] = mode
        pp["feature_set"] = feature_set_name
        pp["target"] = target_col
        pp["fold"] = fold
        pp["heldout_group"] = test_group
        pp["y_true"] = y_test
        pp["y_pred"] = y_pred
        preds.append(pp)

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    summary = {
        "mode": mode,
        "feature_set": feature_set_name,
        "target": target_col,
        "n_folds": int(len(fold_df)),
        "n_rows": int(len(df)),
        "n_features": int(len(features)),
        "mean_accuracy": float(fold_df["accuracy"].mean()),
        "mean_macro_f1": float(fold_df["macro_f1"].mean()),
        "mean_weighted_f1": float(fold_df["weighted_f1"].mean()),
        "mean_balanced_accuracy": float(fold_df["balanced_accuracy"].mean()),
        "min_accuracy": float(fold_df["accuracy"].min()),
        "min_macro_f1": float(fold_df["macro_f1"].min()),
        "mean_absent_true_label_frac": float(fold_df["absent_true_label_frac"].mean()),
    }
    return summary, pred_df


def try_replay(pred_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Optional: if latent factor -> action map exists, choose representative action for predicted LPF.
    This is a sanity replay table, not main classifier metric.
    """
    action_map = load_first_existing(LPF_ACTION_MAP_CANDIDATES, required=False)
    raw_policy = load_first_existing(RAW_POLICY_CANDIDATES, required=False)
    if action_map is None or raw_policy is None or pred_df.empty:
        return None

    action_map = normalize_cols(action_map)
    raw_policy = normalize_cols(raw_policy)

    # raw policy table from 2J is usually condition/action-level, not graph-level.
    # It may not contain graph_id; this is OK for replay sanity audit.
    cond_col = find_col(raw_policy, ["condition", "cond", "condition_name"])
    if cond_col is None:
        return None
    if cond_col != "condition":
        raw_policy = raw_policy.rename(columns={cond_col: "condition"})
    raw_policy["condition"] = raw_policy["condition"].astype(str).str.strip()

    lpf_col = infer_lpf_col(action_map)
    action_col = None
    for c in ["recommended_action", "representative_action", "action_class", "best_action"]:
        if c in action_map.columns:
            action_col = c
            break
    if action_col is None:
        # Fallback: any action-like col
        for c in action_map.columns:
            if "action" in c.lower():
                action_col = c
                break
    if action_col is None:
        return None

    metric_cols = [c for c in ["pred_clean", "R_final", "functional_score", "mean_pred_clean", "mean_R_final"] if c in raw_policy.columns]
    if not metric_cols:
        # Try unprefixed 2J raw policy columns
        metric_cols = [c for c in raw_policy.select_dtypes(include=[np.number]).columns if c not in KEY_COLS]
    if not metric_cols:
        return None

    act_col_raw = None
    for c in ["action_class", "canonical_action_class", "action", "policy"]:
        if c in raw_policy.columns:
            act_col_raw = c
            break
    if act_col_raw is None:
        return None

    mp = action_map[[lpf_col, action_col]].drop_duplicates().rename(
        columns={lpf_col: "y_pred", action_col: "pred_action_class"}
    )

    rows = []
    use_pred = pred_df[pred_df["target"] == "latent_policy_factor"].copy()
    if use_pred.empty:
        return None

    joined = use_pred.merge(mp, on="y_pred", how="left")
    raw_policy[act_col_raw] = raw_policy[act_col_raw].astype(str)
    joined["pred_action_class"] = joined["pred_action_class"].astype(str)

    # If raw policy is condition-level, join on condition + action.
    # If it unexpectedly has graph_id, still prefer graph_id+condition+action.
    join_keys_left = ["condition", "pred_action_class"]
    join_keys_right = ["condition", act_col_raw]
    raw_cols = ["condition", act_col_raw] + metric_cols
    if "graph_id" in raw_policy.columns and "graph_id" in joined.columns:
        join_keys_left = ["graph_id", "condition", "pred_action_class"]
        join_keys_right = ["graph_id", "condition", act_col_raw]
        raw_cols = ["graph_id", "condition", act_col_raw] + metric_cols

    raw_cols = list(dict.fromkeys(raw_cols))
    merged = joined.merge(
        raw_policy[raw_cols].drop_duplicates(),
        left_on=join_keys_left,
        right_on=join_keys_right,
        how="left",
    )

    for (mode, fs), sub in merged.groupby(["mode", "feature_set"]):
        rec = {
            "mode": mode,
            "feature_set": fs,
            "n": int(len(sub)),
            "matched_rate": float(sub[metric_cols].notna().any(axis=1).mean()),
        }
        for m in metric_cols[:10]:
            rec[f"mean_{m}"] = float(pd.to_numeric(sub[m], errors="coerce").mean())
        rows.append(rec)

    return pd.DataFrame(rows)


def main():
    labels = build_lpf_labels()
    features, feature_info = load_feature_table_or_discover()

    data = labels.merge(features, on=KEY_COLS, how="left", validate="one_to_one")
    missing_key_feature_rate = float(data.drop(columns=KEY_COLS + ["latent_policy_factor"], errors="ignore").isna().all(axis=1).mean())

    feature_sets = build_feature_sets(data)
    if not feature_sets:
        raise ValueError("No clean DSTA feature sets available after filtering.")

    groups = infer_groups(data)

    summaries = []
    preds = []
    target_col = "latent_policy_factor"

    for fs_name, cols in feature_sets.items():
        for mode, group_values in groups.items():
            summary, pred_df = evaluate_cv(data, cols, target_col, group_values, mode, fs_name)
            summaries.append(summary)
            preds.append(pred_df)

    summary_df = pd.DataFrame(summaries)
    preds_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()

    # Best by mode
    best_by_mode = (
        summary_df.sort_values(["mode", "mean_macro_f1", "mean_accuracy"], ascending=[True, False, False])
        .groupby("mode", as_index=False)
        .head(1)
    )

    replay_df = try_replay(preds_df)
    if replay_df is not None:
        replay_df.to_csv(OUT_DIR / "da_asa2j1_optional_lpf_replay_summary.csv", index=False, encoding="utf-8-sig")

    # Confusion for best graph/condition/family
    for _, row in best_by_mode.iterrows():
        mode = row["mode"]
        fs = row["feature_set"]
        sub = preds_df[(preds_df["mode"] == mode) & (preds_df["feature_set"] == fs)]
        if len(sub):
            labels_sorted = sorted(set(sub["y_true"]) | set(sub["y_pred"]))
            cm = pd.DataFrame(
                confusion_matrix(sub["y_true"], sub["y_pred"], labels=labels_sorted),
                index=[f"true__{x}" for x in labels_sorted],
                columns=[f"pred__{x}" for x in labels_sorted],
            )
            cm.to_csv(OUT_DIR / f"da_asa2j1_confusion_{mode}_{fs}.csv", encoding="utf-8-sig")

    # Verdict
    best_graph = best_by_mode[best_by_mode["mode"] == "group_graph"]
    best_cond = best_by_mode[best_by_mode["mode"] == "leave_one_condition"]
    best_family = best_by_mode[best_by_mode["mode"] == "leave_one_family"]

    graph_f1 = float(best_graph["mean_macro_f1"].iloc[0]) if len(best_graph) else 0.0
    cond_f1 = float(best_cond["mean_macro_f1"].iloc[0]) if len(best_cond) else 0.0
    fam_f1 = float(best_family["mean_macro_f1"].iloc[0]) if len(best_family) else 0.0

    if graph_f1 >= 0.95 and cond_f1 >= 0.70 and fam_f1 >= 0.55:
        verdict = "PASS_STRONG_DENSE_DSTA_TO_LPF"
    elif graph_f1 >= 0.90 and (cond_f1 >= 0.55 or fam_f1 >= 0.45):
        verdict = "PASS_LITE_DENSE_DSTA_TO_LPF"
    elif graph_f1 >= 0.90:
        verdict = "PASS_KNOWN_CONDITION_ONLY_DENSE_DSTA_TO_LPF"
    else:
        verdict = "FAIL_DENSE_DSTA_TO_LPF"

    diagnostics = {
        "stage": "DA-ASA-2J.1",
        "purpose": "Clean DenseDSTA -> LatentPolicyFactor audit after 2J",
        "verdict": verdict,
        "root": str(ROOT),
        "output_dir": str(OUT_DIR),
        "n_rows": int(len(data)),
        "n_graphs": int(data["graph_id"].nunique()),
        "n_conditions": int(data["condition"].nunique()),
        "n_lpf_classes": int(data["latent_policy_factor"].nunique()),
        "missing_key_feature_rate": missing_key_feature_rate,
        "feature_info": feature_info,
        "feature_sets": {k: len(v) for k, v in feature_sets.items()},
        "best_by_mode": best_by_mode.to_dict(orient="records"),
        "graph_best_macro_f1": graph_f1,
        "condition_best_macro_f1": cond_f1,
        "family_best_macro_f1": fam_f1,
        "primary_caveat": "This audit excludes fold/prediction/outcome/replay leakage columns and focuses on DSTA-like numeric features.",
    }

    # Save
    summary_df.to_csv(OUT_DIR / "da_asa2j1_dsta_to_lpf_cv_summary.csv", index=False, encoding="utf-8-sig")
    best_by_mode.to_csv(OUT_DIR / "da_asa2j1_best_by_mode.csv", index=False, encoding="utf-8-sig")
    preds_df.to_csv(OUT_DIR / "da_asa2j1_dsta_to_lpf_predictions.csv", index=False, encoding="utf-8-sig")
    data.to_csv(OUT_DIR / "da_asa2j1_training_table_used.csv", index=False, encoding="utf-8-sig")
    with open(OUT_DIR / "da_asa2j1_feature_sets.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in feature_sets.items()}, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / "da_asa2j1_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("DA-ASA-2J.1 CLEAN DENSEDSTA -> LPF AUDIT")
    print("=" * 80)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print("\n[BEST BY MODE]")
    print(best_by_mode.to_string(index=False))


if __name__ == "__main__":
    main()
