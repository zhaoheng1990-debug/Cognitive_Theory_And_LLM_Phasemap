# -*- coding: utf-8 -*-
"""
DA-ASA-2H: Mechanism-Factored Policy Rule Audit

Purpose
-------
2G showed that direct action-label prediction does not generalize well to unseen
condition/family settings. 2H tests whether a structured policy decomposition is
more transferable than a single flat action label.

Core idea:
    DSTA features -> factor labels -> recomposed transferable policy

This script does NOT run model forward. It reuses 2F/2G tables.

Expected inputs, relative to this script directory:
    da_asa2f_outputs/da_asa2f_predictions_long.csv
    da_asa2f_outputs/da_asa2f_fallback_feature_table.csv
    da_asa2g_outputs/da_asa2g_feature_sets.json    optional
    da_asa2f_outputs/da_asa2f_feature_sets.json    fallback

Outputs:
    da_asa2h_outputs/
      da_asa2h_diagnostics.json
      da_asa2h_factor_holdout_summary.csv
      da_asa2h_factor_holdout_aggregate.csv
      da_asa2h_factor_predictions_long.csv
      da_asa2h_recomposed_policy_summary.csv
      da_asa2h_recomposed_policy_predictions.csv
      da_asa2h_feature_sets.json
"""

from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "da_asa2h_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KEY_COLS = ["graph_id", "condition"]
RANDOM_SEED = 42

# Conservative preferred feature set order. strict_graph_dsta_core is most leakage-safe;
# dense DSTA is the strongest known-condition representation.
PREFERRED_FEATURE_SETS = [
    "strict_graph_dsta_core",
    "dsta_dense_auto_strict",
    "json__dsta_dense_no_policy_outcome",
    "json__all_numeric_safe_no_policy_outcome",
    "json__policy_diagnostic_safe_numeric",
    "baseline_auto_strict",
    "json__baseline_only",
]


def find_file(candidates: Iterable[str], required: bool = True) -> Optional[Path]:
    for rel in candidates:
        p = ROOT / rel
        if p.exists():
            return p
    # recursive fallback by basename
    for rel in candidates:
        hits = list(ROOT.rglob(Path(rel).name))
        if hits:
            return hits[0]
    if required:
        raise FileNotFoundError(f"Could not find any of: {list(candidates)} under {ROOT}")
    return None


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_key_cols(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    aliases = {
        "graph": "graph_id",
        "gid": "graph_id",
        "cond": "condition",
        "condition_name": "condition",
    }
    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    missing = [c for c in KEY_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"[{name}] missing key columns {missing}. Columns={list(df.columns)}")
    for c in KEY_COLS:
        df[c] = df[c].astype(str).str.strip()
    return df


def canonical_action(x) -> str:
    if pd.isna(x):
        return "MISSING_ACTION"
    s = str(x).strip().upper()
    aliases = {
        "NONE": "NO_INTERVENTION",
        "NOOP": "NO_INTERVENTION",
        "NO_OP": "NO_INTERVENTION",
        "NO INTERVENTION": "NO_INTERVENTION",
        "NO_INTERVENTION": "NO_INTERVENTION",
        "CORE": "CORE_CLOSURE_UPDATE",
        "CORE_UPDATE": "CORE_CLOSURE_UPDATE",
        "CORE_CLOSURE": "CORE_CLOSURE_UPDATE",
        "CORE_CLOSURE_UPDATE": "CORE_CLOSURE_UPDATE",
        "EQUAL": "EQUAL_EVIDENCE_ORDER",
        "EQUAL_EVIDENCE": "EQUAL_EVIDENCE_ORDER",
        "EQUAL_EVIDENCE_ORDER": "EQUAL_EVIDENCE_ORDER",
        "OVERRIDE": "OVERRIDE_THREE_STAGE",
        "OVERRIDE_THREE_STAGE": "OVERRIDE_THREE_STAGE",
        "THREE_STAGE_OVERRIDE": "OVERRIDE_THREE_STAGE",
    }
    if s in aliases:
        return aliases[s]
    if "NO" in s and "INTERVENTION" in s:
        return "NO_INTERVENTION"
    if "CORE" in s or "CLOSURE_UPDATE" in s:
        return "CORE_CLOSURE_UPDATE"
    if "EQUAL" in s:
        return "EQUAL_EVIDENCE_ORDER"
    if "OVERRIDE" in s or "THREE" in s:
        return "OVERRIDE_THREE_STAGE"
    return s


def infer_family(condition: str) -> str:
    c = str(condition).strip().lower()
    if c.startswith("closure"):
        return "closure"
    if c.startswith("competition"):
        return "competition"
    if "hallucination" in c:
        return "hallucination_like"
    if c.startswith("stable"):
        return "stable"
    return "other"


def infer_binding_type(condition: str) -> str:
    c = str(condition).lower()
    if "source_claim" in c:
        return "source_claim"
    if "equal_evidence" in c:
        return "equal_evidence"
    if "direct" in c:
        return "direct_competition"
    if "override" in c:
        return "rule_override"
    if "exception" in c:
        return "exception_binding"
    if "temporal" in c:
        return "temporal_update"
    if "authority" in c:
        return "authority_update"
    if "negation" in c:
        return "negation_rewrite"
    if "update" in c:
        return "canonical_update"
    if "hallucination" in c:
        return "hallucination_like"
    if "weak" in c:
        return "weak_distractor"
    if "stable" in c:
        return "stable"
    return "other"


def make_factor_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "outcome_representative" not in df.columns:
        raise KeyError("labels table must contain outcome_representative")
    df["outcome_representative"] = df["outcome_representative"].map(canonical_action)
    if "raw_policy_class" in df.columns:
        df["raw_policy_class"] = df["raw_policy_class"].map(canonical_action)
    if "condition_family" not in df.columns:
        df["condition_family"] = df["condition"].map(infer_family)

    act = df["outcome_representative"]

    # Binary / low-level transferable factors.
    df["factor_repair_need"] = np.where(act.eq("NO_INTERVENTION"), "NO_REPAIR", "REPAIR")
    df["factor_commit_need"] = np.where(act.isin(["EQUAL_EVIDENCE_ORDER", "OVERRIDE_THREE_STAGE"]), "COMMIT_CORRECTION", "NO_COMMIT_CORRECTION")
    df["factor_attractor_escape_need"] = np.where(act.isin(["CORE_CLOSURE_UPDATE", "OVERRIDE_THREE_STAGE"]), "ATTRACTOR_ESCAPE", "NO_ATTRACTOR_ESCAPE")
    df["factor_order_correction_need"] = np.where(act.isin(["EQUAL_EVIDENCE_ORDER", "OVERRIDE_THREE_STAGE"]), "ORDER_CORRECTION", "NO_ORDER_CORRECTION")
    df["factor_protect_need"] = np.where(act.eq("NO_INTERVENTION"), "PROTECT", "INTERVENE")

    # Repair type only for rows requiring repair. Non-repair remains NONE.
    repair_type_map = {
        "NO_INTERVENTION": "NONE",
        "CORE_CLOSURE_UPDATE": "CORE_CLOSURE_REPAIR",
        "EQUAL_EVIDENCE_ORDER": "EQUAL_EVIDENCE_REPAIR",
        "OVERRIDE_THREE_STAGE": "OVERRIDE_REPAIR",
    }
    df["factor_repair_type"] = act.map(lambda x: repair_type_map.get(x, "OTHER"))

    # Condition/mechanism descriptors for interpretability. These are labels to predict, not input features.
    df["factor_condition_family"] = df["condition_family"].astype(str)
    df["factor_binding_type"] = df["condition"].map(infer_binding_type)

    # A compact rule tuple. This is still less condition-specific than outcome_policy_class,
    # but more structured than flat raw policy.
    df["factor_rule_tuple"] = (
        df["factor_repair_need"].astype(str)
        + "|" + df["factor_attractor_escape_need"].astype(str)
        + "|" + df["factor_commit_need"].astype(str)
    )
    return df


def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    label_path = find_file([
        "da_asa2f_outputs/da_asa2f_predictions_long.csv",
        "da_asa2f_predictions_long.csv",
    ])
    feat_path = find_file([
        "da_asa2f_outputs/da_asa2f_fallback_feature_table.csv",
        "da_asa2f_fallback_feature_table.csv",
    ])
    fs_path = find_file([
        "da_asa2g_outputs/da_asa2g_feature_sets.json",
        "da_asa2f_outputs/da_asa2f_feature_sets.json",
        "da_asa2g_feature_sets.json",
        "da_asa2f_feature_sets.json",
    ])

    labels_raw = normalize_key_cols(pd.read_csv(label_path), "2F predictions_long")
    features = normalize_key_cols(pd.read_csv(feat_path), "2F fallback feature table")
    fs_json = load_json(fs_path)

    # Deduplicate labels: predictions_long repeats keys across feature_set/target.
    base_cols = [
        "graph_id", "condition", "condition_family",
        "raw_policy_class", "outcome_policy_class", "outcome_representative",
    ]
    keep = [c for c in base_cols if c in labels_raw.columns]
    labels = labels_raw[keep].drop_duplicates(KEY_COLS).copy()
    labels = make_factor_labels(labels)

    # Feature set JSON can be either {name:{columns:[...]}} or {name:[...]}
    feature_sets: Dict[str, List[str]] = {}
    for name, val in fs_json.items():
        cols = val.get("columns", []) if isinstance(val, dict) else val
        cols = [c for c in cols if c in features.columns]
        if cols:
            feature_sets[name] = cols

    # Add strict auto feature sets if not present.
    numeric_cols = [c for c in features.select_dtypes(include=[np.number]).columns if c not in KEY_COLS]
    strict_graph = [
        c for c in numeric_cols
        if "dsta" in c.lower()
        and any(tok in c.lower() for tok in ["transport", "delta_align", "rotation_abs", "delta_norm"])
        and not any(bad in c.lower() for bad in ["policy_outcome", "pred_", "true_", "fold"])
    ]
    if strict_graph:
        feature_sets.setdefault("strict_graph_dsta_core_auto", strict_graph[:200])

    dense_dsta = [
        c for c in numeric_cols
        if "dsta" in c.lower()
        and not any(bad in c.lower() for bad in ["policy_outcome", "pred_", "true_", "fold"])
    ]
    if dense_dsta:
        feature_sets.setdefault("dsta_dense_auto", dense_dsta[:250])

    baseline = [
        c for c in numeric_cols
        if any(tok in c.lower() for tok in ["baseline", "boundary_distance"])
        and not any(bad in c.lower() for bad in ["policy_outcome", "pred_", "true_", "fold"])
    ]
    if baseline:
        feature_sets.setdefault("baseline_auto", baseline[:120])

    return labels, features, feature_sets


def metric_row(y_true: List[str], y_pred: List[str], labels_train: Optional[set] = None) -> dict:
    if len(y_true) == 0:
        return {
            "n": 0, "accuracy": np.nan, "macro_f1": np.nan, "weighted_f1": np.nan,
            "balanced_accuracy": np.nan, "absent_true_label_frac": np.nan,
            "n_absent_true_label": np.nan,
        }
    y_true = pd.Series(y_true).astype(str)
    y_pred = pd.Series(y_pred).astype(str)
    absent = 0
    if labels_train is not None:
        absent = int((~y_true.isin(labels_train)).sum())
    try:
        bal = balanced_accuracy_score(y_true, y_pred)
    except Exception:
        bal = np.nan
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(bal) if not pd.isna(bal) else np.nan,
        "absent_true_label_frac": float(absent / len(y_true)),
        "n_absent_true_label": absent,
    }


def fit_predict(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    y_train = y_train.astype(str)
    classes = sorted(y_train.unique().tolist())
    if len(classes) < 2:
        return np.array([classes[0]] * len(X_test), dtype=object)
    # Logistic regression is good enough here and keeps behavior similar to 2F/2G.
    clf = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear" if len(classes) == 2 else "lbfgs",
            random_state=RANDOM_SEED,
        ),
    )
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


def evaluate_holdouts(df: pd.DataFrame, feature_sets: Dict[str, List[str]], targets: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pred_rows = []
    summary_rows = []

    modes = []
    # group_graph: one GroupKFold pass across graph_id.
    modes.append(("group_graph", None))
    # leave one condition/family.
    modes.append(("leave_one_condition", "condition"))
    modes.append(("leave_one_family", "condition_family"))

    for fs_name in PREFERRED_FEATURE_SETS + [k for k in feature_sets.keys() if k not in PREFERRED_FEATURE_SETS]:
        if fs_name not in feature_sets:
            continue
        cols = [c for c in feature_sets[fs_name] if c in df.columns]
        if not cols:
            continue
        for target in targets:
            if target not in df.columns:
                continue
            for mode, group_col in modes:
                if mode == "group_graph":
                    groups = df["graph_id"].astype(str)
                    n_splits = min(5, groups.nunique())
                    if n_splits < 2:
                        continue
                    splitter = GroupKFold(n_splits=n_splits)
                    y_all_true, y_all_pred = [], []
                    for fold, (tr, te) in enumerate(splitter.split(df[cols], df[target], groups=groups)):
                        Xtr, Xte = df.iloc[tr][cols], df.iloc[te][cols]
                        ytr, yte = df.iloc[tr][target].astype(str), df.iloc[te][target].astype(str)
                        yp = fit_predict(Xtr, ytr, Xte)
                        y_all_true.extend(yte.tolist())
                        y_all_pred.extend(yp.tolist())
                        for idx, pred in zip(df.index[te], yp):
                            pred_rows.append({
                                "mode": mode, "holdout": f"fold_{fold}", "feature_set": fs_name,
                                "target": target, "graph_id": df.loc[idx, "graph_id"],
                                "condition": df.loc[idx, "condition"],
                                "condition_family": df.loc[idx, "condition_family"],
                                "true_label": str(df.loc[idx, target]), "pred_label": str(pred),
                                "true_seen_in_train": str(df.loc[idx, target]) in set(ytr.astype(str)),
                            })
                    row = metric_row(y_all_true, y_all_pred, labels_train=None)
                    row.update({"mode": mode, "holdout": "all_folds", "feature_set": fs_name, "target": target, "n_features": len(cols), "n_train_classes_min": np.nan})
                    summary_rows.append(row)
                else:
                    for holdout in sorted(df[group_col].astype(str).unique()):
                        train_mask = df[group_col].astype(str) != holdout
                        test_mask = ~train_mask
                        if train_mask.sum() == 0 or test_mask.sum() == 0:
                            continue
                        Xtr, Xte = df.loc[train_mask, cols], df.loc[test_mask, cols]
                        ytr, yte = df.loc[train_mask, target].astype(str), df.loc[test_mask, target].astype(str)
                        train_classes = set(ytr.unique())
                        yp = fit_predict(Xtr, ytr, Xte)
                        row = metric_row(yte.tolist(), yp.tolist(), labels_train=train_classes)
                        row.update({
                            "mode": mode, "holdout": holdout, "feature_set": fs_name,
                            "target": target, "n_features": len(cols),
                            "n_train_classes_min": int(len(train_classes)),
                        })
                        summary_rows.append(row)
                        for idx, pred in zip(df.loc[test_mask].index, yp):
                            pred_rows.append({
                                "mode": mode, "holdout": holdout, "feature_set": fs_name,
                                "target": target, "graph_id": df.loc[idx, "graph_id"],
                                "condition": df.loc[idx, "condition"],
                                "condition_family": df.loc[idx, "condition_family"],
                                "true_label": str(df.loc[idx, target]), "pred_label": str(pred),
                                "true_seen_in_train": str(df.loc[idx, target]) in train_classes,
                            })
    return pd.DataFrame(summary_rows), pd.DataFrame(pred_rows)


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    group_cols = ["mode", "feature_set", "target"]
    agg = summary.groupby(group_cols, as_index=False).agg(
        n_holdouts=("holdout", "count"),
        n_total=("n", "sum"),
        mean_accuracy=("accuracy", "mean"),
        mean_macro_f1=("macro_f1", "mean"),
        mean_weighted_f1=("weighted_f1", "mean"),
        mean_absent_true_label_frac=("absent_true_label_frac", "mean"),
        min_accuracy=("accuracy", "min"),
        min_macro_f1=("macro_f1", "min"),
    )
    return agg.sort_values(["mode", "target", "mean_macro_f1", "mean_accuracy"], ascending=[True, True, False, False])


def recompose_action_from_factors(rows: pd.DataFrame) -> str:
    # rows contains predictions for one sample across factor targets.
    preds = {r["target"]: str(r["pred_label"]) for _, r in rows.iterrows()}
    repair = preds.get("factor_repair_need", "NO_REPAIR")
    if repair != "REPAIR":
        return "NO_INTERVENTION"
    # Prefer explicit repair_type if predicted.
    repair_type = preds.get("factor_repair_type", "")
    if repair_type == "CORE_CLOSURE_REPAIR":
        return "CORE_CLOSURE_UPDATE"
    if repair_type == "EQUAL_EVIDENCE_REPAIR":
        return "EQUAL_EVIDENCE_ORDER"
    if repair_type == "OVERRIDE_REPAIR":
        return "OVERRIDE_THREE_STAGE"
    # Fall back to factor combination.
    commit = preds.get("factor_commit_need", "NO_COMMIT_CORRECTION")
    escape = preds.get("factor_attractor_escape_need", "NO_ATTRACTOR_ESCAPE")
    order = preds.get("factor_order_correction_need", "NO_ORDER_CORRECTION")
    binding = preds.get("factor_binding_type", "")
    if "override" in binding or "exception" in binding or (commit == "COMMIT_CORRECTION" and escape == "ATTRACTOR_ESCAPE"):
        return "OVERRIDE_THREE_STAGE"
    if "equal" in binding or (commit == "COMMIT_CORRECTION" and order == "ORDER_CORRECTION"):
        return "EQUAL_EVIDENCE_ORDER"
    if escape == "ATTRACTOR_ESCAPE" or repair == "REPAIR":
        return "CORE_CLOSURE_UPDATE"
    return "NO_INTERVENTION"


def build_recomposed_policy(pred_long: pd.DataFrame, labels: pd.DataFrame, feature_set: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    factor_targets = [
        "factor_repair_need",
        "factor_repair_type",
        "factor_commit_need",
        "factor_attractor_escape_need",
        "factor_order_correction_need",
        "factor_binding_type",
    ]
    sub = pred_long[(pred_long["feature_set"] == feature_set) & (pred_long["target"].isin(factor_targets))].copy()
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()

    rec_rows = []
    group_cols = ["mode", "holdout", "graph_id", "condition"]
    for key, grp in sub.groupby(group_cols):
        mode, holdout, graph_id, condition = key
        pred_action = recompose_action_from_factors(grp)
        rec_rows.append({
            "mode": mode,
            "holdout": holdout,
            "feature_set": feature_set,
            "graph_id": graph_id,
            "condition": condition,
            "pred_recomposed_action": pred_action,
        })
    rec = pd.DataFrame(rec_rows)
    lab = labels[KEY_COLS + ["condition_family", "outcome_representative"]].copy()
    rec = rec.merge(lab, on=KEY_COLS, how="left")
    rec["correct"] = rec["pred_recomposed_action"].map(canonical_action) == rec["outcome_representative"].map(canonical_action)

    summ = []
    for (mode, fs), g in rec.groupby(["mode", "feature_set"]):
        summ.append({
            "mode": mode,
            "feature_set": fs,
            "n": int(len(g)),
            "accuracy": float(g["correct"].mean()),
            "macro_f1": float(f1_score(g["outcome_representative"], g["pred_recomposed_action"], average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(g["outcome_representative"], g["pred_recomposed_action"], average="weighted", zero_division=0)),
        })
    return rec, pd.DataFrame(summ)


def choose_verdict(agg: pd.DataFrame) -> Tuple[str, dict]:
    # Main 2H question: do factored labels generalize better than flat outcome_representative?
    info = {}
    if agg.empty:
        return "FAIL_NO_RESULTS", info

    def best(mode, target):
        s = agg[(agg["mode"] == mode) & (agg["target"] == target)]
        if s.empty:
            return None
        return s.sort_values(["mean_macro_f1", "mean_accuracy"], ascending=False).iloc[0].to_dict()

    flat_loc = best("leave_one_condition", "outcome_representative")
    flat_lof = best("leave_one_family", "outcome_representative")
    repair_loc = best("leave_one_condition", "factor_repair_need")
    repair_lof = best("leave_one_family", "factor_repair_need")
    commit_loc = best("leave_one_condition", "factor_commit_need")
    commit_lof = best("leave_one_family", "factor_commit_need")
    escape_loc = best("leave_one_condition", "factor_attractor_escape_need")
    escape_lof = best("leave_one_family", "factor_attractor_escape_need")

    info = {
        "flat_leave_one_condition": flat_loc,
        "flat_leave_one_family": flat_lof,
        "repair_leave_one_condition": repair_loc,
        "repair_leave_one_family": repair_lof,
        "commit_leave_one_condition": commit_loc,
        "commit_leave_one_family": commit_lof,
        "escape_leave_one_condition": escape_loc,
        "escape_leave_one_family": escape_lof,
    }

    flat_loc_f1 = flat_loc["mean_macro_f1"] if flat_loc else 0.0
    flat_lof_f1 = flat_lof["mean_macro_f1"] if flat_lof else 0.0
    factor_loc_best = max([x["mean_macro_f1"] for x in [repair_loc, commit_loc, escape_loc] if x is not None] or [0.0])
    factor_lof_best = max([x["mean_macro_f1"] for x in [repair_lof, commit_lof, escape_lof] if x is not None] or [0.0])

    info["best_factor_leave_one_condition_macro_f1"] = factor_loc_best
    info["best_factor_leave_one_family_macro_f1"] = factor_lof_best
    info["flat_leave_one_condition_macro_f1"] = flat_loc_f1
    info["flat_leave_one_family_macro_f1"] = flat_lof_f1

    if factor_loc_best >= 0.75 and factor_lof_best >= 0.65 and factor_loc_best > flat_loc_f1 + 0.1:
        return "PASS_FACTORED_POLICY_RULE_GENERALIZES", info
    if factor_loc_best > flat_loc_f1 + 0.1 or factor_lof_best > flat_lof_f1 + 0.1:
        return "PARTIAL_PASS_FACTORS_IMPROVE_BUT_RULE_NOT_CLOSED", info
    return "FAIL_FACTORED_RULE_NOT_BETTER_THAN_FLAT_POLICY", info


def main():
    labels, features, feature_sets = load_inputs()
    merged = labels.merge(features, on=KEY_COLS, how="left", validate="one_to_one")
    if len(merged) != len(labels):
        raise ValueError(f"Merge changed row count: labels={len(labels)} merged={len(merged)}")

    targets = [
        "outcome_representative",  # flat baseline from 2G
        "raw_policy_class",        # flat baseline
        "factor_repair_need",
        "factor_repair_type",
        "factor_commit_need",
        "factor_attractor_escape_need",
        "factor_order_correction_need",
        "factor_protect_need",
        "factor_condition_family",
        "factor_binding_type",
        "factor_rule_tuple",
    ]

    summary, pred_long = evaluate_holdouts(merged, feature_sets, targets)
    agg = aggregate_summary(summary)

    # Recomposed action from factors for the best safe strict feature set available.
    recomposed_all = []
    recomposed_summary_all = []
    for fs in ["strict_graph_dsta_core", "strict_graph_dsta_core_auto", "dsta_dense_auto_strict", "json__all_numeric_safe_no_policy_outcome"]:
        if fs in set(pred_long.get("feature_set", [])):
            rec, rs = build_recomposed_policy(pred_long, labels, fs)
            if not rec.empty:
                recomposed_all.append(rec)
                recomposed_summary_all.append(rs)
    recomposed = pd.concat(recomposed_all, ignore_index=True) if recomposed_all else pd.DataFrame()
    recomposed_summary = pd.concat(recomposed_summary_all, ignore_index=True) if recomposed_summary_all else pd.DataFrame()

    verdict, interpretation = choose_verdict(agg)

    # Save outputs.
    summary.to_csv(OUT_DIR / "da_asa2h_factor_holdout_summary.csv", index=False, encoding="utf-8-sig")
    agg.to_csv(OUT_DIR / "da_asa2h_factor_holdout_aggregate.csv", index=False, encoding="utf-8-sig")
    pred_long.to_csv(OUT_DIR / "da_asa2h_factor_predictions_long.csv", index=False, encoding="utf-8-sig")
    recomposed.to_csv(OUT_DIR / "da_asa2h_recomposed_policy_predictions.csv", index=False, encoding="utf-8-sig")
    recomposed_summary.to_csv(OUT_DIR / "da_asa2h_recomposed_policy_summary.csv", index=False, encoding="utf-8-sig")

    fs_out = {k: {"n": len(v), "columns": v} for k, v in feature_sets.items()}
    with open(OUT_DIR / "da_asa2h_feature_sets.json", "w", encoding="utf-8") as f:
        json.dump(fs_out, f, ensure_ascii=False, indent=2)

    diagnostics = {
        "stage": "DA-ASA-2H",
        "purpose": "Mechanism-factored policy rule audit after 2G unseen-condition failure",
        "verdict": verdict,
        "root": str(ROOT),
        "output_dir": str(OUT_DIR),
        "n_rows": int(len(merged)),
        "n_graphs": int(merged["graph_id"].nunique()),
        "n_conditions": int(merged["condition"].nunique()),
        "n_condition_families": int(merged["condition_family"].nunique()),
        "condition_families": sorted(merged["condition_family"].astype(str).unique().tolist()),
        "target_class_counts": {t: merged[t].astype(str).value_counts().to_dict() for t in targets if t in merged.columns},
        "feature_sets": {k: len(v) for k, v in feature_sets.items()},
        "interpretation": interpretation,
        "main_question": "Whether factor labels generalize better than flat outcome_representative under leave-one-condition/family.",
        "note": "A PASS here should be interpreted as policy-rule factor generalization, not as model-forward intervention validation.",
    }
    with open(OUT_DIR / "da_asa2h_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("DA-ASA-2H Mechanism-Factored Policy Rule Audit")
    print("=" * 80)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2)[:5000])
    print("\n[Best aggregate rows]")
    if not agg.empty:
        print(agg.head(30).to_string(index=False))
    if not recomposed_summary.empty:
        print("\n[Recomposed policy summary]")
        print(recomposed_summary.to_string(index=False))
    print(f"\n[OUT] {OUT_DIR}")


if __name__ == "__main__":
    main()
