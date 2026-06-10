# -*- coding: utf-8 -*-
"""
DA-ASA-2I v2: AttractorEscape-First Policy Rule Audit

Purpose
-------
After DA-ASA-2G failed flat unseen-condition generalization and DA-ASA-2H showed
factor labels generalize better than flat action labels, this audit tests a
hierarchical policy rule:

    DSTA -> AttractorEscapeNeed -> subspace factors -> recomposed policy

Main hypothesis
---------------
AttractorEscapeNeed is the most transferable first gate. Policy selection should
not directly predict outcome_representative; it should first split samples into
escape / non-escape subspaces, then infer repair/commit/order-correction only
inside the relevant subspace.

Inputs
------
This script expects to run from the same parent directory as previous outputs.
It will look for:
  da_asa2h_outputs/da_asa2h_factor_predictions_long.csv
  da_asa2h_outputs/da_asa2h_feature_sets.json
  da_asa2f_outputs/da_asa2f_fallback_feature_table.csv
  da_asa2f_outputs/da_asa2f_predictions_long.csv

If labels/features are unavailable, it will stop with a clear error.

Outputs
-------
  da_asa2i_outputs/da_asa2i_diagnostics.json
  da_asa2i_outputs/da_asa2i_gate_summary.csv
  da_asa2i_outputs/da_asa2i_recomposed_policy_summary.csv
  da_asa2i_outputs/da_asa2i_predictions_long.csv
  da_asa2i_outputs/da_asa2i_confusion_recomposed_policy.csv

Notes
-----
This is still table-level policy-rule audit, not model-forward intervention.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "da_asa2i_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KEY_COLS = ["graph_id", "condition"]
RANDOM_SEED = 20260606

FEATURE_TABLE_CANDIDATES = [
    ROOT / "da_asa2f_outputs" / "da_asa2f_fallback_feature_table.csv",
    ROOT / "da_asa2f_fallback_feature_table.csv",
]
FEATURE_SETS_CANDIDATES = [
    ROOT / "da_asa2h_outputs" / "da_asa2h_feature_sets.json",
    ROOT / "da_asa2f_outputs" / "da_asa2f_feature_sets.json",
]
LABEL_CANDIDATES = [
    ROOT / "da_asa2f_outputs" / "da_asa2f_predictions_long.csv",
    ROOT / "da_asa2g_outputs" / "da_asa2g_predictions_long.csv",
    ROOT / "da_asa2h_outputs" / "da_asa2h_factor_predictions_long.csv",
]

# Preferred feature sets from 2H.
PREFERRED_FEATURE_SETS = [
    "strict_graph_dsta_core",
    "strict_graph_dsta_core_auto",
    "dsta_dense_auto",
    "dsta_dense_auto_strict",
    "json__dsta_dense_no_policy_outcome",
    "json__all_numeric_safe_no_policy_outcome",
    "baseline_auto_strict",
]

MODE_ORDER = ["group_graph", "leave_one_condition", "leave_one_family"]


def find_existing(paths: List[Path], name: str) -> Path:
    for p in paths:
        if p.exists():
            print(f"[LOAD] {name}: {p}")
            return p
    raise FileNotFoundError(f"Could not find {name}. Tried: {[str(p) for p in paths]}")


def canonical_action(x) -> str:
    if pd.isna(x):
        return "NO_INTERVENTION"
    s = str(x).strip().upper()
    aliases = {
        "NOOP": "NO_INTERVENTION",
        "NO_OP": "NO_INTERVENTION",
        "NONE": "NO_INTERVENTION",
        "NO INTERVENTION": "NO_INTERVENTION",
        "A0.0_B0.0": "NO_INTERVENTION",
        "CORE": "CORE_CLOSURE_UPDATE",
        "CORE_CLOSURE": "CORE_CLOSURE_UPDATE",
        "CORE_UPDATE": "CORE_CLOSURE_UPDATE",
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
    if "CORE" in s:
        return "CORE_CLOSURE_UPDATE"
    if "EQUAL" in s:
        return "EQUAL_EVIDENCE_ORDER"
    if "OVERRIDE" in s:
        return "OVERRIDE_THREE_STAGE"
    return s


def infer_condition_family(cond: str) -> str:
    c = str(cond).lower()
    if c.startswith("closure_"):
        return "closure"
    if c.startswith("competition_"):
        return "competition"
    if c.startswith("stable_"):
        return "stable"
    if "hallucination" in c:
        return "hallucination_like"
    return "other"


def build_labels_from_prediction_file(df: pd.DataFrame) -> pd.DataFrame:
    """Extract graph-level target labels from 2F/2G/2H long prediction files or summaries."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in KEY_COLS:
        if col not in df.columns:
            raise KeyError(f"label file has no {col}; columns={list(df.columns)[:30]}")
    df["graph_id"] = df["graph_id"].astype(str).str.strip()
    df["condition"] = df["condition"].astype(str).str.strip()

    # Keep only rows with true labels when long CV file has target/y_true.
    labels = df[KEY_COLS].drop_duplicates().copy()

    def pick_truth_col(candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    # Direct columns from 2F/2G predictions_long.
    outcome_col = pick_truth_col(["true_outcome_representative", "outcome_representative", "y_true_outcome_representative"])
    raw_col = pick_truth_col(["true_raw_policy_class", "raw_policy_class", "y_true_raw_policy_class"])

    # Generic long format: target, y_true.
    if (outcome_col is None or raw_col is None) and {"target", "y_true"}.issubset(df.columns):
        long = df.copy()
        for target_name, out_name in [
            ("outcome_representative", "outcome_representative"),
            ("raw_policy_class", "raw_policy_class"),
        ]:
            sub = long[long["target"].astype(str) == target_name]
            if len(sub):
                val = sub.groupby(KEY_COLS)["y_true"].first().reset_index().rename(columns={"y_true": out_name})
                labels = labels.merge(val, on=KEY_COLS, how="left")

    if outcome_col is not None and "outcome_representative" not in labels.columns:
        val = df.groupby(KEY_COLS)[outcome_col].first().reset_index().rename(columns={outcome_col: "outcome_representative"})
        labels = labels.merge(val, on=KEY_COLS, how="left")
    if raw_col is not None and "raw_policy_class" not in labels.columns:
        val = df.groupby(KEY_COLS)[raw_col].first().reset_index().rename(columns={raw_col: "raw_policy_class"})
        labels = labels.merge(val, on=KEY_COLS, how="left")

    # If 2H factor predictions long has factor targets, capture them too.
    if {"target", "y_true"}.issubset(df.columns):
        for target_name in [
            "factor_repair_need", "factor_repair_type", "factor_commit_need",
            "factor_attractor_escape_need", "factor_order_correction_need",
            "factor_protect_need", "factor_binding_type", "factor_rule_tuple",
        ]:
            sub = df[df["target"].astype(str) == target_name]
            if len(sub):
                val = sub.groupby(KEY_COLS)["y_true"].first().reset_index().rename(columns={"y_true": target_name})
                labels = labels.merge(val, on=KEY_COLS, how="left")

    if "outcome_representative" not in labels.columns and "raw_policy_class" in labels.columns:
        labels["outcome_representative"] = labels["raw_policy_class"]
    if "raw_policy_class" not in labels.columns and "outcome_representative" in labels.columns:
        labels["raw_policy_class"] = labels["outcome_representative"]

    if "outcome_representative" not in labels.columns:
        raise KeyError("Could not infer outcome_representative labels from available prediction files.")

    labels["outcome_representative"] = labels["outcome_representative"].map(canonical_action)
    labels["raw_policy_class"] = labels.get("raw_policy_class", labels["outcome_representative"]).map(canonical_action)
    labels["condition_family"] = labels["condition"].map(infer_condition_family)

    # Derive factors if missing.
    labels = derive_policy_factors(labels)
    return labels.drop_duplicates(KEY_COLS).reset_index(drop=True)


def derive_policy_factors(labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.copy()
    act = labels["outcome_representative"].map(canonical_action)

    if "factor_repair_need" not in labels.columns:
        labels["factor_repair_need"] = np.where(act.eq("NO_INTERVENTION"), "NO_REPAIR", "REPAIR")
    if "factor_commit_need" not in labels.columns:
        labels["factor_commit_need"] = np.where(act.isin(["OVERRIDE_THREE_STAGE", "EQUAL_EVIDENCE_ORDER"]), "COMMIT_CORRECTION", "NO_COMMIT_CORRECTION")
    if "factor_attractor_escape_need" not in labels.columns:
        labels["factor_attractor_escape_need"] = np.where(act.isin(["CORE_CLOSURE_UPDATE", "OVERRIDE_THREE_STAGE"]), "ATTRACTOR_ESCAPE", "NO_ATTRACTOR_ESCAPE")
    if "factor_order_correction_need" not in labels.columns:
        labels["factor_order_correction_need"] = np.where(act.isin(["OVERRIDE_THREE_STAGE", "EQUAL_EVIDENCE_ORDER"]), "ORDER_CORRECTION", "NO_ORDER_CORRECTION")
    if "factor_protect_need" not in labels.columns:
        labels["factor_protect_need"] = np.where(act.eq("NO_INTERVENTION"), "PROTECT", "INTERVENE")
    if "factor_repair_type" not in labels.columns:
        labels["factor_repair_type"] = act.map({
            "NO_INTERVENTION": "NONE",
            "CORE_CLOSURE_UPDATE": "CORE_CLOSURE_REPAIR",
            "EQUAL_EVIDENCE_ORDER": "EQUAL_EVIDENCE_REPAIR",
            "OVERRIDE_THREE_STAGE": "OVERRIDE_REPAIR",
        }).fillna("NONE")
    if "factor_binding_type" not in labels.columns:
        labels["factor_binding_type"] = labels["condition"].astype(str)
    if "factor_rule_tuple" not in labels.columns:
        labels["factor_rule_tuple"] = (
            labels["factor_repair_need"].astype(str) + "|" +
            labels["factor_attractor_escape_need"].astype(str) + "|" +
            labels["factor_commit_need"].astype(str)
        )
    return labels


def load_labels() -> pd.DataFrame:
    for p in LABEL_CANDIDATES:
        if p.exists():
            try:
                df = pd.read_csv(p)
                labels = build_labels_from_prediction_file(df)
                print(f"[LOAD] labels: {p} rows={len(labels)}")
                return labels
            except Exception as e:
                print(f"[WARN] failed to use labels from {p}: {e}")
    raise FileNotFoundError("Could not build labels from prediction files.")


def load_features(labels: pd.DataFrame) -> pd.DataFrame:
    p = find_existing(FEATURE_TABLE_CANDIDATES, "feature_table")
    feats = pd.read_csv(p)
    feats.columns = [str(c).strip() for c in feats.columns]
    for col in KEY_COLS:
        if col not in feats.columns:
            raise KeyError(f"feature table missing {col}")
        feats[col] = feats[col].astype(str).str.strip()
    feats = labels[KEY_COLS].drop_duplicates().merge(feats, on=KEY_COLS, how="left")
    return feats


def load_feature_sets(features: pd.DataFrame) -> Dict[str, List[str]]:
    fs = {}
    for p in FEATURE_SETS_CANDIDATES:
        if p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                for k, v in raw.items():
                    cols = v.get("columns", v) if isinstance(v, dict) else v
                    if isinstance(cols, list):
                        cols2 = [c for c in cols if c in features.columns]
                        if len(cols2) >= 2:
                            fs[str(k)] = cols2
                print(f"[LOAD] feature sets: {p}")
                break
            except Exception as e:
                print(f"[WARN] cannot parse feature sets {p}: {e}")

    numeric_cols = [c for c in features.select_dtypes(include=[np.number]).columns if c not in KEY_COLS]
    if not fs:
        fs["all_numeric"] = numeric_cols

    # Auto sets robustly derived from column names.
    dsta_cols = [c for c in numeric_cols if "dsta" in c.lower() or "attractor" in c.lower()]
    strict_dsta = [c for c in dsta_cols if any(t in c.lower() for t in ["transport", "delta_align", "rotation", "delta_norm", "signed_align"])]
    baseline_cols = [c for c in numeric_cols if "baseline" in c.lower() and "policy_outcome" not in c.lower()]
    safe_all = [c for c in numeric_cols if all(bad not in c.lower() for bad in ["policy_outcome", "pred_clean", "r_final", "outcome_"])]
    if len(dsta_cols) >= 2:
        fs["auto_dsta_dense"] = dsta_cols[:250]
    if len(strict_dsta) >= 2:
        fs["auto_strict_dsta"] = strict_dsta[:150]
    if len(baseline_cols) >= 2:
        fs["auto_baseline"] = baseline_cols[:100]
    if len(safe_all) >= 2:
        fs["auto_all_safe"] = safe_all[:300]

    ordered = {}
    for name in PREFERRED_FEATURE_SETS:
        if name in fs:
            ordered[name] = fs[name]
    for k, v in fs.items():
        if k not in ordered:
            ordered[k] = v
    return ordered


def make_model(n_classes: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear" if n_classes <= 2 else "lbfgs",
            random_state=RANDOM_SEED,
        )),
    ])


def split_holdouts(labels: pd.DataFrame, mode: str) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    if mode == "group_graph":
        # One graph-heldout full CV via each graph.
        groups = labels["graph_id"].astype(str).unique().tolist()
        out = []
        for g in groups:
            test = labels["graph_id"].astype(str).values == str(g)
            train = ~test
            out.append((str(g), train, test))
        return out
    if mode == "leave_one_condition":
        vals = labels["condition"].astype(str).unique().tolist()
        return [(v, labels["condition"].astype(str).values != v, labels["condition"].astype(str).values == v) for v in vals]
    if mode == "leave_one_family":
        vals = labels["condition_family"].astype(str).unique().tolist()
        return [(v, labels["condition_family"].astype(str).values != v, labels["condition_family"].astype(str).values == v) for v in vals]
    raise ValueError(mode)




def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with uniquely valued column names for safe concat/merge."""
    df = df.copy()
    seen = {}
    new_cols = []
    for c in df.columns:
        c = str(c)
        if c not in seen:
            seen[c] = 0
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}__dup{seen[c]}")
    df.columns = new_cols
    return df


def unique_list(xs: List[str]) -> List[str]:
    out = []
    seen = set()
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def evaluate_target(
    df: pd.DataFrame,
    features: pd.DataFrame,
    feature_cols: List[str],
    target: str,
    mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    preds = []
    X_all = features[feature_cols].copy()
    y_all = df[target].astype(str).values
    holdouts = split_holdouts(df, mode)
    for holdout, train_mask, test_mask in holdouts:
        y_train = y_all[train_mask]
        y_test = y_all[test_mask]
        absent = np.mean([yy not in set(y_train) for yy in y_test]) if len(y_test) else np.nan
        if len(set(y_train)) < 2:
            y_pred = np.array([pd.Series(y_train).mode().iloc[0] if len(y_train) else "MISSING"] * len(y_test))
        else:
            model = make_model(len(set(y_train)))
            model.fit(X_all.loc[train_mask], y_train)
            y_pred = model.predict(X_all.loc[test_mask])
        acc = accuracy_score(y_test, y_pred) if len(y_test) else np.nan
        macro = f1_score(y_test, y_pred, average="macro", zero_division=0) if len(y_test) else np.nan
        weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0) if len(y_test) else np.nan
        try:
            bal = balanced_accuracy_score(y_test, y_pred)
        except Exception:
            bal = np.nan
        rows.append({
            "mode": mode,
            "holdout": holdout,
            "target": target,
            "n_test": int(test_mask.sum()),
            "n_train": int(train_mask.sum()),
            "n_train_classes": int(len(set(y_train))),
            "n_test_classes": int(len(set(y_test))),
            "absent_true_label_frac": float(absent),
            "accuracy": float(acc),
            "macro_f1": float(macro),
            "weighted_f1": float(weighted),
            "balanced_accuracy": float(bal) if not pd.isna(bal) else np.nan,
        })
        pred_cols = unique_list(KEY_COLS + ["condition_family", target, "outcome_representative"])
        test_rows = df.loc[test_mask, pred_cols].copy()
        if target in test_rows.columns:
            test_rows = test_rows.rename(columns={target: "y_true"})
        else:
            test_rows["y_true"] = y_test
        # Preserve outcome_representative when target is not outcome_representative;
        # when target is outcome_representative, y_true already contains it.
        test_rows["y_pred"] = y_pred
        test_rows["mode"] = mode
        test_rows["holdout"] = holdout
        test_rows["target"] = target
        preds.append(dedupe_columns(test_rows))
    return pd.DataFrame(rows), pd.concat([dedupe_columns(x) for x in preds], ignore_index=True) if preds else pd.DataFrame()


def select_best_feature_set(df: pd.DataFrame, features: pd.DataFrame, feature_sets: Dict[str, List[str]], target: str, mode: str) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    best = None
    all_summaries = []
    all_preds = []
    for name, cols in feature_sets.items():
        if len(cols) < 2:
            continue
        summ, pred = evaluate_target(df, features, cols, target, mode)
        agg_macro = summ["macro_f1"].mean()
        agg_acc = summ["accuracy"].mean()
        all_summaries.append(summ.assign(feature_set=name))
        all_preds.append(pred.assign(feature_set=name))
        score = (agg_macro, agg_acc, -len(cols))
        if best is None or score > best[0]:
            best = (score, name, summ, pred)
    if best is None:
        raise RuntimeError(f"No feature set could evaluate {target}/{mode}")
    return best[1], best[2].assign(feature_set=best[1]), best[3].assign(feature_set=best[1])


def action_from_factors(escape: str, repair: str, commit: str, order: Optional[str] = None) -> str:
    escape_yes = str(escape) == "ATTRACTOR_ESCAPE"
    repair_yes = str(repair) == "REPAIR"
    commit_yes = str(commit) == "COMMIT_CORRECTION"
    order_yes = str(order) == "ORDER_CORRECTION" if order is not None else commit_yes
    if not repair_yes:
        return "NO_INTERVENTION"
    if escape_yes and commit_yes:
        return "OVERRIDE_THREE_STAGE"
    if escape_yes and not commit_yes:
        return "CORE_CLOSURE_UPDATE"
    if (not escape_yes) and (commit_yes or order_yes):
        return "EQUAL_EVIDENCE_ORDER"
    return "NO_INTERVENTION"


def hierarchical_predict(
    labels: pd.DataFrame,
    features: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train hierarchical factors within each split and recompose policy."""
    rows = []
    pred_rows = []
    holdouts = split_holdouts(labels, mode)

    # Fixed feature sets chosen from 2H-ish best defaults.
    fs_escape = next((n for n in ["dsta_dense_auto", "auto_dsta_dense", "strict_graph_dsta_core", "auto_strict_dsta", "json__baseline_only"] if n in feature_sets), list(feature_sets)[0])
    fs_repair = next((n for n in ["strict_graph_dsta_core", "auto_strict_dsta", "dsta_dense_auto", "auto_dsta_dense"] if n in feature_sets), list(feature_sets)[0])
    fs_commit = next((n for n in ["dsta_dense_auto", "auto_dsta_dense", "strict_graph_dsta_core", "auto_strict_dsta"] if n in feature_sets), list(feature_sets)[0])

    for holdout, train_mask, test_mask in holdouts:
        train = labels.loc[train_mask].copy()
        test = labels.loc[test_mask].copy()
        pred_factor = {}
        for target, fs_name in [
            ("factor_attractor_escape_need", fs_escape),
            ("factor_repair_need", fs_repair),
            ("factor_commit_need", fs_commit),
            ("factor_order_correction_need", fs_commit),
        ]:
            y_train = train[target].astype(str).values
            if len(set(y_train)) < 2:
                yp = np.array([pd.Series(y_train).mode().iloc[0]] * len(test))
            else:
                model = make_model(len(set(y_train)))
                cols = feature_sets[fs_name]
                model.fit(features.loc[train_mask, cols], y_train)
                yp = model.predict(features.loc[test_mask, cols])
            pred_factor[target] = yp

        y_pred_action = [
            action_from_factors(e, r, c, o)
            for e, r, c, o in zip(
                pred_factor["factor_attractor_escape_need"],
                pred_factor["factor_repair_need"],
                pred_factor["factor_commit_need"],
                pred_factor["factor_order_correction_need"],
            )
        ]
        y_true = test["outcome_representative"].astype(str).values
        acc = accuracy_score(y_true, y_pred_action)
        macro = f1_score(y_true, y_pred_action, average="macro", zero_division=0)
        weighted = f1_score(y_true, y_pred_action, average="weighted", zero_division=0)
        rows.append({
            "mode": mode,
            "holdout": holdout,
            "n_test": int(test_mask.sum()),
            "accuracy": float(acc),
            "macro_f1": float(macro),
            "weighted_f1": float(weighted),
            "feature_escape": fs_escape,
            "feature_repair": fs_repair,
            "feature_commit": fs_commit,
        })
        pr = test[KEY_COLS + ["condition_family", "outcome_representative"]].copy()
        pr["pred_policy"] = y_pred_action
        for k, v in pred_factor.items():
            pr["pred_" + k] = v
        pr["mode"] = mode
        pr["holdout"] = holdout
        pred_rows.append(pr)

    return pd.DataFrame(rows), pd.concat([dedupe_columns(x) for x in pred_rows], ignore_index=True)


def aggregate_summary(summaries: List[pd.DataFrame]) -> pd.DataFrame:
    if not summaries:
        return pd.DataFrame()
    all_s = pd.concat([dedupe_columns(x) for x in summaries], ignore_index=True)
    return (all_s.groupby(["mode", "target", "feature_set"], as_index=False)
            .agg(n_holdouts=("holdout", "nunique"), n_total=("n_test", "sum"),
                 mean_accuracy=("accuracy", "mean"), mean_macro_f1=("macro_f1", "mean"),
                 mean_weighted_f1=("weighted_f1", "mean"), min_accuracy=("accuracy", "min"),
                 min_macro_f1=("macro_f1", "min"), mean_absent_true_label_frac=("absent_true_label_frac", "mean")))


def main() -> None:
    labels = load_labels()
    features = load_features(labels)
    feature_sets = load_feature_sets(features)
    # Align order.
    labels = labels.sort_values(KEY_COLS).reset_index(drop=True)
    features = labels[KEY_COLS].merge(features, on=KEY_COLS, how="left")

    factor_targets = [
        "factor_attractor_escape_need",
        "factor_repair_need",
        "factor_commit_need",
        "factor_order_correction_need",
        "factor_repair_type",
        "factor_rule_tuple",
    ]

    factor_summaries = []
    factor_preds = []
    best_factor_records = []
    for mode in MODE_ORDER:
        for target in factor_targets + ["outcome_representative"]:
            if target not in labels.columns:
                continue
            fs_name, summ, pred = select_best_feature_set(labels, features, feature_sets, target, mode)
            factor_summaries.append(summ.assign(target=target))
            factor_preds.append(pred)
            best_factor_records.append({
                "mode": mode,
                "target": target,
                "feature_set": fs_name,
                "mean_accuracy": float(summ["accuracy"].mean()),
                "mean_macro_f1": float(summ["macro_f1"].mean()),
                "mean_weighted_f1": float(summ["weighted_f1"].mean()),
                "min_accuracy": float(summ["accuracy"].min()),
                "min_macro_f1": float(summ["macro_f1"].min()),
            })

    h_summaries = []
    h_preds = []
    for mode in MODE_ORDER:
        hs, hp = hierarchical_predict(labels, features, feature_sets, mode)
        h_summaries.append(hs)
        h_preds.append(hp)

    factor_summary = pd.concat([dedupe_columns(x) for x in factor_summaries], ignore_index=True)
    factor_pred = pd.concat([dedupe_columns(x) for x in factor_preds], ignore_index=True)
    h_summary = pd.concat([dedupe_columns(x) for x in h_summaries], ignore_index=True)
    h_pred = pd.concat([dedupe_columns(x) for x in h_preds], ignore_index=True)
    best_df = pd.DataFrame(best_factor_records)

    factor_agg = aggregate_summary(factor_summaries)
    h_agg = (h_summary.groupby("mode", as_index=False)
             .agg(n_holdouts=("holdout", "nunique"), n_total=("n_test", "sum"),
                  mean_accuracy=("accuracy", "mean"), mean_macro_f1=("macro_f1", "mean"),
                  mean_weighted_f1=("weighted_f1", "mean"), min_accuracy=("accuracy", "min"), min_macro_f1=("macro_f1", "min")))

    # Compare 2H/2G flat vs 2I hierarchical.
    def get_best(mode, target):
        sub = best_df[(best_df["mode"] == mode) & (best_df["target"] == target)]
        if sub.empty:
            return None
        return sub.sort_values(["mean_macro_f1", "mean_accuracy"], ascending=False).iloc[0].to_dict()

    flat_loc = get_best("leave_one_condition", "outcome_representative")
    flat_lof = get_best("leave_one_family", "outcome_representative")
    esc_loc = get_best("leave_one_condition", "factor_attractor_escape_need")
    esc_lof = get_best("leave_one_family", "factor_attractor_escape_need")

    h_loc = h_agg[h_agg["mode"] == "leave_one_condition"].iloc[0].to_dict() if len(h_agg[h_agg["mode"] == "leave_one_condition"]) else {}
    h_lof = h_agg[h_agg["mode"] == "leave_one_family"].iloc[0].to_dict() if len(h_agg[h_agg["mode"] == "leave_one_family"]) else {}

    # Verdict logic.
    h_loc_f1 = float(h_loc.get("mean_macro_f1", 0.0))
    h_lof_f1 = float(h_lof.get("mean_macro_f1", 0.0))
    flat_loc_f1 = float(flat_loc.get("mean_macro_f1", 0.0)) if flat_loc else 0.0
    flat_lof_f1 = float(flat_lof.get("mean_macro_f1", 0.0)) if flat_lof else 0.0
    esc_lof_f1 = float(esc_lof.get("mean_macro_f1", 0.0)) if esc_lof else 0.0

    if h_loc_f1 > flat_loc_f1 + 0.10 and h_lof_f1 > flat_lof_f1 + 0.10 and h_lof_f1 >= 0.55:
        verdict = "PASS_HIERARCHICAL_POLICY_RULE_IMPROVES_UNSEEN_GENERALIZATION"
    elif esc_lof_f1 >= 0.60 and (h_loc_f1 > flat_loc_f1 or h_lof_f1 > flat_lof_f1):
        verdict = "PARTIAL_PASS_ESCAPE_GATE_STRONG_RECOMPOSITION_WEAK"
    else:
        verdict = "FAIL_HIERARCHICAL_POLICY_RULE_NOT_IMPROVED"

    diagnostics = {
        "stage": "DA-ASA-2I",
        "purpose": "AttractorEscape-first hierarchical policy rule after 2H partial pass",
        "verdict": verdict,
        "root": str(ROOT),
        "output_dir": str(OUT_DIR),
        "n_rows": int(len(labels)),
        "n_graphs": int(labels["graph_id"].nunique()),
        "n_conditions": int(labels["condition"].nunique()),
        "n_condition_families": int(labels["condition_family"].nunique()),
        "condition_families": sorted(labels["condition_family"].unique().tolist()),
        "target_class_counts": {c: labels[c].value_counts().to_dict() for c in ["outcome_representative"] + factor_targets if c in labels.columns},
        "feature_sets": {k: len(v) for k, v in feature_sets.items()},
        "flat_leave_one_condition": flat_loc,
        "flat_leave_one_family": flat_lof,
        "escape_leave_one_condition": esc_loc,
        "escape_leave_one_family": esc_lof,
        "hierarchical_leave_one_condition": h_loc,
        "hierarchical_leave_one_family": h_lof,
        "interpretation": {
            "main_test": "Does AttractorEscape-first hierarchical recomposition beat flat outcome_representative on leave-one-condition/family?",
            "caveat": "This is table-level rule recomposition audit, not model-forward intervention validation.",
        }
    }

    # Save.
    (OUT_DIR / "da_asa2i_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    factor_summary.to_csv(OUT_DIR / "da_asa2i_factor_holdout_summary.csv", index=False, encoding="utf-8-sig")
    factor_agg.to_csv(OUT_DIR / "da_asa2i_factor_holdout_aggregate.csv", index=False, encoding="utf-8-sig")
    factor_pred.to_csv(OUT_DIR / "da_asa2i_factor_predictions_long.csv", index=False, encoding="utf-8-sig")
    best_df.to_csv(OUT_DIR / "da_asa2i_best_factor_by_mode_target.csv", index=False, encoding="utf-8-sig")
    h_summary.to_csv(OUT_DIR / "da_asa2i_hierarchical_policy_summary.csv", index=False, encoding="utf-8-sig")
    h_agg.to_csv(OUT_DIR / "da_asa2i_hierarchical_policy_aggregate.csv", index=False, encoding="utf-8-sig")
    h_pred.to_csv(OUT_DIR / "da_asa2i_hierarchical_policy_predictions.csv", index=False, encoding="utf-8-sig")
    with open(OUT_DIR / "da_asa2i_feature_sets.json", "w", encoding="utf-8") as f:
        json.dump({k: {"n": len(v), "columns": v} for k, v in feature_sets.items()}, f, ensure_ascii=False, indent=2)

    # Confusion for hierarchical leave_one_condition/family.
    for mode in ["leave_one_condition", "leave_one_family"]:
        sub = h_pred[h_pred["mode"] == mode]
        if len(sub):
            labs = sorted(set(sub["outcome_representative"].astype(str)) | set(sub["pred_policy"].astype(str)))
            cm = pd.DataFrame(confusion_matrix(sub["outcome_representative"], sub["pred_policy"], labels=labs), index=labs, columns=labs)
            cm.to_csv(OUT_DIR / f"da_asa2i_confusion_hierarchical_{mode}.csv", encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("DA-ASA-2I ATTRACTOR-ESCAPE-FIRST POLICY RULE AUDIT")
    print("=" * 80)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print(f"\n[OUT] {OUT_DIR}")


if __name__ == "__main__":
    main()
