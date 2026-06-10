# -*- coding: utf-8 -*-
"""
DA-ASA-2G: Unseen Condition / Family Generalization Audit

Purpose
-------
Audit whether DSTA-derived features that predict OutcomePolicyClass in 2F
continue to select outcome-equivalent policies when a condition or condition
family is held out.

This script is intentionally table-only. It does not run model forward passes.
It consumes DA-ASA-2F outputs when available, and falls back to nearby files.

Primary target for unseen-condition generalization:
    outcome_representative

Why not outcome_policy_class as primary?
    2E's outcome_policy_class is condition-specific, e.g. closure_update__OE01.
    In leave-one-condition-out, that exact class is absent from training by
    construction. Therefore exact outcome_policy_class is included as a stress
    audit, not as the primary generalization target.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


KEY_COLS = ["graph_id", "condition"]
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "da_asa2g_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["outcome_representative", "raw_policy_class", "outcome_policy_class"]
PRIMARY_TARGET = "outcome_representative"

LEAKAGE_PATTERNS = [
    "policy_outcome",
    "predicted",
    "prediction",
    "pred_label",
    "true_label",
    "target_policy",
    "oracle",
    "selected",
    "replay",
    "fold",
    "clean_rate",
    "pred_clean",
    "r_final",
    "mean_r_final",
    "mean_pred_clean",
]

# These are not necessarily leakage in older feature tables, but they often encode
# manual condition/rule labels. Keep out of strict feature sets.
LABELISH_PATTERNS = [
    "is_core_closure",
    "is_equal_evidence",
    "is_override",
    "is_source_claim",
    "is_hallucination",
    "is_protected_no_intervention",
    "policy_id",
]

DSTA_PATTERNS = [
    "dsta_signed_attractor",
    "signed_attractor",
    "transport_signed_align",
    "transport_rotation_abs_deg",
    "transport_delta_norm",
    "delta_align_to_update",
    "delta_align_to_override",
    "center_signed_align",
    "center_rotation_abs_deg",
    "delta_center_norm",
    "d_to_stable",
    "d_to_update",
    "d_to_override",
    "d_to_exception",
    "proj_projection_t",
    "proj_perpendicular_residual",
    "between_excess",
]

STRICT_GRAPH_PATTERNS = [
    "transport_signed_align_vs_clean",
    "transport_rotation_abs_deg",
    "transport_delta_norm",
    "delta_align_to_update",
    "delta_align_to_override",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def find_first(candidates: List[str]) -> Optional[Path]:
    roots = [ROOT, ROOT / "da_asa2f_outputs", ROOT.parent]
    for r in roots:
        for name in candidates:
            p = r / name
            if p.exists():
                return p
    # Recursive fallback, bounded by current script folder.
    for name in candidates:
        hits = list(ROOT.rglob(name))
        if hits:
            return hits[0]
    return None


def read_csv_required(candidates: List[str], desc: str) -> Tuple[pd.DataFrame, Path]:
    p = find_first(candidates)
    if p is None:
        raise FileNotFoundError(f"Could not find {desc}. Tried: {candidates}")
    log(f"[LOAD] {desc}: {p}")
    return pd.read_csv(p), p


def read_json_optional(candidates: List[str]) -> Tuple[Optional[dict], Optional[Path]]:
    p = find_first(candidates)
    if p is None:
        return None, None
    log(f"[LOAD] json: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f), p


def normalize_key_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    ren = {}
    if "graph_id" not in df.columns:
        for alt in ["graph", "gid", "graphid"]:
            if alt in lower:
                ren[lower[alt]] = "graph_id"
                break
    if "condition" not in df.columns:
        for alt in ["cond", "condition_name"]:
            if alt in lower:
                ren[lower[alt]] = "condition"
                break
    if ren:
        df = df.rename(columns=ren)
    missing = [c for c in KEY_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing key columns {missing}; columns={list(df.columns)[:30]}")
    for c in KEY_COLS:
        df[c] = df[c].astype(str).str.strip()
    return df


def canonical_family(condition: str) -> str:
    s = str(condition).strip().lower()
    if s.startswith("stable") or "weak_distractor" in s:
        return "stable"
    if s.startswith("competition"):
        return "competition"
    if s.startswith("closure"):
        return "closure"
    if "hallucination" in s:
        return "hallucination_like"
    return s.split("_")[0] if "_" in s else s


def clean_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    feature_df = normalize_key_cols(feature_df)
    # Keep one row per graph_id+condition. If duplicates exist, average numeric cols.
    numeric_cols = [c for c in feature_df.select_dtypes(include=[np.number]).columns if c not in KEY_COLS]
    if feature_df.duplicated(KEY_COLS).any():
        feature_df = feature_df.groupby(KEY_COLS, as_index=False)[numeric_cols].mean()
    return feature_df


def load_labels() -> Tuple[pd.DataFrame, Path]:
    pred, p = read_csv_required(["da_asa2f_predictions_long.csv"], "2F predictions_long")
    pred = normalize_key_cols(pred)
    needed = KEY_COLS + ["raw_policy_class", "outcome_policy_class", "outcome_representative"]
    missing = [c for c in needed if c not in pred.columns]
    if missing:
        raise KeyError(f"2F predictions_long missing {missing}; columns={list(pred.columns)}")
    keep = needed + (["condition_family"] if "condition_family" in pred.columns else [])
    lab = pred[keep].drop_duplicates(KEY_COLS).copy()
    if "condition_family" not in lab.columns:
        lab["condition_family"] = lab["condition"].map(canonical_family)
    else:
        lab["condition_family"] = lab["condition_family"].fillna(lab["condition"].map(canonical_family))
    for c in ["raw_policy_class", "outcome_policy_class", "outcome_representative", "condition_family"]:
        lab[c] = lab[c].astype(str).str.strip()
    if len(lab) != pred[KEY_COLS].drop_duplicates().shape[0]:
        raise ValueError("Label table failed to deduplicate cleanly.")
    return lab, p


def is_leaky(col: str, strict: bool = True) -> bool:
    cl = col.lower()
    if any(p in cl for p in LEAKAGE_PATTERNS):
        return True
    if strict and any(p in cl for p in LABELISH_PATTERNS):
        return True
    return False


def existing_columns(cols: Iterable[str], df: pd.DataFrame) -> List[str]:
    out = []
    seen = set()
    for c in cols:
        if c in df.columns and c not in KEY_COLS and c not in seen and pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
            seen.add(c)
    return out


def build_feature_sets(feature_df: pd.DataFrame, feature_sets_json: Optional[dict]) -> Dict[str, List[str]]:
    numeric_cols = [c for c in feature_df.select_dtypes(include=[np.number]).columns if c not in KEY_COLS]
    sets: Dict[str, List[str]] = {}

    if isinstance(feature_sets_json, dict):
        for name, obj in feature_sets_json.items():
            if isinstance(obj, dict) and isinstance(obj.get("columns"), list):
                cols = existing_columns(obj["columns"], feature_df)
                cols = [c for c in cols if not is_leaky(c, strict=True)]
                if cols:
                    sets[f"json__{name}"] = cols

    strict_graph = [
        c for c in numeric_cols
        if any(p in c.lower() for p in STRICT_GRAPH_PATTERNS)
        and not is_leaky(c, strict=True)
    ]
    if strict_graph:
        sets["strict_graph_dsta_core"] = strict_graph

    dsta_dense = [
        c for c in numeric_cols
        if any(p in c.lower() for p in DSTA_PATTERNS)
        and not is_leaky(c, strict=True)
    ]
    if dsta_dense:
        sets["dsta_dense_auto_strict"] = dsta_dense

    baseline = [
        c for c in numeric_cols
        if ("baseline" in c.lower() or "boundary_distance_proxy" in c.lower())
        and not is_leaky(c, strict=True)
    ]
    if baseline:
        sets["baseline_auto_strict"] = baseline

    all_safe = [c for c in numeric_cols if not is_leaky(c, strict=True)]
    if all_safe:
        sets["all_numeric_strict_no_labelish"] = all_safe

    # Remove huge duplicate exact sets by content signature.
    dedup: Dict[Tuple[str, ...], str] = {}
    final: Dict[str, List[str]] = {}
    for name, cols in sets.items():
        sig = tuple(cols)
        if sig in dedup:
            continue
        final[name] = cols
        dedup[sig] = name
    return final


def make_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs")),
    ])


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: List[str], target: str) -> Tuple[np.ndarray, Optional[str]]:
    y_train = train[target].astype(str).values
    if len(set(y_train)) < 2:
        # Cannot train a classifier; constant fallback.
        pred = np.array([y_train[0]] * len(test)) if len(y_train) else np.array(["NO_TRAIN_CLASS"] * len(test))
        return pred, "constant_train_single_class"
    try:
        model = make_model()
        model.fit(train[cols], y_train)
        pred = model.predict(test[cols])
        return pred, None
    except Exception as e:
        # Conservative fallback to most frequent class; record the failure.
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(train[cols], y_train)
        pred = dummy.predict(test[cols])
        return pred, f"model_failed_used_dummy:{type(e).__name__}:{e}"


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, train_classes: set) -> Dict[str, float]:
    y_true = np.asarray([str(x) for x in y_true])
    y_pred = np.asarray([str(x) for x in y_pred])
    absent_mask = np.array([x not in train_classes for x in y_true], dtype=bool)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else float("nan"),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) else float("nan"),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)) if len(y_true) else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(set(y_true)) > 1 else float("nan"),
        "absent_true_label_frac": float(absent_mask.mean()) if len(y_true) else float("nan"),
        "n_absent_true_label": int(absent_mask.sum()),
    }


def evaluate_holdouts(df: pd.DataFrame, feature_sets: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    preds = []

    for fs_name, cols in feature_sets.items():
        if not cols:
            continue
        for target in TARGETS:
            if target not in df.columns:
                continue

            # A. GroupKFold by graph_id
            groups = df["graph_id"].astype(str).values
            n_groups = len(set(groups))
            n_splits = min(5, n_groups)
            if n_splits >= 2:
                gkf = GroupKFold(n_splits=n_splits)
                all_true, all_pred = [], []
                errors = []
                for fold, (tr, te) in enumerate(gkf.split(df, df[target], groups=groups)):
                    train, test = df.iloc[tr].copy(), df.iloc[te].copy()
                    pred, err = fit_predict(train, test, cols, target)
                    if err:
                        errors.append(err)
                    train_classes = set(train[target].astype(str))
                    for i, row_idx in enumerate(test.index):
                        preds.append({
                            "mode": "group_graph",
                            "holdout": f"fold_{fold}",
                            "feature_set": fs_name,
                            "target": target,
                            "graph_id": df.loc[row_idx, "graph_id"],
                            "condition": df.loc[row_idx, "condition"],
                            "condition_family": df.loc[row_idx, "condition_family"],
                            "true_label": str(df.loc[row_idx, target]),
                            "pred_label": str(pred[i]),
                            "true_seen_in_train": str(df.loc[row_idx, target]) in train_classes,
                        })
                    all_true.extend(test[target].astype(str).values)
                    all_pred.extend(pred)
                row = metric_row(np.array(all_true), np.array(all_pred), set(df[target].astype(str)))
                row.update({
                    "mode": "group_graph",
                    "holdout": "all_folds",
                    "feature_set": fs_name,
                    "target": target,
                    "n_features": len(cols),
                    "n_train_classes_min": None,
                    "errors": " | ".join(sorted(set(errors))) if errors else "",
                })
                summaries.append(row)

            # B. Leave-one-condition-out
            for cond in sorted(df["condition"].unique()):
                train = df[df["condition"] != cond].copy()
                test = df[df["condition"] == cond].copy()
                pred, err = fit_predict(train, test, cols, target)
                train_classes = set(train[target].astype(str))
                row = metric_row(test[target].astype(str).values, pred, train_classes)
                row.update({
                    "mode": "leave_one_condition",
                    "holdout": cond,
                    "feature_set": fs_name,
                    "target": target,
                    "n_features": len(cols),
                    "n_train_classes_min": len(train_classes),
                    "errors": err or "",
                })
                summaries.append(row)
                for i, row_idx in enumerate(test.index):
                    preds.append({
                        "mode": "leave_one_condition",
                        "holdout": cond,
                        "feature_set": fs_name,
                        "target": target,
                        "graph_id": df.loc[row_idx, "graph_id"],
                        "condition": df.loc[row_idx, "condition"],
                        "condition_family": df.loc[row_idx, "condition_family"],
                        "true_label": str(df.loc[row_idx, target]),
                        "pred_label": str(pred[i]),
                        "true_seen_in_train": str(df.loc[row_idx, target]) in train_classes,
                    })

            # C. Leave-one-family-out
            for fam in sorted(df["condition_family"].unique()):
                train = df[df["condition_family"] != fam].copy()
                test = df[df["condition_family"] == fam].copy()
                pred, err = fit_predict(train, test, cols, target)
                train_classes = set(train[target].astype(str))
                row = metric_row(test[target].astype(str).values, pred, train_classes)
                row.update({
                    "mode": "leave_one_family",
                    "holdout": fam,
                    "feature_set": fs_name,
                    "target": target,
                    "n_features": len(cols),
                    "n_train_classes_min": len(train_classes),
                    "errors": err or "",
                })
                summaries.append(row)
                for i, row_idx in enumerate(test.index):
                    preds.append({
                        "mode": "leave_one_family",
                        "holdout": fam,
                        "feature_set": fs_name,
                        "target": target,
                        "graph_id": df.loc[row_idx, "graph_id"],
                        "condition": df.loc[row_idx, "condition"],
                        "condition_family": df.loc[row_idx, "condition_family"],
                        "true_label": str(df.loc[row_idx, target]),
                        "pred_label": str(pred[i]),
                        "true_seen_in_train": str(df.loc[row_idx, target]) in train_classes,
                    })

    return pd.DataFrame(summaries), pd.DataFrame(preds)


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mode, feature_set, target), sub in summary.groupby(["mode", "feature_set", "target"], dropna=False):
        rows.append({
            "mode": mode,
            "feature_set": feature_set,
            "target": target,
            "n_holdouts": int(len(sub)),
            "n_total": int(sub["n"].sum()),
            "mean_accuracy": float(sub["accuracy"].mean()),
            "mean_macro_f1": float(sub["macro_f1"].mean()),
            "mean_weighted_f1": float(sub["weighted_f1"].mean()),
            "mean_absent_true_label_frac": float(sub["absent_true_label_frac"].mean()),
            "min_accuracy": float(sub["accuracy"].min()),
            "min_macro_f1": float(sub["macro_f1"].min()),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["mode", "target", "mean_macro_f1"], ascending=[True, True, False])
    return out


def optional_replay(preds: pd.DataFrame) -> pd.DataFrame:
    # Optional table replay for predicted representative actions.
    p = find_first([
        "da_asa2d2_policy_outcomes_aggregated.csv",
        "da_asa2d2_policy_outcomes_aggregated.csv",
        "da_asa2d2_final_replay_outputs/da_asa2d2_policy_outcomes_aggregated.csv",
        "da_asa2d2_replay_outputs/da_asa2d2_policy_outcomes_aggregated.csv",
    ])
    if p is None:
        return pd.DataFrame([{"note": "policy outcome table not found; replay skipped"}])
    pol = pd.read_csv(p)
    pol = normalize_key_cols(pol)
    if "action_class" not in pol.columns:
        return pd.DataFrame([{"note": f"policy outcome table {p} has no action_class; replay skipped"}])
    metric_cols = [c for c in ["pred_clean", "R_final", "mean_pred_clean", "mean_R_final"] if c in pol.columns]
    if not metric_cols:
        metric_cols = [c for c in pol.select_dtypes(include=[np.number]).columns if c not in KEY_COLS][:2]
    pp = preds[(preds["target"] == PRIMARY_TARGET)].copy()
    if pp.empty:
        return pd.DataFrame([{"note": "no primary target predictions; replay skipped"}])
    pp = pp.rename(columns={"pred_label": "action_class"})
    merged = pp.merge(pol[KEY_COLS + ["action_class"] + metric_cols], on=KEY_COLS + ["action_class"], how="left")
    rows = []
    for (mode, feature_set), sub in merged.groupby(["mode", "feature_set"]):
        row = {
            "mode": mode,
            "feature_set": feature_set,
            "n": int(len(sub)),
            "matched_rate": float(sub[metric_cols].notna().any(axis=1).mean()) if metric_cols else float("nan"),
        }
        for c in metric_cols:
            row[f"mean_{c}"] = float(pd.to_numeric(sub[c], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def choose_verdict(agg: pd.DataFrame, summary: pd.DataFrame) -> str:
    if agg.empty:
        return "FAIL_NO_RESULTS"

    # Primary: leave-one-condition, outcome_representative, DSTA-like set.
    candidates = agg[
        (agg["mode"] == "leave_one_condition")
        & (agg["target"] == PRIMARY_TARGET)
        & (agg["feature_set"].str.contains("dsta", case=False, na=False))
    ].copy()
    if candidates.empty:
        candidates = agg[(agg["mode"] == "leave_one_condition") & (agg["target"] == PRIMARY_TARGET)].copy()
    if candidates.empty:
        return "FAIL_NO_PRIMARY_LOCO_RESULT"
    best = candidates.sort_values("mean_macro_f1", ascending=False).iloc[0]
    f1 = float(best["mean_macro_f1"])
    absent = float(best["mean_absent_true_label_frac"])

    # Family generalization is often impossible if action class absent; report separately.
    fam = agg[(agg["mode"] == "leave_one_family") & (agg["target"] == PRIMARY_TARGET)]
    fam_absent = float(fam["mean_absent_true_label_frac"].min()) if not fam.empty else 1.0

    if f1 >= 0.85 and absent == 0 and fam_absent > 0:
        return "PASS_STRONG_UNSEEN_CONDITION_BUT_FAMILY_UNSEEN_CLASS_LIMITED"
    if f1 >= 0.85 and absent == 0:
        return "PASS_STRONG_UNSEEN_CONDITION_GENERALIZATION"
    if f1 >= 0.65:
        return "PASS_LITE_UNSEEN_CONDITION_GENERALIZATION"
    return "FAIL_UNSEEN_CONDITION_GENERALIZATION"


def main() -> None:
    labels, labels_path = load_labels()
    features, features_path = read_csv_required([
        "da_asa2f_fallback_feature_table.csv",
        "da_asa2d0h5_feature_table.csv",
        "da_asa2d0h5_dense_feature_table.csv",
        "da_asa2d_feature_table.csv",
    ], "feature table")
    features = clean_features(features)

    fs_json, fs_path = read_json_optional(["da_asa2f_feature_sets.json"])
    feature_sets = build_feature_sets(features, fs_json)
    if not feature_sets:
        raise ValueError("No usable numeric feature sets were built.")

    df = labels.merge(features, on=KEY_COLS, how="left", validate="one_to_one")
    if df.empty:
        raise ValueError("Merged label/feature table is empty.")

    # Ensure feature columns exist and have at least some non-null data after merge.
    pruned_sets = {}
    for name, cols in feature_sets.items():
        cols2 = [c for c in cols if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any()]
        if cols2:
            pruned_sets[name] = cols2
    feature_sets = pruned_sets

    log(f"[INFO] rows={len(df)} graphs={df['graph_id'].nunique()} conditions={df['condition'].nunique()} families={df['condition_family'].nunique()}")
    log(f"[INFO] feature_sets={ {k: len(v) for k,v in feature_sets.items()} }")

    summary, preds = evaluate_holdouts(df, feature_sets)
    agg = aggregate_summary(summary)
    replay = optional_replay(preds)
    verdict = choose_verdict(agg, summary)

    # Save outputs
    summary.to_csv(OUT_DIR / "da_asa2g_holdout_summary.csv", index=False, encoding="utf-8-sig")
    agg.to_csv(OUT_DIR / "da_asa2g_holdout_aggregate.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(OUT_DIR / "da_asa2g_predictions_long.csv", index=False, encoding="utf-8-sig")
    replay.to_csv(OUT_DIR / "da_asa2g_optional_replay_summary.csv", index=False, encoding="utf-8-sig")

    with open(OUT_DIR / "da_asa2g_feature_sets.json", "w", encoding="utf-8") as f:
        json.dump({k: {"n": len(v), "columns": v} for k, v in feature_sets.items()}, f, ensure_ascii=False, indent=2)

    # Best rows for quick inspection
    best_rows = []
    if not agg.empty:
        for mode in sorted(agg["mode"].unique()):
            for target in TARGETS:
                sub = agg[(agg["mode"] == mode) & (agg["target"] == target)].copy()
                if not sub.empty:
                    best_rows.append(sub.sort_values("mean_macro_f1", ascending=False).iloc[0].to_dict())
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(OUT_DIR / "da_asa2g_best_by_mode_target.csv", index=False, encoding="utf-8-sig")

    diagnostics = {
        "stage": "DA-ASA-2G",
        "purpose": "Unseen condition/family generalization audit for DSTA-based OutcomePolicyClass selection",
        "verdict": verdict,
        "root": str(ROOT),
        "output_dir": str(OUT_DIR),
        "labels_file": str(labels_path),
        "feature_file": str(features_path),
        "feature_sets_file": str(fs_path) if fs_path else None,
        "n_rows": int(len(df)),
        "n_graphs": int(df["graph_id"].nunique()),
        "n_conditions": int(df["condition"].nunique()),
        "n_condition_families": int(df["condition_family"].nunique()),
        "condition_families": sorted(df["condition_family"].unique().tolist()),
        "target_class_counts": {t: df[t].value_counts().to_dict() for t in TARGETS if t in df.columns},
        "feature_sets": {k: len(v) for k, v in feature_sets.items()},
        "best_by_mode_target": best_rows,
        "primary_interpretation": {
            "primary_target": PRIMARY_TARGET,
            "why": "outcome_policy_class is condition-specific and is expected to fail exact unseen-condition prediction; outcome_representative tests transferable action-level policy selection.",
        },
    }
    with open(OUT_DIR / "da_asa2g_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("DA-ASA-2G UNSEEN CONDITION / FAMILY GENERALIZATION AUDIT")
    print("=" * 80)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2)[:6000])
    print("\n[BEST]")
    if not best_df.empty:
        print(best_df[["mode", "target", "feature_set", "mean_accuracy", "mean_macro_f1", "mean_absent_true_label_frac"]].to_string(index=False))
    print(f"\n[OUT] {OUT_DIR}")


if __name__ == "__main__":
    main()
