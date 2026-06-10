# -*- coding: utf-8 -*-
"""
SR-1C.1: Cross-Reuse-Group Failure Diagnostics

Purpose
-------
SR-1C showed:

    TopK/VIM features do not beat tabular baseline under group_by_reuse_group.

This diagnostic script answers:
    1. Which reuse groups fail?
    2. Is the failure caused by class distribution shift?
    3. Are TopK models collapsing to majority classes?
    4. Which feature/model combinations are robust within concept but fail across reuse_group?
    5. Does predicted action utility degrade specifically under held-out reuse groups?

Inputs
------
    sr1c_outputs/sr1c_cv_results.csv
    sr1c_outputs/sr1c_predictions.csv
    sr1c_outputs/sr1c_policy_utility.csv
    sr1c_outputs/sr1c_feature_importance.csv
    sr1b1_outputs/sr1b1_topk_vim_features.csv

Outputs
-------
    sr1c1_outputs/sr1c1_reuse_group_confusion.csv
    sr1c1_outputs/sr1c1_group_target_distribution.csv
    sr1c1_outputs/sr1c1_model_collapse_audit.csv
    sr1c1_outputs/sr1c1_concept_vs_reuse_gap.csv
    sr1c1_outputs/sr1c1_top_features_by_failure.csv
    sr1c1_outputs/sr1c1_summary.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score


BASE_DIR = Path(r"C:\Users\ZH\Desktop\AGI\python_script")

IN_CV = BASE_DIR / "sr1c_outputs" / "sr1c_cv_results.csv"
IN_PREDS = BASE_DIR / "sr1c_outputs" / "sr1c_predictions.csv"
IN_UTILITY = BASE_DIR / "sr1c_outputs" / "sr1c_policy_utility.csv"
IN_IMPORTANCE = BASE_DIR / "sr1c_outputs" / "sr1c_feature_importance.csv"
IN_FEATURES = BASE_DIR / "sr1b1_outputs" / "sr1b1_topk_vim_features.csv"

OUT_DIR = BASE_DIR / "sr1c1_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CONFUSION = OUT_DIR / "sr1c1_reuse_group_confusion.csv"
OUT_DIST = OUT_DIR / "sr1c1_group_target_distribution.csv"
OUT_COLLAPSE = OUT_DIR / "sr1c1_model_collapse_audit.csv"
OUT_GAP = OUT_DIR / "sr1c1_concept_vs_reuse_gap.csv"
OUT_TOP_FEATURES = OUT_DIR / "sr1c1_top_features_by_failure.csv"
OUT_SUMMARY = OUT_DIR / "sr1c1_summary.json"


TABULAR_SETS = {
    "boundary_only",
    "reuse_group_only",
    "boundary_plus_group",
    "symbolic_no_concept",
    "symbolic_with_concept",
    "surface_text_only",
    "all_tabular_text",
}


def read_csv(path: Path, required=True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"[LOAD] {path} shape={df.shape}")
    return df


def kind_of_feature_set(fs: str) -> str:
    if fs in TABULAR_SETS:
        return "tabular"
    if str(fs).startswith("topk_"):
        return "topk"
    return "fused"


def safe_macro_f1(y_true, y_pred):
    try:
        return float(f1_score(y_true, y_pred, average="macro"))
    except Exception:
        return np.nan


def class_distribution(series: pd.Series) -> dict:
    vc = series.astype(str).value_counts(dropna=False)
    total = max(1, int(vc.sum()))
    return {str(k): float(v / total) for k, v in vc.items()}


def main():
    cv = read_csv(IN_CV)
    preds = read_csv(IN_PREDS)
    utility = read_csv(IN_UTILITY, required=False)
    importance = read_csv(IN_IMPORTANCE, required=False)
    features = read_csv(IN_FEATURES)

    required_pred = ["task_id", "target", "feature_set", "model", "split", "y_true", "y_pred"]
    missing = [c for c in required_pred if c not in preds.columns]
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")

    required_feat = ["task_id", "reuse_group", "task_family", "target_operator", "concept"]
    missing_feat = [c for c in required_feat if c not in features.columns]
    if missing_feat:
        raise ValueError(f"Missing feature/meta columns: {missing_feat}")

    meta = features[required_feat].copy()
    meta["task_id"] = meta["task_id"].astype(str)

    preds["task_id"] = preds["task_id"].astype(str)
    p = preds.merge(meta, on="task_id", how="left")

    p["feature_kind"] = p["feature_set"].apply(kind_of_feature_set)
    p["correct"] = (p["y_true"].astype(str) == p["y_pred"].astype(str)).astype(int)

    # =========================
    # 1. Reuse-group confusion
    # =========================

    rg = p[p["split"] == "group_by_reuse_group"].copy()

    confusion_rows = []
    for keys, g in rg.groupby(["target", "feature_set", "model", "feature_kind", "reuse_group"], dropna=False):
        target, fs, model, kind, group = keys
        y_true = g["y_true"].astype(str)
        y_pred = g["y_pred"].astype(str)
        labels = sorted(set(y_true) | set(y_pred))
        macro = safe_macro_f1(y_true, y_pred)
        acc = float(accuracy_score(y_true, y_pred))

        row = {
            "target": target,
            "feature_set": fs,
            "model": model,
            "feature_kind": kind,
            "reuse_group": group,
            "n": int(len(g)),
            "accuracy": acc,
            "macro_f1": macro,
            "true_dist": json.dumps(class_distribution(y_true), ensure_ascii=False),
            "pred_dist": json.dumps(class_distribution(y_pred), ensure_ascii=False),
            "majority_pred": str(y_pred.value_counts().idxmax()) if len(y_pred) else "",
            "majority_true": str(y_true.value_counts().idxmax()) if len(y_true) else "",
        }

        for lab in labels:
            row[f"true_{lab}_rate"] = float((y_true == lab).mean())
            row[f"pred_{lab}_rate"] = float((y_pred == lab).mean())

        confusion_rows.append(row)

    confusion_df = pd.DataFrame(confusion_rows)
    confusion_df.to_csv(OUT_CONFUSION, index=False, encoding="utf-8-sig")

    # =========================
    # 2. Group target distribution
    # =========================

    dist_rows = []
    for target in ["gain_oracle_action", "best_policy_action", "best_policy_id"]:
        if target not in features.columns:
            continue
        for group, g in features.groupby("reuse_group", dropna=False):
            y = g[target].astype(str)
            row = {
                "target": target,
                "reuse_group": group,
                "n": int(len(g)),
                "dist_json": json.dumps(class_distribution(y), ensure_ascii=False),
                "majority_class": str(y.value_counts().idxmax()),
                "majority_rate": float(y.value_counts().max() / len(y)),
                "n_classes": int(y.nunique()),
            }
            for cls, rate in class_distribution(y).items():
                row[f"class_{cls}_rate"] = rate
            dist_rows.append(row)

    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(OUT_DIST, index=False, encoding="utf-8-sig")

    # =========================
    # 3. Collapse audit
    # =========================

    collapse_rows = []
    for keys, g in rg.groupby(["target", "feature_set", "model", "feature_kind"], dropna=False):
        target, fs, model, kind = keys
        y_true = g["y_true"].astype(str)
        y_pred = g["y_pred"].astype(str)
        pred_dist = y_pred.value_counts(normalize=True)
        true_dist = y_true.value_counts(normalize=True)

        collapse_rows.append({
            "target": target,
            "feature_set": fs,
            "model": model,
            "feature_kind": kind,
            "n": int(len(g)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": safe_macro_f1(y_true, y_pred),
            "pred_n_classes": int(y_pred.nunique()),
            "true_n_classes": int(y_true.nunique()),
            "pred_majority_class": str(pred_dist.index[0]),
            "pred_majority_rate": float(pred_dist.iloc[0]),
            "true_majority_class": str(true_dist.index[0]),
            "true_majority_rate": float(true_dist.iloc[0]),
            "collapse_score_pred_majority_minus_true_majority": float(pred_dist.iloc[0] - true_dist.iloc[0]),
        })

    collapse_df = pd.DataFrame(collapse_rows)
    collapse_df = collapse_df.sort_values(["target", "macro_f1"], ascending=[True, False])
    collapse_df.to_csv(OUT_COLLAPSE, index=False, encoding="utf-8-sig")

    # =========================
    # 4. Concept-vs-reuse generalization gap
    # =========================

    cv2 = cv.copy()
    cv2["feature_kind"] = cv2["feature_set"].apply(kind_of_feature_set)

    # Pair same target / feature / model between group_by_concept and group_by_reuse_group.
    concept = cv2[cv2["split"] == "group_by_concept"].copy()
    reuse = cv2[cv2["split"] == "group_by_reuse_group"].copy()

    gap = concept.merge(
        reuse,
        on=["target", "feature_set", "model"],
        how="inner",
        suffixes=("_concept", "_reuse_group"),
    )

    if len(gap):
        gap["macro_f1_gap_concept_minus_reuse"] = gap["macro_f1_concept"] - gap["macro_f1_reuse_group"]
        gap["accuracy_gap_concept_minus_reuse"] = gap["accuracy_concept"] - gap["accuracy_reuse_group"]
        gap["feature_kind"] = gap["feature_set"].apply(kind_of_feature_set)
        gap = gap.sort_values("macro_f1_gap_concept_minus_reuse", ascending=False)

    gap.to_csv(OUT_GAP, index=False, encoding="utf-8-sig")

    # =========================
    # 5. Importance summary by feature windows
    # =========================

    top_feat_rows = []
    if not importance.empty:
        imp = importance.copy()
        imp["window"] = "other"
        imp.loc[imp["feature"].str.contains("init_0_6|_L0_|_L1_|_L2_|_L3_|_L4_|_L5_|_L6_", na=False), "window"] = "init"
        imp.loc[imp["feature"].str.contains("mid_sparse_7_19|_L7_|_L10_|_L13_|_L16_|_L19_", na=False), "window"] = "mid"
        imp.loc[imp["feature"].str.contains("boundary_20_25|_L20_|_L21_|_L22_|_L23_|_L24_|_L25_", na=False), "window"] = "boundary"

        for keys, g in imp.groupby(["target", "feature_set", "window"], dropna=False):
            target, fs, window = keys
            top_feat_rows.append({
                "target": target,
                "feature_set": fs,
                "window": window,
                "n_top_features": int(len(g)),
                "importance_sum": float(g["importance"].sum()),
                "importance_mean": float(g["importance"].mean()),
                "top_feature": str(g.sort_values("importance", ascending=False).iloc[0]["feature"]),
                "top_importance": float(g["importance"].max()),
            })

    top_features_df = pd.DataFrame(top_feat_rows)
    if len(top_features_df):
        top_features_df = top_features_df.sort_values(["target", "feature_set", "importance_sum"], ascending=[True, True, False])
    top_features_df.to_csv(OUT_TOP_FEATURES, index=False, encoding="utf-8-sig")

    # =========================
    # Summary verdict
    # =========================

    best_rg_by_kind = (
        collapse_df
        .sort_values(["target", "feature_kind", "macro_f1"], ascending=[True, True, False])
        .groupby(["target", "feature_kind"], as_index=False)
        .head(1)
    )

    # Identify strongest concept-to-reuse collapse.
    if len(gap):
        gap_top = gap.head(20).to_dict(orient="records")
    else:
        gap_top = []

    # Main interpretation.
    main_target = "gain_oracle_action"
    main = best_rg_by_kind[best_rg_by_kind["target"] == main_target]
    best_tab = main[main["feature_kind"] == "tabular"]["macro_f1"].max()
    best_topk = main[main["feature_kind"] == "topk"]["macro_f1"].max()
    best_fused = main[main["feature_kind"] == "fused"]["macro_f1"].max()

    best_tab = float(best_tab) if pd.notna(best_tab) else None
    best_topk = float(best_topk) if pd.notna(best_topk) else None
    best_fused = float(best_fused) if pd.notna(best_fused) else None

    if best_topk is not None and best_tab is not None and best_topk > best_tab + 0.05:
        verdict = "TOPK_HAS_CROSS_REUSE_SIGNAL"
    elif best_fused is not None and best_tab is not None and best_fused > best_tab + 0.05:
        verdict = "FUSED_HAS_CROSS_REUSE_SIGNAL"
    elif best_topk is not None and best_tab is not None and best_topk < best_tab - 0.10:
        verdict = "TOPK_FAILS_CROSS_REUSE_DUE_TO_GROUP_SHIFT"
    else:
        verdict = "MIXED_CROSS_REUSE_DIAGNOSTIC"

    summary = {
        "inputs": {
            "cv": str(IN_CV),
            "preds": str(IN_PREDS),
            "utility": str(IN_UTILITY),
            "importance": str(IN_IMPORTANCE),
            "features": str(IN_FEATURES),
        },
        "n_prediction_rows": int(len(preds)),
        "n_feature_rows": int(len(features)),
        "best_group_by_reuse_group_by_kind": best_rg_by_kind.to_dict(orient="records"),
        "concept_vs_reuse_gap_top": gap_top,
        "main_target": main_target,
        "best_group_by_reuse_group_macro_f1": {
            "tabular": best_tab,
            "topk": best_topk,
            "fused": best_fused,
        },
        "verdict": verdict,
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[SAVED]")
    print(f"  confusion       : {OUT_CONFUSION}")
    print(f"  distribution    : {OUT_DIST}")
    print(f"  collapse audit  : {OUT_COLLAPSE}")
    print(f"  concept/reuse gap: {OUT_GAP}")
    print(f"  top features    : {OUT_TOP_FEATURES}")
    print(f"  summary         : {OUT_SUMMARY}")
    print("\n[VERDICT]")
    print(verdict)


if __name__ == "__main__":
    main()
