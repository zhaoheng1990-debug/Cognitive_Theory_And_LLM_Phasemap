# -*- coding: utf-8 -*-
"""
SR-1C: TopK/VIM vs Tabular Baseline Audit

Purpose
-------
Evaluate whether TopK/VIM features improve StructuralResolution prediction
beyond low-cost tabular metadata, especially under held-out reuse_group split.

Inputs
------
    sr1b1_outputs/sr1b1_topk_vim_features.csv

Outputs
-------
    sr1c_outputs/sr1c_cv_results.csv
    sr1c_outputs/sr1c_predictions.csv
    sr1c_outputs/sr1c_policy_utility.csv
    sr1c_outputs/sr1c_feature_importance.csv
    sr1c_outputs/sr1c_summary.json

Core targets
------------
    gain_oracle_action
    best_policy_action
    best_policy_id

Main comparison
---------------
    tabular baselines:
        boundary_only
        symbolic_no_concept
        all_tabular_text

    TopK/VIM groups:
        topk_init_only
        topk_mid_only
        topk_boundary_only
        topk_init_mid
        topk_mid_boundary
        topk_all

    Fused:
        symbolic_no_concept + topk groups
        all_tabular_text + topk groups

Key criterion
-------------
    TopK/VIM or fused feature groups should improve group_by_reuse_group macro-F1
    over tabular baseline. Otherwise TopK/VIM is not yet adding cross-boundary
    structural-resolution signal.
"""

import json
import warnings
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC


warnings.filterwarnings("ignore")


BASE_DIR = Path(r"C:\Users\ZH\Desktop\AGI\python_script")
INPUT_CSV = BASE_DIR / "sr1b1_outputs" / "sr1b1_topk_vim_features.csv"

OUT_DIR = BASE_DIR / "sr1c_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CV = OUT_DIR / "sr1c_cv_results.csv"
OUT_PREDS = OUT_DIR / "sr1c_predictions.csv"
OUT_UTILITY = OUT_DIR / "sr1c_policy_utility.csv"
OUT_IMPORTANCE = OUT_DIR / "sr1c_feature_importance.csv"
OUT_SUMMARY = OUT_DIR / "sr1c_summary.json"


META_COLS = [
    "task_id",
    "reuse_group",
    "task_family",
    "target_operator",
    "concept",
    "boundary_strength",
    "sr_policy_v1_action",
    "gain_oracle_action",
    "best_policy_action",
    "best_policy_id",
    "correct_vs_no",
    "correct_vs_wrong",
    "correct_vs_random",
    "best_policy_gain",
    "best_policy_vs_no",
    "best_policy_vs_wrong",
    "best_policy_vs_random",
]


TARGETS = [
    "gain_oracle_action",
    "best_policy_action",
    "best_policy_id",
]


SPLITS = [
    "stratified5",
    "group_by_concept",
    "group_by_task_family",
    "group_by_reuse_group",
]


FEATURE_SETS = [
    "boundary_only",
    "reuse_group_only",
    "boundary_plus_group",
    "symbolic_no_concept",
    "symbolic_with_concept",
    "surface_text_only",
    "all_tabular_text",

    "topk_init_only",
    "topk_mid_only",
    "topk_boundary_only",
    "topk_init_mid",
    "topk_mid_boundary",
    "topk_all",

    "symbolic_plus_topk_init",
    "symbolic_plus_topk_mid",
    "symbolic_plus_topk_boundary",
    "symbolic_plus_topk_all",

    "all_tabular_text_plus_topk_all",
]


MODELS = [
    "dummy_most_frequent",
    "logistic_l2",
    "linear_svc",
    "rf_small",
    "extra_trees",
]


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def read_input() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Required input not found: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"[LOAD] {INPUT_CSV}")
    print(f"       shape={df.shape}")
    print(f"       columns={len(df.columns)}")
    return df


def is_topk_feature(c: str) -> bool:
    return str(c).startswith("k50_") or str(c).startswith("k100_") or str(c).startswith("k500_")


def topk_feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    all_topk = [c for c in df.columns if is_topk_feature(c)]

    init = [c for c in all_topk if "_init_0_6_" in c or "_L0_" in c or "_L1_" in c or "_L2_" in c or "_L3_" in c or "_L4_" in c or "_L5_" in c or "_L6_" in c or "_T0_1_" in c or "_T1_2_" in c or "_T2_3_" in c or "_T3_4_" in c or "_T4_5_" in c or "_T5_6_" in c]

    mid = [c for c in all_topk if "_mid_sparse_7_19_" in c or "_L7_" in c or "_L10_" in c or "_L13_" in c or "_L16_" in c or "_L19_" in c or "_T7_10_" in c or "_T10_13_" in c or "_T13_16_" in c or "_T16_19_" in c]

    boundary = [c for c in all_topk if "_boundary_20_25_" in c or "_L20_" in c or "_L21_" in c or "_L22_" in c or "_L23_" in c or "_L24_" in c or "_L25_" in c or "_T20_21_" in c or "_T21_22_" in c or "_T22_23_" in c or "_T23_24_" in c or "_T24_25_" in c]

    init_mid = sorted(set(init + mid + [c for c in all_topk if "_init_plus_mid_" in c]))
    mid_boundary = sorted(set(mid + boundary + [c for c in all_topk if "_mid_plus_boundary_" in c]))

    return {
        "init": sorted(set(init)),
        "mid": sorted(set(mid)),
        "boundary": sorted(set(boundary)),
        "init_mid": init_mid,
        "mid_boundary": mid_boundary,
        "all": all_topk,
    }


def get_feature_spec(df: pd.DataFrame, feature_set: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns numeric_features, categorical_features, text_features.
    """
    groups = topk_feature_groups(df)

    numeric = []
    categorical = []
    text = []

    if feature_set == "boundary_only":
        numeric = ["boundary_strength"]

    elif feature_set == "reuse_group_only":
        categorical = ["reuse_group"]

    elif feature_set == "boundary_plus_group":
        numeric = ["boundary_strength"]
        categorical = ["reuse_group"]

    elif feature_set == "symbolic_no_concept":
        numeric = ["boundary_strength"]
        categorical = ["reuse_group", "task_family", "target_operator"]

    elif feature_set == "symbolic_with_concept":
        numeric = ["boundary_strength"]
        categorical = ["reuse_group", "task_family", "target_operator", "concept"]

    elif feature_set == "surface_text_only":
        text = ["request"] if "request" in df.columns else []

    elif feature_set == "all_tabular_text":
        numeric = ["boundary_strength"]
        categorical = ["reuse_group", "task_family", "target_operator", "concept"]
        text = ["request"] if "request" in df.columns else []

    elif feature_set == "topk_init_only":
        numeric = groups["init"]

    elif feature_set == "topk_mid_only":
        numeric = groups["mid"]

    elif feature_set == "topk_boundary_only":
        numeric = groups["boundary"]

    elif feature_set == "topk_init_mid":
        numeric = groups["init_mid"]

    elif feature_set == "topk_mid_boundary":
        numeric = groups["mid_boundary"]

    elif feature_set == "topk_all":
        numeric = groups["all"]

    elif feature_set == "symbolic_plus_topk_init":
        numeric = ["boundary_strength"] + groups["init"]
        categorical = ["reuse_group", "task_family", "target_operator"]

    elif feature_set == "symbolic_plus_topk_mid":
        numeric = ["boundary_strength"] + groups["mid"]
        categorical = ["reuse_group", "task_family", "target_operator"]

    elif feature_set == "symbolic_plus_topk_boundary":
        numeric = ["boundary_strength"] + groups["boundary"]
        categorical = ["reuse_group", "task_family", "target_operator"]

    elif feature_set == "symbolic_plus_topk_all":
        numeric = ["boundary_strength"] + groups["all"]
        categorical = ["reuse_group", "task_family", "target_operator"]

    elif feature_set == "all_tabular_text_plus_topk_all":
        numeric = ["boundary_strength"] + groups["all"]
        categorical = ["reuse_group", "task_family", "target_operator", "concept"]
        text = ["request"] if "request" in df.columns else []

    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    numeric = [c for c in numeric if c in df.columns]
    categorical = [c for c in categorical if c in df.columns]
    text = [c for c in text if c in df.columns]

    return numeric, categorical, text


def build_pipeline(df: pd.DataFrame, feature_set: str, model_name: str):
    numeric, categorical, text = get_feature_spec(df, feature_set)

    transformers = []

    if numeric:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold(threshold=0.0)),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipe, numeric))

    if categorical:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", make_ohe()),
        ])
        transformers.append(("cat", cat_pipe, categorical))

    if text:
        transformers.append((
            "text",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=1,
                max_features=2000,
            ),
            text[0],
        ))

    if not transformers:
        raise ValueError(f"No features selected for {feature_set}")

    pre = ColumnTransformer(transformers=transformers, remainder="drop")

    if model_name == "dummy_most_frequent":
        clf = DummyClassifier(strategy="most_frequent")

    elif model_name == "logistic_l2":
        clf = LogisticRegression(
            C=0.5,
            solver="lbfgs",
            max_iter=3000,
            class_weight="balanced",
        )

    elif model_name == "linear_svc":
        clf = LinearSVC(
            C=0.5,
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
        )

    elif model_name == "rf_small":
        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=3,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )

    elif model_name == "extra_trees":
        clf = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return Pipeline([("pre", pre), ("clf", clf)])


def get_splits(df: pd.DataFrame, target: str, split_name: str):
    y = df[target].astype(str).values

    if split_name == "stratified5":
        vc = pd.Series(y).value_counts()
        if vc.min() < 2:
            return []
        n_splits = int(min(5, vc.min()))
        n_splits = max(2, n_splits)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return list(splitter.split(df, y))

    if split_name == "group_by_concept":
        groups = df["concept"].astype(str).values
        if len(np.unique(groups)) < 2:
            return []
        splitter = GroupKFold(n_splits=int(min(5, len(np.unique(groups)))))
        return list(splitter.split(df, y, groups))

    if split_name == "group_by_task_family":
        groups = df["task_family"].astype(str).values
        if len(np.unique(groups)) < 2:
            return []
        splitter = GroupKFold(n_splits=int(min(5, len(np.unique(groups)))))
        return list(splitter.split(df, y, groups))

    if split_name == "group_by_reuse_group":
        groups = df["reuse_group"].astype(str).values
        if len(np.unique(groups)) < 2:
            return []
        splitter = GroupKFold(n_splits=int(min(5, len(np.unique(groups)))))
        return list(splitter.split(df, y, groups))

    raise ValueError(f"Unknown split_name: {split_name}")


def cv_predict(df: pd.DataFrame, target: str, feature_set: str, model_name: str, split_name: str):
    splits = get_splits(df, target, split_name)
    if not splits:
        return None, None

    y = df[target].astype(str).values
    pred = np.empty(len(df), dtype=object)
    pred[:] = None

    for fold, (tr, te) in enumerate(splits):
        # Some group splits can create train folds with missing classes.
        # That is allowed, but metrics will expose instability.
        pipe = build_pipeline(df, feature_set, model_name)
        pipe.fit(df.iloc[tr], y[tr])
        pred[te] = pipe.predict(df.iloc[te])

    mask = pd.Series(pred).notna().values

    metrics = {
        "target": target,
        "feature_set": feature_set,
        "model": model_name,
        "split": split_name,
        "n": int(mask.sum()),
        "accuracy": float(accuracy_score(y[mask], pred[mask])),
        "macro_f1": float(f1_score(y[mask], pred[mask], average="macro")),
    }

    pred_df = pd.DataFrame({
        "task_id": df["task_id"].astype(str),
        "target": target,
        "feature_set": feature_set,
        "model": model_name,
        "split": split_name,
        "y_true": y,
        "y_pred": pred,
        "is_correct": (y == pred).astype(int),
    })

    return metrics, pred_df


def action_utility(row, action_value):
    action = str(action_value)
    if action == "direct_reuse":
        return row.get("correct_vs_no", np.nan)
    if action in ["observe", "reject", "no_reuse"]:
        return 0.0
    return np.nan


def policy_id_utility(row, policy_id_value):
    pid = str(policy_id_value)
    c = f"{pid}__policy_gain"
    if c in row.index:
        return row[c]
    return np.nan


def summarize_policy_utility(df: pd.DataFrame, preds: pd.DataFrame):
    rows = []
    if preds.empty:
        return pd.DataFrame(rows)

    pred_keep = preds[[
        "task_id",
        "target",
        "feature_set",
        "model",
        "split",
        "y_true",
        "y_pred",
    ]].copy()

    pred_keep = pred_keep.rename(columns={
        "target": "pred_target_name",
        "y_true": "pred_true",
        "y_pred": "pred_label",
    })

    for target_name in ["gain_oracle_action", "best_policy_action"]:
        sub = pred_keep[pred_keep["pred_target_name"] == target_name].copy()
        if sub.empty:
            continue

        merged = df.merge(sub, on="task_id", how="inner", suffixes=("", "_pred"))

        merged["pred_action_gain_vs_no"] = merged.apply(
            lambda r: action_utility(r, r["pred_label"]),
            axis=1,
        )
        merged["true_action_gain_vs_no"] = merged.apply(
            lambda r: action_utility(r, r["pred_true"]),
            axis=1,
        )

        for keys, g in merged.groupby(["pred_target_name", "feature_set", "model", "split"]):
            pred_mean = pd.to_numeric(g["pred_action_gain_vs_no"], errors="coerce").mean()
            true_mean = pd.to_numeric(g["true_action_gain_vs_no"], errors="coerce").mean()
            rows.append({
                "target": keys[0],
                "feature_set": keys[1],
                "model": keys[2],
                "split": keys[3],
                "n": int(len(g)),
                "mean_pred_action_gain_vs_no": float(pred_mean) if pd.notna(pred_mean) else None,
                "mean_true_action_gain_vs_no": float(true_mean) if pd.notna(true_mean) else None,
                "gain_gap_pred_minus_true": float(pred_mean - true_mean) if pd.notna(pred_mean) and pd.notna(true_mean) else None,
            })

    sub = pred_keep[pred_keep["pred_target_name"] == "best_policy_id"].copy()
    if not sub.empty:
        merged = df.merge(sub, on="task_id", how="inner", suffixes=("", "_pred"))
        merged["pred_policy_gain"] = merged.apply(
            lambda r: policy_id_utility(r, r["pred_label"]),
            axis=1,
        )
        merged["true_policy_gain"] = merged["best_policy_gain"]

        for keys, g in merged.groupby(["pred_target_name", "feature_set", "model", "split"]):
            pred_mean = pd.to_numeric(g["pred_policy_gain"], errors="coerce").mean()
            true_mean = pd.to_numeric(g["true_policy_gain"], errors="coerce").mean()
            rows.append({
                "target": keys[0],
                "feature_set": keys[1],
                "model": keys[2],
                "split": keys[3],
                "n": int(len(g)),
                "mean_pred_policy_gain": float(pred_mean) if pd.notna(pred_mean) else None,
                "mean_true_policy_gain": float(true_mean) if pd.notna(true_mean) else None,
                "gain_gap_pred_minus_true": float(pred_mean - true_mean) if pd.notna(pred_mean) and pd.notna(true_mean) else None,
            })

    return pd.DataFrame(rows)


def fit_importance(df: pd.DataFrame, target: str, feature_set: str):
    """
    Lightweight full-data feature importance using ExtraTrees.
    Only for interpretation, not for proof.
    """
    numeric, categorical, text = get_feature_spec(df, feature_set)

    if text:
        # Avoid huge text importances here.
        text = []

    if not numeric and not categorical:
        return pd.DataFrame()

    pipe = build_pipeline(df, feature_set, "extra_trees")
    y = df[target].astype(str).values
    pipe.fit(df, y)

    clf = pipe.named_steps["clf"]
    pre = pipe.named_steps["pre"]

    if not hasattr(clf, "feature_importances_"):
        return pd.DataFrame()

    try:
        names = pre.get_feature_names_out()
    except Exception:
        names = [f"f_{i}" for i in range(len(clf.feature_importances_))]

    imp = pd.DataFrame({
        "target": target,
        "feature_set": feature_set,
        "feature": names,
        "importance": clf.feature_importances_,
    })

    imp = imp.sort_values("importance", ascending=False).head(100)
    return imp


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["reuse_group", "task_family", "target_operator", "concept"]:
        if c in df.columns:
            df[c] = df[c].fillna("NA").astype(str)

    if "request" not in df.columns:
        df["request"] = ""

    df["request"] = df["request"].fillna("").astype(str)

    if "boundary_strength" in df.columns:
        df["boundary_strength"] = pd.to_numeric(df["boundary_strength"], errors="coerce").fillna(-1)

    # Convert TopK columns to numeric and replace inf with nan.
    topk_cols = [c for c in df.columns if is_topk_feature(c)]
    for c in topk_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[topk_cols] = df[topk_cols].replace([np.inf, -np.inf], np.nan)

    return df


def main():
    df = read_input()
    df = prepare_dataframe(df)

    required = ["task_id", "gain_oracle_action", "best_policy_action", "best_policy_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    groups = topk_feature_groups(df)

    print("[FEATURE GROUPS]")
    for k, v in groups.items():
        print(f"  {k}: {len(v)}")

    result_rows = []
    pred_parts = []

    for target in TARGETS:
        for feature_set in FEATURE_SETS:
            numeric, categorical, text = get_feature_spec(df, feature_set)
            if not numeric and not categorical and not text:
                print(f"[SKIP] no features: {feature_set}")
                continue

            for model_name in MODELS:
                for split_name in SPLITS:
                    try:
                        metrics, pred_df = cv_predict(df, target, feature_set, model_name, split_name)
                        if metrics is None:
                            continue
                        result_rows.append(metrics)
                        pred_parts.append(pred_df)

                        print(
                            f"[CV] target={target:20s} feature={feature_set:30s} "
                            f"model={model_name:20s} split={split_name:20s} "
                            f"acc={metrics['accuracy']:.3f} f1={metrics['macro_f1']:.3f}"
                        )
                    except Exception as e:
                        result_rows.append({
                            "target": target,
                            "feature_set": feature_set,
                            "model": model_name,
                            "split": split_name,
                            "n": 0,
                            "accuracy": np.nan,
                            "macro_f1": np.nan,
                            "error": str(e),
                        })
                        print(f"[WARN] Failed: {target} {feature_set} {model_name} {split_name}: {e}")

    results = pd.DataFrame(result_rows)
    preds = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    utility = summarize_policy_utility(df, preds) if not preds.empty else pd.DataFrame()

    importance_parts = []
    for target in ["gain_oracle_action", "best_policy_action"]:
        for feature_set in [
            "topk_init_only",
            "topk_mid_only",
            "topk_boundary_only",
            "topk_all",
            "symbolic_plus_topk_all",
        ]:
            try:
                imp = fit_importance(df, target, feature_set)
                if not imp.empty:
                    importance_parts.append(imp)
            except Exception as e:
                print(f"[WARN] importance failed: {target} {feature_set}: {e}")

    importance = pd.concat(importance_parts, ignore_index=True) if importance_parts else pd.DataFrame()

    results.to_csv(OUT_CV, index=False, encoding="utf-8-sig")
    preds.to_csv(OUT_PREDS, index=False, encoding="utf-8-sig")
    utility.to_csv(OUT_UTILITY, index=False, encoding="utf-8-sig")
    importance.to_csv(OUT_IMPORTANCE, index=False, encoding="utf-8-sig")

    clean = results.dropna(subset=["macro_f1"]).copy()

    best_by_target_split = (
        clean.sort_values(
            ["target", "split", "macro_f1", "accuracy"],
            ascending=[True, True, False, False],
        )
        .groupby(["target", "split"], as_index=False)
        .head(1)
        if not clean.empty else pd.DataFrame()
    )

    # Compare best tabular vs best topk/fused under group_by_reuse_group.
    rg = clean[clean["split"] == "group_by_reuse_group"].copy()

    tabular_sets = {
        "boundary_only",
        "reuse_group_only",
        "boundary_plus_group",
        "symbolic_no_concept",
        "symbolic_with_concept",
        "surface_text_only",
        "all_tabular_text",
    }

    topk_or_fused_sets = set(FEATURE_SETS) - tabular_sets

    comparison = {}
    for target in TARGETS:
        sub = rg[rg["target"] == target]
        tab = sub[sub["feature_set"].isin(tabular_sets)]
        top = sub[sub["feature_set"].isin(topk_or_fused_sets)]

        best_tab = tab.sort_values("macro_f1", ascending=False).head(1).to_dict(orient="records")
        best_top = top.sort_values("macro_f1", ascending=False).head(1).to_dict(orient="records")

        best_tab_f1 = best_tab[0]["macro_f1"] if best_tab else np.nan
        best_top_f1 = best_top[0]["macro_f1"] if best_top else np.nan

        comparison[target] = {
            "best_tabular": best_tab[0] if best_tab else None,
            "best_topk_or_fused": best_top[0] if best_top else None,
            "delta_topk_minus_tabular_macro_f1": (
                float(best_top_f1 - best_tab_f1)
                if pd.notna(best_top_f1) and pd.notna(best_tab_f1)
                else None
            ),
        }

    # Verdict.
    gain_delta = comparison.get("gain_oracle_action", {}).get("delta_topk_minus_tabular_macro_f1", None)
    best_gain_top = comparison.get("gain_oracle_action", {}).get("best_topk_or_fused", None)

    if gain_delta is not None and gain_delta >= 0.10:
        verdict = "PASS_STRONG_TOPK_VIM_IMPROVES_CROSS_REUSE_GROUP"
    elif gain_delta is not None and gain_delta >= 0.03:
        verdict = "PASS_LITE_TOPK_VIM_IMPROVES_CROSS_REUSE_GROUP"
    elif gain_delta is not None and gain_delta > -0.03:
        verdict = "MIXED_TOPK_VIM_PARITY_WITH_TABULAR"
    else:
        verdict = "FAIL_TOPK_VIM_DOES_NOT_BEAT_TABULAR_YET"

    summary = {
        "input_csv": str(INPUT_CSV),
        "n_rows": int(len(df)),
        "feature_group_counts": {k: len(v) for k, v in groups.items()},
        "targets": TARGETS,
        "feature_sets": FEATURE_SETS,
        "models": MODELS,
        "splits": SPLITS,
        "best_by_target_split": best_by_target_split.to_dict(orient="records"),
        "cross_reuse_group_comparison": comparison,
        "verdict": verdict,
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[SAVED]")
    print(f"  cv results        : {OUT_CV}")
    print(f"  predictions       : {OUT_PREDS}")
    print(f"  utility           : {OUT_UTILITY}")
    print(f"  feature importance: {OUT_IMPORTANCE}")
    print(f"  summary           : {OUT_SUMMARY}")

    print("\n[VERDICT]")
    print(verdict)


if __name__ == "__main__":
    main()
