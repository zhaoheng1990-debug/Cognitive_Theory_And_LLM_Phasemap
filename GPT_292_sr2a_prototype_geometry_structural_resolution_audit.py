# -*- coding: utf-8 -*-
"""
SR-2A: Prototype Geometry StructuralResolution Audit

Purpose
-------
SR-1C.1 showed raw single-prompt TopK/VIM features are strong under
group_by_concept but fail under group_by_reuse_group.

Hypothesis
----------
StructuralResolution is not an absolute property of a target prompt.
It is a relative geometry property:

    target context relative to direct / observe / reject prototypes

This script converts TopK/VIM vectors into fold-local prototype geometry
features and evaluates whether prototype distances/margins generalize better
across held-out reuse groups.

Inputs
------
    sr1b1_outputs/sr1b1_topk_vim_features.csv
    sr1c_outputs/sr1c_cv_results.csv       optional, for baseline comparison

Outputs
-------
    sr2a_outputs/sr2a_cv_results.csv
    sr2a_outputs/sr2a_predictions.csv
    sr2a_outputs/sr2a_policy_utility.csv
    sr2a_outputs/sr2a_summary.json

Key target
----------
    gain_oracle_action

Main PASS criterion
-------------------
    Under group_by_reuse_group:
        prototype_topk_* Macro-F1 > best tabular baseline from SR-1C
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


warnings.filterwarnings("ignore")


BASE_DIR = Path(r"C:\Users\ZH\Desktop\AGI\python_script")

IN_FEATURES = BASE_DIR / "sr1b1_outputs" / "sr1b1_topk_vim_features.csv"
IN_SR1C = BASE_DIR / "sr1c_outputs" / "sr1c_cv_results.csv"

OUT_DIR = BASE_DIR / "sr2a_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CV = OUT_DIR / "sr2a_cv_results.csv"
OUT_PREDS = OUT_DIR / "sr2a_predictions.csv"
OUT_UTILITY = OUT_DIR / "sr2a_policy_utility.csv"
OUT_SUMMARY = OUT_DIR / "sr2a_summary.json"


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
    "topk_init",
    "topk_mid",
    "topk_boundary",
    "topk_init_mid",
    "topk_mid_boundary",
    "topk_all",
]

PROTOTYPE_MODELS = [
    "nearest_proto",
    "proto_logistic",
    "proto_rf",
    "proto_extra_trees",
]

PCA_COMPONENTS = 32
SEED = 42


def read_csv(path: Path, required=True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"[LOAD] {path} shape={df.shape}")
    return df


def is_topk_feature(c: str) -> bool:
    return str(c).startswith("k50_") or str(c).startswith("k100_") or str(c).startswith("k500_")


def topk_feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    all_topk = [
        c for c in df.columns
        if is_topk_feature(c)
        and not str(c).endswith("_top_id")
    ]

    init = [
        c for c in all_topk
        if "_init_0_6_" in c
        or "_L0_" in c or "_L1_" in c or "_L2_" in c
        or "_L3_" in c or "_L4_" in c or "_L5_" in c or "_L6_" in c
        or "_T0_1_" in c or "_T1_2_" in c or "_T2_3_" in c
        or "_T3_4_" in c or "_T4_5_" in c or "_T5_6_" in c
    ]

    mid = [
        c for c in all_topk
        if "_mid_sparse_7_19_" in c
        or "_L7_" in c or "_L10_" in c or "_L13_" in c
        or "_L16_" in c or "_L19_" in c
        or "_T7_10_" in c or "_T10_13_" in c
        or "_T13_16_" in c or "_T16_19_" in c
    ]

    boundary = [
        c for c in all_topk
        if "_boundary_20_25_" in c
        or "_L20_" in c or "_L21_" in c or "_L22_" in c
        or "_L23_" in c or "_L24_" in c or "_L25_" in c
        or "_T20_21_" in c or "_T21_22_" in c
        or "_T22_23_" in c or "_T23_24_" in c or "_T24_25_" in c
    ]

    init_mid = sorted(set(init + mid + [c for c in all_topk if "_init_plus_mid_" in c]))
    mid_boundary = sorted(set(mid + boundary + [c for c in all_topk if "_mid_plus_boundary_" in c]))

    return {
        "topk_init": sorted(set(init)),
        "topk_mid": sorted(set(mid)),
        "topk_boundary": sorted(set(boundary)),
        "topk_init_mid": init_mid,
        "topk_mid_boundary": mid_boundary,
        "topk_all": all_topk,
    }


def get_splits(df: pd.DataFrame, target: str, split_name: str):
    y = df[target].astype(str).values

    if split_name == "stratified5":
        vc = pd.Series(y).value_counts()
        if vc.min() < 2:
            return []
        n_splits = int(min(5, vc.min()))
        n_splits = max(2, n_splits)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
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


def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    An = np.linalg.norm(A, axis=1, keepdims=True)
    Bn = np.linalg.norm(B, axis=1, keepdims=True).T
    denom = np.maximum(An * Bn, 1e-12)
    return A @ B.T / denom


class FoldPrototypeEncoder:
    """
    Fit preprocessing + PCA + class prototypes on train only.
    Transform any matrix into prototype-distance features.
    """

    def __init__(self, n_components=32, seed=42):
        self.n_components = n_components
        self.seed = seed
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.pca = None
        self.classes_ = None
        self.centroids_ = None
        self.global_centroid_ = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        X0 = self.imputer.fit_transform(X)
        X1 = self.scaler.fit_transform(X0)

        n_comp = min(self.n_components, X1.shape[0] - 1, X1.shape[1])
        if n_comp >= 2:
            self.pca = PCA(n_components=n_comp, random_state=self.seed)
            X2 = self.pca.fit_transform(X1)
        else:
            self.pca = None
            X2 = X1

        self.classes_ = np.array(sorted(pd.Series(y).astype(str).unique()))
        centroids = []
        for cls in self.classes_:
            mask = pd.Series(y).astype(str).values == cls
            if mask.sum() == 0:
                centroids.append(np.zeros(X2.shape[1], dtype=np.float64))
            else:
                centroids.append(X2[mask].mean(axis=0))
        self.centroids_ = np.vstack(centroids)
        self.global_centroid_ = X2.mean(axis=0, keepdims=True)

        return self

    def _transform_base(self, X: pd.DataFrame) -> np.ndarray:
        X0 = self.imputer.transform(X)
        X1 = self.scaler.transform(X0)
        if self.pca is not None:
            X2 = self.pca.transform(X1)
        else:
            X2 = X1
        return X2

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X2 = self._transform_base(X)

        dists = np.linalg.norm(X2[:, None, :] - self.centroids_[None, :, :], axis=2)
        cos = cosine_matrix(X2, self.centroids_)

        rows = {}
        for j, cls in enumerate(self.classes_):
            safe = cls.replace(" ", "_").replace("/", "_")
            rows[f"dist_to_{safe}"] = dists[:, j]
            rows[f"cos_to_{safe}"] = cos[:, j]

        # General prototype geometry.
        sorted_d = np.sort(dists, axis=1)
        rows["min_dist"] = sorted_d[:, 0]
        rows["second_min_dist"] = sorted_d[:, 1] if dists.shape[1] >= 2 else np.nan
        rows["dist_margin_second_minus_first"] = (
            rows["second_min_dist"] - rows["min_dist"]
            if dists.shape[1] >= 2 else np.nan
        )
        rows["mean_dist"] = dists.mean(axis=1)
        rows["std_dist"] = dists.std(axis=1)
        rows["max_cos"] = cos.max(axis=1)
        rows["mean_cos"] = cos.mean(axis=1)
        rows["std_cos"] = cos.std(axis=1)

        global_dist = np.linalg.norm(X2 - self.global_centroid_, axis=1)
        rows["dist_to_global_centroid"] = global_dist

        nearest_idx = np.argmin(dists, axis=1)
        nearest_cls = self.classes_[nearest_idx]
        rows["nearest_proto_class"] = nearest_cls

        out = pd.DataFrame(rows)

        # One-hot nearest class for downstream classifier.
        for cls in self.classes_:
            safe = cls.replace(" ", "_").replace("/", "_")
            out[f"nearest_is_{safe}"] = (nearest_cls == cls).astype(int)

        return out

    def predict_nearest(self, X: pd.DataFrame) -> np.ndarray:
        X2 = self._transform_base(X)
        dists = np.linalg.norm(X2[:, None, :] - self.centroids_[None, :, :], axis=2)
        nearest_idx = np.argmin(dists, axis=1)
        return self.classes_[nearest_idx]


def build_proto_model(model_name: str):
    if model_name == "nearest_proto":
        return None

    if model_name == "proto_logistic":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=2000,
                class_weight="balanced",
            )),
        ])

    if model_name == "proto_rf":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=3,
                random_state=SEED,
                class_weight="balanced_subsample",
                n_jobs=-1,
            )),
        ])

    if model_name == "proto_extra_trees":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", ExtraTreesClassifier(
                n_estimators=400,
                max_depth=5,
                min_samples_leaf=2,
                random_state=SEED,
                class_weight="balanced",
                n_jobs=-1,
            )),
        ])

    raise ValueError(f"Unknown prototype model: {model_name}")


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


def run_cv(df: pd.DataFrame, target: str, feature_set: str, model_name: str, split_name: str, feature_cols: List[str]):
    splits = get_splits(df, target, split_name)
    if not splits:
        return None, None

    y = df[target].astype(str).values
    pred = np.empty(len(df), dtype=object)
    pred[:] = None

    Xall = df[feature_cols].copy()
    for c in feature_cols:
        Xall[c] = pd.to_numeric(Xall[c], errors="coerce")
    Xall = Xall.replace([np.inf, -np.inf], np.nan)

    for fold, (tr, te) in enumerate(splits):
        enc = FoldPrototypeEncoder(n_components=PCA_COMPONENTS, seed=SEED)
        enc.fit(Xall.iloc[tr], y[tr])

        if model_name == "nearest_proto":
            pred[te] = enc.predict_nearest(Xall.iloc[te])
        else:
            P_tr = enc.transform(Xall.iloc[tr])
            P_te = enc.transform(Xall.iloc[te])

            # nearest_proto_class is string, remove it for numeric classifiers.
            if "nearest_proto_class" in P_tr.columns:
                P_tr = P_tr.drop(columns=["nearest_proto_class"])
                P_te = P_te.drop(columns=["nearest_proto_class"])

            clf = build_proto_model(model_name)
            clf.fit(P_tr, y[tr])
            pred[te] = clf.predict(P_te)

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


def main():
    df = read_csv(IN_FEATURES)

    for c in ["reuse_group", "task_family", "target_operator", "concept"]:
        if c not in df.columns:
            raise ValueError(f"Missing required meta column: {c}")
        df[c] = df[c].fillna("NA").astype(str)

    if "request" in df.columns:
        df["request"] = df["request"].fillna("").astype(str)

    groups = topk_feature_groups(df)
    print("[FEATURE GROUPS]")
    for k, v in groups.items():
        print(f"  {k}: {len(v)}")

    result_rows = []
    pred_parts = []

    for target in TARGETS:
        if target not in df.columns:
            print(f"[SKIP] missing target {target}")
            continue

        for feature_set in FEATURE_SETS:
            cols = groups.get(feature_set, [])
            if not cols:
                print(f"[SKIP] no cols for {feature_set}")
                continue

            for model_name in PROTOTYPE_MODELS:
                for split_name in SPLITS:
                    try:
                        metrics, pred_df = run_cv(df, target, feature_set, model_name, split_name, cols)
                        if metrics is None:
                            continue

                        result_rows.append(metrics)
                        pred_parts.append(pred_df)

                        print(
                            f"[CV] target={target:20s} fs={feature_set:18s} "
                            f"model={model_name:18s} split={split_name:20s} "
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
                        print(f"[WARN] Failed {target} {feature_set} {model_name} {split_name}: {e}")

    results = pd.DataFrame(result_rows)
    preds = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    utility = summarize_policy_utility(df, preds) if not preds.empty else pd.DataFrame()

    results.to_csv(OUT_CV, index=False, encoding="utf-8-sig")
    preds.to_csv(OUT_PREDS, index=False, encoding="utf-8-sig")
    utility.to_csv(OUT_UTILITY, index=False, encoding="utf-8-sig")

    # Optional comparison against SR-1C tabular baseline.
    sr1c = read_csv(IN_SR1C, required=False)
    comparison = {}

    if not sr1c.empty:
        clean = sr1c.dropna(subset=["macro_f1"]).copy()
        tabular_sets = {
            "boundary_only",
            "reuse_group_only",
            "boundary_plus_group",
            "symbolic_no_concept",
            "symbolic_with_concept",
            "surface_text_only",
            "all_tabular_text",
        }

        proto_clean = results.dropna(subset=["macro_f1"]).copy()

        for target in TARGETS:
            base = clean[
                (clean["target"] == target)
                & (clean["split"] == "group_by_reuse_group")
                & (clean["feature_set"].isin(tabular_sets))
            ]
            proto = proto_clean[
                (proto_clean["target"] == target)
                & (proto_clean["split"] == "group_by_reuse_group")
            ]

            best_base = base.sort_values("macro_f1", ascending=False).head(1).to_dict(orient="records")
            best_proto = proto.sort_values("macro_f1", ascending=False).head(1).to_dict(orient="records")

            b = best_base[0]["macro_f1"] if best_base else np.nan
            p = best_proto[0]["macro_f1"] if best_proto else np.nan

            comparison[target] = {
                "best_sr1c_tabular": best_base[0] if best_base else None,
                "best_sr2a_prototype": best_proto[0] if best_proto else None,
                "delta_prototype_minus_tabular_macro_f1": (
                    float(p - b) if pd.notna(p) and pd.notna(b) else None
                ),
            }

    clean_res = results.dropna(subset=["macro_f1"]).copy()

    best_by_target_split = (
        clean_res
        .sort_values(["target", "split", "macro_f1", "accuracy"], ascending=[True, True, False, False])
        .groupby(["target", "split"], as_index=False)
        .head(1)
        if not clean_res.empty else pd.DataFrame()
    )

    # Verdict.
    gain_delta = comparison.get("gain_oracle_action", {}).get("delta_prototype_minus_tabular_macro_f1", None)

    if gain_delta is not None and gain_delta >= 0.10:
        verdict = "PASS_STRONG_PROTOTYPE_GEOMETRY_IMPROVES_CROSS_REUSE"
    elif gain_delta is not None and gain_delta >= 0.03:
        verdict = "PASS_LITE_PROTOTYPE_GEOMETRY_IMPROVES_CROSS_REUSE"
    elif gain_delta is not None and gain_delta > -0.03:
        verdict = "MIXED_PROTOTYPE_PARITY_WITH_TABULAR"
    else:
        verdict = "FAIL_PROTOTYPE_GEOMETRY_DOES_NOT_SOLVE_CROSS_REUSE"

    summary = {
        "input_features": str(IN_FEATURES),
        "n_rows": int(len(df)),
        "feature_group_counts": {k: len(v) for k, v in groups.items()},
        "targets": TARGETS,
        "feature_sets": FEATURE_SETS,
        "prototype_models": PROTOTYPE_MODELS,
        "splits": SPLITS,
        "pca_components": PCA_COMPONENTS,
        "best_by_target_split": best_by_target_split.to_dict(orient="records"),
        "sr1c_tabular_comparison": comparison,
        "verdict": verdict,
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[SAVED]")
    print(f"  cv results : {OUT_CV}")
    print(f"  predictions: {OUT_PREDS}")
    print(f"  utility    : {OUT_UTILITY}")
    print(f"  summary    : {OUT_SUMMARY}")

    print("\n[VERDICT]")
    print(verdict)


if __name__ == "__main__":
    main()
