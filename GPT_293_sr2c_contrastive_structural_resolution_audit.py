# -*- coding: utf-8 -*-
"""
SR-2C: Contrastive Structural Resolution Audit

Purpose
-------
Correct the object definition:

    StructuralResolution is NOT an isolated property SR(x).
    StructuralResolution is a contrastive property SR(x | reference)
    or SR(target, source_boundary).

This script converts the SR problem into a contrastive ranking task.

For each target task x, construct candidate references:

    direct_reuse
    observe
    reject

For each candidate action a, compute contrastive geometry:

    phi(x, prototype_a)
    phi(x, prototype_a) - phi(x, prototype_other)
    margins among candidate distances
    relative closeness to direct / observe / reject reference regions

Then train a candidate scorer on train folds and select, for each task,
the candidate action with highest score.

This is different from SR-2A:
    SR-2A: classify x by distance to prototypes.
    SR-2C: rank candidate actions by target-reference contrast.

Inputs
------
    sr1b1_outputs/sr1b1_topk_vim_features.csv
    sr1c_outputs/sr1c_cv_results.csv       optional baseline comparison

Outputs
-------
    sr2c_outputs/sr2c_cv_results.csv
    sr2c_outputs/sr2c_predictions.csv
    sr2c_outputs/sr2c_candidate_scores.csv
    sr2c_outputs/sr2c_policy_utility.csv
    sr2c_outputs/sr2c_summary.json

Key split
---------
    group_by_reuse_group
    group_by_reuse_group_non_source

Theory
------
If SR is truly contrastive, this should beat:
    isolated TopK/VIM
    direct/observe/reject prototype classification
    tabular boundary baseline
especially in non-source heldout settings.
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


BASE_DIR = Path(r"C:\Users\ZH\Desktop\AGI\python_script")

IN_FEATURES = BASE_DIR / "sr1b1_outputs" / "sr1b1_topk_vim_features.csv"
IN_SR1C = BASE_DIR / "sr1c_outputs" / "sr1c_cv_results.csv"

OUT_DIR = BASE_DIR / "sr2c_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CV = OUT_DIR / "sr2c_cv_results.csv"
OUT_PREDS = OUT_DIR / "sr2c_predictions.csv"
OUT_CANDIDATES = OUT_DIR / "sr2c_candidate_scores.csv"
OUT_UTILITY = OUT_DIR / "sr2c_policy_utility.csv"
OUT_SUMMARY = OUT_DIR / "sr2c_summary.json"


ACTIONS = ["direct_reuse", "observe", "reject"]

TARGETS = [
    "gain_oracle_action",
    "best_policy_action",
]

SPLITS = [
    "stratified5",
    "group_by_concept",
    "group_by_task_family",
    "group_by_reuse_group",
    "group_by_reuse_group_non_source",
]

FEATURE_SETS = [
    "topk_init",
    "topk_mid",
    "topk_boundary",
    "topk_init_mid",
    "topk_mid_boundary",
    "topk_all",
]

SCORERS = [
    "contrast_logistic",
    "contrast_rf",
    "contrast_extra_trees",
]

PCA_COMPONENTS = 32
SEED = 42
SOURCE_GROUP = "G0_STRICT_SOURCE"


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

    if split_name == "group_by_reuse_group_non_source":
        splits = []
        all_idx = np.arange(len(df))
        for group in sorted(df["reuse_group"].astype(str).unique()):
            if group == SOURCE_GROUP:
                continue
            te = np.where(df["reuse_group"].astype(str).values == group)[0]
            tr = np.setdiff1d(all_idx, te)
            if len(te) > 0 and len(tr) > 0:
                splits.append((tr, te))
        return splits

    raise ValueError(f"Unknown split_name: {split_name}")


def cosine_to_centroid(Z: np.ndarray, c: np.ndarray) -> np.ndarray:
    c = c.reshape(1, -1)
    denom = np.maximum(np.linalg.norm(Z, axis=1) * np.linalg.norm(c), 1e-12)
    return (Z @ c.T).reshape(-1) / denom


class ContrastiveReferenceGeometry:
    """
    Train-fold local contrastive geometry.

    It builds action prototypes from train fold, while always allowing
    frozen source anchors from G0_STRICT_SOURCE/direct_reuse as a known
    source operator memory.

    For each sample x and candidate action a, it returns features describing:
        - distance to candidate reference
        - cosine to candidate reference
        - distance margin vs other references
        - source-anchor closeness
        - candidate rank among references
    """

    def __init__(self, n_components=32, seed=42):
        self.n_components = n_components
        self.seed = seed
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.pca = None
        self.action_centroids_ = {}
        self.source_centroid_ = None
        self.global_centroid_ = None
        self.available_actions_ = []

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, meta_train: pd.DataFrame, X_source_anchor: pd.DataFrame):
        X_fit = pd.concat([X_train, X_source_anchor], axis=0)
        X0 = self.imputer.fit_transform(X_fit)
        X1 = self.scaler.fit_transform(X0)

        n_comp = min(self.n_components, X1.shape[0] - 1, X1.shape[1])
        if n_comp >= 2:
            self.pca = PCA(n_components=n_comp, random_state=self.seed)
            self.pca.fit(X1)
        else:
            self.pca = None

        Z_train = self._base_transform(X_train)
        Z_source = self._base_transform(X_source_anchor)

        self.source_centroid_ = Z_source.mean(axis=0)
        self.global_centroid_ = Z_train.mean(axis=0)

        y = pd.Series(y_train).astype(str).values

        self.action_centroids_ = {}
        self.available_actions_ = []

        for action in ACTIONS:
            mask = y == action
            if mask.sum() > 0:
                self.action_centroids_[action] = Z_train[mask].mean(axis=0)
                self.available_actions_.append(action)
            else:
                # Fallback keeps feature dimensionality stable.
                self.action_centroids_[action] = self.global_centroid_.copy()

        # Override / stabilize direct_reuse with frozen source memory when possible.
        # This encodes the actual self-reflexive setting: direct reuse is anchored
        # by known source operator memory, not only by current train fold.
        self.action_centroids_["direct_reuse"] = (
            0.5 * self.action_centroids_["direct_reuse"] + 0.5 * self.source_centroid_
        )

        return self

    def _base_transform(self, X: pd.DataFrame) -> np.ndarray:
        X0 = self.imputer.transform(X)
        X1 = self.scaler.transform(X0)
        if self.pca is not None:
            return self.pca.transform(X1)
        return X1

    def candidate_features(self, X: pd.DataFrame, task_ids: List[str]) -> pd.DataFrame:
        Z = self._base_transform(X)

        centroids = np.vstack([self.action_centroids_[a] for a in ACTIONS])
        d = np.linalg.norm(Z[:, None, :] - centroids[None, :, :], axis=2)

        cos = []
        for a in ACTIONS:
            cos.append(cosine_to_centroid(Z, self.action_centroids_[a]))
        cos = np.vstack(cos).T

        source_d = np.linalg.norm(Z - self.source_centroid_.reshape(1, -1), axis=1)
        source_cos = cosine_to_centroid(Z, self.source_centroid_)
        global_d = np.linalg.norm(Z - self.global_centroid_.reshape(1, -1), axis=1)

        rows = []

        for i, tid in enumerate(task_ids):
            dist_map = {a: d[i, j] for j, a in enumerate(ACTIONS)}
            cos_map = {a: cos[i, j] for j, a in enumerate(ACTIONS)}

            sorted_actions = sorted(ACTIONS, key=lambda a: dist_map[a])
            rank_map = {a: r + 1 for r, a in enumerate(sorted_actions)}

            min_other = {}
            mean_other = {}
            for a in ACTIONS:
                others = [o for o in ACTIONS if o != a]
                min_other[a] = min(dist_map[o] for o in others)
                mean_other[a] = float(np.mean([dist_map[o] for o in others]))

            for a in ACTIONS:
                others = [o for o in ACTIONS if o != a]
                row = {
                    "task_id": tid,
                    "candidate_action": a,

                    "d_candidate": dist_map[a],
                    "cos_candidate": cos_map[a],
                    "candidate_rank_by_dist": rank_map[a],

                    "d_margin_min_other_minus_candidate": min_other[a] - dist_map[a],
                    "d_margin_mean_other_minus_candidate": mean_other[a] - dist_map[a],

                    "source_d": source_d[i],
                    "source_cos": source_cos[i],
                    "global_d": global_d[i],
                    "source_minus_candidate_d": source_d[i] - dist_map[a],
                    "candidate_minus_source_d": dist_map[a] - source_d[i],

                    "is_direct_candidate": float(a == "direct_reuse"),
                    "is_observe_candidate": float(a == "observe"),
                    "is_reject_candidate": float(a == "reject"),
                }

                # Pairwise signed margins against named alternatives.
                for o in others:
                    row[f"d_{o}_minus_{a}"] = dist_map[o] - dist_map[a]
                    row[f"cos_{a}_minus_{o}"] = cos_map[a] - cos_map[o]

                # Especially important: direct vs reject contrast.
                row["direct_minus_reject_d_margin"] = dist_map["reject"] - dist_map["direct_reuse"]
                row["observe_minus_reject_d_margin"] = dist_map["reject"] - dist_map["observe"]
                row["direct_minus_observe_d_margin"] = dist_map["observe"] - dist_map["direct_reuse"]

                rows.append(row)

        return pd.DataFrame(rows)

    def nearest_action(self, X: pd.DataFrame) -> np.ndarray:
        Z = self._base_transform(X)
        centroids = np.vstack([self.action_centroids_[a] for a in ACTIONS])
        d = np.linalg.norm(Z[:, None, :] - centroids[None, :, :], axis=2)
        return np.array(ACTIONS)[np.argmin(d, axis=1)]


def build_scorer(name: str):
    if name == "contrast_logistic":
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

    if name == "contrast_rf":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=400,
                max_depth=6,
                min_samples_leaf=3,
                random_state=SEED,
                class_weight="balanced_subsample",
                n_jobs=-1,
            )),
        ])

    if name == "contrast_extra_trees":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", ExtraTreesClassifier(
                n_estimators=500,
                max_depth=6,
                min_samples_leaf=2,
                random_state=SEED,
                class_weight="balanced",
                n_jobs=-1,
            )),
        ])

    raise ValueError(f"Unknown scorer: {name}")


def select_action_from_scores(cand: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    idx = cand.groupby("task_id")[score_col].idxmax()
    out = cand.loc[idx, ["task_id", "candidate_action", score_col]].copy()
    out = out.rename(columns={"candidate_action": "y_pred", score_col: "selected_score"})
    return out


def action_utility(row, action_value):
    action = str(action_value)
    if action == "direct_reuse":
        return row.get("correct_vs_no", np.nan)
    if action in ["observe", "reject", "no_reuse"]:
        return 0.0
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

    for target_name in TARGETS:
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

    return pd.DataFrame(rows)


def run_contrastive_cv(df: pd.DataFrame, target: str, feature_set: str, scorer_name: str, split_name: str, feature_cols: List[str]):
    splits = get_splits(df, target, split_name)
    if not splits:
        return None, None, None

    y = df[target].astype(str).values
    pred = np.empty(len(df), dtype=object)
    pred[:] = None

    candidate_parts = []

    Xall = df[feature_cols].copy()
    for c in feature_cols:
        Xall[c] = pd.to_numeric(Xall[c], errors="coerce")
    Xall = Xall.replace([np.inf, -np.inf], np.nan)

    source_mask = (
        (df["reuse_group"].astype(str) == SOURCE_GROUP)
        & (df["gain_oracle_action"].astype(str) == "direct_reuse")
    )
    source_idx_global = np.where(source_mask.values)[0]

    if len(source_idx_global) < 3:
        raise ValueError(f"Not enough source anchors: {len(source_idx_global)}")

    for fold, (tr, te) in enumerate(splits):
        X_source = Xall.iloc[source_idx_global]

        enc = ContrastiveReferenceGeometry(n_components=PCA_COMPONENTS, seed=SEED)
        enc.fit(
            X_train=Xall.iloc[tr],
            y_train=y[tr],
            meta_train=df.iloc[tr],
            X_source_anchor=X_source,
        )

        # Training candidate rows.
        train_tasks = df.iloc[tr]["task_id"].astype(str).tolist()
        C_tr = enc.candidate_features(Xall.iloc[tr], train_tasks)

        true_map = dict(zip(df.iloc[tr]["task_id"].astype(str), y[tr]))
        C_tr["true_action"] = C_tr["task_id"].map(true_map).astype(str)
        C_tr["label"] = (C_tr["candidate_action"] == C_tr["true_action"]).astype(int)

        # Test candidate rows.
        test_tasks = df.iloc[te]["task_id"].astype(str).tolist()
        C_te = enc.candidate_features(Xall.iloc[te], test_tasks)

        if scorer_name == "nearest_contrast":
            # Score direct by closeness margin, observe by middle/uncertainty, reject by reject margin.
            # This is a deterministic non-learned baseline.
            C_te["score"] = 0.0
            C_te.loc[C_te["candidate_action"] == "direct_reuse", "score"] = (
                C_te["direct_minus_reject_d_margin"]
                + 0.5 * C_te["source_vs_reject_margin"]
                - 0.1 * C_te["d_candidate"]
            )
            C_te.loc[C_te["candidate_action"] == "observe", "score"] = (
                -np.abs(C_te["direct_minus_reject_d_margin"])
                + 0.2 * C_te["source_specificity"] if "source_specificity" in C_te.columns else -np.abs(C_te["direct_minus_reject_d_margin"])
            )
            C_te.loc[C_te["candidate_action"] == "reject", "score"] = (
                -C_te["direct_minus_reject_d_margin"]
                + 0.1 * C_te["d_candidate"]
            )
        else:
            Xc_tr = C_tr.drop(columns=["task_id", "candidate_action", "true_action", "label"], errors="ignore")
            yc_tr = C_tr["label"].astype(int).values

            Xc_te = C_te.drop(columns=["task_id", "candidate_action"], errors="ignore")

            scorer = build_scorer(scorer_name)
            scorer.fit(Xc_tr, yc_tr)

            if hasattr(scorer, "predict_proba"):
                C_te["score"] = scorer.predict_proba(Xc_te)[:, 1]
            else:
                # Pipeline always has clf, but keep fallback.
                if hasattr(scorer.named_steps.get("clf", None), "decision_function"):
                    C_te["score"] = scorer.decision_function(Xc_te)
                else:
                    C_te["score"] = scorer.predict(Xc_te)

        chosen = select_action_from_scores(C_te, "score")
        pred_map = dict(zip(chosen["task_id"].astype(str), chosen["y_pred"].astype(str)))
        pred[te] = [pred_map[str(tid)] for tid in df.iloc[te]["task_id"].astype(str)]

        C_te["fold"] = fold
        C_te["target"] = target
        C_te["feature_set"] = feature_set
        C_te["model"] = scorer_name
        C_te["split"] = split_name
        C_te["y_true"] = C_te["task_id"].map(dict(zip(df["task_id"].astype(str), y)))
        C_te["is_true_candidate"] = (C_te["candidate_action"] == C_te["y_true"]).astype(int)
        candidate_parts.append(C_te)

    mask = pd.Series(pred).notna().values
    metrics = {
        "target": target,
        "feature_set": feature_set,
        "model": scorer_name,
        "split": split_name,
        "n": int(mask.sum()),
        "accuracy": float(accuracy_score(y[mask], pred[mask])),
        "macro_f1": float(f1_score(y[mask], pred[mask], average="macro")),
    }

    pred_df = pd.DataFrame({
        "task_id": df["task_id"].astype(str),
        "target": target,
        "feature_set": feature_set,
        "model": scorer_name,
        "split": split_name,
        "y_true": y,
        "y_pred": pred,
        "is_correct": (y == pred).astype(int),
    })

    cand_df = pd.concat(candidate_parts, ignore_index=True) if candidate_parts else pd.DataFrame()

    return metrics, pred_df, cand_df


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["reuse_group", "task_family", "target_operator", "concept"]:
        if c not in df.columns:
            raise ValueError(f"Missing meta column: {c}")
        df[c] = df[c].fillna("NA").astype(str)

    for c in ["gain_oracle_action", "best_policy_action"]:
        if c not in df.columns:
            raise ValueError(f"Missing target column: {c}")
        df[c] = df[c].fillna("NA").astype(str)

    topk_cols = [c for c in df.columns if is_topk_feature(c)]
    for c in topk_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[topk_cols] = df[topk_cols].replace([np.inf, -np.inf], np.nan)

    return df


def main():
    df = read_csv(IN_FEATURES)
    df = prepare_df(df)

    groups = topk_feature_groups(df)

    print("[FEATURE GROUPS]")
    for k, v in groups.items():
        print(f"  {k}: {len(v)}")

    source_mask = (
        (df["reuse_group"].astype(str) == SOURCE_GROUP)
        & (df["gain_oracle_action"].astype(str) == "direct_reuse")
    )
    print(f"[SOURCE] {SOURCE_GROUP} direct anchors = {int(source_mask.sum())}")

    # Add deterministic baseline as a scorer.
    scorers = ["nearest_contrast"] + SCORERS

    result_rows = []
    pred_parts = []
    cand_parts = []

    for target in TARGETS:
        for feature_set in FEATURE_SETS:
            cols = groups.get(feature_set, [])
            if not cols:
                continue

            for scorer in scorers:
                for split in SPLITS:
                    try:
                        metrics, pred_df, cand_df = run_contrastive_cv(df, target, feature_set, scorer, split, cols)
                        if metrics is None:
                            continue

                        result_rows.append(metrics)
                        pred_parts.append(pred_df)
                        if cand_df is not None and not cand_df.empty:
                            cand_parts.append(cand_df)

                        print(
                            f"[CV] target={target:18s} fs={feature_set:18s} "
                            f"model={scorer:24s} split={split:30s} "
                            f"acc={metrics['accuracy']:.3f} f1={metrics['macro_f1']:.3f}"
                        )

                    except Exception as e:
                        result_rows.append({
                            "target": target,
                            "feature_set": feature_set,
                            "model": scorer,
                            "split": split,
                            "n": 0,
                            "accuracy": np.nan,
                            "macro_f1": np.nan,
                            "error": str(e),
                        })
                        print(f"[WARN] Failed {target} {feature_set} {scorer} {split}: {e}")

    results = pd.DataFrame(result_rows)
    preds = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    candidates = pd.concat(cand_parts, ignore_index=True) if cand_parts else pd.DataFrame()
    utility = summarize_policy_utility(df, preds) if not preds.empty else pd.DataFrame()

    results.to_csv(OUT_CV, index=False, encoding="utf-8-sig")
    preds.to_csv(OUT_PREDS, index=False, encoding="utf-8-sig")
    candidates.to_csv(OUT_CANDIDATES, index=False, encoding="utf-8-sig")
    utility.to_csv(OUT_UTILITY, index=False, encoding="utf-8-sig")

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
        res_clean = results.dropna(subset=["macro_f1"]).copy()

        for target in TARGETS:
            comparison[target] = {}
            for split in ["group_by_reuse_group", "group_by_reuse_group_non_source"]:
                base = clean[
                    (clean["target"] == target)
                    & (clean["split"] == "group_by_reuse_group")
                    & (clean["feature_set"].isin(tabular_sets))
                ]
                con = res_clean[
                    (res_clean["target"] == target)
                    & (res_clean["split"] == split)
                ]

                best_base = base.sort_values("macro_f1", ascending=False).head(1).to_dict(orient="records")
                best_con = con.sort_values("macro_f1", ascending=False).head(1).to_dict(orient="records")

                b = best_base[0]["macro_f1"] if best_base else np.nan
                c = best_con[0]["macro_f1"] if best_con else np.nan

                comparison[target][split] = {
                    "best_sr1c_tabular_group_by_reuse_group": best_base[0] if best_base else None,
                    "best_sr2c_contrastive": best_con[0] if best_con else None,
                    "delta_contrastive_minus_tabular_macro_f1": (
                        float(c - b) if pd.notna(c) and pd.notna(b) else None
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

    gain_delta_non_source = (
        comparison
        .get("gain_oracle_action", {})
        .get("group_by_reuse_group_non_source", {})
        .get("delta_contrastive_minus_tabular_macro_f1", None)
    )

    gain_delta_strict = (
        comparison
        .get("gain_oracle_action", {})
        .get("group_by_reuse_group", {})
        .get("delta_contrastive_minus_tabular_macro_f1", None)
    )

    if gain_delta_non_source is not None and gain_delta_non_source >= 0.10:
        verdict = "PASS_STRONG_CONTRASTIVE_SR_IMPROVES_NON_SOURCE_CROSS_REUSE"
    elif gain_delta_non_source is not None and gain_delta_non_source >= 0.03:
        verdict = "PASS_LITE_CONTRASTIVE_SR_IMPROVES_NON_SOURCE_CROSS_REUSE"
    elif gain_delta_strict is not None and gain_delta_strict >= 0.03:
        verdict = "MIXED_CONTRASTIVE_SR_ONLY_STRICT_SPLIT_SIGNAL"
    elif gain_delta_non_source is not None and gain_delta_non_source > -0.03:
        verdict = "MIXED_CONTRASTIVE_SR_PARITY_WITH_TABULAR"
    else:
        verdict = "FAIL_CONTRASTIVE_SR_NOT_ENOUGH_YET"

    summary = {
        "input_features": str(IN_FEATURES),
        "n_rows": int(len(df)),
        "source_group": SOURCE_GROUP,
        "n_source_direct_anchors": int(source_mask.sum()),
        "feature_group_counts": {k: len(v) for k, v in groups.items()},
        "targets": TARGETS,
        "feature_sets": FEATURE_SETS,
        "scorers": scorers,
        "splits": SPLITS,
        "pca_components": PCA_COMPONENTS,
        "best_by_target_split": best_by_target_split.to_dict(orient="records"),
        "sr1c_tabular_comparison": comparison,
        "verdict": verdict,
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[SAVED]")
    print(f"  cv results      : {OUT_CV}")
    print(f"  predictions     : {OUT_PREDS}")
    print(f"  candidate scores: {OUT_CANDIDATES}")
    print(f"  utility         : {OUT_UTILITY}")
    print(f"  summary         : {OUT_SUMMARY}")

    print("\n[VERDICT]")
    print(verdict)


if __name__ == "__main__":
    main()
