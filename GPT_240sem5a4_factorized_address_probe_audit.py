# -*- coding: utf-8 -*-
"""
SEM-5A.4: Factorized Address Probe Audit
========================================

Motivation
----------
SEM-5A.3 showed that Shape+Commit fusion improves memory addressing, but
same_operator / concept disambiguation remains weak.

SEM-5A.4 asks:
  Where is concept identity readable?
  Where is operator / trajectory-shape readable?
  Can probe-guided two-stage retrieval close the gap?

Core tests
----------
1. Concept probe:
   Predict concept_star from Init / Shape / Commit features.

2. Operator probe:
   Predict operator_star from Init / Shape / Commit features.

3. Retrieval baselines:
   ShapeQuery
   CommitQuery
   ShapeCommit best late fusion

4. Oracle upper bounds:
   TrueConcept -> Shape rank within concept
   TrueOperator -> Commit rank within operator

5. Probe-guided two-stage retrieval:
   PredConcept -> Shape rank within predicted concept
   PredOperator -> Commit rank within predicted operator
   Joint score from predicted concept/operator gates

Default inputs
--------------
  sem5a2_outputs/sem5a2_dataset.csv
  sem5a2_outputs/sem5a2_features.csv

Outputs
-------
  sem5a4_outputs/sem5a4_probe_by_split.csv
  sem5a4_outputs/sem5a4_probe_summary.csv
  sem5a4_outputs/sem5a4_retrieval_by_split.csv
  sem5a4_outputs/sem5a4_retrieval_summary.csv
  sem5a4_outputs/sem5a4_factorized_diagnosis.csv
  sem5a4_outputs/sem5a4_verdict.json
  sem5a4_outputs/sem5a4_column_audit.json

Run
---
  python sem5a4_factorized_address_probe_audit.py
"""

from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# =============================================================================
# Config
# =============================================================================

DATASET_PATH = Path("sem5a2_outputs/sem5a2_dataset.csv")
FEATURES_PATH = Path("sem5a2_outputs/sem5a2_features.csv")
OUTPUT_DIR = Path("sem5a4_outputs")

RANDOM_SEED = 42
TOP_KS = (1, 3, 5)
SOFTMAX_TEMPERATURE = 0.10

# Late fusion weights for Shape + Commit.
FUSION_WEIGHTS = [
    (0.50, 0.50),
    (0.60, 0.40),
    (0.70, 0.30),
    (0.40, 0.60),
    (0.30, 0.70),
]

# Candidate gate bonus for predicted concept/operator two-stage retrieval.
GATE_BONUS = 2.0


# =============================================================================
# Utilities
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def layer_number_from_col(col: str) -> Optional[int]:
    s = normalize_name(col)
    patterns = [
        r"(?:^|_)l(?:ayer)?_?0?(\d{1,2})(?:_|$)",
        r"(?:^|_)layer_?0?(\d{1,2})(?:_|$)",
        r"(?:^|_)block_?0?(\d{1,2})(?:_|$)",
        r"(?:^|_)h_?0?(\d{1,2})(?:_|$)",
    ]
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            val = int(m.group(1))
            if 0 <= val <= 80:
                return val

    m = re.search(r"(?:^|_)l0?(\d{1,2})[a-z_]", s)
    if m:
        val = int(m.group(1))
        if 0 <= val <= 80:
            return val

    return None


def select_layer_cols(
    df: pd.DataFrame,
    start: int,
    end: int,
    prefer_keywords: Iterable[str] = (),
    avoid_keywords: Iterable[str] = (),
) -> List[str]:
    cols = []
    prefer = [normalize_name(k) for k in prefer_keywords]
    avoid = [normalize_name(k) for k in avoid_keywords]

    for c in numeric_cols(df):
        lnum = layer_number_from_col(c)
        if lnum is None or not (start <= lnum <= end):
            continue
        name = normalize_name(c)
        if avoid and any(k in name for k in avoid):
            continue
        cols.append(c)

    if prefer:
        preferred = [c for c in cols if any(k in normalize_name(c) for k in prefer)]
        if len(preferred) >= 2:
            return preferred

    return cols


def zscore_train_test(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(x_train, axis=0, keepdims=True)
    sd = np.nanstd(x_train, axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    x_train = (x_train - mu) / sd
    x_test = (x_test - mu) / sd
    x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
    x_test = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0)
    return x_train, x_test


def row_softmax_entropy(sim: np.ndarray, temperature: float = SOFTMAX_TEMPERATURE) -> np.ndarray:
    z = sim / max(temperature, 1e-8)
    z = z - np.max(z, axis=1, keepdims=True)
    p = np.exp(z)
    p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
    h = -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1)
    if sim.shape[1] > 1:
        h = h / math.log(sim.shape[1])
    return h


def recall_at_k(y_true: List[str], ranked_labels: np.ndarray, k: int) -> float:
    hits = 0
    for truth, row in zip(y_true, ranked_labels):
        if truth in row[:k]:
            hits += 1
    return hits / max(1, len(y_true))


def mrr_score(y_true: List[str], ranked_labels: np.ndarray) -> float:
    vals = []
    for truth, row in zip(y_true, ranked_labels):
        pos = np.where(row == truth)[0]
        vals.append(1.0 / (pos[0] + 1) if len(pos) else 0.0)
    return float(np.mean(vals)) if vals else float("nan")


def safe_f1(y_true, y_pred, average="macro") -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def safe_acc(y_true, y_pred) -> float:
    return float(accuracy_score(y_true, y_pred))


# =============================================================================
# Data loading and feature blocks
# =============================================================================

@dataclass
class Block:
    name: str
    cols: List[str]


def merge_dataset_features(dataset: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if "row_id" in dataset.columns and "row_id" in features.columns:
        merged = dataset.merge(features, on="row_id", how="inner", suffixes=("", "_feat"))
    else:
        if len(dataset) != len(features):
            raise ValueError("Dataset/features row count differs and no row_id shared.")
        merged = pd.concat([dataset.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
        merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged


def build_blocks(df: pd.DataFrame) -> Dict[str, Block]:
    init_cols = select_layer_cols(
        df, 0, 6,
        prefer_keywords=["center", "pca", "topk", "init"],
        avoid_keywords=["commit"],
    )
    shape_cols = select_layer_cols(
        df, 7, 19,
        prefer_keywords=["center", "pca", "shape"],
        avoid_keywords=["commit"],
    )
    commit_cols = select_layer_cols(
        df, 23, 25,
        prefer_keywords=["center", "pca", "commit"],
        avoid_keywords=[],
    )

    if not shape_cols or not commit_cols:
        raise RuntimeError("Missing shape or commit layer columns.")

    blocks = {
        "Init": Block("Init_L0_6", init_cols),
        "Shape": Block("Shape_L7_19", shape_cols),
        "Commit": Block("Commit_L23_25", commit_cols),
        "ShapeCommitConcat": Block("ShapeCommit_Concat", list(shape_cols) + list(commit_cols)),
    }
    return blocks


def make_splits(df: pd.DataFrame, n_splits: int = 4) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    y = df["seed_family"].astype(str).values
    groups = df["surface_id"].astype(str).values
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    return [(f"leave_surface_group_{i}", tr, te) for i, (tr, te) in enumerate(gkf.split(df, y, groups))]


# =============================================================================
# Probe
# =============================================================================

def make_probe_model():
    """
    Use LogisticRegression first. If sklearn version has issues, fallback is handled.
    """
    try:
        clf = LogisticRegression(
            max_iter=3000,
            solver="lbfgs",
            random_state=RANDOM_SEED,
        )
        return make_pipeline(StandardScaler(), clf)
    except Exception:
        return make_pipeline(StandardScaler(), LinearSVC(random_state=RANDOM_SEED))


def fit_predict_probe(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: List[str],
    label_col: str,
) -> Tuple[np.ndarray, Dict]:
    x_train = train[cols].to_numpy(dtype=float)
    x_test = test[cols].to_numpy(dtype=float)
    y_train = train[label_col].astype(str).values
    y_test = test[label_col].astype(str).values

    # Robust model fallback.
    try:
        model = make_probe_model()
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
    except Exception:
        # Nearest-centroid fallback.
        x_train_z, x_test_z = zscore_train_test(x_train, x_test)
        labels = sorted(pd.Series(y_train).unique().tolist())
        centers = []
        for lab in labels:
            centers.append(x_train_z[y_train == lab].mean(axis=0))
        centers = np.vstack(centers)
        sim = cosine_similarity(x_test_z, centers)
        pred = np.array(labels, dtype=object)[np.argmax(sim, axis=1)]

    metrics = {
        "label": label_col,
        "acc": safe_acc(y_test, pred),
        "macro_f1": safe_f1(y_test, pred, "macro"),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_classes_train": int(pd.Series(y_train).nunique()),
    }
    return pred.astype(str), metrics


# =============================================================================
# Retrieval
# =============================================================================

def build_centers(
    x_train: np.ndarray,
    y_train: np.ndarray,
    meta_train: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    families = sorted(pd.Series(y_train).unique().astype(str).tolist())
    centers = []
    meta_rows = []
    for fam in families:
        mask = y_train == fam
        centers.append(x_train[mask].mean(axis=0))
        sub = meta_train.loc[mask]
        meta_rows.append({
            "family": fam,
            "concept": str(sub["concept_star"].mode().iloc[0]),
            "operator": str(sub["operator_star"].mode().iloc[0]),
        })
    return np.vstack(centers), np.array(families, dtype=object), pd.DataFrame(meta_rows)


def numeric_similarity(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    y_train = train["seed_family"].astype(str).values
    x_train = train[cols].to_numpy(dtype=float)
    x_test = test[cols].to_numpy(dtype=float)
    x_train, x_test = zscore_train_test(x_train, x_test)
    centers, families, center_meta = build_centers(x_train, y_train, train)
    sim = cosine_similarity(x_test, centers)
    return sim, families, center_meta


def restrict_and_rank(
    sim: np.ndarray,
    families: np.ndarray,
    center_meta: pd.DataFrame,
    test: pd.DataFrame,
    retrieval_mode: str,
    pred_concept: Optional[np.ndarray] = None,
    pred_operator: Optional[np.ndarray] = None,
    gate_mode: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[bool], List[int]]:
    """
    retrieval_mode:
      full
      same_concept
      same_operator
      same_concept_different_operator
      same_operator_different_concept

    gate_mode:
      none
      pred_concept_gate
      pred_operator_gate
      pred_both_gate
    """
    score = sim.copy()

    # Add predicted gates before hard restriction.
    if gate_mode in {"pred_concept_gate", "pred_both_gate"} and pred_concept is not None:
        for i, pc in enumerate(pred_concept):
            score[i, center_meta["concept"].astype(str).values == str(pc)] += GATE_BONUS

    if gate_mode in {"pred_operator_gate", "pred_both_gate"} and pred_operator is not None:
        for i, po in enumerate(pred_operator):
            score[i, center_meta["operator"].astype(str).values == str(po)] += GATE_BONUS

    masked = score.copy()
    valid_rows = []
    candidate_counts = []

    for i in range(score.shape[0]):
        mask = np.ones(score.shape[1], dtype=bool)
        truth = str(test.iloc[i]["seed_family"])
        cval = str(test.iloc[i]["concept_star"])
        oval = str(test.iloc[i]["operator_star"])

        if retrieval_mode in {"same_concept", "same_concept_different_operator"}:
            mask &= (center_meta["concept"].astype(str).values == cval)

        if retrieval_mode in {"same_operator", "same_operator_different_concept"}:
            mask &= (center_meta["operator"].astype(str).values == oval)

        if retrieval_mode == "same_concept_different_operator":
            mask &= ((center_meta["operator"].astype(str).values != oval) |
                     (center_meta["family"].astype(str).values == truth))

        if retrieval_mode == "same_operator_different_concept":
            mask &= ((center_meta["concept"].astype(str).values != cval) |
                     (center_meta["family"].astype(str).values == truth))

        candidate_counts.append(int(mask.sum()))
        if mask.sum() <= 0:
            masked[i, :] = -np.inf
            valid_rows.append(False)
        else:
            masked[i, ~mask] = -np.inf
            valid_rows.append(True)

    order = np.argsort(-masked, axis=1)
    ranked = families[order]
    return ranked, masked, valid_rows, candidate_counts


def score_retrieval(
    query_name: str,
    query_kind: str,
    sim: np.ndarray,
    families: np.ndarray,
    center_meta: pd.DataFrame,
    test: pd.DataFrame,
    retrieval_mode: str,
    pred_concept: Optional[np.ndarray] = None,
    pred_operator: Optional[np.ndarray] = None,
    gate_mode: Optional[str] = None,
) -> Tuple[Dict, pd.DataFrame]:
    ranked, masked, valid_rows, candidate_counts = restrict_and_rank(
        sim,
        families,
        center_meta,
        test,
        retrieval_mode,
        pred_concept=pred_concept,
        pred_operator=pred_operator,
        gate_mode=gate_mode,
    )
    valid = np.array(valid_rows, dtype=bool)

    y = test["seed_family"].astype(str).values[valid]
    ranked_eval = ranked[valid]
    masked_eval = masked[valid]
    candidate_counts_eval = np.array(candidate_counts)[valid]

    if len(y) == 0:
        raise ValueError("No valid retrieval rows.")

    order = np.argsort(-masked_eval, axis=1)
    pred = ranked_eval[:, 0].astype(str)
    ent = row_softmax_entropy(masked_eval)

    metrics = {
        "query_type": query_name,
        "query_kind": query_kind,
        "retrieval_mode": retrieval_mode,
        "gate_mode": gate_mode or "none",
        "n_test_eval": int(len(y)),
        "candidate_count_mean": float(np.mean(candidate_counts_eval)),
        "acc_top1": safe_acc(y, pred),
        "macro_f1": safe_f1(y, pred, "macro"),
        "mrr": mrr_score(list(y), ranked_eval),
        "entropy_mean": float(np.mean(ent)),
    }
    for k in TOP_KS:
        metrics[f"recall_at_{k}"] = recall_at_k(list(y), ranked_eval, min(k, ranked_eval.shape[1]))

    details = pd.DataFrame({
        "query_type": query_name,
        "query_kind": query_kind,
        "retrieval_mode": retrieval_mode,
        "gate_mode": gate_mode or "none",
        "truth_family": y,
        "pred_family": pred,
        "top1_sim": masked_eval[np.arange(len(y)), order[:, 0]],
        "entropy": ent,
        "candidate_count": candidate_counts_eval,
        "concept": test.loc[valid, "concept_star"].astype(str).values,
        "operator": test.loc[valid, "operator_star"].astype(str).values,
        "surface": test.loc[valid, "surface_id"].astype(str).values,
    })
    for k in TOP_KS:
        kk = min(k, ranked_eval.shape[1])
        details[f"hit_at_{k}"] = [truth in set(row[:kk]) for truth, row in zip(y, ranked_eval)]

    return metrics, details


# =============================================================================
# Summary and verdict
# =============================================================================

def summarize_probe(probe_df: pd.DataFrame) -> pd.DataFrame:
    agg = probe_df.groupby(["target", "block"]).agg({
        "acc": ["mean", "std"],
        "macro_f1": ["mean", "std"],
        "n_test": ["sum"],
    })
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index().sort_values(["target", "acc_mean"], ascending=[True, False])


def summarize_retrieval(ret_df: pd.DataFrame) -> pd.DataFrame:
    agg_map = {
        "acc_top1": ["mean", "std"],
        "macro_f1": ["mean", "std"],
        "mrr": ["mean", "std"],
        "entropy_mean": ["mean", "std"],
        "candidate_count_mean": ["mean", "std"],
        "n_test_eval": ["sum"],
    }
    for k in TOP_KS:
        col = f"recall_at_{k}"
        agg_map[col] = ["mean", "std"]

    agg = ret_df.groupby(["retrieval_mode", "query_type", "query_kind", "gate_mode"]).agg(agg_map)
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index().sort_values(["retrieval_mode", "acc_top1_mean"], ascending=[True, False])


def factorized_diagnosis(probe_summary: pd.DataFrame, retrieval_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # Probe best blocks.
    best_concept = probe_summary[probe_summary["target"] == "concept_star"].sort_values("acc_mean", ascending=False).head(1)
    best_operator = probe_summary[probe_summary["target"] == "operator_star"].sort_values("acc_mean", ascending=False).head(1)

    for mode, sub in retrieval_summary.groupby("retrieval_mode"):
        table = sub.set_index("query_type").to_dict(orient="index")

        def get(q, metric="acc_top1_mean"):
            return table.get(q, {}).get(metric, np.nan)

        def best_prefix(prefix):
            keys = [q for q in table if q.startswith(prefix)]
            if not keys:
                return None
            return max(keys, key=lambda q: get(q))

        best_late = best_prefix("LateFusion")
        best_oracle = best_prefix("Oracle")
        best_probe = best_prefix("Probe")
        best_query = max(table.keys(), key=lambda q: get(q))

        row = {
            "retrieval_mode": mode,
            "best_query": best_query,
            "best_acc": get(best_query),
            "best_r3": get(best_query, "recall_at_3_mean"),
            "shape_acc": get("ShapeOnly"),
            "shape_r3": get("ShapeOnly", "recall_at_3_mean"),
            "commit_acc": get("CommitOnly"),
            "commit_r3": get("CommitOnly", "recall_at_3_mean"),
            "concat_acc": get("ShapeCommitConcat"),
            "concat_r3": get("ShapeCommitConcat", "recall_at_3_mean"),
            "best_late_query": best_late,
            "best_late_acc": get(best_late) if best_late else np.nan,
            "best_late_r3": get(best_late, "recall_at_3_mean") if best_late else np.nan,
            "best_oracle_query": best_oracle,
            "best_oracle_acc": get(best_oracle) if best_oracle else np.nan,
            "best_oracle_r3": get(best_oracle, "recall_at_3_mean") if best_oracle else np.nan,
            "best_probe_query": best_probe,
            "best_probe_acc": get(best_probe) if best_probe else np.nan,
            "best_probe_r3": get(best_probe, "recall_at_3_mean") if best_probe else np.nan,
        }
        row["late_minus_shape"] = row["best_late_acc"] - row["shape_acc"]
        row["late_minus_commit"] = row["best_late_acc"] - row["commit_acc"]
        row["probe_minus_late"] = row["best_probe_acc"] - row["best_late_acc"]
        row["oracle_minus_late"] = row["best_oracle_acc"] - row["best_late_acc"]
        rows.append(row)

    diag = pd.DataFrame(rows)

    if not best_concept.empty:
        diag["best_concept_probe_block"] = best_concept.iloc[0]["block"]
        diag["best_concept_probe_acc"] = float(best_concept.iloc[0]["acc_mean"])
    if not best_operator.empty:
        diag["best_operator_probe_block"] = best_operator.iloc[0]["block"]
        diag["best_operator_probe_acc"] = float(best_operator.iloc[0]["acc_mean"])

    return diag


def make_verdict(diag: pd.DataFrame, probe_summary: pd.DataFrame) -> Dict:
    verdict = {
        "status": "UNDETERMINED",
        "pass_lite": False,
        "pass_strong": False,
        "notes": [],
    }
    if diag.empty:
        verdict["status"] = "NO_DIAGNOSIS"
        return verdict

    rows = {r["retrieval_mode"]: r for _, r in diag.iterrows()}

    # Probe facts.
    concept_best = probe_summary[probe_summary["target"] == "concept_star"].sort_values("acc_mean", ascending=False).head(1)
    operator_best = probe_summary[probe_summary["target"] == "operator_star"].sort_values("acc_mean", ascending=False).head(1)

    if not concept_best.empty:
        block = concept_best.iloc[0]["block"]
        acc = float(concept_best.iloc[0]["acc_mean"])
        verdict["notes"].append(f"Best concept probe: {block}, acc={acc:.3f}.")
    if not operator_best.empty:
        block = operator_best.iloc[0]["block"]
        acc = float(operator_best.iloc[0]["acc_mean"])
        verdict["notes"].append(f"Best operator probe: {block}, acc={acc:.3f}.")

    for mode in ["same_concept", "same_operator"]:
        r = rows.get(mode)
        if r is None:
            continue
        if r.get("best_late_acc", 0) > max(r.get("shape_acc", 0), r.get("commit_acc", 0)) + 0.02:
            verdict["notes"].append(f"{mode}: late fusion improves over single branch.")
        if r.get("best_probe_acc", 0) > r.get("best_late_acc", 0) + 0.02:
            verdict["notes"].append(f"{mode}: probe-guided retrieval improves over late fusion.")
        if r.get("best_oracle_acc", 0) > r.get("best_late_acc", 0) + 0.05:
            verdict["notes"].append(f"{mode}: oracle gate shows remaining address headroom.")
        if r.get("best_late_r3", 0) >= 0.90:
            verdict["notes"].append(f"{mode}: late fusion Recall@3 >= 0.90.")

    # PASS-Lite: late/probe/oracle shows factorized gain in either same_concept or same_operator.
    sc = rows.get("same_concept")
    so = rows.get("same_operator")
    sc_lite = sc is not None and (
        sc.get("best_late_acc", 0) > sc.get("shape_acc", 0) + 0.02
        or sc.get("best_probe_acc", 0) > sc.get("best_late_acc", 0) + 0.02
    )
    so_lite = so is not None and (
        so.get("best_late_acc", 0) > max(so.get("shape_acc", 0), so.get("commit_acc", 0)) + 0.02
        or so.get("best_probe_acc", 0) > so.get("best_late_acc", 0) + 0.02
    )
    if sc_lite or so_lite:
        verdict["pass_lite"] = True

    # PASS-Strong: factorized retrieval strong in both modes.
    sc_strong = sc is not None and max(sc.get("best_late_r3", 0), sc.get("best_probe_r3", 0)) >= 0.90 and max(sc.get("best_late_acc", 0), sc.get("best_probe_acc", 0)) >= 0.70
    so_strong = so is not None and max(so.get("best_late_r3", 0), so.get("best_probe_r3", 0)) >= 0.85 and max(so.get("best_late_acc", 0), so.get("best_probe_acc", 0)) >= 0.60
    if sc_strong and so_strong:
        verdict["pass_strong"] = True
        verdict["notes"].append("Factorized addressing reaches strong retrieval in both same-concept and same-operator modes.")

    verdict["status"] = "PASS_STRONG" if verdict["pass_strong"] else ("PASS_LITE" if verdict["pass_lite"] else "NO_PASS_OR_INSUFFICIENT")
    return verdict


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ensure_dir(OUTPUT_DIR)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATASET_PATH}")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing features: {FEATURES_PATH}")

    dataset = pd.read_csv(DATASET_PATH)
    features = pd.read_csv(FEATURES_PATH)
    df = merge_dataset_features(dataset, features)

    blocks = build_blocks(df)
    splits = make_splits(df, n_splits=4)

    retrieval_modes = [
        "full",
        "same_concept",
        "same_concept_different_operator",
        "same_operator",
        "same_operator_different_concept",
    ]

    audit = {
        "dataset_path": str(DATASET_PATH),
        "features_path": str(FEATURES_PATH),
        "n_rows": int(len(df)),
        "n_numeric_cols": int(len(numeric_cols(df))),
        "blocks": {name: {"display": b.name, "n_cols": len(b.cols), "cols_preview": b.cols[:24]} for name, b in blocks.items()},
        "retrieval_modes": retrieval_modes,
        "split_count": len(splits),
        "counts": {
            "concepts": int(df["concept_star"].nunique()),
            "operators": int(df["operator_star"].nunique()),
            "families": int(df["seed_family"].nunique()),
            "surfaces": int(df["surface_id"].nunique()),
        }
    }
    (OUTPUT_DIR / "sem5a4_column_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    probe_rows = []
    ret_rows = []
    ret_details = []

    for split_name, tr, te in splits:
        train = df.iloc[tr].copy().reset_index(drop=True)
        test = df.iloc[te].copy().reset_index(drop=True)

        # Probes.
        probe_preds = {}
        for target in ["concept_star", "operator_star"]:
            for bkey, block in blocks.items():
                if bkey == "ShapeCommitConcat":
                    # Skip concat for readout diagnosis to keep interpretation clean.
                    continue
                pred, metrics = fit_predict_probe(train, test, block.cols, target)
                metrics["split"] = split_name
                metrics["target"] = target
                metrics["block"] = block.name
                probe_rows.append(metrics)
                probe_preds[(target, bkey)] = pred
                print(f"[PROBE] {split_name} / {target} / {block.name}: acc={metrics['acc']:.3f}")

        # Similarities.
        sim_shape, families, meta = numeric_similarity(train, test, blocks["Shape"].cols)
        sim_commit, families2, meta2 = numeric_similarity(train, test, blocks["Commit"].cols)
        sim_concat, families3, meta3 = numeric_similarity(train, test, blocks["ShapeCommitConcat"].cols)

        if list(families) != list(families2) or list(families) != list(families3):
            raise RuntimeError("Family order mismatch.")

        sim_map = {
            "ShapeOnly": ("numeric", sim_shape),
            "CommitOnly": ("numeric", sim_commit),
            "ShapeCommitConcat": ("concat", sim_concat),
        }

        for ws, wc in FUSION_WEIGHTS:
            sim_map[f"LateFusion_s{ws:.1f}_c{wc:.1f}"] = ("late_fusion", ws * sim_shape + wc * sim_commit)

        # Oracle gates.
        true_concept = test["concept_star"].astype(str).values
        true_operator = test["operator_star"].astype(str).values
        sim_map["OracleConceptGate_ShapeRank"] = ("oracle", sim_shape.copy())
        sim_map["OracleOperatorGate_CommitRank"] = ("oracle", sim_commit.copy())
        sim_map["OracleBothGate_LateFusion"] = ("oracle", 0.5 * sim_shape + 0.5 * sim_commit)

        # Probe gates. Pick best available branch by theory:
        # concept from Commit, operator from Shape.
        pred_concept_commit = probe_preds.get(("concept_star", "Commit"))
        pred_operator_shape = probe_preds.get(("operator_star", "Shape"))
        pred_concept_shape = probe_preds.get(("concept_star", "Shape"))
        pred_operator_commit = probe_preds.get(("operator_star", "Commit"))

        for mode in retrieval_modes:
            # Normal queries.
            for qname, (qkind, sim) in sim_map.items():
                gate_mode = None
                pc = None
                po = None

                if qname == "OracleConceptGate_ShapeRank":
                    gate_mode = "pred_concept_gate"
                    pc = true_concept
                elif qname == "OracleOperatorGate_CommitRank":
                    gate_mode = "pred_operator_gate"
                    po = true_operator
                elif qname == "OracleBothGate_LateFusion":
                    gate_mode = "pred_both_gate"
                    pc = true_concept
                    po = true_operator

                metrics, details = score_retrieval(
                    qname, qkind, sim, families, meta, test, mode,
                    pred_concept=pc, pred_operator=po, gate_mode=gate_mode
                )
                metrics["split"] = split_name
                details["split"] = split_name
                ret_rows.append(metrics)
                ret_details.append(details)
                print(f"[RET] {split_name} / {mode} / {qname}: acc={metrics['acc_top1']:.3f}, r3={metrics['recall_at_3']:.3f}")

            # Probe-guided retrievals.
            probe_queries = []

            if pred_concept_commit is not None:
                probe_queries.append(("ProbeCommitConceptGate_ShapeRank", "probe", sim_shape, pred_concept_commit, None, "pred_concept_gate"))
            if pred_operator_shape is not None:
                probe_queries.append(("ProbeShapeOperatorGate_CommitRank", "probe", sim_commit, None, pred_operator_shape, "pred_operator_gate"))
            if pred_concept_commit is not None and pred_operator_shape is not None:
                probe_queries.append(("ProbeBothGate_LateFusion", "probe", 0.5 * sim_shape + 0.5 * sim_commit, pred_concept_commit, pred_operator_shape, "pred_both_gate"))

            # Alternative branch gates for diagnostics.
            if pred_concept_shape is not None:
                probe_queries.append(("ProbeShapeConceptGate_ShapeRank", "probe_alt", sim_shape, pred_concept_shape, None, "pred_concept_gate"))
            if pred_operator_commit is not None:
                probe_queries.append(("ProbeCommitOperatorGate_CommitRank", "probe_alt", sim_commit, None, pred_operator_commit, "pred_operator_gate"))

            for qname, qkind, sim, pc, po, gm in probe_queries:
                metrics, details = score_retrieval(
                    qname, qkind, sim, families, meta, test, mode,
                    pred_concept=pc, pred_operator=po, gate_mode=gm
                )
                metrics["split"] = split_name
                details["split"] = split_name
                ret_rows.append(metrics)
                ret_details.append(details)
                print(f"[RET] {split_name} / {mode} / {qname}: acc={metrics['acc_top1']:.3f}, r3={metrics['recall_at_3']:.3f}")

    probe_df = pd.DataFrame(probe_rows)
    ret_df = pd.DataFrame(ret_rows)
    details_df = pd.concat(ret_details, ignore_index=True) if ret_details else pd.DataFrame()

    probe_by_split_path = OUTPUT_DIR / "sem5a4_probe_by_split.csv"
    probe_summary_path = OUTPUT_DIR / "sem5a4_probe_summary.csv"
    ret_by_split_path = OUTPUT_DIR / "sem5a4_retrieval_by_split.csv"
    ret_summary_path = OUTPUT_DIR / "sem5a4_retrieval_summary.csv"
    details_path = OUTPUT_DIR / "sem5a4_retrieval_details.csv"
    diag_path = OUTPUT_DIR / "sem5a4_factorized_diagnosis.csv"
    verdict_path = OUTPUT_DIR / "sem5a4_verdict.json"

    probe_df.to_csv(probe_by_split_path, index=False, encoding="utf-8-sig")
    ret_df.to_csv(ret_by_split_path, index=False, encoding="utf-8-sig")
    details_df.to_csv(details_path, index=False, encoding="utf-8-sig")

    probe_summary = summarize_probe(probe_df)
    ret_summary = summarize_retrieval(ret_df)
    diag = factorized_diagnosis(probe_summary, ret_summary)

    probe_summary.to_csv(probe_summary_path, index=False, encoding="utf-8-sig")
    ret_summary.to_csv(ret_summary_path, index=False, encoding="utf-8-sig")
    diag.to_csv(diag_path, index=False, encoding="utf-8-sig")

    verdict = make_verdict(diag, probe_summary)
    verdict["output_files"] = {
        "probe_by_split": str(probe_by_split_path),
        "probe_summary": str(probe_summary_path),
        "retrieval_by_split": str(ret_by_split_path),
        "retrieval_summary": str(ret_summary_path),
        "retrieval_details": str(details_path),
        "factorized_diagnosis": str(diag_path),
        "column_audit": str(OUTPUT_DIR / "sem5a4_column_audit.json"),
    }
    verdict_path.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== SEM-5A.4 PROBE SUMMARY ===")
    print(probe_summary.to_string(index=False))
    print("\n=== SEM-5A.4 FACTORIZED DIAGNOSIS ===")
    print(diag.to_string(index=False))
    print("\n=== SEM-5A.4 VERDICT ===")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
