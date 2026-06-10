# -*- coding: utf-8 -*-
"""
SEM-5A.5: Prototype-Gate Memory Addressing Audit
================================================

Motivation
----------
SEM-5A.4 showed:
  - Best concept probe: Commit L23-L25
  - Best operator probe: Shape L7-L19
  - Probe-guided retrieval improves over late fusion
  - Oracle gate is perfect, so the remaining bottleneck is gate quality

SEM-5A.5 removes external supervised probe classifiers.

Instead, it tests whether the memory bank itself contains prototype-addressable
concept/operator structure:

  ConceptGate  = nearest concept prototype in Commit space
  OperatorGate = nearest operator prototype in Shape space

Then combine gates with memory-family retrieval.

Core hypothesis
---------------
MemoryAddress_F is factorized and internally prototype-addressable:

  MemoryAddress_F = (ConceptPrototype_commit, OperatorPrototype_shape)

If prototype gates work, SEM-5 addressing becomes:

  Prompt -> (q_commit, q_shape) -> prototype gates -> MemoryUnit_F

Inputs
------
  sem5a2_outputs/sem5a2_dataset.csv
  sem5a2_outputs/sem5a2_features.csv

Outputs
-------
  sem5a5_outputs/sem5a5_gate_by_split.csv
  sem5a5_outputs/sem5a5_gate_summary.csv
  sem5a5_outputs/sem5a5_retrieval_by_split.csv
  sem5a5_outputs/sem5a5_retrieval_summary.csv
  sem5a5_outputs/sem5a5_prototype_diagnosis.csv
  sem5a5_outputs/sem5a5_verdict.json
  sem5a5_outputs/sem5a5_column_audit.json

Run
---
  python sem5a5_prototype_gate_memory_addressing_audit.py
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupKFold


# =============================================================================
# Config
# =============================================================================

DATASET_PATH = Path("sem5a2_outputs/sem5a2_dataset.csv")
FEATURES_PATH = Path("sem5a2_outputs/sem5a2_features.csv")
OUTPUT_DIR = Path("sem5a5_outputs")

RANDOM_SEED = 42
TOP_KS = (1, 3, 5)
SOFTMAX_TEMPERATURE = 0.10

# Score weights:
# family score + gate bonuses
FAMILY_SHAPE_WEIGHT = 0.50
FAMILY_COMMIT_WEIGHT = 0.50
CONCEPT_GATE_BONUS = 2.0
OPERATOR_GATE_BONUS = 2.0

# Optional soft gate uses similarity-to-prototype as continuous score.
USE_SOFT_GATE = True


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
# Data / blocks
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

    return {
        "Init": Block("Init_L0_6", init_cols),
        "Shape": Block("Shape_L7_19", shape_cols),
        "Commit": Block("Commit_L23_25", commit_cols),
        "ShapeCommit": Block("ShapeCommit_Concat", list(shape_cols) + list(commit_cols)),
    }


def make_splits(df: pd.DataFrame, n_splits: int = 4) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    y = df["seed_family"].astype(str).values
    groups = df["surface_id"].astype(str).values
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    return [(f"leave_surface_group_{i}", tr, te) for i, (tr, te) in enumerate(gkf.split(df, y, groups))]


# =============================================================================
# Prototype gates
# =============================================================================

def prototype_similarities(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: List[str],
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build target prototypes from train, return sim[test, target_prototypes].
    """
    y_train = train[target_col].astype(str).values
    labels = sorted(pd.Series(y_train).unique().tolist())

    x_train = train[cols].to_numpy(dtype=float)
    x_test = test[cols].to_numpy(dtype=float)
    x_train, x_test = zscore_train_test(x_train, x_test)

    centers = []
    for lab in labels:
        centers.append(x_train[y_train == lab].mean(axis=0))
    centers = np.vstack(centers)

    sim = cosine_similarity(x_test, centers)
    return sim, np.array(labels, dtype=object)


def gate_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: List[str],
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    sim, labels = prototype_similarities(train, test, cols, target_col)
    order = np.argsort(-sim, axis=1)
    pred = labels[order[:, 0]].astype(str)

    truth = test[target_col].astype(str).values
    metrics = {
        "target": target_col,
        "acc": safe_acc(truth, pred),
        "macro_f1": safe_f1(truth, pred),
        "recall_at_3": recall_at_k(list(truth), labels[order], min(3, len(labels))),
        "mrr": mrr_score(list(truth), labels[order]),
        "n_test": int(len(test)),
        "n_classes": int(len(labels)),
    }
    return pred, sim, labels, metrics


def family_centers(
    train: pd.DataFrame,
    cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    y_train = train["seed_family"].astype(str).values
    families = sorted(pd.Series(y_train).unique().tolist())

    x_train = train[cols].to_numpy(dtype=float)
    # For family centers we zscore against train only. Caller must zscore test with same params,
    # so we return raw centers? Easier: use helper below for test too.
    labels = np.array(families, dtype=object)
    return labels, y_train, train[cols].to_numpy(dtype=float), train["concept_star"].astype(str).values, train["operator_star"].astype(str).values


def family_similarity(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    y_train = train["seed_family"].astype(str).values
    families = sorted(pd.Series(y_train).unique().tolist())

    x_train = train[cols].to_numpy(dtype=float)
    x_test = test[cols].to_numpy(dtype=float)
    x_train, x_test = zscore_train_test(x_train, x_test)

    centers = []
    meta_rows = []
    for fam in families:
        mask = y_train == fam
        centers.append(x_train[mask].mean(axis=0))
        sub = train.loc[mask]
        meta_rows.append({
            "family": fam,
            "concept": str(sub["concept_star"].mode().iloc[0]),
            "operator": str(sub["operator_star"].mode().iloc[0]),
        })

    centers = np.vstack(centers)
    sim = cosine_similarity(x_test, centers)
    return sim, np.array(families, dtype=object), pd.DataFrame(meta_rows)


def make_gate_scores_for_families(
    center_meta: pd.DataFrame,
    pred_concept: np.ndarray,
    pred_operator: np.ndarray,
    concept_sim: Optional[np.ndarray] = None,
    concept_labels: Optional[np.ndarray] = None,
    operator_sim: Optional[np.ndarray] = None,
    operator_labels: Optional[np.ndarray] = None,
    soft: bool = USE_SOFT_GATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return concept_gate_score[test, family] and operator_gate_score[test, family].
    Hard gate: bonus if family concept/operator matches predicted label.
    Soft gate: similarity score of family concept/operator prototype.
    """
    n = len(pred_concept)
    m = len(center_meta)
    concept_gate = np.zeros((n, m), dtype=float)
    operator_gate = np.zeros((n, m), dtype=float)

    family_concepts = center_meta["concept"].astype(str).values
    family_ops = center_meta["operator"].astype(str).values

    if soft and concept_sim is not None and concept_labels is not None:
        concept_label_to_idx = {str(l): i for i, l in enumerate(concept_labels)}
        for j, c in enumerate(family_concepts):
            idx = concept_label_to_idx.get(str(c))
            if idx is not None:
                concept_gate[:, j] = concept_sim[:, idx]
    else:
        for i, pc in enumerate(pred_concept):
            concept_gate[i, family_concepts == str(pc)] = 1.0

    if soft and operator_sim is not None and operator_labels is not None:
        op_label_to_idx = {str(l): i for i, l in enumerate(operator_labels)}
        for j, o in enumerate(family_ops):
            idx = op_label_to_idx.get(str(o))
            if idx is not None:
                operator_gate[:, j] = operator_sim[:, idx]
    else:
        for i, po in enumerate(pred_operator):
            operator_gate[i, family_ops == str(po)] = 1.0

    return concept_gate, operator_gate


# =============================================================================
# Ranking / metrics
# =============================================================================

def apply_retrieval_restriction(
    score: np.ndarray,
    families: np.ndarray,
    center_meta: pd.DataFrame,
    test: pd.DataFrame,
    retrieval_mode: str,
) -> Tuple[np.ndarray, np.ndarray, List[bool], List[int]]:
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
    query_type: str,
    query_kind: str,
    score: np.ndarray,
    families: np.ndarray,
    center_meta: pd.DataFrame,
    test: pd.DataFrame,
    retrieval_mode: str,
) -> Tuple[Dict, pd.DataFrame]:
    ranked, masked, valid_rows, candidate_counts = apply_retrieval_restriction(
        score, families, center_meta, test, retrieval_mode
    )
    valid = np.array(valid_rows, dtype=bool)
    y = test["seed_family"].astype(str).values[valid]
    ranked_eval = ranked[valid]
    masked_eval = masked[valid]
    candidate_counts_eval = np.array(candidate_counts)[valid]

    if len(y) == 0:
        raise ValueError("No valid retrieval rows.")

    pred = ranked_eval[:, 0].astype(str)
    ent = row_softmax_entropy(masked_eval)

    metrics = {
        "query_type": query_type,
        "query_kind": query_kind,
        "retrieval_mode": retrieval_mode,
        "n_test_eval": int(len(y)),
        "candidate_count_mean": float(np.mean(candidate_counts_eval)),
        "acc_top1": safe_acc(y, pred),
        "macro_f1": safe_f1(y, pred),
        "mrr": mrr_score(list(y), ranked_eval),
        "entropy_mean": float(np.mean(ent)),
    }
    for k in TOP_KS:
        metrics[f"recall_at_{k}"] = recall_at_k(list(y), ranked_eval, min(k, ranked_eval.shape[1]))

    details = pd.DataFrame({
        "query_type": query_type,
        "query_kind": query_kind,
        "retrieval_mode": retrieval_mode,
        "truth_family": y,
        "pred_family": pred,
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
# Summary / Verdict
# =============================================================================

def summarize_gate(gate_df: pd.DataFrame) -> pd.DataFrame:
    agg = gate_df.groupby(["gate_type", "space"]).agg({
        "acc": ["mean", "std"],
        "macro_f1": ["mean", "std"],
        "recall_at_3": ["mean", "std"],
        "mrr": ["mean", "std"],
        "n_test": ["sum"],
    })
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index().sort_values(["gate_type", "acc_mean"], ascending=[True, False])


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
        agg_map[f"recall_at_{k}"] = ["mean", "std"]

    agg = ret_df.groupby(["retrieval_mode", "query_type", "query_kind"]).agg(agg_map)
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index().sort_values(["retrieval_mode", "acc_top1_mean"], ascending=[True, False])


def prototype_diagnosis(gate_summary: pd.DataFrame, ret_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    best_concept = gate_summary[gate_summary["gate_type"] == "ConceptGate"].sort_values("acc_mean", ascending=False).head(1)
    best_operator = gate_summary[gate_summary["gate_type"] == "OperatorGate"].sort_values("acc_mean", ascending=False).head(1)

    for mode, sub in ret_summary.groupby("retrieval_mode"):
        table = sub.set_index("query_type").to_dict(orient="index")

        def get(q, metric="acc_top1_mean"):
            return table.get(q, {}).get(metric, np.nan)

        def best_prefix(prefix):
            keys = [q for q in table if q.startswith(prefix)]
            if not keys:
                return None
            return max(keys, key=lambda q: get(q))

        best_proto = best_prefix("Prototype")
        best_oracle = best_prefix("Oracle")
        best_base = max(["ShapeOnly", "CommitOnly", "ShapeCommitLateFusion"], key=lambda q: get(q))
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
            "late_acc": get("ShapeCommitLateFusion"),
            "late_r3": get("ShapeCommitLateFusion", "recall_at_3_mean"),
            "best_base_query": best_base,
            "best_base_acc": get(best_base),
            "best_proto_query": best_proto,
            "best_proto_acc": get(best_proto) if best_proto else np.nan,
            "best_proto_r3": get(best_proto, "recall_at_3_mean") if best_proto else np.nan,
            "best_oracle_query": best_oracle,
            "best_oracle_acc": get(best_oracle) if best_oracle else np.nan,
            "best_oracle_r3": get(best_oracle, "recall_at_3_mean") if best_oracle else np.nan,
        }
        row["proto_minus_late"] = row["best_proto_acc"] - row["late_acc"]
        row["proto_minus_base"] = row["best_proto_acc"] - row["best_base_acc"]
        row["oracle_minus_proto"] = row["best_oracle_acc"] - row["best_proto_acc"]
        rows.append(row)

    diag = pd.DataFrame(rows)

    if not best_concept.empty:
        diag["best_concept_gate_space"] = best_concept.iloc[0]["space"]
        diag["best_concept_gate_acc"] = float(best_concept.iloc[0]["acc_mean"])
        diag["best_concept_gate_r3"] = float(best_concept.iloc[0]["recall_at_3_mean"])
    if not best_operator.empty:
        diag["best_operator_gate_space"] = best_operator.iloc[0]["space"]
        diag["best_operator_gate_acc"] = float(best_operator.iloc[0]["acc_mean"])
        diag["best_operator_gate_r3"] = float(best_operator.iloc[0]["recall_at_3_mean"])

    return diag


def make_verdict(diag: pd.DataFrame, gate_summary: pd.DataFrame) -> Dict:
    verdict = {
        "status": "UNDETERMINED",
        "pass_lite": False,
        "pass_strong": False,
        "notes": [],
    }
    if diag.empty:
        verdict["status"] = "NO_DIAGNOSIS"
        return verdict

    best_concept = gate_summary[gate_summary["gate_type"] == "ConceptGate"].sort_values("acc_mean", ascending=False).head(1)
    best_operator = gate_summary[gate_summary["gate_type"] == "OperatorGate"].sort_values("acc_mean", ascending=False).head(1)

    if not best_concept.empty:
        verdict["notes"].append(
            f"Best ConceptGate: {best_concept.iloc[0]['space']}, "
            f"acc={best_concept.iloc[0]['acc_mean']:.3f}, "
            f"r3={best_concept.iloc[0]['recall_at_3_mean']:.3f}."
        )
    if not best_operator.empty:
        verdict["notes"].append(
            f"Best OperatorGate: {best_operator.iloc[0]['space']}, "
            f"acc={best_operator.iloc[0]['acc_mean']:.3f}, "
            f"r3={best_operator.iloc[0]['recall_at_3_mean']:.3f}."
        )

    rows = {r["retrieval_mode"]: r for _, r in diag.iterrows()}

    for mode in ["same_concept", "same_operator", "full"]:
        r = rows.get(mode)
        if r is None:
            continue
        if r.get("proto_minus_late", -999) > 0.02:
            verdict["notes"].append(f"{mode}: prototype gate improves over late fusion.")
        if r.get("best_proto_r3", 0) >= 0.90:
            verdict["notes"].append(f"{mode}: prototype-gated retrieval Recall@3 >= 0.90.")
        if r.get("oracle_minus_proto", 0) > 0.05:
            verdict["notes"].append(f"{mode}: oracle still shows remaining headroom.")

    sc = rows.get("same_concept")
    so = rows.get("same_operator")

    sc_lite = sc is not None and (
        sc.get("best_proto_acc", 0) > sc.get("late_acc", 0) + 0.02
        or sc.get("best_proto_r3", 0) >= 0.90
    )
    so_lite = so is not None and (
        so.get("best_proto_acc", 0) > so.get("late_acc", 0) + 0.02
        or so.get("best_proto_r3", 0) >= 0.85
    )

    if sc_lite or so_lite:
        verdict["pass_lite"] = True

    sc_strong = sc is not None and sc.get("best_proto_acc", 0) >= 0.75 and sc.get("best_proto_r3", 0) >= 0.90
    so_strong = so is not None and so.get("best_proto_acc", 0) >= 0.65 and so.get("best_proto_r3", 0) >= 0.85

    if sc_strong and so_strong:
        verdict["pass_strong"] = True
        verdict["notes"].append("Prototype-gated addressing reaches strong criteria in both same-concept and same-operator modes.")

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
        "blocks": {k: {"display": v.name, "n_cols": len(v.cols), "cols_preview": v.cols[:24]} for k, v in blocks.items()},
        "retrieval_modes": retrieval_modes,
        "split_count": len(splits),
        "counts": {
            "concepts": int(df["concept_star"].nunique()),
            "operators": int(df["operator_star"].nunique()),
            "families": int(df["seed_family"].nunique()),
            "surfaces": int(df["surface_id"].nunique()),
        },
        "gate_bonus": {
            "concept": CONCEPT_GATE_BONUS,
            "operator": OPERATOR_GATE_BONUS,
            "soft_gate": USE_SOFT_GATE,
        }
    }
    (OUTPUT_DIR / "sem5a5_column_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    gate_rows = []
    ret_rows = []
    ret_details = []

    for split_name, tr, te in splits:
        train = df.iloc[tr].copy().reset_index(drop=True)
        test = df.iloc[te].copy().reset_index(drop=True)

        # Gate predictions / similarities.
        gate_info = {}

        for gate_type, target, space_key in [
            ("ConceptGate", "concept_star", "Commit"),
            ("ConceptGate", "concept_star", "Shape"),
            ("ConceptGate", "concept_star", "Init"),
            ("OperatorGate", "operator_star", "Shape"),
            ("OperatorGate", "operator_star", "Commit"),
            ("OperatorGate", "operator_star", "Init"),
        ]:
            block = blocks[space_key]
            pred, sim, labels, metrics = gate_predict(train, test, block.cols, target)
            metrics["split"] = split_name
            metrics["gate_type"] = gate_type
            metrics["space"] = block.name
            gate_rows.append(metrics)
            gate_info[(gate_type, space_key)] = {
                "pred": pred,
                "sim": sim,
                "labels": labels,
                "metrics": metrics,
            }
            print(f"[GATE] {split_name} / {gate_type} / {block.name}: acc={metrics['acc']:.3f}, r3={metrics['recall_at_3']:.3f}")

        # Family similarities.
        sim_shape, families, meta = family_similarity(train, test, blocks["Shape"].cols)
        sim_commit, families2, meta2 = family_similarity(train, test, blocks["Commit"].cols)
        sim_late = FAMILY_SHAPE_WEIGHT * sim_shape + FAMILY_COMMIT_WEIGHT * sim_commit

        if list(families) != list(families2):
            raise RuntimeError("Family mismatch between shape and commit.")

        # Best theoretical gates:
        concept_gate = gate_info[("ConceptGate", "Commit")]
        operator_gate = gate_info[("OperatorGate", "Shape")]

        concept_gate_alt = gate_info[("ConceptGate", "Shape")]
        operator_gate_alt = gate_info[("OperatorGate", "Commit")]

        # Make soft/hard family gate scores.
        cg_soft, og_soft = make_gate_scores_for_families(
            meta,
            pred_concept=concept_gate["pred"],
            pred_operator=operator_gate["pred"],
            concept_sim=concept_gate["sim"],
            concept_labels=concept_gate["labels"],
            operator_sim=operator_gate["sim"],
            operator_labels=operator_gate["labels"],
            soft=True,
        )

        cg_hard, og_hard = make_gate_scores_for_families(
            meta,
            pred_concept=concept_gate["pred"],
            pred_operator=operator_gate["pred"],
            soft=False,
        )

        cg_alt_soft, og_alt_soft = make_gate_scores_for_families(
            meta,
            pred_concept=concept_gate_alt["pred"],
            pred_operator=operator_gate_alt["pred"],
            concept_sim=concept_gate_alt["sim"],
            concept_labels=concept_gate_alt["labels"],
            operator_sim=operator_gate_alt["sim"],
            operator_labels=operator_gate_alt["labels"],
            soft=True,
        )

        # Oracle gates.
        true_concept = test["concept_star"].astype(str).values
        true_operator = test["operator_star"].astype(str).values
        oracle_cg, oracle_og = make_gate_scores_for_families(
            meta,
            pred_concept=true_concept,
            pred_operator=true_operator,
            soft=False,
        )

        query_scores = {
            "ShapeOnly": ("baseline", sim_shape),
            "CommitOnly": ("baseline", sim_commit),
            "ShapeCommitLateFusion": ("late_fusion", sim_late),

            # Prototype gates.
            "PrototypeConceptGate_ShapeRank_soft": ("prototype", sim_shape + CONCEPT_GATE_BONUS * cg_soft),
            "PrototypeOperatorGate_CommitRank_soft": ("prototype", sim_commit + OPERATOR_GATE_BONUS * og_soft),
            "PrototypeBothGate_LateFusion_soft": ("prototype", sim_late + CONCEPT_GATE_BONUS * cg_soft + OPERATOR_GATE_BONUS * og_soft),

            "PrototypeConceptGate_ShapeRank_hard": ("prototype", sim_shape + CONCEPT_GATE_BONUS * cg_hard),
            "PrototypeOperatorGate_CommitRank_hard": ("prototype", sim_commit + OPERATOR_GATE_BONUS * og_hard),
            "PrototypeBothGate_LateFusion_hard": ("prototype", sim_late + CONCEPT_GATE_BONUS * cg_hard + OPERATOR_GATE_BONUS * og_hard),

            # Alternative spaces.
            "PrototypeAltConceptShapeGate_ShapeRank_soft": ("prototype_alt", sim_shape + CONCEPT_GATE_BONUS * cg_alt_soft),
            "PrototypeAltOperatorCommitGate_CommitRank_soft": ("prototype_alt", sim_commit + OPERATOR_GATE_BONUS * og_alt_soft),

            # Oracle upper bounds.
            "OracleConceptGate_ShapeRank": ("oracle", sim_shape + CONCEPT_GATE_BONUS * oracle_cg),
            "OracleOperatorGate_CommitRank": ("oracle", sim_commit + OPERATOR_GATE_BONUS * oracle_og),
            "OracleBothGate_LateFusion": ("oracle", sim_late + CONCEPT_GATE_BONUS * oracle_cg + OPERATOR_GATE_BONUS * oracle_og),
        }

        for mode in retrieval_modes:
            for qname, (qkind, score) in query_scores.items():
                metrics, details = score_retrieval(qname, qkind, score, families, meta, test, mode)
                metrics["split"] = split_name
                details["split"] = split_name
                ret_rows.append(metrics)
                ret_details.append(details)
                print(f"[RET] {split_name} / {mode} / {qname}: acc={metrics['acc_top1']:.3f}, r3={metrics['recall_at_3']:.3f}")

    gate_df = pd.DataFrame(gate_rows)
    ret_df = pd.DataFrame(ret_rows)
    details_df = pd.concat(ret_details, ignore_index=True) if ret_details else pd.DataFrame()

    gate_by_split_path = OUTPUT_DIR / "sem5a5_gate_by_split.csv"
    gate_summary_path = OUTPUT_DIR / "sem5a5_gate_summary.csv"
    ret_by_split_path = OUTPUT_DIR / "sem5a5_retrieval_by_split.csv"
    ret_summary_path = OUTPUT_DIR / "sem5a5_retrieval_summary.csv"
    details_path = OUTPUT_DIR / "sem5a5_retrieval_details.csv"
    diag_path = OUTPUT_DIR / "sem5a5_prototype_diagnosis.csv"
    verdict_path = OUTPUT_DIR / "sem5a5_verdict.json"

    gate_df.to_csv(gate_by_split_path, index=False, encoding="utf-8-sig")
    ret_df.to_csv(ret_by_split_path, index=False, encoding="utf-8-sig")
    details_df.to_csv(details_path, index=False, encoding="utf-8-sig")

    gate_summary = summarize_gate(gate_df)
    ret_summary = summarize_retrieval(ret_df)
    diag = prototype_diagnosis(gate_summary, ret_summary)

    gate_summary.to_csv(gate_summary_path, index=False, encoding="utf-8-sig")
    ret_summary.to_csv(ret_summary_path, index=False, encoding="utf-8-sig")
    diag.to_csv(diag_path, index=False, encoding="utf-8-sig")

    verdict = make_verdict(diag, gate_summary)
    verdict["output_files"] = {
        "gate_by_split": str(gate_by_split_path),
        "gate_summary": str(gate_summary_path),
        "retrieval_by_split": str(ret_by_split_path),
        "retrieval_summary": str(ret_summary_path),
        "retrieval_details": str(details_path),
        "prototype_diagnosis": str(diag_path),
        "column_audit": str(OUTPUT_DIR / "sem5a5_column_audit.json"),
    }
    verdict_path.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== SEM-5A.5 GATE SUMMARY ===")
    print(gate_summary.to_string(index=False))
    print("\n=== SEM-5A.5 PROTOTYPE DIAGNOSIS ===")
    print(diag.to_string(index=False))
    print("\n=== SEM-5A.5 VERDICT ===")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
