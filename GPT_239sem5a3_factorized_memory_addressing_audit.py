# -*- coding: utf-8 -*-
"""
SEM-5A.3: Factorized Memory Addressing Audit
============================================

Motivation
----------
SEM-5A.2 found:
  - ShapeQuery_L7_19 beats anonymized TextQuery in non-degenerate hard set.
  - ShapeQuery is good at same_concept/operator disambiguation.
  - ShapeQuery is weak at same_operator/concept disambiguation.
  - CommitQuery helps more in same_operator, suggesting concept/identity appears later.

Hypothesis
----------
Memory addressing is factorized:

  Address_F = ConceptAddress + OperatorShapeAddress

where:
  ShapeQuery_L7_19  ~= operator / trajectory-shape address
  CommitQuery_L23_25 ~= concept / identity confirmation address

This audit compares:
  TextQuery
  TextQuery_Anonymized
  InitQuery_L0_6
  ShapeQuery_L7_19
  CommitQuery_L23_25
  ShapeCommit_Concat
  ShapeCommit_LateFusion
  ConceptThenShape
  OperatorThenCommit

Expected strongest:
  ShapeCommit_LateFusion or ShapeCommit_Concat.

Default inputs
--------------
  sem5a2_outputs/sem5a2_dataset.csv
  sem5a2_outputs/sem5a2_features.csv

Outputs
-------
  sem5a3_outputs/sem5a3_retrieval_by_split.csv
  sem5a3_outputs/sem5a3_summary_by_query_and_mode.csv
  sem5a3_outputs/sem5a3_factorized_summary.csv
  sem5a3_outputs/sem5a3_verdict.json
  sem5a3_outputs/sem5a3_column_audit.json

Run
---
  python sem5a3_factorized_memory_addressing_audit.py
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupKFold


# =============================================================================
# Config
# =============================================================================

DATASET_PATH = Path("sem5a2_outputs/sem5a2_dataset.csv")
FEATURES_PATH = Path("sem5a2_outputs/sem5a2_features.csv")
OUTPUT_DIR = Path("sem5a3_outputs")

RANDOM_SEED = 42
TOP_KS = (1, 3, 5)
SOFTMAX_TEMPERATURE = 0.10

# Fusion weights to test for late fusion:
# score = w_shape * sim_shape + w_commit * sim_commit
FUSION_WEIGHTS = [
    (0.50, 0.50),
    (0.60, 0.40),
    (0.70, 0.30),
    (0.40, 0.60),
    (0.30, 0.70),
]


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


def safe_macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def safe_acc(y_true: List[str], y_pred: List[str]) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(accuracy_score(y_true, y_pred))


# =============================================================================
# Data and feature blocks
# =============================================================================

@dataclass
class QuerySpec:
    name: str
    kind: str  # text, numeric, concat, late_fusion, staged
    cols: Optional[List[str]] = None
    cols_a: Optional[List[str]] = None
    cols_b: Optional[List[str]] = None
    anonymize_text: bool = False
    fusion_weights: Optional[Tuple[float, float]] = None
    staged_mode: Optional[str] = None  # concept_then_shape, operator_then_commit


def merge_dataset_features(dataset: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if "row_id" in dataset.columns and "row_id" in features.columns:
        merged = dataset.merge(features, on="row_id", how="inner", suffixes=("", "_feat"))
    else:
        if len(dataset) != len(features):
            raise ValueError("Dataset/features row count differs and no row_id shared.")
        merged = pd.concat([dataset.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
        merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged


def build_query_specs(df: pd.DataFrame) -> Tuple[List[QuerySpec], Dict[str, List[str]]]:
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

    specs = [
        QuerySpec("TextQuery", "text", anonymize_text=False),
        QuerySpec("TextQuery_Anonymized", "text", anonymize_text=True),
    ]

    if init_cols:
        specs.append(QuerySpec("InitQuery_L0_6", "numeric", cols=init_cols))
    if shape_cols:
        specs.append(QuerySpec("ShapeQuery_L7_19", "numeric", cols=shape_cols))
    if commit_cols:
        specs.append(QuerySpec("CommitQuery_L23_25", "numeric", cols=commit_cols))

    if shape_cols and commit_cols:
        specs.append(QuerySpec("ShapeCommit_Concat", "concat", cols_a=shape_cols, cols_b=commit_cols))
        for ws, wc in FUSION_WEIGHTS:
            specs.append(QuerySpec(
                f"ShapeCommit_LateFusion_s{ws:.1f}_c{wc:.1f}",
                "late_fusion",
                cols_a=shape_cols,
                cols_b=commit_cols,
                fusion_weights=(ws, wc),
            ))
        specs.append(QuerySpec(
            "ConceptThenShape_commitFilter_shapeRank",
            "staged",
            cols_a=commit_cols,
            cols_b=shape_cols,
            staged_mode="concept_then_shape",
        ))
        specs.append(QuerySpec(
            "OperatorThenCommit_shapeFilter_commitRank",
            "staged",
            cols_a=shape_cols,
            cols_b=commit_cols,
            staged_mode="operator_then_commit",
        ))

    blocks = {"init": init_cols, "shape": shape_cols, "commit": commit_cols}
    return specs, blocks


def build_replacement_terms(df: pd.DataFrame) -> Dict[str, str]:
    replacements = {}
    for col, repl in [
        ("concept_star", "CONCEPT"),
        ("concept_phrase", "CONCEPT"),
        ("operator_star", "OPERATOR"),
        ("operator_id", "OPERATOR"),
        ("operator_phrase", "OPERATOR"),
        ("seed_family", "FAMILY"),
    ]:
        if col in df.columns:
            vals = sorted(set(df[col].dropna().astype(str)), key=len, reverse=True)
            for v in vals:
                if v.strip():
                    replacements[v] = repl
    return replacements


def anonymize_prompt_text(text: str, replacements: Dict[str, str]) -> str:
    out = str(text)
    for term, repl in replacements.items():
        term = str(term).strip()
        if not term or len(term) < 2:
            continue
        variants = {
            term,
            term.replace("_", " "),
            term.replace("-", " "),
            term.replace("_", "-"),
            term.replace(" ", "_"),
            term.replace(" ", "-"),
        }
        for v in variants:
            if v and len(v) >= 2:
                out = re.sub(re.escape(v), repl, out, flags=re.IGNORECASE)
    return out


def make_splits(df: pd.DataFrame, n_splits: int = 4) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    y = df["seed_family"].astype(str).values
    groups = df["surface_id"].astype(str).values
    unique_groups = np.unique(groups)
    splits = []
    if len(unique_groups) >= n_splits:
        gkf = GroupKFold(n_splits=min(n_splits, len(unique_groups)))
        for i, (tr, te) in enumerate(gkf.split(df, y, groups)):
            splits.append((f"leave_surface_group_{i}", tr, te))
    else:
        idx = np.arange(len(df))
        for g in unique_groups:
            te = idx[groups == g]
            tr = idx[groups != g]
            splits.append((f"leave_surface_{g}", tr, te))
    return splits


# =============================================================================
# Retrieval core
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


def build_text_centers(
    train_text: List[str],
    y_train: np.ndarray,
    meta_train: pd.DataFrame,
    test_text: List[str],
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1,
        max_features=25000,
    )
    x_train = vectorizer.fit_transform(train_text)
    x_test = vectorizer.transform(test_text)

    families = sorted(pd.Series(y_train).unique().astype(str).tolist())
    centers = []
    meta_rows = []
    for fam in families:
        rows = np.where(y_train == fam)[0]
        centers.append(np.asarray(x_train[rows].mean(axis=0)).ravel())
        sub = meta_train.loc[rows]
        meta_rows.append({
            "family": fam,
            "concept": str(sub["concept_star"].mode().iloc[0]),
            "operator": str(sub["operator_star"].mode().iloc[0]),
        })

    centers = np.vstack(centers)
    sim = cosine_similarity(x_test, centers)
    return sim, np.array(families, dtype=object), pd.DataFrame(meta_rows), x_test


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


def concat_similarity(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols_a: List[str],
    cols_b: List[str],
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    cols = list(cols_a) + list(cols_b)
    return numeric_similarity(train, test, cols)


def late_fusion_similarity(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols_a: List[str],
    cols_b: List[str],
    weights: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    sim_a, families_a, meta_a = numeric_similarity(train, test, cols_a)
    sim_b, families_b, meta_b = numeric_similarity(train, test, cols_b)

    if list(families_a) != list(families_b):
        raise RuntimeError("Family order mismatch in late fusion.")
    wa, wb = weights
    sim = wa * sim_a + wb * sim_b
    return sim, families_a, meta_a


def rank_with_candidate_restriction(
    sim: np.ndarray,
    families: np.ndarray,
    center_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, List[bool], List[int]]:
    masked = sim.copy()
    valid_rows = []
    candidate_counts = []

    for i in range(sim.shape[0]):
        mask = np.ones(sim.shape[1], dtype=bool)
        truth = str(test_meta.iloc[i]["__truth_family__"])
        cval = str(test_meta.iloc[i]["concept_star"])
        oval = str(test_meta.iloc[i]["operator_star"])

        if mode in {"same_concept", "same_concept_different_operator"}:
            mask &= (center_meta["concept"].astype(str).values == cval)

        if mode in {"same_operator", "same_operator_different_concept"}:
            mask &= (center_meta["operator"].astype(str).values == oval)

        if mode == "same_concept_different_operator":
            mask &= ((center_meta["operator"].astype(str).values != oval) |
                     (center_meta["family"].astype(str).values == truth))

        if mode == "same_operator_different_concept":
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


def staged_similarity(
    train: pd.DataFrame,
    test: pd.DataFrame,
    spec: QuerySpec,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    ConceptThenShape:
      use commit sim to choose / weight concept candidates, then shape rank inside predicted concept.
      Implementation: add a large bonus to candidates whose concept equals top commit concept.

    OperatorThenCommit:
      use shape sim to choose / weight operator candidates, then commit rank inside predicted operator.
      Implementation: add a large bonus to candidates whose operator equals top shape operator.
    """
    if spec.staged_mode == "concept_then_shape":
        sim_commit, families, meta = numeric_similarity(train, test, spec.cols_a)
        sim_shape, families2, meta2 = numeric_similarity(train, test, spec.cols_b)
        if list(families) != list(families2):
            raise RuntimeError("Family order mismatch in staged similarity.")

        top_commit_idx = np.argmax(sim_commit, axis=1)
        pred_concepts = meta.iloc[top_commit_idx]["concept"].astype(str).values
        bonus = np.zeros_like(sim_shape)
        for i, pc in enumerate(pred_concepts):
            bonus[i, meta["concept"].astype(str).values == pc] = 2.0
        sim = sim_shape + bonus
        return sim, families, meta

    if spec.staged_mode == "operator_then_commit":
        sim_shape, families, meta = numeric_similarity(train, test, spec.cols_a)
        sim_commit, families2, meta2 = numeric_similarity(train, test, spec.cols_b)
        if list(families) != list(families2):
            raise RuntimeError("Family order mismatch in staged similarity.")

        top_shape_idx = np.argmax(sim_shape, axis=1)
        pred_ops = meta.iloc[top_shape_idx]["operator"].astype(str).values
        bonus = np.zeros_like(sim_commit)
        for i, po in enumerate(pred_ops):
            bonus[i, meta["operator"].astype(str).values == po] = 2.0
        sim = sim_commit + bonus
        return sim, families, meta

    raise ValueError(f"Unknown staged_mode: {spec.staged_mode}")


def evaluate_spec(
    df: pd.DataFrame,
    spec: QuerySpec,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    retrieval_mode: str,
    replacements: Dict[str, str],
) -> Tuple[Dict, pd.DataFrame]:
    train = df.iloc[train_idx].copy().reset_index(drop=True)
    test = df.iloc[test_idx].copy().reset_index(drop=True)

    y_train = train["seed_family"].astype(str).values
    y_test = test["seed_family"].astype(str).values

    train_fams = set(y_train)
    keep = np.array([y in train_fams for y in y_test], dtype=bool)
    test = test.loc[keep].reset_index(drop=True)
    y_test = y_test[keep]
    kept_test_idx = np.array(test_idx)[keep]

    if len(y_test) == 0:
        raise ValueError("No test rows with truth family present in train.")

    if spec.kind == "text":
        train_text = train["prompt"].fillna("").astype(str).tolist()
        test_text = test["prompt"].fillna("").astype(str).tolist()
        if spec.anonymize_text:
            train_text = [anonymize_prompt_text(t, replacements) for t in train_text]
            test_text = [anonymize_prompt_text(t, replacements) for t in test_text]
        sim, families, center_meta, _ = build_text_centers(train_text, y_train, train, test_text)

    elif spec.kind == "numeric":
        sim, families, center_meta = numeric_similarity(train, test, spec.cols)

    elif spec.kind == "concat":
        sim, families, center_meta = concat_similarity(train, test, spec.cols_a, spec.cols_b)

    elif spec.kind == "late_fusion":
        sim, families, center_meta = late_fusion_similarity(train, test, spec.cols_a, spec.cols_b, spec.fusion_weights)

    elif spec.kind == "staged":
        sim, families, center_meta = staged_similarity(train, test, spec)

    else:
        raise ValueError(f"Unknown query kind: {spec.kind}")

    test_meta = test.copy()
    test_meta["__truth_family__"] = y_test

    ranked, masked, valid_rows, candidate_counts = rank_with_candidate_restriction(
        sim, families, center_meta, test_meta, retrieval_mode
    )

    valid = np.array(valid_rows, dtype=bool)
    if not valid.any():
        raise ValueError("No valid rows after candidate restriction.")

    y_eval = y_test[valid]
    ranked_eval = ranked[valid]
    masked_eval = masked[valid]
    kept_eval_idx = kept_test_idx[valid]
    candidate_counts_eval = np.array(candidate_counts)[valid]

    order = np.argsort(-masked_eval, axis=1)
    pred = ranked_eval[:, 0].astype(str)
    ent = row_softmax_entropy(masked_eval)

    metrics = {
        "query_type": spec.name,
        "query_kind": spec.kind,
        "retrieval_mode": retrieval_mode,
        "n_train": int(len(train_idx)),
        "n_test_raw": int(len(test_idx)),
        "n_test_eval": int(len(y_eval)),
        "n_families_train": int(len(families)),
        "candidate_count_mean": float(np.mean(candidate_counts_eval)),
        "candidate_count_min": int(np.min(candidate_counts_eval)),
        "candidate_count_max": int(np.max(candidate_counts_eval)),
        "acc_top1": safe_acc(list(y_eval), list(pred)),
        "macro_f1": safe_macro_f1(list(y_eval), list(pred)),
        "mrr": mrr_score(list(y_eval), ranked_eval),
        "entropy_mean": float(np.mean(ent)),
        "entropy_std": float(np.std(ent)),
    }
    for k in TOP_KS:
        metrics[f"recall_at_{k}"] = recall_at_k(list(y_eval), ranked_eval, min(k, ranked_eval.shape[1]))

    details = pd.DataFrame({
        "query_type": spec.name,
        "query_kind": spec.kind,
        "retrieval_mode": retrieval_mode,
        "truth_family": y_eval,
        "pred_family": pred,
        "top1_sim": masked_eval[np.arange(len(y_eval)), order[:, 0]],
        "entropy": ent,
        "candidate_count": candidate_counts_eval,
        "test_row_index": kept_eval_idx,
        "concept": test.loc[valid, "concept_star"].astype(str).values,
        "operator": test.loc[valid, "operator_star"].astype(str).values,
        "surface": test.loc[valid, "surface_id"].astype(str).values,
    })
    for k in TOP_KS:
        kk = min(k, ranked_eval.shape[1])
        details[f"hit_at_{k}"] = [truth in set(row[:kk]) for truth, row in zip(y_eval, ranked_eval)]

    return metrics, details


# =============================================================================
# Summary / Verdict
# =============================================================================

def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
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
        if col in metrics_df.columns:
            agg_map[col] = ["mean", "std"]

    summary = metrics_df.groupby(["retrieval_mode", "query_type", "query_kind"]).agg(agg_map)
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    return summary.reset_index().sort_values(["retrieval_mode", "acc_top1_mean"], ascending=[True, False])


def factorized_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode, sub in summary.groupby("retrieval_mode"):
        table = sub.set_index("query_type").to_dict(orient="index")
        best = max(table.keys(), key=lambda q: table[q].get("acc_top1_mean", -999))

        def get(q, metric="acc_top1_mean"):
            return table.get(q, {}).get(metric, np.nan)

        shape_keys = [q for q in table if q.startswith("ShapeQuery")]
        commit_keys = [q for q in table if q.startswith("CommitQuery")]
        fusion_keys = [q for q in table if q.startswith("ShapeCommit")]
        staged_keys = [q for q in table if "Then" in q]

        best_shape = max(shape_keys, key=lambda q: get(q)) if shape_keys else None
        best_commit = max(commit_keys, key=lambda q: get(q)) if commit_keys else None
        best_fusion = max(fusion_keys, key=lambda q: get(q)) if fusion_keys else None
        best_staged = max(staged_keys, key=lambda q: get(q)) if staged_keys else None

        row = {
            "retrieval_mode": mode,
            "best_query": best,
            "best_acc": get(best),
            "best_r3": get(best, "recall_at_3_mean"),
            "text_acc": get("TextQuery"),
            "anon_text_acc": get("TextQuery_Anonymized"),
            "init_acc": get("InitQuery_L0_6"),
            "shape_query": best_shape,
            "shape_acc": get(best_shape) if best_shape else np.nan,
            "shape_r3": get(best_shape, "recall_at_3_mean") if best_shape else np.nan,
            "commit_query": best_commit,
            "commit_acc": get(best_commit) if best_commit else np.nan,
            "commit_r3": get(best_commit, "recall_at_3_mean") if best_commit else np.nan,
            "best_fusion_query": best_fusion,
            "best_fusion_acc": get(best_fusion) if best_fusion else np.nan,
            "best_fusion_r3": get(best_fusion, "recall_at_3_mean") if best_fusion else np.nan,
            "best_staged_query": best_staged,
            "best_staged_acc": get(best_staged) if best_staged else np.nan,
            "best_staged_r3": get(best_staged, "recall_at_3_mean") if best_staged else np.nan,
            "candidate_count": get(best_shape, "candidate_count_mean_mean") if best_shape else np.nan,
        }

        row["fusion_minus_shape"] = row["best_fusion_acc"] - row["shape_acc"]
        row["fusion_minus_commit"] = row["best_fusion_acc"] - row["commit_acc"]
        row["fusion_minus_anon_text"] = row["best_fusion_acc"] - row["anon_text_acc"]
        row["staged_minus_shape"] = row["best_staged_acc"] - row["shape_acc"]
        row["staged_minus_commit"] = row["best_staged_acc"] - row["commit_acc"]
        row["staged_minus_anon_text"] = row["best_staged_acc"] - row["anon_text_acc"]

        rows.append(row)

    return pd.DataFrame(rows)


def make_verdict(fsum: pd.DataFrame) -> Dict:
    verdict = {
        "status": "UNDETERMINED",
        "pass_lite": False,
        "pass_strong": False,
        "notes": [],
    }
    if fsum.empty:
        verdict["status"] = "NO_SUMMARY"
        return verdict

    rows = {r["retrieval_mode"]: r for _, r in fsum.iterrows()}

    def fusion_ok(mode: str) -> bool:
        r = rows.get(mode)
        if r is None:
            return False
        return (
            r.get("best_fusion_acc", 0) >= 0.70 and
            r.get("best_fusion_r3", 0) >= 0.90 and
            r.get("fusion_minus_anon_text", -999) > 0.05 and
            r.get("fusion_minus_shape", -999) > 0.02 and
            r.get("candidate_count", 0) >= 3
        )

    def any_factorized_ok(mode: str) -> bool:
        r = rows.get(mode)
        if r is None:
            return False
        best_factorized = max(r.get("best_fusion_acc", 0), r.get("best_staged_acc", 0))
        best_r3 = max(r.get("best_fusion_r3", 0), r.get("best_staged_r3", 0))
        return (
            best_factorized >= 0.65 and
            best_r3 >= 0.85 and
            max(r.get("fusion_minus_anon_text", -999), r.get("staged_minus_anon_text", -999)) > 0.05
        )

    sc = fusion_ok("same_concept")
    so = fusion_ok("same_operator")
    sc_any = any_factorized_ok("same_concept")
    so_any = any_factorized_ok("same_operator")

    for mode in ["full", "same_concept", "same_operator", "same_concept_different_operator", "same_operator_different_concept"]:
        r = rows.get(mode)
        if r is None:
            continue
        if r.get("fusion_minus_shape", -999) > 0.02:
            verdict["notes"].append(f"{mode}: best fusion improves over ShapeQuery.")
        if r.get("fusion_minus_commit", -999) > 0.02:
            verdict["notes"].append(f"{mode}: best fusion improves over CommitQuery.")
        if r.get("fusion_minus_anon_text", -999) > 0.05:
            verdict["notes"].append(f"{mode}: best fusion beats anonymized TextQuery by >5 points.")
        if r.get("best_fusion_r3", 0) >= 0.90:
            verdict["notes"].append(f"{mode}: best fusion Recall@3 >= 0.90.")

    if sc_any or so_any:
        verdict["pass_lite"] = True

    if sc and so:
        verdict["pass_strong"] = True
        verdict["notes"].append("Fusion passes both same-concept and same-operator non-degenerate addressing tests.")

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

    specs, blocks = build_query_specs(df)
    replacements = build_replacement_terms(df)
    splits = make_splits(df, n_splits=4)

    retrieval_modes = [
        "full",
        "same_concept",
        "same_concept_different_operator",
        "same_operator",
        "same_operator_different_concept",
    ]

    column_audit = {
        "dataset_path": str(DATASET_PATH),
        "features_path": str(FEATURES_PATH),
        "n_rows": int(len(df)),
        "n_numeric_cols": int(len(numeric_cols(df))),
        "query_specs": [
            {
                "name": s.name,
                "kind": s.kind,
                "n_cols": len(s.cols or []) if s.cols is not None else None,
                "n_cols_a": len(s.cols_a or []) if s.cols_a is not None else None,
                "n_cols_b": len(s.cols_b or []) if s.cols_b is not None else None,
                "fusion_weights": s.fusion_weights,
                "staged_mode": s.staged_mode,
            }
            for s in specs
        ],
        "blocks": {k: len(v) for k, v in blocks.items()},
        "retrieval_modes": retrieval_modes,
        "split_count": len(splits),
        "dataset_counts": {
            "concepts": int(df["concept_star"].nunique()),
            "operators": int(df["operator_star"].nunique()),
            "families": int(df["seed_family"].nunique()),
            "surfaces": int(df["surface_id"].nunique()),
        }
    }
    (OUTPUT_DIR / "sem5a3_column_audit.json").write_text(
        json.dumps(column_audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    all_metrics = []
    all_details = []

    for split_name, tr, te in splits:
        for mode in retrieval_modes:
            for spec in specs:
                try:
                    metrics, details = evaluate_spec(df, spec, tr, te, mode, replacements)
                    metrics["split"] = split_name
                    details["split"] = split_name
                    all_metrics.append(metrics)
                    all_details.append(details)
                    print(
                        f"[OK] {split_name} / {mode} / {spec.name}: "
                        f"acc={metrics['acc_top1']:.3f}, r@3={metrics.get('recall_at_3', np.nan):.3f}, "
                        f"cand={metrics['candidate_count_mean']:.1f}"
                    )
                except Exception as exc:
                    print(f"[WARN] {split_name} / {mode} / {spec.name} failed: {exc}")

    if not all_metrics:
        raise RuntimeError("No metrics produced.")

    metrics_df = pd.DataFrame(all_metrics)
    details_df = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()

    metrics_path = OUTPUT_DIR / "sem5a3_retrieval_by_split.csv"
    details_path = OUTPUT_DIR / "sem5a3_retrieval_details.csv"
    summary_path = OUTPUT_DIR / "sem5a3_summary_by_query_and_mode.csv"
    fsum_path = OUTPUT_DIR / "sem5a3_factorized_summary.csv"
    verdict_path = OUTPUT_DIR / "sem5a3_verdict.json"

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    details_df.to_csv(details_path, index=False, encoding="utf-8-sig")

    summary = summarize_metrics(metrics_df)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    fsum = factorized_summary(summary)
    fsum.to_csv(fsum_path, index=False, encoding="utf-8-sig")

    verdict = make_verdict(fsum)
    verdict["output_files"] = {
        "retrieval_by_split": str(metrics_path),
        "retrieval_details": str(details_path),
        "summary_by_query_and_mode": str(summary_path),
        "factorized_summary": str(fsum_path),
        "column_audit": str(OUTPUT_DIR / "sem5a3_column_audit.json"),
    }
    verdict_path.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== SEM-5A.3 FACTORIZED SUMMARY ===")
    print(fsum.to_string(index=False))
    print("\n=== SEM-5A.3 VERDICT ===")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
