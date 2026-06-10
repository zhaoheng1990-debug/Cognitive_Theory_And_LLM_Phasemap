# -*- coding: utf-8 -*-
"""
SEM-5A.2: Non-Degenerate Hard-Set Builder + Addressing Audit
============================================================

Purpose
-------
SEM-5A.1 passed Lite, but its same-concept mode was degenerate:
  family = 12, concept = 12
so each concept almost mapped to one family.

SEM-5A.2 fixes this by generating a non-degenerate factorial prompt set:

  concepts × operators × surfaces

Each MemoryFamily is defined as:

  seed_family = concept_star + "__" + operator_star

This makes:
  same_concept retrieval  = operator disambiguation within one concept
  same_operator retrieval = concept disambiguation within one operator

Default design:
  12 concepts × 8 operators × 4 surfaces = 384 prompts

Then the script can do one of two things:

A) Build only:
   Generate sem5a2_outputs/sem5a2_dataset.csv
   You then run your existing feature extraction pipeline on that dataset.

B) Build + Audit:
   If sem5a2_outputs/sem5a2_features.csv exists, this script automatically runs
   the same hard-negative retrieval audit as SEM-5A.1.

Expected workflow
-----------------
1. Run:
     python sem5a2_nondegenerate_hardset_builder_and_audit.py

2. If features are missing, the script writes:
     sem5a2_outputs/sem5a2_dataset.csv

3. Run your feature extractor on sem5a2_dataset.csv and save:
     sem5a2_outputs/sem5a2_features.csv

4. Run this script again:
     python sem5a2_nondegenerate_hardset_builder_and_audit.py

Outputs
-------
  sem5a2_outputs/sem5a2_dataset.csv
  sem5a2_outputs/sem5a2_dataset_audit.json

If features exist:
  sem5a2_outputs/sem5a2_retrieval_by_split.csv
  sem5a2_outputs/sem5a2_summary_by_query_and_mode.csv
  sem5a2_outputs/sem5a2_hard_negative_summary.csv
  sem5a2_outputs/sem5a2_verdict.json
  sem5a2_outputs/sem5a2_column_audit.json

Key verdict
-----------
PASS-Strong requires:
  ShapeQuery_L7_19 beats anonymized TextQuery in both:
    same_concept
    same_operator
  with Shape Recall@3 >= 0.90 in both modes.
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
# Paths
# =============================================================================

OUTPUT_DIR = Path("sem5a2_outputs")
DATASET_PATH = OUTPUT_DIR / "sem5a2_dataset.csv"
FEATURES_PATH = OUTPUT_DIR / "sem5a2_features.csv"

RANDOM_SEED = 42
TOP_KS = (1, 3, 5)
SOFTMAX_TEMPERATURE = 0.10


# =============================================================================
# Non-degenerate factorial prompt set
# =============================================================================

CONCEPTS = [
    ("Gravity", "gravity", "physical phenomenon involving attraction between masses"),
    ("Inertia", "inertia", "tendency of objects to resist changes in motion"),
    ("Photosynthesis", "photosynthesis", "biological process converting light into chemical energy"),
    ("Democracy", "democracy", "political system based on public participation and representation"),
    ("Market", "market", "system where goods, services, and prices interact"),
    ("Memory", "memory", "system for storing and retrieving information"),
    ("TransformerModel", "transformer model", "neural network architecture using attention mechanisms"),
    ("Python", "Python programming", "programming language used for software and data tasks"),
    ("Internet", "internet", "global network connecting computers and services"),
    ("ClimateChange", "climate change", "long-term shifts in climate patterns"),
    ("Ocean", "ocean", "large body of salt water covering much of Earth"),
    ("Robot", "robot", "machine capable of sensing, acting, or automating tasks"),
]

OPERATORS = [
    ("Definition", "definition", "define what the concept is"),
    ("MechanismExplanation", "mechanism", "explain how the concept works internally"),
    ("CausalExplanation", "causal", "explain causes and effects related to the concept"),
    ("RelationMapping", "relation", "map how the concept relates to other concepts"),
    ("PropertyDescription", "property", "describe key properties and features"),
    ("Comparison", "comparison", "compare the concept with a nearby alternative"),
    ("RiskAudit", "risk", "audit possible risks, failure modes, or limitations"),
    ("Planning", "planning", "make a practical plan involving the concept"),
]

SURFACE_TEMPLATES = [
    (
        "surface_direct",
        "Please give a {operator_phrase} for {concept_phrase}. Focus on the core idea rather than examples."
    ),
    (
        "surface_research",
        "I am studying {concept_phrase}. Write a concise {operator_phrase} that would help a research note."
    ),
    (
        "surface_teaching",
        "Teach a beginner about {concept_phrase} by giving a clear {operator_phrase}."
    ),
    (
        "surface_applied",
        "For an applied project involving {concept_phrase}, provide a useful {operator_phrase}."
    ),
]

# Extra challenge surfaces not used by default. Set INCLUDE_EXTRA_SURFACES=True
# if you want 12 × 8 × 6 = 576 prompts.
INCLUDE_EXTRA_SURFACES = False
EXTRA_SURFACE_TEMPLATES = [
    (
        "surface_minimal",
        "{concept_phrase}: {operator_phrase}."
    ),
    (
        "surface_question",
        "What is the best way to produce a {operator_phrase} of {concept_phrase}?"
    ),
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_seed_family(concept_id: str, operator_id: str) -> str:
    return f"{concept_id}__{operator_id}"


def build_dataset() -> pd.DataFrame:
    rows = []
    surfaces = list(SURFACE_TEMPLATES)
    if INCLUDE_EXTRA_SURFACES:
        surfaces += EXTRA_SURFACE_TEMPLATES

    row_id = 0
    for concept_star, concept_phrase, concept_desc in CONCEPTS:
        for operator_star, operator_id, operator_phrase in OPERATORS:
            seed_family = make_seed_family(concept_star, operator_star)
            for surface_id, template in surfaces:
                prompt = template.format(
                    concept_phrase=concept_phrase,
                    operator_phrase=operator_phrase,
                )
                rows.append({
                    "row_id": row_id,
                    "row_type": "sem5a2_hard_factorial",
                    "seed_family": seed_family,
                    "concept_star": concept_star,
                    "concept_phrase": concept_phrase,
                    "concept_description": concept_desc,
                    "operator_star": operator_star,
                    "operator_id": operator_id,
                    "operator_phrase": operator_phrase,
                    "surface_id": surface_id,
                    "prompt": prompt,
                })
                row_id += 1

    return pd.DataFrame(rows)


def audit_dataset(df: pd.DataFrame) -> Dict:
    fam_per_concept = df.groupby("concept_star")["seed_family"].nunique().to_dict()
    fam_per_operator = df.groupby("operator_star")["seed_family"].nunique().to_dict()
    surface_per_family = df.groupby("seed_family")["surface_id"].nunique().to_dict()

    return {
        "n_rows": int(len(df)),
        "n_concepts": int(df["concept_star"].nunique()),
        "n_operators": int(df["operator_star"].nunique()),
        "n_seed_families": int(df["seed_family"].nunique()),
        "n_surfaces": int(df["surface_id"].nunique()),
        "families_per_concept_min": int(min(fam_per_concept.values())),
        "families_per_concept_max": int(max(fam_per_concept.values())),
        "families_per_operator_min": int(min(fam_per_operator.values())),
        "families_per_operator_max": int(max(fam_per_operator.values())),
        "surfaces_per_family_min": int(min(surface_per_family.values())),
        "surfaces_per_family_max": int(max(surface_per_family.values())),
        "expected_same_concept_candidates": int(df["operator_star"].nunique()),
        "expected_same_operator_candidates": int(df["concept_star"].nunique()),
        "pass_nondegenerate": (
            min(fam_per_concept.values()) >= 2
            and min(fam_per_operator.values()) >= 2
            and min(surface_per_family.values()) >= 2
        ),
    }


# =============================================================================
# Retrieval audit, adapted from SEM-5A.1
# =============================================================================

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


@dataclass
class FeatureBlock:
    name: str
    cols: List[str]
    is_text: bool = False
    anonymize_text: bool = False


def merge_dataset_features(dataset: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if "row_id" in dataset.columns and "row_id" in features.columns:
        merged = dataset.merge(features, on="row_id", how="inner", suffixes=("", "_feat"))
    else:
        if len(dataset) != len(features):
            raise ValueError("Dataset/features row count differs and no row_id shared.")
        merged = pd.concat([dataset.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
        merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged


def build_feature_blocks(df: pd.DataFrame) -> List[FeatureBlock]:
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

    blocks = [
        FeatureBlock("TextQuery", [], is_text=True, anonymize_text=False),
        FeatureBlock("TextQuery_Anonymized", [], is_text=True, anonymize_text=True),
    ]
    if init_cols:
        blocks.append(FeatureBlock("InitQuery_L0_6", init_cols))
    if shape_cols:
        blocks.append(FeatureBlock("ShapeQuery_L7_19", shape_cols))
    if commit_cols:
        blocks.append(FeatureBlock("CommitQuery_L23_25", commit_cols))
    return blocks


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


def evaluate_numeric_block(
    df: pd.DataFrame,
    block: FeatureBlock,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    retrieval_mode: str,
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

    x_train = train[block.cols].to_numpy(dtype=float)
    x_test = test[block.cols].to_numpy(dtype=float)
    x_train, x_test = zscore_train_test(x_train, x_test)

    centers, families, center_meta = build_centers(x_train, y_train, train)

    sim = cosine_similarity(x_test, centers)
    test_meta = test.copy()
    test_meta["__truth_family__"] = y_test
    ranked, masked, valid_rows, candidate_counts = rank_with_candidate_restriction(
        sim, families, center_meta, test_meta, retrieval_mode
    )

    valid = np.array(valid_rows, dtype=bool)
    y_eval = y_test[valid]
    ranked_eval = ranked[valid]
    masked_eval = masked[valid]
    kept_eval_idx = kept_test_idx[valid]
    candidate_counts_eval = np.array(candidate_counts)[valid]

    if len(y_eval) == 0:
        raise ValueError("No valid rows after restriction.")

    order = np.argsort(-masked_eval, axis=1)
    pred = ranked_eval[:, 0].astype(str)
    ent = row_softmax_entropy(masked_eval)

    metrics = {
        "query_type": block.name,
        "retrieval_mode": retrieval_mode,
        "n_train": int(len(train_idx)),
        "n_test_raw": int(len(test_idx)),
        "n_test_eval": int(len(y_eval)),
        "n_families_train": int(len(families)),
        "feature_dim": int(len(block.cols)),
        "candidate_count_mean": float(np.mean(candidate_counts_eval)),
        "candidate_count_min": int(np.min(candidate_counts_eval)),
        "candidate_count_max": int(np.max(candidate_counts_eval)),
        "acc_top1": float(accuracy_score(y_eval, pred)),
        "macro_f1": safe_macro_f1(list(y_eval), list(pred)),
        "mrr": mrr_score(list(y_eval), ranked_eval),
        "entropy_mean": float(np.mean(ent)),
        "entropy_std": float(np.std(ent)),
    }
    for k in TOP_KS:
        metrics[f"recall_at_{k}"] = recall_at_k(list(y_eval), ranked_eval, min(k, ranked_eval.shape[1]))

    details = pd.DataFrame({
        "query_type": block.name,
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


def evaluate_text_block(
    df: pd.DataFrame,
    block: FeatureBlock,
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

    train_text = train["prompt"].fillna("").astype(str).tolist()
    test_text = test["prompt"].fillna("").astype(str).tolist()

    if block.anonymize_text:
        train_text = [anonymize_prompt_text(t, replacements) for t in train_text]
        test_text = [anonymize_prompt_text(t, replacements) for t in test_text]

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
        sub = train.loc[rows]
        meta_rows.append({
            "family": fam,
            "concept": str(sub["concept_star"].mode().iloc[0]),
            "operator": str(sub["operator_star"].mode().iloc[0]),
        })
    centers = np.vstack(centers)
    families = np.array(families, dtype=object)
    center_meta = pd.DataFrame(meta_rows)

    sim = cosine_similarity(x_test, centers)
    test_meta = test.copy()
    test_meta["__truth_family__"] = y_test
    ranked, masked, valid_rows, candidate_counts = rank_with_candidate_restriction(
        sim, families, center_meta, test_meta, retrieval_mode
    )

    valid = np.array(valid_rows, dtype=bool)
    y_eval = y_test[valid]
    ranked_eval = ranked[valid]
    masked_eval = masked[valid]
    kept_eval_idx = kept_test_idx[valid]
    candidate_counts_eval = np.array(candidate_counts)[valid]

    if len(y_eval) == 0:
        raise ValueError("No valid rows after restriction.")

    order = np.argsort(-masked_eval, axis=1)
    pred = ranked_eval[:, 0].astype(str)
    ent = row_softmax_entropy(masked_eval)

    metrics = {
        "query_type": block.name,
        "retrieval_mode": retrieval_mode,
        "n_train": int(len(train_idx)),
        "n_test_raw": int(len(test_idx)),
        "n_test_eval": int(len(y_eval)),
        "n_families_train": int(len(families)),
        "feature_dim": int(x_train.shape[1]),
        "candidate_count_mean": float(np.mean(candidate_counts_eval)),
        "candidate_count_min": int(np.min(candidate_counts_eval)),
        "candidate_count_max": int(np.max(candidate_counts_eval)),
        "acc_top1": float(accuracy_score(y_eval, pred)),
        "macro_f1": safe_macro_f1(list(y_eval), list(pred)),
        "mrr": mrr_score(list(y_eval), ranked_eval),
        "entropy_mean": float(np.mean(ent)),
        "entropy_std": float(np.std(ent)),
    }
    for k in TOP_KS:
        metrics[f"recall_at_{k}"] = recall_at_k(list(y_eval), ranked_eval, min(k, ranked_eval.shape[1]))

    details = pd.DataFrame({
        "query_type": block.name,
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


def evaluate_block(
    df: pd.DataFrame,
    block: FeatureBlock,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    retrieval_mode: str,
    replacements: Dict[str, str],
) -> Tuple[Dict, pd.DataFrame]:
    if block.is_text:
        return evaluate_text_block(df, block, train_idx, test_idx, retrieval_mode, replacements)
    return evaluate_numeric_block(df, block, train_idx, test_idx, retrieval_mode)


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

    summary = metrics_df.groupby(["retrieval_mode", "query_type"]).agg(agg_map)
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    return summary.reset_index().sort_values(["retrieval_mode", "acc_top1_mean"], ascending=[True, False])


def hard_negative_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode, sub in summary.groupby("retrieval_mode"):
        table = sub.set_index("query_type").to_dict(orient="index")
        shape_keys = [k for k in table if k.startswith("ShapeQuery")]
        if not shape_keys:
            continue
        shape = max(shape_keys, key=lambda k: table[k].get("acc_top1_mean", -999))

        row = {
            "retrieval_mode": mode,
            "shape_query": shape,
            "shape_acc": table[shape].get("acc_top1_mean"),
            "shape_r3": table[shape].get("recall_at_3_mean"),
            "shape_entropy": table[shape].get("entropy_mean_mean"),
            "candidate_count": table[shape].get("candidate_count_mean_mean"),
        }
        for q, prefix in [
            ("TextQuery", "text"),
            ("TextQuery_Anonymized", "anon_text"),
            ("InitQuery_L0_6", "init"),
            ("CommitQuery_L23_25", "commit"),
        ]:
            if q in table:
                row[f"{prefix}_acc"] = table[q].get("acc_top1_mean")
                row[f"shape_minus_{prefix}"] = row["shape_acc"] - row[f"{prefix}_acc"]
        rows.append(row)
    return pd.DataFrame(rows)


def make_verdict(hard: pd.DataFrame) -> Dict:
    verdict = {
        "status": "UNDETERMINED",
        "pass_lite": False,
        "pass_strong": False,
        "notes": [],
    }
    if hard.empty:
        verdict["status"] = "NO_HARD_SUMMARY"
        return verdict

    rows = {r["retrieval_mode"]: r for _, r in hard.iterrows()}

    def ok(mode: str) -> bool:
        r = rows.get(mode)
        if r is None:
            return False
        return (
            r.get("shape_acc", 0) >= 0.70
            and r.get("shape_r3", 0) >= 0.90
            and r.get("shape_minus_anon_text", -999) > 0.05
            and r.get("candidate_count", 0) >= 3
        )

    sc = ok("same_concept")
    so = ok("same_operator")
    scdo = ok("same_concept_different_operator")
    sodc = ok("same_operator_different_concept")

    for mode in ["same_concept", "same_operator", "same_concept_different_operator", "same_operator_different_concept"]:
        r = rows.get(mode)
        if r is None:
            continue
        if r.get("shape_r3", 0) >= 0.90:
            verdict["notes"].append(f"{mode}: ShapeQuery Recall@3 >= 0.90.")
        if r.get("shape_minus_init", -999) > 0.10:
            verdict["notes"].append(f"{mode}: ShapeQuery exceeds InitQuery by >10 points.")
        if r.get("shape_minus_anon_text", -999) > 0.05:
            verdict["notes"].append(f"{mode}: ShapeQuery exceeds anonymized TextQuery by >5 points.")

    if sc or so or scdo or sodc:
        verdict["pass_lite"] = True

    if sc and so:
        verdict["pass_strong"] = True
        verdict["notes"].append("ShapeQuery passes both non-degenerate same-concept and same-operator restrictions.")

    verdict["status"] = "PASS_STRONG" if verdict["pass_strong"] else ("PASS_LITE" if verdict["pass_lite"] else "NO_PASS_OR_INSUFFICIENT")
    return verdict


def run_audit() -> None:
    dataset = pd.read_csv(DATASET_PATH)
    features = pd.read_csv(FEATURES_PATH)
    df = merge_dataset_features(dataset, features)

    blocks = build_feature_blocks(df)
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
        "blocks": [
            {"name": b.name, "is_text": b.is_text, "anonymize_text": b.anonymize_text, "n_cols": len(b.cols), "cols_preview": b.cols[:24]}
            for b in blocks
        ],
        "retrieval_modes": retrieval_modes,
        "split_count": len(splits),
        "dataset_audit": audit_dataset(dataset),
    }
    (OUTPUT_DIR / "sem5a2_column_audit.json").write_text(json.dumps(column_audit, ensure_ascii=False, indent=2), encoding="utf-8")

    all_metrics = []
    all_details = []

    for split_name, tr, te in splits:
        for mode in retrieval_modes:
            for block in blocks:
                try:
                    metrics, details = evaluate_block(df, block, tr, te, mode, replacements)
                    metrics["split"] = split_name
                    details["split"] = split_name
                    all_metrics.append(metrics)
                    all_details.append(details)
                    print(
                        f"[OK] {split_name} / {mode} / {block.name}: "
                        f"acc={metrics['acc_top1']:.3f}, r@3={metrics.get('recall_at_3', np.nan):.3f}, "
                        f"cand={metrics['candidate_count_mean']:.1f}"
                    )
                except Exception as exc:
                    print(f"[WARN] {split_name} / {mode} / {block.name} failed: {exc}")

    if not all_metrics:
        raise RuntimeError("No audit metrics produced.")

    metrics_df = pd.DataFrame(all_metrics)
    details_df = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()

    metrics_path = OUTPUT_DIR / "sem5a2_retrieval_by_split.csv"
    details_path = OUTPUT_DIR / "sem5a2_retrieval_details.csv"
    summary_path = OUTPUT_DIR / "sem5a2_summary_by_query_and_mode.csv"
    hard_path = OUTPUT_DIR / "sem5a2_hard_negative_summary.csv"
    verdict_path = OUTPUT_DIR / "sem5a2_verdict.json"

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    details_df.to_csv(details_path, index=False, encoding="utf-8-sig")

    summary = summarize_metrics(metrics_df)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    hard = hard_negative_summary(summary)
    hard.to_csv(hard_path, index=False, encoding="utf-8-sig")

    verdict = make_verdict(hard)
    verdict["output_files"] = {
        "retrieval_by_split": str(metrics_path),
        "retrieval_details": str(details_path),
        "summary_by_query_and_mode": str(summary_path),
        "hard_negative_summary": str(hard_path),
        "column_audit": str(OUTPUT_DIR / "sem5a2_column_audit.json"),
    }
    verdict_path.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== SEM-5A.2 HARD NEGATIVE SUMMARY ===")
    print(hard.to_string(index=False))
    print("\n=== SEM-5A.2 VERDICT ===")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    df = build_dataset()
    df.to_csv(DATASET_PATH, index=False, encoding="utf-8-sig")

    audit = audit_dataset(df)
    (OUTPUT_DIR / "sem5a2_dataset_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== SEM-5A.2 DATASET BUILT ===")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"\nDataset written to: {DATASET_PATH}")

    if not FEATURES_PATH.exists():
        print("\n[STOP] Feature file not found.")
        print(f"Run your existing feature extraction pipeline on {DATASET_PATH}")
        print(f"and save features to: {FEATURES_PATH}")
        return

    print(f"\nFeature file found: {FEATURES_PATH}")
    print("Running SEM-5A.2 hard-negative retrieval audit...\n")
    run_audit()


if __name__ == "__main__":
    main()
