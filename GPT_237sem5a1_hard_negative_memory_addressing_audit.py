# -*- coding: utf-8 -*-
"""
SEM-5A.1: Hard Negative Memory Addressing Audit
===============================================

Why this script exists
----------------------
SEM-5A found:
  TextQuery > ShapeQuery
  ShapeQuery > InitQuery

This does NOT cleanly falsify trajectory-query memory addressing, because the
current SEM dataset may leak SeedFamily through lexical concept/operator cues
in the natural-language prompt.

SEM-5A.1 therefore runs harder retrieval settings:

1. Full retrieval
   Same as SEM-5A, for continuity.

2. Same-concept restricted retrieval
   For each test sample, rank only memory units whose concept_star equals the
   test concept. This removes "concept lookup" and forces operator / trajectory
   disambiguation.

3. Same-operator restricted retrieval
   For each test sample, rank only memory units whose operator_star equals the
   test operator. This removes "operator lookup" and forces concept / trajectory
   disambiguation.

4. Anonymized TextQuery
   Replaces visible concept/operator strings with placeholders before TF-IDF.
   This weakens surface lexical leakage.

5. Leave-concept and leave-operator diagnostics when possible
   These are harder but may be partially impossible if labels disappear from
   train splits. The script reports feasible splits only.

Main target
-----------
If ShapeQuery_L7_19 remains strong under candidate-restricted retrieval while
TextQuery / AnonymizedTextQuery drops, SEM-5A.1 supports:

  Prompt_text -> Query_traj -> MemoryUnit_F

more strongly than SEM-5A.

Inputs
------
Default paths:
  sem3a_outputs/sem3a_dataset.csv
  sem3a_outputs/sem3a_features.csv

Outputs
-------
  sem5a1_outputs/sem5a1_retrieval_by_split.csv
  sem5a1_outputs/sem5a1_summary_by_query_and_mode.csv
  sem5a1_outputs/sem5a1_hard_negative_summary.csv
  sem5a1_outputs/sem5a1_verdict.json
  sem5a1_outputs/sem5a1_column_audit.json

Run
---
  python sem5a1_hard_negative_memory_addressing_audit.py
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
from sklearn.model_selection import GroupKFold, StratifiedKFold, LeaveOneGroupOut


# ---------------------------------------------------------------------------
# Hardcoded defaults
# ---------------------------------------------------------------------------

DATASET_PATH = Path("sem3a_outputs/sem3a_dataset.csv")
FEATURES_PATH = Path("sem3a_outputs/sem3a_features.csv")
OUTPUT_DIR = Path("sem5a1_outputs")

RANDOM_SEED = 42
TOP_KS = (1, 3, 5)
SOFTMAX_TEMPERATURE = 0.10
MAX_SPLITS_PER_GROUP_TYPE = 12  # avoids very long leave-one-concept/operator loops


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_safely(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path.resolve()}")
    return pd.read_csv(path)


def normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    norm_to_orig = {normalize_name(c): c for c in columns}
    for cand in candidates:
        key = normalize_name(cand)
        if key in norm_to_orig:
            return norm_to_orig[key]
    return None


def find_by_keywords(columns: Iterable[str], include: Iterable[str], exclude: Iterable[str] = ()) -> Optional[str]:
    include = [x.lower() for x in include]
    exclude = [x.lower() for x in exclude]
    for col in columns:
        name = normalize_name(col)
        if all(k in name for k in include) and not any(k in name for k in exclude):
            return col
    return None


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
    if sim.shape[1] <= 0:
        return np.zeros(sim.shape[0])
    z = sim / max(temperature, 1e-8)
    z = z - np.max(z, axis=1, keepdims=True)
    p = np.exp(z)
    p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
    h = -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1)
    if sim.shape[1] > 1:
        h = h / math.log(sim.shape[1])
    return h


def recall_at_k(y_true: List[str], ranked_labels: np.ndarray, k: int) -> float:
    if len(y_true) == 0:
        return float("nan")
    hits = 0
    for truth, row in zip(y_true, ranked_labels):
        if truth in row[:k]:
            hits += 1
    return hits / len(y_true)


def mrr_score(y_true: List[str], ranked_labels: np.ndarray) -> float:
    vals = []
    for truth, row in zip(y_true, ranked_labels):
        pos = np.where(row == truth)[0]
        vals.append(1.0 / (pos[0] + 1) if len(pos) else 0.0)
    return float(np.mean(vals)) if vals else float("nan")


def safe_macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    if len(y_true) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def safe_acc(y_true: List[str], y_pred: List[str]) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(accuracy_score(y_true, y_pred))


# ---------------------------------------------------------------------------
# Columns and feature blocks
# ---------------------------------------------------------------------------

@dataclass
class ColumnMap:
    id_col: Optional[str]
    prompt_col: str
    family_col: str
    concept_col: Optional[str]
    operator_col: Optional[str]
    surface_col: Optional[str]
    prompt_type_col: Optional[str]
    condition_col: Optional[str]


@dataclass
class FeatureBlock:
    name: str
    cols: List[str]
    is_text: bool = False
    anonymize_text: bool = False


def infer_columns(df: pd.DataFrame) -> ColumnMap:
    cols = list(df.columns)

    id_col = first_existing(cols, [
        "prompt_id", "sample_id", "id", "uid", "case_id", "example_id", "row_id"
    ])

    prompt_col = first_existing(cols, [
        "prompt", "text", "prompt_text", "input", "input_text", "query"
    ]) or find_by_keywords(cols, ["prompt"]) or find_by_keywords(cols, ["text"])

    family_col = first_existing(cols, [
        "seed_family", "SeedFamily", "family", "family_id", "memory_family",
        "MemoryFamily", "seedfamily", "label", "target_family"
    ]) or find_by_keywords(cols, ["family"]) or find_by_keywords(cols, ["seed"])

    concept_col = first_existing(cols, [
        "concept_star", "concept", "Concept", "topic", "entity", "subject"
    ]) or find_by_keywords(cols, ["concept"]) or find_by_keywords(cols, ["topic"])

    operator_col = first_existing(cols, [
        "operator_star", "operator", "OperatorID", "operator_id", "policy_id",
        "PolicyID", "constraint", "task_type", "task"
    ]) or find_by_keywords(cols, ["operator"]) or find_by_keywords(cols, ["policy"])

    surface_col = first_existing(cols, [
        "surface_id", "surface", "surface_group", "surface_family", "template",
        "template_id", "paraphrase_group", "style"
    ]) or find_by_keywords(cols, ["surface"]) or find_by_keywords(cols, ["template"])

    prompt_type_col = first_existing(cols, [
        "prompt_type", "type", "heldout_type", "query_type", "eval_type", "row_type"
    ]) or find_by_keywords(cols, ["prompt", "type"])

    condition_col = first_existing(cols, [
        "condition", "Condition", "mechanism", "family_type", "case_type"
    ]) or find_by_keywords(cols, ["condition"]) or find_by_keywords(cols, ["mechanism"])

    if prompt_col is None:
        raise ValueError("Could not infer prompt/text column. Please add prompt or text.")
    if family_col is None:
        raise ValueError("Could not infer family label column. Please add seed_family / family / label.")

    return ColumnMap(
        id_col=id_col,
        prompt_col=prompt_col,
        family_col=family_col,
        concept_col=concept_col,
        operator_col=operator_col,
        surface_col=surface_col,
        prompt_type_col=prompt_type_col,
        condition_col=condition_col,
    )


def merge_dataset_features(dataset: pd.DataFrame, features: pd.DataFrame, cmap: ColumnMap) -> pd.DataFrame:
    if cmap.id_col and cmap.id_col in dataset.columns and cmap.id_col in features.columns:
        merged = dataset.merge(features, on=cmap.id_col, how="inner", suffixes=("", "_feat"))
    else:
        if len(dataset) != len(features):
            raise ValueError(
                "No shared id column found and dataset/features lengths differ. "
                "Add row_id/prompt_id/sample_id to both files."
            )
        merged = pd.concat(
            [dataset.reset_index(drop=True), features.reset_index(drop=True)],
            axis=1,
        )
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


# ---------------------------------------------------------------------------
# Text anonymization
# ---------------------------------------------------------------------------

def build_replacement_terms(df: pd.DataFrame, cmap: ColumnMap) -> Dict[str, str]:
    replacements = {}

    if cmap.concept_col and cmap.concept_col in df.columns:
        vals = sorted(set(df[cmap.concept_col].dropna().astype(str)), key=len, reverse=True)
        for v in vals:
            if v.strip():
                replacements[v] = "CONCEPT"

    if cmap.operator_col and cmap.operator_col in df.columns:
        vals = sorted(set(df[cmap.operator_col].dropna().astype(str)), key=len, reverse=True)
        for v in vals:
            if v.strip():
                replacements[v] = "OPERATOR"

    if cmap.family_col and cmap.family_col in df.columns:
        vals = sorted(set(df[cmap.family_col].dropna().astype(str)), key=len, reverse=True)
        for v in vals:
            if v.strip():
                replacements[v] = "FAMILY"

    return replacements


def anonymize_prompt_text(text: str, replacements: Dict[str, str]) -> str:
    out = str(text)

    # Replace exact known labels, case-insensitive.
    for term, repl in replacements.items():
        term = str(term).strip()
        if not term or len(term) < 2:
            continue
        out = re.sub(re.escape(term), repl, out, flags=re.IGNORECASE)

        # Common normalized variants.
        variants = {
            term.replace("_", " "),
            term.replace("-", " "),
            term.replace("_", "-"),
            term.replace(" ", "_"),
            term.replace(" ", "-"),
        }
        for v in variants:
            if v and len(v) >= 2:
                out = re.sub(re.escape(v), repl, out, flags=re.IGNORECASE)

    # Replace obvious artificial labels if dataset uses them.
    out = re.sub(r"\bconcept[_\-\s]*[a-z0-9]+\b", "CONCEPT", out, flags=re.IGNORECASE)
    out = re.sub(r"\boperator[_\-\s]*[a-z0-9]+\b", "OPERATOR", out, flags=re.IGNORECASE)
    out = re.sub(r"\bseed[_\-\s]*family[_\-\s]*[a-z0-9]+\b", "FAMILY", out, flags=re.IGNORECASE)

    return out


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def make_standard_splits(df: pd.DataFrame, cmap: ColumnMap, n_splits: int = 5) -> List[Tuple[str, str, np.ndarray, np.ndarray]]:
    y = df[cmap.family_col].astype(str).values
    splits = []

    if cmap.surface_col and cmap.surface_col in df.columns:
        groups = df[cmap.surface_col].astype(str).values
        unique_groups = np.unique(groups)
        if len(unique_groups) >= 2:
            if len(unique_groups) >= n_splits:
                gkf = GroupKFold(n_splits=min(n_splits, len(unique_groups)))
                for i, (tr, te) in enumerate(gkf.split(df, y, groups)):
                    splits.append(("leave_surface", f"leave_surface_group_{i}", tr, te))
            else:
                idx = np.arange(len(df))
                for g in unique_groups:
                    te = idx[groups == g]
                    tr = idx[groups != g]
                    if len(te) and len(tr):
                        splits.append(("leave_surface", f"leave_surface_{g}", tr, te))

    if not splits:
        class_counts = pd.Series(y).value_counts()
        min_class = int(class_counts.min())
        if len(class_counts) >= 2 and min_class >= 2:
            skf = StratifiedKFold(
                n_splits=min(n_splits, min_class),
                shuffle=True,
                random_state=RANDOM_SEED,
            )
            for i, (tr, te) in enumerate(skf.split(df, y)):
                splits.append(("stratified", f"stratified_{i}", tr, te))
        else:
            rng = np.random.default_rng(RANDOM_SEED)
            idx = np.arange(len(df))
            rng.shuffle(idx)
            cut = max(1, int(0.8 * len(idx)))
            splits.append(("random", "random_80_20", idx[:cut], idx[cut:]))

    return splits


def make_leave_group_splits(
    df: pd.DataFrame,
    cmap: ColumnMap,
    group_col: Optional[str],
    split_type: str,
    max_splits: int = MAX_SPLITS_PER_GROUP_TYPE,
) -> List[Tuple[str, str, np.ndarray, np.ndarray]]:
    if group_col is None or group_col not in df.columns:
        return []

    groups = df[group_col].astype(str).values
    y = df[cmap.family_col].astype(str).values
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return []

    splits = []
    logo = LeaveOneGroupOut()
    for i, (tr, te) in enumerate(logo.split(df, y, groups)):
        if i >= max_splits:
            break
        gname = str(groups[te][0])
        splits.append((split_type, f"{split_type}_{gname}", tr, te))
    return splits


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def build_centers(
    x_train: np.ndarray,
    y_train: np.ndarray,
    meta_train: pd.DataFrame,
    cmap: ColumnMap,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    families = sorted(pd.Series(y_train).unique().astype(str).tolist())
    centers = []
    meta_rows = []

    for fam in families:
        mask = y_train == fam
        centers.append(x_train[mask].mean(axis=0))

        row = {"family": fam}
        sub = meta_train.loc[mask]
        if cmap.concept_col and cmap.concept_col in sub.columns:
            row["concept"] = str(sub[cmap.concept_col].mode().iloc[0]) if len(sub[cmap.concept_col].mode()) else ""
        if cmap.operator_col and cmap.operator_col in sub.columns:
            row["operator"] = str(sub[cmap.operator_col].mode().iloc[0]) if len(sub[cmap.operator_col].mode()) else ""
        meta_rows.append(row)

    return np.vstack(centers), np.array(families, dtype=object), pd.DataFrame(meta_rows)


def rank_with_candidate_restriction(
    sim: np.ndarray,
    families: np.ndarray,
    center_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    cmap: ColumnMap,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, List[bool], List[int]]:
    """
    Returns ranked labels and masked sim for each row.
    mode:
      full
      same_concept
      same_operator
      same_concept_different_operator
      same_operator_different_concept
    """
    masked = sim.copy()
    valid_rows = []
    candidate_counts = []

    for i in range(sim.shape[0]):
        mask = np.ones(sim.shape[1], dtype=bool)

        if mode in {"same_concept", "same_concept_different_operator"}:
            if cmap.concept_col is None or "concept" not in center_meta.columns or cmap.concept_col not in test_meta.columns:
                mask[:] = False
            else:
                cval = str(test_meta.iloc[i][cmap.concept_col])
                mask &= (center_meta["concept"].astype(str).values == cval)

        if mode in {"same_operator", "same_operator_different_concept"}:
            if cmap.operator_col is None or "operator" not in center_meta.columns or cmap.operator_col not in test_meta.columns:
                mask[:] = False
            else:
                oval = str(test_meta.iloc[i][cmap.operator_col])
                mask &= (center_meta["operator"].astype(str).values == oval)

        if mode == "same_concept_different_operator":
            if cmap.operator_col and "operator" in center_meta.columns and cmap.operator_col in test_meta.columns:
                oval = str(test_meta.iloc[i][cmap.operator_col])
                # Keep truth family even if same operator; otherwise target may vanish.
                truth = str(test_meta.iloc[i]["__truth_family__"])
                mask &= ((center_meta["operator"].astype(str).values != oval) | (center_meta["family"].astype(str).values == truth))

        if mode == "same_operator_different_concept":
            if cmap.concept_col and "concept" in center_meta.columns and cmap.concept_col in test_meta.columns:
                cval = str(test_meta.iloc[i][cmap.concept_col])
                truth = str(test_meta.iloc[i]["__truth_family__"])
                mask &= ((center_meta["concept"].astype(str).values != cval) | (center_meta["family"].astype(str).values == truth))

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
    cmap: ColumnMap,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    retrieval_mode: str,
) -> Tuple[Dict, pd.DataFrame]:
    train = df.iloc[train_idx].copy().reset_index(drop=True)
    test = df.iloc[test_idx].copy().reset_index(drop=True)

    y_train = train[cmap.family_col].astype(str).values
    y_test = test[cmap.family_col].astype(str).values

    # Remove test examples whose truth family does not exist in train.
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

    centers, families, center_meta = build_centers(x_train, y_train, train, cmap)

    sim = cosine_similarity(x_test, centers)
    test_meta = test.copy()
    test_meta["__truth_family__"] = y_test
    ranked, masked, valid_rows, candidate_counts = rank_with_candidate_restriction(
        sim, families, center_meta, test_meta, cmap, retrieval_mode
    )

    valid = np.array(valid_rows, dtype=bool)
    if not valid.any():
        raise ValueError(f"No valid candidate rows under retrieval mode: {retrieval_mode}")

    y_eval = y_test[valid]
    ranked_eval = ranked[valid]
    masked_eval = masked[valid]
    kept_eval_idx = kept_test_idx[valid]
    candidate_counts_eval = np.array(candidate_counts)[valid]
    pred = ranked_eval[:, 0].astype(str)

    ent = row_softmax_entropy(masked_eval)

    out = {
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
        "acc_top1": safe_acc(list(y_eval), list(pred)),
        "macro_f1": safe_macro_f1(list(y_eval), list(pred)),
        "mrr": mrr_score(list(y_eval), ranked_eval),
        "entropy_mean": float(np.mean(ent)),
        "entropy_std": float(np.std(ent)),
    }
    for k in TOP_KS:
        out[f"recall_at_{k}"] = recall_at_k(list(y_eval), ranked_eval, k=min(k, ranked_eval.shape[1]))

    details = pd.DataFrame({
        "query_type": block.name,
        "retrieval_mode": retrieval_mode,
        "truth_family": y_eval,
        "pred_family": pred,
        "top1_sim": masked_eval[np.arange(len(y_eval)), np.argsort(-masked_eval, axis=1)[:, 0]],
        "entropy": ent,
        "candidate_count": candidate_counts_eval,
        "test_row_index": kept_eval_idx,
    })

    for c, outc in [
        (cmap.prompt_type_col, "prompt_type"),
        (cmap.condition_col, "condition"),
        (cmap.concept_col, "concept"),
        (cmap.operator_col, "operator"),
        (cmap.surface_col, "surface"),
    ]:
        if c and c in test.columns:
            details[outc] = test.loc[valid, c].astype(str).values

    for k in TOP_KS:
        kk = min(k, ranked_eval.shape[1])
        details[f"hit_at_{k}"] = [truth in set(row[:kk]) for truth, row in zip(y_eval, ranked_eval)]

    return out, details


def evaluate_text_block(
    df: pd.DataFrame,
    block: FeatureBlock,
    cmap: ColumnMap,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    retrieval_mode: str,
    replacements: Dict[str, str],
) -> Tuple[Dict, pd.DataFrame]:
    train = df.iloc[train_idx].copy().reset_index(drop=True)
    test = df.iloc[test_idx].copy().reset_index(drop=True)

    y_train = train[cmap.family_col].astype(str).values
    y_test = test[cmap.family_col].astype(str).values

    train_fams = set(y_train)
    keep = np.array([y in train_fams for y in y_test], dtype=bool)
    test = test.loc[keep].reset_index(drop=True)
    y_test = y_test[keep]
    kept_test_idx = np.array(test_idx)[keep]

    if len(y_test) == 0:
        raise ValueError("No test rows with truth family present in train.")

    train_text = train[cmap.prompt_col].fillna("").astype(str).tolist()
    test_text = test[cmap.prompt_col].fillna("").astype(str).tolist()

    if block.anonymize_text:
        train_text = [anonymize_prompt_text(t, replacements) for t in train_text]
        test_text = [anonymize_prompt_text(t, replacements) for t in test_text]

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1,
        max_features=20000,
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
        row = {"family": fam}
        if cmap.concept_col and cmap.concept_col in sub.columns:
            row["concept"] = str(sub[cmap.concept_col].mode().iloc[0]) if len(sub[cmap.concept_col].mode()) else ""
        if cmap.operator_col and cmap.operator_col in sub.columns:
            row["operator"] = str(sub[cmap.operator_col].mode().iloc[0]) if len(sub[cmap.operator_col].mode()) else ""
        meta_rows.append(row)
    centers = np.vstack(centers)
    center_meta = pd.DataFrame(meta_rows)
    families = np.array(families, dtype=object)

    sim = cosine_similarity(x_test, centers)
    test_meta = test.copy()
    test_meta["__truth_family__"] = y_test
    ranked, masked, valid_rows, candidate_counts = rank_with_candidate_restriction(
        sim, families, center_meta, test_meta, cmap, retrieval_mode
    )

    valid = np.array(valid_rows, dtype=bool)
    if not valid.any():
        raise ValueError(f"No valid candidate rows under retrieval mode: {retrieval_mode}")

    y_eval = y_test[valid]
    ranked_eval = ranked[valid]
    masked_eval = masked[valid]
    kept_eval_idx = kept_test_idx[valid]
    candidate_counts_eval = np.array(candidate_counts)[valid]
    pred = ranked_eval[:, 0].astype(str)

    ent = row_softmax_entropy(masked_eval)

    out = {
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
        "acc_top1": safe_acc(list(y_eval), list(pred)),
        "macro_f1": safe_macro_f1(list(y_eval), list(pred)),
        "mrr": mrr_score(list(y_eval), ranked_eval),
        "entropy_mean": float(np.mean(ent)),
        "entropy_std": float(np.std(ent)),
    }
    for k in TOP_KS:
        out[f"recall_at_{k}"] = recall_at_k(list(y_eval), ranked_eval, k=min(k, ranked_eval.shape[1]))

    details = pd.DataFrame({
        "query_type": block.name,
        "retrieval_mode": retrieval_mode,
        "truth_family": y_eval,
        "pred_family": pred,
        "top1_sim": masked_eval[np.arange(len(y_eval)), np.argsort(-masked_eval, axis=1)[:, 0]],
        "entropy": ent,
        "candidate_count": candidate_counts_eval,
        "test_row_index": kept_eval_idx,
    })

    for c, outc in [
        (cmap.prompt_type_col, "prompt_type"),
        (cmap.condition_col, "condition"),
        (cmap.concept_col, "concept"),
        (cmap.operator_col, "operator"),
        (cmap.surface_col, "surface"),
    ]:
        if c and c in test.columns:
            details[outc] = test.loc[valid, c].astype(str).values

    for k in TOP_KS:
        kk = min(k, ranked_eval.shape[1])
        details[f"hit_at_{k}"] = [truth in set(row[:kk]) for truth, row in zip(y_eval, ranked_eval)]

    return out, details


def evaluate_block(
    df: pd.DataFrame,
    block: FeatureBlock,
    cmap: ColumnMap,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    retrieval_mode: str,
    replacements: Dict[str, str],
) -> Tuple[Dict, pd.DataFrame]:
    if block.is_text:
        return evaluate_text_block(df, block, cmap, train_idx, test_idx, retrieval_mode, replacements)
    return evaluate_numeric_block(df, block, cmap, train_idx, test_idx, retrieval_mode)


# ---------------------------------------------------------------------------
# Summary and verdict
# ---------------------------------------------------------------------------

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
    summary = summary.reset_index().sort_values(["retrieval_mode", "acc_top1_mean"], ascending=[True, False])
    return summary


def hard_negative_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode, sub in summary.groupby("retrieval_mode"):
        table = sub.set_index("query_type").to_dict(orient="index")
        shape_keys = [k for k in table if k.startswith("ShapeQuery")]
        init_keys = [k for k in table if k.startswith("InitQuery")]
        commit_keys = [k for k in table if k.startswith("CommitQuery")]

        if not shape_keys:
            continue
        shape = max(shape_keys, key=lambda k: table[k].get("acc_top1_mean", -999))
        text = "TextQuery" if "TextQuery" in table else None
        anon = "TextQuery_Anonymized" if "TextQuery_Anonymized" in table else None
        init = max(init_keys, key=lambda k: table[k].get("acc_top1_mean", -999)) if init_keys else None
        commit = max(commit_keys, key=lambda k: table[k].get("acc_top1_mean", -999)) if commit_keys else None

        row = {
            "retrieval_mode": mode,
            "shape_query": shape,
            "shape_acc": table[shape].get("acc_top1_mean"),
            "shape_r3": table[shape].get("recall_at_3_mean"),
            "shape_entropy": table[shape].get("entropy_mean_mean"),
        }
        if text:
            row["text_acc"] = table[text].get("acc_top1_mean")
            row["shape_minus_text"] = row["shape_acc"] - row["text_acc"]
        if anon:
            row["anon_text_acc"] = table[anon].get("acc_top1_mean")
            row["shape_minus_anon_text"] = row["shape_acc"] - row["anon_text_acc"]
        if init:
            row["init_acc"] = table[init].get("acc_top1_mean")
            row["shape_minus_init"] = row["shape_acc"] - row["init_acc"]
        if commit:
            row["commit_acc"] = table[commit].get("acc_top1_mean")
            row["shape_minus_commit"] = row["shape_acc"] - row["commit_acc"]

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

    # Key modes.
    mode_rows = {r["retrieval_mode"]: r for _, r in hard.iterrows()}

    full = mode_rows.get("full")
    same_concept = mode_rows.get("same_concept")
    same_operator = mode_rows.get("same_operator")

    positive_modes = []

    for mode_name in ["same_concept", "same_operator", "same_concept_different_operator", "same_operator_different_concept"]:
        r = mode_rows.get(mode_name)
        if r is None:
            continue
        shape_acc = r.get("shape_acc", np.nan)
        shape_r3 = r.get("shape_r3", np.nan)
        shape_minus_anon = r.get("shape_minus_anon_text", np.nan)
        shape_minus_init = r.get("shape_minus_init", np.nan)

        if pd.notna(shape_acc) and pd.notna(shape_minus_anon):
            if shape_acc >= 0.70 and shape_minus_anon > 0.05:
                positive_modes.append(mode_name)

        if pd.notna(shape_minus_init) and shape_minus_init > 0.10:
            verdict["notes"].append(f"{mode_name}: ShapeQuery exceeds InitQuery by > 10 points.")

        if pd.notna(shape_r3) and shape_r3 >= 0.90:
            verdict["notes"].append(f"{mode_name}: ShapeQuery Recall@3 >= 0.90.")

    if positive_modes:
        verdict["pass_lite"] = True
        verdict["notes"].append(
            "ShapeQuery beats anonymized TextQuery under at least one hard-negative restriction: "
            + ", ".join(positive_modes)
        )

    # Stronger condition: positive in both same_concept and same_operator, or very high same_concept.
    sc_ok = False
    so_ok = False
    if same_concept is not None:
        sc_ok = (
            same_concept.get("shape_acc", 0) >= 0.75 and
            same_concept.get("shape_minus_anon_text", -999) > 0.05 and
            same_concept.get("shape_r3", 0) >= 0.90
        )
    if same_operator is not None:
        so_ok = (
            same_operator.get("shape_acc", 0) >= 0.75 and
            same_operator.get("shape_minus_anon_text", -999) > 0.05 and
            same_operator.get("shape_r3", 0) >= 0.90
        )

    if sc_ok and so_ok:
        verdict["pass_strong"] = True
        verdict["notes"].append("ShapeQuery passes both same-concept and same-operator hard-negative tests.")

    if verdict["pass_strong"]:
        verdict["status"] = "PASS_STRONG"
    elif verdict["pass_lite"]:
        verdict["status"] = "PASS_LITE"
    else:
        verdict["status"] = "NO_PASS_OR_INSUFFICIENT"

    return verdict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_dir(OUTPUT_DIR)

    dataset = read_csv_safely(DATASET_PATH)
    features = read_csv_safely(FEATURES_PATH)

    cmap0 = infer_columns(dataset)
    df = merge_dataset_features(dataset, features, cmap0)
    cmap = infer_columns(df)

    blocks = build_feature_blocks(df)
    replacements = build_replacement_terms(df, cmap)

    retrieval_modes = ["full"]
    if cmap.concept_col:
        retrieval_modes += ["same_concept", "same_concept_different_operator"]
    if cmap.operator_col:
        retrieval_modes += ["same_operator", "same_operator_different_concept"]

    splits = []
    splits += make_standard_splits(df, cmap, n_splits=5)
    # Diagnostics: these may exclude families not present in train, so they are not the main verdict.
    splits += make_leave_group_splits(df, cmap, cmap.concept_col, "leave_concept")
    splits += make_leave_group_splits(df, cmap, cmap.operator_col, "leave_operator")

    column_audit = {
        "columns": cmap.__dict__,
        "n_rows": int(len(df)),
        "n_numeric_cols": int(len(numeric_cols(df))),
        "blocks": [
            {"name": b.name, "is_text": b.is_text, "anonymize_text": b.anonymize_text, "n_cols": len(b.cols), "cols_preview": b.cols[:24]}
            for b in blocks
        ],
        "retrieval_modes": retrieval_modes,
        "split_count": len(splits),
        "replacement_terms_preview": list(replacements.items())[:50],
        "unique_counts": {
            "family": int(df[cmap.family_col].nunique()) if cmap.family_col else None,
            "concept": int(df[cmap.concept_col].nunique()) if cmap.concept_col else None,
            "operator": int(df[cmap.operator_col].nunique()) if cmap.operator_col else None,
            "surface": int(df[cmap.surface_col].nunique()) if cmap.surface_col else None,
        }
    }
    (OUTPUT_DIR / "sem5a1_column_audit.json").write_text(
        json.dumps(column_audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    all_metrics = []
    all_details = []

    for split_type, split_name, tr, te in splits:
        for mode in retrieval_modes:
            for block in blocks:
                # Leave-concept plus same-concept often has no same concept candidates except truth missing;
                # we still try and log failures.
                try:
                    metrics, details = evaluate_block(df, block, cmap, tr, te, mode, replacements)
                    metrics["split_type"] = split_type
                    metrics["split"] = split_name
                    details["split_type"] = split_type
                    details["split"] = split_name
                    all_metrics.append(metrics)
                    all_details.append(details)
                    print(
                        f"[OK] {split_name} / {mode} / {block.name}: "
                        f"acc={metrics['acc_top1']:.3f}, r@3={metrics.get('recall_at_3', np.nan):.3f}, "
                        f"n={metrics['n_test_eval']}"
                    )
                except Exception as exc:
                    print(f"[WARN] {split_name} / {mode} / {block.name} failed: {exc}")

    if not all_metrics:
        raise RuntimeError("No retrieval metrics produced.")

    metrics_df = pd.DataFrame(all_metrics)
    details_df = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()

    metrics_path = OUTPUT_DIR / "sem5a1_retrieval_by_split.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    details_path = OUTPUT_DIR / "sem5a1_retrieval_details.csv"
    if not details_df.empty:
        details_df.to_csv(details_path, index=False, encoding="utf-8-sig")

    # Main summary only on standard leave-surface / stratified / random splits.
    main_metrics = metrics_df[metrics_df["split_type"].isin(["leave_surface", "stratified", "random"])].copy()
    summary = summarize_metrics(main_metrics)
    summary_path = OUTPUT_DIR / "sem5a1_summary_by_query_and_mode.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    hard = hard_negative_summary(summary)
    hard_path = OUTPUT_DIR / "sem5a1_hard_negative_summary.csv"
    hard.to_csv(hard_path, index=False, encoding="utf-8-sig")

    # Diagnostics summary for leave-concept/operator if any.
    diag_metrics = metrics_df[metrics_df["split_type"].isin(["leave_concept", "leave_operator"])].copy()
    diag_path = OUTPUT_DIR / "sem5a1_leave_concept_operator_diagnostics.csv"
    if not diag_metrics.empty:
        diag_summary = summarize_metrics(diag_metrics)
        diag_summary.to_csv(diag_path, index=False, encoding="utf-8-sig")

    verdict = make_verdict(hard)
    verdict["output_files"] = {
        "retrieval_by_split": str(metrics_path),
        "retrieval_details": str(details_path),
        "summary_by_query_and_mode": str(summary_path),
        "hard_negative_summary": str(hard_path),
        "leave_concept_operator_diagnostics": str(diag_path) if not diag_metrics.empty else None,
        "column_audit": str(OUTPUT_DIR / "sem5a1_column_audit.json"),
    }
    verdict["column_audit"] = column_audit["unique_counts"]

    verdict_path = OUTPUT_DIR / "sem5a1_verdict.json"
    verdict_path.write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== SEM-5A.1 HARD NEGATIVE SUMMARY ===")
    print(hard.to_string(index=False))
    print("\n=== SEM-5A.1 VERDICT ===")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
