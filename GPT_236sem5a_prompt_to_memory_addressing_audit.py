# -*- coding: utf-8 -*-
"""
SEM-5A: Prompt-to-Memory Addressing Audit
========================================

Goal
----
Validate whether a natural-language prompt activates the correct trajectory-geometry
memory unit through an internal latent query state.

Core comparison:
  TextQuery   -> MemoryUnit
  InitQuery   -> MemoryUnit
  ShapeQuery  -> MemoryUnit
  CommitQuery -> MemoryUnit

Main hypothesis:
  ShapeQuery(L7-L19) retrieves SeedFamily / MemoryUnit better than TextQuery
  and better than InitQuery, especially under leave-surface-family-out evaluation.

Expected input files
--------------------
Default relative paths:

  sem3a_outputs/sem3a_dataset.csv
  sem3a_outputs/sem3a_features.csv
  sem4e_outputs/sem4e_memory_units.csv   optional

The script is deliberately defensive:
- It auto-detects family / prompt / concept / operator / surface columns.
- It auto-detects numeric feature columns by layer-number patterns such as L7,
  layer_7, l07, etc.
- If sem4e_memory_units.csv is missing, memory centers are reconstructed from
  sem3a_features.csv inside each train split to avoid heldout leakage.

Outputs
-------
  sem5a_outputs/sem5a_config.json
  sem5a_outputs/sem5a_column_audit.json
  sem5a_outputs/sem5a_retrieval_by_split.csv
  sem5a_outputs/sem5a_summary_by_query.csv
  sem5a_outputs/sem5a_condition_summary.csv
  sem5a_outputs/sem5a_memory_competition_entropy.csv
  sem5a_outputs/sem5a_verdict.json

Author note
-----------
This is an audit script, not a model-training script. It uses only existing
SEM-3A / SEM-4E outputs.
"""

from __future__ import annotations

import json
import math
import os
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
from sklearn.model_selection import GroupKFold, StratifiedKFold


# ---------------------------------------------------------------------------
# Hardcoded defaults. Modify here only if your directory names differ.
# ---------------------------------------------------------------------------

DATASET_PATH = Path("sem3a_outputs/sem3a_dataset.csv")
FEATURES_PATH = Path("sem3a_outputs/sem3a_features.csv")
MEMORY_UNITS_PATH = Path("sem4e_outputs/sem4e_memory_units.csv")
OUTPUT_DIR = Path("sem5a_outputs")

RANDOM_SEED = 42
TOP_KS = (1, 3, 5)
SOFTMAX_TEMPERATURE = 0.10

# Windows users: if you keep scripts in a different directory, either run this
# file from your project root or replace the paths above with absolute paths.


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
    """
    Tries to recover layer number from names like:
      L7_x, l07_pca_3, layer_19_center_pca_0, h_23, block24
    """
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

    # Conservative fallback: strings like "L23CenterPCA0" may normalize to l23centerpca0
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

    # If preferred keyword columns exist in the layer window, use them.
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
    return float(np.mean(vals)) if vals else 0.0


def safe_macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# Data alignment
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


def infer_columns(df: pd.DataFrame) -> ColumnMap:
    cols = list(df.columns)

    id_col = first_existing(cols, [
        "prompt_id", "sample_id", "id", "uid", "case_id", "example_id"
    ])

    prompt_col = first_existing(cols, [
        "prompt", "text", "prompt_text", "input", "input_text", "query"
    ]) or find_by_keywords(cols, ["prompt"]) or find_by_keywords(cols, ["text"])

    family_col = first_existing(cols, [
        "seed_family", "SeedFamily", "family", "family_id", "memory_family",
        "MemoryFamily", "seedfamily", "label", "target_family"
    ]) or find_by_keywords(cols, ["family"]) or find_by_keywords(cols, ["seed"])

    concept_col = first_existing(cols, [
        "concept", "Concept", "topic", "entity", "subject"
    ]) or find_by_keywords(cols, ["concept"]) or find_by_keywords(cols, ["topic"])

    operator_col = first_existing(cols, [
        "operator", "OperatorID", "operator_id", "policy_id", "PolicyID",
        "constraint", "task_type", "task"
    ]) or find_by_keywords(cols, ["operator"]) or find_by_keywords(cols, ["policy"])

    surface_col = first_existing(cols, [
        "surface", "surface_group", "surface_family", "template", "template_id",
        "paraphrase_group", "style", "surface_id"
    ]) or find_by_keywords(cols, ["surface"]) or find_by_keywords(cols, ["template"])

    prompt_type_col = first_existing(cols, [
        "prompt_type", "type", "heldout_type", "query_type", "eval_type"
    ]) or find_by_keywords(cols, ["prompt", "type"])

    condition_col = first_existing(cols, [
        "condition", "Condition", "mechanism", "family_type", "case_type"
    ]) or find_by_keywords(cols, ["condition"]) or find_by_keywords(cols, ["mechanism"])

    if prompt_col is None:
        raise ValueError("Could not infer prompt/text column. Please add a column named prompt or text.")
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
                "Add prompt_id/sample_id to both files."
            )
        merged = pd.concat(
            [dataset.reset_index(drop=True), features.reset_index(drop=True)],
            axis=1,
        )
        # Remove duplicate columns introduced by concat, keeping first occurrence.
        merged = merged.loc[:, ~merged.columns.duplicated()]

    return merged


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

@dataclass
class FeatureBlock:
    name: str
    cols: List[str]
    is_text: bool = False


def build_feature_blocks(df: pd.DataFrame, cmap: ColumnMap) -> List[FeatureBlock]:
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

    blocks = [FeatureBlock("TextQuery", [], is_text=True)]

    if init_cols:
        blocks.append(FeatureBlock("InitQuery_L0_6", init_cols))
    if shape_cols:
        blocks.append(FeatureBlock("ShapeQuery_L7_19", shape_cols))
    if commit_cols:
        blocks.append(FeatureBlock("CommitQuery_L23_25", commit_cols))

    # Fallbacks if no layer-aware columns exist.
    if len(blocks) == 1:
        all_num = numeric_cols(df)
        id_like = {
            c for c in all_num
            if normalize_name(c) in {"id", "prompt_id", "sample_id", "family_id"}
        }
        all_num = [c for c in all_num if c not in id_like]
        if all_num:
            blocks.append(FeatureBlock("AllNumericFallback", all_num))

    return blocks


def make_splits(df: pd.DataFrame, cmap: ColumnMap, n_splits: int = 5) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    y = df[cmap.family_col].astype(str).values
    splits = []

    if cmap.surface_col and cmap.surface_col in df.columns:
        groups = df[cmap.surface_col].astype(str).values
        unique_groups = np.unique(groups)
        if len(unique_groups) >= 2:
            if len(unique_groups) >= n_splits:
                gkf = GroupKFold(n_splits=min(n_splits, len(unique_groups)))
                for i, (tr, te) in enumerate(gkf.split(df, y, groups)):
                    splits.append((f"leave_surface_group_{i}", tr, te))
            else:
                # Manual leave-one-group-out.
                idx = np.arange(len(df))
                for g in unique_groups:
                    te = idx[groups == g]
                    tr = idx[groups != g]
                    if len(te) and len(tr):
                        splits.append((f"leave_surface_{g}", tr, te))

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
                splits.append((f"stratified_{i}", tr, te))
        else:
            # Last resort random split.
            rng = np.random.default_rng(RANDOM_SEED)
            idx = np.arange(len(df))
            rng.shuffle(idx)
            cut = max(1, int(0.8 * len(idx)))
            splits.append(("random_80_20", idx[:cut], idx[cut:]))

    return splits


def evaluate_numeric_block(
    df: pd.DataFrame,
    block: FeatureBlock,
    cmap: ColumnMap,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[Dict, pd.DataFrame]:
    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    y_train = train[cmap.family_col].astype(str).values
    y_test = test[cmap.family_col].astype(str).values

    x_train = train[block.cols].to_numpy(dtype=float)
    x_test = test[block.cols].to_numpy(dtype=float)
    x_train, x_test = zscore_train_test(x_train, x_test)

    families = sorted(pd.Series(y_train).unique().astype(str).tolist())
    centers = []
    for fam in families:
        centers.append(x_train[y_train == fam].mean(axis=0))
    centers = np.vstack(centers)

    # Normalize for cosine.
    sim = cosine_similarity(x_test, centers)
    order = np.argsort(-sim, axis=1)
    ranked = np.array(families, dtype=object)[order]
    pred = ranked[:, 0].astype(str)

    ent = row_softmax_entropy(sim)

    out = {
        "query_type": block.name,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_families_train": int(len(families)),
        "feature_dim": int(len(block.cols)),
        "acc_top1": float(accuracy_score(y_test, pred)),
        "macro_f1": safe_macro_f1(list(y_test), list(pred)),
        "mrr": mrr_score(list(y_test), ranked),
        "entropy_mean": float(np.mean(ent)),
        "entropy_std": float(np.std(ent)),
    }
    for k in TOP_KS:
        out[f"recall_at_{k}"] = recall_at_k(list(y_test), ranked, k=min(k, len(families)))

    details = pd.DataFrame({
        "query_type": block.name,
        "truth_family": y_test,
        "pred_family": pred,
        "top1_sim": sim[np.arange(len(test_idx)), order[:, 0]],
        "entropy": ent,
        "test_row_index": test_idx,
    })

    if cmap.prompt_type_col and cmap.prompt_type_col in test.columns:
        details["prompt_type"] = test[cmap.prompt_type_col].astype(str).values
    if cmap.condition_col and cmap.condition_col in test.columns:
        details["condition"] = test[cmap.condition_col].astype(str).values
    if cmap.concept_col and cmap.concept_col in test.columns:
        details["concept"] = test[cmap.concept_col].astype(str).values
    if cmap.operator_col and cmap.operator_col in test.columns:
        details["operator"] = test[cmap.operator_col].astype(str).values
    if cmap.surface_col and cmap.surface_col in test.columns:
        details["surface"] = test[cmap.surface_col].astype(str).values

    for j, k in enumerate(TOP_KS):
        kk = min(k, len(families))
        details[f"hit_at_{k}"] = [
            truth in set(row[:kk]) for truth, row in zip(y_test, ranked)
        ]

    return out, details


def evaluate_text_block(
    df: pd.DataFrame,
    block: FeatureBlock,
    cmap: ColumnMap,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[Dict, pd.DataFrame]:
    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    y_train = train[cmap.family_col].astype(str).values
    y_test = test[cmap.family_col].astype(str).values

    train_text = train[cmap.prompt_col].fillna("").astype(str).tolist()
    test_text = test[cmap.prompt_col].fillna("").astype(str).tolist()

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
    for fam in families:
        rows = np.where(y_train == fam)[0]
        centers.append(np.asarray(x_train[rows].mean(axis=0)).ravel())
    centers = np.vstack(centers)

    sim = cosine_similarity(x_test, centers)
    order = np.argsort(-sim, axis=1)
    ranked = np.array(families, dtype=object)[order]
    pred = ranked[:, 0].astype(str)

    ent = row_softmax_entropy(sim)

    out = {
        "query_type": block.name,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_families_train": int(len(families)),
        "feature_dim": int(x_train.shape[1]),
        "acc_top1": float(accuracy_score(y_test, pred)),
        "macro_f1": safe_macro_f1(list(y_test), list(pred)),
        "mrr": mrr_score(list(y_test), ranked),
        "entropy_mean": float(np.mean(ent)),
        "entropy_std": float(np.std(ent)),
    }
    for k in TOP_KS:
        out[f"recall_at_{k}"] = recall_at_k(list(y_test), ranked, k=min(k, len(families)))

    details = pd.DataFrame({
        "query_type": block.name,
        "truth_family": y_test,
        "pred_family": pred,
        "top1_sim": sim[np.arange(len(test_idx)), order[:, 0]],
        "entropy": ent,
        "test_row_index": test_idx,
    })

    if cmap.prompt_type_col and cmap.prompt_type_col in test.columns:
        details["prompt_type"] = test[cmap.prompt_type_col].astype(str).values
    if cmap.condition_col and cmap.condition_col in test.columns:
        details["condition"] = test[cmap.condition_col].astype(str).values
    if cmap.concept_col and cmap.concept_col in test.columns:
        details["concept"] = test[cmap.concept_col].astype(str).values
    if cmap.operator_col and cmap.operator_col in test.columns:
        details["operator"] = test[cmap.operator_col].astype(str).values
    if cmap.surface_col and cmap.surface_col in test.columns:
        details["surface"] = test[cmap.surface_col].astype(str).values

    for j, k in enumerate(TOP_KS):
        kk = min(k, len(families))
        details[f"hit_at_{k}"] = [
            truth in set(row[:kk]) for truth, row in zip(y_test, ranked)
        ]

    return out, details


def evaluate_block(
    df: pd.DataFrame,
    block: FeatureBlock,
    cmap: ColumnMap,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[Dict, pd.DataFrame]:
    if block.is_text:
        return evaluate_text_block(df, block, cmap, train_idx, test_idx)
    return evaluate_numeric_block(df, block, cmap, train_idx, test_idx)


# ---------------------------------------------------------------------------
# Condition-level diagnostics
# ---------------------------------------------------------------------------

def summarize_conditions(details: pd.DataFrame) -> pd.DataFrame:
    group_cols = []
    for c in ["query_type", "prompt_type", "condition", "concept", "operator"]:
        if c in details.columns:
            group_cols.append(c)

    if "query_type" not in group_cols:
        return pd.DataFrame()

    rows = []
    for keys, sub in details.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: v for c, v in zip(group_cols, keys)}
        row["n"] = int(len(sub))
        row["acc_top1"] = float((sub["truth_family"] == sub["pred_family"]).mean())
        row["entropy_mean"] = float(sub["entropy"].mean())
        for k in TOP_KS:
            col = f"hit_at_{k}"
            if col in sub.columns:
                row[f"recall_at_{k}"] = float(sub[col].mean())
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def summarize_entropy(details: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["query_type"]
    for c in ["prompt_type", "condition"]:
        if c in details.columns:
            group_cols.append(c)

    rows = []
    for keys, sub in details.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: v for c, v in zip(group_cols, keys)}
        row["n"] = int(len(sub))
        row["entropy_mean"] = float(sub["entropy"].mean())
        row["entropy_std"] = float(sub["entropy"].std(ddof=0))
        row["top1_sim_mean"] = float(sub["top1_sim"].mean())
        row["acc_top1"] = float((sub["truth_family"] == sub["pred_family"]).mean())
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def make_verdict(summary: pd.DataFrame) -> Dict:
    """
    PASS-Lite:
      Best ShapeQuery retrieval > TextQuery retrieval by > 10 percentage points.

    PASS-Strong proxy:
      ShapeQuery best or tied-best among non-commit early interfaces,
      Recall@3 >= 0.9 if available,
      and ShapeQuery > InitQuery.
    """
    verdict = {
        "status": "UNDETERMINED",
        "pass_lite": False,
        "pass_strong_proxy": False,
        "notes": [],
    }

    if summary.empty:
        verdict["status"] = "NO_SUMMARY"
        return verdict

    rows = summary.set_index("query_type").to_dict(orient="index")

    text_acc = rows.get("TextQuery", {}).get("acc_top1_mean")
    shape_names = [q for q in rows if q.startswith("ShapeQuery")]
    init_names = [q for q in rows if q.startswith("InitQuery")]
    commit_names = [q for q in rows if q.startswith("CommitQuery")]

    if shape_names:
        shape_name = max(shape_names, key=lambda q: rows[q].get("acc_top1_mean", -999))
        shape_acc = rows[shape_name].get("acc_top1_mean")
        shape_r3 = rows[shape_name].get("recall_at_3_mean")

        verdict["best_shape_query"] = shape_name
        verdict["shape_acc"] = shape_acc
        verdict["shape_recall_at_3"] = shape_r3

        if text_acc is not None:
            verdict["text_acc"] = text_acc
            verdict["shape_minus_text_acc"] = shape_acc - text_acc
            if shape_acc - text_acc > 0.10:
                verdict["pass_lite"] = True
                verdict["notes"].append("ShapeQuery exceeds TextQuery by > 10 percentage points.")

        if init_names:
            init_name = max(init_names, key=lambda q: rows[q].get("acc_top1_mean", -999))
            init_acc = rows[init_name].get("acc_top1_mean")
            verdict["best_init_query"] = init_name
            verdict["init_acc"] = init_acc
            verdict["shape_minus_init_acc"] = shape_acc - init_acc
            if shape_acc > init_acc:
                verdict["notes"].append("ShapeQuery exceeds InitQuery.")

        # Compare among practical early interfaces: Text + Init + Shape.
        early_candidates = ["TextQuery"] + init_names + shape_names
        best_early = max(early_candidates, key=lambda q: rows.get(q, {}).get("acc_top1_mean", -999))
        verdict["best_early_query"] = best_early

        if best_early == shape_name and (shape_r3 is None or shape_r3 >= 0.90):
            verdict["pass_strong_proxy"] = True
            verdict["notes"].append("ShapeQuery is best early interface and Recall@3 is high or unavailable.")

    else:
        verdict["notes"].append("No ShapeQuery columns detected; cannot test core SEM-5 hypothesis.")

    if verdict["pass_strong_proxy"]:
        verdict["status"] = "PASS_STRONG_PROXY"
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

    # Infer columns from dataset first; if features carries labels too, merged df keeps dataset labels.
    cmap = infer_columns(dataset)
    df = merge_dataset_features(dataset, features, cmap)

    # Re-infer after merge in case columns changed.
    cmap = infer_columns(df)

    blocks = build_feature_blocks(df, cmap)
    splits = make_splits(df, cmap, n_splits=5)

    config = {
        "dataset_path": str(DATASET_PATH),
        "features_path": str(FEATURES_PATH),
        "memory_units_path": str(MEMORY_UNITS_PATH),
        "output_dir": str(OUTPUT_DIR),
        "top_ks": TOP_KS,
        "softmax_temperature": SOFTMAX_TEMPERATURE,
        "n_rows": int(len(df)),
        "n_splits": int(len(splits)),
        "blocks": [
            {"name": b.name, "is_text": b.is_text, "n_cols": len(b.cols), "cols_preview": b.cols[:20]}
            for b in blocks
        ],
    }
    (OUTPUT_DIR / "sem5a_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    column_audit = {
        "columns": cmap.__dict__,
        "all_columns": list(df.columns),
        "numeric_column_count": len(numeric_cols(df)),
        "layer_detected_columns": {
            str(i): [c for c in numeric_cols(df) if layer_number_from_col(c) == i][:20]
            for i in range(0, 32)
        },
    }
    (OUTPUT_DIR / "sem5a_column_audit.json").write_text(
        json.dumps(column_audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    all_metrics = []
    all_details = []

    for split_name, tr, te in splits:
        for block in blocks:
            try:
                metrics, details = evaluate_block(df, block, cmap, tr, te)
                metrics["split"] = split_name
                details["split"] = split_name
                all_metrics.append(metrics)
                all_details.append(details)
                print(f"[OK] {split_name} / {block.name}: acc={metrics['acc_top1']:.3f}, r@3={metrics.get('recall_at_3', np.nan):.3f}")
            except Exception as exc:
                print(f"[WARN] {split_name} / {block.name} failed: {exc}")

    if not all_metrics:
        raise RuntimeError("No retrieval metrics produced. Check column detection and input files.")

    metrics_df = pd.DataFrame(all_metrics)
    details_df = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()

    metrics_path = OUTPUT_DIR / "sem5a_retrieval_by_split.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    # Aggregate summary.
    agg_map = {
        "acc_top1": ["mean", "std"],
        "macro_f1": ["mean", "std"],
        "mrr": ["mean", "std"],
        "entropy_mean": ["mean", "std"],
    }
    for k in TOP_KS:
        col = f"recall_at_{k}"
        if col in metrics_df.columns:
            agg_map[col] = ["mean", "std"]

    summary = metrics_df.groupby("query_type").agg(agg_map)
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index().sort_values("acc_top1_mean", ascending=False)
    summary_path = OUTPUT_DIR / "sem5a_summary_by_query.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    if not details_df.empty:
        details_path = OUTPUT_DIR / "sem5a_retrieval_details.csv"
        details_df.to_csv(details_path, index=False, encoding="utf-8-sig")

        cond = summarize_conditions(details_df)
        cond_path = OUTPUT_DIR / "sem5a_condition_summary.csv"
        cond.to_csv(cond_path, index=False, encoding="utf-8-sig")

        ent = summarize_entropy(details_df)
        ent_path = OUTPUT_DIR / "sem5a_memory_competition_entropy.csv"
        ent.to_csv(ent_path, index=False, encoding="utf-8-sig")
    else:
        cond_path = None
        ent_path = None

    verdict = make_verdict(summary)
    verdict["output_files"] = {
        "retrieval_by_split": str(metrics_path),
        "summary_by_query": str(summary_path),
        "condition_summary": str(cond_path) if cond_path else None,
        "memory_competition_entropy": str(ent_path) if ent_path else None,
        "column_audit": str(OUTPUT_DIR / "sem5a_column_audit.json"),
        "config": str(OUTPUT_DIR / "sem5a_config.json"),
    }

    verdict_path = OUTPUT_DIR / "sem5a_verdict.json"
    verdict_path.write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== SEM-5A SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== SEM-5A VERDICT ===")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
