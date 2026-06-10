# -*- coding: utf-8 -*-
"""
SEM-5B: Memory-Guided Trajectory Replay Audit
=============================================

Purpose
-------
SEM-5A.* established Lite evidence for factorized memory addressing:

  Prompt_text -> (q_concept/commit, q_operator/shape) -> MemoryUnit_F

SEM-5B asks the next question:

  If a MemoryUnit_F is retrieved, does it improve downstream trajectory?

Because we currently have offline feature files rather than live intervention
hooks, this script implements an offline replay/proxy audit.

Core idea
---------
For each held-out prompt, retrieve a MemoryUnit from train memory bank using
factorized query. Then simulate a memory-guided state by blending query features
toward the retrieved memory unit:

  q_guided = (1-alpha) * q + alpha * mu_retrieved

Measure whether q_guided becomes closer to the true family memory unit and
farther from competing families.

This is not generation-level control. It is a trajectory-geometry replay proxy.

Main metrics
------------
1. Retrieval correctness:
   retrieved_family == truth_family

2. Geometry improvement:
   Δsim_true = sim(q_guided, mu_true) - sim(q_raw, mu_true)

3. Margin improvement:
   margin = sim_to_true - max_sim_to_wrong
   Δmargin = margin_guided - margin_raw

4. Rank improvement:
   true family rank under q_guided vs q_raw

5. Wrong-memory control:
   compare retrieved-memory guidance vs shuffled/random-memory guidance

Interpretation
--------------
PASS-Lite:
  Retrieved memory guidance improves true-family margin over raw state and
  over shuffled guidance.

PASS-Strong proxy:
  Same holds in same_concept and same_operator hard modes, with positive
  rank improvement and better-than-shuffle margin.

Inputs
------
  sem5a2_outputs/sem5a2_dataset.csv
  sem5a2_outputs/sem5a2_features.csv

Outputs
-------
  sem5b_outputs/sem5b_replay_by_split.csv
  sem5b_outputs/sem5b_replay_summary.csv
  sem5b_outputs/sem5b_verdict.json
  sem5b_outputs/sem5b_column_audit.json

Run
---
  python sem5b_memory_guided_trajectory_replay_audit.py
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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupKFold


# =============================================================================
# Config
# =============================================================================

DATASET_PATH = Path("sem5a2_outputs/sem5a2_dataset.csv")
FEATURES_PATH = Path("sem5a2_outputs/sem5a2_features.csv")
OUTPUT_DIR = Path("sem5b_outputs")

RANDOM_SEED = 42
ALPHAS = [0.10, 0.20, 0.30, 0.50, 0.70]

# Factorized retrieval score.
SHAPE_WEIGHT = 0.50
COMMIT_WEIGHT = 0.50

# Candidate restriction modes.
RETRIEVAL_MODES = [
    "full",
    "same_concept",
    "same_operator",
    "same_concept_different_operator",
    "same_operator_different_concept",
]

# Replay target spaces.
# shape: does memory improve trajectory-shape address?
# commit: does memory improve identity/commit address?
# joint: concat shape+commit.
REPLAY_SPACES = ["shape", "commit", "joint"]


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


def zscore_train_test(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(x_train, axis=0, keepdims=True)
    sd = np.nanstd(x_train, axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    x_train_z = (x_train - mu) / sd
    x_test_z = (x_test - mu) / sd
    x_train_z = np.nan_to_num(x_train_z, nan=0.0, posinf=0.0, neginf=0.0)
    x_test_z = np.nan_to_num(x_test_z, nan=0.0, posinf=0.0, neginf=0.0)
    return x_train_z, x_test_z, mu, sd


def row_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)


def cos_to_centers(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return row_norm(x) @ row_norm(centers).T


def rank_of_truth(sim: np.ndarray, families: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    order = np.argsort(-sim, axis=1)
    ranks = np.zeros(len(y_true), dtype=int)
    for i, truth in enumerate(y_true):
        pos = np.where(families[order[i]] == truth)[0]
        ranks[i] = int(pos[0] + 1) if len(pos) else len(families) + 1
    return ranks


def margin_to_truth(sim: np.ndarray, families: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_sim = np.zeros(len(y_true), dtype=float)
    wrong_max = np.zeros(len(y_true), dtype=float)

    for i, truth in enumerate(y_true):
        idx = np.where(families == truth)[0]
        if len(idx) == 0:
            true_sim[i] = np.nan
            wrong_max[i] = np.nan
            continue
        tidx = idx[0]
        true_sim[i] = sim[i, tidx]
        wrong = np.ones(sim.shape[1], dtype=bool)
        wrong[tidx] = False
        wrong_max[i] = np.max(sim[i, wrong]) if wrong.any() else -np.inf

    return true_sim, wrong_max, true_sim - wrong_max


def safe_mean(x: np.ndarray) -> float:
    return float(np.nanmean(x)) if len(x) else float("nan")


# =============================================================================
# Data
# =============================================================================

@dataclass
class FeatureBlocks:
    init_cols: List[str]
    shape_cols: List[str]
    commit_cols: List[str]
    joint_cols: List[str]


def merge_dataset_features(dataset: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if "row_id" in dataset.columns and "row_id" in features.columns:
        merged = dataset.merge(features, on="row_id", how="inner", suffixes=("", "_feat"))
    else:
        if len(dataset) != len(features):
            raise ValueError("Dataset/features row count differs and no row_id shared.")
        merged = pd.concat([dataset.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
        merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged


def build_blocks(df: pd.DataFrame) -> FeatureBlocks:
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

    return FeatureBlocks(
        init_cols=init_cols,
        shape_cols=shape_cols,
        commit_cols=commit_cols,
        joint_cols=list(shape_cols) + list(commit_cols),
    )


def make_splits(df: pd.DataFrame, n_splits: int = 4) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    y = df["seed_family"].astype(str).values
    groups = df["surface_id"].astype(str).values
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    return [(f"leave_surface_group_{i}", tr, te) for i, (tr, te) in enumerate(gkf.split(df, y, groups))]


def make_family_centers(
    train: pd.DataFrame,
    x_train: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    y_train = train["seed_family"].astype(str).values
    families = sorted(pd.Series(y_train).unique().tolist())

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

    return np.vstack(centers), np.array(families, dtype=object), pd.DataFrame(meta_rows)


def restrict_candidates(
    center_meta: pd.DataFrame,
    test: pd.DataFrame,
    retrieval_mode: str,
) -> np.ndarray:
    mask = np.ones((len(test), len(center_meta)), dtype=bool)

    for i in range(len(test)):
        truth = str(test.iloc[i]["seed_family"])
        cval = str(test.iloc[i]["concept_star"])
        oval = str(test.iloc[i]["operator_star"])

        if retrieval_mode in {"same_concept", "same_concept_different_operator"}:
            mask[i] &= (center_meta["concept"].astype(str).values == cval)

        if retrieval_mode in {"same_operator", "same_operator_different_concept"}:
            mask[i] &= (center_meta["operator"].astype(str).values == oval)

        if retrieval_mode == "same_concept_different_operator":
            mask[i] &= ((center_meta["operator"].astype(str).values != oval) |
                        (center_meta["family"].astype(str).values == truth))

        if retrieval_mode == "same_operator_different_concept":
            mask[i] &= ((center_meta["concept"].astype(str).values != cval) |
                        (center_meta["family"].astype(str).values == truth))

    return mask


# =============================================================================
# Main replay logic
# =============================================================================

def retrieve_memory(
    sim_shape: np.ndarray,
    sim_commit: np.ndarray,
    families: np.ndarray,
    center_meta: pd.DataFrame,
    test: pd.DataFrame,
    retrieval_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    score = SHAPE_WEIGHT * sim_shape + COMMIT_WEIGHT * sim_commit
    mask = restrict_candidates(center_meta, test, retrieval_mode)
    masked_score = score.copy()
    masked_score[~mask] = -np.inf

    pred_idx = np.argmax(masked_score, axis=1)
    pred_family = families[pred_idx].astype(str)
    candidate_counts = mask.sum(axis=1)
    return pred_idx, pred_family, candidate_counts


def replay_guidance(
    q: np.ndarray,
    retrieved_centers: np.ndarray,
    alpha: float,
) -> np.ndarray:
    return (1.0 - alpha) * q + alpha * retrieved_centers


def evaluate_replay_space(
    split_name: str,
    retrieval_mode: str,
    replay_space: str,
    alpha: float,
    train: pd.DataFrame,
    test: pd.DataFrame,
    families: np.ndarray,
    center_meta: pd.DataFrame,
    q_raw: np.ndarray,
    centers: np.ndarray,
    retrieved_idx: np.ndarray,
    retrieved_family: np.ndarray,
    candidate_counts: np.ndarray,
    rng: np.random.Generator,
) -> List[Dict]:
    y_true = test["seed_family"].astype(str).values

    # Raw.
    sim_raw = cos_to_centers(q_raw, centers)
    true_raw, wrong_raw, margin_raw = margin_to_truth(sim_raw, families, y_true)
    rank_raw = rank_of_truth(sim_raw, families, y_true)

    # Retrieved guidance.
    retrieved_centers = centers[retrieved_idx]
    q_guided = replay_guidance(q_raw, retrieved_centers, alpha)
    sim_guided = cos_to_centers(q_guided, centers)
    true_guided, wrong_guided, margin_guided = margin_to_truth(sim_guided, families, y_true)
    rank_guided = rank_of_truth(sim_guided, families, y_true)

    # Shuffled/random guidance control.
    shuffled_idx = retrieved_idx.copy()
    rng.shuffle(shuffled_idx)
    q_shuffle = replay_guidance(q_raw, centers[shuffled_idx], alpha)
    sim_shuffle = cos_to_centers(q_shuffle, centers)
    true_shuffle, wrong_shuffle, margin_shuffle = margin_to_truth(sim_shuffle, families, y_true)
    rank_shuffle = rank_of_truth(sim_shuffle, families, y_true)

    # Oracle guidance upper bound.
    true_idx = np.array([np.where(families == y)[0][0] for y in y_true], dtype=int)
    q_oracle = replay_guidance(q_raw, centers[true_idx], alpha)
    sim_oracle = cos_to_centers(q_oracle, centers)
    true_oracle, wrong_oracle, margin_oracle = margin_to_truth(sim_oracle, families, y_true)
    rank_oracle = rank_of_truth(sim_oracle, families, y_true)

    rows = []
    for condition, ts, wm, mg, rk in [
        ("raw", true_raw, wrong_raw, margin_raw, rank_raw),
        ("retrieved_guided", true_guided, wrong_guided, margin_guided, rank_guided),
        ("shuffle_guided", true_shuffle, wrong_shuffle, margin_shuffle, rank_shuffle),
        ("oracle_guided", true_oracle, wrong_oracle, margin_oracle, rank_oracle),
    ]:
        rows.append({
            "split": split_name,
            "retrieval_mode": retrieval_mode,
            "replay_space": replay_space,
            "alpha": alpha,
            "condition": condition,
            "n": int(len(test)),
            "retrieval_acc": float(np.mean(retrieved_family == y_true)),
            "candidate_count_mean": float(np.mean(candidate_counts)),
            "true_sim_mean": safe_mean(ts),
            "wrong_max_mean": safe_mean(wm),
            "margin_mean": safe_mean(mg),
            "rank_mean": safe_mean(rk),
            "rank_top1": float(np.mean(rk == 1)),
            "rank_recall_at_3": float(np.mean(rk <= 3)),
        })

    # Add paired deltas against raw and against shuffle for convenience.
    delta_ret_margin = margin_guided - margin_raw
    delta_shuf_margin = margin_shuffle - margin_raw
    delta_oracle_margin = margin_oracle - margin_raw

    delta_ret_rank = rank_raw - rank_guided      # positive = improvement
    delta_shuf_rank = rank_raw - rank_shuffle
    delta_oracle_rank = rank_raw - rank_oracle

    rows.append({
        "split": split_name,
        "retrieval_mode": retrieval_mode,
        "replay_space": replay_space,
        "alpha": alpha,
        "condition": "delta_retrieved_minus_raw",
        "n": int(len(test)),
        "retrieval_acc": float(np.mean(retrieved_family == y_true)),
        "candidate_count_mean": float(np.mean(candidate_counts)),
        "true_sim_mean": safe_mean(true_guided - true_raw),
        "wrong_max_mean": safe_mean(wrong_guided - wrong_raw),
        "margin_mean": safe_mean(delta_ret_margin),
        "rank_mean": safe_mean(delta_ret_rank),
        "rank_top1": float(np.mean(rank_guided == 1) - np.mean(rank_raw == 1)),
        "rank_recall_at_3": float(np.mean(rank_guided <= 3) - np.mean(rank_raw <= 3)),
    })

    rows.append({
        "split": split_name,
        "retrieval_mode": retrieval_mode,
        "replay_space": replay_space,
        "alpha": alpha,
        "condition": "delta_retrieved_minus_shuffle",
        "n": int(len(test)),
        "retrieval_acc": float(np.mean(retrieved_family == y_true)),
        "candidate_count_mean": float(np.mean(candidate_counts)),
        "true_sim_mean": safe_mean(true_guided - true_shuffle),
        "wrong_max_mean": safe_mean(wrong_guided - wrong_shuffle),
        "margin_mean": safe_mean(margin_guided - margin_shuffle),
        "rank_mean": safe_mean(rank_shuffle - rank_guided),
        "rank_top1": float(np.mean(rank_guided == 1) - np.mean(rank_shuffle == 1)),
        "rank_recall_at_3": float(np.mean(rank_guided <= 3) - np.mean(rank_shuffle <= 3)),
    })

    rows.append({
        "split": split_name,
        "retrieval_mode": retrieval_mode,
        "replay_space": replay_space,
        "alpha": alpha,
        "condition": "delta_oracle_minus_raw",
        "n": int(len(test)),
        "retrieval_acc": 1.0,
        "candidate_count_mean": float(np.mean(candidate_counts)),
        "true_sim_mean": safe_mean(true_oracle - true_raw),
        "wrong_max_mean": safe_mean(wrong_oracle - wrong_raw),
        "margin_mean": safe_mean(delta_oracle_margin),
        "rank_mean": safe_mean(delta_oracle_rank),
        "rank_top1": float(np.mean(rank_oracle == 1) - np.mean(rank_raw == 1)),
        "rank_recall_at_3": float(np.mean(rank_oracle <= 3) - np.mean(rank_raw <= 3)),
    })

    return rows


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    agg = rows.groupby(["retrieval_mode", "replay_space", "alpha", "condition"]).agg({
        "n": ["sum"],
        "retrieval_acc": ["mean", "std"],
        "candidate_count_mean": ["mean"],
        "true_sim_mean": ["mean", "std"],
        "wrong_max_mean": ["mean", "std"],
        "margin_mean": ["mean", "std"],
        "rank_mean": ["mean", "std"],
        "rank_top1": ["mean", "std"],
        "rank_recall_at_3": ["mean", "std"],
    })
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index().sort_values(["retrieval_mode", "replay_space", "alpha", "condition"])


def make_verdict(summary: pd.DataFrame) -> Dict:
    verdict = {
        "status": "UNDETERMINED",
        "pass_lite": False,
        "pass_strong_proxy": False,
        "notes": [],
    }

    if summary.empty:
        verdict["status"] = "NO_SUMMARY"
        return verdict

    delta = summary[summary["condition"] == "delta_retrieved_minus_shuffle"].copy()
    raw_delta = summary[summary["condition"] == "delta_retrieved_minus_raw"].copy()
    oracle = summary[summary["condition"] == "delta_oracle_minus_raw"].copy()

    # Find best delta over modes/spaces/alphas by margin.
    if not delta.empty:
        best = delta.sort_values("margin_mean_mean", ascending=False).iloc[0].to_dict()
        verdict["best_retrieved_vs_shuffle"] = {
            "retrieval_mode": best["retrieval_mode"],
            "replay_space": best["replay_space"],
            "alpha": float(best["alpha"]),
            "margin_gain": float(best["margin_mean_mean"]),
            "rank_gain": float(best["rank_mean_mean"]),
            "r3_gain": float(best["rank_recall_at_3_mean"]),
        }
        if best["margin_mean_mean"] > 0.01:
            verdict["pass_lite"] = True
            verdict["notes"].append("Retrieved memory guidance improves margin over shuffled guidance.")

    # Strong proxy: in both same_concept and same_operator, positive margin vs shuffle for some alpha/space.
    strong_modes = []
    for mode in ["same_concept", "same_operator"]:
        sub = delta[delta["retrieval_mode"] == mode]
        if sub.empty:
            continue
        best_mode = sub.sort_values("margin_mean_mean", ascending=False).iloc[0]
        if best_mode["margin_mean_mean"] > 0.01 and best_mode["rank_mean_mean"] >= 0:
            strong_modes.append(mode)
            verdict["notes"].append(
                f"{mode}: retrieved guidance beats shuffled guidance "
                f"(best margin gain={best_mode['margin_mean_mean']:.4f})."
            )

    if set(strong_modes) >= {"same_concept", "same_operator"}:
        verdict["pass_strong_proxy"] = True
        verdict["notes"].append("Retrieved memory guidance is positive in both same-concept and same-operator modes.")

    # Add raw comparison.
    if not raw_delta.empty:
        best_raw = raw_delta.sort_values("margin_mean_mean", ascending=False).iloc[0].to_dict()
        verdict["best_retrieved_vs_raw"] = {
            "retrieval_mode": best_raw["retrieval_mode"],
            "replay_space": best_raw["replay_space"],
            "alpha": float(best_raw["alpha"]),
            "margin_gain": float(best_raw["margin_mean_mean"]),
            "rank_gain": float(best_raw["rank_mean_mean"]),
            "r3_gain": float(best_raw["rank_recall_at_3_mean"]),
        }
        if best_raw["margin_mean_mean"] > 0.01:
            verdict["notes"].append("Retrieved memory guidance improves margin over raw query.")

    if not oracle.empty:
        best_oracle = oracle.sort_values("margin_mean_mean", ascending=False).iloc[0].to_dict()
        verdict["oracle_upper_bound"] = {
            "retrieval_mode": best_oracle["retrieval_mode"],
            "replay_space": best_oracle["replay_space"],
            "alpha": float(best_oracle["alpha"]),
            "margin_gain": float(best_oracle["margin_mean_mean"]),
            "rank_gain": float(best_oracle["rank_mean_mean"]),
            "r3_gain": float(best_oracle["rank_recall_at_3_mean"]),
        }

    verdict["status"] = "PASS_STRONG_PROXY" if verdict["pass_strong_proxy"] else ("PASS_LITE" if verdict["pass_lite"] else "NO_PASS_OR_INSUFFICIENT")
    return verdict


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

    audit = {
        "dataset_path": str(DATASET_PATH),
        "features_path": str(FEATURES_PATH),
        "n_rows": int(len(df)),
        "n_numeric_cols": int(len(numeric_cols(df))),
        "blocks": {
            "init": len(blocks.init_cols),
            "shape": len(blocks.shape_cols),
            "commit": len(blocks.commit_cols),
            "joint": len(blocks.joint_cols),
        },
        "retrieval_modes": RETRIEVAL_MODES,
        "replay_spaces": REPLAY_SPACES,
        "alphas": ALPHAS,
        "split_count": len(splits),
        "counts": {
            "concepts": int(df["concept_star"].nunique()),
            "operators": int(df["operator_star"].nunique()),
            "families": int(df["seed_family"].nunique()),
            "surfaces": int(df["surface_id"].nunique()),
        },
    }
    (OUTPUT_DIR / "sem5b_column_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rng = np.random.default_rng(RANDOM_SEED)
    all_rows = []

    for split_name, tr, te in splits:
        train = df.iloc[tr].copy().reset_index(drop=True)
        test = df.iloc[te].copy().reset_index(drop=True)

        # Shape and commit family similarities used for retrieval.
        x_shape_train = train[blocks.shape_cols].to_numpy(dtype=float)
        x_shape_test = test[blocks.shape_cols].to_numpy(dtype=float)
        x_shape_train_z, x_shape_test_z, _, _ = zscore_train_test(x_shape_train, x_shape_test)
        shape_centers, families, center_meta = make_family_centers(train, x_shape_train_z)
        sim_shape = cos_to_centers(x_shape_test_z, shape_centers)

        x_commit_train = train[blocks.commit_cols].to_numpy(dtype=float)
        x_commit_test = test[blocks.commit_cols].to_numpy(dtype=float)
        x_commit_train_z, x_commit_test_z, _, _ = zscore_train_test(x_commit_train, x_commit_test)
        commit_centers, families2, center_meta2 = make_family_centers(train, x_commit_train_z)
        sim_commit = cos_to_centers(x_commit_test_z, commit_centers)

        if list(families) != list(families2):
            raise RuntimeError("Family order mismatch.")

        # Joint space.
        x_joint_train = train[blocks.joint_cols].to_numpy(dtype=float)
        x_joint_test = test[blocks.joint_cols].to_numpy(dtype=float)
        x_joint_train_z, x_joint_test_z, _, _ = zscore_train_test(x_joint_train, x_joint_test)
        joint_centers, families3, center_meta3 = make_family_centers(train, x_joint_train_z)

        if list(families) != list(families3):
            raise RuntimeError("Family order mismatch for joint.")

        for mode in RETRIEVAL_MODES:
            retrieved_idx, retrieved_family, candidate_counts = retrieve_memory(
                sim_shape, sim_commit, families, center_meta, test, mode
            )
            retrieval_acc = float(np.mean(retrieved_family == test["seed_family"].astype(str).values))
            print(f"[RET] {split_name} / {mode}: retrieval_acc={retrieval_acc:.3f}, cand={candidate_counts.mean():.1f}")

            for alpha in ALPHAS:
                for space in REPLAY_SPACES:
                    if space == "shape":
                        q_raw = x_shape_test_z
                        centers = shape_centers
                    elif space == "commit":
                        q_raw = x_commit_test_z
                        centers = commit_centers
                    elif space == "joint":
                        q_raw = x_joint_test_z
                        centers = joint_centers
                    else:
                        raise ValueError(space)

                    rows = evaluate_replay_space(
                        split_name=split_name,
                        retrieval_mode=mode,
                        replay_space=space,
                        alpha=alpha,
                        train=train,
                        test=test,
                        families=families,
                        center_meta=center_meta,
                        q_raw=q_raw,
                        centers=centers,
                        retrieved_idx=retrieved_idx,
                        retrieved_family=retrieved_family,
                        candidate_counts=candidate_counts,
                        rng=rng,
                    )
                    all_rows.extend(rows)

    result = pd.DataFrame(all_rows)
    by_split_path = OUTPUT_DIR / "sem5b_replay_by_split.csv"
    summary_path = OUTPUT_DIR / "sem5b_replay_summary.csv"
    verdict_path = OUTPUT_DIR / "sem5b_verdict.json"

    result.to_csv(by_split_path, index=False, encoding="utf-8-sig")

    summary = summarize(result)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    verdict = make_verdict(summary)
    verdict["output_files"] = {
        "replay_by_split": str(by_split_path),
        "replay_summary": str(summary_path),
        "column_audit": str(OUTPUT_DIR / "sem5b_column_audit.json"),
    }
    verdict_path.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== SEM-5B BEST SUMMARY ===")
    for cond in ["delta_retrieved_minus_raw", "delta_retrieved_minus_shuffle", "delta_oracle_minus_raw"]:
        sub = summary[summary["condition"] == cond]
        if not sub.empty:
            best = sub.sort_values("margin_mean_mean", ascending=False).head(5)
            print(f"\n[{cond}]")
            print(best[["retrieval_mode", "replay_space", "alpha", "margin_mean_mean", "rank_mean_mean", "rank_recall_at_3_mean"]].to_string(index=False))

    print("\n=== SEM-5B VERDICT ===")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
