# -*- coding: utf-8 -*-
"""
SR-3A: Pairwise Source-Target Dataset Builder

Purpose
-------
SR-2C confirmed that action-prototype contrast is still insufficient.
The object must be explicitly pairwise:

    StructuralResolution = SR(source_case, target_case)

This script builds a pairwise dataset from SR-1B.1 TopK/VIM features.

For each target task, it pairs the target with multiple source cases:
    - strict source direct anchors
    - observe-like anchors
    - reject-like anchors
    - same / different reuse-group cases

Each pair receives labels:
    pair_action_label: direct_reuse / observe / reject
    pair_utility_label: useful / neutral / harmful
    pair_gain: simulated pair gain

The feature vector is contrastive:
    target features
    source features
    target - source
    abs(target - source)
    target * source
    distance/cosine summaries

Inputs
------
    sr1b1_outputs/sr1b1_topk_vim_features.csv

Outputs
-------
    sr3a_outputs/sr3a_pairwise_source_target_dataset.csv
    sr3a_outputs/sr3a_pairwise_summary.json

Notes
-----
This does NOT train a model. It constructs the correct object for SR-3B.

Theory
------
If structural resolution is comparative, the supervised object should be:

    (source operator context, target context) -> reuse mode / utility

not:

    target context -> reuse mode
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(r"C:\Users\ZH\Desktop\AGI\python_script")

IN_FEATURES = BASE_DIR / "sr1b1_outputs" / "sr1b1_topk_vim_features.csv"

OUT_DIR = BASE_DIR / "sr3a_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PAIRWISE = OUT_DIR / "sr3a_pairwise_source_target_dataset.csv"
OUT_SUMMARY = OUT_DIR / "sr3a_pairwise_summary.json"


SOURCE_GROUP = "G0_STRICT_SOURCE"
ACTIONS = ["direct_reuse", "observe", "reject"]

# To keep file size manageable.
MAX_SOURCES_PER_BUCKET = 24
RANDOM_SEED = 42

# Feature windows to use in pairwise deltas.
FEATURE_GROUP = "topk_mid_boundary"


META_COLS = [
    "task_id",
    "reuse_group",
    "task_family",
    "target_operator",
    "concept",
    "boundary_strength",
    "gain_oracle_action",
    "best_policy_action",
    "best_policy_id",
    "correct_vs_no",
    "correct_vs_wrong",
    "correct_vs_random",
    "best_policy_gain",
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
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


def clean_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    med = out.median(axis=0, numeric_only=True)
    out = out.fillna(med).fillna(0.0)
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def sample_sources(df: pd.DataFrame, rng: np.random.Generator) -> Dict[str, pd.DataFrame]:
    """
    Define source buckets.

    direct_source:
        known useful strict source anchors

    observe_source:
        gain_oracle observe rows if available; otherwise boundary_strength==3

    reject_source:
        gain_oracle reject rows from weak/surface/unrelated regions

    same_operator_source:
        useful direct rows not necessarily strict source

    broad_source:
        random rows across the dataset for negative/background pairs
    """
    source_buckets = {}

    direct_source = df[
        (df["reuse_group"].astype(str) == SOURCE_GROUP)
        & (df["gain_oracle_action"].astype(str) == "direct_reuse")
    ].copy()

    if direct_source.empty:
        direct_source = df[df["gain_oracle_action"].astype(str) == "direct_reuse"].copy()

    observe_source = df[df["gain_oracle_action"].astype(str) == "observe"].copy()
    if observe_source.empty and "boundary_strength" in df.columns:
        observe_source = df[pd.to_numeric(df["boundary_strength"], errors="coerce") == 3].copy()

    reject_source = df[df["gain_oracle_action"].astype(str) == "reject"].copy()

    same_operator_source = df[df["gain_oracle_action"].astype(str) == "direct_reuse"].copy()

    broad_source = df.copy()

    buckets = {
        "direct_source": direct_source,
        "observe_source": observe_source,
        "reject_source": reject_source,
        "same_operator_source": same_operator_source,
        "broad_source": broad_source,
    }

    for name, sub in buckets.items():
        if len(sub) > MAX_SOURCES_PER_BUCKET:
            idx = rng.choice(sub.index.values, size=MAX_SOURCES_PER_BUCKET, replace=False)
            sub = sub.loc[idx].copy()
        source_buckets[name] = sub.reset_index(drop=True)

    return source_buckets


def infer_pair_label(target_row: pd.Series, source_bucket_name: str, source_row: pd.Series) -> Tuple[str, str, float]:
    """
    Returns:
        pair_action_label, pair_utility_label, pair_gain

    The label is target-centric but conditioned on source type.

    For known direct source:
        if target gain_oracle_action says direct -> useful direct
        if target observe -> observe
        if target reject -> reject/harmful

    For reject source:
        reuse should generally reject unless target is also reject-like.

    This creates a pairwise supervision signal rather than isolated target labels.
    """
    target_action = str(target_row.get("gain_oracle_action", "unknown"))
    correct_vs_no = float(pd.to_numeric(target_row.get("correct_vs_no", 0.0), errors="coerce"))

    if source_bucket_name in ["direct_source", "same_operator_source"]:
        if target_action == "direct_reuse":
            return "direct_reuse", "useful", correct_vs_no
        if target_action == "observe":
            return "observe", "neutral", 0.0
        return "reject", "harmful", min(0.0, correct_vs_no)

    if source_bucket_name == "observe_source":
        if target_action == "observe":
            return "observe", "neutral", 0.0
        if target_action == "direct_reuse":
            return "direct_reuse", "useful", correct_vs_no
        return "reject", "harmful", min(0.0, correct_vs_no)

    if source_bucket_name in ["reject_source", "broad_source"]:
        if target_action == "reject":
            return "reject", "neutral", 0.0
        if target_action == "observe":
            return "observe", "neutral", 0.0
        # A useful target paired with reject/broad source should learn mismatch.
        return "reject", "harmful", 0.0

    return target_action, "unknown", correct_vs_no


def build_pair_features(
    target_row: pd.Series,
    source_row: pd.Series,
    target_vec: np.ndarray,
    source_vec: np.ndarray,
    feature_cols: List[str],
    source_bucket: str,
) -> Dict:
    diff = target_vec - source_vec
    absdiff = np.abs(diff)
    prod = target_vec * source_vec

    pair_action, pair_utility, pair_gain = infer_pair_label(target_row, source_bucket, source_row)

    out = {
        "pair_id": f"{source_row['task_id']}__PAIR__{target_row['task_id']}__{source_bucket}",
        "source_task_id": str(source_row["task_id"]),
        "target_task_id": str(target_row["task_id"]),
        "source_bucket": source_bucket,

        "source_reuse_group": str(source_row.get("reuse_group", "")),
        "target_reuse_group": str(target_row.get("reuse_group", "")),
        "source_task_family": str(source_row.get("task_family", "")),
        "target_task_family": str(target_row.get("task_family", "")),
        "source_operator": str(source_row.get("target_operator", "")),
        "target_operator": str(target_row.get("target_operator", "")),
        "source_concept": str(source_row.get("concept", "")),
        "target_concept": str(target_row.get("concept", "")),

        "source_gain_oracle_action": str(source_row.get("gain_oracle_action", "")),
        "target_gain_oracle_action": str(target_row.get("gain_oracle_action", "")),
        "target_best_policy_action": str(target_row.get("best_policy_action", "")),
        "target_best_policy_id": str(target_row.get("best_policy_id", "")),

        "pair_action_label": pair_action,
        "pair_utility_label": pair_utility,
        "pair_gain": pair_gain,

        "same_reuse_group": int(str(source_row.get("reuse_group", "")) == str(target_row.get("reuse_group", ""))),
        "same_task_family": int(str(source_row.get("task_family", "")) == str(target_row.get("task_family", ""))),
        "same_operator": int(str(source_row.get("target_operator", "")) == str(target_row.get("target_operator", ""))),
        "same_concept": int(str(source_row.get("concept", "")) == str(target_row.get("concept", ""))),

        "source_boundary_strength": float(pd.to_numeric(source_row.get("boundary_strength", np.nan), errors="coerce")),
        "target_boundary_strength": float(pd.to_numeric(target_row.get("boundary_strength", np.nan), errors="coerce")),
    }

    out["boundary_strength_delta_target_minus_source"] = out["target_boundary_strength"] - out["source_boundary_strength"]
    out["boundary_strength_abs_delta"] = abs(out["boundary_strength_delta_target_minus_source"])

    out["vec_l2"] = float(np.linalg.norm(diff))
    out["vec_l1_mean"] = float(np.mean(absdiff))
    out["vec_linf"] = float(np.max(absdiff))
    out["vec_cos"] = cosine(target_vec, source_vec)
    out["vec_dot_mean"] = float(np.mean(prod))
    out["target_norm"] = float(np.linalg.norm(target_vec))
    out["source_norm"] = float(np.linalg.norm(source_vec))
    out["norm_ratio_target_over_source"] = float(out["target_norm"] / max(out["source_norm"], 1e-12))

    # Aggregate deltas by feature family.
    prefixes = {
        "init": ["_init_0_6_", "_L0_", "_L1_", "_L2_", "_L3_", "_L4_", "_L5_", "_L6_"],
        "mid": ["_mid_sparse_7_19_", "_L7_", "_L10_", "_L13_", "_L16_", "_L19_"],
        "boundary": ["_boundary_20_25_", "_L20_", "_L21_", "_L22_", "_L23_", "_L24_", "_L25_"],
        "entropy": ["entropy"],
        "spread": ["spread"],
        "center": ["center"],
        "jaccard": ["jaccard"],
    }

    for name, pats in prefixes.items():
        idx = [i for i, c in enumerate(feature_cols) if any(p in c for p in pats)]
        if not idx:
            continue
        sub_diff = diff[idx]
        sub_abs = absdiff[idx]
        sub_t = target_vec[idx]
        sub_s = source_vec[idx]
        out[f"{name}_l2"] = float(np.linalg.norm(sub_diff))
        out[f"{name}_l1_mean"] = float(np.mean(sub_abs))
        out[f"{name}_cos"] = cosine(sub_t, sub_s)
        out[f"{name}_delta_mean"] = float(np.mean(sub_diff))
        out[f"{name}_abs_delta_max"] = float(np.max(sub_abs))

    return out


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    df = read_csv(IN_FEATURES)

    required = [
        "task_id", "reuse_group", "task_family", "target_operator", "concept",
        "gain_oracle_action", "best_policy_action", "best_policy_id",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for c in required:
        df[c] = df[c].fillna("NA").astype(str)

    if "correct_vs_no" not in df.columns:
        df["correct_vs_no"] = 0.0

    groups = topk_feature_groups(df)
    feature_cols = groups.get(FEATURE_GROUP, [])
    if not feature_cols:
        raise ValueError(f"No feature columns for FEATURE_GROUP={FEATURE_GROUP}")

    X = clean_numeric(df, feature_cols)

    source_buckets = sample_sources(df, rng)

    print("[SOURCE BUCKETS]")
    for k, v in source_buckets.items():
        print(f"  {k}: {len(v)}")

    rows = []

    for target_idx, target_row in df.iterrows():
        target_vec = X.loc[target_idx].values.astype(np.float64)

        for bucket_name, bucket_df in source_buckets.items():
            for _, source_row in bucket_df.iterrows():
                source_task_id = str(source_row["task_id"])
                # Find source original index.
                src_matches = df.index[df["task_id"].astype(str) == source_task_id].tolist()
                if not src_matches:
                    continue
                source_idx = src_matches[0]
                source_vec = X.loc[source_idx].values.astype(np.float64)

                rows.append(
                    build_pair_features(
                        target_row=target_row,
                        source_row=source_row,
                        target_vec=target_vec,
                        source_vec=source_vec,
                        feature_cols=feature_cols,
                        source_bucket=bucket_name,
                    )
                )

    pair_df = pd.DataFrame(rows)
    pair_df.to_csv(OUT_PAIRWISE, index=False, encoding="utf-8-sig")

    summary = {
        "input_features": str(IN_FEATURES),
        "output_pairwise": str(OUT_PAIRWISE),
        "feature_group": FEATURE_GROUP,
        "n_original_rows": int(len(df)),
        "n_pair_rows": int(len(pair_df)),
        "n_feature_cols_used": int(len(feature_cols)),
        "source_buckets": {k: int(len(v)) for k, v in source_buckets.items()},
        "pair_action_label_counts": pair_df["pair_action_label"].value_counts(dropna=False).to_dict(),
        "pair_utility_label_counts": pair_df["pair_utility_label"].value_counts(dropna=False).to_dict(),
        "target_reuse_group_counts": pair_df["target_reuse_group"].value_counts(dropna=False).to_dict(),
        "source_bucket_counts": pair_df["source_bucket"].value_counts(dropna=False).to_dict(),
        "verdict": "SR3A_PAIRWISE_DATASET_READY",
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[SAVED]")
    print(f"  pairwise: {OUT_PAIRWISE}")
    print(f"  summary : {OUT_SUMMARY}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
