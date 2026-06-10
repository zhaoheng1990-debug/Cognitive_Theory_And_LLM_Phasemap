# -*- coding: utf-8 -*-
"""
OA-5D: Prompt-Conditioned Geometry Audit
=======================================

Motivation
----------
OA-5C.2b detected a strong coupled direction field among task fibers.
New hypothesis:

    The observed fiber/coupling field is not bare W geometry.
    It is prompt-conditioned geometry.

More specifically:

    W = M provides local neighborhood N_epsilon(x)
    Prompt provides directional/spectral bias Sigma_0
    Observed TopK geometry = prompt-conditioned reweighting of local neighborhood

Core prediction
---------------
For same topic/entity but different task prompts:

    1. Bare neighborhood overlap should remain nontrivial.
       N_epsilon(x; topic) is shared.

    2. Directional spectrum should change strongly.
       Prompt selects/reweights directions inside the shared neighborhood.

    3. Therefore:
       identity overlap / center proximity may be moderate,
       but anisotropy / PCA orientation / logit-weighted direction differs.

Operational tests
-----------------
A. Same-topic / different-task neighborhood sharing:
   Jaccard overlap and center distance.

B. Prompt-conditioned direction shift:
   Compare task-conditioned center vectors relative to topic centroid:
       v_task = C(topic, task) - C(topic)
   Measure angles among v_task.

C. Spectrum anisotropy:
   Compare spread/logit_gap/logit_std across tasks.
   Compute task separation in spectrum-feature space.

D. Direction-field reweighting:
   Weighted direction proxy:
       q = center * logit_gap or center / spread
   Compare whether weighted directions separate tasks more than raw centers.

Inputs
------
Uses OA-5C.1 data:
    C:\\Users\\ZH\\Desktop\\AGI\\python_script\\oa5c1_outputs\\oa5c1_topk_center_records.csv

Outputs
-------
oa5d_outputs/
    oa5d_neighborhood_overlap_summary.csv
    oa5d_direction_spectrum_summary.csv
    oa5d_prompt_conditioning_summary.csv
    oa5d_verdict_summary.csv
"""

import json
import math
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

# ============================================================
# PATH CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\ZH\Desktop\AGI\python_script")
INPUT_FILE = BASE_DIR / "oa5c1_outputs" / "oa5c1_topk_center_records.csv"

OUTPUT_DIR = BASE_DIR / "oa5d_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_OVERLAP = OUTPUT_DIR / "oa5d_neighborhood_overlap_summary.csv"
OUT_DIRECTION = OUTPUT_DIR / "oa5d_direction_spectrum_summary.csv"
OUT_CONDITION = OUTPUT_DIR / "oa5d_prompt_conditioning_summary.csv"
OUT_VERDICT = OUTPUT_DIR / "oa5d_verdict_summary.csv"

# ============================================================
# CONFIG
# ============================================================

SHALLOW_GROUPS = {
    "L0": [0],
    "L0_2": [0, 1, 2],
    "L0_6": list(range(0, 7)),
}

RANDOM_SEED = 42

# ============================================================
# HELPERS
# ============================================================

def load_data():
    print("=" * 80)
    print("OA-5D Prompt-Conditioned Geometry Audit")
    print("=" * 80)
    print("INPUT_FILE:", INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    center_cols = [c for c in df.columns if c.startswith("center_")]
    center_cols = sorted(center_cols, key=lambda x: int(x.split("_")[1]))

    if len(center_cols) < 8:
        raise ValueError("No center_* columns found.")

    required = [
        "prompt_id", "topic", "task", "paraphrase_id",
        "layer_index", "k", "topk_ids",
        "spread", "logit_mean", "logit_std", "logit_gap",
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column {c}")

    print("Rows:", len(df))
    print("Center dim:", len(center_cols))
    print("Prompts:", df["prompt_id"].nunique())
    print("Topics:", sorted(df["topic"].unique().tolist()))
    print("Tasks:", sorted(df["task"].unique().tolist()))
    print("K:", sorted(df["k"].unique().tolist()))
    print("Layers:", sorted(df["layer_index"].unique().tolist()))

    return df, center_cols


def parse_ids(s):
    if isinstance(s, list):
        return set(map(str, s))
    if pd.isna(s):
        return set()
    try:
        obj = json.loads(str(s))
        return set(map(str, obj))
    except Exception:
        txt = str(s).strip()
        if txt.startswith("[") and txt.endswith("]"):
            txt = txt[1:-1]
        return set([x.strip() for x in txt.split(",") if x.strip()])


def aggregate_by_prompt(df, center_cols, layers, k):
    sub = df[(df["layer_index"].isin(layers)) & (df["k"] == k)].copy()
    if sub.empty:
        return None, None, None

    rows = []
    X = []
    sets = []

    for pid, g in sub.groupby("prompt_id"):
        first = g.iloc[0]
        x = g[center_cols].to_numpy(dtype=np.float32).mean(axis=0)

        union_ids = set()
        for val in g["topk_ids"].values:
            union_ids |= parse_ids(val)

        rows.append({
            "prompt_id": str(pid),
            "topic": str(first["topic"]),
            "task": str(first["task"]),
            "paraphrase_id": str(first["paraphrase_id"]),
            "spread": float(g["spread"].mean()),
            "logit_mean": float(g["logit_mean"].mean()),
            "logit_std": float(g["logit_std"].mean()),
            "logit_gap": float(g["logit_gap"].mean()),
        })
        X.append(x)
        sets.append(union_ids)

    meta = pd.DataFrame(rows)
    X = np.stack(X).astype(np.float32)
    return meta, X, sets


def cosine_distance_matrix(X):
    X = np.asarray(X, dtype=np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return np.clip(1.0 - X @ X.T, 0.0, 2.0)


def cosine_sim(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


def jaccard(a, b):
    u = len(a | b)
    if u == 0:
        return np.nan
    return len(a & b) / u


def mean(vals):
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def classify_task_cv(meta, features, groups=None):
    y = meta["task"].astype(str).values
    X = np.asarray(features, dtype=np.float32)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ])

    # Stratified CV
    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy").mean()
        f1 = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro").mean()
    except Exception:
        acc, f1 = np.nan, np.nan

    # Leave-topic-out task classification
    logo_acc, logo_f1 = [], []
    if groups is not None:
        logo = LeaveOneGroupOut()
        for tr, te in logo.split(X, y, groups=groups):
            if len(set(y[tr])) < 2 or len(set(y[te])) < 1:
                continue
            try:
                pipe.fit(X[tr], y[tr])
                pred = pipe.predict(X[te])
                logo_acc.append(accuracy_score(y[te], pred))
                logo_f1.append(f1_score(y[te], pred, average="macro", zero_division=0))
            except Exception:
                pass

    return {
        "task_acc_cv": float(np.mean(acc)) if np.isfinite(acc) else np.nan,
        "task_f1_cv": float(np.mean(f1)) if np.isfinite(f1) else np.nan,
        "task_acc_leave_topic_out": float(np.mean(logo_acc)) if logo_acc else np.nan,
        "task_f1_leave_topic_out": float(np.mean(logo_f1)) if logo_f1 else np.nan,
    }


# ============================================================
# ANALYSIS A: NEIGHBORHOOD OVERLAP
# ============================================================

def neighborhood_overlap_analysis(df, center_cols):
    rows = []

    for k in sorted(df["k"].unique()):
        for lg, layers in SHALLOW_GROUPS.items():
            meta, X, sets = aggregate_by_prompt(df, center_cols, layers, k)
            if meta is None:
                continue

            Dcos = cosine_distance_matrix(X)
            Deuc = pairwise_distances(StandardScaler().fit_transform(X), metric="euclidean")

            categories = defaultdict(list)
            jaccs = defaultdict(list)

            n = len(meta)
            for i in range(n):
                for j in range(i + 1, n):
                    same_topic = meta.loc[i, "topic"] == meta.loc[j, "topic"]
                    same_task = meta.loc[i, "task"] == meta.loc[j, "task"]

                    if same_topic and same_task:
                        cat = "same_topic_same_task"
                    elif same_topic and not same_task:
                        cat = "same_topic_diff_task"
                    elif (not same_topic) and same_task:
                        cat = "diff_topic_same_task"
                    else:
                        cat = "diff_topic_diff_task"

                    categories[(cat, "cosine_dist")].append(Dcos[i, j])
                    categories[(cat, "euclidean_dist")].append(Deuc[i, j])
                    jaccs[cat].append(jaccard(sets[i], sets[j]))

            row = {
                "k": int(k),
                "layer_group": lg,
                "n_prompts": len(meta),
            }

            for cat in [
                "same_topic_same_task",
                "same_topic_diff_task",
                "diff_topic_same_task",
                "diff_topic_diff_task",
            ]:
                row[f"{cat}_cosine_dist"] = mean(categories[(cat, "cosine_dist")])
                row[f"{cat}_euclidean_dist"] = mean(categories[(cat, "euclidean_dist")])
                row[f"{cat}_jaccard"] = mean(jaccs[cat])

            # Core ratios
            row["shared_neighborhood_jaccard_ratio_same_topic_diff_task_vs_diff_topic_diff_task"] = (
                row["same_topic_diff_task_jaccard"] / (row["diff_topic_diff_task_jaccard"] + 1e-12)
                if np.isfinite(row["diff_topic_diff_task_jaccard"]) else np.nan
            )
            row["shared_neighborhood_dist_ratio_same_topic_diff_task_vs_diff_topic_diff_task"] = (
                row["same_topic_diff_task_cosine_dist"] / (row["diff_topic_diff_task_cosine_dist"] + 1e-12)
                if np.isfinite(row["diff_topic_diff_task_cosine_dist"]) else np.nan
            )
            row["task_continuity_dist_ratio_same_task_vs_diff_task_same_topic"] = (
                row["same_topic_same_task_cosine_dist"] / (row["same_topic_diff_task_cosine_dist"] + 1e-12)
                if np.isfinite(row["same_topic_diff_task_cosine_dist"]) else np.nan
            )

            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# ANALYSIS B: DIRECTION SPECTRUM
# ============================================================

def direction_spectrum_analysis(df, center_cols):
    rows = []

    for k in sorted(df["k"].unique()):
        for lg, layers in SHALLOW_GROUPS.items():
            meta, X, sets = aggregate_by_prompt(df, center_cols, layers, k)
            if meta is None:
                continue

            # Topic centroid and task directions inside each topic
            for topic, gt in meta.groupby("topic"):
                idx_topic = gt.index.values
                Xtopic = X[idx_topic]
                topic_centroid = Xtopic.mean(axis=0)

                task_dirs = {}
                task_stats = {}

                for task, gtt in gt.groupby("task"):
                    idx = gtt.index.values
                    xmean = X[idx].mean(axis=0)
                    v = xmean - topic_centroid
                    task_dirs[str(task)] = v

                    task_stats[str(task)] = {
                        "spread_mean": float(meta.loc[idx, "spread"].mean()),
                        "logit_gap_mean": float(meta.loc[idx, "logit_gap"].mean()),
                        "logit_std_mean": float(meta.loc[idx, "logit_std"].mean()),
                        "n": int(len(idx)),
                    }

                tasks = sorted(task_dirs.keys())
                for i, ti in enumerate(tasks):
                    for tj in tasks[i + 1:]:
                        vi = task_dirs[ti]
                        vj = task_dirs[tj]

                        rows.append({
                            "k": int(k),
                            "layer_group": lg,
                            "topic": topic,
                            "task_i": ti,
                            "task_j": tj,
                            "direction_cosine": cosine_sim(vi, vj),
                            "direction_angle_proxy": 1.0 - cosine_sim(vi, vj),
                            "norm_i": float(np.linalg.norm(vi)),
                            "norm_j": float(np.linalg.norm(vj)),
                            "spread_i": task_stats[ti]["spread_mean"],
                            "spread_j": task_stats[tj]["spread_mean"],
                            "logit_gap_i": task_stats[ti]["logit_gap_mean"],
                            "logit_gap_j": task_stats[tj]["logit_gap_mean"],
                            "logit_std_i": task_stats[ti]["logit_std_mean"],
                            "logit_std_j": task_stats[tj]["logit_std_mean"],
                        })

    return pd.DataFrame(rows)


# ============================================================
# ANALYSIS C: PROMPT CONDITIONING
# ============================================================

def prompt_conditioning_analysis(df, center_cols):
    rows = []

    for k in sorted(df["k"].unique()):
        for lg, layers in SHALLOW_GROUPS.items():
            meta, X, sets = aggregate_by_prompt(df, center_cols, layers, k)
            if meta is None:
                continue

            # Raw center features
            raw_features = X

            # Spectrum-only features
            spec_features = meta[["spread", "logit_mean", "logit_std", "logit_gap"]].to_numpy(dtype=np.float32)

            # Weighted direction proxies
            gap = meta["logit_gap"].to_numpy(dtype=np.float32)
            spread = meta["spread"].to_numpy(dtype=np.float32)
            weighted_gap = X * gap[:, None]
            weighted_inv_spread = X / (spread[:, None] + 1e-6)

            groups = meta["topic"].astype(str).values

            raw_cls = classify_task_cv(meta, raw_features, groups=groups)
            spec_cls = classify_task_cv(meta, spec_features, groups=groups)
            wg_cls = classify_task_cv(meta, weighted_gap, groups=groups)
            wis_cls = classify_task_cv(meta, weighted_inv_spread, groups=groups)

            # If prompt-conditioning is reweighting directions,
            # weighted features should improve task separability over raw centers
            rows.append({
                "k": int(k),
                "layer_group": lg,
                "n_prompts": int(len(meta)),

                "raw_task_acc_cv": raw_cls["task_acc_cv"],
                "raw_task_f1_cv": raw_cls["task_f1_cv"],
                "raw_task_acc_leave_topic_out": raw_cls["task_acc_leave_topic_out"],
                "raw_task_f1_leave_topic_out": raw_cls["task_f1_leave_topic_out"],

                "spectrum_task_acc_cv": spec_cls["task_acc_cv"],
                "spectrum_task_f1_cv": spec_cls["task_f1_cv"],
                "spectrum_task_acc_leave_topic_out": spec_cls["task_acc_leave_topic_out"],
                "spectrum_task_f1_leave_topic_out": spec_cls["task_f1_leave_topic_out"],

                "weighted_gap_task_acc_cv": wg_cls["task_acc_cv"],
                "weighted_gap_task_f1_cv": wg_cls["task_f1_cv"],
                "weighted_gap_task_acc_leave_topic_out": wg_cls["task_acc_leave_topic_out"],
                "weighted_gap_task_f1_leave_topic_out": wg_cls["task_f1_leave_topic_out"],

                "weighted_invspread_task_acc_cv": wis_cls["task_acc_cv"],
                "weighted_invspread_task_f1_cv": wis_cls["task_f1_cv"],
                "weighted_invspread_task_acc_leave_topic_out": wis_cls["task_acc_leave_topic_out"],
                "weighted_invspread_task_f1_leave_topic_out": wis_cls["task_f1_leave_topic_out"],

                "weighted_gap_gain_leave_topic_out": wg_cls["task_acc_leave_topic_out"] - raw_cls["task_acc_leave_topic_out"],
                "weighted_invspread_gain_leave_topic_out": wis_cls["task_acc_leave_topic_out"] - raw_cls["task_acc_leave_topic_out"],
                "spectrum_only_gap_from_raw_leave_topic_out": spec_cls["task_acc_leave_topic_out"] - raw_cls["task_acc_leave_topic_out"],
            })

    return pd.DataFrame(rows)


# ============================================================
# VERDICT
# ============================================================

def verdict_summary(overlap_df, direction_df, condition_df):
    rows = []

    for _, c in condition_df.iterrows():
        k = c["k"]
        lg = c["layer_group"]

        o = overlap_df[(overlap_df["k"] == k) & (overlap_df["layer_group"] == lg)]
        d = direction_df[(direction_df["k"] == k) & (direction_df["layer_group"] == lg)]

        if o.empty:
            continue
        o = o.iloc[0]

        # Shared neighborhood: same topic diff task more similar than fully diff
        # For Jaccard: ratio > 1 means same-topic-diff-task shares more topk identity
        shared_jacc_ratio = float(o["shared_neighborhood_jaccard_ratio_same_topic_diff_task_vs_diff_topic_diff_task"])
        shared_dist_ratio = float(o["shared_neighborhood_dist_ratio_same_topic_diff_task_vs_diff_topic_diff_task"])
        task_continuity_ratio = float(o["task_continuity_dist_ratio_same_task_vs_diff_task_same_topic"])

        # Direction shift: task direction cosine lower / angle proxy higher
        mean_angle_proxy = float(d["direction_angle_proxy"].mean()) if len(d) else np.nan
        mean_abs_cos = float(np.mean(np.abs(d["direction_cosine"]))) if len(d) else np.nan
        mean_dir_norm = float(np.mean([d["norm_i"].mean(), d["norm_j"].mean()])) if len(d) else np.nan

        raw_acc = float(c["raw_task_acc_leave_topic_out"])
        spectrum_acc = float(c["spectrum_task_acc_leave_topic_out"])
        wg_acc = float(c["weighted_gap_task_acc_leave_topic_out"])
        wis_acc = float(c["weighted_invspread_task_acc_leave_topic_out"])

        best_weighted = max(wg_acc, wis_acc)
        weighted_gain = best_weighted - raw_acc

        # Flags
        shared_neighborhood = (
            np.isfinite(shared_jacc_ratio) and shared_jacc_ratio > 1.05
        ) or (
            np.isfinite(shared_dist_ratio) and shared_dist_ratio < 0.97
        )

        same_task_more_continuous = np.isfinite(task_continuity_ratio) and task_continuity_ratio < 0.98

        direction_reweighted = (
            np.isfinite(mean_angle_proxy) and mean_angle_proxy > 0.05
        ) or (
            np.isfinite(weighted_gain) and weighted_gain > 0.02
        )

        spectrum_carries_task = np.isfinite(spectrum_acc) and spectrum_acc > 0.35

        prompt_conditioning = (
            np.isfinite(raw_acc) and raw_acc > 0.60
        )

        score = sum([
            shared_neighborhood,
            same_task_more_continuous,
            direction_reweighted,
            spectrum_carries_task,
            prompt_conditioning,
        ])

        if score >= 4:
            verdict = "PASS-Strong: prompt-conditioned geometry supported"
        elif score == 3:
            verdict = "PASS-Moderate: partial prompt-conditioned geometry"
        elif score == 2:
            verdict = "PASS-Lite/Mixed"
        else:
            verdict = "FAIL/Mixed"

        rows.append({
            "k": int(k),
            "layer_group": lg,
            "shared_jaccard_ratio_same_topic_diff_task_vs_diff_topic_diff_task": shared_jacc_ratio,
            "shared_distance_ratio_same_topic_diff_task_vs_diff_topic_diff_task": shared_dist_ratio,
            "task_continuity_ratio_same_task_vs_diff_task_same_topic": task_continuity_ratio,
            "mean_direction_angle_proxy": mean_angle_proxy,
            "mean_abs_direction_cosine": mean_abs_cos,
            "mean_direction_norm": mean_dir_norm,
            "raw_task_acc_leave_topic_out": raw_acc,
            "spectrum_task_acc_leave_topic_out": spectrum_acc,
            "weighted_gap_task_acc_leave_topic_out": wg_acc,
            "weighted_invspread_task_acc_leave_topic_out": wis_acc,
            "best_weighted_task_acc_leave_topic_out": best_weighted,
            "weighted_gain_over_raw": weighted_gain,
            "shared_neighborhood": shared_neighborhood,
            "same_task_more_continuous": same_task_more_continuous,
            "direction_reweighted": direction_reweighted,
            "spectrum_carries_task": spectrum_carries_task,
            "prompt_conditioning": prompt_conditioning,
            "score": int(score),
            "verdict": verdict,
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    df, center_cols = load_data()

    print("\nRunning neighborhood overlap analysis...")
    overlap_df = neighborhood_overlap_analysis(df, center_cols)
    overlap_df.to_csv(OUT_OVERLAP, index=False, encoding="utf-8-sig")
    print("Saved:", OUT_OVERLAP)

    print("\nRunning direction spectrum analysis...")
    direction_df = direction_spectrum_analysis(df, center_cols)
    direction_df.to_csv(OUT_DIRECTION, index=False, encoding="utf-8-sig")
    print("Saved:", OUT_DIRECTION)

    print("\nRunning prompt conditioning analysis...")
    condition_df = prompt_conditioning_analysis(df, center_cols)
    condition_df.to_csv(OUT_CONDITION, index=False, encoding="utf-8-sig")
    print("Saved:", OUT_CONDITION)

    print("\nBuilding verdict...")
    verdict_df = verdict_summary(overlap_df, direction_df, condition_df)
    verdict_df.to_csv(OUT_VERDICT, index=False, encoding="utf-8-sig")
    print("Saved:", OUT_VERDICT)

    print("\n=== OA-5D VERDICT PREVIEW ===")
    print(verdict_df.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
