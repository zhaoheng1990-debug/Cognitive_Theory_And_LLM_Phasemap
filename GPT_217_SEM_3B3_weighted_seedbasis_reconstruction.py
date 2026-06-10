# -*- coding: utf-8 -*-
"""
GPT_207_SEM_3B3_weighted_seedbasis_reconstruction.py

SEM-3B.3: Weighted SeedBasis Reconstruction
-------------------------------------------

Goal:
    Test whether SeedFamily center μ_F is better approximated by a weighted
    SeedBasis than by:
      - a single prompt
      - uniform PromptSet
      - random-family basis

Core hypothesis:
    μ_F is not generally realizable as one prompt.
    μ_F may be approximated by:
        μ_F ≈ Σ_i w_i T(p_i)
    where:
        w_i >= 0, Σ_i w_i = 1

Inputs:
    sem3a_outputs/
      sem3a_dataset.csv
      sem3a_features.csv

    sem3b2_outputs/
      sem3b2_seedbasis_dataset.csv
      sem3b2_seedbasis_features.csv
      sem3b2_promptset_centers.csv
      sem3b2_pairwise_regeneration.csv

Outputs:
    sem3b3_outputs/
      sem3b3_config.json
      sem3b3_weighted_basis_weights.csv
      sem3b3_weighted_centers.csv
      sem3b3_pairwise_reconstruction.csv
      sem3b3_strategy_summary.csv
      sem3b3_family_summary.csv
      sem3b3_results_summary.json

Run:
    python GPT_207_SEM_3B3_weighted_seedbasis_reconstruction.py

Notes:
    - This script does NOT forward model again.
    - It operates entirely on SEM-3A original features and SEM-3B.2 basis features.
    - Weighted basis is fit in each feature block separately.
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.stats import ttest_rel, wilcoxon


# ============================================================
# CONFIG
# ============================================================

@dataclass
class CFG:
    SEM3A_DIR: str = "./sem3a_outputs"
    SEM3B2_DIR: str = "./sem3b2_outputs"
    OUTPUT_DIR: str = "./sem3b3_outputs"

    SEM3A_FEATURES: str = "sem3a_features.csv"
    SEM3B2_BASIS_DATASET: str = "sem3b2_seedbasis_dataset.csv"
    SEM3B2_BASIS_FEATURES: str = "sem3b2_seedbasis_features.csv"

    MAIN_BLOCK: str = "center_pca"
    BLOCKS: Tuple[str, ...] = ("center_pca", "decision", "mid", "all", "scalar", "transport", "init")

    # Candidate basis prompt types from SEM-3B.2.
    BASIS_TYPES: Tuple[str, ...] = (
        "nearest_exemplar",
        "nearest_plus_structure",
        "natural_prompt_star",
        "structured_prompt_star",
        "compressed_prompt_star",
        "canonical_minimal_seed",
    )

    # Strategies to evaluate.
    # weighted_* are fitted to approximate μ_F.
    WEIGHTED_STRATEGIES: Tuple[str, ...] = (
        "weighted_simplex_l2",
        "weighted_simplex_cosine",
        "weighted_ridge_simplex",
    )

    RANDOM_SEED: int = 42
    RIDGE_LAMBDA: float = 0.05


cfg = CFG()


# ============================================================
# UTILS
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def json_dump(obj: Any, path: Path):
    def convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, float) and np.isnan(o):
            return None
        return str(o)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=convert)


def cosine_np(a, b, eps=1e-9):
    a = np.asarray(a)
    b = np.asarray(b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ============================================================
# FEATURE BLOCKS
# ============================================================

def get_feature_cols(df: pd.DataFrame, block: str) -> List[str]:
    exclude = {
        "row_id", "row_type", "seed_family", "concept_star", "operator_star",
        "surface_id", "prompt", "basis_prompt_type", "source_row_id",
        "operator_id", "common_structure"
    }
    numeric = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    if block == "all":
        return numeric
    if block == "scalar":
        return [c for c in numeric if "centerPC" not in c]
    if block == "center_pca":
        return [c for c in numeric if "centerPC" in c]
    if block == "transport":
        return [c for c in numeric if any(k in c for k in ["shift", "jaccard"])]
    if block == "init":
        return [c for c in numeric if any(f"_L{i}_" in c for i in range(0, 7))]
    if block == "mid":
        return [c for c in numeric if any(f"_L{i}_" in c for i in range(7, 20))]
    if block == "decision":
        return [c for c in numeric if any(f"_L{i}_" in c for i in range(20, 26))]
    raise ValueError(block)


def align_spaces(original_features: pd.DataFrame, basis_features: pd.DataFrame, block: str):
    cols = sorted(set(get_feature_cols(original_features, block)) & set(get_feature_cols(basis_features, block)))
    if not cols:
        return None, None, []

    combined = pd.concat([
        original_features[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0),
        basis_features[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0),
    ], axis=0)
    scaler = StandardScaler()
    scaler.fit(combined.values.astype(np.float32))

    Xo = scaler.transform(original_features[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32))
    Xb = scaler.transform(basis_features[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32))
    return Xo, Xb, cols


# ============================================================
# LOAD DATA
# ============================================================

def load_inputs(cfg: CFG):
    sem3a_dir = Path(cfg.SEM3A_DIR)
    sem3b2_dir = Path(cfg.SEM3B2_DIR)

    original_features = pd.read_csv(sem3a_dir / cfg.SEM3A_FEATURES)
    basis_dataset = pd.read_csv(sem3b2_dir / cfg.SEM3B2_BASIS_DATASET)
    basis_features = pd.read_csv(sem3b2_dir / cfg.SEM3B2_BASIS_FEATURES)

    # Ensure basis feature table has metadata if not merged.
    if "basis_prompt_type" not in basis_features.columns:
        basis_features = basis_dataset.merge(basis_features, on="row_id", how="left")

    original_features = original_features[original_features["row_type"] == "original_prompt"].copy()

    return original_features, basis_features


# ============================================================
# WEIGHT FITTING
# ============================================================

def simplex_project(w):
    w = np.maximum(w, 0)
    s = w.sum()
    if s <= 1e-12:
        return np.ones_like(w) / len(w)
    return w / s


def fit_weights(B: np.ndarray, target: np.ndarray, strategy: str, ridge_lambda: float):
    """
    B shape: n_basis x d
    target: d
    returns weights n_basis
    """
    n = B.shape[0]
    if n == 1:
        return np.array([1.0])

    init = np.ones(n) / n
    bounds = [(0.0, 1.0) for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if strategy == "weighted_simplex_l2":
        def obj(w):
            pred = w @ B
            return float(np.mean((pred - target) ** 2))
    elif strategy == "weighted_simplex_cosine":
        def obj(w):
            pred = w @ B
            return float(1.0 - cosine_np(pred, target))
    elif strategy == "weighted_ridge_simplex":
        def obj(w):
            pred = w @ B
            ridge = ridge_lambda * np.sum((w - init) ** 2)
            return float(np.mean((pred - target) ** 2) + ridge)
    else:
        raise ValueError(strategy)

    res = minimize(obj, init, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 500, "ftol": 1e-9})
    if not res.success:
        return init
    return simplex_project(res.x)


def build_family_targets_and_basis(original_features, basis_features, Xo, Xb):
    orig = original_features.reset_index(drop=True)
    basis = basis_features.reset_index(drop=True)

    fam_targets = {}
    fam_basis = {}

    for fam in sorted(orig["seed_family"].unique()):
        idx_orig = np.where(orig["seed_family"].values == fam)[0]
        if len(idx_orig) == 0:
            continue
        fam_targets[fam] = Xo[idx_orig, :].mean(axis=0)

        basis_idxs = []
        basis_types = []
        for btype in cfg.BASIS_TYPES:
            idx = np.where((basis["seed_family"].values == fam) & (basis["basis_prompt_type"].values == btype))[0]
            if len(idx):
                basis_idxs.append(int(idx[0]))
                basis_types.append(btype)
        if basis_idxs:
            fam_basis[fam] = {
                "idxs": basis_idxs,
                "types": basis_types,
                "B": Xb[basis_idxs, :],
            }

    return fam_targets, fam_basis


def fit_all_weighted_centers(original_features, basis_features, block: str):
    Xo, Xb, cols = align_spaces(original_features, basis_features, block)
    if Xo is None:
        return pd.DataFrame(), pd.DataFrame(), None, None, []

    fam_targets, fam_basis = build_family_targets_and_basis(original_features, basis_features, Xo, Xb)

    weight_rows = []
    center_rows = []
    center_vectors = {}

    for fam, target in fam_targets.items():
        if fam not in fam_basis:
            continue
        item = fam_basis[fam]
        B = item["B"]
        types = item["types"]

        # Baseline centers
        centers = {
            "uniform_full_basis": np.ones(len(types)) / len(types),
        }
        # Single basis baselines
        for i, t in enumerate(types):
            w = np.zeros(len(types))
            w[i] = 1.0
            centers[f"single_{t}"] = w

        # Weighted strategies
        for strategy in cfg.WEIGHTED_STRATEGIES:
            centers[strategy] = fit_weights(B, target, strategy, cfg.RIDGE_LAMBDA)

        for strategy, w in centers.items():
            center = w @ B
            center_vectors[(fam, strategy)] = center

            weight_rows.append({
                "block": block,
                "seed_family": fam,
                "strategy": strategy,
                "basis_types": "|".join(types),
                "weights_json": json.dumps({t: float(wi) for t, wi in zip(types, w)}, ensure_ascii=False),
                "target_cos": float(cosine_np(center, target)),
                "target_l2": float(np.linalg.norm(center - target)),
                "n_basis": int(len(types)),
            })

            center_rows.append({
                "block": block,
                "seed_family": fam,
                "strategy": strategy,
                "basis_types": "|".join(types),
                "center_vector_json": json.dumps(center.tolist()),
                "target_cos": float(cosine_np(center, target)),
                "target_l2": float(np.linalg.norm(center - target)),
                "n_basis": int(len(types)),
            })

    return pd.DataFrame(weight_rows), pd.DataFrame(center_rows), center_vectors, Xo, cols


# ============================================================
# RECONSTRUCTION COMPARISON
# ============================================================

def compare_weighted_centers(original_features, block: str, center_vectors: Dict[Tuple[str, str], np.ndarray], Xo: np.ndarray):
    orig = original_features.reset_index(drop=True)
    all_keys = list(center_vectors.keys())

    rows = []
    for oi, row in orig.iterrows():
        true_fam = row["seed_family"]
        x = Xo[oi]

        strategies = sorted(set(k[1] for k in all_keys))
        for strategy in strategies:
            key = (true_fam, strategy)
            if key not in center_vectors:
                continue
            same_vec = center_vectors[key]
            same_cos = cosine_np(x, same_vec)
            same_l2 = float(np.linalg.norm(x - same_vec))

            rand_keys = [(fam, s) for (fam, s) in all_keys if s == strategy and fam != true_fam]
            if rand_keys:
                rand_cos = np.array([cosine_np(x, center_vectors[k]) for k in rand_keys])
                rand_l2 = np.array([np.linalg.norm(x - center_vectors[k]) for k in rand_keys])
                rand_mean_cos = float(np.mean(rand_cos))
                rand_max_cos = float(np.max(rand_cos))
                rand_mean_l2 = float(np.mean(rand_l2))
                rand_min_l2 = float(np.min(rand_l2))
            else:
                rand_mean_cos = rand_max_cos = rand_mean_l2 = rand_min_l2 = np.nan

            # nearest family among same strategy
            type_keys = [(fam, s) for (fam, s) in all_keys if s == strategy]
            type_cos = np.array([cosine_np(x, center_vectors[k]) for k in type_keys])
            nn_idx = int(np.argmax(type_cos))
            nn_family = type_keys[nn_idx][0]

            rows.append({
                "row_id": int(row["row_id"]),
                "block": block,
                "strategy": strategy,
                "true_family": true_fam,
                "original_prompt": row["prompt"],
                "same_family_cos": same_cos,
                "same_family_l2": same_l2,
                "random_family_mean_cos": rand_mean_cos,
                "random_family_max_cos": rand_max_cos,
                "random_family_mean_l2": rand_mean_l2,
                "random_family_min_l2": rand_min_l2,
                "cos_margin_vs_random_mean": same_cos - rand_mean_cos if np.isfinite(rand_mean_cos) else np.nan,
                "cos_margin_vs_random_max": same_cos - rand_max_cos if np.isfinite(rand_max_cos) else np.nan,
                "l2_gain_vs_random_mean": rand_mean_l2 - same_l2 if np.isfinite(rand_mean_l2) else np.nan,
                "l2_gain_vs_random_min": rand_min_l2 - same_l2 if np.isfinite(rand_min_l2) else np.nan,
                "nearest_weighted_family": nn_family,
                "nearest_weighted_correct_family": int(nn_family == true_fam),
            })

    return pd.DataFrame(rows)


def summarize_strategy(pair_df: pd.DataFrame):
    rows = []
    for (block, strategy), g in pair_df.groupby(["block", "strategy"]):
        rows.append({
            "block": block,
            "strategy": strategy,
            "n": int(len(g)),
            "mean_same_family_cos": float(g["same_family_cos"].mean()),
            "mean_random_family_mean_cos": float(g["random_family_mean_cos"].mean()),
            "mean_cos_margin_vs_random_mean": float(g["cos_margin_vs_random_mean"].mean()),
            "frac_cos_gt_random_mean": float((g["cos_margin_vs_random_mean"] > 0).mean()),
            "frac_cos_gt_random_max": float((g["cos_margin_vs_random_max"] > 0).mean()),
            "mean_same_family_l2": float(g["same_family_l2"].mean()),
            "mean_l2_gain_vs_random_mean": float(g["l2_gain_vs_random_mean"].mean()),
            "frac_l2_better_than_random_mean": float((g["l2_gain_vs_random_mean"] > 0).mean()),
            "nearest_weighted_correct_family_rate": float(g["nearest_weighted_correct_family"].mean()),
        })
    return pd.DataFrame(rows)


def family_summary(pair_df: pd.DataFrame):
    rows = []
    for (block, strategy, fam), g in pair_df.groupby(["block", "strategy", "true_family"]):
        rows.append({
            "block": block,
            "strategy": strategy,
            "seed_family": fam,
            "n": int(len(g)),
            "mean_cos_margin_vs_random_mean": float(g["cos_margin_vs_random_mean"].mean()),
            "frac_cos_gt_random_mean": float((g["cos_margin_vs_random_mean"] > 0).mean()),
            "mean_l2_gain_vs_random_mean": float(g["l2_gain_vs_random_mean"].mean()),
            "nearest_weighted_correct_family_rate": float(g["nearest_weighted_correct_family"].mean()),
        })
    return pd.DataFrame(rows)


def stats_tests(pair_df: pd.DataFrame):
    rows = []
    for (block, strategy), g in pair_df.groupby(["block", "strategy"]):
        cos_same = g["same_family_cos"].values
        cos_rand = g["random_family_mean_cos"].values
        l2_same = g["same_family_l2"].values
        l2_rand = g["random_family_mean_l2"].values

        try:
            tt = ttest_rel(cos_same, cos_rand, nan_policy="omit")
            t_stat, t_p = float(tt.statistic), float(tt.pvalue)
        except Exception:
            t_stat, t_p = np.nan, np.nan
        try:
            ww = wilcoxon(cos_same - cos_rand)
            w_stat, w_p = float(ww.statistic), float(ww.pvalue)
        except Exception:
            w_stat, w_p = np.nan, np.nan
        try:
            tt2 = ttest_rel(l2_rand, l2_same, nan_policy="omit")
            t2_stat, t2_p = float(tt2.statistic), float(tt2.pvalue)
        except Exception:
            t2_stat, t2_p = np.nan, np.nan

        rows.append({
            "block": block,
            "strategy": strategy,
            "n": int(len(g)),
            "cos_margin_mean": float(np.nanmean(cos_same - cos_rand)),
            "cos_paired_t_stat": t_stat,
            "cos_paired_t_p": t_p,
            "cos_wilcoxon_stat": w_stat,
            "cos_wilcoxon_p": w_p,
            "l2_gain_mean": float(np.nanmean(l2_rand - l2_same)),
            "l2_paired_t_stat": t2_stat,
            "l2_paired_t_p": t2_p,
        })
    return pd.DataFrame(rows)


# ============================================================
# SUMMARY
# ============================================================

def build_results_summary(strategy_df: pd.DataFrame, weights_df: pd.DataFrame):
    summary = {
        "config": asdict(cfg),
        "main_block": cfg.MAIN_BLOCK,
        "verdict": "UNDETERMINED",
        "best_overall": {},
        "best_main_block": {},
        "diagnosis": "",
        "interpretation": [],
    }

    if len(strategy_df):
        df = strategy_df.copy()
        df["score"] = (
            df["mean_cos_margin_vs_random_mean"].fillna(-999)
            + df["frac_cos_gt_random_mean"].fillna(0)
            + df["nearest_weighted_correct_family_rate"].fillna(0)
            + 0.1 * df["mean_l2_gain_vs_random_mean"].fillna(0)
        )
        best = df.sort_values("score", ascending=False).iloc[0].to_dict()
        summary["best_overall"] = best

        main = df[df["block"] == cfg.MAIN_BLOCK]
        if len(main):
            main_best = main.sort_values("score", ascending=False).iloc[0].to_dict()
            summary["best_main_block"] = main_best
        else:
            main_best = best

        def get(block, strategy, field):
            sub = df[(df["block"] == block) & (df["strategy"] == strategy)]
            if len(sub):
                return float(sub.iloc[0].get(field, np.nan))
            return np.nan

        mb = cfg.MAIN_BLOCK
        weighted_best = max([
            get(mb, "weighted_simplex_l2", "mean_cos_margin_vs_random_mean"),
            get(mb, "weighted_simplex_cosine", "mean_cos_margin_vs_random_mean"),
            get(mb, "weighted_ridge_simplex", "mean_cos_margin_vs_random_mean"),
        ])
        uniform = get(mb, "uniform_full_basis", "mean_cos_margin_vs_random_mean")
        nearest = get(mb, "single_nearest_exemplar", "mean_cos_margin_vs_random_mean")
        canonical = get(mb, "single_canonical_minimal_seed", "mean_cos_margin_vs_random_mean")

        if np.isfinite(weighted_best) and weighted_best > max(uniform, nearest, canonical):
            summary["diagnosis"] = "WEIGHTED_SEEDBASIS_IMPROVES"
        elif np.isfinite(uniform) and uniform > max(weighted_best, nearest, canonical):
            summary["diagnosis"] = "UNIFORM_PROMPTSET_BEST"
        elif np.isfinite(nearest) and nearest > max(weighted_best, uniform, canonical):
            summary["diagnosis"] = "NEAREST_EXEMPLAR_BEST"
        elif np.isfinite(canonical) and canonical > max(weighted_best, uniform, nearest):
            summary["diagnosis"] = "CANONICAL_BEST"
        else:
            summary["diagnosis"] = "NO_CLEAR_WINNER"

        margin = main_best.get("mean_cos_margin_vs_random_mean", -999)
        frac = main_best.get("frac_cos_gt_random_mean", 0)
        nn = main_best.get("nearest_weighted_correct_family_rate", 0)
        l2_gain = main_best.get("mean_l2_gain_vs_random_mean", -999)

        if margin > 0 and frac >= 0.85 and nn >= 0.70 and l2_gain > 0:
            summary["verdict"] = "PASS_STRONG_WEIGHTED_SEEDBASIS"
        elif margin > 0 and frac >= 0.70 and nn >= 0.50:
            summary["verdict"] = "PASS_LITE_WEIGHTED_SEEDBASIS"
        elif margin > 0 and frac >= 0.60:
            summary["verdict"] = "PARTIAL_PASS_WEIGHTED_SEEDBASIS_SIGNAL"
        else:
            summary["verdict"] = "NO_PASS_YET"

    # Weight diagnostics.
    if len(weights_df):
        wmain = weights_df[weights_df["block"] == cfg.MAIN_BLOCK]
        if len(wmain):
            summary["weight_summary_main_block"] = {
                "mean_target_cos_by_strategy": wmain.groupby("strategy")["target_cos"].mean().to_dict(),
                "mean_target_l2_by_strategy": wmain.groupby("strategy")["target_l2"].mean().to_dict(),
            }

    summary["interpretation"] = [
        "Weighted SeedBasis fits non-negative simplex weights to approximate each family center μ_F.",
        "A PASS means weighted centers regenerate original trajectories better than random-family weighted centers.",
        "If weighted basis improves over uniform and single prompts, SeedMemory should be modeled as weighted basis.",
        "If weighted basis only fits μ_F but not originals, μ_F may overfit the average and not preserve member-level recoverability.",
    ]
    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(cfg.RANDOM_SEED)
    outdir = Path(cfg.OUTPUT_DIR)
    ensure_dir(str(outdir))
    json_dump(asdict(cfg), outdir / "sem3b3_config.json")

    original_features, basis_features = load_inputs(cfg)

    all_weights = []
    all_centers = []
    all_pair = []

    for block in cfg.BLOCKS:
        print(f"[SEM-3B.3] Processing block={block}")
        weights_df, centers_df, center_vectors, Xo, cols = fit_all_weighted_centers(original_features, basis_features, block)
        if Xo is None:
            continue
        pair_df = compare_weighted_centers(original_features, block, center_vectors, Xo)

        if len(weights_df):
            all_weights.append(weights_df)
        if len(centers_df):
            all_centers.append(centers_df)
        if len(pair_df):
            all_pair.append(pair_df)

    weights = pd.concat(all_weights, ignore_index=True) if all_weights else pd.DataFrame()
    centers = pd.concat(all_centers, ignore_index=True) if all_centers else pd.DataFrame()
    pair = pd.concat(all_pair, ignore_index=True) if all_pair else pd.DataFrame()

    weights.to_csv(outdir / "sem3b3_weighted_basis_weights.csv", index=False, encoding="utf-8-sig")
    centers.to_csv(outdir / "sem3b3_weighted_centers.csv", index=False, encoding="utf-8-sig")
    pair.to_csv(outdir / "sem3b3_pairwise_reconstruction.csv", index=False, encoding="utf-8-sig")

    strat = summarize_strategy(pair) if len(pair) else pd.DataFrame()
    strat.to_csv(outdir / "sem3b3_strategy_summary.csv", index=False, encoding="utf-8-sig")

    fam = family_summary(pair) if len(pair) else pd.DataFrame()
    fam.to_csv(outdir / "sem3b3_family_summary.csv", index=False, encoding="utf-8-sig")

    stats = stats_tests(pair) if len(pair) else pd.DataFrame()
    stats.to_csv(outdir / "sem3b3_statistical_tests.csv", index=False, encoding="utf-8-sig")

    summary = build_results_summary(strat, weights)
    json_dump(summary, outdir / "sem3b3_results_summary.json")

    print("=" * 100)
    print("SEM-3B.3: Weighted SeedBasis Reconstruction")
    print("=" * 100)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nOutput files:")
    for p in [
        "sem3b3_config.json",
        "sem3b3_weighted_basis_weights.csv",
        "sem3b3_weighted_centers.csv",
        "sem3b3_pairwise_reconstruction.csv",
        "sem3b3_strategy_summary.csv",
        "sem3b3_family_summary.csv",
        "sem3b3_statistical_tests.csv",
        "sem3b3_results_summary.json",
    ]:
        print("  -", outdir / p)


if __name__ == "__main__":
    main()
