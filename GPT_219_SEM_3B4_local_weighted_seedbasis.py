# -*- coding: utf-8 -*-
"""
GPT_208_SEM_3B4_local_weighted_seedbasis.py

SEM-3B.4: Local Weighted SeedBasis
----------------------------------

Goal:
    Test whether each original prompt requires its own local weights over a
    family SeedBasis.

Previous result:
    SEM-3B.3 showed family-level WeightedSeedBasis improves center_pca and scalar
    signal, but nearest-family rate remains weak.

Hypothesis:
    MemoryUnit_F = Basis_F + WeightGenerator(q)

Instead of one family-level weight vector:
    μ_F ≈ Σ_j w_F,j T(p_j)

Use local query-conditioned weights:
    T(prompt_i) ≈ Σ_j w_i,j T(p_j)

This script operates entirely on existing features:
    - SEM-3A original prompt features
    - SEM-3B.2 seedbasis features

No model forward is required.

Important caveat:
    This is an upper-bound / oracle reconstruction audit because weights are fit
    directly to each original prompt feature vector. It answers:
        "Can the family basis span local prompt variants?"
    not:
        "Can the system predict the weights without seeing target?"
    If positive, next step is SEM-3B.5 WeightGenerator(q).

Inputs:
    sem3a_outputs/sem3a_features.csv
    sem3b2_outputs/sem3b2_seedbasis_dataset.csv
    sem3b2_outputs/sem3b2_seedbasis_features.csv

Outputs:
    sem3b4_outputs/
      sem3b4_config.json
      sem3b4_local_weights.csv
      sem3b4_pairwise_reconstruction.csv
      sem3b4_strategy_summary.csv
      sem3b4_family_summary.csv
      sem3b4_statistical_tests.csv
      sem3b4_results_summary.json

Run:
    python GPT_208_SEM_3B4_local_weighted_seedbasis.py
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
    OUTPUT_DIR: str = "./sem3b4_outputs"

    SEM3A_FEATURES: str = "sem3a_features.csv"
    SEM3B2_BASIS_DATASET: str = "sem3b2_seedbasis_dataset.csv"
    SEM3B2_BASIS_FEATURES: str = "sem3b2_seedbasis_features.csv"

    MAIN_BLOCK: str = "center_pca"
    BLOCKS: Tuple[str, ...] = ("center_pca", "decision", "mid", "all", "scalar", "transport", "init")

    BASIS_TYPES: Tuple[str, ...] = (
        "nearest_exemplar",
        "nearest_plus_structure",
        "natural_prompt_star",
        "structured_prompt_star",
        "compressed_prompt_star",
        "canonical_minimal_seed",
    )

    LOCAL_STRATEGIES: Tuple[str, ...] = (
        "local_simplex_l2",
        "local_simplex_cosine",
        "local_ridge_simplex",
        "local_top2_simplex_l2",
        "local_top3_simplex_l2",
    )

    RIDGE_LAMBDA: float = 0.05
    RANDOM_SEED: int = 42

cfg = CFG()


# ============================================================
# UTILITIES
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


def simplex_project(w):
    w = np.maximum(w, 0.0)
    s = w.sum()
    if s <= 1e-12:
        return np.ones_like(w) / len(w)
    return w / s


# ============================================================
# LOAD DATA
# ============================================================

def load_inputs(cfg: CFG):
    sem3a_dir = Path(cfg.SEM3A_DIR)
    sem3b2_dir = Path(cfg.SEM3B2_DIR)

    original_features = pd.read_csv(sem3a_dir / cfg.SEM3A_FEATURES)
    basis_dataset = pd.read_csv(sem3b2_dir / cfg.SEM3B2_BASIS_DATASET)
    basis_features = pd.read_csv(sem3b2_dir / cfg.SEM3B2_BASIS_FEATURES)

    if "basis_prompt_type" not in basis_features.columns:
        basis_features = basis_dataset.merge(basis_features, on="row_id", how="left")

    original_features = original_features[original_features["row_type"] == "original_prompt"].copy()
    return original_features, basis_features


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
# LOCAL WEIGHT FITTING
# ============================================================

def get_basis_for_family(basis_features: pd.DataFrame, Xb: np.ndarray, fam: str):
    basis = basis_features.reset_index(drop=True)
    idxs = []
    types = []
    prompts = []
    for btype in cfg.BASIS_TYPES:
        hit = np.where((basis["seed_family"].values == fam) & (basis["basis_prompt_type"].values == btype))[0]
        if len(hit):
            idx = int(hit[0])
            idxs.append(idx)
            types.append(btype)
            prompts.append(str(basis.iloc[idx].get("prompt", "")))
    if not idxs:
        return None
    return {
        "idxs": idxs,
        "types": types,
        "prompts": prompts,
        "B": Xb[idxs, :],
    }


def fit_local_weights(B: np.ndarray, target: np.ndarray, strategy: str, ridge_lambda: float):
    n = B.shape[0]
    if n == 1:
        return np.array([1.0])

    # Optional candidate restriction by nearest basis prompts.
    allowed = np.arange(n)
    if strategy.startswith("local_top"):
        if "top2" in strategy:
            kk = min(2, n)
        elif "top3" in strategy:
            kk = min(3, n)
        else:
            kk = n
        sims = np.array([cosine_np(target, B[i]) for i in range(n)])
        allowed = np.argsort(-sims)[:kk]
        Bfit = B[allowed, :]
        base_strategy = "local_simplex_l2"
    else:
        Bfit = B
        base_strategy = strategy

    m = Bfit.shape[0]
    init = np.ones(m) / m
    bounds = [(0.0, 1.0) for _ in range(m)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if base_strategy == "local_simplex_l2":
        def obj(w):
            pred = w @ Bfit
            return float(np.mean((pred - target) ** 2))
    elif base_strategy == "local_simplex_cosine":
        def obj(w):
            pred = w @ Bfit
            return float(1.0 - cosine_np(pred, target))
    elif base_strategy == "local_ridge_simplex":
        def obj(w):
            pred = w @ Bfit
            ridge = ridge_lambda * np.sum((w - init) ** 2)
            return float(np.mean((pred - target) ** 2) + ridge)
    else:
        raise ValueError(strategy)

    res = minimize(obj, init, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 500, "ftol": 1e-9})
    if not res.success:
        wsmall = init
    else:
        wsmall = simplex_project(res.x)

    w = np.zeros(n)
    w[allowed] = wsmall
    return simplex_project(w)


def build_local_reconstructions(original_features, basis_features, block: str):
    Xo, Xb, cols = align_spaces(original_features, basis_features, block)
    if Xo is None:
        return pd.DataFrame(), pd.DataFrame(), None, None

    orig = original_features.reset_index(drop=True)
    weight_rows = []
    recon_rows = []
    recon_vectors = {}

    for oi, row in orig.iterrows():
        fam = row["seed_family"]
        item = get_basis_for_family(basis_features, Xb, fam)
        if item is None:
            continue

        B = item["B"]
        types = item["types"]
        prompts = item["prompts"]
        target = Xo[oi]

        # Single and uniform baselines.
        strategies = {}
        strategies["uniform_full_basis"] = np.ones(len(types)) / len(types)
        for i, t in enumerate(types):
            w = np.zeros(len(types))
            w[i] = 1.0
            strategies[f"single_{t}"] = w

        for s in cfg.LOCAL_STRATEGIES:
            strategies[s] = fit_local_weights(B, target, s, cfg.RIDGE_LAMBDA)

        for strategy, w in strategies.items():
            pred = w @ B
            key = (int(row["row_id"]), strategy)
            recon_vectors[key] = pred

            weight_rows.append({
                "block": block,
                "row_id": int(row["row_id"]),
                "seed_family": fam,
                "strategy": strategy,
                "basis_types": "|".join(types),
                "weights_json": json.dumps({t: float(wi) for t, wi in zip(types, w)}, ensure_ascii=False),
                "selected_basis_types": "|".join([t for t, wi in zip(types, w) if wi > 1e-6]),
                "target_cos": float(cosine_np(pred, target)),
                "target_l2": float(np.linalg.norm(pred - target)),
                "n_basis": int(len(types)),
            })
            recon_rows.append({
                "block": block,
                "row_id": int(row["row_id"]),
                "seed_family": fam,
                "strategy": strategy,
                "recon_vector_json": json.dumps(pred.tolist()),
                "target_cos": float(cosine_np(pred, target)),
                "target_l2": float(np.linalg.norm(pred - target)),
            })

    return pd.DataFrame(weight_rows), pd.DataFrame(recon_rows), recon_vectors, Xo


# ============================================================
# RANDOM-FAMILY COMPARISON
# ============================================================

def build_random_family_recon_centers(original_features, basis_features, block: str, strategy: str, target: np.ndarray, true_family: str, Xb: np.ndarray):
    """
    For a given target and strategy, fit local weights using each random family's basis.
    This is a strong baseline: it gives random families the same oracle local fit.
    """
    basis = basis_features.reset_index(drop=True)
    fams = sorted([f for f in basis["seed_family"].unique() if f != true_family])
    out = []
    for fam in fams:
        item = get_basis_for_family(basis_features, Xb, fam)
        if item is None:
            continue
        B = item["B"]
        w = fit_local_weights(B, target, strategy, cfg.RIDGE_LAMBDA) if strategy in cfg.LOCAL_STRATEGIES else None
        if strategy == "uniform_full_basis":
            w = np.ones(B.shape[0]) / B.shape[0]
        elif strategy.startswith("single_"):
            btype = strategy.replace("single_", "")
            if btype in item["types"]:
                idx = item["types"].index(btype)
                w = np.zeros(B.shape[0])
                w[idx] = 1.0
            else:
                continue
        if w is None:
            continue
        out.append((fam, w @ B))
    return out


def compare_local_reconstructions(original_features, basis_features, block: str):
    Xo, Xb, cols = align_spaces(original_features, basis_features, block)
    if Xo is None:
        return pd.DataFrame(), pd.DataFrame()

    orig = original_features.reset_index(drop=True)
    weight_rows, recon_rows, recon_vectors, _ = build_local_reconstructions(original_features, basis_features, block)

    rows = []

    strategies = sorted(weight_rows["strategy"].unique()) if len(weight_rows) else []
    for oi, row in orig.iterrows():
        row_id = int(row["row_id"])
        true_family = row["seed_family"]
        target = Xo[oi]

        for strategy in strategies:
            same_key = (row_id, strategy)
            if same_key not in recon_vectors:
                continue

            same_vec = recon_vectors[same_key]
            same_cos = cosine_np(target, same_vec)
            same_l2 = float(np.linalg.norm(target - same_vec))

            randoms = build_random_family_recon_centers(
                original_features, basis_features, block, strategy, target, true_family, Xb
            )
            if randoms:
                rand_cos = np.array([cosine_np(target, vec) for _, vec in randoms])
                rand_l2 = np.array([np.linalg.norm(target - vec) for _, vec in randoms])
                rand_mean_cos = float(np.mean(rand_cos))
                rand_max_cos = float(np.max(rand_cos))
                rand_mean_l2 = float(np.mean(rand_l2))
                rand_min_l2 = float(np.min(rand_l2))
                nearest_idx = int(np.argmax(rand_cos))
                nearest_random_family = randoms[nearest_idx][0]
            else:
                rand_mean_cos = rand_max_cos = rand_mean_l2 = rand_min_l2 = np.nan
                nearest_random_family = ""

            # Nearest among same strategy including true and random families.
            if randoms:
                if same_cos >= rand_max_cos:
                    nearest_family = true_family
                    nearest_correct = 1
                else:
                    nearest_family = nearest_random_family
                    nearest_correct = 0
            else:
                nearest_family = true_family
                nearest_correct = 1

            rows.append({
                "block": block,
                "row_id": row_id,
                "strategy": strategy,
                "true_family": true_family,
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
                "nearest_family": nearest_family,
                "nearest_correct_family": nearest_correct,
                "n_features": len(cols),
            })

    return pd.DataFrame(rows), weight_rows


# ============================================================
# SUMMARIES
# ============================================================

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
            "nearest_correct_family_rate": float(g["nearest_correct_family"].mean()),
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
            "nearest_correct_family_rate": float(g["nearest_correct_family"].mean()),
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
            + df["nearest_correct_family_rate"].fillna(0)
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
        local_best = max([
            get(mb, "local_simplex_l2", "mean_cos_margin_vs_random_mean"),
            get(mb, "local_simplex_cosine", "mean_cos_margin_vs_random_mean"),
            get(mb, "local_ridge_simplex", "mean_cos_margin_vs_random_mean"),
            get(mb, "local_top2_simplex_l2", "mean_cos_margin_vs_random_mean"),
            get(mb, "local_top3_simplex_l2", "mean_cos_margin_vs_random_mean"),
        ])
        uniform = get(mb, "uniform_full_basis", "mean_cos_margin_vs_random_mean")
        nearest = get(mb, "single_nearest_exemplar", "mean_cos_margin_vs_random_mean")

        if np.isfinite(local_best) and local_best > max(uniform, nearest):
            summary["diagnosis"] = "LOCAL_WEIGHTED_SEEDBASIS_IMPROVES"
        elif np.isfinite(nearest) and nearest > max(local_best, uniform):
            summary["diagnosis"] = "NEAREST_EXEMPLAR_REMAINS_BEST"
        elif np.isfinite(uniform) and uniform > max(local_best, nearest):
            summary["diagnosis"] = "UNIFORM_BASIS_REMAINS_BEST"
        else:
            summary["diagnosis"] = "NO_CLEAR_WINNER"

        margin = main_best.get("mean_cos_margin_vs_random_mean", -999)
        frac = main_best.get("frac_cos_gt_random_mean", 0)
        nn = main_best.get("nearest_correct_family_rate", 0)
        l2_gain = main_best.get("mean_l2_gain_vs_random_mean", -999)

        if margin > 0 and frac >= 0.90 and nn >= 0.85 and l2_gain > 0:
            summary["verdict"] = "PASS_STRONG_LOCAL_WEIGHTED_SEEDBASIS"
        elif margin > 0 and frac >= 0.80 and nn >= 0.65:
            summary["verdict"] = "PASS_LITE_LOCAL_WEIGHTED_SEEDBASIS"
        elif margin > 0 and frac >= 0.65:
            summary["verdict"] = "PARTIAL_PASS_LOCAL_WEIGHTED_SIGNAL"
        else:
            summary["verdict"] = "NO_PASS_YET"

    if len(weights_df):
        mainw = weights_df[weights_df["block"] == cfg.MAIN_BLOCK]
        if len(mainw):
            summary["weight_summary_main_block"] = {
                "mean_target_cos_by_strategy": mainw.groupby("strategy")["target_cos"].mean().to_dict(),
                "mean_target_l2_by_strategy": mainw.groupby("strategy")["target_l2"].mean().to_dict(),
            }

    summary["interpretation"] = [
        "This is an oracle/local upper-bound audit: weights are fit to each original prompt feature vector.",
        "A strong PASS means family basis spans prompt-level local variants.",
        "If local weighted succeeds but family-level weighted was weak, next step is a WeightGenerator(q) to predict local weights.",
        "If local weighted still fails, current basis prompts do not span center_pca trajectory variants.",
    ]
    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(cfg.RANDOM_SEED)
    outdir = Path(cfg.OUTPUT_DIR)
    ensure_dir(str(outdir))
    json_dump(asdict(cfg), outdir / "sem3b4_config.json")

    original_features, basis_features = load_inputs(cfg)

    all_pairs = []
    all_weights = []

    for block in cfg.BLOCKS:
        print(f"[SEM-3B.4] Processing block={block}")
        pair, weights = compare_local_reconstructions(original_features, basis_features, block)
        if len(pair):
            all_pairs.append(pair)
        if len(weights):
            all_weights.append(weights)

    pair_df = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame()
    weights_df = pd.concat(all_weights, ignore_index=True) if all_weights else pd.DataFrame()

    pair_df.to_csv(outdir / "sem3b4_pairwise_reconstruction.csv", index=False, encoding="utf-8-sig")
    weights_df.to_csv(outdir / "sem3b4_local_weights.csv", index=False, encoding="utf-8-sig")

    strat = summarize_strategy(pair_df) if len(pair_df) else pd.DataFrame()
    strat.to_csv(outdir / "sem3b4_strategy_summary.csv", index=False, encoding="utf-8-sig")

    fam = family_summary(pair_df) if len(pair_df) else pd.DataFrame()
    fam.to_csv(outdir / "sem3b4_family_summary.csv", index=False, encoding="utf-8-sig")

    stats = stats_tests(pair_df) if len(pair_df) else pd.DataFrame()
    stats.to_csv(outdir / "sem3b4_statistical_tests.csv", index=False, encoding="utf-8-sig")

    summary = build_results_summary(strat, weights_df)
    json_dump(summary, outdir / "sem3b4_results_summary.json")

    print("=" * 100)
    print("SEM-3B.4: Local Weighted SeedBasis")
    print("=" * 100)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nOutput files:")
    for p in [
        "sem3b4_config.json",
        "sem3b4_local_weights.csv",
        "sem3b4_pairwise_reconstruction.csv",
        "sem3b4_strategy_summary.csv",
        "sem3b4_family_summary.csv",
        "sem3b4_statistical_tests.csv",
        "sem3b4_results_summary.json",
    ]:
        print("  -", outdir / p)


if __name__ == "__main__":
    main()
