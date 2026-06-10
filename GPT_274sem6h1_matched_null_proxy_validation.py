# -*- coding: utf-8 -*-
"""
SEM-6H.1: Matched-Null Proxy Write Validation

Purpose
-------
Validate whether real Q_theta candidate selection in SEM-6E/6F is genuinely better
than matched nulls before running expensive live write replay.

This script is intentionally data-schema tolerant. It expects the key input:

    sem6e_outputs/sem6e_candidate_utilities.csv

The file should contain one row per (query/base sample, candidate experience), with:
  - an observed/oracle utility column
  - candidate pool/type metadata
  - candidate feature columns or existing proxy score columns

Core comparisons
----------------
init_only
real_Q_selected
shuffle_Q_selected
permuted_utility_Q_selected
random_pool_selected
same_family_selected
commit_topk_rule
oracle_selected

Outputs
-------
sem6h_outputs/
  sem6h1_selection_rows.csv
  sem6h1_summary.csv
  sem6h1_null_distributions.csv
  sem6h1_verdict.json

Run
---
python GPT_sem6h1_matched_null_proxy_validation.py

Optional:
python GPT_sem6h1_matched_null_proxy_validation.py --input sem6e_outputs/sem6e_candidate_utilities.csv
"""

import argparse
import json
import math
import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 20260606
N_NULL = 100
N_SPLITS = 5

DEFAULT_INPUT = r"sem6e_outputs\sem6e_candidate_utilities.csv"
DEFAULT_OUT_DIR = r"sem6h_outputs"


# -----------------------------
# Column detection
# -----------------------------

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def find_first_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    norm_map = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in norm_map:
            return norm_map[key]

    # fuzzy contains
    for cand in candidates:
        key = _norm(cand)
        for nk, orig in norm_map.items():
            if key and (key in nk or nk in key):
                return orig

    if required:
        raise ValueError(
            f"Could not find required column among candidates={candidates}. "
            f"Available columns={list(df.columns)}"
        )
    return None


def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols: Dict[str, Optional[str]] = {}

    cols["group"] = find_first_col(
        df,
        [
            "query_id", "base_id", "target_id", "test_id", "row_id", "prompt_id",
            "family_query_id", "sample_id", "anchor_id", "source_id"
        ],
        required=False,
    )

    cols["family"] = find_first_col(
        df,
        ["family_id", "seed_family", "memory_family", "target_family", "true_family"],
        required=False,
    )

    cols["candidate_family"] = find_first_col(
        df,
        ["candidate_family", "cand_family", "candidate_seed_family", "memory_candidate_family"],
        required=False,
    )

    cols["candidate_type"] = find_first_col(
        df,
        [
            "candidate_type", "cand_type", "pool_type", "source_pool",
            "candidate_source", "update_type", "experience_type"
        ],
        required=False,
    )

    cols["utility"] = find_first_col(
        df,
        [
            "utility", "delta_utility", "oracle_utility", "true_utility", "live_utility",
            "rank_improvement", "score_improvement", "delta_rank", "target_utility",
            "observed_utility", "y", "label"
        ],
        required=True,
    )

    cols["init"] = find_first_col(
        df,
        [
            "init_utility", "init_score", "baseline_utility", "base_utility",
            "init_rank_improvement", "no_write_utility"
        ],
        required=False,
    )

    cols["existing_q"] = find_first_col(
        df,
        [
            "q_pred", "proxy_pred", "proxy_score", "q_score", "pred_utility",
            "predicted_utility", "qtheta_score", "Q_theta", "proxy_utility"
        ],
        required=False,
    )

    cols["candidate_id"] = find_first_col(
        df,
        ["candidate_id", "cand_id", "experience_id", "memory_id", "candidate_row_id"],
        required=False,
    )

    if cols["group"] is None:
        # Create a fallback group if exactly one candidate per apparent ordered block is impossible.
        # This will not be meaningful for selection, so we fail loudly.
        raise ValueError(
            "Could not detect a grouping column such as query_id/base_id/row_id. "
            "SEM-6H needs multiple candidate rows per query/base sample."
        )

    return cols


def choose_feature_columns(df: pd.DataFrame, detected: Dict[str, Optional[str]]) -> List[str]:
    exclude = {c for c in detected.values() if c is not None}
    exclude_norm = {_norm(c) for c in exclude}

    bad_patterns = [
        "utility", "label", "target", "oracle", "live", "rank_improvement",
        "gen_", "answer", "selected", "verdict"
    ]

    numeric_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        nc = _norm(c)
        if any(p in nc for p in bad_patterns):
            # Avoid obvious leakage labels.
            continue
        if df[c].isna().all():
            continue
        if df[c].nunique(dropna=True) <= 1:
            continue
        numeric_cols.append(c)

    if not numeric_cols:
        raise ValueError(
            "No numeric feature columns detected after leakage filtering. "
            "If sem6e file already has Q scores, keep a q_pred/proxy_score column. "
            f"Columns={list(df.columns)}"
        )
    return numeric_cols


# -----------------------------
# Data helpers
# -----------------------------

def add_candidate_type_flags(df: pd.DataFrame, type_col: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    if type_col is None:
        out["_candidate_type_str"] = ""
    else:
        out["_candidate_type_str"] = out[type_col].astype(str).str.lower()

    s = out["_candidate_type_str"]
    out["_is_same_family"] = s.str.contains("same_family|same-family|same family|family_update", regex=True)
    out["_is_commit_topk"] = s.str.contains("commit|topk|top_k|neighbor", regex=True)
    out["_is_random"] = s.str.contains("random|rand", regex=True)
    return out


def group_iter(df: pd.DataFrame, group_col: str):
    for gid, sub in df.groupby(group_col, sort=False):
        yield gid, sub


def safe_mean(x) -> float:
    arr = pd.Series(x).dropna().astype(float)
    if len(arr) == 0:
        return float("nan")
    return float(arr.mean())


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = RANDOM_SEED) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(float(np.mean(sample)))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# -----------------------------
# Q prediction
# -----------------------------

def build_model(kind: str = "extratrees"):
    if kind == "ridge":
        return make_pipeline(StandardScaler(with_mean=True), Ridge(alpha=10.0, random_state=RANDOM_SEED))
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    if kind == "gbr":
        return GradientBoostingRegressor(random_state=RANDOM_SEED)
    return ExtraTreesRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def crossfit_predict(
    df: pd.DataFrame,
    feature_cols: List[str],
    utility_col: str,
    group_col: str,
    model_kind: str,
    permute_y: bool = False,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, Dict[str, float]]:
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0).to_numpy(dtype=float)
    y = df[utility_col].to_numpy(dtype=float)
    groups = df[group_col].astype(str).to_numpy()

    unique_groups = np.unique(groups)
    n_splits = min(N_SPLITS, len(unique_groups))
    if n_splits < 2:
        raise ValueError(f"Need at least 2 groups for crossfit; found {len(unique_groups)}")

    splitter = GroupKFold(n_splits=n_splits)
    preds = np.full(len(df), np.nan, dtype=float)

    rng = np.random.default_rng(seed)

    fold_metrics = []
    for fold, (tr, te) in enumerate(splitter.split(X, y, groups=groups)):
        y_train = y[tr].copy()
        if permute_y:
            y_train = rng.permutation(y_train)

        model = build_model(model_kind)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X[tr], y_train)
            p = model.predict(X[te])

        preds[te] = p

        if len(np.unique(y[te])) > 1:
            try:
                r2 = r2_score(y[te], p)
            except Exception:
                r2 = np.nan
            corr = np.corrcoef(y[te], p)[0, 1] if np.std(p) > 1e-12 and np.std(y[te]) > 1e-12 else np.nan
        else:
            r2, corr = np.nan, np.nan
        rmse = math.sqrt(mean_squared_error(y[te], p))
        fold_metrics.append({"fold": fold, "r2": r2, "corr": corr, "rmse": rmse})

    metrics = {
        "cv_r2_mean": safe_mean([m["r2"] for m in fold_metrics]),
        "cv_corr_mean": safe_mean([m["corr"] for m in fold_metrics]),
        "cv_rmse_mean": safe_mean([m["rmse"] for m in fold_metrics]),
    }
    return preds, metrics


# -----------------------------
# Selection rules
# -----------------------------

def pick_idx_by_score(sub: pd.DataFrame, score_col: str, maximize: bool = True) -> int:
    vals = sub[score_col].to_numpy(dtype=float)
    if maximize:
        pos = int(np.nanargmax(vals))
    else:
        pos = int(np.nanargmin(vals))
    return int(sub.index[pos])


def pick_idx_random(sub: pd.DataFrame, rng: np.random.Generator) -> int:
    return int(rng.choice(sub.index.to_numpy()))


def pick_idx_mask_then_score(sub: pd.DataFrame, mask_col: str, score_col: str, rng: np.random.Generator) -> int:
    masked = sub[sub[mask_col].fillna(False)]
    if len(masked) == 0:
        return pick_idx_random(sub, rng)
    return pick_idx_by_score(masked, score_col, maximize=True)


def evaluate_selection(
    df: pd.DataFrame,
    group_col: str,
    utility_col: str,
    selection_name: str,
    picker,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    utilities = []
    for gid, sub in group_iter(df, group_col):
        try:
            idx = picker(sub)
        except Exception:
            continue
        row = df.loc[idx].copy()
        util = float(row[utility_col])
        utilities.append(util)
        out = {
            "selection": selection_name,
            "group_id": gid,
            "selected_index": idx,
            "selected_utility": util,
        }
        for c in ["_candidate_type_str", "_is_same_family", "_is_commit_topk", "_is_random"]:
            if c in row.index:
                out[c] = row[c]
        rows.append(out)

    arr = np.asarray(utilities, dtype=float)
    ci_lo, ci_hi = bootstrap_ci(arr) if len(arr) else (np.nan, np.nan)
    summary = {
        "selection": selection_name,
        "n_groups": int(len(arr)),
        "mean_utility": float(np.mean(arr)) if len(arr) else np.nan,
        "median_utility": float(np.median(arr)) if len(arr) else np.nan,
        "std_utility": float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "positive_rate": float(np.mean(arr > 0)) if len(arr) else np.nan,
    }
    return pd.DataFrame(rows), summary


def paired_difference(selection_rows: pd.DataFrame, a: str, b: str) -> Dict[str, float]:
    pa = selection_rows[selection_rows["selection"] == a][["group_id", "selected_utility"]].rename(
        columns={"selected_utility": "a"}
    )
    pb = selection_rows[selection_rows["selection"] == b][["group_id", "selected_utility"]].rename(
        columns={"selected_utility": "b"}
    )
    m = pa.merge(pb, on="group_id", how="inner")
    diff = (m["a"] - m["b"]).to_numpy(dtype=float)
    if len(diff) == 0:
        return {"comparison": f"{a}>{b}", "n": 0, "mean_diff": np.nan, "ci95_lo": np.nan, "ci95_hi": np.nan, "z": np.nan}
    lo, hi = bootstrap_ci(diff)
    z = float(np.mean(diff) / (np.std(diff, ddof=1) / math.sqrt(len(diff)))) if len(diff) > 1 and np.std(diff, ddof=1) > 1e-12 else np.nan
    return {
        "comparison": f"{a}>{b}",
        "n": int(len(diff)),
        "mean_diff": float(np.mean(diff)),
        "ci95_lo": lo,
        "ci95_hi": hi,
        "z": z,
        "win_rate": float(np.mean(diff > 0)),
    }


def run_null_distribution(
    df: pd.DataFrame,
    group_col: str,
    utility_col: str,
    real_score_col: str,
    feature_cols: List[str],
    model_kind: str,
    n_null: int,
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    rng = np.random.default_rng(RANDOM_SEED + 991)
    null_summaries = []
    null_rows_all = []

    for i in range(n_null):
        tmp = df.copy()

        # 1) shuffle_Q: shuffle real predictions globally, preserving marginal score distribution.
        tmp["_shuffle_q_score"] = rng.permutation(tmp[real_score_col].to_numpy(dtype=float))

        rows, summ = evaluate_selection(
            tmp,
            group_col,
            utility_col,
            f"shuffle_Q_selected_{i:03d}",
            lambda sub, col="_shuffle_q_score": pick_idx_by_score(sub, col, maximize=True),
        )
        summ["null_type"] = "shuffle_Q_selected"
        summ["iter"] = i
        null_summaries.append(summ)
        null_rows_all.append(rows)

        # 2) random pool.
        rows, summ = evaluate_selection(
            tmp,
            group_col,
            utility_col,
            f"random_pool_selected_{i:03d}",
            lambda sub, r=rng: pick_idx_random(sub, r),
        )
        summ["null_type"] = "random_pool_selected"
        summ["iter"] = i
        null_summaries.append(summ)
        null_rows_all.append(rows)

        # 3) permuted utility Q: retrain with permuted labels.
        try:
            perm_pred, _ = crossfit_predict(
                tmp, feature_cols, utility_col, group_col,
                model_kind=model_kind, permute_y=True, seed=RANDOM_SEED + 10000 + i
            )
            tmp["_perm_q_score"] = perm_pred
            rows, summ = evaluate_selection(
                tmp,
                group_col,
                utility_col,
                f"permuted_utility_Q_selected_{i:03d}",
                lambda sub, col="_perm_q_score": pick_idx_by_score(sub, col, maximize=True),
            )
            summ["null_type"] = "permuted_utility_Q_selected"
            summ["iter"] = i
            null_summaries.append(summ)
            null_rows_all.append(rows)
        except Exception as e:
            null_summaries.append({
                "selection": f"permuted_utility_Q_selected_{i:03d}",
                "null_type": "permuted_utility_Q_selected",
                "iter": i,
                "error": str(e),
                "n_groups": np.nan,
                "mean_utility": np.nan,
            })

    return pd.DataFrame(null_summaries), null_rows_all


def summarize_null_vs_real(null_df: pd.DataFrame, real_mean: float) -> pd.DataFrame:
    rows = []
    for nt, sub in null_df.groupby("null_type"):
        vals = pd.to_numeric(sub["mean_utility"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            rows.append({"null_type": nt, "null_mean": np.nan, "null_std": np.nan, "z_vs_real": np.nan, "q95": np.nan, "real_gt_q95": False})
            continue
        mu = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        z = float((real_mean - mu) / sd) if sd and sd > 1e-12 else np.nan
        q95 = float(np.percentile(vals, 95))
        rows.append({
            "null_type": nt,
            "null_mean": mu,
            "null_std": sd,
            "z_vs_real": z,
            "q95": q95,
            "real_gt_q95": bool(real_mean > q95),
            "n_null": int(len(vals)),
        })
    return pd.DataFrame(rows)


def make_verdict(summary_df: pd.DataFrame, pair_df: pd.DataFrame, null_vs_real: pd.DataFrame) -> Dict[str, object]:
    means = dict(zip(summary_df["selection"], summary_df["mean_utility"]))

    real = means.get("real_Q_selected", np.nan)
    init = means.get("init_only", np.nan)
    same = means.get("same_family_selected", np.nan)
    rand = means.get("random_pool_selected_mean", np.nan)
    commit = means.get("commit_topk_rule", np.nan)
    oracle = means.get("oracle_selected", np.nan)

    # Null requirements.
    z_req = {}
    for nt in ["shuffle_Q_selected", "permuted_utility_Q_selected", "random_pool_selected"]:
        row = null_vs_real[null_vs_real["null_type"] == nt]
        if len(row):
            z = float(row["z_vs_real"].iloc[0])
            gt_q95 = bool(row["real_gt_q95"].iloc[0])
        else:
            z, gt_q95 = np.nan, False
        z_req[nt] = {"z": z, "real_gt_q95": gt_q95}

    pass_lite = bool(
        np.isfinite(real)
        and (not np.isfinite(init) or real > init)
        and (not np.isfinite(same) or real > same)
    )

    # For random pool, use null mean if summary mean absent.
    rand_null = null_vs_real[null_vs_real["null_type"] == "random_pool_selected"]
    rand_mu = float(rand_null["null_mean"].iloc[0]) if len(rand_null) else rand
    if np.isfinite(rand_mu):
        pass_lite = pass_lite and real > rand_mu

    pass_strong = bool(
        np.isfinite(real)
        and all(v["z"] > 2.0 and v["real_gt_q95"] for v in z_req.values() if np.isfinite(v["z"]))
        and (not np.isfinite(oracle) or oracle >= real)
    )

    pass_milestone = bool(
        pass_strong
        and np.isfinite(commit)
        and real >= commit
    )

    if pass_milestone:
        verdict = "PASS_MILESTONE_REAL_Q_GE_COMMIT_TOPK_AND_MATCHED_NULL"
    elif pass_strong:
        verdict = "PASS_STRONG_MATCHED_NULL_VALIDATED"
    elif pass_lite:
        verdict = "PASS_LITE_REAL_Q_BEATS_BASIC_BASELINES"
    else:
        verdict = "FAIL_OR_INCONCLUSIVE"

    return {
        "verdict": verdict,
        "pass_lite": pass_lite,
        "pass_strong": pass_strong,
        "pass_milestone": pass_milestone,
        "means": {k: (None if pd.isna(v) else float(v)) for k, v in means.items()},
        "null_tests": z_req,
        "notes": [
            "SEM-6H.1 is a statistical matched-null validation using SEM-6E candidate utility table.",
            "It does not perform live model write/replay. If PASS-Strong, run SEM-6H.2 live replay.",
            "If real_Q fails only against commit_topk, Q_theta is useful but not better than the current identity/commit heuristic.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--model_kind", default="extratrees", choices=["extratrees", "rf", "gbr", "ridge"])
    parser.add_argument("--n_null", type=int, default=N_NULL)
    parser.add_argument("--use_existing_q_if_present", action="store_true")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {in_path}\n"
            f"Expected default: {DEFAULT_INPUT}"
        )

    df = pd.read_csv(in_path)
    print(f"[SEM-6H.1] Loaded {in_path} shape={df.shape}")

    detected = detect_columns(df)
    print("[SEM-6H.1] Detected columns:")
    for k, v in detected.items():
        print(f"  {k}: {v}")

    group_col = detected["group"]
    utility_col = detected["utility"]

    df = add_candidate_type_flags(df, detected["candidate_type"])

    # Ensure utility is numeric.
    df[utility_col] = pd.to_numeric(df[utility_col], errors="coerce")
    df = df.dropna(subset=[utility_col, group_col]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows after dropping missing group/utility.")

    feature_cols = choose_feature_columns(df, detected)
    print(f"[SEM-6H.1] Using {len(feature_cols)} feature columns.")
    with open(out_dir / "sem6h1_feature_columns.txt", "w", encoding="utf-8") as f:
        for c in feature_cols:
            f.write(c + "\n")

    # Build real Q score.
    if args.use_existing_q_if_present and detected["existing_q"] is not None:
        q_col = "_real_q_score"
        df[q_col] = pd.to_numeric(df[detected["existing_q"]], errors="coerce")
        if df[q_col].isna().all():
            raise ValueError(f"Existing Q column {detected['existing_q']} is non-numeric/all NaN.")
        df[q_col] = df[q_col].fillna(df[q_col].median())
        q_metrics = {"source": f"existing:{detected['existing_q']}"}
    else:
        pred, q_metrics = crossfit_predict(
            df, feature_cols, utility_col, group_col,
            model_kind=args.model_kind,
            permute_y=False,
            seed=RANDOM_SEED,
        )
        q_col = "_real_q_score"
        df[q_col] = pred
        q_metrics["source"] = f"crossfit:{args.model_kind}"

    print("[SEM-6H.1] Q metrics:", q_metrics)

    rng = np.random.default_rng(RANDOM_SEED)

    # Core selections.
    selection_frames = []
    summaries = []

    # init_only: if init utility exists, summarize by group from init column;
    # otherwise use zero baseline.
    if detected["init"] is not None:
        init_col = detected["init"]
        tmp_rows = []
        vals = []
        for gid, sub in group_iter(df, group_col):
            val = float(pd.to_numeric(sub[init_col], errors="coerce").dropna().iloc[0]) if pd.to_numeric(sub[init_col], errors="coerce").dropna().size else 0.0
            vals.append(val)
            tmp_rows.append({"selection": "init_only", "group_id": gid, "selected_index": -1, "selected_utility": val})
        arr = np.asarray(vals, dtype=float)
        lo, hi = bootstrap_ci(arr)
        selection_frames.append(pd.DataFrame(tmp_rows))
        summaries.append({
            "selection": "init_only",
            "n_groups": len(arr),
            "mean_utility": float(np.mean(arr)),
            "median_utility": float(np.median(arr)),
            "std_utility": float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "positive_rate": float(np.mean(arr > 0)),
        })
    else:
        # SEM-6E utility is often delta relative to init; baseline is 0.
        gids = list(df[group_col].drop_duplicates())
        rows = [{"selection": "init_only", "group_id": gid, "selected_index": -1, "selected_utility": 0.0} for gid in gids]
        selection_frames.append(pd.DataFrame(rows))
        summaries.append({
            "selection": "init_only",
            "n_groups": len(gids),
            "mean_utility": 0.0,
            "median_utility": 0.0,
            "std_utility": 0.0,
            "ci95_lo": 0.0,
            "ci95_hi": 0.0,
            "positive_rate": 0.0,
        })

    for name, picker in [
        ("real_Q_selected", lambda sub: pick_idx_by_score(sub, q_col, maximize=True)),
        ("oracle_selected", lambda sub: pick_idx_by_score(sub, utility_col, maximize=True)),
        ("same_family_selected", lambda sub: pick_idx_mask_then_score(sub, "_is_same_family", q_col, rng)),
        ("commit_topk_rule", lambda sub: pick_idx_mask_then_score(sub, "_is_commit_topk", q_col, rng)),
    ]:
        rows, summ = evaluate_selection(df, group_col, utility_col, name, picker)
        selection_frames.append(rows)
        summaries.append(summ)

    # Mean random pool summary over repeated random choices.
    random_means = []
    random_frames = []
    for i in range(args.n_null):
        rows, summ = evaluate_selection(
            df, group_col, utility_col,
            f"random_pool_selected_repl_{i:03d}",
            lambda sub, r=rng: pick_idx_random(sub, r),
        )
        random_means.append(summ["mean_utility"])
        random_frames.append(rows)
    random_means_arr = np.asarray(random_means, dtype=float)
    summaries.append({
        "selection": "random_pool_selected_mean",
        "n_groups": int(df[group_col].nunique()),
        "mean_utility": float(np.mean(random_means_arr)),
        "median_utility": float(np.median(random_means_arr)),
        "std_utility": float(np.std(random_means_arr, ddof=1)) if len(random_means_arr) > 1 else np.nan,
        "ci95_lo": float(np.percentile(random_means_arr, 2.5)),
        "ci95_hi": float(np.percentile(random_means_arr, 97.5)),
        "positive_rate": np.nan,
    })

    selection_rows = pd.concat(selection_frames, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    # Null distributions.
    null_df, null_rows_all = run_null_distribution(
        df, group_col, utility_col, q_col,
        feature_cols=feature_cols,
        model_kind=args.model_kind,
        n_null=args.n_null,
    )
    null_vs_real = summarize_null_vs_real(
        null_df,
        float(summary_df.loc[summary_df["selection"] == "real_Q_selected", "mean_utility"].iloc[0]),
    )

    # Pairwise differences for named baselines.
    pair_names = ["init_only", "same_family_selected", "commit_topk_rule", "oracle_selected"]
    pair_rows = []
    for b in pair_names:
        pair_rows.append(paired_difference(selection_rows, "real_Q_selected", b))
    pair_df = pd.DataFrame(pair_rows)

    verdict = make_verdict(summary_df, pair_df, null_vs_real)
    verdict["q_metrics"] = q_metrics
    verdict["detected_columns"] = detected
    verdict["n_rows"] = int(len(df))
    verdict["n_groups"] = int(df[group_col].nunique())
    verdict["n_candidates_per_group_mean"] = float(df.groupby(group_col).size().mean())
    verdict["feature_count"] = int(len(feature_cols))

    # Save.
    df.to_csv(out_dir / "sem6h1_scored_candidates.csv", index=False, encoding="utf-8-sig")
    selection_rows.to_csv(out_dir / "sem6h1_selection_rows.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "sem6h1_summary.csv", index=False, encoding="utf-8-sig")
    null_df.to_csv(out_dir / "sem6h1_null_distributions.csv", index=False, encoding="utf-8-sig")
    null_vs_real.to_csv(out_dir / "sem6h1_null_vs_real.csv", index=False, encoding="utf-8-sig")
    pair_df.to_csv(out_dir / "sem6h1_pairwise_realQ_differences.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "sem6h1_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)

    print("\n========== SEM-6H.1 SUMMARY ==========")
    print(summary_df.to_string(index=False))
    print("\n========== NULL VS REAL_Q ==========")
    print(null_vs_real.to_string(index=False))
    print("\n========== REAL_Q PAIRWISE ==========")
    print(pair_df.to_string(index=False))
    print("\n========== VERDICT ==========")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    print(f"\n[SEM-6H.1] Outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
