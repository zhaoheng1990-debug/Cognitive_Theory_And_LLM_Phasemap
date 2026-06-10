# -*- coding: utf-8 -*-
"""
SEM-6H.1b: No-Leak Matched-Null Q Validation

Purpose
-------
Rerun SEM-6H.1 without post-hoc / leakage-prone columns.

This script compares two clean selectors:

1) proxy_only_selected
   Uses the existing proxy_utility column directly for candidate selection.
   This is valid only if proxy_utility was produced before live oracle utility.

2) learned_noleak_Q_selected
   Learns Q from pre-write / structural features only:
     feat_*
     pool flags
     optional one-hot pool / relation metadata
   It explicitly excludes:
     oracle_utility
     candidate_gain
     candidate_margin_gain
     candidate_rank_improvement
     _real_q_score
     any columns containing utility/gain/oracle/rank_improvement/leak labels.

Core baselines
--------------
init_only
oracle_selected
commit_topk_rule
same_family_selected
random_pool_selected_mean
proxy_only_selected
learned_noleak_Q_selected

Matched nulls
-------------
For both proxy_only and learned_noleak_Q:
  shuffle_score_selected
  random_pool_selected
For learned_noleak_Q:
  permuted_utility_Q_selected

Outputs
-------
sem6h1b_outputs/
  sem6h1b_summary.csv
  sem6h1b_null_distributions.csv
  sem6h1b_null_vs_real.csv
  sem6h1b_pairwise_differences.csv
  sem6h1b_selection_rows.csv
  sem6h1b_scored_candidates.csv
  sem6h1b_verdict.json
  sem6h1b_feature_columns.txt

Run
---
python GPT_sem6h1b_noleak_matched_null_validation.py

Optional:
python GPT_sem6h1b_noleak_matched_null_validation.py --input sem6h1_scored_candidates.csv --n_null 100
"""

import argparse
import json
import math
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_SEED = 20260606
DEFAULT_INPUTS = [
    "sem6h1_scored_candidates.csv",
    r"sem6h_outputs\sem6h1_scored_candidates.csv",
    r"sem6e_outputs\sem6e_candidate_utilities.csv",
]
DEFAULT_OUT_DIR = "sem6h1b_outputs"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def find_input_path(user_path: str = None) -> Path:
    if user_path:
        p = Path(user_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Input not found: {p}")

    for x in DEFAULT_INPUTS:
        p = Path(x)
        if p.exists():
            return p
    raise FileNotFoundError(
        "No default input found. Tried:\n" + "\n".join(DEFAULT_INPUTS)
    )


def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    names = {norm(c): c for c in df.columns}

    def pick(cands, required=True):
        for c in cands:
            if norm(c) in names:
                return names[norm(c)]
        for c in cands:
            nc = norm(c)
            for k, v in names.items():
                if nc in k or k in nc:
                    return v
        if required:
            raise ValueError(f"Could not find column among {cands}. Columns={list(df.columns)}")
        return None

    return {
        "group": pick(["family", "query_id", "base_id", "target_id", "row_id", "prompt_id"]),
        "utility": pick(["oracle_utility", "utility", "true_utility", "target_utility", "observed_utility"]),
        "proxy": pick(["proxy_utility", "q_pred", "proxy_score", "q_score", "predicted_utility"], required=False),
        "pool": pick(["pool", "candidate_type", "cand_type", "candidate_source"], required=False),
        "candidate_family": pick(["candidate_family", "cand_family"], required=False),
        "target_concept": pick(["target_concept", "concept", "target_entity"], required=False),
        "target_operator": pick(["target_operator", "operator", "target_policy"], required=False),
        "candidate_concept": pick(["candidate_concept", "cand_concept"], required=False),
        "candidate_operator": pick(["candidate_operator", "cand_operator"], required=False),
    }


def add_flags(df: pd.DataFrame, cols: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    pool_col = cols.get("pool")
    if pool_col:
        s = out[pool_col].astype(str).str.lower()
    else:
        s = pd.Series([""] * len(out), index=out.index)

    out["_is_same_family"] = (
        s.str.contains("same_family|same-family|same family", regex=True)
        if "_is_same_family" not in out.columns
        else out["_is_same_family"].astype(bool)
    )
    out["_is_commit_topk"] = (
        s.str.contains("commit|topk|top_k|neighbor", regex=True)
        if "_is_commit_topk" not in out.columns
        else out["_is_commit_topk"].astype(bool)
    )
    out["_is_random"] = (
        s.str.contains("random|rand", regex=True)
        if "_is_random" not in out.columns
        else out["_is_random"].astype(bool)
    )
    return out


def choose_noleak_features(df: pd.DataFrame, cols: Dict[str, str], include_metadata: bool = True) -> Tuple[List[str], List[str]]:
    """
    Returns numeric_features, categorical_features.
    """
    leak_patterns = [
        "oracle", "utility", "gain", "margin_gain", "rank_improvement",
        "target_utility", "observed", "label", "selected",
        "real_q", "proxy_utility", "q_score", "q_pred", "predicted",
        "answer", "gen_", "verdict"
    ]

    numeric = []
    categorical = []

    # Conservative numeric features: feat_* and flags only.
    for c in df.columns:
        nc = norm(c)
        if any(p in nc for p in leak_patterns):
            continue
        if c in [cols.get("group"), cols.get("utility"), cols.get("proxy")]:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            if nc.startswith("feat_") or nc in ["is_same_family", "is_commit_topk", "is_random"]:
                if df[c].nunique(dropna=True) > 1:
                    numeric.append(c)

    for flag in ["_is_same_family", "_is_commit_topk", "_is_random"]:
        if flag in df.columns and flag not in numeric:
            numeric.append(flag)

    if include_metadata:
        for key in ["pool", "candidate_family", "target_operator", "candidate_operator"]:
            c = cols.get(key)
            if c and c in df.columns and c not in [cols.get("group"), cols.get("utility"), cols.get("proxy")]:
                nc = norm(c)
                if not any(p in nc for p in leak_patterns):
                    if df[c].nunique(dropna=True) > 1:
                        categorical.append(c)

    # Deduplicate preserving order.
    numeric = list(dict.fromkeys(numeric))
    categorical = list(dict.fromkeys(categorical))
    return numeric, categorical


def build_model(numeric_cols: List[str], categorical_cols: List[str], kind: str):
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        # sklearn compatibility
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(("cat", ohe, categorical_cols))
    pre = ColumnTransformer(transformers=transformers, remainder="drop")

    if kind == "ridge":
        reg = Ridge(alpha=10.0, random_state=RANDOM_SEED)
    elif kind == "rf":
        reg = RandomForestRegressor(
            n_estimators=80, min_samples_leaf=3, n_jobs=-1, random_state=RANDOM_SEED
        )
    else:
        reg = ExtraTreesRegressor(
            n_estimators=80, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_SEED
        )
    return Pipeline([("pre", pre), ("reg", reg)])


def crossfit_predict(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    utility_col: str,
    group_col: str,
    kind: str,
    permute_y: bool = False,
    seed: int = RANDOM_SEED,
):
    feature_cols = numeric_cols + categorical_cols
    X = df[feature_cols].copy()
    for c in numeric_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median() if X[c].notna().any() else 0.0)
    for c in categorical_cols:
        X[c] = X[c].astype(str).fillna("NA")

    y = pd.to_numeric(df[utility_col], errors="coerce").to_numpy(dtype=float)
    groups = df[group_col].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))
    if n_splits < 2:
        raise ValueError("Need at least two groups for GroupKFold.")

    splitter = GroupKFold(n_splits=n_splits)
    pred = np.full(len(df), np.nan)
    rng = np.random.default_rng(seed)
    metrics = []

    for fold, (tr, te) in enumerate(splitter.split(X, y, groups)):
        ytr = y[tr].copy()
        if permute_y:
            ytr = rng.permutation(ytr)
        model = build_model(numeric_cols, categorical_cols, kind)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X.iloc[tr], ytr)
            p = model.predict(X.iloc[te])
        pred[te] = p
        corr = np.corrcoef(y[te], p)[0, 1] if np.std(y[te]) > 1e-12 and np.std(p) > 1e-12 else np.nan
        r2 = r2_score(y[te], p) if len(te) > 1 else np.nan
        rmse = math.sqrt(mean_squared_error(y[te], p))
        metrics.append({"fold": fold, "corr": corr, "r2": r2, "rmse": rmse})

    return pred, {
        "cv_corr_mean": float(np.nanmean([m["corr"] for m in metrics])),
        "cv_r2_mean": float(np.nanmean([m["r2"] for m in metrics])),
        "cv_rmse_mean": float(np.nanmean([m["rmse"] for m in metrics])),
    }


def bootstrap_ci(values, n_boot=2000, seed=RANDOM_SEED):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def pick_by_score(sub, score_col):
    vals = pd.to_numeric(sub[score_col], errors="coerce").to_numpy(dtype=float)
    if np.all(~np.isfinite(vals)):
        return int(sub.index[0])
    vals = np.where(np.isfinite(vals), vals, -np.inf)
    return int(sub.index[int(np.argmax(vals))])


def pick_random(sub, rng):
    return int(rng.choice(sub.index.to_numpy()))


def pick_mask_then_score(sub, mask_col, score_col, rng):
    if mask_col in sub.columns:
        m = sub[sub[mask_col].fillna(False).astype(bool)]
        if len(m):
            return pick_by_score(m, score_col)
    return pick_random(sub, rng)


def evaluate_selection(df, group_col, utility_col, selection, picker):
    rows, vals = [], []
    for gid, sub in df.groupby(group_col, sort=False):
        try:
            idx = picker(sub)
        except Exception:
            continue
        val = float(df.loc[idx, utility_col])
        vals.append(val)
        row = {
            "selection": selection,
            "group_id": gid,
            "selected_index": idx,
            "selected_utility": val,
        }
        if "pool" in df.columns:
            row["pool"] = df.loc[idx, "pool"]
        elif "_pool_str" in df.columns:
            row["pool"] = df.loc[idx, "_pool_str"]
        for c in ["candidate_family", "candidate_concept", "candidate_operator", "_is_same_family", "_is_commit_topk", "_is_random"]:
            if c in df.columns:
                row[c] = df.loc[idx, c]
        rows.append(row)

    vals = np.asarray(vals, dtype=float)
    lo, hi = bootstrap_ci(vals) if len(vals) else (np.nan, np.nan)
    summary = {
        "selection": selection,
        "n_groups": int(len(vals)),
        "mean_utility": float(np.mean(vals)) if len(vals) else np.nan,
        "median_utility": float(np.median(vals)) if len(vals) else np.nan,
        "std_utility": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "positive_rate": float(np.mean(vals > 0)) if len(vals) else np.nan,
    }
    return pd.DataFrame(rows), summary


def pairwise(rows_df, a, b):
    A = rows_df[rows_df.selection == a][["group_id", "selected_utility"]].rename(columns={"selected_utility": "a"})
    B = rows_df[rows_df.selection == b][["group_id", "selected_utility"]].rename(columns={"selected_utility": "b"})
    m = A.merge(B, on="group_id")
    diff = (m.a - m.b).to_numpy(dtype=float)
    lo, hi = bootstrap_ci(diff) if len(diff) else (np.nan, np.nan)
    z = np.mean(diff) / (np.std(diff, ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 and np.std(diff, ddof=1) > 1e-12 else np.nan
    return {
        "comparison": f"{a}>{b}",
        "n": int(len(diff)),
        "mean_diff": float(np.mean(diff)) if len(diff) else np.nan,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "z": float(z) if np.isfinite(z) else np.nan,
        "win_rate": float(np.mean(diff > 0)) if len(diff) else np.nan,
    }


def null_for_score(df, group_col, utility_col, score_col, null_name, n_null, rng):
    summaries = []
    for i in range(n_null):
        tmp = df.copy()
        tmp["_null_score"] = rng.permutation(tmp[score_col].to_numpy())
        _, summ = evaluate_selection(
            tmp, group_col, utility_col, f"{null_name}_{i:03d}",
            lambda sub: pick_by_score(sub, "_null_score")
        )
        summ["null_type"] = null_name
        summ["iter"] = i
        summaries.append(summ)
    return pd.DataFrame(summaries)


def random_null(df, group_col, utility_col, n_null, rng):
    summaries = []
    for i in range(n_null):
        _, summ = evaluate_selection(
            df, group_col, utility_col, f"random_pool_selected_{i:03d}",
            lambda sub, r=rng: pick_random(sub, r)
        )
        summ["null_type"] = "random_pool_selected"
        summ["iter"] = i
        summaries.append(summ)
    return pd.DataFrame(summaries)


def permuted_learned_null(df, group_col, utility_col, numeric_cols, categorical_cols, kind, n_null):
    summaries = []
    for i in range(n_null):
        tmp = df.copy()
        pred, _ = crossfit_predict(
            tmp, numeric_cols, categorical_cols, utility_col, group_col, kind,
            permute_y=True, seed=RANDOM_SEED + 1000 + i
        )
        tmp["_perm_q"] = pred
        _, summ = evaluate_selection(
            tmp, group_col, utility_col, f"permuted_utility_Q_selected_{i:03d}",
            lambda sub: pick_by_score(sub, "_perm_q")
        )
        summ["null_type"] = "permuted_utility_Q_selected"
        summ["iter"] = i
        summaries.append(summ)
    return pd.DataFrame(summaries)


def summarize_null(null_df, real_means: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for null_type, sub in null_df.groupby("null_type"):
        vals = pd.to_numeric(sub["mean_utility"], errors="coerce").dropna().to_numpy()
        if len(vals) == 0:
            continue

        # Map null to relevant real selection.
        if null_type.startswith("proxy"):
            real_key = "proxy_only_selected"
        elif null_type.startswith("learned") or null_type.startswith("permuted"):
            real_key = "learned_noleak_Q_selected"
        else:
            # random applies to both; record twice
            for real_key in ["proxy_only_selected", "learned_noleak_Q_selected"]:
                real = real_means.get(real_key, np.nan)
                mu = float(np.mean(vals))
                sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
                z = (real - mu) / sd if np.isfinite(real) and sd and sd > 1e-12 else np.nan
                q95 = float(np.percentile(vals, 95))
                rows.append({
                    "real_selection": real_key,
                    "null_type": null_type,
                    "real_mean": real,
                    "null_mean": mu,
                    "null_std": sd,
                    "z_vs_real": float(z) if np.isfinite(z) else np.nan,
                    "q95": q95,
                    "real_gt_q95": bool(real > q95) if np.isfinite(real) else False,
                    "n_null": int(len(vals)),
                })
            continue

        real = real_means.get(real_key, np.nan)
        mu = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        z = (real - mu) / sd if np.isfinite(real) and sd and sd > 1e-12 else np.nan
        q95 = float(np.percentile(vals, 95))
        rows.append({
            "real_selection": real_key,
            "null_type": null_type,
            "real_mean": real,
            "null_mean": mu,
            "null_std": sd,
            "z_vs_real": float(z) if np.isfinite(z) else np.nan,
            "q95": q95,
            "real_gt_q95": bool(real > q95) if np.isfinite(real) else False,
            "n_null": int(len(vals)),
        })
    return pd.DataFrame(rows)


def verdict(summary_df, null_vs_real, q_metrics, feature_cols):
    means = dict(zip(summary_df.selection, summary_df.mean_utility))
    out = {
        "means": {k: float(v) for k, v in means.items()},
        "q_metrics": q_metrics,
        "feature_columns": feature_cols,
        "verdicts": {},
        "interpretation": []
    }

    for key in ["proxy_only_selected", "learned_noleak_Q_selected"]:
        real = means.get(key, np.nan)
        commit = means.get("commit_topk_rule", np.nan)
        oracle = means.get("oracle_selected", np.nan)
        init = means.get("init_only", 0.0)
        same = means.get("same_family_selected", np.nan)

        sub = null_vs_real[null_vs_real.real_selection == key]
        strong_nulls = sub[(sub.real_gt_q95 == True) & (sub.z_vs_real > 2.0)]
        required = set(sub.null_type.unique())
        passed_required = set(strong_nulls.null_type.unique())

        pass_lite = np.isfinite(real) and real > init and (not np.isfinite(same) or real > same)
        pass_strong = pass_lite and required.issubset(passed_required) and (not np.isfinite(oracle) or oracle >= real)
        pass_milestone = pass_strong and np.isfinite(commit) and real >= commit

        if pass_milestone:
            v = "PASS_MILESTONE_NOLEAK_GE_COMMIT_AND_MATCHED_NULL"
        elif pass_strong:
            v = "PASS_STRONG_NOLEAK_MATCHED_NULL"
        elif pass_lite:
            v = "PASS_LITE_NOLEAK_BEATS_BASIC_BASELINES"
        else:
            v = "FAIL_OR_INCONCLUSIVE"

        out["verdicts"][key] = {
            "verdict": v,
            "pass_lite": bool(pass_lite),
            "pass_strong": bool(pass_strong),
            "pass_milestone": bool(pass_milestone),
            "required_nulls": sorted(required),
            "passed_nulls": sorted(passed_required),
        }

    out["interpretation"].append(
        "If proxy_only passes but learned_noleak fails, existing proxy_utility is useful but the clean feature set is insufficient."
    )
    out["interpretation"].append(
        "If learned_noleak passes commit_topk and matched nulls, SEM-6H.1b supports clean Q_theta selection."
    )
    out["interpretation"].append(
        "If both fail against commit_topk but pass nulls, Q is real but still weaker than identity/commit heuristic."
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--model_kind", default="extratrees", choices=["extratrees", "rf", "ridge"])
    ap.add_argument("--n_null", type=int, default=20)
    ap.add_argument("--no_metadata", action="store_true")
    ap.add_argument("--skip_permuted_null", action="store_true")
    args = ap.parse_args()

    in_path = find_input_path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    cols = detect_columns(df)
    df = add_flags(df, cols)
    if cols.get("pool"):
        df["_pool_str"] = df[cols["pool"]].astype(str)
    else:
        df["_pool_str"] = ""

    utility_col = cols["utility"]
    group_col = cols["group"]
    proxy_col = cols["proxy"]

    df[utility_col] = pd.to_numeric(df[utility_col], errors="coerce")
    df = df.dropna(subset=[utility_col, group_col]).reset_index(drop=True)

    numeric_cols, categorical_cols = choose_noleak_features(df, cols, include_metadata=(not args.no_metadata))
    feature_cols = numeric_cols + categorical_cols

    with open(out_dir / "sem6h1b_feature_columns.txt", "w", encoding="utf-8") as f:
        f.write("# numeric\n")
        for c in numeric_cols:
            f.write(c + "\n")
        f.write("# categorical\n")
        for c in categorical_cols:
            f.write(c + "\n")

    pred, q_metrics = crossfit_predict(
        df, numeric_cols, categorical_cols, utility_col, group_col, args.model_kind,
        permute_y=False, seed=RANDOM_SEED
    )
    df["_learned_noleak_q"] = pred

    if proxy_col:
        df["_proxy_score"] = pd.to_numeric(df[proxy_col], errors="coerce")
        df["_proxy_score"] = df["_proxy_score"].fillna(df["_proxy_score"].median())
    else:
        df["_proxy_score"] = np.nan

    rng = np.random.default_rng(RANDOM_SEED)

    selection_frames = []
    summaries = []

    # init baseline: utility is delta, so zero if no init column.
    gids = list(df[group_col].drop_duplicates())
    init_rows = [{"selection": "init_only", "group_id": g, "selected_index": -1, "selected_utility": 0.0} for g in gids]
    selection_frames.append(pd.DataFrame(init_rows))
    summaries.append({
        "selection": "init_only", "n_groups": len(gids), "mean_utility": 0.0,
        "median_utility": 0.0, "std_utility": 0.0, "ci95_lo": 0.0, "ci95_hi": 0.0, "positive_rate": 0.0
    })

    base_selectors = [
        ("oracle_selected", lambda sub: pick_by_score(sub, utility_col)),
        ("commit_topk_rule", lambda sub: pick_mask_then_score(sub, "_is_commit_topk", "_proxy_score" if proxy_col else "_learned_noleak_q", rng)),
        ("same_family_selected", lambda sub: pick_mask_then_score(sub, "_is_same_family", "_proxy_score" if proxy_col else "_learned_noleak_q", rng)),
        ("learned_noleak_Q_selected", lambda sub: pick_by_score(sub, "_learned_noleak_q")),
    ]
    if proxy_col:
        base_selectors.append(("proxy_only_selected", lambda sub: pick_by_score(sub, "_proxy_score")))

    for name, picker in base_selectors:
        rows, summ = evaluate_selection(df, group_col, utility_col, name, picker)
        selection_frames.append(rows)
        summaries.append(summ)

    # Random mean baseline.
    random_summaries = []
    for i in range(args.n_null):
        _, summ = evaluate_selection(df, group_col, utility_col, f"random_pool_selected_repl_{i:03d}", lambda sub, r=rng: pick_random(sub, r))
        random_summaries.append(summ["mean_utility"])
    arr = np.asarray(random_summaries, dtype=float)
    summaries.append({
        "selection": "random_pool_selected_mean",
        "n_groups": int(df[group_col].nunique()),
        "mean_utility": float(np.mean(arr)),
        "median_utility": float(np.median(arr)),
        "std_utility": float(np.std(arr, ddof=1)),
        "ci95_lo": float(np.percentile(arr, 2.5)),
        "ci95_hi": float(np.percentile(arr, 97.5)),
        "positive_rate": np.nan,
    })

    selection_rows = pd.concat(selection_frames, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    null_frames = []
    null_frames.append(random_null(df, group_col, utility_col, args.n_null, rng))
    null_frames.append(null_for_score(df, group_col, utility_col, "_learned_noleak_q", "learned_shuffle_Q_selected", args.n_null, rng))
    if not args.skip_permuted_null:
        null_frames.append(permuted_learned_null(df, group_col, utility_col, numeric_cols, categorical_cols, args.model_kind, args.n_null))
    if proxy_col:
        null_frames.append(null_for_score(df, group_col, utility_col, "_proxy_score", "proxy_shuffle_Q_selected", args.n_null, rng))

    null_df = pd.concat(null_frames, ignore_index=True)
    means = dict(zip(summary_df.selection, summary_df.mean_utility))
    null_vs = summarize_null(null_df, means)

    pair_rows = []
    for a in ["learned_noleak_Q_selected", "proxy_only_selected"]:
        if a not in set(selection_rows.selection):
            continue
        for b in ["init_only", "same_family_selected", "commit_topk_rule", "oracle_selected"]:
            pair_rows.append(pairwise(selection_rows, a, b))
    pair_df = pd.DataFrame(pair_rows)

    vdict = verdict(summary_df, null_vs, q_metrics, feature_cols)
    vdict["input"] = str(in_path)
    vdict["detected_columns"] = cols
    vdict["n_rows"] = int(len(df))
    vdict["n_groups"] = int(df[group_col].nunique())
    vdict["n_candidates_per_group_mean"] = float(df.groupby(group_col).size().mean())
    vdict["explicitly_excluded_leakage_columns"] = [
        "oracle_utility", "candidate_gain", "candidate_margin_gain",
        "candidate_rank_improvement", "_real_q_score", "proxy_utility for learned-Q"
    ]

    df.to_csv(out_dir / "sem6h1b_scored_candidates.csv", index=False, encoding="utf-8-sig")
    selection_rows.to_csv(out_dir / "sem6h1b_selection_rows.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "sem6h1b_summary.csv", index=False, encoding="utf-8-sig")
    null_df.to_csv(out_dir / "sem6h1b_null_distributions.csv", index=False, encoding="utf-8-sig")
    null_vs.to_csv(out_dir / "sem6h1b_null_vs_real.csv", index=False, encoding="utf-8-sig")
    pair_df.to_csv(out_dir / "sem6h1b_pairwise_differences.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "sem6h1b_verdict.json", "w", encoding="utf-8") as f:
        json.dump(vdict, f, ensure_ascii=False, indent=2)

    print("\n========== SEM-6H.1b SUMMARY ==========")
    print(summary_df.to_string(index=False))
    print("\n========== NULL VS REAL ==========")
    print(null_vs.to_string(index=False))
    print("\n========== PAIRWISE ==========")
    print(pair_df.to_string(index=False))
    print("\n========== VERDICT ==========")
    print(json.dumps(vdict, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
