# -*- coding: utf-8 -*-
"""
SEM-6F: Proxy-Guided Beneficial Experience Write Audit

Why SEM-6F
----------
SEM-6E showed:
    - same-family updates have negative utility;
    - beneficial candidates exist in broader pools;
    - proxy utility is learnable with GroupKFold:
        corr ~ 0.887, R2 ~ 0.787, AUC ~ 0.897.

SEM-6F tests operational write selection:

    Train Q_hat on candidate utilities from other families.
    For held-out family, select candidate with max Q_hat.
    Evaluate selected candidate using the already measured oracle utility.

This is fast and does not rerun the model; it uses SEM-6E's candidate table.

Inputs:
    sem6e_outputs/sem6e_candidate_utilities.csv
or fallback:
    sem6e_candidate_utilities.csv

Outputs:
    sem6f_outputs/
        sem6f_proxy_selection_results.csv
        sem6f_pool_selection_summary.csv
        sem6f_pass_checks.csv
        sem6f_report.json

Core comparisons:
    proxy_selected_best
    oracle_selected_best
    same_family_best
    commit_topk_best
    random_best
    init baseline = 0 utility

If proxy_selected_best > same_family_best and > random_best and > 0:
    Q_proxy is operational for beneficial experience search.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score, f1_score


RANDOM_SEED = 42
OUT_DIR = Path("sem6f_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CANDIDATES = [
    Path("sem6e_outputs/sem6e_candidate_utilities.csv"),
    Path("sem6e_candidate_utilities.csv"),
    Path(r"C:\Users\ZH\Desktop\AGI\python_script\sem6e_outputs\sem6e_candidate_utilities.csv"),
]


def find_input():
    for p in INPUT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("Cannot find sem6e_candidate_utilities.csv. Put this script beside the file or keep sem6e_outputs/.")


def main():
    inp = find_input()
    df = pd.read_csv(inp)

    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    if not feat_cols:
        raise ValueError("No feat_* columns found in SEM-6E candidate utilities.")

    X = df[feat_cols].values.astype(np.float32)
    y = df["oracle_utility"].values.astype(np.float32)
    groups = df["family"].values
    positive = (y > 0).astype(int)

    unique_fams = np.unique(groups)
    n_splits = min(5, len(unique_fams))
    if n_splits < 2:
        raise ValueError("Need at least 2 families for GroupKFold.")

    df["proxy_pred"] = np.nan
    df["proxy_prob_positive"] = np.nan

    gkf = GroupKFold(n_splits=n_splits)

    for train_idx, test_idx in gkf.split(X, y, groups):
        reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        reg.fit(X[train_idx], y[train_idx])
        df.loc[test_idx, "proxy_pred"] = reg.predict(X[test_idx])

        # Logistic only if train has both classes
        ybin = positive[train_idx]
        if len(set(ybin)) == 2:
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
            clf.fit(X[train_idx], ybin)
            df.loc[test_idx, "proxy_prob_positive"] = clf.predict_proba(X[test_idx])[:, 1]
        else:
            df.loc[test_idx, "proxy_prob_positive"] = df.loc[test_idx, "proxy_pred"]

    proxy_r2 = float(r2_score(y, df["proxy_pred"].values))
    proxy_corr = float(np.corrcoef(y, df["proxy_pred"].values)[0, 1]) if np.std(df["proxy_pred"].values) > 1e-8 else 0.0
    proxy_auc = float(roc_auc_score(positive, df["proxy_pred"].values)) if len(set(positive)) == 2 else np.nan

    # Select candidates per family using proxy.
    rows = []
    for fam, sub in df.groupby("family"):
        # full proxy selection across all pools
        proxy_best = sub.sort_values("proxy_pred", ascending=False).iloc[0]
        oracle_best = sub.sort_values("oracle_utility", ascending=False).iloc[0]

        # pool baselines
        def best_pool(pool):
            ss = sub[sub["pool"] == pool]
            if len(ss) == 0:
                return None
            return ss.sort_values("oracle_utility", ascending=False).iloc[0]

        baselines = {
            "same_family_best": best_pool("same_family_update"),
            "commit_topk_best": best_pool("commit_topk_neighbors"),
            "same_operator_best": best_pool("same_operator_other_concept"),
            "same_concept_best": best_pool("same_concept_other_operator"),
            "random_best": best_pool("random_other_family"),
        }

        row = {
            "family": fam,
            "proxy_selected_pool": proxy_best["pool"],
            "proxy_selected_candidate_family": proxy_best["candidate_family"],
            "proxy_selected_utility": float(proxy_best["oracle_utility"]),
            "proxy_selected_pred": float(proxy_best["proxy_pred"]),
            "proxy_selected_positive": int(proxy_best["oracle_utility"] > 0),

            "oracle_selected_pool": oracle_best["pool"],
            "oracle_selected_candidate_family": oracle_best["candidate_family"],
            "oracle_selected_utility": float(oracle_best["oracle_utility"]),
            "oracle_selected_positive": int(oracle_best["oracle_utility"] > 0),
        }

        for name, r in baselines.items():
            if r is None:
                row[name + "_utility"] = np.nan
                row[name + "_pool"] = None
            else:
                row[name + "_utility"] = float(r["oracle_utility"])
                row[name + "_pool"] = r["pool"]

        rows.append(row)

    sel = pd.DataFrame(rows)

    sel_path = OUT_DIR / "sem6f_proxy_selection_results.csv"
    sel.to_csv(sel_path, index=False, encoding="utf-8-sig")

    # Summary
    summary_items = []
    utility_cols = [
        "proxy_selected_utility",
        "oracle_selected_utility",
        "same_family_best_utility",
        "commit_topk_best_utility",
        "same_operator_best_utility",
        "same_concept_best_utility",
        "random_best_utility",
    ]
    for col in utility_cols:
        vals = sel[col].dropna().values
        summary_items.append({
            "selector": col.replace("_utility", ""),
            "n": int(len(vals)),
            "mean_utility": float(np.mean(vals)),
            "median_utility": float(np.median(vals)),
            "positive_rate": float(np.mean(vals > 0)),
            "min_utility": float(np.min(vals)),
            "max_utility": float(np.max(vals)),
        })
    summary = pd.DataFrame(summary_items)
    summary_path = OUT_DIR / "sem6f_selection_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # Pool distribution for proxy selection
    pool_dist = (
        sel.groupby("proxy_selected_pool")
        .agg(n=("family", "count"), mean_utility=("proxy_selected_utility", "mean"), positive_rate=("proxy_selected_positive", "mean"))
        .reset_index()
        .sort_values("n", ascending=False)
    )
    pool_dist_path = OUT_DIR / "sem6f_proxy_pool_distribution.csv"
    pool_dist.to_csv(pool_dist_path, index=False, encoding="utf-8-sig")

    def mean_col(c):
        return float(sel[c].mean())

    proxy_mean = mean_col("proxy_selected_utility")
    samefam_mean = mean_col("same_family_best_utility")
    commit_mean = mean_col("commit_topk_best_utility")
    sameop_mean = mean_col("same_operator_best_utility")
    samecon_mean = mean_col("same_concept_best_utility")
    rand_mean = mean_col("random_best_utility")
    oracle_mean = mean_col("oracle_selected_utility")

    checks = pd.DataFrame([
        {
            "check": "proxy_beats_same_family",
            "value": proxy_mean,
            "baseline": samefam_mean,
            "pass_lite": bool(proxy_mean > samefam_mean),
            "pass_strong": bool(proxy_mean > samefam_mean and proxy_mean > 0),
            "note": "Proxy selected candidate should beat same-family best and be positive."
        },
        {
            "check": "proxy_beats_random",
            "value": proxy_mean,
            "baseline": rand_mean,
            "pass_lite": bool(proxy_mean > rand_mean),
            "pass_strong": bool(proxy_mean > rand_mean and proxy_mean > 0),
            "note": "Proxy selection should beat random candidate search."
        },
        {
            "check": "proxy_competes_with_commit_topk",
            "value": proxy_mean,
            "baseline": commit_mean,
            "pass_lite": bool(proxy_mean >= commit_mean),
            "pass_strong": bool(proxy_mean >= commit_mean and proxy_mean > 0),
            "note": "Proxy selection should match or beat simple commit-topk best."
        },
        {
            "check": "proxy_utility_predictability",
            "value_corr": proxy_corr,
            "value_r2": proxy_r2,
            "value_auc": proxy_auc,
            "pass_lite": bool(proxy_corr > 0.2),
            "pass_strong": bool(proxy_corr > 0.4 and proxy_auc > 0.65),
            "note": "Proxy should predict candidate utility under family-held-out CV."
        },
        {
            "check": "oracle_gap_remaining",
            "value": proxy_mean,
            "oracle": oracle_mean,
            "gap": oracle_mean - proxy_mean,
            "pass_lite": bool(proxy_mean > 0),
            "pass_strong": bool((oracle_mean - proxy_mean) < abs(oracle_mean) * 0.5),
            "note": "Proxy can be positive even if an oracle gap remains."
        },
    ])

    checks_path = OUT_DIR / "sem6f_pass_checks.csv"
    checks.to_csv(checks_path, index=False, encoding="utf-8-sig")

    strong = int(checks["pass_strong"].sum())
    lite = int(checks["pass_lite"].sum())
    verdict = "FAIL"
    if lite >= 2:
        verdict = "PASS-Lite"
    if strong >= 3:
        verdict = "PASS-Strong"

    report = {
        "experiment": "SEM-6F Proxy-Guided Beneficial Experience Write Audit",
        "verdict": verdict,
        "input": str(inp),
        "proxy_utility": {
            "groupkfold_r2": proxy_r2,
            "groupkfold_corr": proxy_corr,
            "groupkfold_auc_positive": proxy_auc,
            "n_candidates": int(len(df)),
            "n_families": int(len(sel)),
        },
        "mean_utilities": {
            "proxy_selected": proxy_mean,
            "oracle_selected": oracle_mean,
            "same_family_best": samefam_mean,
            "commit_topk_best": commit_mean,
            "same_operator_best": sameop_mean,
            "same_concept_best": samecon_mean,
            "random_best": rand_mean,
        },
        "interpretation_rules": [
            "If proxy_selected > same_family_best and > random_best and > 0, Q_proxy is operational for beneficial experience search.",
            "If proxy_selected is close to commit_topk_best, commit-neighborhood search is a strong simple baseline.",
            "If oracle gap remains large, future work should improve Q features / policy."
        ],
        "outputs": {
            "selection_results": str(sel_path),
            "selection_summary": str(summary_path),
            "proxy_pool_distribution": str(pool_dist_path),
            "checks": str(checks_path),
        }
    }

    report_path = OUT_DIR / "sem6f_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 90)
    print("SEM-6F finished")
    print("=" * 90)
    print("Proxy metrics:", report["proxy_utility"])
    print("\nSummary:")
    print(summary)
    print("\nPool distribution:")
    print(pool_dist)
    print("\nChecks:")
    print(checks)
    print("\nVerdict:", verdict)
    print(f"Saved: {sel_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {pool_dist_path}")
    print(f"Saved: {checks_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
