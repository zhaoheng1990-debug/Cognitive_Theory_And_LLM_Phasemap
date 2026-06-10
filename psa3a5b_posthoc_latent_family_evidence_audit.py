# -*- coding: utf-8 -*-
"""
PSA-3A.5B: Post-hoc Latent Family Evidence Audit
================================================

Analyze existing PSA-3A.5 outputs without re-running the model.

This script answers:
1. Did anti-leak gates pass?
2. Did label-free latent trajectory families emerge over layers?
3. Are latent families independent of diagnostic mechanism labels?
4. Did CFT fail because the regression target was wrong?
5. Can TopK/CFT aggregate features classify latent family better than text leakage?

Run:
    python psa3a5b_posthoc_latent_family_evidence_audit.py
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, adjusted_rand_score, normalized_mutual_info_score, r2_score
from scipy.stats import pearsonr, spearmanr


DEFAULT_IN_DIR = r"C:\Users\ZH\Desktop\AGI\outputs\psa3a5_label_free_outputs"


def safe_corr(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(pearsonr(x, y)[0])


def classify_group_cv(df, target, cols, groups):
    X = df[cols].values
    y = df[target].values.astype(int)
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced", C=1.0)
    )
    pred = cross_val_predict(clf, X, y, cv=cv.split(X, y, groups))
    return {
        "acc": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }


def ridge_group_cv(df, target, cols, groups):
    X = df[cols].values
    y = df[target].values.astype(float)
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    pred = np.zeros_like(y, dtype=np.float64)
    for tr, te in cv.split(X, y, groups):
        model = make_pipeline(StandardScaler(with_mean=True, with_std=True), Ridge(alpha=1.0))
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
    spr = None
    if np.std(pred) > 1e-12 and np.std(y) > 1e-12:
        spr = float(spearmanr(pred, y).correlation)
    return {
        "corr": safe_corr(pred, y),
        "spearman": spr,
        "r2": float(r2_score(y, pred)),
    }


def main():
    in_dir = Path(DEFAULT_IN_DIR)

    summary = json.loads((in_dir / "psa3a5_summary.json").read_text(encoding="utf-8"))
    text_leak = json.loads((in_dir / "psa3a5_text_leakage.json").read_text(encoding="utf-8"))
    layer = pd.read_csv(in_dir / "psa3a5_layer_metrics.csv")
    sample = pd.read_csv(in_dir / "psa3a5_sample_metrics.csv")
    pairs = pd.read_csv(in_dir / "psa3a5_pairs_with_latent_family.csv")

    first = layer.iloc[0]
    best_acc_row = layer.loc[layer["latent_acc"].idxmax()]
    max_H_row = layer.loc[layer["H_latent_mean"].idxmax()]
    min_H_row = layer.loc[layer["H_latent_mean"].idxmin()]

    emergence_acc_gain = float(best_acc_row["latent_acc"] - first["latent_acc"])
    mid_H = float(sample["H_latent_mid"].mean())
    init_H = float(sample["H_latent_init"].mean())
    selection_H = float(sample["H_latent_selection"].mean())

    ari = float(adjusted_rand_score(pairs["mechanism_id"], pairs["latent_family_id"]))
    nmi = float(normalized_mutual_info_score(pairs["mechanism_id"], pairs["latent_family_id"]))

    pre_topk_cols = [
        "topk_entropy_delta_init",
        "topk_entropy_delta_mid",
        "topk_entropy_delta_slope_mid",
        "topk_spread_delta_init",
        "topk_spread_delta_mid",
        "center_delta_drift_mid_selection",
    ]

    all_topk_cols = [
        "topk_entropy_delta_init",
        "topk_entropy_delta_mid",
        "topk_entropy_delta_selection",
        "topk_entropy_delta_change_init_selection",
        "topk_entropy_delta_slope_mid",
        "topk_entropy_delta_slope_selection",
        "topk_spread_delta_init",
        "topk_spread_delta_mid",
        "topk_spread_delta_selection",
        "topk_spread_delta_change_init_selection",
        "center_delta_drift_init_selection",
        "center_delta_drift_mid_selection",
    ]

    cft_cols = all_topk_cols + [
        "hidden_delta_vel_mid",
        "hidden_delta_vel_selection",
    ]

    groups = sample["base_id"].values

    cls = {
        "pre_topk_to_latent_family": classify_group_cv(sample, "latent_family_id", pre_topk_cols, groups),
        "all_topk_to_latent_family": classify_group_cv(sample, "latent_family_id", all_topk_cols, groups),
        "cft_proxy_to_latent_family": classify_group_cv(sample, "latent_family_id", cft_cols, groups),
    }

    regress = {}
    for target in ["Selection_eff_init_selection", "Unfolding_eff_init_selection"]:
        regress[target] = {
            "pre_topk": ridge_group_cv(sample, target, pre_topk_cols, groups),
            "all_topk": ridge_group_cv(sample, target, all_topk_cols, groups),
            "cft_proxy": ridge_group_cv(sample, target, cft_cols, groups),
        }

    text_f1 = float(text_leak["text_tfidf_macro_f1_group"])
    cft_f1 = cls["cft_proxy_to_latent_family"]["macro_f1"]
    all_topk_f1 = cls["all_topk_to_latent_family"]["macro_f1"]
    pre_topk_f1 = cls["pre_topk_to_latent_family"]["macro_f1"]

    anti_leak_ok = (
        not summary["anti_leak_flags"]["text_leak_caveat"]
        and not summary["anti_leak_flags"]["first_layer_caveat"]
    )

    label_free_ok = (
        ari < 0.05 and nmi < 0.05
    )

    layer_emergence_ok = (
        emergence_acc_gain > 0.15
        and float(best_acc_row["layer"]) >= 10
    )

    selection_pattern_ok = (
        mid_H > init_H and mid_H > selection_H
    )

    cft_cls_ok = (
        cft_f1 > text_f1 + 0.10
        and cft_f1 >= 0.45
    )

    if anti_leak_ok and label_free_ok and layer_emergence_ok and selection_pattern_ok and cft_cls_ok:
        verdict = "PASS_PSA3A5B_LABEL_FREE_LATENT_FAMILIES_WITH_CFT_CLASSIFICATION"
    elif anti_leak_ok and label_free_ok and layer_emergence_ok and selection_pattern_ok:
        verdict = "PARTIAL_STRONG_LABEL_FREE_LATENT_FAMILIES_CFT_REGRESSION_TARGET_WRONG"
    elif anti_leak_ok and label_free_ok and layer_emergence_ok:
        verdict = "PARTIAL_LABEL_FREE_LATENT_FAMILY_EMERGENCE"
    else:
        verdict = "FAIL_OR_CAVEAT_LABEL_FREE_EVIDENCE"

    out = {
        "verdict": verdict,
        "interpretation": {
            "main_point": (
                "PSA-3A.5 supports label-free latent trajectory families if anti-leak, "
                "label independence, and layer emergence hold. CFT should be tested as a "
                "classifier or family-geometry predictor, not only as entropy-effect regression."
            ),
            "terminology": "selection / commitment / trajectory family; no basin terminology",
        },
        "anti_leak": {
            "text_tfidf_macro_f1": text_f1,
            "text_tfidf_acc": float(text_leak["text_tfidf_acc_group"]),
            "first_layer_acc": float(first["latent_acc"]),
            "first_layer_H": float(first["H_latent_mean"]),
            "anti_leak_ok": bool(anti_leak_ok),
        },
        "layer_emergence": {
            "best_acc_layer": int(best_acc_row["layer"]),
            "best_acc": float(best_acc_row["latent_acc"]),
            "first_layer_acc": float(first["latent_acc"]),
            "emergence_acc_gain": emergence_acc_gain,
            "max_H_layer": int(max_H_row["layer"]),
            "max_H": float(max_H_row["H_latent_mean"]),
            "min_H_layer": int(min_H_row["layer"]),
            "min_H": float(min_H_row["H_latent_mean"]),
            "H_init_mean": init_H,
            "H_mid_mean": mid_H,
            "H_selection_mean": selection_H,
            "selection_pattern_ok": bool(selection_pattern_ok),
        },
        "label_free_diagnostics": {
            "latent_vs_diagnostic_mechanism_ARI": ari,
            "latent_vs_diagnostic_mechanism_NMI": nmi,
            "label_free_ok": bool(label_free_ok),
            "latent_family_counts": {
                str(k): int(v) for k, v in pairs["latent_family_id"].value_counts().sort_index().items()
            },
        },
        "feature_to_latent_family_classification": cls,
        "effect_regression_diagnostic": regress,
        "flags": {
            "anti_leak_ok": bool(anti_leak_ok),
            "label_free_ok": bool(label_free_ok),
            "layer_emergence_ok": bool(layer_emergence_ok),
            "selection_pattern_ok": bool(selection_pattern_ok),
            "cft_classification_ok": bool(cft_cls_ok),
        },
        "source_summary_verdict": summary["verdict"],
    }

    (in_dir / "psa3a5b_posthoc_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (in_dir / "psa3a5b_posthoc_verdict.txt").write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
