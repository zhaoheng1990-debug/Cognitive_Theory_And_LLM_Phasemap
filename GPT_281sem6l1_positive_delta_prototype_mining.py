# -*- coding: utf-8 -*-
r"""
SEM-6L.1: Positive Delta Prototype Mining

Purpose
-------
SEM-6K showed that current action-conditioned Q(E,a) does not close the loop because
the available hand-built action primitives do not have stable positive mean effect.

SEM-6L.1 shifts the question:

    Not: Which action should we choose?
    But: What do positive write deltas have in common?

It mines all live SEM-6 write experiments and extracts the structural signature of
positive rows:

    trajectory_improvement > threshold

from:
    SEM-6H.2 synthetic latent write
    SEM-6H.2b rebuilt MemoryUnit hidden write
    SEM-6J prompt-delta operator write
    SEM-6J.3 address gate
    SEM-6J.3b expanded address gate
    SEM-6K merged action-conditioned dataset

Outputs
-------
sem6l1_outputs/
    sem6l1_all_live_rows.csv
    sem6l1_positive_rows.csv
    sem6l1_positive_enrichment.csv
    sem6l1_numeric_contrasts.csv
    sem6l1_prototype_rules.csv
    sem6l1_positive_cluster_summary.csv
    sem6l1_verdict.json

Run
---
conda activate dhrf_4080s
cd C:\Users\ZH\Desktop\AGI\python_script

python GPT_sem6l1_positive_delta_prototype_mining.py

Optional:
python GPT_sem6l1_positive_delta_prototype_mining.py --threshold 0.01
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_ROOT = r"C:\Users\ZH\Desktop\AGI\python_script"
DEFAULT_OUT_DIR = rf"{DEFAULT_ROOT}\sem6l1_outputs"

DEFAULT_FILES = {
    "6H2_synthetic": rf"{DEFAULT_ROOT}\sem6h2_outputs\sem6h2_live_results.csv",
    "6H2b_memoryunit": rf"{DEFAULT_ROOT}\sem6h2_outputs\sem6h2b_live_results.csv",
    "6J_operator_delta": rf"{DEFAULT_ROOT}\sem6j_outputs\sem6j_live_results.csv",
    "6J3_address_gate": rf"{DEFAULT_ROOT}\sem6j3_outputs\sem6j3_address_gate_live_results.csv",
    "6J3b_address_expanded": rf"{DEFAULT_ROOT}\sem6j3b_outputs\sem6j3b_address_gate_live_results.csv",
    "6K_merged": rf"{DEFAULT_ROOT}\sem6k_outputs\sem6k_action_conditioned_dataset.csv",
}

RANDOM_SEED = 20260606


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def find_metric(df: pd.DataFrame, preferred: str = "") -> Optional[str]:
    if preferred and preferred in df.columns:
        return preferred
    candidates = [
        "trajectory_improvement",
        "live_margin_gain",
        "live_gain",
        "target_value",
        "live_logprob_gap_gain",
        "live_rank_improvement",
    ]
    for c in candidates:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().sum() > 0:
                return c
    for c in df.columns:
        nc = norm(c)
        if any(x in nc for x in ["trajectory_improvement", "live_margin_gain", "live_gain", "target_value"]):
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().sum() > 0:
                return c
    return None


def coerce_action(row: pd.Series, source: str) -> str:
    # Most reliable explicit action fields.
    for c in ["action", "minimal_action", "asa_action"]:
        if c in row.index and pd.notna(row[c]):
            v = str(row[c])
            if v in {"address_gate", "operator_prior_transfer", "no_intervention", "correction_operator"}:
                return v
            if "operator" in v and v != "operator_prior_transfer":
                return "correction_operator"
    route = str(row.get("write_route", ""))
    if route == "write_address_gate":
        return "address_gate"
    if route == "write_operator_prior":
        return "operator_prior_transfer"
    if route == "reject_as_null":
        return "no_intervention"
    if route in {"trigger_correction_operator", "write_residual_mode"}:
        return "correction_operator"
    if "address" in source.lower():
        return "address_gate"
    if source.startswith("6H2"):
        return "hidden_vector_write"
    return "unknown_action"


def load_one(path: str, source: str, preferred_metric: str = "") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        print(f"[skip] {source}: not found {p}")
        return pd.DataFrame()
    df = pd.read_csv(p)
    metric = find_metric(df, preferred_metric)
    if metric is None:
        print(f"[skip] {source}: no metric")
        return pd.DataFrame()
    out = df.copy()
    out["source_experiment"] = source
    out["metric_used"] = metric
    out["live_value"] = pd.to_numeric(out[metric], errors="coerce")
    out = out.dropna(subset=["live_value"]).copy()
    out["action"] = out.apply(lambda r: coerce_action(r, source), axis=1)

    # Normalize fields.
    if "_selection_family" not in out.columns:
        out["_selection_family"] = out.get("selection_family", source)
    if "_group_id" not in out.columns:
        if "family" in out.columns:
            out["_group_id"] = out["family"]
        else:
            out["_group_id"] = out.index.astype(str)

    for c in ["relation_type", "write_route", "target_operator", "candidate_operator",
              "target_concept", "candidate_concept", "pool"]:
        if c not in out.columns:
            out[c] = "NA"

    # Numeric optional fields.
    for c in [
        "_learned_noleak_q", "proxy_utility", "oracle_utility",
        "beta", "delta_scale", "alpha_operator", "beta_precursor", "asa_alpha", "asa_beta",
        "base_margin", "base_rank", "base_logprob_gap_to_top", "live_rank_improvement",
        "n_delta_layers",
    ]:
        if c not in out.columns:
            out[c] = np.nan
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    print(f"[load] {source}: {len(out)} rows, metric={metric}")
    return out


def build_all(files: Dict[str, str], preferred_metric: str = "") -> pd.DataFrame:
    parts = []
    for src, path in files.items():
        part = load_one(path, src, preferred_metric)
        if not part.empty:
            parts.append(part)
    if not parts:
        raise FileNotFoundError("No live rows loaded.")
    df = pd.concat(parts, ignore_index=True)

    # Derived fields.
    df["is_positive_gt0"] = (df["live_value"] > 0).astype(int)
    df["is_positive_ge0"] = (df["live_value"] >= 0).astype(int)
    df["q_proxy_gap"] = df["_learned_noleak_q"] - df["proxy_utility"]
    df["q_oracle_gap"] = df["_learned_noleak_q"] - df["oracle_utility"]
    return df


def categorical_enrichment(df: pd.DataFrame, pos_col: str, out_dir: Path) -> pd.DataFrame:
    cats = [
        "source_experiment", "action", "_selection_family", "relation_type",
        "write_route", "target_operator", "candidate_operator", "pool"
    ]
    rows = []
    base_rate = df[pos_col].mean()
    for c in cats:
        if c not in df.columns:
            continue
        for val, sub in df.groupby(c, dropna=False):
            n = len(sub)
            if n < 5:
                continue
            rate = sub[pos_col].mean()
            lift = rate - base_rate
            # z for binomial difference approximate
            p = base_rate
            se = math.sqrt(max(p * (1 - p), 1e-9) / n)
            z = lift / se if se > 0 else np.nan
            rows.append({
                "feature": c,
                "value": str(val),
                "n": int(n),
                "positive_rate": float(rate),
                "base_rate": float(base_rate),
                "lift": float(lift),
                "z_vs_base": float(z) if np.isfinite(z) else np.nan,
                "mean_live_value": float(pd.to_numeric(sub["live_value"], errors="coerce").mean()),
                "median_live_value": float(pd.to_numeric(sub["live_value"], errors="coerce").median()),
            })
    enr = pd.DataFrame(rows)
    if not enr.empty:
        enr = enr.sort_values(["lift", "n"], ascending=[False, False])
    return enr


def numeric_contrasts(df: pd.DataFrame, pos_col: str) -> pd.DataFrame:
    nums = [
        "_learned_noleak_q", "proxy_utility", "oracle_utility", "q_proxy_gap", "q_oracle_gap",
        "beta", "delta_scale", "alpha_operator", "beta_precursor", "asa_alpha", "asa_beta",
        "base_margin", "base_rank", "base_logprob_gap_to_top", "live_rank_improvement",
        "n_delta_layers",
    ]
    rows = []
    pos = df[df[pos_col] == 1]
    neg = df[df[pos_col] == 0]
    for c in nums:
        if c not in df.columns:
            continue
        x1 = pd.to_numeric(pos[c], errors="coerce").dropna().to_numpy(float)
        x0 = pd.to_numeric(neg[c], errors="coerce").dropna().to_numpy(float)
        if len(x1) < 5 or len(x0) < 5:
            continue
        m1, m0 = np.mean(x1), np.mean(x0)
        s1, s0 = np.std(x1, ddof=1), np.std(x0, ddof=1)
        pooled = math.sqrt(((len(x1)-1)*s1*s1 + (len(x0)-1)*s0*s0) / max(len(x1)+len(x0)-2, 1))
        d = (m1 - m0) / pooled if pooled > 1e-12 else np.nan
        rows.append({
            "feature": c,
            "n_pos": int(len(x1)),
            "n_neg": int(len(x0)),
            "mean_pos": float(m1),
            "mean_neg": float(m0),
            "diff": float(m1-m0),
            "cohen_d": float(d) if np.isfinite(d) else np.nan,
            "median_pos": float(np.median(x1)),
            "median_neg": float(np.median(x0)),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("cohen_d", ascending=False)
    return out


def feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric = [
        "_learned_noleak_q", "proxy_utility", "oracle_utility", "q_proxy_gap", "q_oracle_gap",
        "beta", "delta_scale", "alpha_operator", "beta_precursor", "asa_alpha", "asa_beta",
        "base_margin", "base_rank", "base_logprob_gap_to_top",
        "n_delta_layers",
    ]
    numeric = [c for c in numeric if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().sum() > 0]
    categorical = [
        "source_experiment", "action", "_selection_family", "relation_type",
        "write_route", "target_operator", "candidate_operator", "target_concept", "candidate_concept", "pool"
    ]
    categorical = [c for c in categorical if c in df.columns]
    return numeric, categorical


def train_positive_classifier(df: pd.DataFrame, pos_col: str, out_dir: Path) -> Dict:
    work = df.copy()
    # Remove exact zero baseline rows from training? Keep them: they define no effect.
    if work[pos_col].nunique() < 2:
        return {"error": "Only one class in positive label."}
    numeric, categorical = feature_columns(work)
    X = work[numeric + categorical].copy()
    for c in numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median() if X[c].notna().any() else 0.0)
    for c in categorical:
        X[c] = X[c].astype(str).fillna("NA")
    y = work[pos_col].astype(int).to_numpy()

    groups = None
    for c in ["_group_id", "target_concept", "candidate_concept"]:
        if c in work.columns:
            groups = work[c].astype(str).to_numpy()
            break

    if groups is not None and len(np.unique(groups)) >= 2:
        k = min(5, len(np.unique(groups)))
        splits = list(GroupKFold(n_splits=k).split(X, y, groups))
    else:
        splits = list(StratifiedKFold(n_splits=min(5, np.bincount(y).min()), shuffle=True, random_state=RANDOM_SEED).split(X, y))

    prob = np.full(len(work), np.nan)
    pred = np.full(len(work), -1)
    for tr, te in splits:
        pre = ColumnTransformer(
            [
                ("num", StandardScaler(), numeric),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ],
            remainder="drop",
        )
        clf = ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X.iloc[tr], y[tr])
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        prob[te] = p
        pred[te] = (p >= 0.5).astype(int)

    mask = np.isfinite(prob)
    report = {
        "n": int(len(work)),
        "positive_rate": float(np.mean(y)),
        "features": {"numeric": numeric, "categorical": categorical},
        "cv_auc": float(roc_auc_score(y[mask], prob[mask])) if len(np.unique(y[mask])) == 2 else np.nan,
        "cv_acc": float(accuracy_score(y[mask], pred[mask])),
        "cv_f1": float(f1_score(y[mask], pred[mask], zero_division=0)),
    }

    out = work.copy()
    out["pred_positive_prob"] = prob
    out["pred_positive"] = pred
    out.to_csv(out_dir / "sem6l1_positive_classifier_predictions.csv", index=False, encoding="utf-8-sig")
    return report


def cluster_positive_rows(pos: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    if len(pos) < 20:
        return pd.DataFrame()
    numeric, categorical = feature_columns(pos)
    # Use a manageable set of categorical one-hot + numeric for clusters.
    X = pos[numeric + categorical].copy()
    for c in numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median() if X[c].notna().any() else 0.0)
    for c in categorical:
        X[c] = X[c].astype(str).fillna("NA")

    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    pre = ColumnTransformer([("num", StandardScaler(), numeric), ("cat", enc, categorical)], remainder="drop")
    Xmat = pre.fit_transform(X)
    k = min(6, max(2, len(pos) // 30))
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=20)
    labels = km.fit_predict(Xmat)
    tmp = pos.copy()
    tmp["positive_cluster"] = labels

    rows = []
    for cl, sub in tmp.groupby("positive_cluster"):
        row = {
            "cluster": int(cl),
            "n": int(len(sub)),
            "mean_live_value": float(sub["live_value"].mean()),
            "top_source": sub["source_experiment"].astype(str).mode().iloc[0],
            "top_action": sub["action"].astype(str).mode().iloc[0],
            "top_selection": sub["_selection_family"].astype(str).mode().iloc[0],
            "top_relation": sub["relation_type"].astype(str).mode().iloc[0],
            "top_target_operator": sub["target_operator"].astype(str).mode().iloc[0],
            "top_candidate_operator": sub["candidate_operator"].astype(str).mode().iloc[0],
        }
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("mean_live_value", ascending=False)
    tmp.to_csv(out_dir / "sem6l1_positive_rows_clustered.csv", index=False, encoding="utf-8-sig")
    return summary


def build_prototype_rules(enr: pd.DataFrame, num: pd.DataFrame, cluster: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rules = []
    if not enr.empty:
        # top enriched categorical conditions
        top = enr[(enr["lift"] > 0) & (enr["n"] >= 10)].head(20)
        for _, r in top.iterrows():
            rules.append({
                "rule_type": "categorical_enrichment",
                "condition": f"{r['feature']} == {r['value']}",
                "n": int(r["n"]),
                "positive_rate": float(r["positive_rate"]),
                "lift": float(r["lift"]),
                "mean_live_value": float(r["mean_live_value"]),
                "interpretation": "candidate positive-write prototype condition",
            })
    if not num.empty:
        topn = num[num["cohen_d"].abs() > 0.2].head(10)
        for _, r in topn.iterrows():
            direction = "high" if r["diff"] > 0 else "low"
            rules.append({
                "rule_type": "numeric_contrast",
                "condition": f"{direction} {r['feature']}",
                "n": int(min(r["n_pos"], r["n_neg"])),
                "positive_rate": np.nan,
                "lift": np.nan,
                "mean_live_value": np.nan,
                "interpretation": f"positive rows have {direction} {r['feature']} (d={r['cohen_d']:.3f})",
            })
    if cluster is not None and not cluster.empty:
        for _, r in cluster.head(10).iterrows():
            rules.append({
                "rule_type": "positive_cluster",
                "condition": (
                    f"cluster {r['cluster']}: source={r['top_source']}, action={r['top_action']}, "
                    f"selection={r['top_selection']}, relation={r['top_relation']}, "
                    f"target_op={r['top_target_operator']}, cand_op={r['top_candidate_operator']}"
                ),
                "n": int(r["n"]),
                "positive_rate": np.nan,
                "lift": np.nan,
                "mean_live_value": float(r["mean_live_value"]),
                "interpretation": "clustered positive prototype",
            })
    rules_df = pd.DataFrame(rules)
    rules_df.to_csv(out_dir / "sem6l1_prototype_rules.csv", index=False, encoding="utf-8-sig")
    return rules_df


def make_verdict(df: pd.DataFrame, pos: pd.DataFrame, enr: pd.DataFrame, clf_report: Dict, rules: pd.DataFrame, threshold: float) -> Dict:
    positive_rate = len(pos) / len(df) if len(df) else 0.0
    auc = clf_report.get("cv_auc", np.nan) if isinstance(clf_report, dict) else np.nan
    n_rules = len(rules) if rules is not None else 0

    pass_lite = bool(len(pos) >= 50 and n_rules >= 3)
    pass_strong = bool(pass_lite and np.isfinite(auc) and auc > 0.70)

    if pass_strong:
        verdict = "PASS_STRONG_POSITIVE_PROTOTYPES_FOUND"
    elif pass_lite:
        verdict = "PASS_LITE_POSITIVE_PROTOTYPES_FOUND"
    else:
        verdict = "FAIL_OR_INCONCLUSIVE"

    return {
        "stage": "SEM-6L.1",
        "mode": "positive_delta_prototype_mining",
        "threshold": threshold,
        "verdict": verdict,
        "pass_lite": pass_lite,
        "pass_strong": pass_strong,
        "n_all_rows": int(len(df)),
        "n_positive_rows": int(len(pos)),
        "positive_rate": float(positive_rate),
        "positive_classifier": clf_report,
        "n_prototype_rules": int(n_rules),
        "notes": [
            "6L.1 does not validate a live controller; it mines positive write prototypes.",
            "PASS means positive live deltas have structured enrichment/signature.",
            "If no strong prototypes appear, current write primitive family remains underpowered."
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--metric", default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = DEFAULT_FILES
    df = build_all(files, args.metric)
    df["is_positive"] = (df["live_value"] > args.threshold).astype(int)

    df.to_csv(out_dir / "sem6l1_all_live_rows.csv", index=False, encoding="utf-8-sig")
    pos = df[df["is_positive"] == 1].copy()
    pos.to_csv(out_dir / "sem6l1_positive_rows.csv", index=False, encoding="utf-8-sig")

    enr = categorical_enrichment(df, "is_positive", out_dir)
    enr.to_csv(out_dir / "sem6l1_positive_enrichment.csv", index=False, encoding="utf-8-sig")

    num = numeric_contrasts(df, "is_positive")
    num.to_csv(out_dir / "sem6l1_numeric_contrasts.csv", index=False, encoding="utf-8-sig")

    clf_report = train_positive_classifier(df, "is_positive", out_dir)
    with open(out_dir / "sem6l1_positive_classifier_report.json", "w", encoding="utf-8") as f:
        json.dump(clf_report, f, ensure_ascii=False, indent=2)

    cluster = cluster_positive_rows(pos, out_dir)
    cluster.to_csv(out_dir / "sem6l1_positive_cluster_summary.csv", index=False, encoding="utf-8-sig")

    rules = build_prototype_rules(enr, num, cluster, out_dir)

    verdict = make_verdict(df, pos, enr, clf_report, rules, args.threshold)
    verdict["outputs"] = {
        "all_rows": str(out_dir / "sem6l1_all_live_rows.csv"),
        "positive_rows": str(out_dir / "sem6l1_positive_rows.csv"),
        "positive_enrichment": str(out_dir / "sem6l1_positive_enrichment.csv"),
        "numeric_contrasts": str(out_dir / "sem6l1_numeric_contrasts.csv"),
        "prototype_rules": str(out_dir / "sem6l1_prototype_rules.csv"),
        "positive_cluster_summary": str(out_dir / "sem6l1_positive_cluster_summary.csv"),
        "classifier_report": str(out_dir / "sem6l1_positive_classifier_report.json"),
    }
    with open(out_dir / "sem6l1_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)

    print("\n========== SEM-6L.1 ==========")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    print("\nTop enrichment:")
    print(enr.head(20).to_string(index=False))
    print("\nNumeric contrasts:")
    print(num.head(15).to_string(index=False))
    print("\nPrototype rules:")
    print(rules.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
