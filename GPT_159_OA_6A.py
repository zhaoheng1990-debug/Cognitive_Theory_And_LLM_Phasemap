# -*- coding: utf-8 -*-
"""
OA-6A: Direction Spectrum Evolution Audit
=========================================

Core question
-------------
After OA-5D, we have:

    W provides shared local neighborhood.
    Prompt injects direction-spectrum bias.
    Observed TopK center geometry is prompt-conditioned local geometry.

OA-6A asks:

    Does the prompt-conditioned direction spectrum evolve layer by layer?

More concretely:

    Prompt -> Sigma_0
    Sigma_0 -> Sigma_l
    Direction dominance grows or reorganizes across layers.

Hypothesis
----------
For the same topic, different task prompts initially activate different directions
inside a shared local neighborhood. During layer propagation, one or more
directions become dominant.

This experiment measures:

    1. Task-direction separability per layer.
    2. Topic-conditioned direction dominance per layer.
    3. PCA/eigen spectrum of task-centroid geometry per layer.
    4. Whether shallow direction spectrum predicts later direction dominance.
    5. Whether there is a transition band analogous to tau.

Input
-----
Default:
    C:\\Users\\ZH\\Desktop\\AGI\\python_script\\oa5c1_outputs\\oa5c1_topk_center_records.csv

Required columns:
    prompt_id, topic, task, paraphrase_id, layer_index, k
    center_0 ... center_D

Outputs
-------
oa6a_outputs/
    oa6a_layer_spectrum_summary.csv
    oa6a_topic_task_direction_summary.csv
    oa6a_shallow_to_late_prediction.csv
    oa6a_tau_candidate_summary.csv
    oa6a_verdict_summary.csv
"""

import json
import math
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.metrics import r2_score, accuracy_score, f1_score

warnings.filterwarnings("ignore")

# ============================================================
# PATH CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\ZH\Desktop\AGI\python_script")
INPUT_FILE = BASE_DIR / "oa5c1_outputs" / "oa5c1_topk_center_records.csv"

OUTPUT_DIR = BASE_DIR / "oa6a_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_LAYER = OUTPUT_DIR / "oa6a_layer_spectrum_summary.csv"
OUT_DIRECTION = OUTPUT_DIR / "oa6a_topic_task_direction_summary.csv"
OUT_PRED = OUTPUT_DIR / "oa6a_shallow_to_late_prediction.csv"
OUT_TAU = OUTPUT_DIR / "oa6a_tau_candidate_summary.csv"
OUT_VERDICT = OUTPUT_DIR / "oa6a_verdict_summary.csv"

RANDOM_SEED = 42

# Layer windows. Adjust if model has different depth.
SHALLOW_LAYERS = list(range(0, 7))
MID_LAYERS = list(range(7, 20))
CRIT_LAYERS = list(range(20, 23))
LATE_LAYERS = list(range(23, 27))

# ============================================================
# HELPERS
# ============================================================

def infer_center_cols(df):
    center_cols = [c for c in df.columns if c.startswith("center_")]
    if not center_cols:
        raise ValueError("No center_* columns found.")
    return sorted(center_cols, key=lambda x: int(x.split("_")[1]))


def normalize_rows(X):
    X = np.asarray(X, dtype=np.float64)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def cosine_sim(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


def safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def eig_spectrum(X):
    """
    Return eigen-spectrum of row vectors after centering.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] < 3:
        return {
            "pc1_var": np.nan,
            "pc2_var": np.nan,
            "pc3_var": np.nan,
            "pc12_var": np.nan,
            "pc123_var": np.nan,
            "dominance_ratio": np.nan,
            "effective_rank": np.nan,
        }

    Xc = X - X.mean(axis=0, keepdims=True)
    try:
        pca = PCA(n_components=min(5, Xc.shape[0], Xc.shape[1]), random_state=RANDOM_SEED)
        pca.fit(Xc)
        ev = pca.explained_variance_ratio_
    except Exception:
        return {
            "pc1_var": np.nan,
            "pc2_var": np.nan,
            "pc3_var": np.nan,
            "pc12_var": np.nan,
            "pc123_var": np.nan,
            "dominance_ratio": np.nan,
            "effective_rank": np.nan,
        }

    pc1 = float(ev[0]) if len(ev) > 0 else np.nan
    pc2 = float(ev[1]) if len(ev) > 1 else 0.0
    pc3 = float(ev[2]) if len(ev) > 2 else 0.0
    pc12 = pc1 + pc2
    pc123 = pc12 + pc3

    # dominance ratio: how much first direction dominates second direction
    dom = pc1 / (pc2 + 1e-12) if pc2 > 0 else np.inf

    # effective rank of spectrum
    p = ev / (ev.sum() + 1e-12)
    ent = -np.sum(p * np.log(p + 1e-12))
    erank = float(np.exp(ent))

    return {
        "pc1_var": pc1,
        "pc2_var": pc2,
        "pc3_var": pc3,
        "pc12_var": pc12,
        "pc123_var": pc123,
        "dominance_ratio": float(dom),
        "effective_rank": erank,
    }


def aggregate_prompt_layer(df, center_cols, k):
    sub = df[df["k"] == k].copy()
    rows = []
    X = []

    for (pid, layer), g in sub.groupby(["prompt_id", "layer_index"]):
        first = g.iloc[0]
        rows.append({
            "prompt_id": str(pid),
            "topic": str(first["topic"]),
            "task": str(first["task"]),
            "paraphrase_id": str(first["paraphrase_id"]),
            "layer_index": int(layer),
            "k": int(k),
            "spread": float(g["spread"].mean()) if "spread" in g.columns else np.nan,
            "logit_gap": float(g["logit_gap"].mean()) if "logit_gap" in g.columns else np.nan,
            "logit_std": float(g["logit_std"].mean()) if "logit_std" in g.columns else np.nan,
        })
        X.append(g[center_cols].to_numpy(dtype=np.float32).mean(axis=0))

    meta = pd.DataFrame(rows)
    X = np.stack(X).astype(np.float32)
    return meta, X


def build_layer_matrix(meta, X, layer):
    idx = meta.index[meta["layer_index"] == layer].to_numpy()
    return meta.loc[idx].reset_index(drop=True), X[idx]


def classify_task_leave_topic(meta_l, X_l):
    y = meta_l["task"].astype(str).values
    groups = meta_l["topic"].astype(str).values
    if len(np.unique(y)) < 2 or len(np.unique(groups)) < 2:
        return {"acc": np.nan, "f1": np.nan}

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ])

    accs, f1s = [], []
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))

    for tr, te in gkf.split(X_l, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        try:
            pipe.fit(X_l[tr], y[tr])
            pred = pipe.predict(X_l[te])
            accs.append(accuracy_score(y[te], pred))
            f1s.append(f1_score(y[te], pred, average="macro", zero_division=0))
        except Exception:
            pass

    return {
        "acc": float(np.mean(accs)) if accs else np.nan,
        "f1": float(np.mean(f1s)) if f1s else np.nan,
    }


# ============================================================
# ANALYSIS 1: LAYER SPECTRUM
# ============================================================

def layer_spectrum_analysis(df, center_cols):
    rows = []

    for k in sorted(df["k"].unique()):
        meta, X = aggregate_prompt_layer(df, center_cols, k)

        for layer in sorted(meta["layer_index"].unique()):
            ml, Xl = build_layer_matrix(meta, X, layer)

            spec_all = eig_spectrum(StandardScaler().fit_transform(Xl))
            clf = classify_task_leave_topic(ml, Xl)

            row = {
                "k": int(k),
                "layer_index": int(layer),
                "n_prompts": int(len(ml)),
                "n_topics": int(ml["topic"].nunique()),
                "n_tasks": int(ml["task"].nunique()),
                "task_acc_leave_topic": clf["acc"],
                "task_f1_leave_topic": clf["f1"],
                **{f"global_{kk}": vv for kk, vv in spec_all.items()},
            }

            # Topic-conditioned task centroid spectrum
            topic_specs = []
            for topic, gt in ml.groupby("topic"):
                task_centers = []
                for task, gtt in gt.groupby("task"):
                    idx = gtt.index.to_numpy()
                    task_centers.append(Xl[idx].mean(axis=0))
                if len(task_centers) >= 3:
                    topic_specs.append(eig_spectrum(np.stack(task_centers)))

            for key in ["pc1_var", "pc2_var", "pc3_var", "dominance_ratio", "effective_rank"]:
                vals = [s[key] for s in topic_specs if np.isfinite(s[key])]
                row[f"topic_task_{key}_mean"] = float(np.mean(vals)) if vals else np.nan
                row[f"topic_task_{key}_std"] = float(np.std(vals)) if vals else np.nan

            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# ANALYSIS 2: TOPIC-TASK DIRECTION FIELD
# ============================================================

def topic_task_direction_analysis(df, center_cols):
    rows = []

    for k in sorted(df["k"].unique()):
        meta, X = aggregate_prompt_layer(df, center_cols, k)

        for layer in sorted(meta["layer_index"].unique()):
            ml, Xl = build_layer_matrix(meta, X, layer)

            for topic, gt in ml.groupby("topic"):
                topic_idx = gt.index.to_numpy()
                topic_center = Xl[topic_idx].mean(axis=0)

                task_dirs = {}
                for task, gtt in gt.groupby("task"):
                    idx = gtt.index.to_numpy()
                    task_center = Xl[idx].mean(axis=0)
                    v = task_center - topic_center
                    task_dirs[str(task)] = v

                tasks = sorted(task_dirs.keys())

                # Pairwise task direction angular separation
                pair_cos = []
                pair_angle = []
                for i, ti in enumerate(tasks):
                    for tj in tasks[i + 1:]:
                        c = cosine_sim(task_dirs[ti], task_dirs[tj])
                        pair_cos.append(c)
                        pair_angle.append(1.0 - c)

                norms = [float(np.linalg.norm(task_dirs[t])) for t in tasks]

                if len(task_dirs) >= 3:
                    spec = eig_spectrum(np.stack([task_dirs[t] for t in tasks]))
                else:
                    spec = {k2: np.nan for k2 in [
                        "pc1_var", "pc2_var", "pc3_var",
                        "pc12_var", "pc123_var",
                        "dominance_ratio", "effective_rank"
                    ]}

                rows.append({
                    "k": int(k),
                    "layer_index": int(layer),
                    "topic": str(topic),
                    "n_tasks": int(len(tasks)),
                    "mean_task_dir_norm": float(np.mean(norms)) if norms else np.nan,
                    "std_task_dir_norm": float(np.std(norms)) if norms else np.nan,
                    "mean_pair_direction_cos": float(np.mean(pair_cos)) if pair_cos else np.nan,
                    "mean_pair_direction_angle": float(np.mean(pair_angle)) if pair_angle else np.nan,
                    **spec,
                })

    return pd.DataFrame(rows)


# ============================================================
# ANALYSIS 3: SHALLOW -> LATE PREDICTION
# ============================================================

def summarize_window(layer_df, layers, prefix):
    sub = layer_df[layer_df["layer_index"].isin(layers)]
    g = sub.groupby("k")
    rows = []
    for k, gg in g:
        row = {"k": int(k)}
        for col in [
            "task_acc_leave_topic",
            "task_f1_leave_topic",
            "global_pc1_var",
            "global_dominance_ratio",
            "global_effective_rank",
            "topic_task_pc1_var_mean",
            "topic_task_dominance_ratio_mean",
            "topic_task_effective_rank_mean",
        ]:
            row[f"{prefix}_{col}"] = float(gg[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def shallow_to_late_prediction(layer_df, direction_df):
    rows = []

    for k in sorted(layer_df["k"].unique()):
        ldf = layer_df[layer_df["k"] == k].copy()

        # build layer-level sequence features
        shallow = ldf[ldf["layer_index"].isin(SHALLOW_LAYERS)]
        mid = ldf[ldf["layer_index"].isin(MID_LAYERS)]
        crit = ldf[ldf["layer_index"].isin(CRIT_LAYERS)]
        late = ldf[ldf["layer_index"].isin(LATE_LAYERS)]

        if shallow.empty or crit.empty or late.empty:
            continue

        def m(df_, col):
            return float(df_[col].mean()) if len(df_) else np.nan

        row = {
            "k": int(k),

            "shallow_task_f1": m(shallow, "task_f1_leave_topic"),
            "mid_task_f1": m(mid, "task_f1_leave_topic"),
            "crit_task_f1": m(crit, "task_f1_leave_topic"),
            "late_task_f1": m(late, "task_f1_leave_topic"),

            "shallow_pc1": m(shallow, "topic_task_pc1_var_mean"),
            "mid_pc1": m(mid, "topic_task_pc1_var_mean"),
            "crit_pc1": m(crit, "topic_task_pc1_var_mean"),
            "late_pc1": m(late, "topic_task_pc1_var_mean"),

            "shallow_dom": m(shallow, "topic_task_dominance_ratio_mean"),
            "mid_dom": m(mid, "topic_task_dominance_ratio_mean"),
            "crit_dom": m(crit, "topic_task_dominance_ratio_mean"),
            "late_dom": m(late, "topic_task_dominance_ratio_mean"),

            "shallow_erank": m(shallow, "topic_task_effective_rank_mean"),
            "mid_erank": m(mid, "topic_task_effective_rank_mean"),
            "crit_erank": m(crit, "topic_task_effective_rank_mean"),
            "late_erank": m(late, "topic_task_effective_rank_mean"),

            "pc1_gain_shallow_to_crit": m(crit, "topic_task_pc1_var_mean") - m(shallow, "topic_task_pc1_var_mean"),
            "pc1_gain_shallow_to_late": m(late, "topic_task_pc1_var_mean") - m(shallow, "topic_task_pc1_var_mean"),
            "dom_gain_shallow_to_crit": m(crit, "topic_task_dominance_ratio_mean") - m(shallow, "topic_task_dominance_ratio_mean"),
            "dom_gain_shallow_to_late": m(late, "topic_task_dominance_ratio_mean") - m(shallow, "topic_task_dominance_ratio_mean"),
            "erank_drop_shallow_to_late": m(shallow, "topic_task_effective_rank_mean") - m(late, "topic_task_effective_rank_mean"),
        }

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# ANALYSIS 4: TAU CANDIDATE
# ============================================================

def tau_candidate_analysis(layer_df):
    rows = []

    for k in sorted(layer_df["k"].unique()):
        ldf = layer_df[layer_df["k"] == k].sort_values("layer_index").copy()

        # Use smoothed dominance / pc1 / task_f1 as observable.
        for obs in [
            "topic_task_pc1_var_mean",
            "topic_task_dominance_ratio_mean",
            "task_f1_leave_topic",
            "global_pc1_var",
        ]:
            vals = ldf[obs].to_numpy(dtype=np.float64)
            layers = ldf["layer_index"].to_numpy(dtype=int)
            if len(vals) < 5 or np.all(~np.isfinite(vals)):
                continue

            vals = pd.Series(vals).interpolate(limit_direction="both").to_numpy()
            dvals = np.diff(vals)

            # Candidate tau = largest positive derivative in L15-L24
            mask = (layers[:-1] >= 15) & (layers[:-1] <= 24)
            if mask.sum() == 0:
                tau = np.nan
                max_jump = np.nan
            else:
                local_idx = np.where(mask)[0]
                best = local_idx[np.argmax(np.abs(dvals[local_idx]))]
                tau = int(layers[best])
                max_jump = float(dvals[best])

            rows.append({
                "k": int(k),
                "observable": obs,
                "tau_candidate_layer": tau,
                "max_abs_jump": abs(max_jump) if np.isfinite(max_jump) else np.nan,
                "signed_jump": max_jump,
                "in_L20_L22": bool(tau in [20, 21, 22]) if np.isfinite(tau) else False,
                "value_L0_6_mean": float(ldf[ldf["layer_index"].isin(SHALLOW_LAYERS)][obs].mean()),
                "value_L7_19_mean": float(ldf[ldf["layer_index"].isin(MID_LAYERS)][obs].mean()),
                "value_L20_22_mean": float(ldf[ldf["layer_index"].isin(CRIT_LAYERS)][obs].mean()),
                "value_L23_26_mean": float(ldf[ldf["layer_index"].isin(LATE_LAYERS)][obs].mean()),
            })

    return pd.DataFrame(rows)


# ============================================================
# VERDICT
# ============================================================

def verdict_summary(layer_df, pred_df, tau_df):
    rows = []

    for k in sorted(layer_df["k"].unique()):
        ldf = layer_df[layer_df["k"] == k]
        pdf = pred_df[pred_df["k"] == k]
        tdf = tau_df[tau_df["k"] == k]

        if pdf.empty:
            continue

        p = pdf.iloc[0]

        # Core evidence flags
        task_separable = (
            np.isfinite(p["shallow_task_f1"]) and p["shallow_task_f1"] > 0.45
        ) or (
            np.isfinite(p["mid_task_f1"]) and p["mid_task_f1"] > 0.50
        )

        pc1_growth = (
            np.isfinite(p["pc1_gain_shallow_to_late"]) and
            p["pc1_gain_shallow_to_late"] > 0.03
        )

        dominance_growth = (
            np.isfinite(p["dom_gain_shallow_to_late"]) and
            p["dom_gain_shallow_to_late"] > 0.10
        )

        erank_collapse = (
            np.isfinite(p["erank_drop_shallow_to_late"]) and
            p["erank_drop_shallow_to_late"] > 0.05
        )

        tau_support = bool(tdf["in_L20_L22"].mean() >= 0.25) if len(tdf) else False

        score = sum([
            task_separable,
            pc1_growth,
            dominance_growth,
            erank_collapse,
            tau_support,
        ])

        if score >= 4:
            verdict = "PASS-Strong: direction spectrum evolves into dominant geometry"
        elif score == 3:
            verdict = "PASS-Moderate: partial direction-spectrum evolution"
        elif score == 2:
            verdict = "PASS-Lite/Mixed"
        else:
            verdict = "FAIL/Mixed"

        rows.append({
            "k": int(k),
            "task_separable": bool(task_separable),
            "pc1_growth": bool(pc1_growth),
            "dominance_growth": bool(dominance_growth),
            "erank_collapse": bool(erank_collapse),
            "tau_support": bool(tau_support),
            "score": int(score),
            "verdict": verdict,

            "shallow_task_f1": p["shallow_task_f1"],
            "mid_task_f1": p["mid_task_f1"],
            "crit_task_f1": p["crit_task_f1"],
            "late_task_f1": p["late_task_f1"],

            "shallow_pc1": p["shallow_pc1"],
            "crit_pc1": p["crit_pc1"],
            "late_pc1": p["late_pc1"],

            "pc1_gain_shallow_to_late": p["pc1_gain_shallow_to_late"],
            "dom_gain_shallow_to_late": p["dom_gain_shallow_to_late"],
            "erank_drop_shallow_to_late": p["erank_drop_shallow_to_late"],

            "tau_L20_L22_frac": float(tdf["in_L20_L22"].mean()) if len(tdf) else np.nan,
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("OA-6A Direction Spectrum Evolution Audit")
    print("=" * 80)

    print("Input:", INPUT_FILE)
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    center_cols = infer_center_cols(df)

    print("Rows:", len(df))
    print("Center dim:", len(center_cols))
    print("Topics:", sorted(df["topic"].unique().tolist()))
    print("Tasks:", sorted(df["task"].unique().tolist()))
    print("K:", sorted(df["k"].unique().tolist()))
    print("Layers:", sorted(df["layer_index"].unique().tolist()))

    print("\n[1/5] Layer spectrum analysis...")
    layer_df = layer_spectrum_analysis(df, center_cols)
    layer_df.to_csv(OUT_LAYER, index=False, encoding="utf-8-sig")
    print("Saved:", OUT_LAYER)

    print("\n[2/5] Topic-task direction field analysis...")
    direction_df = topic_task_direction_analysis(df, center_cols)
    direction_df.to_csv(OUT_DIRECTION, index=False, encoding="utf-8-sig")
    print("Saved:", OUT_DIRECTION)

    print("\n[3/5] Shallow-to-late summary...")
    pred_df = shallow_to_late_prediction(layer_df, direction_df)
    pred_df.to_csv(OUT_PRED, index=False, encoding="utf-8-sig")
    print("Saved:", OUT_PRED)

    print("\n[4/5] Tau candidate analysis...")
    tau_df = tau_candidate_analysis(layer_df)
    tau_df.to_csv(OUT_TAU, index=False, encoding="utf-8-sig")
    print("Saved:", OUT_TAU)

    print("\n[5/5] Verdict...")
    verdict_df = verdict_summary(layer_df, pred_df, tau_df)
    verdict_df.to_csv(OUT_VERDICT, index=False, encoding="utf-8-sig")
    print("Saved:", OUT_VERDICT)

    print("\n=== OA-6A VERDICT ===")
    print(verdict_df.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
