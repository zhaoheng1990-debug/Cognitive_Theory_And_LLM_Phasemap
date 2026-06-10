# -*- coding: utf-8 -*-
"""
GPT_215_SEM_4C_latent_seed_window_localization.py

SEM-4C: Latent Seed Window Localization
---------------------------------------

Goal:
    Localize which layer window acts as the LatentSeed bridge between
    Prompt_text and trajectory center μF.

Background:
    SEM-4B showed:
        LatentSeed > TextSeed for SeedFamily recovery under leave-surface-group CV.
    But SEM-4B's best block was full center_pca, which may include non-init layers.

    SEM-4C asks:
        Is the bridge already present in shallow init windows?
        Or does it require later trajectory geometry?

Core questions:
    Q1. Which layer window gives best SeedFamily recovery?
    Q2. Which window best retrieves μF family centers?
    Q3. Is L0-L6 enough, or is full trajectory center_pca necessary?
    Q4. Does window performance follow model-theory expectations:
        shallow init -> latent seed,
        mid/decision -> trajectory seed?

Inputs:
    sem3a_outputs/
      sem3a_dataset.csv
      sem3a_features.csv

Outputs:
    sem4c_outputs/
      sem4c_config.json
      sem4c_window_inventory.csv
      sem4c_seedfamily_window_cv.csv
      sem4c_muF_window_retrieval.csv
      sem4c_window_rank_summary.csv
      sem4c_results_summary.json

Run:
    python GPT_215_SEM_4C_latent_seed_window_localization.py
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIG
# ============================================================

@dataclass
class CFG:
    SEM3A_DIR: str = "./sem3a_outputs"
    OUTPUT_DIR: str = "./sem4c_outputs"

    SEM3A_DATASET: str = "sem3a_dataset.csv"
    SEM3A_FEATURES: str = "sem3a_features.csv"

    RANDOM_SEED: int = 42
    N_SPLITS: int = 4

    # Core layer windows.
    WINDOWS: Tuple[str, ...] = (
        "L0_1",
        "L0_2",
        "L0_3",
        "L0_4",
        "L0_5",
        "L0_6",
        "L2_5",
        "L3_6",
        "L5_6",
        "L7_8",
        "L7_11",
        "L9_11",
        "L7_19",
        "L20_22",
        "L20_25",
        "L23_25",
        "L0_25",
    )

    FEATURE_FAMILIES: Tuple[str, ...] = (
        "center_pca",
        "scalar",
        "transport",
        "all",
    )

    MAIN_INIT_WINDOW: str = "L0_6"
    MAIN_FULL_WINDOW: str = "L0_25"


cfg = CFG()


# ============================================================
# UTILS
# ============================================================

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
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def safe_macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


# ============================================================
# LOAD
# ============================================================

def load_inputs(cfg: CFG):
    sem3a = Path(cfg.SEM3A_DIR)
    dataset = pd.read_csv(sem3a / cfg.SEM3A_DATASET)
    features = pd.read_csv(sem3a / cfg.SEM3A_FEATURES)

    if "prompt" not in features.columns:
        features = dataset.merge(features, on="row_id", how="left")
    return dataset, features


# ============================================================
# FEATURE WINDOWS
# ============================================================

META_COLS = {
    "row_id", "row_type", "seed_family", "concept_star", "operator_star",
    "surface_id", "prompt", "operator_id", "concept_family", "operator_family",
    "question_form", "specificity_level", "surface_form", "domain_frame",
    "constraint_polarity", "expected_prompt_function"
}


def parse_window(name: str):
    # format "L0_6"
    a, b = name.replace("L", "").split("_")
    return int(a), int(b)


def numeric_cols(df: pd.DataFrame):
    return [c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]


def layer_in_col(c: str, lo: int, hi: int) -> bool:
    return any(f"_L{i}_" in c for i in range(lo, hi + 1))


def get_cols(df: pd.DataFrame, window: str, family: str):
    lo, hi = parse_window(window)
    num = numeric_cols(df)
    cols = [c for c in num if layer_in_col(c, lo, hi)]

    if family == "center_pca":
        cols = [c for c in cols if "centerPC" in c]
    elif family == "scalar":
        cols = [c for c in cols if "centerPC" not in c]
    elif family == "transport":
        cols = [c for c in cols if any(k in c for k in ["shift", "jaccard"])]
    elif family == "all":
        pass
    else:
        raise ValueError(family)
    return cols


def make_X(df: pd.DataFrame, cols: List[str]):
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    return df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32).values


# ============================================================
# CV
# ============================================================

def cv_classify(X, y, groups=None, mode="stratified", random_seed=42):
    y = np.asarray(y)
    if X.shape[1] == 0:
        return {"acc": np.nan, "macro_f1": np.nan, "n": len(y)}

    if mode == "stratified":
        labels = pd.factorize(y)[0]
        counts = np.bincount(labels)
        n_splits = min(cfg.N_SPLITS, counts.min())
        if n_splits < 2:
            return {"acc": np.nan, "macro_f1": np.nan, "n": len(y)}
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        splits = splitter.split(X, y)
    elif mode == "group":
        if groups is None or len(np.unique(groups)) < 2:
            return {"acc": np.nan, "macro_f1": np.nan, "n": len(y)}
        splitter = GroupKFold(n_splits=min(cfg.N_SPLITS, len(np.unique(groups))))
        splits = splitter.split(X, y, groups=groups)
    else:
        raise ValueError(mode)

    trues, preds = [], []
    for tr, te in splits:
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        try:
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")
            )
            clf.fit(Xtr, ytr)
            yp = clf.predict(Xte)
        except Exception:
            clf = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
            clf.fit(Xtr, ytr)
            yp = clf.predict(Xte)
        trues.extend(yte)
        preds.extend(yp)

    return {
        "acc": float(accuracy_score(trues, preds)),
        "macro_f1": float(safe_macro_f1(trues, preds)),
        "n": int(len(trues)),
    }


def run_window_cv(features: pd.DataFrame, cfg: CFG):
    df = features[features["row_type"] == "original_prompt"].copy().reset_index(drop=True)
    y = df["seed_family"].astype(str).values
    groups = df["surface_id"].astype(str).values if "surface_id" in df.columns else None

    rows = []
    for window in cfg.WINDOWS:
        for family in cfg.FEATURE_FAMILIES:
            cols = get_cols(df, window, family)
            X = make_X(df, cols)
            if X.shape[1] == 0:
                rows.append({
                    "window": window,
                    "feature_family": family,
                    "cv": "NA",
                    "n_features": 0,
                    "acc": np.nan,
                    "macro_f1": np.nan,
                    "n": len(df),
                })
                continue
            for mode, grp in [("stratified", None), ("leave_surface_group", groups)]:
                out = cv_classify(X, y, groups=grp, mode="group" if mode == "leave_surface_group" else "stratified", random_seed=cfg.RANDOM_SEED)
                rows.append({
                    "window": window,
                    "feature_family": family,
                    "cv": mode,
                    "n_features": X.shape[1],
                    **out,
                })
    return pd.DataFrame(rows)


# ============================================================
# μF RETRIEVAL
# ============================================================

def run_muF_retrieval(features: pd.DataFrame, cfg: CFG):
    all_df = features.reset_index(drop=True)
    original_mask = (all_df["row_type"] == "original_prompt").values
    orig_indices = np.where(original_mask)[0]

    rows = []
    for window in cfg.WINDOWS:
        for family in cfg.FEATURE_FAMILIES:
            cols = get_cols(all_df, window, family)
            X = make_X(all_df, cols)
            if X.shape[1] == 0:
                rows.append({
                    "window": window,
                    "feature_family": family,
                    "target": "muF_family_center",
                    "n_features": 0,
                    "nearest_family_acc": np.nan,
                    "frac_true_gt_random_mean": np.nan,
                    "mean_margin_vs_random_mean": np.nan,
                    "n": int(len(orig_indices)),
                })
                continue

            scaler = StandardScaler()
            Xz = scaler.fit_transform(X)

            centers = {}
            for fam in sorted(all_df["seed_family"].unique()):
                idx = np.where((all_df["seed_family"].values == fam) & original_mask)[0]
                if len(idx):
                    centers[fam] = Xz[idx].mean(axis=0)

            fams = list(centers.keys())
            C = np.vstack([centers[f] for f in fams])

            correct = []
            gt_mean = []
            margins = []
            for idx in orig_indices:
                x = Xz[idx]
                true_fam = all_df.iloc[idx]["seed_family"]
                sims = np.array([cosine_np(x, c) for c in C])
                pred = fams[int(np.argmax(sims))]
                correct.append(int(pred == true_fam))

                true_sim = sims[fams.index(true_fam)]
                other = np.array([s for f, s in zip(fams, sims) if f != true_fam])
                gt_mean.append(float(true_sim > other.mean()))
                margins.append(float(true_sim - other.mean()))

            rows.append({
                "window": window,
                "feature_family": family,
                "target": "muF_family_center",
                "n_features": X.shape[1],
                "nearest_family_acc": float(np.mean(correct)),
                "frac_true_gt_random_mean": float(np.mean(gt_mean)),
                "mean_margin_vs_random_mean": float(np.mean(margins)),
                "n": int(len(orig_indices)),
            })
    return pd.DataFrame(rows)


# ============================================================
# INVENTORY / RANKING / SUMMARY
# ============================================================

def build_inventory(features: pd.DataFrame, cfg: CFG):
    rows = []
    for window in cfg.WINDOWS:
        for family in cfg.FEATURE_FAMILIES:
            cols = get_cols(features, window, family)
            rows.append({
                "window": window,
                "feature_family": family,
                "n_features": len(cols),
                "example_cols": json.dumps(cols[:8], ensure_ascii=False),
            })
    return pd.DataFrame(rows)


def build_rank_summary(cv_df: pd.DataFrame, retr_df: pd.DataFrame):
    rows = []

    if len(cv_df):
        for cv in ["stratified", "leave_surface_group"]:
            sub = cv_df[cv_df["cv"] == cv].dropna(subset=["macro_f1"]).copy()
            if len(sub):
                sub["rank_macro_f1"] = sub["macro_f1"].rank(ascending=False, method="min")
                for _, r in sub.sort_values("rank_macro_f1").head(20).iterrows():
                    rows.append({
                        "table": "seedfamily_cv",
                        "metric": "macro_f1",
                        "cv_or_target": cv,
                        "window": r["window"],
                        "feature_family": r["feature_family"],
                        "score": r["macro_f1"],
                        "acc": r["acc"],
                        "n_features": r["n_features"],
                        "rank": int(r["rank_macro_f1"]),
                    })

    if len(retr_df):
        sub = retr_df.dropna(subset=["nearest_family_acc"]).copy()
        if len(sub):
            sub["rank_retrieval"] = sub["nearest_family_acc"].rank(ascending=False, method="min")
            for _, r in sub.sort_values(["rank_retrieval", "mean_margin_vs_random_mean"], ascending=[True, False]).head(20).iterrows():
                rows.append({
                    "table": "muF_retrieval",
                    "metric": "nearest_family_acc",
                    "cv_or_target": "muF_family_center",
                    "window": r["window"],
                    "feature_family": r["feature_family"],
                    "score": r["nearest_family_acc"],
                    "acc": r["nearest_family_acc"],
                    "n_features": r["n_features"],
                    "rank": int(r["rank_retrieval"]),
                    "mean_margin_vs_random_mean": r["mean_margin_vs_random_mean"],
                })
    return pd.DataFrame(rows)


def parse_window_end(window: str):
    lo, hi = parse_window(window)
    return hi


def build_results_summary(cv_df, retr_df, rank_df):
    summary = {
        "config": asdict(cfg),
        "verdict": "UNDETERMINED",
        "best": {},
        "diagnosis": "",
        "window_interpretation": {},
        "interpretation": [],
    }

    # Best leave-surface CV as main robustness criterion
    robust = cv_df[(cv_df["cv"] == "leave_surface_group") & cv_df["macro_f1"].notna()].copy()
    if len(robust):
        best_cv = robust.sort_values("macro_f1", ascending=False).iloc[0].to_dict()
        summary["best"]["seedfamily_leave_surface"] = best_cv

        init = robust[robust["window"].isin(["L0_1", "L0_2", "L0_3", "L0_4", "L0_5", "L0_6", "L2_5", "L3_6", "L5_6"])]
        if len(init):
            best_init = init.sort_values("macro_f1", ascending=False).iloc[0].to_dict()
            summary["best"]["best_init_window"] = best_init

        mid = robust[robust["window"].isin(["L7_8", "L7_11", "L9_11", "L7_19"])]
        if len(mid):
            best_mid = mid.sort_values("macro_f1", ascending=False).iloc[0].to_dict()
            summary["best"]["best_mid_window"] = best_mid

        decision = robust[robust["window"].isin(["L20_22", "L20_25", "L23_25"])]
        if len(decision):
            best_dec = decision.sort_values("macro_f1", ascending=False).iloc[0].to_dict()
            summary["best"]["best_decision_window"] = best_dec

    if len(retr_df):
        best_retr = retr_df.dropna(subset=["nearest_family_acc"]).sort_values(
            ["nearest_family_acc", "mean_margin_vs_random_mean"],
            ascending=[False, False]
        ).iloc[0].to_dict()
        summary["best"]["muF_retrieval"] = best_retr

        init_retr = retr_df[retr_df["window"].isin(["L0_1", "L0_2", "L0_3", "L0_4", "L0_5", "L0_6", "L2_5", "L3_6", "L5_6"])].dropna(subset=["nearest_family_acc"])
        if len(init_retr):
            summary["best"]["muF_retrieval_init"] = init_retr.sort_values(
                ["nearest_family_acc", "mean_margin_vs_random_mean"],
                ascending=[False, False]
            ).iloc[0].to_dict()

    # Diagnosis
    best_all = summary["best"].get("seedfamily_leave_surface", {})
    best_init = summary["best"].get("best_init_window", {})
    best_mid = summary["best"].get("best_mid_window", {})
    best_dec = summary["best"].get("best_decision_window", {})

    all_f1 = best_all.get("macro_f1", np.nan)
    init_f1 = best_init.get("macro_f1", np.nan)
    mid_f1 = best_mid.get("macro_f1", np.nan)
    dec_f1 = best_dec.get("macro_f1", np.nan)

    if np.isfinite(init_f1) and np.isfinite(all_f1) and init_f1 >= all_f1 - 0.05 and init_f1 >= 0.80:
        summary["diagnosis"] = "INIT_WINDOW_SUFFICIENT_FOR_LATENT_SEED"
    elif np.isfinite(mid_f1) and mid_f1 > np.nan_to_num(init_f1) + 0.05:
        summary["diagnosis"] = "MID_TRAJECTORY_GEOMETRY_REQUIRED"
    elif np.isfinite(dec_f1) and dec_f1 > max(np.nan_to_num(init_f1), np.nan_to_num(mid_f1)) + 0.05:
        summary["diagnosis"] = "DECISION_TRAJECTORY_GEOMETRY_REQUIRED"
    elif np.isfinite(init_f1) and init_f1 >= 0.65:
        summary["diagnosis"] = "INIT_WINDOW_PARTIAL_BRIDGE"
    else:
        summary["diagnosis"] = "NO_CLEAR_LATENT_SEED_WINDOW"

    # Verdict
    if summary["diagnosis"] == "INIT_WINDOW_SUFFICIENT_FOR_LATENT_SEED":
        summary["verdict"] = "PASS_STRONG_INIT_LATENT_SEED"
    elif summary["diagnosis"] == "INIT_WINDOW_PARTIAL_BRIDGE":
        summary["verdict"] = "PASS_LITE_INIT_LATENT_SEED"
    elif summary["diagnosis"] in ["MID_TRAJECTORY_GEOMETRY_REQUIRED", "DECISION_TRAJECTORY_GEOMETRY_REQUIRED"]:
        summary["verdict"] = "PASS_LITE_LATENT_SEED_EXISTS_BUT_NOT_INIT_ONLY"
    elif np.isfinite(all_f1) and all_f1 >= 0.65:
        summary["verdict"] = "PARTIAL_PASS_LATENT_SEED_WINDOW_SIGNAL"
    else:
        summary["verdict"] = "NO_PASS_YET"

    summary["window_interpretation"] = {
        "init_windows": "candidate text->latent seed initialization bridge",
        "mid_windows": "candidate trajectory-shaping / operator interaction bridge",
        "decision_windows": "trajectory seed / μF identity may already be decision geometry, not init seed",
        "full_window": "upper bound; may leak full trajectory identity and should not be equated with initialization",
    }

    summary["interpretation"] = [
        "SEM-4C localizes which layer window is sufficient for latent seed recovery.",
        "If init windows approach full-window performance, latent seed is an initialization object.",
        "If only mid/decision/full windows perform strongly, μF is closer to trajectory seed than shallow seed.",
        "This avoids the failed μF->Prompt* regeneration route by locating the internal bridge directly.",
    ]
    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_dir(cfg.OUTPUT_DIR)
    outdir = Path(cfg.OUTPUT_DIR)
    json_dump(asdict(cfg), outdir / "sem4c_config.json")

    dataset, features = load_inputs(cfg)

    inventory = build_inventory(features, cfg)
    inventory.to_csv(outdir / "sem4c_window_inventory.csv", index=False, encoding="utf-8-sig")

    cv_df = run_window_cv(features, cfg)
    cv_df.to_csv(outdir / "sem4c_seedfamily_window_cv.csv", index=False, encoding="utf-8-sig")

    retr_df = run_muF_retrieval(features, cfg)
    retr_df.to_csv(outdir / "sem4c_muF_window_retrieval.csv", index=False, encoding="utf-8-sig")

    rank_df = build_rank_summary(cv_df, retr_df)
    rank_df.to_csv(outdir / "sem4c_window_rank_summary.csv", index=False, encoding="utf-8-sig")

    summary = build_results_summary(cv_df, retr_df, rank_df)
    json_dump(summary, outdir / "sem4c_results_summary.json")

    print("=" * 100)
    print("SEM-4C: Latent Seed Window Localization")
    print("=" * 100)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nOutput files:")
    for p in [
        "sem4c_config.json",
        "sem4c_window_inventory.csv",
        "sem4c_seedfamily_window_cv.csv",
        "sem4c_muF_window_retrieval.csv",
        "sem4c_window_rank_summary.csv",
        "sem4c_results_summary.json",
    ]:
        print("  -", outdir / p)


if __name__ == "__main__":
    main()
