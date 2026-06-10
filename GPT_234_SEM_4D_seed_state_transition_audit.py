# -*- coding: utf-8 -*-
"""
GPT_216_SEM_4D_seed_state_transition_audit.py

SEM-4D: Seed State Transition Audit
-----------------------------------

Goal:
    Test whether the internal seed evolves through a predictable state chain:

        Seed_init       -> Seed_traj_shape      -> Seed_commit
        L0-L6              L7-L19                  L23-L25

Background:
    SEM-4C found:
      - init windows contain partial SeedFamily signal.
      - μF retrieval peaks in L7-L19 center_pca.
      - SeedFamily recovery peaks in L23-L25 center_pca.
    Therefore μF is not shallow prompt seed, but a post-dynamics trajectory
    attractor center.

Core questions:
    Q1. Can Seed_init predict Seed_traj_shape?
    Q2. Can Seed_traj_shape predict Seed_commit?
    Q3. Is information gain largest from init -> traj_shape or traj_shape -> commit?
    Q4. Is the transition chain more predictive than direct init -> commit?
    Q5. Does a low-dimensional transition map exist?

Inputs:
    sem3a_outputs/
      sem3a_dataset.csv
      sem3a_features.csv

Outputs:
    sem4d_outputs/
      sem4d_config.json
      sem4d_state_feature_inventory.csv
      sem4d_transition_regression.csv
      sem4d_transition_retrieval.csv
      sem4d_information_gain.csv
      sem4d_state_predictor_cv.csv
      sem4d_results_summary.json

Run:
    python GPT_216_SEM_4D_seed_state_transition_audit.py

Notes:
    - No model forward is needed.
    - It operates on SEM-3A features.
    - This is not prompt regeneration; it tests internal seed-state transitions.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, LogisticRegression, RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import r2_score, accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIG
# ============================================================

@dataclass
class CFG:
    SEM3A_DIR: str = "./sem3a_outputs"
    OUTPUT_DIR: str = "./sem4d_outputs"

    SEM3A_DATASET: str = "sem3a_dataset.csv"
    SEM3A_FEATURES: str = "sem3a_features.csv"

    RANDOM_SEED: int = 42
    N_SPLITS: int = 4

    # State windows derived from SEM-4C.
    INIT_WINDOW: str = "L0_6"
    SHAPE_WINDOW: str = "L7_19"
    COMMIT_WINDOW: str = "L23_25"

    # Feature family to use as state coordinates.
    # center_pca is the main geometric state; all/scalar can be audited too.
    FEATURE_FAMILIES: Tuple[str, ...] = ("center_pca", "scalar", "transport", "all")

    MAIN_FEATURE_FAMILY: str = "center_pca"

    # Regression models.
    RIDGE_ALPHA: float = 10.0
    PLS_COMPONENTS: int = 16

    # Optional PCA compression before transition regression.
    PCA_DIMS: Tuple[int, ...] = (8, 16, 32, 64, 128)


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
    a = np.asarray(a)
    b = np.asarray(b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def mean_row_cosine(Y_true, Y_pred):
    vals = [cosine_np(a, b) for a, b in zip(Y_true, Y_pred)]
    return float(np.mean(vals))


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
# FEATURE SELECTION
# ============================================================

META_COLS = {
    "row_id", "row_type", "seed_family", "concept_star", "operator_star",
    "surface_id", "prompt", "operator_id", "concept_family", "operator_family",
    "question_form", "specificity_level", "surface_form", "domain_frame",
    "constraint_polarity", "expected_prompt_function"
}


def parse_window(window: str):
    a, b = window.replace("L", "").split("_")
    return int(a), int(b)


def numeric_cols(df: pd.DataFrame):
    return [c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]


def layer_in_col(c: str, lo: int, hi: int):
    return any(f"_L{i}_" in c for i in range(lo, hi + 1))


def get_cols(df: pd.DataFrame, window: str, family: str):
    lo, hi = parse_window(window)
    cols = [c for c in numeric_cols(df) if layer_in_col(c, lo, hi)]

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


def get_state_matrices(features: pd.DataFrame, family: str):
    df = features[features["row_type"] == "original_prompt"].copy().reset_index(drop=True)

    cols_init = get_cols(df, cfg.INIT_WINDOW, family)
    cols_shape = get_cols(df, cfg.SHAPE_WINDOW, family)
    cols_commit = get_cols(df, cfg.COMMIT_WINDOW, family)

    X_init = make_X(df, cols_init)
    X_shape = make_X(df, cols_shape)
    X_commit = make_X(df, cols_commit)

    return df, {
        "init": (X_init, cols_init),
        "shape": (X_shape, cols_shape),
        "commit": (X_commit, cols_commit),
    }


# ============================================================
# TRANSITION REGRESSION
# ============================================================

def cv_transition_regression(X, Y, groups=None, model_type="ridge", pca_dim=None):
    """
    Predict Y state from X state using CV.
    Returns R2, cosine, L2 error.
    """
    if X.shape[1] == 0 or Y.shape[1] == 0:
        return {"r2": np.nan, "cos": np.nan, "l2": np.nan, "n": len(X)}

    unique_groups = np.unique(groups) if groups is not None else []
    if groups is not None and len(unique_groups) >= 2:
        n_splits = min(cfg.N_SPLITS, len(unique_groups))
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(X, groups=groups)
        cv_name = "group_surface"
    else:
        n_splits = min(cfg.N_SPLITS, len(X))
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=cfg.RANDOM_SEED)
        splits = splitter.split(X)
        cv_name = "kfold"

    Y_true_all = []
    Y_pred_all = []

    for train_idx, test_idx in splits:
        Xtr, Xte = X[train_idx], X[test_idx]
        Ytr, Yte = Y[train_idx], Y[test_idx]

        sx = StandardScaler()
        sy = StandardScaler()
        Xtrz = sx.fit_transform(Xtr)
        Xtez = sx.transform(Xte)
        Ytrz = sy.fit_transform(Ytr)
        Ytez = sy.transform(Yte)

        if pca_dim is not None:
            dim = min(pca_dim, Xtrz.shape[1], Xtrz.shape[0] - 1)
            if dim >= 1:
                pca = PCA(n_components=dim, random_state=cfg.RANDOM_SEED)
                Xtrz2 = pca.fit_transform(Xtrz)
                Xtez2 = pca.transform(Xtez)
            else:
                Xtrz2, Xtez2 = Xtrz, Xtez
        else:
            Xtrz2, Xtez2 = Xtrz, Xtez

        if model_type == "pls":
            n_comp = min(cfg.PLS_COMPONENTS, Xtrz2.shape[1], Ytrz.shape[1], Xtrz2.shape[0] - 1)
            if n_comp < 1:
                model = Ridge(alpha=cfg.RIDGE_ALPHA)
            else:
                model = PLSRegression(n_components=n_comp)
        else:
            model = Ridge(alpha=cfg.RIDGE_ALPHA)

        model.fit(Xtrz2, Ytrz)
        pred_z = model.predict(Xtez2)
        pred = sy.inverse_transform(pred_z)

        Y_true_all.append(Yte)
        Y_pred_all.append(pred)

    Y_true = np.vstack(Y_true_all)
    Y_pred = np.vstack(Y_pred_all)

    return {
        "cv": cv_name,
        "r2": float(r2_score(Y_true, Y_pred, multioutput="variance_weighted")),
        "cos": float(mean_row_cosine(Y_true, Y_pred)),
        "l2": float(np.mean(np.linalg.norm(Y_true - Y_pred, axis=1))),
        "n": int(len(Y_true)),
    }


def run_transition_regression(features: pd.DataFrame, cfg: CFG):
    rows = []
    for family in cfg.FEATURE_FAMILIES:
        df, states = get_state_matrices(features, family)
        groups = df["surface_id"].astype(str).values if "surface_id" in df.columns else None

        transitions = [
            ("init_to_shape", "init", "shape"),
            ("shape_to_commit", "shape", "commit"),
            ("init_to_commit", "init", "commit"),
            ("init_plus_shape_to_commit", "init+shape", "commit"),
        ]

        for name, src, tgt in transitions:
            if src == "init+shape":
                X = np.concatenate([states["init"][0], states["shape"][0]], axis=1)
                src_features = states["init"][0].shape[1] + states["shape"][0].shape[1]
            else:
                X = states[src][0]
                src_features = X.shape[1]
            Y = states[tgt][0]

            for model_type in ["ridge", "pls"]:
                # Raw
                out = cv_transition_regression(X, Y, groups=groups, model_type=model_type, pca_dim=None)
                rows.append({
                    "feature_family": family,
                    "transition": name,
                    "source": src,
                    "target": tgt,
                    "model_type": model_type,
                    "pca_dim": "none",
                    "n_source_features": int(src_features),
                    "n_target_features": int(Y.shape[1]),
                    **out,
                })

                # PCA compressed
                for dim in cfg.PCA_DIMS:
                    if dim < X.shape[1]:
                        out = cv_transition_regression(X, Y, groups=groups, model_type=model_type, pca_dim=dim)
                        rows.append({
                            "feature_family": family,
                            "transition": name,
                            "source": src,
                            "target": tgt,
                            "model_type": model_type,
                            "pca_dim": int(dim),
                            "n_source_features": int(src_features),
                            "n_target_features": int(Y.shape[1]),
                            **out,
                        })
    return pd.DataFrame(rows)


# ============================================================
# RETRIEVAL THROUGH STATE TRANSITIONS
# ============================================================

def family_centers(X, y):
    centers = {}
    for fam in sorted(np.unique(y)):
        idx = np.where(y == fam)[0]
        centers[fam] = X[idx].mean(axis=0)
    return centers


def nearest_center_acc(X, y, centers):
    fams = list(centers.keys())
    C = np.vstack([centers[f] for f in fams])
    correct = []
    margins = []
    for x, true in zip(X, y):
        sims = np.array([cosine_np(x, c) for c in C])
        pred = fams[int(np.argmax(sims))]
        correct.append(int(pred == true))
        true_sim = sims[fams.index(true)]
        other = np.array([s for f, s in zip(fams, sims) if f != true])
        margins.append(float(true_sim - other.mean()))
    return float(np.mean(correct)), float(np.mean(margins))


def run_transition_retrieval(features: pd.DataFrame, cfg: CFG):
    """
    Evaluate family-center retrieval accuracy using init/shape/commit states,
    plus predicted shape/commit states from transitions.
    """
    rows = []
    for family in cfg.FEATURE_FAMILIES:
        df, states = get_state_matrices(features, family)
        y = df["seed_family"].astype(str).values
        groups = df["surface_id"].astype(str).values if "surface_id" in df.columns else None

        # Standardize each state globally for retrieval.
        standardized = {}
        for name, (X, cols) in states.items():
            if X.shape[1] == 0:
                continue
            scaler = StandardScaler()
            standardized[name] = scaler.fit_transform(X)

        for state_name, Xz in standardized.items():
            centers = family_centers(Xz, y)
            acc, margin = nearest_center_acc(Xz, y, centers)
            rows.append({
                "feature_family": family,
                "state": state_name,
                "representation": "observed",
                "target": "seed_family_center",
                "nearest_family_acc": acc,
                "mean_margin_vs_random_mean": margin,
                "n_features": int(Xz.shape[1]),
                "n": int(len(y)),
            })

        # Predicted shape from init, predicted commit from shape.
        # Use in-sample CV predictions to avoid pure leakage.
        for transition, src, tgt in [
            ("init_to_shape", "init", "shape"),
            ("shape_to_commit", "shape", "commit"),
            ("init_to_commit", "init", "commit"),
        ]:
            if src not in standardized or tgt not in standardized:
                continue
            X = states[src][0]
            Y = states[tgt][0]
            if X.shape[1] == 0 or Y.shape[1] == 0:
                continue

            pred = cross_val_predict_state(X, Y, groups=groups)
            scaler_y = StandardScaler()
            Yz = scaler_y.fit_transform(Y)
            predz = scaler_y.transform(pred)

            centers = family_centers(Yz, y)
            acc, margin = nearest_center_acc(predz, y, centers)
            rows.append({
                "feature_family": family,
                "state": tgt,
                "representation": f"predicted_{transition}",
                "target": "seed_family_center",
                "nearest_family_acc": acc,
                "mean_margin_vs_random_mean": margin,
                "n_features": int(Y.shape[1]),
                "n": int(len(y)),
            })

    return pd.DataFrame(rows)


def cross_val_predict_state(X, Y, groups=None):
    if groups is not None and len(np.unique(groups)) >= 2:
        splitter = GroupKFold(n_splits=min(cfg.N_SPLITS, len(np.unique(groups))))
        splits = splitter.split(X, groups=groups)
    else:
        splitter = KFold(n_splits=min(cfg.N_SPLITS, len(X)), shuffle=True, random_state=cfg.RANDOM_SEED)
        splits = splitter.split(X)

    pred_all = np.zeros_like(Y, dtype=np.float32)
    for tr, te in splits:
        sx = StandardScaler()
        sy = StandardScaler()
        Xtr = sx.fit_transform(X[tr])
        Xte = sx.transform(X[te])
        Ytr = sy.fit_transform(Y[tr])

        model = Ridge(alpha=cfg.RIDGE_ALPHA)
        model.fit(Xtr, Ytr)
        predz = model.predict(Xte)
        pred = sy.inverse_transform(predz)
        pred_all[te] = pred
    return pred_all


# ============================================================
# STATE PREDICTOR CV
# ============================================================

def state_family_cv(features: pd.DataFrame, cfg: CFG):
    rows = []
    for family in cfg.FEATURE_FAMILIES:
        df, states = get_state_matrices(features, family)
        y = df["seed_family"].astype(str).values
        groups = df["surface_id"].astype(str).values if "surface_id" in df.columns else None

        combos = {
            "init": states["init"][0],
            "shape": states["shape"][0],
            "commit": states["commit"][0],
            "init_plus_shape": np.concatenate([states["init"][0], states["shape"][0]], axis=1),
            "shape_plus_commit": np.concatenate([states["shape"][0], states["commit"][0]], axis=1),
            "init_shape_commit": np.concatenate([states["init"][0], states["shape"][0], states["commit"][0]], axis=1),
        }

        for name, X in combos.items():
            if X.shape[1] == 0:
                continue
            out = cv_classify_state(X, y, groups)
            rows.append({
                "feature_family": family,
                "state_combo": name,
                "n_features": int(X.shape[1]),
                **out,
            })
    return pd.DataFrame(rows)


def cv_classify_state(X, y, groups=None):
    y = np.asarray(y)
    if groups is not None and len(np.unique(groups)) >= 2:
        splitter = GroupKFold(n_splits=min(cfg.N_SPLITS, len(np.unique(groups))))
        splits = splitter.split(X, y, groups)
        cv = "leave_surface_group"
    else:
        labels = pd.factorize(y)[0]
        n_splits = min(cfg.N_SPLITS, np.bincount(labels).min())
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.RANDOM_SEED)
        splits = splitter.split(X, y)
        cv = "stratified"

    trues, preds = [], []
    for tr, te in splits:
        try:
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")
            )
            clf.fit(X[tr], y[tr])
            yp = clf.predict(X[te])
        except Exception:
            clf = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
            clf.fit(X[tr], y[tr])
            yp = clf.predict(X[te])
        trues.extend(y[te])
        preds.extend(yp)

    return {
        "cv": cv,
        "acc": float(accuracy_score(trues, preds)),
        "macro_f1": float(safe_macro_f1(trues, preds)),
        "n": int(len(trues)),
    }


# ============================================================
# INFORMATION GAIN
# ============================================================

def build_information_gain(predictor_cv: pd.DataFrame, transition_df: pd.DataFrame, retrieval_df: pd.DataFrame):
    rows = []
    for family in cfg.FEATURE_FAMILIES:
        sub = predictor_cv[predictor_cv["feature_family"] == family]
        def best_f1(combo):
            r = sub[sub["state_combo"] == combo]
            return float(r["macro_f1"].max()) if len(r) else np.nan

        init = best_f1("init")
        shape = best_f1("shape")
        commit = best_f1("commit")
        init_shape = best_f1("init_plus_shape")
        shape_commit = best_f1("shape_plus_commit")
        all_combo = best_f1("init_shape_commit")

        rows.append({
            "feature_family": family,
            "metric": "seedfamily_macro_f1",
            "init": init,
            "shape": shape,
            "commit": commit,
            "init_plus_shape": init_shape,
            "shape_plus_commit": shape_commit,
            "all": all_combo,
            "gain_shape_over_init": shape - init if np.isfinite(shape) and np.isfinite(init) else np.nan,
            "gain_commit_over_shape": commit - shape if np.isfinite(commit) and np.isfinite(shape) else np.nan,
            "gain_init_shape_over_init": init_shape - init if np.isfinite(init_shape) and np.isfinite(init) else np.nan,
            "gain_shape_commit_over_shape": shape_commit - shape if np.isfinite(shape_commit) and np.isfinite(shape) else np.nan,
        })

    # Transition R2 best summaries
    if len(transition_df):
        for family in cfg.FEATURE_FAMILIES:
            sub = transition_df[transition_df["feature_family"] == family]
            row = {"feature_family": family, "metric": "transition_best_r2"}
            for trans in ["init_to_shape", "shape_to_commit", "init_to_commit", "init_plus_shape_to_commit"]:
                r = sub[sub["transition"] == trans]
                row[trans] = float(r["r2"].max()) if len(r) else np.nan
            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# INVENTORY / SUMMARY
# ============================================================

def build_inventory(features: pd.DataFrame, cfg: CFG):
    rows = []
    for family in cfg.FEATURE_FAMILIES:
        for state, window in [("init", cfg.INIT_WINDOW), ("shape", cfg.SHAPE_WINDOW), ("commit", cfg.COMMIT_WINDOW)]:
            cols = get_cols(features, window, family)
            rows.append({
                "state": state,
                "window": window,
                "feature_family": family,
                "n_features": len(cols),
                "example_cols": json.dumps(cols[:8], ensure_ascii=False),
            })
    return pd.DataFrame(rows)


def build_results_summary(trans_df, retr_df, ig_df, predictor_cv):
    summary = {
        "config": asdict(cfg),
        "verdict": "UNDETERMINED",
        "best": {},
        "diagnosis": "",
        "interpretation": [],
    }

    main = cfg.MAIN_FEATURE_FAMILY

    def best_transition(fam, trans):
        sub = trans_df[(trans_df["feature_family"] == fam) & (trans_df["transition"] == trans)]
        if len(sub) == 0:
            return {}
        return sub.sort_values(["r2", "cos"], ascending=[False, False]).iloc[0].to_dict()

    summary["best"]["init_to_shape"] = best_transition(main, "init_to_shape")
    summary["best"]["shape_to_commit"] = best_transition(main, "shape_to_commit")
    summary["best"]["init_to_commit"] = best_transition(main, "init_to_commit")
    summary["best"]["init_plus_shape_to_commit"] = best_transition(main, "init_plus_shape_to_commit")

    # Best classifier states
    if len(predictor_cv):
        sub = predictor_cv[predictor_cv["feature_family"] == main]
        if len(sub):
            summary["best"]["state_family_cv"] = sub.sort_values("macro_f1", ascending=False).iloc[0].to_dict()

    # Best retrieval
    if len(retr_df):
        sub = retr_df[retr_df["feature_family"] == main]
        if len(sub):
            summary["best"]["state_retrieval"] = sub.sort_values(["nearest_family_acc", "mean_margin_vs_random_mean"], ascending=[False, False]).iloc[0].to_dict()

    # Diagnosis logic
    init_shape_r2 = summary["best"]["init_to_shape"].get("r2", np.nan)
    shape_commit_r2 = summary["best"]["shape_to_commit"].get("r2", np.nan)
    init_commit_r2 = summary["best"]["init_to_commit"].get("r2", np.nan)
    init_shape_commit_r2 = summary["best"]["init_plus_shape_to_commit"].get("r2", np.nan)

    shape_retr = retr_df[
        (retr_df["feature_family"] == main) &
        (retr_df["state"] == "shape") &
        (retr_df["representation"] == "observed")
    ]
    commit_retr = retr_df[
        (retr_df["feature_family"] == main) &
        (retr_df["state"] == "commit") &
        (retr_df["representation"] == "observed")
    ]

    shape_acc = float(shape_retr["nearest_family_acc"].max()) if len(shape_retr) else np.nan
    commit_acc = float(commit_retr["nearest_family_acc"].max()) if len(commit_retr) else np.nan

    if np.nan_to_num(shape_commit_r2) > 0.50 and np.nan_to_num(init_shape_r2) > 0.30:
        summary["diagnosis"] = "SEED_STATE_CHAIN_PREDICTABLE"
    elif np.nan_to_num(shape_commit_r2) > 0.50:
        summary["diagnosis"] = "SHAPE_TO_COMMIT_PREDICTABLE_INIT_WEAK"
    elif np.nan_to_num(init_shape_commit_r2) > np.nan_to_num(init_commit_r2) + 0.10:
        summary["diagnosis"] = "SHAPE_MEDIATES_INIT_TO_COMMIT"
    elif np.nan_to_num(commit_acc) >= 0.90 and np.nan_to_num(shape_acc) >= 0.90:
        summary["diagnosis"] = "STATE_IDENTITY_RETRIEVAL_STRONG_BUT_TRANSITION_WEAK"
    else:
        summary["diagnosis"] = "NO_CLEAR_SEED_STATE_TRANSITION"

    # Verdict
    if summary["diagnosis"] == "SEED_STATE_CHAIN_PREDICTABLE":
        summary["verdict"] = "PASS_STRONG_SEED_STATE_TRANSITION"
    elif summary["diagnosis"] in ["SHAPE_TO_COMMIT_PREDICTABLE_INIT_WEAK", "SHAPE_MEDIATES_INIT_TO_COMMIT"]:
        summary["verdict"] = "PASS_LITE_SEED_STATE_TRANSITION"
    elif summary["diagnosis"] == "STATE_IDENTITY_RETRIEVAL_STRONG_BUT_TRANSITION_WEAK":
        summary["verdict"] = "PARTIAL_PASS_STATE_IDENTITY_ONLY"
    else:
        summary["verdict"] = "NO_PASS_YET"

    summary["interpretation"] = [
        "init state is L0-L6; shape state is L7-L19; commit state is L23-L25.",
        "Transition regression tests whether one state can predict the next geometry.",
        "Retrieval tests whether each state can identify the same SeedFamily center.",
        "If transitions are weak but retrieval is strong, states share identity but not linear dynamics.",
        "If shape mediates init->commit, μF is a trajectory seed produced by middle-layer shaping."
    ]
    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_dir(cfg.OUTPUT_DIR)
    outdir = Path(cfg.OUTPUT_DIR)
    json_dump(asdict(cfg), outdir / "sem4d_config.json")

    dataset, features = load_inputs(cfg)

    inventory = build_inventory(features, cfg)
    inventory.to_csv(outdir / "sem4d_state_feature_inventory.csv", index=False, encoding="utf-8-sig")

    transition_df = run_transition_regression(features, cfg)
    transition_df.to_csv(outdir / "sem4d_transition_regression.csv", index=False, encoding="utf-8-sig")

    retrieval_df = run_transition_retrieval(features, cfg)
    retrieval_df.to_csv(outdir / "sem4d_transition_retrieval.csv", index=False, encoding="utf-8-sig")

    predictor_cv = state_family_cv(features, cfg)
    predictor_cv.to_csv(outdir / "sem4d_state_predictor_cv.csv", index=False, encoding="utf-8-sig")

    ig_df = build_information_gain(predictor_cv, transition_df, retrieval_df)
    ig_df.to_csv(outdir / "sem4d_information_gain.csv", index=False, encoding="utf-8-sig")

    summary = build_results_summary(transition_df, retrieval_df, ig_df, predictor_cv)
    json_dump(summary, outdir / "sem4d_results_summary.json")

    print("=" * 100)
    print("SEM-4D: Seed State Transition Audit")
    print("=" * 100)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nOutput files:")
    for p in [
        "sem4d_config.json",
        "sem4d_state_feature_inventory.csv",
        "sem4d_transition_regression.csv",
        "sem4d_transition_retrieval.csv",
        "sem4d_information_gain.csv",
        "sem4d_state_predictor_cv.csv",
        "sem4d_results_summary.json",
    ]:
        print("  -", outdir / p)


if __name__ == "__main__":
    main()
