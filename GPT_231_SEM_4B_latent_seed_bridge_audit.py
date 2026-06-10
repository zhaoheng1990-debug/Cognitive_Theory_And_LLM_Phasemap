# -*- coding: utf-8 -*-
"""
GPT_214_SEM_4B_latent_seed_bridge_audit.py

SEM-4B: Latent Seed Bridge Audit
--------------------------------

Goal:
    Test whether shallow latent seed bridges text prompt and trajectory seed μ_F.

Motivation:
    SEM-3 showed:
        Trajectory -> SeedFamily / μ_F is recoverable.
        μ_F -> natural-language Prompt* regeneration is difficult.

    The correction:
        Prompt_text is not the seed ontology.
        Prompt_text is a seed carrier.
        The model first maps:
            Prompt_text -> Seed_latent
        through tokenizer / embedding / shallow layers / TopK initialization.

    Therefore, the real bridge should be:

        Prompt_text
          -> Seed_latent^{init window}
          -> TrajectoryFamily / μ_F

    not:
        Prompt_text <-> μ_F directly.

Core questions:
    Q1. Is LatentSeed more predictive of SeedFamily than TextSeed?
    Q2. Is LatentSeed closer to μ_F than TextSeed-derived proxies?
    Q3. Which init window best bridges Prompt_text and μ_F?
    Q4. Does LatentSeed recover PromptStructure / OperatorID better than text surface features?

Inputs:
    sem3a_outputs/
      sem3a_dataset.csv
      sem3a_features.csv
      sem3c0b_family_center_features.csv  [optional location: sem3c0b_outputs]
      sem3c0b_prompt_structure_labels.csv [optional]
      sem3c0b_seed_family_structure.csv   [optional]

Outputs:
    sem4b_outputs/
      sem4b_config.json
      sem4b_feature_inventory.csv
      sem4b_seedfamily_cv.csv
      sem4b_muF_retrieval.csv
      sem4b_representation_comparison.csv
      sem4b_operatorid_cv.csv
      sem4b_results_summary.json

Run:
    python GPT_214_SEM_4B_latent_seed_bridge_audit.py

Notes:
    - This script does NOT forward model again.
    - It uses existing SEM-3A features.
    - TextSeed features are deliberately simple lexical/surface features to avoid
      external embedding dependencies.
    - LatentSeed features are shallow/init TopK-derived features already extracted in SEM-3A.
"""

import json
import math
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# CONFIG
# ============================================================

@dataclass
class CFG:
    SEM3A_DIR: str = "./sem3a_outputs"
    SEM3C0B_DIR: str = "./sem3c0b_outputs"
    OUTPUT_DIR: str = "./sem4b_outputs"

    SEM3A_DATASET: str = "sem3a_dataset.csv"
    SEM3A_FEATURES: str = "sem3a_features.csv"

    # optional prior outputs
    SEM3C0B_FAMILY_CENTERS: str = "sem3c0b_family_center_features.csv"
    SEM3C0B_PROMPT_LABELS: str = "sem3c0b_prompt_structure_labels.csv"
    SEM3C0B_FAMILY_STRUCTURE: str = "sem3c0b_seed_family_structure.csv"

    RANDOM_SEED: int = 42
    N_SPLITS: int = 4

    # Blocks
    LATENT_BLOCKS: Tuple[str, ...] = (
        "init",
        "init_center_pca",
        "init_scalar",
        "early_0_2",
        "mid_init_3_6",
        "center_pca",
        "scalar",
        "all",
    )

    TEXT_BLOCKS: Tuple[str, ...] = (
        "text_tfidf_word",
        "text_tfidf_char",
        "text_surface_stats",
        "text_all",
    )

    # Main comparison
    MAIN_LATENT_BLOCK: str = "init_center_pca"
    MAIN_TEXT_BLOCK: str = "text_all"

    MAX_TFIDF_WORD_FEATURES: int = 256
    MAX_TFIDF_CHAR_FEATURES: int = 256


cfg = CFG()


# ============================================================
# UTILITIES
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


def safe_macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


# ============================================================
# LOAD INPUTS
# ============================================================

def load_inputs(cfg: CFG):
    sem3a_dir = Path(cfg.SEM3A_DIR)
    sem3c_dir = Path(cfg.SEM3C0B_DIR)

    dataset = pd.read_csv(sem3a_dir / cfg.SEM3A_DATASET)
    features = pd.read_csv(sem3a_dir / cfg.SEM3A_FEATURES)

    # Ensure features include metadata.
    if "prompt" not in features.columns:
        features = dataset.merge(features, on="row_id", how="left")

    family_centers_path = sem3c_dir / cfg.SEM3C0B_FAMILY_CENTERS
    family_centers = pd.read_csv(family_centers_path) if family_centers_path.exists() else pd.DataFrame()

    prompt_labels_path = sem3c_dir / cfg.SEM3C0B_PROMPT_LABELS
    prompt_labels = pd.read_csv(prompt_labels_path) if prompt_labels_path.exists() else pd.DataFrame()

    family_structure_path = sem3c_dir / cfg.SEM3C0B_FAMILY_STRUCTURE
    family_structure = pd.read_csv(family_structure_path) if family_structure_path.exists() else pd.DataFrame()

    return dataset, features, family_centers, prompt_labels, family_structure


# ============================================================
# FEATURE BLOCKS
# ============================================================

META_COLS = {
    "row_id", "row_type", "seed_family", "concept_star", "operator_star",
    "surface_id", "prompt", "operator_id", "concept_family", "operator_family",
    "question_form", "specificity_level", "surface_form", "domain_frame",
    "constraint_polarity", "expected_prompt_function"
}


def numeric_feature_cols(df: pd.DataFrame):
    return [c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]


def contains_layer_token(col: str, layers: range):
    return any(f"_L{i}_" in col for i in layers)


def get_latent_cols(features: pd.DataFrame, block: str):
    numeric = numeric_feature_cols(features)

    if block == "all":
        return numeric
    if block == "center_pca":
        return [c for c in numeric if "centerPC" in c]
    if block == "scalar":
        return [c for c in numeric if "centerPC" not in c]

    if block == "init":
        return [c for c in numeric if contains_layer_token(c, range(0, 7))]
    if block == "init_center_pca":
        return [c for c in numeric if "centerPC" in c and contains_layer_token(c, range(0, 7))]
    if block == "init_scalar":
        return [c for c in numeric if "centerPC" not in c and contains_layer_token(c, range(0, 7))]
    if block == "early_0_2":
        return [c for c in numeric if contains_layer_token(c, range(0, 3))]
    if block == "mid_init_3_6":
        return [c for c in numeric if contains_layer_token(c, range(3, 7))]

    raise ValueError(block)


def make_X(df: pd.DataFrame, cols: List[str]):
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    return df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32).values


def build_text_features(df: pd.DataFrame, cfg: CFG):
    prompts = df["prompt"].astype(str).fillna("").tolist()

    # word TFIDF
    word_vec = TfidfVectorizer(
        max_features=cfg.MAX_TFIDF_WORD_FEATURES,
        ngram_range=(1, 2),
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    X_word = word_vec.fit_transform(prompts).toarray().astype(np.float32)
    word_cols = [f"text_word_{t}" for t in word_vec.get_feature_names_out()]

    # char TFIDF
    char_vec = TfidfVectorizer(
        max_features=cfg.MAX_TFIDF_CHAR_FEATURES,
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
    )
    X_char = char_vec.fit_transform(prompts).toarray().astype(np.float32)
    char_cols = [f"text_char_{i}" for i in range(X_char.shape[1])]

    # surface stats
    stats = []
    for p in prompts:
        words = re.findall(r"\b\w+\b", p)
        stats.append({
            "text_len_chars": len(p),
            "text_len_words": len(words),
            "text_num_digits": sum(ch.isdigit() for ch in p),
            "text_num_upper": sum(ch.isupper() for ch in p),
            "text_num_punct": sum((not ch.isalnum() and not ch.isspace()) for ch in p),
            "text_has_question": int("?" in p),
            "text_has_colon": int(":" in p),
            "text_has_concept": int("concept" in p.lower()),
            "text_has_operator": int("operator" in p.lower()),
        })
    stats_df = pd.DataFrame(stats).astype(np.float32)
    X_stats = stats_df.values
    stat_cols = list(stats_df.columns)

    blocks = {
        "text_tfidf_word": (X_word, word_cols),
        "text_tfidf_char": (X_char, char_cols),
        "text_surface_stats": (X_stats, stat_cols),
        "text_all": (np.concatenate([X_word, X_char, X_stats], axis=1), word_cols + char_cols + stat_cols),
    }
    return blocks


# ============================================================
# CV CLASSIFICATION
# ============================================================

def cv_classify(X, y, groups=None, cv_mode="stratified", random_seed=42, model_name="logreg"):
    y = np.asarray(y)
    results = []

    if cv_mode == "stratified":
        n_splits = min(cfg.N_SPLITS, np.min(np.bincount(pd.factorize(y)[0])))
        if n_splits < 2:
            return {"acc": np.nan, "macro_f1": np.nan, "n": len(y)}
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        splits = splitter.split(X, y)
    elif cv_mode == "group":
        if groups is None:
            return {"acc": np.nan, "macro_f1": np.nan, "n": len(y)}
        unique_groups = np.unique(groups)
        if len(unique_groups) < 2:
            return {"acc": np.nan, "macro_f1": np.nan, "n": len(y)}
        n_splits = min(cfg.N_SPLITS, len(unique_groups))
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(X, y, groups=groups)
    else:
        raise ValueError(cv_mode)

    preds = []
    trues = []
    for train_idx, test_idx in splits:
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        if model_name == "ridge":
            clf = make_pipeline(StandardScaler(with_mean=True), RidgeClassifier(alpha=1.0))
        else:
            clf = make_pipeline(
                StandardScaler(with_mean=True),
                LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")
            )
        try:
            clf.fit(Xtr, ytr)
            yp = clf.predict(Xte)
        except Exception:
            # fallback
            clf = make_pipeline(StandardScaler(with_mean=True), RidgeClassifier(alpha=1.0))
            clf.fit(Xtr, ytr)
            yp = clf.predict(Xte)

        preds.extend(list(yp))
        trues.extend(list(yte))

    return {
        "acc": float(accuracy_score(trues, preds)),
        "macro_f1": float(safe_macro_f1(trues, preds)),
        "n": int(len(trues)),
    }


def seedfamily_cv(features: pd.DataFrame, cfg: CFG):
    df = features[features["row_type"] == "original_prompt"].copy().reset_index(drop=True)
    y = df["seed_family"].astype(str).values
    groups_surface = df["surface_id"].astype(str).values if "surface_id" in df.columns else None

    rows = []

    # Latent blocks
    for block in cfg.LATENT_BLOCKS:
        cols = get_latent_cols(df, block)
        X = make_X(df, cols)
        if X.shape[1] == 0:
            continue
        for cv_mode, groups in [("stratified", None), ("group", groups_surface)]:
            out = cv_classify(X, y, groups=groups, cv_mode=cv_mode, random_seed=cfg.RANDOM_SEED)
            rows.append({
                "representation": "latent_seed",
                "block": block,
                "cv": cv_mode if cv_mode == "stratified" else "leave_surface_group",
                "target": "seed_family",
                "n_features": X.shape[1],
                **out,
            })

    # Text blocks
    text_blocks = build_text_features(df, cfg)
    for block, (X, cols) in text_blocks.items():
        for cv_mode, groups in [("stratified", None), ("group", groups_surface)]:
            out = cv_classify(X, y, groups=groups, cv_mode=cv_mode, random_seed=cfg.RANDOM_SEED)
            rows.append({
                "representation": "text_seed",
                "block": block,
                "cv": cv_mode if cv_mode == "stratified" else "leave_surface_group",
                "target": "seed_family",
                "n_features": X.shape[1],
                **out,
            })

    return pd.DataFrame(rows)


# ============================================================
# μF RETRIEVAL
# ============================================================

def build_muF_centers_from_features(features: pd.DataFrame, block: str):
    """
    Compute μF directly from original prompt features in a given block.
    This avoids dependency on 3C0b center vectors and ensures feature-space match.
    """
    df = features.reset_index(drop=True)
    cols = get_latent_cols(df, block) if not block.startswith("text_") else []
    if not cols:
        return None, None, []
    X = make_X(df, cols)
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    centers = {}
    original_mask = (df["row_type"] == "original_prompt").values
    for fam in sorted(df["seed_family"].unique()):
        idx = np.where((df["seed_family"].values == fam) & original_mask)[0]
        if len(idx):
            centers[fam] = Xz[idx].mean(axis=0)
    return centers, (Xz, scaler, df), cols


def retrieval_to_muF_latent(features: pd.DataFrame, cfg: CFG):
    df = features[features["row_type"] == "original_prompt"].copy().reset_index(drop=True)

    rows = []
    for block in cfg.LATENT_BLOCKS:
        cols = get_latent_cols(features, block)
        if not cols:
            continue

        # Fit scaler on all rows then build family centers on originals.
        all_df = features.reset_index(drop=True)
        Xall = make_X(all_df, cols)
        scaler = StandardScaler()
        Xall_z = scaler.fit_transform(Xall)

        centers = {}
        for fam in sorted(all_df["seed_family"].unique()):
            idx = np.where((all_df["seed_family"].values == fam) & (all_df["row_type"].values == "original_prompt"))[0]
            if len(idx):
                centers[fam] = Xall_z[idx].mean(axis=0)

        center_keys = list(centers.keys())
        C = np.vstack([centers[k] for k in center_keys])

        # Evaluate retrieval for original prompts.
        orig_idx = np.where(all_df["row_type"].values == "original_prompt")[0]
        correct = []
        true_gt_random_mean = []
        true_margin_mean = []
        nearest = []

        for idx in orig_idx:
            x = Xall_z[idx]
            true_fam = all_df.iloc[idx]["seed_family"]
            sims = np.array([cosine_np(x, c) for c in C])
            best_i = int(np.argmax(sims))
            pred_fam = center_keys[best_i]
            nearest.append(pred_fam)
            correct.append(int(pred_fam == true_fam))

            if true_fam in center_keys:
                true_sim = sims[center_keys.index(true_fam)]
                other_sims = np.array([s for k, s in zip(center_keys, sims) if k != true_fam])
                true_gt_random_mean.append(float(true_sim > other_sims.mean()))
                true_margin_mean.append(float(true_sim - other_sims.mean()))

        rows.append({
            "representation": "latent_seed",
            "block": block,
            "target": "muF_family_center",
            "n_features": len(cols),
            "nearest_family_acc": float(np.mean(correct)),
            "frac_true_gt_random_mean": float(np.mean(true_gt_random_mean)),
            "mean_margin_vs_random_mean": float(np.mean(true_margin_mean)),
            "n": int(len(correct)),
        })

    return pd.DataFrame(rows)


def retrieval_to_muF_text(features: pd.DataFrame, cfg: CFG):
    df = features[features["row_type"] == "original_prompt"].copy().reset_index(drop=True)
    text_blocks = build_text_features(df, cfg)

    rows = []
    for block, (X, cols) in text_blocks.items():
        scaler = StandardScaler(with_mean=False) if "tfidf" in block else StandardScaler()
        try:
            Xz = scaler.fit_transform(X)
        except Exception:
            Xz = StandardScaler().fit_transform(X)

        centers = {}
        for fam in sorted(df["seed_family"].unique()):
            idx = np.where(df["seed_family"].values == fam)[0]
            if len(idx):
                centers[fam] = Xz[idx].mean(axis=0)

        center_keys = list(centers.keys())
        C = np.vstack([centers[k] for k in center_keys])

        correct = []
        true_gt_random_mean = []
        true_margin_mean = []
        for i, row in df.iterrows():
            x = Xz[i]
            true_fam = row["seed_family"]
            sims = np.array([cosine_np(x, c) for c in C])
            pred = center_keys[int(np.argmax(sims))]
            correct.append(int(pred == true_fam))
            true_sim = sims[center_keys.index(true_fam)]
            other = np.array([s for k, s in zip(center_keys, sims) if k != true_fam])
            true_gt_random_mean.append(float(true_sim > other.mean()))
            true_margin_mean.append(float(true_sim - other.mean()))

        rows.append({
            "representation": "text_seed",
            "block": block,
            "target": "muF_family_center_proxy",
            "n_features": X.shape[1],
            "nearest_family_acc": float(np.mean(correct)),
            "frac_true_gt_random_mean": float(np.mean(true_gt_random_mean)),
            "mean_margin_vs_random_mean": float(np.mean(true_margin_mean)),
            "n": int(len(correct)),
        })

    return pd.DataFrame(rows)


# ============================================================
# OPERATOR / PROMPT STRUCTURE RECOVERY
# ============================================================

def attach_prompt_labels(features: pd.DataFrame, prompt_labels: pd.DataFrame):
    if prompt_labels.empty:
        return features
    cols = [c for c in prompt_labels.columns if c not in features.columns or c == "row_id"]
    return features.merge(prompt_labels[cols], on="row_id", how="left")


def operatorid_cv(features: pd.DataFrame, prompt_labels: pd.DataFrame, cfg: CFG):
    if prompt_labels.empty:
        return pd.DataFrame()

    merged = attach_prompt_labels(features, prompt_labels)
    df = merged[(merged["row_type"] == "original_prompt") & merged["operator_id"].notna()].copy().reset_index(drop=True)
    if len(df) < 5:
        return pd.DataFrame()

    y = df["operator_id"].astype(str).values
    groups_surface = df["surface_id"].astype(str).values if "surface_id" in df.columns else None

    rows = []
    for block in cfg.LATENT_BLOCKS:
        cols = get_latent_cols(df, block)
        if not cols:
            continue
        X = make_X(df, cols)
        for cv_mode, groups in [("stratified", None), ("group", groups_surface)]:
            out = cv_classify(X, y, groups=groups, cv_mode=cv_mode, random_seed=cfg.RANDOM_SEED)
            rows.append({
                "representation": "latent_seed",
                "block": block,
                "cv": cv_mode if cv_mode == "stratified" else "leave_surface_group",
                "target": "operator_id",
                "n_features": X.shape[1],
                **out,
            })

    text_blocks = build_text_features(df, cfg)
    for block, (X, cols) in text_blocks.items():
        for cv_mode, groups in [("stratified", None), ("group", groups_surface)]:
            out = cv_classify(X, y, groups=groups, cv_mode=cv_mode, random_seed=cfg.RANDOM_SEED)
            rows.append({
                "representation": "text_seed",
                "block": block,
                "cv": cv_mode if cv_mode == "stratified" else "leave_surface_group",
                "target": "operator_id",
                "n_features": X.shape[1],
                **out,
            })

    return pd.DataFrame(rows)


# ============================================================
# COMPARISON / SUMMARY
# ============================================================

def build_feature_inventory(features: pd.DataFrame, cfg: CFG):
    rows = []
    for block in cfg.LATENT_BLOCKS:
        cols = get_latent_cols(features, block)
        rows.append({
            "representation": "latent_seed",
            "block": block,
            "n_features": len(cols),
            "example_cols": json.dumps(cols[:10], ensure_ascii=False),
        })
    # Text feature counts only after build on originals.
    df = features[features["row_type"] == "original_prompt"].copy()
    text_blocks = build_text_features(df, cfg)
    for block, (X, cols) in text_blocks.items():
        rows.append({
            "representation": "text_seed",
            "block": block,
            "n_features": X.shape[1],
            "example_cols": json.dumps(cols[:10], ensure_ascii=False),
        })
    return pd.DataFrame(rows)


def compare_representations(seed_cv, retrieval, operator_cv):
    rows = []
    for table_name, df, metric in [
        ("seedfamily_cv", seed_cv, "macro_f1"),
        ("muF_retrieval", retrieval, "nearest_family_acc"),
        ("operatorid_cv", operator_cv, "macro_f1"),
    ]:
        if df is None or df.empty or metric not in df.columns:
            continue
        for target in df["target"].unique():
            sub = df[df["target"] == target].copy()
            text_best = sub[sub["representation"] == "text_seed"][metric].max() if len(sub[sub["representation"] == "text_seed"]) else np.nan
            latent_best = sub[sub["representation"] == "latent_seed"][metric].max() if len(sub[sub["representation"] == "latent_seed"]) else np.nan
            rows.append({
                "table": table_name,
                "target": target,
                "metric": metric,
                "text_best": float(text_best) if np.isfinite(text_best) else np.nan,
                "latent_best": float(latent_best) if np.isfinite(latent_best) else np.nan,
                "latent_minus_text": float(latent_best - text_best) if np.isfinite(latent_best) and np.isfinite(text_best) else np.nan,
                "winner": "latent_seed" if np.isfinite(latent_best) and (not np.isfinite(text_best) or latent_best > text_best) else "text_seed",
            })
    return pd.DataFrame(rows)


def build_results_summary(seed_cv, retrieval, operator_cv, comparison):
    summary = {
        "config": asdict(cfg),
        "verdict": "UNDETERMINED",
        "best": {},
        "diagnosis": "",
        "interpretation": [],
    }

    def best_row(df, metric, representation=None):
        if df is None or df.empty or metric not in df.columns:
            return {}
        sub = df.copy()
        if representation is not None:
            sub = sub[sub["representation"] == representation]
        if sub.empty:
            return {}
        return sub.sort_values(metric, ascending=False).iloc[0].to_dict()

    summary["best"]["seedfamily_text"] = best_row(seed_cv, "macro_f1", "text_seed")
    summary["best"]["seedfamily_latent"] = best_row(seed_cv, "macro_f1", "latent_seed")
    summary["best"]["retrieval_text"] = best_row(retrieval, "nearest_family_acc", "text_seed")
    summary["best"]["retrieval_latent"] = best_row(retrieval, "nearest_family_acc", "latent_seed")
    summary["best"]["operator_text"] = best_row(operator_cv, "macro_f1", "text_seed")
    summary["best"]["operator_latent"] = best_row(operator_cv, "macro_f1", "latent_seed")

    # Main diagnosis
    latent_retr = summary["best"]["retrieval_latent"].get("nearest_family_acc", np.nan)
    text_retr = summary["best"]["retrieval_text"].get("nearest_family_acc", np.nan)
    latent_cv = summary["best"]["seedfamily_latent"].get("macro_f1", np.nan)
    text_cv = summary["best"]["seedfamily_text"].get("macro_f1", np.nan)

    if np.isfinite(latent_retr) and np.isfinite(text_retr) and latent_retr > text_retr + 0.05:
        summary["diagnosis"] = "LATENT_SEED_BRIDGES_MUF_BETTER_THAN_TEXT"
    elif np.isfinite(latent_cv) and np.isfinite(text_cv) and latent_cv > text_cv + 0.05:
        summary["diagnosis"] = "LATENT_SEED_CLASSIFIES_SEEDFAMILY_BETTER_THAN_TEXT"
    elif np.isfinite(text_retr) and np.isfinite(latent_retr) and text_retr >= latent_retr:
        summary["diagnosis"] = "TEXT_SEED_REMAINS_COMPETITIVE"
    else:
        summary["diagnosis"] = "MIXED_OR_INSUFFICIENT"

    # Verdict
    if summary["diagnosis"] == "LATENT_SEED_BRIDGES_MUF_BETTER_THAN_TEXT" and latent_retr >= 0.80:
        summary["verdict"] = "PASS_STRONG_LATENT_SEED_BRIDGE"
    elif summary["diagnosis"].startswith("LATENT") and (np.nan_to_num(latent_retr) >= 0.65 or np.nan_to_num(latent_cv) >= 0.65):
        summary["verdict"] = "PASS_LITE_LATENT_SEED_BRIDGE"
    elif np.nan_to_num(latent_retr) > 0.5 or np.nan_to_num(latent_cv) > 0.5:
        summary["verdict"] = "PARTIAL_PASS_LATENT_SEED_SIGNAL"
    else:
        summary["verdict"] = "NO_PASS_YET"

    summary["comparison"] = comparison.to_dict(orient="records") if comparison is not None and len(comparison) else []

    summary["interpretation"] = [
        "Prompt_text is treated as external seed carrier, not seed ontology.",
        "LatentSeed is operationalized as shallow/init TopK-derived features from SEM-3A.",
        "TrajectorySeed is operationalized as μF family center in the same latent feature space.",
        "If LatentSeed beats TextSeed, SEM should shift from prompt-regeneration to latent-seed retrieval/control.",
    ]
    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_dir(cfg.OUTPUT_DIR)
    outdir = Path(cfg.OUTPUT_DIR)
    json_dump(asdict(cfg), outdir / "sem4b_config.json")

    dataset, features, family_centers, prompt_labels, family_structure = load_inputs(cfg)

    inventory = build_feature_inventory(features, cfg)
    inventory.to_csv(outdir / "sem4b_feature_inventory.csv", index=False, encoding="utf-8-sig")

    seed_cv = seedfamily_cv(features, cfg)
    seed_cv.to_csv(outdir / "sem4b_seedfamily_cv.csv", index=False, encoding="utf-8-sig")

    retr_lat = retrieval_to_muF_latent(features, cfg)
    retr_txt = retrieval_to_muF_text(features, cfg)
    retrieval = pd.concat([retr_lat, retr_txt], ignore_index=True)
    retrieval.to_csv(outdir / "sem4b_muF_retrieval.csv", index=False, encoding="utf-8-sig")

    operator_cv_df = operatorid_cv(features, prompt_labels, cfg)
    operator_cv_df.to_csv(outdir / "sem4b_operatorid_cv.csv", index=False, encoding="utf-8-sig")

    comparison = compare_representations(seed_cv, retrieval, operator_cv_df)
    comparison.to_csv(outdir / "sem4b_representation_comparison.csv", index=False, encoding="utf-8-sig")

    summary = build_results_summary(seed_cv, retrieval, operator_cv_df, comparison)
    json_dump(summary, outdir / "sem4b_results_summary.json")

    print("=" * 100)
    print("SEM-4B: Latent Seed Bridge Audit")
    print("=" * 100)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nOutput files:")
    for p in [
        "sem4b_config.json",
        "sem4b_feature_inventory.csv",
        "sem4b_seedfamily_cv.csv",
        "sem4b_muF_retrieval.csv",
        "sem4b_operatorid_cv.csv",
        "sem4b_representation_comparison.csv",
        "sem4b_results_summary.json",
    ]:
        print("  -", outdir / p)


if __name__ == "__main__":
    main()
