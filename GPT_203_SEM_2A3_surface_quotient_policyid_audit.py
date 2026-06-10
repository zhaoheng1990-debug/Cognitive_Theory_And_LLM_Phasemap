# -*- coding: utf-8 -*-
"""
GPT_199_SEM_2A3_surface_quotient_policyid_audit.py

SEM-2A.3: Surface-Quotient PolicyID Audit
-----------------------------------------
Goal:
    Test whether PolicyID / Constraint signal becomes recoverable after explicitly
    quotienting out SurfaceForm effects from trajectory features.

Background:
    SEM-2A.2 found:
        - Constraint recovery is strong under ordinary CV.
        - Constraint recovery is strong under Leave-One-Concept-Out.
        - Constraint recovery fails under Leave-One-Surface-Family-Out.
        - Constraint residual space is low-rank, rank90 ~= 6.

New decomposition:
    Prompt = (Concept, PolicyID, SurfaceForm)

Feature model:
    TrajectoryFeature ~= ConceptEffect + PolicyEffect + SurfaceEffect + Noise

This experiment compares:
    raw features
    concept-quotient features
    surface-quotient features
    concept+surface quotient features
    two-way additive residual features

Main target:
    TrajectoryResidual -> PolicyID / Constraint

Default:
    Reads existing SEM-2A.2 outputs if present:
        ./sem2a2_outputs/sem2a2_features.csv
    Otherwise extracts features again from Qwen.

Outputs:
    sem2a3_outputs/
        sem2a3_config.json
        sem2a3_dataset.csv
        sem2a3_features_raw.csv
        sem2a3_results_summary.json
        sem2a3_quotient_ablation.csv
        sem2a3_baseline_comparison.csv
        sem2a3_constraint_rank_by_transform.csv
        sem2a3_feature_importance.csv

Run:
    python GPT_199_SEM_2A3_surface_quotient_policyid_audit.py
"""

import os
import json
import random
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Any, Dict

import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA


# ============================================================
# CONFIG
# ============================================================

@dataclass
class CFG:
    model_key: str = "qwen"
    model_path: str = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

    # Reuse 2A.2 features whenever possible.
    reuse_features_if_exists: bool = True
    sem2a2_features_path: str = r"./sem2a2_outputs/sem2a2_features.csv"

    save_dir: str = "./sem2a3_outputs"

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "auto"

    topk: int = 100
    batch_size: int = 4
    max_length: int = 256

    init_layers: Tuple[int, ...] = tuple(range(0, 7))
    mid_layers: Tuple[int, ...] = tuple(range(7, 20))
    decision_layers: Tuple[int, ...] = tuple(range(20, 26))
    full_layers: Tuple[int, ...] = tuple(range(0, 26))

    n_splits: int = 4
    random_state: int = 42
    use_random_forest: bool = True
    save_feature_importance: bool = True

    # 24 concepts x 10 constraints x 8 styles = 1920 prompts.
    max_rows: int = 0

    # 2A.3 is PolicyID-focused by default.
    targets: Tuple[str, ...] = ("constraint",)


cfg = CFG()


# ============================================================
# UTILITIES
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_torch_dtype(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    return "auto"


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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=convert)


def safe_mean(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def safe_std(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.std(xs)) if xs else float("nan")


def cosine_np(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


# ============================================================
# DATASET: same as SEM-2A.2
# ============================================================

CONCEPTS = [
    "Paris", "Tokyo", "Newton", "Einstein", "Python", "Transformer",
    "Apple", "Hospital", "Democracy", "Evolution", "Gravity", "Market",
    "Language", "Memory", "Cancer", "Court", "School", "Ocean",
    "Desert", "Music", "Painting", "Energy", "Robot", "Internet",
]

CONSTRAINTS = [
    "definition",
    "relation",
    "property",
    "list",
    "cause",
    "compare",
    "mechanism",
    "risk",
    "plan",
    "counterfactual",
]

TEMPLATES = {
    "definition": [
        "What is {concept}?",
        "Give a concise definition of {concept}.",
        "For {concept}, produce the answer using the 'what it is' operation.",
        "Apply the identity-description operation to {concept}.",
        "Handle {concept} by saying what kind of thing it is.",
        "For {concept}, return a short identity statement.",
        "Use operation code A: describe the basic identity of {concept}.",
        "In the shared analysis frame, resolve {concept} by identifying it.",
    ],
    "relation": [
        "What is {concept} related to?",
        "Identify the broader category or domain associated with {concept}.",
        "For {concept}, produce the answer using the 'connection' operation.",
        "Apply the relation-mapping operation to {concept}.",
        "Handle {concept} by placing it in a broader context.",
        "For {concept}, return its main association.",
        "Use operation code B: connect {concept} to its domain.",
        "In the shared analysis frame, resolve {concept} by mapping its relation.",
    ],
    "property": [
        "What are notable features of {concept}?",
        "Describe key characteristics of {concept}.",
        "For {concept}, produce the answer using the 'attribute' operation.",
        "Apply the feature-extraction operation to {concept}.",
        "Handle {concept} by giving traits rather than examples.",
        "For {concept}, return characteristic attributes.",
        "Use operation code C: extract properties of {concept}.",
        "In the shared analysis frame, resolve {concept} by listing attributes.",
    ],
    "list": [
        "List examples related to {concept}.",
        "Name several well-known items associated with {concept}.",
        "For {concept}, produce the answer using the 'enumeration' operation.",
        "Apply the example-generation operation to {concept}.",
        "Handle {concept} by giving multiple related items.",
        "For {concept}, return a compact set of examples.",
        "Use operation code D: enumerate related instances for {concept}.",
        "In the shared analysis frame, resolve {concept} by expanding examples.",
    ],
    "cause": [
        "What causes or explains {concept}?",
        "Explain why {concept} occurs or matters.",
        "For {concept}, produce the answer using the 'explanation' operation.",
        "Apply the causal-account operation to {concept}.",
        "Handle {concept} by giving reasons or drivers.",
        "For {concept}, return a short causal account.",
        "Use operation code E: explain drivers of {concept}.",
        "In the shared analysis frame, resolve {concept} by tracing causes.",
    ],
    "compare": [
        "Compare {concept} with a similar concept.",
        "Contrast {concept} with another relevant example.",
        "For {concept}, produce the answer using the 'contrast' operation.",
        "Apply the comparison operation to {concept}.",
        "Handle {concept} by separating it from a related case.",
        "For {concept}, return similarities and differences.",
        "Use operation code F: contrast {concept} with a neighbor.",
        "In the shared analysis frame, resolve {concept} by comparing it.",
    ],
    "mechanism": [
        "How does {concept} work?",
        "Describe the internal process behind {concept}.",
        "For {concept}, produce the answer using the 'process' operation.",
        "Apply the mechanism-tracing operation to {concept}.",
        "Handle {concept} by describing components and steps.",
        "For {concept}, return the working process.",
        "Use operation code G: trace how {concept} functions.",
        "In the shared analysis frame, resolve {concept} by explaining mechanism.",
    ],
    "risk": [
        "What risks are associated with {concept}?",
        "Identify vulnerabilities or dangers related to {concept}.",
        "For {concept}, produce the answer using the 'failure' operation.",
        "Apply the risk-audit operation to {concept}.",
        "Handle {concept} by looking for failure modes.",
        "For {concept}, return likely problems or hazards.",
        "Use operation code H: audit risks of {concept}.",
        "In the shared analysis frame, resolve {concept} by checking failure modes.",
    ],
    "plan": [
        "Make a plan involving {concept}.",
        "Outline steps to use or study {concept}.",
        "For {concept}, produce the answer using the 'procedure' operation.",
        "Apply the planning operation to {concept}.",
        "Handle {concept} by giving an ordered course of action.",
        "For {concept}, return practical steps.",
        "Use operation code I: plan around {concept}.",
        "In the shared analysis frame, resolve {concept} by sequencing actions.",
    ],
    "counterfactual": [
        "What if {concept} were different?",
        "Imagine an alternative version of {concept} and analyze it.",
        "For {concept}, produce the answer using the 'alternative' operation.",
        "Apply the counterfactual operation to {concept}.",
        "Handle {concept} by changing an assumption.",
        "For {concept}, return a hypothetical variation.",
        "Use operation code J: alter a condition involving {concept}.",
        "In the shared analysis frame, resolve {concept} by considering an alternative.",
    ],
}


def build_sem2a3_dataset() -> pd.DataFrame:
    rows = []
    for concept in CONCEPTS:
        for constraint in CONSTRAINTS:
            prompts = TEMPLATES[constraint]
            for style_id, prompt in enumerate(prompts):
                seed_id = f"{concept}__{constraint}"
                rows.append({
                    "prompt_id": len(rows),
                    "concept": concept,
                    "constraint": constraint,
                    "seed_id": seed_id,
                    "style_id": style_id,
                    "prompt": prompt.format(concept=concept),
                })
    df = pd.DataFrame(rows)
    if cfg.max_rows and cfg.max_rows > 0:
        df = df.sample(n=min(cfg.max_rows, len(df)), random_state=cfg.random_state).reset_index(drop=True)
        df["prompt_id"] = np.arange(len(df))
    return df


# ============================================================
# MODEL LOADING AND FEATURE EXTRACTION
# ============================================================

def load_model_and_tokenizer(cfg: CFG):
    print(f"[SEM-2A.3] Loading model: {cfg.model_path}")
    dtype = get_torch_dtype(cfg.dtype)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
        device_map="auto" if cfg.device == "cuda" else None,
    )
    if cfg.device != "cuda":
        model.to(cfg.device)
    model.eval()
    return model, tokenizer


def get_lm_head_weight(model) -> torch.Tensor:
    if hasattr(model, "get_output_embeddings") and model.get_output_embeddings() is not None:
        return model.get_output_embeddings().weight
    raise RuntimeError("Could not find output embedding / lm_head weight.")


@torch.no_grad()
def extract_batch_features(prompts: List[str], model, tokenizer, cfg: CFG) -> List[dict]:
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=cfg.max_length)
    enc = {k: v.to(model.device) for k, v in enc.items()}

    out = model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
    hidden_states = out.hidden_states
    W = get_lm_head_weight(model).to(model.device)

    attention_mask = enc["attention_mask"]
    last_pos = attention_mask.sum(dim=1) - 1

    batch_features = []

    for bi in range(len(prompts)):
        per_layer = {}
        centers = {}
        spreads = {}
        entropies = {}
        topk_mean_logits = {}
        topk_gap_logits = {}
        center_norms = {}

        prev_center = None
        prev_set = None

        for layer in cfg.full_layers:
            hs_index = layer + 1
            if hs_index >= len(hidden_states):
                continue

            h = hidden_states[hs_index][bi, last_pos[bi], :]
            logits = torch.matmul(h, W.T)
            vals, ids = torch.topk(logits, k=cfg.topk, dim=-1)

            ids_np = ids.detach().cpu().numpy().astype(np.int64)
            vals_np = vals.detach().float().cpu().numpy()
            emb = W[ids].detach().float()
            center = emb.mean(dim=0)
            center_np = center.cpu().numpy()

            emb_norm = torch.nn.functional.normalize(emb, dim=-1)
            c_norm = torch.nn.functional.normalize(center.unsqueeze(0), dim=-1)
            cos = (emb_norm * c_norm).sum(dim=-1)
            spread = float((1.0 - cos).mean().detach().cpu())

            probs = torch.softmax(vals.float(), dim=-1)
            entropy = float((-(probs * torch.log(probs + 1e-12)).sum()).detach().cpu())

            id_set = set(ids_np.tolist())

            centers[layer] = center_np
            spreads[layer] = spread
            entropies[layer] = entropy
            topk_mean_logits[layer] = float(vals_np.mean())
            topk_gap_logits[layer] = float(vals_np[0] - vals_np[-1])
            center_norms[layer] = float(np.linalg.norm(center_np))

            prefix = f"L{layer}"
            per_layer[f"{prefix}_spread"] = spread
            per_layer[f"{prefix}_entropy"] = entropy
            per_layer[f"{prefix}_topk_mean_logit"] = float(vals_np.mean())
            per_layer[f"{prefix}_topk_gap_logit"] = float(vals_np[0] - vals_np[-1])
            per_layer[f"{prefix}_center_norm"] = float(np.linalg.norm(center_np))

            if prev_center is not None:
                per_layer[f"{prefix}_center_shift_cosdist"] = 1.0 - cosine_np(prev_center, center_np)
                per_layer[f"{prefix}_center_shift_l2"] = float(np.linalg.norm(center_np - prev_center))
            else:
                per_layer[f"{prefix}_center_shift_cosdist"] = 0.0
                per_layer[f"{prefix}_center_shift_l2"] = 0.0

            if prev_set is not None:
                jac = jaccard(prev_set, id_set)
                per_layer[f"{prefix}_topk_jaccard_prev"] = jac
                per_layer[f"{prefix}_topk_jaccard_dist_prev"] = 1.0 - jac
            else:
                per_layer[f"{prefix}_topk_jaccard_prev"] = 1.0
                per_layer[f"{prefix}_topk_jaccard_dist_prev"] = 0.0

            prev_center = center_np
            prev_set = id_set

        def add_window_features(name: str, layers: Tuple[int, ...]):
            valid = [l for l in layers if l in centers]
            if not valid:
                return

            sp = [spreads[l] for l in valid]
            en = [entropies[l] for l in valid]
            gap = [topk_gap_logits[l] for l in valid]
            meanlog = [topk_mean_logits[l] for l in valid]
            cn = [center_norms[l] for l in valid]

            per_layer[f"{name}_spread_mean"] = safe_mean(sp)
            per_layer[f"{name}_spread_std"] = safe_std(sp)
            per_layer[f"{name}_entropy_mean"] = safe_mean(en)
            per_layer[f"{name}_entropy_std"] = safe_std(en)
            per_layer[f"{name}_gap_mean"] = safe_mean(gap)
            per_layer[f"{name}_gap_std"] = safe_std(gap)
            per_layer[f"{name}_meanlogit_mean"] = safe_mean(meanlog)
            per_layer[f"{name}_center_norm_mean"] = safe_mean(cn)

            shift_cos_vals, shift_l2_vals, jacdist_vals = [], [], []
            for l in valid[1:]:
                shift_cos_vals.append(per_layer.get(f"L{l}_center_shift_cosdist", np.nan))
                shift_l2_vals.append(per_layer.get(f"L{l}_center_shift_l2", np.nan))
                jacdist_vals.append(per_layer.get(f"L{l}_topk_jaccard_dist_prev", np.nan))

            per_layer[f"{name}_center_shift_cosdist_mean"] = safe_mean(shift_cos_vals)
            per_layer[f"{name}_center_shift_cosdist_std"] = safe_std(shift_cos_vals)
            per_layer[f"{name}_center_shift_l2_mean"] = safe_mean(shift_l2_vals)
            per_layer[f"{name}_center_shift_l2_std"] = safe_std(shift_l2_vals)
            per_layer[f"{name}_jaccard_dist_mean"] = safe_mean(jacdist_vals)
            per_layer[f"{name}_jaccard_dist_std"] = safe_std(jacdist_vals)

            if len(valid) >= 2:
                start = centers[valid[0]]
                end = centers[valid[-1]]
                per_layer[f"{name}_start_end_cosdist"] = 1.0 - cosine_np(start, end)
                per_layer[f"{name}_start_end_l2"] = float(np.linalg.norm(end - start))

                path_len = 0.0
                for a, b in zip(valid[:-1], valid[1:]):
                    path_len += float(np.linalg.norm(centers[b] - centers[a]))
                chord = float(np.linalg.norm(end - start)) + 1e-9
                per_layer[f"{name}_path_len"] = path_len
                per_layer[f"{name}_detour_ratio"] = path_len / chord
            else:
                per_layer[f"{name}_start_end_cosdist"] = 0.0
                per_layer[f"{name}_start_end_l2"] = 0.0
                per_layer[f"{name}_path_len"] = 0.0
                per_layer[f"{name}_detour_ratio"] = 0.0

        add_window_features("init", cfg.init_layers)
        add_window_features("mid", cfg.mid_layers)
        add_window_features("decision", cfg.decision_layers)
        add_window_features("full", cfg.full_layers)

        batch_features.append(per_layer)

    return batch_features


def extract_all_features(df: pd.DataFrame, model, tokenizer, cfg: CFG) -> pd.DataFrame:
    all_features = []
    prompts = df["prompt"].tolist()
    for start in range(0, len(prompts), cfg.batch_size):
        end = min(start + cfg.batch_size, len(prompts))
        print(f"[SEM-2A.3] Feature extraction batch {start}:{end} / {len(prompts)}")
        feats = extract_batch_features(prompts[start:end], model, tokenizer, cfg)
        all_features.extend(feats)
    feat_df = pd.DataFrame(all_features)
    return pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)


def load_or_extract_features(cfg: CFG, save_dir: Path) -> pd.DataFrame:
    if cfg.reuse_features_if_exists and Path(cfg.sem2a2_features_path).exists():
        print(f"[SEM-2A.3] Reusing features from {cfg.sem2a2_features_path}")
        feat_df = pd.read_csv(cfg.sem2a2_features_path)
        return feat_df

    df = build_sem2a3_dataset()
    model, tokenizer = load_model_and_tokenizer(cfg)
    feat_df = extract_all_features(df, model, tokenizer, cfg)
    return feat_df


# ============================================================
# FEATURE COLUMNS AND QUOTIENT TRANSFORMS
# ============================================================

def get_feature_columns(df: pd.DataFrame, window: str = "all") -> List[str]:
    exclude = {"prompt_id", "concept", "constraint", "seed_id", "style_id", "prompt"}
    numeric_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    if window == "all":
        return numeric_cols
    if window == "init":
        prefixes = [f"L{i}_" for i in range(0, 7)] + ["init_"]
    elif window == "mid":
        prefixes = [f"L{i}_" for i in range(7, 20)] + ["mid_"]
    elif window == "decision":
        prefixes = [f"L{i}_" for i in range(20, 26)] + ["decision_"]
    elif window == "full_summary":
        prefixes = ["init_", "mid_", "decision_", "full_"]
    else:
        raise ValueError(window)
    return [c for c in numeric_cols if any(c.startswith(p) for p in prefixes)]


def make_feature_matrix(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.astype(np.float32)


def residualize_by_group_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    groups_train,
    groups_test,
    mode: str = "subtract",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train-fold-safe group residualization.
    For known groups in train: subtract train group mean.
    For unseen groups in test: subtract global train mean.

    This is important for leave-one-style-out or leave-one-concept-out:
    the held-out group has no train group mean, so the transform cannot leak test data.
    """
    Xtr = X_train.copy()
    Xte = X_test.copy()
    global_mean = X_train.mean(axis=0)

    group_means = {}
    for g in pd.Series(groups_train).unique():
        idx = (pd.Series(groups_train).values == g)
        group_means[g] = X_train.loc[idx, :].mean(axis=0)

    def get_mean(g):
        return group_means.get(g, global_mean)

    for i, g in enumerate(groups_train):
        Xtr.iloc[i, :] = Xtr.iloc[i, :] - get_mean(g)

    for i, g in enumerate(groups_test):
        Xte.iloc[i, :] = Xte.iloc[i, :] - get_mean(g)

    return Xtr, Xte


def additive_residual_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    concept_train,
    concept_test,
    style_train,
    style_test,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fold-safe two-way additive residual:
        X_resid = X - mean(concept) - mean(style) + global_mean

    If a concept/style is unseen in test, use global mean for that component.
    """
    global_mean = X_train.mean(axis=0)

    c_means = {}
    for c in pd.Series(concept_train).unique():
        idx = (pd.Series(concept_train).values == c)
        c_means[c] = X_train.loc[idx, :].mean(axis=0)

    s_means = {}
    for s in pd.Series(style_train).unique():
        idx = (pd.Series(style_train).values == s)
        s_means[s] = X_train.loc[idx, :].mean(axis=0)

    def cmean(c):
        return c_means.get(c, global_mean)

    def smean(s):
        return s_means.get(s, global_mean)

    Xtr = X_train.copy()
    Xte = X_test.copy()

    for i, (c, s) in enumerate(zip(concept_train, style_train)):
        Xtr.iloc[i, :] = Xtr.iloc[i, :] - cmean(c) - smean(s) + global_mean

    for i, (c, s) in enumerate(zip(concept_test, style_test)):
        Xte.iloc[i, :] = Xte.iloc[i, :] - cmean(c) - smean(s) + global_mean

    return Xtr, Xte


# ============================================================
# MODELING
# ============================================================

def make_classifier(model_name: str, cfg: CFG):
    if model_name == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=4000, solver="lbfgs", class_weight="balanced")),
        ])
    if model_name == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RidgeClassifier(class_weight="balanced")),
        ])
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=cfg.random_state,
            class_weight="balanced_subsample",
            max_depth=None,
            n_jobs=-1,
        )
    raise ValueError(model_name)


def get_splits(df: pd.DataFrame, y: np.ndarray, cv_mode: str, group_col: str, cfg: CFG):
    dummyX = np.zeros((len(df), 1))
    if cv_mode == "stratified":
        min_count = np.min(np.bincount(y))
        n_splits = max(2, min(cfg.n_splits, int(min_count)))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)
        return list(cv.split(dummyX, y))
    if cv_mode == "group":
        groups = df[group_col].values
        n_groups = len(np.unique(groups))
        cv = GroupKFold(n_splits=min(cfg.n_splits, n_groups))
        return list(cv.split(dummyX, y, groups=groups))
    raise ValueError(cv_mode)


def transform_train_test(
    transform_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if transform_name == "raw":
        return X_train, X_test

    if transform_name == "concept_quotient":
        return residualize_by_group_train_test(
            X_train, X_test,
            df_train["concept"].values, df_test["concept"].values,
        )

    if transform_name == "style_quotient":
        return residualize_by_group_train_test(
            X_train, X_test,
            df_train["style_id"].values, df_test["style_id"].values,
        )

    if transform_name == "concept_style_additive_residual":
        return additive_residual_train_test(
            X_train, X_test,
            df_train["concept"].values, df_test["concept"].values,
            df_train["style_id"].values, df_test["style_id"].values,
        )

    if transform_name == "global_centered":
        mu = X_train.mean(axis=0)
        return X_train - mu, X_test - mu

    raise ValueError(transform_name)


def evaluate_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    model_name: str,
    cv_mode: str,
    group_col: str,
    transform_name: str,
    cfg: CFG,
) -> Dict[str, Any]:
    X_all = make_feature_matrix(df, feature_cols)
    y_raw = df[target_col].astype(str).values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    splits = get_splits(df, y, cv_mode, group_col, cfg)
    y_true_all, y_pred_all = [], []

    for fold, (tr, te) in enumerate(splits):
        if set(y[te]) - set(y[tr]):
            return {
                "target": target_col,
                "model": model_name,
                "cv_mode": cv_mode,
                "group_col": group_col,
                "transform": transform_name,
                "accuracy": float("nan"),
                "macro_f1": float("nan"),
                "weighted_f1": float("nan"),
                "skipped": True,
                "skip_reason": "test contains unseen target classes",
                "n_features": len(feature_cols),
            }

        X_train = X_all.iloc[tr, :].reset_index(drop=True)
        X_test = X_all.iloc[te, :].reset_index(drop=True)
        df_train = df.iloc[tr, :].reset_index(drop=True)
        df_test = df.iloc[te, :].reset_index(drop=True)

        Xtr_t, Xte_t = transform_train_test(transform_name, X_train, X_test, df_train, df_test)

        clf = make_classifier(model_name, cfg)
        clf.fit(Xtr_t.values.astype(np.float32), y[tr])
        pred = clf.predict(Xte_t.values.astype(np.float32))

        y_true_all.extend(y[te].tolist())
        y_pred_all.extend(pred.tolist())

    return {
        "target": target_col,
        "model": model_name,
        "cv_mode": cv_mode,
        "group_col": group_col,
        "transform": transform_name,
        "n": int(len(df)),
        "n_classes": int(len(le.classes_)),
        "n_features": int(len(feature_cols)),
        "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
        "macro_f1": float(f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_all, y_pred_all).tolist(),
        "skipped": False,
    }


def evaluate_surface_baseline(df: pd.DataFrame, target_col: str, cv_mode: str, group_col: str, cfg: CFG):
    y_raw = df[target_col].astype(str).values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    prompts = df["prompt"].values

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
        ("clf", LinearSVC(class_weight="balanced", max_iter=12000)),
    ])

    splits = get_splits(df, y, cv_mode, group_col, cfg)
    y_true_all, y_pred_all = [], []

    for tr, te in splits:
        if set(y[te]) - set(y[tr]):
            return {
                "target": target_col,
                "model": "surface_tfidf_linearsvc",
                "cv_mode": cv_mode,
                "group_col": group_col,
                "accuracy": float("nan"),
                "macro_f1": float("nan"),
                "weighted_f1": float("nan"),
                "n_classes": int(len(le.classes_)),
                "skipped": True,
                "skip_reason": "test contains unseen target classes",
            }
        pipe.fit(prompts[tr], y[tr])
        pred = pipe.predict(prompts[te])
        y_true_all.extend(y[te].tolist())
        y_pred_all.extend(pred.tolist())

    return {
        "target": target_col,
        "model": "surface_tfidf_linearsvc",
        "cv_mode": cv_mode,
        "group_col": group_col,
        "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
        "macro_f1": float(f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0)),
        "n_classes": int(len(le.classes_)),
        "skipped": False,
    }


def run_quotient_ablation(feat_df: pd.DataFrame, cfg: CFG) -> Tuple[pd.DataFrame, pd.DataFrame]:
    targets = list(cfg.targets)
    windows = ["init", "mid", "decision", "full_summary", "all"]
    models = ["ridge", "logreg"] + (["rf"] if cfg.use_random_forest else [])
    transforms = [
        "raw",
        "global_centered",
        "concept_quotient",
        "style_quotient",
        "concept_style_additive_residual",
    ]
    cv_setups = [
        ("stratified", None),
        ("group", "style_id"),
        ("group", "concept"),
    ]

    results = []
    for target in targets:
        for window in windows:
            feature_cols = get_feature_columns(feat_df, window)
            if not feature_cols:
                continue
            for model_name in models:
                for transform in transforms:
                    for cv_mode, group_col in cv_setups:
                        if target == "concept" and group_col == "concept":
                            continue
                        print(f"[SEM-2A.3] Eval target={target}, window={window}, model={model_name}, transform={transform}, cv={cv_mode}, group={group_col}")
                        r = evaluate_cv(
                            feat_df, feature_cols, target, model_name, cv_mode, group_col, transform, cfg
                        )
                        r["window"] = window
                        r["feature_type"] = "trajectory"
                        results.append(r)

    baselines = []
    for target in targets:
        for cv_mode, group_col in cv_setups:
            if target == "concept" and group_col == "concept":
                continue
            print(f"[SEM-2A.3] Surface baseline target={target}, cv={cv_mode}, group={group_col}")
            b = evaluate_surface_baseline(feat_df, target, cv_mode, group_col, cfg)
            b["feature_type"] = "surface_only"
            b["window"] = "prompt_text"
            b["transform"] = "surface_raw"
            baselines.append(b)

    return pd.DataFrame(results), pd.DataFrame(baselines)


# ============================================================
# RANK AUDIT BY TRANSFORM
# ============================================================

def offline_transform_all(
    df: pd.DataFrame,
    feature_cols: List[str],
    transform_name: str,
) -> pd.DataFrame:
    """
    Offline residuals for rank audit only. This may use all data because it is
    descriptive geometry, not predictive held-out evaluation.
    """
    X0 = make_feature_matrix(df, feature_cols)
    X = pd.DataFrame(StandardScaler().fit_transform(X0), columns=feature_cols)

    if transform_name == "raw":
        return X

    if transform_name == "global_centered":
        return X - X.mean(axis=0)

    if transform_name == "concept_quotient":
        Y = X.copy()
        for concept in df["concept"].unique():
            idx = df["concept"] == concept
            Y.loc[idx, :] = X.loc[idx, :] - X.loc[idx, :].mean(axis=0)
        return Y

    if transform_name == "style_quotient":
        Y = X.copy()
        for style in df["style_id"].unique():
            idx = df["style_id"] == style
            Y.loc[idx, :] = X.loc[idx, :] - X.loc[idx, :].mean(axis=0)
        return Y

    if transform_name == "concept_style_additive_residual":
        global_mean = X.mean(axis=0)
        c_means = {c: X.loc[df["concept"] == c, :].mean(axis=0) for c in df["concept"].unique()}
        s_means = {s: X.loc[df["style_id"] == s, :].mean(axis=0) for s in df["style_id"].unique()}
        Y = X.copy()
        for i, row in df.reset_index(drop=True).iterrows():
            Y.iloc[i, :] = X.iloc[i, :] - c_means[row["concept"]] - s_means[row["style_id"]] + global_mean
        return Y

    raise ValueError(transform_name)


def constraint_rank_by_transform(feat_df: pd.DataFrame, cfg: CFG) -> pd.DataFrame:
    rows = []
    feature_cols = get_feature_columns(feat_df, "all")
    transforms = [
        "raw",
        "global_centered",
        "concept_quotient",
        "style_quotient",
        "concept_style_additive_residual",
    ]

    for transform in transforms:
        X = offline_transform_all(feat_df, feature_cols, transform)

        # Aggregate by constraint.
        centroids = []
        labels = []
        for constraint in sorted(feat_df["constraint"].unique()):
            idx = feat_df["constraint"] == constraint
            centroids.append(X.loc[idx, :].mean(axis=0).values)
            labels.append(constraint)
        C = np.vstack(centroids)

        n_comp = min(C.shape[0], C.shape[1])
        pca = PCA(n_components=n_comp, random_state=cfg.random_state)
        pca.fit(C)

        cumsum = np.cumsum(pca.explained_variance_ratio_)

        for i, ev in enumerate(pca.explained_variance_ratio_):
            rows.append({
                "transform": transform,
                "component": i + 1,
                "explained_variance_ratio": float(ev),
                "cumulative_explained_variance": float(cumsum[i]),
                "rank_estimate": np.nan,
                "n_constraints": len(labels),
                "n_features": len(feature_cols),
            })

        for threshold in [0.8, 0.9, 0.95]:
            k = int(np.searchsorted(cumsum, threshold) + 1)
            rows.append({
                "transform": transform,
                "component": f"rank_for_{threshold}",
                "explained_variance_ratio": np.nan,
                "cumulative_explained_variance": threshold,
                "rank_estimate": k,
                "n_constraints": len(labels),
                "n_features": len(feature_cols),
            })

    return pd.DataFrame(rows)


def feature_importance_rf(feat_df: pd.DataFrame, feature_cols: List[str], cfg: CFG) -> pd.DataFrame:
    X = make_feature_matrix(feat_df, feature_cols)
    y = LabelEncoder().fit_transform(feat_df["constraint"].astype(str).values)
    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=cfg.random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    rf.fit(X.values.astype(np.float32), y)
    return pd.DataFrame({
        "target": "constraint",
        "feature": feature_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)


# ============================================================
# SUMMARY
# ============================================================

def best_row(df, mask):
    sub = df[mask & (df["skipped"] != True)]
    if not len(sub):
        return None
    return sub.sort_values(["accuracy", "macro_f1"], ascending=False).iloc[0].to_dict()


def summarize_results(result_df: pd.DataFrame, baseline_df: pd.DataFrame, rank_df: pd.DataFrame, cfg: CFG) -> Dict[str, Any]:
    summary = {
        "config": asdict(cfg),
        "best_overall": {},
        "key_tests": {},
        "rank_by_transform": {},
        "verdict": "UNDETERMINED",
        "interpretation": [],
    }

    for transform in sorted(result_df["transform"].dropna().unique()):
        br = best_row(result_df, (result_df["target"] == "constraint") & (result_df["transform"] == transform))
        if br:
            summary["best_overall"][transform] = br

    # Key tests: style-heldout and concept-heldout by transform.
    for transform in sorted(result_df["transform"].dropna().unique()):
        for group_col in ["style_id", "concept"]:
            br = best_row(
                result_df,
                (result_df["target"] == "constraint") &
                (result_df["transform"] == transform) &
                (result_df["group_col"] == group_col)
            )
            key = f"{transform}__group_{group_col}"
            if br:
                surf = best_row(
                    baseline_df,
                    (baseline_df["target"] == "constraint") &
                    (baseline_df["group_col"] == group_col)
                )
                summary["key_tests"][key] = {
                    "best_traj_acc": float(br["accuracy"]),
                    "best_traj_macro_f1": float(br["macro_f1"]),
                    "best_traj_window": br.get("window"),
                    "best_traj_model": br.get("model"),
                    "surface_acc": float(surf["accuracy"]) if surf else None,
                    "traj_minus_surface": float(br["accuracy"] - surf["accuracy"]) if surf else None,
                }

    # Rank.
    for transform in sorted(rank_df["transform"].dropna().unique()):
        sub = rank_df[(rank_df["transform"] == transform) & (rank_df["component"].astype(str).str.startswith("rank_for_"))]
        summary["rank_by_transform"][transform] = {
            str(row["component"]): int(row["rank_estimate"])
            for _, row in sub.iterrows()
        }

    # Verdict.
    # Main aim: style-heldout improves over raw and surface baseline after style/concept residualization.
    raw_style = summary["key_tests"].get("raw__group_style_id", {}).get("best_traj_acc", 0.0)
    styleq_style = summary["key_tests"].get("style_quotient__group_style_id", {}).get("best_traj_acc", 0.0)
    cs_style = summary["key_tests"].get("concept_style_additive_residual__group_style_id", {}).get("best_traj_acc", 0.0)

    raw_concept = summary["key_tests"].get("raw__group_concept", {}).get("best_traj_acc", 0.0)
    cs_concept = summary["key_tests"].get("concept_style_additive_residual__group_concept", {}).get("best_traj_acc", 0.0)

    surface_style = summary["key_tests"].get("raw__group_style_id", {}).get("surface_acc", None)
    best_style_after = max(styleq_style, cs_style)
    best_style_margin = best_style_after - surface_style if surface_style is not None else -999

    rank90_cs = summary["rank_by_transform"].get("concept_style_additive_residual", {}).get("rank_for_0.9", 999)

    if best_style_after >= 0.70 and best_style_margin >= 0.10 and cs_concept >= 0.85 and rank90_cs <= 8:
        summary["verdict"] = "PASS_STRONG_SURFACE_QUOTIENT_POLICYID"
    elif best_style_after >= 0.55 and best_style_margin >= 0.05 and cs_concept >= 0.80 and rank90_cs <= 8:
        summary["verdict"] = "PASS_LITE_SURFACE_QUOTIENT_POLICYID"
    elif best_style_after > raw_style + 0.10:
        summary["verdict"] = "PARTIAL_PASS_SURFACE_QUOTIENT_IMPROVES_STYLE_HELDOUT"
    elif cs_concept >= 0.85 and rank90_cs <= 8:
        summary["verdict"] = "PARTIAL_PASS_POLICY_LOW_RANK_CONCEPT_HELDOUT"
    else:
        summary["verdict"] = "NO_PASS_YET"

    summary["interpretation"].append(
        "Compare raw__group_style_id with style_quotient__group_style_id and concept_style_additive_residual__group_style_id."
    )
    summary["interpretation"].append(
        "If quotient transforms improve style-heldout accuracy, PolicyID signal was partly hidden under SurfaceForm directions."
    )
    summary["interpretation"].append(
        "If concept-heldout remains high after quotienting, PolicyID generalizes across Concept."
    )
    summary["interpretation"].append(
        "Rank estimates are descriptive, not held-out predictive evidence."
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    warnings.filterwarnings("ignore")
    set_seed(cfg.seed)
    ensure_dir(cfg.save_dir)
    save_dir = Path(cfg.save_dir)

    print("=" * 100)
    print("SEM-2A.3: Surface-Quotient PolicyID Audit")
    print("=" * 100)
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    json_dump(asdict(cfg), save_dir / "sem2a3_config.json")

    dataset = build_sem2a3_dataset()
    dataset.to_csv(save_dir / "sem2a3_dataset.csv", index=False, encoding="utf-8-sig")

    feat_df = load_or_extract_features(cfg, save_dir)
    feat_df.to_csv(save_dir / "sem2a3_features_raw.csv", index=False, encoding="utf-8-sig")
    print(f"[SEM-2A.3] Feature table shape={feat_df.shape}")

    result_df, baseline_df = run_quotient_ablation(feat_df, cfg)
    result_df.to_csv(save_dir / "sem2a3_quotient_ablation.csv", index=False, encoding="utf-8-sig")
    baseline_df.to_csv(save_dir / "sem2a3_baseline_comparison.csv", index=False, encoding="utf-8-sig")

    rank_df = constraint_rank_by_transform(feat_df, cfg)
    rank_df.to_csv(save_dir / "sem2a3_constraint_rank_by_transform.csv", index=False, encoding="utf-8-sig")

    if cfg.save_feature_importance:
        all_cols = get_feature_columns(feat_df, "all")
        imp_df = feature_importance_rf(feat_df, all_cols, cfg).head(200)
        imp_df.to_csv(save_dir / "sem2a3_feature_importance.csv", index=False, encoding="utf-8-sig")

    summary = summarize_results(result_df, baseline_df, rank_df, cfg)
    json_dump(summary, save_dir / "sem2a3_results_summary.json")

    print("\n" + "=" * 100)
    print("[SEM-2A.3] SUMMARY")
    print("=" * 100)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n[SEM-2A.3] Output files:")
    for p in [
        "sem2a3_config.json",
        "sem2a3_dataset.csv",
        "sem2a3_features_raw.csv",
        "sem2a3_quotient_ablation.csv",
        "sem2a3_baseline_comparison.csv",
        "sem2a3_constraint_rank_by_transform.csv",
        "sem2a3_feature_importance.csv",
        "sem2a3_results_summary.json",
    ]:
        print("  -", save_dir / p)


if __name__ == "__main__":
    main()
