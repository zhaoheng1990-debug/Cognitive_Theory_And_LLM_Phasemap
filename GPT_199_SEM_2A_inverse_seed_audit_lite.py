# -*- coding: utf-8 -*-
"""
GPT_197_SEM_2A1_surface_controlled_inverse_seed_audit.py

SEM-2A.1: Surface-Controlled Inverse Seed Audit
------------------------------------------------
Goal:
    Audit whether Trajectory -> Concept / Constraint / Seed recovery remains
    after stronger surface controls.

Compared with SEM-2A:
    1. Larger concept set.
    2. More constraint/policy families.
    3. Shared surface families across all concepts and constraints.
    4. Leave-One-Surface-Family-Out.
    5. Leave-One-Concept-Out for constraint recovery.
    6. More direct test of:
          Dim(Constraint) << Dim(Concept)
       and
          Seed=(Concept, Constraint) -> Seed=(Concept, PolicyID)

Default:
    Qwen only, hard-coded model path.
    Cross-model version should become SEM-2B after this passes.

Outputs:
    sem2a1_outputs/
        sem2a1_dataset.csv
        sem2a1_features.csv
        sem2a1_results_summary.json
        sem2a1_window_ablation.csv
        sem2a1_baseline_comparison.csv
        sem2a1_constraint_rank.csv
        sem2a1_feature_importance.csv

Run:
    python GPT_197_SEM_2A1_surface_controlled_inverse_seed_audit.py
"""

import os
import json
import math
import random
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any

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

    save_dir: str = "./sem2a1_outputs"

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "auto"

    topk: int = 100
    batch_size: int = 4
    max_length: int = 256

    # Qwen convention: hidden_states[layer + 1] = block layer L.
    init_layers: Tuple[int, ...] = tuple(range(0, 7))
    mid_layers: Tuple[int, ...] = tuple(range(7, 20))
    decision_layers: Tuple[int, ...] = tuple(range(20, 26))
    full_layers: Tuple[int, ...] = tuple(range(0, 26))

    n_splits: int = 5
    random_state: int = 42
    use_random_forest: bool = True
    save_feature_importance: bool = True

    # Dataset controls
    # 24 concepts x 10 constraints x 4 surface families = 960 prompts.
    # First pass may take some time. Set max_rows for a quick smoke test.
    max_rows: int = 0  # 0 = all


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
        if pd.isna(o) if not isinstance(o, (list, dict, tuple, str)) else False:
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
    if u == 0:
        return 0.0
    return len(a & b) / u


# ============================================================
# DATASET
# ============================================================

CONCEPTS = [
    "Paris", "Tokyo", "Newton", "Einstein", "Python", "Transformer",
    "Apple", "Hospital", "Democracy", "Evolution", "Gravity", "Market",
    "Language", "Memory", "Cancer", "Court", "School", "Ocean",
    "Desert", "Music", "Painting", "Energy", "Robot", "Internet",
]

# 10 low-rank policy candidates.
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

# Shared templates by constraint and surface family.
# The {concept} placeholder is the only concept-specific lexical insertion.
# style_id is deliberately shared across all constraints but uses different linguistic surfaces.
TEMPLATES = {
    "definition": [
        "What is {concept}?",
        "Explain the meaning of {concept}.",
        "Give a concise definition of {concept}.",
        "In one sentence, describe what {concept} is.",
    ],
    "relation": [
        "What is {concept} related to?",
        "Identify the main category or domain associated with {concept}.",
        "Where does {concept} belong conceptually?",
        "State the broader relation that connects {concept} to its field.",
    ],
    "property": [
        "What are notable features of {concept}?",
        "Describe key characteristics of {concept}.",
        "What properties are commonly associated with {concept}?",
        "List the main attributes of {concept}.",
    ],
    "list": [
        "List examples related to {concept}.",
        "Name several well-known items associated with {concept}.",
        "Give examples connected with {concept}.",
        "Provide a short list of things linked to {concept}.",
    ],
    "cause": [
        "What causes or explains {concept}?",
        "Explain the causes behind {concept}.",
        "Why does {concept} occur or matter?",
        "Describe causal factors related to {concept}.",
    ],
    "compare": [
        "Compare {concept} with a similar concept.",
        "How is {concept} different from related ideas?",
        "Contrast {concept} with another relevant example.",
        "Explain similarities and differences involving {concept}.",
    ],
    "mechanism": [
        "What mechanism underlies {concept}?",
        "Explain how {concept} works.",
        "Describe the internal process behind {concept}.",
        "What are the steps or components that make {concept} function?",
    ],
    "risk": [
        "What risks are associated with {concept}?",
        "Audit possible failure modes of {concept}.",
        "Identify vulnerabilities or dangers related to {concept}.",
        "What could go wrong with {concept}?",
    ],
    "plan": [
        "Make a plan involving {concept}.",
        "Design a practical strategy for working with {concept}.",
        "Outline steps to use or study {concept}.",
        "Create an action plan related to {concept}.",
    ],
    "counterfactual": [
        "What if {concept} were different?",
        "Consider a counterfactual scenario involving {concept}.",
        "How would things change if {concept} did not exist?",
        "Imagine an alternative version of {concept} and analyze it.",
    ],
}


def build_sem2a1_dataset() -> pd.DataFrame:
    rows = []
    for concept in CONCEPTS:
        for constraint in CONSTRAINTS:
            prompts = TEMPLATES[constraint]
            for style_id, template in enumerate(prompts):
                prompt = template.format(concept=concept)
                seed_id = f"{concept}__{constraint}"
                rows.append({
                    "prompt_id": len(rows),
                    "concept": concept,
                    "constraint": constraint,
                    "seed_id": seed_id,
                    "style_id": style_id,
                    "prompt": prompt,
                })

    df = pd.DataFrame(rows)
    if cfg.max_rows and cfg.max_rows > 0:
        # Stratified-ish smoke sample: first rows only is bad, sample random.
        df = df.sample(n=min(cfg.max_rows, len(df)), random_state=cfg.random_state).reset_index(drop=True)
        df["prompt_id"] = np.arange(len(df))
    return df


# ============================================================
# MODEL LOADING
# ============================================================

def load_model_and_tokenizer(cfg: CFG):
    print(f"[SEM-2A.1] Loading model: {cfg.model_path}")
    dtype = get_torch_dtype(cfg.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
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


# ============================================================
# FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_batch_features(prompts: List[str], model, tokenizer, cfg: CFG) -> List[Dict[str, Any]]:
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg.max_length,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    out = model(
        **enc,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    hidden_states = out.hidden_states
    W = get_lm_head_weight(model).to(model.device)
    attention_mask = enc["attention_mask"]
    last_pos = attention_mask.sum(dim=1) - 1

    batch_features = []

    for bi in range(len(prompts)):
        per_layer = {}
        topk_sets = {}
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

            topk_sets[layer] = id_set
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

            shift_cos_vals = []
            shift_l2_vals = []
            jacdist_vals = []
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

                # Simple curvature proxy: path length / chord length
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
        print(f"[SEM-2A.1] Feature extraction batch {start}:{end} / {len(prompts)}")
        feats = extract_batch_features(prompts[start:end], model, tokenizer, cfg)
        all_features.extend(feats)

    feat_df = pd.DataFrame(all_features)
    return pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)


# ============================================================
# MODELING
# ============================================================

def get_feature_columns(df: pd.DataFrame, window: str = "all") -> List[str]:
    exclude = {"prompt_id", "concept", "constraint", "seed_id", "style_id", "prompt"}
    numeric_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)

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


def make_classifier(model_name: str, cfg: CFG):
    if model_name == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=3000,
                solver="lbfgs",
                class_weight="balanced",
            )),
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


def evaluate_classifier_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    model_name: str,
    cv_mode: str,
    group_col: str = None,
    cfg: CFG = cfg,
) -> Dict[str, Any]:
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32)
    y_raw = df[target_col].astype(str).values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    if len(np.unique(y)) < 2:
        return {"error": "target has <2 classes", "target": target_col}

    clf = make_classifier(model_name, cfg)

    if cv_mode == "stratified":
        cv = StratifiedKFold(n_splits=min(cfg.n_splits, np.min(np.bincount(y))), shuffle=True, random_state=cfg.random_state)
        splits = list(cv.split(X, y))
    elif cv_mode == "group":
        if group_col is None:
            raise ValueError("group_col required")
        groups = df[group_col].values
        n_groups = len(np.unique(groups))
        cv = GroupKFold(n_splits=min(cfg.n_splits, n_groups))
        splits = list(cv.split(X, y, groups=groups))
    else:
        raise ValueError(cv_mode)

    y_true_all, y_pred_all = [], []
    for fold, (tr, te) in enumerate(splits):
        # In group by concept, concept target cannot be evaluated if held-out concept unseen.
        # sklearn classifiers cannot predict unseen labels. Detect and skip impossible cases.
        if set(y[te]) - set(y[tr]):
            return {
                "target": target_col,
                "model": model_name,
                "cv_mode": cv_mode,
                "group_col": group_col,
                "n": int(len(df)),
                "n_classes": int(len(le.classes_)),
                "n_features": int(len(feature_cols)),
                "accuracy": float("nan"),
                "macro_f1": float("nan"),
                "weighted_f1": float("nan"),
                "skipped": True,
                "skip_reason": "test contains unseen target classes",
            }
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        y_true_all.extend(y[te].tolist())
        y_pred_all.extend(pred.tolist())

    acc = accuracy_score(y_true_all, y_pred_all)
    macro_f1 = f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0)

    return {
        "target": target_col,
        "model": model_name,
        "cv_mode": cv_mode,
        "group_col": group_col,
        "n": int(len(df)),
        "n_classes": int(len(le.classes_)),
        "n_features": int(len(feature_cols)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classes": le.classes_.tolist(),
        "confusion_matrix": confusion_matrix(y_true_all, y_pred_all).tolist(),
        "skipped": False,
    }


def evaluate_surface_baseline(df: pd.DataFrame, target_col: str, cv_mode: str, group_col: str = None, cfg: CFG = cfg):
    y_raw = df[target_col].astype(str).values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    prompts = df["prompt"].values

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
        ("clf", LinearSVC(class_weight="balanced", max_iter=10000)),
    ])

    if cv_mode == "stratified":
        cv = StratifiedKFold(n_splits=min(cfg.n_splits, np.min(np.bincount(y))), shuffle=True, random_state=cfg.random_state)
        splits = list(cv.split(prompts, y))
    elif cv_mode == "group":
        groups = df[group_col].values
        n_groups = len(np.unique(groups))
        cv = GroupKFold(n_splits=min(cfg.n_splits, n_groups))
        splits = list(cv.split(prompts, y, groups=groups))
    else:
        raise ValueError(cv_mode)

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


def evaluate_layer_shuffle(df: pd.DataFrame, target_col: str, cfg: CFG):
    random_state = np.random.RandomState(cfg.random_state)
    shuffled = df.copy()

    layer_groups = {}
    for c in df.columns:
        if c.startswith("L") and "_" in c:
            layer_prefix = c.split("_")[0]
            suffix = c[len(layer_prefix):]
            if layer_prefix[1:].isdigit():
                layer_groups.setdefault(suffix, []).append(c)

    for suffix, cols in layer_groups.items():
        cols_sorted = sorted(cols, key=lambda x: int(x.split("_")[0][1:]))
        vals = shuffled[cols_sorted].values.copy()
        for i in range(vals.shape[0]):
            perm = random_state.permutation(vals.shape[1])
            vals[i, :] = vals[i, perm]
        shuffled[cols_sorted] = vals

    feature_cols = get_feature_columns(shuffled, "all")
    return evaluate_classifier_cv(shuffled, feature_cols, target_col, "ridge", "stratified", None, cfg)


def run_all_evaluations(feat_df: pd.DataFrame, cfg: CFG) -> Tuple[pd.DataFrame, pd.DataFrame]:
    targets = ["concept", "constraint", "seed_id"]
    windows = ["init", "mid", "decision", "full_summary", "all"]
    model_names = ["ridge", "logreg"] + (["rf"] if cfg.use_random_forest else [])

    results = []

    # Trajectory window/model ablations.
    for target in targets:
        for window in windows:
            feature_cols = get_feature_columns(feat_df, window)
            if len(feature_cols) == 0:
                continue

            for model_name in model_names:
                for cv_mode, group_col in [
                    ("stratified", None),
                    ("group", "style_id"),   # leave-one-surface-family-out
                    ("group", "concept"),    # leave-one-concept-out; meaningful for constraint only
                ]:
                    if target == "concept" and group_col == "concept":
                        continue  # impossible: held-out concept label unseen.
                    print(f"[SEM-2A.1] Eval target={target}, window={window}, model={model_name}, cv={cv_mode}, group={group_col}")
                    r = evaluate_classifier_cv(feat_df, feature_cols, target, model_name, cv_mode, group_col, cfg)
                    r["window"] = window
                    r["feature_type"] = "trajectory"
                    results.append(r)

    # Baselines.
    baselines = []
    for target in targets:
        for cv_mode, group_col in [
            ("stratified", None),
            ("group", "style_id"),
            ("group", "concept"),
        ]:
            if target == "concept" and group_col == "concept":
                continue
            print(f"[SEM-2A.1] Surface baseline target={target}, cv={cv_mode}, group={group_col}")
            b = evaluate_surface_baseline(feat_df, target, cv_mode, group_col, cfg)
            b["feature_type"] = "surface_only"
            b["window"] = "prompt_text"
            baselines.append(b)

        print(f"[SEM-2A.1] Layer-shuffle baseline target={target}")
        ls = evaluate_layer_shuffle(feat_df, target, cfg)
        ls["feature_type"] = "layer_shuffled_trajectory"
        ls["window"] = "all"
        baselines.append(ls)

    return pd.DataFrame(results), pd.DataFrame(baselines)


def feature_importance_rf(feat_df: pd.DataFrame, target: str, feature_cols: List[str], cfg: CFG) -> pd.DataFrame:
    X = feat_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32)
    y = LabelEncoder().fit_transform(feat_df[target].astype(str).values)
    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=cfg.random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    rf.fit(X, y)
    return pd.DataFrame({
        "target": target,
        "feature": feature_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)


def constraint_rank_audit(feat_df: pd.DataFrame, cfg: CFG) -> pd.DataFrame:
    """
    Estimate low-rank structure of constraint signatures after removing concept means.

    Procedure:
      1. For numeric features, compute residual:
            feature - mean(feature | concept)
      2. Aggregate residual by constraint.
      3. PCA on constraint centroids.
    """
    feature_cols = get_feature_columns(feat_df, "all")
    X = feat_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=feature_cols)

    residual = X.copy()
    for concept in feat_df["concept"].unique():
        idx = feat_df["concept"] == concept
        residual.loc[idx, :] = X.loc[idx, :] - X.loc[idx, :].mean(axis=0)

    # Constraint centroids in concept-residual space.
    centroids = []
    labels = []
    for constraint in sorted(feat_df["constraint"].unique()):
        idx = feat_df["constraint"] == constraint
        centroids.append(residual.loc[idx, :].mean(axis=0).values)
        labels.append(constraint)
    C = np.vstack(centroids)

    n_comp = min(C.shape[0], C.shape[1])
    pca = PCA(n_components=n_comp, random_state=cfg.random_state)
    pca.fit(C)

    rows = []
    cum = 0.0
    for i, ev in enumerate(pca.explained_variance_ratio_):
        cum += float(ev)
        rows.append({
            "component": i + 1,
            "explained_variance_ratio": float(ev),
            "cumulative_explained_variance": float(cum),
            "n_constraints": len(labels),
            "n_features": len(feature_cols),
        })

    # Estimate rank thresholds.
    for threshold in [0.8, 0.9, 0.95]:
        k = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), threshold) + 1)
        rows.append({
            "component": f"rank_for_{threshold}",
            "explained_variance_ratio": np.nan,
            "cumulative_explained_variance": threshold,
            "n_constraints": len(labels),
            "n_features": len(feature_cols),
            "rank_estimate": k,
        })

    return pd.DataFrame(rows)


def summarize_results(result_df: pd.DataFrame, baseline_df: pd.DataFrame, rank_df: pd.DataFrame, cfg: CFG) -> Dict[str, Any]:
    combined = pd.concat([result_df, baseline_df], ignore_index=True, sort=False)

    summary = {
        "config": asdict(cfg),
        "best_by_target": {},
        "surface_control": {},
        "constraint_low_rank": {},
        "verdict": "UNDETERMINED",
        "interpretation": [],
        "thresholds": {
            "pass_lite": {
                "concept_style_acc": 0.75,
                "constraint_style_acc": 0.70,
                "constraint_concept_heldout_acc": 0.70,
                "seed_style_acc": 0.35,
                "constraint_rank90": 5,
            },
            "pass_strong": {
                "concept_style_acc": 0.85,
                "constraint_style_acc": 0.80,
                "constraint_concept_heldout_acc": 0.80,
                "seed_style_acc": 0.50,
                "constraint_rank90": 5,
            }
        }
    }

    for target in ["concept", "constraint", "seed_id"]:
        sub = combined[(combined["target"] == target) & (combined["feature_type"] == "trajectory") & (combined["skipped"] != True)]
        if len(sub):
            best = sub.sort_values(["accuracy", "macro_f1"], ascending=False).iloc[0].to_dict()
            summary["best_by_target"][target] = best

    # Best under style-heldout for surface control.
    for target in ["concept", "constraint", "seed_id"]:
        for group_col in ["style_id", "concept"]:
            if target == "concept" and group_col == "concept":
                continue
            traj = combined[
                (combined["target"] == target) &
                (combined["feature_type"] == "trajectory") &
                (combined["group_col"] == group_col) &
                (combined["skipped"] != True)
            ]
            surf = combined[
                (combined["target"] == target) &
                (combined["feature_type"] == "surface_only") &
                (combined["group_col"] == group_col) &
                (combined["skipped"] != True)
            ]
            key = f"{target}_group_{group_col}"
            if len(traj):
                summary["surface_control"][key] = {
                    "best_traj_acc": float(traj["accuracy"].max()),
                    "best_traj_macro_f1": float(traj.sort_values("accuracy", ascending=False).iloc[0]["macro_f1"]),
                    "best_surface_acc": float(surf["accuracy"].max()) if len(surf) else None,
                    "traj_minus_surface": float(traj["accuracy"].max() - surf["accuracy"].max()) if len(surf) else None,
                }

    # Constraint rank.
    rank_rows = rank_df[rank_df["component"].astype(str).str.startswith("rank_for_")]
    for _, row in rank_rows.iterrows():
        summary["constraint_low_rank"][str(row["component"])] = int(row["rank_estimate"])

    # Verdict logic.
    concept_style = summary["surface_control"].get("concept_group_style_id", {}).get("best_traj_acc", 0.0)
    constraint_style = summary["surface_control"].get("constraint_group_style_id", {}).get("best_traj_acc", 0.0)
    constraint_concept = summary["surface_control"].get("constraint_group_concept", {}).get("best_traj_acc", 0.0)
    seed_style = summary["surface_control"].get("seed_id_group_style_id", {}).get("best_traj_acc", 0.0)
    rank90 = summary["constraint_low_rank"].get("rank_for_0.9", 999)

    pass_lite = (
        concept_style >= 0.75 and
        constraint_style >= 0.70 and
        constraint_concept >= 0.70 and
        seed_style >= 0.35 and
        rank90 <= 5
    )
    pass_strong = (
        concept_style >= 0.85 and
        constraint_style >= 0.80 and
        constraint_concept >= 0.80 and
        seed_style >= 0.50 and
        rank90 <= 5
    )

    if pass_strong:
        summary["verdict"] = "PASS_STRONG_SURFACE_CONTROLLED"
    elif pass_lite:
        summary["verdict"] = "PASS_LITE_SURFACE_CONTROLLED"
    else:
        # Partial labels
        if constraint_style >= 0.70 and constraint_concept >= 0.70 and rank90 <= 5:
            summary["verdict"] = "PARTIAL_PASS_CONSTRAINT_LOW_RANK"
        elif constraint_style >= 0.70:
            summary["verdict"] = "PARTIAL_PASS_CONSTRAINT_RECOVERY"
        else:
            summary["verdict"] = "NO_PASS_YET"

    summary["interpretation"].append(
        "If constraint_group_concept is high, constraint/policy signatures generalize across held-out concepts."
    )
    summary["interpretation"].append(
        "If rank_for_0.9 <= 5, constraint residual signatures are low-rank under concept-mean quotient."
    )
    summary["interpretation"].append(
        "If trajectory beats surface in style-heldout, surface leakage is reduced."
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
    print("SEM-2A.1: Surface-Controlled Inverse Seed Audit")
    print("=" * 100)
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    json_dump(asdict(cfg), save_dir / "sem2a1_config.json")

    df = build_sem2a1_dataset()
    df.to_csv(save_dir / "sem2a1_dataset.csv", index=False, encoding="utf-8-sig")
    print(f"[SEM-2A.1] Dataset saved: {save_dir / 'sem2a1_dataset.csv'} n={len(df)}")

    model, tokenizer = load_model_and_tokenizer(cfg)

    feat_df = extract_all_features(df, model, tokenizer, cfg)
    feat_df.to_csv(save_dir / "sem2a1_features.csv", index=False, encoding="utf-8-sig")
    print(f"[SEM-2A.1] Features saved: {save_dir / 'sem2a1_features.csv'} shape={feat_df.shape}")

    result_df, baseline_df = run_all_evaluations(feat_df, cfg)
    result_df.to_csv(save_dir / "sem2a1_window_ablation.csv", index=False, encoding="utf-8-sig")
    baseline_df.to_csv(save_dir / "sem2a1_baseline_comparison.csv", index=False, encoding="utf-8-sig")

    rank_df = constraint_rank_audit(feat_df, cfg)
    rank_df.to_csv(save_dir / "sem2a1_constraint_rank.csv", index=False, encoding="utf-8-sig")

    if cfg.save_feature_importance:
        imps = []
        all_cols = get_feature_columns(feat_df, "all")
        for target in ["concept", "constraint", "seed_id"]:
            print(f"[SEM-2A.1] Feature importance for {target}")
            imps.append(feature_importance_rf(feat_df, target, all_cols, cfg).head(150))
        imp_df = pd.concat(imps, ignore_index=True)
        imp_df.to_csv(save_dir / "sem2a1_feature_importance.csv", index=False, encoding="utf-8-sig")

    summary = summarize_results(result_df, baseline_df, rank_df, cfg)
    json_dump(summary, save_dir / "sem2a1_results_summary.json")

    print("\n" + "=" * 100)
    print("[SEM-2A.1] SUMMARY")
    print("=" * 100)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n[SEM-2A.1] Output files:")
    for p in [
        "sem2a1_config.json",
        "sem2a1_dataset.csv",
        "sem2a1_features.csv",
        "sem2a1_window_ablation.csv",
        "sem2a1_baseline_comparison.csv",
        "sem2a1_constraint_rank.csv",
        "sem2a1_feature_importance.csv",
        "sem2a1_results_summary.json",
    ]:
        print("  -", save_dir / p)


if __name__ == "__main__":
    main()
