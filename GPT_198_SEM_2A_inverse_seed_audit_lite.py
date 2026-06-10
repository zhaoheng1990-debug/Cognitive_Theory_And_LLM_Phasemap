# -*- coding: utf-8 -*-
"""
GPT_196_SEM_2A_inverse_seed_audit_lite.py

SEM-2A: Inverse Seed Audit Lite
--------------------------------
Goal:
    Test whether internal trajectory signatures can recover Seed=(Concept, Constraint).

Core hypothesis:
    Prompt = Concept + Constraint
    Concept   -> x0
    Constraint-> Sigma0

    If trajectory signatures preserve Seed information, then:
        Trajectory -> Concept
        Trajectory -> Constraint
        Trajectory -> (Concept, Constraint)

Default:
    Single-model first pass on Qwen2.5-1.5B-Instruct.
    Model path is hard-coded per project convention.

Outputs:
    sem2a_outputs/
        sem2a_dataset.csv
        sem2a_features.csv
        sem2a_results_summary.json
        sem2a_window_ablation.csv
        sem2a_baseline_comparison.csv
        sem2a_feature_importance.csv

Notes:
    - This script uses hidden states at the final input token position.
    - TopK is computed as TopK(H_l W^T), i.e. vocabulary-induced semantic neighborhood.
    - It does NOT require generation; it audits internal trajectory signatures.
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
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# ============================================================
# CONFIG
# ============================================================

@dataclass
class CFG:
    # Hard-coded model path from GPT_195_ASA_9.py
    model_key: str = "qwen"
    model_path: str = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

    save_dir: str = "./sem2a_outputs"

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "auto"  # "auto", "float16", "bfloat16", "float32"

    # Extraction
    topk: int = 100
    batch_size: int = 4
    max_length: int = 256

    # Qwen 28-layer convention:
    # hidden_states[0] = embedding output
    # hidden_states[1] = after block 0
    # ...
    # This script renames block layers as L0-L27, i.e. hidden_states[layer+1].
    init_layers: Tuple[int, ...] = tuple(range(0, 7))       # L0-L6
    mid_layers: Tuple[int, ...] = tuple(range(7, 20))       # L7-L19
    decision_layers: Tuple[int, ...] = tuple(range(20, 26)) # L20-L25
    full_layers: Tuple[int, ...] = tuple(range(0, 26))      # L0-L25

    # CV
    n_splits: int = 4
    random_state: int = 42

    # Quick controls
    use_random_forest: bool = True
    save_feature_importance: bool = True


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

def build_sem2a_dataset() -> pd.DataFrame:
    """
    6 concepts x 4 constraints x 4 paraphrases = 96 prompts.
    Uses 4 paraphrase styles shared across all seeds for leave-one-style-out audit.
    """
    concepts = {
        "Paris": {
            "definition": [
                "What is Paris?",
                "Explain what Paris is.",
                "Give a concise definition of Paris.",
                "Describe Paris in one sentence.",
            ],
            "relation": [
                "Which country is Paris located in?",
                "Paris belongs to which country?",
                "What is the national relation of Paris?",
                "Identify the country associated with Paris.",
            ],
            "property": [
                "What are notable features of Paris?",
                "Describe the main characteristics of Paris.",
                "What is Paris known for?",
                "List important properties of Paris.",
            ],
            "list": [
                "Name famous landmarks in Paris.",
                "List well-known places in Paris.",
                "What are major attractions in Paris?",
                "Give examples of famous Paris landmarks.",
            ],
        },
        "Tokyo": {
            "definition": [
                "What is Tokyo?",
                "Explain what Tokyo is.",
                "Give a concise definition of Tokyo.",
                "Describe Tokyo in one sentence.",
            ],
            "relation": [
                "Which country is Tokyo located in?",
                "Tokyo belongs to which country?",
                "What is the national relation of Tokyo?",
                "Identify the country associated with Tokyo.",
            ],
            "property": [
                "What are notable features of Tokyo?",
                "Describe the main characteristics of Tokyo.",
                "What is Tokyo known for?",
                "List important properties of Tokyo.",
            ],
            "list": [
                "Name famous landmarks in Tokyo.",
                "List well-known places in Tokyo.",
                "What are major attractions in Tokyo?",
                "Give examples of famous Tokyo landmarks.",
            ],
        },
        "Newton": {
            "definition": [
                "Who was Newton?",
                "Explain who Newton was.",
                "Give a concise definition of Newton as a historical figure.",
                "Describe Newton in one sentence.",
            ],
            "relation": [
                "What field is Newton most associated with?",
                "Newton belongs to which area of science?",
                "Identify the scientific domain associated with Newton.",
                "What is Newton's relation to physics?",
            ],
            "property": [
                "What are notable features of Newton's work?",
                "Describe the main characteristics of Newton's contributions.",
                "What is Newton known for?",
                "List important properties of Newton's scientific legacy.",
            ],
            "list": [
                "Name famous discoveries associated with Newton.",
                "List well-known ideas from Newton.",
                "What are major contributions of Newton?",
                "Give examples of Newton's scientific achievements.",
            ],
        },
        "Python": {
            "definition": [
                "What is Python in computing?",
                "Explain what the Python programming language is.",
                "Give a concise definition of Python as a programming language.",
                "Describe Python in one sentence.",
            ],
            "relation": [
                "Python belongs to which category of technology?",
                "What is Python's relation to programming?",
                "Identify the software domain associated with Python.",
                "What kind of language is Python?",
            ],
            "property": [
                "What are notable features of Python?",
                "Describe the main characteristics of Python.",
                "What is Python known for?",
                "List important properties of the Python language.",
            ],
            "list": [
                "Name common uses of Python.",
                "List well-known applications of Python.",
                "What are major tasks people use Python for?",
                "Give examples of Python use cases.",
            ],
        },
        "Apple": {
            "definition": [
                "What is an apple?",
                "Explain what an apple is.",
                "Give a concise definition of apple as a fruit.",
                "Describe an apple in one sentence.",
            ],
            "relation": [
                "An apple belongs to which food category?",
                "What is the biological category of an apple?",
                "Identify the category associated with apples.",
                "What is the relation between apple and fruit?",
            ],
            "property": [
                "What are notable features of an apple?",
                "Describe the main characteristics of apples.",
                "What are apples known for?",
                "List important properties of apples.",
            ],
            "list": [
                "Name common varieties of apples.",
                "List well-known types of apples.",
                "What are major examples of apple varieties?",
                "Give examples of common apples.",
            ],
        },
        "Transformer": {
            "definition": [
                "What is a Transformer model?",
                "Explain what a Transformer is in machine learning.",
                "Give a concise definition of Transformer architecture.",
                "Describe a Transformer model in one sentence.",
            ],
            "relation": [
                "Transformer models belong to which area of AI?",
                "What is the relation between Transformers and neural networks?",
                "Identify the machine learning category associated with Transformers.",
                "What kind of architecture is a Transformer?",
            ],
            "property": [
                "What are notable features of Transformer models?",
                "Describe the main characteristics of Transformers.",
                "What are Transformers known for?",
                "List important properties of Transformer architecture.",
            ],
            "list": [
                "Name common applications of Transformers.",
                "List well-known uses of Transformer models.",
                "What are major tasks Transformers are used for?",
                "Give examples of Transformer-based applications.",
            ],
        },
    }

    rows = []
    for concept, cdict in concepts.items():
        for constraint, prompts in cdict.items():
            assert len(prompts) == 4
            for style_id, prompt in enumerate(prompts):
                seed_id = f"{concept}__{constraint}"
                rows.append({
                    "prompt_id": len(rows),
                    "concept": concept,
                    "constraint": constraint,
                    "seed_id": seed_id,
                    "style_id": style_id,
                    "prompt": prompt,
                })

    return pd.DataFrame(rows)


# ============================================================
# MODEL LOADING
# ============================================================

def load_model_and_tokenizer(cfg: CFG):
    print(f"[SEM-2A] Loading model: {cfg.model_path}")
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
    """
    Returns W with shape [vocab, hidden].
    For tied embeddings this may be embed_tokens.
    """
    if hasattr(model, "get_output_embeddings") and model.get_output_embeddings() is not None:
        W = model.get_output_embeddings().weight
        return W
    raise RuntimeError("Could not find output embedding / lm_head weight.")


# ============================================================
# FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_batch_features(
    prompts: List[str],
    model,
    tokenizer,
    cfg: CFG,
) -> List[Dict[str, Any]]:
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
    # hidden_states[0] embedding, hidden_states[1] after block 0, etc.
    W = get_lm_head_weight(model)
    W = W.to(model.device)

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

        # only block layers, cfg.full_layers uses L indices -> hidden_states[L+1]
        for layer in cfg.full_layers:
            hs_index = layer + 1
            if hs_index >= len(hidden_states):
                continue

            h = hidden_states[hs_index][bi, last_pos[bi], :]  # [hidden]
            logits = torch.matmul(h, W.T)  # [vocab]
            vals, ids = torch.topk(logits, k=cfg.topk, dim=-1)

            ids_np = ids.detach().cpu().numpy().astype(np.int64)
            vals_np = vals.detach().float().cpu().numpy()
            emb = W[ids].detach().float()  # [k, hidden]
            center = emb.mean(dim=0)
            center_np = center.cpu().numpy()

            # spread as mean 1-cos to center
            emb_norm = torch.nn.functional.normalize(emb, dim=-1)
            c_norm = torch.nn.functional.normalize(center.unsqueeze(0), dim=-1)
            cos = (emb_norm * c_norm).sum(dim=-1)
            spread = float((1.0 - cos).mean().detach().cpu())

            # entropy over topk logits
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
            else:
                per_layer[f"{prefix}_center_shift_cosdist"] = 0.0

            if prev_set is not None:
                per_layer[f"{prefix}_topk_jaccard_prev"] = jaccard(prev_set, id_set)
                per_layer[f"{prefix}_topk_jaccard_dist_prev"] = 1.0 - jaccard(prev_set, id_set)
            else:
                per_layer[f"{prefix}_topk_jaccard_prev"] = 1.0
                per_layer[f"{prefix}_topk_jaccard_dist_prev"] = 0.0

            prev_center = center_np
            prev_set = id_set

        # Window summary features
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

            # trajectory distances inside the window
            shift_vals = []
            jacdist_vals = []
            for l in valid[1:]:
                shift_vals.append(per_layer.get(f"L{l}_center_shift_cosdist", np.nan))
                jacdist_vals.append(per_layer.get(f"L{l}_topk_jaccard_dist_prev", np.nan))
            per_layer[f"{name}_center_shift_mean"] = safe_mean(shift_vals)
            per_layer[f"{name}_center_shift_std"] = safe_std(shift_vals)
            per_layer[f"{name}_jaccard_dist_mean"] = safe_mean(jacdist_vals)
            per_layer[f"{name}_jaccard_dist_std"] = safe_std(jacdist_vals)

            # start-end direction
            if len(valid) >= 2:
                start = centers[valid[0]]
                end = centers[valid[-1]]
                per_layer[f"{name}_start_end_cosdist"] = 1.0 - cosine_np(start, end)
                per_layer[f"{name}_start_end_l2"] = float(np.linalg.norm(end - start))
            else:
                per_layer[f"{name}_start_end_cosdist"] = 0.0
                per_layer[f"{name}_start_end_l2"] = 0.0

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
        batch_prompts = prompts[start:end]
        print(f"[SEM-2A] Feature extraction batch {start}:{end} / {len(prompts)}")
        feats = extract_batch_features(batch_prompts, model, tokenizer, cfg)
        all_features.extend(feats)

    feat_df = pd.DataFrame(all_features)
    out = pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
    return out


# ============================================================
# MODELING
# ============================================================

def get_feature_columns(df: pd.DataFrame, window: str = "all") -> List[str]:
    exclude = {"prompt_id", "concept", "constraint", "seed_id", "style_id", "prompt"}
    numeric_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        numeric_cols.append(c)

    if window == "all":
        return numeric_cols
    if window == "init":
        return [c for c in numeric_cols if c.startswith("init_") or c.startswith("L0_") or c.startswith("L1_") or c.startswith("L2_") or c.startswith("L3_") or c.startswith("L4_") or c.startswith("L5_") or c.startswith("L6_")]
    if window == "mid":
        prefixes = [f"L{i}_" for i in range(7, 20)] + ["mid_"]
        return [c for c in numeric_cols if any(c.startswith(p) for p in prefixes)]
    if window == "decision":
        prefixes = [f"L{i}_" for i in range(20, 26)] + ["decision_"]
        return [c for c in numeric_cols if any(c.startswith(p) for p in prefixes)]
    if window == "full_summary":
        return [c for c in numeric_cols if c.startswith("full_") or c.startswith("init_") or c.startswith("mid_") or c.startswith("decision_")]
    raise ValueError(f"Unknown window: {window}")


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

    if model_name == "logreg":
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                class_weight="balanced",
            )),
        ])
    elif model_name == "ridge":
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RidgeClassifier(class_weight="balanced")),
        ])
    elif model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            random_state=cfg.random_state,
            class_weight="balanced_subsample",
            max_depth=None,
            n_jobs=-1,
        )
    else:
        raise ValueError(model_name)

    if cv_mode == "stratified":
        cv = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
        splits = list(cv.split(X, y))
    elif cv_mode == "group":
        if group_col is None:
            raise ValueError("group_col required for group CV")
        groups = df[group_col].values
        cv = GroupKFold(n_splits=min(cfg.n_splits, len(np.unique(groups))))
        splits = list(cv.split(X, y, groups=groups))
    else:
        raise ValueError(cv_mode)

    y_true_all = []
    y_pred_all = []

    for fold, (tr, te) in enumerate(splits):
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        y_true_all.extend(y[te].tolist())
        y_pred_all.extend(pred.tolist())

    acc = accuracy_score(y_true_all, y_pred_all)
    macro_f1 = f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0)

    result = {
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
    }
    return result


def evaluate_surface_baseline(df: pd.DataFrame, target_col: str, cv_mode: str, group_col: str = None, cfg: CFG = cfg):
    y_raw = df[target_col].astype(str).values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    prompts = df["prompt"].values

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
        ("clf", LinearSVC(class_weight="balanced", max_iter=5000)),
    ])

    if cv_mode == "stratified":
        cv = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
        splits = list(cv.split(prompts, y))
    elif cv_mode == "group":
        groups = df[group_col].values
        cv = GroupKFold(n_splits=min(cfg.n_splits, len(np.unique(groups))))
        splits = list(cv.split(prompts, y, groups=groups))
    else:
        raise ValueError(cv_mode)

    y_true_all, y_pred_all = [], []
    for tr, te in splits:
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
    }


def evaluate_layer_shuffle(df: pd.DataFrame, target_col: str, cfg: CFG):
    """
    Simple layer-shuffle baseline:
    shuffle numeric layer-specific columns across layer indices within each row
    while preserving window summaries. This is an approximate control.
    """
    random_state = np.random.RandomState(cfg.random_state)
    shuffled = df.copy()

    layer_groups = {}
    for c in df.columns:
        if c.startswith("L") and "_" in c:
            layer_prefix = c.split("_")[0]
            suffix = c[len(layer_prefix):]
            layer_groups.setdefault(suffix, []).append(c)

    for suffix, cols in layer_groups.items():
        # Only shuffle among existing layer columns with same suffix.
        cols_sorted = sorted(cols, key=lambda x: int(x.split("_")[0][1:]))
        vals = shuffled[cols_sorted].values.copy()
        for i in range(vals.shape[0]):
            perm = random_state.permutation(vals.shape[1])
            vals[i, :] = vals[i, perm]
        shuffled[cols_sorted] = vals

    feature_cols = get_feature_columns(shuffled, "all")
    return evaluate_classifier_cv(
        shuffled,
        feature_cols=feature_cols,
        target_col=target_col,
        model_name="ridge",
        cv_mode="stratified",
        group_col=None,
        cfg=cfg,
    )


def run_all_evaluations(feat_df: pd.DataFrame, cfg: CFG) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Combined seed label already in df as seed_id.
    targets = ["concept", "constraint", "seed_id"]
    windows = ["init", "mid", "decision", "full_summary", "all"]
    model_names = ["ridge", "logreg"] + (["rf"] if cfg.use_random_forest else [])

    results = []

    for target in targets:
        for window in windows:
            feature_cols = get_feature_columns(feat_df, window)
            if len(feature_cols) == 0:
                continue

            for model_name in model_names:
                print(f"[SEM-2A] Eval target={target}, window={window}, model={model_name}, stratified")
                r = evaluate_classifier_cv(feat_df, feature_cols, target, model_name, "stratified", None, cfg)
                r["window"] = window
                r["feature_type"] = "trajectory"
                results.append(r)

                # Leave-one-style-out: group by style_id
                print(f"[SEM-2A] Eval target={target}, window={window}, model={model_name}, group=style_id")
                rg = evaluate_classifier_cv(feat_df, feature_cols, target, model_name, "group", "style_id", cfg)
                rg["window"] = window
                rg["feature_type"] = "trajectory"
                results.append(rg)

    # Baselines
    baselines = []
    for target in targets:
        print(f"[SEM-2A] Surface baseline target={target}")
        baselines.append(evaluate_surface_baseline(feat_df, target, "stratified", None, cfg))
        baselines[-1]["feature_type"] = "surface_only"
        baselines[-1]["window"] = "prompt_text"

        baselines.append(evaluate_surface_baseline(feat_df, target, "group", "style_id", cfg))
        baselines[-1]["feature_type"] = "surface_only"
        baselines[-1]["window"] = "prompt_text"

        print(f"[SEM-2A] Layer-shuffle baseline target={target}")
        ls = evaluate_layer_shuffle(feat_df, target, cfg)
        ls["feature_type"] = "layer_shuffled_trajectory"
        ls["window"] = "all"
        baselines.append(ls)

    result_df = pd.DataFrame(results)
    baseline_df = pd.DataFrame(baselines)
    return result_df, baseline_df


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
    imp = pd.DataFrame({
        "target": target,
        "feature": feature_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    return imp


def summarize_results(result_df: pd.DataFrame, baseline_df: pd.DataFrame, cfg: CFG) -> Dict[str, Any]:
    combined = pd.concat([result_df, baseline_df], ignore_index=True, sort=False)

    summary = {
        "config": asdict(cfg),
        "best_by_target": {},
        "pass_lite_thresholds": {
            "concept_accuracy": 0.75,
            "constraint_accuracy": 0.65,
            "seed_id_accuracy": 0.45,
        },
        "verdict": "UNDETERMINED",
        "notes": [],
    }

    for target in ["concept", "constraint", "seed_id"]:
        sub = combined[(combined["target"] == target) & (combined["feature_type"] == "trajectory")]
        if len(sub):
            best = sub.sort_values(["accuracy", "macro_f1"], ascending=False).iloc[0].to_dict()
            summary["best_by_target"][target] = best

    concept_acc = summary["best_by_target"].get("concept", {}).get("accuracy", 0.0)
    constraint_acc = summary["best_by_target"].get("constraint", {}).get("accuracy", 0.0)
    seed_acc = summary["best_by_target"].get("seed_id", {}).get("accuracy", 0.0)

    pass_lite = (
        concept_acc >= 0.75 and
        constraint_acc >= 0.65 and
        seed_acc >= 0.45
    )

    pass_strong = (
        concept_acc >= 0.90 and
        constraint_acc >= 0.80 and
        seed_acc >= 0.65
    )

    if pass_strong:
        summary["verdict"] = "PASS_STRONG"
    elif pass_lite:
        summary["verdict"] = "PASS_LITE"
    else:
        summary["verdict"] = "NO_PASS_YET"

    # Compare trajectory vs surface baseline best
    for target in ["concept", "constraint", "seed_id"]:
        traj = combined[(combined["target"] == target) & (combined["feature_type"] == "trajectory")]
        surf = combined[(combined["target"] == target) & (combined["feature_type"] == "surface_only")]
        if len(traj) and len(surf):
            best_traj = float(traj["accuracy"].max())
            best_surf = float(surf["accuracy"].max())
            summary[f"{target}_best_traj_acc"] = best_traj
            summary[f"{target}_best_surface_acc"] = best_surf
            summary[f"{target}_traj_minus_surface"] = best_traj - best_surf

    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    warnings.filterwarnings("ignore")
    set_seed(cfg.seed)
    ensure_dir(cfg.save_dir)
    save_dir = Path(cfg.save_dir)

    print("=" * 90)
    print("SEM-2A: Inverse Seed Audit Lite")
    print("=" * 90)
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))

    # Save config
    json_dump(asdict(cfg), save_dir / "sem2a_config.json")

    # Dataset
    df = build_sem2a_dataset()
    df.to_csv(save_dir / "sem2a_dataset.csv", index=False, encoding="utf-8-sig")
    print(f"[SEM-2A] Dataset saved: {save_dir / 'sem2a_dataset.csv'} n={len(df)}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Extract features
    feat_df = extract_all_features(df, model, tokenizer, cfg)
    feat_df.to_csv(save_dir / "sem2a_features.csv", index=False, encoding="utf-8-sig")
    print(f"[SEM-2A] Features saved: {save_dir / 'sem2a_features.csv'} shape={feat_df.shape}")

    # Evaluations
    result_df, baseline_df = run_all_evaluations(feat_df, cfg)

    result_df.to_csv(save_dir / "sem2a_window_ablation.csv", index=False, encoding="utf-8-sig")
    baseline_df.to_csv(save_dir / "sem2a_baseline_comparison.csv", index=False, encoding="utf-8-sig")

    # Feature importance
    if cfg.save_feature_importance:
        imps = []
        all_cols = get_feature_columns(feat_df, "all")
        for target in ["concept", "constraint", "seed_id"]:
            print(f"[SEM-2A] Feature importance for {target}")
            imps.append(feature_importance_rf(feat_df, target, all_cols, cfg).head(100))
        imp_df = pd.concat(imps, ignore_index=True)
        imp_df.to_csv(save_dir / "sem2a_feature_importance.csv", index=False, encoding="utf-8-sig")

    # Summary
    summary = summarize_results(result_df, baseline_df, cfg)
    json_dump(summary, save_dir / "sem2a_results_summary.json")

    # Human-readable summary
    print("\n" + "=" * 90)
    print("[SEM-2A] SUMMARY")
    print("=" * 90)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n[SEM-2A] Output files:")
    for p in [
        "sem2a_config.json",
        "sem2a_dataset.csv",
        "sem2a_features.csv",
        "sem2a_window_ablation.csv",
        "sem2a_baseline_comparison.csv",
        "sem2a_feature_importance.csv",
        "sem2a_results_summary.json",
    ]:
        print("  -", save_dir / p)


if __name__ == "__main__":
    main()
