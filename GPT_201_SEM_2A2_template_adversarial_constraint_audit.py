# -*- coding: utf-8 -*-
"""
GPT_198_SEM_2A2_template_adversarial_constraint_audit.py

SEM-2A.2: Template-Adversarial Constraint Audit
------------------------------------------------
Goal:
    Specifically attack surface/template leakage in SEM-2A.1.

Main question:
    Does trajectory signature recover Constraint/PolicyID under adversarial surface control?

Core hypothesis:
    Prompt = Concept + PolicyID
    Concept  -> high-dimensional x0
    PolicyID -> low-rank direction-spectrum / navigation policy

Compared with SEM-2A.1:
    1. Keep 24 concepts and 10 constraints.
    2. Use 8 surface families per constraint.
    3. Include "shared_frame" styles where multiple constraints use nearly identical sentence shells.
    4. Include adversarial generic wrappers that reduce obvious lexical cueing.
    5. Evaluate:
        - Stratified CV
        - Leave-One-Surface-Family-Out
        - Leave-One-Concept-Out for constraint recovery
        - Surface-only baseline
        - Layer-shuffled trajectory baseline
        - Constraint rank audit after concept quotient
    6. Main success criterion:
        constraint_group_style_id trajectory acc > 0.70
        and trajectory > surface baseline
        and rank90 <= 6~8.

Default:
    Qwen only, hard-coded path.

Outputs:
    sem2a2_outputs/
        sem2a2_dataset.csv
        sem2a2_features.csv
        sem2a2_results_summary.json
        sem2a2_window_ablation.csv
        sem2a2_baseline_comparison.csv
        sem2a2_constraint_rank.csv
        sem2a2_feature_importance.csv

Run:
    python GPT_198_SEM_2A2_template_adversarial_constraint_audit.py
"""

import os
import json
import random
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Any

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
    model_path: str = r"D:\model\models--Qwen--Qwen2.5B-Instruct\main"  # fallback handled below
    model_path_alt: str = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

    save_dir: str = "./sem2a2_outputs"

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
    # Set max_rows for smoke test; 0 = all.
    max_rows: int = 0

    # If true, only focus on constraint target to speed up modeling.
    # Feature extraction cost is unchanged, but evaluation is shorter.
    constraint_focus_only: bool = False


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

# 8 surface families.
# Important design:
# - style 0/1 are more natural and label-like.
# - style 2/3 use shared shells across constraints.
# - style 4/5 reduce explicit lexical cues by using operation-neutral language.
# - style 6/7 are meta/adversarial wrappers.
#
# This does NOT remove all lexical leakage. It is a stronger control than 2A.1,
# not a final adversarial benchmark.
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


def build_sem2a2_dataset() -> pd.DataFrame:
    rows = []
    for concept in CONCEPTS:
        for constraint in CONSTRAINTS:
            prompts = TEMPLATES[constraint]
            assert len(prompts) == 8
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
# MODEL LOADING
# ============================================================

def resolve_model_path(cfg: CFG) -> str:
    p = Path(cfg.model_path)
    if p.exists():
        return str(p)
    p2 = Path(cfg.model_path_alt)
    if p2.exists():
        return str(p2)
    # Return alt anyway because Windows path may not exist when inspected from non-Windows shell,
    # but will exist on user's machine.
    return cfg.model_path_alt


def load_model_and_tokenizer(cfg: CFG):
    model_path = resolve_model_path(cfg)
    print(f"[SEM-2A.2] Loading model: {model_path}")
    dtype = get_torch_dtype(cfg.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
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
def extract_batch_features(prompts: List[str], model, tokenizer, cfg: CFG) -> List[dict]:
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
        print(f"[SEM-2A.2] Feature extraction batch {start}:{end} / {len(prompts)}")
        feats = extract_batch_features(prompts[start:end], model, tokenizer, cfg)
        all_features.extend(feats)

    feat_df = pd.DataFrame(all_features)
    return pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)


# ============================================================
# MODELING
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


def get_splits(X, y, df, cv_mode, group_col, cfg):
    if cv_mode == "stratified":
        min_count = np.min(np.bincount(y))
        n_splits = max(2, min(cfg.n_splits, int(min_count)))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)
        return list(cv.split(X, y))

    if cv_mode == "group":
        groups = df[group_col].values
        n_groups = len(np.unique(groups))
        cv = GroupKFold(n_splits=min(cfg.n_splits, n_groups))
        return list(cv.split(X, y, groups=groups))

    raise ValueError(cv_mode)


def evaluate_classifier_cv(df, feature_cols, target_col, model_name, cv_mode, group_col=None, cfg=cfg):
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32)
    y_raw = df[target_col].astype(str).values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    if len(np.unique(y)) < 2:
        return {"error": "target has <2 classes", "target": target_col}

    clf = make_classifier(model_name, cfg)
    splits = get_splits(X, y, df, cv_mode, group_col, cfg)

    y_true_all, y_pred_all = [], []
    for tr, te in splits:
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

    return {
        "target": target_col,
        "model": model_name,
        "cv_mode": cv_mode,
        "group_col": group_col,
        "n": int(len(df)),
        "n_classes": int(len(le.classes_)),
        "n_features": int(len(feature_cols)),
        "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
        "macro_f1": float(f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0)),
        "classes": le.classes_.tolist(),
        "confusion_matrix": confusion_matrix(y_true_all, y_pred_all).tolist(),
        "skipped": False,
    }


def evaluate_surface_baseline(df, target_col, cv_mode, group_col=None, cfg=cfg):
    y_raw = df[target_col].astype(str).values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    prompts = df["prompt"].values

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
        ("clf", LinearSVC(class_weight="balanced", max_iter=12000)),
    ])

    splits = get_splits(prompts, y, df, cv_mode, group_col, cfg)

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


def evaluate_layer_shuffle(df, target_col, cfg):
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


def run_all_evaluations(feat_df: pd.DataFrame, cfg: CFG):
    targets = ["constraint"] if cfg.constraint_focus_only else ["concept", "constraint", "seed_id"]
    windows = ["init", "mid", "decision", "full_summary", "all"]
    model_names = ["ridge", "logreg"] + (["rf"] if cfg.use_random_forest else [])

    results = []

    for target in targets:
        for window in windows:
            feature_cols = get_feature_columns(feat_df, window)
            if not feature_cols:
                continue

            for model_name in model_names:
                for cv_mode, group_col in [
                    ("stratified", None),
                    ("group", "style_id"),
                    ("group", "concept"),
                ]:
                    if target == "concept" and group_col == "concept":
                        continue
                    print(f"[SEM-2A.2] Eval target={target}, window={window}, model={model_name}, cv={cv_mode}, group={group_col}")
                    r = evaluate_classifier_cv(feat_df, feature_cols, target, model_name, cv_mode, group_col, cfg)
                    r["window"] = window
                    r["feature_type"] = "trajectory"
                    results.append(r)

    baselines = []
    for target in targets:
        for cv_mode, group_col in [
            ("stratified", None),
            ("group", "style_id"),
            ("group", "concept"),
        ]:
            if target == "concept" and group_col == "concept":
                continue
            print(f"[SEM-2A.2] Surface baseline target={target}, cv={cv_mode}, group={group_col}")
            b = evaluate_surface_baseline(feat_df, target, cv_mode, group_col, cfg)
            b["feature_type"] = "surface_only"
            b["window"] = "prompt_text"
            baselines.append(b)

        print(f"[SEM-2A.2] Layer-shuffle baseline target={target}")
        ls = evaluate_layer_shuffle(feat_df, target, cfg)
        ls["feature_type"] = "layer_shuffled_trajectory"
        ls["window"] = "all"
        baselines.append(ls)

    return pd.DataFrame(results), pd.DataFrame(baselines)


def feature_importance_rf(feat_df, target, feature_cols, cfg):
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


def constraint_rank_audit(feat_df, cfg):
    feature_cols = get_feature_columns(feat_df, "all")
    X0 = feat_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = pd.DataFrame(StandardScaler().fit_transform(X0), columns=feature_cols)

    residual = X.copy()
    # quotient concept means
    for concept in feat_df["concept"].unique():
        idx = feat_df["concept"] == concept
        residual.loc[idx, :] = X.loc[idx, :] - X.loc[idx, :].mean(axis=0)

    centroids, labels = [], []
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
            "rank_estimate": np.nan,
        })

    cumsum = np.cumsum(pca.explained_variance_ratio_)
    for threshold in [0.8, 0.9, 0.95]:
        k = int(np.searchsorted(cumsum, threshold) + 1)
        rows.append({
            "component": f"rank_for_{threshold}",
            "explained_variance_ratio": np.nan,
            "cumulative_explained_variance": threshold,
            "n_constraints": len(labels),
            "n_features": len(feature_cols),
            "rank_estimate": k,
        })

    return pd.DataFrame(rows)


def summarize_results(result_df, baseline_df, rank_df, cfg):
    combined = pd.concat([result_df, baseline_df], ignore_index=True, sort=False)

    summary = {
        "config": asdict(cfg),
        "best_by_target": {},
        "surface_control": {},
        "constraint_low_rank": {},
        "verdict": "UNDETERMINED",
        "thresholds": {
            "pass_lite": {
                "constraint_style_acc": 0.55,
                "constraint_style_margin_over_surface": 0.05,
                "constraint_concept_heldout_acc": 0.80,
                "constraint_rank90": 8,
            },
            "pass_strong": {
                "constraint_style_acc": 0.70,
                "constraint_style_margin_over_surface": 0.10,
                "constraint_concept_heldout_acc": 0.90,
                "constraint_rank90": 6,
            },
            "pass_milestone": {
                "constraint_style_acc": 0.80,
                "constraint_style_margin_over_surface": 0.15,
                "constraint_concept_heldout_acc": 0.95,
                "constraint_rank90": 5,
            }
        }
    }

    for target in sorted(combined["target"].dropna().unique()):
        sub = combined[
            (combined["target"] == target) &
            (combined["feature_type"] == "trajectory") &
            (combined["skipped"] != True)
        ]
        if len(sub):
            best = sub.sort_values(["accuracy", "macro_f1"], ascending=False).iloc[0].to_dict()
            summary["best_by_target"][target] = best

    for target in sorted(combined["target"].dropna().unique()):
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
                best_traj_row = traj.sort_values(["accuracy", "macro_f1"], ascending=False).iloc[0]
                best_traj_acc = float(best_traj_row["accuracy"])
                best_surf_acc = float(surf["accuracy"].max()) if len(surf) else None
                summary["surface_control"][key] = {
                    "best_traj_acc": best_traj_acc,
                    "best_traj_macro_f1": float(best_traj_row["macro_f1"]),
                    "best_traj_window": best_traj_row.get("window", None),
                    "best_traj_model": best_traj_row.get("model", None),
                    "best_surface_acc": best_surf_acc,
                    "traj_minus_surface": (best_traj_acc - best_surf_acc) if best_surf_acc is not None else None,
                }

    rank_rows = rank_df[rank_df["component"].astype(str).str.startswith("rank_for_")]
    for _, row in rank_rows.iterrows():
        summary["constraint_low_rank"][str(row["component"])] = int(row["rank_estimate"])

    c_style = summary["surface_control"].get("constraint_group_style_id", {}).get("best_traj_acc", 0.0)
    c_style_margin = summary["surface_control"].get("constraint_group_style_id", {}).get("traj_minus_surface", -999)
    c_concept = summary["surface_control"].get("constraint_group_concept", {}).get("best_traj_acc", 0.0)
    rank90 = summary["constraint_low_rank"].get("rank_for_0.9", 999)

    if c_style >= 0.80 and c_style_margin >= 0.15 and c_concept >= 0.95 and rank90 <= 5:
        summary["verdict"] = "PASS_MILESTONE_TEMPLATE_ADVERSARIAL_LOW_RANK"
    elif c_style >= 0.70 and c_style_margin >= 0.10 and c_concept >= 0.90 and rank90 <= 6:
        summary["verdict"] = "PASS_STRONG_TEMPLATE_ADVERSARIAL_LOW_RANK"
    elif c_style >= 0.55 and c_style_margin >= 0.05 and c_concept >= 0.80 and rank90 <= 8:
        summary["verdict"] = "PASS_LITE_TEMPLATE_ADVERSARIAL_LOW_RANK"
    elif c_concept >= 0.90 and rank90 <= 8:
        summary["verdict"] = "PARTIAL_PASS_CONCEPT_HELDOUT_LOW_RANK"
    elif c_style > 0.0 and c_style_margin > 0.0:
        summary["verdict"] = "PARTIAL_PASS_SURFACE_CONTROL"
    else:
        summary["verdict"] = "NO_PASS_YET"

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
    print("SEM-2A.2: Template-Adversarial Constraint Audit")
    print("=" * 100)
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    json_dump(asdict(cfg), save_dir / "sem2a2_config.json")

    df = build_sem2a2_dataset()
    df.to_csv(save_dir / "sem2a2_dataset.csv", index=False, encoding="utf-8-sig")
    print(f"[SEM-2A.2] Dataset saved: {save_dir / 'sem2a2_dataset.csv'} n={len(df)}")

    model, tokenizer = load_model_and_tokenizer(cfg)

    feat_df = extract_all_features(df, model, tokenizer, cfg)
    feat_df.to_csv(save_dir / "sem2a2_features.csv", index=False, encoding="utf-8-sig")
    print(f"[SEM-2A.2] Features saved: {save_dir / 'sem2a2_features.csv'} shape={feat_df.shape}")

    result_df, baseline_df = run_all_evaluations(feat_df, cfg)
    result_df.to_csv(save_dir / "sem2a2_window_ablation.csv", index=False, encoding="utf-8-sig")
    baseline_df.to_csv(save_dir / "sem2a2_baseline_comparison.csv", index=False, encoding="utf-8-sig")

    rank_df = constraint_rank_audit(feat_df, cfg)
    rank_df.to_csv(save_dir / "sem2a2_constraint_rank.csv", index=False, encoding="utf-8-sig")

    if cfg.save_feature_importance:
        imps = []
        all_cols = get_feature_columns(feat_df, "all")
        targets = ["constraint"] if cfg.constraint_focus_only else ["concept", "constraint", "seed_id"]
        for target in targets:
            print(f"[SEM-2A.2] Feature importance for {target}")
            imps.append(feature_importance_rf(feat_df, target, all_cols, cfg).head(150))
        imp_df = pd.concat(imps, ignore_index=True)
        imp_df.to_csv(save_dir / "sem2a2_feature_importance.csv", index=False, encoding="utf-8-sig")

    summary = summarize_results(result_df, baseline_df, rank_df, cfg)
    json_dump(summary, save_dir / "sem2a2_results_summary.json")

    print("\n" + "=" * 100)
    print("[SEM-2A.2] SUMMARY")
    print("=" * 100)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n[SEM-2A.2] Output files:")
    for p in [
        "sem2a2_config.json",
        "sem2a2_dataset.csv",
        "sem2a2_features.csv",
        "sem2a2_window_ablation.csv",
        "sem2a2_baseline_comparison.csv",
        "sem2a2_constraint_rank.csv",
        "sem2a2_feature_importance.csv",
        "sem2a2_results_summary.json",
    ]:
        print("  -", save_dir / p)


if __name__ == "__main__":
    main()
