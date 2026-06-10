# -*- coding: utf-8 -*-
r"""
SOB-V2B: Learned Internal Visibility Axis

Purpose
-------
SOB-V2 showed:
  1. Internal subject-state separability exists across all three models.
  2. But hand-built CategoryLogitMass visibility is not a stable universal observable.

SOB-V2B replaces hand-built visibility with learned internal axes.

Core objects
------------
eta_core:
  learned from S0/S1/S2/S3 to predict state_rank over core tasks.

eta_transfer:
  learned from S3 vs S4 over transfer tasks.

Main hypotheses
---------------
H1 Learned core axis exists:
    Internal features can predict core subject rank S0-S3 across held-out tasks.

H2 Learned transfer axis exists:
    Internal features can discriminate S3 vs S4 on transfer tasks.

H3 Learned axes outperform hand-built category-logit visibility:
    LearnedCoreCorr > CategoryCoreCorr
    LearnedTransferF1/AUC > CategoryTransferGap

H4 Axis structure is present in hidden / TopK features:
    TopK/VIM-only features recover a meaningful fraction of eta_core / eta_transfer.

Output
------
C:\Users\ZH\Desktop\AGI\outputs\SOB-V2B

Default models
--------------
Qwen:
  D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main

Llama:
  D:\model\Llama-3.2-1B-Instruct

Gemma:
  D:\model\gemma-2-2b-it
"""

import os
import re
import json
import time
import math
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ============================================================
# Constants
# ============================================================

EXPERIMENT_ID = "SOB-V2B"
OUT_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\SOB-V2B")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SPECS = {
    "qwen": {
        "model_name": "Qwen2.5-1.5B-Instruct",
        "path": r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main",
        "trust_remote_code": True,
    },
    "llama": {
        "model_name": "Llama-3.2-1B-Instruct",
        "path": r"D:\model\Llama-3.2-1B-Instruct",
        "trust_remote_code": True,
    },
    "gemma": {
        "model_name": "gemma-2-2b-it",
        "path": r"D:\model\gemma-2-2b-it",
        "trust_remote_code": True,
    },
}

MODELS_TO_RUN = ["qwen", "llama", "gemma"]

RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

MAX_INPUT_LEN = 1200
TOPK = 80
N_SPLITS = 4

CORE_STATES = ["S0", "S1", "S2", "S3"]
TRANSFER_STATES = ["S3", "S4"]

CORE_CATEGORIES = ["definition", "calculation", "geometry", "edge"]
TRANSFER_CATEGORIES = ["physics", "optimization", "machine_learning", "control", "new_domain"]
ALL_CATEGORIES = ["symbol"] + CORE_CATEGORIES + TRANSFER_CATEGORIES

SUBJECT_STATES = [
    {
        "state_id": "S0",
        "state_rank": 0,
        "state_name": "symbol_only",
        "profile": (
            "You have only seen the symbols d/dx and f'(x). "
            "You do not know what they mean. You cannot define derivative, compute derivatives, explain tangent slope, use limits, or apply it."
        ),
    },
    {
        "state_id": "S1",
        "state_rank": 1,
        "state_name": "definition_known",
        "profile": (
            "You know the derivative only as a formal limit definition and instantaneous rate of change. "
            "You cannot reliably compute derivatives, explain geometry, or transfer it to other domains."
        ),
    },
    {
        "state_id": "S2",
        "state_rank": 2,
        "state_name": "calculation_ability",
        "profile": (
            "You know the limit definition and can compute derivatives using power, product, quotient, and chain rules. "
            "Your geometric and transfer understanding is still limited."
        ),
    },
    {
        "state_id": "S3",
        "state_rank": 3,
        "state_name": "geometric_understanding",
        "profile": (
            "You know definition, calculation, tangent slope, local linear approximation, and edge cases such as corners and continuity versus differentiability."
        ),
    },
    {
        "state_id": "S4",
        "state_rank": 4,
        "state_name": "cross_domain_transfer",
        "profile": (
            "You understand derivative as first-order local change structure and can transfer it across physics, optimization, machine learning, control, economics, biology, and unfamiliar dynamic systems."
        ),
    },
]

TASKS = [
    # Core tasks
    {
        "task_id": "CORE_concept_map",
        "task_family": "core",
        "question": "Think about the derivative as an object. What internal concepts should become visible for this learner state?",
    },
    {
        "task_id": "CORE_edge_case",
        "task_family": "core",
        "question": "Think about differentiability, continuity, corners, and local linearity from this learner state.",
    },
    {
        "task_id": "CORE_compression",
        "task_family": "core",
        "question": "Think about how this learner state would compress derivative into a minimal explanation.",
    },
    {
        "task_id": "CORE_counterfactual",
        "task_family": "core",
        "question": "Think about what this learner state can infer if derivatives do not exist or fail at a point.",
    },

    # Transfer tasks
    {
        "task_id": "TRANSFER_physics",
        "task_family": "transfer",
        "question": "Think about whether this learner state can transfer derivative to position, velocity, and acceleration.",
    },
    {
        "task_id": "TRANSFER_optimization",
        "task_family": "transfer",
        "question": "Think about whether this learner state can transfer derivative to gradient descent and optimization.",
    },
    {
        "task_id": "TRANSFER_ml",
        "task_family": "transfer",
        "question": "Think about whether this learner state can transfer derivative to gradients and backpropagation in machine learning.",
    },
    {
        "task_id": "TRANSFER_control",
        "task_family": "transfer",
        "question": "Think about whether this learner state can transfer derivative to feedback, sensitivity, and control systems.",
    },
    {
        "task_id": "TRANSFER_newdomain",
        "task_family": "transfer",
        "question": "Think about whether this learner state can transfer derivative to economics, biology, or another unfamiliar dynamic system.",
    },
]

CATEGORY_TERMS = {
    "symbol": [
        "d/dx", "f'", "prime", "symbol", "notation", "derivative symbol"
    ],
    "definition": [
        "limit", "difference quotient", "h approaches zero", "instantaneous rate", "rate of change",
        "definition", "delta x", "delta y"
    ],
    "calculation": [
        "power rule", "product rule", "quotient rule", "chain rule", "differentiate",
        "compute", "derivative rule", "algebraic rule"
    ],
    "geometry": [
        "tangent", "slope", "secant", "local linear", "linear approximation",
        "first order", "line touches", "graph"
    ],
    "edge": [
        "continuous", "differentiable", "corner", "cusp", "left derivative",
        "right derivative", "absolute value", "not differentiable"
    ],
    "physics": [
        "position", "velocity", "acceleration", "motion", "time", "speed",
        "change of position", "change of velocity"
    ],
    "optimization": [
        "gradient", "gradient descent", "objective", "loss", "minimize",
        "steepest", "parameter update", "optimization"
    ],
    "machine_learning": [
        "backpropagation", "neural network", "weight", "loss function",
        "learning", "chain rule", "parameter", "error signal"
    ],
    "control": [
        "control", "controller", "feedback", "sensitivity", "stability",
        "state", "response", "adjustment"
    ],
    "new_domain": [
        "economics", "biology", "population", "growth rate", "marginal",
        "sensitivity", "dynamic system", "cost", "benefit"
    ],
}

# ============================================================
# Utilities
# ============================================================

def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def now_time():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def softmax_np(x):
    arr = np.asarray(x, dtype=np.float64)
    arr = arr - np.nanmax(arr)
    p = np.exp(arr)
    return p / (np.nansum(p) + 1e-12)


def entropy_bits(probs):
    p = np.asarray(probs, dtype=np.float64)
    p = p / (np.nansum(p) + 1e-12)
    return float(-np.nansum(p * np.log2(p + 1e-12)))


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / den)


def topk_indices(vals, k):
    vals = np.asarray(vals)
    k_eff = min(k, len(vals))
    idx = np.argpartition(-vals, k_eff - 1)[:k_eff]
    idx = idx[np.argsort(-vals[idx])]
    return idx


def center_vec(ids, vals, W):
    ids = np.asarray(ids, dtype=np.int64)
    vals = np.asarray(vals, dtype=np.float64)
    if ids.size == 0:
        return np.zeros(W.shape[1], dtype=np.float64)
    p = softmax_np(vals)
    return (W[ids] * p[:, None]).sum(axis=0)


def spread_vec(ids, W):
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size <= 1:
        return 0.0
    E = W[ids]
    c = E.mean(axis=0, keepdims=True)
    return float(np.mean(np.linalg.norm(E - c, axis=1)))


def numeric_cols(df, cols):
    out = []
    for c in cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def cols_by_prefix(df, prefixes):
    out = []
    for c in df.columns:
        for p in prefixes:
            if c.startswith(p):
                out.append(c)
                break
    return numeric_cols(df, out)


def unique_keep_order(xs):
    return list(dict.fromkeys(xs))


def build_prompt(subject_state, task):
    return (
        "Internal-state probe only. Do not produce an explanatory answer.\n\n"
        "Read the learner state and task, then silently form the internal representation that would be needed.\n\n"
        f"Learner state:\n{subject_state['profile']}\n\n"
        "Object of cognition: derivative.\n\n"
        f"Probe task:\n{task['question']}\n\n"
        "End with exactly this token: READY"
    )


def build_dataset():
    rows = []
    for st in SUBJECT_STATES:
        for task in TASKS:
            rows.append({
                "row_id": len(rows),
                "state_id": st["state_id"],
                "state_rank": st["state_rank"],
                "state_name": st["state_name"],
                "task_id": task["task_id"],
                "task_family": task["task_family"],
                "question": task["question"],
                "prompt": build_prompt(st, task),
            })
    return pd.DataFrame(rows)


# ============================================================
# ML utility
# ============================================================

def cv_indices(df, y=None, group_col="task_id", stratify=False):
    n = len(df)
    groups = df[group_col].values if group_col in df.columns else None
    if groups is not None and len(np.unique(groups)) >= min(N_SPLITS, n):
        ns = min(N_SPLITS, len(np.unique(groups)))
        cv = GroupKFold(n_splits=ns)
        y_dummy = np.zeros(n) if y is None else y
        return list(cv.split(np.zeros(n), y_dummy, groups=groups))

    if stratify and y is not None:
        vals, counts = np.unique(y, return_counts=True)
        ns = min(N_SPLITS, int(counts.min())) if len(counts) else 2
        if len(vals) >= 2 and ns >= 2:
            cv = StratifiedKFold(n_splits=ns, shuffle=True, random_state=RANDOM_SEED)
            return list(cv.split(np.zeros(n), y))

    ns = min(N_SPLITS, n)
    cv = KFold(n_splits=ns, shuffle=True, random_state=RANDOM_SEED)
    return list(cv.split(np.zeros(n)))


def make_binary_classifier():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=RANDOM_SEED)),
    ])


def make_regressor(alpha=1.0):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=alpha)),
    ])


def cv_regression_predictions(df, cols, target, group_col="task_id"):
    cols = numeric_cols(df, cols)
    d = df.dropna(subset=[target]).copy()
    if len(d) < 8 or not cols:
        return None
    y = d[target].astype(float).values
    X = d[cols].replace([np.inf, -np.inf], np.nan).values
    pred = np.zeros(len(d), dtype=float)
    used = 0
    for train_idx, test_idx in cv_indices(d, None, group_col=group_col):
        reg = make_regressor()
        reg.fit(X[train_idx], y[train_idx])
        pred[test_idx] = reg.predict(X[test_idx])
        used += 1
    if used == 0:
        return None
    corr = np.nan
    if np.std(y) > 1e-12 and np.std(pred) > 1e-12:
        corr = float(np.corrcoef(y, pred)[0, 1])
    return {
        "df": d,
        "y": y,
        "pred": pred,
        "r2": safe_float(r2_score(y, pred)),
        "corr": corr,
        "n": int(len(d)),
        "n_features": int(len(cols)),
    }


def cv_binary_predictions(df, cols, target, group_col="task_id"):
    cols = numeric_cols(df, cols)
    d = df.dropna(subset=[target]).copy()
    if len(d) < 8 or not cols:
        return None
    y = d[target].astype(int).values
    if len(np.unique(y)) < 2:
        return None
    X = d[cols].replace([np.inf, -np.inf], np.nan).values
    pred = np.zeros(len(d), dtype=int)
    prob = np.zeros(len(d), dtype=float)
    used = 0
    for train_idx, test_idx in cv_indices(d, y, group_col=group_col, stratify=True):
        if len(np.unique(y[train_idx])) < 2:
            continue
        clf = make_binary_classifier()
        clf.fit(X[train_idx], y[train_idx])
        p = clf.predict_proba(X[test_idx])[:, 1]
        prob[test_idx] = p
        pred[test_idx] = (p >= 0.5).astype(int)
        used += 1
    if used == 0:
        return None
    auc = np.nan
    try:
        auc = safe_float(roc_auc_score(y, prob))
    except Exception:
        pass
    return {
        "df": d,
        "y": y,
        "pred": pred,
        "prob": prob,
        "auc": auc,
        "acc": safe_float(accuracy_score(y, pred)),
        "macro_f1": safe_float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1": safe_float(f1_score(y, pred, zero_division=0)),
        "balanced_acc": safe_float(balanced_accuracy_score(y, pred)),
        "n": int(len(d)),
        "n_features": int(len(cols)),
    }


def train_full_regression_axis(df, cols, target):
    cols = numeric_cols(df, cols)
    d = df.dropna(subset=[target]).copy()
    if len(d) < 4 or not cols:
        return None
    X = d[cols].replace([np.inf, -np.inf], np.nan).values
    y = d[target].astype(float).values
    reg = make_regressor()
    reg.fit(X, y)
    pred_all = reg.predict(df[cols].replace([np.inf, -np.inf], np.nan).values)
    return pred_all


def train_full_binary_axis(df, cols, target):
    cols = numeric_cols(df, cols)
    d = df.dropna(subset=[target]).copy()
    if len(d) < 4 or not cols:
        return None
    y = d[target].astype(int).values
    if len(np.unique(y)) < 2:
        return None
    X = d[cols].replace([np.inf, -np.inf], np.nan).values
    clf = make_binary_classifier()
    clf.fit(X, y)
    prob_all = clf.predict_proba(df[cols].replace([np.inf, -np.inf], np.nan).values)[:, 1]
    return prob_all


# ============================================================
# Model helpers
# ============================================================

def load_model_and_tokenizer(path, trust_remote_code=True):
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
        torch_dtype=DTYPE,
        device_map=None,
    )
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def format_chat(tokenizer, prompt):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt
    return prompt


@torch.no_grad()
def forward_hidden(model, tokenizer, prompt):
    text = format_chat(tokenizer, prompt)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LEN)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    outputs = model(**enc, output_hidden_states=True, use_cache=False)
    hs = [h[:, -1, :].float().detach().cpu().numpy()[0] for h in outputs.hidden_states]
    return hs


def lm_head_weight(model):
    return model.get_output_embeddings().weight.detach().float().cpu().numpy()


def category_token_ids(tokenizer):
    out = {}
    for cat, terms in CATEGORY_TERMS.items():
        ids = []
        for term in terms:
            variants = [term, " " + term, "\n" + term]
            for v in variants:
                toks = tokenizer(v, add_special_tokens=False).input_ids
                if toks:
                    ids.append(toks[-1])
        out[cat] = sorted(set(ids))
    return out


def to_hidx(layer_num, n_hidden_states):
    return max(0, min(n_hidden_states - 1, layer_num + 1))


def span_layers(a, b, n_hidden_states):
    return [to_hidx(i, n_hidden_states) for i in range(max(0, a), max(a, b) + 1)]


def windows_for_model(model_key, n_hidden_states):
    n_layers = n_hidden_states - 1
    def rel(p):
        return int(round(n_layers * p))

    if model_key == "qwen":
        return {
            "init": span_layers(0, 6, n_hidden_states),
            "middle": span_layers(7, 19, n_hidden_states),
            "competition": span_layers(20, 22, n_hidden_states),
            "commit": span_layers(23, 25, n_hidden_states),
            "final": span_layers(27, 27, n_hidden_states) if n_layers >= 27 else span_layers(n_layers, n_layers, n_hidden_states),
        }
    elif model_key == "llama":
        return {
            "init": span_layers(0, 2, n_hidden_states),
            "middle": span_layers(rel(0.25), rel(0.62), n_hidden_states),
            "competition": span_layers(rel(0.63), rel(0.75), n_hidden_states),
            "commit": span_layers(rel(0.76), rel(0.92), n_hidden_states),
            "final": span_layers(n_layers, n_layers, n_hidden_states),
        }
    elif model_key == "gemma":
        return {
            "init": span_layers(0, 2, n_hidden_states),
            "middle": span_layers(7, 17, n_hidden_states) if n_layers >= 18 else span_layers(rel(0.25), rel(0.65), n_hidden_states),
            "competition": span_layers(18, 20, n_hidden_states) if n_layers >= 21 else span_layers(rel(0.66), rel(0.75), n_hidden_states),
            "commit": span_layers(21, 23, n_hidden_states) if n_layers >= 24 else span_layers(rel(0.76), rel(0.92), n_hidden_states),
            "final": span_layers(n_layers, n_hidden_states - 1, n_hidden_states),
        }
    else:
        return {
            "init": span_layers(0, rel(0.20), n_hidden_states),
            "middle": span_layers(rel(0.25), rel(0.65), n_hidden_states),
            "competition": span_layers(rel(0.66), rel(0.78), n_hidden_states),
            "commit": span_layers(rel(0.80), rel(0.92), n_hidden_states),
            "final": span_layers(n_layers, n_layers, n_hidden_states),
        }


def category_scores_from_hidden(h, W, cat_ids):
    logits = h @ W.T
    scores = {}
    for cat in ALL_CATEGORIES:
        ids = cat_ids.get(cat, [])
        if ids:
            scores[cat] = float(np.max(logits[ids]))
        else:
            scores[cat] = np.nan
    return scores


def hidden_summary_features(feat, wname, hvecs):
    H = np.asarray(hvecs, dtype=np.float64)
    if H.ndim != 2 or H.shape[0] == 0:
        return
    mean = H.mean(axis=0)
    std = H.std(axis=0)
    # Dimensionality can be large; store top norm / statistical summaries only.
    norms = np.linalg.norm(H, axis=1)
    feat[f"{wname}_hidden_norm_mean"] = float(np.mean(norms))
    feat[f"{wname}_hidden_norm_std"] = float(np.std(norms))
    feat[f"{wname}_hidden_abs_mean"] = float(np.mean(np.abs(H)))
    feat[f"{wname}_hidden_std_mean"] = float(np.mean(std))
    if H.shape[0] >= 2:
        drifts = [1.0 - cosine(H[i], H[i+1]) for i in range(H.shape[0]-1)]
        feat[f"{wname}_hidden_drift_mean"] = float(np.mean(drifts))
        feat[f"{wname}_hidden_drift_max"] = float(np.max(drifts))
    else:
        feat[f"{wname}_hidden_drift_mean"] = 0.0
        feat[f"{wname}_hidden_drift_max"] = 0.0

    # Store compact sampled dimensions for learned axis.
    # Deterministic projection by evenly spaced dimensions.
    d = H.shape[1]
    sample_n = min(96, d)
    idx = np.linspace(0, d - 1, sample_n).round().astype(int)
    for j, k in enumerate(idx):
        feat[f"{wname}_hmean_{j:03d}"] = float(mean[k])
        feat[f"{wname}_hstd_{j:03d}"] = float(std[k])


def extract_row_features(row, model, tokenizer, W, cat_ids, model_key):
    hs = forward_hidden(model, tokenizer, row["prompt"])
    windows = windows_for_model(model_key, len(hs))
    feat = {
        "n_hidden_states": len(hs),
        "n_layers": len(hs) - 1,
    }

    # Windowed category-logit visibility, TopK/VIM, hidden summaries.
    for wname, hidxs in windows.items():
        cat_seq = {cat: [] for cat in ALL_CATEGORIES}
        topk_centers = []
        topk_entropy_vals = []
        topk_spread_vals = []

        for hidx in hidxs:
            h = hs[hidx]
            scores = category_scores_from_hidden(h, W, cat_ids)
            for cat in ALL_CATEGORIES:
                cat_seq[cat].append(scores.get(cat, np.nan))

            logits = h @ W.T
            ids = topk_indices(logits, TOPK)
            vals = logits[ids]
            p = softmax_np(vals)
            topk_entropy_vals.append(entropy_bits(p))
            topk_spread_vals.append(spread_vec(ids, W))
            topk_centers.append(center_vec(ids, vals, W))

        # Category probabilities.
        cat_mat = np.array([[cat_seq[cat][i] for cat in ALL_CATEGORIES] for i in range(len(hidxs))], dtype=np.float64)
        cat_probs = []
        for i in range(cat_mat.shape[0]):
            row_scores = cat_mat[i]
            finite = np.isfinite(row_scores)
            if finite.sum() == 0:
                cat_probs.append(np.ones(len(ALL_CATEGORIES)) / len(ALL_CATEGORIES))
            else:
                row_scores = np.where(finite, row_scores, np.nanmin(row_scores[finite]) - 10.0)
                cat_probs.append(softmax_np(row_scores))
        cat_probs = np.array(cat_probs, dtype=np.float64)
        mean_probs = cat_probs.mean(axis=0)

        for j, cat in enumerate(ALL_CATEGORIES):
            vals = np.asarray(cat_seq[cat], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            feat[f"{wname}_catlogit_{cat}_mean"] = float(vals.mean()) if vals.size else np.nan
            feat[f"{wname}_catprob_{cat}_mean"] = float(mean_probs[j])

        feat[f"{wname}_cat_entropy_mean"] = float(np.mean([entropy_bits(p) for p in cat_probs]))
        feat[f"{wname}_core_mass"] = float(sum(mean_probs[ALL_CATEGORIES.index(c)] for c in CORE_CATEGORIES))
        feat[f"{wname}_transfer_mass"] = float(sum(mean_probs[ALL_CATEGORIES.index(c)] for c in TRANSFER_CATEGORIES))
        feat[f"{wname}_symbol_mass"] = float(mean_probs[ALL_CATEGORIES.index("symbol")])
        feat[f"{wname}_core_minus_symbol"] = feat[f"{wname}_core_mass"] - feat[f"{wname}_symbol_mass"]
        feat[f"{wname}_transfer_minus_core"] = feat[f"{wname}_transfer_mass"] - feat[f"{wname}_core_mass"]

        feat[f"{wname}_topk_entropy_mean"] = float(np.mean(topk_entropy_vals))
        feat[f"{wname}_topk_spread_mean"] = float(np.mean(topk_spread_vals))
        if len(topk_centers) >= 2:
            drifts = [1.0 - cosine(topk_centers[i], topk_centers[i+1]) for i in range(len(topk_centers)-1)]
            feat[f"{wname}_topk_center_drift_mean"] = float(np.mean(drifts))
            feat[f"{wname}_topk_center_drift_max"] = float(np.max(drifts))
        else:
            feat[f"{wname}_topk_center_drift_mean"] = 0.0
            feat[f"{wname}_topk_center_drift_max"] = 0.0

        hidden_summary_features(feat, wname, [hs[i] for i in hidxs])

    # Hand-built category visibility from V2.
    feat["V_core_category"] = float(np.nanmean([
        feat.get("commit_core_mass", np.nan),
        feat.get("middle_core_mass", np.nan),
    ]))
    feat["V_transfer_category"] = float(np.nanmean([
        feat.get("commit_transfer_mass", np.nan),
        feat.get("middle_transfer_mass", np.nan),
    ]))
    feat["V_symbol_category"] = float(np.nanmean([
        feat.get("init_symbol_mass", np.nan),
        feat.get("commit_symbol_mass", np.nan),
    ]))
    feat["V_all_category"] = float(np.nanmean([feat["V_core_category"], feat["V_transfer_category"]]))

    return feat


# ============================================================
# Feature groups and axis evaluation
# ============================================================

def build_feature_groups(df):
    category_cols = cols_by_prefix(df, [
        "init_catprob_", "middle_catprob_", "competition_catprob_", "commit_catprob_", "final_catprob_",
        "init_core_", "middle_core_", "commit_core_", "final_core_",
        "init_transfer_", "middle_transfer_", "commit_transfer_", "final_transfer_",
        "init_symbol_", "middle_symbol_", "commit_symbol_", "final_symbol_",
        "V_core_category", "V_transfer_category", "V_symbol_category", "V_all_category",
    ])
    topk_cols = cols_by_prefix(df, [
        "init_topk_", "middle_topk_", "competition_topk_", "commit_topk_", "final_topk_"
    ])
    hidden_cols = cols_by_prefix(df, [
        "init_hidden_", "middle_hidden_", "competition_hidden_", "commit_hidden_", "final_hidden_",
        "init_hmean_", "middle_hmean_", "competition_hmean_", "commit_hmean_", "final_hmean_",
        "init_hstd_", "middle_hstd_", "competition_hstd_", "commit_hstd_", "final_hstd_",
    ])
    all_cols = unique_keep_order(category_cols + topk_cols + hidden_cols)
    no_category_cols = unique_keep_order(topk_cols + hidden_cols)
    return {
        "CategoryOnly": category_cols,
        "TopKOnly": topk_cols,
        "HiddenSummary": hidden_cols,
        "TopKPlusHidden": no_category_cols,
        "AllInternal": all_cols,
    }


def state_axis_summary(features_df):
    groups = build_feature_groups(features_df)
    rows = []

    # Core axis: S0-S3 core tasks only.
    core_df = features_df[
        (features_df["task_family"] == "core") &
        (features_df["state_id"].isin(CORE_STATES))
    ].copy()

    for gname, cols in groups.items():
        res = cv_regression_predictions(core_df, cols, target="state_rank", group_col="task_id")
        if res is None:
            rows.append({
                "axis": "eta_core",
                "feature_group": gname,
                "task": "core_rank_regression",
                "valid": False,
            })
            continue
        rows.append({
            "axis": "eta_core",
            "feature_group": gname,
            "task": "core_rank_regression",
            "valid": True,
            "n": res["n"],
            "n_features": res["n_features"],
            "r2": res["r2"],
            "corr": res["corr"],
            "auc": np.nan,
            "acc": np.nan,
            "macro_f1": np.nan,
            "balanced_acc": np.nan,
        })

    # Transfer axis: S3 vs S4 transfer tasks only.
    trans_df = features_df[
        (features_df["task_family"] == "transfer") &
        (features_df["state_id"].isin(TRANSFER_STATES))
    ].copy()
    trans_df["is_S4"] = (trans_df["state_id"] == "S4").astype(int)

    for gname, cols in groups.items():
        res = cv_binary_predictions(trans_df, cols, target="is_S4", group_col="task_id")
        if res is None:
            rows.append({
                "axis": "eta_transfer",
                "feature_group": gname,
                "task": "S3_vs_S4_transfer",
                "valid": False,
            })
            continue
        rows.append({
            "axis": "eta_transfer",
            "feature_group": gname,
            "task": "S3_vs_S4_transfer",
            "valid": True,
            "n": res["n"],
            "n_features": res["n_features"],
            "r2": np.nan,
            "corr": np.nan,
            "auc": res["auc"],
            "acc": res["acc"],
            "macro_f1": res["macro_f1"],
            "balanced_acc": res["balanced_acc"],
        })

    return pd.DataFrame(rows), groups


def add_learned_axes(features_df, groups):
    df = features_df.copy()

    core_train = df[
        (df["task_family"] == "core") &
        (df["state_id"].isin(CORE_STATES))
    ].copy()

    transfer_train = df[
        (df["task_family"] == "transfer") &
        (df["state_id"].isin(TRANSFER_STATES))
    ].copy()
    transfer_train["is_S4"] = (transfer_train["state_id"] == "S4").astype(int)

    # Choose AllInternal for primary learned axis.
    all_cols = groups["AllInternal"]
    topk_hidden_cols = groups["TopKPlusHidden"]

    core_pred_all = train_full_regression_axis(core_train, all_cols, "state_rank")
    if core_pred_all is not None:
        # Careful: returned on core_train if passed core_train. Need train model manually on core_train and predict df.
        pass

    # Manual fit/predict full because train_full returned wrong target shape relative to df.
    def fit_core_predict(cols, suffix):
        cols = numeric_cols(df, cols)
        if not cols:
            df[f"eta_core_{suffix}"] = np.nan
            return
        tr = core_train.dropna(subset=["state_rank"]).copy()
        Xtr = tr[cols].replace([np.inf, -np.inf], np.nan).values
        ytr = tr["state_rank"].astype(float).values
        reg = make_regressor()
        reg.fit(Xtr, ytr)
        Xall = df[cols].replace([np.inf, -np.inf], np.nan).values
        df[f"eta_core_{suffix}"] = reg.predict(Xall)

    def fit_transfer_predict(cols, suffix):
        cols = numeric_cols(df, cols)
        if not cols:
            df[f"eta_transfer_{suffix}"] = np.nan
            return
        tr = transfer_train.dropna(subset=["is_S4"]).copy()
        ytr = tr["is_S4"].astype(int).values
        if len(np.unique(ytr)) < 2:
            df[f"eta_transfer_{suffix}"] = np.nan
            return
        Xtr = tr[cols].replace([np.inf, -np.inf], np.nan).values
        clf = make_binary_classifier()
        clf.fit(Xtr, ytr)
        Xall = df[cols].replace([np.inf, -np.inf], np.nan).values
        df[f"eta_transfer_{suffix}"] = clf.predict_proba(Xall)[:, 1]

    fit_core_predict(all_cols, "all")
    fit_core_predict(topk_hidden_cols, "topk_hidden")
    fit_transfer_predict(all_cols, "all")
    fit_transfer_predict(topk_hidden_cols, "topk_hidden")

    return df


def learned_axis_state_summary(axis_df):
    agg = (
        axis_df
        .groupby(["model_key", "state_id", "state_rank", "state_name"], as_index=False)
        .agg(
            eta_core_all=("eta_core_all", "mean"),
            eta_core_topk_hidden=("eta_core_topk_hidden", "mean"),
            eta_transfer_all=("eta_transfer_all", "mean"),
            eta_transfer_topk_hidden=("eta_transfer_topk_hidden", "mean"),
            V_core_category=("V_core_category", "mean"),
            V_transfer_category=("V_transfer_category", "mean"),
            n=("row_id", "size"),
        )
    )
    return agg.sort_values(["model_key", "state_rank"])


def pearson_state(sub, col, max_rank=None):
    d = sub.copy()
    if max_rank is not None:
        d = d[d["state_rank"] <= max_rank]
    vals = d[col].values.astype(float)
    ranks = d["state_rank"].values.astype(float)
    mask = np.isfinite(vals) & np.isfinite(ranks)
    vals = vals[mask]
    ranks = ranks[mask]
    if len(vals) < 2 or np.std(vals) < 1e-12:
        return np.nan
    return float(np.corrcoef(ranks, vals)[0, 1])


def spearman_state(sub, col, max_rank=None):
    d = sub.copy()
    if max_rank is not None:
        d = d[d["state_rank"] <= max_rank]
    vals = d[col].values.astype(float)
    ranks = d["state_rank"].values.astype(float)
    mask = np.isfinite(vals) & np.isfinite(ranks)
    vals = vals[mask]
    ranks = ranks[mask]
    if len(vals) < 2 or np.std(vals) < 1e-12:
        return np.nan
    return float(np.corrcoef(ranks, pd.Series(vals).rank().values)[0, 1])


def state_val(sub, state_id, col):
    x = sub[sub["state_id"] == state_id]
    if len(x) == 0:
        return np.nan
    return safe_float(x[col].iloc[0])


def verdict_from_summaries(model_key, axis_summary, learned_state_summary):
    ax = axis_summary.copy()
    st = learned_state_summary[learned_state_summary["model_key"] == model_key].copy().sort_values("state_rank")

    def get_axis(axis, group, metric):
        r = ax[(ax["axis"] == axis) & (ax["feature_group"] == group)]
        if len(r) == 0:
            return np.nan
        return safe_float(r.iloc[0].get(metric, np.nan))

    core_corr_all_cv = get_axis("eta_core", "AllInternal", "corr")
    core_r2_all_cv = get_axis("eta_core", "AllInternal", "r2")
    core_corr_cat_cv = get_axis("eta_core", "CategoryOnly", "corr")
    transfer_auc_all_cv = get_axis("eta_transfer", "AllInternal", "auc")
    transfer_f1_all_cv = get_axis("eta_transfer", "AllInternal", "macro_f1")
    transfer_auc_cat_cv = get_axis("eta_transfer", "CategoryOnly", "auc")

    core_corr_topk_hidden_cv = get_axis("eta_core", "TopKPlusHidden", "corr")
    transfer_auc_topk_hidden_cv = get_axis("eta_transfer", "TopKPlusHidden", "auc")

    # Learned full-axis state summary.
    core_p_all = pearson_state(st, "eta_core_all", max_rank=3)
    core_s_all = spearman_state(st, "eta_core_all", max_rank=3)
    core_vals = st[st["state_rank"] <= 3]["eta_core_all"].values.astype(float)
    core_strict = bool(np.all(np.diff(core_vals) > 0)) if len(core_vals) >= 2 else False

    trans_s3 = state_val(st, "S3", "eta_transfer_all")
    trans_s4 = state_val(st, "S4", "eta_transfer_all")
    trans_gain = trans_s4 - trans_s3 if np.isfinite(trans_s4) and np.isfinite(trans_s3) else np.nan

    cat_core_p = pearson_state(st, "V_core_category", max_rank=3)
    cat_trans_gain = state_val(st, "S4", "V_transfer_category") - state_val(st, "S3", "V_transfer_category")

    learned_improves_core = bool(np.isfinite(core_corr_all_cv) and np.isfinite(core_corr_cat_cv) and core_corr_all_cv > core_corr_cat_cv + 0.05)
    learned_improves_transfer = bool(np.isfinite(transfer_auc_all_cv) and np.isfinite(transfer_auc_cat_cv) and transfer_auc_all_cv > transfer_auc_cat_cv + 0.05)

    core_pass = bool(np.isfinite(core_corr_all_cv) and core_corr_all_cv >= 0.50)
    transfer_pass = bool(np.isfinite(transfer_auc_all_cv) and transfer_auc_all_cv >= 0.65)
    topo_hidden_support = (
        (np.isfinite(core_corr_topk_hidden_cv) and core_corr_topk_hidden_cv >= 0.35) or
        (np.isfinite(transfer_auc_topk_hidden_cv) and transfer_auc_topk_hidden_cv >= 0.60)
    )

    if core_pass and transfer_pass and topo_hidden_support:
        verdict = "PASS_STRONG_LEARNED_INTERNAL_VISIBILITY_AXIS"
    elif core_pass and transfer_pass:
        verdict = "PASS_LEARNED_INTERNAL_VISIBILITY_AXIS"
    elif core_pass or transfer_pass:
        verdict = "PARTIAL_LEARNED_INTERNAL_VISIBILITY_AXIS"
    else:
        verdict = "FAIL_LEARNED_INTERNAL_VISIBILITY_AXIS"

    return {
        "model_key": model_key,
        "verdict": verdict,

        "cv_eta_core_all_corr": core_corr_all_cv,
        "cv_eta_core_all_r2": core_r2_all_cv,
        "cv_eta_core_category_corr": core_corr_cat_cv,
        "cv_eta_core_topk_hidden_corr": core_corr_topk_hidden_cv,

        "cv_eta_transfer_all_auc": transfer_auc_all_cv,
        "cv_eta_transfer_all_macro_f1": transfer_f1_all_cv,
        "cv_eta_transfer_category_auc": transfer_auc_cat_cv,
        "cv_eta_transfer_topk_hidden_auc": transfer_auc_topk_hidden_cv,

        "learned_axis_core_state_pearson_S0_S3": core_p_all,
        "learned_axis_core_state_spearman_S0_S3": core_s_all,
        "learned_axis_core_strict_S0_S3": core_strict,
        "learned_axis_transfer_gain_S4_minus_S3": trans_gain,

        "category_core_state_pearson_S0_S3": cat_core_p,
        "category_transfer_gain_S4_minus_S3": cat_trans_gain,

        "learned_improves_core_over_category": learned_improves_core,
        "learned_improves_transfer_over_category": learned_improves_transfer,
        "topk_hidden_axis_support": topo_hidden_support,

        "eta_core_all_S0": state_val(st, "S0", "eta_core_all"),
        "eta_core_all_S1": state_val(st, "S1", "eta_core_all"),
        "eta_core_all_S2": state_val(st, "S2", "eta_core_all"),
        "eta_core_all_S3": state_val(st, "S3", "eta_core_all"),
        "eta_core_all_S4": state_val(st, "S4", "eta_core_all"),
        "eta_transfer_all_S3": trans_s3,
        "eta_transfer_all_S4": trans_s4,
    }


# ============================================================
# Runner
# ============================================================

def run_one_model(model_key, dataset):
    spec = MODEL_SPECS[model_key]
    model_out = OUT_DIR / model_key
    model_out.mkdir(parents=True, exist_ok=True)

    info = {
        "experiment_id": EXPERIMENT_ID,
        "model_key": model_key,
        "model_name": spec["model_name"],
        "path": spec["path"],
        "exists": os.path.exists(spec["path"]),
        "status": "pending",
        "started_at": now_time(),
    }

    if not os.path.exists(spec["path"]):
        info.update({
            "status": "missing_path",
            "error": f"Path not found: {spec['path']}",
        })
        return info

    model = None
    try:
        t0 = time.time()
        model, tokenizer = load_model_and_tokenizer(spec["path"], spec.get("trust_remote_code", True))
        W = lm_head_weight(model)
        cat_ids = category_token_ids(tokenizer)

        rows = []
        errors = []
        for i, row in dataset.iterrows():
            if i % 10 == 0:
                print(f"[{model_key}] extracting internal row {i}/{len(dataset)}", flush=True)
            try:
                feat = extract_row_features(row, model, tokenizer, W, cat_ids, model_key)
                rows.append({**row.to_dict(), "model_key": model_key, "model_name": spec["model_name"], **feat})
            except Exception as e:
                errors.append({
                    "row_id": int(row["row_id"]),
                    "state_id": row["state_id"],
                    "task_id": row["task_id"],
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
                print(f"[warn][{model_key}] row {row['row_id']} failed: {type(e).__name__}: {e}", flush=True)

        features_df = pd.DataFrame(rows)
        error_df = pd.DataFrame(errors)

        features_df.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_internal_features.csv", index=False, encoding="utf-8-sig")
        error_df.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_errors.csv", index=False, encoding="utf-8-sig")

        if features_df.empty:
            info.update({
                "status": "empty_features",
                "n_errors": int(len(error_df)),
            })
            return info

        axis_summary, groups = state_axis_summary(features_df)
        axis_df = add_learned_axes(features_df, groups)
        learned_state = learned_axis_state_summary(axis_df)
        verdict = verdict_from_summaries(model_key, axis_summary, learned_state)

        axis_summary.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_axis_cv_summary.csv", index=False, encoding="utf-8-sig")
        axis_df.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_axis_features.csv", index=False, encoding="utf-8-sig")
        learned_state.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_learned_axis_state_summary.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([verdict]).to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_verdict_summary.csv", index=False, encoding="utf-8-sig")

        info.update({
            "status": "done",
            "finished_at": now_time(),
            "runtime_sec": time.time() - t0,
            "n_rows": int(len(features_df)),
            "n_errors": int(len(error_df)),
            "verdict": verdict.get("verdict", "UNDETERMINED"),
            "metrics": verdict,
            "outputs": {
                "internal_features": str(model_out / f"{EXPERIMENT_ID}_{model_key}_internal_features.csv"),
                "axis_cv_summary": str(model_out / f"{EXPERIMENT_ID}_{model_key}_axis_cv_summary.csv"),
                "axis_features": str(model_out / f"{EXPERIMENT_ID}_{model_key}_axis_features.csv"),
                "learned_axis_state_summary": str(model_out / f"{EXPERIMENT_ID}_{model_key}_learned_axis_state_summary.csv"),
                "verdict_summary": str(model_out / f"{EXPERIMENT_ID}_{model_key}_verdict_summary.csv"),
            },
        })

        with open(model_out / f"{EXPERIMENT_ID}_{model_key}_verdict.json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

    except Exception as e:
        info.update({
            "status": "failed",
            "finished_at": now_time(),
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc(),
        })

    finally:
        try:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    return info


def cross_model_summary(model_infos):
    rows = []
    for info in model_infos:
        m = info.get("metrics", {}) or {}
        rows.append({
            "model_key": info.get("model_key"),
            "model_name": info.get("model_name"),
            "status": info.get("status"),
            "verdict": info.get("verdict"),

            "cv_eta_core_all_corr": m.get("cv_eta_core_all_corr"),
            "cv_eta_core_all_r2": m.get("cv_eta_core_all_r2"),
            "cv_eta_core_category_corr": m.get("cv_eta_core_category_corr"),
            "cv_eta_core_topk_hidden_corr": m.get("cv_eta_core_topk_hidden_corr"),

            "cv_eta_transfer_all_auc": m.get("cv_eta_transfer_all_auc"),
            "cv_eta_transfer_all_macro_f1": m.get("cv_eta_transfer_all_macro_f1"),
            "cv_eta_transfer_category_auc": m.get("cv_eta_transfer_category_auc"),
            "cv_eta_transfer_topk_hidden_auc": m.get("cv_eta_transfer_topk_hidden_auc"),

            "learned_axis_core_state_pearson_S0_S3": m.get("learned_axis_core_state_pearson_S0_S3"),
            "learned_axis_core_state_spearman_S0_S3": m.get("learned_axis_core_state_spearman_S0_S3"),
            "learned_axis_core_strict_S0_S3": m.get("learned_axis_core_strict_S0_S3"),
            "learned_axis_transfer_gain_S4_minus_S3": m.get("learned_axis_transfer_gain_S4_minus_S3"),

            "category_core_state_pearson_S0_S3": m.get("category_core_state_pearson_S0_S3"),
            "category_transfer_gain_S4_minus_S3": m.get("category_transfer_gain_S4_minus_S3"),

            "learned_improves_core_over_category": m.get("learned_improves_core_over_category"),
            "learned_improves_transfer_over_category": m.get("learned_improves_transfer_over_category"),
            "topk_hidden_axis_support": m.get("topk_hidden_axis_support"),

            "eta_core_all_S0": m.get("eta_core_all_S0"),
            "eta_core_all_S1": m.get("eta_core_all_S1"),
            "eta_core_all_S2": m.get("eta_core_all_S2"),
            "eta_core_all_S3": m.get("eta_core_all_S3"),
            "eta_core_all_S4": m.get("eta_core_all_S4"),
            "eta_transfer_all_S3": m.get("eta_transfer_all_S3"),
            "eta_transfer_all_S4": m.get("eta_transfer_all_S4"),

            "n_rows": info.get("n_rows"),
            "n_errors": info.get("n_errors"),
            "error": info.get("error"),
        })
    return pd.DataFrame(rows)


def global_verdict(cross):
    done = cross[cross["status"] == "done"].copy()
    if len(done) == 0:
        return "FAIL_SOBV2B_NO_MODELS"

    verdicts = done["verdict"].astype(str).tolist()
    pass_strong = sum(v.startswith("PASS_STRONG") for v in verdicts)
    pass_any = sum(v.startswith("PASS") for v in verdicts)
    partial = sum(v.startswith("PARTIAL") for v in verdicts)

    if pass_strong == len(done):
        return "PASS_STRONG_SOBV2B_CROSS_MODEL"
    if pass_any == len(done):
        return "PASS_SOBV2B_CROSS_MODEL"
    if pass_any + partial == len(done):
        return "PARTIAL_SOBV2B_CROSS_MODEL"
    if pass_any >= 2:
        return "MIXED_POSITIVE_SOBV2B"
    if pass_any >= 1:
        return "MIXED_SOBV2B"
    return "FAIL_SOBV2B"


def main():
    set_seed()
    print("=" * 80)
    print(f"[{EXPERIMENT_ID}] Learned Internal Visibility Axis")
    print(f"Device: {DEVICE}, dtype: {DTYPE}")
    print(f"Output: {OUT_DIR}")
    print("=" * 80)

    dataset = build_dataset()
    dataset_path = OUT_DIR / f"{EXPERIMENT_ID}_dataset.csv"
    dataset.to_csv(dataset_path, index=False, encoding="utf-8-sig")

    config = {
        "experiment_id": EXPERIMENT_ID,
        "created_at": now_time(),
        "out_dir": str(OUT_DIR),
        "device": DEVICE,
        "dtype": str(DTYPE),
        "models_to_run": MODELS_TO_RUN,
        "model_specs": MODEL_SPECS,
        "subject_states": SUBJECT_STATES,
        "tasks": TASKS,
        "all_categories": ALL_CATEGORIES,
        "category_terms": CATEGORY_TERMS,
        "topk": TOPK,
        "max_input_len": MAX_INPUT_LEN,
        "random_seed": RANDOM_SEED,
    }
    with open(OUT_DIR / f"{EXPERIMENT_ID}_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    model_infos = []
    for model_key in MODELS_TO_RUN:
        print("=" * 80)
        print(f"[{EXPERIMENT_ID}] Running model: {model_key}")
        info = run_one_model(model_key, dataset)
        model_infos.append(info)
        print(json.dumps({
            "model_key": info.get("model_key"),
            "status": info.get("status"),
            "verdict": info.get("verdict"),
            "metrics": info.get("metrics"),
            "error": info.get("error"),
        }, ensure_ascii=False, indent=2))

    cross = cross_model_summary(model_infos)
    cross_path = OUT_DIR / f"{EXPERIMENT_ID}_cross_model_summary.csv"
    cross.to_csv(cross_path, index=False, encoding="utf-8-sig")

    gv = global_verdict(cross)
    global_obj = {
        "experiment_id": EXPERIMENT_ID,
        "verdict": gv,
        "n_models": len(model_infos),
        "n_models_done": int((cross["status"] == "done").sum()) if "status" in cross.columns else 0,
        "model_infos": model_infos,
        "outputs": {
            "dataset": str(dataset_path),
            "cross_model_summary": str(cross_path),
            "out_dir": str(OUT_DIR),
        },
        "interpretation": {
            "eta_core": "Learned internal axis for core visibility rank S0-S3.",
            "eta_transfer": "Learned internal axis for S3 vs S4 transfer visibility.",
            "CategoryOnly": "Hand-built category-logit visibility features.",
            "TopKPlusHidden": "No category-token feature group using TopK/VIM and hidden summaries.",
            "PASS_STRONG_LEARNED_INTERNAL_VISIBILITY_AXIS": "Core and transfer axes pass, and TopK/hidden supports the axes.",
            "PASS_LEARNED_INTERNAL_VISIBILITY_AXIS": "Core and transfer learned axes pass.",
            "PARTIAL_LEARNED_INTERNAL_VISIBILITY_AXIS": "Only one learned axis passes.",
            "FAIL_LEARNED_INTERNAL_VISIBILITY_AXIS": "No stable learned internal visibility axis.",
        },
    }

    with open(OUT_DIR / f"{EXPERIMENT_ID}_verdict.json", "w", encoding="utf-8") as f:
        json.dump(global_obj, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(json.dumps(global_obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
