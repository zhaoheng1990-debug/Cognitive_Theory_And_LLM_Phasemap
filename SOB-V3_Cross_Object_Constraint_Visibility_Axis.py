# -*- coding: utf-8 -*-
r"""
SOB-V3: Cross-Object Constraint Visibility Axis

Purpose
-------
SOB-V2B confirmed learned internal visibility axes for O = Derivative:
  eta_core and eta_transfer exist, with strong Qwen/Llama support and
  TopK+Hidden support for Gemma.

SOB-V3 tests whether this is object-specific but structurally general.

Objects
-------
O in:
  derivative
  integral
  gradient
  eigenvector
  manifold

Main questions
--------------
Q1. For each object O_i, does a learned internal eta_core(O_i) axis exist?
    eta_core predicts S0-S3 subject-state rank over core tasks.

Q2. For each object O_i, does a learned internal eta_transfer(O_i) axis exist?
    eta_transfer separates S3 vs S4 over transfer tasks.

Q3. Are eta axes object-specific but structurally analogous?
    We test cross-object generalization:
       train eta_core on object A -> predict S0-S3 rank on object B
       train eta_transfer on object A -> distinguish S3/S4 on object B

Q4. Does TopK+Hidden recover the object axes without relying on category token mass?

Output
------
C:\Users\ZH\Desktop\AGI\outputs\SOB-V3

Default models
--------------
Qwen:
  D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main

Llama:
  D:\model\Llama-3.2-1B-Instruct

Gemma:
  D:\model\gemma-2-2b-it

Notes
-----
This script does not generate explanatory answers.
It extracts hidden-state, TopK/VIM, and object/category-logit features from
internal probes ending in READY.
"""

import os
import re
import json
import time
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

EXPERIMENT_ID = "SOB-V3"
OUT_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\SOB-V3")
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

MAX_INPUT_LEN = 1300
TOPK = 80
N_SPLITS = 4

CORE_STATES = ["S0", "S1", "S2", "S3"]
TRANSFER_STATES = ["S3", "S4"]

OBJECTS = {
    "derivative": {
        "name": "derivative",
        "symbol_terms": ["d/dx", "f'", "prime", "derivative symbol", "differentiation"],
        "definition_terms": ["limit", "difference quotient", "instantaneous rate", "rate of change", "definition"],
        "calculation_terms": ["power rule", "product rule", "quotient rule", "chain rule", "differentiate", "compute"],
        "geometry_terms": ["tangent", "slope", "secant", "local linear", "linear approximation", "first order"],
        "edge_terms": ["continuous", "differentiable", "corner", "cusp", "left derivative", "right derivative"],
        "transfer_terms": ["velocity", "acceleration", "gradient descent", "backpropagation", "feedback", "marginal"],
        "core_question": "Think about derivative as an object: definition, calculation rules, tangent geometry, local linearity, and edge cases.",
        "transfer_question": "Think about transferring derivative to physics, optimization, machine learning, control, or economics.",
    },
    "integral": {
        "name": "integral",
        "symbol_terms": ["integral", "∫", "dx", "antiderivative", "summation"],
        "definition_terms": ["limit", "Riemann sum", "area", "accumulation", "partition", "definition"],
        "calculation_terms": ["antiderivative", "substitution", "integration by parts", "fundamental theorem", "compute"],
        "geometry_terms": ["area under curve", "accumulated area", "signed area", "region", "curve"],
        "edge_terms": ["improper integral", "convergence", "divergence", "discontinuity", "singular", "finite area"],
        "transfer_terms": ["work", "probability density", "mass", "total change", "expected value", "accumulation"],
        "core_question": "Think about integral as an object: definition, accumulation, area, calculation methods, and convergence edge cases.",
        "transfer_question": "Think about transferring integral to physics, probability, statistics, economics, or accumulated change.",
    },
    "gradient": {
        "name": "gradient",
        "symbol_terms": ["gradient", "nabla", "∇", "partial derivative", "vector"],
        "definition_terms": ["partial derivatives", "directional derivative", "rate of change", "definition", "scalar field"],
        "calculation_terms": ["compute partial", "Jacobian", "chain rule", "differentiate", "component"],
        "geometry_terms": ["steepest ascent", "level set", "normal direction", "direction", "slope field"],
        "edge_terms": ["non-differentiable", "critical point", "saddle", "flat region", "constraint"],
        "transfer_terms": ["gradient descent", "loss landscape", "optimization", "backpropagation", "force field", "control"],
        "core_question": "Think about gradient as an object: partial derivatives, direction of steepest change, level sets, and critical cases.",
        "transfer_question": "Think about transferring gradient to optimization, machine learning, physics fields, or control.",
    },
    "eigenvector": {
        "name": "eigenvector",
        "symbol_terms": ["eigenvector", "eigenvalue", "lambda", "Av", "linear transformation"],
        "definition_terms": ["Av equals lambda v", "unchanged direction", "linear map", "definition", "scale"],
        "calculation_terms": ["determinant", "characteristic polynomial", "solve", "matrix", "linear algebra"],
        "geometry_terms": ["invariant direction", "stretch", "scale", "rotation", "axis", "transformation"],
        "edge_terms": ["degenerate", "repeated eigenvalue", "defective", "zero eigenvalue", "complex eigenvalue"],
        "transfer_terms": ["PCA", "stability", "dynamical system", "quantum state", "principal component", "spectral"],
        "core_question": "Think about eigenvector as an object: invariant direction, scaling, matrix calculation, and degenerate edge cases.",
        "transfer_question": "Think about transferring eigenvectors to PCA, dynamical systems, stability, quantum mechanics, or network analysis.",
    },
    "manifold": {
        "name": "manifold",
        "symbol_terms": ["manifold", "M", "chart", "atlas", "surface"],
        "definition_terms": ["locally Euclidean", "chart", "coordinate", "atlas", "topological space", "definition"],
        "calculation_terms": ["coordinate chart", "tangent space", "map", "dimension", "local coordinate"],
        "geometry_terms": ["curvature", "surface", "tangent", "geodesic", "local patch", "connection"],
        "edge_terms": ["singularity", "boundary", "non-manifold", "chart overlap", "self-intersection"],
        "transfer_terms": ["data manifold", "state space", "configuration space", "latent space", "robotics", "general relativity"],
        "core_question": "Think about manifold as an object: local coordinates, charts, tangent spaces, curvature, and boundary/singularity cases.",
        "transfer_question": "Think about transferring manifold to machine learning latent spaces, robotics, physics, or scientific state spaces.",
    },
}

SUBJECT_STATES = [
    {
        "state_id": "S0",
        "state_rank": 0,
        "state_name": "symbol_only",
        "profile": (
            "You have only seen the symbol or name of the object. "
            "You do not know its definition, computation, geometry, edge cases, or applications."
        ),
    },
    {
        "state_id": "S1",
        "state_rank": 1,
        "state_name": "definition_known",
        "profile": (
            "You know only the formal definition and a few basic words. "
            "You cannot reliably compute with it, understand its geometry, or transfer it to other domains."
        ),
    },
    {
        "state_id": "S2",
        "state_rank": 2,
        "state_name": "calculation_ability",
        "profile": (
            "You know the definition and can perform standard calculations or procedures. "
            "Your geometric understanding, edge-case understanding, and cross-domain transfer are still limited."
        ),
    },
    {
        "state_id": "S3",
        "state_rank": 3,
        "state_name": "geometric_understanding",
        "profile": (
            "You know definition, calculation, geometric meaning, and important boundary or edge cases. "
            "You are not yet an expert in cross-domain transfer."
        ),
    },
    {
        "state_id": "S4",
        "state_rank": 4,
        "state_name": "cross_domain_transfer",
        "profile": (
            "You understand the object as a transferable structural tool and can apply it across mathematics, physics, machine learning, control, economics, biology, and unfamiliar dynamic systems."
        ),
    },
]

TASKS = [
    {"task_id": "CORE_1", "task_family": "core", "question_key": "core_question"},
    {"task_id": "CORE_2", "task_family": "core", "question_key": "core_question"},
    {"task_id": "TRANSFER_1", "task_family": "transfer", "question_key": "transfer_question"},
    {"task_id": "TRANSFER_2", "task_family": "transfer", "question_key": "transfer_question"},
]

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


def build_prompt(obj_key, obj, subject_state, task):
    question = obj[task["question_key"]]
    return (
        "Internal-state probe only. Do not produce an explanatory answer.\n\n"
        "Read the learner state, object, and task. Silently form the internal representation that would be needed.\n\n"
        f"Learner state:\n{subject_state['profile']}\n\n"
        f"Object of cognition: {obj['name']}.\n\n"
        f"Probe task:\n{question}\n\n"
        "End with exactly this token: READY"
    )


def build_dataset():
    rows = []
    for obj_key, obj in OBJECTS.items():
        for st in SUBJECT_STATES:
            for task in TASKS:
                rows.append({
                    "row_id": len(rows),
                    "object_key": obj_key,
                    "object_name": obj["name"],
                    "state_id": st["state_id"],
                    "state_rank": st["state_rank"],
                    "state_name": st["state_name"],
                    "task_id": f"{obj_key}_{task['task_id']}",
                    "task_family": task["task_family"],
                    "question": obj[task["question_key"]],
                    "prompt": build_prompt(obj_key, obj, st, task),
                })
    return pd.DataFrame(rows)


# ============================================================
# ML helpers
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


def make_regressor():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=1.0)),
    ])


def cv_regression(df, cols, target, group_col="task_id"):
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
        "n": int(len(d)),
        "n_features": int(len(cols)),
        "r2": safe_float(r2_score(y, pred)),
        "corr": corr,
    }


def cv_binary(df, cols, target, group_col="task_id"):
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
        "n": int(len(d)),
        "n_features": int(len(cols)),
        "auc": auc,
        "acc": safe_float(accuracy_score(y, pred)),
        "macro_f1": safe_float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_acc": safe_float(balanced_accuracy_score(y, pred)),
    }


def train_predict_regression(train_df, test_df, cols, target):
    cols = numeric_cols(train_df, cols)
    cols = [c for c in cols if c in test_df.columns]
    if not cols:
        return None
    tr = train_df.dropna(subset=[target]).copy()
    if len(tr) < 4:
        return None
    Xtr = tr[cols].replace([np.inf, -np.inf], np.nan).values
    ytr = tr[target].astype(float).values
    Xte = test_df[cols].replace([np.inf, -np.inf], np.nan).values
    reg = make_regressor()
    reg.fit(Xtr, ytr)
    return reg.predict(Xte)


def train_predict_binary(train_df, test_df, cols, target):
    cols = numeric_cols(train_df, cols)
    cols = [c for c in cols if c in test_df.columns]
    if not cols:
        return None
    tr = train_df.dropna(subset=[target]).copy()
    if len(tr) < 4:
        return None
    ytr = tr[target].astype(int).values
    if len(np.unique(ytr)) < 2:
        return None
    Xtr = tr[cols].replace([np.inf, -np.inf], np.nan).values
    Xte = test_df[cols].replace([np.inf, -np.inf], np.nan).values
    clf = make_binary_classifier()
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


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
    for obj_key, obj in OBJECTS.items():
        cats = {
            "symbol": obj["symbol_terms"],
            "definition": obj["definition_terms"],
            "calculation": obj["calculation_terms"],
            "geometry": obj["geometry_terms"],
            "edge": obj["edge_terms"],
            "transfer": obj["transfer_terms"],
        }
        for cat, terms in cats.items():
            key = f"{obj_key}_{cat}"
            ids = []
            for term in terms:
                for v in [term, " " + term, "\n" + term]:
                    toks = tokenizer(v, add_special_tokens=False).input_ids
                    if toks:
                        ids.append(toks[-1])
            out[key] = sorted(set(ids))
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
            "final": span_layers(n_layers, n_layers, n_hidden_states),
        }
    else:
        return {
            "init": span_layers(0, rel(0.20), n_hidden_states),
            "middle": span_layers(rel(0.25), rel(0.65), n_hidden_states),
            "competition": span_layers(rel(0.66), rel(0.78), n_hidden_states),
            "commit": span_layers(rel(0.80), rel(0.92), n_hidden_states),
            "final": span_layers(n_layers, n_layers, n_hidden_states),
        }


def object_category_scores(h, W, cat_ids, obj_key):
    logits = h @ W.T
    cats = ["symbol", "definition", "calculation", "geometry", "edge", "transfer"]
    scores = {}
    for cat in cats:
        ids = cat_ids.get(f"{obj_key}_{cat}", [])
        scores[cat] = float(np.max(logits[ids])) if ids else np.nan
    return scores


def hidden_summary_features(feat, wname, hvecs):
    H = np.asarray(hvecs, dtype=np.float64)
    if H.ndim != 2 or H.shape[0] == 0:
        return
    mean = H.mean(axis=0)
    std = H.std(axis=0)
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

    d = H.shape[1]
    sample_n = min(96, d)
    idx = np.linspace(0, d - 1, sample_n).round().astype(int)
    for j, k in enumerate(idx):
        feat[f"{wname}_hmean_{j:03d}"] = float(mean[k])
        feat[f"{wname}_hstd_{j:03d}"] = float(std[k])


def extract_row_features(row, model, tokenizer, W, cat_ids, model_key):
    hs = forward_hidden(model, tokenizer, row["prompt"])
    windows = windows_for_model(model_key, len(hs))
    obj_key = row["object_key"]
    feat = {
        "n_hidden_states": len(hs),
        "n_layers": len(hs) - 1,
    }

    for wname, hidxs in windows.items():
        cat_seq = {cat: [] for cat in ["symbol", "definition", "calculation", "geometry", "edge", "transfer"]}
        topk_centers, topk_entropy_vals, topk_spread_vals = [], [], []

        for hidx in hidxs:
            h = hs[hidx]
            scores = object_category_scores(h, W, cat_ids, obj_key)
            for cat in cat_seq:
                cat_seq[cat].append(scores.get(cat, np.nan))

            logits = h @ W.T
            ids = topk_indices(logits, TOPK)
            vals = logits[ids]
            p = softmax_np(vals)
            topk_entropy_vals.append(entropy_bits(p))
            topk_spread_vals.append(spread_vec(ids, W))
            topk_centers.append(center_vec(ids, vals, W))

        cats = list(cat_seq.keys())
        cat_mat = np.array([[cat_seq[cat][i] for cat in cats] for i in range(len(hidxs))], dtype=np.float64)
        cat_probs = []
        for i in range(cat_mat.shape[0]):
            row_scores = cat_mat[i]
            finite = np.isfinite(row_scores)
            if finite.sum() == 0:
                cat_probs.append(np.ones(len(cats)) / len(cats))
            else:
                row_scores = np.where(finite, row_scores, np.nanmin(row_scores[finite]) - 10.0)
                cat_probs.append(softmax_np(row_scores))
        cat_probs = np.array(cat_probs, dtype=np.float64)
        mean_probs = cat_probs.mean(axis=0)

        for j, cat in enumerate(cats):
            vals = np.asarray(cat_seq[cat], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            feat[f"{wname}_catlogit_{cat}_mean"] = float(vals.mean()) if vals.size else np.nan
            feat[f"{wname}_catprob_{cat}_mean"] = float(mean_probs[j])

        feat[f"{wname}_cat_entropy_mean"] = float(np.mean([entropy_bits(p) for p in cat_probs]))
        feat[f"{wname}_core_mass"] = float(
            mean_probs[cats.index("definition")] +
            mean_probs[cats.index("calculation")] +
            mean_probs[cats.index("geometry")] +
            mean_probs[cats.index("edge")]
        )
        feat[f"{wname}_transfer_mass"] = float(mean_probs[cats.index("transfer")])
        feat[f"{wname}_symbol_mass"] = float(mean_probs[cats.index("symbol")])
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

    return feat


# ============================================================
# Analysis
# ============================================================

def build_feature_groups(df):
    category_cols = cols_by_prefix(df, [
        "init_catprob_", "middle_catprob_", "competition_catprob_", "commit_catprob_", "final_catprob_",
        "init_core_", "middle_core_", "commit_core_", "final_core_",
        "init_transfer_", "middle_transfer_", "commit_transfer_", "final_transfer_",
        "init_symbol_", "middle_symbol_", "commit_symbol_", "final_symbol_",
        "V_core_category", "V_transfer_category", "V_symbol_category",
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
    topk_hidden_cols = unique_keep_order(topk_cols + hidden_cols)
    return {
        "CategoryOnly": category_cols,
        "TopKOnly": topk_cols,
        "HiddenSummary": hidden_cols,
        "TopKPlusHidden": topk_hidden_cols,
        "AllInternal": all_cols,
    }


def within_object_axis_eval(features_df, groups):
    rows = []
    for obj_key, odf in features_df.groupby("object_key"):
        core_df = odf[(odf["task_family"] == "core") & (odf["state_id"].isin(CORE_STATES))].copy()
        for gname, cols in groups.items():
            res = cv_regression(core_df, cols, "state_rank", group_col="task_id")
            rows.append({
                "object_key": obj_key,
                "axis": "eta_core",
                "feature_group": gname,
                "valid": res is not None,
                "n": res["n"] if res else np.nan,
                "n_features": res["n_features"] if res else len(cols),
                "r2": res["r2"] if res else np.nan,
                "corr": res["corr"] if res else np.nan,
                "auc": np.nan,
                "macro_f1": np.nan,
                "balanced_acc": np.nan,
            })

        trans_df = odf[(odf["task_family"] == "transfer") & (odf["state_id"].isin(TRANSFER_STATES))].copy()
        trans_df["is_S4"] = (trans_df["state_id"] == "S4").astype(int)
        for gname, cols in groups.items():
            res = cv_binary(trans_df, cols, "is_S4", group_col="task_id")
            rows.append({
                "object_key": obj_key,
                "axis": "eta_transfer",
                "feature_group": gname,
                "valid": res is not None,
                "n": res["n"] if res else np.nan,
                "n_features": res["n_features"] if res else len(cols),
                "r2": np.nan,
                "corr": np.nan,
                "auc": res["auc"] if res else np.nan,
                "macro_f1": res["macro_f1"] if res else np.nan,
                "balanced_acc": res["balanced_acc"] if res else np.nan,
            })
    return pd.DataFrame(rows)


def cross_object_axis_eval(features_df, groups):
    rows = []
    objects = sorted(features_df["object_key"].unique().tolist())
    for train_obj in objects:
        for test_obj in objects:
            if train_obj == test_obj:
                continue

            train = features_df[features_df["object_key"] == train_obj].copy()
            test = features_df[features_df["object_key"] == test_obj].copy()

            train_core = train[(train["task_family"] == "core") & (train["state_id"].isin(CORE_STATES))].copy()
            test_core = test[(test["task_family"] == "core") & (test["state_id"].isin(CORE_STATES))].copy()

            for gname in ["TopKPlusHidden", "AllInternal"]:
                cols = groups[gname]
                pred = train_predict_regression(train_core, test_core, cols, "state_rank")
                if pred is None:
                    rows.append({
                        "train_object": train_obj,
                        "test_object": test_obj,
                        "axis": "eta_core_cross_object",
                        "feature_group": gname,
                        "valid": False,
                    })
                else:
                    y = test_core["state_rank"].astype(float).values
                    corr = np.nan
                    if np.std(y) > 1e-12 and np.std(pred) > 1e-12:
                        corr = float(np.corrcoef(y, pred)[0, 1])
                    rows.append({
                        "train_object": train_obj,
                        "test_object": test_obj,
                        "axis": "eta_core_cross_object",
                        "feature_group": gname,
                        "valid": True,
                        "n": int(len(y)),
                        "r2": safe_float(r2_score(y, pred)),
                        "corr": corr,
                    })

            train_trans = train[(train["task_family"] == "transfer") & (train["state_id"].isin(TRANSFER_STATES))].copy()
            test_trans = test[(test["task_family"] == "transfer") & (test["state_id"].isin(TRANSFER_STATES))].copy()
            train_trans["is_S4"] = (train_trans["state_id"] == "S4").astype(int)
            test_trans["is_S4"] = (test_trans["state_id"] == "S4").astype(int)

            for gname in ["TopKPlusHidden", "AllInternal"]:
                cols = groups[gname]
                prob = train_predict_binary(train_trans, test_trans, cols, "is_S4")
                if prob is None:
                    rows.append({
                        "train_object": train_obj,
                        "test_object": test_obj,
                        "axis": "eta_transfer_cross_object",
                        "feature_group": gname,
                        "valid": False,
                    })
                else:
                    y = test_trans["is_S4"].astype(int).values
                    pred = (prob >= 0.5).astype(int)
                    auc = np.nan
                    try:
                        auc = safe_float(roc_auc_score(y, prob))
                    except Exception:
                        pass
                    rows.append({
                        "train_object": train_obj,
                        "test_object": test_obj,
                        "axis": "eta_transfer_cross_object",
                        "feature_group": gname,
                        "valid": True,
                        "n": int(len(y)),
                        "auc": auc,
                        "macro_f1": safe_float(f1_score(y, pred, average="macro", zero_division=0)),
                        "balanced_acc": safe_float(balanced_accuracy_score(y, pred)),
                    })
    return pd.DataFrame(rows)


def object_verdicts(within_df, cross_df):
    rows = []
    for obj_key in sorted(within_df["object_key"].unique().tolist()):
        w = within_df[within_df["object_key"] == obj_key]
        core = w[(w["axis"] == "eta_core") & (w["feature_group"] == "TopKPlusHidden")]
        trans = w[(w["axis"] == "eta_transfer") & (w["feature_group"] == "TopKPlusHidden")]

        core_corr = safe_float(core["corr"].iloc[0]) if len(core) else np.nan
        trans_auc = safe_float(trans["auc"].iloc[0]) if len(trans) else np.nan

        core_pass = bool(np.isfinite(core_corr) and core_corr >= 0.50)
        trans_pass = bool(np.isfinite(trans_auc) and trans_auc >= 0.70)

        if core_pass and trans_pass:
            verdict = "PASS_OBJECT_AXIS"
        elif core_pass or trans_pass:
            verdict = "PARTIAL_OBJECT_AXIS"
        else:
            verdict = "FAIL_OBJECT_AXIS"

        rows.append({
            "object_key": obj_key,
            "verdict": verdict,
            "topk_hidden_core_corr": core_corr,
            "topk_hidden_transfer_auc": trans_auc,
        })
    return pd.DataFrame(rows)


def model_global_verdict(obj_verdict_df, within_df, cross_df):
    pass_count = int(obj_verdict_df["verdict"].astype(str).str.startswith("PASS").sum())
    partial_count = int(obj_verdict_df["verdict"].astype(str).str.startswith("PARTIAL").sum())
    n_obj = int(len(obj_verdict_df))

    cross_core = cross_df[
        (cross_df["axis"] == "eta_core_cross_object") &
        (cross_df["feature_group"] == "TopKPlusHidden") &
        (cross_df["valid"] == True)
    ]
    cross_transfer = cross_df[
        (cross_df["axis"] == "eta_transfer_cross_object") &
        (cross_df["feature_group"] == "TopKPlusHidden") &
        (cross_df["valid"] == True)
    ]

    mean_cross_core_corr = safe_float(cross_core["corr"].mean()) if len(cross_core) else np.nan
    mean_cross_transfer_auc = safe_float(cross_transfer["auc"].mean()) if len(cross_transfer) else np.nan

    cross_support = (
        (np.isfinite(mean_cross_core_corr) and mean_cross_core_corr >= 0.25) or
        (np.isfinite(mean_cross_transfer_auc) and mean_cross_transfer_auc >= 0.60)
    )

    if pass_count == n_obj and cross_support:
        verdict = "PASS_STRONG_CROSS_OBJECT_VISIBILITY_AXIS"
    elif pass_count + partial_count == n_obj and cross_support:
        verdict = "PASS_CROSS_OBJECT_VISIBILITY_AXIS"
    elif pass_count >= max(1, int(np.ceil(n_obj * 0.5))):
        verdict = "PARTIAL_CROSS_OBJECT_VISIBILITY_AXIS"
    else:
        verdict = "FAIL_CROSS_OBJECT_VISIBILITY_AXIS"

    return {
        "verdict": verdict,
        "n_objects": n_obj,
        "n_object_pass": pass_count,
        "n_object_partial": partial_count,
        "mean_cross_object_core_corr_topk_hidden": mean_cross_core_corr,
        "mean_cross_object_transfer_auc_topk_hidden": mean_cross_transfer_auc,
        "cross_object_support": cross_support,
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
            if i % 20 == 0:
                print(f"[{model_key}] extracting row {i}/{len(dataset)}", flush=True)
            try:
                feat = extract_row_features(row, model, tokenizer, W, cat_ids, model_key)
                rows.append({**row.to_dict(), "model_key": model_key, "model_name": spec["model_name"], **feat})
            except Exception as e:
                errors.append({
                    "row_id": int(row["row_id"]),
                    "object_key": row["object_key"],
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

        groups = build_feature_groups(features_df)
        within = within_object_axis_eval(features_df, groups)
        cross = cross_object_axis_eval(features_df, groups)
        obj_verdict = object_verdicts(within, cross)
        gv = model_global_verdict(obj_verdict, within, cross)

        within.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_within_object_axis_summary.csv", index=False, encoding="utf-8-sig")
        cross.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_cross_object_axis_summary.csv", index=False, encoding="utf-8-sig")
        obj_verdict.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_object_verdict_summary.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([{**{"model_key": model_key}, **gv}]).to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_verdict_summary.csv", index=False, encoding="utf-8-sig")

        info.update({
            "status": "done",
            "finished_at": now_time(),
            "runtime_sec": time.time() - t0,
            "n_rows": int(len(features_df)),
            "n_errors": int(len(error_df)),
            "verdict": gv.get("verdict", "UNDETERMINED"),
            "metrics": gv,
            "outputs": {
                "internal_features": str(model_out / f"{EXPERIMENT_ID}_{model_key}_internal_features.csv"),
                "within_object_axis_summary": str(model_out / f"{EXPERIMENT_ID}_{model_key}_within_object_axis_summary.csv"),
                "cross_object_axis_summary": str(model_out / f"{EXPERIMENT_ID}_{model_key}_cross_object_axis_summary.csv"),
                "object_verdict_summary": str(model_out / f"{EXPERIMENT_ID}_{model_key}_object_verdict_summary.csv"),
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
            "n_objects": m.get("n_objects"),
            "n_object_pass": m.get("n_object_pass"),
            "n_object_partial": m.get("n_object_partial"),
            "mean_cross_object_core_corr_topk_hidden": m.get("mean_cross_object_core_corr_topk_hidden"),
            "mean_cross_object_transfer_auc_topk_hidden": m.get("mean_cross_object_transfer_auc_topk_hidden"),
            "cross_object_support": m.get("cross_object_support"),
            "n_rows": info.get("n_rows"),
            "n_errors": info.get("n_errors"),
            "error": info.get("error"),
        })
    return pd.DataFrame(rows)


def global_verdict(cross):
    done = cross[cross["status"] == "done"].copy()
    if len(done) == 0:
        return "FAIL_SOBV3_NO_MODELS"

    verdicts = done["verdict"].astype(str).tolist()
    pass_strong = sum(v.startswith("PASS_STRONG") for v in verdicts)
    pass_any = sum(v.startswith("PASS") for v in verdicts)
    partial = sum(v.startswith("PARTIAL") for v in verdicts)

    if pass_strong == len(done):
        return "PASS_STRONG_SOBV3_CROSS_MODEL"
    if pass_any == len(done):
        return "PASS_SOBV3_CROSS_MODEL"
    if pass_any + partial == len(done):
        return "PARTIAL_SOBV3_CROSS_MODEL"
    if pass_any >= 2:
        return "MIXED_POSITIVE_SOBV3"
    if pass_any >= 1:
        return "MIXED_SOBV3"
    return "FAIL_SOBV3"


def main():
    set_seed()
    print("=" * 80)
    print(f"[{EXPERIMENT_ID}] Cross-Object Constraint Visibility Axis")
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
        "objects": OBJECTS,
        "subject_states": SUBJECT_STATES,
        "tasks": TASKS,
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
            "within_object_axis": "Whether each object has its own eta_core and eta_transfer axes.",
            "cross_object_axis": "Whether an axis learned on one object transfers to another object.",
            "TopKPlusHidden": "Feature group excluding object-specific category token mass, using TopK/VIM and hidden summaries.",
            "PASS_STRONG_CROSS_OBJECT_VISIBILITY_AXIS": "All objects pass and cross-object support exists.",
            "PASS_CROSS_OBJECT_VISIBILITY_AXIS": "All objects pass or partially pass with cross-object support.",
            "PARTIAL_CROSS_OBJECT_VISIBILITY_AXIS": "At least half of objects pass.",
        },
    }

    with open(OUT_DIR / f"{EXPERIMENT_ID}_verdict.json", "w", encoding="utf-8") as f:
        json.dump(global_obj, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(json.dumps(global_obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
