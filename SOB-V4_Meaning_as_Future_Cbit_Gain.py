# -*- coding: utf-8 -*-
r"""
SOB-V4: Meaning as Future Cbit Gain

Purpose
-------
SOB-V1~V3 established:
  1. Constraint Visibility V(S,O) is measurable at text and internal-state levels.
  2. Learned internal visibility axes eta_O exist.
  3. eta_O transfers across objects and models.

SOB-V4 tests the next theoretical claim:

    Meaning(S,O) ≈ Future Cbit Gain

That is:
  An object O is meaningful to a subject state S if visibility of O improves
  future compression / trajectory quality / transfer performance on downstream tasks.

Core idea
---------
For each object O and subject state S:
  1. Extract current internal visibility proxies:
       eta_core_current
       eta_transfer_current
       category visibility
       TopK/Hidden visibility features

  2. Generate downstream future tasks:
       same-object transfer
       cross-object near transfer
       far transfer / application
       counterfactual use
       explanation compression

  3. Score generated downstream answers with deterministic rubrics:
       Cbit_future_proxy
       TrajectoryQuality_future_proxy
       AnswerQuality_future_proxy

  4. Test:
       Current internal V(S,O) -> Future Cbit / TQ / AQ

Main hypotheses
---------------
H1:
  Current eta_transfer / TopK+Hidden visibility predicts future Cbit proxy.

H2:
  Future Cbit is better predicted by learned/internal visibility than by
  raw subject rank or object category token mass alone.

H3:
  Transfer visibility is a stronger predictor of future utility than core visibility.

Output
------
C:\Users\ZH\Desktop\AGI\outputs\SOB-V4

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
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ============================================================
# Constants
# ============================================================

EXPERIMENT_ID = "SOB-V4"
OUT_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\SOB-V4")
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

MAX_INPUT_LEN = 1400
MAX_NEW_TOKENS = 220
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
        "future_tasks": [
            ("same_transfer", "Use derivative to explain why velocity is instantaneous change of position."),
            ("near_transfer", "Use derivative-like local change to explain gradient descent."),
            ("far_transfer", "Use derivative as a structural analogy for marginal cost in economics."),
            ("counterfactual", "Explain what breaks if local change cannot be defined."),
            ("compression", "Compress derivative into one reusable principle for future problem solving."),
        ],
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
        "future_tasks": [
            ("same_transfer", "Use integral to explain accumulated distance from velocity over time."),
            ("near_transfer", "Use integral to explain expected value under a probability density."),
            ("far_transfer", "Use integral as a structural analogy for total social cost accumulated over time."),
            ("counterfactual", "Explain what breaks if accumulation over continuous quantities cannot be defined."),
            ("compression", "Compress integral into one reusable principle for future problem solving."),
        ],
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
        "future_tasks": [
            ("same_transfer", "Use gradient to explain steepest ascent on a landscape."),
            ("near_transfer", "Use gradient to explain why optimization updates parameters."),
            ("far_transfer", "Use gradient as an analogy for directional pressure in a policy decision landscape."),
            ("counterfactual", "Explain what breaks if a system cannot estimate direction of steepest change."),
            ("compression", "Compress gradient into one reusable principle for future problem solving."),
        ],
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
        "future_tasks": [
            ("same_transfer", "Use eigenvector to explain an invariant direction under a linear transformation."),
            ("near_transfer", "Use eigenvector to explain principal components in data."),
            ("far_transfer", "Use eigenvector as an analogy for stable strategic direction under organizational transformation."),
            ("counterfactual", "Explain what breaks if a transformation has no stable or invariant directions."),
            ("compression", "Compress eigenvector into one reusable principle for future problem solving."),
        ],
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
        "future_tasks": [
            ("same_transfer", "Use manifold to explain why a curved surface can look flat locally."),
            ("near_transfer", "Use manifold to explain latent spaces in machine learning."),
            ("far_transfer", "Use manifold as an analogy for a scientific field with local methods and global structure."),
            ("counterfactual", "Explain what breaks if local coordinate patches cannot be consistently connected."),
            ("compression", "Compress manifold into one reusable principle for future problem solving."),
        ],
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

CURRENT_PROBES = [
    {"probe_id": "CURRENT_CORE", "task_family": "current_core"},
    {"probe_id": "CURRENT_TRANSFER", "task_family": "current_transfer"},
]

# ============================================================
# Basic utilities
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


def count_words(s):
    return len(re.findall(r"\b[\w'-]+\b", str(s)))


def normalize_text(s):
    return str(s).lower().replace("’", "'").replace("“", '"').replace("”", '"')


def marker_hits(text, markers):
    t = normalize_text(text)
    hits = []
    for m in markers:
        if normalize_text(m) in t:
            hits.append(m)
    return sorted(set(hits))


def marker_score(text, markers):
    if not markers:
        return 0.0, []
    hits = marker_hits(text, markers)
    return float(len(hits) / max(1, len(markers))), hits


def compression_quality(text):
    n = count_words(text)
    if n <= 0:
        return 0.0
    if 25 <= n <= 120:
        return 1.0
    if n < 25:
        return max(0.0, n / 25.0)
    return max(0.0, 1.0 - (n - 120) / 260.0)


def lexical_diversity(text):
    words = re.findall(r"\b[a-zA-Z][a-zA-Z'-]*\b", normalize_text(text))
    if not words:
        return 0.0
    return float(len(set(words)) / max(1, len(words)))


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

# ============================================================
# Dataset
# ============================================================

def build_current_prompt(obj, subject_state, probe):
    if probe["task_family"] == "current_core":
        question = obj["core_question"]
    else:
        question = "Think about whether this learner state can transfer " + obj["name"] + " across future unfamiliar domains."

    return (
        "Internal-state probe only. Do not produce an explanatory answer.\n\n"
        "Read the learner state, object, and task. Silently form the internal representation that would be needed.\n\n"
        f"Learner state:\n{subject_state['profile']}\n\n"
        f"Object of cognition: {obj['name']}.\n\n"
        f"Probe task:\n{question}\n\n"
        "End with exactly this token: READY"
    )


def build_future_prompt(obj, subject_state, task_type, task_text):
    return (
        "You are participating in a future Cbit gain audit.\n\n"
        "Role-play the assigned learner state exactly. "
        "Answer only using what this learner state could plausibly use. "
        "If the learner cannot perform the transfer, say so directly.\n\n"
        f"Learner state:\n{subject_state['profile']}\n\n"
        f"Object previously studied: {obj['name']}.\n\n"
        f"Future task type: {task_type}\n"
        f"Future task:\n{task_text}\n\n"
        "Answer concisely. Focus on reusable structure, not memorized wording."
    )


def build_current_dataset():
    rows = []
    for obj_key, obj in OBJECTS.items():
        for st in SUBJECT_STATES:
            for probe in CURRENT_PROBES:
                rows.append({
                    "row_id": len(rows),
                    "object_key": obj_key,
                    "object_name": obj["name"],
                    "state_id": st["state_id"],
                    "state_rank": st["state_rank"],
                    "state_name": st["state_name"],
                    "probe_id": probe["probe_id"],
                    "task_family": probe["task_family"],
                    "prompt": build_current_prompt(obj, st, probe),
                })
    return pd.DataFrame(rows)


def build_future_dataset():
    rows = []
    for obj_key, obj in OBJECTS.items():
        for st in SUBJECT_STATES:
            for task_type, task_text in obj["future_tasks"]:
                rows.append({
                    "future_row_id": len(rows),
                    "object_key": obj_key,
                    "object_name": obj["name"],
                    "state_id": st["state_id"],
                    "state_rank": st["state_rank"],
                    "state_name": st["state_name"],
                    "future_task_type": task_type,
                    "future_task": task_text,
                    "prompt": build_future_prompt(obj, st, task_type, task_text),
                })
    return pd.DataFrame(rows)

# ============================================================
# ML helpers
# ============================================================

def cv_indices(df, y=None, group_col="object_key", stratify=False):
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


def make_regressor():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=1.0)),
    ])


def make_binary_classifier():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=RANDOM_SEED)),
    ])


def cv_regression(df, cols, target, group_col="object_key"):
    cols = numeric_cols(df, cols)
    d = df.dropna(subset=[target]).copy()
    if len(d) < 10 or not cols:
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


def cv_binary(df, cols, target, group_col="object_key"):
    cols = numeric_cols(df, cols)
    d = df.dropna(subset=[target]).copy()
    if len(d) < 10 or not cols:
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


@torch.no_grad()
def generate_answer(model, tokenizer, prompt):
    text = format_chat(tokenizer, prompt)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LEN)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    out = model.generate(
        **enc,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = out[0, enc["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


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


def extract_current_features(row, model, tokenizer, W, cat_ids, model_key):
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
# Future answer scoring
# ============================================================

def future_markers_for(obj_key, task_type):
    obj = OBJECTS[obj_key]
    base = []
    if task_type == "same_transfer":
        base = obj["definition_terms"] + obj["geometry_terms"] + obj["transfer_terms"]
    elif task_type == "near_transfer":
        base = obj["transfer_terms"] + obj["geometry_terms"]
    elif task_type == "far_transfer":
        base = obj["transfer_terms"] + ["analogy", "structure", "local", "system", "constraint", "mapping"]
    elif task_type == "counterfactual":
        base = obj["edge_terms"] + ["break", "cannot", "fail", "undefined", "limit", "constraint", "no longer"]
    elif task_type == "compression":
        base = obj["definition_terms"] + obj["geometry_terms"] + ["principle", "reusable", "compress", "one idea", "structure"]
    else:
        base = obj["definition_terms"] + obj["transfer_terms"]

    general = [
        "because", "therefore", "local", "structure", "constraint", "transfer",
        "principle", "same idea", "mapping", "explains", "predict", "use"
    ]
    return list(dict.fromkeys(base + general))


def score_future_answer(row):
    text = str(row.get("answer_text", ""))
    obj_key = row["object_key"]
    task_type = row["future_task_type"]

    markers = future_markers_for(obj_key, task_type)
    marker_s, hits = marker_score(text, markers)

    n = count_words(text)
    length_quality = 0.0
    if n > 8:
        length_quality = min(1.0, math.log1p(n) / math.log1p(140))

    div = lexical_diversity(text)
    comp = compression_quality(text)

    transfer_hits = marker_hits(text, ["transfer", "analogy", "same idea", "mapping", "across", "reusable", "principle"])
    causal_hits = marker_hits(text, ["because", "therefore", "so", "explains", "break", "cannot", "if"])
    boundary_hits = marker_hits(text, ["edge", "break", "fail", "undefined", "not", "cannot", "boundary", "constraint"])
    limitation_hits = marker_hits(text, ["do not know", "don't know", "cannot", "not enough", "only seen", "limited"])

    transfer_s = min(1.0, len(transfer_hits) / 3.0)
    causal_s = min(1.0, len(causal_hits) / 3.0)
    boundary_s = min(1.0, len(boundary_hits) / 3.0)

    # Cbit_future_proxy: reusable structural compression, not just answer length.
    cbit_future = (
        0.42 * marker_s +
        0.22 * transfer_s +
        0.18 * causal_s +
        0.10 * comp +
        0.08 * div
    )

    # TQ_future_proxy: trajectory health = structure + transfer + boundary handling.
    tq_future = (
        0.36 * marker_s +
        0.22 * transfer_s +
        0.18 * boundary_s +
        0.14 * causal_s +
        0.10 * length_quality
    )

    # AQ_future_proxy: answer-level usefulness.
    aq_future = (
        0.50 * marker_s +
        0.20 * causal_s +
        0.15 * transfer_s +
        0.10 * length_quality +
        0.05 * div
    )

    # Penalize low states overclaiming only mildly; this experiment cares about future utility.
    overclaim_penalty = 0.0
    if row["state_id"] in ["S0", "S1"] and marker_s > 0.45 and not limitation_hits:
        overclaim_penalty = 0.08

    cbit_future = max(0.0, min(1.0, cbit_future - overclaim_penalty))
    tq_future = max(0.0, min(1.0, tq_future - 0.5 * overclaim_penalty))
    aq_future = max(0.0, min(1.0, aq_future - 0.5 * overclaim_penalty))

    return {
        "future_marker_score": marker_s,
        "future_marker_hits": "; ".join(hits),
        "future_transfer_score": transfer_s,
        "future_causal_score": causal_s,
        "future_boundary_score": boundary_s,
        "future_word_count": n,
        "future_compression_quality": comp,
        "future_lexical_diversity": div,
        "overclaim_penalty": overclaim_penalty,
        "Cbit_future_proxy": cbit_future,
        "TQ_future_proxy": tq_future,
        "AQ_future_proxy": aq_future,
        "future_success": int(cbit_future >= 0.45 and tq_future >= 0.40),
    }


def score_future_df(future_answers):
    rows = []
    for _, row in future_answers.iterrows():
        d = row.to_dict()
        d.update(score_future_answer(d))
        rows.append(d)
    return pd.DataFrame(rows)


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
        "SubjectRankOnly": ["state_rank"],
        "CategoryOnly": category_cols,
        "TopKOnly": topk_cols,
        "HiddenSummary": hidden_cols,
        "TopKPlusHidden": topk_hidden_cols,
        "AllInternal": all_cols,
        "AllPlusRank": unique_keep_order(["state_rank"] + all_cols),
    }


def aggregate_current_features(current_features):
    # Combine CURRENT_CORE and CURRENT_TRANSFER into one row per object/state.
    num_cols = [c for c in current_features.columns if pd.api.types.is_numeric_dtype(current_features[c])]
    keep_num = [c for c in num_cols if c not in ["row_id", "state_rank"]]
    agg = current_features.groupby(["model_key", "object_key", "object_name", "state_id", "state_rank", "state_name"], as_index=False)[keep_num].mean()
    return agg


def aggregate_future_scores(future_scored):
    agg = (
        future_scored
        .groupby(["model_key", "object_key", "object_name", "state_id", "state_rank", "state_name"], as_index=False)
        .agg(
            Cbit_future_proxy=("Cbit_future_proxy", "mean"),
            TQ_future_proxy=("TQ_future_proxy", "mean"),
            AQ_future_proxy=("AQ_future_proxy", "mean"),
            future_success_rate=("future_success", "mean"),
            future_marker_score=("future_marker_score", "mean"),
            n_future=("future_row_id", "size"),
        )
    )
    return agg


def run_predictive_analysis(merged):
    groups = build_feature_groups(merged)
    targets = ["Cbit_future_proxy", "TQ_future_proxy", "AQ_future_proxy"]
    rows = []

    for target in targets:
        for gname, cols in groups.items():
            res = cv_regression(merged, cols, target, group_col="object_key")
            rows.append({
                "target": target,
                "feature_group": gname,
                "task": "future_regression",
                "valid": res is not None,
                "n": res["n"] if res else np.nan,
                "n_features": res["n_features"] if res else len(cols),
                "r2": res["r2"] if res else np.nan,
                "corr": res["corr"] if res else np.nan,
                "auc": np.nan,
                "macro_f1": np.nan,
            })

    for gname, cols in groups.items():
        res = cv_binary(merged, cols, "future_success", group_col="object_key")
        rows.append({
            "target": "future_success",
            "feature_group": gname,
            "task": "future_success_classification",
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


def meaning_verdict(pred_summary):
    def get_corr(target, group):
        x = pred_summary[
            (pred_summary["target"] == target) &
            (pred_summary["feature_group"] == group) &
            (pred_summary["valid"] == True)
        ]
        if len(x) == 0:
            return np.nan
        return safe_float(x["corr"].iloc[0])

    def get_auc(target, group):
        x = pred_summary[
            (pred_summary["target"] == target) &
            (pred_summary["feature_group"] == group) &
            (pred_summary["valid"] == True)
        ]
        if len(x) == 0:
            return np.nan
        return safe_float(x["auc"].iloc[0])

    cbit_rank = get_corr("Cbit_future_proxy", "SubjectRankOnly")
    cbit_cat = get_corr("Cbit_future_proxy", "CategoryOnly")
    cbit_topkhidden = get_corr("Cbit_future_proxy", "TopKPlusHidden")
    cbit_all = get_corr("Cbit_future_proxy", "AllInternal")
    tq_topkhidden = get_corr("TQ_future_proxy", "TopKPlusHidden")
    aq_topkhidden = get_corr("AQ_future_proxy", "TopKPlusHidden")
    success_auc_topkhidden = get_auc("future_success", "TopKPlusHidden")

    internal_predicts_future = bool(
        (np.isfinite(cbit_topkhidden) and cbit_topkhidden >= 0.45) or
        (np.isfinite(cbit_all) and cbit_all >= 0.45)
    )
    internal_beats_rank = bool(np.isfinite(cbit_topkhidden) and np.isfinite(cbit_rank) and cbit_topkhidden > cbit_rank + 0.05)
    internal_beats_category = bool(np.isfinite(cbit_topkhidden) and np.isfinite(cbit_cat) and cbit_topkhidden > cbit_cat + 0.05)

    tq_support = bool(np.isfinite(tq_topkhidden) and tq_topkhidden >= 0.35)
    aq_support = bool(np.isfinite(aq_topkhidden) and aq_topkhidden >= 0.35)
    success_support = bool(np.isfinite(success_auc_topkhidden) and success_auc_topkhidden >= 0.65)

    if internal_predicts_future and tq_support and (internal_beats_rank or internal_beats_category):
        verdict = "PASS_STRONG_MEANING_AS_FUTURE_CBIT_GAIN"
    elif internal_predicts_future and (tq_support or aq_support or success_support):
        verdict = "PASS_MEANING_AS_FUTURE_CBIT_GAIN"
    elif internal_predicts_future or tq_support or success_support:
        verdict = "PARTIAL_MEANING_AS_FUTURE_CBIT_GAIN"
    else:
        verdict = "FAIL_MEANING_AS_FUTURE_CBIT_GAIN"

    return {
        "verdict": verdict,
        "cbit_corr_subject_rank": cbit_rank,
        "cbit_corr_category": cbit_cat,
        "cbit_corr_topk_hidden": cbit_topkhidden,
        "cbit_corr_all_internal": cbit_all,
        "tq_corr_topk_hidden": tq_topkhidden,
        "aq_corr_topk_hidden": aq_topkhidden,
        "future_success_auc_topk_hidden": success_auc_topkhidden,
        "internal_predicts_future_cbit": internal_predicts_future,
        "internal_beats_rank": internal_beats_rank,
        "internal_beats_category": internal_beats_category,
        "tq_support": tq_support,
        "aq_support": aq_support,
        "success_support": success_support,
    }

# ============================================================
# Runner
# ============================================================

def run_one_model(model_key, current_dataset, future_dataset):
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

        # Current internal feature extraction.
        current_rows = []
        current_errors = []
        for i, row in current_dataset.iterrows():
            if i % 20 == 0:
                print(f"[{model_key}] current internal row {i}/{len(current_dataset)}", flush=True)
            try:
                feat = extract_current_features(row, model, tokenizer, W, cat_ids, model_key)
                current_rows.append({**row.to_dict(), "model_key": model_key, "model_name": spec["model_name"], **feat})
            except Exception as e:
                current_errors.append({
                    "row_id": int(row["row_id"]),
                    "object_key": row["object_key"],
                    "state_id": row["state_id"],
                    "probe_id": row["probe_id"],
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
                print(f"[warn][{model_key}] current row {row['row_id']} failed: {type(e).__name__}: {e}", flush=True)

        current_features = pd.DataFrame(current_rows)
        current_error_df = pd.DataFrame(current_errors)

        # Future answer generation.
        future_rows = []
        future_errors = []
        for i, row in future_dataset.iterrows():
            if i % 25 == 0:
                print(f"[{model_key}] future answer row {i}/{len(future_dataset)}", flush=True)
            try:
                ans = generate_answer(model, tokenizer, row["prompt"])
                future_rows.append({**row.to_dict(), "model_key": model_key, "model_name": spec["model_name"], "answer_text": ans})
            except Exception as e:
                future_errors.append({
                    "future_row_id": int(row["future_row_id"]),
                    "object_key": row["object_key"],
                    "state_id": row["state_id"],
                    "future_task_type": row["future_task_type"],
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
                print(f"[warn][{model_key}] future row {row['future_row_id']} failed: {type(e).__name__}: {e}", flush=True)

        future_answers = pd.DataFrame(future_rows)
        future_scored = score_future_df(future_answers) if not future_answers.empty else pd.DataFrame()

        current_features.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_current_internal_features.csv", index=False, encoding="utf-8-sig")
        current_error_df.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_current_errors.csv", index=False, encoding="utf-8-sig")
        future_answers.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_future_answers.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(future_errors).to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_future_errors.csv", index=False, encoding="utf-8-sig")
        future_scored.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_future_scored_answers.csv", index=False, encoding="utf-8-sig")

        if current_features.empty or future_scored.empty:
            info.update({
                "status": "empty_outputs",
                "n_current_rows": int(len(current_features)),
                "n_future_rows": int(len(future_scored)),
                "n_errors": int(len(current_errors) + len(future_errors)),
            })
            return info

        current_agg = aggregate_current_features(current_features)
        future_agg = aggregate_future_scores(future_scored)

        merged = current_agg.merge(
            future_agg,
            on=["model_key", "object_key", "object_name", "state_id", "state_rank", "state_name"],
            how="inner",
        )
        merged["future_success"] = (merged["future_success_rate"] >= 0.5).astype(int)

        pred_summary = run_predictive_analysis(merged)
        verdict = meaning_verdict(pred_summary)

        current_agg.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_current_state_features.csv", index=False, encoding="utf-8-sig")
        future_agg.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_future_state_scores.csv", index=False, encoding="utf-8-sig")
        merged.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_merged_current_future.csv", index=False, encoding="utf-8-sig")
        pred_summary.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_prediction_summary.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([{**{"model_key": model_key}, **verdict}]).to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_verdict_summary.csv", index=False, encoding="utf-8-sig")

        info.update({
            "status": "done",
            "finished_at": now_time(),
            "runtime_sec": time.time() - t0,
            "n_current_rows": int(len(current_features)),
            "n_future_rows": int(len(future_scored)),
            "n_errors": int(len(current_errors) + len(future_errors)),
            "verdict": verdict.get("verdict", "UNDETERMINED"),
            "metrics": verdict,
            "outputs": {
                "current_internal_features": str(model_out / f"{EXPERIMENT_ID}_{model_key}_current_internal_features.csv"),
                "future_answers": str(model_out / f"{EXPERIMENT_ID}_{model_key}_future_answers.csv"),
                "future_scored_answers": str(model_out / f"{EXPERIMENT_ID}_{model_key}_future_scored_answers.csv"),
                "current_state_features": str(model_out / f"{EXPERIMENT_ID}_{model_key}_current_state_features.csv"),
                "future_state_scores": str(model_out / f"{EXPERIMENT_ID}_{model_key}_future_state_scores.csv"),
                "merged_current_future": str(model_out / f"{EXPERIMENT_ID}_{model_key}_merged_current_future.csv"),
                "prediction_summary": str(model_out / f"{EXPERIMENT_ID}_{model_key}_prediction_summary.csv"),
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
            "cbit_corr_subject_rank": m.get("cbit_corr_subject_rank"),
            "cbit_corr_category": m.get("cbit_corr_category"),
            "cbit_corr_topk_hidden": m.get("cbit_corr_topk_hidden"),
            "cbit_corr_all_internal": m.get("cbit_corr_all_internal"),
            "tq_corr_topk_hidden": m.get("tq_corr_topk_hidden"),
            "aq_corr_topk_hidden": m.get("aq_corr_topk_hidden"),
            "future_success_auc_topk_hidden": m.get("future_success_auc_topk_hidden"),
            "internal_predicts_future_cbit": m.get("internal_predicts_future_cbit"),
            "internal_beats_rank": m.get("internal_beats_rank"),
            "internal_beats_category": m.get("internal_beats_category"),
            "tq_support": m.get("tq_support"),
            "aq_support": m.get("aq_support"),
            "success_support": m.get("success_support"),
            "n_current_rows": info.get("n_current_rows"),
            "n_future_rows": info.get("n_future_rows"),
            "n_errors": info.get("n_errors"),
            "error": info.get("error"),
        })
    return pd.DataFrame(rows)


def global_verdict(cross):
    done = cross[cross["status"] == "done"].copy()
    if len(done) == 0:
        return "FAIL_SOBV4_NO_MODELS"

    verdicts = done["verdict"].astype(str).tolist()
    pass_strong = sum(v.startswith("PASS_STRONG") for v in verdicts)
    pass_any = sum(v.startswith("PASS") for v in verdicts)
    partial = sum(v.startswith("PARTIAL") for v in verdicts)

    if pass_strong == len(done):
        return "PASS_STRONG_SOBV4_CROSS_MODEL"
    if pass_any == len(done):
        return "PASS_SOBV4_CROSS_MODEL"
    if pass_any + partial == len(done):
        return "PARTIAL_SOBV4_CROSS_MODEL"
    if pass_any >= 2:
        return "MIXED_POSITIVE_SOBV4"
    if pass_any >= 1:
        return "MIXED_SOBV4"
    return "FAIL_SOBV4"


def main():
    set_seed()
    print("=" * 80)
    print(f"[{EXPERIMENT_ID}] Meaning as Future Cbit Gain")
    print(f"Device: {DEVICE}, dtype: {DTYPE}")
    print(f"Output: {OUT_DIR}")
    print("=" * 80)

    current_dataset = build_current_dataset()
    future_dataset = build_future_dataset()

    current_dataset_path = OUT_DIR / f"{EXPERIMENT_ID}_current_dataset.csv"
    future_dataset_path = OUT_DIR / f"{EXPERIMENT_ID}_future_dataset.csv"
    current_dataset.to_csv(current_dataset_path, index=False, encoding="utf-8-sig")
    future_dataset.to_csv(future_dataset_path, index=False, encoding="utf-8-sig")

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
        "topk": TOPK,
        "max_input_len": MAX_INPUT_LEN,
        "max_new_tokens": MAX_NEW_TOKENS,
        "random_seed": RANDOM_SEED,
    }
    with open(OUT_DIR / f"{EXPERIMENT_ID}_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    model_infos = []
    for model_key in MODELS_TO_RUN:
        print("=" * 80)
        print(f"[{EXPERIMENT_ID}] Running model: {model_key}")
        info = run_one_model(model_key, current_dataset, future_dataset)
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
            "current_dataset": str(current_dataset_path),
            "future_dataset": str(future_dataset_path),
            "cross_model_summary": str(cross_path),
            "out_dir": str(OUT_DIR),
        },
        "interpretation": {
            "Cbit_future_proxy": "Deterministic future-task structural compression score.",
            "TQ_future_proxy": "Future trajectory quality proxy.",
            "AQ_future_proxy": "Future answer usefulness proxy.",
            "Meaning_as_future_Cbit_gain": "Current internal visibility predicts future Cbit/TQ/AQ.",
            "PASS_STRONG_MEANING_AS_FUTURE_CBIT_GAIN": "Internal visibility predicts future Cbit and beats rank/category baselines with TQ support.",
            "PASS_MEANING_AS_FUTURE_CBIT_GAIN": "Internal visibility predicts future Cbit with future utility support.",
            "PARTIAL_MEANING_AS_FUTURE_CBIT_GAIN": "Some future utility prediction exists but not all criteria pass.",
        },
    }

    with open(OUT_DIR / f"{EXPERIMENT_ID}_verdict.json", "w", encoding="utf-8") as f:
        json.dump(global_obj, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(json.dumps(global_obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
