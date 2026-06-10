# -*- coding: utf-8 -*-
r"""
SOB-V4B: Relevance-Controlled Meaning Audit

Purpose
-------
SOB-V4 failed the direct hypothesis:

    V_internal(S,O) -> Future Cbit Gain

Interpretation:
    Constraint Visibility V(S,O) is not Meaning itself.
    V is a necessary precursor, but future utility also requires:
      1. Relevance(O,T_future)
      2. Use(S,O,T_future)

SOB-V4B tests the corrected hypothesis:

    Meaning(S,O,T) ≈ V(S,O) × Relevance(O,T) × Use(S,O,T)

where:
    Relevance is experimentally assigned by future task type:
        high / medium / low
    Use is scored from whether generated answer actually uses the object O
        in a structurally valid way.

Main hypotheses
---------------
H1:
    V alone is weak.

H2:
    V × Relevance predicts Future Cbit better than V alone.

H3:
    V × Relevance × Use predicts Future Cbit / TQ / AQ best.

H4:
    High relevance tasks should show stronger V -> FutureCbit relation
    than medium / low relevance tasks.

Output
------
C:\Users\ZH\Desktop\AGI\outputs\SOB-V4B

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

EXPERIMENT_ID = "SOB-V4B"
OUT_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\SOB-V4B")
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

RELEVANCE_WEIGHT = {
    "high": 1.0,
    "medium": 0.55,
    "low": 0.10,
}

OBJECTS = {
    "derivative": {
        "name": "derivative",
        "terms": {
            "symbol": ["d/dx", "f'", "prime", "derivative", "differentiation"],
            "core": ["limit", "instantaneous", "rate of change", "tangent", "slope", "local linear", "differentiable"],
            "use": ["local change", "instantaneous change", "slope", "rate", "sensitivity", "marginal", "gradient"],
        },
        "core_question": "Think about derivative as an object: definition, local change, tangent geometry, and edge cases.",
        "future_tasks": [
            ("high", "H1_velocity", "A vehicle position changes over time. Use derivative to explain instantaneous velocity."),
            ("high", "H2_gradient_descent", "A loss function changes as parameters change. Use derivative to explain why gradient descent can reduce loss."),
            ("medium", "M1_marginal_cost", "A business studies cost as production changes. Derivative may help, but other tools may also help. Explain whether and how it helps."),
            ("medium", "M2_sensitivity", "A climate model output changes with one parameter. Explain how derivative-like local sensitivity may help."),
            ("low", "L1_historical_cause", "Explain why a historical revolution happened. Do not force derivative unless it is genuinely useful."),
            ("low", "L2_poem_theme", "Explain the theme of a poem. Do not force derivative unless it is genuinely useful."),
        ],
    },
    "integral": {
        "name": "integral",
        "terms": {
            "symbol": ["integral", "∫", "dx", "antiderivative"],
            "core": ["Riemann sum", "area", "accumulation", "partition", "antiderivative", "convergence"],
            "use": ["accumulation", "total amount", "area under", "sum over", "expected value", "aggregate"],
        },
        "core_question": "Think about integral as an object: accumulation, area, continuous summation, and convergence.",
        "future_tasks": [
            ("high", "H1_accumulated_distance", "Velocity is known over time. Use integral to explain total distance or displacement."),
            ("high", "H2_probability_density", "A probability density is given. Use integral to explain total probability or expected value."),
            ("medium", "M1_total_cost", "A cost rate changes over time. Integral may help estimate total cost. Explain when it helps."),
            ("medium", "M2_population_growth", "A growth rate varies over time. Explain how accumulation could help understand total population change."),
            ("low", "L1_legal_argument", "Analyze a legal argument about responsibility. Do not force integral unless genuinely useful."),
            ("low", "L2_visual_style", "Describe the style of a painting. Do not force integral unless genuinely useful."),
        ],
    },
    "gradient": {
        "name": "gradient",
        "terms": {
            "symbol": ["gradient", "∇", "nabla", "partial derivative"],
            "core": ["partial derivative", "directional derivative", "steepest ascent", "level set", "critical point"],
            "use": ["steepest", "direction of change", "optimization", "loss landscape", "update", "sensitivity"],
        },
        "core_question": "Think about gradient as an object: direction of steepest change, partial derivatives, level sets, and critical points.",
        "future_tasks": [
            ("high", "H1_optimization", "A model must minimize a loss function. Use gradient to explain direction selection."),
            ("high", "H2_surface_navigation", "You are on a height landscape. Use gradient to explain steepest ascent or descent."),
            ("medium", "M1_policy_pressure", "A policy decision has several variables. Gradient-like directional pressure may help. Explain when it helps."),
            ("medium", "M2_resource_allocation", "Resources affect outcomes. Explain how a gradient-like idea might guide marginal changes."),
            ("low", "L1_story_character", "Analyze a fictional character's motivation. Do not force gradient unless genuinely useful."),
            ("low", "L2_food_recipe", "Explain how to make a soup. Do not force gradient unless genuinely useful."),
        ],
    },
    "eigenvector": {
        "name": "eigenvector",
        "terms": {
            "symbol": ["eigenvector", "eigenvalue", "lambda", "Av"],
            "core": ["invariant direction", "linear transformation", "scale", "matrix", "characteristic"],
            "use": ["invariant direction", "stable direction", "principal component", "mode", "spectral", "unchanged direction"],
        },
        "core_question": "Think about eigenvector as an object: invariant direction, scaling under transformation, and degeneracy.",
        "future_tasks": [
            ("high", "H1_pca", "Data vary in many directions. Use eigenvectors to explain principal components."),
            ("high", "H2_stability", "A linear dynamical system repeats a transformation. Use eigenvectors to explain stable modes."),
            ("medium", "M1_org_strategy", "An organization changes but some strategic direction remains stable. Use eigenvector as analogy only if helpful."),
            ("medium", "M2_network_influence", "A network has repeated influence propagation. Explain whether eigenvector-like centrality helps."),
            ("low", "L1_poetry_rhyme", "Analyze rhyme in a poem. Do not force eigenvector unless genuinely useful."),
            ("low", "L2_cooking", "Explain how to roast vegetables. Do not force eigenvector unless genuinely useful."),
        ],
    },
    "manifold": {
        "name": "manifold",
        "terms": {
            "symbol": ["manifold", "chart", "atlas", "surface"],
            "core": ["locally Euclidean", "coordinate", "tangent space", "curvature", "geodesic", "local patch"],
            "use": ["local patch", "global structure", "latent space", "state space", "configuration space", "local coordinates"],
        },
        "core_question": "Think about manifold as an object: local coordinates, charts, tangent spaces, curvature, and boundary cases.",
        "future_tasks": [
            ("high", "H1_data_manifold", "Data lie on a lower-dimensional structure. Use manifold to explain latent structure."),
            ("high", "H2_robot_configuration", "A robot arm has constrained configurations. Use manifold to explain configuration space."),
            ("medium", "M1_scientific_field", "A scientific field has local methods and global structure. Explain whether manifold is a helpful analogy."),
            ("medium", "M2_organization_space", "An organization has local teams and global coordination. Explain if manifold-like local/global structure helps."),
            ("low", "L1_basic_arithmetic", "Explain why 2+2=4. Do not force manifold unless genuinely useful."),
            ("low", "L2_movie_review", "Review a comedy movie. Do not force manifold unless genuinely useful."),
        ],
    },
}

SUBJECT_STATES = [
    {
        "state_id": "S0",
        "state_rank": 0,
        "state_name": "symbol_only",
        "profile": "You have only seen the name or symbol. You do not know definition, calculation, geometry, edge cases, or applications.",
    },
    {
        "state_id": "S1",
        "state_rank": 1,
        "state_name": "definition_known",
        "profile": "You know only the formal definition and a few basic words. You cannot reliably calculate, reason geometrically, or transfer.",
    },
    {
        "state_id": "S2",
        "state_rank": 2,
        "state_name": "calculation_ability",
        "profile": "You know definition and standard procedures. Geometry, edge cases, and transfer are limited.",
    },
    {
        "state_id": "S3",
        "state_rank": 3,
        "state_name": "geometric_understanding",
        "profile": "You know definition, procedures, geometry, and edge cases. Cross-domain transfer is not yet expert-level.",
    },
    {
        "state_id": "S4",
        "state_rank": 4,
        "state_name": "cross_domain_transfer",
        "profile": "You understand the object as a transferable structural tool across domains.",
    },
]

# ============================================================
# Utility
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

def normalize_text(s):
    return str(s).lower().replace("’", "'").replace("“", '"').replace("”", '"')

def count_words(s):
    return len(re.findall(r"\b[\w'-]+\b", str(s)))

def lexical_diversity(text):
    words = re.findall(r"\b[a-zA-Z][a-zA-Z'-]*\b", normalize_text(text))
    if not words:
        return 0.0
    return float(len(set(words)) / max(1, len(words)))

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

# ============================================================
# Dataset
# ============================================================

def build_current_prompt(obj, subject_state):
    return (
        "Internal-state probe only. Do not produce an explanatory answer.\n\n"
        "Read the learner state and object. Silently form the internal representation that would be needed.\n\n"
        f"Learner state:\n{subject_state['profile']}\n\n"
        f"Object of cognition: {obj['name']}.\n\n"
        f"Probe task:\n{obj['core_question']}\n\n"
        "End with exactly this token: READY"
    )

def build_future_prompt(obj, subject_state, relevance, task_id, task_text):
    return (
        "You are participating in a relevance-controlled meaning audit.\n\n"
        "Role-play the assigned learner state exactly. "
        "Use the studied object only if it is genuinely relevant. "
        "For low-relevance tasks, do not force the object; say it is not useful if appropriate.\n\n"
        f"Learner state:\n{subject_state['profile']}\n\n"
        f"Studied object: {obj['name']}.\n\n"
        f"Experimentally assigned relevance: {relevance}\n"
        f"Future task ID: {task_id}\n"
        f"Future task:\n{task_text}\n\n"
        "Answer concisely. Focus on whether the studied object helps compress the future task."
    )

def build_current_dataset():
    rows = []
    for obj_key, obj in OBJECTS.items():
        for st in SUBJECT_STATES:
            rows.append({
                "row_id": len(rows),
                "object_key": obj_key,
                "object_name": obj["name"],
                "state_id": st["state_id"],
                "state_rank": st["state_rank"],
                "state_name": st["state_name"],
                "prompt": build_current_prompt(obj, st),
            })
    return pd.DataFrame(rows)

def build_future_dataset():
    rows = []
    for obj_key, obj in OBJECTS.items():
        for st in SUBJECT_STATES:
            for relevance, task_id, task_text in obj["future_tasks"]:
                rows.append({
                    "future_row_id": len(rows),
                    "object_key": obj_key,
                    "object_name": obj["name"],
                    "state_id": st["state_id"],
                    "state_rank": st["state_rank"],
                    "state_name": st["state_name"],
                    "relevance": relevance,
                    "relevance_weight": RELEVANCE_WEIGHT[relevance],
                    "future_task_id": task_id,
                    "future_task": task_text,
                    "prompt": build_future_prompt(obj, st, relevance, task_id, task_text),
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
        for cat, terms in obj["terms"].items():
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

def object_category_scores(h, W, cat_ids, obj_key):
    logits = h @ W.T
    cats = ["symbol", "core", "use"]
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
    feat = {"n_hidden_states": len(hs), "n_layers": len(hs)-1}

    for wname, hidxs in windows.items():
        cat_seq = {cat: [] for cat in ["symbol", "core", "use"]}
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
        cat_probs = np.array(cat_probs)
        mean_probs = cat_probs.mean(axis=0)

        for j, cat in enumerate(cats):
            feat[f"{wname}_catprob_{cat}_mean"] = float(mean_probs[j])
            vals = np.asarray(cat_seq[cat], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            feat[f"{wname}_catlogit_{cat}_mean"] = float(vals.mean()) if vals.size else np.nan

        feat[f"{wname}_cat_entropy_mean"] = float(np.mean([entropy_bits(p) for p in cat_probs]))
        feat[f"{wname}_core_mass"] = float(mean_probs[cats.index("core")])
        feat[f"{wname}_use_mass"] = float(mean_probs[cats.index("use")])
        feat[f"{wname}_symbol_mass"] = float(mean_probs[cats.index("symbol")])
        feat[f"{wname}_use_minus_symbol"] = feat[f"{wname}_use_mass"] - feat[f"{wname}_symbol_mass"]
        feat[f"{wname}_use_minus_core"] = feat[f"{wname}_use_mass"] - feat[f"{wname}_core_mass"]

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

    feat["V_core_current"] = float(np.nanmean([feat.get("commit_core_mass", np.nan), feat.get("middle_core_mass", np.nan)]))
    feat["V_use_current"] = float(np.nanmean([feat.get("commit_use_mass", np.nan), feat.get("middle_use_mass", np.nan)]))
    feat["V_symbol_current"] = float(np.nanmean([feat.get("init_symbol_mass", np.nan), feat.get("commit_symbol_mass", np.nan)]))
    return feat

# ============================================================
# Scoring
# ============================================================

def score_future_answer(row):
    text = str(row.get("answer_text", ""))
    obj_key = row["object_key"]
    obj = OBJECTS[obj_key]
    relevance = row["relevance"]
    task = row["future_task"]

    core_terms = obj["terms"]["core"]
    use_terms = obj["terms"]["use"]
    object_terms = obj["terms"]["symbol"] + core_terms + use_terms
    general_structural = ["because", "therefore", "structure", "constraint", "mapping", "principle", "helps", "not useful", "not relevant", "only if"]

    object_s, object_hits = marker_score(text, object_terms)
    core_s, core_hits = marker_score(text, core_terms)
    use_s, use_hits = marker_score(text, use_terms)
    general_s, general_hits = marker_score(text, general_structural)

    n = count_words(text)
    div = lexical_diversity(text)
    comp = compression_quality(text)

    negation_hits = marker_hits(text, ["not useful", "not relevant", "do not force", "does not help", "not the right tool", "limited relevance"])
    force_hits = marker_hits(text, ["can be used", "helps", "use", "apply", "explains", "maps", "analogous"])

    # Use score: object is actually used structurally.
    raw_use = 0.55 * use_s + 0.25 * core_s + 0.10 * general_s + 0.10 * div
    object_use_score = max(0.0, min(1.0, raw_use))

    # Correct-use respects relevance. For low relevance, correctly rejecting forced use is good.
    if relevance == "low":
        if negation_hits:
            relevance_alignment = 1.0
            corrected_use = 0.15 + 0.70 * min(1.0, len(negation_hits) / 2.0)
        else:
            relevance_alignment = max(0.0, 1.0 - object_use_score)
            corrected_use = max(0.0, 0.35 - 0.35 * object_use_score)
    elif relevance == "medium":
        relevance_alignment = 0.5 + 0.5 * object_use_score
        corrected_use = 0.45 * object_use_score + 0.25 * general_s + 0.15 * comp + 0.15 * div
    else:
        relevance_alignment = object_use_score
        corrected_use = 0.58 * object_use_score + 0.18 * general_s + 0.14 * comp + 0.10 * div

    corrected_use = max(0.0, min(1.0, corrected_use))
    relevance_alignment = max(0.0, min(1.0, relevance_alignment))

    # Cbit future: rewards appropriate compression of future task.
    cbit_future = (
        0.45 * corrected_use +
        0.25 * relevance_alignment +
        0.15 * general_s +
        0.10 * comp +
        0.05 * div
    )

    tq_future = (
        0.38 * corrected_use +
        0.22 * object_use_score +
        0.20 * relevance_alignment +
        0.10 * general_s +
        0.10 * comp
    )

    aq_future = (
        0.38 * corrected_use +
        0.22 * relevance_alignment +
        0.18 * general_s +
        0.12 * comp +
        0.10 * div
    )

    cbit_future = max(0.0, min(1.0, cbit_future))
    tq_future = max(0.0, min(1.0, tq_future))
    aq_future = max(0.0, min(1.0, aq_future))

    # Success definition is relevance-aware.
    future_success = int(cbit_future >= 0.48 and relevance_alignment >= 0.45)

    return {
        "future_word_count": n,
        "object_marker_score": object_s,
        "core_marker_score": core_s,
        "use_marker_score": use_s,
        "general_structural_score": general_s,
        "object_hits": "; ".join(object_hits),
        "use_hits": "; ".join(use_hits),
        "negation_hits": "; ".join(negation_hits),
        "force_hits": "; ".join(force_hits),
        "future_compression_quality": comp,
        "future_lexical_diversity": div,
        "object_use_score": object_use_score,
        "relevance_alignment": relevance_alignment,
        "corrected_use_score": corrected_use,
        "Cbit_future_proxy": cbit_future,
        "TQ_future_proxy": tq_future,
        "AQ_future_proxy": aq_future,
        "future_success": future_success,
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
        "init_use_", "middle_use_", "commit_use_", "final_use_",
        "init_symbol_", "middle_symbol_", "commit_symbol_", "final_symbol_",
        "V_core_current", "V_use_current", "V_symbol_current",
    ])
    topk_cols = cols_by_prefix(df, ["init_topk_", "middle_topk_", "competition_topk_", "commit_topk_", "final_topk_"])
    hidden_cols = cols_by_prefix(df, [
        "init_hidden_", "middle_hidden_", "competition_hidden_", "commit_hidden_", "final_hidden_",
        "init_hmean_", "middle_hmean_", "competition_hmean_", "commit_hmean_", "final_hmean_",
        "init_hstd_", "middle_hstd_", "competition_hstd_", "commit_hstd_", "final_hstd_",
    ])
    all_cols = unique_keep_order(category_cols + topk_cols + hidden_cols)
    topk_hidden_cols = unique_keep_order(topk_cols + hidden_cols)
    interaction_cols = [
        "Meaning_VR", "Meaning_VRU", "Meaning_rank_relevance",
        "V_use_current", "relevance_weight", "corrected_use_score", "object_use_score"
    ]
    return {
        "VOnly": ["V_use_current", "V_core_current"],
        "VxRelevance": ["Meaning_VR", "V_use_current", "relevance_weight"],
        "VxRelevancexUse": ["Meaning_VRU", "Meaning_VR", "V_use_current", "relevance_weight", "corrected_use_score", "object_use_score"],
        "RankOnly": ["state_rank"],
        "RankxRelevance": ["Meaning_rank_relevance", "state_rank", "relevance_weight"],
        "CategoryOnly": category_cols,
        "TopKPlusHidden": topk_hidden_cols,
        "AllInternal": all_cols,
        "AllMeaning": unique_keep_order(all_cols + interaction_cols + ["state_rank"]),
    }

def make_analysis_table(current_features, future_scored):
    # Current: one row per object/state.
    num_cols = [c for c in current_features.columns if pd.api.types.is_numeric_dtype(current_features[c])]
    keep_num = [c for c in num_cols if c not in ["row_id", "state_rank"]]
    cur = current_features.groupby(["model_key", "object_key", "object_name", "state_id", "state_rank", "state_name"], as_index=False)[keep_num].mean()

    fut = future_scored.copy()
    merged = fut.merge(
        cur,
        on=["model_key", "object_key", "object_name", "state_id", "state_rank", "state_name"],
        how="inner",
    )

    merged["Meaning_VR"] = merged["V_use_current"] * merged["relevance_weight"]
    merged["Meaning_VRU"] = merged["V_use_current"] * merged["relevance_weight"] * merged["corrected_use_score"]
    merged["Meaning_rank_relevance"] = merged["state_rank"].astype(float) * merged["relevance_weight"]
    return merged

def predictive_summary(analysis_df):
    groups = build_feature_groups(analysis_df)
    targets = ["Cbit_future_proxy", "TQ_future_proxy", "AQ_future_proxy"]
    rows = []
    for target in targets:
        for gname, cols in groups.items():
            res = cv_regression(analysis_df, cols, target, group_col="object_key")
            rows.append({
                "target": target,
                "feature_group": gname,
                "task": "regression",
                "valid": res is not None,
                "n": res["n"] if res else np.nan,
                "n_features": res["n_features"] if res else len(cols),
                "r2": res["r2"] if res else np.nan,
                "corr": res["corr"] if res else np.nan,
                "auc": np.nan,
                "macro_f1": np.nan,
            })

    for gname, cols in groups.items():
        res = cv_binary(analysis_df, cols, "future_success", group_col="object_key")
        rows.append({
            "target": "future_success",
            "feature_group": gname,
            "task": "classification",
            "valid": res is not None,
            "n": res["n"] if res else np.nan,
            "n_features": res["n_features"] if res else len(cols),
            "r2": np.nan,
            "corr": np.nan,
            "auc": res["auc"] if res else np.nan,
            "macro_f1": res["macro_f1"] if res else np.nan,
            "balanced_acc": res["balanced_acc"] if res else np.nan,
        })

    # Within relevance strata: V_use_current -> Cbit_future
    for relevance, sub in analysis_df.groupby("relevance"):
        res = cv_regression(sub, ["V_use_current", "V_core_current"], "Cbit_future_proxy", group_col="object_key")
        rows.append({
            "target": f"Cbit_future_proxy_{relevance}",
            "feature_group": "VOnly_within_relevance",
            "task": "stratified_regression",
            "valid": res is not None,
            "n": res["n"] if res else np.nan,
            "n_features": res["n_features"] if res else 2,
            "r2": res["r2"] if res else np.nan,
            "corr": res["corr"] if res else np.nan,
            "auc": np.nan,
            "macro_f1": np.nan,
        })

    return pd.DataFrame(rows)

def verdict_from_prediction(pred):
    def corr(target, group):
        x = pred[(pred["target"] == target) & (pred["feature_group"] == group) & (pred["valid"] == True)]
        if len(x) == 0:
            return np.nan
        return safe_float(x["corr"].iloc[0])

    def auc(target, group):
        x = pred[(pred["target"] == target) & (pred["feature_group"] == group) & (pred["valid"] == True)]
        if len(x) == 0:
            return np.nan
        return safe_float(x["auc"].iloc[0])

    c_v = corr("Cbit_future_proxy", "VOnly")
    c_vr = corr("Cbit_future_proxy", "VxRelevance")
    c_vru = corr("Cbit_future_proxy", "VxRelevancexUse")
    c_rank = corr("Cbit_future_proxy", "RankOnly")
    c_rank_rel = corr("Cbit_future_proxy", "RankxRelevance")
    c_topk = corr("Cbit_future_proxy", "TopKPlusHidden")
    c_allmeaning = corr("Cbit_future_proxy", "AllMeaning")

    tq_vru = corr("TQ_future_proxy", "VxRelevancexUse")
    aq_vru = corr("AQ_future_proxy", "VxRelevancexUse")
    auc_vru = auc("future_success", "VxRelevancexUse")
    auc_allmeaning = auc("future_success", "AllMeaning")

    high_corr = corr("Cbit_future_proxy_high", "VOnly_within_relevance")
    med_corr = corr("Cbit_future_proxy_medium", "VOnly_within_relevance")
    low_corr = corr("Cbit_future_proxy_low", "VOnly_within_relevance")

    relevance_improves = bool(np.isfinite(c_vr) and np.isfinite(c_v) and c_vr > c_v + 0.05)
    use_improves = bool(np.isfinite(c_vru) and np.isfinite(c_vr) and c_vru > c_vr + 0.05)
    meaning_predicts = bool(np.isfinite(c_vru) and c_vru >= 0.45)
    meaning_beats_rank = bool(np.isfinite(c_vru) and np.isfinite(c_rank_rel) and c_vru > c_rank_rel + 0.05)
    utility_support = bool(
        (np.isfinite(tq_vru) and tq_vru >= 0.35) or
        (np.isfinite(aq_vru) and aq_vru >= 0.35) or
        (np.isfinite(auc_vru) and auc_vru >= 0.65) or
        (np.isfinite(auc_allmeaning) and auc_allmeaning >= 0.65)
    )
    relevance_gradient = bool(np.isfinite(high_corr) and np.isfinite(low_corr) and high_corr > low_corr)

    if meaning_predicts and relevance_improves and utility_support:
        verdict = "PASS_STRONG_RELEVANCE_CONTROLLED_MEANING"
    elif meaning_predicts and utility_support:
        verdict = "PASS_RELEVANCE_CONTROLLED_MEANING"
    elif relevance_improves or use_improves or relevance_gradient:
        verdict = "PARTIAL_RELEVANCE_CONTROLLED_MEANING"
    else:
        verdict = "FAIL_RELEVANCE_CONTROLLED_MEANING"

    return {
        "verdict": verdict,
        "cbit_corr_VOnly": c_v,
        "cbit_corr_VxRelevance": c_vr,
        "cbit_corr_VxRelevancexUse": c_vru,
        "cbit_corr_RankOnly": c_rank,
        "cbit_corr_RankxRelevance": c_rank_rel,
        "cbit_corr_TopKPlusHidden": c_topk,
        "cbit_corr_AllMeaning": c_allmeaning,
        "tq_corr_VxRelevancexUse": tq_vru,
        "aq_corr_VxRelevancexUse": aq_vru,
        "future_success_auc_VxRelevancexUse": auc_vru,
        "future_success_auc_AllMeaning": auc_allmeaning,
        "high_relevance_V_corr": high_corr,
        "medium_relevance_V_corr": med_corr,
        "low_relevance_V_corr": low_corr,
        "relevance_improves_over_V": relevance_improves,
        "use_improves_over_VR": use_improves,
        "meaning_predicts_future_cbit": meaning_predicts,
        "meaning_beats_rank_relevance": meaning_beats_rank,
        "utility_support": utility_support,
        "relevance_gradient": relevance_gradient,
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
        info.update({"status": "missing_path", "error": f"Path not found: {spec['path']}"})
        return info

    model = None
    try:
        t0 = time.time()
        model, tokenizer = load_model_and_tokenizer(spec["path"], spec.get("trust_remote_code", True))
        W = lm_head_weight(model)
        cat_ids = category_token_ids(tokenizer)

        current_rows, current_errors = [], []
        for i, row in current_dataset.iterrows():
            if i % 10 == 0:
                print(f"[{model_key}] current row {i}/{len(current_dataset)}", flush=True)
            try:
                feat = extract_current_features(row, model, tokenizer, W, cat_ids, model_key)
                current_rows.append({**row.to_dict(), "model_key": model_key, "model_name": spec["model_name"], **feat})
            except Exception as e:
                current_errors.append({
                    "row_id": int(row["row_id"]),
                    "object_key": row["object_key"],
                    "state_id": row["state_id"],
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })

        current_features = pd.DataFrame(current_rows)

        future_rows, future_errors = [], []
        for i, row in future_dataset.iterrows():
            if i % 25 == 0:
                print(f"[{model_key}] future row {i}/{len(future_dataset)}", flush=True)
            try:
                ans = generate_answer(model, tokenizer, row["prompt"])
                future_rows.append({**row.to_dict(), "model_key": model_key, "model_name": spec["model_name"], "answer_text": ans})
            except Exception as e:
                future_errors.append({
                    "future_row_id": int(row["future_row_id"]),
                    "object_key": row["object_key"],
                    "state_id": row["state_id"],
                    "future_task_id": row["future_task_id"],
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })

        future_answers = pd.DataFrame(future_rows)
        future_scored = score_future_df(future_answers) if not future_answers.empty else pd.DataFrame()

        current_features.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_current_internal_features.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(current_errors).to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_current_errors.csv", index=False, encoding="utf-8-sig")
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

        analysis_df = make_analysis_table(current_features, future_scored)
        pred = predictive_summary(analysis_df)
        verdict = verdict_from_prediction(pred)

        analysis_df.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_analysis_table.csv", index=False, encoding="utf-8-sig")
        pred.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_prediction_summary.csv", index=False, encoding="utf-8-sig")
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
                "analysis_table": str(model_out / f"{EXPERIMENT_ID}_{model_key}_analysis_table.csv"),
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
            "cbit_corr_VOnly": m.get("cbit_corr_VOnly"),
            "cbit_corr_VxRelevance": m.get("cbit_corr_VxRelevance"),
            "cbit_corr_VxRelevancexUse": m.get("cbit_corr_VxRelevancexUse"),
            "cbit_corr_RankOnly": m.get("cbit_corr_RankOnly"),
            "cbit_corr_RankxRelevance": m.get("cbit_corr_RankxRelevance"),
            "cbit_corr_TopKPlusHidden": m.get("cbit_corr_TopKPlusHidden"),
            "cbit_corr_AllMeaning": m.get("cbit_corr_AllMeaning"),
            "tq_corr_VxRelevancexUse": m.get("tq_corr_VxRelevancexUse"),
            "aq_corr_VxRelevancexUse": m.get("aq_corr_VxRelevancexUse"),
            "future_success_auc_VxRelevancexUse": m.get("future_success_auc_VxRelevancexUse"),
            "future_success_auc_AllMeaning": m.get("future_success_auc_AllMeaning"),
            "high_relevance_V_corr": m.get("high_relevance_V_corr"),
            "medium_relevance_V_corr": m.get("medium_relevance_V_corr"),
            "low_relevance_V_corr": m.get("low_relevance_V_corr"),
            "relevance_improves_over_V": m.get("relevance_improves_over_V"),
            "use_improves_over_VR": m.get("use_improves_over_VR"),
            "meaning_predicts_future_cbit": m.get("meaning_predicts_future_cbit"),
            "meaning_beats_rank_relevance": m.get("meaning_beats_rank_relevance"),
            "utility_support": m.get("utility_support"),
            "relevance_gradient": m.get("relevance_gradient"),
            "n_current_rows": info.get("n_current_rows"),
            "n_future_rows": info.get("n_future_rows"),
            "n_errors": info.get("n_errors"),
            "error": info.get("error"),
        })
    return pd.DataFrame(rows)

def global_verdict(cross):
    done = cross[cross["status"] == "done"].copy()
    if len(done) == 0:
        return "FAIL_SOBV4B_NO_MODELS"
    verdicts = done["verdict"].astype(str).tolist()
    pass_strong = sum(v.startswith("PASS_STRONG") for v in verdicts)
    pass_any = sum(v.startswith("PASS") for v in verdicts)
    partial = sum(v.startswith("PARTIAL") for v in verdicts)

    if pass_strong == len(done):
        return "PASS_STRONG_SOBV4B_CROSS_MODEL"
    if pass_any == len(done):
        return "PASS_SOBV4B_CROSS_MODEL"
    if pass_any + partial == len(done):
        return "PARTIAL_SOBV4B_CROSS_MODEL"
    if pass_any >= 2:
        return "MIXED_POSITIVE_SOBV4B"
    if pass_any >= 1:
        return "MIXED_SOBV4B"
    return "FAIL_SOBV4B"

def main():
    set_seed()
    print("=" * 80)
    print(f"[{EXPERIMENT_ID}] Relevance-Controlled Meaning Audit")
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
        "relevance_weight": RELEVANCE_WEIGHT,
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
            "VOnly": "Current visibility alone.",
            "VxRelevance": "Visibility multiplied by experimentally assigned relevance.",
            "VxRelevancexUse": "Visibility multiplied by relevance and actual object-use score.",
            "Meaning": "Operationalized as V(S,O) × Relevance(O,T) × Use(S,O,T).",
            "PASS_STRONG_RELEVANCE_CONTROLLED_MEANING": "Meaning product predicts future Cbit and improves over V alone with utility support.",
            "PASS_RELEVANCE_CONTROLLED_MEANING": "Meaning product predicts future Cbit with utility support.",
            "PARTIAL_RELEVANCE_CONTROLLED_MEANING": "Relevance or use improves prediction but full future Cbit criterion is not met.",
        },
    }

    with open(OUT_DIR / f"{EXPERIMENT_ID}_verdict.json", "w", encoding="utf-8") as f:
        json.dump(global_obj, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(json.dumps(global_obj, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
