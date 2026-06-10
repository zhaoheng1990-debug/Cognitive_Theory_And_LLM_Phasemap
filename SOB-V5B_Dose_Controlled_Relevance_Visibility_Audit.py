# -*- coding: utf-8 -*-
r"""
SOB-V5B: Dose-Controlled Relevance-Conditioned Visibility Intervention

Purpose
-------
SOB-V5A produced a partial causal result:

    do(V ↑) -> Use_pre_hat ↑
    do(V ↑) -> Meaning_pre ↑

but did not strongly establish:

    do(V ↑) -> Cbit_future ↑

Main limitations in V5A:
  1. One single visibility lesson dose.
  2. No sham lesson control.
  3. Cbit score mixed useful object-use and correct non-use.
  4. Low-relevance overuse risk was not isolated.

SOB-V5B fixes this.

Core hypotheses
---------------
H1 Dose-response:
    Visibility dose ↑ -> Use_pre_hat ↑ -> Meaning_pre ↑

H2 Relevance-conditioned causal effect:
    In high/medium relevance tasks:
        dose ↑ -> useful_use_Cbit ↑

H3 Safety:
    In low relevance tasks:
        dose ↑ should NOT increase overuse penalty.
        correct non-use should remain stable or improve.

H4 Sham control:
    true visibility lessons should outperform sham verbosity control.

Intervention arms
-----------------
control:
    no added lesson.

sham:
    matched generic study instruction with no object-specific constraint content.

visibility_low:
    one-sentence object constraint cue.

visibility_medium:
    definition + geometry/use constraints.

visibility_high:
    definition + geometry/use + boundary/non-use constraints.

Output
------
C:\Users\ZH\Desktop\AGI\outputs\SOB-V5B

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ============================================================
# Constants
# ============================================================

EXPERIMENT_ID = "SOB-V5B"
OUT_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\SOB-V5B")
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

MAX_INPUT_LEN = 1700
MAX_NEW_TOKENS = 220
TOPK = 80
N_SPLITS = 5

RELEVANCE_WEIGHT = {
    "high": 1.0,
    "medium": 0.55,
    "low": 0.10,
}

CONDITIONS = [
    ("control", 0.0, "none"),
    ("sham", 0.0, "sham"),
    ("visibility_low", 0.33, "true"),
    ("visibility_medium", 0.66, "true"),
    ("visibility_high", 1.0, "true"),
]

LOW_STATES = {"S0", "S1", "S2"}

OBJECTS = {
    "derivative": {
        "name": "derivative",
        "terms": {
            "symbol": ["d/dx", "f'", "prime", "derivative", "differentiation"],
            "core": ["limit", "instantaneous", "rate of change", "tangent", "slope", "local linear", "differentiable"],
            "use": ["local change", "instantaneous change", "slope", "rate", "sensitivity", "marginal", "gradient"],
        },
        "lesson_low": "Derivative means local rate of change.",
        "lesson_medium": (
            "A derivative makes local change visible: instantaneous rate, tangent slope, and local linear approximation."
        ),
        "lesson_high": (
            "A derivative makes local change visible: instantaneous rate, tangent slope, and local linear approximation. "
            "Use it for pointwise change, sensitivity, marginal change, and update direction. "
            "Do not force it for narrative, aesthetic, or historical tasks without a local-change structure."
        ),
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
        "lesson_low": "Integral means accumulation or continuous summation.",
        "lesson_medium": (
            "An integral makes accumulation visible: local contributions combine into a total amount, area, probability mass, or total change."
        ),
        "lesson_high": (
            "An integral makes accumulation visible: local contributions combine into a total amount, area, probability mass, or total change. "
            "Use it for accumulated distance, total cost, expected value, work, or aggregate quantity. "
            "Do not force it when the task has no accumulation structure."
        ),
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
        "lesson_low": "Gradient means direction of steepest change.",
        "lesson_medium": (
            "A gradient makes direction-of-change visible: steepest increase/decrease, level set normal, and local optimization direction."
        ),
        "lesson_high": (
            "A gradient makes direction-of-change visible: steepest increase/decrease, level set normal, and local optimization direction. "
            "Use it for optimization, loss landscapes, sensitivity to variables, or directional pressure. "
            "Do not force it for recipes, literary motives, or tasks without directional change."
        ),
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
        "lesson_low": "Eigenvector means an invariant direction under a transformation.",
        "lesson_medium": (
            "An eigenvector makes invariant direction visible: a linear transformation scales it without changing its direction."
        ),
        "lesson_high": (
            "An eigenvector makes invariant direction visible: a linear transformation scales it without changing its direction. "
            "Use it for stable modes, repeated transformations, PCA, spectral structure, or centrality. "
            "Do not force it when there is no transformation, stability, or repeated linear influence structure."
        ),
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
        "lesson_low": "Manifold means a complex space that is simple locally.",
        "lesson_medium": (
            "A manifold makes local-global structure visible: local coordinate patches connect into a global structure."
        ),
        "lesson_high": (
            "A manifold makes local-global structure visible: local coordinate patches connect into a global structure. "
            "Use it for curved surfaces, latent spaces, configuration spaces, state spaces, or local methods embedded in global structure. "
            "Do not force it for unrelated arithmetic, reviews, or tasks without local-global structure."
        ),
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

def intervention_text(obj, condition):
    if condition == "control":
        return ""
    if condition == "sham":
        return (
            "General study note: read the task carefully, avoid overclaiming, and answer concisely. "
            "Use only relevant ideas and avoid forcing unrelated concepts."
        )
    if condition == "visibility_low":
        return obj["lesson_low"]
    if condition == "visibility_medium":
        return obj["lesson_medium"]
    if condition == "visibility_high":
        return obj["lesson_high"]
    return ""

def build_current_prompt(obj, subject_state, condition):
    extra = intervention_text(obj, condition)
    extra_block = ""
    if extra:
        extra_block = f"\n\nIntervention note:\n{extra}\n"
    return (
        "Internal-state probe only. Do not produce an explanatory answer.\n\n"
        "Read the learner state, object, and possible intervention. Silently form the internal representation that would be needed.\n\n"
        f"Learner state:\n{subject_state['profile']}\n\n"
        f"Object of cognition: {obj['name']}.\n"
        f"{extra_block}\n"
        f"Probe task:\n{obj['core_question']}\n\n"
        "End with exactly this token: READY"
    )

def build_preuse_prompt(obj, subject_state, relevance, task_id, task_text, condition):
    extra = intervention_text(obj, condition)
    extra_block = ""
    if extra:
        extra_block = f"\n\nIntervention note:\n{extra}\n"
    return (
        "Internal pre-use prediction probe only. Do not answer the future task.\n\n"
        "Read the learner state, studied object, possible intervention, relevance label, and future task. "
        "Silently form the representation of whether this learner will actually use the studied object appropriately. "
        "End with exactly this token: READY\n\n"
        f"Learner state:\n{subject_state['profile']}\n\n"
        f"Studied object: {obj['name']}.\n"
        f"{extra_block}\n"
        f"Experimentally assigned relevance: {relevance}\n"
        f"Future task ID: {task_id}\n"
        f"Future task:\n{task_text}\n\n"
        "READY"
    )

def build_future_prompt(obj, subject_state, relevance, task_id, task_text, condition):
    extra = intervention_text(obj, condition)
    extra_block = ""
    if extra:
        extra_block = f"\n\nBefore answering, the learner receives this intervention note:\n{extra}\n"
    return (
        "You are participating in a dose-controlled relevance-conditioned meaning audit.\n\n"
        "Role-play the assigned learner state exactly, but include any intervention note if provided. "
        "Use the studied object only if it is genuinely relevant. "
        "For low-relevance tasks, do not force the object; say it is not useful if appropriate.\n\n"
        f"Learner state:\n{subject_state['profile']}\n\n"
        f"Studied object: {obj['name']}.\n"
        f"{extra_block}\n"
        f"Condition: {condition}\n"
        f"Experimentally assigned relevance: {relevance}\n"
        f"Future task ID: {task_id}\n"
        f"Future task:\n{task_text}\n\n"
        "Answer concisely. Focus on whether the studied object helps compress the future task."
    )

def build_datasets():
    current_rows = []
    preuse_rows = []
    future_rows = []

    for obj_key, obj in OBJECTS.items():
        for st in SUBJECT_STATES:
            for condition, dose, arm_type in CONDITIONS:
                current_rows.append({
                    "current_row_id": len(current_rows),
                    "object_key": obj_key,
                    "object_name": obj["name"],
                    "state_id": st["state_id"],
                    "state_rank": st["state_rank"],
                    "state_name": st["state_name"],
                    "condition": condition,
                    "dose": dose,
                    "is_true_visibility": int(arm_type == "true"),
                    "is_sham": int(arm_type == "sham"),
                    "low_state": int(st["state_id"] in LOW_STATES),
                    "prompt": build_current_prompt(obj, st, condition),
                })

                for relevance, task_id, task_text in obj["future_tasks"]:
                    common = {
                        "object_key": obj_key,
                        "object_name": obj["name"],
                        "state_id": st["state_id"],
                        "state_rank": st["state_rank"],
                        "state_name": st["state_name"],
                        "condition": condition,
                        "dose": dose,
                        "is_true_visibility": int(arm_type == "true"),
                        "is_sham": int(arm_type == "sham"),
                        "low_state": int(st["state_id"] in LOW_STATES),
                        "relevance": relevance,
                        "relevance_weight": RELEVANCE_WEIGHT[relevance],
                        "future_task_id": task_id,
                        "future_task": task_text,
                    }
                    preuse_rows.append({
                        "preuse_row_id": len(preuse_rows),
                        **common,
                        "prompt": build_preuse_prompt(obj, st, relevance, task_id, task_text, condition),
                    })
                    future_rows.append({
                        "future_row_id": len(future_rows),
                        **common,
                        "prompt": build_future_prompt(obj, st, relevance, task_id, task_text, condition),
                    })

    return pd.DataFrame(current_rows), pd.DataFrame(preuse_rows), pd.DataFrame(future_rows)

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

def make_binary_classifier():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=RANDOM_SEED)),
    ])

def cv_binary_probs(df, cols, target, group_col="object_key"):
    cols = numeric_cols(df, cols)
    d = df.dropna(subset=[target]).copy()
    if len(d) < 10 or not cols:
        return None
    y = d[target].astype(int).values
    if len(np.unique(y)) < 2:
        return None
    X = d[cols].replace([np.inf, -np.inf], np.nan).values
    prob = np.zeros(len(d), dtype=float)
    used = 0
    for train_idx, test_idx in cv_indices(d, y, group_col=group_col, stratify=True):
        if len(np.unique(y[train_idx])) < 2:
            continue
        clf = make_binary_classifier()
        clf.fit(X[train_idx], y[train_idx])
        prob[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
        used += 1
    if used == 0:
        return None
    return d.index.values, prob

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

def extract_features(row, model, tokenizer, W, cat_ids, model_key, prefix):
    hs = forward_hidden(model, tokenizer, row["prompt"])
    windows = windows_for_model(model_key, len(hs))
    obj_key = row["object_key"]
    feat = {f"{prefix}_n_hidden_states": len(hs), f"{prefix}_n_layers": len(hs)-1}

    for wname, hidxs in windows.items():
        wp = f"{prefix}_{wname}"
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
            feat[f"{wp}_catprob_{cat}_mean"] = float(mean_probs[j])
            vals = np.asarray(cat_seq[cat], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            feat[f"{wp}_catlogit_{cat}_mean"] = float(vals.mean()) if vals.size else np.nan

        feat[f"{wp}_cat_entropy_mean"] = float(np.mean([entropy_bits(p) for p in cat_probs]))
        feat[f"{wp}_core_mass"] = float(mean_probs[cats.index("core")])
        feat[f"{wp}_use_mass"] = float(mean_probs[cats.index("use")])
        feat[f"{wp}_symbol_mass"] = float(mean_probs[cats.index("symbol")])
        feat[f"{wp}_use_minus_symbol"] = feat[f"{wp}_use_mass"] - feat[f"{wp}_symbol_mass"]
        feat[f"{wp}_use_minus_core"] = feat[f"{wp}_use_mass"] - feat[f"{wp}_core_mass"]

        feat[f"{wp}_topk_entropy_mean"] = float(np.mean(topk_entropy_vals))
        feat[f"{wp}_topk_spread_mean"] = float(np.mean(topk_spread_vals))
        if len(topk_centers) >= 2:
            drifts = [1.0 - cosine(topk_centers[i], topk_centers[i+1]) for i in range(len(topk_centers)-1)]
            feat[f"{wp}_topk_center_drift_mean"] = float(np.mean(drifts))
            feat[f"{wp}_topk_center_drift_max"] = float(np.max(drifts))
        else:
            feat[f"{wp}_topk_center_drift_mean"] = 0.0
            feat[f"{wp}_topk_center_drift_max"] = 0.0

        hidden_summary_features(feat, wp, [hs[i] for i in hidxs])

    feat[f"{prefix}_V_core"] = float(np.nanmean([
        feat.get(f"{prefix}_commit_core_mass", np.nan),
        feat.get(f"{prefix}_middle_core_mass", np.nan)
    ]))
    feat[f"{prefix}_V_use"] = float(np.nanmean([
        feat.get(f"{prefix}_commit_use_mass", np.nan),
        feat.get(f"{prefix}_middle_use_mass", np.nan)
    ]))
    feat[f"{prefix}_V_symbol"] = float(np.nanmean([
        feat.get(f"{prefix}_init_symbol_mass", np.nan),
        feat.get(f"{prefix}_commit_symbol_mass", np.nan)
    ]))
    return feat

# ============================================================
# Scoring
# ============================================================

def score_future_answer(row):
    text = str(row.get("answer_text", ""))
    obj_key = row["object_key"]
    obj = OBJECTS[obj_key]
    relevance = row["relevance"]

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
    raw_use = 0.55 * use_s + 0.25 * core_s + 0.10 * general_s + 0.10 * div
    object_use_score = max(0.0, min(1.0, raw_use))

    # Correct non-use and overuse are explicitly separated.
    if relevance == "low":
        correct_nonuse_score = min(1.0, len(negation_hits) / 2.0) if negation_hits else max(0.0, 1.0 - object_use_score)
        useful_use_cbit = 0.0
        overuse_penalty = object_use_score * (0.85 if not negation_hits else 0.30)
        corrected_use = 0.15 + 0.75 * correct_nonuse_score
        relevance_alignment = correct_nonuse_score
    elif relevance == "medium":
        correct_nonuse_score = 0.0
        useful_use_cbit = 0.50 * object_use_score + 0.25 * general_s + 0.15 * comp + 0.10 * div
        overuse_penalty = 0.10 * max(0.0, object_use_score - 0.70)
        corrected_use = useful_use_cbit - overuse_penalty
        relevance_alignment = 0.45 + 0.55 * object_use_score
    else:
        correct_nonuse_score = 0.0
        useful_use_cbit = 0.58 * object_use_score + 0.18 * general_s + 0.14 * comp + 0.10 * div
        overuse_penalty = 0.0
        corrected_use = useful_use_cbit
        relevance_alignment = object_use_score

    useful_use_cbit = max(0.0, min(1.0, useful_use_cbit))
    overuse_penalty = max(0.0, min(1.0, overuse_penalty))
    corrected_use = max(0.0, min(1.0, corrected_use))
    relevance_alignment = max(0.0, min(1.0, relevance_alignment))

    # Total future Cbit rewards useful use in relevant tasks and correct non-use in low relevance.
    if relevance == "low":
        cbit_future = (
            0.55 * correct_nonuse_score +
            0.20 * general_s +
            0.15 * comp +
            0.10 * div -
            0.50 * overuse_penalty
        )
    else:
        cbit_future = (
            0.52 * useful_use_cbit +
            0.20 * relevance_alignment +
            0.13 * general_s +
            0.10 * comp +
            0.05 * div -
            0.20 * overuse_penalty
        )

    tq_future = (
        0.38 * corrected_use +
        0.22 * object_use_score +
        0.20 * relevance_alignment +
        0.10 * general_s +
        0.10 * comp -
        0.20 * overuse_penalty
    )

    aq_future = (
        0.38 * corrected_use +
        0.22 * relevance_alignment +
        0.18 * general_s +
        0.12 * comp +
        0.10 * div -
        0.20 * overuse_penalty
    )

    cbit_future = max(0.0, min(1.0, cbit_future))
    tq_future = max(0.0, min(1.0, tq_future))
    aq_future = max(0.0, min(1.0, aq_future))
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
        "future_compression_quality": comp,
        "future_lexical_diversity": div,
        "object_use_score": object_use_score,
        "use_binary": int(corrected_use >= 0.45),
        "relevance_alignment": relevance_alignment,
        "corrected_use_score": corrected_use,
        "useful_use_Cbit": useful_use_cbit,
        "correct_nonuse_Cbit": correct_nonuse_score,
        "overuse_penalty": overuse_penalty,
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

def make_analysis_table(current_features, preuse_features, future_scored):
    cur_num = [c for c in current_features.columns if pd.api.types.is_numeric_dtype(current_features[c])]
    cur_keep = [c for c in cur_num if c not in ["current_row_id", "state_rank", "dose", "is_true_visibility", "is_sham", "low_state"]]
    cur = current_features.groupby(
        ["model_key", "object_key", "object_name", "state_id", "state_rank", "state_name",
         "condition", "dose", "is_true_visibility", "is_sham", "low_state"],
        as_index=False
    )[cur_keep].mean()

    fut = future_scored.copy()
    merged = fut.merge(
        cur,
        on=["model_key", "object_key", "object_name", "state_id", "state_rank", "state_name",
            "condition", "dose", "is_true_visibility", "is_sham", "low_state"],
        how="inner",
    )

    pre = preuse_features.drop(columns=["prompt"], errors="ignore")
    merged = merged.merge(
        pre,
        on=["model_key", "model_name", "object_key", "object_name", "state_id", "state_rank", "state_name",
            "condition", "dose", "is_true_visibility", "is_sham", "low_state",
            "relevance", "relevance_weight", "future_task_id", "future_task"],
        how="inner",
        suffixes=("", "_preuse_dup"),
    )
    return merged

def feature_groups(df):
    current_cols = cols_by_prefix(df, ["current_"])
    preuse_cols = cols_by_prefix(df, ["preuse_"])
    use_pred_cols = unique_keep_order(current_cols + preuse_cols + [
        "state_rank", "relevance_weight", "dose", "is_true_visibility", "is_sham", "low_state"
    ])
    return {
        "UsePred": use_pred_cols,
    }

def add_use_pre_predictions(analysis_df):
    df = analysis_df.copy()
    df["Meaning_VR"] = df["current_V_use"] * df["relevance_weight"]
    df["dose_x_low"] = df["dose"] * df["low_state"]
    df["dose_x_rel"] = df["dose"] * df["relevance_weight"]
    df["dose_x_low_x_rel"] = df["dose"] * df["low_state"] * df["relevance_weight"]

    groups = feature_groups(df)
    res = cv_binary_probs(df, groups["UsePred"], "use_binary", group_col="object_key")
    if res is None:
        df["Use_pre_hat"] = np.nan
    else:
        idx, prob = res
        df["Use_pre_hat"] = np.nan
        df.loc[idx, "Use_pre_hat"] = prob

    df["Meaning_pre"] = df["current_V_use"] * df["relevance_weight"] * df["Use_pre_hat"]
    df["Meaning_oracle"] = df["current_V_use"] * df["relevance_weight"] * df["corrected_use_score"]
    return df

def paired_deltas(analysis_df):
    keys = ["model_key", "object_key", "state_id", "future_task_id", "relevance"]
    ctrl = analysis_df[analysis_df["condition"] == "control"].copy()
    base_cols = [
        "Cbit_future_proxy", "TQ_future_proxy", "AQ_future_proxy",
        "useful_use_Cbit", "correct_nonuse_Cbit", "overuse_penalty",
        "corrected_use_score", "object_use_score", "relevance_alignment",
        "future_success", "current_V_use", "current_V_core", "Use_pre_hat",
        "Meaning_pre", "Meaning_oracle", "state_rank", "low_state", "relevance_weight"
    ]
    ctrl_small = ctrl[keys + base_cols].rename(columns={c: f"{c}_control" for c in base_cols})
    rows = []
    for condition, dose, arm_type in CONDITIONS:
        if condition == "control":
            continue
        sub = analysis_df[analysis_df["condition"] == condition].copy()
        sub_small = sub[keys + base_cols].rename(columns={c: f"{c}_intervention" for c in base_cols})
        pair = ctrl_small.merge(sub_small, on=keys, how="inner")
        pair["condition"] = condition
        pair["dose"] = dose
        pair["arm_type"] = arm_type
        pair["is_true_visibility"] = int(arm_type == "true")
        pair["is_sham"] = int(arm_type == "sham")
        for metric in base_cols:
            if f"{metric}_intervention" in pair.columns and f"{metric}_control" in pair.columns:
                pair[f"delta_{metric}"] = pair[f"{metric}_intervention"] - pair[f"{metric}_control"]
        pair["low_state"] = pair["low_state_control"]
        pair["state_rank"] = pair["state_rank_control"]
        pair["relevance_weight"] = pair["relevance_weight_control"]
        rows.append(pair)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def summarize_pair(pair):
    groups = []
    groups.append(("all", pair))
    groups.append(("true_visibility", pair[pair["is_true_visibility"] == 1]))
    groups.append(("sham", pair[pair["is_sham"] == 1]))
    groups.append(("true_low_states", pair[(pair["is_true_visibility"] == 1) & (pair["low_state"] == 1)]))
    groups.append(("true_high_states", pair[(pair["is_true_visibility"] == 1) & (pair["low_state"] == 0)]))
    for rel in ["high", "medium", "low"]:
        groups.append((f"true_{rel}", pair[(pair["is_true_visibility"] == 1) & (pair["relevance"] == rel)]))
        groups.append((f"true_low_states_{rel}", pair[(pair["is_true_visibility"] == 1) & (pair["low_state"] == 1) & (pair["relevance"] == rel)]))
    for cond in ["visibility_low", "visibility_medium", "visibility_high"]:
        groups.append((cond, pair[pair["condition"] == cond]))
        groups.append((f"{cond}_low_states", pair[(pair["condition"] == cond) & (pair["low_state"] == 1)]))
        for rel in ["high", "medium", "low"]:
            groups.append((f"{cond}_low_states_{rel}", pair[(pair["condition"] == cond) & (pair["low_state"] == 1) & (pair["relevance"] == rel)]))

    metrics = [
        "delta_Cbit_future_proxy", "delta_TQ_future_proxy", "delta_AQ_future_proxy",
        "delta_useful_use_Cbit", "delta_correct_nonuse_Cbit", "delta_overuse_penalty",
        "delta_corrected_use_score", "delta_current_V_use", "delta_Use_pre_hat",
        "delta_Meaning_pre", "delta_Meaning_oracle", "delta_future_success",
    ]

    rows = []
    for name, sub in groups:
        row = {"group": name, "n": int(len(sub))}
        for m in metrics:
            if m in sub.columns and len(sub):
                vals = pd.to_numeric(sub[m], errors="coerce").dropna()
                row[f"{m}_mean"] = safe_float(vals.mean()) if len(vals) else np.nan
                row[f"{m}_median"] = safe_float(vals.median()) if len(vals) else np.nan
                row[f"{m}_positive_rate"] = safe_float((vals > 0).mean()) if len(vals) else np.nan
            else:
                row[f"{m}_mean"] = np.nan
                row[f"{m}_median"] = np.nan
                row[f"{m}_positive_rate"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def dose_response(pair):
    rows = []
    d = pair[(pair["is_true_visibility"] == 1) & (pair["low_state"] == 1)].copy()
    metrics = [
        "delta_Use_pre_hat", "delta_Meaning_pre", "delta_Cbit_future_proxy",
        "delta_useful_use_Cbit", "delta_correct_nonuse_Cbit", "delta_overuse_penalty"
    ]
    for rel in ["all", "high", "medium", "low"]:
        sub = d if rel == "all" else d[d["relevance"] == rel]
        for m in metrics:
            vals = sub[["dose", m]].dropna()
            corr = np.nan
            if len(vals) >= 5 and vals["dose"].std() > 1e-12 and vals[m].std() > 1e-12:
                corr = float(np.corrcoef(vals["dose"], vals[m])[0, 1])
            rows.append({
                "scope": rel,
                "metric": m,
                "n": int(len(vals)),
                "dose_corr": corr,
                "mean": safe_float(vals[m].mean()) if len(vals) else np.nan,
            })
    return pd.DataFrame(rows)

def verdict_from_summary(summary, dose_df):
    def g(group, metric):
        row = summary[summary["group"] == group]
        if len(row) == 0:
            return np.nan
        col = f"{metric}_mean"
        return safe_float(row[col].iloc[0]) if col in row.columns else np.nan

    def dc(scope, metric):
        row = dose_df[(dose_df["scope"] == scope) & (dose_df["metric"] == metric)]
        if len(row) == 0:
            return np.nan
        return safe_float(row["dose_corr"].iloc[0])

    # Main effect: true visibility vs control.
    true_low_usepre = g("true_low_states", "delta_Use_pre_hat")
    true_low_meaning = g("true_low_states", "delta_Meaning_pre")
    true_low_cbit = g("true_low_states", "delta_Cbit_future_proxy")

    high_useful = g("true_low_states_high", "delta_useful_use_Cbit")
    med_useful = g("true_low_states_medium", "delta_useful_use_Cbit")
    low_overuse = g("true_low_states_low", "delta_overuse_penalty")
    low_nonuse = g("true_low_states_low", "delta_correct_nonuse_Cbit")

    sham_low_cbit = g("sham", "delta_Cbit_future_proxy")
    true_low_high_cbit = g("true_low_states_high", "delta_Cbit_future_proxy")
    true_low_med_cbit = g("true_low_states_medium", "delta_Cbit_future_proxy")
    true_low_low_cbit = g("true_low_states_low", "delta_Cbit_future_proxy")

    dose_usepre_corr = dc("all", "delta_Use_pre_hat")
    dose_meaning_corr = dc("all", "delta_Meaning_pre")
    dose_high_useful_corr = dc("high", "delta_useful_use_Cbit")
    dose_low_overuse_corr = dc("low", "delta_overuse_penalty")

    dose_response_support = bool(
        (np.isfinite(dose_usepre_corr) and dose_usepre_corr > 0.25) or
        (np.isfinite(dose_meaning_corr) and dose_meaning_corr > 0.25)
    )
    relevance_cbit_support = bool(
        (np.isfinite(true_low_high_cbit) and true_low_high_cbit > 0.02) or
        (np.isfinite(true_low_med_cbit) and true_low_med_cbit > 0.02)
    )
    useful_use_support = bool(
        (np.isfinite(high_useful) and high_useful > 0.02) or
        (np.isfinite(med_useful) and med_useful > 0.02)
    )
    meaning_support = bool(np.isfinite(true_low_meaning) and true_low_meaning > 0.01)
    usepre_support = bool(np.isfinite(true_low_usepre) and true_low_usepre > 0.01)
    safety_support = bool(
        (not np.isfinite(low_overuse)) or
        (low_overuse < 0.03)
    )
    sham_control_support = bool(
        np.isfinite(true_low_cbit) and np.isfinite(sham_low_cbit) and true_low_cbit > sham_low_cbit + 0.005
    ) if np.isfinite(true_low_cbit) and np.isfinite(sham_low_cbit) else False

    if (dose_response_support and meaning_support and useful_use_support and relevance_cbit_support and safety_support):
        verdict = "PASS_STRONG_DOSE_CONTROLLED_VISIBILITY_CAUSAL"
    elif (meaning_support and usepre_support and useful_use_support and safety_support):
        verdict = "PASS_DOSE_CONTROLLED_VISIBILITY_CAUSAL"
    elif (meaning_support or usepre_support or relevance_cbit_support or dose_response_support):
        verdict = "PARTIAL_DOSE_CONTROLLED_VISIBILITY_CAUSAL"
    else:
        verdict = "FAIL_DOSE_CONTROLLED_VISIBILITY_CAUSAL"

    return {
        "verdict": verdict,
        "true_low_delta_usepre": true_low_usepre,
        "true_low_delta_meaning_pre": true_low_meaning,
        "true_low_delta_cbit": true_low_cbit,
        "true_low_high_delta_cbit": true_low_high_cbit,
        "true_low_medium_delta_cbit": true_low_med_cbit,
        "true_low_low_delta_cbit": true_low_low_cbit,
        "true_low_high_delta_useful_use_Cbit": high_useful,
        "true_low_medium_delta_useful_use_Cbit": med_useful,
        "true_low_low_delta_correct_nonuse_Cbit": low_nonuse,
        "true_low_low_delta_overuse_penalty": low_overuse,
        "sham_delta_cbit": sham_low_cbit,
        "dose_corr_usepre": dose_usepre_corr,
        "dose_corr_meaning_pre": dose_meaning_corr,
        "dose_corr_high_useful_use_Cbit": dose_high_useful_corr,
        "dose_corr_low_overuse_penalty": dose_low_overuse_corr,
        "dose_response_support": dose_response_support,
        "meaning_support": meaning_support,
        "usepre_support": usepre_support,
        "relevance_cbit_support": relevance_cbit_support,
        "useful_use_support": useful_use_support,
        "safety_support": safety_support,
        "sham_control_support": sham_control_support,
    }

# ============================================================
# Runner
# ============================================================

def run_one_model(model_key, current_dataset, preuse_dataset, future_dataset):
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
            if i % 50 == 0:
                print(f"[{model_key}] current row {i}/{len(current_dataset)}", flush=True)
            try:
                feat = extract_features(row, model, tokenizer, W, cat_ids, model_key, "current")
                current_rows.append({**row.to_dict(), "model_key": model_key, "model_name": spec["model_name"], **feat})
            except Exception as e:
                current_errors.append({
                    "current_row_id": int(row["current_row_id"]),
                    "object_key": row["object_key"],
                    "state_id": row["state_id"],
                    "condition": row["condition"],
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })

        preuse_rows, preuse_errors = [], []
        for i, row in preuse_dataset.iterrows():
            if i % 100 == 0:
                print(f"[{model_key}] preuse row {i}/{len(preuse_dataset)}", flush=True)
            try:
                feat = extract_features(row, model, tokenizer, W, cat_ids, model_key, "preuse")
                preuse_rows.append({**row.to_dict(), "model_key": model_key, "model_name": spec["model_name"], **feat})
            except Exception as e:
                preuse_errors.append({
                    "preuse_row_id": int(row["preuse_row_id"]),
                    "object_key": row["object_key"],
                    "state_id": row["state_id"],
                    "condition": row["condition"],
                    "future_task_id": row["future_task_id"],
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })

        future_rows, future_errors = [], []
        for i, row in future_dataset.iterrows():
            if i % 100 == 0:
                print(f"[{model_key}] future row {i}/{len(future_dataset)}", flush=True)
            try:
                ans = generate_answer(model, tokenizer, row["prompt"])
                future_rows.append({**row.to_dict(), "model_key": model_key, "model_name": spec["model_name"], "answer_text": ans})
            except Exception as e:
                future_errors.append({
                    "future_row_id": int(row["future_row_id"]),
                    "object_key": row["object_key"],
                    "state_id": row["state_id"],
                    "condition": row["condition"],
                    "future_task_id": row["future_task_id"],
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })

        current_features = pd.DataFrame(current_rows)
        preuse_features = pd.DataFrame(preuse_rows)
        future_answers = pd.DataFrame(future_rows)
        future_scored = score_future_df(future_answers) if not future_answers.empty else pd.DataFrame()

        current_features.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_current_internal_features.csv", index=False, encoding="utf-8-sig")
        preuse_features.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_preuse_internal_features.csv", index=False, encoding="utf-8-sig")
        future_answers.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_future_answers.csv", index=False, encoding="utf-8-sig")
        future_scored.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_future_scored_answers.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(current_errors).to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_current_errors.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(preuse_errors).to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_preuse_errors.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(future_errors).to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_future_errors.csv", index=False, encoding="utf-8-sig")

        if current_features.empty or preuse_features.empty or future_scored.empty:
            info.update({
                "status": "empty_outputs",
                "n_current_rows": int(len(current_features)),
                "n_preuse_rows": int(len(preuse_features)),
                "n_future_rows": int(len(future_scored)),
                "n_errors": int(len(current_errors) + len(preuse_errors) + len(future_errors)),
            })
            return info

        analysis = make_analysis_table(current_features, preuse_features, future_scored)
        analysis = add_use_pre_predictions(analysis)
        pair = paired_deltas(analysis)
        summary = summarize_pair(pair)
        dose = dose_response(pair)
        verdict = verdict_from_summary(summary, dose)

        analysis.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_analysis_table.csv", index=False, encoding="utf-8-sig")
        pair.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_paired_deltas.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_delta_summary.csv", index=False, encoding="utf-8-sig")
        dose.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_dose_response_summary.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([{**{"model_key": model_key}, **verdict}]).to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_verdict_summary.csv", index=False, encoding="utf-8-sig")

        info.update({
            "status": "done",
            "finished_at": now_time(),
            "runtime_sec": time.time() - t0,
            "n_current_rows": int(len(current_features)),
            "n_preuse_rows": int(len(preuse_features)),
            "n_future_rows": int(len(future_scored)),
            "n_pairs": int(len(pair)),
            "n_errors": int(len(current_errors) + len(preuse_errors) + len(future_errors)),
            "verdict": verdict.get("verdict", "UNDETERMINED"),
            "metrics": verdict,
            "outputs": {
                "current_internal_features": str(model_out / f"{EXPERIMENT_ID}_{model_key}_current_internal_features.csv"),
                "preuse_internal_features": str(model_out / f"{EXPERIMENT_ID}_{model_key}_preuse_internal_features.csv"),
                "future_answers": str(model_out / f"{EXPERIMENT_ID}_{model_key}_future_answers.csv"),
                "future_scored_answers": str(model_out / f"{EXPERIMENT_ID}_{model_key}_future_scored_answers.csv"),
                "analysis_table": str(model_out / f"{EXPERIMENT_ID}_{model_key}_analysis_table.csv"),
                "paired_deltas": str(model_out / f"{EXPERIMENT_ID}_{model_key}_paired_deltas.csv"),
                "delta_summary": str(model_out / f"{EXPERIMENT_ID}_{model_key}_delta_summary.csv"),
                "dose_response_summary": str(model_out / f"{EXPERIMENT_ID}_{model_key}_dose_response_summary.csv"),
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
            "true_low_delta_usepre": m.get("true_low_delta_usepre"),
            "true_low_delta_meaning_pre": m.get("true_low_delta_meaning_pre"),
            "true_low_delta_cbit": m.get("true_low_delta_cbit"),
            "true_low_high_delta_cbit": m.get("true_low_high_delta_cbit"),
            "true_low_medium_delta_cbit": m.get("true_low_medium_delta_cbit"),
            "true_low_low_delta_cbit": m.get("true_low_low_delta_cbit"),
            "true_low_high_delta_useful_use_Cbit": m.get("true_low_high_delta_useful_use_Cbit"),
            "true_low_medium_delta_useful_use_Cbit": m.get("true_low_medium_delta_useful_use_Cbit"),
            "true_low_low_delta_correct_nonuse_Cbit": m.get("true_low_low_delta_correct_nonuse_Cbit"),
            "true_low_low_delta_overuse_penalty": m.get("true_low_low_delta_overuse_penalty"),
            "sham_delta_cbit": m.get("sham_delta_cbit"),
            "dose_corr_usepre": m.get("dose_corr_usepre"),
            "dose_corr_meaning_pre": m.get("dose_corr_meaning_pre"),
            "dose_corr_high_useful_use_Cbit": m.get("dose_corr_high_useful_use_Cbit"),
            "dose_corr_low_overuse_penalty": m.get("dose_corr_low_overuse_penalty"),
            "dose_response_support": m.get("dose_response_support"),
            "meaning_support": m.get("meaning_support"),
            "usepre_support": m.get("usepre_support"),
            "relevance_cbit_support": m.get("relevance_cbit_support"),
            "useful_use_support": m.get("useful_use_support"),
            "safety_support": m.get("safety_support"),
            "sham_control_support": m.get("sham_control_support"),
            "n_current_rows": info.get("n_current_rows"),
            "n_preuse_rows": info.get("n_preuse_rows"),
            "n_future_rows": info.get("n_future_rows"),
            "n_pairs": info.get("n_pairs"),
            "n_errors": info.get("n_errors"),
            "error": info.get("error"),
        })
    return pd.DataFrame(rows)

def global_verdict(cross):
    done = cross[cross["status"] == "done"].copy()
    if len(done) == 0:
        return "FAIL_SOBV5B_NO_MODELS"
    verdicts = done["verdict"].astype(str).tolist()
    pass_strong = sum(v.startswith("PASS_STRONG") for v in verdicts)
    pass_any = sum(v.startswith("PASS") for v in verdicts)
    partial = sum(v.startswith("PARTIAL") for v in verdicts)
    if pass_strong == len(done):
        return "PASS_STRONG_SOBV5B_CROSS_MODEL"
    if pass_any == len(done):
        return "PASS_SOBV5B_CROSS_MODEL"
    if pass_any + partial == len(done):
        return "PARTIAL_SOBV5B_CROSS_MODEL"
    if pass_any >= 2:
        return "MIXED_POSITIVE_SOBV5B"
    if pass_any >= 1:
        return "MIXED_SOBV5B"
    return "FAIL_SOBV5B"

def main():
    set_seed()
    print("=" * 80)
    print(f"[{EXPERIMENT_ID}] Dose-Controlled Relevance-Conditioned Visibility Intervention")
    print(f"Device: {DEVICE}, dtype: {DTYPE}")
    print(f"Output: {OUT_DIR}")
    print("=" * 80)

    current_dataset, preuse_dataset, future_dataset = build_datasets()

    current_dataset_path = OUT_DIR / f"{EXPERIMENT_ID}_current_dataset.csv"
    preuse_dataset_path = OUT_DIR / f"{EXPERIMENT_ID}_preuse_dataset.csv"
    future_dataset_path = OUT_DIR / f"{EXPERIMENT_ID}_future_dataset.csv"

    current_dataset.to_csv(current_dataset_path, index=False, encoding="utf-8-sig")
    preuse_dataset.to_csv(preuse_dataset_path, index=False, encoding="utf-8-sig")
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
        "conditions": CONDITIONS,
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
        info = run_one_model(model_key, current_dataset, preuse_dataset, future_dataset)
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
            "preuse_dataset": str(preuse_dataset_path),
            "future_dataset": str(future_dataset_path),
            "cross_model_summary": str(cross_path),
            "out_dir": str(OUT_DIR),
        },
        "interpretation": {
            "dose": "0 control/sham, 0.33 low, 0.66 medium, 1.0 high.",
            "sham": "generic study note without object-specific constraint visibility.",
            "useful_use_Cbit": "Future Cbit from relevant object use.",
            "correct_nonuse_Cbit": "Future Cbit from correctly refusing irrelevant use.",
            "overuse_penalty": "Penalty for forcing the object on low-relevance tasks.",
            "PASS_STRONG_DOSE_CONTROLLED_VISIBILITY_CAUSAL": "Dose response, MeaningPre, useful-use Cbit, relevance-conditioned support, and safety all hold.",
            "PASS_DOSE_CONTROLLED_VISIBILITY_CAUSAL": "Meaning/usepre and useful use hold with safety.",
            "PARTIAL_DOSE_CONTROLLED_VISIBILITY_CAUSAL": "Some causal support exists but full criterion is not met.",
        },
    }

    with open(OUT_DIR / f"{EXPERIMENT_ID}_verdict.json", "w", encoding="utf-8") as f:
        json.dump(global_obj, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(json.dumps(global_obj, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
