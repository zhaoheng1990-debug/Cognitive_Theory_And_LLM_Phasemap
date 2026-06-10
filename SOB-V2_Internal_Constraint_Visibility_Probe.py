# -*- coding: utf-8 -*-
r"""
SOB-V2: Internal Constraint Visibility Probe

Purpose
-------
SOB-V1 established text-level evidence that Constraint Visibility V(S,O)
is measurable under subject-state control.

SOB-V2 moves from generated text to internal state observables.

Core question
-------------
For the same object O = Derivative and different subject states S_k,
does the model internally form different visible constraint structures
\hat P_S = (\hat\Omega_S, \hat F_S)?

We test this using internal probes:
  1. Category-logit visibility over derivative constraint families.
  2. TopK/VIM neighborhood statistics.
  3. State-rank gradients of internal core and transfer visibility.
  4. Internal subject-state separability.

Object
------
O = Derivative

Subject states
--------------
S0 = symbol-only
S1 = definition-known
S2 = calculation ability
S3 = geometric understanding
S4 = cross-domain transfer

Main hypotheses
---------------
H1 Core internal visibility:
    V_core_internal(S0) < S1 < S2 < S3

H2 Transfer internal visibility:
    V_transfer_internal(S4) > V_transfer_internal(S3)

H3 Internal state separability:
    Hidden / category / TopK features can classify subject state above chance.

Output
------
C:\Users\ZH\Desktop\AGI\outputs\SOB-V2

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
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.multiclass import OneVsRestClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ============================================================
# Constants
# ============================================================

EXPERIMENT_ID = "SOB-V2"
OUT_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\SOB-V2")
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
        "task_id": "TRANSFER_physics",
        "task_family": "transfer",
        "question": "Think about whether this learner state can transfer derivative to position, velocity, and acceleration.",
    },
    {
        "task_id": "TRANSFER_optimization_ml",
        "task_family": "transfer",
        "question": "Think about whether this learner state can transfer derivative to gradients, optimization, and learning.",
    },
    {
        "task_id": "TRANSFER_control_newdomain",
        "task_family": "transfer",
        "question": "Think about whether this learner state can transfer derivative to feedback, sensitivity, economics, or biology.",
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


def build_prompt(subject_state, task):
    return (
        "Internal-state probe only. Do not produce an answer.\n\n"
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


def make_multiclass_classifier():
    base = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=RANDOM_SEED)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", OneVsRestClassifier(base)),
    ])


def make_regressor():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=1.0)),
    ])


def eval_multiclass(df, cols, target="state_id", group_col="task_id"):
    cols = numeric_cols(df, cols)
    if target not in df.columns or not cols:
        return {"valid": False, "reason": "missing"}
    d = df.dropna(subset=[target]).copy()
    if len(d) < 15:
        return {"valid": False, "reason": "too_few", "n": len(d)}
    y = d[target].astype(str).values
    if len(np.unique(y)) < 2:
        return {"valid": False, "reason": "single_class", "n": len(d)}
    X = d[cols].replace([np.inf, -np.inf], np.nan).values

    pred = np.array([""] * len(d), dtype=object)
    used = 0
    for train_idx, test_idx in cv_indices(d, y, group_col=group_col, stratify=True):
        if len(np.unique(y[train_idx])) < 2:
            continue
        clf = make_multiclass_classifier()
        clf.fit(X[train_idx], y[train_idx])
        pred[test_idx] = clf.predict(X[test_idx])
        used += 1
    if used == 0:
        return {"valid": False, "reason": "no_folds", "n": len(d)}

    return {
        "valid": True,
        "n": int(len(d)),
        "n_features": int(len(cols)),
        "n_classes": int(len(np.unique(y))),
        "acc": safe_float(accuracy_score(y, pred)),
        "macro_f1": safe_float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_acc": safe_float(balanced_accuracy_score(y, pred)),
    }


def eval_regression(df, cols, target="state_rank", group_col="task_id"):
    cols = numeric_cols(df, cols)
    if target not in df.columns or not cols:
        return {"valid": False, "reason": "missing"}
    d = df.dropna(subset=[target]).copy()
    if len(d) < 15:
        return {"valid": False, "reason": "too_few", "n": len(d)}
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
        return {"valid": False, "reason": "no_folds", "n": len(d)}
    corr = np.nan
    if np.std(pred) > 1e-12 and np.std(y) > 1e-12:
        corr = float(np.corrcoef(y, pred)[0, 1])
    return {
        "valid": True,
        "n": int(len(d)),
        "n_features": int(len(cols)),
        "r2": safe_float(r2_score(y, pred)),
        "corr": corr,
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


def extract_row_features(row, model, tokenizer, W, cat_ids, model_key):
    hs = forward_hidden(model, tokenizer, row["prompt"])
    windows = windows_for_model(model_key, len(hs))
    feat = {
        "n_hidden_states": len(hs),
        "n_layers": len(hs) - 1,
    }

    # Windowed category-logit visibility.
    for wname, hidxs in windows.items():
        cat_seq = {cat: [] for cat in ALL_CATEGORIES}
        topk_centers = []
        topk_entropy_vals = []
        topk_spread_vals = []
        topk_overlap_symbol = []
        topk_overlap_core = []
        topk_overlap_transfer = []

        symbol_ids = set(cat_ids.get("symbol", []))
        core_ids = set()
        transfer_ids = set()
        for c in CORE_CATEGORIES:
            core_ids |= set(cat_ids.get(c, []))
        for c in TRANSFER_CATEGORIES:
            transfer_ids |= set(cat_ids.get(c, []))

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

            idset = set(ids.tolist())
            topk_overlap_symbol.append(len(idset & symbol_ids) / max(1, len(symbol_ids)))
            topk_overlap_core.append(len(idset & core_ids) / max(1, len(core_ids)))
            topk_overlap_transfer.append(len(idset & transfer_ids) / max(1, len(transfer_ids)))

        # Category softmax per layer, then average probability mass.
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
        feat[f"{wname}_topk_symbol_overlap_mean"] = float(np.mean(topk_overlap_symbol))
        feat[f"{wname}_topk_core_overlap_mean"] = float(np.mean(topk_overlap_core))
        feat[f"{wname}_topk_transfer_overlap_mean"] = float(np.mean(topk_overlap_transfer))

        if len(topk_centers) >= 2:
            drifts = [1.0 - cosine(topk_centers[i], topk_centers[i + 1]) for i in range(len(topk_centers) - 1)]
            feat[f"{wname}_topk_center_drift_mean"] = float(np.mean(drifts))
            feat[f"{wname}_topk_center_drift_max"] = float(np.max(drifts))
        else:
            feat[f"{wname}_topk_center_drift_mean"] = 0.0
            feat[f"{wname}_topk_center_drift_max"] = 0.0

    # Primary internal observables.
    # Commit window is treated as identity / visible structure readout.
    # Middle window is treated as trajectory-shape / operator-readout.
    feat["V_core_internal"] = float(np.nanmean([
        feat.get("commit_core_mass", np.nan),
        feat.get("middle_core_mass", np.nan),
    ]))
    feat["V_transfer_internal"] = float(np.nanmean([
        feat.get("commit_transfer_mass", np.nan),
        feat.get("middle_transfer_mass", np.nan),
    ]))
    feat["V_symbol_internal"] = float(np.nanmean([
        feat.get("init_symbol_mass", np.nan),
        feat.get("commit_symbol_mass", np.nan),
    ]))
    feat["V_all_internal"] = float(np.nanmean([
        feat["V_core_internal"],
        feat["V_transfer_internal"],
    ]))
    feat["InternalVisibilityRatio_core_over_symbol"] = feat["V_core_internal"] / (feat["V_symbol_internal"] + 1e-8)
    feat["InternalVisibilityRatio_transfer_over_core"] = feat["V_transfer_internal"] / (feat["V_core_internal"] + 1e-8)

    return feat


# ============================================================
# Analysis
# ============================================================

def aggregate_state_visibility(features_df):
    agg = (
        features_df
        .groupby(["model_key", "state_id", "state_rank", "state_name"], as_index=False)
        .agg(
            V_core_internal=("V_core_internal", "mean"),
            V_transfer_internal=("V_transfer_internal", "mean"),
            V_symbol_internal=("V_symbol_internal", "mean"),
            V_all_internal=("V_all_internal", "mean"),
            ratio_core_over_symbol=("InternalVisibilityRatio_core_over_symbol", "mean"),
            ratio_transfer_over_core=("InternalVisibilityRatio_transfer_over_core", "mean"),
            n=("V_all_internal", "size"),
        )
    )
    return agg.sort_values(["model_key", "state_rank"])


def pearson_rank(sub, col):
    vals = sub[col].values.astype(float)
    ranks = sub["state_rank"].values.astype(float)
    mask = np.isfinite(vals) & np.isfinite(ranks)
    vals = vals[mask]
    ranks = ranks[mask]
    if len(vals) < 2 or np.std(vals) < 1e-12:
        return np.nan
    return float(np.corrcoef(ranks, vals)[0, 1])


def spearman_rank(sub, col):
    vals = sub[col].values.astype(float)
    ranks = sub["state_rank"].values.astype(float)
    mask = np.isfinite(vals) & np.isfinite(ranks)
    vals = vals[mask]
    ranks = ranks[mask]
    if len(vals) < 2 or np.std(vals) < 1e-12:
        return np.nan
    order_vals = pd.Series(vals).rank().values
    return float(np.corrcoef(ranks, order_vals)[0, 1])


def get_state_val(sub, state_id, col):
    x = sub[sub["state_id"] == state_id]
    if len(x) == 0:
        return np.nan
    return safe_float(x[col].iloc[0])


def build_feature_groups(df):
    category_cols = cols_by_prefix(df, [
        "init_catprob_", "middle_catprob_", "competition_catprob_", "commit_catprob_", "final_catprob_",
        "init_core_", "middle_core_", "commit_core_", "final_core_",
        "init_transfer_", "middle_transfer_", "commit_transfer_", "final_transfer_",
        "init_symbol_", "middle_symbol_", "commit_symbol_", "final_symbol_",
        "InternalVisibilityRatio_", "V_core_internal", "V_transfer_internal", "V_symbol_internal", "V_all_internal",
    ])
    topk_cols = cols_by_prefix(df, [
        "init_topk_", "middle_topk_", "competition_topk_", "commit_topk_", "final_topk_"
    ])
    all_cols = unique_keep_order(category_cols + topk_cols)
    return {
        "CategoryInternal": category_cols,
        "TopKInternal": topk_cols,
        "AllInternal": all_cols,
    }


def run_probe_evals(features_df):
    groups = build_feature_groups(features_df)
    rows = []
    for gname, cols in groups.items():
        mc = eval_multiclass(features_df, cols, target="state_id", group_col="task_id")
        rows.append({
            "feature_group": gname,
            "task": "state_classification",
            "target": "state_id",
            "n_features": len(cols),
            "valid": mc.get("valid", False),
            "acc": mc.get("acc", np.nan),
            "macro_f1": mc.get("macro_f1", np.nan),
            "balanced_acc": mc.get("balanced_acc", np.nan),
            "r2": np.nan,
            "corr": np.nan,
            "n": mc.get("n", np.nan),
        })
        rg = eval_regression(features_df, cols, target="state_rank", group_col="task_id")
        rows.append({
            "feature_group": gname,
            "task": "state_rank_regression",
            "target": "state_rank",
            "n_features": len(cols),
            "valid": rg.get("valid", False),
            "acc": np.nan,
            "macro_f1": np.nan,
            "balanced_acc": np.nan,
            "r2": rg.get("r2", np.nan),
            "corr": rg.get("corr", np.nan),
            "n": rg.get("n", np.nan),
        })
    return pd.DataFrame(rows)


def verdict_from_state_summary(model_key, state_summary, probe_summary):
    sub = state_summary[state_summary["model_key"] == model_key].sort_values("state_rank")
    core_sub = sub[sub["state_rank"] <= 3].copy()

    core_vals = core_sub["V_core_internal"].values.astype(float)
    core_diffs = np.diff(core_vals)
    core_strict = bool(np.all(core_diffs > 0))
    core_nondecreasing = bool(np.all(core_diffs >= -1e-9))
    core_p = pearson_rank(core_sub, "V_core_internal")
    core_s = spearman_rank(core_sub, "V_core_internal")

    transfer_s3 = get_state_val(sub, "S3", "V_transfer_internal")
    transfer_s4 = get_state_val(sub, "S4", "V_transfer_internal")
    transfer_gain = transfer_s4 - transfer_s3 if np.isfinite(transfer_s4) and np.isfinite(transfer_s3) else np.nan
    transfer_pass = bool(np.isfinite(transfer_gain) and transfer_gain > 0)

    all_p = pearson_rank(sub, "V_all_internal")
    all_s = spearman_rank(sub, "V_all_internal")

    best_cls_f1 = np.nan
    best_rank_corr = np.nan
    ps = probe_summary.copy()
    if len(ps):
        cls = ps[ps["task"] == "state_classification"]
        reg = ps[ps["task"] == "state_rank_regression"]
        if len(cls):
            best_cls_f1 = safe_float(cls["macro_f1"].max())
        if len(reg):
            best_rank_corr = safe_float(reg["corr"].max())

    sep_pass = bool(np.isfinite(best_cls_f1) and best_cls_f1 > 0.35) or bool(np.isfinite(best_rank_corr) and best_rank_corr > 0.45)

    if core_strict and core_p >= 0.80 and transfer_pass and sep_pass:
        verdict = "PASS_STRONG_INTERNAL_VISIBILITY"
    elif core_p >= 0.60 and transfer_pass and sep_pass:
        verdict = "PASS_INTERNAL_VISIBILITY"
    elif (core_p >= 0.50 or transfer_pass) and sep_pass:
        verdict = "PARTIAL_INTERNAL_VISIBILITY"
    else:
        verdict = "FAIL_INTERNAL_VISIBILITY"

    return {
        "model_key": model_key,
        "verdict": verdict,
        "core_strict_monotonic_S0_S3": core_strict,
        "core_nondecreasing_S0_S3": core_nondecreasing,
        "pearson_state_core_internal_S0_S3": core_p,
        "spearman_state_core_internal_S0_S3": core_s,
        "transfer_S4_greater_S3": transfer_pass,
        "transfer_gain_S4_minus_S3": transfer_gain,
        "pearson_state_all_internal_S0_S4": all_p,
        "spearman_state_all_internal_S0_S4": all_s,
        "best_state_classification_macro_f1": best_cls_f1,
        "best_state_rank_regression_corr": best_rank_corr,
        "internal_state_separability_pass": sep_pass,
        "V_core_internal_S0": get_state_val(sub, "S0", "V_core_internal"),
        "V_core_internal_S1": get_state_val(sub, "S1", "V_core_internal"),
        "V_core_internal_S2": get_state_val(sub, "S2", "V_core_internal"),
        "V_core_internal_S3": get_state_val(sub, "S3", "V_core_internal"),
        "V_core_internal_S4": get_state_val(sub, "S4", "V_core_internal"),
        "V_transfer_internal_S3": transfer_s3,
        "V_transfer_internal_S4": transfer_s4,
        "V_symbol_internal_S0": get_state_val(sub, "S0", "V_symbol_internal"),
        "V_all_internal_S0": get_state_val(sub, "S0", "V_all_internal"),
        "V_all_internal_S1": get_state_val(sub, "S1", "V_all_internal"),
        "V_all_internal_S2": get_state_val(sub, "S2", "V_all_internal"),
        "V_all_internal_S3": get_state_val(sub, "S3", "V_all_internal"),
        "V_all_internal_S4": get_state_val(sub, "S4", "V_all_internal"),
        "core_diffs_S0_S3": ";".join([f"{x:.6f}" for x in core_diffs]),
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

        state_summary = aggregate_state_visibility(features_df)
        probe_summary = run_probe_evals(features_df)
        verdict = verdict_from_state_summary(model_key, state_summary, probe_summary)

        state_summary.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_state_visibility_summary.csv", index=False, encoding="utf-8-sig")
        probe_summary.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_probe_summary.csv", index=False, encoding="utf-8-sig")
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
                "state_visibility_summary": str(model_out / f"{EXPERIMENT_ID}_{model_key}_state_visibility_summary.csv"),
                "probe_summary": str(model_out / f"{EXPERIMENT_ID}_{model_key}_probe_summary.csv"),
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
            "core_strict_monotonic_S0_S3": m.get("core_strict_monotonic_S0_S3"),
            "pearson_state_core_internal_S0_S3": m.get("pearson_state_core_internal_S0_S3"),
            "spearman_state_core_internal_S0_S3": m.get("spearman_state_core_internal_S0_S3"),
            "transfer_S4_greater_S3": m.get("transfer_S4_greater_S3"),
            "transfer_gain_S4_minus_S3": m.get("transfer_gain_S4_minus_S3"),
            "pearson_state_all_internal_S0_S4": m.get("pearson_state_all_internal_S0_S4"),
            "spearman_state_all_internal_S0_S4": m.get("spearman_state_all_internal_S0_S4"),
            "best_state_classification_macro_f1": m.get("best_state_classification_macro_f1"),
            "best_state_rank_regression_corr": m.get("best_state_rank_regression_corr"),
            "internal_state_separability_pass": m.get("internal_state_separability_pass"),
            "V_core_internal_S0": m.get("V_core_internal_S0"),
            "V_core_internal_S1": m.get("V_core_internal_S1"),
            "V_core_internal_S2": m.get("V_core_internal_S2"),
            "V_core_internal_S3": m.get("V_core_internal_S3"),
            "V_core_internal_S4": m.get("V_core_internal_S4"),
            "V_transfer_internal_S3": m.get("V_transfer_internal_S3"),
            "V_transfer_internal_S4": m.get("V_transfer_internal_S4"),
            "V_symbol_internal_S0": m.get("V_symbol_internal_S0"),
            "V_all_internal_S0": m.get("V_all_internal_S0"),
            "V_all_internal_S1": m.get("V_all_internal_S1"),
            "V_all_internal_S2": m.get("V_all_internal_S2"),
            "V_all_internal_S3": m.get("V_all_internal_S3"),
            "V_all_internal_S4": m.get("V_all_internal_S4"),
            "core_diffs_S0_S3": m.get("core_diffs_S0_S3"),
            "n_rows": info.get("n_rows"),
            "n_errors": info.get("n_errors"),
            "error": info.get("error"),
        })
    return pd.DataFrame(rows)


def global_verdict(cross):
    done = cross[cross["status"] == "done"].copy()
    if len(done) == 0:
        return "FAIL_SOBV2_NO_MODELS"

    verdicts = done["verdict"].astype(str).tolist()
    pass_strong = sum(v.startswith("PASS_STRONG") for v in verdicts)
    pass_any = sum(v.startswith("PASS") for v in verdicts)
    partial = sum(v.startswith("PARTIAL") for v in verdicts)

    if pass_strong == len(done):
        return "PASS_STRONG_SOBV2_CROSS_MODEL"
    if pass_any == len(done):
        return "PASS_SOBV2_CROSS_MODEL"
    if pass_any + partial == len(done):
        return "PARTIAL_SOBV2_CROSS_MODEL"
    if pass_any >= 2:
        return "MIXED_POSITIVE_SOBV2"
    if pass_any >= 1:
        return "MIXED_SOBV2"
    return "FAIL_SOBV2"


def main():
    set_seed()
    print("=" * 80)
    print(f"[{EXPERIMENT_ID}] Internal Constraint Visibility Probe")
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
            "V_core_internal": "Internal category-mass visibility over derivative definition/calculation/geometry/edge constraints.",
            "V_transfer_internal": "Internal category-mass visibility over physics/optimization/ML/control/new-domain transfer constraints.",
            "state_separability": "Whether internal features classify subject state or predict state rank above chance.",
            "PASS_STRONG_INTERNAL_VISIBILITY": "Core internal visibility is monotonic S0-S3, transfer S4>S3, and internal state separability passes.",
            "PASS_INTERNAL_VISIBILITY": "Core/transfer internal visibility is supported with internal separability.",
            "PARTIAL_INTERNAL_VISIBILITY": "Some internal visibility structure exists but not all criteria pass.",
            "FAIL_INTERNAL_VISIBILITY": "No stable internal visibility structure detected.",
        },
    }

    with open(OUT_DIR / f"{EXPERIMENT_ID}_verdict.json", "w", encoding="utf-8") as f:
        json.dump(global_obj, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(json.dumps(global_obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
