# -*- coding: utf-8 -*-
r"""
PSA-2B: Unified Full-Coverage ConstraintField -> Effective Cbit Audit

Purpose
-------
CFT-4 / PSA-2 was positive, but had a caveat:
  - CFT features and PSA features came from different prior runs.
  - row-level join coverage was 44.4%.

PSA-2B removes this caveat by extracting both feature families from the same
forward pass over the same dataset:

  1. CFT-style no-answer constraint-field proxy:
       TopK_aligned + Dynamics_aligned

  2. PSA-style possibility-space / Cbit features:
       H_answer(l), N_eff(l), ΔCbit(init->basin/final)

  3. Effective Cbit:
       Cbit_eff = ΔCbit * Q_c

where Q_c is a lightweight structural-quality proxy built from:
  - correctness
  - mechanism type
  - random-broken penalty

Default model
-------------
Qwen only, for continuity with PSA-1 and CFT-4/PSA-2:

  D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main

Optional:
  Set MODELS_TO_RUN = ["qwen", "llama", "gemma"] for cross-model.

Outputs
-------
C:\Users\ZH\Desktop\AGI\outputs\psa2b_outputs\
  psa2b_<model>_features.csv
  psa2b_<model>_model_summary.csv
  psa2b_<model>_group_summary.csv
  psa2b_<model>_verdict.json
  psa2b_cross_model_summary.csv
  psa2b_verdict.json

Verdict logic
-------------
PASS_PSA2B_CONSTRAINT_TO_EFFECTIVE_CBIT:
  CFT_NoAnswer predicts Cbit_eff_basin and correct samples have higher Cbit_eff.

PARTIAL_PSA2B_RAW_CBIT_ONLY:
  CFT_NoAnswer predicts raw Cbit, but effective Cbit separation is weak.

FAIL_PSA2B:
  No stable link between constraint proxy and Cbit measures.
"""

import os
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
    roc_auc_score,
    balanced_accuracy_score,
    r2_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.multiclass import OneVsRestClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# Config
# -----------------------------

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

MODELS_TO_RUN = ["qwen"]

OUT_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\psa2b_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# 24*3*9 = 648 rows; same scale as CFT-3B.
N_GRAPHS = 24
N_SURFACES = 3
TOPK = 80
MAX_LEN = 512
N_SPLITS = 4

CANDIDATES = ["C", "E", "OTHER"]

# Use CFT-3B style dataset for direct comparability.
ENTITIES_A = [
    "mira", "nolan", "sora", "tavin", "lena", "orion",
    "vexa", "darin", "selka", "brin", "kael", "nira",
    "faro", "yuna", "pavo", "lumi", "taro", "zena",
    "bex", "rilo", "sena", "mako", "viro", "doma",
]

GROUPS_B = [
    "amber guild", "blue circle", "crimson order", "delta clan", "ember house", "frost league",
    "green lodge", "harbor unit", "iron band", "jade group", "kepler sect", "lunar team",
    "marble council", "north ring", "opal class", "prime cohort", "quartz line", "river squad",
    "silver branch", "terra block", "umber cell", "violet union",
]

GROUPS_C = [
    "atlas zone", "boreal zone", "cobalt zone", "dawn zone", "equinox zone", "falcon zone",
    "garden zone", "helios zone", "ivory zone", "juniper zone", "kestrel zone", "lotus zone",
    "meteor zone", "nova zone", "onyx zone", "prairie zone", "quasar zone", "raven zone",
    "summit zone", "tundra zone", "umbra zone", "velvet zone",
]

GROUPS_E = [
    "arcadia zone", "bright zone", "cinder zone", "dusk zone", "elm zone", "fern zone",
    "granite zone", "hazel zone", "indigo zone", "jasmine zone", "kite zone", "lagoon zone",
    "meadow zone", "nebula zone", "oasis zone", "polar zone", "quiet zone", "reed zone",
    "sunset zone", "thistle zone", "ultra zone", "valley zone",
]

SURFACES = [
    "Use the facts below and answer with only C, E, or OTHER.\nFacts:\n{facts}\nQuestion: Which final group does {A} belong to?\nAnswer:",
    "Read the mini knowledge base. Reply only with C, E, or OTHER.\n{facts}\nQuery: final membership of {A}?\nAnswer:",
    "Given these relations, decide the final target for {A}. Output exactly C, E, or OTHER.\n{facts}\nFinal answer:",
]


# -----------------------------
# Dataset
# -----------------------------

def build_dataset():
    rows = []
    conditions = [
        "stable_clean",
        "stable_redundant",
        "competition_equal",
        "source_trusted_C",
        "source_trusted_E",
        "closure_update_C",
        "closure_update_E",
        "logic_contradiction",
        "random_broken",
    ]

    for gi in range(N_GRAPHS):
        A = ENTITIES_A[gi % len(ENTITIES_A)]
        B = GROUPS_B[gi % len(GROUPS_B)]
        C = GROUPS_C[gi % len(GROUPS_C)]
        E = GROUPS_E[gi % len(GROUPS_E)]
        graph_id = f"g{gi:03d}"

        for si in range(N_SURFACES):
            template = SURFACES[si % len(SURFACES)]
            surface_id = f"surface_{si}"

            for cond in conditions:
                if cond == "stable_clean":
                    facts = f"{A} belongs to {B}.\n{B} belongs to C.\nC means {C}. E means {E}."
                    mechanism, submechanism, gold = "stable", "stable_core", "C"

                elif cond == "stable_redundant":
                    facts = f"{A} belongs to {B}.\n{B} belongs to C.\nC means {C}. E means {E}.\nNote: C is the final group in this example."
                    mechanism, submechanism, gold = "stable", "stable_redundant", "C"

                elif cond == "competition_equal":
                    facts = f"{A} belongs to {B}.\n{B} may belong to C.\n{B} may belong to E.\nC means {C}. E means {E}. Both options are equally supported."
                    mechanism, submechanism, gold = "competition", "competition_equal", "OTHER"

                elif cond == "source_trusted_C":
                    facts = f"Source 1 says: {B} belongs to E.\nTrusted source says: {B} belongs to C.\n{A} belongs to {B}.\nC means {C}. E means {E}. Use the trusted source."
                    mechanism, submechanism, gold = "competition", "competition_source_C", "C"

                elif cond == "source_trusted_E":
                    facts = f"Source 1 says: {B} belongs to C.\nTrusted source says: {B} belongs to E.\n{A} belongs to {B}.\nC means {C}. E means {E}. Use the trusted source."
                    mechanism, submechanism, gold = "competition", "competition_source_E", "E"

                elif cond == "closure_update_C":
                    facts = f"Old rule: {B} belongs to E.\nNew rule: {B} belongs to C.\n{A} belongs to {B}.\nC means {C}. E means {E}. Use the new rule."
                    mechanism, submechanism, gold = "closure", "closure_update_C", "C"

                elif cond == "closure_update_E":
                    facts = f"Old rule: {B} belongs to C.\nNew rule: {B} belongs to E.\n{A} belongs to {B}.\nC means {C}. E means {E}. Use the new rule."
                    mechanism, submechanism, gold = "closure", "closure_update_E", "E"

                elif cond == "logic_contradiction":
                    facts = f"Rule: {A} cannot belong to both C and E.\nClaim 1: {A} belongs to C.\nClaim 2: {A} belongs to E.\nC means {C}. E means {E}. If the facts contradict, answer OTHER."
                    mechanism, submechanism, gold = "logic", "logic_direct", "OTHER"

                elif cond == "random_broken":
                    facts = f"{A} belongs to {B}.\nC means {C}. E means {E}.\nNo final membership relation from {B} to C or E is provided."
                    mechanism, submechanism, gold = "random", "random_broken", "OTHER"

                prompt = template.format(facts=facts, A=A)
                rows.append({
                    "row_id": len(rows),
                    "graph_id": graph_id,
                    "surface_id": surface_id,
                    "condition": cond,
                    "mechanism": mechanism,
                    "submechanism": submechanism,
                    "gold": gold,
                    "A": A,
                    "B": B,
                    "C_text": C,
                    "E_text": E,
                    "prompt": prompt,
                })

    return pd.DataFrame(rows)


# -----------------------------
# Basic math / ML utilities
# -----------------------------

def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def softmax_np(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    p = np.exp(x)
    return p / (np.sum(p) + 1e-12)


def entropy_bits_from_scores(scores):
    p = softmax_np(scores)
    h = -float(np.sum(p * np.log2(p + 1e-12)))
    return h, p


def n_eff(h):
    return float(2.0 ** h) if np.isfinite(h) else np.nan


def area(xs):
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    if arr.size == 1:
        return float(arr[0])
    return float(np.sum((arr[:-1] + arr[1:]) * 0.5))


def slope(xs):
    arr = np.asarray(xs, dtype=np.float64)
    if arr.size < 2 or np.any(~np.isfinite(arr)):
        return np.nan
    x = np.arange(arr.size, dtype=np.float64)
    try:
        return float(np.polyfit(x, arr, 1)[0])
    except Exception:
        return np.nan


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / den)


def numeric_cols(df, cols):
    return [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


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


def cv_indices(df, y=None, group_col="graph_id", stratify=False):
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


def eval_binary(df, cols, target="gen_error", group_col="graph_id"):
    cols = numeric_cols(df, cols)
    if target not in df.columns or not cols:
        return {"valid": False, "reason": "missing"}
    d = df.dropna(subset=[target]).copy()
    if len(d) < 20:
        return {"valid": False, "reason": "too_few", "n": len(d)}
    y = d[target].astype(int).values
    if len(np.unique(y)) < 2:
        return {"valid": False, "reason": "single_class", "n": len(d)}
    X = d[cols].replace([np.inf, -np.inf], np.nan).values

    prob = np.zeros(len(d), dtype=float)
    pred = np.zeros(len(d), dtype=int)
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
        return {"valid": False, "reason": "no_folds", "n": len(d)}
    try:
        auc = roc_auc_score(y, prob)
    except Exception:
        auc = np.nan
    return {
        "valid": True,
        "n": int(len(d)),
        "n_features": int(len(cols)),
        "auc": safe_float(auc),
        "acc": safe_float(accuracy_score(y, pred)),
        "f1": safe_float(f1_score(y, pred, zero_division=0)),
        "bal_acc": safe_float(balanced_accuracy_score(y, pred)),
    }


def eval_multiclass(df, cols, target="submechanism", group_col="graph_id"):
    cols = numeric_cols(df, cols)
    if target not in df.columns or not cols:
        return {"valid": False, "reason": "missing"}
    d = df.dropna(subset=[target]).copy()
    if len(d) < 20:
        return {"valid": False, "reason": "too_few", "n": len(d)}
    y = d[target].astype(str).values
    if len(np.unique(y)) < 2:
        return {"valid": False, "reason": "single_class", "n": len(d)}
    X = d[cols].replace([np.inf, -np.inf], np.nan).values

    pred = np.array([""] * len(d), dtype=object)
    used = 0
    for train_idx, test_idx in cv_indices(d, y, group_col=group_col):
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
        "weighted_f1": safe_float(f1_score(y, pred, average="weighted", zero_division=0)),
        "bal_acc": safe_float(balanced_accuracy_score(y, pred)),
    }


def eval_regression(df, cols, target, group_col="graph_id"):
    cols = numeric_cols(df, cols)
    if target not in df.columns or not cols:
        return {"valid": False, "reason": "missing"}
    d = df.dropna(subset=[target]).copy()
    if len(d) < 20:
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
    try:
        corr = np.corrcoef(y, pred)[0, 1]
    except Exception:
        corr = np.nan
    return {
        "valid": True,
        "n": int(len(d)),
        "n_features": int(len(cols)),
        "r2": safe_float(r2_score(y, pred)),
        "corr": safe_float(corr),
    }


# -----------------------------
# Model helpers
# -----------------------------

def load_model_and_tokenizer(path, trust_remote_code=True):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code, local_files_only=True)
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


def lm_head_weight(model):
    return model.get_output_embeddings().weight.detach().float().cpu().numpy()


def get_candidate_token_ids(tokenizer):
    variants = {
        "C": ["C", " C", "\nC"],
        "E": ["E", " E", "\nE"],
        "OTHER": ["OTHER", " OTHER", "\nOTHER", "Other", " other"],
    }
    out = {}
    for label, texts in variants.items():
        ids = []
        for t in texts:
            enc = tokenizer(t, add_special_tokens=False).input_ids
            if len(enc):
                ids.append(enc[-1])
        out[label] = sorted(set(ids))
    return out


@torch.no_grad()
def forward_hidden_logits(model, tokenizer, text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    outputs = model(**enc, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states
    logits = outputs.logits[:, -1, :].float().detach().cpu().numpy()[0]
    hs = [h[:, -1, :].float().detach().cpu().numpy()[0] for h in hidden_states]
    return hs, logits


def to_hidx(layer_num, n_hidden_states):
    return max(0, min(n_hidden_states - 1, layer_num + 1))


def contiguous(a, b, n_hidden_states):
    return [to_hidx(i, n_hidden_states) for i in range(max(0, a), max(a, b) + 1)]


def aligned_windows(model_key, n_hidden_states):
    n_layers = n_hidden_states - 1
    def rel(p):
        return int(round(n_layers * p))

    if model_key == "qwen":
        topology = contiguous(9, 11, n_hidden_states) if n_layers >= 12 else contiguous(rel(.30), rel(.40), n_hidden_states)
        mechanism = contiguous(5, 6, n_hidden_states) if n_layers >= 7 else contiguous(rel(.15), rel(.25), n_hidden_states)
        downstream = contiguous(5, 6, n_hidden_states) if n_layers >= 7 else contiguous(rel(.15), rel(.25), n_hidden_states)
        decision = contiguous(20, 25, n_hidden_states) if n_layers >= 26 else contiguous(rel(.70), rel(.92), n_hidden_states)
    elif model_key == "llama":
        topology = contiguous(0, 1, n_hidden_states)
        mechanism = contiguous(2, 5, n_hidden_states) if n_layers >= 6 else contiguous(rel(.15), rel(.35), n_hidden_states)
        downstream = contiguous(0, 2, n_hidden_states)
        decision = contiguous(rel(.68), rel(.92), n_hidden_states)
    elif model_key == "gemma":
        topology = contiguous(0, 1, n_hidden_states)
        mechanism = contiguous(7, 8, n_hidden_states) if n_layers >= 9 else contiguous(rel(.30), rel(.45), n_hidden_states)
        downstream = contiguous(0, 1, n_hidden_states)
        decision = contiguous(18, 23, n_hidden_states) if n_layers >= 24 else contiguous(rel(.68), rel(.92), n_hidden_states)
    else:
        topology = contiguous(rel(.05), rel(.15), n_hidden_states)
        mechanism = contiguous(rel(.15), rel(.30), n_hidden_states)
        downstream = contiguous(rel(.05), rel(.20), n_hidden_states)
        decision = contiguous(rel(.70), rel(.92), n_hidden_states)

    if len(decision) >= 4:
        mid = len(decision) // 2
        early, late = decision[:mid], decision[mid:]
    else:
        early, late = decision, decision

    return {
        "topology": sorted(set(topology)),
        "mechanism_seed": sorted(set(mechanism)),
        "downstream_seed": sorted(set(downstream)),
        "decision": sorted(set(decision)),
        "decision_early": sorted(set(early)),
        "decision_late": sorted(set(late)),
    }


def layer_groups(n_hidden_states):
    n_layers = n_hidden_states - 1
    def hidx(layer):
        return max(0, min(n_hidden_states - 1, layer + 1))
    def span(a, b):
        return [hidx(l) for l in range(max(0, a), max(a, b) + 1)]

    if n_layers >= 28:
        return {
            "init": span(0, 6),
            "middle": span(7, 19),
            "boundary": span(20, 22),
            "basin": span(23, 25),
            "final": span(27, 27),
        }

    def rel(p):
        return int(round(n_layers * p))

    return {
        "init": span(0, rel(.22)),
        "middle": span(rel(.25), rel(.68)),
        "boundary": span(rel(.70), rel(.78)),
        "basin": span(rel(.80), rel(.90)),
        "final": span(n_layers, n_layers),
    }


def topk_for_hidden(h, W, k=TOPK):
    logits = h @ W.T
    k_eff = min(k, len(logits))
    idx = np.argpartition(-logits, k_eff - 1)[:k_eff]
    idx = idx[np.argsort(-logits[idx])]
    vals = logits[idx]
    return idx.astype(np.int64), vals.astype(np.float64)


def center_vec(ids, vals, W):
    ids = np.asarray(ids)
    vals = np.asarray(vals, dtype=np.float64)
    if ids.size == 0:
        return np.zeros(W.shape[1], dtype=np.float64)
    p = softmax_np(vals)
    return (W[ids] * p[:, None]).sum(axis=0)


def topk_entropy(vals):
    h, _ = entropy_bits_from_scores(vals)
    return h


def topk_spread(ids, W):
    ids = np.asarray(ids)
    if ids.size <= 1:
        return 0.0
    E = W[ids]
    c = E.mean(axis=0, keepdims=True)
    return float(np.mean(np.linalg.norm(E - c, axis=1)))


def score_candidate_tokens(logits, candidate_token_ids):
    scores = {}
    for label in CANDIDATES:
        ids = candidate_token_ids.get(label, [])
        scores[label] = float(np.max(logits[ids])) if ids else -1e9
    return scores


def window_topk_features(feat, prefix, hs, hidxs, W):
    centers, entropies, spreads = [], [], []
    for local_i, hidx in enumerate(hidxs):
        ids, vals = topk_for_hidden(hs[hidx], W, TOPK)
        ent = topk_entropy(vals)
        spr = topk_spread(ids, W)
        ctr = center_vec(ids, vals, W)
        centers.append(ctr)
        entropies.append(ent)
        spreads.append(spr)
        feat[f"topk_{prefix}_entropy_{local_i}"] = ent
        feat[f"topk_{prefix}_spread_{local_i}"] = spr
        feat[f"topk_{prefix}_top1_id_{local_i}"] = int(ids[0]) if len(ids) else -1

    if entropies:
        feat[f"topk_{prefix}_entropy_mean"] = float(np.mean(entropies))
        feat[f"topk_{prefix}_entropy_slope"] = slope(entropies)
        feat[f"topk_{prefix}_spread_mean"] = float(np.mean(spreads))
        feat[f"topk_{prefix}_spread_slope"] = slope(spreads)

    if len(centers) >= 2:
        drifts = [1.0 - cosine(centers[i], centers[i + 1]) for i in range(len(centers) - 1)]
        feat[f"topk_{prefix}_center_drift_mean"] = float(np.mean(drifts))
        feat[f"topk_{prefix}_center_drift_area"] = area(drifts)
        feat[f"topk_{prefix}_center_drift_slope"] = slope(drifts)
    else:
        feat[f"topk_{prefix}_center_drift_mean"] = 0.0
        feat[f"topk_{prefix}_center_drift_area"] = 0.0
        feat[f"topk_{prefix}_center_drift_slope"] = 0.0


def window_margin_features(feat, prefix, hs, hidxs, W, candidate_token_ids):
    margins = []
    other_gaps = []
    Hs = []
    Ns = []
    for local_i, hidx in enumerate(hidxs):
        logits = hs[hidx] @ W.T
        scores = score_candidate_tokens(logits, candidate_token_ids)
        arr = [scores[x] for x in CANDIDATES]
        h_ans, p_ans = entropy_bits_from_scores(arr)
        margin = scores["C"] - scores["E"]
        ogap = scores["OTHER"] - max(scores["C"], scores["E"])

        margins.append(margin)
        other_gaps.append(ogap)
        Hs.append(h_ans)
        Ns.append(n_eff(h_ans))

        feat[f"answer_H_bits_{prefix}_{local_i}"] = h_ans
        feat[f"answer_N_eff_{prefix}_{local_i}"] = n_eff(h_ans)
        feat[f"ans_margin_{prefix}_{local_i}"] = margin
        feat[f"other_gap_{prefix}_{local_i}"] = ogap
        feat[f"answer_score_C_{prefix}_{local_i}"] = scores["C"]
        feat[f"answer_score_E_{prefix}_{local_i}"] = scores["E"]
        feat[f"answer_score_OTHER_{prefix}_{local_i}"] = scores["OTHER"]

    if margins:
        m = np.asarray(margins, dtype=np.float64)
        am = np.abs(m)
        feat[f"ans_margin_{prefix}_mean"] = float(np.mean(m))
        feat[f"ans_margin_{prefix}_std"] = float(np.std(m))
        feat[f"ans_margin_{prefix}_slope"] = slope(m)
        feat[f"ans_margin_{prefix}_area"] = area(m)
        feat[f"ans_margin_{prefix}_minabs"] = float(np.min(am))
        feat[f"commit_abs_{prefix}_mean"] = float(np.mean(am))
        feat[f"commit_abs_{prefix}_std"] = float(np.std(am))
        feat[f"commit_abs_{prefix}_slope"] = slope(am)
        feat[f"commit_abs_{prefix}_area"] = area(am)
        feat[f"margin_sign_changes_{prefix}"] = float(np.sum(np.sign(m[:-1]) * np.sign(m[1:]) < 0)) if len(m) > 1 else 0.0
        feat[f"answer_H_{prefix}_mean"] = float(np.mean(Hs))
        feat[f"answer_H_{prefix}_min"] = float(np.min(Hs))
        feat[f"answer_H_{prefix}_max"] = float(np.max(Hs))
        feat[f"answer_N_eff_{prefix}_mean"] = float(np.mean(Ns))

        if len(m) >= 2:
            dm = np.diff(m)
            feat[f"dyn_{prefix}_velocity_mean"] = float(np.mean(dm))
            feat[f"dyn_{prefix}_velocity_abs_mean"] = float(np.mean(np.abs(dm)))
            feat[f"dyn_{prefix}_velocity_std"] = float(np.std(dm))
            feat[f"dyn_{prefix}_velocity_area"] = area(dm)
            if len(dm) >= 2:
                ddm = np.diff(dm)
                feat[f"dyn_{prefix}_curvature_abs_mean"] = float(np.mean(np.abs(ddm)))
                feat[f"dyn_{prefix}_accel_std"] = float(np.std(ddm))
            else:
                feat[f"dyn_{prefix}_curvature_abs_mean"] = 0.0
                feat[f"dyn_{prefix}_accel_std"] = 0.0

    if other_gaps:
        og = np.asarray(other_gaps, dtype=np.float64)
        feat[f"other_gap_{prefix}_mean"] = float(np.mean(og))
        feat[f"other_gap_{prefix}_slope"] = slope(og)
        feat[f"other_gap_{prefix}_area"] = area(og)


def extract_features(row, model, tokenizer, W, cand_ids, model_key):
    text = format_chat(tokenizer, row["prompt"])
    hs, final_logits = forward_hidden_logits(model, tokenizer, text)

    windows = aligned_windows(model_key, len(hs))
    lgroups = layer_groups(len(hs))

    final_scores = score_candidate_tokens(final_logits, cand_ids)
    final_pred = max(final_scores, key=final_scores.get)
    final_margin = final_scores["C"] - final_scores["E"]

    feat = {
        "answer_pred": final_pred,
        "score_C": final_scores["C"],
        "score_E": final_scores["E"],
        "score_OTHER": final_scores["OTHER"],
        "final_margin_C_minus_E": final_margin,
        "commit_abs_final_margin": abs(final_margin),
        "commit_high_abs_margin": int(abs(final_margin) >= 1.0),
        "gen_correct": int(final_pred == row["gold"]),
        "gen_error": int(final_pred != row["gold"]),
        "n_hidden_states": len(hs),
        "n_layers": len(hs) - 1,
    }

    # CFT aligned TopK windows.
    for pname in ["topology", "mechanism_seed", "downstream_seed", "decision"]:
        window_topk_features(feat, pname, hs, windows[pname], W)

    # Aligned margin/dynamics windows.
    for pname in ["mechanism_seed", "downstream_seed", "decision", "decision_early", "decision_late"]:
        window_margin_features(feat, pname, hs, windows[pname], W, cand_ids)

    # PSA-style layer-group answer entropy.
    for gname, hidxs in lgroups.items():
        window_margin_features(feat, f"psa_{gname}", hs, hidxs, W, cand_ids)

    # Cbit gains from PSA layer groups.
    H_init = feat.get("answer_H_psa_init_mean", np.nan)
    H_middle = feat.get("answer_H_psa_middle_mean", np.nan)
    H_boundary = feat.get("answer_H_psa_boundary_mean", np.nan)
    H_basin = feat.get("answer_H_psa_basin_mean", np.nan)
    H_final = feat.get("answer_H_psa_final_mean", np.nan)

    feat["raw_cbit_init_to_boundary"] = H_init - H_boundary
    feat["raw_cbit_init_to_basin"] = H_init - H_basin
    feat["raw_cbit_init_to_final"] = H_init - H_final
    feat["raw_cbit_middle_to_basin"] = H_middle - H_basin
    feat["raw_cbit_boundary_to_basin"] = H_boundary - H_basin

    # DYN4-style aligned composite.
    early_abs = feat.get("commit_abs_decision_early_mean", np.nan)
    late_abs = feat.get("commit_abs_decision_late_mean", np.nan)
    feat["dyn4_aligned_late_minus_early_abs"] = safe_float(late_abs) - safe_float(early_abs)
    feat["dyn4_aligned_late_over_early_abs"] = safe_float(late_abs) / (safe_float(early_abs) + 1e-6)
    feat["dyn4_aligned_boundary_residence_inv"] = 1.0 / (safe_float(feat.get("ans_margin_decision_minabs", np.nan)) + 1e-6)

    # Binding proxies, kept as answer-readout controls, not no-answer.
    feat["binding_priority_proxy_CminusE"] = final_margin
    feat["binding_priority_proxy_OTHERmax"] = final_scores["OTHER"] - max(final_scores["C"], final_scores["E"])

    # Effective Cbit quality proxy.
    q = 1.0 if feat["gen_error"] == 0 else 0.0
    mech = str(row.get("mechanism", ""))
    cond = str(row.get("condition", ""))
    if any(x in mech for x in ["stable", "closure", "competition", "logic", "random"]):
        q += 0.10
    if "random_broken" in cond and feat["gen_error"] == 1:
        q -= 0.10
    q = max(0.0, min(1.0, q))
    feat["Q_correctness_structural"] = q
    feat["Cbit_eff_basin"] = feat["raw_cbit_init_to_basin"] * q
    feat["Cbit_eff_final"] = feat["raw_cbit_init_to_final"] * q

    return feat


def build_feature_groups(df):
    topk = unique_keep_order(
        cols_by_prefix(df, ["topk_topology_"]) +
        cols_by_prefix(df, ["topk_mechanism_seed_"]) +
        cols_by_prefix(df, ["topk_downstream_seed_"]) +
        cols_by_prefix(df, ["topk_decision_"])
    )

    answer = numeric_cols(df, [
        "score_C", "score_E", "score_OTHER", "final_margin_C_minus_E",
        "commit_abs_final_margin", "commit_high_abs_margin",
        "binding_priority_proxy_CminusE", "binding_priority_proxy_OTHERmax",
    ]) + cols_by_prefix(df, ["ans_margin_"])

    dynamics = unique_keep_order(
        cols_by_prefix(df, ["dyn_", "dyn4_"]) +
        cols_by_prefix(df, ["commit_abs_"]) +
        cols_by_prefix(df, ["other_gap_"]) +
        cols_by_prefix(df, ["margin_sign_changes_"])
    )
    dynamics = [c for c in dynamics if c not in answer]

    cft = unique_keep_order(topk + dynamics)
    return {
        "TopK_aligned": numeric_cols(df, topk),
        "Dynamics_aligned": numeric_cols(df, dynamics),
        "CFT_NoAnswer": numeric_cols(df, cft),
        "AnswerReadout": numeric_cols(df, answer),
    }


def run_evaluation(df, groups):
    rows = []
    for gname, cols in groups.items():
        for target in [
            "raw_cbit_init_to_basin", "raw_cbit_init_to_final",
            "Cbit_eff_basin", "Cbit_eff_final",
            "Q_correctness_structural",
        ]:
            res = eval_regression(df, cols, target, group_col="graph_id")
            rows.append({
                "feature_group": gname,
                "target": target,
                "task": "regression",
                "n_features": len(cols),
                "r2": res.get("r2", np.nan),
                "corr": res.get("corr", np.nan),
                "auc": np.nan,
                "f1": np.nan,
                "acc": np.nan,
                "valid": res.get("valid", False),
                "n": res.get("n", np.nan),
            })
        for target in ["gen_error"]:
            res = eval_binary(df, cols, target, group_col="graph_id")
            rows.append({
                "feature_group": gname,
                "target": target,
                "task": "binary",
                "n_features": len(cols),
                "r2": np.nan,
                "corr": np.nan,
                "auc": res.get("auc", np.nan),
                "f1": res.get("f1", np.nan),
                "acc": res.get("acc", np.nan),
                "valid": res.get("valid", False),
                "n": res.get("n", np.nan),
            })
        for target in ["mechanism", "submechanism"]:
            res = eval_multiclass(df, cols, target=target, group_col="graph_id")
            rows.append({
                "feature_group": gname,
                "target": target,
                "task": "multiclass",
                "n_features": len(cols),
                "r2": np.nan,
                "corr": np.nan,
                "auc": np.nan,
                "f1": res.get("macro_f1", np.nan),
                "acc": res.get("acc", np.nan),
                "valid": res.get("valid", False),
                "n": res.get("n", np.nan),
            })

    return pd.DataFrame(rows)


def group_summary(df):
    rows = []
    if "gen_error" in df.columns:
        for val, sub in df.groupby("gen_error"):
            rows.append({
                "group_type": "gen_error",
                "group": int(val),
                "n": int(len(sub)),
                "raw_cbit_basin_mean": safe_float(sub["raw_cbit_init_to_basin"].mean()),
                "raw_cbit_final_mean": safe_float(sub["raw_cbit_init_to_final"].mean()),
                "cbit_eff_basin_mean": safe_float(sub["Cbit_eff_basin"].mean()),
                "cbit_eff_final_mean": safe_float(sub["Cbit_eff_final"].mean()),
                "Q_mean": safe_float(sub["Q_correctness_structural"].mean()),
                "error_rate": safe_float(sub["gen_error"].mean()),
            })
    for col in ["condition", "mechanism", "submechanism"]:
        if col in df.columns:
            for val, sub in df.groupby(col):
                rows.append({
                    "group_type": col,
                    "group": str(val),
                    "n": int(len(sub)),
                    "raw_cbit_basin_mean": safe_float(sub["raw_cbit_init_to_basin"].mean()),
                    "raw_cbit_final_mean": safe_float(sub["raw_cbit_init_to_final"].mean()),
                    "cbit_eff_basin_mean": safe_float(sub["Cbit_eff_basin"].mean()),
                    "cbit_eff_final_mean": safe_float(sub["Cbit_eff_final"].mean()),
                    "Q_mean": safe_float(sub["Q_correctness_structural"].mean()),
                    "error_rate": safe_float(sub["gen_error"].mean()),
                })
    return pd.DataFrame(rows)


def get_metric(model_summary, group, target, key):
    r = model_summary[(model_summary["feature_group"] == group) & (model_summary["target"] == target)]
    if len(r) == 0:
        return np.nan
    return safe_float(r.iloc[0].get(key, np.nan))


def verdict_for_model(model_key, df, model_summary, gsummary):
    metrics = {
        "topk_eff_basin_corr": get_metric(model_summary, "TopK_aligned", "Cbit_eff_basin", "corr"),
        "dyn_eff_basin_corr": get_metric(model_summary, "Dynamics_aligned", "Cbit_eff_basin", "corr"),
        "cft_eff_basin_corr": get_metric(model_summary, "CFT_NoAnswer", "Cbit_eff_basin", "corr"),
        "topk_eff_basin_r2": get_metric(model_summary, "TopK_aligned", "Cbit_eff_basin", "r2"),
        "dyn_eff_basin_r2": get_metric(model_summary, "Dynamics_aligned", "Cbit_eff_basin", "r2"),
        "cft_eff_basin_r2": get_metric(model_summary, "CFT_NoAnswer", "Cbit_eff_basin", "r2"),
        "topk_raw_basin_corr": get_metric(model_summary, "TopK_aligned", "raw_cbit_init_to_basin", "corr"),
        "cft_raw_basin_corr": get_metric(model_summary, "CFT_NoAnswer", "raw_cbit_init_to_basin", "corr"),
        "cft_gen_error_auc": get_metric(model_summary, "CFT_NoAnswer", "gen_error", "auc"),
        "topk_gen_error_auc": get_metric(model_summary, "TopK_aligned", "gen_error", "auc"),
    }

    ge = gsummary[gsummary["group_type"] == "gen_error"]
    corr = ge[ge["group"].astype(str) == "0"]
    wrong = ge[ge["group"].astype(str) == "1"]

    if len(corr) and len(wrong):
        metrics["correct_raw_basin_mean"] = safe_float(corr.iloc[0]["raw_cbit_basin_mean"])
        metrics["wrong_raw_basin_mean"] = safe_float(wrong.iloc[0]["raw_cbit_basin_mean"])
        metrics["correct_eff_basin_mean"] = safe_float(corr.iloc[0]["cbit_eff_basin_mean"])
        metrics["wrong_eff_basin_mean"] = safe_float(wrong.iloc[0]["cbit_eff_basin_mean"])
        metrics["raw_correct_minus_wrong"] = metrics["correct_raw_basin_mean"] - metrics["wrong_raw_basin_mean"]
        metrics["eff_correct_minus_wrong"] = metrics["correct_eff_basin_mean"] - metrics["wrong_eff_basin_mean"]
    else:
        metrics["correct_raw_basin_mean"] = np.nan
        metrics["wrong_raw_basin_mean"] = np.nan
        metrics["correct_eff_basin_mean"] = np.nan
        metrics["wrong_eff_basin_mean"] = np.nan
        metrics["raw_correct_minus_wrong"] = np.nan
        metrics["eff_correct_minus_wrong"] = np.nan

    cft_eff_corr = metrics["cft_eff_basin_corr"]
    eff_gap = metrics["eff_correct_minus_wrong"]
    cft_eff_gain_over_topk = metrics["cft_eff_basin_corr"] - metrics["topk_eff_basin_corr"]

    reasons = []
    verdict = "UNDETERMINED"

    if np.isfinite(cft_eff_corr) and cft_eff_corr > 0.30 and np.isfinite(eff_gap) and eff_gap > 0:
        verdict = "PASS_PSA2B_CONSTRAINT_TO_EFFECTIVE_CBIT"
        reasons.append("CFT no-answer proxy predicts Cbit_eff and correct samples have higher Cbit_eff.")
    elif np.isfinite(metrics["cft_raw_basin_corr"]) and metrics["cft_raw_basin_corr"] > 0.30:
        verdict = "PARTIAL_PSA2B_RAW_CBIT_ONLY"
        reasons.append("CFT no-answer proxy predicts raw Cbit but effective separation is weak.")
    elif np.isfinite(eff_gap) and eff_gap > 0:
        verdict = "PARTIAL_PSA2B_EFFECTIVE_CBIT_SEPARATES"
        reasons.append("Effective Cbit separates correct/wrong but CFT prediction is weak.")
    else:
        verdict = "FAIL_PSA2B"

    if np.isfinite(cft_eff_gain_over_topk) and cft_eff_gain_over_topk > 0:
        reasons.append("CFT improves Cbit_eff prediction over TopK.")

    return {
        "model_key": model_key,
        "verdict": verdict,
        "reasons": reasons,
        "metrics": metrics,
    }


def run_one_model(model_key):
    spec = MODEL_SPECS[model_key]
    model_out = OUT_DIR / model_key
    model_out.mkdir(parents=True, exist_ok=True)

    info = {
        "model_key": model_key,
        "model_name": spec["model_name"],
        "path": spec["path"],
        "exists": os.path.exists(spec["path"]),
        "status": "pending",
    }

    if not os.path.exists(spec["path"]):
        info.update({"status": "missing_path", "error": f"Path not found: {spec['path']}"})
        return info

    dataset = build_dataset()
    dataset.to_csv(model_out / f"psa2b_{model_key}_dataset.csv", index=False, encoding="utf-8-sig")

    model = None
    try:
        t0 = time.time()
        model, tokenizer = load_model_and_tokenizer(spec["path"], spec.get("trust_remote_code", True))
        W = lm_head_weight(model)
        cand_ids = get_candidate_token_ids(tokenizer)

        rows, errors = [], []
        for i, row in dataset.iterrows():
            if i % 20 == 0:
                print(f"[{model_key}] extracting row {i}/{len(dataset)}", flush=True)
            try:
                feat = extract_features(row, model, tokenizer, W, cand_ids, model_key)
                rows.append({**row.to_dict(), **feat})
            except Exception as e:
                errors.append({
                    "row_id": int(row["row_id"]),
                    "condition": row["condition"],
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
                if len(errors) <= 5:
                    print(f"[warn][{model_key}] row {row['row_id']} failed: {type(e).__name__}: {e}", flush=True)

        feat_df = pd.DataFrame(rows)
        err_df = pd.DataFrame(errors)

        feat_df.to_csv(model_out / f"psa2b_{model_key}_features.csv", index=False, encoding="utf-8-sig")
        err_df.to_csv(model_out / f"psa2b_{model_key}_errors.csv", index=False, encoding="utf-8-sig")

        if feat_df.empty:
            info.update({"status": "feature_empty", "n_errors": len(errors)})
            return info

        groups = build_feature_groups(feat_df)
        msummary = run_evaluation(feat_df, groups)
        gsummary = group_summary(feat_df)

        msummary.to_csv(model_out / f"psa2b_{model_key}_model_summary.csv", index=False, encoding="utf-8-sig")
        gsummary.to_csv(model_out / f"psa2b_{model_key}_group_summary.csv", index=False, encoding="utf-8-sig")

        verdict = verdict_for_model(model_key, feat_df, msummary, gsummary)
        verdict.update({
            "status": "done",
            "n_rows": int(len(feat_df)),
            "n_graphs": int(feat_df["graph_id"].nunique()) if "graph_id" in feat_df.columns else None,
            "n_errors": int(len(errors)),
            "runtime_sec": time.time() - t0,
            "feature_set_sizes": {k: len(v) for k, v in groups.items()},
            "outputs": {
                "features": str(model_out / f"psa2b_{model_key}_features.csv"),
                "model_summary": str(model_out / f"psa2b_{model_key}_model_summary.csv"),
                "group_summary": str(model_out / f"psa2b_{model_key}_group_summary.csv"),
            },
            "top_rows": msummary.sort_values(["target", "corr", "r2"], ascending=[True, False, False]).head(20).to_dict(orient="records"),
        })

        with open(model_out / f"psa2b_{model_key}_verdict.json", "w", encoding="utf-8") as f:
            json.dump(verdict, f, ensure_ascii=False, indent=2)

        info.update(verdict)

    except Exception as e:
        info.update({
            "status": "failed",
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc(),
        })

    finally:
        try:
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass

    return info


def main():
    all_infos = []

    for model_key in MODELS_TO_RUN:
        print("=" * 80)
        print(f"[PSA-2B] Running model: {model_key}")
        info = run_one_model(model_key)
        all_infos.append(info)
        print(json.dumps({
            "model_key": info.get("model_key"),
            "status": info.get("status"),
            "verdict": info.get("verdict"),
            "metrics": info.get("metrics"),
            "error": info.get("error"),
        }, ensure_ascii=False, indent=2))

    rows = []
    for info in all_infos:
        m = info.get("metrics", {}) or {}
        rows.append({
            "model_key": info.get("model_key"),
            "status": info.get("status"),
            "verdict": info.get("verdict"),
            "cft_eff_basin_corr": m.get("cft_eff_basin_corr"),
            "topk_eff_basin_corr": m.get("topk_eff_basin_corr"),
            "dyn_eff_basin_corr": m.get("dyn_eff_basin_corr"),
            "cft_eff_basin_r2": m.get("cft_eff_basin_r2"),
            "topk_eff_basin_r2": m.get("topk_eff_basin_r2"),
            "cft_raw_basin_corr": m.get("cft_raw_basin_corr"),
            "cft_gen_error_auc": m.get("cft_gen_error_auc"),
            "topk_gen_error_auc": m.get("topk_gen_error_auc"),
            "eff_correct_minus_wrong": m.get("eff_correct_minus_wrong"),
            "raw_correct_minus_wrong": m.get("raw_correct_minus_wrong"),
            "n_rows": info.get("n_rows"),
            "n_errors": info.get("n_errors"),
            "error": info.get("error"),
        })

    cross = pd.DataFrame(rows)
    cross.to_csv(OUT_DIR / "psa2b_cross_model_summary.csv", index=False, encoding="utf-8-sig")

    done = [x for x in all_infos if x.get("status") == "done"]
    pass_count = sum(str(x.get("verdict", "")).startswith("PASS") for x in done)
    partial_count = sum(str(x.get("verdict", "")).startswith("PARTIAL") for x in done)

    if len(done) == 0:
        global_verdict = "FAIL_PSA2B_NO_MODELS"
    elif pass_count == len(done):
        global_verdict = "PASS_PSA2B"
    elif pass_count + partial_count == len(done):
        global_verdict = "PARTIAL_PSA2B"
    elif pass_count >= 1:
        global_verdict = "MIXED_PSA2B"
    else:
        global_verdict = "FAIL_PSA2B"

    global_obj = {
        "verdict": global_verdict,
        "n_models_done": len(done),
        "n_models_pass": pass_count,
        "n_models_partial": partial_count,
        "models": all_infos,
        "outputs": {
            "cross_summary": str(OUT_DIR / "psa2b_cross_model_summary.csv"),
        },
    }

    with open(OUT_DIR / "psa2b_verdict.json", "w", encoding="utf-8") as f:
        json.dump(global_obj, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(json.dumps(global_obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
