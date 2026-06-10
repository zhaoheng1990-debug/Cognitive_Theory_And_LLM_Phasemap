# -*- coding: utf-8 -*-
r"""
CFT-3: Cross-Model Constraint Field Proxy Audit

Goal
----
Validate whether the CFT-2 no-answer constraint field proxy generalizes across:

  Qwen  : D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main
  Llama : D:\model\Llama-3.2-1B-Instruct
  Gemma : D:\model\gemma-2-2b-it

Theory
------
CFT-2 on Qwen showed:

  F_c^(2) = TopK + Binding/Priority + DYN4 Convergence

can strongly predict:
  - GenError
  - mechanism
  - submechanism
  - commitment magnitude

without AnswerReadout.

CFT-3 asks:

  Does this no-answer constraint-field proxy exist across model-specific manifolds M_m?

Important
---------
This script is self-contained and generates its own controlled relational-closure dataset.
It does NOT depend on previous Qwen feature CSVs.

Outputs
-------
C:\Users\ZH\Desktop\AGI\outputs\cft3_outputs\
  cft3_cross_model_summary.csv
  cft3_<model>_dataset.csv
  cft3_<model>_features.csv
  cft3_<model>_model_summary.csv
  cft3_<model>_verdict.json
  cft3_verdict.json

Notes
-----
This is a Lite cross-model version for feasibility.
It extracts:
  - TopK topology features
  - Binding/Priority direction proxies
  - DYN4 convergence proxies
  - AnswerReadout baselines

If a model path is missing or load fails, it skips that model and records the error.
"""

import os
import re
import json
import math
import time
import traceback
import warnings
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

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

from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# Paths
# -----------------------------

MODEL_SPECS = [
    {
        "model_key": "qwen",
        "model_name": "Qwen2.5-1.5B-Instruct",
        "path": r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main",
        "trust_remote_code": True,
    },
    {
        "model_key": "llama",
        "model_name": "Llama-3.2-1B-Instruct",
        "path": r"D:\model\Llama-3.2-1B-Instruct",
        "trust_remote_code": True,
    },
    {
        "model_key": "gemma",
        "model_name": "gemma-2-2b-it",
        "path": r"D:\model\gemma-2-2b-it",
        "trust_remote_code": True,
    },
]

OUT_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\cft3_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Lite config
N_GRAPHS = 24
N_SURFACES = 3
TOPK = 80
MAX_NEW_TOKENS = 6
N_SPLITS = 4

# If VRAM is tight, reduce N_GRAPHS or TOPK.



def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        y = float(x)
        return y
    except Exception:
        return default


# -----------------------------
# Dataset
# -----------------------------

ENTITIES_A = [
    "mira", "nolan", "sora", "tavin", "lena", "orion",
    "vexa", "darin", "selka", "brin", "kael", "nira",
    "faro", "yuna", "pavo", "lumi", "taro", "zena",
    "bex", "rilo", "sena", "mako", "viro", "doma",
    "hani", "lexa", "nemo", "runa", "silo", "tova",
]

GROUPS_B = [
    "amber guild", "blue circle", "crimson order", "delta clan", "ember house", "frost league",
    "green lodge", "harbor unit", "iron band", "jade group", "kepler sect", "lunar team",
    "marble council", "north ring", "opal class", "prime cohort", "quartz line", "river squad",
    "silver branch", "terra block", "umber cell", "violet union", "willow set", "zenith party",
]

GROUPS_C = [
    "atlas zone", "boreal zone", "cobalt zone", "dawn zone", "equinox zone", "falcon zone",
    "garden zone", "helios zone", "ivory zone", "juniper zone", "kestrel zone", "lotus zone",
    "meteor zone", "nova zone", "onyx zone", "prairie zone", "quasar zone", "raven zone",
    "summit zone", "tundra zone", "umbra zone", "velvet zone", "willow zone", "zephyr zone",
]

GROUPS_E = [
    "arcadia zone", "bright zone", "cinder zone", "dusk zone", "elm zone", "fern zone",
    "granite zone", "hazel zone", "indigo zone", "jasmine zone", "kite zone", "lagoon zone",
    "meadow zone", "nebula zone", "oasis zone", "polar zone", "quiet zone", "reed zone",
    "sunset zone", "thistle zone", "ultra zone", "valley zone", "winter zone", "zen zone",
]


SURFACES = [
    "Use the facts below and answer with only C, E, or OTHER.\nFacts:\n{facts}\nQuestion: Which final group does {A} belong to?\nAnswer:",
    "Read the mini knowledge base. Reply only with C, E, or OTHER.\n{facts}\nQuery: final membership of {A}?\nAnswer:",
    "Given these relations, decide the final target for {A}. Output exactly C, E, or OTHER.\n{facts}\nFinal answer:",
]


def build_dataset():
    rows = []
    rng = np.random.default_rng(RANDOM_SEED)

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
            surface_id = f"surface_{si}"
            template = SURFACES[si % len(SURFACES)]

            for cond in conditions:
                mechanism = None
                submechanism = None
                gold = None

                if cond == "stable_clean":
                    facts = f"{A} belongs to {B}.\n{B} belongs to {C}."
                    mechanism = "stable"
                    submechanism = "stable_core"
                    gold = "C"

                elif cond == "stable_redundant":
                    facts = f"{A} belongs to {B}.\n{B} belongs to {C}.\nNote: {C} is the final group in this example."
                    mechanism = "stable"
                    submechanism = "stable_redundant"
                    gold = "C"

                elif cond == "competition_equal":
                    facts = f"{A} belongs to {B}.\n{B} may belong to {C}.\n{B} may belong to {E}.\nBoth options are equally supported."
                    mechanism = "competition"
                    submechanism = "competition_equal"
                    gold = "OTHER"

                elif cond == "source_trusted_C":
                    facts = f"Source 1 says: {B} belongs to {E}.\nTrusted source says: {B} belongs to {C}.\n{A} belongs to {B}.\nUse the trusted source."
                    mechanism = "competition"
                    submechanism = "competition_source_C"
                    gold = "C"

                elif cond == "source_trusted_E":
                    facts = f"Source 1 says: {B} belongs to {C}.\nTrusted source says: {B} belongs to {E}.\n{A} belongs to {B}.\nUse the trusted source."
                    mechanism = "competition"
                    submechanism = "competition_source_E"
                    gold = "E"

                elif cond == "closure_update_C":
                    facts = f"Old rule: {B} belongs to {E}.\nNew rule: {B} belongs to {C}.\n{A} belongs to {B}.\nUse the new rule."
                    mechanism = "closure"
                    submechanism = "closure_update_C"
                    gold = "C"

                elif cond == "closure_update_E":
                    facts = f"Old rule: {B} belongs to {C}.\nNew rule: {B} belongs to {E}.\n{A} belongs to {B}.\nUse the new rule."
                    mechanism = "closure"
                    submechanism = "closure_update_E"
                    gold = "E"

                elif cond == "logic_contradiction":
                    facts = f"Rule: {A} cannot belong to both {C} and {E}.\nClaim 1: {A} belongs to {C}.\nClaim 2: {A} belongs to {E}.\nIf the facts contradict, answer OTHER."
                    mechanism = "logic"
                    submechanism = "logic_direct"
                    gold = "OTHER"

                elif cond == "random_broken":
                    Z = GROUPS_E[(gi + 7) % len(GROUPS_E)]
                    facts = f"{A} belongs to {B}.\n{B} is mentioned near {Z}.\nNo final membership relation is provided."
                    mechanism = "random"
                    submechanism = "random_broken"
                    gold = "OTHER"

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
                    "C": C,
                    "E": E,
                    "prompt": prompt,
                })

    return pd.DataFrame(rows)


# -----------------------------
# Model helpers
# -----------------------------

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
        messages = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return prompt


def get_choice_token_ids(tokenizer):
    # We use multiple variants because tokenization differs across models.
    variants = {
        "C": ["C", " C", "\nC"],
        "E": ["E", " E", "\nE"],
        "OTHER": ["OTHER", " OTHER", "\nOTHER", "Other", " other"],
    }
    out = {}
    for label, toks in variants.items():
        ids = []
        for t in toks:
            enc = tokenizer(t, add_special_tokens=False).input_ids
            if len(enc) >= 1:
                ids.append(enc[-1])
        # unique
        out[label] = sorted(set(ids))
    return out


@torch.no_grad()
def forward_hidden_logits(model, tokenizer, text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    outputs = model(**enc, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states  # tuple: emb + layers
    logits = outputs.logits[:, -1, :].float().detach().cpu().numpy()[0]
    # last token hidden for each layer/state
    hs = []
    for h in hidden_states:
        hs.append(h[:, -1, :].float().detach().cpu().numpy()[0])
    return hs, logits


def layer_indices(n_hidden_states):
    # hidden_states includes embedding at 0.
    # We map model depths to relative windows.
    # decision main window roughly 70%-92% of layers.
    n_layers = n_hidden_states - 1
    if n_layers <= 0:
        return {
            "init": list(range(0, n_hidden_states)),
            "main": list(range(0, n_hidden_states)),
            "early": list(range(0, n_hidden_states)),
            "late": list(range(0, n_hidden_states)),
        }

    def hs_idx(layer_num):
        # transformer layer L maps to hidden_states index L+1
        return max(0, min(n_hidden_states - 1, layer_num + 1))

    init_layers = list(range(0, min(n_layers, 6) + 1))
    main_start = int(round(n_layers * 0.70))
    main_end = int(round(n_layers * 0.92))
    main_layers = list(range(main_start, max(main_start + 1, main_end) + 1))

    early_start = int(round(n_layers * 0.70))
    early_end = int(round(n_layers * 0.80))
    late_start = int(round(n_layers * 0.80))
    late_end = int(round(n_layers * 0.92))

    return {
        "init": [hs_idx(l) for l in init_layers],
        "main": [hs_idx(l) for l in main_layers],
        "early": [hs_idx(l) for l in range(early_start, max(early_start + 1, early_end) + 1)],
        "late": [hs_idx(l) for l in range(late_start, max(late_start + 1, late_end) + 1)],
    }


def lm_head_weight(model):
    # Most CausalLMs expose output embeddings.
    W = model.get_output_embeddings().weight.detach().float().cpu().numpy()
    return W


def topk_for_hidden(h, W, k=TOPK):
    logits = h @ W.T
    if k >= len(logits):
        idx = np.argsort(-logits)
    else:
        idx = np.argpartition(-logits, k)[:k]
        idx = idx[np.argsort(-logits[idx])]
    vals = logits[idx]
    return idx.astype(np.int64), vals.astype(np.float64)


def entropy_from_vals(vals):
    v = np.asarray(vals, dtype=np.float64)
    if v.size == 0:
        return np.nan
    v = v - np.max(v)
    p = np.exp(v)
    p = p / (p.sum() + 1e-12)
    return float(-np.sum(p * np.log(p + 1e-12)))


def spread_from_ids(ids, W):
    if len(ids) <= 1:
        return 0.0
    E = W[np.asarray(ids)]
    c = E.mean(axis=0, keepdims=True)
    d = np.linalg.norm(E - c, axis=1)
    return float(np.mean(d))


def center_vec(ids, vals, W):
    ids = np.asarray(ids)
    vals = np.asarray(vals, dtype=np.float64)
    if len(ids) == 0:
        return np.zeros(W.shape[1], dtype=np.float64)
    v = vals - vals.max()
    p = np.exp(v)
    p = p / (p.sum() + 1e-12)
    return (W[ids] * p[:, None]).sum(axis=0)


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / den)


def slope(xs):
    arr = np.asarray(xs, dtype=np.float64)
    if arr.size < 2 or np.any(~np.isfinite(arr)):
        return np.nan
    x = np.arange(arr.size, dtype=np.float64)
    try:
        return float(np.polyfit(x, arr, 1)[0])
    except Exception:
        return np.nan


def area(xs):
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    if arr.size == 1:
        return float(arr[0])
    return float(np.sum((arr[:-1] + arr[1:]) * 0.5))


def last_token_choice(logits, choice_ids):
    scores = {}
    for label, ids in choice_ids.items():
        if len(ids) == 0:
            scores[label] = -1e9
        else:
            scores[label] = float(np.max(logits[ids]))
    pred = max(scores, key=scores.get)
    return pred, scores


def extract_features_for_row(row, model, tokenizer, W, choice_ids):
    text = format_chat(tokenizer, row["prompt"])
    hs, final_logits = forward_hidden_logits(model, tokenizer, text)

    idxs = layer_indices(len(hs))

    pred, scores = last_token_choice(final_logits, choice_ids)
    C_score = scores.get("C", -1e9)
    E_score = scores.get("E", -1e9)
    O_score = scores.get("OTHER", -1e9)
    final_margin = C_score - E_score
    commit_abs = abs(final_margin)

    feat = {
        "answer_pred": pred,
        "score_C": C_score,
        "score_E": E_score,
        "score_OTHER": O_score,
        "final_margin_C_minus_E": final_margin,
        "commit_abs_final_margin": commit_abs,
        "commit_high_abs_margin": int(commit_abs >= 1.0),
        "gen_correct": int(pred == row["gold"]),
        "gen_error": int(pred != row["gold"]),
    }

    # Layerwise answer margins and TopK
    centers = []
    entropies = []
    spreads = []
    ans_margins = []

    all_main_layers = idxs["main"]
    for local_i, hidx in enumerate(all_main_layers):
        h = hs[hidx]
        # Answer margin through lm_head
        logits = h @ W.T
        c_layer = max([logits[i] for i in choice_ids["C"]]) if choice_ids["C"] else -1e9
        e_layer = max([logits[i] for i in choice_ids["E"]]) if choice_ids["E"] else -1e9
        margin = float(c_layer - e_layer)
        ans_margins.append(margin)
        feat[f"ans_margin_main_{local_i}"] = margin

        ids, vals = topk_for_hidden(h, W, k=TOPK)
        ent = entropy_from_vals(vals)
        spr = spread_from_ids(ids, W)
        ctr = center_vec(ids, vals, W)
        centers.append(ctr)
        entropies.append(ent)
        spreads.append(spr)

        feat[f"topk_main_entropy_{local_i}"] = ent
        feat[f"topk_main_spread_{local_i}"] = spr
        feat[f"topk_main_top1_id_{local_i}"] = int(ids[0]) if len(ids) else -1

    if ans_margins:
        feat["ans_margin_main_mean"] = float(np.mean(ans_margins))
        feat["ans_margin_main_std"] = float(np.std(ans_margins))
        feat["ans_margin_main_slope"] = slope(ans_margins)
        feat["ans_margin_main_area"] = area(ans_margins)
        feat["ans_margin_main_minabs"] = float(np.min(np.abs(ans_margins)))
        feat["ans_margin_first_flip"] = float(np.sum(np.sign(ans_margins[:-1]) * np.sign(ans_margins[1:]) < 0)) if len(ans_margins) > 1 else 0.0

        abs_m = np.abs(np.asarray(ans_margins, dtype=np.float64))
        feat["dyn4_abs_margin_area"] = area(abs_m)
        feat["dyn4_abs_margin_slope"] = slope(abs_m)
        feat["dyn4_margin_area"] = area(ans_margins)
        feat["dyn4_margin_slope"] = slope(ans_margins)
        feat["dyn4_start_abs"] = float(abs_m[0])
        feat["dyn4_final_abs"] = float(abs_m[-1])
        if len(abs_m) >= 4:
            mid = len(abs_m) // 2
            early = float(np.mean(abs_m[:mid]))
            late = float(np.mean(abs_m[mid:]))
            feat["dyn4_late_minus_early_abs"] = late - early
            feat["dyn4_late_over_early_abs"] = late / (early + 1e-6)
        else:
            feat["dyn4_late_minus_early_abs"] = np.nan
            feat["dyn4_late_over_early_abs"] = np.nan

        q25 = np.quantile(abs_m, 0.25)
        q50 = np.quantile(abs_m, 0.50)
        feat["dyn4_boundary_residence_q25"] = float(np.mean(abs_m <= q25))
        feat["dyn4_boundary_residence_q50"] = float(np.mean(abs_m <= q50))

        if len(ans_margins) >= 2:
            dm = np.diff(np.asarray(ans_margins, dtype=np.float64))
            feat["dyn_dmargin_mean"] = float(np.mean(dm))
            feat["dyn_dmargin_std"] = float(np.std(dm))
            feat["dyn_abs_dmargin_mean"] = float(np.mean(np.abs(dm)))
            feat["dyn_dmargin_area"] = area(dm)
            feat["dyn2_velocity_mean"] = float(np.mean(dm))
            feat["dyn2_velocity_abs_mean"] = float(np.mean(np.abs(dm)))
            feat["dyn2_velocity_std"] = float(np.std(dm))
            if len(dm) >= 2:
                ddm = np.diff(dm)
                feat["dyn_second_abs_mean"] = float(np.mean(np.abs(ddm)))
                feat["dyn_curvature_proxy"] = float(np.mean(np.abs(ddm)))
                feat["dyn2_accel_abs_mean"] = float(np.mean(np.abs(ddm)))
                feat["dyn2_accel_std"] = float(np.std(ddm))
            else:
                feat["dyn_second_abs_mean"] = 0.0
                feat["dyn_curvature_proxy"] = 0.0
                feat["dyn2_accel_abs_mean"] = 0.0
                feat["dyn2_accel_std"] = 0.0
            feat["dyn2_sign_changes"] = float(np.sum(np.sign(ans_margins[:-1]) * np.sign(ans_margins[1:]) < 0))
        else:
            for k in [
                "dyn_dmargin_mean", "dyn_dmargin_std", "dyn_abs_dmargin_mean", "dyn_dmargin_area",
                "dyn_second_abs_mean", "dyn_curvature_proxy", "dyn2_velocity_mean",
                "dyn2_velocity_abs_mean", "dyn2_velocity_std", "dyn2_accel_abs_mean",
                "dyn2_accel_std", "dyn2_sign_changes"
            ]:
                feat[k] = 0.0

    if entropies:
        feat["topk_main_entropy_mean"] = float(np.mean(entropies))
        feat["topk_main_entropy_slope"] = slope(entropies)
        feat["topk_main_spread_mean"] = float(np.mean(spreads))
        feat["topk_main_spread_slope"] = slope(spreads)

    if len(centers) >= 2:
        drifts = [1.0 - cosine(centers[i], centers[i + 1]) for i in range(len(centers) - 1)]
        feat["topk_center_drift_mean"] = float(np.mean(drifts))
        feat["topk_center_drift_area"] = area(drifts)
        feat["topk_center_drift_slope"] = slope(drifts)
    else:
        feat["topk_center_drift_mean"] = 0.0
        feat["topk_center_drift_area"] = 0.0
        feat["topk_center_drift_slope"] = 0.0

    # Init TopK features
    init_ent = []
    init_spr = []
    for local_i, hidx in enumerate(idxs["init"]):
        h = hs[hidx]
        ids, vals = topk_for_hidden(h, W, k=TOPK)
        init_ent.append(entropy_from_vals(vals))
        init_spr.append(spread_from_ids(ids, W))
    feat["topk_init_entropy_mean"] = float(np.mean(init_ent)) if init_ent else np.nan
    feat["topk_init_spread_mean"] = float(np.mean(init_spr)) if init_spr else np.nan

    # Simple binding/priority probes through prompt variants.
    # We don't run extra model calls here; use final hidden/logit directional proxies.
    feat["binding_priority_proxy_CminusE"] = final_margin
    feat["binding_priority_proxy_OTHERmax"] = O_score - max(C_score, E_score)

    return feat


# -----------------------------
# Evaluation
# -----------------------------

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


def cols_containing(df, patterns):
    out = []
    for c in df.columns:
        lc = c.lower()
        if any(p.lower() in lc for p in patterns):
            out.append(c)
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
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_SEED,
        )),
    ])


def make_multiclass_classifier():
    base = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_SEED,
    )
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
        return {"valid": False, "reason": "missing_target_or_features"}
    d = df.dropna(subset=[target]).copy()
    if len(d) < 20:
        return {"valid": False, "reason": "too_few_rows", "n": len(d)}
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
        return {"valid": False, "reason": "no_valid_folds", "n": len(d)}

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


def eval_multiclass(df, cols, target="mechanism", group_col="graph_id"):
    cols = numeric_cols(df, cols)
    if target not in df.columns or not cols:
        return {"valid": False, "reason": "missing_target_or_features"}
    d = df.dropna(subset=[target]).copy()
    if len(d) < 20:
        return {"valid": False, "reason": "too_few_rows", "n": len(d)}
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
        return {"valid": False, "reason": "no_valid_folds", "n": len(d)}

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


def eval_regression(df, cols, target="commit_abs_final_margin", group_col="graph_id"):
    cols = numeric_cols(df, cols)
    if target not in df.columns or not cols:
        return {"valid": False, "reason": "missing_target_or_features"}
    d = df.dropna(subset=[target]).copy()
    if len(d) < 20:
        return {"valid": False, "reason": "too_few_rows", "n": len(d)}
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
        return {"valid": False, "reason": "no_valid_folds", "n": len(d)}

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


def build_feature_groups(df):
    topk = cols_by_prefix(df, ["topk_"])
    answer = numeric_cols(df, [
        "score_C", "score_E", "score_OTHER",
        "final_margin_C_minus_E",
        "commit_abs_final_margin",
        "commit_high_abs_margin",
        "binding_priority_proxy_CminusE",
        "binding_priority_proxy_OTHERmax",
    ]) + cols_by_prefix(df, ["ans_margin_"])

    dyn = cols_by_prefix(df, ["dyn_", "dyn2_", "dyn4_"])
    binding = numeric_cols(df, [
        "binding_priority_proxy_CminusE",
        "binding_priority_proxy_OTHERmax",
    ])

    # no answer readout excludes direct scores/final margin, but keeps TopK and dyn4.
    cft_no_answer = unique_keep_order(topk + [c for c in dyn if c not in answer])
    cft_with_answer = unique_keep_order(cft_no_answer + answer)
    topk_plus_dyn = unique_keep_order(topk + [c for c in dyn if c not in answer])

    groups = {
        "TopK": topk,
        "AnswerReadout": answer,
        "Dynamics": [c for c in dyn if c not in answer],
        "BindingPriority": binding,
        "TopK_plus_Dynamics": topk_plus_dyn,
        "CFT3_NoAnswer": cft_no_answer,
        "CFT3_WithAnswer": cft_with_answer,
    }
    return {k: numeric_cols(df, v) for k, v in groups.items() if len(numeric_cols(df, v)) > 0}


def evaluate_feature_groups(df, groups):
    rows = []
    for name, cols in groups.items():
        gen = eval_binary(df, cols, target="gen_error", group_col="graph_id")
        mech = eval_multiclass(df, cols, target="mechanism", group_col="graph_id")
        subm = eval_multiclass(df, cols, target="submechanism", group_col="graph_id")
        commit = eval_regression(df, cols, target="commit_abs_final_margin", group_col="graph_id")

        rows.append({
            "feature_group": name,
            "n_features": len(cols),
            "gen_auc": gen.get("auc", np.nan),
            "gen_f1": gen.get("f1", np.nan),
            "gen_acc": gen.get("acc", np.nan),
            "mechanism_macro_f1": mech.get("macro_f1", np.nan),
            "submechanism_macro_f1": subm.get("macro_f1", np.nan),
            "commit_abs_r2": commit.get("r2", np.nan),
            "commit_abs_corr": commit.get("corr", np.nan),
            "valid_gen": gen.get("valid", False),
            "valid_mech": mech.get("valid", False),
            "valid_submech": subm.get("valid", False),
            "valid_commit": commit.get("valid", False),
        })
    return pd.DataFrame(rows).sort_values(
        ["gen_auc", "submechanism_macro_f1", "commit_abs_r2"],
        ascending=False,
        na_position="last",
    )


def get_metric(summary, group, key):
    r = summary[summary["feature_group"] == group]
    if len(r) == 0:
        return np.nan
    return safe_float(r.iloc[0].get(key, np.nan))


def verdict_for_model(summary):
    topk_gen = get_metric(summary, "TopK", "gen_auc")
    cft_no_gen = get_metric(summary, "CFT3_NoAnswer", "gen_auc")
    cft_with_gen = get_metric(summary, "CFT3_WithAnswer", "gen_auc")
    answer_gen = get_metric(summary, "AnswerReadout", "gen_auc")
    topk_sub = get_metric(summary, "TopK", "submechanism_macro_f1")
    cft_sub = get_metric(summary, "CFT3_NoAnswer", "submechanism_macro_f1")
    topk_commit = get_metric(summary, "TopK", "commit_abs_r2")
    cft_commit = get_metric(summary, "CFT3_NoAnswer", "commit_abs_r2")
    ans_commit = get_metric(summary, "AnswerReadout", "commit_abs_r2")

    reasons = []
    verdict = "UNDETERMINED"

    if not np.isfinite(topk_gen) or not np.isfinite(cft_no_gen):
        verdict = "FAIL_EVAL_INVALID"
        reasons.append("TopK or CFT3_NoAnswer not evaluable.")
    elif cft_no_gen > topk_gen + 0.05 and cft_sub > topk_sub + 0.02 and cft_commit > max(0.50, topk_commit + 0.20):
        verdict = "PASS_CFT3_MODEL_STRONG"
        reasons.append("CFT3 no-answer proxy improves GenError, submechanism, and commitment over TopK.")
    elif cft_no_gen > topk_gen + 0.03:
        verdict = "PASS_CFT3_MODEL_GEN_GAIN"
        reasons.append("CFT3 no-answer proxy improves GenError over TopK.")
    elif cft_no_gen > topk_gen + 0.01:
        verdict = "PARTIAL_CFT3_MODEL_SMALL_GAIN"
        reasons.append("CFT3 no-answer proxy gives small GenError gain.")
    else:
        verdict = "FAIL_CFT3_MODEL_NO_GAIN"
        reasons.append("CFT3 no-answer proxy does not improve over TopK.")

    return {
        "verdict": verdict,
        "reasons": reasons,
        "metrics": {
            "topk_gen_auc": topk_gen,
            "cft3_noanswer_gen_auc": cft_no_gen,
            "cft3_withanswer_gen_auc": cft_with_gen,
            "answer_gen_auc": answer_gen,
            "topk_submechanism_macro_f1": topk_sub,
            "cft3_noanswer_submechanism_macro_f1": cft_sub,
            "topk_commit_r2": topk_commit,
            "cft3_noanswer_commit_r2": cft_commit,
            "answer_commit_r2": ans_commit,
        }
    }


def run_one_model(spec):
    model_key = spec["model_key"]
    model_name = spec["model_name"]
    path = spec["path"]

    model_out = OUT_DIR / model_key
    model_out.mkdir(parents=True, exist_ok=True)

    info = {
        "model_key": model_key,
        "model_name": model_name,
        "path": path,
        "exists": os.path.exists(path),
        "status": "pending",
    }

    if not os.path.exists(path):
        info["status"] = "missing_path"
        info["error"] = f"Path not found: {path}"
        return info

    dataset = build_dataset()
    dataset.to_csv(model_out / f"cft3_{model_key}_dataset.csv", index=False, encoding="utf-8-sig")

    try:
        tokenizer, model = None, None
        model, tokenizer = load_model_and_tokenizer(path, trust_remote_code=spec.get("trust_remote_code", True))
        W = lm_head_weight(model)
        choice_ids = get_choice_token_ids(tokenizer)

        rows = []
        errors = []
        t0 = time.time()

        for i, row in dataset.iterrows():
            if i % 20 == 0:
                print(f"[{model_key}] extracting row {i}/{len(dataset)}", flush=True)
            try:
                feat = extract_features_for_row(row, model, tokenizer, W, choice_ids)
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
                    print(f"[warn][{model_key}] row {row['row_id']} failed: {type(e).__name__}: {e}")

        feat_df = pd.DataFrame(rows)
        err_df = pd.DataFrame(errors)
        feat_df.to_csv(model_out / f"cft3_{model_key}_features.csv", index=False, encoding="utf-8-sig")
        err_df.to_csv(model_out / f"cft3_{model_key}_errors.csv", index=False, encoding="utf-8-sig")

        if feat_df.empty:
            info["status"] = "feature_empty"
            info["n_errors"] = len(errors)
            return info

        groups = build_feature_groups(feat_df)
        summary = evaluate_feature_groups(feat_df, groups)
        summary.to_csv(model_out / f"cft3_{model_key}_model_summary.csv", index=False, encoding="utf-8-sig")

        verdict = verdict_for_model(summary)
        verdict.update({
            "model_key": model_key,
            "model_name": model_name,
            "n_rows": int(len(feat_df)),
            "n_graphs": int(feat_df["graph_id"].nunique()) if "graph_id" in feat_df.columns else None,
            "n_errors": int(len(errors)),
            "runtime_sec": time.time() - t0,
            "feature_set_sizes": {k: len(v) for k, v in groups.items()},
            "top_rows": summary.head(12).to_dict(orient="records"),
        })
        with open(model_out / f"cft3_{model_key}_verdict.json", "w", encoding="utf-8") as f:
            json.dump(verdict, f, ensure_ascii=False, indent=2)

        info.update(verdict)
        info["status"] = "done"

    except Exception as e:
        info["status"] = "failed"
        info["error_type"] = type(e).__name__
        info["error"] = str(e)
        info["traceback"] = traceback.format_exc()

    finally:
        try:
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass

    return info


def main():
    all_infos = []
    for spec in MODEL_SPECS:
        print("=" * 80)
        print(f"[CFT-3] Running model: {spec['model_key']} | {spec['path']}")
        info = run_one_model(spec)
        all_infos.append(info)
        print(json.dumps({
            "model_key": info.get("model_key"),
            "status": info.get("status"),
            "verdict": info.get("verdict"),
            "metrics": info.get("metrics"),
            "error": info.get("error"),
        }, ensure_ascii=False, indent=2))

    # Cross-model summary
    rows = []
    for info in all_infos:
        metrics = info.get("metrics", {}) or {}
        rows.append({
            "model_key": info.get("model_key"),
            "model_name": info.get("model_name"),
            "path": info.get("path"),
            "exists": info.get("exists"),
            "status": info.get("status"),
            "verdict": info.get("verdict"),
            "topk_gen_auc": metrics.get("topk_gen_auc"),
            "cft3_noanswer_gen_auc": metrics.get("cft3_noanswer_gen_auc"),
            "cft3_gain_over_topk": safe_float(metrics.get("cft3_noanswer_gen_auc")) - safe_float(metrics.get("topk_gen_auc")),
            "answer_gen_auc": metrics.get("answer_gen_auc"),
            "topk_submechanism_macro_f1": metrics.get("topk_submechanism_macro_f1"),
            "cft3_noanswer_submechanism_macro_f1": metrics.get("cft3_noanswer_submechanism_macro_f1"),
            "topk_commit_r2": metrics.get("topk_commit_r2"),
            "cft3_noanswer_commit_r2": metrics.get("cft3_noanswer_commit_r2"),
            "answer_commit_r2": metrics.get("answer_commit_r2"),
            "n_rows": info.get("n_rows"),
            "n_errors": info.get("n_errors"),
            "runtime_sec": info.get("runtime_sec"),
            "error": info.get("error"),
        })

    cross = pd.DataFrame(rows)
    cross.to_csv(OUT_DIR / "cft3_cross_model_summary.csv", index=False, encoding="utf-8-sig")

    done = [r for r in rows if r.get("status") == "done"]
    pass_count = sum(str(r.get("verdict", "")).startswith("PASS") for r in done)

    if len(done) == 0:
        global_verdict = "FAIL_NO_MODELS_RAN"
    elif pass_count == len(done) and len(done) >= 2:
        global_verdict = "PASS_CFT3_CROSS_MODEL"
    elif pass_count >= 1 and len(done) >= 2:
        global_verdict = "PARTIAL_CFT3_CROSS_MODEL_MIXED"
    elif pass_count == 1:
        global_verdict = "PARTIAL_CFT3_SINGLE_MODEL_PASS"
    else:
        global_verdict = "FAIL_CFT3_NO_MODEL_PASS"

    global_obj = {
        "verdict": global_verdict,
        "n_models_done": len(done),
        "n_models_pass": pass_count,
        "models": all_infos,
        "summary_csv": str(OUT_DIR / "cft3_cross_model_summary.csv"),
    }
    with open(OUT_DIR / "cft3_verdict.json", "w", encoding="utf-8") as f:
        json.dump(global_obj, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(json.dumps(global_obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
