# -*- coding: utf-8 -*-
r"""
CFT-3B: Model-Specific Window-Aligned Constraint Field Proxy Audit

Purpose
-------
CFT-3 showed mixed-positive cross-model results:

  Qwen  : PASS-Strong
  Llama : small positive gain
  Gemma : positive GenError gain

But CFT-3 used a generic relative decision window for all models.
CM-4D showed that initialization / mechanism / downstream coupling windows are model-specific.

CFT-3B tests whether model-specific window alignment strengthens CFT3_NoAnswer.

Model-specific window priors from CM-4D:
  Qwen:
    mechanism/downstream peak ≈ L5-L6
    topology stabilization ≈ L9-L11
    decision window ≈ L20-L25

  Llama:
    topology peak ≈ L0-L1
    mechanism peak ≈ L2-L5
    downstream peak ≈ L0-L2
    decision window is model-depth specific; use relative late band

  Gemma:
    topology/downstream peak ≈ L0-L1
    mechanism peak ≈ L7-L8
    decision window ≈ L18-L23

This script is self-contained and reruns the three models with a smaller default dataset than full CFT-3.
It extracts features in aligned windows:
  - topk_topology_window
  - mechanism_seed_window
  - downstream_seed_window
  - decision_window
  - early/late decision dynamics

Outputs
-------
C:\Users\ZH\Desktop\AGI\outputs\cft3b_outputs\
  cft3b_cross_model_summary.csv
  cft3b_verdict.json
  <model>\cft3b_<model>_features.csv
  <model>\cft3b_<model>_model_summary.csv
  <model>\cft3b_<model>_verdict.json
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

MODEL_SPECS = [
    {
        "model_key": "qwen",
        "model_name": "Qwen2.5-1.5B-Instruct",
        "path": r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main",
        "trust_remote_code": True,
        "window_mode": "qwen",
    },
    {
        "model_key": "llama",
        "model_name": "Llama-3.2-1B-Instruct",
        "path": r"D:\model\Llama-3.2-1B-Instruct",
        "trust_remote_code": True,
        "window_mode": "llama",
    },
    {
        "model_key": "gemma",
        "model_name": "gemma-2-2b-it",
        "path": r"D:\model\gemma-2-2b-it",
        "trust_remote_code": True,
        "window_mode": "gemma",
    },
]

OUT_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\cft3b_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Use CFT-3 scale by default. Reduce for smoke if needed.
N_GRAPHS = 24
N_SURFACES = 3
TOPK = 80
N_SPLITS = 4


# -----------------------------
# Dataset
# -----------------------------

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
                    facts = f"{A} belongs to {B}.\n{B} belongs to {C}."
                    mechanism, submechanism, gold = "stable", "stable_core", "C"

                elif cond == "stable_redundant":
                    facts = f"{A} belongs to {B}.\n{B} belongs to {C}.\nNote: {C} is the final group in this example."
                    mechanism, submechanism, gold = "stable", "stable_redundant", "C"

                elif cond == "competition_equal":
                    facts = f"{A} belongs to {B}.\n{B} may belong to {C}.\n{B} may belong to {E}.\nBoth options are equally supported."
                    mechanism, submechanism, gold = "competition", "competition_equal", "OTHER"

                elif cond == "source_trusted_C":
                    facts = f"Source 1 says: {B} belongs to {E}.\nTrusted source says: {B} belongs to {C}.\n{A} belongs to {B}.\nUse the trusted source."
                    mechanism, submechanism, gold = "competition", "competition_source_C", "C"

                elif cond == "source_trusted_E":
                    facts = f"Source 1 says: {B} belongs to {C}.\nTrusted source says: {B} belongs to {E}.\n{A} belongs to {B}.\nUse the trusted source."
                    mechanism, submechanism, gold = "competition", "competition_source_E", "E"

                elif cond == "closure_update_C":
                    facts = f"Old rule: {B} belongs to {E}.\nNew rule: {B} belongs to {C}.\n{A} belongs to {B}.\nUse the new rule."
                    mechanism, submechanism, gold = "closure", "closure_update_C", "C"

                elif cond == "closure_update_E":
                    facts = f"Old rule: {B} belongs to {C}.\nNew rule: {B} belongs to {E}.\n{A} belongs to {B}.\nUse the new rule."
                    mechanism, submechanism, gold = "closure", "closure_update_E", "E"

                elif cond == "logic_contradiction":
                    facts = f"Rule: {A} cannot belong to both {C} and {E}.\nClaim 1: {A} belongs to {C}.\nClaim 2: {A} belongs to {E}.\nIf the facts contradict, answer OTHER."
                    mechanism, submechanism, gold = "logic", "logic_direct", "OTHER"

                elif cond == "random_broken":
                    Z = GROUPS_E[(gi + 7) % len(GROUPS_E)]
                    facts = f"{A} belongs to {B}.\n{B} is mentioned near {Z}.\nNo final membership relation is provided."
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
                    "C": C,
                    "E": E,
                    "prompt": prompt,
                })

    return pd.DataFrame(rows)


# -----------------------------
# Helpers
# -----------------------------

def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


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


def lm_head_weight(model):
    return model.get_output_embeddings().weight.detach().float().cpu().numpy()


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


def get_choice_token_ids(tokenizer):
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
            if len(enc):
                ids.append(enc[-1])
        out[label] = sorted(set(ids))
    return out


@torch.no_grad()
def forward_hidden_logits(model, tokenizer, text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
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
        # CM-4D: mechanism/downstream L5-L6, topology L9-L11, decision L20-L25
        topo = contiguous(9, 11, n_hidden_states) if n_layers >= 12 else contiguous(rel(.30), rel(.40), n_hidden_states)
        mech = contiguous(5, 6, n_hidden_states) if n_layers >= 7 else contiguous(rel(.15), rel(.25), n_hidden_states)
        down = contiguous(5, 6, n_hidden_states) if n_layers >= 7 else contiguous(rel(.15), rel(.25), n_hidden_states)
        decision = contiguous(20, 25, n_hidden_states) if n_layers >= 26 else contiguous(rel(.70), rel(.92), n_hidden_states)

    elif model_key == "llama":
        # CM-4D: topology L0-L1, mechanism L2-L5, downstream L0-L2
        topo = contiguous(0, 1, n_hidden_states)
        mech = contiguous(2, 5, n_hidden_states) if n_layers >= 6 else contiguous(rel(.15), rel(.35), n_hidden_states)
        down = contiguous(0, 2, n_hidden_states)
        decision = contiguous(rel(.68), rel(.92), n_hidden_states)

    elif model_key == "gemma":
        # CM-4D: topology/downstream L0-L1, mechanism L7-L8, decision L18-L23
        topo = contiguous(0, 1, n_hidden_states)
        mech = contiguous(7, 8, n_hidden_states) if n_layers >= 9 else contiguous(rel(.30), rel(.45), n_hidden_states)
        down = contiguous(0, 1, n_hidden_states)
        decision = contiguous(18, 23, n_hidden_states) if n_layers >= 24 else contiguous(rel(.68), rel(.92), n_hidden_states)

    else:
        topo = contiguous(rel(.05), rel(.15), n_hidden_states)
        mech = contiguous(rel(.15), rel(.30), n_hidden_states)
        down = contiguous(rel(.05), rel(.20), n_hidden_states)
        decision = contiguous(rel(.70), rel(.92), n_hidden_states)

    # split decision into early/late
    if len(decision) >= 4:
        mid = len(decision) // 2
        dec_early = decision[:mid]
        dec_late = decision[mid:]
    else:
        dec_early = decision
        dec_late = decision

    return {
        "topology": sorted(set(topo)),
        "mechanism_seed": sorted(set(mech)),
        "downstream_seed": sorted(set(down)),
        "decision": sorted(set(decision)),
        "decision_early": sorted(set(dec_early)),
        "decision_late": sorted(set(dec_late)),
    }


def topk_for_hidden(h, W, k=TOPK):
    logits = h @ W.T
    k_eff = min(k, len(logits))
    idx = np.argpartition(-logits, k_eff - 1)[:k_eff]
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
    ids = np.asarray(ids)
    if ids.size <= 1:
        return 0.0
    E = W[ids]
    c = E.mean(axis=0, keepdims=True)
    return float(np.mean(np.linalg.norm(E - c, axis=1)))


def center_vec(ids, vals, W):
    ids = np.asarray(ids)
    vals = np.asarray(vals, dtype=np.float64)
    if ids.size == 0:
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


def score_choices_from_logits(logits, choice_ids):
    scores = {}
    for label, ids in choice_ids.items():
        scores[label] = float(np.max(logits[ids])) if ids else -1e9
    pred = max(scores, key=scores.get)
    return pred, scores


def window_topk_features(feat, prefix, hs, hidxs, W):
    centers, entropies, spreads = [], [], []
    for local_i, hidx in enumerate(hidxs):
        ids, vals = topk_for_hidden(hs[hidx], W, k=TOPK)
        ent = entropy_from_vals(vals)
        spr = spread_from_ids(ids, W)
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


def window_margin_features(feat, prefix, hs, hidxs, W, choice_ids):
    margins = []
    other_gaps = []
    for local_i, hidx in enumerate(hidxs):
        logits = hs[hidx] @ W.T
        c = max([logits[i] for i in choice_ids["C"]]) if choice_ids["C"] else -1e9
        e = max([logits[i] for i in choice_ids["E"]]) if choice_ids["E"] else -1e9
        o = max([logits[i] for i in choice_ids["OTHER"]]) if choice_ids["OTHER"] else -1e9
        m = float(c - e)
        og = float(o - max(c, e))
        margins.append(m)
        other_gaps.append(og)
        feat[f"ans_margin_{prefix}_{local_i}"] = m
        feat[f"other_gap_{prefix}_{local_i}"] = og

    if margins:
        arr = np.asarray(margins, dtype=np.float64)
        abs_arr = np.abs(arr)
        feat[f"ans_margin_{prefix}_mean"] = float(np.mean(arr))
        feat[f"ans_margin_{prefix}_std"] = float(np.std(arr))
        feat[f"ans_margin_{prefix}_slope"] = slope(arr)
        feat[f"ans_margin_{prefix}_area"] = area(arr)
        feat[f"ans_margin_{prefix}_minabs"] = float(np.min(abs_arr))
        feat[f"commit_abs_{prefix}_area"] = area(abs_arr)
        feat[f"commit_abs_{prefix}_slope"] = slope(abs_arr)
        feat[f"commit_abs_{prefix}_mean"] = float(np.mean(abs_arr))
        feat[f"commit_abs_{prefix}_std"] = float(np.std(abs_arr))
        feat[f"margin_sign_changes_{prefix}"] = float(np.sum(np.sign(arr[:-1]) * np.sign(arr[1:]) < 0)) if arr.size > 1 else 0.0

        if arr.size >= 2:
            dm = np.diff(arr)
            feat[f"dyn_{prefix}_velocity_mean"] = float(np.mean(dm))
            feat[f"dyn_{prefix}_velocity_abs_mean"] = float(np.mean(np.abs(dm)))
            feat[f"dyn_{prefix}_velocity_std"] = float(np.std(dm))
            feat[f"dyn_{prefix}_velocity_area"] = area(dm)
            if dm.size >= 2:
                ddm = np.diff(dm)
                feat[f"dyn_{prefix}_curvature_abs_mean"] = float(np.mean(np.abs(ddm)))
                feat[f"dyn_{prefix}_accel_std"] = float(np.std(ddm))
            else:
                feat[f"dyn_{prefix}_curvature_abs_mean"] = 0.0
                feat[f"dyn_{prefix}_accel_std"] = 0.0
        else:
            feat[f"dyn_{prefix}_velocity_mean"] = 0.0
            feat[f"dyn_{prefix}_velocity_abs_mean"] = 0.0
            feat[f"dyn_{prefix}_velocity_std"] = 0.0
            feat[f"dyn_{prefix}_velocity_area"] = 0.0
            feat[f"dyn_{prefix}_curvature_abs_mean"] = 0.0
            feat[f"dyn_{prefix}_accel_std"] = 0.0

    if other_gaps:
        og = np.asarray(other_gaps, dtype=np.float64)
        feat[f"other_gap_{prefix}_mean"] = float(np.mean(og))
        feat[f"other_gap_{prefix}_slope"] = slope(og)
        feat[f"other_gap_{prefix}_area"] = area(og)


def extract_features_for_row(row, model, tokenizer, W, choice_ids, model_key):
    text = format_chat(tokenizer, row["prompt"])
    hs, final_logits = forward_hidden_logits(model, tokenizer, text)
    windows = aligned_windows(model_key, len(hs))

    pred, scores = score_choices_from_logits(final_logits, choice_ids)
    final_margin = scores["C"] - scores["E"]
    commit_abs = abs(final_margin)

    feat = {
        "answer_pred": pred,
        "score_C": scores["C"],
        "score_E": scores["E"],
        "score_OTHER": scores["OTHER"],
        "final_margin_C_minus_E": final_margin,
        "commit_abs_final_margin": commit_abs,
        "commit_high_abs_margin": int(commit_abs >= 1.0),
        "gen_correct": int(pred == row["gold"]),
        "gen_error": int(pred != row["gold"]),
        "n_hidden_states": len(hs),
        "n_layers": len(hs) - 1,
        "window_topology": ",".join(map(str, windows["topology"])),
        "window_mechanism_seed": ",".join(map(str, windows["mechanism_seed"])),
        "window_downstream_seed": ",".join(map(str, windows["downstream_seed"])),
        "window_decision": ",".join(map(str, windows["decision"])),
    }

    # TopK windows
    for pname, hidxs in [
        ("topology", windows["topology"]),
        ("mechanism_seed", windows["mechanism_seed"]),
        ("downstream_seed", windows["downstream_seed"]),
        ("decision", windows["decision"]),
    ]:
        window_topk_features(feat, pname, hs, hidxs, W)

    # Margin/dynamics windows
    for pname, hidxs in [
        ("mechanism_seed", windows["mechanism_seed"]),
        ("downstream_seed", windows["downstream_seed"]),
        ("decision", windows["decision"]),
        ("decision_early", windows["decision_early"]),
        ("decision_late", windows["decision_late"]),
    ]:
        window_margin_features(feat, pname, hs, hidxs, W, choice_ids)

    # DYN4-style aligned composite
    dec_abs_mean = feat.get("commit_abs_decision_mean", np.nan)
    late_abs_mean = feat.get("commit_abs_decision_late_mean", np.nan)
    early_abs_mean = feat.get("commit_abs_decision_early_mean", np.nan)
    feat["dyn4_aligned_late_minus_early_abs"] = safe_float(late_abs_mean) - safe_float(early_abs_mean)
    feat["dyn4_aligned_late_over_early_abs"] = safe_float(late_abs_mean) / (safe_float(early_abs_mean) + 1e-6)
    feat["dyn4_aligned_boundary_residence_inv"] = 1.0 / (safe_float(feat.get("ans_margin_decision_minabs", np.nan)) + 1e-6)

    # Binding/priority lightweight proxies
    feat["binding_priority_proxy_CminusE"] = final_margin
    feat["binding_priority_proxy_OTHERmax"] = scores["OTHER"] - max(scores["C"], scores["E"])

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
        return {"valid": False}
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
        return {"valid": False}
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
        return {"valid": False}
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
    topk_topology = cols_by_prefix(df, ["topk_topology_"])
    topk_mech = cols_by_prefix(df, ["topk_mechanism_seed_"])
    topk_down = cols_by_prefix(df, ["topk_downstream_seed_"])
    topk_decision = cols_by_prefix(df, ["topk_decision_"])
    topk_all = unique_keep_order(topk_topology + topk_mech + topk_down + topk_decision)

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

    binding = numeric_cols(df, ["binding_priority_proxy_CminusE", "binding_priority_proxy_OTHERmax"])

    cft_no_answer = unique_keep_order(topk_all + dynamics)
    cft_with_answer = unique_keep_order(cft_no_answer + answer)

    groups = {
        "TopK_topology": topk_topology,
        "TopK_mechanism_seed": topk_mech,
        "TopK_downstream_seed": topk_down,
        "TopK_decision": topk_decision,
        "TopK_all_aligned": topk_all,
        "AnswerReadout": answer,
        "Dynamics_aligned": dynamics,
        "BindingPriority": binding,
        "TopK_plus_Dynamics_aligned": unique_keep_order(topk_all + dynamics),
        "CFT3B_NoAnswer": cft_no_answer,
        "CFT3B_WithAnswer": cft_with_answer,
    }
    return {k: numeric_cols(df, v) for k, v in groups.items() if len(numeric_cols(df, v)) > 0}


def evaluate_groups(df, groups):
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
    topk = get_metric(summary, "TopK_all_aligned", "gen_auc")
    cft = get_metric(summary, "CFT3B_NoAnswer", "gen_auc")
    cft_ans = get_metric(summary, "CFT3B_WithAnswer", "gen_auc")
    ans = get_metric(summary, "AnswerReadout", "gen_auc")

    topk_sub = get_metric(summary, "TopK_all_aligned", "submechanism_macro_f1")
    cft_sub = get_metric(summary, "CFT3B_NoAnswer", "submechanism_macro_f1")

    topk_r2 = get_metric(summary, "TopK_all_aligned", "commit_abs_r2")
    cft_r2 = get_metric(summary, "CFT3B_NoAnswer", "commit_abs_r2")
    ans_r2 = get_metric(summary, "AnswerReadout", "commit_abs_r2")

    reasons = []
    if not np.isfinite(topk) or not np.isfinite(cft):
        verdict = "FAIL_EVAL_INVALID"
        reasons.append("TopK_all_aligned or CFT3B_NoAnswer not evaluable.")
    elif cft > topk + 0.05 and cft_sub > topk_sub + 0.02 and cft_r2 > max(0.5, topk_r2 + 0.2):
        verdict = "PASS_CFT3B_MODEL_STRONG"
        reasons.append("Window-aligned no-answer proxy improves GenError, submechanism, and commitment over aligned TopK.")
    elif cft > topk + 0.03:
        verdict = "PASS_CFT3B_MODEL_GEN_GAIN"
        reasons.append("Window-aligned no-answer proxy improves GenError over aligned TopK.")
    elif cft > topk + 0.01:
        verdict = "PARTIAL_CFT3B_MODEL_SMALL_GAIN"
        reasons.append("Window-aligned no-answer proxy gives small GenError gain.")
    else:
        verdict = "FAIL_CFT3B_MODEL_NO_GAIN"
        reasons.append("Window-aligned no-answer proxy does not improve over aligned TopK.")

    return {
        "verdict": verdict,
        "reasons": reasons,
        "metrics": {
            "topk_aligned_gen_auc": topk,
            "cft3b_noanswer_gen_auc": cft,
            "cft3b_withanswer_gen_auc": cft_ans,
            "answer_gen_auc": ans,
            "topk_aligned_submechanism_macro_f1": topk_sub,
            "cft3b_noanswer_submechanism_macro_f1": cft_sub,
            "topk_aligned_commit_r2": topk_r2,
            "cft3b_noanswer_commit_r2": cft_r2,
            "answer_commit_r2": ans_r2,
        }
    }


def run_one_model(spec):
    model_key = spec["model_key"]
    path = spec["path"]
    model_out = OUT_DIR / model_key
    model_out.mkdir(parents=True, exist_ok=True)

    info = {
        "model_key": model_key,
        "model_name": spec["model_name"],
        "path": path,
        "exists": os.path.exists(path),
        "status": "pending",
    }

    if not os.path.exists(path):
        info.update({"status": "missing_path", "error": f"Path not found: {path}"})
        return info

    dataset = build_dataset()
    dataset.to_csv(model_out / f"cft3b_{model_key}_dataset.csv", index=False, encoding="utf-8-sig")

    model = None
    try:
        t0 = time.time()
        model, tokenizer = load_model_and_tokenizer(path, trust_remote_code=spec.get("trust_remote_code", True))
        W = lm_head_weight(model)
        choice_ids = get_choice_token_ids(tokenizer)

        rows, errors = [], []
        for i, row in dataset.iterrows():
            if i % 20 == 0:
                print(f"[{model_key}] extracting row {i}/{len(dataset)}", flush=True)
            try:
                feat = extract_features_for_row(row, model, tokenizer, W, choice_ids, model_key)
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
        feat_df.to_csv(model_out / f"cft3b_{model_key}_features.csv", index=False, encoding="utf-8-sig")
        err_df.to_csv(model_out / f"cft3b_{model_key}_errors.csv", index=False, encoding="utf-8-sig")

        if feat_df.empty:
            info.update({"status": "feature_empty", "n_errors": len(errors)})
            return info

        groups = build_feature_groups(feat_df)
        summary = evaluate_groups(feat_df, groups)
        summary.to_csv(model_out / f"cft3b_{model_key}_model_summary.csv", index=False, encoding="utf-8-sig")

        verdict = verdict_for_model(summary)
        verdict.update({
            "model_key": model_key,
            "model_name": spec["model_name"],
            "n_rows": int(len(feat_df)),
            "n_graphs": int(feat_df["graph_id"].nunique()) if "graph_id" in feat_df.columns else None,
            "n_errors": int(len(errors)),
            "runtime_sec": time.time() - t0,
            "feature_set_sizes": {k: len(v) for k, v in groups.items()},
            "top_rows": summary.head(12).to_dict(orient="records"),
        })

        with open(model_out / f"cft3b_{model_key}_verdict.json", "w", encoding="utf-8") as f:
            json.dump(verdict, f, ensure_ascii=False, indent=2)

        info.update(verdict)
        info["status"] = "done"

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

    for spec in MODEL_SPECS:
        print("=" * 80)
        print(f"[CFT-3B] Running model: {spec['model_key']} | {spec['path']}")
        info = run_one_model(spec)
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
            "model_name": info.get("model_name"),
            "path": info.get("path"),
            "exists": info.get("exists"),
            "status": info.get("status"),
            "verdict": info.get("verdict"),
            "topk_aligned_gen_auc": m.get("topk_aligned_gen_auc"),
            "cft3b_noanswer_gen_auc": m.get("cft3b_noanswer_gen_auc"),
            "cft3b_gain_over_topk": safe_float(m.get("cft3b_noanswer_gen_auc")) - safe_float(m.get("topk_aligned_gen_auc")),
            "answer_gen_auc": m.get("answer_gen_auc"),
            "topk_aligned_submechanism_macro_f1": m.get("topk_aligned_submechanism_macro_f1"),
            "cft3b_noanswer_submechanism_macro_f1": m.get("cft3b_noanswer_submechanism_macro_f1"),
            "topk_aligned_commit_r2": m.get("topk_aligned_commit_r2"),
            "cft3b_noanswer_commit_r2": m.get("cft3b_noanswer_commit_r2"),
            "answer_commit_r2": m.get("answer_commit_r2"),
            "n_rows": info.get("n_rows"),
            "n_errors": info.get("n_errors"),
            "runtime_sec": info.get("runtime_sec"),
            "error": info.get("error"),
        })

    cross = pd.DataFrame(rows)
    cross.to_csv(OUT_DIR / "cft3b_cross_model_summary.csv", index=False, encoding="utf-8-sig")

    done = [r for r in rows if r.get("status") == "done"]
    pass_count = sum(str(r.get("verdict", "")).startswith("PASS") for r in done)

    if len(done) == 0:
        global_verdict = "FAIL_NO_MODELS_RAN"
    elif pass_count == len(done) and len(done) >= 2:
        global_verdict = "PASS_CFT3B_CROSS_MODEL"
    elif pass_count >= 1 and len(done) >= 2:
        global_verdict = "PARTIAL_CFT3B_CROSS_MODEL_MIXED"
    elif pass_count == 1:
        global_verdict = "PARTIAL_CFT3B_SINGLE_MODEL_PASS"
    else:
        global_verdict = "FAIL_CFT3B_NO_MODEL_PASS"

    global_obj = {
        "verdict": global_verdict,
        "n_models_done": len(done),
        "n_models_pass": pass_count,
        "models": all_infos,
        "summary_csv": str(OUT_DIR / "cft3b_cross_model_summary.csv"),
    }

    with open(OUT_DIR / "cft3b_verdict.json", "w", encoding="utf-8") as f:
        json.dump(global_obj, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(json.dumps(global_obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
