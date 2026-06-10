# -*- coding: utf-8 -*-
r"""
PSA-1: Possibility Space / Cbit Audit

Goal
----
Measure whether an LLM maintains an effective possibility space during inference,
and quantify how many Cbits are gained as the model compresses candidate possibilities.

Core definitions
----------------
Given a candidate set Ω = {s_i} and per-layer scores z_i(l):

    p_i(l) = softmax(z_i(l))
    H(l)   = -Σ p_i(l) log2 p_i(l)
    N_eff(l) = 2^H(l)

Cbit gain from layer a to b:

    ΔCbit(a→b) = H(a) - H(b)
               = log2(N_eff(a) / N_eff(b))

Interpretation
--------------
If N_eff is high in early/middle layers and low near output, the model is
compressing a possibility space.

If multiple candidate structures are distinguishable, compete, and can collapse
to different final answers, then the possibility space has operational existence.

This script measures three projections:

1. Answer possibility space:
   Ω_answer = {C, E, OTHER}

2. Sequence-label possibility space:
   Ω_seq = {C, E, OTHER} using short sequence log-likelihood.

3. TopK/VIM local neighborhood volume:
   TopK entropy/spread/center drift as a proxy for local effective volume.

Hardcoded local model paths
---------------------------
Qwen:
  D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main

Llama:
  D:\model\Llama-3.2-1B-Instruct

Gemma:
  D:\model\gemma-2-2b-it

Default only runs Qwen for speed.
Set MODELS_TO_RUN = ["qwen", "llama", "gemma"] for cross-model audit.

Outputs
-------
C:\Users\ZH\Desktop\AGI\outputs\psa1_outputs\
  psa1_<model>_features.csv
  psa1_<model>_layer_summary.csv
  psa1_<model>_condition_summary.csv
  psa1_<model>_verdict.json
  psa1_cross_model_summary.csv
  psa1_verdict.json
"""

import os
import json
import math
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore")

# -----------------------------
# Configuration
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

# Start with qwen. Expand after smoke test.
MODELS_TO_RUN = ["qwen"]

OUT_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\psa1_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

RANDOM_SEED = 42
N_GRAPHS = 16
N_SURFACES = 2
TOPK = 80
MAX_LEN = 512

# Candidate labels. Keep short and robust.
CANDIDATES = ["C", "E", "OTHER"]


# -----------------------------
# Dataset
# -----------------------------

ENTITIES_A = [
    "mira", "nolan", "sora", "tavin", "lena", "orion", "vexa", "darin",
    "selka", "brin", "kael", "nira", "faro", "yuna", "pavo", "lumi",
]

GROUPS_B = [
    "amber guild", "blue circle", "crimson order", "delta clan",
    "ember house", "frost league", "green lodge", "harbor unit",
    "iron band", "jade group", "kepler sect", "lunar team",
    "marble council", "north ring", "opal class", "prime cohort",
]

GROUPS_C = [
    "atlas zone", "boreal zone", "cobalt zone", "dawn zone",
    "equinox zone", "falcon zone", "garden zone", "helios zone",
    "ivory zone", "juniper zone", "kestrel zone", "lotus zone",
    "meteor zone", "nova zone", "onyx zone", "prairie zone",
]

GROUPS_E = [
    "arcadia zone", "bright zone", "cinder zone", "dusk zone",
    "elm zone", "fern zone", "granite zone", "hazel zone",
    "indigo zone", "jasmine zone", "kite zone", "lagoon zone",
    "meadow zone", "nebula zone", "oasis zone", "polar zone",
]

SURFACES = [
    "Use the facts below and answer with only C, E, or OTHER.\nFacts:\n{facts}\nQuestion: Which final group does {A} belong to?\nAnswer:",
    "Read the mini knowledge base. Reply only with C, E, or OTHER.\n{facts}\nQuery: final membership of {A}?\nAnswer:",
]

CONDITIONS = [
    "stable_clean",
    "weak_distractor",
    "ambiguous_branch",
    "strong_branch_conflict",
    "closure_negation_conflict",
    "closure_update_conflict",
    "source_trusted_C",
    "source_trusted_E",
    "random_broken",
]


def build_dataset():
    rows = []
    for gi in range(N_GRAPHS):
        A = ENTITIES_A[gi % len(ENTITIES_A)]
        B = GROUPS_B[gi % len(GROUPS_B)]
        C = GROUPS_C[gi % len(GROUPS_C)]
        E = GROUPS_E[gi % len(GROUPS_E)]
        graph_id = f"g{gi:03d}"

        for si in range(N_SURFACES):
            template = SURFACES[si % len(SURFACES)]
            surface_id = f"surface_{si}"

            for cond in CONDITIONS:
                if cond == "stable_clean":
                    facts = f"{A} belongs to {B}.\n{B} belongs to C.\nC means {C}."
                    gold = "C"
                    mechanism = "stable"

                elif cond == "weak_distractor":
                    facts = f"{A} belongs to {B}.\n{B} belongs to C.\nC means {C}.\nA different group E means {E}, but no relation points there."
                    gold = "C"
                    mechanism = "stable_shift"

                elif cond == "ambiguous_branch":
                    facts = f"{A} belongs to {B}.\n{B} may belong to C.\n{B} may belong to E.\nC means {C}. E means {E}. Both branches are equally supported."
                    gold = "OTHER"
                    mechanism = "competition"

                elif cond == "strong_branch_conflict":
                    facts = f"{A} belongs to {B}.\nFirst note: {B} belongs to C.\nSecond note: {B} belongs to E.\nC means {C}. E means {E}. The second note is more recent."
                    gold = "E"
                    mechanism = "competition_conflict"

                elif cond == "closure_negation_conflict":
                    facts = f"{A} belongs to {B}.\nEarlier: {B} belongs to C.\nCorrection: {B} does not belong to C; {B} belongs to E.\nC means {C}. E means {E}."
                    gold = "E"
                    mechanism = "closure"

                elif cond == "closure_update_conflict":
                    facts = f"Old rule: {B} belongs to C.\nNew rule: {B} belongs to E.\n{A} belongs to {B}.\nC means {C}. E means {E}. Use the new rule."
                    gold = "E"
                    mechanism = "closure"

                elif cond == "source_trusted_C":
                    facts = f"Source 1 says: {B} belongs to E.\nTrusted source says: {B} belongs to C.\n{A} belongs to {B}.\nC means {C}. E means {E}. Use the trusted source."
                    gold = "C"
                    mechanism = "source_binding"

                elif cond == "source_trusted_E":
                    facts = f"Source 1 says: {B} belongs to C.\nTrusted source says: {B} belongs to E.\n{A} belongs to {B}.\nC means {C}. E means {E}. Use the trusted source."
                    gold = "E"
                    mechanism = "source_binding"

                elif cond == "random_broken":
                    facts = f"{A} belongs to {B}.\nC means {C}. E means {E}.\nNo final membership relation from {B} to C or E is given."
                    gold = "OTHER"
                    mechanism = "random_broken"

                prompt = template.format(facts=facts, A=A)
                rows.append({
                    "row_id": len(rows),
                    "graph_id": graph_id,
                    "surface_id": surface_id,
                    "condition": cond,
                    "mechanism": mechanism,
                    "gold": gold,
                    "A": A,
                    "B": B,
                    "C_text": C,
                    "E_text": E,
                    "prompt": prompt,
                })
    return pd.DataFrame(rows)


# -----------------------------
# Math utilities
# -----------------------------

def softmax_np(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    p = np.exp(x)
    return p / (np.sum(p) + 1e-12)


def entropy_bits_from_scores(scores):
    p = softmax_np(scores)
    h = -float(np.sum(p * np.log2(p + 1e-12)))
    return h, p


def n_eff_from_entropy(h):
    if not np.isfinite(h):
        return np.nan
    return float(2.0 ** h)


def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


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
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt
    return prompt


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
            if len(enc) >= 1:
                ids.append(enc[-1])
        out[label] = sorted(set(ids))
    return out


def get_candidate_sequences(tokenizer):
    seqs = {}
    for label in CANDIDATES:
        # leading space often matches next-token continuation better.
        ids = tokenizer(" " + label, add_special_tokens=False).input_ids
        if not ids:
            ids = tokenizer(label, add_special_tokens=False).input_ids
        seqs[label] = ids
    return seqs


@torch.no_grad()
def forward_hidden_logits(model, tokenizer, text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    outputs = model(**enc, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states
    logits = outputs.logits[:, -1, :].float().detach().cpu().numpy()[0]
    hs = [h[:, -1, :].float().detach().cpu().numpy()[0] for h in hidden_states]
    return hs, logits, enc


@torch.no_grad()
def sequence_logprob(model, tokenizer, prompt_text, seq_ids):
    """Compute log P(seq | prompt) with teacher forcing."""
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=MAX_LEN).input_ids[0].tolist()
    full_ids = prompt_ids + list(seq_ids)

    if len(full_ids) > MAX_LEN:
        full_ids = full_ids[-MAX_LEN:]

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=DEVICE)
    outputs = model(input_ids=input_ids, use_cache=False)
    logits = outputs.logits[0].float()

    # Sum logprob of seq tokens. Token at position j is predicted by logits[j-1].
    start = len(full_ids) - len(seq_ids)
    logp = 0.0
    for pos in range(start, len(full_ids)):
        if pos <= 0:
            continue
        token_id = full_ids[pos]
        lp = torch.log_softmax(logits[pos - 1], dim=-1)[token_id]
        logp += float(lp.detach().cpu())
    return logp


def lm_head_weight(model):
    return model.get_output_embeddings().weight.detach().float().cpu().numpy()


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
            "prefinal": span(26, 26),
            "final": span(27, 27),
            "all": list(range(n_hidden_states)),
        }

    # Relative fallback for Llama/Gemma etc.
    def rel(p):
        return int(round(n_layers * p))

    return {
        "init": span(0, rel(0.22)),
        "middle": span(rel(0.25), rel(0.68)),
        "boundary": span(rel(0.70), rel(0.78)),
        "basin": span(rel(0.80), rel(0.90)),
        "prefinal": span(rel(0.92), rel(0.95)),
        "final": span(n_layers, n_layers),
        "all": list(range(n_hidden_states)),
    }


def score_candidate_tokens(logits, candidate_token_ids):
    scores = {}
    for label in CANDIDATES:
        ids = candidate_token_ids.get(label, [])
        if not ids:
            scores[label] = -1e9
        else:
            scores[label] = float(np.max(logits[ids]))
    return scores


def extract_row_features(row, model, tokenizer, W, candidate_token_ids, candidate_seqs):
    prompt_text = format_chat(tokenizer, row["prompt"])
    hs, final_logits, enc = forward_hidden_logits(model, tokenizer, prompt_text)
    groups = layer_groups(len(hs))

    feat = {}

    # Sequence scores at final output only.
    seq_scores = {}
    for label, seq in candidate_seqs.items():
        seq_scores[label] = sequence_logprob(model, tokenizer, prompt_text, seq)

    h_seq, p_seq = entropy_bits_from_scores([seq_scores[x] for x in CANDIDATES])
    feat["seq_H_bits_final"] = h_seq
    feat["seq_N_eff_final"] = n_eff_from_entropy(h_seq)
    for label, p in zip(CANDIDATES, p_seq):
        feat[f"seq_p_{label}_final"] = float(p)
        feat[f"seq_score_{label}_final"] = float(seq_scores[label])

    seq_pred = CANDIDATES[int(np.argmax([seq_scores[x] for x in CANDIDATES]))]
    feat["seq_pred"] = seq_pred
    feat["seq_correct"] = int(seq_pred == row["gold"])

    # Per-layer answer possibility entropy and TopK volume.
    centers = []
    answer_H = []
    answer_N = []
    topk_H = []
    topk_N = []
    topk_spreads = []
    margins_CE = []

    for lidx, h in enumerate(hs):
        logits = h @ W.T
        cand_scores = score_candidate_tokens(logits, candidate_token_ids)
        scores_arr = [cand_scores[x] for x in CANDIDATES]
        h_ans, p_ans = entropy_bits_from_scores(scores_arr)
        n_ans = n_eff_from_entropy(h_ans)

        ids, vals = topk_for_hidden(h, W, TOPK)
        h_topk = topk_entropy(vals)
        n_topk = n_eff_from_entropy(h_topk)
        spread = topk_spread(ids, W)
        center = center_vec(ids, vals, W)

        centers.append(center)
        answer_H.append(h_ans)
        answer_N.append(n_ans)
        topk_H.append(h_topk)
        topk_N.append(n_topk)
        topk_spreads.append(spread)
        margins_CE.append(cand_scores["C"] - cand_scores["E"])

        feat[f"answer_H_bits_L{lidx}"] = h_ans
        feat[f"answer_N_eff_L{lidx}"] = n_ans
        feat[f"answer_margin_C_minus_E_L{lidx}"] = float(cand_scores["C"] - cand_scores["E"])
        feat[f"answer_score_C_L{lidx}"] = float(cand_scores["C"])
        feat[f"answer_score_E_L{lidx}"] = float(cand_scores["E"])
        feat[f"answer_score_OTHER_L{lidx}"] = float(cand_scores["OTHER"])
        for label, p in zip(CANDIDATES, p_ans):
            feat[f"answer_p_{label}_L{lidx}"] = float(p)

        feat[f"topk_H_bits_L{lidx}"] = h_topk
        feat[f"topk_N_eff_L{lidx}"] = n_topk
        feat[f"topk_spread_L{lidx}"] = spread
        feat[f"topk_top1_id_L{lidx}"] = int(ids[0]) if len(ids) else -1

    # Layer group aggregates.
    def group_vals(arr, name):
        idxs = groups.get(name, [])
        vals = [arr[i] for i in idxs if 0 <= i < len(arr)]
        return vals

    for g in ["init", "middle", "boundary", "basin", "prefinal", "final"]:
        ah = group_vals(answer_H, g)
        an = group_vals(answer_N, g)
        th = group_vals(topk_H, g)
        tn = group_vals(topk_N, g)
        sp = group_vals(topk_spreads, g)
        mg = group_vals(margins_CE, g)

        feat[f"{g}_answer_H_mean"] = float(np.mean(ah)) if ah else np.nan
        feat[f"{g}_answer_H_min"] = float(np.min(ah)) if ah else np.nan
        feat[f"{g}_answer_H_max"] = float(np.max(ah)) if ah else np.nan
        feat[f"{g}_answer_N_eff_mean"] = float(np.mean(an)) if an else np.nan
        feat[f"{g}_topk_H_mean"] = float(np.mean(th)) if th else np.nan
        feat[f"{g}_topk_N_eff_mean"] = float(np.mean(tn)) if tn else np.nan
        feat[f"{g}_topk_spread_mean"] = float(np.mean(sp)) if sp else np.nan
        feat[f"{g}_margin_CE_mean"] = float(np.mean(mg)) if mg else np.nan
        feat[f"{g}_margin_CE_slope"] = slope(mg) if mg else np.nan

    # Cbit gains for answer possibility compression.
    def first_val(arr, g):
        vals = group_vals(arr, g)
        return vals[0] if vals else np.nan

    def mean_val(arr, g):
        vals = group_vals(arr, g)
        return float(np.mean(vals)) if vals else np.nan

    H_init = mean_val(answer_H, "init")
    H_middle = mean_val(answer_H, "middle")
    H_boundary = mean_val(answer_H, "boundary")
    H_basin = mean_val(answer_H, "basin")
    H_final = mean_val(answer_H, "final")

    feat["Cbit_answer_init_to_boundary"] = H_init - H_boundary
    feat["Cbit_answer_init_to_basin"] = H_init - H_basin
    feat["Cbit_answer_init_to_final"] = H_init - H_final
    feat["Cbit_answer_middle_to_basin"] = H_middle - H_basin
    feat["Cbit_answer_boundary_to_basin"] = H_boundary - H_basin

    # TopK entropy changes are not Cbits over candidate structure, but local-volume proxy.
    TH_init = mean_val(topk_H, "init")
    TH_middle = mean_val(topk_H, "middle")
    TH_boundary = mean_val(topk_H, "boundary")
    TH_basin = mean_val(topk_H, "basin")
    TH_final = mean_val(topk_H, "final")

    feat["TopK_entropy_delta_init_to_boundary"] = TH_init - TH_boundary
    feat["TopK_entropy_delta_init_to_basin"] = TH_init - TH_basin
    feat["TopK_entropy_delta_init_to_final"] = TH_init - TH_final

    # TopK center drift over all layers.
    drifts = []
    for i in range(len(centers) - 1):
        drifts.append(1.0 - cosine(centers[i], centers[i + 1]))
    feat["topk_center_drift_mean"] = float(np.mean(drifts)) if drifts else 0.0
    feat["topk_center_drift_area"] = area(drifts)

    final_token_scores = score_candidate_tokens(final_logits, candidate_token_ids)
    final_pred = max(final_token_scores, key=final_token_scores.get)
    feat["token_pred"] = final_pred
    feat["token_correct"] = int(final_pred == row["gold"])
    feat["token_final_margin_C_minus_E"] = final_token_scores["C"] - final_token_scores["E"]

    return feat


# -----------------------------
# Summaries and verdict
# -----------------------------

def summarize_layers(df):
    rows = []
    layer_cols = [c for c in df.columns if c.startswith("answer_H_bits_L")]
    layer_ids = []
    for c in layer_cols:
        try:
            layer_ids.append(int(c.split("_L")[-1]))
        except Exception:
            pass
    layer_ids = sorted(set(layer_ids))

    for l in layer_ids:
        hcol = f"answer_H_bits_L{l}"
        ncol = f"answer_N_eff_L{l}"
        tcol = f"topk_H_bits_L{l}"
        scol = f"topk_spread_L{l}"
        mcol = f"answer_margin_C_minus_E_L{l}"

        rows.append({
            "layer": l,
            "answer_H_mean": df[hcol].mean(),
            "answer_H_std": df[hcol].std(),
            "answer_N_eff_mean": df[ncol].mean(),
            "topk_H_mean": df[tcol].mean(),
            "topk_spread_mean": df[scol].mean(),
            "margin_CE_mean": df[mcol].mean(),
            "margin_CE_abs_mean": df[mcol].abs().mean(),
        })
    return pd.DataFrame(rows)


def summarize_conditions(df):
    agg = df.groupby("condition").agg(
        n=("row_id", "count"),
        token_acc=("token_correct", "mean"),
        seq_acc=("seq_correct", "mean"),
        init_answer_H=("init_answer_H_mean", "mean"),
        middle_answer_H=("middle_answer_H_mean", "mean"),
        boundary_answer_H=("boundary_answer_H_mean", "mean"),
        basin_answer_H=("basin_answer_H_mean", "mean"),
        final_answer_H=("final_answer_H_mean", "mean"),
        init_N_eff=("init_answer_N_eff_mean", "mean"),
        boundary_N_eff=("boundary_answer_N_eff_mean", "mean"),
        basin_N_eff=("basin_answer_N_eff_mean", "mean"),
        final_N_eff=("final_answer_N_eff_mean", "mean"),
        Cbit_init_to_boundary=("Cbit_answer_init_to_boundary", "mean"),
        Cbit_init_to_basin=("Cbit_answer_init_to_basin", "mean"),
        Cbit_init_to_final=("Cbit_answer_init_to_final", "mean"),
        Cbit_boundary_to_basin=("Cbit_answer_boundary_to_basin", "mean"),
        init_topk_H=("init_topk_H_mean", "mean"),
        boundary_topk_H=("boundary_topk_H_mean", "mean"),
        basin_topk_H=("basin_topk_H_mean", "mean"),
        final_topk_H=("final_topk_H_mean", "mean"),
        topk_drift=("topk_center_drift_mean", "mean"),
    ).reset_index()
    return agg


def make_verdict(model_key, feat_df, layer_summary, condition_summary):
    reasons = []

    init_H = feat_df["init_answer_H_mean"].mean()
    basin_H = feat_df["basin_answer_H_mean"].mean()
    final_H = feat_df["final_answer_H_mean"].mean()
    init_N = feat_df["init_answer_N_eff_mean"].mean()
    final_N = feat_df["final_answer_N_eff_mean"].mean()
    cbit_if = feat_df["Cbit_answer_init_to_final"].mean()
    cbit_ib = feat_df["Cbit_answer_init_to_basin"].mean()

    # Existence: early effective candidate number larger than 1.5 and nontrivial entropy.
    exists_answer_space = bool(init_N > 1.5 and init_H > 0.5)
    compresses = bool(cbit_if > 0.15 or cbit_ib > 0.15)

    # Competition existence: ambiguous/competition should retain higher entropy than stable near boundary/basin.
    cond = condition_summary.set_index("condition")
    competition_conditions = [c for c in cond.index if "ambiguous" in c or "branch_conflict" in c]
    stable_conditions = [c for c in cond.index if "stable" in c]
    closure_conditions = [c for c in cond.index if "closure" in c]

    comp_boundary_H = cond.loc[competition_conditions, "boundary_answer_H"].mean() if competition_conditions else np.nan
    stable_boundary_H = cond.loc[stable_conditions, "boundary_answer_H"].mean() if stable_conditions else np.nan
    competition_retains_space = bool(np.isfinite(comp_boundary_H) and np.isfinite(stable_boundary_H) and comp_boundary_H >= stable_boundary_H)

    # Layer compression shape: H should generally decline by basin/final.
    if exists_answer_space:
        reasons.append("Initial answer-candidate effective space is nontrivial.")
    if compresses:
        reasons.append("Answer candidate entropy decreases, yielding positive Cbit gain.")
    if competition_retains_space:
        reasons.append("Competition-like conditions retain larger boundary possibility space than stable conditions.")

    if exists_answer_space and compresses and competition_retains_space:
        verdict = "PASS_POSSIBILITY_SPACE_AND_CBIT_MEASURABLE"
    elif exists_answer_space and compresses:
        verdict = "PARTIAL_CBIT_GAIN_MEASURABLE"
    elif exists_answer_space:
        verdict = "PARTIAL_POSSIBILITY_SPACE_EXISTS_NO_CLEAR_COMPRESSION"
    else:
        verdict = "FAIL_NO_EFFECTIVE_POSSIBILITY_SPACE_DETECTED"

    return {
        "model_key": model_key,
        "verdict": verdict,
        "reasons": reasons,
        "metrics": {
            "init_answer_H_mean": safe_float(init_H),
            "basin_answer_H_mean": safe_float(basin_H),
            "final_answer_H_mean": safe_float(final_H),
            "init_answer_N_eff_mean": safe_float(init_N),
            "final_answer_N_eff_mean": safe_float(final_N),
            "mean_Cbit_init_to_basin": safe_float(cbit_ib),
            "mean_Cbit_init_to_final": safe_float(cbit_if),
            "competition_boundary_H": safe_float(comp_boundary_H),
            "stable_boundary_H": safe_float(stable_boundary_H),
            "token_acc": safe_float(feat_df["token_correct"].mean()),
            "seq_acc": safe_float(feat_df["seq_correct"].mean()),
            "n_rows": int(len(feat_df)),
        },
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
        info["status"] = "missing_path"
        info["error"] = f"Path not found: {spec['path']}"
        return info

    dataset = build_dataset()
    dataset.to_csv(model_out / f"psa1_{model_key}_dataset.csv", index=False, encoding="utf-8-sig")

    model = None
    try:
        t0 = time.time()
        model, tokenizer = load_model_and_tokenizer(spec["path"], spec.get("trust_remote_code", True))
        W = lm_head_weight(model)
        cand_token_ids = get_candidate_token_ids(tokenizer)
        cand_seqs = get_candidate_sequences(tokenizer)

        rows = []
        errors = []

        for i, row in dataset.iterrows():
            if i % 20 == 0:
                print(f"[{model_key}] extracting row {i}/{len(dataset)}", flush=True)
            try:
                feat = extract_row_features(row, model, tokenizer, W, cand_token_ids, cand_seqs)
                rows.append({**row.to_dict(), **feat})
            except Exception as e:
                errors.append({
                    "row_id": int(row["row_id"]),
                    "condition": row["condition"],
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
                print(f"[warn][{model_key}] row {row['row_id']} failed: {type(e).__name__}: {e}", flush=True)

        feat_df = pd.DataFrame(rows)
        err_df = pd.DataFrame(errors)

        feat_df.to_csv(model_out / f"psa1_{model_key}_features.csv", index=False, encoding="utf-8-sig")
        err_df.to_csv(model_out / f"psa1_{model_key}_errors.csv", index=False, encoding="utf-8-sig")

        if feat_df.empty:
            info["status"] = "feature_empty"
            info["n_errors"] = len(errors)
            return info

        layer_summary = summarize_layers(feat_df)
        condition_summary = summarize_conditions(feat_df)

        layer_summary.to_csv(model_out / f"psa1_{model_key}_layer_summary.csv", index=False, encoding="utf-8-sig")
        condition_summary.to_csv(model_out / f"psa1_{model_key}_condition_summary.csv", index=False, encoding="utf-8-sig")

        verdict = make_verdict(model_key, feat_df, layer_summary, condition_summary)
        verdict.update({
            "status": "done",
            "runtime_sec": time.time() - t0,
            "n_errors": int(len(errors)),
            "outputs": {
                "features": str(model_out / f"psa1_{model_key}_features.csv"),
                "layer_summary": str(model_out / f"psa1_{model_key}_layer_summary.csv"),
                "condition_summary": str(model_out / f"psa1_{model_key}_condition_summary.csv"),
            }
        })

        with open(model_out / f"psa1_{model_key}_verdict.json", "w", encoding="utf-8") as f:
            json.dump(verdict, f, ensure_ascii=False, indent=2)

        info.update(verdict)

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

    for model_key in MODELS_TO_RUN:
        print("=" * 80)
        print(f"[PSA-1] Running model: {model_key}")
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
            "model_name": info.get("model_name"),
            "status": info.get("status"),
            "verdict": info.get("verdict"),
            "init_answer_H_mean": m.get("init_answer_H_mean"),
            "basin_answer_H_mean": m.get("basin_answer_H_mean"),
            "final_answer_H_mean": m.get("final_answer_H_mean"),
            "init_answer_N_eff_mean": m.get("init_answer_N_eff_mean"),
            "final_answer_N_eff_mean": m.get("final_answer_N_eff_mean"),
            "mean_Cbit_init_to_basin": m.get("mean_Cbit_init_to_basin"),
            "mean_Cbit_init_to_final": m.get("mean_Cbit_init_to_final"),
            "competition_boundary_H": m.get("competition_boundary_H"),
            "stable_boundary_H": m.get("stable_boundary_H"),
            "token_acc": m.get("token_acc"),
            "seq_acc": m.get("seq_acc"),
            "n_rows": m.get("n_rows"),
            "n_errors": info.get("n_errors"),
            "error": info.get("error"),
        })

    cross = pd.DataFrame(rows)
    cross.to_csv(OUT_DIR / "psa1_cross_model_summary.csv", index=False, encoding="utf-8-sig")

    done = [x for x in all_infos if x.get("status") == "done"]
    pass_count = sum(str(x.get("verdict", "")).startswith("PASS") for x in done)
    partial_count = sum(str(x.get("verdict", "")).startswith("PARTIAL") for x in done)

    if len(done) == 0:
        global_verdict = "FAIL_NO_MODELS_RAN"
    elif pass_count == len(done):
        global_verdict = "PASS_PSA1"
    elif pass_count + partial_count == len(done):
        global_verdict = "PARTIAL_PSA1"
    else:
        global_verdict = "MIXED_PSA1"

    global_obj = {
        "verdict": global_verdict,
        "n_models_done": len(done),
        "n_models_pass": pass_count,
        "n_models_partial": partial_count,
        "models": all_infos,
        "summary_csv": str(OUT_DIR / "psa1_cross_model_summary.csv"),
    }

    with open(OUT_DIR / "psa1_verdict.json", "w", encoding="utf-8") as f:
        json.dump(global_obj, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(json.dumps(global_obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
