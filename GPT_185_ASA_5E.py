
# ============================================================
# ASA-5E: Safe Curvature Direction Audit
#
# Motivation:
#   ASA-5 / ASA-5D found a weak acceleration_residual signal, but:
#     - raw acceleration_delta failed
#     - residual norm after shared-subspace removal was ~0
#     - normalizing that near-zero residual created a numerical caveat
#
# ASA-5E changes the direction construction:
#   Instead of using:
#       a_S - a_Q
#   or near-zero residual vectors,
#
#   it learns curvature-sensitive directions from training data:
#       a_l = H_{l+2} - 2H_{l+1} + H_l
#       a_l -> DeltaU
#
# Main controls:
#   curvature_ridge          : Ridge coefficient direction a_l -> DeltaU
#   curvature_pls_like       : Covariance direction Cov(a_l, DeltaU)
#   curvature_phase_delta    : a_stable - a_closure, raw old baseline
#   velocity_ridge           : Ridge direction v_l -> DeltaU
#   position_ridge           : Ridge direction H_l -> DeltaU
#   random
#   answer positive control
#   shuffle_curvature_ridge  : DeltaU-shuffled Ridge directions
#
# Safety:
#   No normalization of near-zero residual vectors.
#   Any direction with norm < min_dir_norm is dropped.
#
# Default:
#   Qwen, Q->S, layers L15-L19, decision L20-L25.
#
# Outputs:
#   asa5e_outputs/
#     asa5e_config.json
#     asa5e_dataset.csv
#     asa5e_baseline_features.csv
#     asa5e_split.csv
#     asa5e_direction_audit.csv
#     asa5e_steering_results.csv
#     asa5e_specificity_summary.csv
#     asa5e_shuffle_specificity_significance.csv
#     asa5e_dose_response.csv
#     asa5e_verdict.csv
#     asa5e_summary.json
# ============================================================

import os
import gc
import re
import json
import random
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import GroupShuffleSplit
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

QWEN_PATH = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"
LLAMA_PATH = r"D:\model\Llama-3.2-1B-Instruct"
GEMMA_PATH = r"D:\model\gemma-2-2b-it"

MODEL_PATHS = {
    "qwen": QWEN_PATH,
    "llama": LLAMA_PATH,
    "gemma": GEMMA_PATH,
}

@dataclass
class Config:
    model_key: str = "qwen"
    model_path: str = "auto"
    save_dir: str = "./asa5e_outputs"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float16"
    seed: int = 42

    n_graphs: int = 96
    max_len: int = 260
    batch_size: int = 4
    test_size: float = 0.30

    # curvature a_l for l=15..18 modifies output at layer l+1 => H16..H19
    accel_layers: Tuple[int, ...] = tuple(range(15, 19))
    velocity_layers: Tuple[int, ...] = tuple(range(15, 20))
    position_layers: Tuple[int, ...] = tuple(range(15, 20))
    decision_layers: Tuple[int, ...] = tuple(range(20, 26))

    # Steering dose
    alphas: Tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30)

    # Main direction
    run_q_to_s: bool = True
    run_k_to_s: bool = False
    run_s_to_q: bool = False

    # Direction learning
    ridge_alpha: float = 10.0
    min_dir_norm: float = 1e-4

    # Shuffle controls
    n_shuffles: int = 50
    shuffle_alphas: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)

    # Controls
    include_answer_control: bool = True
    include_random_control: bool = True
    include_phase_delta_control: bool = True
    include_curvature_ridge: bool = True
    include_curvature_pls_like: bool = True
    include_velocity_ridge: bool = True
    include_position_ridge: bool = True

CFG = Config()

# ============================================================
# SEED / PATH
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CFG.seed)

def dtype_from_cfg(cfg):
    if cfg.dtype == "float16" and cfg.device == "cuda":
        return torch.float16
    if cfg.dtype == "bfloat16" and cfg.device == "cuda":
        return torch.bfloat16
    return torch.float32

def resolve_model_path(cfg):
    if cfg.model_path == "auto" or "YOUR_SNAPSHOT" in str(cfg.model_path):
        return MODEL_PATHS[cfg.model_key]
    return cfg.model_path

# ============================================================
# DATASET
# ============================================================

LABEL_CANDIDATES = [
    "Red", "Blue", "Green", "Yellow", "North", "South", "East", "West",
    "Copper", "Silver", "Gold", "Iron", "Circle", "Square", "Triangle", "Star",
    "River", "Mountain", "Forest", "Ocean", "Sun", "Moon", "Cloud", "Stone",
    "Alpha", "Beta", "Gamma", "Delta", "Apple", "Orange", "Lemon", "Pear",
]

ENTITIES = [
    ("Ava", "Bela", "Cora"),
    ("Darin", "Elo", "Faye"),
    ("Galen", "Hera", "Ivo"),
    ("Juno", "Kira", "Lio"),
    ("Mira", "Nero", "Orin"),
    ("Pia", "Quin", "Rhea"),
    ("Sola", "Taro", "Una"),
    ("Vera", "Wen", "Xio"),
    ("Yara", "Zeno", "Nia"),
    ("Orla", "Pavel", "Rin"),
    ("Nora", "Silas", "Tess"),
    ("Uma", "Vito", "Willa"),
    ("Xena", "Yuri", "Zara"),
    ("Iris", "Kai", "Lena"),
    ("Omar", "Priya", "Quill"),
    ("Ravi", "Sara", "Theo"),
]

REL_WORDS = [
    ("belongs to", "is located at"),
    ("is assigned to", "maps to"),
    ("is part of", "points to"),
    ("is grouped under", "has label"),
    ("routes through", "ends at"),
    ("is linked with", "resolves to"),
]

STABLE_CONDS = ["clean", "redundant", "irrelevant", "paraphrase"]
COMPETITION_CONDS = ["weak_distractor", "competition_balanced", "direct_conflict"]
CLOSURE_CONDS = ["closure_update", "closure_override", "exception_override"]
ALL_CONDS = STABLE_CONDS + COMPETITION_CONDS + CLOSURE_CONDS

def mechanism_of(cond):
    if cond in STABLE_CONDS:
        return "stable"
    if cond in COMPETITION_CONDS:
        return "competition"
    if cond in CLOSURE_CONDS:
        return "closure"
    raise ValueError(cond)

def continuation_ids(tokenizer, text):
    return tokenizer(" " + text, add_special_tokens=False)["input_ids"]

def select_single_token_labels(tokenizer, save_dir):
    rows, chosen = [], []
    for lab in LABEL_CANDIDATES:
        ids = continuation_ids(tokenizer, lab)
        rows.append({"label": lab, "ids": str(ids), "len": len(ids), "single": int(len(ids) == 1)})
        if len(ids) == 1:
            chosen.append(lab)
    pd.DataFrame(rows).to_csv(save_dir / "asa5e_label_tokenization_audit.csv", index=False, encoding="utf-8-sig")
    if len(chosen) < 12:
        raise RuntimeError(f"Need >=12 single-token labels; got {chosen}")
    return chosen[:12]

def make_prompt(cond, a, b, d, clean_label, conflict_label, aux_label, relation1, relation2):
    if cond == "clean":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} {relation1} {b}.",
            f"Fact 2: {b} {relation2} {clean_label}.",
            f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
            "Answer with exactly one word:",
        ]
    elif cond == "redundant":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} {relation1} {b}.",
            f"Fact 2: {b} {relation2} {clean_label}.",
            f"Confirmation: {a} still goes through {b}.",
            f"Confirmation: {b} still points to {clean_label}.",
            f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
            "Answer with exactly one word:",
        ]
    elif cond == "irrelevant":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} {relation1} {b}.",
            f"Fact 2: {b} {relation2} {clean_label}.",
            f"Irrelevant fact: {d} is associated with {aux_label}.",
            f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
            "Answer with exactly one word:",
        ]
    elif cond == "paraphrase":
        lines = [
            f"In this example, {a} reaches {b}.",
            f"The destination connected to {b} is {clean_label}.",
            f"Based on those two links, select the label for {a}: {clean_label} or {conflict_label}.",
            "Answer:",
        ]
    elif cond == "weak_distractor":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} {relation1} {b}.",
            f"Fact 2: {b} {relation2} {clean_label}.",
            f"Weak note: some unrelated source mentions {conflict_label}, but does not update the graph.",
            f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
            "Answer with exactly one word:",
        ]
    elif cond == "competition_balanced":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} {relation1} {b}.",
            f"Fact 2: {b} {relation2} {clean_label}.",
            f"Competing fact: {a} is also associated with {conflict_label}.",
            f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
            "Answer with exactly one word:",
        ]
    elif cond == "direct_conflict":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} {relation1} {b}.",
            f"Fact 2: {b} {relation2} {clean_label}.",
            f"Direct conflicting fact: {a} {relation2} {conflict_label}.",
            f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
            "Answer with exactly one word:",
        ]
    elif cond == "closure_update":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} {relation1} {b}.",
            f"Old record: {b} {relation2} {clean_label}.",
            f"Updated record: {b} {relation2} {conflict_label}, not {clean_label}.",
            f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
            "Answer with exactly one word:",
        ]
    elif cond == "closure_override":
        lines = [
            "You are given a rule system.",
            f"General rule: items that {relation1} {b} receive label {clean_label}.",
            f"Override rule: in this case, items that {relation1} {b} receive label {conflict_label}.",
            f"Fact: {a} {relation1} {b}.",
            f"Question: Which label applies to {a}: {clean_label} or {conflict_label}?",
            "Answer with exactly one word:",
        ]
    elif cond == "exception_override":
        lines = [
            "You are given a rule system.",
            f"General rule: items connected to {b} use label {clean_label}.",
            f"Exception: {a} is a special case and uses label {conflict_label}.",
            f"Fact: {a} is connected to {b}.",
            f"Question: Which label applies to {a}: {clean_label} or {conflict_label}?",
            "Answer with exactly one word:",
        ]
    else:
        raise ValueError(cond)
    return "\n".join(lines)

def build_dataset(tokenizer, cfg, save_dir):
    labels = select_single_token_labels(tokenizer, save_dir)
    rows, graph_count = [], 0
    n_labels = len(labels)
    for ent_i, (a, b, d) in enumerate(ENTITIES):
        for rel_i, (r1, r2) in enumerate(REL_WORDS):
            if graph_count >= cfg.n_graphs:
                break
            clean_label = labels[(2 * graph_count) % n_labels]
            conflict_label = labels[(2 * graph_count + 1) % n_labels]
            aux_label = labels[(2 * graph_count + 2) % n_labels]
            clean_id = continuation_ids(tokenizer, clean_label)[0]
            conflict_id = continuation_ids(tokenizer, conflict_label)[0]
            for cond in ALL_CONDS:
                rows.append({
                    "prompt_id": f"g{graph_count:03d}_{cond}",
                    "graph_id": graph_count,
                    "entity_id": ent_i,
                    "relation_id": rel_i,
                    "condition": cond,
                    "mechanism": mechanism_of(cond),
                    "clean_label": clean_label,
                    "conflict_label": conflict_label,
                    "clean_token_id": clean_id,
                    "conflict_token_id": conflict_id,
                    "text": make_prompt(cond, a, b, d, clean_label, conflict_label, aux_label, r1, r2),
                })
            graph_count += 1
        if graph_count >= cfg.n_graphs:
            break
    df = pd.DataFrame(rows)
    df.to_csv(save_dir / "asa5e_dataset.csv", index=False, encoding="utf-8-sig")
    return df

# ============================================================
# MODEL UTILS
# ============================================================

def load_model_and_tokenizer(cfg):
    cfg.model_path = resolve_model_path(cfg)
    print(f"[MODEL] model_key={cfg.model_key} resolved_path={cfg.model_path}")
    tok = AutoTokenizer.from_pretrained(cfg.model_path, local_files_only=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype_from_cfg(cfg),
        device_map="auto" if cfg.device == "cuda" else None,
    )
    if cfg.device == "cpu":
        model.to(cfg.device)
    model.eval()
    return model, tok

def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise RuntimeError("Cannot locate transformer layers")

def get_lm_head_weight(model):
    if hasattr(model, "lm_head"):
        return model.lm_head.weight.detach()
    raise RuntimeError("Cannot locate lm_head")

def last_positions(attention_mask):
    return torch.full(
        (attention_mask.shape[0],),
        attention_mask.shape[1] - 1,
        dtype=torch.long,
        device=attention_mask.device,
    )

def normalize_np(v, eps=1e-8):
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < eps:
        return None, float(n)
    return (v / n).astype(np.float32), float(n)

# ============================================================
# BASELINE EXTRACTION
# ============================================================

def extract_baseline(model, tokenizer, df, cfg, save_dir):
    print("[EXTRACT] baseline hidden states / decision targets")
    W = get_lm_head_weight(model).detach().float()
    W_device = W.to(model.device)

    n = len(df)
    needed_layers = sorted(set(
        list(cfg.position_layers)
        + [l+1 for l in cfg.velocity_layers]
        + list(cfg.velocity_layers)
        + [l+2 for l in cfg.accel_layers]
        + [l+1 for l in cfg.accel_layers]
        + list(cfg.accel_layers)
        + list(cfg.decision_layers)
    ))
    h_by_layer = {l: None for l in needed_layers}
    R_by_layer = {l: np.zeros(n, dtype=np.float32) for l in cfg.decision_layers}
    final_clean_logits = np.zeros(n, dtype=np.float32)
    final_conflict_logits = np.zeros(n, dtype=np.float32)

    texts = df["text"].tolist()
    clean_ids_all = df["clean_token_id"].values.astype(np.int64)
    conflict_ids_all = df["conflict_token_id"].values.astype(np.int64)

    with torch.no_grad():
        for start in range(0, n, cfg.batch_size):
            end = min(n, start + cfg.batch_size)
            inputs = tokenizer(
                texts[start:end],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.max_len,
            ).to(model.device)
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            hstates = outputs.hidden_states
            pos = last_positions(inputs["attention_mask"])
            bsz = end - start
            idx_t = torch.arange(bsz, device=model.device)
            cids = torch.tensor(clean_ids_all[start:end], dtype=torch.long, device=model.device)
            eids = torch.tensor(conflict_ids_all[start:end], dtype=torch.long, device=model.device)

            for l in needed_layers:
                h = hstates[l + 1][idx_t, pos, :].detach().float().cpu().numpy().astype(np.float32)
                if h_by_layer[l] is None:
                    h_by_layer[l] = np.zeros((n, h.shape[1]), dtype=np.float32)
                h_by_layer[l][start:end] = h

            for l in cfg.decision_layers:
                ht = hstates[l + 1][idx_t, pos, :].detach().float()
                clean_logits = torch.sum(ht * W_device[cids].float(), dim=1)
                conflict_logits = torch.sum(ht * W_device[eids].float(), dim=1)
                R_by_layer[l][start:end] = (clean_logits - conflict_logits).detach().cpu().numpy().astype(np.float32)

            final_logits = outputs.logits[idx_t, pos, :].detach().float()
            final_clean_logits[start:end] = final_logits[idx_t, cids].detach().cpu().numpy().astype(np.float32)
            final_conflict_logits[start:end] = final_logits[idx_t, eids].detach().cpu().numpy().astype(np.float32)

            print(f"  extracted {end}/{n}")
            del outputs, hstates, inputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    out = df.copy()
    for l in cfg.decision_layers:
        out[f"R_L{l}"] = R_by_layer[l]
    out["final_clean_logit"] = final_clean_logits
    out["final_conflict_logit"] = final_conflict_logits
    out["pair_choice"] = np.where(final_clean_logits >= final_conflict_logits, "clean", "conflict")

    clean_anchor = out[out["condition"] == "clean"].set_index("graph_id")
    dR_cols = []
    for l in cfg.decision_layers:
        vals = []
        for _, row in out.iterrows():
            vals.append(row[f"R_L{l}"] - clean_anchor.loc[row["graph_id"], f"R_L{l}"])
        col = f"dR_L{l}"
        out[col] = np.asarray(vals, dtype=np.float32)
        dR_cols.append(col)

    X_dec = out[dR_cols].values.astype(np.float32)
    pca = PCA(n_components=1)
    du = pca.fit_transform(X_dec)[:, 0]
    if np.mean(du[out["mechanism"] == "stable"]) < np.mean(du[out["mechanism"] == "closure"]):
        du = -du
        pca.components_[0] *= -1
    out["DeltaU"] = du.astype(np.float32)
    out.to_csv(save_dir / "asa5e_baseline_features.csv", index=False, encoding="utf-8-sig")
    print("[PCA] DeltaU explained variance:", float(pca.explained_variance_ratio_[0]))
    return out, h_by_layer, pca

# ============================================================
# SPLIT
# ============================================================

def make_split(df, cfg, save_dir):
    graph_ids = df["graph_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.seed)
    tr, te = next(gss.split(df, groups=graph_ids))
    test_set = set(te.tolist())
    split_df = pd.DataFrame({
        "prompt_id": df["prompt_id"],
        "graph_id": df["graph_id"],
        "condition": df["condition"],
        "mechanism": df["mechanism"],
        "split": ["test" if i in test_set else "train" for i in range(len(df))]
    })
    split_df.to_csv(save_dir / "asa5e_split.csv", index=False, encoding="utf-8-sig")
    return np.asarray(tr), np.asarray(te), split_df

# ============================================================
# DIRECTION LEARNING
# ============================================================

def mean_by_mechanism(X, base_df, train_idx, mech):
    idx = [i for i in train_idx if base_df.iloc[i]["mechanism"] == mech]
    idx = np.asarray(idx, dtype=int)
    return X[idx].mean(axis=0)

def fit_vector_direction(X, y, method, alpha=10.0, min_norm=1e-4):
    """
    Returns unit vector and audit.
    method:
      ridge: Ridge coefficient vector
      pls_like: covariance direction X^T y
    """
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    y0 = y - y.mean()
    X0 = X - X.mean(axis=0, keepdims=True)

    if method == "ridge":
        reg = Ridge(alpha=alpha)
        reg.fit(X0, y0)
        coef = reg.coef_.astype(np.float32)
        pred = reg.predict(X0)
        train_r2 = r2_score(y0, pred)
        train_corr = np.corrcoef(y0, pred)[0, 1] if np.std(pred) > 1e-8 else 0.0
    elif method == "pls_like":
        coef = (X0.T @ y0 / max(1, len(y0))).astype(np.float32)
        pred = X0 @ coef
        train_r2 = r2_score(y0, pred) if np.std(pred) > 1e-8 else np.nan
        train_corr = np.corrcoef(y0, pred)[0, 1] if np.std(pred) > 1e-8 else 0.0
    else:
        raise ValueError(method)

    unit, norm = normalize_np(coef, eps=min_norm)
    valid = unit is not None
    return unit, {
        "method": method,
        "coef_norm": norm,
        "valid": bool(valid),
        "train_r2": float(train_r2) if np.isfinite(train_r2) else np.nan,
        "train_corr": float(train_corr) if np.isfinite(train_corr) else np.nan,
    }

def fit_directions(base_df, h_by_layer, train_idx, cfg, save_dir):
    y = base_df["DeltaU"].values.astype(np.float32)

    directions = {
        "curvature_ridge": {},
        "curvature_pls_like": {},
        "curvature_phase_delta": {},
        "velocity_ridge": {},
        "position_ridge": {},
        "random": {},
    }
    audits = []
    rng = np.random.default_rng(cfg.seed + 111)

    # Curvature directions a_l -> DeltaU, hook at l+1.
    for l in cfg.accel_layers:
        A = h_by_layer[l + 2] - 2 * h_by_layer[l + 1] + h_by_layer[l]
        Xtr = A[train_idx]

        # Ridge
        unit, au = fit_vector_direction(
            Xtr, y[train_idx], method="ridge",
            alpha=cfg.ridge_alpha, min_norm=cfg.min_dir_norm
        )
        au.update({"control": "curvature_ridge", "source_layer": l, "hook_layer": l + 1})
        audits.append(au)
        if unit is not None:
            directions["curvature_ridge"][l + 1] = unit

        # PLS-like covariance
        unit, au = fit_vector_direction(
            Xtr, y[train_idx], method="pls_like",
            alpha=cfg.ridge_alpha, min_norm=cfg.min_dir_norm
        )
        au.update({"control": "curvature_pls_like", "source_layer": l, "hook_layer": l + 1})
        audits.append(au)
        if unit is not None:
            directions["curvature_pls_like"][l + 1] = unit

        # Raw phase delta old baseline: stable - closure
        acc_s = mean_by_mechanism(A, base_df, train_idx, "stable")
        acc_q = mean_by_mechanism(A, base_df, train_idx, "closure")
        delta = acc_s - acc_q
        unit, norm = normalize_np(delta, eps=cfg.min_dir_norm)
        audits.append({
            "control": "curvature_phase_delta",
            "method": "phase_mean_diff",
            "source_layer": l,
            "hook_layer": l + 1,
            "coef_norm": norm,
            "valid": bool(unit is not None),
            "train_r2": np.nan,
            "train_corr": np.nan,
        })
        if unit is not None:
            directions["curvature_phase_delta"][l + 1] = unit

    # Velocity ridge v_l -> DeltaU, hook at l+1.
    for l in cfg.velocity_layers:
        V = h_by_layer[l + 1] - h_by_layer[l]
        Xtr = V[train_idx]
        unit, au = fit_vector_direction(
            Xtr, y[train_idx], method="ridge",
            alpha=cfg.ridge_alpha, min_norm=cfg.min_dir_norm
        )
        au.update({"control": "velocity_ridge", "source_layer": l, "hook_layer": l + 1})
        audits.append(au)
        if unit is not None:
            directions["velocity_ridge"][l + 1] = unit

    # Position ridge H_l -> DeltaU, hook at l.
    for l in cfg.position_layers:
        H = h_by_layer[l]
        Xtr = H[train_idx]
        unit, au = fit_vector_direction(
            Xtr, y[train_idx], method="ridge",
            alpha=cfg.ridge_alpha, min_norm=cfg.min_dir_norm
        )
        au.update({"control": "position_ridge", "source_layer": l, "hook_layer": l})
        audits.append(au)
        if unit is not None:
            directions["position_ridge"][l] = unit

    # Random directions for curvature hook layers.
    hook_layers = sorted(set([l + 1 for l in cfg.accel_layers]))
    dim = next(iter(h_by_layer.values())).shape[1]
    for l in hook_layers:
        rd = rng.standard_normal(dim).astype(np.float32)
        unit, norm = normalize_np(rd, eps=cfg.min_dir_norm)
        directions["random"][l] = unit
        audits.append({
            "control": "random",
            "method": "random_normal",
            "source_layer": l - 1,
            "hook_layer": l,
            "coef_norm": norm,
            "valid": True,
            "train_r2": np.nan,
            "train_corr": np.nan,
        })

    # Shuffle curvature ridge directions: y permuted.
    shuffle_dirs = {}
    for s in range(cfg.n_shuffles):
        rng_s = np.random.default_rng(cfg.seed + 10000 + s)
        y_perm = y[train_idx].copy()
        rng_s.shuffle(y_perm)
        sh = {}
        for l in cfg.accel_layers:
            A = h_by_layer[l + 2] - 2 * h_by_layer[l + 1] + h_by_layer[l]
            Xtr = A[train_idx]
            unit, au = fit_vector_direction(
                Xtr, y_perm, method="ridge",
                alpha=cfg.ridge_alpha, min_norm=cfg.min_dir_norm
            )
            audits.append({
                **au,
                "control": f"shuffle_curvature_ridge_{s:03d}",
                "source_layer": l,
                "hook_layer": l + 1,
            })
            if unit is not None:
                sh[l + 1] = unit
        shuffle_dirs[s] = sh

    audit_df = pd.DataFrame(audits)
    audit_df.to_csv(save_dir / "asa5e_direction_audit.csv", index=False, encoding="utf-8-sig")
    return directions, shuffle_dirs

# ============================================================
# STEERING
# ============================================================

def make_vector_hook(direction_by_layer, alpha, mode, W_device, clean_ids_batch=None, conflict_ids_batch=None):
    dir_tensors = {
        int(l): torch.tensor(v, dtype=torch.float32, device=W_device.device)
        for l, v in direction_by_layer.items() if v is not None
    }

    def hook_factory(l):
        def hook(module, inputs, output):
            if alpha == 0.0:
                return output
            if l not in dir_tensors and mode != "answer":
                return output

            if isinstance(output, tuple):
                h = output[0]
                rest = output[1:]
            else:
                h = output
                rest = None

            B, S, D = h.shape
            pos = torch.full((B,), S - 1, dtype=torch.long, device=h.device)
            idx = torch.arange(B, device=h.device)

            if mode == "answer":
                d_batch = W_device[clean_ids_batch].float() - W_device[conflict_ids_batch].float()
                d_batch = d_batch / (torch.norm(d_batch, dim=1, keepdim=True) + 1e-8)
                d_batch = d_batch.to(h.dtype)
            else:
                d = dir_tensors[l].to(h.device, h.dtype)
                d_batch = d[None, :].expand(B, -1)

            h_last = h[idx, pos, :]
            scale = h_last.float().std(dim=-1, keepdim=True).to(h.dtype)
            h_new_last = h_last + alpha * scale * d_batch

            h2 = h.clone()
            h2[idx, pos, :] = h_new_last

            if rest is not None:
                return (h2,) + rest
            return h2
        return hook
    return hook_factory

def forward_eval(model, tokenizer, df_subset, cfg, pca_decision,
                 clean_anchor_R, direction_by_layer=None,
                 layers_to_hook=None, alpha=0.0, mode="vector"):
    W = get_lm_head_weight(model).detach().float()
    W_device = W.to(model.device)
    rows = []

    texts = df_subset["text"].tolist()
    clean_ids_all = df_subset["clean_token_id"].values.astype(np.int64)
    conflict_ids_all = df_subset["conflict_token_id"].values.astype(np.int64)
    layer_modules = get_layers(model)

    with torch.no_grad():
        for start in range(0, len(df_subset), cfg.batch_size):
            end = min(len(df_subset), start + cfg.batch_size)
            batch = df_subset.iloc[start:end].reset_index(drop=True)

            inputs = tokenizer(
                texts[start:end],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.max_len,
            ).to(model.device)

            cids = torch.tensor(clean_ids_all[start:end], dtype=torch.long, device=model.device)
            eids = torch.tensor(conflict_ids_all[start:end], dtype=torch.long, device=model.device)

            handles = []
            if alpha != 0.0 and layers_to_hook is not None:
                hook_factory = make_vector_hook(
                    direction_by_layer=direction_by_layer or {},
                    alpha=float(alpha),
                    mode=mode,
                    W_device=W_device,
                    clean_ids_batch=cids,
                    conflict_ids_batch=eids,
                )
                for l in layers_to_hook:
                    if l < len(layer_modules):
                        handles.append(layer_modules[l].register_forward_hook(hook_factory(l)))

            outputs = model(**inputs, output_hidden_states=True, use_cache=False)

            for h in handles:
                h.remove()

            hstates = outputs.hidden_states
            pos = last_positions(inputs["attention_mask"])
            B = end - start
            idx_t = torch.arange(B, device=model.device)

            dR_mat = []
            for l in cfg.decision_layers:
                h = hstates[l + 1][idx_t, pos, :].detach().float()
                clean_logits = torch.sum(h * W_device[cids].float(), dim=1)
                conflict_logits = torch.sum(h * W_device[eids].float(), dim=1)
                R = (clean_logits - conflict_logits).detach().cpu().numpy().astype(np.float32)
                dR = []
                for bi, (_, row) in enumerate(batch.iterrows()):
                    g = int(row["graph_id"])
                    dR.append(R[bi] - clean_anchor_R[g][l])
                dR_mat.append(dR)
            dR_mat = np.asarray(dR_mat, dtype=np.float32).T
            du = pca_decision.transform(dR_mat)[:, 0].astype(np.float32)

            final_logits = outputs.logits[idx_t, pos, :].detach().float()
            final_clean = final_logits[idx_t, cids].detach().cpu().numpy()
            final_conflict = final_logits[idx_t, eids].detach().cpu().numpy()
            pair_choice = np.where(final_clean >= final_conflict, "clean", "conflict")

            for bi, (_, row) in enumerate(batch.iterrows()):
                rows.append({
                    "prompt_id": row["prompt_id"],
                    "graph_id": int(row["graph_id"]),
                    "condition": row["condition"],
                    "mechanism": row["mechanism"],
                    "clean_label": row["clean_label"],
                    "conflict_label": row["conflict_label"],
                    "DeltaU": float(du[bi]),
                    "final_clean_logit": float(final_clean[bi]),
                    "final_conflict_logit": float(final_conflict[bi]),
                    "pair_choice": str(pair_choice[bi]),
                })

            del outputs, hstates, inputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return pd.DataFrame(rows)

# ============================================================
# EVAL
# ============================================================

def safe_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else np.nan

def summarize_specificity(results_df, base_df, direction_targets):
    merged = results_df.merge(
        base_df[["prompt_id", "DeltaU", "pair_choice"]],
        on="prompt_id",
        suffixes=("", "_base")
    )
    merged["shift"] = merged["DeltaU"] - merged["DeltaU_base"]
    merged["abs_shift"] = merged["shift"].abs()
    merged["target_hit"] = np.where(merged["target_pair_choice"] == merged["pair_choice"], 1, 0)

    rows = []
    for (direction, control, alpha), g in merged.groupby(["direction", "control", "alpha"]):
        target_mech = direction_targets[direction]
        target = g[g["mechanism"] == target_mech]
        nontarget = g[g["mechanism"] != target_mech]
        rows.append({
            "direction": direction,
            "control": control,
            "alpha": float(alpha),
            "target_mechanism": target_mech,
            "target_abs_shift": safe_mean(target["abs_shift"]) if len(target) else np.nan,
            "nontarget_abs_shift": safe_mean(nontarget["abs_shift"]) if len(nontarget) else np.nan,
            "specificity": safe_mean(target["abs_shift"]) - safe_mean(nontarget["abs_shift"]) if len(target) and len(nontarget) else np.nan,
            "target_signed_shift": safe_mean(target["shift"]) if len(target) else np.nan,
            "nontarget_signed_shift": safe_mean(nontarget["shift"]) if len(nontarget) else np.nan,
            "target_hit_rate": safe_mean(target["target_hit"]) if len(target) else np.nan,
            "nontarget_hit_rate": safe_mean(nontarget["target_hit"]) if len(nontarget) else np.nan,
            "n": int(len(g)),
        })
    return pd.DataFrame(rows), merged

def dose_response(spec_df):
    rows = []
    for keys, g in spec_df.groupby(["direction", "control"]):
        g = g.sort_values("alpha")
        x = g["alpha"].values.astype(float)
        y = g["specificity"].values.astype(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 3 and np.std(x[m]) > 1e-8:
            coef = np.polyfit(x[m], y[m], deg=1)
            pred = np.polyval(coef, x[m])
            r2 = r2_score(y[m], pred)
            slope = float(coef[0])
        else:
            slope, r2 = np.nan, np.nan
        rows.append({
            "direction": keys[0],
            "control": keys[1],
            "specificity_slope_vs_alpha": slope,
            "specificity_r2_vs_alpha": float(r2) if np.isfinite(r2) else np.nan,
        })
    return pd.DataFrame(rows)

def shuffle_significance(spec_df, cfg):
    rows = []
    main_control = "curvature_ridge"
    for direction in sorted(spec_df["direction"].unique()):
        for alpha in cfg.shuffle_alphas:
            main = spec_df[
                (spec_df["direction"] == direction) &
                (np.isclose(spec_df["alpha"], alpha)) &
                (spec_df["control"] == main_control)
            ]
            if len(main) == 0:
                continue
            proto_spec = float(main.iloc[0]["specificity"])

            sh = spec_df[
                (spec_df["direction"] == direction) &
                (np.isclose(spec_df["alpha"], alpha)) &
                (spec_df["control"].str.startswith("shuffle_curvature_ridge_"))
            ]["specificity"].values.astype(float)

            def spec_of(ctrl):
                r = spec_df[
                    (spec_df["direction"] == direction) &
                    (np.isclose(spec_df["alpha"], alpha)) &
                    (spec_df["control"] == ctrl)
                ]
                return float(r.iloc[0]["specificity"]) if len(r) else np.nan

            rand = spec_of("random")
            vel = spec_of("velocity_ridge")
            pos = spec_of("position_ridge")
            phase = spec_of("curvature_phase_delta")
            pls = spec_of("curvature_pls_like")

            mean = float(np.nanmean(sh)) if len(sh) else np.nan
            std = float(np.nanstd(sh) + 1e-8) if len(sh) else np.nan
            z = float((proto_spec - mean) / std) if len(sh) else np.nan
            pct = float(np.mean(sh <= proto_spec)) if len(sh) else np.nan
            q95 = float(np.quantile(sh, 0.95)) if len(sh) else np.nan

            rows.append({
                "direction": direction,
                "alpha": float(alpha),
                "curvature_ridge_specificity": proto_spec,
                "shuffle_specificity_mean": mean,
                "shuffle_specificity_std": std,
                "shuffle_z_specificity": z,
                "shuffle_percentile": pct,
                "shuffle_q95": q95,
                "proto_exceeds_shuffle_q95": bool(np.isfinite(proto_spec) and np.isfinite(q95) and proto_spec > q95),
                "random_specificity": rand,
                "velocity_ridge_specificity": vel,
                "position_ridge_specificity": pos,
                "curvature_phase_delta_specificity": phase,
                "curvature_pls_like_specificity": pls,
                "ridge_minus_random": proto_spec - rand if np.isfinite(rand) else np.nan,
                "ridge_minus_velocity": proto_spec - vel if np.isfinite(vel) else np.nan,
                "ridge_minus_position": proto_spec - pos if np.isfinite(pos) else np.nan,
                "ridge_minus_phase_delta": proto_spec - phase if np.isfinite(phase) else np.nan,
                "ridge_minus_pls_like": proto_spec - pls if np.isfinite(pls) else np.nan,
                "pass_lite": bool(proto_spec > 0 and np.isfinite(rand) and proto_spec > rand),
                "pass_strong_z2": bool(proto_spec > 0 and np.isfinite(z) and z > 2),
            })
    return pd.DataFrame(rows)

# ============================================================
# MAIN
# ============================================================

def main(cfg):
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "asa5e_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    model, tokenizer = load_model_and_tokenizer(cfg)
    df = build_dataset(tokenizer, cfg, save_dir)

    base_df, h_by_layer, pca_decision = extract_baseline(model, tokenizer, df, cfg, save_dir)
    train_idx, test_idx, split_df = make_split(base_df, cfg, save_dir)

    clean_anchor_R = {}
    clean_base = base_df[base_df["condition"] == "clean"].set_index("graph_id")
    for g, row in clean_base.iterrows():
        clean_anchor_R[int(g)] = {l: float(row[f"R_L{l}"]) for l in cfg.decision_layers}

    directions, shuffle_dirs = fit_directions(base_df, h_by_layer, train_idx, cfg, save_dir)

    test_df = base_df.iloc[test_idx].reset_index(drop=True)
    base_test = base_df.iloc[test_idx][["prompt_id", "DeltaU", "pair_choice"]].reset_index(drop=True)

    direction_list = []
    direction_targets = {}
    if cfg.run_q_to_s:
        direction_list.append("Q_to_S")
        direction_targets["Q_to_S"] = "closure"
    if cfg.run_k_to_s:
        direction_list.append("K_to_S")
        direction_targets["K_to_S"] = "competition"
    if cfg.run_s_to_q:
        direction_list.append("S_to_Q")
        direction_targets["S_to_Q"] = "stable"

    results = []
    for direction in direction_list:
        target_pair = "conflict" if direction == "S_to_Q" else "clean"
        sign = -1.0 if direction == "S_to_Q" else +1.0

        for alpha in cfg.alphas:
            control_maps = {}

            if cfg.include_curvature_ridge:
                control_maps["curvature_ridge"] = (directions["curvature_ridge"], list(directions["curvature_ridge"].keys()), "vector")
            if cfg.include_curvature_pls_like:
                control_maps["curvature_pls_like"] = (directions["curvature_pls_like"], list(directions["curvature_pls_like"].keys()), "vector")
            if cfg.include_phase_delta_control:
                control_maps["curvature_phase_delta"] = (directions["curvature_phase_delta"], list(directions["curvature_phase_delta"].keys()), "vector")
            if cfg.include_velocity_ridge:
                control_maps["velocity_ridge"] = (directions["velocity_ridge"], list(directions["velocity_ridge"].keys()), "vector")
            if cfg.include_position_ridge:
                control_maps["position_ridge"] = (directions["position_ridge"], list(directions["position_ridge"].keys()), "vector")
            if cfg.include_random_control:
                rand_layers = list(directions["curvature_ridge"].keys()) or sorted([l + 1 for l in cfg.accel_layers])
                rand_map = {l: directions["random"][l] for l in rand_layers if l in directions["random"]}
                control_maps["random"] = (rand_map, list(rand_map.keys()), "vector")
            if cfg.include_answer_control:
                ans_layers = list(directions["curvature_ridge"].keys()) or sorted([l + 1 for l in cfg.accel_layers])
                ans_map = {l: np.ones_like(next(iter(directions["random"].values()))) for l in ans_layers}
                control_maps["answer"] = (ans_map, ans_layers, "answer")

            if alpha in cfg.shuffle_alphas:
                for s, sh in shuffle_dirs.items():
                    if len(sh) > 0:
                        control_maps[f"shuffle_curvature_ridge_{s:03d}"] = (sh, list(sh.keys()), "vector")

            for control, (dmap, layers, mode) in control_maps.items():
                if len(layers) == 0:
                    print(f"[SKIP] control={control} has no valid layers")
                    continue
                signed_map = {l: (sign * v).astype(np.float32) for l, v in dmap.items()}
                print(f"[STEER] dir={direction} alpha={alpha} control={control}")
                out = forward_eval(
                    model=model,
                    tokenizer=tokenizer,
                    df_subset=test_df,
                    cfg=cfg,
                    pca_decision=pca_decision,
                    clean_anchor_R=clean_anchor_R,
                    direction_by_layer=signed_map,
                    layers_to_hook=layers,
                    alpha=float(alpha),
                    mode=mode,
                )
                out["direction"] = direction
                out["alpha"] = float(alpha)
                out["control"] = control
                out["target_pair_choice"] = target_pair
                results.append(out)

    results_df = pd.concat(results, ignore_index=True)
    results_df.to_csv(save_dir / "asa5e_steering_results.csv", index=False, encoding="utf-8-sig")

    spec_df, merged = summarize_specificity(results_df, base_test, direction_targets)
    spec_df.to_csv(save_dir / "asa5e_specificity_summary.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(save_dir / "asa5e_merged_shift_table.csv", index=False, encoding="utf-8-sig")

    dose_df = dose_response(spec_df)
    dose_df.to_csv(save_dir / "asa5e_dose_response.csv", index=False, encoding="utf-8-sig")

    sig_df = shuffle_significance(spec_df, cfg)
    sig_df.to_csv(save_dir / "asa5e_shuffle_specificity_significance.csv", index=False, encoding="utf-8-sig")

    verdict = sig_df[np.isclose(sig_df["alpha"], max(cfg.shuffle_alphas))].copy()
    verdict.to_csv(save_dir / "asa5e_verdict.csv", index=False, encoding="utf-8-sig")

    summary = {
        "audit": "ASA-5E Safe Curvature Direction Audit",
        "config": asdict(cfg),
        "n_test_prompts": int(len(test_df)),
        "deltaU_pca_explained_variance": float(pca_decision.explained_variance_ratio_[0]),
        "baseline_pair_choice_rate_test": base_test["pair_choice"].value_counts(normalize=True).to_dict(),
        "key_outputs": {
            "steering_results": "asa5e_steering_results.csv",
            "specificity": "asa5e_specificity_summary.csv",
            "dose_response": "asa5e_dose_response.csv",
            "shuffle_specificity_significance": "asa5e_shuffle_specificity_significance.csv",
            "verdict": "asa5e_verdict.csv",
            "direction_audit": "asa5e_direction_audit.csv",
        },
        "note": (
            "ASA-5E replaces near-zero residual-normalized acceleration directions with safe learned curvature directions. "
            "Main direction is curvature_ridge learned from a_l -> DeltaU. Directions with norm < min_dir_norm are dropped. "
            "Answer-token logits/ranks/margins are not used to construct curvature directions; DeltaU is used as the answerless target."
        )
    }
    with open(save_dir / "asa5e_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nASA-5E complete.")
    print(verdict)

if __name__ == "__main__":
    main(CFG)
