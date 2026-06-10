
# ============================================================
# ASA-5: Trajectory Acceleration Steering
#
# Motivation:
#   ASA-1/2/3 tested static state / predictor / flow direction steering.
#   ASA-4 tests local operator-matched steering.
#
#   ASA-5 tests a different hypothesis:
#       basin control may require sustained acceleration / curvature control
#       over a layer interval, not a single state/vector perturbation.
#
# Core objects:
#   position:      H_l
#   velocity:      v_l = H_{l+1} - H_l
#   acceleration:  a_l = H_{l+2} - 2H_{l+1} + H_l
#
# Main target:
#   Q -> S  (closure -> stable)
#
# Main steering window:
#   L15-L19, implemented as acceleration terms a_l for l=15..18
#   because a_l modifies H_{l+1}; for l=18 this modifies H19.
#
# Controls:
#   acceleration_delta: a_S - a_Q
#   acceleration_stable: a_S
#   velocity_delta: v_S - v_Q
#   position_delta: H_S - H_Q
#   random
#   shuffled acceleration deltas
#   answer direction positive control
#
# No answer-token logits/ranks/margins are used to construct steering directions.
# R_L and DeltaU are used only as evaluation targets.
#
# Outputs:
#   asa5_outputs/
#     asa5_config.json
#     asa5_dataset.csv
#     asa5_split.csv
#     asa5_baseline_features.csv
#     asa5_direction_audit.csv
#     asa5_steering_results.csv
#     asa5_specificity_summary.csv
#     asa5_dose_response.csv
#     asa5_shuffle_specificity_significance.csv
#     asa5_curvature_alignment.csv
#     asa5_verdict.csv
#     asa5_summary.json
# ============================================================

import os
import re
import gc
import json
import math
import random
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import GroupShuffleSplit
from sklearn.decomposition import PCA
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
    save_dir: str = "./asa5_outputs"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float16"
    seed: int = 42

    n_graphs: int = 96
    max_len: int = 260
    batch_size: int = 4
    test_size: float = 0.30

    # Qwen-1.5B layer conventions.
    # We need H_l,H_{l+1},H_{l+2} for acceleration.
    accel_layers: Tuple[int, ...] = tuple(range(15, 19))   # a_l modifies H_{l+1}, l=15..18
    velocity_layers: Tuple[int, ...] = tuple(range(15, 20)) # v_l modifies H_{l+1}, l=15..19
    position_layers: Tuple[int, ...] = tuple(range(15, 20)) # position perturb at layer output l

    decision_layers: Tuple[int, ...] = tuple(range(20, 26))

    alphas: Tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30)

    # Main direction: Q -> S
    run_q_to_s: bool = True
    run_k_to_s: bool = False
    run_s_to_q: bool = False

    # Residualize against shared PCA subspace of position/velocity/accel directions.
    shared_pca_dim: int = 8

    # Shuffle controls.
    n_shuffles: int = 50
    shuffle_alphas: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)

    include_answer_control: bool = True
    include_random_control: bool = True
    include_position_control: bool = True
    include_velocity_control: bool = True
    include_acceleration_control: bool = True

CFG = Config()

# ============================================================
# SEED / PATH
# ============================================================

def set_seed(seed: int):
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
    pd.DataFrame(rows).to_csv(save_dir / "asa5_label_tokenization_audit.csv", index=False, encoding="utf-8-sig")
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
    df.to_csv(save_dir / "asa5_dataset.csv", index=False, encoding="utf-8-sig")
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

def normalize_np(v):
    return v / (np.linalg.norm(v) + 1e-8)

def remove_shared_np(v, basis):
    if basis is None or basis.size == 0:
        return v
    return v - basis.T @ (basis @ v)

# ============================================================
# BASELINE EXTRACTION
# ============================================================

def extract_baseline(model, tokenizer, df, cfg, save_dir):
    print("[EXTRACT] baseline hidden states / targets")
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
    out.to_csv(save_dir / "asa5_baseline_features.csv", index=False, encoding="utf-8-sig")
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
    split_df.to_csv(save_dir / "asa5_split.csv", index=False, encoding="utf-8-sig")
    return np.asarray(tr), np.asarray(te), split_df

# ============================================================
# DIRECTION CONSTRUCTION
# ============================================================

def mean_by_mechanism(X, base_df, train_idx, mech):
    idx = [i for i in train_idx if base_df.iloc[i]["mechanism"] == mech]
    idx = np.asarray(idx, dtype=int)
    return X[idx].mean(axis=0)

def fit_directions(base_df, h_by_layer, train_idx, cfg, save_dir):
    """
    Build per-layer position/velocity/acceleration directions.
    For Q->S:
        delta_pos = mean(H_l|S) - mean(H_l|Q)
        delta_vel = mean(v_l|S) - mean(v_l|Q)
        delta_acc = mean(a_l|S) - mean(a_l|Q)
    and residualize versions by removing shared PCA subspace.
    """
    directions = {
        "position_delta": {},
        "velocity_delta": {},
        "acceleration_delta": {},
        "acceleration_stable": {},
        "acceleration_residual": {},
        "velocity_residual": {},
        "position_residual": {},
        "random": {},
    }
    audits = []
    all_vecs = []

    for l in cfg.position_layers:
        H = h_by_layer[l]
        d = mean_by_mechanism(H, base_df, train_idx, "stable") - mean_by_mechanism(H, base_df, train_idx, "closure")
        all_vecs.append(d[None, :])
    for l in cfg.velocity_layers:
        V = h_by_layer[l + 1] - h_by_layer[l]
        d = mean_by_mechanism(V, base_df, train_idx, "stable") - mean_by_mechanism(V, base_df, train_idx, "closure")
        all_vecs.append(d[None, :])
    for l in cfg.accel_layers:
        A = h_by_layer[l + 2] - 2 * h_by_layer[l + 1] + h_by_layer[l]
        d = mean_by_mechanism(A, base_df, train_idx, "stable") - mean_by_mechanism(A, base_df, train_idx, "closure")
        all_vecs.append(d[None, :])

    all_vecs = np.concatenate(all_vecs, axis=0).astype(np.float32)
    pca_shared = PCA(n_components=min(cfg.shared_pca_dim, all_vecs.shape[0], all_vecs.shape[1]))
    pca_shared.fit(all_vecs)
    shared_basis = pca_shared.components_.astype(np.float32)

    rng = np.random.default_rng(cfg.seed + 555)

    # Position
    for l in cfg.position_layers:
        H = h_by_layer[l]
        pos_delta = mean_by_mechanism(H, base_df, train_idx, "stable") - mean_by_mechanism(H, base_df, train_idx, "closure")
        pos_res = remove_shared_np(pos_delta, shared_basis)
        directions["position_delta"][l] = normalize_np(pos_delta).astype(np.float32)
        directions["position_residual"][l] = normalize_np(pos_res).astype(np.float32)

    # Velocity
    for l in cfg.velocity_layers:
        V = h_by_layer[l + 1] - h_by_layer[l]
        vel_delta = mean_by_mechanism(V, base_df, train_idx, "stable") - mean_by_mechanism(V, base_df, train_idx, "closure")
        vel_res = remove_shared_np(vel_delta, shared_basis)
        directions["velocity_delta"][l] = normalize_np(vel_delta).astype(np.float32)
        directions["velocity_residual"][l] = normalize_np(vel_res).astype(np.float32)

    # Acceleration
    for l in cfg.accel_layers:
        A = h_by_layer[l + 2] - 2 * h_by_layer[l + 1] + h_by_layer[l]
        acc_s = mean_by_mechanism(A, base_df, train_idx, "stable")
        acc_q = mean_by_mechanism(A, base_df, train_idx, "closure")
        acc_delta = acc_s - acc_q
        acc_res = remove_shared_np(acc_delta, shared_basis)

        directions["acceleration_delta"][l + 1] = normalize_np(acc_delta).astype(np.float32)  # hook at H_{l+1}
        directions["acceleration_stable"][l + 1] = normalize_np(acc_s).astype(np.float32)
        directions["acceleration_residual"][l + 1] = normalize_np(acc_res).astype(np.float32)

        audits.append({
            "accel_source_l": l,
            "hook_layer": l + 1,
            "norm_acc_stable": float(np.linalg.norm(acc_s)),
            "norm_acc_closure": float(np.linalg.norm(acc_q)),
            "norm_acc_delta": float(np.linalg.norm(acc_delta)),
            "norm_acc_residual": float(np.linalg.norm(acc_res)),
            "cos_accS_accQ": float(np.dot(normalize_np(acc_s), normalize_np(acc_q))),
            "cos_delta_residual": float(np.dot(normalize_np(acc_delta), normalize_np(acc_res))),
        })

    # Random per hook layer.
    hook_layers = set(list(cfg.position_layers) + [l+1 for l in cfg.velocity_layers] + [l+1 for l in cfg.accel_layers])
    dim = next(iter(h_by_layer.values())).shape[1]
    for l in hook_layers:
        rd = rng.standard_normal(dim).astype(np.float32)
        directions["random"][l] = normalize_np(rd).astype(np.float32)

    # Shuffle acceleration residual directions.
    shuffle_dirs = {}
    for s in range(cfg.n_shuffles):
        rng_s = np.random.default_rng(cfg.seed + 10000 + s)
        shuffled_mech = base_df["mechanism"].values.copy()
        train_mech = shuffled_mech[train_idx].copy()
        rng_s.shuffle(train_mech)
        shuffled_mech[train_idx] = train_mech

        sh = {}
        for l in cfg.accel_layers:
            A = h_by_layer[l + 2] - 2 * h_by_layer[l + 1] + h_by_layer[l]
            stable_idx = train_idx[shuffled_mech[train_idx] == "stable"]
            closure_idx = train_idx[shuffled_mech[train_idx] == "closure"]
            if len(stable_idx) < 3 or len(closure_idx) < 3:
                stable_idx = train_idx
                closure_idx = train_idx
            acc_delta = A[stable_idx].mean(axis=0) - A[closure_idx].mean(axis=0)
            acc_res = remove_shared_np(acc_delta, shared_basis)
            sh[l + 1] = normalize_np(acc_res).astype(np.float32)
        shuffle_dirs[s] = sh

    pd.DataFrame(audits).to_csv(save_dir / "asa5_direction_audit.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        save_dir / "asa5_dirs_and_shared_basis.npz",
        shared_basis=shared_basis,
        **{f"acc_res_L{l}": v for l, v in directions["acceleration_residual"].items()},
        **{f"acc_delta_L{l}": v for l, v in directions["acceleration_delta"].items()},
        **{f"vel_res_L{l}": v for l, v in directions["velocity_residual"].items()},
        **{f"pos_res_L{l}": v for l, v in directions["position_residual"].items()},
    )
    return directions, shuffle_dirs, shared_basis

# ============================================================
# STEERED FORWARD
# ============================================================

def make_vector_hook(direction_by_layer, alpha, mode, W_device, clean_ids_batch=None, conflict_ids_batch=None):
    dir_tensors = {
        int(l): torch.tensor(v, dtype=torch.float32, device=W_device.device)
        for l, v in direction_by_layer.items()
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

            # Scale by last-token hidden std, not norm.
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
            "target_abs_shift": float(target["abs_shift"].mean()) if len(target) else np.nan,
            "nontarget_abs_shift": float(nontarget["abs_shift"].mean()) if len(nontarget) else np.nan,
            "specificity": float(target["abs_shift"].mean() - nontarget["abs_shift"].mean()) if len(target) and len(nontarget) else np.nan,
            "target_signed_shift": float(target["shift"].mean()) if len(target) else np.nan,
            "nontarget_signed_shift": float(nontarget["shift"].mean()) if len(nontarget) else np.nan,
            "target_hit_rate": float(target["target_hit"].mean()) if len(target) else np.nan,
            "nontarget_hit_rate": float(nontarget["target_hit"].mean()) if len(nontarget) else np.nan,
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
    main_control = "acceleration_residual"
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
                (spec_df["control"].str.startswith("shuffle_acceleration_residual_"))
            ]["specificity"].values.astype(float)

            random = spec_df[
                (spec_df["direction"] == direction) &
                (np.isclose(spec_df["alpha"], alpha)) &
                (spec_df["control"] == "random")
            ]
            rand_spec = float(random.iloc[0]["specificity"]) if len(random) else np.nan

            vel = spec_df[
                (spec_df["direction"] == direction) &
                (np.isclose(spec_df["alpha"], alpha)) &
                (spec_df["control"] == "velocity_residual")
            ]
            vel_spec = float(vel.iloc[0]["specificity"]) if len(vel) else np.nan

            pos = spec_df[
                (spec_df["direction"] == direction) &
                (np.isclose(spec_df["alpha"], alpha)) &
                (spec_df["control"] == "position_residual")
            ]
            pos_spec = float(pos.iloc[0]["specificity"]) if len(pos) else np.nan

            mean = float(np.nanmean(sh)) if len(sh) else np.nan
            std = float(np.nanstd(sh) + 1e-8) if len(sh) else np.nan
            z = float((proto_spec - mean) / std) if len(sh) else np.nan

            rows.append({
                "direction": direction,
                "alpha": float(alpha),
                "acceleration_specificity": proto_spec,
                "shuffle_specificity_mean": mean,
                "shuffle_specificity_std": std,
                "shuffle_z_specificity": z,
                "random_specificity": rand_spec,
                "velocity_specificity": vel_spec,
                "position_specificity": pos_spec,
                "acc_minus_random": proto_spec - rand_spec if np.isfinite(rand_spec) else np.nan,
                "acc_minus_velocity": proto_spec - vel_spec if np.isfinite(vel_spec) else np.nan,
                "acc_minus_position": proto_spec - pos_spec if np.isfinite(pos_spec) else np.nan,
                "pass_lite": bool(proto_spec > 0 and np.isfinite(rand_spec) and proto_spec > rand_spec),
                "pass_strong_z2": bool(proto_spec > 0 and np.isfinite(z) and z > 2),
            })
    return pd.DataFrame(rows)

# ============================================================
# MAIN
# ============================================================

def main(cfg):
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "asa5_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    model, tokenizer = load_model_and_tokenizer(cfg)
    df = build_dataset(tokenizer, cfg, save_dir)

    base_df, h_by_layer, pca_decision = extract_baseline(model, tokenizer, df, cfg, save_dir)
    train_idx, test_idx, split_df = make_split(base_df, cfg, save_dir)

    clean_anchor_R = {}
    clean_base = base_df[base_df["condition"] == "clean"].set_index("graph_id")
    for g, row in clean_base.iterrows():
        clean_anchor_R[int(g)] = {l: float(row[f"R_L{l}"]) for l in cfg.decision_layers}

    directions, shuffle_dirs, shared_basis = fit_directions(base_df, h_by_layer, train_idx, cfg, save_dir)

    test_df = base_df.iloc[test_idx].reset_index(drop=True)
    base_test = base_df.iloc[test_idx][["prompt_id", "DeltaU", "pair_choice"]].reset_index(drop=True)

    # Main directions.
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

            if cfg.include_acceleration_control:
                control_maps["acceleration_delta"] = (directions["acceleration_delta"], list(directions["acceleration_delta"].keys()), "vector")
                control_maps["acceleration_residual"] = (directions["acceleration_residual"], list(directions["acceleration_residual"].keys()), "vector")
                control_maps["acceleration_stable"] = (directions["acceleration_stable"], list(directions["acceleration_stable"].keys()), "vector")

            if cfg.include_velocity_control:
                control_maps["velocity_delta"] = (directions["velocity_delta"], list(directions["velocity_delta"].keys()), "vector")
                control_maps["velocity_residual"] = (directions["velocity_residual"], list(directions["velocity_residual"].keys()), "vector")

            if cfg.include_position_control:
                control_maps["position_delta"] = (directions["position_delta"], list(directions["position_delta"].keys()), "vector")
                control_maps["position_residual"] = (directions["position_residual"], list(directions["position_residual"].keys()), "vector")

            if cfg.include_random_control:
                # Use acceleration hook layers for random baseline to match main control.
                rand_layers = list(directions["acceleration_residual"].keys())
                rand_map = {l: directions["random"][l] for l in rand_layers}
                control_maps["random"] = (rand_map, rand_layers, "vector")

            if cfg.include_answer_control:
                ans_layers = list(directions["acceleration_residual"].keys())
                # direction map not used for answer, but hook expects valid layers.
                ans_map = {l: directions["acceleration_residual"][l] for l in ans_layers}
                control_maps["answer"] = (ans_map, ans_layers, "answer")

            if alpha in cfg.shuffle_alphas:
                for s, sh in shuffle_dirs.items():
                    control_maps[f"shuffle_acceleration_residual_{s:03d}"] = (sh, list(sh.keys()), "vector")

            for control, (dmap, layers, mode) in control_maps.items():
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
    results_df.to_csv(save_dir / "asa5_steering_results.csv", index=False, encoding="utf-8-sig")

    spec_df, merged = summarize_specificity(results_df, base_test, direction_targets)
    spec_df.to_csv(save_dir / "asa5_specificity_summary.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(save_dir / "asa5_merged_shift_table.csv", index=False, encoding="utf-8-sig")

    dose_df = dose_response(spec_df)
    dose_df.to_csv(save_dir / "asa5_dose_response.csv", index=False, encoding="utf-8-sig")

    sig_df = shuffle_significance(spec_df, cfg)
    sig_df.to_csv(save_dir / "asa5_shuffle_specificity_significance.csv", index=False, encoding="utf-8-sig")

    verdict = sig_df[np.isclose(sig_df["alpha"], max(cfg.shuffle_alphas))].copy()
    verdict.to_csv(save_dir / "asa5_verdict.csv", index=False, encoding="utf-8-sig")

    summary = {
        "audit": "ASA-5 Trajectory Acceleration Steering",
        "config": asdict(cfg),
        "n_test_prompts": int(len(test_df)),
        "deltaU_pca_explained_variance": float(pca_decision.explained_variance_ratio_[0]),
        "baseline_pair_choice_rate_test": base_test["pair_choice"].value_counts(normalize=True).to_dict(),
        "key_outputs": {
            "steering_results": "asa5_steering_results.csv",
            "specificity": "asa5_specificity_summary.csv",
            "dose_response": "asa5_dose_response.csv",
            "shuffle_specificity_significance": "asa5_shuffle_specificity_significance.csv",
            "verdict": "asa5_verdict.csv",
            "direction_audit": "asa5_direction_audit.csv",
        },
        "note": (
            "ASA-5 tests sustained acceleration/curvature steering over L15-L19. "
            "Directions are built from position/velocity/acceleration differences between stable and closure trajectories. "
            "Answer-token logits/ranks/margins are not used to build steering directions. "
            "R_L and DeltaU are evaluation targets only."
        )
    }
    with open(save_dir / "asa5_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nASA-5 complete.")
    print(verdict)

if __name__ == "__main__":
    main(CFG)
