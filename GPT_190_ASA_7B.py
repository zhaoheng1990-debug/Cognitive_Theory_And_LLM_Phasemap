
# ============================================================
# ASA-7B: Best Combo Shuffle Validation
#
# Motivation:
#   ASA-7A found:
#
#       same_combo(alpha=0.3, beta=0.3)
#
#   outperforms operator_only and precursor_only, with small positive
#   additive synergy.
#
#   ASA-7B validates that the best combo is not merely the result of
#   stacking two nonspecific perturbations.
#
# Main test:
#   fixed combo:
#       operator = stable operator A_S
#       precursor = position_ridge w_DeltaU
#       op_alpha = 0.3
#       pre_beta = 0.3
#
# Controls:
#   - none
#   - operator_only
#   - precursor_only
#   - same_combo
#   - self_operator_combo          closure operator + precursor
#   - random_operator_combo        random transition operator + precursor
#   - operator_shuffle_precursor   stable operator + shuffled precursor
#   - shuffle_operator_precursor   shuffled stable operator + real precursor
#   - shuffle_both_combo           shuffled operator + shuffled precursor
#   - random_both_combo            random operator + random precursor
#   - answer
#
# Shuffle count:
#   default 50, only at best dose.
#
# Outputs:
#   asa7b_outputs/
#     asa7b_config.json
#     asa7b_dataset.csv
#     asa7b_baseline_features.csv
#     asa7b_operator_audit.csv
#     asa7b_direction_audit.csv
#     asa7b_steering_results.csv
#     asa7b_specificity_summary.csv
#     asa7b_shuffle_validation.csv
#     asa7b_verdict.csv
#     asa7b_summary.json
# ============================================================

import os
import gc
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
    save_dir: str = "./asa7b_outputs"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float16"
    seed: int = 42

    n_graphs: int = 96
    max_len: int = 260
    batch_size: int = 4
    test_size: float = 0.30

    # ASA-4 / ASA-7A effective operator window.
    operator_layers: Tuple[int, ...] = tuple(range(15, 20))

    # ASA-6B / ASA-7A effective precursor window.
    precursor_layers: Tuple[int, ...] = (17, 18, 19)

    decision_layers: Tuple[int, ...] = tuple(range(20, 26))

    # Best combo dose from ASA-7A.
    op_alpha: float = 0.30
    pre_beta: float = 0.30

    # Operator learning.
    pca_dim: int = 128
    operator_ridge_alpha: float = 10.0

    # Precursor learning.
    precursor_ridge_alpha: float = 10.0
    min_dir_norm: float = 1e-4

    # Validation shuffles.
    n_shuffles: int = 50

    include_answer_control: bool = True

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
    pd.DataFrame(rows).to_csv(save_dir / "asa7b_label_tokenization_audit.csv", index=False, encoding="utf-8-sig")
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
    df.to_csv(save_dir / "asa7b_dataset.csv", index=False, encoding="utf-8-sig")
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
    print("[EXTRACT] baseline hidden states and targets")
    W = get_lm_head_weight(model).detach().float()
    W_device = W.to(model.device)

    n = len(df)
    needed_layers = sorted(set(
        list(cfg.operator_layers)
        + [l+1 for l in cfg.operator_layers]
        + list(cfg.precursor_layers)
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
    out.to_csv(save_dir / "asa7b_baseline_features.csv", index=False, encoding="utf-8-sig")
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
    split_df.to_csv(save_dir / "asa7b_split.csv", index=False, encoding="utf-8-sig")
    return np.asarray(tr), np.asarray(te), split_df

# ============================================================
# OPERATOR FIT
# ============================================================

class LowDimOperator:
    def __init__(self, pca, ridge, layer):
        self.pca = pca
        self.ridge = ridge
        self.layer = layer

    def predict_torch(self, h_in: torch.Tensor) -> torch.Tensor:
        device = h_in.device
        dtype = h_in.dtype
        mean = torch.tensor(self.pca.mean_, device=device, dtype=dtype)
        comps = torch.tensor(self.pca.components_, device=device, dtype=dtype)
        coef = torch.tensor(self.ridge.coef_, device=device, dtype=dtype)
        intercept = torch.tensor(self.ridge.intercept_, device=device, dtype=dtype)
        z = (h_in - mean) @ comps.T
        z_next = z @ coef.T + intercept
        h_pred = z_next @ comps + mean
        return h_pred

def fit_operator_family(base_df, h_by_layer, train_idx, cfg, save_dir):
    operators = {}
    audits = []
    rng = np.random.default_rng(cfg.seed + 700)

    for l in cfg.operator_layers:
        H0 = h_by_layer[l]
        H1 = h_by_layer[l + 1]
        H_comb = np.concatenate([H0[train_idx], H1[train_idx]], axis=0)
        dim = min(cfg.pca_dim, H_comb.shape[0] - 1, H_comb.shape[1])
        pca = PCA(n_components=dim, random_state=cfg.seed)
        pca.fit(H_comb)
        z0_all = pca.transform(H0)
        z1_all = pca.transform(H1)

        operators[l] = {}

        for mech in ["stable", "closure"]:
            idx = np.asarray([i for i in train_idx if base_df.iloc[i]["mechanism"] == mech], dtype=int)
            reg = Ridge(alpha=cfg.operator_ridge_alpha)
            reg.fit(z0_all[idx], z1_all[idx])
            pred = reg.predict(z0_all[idx])
            r2 = r2_score(z1_all[idx], pred)
            corr = np.corrcoef(z1_all[idx].ravel(), pred.ravel())[0, 1] if np.std(pred) > 1e-8 else 0.0
            operators[l][mech] = LowDimOperator(pca, reg, l)
            audits.append({
                "layer": l,
                "operator_type": mech,
                "n_train": int(len(idx)),
                "pca_dim": int(dim),
                "pca_var_sum": float(np.sum(pca.explained_variance_ratio_)),
                "train_r2_znext": float(r2),
                "train_flat_corr_znext": float(corr),
            })

        # Random operator: random target permutation.
        perm = train_idx.copy()
        rng.shuffle(perm)
        reg = Ridge(alpha=cfg.operator_ridge_alpha)
        reg.fit(z0_all[train_idx], z1_all[perm])
        operators[l]["random"] = LowDimOperator(pca, reg, l)
        audits.append({
            "layer": l,
            "operator_type": "random_perm",
            "n_train": int(len(train_idx)),
            "pca_dim": int(dim),
            "pca_var_sum": float(np.sum(pca.explained_variance_ratio_)),
            "train_r2_znext": np.nan,
            "train_flat_corr_znext": np.nan,
        })

    # Shuffle stable operators: shuffled mechanism labels.
    shuffle_ops = {}
    for s in range(cfg.n_shuffles):
        rng_s = np.random.default_rng(cfg.seed + 10000 + s)
        shuffled_mech = base_df["mechanism"].values.copy()
        train_mech = shuffled_mech[train_idx].copy()
        rng_s.shuffle(train_mech)
        shuffled_mech[train_idx] = train_mech

        shuffle_ops[s] = {}
        for l in cfg.operator_layers:
            H0 = h_by_layer[l]
            H1 = h_by_layer[l + 1]
            pca = operators[l]["stable"].pca
            z0_all = pca.transform(H0)
            z1_all = pca.transform(H1)
            idx = train_idx[shuffled_mech[train_idx] == "stable"]
            if len(idx) < 4:
                idx = train_idx
            reg = Ridge(alpha=cfg.operator_ridge_alpha)
            reg.fit(z0_all[idx], z1_all[idx])
            shuffle_ops[s][l] = LowDimOperator(pca, reg, l)

    pd.DataFrame(audits).to_csv(save_dir / "asa7b_operator_audit.csv", index=False, encoding="utf-8-sig")
    return operators, shuffle_ops

# ============================================================
# PRECURSOR FIT
# ============================================================

def fit_position_ridge_directions(base_df, h_by_layer, train_idx, cfg, save_dir):
    y = base_df["DeltaU"].values.astype(np.float32)
    dirs = {}
    audits = []

    for l in cfg.precursor_layers:
        H = h_by_layer[l].astype(np.float32)
        X0 = H[train_idx] - H[train_idx].mean(axis=0, keepdims=True)
        y0 = y[train_idx] - y[train_idx].mean()
        reg = Ridge(alpha=cfg.precursor_ridge_alpha)
        reg.fit(X0, y0)
        coef = reg.coef_.astype(np.float32)
        unit, norm = normalize_np(coef, eps=cfg.min_dir_norm)
        pred = reg.predict(X0)
        dirs[l] = unit
        audits.append({
            "layer": l,
            "direction_type": "position_ridge",
            "coef_norm": norm,
            "valid": bool(unit is not None),
            "train_r2": float(r2_score(y0, pred)),
            "train_corr": float(np.corrcoef(y0, pred)[0, 1]) if np.std(pred) > 1e-8 else 0.0,
        })

    # Shuffled precursor directions.
    shuffle_dirs = {}
    for s in range(cfg.n_shuffles):
        rng_s = np.random.default_rng(cfg.seed + 20000 + s)
        y_perm = y[train_idx].copy()
        rng_s.shuffle(y_perm)
        sh = {}
        for l in cfg.precursor_layers:
            H = h_by_layer[l].astype(np.float32)
            X0 = H[train_idx] - H[train_idx].mean(axis=0, keepdims=True)
            y0 = y_perm - y_perm.mean()
            reg = Ridge(alpha=cfg.precursor_ridge_alpha)
            reg.fit(X0, y0)
            coef = reg.coef_.astype(np.float32)
            unit, norm = normalize_np(coef, eps=cfg.min_dir_norm)
            sh[l] = unit
            audits.append({
                "layer": l,
                "direction_type": f"shuffle_position_ridge_{s:03d}",
                "coef_norm": norm,
                "valid": bool(unit is not None),
                "train_r2": np.nan,
                "train_corr": np.nan,
            })
        shuffle_dirs[s] = sh

    # Random precursor.
    rng = np.random.default_rng(cfg.seed + 900)
    random_dirs = {}
    dim = h_by_layer[cfg.precursor_layers[0]].shape[1]
    for l in cfg.precursor_layers:
        rd = rng.standard_normal(dim).astype(np.float32)
        unit, norm = normalize_np(rd, eps=cfg.min_dir_norm)
        random_dirs[l] = unit
        audits.append({
            "layer": l,
            "direction_type": "random",
            "coef_norm": norm,
            "valid": True,
            "train_r2": np.nan,
            "train_corr": np.nan,
        })

    pd.DataFrame(audits).to_csv(save_dir / "asa7b_direction_audit.csv", index=False, encoding="utf-8-sig")
    return dirs, shuffle_dirs, random_dirs

# ============================================================
# HOOKS
# ============================================================

def make_combo_hook_factory(
    stable_ops,
    precursor_dirs,
    cfg,
    control_mode,
    W_device,
    clean_ids_batch=None,
    conflict_ids_batch=None,
):
    """
    control_mode determines which operator/direction maps are passed.

    For controls:
      same_combo: stable operator + real precursor
      self_operator_combo: closure operator + real precursor
      random_operator_combo: random operator + real precursor
      operator_shuffle_precursor: stable operator + shuffled precursor
      shuffle_operator_precursor: shuffled operator + real precursor
      shuffle_both_combo: shuffled operator + shuffled precursor
      random_both_combo: random operator + random precursor
    """
    pre_tensors = {
        int(l): torch.tensor(v, dtype=torch.float32, device=W_device.device)
        for l, v in precursor_dirs.items() if v is not None
    }

    def hook_factory(l):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                h_out = output[0]
                rest = output[1:]
            else:
                h_out = output
                rest = None

            B, S, D = h_out.shape
            pos = torch.full((B,), S - 1, dtype=torch.long, device=h_out.device)
            idx = torch.arange(B, device=h_out.device)

            h_in = inputs[0]
            h_real_last = h_out[idx, pos, :]
            h_in_last = h_in[idx, pos, :]
            h_new = h_real_last

            def apply_operator(h_base):
                if l not in stable_ops:
                    return h_base
                h_target = stable_ops[l].predict_torch(h_in_last.float()).to(h_base.dtype)
                return (1.0 - cfg.op_alpha) * h_base + cfg.op_alpha * h_target

            def apply_precursor(h_base):
                if l not in pre_tensors:
                    return h_base
                d = pre_tensors[l].to(h_base.device, h_base.dtype)
                scale = h_base.float().std(dim=-1, keepdim=True).to(h_base.dtype)
                return h_base + cfg.pre_beta * scale * d[None, :].expand(B, -1)

            def apply_answer(h_base):
                ans = W_device[clean_ids_batch].float() - W_device[conflict_ids_batch].float()
                ans = ans / (torch.norm(ans, dim=1, keepdim=True) + 1e-8)
                scale = h_base.float().std(dim=-1, keepdim=True).to(h_base.dtype)
                return h_base + cfg.pre_beta * scale * ans.to(h_base.dtype)

            if control_mode == "none":
                return output
            elif control_mode == "operator_only":
                h_new = apply_operator(h_new)
            elif control_mode == "precursor_only":
                h_new = apply_precursor(h_new)
            elif control_mode in (
                "same_combo",
                "self_operator_combo",
                "random_operator_combo",
                "operator_shuffle_precursor",
                "shuffle_operator_precursor",
                "shuffle_both_combo",
                "random_both_combo",
            ):
                h_new = apply_operator(h_new)
                h_new = apply_precursor(h_new)
            elif control_mode == "answer":
                h_new = apply_answer(h_new)
            else:
                raise ValueError(control_mode)

            if torch.allclose(h_new, h_real_last):
                return output

            h2 = h_out.clone()
            h2[idx, pos, :] = h_new

            if rest is not None:
                return (h2,) + rest
            return h2
        return hook
    return hook_factory

# ============================================================
# FORWARD EVAL
# ============================================================

def forward_eval(model, tokenizer, df_subset, cfg, pca_decision,
                 clean_anchor_R, op_map, pre_map, control_mode):
    W = get_lm_head_weight(model).detach().float()
    W_device = W.to(model.device)
    rows = []

    texts = df_subset["text"].tolist()
    clean_ids_all = df_subset["clean_token_id"].values.astype(np.int64)
    conflict_ids_all = df_subset["conflict_token_id"].values.astype(np.int64)
    layer_modules = get_layers(model)

    hook_layers = sorted(set(list(cfg.operator_layers) + list(cfg.precursor_layers)))

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
            if control_mode != "none":
                hook_factory = make_combo_hook_factory(
                    stable_ops=op_map,
                    precursor_dirs=pre_map,
                    cfg=cfg,
                    control_mode=control_mode,
                    W_device=W_device,
                    clean_ids_batch=cids,
                    conflict_ids_batch=eids,
                )
                for l in hook_layers:
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
# SUMMARY
# ============================================================

def summarize_specificity(results_df, base_test):
    merged = results_df.merge(
        base_test[["prompt_id", "DeltaU", "pair_choice"]],
        on="prompt_id",
        suffixes=("", "_base")
    )
    merged["shift"] = merged["DeltaU"] - merged["DeltaU_base"]
    merged["abs_shift"] = merged["shift"].abs()
    merged["target_hit"] = (merged["pair_choice"] == "clean").astype(int)

    rows = []
    for control, g in merged.groupby("control"):
        target = g[g["mechanism"] == "closure"]
        nontarget = g[g["mechanism"] != "closure"]
        rows.append({
            "control": control,
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

def shuffle_validation(spec_df):
    main = spec_df[spec_df["control"] == "same_combo"]
    if len(main) == 0:
        return pd.DataFrame()
    main_spec = float(main.iloc[0]["specificity"])

    rows = []
    for prefix in [
        "operator_shuffle_precursor",
        "shuffle_operator_precursor",
        "shuffle_both_combo",
    ]:
        vals = spec_df[spec_df["control"].str.startswith(prefix)]["specificity"].values.astype(float)
        if len(vals) == 0:
            continue
        mean = float(np.nanmean(vals))
        std = float(np.nanstd(vals) + 1e-8)
        z = float((main_spec - mean) / std)
        pct = float(np.mean(vals <= main_spec))
        q95 = float(np.quantile(vals, 0.95))
        rows.append({
            "control_family": prefix,
            "same_combo_specificity": main_spec,
            "shuffle_mean": mean,
            "shuffle_std": std,
            "shuffle_z": z,
            "shuffle_percentile": pct,
            "shuffle_q95": q95,
            "exceeds_q95": bool(main_spec > q95),
            "n": int(len(vals)),
        })

    # Also compare to deterministic nonspecific controls.
    for ctrl in ["self_operator_combo", "random_operator_combo", "random_both_combo", "operator_only", "precursor_only"]:
        r = spec_df[spec_df["control"] == ctrl]
        if len(r):
            rows.append({
                "control_family": ctrl,
                "same_combo_specificity": main_spec,
                "shuffle_mean": float(r.iloc[0]["specificity"]),
                "shuffle_std": np.nan,
                "shuffle_z": np.nan,
                "shuffle_percentile": np.nan,
                "shuffle_q95": np.nan,
                "exceeds_q95": bool(main_spec > float(r.iloc[0]["specificity"])),
                "n": 1,
            })

    return pd.DataFrame(rows)

# ============================================================
# MAIN
# ============================================================

def main(cfg):
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "asa7b_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    model, tokenizer = load_model_and_tokenizer(cfg)
    df = build_dataset(tokenizer, cfg, save_dir)

    base_df, h_by_layer, pca_decision = extract_baseline(model, tokenizer, df, cfg, save_dir)
    train_idx, test_idx, split_df = make_split(base_df, cfg, save_dir)

    clean_anchor_R = {}
    clean_base = base_df[base_df["condition"] == "clean"].set_index("graph_id")
    for g, row in clean_base.iterrows():
        clean_anchor_R[int(g)] = {l: float(row[f"R_L{l}"]) for l in cfg.decision_layers}

    op_family, shuffle_ops = fit_operator_family(base_df, h_by_layer, train_idx, cfg, save_dir)
    real_pre, shuffle_pre, random_pre = fit_position_ridge_directions(base_df, h_by_layer, train_idx, cfg, save_dir)

    test_df = base_df.iloc[test_idx].reset_index(drop=True)
    base_test = base_df.iloc[test_idx][["prompt_id", "DeltaU", "pair_choice"]].reset_index(drop=True)

    # Deterministic maps.
    stable_op = {l: op_family[l]["stable"] for l in cfg.operator_layers}
    closure_op = {l: op_family[l]["closure"] for l in cfg.operator_layers}
    random_op = {l: op_family[l]["random"] for l in cfg.operator_layers}

    runs = []
    runs.append(("none", {}, {}))
    runs.append(("operator_only", stable_op, {}))
    runs.append(("precursor_only", {}, real_pre))
    runs.append(("same_combo", stable_op, real_pre))
    runs.append(("self_operator_combo", closure_op, real_pre))
    runs.append(("random_operator_combo", random_op, real_pre))
    runs.append(("random_both_combo", random_op, random_pre))
    if cfg.include_answer_control:
        runs.append(("answer", {}, real_pre))

    # Shuffle families.
    for s in range(cfg.n_shuffles):
        runs.append((f"operator_shuffle_precursor_{s:03d}", stable_op, shuffle_pre[s]))
        runs.append((f"shuffle_operator_precursor_{s:03d}", shuffle_ops[s], real_pre))
        runs.append((f"shuffle_both_combo_{s:03d}", shuffle_ops[s], shuffle_pre[s]))

    results = []
    for control, op_map, pre_map in runs:
        print(f"[RUN] control={control}")
        out = forward_eval(
            model=model,
            tokenizer=tokenizer,
            df_subset=test_df,
            cfg=cfg,
            pca_decision=pca_decision,
            clean_anchor_R=clean_anchor_R,
            op_map=op_map,
            pre_map=pre_map,
            control_mode=("same_combo" if control.startswith(("operator_shuffle", "shuffle_operator", "shuffle_both")) else control),
        )
        out["control"] = control
        results.append(out)

    results_df = pd.concat(results, ignore_index=True)
    results_df.to_csv(save_dir / "asa7b_steering_results.csv", index=False, encoding="utf-8-sig")

    spec_df, merged = summarize_specificity(results_df, base_test)
    spec_df.to_csv(save_dir / "asa7b_specificity_summary.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(save_dir / "asa7b_merged_shift_table.csv", index=False, encoding="utf-8-sig")

    val_df = shuffle_validation(spec_df)
    val_df.to_csv(save_dir / "asa7b_shuffle_validation.csv", index=False, encoding="utf-8-sig")

    # Verdict heuristic.
    main_spec = float(spec_df[spec_df["control"] == "same_combo"].iloc[0]["specificity"])
    op_spec = float(spec_df[spec_df["control"] == "operator_only"].iloc[0]["specificity"])
    pre_spec = float(spec_df[spec_df["control"] == "precursor_only"].iloc[0]["specificity"])

    # Main shuffle z against both-shuffled is the strongest null.
    both = val_df[val_df["control_family"] == "shuffle_both_combo"]
    if len(both):
        both_z = float(both.iloc[0]["shuffle_z"])
        both_exceeds = bool(both.iloc[0]["exceeds_q95"])
    else:
        both_z = np.nan
        both_exceeds = False

    # Check precursor shuffle and operator shuffle too.
    op_sh = val_df[val_df["control_family"] == "shuffle_operator_precursor"]
    pre_sh = val_df[val_df["control_family"] == "operator_shuffle_precursor"]
    op_sh_z = float(op_sh.iloc[0]["shuffle_z"]) if len(op_sh) else np.nan
    pre_sh_z = float(pre_sh.iloc[0]["shuffle_z"]) if len(pre_sh) else np.nan

    gain_over_op = main_spec - op_spec
    gain_over_pre = main_spec - pre_spec
    additive_synergy = main_spec - op_spec - pre_spec

    if main_spec > op_spec and main_spec > pre_spec and both_z > 2 and both_exceeds:
        verdict = "PASS_STRONG_COMBO_VALIDATED"
    elif main_spec > op_spec and main_spec > pre_spec and both_z > 1.5:
        verdict = "PASS_LITE_COMBO_VALIDATED"
    elif main_spec > op_spec:
        verdict = "COMBO_GAIN_BUT_NULL_WEAK"
    else:
        verdict = "NO_COMBO_VALIDATION"

    verdict_df = pd.DataFrame([{
        "audit": "ASA-7B Best Combo Shuffle Validation",
        "verdict": verdict,
        "same_combo_specificity": main_spec,
        "operator_only_specificity": op_spec,
        "precursor_only_specificity": pre_spec,
        "gain_over_operator": gain_over_op,
        "gain_over_precursor": gain_over_pre,
        "additive_synergy": additive_synergy,
        "z_vs_shuffle_both": both_z,
        "z_vs_shuffle_operator": op_sh_z,
        "z_vs_shuffle_precursor": pre_sh_z,
        "exceeds_shuffle_both_q95": both_exceeds,
    }])
    verdict_df.to_csv(save_dir / "asa7b_verdict.csv", index=False, encoding="utf-8-sig")

    summary = {
        "audit": "ASA-7B Best Combo Shuffle Validation",
        "config": asdict(cfg),
        "n_test_prompts": int(len(test_df)),
        "deltaU_pca_explained_variance": float(pca_decision.explained_variance_ratio_[0]),
        "num_runs": int(len(runs)),
        "verdict": verdict,
        "main_metrics": {
            "same_combo_specificity": main_spec,
            "operator_only_specificity": op_spec,
            "precursor_only_specificity": pre_spec,
            "gain_over_operator": gain_over_op,
            "gain_over_precursor": gain_over_pre,
            "additive_synergy": additive_synergy,
            "z_vs_shuffle_both": both_z,
            "z_vs_shuffle_operator": op_sh_z,
            "z_vs_shuffle_precursor": pre_sh_z,
        },
        "key_outputs": {
            "steering_results": "asa7b_steering_results.csv",
            "specificity": "asa7b_specificity_summary.csv",
            "shuffle_validation": "asa7b_shuffle_validation.csv",
            "verdict": "asa7b_verdict.csv",
            "operator_audit": "asa7b_operator_audit.csv",
            "direction_audit": "asa7b_direction_audit.csv",
        },
        "note": (
            "ASA-7B validates the best ASA-7A same_combo at alpha=beta=0.3 against matched shuffle/null controls. "
            "The strongest null is shuffle_both_combo: shuffled operator plus shuffled precursor."
        )
    }

    with open(save_dir / "asa7b_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nASA-7B complete.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main(CFG)
