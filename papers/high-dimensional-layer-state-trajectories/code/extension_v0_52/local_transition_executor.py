#!/usr/bin/env python
"""Mechanism-conditioned local transition executor used by the public control rerun.

The implementation fits mechanism labels, local transition operators and a train-only
action policy. Its output endpoint is a bounded internal or candidate-boundary effect.
"""

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
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

QWEN_PATH = r"Qwen/Qwen2.5-1.5B-Instruct"
LLAMA_PATH = r"meta-llama/Llama-3.2-1B-Instruct"
GEMMA_PATH = r"google/gemma-2-2b-it"

MODEL_PATHS = {
    "qwen": QWEN_PATH,
    "llama": LLAMA_PATH,
    "gemma": GEMMA_PATH,
}

@dataclass
class Config:
    model_key: str = "qwen"
    model_path: str = "auto"
    save_dir: str = "./local_transition_outputs"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float16"
    seed: int = 42

    n_graphs: int = 96
    max_len: int = 260
    batch_size: int = 4
    test_size: float = 0.30

    operator_layers: Tuple[int, ...] = tuple(range(15, 20))
    precursor_layers: Tuple[int, ...] = (17, 18, 19)
    decision_layers: Tuple[int, ...] = tuple(range(20, 26))

    pca_dim: int = 128
    operator_ridge_alpha: float = 10.0
    probe_ridge_alpha: float = 10.0
    precursor_ridge_alpha: float = 10.0
    min_dir_norm: float = 1e-4

    fixed_alpha: float = 0.30
    fixed_beta: float = 0.30

    policy_actions: Tuple[Tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.1, 0.1),
        (0.2, 0.2),
        (0.3, 0.3),
        (0.3, 0.0),
        (0.0, 0.3),
        (0.3, 0.2),
        (0.2, 0.3),
    )

    mechanism_clf_type: str = "random_forest"
    policy_clf_type: str = "random_forest"

    # Shuffle validation.
    n_shuffles: int = 30

    # To control compute, each shuffle evaluates only these null families.
    eval_shuffle_mech_policy: bool = True
    eval_shuffle_policy_label: bool = True
    eval_shuffle_both: bool = True

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

MECH_TO_ID = {"stable": 0, "competition": 1, "closure": 2}
ID_TO_MECH = {v: k for k, v in MECH_TO_ID.items()}

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
    pd.DataFrame(rows).to_csv(save_dir / "local_transition_label_tokenization_audit.csv", index=False, encoding="utf-8-sig")
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
                mech = mechanism_of(cond)
                rows.append({
                    "prompt_id": f"g{graph_count:03d}_{cond}",
                    "graph_id": graph_count,
                    "entity_id": ent_i,
                    "relation_id": rel_i,
                    "condition": cond,
                    "mechanism": mech,
                    "mechanism_id": MECH_TO_ID[mech],
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
    df.to_csv(save_dir / "local_transition_dataset.csv", index=False, encoding="utf-8-sig")
    return df

# ============================================================
# MODEL UTILS
# ============================================================

def load_model_and_tokenizer(cfg):
    cfg.model_path = resolve_model_path(cfg)
    print(f"[MODEL] model_key={cfg.model_key} resolved_path={cfg.model_path}")
    tok = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
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

def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) < 1e-12 or np.std(y[m]) < 1e-12:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])

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
    out["D_B_proxy"] = np.min(np.abs(X_dec), axis=1).astype(np.float32)
    out["R_final_margin"] = final_clean_logits - final_conflict_logits

    out.to_csv(save_dir / "local_transition_baseline_features.csv", index=False, encoding="utf-8-sig")
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
    split_df.to_csv(save_dir / "local_transition_split.csv", index=False, encoding="utf-8-sig")
    return np.asarray(tr), np.asarray(te), split_df

# ============================================================
# OPERATOR / PROBES
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

def fit_stable_operators(base_df, h_by_layer, train_idx, cfg, save_dir):
    operators = {}
    audits = []
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
                "layer": l, "mechanism": mech, "n_train": int(len(idx)),
                "pca_dim": int(dim), "pca_var_sum": float(np.sum(pca.explained_variance_ratio_)),
                "train_r2_znext": float(r2), "train_flat_corr_znext": float(corr),
            })
    pd.DataFrame(audits).to_csv(save_dir / "local_transition_operator_audit.csv", index=False, encoding="utf-8-sig")
    return operators

def fit_ridge_vector(X_train, y_train, alpha, min_norm):
    X = X_train.astype(np.float32)
    y = y_train.astype(np.float32)
    X_mean = X.mean(axis=0, keepdims=True)
    y_mean = float(y.mean())
    X0 = X - X_mean
    y0 = y - y_mean
    reg = Ridge(alpha=alpha)
    reg.fit(X0, y0)
    coef = reg.coef_.astype(np.float32)
    unit, norm = normalize_np(coef, eps=min_norm)
    pred = reg.predict(X0)
    return {
        "coef": coef, "unit": unit, "norm": norm,
        "x_mean": X_mean.astype(np.float32).reshape(-1),
        "y_mean": y_mean,
        "train_r2": float(r2_score(y0, pred)),
        "train_corr": safe_corr(y0, pred),
    }

def fit_precursor_and_probes(base_df, h_by_layer, train_idx, cfg, save_dir):
    y_du = base_df["DeltaU"].values.astype(np.float32)
    y_db = base_df["D_B_proxy"].values.astype(np.float32)
    dirs = {}
    probes = {}
    audits = []
    vel_stats = {}

    for l in cfg.precursor_layers:
        H = h_by_layer[l].astype(np.float32)
        du_fit = fit_ridge_vector(H[train_idx], y_du[train_idx], cfg.precursor_ridge_alpha, cfg.min_dir_norm)
        dirs[l] = du_fit["unit"]
        probes[(l, "DeltaU")] = du_fit
        db_fit = fit_ridge_vector(H[train_idx], y_db[train_idx], cfg.probe_ridge_alpha, cfg.min_dir_norm)
        probes[(l, "D_B")] = db_fit
        audits.append({"layer": l, "target": "DeltaU", "coef_norm": du_fit["norm"], "valid": bool(du_fit["unit"] is not None), "train_r2": du_fit["train_r2"], "train_corr": du_fit["train_corr"]})
        audits.append({"layer": l, "target": "D_B", "coef_norm": db_fit["norm"], "valid": bool(db_fit["unit"] is not None), "train_r2": db_fit["train_r2"], "train_corr": db_fit["train_corr"]})

    for l in cfg.operator_layers:
        H = h_by_layer[l].astype(np.float32)
        if (l, "DeltaU") not in probes:
            du_fit = fit_ridge_vector(H[train_idx], y_du[train_idx], cfg.probe_ridge_alpha, cfg.min_dir_norm)
            db_fit = fit_ridge_vector(H[train_idx], y_db[train_idx], cfg.probe_ridge_alpha, cfg.min_dir_norm)
            probes[(l, "DeltaU")] = du_fit
            probes[(l, "D_B")] = db_fit
            audits.append({"layer": l, "target": "DeltaU", "coef_norm": du_fit["norm"], "valid": bool(du_fit["unit"] is not None), "train_r2": du_fit["train_r2"], "train_corr": du_fit["train_corr"]})
            audits.append({"layer": l, "target": "D_B", "coef_norm": db_fit["norm"], "valid": bool(db_fit["unit"] is not None), "train_r2": db_fit["train_r2"], "train_corr": db_fit["train_corr"]})

        V = h_by_layer[l + 1] - h_by_layer[l]
        vn = np.linalg.norm(V[train_idx], axis=1)
        vel_stats[l] = {
            "mean": float(np.mean(vn)),
            "std": float(np.std(vn) + 1e-8),
            "q50": float(np.quantile(vn, 0.50)),
        }

    pd.DataFrame(audits).to_csv(save_dir / "local_transition_probe_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{
        "layer": l, "direction": "position_ridge_DeltaU",
        "valid": bool(dirs.get(l) is not None),
        "norm": float(np.linalg.norm(dirs[l])) if dirs.get(l) is not None else np.nan,
    } for l in cfg.precursor_layers]).to_csv(save_dir / "local_transition_direction_audit.csv", index=False, encoding="utf-8-sig")
    return dirs, probes, vel_stats

# ============================================================
# FEATURE / CLASSIFIERS / POLICY
# ============================================================

def predict_linear_np(H, fit):
    X = H.astype(np.float32)
    return ((X - fit["x_mean"][None, :]) @ fit["coef"] + fit["y_mean"]).astype(np.float32)

def base_state_features_for_samples(base_df, h_by_layer, probes, vel_stats, sample_idx, cfg):
    rows = []
    for idx in sample_idx:
        row = base_df.iloc[idx]
        for l in cfg.operator_layers:
            H = h_by_layer[l][idx:idx+1]
            du_hat = float(predict_linear_np(H, probes[(l, "DeltaU")])[0])
            db_hat = float(predict_linear_np(H, probes[(l, "D_B")])[0])
            if l + 1 in h_by_layer:
                vel = float(np.linalg.norm(h_by_layer[l + 1][idx] - h_by_layer[l][idx]))
            else:
                vel = vel_stats[l]["mean"]
            vs = vel_stats[l]
            rows.append({
                "prompt_id": row["prompt_id"],
                "graph_id": int(row["graph_id"]),
                "sample_index": int(idx),
                "condition": row["condition"],
                "mechanism": row["mechanism"],
                "mechanism_id": int(row["mechanism_id"]),
                "layer": int(l),
                "du_hat": du_hat,
                "abs_du_hat": abs(du_hat),
                "db_hat": db_hat,
                "vel_norm": vel,
                "vel_z": (vel - vs["mean"]) / vs["std"],
                "layer_norm": (l - min(cfg.operator_layers)) / max(1, (max(cfg.operator_layers) - min(cfg.operator_layers))),
                "is_precursor_layer": int(l in cfg.precursor_layers),
                "baseline_DeltaU": float(row["DeltaU"]),
                "baseline_DB": float(row["D_B_proxy"]),
            })
    return pd.DataFrame(rows)

MECH_FEATURE_COLS = [
    "du_hat", "abs_du_hat", "db_hat", "vel_norm", "vel_z",
    "layer_norm", "is_precursor_layer", "baseline_DeltaU", "baseline_DB",
]

POLICY_BASE_COLS = MECH_FEATURE_COLS
POLICY_PRED_MECH_COLS = POLICY_BASE_COLS + [
    "p_mech_stable", "p_mech_competition", "p_mech_closure",
    "pred_mech_stable", "pred_mech_competition", "pred_mech_closure",
]

def fit_mechanism_classifier(train_features, cfg, save_dir, shuffle_labels=False, shuffle_id=0):
    X = train_features[MECH_FEATURE_COLS].values.astype(np.float32)
    y = train_features["mechanism_id"].values.astype(int).copy()
    if shuffle_labels:
        rng = np.random.default_rng(cfg.seed + 1000 + shuffle_id)
        rng.shuffle(y)

    clf = RandomForestClassifier(
        n_estimators=250,
        max_depth=7,
        min_samples_leaf=6,
        class_weight="balanced_subsample",
        random_state=cfg.seed + shuffle_id,
    )
    clf.fit(X, y)
    pred = clf.predict(X)

    audit = {
        "shuffle_labels": bool(shuffle_labels),
        "shuffle_id": int(shuffle_id),
        "train_accuracy": float(accuracy_score(y, pred)),
        "train_macro_f1": float(f1_score(y, pred, average="macro")),
        "n_rows": int(len(y)),
    }
    return clf, audit

def attach_predicted_mechanism(features, clf):
    X = features[MECH_FEATURE_COLS].values.astype(np.float32)
    pred = clf.predict(X)
    prob = clf.predict_proba(X)
    out = features.copy()
    out["pred_mechanism_id"] = pred
    out["pred_mechanism"] = [ID_TO_MECH[int(x)] for x in pred]
    for k in range(3):
        out[f"p_mech_{ID_TO_MECH[k]}"] = prob[:, k]
        out[f"pred_mech_{ID_TO_MECH[k]}"] = (pred == k).astype(int)
    return out

def action_to_label(action):
    return f"a{action[0]:.1f}_b{action[1]:.1f}"

def fit_gain_policy(train_features, grid_labels, feature_cols, cfg, shuffle_policy_labels=False, shuffle_id=0):
    df = train_features.merge(grid_labels[["prompt_id", "best_action_label"]], on="prompt_id", how="left")
    df = df.dropna(subset=["best_action_label"]).copy()
    X = df[feature_cols].values.astype(np.float32)
    y = df["best_action_label"].values.copy()

    if shuffle_policy_labels:
        rng = np.random.default_rng(cfg.seed + 2000 + shuffle_id)
        rng.shuffle(y)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        random_state=cfg.seed + 100 + shuffle_id,
    )
    clf.fit(X, y)
    pred = clf.predict(X)
    audit = {
        "shuffle_policy_labels": bool(shuffle_policy_labels),
        "shuffle_id": int(shuffle_id),
        "n_train_rows": int(len(df)),
        "n_classes": int(len(np.unique(y))),
        "train_accuracy": float(accuracy_score(y, pred)),
        "train_macro_f1": float(f1_score(y, pred, average="macro")),
        "class_counts": pd.Series(y).value_counts().to_dict(),
    }
    return clf, audit

def policy_predict_actions(clf, feature_rows, feature_cols):
    X = feature_rows[feature_cols].values.astype(np.float32)
    labels = clf.predict(X)
    prob = clf.predict_proba(X) if hasattr(clf, "predict_proba") else np.ones((len(labels),1))
    conf = prob.max(axis=1)

    actions = []
    for lab, c in zip(labels, conf):
        try:
            a_str, b_str = lab.split("_")
            a = float(a_str[1:])
            b = float(b_str[1:])
        except Exception:
            a, b = CFG.fixed_alpha, CFG.fixed_beta
        actions.append((a, b, float(c), lab))
    out = feature_rows[["prompt_id", "layer"]].copy()
    out["alpha"] = [x[0] for x in actions]
    out["beta"] = [x[1] for x in actions]
    out["policy_conf"] = [x[2] for x in actions]
    out["action_label"] = [x[3] for x in actions]
    return out

# ============================================================
# HOOKS / EVAL
# ============================================================

def make_action_lookup(policy_actions_df):
    d = {}
    if policy_actions_df is None or len(policy_actions_df) == 0:
        return d
    for _, r in policy_actions_df.iterrows():
        d[(str(r["prompt_id"]), int(r["layer"]))] = (float(r["alpha"]), float(r["beta"]), float(r.get("policy_conf", 1.0)), str(r.get("action_label", "")))
    return d

def make_combo_hook_factory(
    cfg,
    stable_ops,
    precursor_dirs,
    control_mode,
    W_device,
    action_lookup=None,
    trace_buffer=None,
    clean_ids_batch=None,
    conflict_ids_batch=None,
    prompt_ids_batch=None,
):
    pre_tensors = {
        int(l): torch.tensor(v, dtype=torch.float32, device=W_device.device)
        for l, v in precursor_dirs.items() if v is not None
    }

    def hook_factory(l):
        def hook(module, inputs, output):
            if control_mode == "none":
                return output

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
            h_in_last = h_in[idx, pos, :]
            h_real_last = h_out[idx, pos, :]
            h_new = h_real_last

            alpha = torch.zeros((B, 1), device=h_out.device, dtype=h_out.dtype)
            beta = torch.zeros((B, 1), device=h_out.device, dtype=h_out.dtype)
            confs = []
            labels = []

            if control_mode == "operator_only_fixed":
                alpha[:] = cfg.fixed_alpha
            elif control_mode == "precursor_only_fixed":
                beta[:] = cfg.fixed_beta
            elif control_mode == "same_combo_fixed":
                alpha[:] = cfg.fixed_alpha
                beta[:] = cfg.fixed_beta
            elif control_mode.startswith("policy_") or control_mode.startswith("shuffle_") or control_mode == "real_predmech_policy":
                for bi, pid in enumerate(prompt_ids_batch):
                    a, b, c, lab = action_lookup.get((str(pid), int(l)), (cfg.fixed_alpha, cfg.fixed_beta, 0.0, "fallback"))
                    alpha[bi, 0] = a
                    beta[bi, 0] = b
                    confs.append(c)
                    labels.append(lab)
            elif control_mode == "random_gain_combo":
                g = ((l * 37) % 100) / 100.0
                alpha[:] = cfg.fixed_alpha * g
                beta[:] = cfg.fixed_beta * g
            elif control_mode == "answer":
                beta[:] = cfg.fixed_beta
            else:
                raise ValueError(control_mode)

            def apply_operator(h_base):
                if l not in stable_ops or torch.max(alpha).item() == 0.0:
                    return h_base
                h_target = stable_ops[l]["stable"].predict_torch(h_in_last.float()).to(h_base.dtype)
                return (1.0 - alpha) * h_base + alpha * h_target

            def apply_precursor(h_base):
                if l not in pre_tensors or torch.max(beta).item() == 0.0:
                    return h_base
                d = pre_tensors[l].to(h_base.device, h_base.dtype)
                scale = h_base.float().std(dim=-1, keepdim=True).to(h_base.dtype)
                return h_base + beta * scale * d[None, :].expand(B, -1)

            def apply_answer(h_base):
                ans = W_device[clean_ids_batch].float() - W_device[conflict_ids_batch].float()
                ans = ans / (torch.norm(ans, dim=1, keepdim=True) + 1e-8)
                scale = h_base.float().std(dim=-1, keepdim=True).to(h_base.dtype)
                return h_base + beta * scale * ans.to(h_base.dtype)

            if control_mode == "answer":
                h_new = apply_answer(h_new)
            else:
                h_new = apply_operator(h_new)
                h_new = apply_precursor(h_new)

            if trace_buffer is not None:
                trace_buffer.append({
                    "control": control_mode,
                    "layer": int(l),
                    "mean_alpha": float(alpha.detach().float().mean().cpu().item()),
                    "mean_beta": float(beta.detach().float().mean().cpu().item()),
                    "mean_policy_conf": float(np.mean(confs)) if confs else np.nan,
                    "most_common_action": max(set(labels), key=labels.count) if labels else "",
                    "batch_size": int(B),
                })

            if torch.allclose(h_new, h_real_last):
                return output

            h2 = h_out.clone()
            h2[idx, pos, :] = h_new

            if rest is not None:
                return (h2,) + rest
            return h2
        return hook
    return hook_factory

def forward_eval(model, tokenizer, df_subset, cfg, pca_decision,
                 clean_anchor_R, stable_ops, precursor_dirs,
                 control_mode, policy_actions_df=None):
    W = get_lm_head_weight(model).detach().float()
    W_device = W.to(model.device)
    rows = []
    trace_buffer = []
    action_lookup = make_action_lookup(policy_actions_df)

    texts = df_subset["text"].tolist()
    clean_ids_all = df_subset["clean_token_id"].values.astype(np.int64)
    conflict_ids_all = df_subset["conflict_token_id"].values.astype(np.int64)
    prompt_ids_all = df_subset["prompt_id"].astype(str).values.tolist()
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
            pids = prompt_ids_all[start:end]

            handles = []
            if control_mode != "none":
                hook_factory = make_combo_hook_factory(
                    cfg=cfg,
                    stable_ops=stable_ops,
                    precursor_dirs=precursor_dirs,
                    control_mode=control_mode,
                    W_device=W_device,
                    action_lookup=action_lookup,
                    trace_buffer=trace_buffer,
                    clean_ids_batch=cids,
                    conflict_ids_batch=eids,
                    prompt_ids_batch=pids,
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

    return pd.DataFrame(rows), pd.DataFrame(trace_buffer)

# ============================================================
# POLICY GRID / LABELS
# ============================================================

def action_to_label(action):
    return f"a{action[0]:.1f}_b{action[1]:.1f}"

def run_policy_train_grid(model, tokenizer, policy_train_df, cfg, pca_decision, clean_anchor_R,
                          stable_ops, precursor_dirs, base_train, save_dir):
    grid_results = []
    for a, b in cfg.policy_actions:
        label = action_to_label((a, b))
        rows = []
        for pid in policy_train_df["prompt_id"].astype(str):
            for l in cfg.operator_layers:
                rows.append({"prompt_id": pid, "layer": l, "alpha": a, "beta": b, "policy_conf": 1.0, "action_label": label})
        action_df = pd.DataFrame(rows)
        out, _ = forward_eval(
            model=model,
            tokenizer=tokenizer,
            df_subset=policy_train_df,
            cfg=cfg,
            pca_decision=pca_decision,
            clean_anchor_R=clean_anchor_R,
            stable_ops=stable_ops,
            precursor_dirs=precursor_dirs,
            control_mode="policy_grid",
            policy_actions_df=action_df,
        )
        out["action_label"] = label
        out["alpha"] = a
        out["beta"] = b
        grid_results.append(out)

    grid_df = pd.concat(grid_results, ignore_index=True)
    grid_df.to_csv(save_dir / "local_transition_policy_train_grid_results.csv", index=False, encoding="utf-8-sig")

    m = grid_df.merge(base_train[["prompt_id", "DeltaU", "mechanism"]], on="prompt_id", suffixes=("", "_base"))
    m["shift"] = m["DeltaU"] - m["DeltaU_base"]
    m["abs_shift"] = m["shift"].abs()

    labels = []
    for pid, g in m.groupby("prompt_id"):
        mech = str(g.iloc[0]["mechanism"])
        if mech == "closure":
            best = g.sort_values("shift", ascending=False).iloc[0]
        else:
            best = g.sort_values("abs_shift", ascending=True).iloc[0]
        labels.append({
            "prompt_id": pid,
            "mechanism": mech,
            "best_action_label": best["action_label"],
            "best_alpha": float(best["alpha"]),
            "best_beta": float(best["beta"]),
            "best_shift": float(best["shift"]),
            "best_abs_shift": float(best["abs_shift"]),
        })
    label_df = pd.DataFrame(labels)
    label_df.to_csv(save_dir / "local_transition_policy_train_grid_labels.csv", index=False, encoding="utf-8-sig")
    return grid_df, label_df

def policy_predict_actions(clf, feature_rows, feature_cols):
    X = feature_rows[feature_cols].values.astype(np.float32)
    labels = clf.predict(X)
    prob = clf.predict_proba(X) if hasattr(clf, "predict_proba") else np.ones((len(labels),1))
    conf = prob.max(axis=1)
    actions = []
    for lab, c in zip(labels, conf):
        try:
            a_str, b_str = lab.split("_")
            a = float(a_str[1:])
            b = float(b_str[1:])
        except Exception:
            a, b = CFG.fixed_alpha, CFG.fixed_beta
        actions.append((a, b, float(c), lab))
    out = feature_rows[["prompt_id", "layer"]].copy()
    out["alpha"] = [x[0] for x in actions]
    out["beta"] = [x[1] for x in actions]
    out["policy_conf"] = [x[2] for x in actions]
    out["action_label"] = [x[3] for x in actions]
    return out

# ============================================================
# SUMMARY
# ============================================================

def summarize_specificity(results_df, base_test):
    merged = results_df.merge(base_test[["prompt_id", "DeltaU", "pair_choice"]], on="prompt_id", suffixes=("", "_base"))
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
    main = spec_df[spec_df["control"] == "real_predmech_policy"]
    if len(main) == 0:
        return pd.DataFrame()
    main_spec = float(main.iloc[0]["specificity"])
    rows = []
    for prefix in ["shuffle_mech_policy", "shuffle_policy_label", "shuffle_both"]:
        vals = spec_df[spec_df["control"].str.startswith(prefix)]["specificity"].values.astype(float)
        if len(vals) == 0:
            continue
        mean = float(np.nanmean(vals))
        std = float(np.nanstd(vals) + 1e-8)
        z = float((main_spec - mean) / std)
        pct = float(np.mean(vals <= main_spec))
        q95 = float(np.quantile(vals, 0.95))
        rows.append({
            "null_family": prefix,
            "real_predmech_specificity": main_spec,
            "null_mean": mean,
            "null_std": std,
            "z": z,
            "percentile": pct,
            "q95": q95,
            "exceeds_q95": bool(main_spec > q95),
            "n": int(len(vals)),
        })
    for ctrl in ["same_combo_fixed", "operator_only_fixed", "precursor_only_fixed", "random_gain_combo", "answer"]:
        r = spec_df[spec_df["control"] == ctrl]
        if len(r):
            rows.append({
                "null_family": ctrl,
                "real_predmech_specificity": main_spec,
                "null_mean": float(r.iloc[0]["specificity"]),
                "null_std": np.nan,
                "z": np.nan,
                "percentile": np.nan,
                "q95": np.nan,
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
    with open(save_dir / "local_transition_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    model, tokenizer = load_model_and_tokenizer(cfg)
    df = build_dataset(tokenizer, cfg, save_dir)
    base_df, h_by_layer, pca_decision = extract_baseline(model, tokenizer, df, cfg, save_dir)
    train_idx, test_idx, split_df = make_split(base_df, cfg, save_dir)

    clean_anchor_R = {}
    clean_base = base_df[base_df["condition"] == "clean"].set_index("graph_id")
    for g, row in clean_base.iterrows():
        clean_anchor_R[int(g)] = {l: float(row[f"R_L{l}"]) for l in cfg.decision_layers}

    stable_ops = fit_stable_operators(base_df, h_by_layer, train_idx, cfg, save_dir)
    precursor_dirs, probes, vel_stats = fit_precursor_and_probes(base_df, h_by_layer, train_idx, cfg, save_dir)

    policy_train_df = base_df.iloc[train_idx].reset_index(drop=True)
    test_df = base_df.iloc[test_idx].reset_index(drop=True)
    base_train = base_df.iloc[train_idx][["prompt_id", "DeltaU", "mechanism"]].reset_index(drop=True)
    base_test = base_df.iloc[test_idx][["prompt_id", "DeltaU", "pair_choice"]].reset_index(drop=True)

    train_base_features = base_state_features_for_samples(base_df, h_by_layer, probes, vel_stats, train_idx, cfg)
    test_base_features = base_state_features_for_samples(base_df, h_by_layer, probes, vel_stats, test_idx, cfg)

    # Policy-train dose grid.
    print("[POLICY] Running policy-train dose grid...")
    grid_df, grid_labels = run_policy_train_grid(
        model=model,
        tokenizer=tokenizer,
        policy_train_df=policy_train_df,
        cfg=cfg,
        pca_decision=pca_decision,
        clean_anchor_R=clean_anchor_R,
        stable_ops=stable_ops,
        precursor_dirs=precursor_dirs,
        base_train=base_train,
        save_dir=save_dir,
    )

    # Real mechanism classifier and real policy.
    real_mech_clf, real_mech_audit = fit_mechanism_classifier(train_base_features, cfg, save_dir, shuffle_labels=False, shuffle_id=0)
    train_real_pred = attach_predicted_mechanism(train_base_features, real_mech_clf)
    test_real_pred = attach_predicted_mechanism(test_base_features, real_mech_clf)
    real_policy_clf, real_policy_audit = fit_gain_policy(train_real_pred, grid_labels, POLICY_PRED_MECH_COLS, cfg, shuffle_policy_labels=False, shuffle_id=0)
    real_actions = policy_predict_actions(real_policy_clf, test_real_pred, POLICY_PRED_MECH_COLS)
    real_actions.to_csv(save_dir / "local_transition_real_policy_actions.csv", index=False, encoding="utf-8-sig")

    # Deterministic controls.
    controls = [
        ("none", None),
        ("operator_only_fixed", None),
        ("precursor_only_fixed", None),
        ("same_combo_fixed", None),
        ("real_predmech_policy", real_actions),
        ("random_gain_combo", None),
    ]
    if cfg.include_answer_control:
        controls.append(("answer", None))

    shuffle_audits = []

    # Shuffle null policies.
    for s in range(cfg.n_shuffles):
        print(f"[SHUFFLE] {s+1}/{cfg.n_shuffles}")

        if cfg.eval_shuffle_mech_policy:
            sh_mech_clf, sh_mech_audit = fit_mechanism_classifier(train_base_features, cfg, save_dir, shuffle_labels=True, shuffle_id=s)
            train_sh_pred = attach_predicted_mechanism(train_base_features, sh_mech_clf)
            test_sh_pred = attach_predicted_mechanism(test_base_features, sh_mech_clf)
            sh_policy_clf, sh_pol_audit = fit_gain_policy(train_sh_pred, grid_labels, POLICY_PRED_MECH_COLS, cfg, shuffle_policy_labels=False, shuffle_id=s)
            sh_actions = policy_predict_actions(sh_policy_clf, test_sh_pred, POLICY_PRED_MECH_COLS)
            controls.append((f"shuffle_mech_policy_{s:03d}", sh_actions))
            shuffle_audits.append({"shuffle": s, "family": "shuffle_mech_policy", **sh_mech_audit, **{f"policy_{k}": v for k, v in sh_pol_audit.items() if k not in ("class_counts",)}})

        if cfg.eval_shuffle_policy_label:
            sh_policy_clf, sh_pol_audit = fit_gain_policy(train_real_pred, grid_labels, POLICY_PRED_MECH_COLS, cfg, shuffle_policy_labels=True, shuffle_id=s)
            sh_actions = policy_predict_actions(sh_policy_clf, test_real_pred, POLICY_PRED_MECH_COLS)
            controls.append((f"shuffle_policy_label_{s:03d}", sh_actions))
            shuffle_audits.append({"shuffle": s, "family": "shuffle_policy_label", **{f"policy_{k}": v for k, v in sh_pol_audit.items() if k not in ("class_counts",)}})

        if cfg.eval_shuffle_both:
            sh_mech_clf, sh_mech_audit = fit_mechanism_classifier(train_base_features, cfg, save_dir, shuffle_labels=True, shuffle_id=1000+s)
            train_sh_pred = attach_predicted_mechanism(train_base_features, sh_mech_clf)
            test_sh_pred = attach_predicted_mechanism(test_base_features, sh_mech_clf)
            sh_policy_clf, sh_pol_audit = fit_gain_policy(train_sh_pred, grid_labels, POLICY_PRED_MECH_COLS, cfg, shuffle_policy_labels=True, shuffle_id=1000+s)
            sh_actions = policy_predict_actions(sh_policy_clf, test_sh_pred, POLICY_PRED_MECH_COLS)
            controls.append((f"shuffle_both_{s:03d}", sh_actions))
            shuffle_audits.append({"shuffle": s, "family": "shuffle_both", **sh_mech_audit, **{f"policy_{k}": v for k, v in sh_pol_audit.items() if k not in ("class_counts",)}})

    pd.DataFrame(shuffle_audits).to_csv(save_dir / "local_transition_shuffle_policy_audits.csv", index=False, encoding="utf-8-sig")
    with open(save_dir / "local_transition_real_mechanism_audit.json", "w", encoding="utf-8") as f:
        json.dump(real_mech_audit, f, indent=2, ensure_ascii=False)
    with open(save_dir / "local_transition_real_policy_audit.json", "w", encoding="utf-8") as f:
        json.dump(real_policy_audit, f, indent=2, ensure_ascii=False)

    # Evaluate all controls.
    results = []
    traces = []
    for control, action_df in controls:
        print(f"[RUN] control={control}")
        out, tr = forward_eval(
            model=model,
            tokenizer=tokenizer,
            df_subset=test_df,
            cfg=cfg,
            pca_decision=pca_decision,
            clean_anchor_R=clean_anchor_R,
            stable_ops=stable_ops,
            precursor_dirs=precursor_dirs,
            control_mode=control,
            policy_actions_df=action_df,
        )
        out["control"] = control
        results.append(out)
        if len(tr):
            tr["control_run"] = control
            traces.append(tr)

    results_df = pd.concat(results, ignore_index=True)
    results_df.to_csv(save_dir / "local_transition_steering_results.csv", index=False, encoding="utf-8-sig")

    spec_df, merged = summarize_specificity(results_df, base_test)
    spec_df.to_csv(save_dir / "local_transition_specificity_summary.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(save_dir / "local_transition_merged_shift_table.csv", index=False, encoding="utf-8-sig")

    trace_df = pd.concat(traces, ignore_index=True) if len(traces) else pd.DataFrame()
    trace_df.to_csv(save_dir / "local_transition_control_trace.csv", index=False, encoding="utf-8-sig")

    val_df = shuffle_validation(spec_df)
    val_df.to_csv(save_dir / "local_transition_shuffle_validation.csv", index=False, encoding="utf-8-sig")

    # Verdict.
    real = spec_df[spec_df["control"] == "real_predmech_policy"]
    real_spec = float(real.iloc[0]["specificity"]) if len(real) else np.nan
    fixed = spec_df[spec_df["control"] == "same_combo_fixed"]
    fixed_spec = float(fixed.iloc[0]["specificity"]) if len(fixed) else np.nan

    both = val_df[val_df["null_family"] == "shuffle_both"]
    both_z = float(both.iloc[0]["z"]) if len(both) else np.nan
    both_exceeds = bool(both.iloc[0]["exceeds_q95"]) if len(both) else False

    mechnull = val_df[val_df["null_family"] == "shuffle_mech_policy"]
    mech_z = float(mechnull.iloc[0]["z"]) if len(mechnull) else np.nan
    polnull = val_df[val_df["null_family"] == "shuffle_policy_label"]
    pol_z = float(polnull.iloc[0]["z"]) if len(polnull) else np.nan

    gain_over_fixed = real_spec - fixed_spec

    if np.isfinite(real_spec) and gain_over_fixed > 0.10 and np.isfinite(both_z) and both_z > 2 and both_exceeds:
        verdict = "PASS_STRONG_VALIDATED_CLOSED_LOOP"
    elif np.isfinite(real_spec) and gain_over_fixed > 0.10 and np.isfinite(both_z) and both_z > 1.5:
        verdict = "PASS_LITE_VALIDATED_CLOSED_LOOP"
    elif np.isfinite(real_spec) and gain_over_fixed > 0.10:
        verdict = "GAIN_OVER_FIXED_BUT_NULL_WEAK"
    else:
        verdict = "NO_VALIDATED_CLOSED_LOOP_GAIN"

    verdict_df = pd.DataFrame([{
        "audit": "local transition control Matched Shuffle Validation",
        "verdict": verdict,
        "real_predmech_specificity": real_spec,
        "fixed_combo_specificity": fixed_spec,
        "gain_over_fixed": gain_over_fixed,
        "z_vs_shuffle_both": both_z,
        "z_vs_shuffle_mech_policy": mech_z,
        "z_vs_shuffle_policy_label": pol_z,
        "exceeds_shuffle_both_q95": both_exceeds,
        "n_shuffles": cfg.n_shuffles,
        "real_mech_train_accuracy": real_mech_audit.get("train_accuracy"),
        "real_mech_train_macro_f1": real_mech_audit.get("train_macro_f1"),
        "real_policy_train_accuracy": real_policy_audit.get("train_accuracy"),
        "real_policy_train_macro_f1": real_policy_audit.get("train_macro_f1"),
    }])
    verdict_df.to_csv(save_dir / "local_transition_verdict.csv", index=False, encoding="utf-8-sig")

    summary = {
        "audit": "local transition control Matched Shuffle / Null Validation for Predicted-Mechanism Closed-Loop Control",
        "config": asdict(cfg),
        "n_policy_train_prompts": int(len(policy_train_df)),
        "n_test_prompts": int(len(test_df)),
        "deltaU_pca_explained_variance": float(pca_decision.explained_variance_ratio_[0]),
        "verdict": verdict,
        "main_metrics": {
            "real_predmech_specificity": real_spec,
            "fixed_combo_specificity": fixed_spec,
            "gain_over_fixed": gain_over_fixed,
            "z_vs_shuffle_both": both_z,
            "z_vs_shuffle_mech_policy": mech_z,
            "z_vs_shuffle_policy_label": pol_z,
            "exceeds_shuffle_both_q95": both_exceeds,
        },
        "real_mechanism_audit": real_mech_audit,
        "real_policy_audit": real_policy_audit,
        "key_outputs": {
            "specificity": "local_transition_specificity_summary.csv",
            "shuffle_validation": "local_transition_shuffle_validation.csv",
            "verdict": "local_transition_verdict.csv",
            "steering_results": "local_transition_steering_results.csv",
            "policy_train_grid_results": "local_transition_policy_train_grid_results.csv",
            "policy_train_grid_labels": "local_transition_policy_train_grid_labels.csv",
            "shuffle_policy_audits": "local_transition_shuffle_policy_audits.csv",
            "real_policy_actions": "local_transition_real_policy_actions.csv",
            "control_trace": "local_transition_control_trace.csv",
        },
        "note": (
            "local transition control validates preceding local-control analysis against shuffled mechanism classifier, shuffled policy labels, and shuffled-both nulls. "
            "If PASS_STRONG, predicted-mechanism closed-loop control can be treated as validated trajectory-level control. "
            "Generation-level hit rate is still a separate objective."
        )
    }
    with open(save_dir / "local_transition_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nlocal transition control complete.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main(CFG)
