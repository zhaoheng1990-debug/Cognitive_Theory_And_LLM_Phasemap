
# ============================================================
# ASA-6D: True-Ocont Mediation Audit
#
# Purpose:
#   ASA-6C showed that position_ridge is best interpreted as an
#   ORDER_PRECURSOR, but it used a crude Oproxy:
#
#       Oproxy_l = PC1(H_{l+1} - H_l)
#
#   ASA-6D replaces that with a richer, 7E-inspired continuous
#   morphism / transport feature:
#
#       Ocont_like_l =
#       [
#         dR_l,
#         dB_l,
#         dEntropy_l,
#         TopK center transport,
#         TopK set turnover,
#         TopK spread change,
#         TopK entropy change
#       ]
#
#   Then it tests whether position_score is mediated by this Ocont_like
#   channel:
#
#       H_l -> position_score -> Ocont_like -> DeltaU
#
#   or whether position_ridge remains an independent order-precursor.
#
# Main outputs:
#   asa6d_outputs/
#     asa6d_summary.json
#     asa6d_verdict.csv
#     asa6d_baseline_features.csv
#     asa6d_o_cont_like_features.csv
#     asa6d_position_scores.csv
#     asa6d_mediation.csv
#     asa6d_direction_alignment.csv
#     asa6d_orthogonal_survival.csv
#
# Important:
#   This is not the original 7E artifact format. It reconstructs a
#   "trueer" Ocont-like object on the ASA dataset using both answerless
#   margin transport and TopK/VIM neighborhood transport.
#
#   If you later have exact 7E O_cont artifacts aligned to this dataset,
#   use those to replace Ocont_like_l and rerun the same mediation block.
# ============================================================

import os
import gc
import json
import random
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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
    save_dir: str = "./asa6d_outputs"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float16"
    seed: int = 42

    n_graphs: int = 96
    max_len: int = 260
    batch_size: int = 4
    test_size: float = 0.30

    # ASA-6B/6C core layers.
    probe_layers: Tuple[int, ...] = (17, 18, 19)

    # Need l+1 for Ocont-like transition.
    transition_layers: Tuple[int, ...] = (17, 18, 19)

    # Decision layers used to build DeltaU.
    decision_layers: Tuple[int, ...] = tuple(range(20, 26))

    # TopK/VIM settings.
    topk: int = 500
    compute_topk_features: bool = True

    # Ridge settings.
    ridge_alpha: float = 10.0
    min_dir_norm: float = 1e-4
    n_splits: int = 5

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
    pd.DataFrame(rows).to_csv(save_dir / "asa6d_label_tokenization_audit.csv", index=False, encoding="utf-8-sig")
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
    df.to_csv(save_dir / "asa6d_dataset.csv", index=False, encoding="utf-8-sig")
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

def binary_entropy_from_logit_margin(R):
    p = 1.0 / (1.0 + np.exp(-np.clip(R, -50, 50)))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    h = -(p * np.log(p) + (1-p) * np.log(1-p)) / np.log(2.0)
    return h.astype(np.float32)

def row_cosine(A, B):
    num = np.sum(A * B, axis=1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-8
    return (num / den).astype(np.float32)

def topk_entropy_from_logits(vals):
    vals = vals.astype(np.float32)
    vals = vals - np.max(vals, axis=1, keepdims=True)
    p = np.exp(vals)
    p = p / (np.sum(p, axis=1, keepdims=True) + 1e-8)
    ent = -np.sum(p * np.log(p + 1e-8), axis=1) / np.log(vals.shape[1])
    return ent.astype(np.float32)

def jaccard_rows(ids1, ids2):
    out = np.zeros(ids1.shape[0], dtype=np.float32)
    for i in range(ids1.shape[0]):
        a = set(ids1[i].tolist())
        b = set(ids2[i].tolist())
        out[i] = len(a & b) / max(1, len(a | b))
    return out

# ============================================================
# EXTRACTION
# ============================================================

def extract_all(model, tokenizer, df, cfg, save_dir):
    print("[EXTRACT] hidden, R, and TopK/VIM features")
    W = get_lm_head_weight(model).detach().float()
    W_cpu = W.cpu().numpy().astype(np.float32)
    W_device = W.to(model.device)
    W_normed_cpu = W_cpu / (np.linalg.norm(W_cpu, axis=1, keepdims=True) + 1e-8)

    n = len(df)
    needed_layers = sorted(set(
        list(cfg.probe_layers)
        + [l + 1 for l in cfg.transition_layers]
        + list(cfg.decision_layers)
    ))

    h_by_layer = {l: None for l in needed_layers}
    R_by_layer = {l: np.zeros(n, dtype=np.float32) for l in sorted(set(needed_layers + list(cfg.decision_layers)))}
    Hbin_by_layer = {l: np.zeros(n, dtype=np.float32) for l in sorted(set(needed_layers + list(cfg.decision_layers)))}

    topk_ids_by_layer = {}
    topk_vals_by_layer = {}
    center_by_layer = {}
    spread_by_layer = {}
    topkent_by_layer = {}

    for l in needed_layers:
        topk_ids_by_layer[l] = None
        topk_vals_by_layer[l] = None
        center_by_layer[l] = None
        spread_by_layer[l] = np.zeros(n, dtype=np.float32)
        topkent_by_layer[l] = np.zeros(n, dtype=np.float32)

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
                ht = hstates[l + 1][idx_t, pos, :].detach().float()
                h_np = ht.cpu().numpy().astype(np.float32)

                if h_by_layer[l] is None:
                    h_by_layer[l] = np.zeros((n, h_np.shape[1]), dtype=np.float32)
                h_by_layer[l][start:end] = h_np

                clean_logits = torch.sum(ht * W_device[cids].float(), dim=1)
                conflict_logits = torch.sum(ht * W_device[eids].float(), dim=1)
                R = (clean_logits - conflict_logits).detach().cpu().numpy().astype(np.float32)
                R_by_layer[l][start:end] = R
                Hbin_by_layer[l][start:end] = binary_entropy_from_logit_margin(R)

                if cfg.compute_topk_features:
                    logits = (ht @ W_device.T).detach().float()
                    vals, ids = torch.topk(logits, k=cfg.topk, dim=1)
                    vals_np = vals.cpu().numpy().astype(np.float32)
                    ids_np = ids.cpu().numpy().astype(np.int32)

                    if topk_ids_by_layer[l] is None:
                        topk_ids_by_layer[l] = np.zeros((n, cfg.topk), dtype=np.int32)
                        topk_vals_by_layer[l] = np.zeros((n, cfg.topk), dtype=np.float32)
                        center_by_layer[l] = np.zeros((n, W_cpu.shape[1]), dtype=np.float32)

                    topk_ids_by_layer[l][start:end] = ids_np
                    topk_vals_by_layer[l][start:end] = vals_np

                    emb = W_normed_cpu[ids_np]  # [B,K,D]
                    center = emb.mean(axis=1)
                    center = center / (np.linalg.norm(center, axis=1, keepdims=True) + 1e-8)
                    center_by_layer[l][start:end] = center.astype(np.float32)

                    cos_to_center = np.sum(emb * center[:, None, :], axis=2)
                    spread_by_layer[l][start:end] = np.mean(1.0 - cos_to_center, axis=1).astype(np.float32)
                    topkent_by_layer[l][start:end] = topk_entropy_from_logits(vals_np)

            print(f"  extracted {end}/{n}")

            del outputs, hstates, inputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Build baseline dataframe.
    out = df.copy()
    for l in cfg.decision_layers:
        out[f"R_L{l}"] = R_by_layer[l]
        out[f"Hbin_L{l}"] = Hbin_by_layer[l]

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
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(X_dec)
    du = pcs[:, 0].astype(np.float32)
    if np.mean(du[out["mechanism"] == "stable"]) < np.mean(du[out["mechanism"] == "closure"]):
        du = -du
        pcs[:, 0] = du
        pca.components_[0] *= -1

    out["DeltaU"] = du.astype(np.float32)
    out["ShapePC2"] = pcs[:, 1].astype(np.float32)
    out["ShapePC3"] = pcs[:, 2].astype(np.float32)
    out["D_B_proxy"] = np.min(np.abs(X_dec), axis=1).astype(np.float32)
    out["U_K_proxy"] = (-out["D_B_proxy"]).astype(np.float32)

    out.to_csv(save_dir / "asa6d_baseline_features.csv", index=False, encoding="utf-8-sig")
    print("[PCA] DeltaU explained variance:", float(pca.explained_variance_ratio_[0]))

    topk_pack = {
        "ids": topk_ids_by_layer,
        "vals": topk_vals_by_layer,
        "center": center_by_layer,
        "spread": spread_by_layer,
        "entropy": topkent_by_layer,
    }

    return out, h_by_layer, R_by_layer, Hbin_by_layer, topk_pack, pca

# ============================================================
# SPLIT + ML HELPERS
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
    split_df.to_csv(save_dir / "asa6d_split.csv", index=False, encoding="utf-8-sig")
    return np.asarray(tr), np.asarray(te), split_df

def fit_ridge_direction(X, y, alpha=10.0, min_norm=1e-4):
    X = X.astype(np.float32)
    y = np.asarray(y, dtype=np.float32)
    X0 = X - X.mean(axis=0, keepdims=True)
    y0 = y - y.mean()
    reg = Ridge(alpha=alpha)
    reg.fit(X0, y0)
    coef = reg.coef_.astype(np.float32)
    pred = reg.predict(X0)
    r2 = r2_score(y0, pred)
    corr = safe_corr(y0, pred)
    unit, norm = normalize_np(coef, eps=min_norm)
    return unit, {
        "coef_norm": norm,
        "valid": bool(unit is not None),
        "train_r2": float(r2) if np.isfinite(r2) else np.nan,
        "train_corr": float(corr) if np.isfinite(corr) else np.nan,
    }

def ridge_group_cv_predict(X, y, groups, cfg, alpha=1.0):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if X.ndim == 1:
        X = X[:, None]
    n_splits = min(cfg.n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    pred = np.zeros(len(y), dtype=np.float32)
    for tr, te in gkf.split(X, y, groups):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])
        pipe.fit(X[tr], y[tr])
        pred[te] = pipe.predict(X[te]).astype(np.float32)
    return pred

def vector_cos(u, v):
    if u is None or v is None:
        return np.nan
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-8 or nv < 1e-8:
        return np.nan
    return float(np.dot(u, v) / (nu * nv))

def project_out(u, basis_vecs):
    if u is None:
        return None
    x = np.asarray(u, dtype=np.float32).copy()
    for b in basis_vecs:
        if b is None:
            continue
        b = np.asarray(b, dtype=np.float32)
        nb = np.linalg.norm(b)
        if nb < 1e-8:
            continue
        bh = b / nb
        x = x - np.dot(x, bh) * bh
    unit, _ = normalize_np(x, eps=1e-8)
    return unit

# ============================================================
# OCONT-LIKE FEATURE CONSTRUCTION
# ============================================================

def build_o_cont_like(base_df, R_by_layer, Hbin_by_layer, topk_pack, cfg, save_dir):
    rows = []
    features_by_layer = {}

    for l in cfg.transition_layers:
        lp1 = l + 1
        n = len(base_df)

        dR = R_by_layer[lp1] - R_by_layer[l]
        dB = np.abs(R_by_layer[lp1]) - np.abs(R_by_layer[l])
        dH = Hbin_by_layer[lp1] - Hbin_by_layer[l]

        feature_cols = {
            f"O_L{l}_dR": dR.astype(np.float32),
            f"O_L{l}_dAbsR": dB.astype(np.float32),
            f"O_L{l}_dEntropy": dH.astype(np.float32),
        }

        if cfg.compute_topk_features:
            C0 = topk_pack["center"][l]
            C1 = topk_pack["center"][lp1]
            center_dist = 1.0 - row_cosine(C0, C1)
            jacc = jaccard_rows(topk_pack["ids"][l], topk_pack["ids"][lp1])
            wjacc_dist = 1.0 - jacc  # simple unweighted proxy
            dspread = topk_pack["spread"][lp1] - topk_pack["spread"][l]
            dent = topk_pack["entropy"][lp1] - topk_pack["entropy"][l]

            feature_cols.update({
                f"O_L{l}_topk_center_dist": center_dist.astype(np.float32),
                f"O_L{l}_topk_set_turnover": wjacc_dist.astype(np.float32),
                f"O_L{l}_dSpread": dspread.astype(np.float32),
                f"O_L{l}_dTopKEntropy": dent.astype(np.float32),
            })

        O = pd.DataFrame(feature_cols)
        # Standardized PC1 as compact Ocont-like scalar.
        X = O.values.astype(np.float32)
        Xs = StandardScaler().fit_transform(X)
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(Xs)[:, 0].astype(np.float32)
        if safe_corr(pc1, base_df["DeltaU"].values) < 0:
            pc1 = -pc1
        O[f"Ocont_like_PC1_L{l}"] = pc1
        O["prompt_id"] = base_df["prompt_id"].values
        O["graph_id"] = base_df["graph_id"].values
        O["condition"] = base_df["condition"].values
        O["mechanism"] = base_df["mechanism"].values
        O["layer"] = l

        features_by_layer[l] = O
        rows.append(O)

    full = pd.concat(rows, ignore_index=True)
    full.to_csv(save_dir / "asa6d_o_cont_like_features_long.csv", index=False, encoding="utf-8-sig")

    # Wide form for convenience.
    wide = base_df[["prompt_id", "graph_id", "condition", "mechanism", "DeltaU", "D_B_proxy", "U_K_proxy", "ShapePC2", "ShapePC3"]].copy()
    for l, O in features_by_layer.items():
        for col in O.columns:
            if col.startswith(f"O_L{l}_") or col.startswith("Ocont_like_PC1"):
                wide[col] = O[col].values
    wide.to_csv(save_dir / "asa6d_o_cont_like_features.csv", index=False, encoding="utf-8-sig")

    return features_by_layer, wide

# ============================================================
# MAIN AUDIT
# ============================================================

def main(cfg):
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "asa6d_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    model, tokenizer = load_model_and_tokenizer(cfg)
    df = build_dataset(tokenizer, cfg, save_dir)

    base_df, h_by_layer, R_by_layer, Hbin_by_layer, topk_pack, pca_decision = extract_all(model, tokenizer, df, cfg, save_dir)
    train_idx, test_idx, split_df = make_split(base_df, cfg, save_dir)
    O_by_layer, O_wide = build_o_cont_like(base_df, R_by_layer, Hbin_by_layer, topk_pack, cfg, save_dir)

    groups = base_df["graph_id"].values
    y = base_df["DeltaU"].values.astype(np.float32)

    # Answer direction.
    W = get_lm_head_weight(model).detach().float().cpu().numpy().astype(np.float32)
    clean_ids = base_df["clean_token_id"].values.astype(int)
    conflict_ids = base_df["conflict_token_id"].values.astype(int)
    ans_dir = (W[clean_ids] - W[conflict_ids]).mean(axis=0).astype(np.float32)
    ans_dir, _ = normalize_np(ans_dir, eps=cfg.min_dir_norm)

    # Collect results.
    score_rows = []
    med_rows = []
    align_rows = []
    orth_rows = []
    pred_rows = []

    for l in cfg.probe_layers:
        H = h_by_layer[l]
        # Position ridge direction H_l -> DeltaU.
        w_du, audit_du = fit_ridge_direction(H[train_idx], y[train_idx], cfg.ridge_alpha, cfg.min_dir_norm)
        pos_score = H @ w_du

        # Ocont-like features for same l.
        Odf = O_by_layer[l]
        O_cols = [c for c in Odf.columns if c.startswith(f"O_L{l}_")]
        OX = Odf[O_cols].values.astype(np.float32)
        O_pc1 = Odf[f"Ocont_like_PC1_L{l}"].values.astype(np.float32)

        # Learn H_l -> Ocont_like_PC1 direction.
        w_o, audit_o = fit_ridge_direction(H[train_idx], O_pc1[train_idx], cfg.ridge_alpha, cfg.min_dir_norm)
        o_score_from_H = H @ w_o if w_o is not None else np.zeros(len(base_df), dtype=np.float32)

        # Learn H_l -> DB direction.
        w_db, audit_db = fit_ridge_direction(H[train_idx], base_df["D_B_proxy"].values[train_idx], cfg.ridge_alpha, cfg.min_dir_norm)
        w_shape2, audit_s2 = fit_ridge_direction(H[train_idx], base_df["ShapePC2"].values[train_idx], cfg.ridge_alpha, cfg.min_dir_norm)

        # Prediction audits.
        pred_pos = ridge_group_cv_predict(pos_score, y, groups, cfg)
        pred_o = ridge_group_cv_predict(OX, y, groups, cfg)
        pred_opc1 = ridge_group_cv_predict(O_pc1, y, groups, cfg)
        pred_pos_o = ridge_group_cv_predict(np.column_stack([pos_score, OX]), y, groups, cfg)
        pred_pos_opc1 = ridge_group_cv_predict(np.column_stack([pos_score, O_pc1]), y, groups, cfg)

        def add_pred(name, pred):
            pred_rows.append({
                "layer": l,
                "feature_set": name,
                "group_cv_r2": float(r2_score(y, pred)),
                "group_cv_corr": safe_corr(y, pred),
            })

        add_pred("position_score", pred_pos)
        add_pred("Ocont_like_features", pred_o)
        add_pred("Ocont_like_PC1", pred_opc1)
        add_pred("position_plus_Ocont_features", pred_pos_o)
        add_pred("position_plus_Ocont_PC1", pred_pos_opc1)

        # Mediation style increments.
        r2_pos = r2_score(y, pred_pos)
        r2_o = r2_score(y, pred_o)
        r2_opc1 = r2_score(y, pred_opc1)
        r2_pos_o = r2_score(y, pred_pos_o)
        r2_pos_opc1 = r2_score(y, pred_pos_opc1)

        med_rows.append({
            "layer": l,
            "r2_position": float(r2_pos),
            "r2_Ocont_features": float(r2_o),
            "r2_Ocont_PC1": float(r2_opc1),
            "r2_position_plus_Ocont_features": float(r2_pos_o),
            "r2_position_plus_Ocont_PC1": float(r2_pos_opc1),
            "O_gain_over_position": float(r2_pos_o - r2_pos),
            "position_gain_over_O": float(r2_pos_o - r2_o),
            "O_PC1_gain_over_position": float(r2_pos_opc1 - r2_pos),
            "position_gain_over_O_PC1": float(r2_pos_opc1 - r2_opc1),
            "position_mediated_by_O_fraction": float(max(0.0, min(1.0, (r2_pos - max(0.0, r2_pos_o - r2_o)) / (abs(r2_pos) + 1e-8)))) if np.isfinite(r2_pos) else np.nan,
        })

        # Direction alignments in hidden space.
        align_rows.append({
            "layer": l,
            "cos_wDeltaU_wOcontPC1": vector_cos(w_du, w_o),
            "abs_cos_wDeltaU_wOcontPC1": abs(vector_cos(w_du, w_o)) if np.isfinite(vector_cos(w_du, w_o)) else np.nan,
            "cos_wDeltaU_wDB": vector_cos(w_du, w_db),
            "abs_cos_wDeltaU_wDB": abs(vector_cos(w_du, w_db)) if np.isfinite(vector_cos(w_du, w_db)) else np.nan,
            "cos_wDeltaU_wShapePC2": vector_cos(w_du, w_shape2),
            "abs_cos_wDeltaU_wShapePC2": abs(vector_cos(w_du, w_shape2)) if np.isfinite(vector_cos(w_du, w_shape2)) else np.nan,
            "cos_wDeltaU_answer": vector_cos(w_du, ans_dir),
            "abs_cos_wDeltaU_answer": abs(vector_cos(w_du, ans_dir)) if np.isfinite(vector_cos(w_du, ans_dir)) else np.nan,
            "H_to_DeltaU_train_corr": audit_du["train_corr"],
            "H_to_OcontPC1_train_corr": audit_o["train_corr"],
            "H_to_DB_train_corr": audit_db["train_corr"],
        })

        # Orthogonal survival: project w_DU out of O/DB/answer directions.
        basis_sets = {
            "none": [],
            "answer": [ans_dir],
            "OcontPC1": [w_o],
            "DB": [w_db],
            "OcontPC1_plus_DB": [w_o, w_db],
            "OcontPC1_plus_DB_plus_answer": [w_o, w_db, ans_dir],
            "OcontPC1_plus_DB_plus_ShapePC2": [w_o, w_db, w_shape2],
        }

        base_corr = safe_corr(pos_score, y)

        for name, basis in basis_sets.items():
            w_res = project_out(w_du, basis)
            if w_res is None:
                score = np.zeros(len(y), dtype=np.float32)
                valid = False
            else:
                score = H @ w_res
                valid = True
            corr = safe_corr(score, y)
            orth_rows.append({
                "layer": l,
                "orthogonalization": name,
                "score_corr_DeltaU": corr,
                "survival_ratio": abs(corr) / (abs(base_corr) + 1e-8) if np.isfinite(corr) and np.isfinite(base_corr) else np.nan,
                "direction_valid": bool(valid),
            })

        # Save per-sample scores.
        tmp = pd.DataFrame({
            "prompt_id": base_df["prompt_id"],
            "graph_id": base_df["graph_id"],
            "condition": base_df["condition"],
            "mechanism": base_df["mechanism"],
            "layer": l,
            "DeltaU": y,
            "position_score": pos_score,
            "Ocont_like_PC1": O_pc1,
            "O_score_from_H": o_score_from_H,
            "D_B_proxy": base_df["D_B_proxy"].values,
        })
        score_rows.append(tmp)

    pd.concat(score_rows, ignore_index=True).to_csv(save_dir / "asa6d_position_o_scores_long.csv", index=False, encoding="utf-8-sig")

    med_df = pd.DataFrame(med_rows)
    med_df.to_csv(save_dir / "asa6d_mediation.csv", index=False, encoding="utf-8-sig")

    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(save_dir / "asa6d_prediction_audit.csv", index=False, encoding="utf-8-sig")

    align_df = pd.DataFrame(align_rows)
    align_df.to_csv(save_dir / "asa6d_direction_alignment.csv", index=False, encoding="utf-8-sig")

    orth_df = pd.DataFrame(orth_rows)
    orth_df.to_csv(save_dir / "asa6d_orthogonal_survival.csv", index=False, encoding="utf-8-sig")

    # Verdict heuristic.
    avg = {
        "position_r2": float(med_df["r2_position"].mean()),
        "Ocont_r2": float(med_df["r2_Ocont_features"].mean()),
        "position_plus_Ocont_r2": float(med_df["r2_position_plus_Ocont_features"].mean()),
        "position_gain_over_O": float(med_df["position_gain_over_O"].mean()),
        "O_gain_over_position": float(med_df["O_gain_over_position"].mean()),
        "abs_cos_DU_Ocont": float(align_df["abs_cos_wDeltaU_wOcontPC1"].mean()),
        "abs_cos_DU_DB": float(align_df["abs_cos_wDeltaU_wDB"].mean()),
        "abs_cos_DU_answer": float(align_df["abs_cos_wDeltaU_answer"].mean()),
    }

    orth_o = orth_df[orth_df["orthogonalization"] == "OcontPC1"]
    orth_all = orth_df[orth_df["orthogonalization"] == "OcontPC1_plus_DB_plus_answer"]
    avg["survival_after_O_orthogonal"] = float(orth_o["survival_ratio"].mean()) if len(orth_o) else np.nan
    avg["survival_after_O_DB_answer_orthogonal"] = float(orth_all["survival_ratio"].mean()) if len(orth_all) else np.nan

    # Interpretation:
    # - If Ocont predicts DeltaU well and position adds little over O, operator mediation.
    # - If position remains strong over O and survives O-orthogonalization, independent order precursor.
    # - If answer cos high or answer orth kills it, answer artifact.
    if avg["abs_cos_DU_answer"] > 0.25:
        verdict = "ANSWER_ARTIFACT_RISK"
    elif avg["Ocont_r2"] > 0.5 and avg["position_gain_over_O"] < 0.05 and avg["survival_after_O_orthogonal"] < 0.5:
        verdict = "O_CONT_MEDIATED_OPERATOR_PRECURSOR"
    elif avg["position_gain_over_O"] > 0.05 and avg["survival_after_O_orthogonal"] > 0.6:
        verdict = "INDEPENDENT_ORDER_PRECURSOR_CONFIRMED"
    elif avg["Ocont_r2"] > 0.5 and avg["position_plus_Ocont_r2"] > max(avg["position_r2"], avg["Ocont_r2"]) + 0.03:
        verdict = "DUAL_CHANNEL_POSITION_AND_O_CONT"
    else:
        verdict = "MIXED_OR_UNRESOLVED"

    verdict_df = pd.DataFrame([{
        "audit": "ASA-6D True-Ocont Mediation Audit",
        "verdict": verdict,
        **avg,
    }])
    verdict_df.to_csv(save_dir / "asa6d_verdict.csv", index=False, encoding="utf-8-sig")

    summary = {
        "audit": "ASA-6D True-Ocont Mediation Audit",
        "config": asdict(cfg),
        "n_prompts": int(len(base_df)),
        "deltaU_pca_explained_variance": float(pca_decision.explained_variance_ratio_[0]),
        "verdict": verdict,
        "averages": avg,
        "key_outputs": {
            "baseline_features": "asa6d_baseline_features.csv",
            "o_cont_like_features": "asa6d_o_cont_like_features.csv",
            "scores": "asa6d_position_o_scores_long.csv",
            "prediction_audit": "asa6d_prediction_audit.csv",
            "mediation": "asa6d_mediation.csv",
            "direction_alignment": "asa6d_direction_alignment.csv",
            "orthogonal_survival": "asa6d_orthogonal_survival.csv",
            "verdict": "asa6d_verdict.csv",
        },
        "important_caveat": (
            "Ocont_like here is reconstructed from local margin transport and TopK/VIM transport on the ASA dataset. "
            "It is closer to PhaseMap O_cont than ASA-6C's hidden-transport PC1, but it is still not necessarily identical "
            "to the original 7E O_cont artifacts."
        )
    }

    with open(save_dir / "asa6d_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nASA-6D complete.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main(CFG)
