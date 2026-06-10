
# ============================================================
# ASA-6C: Position Ridge Ontology Decomposition Audit
#
# Purpose:
#   ASA-6B established position_ridge as a reliable answerless
#   DeltaU trajectory modulator.
#
#   ASA-6C no longer asks "does it steer?"
#   It asks:
#
#       What is position_ridge actually reading?
#
# Candidate ontologies:
#   A. Operator precursor:
#       w_DeltaU aligns with w_Oproxy, and mediation through Oproxy explains DeltaU.
#
#   B. Boundary / basin-position readout:
#       w_DeltaU aligns with w_DB, and boundary features explain DeltaU.
#
#   C. Independent order-parameter precursor:
#       w_DeltaU predicts future DeltaU but remains distinct from Oproxy / DB / UK.
#
#   D. Answer artifact:
#       w_DeltaU aligns with answer direction w_C - w_E.
#
# This script is compute-controlled:
#   - No hooked steering loops
#   - One baseline hidden extraction pass
#   - Offline Ridge direction / mediation / alignment audits
#
# Main layers:
#   L17, L18, L19
#
# Outputs:
#   asa6c_outputs/
#     asa6c_config.json
#     asa6c_dataset.csv
#     asa6c_baseline_features.csv
#     asa6c_split.csv
#     asa6c_direction_alignment.csv
#     asa6c_target_prediction.csv
#     asa6c_mediation.csv
#     asa6c_orthogonal_residual_prediction.csv
#     asa6c_answer_alignment.csv
#     asa6c_ontology_scores.csv
#     asa6c_verdict.csv
#     asa6c_summary.json
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
from sklearn.metrics import r2_score, roc_auc_score
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
    save_dir: str = "./asa6c_outputs"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float16"
    seed: int = 42

    n_graphs: int = 96
    max_len: int = 260
    batch_size: int = 4
    test_size: float = 0.30

    # Core layers where ASA-6B found signal.
    probe_layers: Tuple[int, ...] = (17, 18, 19)

    # Need H_{l+1} for Oproxy/velocity.
    extra_layers: Tuple[int, ...] = (20,)

    # Decision layers used to build DeltaU.
    decision_layers: Tuple[int, ...] = tuple(range(20, 26))

    # Ridge settings.
    ridge_alpha: float = 10.0
    min_dir_norm: float = 1e-4

    # GroupKFold for prediction audit.
    n_splits: int = 5

    # Optional: include test split diagnostics.
    include_group_cv: bool = True

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
    pd.DataFrame(rows).to_csv(save_dir / "asa6c_label_tokenization_audit.csv", index=False, encoding="utf-8-sig")
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
    df.to_csv(save_dir / "asa6c_dataset.csv", index=False, encoding="utf-8-sig")
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

# ============================================================
# BASELINE EXTRACTION
# ============================================================

def extract_baseline(model, tokenizer, df, cfg, save_dir):
    print("[EXTRACT] baseline hidden states / decision targets")
    W = get_lm_head_weight(model).detach().float()
    W_device = W.to(model.device)

    n = len(df)
    needed_layers = sorted(set(
        list(cfg.probe_layers)
        + list(cfg.extra_layers)
        + [l + 1 for l in cfg.probe_layers]
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

    # Derived proxy targets.
    dR_mat = out[dR_cols].values.astype(np.float32)
    out["D_B_proxy"] = np.min(np.abs(dR_mat), axis=1).astype(np.float32)       # distance-to-boundary proxy
    out["U_K_proxy"] = (-out["D_B_proxy"]).astype(np.float32)                 # competition intensity proxy
    out["Hshape_slope"] = (out["dR_L25"] - out["dR_L20"]).astype(np.float32)
    out["Hshape_range"] = (np.max(dR_mat, axis=1) - np.min(dR_mat, axis=1)).astype(np.float32)
    out["Hshape_min"] = np.min(dR_mat, axis=1).astype(np.float32)
    out["Hshape_max"] = np.max(dR_mat, axis=1).astype(np.float32)
    out["Hshape_l2"] = np.linalg.norm(dR_mat, axis=1).astype(np.float32)

    out.to_csv(save_dir / "asa6c_baseline_features.csv", index=False, encoding="utf-8-sig")
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
    split_df.to_csv(save_dir / "asa6c_split.csv", index=False, encoding="utf-8-sig")
    return np.asarray(tr), np.asarray(te), split_df

# ============================================================
# DIRECTION / TARGET UTILS
# ============================================================

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

def ridge_cv_predict(X, y, groups, cfg):
    X = X.astype(np.float32)
    y = np.asarray(y, dtype=np.float32)
    groups = np.asarray(groups)

    n_splits = min(cfg.n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    pred = np.zeros_like(y, dtype=np.float32)

    for tr, te in gkf.split(X, y, groups):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=cfg.ridge_alpha)),
        ])
        pipe.fit(X[tr], y[tr])
        pred[te] = pipe.predict(X[te]).astype(np.float32)

    return pred

def residualize(y, covariates):
    y = np.asarray(y, dtype=np.float32)
    C = np.asarray(covariates, dtype=np.float32)
    if C.ndim == 1:
        C = C[:, None]
    C0 = C - C.mean(axis=0, keepdims=True)
    y0 = y - y.mean()
    reg = Ridge(alpha=1.0)
    reg.fit(C0, y0)
    res = y0 - reg.predict(C0)
    return res.astype(np.float32)

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
# PROXY TARGETS
# ============================================================

def build_proxy_targets(base_df, h_by_layer, cfg):
    """
    Build target variables for ontology decomposition.

    DeltaU: order observable.
    D_B_proxy: min abs dR20:25.
    U_K_proxy: -D_B_proxy.
    Hshape_*: trajectory-shape scalars.

    Oproxy_l:
      Hidden transport at l:
        v_l = H_{l+1} - H_l
      We compress v_l into PC1 across all samples.
      This is not O_cont from previous PhaseMap scripts, but a local hidden transport proxy.
      It is enough for "operator precursor vs position readout" screening.
    """
    targets = {
        "DeltaU": base_df["DeltaU"].values.astype(np.float32),
        "D_B_proxy": base_df["D_B_proxy"].values.astype(np.float32),
        "U_K_proxy": base_df["U_K_proxy"].values.astype(np.float32),
        "Hshape_slope": base_df["Hshape_slope"].values.astype(np.float32),
        "Hshape_range": base_df["Hshape_range"].values.astype(np.float32),
        "Hshape_l2": base_df["Hshape_l2"].values.astype(np.float32),
    }

    # Hshape_PC1 over decision dR columns.
    dR_cols = [f"dR_L{l}" for l in cfg.decision_layers]
    Xshape = base_df[dR_cols].values.astype(np.float32)
    pca_shape = PCA(n_components=1)
    pc = pca_shape.fit_transform(Xshape)[:, 0].astype(np.float32)
    if safe_corr(pc, base_df["DeltaU"].values) < 0:
        pc = -pc
    targets["Hshape_PC1"] = pc

    # Layerwise Oproxy.
    for l in cfg.probe_layers:
        if (l + 1) not in h_by_layer:
            continue
        V = h_by_layer[l + 1] - h_by_layer[l]
        pca = PCA(n_components=1)
        op = pca.fit_transform(V)[:, 0].astype(np.float32)
        if safe_corr(op, targets["DeltaU"]) < 0:
            op = -op
        targets[f"Oproxy_L{l}"] = op

    return targets

# ============================================================
# MAIN AUDIT
# ============================================================

def main(cfg):
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "asa6c_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    model, tokenizer = load_model_and_tokenizer(cfg)
    df = build_dataset(tokenizer, cfg, save_dir)

    base_df, h_by_layer, pca_decision = extract_baseline(model, tokenizer, df, cfg, save_dir)
    train_idx, test_idx, split_df = make_split(base_df, cfg, save_dir)
    targets = build_proxy_targets(base_df, h_by_layer, cfg)

    groups = base_df["graph_id"].values
    W = get_lm_head_weight(model).detach().float().cpu().numpy().astype(np.float32)
    clean_ids = base_df["clean_token_id"].values.astype(int)
    conflict_ids = base_df["conflict_token_id"].values.astype(int)
    ans_dirs = W[clean_ids] - W[conflict_ids]
    ans_dir_global = ans_dirs.mean(axis=0).astype(np.float32)
    ans_dir_global, _ = normalize_np(ans_dir_global, eps=cfg.min_dir_norm)

    # --------------------------------------------------------
    # 1) Fit layerwise directions for multiple targets.
    # --------------------------------------------------------
    target_names_global = [
        "DeltaU",
        "D_B_proxy",
        "U_K_proxy",
        "Hshape_slope",
        "Hshape_range",
        "Hshape_l2",
        "Hshape_PC1",
    ]

    direction_dict = {}
    pred_rows = []
    align_rows = []
    answer_rows = []
    orth_rows = []

    for l in cfg.probe_layers:
        X = h_by_layer[l]
        direction_dict[l] = {}

        target_names = list(target_names_global)
        if f"Oproxy_L{l}" in targets:
            target_names.append(f"Oproxy_L{l}")

        # Fit all target directions.
        for tname in target_names:
            y = targets[tname]
            unit, audit = fit_ridge_direction(
                X[train_idx],
                y[train_idx],
                alpha=cfg.ridge_alpha,
                min_norm=cfg.min_dir_norm
            )
            direction_dict[l][tname] = unit

            # group CV prediction from H_l to target
            if cfg.include_group_cv:
                pred = ridge_cv_predict(X, y, groups, cfg)
                cv_r2 = r2_score(y, pred)
                cv_corr = safe_corr(y, pred)
            else:
                cv_r2, cv_corr = np.nan, np.nan

            pred_rows.append({
                "layer": l,
                "target": tname,
                **audit,
                "group_cv_r2": float(cv_r2) if np.isfinite(cv_r2) else np.nan,
                "group_cv_corr": float(cv_corr) if np.isfinite(cv_corr) else np.nan,
            })

        # Pairwise alignment against DeltaU direction.
        w_du = direction_dict[l].get("DeltaU")
        for tname in target_names:
            if tname == "DeltaU":
                continue
            w = direction_dict[l].get(tname)
            align_rows.append({
                "layer": l,
                "source": "DeltaU",
                "target": tname,
                "cosine": vector_cos(w_du, w),
                "abs_cosine": abs(vector_cos(w_du, w)) if np.isfinite(vector_cos(w_du, w)) else np.nan,
            })

        # Answer alignment.
        answer_rows.append({
            "layer": l,
            "target": "DeltaU",
            "cos_to_global_answer_dir": vector_cos(w_du, ans_dir_global),
            "abs_cos_to_global_answer_dir": abs(vector_cos(w_du, ans_dir_global)) if np.isfinite(vector_cos(w_du, ans_dir_global)) else np.nan,
        })

        # Orthogonal residual prediction:
        # score = H_l dot w_DeltaU with components projected out.
        candidates = {
            "answer_orthogonal": [ans_dir_global],
            "Oproxy_orthogonal": [direction_dict[l].get(f"Oproxy_L{l}")],
            "DB_orthogonal": [direction_dict[l].get("D_B_proxy")],
            "UK_orthogonal": [direction_dict[l].get("U_K_proxy")],
            "HshapePC1_orthogonal": [direction_dict[l].get("Hshape_PC1")],
            "O_DB_orthogonal": [direction_dict[l].get(f"Oproxy_L{l}"), direction_dict[l].get("D_B_proxy")],
            "O_DB_Hshape_orthogonal": [direction_dict[l].get(f"Oproxy_L{l}"), direction_dict[l].get("D_B_proxy"), direction_dict[l].get("Hshape_PC1")],
        }

        y_du = targets["DeltaU"]
        base_score = X @ w_du if w_du is not None else np.zeros(len(base_df))
        orth_rows.append({
            "layer": l,
            "orthogonalization": "none",
            "score_corr_DeltaU": safe_corr(base_score, y_du),
            "score_r2_DeltaU": r2_score(y_du, base_score) if np.std(base_score) > 1e-8 else np.nan,
            "direction_valid": bool(w_du is not None),
        })

        for cname, basis in candidates.items():
            w_res = project_out(w_du, basis)
            if w_res is None:
                score = np.zeros(len(base_df))
                valid = False
            else:
                score = X @ w_res
                valid = True
            orth_rows.append({
                "layer": l,
                "orthogonalization": cname,
                "score_corr_DeltaU": safe_corr(score, y_du),
                "score_r2_DeltaU": r2_score(y_du, score) if np.std(score) > 1e-8 else np.nan,
                "direction_valid": bool(valid),
            })

    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(save_dir / "asa6c_target_prediction.csv", index=False, encoding="utf-8-sig")

    align_df = pd.DataFrame(align_rows)
    align_df.to_csv(save_dir / "asa6c_direction_alignment.csv", index=False, encoding="utf-8-sig")

    answer_df = pd.DataFrame(answer_rows)
    answer_df.to_csv(save_dir / "asa6c_answer_alignment.csv", index=False, encoding="utf-8-sig")

    orth_df = pd.DataFrame(orth_rows)
    orth_df.to_csv(save_dir / "asa6c_orthogonal_residual_prediction.csv", index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # 2) Mediation-like audit.
    # --------------------------------------------------------
    # For each layer:
    #   position_score = H_l dot w_DeltaU
    #   O_score        = Oproxy_Ll
    #   DB             = D_B_proxy
    # Regress DeltaU on:
    #   position only
    #   O only
    #   DB only
    #   position + O
    #   position + DB
    #   position + O + DB
    # If position contribution collapses after O/DB, it is mediated.
    mediation_rows = []

    def fit_ridge_simple(Xfeat, y, groups=None):
        Xfeat = np.asarray(Xfeat, dtype=np.float32)
        if Xfeat.ndim == 1:
            Xfeat = Xfeat[:, None]
        y = np.asarray(y, dtype=np.float32)

        if groups is None:
            reg = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ])
            reg.fit(Xfeat, y)
            pred = reg.predict(Xfeat)
            return r2_score(y, pred), safe_corr(y, pred)
        else:
            pred = np.zeros_like(y)
            n_splits = min(cfg.n_splits, len(np.unique(groups)))
            gkf = GroupKFold(n_splits=n_splits)
            for tr, te in gkf.split(Xfeat, y, groups):
                reg = Pipeline([
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=1.0)),
                ])
                reg.fit(Xfeat[tr], y[tr])
                pred[te] = reg.predict(Xfeat[te])
            return r2_score(y, pred), safe_corr(y, pred)

    y_du = targets["DeltaU"]
    for l in cfg.probe_layers:
        X = h_by_layer[l]
        w_du = direction_dict[l].get("DeltaU")
        if w_du is None:
            continue
        pos_score = X @ w_du
        db = targets["D_B_proxy"]
        uk = targets["U_K_proxy"]
        hs = targets["Hshape_PC1"]
        op = targets.get(f"Oproxy_L{l}", np.zeros(len(base_df), dtype=np.float32))

        feature_sets = {
            "position_only": pos_score,
            "Oproxy_only": op,
            "DB_only": db,
            "UK_only": uk,
            "HshapePC1_only": hs,
            "position_plus_O": np.column_stack([pos_score, op]),
            "position_plus_DB": np.column_stack([pos_score, db]),
            "position_plus_UK": np.column_stack([pos_score, uk]),
            "position_plus_HshapePC1": np.column_stack([pos_score, hs]),
            "O_plus_DB": np.column_stack([op, db]),
            "position_plus_O_DB": np.column_stack([pos_score, op, db]),
            "position_plus_O_DB_Hshape": np.column_stack([pos_score, op, db, hs]),
            "all_proxy": np.column_stack([pos_score, op, db, uk, hs]),
        }

        for fname, Xf in feature_sets.items():
            r2, corr = fit_ridge_simple(Xf, y_du, groups=groups)
            mediation_rows.append({
                "layer": l,
                "feature_set": fname,
                "group_cv_r2": float(r2) if np.isfinite(r2) else np.nan,
                "group_cv_corr": float(corr) if np.isfinite(corr) else np.nan,
            })

    mediation_df = pd.DataFrame(mediation_rows)
    mediation_df.to_csv(save_dir / "asa6c_mediation.csv", index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # 3) Ontology scoring.
    # --------------------------------------------------------
    # Heuristic scores:
    #   operator_score: |cos(wDU,wO)| + mediation improvement of O.
    #   boundary_score: |cos(wDU,wDB)| + DB mediation.
    #   order_precursor_score: DU prediction high + residual after O/DB/Hshape remains.
    #   answer_artifact_score: |cos(wDU, answer)|.
    ontology_rows = []

    for l in cfg.probe_layers:
        # Direction cosines
        def get_cos(t):
            r = align_df[(align_df["layer"] == l) & (align_df["target"] == t)]
            return float(r.iloc[0]["abs_cosine"]) if len(r) else np.nan

        cos_o = get_cos(f"Oproxy_L{l}")
        cos_db = get_cos("D_B_proxy")
        cos_uk = get_cos("U_K_proxy")
        cos_hs = get_cos("Hshape_PC1")

        ar = answer_df[answer_df["layer"] == l]
        cos_ans = float(ar.iloc[0]["abs_cos_to_global_answer_dir"]) if len(ar) else np.nan

        # CV target prediction
        pr_du = pred_df[(pred_df["layer"] == l) & (pred_df["target"] == "DeltaU")]
        du_cv = float(pr_du.iloc[0]["group_cv_corr"]) if len(pr_du) else np.nan

        pr_o = pred_df[(pred_df["layer"] == l) & (pred_df["target"] == f"Oproxy_L{l}")]
        o_cv = float(pr_o.iloc[0]["group_cv_corr"]) if len(pr_o) else np.nan

        pr_db = pred_df[(pred_df["layer"] == l) & (pred_df["target"] == "D_B_proxy")]
        db_cv = float(pr_db.iloc[0]["group_cv_corr"]) if len(pr_db) else np.nan

        # Orthogonal survival
        base = orth_df[(orth_df["layer"] == l) & (orth_df["orthogonalization"] == "none")]
        base_corr = float(base.iloc[0]["score_corr_DeltaU"]) if len(base) else np.nan

        def survival(name):
            r = orth_df[(orth_df["layer"] == l) & (orth_df["orthogonalization"] == name)]
            if len(r) == 0 or not np.isfinite(base_corr) or abs(base_corr) < 1e-8:
                return np.nan
            return float(abs(r.iloc[0]["score_corr_DeltaU"]) / (abs(base_corr) + 1e-8))

        surv_ans = survival("answer_orthogonal")
        surv_o = survival("Oproxy_orthogonal")
        surv_db = survival("DB_orthogonal")
        surv_o_db_hs = survival("O_DB_Hshape_orthogonal")

        # Mediation summaries
        med = mediation_df[mediation_df["layer"] == l]
        def med_r2(fs):
            r = med[med["feature_set"] == fs]
            return float(r.iloc[0]["group_cv_r2"]) if len(r) else np.nan

        r2_pos = med_r2("position_only")
        r2_o = med_r2("Oproxy_only")
        r2_db = med_r2("DB_only")
        r2_hs = med_r2("HshapePC1_only")
        r2_pos_o = med_r2("position_plus_O")
        r2_pos_db = med_r2("position_plus_DB")
        r2_pos_o_db_hs = med_r2("position_plus_O_DB_Hshape")

        # Additional explanatory gain over O/DB/Hshape without position
        r2_o_db = med_r2("O_plus_DB")
        pos_increment_over_o_db = r2_pos_o_db_hs - r2_o_db if np.isfinite(r2_pos_o_db_hs) and np.isfinite(r2_o_db) else np.nan

        operator_score = np.nanmean([cos_o, o_cv, 1.0 - surv_o if np.isfinite(surv_o) else np.nan])
        boundary_score = np.nanmean([cos_db, db_cv, 1.0 - surv_db if np.isfinite(surv_db) else np.nan])
        answer_score = np.nanmean([cos_ans, 1.0 - surv_ans if np.isfinite(surv_ans) else np.nan])
        independent_order_score = np.nanmean([
            du_cv,
            surv_o_db_hs,
            pos_increment_over_o_db if np.isfinite(pos_increment_over_o_db) else np.nan
        ])

        ontology_rows.append({
            "layer": l,
            "du_cv_corr": du_cv,
            "cos_DU_Oproxy": cos_o,
            "cos_DU_DB": cos_db,
            "cos_DU_UK": cos_uk,
            "cos_DU_HshapePC1": cos_hs,
            "cos_DU_answer": cos_ans,
            "survival_answer_orthogonal": surv_ans,
            "survival_O_orthogonal": surv_o,
            "survival_DB_orthogonal": surv_db,
            "survival_O_DB_Hshape_orthogonal": surv_o_db_hs,
            "r2_position_only": r2_pos,
            "r2_O_only": r2_o,
            "r2_DB_only": r2_db,
            "r2_HshapePC1_only": r2_hs,
            "r2_position_plus_O": r2_pos_o,
            "r2_position_plus_DB": r2_pos_db,
            "r2_position_plus_O_DB_Hshape": r2_pos_o_db_hs,
            "position_increment_over_O_DB": pos_increment_over_o_db,
            "operator_precursor_score": operator_score,
            "boundary_position_score": boundary_score,
            "answer_artifact_score": answer_score,
            "independent_order_precursor_score": independent_order_score,
        })

    ontology_df = pd.DataFrame(ontology_rows)
    ontology_df.to_csv(save_dir / "asa6c_ontology_scores.csv", index=False, encoding="utf-8-sig")

    # Final verdict aggregation.
    avg = ontology_df.mean(numeric_only=True).to_dict()

    scores = {
        "operator_precursor": avg.get("operator_precursor_score", np.nan),
        "boundary_position": avg.get("boundary_position_score", np.nan),
        "answer_artifact": avg.get("answer_artifact_score", np.nan),
        "independent_order_precursor": avg.get("independent_order_precursor_score", np.nan),
    }
    sorted_scores = sorted(scores.items(), key=lambda x: -np.nan_to_num(x[1], nan=-999))

    if len(sorted_scores) > 0:
        top_label, top_score = sorted_scores[0]
    else:
        top_label, top_score = "undetermined", np.nan

    # Conservative interpretation.
    if top_label == "answer_artifact" and top_score > 0.5:
        verdict_label = "ANSWER_ARTIFACT_RISK"
    elif top_label == "operator_precursor" and top_score > 0.45:
        verdict_label = "OPERATOR_PRECURSOR"
    elif top_label == "boundary_position" and top_score > 0.45:
        verdict_label = "BOUNDARY_POSITION"
    elif top_label == "independent_order_precursor" and top_score > 0.45:
        verdict_label = "ORDER_PRECURSOR"
    else:
        verdict_label = "MIXED_OR_UNRESOLVED"

    verdict_df = pd.DataFrame([{
        "audit": "ASA-6C Position Ridge Ontology Decomposition",
        "verdict": verdict_label,
        "top_score_label": top_label,
        "top_score": top_score,
        **{f"score_{k}": v for k, v in scores.items()},
        "mean_du_cv_corr": avg.get("du_cv_corr", np.nan),
        "mean_cos_DU_Oproxy": avg.get("cos_DU_Oproxy", np.nan),
        "mean_cos_DU_DB": avg.get("cos_DU_DB", np.nan),
        "mean_cos_DU_HshapePC1": avg.get("cos_DU_HshapePC1", np.nan),
        "mean_cos_DU_answer": avg.get("cos_DU_answer", np.nan),
        "mean_survival_answer_orthogonal": avg.get("survival_answer_orthogonal", np.nan),
        "mean_survival_O_orthogonal": avg.get("survival_O_orthogonal", np.nan),
        "mean_survival_DB_orthogonal": avg.get("survival_DB_orthogonal", np.nan),
        "mean_survival_O_DB_Hshape_orthogonal": avg.get("survival_O_DB_Hshape_orthogonal", np.nan),
    }])
    verdict_df.to_csv(save_dir / "asa6c_verdict.csv", index=False, encoding="utf-8-sig")

    summary = {
        "audit": "ASA-6C Position Ridge Ontology Decomposition",
        "config": asdict(cfg),
        "n_prompts": int(len(base_df)),
        "deltaU_pca_explained_variance": float(pca_decision.explained_variance_ratio_[0]),
        "verdict": verdict_label,
        "scores": scores,
        "top_score": {"label": top_label, "value": top_score},
        "key_outputs": {
            "baseline_features": "asa6c_baseline_features.csv",
            "target_prediction": "asa6c_target_prediction.csv",
            "direction_alignment": "asa6c_direction_alignment.csv",
            "mediation": "asa6c_mediation.csv",
            "orthogonal_residual_prediction": "asa6c_orthogonal_residual_prediction.csv",
            "answer_alignment": "asa6c_answer_alignment.csv",
            "ontology_scores": "asa6c_ontology_scores.csv",
            "verdict": "asa6c_verdict.csv",
        },
        "important_caveat": (
            "Oproxy here is a hidden-transport PC1 proxy, not the full PhaseMap O_cont variable. "
            "If the audit suggests operator precursor, confirm later with true O_cont artifacts."
        )
    }

    with open(save_dir / "asa6c_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nASA-6C complete.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main(CFG)
