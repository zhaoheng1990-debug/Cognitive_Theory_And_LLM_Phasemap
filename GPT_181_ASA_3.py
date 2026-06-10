
# ============================================================
# ASA-3: Answerless TopK Flow / Operator Steering
#
# Motivation:
#   ASA-1 / ASA-2 showed that static state steering directions
#   (prototype mean differences, linear predictor coefficients on C_l)
#   do not produce robust mechanism-specific causal control.
#
# ASA-3 changes the intervention object:
#
#   state steering:
#       H_l' = H_l + delta
#
#   flow/operator steering:
#       H_{l+1}' = H_{l+1} + delta_O_l
#
# where delta_O_l is learned from answerless TopK/VIM flow:
#
#       dC_l = C_{l+1} - C_l
#       dC_l -> DeltaU
#
# No answer-token logit/rank/margin is used to build steering directions.
# R_l and DeltaU are used only as evaluation targets.
#
# Default:
#   Qwen2.5-1.5B-Instruct
#   train directions on L0-L19 TopK centers / flows
#   evaluate steering in L9-L14 and L15-L19
#
# Outputs:
#   asa3_outputs/
#       asa3_config.json
#       asa3_dataset.csv
#       asa3_split.csv
#       asa3_baseline_features.csv
#       asa3_flow_direction_audit.csv
#       asa3_steering_results.csv
#       asa3_dose_response.csv
#       asa3_specificity_summary.csv
#       asa3_shuffle_specificity_each.csv
#       asa3_shuffle_specificity_significance.csv
#       asa3_verdict.csv
#       asa3_summary.json
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
from typing import Dict, List, Tuple, Any

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
    save_dir: str = "./asa3_outputs"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float16"
    seed: int = 42

    n_graphs: int = 96
    max_len: int = 260
    batch_size: int = 4
    topk: int = 512

    # For Qwen-1.5B with 28 layers:
    proto_layers: Tuple[int, ...] = tuple(range(0, 20))       # centers / flows source
    decision_layers: Tuple[int, ...] = tuple(range(20, 26))   # DeltaU target

    # Flow l means C_{l+1} - C_l and hook after layer l.
    steering_windows: Dict[str, Tuple[int, ...]] = None

    alphas: Tuple[float, ...] = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10)
    test_size: float = 0.30

    # Main directions:
    #   Q_to_S: closure -> stable, target mechanism = closure
    #   K_to_S: competition -> stable, target mechanism = competition
    run_q_to_s: bool = True
    run_k_to_s: bool = True
    run_s_to_q: bool = False

    # Shared subspace residualization for flow directions.
    shared_pca_dim: int = 8

    # Shuffle distribution for specificity.
    n_shuffles: int = 50
    shuffle_alphas: Tuple[float, ...] = (0.02, 0.05, 0.10)

    ridge_alpha: float = 10.0

    # Controls
    include_answer_control: bool = True
    include_random_control: bool = True
    include_orthogonal_control: bool = True

CFG = Config()
if CFG.steering_windows is None:
    CFG.steering_windows = {
        "L9_14_flow": tuple(range(9, 15)),
        "L15_19_flow": tuple(range(15, 20)),
    }

# ============================================================
# REPRO
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CFG.seed)

def dtype_from_cfg(cfg: Config):
    if cfg.dtype == "float16" and cfg.device == "cuda":
        return torch.float16
    if cfg.dtype == "bfloat16" and cfg.device == "cuda":
        return torch.bfloat16
    return torch.float32

def resolve_model_path(cfg: Config):
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

def mechanism_of(cond: str):
    if cond in STABLE_CONDS:
        return "stable"
    if cond in COMPETITION_CONDS:
        return "competition"
    if cond in CLOSURE_CONDS:
        return "closure"
    raise ValueError(cond)

def continuation_ids(tokenizer, text: str):
    return tokenizer(" " + text, add_special_tokens=False)["input_ids"]

def select_single_token_labels(tokenizer, save_dir: Path):
    rows = []
    chosen = []
    for lab in LABEL_CANDIDATES:
        ids = continuation_ids(tokenizer, lab)
        rows.append({"label": lab, "ids": str(ids), "len": len(ids), "single": int(len(ids) == 1)})
        if len(ids) == 1:
            chosen.append(lab)
    pd.DataFrame(rows).to_csv(save_dir / "asa3_label_tokenization_audit.csv", index=False, encoding="utf-8-sig")
    if len(chosen) < 12:
        raise RuntimeError(f"Need at least 12 single-token labels; got {chosen}")
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

def build_dataset(tokenizer, cfg: Config, save_dir: Path):
    labels = select_single_token_labels(tokenizer, save_dir)
    rows = []
    graph_count = 0
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
    df.to_csv(save_dir / "asa3_dataset.csv", index=False, encoding="utf-8-sig")
    return df

# ============================================================
# MODEL UTILS
# ============================================================

def load_model_and_tokenizer(cfg: Config):
    cfg.model_path = resolve_model_path(cfg)
    print(f"[MODEL] model_key={cfg.model_key} resolved_path={cfg.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
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
    return model, tokenizer

def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise RuntimeError("Cannot locate transformer layers.")

def get_lm_head_weight(model):
    if hasattr(model, "lm_head"):
        return model.lm_head.weight.detach()
    raise RuntimeError("Cannot locate lm_head.")

def last_positions(attention_mask):
    return torch.full(
        (attention_mask.shape[0],),
        attention_mask.shape[1] - 1,
        dtype=torch.long,
        device=attention_mask.device,
    )

def normalize_rows_np(X, eps=1e-8):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

def remove_shared_np(v, basis):
    if basis is None or basis.size == 0:
        return v
    # basis rows are components.
    return v - basis.T @ (basis @ v)

def orthogonalize_to_answer_batch(direction, ans_dir):
    # direction: [d] torch, ans_dir: [B,d] torch
    d = direction[None, :].expand_as(ans_dir)
    denom = torch.sum(ans_dir * ans_dir, dim=1, keepdim=True) + 1e-8
    proj = torch.sum(d * ans_dir, dim=1, keepdim=True) / denom * ans_dir
    out = d - proj
    out = out / (torch.norm(out, dim=1, keepdim=True) + 1e-8)
    return out

# ============================================================
# BASELINE EXTRACTION
# ============================================================

def extract_baseline(model, tokenizer, df, cfg: Config, save_dir: Path):
    print("[EXTRACT] baseline hidden/topk/R")
    W = get_lm_head_weight(model).detach().float()
    W_device = W.to(model.device)
    W_norm = torch.nn.functional.normalize(W_device.float(), dim=1)

    n = len(df)
    all_flow_layers = sorted(set(list(cfg.proto_layers) + [max(cfg.proto_layers) + 1]))
    max_needed_layer = max(max(all_flow_layers), max(cfg.decision_layers))

    centers = {l: None for l in all_flow_layers}
    R_by_layer = {l: np.zeros(n, dtype=np.float32) for l in cfg.decision_layers}
    final_clean_logits = np.zeros(n, dtype=np.float32)
    final_conflict_logits = np.zeros(n, dtype=np.float32)

    texts = df["text"].tolist()
    clean_ids_all = df["clean_token_id"].values.astype(np.int64)
    conflict_ids_all = df["conflict_token_id"].values.astype(np.int64)

    with torch.no_grad():
        for start in range(0, n, cfg.batch_size):
            end = min(n, start + cfg.batch_size)
            batch_texts = texts[start:end]
            inputs = tokenizer(
                batch_texts,
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

            for l in all_flow_layers:
                h = hstates[l + 1][idx_t, pos, :].detach().float()
                logits = h @ W_device.float().T
                vals, ids = torch.topk(logits, k=cfg.topk, dim=1)
                emb = W_norm[ids].float()
                center = emb.mean(dim=1)
                center = torch.nn.functional.normalize(center, dim=1)
                if centers[l] is None:
                    centers[l] = np.zeros((n, center.shape[1]), dtype=np.float32)
                centers[l][start:end] = center.detach().cpu().numpy().astype(np.float32)

            for l in cfg.decision_layers:
                h = hstates[l + 1][idx_t, pos, :].detach().float()
                clean_logits = torch.sum(h * W_device[cids].float(), dim=1)
                conflict_logits = torch.sum(h * W_device[eids].float(), dim=1)
                R_by_layer[l][start:end] = (clean_logits - conflict_logits).detach().cpu().numpy().astype(np.float32)

            # final next-token logits
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
    # Orient stable high, closure low.
    if np.mean(du[out["mechanism"] == "stable"]) < np.mean(du[out["mechanism"] == "closure"]):
        du = -du
        pca.components_[0] *= -1
    out["DeltaU"] = du.astype(np.float32)
    out.to_csv(save_dir / "asa3_baseline_features.csv", index=False, encoding="utf-8-sig")

    print("[PCA] DeltaU explained variance:", float(pca.explained_variance_ratio_[0]))
    return out, centers, pca

# ============================================================
# SPLIT
# ============================================================

def make_split(df, cfg: Config, save_dir: Path):
    graph_ids = df["graph_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.seed)
    tr, te = next(gss.split(df, groups=graph_ids))
    split_df = pd.DataFrame({
        "prompt_id": df["prompt_id"],
        "graph_id": df["graph_id"],
        "condition": df["condition"],
        "mechanism": df["mechanism"],
        "split": ["test" if i in set(te) else "train" for i in range(len(df))]
    })
    split_df.to_csv(save_dir / "asa3_split.csv", index=False, encoding="utf-8-sig")
    return np.asarray(tr), np.asarray(te), split_df

# ============================================================
# FLOW DIRECTIONS
# ============================================================

def fit_flow_directions(base_df, centers, train_idx, cfg: Config, save_dir: Path):
    """
    Fit per-layer Ridge:
        dC_l = C_{l+1} - C_l -> DeltaU
    Direction is ridge coef vector in hidden dimension.
    """
    y = base_df["DeltaU"].values.astype(np.float32)
    flow_layers = list(cfg.proto_layers)
    dirs_raw = {}
    dirs_resid = {}
    dirs_shuffle = {}
    random_dirs = {}
    audits = []
    all_train_flows = []

    # Shared flow subspace over all train dC.
    for l in flow_layers:
        if (l + 1) not in centers:
            continue
        dC = centers[l + 1] - centers[l]
        all_train_flows.append(dC[train_idx])
    all_train_flows = np.concatenate(all_train_flows, axis=0)
    pca_shared = PCA(n_components=min(cfg.shared_pca_dim, all_train_flows.shape[1]))
    pca_shared.fit(all_train_flows)
    shared_basis = pca_shared.components_.astype(np.float32)

    rng = np.random.default_rng(cfg.seed + 777)

    for l in flow_layers:
        if (l + 1) not in centers:
            continue
        dC = centers[l + 1] - centers[l]
        Xtr = dC[train_idx].astype(np.float32)
        ytr = y[train_idx].astype(np.float32)

        reg = Ridge(alpha=cfg.ridge_alpha)
        reg.fit(Xtr, ytr)
        pred = reg.predict(Xtr)
        r2 = r2_score(ytr, pred)
        corr = float(np.corrcoef(ytr, pred)[0, 1]) if np.std(pred) > 1e-8 else 0.0

        coef = reg.coef_.astype(np.float32)
        coef = coef / (np.linalg.norm(coef) + 1e-8)
        coef_res = remove_shared_np(coef, shared_basis)
        coef_res = coef_res / (np.linalg.norm(coef_res) + 1e-8)

        dirs_raw[l] = coef
        dirs_resid[l] = coef_res

        rd = rng.standard_normal(size=coef.shape).astype(np.float32)
        rd = rd / (np.linalg.norm(rd) + 1e-8)
        random_dirs[l] = rd

        audits.append({
            "layer": l,
            "train_r2_dC_to_DeltaU": float(r2),
            "train_corr_dC_to_DeltaU": float(corr),
            "coef_norm": float(np.linalg.norm(reg.coef_)),
            "residual_norm_after_shared_removal": float(np.linalg.norm(coef_res)),
        })

    # Shuffle directions: y permuted, fit same Ridge.
    for s in range(cfg.n_shuffles):
        rng_s = np.random.default_rng(cfg.seed + 10000 + s)
        y_perm = y[train_idx].copy()
        rng_s.shuffle(y_perm)
        dirs_s = {}
        for l in flow_layers:
            if (l + 1) not in centers:
                continue
            dC = centers[l + 1] - centers[l]
            Xtr = dC[train_idx].astype(np.float32)
            reg = Ridge(alpha=cfg.ridge_alpha)
            reg.fit(Xtr, y_perm)
            coef = reg.coef_.astype(np.float32)
            coef = coef / (np.linalg.norm(coef) + 1e-8)
            coef_res = remove_shared_np(coef, shared_basis)
            coef_res = coef_res / (np.linalg.norm(coef_res) + 1e-8)
            dirs_s[l] = coef_res
        dirs_shuffle[s] = dirs_s

    pd.DataFrame(audits).to_csv(save_dir / "asa3_flow_direction_audit.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        save_dir / "asa3_flow_dirs_and_shared_basis.npz",
        shared_basis=shared_basis,
        **{f"raw_L{l}": v for l, v in dirs_raw.items()},
        **{f"resid_L{l}": v for l, v in dirs_resid.items()},
        **{f"random_L{l}": v for l, v in random_dirs.items()},
    )
    return dirs_raw, dirs_resid, random_dirs, dirs_shuffle, shared_basis

# ============================================================
# STEERED FORWARD
# ============================================================

def make_hook(direction_by_layer: Dict[int, np.ndarray],
              layers_to_hook: List[int],
              alpha: float,
              cfg: Config,
              mode: str,
              W_device,
              clean_ids_batch=None,
              conflict_ids_batch=None):
    """
    Hook modifies layer output after transformer layer l.
    Direction is added to all last positions? Simpler and robust: add to the entire sequence hidden states.
    Since prompt is short and only final position matters, adding to all tokens is a stronger operator-flow perturbation.
    To make it less global, set only_last_token=True in future.
    """
    dir_tensors = {
        l: torch.tensor(direction_by_layer[l], dtype=torch.float32, device=W_device.device)
        for l in layers_to_hook if l in direction_by_layer
    }

    def hook_factory(l):
        def hook(module, inputs, output):
            if l not in dir_tensors or alpha == 0.0:
                return output

            if isinstance(output, tuple):
                h = output[0]
                rest = output[1:]
            else:
                h = output
                rest = None

            d = dir_tensors[l].to(device=h.device, dtype=h.dtype)

            # Per-sample answer direction if needed.
            if mode == "answer":
                ans = W_device[clean_ids_batch].float() - W_device[conflict_ids_batch].float()
                ans = ans / (torch.norm(ans, dim=1, keepdim=True) + 1e-8)
                d_batch = ans.to(device=h.device, dtype=h.dtype)
            elif mode.startswith("orthogonal"):
                ans = W_device[clean_ids_batch].float() - W_device[conflict_ids_batch].float()
                ans = ans / (torch.norm(ans, dim=1, keepdim=True) + 1e-8)
                d_batch = orthogonalize_to_answer_batch(d.float(), ans).to(dtype=h.dtype, device=h.device)
            else:
                d_batch = d[None, :].expand(h.shape[0], -1)

            # Scale by per-sample hidden std.
            scale = h.float().std(dim=-1, keepdim=True).mean(dim=1, keepdim=True)  # [B,1]
            scale = scale.to(device=h.device, dtype=h.dtype)

            # Add to all sequence positions as a layer-output transport perturbation.
            # h is [B, S, D]. scale is [B, 1, 1]; d_batch[:, None, :] is [B, 1, D].
            # Do NOT insert an extra dimension into scale; otherwise h becomes 4D and Qwen RoPE breaks.
            perturb = alpha * scale * d_batch[:, None, :]
            h2 = h + perturb

            if rest is not None:
                return (h2,) + rest
            return h2
        return hook
    return hook_factory

def forward_eval(model, tokenizer, df_subset, cfg: Config, pca_decision,
                 clean_anchor_R: Dict[int, Dict[int, float]],
                 direction_by_layer=None,
                 layers_to_hook=None,
                 alpha=0.0,
                 control_mode="none"):
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
            if direction_by_layer is not None and layers_to_hook is not None and alpha != 0.0:
                hook_factory = make_hook(
                    direction_by_layer=direction_by_layer,
                    layers_to_hook=list(layers_to_hook),
                    alpha=alpha,
                    cfg=cfg,
                    mode=control_mode,
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
            bsz = end - start
            idx_t = torch.arange(bsz, device=model.device)

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
# EVAL SUMMARIES
# ============================================================

def summarize_specificity(results_df, base_df, direction_targets):
    """
    Specificity = target_abs_shift - nontarget_abs_shift
    """
    rows = []
    merged = results_df.merge(
        base_df[["prompt_id", "DeltaU", "pair_choice"]],
        on="prompt_id",
        suffixes=("", "_base")
    )
    merged["shift"] = merged["DeltaU"] - merged["DeltaU_base"]
    merged["abs_shift"] = merged["shift"].abs()
    merged["target_hit"] = np.where(merged["target_pair_choice"] == merged["pair_choice"], 1, 0)

    for (direction, control, window, alpha), g in merged.groupby(["direction", "control", "window", "alpha"]):
        target_mech = direction_targets[direction]
        target = g[g["mechanism"] == target_mech]
        nontarget = g[g["mechanism"] != target_mech]
        rows.append({
            "direction": direction,
            "control": control,
            "window": window,
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
    for keys, g in spec_df.groupby(["direction", "control", "window"]):
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
            slope = np.nan
            r2 = np.nan
        rows.append({
            "direction": keys[0],
            "control": keys[1],
            "window": keys[2],
            "specificity_slope_vs_alpha": slope,
            "specificity_r2_vs_alpha": float(r2) if np.isfinite(r2) else np.nan,
        })
    return pd.DataFrame(rows)

def shuffle_specificity_significance(spec_df, cfg: Config):
    rows = []
    for direction in sorted(spec_df["direction"].unique()):
        for window in sorted(spec_df["window"].unique()):
            for alpha in cfg.shuffle_alphas:
                main = spec_df[
                    (spec_df["direction"] == direction) &
                    (spec_df["window"] == window) &
                    (np.isclose(spec_df["alpha"], alpha)) &
                    (spec_df["control"] == "predictor_flow_residual")
                ]
                if len(main) == 0:
                    continue
                proto_spec = float(main.iloc[0]["specificity"])

                sh = spec_df[
                    (spec_df["direction"] == direction) &
                    (spec_df["window"] == window) &
                    (np.isclose(spec_df["alpha"], alpha)) &
                    (spec_df["control"].str.startswith("shuffle_flow_residual_"))
                ]["specificity"].values.astype(float)

                rand = spec_df[
                    (spec_df["direction"] == direction) &
                    (spec_df["window"] == window) &
                    (np.isclose(spec_df["alpha"], alpha)) &
                    (spec_df["control"] == "random")
                ]
                rand_spec = float(rand.iloc[0]["specificity"]) if len(rand) else np.nan

                orth = spec_df[
                    (spec_df["direction"] == direction) &
                    (spec_df["window"] == window) &
                    (np.isclose(spec_df["alpha"], alpha)) &
                    (spec_df["control"] == "orthogonal_predictor_flow_residual")
                ]
                orth_spec = float(orth.iloc[0]["specificity"]) if len(orth) else np.nan

                mean = float(np.nanmean(sh)) if len(sh) else np.nan
                std = float(np.nanstd(sh) + 1e-8) if len(sh) else np.nan
                z = float((proto_spec - mean) / std) if len(sh) else np.nan

                rows.append({
                    "direction": direction,
                    "window": window,
                    "alpha": float(alpha),
                    "proto_specificity": proto_spec,
                    "shuffle_specificity_mean": mean,
                    "shuffle_specificity_std": std,
                    "shuffle_z_specificity": z,
                    "random_specificity": rand_spec,
                    "proto_minus_random_specificity": proto_spec - rand_spec if np.isfinite(rand_spec) else np.nan,
                    "orthogonal_specificity": orth_spec,
                    "orthogonal_survival_ratio": abs(orth_spec) / (abs(proto_spec) + 1e-8) if np.isfinite(orth_spec) else np.nan,
                    "pass_lite": bool(proto_spec > 0 and np.isfinite(rand_spec) and proto_spec > rand_spec),
                    "pass_strong_z2": bool(proto_spec > 0 and np.isfinite(z) and z > 2.0),
                    "pass_very_strong_hint": bool(proto_spec > 0 and np.isfinite(z) and z > 2.0 and np.isfinite(orth_spec) and abs(orth_spec) / (abs(proto_spec) + 1e-8) > 0.3),
                })
    return pd.DataFrame(rows)

# ============================================================
# MAIN
# ============================================================

def main(cfg: Config):
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "asa3_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    model, tokenizer = load_model_and_tokenizer(cfg)
    df = build_dataset(tokenizer, cfg, save_dir)

    base_df, centers, pca_decision = extract_baseline(model, tokenizer, df, cfg, save_dir)
    train_idx, test_idx, split_df = make_split(base_df, cfg, save_dir)

    # Clean anchor R for DeltaU transform under interventions.
    clean_anchor_R = {}
    clean_base = base_df[base_df["condition"] == "clean"].set_index("graph_id")
    for g, row in clean_base.iterrows():
        clean_anchor_R[int(g)] = {l: float(row[f"R_L{l}"]) for l in cfg.decision_layers}

    dirs_raw, dirs_resid, random_dirs, dirs_shuffle, shared_basis = fit_flow_directions(
        base_df, centers, train_idx, cfg, save_dir
    )

    test_df = base_df.iloc[test_idx].reset_index(drop=True)
    base_test = base_df.iloc[test_idx][["prompt_id", "DeltaU", "pair_choice"]].reset_index(drop=True)

    # Direction targets.
    directions = []
    direction_targets = {}
    if cfg.run_k_to_s:
        directions.append("K_to_S")
        direction_targets["K_to_S"] = "competition"
    if cfg.run_q_to_s:
        directions.append("Q_to_S")
        direction_targets["Q_to_S"] = "closure"
    if cfg.run_s_to_q:
        directions.append("S_to_Q")
        direction_targets["S_to_Q"] = "stable"

    # Controls.
    control_dir_map = {
        "predictor_flow_raw": dirs_raw,
        "predictor_flow_residual": dirs_resid,
        "random": random_dirs,
    }
    if cfg.include_orthogonal_control:
        control_dir_map["orthogonal_predictor_flow_residual"] = dirs_resid
    if cfg.include_answer_control:
        # answer control uses dirs_resid just as placeholder; hook mode uses answer direction instead.
        control_dir_map["answer"] = dirs_resid

    # Add shuffle controls.
    for s, dct in dirs_shuffle.items():
        control_dir_map[f"shuffle_flow_residual_{s:03d}"] = dct

    results = []
    for direction in directions:
        # Desired target pair choice.
        # K/Q -> S: target clean. S -> Q: target conflict.
        target_pair = "conflict" if direction == "S_to_Q" else "clean"

        # Sign of predictor direction.
        # DeltaU is oriented stable high, closure low.
        # K/Q -> S should increase DeltaU; S->Q should decrease DeltaU.
        sign = -1.0 if direction == "S_to_Q" else +1.0

        for window_name, layers in cfg.steering_windows.items():
            layers = list(layers)

            for alpha in cfg.alphas:
                for control, dirs in control_dir_map.items():
                    # Reduce shuffle cost: for shuffle controls, only run requested alpha values.
                    if control.startswith("shuffle_flow_residual_") and alpha not in cfg.shuffle_alphas:
                        continue

                    # Apply sign to direction dictionaries unless answer/random? Random can also be signed without effect.
                    signed_dirs = {}
                    for l, v in dirs.items():
                        signed_dirs[l] = (sign * v).astype(np.float32)

                    if control == "answer":
                        mode = "answer"
                    elif control == "orthogonal_predictor_flow_residual":
                        mode = "orthogonal_predictor_flow_residual"
                    else:
                        mode = control

                    print(f"[STEER] dir={direction} window={window_name} alpha={alpha} control={control}")
                    out = forward_eval(
                        model=model,
                        tokenizer=tokenizer,
                        df_subset=test_df,
                        cfg=cfg,
                        pca_decision=pca_decision,
                        clean_anchor_R=clean_anchor_R,
                        direction_by_layer=signed_dirs,
                        layers_to_hook=layers,
                        alpha=float(alpha),
                        control_mode=mode,
                    )
                    out["direction"] = direction
                    out["window"] = window_name
                    out["alpha"] = float(alpha)
                    out["control"] = control
                    out["target_pair_choice"] = target_pair
                    results.append(out)

    results_df = pd.concat(results, ignore_index=True)
    results_df.to_csv(save_dir / "asa3_steering_results.csv", index=False, encoding="utf-8-sig")

    spec_df, merged = summarize_specificity(results_df, base_test, direction_targets)
    spec_df.to_csv(save_dir / "asa3_specificity_summary.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(save_dir / "asa3_merged_shift_table.csv", index=False, encoding="utf-8-sig")

    dose_df = dose_response(spec_df)
    dose_df.to_csv(save_dir / "asa3_dose_response.csv", index=False, encoding="utf-8-sig")

    sig_df = shuffle_specificity_significance(spec_df, cfg)
    sig_df.to_csv(save_dir / "asa3_shuffle_specificity_significance.csv", index=False, encoding="utf-8-sig")

    # Main verdict at alpha=max(shuffle_alphas), predictor_flow_residual, each direction/window.
    verdict = sig_df[np.isclose(sig_df["alpha"], max(cfg.shuffle_alphas))].copy()
    verdict.to_csv(save_dir / "asa3_verdict.csv", index=False, encoding="utf-8-sig")

    summary = {
        "audit": "ASA-3 Answerless TopK Flow / Operator Steering",
        "config": asdict(cfg),
        "n_test_prompts": int(len(test_df)),
        "deltaU_pca_explained_variance": float(pca_decision.explained_variance_ratio_[0]),
        "baseline_pair_choice_rate_test": base_test["pair_choice"].value_counts(normalize=True).to_dict(),
        "key_outputs": {
            "steering_results": "asa3_steering_results.csv",
            "specificity": "asa3_specificity_summary.csv",
            "dose_response": "asa3_dose_response.csv",
            "shuffle_specificity_significance": "asa3_shuffle_specificity_significance.csv",
            "verdict": "asa3_verdict.csv",
            "flow_direction_audit": "asa3_flow_direction_audit.csv",
        },
        "note": (
            "ASA-3 steers layer outputs with directions derived from answerless TopK-center flow "
            "dC_l = C_{l+1} - C_l -> DeltaU. Answer-token logits/ranks/margins are not used to build "
            "steering directions. R_L and DeltaU are used only as evaluation targets."
        )
    }
    with open(save_dir / "asa3_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nASA-3 complete.")
    print(verdict)

if __name__ == "__main__":
    main(CFG)
