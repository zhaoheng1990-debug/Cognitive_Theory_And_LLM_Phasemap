
# ============================================================
# ASA-4: Operator-Matched Steering
#
# Core idea:
#   ASA-1/2/3 tried linear additive directions:
#       H' = H + alpha * delta
#
#   ASA-4 tests a different control object:
#       mechanism-conditioned local transition operators.
#
#   For a mechanism M, fit a low-dimensional local operator:
#       z_{l+1} = A_M^{(l)} z_l + b_M^{(l)}
#
#   Then for a target closure prompt, partially replace the real layer output:
#       H_{l+1}' = (1-alpha) H_{l+1}^{real} + alpha * PCA^{-1}(A_S z_l + b_S)
#
#   This tests whether a learned stable convergence operator can steer closure
#   prompts toward the stable basin better than identity/random/shuffle controls.
#
# Notes:
#   - Answer-token logits/ranks/margins are not used to build operators.
#   - R_L and DeltaU are used only as evaluation targets.
#   - Default is Qwen, Q->S, L15-L19.
#
# Outputs:
#   asa4_outputs/
#       asa4_summary.json
#       asa4_verdict.csv
#       asa4_specificity_summary.csv
#       asa4_shuffle_specificity_significance.csv
#       asa4_steering_results.csv
#       asa4_operator_audit.csv
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
    save_dir: str = "./asa4_outputs"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float16"
    seed: int = 42

    n_graphs: int = 96
    max_len: int = 260
    batch_size: int = 4
    test_size: float = 0.30

    # Qwen-1.5B layer conventions.
    operator_layers: Tuple[int, ...] = tuple(range(15, 20))   # fit/intervene l -> l+1
    decision_layers: Tuple[int, ...] = tuple(range(20, 26))

    alphas: Tuple[float, ...] = (0.0, 0.02, 0.05, 0.10, 0.20, 0.30)

    # Low-dimensional operator.
    pca_dim: int = 128
    ridge_alpha: float = 10.0

    # Main steering directions.
    run_q_to_s: bool = True
    run_k_to_s: bool = False
    run_s_to_q: bool = False

    # Shuffle distribution.
    n_shuffles: int = 50
    shuffle_alphas: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)

    # Controls.
    include_stable_operator: bool = True
    include_self_operator: bool = True       # closure operator for Q->S target / competition operator for K->S
    include_identity_control: bool = True
    include_random_operator: bool = True
    include_answer_control: bool = True

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
    pd.DataFrame(rows).to_csv(save_dir / "asa4_label_tokenization_audit.csv", index=False, encoding="utf-8-sig")
    if len(chosen) < 12:
        raise RuntimeError(f"Need >=12 single token labels; got {chosen}")
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
    df.to_csv(save_dir / "asa4_dataset.csv", index=False, encoding="utf-8-sig")
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

# ============================================================
# BASELINE EXTRACTION
# ============================================================

def extract_baseline(model, tokenizer, df, cfg, save_dir):
    print("[EXTRACT] baseline hidden states and targets")
    W = get_lm_head_weight(model).detach().float()
    W_device = W.to(model.device)

    n = len(df)
    needed_layers = sorted(set(list(cfg.operator_layers) + [l+1 for l in cfg.operator_layers] + list(cfg.decision_layers)))
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
    out.to_csv(save_dir / "asa4_baseline_features.csv", index=False, encoding="utf-8-sig")
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
    split_df.to_csv(save_dir / "asa4_split.csv", index=False, encoding="utf-8-sig")
    return np.asarray(tr), np.asarray(te), split_df

# ============================================================
# OPERATORS
# ============================================================

class LowDimOperator:
    def __init__(self, pca, ridge, layer):
        self.pca = pca
        self.ridge = ridge
        self.layer = layer

    def predict_torch(self, h_in: torch.Tensor) -> torch.Tensor:
        """
        h_in: [B,D] float torch
        returns h_pred_next: [B,D]
        Uses sklearn fitted pca/ridge converted to torch tensors on fly.
        """
        device = h_in.device
        dtype = h_in.dtype

        mean = torch.tensor(self.pca.mean_, device=device, dtype=dtype)
        comps = torch.tensor(self.pca.components_, device=device, dtype=dtype)  # [r,D]

        coef = torch.tensor(self.ridge.coef_, device=device, dtype=dtype)       # [r,r]
        intercept = torch.tensor(self.ridge.intercept_, device=device, dtype=dtype)  # [r]

        z = (h_in - mean) @ comps.T
        z_next = z @ coef.T + intercept
        h_pred = z_next @ comps + mean
        return h_pred

def fit_mechanism_operators(base_df, h_by_layer, train_idx, cfg, save_dir):
    """
    For each layer l and mechanism M, fit low-dim z_{l+1}=A_M z_l+b.
    PCA is fitted per layer on combined H_l/H_{l+1} from train.
    """
    operators = {}
    audits = []
    rng = np.random.default_rng(cfg.seed + 1234)

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
        for mech in ["stable", "competition", "closure"]:
            idx = [i for i in train_idx if base_df.iloc[i]["mechanism"] == mech]
            idx = np.asarray(idx, dtype=int)
            if len(idx) < dim + 2:
                # Ridge can still fit with fewer samples, but audit notes it.
                pass
            reg = Ridge(alpha=cfg.ridge_alpha)
            reg.fit(z0_all[idx], z1_all[idx])
            pred = reg.predict(z0_all[idx])
            r2 = r2_score(z1_all[idx], pred)
            corr = np.corrcoef(z1_all[idx].ravel(), pred.ravel())[0, 1] if np.std(pred) > 1e-8 else 0.0
            operators[l][mech] = LowDimOperator(pca, reg, l)
            audits.append({
                "layer": l,
                "mechanism": mech,
                "n_train": int(len(idx)),
                "pca_dim": int(dim),
                "pca_var_sum": float(np.sum(pca.explained_variance_ratio_)),
                "train_r2_znext": float(r2),
                "train_flat_corr_znext": float(corr),
            })

        # Random operator control: random affine in z-space near identity-ish.
        rreg = Ridge(alpha=cfg.ridge_alpha)
        # Fit random target permutation to preserve scale but destroy mechanism.
        perm = train_idx.copy()
        rng.shuffle(perm)
        rreg.fit(z0_all[train_idx], z1_all[perm])
        operators[l]["random"] = LowDimOperator(pca, rreg, l)

    # Shuffle operators: mechanism labels shuffled, stable pseudo-operator.
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
            # reuse stable pca from fitted operator
            pca = operators[l]["stable"].pca
            z0_all = pca.transform(H0)
            z1_all = pca.transform(H1)
            idx = train_idx[shuffled_mech[train_idx] == "stable"]
            if len(idx) < 4:
                idx = train_idx
            reg = Ridge(alpha=cfg.ridge_alpha)
            reg.fit(z0_all[idx], z1_all[idx])
            shuffle_ops[s][l] = LowDimOperator(pca, reg, l)

    pd.DataFrame(audits).to_csv(save_dir / "asa4_operator_audit.csv", index=False, encoding="utf-8-sig")
    return operators, shuffle_ops

# ============================================================
# STEERED FORWARD
# ============================================================

def make_operator_hook(operators_for_layers,
                       layers_to_hook,
                       alpha,
                       mode,
                       W_device,
                       clean_ids_batch=None,
                       conflict_ids_batch=None):
    def hook_factory(l):
        op = operators_for_layers.get(l, None)

        def hook(module, inputs, output):
            if alpha == 0.0:
                return output

            if isinstance(output, tuple):
                h_out = output[0]
                rest = output[1:]
            else:
                h_out = output
                rest = None

            if l not in layers_to_hook:
                return output

            h_in = inputs[0]  # [B,S,D]
            B, S, D = h_out.shape
            pos = torch.full((B,), S - 1, dtype=torch.long, device=h_out.device)
            idx = torch.arange(B, device=h_out.device)

            h_real_last = h_out[idx, pos, :]
            h_in_last = h_in[idx, pos, :]

            if mode == "identity":
                h_target_last = h_in_last
            elif mode == "answer":
                ans = W_device[clean_ids_batch].float() - W_device[conflict_ids_batch].float()
                ans = ans / (torch.norm(ans, dim=1, keepdim=True) + 1e-8)
                scale = h_out.float().std(dim=-1).mean(dim=1, keepdim=True).to(h_out.dtype)
                h_target_last = h_real_last + scale * ans.to(h_out.dtype)
            else:
                if op is None:
                    return output
                h_target_last = op.predict_torch(h_in_last.float()).to(h_out.dtype)

            h_new_last = (1.0 - alpha) * h_real_last + alpha * h_target_last

            h2 = h_out.clone()
            h2[idx, pos, :] = h_new_last

            if rest is not None:
                return (h2,) + rest
            return h2
        return hook
    return hook_factory

def forward_eval(model, tokenizer, df_subset, cfg, pca_decision,
                 clean_anchor_R, operators_for_layers=None,
                 layers_to_hook=None, alpha=0.0, mode="none"):
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
                hook_factory = make_operator_hook(
                    operators_for_layers=operators_for_layers or {},
                    layers_to_hook=set(layers_to_hook),
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
# SUMMARIES
# ============================================================

def summarize_specificity(results_df, base_df, direction_targets):
    merged = results_df.merge(
        base_df[["prompt_id", "DeltaU", "pair_choice"]],
        on="prompt_id",
        suffixes=("", "_base"),
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
    for direction in sorted(spec_df["direction"].unique()):
        for alpha in cfg.shuffle_alphas:
            main = spec_df[
                (spec_df["direction"] == direction) &
                (np.isclose(spec_df["alpha"], alpha)) &
                (spec_df["control"] == "stable_operator")
            ]
            if len(main) == 0:
                continue
            proto_spec = float(main.iloc[0]["specificity"])
            sh = spec_df[
                (spec_df["direction"] == direction) &
                (np.isclose(spec_df["alpha"], alpha)) &
                (spec_df["control"].str.startswith("shuffle_stable_operator_"))
            ]["specificity"].values.astype(float)

            identity = spec_df[
                (spec_df["direction"] == direction) &
                (np.isclose(spec_df["alpha"], alpha)) &
                (spec_df["control"] == "identity")
            ]
            identity_spec = float(identity.iloc[0]["specificity"]) if len(identity) else np.nan

            selfop = spec_df[
                (spec_df["direction"] == direction) &
                (np.isclose(spec_df["alpha"], alpha)) &
                (spec_df["control"] == "self_operator")
            ]
            self_spec = float(selfop.iloc[0]["specificity"]) if len(selfop) else np.nan

            rand = spec_df[
                (spec_df["direction"] == direction) &
                (np.isclose(spec_df["alpha"], alpha)) &
                (spec_df["control"] == "random_operator")
            ]
            rand_spec = float(rand.iloc[0]["specificity"]) if len(rand) else np.nan

            mean = float(np.nanmean(sh)) if len(sh) else np.nan
            std = float(np.nanstd(sh) + 1e-8) if len(sh) else np.nan
            z = float((proto_spec - mean) / std) if len(sh) else np.nan

            rows.append({
                "direction": direction,
                "alpha": float(alpha),
                "stable_operator_specificity": proto_spec,
                "shuffle_specificity_mean": mean,
                "shuffle_specificity_std": std,
                "shuffle_z_specificity": z,
                "identity_specificity": identity_spec,
                "self_operator_specificity": self_spec,
                "random_operator_specificity": rand_spec,
                "stable_minus_identity": proto_spec - identity_spec if np.isfinite(identity_spec) else np.nan,
                "stable_minus_self": proto_spec - self_spec if np.isfinite(self_spec) else np.nan,
                "stable_minus_random": proto_spec - rand_spec if np.isfinite(rand_spec) else np.nan,
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
    with open(save_dir / "asa4_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    model, tokenizer = load_model_and_tokenizer(cfg)
    df = build_dataset(tokenizer, cfg, save_dir)

    base_df, h_by_layer, pca_decision = extract_baseline(model, tokenizer, df, cfg, save_dir)
    train_idx, test_idx, split_df = make_split(base_df, cfg, save_dir)

    clean_anchor_R = {}
    clean_base = base_df[base_df["condition"] == "clean"].set_index("graph_id")
    for g, row in clean_base.iterrows():
        clean_anchor_R[int(g)] = {l: float(row[f"R_L{l}"]) for l in cfg.decision_layers}

    operators, shuffle_ops = fit_mechanism_operators(base_df, h_by_layer, train_idx, cfg, save_dir)

    test_df = base_df.iloc[test_idx].reset_index(drop=True)
    base_test = base_df.iloc[test_idx][["prompt_id", "DeltaU", "pair_choice"]].reset_index(drop=True)

    directions, direction_targets = [], {}
    if cfg.run_q_to_s:
        directions.append("Q_to_S")
        direction_targets["Q_to_S"] = "closure"
    if cfg.run_k_to_s:
        directions.append("K_to_S")
        direction_targets["K_to_S"] = "competition"
    if cfg.run_s_to_q:
        directions.append("S_to_Q")
        direction_targets["S_to_Q"] = "stable"

    results = []
    layers = list(cfg.operator_layers)

    for direction in directions:
        target_pair = "conflict" if direction == "S_to_Q" else "clean"
        if direction == "Q_to_S":
            self_mech = "closure"
        elif direction == "K_to_S":
            self_mech = "competition"
        else:
            self_mech = "stable"

        for alpha in cfg.alphas:
            control_ops = {}

            # Main stable operator.
            if cfg.include_stable_operator:
                control_ops["stable_operator"] = {l: operators[l]["stable"] for l in layers}

            # Self operator control.
            if cfg.include_self_operator:
                control_ops["self_operator"] = {l: operators[l][self_mech] for l in layers}

            # Identity control.
            if cfg.include_identity_control:
                control_ops["identity"] = {}

            # Random operator.
            if cfg.include_random_operator:
                control_ops["random_operator"] = {l: operators[l]["random"] for l in layers}

            # Answer direction positive control.
            if cfg.include_answer_control:
                control_ops["answer"] = {}

            # Shuffle stable operator controls only at selected alphas.
            if alpha in cfg.shuffle_alphas:
                for s, ops_s in shuffle_ops.items():
                    control_ops[f"shuffle_stable_operator_{s:03d}"] = ops_s

            for control, ops_for_layers in control_ops.items():
                if control == "identity":
                    mode = "identity"
                elif control == "answer":
                    mode = "answer"
                else:
                    mode = "operator"

                print(f"[STEER] direction={direction} alpha={alpha} control={control}")
                out = forward_eval(
                    model=model,
                    tokenizer=tokenizer,
                    df_subset=test_df,
                    cfg=cfg,
                    pca_decision=pca_decision,
                    clean_anchor_R=clean_anchor_R,
                    operators_for_layers=ops_for_layers,
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
    results_df.to_csv(save_dir / "asa4_steering_results.csv", index=False, encoding="utf-8-sig")

    spec_df, merged = summarize_specificity(results_df, base_test, direction_targets)
    spec_df.to_csv(save_dir / "asa4_specificity_summary.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(save_dir / "asa4_merged_shift_table.csv", index=False, encoding="utf-8-sig")

    dose_df = dose_response(spec_df)
    dose_df.to_csv(save_dir / "asa4_dose_response.csv", index=False, encoding="utf-8-sig")

    sig_df = shuffle_significance(spec_df, cfg)
    sig_df.to_csv(save_dir / "asa4_shuffle_specificity_significance.csv", index=False, encoding="utf-8-sig")

    verdict = sig_df[np.isclose(sig_df["alpha"], max(cfg.shuffle_alphas))].copy()
    verdict.to_csv(save_dir / "asa4_verdict.csv", index=False, encoding="utf-8-sig")

    summary = {
        "audit": "ASA-4 Operator-Matched Steering",
        "config": asdict(cfg),
        "n_test_prompts": int(len(test_df)),
        "deltaU_pca_explained_variance": float(pca_decision.explained_variance_ratio_[0]),
        "baseline_pair_choice_rate_test": base_test["pair_choice"].value_counts(normalize=True).to_dict(),
        "key_outputs": {
            "steering_results": "asa4_steering_results.csv",
            "specificity": "asa4_specificity_summary.csv",
            "dose_response": "asa4_dose_response.csv",
            "shuffle_specificity_significance": "asa4_shuffle_specificity_significance.csv",
            "verdict": "asa4_verdict.csv",
            "operator_audit": "asa4_operator_audit.csv",
        },
        "note": (
            "ASA-4 tests mechanism-conditioned local transition operators. "
            "It partially replaces real H_{l+1} with stable-operator-predicted H_{l+1}. "
            "Answer-token logits/ranks/margins are not used to fit operators. R_L and DeltaU are evaluation targets only."
        )
    }
    with open(save_dir / "asa4_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nASA-4 complete.")
    print(verdict)

if __name__ == "__main__":
    main(CFG)
