
# ============================================================
# ASA-7A: Operator + Order-Precursor Combination Audit
#
# Motivation:
#   ASA-4:
#       mechanism operator steering PASS-Strong
#       H_{l+1}' = (1-alpha) H_{l+1} + alpha A_S(H_l)
#
#   ASA-6B/C/D:
#       position_ridge / order-precursor channel PASS-Strong
#       H_l' = H_l + beta * w_DeltaU
#
#   ASA-7A asks:
#       Do these two controls combine additively, redundantly, or synergistically?
#
# Core controls:
#   1. none
#   2. operator_only
#   3. precursor_only
#   4. operator_plus_precursor_same_window
#   5. operator_then_precursor
#   6. precursor_then_operator
#
# Main windows:
#   operator_window  = L15-L19  (ASA-4 effective window)
#   precursor_window = L17-L19  (ASA-6B effective window)
#
# Main direction:
#   Q -> S, closure -> stable / clean
#
# Main metric:
#   DeltaU specificity:
#       target closure abs shift - non-target abs shift
#
# Key synergy score:
#   synergy = shift(combo) - shift(operator_only) - shift(precursor_only)
#
# Compute control:
#   No 50/100 shuffle by default.
#   This is a mechanism-combination audit, not a new significance audit.
#
# Outputs:
#   asa7a_outputs/
#     asa7a_config.json
#     asa7a_dataset.csv
#     asa7a_baseline_features.csv
#     asa7a_operator_audit.csv
#     asa7a_direction_audit.csv
#     asa7a_steering_results.csv
#     asa7a_specificity_summary.csv
#     asa7a_synergy_summary.csv
#     asa7a_verdict.csv
#     asa7a_summary.json
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
    save_dir: str = "./asa7a_outputs"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float16"
    seed: int = 42

    n_graphs: int = 96
    max_len: int = 260
    batch_size: int = 4
    test_size: float = 0.30

    # ASA-4 effective operator window: fit l -> l+1 and hook layer l.
    operator_layers: Tuple[int, ...] = tuple(range(15, 20))

    # ASA-6B effective precursor window: hook layer l.
    precursor_layers: Tuple[int, ...] = (17, 18, 19)

    # Decision layers.
    decision_layers: Tuple[int, ...] = tuple(range(20, 26))

    # Operator learning.
    pca_dim: int = 128
    operator_ridge_alpha: float = 10.0

    # Precursor learning.
    precursor_ridge_alpha: float = 10.0
    min_dir_norm: float = 1e-4

    # Doses. Keep compact.
    op_alphas: Tuple[float, ...] = (0.0, 0.10, 0.20, 0.30)
    pre_betas: Tuple[float, ...] = (0.0, 0.10, 0.20, 0.30)

    # Combination grid: only matched doses by default to control compute.
    matched_combo_only: bool = True

    # Include controls.
    include_operator_only: bool = True
    include_precursor_only: bool = True
    include_same_window_combo: bool = True
    include_ordered_combos: bool = True
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
    pd.DataFrame(rows).to_csv(save_dir / "asa7a_label_tokenization_audit.csv", index=False, encoding="utf-8-sig")
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
    df.to_csv(save_dir / "asa7a_dataset.csv", index=False, encoding="utf-8-sig")
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
    out.to_csv(save_dir / "asa7a_baseline_features.csv", index=False, encoding="utf-8-sig")
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
    split_df.to_csv(save_dir / "asa7a_split.csv", index=False, encoding="utf-8-sig")
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
                "layer": l,
                "mechanism": mech,
                "n_train": int(len(idx)),
                "pca_dim": int(dim),
                "pca_var_sum": float(np.sum(pca.explained_variance_ratio_)),
                "train_r2_znext": float(r2),
                "train_flat_corr_znext": float(corr),
            })

    pd.DataFrame(audits).to_csv(save_dir / "asa7a_operator_audit.csv", index=False, encoding="utf-8-sig")
    return operators

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
            "coef_norm": norm,
            "valid": bool(unit is not None),
            "train_r2": float(r2_score(y0, pred)),
            "train_corr": float(np.corrcoef(y0, pred)[0, 1]) if np.std(pred) > 1e-8 else 0.0,
        })

    pd.DataFrame(audits).to_csv(save_dir / "asa7a_direction_audit.csv", index=False, encoding="utf-8-sig")
    return dirs

# ============================================================
# HOOKS
# ============================================================

def make_combo_hook_factory(
    cfg,
    layer_role_map,
    stable_ops,
    precursor_dirs,
    op_alpha,
    pre_beta,
    control_mode,
    W_device,
    clean_ids_batch=None,
    conflict_ids_batch=None,
):
    """
    control_mode:
      none
      operator_only
      precursor_only
      same_combo
      operator_then_precursor
      precursor_then_operator
      answer
    """

    pre_tensors = {
        int(l): torch.tensor(v, dtype=torch.float32, device=W_device.device)
        for l, v in precursor_dirs.items() if v is not None
    }

    def hook_factory(l):
        def hook(module, inputs, output):
            if control_mode == "none" or (op_alpha == 0.0 and pre_beta == 0.0):
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
            h_real_last = h_out[idx, pos, :]
            h_in_last = h_in[idx, pos, :]
            h_new = h_real_last

            def apply_operator(h_base):
                if l not in stable_ops or op_alpha == 0.0:
                    return h_base
                h_target = stable_ops[l]["stable"].predict_torch(h_in_last.float()).to(h_base.dtype)
                return (1.0 - op_alpha) * h_base + op_alpha * h_target

            def apply_precursor(h_base):
                if l not in pre_tensors or pre_beta == 0.0:
                    return h_base
                d = pre_tensors[l].to(h_base.device, h_base.dtype)
                scale = h_base.float().std(dim=-1, keepdim=True).to(h_base.dtype)
                return h_base + pre_beta * scale * d[None, :].expand(B, -1)

            def apply_answer(h_base):
                if pre_beta == 0.0:
                    return h_base
                ans = W_device[clean_ids_batch].float() - W_device[conflict_ids_batch].float()
                ans = ans / (torch.norm(ans, dim=1, keepdim=True) + 1e-8)
                scale = h_base.float().std(dim=-1, keepdim=True).to(h_base.dtype)
                return h_base + pre_beta * scale * ans.to(h_base.dtype)

            if control_mode == "operator_only":
                h_new = apply_operator(h_new)
            elif control_mode == "precursor_only":
                h_new = apply_precursor(h_new)
            elif control_mode == "same_combo":
                # operator and precursor at each eligible layer; operator first within layer.
                h_new = apply_operator(h_new)
                h_new = apply_precursor(h_new)
            elif control_mode == "operator_then_precursor":
                # Early operator layers L15-L16, late precursor L17-L19.
                if l in (15, 16):
                    h_new = apply_operator(h_new)
                if l in (17, 18, 19):
                    h_new = apply_precursor(h_new)
            elif control_mode == "precursor_then_operator":
                # Precursor on L17, then operator on L18-L19.
                if l == 17:
                    h_new = apply_precursor(h_new)
                if l in (18, 19):
                    h_new = apply_operator(h_new)
            elif control_mode == "answer":
                h_new = apply_answer(h_new)

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
                 clean_anchor_R, stable_ops, precursor_dirs,
                 op_alpha=0.0, pre_beta=0.0, control_mode="none"):
    W = get_lm_head_weight(model).detach().float()
    W_device = W.to(model.device)
    rows = []

    texts = df_subset["text"].tolist()
    clean_ids_all = df_subset["clean_token_id"].values.astype(np.int64)
    conflict_ids_all = df_subset["conflict_token_id"].values.astype(np.int64)
    layer_modules = get_layers(model)

    # Hook union.
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

            hook_factory = make_combo_hook_factory(
                cfg=cfg,
                layer_role_map=None,
                stable_ops=stable_ops,
                precursor_dirs=precursor_dirs,
                op_alpha=float(op_alpha),
                pre_beta=float(pre_beta),
                control_mode=control_mode,
                W_device=W_device,
                clean_ids_batch=cids,
                conflict_ids_batch=eids,
            )

            handles = []
            if control_mode != "none" and (op_alpha != 0.0 or pre_beta != 0.0):
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
    for (control, op_alpha, pre_beta), g in merged.groupby(["control", "op_alpha", "pre_beta"]):
        target = g[g["mechanism"] == "closure"]
        nontarget = g[g["mechanism"] != "closure"]
        rows.append({
            "control": control,
            "op_alpha": float(op_alpha),
            "pre_beta": float(pre_beta),
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

def compute_synergy(spec_df):
    rows = []
    # Use matched beta=op_alpha comparisons.
    for a in sorted(set(spec_df["op_alpha"]).union(set(spec_df["pre_beta"]))):
        if a == 0:
            continue
        op = spec_df[(spec_df["control"] == "operator_only") & np.isclose(spec_df["op_alpha"], a)]
        pr = spec_df[(spec_df["control"] == "precursor_only") & np.isclose(spec_df["pre_beta"], a)]
        if len(op) == 0 or len(pr) == 0:
            continue
        op_spec = float(op.iloc[0]["specificity"])
        pr_spec = float(pr.iloc[0]["specificity"])

        for ctrl in ["same_combo", "operator_then_precursor", "precursor_then_operator"]:
            combo = spec_df[
                (spec_df["control"] == ctrl) &
                (np.isclose(spec_df["op_alpha"], a)) &
                (np.isclose(spec_df["pre_beta"], a))
            ]
            if len(combo) == 0:
                continue
            combo_spec = float(combo.iloc[0]["specificity"])
            rows.append({
                "dose": float(a),
                "combo_control": ctrl,
                "operator_specificity": op_spec,
                "precursor_specificity": pr_spec,
                "combo_specificity": combo_spec,
                "combo_minus_operator": combo_spec - op_spec,
                "combo_minus_precursor": combo_spec - pr_spec,
                "additive_synergy": combo_spec - op_spec - pr_spec,
                "max_gain_over_single": combo_spec - max(op_spec, pr_spec),
                "relative_to_best_single": combo_spec / (max(abs(op_spec), abs(pr_spec), 1e-8)),
            })

    return pd.DataFrame(rows)

# ============================================================
# MAIN
# ============================================================

def main(cfg):
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "asa7a_config.json", "w", encoding="utf-8") as f:
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
    precursor_dirs = fit_position_ridge_directions(base_df, h_by_layer, train_idx, cfg, save_dir)

    test_df = base_df.iloc[test_idx].reset_index(drop=True)
    base_test = base_df.iloc[test_idx][["prompt_id", "DeltaU", "pair_choice"]].reset_index(drop=True)

    runs = []
    # Baseline.
    runs.append(("none", 0.0, 0.0))

    if cfg.include_operator_only:
        for a in cfg.op_alphas:
            if a == 0:
                continue
            runs.append(("operator_only", float(a), 0.0))

    if cfg.include_precursor_only:
        for b in cfg.pre_betas:
            if b == 0:
                continue
            runs.append(("precursor_only", 0.0, float(b)))

    if cfg.include_same_window_combo or cfg.include_ordered_combos:
        if cfg.matched_combo_only:
            doses = sorted(set(cfg.op_alphas).intersection(set(cfg.pre_betas)))
            doses = [d for d in doses if d != 0]
            combo_pairs = [(float(d), float(d)) for d in doses]
        else:
            combo_pairs = [(float(a), float(b)) for a in cfg.op_alphas for b in cfg.pre_betas if a != 0 and b != 0]

        for a, b in combo_pairs:
            if cfg.include_same_window_combo:
                runs.append(("same_combo", a, b))
            if cfg.include_ordered_combos:
                runs.append(("operator_then_precursor", a, b))
                runs.append(("precursor_then_operator", a, b))

    if cfg.include_answer_control:
        for b in cfg.pre_betas:
            if b == 0:
                continue
            runs.append(("answer", 0.0, float(b)))

    # Deduplicate preserving order.
    seen = set()
    dedup = []
    for r in runs:
        if r not in seen:
            dedup.append(r)
            seen.add(r)

    results = []
    for control, a, b in dedup:
        print(f"[RUN] control={control} op_alpha={a} pre_beta={b}")
        out = forward_eval(
            model=model,
            tokenizer=tokenizer,
            df_subset=test_df,
            cfg=cfg,
            pca_decision=pca_decision,
            clean_anchor_R=clean_anchor_R,
            stable_ops=stable_ops,
            precursor_dirs=precursor_dirs,
            op_alpha=a,
            pre_beta=b,
            control_mode=control,
        )
        out["control"] = control
        out["op_alpha"] = float(a)
        out["pre_beta"] = float(b)
        results.append(out)

    results_df = pd.concat(results, ignore_index=True)
    results_df.to_csv(save_dir / "asa7a_steering_results.csv", index=False, encoding="utf-8-sig")

    spec_df, merged = summarize_specificity(results_df, base_test)
    spec_df.to_csv(save_dir / "asa7a_specificity_summary.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(save_dir / "asa7a_merged_shift_table.csv", index=False, encoding="utf-8-sig")

    synergy_df = compute_synergy(spec_df)
    synergy_df.to_csv(save_dir / "asa7a_synergy_summary.csv", index=False, encoding="utf-8-sig")

    # Verdict.
    best = spec_df.sort_values("specificity", ascending=False).head(1).iloc[0].to_dict()
    best_syn = synergy_df.sort_values("max_gain_over_single", ascending=False).head(1)
    if len(best_syn):
        best_syn_row = best_syn.iloc[0].to_dict()
    else:
        best_syn_row = {}

    # Simple verdict heuristic.
    if len(best_syn) and best_syn_row.get("max_gain_over_single", -999) > 0.10 and best_syn_row.get("additive_synergy", -999) > 0.0:
        verdict = "SYNERGISTIC_COMBINATION"
    elif len(best_syn) and best_syn_row.get("max_gain_over_single", -999) > 0.05:
        verdict = "COMBO_IMPROVES_BEST_SINGLE"
    elif str(best.get("control")) in ("operator_only", "precursor_only"):
        verdict = "NO_COMBO_GAIN_BEST_SINGLE_DOMINATES"
    else:
        verdict = "MIXED_OR_UNRESOLVED"

    verdict_df = pd.DataFrame([{
        "audit": "ASA-7A Operator + Order-Precursor Combination Audit",
        "verdict": verdict,
        "best_control": best.get("control"),
        "best_op_alpha": best.get("op_alpha"),
        "best_pre_beta": best.get("pre_beta"),
        "best_specificity": best.get("specificity"),
        **{f"best_synergy_{k}": v for k, v in best_syn_row.items()},
    }])
    verdict_df.to_csv(save_dir / "asa7a_verdict.csv", index=False, encoding="utf-8-sig")

    summary = {
        "audit": "ASA-7A Operator + Order-Precursor Combination Audit",
        "config": asdict(cfg),
        "n_test_prompts": int(len(test_df)),
        "deltaU_pca_explained_variance": float(pca_decision.explained_variance_ratio_[0]),
        "num_runs": int(len(dedup)),
        "verdict": verdict,
        "best_specificity_row": best,
        "best_synergy_row": best_syn_row,
        "key_outputs": {
            "steering_results": "asa7a_steering_results.csv",
            "specificity": "asa7a_specificity_summary.csv",
            "synergy": "asa7a_synergy_summary.csv",
            "verdict": "asa7a_verdict.csv",
            "operator_audit": "asa7a_operator_audit.csv",
            "direction_audit": "asa7a_direction_audit.csv",
        },
        "note": (
            "ASA-7A tests whether ASA-4 operator control and ASA-6 order-precursor modulation combine additively, redundantly, or synergistically. "
            "This script does not run matched shuffles; ASA-4 and ASA-6B already established individual significance. "
            "If a combination improves over the best single control, rerun the best combo with shuffle controls."
        )
    }

    with open(save_dir / "asa7a_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nASA-7A complete.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main(CFG)
