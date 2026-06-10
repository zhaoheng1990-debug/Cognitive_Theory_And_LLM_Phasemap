# ============================================================
# ASA-2: Answerless Predictor-Derived Steering
# ------------------------------------------------------------
# Goal:
#   Test whether answerless TopK/VIM mechanism prototypes are not only
#   predictive signatures, but causal control directions over later
#   geodesic-dominance / answer-basin dynamics.
#
# Core intervention:
#   Build mechanism prototypes from TopK centers only:
#       P_S(l), P_K(l), P_Q(l)
#   Construct directions:
#       Q -> S : P_S - P_Q
#       K -> S : P_S - P_K
#       S -> Q : P_Q - P_S  [exploratory]
#   Inject small perturbations into residual hidden states in an early/mid
#   layer window, then measure shifts in later DeltaU and pairwise generation.
#
# Forbidden for prototype construction / steering directions:
#   logit(C), logit(E), rank(C), rank(E), R_l = logit(C)-logit(E)
#
# Allowed for prototype construction:
#   TopK token identity via center in W embedding space, spread, entropy.
#   This first ASA-1 implementation uses TopK center prototypes because they
#   live directly in hidden/lm-head dimension and can be injected as directions.
#
# Outputs:
#   asa1_outputs/
#       asa1_dataset.csv
#       asa1_split.csv
#       asa1_baseline_features.csv
#       asa1_prototype_audit.csv
#       asa1_direction_audit.csv
#       asa1_steering_results.csv
#       asa1_dose_response.csv
#       asa1_specificity_summary.csv
#       asa1_summary.json
# ============================================================

import os
import re
import gc
import json
import math
import random
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

QWEN_PATH = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"
LLAMA_PATH = r"D:\model\Llama-3.2-1B-Instruct"
GEMMA_PATH = r"D:\model\gemma-2-2b-it"

MODEL_SPECS = {
    "qwen": QWEN_PATH,
    "llama": LLAMA_PATH,
    "gemma": GEMMA_PATH,
}


def resolve_model_path(model_key: str, model_path: str) -> str:
    raw = str(model_path or "auto")
    raw_lower = raw.lower()
    markers = ["auto", "your_snapshot", "your-snapshot", "snapshot目录", "<", ">"]
    if raw.strip() and not any(m in raw_lower for m in markers):
        return raw
    key = str(model_key).lower().strip()
    if key not in MODEL_SPECS:
        raise ValueError(f"Unknown model_key={model_key!r}; choose {list(MODEL_SPECS)} or set model_path.")
    return MODEL_SPECS[key]


@dataclass
class Config:
    model_key: str = "qwen"
    model_path: str = "auto"
    save_dir: str = "./asa2_outputs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float16"  # float16 or float32
    seed: int = 42

    n_graphs: int = 96
    max_len: int = 260
    batch_size: int = 4

    topk: int = 512

    # Prototype layers are where TopK centers are extracted.
    proto_layers: Tuple[int, ...] = tuple(range(0, 20))

    # Decision layers used only to construct DeltaU target/evaluation.
    decision_layers: Tuple[int, ...] = tuple(range(20, 26))

    # ASA-1C focuses on shuffle-distribution significance of closure-specific steering in the Q->S window.
    steering_windows: Dict[str, Tuple[int, ...]] = None

    # Dose response alpha grid.
    alphas: Tuple[float, ...] = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10)

    # Split.
    test_size: float = 0.30

    # ASA-1C focuses on closure -> stable repair and shuffle-specificity statistics.
    run_s_to_q: bool = False

    # Residual prototype settings.
    shared_pca_dim: int = 8
    n_shuffles: int = 50
    shuffle_alphas: Tuple[float, ...] = (0.02, 0.05, 0.10)


CFG = Config()
if CFG.steering_windows is None:
    CFG.steering_windows = {
        "L15_19": tuple(range(15, 20)),
    }

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
    ("Ava", "Bela", "Cora"), ("Darin", "Elo", "Faye"), ("Galen", "Hera", "Ivo"),
    ("Juno", "Kira", "Lio"), ("Mira", "Nero", "Orin"), ("Pia", "Quin", "Rhea"),
    ("Sola", "Taro", "Una"), ("Vera", "Wen", "Xio"), ("Yara", "Zeno", "Nia"),
    ("Orla", "Pavel", "Rin"), ("Nora", "Silas", "Tess"), ("Uma", "Vito", "Willa"),
]

REL_WORDS = [
    ("belongs to", "is located at"),
    ("is assigned to", "maps to"),
    ("is part of", "points to"),
    ("is grouped under", "has label"),
]

STABLE_CONDS = ["clean", "redundant", "paraphrase", "irrelevant"]
COMPETITION_CONDS = ["weak_distractor", "competition_balanced", "direct_conflict"]
CLOSURE_CONDS = ["closure_update", "closure_override", "exception_override"]
ALL_CONDS = STABLE_CONDS + COMPETITION_CONDS + CLOSURE_CONDS

MECHANISM = {**{c: "stable" for c in STABLE_CONDS},
             **{c: "competition" for c in COMPETITION_CONDS},
             **{c: "closure" for c in CLOSURE_CONDS}}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def continuation_ids(tokenizer, label: str):
    return tokenizer(" " + label, add_special_tokens=False)["input_ids"]


def select_single_token_labels(tokenizer, save_dir: Path) -> List[str]:
    rows = []
    chosen = []
    for lab in LABEL_CANDIDATES:
        ids = continuation_ids(tokenizer, lab)
        ok = len(ids) == 1
        rows.append({"label": lab, "ids": str(ids), "len": len(ids), "single": int(ok)})
        if ok:
            chosen.append(lab)
    pd.DataFrame(rows).to_csv(save_dir / "asa1_label_tokenization_audit.csv", index=False, encoding="utf-8-sig")
    if len(chosen) < 12:
        raise RuntimeError(f"Need at least 12 single-token labels; found {len(chosen)}: {chosen}")
    return chosen[:12]


def make_prompt(cond, a, b, d, clean_label, conflict_label, aux_label, r1, r2):
    if cond == "clean":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} {r1} {b}.",
            f"Fact 2: {b} {r2} {clean_label}.",
            f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
            "Answer with exactly one word:",
        ]
    elif cond == "redundant":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} {r1} {b}.",
            f"Fact 2: {b} {r2} {clean_label}.",
            f"Repeated confirmation: {a} still goes through {b}.",
            f"Repeated confirmation: {b} still points to {clean_label}.",
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
    elif cond == "irrelevant":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} {r1} {b}.",
            f"Fact 2: {b} {r2} {clean_label}.",
            f"Irrelevant fact: {d} is associated with {aux_label}.",
            f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
            "Answer with exactly one word:",
        ]
    elif cond == "weak_distractor":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} {r1} {b}.",
            f"Fact 2: {b} {r2} {clean_label}.",
            f"Weak note: some unrelated source mentions {conflict_label}, but does not update the graph.",
            f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
            "Answer with exactly one word:",
        ]
    elif cond == "competition_balanced":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} {r1} {b}.",
            f"Fact 2: {b} {r2} {clean_label}.",
            f"Competing fact: {a} is also associated with {conflict_label}.",
            f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
            "Answer with exactly one word:",
        ]
    elif cond == "direct_conflict":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} {r1} {b}.",
            f"Fact 2: {b} {r2} {clean_label}.",
            f"Direct conflicting fact: {a} {r2} {conflict_label}.",
            f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
            "Answer with exactly one word:",
        ]
    elif cond == "closure_update":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} {r1} {b}.",
            f"Old record: {b} {r2} {clean_label}.",
            f"Updated record: {b} {r2} {conflict_label}, not {clean_label}.",
            f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
            "Answer with exactly one word:",
        ]
    elif cond == "closure_override":
        lines = [
            "You are given a rule system.",
            f"General rule: items that {r1} {b} receive label {clean_label}.",
            f"Override rule: in this case, items that {r1} {b} receive label {conflict_label}.",
            f"Fact: {a} {r1} {b}.",
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


def build_dataset(tokenizer, cfg: Config, save_dir: Path) -> pd.DataFrame:
    labels = select_single_token_labels(tokenizer, save_dir)
    rows = []
    graph_id = 0
    n_labels = len(labels)
    for ent_i, (a, b, d) in enumerate(ENTITIES):
        for rel_i, (r1, r2) in enumerate(REL_WORDS):
            if graph_id >= cfg.n_graphs:
                break
            clean_label = labels[(2 * graph_id) % n_labels]
            conflict_label = labels[(2 * graph_id + 1) % n_labels]
            aux_label = labels[(2 * graph_id + 2) % n_labels]
            clean_id = continuation_ids(tokenizer, clean_label)[0]
            conflict_id = continuation_ids(tokenizer, conflict_label)[0]
            for cond in ALL_CONDS:
                text = make_prompt(cond, a, b, d, clean_label, conflict_label, aux_label, r1, r2)
                rows.append({
                    "prompt_id": f"g{graph_id:03d}_{cond}",
                    "graph_id": graph_id,
                    "entity_id": ent_i,
                    "relation_id": rel_i,
                    "condition": cond,
                    "mechanism": MECHANISM[cond],
                    "clean_label": clean_label,
                    "conflict_label": conflict_label,
                    "clean_token_id": clean_id,
                    "conflict_token_id": conflict_id,
                    "text": text,
                })
            graph_id += 1
        if graph_id >= cfg.n_graphs:
            break
    df = pd.DataFrame(rows)
    df.to_csv(save_dir / "asa1_dataset.csv", index=False, encoding="utf-8-sig")
    return df

# ============================================================
# MODEL UTILS
# ============================================================


def load_tokenizer_and_model(cfg: Config):
    cfg.model_path = resolve_model_path(cfg.model_key, cfg.model_path)
    print(f"[MODEL] model_key={cfg.model_key} resolved_path={cfg.model_path}")
    if "YOUR_SNAPSHOT" in str(cfg.model_path):
        raise RuntimeError("Model path still contains YOUR_SNAPSHOT after resolution.")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.float16 if cfg.dtype == "float16" and cfg.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if cfg.device == "cuda" else None,
    )
    if cfg.device == "cpu":
        model.to("cpu")
    model.eval()
    return tokenizer, model


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise RuntimeError("Cannot locate decoder layers for hook registration.")


def get_lm_head_weight(model):
    if hasattr(model, "lm_head"):
        return model.lm_head.weight.detach()
    raise RuntimeError("Cannot locate lm_head.weight")


def last_positions(attention_mask):
    return torch.full((attention_mask.shape[0],), attention_mask.shape[1] - 1,
                      dtype=torch.long, device=attention_mask.device)


def normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)

# ============================================================
# BASELINE EXTRACTION
# ============================================================


def extract_baseline_topk_and_decision(model, tokenizer, df: pd.DataFrame, cfg: Config, save_dir: Path):
    layers_obj = get_layers(model)
    num_layers = len(layers_obj)
    print(f"[INFO] num_layers={num_layers}")
    max_layer = max(max(cfg.proto_layers), max(cfg.decision_layers))
    if max_layer >= num_layers:
        raise RuntimeError(f"Requested layer {max_layer}, but model has {num_layers} layers. Adjust cfg layers.")

    W = get_lm_head_weight(model).detach().float()
    W_dev = W.to(model.device)
    W_norm = torch.nn.functional.normalize(W_dev.float(), dim=1)
    dim = W.shape[1]

    n = len(df)
    centers = {l: np.zeros((n, dim), dtype=np.float32) for l in cfg.proto_layers}
    spreads = {l: np.zeros(n, dtype=np.float32) for l in cfg.proto_layers}
    entropies = {l: np.zeros(n, dtype=np.float32) for l in cfg.proto_layers}
    R_by_layer = {l: np.zeros(n, dtype=np.float32) for l in cfg.decision_layers}
    final_R = np.zeros(n, dtype=np.float32)

    texts = df["text"].tolist()
    cids_all = df["clean_token_id"].values.astype(np.int64)
    eids_all = df["conflict_token_id"].values.astype(np.int64)
    needed = sorted(set(cfg.proto_layers) | set(cfg.decision_layers))

    with torch.no_grad():
        for start in range(0, n, cfg.batch_size):
            end = min(n, start + cfg.batch_size)
            inputs = tokenizer(texts[start:end], return_tensors="pt", padding=True,
                               truncation=True, max_length=cfg.max_len).to(model.device)
            out = model(**inputs, output_hidden_states=True, use_cache=False)
            hstates = out.hidden_states  # embeddings + layers
            pos = last_positions(inputs["attention_mask"])
            bsz = end - start

            cids = torch.tensor(cids_all[start:end], dtype=torch.long, device=model.device)
            eids = torch.tensor(eids_all[start:end], dtype=torch.long, device=model.device)

            for l in needed:
                h = hstates[l + 1][torch.arange(bsz, device=model.device), pos, :].detach().float()

                if l in R_by_layer:
                    cl = torch.sum(h * W_dev[cids].float(), dim=1)
                    el = torch.sum(h * W_dev[eids].float(), dim=1)
                    R_by_layer[l][start:end] = (cl - el).detach().cpu().numpy().astype(np.float32)

                if l in cfg.proto_layers:
                    logits = h @ W_dev.float().T
                    vals, ids = torch.topk(logits, k=cfg.topk, dim=1)
                    emb = W_norm[ids].float()
                    center = emb.mean(dim=1)
                    center = torch.nn.functional.normalize(center, dim=1)
                    cos_to_center = torch.sum(emb * center[:, None, :], dim=2)
                    spread = (1.0 - cos_to_center).mean(dim=1)
                    p = torch.softmax(vals.float(), dim=1)
                    entropy = -(p * torch.log(p + 1e-8)).sum(dim=1) / math.log(cfg.topk)
                    centers[l][start:end] = center.detach().cpu().numpy().astype(np.float32)
                    spreads[l][start:end] = spread.detach().cpu().numpy().astype(np.float32)
                    entropies[l][start:end] = entropy.detach().cpu().numpy().astype(np.float32)

            # Final pairwise choice from actual final logits at prompt position.
            h_final = hstates[-1][torch.arange(bsz, device=model.device), pos, :].detach().float()
            cl = torch.sum(h_final * W_dev[cids].float(), dim=1)
            el = torch.sum(h_final * W_dev[eids].float(), dim=1)
            final_R[start:end] = (cl - el).detach().cpu().numpy().astype(np.float32)

            print(f"  extracted baseline {end}/{n}")
            del out, hstates, inputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    feat_df = df.copy()
    for l in cfg.decision_layers:
        feat_df[f"R_L{l}"] = R_by_layer[l]
    feat_df["final_R"] = final_R
    feat_df["pair_choice"] = np.where(final_R >= 0, "clean", "conflict")

    # Clean-relative decision dR and DeltaU.
    clean_df = feat_df[feat_df["condition"] == "clean"].set_index("graph_id")
    d_cols = []
    for l in cfg.decision_layers:
        vals = []
        for _, row in feat_df.iterrows():
            vals.append(row[f"R_L{l}"] - clean_df.loc[row["graph_id"], f"R_L{l}"])
        col = f"dR_L{l}"
        feat_df[col] = np.asarray(vals, dtype=np.float32)
        d_cols.append(col)

    X_dec = feat_df[d_cols].values.astype(np.float32)
    pca = PCA(n_components=1)
    delta_u = pca.fit_transform(X_dec)[:, 0].astype(np.float32)
    # Orient stable > closure.
    if np.mean(delta_u[feat_df["mechanism"] == "stable"]) < np.mean(delta_u[feat_df["mechanism"] == "closure"]):
        delta_u = -delta_u
    feat_df["DeltaU"] = delta_u

    feat_df.to_csv(save_dir / "asa1_baseline_features.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        save_dir / "asa1_topk_centers_by_layer.npz",
        **{f"center_L{l}": centers[l] for l in cfg.proto_layers},
        **{f"spread_L{l}": spreads[l] for l in cfg.proto_layers},
        **{f"entropy_L{l}": entropies[l] for l in cfg.proto_layers},
    )
    return feat_df, centers, pca, d_cols, W.detach().cpu().numpy().astype(np.float32)


# ============================================================
# SPLIT + ANSWERLESS PREDICTOR DIRECTIONS
# ============================================================

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def make_graph_split(df: pd.DataFrame, cfg: Config, save_dir: Path) -> pd.DataFrame:
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.seed)
    idx = np.arange(len(df))
    train_idx, test_idx = next(gss.split(idx, groups=df["graph_id"].values))
    split = pd.DataFrame({"row_index": idx, "split": "train"})
    split.loc[test_idx, "split"] = "test"
    split["prompt_id"] = df["prompt_id"].values
    split["graph_id"] = df["graph_id"].values
    split["condition"] = df["condition"].values
    split["mechanism"] = df["mechanism"].values
    split.to_csv(save_dir / "asa2_split.csv", index=False, encoding="utf-8-sig")
    return split


def _fit_ridge_direction(X: np.ndarray, y: np.ndarray, alpha: float = 10.0) -> np.ndarray:
    """Fit standardized Ridge and return the coefficient direction in raw X coordinates."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)
    reg = Ridge(alpha=alpha)
    reg.fit(Xs, y)
    coef_scaled = reg.coef_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)
    coef_raw = coef_scaled / (scale + 1e-8)
    coef_raw = coef_raw / (np.linalg.norm(coef_raw) + 1e-8)
    return coef_raw.astype(np.float32)


def _remove_subspace(vec: np.ndarray, basis: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).copy()
    if basis is not None and len(basis) > 0:
        B = np.asarray(basis, dtype=np.float32)
        v = v - B.T @ (B @ v)
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)


def build_predictor_directions(base_df: pd.DataFrame, split: pd.DataFrame, centers: Dict[int, np.ndarray], cfg: Config, save_dir: Path):
    """
    ASA-2 direction builder.

    Difference from ASA-1:
      ASA-1 used mean mechanism prototype differences P_S-P_Q.
      ASA-2 trains an answerless predictor C_l -> DeltaU and uses the learned
      coefficient vector as a local DeltaU-sensitive steering direction.

    Forbidden for direction construction:
      answer token logits/ranks/margins.
    Used for direction construction:
      TopK center C_l only, plus DeltaU target from held-out decision PCA.
      DeltaU is an evaluation target, not an input answer feature.
    """
    train_mask = split["split"].values == "train"
    y = base_df["DeltaU"].values.astype(np.float32)
    rng = np.random.default_rng(cfg.seed + 3307)

    pred_dirs = {"increase_DeltaU": {}, "decrease_DeltaU": {}}
    pred_residual_dirs = {"increase_DeltaU": {}, "decrease_DeltaU": {}}
    shared_basis = {}
    shuffle_dirs = []
    rows = []

    for l in cfg.proto_layers:
        C = centers[l].astype(np.float32)
        X_train = C[train_mask]
        y_train = y[train_mask]

        # Shared high-variance subspace for optional residualization.
        n_comp = min(int(cfg.shared_pca_dim), X_train.shape[0] - 1, X_train.shape[1])
        if n_comp > 0:
            pca_shared = PCA(n_components=n_comp)
            pca_shared.fit(X_train)
            B = pca_shared.components_.astype(np.float32)
        else:
            B = np.zeros((0, X_train.shape[1]), dtype=np.float32)
        shared_basis[l] = B

        coef = _fit_ridge_direction(X_train, y_train, alpha=10.0)
        # By construction, +coef should increase DeltaU. DeltaU was oriented stable > closure.
        pred_dirs["increase_DeltaU"][l] = coef
        pred_dirs["decrease_DeltaU"][l] = (-coef / (np.linalg.norm(coef) + 1e-8)).astype(np.float32)
        pred_residual_dirs["increase_DeltaU"][l] = _remove_subspace(coef, B)
        pred_residual_dirs["decrease_DeltaU"][l] = _remove_subspace(-coef, B)

        # Audit layer-wise predictor quality on train for interpretability.
        pred_train = X_train @ coef
        corr = float(np.corrcoef(pred_train, y_train)[0, 1]) if np.std(pred_train) > 1e-8 else 0.0
        rows.append({
            "layer": int(l),
            "direction_norm": float(np.linalg.norm(coef)),
            "train_corr_centercoef_deltaU": corr,
            "shared_pca_dim": int(n_comp),
        })

    # Shuffle target DeltaU to create null predictor-derived directions.
    train_indices = np.where(train_mask)[0]
    y_train_true = y[train_indices].copy()
    for si in range(int(cfg.n_shuffles)):
        y_perm = rng.permutation(y_train_true)
        sd = {"increase_DeltaU": {}, "decrease_DeltaU": {}}
        sdr = {"increase_DeltaU": {}, "decrease_DeltaU": {}}
        for l in cfg.proto_layers:
            C = centers[l].astype(np.float32)
            X_train = C[train_mask]
            coef = _fit_ridge_direction(X_train, y_perm, alpha=10.0)
            sd["increase_DeltaU"][l] = coef
            sd["decrease_DeltaU"][l] = (-coef / (np.linalg.norm(coef) + 1e-8)).astype(np.float32)
            sdr["increase_DeltaU"][l] = _remove_subspace(coef, shared_basis[l])
            sdr["decrease_DeltaU"][l] = _remove_subspace(-coef, shared_basis[l])
        shuffle_dirs.append({"raw": sd, "residual": sdr})

    audit = pd.DataFrame(rows)
    audit.to_csv(save_dir / "asa2_predictor_direction_audit.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        save_dir / "asa2_predictor_dirs_and_shared_basis.npz",
        **{f"pred_raw_inc_L{l}": pred_dirs["increase_DeltaU"][l] for l in cfg.proto_layers},
        **{f"pred_res_inc_L{l}": pred_residual_dirs["increase_DeltaU"][l] for l in cfg.proto_layers},
        **{f"shared_basis_L{l}": shared_basis[l] for l in cfg.proto_layers},
    )
    return {
        "predictor_raw": pred_dirs,
        "predictor_residual": pred_residual_dirs,
        "shared_basis": shared_basis,
        "shuffle_dirs": shuffle_dirs,
    }


def build_direction(dirs, direction_name: str, layer: int, variant: str = "raw", shuffle_idx: Optional[int] = None) -> np.ndarray:
    # Q_to_S/K_to_S target stable => increase DeltaU. S_to_Q target closure/conflict => decrease DeltaU.
    sign_name = "increase_DeltaU" if direction_name in ["Q_to_S", "K_to_S"] else "decrease_DeltaU"
    if shuffle_idx is not None:
        d = dirs["shuffle_dirs"][int(shuffle_idx)][variant][sign_name][layer]
    else:
        key = "predictor_residual" if variant == "residual" else "predictor_raw"
        d = dirs[key][sign_name][layer]
    return (d / (np.linalg.norm(d) + 1e-8)).astype(np.float32)


def audit_directions(dirs, cfg: Config, save_dir: Path):
    rows = []
    for l in cfg.proto_layers:
        raw = build_direction(dirs, "Q_to_S", l, variant="raw")
        res = build_direction(dirs, "Q_to_S", l, variant="residual")
        rows.append({
            "layer": int(l),
            "raw_residual_cos": float(np.dot(raw, res)),
            "raw_norm": float(np.linalg.norm(raw)),
            "residual_norm": float(np.linalg.norm(res)),
        })
    pd.DataFrame(rows).to_csv(save_dir / "asa2_direction_audit.csv", index=False, encoding="utf-8-sig")


def project_orthogonal(vecs: torch.Tensor, ans_dirs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    coeff = torch.sum(vecs * ans_dirs, dim=1, keepdim=True) / (torch.sum(ans_dirs * ans_dirs, dim=1, keepdim=True) + eps)
    out = vecs - coeff * ans_dirs
    return out / (torch.norm(out, dim=1, keepdim=True) + eps)


class SteeringHookManager:
    def __init__(self, model, target_layers: List[int]):
        self.model = model
        self.target_layers = list(target_layers)
        self.handles = []
        self.current_dirs: Dict[int, torch.Tensor] = {}
        self.alpha = 0.0

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            if self.alpha == 0.0 or layer_idx not in self.current_dirs:
                return output
            if isinstance(output, tuple):
                h = output[0]
                rest = output[1:]
            else:
                h = output
                rest = None
            dirs = self.current_dirs[layer_idx].to(h.device).to(h.dtype)
            # left padding: last token is final sequence position
            pos = h.shape[1] - 1
            scale = torch.std(h[:, pos, :].float(), dim=1, keepdim=True).to(h.dtype) + 1e-6
            h2 = h.clone()
            h2[:, pos, :] = h2[:, pos, :] + float(self.alpha) * scale * dirs
            if rest is not None:
                return (h2,) + rest
            return h2
        return hook

    def register(self):
        layers = get_layers(self.model)
        for l in self.target_layers:
            self.handles.append(layers[l].register_forward_hook(self._make_hook(l)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def make_batch_dirs(direction_name: str, control: str, batch_rows: pd.DataFrame, layers: List[int], dirs,
                    W_np: np.ndarray, cfg: Config, device) -> Dict[int, torch.Tensor]:
    out = {}
    bsz = len(batch_rows)
    dim = W_np.shape[1]
    rng = np.random.default_rng(cfg.seed + 991)

    cids = batch_rows["clean_token_id"].values.astype(np.int64)
    eids = batch_rows["conflict_token_id"].values.astype(np.int64)
    W = torch.tensor(W_np, dtype=torch.float32, device=device)
    clean_w = W[torch.tensor(cids, device=device)]
    conflict_w = W[torch.tensor(eids, device=device)]

    if direction_name in ["Q_to_S", "K_to_S"]:
        ans = clean_w - conflict_w
    else:
        ans = conflict_w - clean_w
    ans = ans / (torch.norm(ans, dim=1, keepdim=True) + 1e-8)

    shuffle_idx = None
    if control.startswith("shuffle_predictor_residual_"):
        shuffle_idx = int(control.split("_")[-1])
    if control.startswith("shuffle_predictor_raw_"):
        shuffle_idx = int(control.split("_")[-1])

    for l in layers:
        if control == "predictor_raw":
            d = build_direction(dirs, direction_name, l, variant="raw")
            mat = np.repeat(d[None, :], bsz, axis=0)
            td = torch.tensor(mat, dtype=torch.float32, device=device)
            td = td / (torch.norm(td, dim=1, keepdim=True) + 1e-8)
        elif control == "predictor_residual":
            d = build_direction(dirs, direction_name, l, variant="residual")
            mat = np.repeat(d[None, :], bsz, axis=0)
            td = torch.tensor(mat, dtype=torch.float32, device=device)
            td = td / (torch.norm(td, dim=1, keepdim=True) + 1e-8)
        elif control == "orthogonal_predictor_raw":
            d = build_direction(dirs, direction_name, l, variant="raw")
            mat = np.repeat(d[None, :], bsz, axis=0)
            td = torch.tensor(mat, dtype=torch.float32, device=device)
            td = td / (torch.norm(td, dim=1, keepdim=True) + 1e-8)
            td = project_orthogonal(td, ans)
        elif control == "orthogonal_predictor_residual":
            d = build_direction(dirs, direction_name, l, variant="residual")
            mat = np.repeat(d[None, :], bsz, axis=0)
            td = torch.tensor(mat, dtype=torch.float32, device=device)
            td = td / (torch.norm(td, dim=1, keepdim=True) + 1e-8)
            td = project_orthogonal(td, ans)
        elif control == "random":
            mat = rng.standard_normal((bsz, dim)).astype(np.float32)
            mat = normalize_rows(mat)
            td = torch.tensor(mat, dtype=torch.float32, device=device)
        elif control == "answer":
            td = ans
        elif shuffle_idx is not None:
            variant = "residual" if "residual" in control else "raw"
            d = build_direction(dirs, direction_name, l, variant=variant, shuffle_idx=shuffle_idx)
            mat = np.repeat(d[None, :], bsz, axis=0)
            td = torch.tensor(mat, dtype=torch.float32, device=device)
            td = td / (torch.norm(td, dim=1, keepdim=True) + 1e-8)
        else:
            raise ValueError(control)
        out[l] = td
    return out


def run_forward_with_optional_steering(model, tokenizer, df_eval: pd.DataFrame, cfg: Config,
                                       W_np: np.ndarray,
                                       pca: PCA,
                                       baseline_clean_R: Dict[int, Dict[int, float]],
                                       direction_name: str,
                                       control: str,
                                       alpha: float,
                                       window_layers: List[int],
                                       dirs=None) -> pd.DataFrame:
    device = model.device
    W_dev = torch.tensor(W_np, dtype=torch.float32, device=device)
    decision_layers = list(cfg.decision_layers)
    rows = []

    hook_mgr = SteeringHookManager(model, window_layers)
    if alpha != 0.0:
        hook_mgr.register()

    texts = df_eval["text"].tolist()

    try:
        with torch.no_grad():
            for start in range(0, len(df_eval), cfg.batch_size):
                end = min(len(df_eval), start + cfg.batch_size)
                batch = df_eval.iloc[start:end].reset_index(drop=True)

                if alpha != 0.0:
                    hook_mgr.current_dirs = make_batch_dirs(
                        direction_name=direction_name,
                        control=control,
                        batch_rows=batch,
                        layers=window_layers,
                        dirs=dirs,
                        W_np=W_np,
                        cfg=cfg,
                        device=device,
                    )
                    hook_mgr.alpha = alpha

                inputs = tokenizer(texts[start:end], return_tensors="pt", padding=True,
                                   truncation=True, max_length=cfg.max_len).to(device)
                out = model(**inputs, output_hidden_states=True, use_cache=False)
                hstates = out.hidden_states
                pos = last_positions(inputs["attention_mask"])
                bsz = end - start

                cids = torch.tensor(batch["clean_token_id"].values.astype(np.int64), dtype=torch.long, device=device)
                eids = torch.tensor(batch["conflict_token_id"].values.astype(np.int64), dtype=torch.long, device=device)

                row_base = [batch.iloc[bi].to_dict() for bi in range(bsz)]
                R_mat = []
                for l in decision_layers:
                    h = hstates[l + 1][torch.arange(bsz, device=device), pos, :].detach().float()
                    cl = torch.sum(h * W_dev[cids].float(), dim=1)
                    el = torch.sum(h * W_dev[eids].float(), dim=1)
                    R = (cl - el).detach().cpu().numpy().astype(np.float32)
                    R_mat.append(R)
                R_mat = np.stack(R_mat, axis=1)

                dR = np.zeros_like(R_mat, dtype=np.float32)
                for bi in range(bsz):
                    gi = int(batch.iloc[bi]["graph_id"])
                    for li, l in enumerate(decision_layers):
                        dR[bi, li] = R_mat[bi, li] - baseline_clean_R[gi][l]
                du = pca.transform(dR)[:, 0].astype(np.float32)

                hf = hstates[-1][torch.arange(bsz, device=device), pos, :].detach().float()
                cl = torch.sum(hf * W_dev[cids].float(), dim=1)
                el = torch.sum(hf * W_dev[eids].float(), dim=1)
                final_R = (cl - el).detach().cpu().numpy().astype(np.float32)
                pair_choice = np.where(final_R >= 0, "clean", "conflict")

                for bi in range(bsz):
                    rec = row_base[bi]
                    outrow = {
                        "prompt_id": rec["prompt_id"],
                        "graph_id": int(rec["graph_id"]),
                        "condition": rec["condition"],
                        "mechanism": rec["mechanism"],
                        "direction": direction_name,
                        "control": control,
                        "alpha": float(alpha),
                        "window": f"L{window_layers[0]}_{window_layers[-1]}",
                        "DeltaU_steered": float(du[bi]),
                        "final_R_steered": float(final_R[bi]),
                        "pair_choice_steered": str(pair_choice[bi]),
                    }
                    for li, l in enumerate(decision_layers):
                        outrow[f"R_L{l}_steered"] = float(R_mat[bi, li])
                        outrow[f"dR_L{l}_steered"] = float(dR[bi, li])
                    rows.append(outrow)

                print(f"    {direction_name}/{control}/alpha={alpha:.3g}: {end}/{len(df_eval)}")
                del out, hstates, inputs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        hook_mgr.remove()

    return pd.DataFrame(rows)


# ============================================================
# EVALUATION
# ============================================================


def summarize_results(results: pd.DataFrame, base_lookup: pd.DataFrame, save_dir: Path):
    base_cols = ["prompt_id", "DeltaU", "final_R", "pair_choice"]
    merged = results.merge(base_lookup[base_cols], on="prompt_id", how="left")
    merged = merged.rename(columns={"DeltaU": "DeltaU_base", "final_R": "final_R_base", "pair_choice": "pair_choice_base"})
    merged["DeltaU_shift"] = merged["DeltaU_steered"] - merged["DeltaU_base"]
    merged["final_R_shift"] = merged["final_R_steered"] - merged["final_R_base"]
    merged["target_clean"] = merged["direction"].isin(["Q_to_S", "K_to_S"]).astype(int)
    merged["target_hit"] = np.where(
        merged["target_clean"] == 1,
        (merged["pair_choice_steered"] == "clean").astype(int),
        (merged["pair_choice_steered"] == "conflict").astype(int),
    )
    merged.to_csv(save_dir / "asa2_steering_results.csv", index=False, encoding="utf-8-sig")

    group_cols = ["direction", "control", "window", "alpha", "mechanism"]
    dose = merged.groupby(group_cols).agg(
        n=("prompt_id", "count"),
        mean_deltaU_base=("DeltaU_base", "mean"),
        mean_deltaU_steered=("DeltaU_steered", "mean"),
        mean_deltaU_shift=("DeltaU_shift", "mean"),
        mean_abs_deltaU_shift=("DeltaU_shift", lambda x: float(np.mean(np.abs(x)))),
        target_hit_rate=("target_hit", "mean"),
        clean_choice_rate=("pair_choice_steered", lambda x: float(np.mean(np.asarray(x) == "clean"))),
        conflict_choice_rate=("pair_choice_steered", lambda x: float(np.mean(np.asarray(x) == "conflict"))),
        mean_final_R_shift=("final_R_shift", "mean"),
    ).reset_index()
    dose.to_csv(save_dir / "asa2_dose_response.csv", index=False, encoding="utf-8-sig")

    target_mech = {"Q_to_S": "closure", "K_to_S": "competition", "S_to_Q": "stable"}
    spec_rows = []
    for (direction, control, window, alpha), sub in merged.groupby(["direction", "control", "window", "alpha"]):
        tm = target_mech.get(direction)
        if tm is None:
            continue
        target = sub[sub["mechanism"] == tm]
        nontarget = sub[sub["mechanism"] != tm]
        if len(target) == 0 or len(nontarget) == 0:
            continue
        spec_rows.append({
            "direction": direction,
            "control": control,
            "window": window,
            "alpha": float(alpha),
            "target_mechanism": tm,
            "target_abs_shift": float(np.mean(np.abs(target["DeltaU_shift"]))),
            "nontarget_abs_shift": float(np.mean(np.abs(nontarget["DeltaU_shift"]))),
            "specificity_abs_shift": float(np.mean(np.abs(target["DeltaU_shift"])) - np.mean(np.abs(nontarget["DeltaU_shift"]))),
            "target_mean_shift": float(np.mean(target["DeltaU_shift"])),
            "nontarget_mean_shift": float(np.mean(nontarget["DeltaU_shift"])),
            "target_hit_rate": float(np.mean(target["target_hit"])),
            "nontarget_hit_rate": float(np.mean(nontarget["target_hit"])),
        })
    spec = pd.DataFrame(spec_rows)
    spec.to_csv(save_dir / "asa2_specificity_summary.csv", index=False, encoding="utf-8-sig")

    # Shuffle specificity significance.
    proto_spec = spec[spec["control"] == "predictor_residual"][[
        "direction", "window", "alpha", "specificity_abs_shift", "target_abs_shift", "nontarget_abs_shift", "target_hit_rate", "nontarget_hit_rate"
    ]].rename(columns={
        "specificity_abs_shift": "proto_specificity",
        "target_abs_shift": "proto_target_abs_shift",
        "nontarget_abs_shift": "proto_nontarget_abs_shift",
        "target_hit_rate": "proto_target_hit_rate",
        "nontarget_hit_rate": "proto_nontarget_hit_rate",
    })
    orth_spec = spec[spec["control"] == "orthogonal_predictor_residual"][[
        "direction", "window", "alpha", "specificity_abs_shift", "target_abs_shift", "nontarget_abs_shift"
    ]].rename(columns={
        "specificity_abs_shift": "orthogonal_specificity",
        "target_abs_shift": "orthogonal_target_abs_shift",
        "nontarget_abs_shift": "orthogonal_nontarget_abs_shift",
    })
    rand_spec = spec[spec["control"] == "random"][[
        "direction", "window", "alpha", "specificity_abs_shift", "target_abs_shift", "nontarget_abs_shift"
    ]].rename(columns={
        "specificity_abs_shift": "random_specificity",
        "target_abs_shift": "random_target_abs_shift",
        "nontarget_abs_shift": "random_nontarget_abs_shift",
    })
    sh_spec_each = spec[spec["control"].astype(str).str.startswith("shuffle_predictor_residual_")][[
        "direction", "window", "alpha", "control", "specificity_abs_shift", "target_abs_shift", "nontarget_abs_shift", "target_hit_rate", "nontarget_hit_rate"
    ]].rename(columns={
        "specificity_abs_shift": "shuffle_specificity",
        "target_abs_shift": "shuffle_target_abs_shift",
        "nontarget_abs_shift": "shuffle_nontarget_abs_shift",
        "target_hit_rate": "shuffle_target_hit_rate",
        "nontarget_hit_rate": "shuffle_nontarget_hit_rate",
    })
    sh_spec_each.to_csv(save_dir / "asa2_shuffle_specificity_each.csv", index=False, encoding="utf-8-sig")
    if len(sh_spec_each) > 0:
        sh_spec_sum = sh_spec_each.groupby(["direction", "window", "alpha"]).agg(
            shuffle_n=("control", "count"),
            shuffle_specificity_mean=("shuffle_specificity", "mean"),
            shuffle_specificity_std=("shuffle_specificity", "std"),
            shuffle_specificity_p95=("shuffle_specificity", lambda x: float(np.percentile(x, 95))),
            shuffle_specificity_p05=("shuffle_specificity", lambda x: float(np.percentile(x, 5))),
            shuffle_target_abs_mean=("shuffle_target_abs_shift", "mean"),
            shuffle_nontarget_abs_mean=("shuffle_nontarget_abs_shift", "mean"),
            shuffle_target_hit_mean=("shuffle_target_hit_rate", "mean"),
        ).reset_index()
        spec_sig = proto_spec.merge(sh_spec_sum, on=["direction", "window", "alpha"], how="left")
        spec_sig = spec_sig.merge(orth_spec, on=["direction", "window", "alpha"], how="left")
        spec_sig = spec_sig.merge(rand_spec, on=["direction", "window", "alpha"], how="left")
        spec_sig["proto_minus_shuffle_specificity"] = spec_sig["proto_specificity"] - spec_sig["shuffle_specificity_mean"]
        spec_sig["shuffle_z_specificity"] = spec_sig["proto_minus_shuffle_specificity"] / (spec_sig["shuffle_specificity_std"].fillna(0) + 1e-8)
        spec_sig["proto_minus_random_specificity"] = spec_sig["proto_specificity"] - spec_sig["random_specificity"]
        spec_sig["orthogonal_survival_ratio"] = spec_sig["orthogonal_specificity"] / (spec_sig["proto_specificity"].replace(0, np.nan))
    else:
        spec_sig = pd.DataFrame()
    spec_sig.to_csv(save_dir / "asa2_shuffle_specificity_significance.csv", index=False, encoding="utf-8-sig")

    return merged, dose, spec, spec_sig


# ============================================================
# MAIN
# ============================================================


def main(cfg: Config):
    cfg.save_dir = "./asa2_outputs"
    set_seed(cfg.seed)
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "asa2_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False, default=str)

    tokenizer, model = load_tokenizer_and_model(cfg)
    df = build_dataset(tokenizer, cfg, save_dir)
    df.to_csv(save_dir / "asa2_dataset.csv", index=False, encoding="utf-8-sig")
    base_df, centers, pca, d_cols, W_np = extract_baseline_topk_and_decision(model, tokenizer, df, cfg, save_dir)
    # Copy with ASA2 names too.
    base_df.to_csv(save_dir / "asa2_baseline_features.csv", index=False, encoding="utf-8-sig")
    split = make_graph_split(df, cfg, save_dir)
    dirs = build_predictor_directions(base_df, split, centers, cfg, save_dir)
    audit_directions(dirs, cfg, save_dir)

    clean_df = base_df[base_df["condition"] == "clean"].set_index("graph_id")
    baseline_clean_R = {
        int(gi): {int(l): float(row[f"R_L{l}"]) for l in cfg.decision_layers}
        for gi, row in clean_df.iterrows()
    }
    test_indices = split.index[split["split"] == "test"].values
    df_test = df.iloc[test_indices].reset_index(drop=True)

    directions = ["Q_to_S", "K_to_S", "S_to_Q"] if cfg.run_s_to_q else ["Q_to_S", "K_to_S"]
    base_controls = [
        "predictor_raw", "predictor_residual",
        "orthogonal_predictor_raw", "orthogonal_predictor_residual",
        "random", "answer",
    ]
    shuffle_controls = [f"shuffle_predictor_residual_{i:03d}" for i in range(int(cfg.n_shuffles))]

    all_results = []
    for window_name, layers in cfg.steering_windows.items():
        layers = list(layers)
        for direction in directions:
            for control in base_controls + shuffle_controls:
                for alpha in cfg.alphas:
                    if control.startswith("shuffle_predictor_residual_") and float(alpha) not in set(map(float, cfg.shuffle_alphas)):
                        continue
                    if control.startswith("shuffle_predictor_residual_") and float(alpha) == 0.0:
                        continue
                    print(f"\n[ASA2-STEER] window={window_name} direction={direction} control={control} alpha={alpha}")
                    res = run_forward_with_optional_steering(
                        model=model,
                        tokenizer=tokenizer,
                        df_eval=df_test,
                        cfg=cfg,
                        W_np=W_np,
                        pca=pca,
                        baseline_clean_R=baseline_clean_R,
                        direction_name=direction,
                        control=control,
                        alpha=float(alpha),
                        window_layers=layers,
                        dirs=dirs,
                    )
                    res["window_name"] = window_name
                    all_results.append(res)

    results = pd.concat(all_results, ignore_index=True)
    base_lookup = base_df[["prompt_id", "DeltaU", "final_R", "pair_choice"]].copy()
    merged, dose, spec, spec_sig = summarize_results(results, base_lookup, save_dir)

    # Compact verdict at max alpha.
    verdict_rows = []
    if len(spec_sig) > 0:
        max_alpha = float(np.nanmax(spec_sig["alpha"].values))
        for _, row in spec_sig[spec_sig["alpha"] == max_alpha].iterrows():
            verdict_rows.append({
                "direction": row["direction"],
                "window": row["window"],
                "alpha": float(row["alpha"]),
                "proto_specificity": float(row["proto_specificity"]),
                "shuffle_specificity_mean": float(row["shuffle_specificity_mean"]),
                "shuffle_specificity_std": float(row["shuffle_specificity_std"]),
                "shuffle_z_specificity": float(row["shuffle_z_specificity"]),
                "random_specificity": float(row["random_specificity"]),
                "orthogonal_specificity": float(row["orthogonal_specificity"]),
                "orthogonal_survival_ratio": float(row["orthogonal_survival_ratio"]),
                "pass_lite": bool((row["proto_specificity"] > 0) and (row["proto_specificity"] > row["random_specificity"])),
                "pass_strong": bool((row["shuffle_z_specificity"] > 2.0) and (row["proto_specificity"] > 0)),
                "pass_very_strong_hint": bool((row["shuffle_z_specificity"] > 2.0) and (row["orthogonal_survival_ratio"] > 0.3)),
            })
    verdict = pd.DataFrame(verdict_rows)
    verdict.to_csv(save_dir / "asa2_verdict.csv", index=False, encoding="utf-8-sig")

    summary = {
        "audit": "ASA-2 Answerless Predictor-Derived Steering",
        "config": asdict(cfg),
        "n_test_prompts": int(len(df_test)),
        "deltaU_pca_explained_variance": float(pca.explained_variance_ratio_[0]),
        "baseline_pair_choice_rate_test": df_test.merge(base_lookup, on="prompt_id", how="left")["pair_choice"].value_counts(normalize=True).to_dict(),
        "key_outputs": {
            "steering_results": "asa2_steering_results.csv",
            "dose_response": "asa2_dose_response.csv",
            "specificity": "asa2_specificity_summary.csv",
            "shuffle_specificity_significance": "asa2_shuffle_specificity_significance.csv",
            "asa2_verdict": "asa2_verdict.csv",
            "predictor_direction_audit": "asa2_predictor_direction_audit.csv",
        },
        "note": (
            "ASA-2 uses directions derived from an answerless TopK-center -> DeltaU predictor, rather than mechanism mean prototypes. "
            "Answer-token logits/ranks/margins are not used to build steering directions. R_L and DeltaU are used only as evaluation targets."
        ),
    }
    with open(save_dir / "asa2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nASA-2 complete. Outputs saved to:", save_dir)
    if len(verdict):
        print(verdict)
    print("Top dose rows:")
    print(dose.sort_values("mean_abs_deltaU_shift", ascending=False).head(20))


if __name__ == "__main__":
    main(CFG)
