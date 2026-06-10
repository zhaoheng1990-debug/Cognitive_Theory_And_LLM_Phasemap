# ============================================================
# CM-4D: Cross-Model Initialization Window Discovery Audit
#
# Purpose:
#   Models have different depths and internal phase partitioning.
#   CM-4D maps, for each model:
#       1. topology emergence window
#       2. mechanism direction-spectrum window
#       3. downstream DeltaU coupling window
#
# It fixes a key CM-4C caveat:
#   weak downstream coupling may be caused by using the wrong
#   initialization window.
#
# Run:
#   python cm4d_initialization_window_discovery_audit.py
#
# Outputs:
#   cm4d_outputs/
#       cm4d_model_summary.csv
#       cm4d_cross_model_isomorphism.csv
#       cm4d_overall_summary.json
#       <model>_window_scan.csv
#       <model>_best_summary.json
#       <model>_best_<topology|mechanism|downstream>_features.csv
#       <model>_best_<topology|mechanism|downstream>_geometry.npz
#
# ============================================================

import os
import gc
import json
import math
import random
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

SAVE_DIR = Path("cm4d_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 4
MAX_LEN = 320
N_GRAPHS = 36

# Fast but still enough for window discovery.
K_LIST = [100, 500]

# Scan only early 45% depth.
MAX_INIT_FRAC = 0.45

# Sliding window lengths in number of layers.
WINDOW_LENGTHS = [2, 3, 4, 5, 6, 8]

# Step size for window starts.
WINDOW_STEP = 1

# Decision target window for DeltaU.
DECISION_FRAC = (0.70, 0.92)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

REL_PRESERVE_CONDS = ["rename", "permuted", "redundant", "irrelevant", "paraphrase"]
STRUCTURE_CHANGE_CONDS = [
    "weak_distractor",
    "competition_balanced",
    "direct_conflict",
    "closure_update",
    "closure_override",
    "exception_override",
]
ALL_CONDS = ["clean"] + REL_PRESERVE_CONDS + STRUCTURE_CHANGE_CONDS

MECHANISM = {
    "clean": "stable",
    "rename": "stable",
    "permuted": "stable",
    "redundant": "stable",
    "irrelevant": "stable",
    "paraphrase": "stable",
    "weak_distractor": "stable_shift",
    "competition_balanced": "competition",
    "direct_conflict": "competition",
    "closure_update": "closure",
    "closure_override": "closure",
    "exception_override": "closure",
}

COND_CLASS = {
    "clean": "clean",
    **{c: "relation_preserving" for c in REL_PRESERVE_CONDS},
    **{c: "structure_changing" for c in STRUCTURE_CHANGE_CONDS},
}

FEATURE_COLS = [
    "rel_center_dist",
    "rel_jaccard_dist",
    "rel_weighted_jaccard_dist",
    "rel_composite_dist",
    "spread",
    "entropy",
]

# ============================================================
# SEED
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ============================================================
# LABELS / DATASET
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
]

REL_WORDS = [
    ("belongs to", "is located at"),
    ("is assigned to", "maps to"),
    ("is part of", "points to"),
    ("is grouped under", "has label"),
]

def continuation_ids(tokenizer, text):
    return tokenizer(" " + text, add_special_tokens=False)["input_ids"]

def audit_common_single_token_labels(model_paths):
    tokenizers = {}
    rows = []
    common = []
    for name, path in model_paths.items():
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
        tokenizers[name] = tok

    for lab in LABEL_CANDIDATES:
        ok = True
        row = {"label": lab}
        for name, tok in tokenizers.items():
            ids = continuation_ids(tok, lab)
            row[f"{name}_ids"] = str(ids)
            row[f"{name}_len"] = len(ids)
            if len(ids) != 1:
                ok = False
        row["is_common_single_token"] = ok
        rows.append(row)
        if ok:
            common.append(lab)

    pd.DataFrame(rows).to_csv(SAVE_DIR / "common_label_tokenization_audit.csv", index=False, encoding="utf-8-sig")
    if len(common) < 12:
        raise RuntimeError(f"Need at least 12 common single-token labels. Found {len(common)}: {common}")
    return common[:12]

def make_prompt(cond, a, b, d, clean_label, conflict_label, aux_label, relation1, relation2):
    if cond == "clean":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} {relation1} {b}.",
            f"Fact 2: {b} {relation2} {clean_label}.",
            f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
            "Answer with exactly one word:",
        ]
    elif cond == "rename":
        lines = [
            "Consider a tiny mapping network.",
            f"Link one: {a} {relation1} {b}.",
            f"Link two: {b} {relation2} {clean_label}.",
            f"Choose the label reached from {a}: {clean_label} or {conflict_label}.",
            "One word answer:",
        ]
    elif cond == "permuted":
        lines = [
            "You are given a small relation graph.",
            f"Fact 2: {b} {relation2} {clean_label}.",
            f"Fact 1: {a} {relation1} {b}.",
            f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
            "Answer with exactly one word:",
        ]
    elif cond == "redundant":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} {relation1} {b}.",
            f"Fact 2: {b} {relation2} {clean_label}.",
            f"Repeated confirmation: {a} still goes through {b}.",
            f"Repeated confirmation: {b} still points to {clean_label}.",
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

def build_dataset(common_labels):
    rows = []
    graph_count = 0
    n_labels = len(common_labels)
    for ent_i, (a, b, d) in enumerate(ENTITIES):
        for rel_i, (r1, r2) in enumerate(REL_WORDS):
            if graph_count >= N_GRAPHS:
                break
            clean_label = common_labels[(2 * graph_count) % n_labels]
            conflict_label = common_labels[(2 * graph_count + 1) % n_labels]
            aux_label = common_labels[(2 * graph_count + 2) % n_labels]
            for cond in ALL_CONDS:
                rows.append({
                    "prompt_id": f"g{graph_count:03d}_{cond}",
                    "graph_id": graph_count,
                    "entity_id": ent_i,
                    "relation_id": rel_i,
                    "condition": cond,
                    "condition_class": COND_CLASS[cond],
                    "mechanism": MECHANISM[cond],
                    "clean_label": clean_label,
                    "conflict_label": conflict_label,
                    "aux_label": aux_label,
                    "text": make_prompt(cond, a, b, d, clean_label, conflict_label, aux_label, r1, r2),
                })
            graph_count += 1
        if graph_count >= N_GRAPHS:
            break
    return pd.DataFrame(rows)

# ============================================================
# MODEL UTILS
# ============================================================

def load_tokenizer(path):
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok

def load_model(path):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    if DEVICE == "cpu":
        model.to(DEVICE)
    model.eval()
    return model

def get_num_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "layers"):
        return len(model.layers)
    raise RuntimeError("Cannot locate layers")

def get_lm_head_weight(model):
    if hasattr(model, "lm_head"):
        return model.lm_head.weight.detach()
    raise RuntimeError("Cannot locate lm_head")

def last_positions(attention_mask):
    return torch.full((attention_mask.shape[0],), attention_mask.shape[1] - 1, dtype=torch.long, device=attention_mask.device)

def decision_layers(num_layers):
    a, b = DECISION_FRAC
    l0 = max(0, min(num_layers - 1, int(math.floor(num_layers * a))))
    l1 = max(l0, min(num_layers - 1, int(math.floor(num_layers * b))))
    return list(range(l0, l1 + 1))

def scan_windows(num_layers):
    max_layer = max(1, int(math.floor((num_layers - 1) * MAX_INIT_FRAC)))
    windows = []
    for length in WINDOW_LENGTHS:
        if length > max_layer + 1:
            continue
        for start in range(0, max_layer - length + 2, WINDOW_STEP):
            layers = list(range(start, start + length))
            layers = [l for l in layers if l < num_layers]
            if len(layers) >= 2:
                windows.append({
                    "window": f"L{layers[0]}_{layers[-1]}",
                    "start": layers[0],
                    "end": layers[-1],
                    "length": len(layers),
                    "start_frac": layers[0] / max(1, num_layers - 1),
                    "end_frac": layers[-1] / max(1, num_layers - 1),
                    "layers": layers,
                })
    # Deduplicate.
    seen, out = set(), []
    for w in windows:
        key = tuple(w["layers"])
        if key not in seen:
            seen.add(key)
            out.append(w)
    return out

def safe_auc(y, score):
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y, score))
    except Exception:
        return np.nan

def cosine_matrix(X):
    X = np.asarray(X, dtype=np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return X @ X.T

def jaccard_from_arrays(a, b):
    sa = set(a.tolist())
    sb = set(b.tolist())
    return len(sa & sb) / max(1, len(sa | sb))

def weighted_jaccard(ids_a, vals_a, ids_b, vals_b):
    wa = vals_a - np.min(vals_a)
    wb = vals_b - np.min(vals_b)
    wa = wa / (np.sum(wa) + 1e-8)
    wb = wb / (np.sum(wb) + 1e-8)
    da = {int(i): float(w) for i, w in zip(ids_a, wa)}
    db = {int(i): float(w) for i, w in zip(ids_b, wb)}
    keys = set(da.keys()) | set(db.keys())
    num = sum(min(da.get(k, 0.0), db.get(k, 0.0)) for k in keys)
    den = sum(max(da.get(k, 0.0), db.get(k, 0.0)) for k in keys) + 1e-8
    return float(num / den)

# ============================================================
# EVALUATION
# ============================================================

def logistic_mechanism_cv(feat_df):
    df2 = feat_df[feat_df["condition"] != "clean"].copy()
    mech_map = {"stable": 0, "stable_shift": 0, "competition": 1, "closure": 2}
    y = df2["mechanism"].map(mech_map).values.astype(int)
    groups = df2["graph_id"].values.astype(int)
    X = df2[FEATURE_COLS].values.astype(np.float32)

    if len(np.unique(y)) < 2 or len(np.unique(groups)) < 3:
        return np.nan, np.nan

    preds = np.zeros_like(y)
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for tr, te in gkf.split(X, y, groups):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        clf.fit(X[tr], y[tr])
        preds[te] = clf.predict(X[te])

    return float(accuracy_score(y, preds)), float(f1_score(y, preds, average="macro", zero_division=0))

def ridge_deltaU_cv(feat_df):
    y = feat_df["DeltaU"].values.astype(float)
    groups = feat_df["graph_id"].values.astype(int)
    X = feat_df[FEATURE_COLS].values.astype(np.float32)

    if len(np.unique(groups)) < 3:
        return np.nan, np.nan

    pred = np.zeros_like(y)
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for tr, te in gkf.split(X, y, groups):
        reg = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0)),
        ])
        reg.fit(X[tr], y[tr])
        pred[te] = reg.predict(X[te])

    r2 = float(r2_score(y, pred))
    corr = float(np.corrcoef(y, pred)[0, 1]) if np.std(pred) > 1e-8 else 0.0
    return r2, corr

def compute_feature_df(model_df, layer_data, k, layers):
    n = len(model_df)
    centers = np.mean([layer_data[k]["centers"][l] for l in layers], axis=0)
    center_dist = 1.0 - cosine_matrix(centers)
    spread = np.mean([layer_data[k]["spread"][l] for l in layers], axis=0)
    entropy = np.mean([layer_data[k]["entropy"][l] for l in layers], axis=0)

    clean_idx = {}
    for idx, row in model_df.reset_index(drop=True).iterrows():
        if row["condition"] == "clean":
            clean_idx[int(row["graph_id"])] = idx

    rows = []
    preserve_d, structure_d = [], []
    y_sep, score_sep = [], []

    for i, row in model_df.reset_index(drop=True).iterrows():
        gi = int(row["graph_id"])
        cidx = clean_idx[gi]
        cd = float(center_dist[i, cidx])

        if i != cidx:
            jacs, wjacs = [], []
            for l in layers:
                ids_i = layer_data[k]["ids"][l][i]
                ids_c = layer_data[k]["ids"][l][cidx]
                vals_i = layer_data[k]["vals"][l][i]
                vals_c = layer_data[k]["vals"][l][cidx]
                jacs.append(jaccard_from_arrays(ids_i, ids_c))
                wjacs.append(weighted_jaccard(ids_i, vals_i, ids_c, vals_c))
            jac_dist = 1.0 - float(np.mean(jacs))
            wjac_dist = 1.0 - float(np.mean(wjacs))
        else:
            jac_dist, wjac_dist = 0.0, 0.0

        composite = 0.50 * cd + 0.25 * jac_dist + 0.25 * wjac_dist

        if row["condition"] in REL_PRESERVE_CONDS:
            preserve_d.append(composite)
            y_sep.append(0)
            score_sep.append(composite)
        elif row["condition"] in STRUCTURE_CHANGE_CONDS:
            structure_d.append(composite)
            y_sep.append(1)
            score_sep.append(composite)

        rows.append({
            "prompt_id": row["prompt_id"],
            "graph_id": gi,
            "condition": row["condition"],
            "condition_class": row["condition_class"],
            "mechanism": row["mechanism"],
            "rel_center_dist": cd,
            "rel_jaccard_dist": jac_dist,
            "rel_weighted_jaccard_dist": wjac_dist,
            "rel_composite_dist": composite,
            "spread": float(spread[i]),
            "entropy": float(entropy[i]),
            "DeltaU": float(row["DeltaU"]),
        })

    feat_df = pd.DataFrame(rows)

    preserve_mean = float(np.mean(preserve_d))
    structure_mean = float(np.mean(structure_d))
    sep_auc = safe_auc(y_sep, score_sep)
    invariance_score = 1.0 - preserve_mean
    structure_lift = (structure_mean - preserve_mean) / (preserve_mean + 1e-8)

    return feat_df, {
        "preserve_distance_mean": preserve_mean,
        "structure_change_distance_mean": structure_mean,
        "invariance_score_1_minus_preserve": invariance_score,
        "structure_separation_auc": sep_auc,
        "structure_lift_over_preserve": structure_lift,
    }, center_dist

# ============================================================
# MODEL RUN
# ============================================================

def extract_model(model_key, model_path, df):
    print(f"\n========== {model_key} ==========")
    tokenizer = load_tokenizer(model_path)
    model = load_model(model_path)
    num_layers = get_num_layers(model)
    dec_layers = decision_layers(num_layers)
    windows = scan_windows(num_layers)

    print("num_layers:", num_layers)
    print("decision_layers:", dec_layers)
    print("n_scan_windows:", len(windows))

    W = get_lm_head_weight(model).detach().float().to(model.device)
    W_norm = torch.nn.functional.normalize(W.float(), dim=1)

    # Token audit.
    clean_ids, conflict_ids = [], []
    token_rows = []
    for _, row in df.iterrows():
        cids = continuation_ids(tokenizer, row["clean_label"])
        eids = continuation_ids(tokenizer, row["conflict_label"])
        token_rows.append({
            "prompt_id": row["prompt_id"],
            "clean_label": row["clean_label"],
            "conflict_label": row["conflict_label"],
            "clean_ids": str(cids),
            "conflict_ids": str(eids),
            "both_single": int(len(cids) == 1 and len(eids) == 1),
        })
        if len(cids) != 1 or len(eids) != 1:
            raise RuntimeError(f"{model_key}: non-single token label: {row['clean_label']} {cids}, {row['conflict_label']} {eids}")
        clean_ids.append(cids[0])
        conflict_ids.append(eids[0])
    pd.DataFrame(token_rows).to_csv(SAVE_DIR / f"{model_key}_tokenization_audit.csv", index=False, encoding="utf-8-sig")

    n = len(df)
    texts = df["text"].tolist()
    max_k = max(K_LIST)

    max_init_layer = max([max(w["layers"]) for w in windows])
    needed_layers = sorted(set(list(range(0, max_init_layer + 1)) + dec_layers))

    layer_data = {
        k: {"centers": {}, "ids": {}, "vals": {}, "spread": {}, "entropy": {}}
        for k in K_LIST
    }
    R_by_layer = {l: np.zeros(n, dtype=np.float32) for l in dec_layers}

    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            end = min(n, start + BATCH_SIZE)
            inputs = tokenizer(texts[start:end], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(model.device)
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            hstates = outputs.hidden_states
            pos = last_positions(inputs["attention_mask"])
            bsz = end - start

            cids_t = torch.tensor(clean_ids[start:end], dtype=torch.long, device=model.device)
            eids_t = torch.tensor(conflict_ids[start:end], dtype=torch.long, device=model.device)

            for l in needed_layers:
                h = hstates[l + 1][torch.arange(bsz, device=model.device), pos, :].detach().float()

                if l in R_by_layer:
                    rc = torch.sum(h * W[cids_t].float(), dim=1)
                    re = torch.sum(h * W[eids_t].float(), dim=1)
                    R_by_layer[l][start:end] = (rc - re).detach().cpu().numpy().astype(np.float32)

                if l <= max_init_layer:
                    logits = h @ W.float().T
                    vals, ids = torch.topk(logits, k=max_k, dim=1)

                    for k in K_LIST:
                        ids_k = ids[:, :k]
                        vals_k = vals[:, :k].float()
                        emb = W_norm[ids_k].float()
                        center = emb.mean(dim=1)
                        center = torch.nn.functional.normalize(center, dim=1)
                        cos_to_center = torch.sum(emb * center[:, None, :], dim=2)
                        spread = (1.0 - cos_to_center).mean(dim=1)
                        p = torch.softmax(vals_k, dim=1)
                        entropy = -(p * torch.log(p + 1e-8)).sum(dim=1) / math.log(k)

                        if l not in layer_data[k]["centers"]:
                            dim = center.shape[1]
                            layer_data[k]["centers"][l] = np.zeros((n, dim), dtype=np.float32)
                            layer_data[k]["ids"][l] = np.zeros((n, k), dtype=np.int32)
                            layer_data[k]["vals"][l] = np.zeros((n, k), dtype=np.float32)
                            layer_data[k]["spread"][l] = np.zeros(n, dtype=np.float32)
                            layer_data[k]["entropy"][l] = np.zeros(n, dtype=np.float32)

                        layer_data[k]["centers"][l][start:end] = center.detach().cpu().numpy().astype(np.float32)
                        layer_data[k]["ids"][l][start:end] = ids_k.detach().cpu().numpy().astype(np.int32)
                        layer_data[k]["vals"][l][start:end] = vals_k.detach().cpu().numpy().astype(np.float32)
                        layer_data[k]["spread"][l][start:end] = spread.detach().cpu().numpy().astype(np.float32)
                        layer_data[k]["entropy"][l][start:end] = entropy.detach().cpu().numpy().astype(np.float32)

            print(f"  forward processed {end}/{n}")
            del outputs, hstates, inputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Build DeltaU target.
    model_df = df.copy()
    for l in dec_layers:
        model_df[f"R_L{l}"] = R_by_layer[l]

    clean_df = model_df[model_df["condition"] == "clean"].set_index("graph_id")
    d_cols = []
    for l in dec_layers:
        vals = []
        for _, row in model_df.iterrows():
            vals.append(row[f"R_L{l}"] - clean_df.loc[row["graph_id"], f"R_L{l}"])
        col = f"dR_L{l}"
        model_df[col] = np.asarray(vals, dtype=np.float32)
        d_cols.append(col)

    X_dec = model_df[d_cols].values.astype(np.float32)
    pca = PCA(n_components=min(3, X_dec.shape[1]))
    pcs = pca.fit_transform(X_dec)
    du = pcs[:, 0]
    if np.mean(du[model_df["mechanism"] == "stable"]) < np.mean(du[model_df["mechanism"] == "closure"]):
        du = -du
    model_df["DeltaU"] = du.astype(np.float32)

    model_df.to_csv(SAVE_DIR / f"{model_key}_prompt_table.csv", index=False, encoding="utf-8-sig")

    scan_rows = []
    best_topology = None
    best_mechanism = None
    best_downstream = None
    best_overall = None
    saved = {}

    for wi, w in enumerate(windows):
        print(f"  scanning window {wi+1}/{len(windows)}: {w['window']} layers={w['layers']}")
        for k in K_LIST:
            feat_df, geom_stats, center_dist = compute_feature_df(model_df, layer_data, k, w["layers"])
            mech_acc, mech_f1 = logistic_mechanism_cv(feat_df)
            du_r2, du_corr = ridge_deltaU_cv(feat_df)

            row = {
                "model_key": model_key,
                "k": k,
                **{kk: vv for kk, vv in w.items() if kk != "layers"},
                "layers": str(w["layers"]),
                **geom_stats,
                "mechanism_acc_group": mech_acc,
                "mechanism_macro_f1_group": mech_f1,
                "downstream_deltaU_r2_group": du_r2,
                "downstream_deltaU_corr_group": du_corr,
            }
            # Component scores.
            row["topology_score"] = row["invariance_score_1_minus_preserve"]
            row["mechanism_score"] = mech_f1 if not np.isnan(mech_f1) else -999
            row["downstream_score"] = du_corr if not np.isnan(du_corr) else -999
            row["overall_score"] = (
                0.5 * row["topology_score"]
                + 0.8 * max(row["mechanism_score"], 0)
                + 0.8 * max(row["downstream_score"], 0)
            )
            scan_rows.append(row)

            candidates = {
                "topology": row["topology_score"],
                "mechanism": row["mechanism_score"],
                "downstream": row["downstream_score"],
                "overall": row["overall_score"],
            }

            for key, score in candidates.items():
                cur = {"topology": best_topology, "mechanism": best_mechanism, "downstream": best_downstream, "overall": best_overall}[key]
                if cur is None or score > cur["score"]:
                    entry = {"score": score, "row": row, "feat_df": feat_df.copy(), "center_dist": center_dist.astype(np.float32)}
                    if key == "topology":
                        best_topology = entry
                    elif key == "mechanism":
                        best_mechanism = entry
                    elif key == "downstream":
                        best_downstream = entry
                    else:
                        best_overall = entry

    scan_df = pd.DataFrame(scan_rows)
    scan_df.to_csv(SAVE_DIR / f"{model_key}_window_scan.csv", index=False, encoding="utf-8-sig")

    best_map = {
        "topology": best_topology,
        "mechanism": best_mechanism,
        "downstream": best_downstream,
        "overall": best_overall,
    }

    best_summary = {"model_key": model_key, "num_layers": num_layers, "decision_layers": dec_layers}
    for key, entry in best_map.items():
        row = entry["row"]
        best_summary[f"{key}_best_score"] = float(entry["score"])
        best_summary[f"{key}_best_k"] = int(row["k"])
        best_summary[f"{key}_best_window"] = row["window"]
        best_summary[f"{key}_best_layers"] = row["layers"]
        best_summary[f"{key}_best_start_frac"] = float(row["start_frac"])
        best_summary[f"{key}_best_end_frac"] = float(row["end_frac"])
        best_summary[f"{key}_best_mechanism_f1"] = float(row["mechanism_macro_f1_group"]) if not np.isnan(row["mechanism_macro_f1_group"]) else None
        best_summary[f"{key}_best_deltaU_corr"] = float(row["downstream_deltaU_corr_group"]) if not np.isnan(row["downstream_deltaU_corr_group"]) else None
        best_summary[f"{key}_best_invariance_score"] = float(row["invariance_score_1_minus_preserve"]) if not np.isnan(row["invariance_score_1_minus_preserve"]) else None

        entry["feat_df"].to_csv(SAVE_DIR / f"{model_key}_best_{key}_features.csv", index=False, encoding="utf-8-sig")
        np.savez_compressed(
            SAVE_DIR / f"{model_key}_best_{key}_geometry.npz",
            distance_matrix=entry["center_dist"],
            prompt_ids=model_df["prompt_id"].values,
            graph_ids=model_df["graph_id"].values,
            conditions=model_df["condition"].values,
            mechanisms=model_df["mechanism"].values,
        )

    with open(SAVE_DIR / f"{model_key}_best_summary.json", "w", encoding="utf-8") as f:
        json.dump(best_summary, f, indent=2, ensure_ascii=False)

    del model, tokenizer, W, W_norm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_summary

# ============================================================
# CROSS MODEL
# ============================================================

def upper_tri_values(M):
    return M[np.triu_indices_from(M, k=1)]

def rank_corr(x, y):
    xr = pd.Series(x).rank().values
    yr = pd.Series(y).rank().values
    if np.std(xr) < 1e-8 or np.std(yr) < 1e-8:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])

def compute_cross_model(model_keys, best_type):
    mats = {}
    for mk in model_keys:
        data = np.load(SAVE_DIR / f"{mk}_best_{best_type}_geometry.npz", allow_pickle=True)
        mats[mk] = data["distance_matrix"]

    rows = []
    for a, b in combinations(model_keys, 2):
        va = upper_tri_values(mats[a])
        vb = upper_tri_values(mats[b])
        rows.append({
            "best_type": best_type,
            "model_a": a,
            "model_b": b,
            "pearson_distance_corr": float(np.corrcoef(va, vb)[0, 1]) if np.std(va) > 1e-8 and np.std(vb) > 1e-8 else np.nan,
            "spearman_distance_corr": rank_corr(va, vb),
        })
    return rows

# ============================================================
# MAIN
# ============================================================

def main():
    print("Auditing common single-token labels across all models...")
    common_labels = audit_common_single_token_labels(MODEL_SPECS)
    print("Common labels:", common_labels)

    df = build_dataset(common_labels)
    df.to_csv(SAVE_DIR / "cm4d_prompt_dataset.csv", index=False, encoding="utf-8-sig")
    print("Dataset:", len(df), "prompts")

    summaries = []
    for mk, path in MODEL_SPECS.items():
        summaries.append(extract_model(mk, path, df))

    model_summary = pd.DataFrame(summaries)
    model_summary.to_csv(SAVE_DIR / "cm4d_model_summary.csv", index=False, encoding="utf-8-sig")

    cross_rows = []
    for best_type in ["topology", "mechanism", "downstream", "overall"]:
        cross_rows.extend(compute_cross_model(list(MODEL_SPECS.keys()), best_type))
    cross_df = pd.DataFrame(cross_rows)
    cross_df.to_csv(SAVE_DIR / "cm4d_cross_model_isomorphism.csv", index=False, encoding="utf-8-sig")

    overall = {
        "audit": "CM-4D Initialization Window Discovery Audit",
        "models": summaries,
        "cross_model_isomorphism": cross_df.to_dict(orient="records"),
        "notes": "Scans early-depth windows to locate topology, mechanism, downstream coupling and overall initialization windows per model."
    }
    with open(SAVE_DIR / "cm4d_overall_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    print("\nCM-4D complete.")
    print(model_summary)
    print(cross_df)

if __name__ == "__main__":
    main()
