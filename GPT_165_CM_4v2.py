# ============================================================
# CM-4B FAST: Prompt Initialization Geometry Isomorphism Audit
#
# Fast version:
#   - Avoids all-pairs Jaccard over 432 prompts.
#   - Computes full pairwise geometry only for center distance.
#   - Computes Jaccard / weighted Jaccard only for clean-vs-variant pairs.
#   - Keeps the core logic:
#       relation-preserving invariance
#       structure-changing separation
#       mechanism geometry
#       cross-model center-geometry isomorphism
#
# Run:
#   python cm4b_fast_prompt_initialization_geometry_isomorphism_audit.py
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

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

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

SAVE_DIR = Path("cm4b_fast_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 4
MAX_LEN = 320

# Fast default.
K_LIST = [100, 500]

# Fewer windows; enough for prompt initialization.
WINDOW_SPECS = {
    "abs_L0_6": "abs_L0_6",
    "rel_0_25": (0.00, 0.25),
    "rel_15_35": (0.15, 0.35),
}

DECISION_FRAC = (0.70, 0.92)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

N_GRAPHS = 36

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
                text = make_prompt(cond, a, b, d, clean_label, conflict_label, aux_label, r1, r2)
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
                    "text": text,
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
    return torch.full(
        (attention_mask.shape[0],),
        attention_mask.shape[1] - 1,
        dtype=torch.long,
        device=attention_mask.device,
    )

def layer_windows(num_layers):
    out = {}
    for name, spec in WINDOW_SPECS.items():
        if isinstance(spec, str) and spec == "abs_L0_6":
            out[name] = list(range(0, min(6, num_layers - 1) + 1))
        else:
            a, b = spec
            l0 = max(0, min(num_layers - 1, int(math.floor(num_layers * a))))
            l1 = max(l0, min(num_layers - 1, int(math.floor(num_layers * b))))
            out[name] = list(range(l0, l1 + 1))
    return out

def decision_layers(num_layers):
    a, b = DECISION_FRAC
    l0 = max(0, min(num_layers - 1, int(math.floor(num_layers * a))))
    l1 = max(l0, min(num_layers - 1, int(math.floor(num_layers * b))))
    return list(range(l0, l1 + 1))

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
# EXTRACTION
# ============================================================

def extract_model_geometry(model_key, model_path, df):
    print(f"\n========== {model_key} ==========")
    tokenizer = load_tokenizer(model_path)
    model = load_model(model_path)
    num_layers = get_num_layers(model)
    windows = layer_windows(num_layers)
    dec_layers = decision_layers(num_layers)
    print(f"num_layers={num_layers}")
    print(f"windows={windows}")
    print(f"decision_layers={dec_layers}")

    W = get_lm_head_weight(model).detach().float().to(model.device)
    W_norm = torch.nn.functional.normalize(W.float(), dim=1)

    # Tokenization audit and ids.
    clean_ids = []
    conflict_ids = []
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
            raise RuntimeError(f"{model_key}: non-single token label {row['clean_label']} {cids}, {row['conflict_label']} {eids}")
        clean_ids.append(cids[0])
        conflict_ids.append(eids[0])
    pd.DataFrame(token_rows).to_csv(SAVE_DIR / f"{model_key}_tokenization_audit.csv", index=False, encoding="utf-8-sig")

    texts = df["text"].tolist()
    n = len(df)
    max_k = max(K_LIST)

    needed_layers = sorted(set([l for layers in windows.values() for l in layers] + dec_layers))

    layer_data = {
        k: {
            "centers": {},
            "ids": {},
            "vals": {},
            "spread": {},
            "entropy": {},
        } for k in K_LIST
    }
    R_by_layer = {l: np.zeros(n, dtype=np.float32) for l in dec_layers}

    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            end = min(n, start + BATCH_SIZE)
            batch = texts[start:end]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(model.device)

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

    clean_idx = {}
    for idx, row in model_df.reset_index(drop=True).iterrows():
        if row["condition"] == "clean":
            clean_idx[int(row["graph_id"])] = idx

    summaries = []
    best = None
    best_score = -1e9
    best_distance_matrix = None
    best_feature_df = None

    for k in K_LIST:
        for win_name, layers in windows.items():
            print(f"  geometry {model_key} k={k} window={win_name} layers={layers}")

            centers = np.mean([layer_data[k]["centers"][l] for l in layers], axis=0)
            center_dist = 1.0 - cosine_matrix(centers)
            spread = np.mean([layer_data[k]["spread"][l] for l in layers], axis=0)
            entropy = np.mean([layer_data[k]["entropy"][l] for l in layers], axis=0)

            feat_rows = []
            preserve_d = []
            struct_d = []
            y_sep = []
            sep_score = []

            for i, row in model_df.reset_index(drop=True).iterrows():
                gi = int(row["graph_id"])
                cidx = clean_idx[gi]
                cd = float(center_dist[i, cidx])

                jacs = []
                wjacs = []
                if i != cidx:
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
                    jac_dist = 0.0
                    wjac_dist = 0.0

                composite = 0.50 * cd + 0.25 * jac_dist + 0.25 * wjac_dist

                if row["condition"] in REL_PRESERVE_CONDS:
                    preserve_d.append(composite)
                    y_sep.append(0)
                    sep_score.append(composite)
                elif row["condition"] in STRUCTURE_CHANGE_CONDS:
                    struct_d.append(composite)
                    y_sep.append(1)
                    sep_score.append(composite)

                feat_rows.append({
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

            feat_df = pd.DataFrame(feat_rows)

            preserve_mean = float(np.mean(preserve_d))
            struct_mean = float(np.mean(struct_d))
            sep_auc = safe_auc(y_sep, sep_score)
            invariance_score = 1.0 - preserve_mean
            structure_lift = (struct_mean - preserve_mean) / (preserve_mean + 1e-8)

            # Mechanism classification excluding clean.
            mech_df = feat_df[feat_df["condition"] != "clean"].copy()
            mech_map = {"stable": 0, "stable_shift": 0, "competition": 1, "closure": 2}
            y_mech = mech_df["mechanism"].map(mech_map).values
            groups = mech_df["graph_id"].values
            X = mech_df[["rel_center_dist", "rel_jaccard_dist", "rel_weighted_jaccard_dist", "rel_composite_dist", "spread", "entropy"]].values.astype(np.float32)

            mech_acc, mech_f1 = np.nan, np.nan
            if len(np.unique(y_mech)) >= 2 and len(np.unique(groups)) >= 3:
                preds = np.zeros_like(y_mech)
                gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
                for tr, te in gkf.split(X, y_mech, groups):
                    clf = Pipeline([
                        ("scaler", StandardScaler()),
                        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
                    ])
                    clf.fit(X[tr], y_mech[tr])
                    preds[te] = clf.predict(X[te])
                mech_acc = float(accuracy_score(y_mech, preds))
                mech_f1 = float(f1_score(y_mech, preds, average="macro", zero_division=0))

            # Downstream DeltaU prediction.
            y_reg = feat_df["DeltaU"].values.astype(np.float32)
            groups2 = feat_df["graph_id"].values
            X2 = feat_df[["rel_center_dist", "rel_jaccard_dist", "rel_weighted_jaccard_dist", "rel_composite_dist", "spread", "entropy"]].values.astype(np.float32)

            du_r2, du_corr = np.nan, np.nan
            if len(np.unique(groups2)) >= 3:
                pred = np.zeros_like(y_reg)
                gkf = GroupKFold(n_splits=min(5, len(np.unique(groups2))))
                for tr, te in gkf.split(X2, y_reg, groups2):
                    reg = Pipeline([
                        ("scaler", StandardScaler()),
                        ("ridge", Ridge(alpha=10.0)),
                    ])
                    reg.fit(X2[tr], y_reg[tr])
                    pred[te] = reg.predict(X2[te])
                du_r2 = float(r2_score(y_reg, pred))
                du_corr = float(np.corrcoef(y_reg, pred)[0, 1]) if np.std(pred) > 1e-8 else 0.0

            score = 0.0
            for val, w in [
                (invariance_score, 1.0),
                (sep_auc, 1.0),
                (max(mech_acc, 0) if not np.isnan(mech_acc) else np.nan, 0.7),
                (max(du_corr, 0) if not np.isnan(du_corr) else np.nan, 0.5),
            ]:
                if not np.isnan(val):
                    score += w * val

            row_summary = {
                "model_key": model_key,
                "k": k,
                "window": win_name,
                "layers": str(layers),
                "preserve_distance_mean": preserve_mean,
                "structure_change_distance_mean": struct_mean,
                "invariance_score_1_minus_preserve": invariance_score,
                "structure_separation_auc": sep_auc,
                "structure_lift_over_preserve": structure_lift,
                "mechanism_acc_group": mech_acc,
                "mechanism_macro_f1_group": mech_f1,
                "downstream_deltaU_r2_group": du_r2,
                "downstream_deltaU_corr_group": du_corr,
                "overall_geometry_score": score,
            }
            summaries.append(row_summary)

            if score > best_score:
                best_score = score
                best = row_summary
                best_distance_matrix = center_dist.astype(np.float32)
                best_feature_df = feat_df.copy()

    pd.DataFrame(summaries).to_csv(SAVE_DIR / f"{model_key}_pairwise_geometry_summary.csv", index=False, encoding="utf-8-sig")
    best_feature_df.to_csv(SAVE_DIR / f"{model_key}_best_clean_relative_features.csv", index=False, encoding="utf-8-sig")
    best_feature_df.groupby(["mechanism", "condition"])[
        ["rel_center_dist", "rel_jaccard_dist", "rel_weighted_jaccard_dist", "rel_composite_dist", "DeltaU"]
    ].mean().reset_index().to_csv(SAVE_DIR / f"{model_key}_mechanism_geometry_summary.csv", index=False, encoding="utf-8-sig")

    np.savez_compressed(
        SAVE_DIR / f"{model_key}_best_geometry.npz",
        distance_matrix=best_distance_matrix,
        prompt_ids=model_df["prompt_id"].values,
        graph_ids=model_df["graph_id"].values,
        conditions=model_df["condition"].values,
        mechanisms=model_df["mechanism"].values,
    )

    with open(SAVE_DIR / f"{model_key}_best_summary.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)

    del model, tokenizer, W, W_norm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best

# ============================================================
# CROSS-MODEL
# ============================================================

def upper_tri_values(M):
    return M[np.triu_indices_from(M, k=1)]

def rank_corr(x, y):
    xr = pd.Series(x).rank().values
    yr = pd.Series(y).rank().values
    if np.std(xr) < 1e-8 or np.std(yr) < 1e-8:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])

def compute_cross_model(model_keys):
    mats = {}
    for mk in model_keys:
        data = np.load(SAVE_DIR / f"{mk}_best_geometry.npz", allow_pickle=True)
        mats[mk] = data["distance_matrix"]

    rows = []
    for a, b in combinations(model_keys, 2):
        va = upper_tri_values(mats[a])
        vb = upper_tri_values(mats[b])
        rows.append({
            "model_a": a,
            "model_b": b,
            "pearson_distance_corr": float(np.corrcoef(va, vb)[0, 1]) if np.std(va) > 1e-8 and np.std(vb) > 1e-8 else np.nan,
            "spearman_distance_corr": rank_corr(va, vb),
        })
    out = pd.DataFrame(rows)
    out.to_csv(SAVE_DIR / "cm4b_fast_cross_model_isomorphism.csv", index=False, encoding="utf-8-sig")
    return out

# ============================================================
# MAIN
# ============================================================

def main():
    print("Auditing common single-token labels across all models...")
    common_labels = audit_common_single_token_labels(MODEL_SPECS)
    print("Common single-token labels:", common_labels)

    df = build_dataset(common_labels)
    df.to_csv(SAVE_DIR / "cm4b_fast_prompt_dataset.csv", index=False, encoding="utf-8-sig")
    print("Dataset:", len(df), "prompts")

    best_rows = []
    for mk, path in MODEL_SPECS.items():
        best = extract_model_geometry(mk, path, df)
        best_rows.append(best)

    model_summary = pd.DataFrame(best_rows)
    model_summary.to_csv(SAVE_DIR / "cm4b_fast_model_summary.csv", index=False, encoding="utf-8-sig")

    cross = compute_cross_model(list(MODEL_SPECS.keys()))

    overall = {
        "audit": "CM-4B FAST Prompt Initialization Geometry Isomorphism Audit",
        "note": "Fast version: full pairwise only for center-distance; Jaccard only for clean-vs-variant pairs.",
        "models": best_rows,
        "cross_model_isomorphism": cross.to_dict(orient="records"),
    }

    with open(SAVE_DIR / "cm4b_fast_overall_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    print("\nCM-4B FAST complete.")
    print(model_summary)
    print(cross)

if __name__ == "__main__":
    main()
