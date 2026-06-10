# ============================================================
# CM-5C: Non-R O-flow Audit
#
# Goal:
#   Move beyond answer-margin O_flow:
#
#       O_adv = dR_{l+1} - dR_l
#
#   and test whether non-R neighborhood dynamics predict later
#   geodesic dominance:
#
#       O_topo_predecision -> DeltaU_decision
#
# Non-R O_topo features:
#   For each transition l -> l+1:
#       center_shift      = 1 - cos(C_{l+1}, C_l)
#       jaccard_turnover  = 1 - Jaccard(TopK_{l+1}, TopK_l)
#       wjaccard_turnover = 1 - weighted Jaccard(TopK logits)
#       spread_delta      = spread_{l+1} - spread_l
#       entropy_delta     = entropy_{l+1} - entropy_l
#
# Target:
#   DeltaU_decision = PC1(dR over decision window)
#
# Leakage control:
#   O_topo transitions are restricted to l+1 < decision_start.
#
# Run:
#   python cm5c_nonR_topo_oflow_audit.py
#
# Outputs:
#   cm5c_outputs/
#       cm5c_model_summary.csv
#       cm5c_cross_model_profile_corr.csv
#       cm5c_overall_summary.json
#       <model>_cm5c_summary.json
#       <model>_nonR_window_summary.csv
#       <model>_predecision_layer_corr.csv
#       <model>_R_dataset.csv
#
# ============================================================

import os
import re
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score
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

CM4D_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\cm4d_outputs")
SAVE_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\cm5c_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 4
MAX_LEN = 320
N_GRAPHS = 36

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

K_LIST = [100, 500]
DECISION_FRAC = (0.70, 0.92)

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

MECH_MAP3 = {"stable": 0, "stable_shift": 0, "competition": 1, "closure": 2}

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
# DATASET
# ============================================================

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

def parse_layers(s):
    nums = re.findall(r"\d+", str(s))
    return [int(x) for x in nums]

def load_cm4d_windows(model_key):
    path = CM4D_DIR / f"{model_key}_best_summary.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for kind in ["topology", "mechanism", "downstream", "overall"]:
        key = f"{kind}_best_layers"
        if key in data:
            out[kind] = parse_layers(data[key])
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

def corr_safe(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def ridge_cv(X, y, groups):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    if X.shape[1] == 0 or len(np.unique(groups)) < 3:
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
    return float(r2_score(y, pred)), corr_safe(y, pred)

def logistic_cv(X, y, groups):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    if X.shape[1] == 0 or len(np.unique(y)) < 2 or len(np.unique(groups)) < 3:
        return np.nan, np.nan
    pred = np.zeros_like(y)
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for tr, te in gkf.split(X, y, groups):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    return float(accuracy_score(y, pred)), float(f1_score(y, pred, average="macro", zero_division=0))

def interpolate_profile(values, n_points=50):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return np.full(n_points, values[0], dtype=float)
    x = np.linspace(0, 1, len(values))
    xi = np.linspace(0, 1, n_points)
    return np.interp(xi, x, values)

def row_cosine(a, b):
    return np.sum(a * b, axis=1) / ((np.linalg.norm(a, axis=1) + 1e-8) * (np.linalg.norm(b, axis=1) + 1e-8))

def jaccard_arrays(a, b):
    # a, b shape [n, k]
    out = np.zeros(a.shape[0], dtype=np.float32)
    for i in range(a.shape[0]):
        sa = set(a[i].tolist())
        sb = set(b[i].tolist())
        out[i] = len(sa & sb) / max(1, len(sa | sb))
    return out

def weighted_jaccard_arrays(ids_a, vals_a, ids_b, vals_b):
    out = np.zeros(ids_a.shape[0], dtype=np.float32)
    for i in range(ids_a.shape[0]):
        ia = ids_a[i]
        ib = ids_b[i]
        va = vals_a[i] - np.min(vals_a[i])
        vb = vals_b[i] - np.min(vals_b[i])
        va = va / (np.sum(va) + 1e-8)
        vb = vb / (np.sum(vb) + 1e-8)
        da = {int(k): float(v) for k, v in zip(ia, va)}
        db = {int(k): float(v) for k, v in zip(ib, vb)}
        keys = set(da.keys()) | set(db.keys())
        num = sum(min(da.get(k, 0.0), db.get(k, 0.0)) for k in keys)
        den = sum(max(da.get(k, 0.0), db.get(k, 0.0)) for k in keys) + 1e-8
        out[i] = num / den
    return out

# ============================================================
# EXTRACTION
# ============================================================

def extract_all(model_key, model_path, df):
    print(f"\n========== {model_key} ==========")
    tok = load_tokenizer(model_path)
    model = load_model(model_path)
    L = get_num_layers(model)
    W = get_lm_head_weight(model).detach().float().to(model.device)
    W_norm = torch.nn.functional.normalize(W.float(), dim=1)

    dec_layers = decision_layers(L)
    dec_start = min(dec_layers)
    pre_max = max(0, dec_start - 2)

    print("num_layers:", L)
    print("decision_layers:", dec_layers)
    print("predecision max transition:", pre_max)

    clean_ids, conflict_ids = [], []
    tok_rows = []
    for _, row in df.iterrows():
        cids = continuation_ids(tok, row["clean_label"])
        eids = continuation_ids(tok, row["conflict_label"])
        tok_rows.append({
            "prompt_id": row["prompt_id"],
            "clean_label": row["clean_label"],
            "conflict_label": row["conflict_label"],
            "clean_ids": str(cids),
            "conflict_ids": str(eids),
            "both_single": int(len(cids) == 1 and len(eids) == 1),
        })
        if len(cids) != 1 or len(eids) != 1:
            raise RuntimeError(f"{model_key}: non-single labels {row['clean_label']} {cids}, {row['conflict_label']} {eids}")
        clean_ids.append(cids[0])
        conflict_ids.append(eids[0])
    pd.DataFrame(tok_rows).to_csv(SAVE_DIR / f"{model_key}_tokenization_audit.csv", index=False, encoding="utf-8-sig")

    needed_layers = sorted(set(list(range(0, pre_max + 2)) + dec_layers))  # need l and l+1 until pre_max+1
    n = len(df)
    texts = df["text"].tolist()
    max_k = max(K_LIST)

    R = np.zeros((n, L), dtype=np.float32)
    topo = {
        k: {
            "centers": {},
            "ids": {},
            "vals": {},
            "spread": {},
            "entropy": {},
        }
        for k in K_LIST
    }

    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            end = min(n, start + BATCH_SIZE)
            inputs = tok(texts[start:end], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(model.device)
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            hstates = outputs.hidden_states
            pos = last_positions(inputs["attention_mask"])
            bsz = end - start

            cids_t = torch.tensor(clean_ids[start:end], dtype=torch.long, device=model.device)
            eids_t = torch.tensor(conflict_ids[start:end], dtype=torch.long, device=model.device)

            for l in needed_layers:
                h = hstates[l + 1][torch.arange(bsz, device=model.device), pos, :].detach().float()

                # R for all needed decision layers and also predecision if desired.
                rc = torch.sum(h * W[cids_t].float(), dim=1)
                re = torch.sum(h * W[eids_t].float(), dim=1)
                R[start:end, l] = (rc - re).detach().cpu().numpy().astype(np.float32)

                if l <= pre_max + 1:
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

                        if l not in topo[k]["centers"]:
                            dim = center.shape[1]
                            topo[k]["centers"][l] = np.zeros((n, dim), dtype=np.float32)
                            topo[k]["ids"][l] = np.zeros((n, k), dtype=np.int32)
                            topo[k]["vals"][l] = np.zeros((n, k), dtype=np.float32)
                            topo[k]["spread"][l] = np.zeros(n, dtype=np.float32)
                            topo[k]["entropy"][l] = np.zeros(n, dtype=np.float32)

                        topo[k]["centers"][l][start:end] = center.detach().cpu().numpy().astype(np.float32)
                        topo[k]["ids"][l][start:end] = ids_k.detach().cpu().numpy().astype(np.int32)
                        topo[k]["vals"][l][start:end] = vals_k.detach().cpu().numpy().astype(np.float32)
                        topo[k]["spread"][l][start:end] = spread.detach().cpu().numpy().astype(np.float32)
                        topo[k]["entropy"][l][start:end] = entropy.detach().cpu().numpy().astype(np.float32)

            print(f"  processed {end}/{n}")

            del outputs, hstates, inputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    del model, tok, W, W_norm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return R, topo, L, dec_layers, pre_max

# ============================================================
# PROCESS MODEL
# ============================================================

def process_model(model_key, model_path, df):
    R, topo, L, dec_layers, pre_max = extract_all(model_key, model_path, df)
    dec_start = min(dec_layers)
    cm4d = load_cm4d_windows(model_key)

    model_df = df.copy()
    for l in range(L):
        model_df[f"R_L{l}"] = R[:, l]

    # clean-relative dR for target only.
    clean_idx = {int(row.graph_id): idx for idx, row in model_df.reset_index(drop=True).iterrows() if row["condition"] == "clean"}
    dR = np.zeros_like(R)
    for i, row in model_df.reset_index(drop=True).iterrows():
        ci = clean_idx[int(row["graph_id"])]
        dR[i] = R[i] - R[ci]
    for l in range(L):
        model_df[f"dR_L{l}"] = dR[:, l]

    X_dec = dR[:, dec_layers]
    pca = PCA(n_components=min(3, X_dec.shape[1]))
    pcs = pca.fit_transform(X_dec)
    deltaU = pcs[:, 0]
    if np.mean(deltaU[model_df["mechanism"] == "stable"]) < np.mean(deltaU[model_df["mechanism"] == "closure"]):
        deltaU = -deltaU
    model_df["DeltaU_decision"] = deltaU.astype(np.float32)

    y3 = model_df["mechanism"].map(MECH_MAP3).values.astype(int)
    groups = model_df["graph_id"].values.astype(int)

    # Window defs, strictly predecision.
    def trans_range(a, b):
        a = max(0, min(pre_max, a))
        b = max(a, min(pre_max, b))
        if b < a:
            return []
        return list(range(a, b + 1))

    early_end = max(1, int(math.floor((L - 1) * 0.25)))
    mid_end = max(early_end + 1, int(math.floor((L - 1) * 0.55)))

    topology = cm4d.get("topology", list(range(0, min(2, L))))
    mechanism = cm4d.get("mechanism", list(range(0, min(6, L))))
    downstream = cm4d.get("downstream", list(range(0, min(3, L))))
    overall = cm4d.get("overall", downstream)

    t_end = max(topology) if topology else 0
    m_end = max(mechanism) if mechanism else 0
    d_end = max(downstream) if downstream else 0
    o_end = max(overall) if overall else 0

    window_defs = {
        "topo_early_0_25depth": trans_range(0, early_end),
        "topo_mid_25_55depth": trans_range(early_end + 1, mid_end),
        "topo_predecision_all": trans_range(0, pre_max),
        "topo_after_topology_until_predecision": trans_range(t_end, pre_max),
        "topo_after_mechanism_until_predecision": trans_range(m_end, pre_max),
        "topo_after_downstream_until_predecision": trans_range(d_end, pre_max),
        "topo_after_overall_until_predecision": trans_range(o_end, pre_max),
        "topo_cm4d_topology_only": trans_range(min(topology) if topology else 0, max(topology)-1 if len(topology) > 1 else max(topology) if topology else 0),
        "topo_cm4d_mechanism_only": trans_range(min(mechanism) if mechanism else 0, max(mechanism)-1 if len(mechanism) > 1 else max(mechanism) if mechanism else 0),
        "topo_cm4d_downstream_only": trans_range(min(downstream) if downstream else 0, max(downstream)-1 if len(downstream) > 1 else max(downstream) if downstream else 0),
    }

    # Build per-transition O_topo for each k.
    O_by_k = {}
    layer_corr_rows = []

    for k in K_LIST:
        # feature tensor n x transitions x f
        features = []
        names = []
        for l in range(0, pre_max + 1):
            C0 = topo[k]["centers"][l]
            C1 = topo[k]["centers"][l + 1]
            center_shift = 1.0 - row_cosine(C0, C1)

            ids0 = topo[k]["ids"][l]
            ids1 = topo[k]["ids"][l + 1]
            vals0 = topo[k]["vals"][l]
            vals1 = topo[k]["vals"][l + 1]

            jac = jaccard_arrays(ids0, ids1)
            wjac = weighted_jaccard_arrays(ids0, vals0, ids1, vals1)

            spread_delta = topo[k]["spread"][l + 1] - topo[k]["spread"][l]
            entropy_delta = topo[k]["entropy"][l + 1] - topo[k]["entropy"][l]
            spread_abs_delta = np.abs(spread_delta)
            entropy_abs_delta = np.abs(entropy_delta)

            F = np.column_stack([
                center_shift,
                1.0 - jac,
                1.0 - wjac,
                spread_delta,
                entropy_delta,
                spread_abs_delta,
                entropy_abs_delta,
            ]).astype(np.float32)
            features.append(F)
            names = [
                "center_shift",
                "jaccard_turnover",
                "weighted_jaccard_turnover",
                "spread_delta",
                "entropy_delta",
                "spread_abs_delta",
                "entropy_abs_delta",
            ]

            # layerwise summary: use all features to ridge DeltaU for each single transition.
            r2_l, corr_l = ridge_cv(F, deltaU, groups)
            acc_l, f1_l = logistic_cv(F, y3, groups)
            layer_corr_rows.append({
                "model_key": model_key,
                "k": k,
                "transition_layer": l,
                "transition": f"L{l}->L{l+1}",
                "layer_frac": l / max(1, L - 2),
                "decision_start": dec_start,
                "ridge_deltaU_r2": r2_l,
                "ridge_deltaU_corr": corr_l,
                "mechanism_acc": acc_l,
                "mechanism_macro_f1": f1_l,
            })

        O_by_k[k] = np.stack(features, axis=1)  # n x T x f

    layer_corr_df = pd.DataFrame(layer_corr_rows)
    layer_corr_df.to_csv(SAVE_DIR / f"{model_key}_predecision_layer_corr.csv", index=False, encoding="utf-8-sig")

    # ============================================================
    # FEATURE-ROW EXPORT FOR STRICT GROUPED BOOTSTRAP
    # ============================================================
    # This is the critical Paper4 v0.9/v1.0 patch.
    # The original CM-5C saved only summaries. That is enough for point
    # estimates but not enough for grouped bootstrap CI of the non-R
    # vocabulary-neighborhood flow. We therefore export one row per:
    #
    #     prompt_id × k × transition_layer
    #
    # Required downstream bootstrap columns:
    #     graph_id, prompt_id, mechanism, DeltaU_decision,
    #     k, transition_layer,
    #     center_shift, jaccard_turnover, weighted_jaccard_turnover,
    #     spread_delta, entropy_delta, spread_abs_delta, entropy_abs_delta
    #
    feature_names = [
        "center_shift",
        "jaccard_turnover",
        "weighted_jaccard_turnover",
        "spread_delta",
        "entropy_delta",
        "spread_abs_delta",
        "entropy_abs_delta",
    ]

    feature_rows = []
    meta_df = model_df.reset_index(drop=True)

    for k in K_LIST:
        O_export = O_by_k[k]  # shape: n_prompt x n_transition x n_feature
        n_prompt, n_transition, n_feature = O_export.shape
        for i in range(n_prompt):
            meta = meta_df.iloc[i]
            for l in range(n_transition):
                row = {
                    "model_key": model_key,
                    "num_layers": L,
                    "decision_start": dec_start,
                    "decision_layers": str(dec_layers),
                    "predecision_max_transition": pre_max,
                    "graph_id": int(meta["graph_id"]),
                    "prompt_id": str(meta["prompt_id"]),
                    "condition": str(meta["condition"]),
                    "condition_class": str(meta["condition_class"]),
                    "mechanism": str(meta["mechanism"]),
                    "clean_label": str(meta["clean_label"]),
                    "conflict_label": str(meta["conflict_label"]),
                    "k": int(k),
                    "transition_layer": int(l),
                    "transition": f"L{l}->L{l+1}",
                    "layer_frac": float(l / max(1, L - 2)),
                    "DeltaU_decision": float(deltaU[i]),
                }
                for j, fname in enumerate(feature_names):
                    row[fname] = float(O_export[i, l, j])
                feature_rows.append(row)

    feature_df = pd.DataFrame(feature_rows)
    feature_path = SAVE_DIR / f"{model_key}_cm5c_nonR_feature_rows.csv"
    feature_df.to_csv(feature_path, index=False, encoding="utf-8-sig")
    print(f"[FEATURE_EXPORT] saved {len(feature_df)} rows -> {feature_path}")

    # Also save compact parquet if pyarrow/fastparquet is available.
    try:
        parquet_path = SAVE_DIR / f"{model_key}_cm5c_nonR_feature_rows.parquet"
        feature_df.to_parquet(parquet_path, index=False)
        print(f"[FEATURE_EXPORT] saved parquet -> {parquet_path}")
    except Exception as e:
        print(f"[FEATURE_EXPORT] parquet skipped: {repr(e)}")

    # Window evaluations.
    rows = []
    best_artifacts = {}

    for k in K_LIST:
        O = O_by_k[k]
        for win_name, layers in window_defs.items():
            layers = [l for l in layers if 0 <= l <= pre_max]
            if len(layers) == 0:
                continue

            X_seq = O[:, layers, :]  # n x t x f

            # Aggregates and flattened profile.
            X_flat = X_seq.reshape(X_seq.shape[0], -1)
            X_mean = X_seq.mean(axis=1)
            X_sum = X_seq.sum(axis=1)
            X_abs_sum = np.abs(X_seq).sum(axis=1)
            X_max = X_seq.max(axis=1)
            X_min = X_seq.min(axis=1)

            X_aug = np.column_stack([X_flat, X_mean, X_sum, X_abs_sum, X_max, X_min]).astype(np.float32)

            r2, corr = ridge_cv(X_aug, deltaU, groups)
            acc, f1 = logistic_cv(X_aug, y3, groups)

            # Simple scalar diagnostics: center shift sum, jaccard turnover sum.
            center_sum = X_seq[:, :, 0].sum(axis=1)
            jac_sum = X_seq[:, :, 1].sum(axis=1)
            wjac_sum = X_seq[:, :, 2].sum(axis=1)
            entropy_abs_sum = X_seq[:, :, 6].sum(axis=1)

            row = {
                "model_key": model_key,
                "k": k,
                "window": win_name,
                "transition_layers": str(layers),
                "n_transitions": len(layers),
                "decision_start": dec_start,
                "decision_layers": str(dec_layers),
                "leakage_free": True,
                "ridge_deltaU_r2": r2,
                "ridge_deltaU_corr": corr,
                "mechanism_acc": acc,
                "mechanism_macro_f1": f1,
                "center_sum_corr_deltaU": corr_safe(center_sum, deltaU),
                "jaccard_sum_corr_deltaU": corr_safe(jac_sum, deltaU),
                "weighted_jaccard_sum_corr_deltaU": corr_safe(wjac_sum, deltaU),
                "entropy_abs_sum_corr_deltaU": corr_safe(entropy_abs_sum, deltaU),
            }
            rows.append(row)

            best_artifacts[(k, win_name)] = {
                "X_aug": X_aug,
                "layers": layers,
                "row": row,
            }

    win_df = pd.DataFrame(rows)
    win_df.to_csv(SAVE_DIR / f"{model_key}_nonR_window_summary.csv", index=False, encoding="utf-8-sig")

    best_delta = win_df.sort_values("ridge_deltaU_corr", ascending=False).iloc[0].to_dict()
    best_mech = win_df.sort_values("mechanism_macro_f1", ascending=False).iloc[0].to_dict()
    peak_layer = layer_corr_df.sort_values("ridge_deltaU_corr", ascending=False).iloc[0].to_dict()

    # Save profile for cross-model: best delta window single-transition corr curve by k best.
    best_k = int(best_delta["k"])
    curve = layer_corr_df[layer_corr_df["k"] == best_k].sort_values("transition_layer")
    profile_corr = interpolate_profile(curve["ridge_deltaU_corr"].values, 50)
    profile_f1 = interpolate_profile(curve["mechanism_macro_f1"].fillna(0).values, 50)
    np.savez_compressed(
        SAVE_DIR / f"{model_key}_nonR_profiles.npz",
        profile_corr=profile_corr,
        profile_f1=profile_f1,
        raw_corr=curve["ridge_deltaU_corr"].values,
        raw_f1=curve["mechanism_macro_f1"].fillna(0).values,
        layer_frac=curve["layer_frac"].values,
    )

    model_df.to_csv(SAVE_DIR / f"{model_key}_R_dataset.csv", index=False, encoding="utf-8-sig")

    summary = {
        "model_key": model_key,
        "num_layers": L,
        "decision_layers": dec_layers,
        "decision_start": dec_start,
        "predecision_max_transition": pre_max,
        "cm4d_windows": cm4d,
        "deltaU_pc_variance": pca.explained_variance_ratio_.tolist(),
        "deltaU_pc1_variance": float(pca.explained_variance_ratio_[0]),
        "best_nonR_deltaU_window": best_delta,
        "best_nonR_mechanism_window": best_mech,
        "peak_nonR_layer_corr": peak_layer,
        "pass_nonR_deltaU_corr_gt_0p3": bool(best_delta["ridge_deltaU_corr"] > 0.3),
        "pass_nonR_mechanism_f1_gt_0p6": bool(best_mech["mechanism_macro_f1"] > 0.6),
    }

    with open(SAVE_DIR / f"{model_key}_cm5c_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary

def compute_cross_model_profiles():
    profiles = {}
    for mk in MODEL_SPECS:
        data = np.load(SAVE_DIR / f"{mk}_nonR_profiles.npz", allow_pickle=True)
        profiles[mk] = {
            "corr": data["profile_corr"],
            "f1": data["profile_f1"],
        }

    rows = []
    for a, b in combinations(MODEL_SPECS.keys(), 2):
        for typ in ["corr", "f1"]:
            rows.append({
                "profile_type": typ,
                "model_a": a,
                "model_b": b,
                "pearson_profile_corr": corr_safe(profiles[a][typ], profiles[b][typ]),
            })
    out = pd.DataFrame(rows)
    out.to_csv(SAVE_DIR / "cm5c_cross_model_profile_corr.csv", index=False, encoding="utf-8-sig")
    return out

# ============================================================
# MAIN
# ============================================================

def main():
    print("Auditing common single-token labels...")
    common_labels = audit_common_single_token_labels(MODEL_SPECS)
    print("Common labels:", common_labels)

    df = build_dataset(common_labels)
    df.to_csv(SAVE_DIR / "cm5c_prompt_dataset.csv", index=False, encoding="utf-8-sig")
    print("Dataset:", len(df), "prompts")

    summaries = []
    for mk, path in MODEL_SPECS.items():
        summaries.append(process_model(mk, path, df))

    flat = []
    for s in summaries:
        bd = s["best_nonR_deltaU_window"]
        bm = s["best_nonR_mechanism_window"]
        peak = s["peak_nonR_layer_corr"]
        flat.append({
            "model_key": s["model_key"],
            "num_layers": s["num_layers"],
            "decision_layers": str(s["decision_layers"]),
            "decision_start": s["decision_start"],
            "predecision_max_transition": s["predecision_max_transition"],
            "deltaU_pc1_variance": s["deltaU_pc1_variance"],
            "best_nonR_deltaU_window": bd["window"],
            "best_nonR_deltaU_k": bd["k"],
            "best_nonR_deltaU_layers": bd["transition_layers"],
            "best_nonR_deltaU_corr": bd["ridge_deltaU_corr"],
            "best_nonR_deltaU_r2": bd["ridge_deltaU_r2"],
            "best_nonR_mechanism_window": bm["window"],
            "best_nonR_mechanism_k": bm["k"],
            "best_nonR_mechanism_layers": bm["transition_layers"],
            "best_nonR_mechanism_f1": bm["mechanism_macro_f1"],
            "best_nonR_mechanism_acc": bm["mechanism_acc"],
            "peak_nonR_transition": peak.get("transition", None),
            "peak_nonR_layer_frac": peak.get("layer_frac", None),
            "peak_nonR_deltaU_corr": peak.get("ridge_deltaU_corr", None),
            "pass_nonR_deltaU_corr_gt_0p3": s["pass_nonR_deltaU_corr_gt_0p3"],
            "pass_nonR_mechanism_f1_gt_0p6": s["pass_nonR_mechanism_f1_gt_0p6"],
        })
    model_summary = pd.DataFrame(flat)
    model_summary.to_csv(SAVE_DIR / "cm5c_model_summary.csv", index=False, encoding="utf-8-sig")

    cross = compute_cross_model_profiles()

    overall = {
        "audit": "CM-5C Non-R TopK/VIM O-flow Audit",
        "definition": (
            "Uses predecision TopK/VIM neighborhood transition features, not dR differences, "
            "to predict later decision-window DeltaU."
        ),
        "models": summaries,
        "cross_model_profile_corr": cross.to_dict(orient="records"),
    }

    with open(SAVE_DIR / "cm5c_overall_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    print("\nCM-5C complete.")
    print(model_summary)
    print(cross)
    print("\n[FEATURE_EXPORT_CHECK]")
    for mk in MODEL_SPECS:
        fp = SAVE_DIR / f"{mk}_cm5c_nonR_feature_rows.csv"
        print(mk, "exists=", fp.exists(), "path=", fp)

if __name__ == "__main__":
    main()
