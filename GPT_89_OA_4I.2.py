# ============================================================
# OA-4I.2: TopK Initialization Decomposition Audit
#
# Goal:
#   Decompose shallow TopK initialization signal into:
#
#       surface component
#       entity component
#       relation wording component
#       structure component
#
# Core hypothesis:
#   Prompt -> Tokenizer/W -> TopK(H0:6) -> Omega_init -> DeltaR20:25
#
# Main questions:
#   1. Does structure explain shallow TopK init beyond surface?
#   2. Can TopK init predict DeltaR20:25 across held-out entities/surfaces/relations?
#   3. Which factor dominates TopK_init variance?
#
# Outputs:
#   oa4i2_dataset.csv
#   oa4i2_summary.json
#   oa4i2_factor_variance.csv
#   oa4i2_feature_importance.csv
# ============================================================

import json
import gc
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    r2_score,
    roc_auc_score,
    accuracy_score,
    f1_score,
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

SAVE_DIR = Path("./oa4i2_topk_decomposition_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

BATCH_SIZE = 4
MAX_LEN = 256

# hidden_states[0] = embedding output
# hidden_states[1] = after transformer layer 0
# ...
INIT_STATE_INDICES = list(range(0, 8))  # embedding + L0-L6
DECISION_LAYERS = [20, 21, 22, 23, 24, 25]

TOPK_LIST = [100, 500, 1000]
MAX_TOPK = max(TOPK_LIST)

# You can reduce these if runtime is too high.
ENTITY_LIMIT = 10
RELATION_LIMIT = 4
SURFACE_LIMIT = 5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model/tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    device_map="auto" if DEVICE == "cuda" else None,
    local_files_only=True,
    trust_remote_code=True,
)

if DEVICE == "cpu":
    model.to(DEVICE)

model.eval()

num_layers = len(model.model.layers)
print("Num layers:", num_layers)

if max(DECISION_LAYERS) >= num_layers:
    raise RuntimeError(
        f"DECISION_LAYERS includes {max(DECISION_LAYERS)}, but model has only {num_layers} layers."
    )

W_cpu = model.lm_head.weight.detach().float().cpu().numpy()
W_norm = W_cpu / (np.linalg.norm(W_cpu, axis=1, keepdims=True) + 1e-9)

VOCAB_SIZE, DIM = W_norm.shape
print("W shape:", W_norm.shape)

# ============================================================
# TOKEN HELPERS
# ============================================================

def encode_no_special(text):
    return tokenizer(text, add_special_tokens=False)["input_ids"]

def single_token_id(text):
    candidates = [
        text,
        " " + text,
        text.lower(),
        " " + text.lower(),
    ]

    for cand in candidates:
        ids = encode_no_special(cand)
        if len(ids) == 1:
            return ids[0]

    return None

def is_single_token(text):
    return single_token_id(text) is not None

def audit_single_token_list(items, name):
    ok = []
    bad = []

    for x in items:
        if is_single_token(x):
            ok.append(x)
        else:
            bad.append(x)

    print(f"{name}: valid={len(ok)}, skipped={len(bad)}")
    if bad:
        print("Skipped:", bad)

    return ok

# ============================================================
# FACTORS
# ============================================================

# Entity factor: A, B are relation nodes.
# C/E are answer labels, selected to be single-token and low-overlap.
RAW_ENTITY_GRAPHS = [
    ("Paris",   "France",   "Red",    "Blue"),
    ("Berlin",  "Germany",  "Green",  "Yellow"),
    ("Tokyo",   "Japan",    "North",  "South"),
    ("Beijing", "China",    "East",   "West"),
    ("doctor",  "hospital", "Gold",   "Iron"),
    ("judge",   "court",    "Apple",  "Orange"),
    ("teacher", "school",   "River",  "Mountain"),
    ("dog",     "animal",   "Sun",    "Moon"),
    ("cat",     "animal",   "Cloud",  "Stone"),
    ("river",   "water",    "Circle", "Square"),
    ("tree",    "plant",    "Copper", "Silver"),
    ("chef",    "kitchen",  "Lemon",  "Pear"),
]

ENTITY_GRAPHS = []
for g in RAW_ENTITY_GRAPHS:
    a, b, c, e = g
    if all(is_single_token(x) for x in [a, b, c, e]):
        ENTITY_GRAPHS.append(g)
    else:
        print("Skipping non-single-token entity graph:", g)

ENTITY_GRAPHS = ENTITY_GRAPHS[:ENTITY_LIMIT]

if len(ENTITY_GRAPHS) < 6:
    raise RuntimeError("Too few valid entity graphs. Adjust labels/entities.")

# Relation wording factor.
# Each relation form has:
#   edge_A_B phrase
#   edge_B_label phrase
#   question phrase
RELATION_FORMS = [
    {
        "relation_id": "belongs_associated",
        "edge_ab": "{A} belongs to {B}.",
        "edge_bc": "{B} is associated with {X}.",
        "question": "Which label is {A} associated with?",
    },
    {
        "relation_id": "inside_maps",
        "edge_ab": "{A} is inside {B}.",
        "edge_bc": "{B} maps to {X}.",
        "question": "Which label does {A} map to?",
    },
    {
        "relation_id": "member_category",
        "edge_ab": "{A} is a member of {B}.",
        "edge_bc": "{B} is linked to {X}.",
        "question": "Which label is {A} linked to?",
    },
    {
        "relation_id": "contained_points",
        "edge_ab": "{A} is contained in {B}.",
        "edge_bc": "{B} points toward {X}.",
        "question": "Which label does {A} point toward?",
    },
]

RELATION_FORMS = RELATION_FORMS[:RELATION_LIMIT]

# Surface factor: wrappers only.
SURFACES = [
    {
        "surface_id": "plain",
        "prefix": "",
        "style_instruction": "",
    },
    {
        "surface_id": "please_consider",
        "prefix": "Please consider the following relation graph.\n",
        "style_instruction": "",
    },
    {
        "surface_id": "given_facts",
        "prefix": "Given the following facts, answer the question.\n",
        "style_instruction": "",
    },
    {
        "surface_id": "knowledge_base",
        "prefix": "Knowledge base:\n",
        "style_instruction": "",
    },
    {
        "surface_id": "reason_carefully",
        "prefix": "Reason carefully using only the provided relations.\n",
        "style_instruction": "",
    },
]

SURFACES = SURFACES[:SURFACE_LIMIT]

# Structure factor.
# C = clean label.
# E = conflict label.
STRUCTURES = [
    {
        "structure_id": "clean",
        "mechanism": "stable",
        "closure_like": 0,
    },
    {
        "structure_id": "weak_distractor",
        "mechanism": "stable_shift",
        "closure_like": 0,
    },
    {
        "structure_id": "ambiguous_branch",
        "mechanism": "competition",
        "closure_like": 0,
    },
    {
        "structure_id": "direct_conflict",
        "mechanism": "competition",
        "closure_like": 0,
    },
    {
        "structure_id": "exception_override",
        "mechanism": "closure",
        "closure_like": 1,
    },
    {
        "structure_id": "rule_update",
        "mechanism": "closure",
        "closure_like": 1,
    },
    {
        "structure_id": "meta_override",
        "mechanism": "closure",
        "closure_like": 1,
    },
]

# ============================================================
# PROMPT GENERATION
# ============================================================

def render_edge_ab(rel, A, B):
    return rel["edge_ab"].format(A=A, B=B)

def render_edge_bx(rel, B, X):
    return rel["edge_bc"].format(B=B, X=X)

def render_question(rel, A):
    return rel["question"].format(A=A)

def make_prompt(A, B, C, E, rel, structure_id, surface):
    option_line = f"Possible answer labels: {C} or {E}."

    edge_ab = render_edge_ab(rel, A, B)
    edge_bc_clean = render_edge_bx(rel, B, C)
    edge_bc_conflict = render_edge_bx(rel, B, E)
    question = render_question(rel, A)

    if structure_id == "clean":
        core = [
            option_line,
            f"Fact 1: {edge_ab}",
            f"Fact 2: {edge_bc_clean}",
            f"Question: {question}",
            f"Answer with exactly one word: {C} or {E}.",
            "Answer:",
        ]

    elif structure_id == "weak_distractor":
        core = [
            option_line,
            f"Fact 1: {edge_ab}",
            f"Fact 2: {edge_bc_clean}",
            f"Unrelated note: another object is sometimes described with label {E}.",
            f"Question: {question}",
            f"Answer with exactly one word: {C} or {E}.",
            "Answer:",
        ]

    elif structure_id == "ambiguous_branch":
        core = [
            option_line,
            f"Fact 1: {edge_ab}",
            f"Fact 2: {edge_bc_clean}",
            f"Ambiguous branch: some descriptions also connect {A} with {E}.",
            f"Question: {question}",
            f"Answer with exactly one word: {C} or {E}.",
            "Answer:",
        ]

    elif structure_id == "direct_conflict":
        core = [
            option_line,
            f"Fact 1: {edge_ab}",
            f"Fact 2: {edge_bc_clean}",
            f"Fact 3: {A} is associated with {E}.",
            f"Question: {question}",
            f"Answer with exactly one word: {C} or {E}.",
            "Answer:",
        ]

    elif structure_id == "exception_override":
        core = [
            option_line,
            f"General rule: objects connected through {B} use label {C}.",
            f"Exception: {A} is a special case using label {E}.",
            f"Fact: {edge_ab}",
            f"Question: {question}",
            f"Answer with exactly one word: {C} or {E}.",
            "Answer:",
        ]

    elif structure_id == "rule_update":
        core = [
            option_line,
            f"Old rule: {edge_bc_clean}",
            f"New rule: in this graph, {edge_bc_conflict}",
            f"Fact: {edge_ab}",
            f"Question: {question}",
            f"Answer with exactly one word: {C} or {E}.",
            "Answer:",
        ]

    elif structure_id == "meta_override":
        core = [
            option_line,
            "Use only the latest rule. Ignore older or general rules.",
            f"Latest rule: {edge_bc_conflict}",
            f"Fact: {edge_ab}",
            f"Question: {question}",
            f"Answer with exactly one word: {C} or {E}.",
            "Answer:",
        ]

    else:
        raise ValueError(structure_id)

    text = "\n".join(core)
    return surface["prefix"] + text

# ============================================================
# BUILD DATASET
# ============================================================

rows = []

for entity_idx, (A, B, C, E) in enumerate(ENTITY_GRAPHS):
    for rel_idx, rel in enumerate(RELATION_FORMS):
        for surface_idx, surface in enumerate(SURFACES):
            for struct_idx, struct in enumerate(STRUCTURES):
                rows.append({
                    "entity_idx": entity_idx,
                    "relation_idx": rel_idx,
                    "surface_idx": surface_idx,
                    "structure_idx": struct_idx,

                    "entity_id": f"E{entity_idx}",
                    "relation_id": rel["relation_id"],
                    "surface_id": surface["surface_id"],
                    "structure_id": struct["structure_id"],

                    "mechanism": struct["mechanism"],
                    "closure_like": int(struct["closure_like"]),

                    "A": A,
                    "B": B,
                    "C": C,
                    "E": E,

                    "prompt": make_prompt(
                        A, B, C, E,
                        rel,
                        struct["structure_id"],
                        surface,
                    ),
                })

df = pd.DataFrame(rows)

print("Dataset rows:", len(df))
print("Entities:", df["entity_id"].nunique())
print("Relations:", df["relation_id"].nunique())
print("Surfaces:", df["surface_id"].nunique())
print("Structures:", df["structure_id"].nunique())

# ============================================================
# TOPK / FORWARD HELPERS
# ============================================================

def state_name(si):
    if si == 0:
        return "emb"
    return f"L{si-1}"

def entropy_np(vals):
    vals = vals.astype(np.float64)
    vals = vals - np.max(vals)
    p = np.exp(vals)
    p = p / (np.sum(p) + 1e-12)
    return float(-np.sum(p * np.log(p + 1e-12)))

@torch.no_grad()
def compute_batch(prompts, clean_labels, conflict_labels):
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
    )

    enc = {k: v.to(model.device) for k, v in enc.items()}

    out = model(
        **enc,
        output_hidden_states=True,
        use_cache=False,
    )

    hs = out.hidden_states
    attn = enc["attention_mask"]
    last_idx = attn.sum(dim=1) - 1

    rows = []

    centers_payload = {}
    topids_payload = {}

    for si in INIT_STATE_INDICES:
        lname = state_name(si)
        for k in TOPK_LIST:
            key = f"{lname}_k{k}"
            centers_payload[key] = []
            topids_payload[key] = []

    for bi in range(len(prompts)):
        c_id = single_token_id(clean_labels[bi])
        e_id = single_token_id(conflict_labels[bi])

        if c_id is None or e_id is None:
            raise RuntimeError(f"Non-single-token label: {clean_labels[bi]} / {conflict_labels[bi]}")

        item = {}

        # ---------- Shallow initialization TopK ----------
        for si in INIT_STATE_INDICES:
            lname = state_name(si)

            h = hs[si][bi, last_idx[bi], :]
            h = h.to(model.lm_head.weight.dtype)

            logits = model.lm_head(h)

            logit_c = logits[c_id]
            logit_e = logits[e_id]

            rank_c = int((logits > logit_c).sum().detach().cpu().item() + 1)
            rank_e = int((logits > logit_e).sum().detach().cpu().item() + 1)

            item[f"init_{lname}_R_CminusE"] = float((logit_c - logit_e).detach().cpu())
            item[f"init_{lname}_rank_C"] = rank_c
            item[f"init_{lname}_rank_E"] = rank_e
            item[f"init_{lname}_rank_gap_EminusC"] = float(rank_e - rank_c)

            top_vals, top_idx = torch.topk(logits, k=MAX_TOPK)

            vals = top_vals.detach().float().cpu().numpy()
            idx = top_idx.detach().cpu().numpy().astype(np.int32)

            for k in TOPK_LIST:
                vals_k = vals[:k]
                idx_k = idx[:k]

                item[f"init_{lname}_k{k}_top1"] = float(vals_k[0])
                item[f"init_{lname}_k{k}_kth"] = float(vals_k[-1])
                item[f"init_{lname}_k{k}_meanlogit"] = float(np.mean(vals_k))
                item[f"init_{lname}_k{k}_stdlogit"] = float(np.std(vals_k))
                item[f"init_{lname}_k{k}_gap_top1_kth"] = float(vals_k[0] - vals_k[-1])
                item[f"init_{lname}_k{k}_entropy"] = entropy_np(vals_k)
                item[f"init_{lname}_k{k}_has_C"] = int(c_id in set(idx_k.tolist()))
                item[f"init_{lname}_k{k}_has_E"] = int(e_id in set(idx_k.tolist()))

                center = np.mean(W_norm[idx_k], axis=0)
                center = center / (np.linalg.norm(center) + 1e-9)

                key = f"{lname}_k{k}"
                centers_payload[key].append(center.astype(np.float32))
                topids_payload[key].append(idx_k.copy())

        # ---------- Decision trajectory R20:25 ----------
        for layer in DECISION_LAYERS:
            h = hs[layer + 1][bi, last_idx[bi], :]
            h = h.to(model.lm_head.weight.dtype)

            logits = model.lm_head(h)
            r = float((logits[c_id] - logits[e_id]).detach().cpu())

            item[f"R{layer}"] = r

        rows.append(item)

    return rows, centers_payload, topids_payload

# ============================================================
# RUN FORWARD
# ============================================================

all_rows = []

center_store = {}
topid_store = {}

for si in INIT_STATE_INDICES:
    lname = state_name(si)
    for k in TOPK_LIST:
        key = f"{lname}_k{k}"
        center_store[key] = []
        topid_store[key] = []

for start in range(0, len(df), BATCH_SIZE):
    sub = df.iloc[start:start+BATCH_SIZE]

    rows, centers_payload, topids_payload = compute_batch(
        sub["prompt"].tolist(),
        sub["C"].tolist(),
        sub["E"].tolist(),
    )

    all_rows.extend(rows)

    for key in center_store.keys():
        center_store[key].extend(centers_payload[key])
        topid_store[key].extend(topids_payload[key])

    print(f"Processed {min(start+BATCH_SIZE, len(df))}/{len(df)}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

feat_df = pd.DataFrame(all_rows)
df = pd.concat([df.reset_index(drop=True), feat_df], axis=1)

for key in center_store:
    center_store[key] = np.stack(center_store[key], axis=0)
    topid_store[key] = np.stack(topid_store[key], axis=0)

# ============================================================
# CLEAN BASELINES
#
# Clean baseline is same:
#   entity_id + relation_id + surface_id
#
# This isolates structure change while preserving entity/relation/surface.
# ============================================================

clean_key_cols = ["entity_id", "relation_id", "surface_id"]

clean_lookup = {}
clean_rows = df[df["structure_id"] == "clean"]

for idx, row in clean_rows.iterrows():
    key = tuple(row[c] for c in clean_key_cols)
    clean_lookup[key] = idx

if len(clean_lookup) != (
    df["entity_id"].nunique()
    * df["relation_id"].nunique()
    * df["surface_id"].nunique()
):
    raise RuntimeError("Clean baseline lookup incomplete.")

# ---------- Delta R20:25 ----------
for layer in DECISION_LAYERS:
    clean_vals = []

    for _, row in df.iterrows():
        key = tuple(row[c] for c in clean_key_cols)
        clean_idx = clean_lookup[key]
        clean_vals.append(df.loc[clean_idx, f"R{layer}"])

    df[f"R{layer}_clean"] = clean_vals
    df[f"dR{layer}"] = df[f"R{layer}"] - df[f"R{layer}_clean"]

dR_cols = [f"dR{x}" for x in DECISION_LAYERS]

df["deltaR_l2"] = np.sqrt(np.sum(np.square(df[dR_cols].values), axis=1))
df["deltaR_mean"] = np.mean(df[dR_cols].values, axis=1)
df["deltaR_min"] = np.min(df[dR_cols].values, axis=1)
df["deltaR_final25"] = df["dR25"]

# ---------- TopK deltas ----------
def cosine_rows(a, b):
    return np.sum(a * b, axis=1) / (
        (np.linalg.norm(a, axis=1) + 1e-9)
        * (np.linalg.norm(b, axis=1) + 1e-9)
    )

for key in center_store.keys():
    centers = center_store[key]
    topids = topid_store[key]

    clean_centers = np.zeros_like(centers)
    clean_topids = np.zeros_like(topids)

    for i, row in df[clean_key_cols].iterrows():
        ckey = tuple(row[c] for c in clean_key_cols)
        clean_idx = clean_lookup[ckey]
        clean_centers[i] = centers[clean_idx]
        clean_topids[i] = topids[clean_idx]

    cos = cosine_rows(centers, clean_centers)

    df[f"topk_{key}_center_cos_to_clean"] = cos
    df[f"topk_{key}_center_dist_to_clean"] = 1.0 - cos

    k = int(key.split("_k")[-1])

    jaccards = []
    for i in range(len(df)):
        s1 = set(topids[i, :k].tolist())
        s2 = set(clean_topids[i, :k].tolist())
        inter = len(s1.intersection(s2))
        union = len(s1.union(s2))
        jaccards.append(inter / max(union, 1))

    df[f"topk_{key}_jaccard_to_clean"] = jaccards
    df[f"topk_{key}_jaccard_loss"] = 1.0 - np.array(jaccards)

# ============================================================
# FEATURE SETS
# ============================================================

rank_features = [
    c for c in df.columns
    if c.startswith("init_")
    and (
        c.endswith("_R_CminusE")
        or "_rank_" in c
    )
]

topk_logit_features = [
    c for c in df.columns
    if c.startswith("init_")
    and any(s in c for s in [
        "_top1",
        "_kth",
        "_meanlogit",
        "_stdlogit",
        "_gap_top1_kth",
        "_entropy",
        "_has_C",
        "_has_E",
    ])
]

topk_delta_features = [
    c for c in df.columns
    if c.startswith("topk_")
    and any(s in c for s in [
        "center_cos_to_clean",
        "center_dist_to_clean",
        "jaccard_to_clean",
        "jaccard_loss",
    ])
]

all_topk_features = rank_features + topk_logit_features + topk_delta_features

feature_sets = {
    "rank_features": rank_features,
    "topk_logit_features": topk_logit_features,
    "topk_delta_features": topk_delta_features,
    "all_topk_features": all_topk_features,
}

print("Feature counts:")
for k, v in feature_sets.items():
    print(k, len(v))

# ============================================================
# FACTOR VARIANCE DECOMPOSITION
#
# Marginal eta^2:
#   SS_factor / SS_total
#
# This is not full factorial ANOVA, but is robust and cheap for
# high-dimensional TopK feature matrices.
# ============================================================

def factor_eta2(feature_cols, factor_col):
    X = df[feature_cols].values.astype(float)

    # Standardize feature dimensions to avoid one dimension dominating.
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-9)

    grand = X.mean(axis=0, keepdims=True)
    ss_total = np.sum((X - grand) ** 2)

    ss_factor = 0.0
    for _, sub in df.groupby(factor_col):
        idx = sub.index.values
        m = X[idx].mean(axis=0, keepdims=True)
        ss_factor += len(idx) * np.sum((m - grand) ** 2)

    return float(ss_factor / (ss_total + 1e-12))

factor_cols = [
    "entity_id",
    "relation_id",
    "surface_id",
    "structure_id",
    "mechanism",
]

factor_variance_rows = []

for fs_name, cols in feature_sets.items():
    for fac in factor_cols:
        factor_variance_rows.append({
            "feature_set": fs_name,
            "factor": fac,
            "eta2": factor_eta2(cols, fac),
        })

# Also decompose decision trajectory.
for fac in factor_cols:
    factor_variance_rows.append({
        "feature_set": "decision_dR",
        "factor": fac,
        "eta2": factor_eta2(dR_cols, fac),
    })

factor_variance_df = pd.DataFrame(factor_variance_rows)

# ============================================================
# PREDICTION HELPERS
# ============================================================

def group_regression(feature_cols, target="deltaR_l2", group_col="entity_id"):
    X = df[feature_cols].values.astype(float)
    y = df[target].values.astype(float)
    groups = df[group_col].values

    unique = np.unique(groups)
    n_splits = min(5, len(unique))

    if n_splits < 2:
        return {"error": f"Not enough groups for {group_col}"}

    cv = GroupKFold(n_splits=n_splits)

    pred = np.zeros(len(y), dtype=float)

    for tr, te in cv.split(X, y, groups):
        reg = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ])
        reg.fit(X[tr], y[tr])
        pred[te] = reg.predict(X[te])

    corr = float(np.corrcoef(y, pred)[0, 1]) if np.std(pred) > 1e-9 else np.nan

    return {
        "target": target,
        "group_col": group_col,
        "n_splits": int(n_splits),
        "r2": float(r2_score(y, pred)),
        "corr": corr,
    }

def group_binary(feature_cols, target_col="closure_like", group_col="entity_id"):
    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(int)
    groups = df[group_col].values

    unique = np.unique(groups)
    n_splits = min(5, len(unique))

    if n_splits < 2:
        return {"error": f"Not enough groups for {group_col}"}

    cv = GroupKFold(n_splits=n_splits)

    prob = np.zeros(len(y), dtype=float)

    for tr, te in cv.split(X, y, groups):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                solver="lbfgs",
            )),
        ])

        clf.fit(X[tr], y[tr])
        prob[te] = clf.predict_proba(X[te])[:, 1]

    pred = (prob >= 0.5).astype(int)

    return {
        "target": target_col,
        "group_col": group_col,
        "n_splits": int(n_splits),
        "auc": float(roc_auc_score(y, prob)),
        "acc": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
    }

def leave_one_group_regression(feature_cols, target="deltaR_l2", group_col="surface_id"):
    X = df[feature_cols].values.astype(float)
    y = df[target].values.astype(float)
    groups = df[group_col].values

    logo = LeaveOneGroupOut()
    pred = np.zeros(len(y), dtype=float)

    for tr, te in logo.split(X, y, groups):
        reg = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ])
        reg.fit(X[tr], y[tr])
        pred[te] = reg.predict(X[te])

    corr = float(np.corrcoef(y, pred)[0, 1]) if np.std(pred) > 1e-9 else np.nan

    return {
        "target": target,
        "group_col": group_col,
        "n_groups": int(len(np.unique(groups))),
        "r2": float(r2_score(y, pred)),
        "corr": corr,
    }

def leave_one_group_binary(feature_cols, target_col="closure_like", group_col="surface_id"):
    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(int)
    groups = df[group_col].values

    logo = LeaveOneGroupOut()
    prob = np.zeros(len(y), dtype=float)

    for tr, te in logo.split(X, y, groups):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                solver="lbfgs",
            )),
        ])

        clf.fit(X[tr], y[tr])
        prob[te] = clf.predict_proba(X[te])[:, 1]

    pred = (prob >= 0.5).astype(int)

    return {
        "target": target_col,
        "group_col": group_col,
        "n_groups": int(len(np.unique(groups))),
        "auc": float(roc_auc_score(y, prob)),
        "acc": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
    }

# ============================================================
# RUN PREDICTION BATTERY
# ============================================================

group_cols_for_cv = [
    "entity_id",
    "surface_id",
    "relation_id",
]

regression_results = {}
binary_results = {}
logo_regression_results = {}
logo_binary_results = {}

for fs_name, cols in feature_sets.items():
    regression_results[fs_name] = {}
    binary_results[fs_name] = {}
    logo_regression_results[fs_name] = {}
    logo_binary_results[fs_name] = {}

    for gcol in group_cols_for_cv:
        regression_results[fs_name][gcol] = group_regression(
            cols,
            target="deltaR_l2",
            group_col=gcol,
        )

        binary_results[fs_name][gcol] = group_binary(
            cols,
            target_col="closure_like",
            group_col=gcol,
        )

        logo_regression_results[fs_name][gcol] = leave_one_group_regression(
            cols,
            target="deltaR_l2",
            group_col=gcol,
        )

        logo_binary_results[fs_name][gcol] = leave_one_group_binary(
            cols,
            target_col="closure_like",
            group_col=gcol,
        )

# ============================================================
# FACTOR-ONLY BASELINES
#
# These help check leakage:
#   If surface-only predicts everything, TopK may be surface signature.
#   If structure-only predicts, task design is clean but expected.
# ============================================================

def factor_only_regression(factors, target="deltaR_l2", group_col="entity_id"):
    X_df = df[factors].astype(str)
    y = df[target].values.astype(float)
    groups = df[group_col].values

    n_splits = min(5, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)

    pred = np.zeros(len(y), dtype=float)

    for tr, te in cv.split(X_df, y, groups):
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), factors),
        ])

        reg = Pipeline([
            ("onehot", pre),
            ("ridge", Ridge(alpha=1.0)),
        ])

        reg.fit(X_df.iloc[tr], y[tr])
        pred[te] = reg.predict(X_df.iloc[te])

    corr = float(np.corrcoef(y, pred)[0, 1]) if np.std(pred) > 1e-9 else np.nan

    return {
        "factors": factors,
        "target": target,
        "group_col": group_col,
        "r2": float(r2_score(y, pred)),
        "corr": corr,
    }

factor_baselines = {
    "surface_only": factor_only_regression(["surface_id"], "deltaR_l2", "entity_id"),
    "entity_only": factor_only_regression(["entity_id"], "deltaR_l2", "surface_id"),
    "relation_only": factor_only_regression(["relation_id"], "deltaR_l2", "entity_id"),
    "structure_only": factor_only_regression(["structure_id"], "deltaR_l2", "entity_id"),
    "surface_plus_structure": factor_only_regression(["surface_id", "structure_id"], "deltaR_l2", "entity_id"),
    "all_factors": factor_only_regression(["entity_id", "relation_id", "surface_id", "structure_id"], "deltaR_l2", "entity_id"),
}

# ============================================================
# FEATURE IMPORTANCE
#
# Use one full-data ridge model to rank broad feature names.
# This is descriptive, not the main evidence.
# ============================================================

def compute_feature_importance(feature_cols, target="deltaR_l2"):
    X = df[feature_cols].values.astype(float)
    y = df[target].values.astype(float)

    model_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])

    model_pipe.fit(X, y)

    coef = model_pipe.named_steps["ridge"].coef_

    rows = []
    for c, w in zip(feature_cols, coef):
        rows.append({
            "feature": c,
            "coef": float(w),
            "abs_coef": float(abs(w)),
        })

    return pd.DataFrame(rows).sort_values("abs_coef", ascending=False)

importance_df = compute_feature_importance(all_topk_features, "deltaR_l2")

# ============================================================
# CONDITION / FACTOR MEANS
# ============================================================

compact_cols = [
    "deltaR_l2",
    "deltaR_mean",
    "deltaR_min",
    "deltaR_final25",
]

# Add a small selected subset of TopK init observables.
for c in topk_delta_features:
    if (
        ("L0_k500" in c or "L2_k500" in c or "L6_k500" in c)
        and (
            "center_dist_to_clean" in c
            or "jaccard_loss" in c
        )
    ):
        compact_cols.append(c)

compact_cols = compact_cols[:80]

condition_means = (
    df.groupby(["structure_id", "mechanism"])[compact_cols]
    .mean()
    .reset_index()
    .to_dict(orient="records")
)

surface_means = (
    df.groupby("surface_id")[compact_cols]
    .mean()
    .reset_index()
    .to_dict(orient="records")
)

relation_means = (
    df.groupby("relation_id")[compact_cols]
    .mean()
    .reset_index()
    .to_dict(orient="records")
)

# ============================================================
# SUMMARY
# ============================================================

summary = {
    "experiment": "OA-4I.2 TopK Initialization Decomposition Audit",
    "n_rows": int(len(df)),
    "n_entities": int(df["entity_id"].nunique()),
    "n_relations": int(df["relation_id"].nunique()),
    "n_surfaces": int(df["surface_id"].nunique()),
    "n_structures": int(df["structure_id"].nunique()),

    "init_state_indices": INIT_STATE_INDICES,
    "decision_layers": DECISION_LAYERS,
    "topk_list": TOPK_LIST,

    "feature_counts": {
        k: len(v) for k, v in feature_sets.items()
    },

    "factor_variance_eta2": factor_variance_df.to_dict(orient="records"),

    "groupkfold_regression_deltaR_l2": regression_results,
    "groupkfold_binary_closure_like": binary_results,

    "leave_one_group_regression_deltaR_l2": logo_regression_results,
    "leave_one_group_binary_closure_like": logo_binary_results,

    "factor_only_baselines": factor_baselines,

    "condition_means": condition_means,
    "surface_means": surface_means,
    "relation_means": relation_means,

    "top_50_feature_importance": importance_df.head(50).to_dict(orient="records"),
}

# ============================================================
# SAVE
# ============================================================

df.to_csv(
    SAVE_DIR / "oa4i2_dataset.csv",
    index=False,
    encoding="utf-8-sig",
)

factor_variance_df.to_csv(
    SAVE_DIR / "oa4i2_factor_variance.csv",
    index=False,
    encoding="utf-8-sig",
)

importance_df.to_csv(
    SAVE_DIR / "oa4i2_feature_importance.csv",
    index=False,
    encoding="utf-8-sig",
)

with open(SAVE_DIR / "oa4i2_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\nSaved to:", SAVE_DIR.resolve())