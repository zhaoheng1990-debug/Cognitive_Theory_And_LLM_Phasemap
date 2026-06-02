# ============================================================
# OA-4I.1: Prompt Initialization TopK Neighborhood Audit
#
# Goal:
#   OA-4I showed shallow C/E margin Rinit does not strongly
#   predict DeltaR20:25.
#
#   This experiment tests whether shallow TopK neighborhoods
#   are a better observable for Prompt initialization:
#
#       Prompt -> TopK(H_init) -> DeltaR20:25
#
# Input:
#   oa4a2_lite_dataset.csv
#
# Outputs:
#   oa4i1_topk_dataset.csv
#   oa4i1_topk_summary.json
# ============================================================

import json
import re
import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score, f1_score

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

CSV_PATH = r"C:\Windows\System32\oa4a2_lite_outputs\oa4a2_lite_dataset.csv"

SAVE_DIR = Path("./oa4i1_topk_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 4
MAX_LEN = 256

# hidden_states[0] = embedding output
# hidden_states[1] = after transformer layer 0
# ...
INIT_STATE_INDICES = list(range(0, 8))  # embedding + L0-L6

DECISION_LAYERS = [20, 21, 22, 23, 24, 25]

TOPK_LIST = [100, 500, 1000]
MAX_TOPK = max(TOPK_LIST)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

# Keep only raw columns. This prevents old R/dR/clean columns
# from previous runs causing duplicated-column bugs.
keep_cols = [
    "graph_id",
    "condition",
    "surface_id",
    "A",
    "B",
    "C",
    "E",
    "prompt",
]

for c in keep_cols:
    if c not in df.columns:
        raise RuntimeError(f"Missing required column: {c}")

df = df[keep_cols].copy()

print("Rows:", len(df))
print("Graphs:", df["graph_id"].nunique())
print("Conditions:", df["condition"].nunique())
print("Surfaces:", df["surface_id"].nunique())

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

# CPU normalized W for TopK center features
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

# audit labels
for _, row in df[["C", "E"]].drop_duplicates().iterrows():
    c_id = single_token_id(row["C"])
    e_id = single_token_id(row["E"])
    if c_id is None or e_id is None:
        raise RuntimeError(f"Non-single token label: {row['C']} / {row['E']}")

# ============================================================
# FORWARD
# ============================================================

def layer_name_from_state_index(si):
    if si == 0:
        return "emb"
    return f"L{si-1}"

def entropy_from_top_logits(top_vals):
    """
    top_vals: torch tensor [k]
    returns entropy over TopK softmax.
    """
    p = torch.softmax(top_vals.float(), dim=-1)
    ent = -torch.sum(p * torch.log(p + 1e-12))
    return float(ent.detach().cpu())

@torch.no_grad()
def compute_batch(prompts, clean_targets, conflict_targets):
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

    # For center vectors and top ids we return side payloads
    # keyed by feature name.
    centers_payload = {}
    topids_payload = {}

    for si in INIT_STATE_INDICES:
        lname = layer_name_from_state_index(si)
        for k in TOPK_LIST:
            centers_payload[f"{lname}_k{k}"] = []
            topids_payload[f"{lname}_k{k}"] = []

    for bi in range(len(prompts)):
        c_id = single_token_id(clean_targets[bi])
        e_id = single_token_id(conflict_targets[bi])

        item = {}

        # ---------- Init TopK features ----------
        for si in INIT_STATE_INDICES:
            lname = layer_name_from_state_index(si)

            h = hs[si][bi, last_idx[bi], :]
            h = h.to(model.lm_head.weight.dtype)

            logits = model.lm_head(h)

            # Answer ranks are not the main hypothesis, but useful controls.
            logit_c = logits[c_id]
            logit_e = logits[e_id]
            rank_c = int((logits > logit_c).sum().detach().cpu().item() + 1)
            rank_e = int((logits > logit_e).sum().detach().cpu().item() + 1)

            item[f"init_{lname}_R_CminusE"] = float((logit_c - logit_e).detach().cpu())
            item[f"init_{lname}_rank_C"] = rank_c
            item[f"init_{lname}_rank_E"] = rank_e
            item[f"init_{lname}_rank_gap_EminusC"] = float(rank_e - rank_c)

            top_vals, top_idx = torch.topk(logits, k=MAX_TOPK)
            top_vals_cpu = top_vals.detach().float().cpu().numpy()
            top_idx_cpu = top_idx.detach().cpu().numpy().astype(np.int32)

            for k in TOPK_LIST:
                idx_k = top_idx_cpu[:k]
                vals_k = top_vals_cpu[:k]

                item[f"init_{lname}_k{k}_top1"] = float(vals_k[0])
                item[f"init_{lname}_k{k}_kth"] = float(vals_k[-1])
                item[f"init_{lname}_k{k}_meanlogit"] = float(np.mean(vals_k))
                item[f"init_{lname}_k{k}_stdlogit"] = float(np.std(vals_k))
                item[f"init_{lname}_k{k}_gap_top1_kth"] = float(vals_k[0] - vals_k[-1])
                item[f"init_{lname}_k{k}_entropy"] = entropy_from_top_logits(
                    torch.tensor(vals_k)
                )

                item[f"init_{lname}_k{k}_has_C"] = int(c_id in set(idx_k.tolist()))
                item[f"init_{lname}_k{k}_has_E"] = int(e_id in set(idx_k.tolist()))

                # center over normalized W rows
                center = np.mean(W_norm[idx_k], axis=0)
                center = center / (np.linalg.norm(center) + 1e-9)

                centers_payload[f"{lname}_k{k}"].append(center.astype(np.float32))
                topids_payload[f"{lname}_k{k}"].append(idx_k.copy())

        # ---------- Decision R20:25 ----------
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

# global center/topid storage
center_store = {}
topid_store = {}

for si in INIT_STATE_INDICES:
    lname = layer_name_from_state_index(si)
    for k in TOPK_LIST:
        key = f"{lname}_k{k}"
        center_store[key] = []
        topid_store[key] = []

for start in range(0, len(df), BATCH_SIZE):
    sub = df.iloc[start:start + BATCH_SIZE]

    rows, centers_payload, topids_payload = compute_batch(
        sub["prompt"].tolist(),
        sub["C"].tolist(),
        sub["E"].tolist(),
    )

    all_rows.extend(rows)

    for key in center_store.keys():
        center_store[key].extend(centers_payload[key])
        topid_store[key].extend(topids_payload[key])

    print(f"{min(start+BATCH_SIZE, len(df))}/{len(df)}")

feat_df = pd.DataFrame(all_rows)
df = pd.concat([df.reset_index(drop=True), feat_df], axis=1)

# Convert center/topid lists to arrays
for key in list(center_store.keys()):
    center_store[key] = np.stack(center_store[key], axis=0)
    topid_store[key] = np.stack(topid_store[key], axis=0)

# ============================================================
# Compute DeltaR20:25 relative to clean per graph + surface
# ============================================================

for layer in DECISION_LAYERS:
    clean_base = (
        df[df["condition"] == "clean"]
        [["graph_id", "surface_id", f"R{layer}"]]
        .copy()
        .rename(columns={f"R{layer}": f"R{layer}_clean"})
    )

    df = df.merge(
        clean_base,
        on=["graph_id", "surface_id"],
        how="left",
        validate="many_to_one",
    )

    df[f"dR{layer}"] = df[f"R{layer}"] - df[f"R{layer}_clean"]

dR_cols = [f"dR{x}" for x in DECISION_LAYERS]

df["deltaR_l2"] = np.sqrt(np.sum(np.square(df[dR_cols].values), axis=1))
df["deltaR_mean"] = np.mean(df[dR_cols].values, axis=1)
df["deltaR_min"] = np.min(df[dR_cols].values, axis=1)

# ============================================================
# TopK delta vs clean: center cosine / Jaccard
# ============================================================

# Build clean index lookup for graph_id + surface_id
clean_rows = df[df["condition"] == "clean"][["graph_id", "surface_id"]].copy()
clean_lookup = {}
for idx, row in clean_rows.iterrows():
    clean_lookup[(int(row["graph_id"]), int(row["surface_id"]))] = idx

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

    for i, row in df[["graph_id", "surface_id"]].iterrows():
        clean_idx = clean_lookup[(int(row["graph_id"]), int(row["surface_id"]))]
        clean_centers[i] = centers[clean_idx]
        clean_topids[i] = topids[clean_idx]

    cos = cosine_rows(centers, clean_centers)
    df[f"topk_{key}_center_cos_to_clean"] = cos
    df[f"topk_{key}_center_dist_to_clean"] = 1.0 - cos

    # Jaccard
    jaccards = []
    k = int(key.split("_k")[-1])

    for i in range(len(df)):
        s1 = set(topids[i, :k].tolist())
        s2 = set(clean_topids[i, :k].tolist())
        inter = len(s1.intersection(s2))
        union = len(s1.union(s2))
        jaccards.append(inter / max(union, 1))

    df[f"topk_{key}_jaccard_to_clean"] = jaccards
    df[f"topk_{key}_jaccard_loss"] = 1.0 - np.array(jaccards)

# ============================================================
# Feature sets
# ============================================================

rank_features = [
    c for c in df.columns
    if c.startswith("init_")
    and (
        "_rank_" in c
        or c.endswith("_R_CminusE")
    )
]

topk_logit_features = [
    c for c in df.columns
    if c.startswith("init_")
    and any(x in c for x in [
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
    and any(x in c for x in [
        "center_cos_to_clean",
        "center_dist_to_clean",
        "jaccard_to_clean",
        "jaccard_loss",
    ])
]

all_topk_features = rank_features + topk_logit_features + topk_delta_features

print("rank_features:", len(rank_features))
print("topk_logit_features:", len(topk_logit_features))
print("topk_delta_features:", len(topk_delta_features))
print("all_topk_features:", len(all_topk_features))

# ============================================================
# Regression: init TopK -> DeltaR20:25
# ============================================================

def group_regression(feature_cols, target="deltaR_l2", group_col="graph_id"):
    X = df[feature_cols].values.astype(float)
    y = df[target].values.astype(float)
    groups = df[group_col].values

    n_splits = min(5, len(np.unique(groups)))
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

# optional classification target:
# closure-like conditions are the strongest rewrite conditions
df["is_closure_like"] = df["condition"].isin([
    "exception_override",
    "rule_update",
    "meta_override",
]).astype(int)

def group_binary(feature_cols, target_col="is_closure_like", group_col="graph_id"):
    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(int)
    groups = df[group_col].values

    n_splits = min(5, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)

    prob = np.zeros(len(y), dtype=float)

    for tr, te in cv.split(X, y, groups):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=3000, class_weight="balanced")),
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

regression_results = {
    "rank_features": group_regression(rank_features, "deltaR_l2"),
    "topk_logit_features": group_regression(topk_logit_features, "deltaR_l2"),
    "topk_delta_features": group_regression(topk_delta_features, "deltaR_l2"),
    "all_topk_features": group_regression(all_topk_features, "deltaR_l2"),
}

binary_results = {
    "rank_features": group_binary(rank_features, "is_closure_like"),
    "topk_logit_features": group_binary(topk_logit_features, "is_closure_like"),
    "topk_delta_features": group_binary(topk_delta_features, "is_closure_like"),
    "all_topk_features": group_binary(all_topk_features, "is_closure_like"),
}

# ============================================================
# Variance audit
# Does surface dominate shallow TopK initialization?
# ============================================================

def variance_ratio(cols):
    surface_var = (
        df.groupby(["graph_id", "condition"])[cols]
        .var()
        .mean()
        .mean()
    )

    condition_var = (
        df.groupby(["graph_id"])[cols]
        .var()
        .mean()
        .mean()
    )

    return {
        "surface_variance": float(surface_var),
        "condition_variance": float(condition_var),
        "surface_over_condition": float(surface_var / (condition_var + 1e-12)),
    }

variance_results = {
    "rank_features": variance_ratio(rank_features),
    "topk_logit_features": variance_ratio(topk_logit_features),
    "topk_delta_features": variance_ratio(topk_delta_features),
    "decision_dR": variance_ratio(dR_cols),
}

# ============================================================
# Condition means
# ============================================================

condition_means_cols = [
    "deltaR_l2",
    "deltaR_mean",
    "deltaR_min",
]

# add compact TopK observables
for c in topk_delta_features:
    if c.endswith("center_dist_to_clean") or c.endswith("jaccard_loss"):
        condition_means_cols.append(c)

# avoid too wide condition means
condition_means_cols = condition_means_cols[:60]

condition_means = (
    df.groupby("condition")[condition_means_cols]
    .mean()
    .reset_index()
    .to_dict(orient="records")
)

# ============================================================
# SAVE
# ============================================================

summary = {
    "experiment": "OA-4I.1 Prompt Initialization TopK Neighborhood Audit",
    "n_rows": int(len(df)),
    "n_graphs": int(df["graph_id"].nunique()),
    "n_conditions": int(df["condition"].nunique()),
    "n_surfaces": int(df["surface_id"].nunique()),
    "init_state_indices": INIT_STATE_INDICES,
    "topk_list": TOPK_LIST,

    "feature_counts": {
        "rank_features": len(rank_features),
        "topk_logit_features": len(topk_logit_features),
        "topk_delta_features": len(topk_delta_features),
        "all_topk_features": len(all_topk_features),
    },

    "regression_topk_init_to_deltaR": regression_results,
    "binary_topk_init_to_closure_like": binary_results,
    "variance_audit": variance_results,
    "condition_means": condition_means,
}

df.to_csv(
    SAVE_DIR / "oa4i1_topk_dataset.csv",
    index=False,
    encoding="utf-8-sig",
)

with open(SAVE_DIR / "oa4i1_topk_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\nSaved to:", SAVE_DIR.resolve())