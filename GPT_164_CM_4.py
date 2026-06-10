# ============================================================
# CM-4: Cross-Model Prompt Initialization Operator Audit
#
# Goal:
#   Validate the updated prompt theory:
#
#       Prompt --I_m--> (x0, Sigma0, C0)
#
#   In each model m, prompt is not an answer bias, but an
#   initialization operator that creates a shallow TopK / direction
#   spectrum. We test whether shallow TopK initialization features:
#
#       1. encode STRUCTURE more than surface/entity
#       2. predict model-specific DeltaU / PC1 from CM-1 style dR
#       3. generalize across surface families
#
# Models:
#   Qwen2.5-1.5B-Instruct
#   Llama-3.2-1B-Instruct
#   Gemma-2-2B-it
#
# Output:
#   cm4_outputs/
#       cm4_model_summary.csv
#       cm4_overall_summary.json
#       <model>_prompt_init_dataset.csv
#       <model>_factor_variance.csv
#       <model>_prediction_summary.json
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

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    roc_auc_score,
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

QWEN_PATH = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"
LLAMA_PATH = r"D:\model\Llama-3.2-1B-Instruct"
GEMMA_PATH = r"D:\model\gemma-2-2b-it"

MODEL_SPECS = {
    "qwen": {
        "path": QWEN_PATH,
        "max_len": 256,
        "dtype": "auto",
    },
    "llama": {
        "path": LLAMA_PATH,
        "max_len": 256,
        "dtype": "auto",
    },
    "gemma": {
        "path": GEMMA_PATH,
        "max_len": 256,
        "dtype": "auto",
    },
}

SAVE_DIR = Path("cm4_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 4
TOPK = 256

# Use shallow layers only for prompt initialization.
INIT_LAYER_FRAC = 0.25

# Use late/decision layers for DeltaU reconstruction.
DECISION_LAYER_FRAC_RANGE = (0.70, 0.92)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

LABELS = [
    ("Red", "Blue"),
    ("Green", "Yellow"),
    ("North", "South"),
    ("East", "West"),
    ("Copper", "Silver"),
    ("Circle", "Square"),
]

ENTITIES = [
    ("Ava", "Bela", "Cora"),
    ("Darin", "Elo", "Faye"),
    ("Galen", "Hera", "Ivo"),
    ("Juno", "Kira", "Lio"),
    ("Mira", "Nero", "Orin"),
    ("Pia", "Quin", "Rhea"),
]

RELATIONS = [
    ("belongs to", "is located at"),
    ("is assigned to", "maps to"),
    ("is part of", "points to"),
]

SURFACES = {
    "plain": {
        "clean": [
            "{a} {r1} {b}.",
            "{b} {r2} {c}.",
            "Question: which label is associated with {a}: {c} or {e}?",
            "Answer with one word:",
        ],
        "weak": [
            "{a} {r1} {b}.",
            "{b} {r2} {c}.",
            "A weak note says {a} may be associated with {e}.",
            "Question: which label is associated with {a}: {c} or {e}?",
            "Answer with one word:",
        ],
        "compete": [
            "{a} {r1} {b}.",
            "{b} {r2} {c}.",
            "{a} is also associated with {e}.",
            "Question: which label is associated with {a}: {c} or {e}?",
            "Answer with one word:",
        ],
        "closure": [
            "{a} {r1} {b}.",
            "Old record: {b} {r2} {c}.",
            "Updated record: {b} {r2} {e}, not {c}.",
            "Question: which label is associated with {a}: {c} or {e}?",
            "Answer with one word:",
        ],
    },
    "evidence": {
        "clean": [
            "Evidence 1 states that {a} {r1} {b}.",
            "Evidence 2 states that {b} {r2} {c}.",
            "Using the evidence, choose the correct label for {a}: {c} or {e}.",
            "Answer:",
        ],
        "weak": [
            "Evidence 1 states that {a} {r1} {b}.",
            "Evidence 2 states that {b} {r2} {c}.",
            "A weak side note mentions {e}, but does not update the relation.",
            "Using the evidence, choose the correct label for {a}: {c} or {e}.",
            "Answer:",
        ],
        "compete": [
            "Evidence 1 states that {a} {r1} {b}.",
            "Evidence 2 states that {b} {r2} {c}.",
            "Evidence 3 states that {a} {r2} {e}.",
            "Using the evidence, choose the correct label for {a}: {c} or {e}.",
            "Answer:",
        ],
        "closure": [
            "Evidence 1 states that {a} {r1} {b}.",
            "Earlier evidence says {b} {r2} {c}.",
            "Newer evidence replaces it: {b} {r2} {e}.",
            "Using the latest evidence, choose the label for {a}: {c} or {e}.",
            "Answer:",
        ],
    },
    "rule": {
        "clean": [
            "Rule: anything that {r1} {b} receives label {c}.",
            "{a} {r1} {b}.",
            "What label does {a} receive: {c} or {e}?",
            "Answer:",
        ],
        "weak": [
            "Rule: anything that {r1} {b} receives label {c}.",
            "{a} {r1} {b}.",
            "Distractor: another unrelated item receives {e}.",
            "What label does {a} receive: {c} or {e}?",
            "Answer:",
        ],
        "compete": [
            "Rule: anything that {r1} {b} receives label {c}.",
            "{a} {r1} {b}.",
            "Competing claim: {a} receives label {e}.",
            "What label does {a} receive: {c} or {e}?",
            "Answer:",
        ],
        "closure": [
            "Old rule: anything that {r1} {b} receives label {c}.",
            "Override rule: anything that {r1} {b} now receives label {e}.",
            "{a} {r1} {b}.",
            "Using the override rule, what label does {a} receive: {c} or {e}?",
            "Answer:",
        ],
    },
    "story": {
        "clean": [
            "In a small graph, {a} travels through {b}.",
            "The path from {b} ends at {c}.",
            "Where does the path from {a} end: {c} or {e}?",
            "Answer:",
        ],
        "weak": [
            "In a small graph, {a} travels through {b}.",
            "The path from {b} ends at {c}.",
            "A rumor mentions {e}, but no path uses it.",
            "Where does the path from {a} end: {c} or {e}?",
            "Answer:",
        ],
        "compete": [
            "In a small graph, {a} travels through {b}.",
            "The path from {b} ends at {c}.",
            "Another branch sends {a} toward {e}.",
            "Where does the path from {a} end: {c} or {e}?",
            "Answer:",
        ],
        "closure": [
            "In a small graph, {a} travels through {b}.",
            "The old path from {b} ended at {c}.",
            "The path was revised: {b} now ends at {e}.",
            "Where does the path from {a} end now: {c} or {e}?",
            "Answer:",
        ],
    },
}

STRUCTURES = ["clean", "weak", "compete", "closure"]
MECHANISM_LABEL = {
    "clean": "stable",
    "weak": "stable",
    "compete": "competition",
    "closure": "closure",
}
CLOSURE_LIKE = {"closure"}

def make_prompt(lines, **kw):
    return "\n".join([x.format(**kw) for x in lines])

def build_dataset():
    rows = []
    idx = 0

    for ent_i, (a, b, aux) in enumerate(ENTITIES):
        c, e = LABELS[ent_i % len(LABELS)]

        for rel_i, (r1, r2) in enumerate(RELATIONS):
            for surface_name, surface_templates in SURFACES.items():
                for structure in STRUCTURES:
                    lines = surface_templates[structure]
                    text = make_prompt(
                        lines,
                        a=a,
                        b=b,
                        aux=aux,
                        c=c,
                        e=e,
                        r1=r1,
                        r2=r2,
                    )
                    rows.append({
                        "idx": idx,
                        "entity_id": ent_i,
                        "relation_id": rel_i,
                        "surface": surface_name,
                        "structure": structure,
                        "mechanism": MECHANISM_LABEL[structure],
                        "closure_like": int(structure in CLOSURE_LIKE),
                        "clean_label": c,
                        "conflict_label": e,
                        "text": text,
                    })
                    idx += 1

    return pd.DataFrame(rows)

DATASET = build_dataset()


# ============================================================
# TOKEN HELPERS
# ============================================================

def token_ids_for_label(tokenizer, label):
    # continuation tokenization
    ids = tokenizer(" " + label, add_special_tokens=False)["input_ids"]
    return ids

def first_token_id_for_label(tokenizer, label):
    ids = token_ids_for_label(tokenizer, label)
    if len(ids) < 1:
        raise RuntimeError(f"No ids for label {label}")
    return ids[0]


# ============================================================
# MODEL HELPERS
# ============================================================

def get_dtype(spec):
    if spec.get("dtype", "auto") == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    return getattr(torch, spec["dtype"])

def load_model_and_tokenizer(path, dtype):
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model.to("cpu")
    model.eval()

    return model, tokenizer

def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "layers"):
        return model.layers
    raise RuntimeError("Cannot locate model layers")

def get_lm_head_weight(model):
    if hasattr(model, "lm_head"):
        return model.lm_head.weight.detach()
    if hasattr(model, "embed_out"):
        return model.embed_out.weight.detach()
    raise RuntimeError("Cannot locate lm_head weight")

def last_token_positions(attention_mask):
    # with left padding, final real token is at the last column
    return torch.full(
        (attention_mask.shape[0],),
        attention_mask.shape[1] - 1,
        dtype=torch.long,
        device=attention_mask.device,
    )

def layer_windows(num_layers):
    init_end = max(2, int(round(num_layers * INIT_LAYER_FRAC)))
    init_layers = list(range(0, min(init_end + 1, num_layers)))

    a, b = DECISION_LAYER_FRAC_RANGE
    l0 = max(0, int(math.floor(num_layers * a)))
    l1 = min(num_layers - 1, int(math.floor(num_layers * b)))
    decision_layers = list(range(l0, l1 + 1))

    return init_layers, decision_layers


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_model_dataset(model_key, spec):
    print(f"\n==============================")
    print(f"Running CM-4 for {model_key}")
    print(f"Path: {spec['path']}")
    print(f"==============================")

    dtype = get_dtype(spec)
    model, tokenizer = load_model_and_tokenizer(spec["path"], dtype)
    layers = get_layers(model)
    num_layers = len(layers)
    init_layers, decision_layers = layer_windows(num_layers)

    print(f"num_layers={num_layers}")
    print(f"init_layers={init_layers}")
    print(f"decision_layers={decision_layers}")

    W = get_lm_head_weight(model).detach().to(DEVICE if torch.cuda.is_available() else "cpu")
    W_norm = torch.nn.functional.normalize(W.float(), dim=1)

    df = DATASET.copy()

    clean_ids = []
    conflict_ids = []
    for _, row in df.iterrows():
        clean_ids.append(first_token_id_for_label(tokenizer, row["clean_label"]))
        conflict_ids.append(first_token_id_for_label(tokenizer, row["conflict_label"]))
    df["clean_token_id"] = clean_ids
    df["conflict_token_id"] = conflict_ids

    rows = []

    texts = df["text"].tolist()
    max_len = spec.get("max_len", 256)

    with torch.no_grad():
        for start in range(0, len(df), BATCH_SIZE):
            end = min(len(df), start + BATCH_SIZE)
            batch = df.iloc[start:end]
            batch_texts = texts[start:end]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )

            hidden_states = outputs.hidden_states
            # hidden_states[0] is embedding; hidden_states[1:] are layer outputs
            last_idx = last_token_positions(inputs["attention_mask"])

            cids = torch.tensor(batch["clean_token_id"].values, dtype=torch.long, device=W.device)
            eids = torch.tensor(batch["conflict_token_id"].values, dtype=torch.long, device=W.device)

            record = batch.reset_index(drop=True).to_dict(orient="records")

            # initialize per-row output
            out_rows = []
            for r in record:
                out = dict(r)
                out_rows.append(out)

            # decision R_l and dR features
            for l in range(num_layers):
                hs = hidden_states[l + 1] if (l + 1) < len(hidden_states) else hidden_states[-1]
                h = hs[
                    torch.arange(hs.shape[0], device=hs.device),
                    last_idx,
                    :
                ].detach().float().to(W.device)

                clean_logits = torch.sum(h * W[cids].float(), dim=1)
                conflict_logits = torch.sum(h * W[eids].float(), dim=1)
                R = (clean_logits - conflict_logits).detach().cpu().numpy()

                for bi, val in enumerate(R):
                    out_rows[bi][f"R_L{l}"] = float(val)

            # shallow TopK features
            for l in init_layers:
                hs = hidden_states[l + 1] if (l + 1) < len(hidden_states) else hidden_states[-1]
                h = hs[
                    torch.arange(hs.shape[0], device=hs.device),
                    last_idx,
                    :
                ].detach().float().to(W.device)

                logits = h @ W.float().T
                vals, ids = torch.topk(logits, k=min(TOPK, logits.shape[1]), dim=1)

                top_emb = W_norm[ids].float()
                center = top_emb.mean(dim=1)
                center = torch.nn.functional.normalize(center, dim=1)

                spread = 1.0 - torch.sum(top_emb * center[:, None, :], dim=2)
                spread_mean = spread.mean(dim=1)
                spread_std = spread.std(dim=1)

                top_mean = vals.float().mean(dim=1)
                top_std = vals.float().std(dim=1)
                top_gap = vals[:, 0].float() - vals[:, -1].float()

                # candidate membership / rank proxy
                ids_cpu = ids.detach().cpu().numpy()
                c_np = cids.detach().cpu().numpy()
                e_np = eids.detach().cpu().numpy()

                for bi in range(len(out_rows)):
                    token_set = set(ids_cpu[bi].tolist())
                    out_rows[bi][f"init_L{l}_topk_mean"] = float(top_mean[bi].detach().cpu())
                    out_rows[bi][f"init_L{l}_topk_std"] = float(top_std[bi].detach().cpu())
                    out_rows[bi][f"init_L{l}_topk_gap"] = float(top_gap[bi].detach().cpu())
                    out_rows[bi][f"init_L{l}_spread_mean"] = float(spread_mean[bi].detach().cpu())
                    out_rows[bi][f"init_L{l}_spread_std"] = float(spread_std[bi].detach().cpu())
                    out_rows[bi][f"init_L{l}_clean_in_topk"] = int(c_np[bi] in token_set)
                    out_rows[bi][f"init_L{l}_conflict_in_topk"] = int(e_np[bi] in token_set)

            rows.extend(out_rows)

            print(f"  processed {end}/{len(df)}")

            del outputs, hidden_states, inputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    model_df = pd.DataFrame(rows)

    # Compute dR relative to clean condition for same entity/relation/surface.
    # This is the clean-relative structure component.
    key_cols = ["entity_id", "relation_id", "surface"]

    clean_df = model_df[model_df["structure"] == "clean"].copy()
    clean_map = clean_df.set_index(key_cols)

    for col in list(model_df.columns):
        if col.startswith("R_L") or col.startswith("init_L"):
            clean_vals = []
            for _, row in model_df.iterrows():
                key = tuple(row[k] for k in key_cols)
                clean_vals.append(clean_map.loc[key, col])
            model_df[f"d_{col}"] = model_df[col].values - np.asarray(clean_vals, dtype=np.float32)

    # Decision DeltaR vector
    decision_R_cols = [f"R_L{l}" for l in decision_layers]
    decision_d_cols = [f"d_R_L{l}" for l in decision_layers]

    X_dec = model_df[decision_d_cols].values.astype(np.float32)
    pca = PCA(n_components=min(3, X_dec.shape[1]))
    pcs = pca.fit_transform(X_dec)
    delta_u = pcs[:, 0]

    # Orient so clean/stable is greater than closure if possible
    mean_clean = np.mean(delta_u[model_df["structure"].values == "clean"])
    mean_closure = np.mean(delta_u[model_df["structure"].values == "closure"])
    if mean_clean < mean_closure:
        delta_u = -delta_u

    model_df["DeltaU_CM4"] = delta_u.astype(np.float32)

    # Feature groups
    raw_topk_cols = [c for c in model_df.columns if c.startswith("init_L")]
    delta_topk_cols = [c for c in model_df.columns if c.startswith("d_init_L")]
    rank_cols = [c for c in raw_topk_cols + delta_topk_cols if ("clean_in_topk" in c or "conflict_in_topk" in c)]
    logit_cols = [c for c in raw_topk_cols + delta_topk_cols if ("topk_" in c)]
    spread_cols = [c for c in raw_topk_cols + delta_topk_cols if ("spread_" in c)]

    # Remove label identity artifacts from raw groups when possible by evaluating delta groups separately.
    feature_groups = {
        "raw_topk_all": raw_topk_cols,
        "delta_topk_all": delta_topk_cols,
        "delta_logit": [c for c in delta_topk_cols if "topk_" in c],
        "delta_spread": [c for c in delta_topk_cols if "spread_" in c],
        "delta_rank": [c for c in delta_topk_cols if ("clean_in_topk" in c or "conflict_in_topk" in c)],
    }

    # Factor variance: eta^2 of each factor for each feature group first PC
    factor_rows = []
    factors = ["entity_id", "relation_id", "surface", "structure", "mechanism"]

    def eta_squared(y, groups):
        y = np.asarray(y, dtype=np.float64)
        groups = np.asarray(groups)
        grand = np.mean(y)
        ss_total = np.sum((y - grand) ** 2) + 1e-12
        ss_between = 0.0
        for g in np.unique(groups):
            mask = groups == g
            ss_between += np.sum(mask) * (np.mean(y[mask]) - grand) ** 2
        return float(ss_between / ss_total)

    for group_name, cols in feature_groups.items():
        cols = [c for c in cols if c in model_df.columns]
        if len(cols) == 0:
            continue
        X = model_df[cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        Xs = StandardScaler().fit_transform(X)
        pc = PCA(n_components=1).fit_transform(Xs)[:, 0]
        for factor in factors:
            factor_rows.append({
                "model_key": model_key,
                "feature_group": group_name,
                "factor": factor,
                "eta2": eta_squared(pc, model_df[factor].values),
                "n_features": len(cols),
            })

    factor_df = pd.DataFrame(factor_rows)

    # Prediction tests:
    pred_rows = []
    y_reg = model_df["DeltaU_CM4"].values.astype(np.float32)
    y_bin = model_df["closure_like"].values.astype(int)

    def safe_auc(y, score):
        if len(np.unique(y)) < 2:
            return np.nan
        return float(roc_auc_score(y, score))

    def run_group_regression(group_name, cols, group_col):
        cols = [c for c in cols if c in model_df.columns]
        if len(cols) == 0:
            return None
        X = model_df[cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        groups = model_df[group_col].values

        preds = np.zeros(len(model_df), dtype=np.float32)
        gkf = GroupKFold(n_splits=min(4, len(np.unique(groups))))
        for tr, te in gkf.split(X, y_reg, groups):
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=10.0)),
            ])
            pipe.fit(X[tr], y_reg[tr])
            preds[te] = pipe.predict(X[te])

        return {
            "task": "DeltaU_regression",
            "feature_group": group_name,
            "heldout_group": group_col,
            "r2": float(r2_score(y_reg, preds)),
            "corr": float(np.corrcoef(y_reg, preds)[0, 1]) if np.std(preds) > 1e-8 else 0.0,
            "n_features": len(cols),
        }

    def run_group_binary(group_name, cols, group_col):
        cols = [c for c in cols if c in model_df.columns]
        if len(cols) == 0:
            return None
        X = model_df[cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        groups = model_df[group_col].values

        probs = np.zeros(len(model_df), dtype=np.float32)
        gkf = GroupKFold(n_splits=min(4, len(np.unique(groups))))
        for tr, te in gkf.split(X, y_bin, groups):
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ])
            pipe.fit(X[tr], y_bin[tr])
            probs[te] = pipe.predict_proba(X[te])[:, 1]

        pred = (probs >= 0.5).astype(int)
        return {
            "task": "closure_like_binary",
            "feature_group": group_name,
            "heldout_group": group_col,
            "auc": safe_auc(y_bin, probs),
            "accuracy": float(accuracy_score(y_bin, pred)),
            "f1": float(f1_score(y_bin, pred, zero_division=0)),
            "n_features": len(cols),
        }

    for group_name, cols in feature_groups.items():
        for holdout in ["entity_id", "surface", "relation_id"]:
            rr = run_group_regression(group_name, cols, holdout)
            if rr is not None:
                pred_rows.append(rr)
            br = run_group_binary(group_name, cols, holdout)
            if br is not None:
                pred_rows.append(br)

    pred_df = pd.DataFrame(pred_rows)

    # Save
    model_df.to_csv(SAVE_DIR / f"{model_key}_prompt_init_dataset.csv", index=False, encoding="utf-8-sig")
    factor_df.to_csv(SAVE_DIR / f"{model_key}_factor_variance.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(SAVE_DIR / f"{model_key}_prediction_summary.csv", index=False, encoding="utf-8-sig")

    # Summary
    # Pick strongest results
    best_delta_surface = pred_df[
        (pred_df["task"] == "DeltaU_regression") &
        (pred_df["heldout_group"] == "surface")
    ].sort_values("r2", ascending=False).head(1)

    best_closure_surface = pred_df[
        (pred_df["task"] == "closure_like_binary") &
        (pred_df["heldout_group"] == "surface")
    ].sort_values("auc", ascending=False).head(1)

    # factor structure/surface ratio for delta_topk_all
    fv = factor_df[factor_df["feature_group"] == "delta_topk_all"]
    eta_structure = float(fv[fv["factor"] == "structure"]["eta2"].iloc[0]) if len(fv[fv["factor"] == "structure"]) else np.nan
    eta_surface = float(fv[fv["factor"] == "surface"]["eta2"].iloc[0]) if len(fv[fv["factor"] == "surface"]) else np.nan
    structure_surface_ratio = eta_structure / (eta_surface + 1e-12)

    summary = {
        "model_key": model_key,
        "model_path": spec["path"],
        "num_layers": num_layers,
        "init_layers": init_layers,
        "decision_layers": decision_layers,
        "pc_variance_DeltaR_decision": pca.explained_variance_ratio_.tolist(),
        "pc1_variance_DeltaR_decision": float(pca.explained_variance_ratio_[0]),
        "delta_topk_eta2_structure": eta_structure,
        "delta_topk_eta2_surface": eta_surface,
        "delta_topk_structure_surface_ratio": float(structure_surface_ratio),
        "best_surface_heldout_DeltaU": best_delta_surface.to_dict(orient="records"),
        "best_surface_heldout_closure": best_closure_surface.to_dict(orient="records"),
        "pass_structure_gt_surface": bool(eta_structure > eta_surface),
        "pass_structure_ratio_gt_3": bool(structure_surface_ratio > 3.0),
        "pass_surface_heldout_deltaU_corr_0p3": bool(
            len(best_delta_surface) > 0 and float(best_delta_surface["corr"].iloc[0]) > 0.3
        ),
        "pass_surface_heldout_closure_auc_0p65": bool(
            len(best_closure_surface) > 0 and float(best_closure_surface["auc"].iloc[0]) > 0.65
        ),
    }

    with open(SAVE_DIR / f"{model_key}_prompt_init_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # cleanup
    del model, tokenizer, W, W_norm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    all_summaries = []

    for model_key, spec in MODEL_SPECS.items():
        summary = extract_model_dataset(model_key, spec)
        all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(SAVE_DIR / "cm4_model_summary.csv", index=False, encoding="utf-8-sig")

    overall = {
        "audit": "CM-4 Prompt Initialization Operator Audit",
        "theory": "Prompt -> (x0, Sigma0, C0); shallow TopK delta should expose structure more than surface.",
        "models": all_summaries,
        "pass_all_structure_gt_surface": all(s["pass_structure_gt_surface"] for s in all_summaries),
        "pass_all_structure_ratio_gt_3": all(s["pass_structure_ratio_gt_3"] for s in all_summaries),
        "pass_all_surface_deltaU_corr_0p3": all(s["pass_surface_heldout_deltaU_corr_0p3"] for s in all_summaries),
        "pass_all_surface_closure_auc_0p65": all(s["pass_surface_heldout_closure_auc_0p65"] for s in all_summaries),
    }

    with open(SAVE_DIR / "cm4_overall_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    print("\nCM-4 complete.")
    print(summary_df)


if __name__ == "__main__":
    main()
