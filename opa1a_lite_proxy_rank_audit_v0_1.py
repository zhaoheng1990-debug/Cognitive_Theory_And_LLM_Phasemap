
# ============================================================
# OPA-1A-lite: Ontology Proxy Rank Audit
#
# Goal:
#   After PSG correction, compare proxy strength on an answer-pair dataset:
#
#       TopK/VIM readback flow
#       PSG nulls: rowpermW / gaussianR / randomK
#       hidden trajectory proxies
#       hidden precursor proxies
#       H_shape / DeltaR decision geometry
#       hidden O-flow
#
#   Target:
#       DeltaU_decision = PC1(clean-relative dR over decision layers)
#       mechanism / phase labels from condition metadata
#
# Core judgment:
#   If TopK realW ~= rowpermW/gaussianR but hidden/Hshape/O-flow dominate,
#   TopK remains PSG-corrected readback observable, while hidden trajectory
#   variables become stronger ontology proxies.
#
# Run:
#   python opa1a_lite_proxy_rank_audit_v0_1.py
#
# Default input:
#   C:\Users\ZH\Desktop\AGI\data\opa1a_lite_answer_pair_dataset_v0_1.csv
#
# Default output:
#   C:\Users\ZH\Desktop\AGI\outputs\opa1a_lite_outputs
#
# ============================================================

import os
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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge, LogisticRegression
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

RUN_MODEL_KEYS = ["qwen"]

PROMPT_CSV = Path(r"C:\Users\ZH\Desktop\AGI\data\opa1a_lite_answer_pair_dataset_v0_1.csv")
SAVE_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\opa1a_lite_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 4
MAX_LEN = 320
TOPK_K = 100
GAUSSIAN_SEED = 123
USE_CHAT_TEMPLATE = False

# Qwen baseline windows. For cross-model later, this script uses fractional
# decision windows if the model layer count differs.
QWEN_PRE_LAYERS = list(range(7, 20))
QWEN_PRECURSOR_LAYERS = [17, 18, 19]
QWEN_DECISION_LAYERS = [20, 21, 22, 23, 24, 25]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

MECH_MAP = {
    "stable": 0,
    "stable_shift": 0,
    "competition": 1,
    "source_ambiguity": 1,
    "closure": 2,
}

PHASE_MAP = {
    "stable": 0,
    "stable_shift": 0,
    "competition": 1,
    "source_ambiguity": 1,
    "closure": 2,
}

# ============================================================
# UTILS
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

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
    raise RuntimeError("Cannot locate transformer layers.")

def get_lm_head_weight(model):
    if hasattr(model, "lm_head"):
        return model.lm_head.weight.detach()
    raise RuntimeError("Cannot locate lm_head.weight.")

def continuation_ids(tokenizer, text):
    return tokenizer(" " + str(text), add_special_tokens=False)["input_ids"]

def last_positions(attention_mask):
    return torch.full(
        (attention_mask.shape[0],),
        attention_mask.shape[1] - 1,
        dtype=torch.long,
        device=attention_mask.device,
    )

def map_qwen_layers_to_model(qwen_layers, num_layers):
    # qwen has 28 layers indexed 0..27. Map by fraction.
    mapped = []
    for l in qwen_layers:
        frac = l / 27.0
        ml = int(round(frac * (num_layers - 1)))
        ml = max(0, min(num_layers - 1, ml))
        mapped.append(ml)
    return sorted(set(mapped))

def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def safe_auc_binary(y, score):
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y, score))
    except Exception:
        return np.nan

def ridge_group_cv(X, y, groups, n_splits=5):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[1] == 0 or len(np.unique(groups)) < 3:
        return np.nan, np.nan
    pred = np.zeros_like(y, dtype=float)
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    for tr, te in gkf.split(X, y, groups):
        reg = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0)),
        ])
        reg.fit(X[tr], y[tr])
        pred[te] = reg.predict(X[te])
    return float(r2_score(y, pred)), safe_corr(y, pred)

def logistic_group_cv(X, y, groups, n_splits=5):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[1] == 0 or len(np.unique(y)) < 2 or len(np.unique(groups)) < 3:
        return np.nan, np.nan
    pred = np.zeros_like(y, dtype=int)
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    for tr, te in gkf.split(X, y, groups):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    return float(accuracy_score(y, pred)), float(f1_score(y, pred, average="macro", zero_division=0))

def pairwise_cosine_distance_matrix(X):
    X = np.asarray(X, dtype=np.float32)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    sim = Xn @ Xn.T
    return 1.0 - sim

def upper_tri_vector(M):
    idx = np.triu_indices(M.shape[0], k=1)
    return M[idx]

def row_cos(a, b):
    return np.sum(a * b, axis=1) / ((np.linalg.norm(a, axis=1) + 1e-8) * (np.linalg.norm(b, axis=1) + 1e-8))

def topk_centers_from_matrix(H, R, k=100):
    # H: [n,d], R: [v,d], returns [n,d] normalized centers.
    Ht = torch.tensor(H, dtype=torch.float32)
    Rt = torch.tensor(R, dtype=torch.float32)
    Rt_norm = torch.nn.functional.normalize(Rt, dim=1)
    out = []
    bs = 32
    with torch.no_grad():
        for s in range(0, Ht.shape[0], bs):
            h = Ht[s:s+bs]
            logits = h @ Rt.T
            _, ids = torch.topk(logits, k=k, dim=1)
            emb = Rt_norm[ids]
            center = emb.mean(dim=1)
            center = torch.nn.functional.normalize(center, dim=1)
            out.append(center.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)

def randomk_centers(R, n, k=100, seed=42):
    rng = np.random.default_rng(seed)
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-8)
    centers = []
    for _ in range(n):
        ids = rng.choice(R.shape[0], size=k, replace=False)
        c = Rn[ids].mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-8)
        centers.append(c.astype(np.float32))
    return np.vstack(centers)

def make_topk_flow_features(centers_by_layer, layers):
    feats = []
    for l0, l1 in zip(layers[:-1], layers[1:]):
        C0 = centers_by_layer[l0]
        C1 = centers_by_layer[l1]
        shift = 1.0 - row_cos(C0, C1)
        feats.append(shift.reshape(-1, 1))
    if not feats:
        return np.zeros((next(iter(centers_by_layer.values())).shape[0], 0), dtype=np.float32)
    X = np.concatenate(feats, axis=1)
    return np.column_stack([
        X,
        X.mean(axis=1),
        X.sum(axis=1),
        X.max(axis=1),
        X.std(axis=1),
    ]).astype(np.float32)

def make_hidden_oflow(H, layers):
    feats = []
    for l0, l1 in zip(layers[:-1], layers[1:]):
        A = H[:, l0, :]
        B = H[:, l1, :]
        shift = 1.0 - row_cos(A, B)
        delta_norm = np.linalg.norm(B - A, axis=1)
        feats.append(np.column_stack([shift, delta_norm]))
    if not feats:
        return np.zeros((H.shape[0], 0), dtype=np.float32)
    X = np.concatenate(feats, axis=1)
    return np.column_stack([
        X,
        X.mean(axis=1),
        X.sum(axis=1),
        X.max(axis=1),
        X.std(axis=1),
    ]).astype(np.float32)

def proxy_eval_rows(model_key, proxy_name, X, deltaU, phase_y, mechanism_y, groups):
    r2, corr = ridge_group_cv(X, deltaU, groups)
    phase_acc, phase_f1 = logistic_group_cv(X, phase_y, groups)
    mech_acc, mech_f1 = logistic_group_cv(X, mechanism_y, groups)
    return {
        "model_key": model_key,
        "proxy": proxy_name,
        "n_samples": int(len(deltaU)),
        "n_features": int(np.asarray(X).shape[1] if np.asarray(X).ndim > 1 else 1),
        "deltaU_r2": r2,
        "deltaU_corr": corr,
        "phase_acc": phase_acc,
        "phase_macro_f1": phase_f1,
        "mechanism_acc": mech_acc,
        "mechanism_macro_f1": mech_f1,
    }

# ============================================================
# DATA
# ============================================================

def load_dataset(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = ["prompt_id", "prompt", "clean_answer", "conflict_answer", "group_id", "mechanism"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    df = df.copy()
    df["prompt"] = df["prompt"].astype(str)
    df["clean_answer"] = df["clean_answer"].astype(str)
    df["conflict_answer"] = df["conflict_answer"].astype(str)
    df["group_id"] = df["group_id"].astype(int)
    df["mechanism"] = df["mechanism"].astype(str)

    bad = [m for m in sorted(df["mechanism"].unique()) if m not in MECH_MAP]
    if bad:
        raise RuntimeError(f"Unknown mechanism labels: {bad}. Update MECH_MAP.")

    df["mechanism_y"] = df["mechanism"].map(MECH_MAP).astype(int)
    df["phase_y"] = df["mechanism"].map(PHASE_MAP).astype(int)
    return df

# ============================================================
# EXTRACTION
# ============================================================

def extract_model_arrays(model_key, model_path, df):
    print(f"\n========== {model_key} ==========")
    tokenizer = load_tokenizer(model_path)
    model = load_model(model_path)
    num_layers = get_num_layers(model)
    W = get_lm_head_weight(model).detach().float().to(model.device)
    W_cpu = W.cpu().numpy().astype(np.float32)

    pre_layers = map_qwen_layers_to_model(QWEN_PRE_LAYERS, num_layers)
    precursor_layers = map_qwen_layers_to_model(QWEN_PRECURSOR_LAYERS, num_layers)
    decision_layers = map_qwen_layers_to_model(QWEN_DECISION_LAYERS, num_layers)
    obs_layers = sorted(set(pre_layers + precursor_layers + decision_layers))

    print("num_layers:", num_layers)
    print("pre_layers:", pre_layers)
    print("precursor_layers:", precursor_layers)
    print("decision_layers:", decision_layers)

    n = len(df)
    d_model = W.shape[1]
    H = np.zeros((n, num_layers, d_model), dtype=np.float32)
    R = np.zeros((n, num_layers), dtype=np.float32)

    clean_ids = []
    conflict_ids = []
    token_rows = []

    for _, row in df.iterrows():
        cids = continuation_ids(tokenizer, row["clean_answer"])
        eids = continuation_ids(tokenizer, row["conflict_answer"])
        token_rows.append({
            "prompt_id": row["prompt_id"],
            "clean_answer": row["clean_answer"],
            "conflict_answer": row["conflict_answer"],
            "clean_ids": str(cids),
            "conflict_ids": str(eids),
            "clean_len": len(cids),
            "conflict_len": len(eids),
            "both_single": int(len(cids) == 1 and len(eids) == 1),
        })
        if len(cids) != 1 or len(eids) != 1:
            raise RuntimeError(
                f"Non-single answer token in {model_key}: "
                f"{row['clean_answer']}={cids}, {row['conflict_answer']}={eids}. "
                "Use labels that are single continuation tokens for all models."
            )
        clean_ids.append(cids[0])
        conflict_ids.append(eids[0])

    pd.DataFrame(token_rows).to_csv(SAVE_DIR / f"{model_key}_tokenization_audit.csv", index=False, encoding="utf-8-sig")

    prompts = df["prompt"].tolist()
    if USE_CHAT_TEMPLATE and hasattr(tokenizer, "apply_chat_template"):
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            end = min(n, start + BATCH_SIZE)
            batch_text = prompts[start:end]
            inputs = tokenizer(
                batch_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
            ).to(model.device)

            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            hstates = outputs.hidden_states
            pos = last_positions(inputs["attention_mask"])
            bsz = end - start

            cids_t = torch.tensor(clean_ids[start:end], dtype=torch.long, device=model.device)
            eids_t = torch.tensor(conflict_ids[start:end], dtype=torch.long, device=model.device)

            for l in range(num_layers):
                h = hstates[l + 1][torch.arange(bsz, device=model.device), pos, :].detach().float()
                H[start:end, l, :] = h.cpu().numpy().astype(np.float32)

                rc = torch.sum(h * W[cids_t].float(), dim=1)
                re = torch.sum(h * W[eids_t].float(), dim=1)
                R[start:end, l] = (rc - re).cpu().numpy().astype(np.float32)

            print(f"processed {end}/{n}")

            del outputs, hstates, inputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    del model, tokenizer, W
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_key": model_key,
        "num_layers": num_layers,
        "pre_layers": pre_layers,
        "precursor_layers": precursor_layers,
        "decision_layers": decision_layers,
        "obs_layers": obs_layers,
        "H": H,
        "R": R,
        "W": W_cpu,
    }

# ============================================================
# MODEL AUDIT
# ============================================================

def run_model_audit(model_key, model_path, df):
    data = extract_model_arrays(model_key, model_path, df)
    H = data["H"]
    R = data["R"]
    W = data["W"]
    n, L, d = H.shape

    pre_layers = data["pre_layers"]
    precursor_layers = data["precursor_layers"]
    decision_layers = data["decision_layers"]
    obs_layers = data["obs_layers"]

    groups = df["group_id"].values.astype(int)
    mechanism_y = df["mechanism_y"].values.astype(int)
    phase_y = df["phase_y"].values.astype(int)

    # Clean-relative dR by group. If a group has no clean condition, use group mean baseline.
    dR = np.zeros_like(R)
    for gid, sub in df.groupby("group_id"):
        idx = sub.index.to_numpy()
        clean_idx = sub[sub["condition"].astype(str) == "clean"].index.to_numpy() if "condition" in df.columns else []
        if len(clean_idx) > 0:
            base = R[clean_idx[0]]
        else:
            base = R[idx].mean(axis=0)
        dR[idx] = R[idx] - base

    X_decision_dR = dR[:, decision_layers]
    pca = PCA(n_components=min(3, X_decision_dR.shape[1]))
    pcs = pca.fit_transform(X_decision_dR)
    deltaU = pcs[:, 0]

    # Orient positive toward stable / away from closure if possible.
    stable_mask = df["mechanism"].isin(["stable", "stable_shift"]).values
    closure_mask = df["mechanism"].isin(["closure"]).values
    if stable_mask.sum() > 0 and closure_mask.sum() > 0:
        if np.mean(deltaU[stable_mask]) < np.mean(deltaU[closure_mask]):
            deltaU = -deltaU

    audit_df = df.copy()
    for l in range(L):
        audit_df[f"R_L{l}"] = R[:, l]
        audit_df[f"dR_L{l}"] = dR[:, l]
    audit_df["DeltaU_decision"] = deltaU.astype(np.float32)
    audit_df.to_csv(SAVE_DIR / f"{model_key}_opa1a_lite_R_deltaU_dataset.csv", index=False, encoding="utf-8-sig")

    np.save(SAVE_DIR / f"{model_key}_hidden_last_token_layers.npy", H)
    np.save(SAVE_DIR / f"{model_key}_R_layers.npy", R)
    np.save(SAVE_DIR / f"{model_key}_dR_layers.npy", dR)

    # Hidden proxy features.
    hidden_pre_mean = H[:, pre_layers, :].mean(axis=1)
    hidden_precursor_mean = H[:, precursor_layers, :].mean(axis=1)
    hidden_decision_mean = H[:, decision_layers, :].mean(axis=1)

    Hshape = dR[:, decision_layers].astype(np.float32)
    Rshape = R[:, decision_layers].astype(np.float32)
    hidden_oflow_pre = make_hidden_oflow(H, pre_layers)
    hidden_oflow_precursor = make_hidden_oflow(H, precursor_layers)
    hidden_oflow_decision = make_hidden_oflow(H, decision_layers)

    # PSG / TopK readback flow features.
    rng = np.random.default_rng(GAUSSIAN_SEED)
    rowperm = W.copy()
    rng.shuffle(rowperm, axis=0)

    gaussian = rng.normal(size=W.shape).astype(np.float32)
    # Match approximate row norms to W to avoid trivial scale differences.
    w_norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-8
    g_norms = np.linalg.norm(gaussian, axis=1, keepdims=True) + 1e-8
    gaussian = gaussian / g_norms * np.mean(w_norms)

    centers_real = {}
    centers_rowperm = {}
    centers_gaussian = {}
    centers_randomK = {}

    for l in obs_layers:
        Hl = H[:, l, :]
        centers_real[l] = topk_centers_from_matrix(Hl, W, k=TOPK_K)
        centers_rowperm[l] = topk_centers_from_matrix(Hl, rowperm, k=TOPK_K)
        centers_gaussian[l] = topk_centers_from_matrix(Hl, gaussian, k=TOPK_K)
        centers_randomK[l] = randomk_centers(W, n=n, k=TOPK_K, seed=GAUSSIAN_SEED + l)

    topk_realW_flow = make_topk_flow_features(centers_real, obs_layers)
    topk_rowpermW_flow = make_topk_flow_features(centers_rowperm, obs_layers)
    topk_gaussianR_flow = make_topk_flow_features(centers_gaussian, obs_layers)
    randomK_flow = make_topk_flow_features(centers_randomK, obs_layers)

    # DirectH readback equivalent: use hidden pairwise geometry itself over obs layers.
    directH_flow = make_hidden_oflow(H, obs_layers)

    feature_map = {
        "hidden_pre_mean": hidden_pre_mean,
        "hidden_precursor_mean": hidden_precursor_mean,
        "hidden_decision_mean": hidden_decision_mean,
        "Hshape_dR_decision": Hshape,
        "Rshape_decision": Rshape,
        "hidden_oflow_pre": hidden_oflow_pre,
        "hidden_oflow_precursor": hidden_oflow_precursor,
        "hidden_oflow_decision": hidden_oflow_decision,
        "topk_realW_flow": topk_realW_flow,
        "topk_rowpermW_flow": topk_rowpermW_flow,
        "topk_gaussianR_flow": topk_gaussianR_flow,
        "randomK_flow": randomK_flow,
        "directH_flow": directH_flow,
    }

    for name, X in feature_map.items():
        np.save(SAVE_DIR / f"{model_key}_features_{name}.npy", np.asarray(X, dtype=np.float32))

    # Proxy rank.
    rows = []
    for name, X in feature_map.items():
        rows.append(proxy_eval_rows(model_key, name, X, deltaU, phase_y, mechanism_y, groups))

    rank_df = pd.DataFrame(rows)
    rank_df["ontology_proxy_score"] = (
        rank_df["deltaU_corr"].fillna(0).abs()
        + rank_df["deltaU_r2"].fillna(0).clip(lower=0)
        + rank_df["phase_macro_f1"].fillna(0)
        + rank_df["mechanism_macro_f1"].fillna(0)
    )
    rank_df = rank_df.sort_values("ontology_proxy_score", ascending=False)
    rank_df.to_csv(SAVE_DIR / f"{model_key}_opa1a_lite_proxy_rank.csv", index=False, encoding="utf-8-sig")

    # PSG residual / geometry audit over obs layers.
    geom_rows = []
    for l in obs_layers:
        hidden_vec = upper_tri_vector(pairwise_cosine_distance_matrix(H[:, l, :]))
        for name, centers in [
            ("realW", centers_real[l]),
            ("rowpermW", centers_rowperm[l]),
            ("gaussianR", centers_gaussian[l]),
            ("randomK", centers_randomK[l]),
        ]:
            proxy_vec = upper_tri_vector(pairwise_cosine_distance_matrix(centers))
            geom_rows.append({
                "model_key": model_key,
                "layer": int(l),
                "proxy": name,
                "pearson_hidden_geometry": safe_corr(hidden_vec, proxy_vec),
            })

    geom_df = pd.DataFrame(geom_rows)
    geom_df.to_csv(SAVE_DIR / f"{model_key}_opa1a_lite_psg_geometry.csv", index=False, encoding="utf-8-sig")

    geom_summary = geom_df.groupby("proxy")["pearson_hidden_geometry"].mean().to_dict()
    real_minus_rowperm = geom_summary.get("realW", np.nan) - geom_summary.get("rowpermW", np.nan)
    real_minus_gaussian = geom_summary.get("realW", np.nan) - geom_summary.get("gaussianR", np.nan)
    real_minus_randomK = geom_summary.get("realW", np.nan) - geom_summary.get("randomK", np.nan)

    # Compare proxy buckets.
    best = rank_df.iloc[0].to_dict()
    topk_real = rank_df[rank_df["proxy"] == "topk_realW_flow"].iloc[0].to_dict()
    rowperm_rank = rank_df[rank_df["proxy"] == "topk_rowpermW_flow"].iloc[0].to_dict()
    gaussian_rank = rank_df[rank_df["proxy"] == "topk_gaussianR_flow"].iloc[0].to_dict()
    hshape_rank = rank_df[rank_df["proxy"] == "Hshape_dR_decision"].iloc[0].to_dict()
    precursor_rank = rank_df[rank_df["proxy"] == "hidden_precursor_mean"].iloc[0].to_dict()

    verdict = {
        "psg_residual_supported": bool(real_minus_rowperm > 0.05 and real_minus_gaussian > 0.0),
        "topk_readback_supported": bool(real_minus_randomK > 0.1),
        "hidden_or_hshape_beats_topk": bool(
            max(float(hshape_rank["ontology_proxy_score"]), float(precursor_rank["ontology_proxy_score"]))
            > float(topk_real["ontology_proxy_score"])
        ),
    }

    summary = {
        "model_key": model_key,
        "model_path": model_path,
        "n_prompts": int(n),
        "num_layers": int(L),
        "pre_layers": pre_layers,
        "precursor_layers": precursor_layers,
        "decision_layers": decision_layers,
        "obs_layers": obs_layers,
        "deltaU_pc1_variance": float(pca.explained_variance_ratio_[0]),
        "best_proxy": best,
        "topk_realW_proxy": topk_real,
        "topk_rowpermW_proxy": rowperm_rank,
        "topk_gaussianR_proxy": gaussian_rank,
        "hshape_proxy": hshape_rank,
        "hidden_precursor_proxy": precursor_rank,
        "psg_geometry_mean": geom_summary,
        "real_minus_rowperm_geometry": float(real_minus_rowperm),
        "real_minus_gaussian_geometry": float(real_minus_gaussian),
        "real_minus_randomK_geometry": float(real_minus_randomK),
        "verdict": verdict,
        "interpretation_guardrail": (
            "If realW does not exceed rowpermW/gaussianR, TopK/VIM stays a PSG-corrected "
            "readback observable. If hidden/Hshape/O-flow proxies outperform TopK on DeltaU/phase, "
            "they become stronger trajectory-ontology proxies."
        ),
    }

    with open(SAVE_DIR / f"{model_key}_opa1a_lite_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary, rank_df, geom_df

# ============================================================
# MAIN
# ============================================================

def main():
    print("OPA-1A-lite Proxy Rank Audit")
    print("Input:", PROMPT_CSV)
    print("Output:", SAVE_DIR)
    print("Device:", DEVICE)

    df = load_dataset(PROMPT_CSV)
    df.to_csv(SAVE_DIR / "opa1a_lite_input_used.csv", index=False, encoding="utf-8-sig")
    print("Dataset rows:", len(df))
    print("Groups:", df["group_id"].nunique())
    print("Mechanisms:", df["mechanism"].value_counts().to_dict())

    all_summaries = []
    all_rank = []
    all_geom = []

    for model_key in RUN_MODEL_KEYS:
        if model_key not in MODEL_SPECS:
            raise RuntimeError(f"Unknown model key: {model_key}")
        summary, rank_df, geom_df = run_model_audit(model_key, MODEL_SPECS[model_key], df)
        all_summaries.append(summary)
        all_rank.append(rank_df)
        all_geom.append(geom_df)

    rank_all = pd.concat(all_rank, ignore_index=True)
    geom_all = pd.concat(all_geom, ignore_index=True)

    rank_all.to_csv(SAVE_DIR / "opa1a_lite_proxy_rank_all_models.csv", index=False, encoding="utf-8-sig")
    geom_all.to_csv(SAVE_DIR / "opa1a_lite_psg_geometry_all_models.csv", index=False, encoding="utf-8-sig")

    overall = {
        "audit": "OPA-1A-lite Ontology Proxy Rank Audit",
        "input_csv": str(PROMPT_CSV),
        "save_dir": str(SAVE_DIR),
        "run_model_keys": RUN_MODEL_KEYS,
        "summaries": all_summaries,
    }
    with open(SAVE_DIR / "opa1a_lite_overall_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print("Main outputs:")
    print(SAVE_DIR / "opa1a_lite_proxy_rank_all_models.csv")
    print(SAVE_DIR / "opa1a_lite_psg_geometry_all_models.csv")
    print(SAVE_DIR / "opa1a_lite_overall_summary.json")

if __name__ == "__main__":
    main()
