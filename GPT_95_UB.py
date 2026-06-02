# ============================================================
# UA-2B: Latent Potential Reconstruction Audit
#
# Goal:
#   Infer latent potential coordinate U_hat from observable dynamics,
#   rather than hand-defining U.
#
# Core test:
#   Observables from L7-L19
#       -> predict L20-L22 critical state
#       -> predict L23-L26 basin state
#       -> predict final answer phase
#
# If a low-dimensional latent U_hat outperforms raw single metrics
# and remains interpretable, it becomes a strong U candidate.
# ============================================================

import os
import gc
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

SAVE_DIR = Path("./ua2b_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SEED = 42
MAX_LEN = 280

N_GRAPHS = 96

TRACK_LAYERS = list(range(0, 28))

MID_LAYERS = list(range(7, 20))
CRIT_LAYERS = list(range(20, 23))
BASIN_LAYERS = list(range(23, 27))
FINAL_LAYER = 27

# Candidate answer labels. Prefer single-token labels.
LABEL_POOL = [
    "Red", "Blue", "Green", "Yellow",
    "North", "South", "East", "West",
    "Alpha", "Beta", "Gamma", "Delta",
    "Circle", "Square", "Triangle", "Star",
]

CONDITIONS = [
    "stable_positive",
    "weak_positive",
    "true_compete_balanced",
    "true_compete_order_C_first",
    "true_compete_order_E_first",
    "direct_negative",
    "update_negative",
    "exception_negative",
]

PHASE_MAP = {
    "stable_positive": "positive",
    "weak_positive": "positive",
    "true_compete_balanced": "critical",
    "true_compete_order_C_first": "critical",
    "true_compete_order_E_first": "critical",
    "direct_negative": "negative",
    "update_negative": "negative",
    "exception_negative": "negative",
}

PHASE_ID = {"positive": 0, "critical": 1, "negative": 2}

# ============================================================
# SEED
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ============================================================
# LOAD MODEL
# ============================================================

print("[LOAD] Loading model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    device_map="auto" if DEVICE == "cuda" else None,
    trust_remote_code=True,
    local_files_only=True,
)

if DEVICE == "cpu":
    model.to(DEVICE)

model.eval()

num_layers = len(model.model.layers)
print("[LOAD] num_layers =", num_layers)

if max(TRACK_LAYERS) >= num_layers:
    raise ValueError("TRACK_LAYERS exceeds model depth.")

# ============================================================
# TOKEN HELPERS
# ============================================================

def continuation_ids(text):
    return tokenizer(" " + text, add_special_tokens=False)["input_ids"]

def tokenization_audit(labels):
    print("\n[LABEL TOKENIZATION]")
    usable = []
    for lab in labels:
        ids = continuation_ids(lab)
        print(f"{lab:<12} ids={ids} len={len(ids)}")
        if len(ids) >= 1:
            usable.append(lab)
    return usable

LABEL_POOL = tokenization_audit(LABEL_POOL)

def label_score(logits_vec, label):
    ids = continuation_ids(label)
    vals = []
    for tid in ids:
        if tid < logits_vec.shape[-1]:
            vals.append(float(logits_vec[tid]))
    if not vals:
        return -1e9
    return float(np.mean(vals))

def entropy_binary(logit_c, logit_e):
    z = np.array([logit_c, logit_e], dtype=np.float64)
    z = z - np.max(z)
    p = np.exp(z) / np.sum(np.exp(z))
    return float(-np.sum(p * np.log(p + 1e-12)))

# ============================================================
# DATASET
# ============================================================

def make_entity(prefix, i):
    return f"{prefix}{i:03d}"

def make_prompt(a, b, clean_label, conflict_label, condition):
    option_line = f"Possible labels: {clean_label} or {conflict_label}."

    if condition == "stable_positive":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} maps to {clean_label}.",
            f"Question: Which label does {a} map to?",
            "Answer with exactly one label."
        ]

    elif condition == "weak_positive":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} usually maps to {clean_label}.",
            f"Note: some unrelated cases may map to {conflict_label}, but not this one.",
            f"Question: Which label does {a} map to?",
            "Answer with exactly one label."
        ]

    elif condition == "true_compete_balanced":
        lines = [
            "You are given a small relation graph with competing evidence.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} maps to {clean_label}.",
            f"Fact 3: a parallel source says {a} maps to {conflict_label}.",
            f"Question: Which label does {a} map to?",
            "Answer with exactly one label."
        ]

    elif condition == "true_compete_order_C_first":
        lines = [
            "You are given a small relation graph with competing evidence.",
            option_line,
            f"Fact 1: {a} maps to {clean_label}.",
            f"Fact 2: {a} belongs to {b}.",
            f"Fact 3: {b} is also associated with {conflict_label}.",
            f"Question: Which label does {a} map to?",
            "Answer with exactly one label."
        ]

    elif condition == "true_compete_order_E_first":
        lines = [
            "You are given a small relation graph with competing evidence.",
            option_line,
            f"Fact 1: {a} is associated with {conflict_label}.",
            f"Fact 2: {a} belongs to {b}.",
            f"Fact 3: {b} maps to {clean_label}.",
            f"Question: Which label does {a} map to?",
            "Answer with exactly one label."
        ]

    elif condition == "direct_negative":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} maps to {clean_label}.",
            f"Fact 3: Direct rule: {a} maps to {conflict_label}.",
            f"Question: Which label does {a} map to?",
            "Answer with exactly one label."
        ]

    elif condition == "update_negative":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Old fact: {a} belonged to {b}, and {b} mapped to {clean_label}.",
            f"Update: {a} has been reassigned to {conflict_label}.",
            f"Question: Which label does {a} map to now?",
            "Answer with exactly one label."
        ]

    elif condition == "exception_negative":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"General rule: items in {b} map to {clean_label}.",
            f"Exception: {a} specifically maps to {conflict_label}.",
            f"Question: Which label does {a} map to?",
            "Answer with exactly one label."
        ]

    else:
        raise ValueError(condition)

    return "\n".join(lines)

def build_dataset(n_graphs):
    rows = []
    label_pairs = []
    labs = LABEL_POOL[:12]
    for i in range(0, len(labs) - 1, 2):
        label_pairs.append((labs[i], labs[i + 1]))

    for gid in range(n_graphs):
        a = make_entity("A", gid)
        b = make_entity("B", gid)
        clean_label, conflict_label = label_pairs[gid % len(label_pairs)]

        for cond in CONDITIONS:
            prompt = make_prompt(a, b, clean_label, conflict_label, cond)
            rows.append({
                "graph_id": gid,
                "condition": cond,
                "phase": PHASE_MAP[cond],
                "phase_id": PHASE_ID[PHASE_MAP[cond]],
                "a": a,
                "b": b,
                "clean_label": clean_label,
                "conflict_label": conflict_label,
                "prompt": prompt,
            })

    return pd.DataFrame(rows)

df_tasks = build_dataset(N_GRAPHS)

# ============================================================
# MODEL EXTRACTION
# ============================================================

@torch.no_grad()
def extract_layer_logits(prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
    ).to(model.device)

    out = model(
        **inputs,
        output_hidden_states=True,
        use_cache=False,
    )

    hidden_states = out.hidden_states
    layer_logits = {}

    for l in TRACK_LAYERS:
        h = hidden_states[l + 1][0, -1, :]
        logits = model.lm_head(h).detach().float().cpu().numpy()
        layer_logits[l] = logits

    del out, inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return layer_logits

def extract_observables(row):
    layer_logits = extract_layer_logits(row["prompt"])
    clean = row["clean_label"]
    conflict = row["conflict_label"]

    out = {
        "graph_id": row["graph_id"],
        "condition": row["condition"],
        "phase": row["phase"],
        "phase_id": row["phase_id"],
        "clean_label": clean,
        "conflict_label": conflict,
    }

    R_vals = []

    for l in TRACK_LAYERS:
        logits = layer_logits[l]
        sc = label_score(logits, clean)
        se = label_score(logits, conflict)
        R = sc - se
        H2 = entropy_binary(sc, se)

        out[f"R_{l}"] = R
        out[f"H2_{l}"] = H2
        out[f"absR_{l}"] = abs(R)

        R_vals.append(R)

    R_vals = np.array(R_vals, dtype=np.float64)

    # Derivatives over layer
    dR = np.diff(R_vals)
    ddR = np.diff(dR)

    for i, l in enumerate(TRACK_LAYERS[:-1]):
        out[f"dR_{l}_{l+1}"] = dR[i]

    for i, l in enumerate(TRACK_LAYERS[:-2]):
        out[f"ddR_{l}_{l+2}"] = ddR[i]

    return out

# ============================================================
# RUN EXTRACTION
# ============================================================

records = []

print("\n[RUN] Extracting observables...")

for idx, row in df_tasks.iterrows():
    print(f"[{idx+1}/{len(df_tasks)}] graph={row.graph_id} cond={row.condition}")
    rec = extract_observables(row)
    records.append(rec)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

df = pd.DataFrame(records)

# ============================================================
# CLEAN-RELATIVE DELTA FEATURES
# ============================================================

print("\n[FEATURE] Building clean-relative deltas...")

clean_rows = (
    df[df["condition"] == "stable_positive"]
    .set_index("graph_id")
)

for l in TRACK_LAYERS:
    base = clean_rows[f"R_{l}"].to_dict()
    df[f"dltR_{l}"] = df.apply(lambda r: r[f"R_{l}"] - base[r["graph_id"]], axis=1)

for l in TRACK_LAYERS[:-1]:
    col = f"dR_{l}_{l+1}"
    base = clean_rows[col].to_dict()
    df[f"dlt_{col}"] = df.apply(lambda r: r[col] - base[r["graph_id"]], axis=1)

# ============================================================
# TARGETS
# ============================================================

def mean_cols(row, prefix, layers):
    return float(np.mean([row[f"{prefix}_{l}"] for l in layers]))

df["target_crit_R_mean"] = df.apply(lambda r: mean_cols(r, "R", CRIT_LAYERS), axis=1)
df["target_basin_R_mean"] = df.apply(lambda r: mean_cols(r, "R", BASIN_LAYERS), axis=1)
df["target_final_R"] = df[f"R_{FINAL_LAYER}"]

df["target_crit_dltR_mean"] = df.apply(lambda r: mean_cols(r, "dltR", CRIT_LAYERS), axis=1)
df["target_basin_dltR_mean"] = df.apply(lambda r: mean_cols(r, "dltR", BASIN_LAYERS), axis=1)
df["target_final_dltR"] = df[f"dltR_{FINAL_LAYER}"]

df["gen_negative_proxy"] = (df["target_final_R"] < 0).astype(int)

# ============================================================
# FEATURE SETS
# ============================================================

def cols_R(layers):
    return [f"R_{l}" for l in layers]

def cols_dltR(layers):
    return [f"dltR_{l}" for l in layers]

def cols_H(layers):
    return [f"H2_{l}" for l in layers]

def cols_absR(layers):
    return [f"absR_{l}" for l in layers]

def cols_dR(layers):
    return [f"dR_{l}_{l+1}" for l in layers if l < max(TRACK_LAYERS)]

def cols_dltdR(layers):
    return [f"dlt_dR_{l}_{l+1}" for l in layers if l < max(TRACK_LAYERS)]

FEATURE_SETS = {
    "mid_R_raw": cols_R(MID_LAYERS),
    "mid_dltR_raw": cols_dltR(MID_LAYERS),
    "mid_R_H_abs": cols_R(MID_LAYERS) + cols_H(MID_LAYERS) + cols_absR(MID_LAYERS),
    "mid_dynamics": cols_R(MID_LAYERS) + cols_dR(MID_LAYERS[:-1]),
    "mid_dlt_dynamics": cols_dltR(MID_LAYERS) + cols_dltdR(MID_LAYERS[:-1]),
    "mid_all_observed": (
        cols_R(MID_LAYERS)
        + cols_dltR(MID_LAYERS)
        + cols_H(MID_LAYERS)
        + cols_absR(MID_LAYERS)
        + cols_dR(MID_LAYERS[:-1])
        + cols_dltdR(MID_LAYERS[:-1])
    ),
}

REG_TARGETS = [
    "target_crit_R_mean",
    "target_basin_R_mean",
    "target_final_R",
    "target_crit_dltR_mean",
    "target_basin_dltR_mean",
    "target_final_dltR",
]

CLS_TARGETS = [
    "phase_id",
    "gen_negative_proxy",
]

# ============================================================
# LATENT U MODELS
# ============================================================

def latent_transform_train_test(X_train, X_test, y_train=None, method="pca", n_components=1):
    """
    Fit latent projection using train only.
    Returns U_train, U_test.
    """
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    if method == "pca":
        model_lat = PCA(n_components=n_components, random_state=SEED)
        Utr = model_lat.fit_transform(Xtr)
        Ute = model_lat.transform(Xte)

    elif method == "fa":
        model_lat = FactorAnalysis(n_components=n_components, random_state=SEED)
        Utr = model_lat.fit_transform(Xtr)
        Ute = model_lat.transform(Xte)

    elif method == "pls":
        if y_train is None:
            raise ValueError("PLS requires y_train")
        model_lat = PLSRegression(n_components=n_components)
        Utr = model_lat.fit_transform(Xtr, y_train)[0]
        Ute = model_lat.transform(Xte)

    else:
        raise ValueError(method)

    return Utr, Ute

def evaluate_regression_latent(df, feature_cols, target_col, method, n_components):
    groups = df["graph_id"].values
    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)

    gkf = GroupKFold(n_splits=6)

    preds = np.zeros(len(df), dtype=np.float64)

    for tr, te in gkf.split(X, y, groups):
        Utr, Ute = latent_transform_train_test(
            X[tr], X[te],
            y_train=y[tr],
            method=method,
            n_components=n_components,
        )

        reg = Ridge(alpha=1.0)
        reg.fit(Utr, y[tr])
        preds[te] = reg.predict(Ute)

    r2 = r2_score(y, preds)
    corr = float(np.corrcoef(y, preds)[0, 1]) if np.std(preds) > 1e-8 else 0.0

    return r2, corr

def evaluate_regression_raw(df, feature_cols, target_col):
    groups = df["graph_id"].values
    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)

    gkf = GroupKFold(n_splits=6)

    preds = np.zeros(len(df), dtype=np.float64)

    for tr, te in gkf.split(X, y, groups):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ])
        pipe.fit(X[tr], y[tr])
        preds[te] = pipe.predict(X[te])

    r2 = r2_score(y, preds)
    corr = float(np.corrcoef(y, preds)[0, 1]) if np.std(preds) > 1e-8 else 0.0

    return r2, corr

def evaluate_classification_latent(df, feature_cols, target_col, method, n_components):
    groups = df["graph_id"].values
    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values.astype(int)

    gkf = GroupKFold(n_splits=6)

    preds = np.zeros(len(df), dtype=int)
    probs = None

    binary = len(np.unique(y)) == 2
    if binary:
        probs = np.zeros(len(df), dtype=np.float64)

    for tr, te in gkf.split(X, y, groups):
        Utr, Ute = latent_transform_train_test(
            X[tr], X[te],
            y_train=y[tr],
            method=method,
            n_components=n_components,
        )

        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(Utr, y[tr])
        preds[te] = clf.predict(Ute)

        if binary:
            probs[te] = clf.predict_proba(Ute)[:, 1]

    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="macro")

    auc = np.nan
    if binary:
        try:
            auc = roc_auc_score(y, probs)
        except Exception:
            auc = np.nan

    return acc, f1, auc

def evaluate_classification_raw(df, feature_cols, target_col):
    groups = df["graph_id"].values
    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values.astype(int)

    gkf = GroupKFold(n_splits=6)

    preds = np.zeros(len(df), dtype=int)
    binary = len(np.unique(y)) == 2
    probs = np.zeros(len(df), dtype=np.float64) if binary else None

    for tr, te in gkf.split(X, y, groups):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])
        pipe.fit(X[tr], y[tr])
        preds[te] = pipe.predict(X[te])

        if binary:
            probs[te] = pipe.predict_proba(X[te])[:, 1]

    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="macro")

    auc = np.nan
    if binary:
        try:
            auc = roc_auc_score(y, probs)
        except Exception:
            auc = np.nan

    return acc, f1, auc

# ============================================================
# EVALUATION
# ============================================================

print("\n[EVAL] Evaluating raw vs latent U...")

reg_rows = []
cls_rows = []

LATENT_METHODS = ["pca", "fa", "pls"]
N_COMPONENTS_LIST = [1, 2, 3]

for fs_name, fs_cols in FEATURE_SETS.items():
    fs_cols = [c for c in fs_cols if c in df.columns]

    for target in REG_TARGETS:
        r2, corr = evaluate_regression_raw(df, fs_cols, target)
        reg_rows.append({
            "feature_set": fs_name,
            "target": target,
            "model": "raw_ridge",
            "latent_method": "none",
            "n_components": len(fs_cols),
            "r2": r2,
            "corr": corr,
        })

        for method in LATENT_METHODS:
            for nc in N_COMPONENTS_LIST:
                try:
                    r2, corr = evaluate_regression_latent(df, fs_cols, target, method, nc)
                    reg_rows.append({
                        "feature_set": fs_name,
                        "target": target,
                        "model": "latent_ridge",
                        "latent_method": method,
                        "n_components": nc,
                        "r2": r2,
                        "corr": corr,
                    })
                except Exception as e:
                    reg_rows.append({
                        "feature_set": fs_name,
                        "target": target,
                        "model": "latent_ridge",
                        "latent_method": method,
                        "n_components": nc,
                        "r2": np.nan,
                        "corr": np.nan,
                        "error": str(e),
                    })

    for target in CLS_TARGETS:
        acc, f1, auc = evaluate_classification_raw(df, fs_cols, target)
        cls_rows.append({
            "feature_set": fs_name,
            "target": target,
            "model": "raw_logreg",
            "latent_method": "none",
            "n_components": len(fs_cols),
            "acc": acc,
            "macro_f1": f1,
            "auc": auc,
        })

        for method in LATENT_METHODS:
            for nc in N_COMPONENTS_LIST:
                try:
                    acc, f1, auc = evaluate_classification_latent(df, fs_cols, target, method, nc)
                    cls_rows.append({
                        "feature_set": fs_name,
                        "target": target,
                        "model": "latent_logreg",
                        "latent_method": method,
                        "n_components": nc,
                        "acc": acc,
                        "macro_f1": f1,
                        "auc": auc,
                    })
                except Exception as e:
                    cls_rows.append({
                        "feature_set": fs_name,
                        "target": target,
                        "model": "latent_logreg",
                        "latent_method": method,
                        "n_components": nc,
                        "acc": np.nan,
                        "macro_f1": np.nan,
                        "auc": np.nan,
                        "error": str(e),
                    })

df_reg = pd.DataFrame(reg_rows)
df_cls = pd.DataFrame(cls_rows)

# ============================================================
# EXTRACT BEST LATENT U FOR INSPECTION
# ============================================================

print("\n[U] Extracting best U_hat coordinates...")

primary_features = FEATURE_SETS["mid_all_observed"]
primary_features = [c for c in primary_features if c in df.columns]

primary_target = "target_crit_dltR_mean"

X = df[primary_features].values.astype(np.float64)
y = df[primary_target].values.astype(np.float64)

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

pls = PLSRegression(n_components=3)
U_hat = pls.fit_transform(Xs, y)[0]

for i in range(U_hat.shape[1]):
    df[f"Uhat_{i+1}"] = U_hat[:, i]

u_summary = (
    df.groupby(["condition", "phase"])
    .agg(
        n=("Uhat_1", "count"),
        U1_mean=("Uhat_1", "mean"),
        U1_std=("Uhat_1", "std"),
        U2_mean=("Uhat_2", "mean"),
        U2_std=("Uhat_2", "std"),
        U3_mean=("Uhat_3", "mean"),
        U3_std=("Uhat_3", "std"),
        crit_dltR=("target_crit_dltR_mean", "mean"),
        basin_dltR=("target_basin_dltR_mean", "mean"),
        final_R=("target_final_R", "mean"),
    )
    .reset_index()
)

# Loading / feature contribution
loadings = pd.DataFrame({
    "feature": primary_features,
    "pls_coef": pls.coef_.ravel()[:len(primary_features)] if pls.coef_.size >= len(primary_features) else np.nan,
})

# ============================================================
# SAVE
# ============================================================

df.to_csv(SAVE_DIR / "ua2b_dataset_with_Uhat.csv", index=False, encoding="utf-8-sig")
df_reg.to_csv(SAVE_DIR / "ua2b_regression_summary.csv", index=False, encoding="utf-8-sig")
df_cls.to_csv(SAVE_DIR / "ua2b_classification_summary.csv", index=False, encoding="utf-8-sig")
u_summary.to_csv(SAVE_DIR / "ua2b_Uhat_condition_summary.csv", index=False, encoding="utf-8-sig")
loadings.to_csv(SAVE_DIR / "ua2b_Uhat_feature_loadings.csv", index=False, encoding="utf-8-sig")

config = {
    "model_path": MODEL_PATH,
    "n_graphs": N_GRAPHS,
    "conditions": CONDITIONS,
    "phase_map": PHASE_MAP,
    "mid_layers": MID_LAYERS,
    "critical_layers": CRIT_LAYERS,
    "basin_layers": BASIN_LAYERS,
    "final_layer": FINAL_LAYER,
    "primary_latent_target": primary_target,
    "interpretation": {
        "PASS_Strong": "Low-dimensional Uhat from L7-L19 predicts L20-L22 and L23-L26 better than raw single metrics and separates phases.",
        "PASS_Lite": "Uhat predicts critical/basin states but does not clearly outperform raw deltaR.",
        "FAIL": "Uhat adds no signal beyond raw observables."
    }
}

with open(SAVE_DIR / "ua2b_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n========== UA-2B REGRESSION TOP ==========")
print(
    df_reg.sort_values(["target", "r2"], ascending=[True, False])
    .groupby("target")
    .head(8)
    .to_string(index=False)
)

print("\n========== UA-2B CLASSIFICATION TOP ==========")
print(
    df_cls.sort_values(["target", "macro_f1"], ascending=[True, False])
    .groupby("target")
    .head(8)
    .to_string(index=False)
)

print("\n========== Uhat CONDITION SUMMARY ==========")
print(u_summary.to_string(index=False))

print("\n[DONE] Outputs saved to:", SAVE_DIR)