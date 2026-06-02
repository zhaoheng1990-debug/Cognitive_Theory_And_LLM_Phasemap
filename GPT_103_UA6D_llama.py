# UA-6D: Order Parameter Universality Audit
# Test whether DeltaU = PC1(deltaR_20_25) generalizes across models.

import json
import gc
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = r"D:\model\Llama-3.2-1B-Instruct"
MODEL_NAME = "replace_with_model_name"

SAVE_DIR = Path(f"./ua6d_outputs_{MODEL_NAME}")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SEED = 42
N_GRAPHS = 96
MAX_LEN = 260

# Use relative layer fractions to support different depths.
# For 28-layer Qwen, these correspond roughly to 20-25.
REL_LAYER_START = 0.72
REL_LAYER_END = 0.93

PHASE_ID = {"positive": 0, "critical": 1, "negative": 2}

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

LABEL_POOL = [
    "Red", "Blue", "Green", "Yellow",
    "North", "South", "East", "West",
    "Alpha", "Beta", "Gamma", "Delta",
    "Circle", "Square", "Triangle", "Star",
    "Copper", "Silver", "Gold", "Iron",
    "Apple", "Orange", "Lemon", "Pear",
]

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

print("[LOAD] Loading model:", MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"

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

TRACK_LAYERS = list(range(num_layers))

# Relative version of Qwen L20-L25.
start = int(round((num_layers - 1) * REL_LAYER_START))
end = int(round((num_layers - 1) * REL_LAYER_END))
DELTA_LAYERS = list(range(start, end + 1))

# Keep 6 layers if possible, to match UA-3A / UA-4B.
if len(DELTA_LAYERS) > 6:
    DELTA_LAYERS = DELTA_LAYERS[:6]

print("[LAYERS] DELTA_LAYERS =", DELTA_LAYERS)

# ============================================================
# TOKEN HELPERS
# ============================================================

def continuation_ids(text):
    return tokenizer(" " + text, add_special_tokens=False)["input_ids"]

def usable_labels(labels):
    out = []
    print("\n[LABEL TOKENIZATION]")
    for lab in labels:
        ids = continuation_ids(lab)
        print(f"{lab:<12} ids={ids} len={len(ids)}")
        if len(ids) >= 1:
            out.append(lab)
    return out

LABEL_POOL = usable_labels(LABEL_POOL)

def label_score(logits_vec, label):
    ids = continuation_ids(label)
    vals = []
    for tid in ids:
        if tid < logits_vec.shape[-1]:
            vals.append(float(logits_vec[tid]))
    if not vals:
        return -1e9
    return float(np.mean(vals))

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
            "Answer with exactly one label.",
        ]

    elif condition == "weak_positive":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} usually maps to {clean_label}.",
            f"Note: some unrelated cases may map to {conflict_label}, but not this one.",
            f"Question: Which label does {a} map to?",
            "Answer with exactly one label.",
        ]

    elif condition == "true_compete_balanced":
        lines = [
            "You are given a small relation graph with competing evidence.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} maps to {clean_label}.",
            f"Fact 3: a parallel source says {a} maps to {conflict_label}.",
            f"Question: Which label does {a} map to?",
            "Answer with exactly one label.",
        ]

    elif condition == "true_compete_order_C_first":
        lines = [
            "You are given a small relation graph with competing evidence.",
            option_line,
            f"Fact 1: {a} maps to {clean_label}.",
            f"Fact 2: {a} belongs to {b}.",
            f"Fact 3: {b} is also associated with {conflict_label}.",
            f"Question: Which label does {a} map to?",
            "Answer with exactly one label.",
        ]

    elif condition == "true_compete_order_E_first":
        lines = [
            "You are given a small relation graph with competing evidence.",
            option_line,
            f"Fact 1: {a} is associated with {conflict_label}.",
            f"Fact 2: {a} belongs to {b}.",
            f"Fact 3: {b} maps to {clean_label}.",
            f"Question: Which label does {a} map to?",
            "Answer with exactly one label.",
        ]

    elif condition == "direct_negative":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} maps to {clean_label}.",
            f"Fact 3: Direct rule: {a} maps to {conflict_label}.",
            f"Question: Which label does {a} map to?",
            "Answer with exactly one label.",
        ]

    elif condition == "update_negative":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Old fact: {a} belonged to {b}, and {b} mapped to {clean_label}.",
            f"Update: {a} has been reassigned to {conflict_label}.",
            f"Question: Which label does {a} map to now?",
            "Answer with exactly one label.",
        ]

    elif condition == "exception_negative":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"General rule: items in {b} map to {clean_label}.",
            f"Exception: {a} specifically maps to {conflict_label}.",
            f"Question: Which label does {a} map to?",
            "Answer with exactly one label.",
        ]

    else:
        raise ValueError(condition)

    return "\n".join(lines)

def build_dataset(n_graphs):
    labs = LABEL_POOL[:12]
    pairs = [(labs[i], labs[i+1]) for i in range(0, len(labs)-1, 2)]

    rows = []
    for gid in range(n_graphs):
        a = make_entity("A", gid)
        b = make_entity("B", gid)
        clean_label, conflict_label = pairs[gid % len(pairs)]

        for cond in CONDITIONS:
            rows.append({
                "graph_id": gid,
                "condition": cond,
                "phase": PHASE_MAP[cond],
                "phase_id": PHASE_ID[PHASE_MAP[cond]],
                "clean_label": clean_label,
                "conflict_label": conflict_label,
                "prompt": make_prompt(a, b, clean_label, conflict_label, cond),
            })

    return pd.DataFrame(rows)

df_tasks = build_dataset(N_GRAPHS)

# ============================================================
# EXTRACTION
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

records = []

print("\n[RUN] Extracting observables...")

for idx, row in df_tasks.iterrows():
    print(f"[{idx+1}/{len(df_tasks)}] graph={row.graph_id} cond={row.condition}")

    layer_logits = extract_layer_logits(row["prompt"])
    rec = {
        "graph_id": row["graph_id"],
        "condition": row["condition"],
        "phase": row["phase"],
        "phase_id": row["phase_id"],
        "clean_label": row["clean_label"],
        "conflict_label": row["conflict_label"],
    }

    for l in TRACK_LAYERS:
        logits = layer_logits[l]
        sc = label_score(logits, row["clean_label"])
        se = label_score(logits, row["conflict_label"])
        rec[f"R_{l}"] = sc - se

    records.append(rec)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

df = pd.DataFrame(records)

# ============================================================
# CLEAN-RELATIVE DELTA R
# ============================================================

clean = df[df["condition"] == "stable_positive"].set_index("graph_id")

for l in TRACK_LAYERS:
    base = clean[f"R_{l}"].to_dict()
    df[f"dltR_{l}"] = df.apply(lambda r: r[f"R_{l}"] - base[r["graph_id"]], axis=1)

DELTA_FEATURES = [f"dltR_{l}" for l in DELTA_LAYERS]

# Final proxy: conflict if final R < 0
FINAL_LAYER = max(TRACK_LAYERS)
df["target_final_R"] = df[f"R_{FINAL_LAYER}"]
df["gen_negative_proxy"] = (df["target_final_R"] < 0).astype(int)

# ============================================================
# PCA ORDER PARAMETER
# ============================================================

X_raw = df[DELTA_FEATURES].values.astype(float)
Xs = StandardScaler().fit_transform(X_raw)

pca = PCA(n_components=min(6, len(DELTA_FEATURES)), random_state=SEED)
Z = pca.fit_transform(Xs)

pc1 = Z[:, 0]

df["DeltaU_PC1_raw"] = pc1

# Orient stable_positive positive
if df.loc[df["condition"] == "stable_positive", "DeltaU_PC1_raw"].mean() < 0:
    pc1 = -pc1
    pca.components_[0] *= -1

df["DeltaU_PC1"] = pc1

variance_summary = pd.DataFrame({
    "component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
})

pc1_loadings = pd.DataFrame({
    "feature": DELTA_FEATURES,
    "layer": DELTA_LAYERS,
    "PC1_loading": pca.components_[0],
    "abs_loading": np.abs(pca.components_[0]),
})

phase_summary = (
    df.groupby("phase")
    .agg(
        n=("DeltaU_PC1", "count"),
        DeltaU_mean=("DeltaU_PC1", "mean"),
        DeltaU_std=("DeltaU_PC1", "std"),
        gen_negative_rate=("gen_negative_proxy", "mean"),
        final_R=("target_final_R", "mean"),
    )
    .reset_index()
)

condition_summary = (
    df.groupby(["condition", "phase"])
    .agg(
        n=("DeltaU_PC1", "count"),
        DeltaU_mean=("DeltaU_PC1", "mean"),
        DeltaU_std=("DeltaU_PC1", "std"),
        gen_negative_rate=("gen_negative_proxy", "mean"),
        final_R=("target_final_R", "mean"),
    )
    .reset_index()
)

# ============================================================
# CRITICAL BAND ESTIMATION
# ============================================================

phase_means = phase_summary.set_index("phase")["DeltaU_mean"].to_dict()

u_pos = phase_means.get("positive", np.nan)
u_crit = phase_means.get("critical", np.nan)
u_neg = phase_means.get("negative", np.nan)

U_minus = (u_neg + u_crit) / 2
U_plus = (u_crit + u_pos) / 2
band_width = U_plus - U_minus

def band_phase(u):
    if u > U_plus:
        return "positive"
    if u < U_minus:
        return "negative"
    return "critical"

df["phase_pred_band"] = df["DeltaU_PC1"].apply(band_phase)

band_summary = pd.DataFrame([{
    "U_minus": U_minus,
    "U_center": u_crit,
    "U_plus": U_plus,
    "band_width": band_width,
    "positive_mean": u_pos,
    "critical_mean": u_crit,
    "negative_mean": u_neg,
    "monotonic_pos_gt_crit_gt_neg": bool(u_pos > u_crit > u_neg),
}])

band_classification = pd.DataFrame([{
    "phase_acc": accuracy_score(df["phase"], df["phase_pred_band"]),
    "phase_macro_f1": f1_score(df["phase"], df["phase_pred_band"], average="macro"),
}])

# ============================================================
# PREDICTION TEST
# ============================================================

groups = df["graph_id"].values

def cls_cv(cols, target):
    X = df[cols].values.astype(float)
    y = df[target].values.astype(int)

    preds = np.zeros(len(df), dtype=int)
    binary = len(np.unique(y)) == 2
    probs = np.zeros(len(df)) if binary else None

    gkf = GroupKFold(n_splits=6)

    for tr, te in gkf.split(X, y, groups):
        model_cv = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ])
        model_cv.fit(X[tr], y[tr])
        preds[te] = model_cv.predict(X[te])
        if binary:
            probs[te] = model_cv.predict_proba(X[te])[:, 1]

    out = {
        "acc": accuracy_score(y, preds),
        "macro_f1": f1_score(y, preds, average="macro"),
        "auc": np.nan,
    }

    if binary:
        try:
            out["auc"] = roc_auc_score(y, probs)
        except Exception:
            pass

    return out

def reg_cv(cols, target):
    X = df[cols].values.astype(float)
    y = df[target].values.astype(float)

    preds = np.zeros(len(df))
    gkf = GroupKFold(n_splits=6)

    for tr, te in gkf.split(X, y, groups):
        model_cv = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ])
        model_cv.fit(X[tr], y[tr])
        preds[te] = model_cv.predict(X[te])

    return {
        "r2": r2_score(y, preds),
        "corr": np.corrcoef(y, preds)[0, 1] if np.std(preds) > 1e-8 else 0,
    }

pred_rows = []

feature_sets = {
    "DeltaU_PC1_only": ["DeltaU_PC1"],
    "raw_deltaR_window": DELTA_FEATURES,
}

for fs_name, cols in feature_sets.items():
    out = cls_cv(cols, "phase_id")
    pred_rows.append({
        "task": "classification",
        "target": "phase_id",
        "feature_set": fs_name,
        **out,
        "r2": np.nan,
        "corr": np.nan,
    })

    out = cls_cv(cols, "gen_negative_proxy")
    pred_rows.append({
        "task": "classification",
        "target": "gen_negative_proxy",
        "feature_set": fs_name,
        **out,
        "r2": np.nan,
        "corr": np.nan,
    })

    out = reg_cv(cols, "target_final_R")
    pred_rows.append({
        "task": "regression",
        "target": "target_final_R",
        "feature_set": fs_name,
        "acc": np.nan,
        "macro_f1": np.nan,
        "auc": np.nan,
        **out,
    })

prediction_summary = pd.DataFrame(pred_rows)

# ============================================================
# DIAGNOSTICS
# ============================================================

pc1_var = float(variance_summary.loc[0, "explained_variance_ratio"])
mono = bool(band_summary.loc[0, "monotonic_pos_gt_crit_gt_neg"])

diagnostics = pd.DataFrame([{
    "model_name": MODEL_NAME,
    "num_layers": num_layers,
    "delta_layers": str(DELTA_LAYERS),
    "pc1_variance": pc1_var,
    "pc123_cumulative": float(variance_summary.loc[min(2, len(variance_summary)-1), "cumulative_variance"]),
    "phase_order_monotonic": mono,
    "band_width": float(band_width),
    "band_phase_macro_f1": float(band_classification.loc[0, "phase_macro_f1"]),
    "PASS_Lite": bool(pc1_var > 0.65),
    "PASS": bool(pc1_var > 0.75 and mono),
    "PASS_Strong": bool(pc1_var > 0.80 and mono and band_classification.loc[0, "phase_macro_f1"] > 0.45),
}])

# ============================================================
# SAVE
# ============================================================

df.to_csv(SAVE_DIR / "ua6d_dataset_with_DeltaU.csv", index=False, encoding="utf-8-sig")
variance_summary.to_csv(SAVE_DIR / "ua6d_variance_summary.csv", index=False, encoding="utf-8-sig")
pc1_loadings.to_csv(SAVE_DIR / "ua6d_pc1_loadings.csv", index=False, encoding="utf-8-sig")
phase_summary.to_csv(SAVE_DIR / "ua6d_phase_summary.csv", index=False, encoding="utf-8-sig")
condition_summary.to_csv(SAVE_DIR / "ua6d_condition_summary.csv", index=False, encoding="utf-8-sig")
band_summary.to_csv(SAVE_DIR / "ua6d_band_summary.csv", index=False, encoding="utf-8-sig")
band_classification.to_csv(SAVE_DIR / "ua6d_band_classification.csv", index=False, encoding="utf-8-sig")
prediction_summary.to_csv(SAVE_DIR / "ua6d_prediction_summary.csv", index=False, encoding="utf-8-sig")
diagnostics.to_csv(SAVE_DIR / "ua6d_diagnostics.csv", index=False, encoding="utf-8-sig")

with open(SAVE_DIR / "ua6d_config.json", "w", encoding="utf-8") as f:
    json.dump({
        "model_path": MODEL_PATH,
        "model_name": MODEL_NAME,
        "num_layers": num_layers,
        "delta_layers": DELTA_LAYERS,
        "relative_layer_start": REL_LAYER_START,
        "relative_layer_end": REL_LAYER_END,
        "hypothesis": "DeltaU = PC1(deltaR_window) is a cross-model order parameter.",
        "success_criteria": {
            "PASS_Lite": "PC1 variance > 0.65",
            "PASS": "PC1 variance > 0.75 and phase means monotonic",
            "PASS_Strong": "PC1 variance > 0.80, phase means monotonic, band classifier macro-F1 > 0.45"
        }
    }, f, ensure_ascii=False, indent=2)

print("\n========== UA-6D DIAGNOSTICS ==========")
print(diagnostics.to_string(index=False))

print("\n========== UA-6D VARIANCE ==========")
print(variance_summary.to_string(index=False))

print("\n========== UA-6D PHASE SUMMARY ==========")
print(phase_summary.to_string(index=False))

print("\n========== UA-6D BAND SUMMARY ==========")
print(band_summary.to_string(index=False))

print("\n========== UA-6D PREDICTION SUMMARY ==========")
print(prediction_summary.to_string(index=False))

print("\n[DONE] Saved to:", SAVE_DIR)