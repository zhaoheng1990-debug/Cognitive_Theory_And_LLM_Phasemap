# ============================================================
# HA-1: Hallucination Basin Crossing Audit
#
# Goal:
#   Test whether hallucination corresponds to basin crossing:
#
#       ΔU = PC1(ΔR_20:25)
#       ΔU enters critical band
#       -> Gen_E probability increases
#
# Core readout:
#   R_l = logit_l(C) - logit_l(E)
#   ΔR_l = R_l(condition) - R_l(clean)
#   ΔU = PC1(ΔR_20:25)
#
# Outputs:
#   ha1_outputs/ha1_dataset.csv
#   ha1_outputs/ha1_layer_R.csv
#   ha1_outputs/ha1_deltaR.csv
#   ha1_outputs/ha1_summary.json
#   ha1_outputs/ha1_phase_band.csv
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
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

SAVE_DIR = Path("./ha1_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SEED = 42
N_GRAPHS = 96
BATCH_SIZE = 8
MAX_LEN = 256
GEN_MAX_NEW_TOKENS = 4

TRACK_LAYERS = list(range(20, 28))
DELTA_LAYERS = list(range(20, 26))

CONDITIONS = [
    ("clean", 0.0, "positive"),
    ("weak_distractor", 1.0, "positive"),
    ("ambiguous_branch", 2.0, "critical"),
    ("strong_conflict", 3.0, "negative"),
    ("closure_negation", 4.0, "negative"),
    ("closure_rewrite", 5.0, "negative"),
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
# LOAD MODEL
# ============================================================

print("Loading model...")

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

if max(TRACK_LAYERS) >= num_layers:
    raise RuntimeError(f"Model has {num_layers} layers, but TRACK_LAYERS includes {max(TRACK_LAYERS)}.")

# ============================================================
# LABEL SELECTION
# ============================================================

def token_ids_no_special(text):
    return tokenizer(text, add_special_tokens=False)["input_ids"]

def continuation_ids(label):
    return token_ids_no_special(" " + label)

def select_single_token_labels():
    candidates = [
        "Red", "Blue", "Green", "Yellow",
        "North", "South", "East", "West",
        "Alpha", "Beta", "Gamma", "Delta",
        "Circle", "Square", "Triangle", "Star",
        "Copper", "Silver", "Gold", "Iron",
        "Apple", "Orange", "Lemon", "Pear",
        "River", "Mountain", "Forest", "Ocean",
        "Sun", "Moon", "Cloud", "Stone",
    ]

    single = []
    print("\nLabel tokenization audit:")
    for lab in candidates:
        ids = continuation_ids(lab)
        print(f"{lab:<10} ids={ids} len={len(ids)}")
        if len(ids) == 1:
            single.append(lab)

    if len(single) < 12:
        raise RuntimeError("Need at least 12 single-token labels.")

    return single[:12]

LABEL_POOL = select_single_token_labels()
print("Using labels:", LABEL_POOL)

# ============================================================
# DATASET
# ============================================================

def make_entity(prefix, i):
    return f"{prefix}{i:03d}"

def make_prompt(a, b, c_label, e_label, condition):
    option = f"Answer with exactly one word: {c_label} or {e_label}."

    if condition == "clean":
        return "\n".join([
            "You are given a small relation graph.",
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is associated with {c_label}.",
            f"Question: Which label is {a} associated with?",
            option,
            "Answer:"
        ])

    if condition == "weak_distractor":
        return "\n".join([
            "You are given a small relation graph.",
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is associated with {c_label}.",
            f"Note: Some unrelated examples may mention {e_label}.",
            f"Question: Which label is {a} associated with?",
            option,
            "Answer:"
        ])

    if condition == "ambiguous_branch":
        return "\n".join([
            "You are given a small relation graph.",
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is usually associated with {c_label}.",
            f"Fact 3: In another context, {a} may be associated with {e_label}.",
            f"Question: Which label is {a} associated with in the given graph?",
            option,
            "Answer:"
        ])

    if condition == "strong_conflict":
        return "\n".join([
            "You are given a small relation graph.",
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is associated with {c_label}.",
            f"Conflicting fact: {a} is associated with {e_label}.",
            f"Question: Which label is {a} associated with?",
            option,
            "Answer:"
        ])

    if condition == "closure_negation":
        return "\n".join([
            "You are given a small relation graph.",
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is associated with {c_label}.",
            f"Correction: {a} is not associated with {c_label}; {a} is associated with {e_label}.",
            f"Question: Which label is {a} associated with?",
            option,
            "Answer:"
        ])

    if condition == "closure_rewrite":
        return "\n".join([
            "You are given a small relation graph.",
            f"Initial rule: if an item belongs to {b}, it is associated with {c_label}.",
            f"Updated rule: for this item, the association is overridden to {e_label}.",
            f"Fact: {a} belongs to {b}.",
            f"Question: Which label is {a} associated with after the update?",
            option,
            "Answer:"
        ])

    raise ValueError(condition)

def build_dataset():
    rows = []
    for gid in range(N_GRAPHS):
        a = make_entity("A", gid)
        b = make_entity("B", gid)

        c_label = LABEL_POOL[(2 * gid) % len(LABEL_POOL)]
        e_label = LABEL_POOL[(2 * gid + 1) % len(LABEL_POOL)]

        if c_label == e_label:
            e_label = LABEL_POOL[(2 * gid + 2) % len(LABEL_POOL)]

        for condition, eps, phase in CONDITIONS:
            prompt = make_prompt(a, b, c_label, e_label, condition)
            rows.append({
                "graph_id": gid,
                "condition": condition,
                "epsilon": eps,
                "phase_target": phase,
                "A": a,
                "B": b,
                "C_label": c_label,
                "E_label": e_label,
                "prompt": prompt,
            })

    return pd.DataFrame(rows)

df = build_dataset()
df.to_csv(SAVE_DIR / "ha1_dataset.csv", index=False)
print("Dataset:", df.shape)

# ============================================================
# MODEL READOUT
# ============================================================

@torch.no_grad()
def batch_layer_logits(prompts, c_labels, e_labels):
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
    ).to(model.device)

    outputs = model(
        **enc,
        output_hidden_states=True,
        use_cache=False,
    )

    hidden_states = outputs.hidden_states

    attention_mask = enc["attention_mask"]
    last_indices = attention_mask.sum(dim=1) - 1

    rows = []

    for bi in range(len(prompts)):
        c_id = continuation_ids(c_labels[bi])[0]
        e_id = continuation_ids(e_labels[bi])[0]

        for layer in TRACK_LAYERS:
            # hidden_states index: 0 embedding, 1 after layer0, ...
            hs = hidden_states[layer + 1][bi, last_indices[bi], :]
            logits = model.lm_head(hs).float()

            c_logit = logits[c_id].item()
            e_logit = logits[e_id].item()
            R = c_logit - e_logit

            rows.append({
                "batch_index": bi,
                "layer": layer,
                "C_id": c_id,
                "E_id": e_id,
                "C_logit": c_logit,
                "E_logit": e_logit,
                "R": R,
            })

    return rows

@torch.no_grad()
def batch_generate(prompts):
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
    ).to(model.device)

    out = model.generate(
        **enc,
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    gen_texts = []
    for i in range(out.shape[0]):
        new_ids = out[i, enc["input_ids"].shape[1]:]
        gen_texts.append(tokenizer.decode(new_ids, skip_special_tokens=True).strip())

    return gen_texts

# ============================================================
# RUN READOUT
# ============================================================

layer_rows = []
gen_rows = []

print("\nRunning layer readout and generation...")

for start in range(0, len(df), BATCH_SIZE):
    batch = df.iloc[start:start + BATCH_SIZE].reset_index(drop=True)

    prompts = batch["prompt"].tolist()
    c_labels = batch["C_label"].tolist()
    e_labels = batch["E_label"].tolist()

    read_rows = batch_layer_logits(prompts, c_labels, e_labels)

    for r in read_rows:
        source = batch.iloc[r["batch_index"]]
        r.update({
            "row_id": int(start + r["batch_index"]),
            "graph_id": int(source["graph_id"]),
            "condition": source["condition"],
            "epsilon": float(source["epsilon"]),
            "phase_target": source["phase_target"],
            "C_label": source["C_label"],
            "E_label": source["E_label"],
        })
        layer_rows.append(r)

    gens = batch_generate(prompts)

    for i, g in enumerate(gens):
        source = batch.iloc[i]
        c = source["C_label"]
        e = source["E_label"]

        gen_c = int(g.lower().startswith(c.lower()))
        gen_e = int(g.lower().startswith(e.lower()))

        gen_rows.append({
            "row_id": int(start + i),
            "graph_id": int(source["graph_id"]),
            "condition": source["condition"],
            "epsilon": float(source["epsilon"]),
            "phase_target": source["phase_target"],
            "C_label": c,
            "E_label": e,
            "generation": g,
            "Gen_C": gen_c,
            "Gen_E": gen_e,
        })

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

layer_df = pd.DataFrame(layer_rows)
gen_df = pd.DataFrame(gen_rows)

layer_df.to_csv(SAVE_DIR / "ha1_layer_R.csv", index=False)
gen_df.to_csv(SAVE_DIR / "ha1_generation.csv", index=False)

print("Layer R:", layer_df.shape)
print("Generation:", gen_df.shape)

# ============================================================
# COMPUTE ΔR
# ============================================================

print("\nComputing ΔR...")

wide = layer_df.pivot_table(
    index=["row_id", "graph_id", "condition", "epsilon", "phase_target", "C_label", "E_label"],
    columns="layer",
    values="R",
).reset_index()

wide.columns = [str(c) if isinstance(c, int) else c for c in wide.columns]

clean = wide[wide["condition"] == "clean"].copy()
clean_cols = ["graph_id"] + [str(l) for l in TRACK_LAYERS]
clean = clean[clean_cols].rename(columns={str(l): f"clean_R_{l}" for l in TRACK_LAYERS})

delta = wide.merge(clean, on="graph_id", how="left")

for l in TRACK_LAYERS:
    delta[f"R_{l}"] = delta[str(l)]
    delta[f"dR_{l}"] = delta[str(l)] - delta[f"clean_R_{l}"]

delta = delta.merge(gen_df[["row_id", "generation", "Gen_C", "Gen_E"]], on="row_id", how="left")

delta_cols = [
    "row_id", "graph_id", "condition", "epsilon", "phase_target",
    "C_label", "E_label", "generation", "Gen_C", "Gen_E"
] + [f"R_{l}" for l in TRACK_LAYERS] + [f"dR_{l}" for l in TRACK_LAYERS]

delta = delta[delta_cols]
delta.to_csv(SAVE_DIR / "ha1_deltaR.csv", index=False)

# ============================================================
# FIT ΔU = PC1(ΔR20:25)
# ============================================================

print("\nFitting ΔU = PC1(ΔR20:25)...")

X = delta[[f"dR_{l}" for l in DELTA_LAYERS]].values.astype(np.float32)

pca = PCA(n_components=3, random_state=SEED)
Z = pca.fit_transform(X)

delta["DeltaU_PC1_raw"] = Z[:, 0]
delta["DeltaU_PC2"] = Z[:, 1]
delta["DeltaU_PC3"] = Z[:, 2]

# Orient PC1 so clean/positive tends high and negative tends low
phase_mean = delta.groupby("phase_target")["DeltaU_PC1_raw"].mean().to_dict()
if phase_mean.get("positive", 0) < phase_mean.get("negative", 0):
    delta["DeltaU"] = -delta["DeltaU_PC1_raw"]
    pc1_loading = (-pca.components_[0]).tolist()
else:
    delta["DeltaU"] = delta["DeltaU_PC1_raw"]
    pc1_loading = pca.components_[0].tolist()

delta.to_csv(SAVE_DIR / "ha1_deltaU.csv", index=False)

# ============================================================
# ESTIMATE CRITICAL BAND
# ============================================================

phase_stats = delta.groupby("phase_target")["DeltaU"].agg(["mean", "std", "count"]).reset_index()

crit_vals = delta[delta["phase_target"] == "critical"]["DeltaU"].values

if len(crit_vals) >= 4:
    U_minus = float(np.percentile(crit_vals, 10))
    U_center = float(np.percentile(crit_vals, 50))
    U_plus = float(np.percentile(crit_vals, 90))
else:
    U_minus = float(delta["DeltaU"].quantile(0.33))
    U_center = float(delta["DeltaU"].quantile(0.50))
    U_plus = float(delta["DeltaU"].quantile(0.67))

delta["in_critical_band"] = ((delta["DeltaU"] >= U_minus) & (delta["DeltaU"] <= U_plus)).astype(int)

band_summary = delta.groupby(["phase_target", "condition"])["in_critical_band"].mean().reset_index()
band_summary.to_csv(SAVE_DIR / "ha1_phase_band.csv", index=False)

# ============================================================
# PREDICT Gen_E FROM ΔU
# ============================================================

print("\nEvaluating ΔU -> Gen_E...")

eval_df = delta.copy()
y = eval_df["Gen_E"].values.astype(int)
u = eval_df[["DeltaU"]].values.astype(np.float32)

metrics = {}

if len(np.unique(y)) > 1:
    auc = roc_auc_score(y, -eval_df["DeltaU"].values)
    metrics["auc_DeltaU_to_GenE_negative_direction"] = float(auc)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    preds = np.zeros(len(eval_df))
    for train_idx, test_idx in skf.split(u, y):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000)),
        ])
        clf.fit(u[train_idx], y[train_idx])
        preds[test_idx] = clf.predict_proba(u[test_idx])[:, 1]

    pred_label = (preds >= 0.5).astype(int)

    metrics["cv_auc"] = float(roc_auc_score(y, preds))
    metrics["cv_acc"] = float(accuracy_score(y, pred_label))
    metrics["cv_f1"] = float(f1_score(y, pred_label, zero_division=0))

    eval_df["Pred_GenE_prob_from_DeltaU"] = preds
else:
    metrics["auc_DeltaU_to_GenE_negative_direction"] = None
    metrics["cv_auc"] = None
    metrics["cv_acc"] = None
    metrics["cv_f1"] = None
    eval_df["Pred_GenE_prob_from_DeltaU"] = np.nan

eval_df.to_csv(SAVE_DIR / "ha1_eval.csv", index=False)

# ============================================================
# RESPONSE FUNCTION χ APPROXIMATION
# ============================================================

print("\nEstimating response curve...")

bins = np.quantile(delta["DeltaU"], np.linspace(0, 1, 8))
bins = np.unique(bins)

if len(bins) >= 3:
    delta["U_bin"] = pd.cut(delta["DeltaU"], bins=bins, include_lowest=True)
    curve = delta.groupby("U_bin").agg(
        DeltaU_mean=("DeltaU", "mean"),
        GenE_rate=("Gen_E", "mean"),
        count=("Gen_E", "count"),
        in_band_rate=("in_critical_band", "mean"),
    ).reset_index()

    curve["chi_abs"] = curve["GenE_rate"].diff().abs() / curve["DeltaU_mean"].diff().abs()
else:
    curve = pd.DataFrame()

curve.to_csv(SAVE_DIR / "ha1_response_curve.csv", index=False)

inside = delta[delta["in_critical_band"] == 1]
outside = delta[delta["in_critical_band"] == 0]

response_summary = {
    "inside_band_GenE_rate": float(inside["Gen_E"].mean()) if len(inside) else None,
    "outside_band_GenE_rate": float(outside["Gen_E"].mean()) if len(outside) else None,
    "inside_band_count": int(len(inside)),
    "outside_band_count": int(len(outside)),
}

# ============================================================
# SUMMARY
# ============================================================

summary = {
    "experiment": "HA-1 Hallucination Basin Crossing Audit",
    "model_path": MODEL_PATH,
    "n_rows": int(len(delta)),
    "n_graphs": int(N_GRAPHS),
    "conditions": CONDITIONS,
    "track_layers": TRACK_LAYERS,
    "delta_layers": DELTA_LAYERS,
    "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    "pc1_loading_oriented": pc1_loading,
    "phase_mean_DeltaU": delta.groupby("phase_target")["DeltaU"].mean().to_dict(),
    "condition_mean_DeltaU": delta.groupby("condition")["DeltaU"].mean().to_dict(),
    "condition_GenE_rate": delta.groupby("condition")["Gen_E"].mean().to_dict(),
    "critical_band": {
        "U_minus": U_minus,
        "U_center": U_center,
        "U_plus": U_plus,
        "band_width": U_plus - U_minus,
    },
    "band_response": response_summary,
    "metrics": metrics,
}

with open(SAVE_DIR / "ha1_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\nDone.")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nSaved to: {SAVE_DIR.resolve()}")