# ============================================================
# UA-7A Lite: LoRA Potential Landscape Audit
#
# Compare:
#   Base Qwen2.5-1.5B-Instruct
#   Base + Code LoRA
#
# Core question:
#   Does LoRA shift internal basin dynamics?
#
# Outputs:
#   ua7a_dataset_base_lora.csv
#   ua7a_condition_summary.csv
#   ua7a_lora_shift_summary.csv
#   ua7a_pca_summary.csv
#   ua7a_pc1_loadings.csv
# ============================================================

import gc
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"
LORA_PATH = r"D:\model\Qwen2.5-1.5B-Instruct-Code-LoRA-r16v2"

SAVE_DIR = Path("./ua7a_lora_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SEED = 42
N_GRAPHS = 48
MAX_LEN = 260

TRACK_LAYERS = list(range(0, 28))
DELTA_LAYERS = list(range(20, 26))

CONDITIONS = [
    "stable_positive",
    "true_compete_balanced",
    "direct_negative",
]

PHASE_MAP = {
    "stable_positive": "positive",
    "true_compete_balanced": "critical",
    "direct_negative": "negative",
}

LABEL_PAIRS = [
    ("Red", "Blue"),
    ("Green", "Yellow"),
    ("North", "South"),
    ("East", "West"),
    ("Alpha", "Beta"),
    ("Circle", "Square"),
]

# ============================================================
# SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# LOAD TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    local_files_only=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"

# ============================================================
# HELPERS
# ============================================================

def continuation_ids(label):
    return tokenizer(" " + label, add_special_tokens=False)["input_ids"]

def label_score(logits_vec, label):
    ids = continuation_ids(label)
    vals = []
    for tid in ids:
        if tid < logits_vec.shape[-1]:
            vals.append(float(logits_vec[tid]))
    return float(np.mean(vals)) if vals else -1e9

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

    else:
        raise ValueError(condition)

    return "\n".join(lines)

def build_dataset():
    rows = []
    for gid in range(N_GRAPHS):
        a = make_entity("A", gid)
        b = make_entity("B", gid)
        clean_label, conflict_label = LABEL_PAIRS[gid % len(LABEL_PAIRS)]

        for cond in CONDITIONS:
            rows.append({
                "graph_id": gid,
                "condition": cond,
                "phase": PHASE_MAP[cond],
                "clean_label": clean_label,
                "conflict_label": conflict_label,
                "prompt": make_prompt(a, b, clean_label, conflict_label, cond),
            })

    return pd.DataFrame(rows)

@torch.no_grad()
def extract_R_trajectory(model, prompt, clean_label, conflict_label):
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
    rec = {}

    for l in TRACK_LAYERS:
        h = hidden_states[l + 1][0, -1, :]
        logits = model.lm_head(h).detach().float().cpu().numpy()

        sc = label_score(logits, clean_label)
        se = label_score(logits, conflict_label)

        rec[f"R_{l}"] = sc - se

    del out, inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rec

def load_base_model():
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
        local_files_only=True,
    )
    if DEVICE == "cpu":
        model.to(DEVICE)
    model.eval()
    return model

def load_lora_model():
    base = load_base_model()
    model = PeftModel.from_pretrained(
        base,
        LORA_PATH,
        local_files_only=True,
    )
    model.eval()
    return model

def run_model(model_name, model, df_tasks):
    records = []

    for idx, row in df_tasks.iterrows():
        print(f"[{model_name}] {idx+1}/{len(df_tasks)} graph={row.graph_id} cond={row.condition}")

        traj = extract_R_trajectory(
            model,
            row["prompt"],
            row["clean_label"],
            row["conflict_label"],
        )

        rec = {
            "model_variant": model_name,
            "graph_id": row["graph_id"],
            "condition": row["condition"],
            "phase": row["phase"],
            "clean_label": row["clean_label"],
            "conflict_label": row["conflict_label"],
        }
        rec.update(traj)
        records.append(rec)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(records)

# ============================================================
# RUN
# ============================================================

df_tasks = build_dataset()

print("\n========== RUN BASE ==========")
base_model = load_base_model()
df_base = run_model("base", base_model, df_tasks)

del base_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n========== RUN LORA ==========")
lora_model = load_lora_model()
df_lora = run_model("lora", lora_model, df_tasks)

del lora_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

df = pd.concat([df_base, df_lora], ignore_index=True)

# ============================================================
# CLEAN-RELATIVE ΔR WITHIN EACH MODEL VARIANT
# ============================================================

for mv in ["base", "lora"]:
    sub_clean = (
        df[(df["model_variant"] == mv) & (df["condition"] == "stable_positive")]
        .set_index("graph_id")
    )

    for l in TRACK_LAYERS:
        base_map = sub_clean[f"R_{l}"].to_dict()
        mask = df["model_variant"] == mv
        df.loc[mask, f"dltR_{l}"] = df.loc[mask].apply(
            lambda r: r[f"R_{l}"] - base_map[r["graph_id"]],
            axis=1,
        )

# ============================================================
# LoRA - Base shift
# ============================================================

merge_cols = ["graph_id", "condition", "phase", "clean_label", "conflict_label"]

base_wide = df[df["model_variant"] == "base"].set_index(["graph_id", "condition"])
lora_wide = df[df["model_variant"] == "lora"].set_index(["graph_id", "condition"])

shift_records = []

for idx in base_wide.index:
    b = base_wide.loc[idx]
    lo = lora_wide.loc[idx]

    rec = {
        "graph_id": idx[0],
        "condition": idx[1],
        "phase": b["phase"],
    }

    for l in TRACK_LAYERS:
        rec[f"R_shift_{l}"] = lo[f"R_{l}"] - b[f"R_{l}"]
        rec[f"dltR_shift_{l}"] = lo[f"dltR_{l}"] - b[f"dltR_{l}"]

    shift_records.append(rec)

df_shift = pd.DataFrame(shift_records)

# ============================================================
# PCA ΔU on base + lora jointly
# ============================================================

DELTA_FEATURES = [f"dltR_{l}" for l in DELTA_LAYERS]

X = df[DELTA_FEATURES].values.astype(float)
Xs = StandardScaler().fit_transform(X)

pca = PCA(n_components=len(DELTA_FEATURES), random_state=SEED)
Z = pca.fit_transform(Xs)

df["DeltaU_PC1_raw"] = Z[:, 0]

# Orient stable_positive positive
if df.loc[df["condition"] == "stable_positive", "DeltaU_PC1_raw"].mean() < 0:
    df["DeltaU_PC1"] = -df["DeltaU_PC1_raw"]
    pca.components_[0] *= -1
else:
    df["DeltaU_PC1"] = df["DeltaU_PC1_raw"]

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

# ============================================================
# SUMMARIES
# ============================================================

condition_summary = (
    df.groupby(["model_variant", "condition", "phase"])
    .agg(
        n=("DeltaU_PC1", "count"),
        DeltaU_mean=("DeltaU_PC1", "mean"),
        DeltaU_std=("DeltaU_PC1", "std"),
        final_R=(f"R_{TRACK_LAYERS[-1]}", "mean"),
        final_dltR=(f"dltR_{TRACK_LAYERS[-1]}", "mean"),
    )
    .reset_index()
)

shift_summary = (
    df_shift.groupby(["condition", "phase"])
    .agg(
        n=("graph_id", "count"),
        **{f"R_shift_{l}_mean": (f"R_shift_{l}", "mean") for l in TRACK_LAYERS},
        **{f"dltR_shift_{l}_mean": (f"dltR_shift_{l}", "mean") for l in TRACK_LAYERS},
    )
    .reset_index()
)

# compact layer-window shift summary
window_rows = []
for cond, sub in df_shift.groupby("condition"):
    for layer_group, layers in {
        "mid_7_19": list(range(7, 20)),
        "boundary_20_22": list(range(20, 23)),
        "basin_23_25": list(range(23, 26)),
        "late_26_27": list(range(26, 28)),
    }.items():
        r_cols = [f"R_shift_{l}" for l in layers if f"R_shift_{l}" in sub.columns]
        d_cols = [f"dltR_shift_{l}" for l in layers if f"dltR_shift_{l}" in sub.columns]

        window_rows.append({
            "condition": cond,
            "phase": sub["phase"].iloc[0],
            "layer_group": layer_group,
            "R_shift_mean": sub[r_cols].mean(axis=1).mean(),
            "R_shift_std": sub[r_cols].mean(axis=1).std(),
            "dltR_shift_mean": sub[d_cols].mean(axis=1).mean(),
            "dltR_shift_std": sub[d_cols].mean(axis=1).std(),
        })

window_shift_summary = pd.DataFrame(window_rows)

# DeltaU shift
du_base = df[df["model_variant"] == "base"][["graph_id", "condition", "DeltaU_PC1"]].rename(
    columns={"DeltaU_PC1": "DeltaU_base"}
)
du_lora = df[df["model_variant"] == "lora"][["graph_id", "condition", "DeltaU_PC1"]].rename(
    columns={"DeltaU_PC1": "DeltaU_lora"}
)

du_shift = du_base.merge(du_lora, on=["graph_id", "condition"])
du_shift["DeltaU_shift"] = du_shift["DeltaU_lora"] - du_shift["DeltaU_base"]

du_shift = du_shift.merge(
    df_tasks[["graph_id", "condition", "phase"]].drop_duplicates(),
    on=["graph_id", "condition"],
    how="left",
)

deltaU_shift_summary = (
    du_shift.groupby(["condition", "phase"])
    .agg(
        n=("graph_id", "count"),
        DeltaU_base_mean=("DeltaU_base", "mean"),
        DeltaU_lora_mean=("DeltaU_lora", "mean"),
        DeltaU_shift_mean=("DeltaU_shift", "mean"),
        DeltaU_shift_std=("DeltaU_shift", "std"),
    )
    .reset_index()
)

# ============================================================
# SAVE
# ============================================================

df.to_csv(SAVE_DIR / "ua7a_dataset_base_lora.csv", index=False, encoding="utf-8-sig")
df_shift.to_csv(SAVE_DIR / "ua7a_layerwise_lora_shift.csv", index=False, encoding="utf-8-sig")
condition_summary.to_csv(SAVE_DIR / "ua7a_condition_summary.csv", index=False, encoding="utf-8-sig")
shift_summary.to_csv(SAVE_DIR / "ua7a_shift_summary_wide.csv", index=False, encoding="utf-8-sig")
window_shift_summary.to_csv(SAVE_DIR / "ua7a_window_shift_summary.csv", index=False, encoding="utf-8-sig")
deltaU_shift_summary.to_csv(SAVE_DIR / "ua7a_deltaU_shift_summary.csv", index=False, encoding="utf-8-sig")
variance_summary.to_csv(SAVE_DIR / "ua7a_pca_summary.csv", index=False, encoding="utf-8-sig")
pc1_loadings.to_csv(SAVE_DIR / "ua7a_pc1_loadings.csv", index=False, encoding="utf-8-sig")

with open(SAVE_DIR / "ua7a_config.json", "w", encoding="utf-8") as f:
    json.dump({
        "base_model": BASE_MODEL,
        "lora_path": LORA_PATH,
        "n_graphs": N_GRAPHS,
        "conditions": CONDITIONS,
        "track_layers": TRACK_LAYERS,
        "delta_layers": DELTA_LAYERS,
        "hypothesis": "LoRA modifies potential landscape, visible as systematic DeltaR / DeltaU trajectory shift.",
        "success_criteria": {
            "PASS_Lite": "LoRA changes final_R or DeltaU consistently.",
            "PASS": "LoRA shifts DeltaR over mid/boundary/basin windows, not only final layer.",
            "PASS_Strong": "LoRA preserves low-dimensional PC1 structure while translating DeltaU trajectories."
        }
    }, f, ensure_ascii=False, indent=2)

# ============================================================
# PRINT
# ============================================================

print("\n========== UA-7A PCA SUMMARY ==========")
print(variance_summary.to_string(index=False))

print("\n========== UA-7A PC1 LOADINGS ==========")
print(pc1_loadings.to_string(index=False))

print("\n========== UA-7A CONDITION SUMMARY ==========")
print(condition_summary.to_string(index=False))

print("\n========== UA-7A WINDOW SHIFT SUMMARY ==========")
print(window_shift_summary.to_string(index=False))

print("\n========== UA-7A DELTAU SHIFT SUMMARY ==========")
print(deltaU_shift_summary.to_string(index=False))

print("\n[DONE] Saved to:", SAVE_DIR)