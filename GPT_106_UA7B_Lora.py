# ============================================================
# UA-7B: Multi-LoRA Landscape Audit
# Base vs Code / Math / Medical LoRA
# ============================================================

import gc, json, random
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

LORA_SPECS = {
    "code_lora": r"D:\model\Qwen2.5-1.5B-Instruct-Code-LoRA-r16v2",
    "math_lora": r"D:\model\Qwen2.5-1.5B-Math-GRPO-LoRA",
    "medical_lora": r"D:\model\qwen2.5-1.5b-medical-lora",
}

SAVE_DIR = Path("./ua7b_multi_lora_outputs")
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
# TOKENIZER
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

def load_lora_model(lora_path):
    base = load_base_model()
    model = PeftModel.from_pretrained(
        base,
        lora_path,
        local_files_only=True,
    )
    model.eval()
    return model

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

def run_variant(model_variant, model, df_tasks):
    records = []

    for idx, row in df_tasks.iterrows():
        print(f"[{model_variant}] {idx+1}/{len(df_tasks)} graph={row.graph_id} cond={row.condition}")

        traj = extract_R_trajectory(
            model,
            row["prompt"],
            row["clean_label"],
            row["conflict_label"],
        )

        rec = {
            "model_variant": model_variant,
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

def unload_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================
# RUN ALL VARIANTS
# ============================================================

df_tasks = build_dataset()
all_dfs = []

print("\n========== RUN BASE ==========")
model = load_base_model()
all_dfs.append(run_variant("base", model, df_tasks))
unload_model(model)

for name, path in LORA_SPECS.items():
    print(f"\n========== RUN {name} ==========")
    model = load_lora_model(path)
    all_dfs.append(run_variant(name, model, df_tasks))
    unload_model(model)

df = pd.concat(all_dfs, ignore_index=True)

# ============================================================
# CLEAN-RELATIVE ΔR WITHIN EACH VARIANT
# ============================================================

variants = ["base"] + list(LORA_SPECS.keys())

for mv in variants:
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
# JOINT PCA: ΔU = PC1(dltR_20...25)
# ============================================================

DELTA_FEATURES = [f"dltR_{l}" for l in DELTA_LAYERS]

X = df[DELTA_FEATURES].values.astype(float)
Xs = StandardScaler().fit_transform(X)

pca = PCA(n_components=len(DELTA_FEATURES), random_state=SEED)
Z = pca.fit_transform(Xs)

df["DeltaU_PC1_raw"] = Z[:, 0]

if df.loc[df["condition"] == "stable_positive", "DeltaU_PC1_raw"].mean() < 0:
    df["DeltaU_PC1"] = -df["DeltaU_PC1_raw"]
    pca.components_[0] *= -1
else:
    df["DeltaU_PC1"] = df["DeltaU_PC1_raw"]

pca_summary = pd.DataFrame({
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
# CONDITION SUMMARY
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

# ============================================================
# LoRA SHIFT vs BASE
# ============================================================

base_df = df[df["model_variant"] == "base"].set_index(["graph_id", "condition"])

shift_records = []

for mv in LORA_SPECS.keys():
    lora_df = df[df["model_variant"] == mv].set_index(["graph_id", "condition"])

    for idx in base_df.index:
        b = base_df.loc[idx]
        lo = lora_df.loc[idx]

        rec = {
            "lora_variant": mv,
            "graph_id": idx[0],
            "condition": idx[1],
            "phase": b["phase"],
        }

        for l in TRACK_LAYERS:
            rec[f"R_shift_{l}"] = lo[f"R_{l}"] - b[f"R_{l}"]
            rec[f"dltR_shift_{l}"] = lo[f"dltR_{l}"] - b[f"dltR_{l}"]

        rec["DeltaU_base"] = b["DeltaU_PC1"]
        rec["DeltaU_lora"] = lo["DeltaU_PC1"]
        rec["DeltaU_shift"] = lo["DeltaU_PC1"] - b["DeltaU_PC1"]

        shift_records.append(rec)

df_shift = pd.DataFrame(shift_records)

# ============================================================
# SUMMARIES
# ============================================================

deltaU_shift_summary = (
    df_shift.groupby(["lora_variant", "condition", "phase"])
    .agg(
        n=("graph_id", "count"),
        DeltaU_base_mean=("DeltaU_base", "mean"),
        DeltaU_lora_mean=("DeltaU_lora", "mean"),
        DeltaU_shift_mean=("DeltaU_shift", "mean"),
        DeltaU_shift_std=("DeltaU_shift", "std"),
    )
    .reset_index()
)

window_rows = []

WINDOWS = {
    "shallow_0_6": list(range(0, 7)),
    "mid_7_19": list(range(7, 20)),
    "boundary_20_22": list(range(20, 23)),
    "basin_23_25": list(range(23, 26)),
    "late_26_27": list(range(26, 28)),
}

for (mv, cond), sub in df_shift.groupby(["lora_variant", "condition"]):
    for wname, layers in WINDOWS.items():
        r_cols = [f"R_shift_{l}" for l in layers]
        d_cols = [f"dltR_shift_{l}" for l in layers]

        window_rows.append({
            "lora_variant": mv,
            "condition": cond,
            "phase": sub["phase"].iloc[0],
            "layer_group": wname,
            "R_shift_mean": sub[r_cols].mean(axis=1).mean(),
            "R_shift_std": sub[r_cols].mean(axis=1).std(),
            "dltR_shift_mean": sub[d_cols].mean(axis=1).mean(),
            "dltR_shift_std": sub[d_cols].mean(axis=1).std(),
        })

window_shift_summary = pd.DataFrame(window_rows)

# Which phase is most sensitive to each LoRA?
sensitivity_summary = (
    df_shift.groupby(["lora_variant", "phase"])
    .agg(
        n=("graph_id", "count"),
        abs_DeltaU_shift_mean=("DeltaU_shift", lambda x: np.mean(np.abs(x))),
        DeltaU_shift_mean=("DeltaU_shift", "mean"),
        DeltaU_shift_std=("DeltaU_shift", "std"),
    )
    .reset_index()
)

# Pairwise LoRA effect similarity by condition-level DeltaU shifts
pivot = deltaU_shift_summary.pivot_table(
    index=["condition", "phase"],
    columns="lora_variant",
    values="DeltaU_shift_mean",
)

corr = pivot.corr().reset_index().rename(columns={"lora_variant": "variant"})
pairwise_lora_shift_correlation = corr

# ============================================================
# SAVE
# ============================================================

df.to_csv(SAVE_DIR / "ua7b_dataset_all_variants.csv", index=False, encoding="utf-8-sig")
df_shift.to_csv(SAVE_DIR / "ua7b_lora_shift_dataset.csv", index=False, encoding="utf-8-sig")
condition_summary.to_csv(SAVE_DIR / "ua7b_condition_summary.csv", index=False, encoding="utf-8-sig")
deltaU_shift_summary.to_csv(SAVE_DIR / "ua7b_deltaU_shift_summary.csv", index=False, encoding="utf-8-sig")
window_shift_summary.to_csv(SAVE_DIR / "ua7b_window_shift_summary.csv", index=False, encoding="utf-8-sig")
sensitivity_summary.to_csv(SAVE_DIR / "ua7b_phase_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
pairwise_lora_shift_correlation.to_csv(SAVE_DIR / "ua7b_pairwise_lora_shift_correlation.csv", index=False, encoding="utf-8-sig")
pca_summary.to_csv(SAVE_DIR / "ua7b_pca_summary.csv", index=False, encoding="utf-8-sig")
pc1_loadings.to_csv(SAVE_DIR / "ua7b_pc1_loadings.csv", index=False, encoding="utf-8-sig")

with open(SAVE_DIR / "ua7b_config.json", "w", encoding="utf-8") as f:
    json.dump({
        "base_model": BASE_MODEL,
        "lora_specs": LORA_SPECS,
        "n_graphs": N_GRAPHS,
        "conditions": CONDITIONS,
        "track_layers": TRACK_LAYERS,
        "delta_layers": DELTA_LAYERS,
        "hypothesis": "Different LoRAs act as controlled potential-landscape perturbations.",
        "success_criteria": {
            "PASS_Lite": "Each LoRA produces nonzero DeltaU shift relative to base.",
            "PASS": "LoRA effects concentrate more in critical/competition phase than stable phase.",
            "PASS_Strong": "PC1 remains dominant while LoRAs produce distinct, structured phase-specific shifts."
        }
    }, f, ensure_ascii=False, indent=2)

# ============================================================
# PRINT
# ============================================================

print("\n========== UA-7B PCA SUMMARY ==========")
print(pca_summary.to_string(index=False))

print("\n========== UA-7B PC1 LOADINGS ==========")
print(pc1_loadings.to_string(index=False))

print("\n========== UA-7B CONDITION SUMMARY ==========")
print(condition_summary.to_string(index=False))

print("\n========== UA-7B DELTAU SHIFT SUMMARY ==========")
print(deltaU_shift_summary.to_string(index=False))

print("\n========== UA-7B WINDOW SHIFT SUMMARY ==========")
print(window_shift_summary.to_string(index=False))

print("\n========== UA-7B PHASE SENSITIVITY SUMMARY ==========")
print(sensitivity_summary.to_string(index=False))

print("\n========== UA-7B PAIRWISE LORA SHIFT CORRELATION ==========")
print(pairwise_lora_shift_correlation.to_string(index=False))

print("\n[DONE] Saved to:", SAVE_DIR)