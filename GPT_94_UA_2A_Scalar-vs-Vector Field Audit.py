# ============================================================
# UA-2A: Scalar-vs-Vector Field Audit
# Test whether layerwise basin dynamics is scalar-potential-like
# or vector-field-like with non-conservative circulation.
# ============================================================

import os
import gc
import math
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"
SAVE_DIR = Path("./ua2a_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SEED = 42
MAX_LEN = 320

# Qwen2.5-1.5B has 28 layers: 0-27
TRACK_LAYERS = list(range(0, 28))

# Layer windows
WINDOWS = {
    "L0_6_init_manifold": list(range(0, 7)),
    "L7_19_accumulation": list(range(7, 20)),
    "L20_22_tau_entry": list(range(20, 23)),
    "L23_26_basin_formation": list(range(23, 27)),
    "L0_27_full": list(range(0, 28)),
}

# Candidate basin labels must be single or short continuations.
# You can replace them with domain-specific labels.
BASINS = {
    "sights": ["monument", "museum", "landmark", "attraction"],
    "capital": ["capital", "government", "administration", "state"],
    "europe_city": ["Europe", "European", "city", "France"],
    "history_culture": ["history", "culture", "heritage", "ancient"],
}

# Prompt path families.
# The same basin set is activated in different cyclic orders.
PROMPT_TEMPLATES = [
    {
        "path": "A_B_C_D_A",
        "order": ["sights", "capital", "europe_city", "history_culture", "sights"],
        "prompt": (
            "Consider Paris from four perspectives in this exact order: "
            "tourist landmarks, national capital, European city, historical culture, "
            "then return to tourist landmarks. "
            "Question: What perspective should dominate when answering about Paris?"
        ),
    },
    {
        "path": "A_D_C_B_A",
        "order": ["sights", "history_culture", "europe_city", "capital", "sights"],
        "prompt": (
            "Consider Paris from four perspectives in this exact order: "
            "tourist landmarks, historical culture, European city, national capital, "
            "then return to tourist landmarks. "
            "Question: What perspective should dominate when answering about Paris?"
        ),
    },
    {
        "path": "A_C_B_D_A",
        "order": ["sights", "europe_city", "capital", "history_culture", "sights"],
        "prompt": (
            "Consider Paris from four perspectives in this exact order: "
            "tourist landmarks, European city, national capital, historical culture, "
            "then return to tourist landmarks. "
            "Question: What perspective should dominate when answering about Paris?"
        ),
    },
    {
        "path": "B_A_D_C_B",
        "order": ["capital", "sights", "history_culture", "europe_city", "capital"],
        "prompt": (
            "Consider Paris from four perspectives in this exact order: "
            "national capital, tourist landmarks, historical culture, European city, "
            "then return to national capital. "
            "Question: What perspective should dominate when answering about Paris?"
        ),
    },
]

N_REPEATS = 16

# ============================================================
# SEED
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ============================================================
# LOAD MODEL
# ============================================================

print("\n[LOAD] Loading model...")

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
    raise ValueError(f"TRACK_LAYERS exceeds model depth: {num_layers}")

# ============================================================
# TOKEN HELPERS
# ============================================================

def continuation_ids(text: str):
    return tokenizer(" " + text, add_special_tokens=False)["input_ids"]

def score_label_from_logits(logits_vec, label: str):
    """
    Approximate basin support by averaging logits of continuation tokens.
    For multi-token labels, average token logits.
    """
    ids = continuation_ids(label)
    vals = []
    for tid in ids:
        if tid < logits_vec.shape[-1]:
            vals.append(float(logits_vec[tid]))
    if not vals:
        return -1e9
    return float(np.mean(vals))

def basin_score(logits_vec, basin_words):
    """
    Basin support score = logsumexp over label-word scores.
    """
    scores = np.array([score_label_from_logits(logits_vec, w) for w in basin_words], dtype=np.float64)
    m = np.max(scores)
    return float(m + np.log(np.sum(np.exp(scores - m))))

# ============================================================
# HIDDEN / LOGIT EXTRACTION
# ============================================================

@torch.no_grad()
def extract_layer_logits(prompt: str):
    """
    Returns:
      layer_logits: dict layer -> vocab logits at final prompt token.
    """
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
    # hidden_states[0] = embedding output
    # hidden_states[i+1] = after layer i

    lm_head = model.lm_head

    layer_logits = {}

    for l in TRACK_LAYERS:
        h = hidden_states[l + 1][0, -1, :]
        logits = lm_head(h).detach().float().cpu().numpy()
        layer_logits[l] = logits

    del out, inputs
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return layer_logits

def compute_U_trajectory(prompt: str):
    """
    U_l vector over basins:
      U_i(l) = -log softmax over basin scores.
    We compute basin scores from layer logits, softmax across basins,
    then U=-log P.
    """
    layer_logits = extract_layer_logits(prompt)

    rows = []
    basin_names = list(BASINS.keys())

    for l in TRACK_LAYERS:
        logits_vec = layer_logits[l]

        raw_scores = np.array([
            basin_score(logits_vec, BASINS[b])
            for b in basin_names
        ], dtype=np.float64)

        # Softmax across basin scores
        m = np.max(raw_scores)
        probs = np.exp(raw_scores - m)
        probs = probs / np.sum(probs)

        U = -np.log(probs + 1e-12)

        row = {"layer": l}
        for i, b in enumerate(basin_names):
            row[f"score_{b}"] = raw_scores[i]
            row[f"P_{b}"] = probs[i]
            row[f"U_{b}"] = U[i]
        rows.append(row)

    return pd.DataFrame(rows)

# ============================================================
# FIELD / CIRCULATION METRICS
# ============================================================

def compute_flow_metrics(df_u: pd.DataFrame, path_name: str, prompt_id: int):
    """
    Layerwise flow:
      F_l = U_{l+1} - U_l

    We test non-conservative behavior via:
      1. signed loop-like circulation in projected PC plane
      2. path work asymmetry
      3. curl score per layer window

    Since U trajectory is in high-D basin space, we estimate circulation
    by projecting trajectory to first two PCA axes and computing polygon signed area.
    Nonzero signed area = rotational component proxy.
    """
    basin_names = list(BASINS.keys())
    U_cols = [f"U_{b}" for b in basin_names]

    U = df_u[U_cols].values.astype(np.float64)
    layers = df_u["layer"].values

    # Center for PCA
    X = U - U.mean(axis=0, keepdims=True)

    # PCA by SVD
    try:
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        Z = X @ vt[:2].T
    except Exception:
        Z = X[:, :2]

    out_rows = []

    for wname, wlayers in WINDOWS.items():
        idx = [i for i, l in enumerate(layers) if l in wlayers]
        if len(idx) < 3:
            continue

        Zw = Z[idx]
        Uw = U[idx]

        # Signed area in PCA plane; proxy for circulation / rotational flow.
        x = Zw[:, 0]
        y = Zw[:, 1]
        signed_area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])

        # Path length
        diffs = np.diff(Zw, axis=0)
        path_len = float(np.sum(np.linalg.norm(diffs, axis=1)) + 1e-12)

        # Normalize area by path_len^2
        curl_area_score = float(abs(signed_area) / (path_len ** 2 + 1e-12))

        # Work-like metric in U space:
        # F_l = U_{l+1}-U_l
        # dU_l = same displacement, so raw self-work is positive.
        # To capture directional rotation, use successive flow turning.
        F = np.diff(Uw, axis=0)
        norms = np.linalg.norm(F, axis=1) + 1e-12

        if len(F) >= 2:
            cos_turn = np.sum(F[:-1] * F[1:], axis=1) / (norms[:-1] * norms[1:])
            mean_turn = float(np.mean(np.arccos(np.clip(cos_turn, -1, 1))))
            turn_energy = float(np.mean(1 - cos_turn))
        else:
            mean_turn = 0.0
            turn_energy = 0.0

        # Antisymmetric flow proxy:
        # Sum pairwise oriented 2D cross products in PCA plane.
        if len(diffs) >= 2:
            cross_vals = diffs[:-1, 0] * diffs[1:, 1] - diffs[:-1, 1] * diffs[1:, 0]
            oriented_turn = float(np.sum(cross_vals))
            abs_oriented_turn = float(np.sum(np.abs(cross_vals)))
            curl_turn_score = float(abs(oriented_turn) / (abs_oriented_turn + 1e-12))
        else:
            oriented_turn = 0.0
            curl_turn_score = 0.0

        # Scalar-field-like monotonicity:
        # If one basin simply wins, min(U) should tend to decrease.
        minU = np.min(Uw, axis=1)
        d_minU = np.diff(minU)
        monotonic_drop_frac = float(np.mean(d_minU <= 0)) if len(d_minU) else 0.0

        out_rows.append({
            "prompt_id": prompt_id,
            "path": path_name,
            "window": wname,
            "signed_area": float(signed_area),
            "curl_area_score": curl_area_score,
            "mean_turn_rad": mean_turn,
            "turn_energy": turn_energy,
            "oriented_turn": oriented_turn,
            "curl_turn_score": curl_turn_score,
            "path_len": path_len,
            "monotonic_drop_frac": monotonic_drop_frac,
        })

    return pd.DataFrame(out_rows)

# ============================================================
# MAIN RUN
# ============================================================

all_u = []
all_metrics = []

print("\n[RUN] Starting UA-2A...")

prompt_id = 0

for rep in range(N_REPEATS):
    for item in PROMPT_TEMPLATES:
        prompt_id += 1
        path_name = item["path"]
        prompt = item["prompt"]

        print(f"[RUN] rep={rep+1}/{N_REPEATS} path={path_name}")

        df_u = compute_U_trajectory(prompt)
        df_u["prompt_id"] = prompt_id
        df_u["rep"] = rep
        df_u["path"] = path_name
        df_u["prompt"] = prompt

        df_m = compute_flow_metrics(df_u, path_name, prompt_id)
        df_m["rep"] = rep

        all_u.append(df_u)
        all_metrics.append(df_m)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

df_U = pd.concat(all_u, ignore_index=True)
df_M = pd.concat(all_metrics, ignore_index=True)

# ============================================================
# SUMMARY
# ============================================================

summary = (
    df_M
    .groupby(["window"])
    .agg(
        n=("curl_area_score", "count"),
        curl_area_mean=("curl_area_score", "mean"),
        curl_area_std=("curl_area_score", "std"),
        curl_turn_mean=("curl_turn_score", "mean"),
        curl_turn_std=("curl_turn_score", "std"),
        turn_energy_mean=("turn_energy", "mean"),
        monotonic_drop_frac_mean=("monotonic_drop_frac", "mean"),
        signed_area_mean=("signed_area", "mean"),
        signed_area_abs_mean=("signed_area", lambda x: np.mean(np.abs(x))),
    )
    .reset_index()
)

path_summary = (
    df_M
    .groupby(["path", "window"])
    .agg(
        n=("curl_area_score", "count"),
        curl_area_mean=("curl_area_score", "mean"),
        curl_turn_mean=("curl_turn_score", "mean"),
        turn_energy_mean=("turn_energy", "mean"),
        monotonic_drop_frac_mean=("monotonic_drop_frac", "mean"),
        signed_area_mean=("signed_area", "mean"),
    )
    .reset_index()
)

# Simple classification
def classify_curl(x):
    if x < 0.05:
        return "scalar_like"
    elif x < 0.20:
        return "mixed"
    else:
        return "vector_like"

summary["field_class_by_area"] = summary["curl_area_mean"].apply(classify_curl)
summary["field_class_by_turn"] = summary["curl_turn_mean"].apply(classify_curl)

# Save
df_U.to_csv(SAVE_DIR / "ua2a_U_trajectories.csv", index=False, encoding="utf-8-sig")
df_M.to_csv(SAVE_DIR / "ua2a_curl_metrics.csv", index=False, encoding="utf-8-sig")
summary.to_csv(SAVE_DIR / "ua2a_summary.csv", index=False, encoding="utf-8-sig")
path_summary.to_csv(SAVE_DIR / "ua2a_path_summary.csv", index=False, encoding="utf-8-sig")

with open(SAVE_DIR / "ua2a_config.json", "w", encoding="utf-8") as f:
    json.dump({
        "model_path": MODEL_PATH,
        "basins": BASINS,
        "windows": WINDOWS,
        "n_repeats": N_REPEATS,
        "interpretation": {
            "curl_score_lt_0.05": "scalar_like",
            "curl_score_0.05_to_0.20": "mixed",
            "curl_score_gt_0.20": "vector_like"
        }
    }, f, ensure_ascii=False, indent=2)

print("\n[DONE] Saved outputs to:", SAVE_DIR)

print("\n========== UA-2A SUMMARY ==========")
print(summary.to_string(index=False))

print("\n========== PATH SUMMARY ==========")
print(path_summary.to_string(index=False))