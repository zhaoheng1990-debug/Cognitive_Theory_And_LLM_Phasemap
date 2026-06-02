# ============================================================
# Neighborhood-Audit-1c
# Corpus Robustness for Axiom 1
#
# Goal:
#   Test whether Axiom-1 evidence survives corpus shift:
#
#       H_l <-> N_k(H_l) = TopK(H_l W^T)
#
#   Neighborhood-Audit-1:
#       showed TopK token identity preserves hidden topology
#       on default-template corpus.
#
#   Neighborhood-Audit-1b:
#       showed the effect is not merely final-layer / L27 artifact.
#
#   Neighborhood-Audit-1c:
#       reruns the TopK identity topology audit on real natural texts,
#       then compares:
#
#           default_template corpus
#           real_text corpus
#
# Main outputs:
#   ./neighborhood_audit1_realtext_outputs/
#   ./neighborhood_audit1c_outputs/
#
# Key files:
#   neighborhood_audit1c_corpus_comparison.csv
#   neighborhood_audit1c_robustness_decision.csv
#   neighborhood_audit1c_nonfinal_summary.csv
#   neighborhood_audit1c_layer_band_comparison.csv
# ============================================================

import os
import re
import gc
import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

# Put your real natural-text corpus here.
# One non-empty line = one sample.
# If the file has too few lines, the script will try paragraph/sentence splitting.
TEXT_FILE = r"C:\Users\ZH\Desktop\AGI\data\natural_wiki_texts_450_balanced_diverse_no_meta_tail.txt"

# Existing default-template outputs from Neighborhood-Audit-1.
DEFAULT_TEMPLATE_DIR = Path("./neighborhood_audit1_outputs")

# New real-text run output.
REALTEXT_SAVE_DIR = Path("./neighborhood_audit1_realtext_outputs")
REALTEXT_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Comparison output.
SAVE_DIR = Path("./neighborhood_audit1c_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

TRACK_LAYERS = None

TOPK_LIST = [20, 50, 100, 200, 500, 1000, 2000, 5000]
TOPK_MAX = max(TOPK_LIST)

BATCH_SIZE = 4
MAX_LEN = 256
MAX_TEXTS = 450
MIN_TEXTS = 64

KNN_K = 10

MODES = ["center", "center+spread"]
BASELINES = ["real_topk", "random", "shuffled_topk", "input_token"]

FINAL_LAYER = 27

LAYER_BANDS = {
    "L0_2_input_boundary": list(range(0, 3)),
    "L3_17_latent_bulk": list(range(3, 18)),
    "L18_23_recoupling_ramp": list(range(18, 24)),
    "L24_26_pre_final_peak": list(range(24, 27)),
    "L27_final_readout": [27],
}

STRONG_TOPO_THRESHOLD = 0.45
STRONG_LIFT_RANDOM_THRESHOLD = 0.25
STRONG_LIFT_SHUFFLED_THRESHOLD = 0.25
STRONG_LIFT_INPUT_THRESHOLD = 0.10
NONFINAL_RATIO_THRESHOLD = 0.65

# ============================================================
# SEED
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)

# ============================================================
# TEXT LOADING
# ============================================================

def normalize_text(s: str) -> str:
    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def split_long_text_to_sentences(text: str):
    # Basic multilingual sentence splitting.
    parts = re.split(r"(?<=[。！？.!?])\s+", text)
    out = []
    for p in parts:
        p = normalize_text(p)
        if len(p) >= 40:
            out.append(p)
    return out


def load_real_texts(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"TEXT_FILE does not exist: {p}\n"
            "Please create a real natural-text corpus file first."
        )

    raw = p.read_text(encoding="utf-8", errors="ignore")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    # First try line-based samples.
    lines = [normalize_text(x) for x in raw.split("\n")]
    lines = [x for x in lines if len(x) >= 40]

    if len(lines) >= MIN_TEXTS:
        texts = lines
    else:
        # Try paragraph split.
        paragraphs = re.split(r"\n\s*\n+", raw)
        paragraphs = [normalize_text(x) for x in paragraphs]
        paragraphs = [x for x in paragraphs if len(x) >= 40]

        if len(paragraphs) >= MIN_TEXTS:
            texts = paragraphs
        else:
            # Last fallback: sentence split.
            texts = split_long_text_to_sentences(raw)

    # Filter extreme length before tokenizer truncation.
    texts = [x for x in texts if 40 <= len(x) <= 3000]

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(texts)

    if len(texts) < MIN_TEXTS:
        raise RuntimeError(
            f"Too few usable text samples: {len(texts)}. "
            f"Need at least {MIN_TEXTS}, preferably 200+."
        )

    texts = texts[:MAX_TEXTS]
    return texts


TEXTS = load_real_texts(TEXT_FILE)

with open(REALTEXT_SAVE_DIR / "neighborhood_audit1c_realtext_prompts.txt", "w", encoding="utf-8") as f:
    for i, t in enumerate(TEXTS):
        f.write(f"[{i:04d}] {t}\n")

print("\n============================================================")
print("Neighborhood-Audit-1c Input Summary")
print("============================================================")
print("Text file:", TEXT_FILE)
print("Num real-text samples:", len(TEXTS))
print("Device:", DEVICE)
print("TopK list:", TOPK_LIST)
print("Modes:", MODES)

# ============================================================
# MODEL
# ============================================================

if "你的snapshot目录" in MODEL_PATH:
    raise RuntimeError("Please replace MODEL_PATH with your actual local HuggingFace snapshot path.")

print("\nLoading model...")

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

if TRACK_LAYERS is None:
    TRACK_LAYERS = list(range(num_layers))

print("Num transformer layers:", num_layers)
print("Track layers:", TRACK_LAYERS)
print("LM head weight:", tuple(model.lm_head.weight.shape))

VOCAB_SIZE = model.lm_head.weight.shape[0]
HIDDEN_DIM = model.lm_head.weight.shape[1]

if TOPK_MAX > VOCAB_SIZE:
    raise RuntimeError(f"TOPK_MAX={TOPK_MAX} > vocab_size={VOCAB_SIZE}")

W = model.lm_head.weight
W_float_cpu = W.detach().float().cpu()

# ============================================================
# HELPERS
# ============================================================

def batch_iter(xs, batch_size):
    for i in range(0, len(xs), batch_size):
        yield i, xs[i:i + batch_size]


def pairwise_cosine_vector(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 3:
        return np.array([], dtype=np.float64)

    d = cosine_distances(x)
    iu = np.triu_indices_from(d, k=1)
    return d[iu]


def dist_spearman(x, y):
    dx = pairwise_cosine_vector(x)
    dy = pairwise_cosine_vector(y)

    if len(dx) == 0 or len(dy) == 0:
        return np.nan

    if np.std(dx) < 1e-12 or np.std(dy) < 1e-12:
        return np.nan

    r = spearmanr(dx, dy).correlation
    return float(r) if np.isfinite(r) else np.nan


def knn_overlap(x, y, k=10):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n = len(x)
    if n <= k + 1:
        k = max(1, n - 2)

    dx = cosine_distances(x)
    dy = cosine_distances(y)

    np.fill_diagonal(dx, np.inf)
    np.fill_diagonal(dy, np.inf)

    nx = np.argsort(dx, axis=1)[:, :k]
    ny = np.argsort(dy, axis=1)[:, :k]

    scores = []
    for i in range(n):
        scores.append(len(set(nx[i]).intersection(set(ny[i]))) / float(k))

    return float(np.mean(scores))


def linear_cka(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)

    kx = x @ x.T
    ky = y @ y.T

    kx = kx - kx.mean(axis=0, keepdims=True) - kx.mean(axis=1, keepdims=True) + kx.mean()
    ky = ky - ky.mean(axis=0, keepdims=True) - ky.mean(axis=1, keepdims=True) + ky.mean()

    hsic = np.sum(kx * ky)
    norm = np.sqrt(np.sum(kx * kx) * np.sum(ky * ky)) + 1e-12

    val = hsic / norm
    return float(val) if np.isfinite(val) else np.nan


def topology_metrics(hidden, rep, knn_k=10):
    hidden = np.asarray(hidden, dtype=np.float64)
    rep = np.asarray(rep, dtype=np.float64)

    hidden_z = StandardScaler().fit_transform(hidden)
    rep_z = StandardScaler().fit_transform(rep)

    sp = dist_spearman(hidden_z, rep_z)
    knn = knn_overlap(hidden_z, rep_z, k=knn_k)
    cka = linear_cka(hidden_z, rep_z)

    sp_for_topo = 0.0 if (not np.isfinite(sp) or sp < 0) else sp
    knn_for_topo = 0.0 if not np.isfinite(knn) else knn
    cka_for_topo = 0.0 if not np.isfinite(cka) else cka

    topo = float(np.mean([sp_for_topo, knn_for_topo, cka_for_topo]))

    return {
        "DistSpearman": float(sp) if np.isfinite(sp) else np.nan,
        "KNN": float(knn) if np.isfinite(knn) else np.nan,
        "CKA": float(cka) if np.isfinite(cka) else np.nan,
        "Topo": topo,
    }


def compute_center_and_spread_from_ids(token_ids_2d, W_float_cpu):
    ids = torch.tensor(token_ids_2d, dtype=torch.long)

    with torch.no_grad():
        emb = W_float_cpu[ids]
        center = emb.mean(dim=1)

        emb_n = emb / (emb.norm(dim=-1, keepdim=True) + 1e-12)
        center_n = center / (center.norm(dim=-1, keepdim=True) + 1e-12)

        cos = (emb_n * center_n[:, None, :]).sum(dim=-1)
        spread = (1.0 - cos).mean(dim=1, keepdim=True)

    return center.numpy().astype(np.float32), spread.numpy().astype(np.float32)


def make_rep(center, spread, mode):
    if mode == "center":
        return center
    if mode == "center+spread":
        return np.concatenate([center, spread], axis=1)
    raise ValueError(f"Unknown mode: {mode}")


def prompt_token_baseline_ids(input_ids_list, k, vocab_size):
    rows = []

    for ids in input_ids_list:
        clean_ids = [int(x) for x in ids if 0 <= int(x) < vocab_size]

        if len(clean_ids) == 0:
            clean_ids = [0]

        reps = []
        while len(reps) < k:
            reps.extend(clean_ids)

        rows.append(reps[:k])

    return np.asarray(rows, dtype=np.int64)


def safe_mean(x):
    x = pd.to_numeric(x, errors="coerce")
    return float(x.mean()) if len(x) else np.nan


def safe_max(x):
    x = pd.to_numeric(x, errors="coerce")
    return float(x.max()) if len(x) else np.nan


def summarize_scope(df, corpus_name, scope_name):
    return {
        "corpus": corpus_name,
        "scope": scope_name,
        "n_rows": int(len(df)),
        "mean_topo": safe_mean(df["Topo"]) if len(df) else np.nan,
        "max_topo": safe_max(df["Topo"]) if len(df) else np.nan,
        "mean_lift_random": safe_mean(df["LiftTopo_vs_random"]) if len(df) else np.nan,
        "mean_lift_shuffled": safe_mean(df["LiftTopo_vs_shuffled"]) if len(df) else np.nan,
        "mean_lift_input_token": safe_mean(df["LiftTopo_vs_input_token"]) if len(df) else np.nan,
        "max_lift_random": safe_max(df["LiftTopo_vs_random"]) if len(df) else np.nan,
        "max_lift_shuffled": safe_max(df["LiftTopo_vs_shuffled"]) if len(df) else np.nan,
        "max_lift_input_token": safe_max(df["LiftTopo_vs_input_token"]) if len(df) else np.nan,
    }


def add_layer_band(df):
    df = df.copy()
    df["layer_band"] = "unassigned"
    for band, layers in LAYER_BANDS.items():
        df.loc[df["layer"].isin(layers), "layer_band"] = band
    return df

# ============================================================
# FORWARD PASS
# ============================================================

print("\nCollecting hidden states and TopK token ids on real-text corpus...")

hidden_by_layer = {l: [] for l in TRACK_LAYERS}
topk_ids_by_layer = {l: [] for l in TRACK_LAYERS}
all_input_ids_no_pad = []

with torch.no_grad():
    for start_idx, batch_texts in batch_iter(TEXTS, BATCH_SIZE):
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        )

        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)

        for row_ids, row_mask in zip(enc["input_ids"].tolist(), enc["attention_mask"].tolist()):
            ids = [tok for tok, m in zip(row_ids, row_mask) if m == 1]
            all_input_ids_no_pad.append(ids)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        last_pos = attention_mask.sum(dim=1) - 1

        for layer in TRACK_LAYERS:
            h = outputs.hidden_states[layer + 1]
            h_last = h[torch.arange(h.shape[0], device=h.device), last_pos]

            hidden_by_layer[layer].append(
                h_last.detach().float().cpu().numpy().astype(np.float32)
            )

            h_for_logits = h_last.to(dtype=W.dtype)
            logits = torch.matmul(h_for_logits, W.T)

            top_ids = torch.topk(logits, k=TOPK_MAX, dim=-1).indices
            topk_ids_by_layer[layer].append(
                top_ids.detach().cpu().numpy().astype(np.int64)
            )

            del logits, top_ids, h_for_logits

        del outputs, input_ids, attention_mask

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"  processed {min(start_idx + len(batch_texts), len(TEXTS))}/{len(TEXTS)} texts")

for layer in TRACK_LAYERS:
    hidden_by_layer[layer] = np.concatenate(hidden_by_layer[layer], axis=0)
    topk_ids_by_layer[layer] = np.concatenate(topk_ids_by_layer[layer], axis=0)

print("\nCollected:")
print("  hidden shape example:", hidden_by_layer[TRACK_LAYERS[0]].shape)
print("  topk ids shape example:", topk_ids_by_layer[TRACK_LAYERS[0]].shape)

# ============================================================
# METRIC COMPUTATION
# ============================================================

print("\nComputing topology metrics on real-text corpus...")

metrics_rows = []
rng = np.random.default_rng(RANDOM_SEED)

for layer in TRACK_LAYERS:
    H = hidden_by_layer[layer]
    top_ids_max = topk_ids_by_layer[layer]

    for k in TOPK_LIST:
        real_ids = top_ids_max[:, :k]

        random_ids = rng.integers(
            low=0,
            high=VOCAB_SIZE,
            size=real_ids.shape,
            dtype=np.int64,
        )

        perm = rng.permutation(len(real_ids))
        shuffled_ids = real_ids[perm]

        input_ids_baseline = prompt_token_baseline_ids(
            all_input_ids_no_pad,
            k,
            VOCAB_SIZE,
        )

        ids_by_baseline = {
            "real_topk": real_ids,
            "random": random_ids,
            "shuffled_topk": shuffled_ids,
            "input_token": input_ids_baseline,
        }

        for baseline_name, ids in ids_by_baseline.items():
            center, spread = compute_center_and_spread_from_ids(ids, W_float_cpu)

            for mode in MODES:
                rep = make_rep(center, spread, mode)
                m = topology_metrics(H, rep, knn_k=KNN_K)

                row = {
                    "layer": layer,
                    "layer_name": f"L{layer}",
                    "topk": k,
                    "mode": mode,
                    "baseline": baseline_name,
                    "corpus": "real_text",
                    "n_samples": len(TEXTS),
                }
                row.update(m)
                metrics_rows.append(row)

    print(f"  layer L{layer} done")

metrics_df = pd.DataFrame(metrics_rows)

random_lookup = (
    metrics_df[metrics_df["baseline"] == "random"]
    .set_index(["layer", "topk", "mode"])["Topo"]
    .to_dict()
)

shuffled_lookup = (
    metrics_df[metrics_df["baseline"] == "shuffled_topk"]
    .set_index(["layer", "topk", "mode"])["Topo"]
    .to_dict()
)

input_lookup = (
    metrics_df[metrics_df["baseline"] == "input_token"]
    .set_index(["layer", "topk", "mode"])["Topo"]
    .to_dict()
)

def get_lookup_val(row, lookup):
    return lookup.get((row["layer"], row["topk"], row["mode"]), np.nan)

metrics_df["RandTopo"] = metrics_df.apply(lambda r: get_lookup_val(r, random_lookup), axis=1)
metrics_df["ShuffleTopo"] = metrics_df.apply(lambda r: get_lookup_val(r, shuffled_lookup), axis=1)
metrics_df["InputTokenTopo"] = metrics_df.apply(lambda r: get_lookup_val(r, input_lookup), axis=1)

metrics_df["LiftTopo_vs_random"] = metrics_df["Topo"] - metrics_df["RandTopo"]
metrics_df["LiftTopo_vs_shuffled"] = metrics_df["Topo"] - metrics_df["ShuffleTopo"]
metrics_df["LiftTopo_vs_input_token"] = metrics_df["Topo"] - metrics_df["InputTokenTopo"]

# ============================================================
# REAL-TEXT SUMMARIES
# ============================================================

real_df = metrics_df[metrics_df["baseline"] == "real_topk"].copy()

summary_by_topk_mode = (
    real_df
    .groupby(["topk", "mode"], as_index=False)
    .agg(
        MeanSp=("DistSpearman", "mean"),
        MeanKNN=("KNN", "mean"),
        MeanCKA=("CKA", "mean"),
        MeanTopo=("Topo", "mean"),
        MeanRandTopo=("RandTopo", "mean"),
        MeanShuffleTopo=("ShuffleTopo", "mean"),
        MeanInputTokenTopo=("InputTokenTopo", "mean"),
        MeanLiftRandom=("LiftTopo_vs_random", "mean"),
        MeanLiftShuffled=("LiftTopo_vs_shuffled", "mean"),
        MeanLiftInputToken=("LiftTopo_vs_input_token", "mean"),
    )
    .sort_values(["mode", "topk"])
)

layerwise_best = (
    real_df
    .sort_values("Topo", ascending=False)
    .groupby("layer", as_index=False)
    .first()
    .sort_values("layer")
)

global_best = real_df.sort_values("Topo", ascending=False).head(40).copy()

nonfinal = real_df[real_df["layer"] != FINAL_LAYER].copy()
final = real_df[real_df["layer"] == FINAL_LAYER].copy()

nonfinal_summary_rows = []
nonfinal_summary_rows.append(summarize_scope(real_df, "real_text", "all_layers"))
nonfinal_summary_rows.append(summarize_scope(nonfinal, "real_text", "nonfinal_L0_26"))
nonfinal_summary_rows.append(summarize_scope(final, "real_text", "final_L27"))

for mode in sorted(real_df["mode"].unique()):
    nonfinal_summary_rows.append(
        summarize_scope(real_df[real_df["mode"] == mode], "real_text", f"all_layers_{mode}")
    )
    nonfinal_summary_rows.append(
        summarize_scope(nonfinal[nonfinal["mode"] == mode], "real_text", f"nonfinal_{mode}")
    )
    nonfinal_summary_rows.append(
        summarize_scope(final[final["mode"] == mode], "real_text", f"final_{mode}")
    )

nonfinal_summary = pd.DataFrame(nonfinal_summary_rows)

real_banded = add_layer_band(real_df)

layer_band_rows = []
for band in LAYER_BANDS.keys():
    sub = real_banded[real_banded["layer_band"] == band]
    layer_band_rows.append(summarize_scope(sub, "real_text", band))

    for mode in sorted(real_df["mode"].unique()):
        layer_band_rows.append(
            summarize_scope(
                sub[sub["mode"] == mode],
                "real_text",
                f"{band}__{mode}",
            )
        )

layer_band_summary = pd.DataFrame(layer_band_rows)

scaling_nonfinal = (
    nonfinal
    .groupby(["topk", "mode"], as_index=False)
    .agg(
        MeanSp=("DistSpearman", "mean"),
        MeanKNN=("KNN", "mean"),
        MeanCKA=("CKA", "mean"),
        MeanTopo=("Topo", "mean"),
        MeanRandTopo=("RandTopo", "mean"),
        MeanShuffleTopo=("ShuffleTopo", "mean"),
        MeanInputTokenTopo=("InputTokenTopo", "mean"),
        MeanLiftRandom=("LiftTopo_vs_random", "mean"),
        MeanLiftShuffled=("LiftTopo_vs_shuffled", "mean"),
        MeanLiftInputToken=("LiftTopo_vs_input_token", "mean"),
    )
    .sort_values(["mode", "topk"])
)

scaling_checks = []
for mode in sorted(scaling_nonfinal["mode"].unique()):
    sub = scaling_nonfinal[scaling_nonfinal["mode"] == mode].sort_values("topk")
    vals = sub["MeanTopo"].values
    topks = sub["topk"].values

    if len(vals) >= 2:
        diffs = np.diff(vals)
        positive_steps = int((diffs > 0).sum())
        total_steps = int(len(diffs))
        positive_step_ratio = positive_steps / total_steps
        overall_gain = float(vals[-1] - vals[0])
    else:
        positive_steps = 0
        total_steps = 0
        positive_step_ratio = np.nan
        overall_gain = np.nan

    scaling_checks.append({
        "corpus": "real_text",
        "mode": mode,
        "min_topk": int(topks[0]) if len(topks) else np.nan,
        "max_topk": int(topks[-1]) if len(topks) else np.nan,
        "mean_topo_at_min_topk": float(vals[0]) if len(vals) else np.nan,
        "mean_topo_at_max_topk": float(vals[-1]) if len(vals) else np.nan,
        "overall_gain": overall_gain,
        "positive_adjacent_steps": positive_steps,
        "total_adjacent_steps": total_steps,
        "positive_step_ratio": positive_step_ratio,
    })

scaling_check_df = pd.DataFrame(scaling_checks)

# ============================================================
# SAVE REAL-TEXT AUDIT OUTPUTS
# ============================================================

metrics_df.to_csv(REALTEXT_SAVE_DIR / "neighborhood_audit1_metrics.csv", index=False)
summary_by_topk_mode.to_csv(REALTEXT_SAVE_DIR / "neighborhood_audit1_summary_by_topk_mode.csv", index=False)
layerwise_best.to_csv(REALTEXT_SAVE_DIR / "neighborhood_audit1_layerwise_best.csv", index=False)
global_best.to_csv(REALTEXT_SAVE_DIR / "neighborhood_audit1_global_best.csv", index=False)

nonfinal_summary.to_csv(REALTEXT_SAVE_DIR / "neighborhood_audit1c_realtext_nonfinal_summary.csv", index=False)
layer_band_summary.to_csv(REALTEXT_SAVE_DIR / "neighborhood_audit1c_realtext_layer_band_summary.csv", index=False)
scaling_nonfinal.to_csv(REALTEXT_SAVE_DIR / "neighborhood_audit1c_realtext_scaling_nonfinal.csv", index=False)
scaling_check_df.to_csv(REALTEXT_SAVE_DIR / "neighborhood_audit1c_realtext_scaling_check.csv", index=False)

# ============================================================
# COMPARISON WITH DEFAULT TEMPLATE
# ============================================================

print("\nComparing default_template vs real_text...")

comparison_rows = []
band_comparison_rows = []
scaling_comparison_rows = []

corpus_dirs = {
    "default_template": DEFAULT_TEMPLATE_DIR,
    "real_text": REALTEXT_SAVE_DIR,
}

for corpus_name, folder in corpus_dirs.items():
    m_path = folder / "neighborhood_audit1_metrics.csv"
    if not m_path.exists():
        comparison_rows.append({
            "corpus": corpus_name,
            "status": "missing_metrics",
            "folder": str(folder),
        })
        continue

    m = pd.read_csv(m_path)
    m = m[m["baseline"] == "real_topk"].copy()

    for c in ["layer", "topk"]:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce")

    m_nonfinal = m[m["layer"] != FINAL_LAYER]
    m_final = m[m["layer"] == FINAL_LAYER]

    comparison_rows.append({
        "corpus": corpus_name,
        "status": "ok",
        "folder": str(folder),
        "n_rows": int(len(m)),
        "mean_topo_all": safe_mean(m["Topo"]),
        "mean_topo_nonfinal": safe_mean(m_nonfinal["Topo"]),
        "mean_topo_final": safe_mean(m_final["Topo"]),
        "best_topo_all": safe_max(m["Topo"]),
        "best_topo_nonfinal": safe_max(m_nonfinal["Topo"]),
        "best_topo_final": safe_max(m_final["Topo"]),
        "nonfinal_to_final_best_ratio": (
            safe_max(m_nonfinal["Topo"]) / safe_max(m_final["Topo"])
            if safe_max(m_final["Topo"]) > 0
            else np.nan
        ),
        "mean_lift_random_nonfinal": safe_mean(m_nonfinal["LiftTopo_vs_random"]),
        "mean_lift_shuffled_nonfinal": safe_mean(m_nonfinal["LiftTopo_vs_shuffled"]),
        "mean_lift_input_nonfinal": safe_mean(m_nonfinal["LiftTopo_vs_input_token"]),
    })

    m_banded = add_layer_band(m)
    for band in LAYER_BANDS.keys():
        sub = m_banded[m_banded["layer_band"] == band]
        band_comparison_rows.append({
            "corpus": corpus_name,
            "layer_band": band,
            "n_rows": int(len(sub)),
            "mean_topo": safe_mean(sub["Topo"]),
            "max_topo": safe_max(sub["Topo"]),
            "mean_lift_random": safe_mean(sub["LiftTopo_vs_random"]),
            "mean_lift_shuffled": safe_mean(sub["LiftTopo_vs_shuffled"]),
            "mean_lift_input": safe_mean(sub["LiftTopo_vs_input_token"]),
        })

    sc = (
        m_nonfinal
        .groupby(["topk", "mode"], as_index=False)
        .agg(
            MeanTopo=("Topo", "mean"),
            MeanLiftRandom=("LiftTopo_vs_random", "mean"),
            MeanLiftShuffled=("LiftTopo_vs_shuffled", "mean"),
            MeanLiftInput=("LiftTopo_vs_input_token", "mean"),
        )
    )

    for _, row in sc.iterrows():
        scaling_comparison_rows.append({
            "corpus": corpus_name,
            "topk": int(row["topk"]),
            "mode": row["mode"],
            "MeanTopo": float(row["MeanTopo"]),
            "MeanLiftRandom": float(row["MeanLiftRandom"]),
            "MeanLiftShuffled": float(row["MeanLiftShuffled"]),
            "MeanLiftInput": float(row["MeanLiftInput"]),
        })

corpus_comparison = pd.DataFrame(comparison_rows)
layer_band_comparison = pd.DataFrame(band_comparison_rows)
scaling_comparison = pd.DataFrame(scaling_comparison_rows)

# ============================================================
# ROBUSTNESS DECISION
# ============================================================

rt_row = corpus_comparison[
    (corpus_comparison["corpus"] == "real_text")
    & (corpus_comparison["status"] == "ok")
].iloc[0]

decision_rows = []

decision_rows.append({
    "criterion": "realtext_nonfinal_mean_topo_above_threshold",
    "value": rt_row["mean_topo_nonfinal"],
    "threshold": STRONG_TOPO_THRESHOLD,
    "pass": bool(rt_row["mean_topo_nonfinal"] >= STRONG_TOPO_THRESHOLD),
    "interpretation": "Real-text non-final layers preserve hidden topology.",
})

decision_rows.append({
    "criterion": "realtext_nonfinal_lift_vs_random_above_threshold",
    "value": rt_row["mean_lift_random_nonfinal"],
    "threshold": STRONG_LIFT_RANDOM_THRESHOLD,
    "pass": bool(rt_row["mean_lift_random_nonfinal"] >= STRONG_LIFT_RANDOM_THRESHOLD),
    "interpretation": "Real-text non-final signal is not random vocabulary embedding effect.",
})

decision_rows.append({
    "criterion": "realtext_nonfinal_lift_vs_shuffled_above_threshold",
    "value": rt_row["mean_lift_shuffled_nonfinal"],
    "threshold": STRONG_LIFT_SHUFFLED_THRESHOLD,
    "pass": bool(rt_row["mean_lift_shuffled_nonfinal"] >= STRONG_LIFT_SHUFFLED_THRESHOLD),
    "interpretation": "Real-text non-final signal is sample-specific, not global TopK bias.",
})

decision_rows.append({
    "criterion": "realtext_nonfinal_lift_vs_input_above_threshold",
    "value": rt_row["mean_lift_input_nonfinal"],
    "threshold": STRONG_LIFT_INPUT_THRESHOLD,
    "pass": bool(rt_row["mean_lift_input_nonfinal"] >= STRONG_LIFT_INPUT_THRESHOLD),
    "interpretation": "Real-text non-final signal is not merely lexical prompt overlap.",
})

decision_rows.append({
    "criterion": "realtext_nonfinal_best_not_collapsed_vs_final",
    "value": rt_row["nonfinal_to_final_best_ratio"],
    "threshold": NONFINAL_RATIO_THRESHOLD,
    "pass": bool(rt_row["nonfinal_to_final_best_ratio"] >= NONFINAL_RATIO_THRESHOLD),
    "interpretation": "Real-text non-final best does not collapse relative to final layer.",
})

# Scaling pass.
scaling_pass = bool(
    len(scaling_check_df) > 0
    and (scaling_check_df["overall_gain"] > 0).all()
    and (scaling_check_df["positive_step_ratio"] >= 0.70).all()
)

decision_rows.append({
    "criterion": "realtext_nonfinal_topk_scaling_positive",
    "value": float(scaling_check_df["positive_step_ratio"].min()) if len(scaling_check_df) else np.nan,
    "threshold": 0.70,
    "pass": scaling_pass,
    "interpretation": "TopK scaling persists on real-text corpus after excluding L27.",
})

robustness_decision = pd.DataFrame(decision_rows)
overall_pass = bool(robustness_decision["pass"].all())

robustness_decision.loc[len(robustness_decision)] = {
    "criterion": "overall_neighborhood_audit1c_decision",
    "value": np.nan,
    "threshold": np.nan,
    "pass": overall_pass,
    "interpretation": (
        "PASS: Axiom-1 evidence survives corpus shift to real natural texts."
        if overall_pass
        else "MIXED: inspect failed criteria before claiming corpus robustness."
    ),
}

# ============================================================
# PLOTS
# ============================================================

# 1. Corpus comparison bar chart.
plt.figure(figsize=(9, 5))
plot_df = corpus_comparison[corpus_comparison["status"] == "ok"].copy()
x = np.arange(len(plot_df))
width = 0.25

plt.bar(x - width, plot_df["mean_topo_nonfinal"], width, label="nonfinal mean")
plt.bar(x, plot_df["mean_topo_final"], width, label="final mean")
plt.bar(x + width, plot_df["best_topo_nonfinal"], width, label="nonfinal best")

plt.xticks(x, plot_df["corpus"], rotation=20)
plt.ylabel("Topo")
plt.title("Neighborhood-Audit-1c: corpus robustness summary")
plt.legend()
plt.tight_layout()
plt.savefig(SAVE_DIR / "neighborhood_audit1c_corpus_topo_comparison.png", dpi=180)
plt.close()

# 2. Layer band comparison.
band_pivot = layer_band_comparison.pivot_table(
    index="layer_band",
    columns="corpus",
    values="mean_topo",
    aggfunc="mean",
)

plt.figure(figsize=(10, 5))
band_pivot.plot(kind="bar", ax=plt.gca())
plt.ylabel("Mean Topo")
plt.title("Neighborhood-Audit-1c: layer-band mean Topo by corpus")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(SAVE_DIR / "neighborhood_audit1c_layer_band_comparison.png", dpi=180)
plt.close()

# 3. Scaling comparison by mode.
for mode in sorted(scaling_comparison["mode"].unique()):
    sub = scaling_comparison[scaling_comparison["mode"] == mode].copy()

    plt.figure(figsize=(8, 5))

    for corpus in sorted(sub["corpus"].unique()):
        ss = sub[sub["corpus"] == corpus].sort_values("topk")
        plt.plot(ss["topk"], ss["MeanTopo"], marker="o", label=corpus)

    plt.xscale("log")
    plt.xlabel("TopK")
    plt.ylabel("Non-final Mean Topo")
    plt.title(f"Neighborhood-Audit-1c: non-final scaling comparison ({mode})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"neighborhood_audit1c_scaling_comparison_{mode}.png", dpi=180)
    plt.close()

# 4. Real-text heatmap.
rt_heat = real_df.pivot_table(index="layer", columns="topk", values="Topo", aggfunc="mean")
plt.figure(figsize=(9, 6))
plt.imshow(rt_heat.values, aspect="auto")
plt.colorbar(label="Topo")
plt.xticks(range(len(rt_heat.columns)), rt_heat.columns, rotation=45)
plt.yticks(range(len(rt_heat.index)), rt_heat.index)
plt.xlabel("TopK")
plt.ylabel("Layer")
plt.title("Neighborhood-Audit-1c: real-text Topo heatmap")
plt.tight_layout()
plt.savefig(SAVE_DIR / "neighborhood_audit1c_realtext_topo_heatmap.png", dpi=180)
plt.close()

# ============================================================
# SAVE COMPARISON OUTPUTS
# ============================================================

corpus_comparison_path = SAVE_DIR / "neighborhood_audit1c_corpus_comparison.csv"
layer_band_comparison_path = SAVE_DIR / "neighborhood_audit1c_layer_band_comparison.csv"
scaling_comparison_path = SAVE_DIR / "neighborhood_audit1c_scaling_comparison.csv"
robustness_decision_path = SAVE_DIR / "neighborhood_audit1c_robustness_decision.csv"
nonfinal_summary_path = SAVE_DIR / "neighborhood_audit1c_realtext_nonfinal_summary.csv"
layer_band_summary_path = SAVE_DIR / "neighborhood_audit1c_realtext_layer_band_summary.csv"
scaling_check_path = SAVE_DIR / "neighborhood_audit1c_realtext_scaling_check.csv"

corpus_comparison.to_csv(corpus_comparison_path, index=False)
layer_band_comparison.to_csv(layer_band_comparison_path, index=False)
scaling_comparison.to_csv(scaling_comparison_path, index=False)
robustness_decision.to_csv(robustness_decision_path, index=False)
nonfinal_summary.to_csv(nonfinal_summary_path, index=False)
layer_band_summary.to_csv(layer_band_summary_path, index=False)
scaling_check_df.to_csv(scaling_check_path, index=False)

# ============================================================
# PRINT RESULTS
# ============================================================

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 240)

print("\n============================================================")
print("Neighborhood-Audit-1c Results")
print("============================================================\n")

print("Real-text nonfinal summary:")
print(nonfinal_summary.to_string(index=False))

print("\nReal-text TopK scaling check:")
print(scaling_check_df.to_string(index=False))

print("\nCorpus comparison:")
print(corpus_comparison.to_string(index=False))

print("\nLayer-band comparison:")
print(layer_band_comparison.to_string(index=False))

print("\nRobustness decision:")
print(robustness_decision.to_string(index=False))

print("\nReal-text global best Top 20:")
view_cols = [
    "layer",
    "layer_name",
    "topk",
    "mode",
    "DistSpearman",
    "KNN",
    "CKA",
    "Topo",
    "RandTopo",
    "ShuffleTopo",
    "InputTokenTopo",
    "LiftTopo_vs_random",
    "LiftTopo_vs_shuffled",
    "LiftTopo_vs_input_token",
]
existing_view_cols = [c for c in view_cols if c in global_best.columns]
print(global_best[existing_view_cols].head(20).to_string(index=False))

print("\nSaved real-text audit outputs:")
for p in [
    REALTEXT_SAVE_DIR / "neighborhood_audit1_metrics.csv",
    REALTEXT_SAVE_DIR / "neighborhood_audit1_summary_by_topk_mode.csv",
    REALTEXT_SAVE_DIR / "neighborhood_audit1_layerwise_best.csv",
    REALTEXT_SAVE_DIR / "neighborhood_audit1_global_best.csv",
    REALTEXT_SAVE_DIR / "neighborhood_audit1c_realtext_prompts.txt",
]:
    print(" ", p)

print("\nSaved 1c comparison outputs:")
for p in [
    corpus_comparison_path,
    layer_band_comparison_path,
    scaling_comparison_path,
    robustness_decision_path,
    nonfinal_summary_path,
    layer_band_summary_path,
    scaling_check_path,
]:
    print(" ", p)

print("\nSaved plots:")
for p in sorted(SAVE_DIR.glob("*.png")):
    print(" ", p)

print("\n============================================================")
print("Neighborhood-Audit-1c Interpretation Guide")
print("============================================================\n")

print("Core question:")
print("  Does Axiom-1 survive corpus shift from default templates to real natural texts?")
print()
print("Strong positive result if:")
print("  1. overall_neighborhood_audit1c_decision = PASS")
print("  2. real_text nonfinal mean Topo remains high")
print("  3. real_text lift vs random / shuffled / input-token remains positive and large")
print("  4. TopK scaling remains positive on real_text")
print("  5. nonfinal best does not collapse relative to final best")
print()
print("If PASS:")
print("  Axiom-1 evidence is no longer just default-template behavior.")
print("  It becomes corpus-robust evidence for vocabulary-induced semantic neighborhoods.")
print()
print("If MIXED:")
print("  Inspect which criterion failed.")
print("  Common failure patterns:")
print("    - real_text too heterogeneous or too few samples")
print("    - input-token baseline too high due to line-level lexical overlap")
print("    - final layer dominates more strongly than in default_template")
print()
print("Done.")