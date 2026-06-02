# ============================================================
# Neighborhood-Audit-1
# TopK Identity Topology Replication and Hardening
#
# Goal:
#   Solidify DHRF Axiom 1:
#
#       H_l <-> N_k(H_l) = TopK(H_l W^T)
#
#   We test whether TopK token identity sets preserve hidden-state topology.
#
# Key constraints:
#   - Use TopK token ids only.
#   - Do NOT use logit values.
#   - Do NOT use probabilities.
#   - Do NOT use rank scores.
#
# Representations:
#   1. center:
#        C_k(H_l) = mean embedding of TopK token ids
#
#   2. center+spread:
#        [C_k(H_l), r_k(H_l)]
#        where r_k is mean cosine spread of TopK embeddings around C_k.
#
# Baselines:
#   1. random:
#        random token ids per sample
#
#   2. shuffled_topk:
#        real TopK neighborhoods shuffled across samples
#
#   3. input_token:
#        prompt token ids repeated/truncated to k,
#        testing whether the effect is merely lexical prompt overlap
#
# Metrics:
#   - DistSpearman: Spearman correlation between pairwise cosine distances
#   - KNN overlap: average kNN overlap between hidden space and neighborhood space
#   - Linear CKA
#   - Topo: mean(DistSpearman_clipped, KNN, CKA)
#   - LiftTopo: Topo(real) - Topo(random)
#
# Outputs:
#   ./neighborhood_audit1_outputs/
#
# Main files:
#   neighborhood_audit1_metrics.csv
#   neighborhood_audit1_summary_by_topk_mode.csv
#   neighborhood_audit1_layerwise_best.csv
#   neighborhood_audit1_global_best.csv
#   neighborhood_audit1_prompts.txt
# ============================================================

import os
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

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

SAVE_DIR = Path("./neighborhood_audit1_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# If None, all transformer block outputs are used.
# For Qwen2.5-1.5B-Instruct this is usually 0..27.
TRACK_LAYERS = None

TOPK_LIST = [20, 50, 100, 200, 500, 1000, 2000, 5000]
TOPK_MAX = max(TOPK_LIST)

BATCH_SIZE = 4
MAX_LEN = 256

KNN_K = 10

# Use built-in prompts unless you provide a text file.
# If a file exists, one non-empty line = one sample.
TEXT_FILE = None
# Example:
# TEXT_FILE = r"C:\Users\ZH\Desktop\AGI\data\natural_texts_240.txt"

# To save memory, only these modes are computed.
MODES = ["center", "center+spread"]

BASELINES = ["real_topk", "random", "shuffled_topk", "input_token"]

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
# TEXT DATA
# ============================================================

def make_default_texts():
    """
    Built-in 240-ish semantically diverse natural texts.
    These are not meant to be a benchmark dataset.
    They are meant to produce diverse hidden states.
    Replace with your own 200+ natural semantic texts if available.
    """
    subjects = [
        "a biologist studying coral reefs",
        "a historian comparing ancient trade routes",
        "a software engineer debugging a distributed system",
        "a physicist explaining thermal equilibrium",
        "a doctor reviewing symptoms before diagnosis",
        "a teacher designing a lesson for children",
        "an economist modeling inflation expectations",
        "a lawyer interpreting a contract clause",
        "a musician analyzing a violin concerto",
        "a chef adjusting a recipe after tasting it",
        "an architect planning a public library",
        "a farmer observing soil moisture after rain",
        "a linguist studying metaphor in poetry",
        "a mathematician proving a lemma",
        "a psychologist analyzing memory formation",
        "a climate scientist studying monsoon patterns",
        "a pilot checking instruments before landing",
        "a journalist verifying conflicting sources",
        "a gardener pruning a young fruit tree",
        "a robotics engineer tuning a navigation policy",
    ]

    actions = [
        "noticed that a small local change can alter the behavior of the whole system",
        "distinguished surface variation from deeper structural change",
        "mapped several observations into a smaller set of stable relations",
        "found that a misleading cue can pull attention away from the correct explanation",
        "explained why the same object can appear different under a new coordinate system",
        "tracked how uncertainty gradually collapses into a decision",
        "compared a stable pattern with a disrupted pattern",
        "identified a boundary where the system changes regime",
        "showed that a redundant signal can preserve meaning while changing form",
        "tested whether a relation remains invariant under renaming",
        "summarized a large context into a compact relational description",
        "observed that a weak perturbation changes local details but not the global structure",
    ]

    domains = [
        "in a laboratory notebook",
        "during a classroom discussion",
        "while reading a long technical report",
        "in a noisy field experiment",
        "inside a simulated environment",
        "during a careful peer review",
        "while comparing two alternative hypotheses",
        "in a planning meeting",
        "while reconstructing a causal chain",
        "during a failure analysis",
        "when translating an abstract theory into a practical rule",
        "while checking whether a conclusion follows from the evidence",
    ]

    conclusions = [
        "The important point is not the isolated object, but the relation that remains stable.",
        "A useful representation should preserve neighborhood structure under mild perturbation.",
        "If the local coordinates move but the relational pattern remains, the system may still be stable.",
        "A good diagnostic separates surface change from structural change.",
        "The same evidence can support different outputs when the closure relation is broken.",
        "A hidden state may be better understood through the semantic neighborhood it induces.",
        "The transition from uncertainty to commitment often appears before the final answer.",
        "The simplest useful invariant is the one that survives renaming, paraphrase, and redundancy.",
    ]

    texts = []
    for s in subjects:
        for a in actions:
            d = random.choice(domains)
            c = random.choice(conclusions)
            texts.append(f"{s.capitalize()} {a} {d}. {c}")

    # Deterministic shuffle.
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(texts)

    return texts[:240]


def load_texts():
    if TEXT_FILE is not None and Path(TEXT_FILE).exists():
        lines = Path(TEXT_FILE).read_text(encoding="utf-8").splitlines()
        texts = [x.strip() for x in lines if x.strip()]
        if len(texts) < 32:
            raise RuntimeError("TEXT_FILE has too few non-empty lines. Use at least 32, preferably 200+.")
        return texts

    return make_default_texts()


TEXTS = load_texts()

print("\n============================================================")
print("Neighborhood-Audit-1 Input Summary")
print("============================================================")
print("Num texts:", len(TEXTS))
print("Device:", DEVICE)
print("TopK list:", TOPK_LIST)
print("Modes:", MODES)
print("Baselines:", BASELINES)

with open(SAVE_DIR / "neighborhood_audit1_prompts.txt", "w", encoding="utf-8") as f:
    for i, t in enumerate(TEXTS):
        f.write(f"[{i:04d}] {t}\n")

# ============================================================
# MODEL
# ============================================================

if "你的snapshot目录" in MODEL_PATH:
    raise RuntimeError(
        "Please replace MODEL_PATH with your actual local HuggingFace snapshot path."
    )

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

# Keep LM head / token embedding matrix on model device.
# We use lm_head weights as vocabulary anchors.
W = model.lm_head.weight
W_float = W.detach().float()

# ============================================================
# HELPERS
# ============================================================

def batch_iter(xs, batch_size):
    for i in range(0, len(xs), batch_size):
        yield i, xs[i:i + batch_size]


def l2_normalize_np(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float64)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def pairwise_cosine_vector(x):
    """
    Return upper-triangular pairwise cosine distance vector.
    """
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
    """
    Linear CKA with centered Gram matrices.
    """
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
    """
    Compare hidden geometry with neighborhood representation geometry.
    """
    hidden = np.asarray(hidden, dtype=np.float64)
    rep = np.asarray(rep, dtype=np.float64)

    # Standardize feature dimensions for fair comparison.
    hidden_z = StandardScaler().fit_transform(hidden)
    rep_z = StandardScaler().fit_transform(rep)

    sp = dist_spearman(hidden_z, rep_z)
    knn = knn_overlap(hidden_z, rep_z, k=knn_k)
    cka = linear_cka(hidden_z, rep_z)

    # Topo is explicitly defined here.
    # Spearman can be negative, so we clip only for the composite score.
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
    """
    token_ids_2d: numpy array [n, k]
    W_float_cpu: torch tensor [vocab, dim] on CPU float32
    Returns:
      center: [n, d] numpy
      spread: [n, 1] numpy
    """
    ids = torch.tensor(token_ids_2d, dtype=torch.long)

    with torch.no_grad():
        emb = W_float_cpu[ids]  # n,k,d
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
    """
    For each sample, use its own prompt token ids, repeated/truncated to k.
    This baseline tests whether lexical prompt overlap alone explains the effect.
    """
    rows = []

    for ids in input_ids_list:
        clean_ids = [int(x) for x in ids if int(x) >= 0 and int(x) < vocab_size]

        if len(clean_ids) == 0:
            clean_ids = [0]

        reps = []
        while len(reps) < k:
            reps.extend(clean_ids)

        rows.append(reps[:k])

    return np.asarray(rows, dtype=np.int64)


# ============================================================
# FORWARD PASS: COLLECT HIDDEN STATES AND TOPK IDS
# ============================================================

print("\nCollecting hidden states and TopK token ids...")

# Store layer -> hidden matrix [n,d]
hidden_by_layer = {l: [] for l in TRACK_LAYERS}

# Store layer -> topk ids [n, TOPK_MAX]
topk_ids_by_layer = {l: [] for l in TRACK_LAYERS}

# Store original prompt input ids for input-token baseline.
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

        # Non-pad token ids for input baseline.
        for row_ids, row_mask in zip(enc["input_ids"].tolist(), enc["attention_mask"].tolist()):
            ids = [tok for tok, m in zip(row_ids, row_mask) if m == 1]
            all_input_ids_no_pad.append(ids)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # Last non-pad token position.
        last_pos = attention_mask.sum(dim=1) - 1

        for layer in TRACK_LAYERS:
            # Transformer block output layer l corresponds to hidden_states[l+1].
            h = outputs.hidden_states[layer + 1]
            h_last = h[torch.arange(h.shape[0], device=h.device), last_pos]

            hidden_by_layer[layer].append(h_last.detach().float().cpu().numpy().astype(np.float32))

            # Compute logits only to get TopK ids.
            # We do NOT store or use logit values.
            h_for_logits = h_last.to(dtype=W.dtype)
            logits = torch.matmul(h_for_logits, W.T)

            top_ids = torch.topk(logits, k=TOPK_MAX, dim=-1).indices
            topk_ids_by_layer[layer].append(top_ids.detach().cpu().numpy().astype(np.int64))

            del logits, top_ids, h_for_logits

        del outputs, input_ids, attention_mask

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"  processed {min(start_idx + len(batch_texts), len(TEXTS))}/{len(TEXTS)} texts")

for layer in TRACK_LAYERS:
    hidden_by_layer[layer] = np.concatenate(hidden_by_layer[layer], axis=0)
    topk_ids_by_layer[layer] = np.concatenate(topk_ids_by_layer[layer], axis=0)

W_float_cpu = W.detach().float().cpu()

print("\nCollected:")
print("  hidden shape example:", hidden_by_layer[TRACK_LAYERS[0]].shape)
print("  topk ids shape example:", topk_ids_by_layer[TRACK_LAYERS[0]].shape)

# ============================================================
# METRIC COMPUTATION
# ============================================================

print("\nComputing topology metrics...")

metrics_rows = []
rng = np.random.default_rng(RANDOM_SEED)

for layer in TRACK_LAYERS:
    H = hidden_by_layer[layer]
    top_ids_max = topk_ids_by_layer[layer]

    for k in TOPK_LIST:
        real_ids = top_ids_max[:, :k]

        # Baseline ids.
        random_ids = rng.integers(low=0, high=VOCAB_SIZE, size=real_ids.shape, dtype=np.int64)

        perm = rng.permutation(len(real_ids))
        shuffled_ids = real_ids[perm]

        input_ids_baseline = prompt_token_baseline_ids(all_input_ids_no_pad, k, VOCAB_SIZE)

        ids_by_baseline = {
            "real_topk": real_ids,
            "random": random_ids,
            "shuffled_topk": shuffled_ids,
            "input_token": input_ids_baseline,
        }

        reps_cache = {}

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
                    "n_samples": len(TEXTS),
                }
                row.update(m)
                metrics_rows.append(row)

            reps_cache[baseline_name] = (center, spread)

        # Explicit cleanup.
        del reps_cache

    print(f"  layer L{layer} done")

metrics_df = pd.DataFrame(metrics_rows)

# Compute lift relative to random baseline for the same layer/topk/mode.
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
# SUMMARIES
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

global_best = real_df.sort_values("Topo", ascending=False).head(20).copy()

# ============================================================
# OPTIONAL PLOTS
# ============================================================

try:
    import matplotlib.pyplot as plt

    # Plot 1: scaling by topk / mode.
    for mode in MODES:
        sub = summary_by_topk_mode[summary_by_topk_mode["mode"] == mode]
        plt.figure(figsize=(8, 5))
        plt.plot(sub["topk"], sub["MeanTopo"], marker="o", label="real TopK")
        plt.plot(sub["topk"], sub["MeanRandTopo"], marker="o", label="random")
        plt.plot(sub["topk"], sub["MeanShuffleTopo"], marker="o", label="shuffled TopK")
        plt.plot(sub["topk"], sub["MeanInputTokenTopo"], marker="o", label="input-token")
        plt.xscale("log")
        plt.xlabel("TopK")
        plt.ylabel("Mean Topo")
        plt.title(f"Neighborhood-Audit-1 scaling: {mode}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(SAVE_DIR / f"neighborhood_audit1_scaling_{mode}.png", dpi=160)
        plt.close()

    # Plot 2: layerwise Topo for best topk/mode by layer.
    plt.figure(figsize=(10, 5))
    plt.plot(layerwise_best["layer"], layerwise_best["Topo"], marker="o")
    plt.xlabel("Layer")
    plt.ylabel("Best Topo")
    plt.title("Neighborhood-Audit-1 layerwise best Topo")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "neighborhood_audit1_layerwise_best_topo.png", dpi=160)
    plt.close()

except Exception as e:
    print("Plotting skipped due to:", repr(e))

# ============================================================
# SAVE
# ============================================================

metrics_path = SAVE_DIR / "neighborhood_audit1_metrics.csv"
summary_path = SAVE_DIR / "neighborhood_audit1_summary_by_topk_mode.csv"
layerwise_path = SAVE_DIR / "neighborhood_audit1_layerwise_best.csv"
global_best_path = SAVE_DIR / "neighborhood_audit1_global_best.csv"

metrics_df.to_csv(metrics_path, index=False)
summary_by_topk_mode.to_csv(summary_path, index=False)
layerwise_best.to_csv(layerwise_path, index=False)
global_best.to_csv(global_best_path, index=False)

print("\n============================================================")
print("Neighborhood-Audit-1 Results")
print("============================================================\n")

print("Summary by TopK / Mode:")
print(summary_by_topk_mode.to_string(index=False))

print("\nLayerwise Best:")
print(
    layerwise_best[
        [
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
    ].to_string(index=False)
)

print("\nGlobal Best Top 20:")
print(
    global_best[
        [
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
    ].to_string(index=False)
)

print("\nSaved outputs:")
print(" ", metrics_path)
print(" ", summary_path)
print(" ", layerwise_path)
print(" ", global_best_path)
print(" ", SAVE_DIR / "neighborhood_audit1_prompts.txt")
print(" ", SAVE_DIR / "neighborhood_audit1_scaling_center.png")
print(" ", SAVE_DIR / "neighborhood_audit1_scaling_center+spread.png")
print(" ", SAVE_DIR / "neighborhood_audit1_layerwise_best_topo.png")

# ============================================================
# INTERPRETATION GUIDE
# ============================================================

print("\n============================================================")
print("Neighborhood-Audit-1 Interpretation Guide")
print("============================================================\n")

print("Axiom-1 core claim:")
print("  H_l <-> N_k(H_l) = TopK(H_l W^T)")
print()
print("Strong positive evidence if:")
print("  1. real_topk Topo is much higher than random Topo.")
print("  2. real_topk Topo is higher than shuffled_topk Topo.")
print("  3. real_topk Topo is higher than input_token Topo.")
print("  4. Topo increases as TopK grows.")
print("  5. center+spread is equal to or stronger than center.")
print("  6. high Topo persists across many layers, not only final layer.")
print()
print("Interpretation:")
print("  If random is near zero but real_topk is high:")
print("    token identity neighborhoods preserve hidden geometry.")
print()
print("  If shuffled_topk is much lower than real_topk:")
print("    the signal is sample-specific, not just vocabulary-wide token bias.")
print()
print("  If input_token is lower than real_topk:")
print("    the signal is not merely lexical overlap in the prompt.")
print()
print("  If larger TopK performs better:")
print("    small TopK behaves like output candidates, while large TopK behaves like a semantic neighborhood.")
print()
print("  If center+spread improves over center:")
print("    neighborhood geometry contains useful spread/freedom information beyond centroid position.")
print()
print("Failure / caveat patterns:")
print("  If shuffled_topk is close to real_topk:")
print("    token-set distribution may be dominated by global vocabulary bias.")
print()
print("  If input_token is close to real_topk:")
print("    prompt lexical overlap may explain too much of the signal.")
print()
print("  If only final layers work:")
print("    the result may be output-readout geometry rather than internal-state geometry.")
print()
print("Done.")