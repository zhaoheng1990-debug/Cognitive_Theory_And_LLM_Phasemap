# ============================================================
# Information-Audit-1
# TopK Information Fidelity / Compression Efficiency
#
# Question:
#   Is TopK token identity set S_k(H)=TopK(HW^T)
#   an efficient lossy code of hidden geometry?
#
# Outputs:
#   1. fidelity vs k
#   2. code length vs k
#   3. compression efficiency
#   4. effective-rank retention
#   5. random / permuted baselines
# ============================================================

import gc, math, json, random, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ================= CONFIG =================

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"
CORPUS_TXT = r"C:\Users\ZH\Desktop\AGI\data\natural_wiki_texts_450_balanced_diverse_no_meta_tail.txt"

SAVE_DIR = Path("./information_audit_1_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SEED = 42
MAX_SAMPLES = 240
BATCH_SIZE = 8
MAX_LEN = 220

LAYERS = list(range(0, 28))
TOPKS = [20, 50, 100, 200, 500, 1000, 2000, 5000, 8000]
KNN_LIST = [5, 10, 20, 50]
N_RANDOM_SEEDS = 3


# ================= UTILS =================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def row_norm(x, eps=1e-8):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def load_prompts():
    p = Path(CORPUS_TXT)
    if p.exists():
        lines = [x.strip() for x in open(p, encoding="utf-8") if x.strip()]
        random.shuffle(lines)
        return lines[:MAX_SAMPLES]

    print("[WARN] corpus not found, using synthetic prompts.")
    return [
        f"Fact 1: EntityA{i} belongs to GroupB{i}. "
        f"Fact 2: GroupB{i} is located in PlaceC{i}. "
        f"Question: Where is EntityA{i} located?"
        for i in range(MAX_SAMPLES)
    ]


def log2_comb_approx(n, k):
    # Stirling entropy approximation: log2 C(n,k) ≈ n H(k/n)
    p = k / n
    if p <= 0 or p >= 1:
        return 0.0
    H = -p * math.log2(p) - (1-p) * math.log2(1-p)
    return n * H


def effective_rank(X, eps=1e-12):
    X = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    p = s / (s.sum() + eps)
    ent = -np.sum(p * np.log(p + eps))
    return float(np.exp(ent))


def dist_spearman(A, B):
    DA = pairwise_distances(A, metric="cosine")
    DB = pairwise_distances(B, metric="cosine")
    iu = np.triu_indices_from(DA, k=1)
    r = spearmanr(DA[iu], DB[iu]).correlation
    return 0.0 if np.isnan(r) else float(r)


def knn_overlap(A, B, k):
    DA = pairwise_distances(A, metric="cosine")
    DB = pairwise_distances(B, metric="cosine")
    np.fill_diagonal(DA, np.inf)
    np.fill_diagonal(DB, np.inf)
    nnA = np.argsort(DA, axis=1)[:, :k]
    nnB = np.argsort(DB, axis=1)[:, :k]
    return float(np.mean([
        len(set(nnA[i]).intersection(set(nnB[i]))) / k
        for i in range(A.shape[0])
    ]))


def topo_metrics(H, Z):
    out = {}
    out["dist_spearman"] = dist_spearman(H, Z)
    for kk in KNN_LIST:
        out[f"knn_{kk}"] = knn_overlap(H, Z, kk)
    out["topo"] = 0.5 * out["dist_spearman"] + 0.5 * out["knn_10"]
    return out


def get_topk_ids(H, W, k, device):
    Ht = torch.tensor(H, dtype=torch.float32, device=device)
    Wt = torch.tensor(W, dtype=torch.float32, device=device)
    ids_all = []
    with torch.no_grad():
        for s in range(0, H.shape[0], 64):
            logits = Ht[s:s+64] @ Wt.T
            ids = torch.topk(logits, k=k, dim=1).indices.cpu().numpy()
            ids_all.append(ids)
    del Ht, Wt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(ids_all, axis=0)


def center_spread(ids, W_lookup):
    vecs = W_lookup[ids]
    center = row_norm(vecs.mean(axis=1))
    cos = np.einsum("nkd,nd->nk", vecs, center)
    spread = (1.0 - cos).mean(axis=1, keepdims=True)
    spread = (spread - spread.mean()) / (spread.std() + 1e-8)
    return np.concatenate([center, spread], axis=1).astype(np.float32)


# ================= LOAD MODEL =================

set_seed(SEED)

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

W_real = model.lm_head.weight.detach().float().cpu().numpy().astype(np.float32)
W_real = row_norm(W_real)
VOCAB, D = W_real.shape

print("W_real:", W_real.shape)


# ================= EXTRACT HIDDEN =================

prompts = load_prompts()
print("N prompts:", len(prompts))

hidden = {l: [] for l in LAYERS}

for s in range(0, len(prompts), BATCH_SIZE):
    batch = prompts[s:s+BATCH_SIZE]
    enc = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
    ).to(model.device)

    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, use_cache=False)

    last = enc["attention_mask"].sum(dim=1) - 1

    for l in LAYERS:
        if l >= len(out.hidden_states):
            continue
        hs = out.hidden_states[l]
        picked = hs[torch.arange(hs.shape[0], device=hs.device), last]
        hidden[l].append(picked.detach().float().cpu().numpy())

    del enc, out
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

for l in list(hidden.keys()):
    if hidden[l]:
        hidden[l] = row_norm(np.concatenate(hidden[l], axis=0).astype(np.float32))
    else:
        del hidden[l]

LAYERS = list(hidden.keys())
print("Layers:", LAYERS)


# ================= MAIN IA-1 =================

rows = []

for rs in range(N_RANDOM_SEEDS):
    rng = np.random.default_rng(SEED + 1000 + rs)

    W_rand = rng.normal(size=W_real.shape).astype(np.float32)
    W_rand = row_norm(W_rand)

    perm = rng.permutation(VOCAB)

    for l in LAYERS:
        H = hidden[l]
        h_erank = effective_rank(H)

        for k in TOPKS:
            if k >= VOCAB:
                continue

            code_bits = log2_comb_approx(VOCAB, k)
            code_bits_per_sample = code_bits

            print(f"seed={rs} layer={l} topk={k}")

            # Real TopK code
            ids_real = get_topk_ids(H, W_real, k, model.device)
            Z_real = center_spread(ids_real, W_real)
            m = topo_metrics(H, Z_real)
            z_erank = effective_rank(Z_real)

            rows.append({
                "seed": rs,
                "layer": l,
                "topk": k,
                "condition": "real_topk_code",
                "vocab": VOCAB,
                "code_bits_log2_comb": code_bits_per_sample,
                "hidden_effective_rank": h_erank,
                "code_effective_rank": z_erank,
                "erank_retention": z_erank / (h_erank + 1e-8),
                "compression_eff_topo_per_kbit": m["topo"] / (code_bits_per_sample / 1000.0),
                **m,
            })

            # Random W TopK code
            ids_rand = get_topk_ids(H, W_rand, k, model.device)
            Z_rand = center_spread(ids_rand, W_rand)
            m = topo_metrics(H, Z_rand)
            z_erank = effective_rank(Z_rand)

            rows.append({
                "seed": rs,
                "layer": l,
                "topk": k,
                "condition": "random_W_topk_code",
                "vocab": VOCAB,
                "code_bits_log2_comb": code_bits_per_sample,
                "hidden_effective_rank": h_erank,
                "code_effective_rank": z_erank,
                "erank_retention": z_erank / (h_erank + 1e-8),
                "compression_eff_topo_per_kbit": m["topo"] / (code_bits_per_sample / 1000.0),
                **m,
            })

            # Real ids but broken semantic binding
            ids_perm = perm[ids_real]
            Z_perm = center_spread(ids_perm, W_real)
            m = topo_metrics(H, Z_perm)
            z_erank = effective_rank(Z_perm)

            rows.append({
                "seed": rs,
                "layer": l,
                "topk": k,
                "condition": "real_ids_permuted_lookup",
                "vocab": VOCAB,
                "code_bits_log2_comb": code_bits_per_sample,
                "hidden_effective_rank": h_erank,
                "code_effective_rank": z_erank,
                "erank_retention": z_erank / (h_erank + 1e-8),
                "compression_eff_topo_per_kbit": m["topo"] / (code_bits_per_sample / 1000.0),
                **m,
            })

            # Pure random token sets
            ids_noise = rng.integers(0, VOCAB, size=(H.shape[0], k))
            Z_noise = center_spread(ids_noise, W_real)
            m = topo_metrics(H, Z_noise)
            z_erank = effective_rank(Z_noise)

            rows.append({
                "seed": rs,
                "layer": l,
                "topk": k,
                "condition": "random_token_sets",
                "vocab": VOCAB,
                "code_bits_log2_comb": code_bits_per_sample,
                "hidden_effective_rank": h_erank,
                "code_effective_rank": z_erank,
                "erank_retention": z_erank / (h_erank + 1e-8),
                "compression_eff_topo_per_kbit": m["topo"] / (code_bits_per_sample / 1000.0),
                **m,
            })

            del ids_real, ids_rand, ids_perm, ids_noise
            del Z_real, Z_rand, Z_perm, Z_noise
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


df = pd.DataFrame(rows)
df.to_csv(SAVE_DIR / "ia1_full_results.csv", index=False, encoding="utf-8-sig")

summary = (
    df.groupby(["condition", "topk"])
    [["topo", "dist_spearman", "knn_5", "knn_10", "knn_20", "knn_50",
      "code_bits_log2_comb", "hidden_effective_rank", "code_effective_rank",
      "erank_retention", "compression_eff_topo_per_kbit"]]
    .agg(["mean", "std"])
)

summary.to_csv(SAVE_DIR / "ia1_summary_by_topk.csv", encoding="utf-8-sig")

# Saturation curve: real only
real_df = df[df["condition"] == "real_topk_code"]
sat = (
    real_df.groupby("topk")
    [["topo", "compression_eff_topo_per_kbit", "erank_retention"]]
    .mean()
    .reset_index()
)
sat.to_csv(SAVE_DIR / "ia1_real_saturation_curve.csv", index=False, encoding="utf-8-sig")

# Decision
main_k = 5000 if 5000 in TOPKS else max(TOPKS)
main = df[(df["topk"] == main_k) & (df["layer"] < max(LAYERS))]

pivot = main.groupby("condition")["topo"].agg(["mean", "std"]).reset_index()
pivot.to_csv(SAVE_DIR / "ia1_main_topo_pivot.csv", index=False, encoding="utf-8-sig")

def get_mean(cond):
    v = pivot[pivot["condition"] == cond]["mean"].values
    return float(v[0]) if len(v) else float("nan")

real = get_mean("real_topk_code")
randW = get_mean("random_W_topk_code")
perm = get_mean("real_ids_permuted_lookup")
noise = get_mean("random_token_sets")

decision = {
    "main_topk": main_k,
    "real_topo": real,
    "random_W_topo": randW,
    "permuted_lookup_topo": perm,
    "random_token_sets_topo": noise,
    "lift_vs_random_W": real - randW,
    "lift_vs_permuted_lookup": real - perm,
    "lift_vs_random_sets": real - noise,
    "pass_real_above_0_45": real > 0.45,
    "pass_lift_vs_random_W_above_0_20": (real - randW) > 0.20,
    "pass_lift_vs_permuted_above_0_15": (real - perm) > 0.15,
    "pass_lift_vs_random_sets_above_0_20": (real - noise) > 0.20,
}

decision["OVERALL_PASS"] = all([
    decision["pass_real_above_0_45"],
    decision["pass_lift_vs_random_W_above_0_20"],
    decision["pass_lift_vs_permuted_above_0_15"],
    decision["pass_lift_vs_random_sets_above_0_20"],
])

with open(SAVE_DIR / "ia1_decision.json", "w", encoding="utf-8") as f:
    json.dump(decision, f, ensure_ascii=False, indent=2)

print("\n========== IA-1 Decision ==========")
print(json.dumps(decision, ensure_ascii=False, indent=2))
print("\nSaved to:", SAVE_DIR)