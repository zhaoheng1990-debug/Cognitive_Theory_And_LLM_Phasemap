# ============================================================
# Neighborhood-Audit-1d: Embedding Bias Control
# Tests whether Audit-5B is caused by pretrained W geometry bias
# ============================================================

import gc, json, random, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"
CORPUS_TXT = r"C:\Users\ZH\Desktop\AGI\data\natural_wiki_texts_450_balanced_diverse_no_meta_tail.txt"

SAVE_DIR = Path("./neighborhood_audit_1d_embedding_bias")
SAVE_DIR.mkdir(exist_ok=True, parents=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SEED = 42
MAX_SAMPLES = 240
BATCH_SIZE = 8
MAX_LEN = 220

LAYERS = list(range(0, 28))
TOPKS = [100, 500, 1000, 2000, 5000]
N_RANDOM_SEEDS = 3
KNN_K = 10


# ---------------- UTILS ----------------
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


def dist_spearman(A, B):
    DA = pairwise_distances(A, metric="cosine")
    DB = pairwise_distances(B, metric="cosine")
    iu = np.triu_indices_from(DA, k=1)
    r = spearmanr(DA[iu], DB[iu]).correlation
    return 0.0 if np.isnan(r) else float(r)


def knn_overlap(A, B, k=10):
    DA = pairwise_distances(A, metric="cosine")
    DB = pairwise_distances(B, metric="cosine")
    np.fill_diagonal(DA, np.inf)
    np.fill_diagonal(DB, np.inf)
    nA = np.argsort(DA, axis=1)[:, :k]
    nB = np.argsort(DB, axis=1)[:, :k]
    return float(np.mean([
        len(set(nA[i]).intersection(set(nB[i]))) / k
        for i in range(A.shape[0])
    ]))


def topo_metrics(H, Z):
    sp = dist_spearman(H, Z)
    ko = knn_overlap(H, Z, KNN_K)
    return {
        "dist_spearman": sp,
        "knn_overlap": ko,
        "topo": 0.5 * sp + 0.5 * ko,
    }


def get_topk_ids(H, W, k, device):
    Ht = torch.tensor(H, dtype=torch.float32, device=device)
    Wt = torch.tensor(W, dtype=torch.float32, device=device)

    out = []
    with torch.no_grad():
        for s in range(0, H.shape[0], 64):
            logits = Ht[s:s+64] @ Wt.T
            out.append(torch.topk(logits, k=k, dim=1).indices.cpu().numpy())

    del Ht, Wt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.concatenate(out, axis=0)


def center_spread(ids, W_lookup):
    vecs = W_lookup[ids]              # [N,K,D]
    c = row_norm(vecs.mean(axis=1))   # [N,D]
    cos = np.einsum("nkd,nd->nk", vecs, c)
    spread = (1.0 - cos).mean(axis=1, keepdims=True)
    spread = (spread - spread.mean()) / (spread.std() + 1e-8)
    return np.concatenate([c, spread], axis=1).astype(np.float32)


# ---------------- LOAD MODEL ----------------
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


# ---------------- EXTRACT HIDDEN ----------------
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


# ---------------- MAIN AUDIT ----------------
rows = []

for rs in range(N_RANDOM_SEEDS):
    rng = np.random.default_rng(SEED + 1000 + rs)

    W_rand = rng.normal(size=W_real.shape).astype(np.float32)
    W_rand = row_norm(W_rand)

    perm = rng.permutation(VOCAB)

    for l in LAYERS:
        H = hidden[l]

        for k in TOPKS:
            print(f"seed={rs} layer={l} topk={k}")

            # 1. Real W: original Audit-5B condition
            ids_real = get_topk_ids(H, W_real, k, model.device)
            Z = center_spread(ids_real, W_real)
            rows.append({
                "seed": rs, "layer": l, "topk": k,
                "condition": "real_W_real_lookup",
                **topo_metrics(H, Z)
            })

            # 2. Random W for scoring and lookup
            ids_rand = get_topk_ids(H, W_rand, k, model.device)
            Z = center_spread(ids_rand, W_rand)
            rows.append({
                "seed": rs, "layer": l, "topk": k,
                "condition": "rand_W_rand_lookup",
                **topo_metrics(H, Z)
            })

            # 3. Random TopK ids, but real W lookup
            Z = center_spread(ids_rand, W_real)
            rows.append({
                "seed": rs, "layer": l, "topk": k,
                "condition": "rand_ids_real_lookup",
                **topo_metrics(H, Z)
            })

            # 4. Real TopK ids, but broken token-id -> embedding binding
            ids_perm = perm[ids_real]
            Z = center_spread(ids_perm, W_real)
            rows.append({
                "seed": rs, "layer": l, "topk": k,
                "condition": "real_ids_permuted_lookup",
                **topo_metrics(H, Z)
            })

            # 5. Pure random token sets
            ids_noise = rng.integers(0, VOCAB, size=(H.shape[0], k))
            Z = center_spread(ids_noise, W_real)
            rows.append({
                "seed": rs, "layer": l, "topk": k,
                "condition": "random_token_sets",
                **topo_metrics(H, Z)
            })

            del ids_real, ids_rand, ids_perm, ids_noise, Z
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ---------------- SAVE + DECISION ----------------
df = pd.DataFrame(rows)
df.to_csv(SAVE_DIR / "audit1d_full_results.csv", index=False, encoding="utf-8-sig")

summary = (
    df.groupby(["condition", "topk"])[["dist_spearman", "knn_overlap", "topo"]]
    .agg(["mean", "std"])
)
summary.to_csv(SAVE_DIR / "audit1d_summary.csv", encoding="utf-8-sig")

# Main decision: nonfinal, largest TopK
main = df[(df["layer"] < max(LAYERS)) & (df["topk"] == max(TOPKS))]
pivot = main.groupby("condition")["topo"].agg(["mean", "std"]).reset_index()
pivot.to_csv(SAVE_DIR / "audit1d_main_pivot.csv", index=False, encoding="utf-8-sig")

def mean_of(cond):
    v = pivot[pivot["condition"] == cond]["mean"].values
    return float(v[0]) if len(v) else float("nan")

real = mean_of("real_W_real_lookup")
rand = mean_of("rand_W_rand_lookup")
rand_ids = mean_of("rand_ids_real_lookup")
perm_lookup = mean_of("real_ids_permuted_lookup")
random_sets = mean_of("random_token_sets")

decision = {
    "real": real,
    "rand_W_rand_lookup": rand,
    "rand_ids_real_lookup": rand_ids,
    "real_ids_permuted_lookup": perm_lookup,
    "random_token_sets": random_sets,

    "lift_vs_rand_W": real - rand,
    "lift_vs_rand_ids_real_lookup": real - rand_ids,
    "lift_vs_permuted_lookup": real - perm_lookup,
    "lift_vs_random_sets": real - random_sets,

    "pass_real_above_0_45": real > 0.45,
    "pass_lift_vs_rand_W_above_0_20": (real - rand) > 0.20,
    "pass_lift_vs_rand_ids_above_0_15": (real - rand_ids) > 0.15,
    "pass_lift_vs_perm_lookup_above_0_15": (real - perm_lookup) > 0.15,
    "pass_lift_vs_random_sets_above_0_20": (real - random_sets) > 0.20,
}

decision["OVERALL_PASS"] = all([
    decision["pass_real_above_0_45"],
    decision["pass_lift_vs_rand_W_above_0_20"],
    decision["pass_lift_vs_rand_ids_above_0_15"],
    decision["pass_lift_vs_perm_lookup_above_0_15"],
    decision["pass_lift_vs_random_sets_above_0_20"],
])

with open(SAVE_DIR / "audit1d_decision.json", "w", encoding="utf-8") as f:
    json.dump(decision, f, ensure_ascii=False, indent=2)

print("\n========== AUDIT-1D DECISION ==========")
print(json.dumps(decision, ensure_ascii=False, indent=2))
print("\nSaved to:", SAVE_DIR)