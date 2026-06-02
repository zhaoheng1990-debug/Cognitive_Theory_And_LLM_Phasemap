# ============================================================
# OA-1: W Geometry Audit Lite
# Goal:
#   Audit whether vocabulary/lm_head W already contains:
#   1) semantic local clusters
#   2) closure triangle geometry
#   3) evidence-rule binding axis hints
# ============================================================

import os, random, math, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# =========================
# CONFIG
# =========================

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"
SAVE_DIR = Path("./oa1_w_geometry_outputs")
SAVE_DIR.mkdir(exist_ok=True, parents=True)

SEED = 42
TOPN_NEIGHBORS = 30
RANDOM_BASELINE_N = 2000

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =========================
# LOAD
# =========================

print("Loading model/tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    local_files_only=True,
    trust_remote_code=True,
)
model.eval()

W = model.lm_head.weight.detach().float().cpu().numpy()
W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)

vocab_size, dim = W.shape
print("W shape:", W.shape)

id_to_tok = {i: tokenizer.decode([i]) for i in range(vocab_size)}

# =========================
# HELPERS
# =========================

def token_id(text):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) == 1:
        return ids[0]
    ids2 = tokenizer(" " + text, add_special_tokens=False)["input_ids"]
    if len(ids2) == 1:
        return ids2[0]
    return None

def nearest(tok, n=TOPN_NEIGHBORS):
    tid = token_id(tok)
    if tid is None:
        return None
    sims = W @ W[tid]
    idx = np.argsort(-sims)[:n+1]
    rows = []
    for j in idx:
        if j == tid:
            continue
        rows.append({
            "query": tok,
            "neighbor_id": int(j),
            "neighbor_token": id_to_tok[j],
            "cos": float(sims[j]),
        })
        if len(rows) >= n:
            break
    return rows

def cos(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a)+1e-9)*(np.linalg.norm(b)+1e-9)))

def get_vec(tok):
    tid = token_id(tok)
    if tid is None:
        return None
    return W[tid]

def random_pair_cos(n=RANDOM_BASELINE_N):
    ids1 = np.random.randint(0, vocab_size, size=n)
    ids2 = np.random.randint(0, vocab_size, size=n)
    return np.array([cos(W[a], W[b]) for a, b in zip(ids1, ids2)])

# =========================
# OA-1A: LOCAL SEMANTIC CLUSTERS
# =========================

cluster_seeds = [
    "France", "Paris", "Germany", "Berlin",
    "doctor", "hospital", "law", "court",
    "evidence", "source", "rule", "exception",
    "before", "after", "because", "therefore",
    "animal", "dog", "cat", "bird",
    "China", "Beijing", "Japan", "Tokyo",
]

cluster_rows = []
for t in cluster_seeds:
    rows = nearest(t)
    if rows:
        cluster_rows.extend(rows)

pd.DataFrame(cluster_rows).to_csv(SAVE_DIR / "oa1a_local_neighbors.csv", index=False, encoding="utf-8-sig")

# =========================
# OA-1B: CLOSURE TRIANGLE GEOMETRY
# A->B, B->C should imply A near C / triangle coherence
# =========================

closure_triples = [
    ("Paris", "France", "Europe"),
    ("Berlin", "Germany", "Europe"),
    ("Tokyo", "Japan", "Asia"),
    ("Beijing", "China", "Asia"),
    ("dog", "animal", "life"),
    ("cat", "animal", "life"),
    ("doctor", "hospital", "medicine"),
    ("judge", "court", "law"),
]

rand_cos = random_pair_cos()
rand_mean, rand_std = float(rand_cos.mean()), float(rand_cos.std())

tri_rows = []
for a,b,c in closure_triples:
    va, vb, vc = get_vec(a), get_vec(b), get_vec(c)
    if va is None or vb is None or vc is None:
        tri_rows.append({"A":a, "B":b, "C":c, "valid":False})
        continue

    ab = cos(va, vb)
    bc = cos(vb, vc)
    ac = cos(va, vc)
    closure_score = (ab + bc + ac) / 3
    z = (closure_score - rand_mean) / (rand_std + 1e-9)

    tri_rows.append({
        "A": a, "B": b, "C": c,
        "valid": True,
        "cos_AB": ab,
        "cos_BC": bc,
        "cos_AC": ac,
        "closure_score_mean_cos": closure_score,
        "random_mean": rand_mean,
        "random_std": rand_std,
        "closure_z_vs_random": z,
    })

pd.DataFrame(tri_rows).to_csv(SAVE_DIR / "oa1b_closure_triangles.csv", index=False, encoding="utf-8-sig")

# =========================
# OA-1C: EVIDENCE-RULE AXIS HINT
# Check whether evidence tokens and rule tokens form separable poles.
# =========================

evidence_tokens = ["evidence", "source", "proof", "claim", "citation", "document", "record"]
rule_tokens = ["rule", "exception", "override", "policy", "law", "authority", "requirement"]

E_vecs, E_valid = [], []
R_vecs, R_valid = [], []

for t in evidence_tokens:
    v = get_vec(t)
    if v is not None:
        E_vecs.append(v); E_valid.append(t)

for t in rule_tokens:
    v = get_vec(t)
    if v is not None:
        R_vecs.append(v); R_valid.append(t)

E_cent = np.mean(E_vecs, axis=0)
R_cent = np.mean(R_vecs, axis=0)
axis_ER = R_cent - E_cent
axis_ER = axis_ER / (np.linalg.norm(axis_ER) + 1e-9)

axis_rows = []
for t in E_valid + R_valid:
    v = get_vec(t)
    z = cos(v, axis_ER)
    axis_rows.append({
        "token": t,
        "group": "evidence" if t in E_valid else "rule",
        "z_ER_rule_positive": z,
    })

# random projection baseline
rand_ids = np.random.randint(0, vocab_size, size=RANDOM_BASELINE_N)
rand_proj = np.array([cos(W[i], axis_ER) for i in rand_ids])

summary = {
    "valid_evidence_tokens": E_valid,
    "valid_rule_tokens": R_valid,
    "mean_evidence_z": float(np.mean([r["z_ER_rule_positive"] for r in axis_rows if r["group"]=="evidence"])),
    "mean_rule_z": float(np.mean([r["z_ER_rule_positive"] for r in axis_rows if r["group"]=="rule"])),
    "random_projection_mean": float(rand_proj.mean()),
    "random_projection_std": float(rand_proj.std()),
}

pd.DataFrame(axis_rows).to_csv(SAVE_DIR / "oa1c_evidence_rule_axis.csv", index=False, encoding="utf-8-sig")
with open(SAVE_DIR / "oa1_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\nOA-1 finished.")
print("Saved to:", SAVE_DIR.resolve())
print(json.dumps(summary, ensure_ascii=False, indent=2))