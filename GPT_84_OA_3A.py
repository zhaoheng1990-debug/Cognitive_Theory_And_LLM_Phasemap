# ============================================================
# OA-3A: W-Basin Geometry -> ΔR20:25 Audit
#
# Goal:
#   Test whether static basin geometry in W predicts
#   inference-time constraint energy ΔR20:25.
#
# Key improvements over Gemini OA3:
#   1. Target = ΔR20:25, not hidden velocity/acceleration
#   2. GroupKFold by basin_pair, not graph_id
#   3. Real-W vs Random-W vs Permuted-W baselines
#   4. More basin pairs to reduce concept-pair leakage
#
# Outputs:
#   oa3a_dataset.csv
#   oa3a_correlations.csv
#   oa3a_summary.json
# ============================================================

import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

SAVE_DIR = Path("./oa3a_basin_deltaR_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_REPEAT = 2
MAX_LEN = 260
BATCH_SIZE = 16

TRACK_LAYERS = list(range(20, 26))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model/tokenizer...")

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

W_raw = model.lm_head.weight.detach().float().cpu().numpy()
W_real = W_raw / (np.linalg.norm(W_raw, axis=1, keepdims=True) + 1e-9)

VOCAB_SIZE, DIM = W_real.shape
print("W shape:", W_real.shape)

# Baseline W_rand
rng = np.random.default_rng(SEED)
W_rand = rng.normal(size=W_real.shape).astype(np.float32)
W_rand = W_rand / (np.linalg.norm(W_rand, axis=1, keepdims=True) + 1e-9)

# Baseline W_perm
perm = rng.permutation(VOCAB_SIZE)
W_perm = W_real[perm]

# ============================================================
# HELPERS
# ============================================================

def encode_no_special(text: str):
    return tokenizer(text, add_special_tokens=False)["input_ids"]

def single_token_id(text: str):
    candidates = [
        text,
        " " + text,
        text.lower(),
        " " + text.lower(),
    ]
    for cand in candidates:
        ids = encode_no_special(cand)
        if len(ids) == 1:
            return ids[0]
    return None

def is_single_token(text: str):
    return single_token_id(text) is not None

def cos(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)))

def get_vec(W, text: str):
    tid = single_token_id(text)
    if tid is None:
        return None
    return W[tid]

def density(W, vec, k=50):
    sims = W @ vec
    top = np.sort(sims)[-k-1:-1]
    return float(np.mean(top))

# ============================================================
# BASIN PAIRS
# Clean basin C vs conflict basin E.
# Keep mostly single-token common concepts.
# ============================================================

raw_basin_pairs = [
    ("Europe", "Asia"),
    ("Asia", "Europe"),
    ("medicine", "law"),
    ("law", "medicine"),
    ("education", "finance"),
    ("finance", "education"),
    ("food", "science"),
    ("science", "food"),
    ("nature", "industry"),
    ("industry", "nature"),
    ("space", "earth"),
    ("earth", "space"),
  ]

basin_pairs = []
for c, e in raw_basin_pairs:
    if is_single_token(c) and is_single_token(e):
        basin_pairs.append((c, e))
    else:
        print("Skipping non-single-token basin pair:", c, e)

if len(basin_pairs) < 12:
    raise RuntimeError("Too few valid basin pairs. Add more single-token basin labels.")

print("Valid basin pairs:", len(basin_pairs))

# ============================================================
# RELATION SKELETONS
# A belongs_to B, B located/associated with C
# ============================================================

raw_skeletons = [
    ("Paris", "France"),
    ("Berlin", "Germany"),
    ("Tokyo", "Japan"),
    ("Beijing", "China"),
    ("doctor", "hospital"),
    ("judge", "court"),
    ("teacher", "school"),
    ("pilot", "airport"),
 ]

skeletons = []
for a, b in raw_skeletons:
    if is_single_token(a) and is_single_token(b):
        skeletons.append((a, b))
    else:
        print("Skipping non-single-token skeleton:", a, b)

if len(skeletons) < 8:
    raise RuntimeError("Too few valid skeletons.")

print("Valid skeletons:", len(skeletons))

# ============================================================
# STATIC W BASIN FEATURES
# ============================================================

def basin_features(W, a, b, c, e):
    va = get_vec(W, a)
    vb = get_vec(W, b)
    vc = get_vec(W, c)
    ve = get_vec(W, e)

    if any(v is None for v in [va, vb, vc, ve]):
        raise RuntimeError("Token vector missing.")

    ab = cos(va, vb)
    bc = cos(vb, vc)
    ac = cos(va, vc)

    be = cos(vb, ve)
    ae = cos(va, ve)
    ce = cos(vc, ve)

    clean_closure = float(np.mean([ab, bc, ac]))
    conflict_closure = float(np.mean([ab, be, ae]))

    dc = density(W, vc, k=50)
    de = density(W, ve, k=50)

    # Proposed basin geometry variables
    basin_similarity_CE = ce
    basin_density_gap_E_minus_C = de - dc
    basin_barrier = de - dc + ce
    basin_pull_conflict = float(np.mean([ae, be, de, ce]))
    basin_pull_clean = float(np.mean([ac, bc, dc]))

    return {
        "AB": ab,
        "BC": bc,
        "AC": ac,
        "BE": be,
        "AE": ae,
        "CE": ce,
        "clean_closure": clean_closure,
        "conflict_closure": conflict_closure,
        "closure_gap_C_minus_E": clean_closure - conflict_closure,
        "C_density50": dc,
        "E_density50": de,
        "density_gap_E_minus_C": basin_density_gap_E_minus_C,
        "basin_similarity_CE": basin_similarity_CE,
        "basin_barrier": basin_barrier,
        "basin_pull_conflict": basin_pull_conflict,
        "basin_pull_clean": basin_pull_clean,
        "basin_pull_gap_E_minus_C": basin_pull_conflict - basin_pull_clean,
    }

# ============================================================
# PROMPTS
# ============================================================

CONDITIONS = [
    ("clean", "stable"),
    ("weak_distractor", "stable"),
    ("ambiguous_branch", "competition"),
    ("direct_conflict", "competition"),
    ("closure_update", "closure"),
    ("closure_exception", "closure"),
]

def make_prompt(a, b, c, e, condition):
    option = f"Possible answer labels: {c} or {e}."

    if condition == "clean":
        lines = [
            "You are given a small relation graph.",
            option,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is associated with {c}.",
            f"Question: Which label is {a} associated with?",
            f"Answer with exactly one word: {c} or {e}.",
            "Answer:",
        ]

    elif condition == "weak_distractor":
        lines = [
            "You are given a small relation graph.",
            option,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is associated with {c}.",
            f"Weak note: some unrelated descriptions mention {e}.",
            f"Question: Which label is {a} associated with?",
            f"Answer with exactly one word: {c} or {e}.",
            "Answer:",
        ]

    elif condition == "ambiguous_branch":
        lines = [
            "You are given a small relation graph.",
            option,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is associated with {c}.",
            f"Ambiguous branch: some descriptions also connect {a} with {e}.",
            f"Question: Which label is {a} associated with?",
            f"Answer with exactly one word: {c} or {e}.",
            "Answer:",
        ]

    elif condition == "direct_conflict":
        lines = [
            "You are given a small relation graph.",
            option,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is associated with {c}.",
            f"Fact 3: {a} is associated with {e}.",
            f"Question: Which label is {a} associated with?",
            f"Answer with exactly one word: {c} or {e}.",
            "Answer:",
        ]

    elif condition == "closure_update":
        lines = [
            "You are given a small relation graph.",
            option,
            f"Initial rule: {b} is associated with {c}.",
            f"Update: in this graph, {b} is now associated with {e}.",
            f"Fact: {a} belongs to {b}.",
            f"Question: Which label is {a} associated with after the update?",
            f"Answer with exactly one word: {c} or {e}.",
            "Answer:",
        ]

    elif condition == "closure_exception":
        lines = [
            "You are given a small relation graph.",
            option,
            f"General rule: items belonging to {b} are associated with {c}.",
            f"Exception: {a} is a special case associated with {e}.",
            f"Fact: {a} belongs to {b}.",
            f"Question: Which label is {a} associated with?",
            f"Answer with exactly one word: {c} or {e}.",
            "Answer:",
        ]

    else:
        raise ValueError(condition)

    return "\n".join(lines)

# ============================================================
# BUILD DATASET
# ============================================================

rows = []
gid = 0

for rep in range(N_REPEAT):
    for pair_idx, (c, e) in enumerate(basin_pairs[:12]):
        for skel_idx, (a, b) in enumerate(skeletons[:8]):

            real_feat = basin_features(W_real, a, b, c, e)
            rand_feat = basin_features(W_rand, a, b, c, e)
            perm_feat = basin_features(W_perm, a, b, c, e)

            for condition, mechanism in CONDITIONS:
                  row = {
                    "graph_id": gid,
                    "rep": rep,
                    "basin_pair_id": pair_idx,
                    "basin_pair": f"{c}__vs__{e}",
                    "skeleton_id": skel_idx,
                    "A": a,
                    "B": b,
                    "C_clean": c,
                    "E_conflict": e,
                    "condition": condition,
                    "mechanism": mechanism,
                    "prompt": make_prompt(a, b, c, e, condition),
                }

                for k, v in real_feat.items():
                    row[f"Wreal_{k}"] = v
                for k, v in rand_feat.items():
                    row[f"Wrand_{k}"] = v
                for k, v in perm_feat.items():
                    row[f"Wperm_{k}"] = v

                rows.append(row)

            gid += 1

df = pd.DataFrame(rows)
print("Rows:", len(df))
print("Graph ids:", df["graph_id"].nunique())
print("Basin pairs:", df["basin_pair"].nunique())

# ============================================================
# FORWARD: R_l = logit(C)-logit(E)
# ============================================================

@torch.no_grad()
def compute_R_batch(prompts, clean_targets, conflict_targets):
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    out = model(
        **enc,
        output_hidden_states=True,
        use_cache=False,
    )

    hs = out.hidden_states
    attn = enc["attention_mask"]
    last_idx = attn.sum(dim=1) - 1

    out_rows = []

    for bi in range(len(prompts)):
        c_id = single_token_id(clean_targets[bi])
        e_id = single_token_id(conflict_targets[bi])

        item = {}
        for layer in TRACK_LAYERS:
            h = hs[layer + 1][bi, last_idx[bi], :]
            logits = model.lm_head(h).float()
            r = float(logits[c_id].detach().cpu() - logits[e_id].detach().cpu())
            item[f"R{layer}"] = r

        out_rows.append(item)

    return out_rows

R_rows = []
for start in range(0, len(df), BATCH_SIZE):
    sub = df.iloc[start:start + BATCH_SIZE]
    R_rows.extend(
        compute_R_batch(
            sub["prompt"].tolist(),
            sub["C_clean"].tolist(),
            sub["E_conflict"].tolist(),
        )
    )
    print(f"Processed {min(start+BATCH_SIZE, len(df))}/{len(df)}")

R_df = pd.DataFrame(R_rows)
df = pd.concat([df.reset_index(drop=True), R_df], axis=1)

# ============================================================
# ΔR relative to clean condition per graph
# ============================================================

for layer in TRACK_LAYERS:
    base = (
        df[df["condition"] == "clean"][["graph_id", f"R{layer}"]]
        .rename(columns={f"R{layer}": f"R{layer}_clean"})
    )
    df = df.merge(base, on="graph_id", how="left")
    df[f"dR{layer}"] = df[f"R{layer}"] - df[f"R{layer}_clean"]

dR_cols = [f"dR{l}" for l in TRACK_LAYERS]

df["deltaR_l2"] = np.sqrt(np.sum(np.square(df[dR_cols].values), axis=1))
df["deltaR_mean"] = np.mean(df[dR_cols].values, axis=1)
df["deltaR_min"] = np.min(df[dR_cols].values, axis=1)
df["deltaR_final"] = df["dR25"]

# ============================================================
# ANALYSIS
# ============================================================

feature_names = [
    "AB",
    "BC",
    "AC",
    "BE",
    "AE",
    "CE",
    "clean_closure",
    "conflict_closure",
    "closure_gap_C_minus_E",
    "C_density50",
    "E_density50",
    "density_gap_E_minus_C",
    "basin_similarity_CE",
    "basin_barrier",
    "basin_pull_conflict",
    "basin_pull_clean",
    "basin_pull_gap_E_minus_C",
]

target_cols = dR_cols + [
    "deltaR_l2",
    "deltaR_mean",
    "deltaR_min",
    "deltaR_final",
]

def corr_table(prefix):
    rows = []
    for fn in feature_names:
        wc = f"{prefix}_{fn}"
        for tc in target_cols:
            x = df[wc].values.astype(float)
            y = df[tc].values.astype(float)
            if np.std(x) < 1e-9 or np.std(y) < 1e-9:
                r = np.nan
            else:
                r = float(np.corrcoef(x, y)[0, 1])
            rows.append({
                "prefix": prefix,
                "W_feature": wc,
                "target": tc,
                "pearson": r,
                "abs_corr": abs(r) if not np.isnan(r) else np.nan,
            })
    return pd.DataFrame(rows)

corr_df = pd.concat([
    corr_table("Wreal"),
    corr_table("Wrand"),
    corr_table("Wperm"),
], ignore_index=True)

def cv_regression(prefix, target="deltaR_l2", group_col="basin_pair_id"):
    cols = [f"{prefix}_{fn}" for fn in feature_names]
    X = df[cols].values.astype(float)
    y = df[target].values.astype(float)
    groups = df[group_col].values

    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))

    cv = GroupKFold(n_splits=n_splits)

    pred = np.zeros(len(y), dtype=float)

    for tr, te in cv.split(X, y, groups):
        reg = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ])
        reg.fit(X[tr], y[tr])
        pred[te] = reg.predict(X[te])

    r2 = float(r2_score(y, pred))
    corr = float(np.corrcoef(y, pred)[0, 1]) if np.std(pred) > 1e-9 else np.nan

    return {
        "prefix": prefix,
        "target": target,
        "group_col": group_col,
        "n_splits": int(n_splits),
        "r2": r2,
        "corr": corr,
    }

summary = {
    "n_rows": int(len(df)),
    "n_graphs": int(df["graph_id"].nunique()),
    "n_basin_pairs": int(df["basin_pair"].nunique()),
    "n_skeletons": int(df["skeleton_id"].nunique()),
    "n_repeat": int(N_REPEAT),
    "target": "deltaR_l2",
    "basin_pair_group_cv": [
        cv_regression("Wreal", target="deltaR_l2", group_col="basin_pair_id"),
        cv_regression("Wrand", target="deltaR_l2", group_col="basin_pair_id"),
        cv_regression("Wperm", target="deltaR_l2", group_col="basin_pair_id"),
    ],
    "skeleton_group_cv": [
        cv_regression("Wreal", target="deltaR_l2", group_col="skeleton_id"),
        cv_regression("Wrand", target="deltaR_l2", group_col="skeleton_id"),
        cv_regression("Wperm", target="deltaR_l2", group_col="skeleton_id"),
    ],
    "top_real_correlations": (
        corr_df[corr_df["prefix"] == "Wreal"]
        .sort_values("abs_corr", ascending=False)
        .head(30)
        .to_dict(orient="records")
    ),
    "top_rand_correlations": (
        corr_df[corr_df["prefix"] == "Wrand"]
        .sort_values("abs_corr", ascending=False)
        .head(10)
        .to_dict(orient="records")
    ),
    "top_perm_correlations": (
        corr_df[corr_df["prefix"] == "Wperm"]
        .sort_values("abs_corr", ascending=False)
        .head(10)
        .to_dict(orient="records")
    ),
}

# ============================================================
# SAVE
# ============================================================

df.to_csv(SAVE_DIR / "oa3a_dataset.csv", index=False, encoding="utf-8-sig")
corr_df.to_csv(SAVE_DIR / "oa3a_correlations.csv", index=False, encoding="utf-8-sig")

with open(SAVE_DIR / "oa3a_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\nOA-3A finished.")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("Saved to:", SAVE_DIR.resolve())