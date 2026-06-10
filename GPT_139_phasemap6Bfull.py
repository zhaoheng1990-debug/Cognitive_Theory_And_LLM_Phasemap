# ============================================================
# PhaseMap-6B-full
# TopK Layerwise Operator Re-Identification
#
# Goal:
#   Build transition-level dataset:
#       N_k(H_l) -> N_k(H_{l+1})
#
# Working operator families:
#   Preservation
#   Rotation
#   Reconstruction
#   Regression / Backtracking
#
# This does NOT assume these four are complete.
# It tests their coverage and residual unknown transitions.
# ============================================================

import gc
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================

MODEL_PATH = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"
DATASET_PATH = Path(r"C:\Windows\System32\phasemap5a_outputs\phasemap5a_dataset.csv")

OUT_DIR = Path("./phasemap6b_full_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SEED = 42
MAX_LEN = 256
BATCH_SIZE = 4

# Qwen2.5-1.5B has layers 0-27 in our previous convention
TRACK_LAYERS = list(range(7, 27))      # transitions L7->L8 ... L25->L26
TRANSITION_LAYERS = list(range(7, 26)) # l -> l+1
K = 500                                # main operator scale
K_SMALL = 100
K_LARGE = 1000

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(DATASET_PATH)

# Ensure no duplicate columns if 5C appended extras
df = df.loc[:, ~df.columns.duplicated()].copy()

required = ["prompt", "graph_id", "condition", "phase", "C_id", "E_id"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing columns in dataset: {missing}")

print("Dataset:", df.shape)

# =========================
# LOAD MODEL
# =========================

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

W_cpu = model.lm_head.weight.detach().float().cpu().numpy().astype(np.float32)
Wn_cpu = W_cpu / (np.linalg.norm(W_cpu, axis=1, keepdims=True) + 1e-8)

print("Model loaded.")
print("LM head:", W_cpu.shape)

# =========================
# FORWARD
# =========================

@torch.no_grad()
def forward_hidden(prompts):
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
    ).to(model.device)

    out = model(
        **enc,
        output_hidden_states=True,
        use_cache=False,
    )

    attn = enc["attention_mask"]
    last_pos = attn.sum(dim=1) - 1

    hs = out.hidden_states
    result = {}

    for l in TRACK_LAYERS:
        h = hs[l + 1]
        vecs = []
        for i, p in enumerate(last_pos):
            vecs.append(h[i, p].detach().float().cpu().numpy())
        result[l] = np.stack(vecs)

    return result

# =========================
# TOPK STATE
# =========================

def topk_state(h, c_id, e_id, k):
    logits = h @ W_cpu.T

    top_idx = np.argpartition(-logits, k)[:k]
    top_scores = logits[top_idx]
    top_idx = top_idx[np.argsort(-top_scores)]

    emb = W_cpu[top_idx]
    embn = Wn_cpu[top_idx]

    center = emb.mean(axis=0)
    center_norm = np.linalg.norm(center) + 1e-8

    cos_to_center = (emb @ center) / (
        (np.linalg.norm(emb, axis=1) + 1e-8) * center_norm
    )

    spread = float(np.mean(1.0 - cos_to_center))

    c_logit = float(logits[c_id])
    e_logit = float(logits[e_id])
    R = c_logit - e_logit

    # Rank gap: positive means E is below C; negative means E outranks C
    # Use argsort once; vocab is large but dataset is manageable.
    order = np.argsort(-logits)
    c_rank = int(np.where(order == c_id)[0][0])
    e_rank = int(np.where(order == e_id)[0][0])
    rank_gap = e_rank - c_rank

    return {
        "logits": logits,
        "top_idx": top_idx,
        "top_set": set(top_idx.tolist()),
        "center": center,
        "spread": spread,
        "R": R,
        "c_rank": c_rank,
        "e_rank": e_rank,
        "rank_gap": rank_gap,
    }


def transition_features(state_prev, state_cur, state_next=None, state_init=None):
    a = state_prev
    b = state_cur

    set_a = a["top_set"]
    set_b = b["top_set"]

    inter = len(set_a & set_b)
    union = len(set_a | set_b)

    jacc = inter / max(union, 1)

    ca = a["center"]
    cb = b["center"]

    center_cos = float(
        np.dot(ca, cb)
        / ((np.linalg.norm(ca) + 1e-8) * (np.linalg.norm(cb) + 1e-8))
    )
    center_step = 1.0 - center_cos

    spread_delta = b["spread"] - a["spread"]
    R_delta = b["R"] - a["R"]
    rank_gap_delta = b["rank_gap"] - a["rank_gap"]

    # Regression / backtracking metrics
    velocity_reversal_R = 0
    center_backtrack_prev = 0.0
    center_backtrack_init = 0.0

    if state_next is not None:
        # Here state_next can be previous-previous when passed accordingly.
        pass

    if state_init is not None:
        ci = state_init["center"]
        cos_a_init = float(
            np.dot(ca, ci)
            / ((np.linalg.norm(ca) + 1e-8) * (np.linalg.norm(ci) + 1e-8))
        )
        cos_b_init = float(
            np.dot(cb, ci)
            / ((np.linalg.norm(cb) + 1e-8) * (np.linalg.norm(ci) + 1e-8))
        )
        # positive means current transition moved closer to init
        center_backtrack_init = cos_b_init - cos_a_init

    return {
        "jaccard": float(jacc),
        "center_cos": center_cos,
        "center_step": float(center_step),
        "spread_l": float(a["spread"]),
        "spread_l1": float(b["spread"]),
        "spread_delta": float(spread_delta),
        "R_l": float(a["R"]),
        "R_l1": float(b["R"]),
        "R_delta": float(R_delta),
        "rank_gap_l": float(a["rank_gap"]),
        "rank_gap_l1": float(b["rank_gap"]),
        "rank_gap_delta": float(rank_gap_delta),
        "center_backtrack_init": float(center_backtrack_init),
    }


# =========================
# HEURISTIC OPERATOR LABELS
# =========================

def label_operator(row):
    """
    Four working morphism families.

    Preservation:
        high overlap, small movement, small spread change.

    Rotation:
        partial preservation but decision axis / rank direction changes.

    Reconstruction:
        low overlap or large center/spread movement.

    Regression:
        reversal / backtracking toward previous or initial state.
    """

    j = row["jaccard"]
    step = row["center_step"]
    sd = row["spread_delta"]
    rd = row["R_delta"]
    rgd = row["rank_gap_delta"]
    back_init = row["center_backtrack_init"]
    r_reversal = row["R_velocity_reversal"]
    rg_reversal = row["rank_gap_reversal"]

    # Regression first: explicit reversal/backtrack
    if r_reversal or rg_reversal or back_init > 0.025:
        return "Regression"

    # Reconstruction: large semantic replacement
    if j < 0.32 or step > 0.12 or abs(sd) > 0.012:
        return "Reconstruction"

    # Preservation: nearly same neighborhood
    if j > 0.50 and step < 0.06 and abs(sd) < 0.006:
        return "Preservation"

    # Rotation: neighborhood partly preserved but axis changes
    if j >= 0.32 and j <= 0.60 and (abs(rd) > 0.5 or abs(rgd) > 500 or step >= 0.04):
        return "Rotation"

    return "Preservation"


# =========================
# EXTRACT TRANSITIONS
# =========================

all_rows = []

print("Extracting transition states...")

for start in range(0, len(df), BATCH_SIZE):
    sub = df.iloc[start:start + BATCH_SIZE].reset_index(drop=True)
    H = forward_hidden(sub["prompt"].tolist())

    for i in range(len(sub)):
        global_i = start + i
        c_id = int(sub.loc[i, "C_id"])
        e_id = int(sub.loc[i, "E_id"])

        states = {}

        for l in TRACK_LAYERS:
            states[l] = topk_state(H[l][i], c_id, e_id, K)

        init_state = states[TRACK_LAYERS[0]]

        # Precompute R and rank velocities for reversal
        R_by_layer = {l: states[l]["R"] for l in TRACK_LAYERS}
        gap_by_layer = {l: states[l]["rank_gap"] for l in TRACK_LAYERS}

        for l in TRANSITION_LAYERS:
            feat = transition_features(
                states[l],
                states[l + 1],
                state_init=init_state,
            )

            # previous velocity for reversal
            if l > TRANSITION_LAYERS[0]:
                prev_R_delta = R_by_layer[l] - R_by_layer[l - 1]
                cur_R_delta = R_by_layer[l + 1] - R_by_layer[l]

                prev_gap_delta = gap_by_layer[l] - gap_by_layer[l - 1]
                cur_gap_delta = gap_by_layer[l + 1] - gap_by_layer[l]

                feat["R_prev_delta"] = float(prev_R_delta)
                feat["R_velocity_reversal"] = int(prev_R_delta * cur_R_delta < 0)

                feat["rank_gap_prev_delta"] = float(prev_gap_delta)
                feat["rank_gap_reversal"] = int(prev_gap_delta * cur_gap_delta < 0)
            else:
                feat["R_prev_delta"] = 0.0
                feat["R_velocity_reversal"] = 0
                feat["rank_gap_prev_delta"] = 0.0
                feat["rank_gap_reversal"] = 0

            # backtrack to previous previous center
            if l > TRANSITION_LAYERS[0]:
                c_prevprev = states[l - 1]["center"]
                c_prev = states[l]["center"]
                c_cur = states[l + 1]["center"]

                cos_prev_to_pp = float(
                    np.dot(c_prev, c_prevprev)
                    / ((np.linalg.norm(c_prev) + 1e-8) * (np.linalg.norm(c_prevprev) + 1e-8))
                )
                cos_cur_to_pp = float(
                    np.dot(c_cur, c_prevprev)
                    / ((np.linalg.norm(c_cur) + 1e-8) * (np.linalg.norm(c_prevprev) + 1e-8))
                )
                feat["center_backtrack_prevprev"] = float(cos_cur_to_pp - cos_prev_to_pp)
            else:
                feat["center_backtrack_prevprev"] = 0.0

            row = {
                "sample_id": int(global_i),
                "graph_id": int(sub.loc[i, "graph_id"]),
                "condition": sub.loc[i, "condition"],
                "phase": sub.loc[i, "phase"],
                "layer": int(l),
                "transition": f"L{l}_to_L{l+1}",
                "C_id": c_id,
                "E_id": e_id,
            }
            row.update(feat)

            all_rows.append(row)

    del H
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

transition_df = pd.DataFrame(all_rows)

# heuristic labels
transition_df["heuristic_operator"] = transition_df.apply(label_operator, axis=1)

# =========================
# UNSUPERVISED CLUSTERING
# =========================

meta_cols = {
    "sample_id", "graph_id", "condition", "phase", "layer", "transition",
    "C_id", "E_id", "heuristic_operator",
}

feature_cols = [
    c for c in transition_df.columns
    if c not in meta_cols
    and pd.api.types.is_numeric_dtype(transition_df[c])
]

X = transition_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
Xs = StandardScaler().fit_transform(X)

pca = PCA(n_components=min(10, Xs.shape[1]))
Xp = pca.fit_transform(Xs)

for i in range(Xp.shape[1]):
    transition_df[f"op_pc{i+1}"] = Xp[:, i]

pca_info = [
    {"pc": i + 1, "explained_variance": float(v)}
    for i, v in enumerate(pca.explained_variance_ratio_)
]

cluster_results = []

for KK in range(2, 11):
    km = KMeans(n_clusters=KK, random_state=42, n_init=30)
    labels = km.fit_predict(Xs)
    sil = silhouette_score(Xs, labels)

    cluster_results.append({
        "K": int(KK),
        "silhouette": float(sil),
        "cluster_sizes": {
            str(i): int(np.sum(labels == i))
            for i in range(KK)
        },
    })

best = max(cluster_results, key=lambda x: x["silhouette"])
BEST_K = int(best["K"])

km = KMeans(n_clusters=BEST_K, random_state=42, n_init=50)
transition_df["operator_cluster"] = km.fit_predict(Xs)

# =========================
# SUMMARIES
# =========================

operator_summary = transition_df.groupby(
    ["heuristic_operator"]
).agg(
    n=("heuristic_operator", "size"),
    jaccard_mean=("jaccard", "mean"),
    center_step_mean=("center_step", "mean"),
    spread_delta_mean=("spread_delta", "mean"),
    R_delta_mean=("R_delta", "mean"),
    R_reversal_mean=("R_velocity_reversal", "mean"),
    rank_gap_reversal_mean=("rank_gap_reversal", "mean"),
    backtrack_init_mean=("center_backtrack_init", "mean"),
    backtrack_prevprev_mean=("center_backtrack_prevprev", "mean"),
).reset_index()

layer_profile = transition_df.groupby(
    ["layer", "heuristic_operator"]
).size().reset_index(name="count")

layer_total = layer_profile.groupby("layer")["count"].transform("sum")
layer_profile["frac"] = layer_profile["count"] / layer_total

condition_profile = transition_df.groupby(
    ["condition", "heuristic_operator"]
).size().reset_index(name="count")

condition_total = condition_profile.groupby("condition")["count"].transform("sum")
condition_profile["frac"] = condition_profile["count"] / condition_total

phase_profile = transition_df.groupby(
    ["phase", "heuristic_operator"]
).size().reset_index(name="count")

phase_total = phase_profile.groupby("phase")["count"].transform("sum")
phase_profile["frac"] = phase_profile["count"] / phase_total

cluster_summary = transition_df.groupby(
    ["operator_cluster"]
).agg(
    n=("operator_cluster", "size"),
    jaccard_mean=("jaccard", "mean"),
    center_step_mean=("center_step", "mean"),
    spread_delta_mean=("spread_delta", "mean"),
    R_delta_mean=("R_delta", "mean"),
    R_reversal_mean=("R_velocity_reversal", "mean"),
    rank_gap_reversal_mean=("rank_gap_reversal", "mean"),
    backtrack_init_mean=("center_backtrack_init", "mean"),
).reset_index()

cluster_operator_cross = transition_df.groupby(
    ["operator_cluster", "heuristic_operator"]
).size().reset_index(name="count")

# Unknown coverage: transitions not confidently represented by four labels.
# Here heuristic always labels, so define weak-confidence by middle ambiguous region.
transition_df["weak_confidence"] = (
    (transition_df["jaccard"].between(0.30, 0.38))
    & (transition_df["center_step"].between(0.055, 0.09))
    & (transition_df["R_velocity_reversal"] == 0)
).astype(int)

coverage_summary = {
    "n_transitions": int(len(transition_df)),
    "operator_counts": transition_df["heuristic_operator"].value_counts().to_dict(),
    "weak_confidence_frac": float(transition_df["weak_confidence"].mean()),
    "best_K": int(BEST_K),
    "best_silhouette": float(best["silhouette"]),
}

# =========================
# SAVE
# =========================

transition_df.to_csv(OUT_DIR / "phasemap6b_full_transition_dataset.csv", index=False)
operator_summary.to_csv(OUT_DIR / "phasemap6b_full_operator_summary.csv", index=False)
layer_profile.to_csv(OUT_DIR / "phasemap6b_full_layer_profile.csv", index=False)
condition_profile.to_csv(OUT_DIR / "phasemap6b_full_condition_profile.csv", index=False)
phase_profile.to_csv(OUT_DIR / "phasemap6b_full_phase_profile.csv", index=False)
cluster_summary.to_csv(OUT_DIR / "phasemap6b_full_cluster_summary.csv", index=False)
cluster_operator_cross.to_csv(OUT_DIR / "phasemap6b_full_cluster_operator_cross.csv", index=False)
pd.DataFrame(cluster_results).to_csv(OUT_DIR / "phasemap6b_full_k_selection.csv", index=False)

summary = {
    "n_samples": int(len(df)),
    "n_transitions": int(len(transition_df)),
    "transition_layers": TRANSITION_LAYERS,
    "K": int(K),
    "pca": pca_info,
    "cluster_results": cluster_results,
    "coverage_summary": coverage_summary,
    "operator_summary": operator_summary.to_dict(orient="records"),
    "cluster_summary": cluster_summary.to_dict(orient="records"),
}

with open(OUT_DIR / "phasemap6b_full_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\nDone.")
print("Saved to:", OUT_DIR)
print("\nCoverage:")
print(coverage_summary)
print("\nOperator summary:")
print(operator_summary)
print("\nBest K:", BEST_K, "silhouette:", best["silhouette"])