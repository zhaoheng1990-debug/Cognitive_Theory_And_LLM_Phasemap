# ============================================================
# PhaseMap-5A
# TopK Trajectory State Audit
#
# Goal:
#   Test whether L7-L19 TopK organization state predicts
#   PhaseMap-4A rollout failures better than R-space features.
#
# Core hypothesis:
#   Z not in R-space.
#   Z ≈ TopK trajectory organization.
# ============================================================

import os, gc, json, random, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error, roc_auc_score,
    accuracy_score, f1_score
)

warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================

MODEL_PATH = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

SAVE_DIR = Path("./phasemap5a_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SEED = 42
N_GRAPHS = 96
MAX_LEN = 256

TRACK_LAYERS = list(range(7, 20))      # L7-L19
DECISION_LAYERS = list(range(20, 26)) # R20-R25
TOPK_LIST = [50, 100, 200, 500, 1000]

BATCH_SIZE = 4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =========================
# LOAD MODEL
# =========================

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    device_map="auto" if DEVICE == "cuda" else None,
    local_files_only=True,
    trust_remote_code=True
)

if DEVICE == "cpu":
    model.to(DEVICE)

model.eval()

W = model.lm_head.weight.detach().float()
W_cpu = W.cpu().numpy().astype(np.float32)

print("Model loaded.")
print("LM head:", W.shape)

# =========================
# LABELS
# =========================

def token_ids(text):
    return tokenizer(text, add_special_tokens=False)["input_ids"]

def continuation_id(label):
    ids = token_ids(" " + label)
    if len(ids) != 1:
        return None
    return ids[0]

CANDIDATES = [
    "Red", "Blue", "Green", "Yellow",
    "North", "South", "East", "West",
    "Alpha", "Beta", "Gamma", "Delta",
    "Circle", "Square", "Triangle", "Star",
    "Copper", "Silver", "Gold", "Iron",
]

single_labels = []
for x in CANDIDATES:
    tid = continuation_id(x)
    if tid is not None:
        single_labels.append((x, tid))

if len(single_labels) < 8:
    raise RuntimeError("Not enough single-token labels.")

LABELS = single_labels[:12]
print("Labels:", LABELS)

# =========================
# DATASET
# =========================

CONDITIONS = [
    ("stable_positive", "positive"),
    ("weak_positive", "positive"),
    ("true_compete_balanced", "critical"),
    ("true_compete_order_C_first", "critical"),
    ("true_compete_order_E_first", "critical"),
    ("direct_negative", "negative"),
    ("update_negative", "negative"),
    ("exception_negative", "negative"),
]

def make_entity(prefix, i):
    return f"{prefix}{i:03d}"

def make_prompt(a, b, c_label, e_label, condition):
    if condition == "stable_positive":
        return f"""You are given a relation graph.
Fact 1: {a} belongs to {b}.
Fact 2: {b} is associated with {c_label}.
Question: Which label is {a} associated with?
Answer with exactly one word: {c_label} or {e_label}.
Answer:"""

    if condition == "weak_positive":
        return f"""You are given a relation graph.
Fact 1: {a} belongs to {b}.
Fact 2: {b} is associated with {c_label}.
Note: another unrelated object may be associated with {e_label}.
Question: Which label is {a} associated with?
Answer with exactly one word: {c_label} or {e_label}.
Answer:"""

    if condition == "true_compete_balanced":
        return f"""You are given a relation graph with two possible branches.
Fact 1: {a} belongs to {b}.
Fact 2: {b} is associated with {c_label}.
Fact 3: {a} may also be linked to a conflicting branch associated with {e_label}.
Question: Which label is {a} associated with?
Answer with exactly one word: {c_label} or {e_label}.
Answer:"""

    if condition == "true_compete_order_C_first":
        return f"""You are given a relation graph.
Primary fact: {a} belongs to {b}.
Primary rule: {b} is associated with {c_label}.
Competing note: another source claims {a} is associated with {e_label}.
Question: Which label is {a} associated with?
Answer with exactly one word: {c_label} or {e_label}.
Answer:"""

    if condition == "true_compete_order_E_first":
        return f"""You are given a relation graph.
Competing note: another source claims {a} is associated with {e_label}.
Primary fact: {a} belongs to {b}.
Primary rule: {b} is associated with {c_label}.
Question: Which label is {a} associated with?
Answer with exactly one word: {c_label} or {e_label}.
Answer:"""

    if condition == "direct_negative":
        return f"""You are given a relation graph.
Fact 1: {a} belongs to {b}.
Fact 2: {b} is associated with {c_label}.
Correction: {a} is directly associated with {e_label}.
Question: Which label is {a} associated with?
Answer with exactly one word: {c_label} or {e_label}.
Answer:"""

    if condition == "update_negative":
        return f"""You are given a relation graph.
Old rule: {b} was associated with {c_label}.
Updated rule: {b} is now associated with {e_label}.
Fact: {a} belongs to {b}.
Question: Which label is {a} associated with now?
Answer with exactly one word: {c_label} or {e_label}.
Answer:"""

    if condition == "exception_negative":
        return f"""You are given a relation graph.
General rule: things belonging to {b} are associated with {c_label}.
Exception: {a} is associated with {e_label}.
Question: Which label is {a} associated with?
Answer with exactly one word: {c_label} or {e_label}.
Answer:"""

    raise ValueError(condition)

rows = []

for g in range(N_GRAPHS):
    a = make_entity("A", g)
    b = make_entity("B", g)

    c_label, c_id = LABELS[(2 * g) % len(LABELS)]
    e_label, e_id = LABELS[(2 * g + 1) % len(LABELS)]

    for cond, phase in CONDITIONS:
        rows.append({
            "graph_id": g,
            "condition": cond,
            "phase": phase,
            "A": a,
            "B": b,
            "C_label": c_label,
            "E_label": e_label,
            "C_id": c_id,
            "E_id": e_id,
            "prompt": make_prompt(a, b, c_label, e_label, cond),
        })

df = pd.DataFrame(rows)
print("Dataset:", df.shape)

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
        use_cache=False
    )

    # last non-pad position
    attn = enc["attention_mask"]
    last_pos = attn.sum(dim=1) - 1

    hs = out.hidden_states  # embedding + layers
    result = {}

    for l in sorted(set(TRACK_LAYERS + DECISION_LAYERS + [26, 27])):
        h = hs[l + 1]  # layer output
        vecs = []
        for i, p in enumerate(last_pos):
            vecs.append(h[i, p].detach().float().cpu().numpy())
        result[l] = np.stack(vecs)

    return result

# =========================
# FEATURE EXTRACTION
# =========================

def topk_features_from_hidden(H_by_layer, c_ids, e_ids):
    feats = []

    for i in range(len(c_ids)):
        row = {}

        prev_sets = {}
        prev_centers = {}

        for k in TOPK_LIST:
            centers = []
            spreads = []
            c_ranks = []
            e_ranks = []
            jaccs = []
            center_steps = []

            last_set = None
            last_center = None

            for l in TRACK_LAYERS:
                h = H_by_layer[l][i]
                logits = h @ W_cpu.T

                top_idx = np.argpartition(-logits, k)[:k]
                top_scores = logits[top_idx]
                order = np.argsort(-top_scores)
                top_idx = top_idx[order]

                emb = W_cpu[top_idx]
                center = emb.mean(axis=0)
                center_norm = np.linalg.norm(center) + 1e-8
                emb_norm = np.linalg.norm(emb, axis=1) + 1e-8
                cos = (emb @ center) / (emb_norm * center_norm)
                spread = float(np.mean(1.0 - cos))

                rank_order = np.argsort(-logits)
                c_rank = int(np.where(rank_order == c_ids[i])[0][0])
                e_rank = int(np.where(rank_order == e_ids[i])[0][0])

                centers.append(center)
                spreads.append(spread)
                c_ranks.append(c_rank)
                e_ranks.append(e_rank)

                cur_set = set(top_idx.tolist())
                if last_set is not None:
                    inter = len(cur_set & last_set)
                    union = len(cur_set | last_set)
                    jaccs.append(inter / max(union, 1))

                    step = 1.0 - float(
                        np.dot(center, last_center)
                        / ((np.linalg.norm(center) + 1e-8) * (np.linalg.norm(last_center) + 1e-8))
                    )
                    center_steps.append(step)

                last_set = cur_set
                last_center = center

            centers = np.stack(centers)
            spreads = np.array(spreads)
            c_ranks = np.array(c_ranks)
            e_ranks = np.array(e_ranks)
            jaccs = np.array(jaccs)
            center_steps = np.array(center_steps)

            row[f"k{k}_spread_mean"] = spreads.mean()
            row[f"k{k}_spread_std"] = spreads.std()
            row[f"k{k}_spread_slope"] = np.polyfit(np.arange(len(spreads)), spreads, 1)[0]

            row[f"k{k}_jacc_mean"] = jaccs.mean() if len(jaccs) else 0
            row[f"k{k}_jacc_min"] = jaccs.min() if len(jaccs) else 0

            row[f"k{k}_center_step_mean"] = center_steps.mean() if len(center_steps) else 0
            row[f"k{k}_center_step_max"] = center_steps.max() if len(center_steps) else 0

            row[f"k{k}_rank_gap_mean"] = np.mean(e_ranks - c_ranks)
            row[f"k{k}_rank_gap_slope"] = np.polyfit(np.arange(len(c_ranks)), e_ranks - c_ranks, 1)[0]

            # trajectory curvature
            diffs = np.diff(centers, axis=0)
            speeds = np.linalg.norm(diffs, axis=1)
            row[f"k{k}_traj_speed_mean"] = speeds.mean()
            row[f"k{k}_traj_speed_std"] = speeds.std()
            row[f"k{k}_traj_speed_max"] = speeds.max()

            if len(diffs) >= 2:
                accel = np.diff(diffs, axis=0)
                acc_norm = np.linalg.norm(accel, axis=1)
                row[f"k{k}_traj_accel_mean"] = acc_norm.mean()
                row[f"k{k}_traj_accel_max"] = acc_norm.max()
            else:
                row[f"k{k}_traj_accel_mean"] = 0
                row[f"k{k}_traj_accel_max"] = 0

        feats.append(row)

    return pd.DataFrame(feats)
def w_geometry_features(c_ids, e_ids, k_list=(50, 100, 500, 1000)):
    feats = []

    Wn = W_cpu / (np.linalg.norm(W_cpu, axis=1, keepdims=True) + 1e-8)

    for c_id, e_id in zip(c_ids, e_ids):
        row = {}

        wc = Wn[c_id]
        we = Wn[e_id]

        ce_cos = float(np.dot(wc, we))
        row["W_CE_cos"] = ce_cos
        row["W_CE_dist"] = float(1.0 - ce_cos)

        sim_c = Wn @ wc
        sim_e = Wn @ we

        for k in k_list:
            top_c = np.argpartition(-sim_c, k + 1)[:k + 1]
            top_e = np.argpartition(-sim_e, k + 1)[:k + 1]

            top_c = top_c[top_c != c_id][:k]
            top_e = top_e[top_e != e_id][:k]

            c_density = float(np.mean(sim_c[top_c]))
            e_density = float(np.mean(sim_e[top_e]))

            c_spread = float(np.std(sim_c[top_c]))
            e_spread = float(np.std(sim_e[top_e]))

            overlap = len(set(top_c.tolist()) & set(top_e.tolist()))
            union = len(set(top_c.tolist()) | set(top_e.tolist()))

            row[f"W_C_density_k{k}"] = c_density
            row[f"W_E_density_k{k}"] = e_density
            row[f"W_density_gap_k{k}"] = c_density - e_density

            row[f"W_C_spread_k{k}"] = c_spread
            row[f"W_E_spread_k{k}"] = e_spread
            row[f"W_spread_gap_k{k}"] = c_spread - e_spread

            row[f"W_CE_jaccard_k{k}"] = overlap / max(union, 1)

            # C 是否在 E 的邻域，E 是否在 C 的邻域
            row[f"W_E_in_C_top{k}"] = int(e_id in top_c)
            row[f"W_C_in_E_top{k}"] = int(c_id in top_e)

        feats.append(row)

    return pd.DataFrame(feats)

def r_space_features(H_by_layer, c_ids, e_ids):
    feats = []

    for i in range(len(c_ids)):
        row = {}
        R = []

        for l in DECISION_LAYERS:
            h = H_by_layer[l][i]
            logits = h @ W_cpu.T
            r = float(logits[c_ids[i]] - logits[e_ids[i]])
            R.append(r)
            row[f"R{l}"] = r

        R = np.array(R)
        v = np.diff(R)
        a = np.diff(v)

        row["R_mean"] = R.mean()
        row["R_min"] = R.min()
        row["R_max"] = R.max()
        row["R_final"] = R[-1]
        row["R_slope"] = np.polyfit(np.arange(len(R)), R, 1)[0]
        row["v_mean"] = v.mean()
        row["v_std"] = v.std()
        row["a_mean"] = a.mean() if len(a) else 0
        row["a_std"] = a.std() if len(a) else 0
        row["R_abs_min"] = np.min(np.abs(R))
        row["R_cross_zero"] = int(np.any(np.sign(R[:-1]) != np.sign(R[1:])))

        feats.append(row)

    return pd.DataFrame(feats)

# =========================
# COLLECT HIDDEN STATES
# =========================

all_topk = []
all_r = []

print("Extracting features...")

for start in range(0, len(df), BATCH_SIZE):
    sub = df.iloc[start:start+BATCH_SIZE].reset_index(drop=True)
    H = forward_hidden(sub["prompt"].tolist())

    c_ids = sub["C_id"].tolist()
    e_ids = sub["E_id"].tolist()

    topk_f = topk_features_from_hidden(H, c_ids, e_ids)
    r_f = r_space_features(H, c_ids, e_ids)

    all_topk.append(topk_f)
    all_r.append(r_f)

    del H
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

topk_df = pd.concat(all_topk, ignore_index=True)
r_df = pd.concat(all_r, ignore_index=True)

w_df = w_geometry_features(
    df["C_id"].tolist(),
    df["E_id"].tolist()
)

full = pd.concat(
    [df.reset_index(drop=True), r_df, topk_df, w_df],
    axis=1
)

# =========================
# DEFINE FAILURE TARGETS
# =========================
# Proxy targets for 4A rollout failure:
#   1. Gen_E proxy: final R26/R25 negative
#   2. Overshoot proxy: large late R jump
#   3. Oscillation proxy: sign changes in velocity
# Later you can replace these with real 4A rollout error columns.

R_cols = [f"R{l}" for l in DECISION_LAYERS]
R_mat = full[R_cols].values

v_mat = np.diff(R_mat, axis=1)

full["target_GenE_proxy"] = (full["R_final"] < 0).astype(int)
full["target_overshoot_proxy"] = (
    np.abs(v_mat[:, -1]) > np.percentile(np.abs(v_mat[:, -1]), 75)
).astype(int)

sign_changes = []
for v in v_mat:
    sign_changes.append(np.sum(np.sign(v[:-1]) != np.sign(v[1:])))
full["target_oscillation_proxy"] = (np.array(sign_changes) >= 2).astype(int)

# regression target: instability energy
full["target_instability_energy"] = (
    np.abs(full["R_final"])
    + np.std(v_mat, axis=1)
    + full["R_abs_min"]
)

# =========================
# EVALUATION
# =========================

meta_cols = {
    "graph_id", "condition", "phase", "A", "B", "C_label", "E_label",
    "C_id", "E_id", "prompt",
    "target_GenE_proxy", "target_overshoot_proxy",
    "target_oscillation_proxy", "target_instability_energy"
}

r_feature_cols = [c for c in r_df.columns]
topk_feature_cols = [c for c in topk_df.columns]
w_feature_cols = [c for c in w_df.columns]

hybrid_feature_cols = r_feature_cols + topk_feature_cols
w_conditioned_feature_cols = r_feature_cols + topk_feature_cols + w_feature_cols

def eval_classification(feature_cols, target_col, name):
    X = full[feature_cols].values
    y = full[target_col].values
    groups = full["graph_id"].values

    rows = []

    gkf = GroupKFold(n_splits=5)

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue

        model_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])

        model_pipe.fit(X[tr], y[tr])
        prob = model_pipe.predict_proba(X[te])[:, 1]
        pred = (prob >= 0.5).astype(int)

        rows.append({
            "feature_set": name,
            "target": target_col,
            "fold": fold,
            "auc": roc_auc_score(y[te], prob),
            "acc": accuracy_score(y[te], pred),
            "f1": f1_score(y[te], pred),
        })

    return pd.DataFrame(rows)

def eval_regression(feature_cols, target_col, name):
    X = full[feature_cols].values
    y = full[target_col].values
    groups = full["graph_id"].values

    rows = []

    gkf = GroupKFold(n_splits=5)

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        model_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0)),
        ])

        model_pipe.fit(X[tr], y[tr])
        pred = model_pipe.predict(X[te])

        rows.append({
            "feature_set": name,
            "target": target_col,
            "fold": fold,
            "r2": r2_score(y[te], pred),
            "mae": mean_absolute_error(y[te], pred),
            "corr": np.corrcoef(y[te], pred)[0, 1],
        })

    return pd.DataFrame(rows)

cls_results = []
reg_results = []

for target in [
    "target_GenE_proxy",
    "target_overshoot_proxy",
    "target_oscillation_proxy",
]:
    cls_results.append(eval_classification(r_feature_cols, target, "R_space"))
    cls_results.append(eval_classification(topk_feature_cols, target, "TopK_L7_19"))
    cls_results.append(eval_classification(w_feature_cols, target, "W_only"))
    cls_results.append(eval_classification(hybrid_feature_cols, target, "Hybrid_R_plus_TopK"))
    cls_results.append(eval_classification(w_conditioned_feature_cols, target, "W_conditioned_Hybrid"))

reg_results.append(eval_regression(r_feature_cols, "target_instability_energy", "R_space"))
reg_results.append(eval_regression(topk_feature_cols, "target_instability_energy", "TopK_L7_19"))
reg_results.append(eval_regression(w_feature_cols, "target_instability_energy", "W_only"))
reg_results.append(eval_regression(hybrid_feature_cols, "target_instability_energy", "Hybrid_R_plus_TopK"))
reg_results.append(eval_regression(w_conditioned_feature_cols, "target_instability_energy", "W_conditioned_Hybrid"))

cls_out = pd.concat(cls_results, ignore_index=True)
reg_out = pd.concat(reg_results, ignore_index=True)

cls_summary = cls_out.groupby(
    ["target", "feature_set"]
).agg(
    auc_mean=("auc", "mean"),
    auc_std=("auc", "std"),
    acc_mean=("acc", "mean"),
    acc_std=("acc", "std"),
    f1_mean=("f1", "mean"),
    f1_std=("f1", "std"),
).reset_index()

reg_summary = reg_out.groupby(
    ["target", "feature_set"]
).agg(
    r2_mean=("r2", "mean"),
    r2_std=("r2", "std"),
    mae_mean=("mae", "mean"),
    mae_std=("mae", "std"),
    corr_mean=("corr", "mean"),
    corr_std=("corr", "std"),
).reset_index()

# =========================
# SAVE
# =========================

full.to_csv(SAVE_DIR / "phasemap5a_dataset.csv", index=False)
w_df.to_csv(SAVE_DIR / "phasemap5c_w_features.csv", index=False)
topk_df.to_csv(SAVE_DIR / "phasemap5a_topk_features.csv", index=False)
r_df.to_csv(SAVE_DIR / "phasemap5a_r_features.csv", index=False)

cls_out.to_csv(SAVE_DIR / "phasemap5a_classification_cv.csv", index=False)
reg_out.to_csv(SAVE_DIR / "phasemap5a_regression_cv.csv", index=False)

cls_summary.to_csv(SAVE_DIR / "phasemap5a_classification_summary.csv", index=False)
reg_summary.to_csv(SAVE_DIR / "phasemap5a_regression_summary.csv", index=False)


summary = {
    "n_samples": int(len(full)),
    "n_graphs": int(full["graph_id"].nunique()),
    "topk_list": TOPK_LIST,
    "track_layers": TRACK_LAYERS,
    "decision_layers": DECISION_LAYERS,
    "classification_summary": cls_summary.to_dict(orient="records"),
    "regression_summary": reg_summary.to_dict(orient="records"),
}

with open(SAVE_DIR / "phasemap5a_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\nDone.")
print("Saved to:", SAVE_DIR)
print("\nClassification summary:")
print(cls_summary)

print("\nRegression summary:")
print(reg_summary)