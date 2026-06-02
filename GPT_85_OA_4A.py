# ============================================================
# OA-4A: Prompt Initial Condition -> Constraint Trajectory Audit
#
# Core hypothesis:
#   Prompt is an initial-condition generator:
#
#       Prompt -> xi0 = (R0, B0, F0)
#
#   and inference-time local constraint state is:
#
#       C20:25 = f(xi0, W, T)
#
# Main test:
#   Fix W, model, answer labels, and graph objects.
#   Change only prompt constraint structure.
#   Measure whether xi0 predicts DeltaR20:25.
#
# Outputs:
#   oa4a_dataset.csv
#   oa4a_correlations.csv
#   oa4a_summary.json
# ============================================================

import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    r2_score,
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

SAVE_DIR = Path("./oa4a_prompt_initial_condition_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
MAX_LEN = 260
BATCH_SIZE = 8

N_REPEAT = 4

TRACK_LAYERS = list(range(20, 26))  # L20-L25

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
W = W_raw / (np.linalg.norm(W_raw, axis=1, keepdims=True) + 1e-9)

print("W shape:", W.shape)

# ============================================================
# TOKEN HELPERS
# ============================================================

def encode_no_special(text: str):
    return tokenizer(text, add_special_tokens=False)["input_ids"]

def single_token_id(text: str):
    candidates = [text, " " + text, text.lower(), " " + text.lower()]
    for cand in candidates:
        ids = encode_no_special(cand)
        if len(ids) == 1:
            return ids[0]
    return None

def is_single_token(text: str):
    return single_token_id(text) is not None

def get_vec(text: str):
    tid = single_token_id(text)
    if tid is None:
        return None
    return W[tid]

def cos(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)))

def density(vec, k=50):
    sims = W @ vec
    top = np.sort(sims)[-k-1:-1]
    return float(np.mean(top))

# ============================================================
# GRAPH OBJECTS
# Use single-token labels when possible.
# A belongs_to B, B associated_with C, conflict basin E.
# ============================================================

raw_graphs = [
    ("Paris", "France", "Europe", "Asia"),
    ("Berlin", "Germany", "Europe", "Asia"),
    ("Tokyo", "Japan", "Asia", "Europe"),
    ("Beijing", "China", "Asia", "Europe"),
    ("doctor", "hospital", "medicine", "law"),
    ("judge", "court", "law", "medicine"),
    ("teacher", "school", "education", "finance"),
    ("pilot", "airport", "aviation", "medicine"),
    ("dog", "animal", "life", "machine"),
    ("cat", "animal", "life", "machine"),
    ("river", "water", "nature", "finance"),
    ("sun", "star", "space", "court"),
]

graphs = []
for a, b, c, e in raw_graphs:
    if all(is_single_token(x) for x in [a, b, c, e]):
        graphs.append((a, b, c, e))
    else:
        print("Skipping non-single-token graph:", (a, b, c, e))

if len(graphs) < 6:
    raise RuntimeError("Too few valid single-token graphs.")

print("Valid graphs:", len(graphs))

# ============================================================
# PROMPT CONDITIONS
#
# xi0 = (R0, B0, F0)
#
# R0: initial rule bias toward C minus E.
#     positive -> supports clean C
#     near 0   -> ambiguous / boundary
#     negative -> supports conflict E
#
# B0: boundary pressure.
#     larger means closer to conflict boundary.
#
# F0: freedom / allowed reinterpretation degree.
# ============================================================

CONDITIONS = [
    # condition, mechanism, R0, B0, F0
    ("clean",                 "stable",      +2.0, 0.0, 0.0),
    ("weak_distractor",       "stable",      +1.5, 0.5, 1.0),
    ("ambiguous_branch",      "competition", +0.2, 2.0, 2.0),
    ("direct_conflict",       "competition", -0.8, 3.0, 2.5),
    ("exception_override",    "closure",     -1.5, 4.0, 3.5),
    ("rule_update",           "closure",     -1.8, 4.5, 4.0),
    ("meta_override",         "closure",     -2.2, 5.0, 5.0),
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

    elif condition == "exception_override":
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

    elif condition == "rule_update":
        lines = [
            "You are given a small relation graph.",
            option,
            f"Old rule: {b} is associated with {c}.",
            f"New rule: in this graph, {b} is now associated with {e}.",
            f"Fact: {a} belongs to {b}.",
            f"Question: Which label is {a} associated with after the update?",
            f"Answer with exactly one word: {c} or {e}.",
            "Answer:",
        ]

    elif condition == "meta_override":
        lines = [
            "You are given a small relation graph.",
            option,
            "Use only the latest rule. Ignore older or general rules.",
            f"Latest rule: {b} is associated with {e}.",
            f"Fact: {a} belongs to {b}.",
            f"Question: Which label is {a} associated with under the latest rule?",
            f"Answer with exactly one word: {c} or {e}.",
            "Answer:",
        ]

    else:
        raise ValueError(condition)

    return "\n".join(lines)

# ============================================================
# STATIC W FEATURES
# These are included as comparison baseline, not main target.
# ============================================================

def W_features(a, b, c, e):
    va, vb, vc, ve = get_vec(a), get_vec(b), get_vec(c), get_vec(e)

    ab = cos(va, vb)
    bc = cos(vb, vc)
    ac = cos(va, vc)
    be = cos(vb, ve)
    ae = cos(va, ve)
    ce = cos(vc, ve)

    clean_closure = float(np.mean([ab, bc, ac]))
    conflict_closure = float(np.mean([ab, be, ae]))

    dc = density(vc, k=50)
    de = density(ve, k=50)

    return {
        "W_AB": ab,
        "W_BC": bc,
        "W_AC": ac,
        "W_BE": be,
        "W_AE": ae,
        "W_CE": ce,
        "W_clean_closure": clean_closure,
        "W_conflict_closure": conflict_closure,
        "W_closure_gap": clean_closure - conflict_closure,
        "W_C_density50": dc,
        "W_E_density50": de,
        "W_density_gap_E_minus_C": de - dc,
    }

# ============================================================
# BUILD DATASET
# ============================================================

rows = []
graph_id = 0

for rep in range(N_REPEAT):
    for base_gid, (a, b, c, e) in enumerate(graphs):
        wf = W_features(a, b, c, e)

        for condition, mechanism, R0, B0, F0 in CONDITIONS:
            rows.append({
                "graph_id": graph_id,
                "base_graph_id": base_gid,
                "rep": rep,
                "A": a,
                "B": b,
                "C_clean": c,
                "E_conflict": e,
                "condition": condition,
                "mechanism": mechanism,
                "R0_prompt_bias": R0,
                "B0_boundary_pressure": B0,
                "F0_freedom": F0,
                "xi0_energy": float(np.sqrt(R0**2 + B0**2 + F0**2)),
                "prompt": make_prompt(a, b, c, e, condition),
                **wf,
            })

        graph_id += 1

df = pd.DataFrame(rows)

print("Rows:", len(df))
print("Graphs:", df["graph_id"].nunique())
print("Base graphs:", df["base_graph_id"].nunique())

# ============================================================
# FORWARD: R_l = logit(C) - logit(E)
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

    batch_rows = []

    for bi in range(len(prompts)):
        c_id = single_token_id(clean_targets[bi])
        e_id = single_token_id(conflict_targets[bi])

        item = {}

        for layer in TRACK_LAYERS:
            h = hs[layer + 1][bi, last_idx[bi], :]
            logits = model.lm_head(h).float()
            r = float(logits[c_id].detach().cpu() - logits[e_id].detach().cpu())
            item[f"R{layer}"] = r

        batch_rows.append(item)

    return batch_rows

print("Running forward...")

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
    print(f"Processed {min(start + BATCH_SIZE, len(df))}/{len(df)}")

R_df = pd.DataFrame(R_rows)
df = pd.concat([df.reset_index(drop=True), R_df], axis=1)

# ============================================================
# DeltaR relative to clean condition per graph_id
# ============================================================

for layer in TRACK_LAYERS:
    clean_base = (
        df[df["condition"] == "clean"][["graph_id", f"R{layer}"]]
        .rename(columns={f"R{layer}": f"R{layer}_clean"})
    )
    df = df.merge(clean_base, on="graph_id", how="left")
    df[f"dR{layer}"] = df[f"R{layer}"] - df[f"R{layer}_clean"]

dR_cols = [f"dR{l}" for l in TRACK_LAYERS]

df["deltaR_l2"] = np.sqrt(np.sum(np.square(df[dR_cols].values), axis=1))
df["deltaR_mean"] = np.mean(df[dR_cols].values, axis=1)
df["deltaR_min"] = np.min(df[dR_cols].values, axis=1)
df["deltaR_final25"] = df["dR25"]

df["is_closure"] = (df["mechanism"] == "closure").astype(int)
df["is_competition_or_closure"] = (df["mechanism"] != "stable").astype(int)

# ============================================================
# CORRELATION TABLE
# ============================================================

xi_cols = [
    "R0_prompt_bias",
    "B0_boundary_pressure",
    "F0_freedom",
    "xi0_energy",
]

W_cols = [
    "W_AB",
    "W_BC",
    "W_AC",
    "W_BE",
    "W_AE",
    "W_CE",
    "W_clean_closure",
    "W_conflict_closure",
    "W_closure_gap",
    "W_C_density50",
    "W_E_density50",
    "W_density_gap_E_minus_C",
]

target_cols = dR_cols + [
    "deltaR_l2",
    "deltaR_mean",
    "deltaR_min",
    "deltaR_final25",
]

corr_rows = []

for group_name, cols in [("xi0", xi_cols), ("W", W_cols), ("xi0_plus_W", xi_cols + W_cols)]:
    for xcol in cols:
        for ycol in target_cols:
            x = df[xcol].values.astype(float)
            y = df[ycol].values.astype(float)

            if np.std(x) < 1e-9 or np.std(y) < 1e-9:
                r = np.nan
            else:
                r = float(np.corrcoef(x, y)[0, 1])

            corr_rows.append({
                "feature_group": group_name,
                "x_feature": xcol,
                "target": ycol,
                "pearson": r,
                "abs_corr": abs(r) if not np.isnan(r) else np.nan,
            })

corr_df = pd.DataFrame(corr_rows).sort_values("abs_corr", ascending=False)

# ============================================================
# CV HELPERS
# ============================================================

def regression_cv(feature_cols, target="deltaR_l2", group_col="base_graph_id"):
    X = df[feature_cols].values.astype(float)
    y = df[target].values.astype(float)
    groups = df[group_col].values

    n_splits = min(5, len(np.unique(groups)))
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
        "target": target,
        "group_col": group_col,
        "n_splits": int(n_splits),
        "r2": r2,
        "corr": corr,
    }

def binary_cv(feature_cols, target_col="is_closure", group_col="base_graph_id"):
    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(int)
    groups = df[group_col].values

    n_splits = min(5, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)

    prob = np.zeros(len(y), dtype=float)

    for tr, te in cv.split(X, y, groups):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                solver="lbfgs",
            )),
        ])
        clf.fit(X[tr], y[tr])
        prob[te] = clf.predict_proba(X[te])[:, 1]

    pred = (prob >= 0.5).astype(int)

    return {
        "target": target_col,
        "group_col": group_col,
        "n_splits": int(n_splits),
        "auc": float(roc_auc_score(y, prob)),
        "acc": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
    }

# ============================================================
# SUMMARY
# ============================================================

summary = {
    "experiment": "OA-4A Prompt Initial Condition Audit",
    "hypothesis": "C20:25 = f(xi0, W, T), where xi0=(R0,B0,F0) is prompt-derived initial condition.",
    "n_rows": int(len(df)),
    "n_graphs": int(df["graph_id"].nunique()),
    "n_base_graphs": int(df["base_graph_id"].nunique()),
    "n_repeat": int(N_REPEAT),
    "track_layers": TRACK_LAYERS,

    "feature_sets": {
        "xi0": xi_cols,
        "W": W_cols,
        "xi0_plus_W": xi_cols + W_cols,
    },

    "regression_deltaR_l2": {
        "xi0_only": regression_cv(xi_cols, target="deltaR_l2", group_col="base_graph_id"),
        "W_only": regression_cv(W_cols, target="deltaR_l2", group_col="base_graph_id"),
        "xi0_plus_W": regression_cv(xi_cols + W_cols, target="deltaR_l2", group_col="base_graph_id"),
    },

    "binary_predict_closure": {
        "xi0_only": binary_cv(xi_cols, target_col="is_closure", group_col="base_graph_id"),
        "W_only": binary_cv(W_cols, target_col="is_closure", group_col="base_graph_id"),
        "xi0_plus_W": binary_cv(xi_cols + W_cols, target_col="is_closure", group_col="base_graph_id"),
    },

    "binary_predict_nonstable": {
        "xi0_only": binary_cv(xi_cols, target_col="is_competition_or_closure", group_col="base_graph_id"),
        "W_only": binary_cv(W_cols, target_col="is_competition_or_closure", group_col="base_graph_id"),
        "xi0_plus_W": binary_cv(xi_cols + W_cols, target_col="is_competition_or_closure", group_col="base_graph_id"),
    },

    "condition_means": (
        df.groupby(["condition", "mechanism"])
        [["R0_prompt_bias", "B0_boundary_pressure", "F0_freedom", "deltaR_l2", "deltaR_mean", "deltaR_min", "deltaR_final25"]]
        .mean()
        .reset_index()
        .to_dict(orient="records")
    ),

    "top_correlations": corr_df.head(40).to_dict(orient="records"),
}

# ============================================================
# SAVE
# ============================================================

df.to_csv(SAVE_DIR / "oa4a_dataset.csv", index=False, encoding="utf-8-sig")
corr_df.to_csv(SAVE_DIR / "oa4a_correlations.csv", index=False, encoding="utf-8-sig")

with open(SAVE_DIR / "oa4a_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\nOA-4A finished.")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("Saved to:", SAVE_DIR.resolve())