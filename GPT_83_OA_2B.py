# ============================================================
# OA-2B: W-Closure -> C Dynamics Coupling Audit
#
# Goal:
#   Test whether static closure geometry in W predicts
#   inference-time constraint trajectory ΔR20:25.
#
# Core hypothesis:
#   W_closure_strength(A,B,C)
#       -> ΔR20:25 / closure collapse
#
# Model:
#   Qwen2.5-1.5B-Instruct or compatible causal LM
#
# Outputs:
#   oa2b_dataset.csv
#   oa2b_correlations.csv
#   oa2b_summary.json
# ============================================================

import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

SAVE_DIR = Path("./oa2b_w_closure_outputs")
SAVE_DIR.mkdir(exist_ok=True, parents=True)

SEED = 42
N_REPEAT_PER_TRIPLE = 8
MAX_LEN = 260
BATCH_SIZE = 8

TRACK_LAYERS = list(range(20, 26))  # ΔR20:25

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

VOCAB_SIZE, HIDDEN_DIM = W.shape

print("W shape:", W.shape)

# ============================================================
# TOKEN HELPERS
# ============================================================

def encode_no_special(text: str):
    return tokenizer(
        text,
        add_special_tokens=False,
        return_tensors=None,
    )["input_ids"]

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

def get_vec(text: str):
    tid = single_token_id(text)
    if tid is None:
        return None, None
    return W[tid], tid

def cos(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)))

def valid_token(text: str):
    return single_token_id(text) is not None

# ============================================================
# LABEL POOL
# ============================================================

candidate_labels = [
    "Red", "Blue", "Green", "Yellow",
    "North", "South", "East", "West",
    "Alpha", "Beta", "Gamma", "Delta",
    "Circle", "Square", "Triangle", "Star",
    "Copper", "Silver", "Gold", "Iron",
    "Apple", "Orange", "Lemon", "Pear",
    "River", "Mountain", "Forest", "Ocean",
    "Sun", "Moon", "Cloud", "Stone",
]

LABEL_POOL = [x for x in candidate_labels if single_token_id(x) is not None]

if len(LABEL_POOL) < 12:
    raise RuntimeError("Need at least 12 single-token labels.")

print("Using label pool:", LABEL_POOL)

# ============================================================
# CLOSURE TRIPLES
#
# Each item:
#   A belongs_to B
#   B located_in / part_of C
#
# conflict_C is alternative closure target.
# ============================================================

raw_triples = [
    ("Paris", "France", "Europe", "Asia"),
    ("Berlin", "Germany", "Europe", "Asia"),
    ("Tokyo", "Japan", "Asia", "Europe"),
    ("Beijing", "China", "Asia", "Europe"),

    ("Madrid", "Spain", "Europe", "Africa"),
    ("Rome", "Italy", "Europe", "Asia"),
    ("Seoul", "Korea", "Asia", "Europe"),
    ("Ottawa", "Canada", "America", "Europe"),

    ("doctor", "hospital", "medicine", "law"),
    ("nurse", "hospital", "medicine", "finance"),
    ("judge", "court", "law", "medicine"),
    ("lawyer", "court", "law", "biology"),

    ("teacher", "school", "education", "medicine"),
    ("student", "school", "education", "law"),
    ("chef", "kitchen", "food", "finance"),
    ("pilot", "airport", "aviation", "medicine"),

    ("dog", "animal", "life", "machine"),
    ("cat", "animal", "life", "machine"),
    ("bird", "animal", "life", "stone"),
    ("tree", "plant", "life", "machine"),

    ("river", "water", "nature", "finance"),
    ("mountain", "earth", "nature", "law"),
    ("sun", "star", "space", "court"),
    ("moon", "satellite", "space", "hospital"),
]

triples = []
for item in raw_triples:
    a, b, c, e = item
    va, _ = get_vec(a)
    vb, _ = get_vec(b)
    vc, _ = get_vec(c)
    ve, _ = get_vec(e)
    if va is not None and vb is not None and vc is not None and ve is not None:
        triples.append(item)
    else:
        print("Skipping non-single-token triple:", item)

if len(triples) < 8:
    raise RuntimeError("Too few valid triples. Add more single-token triples.")

print(f"Valid triples: {len(triples)}")

# ============================================================
# STATIC W CLOSURE FEATURES
# ============================================================

def closure_features(a, b, c, e):
    va, _ = get_vec(a)
    vb, _ = get_vec(b)
    vc, _ = get_vec(c)
    ve, _ = get_vec(e)

    ab = cos(va, vb)
    bc = cos(vb, vc)
    ac = cos(va, vc)

    be = cos(vb, ve)
    ae = cos(va, ve)
    ce = cos(vc, ve)

    clean_closure = float(np.mean([ab, bc, ac]))
    conflict_closure = float(np.mean([ab, be, ae]))
    closure_gap = clean_closure - conflict_closure

    # local density around clean target C and conflict target E
    sims_c = W @ vc
    sims_e = W @ ve

    density_c_50 = float(np.mean(np.sort(sims_c)[-51:-1]))
    density_e_50 = float(np.mean(np.sort(sims_e)[-51:-1]))
    density_gap_c_minus_e = density_c_50 - density_e_50

    return {
        "W_AB": ab,
        "W_BC": bc,
        "W_AC": ac,
        "W_BE": be,
        "W_AE": ae,
        "W_CE": ce,
        "W_clean_closure": clean_closure,
        "W_conflict_closure": conflict_closure,
        "W_closure_gap_C_minus_E": closure_gap,
        "W_C_density50": density_c_50,
        "W_E_density50": density_e_50,
        "W_density_gap_C_minus_E": density_gap_c_minus_e,
    }

# ============================================================
# PROMPT CONDITIONS
# ============================================================

CONDITIONS = [
    ("clean", 0.0, "stable"),
    ("surface_irrelevant", 0.5, "stable"),
    ("weak_branch", 1.0, "competition"),
    ("ambiguous_branch", 2.0, "competition"),
    ("direct_conflict", 3.0, "competition"),
    ("closure_negation", 5.0, "closure"),
    ("closure_update", 5.0, "closure"),
    ("closure_exception", 5.0, "closure"),
]

def make_prompt(a, b, c, e, variant):
    option_line = f"Possible answers: {c} or {e}."

    if variant == "clean":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located in {c}.",
            f"Question: Where is {a} located?",
            f"Answer with exactly one word: {c} or {e}.",
            "Answer:",
        ]

    elif variant == "surface_irrelevant":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located in {c}.",
            f"Irrelevant note: another unrelated object is associated with {e}.",
            f"Question: Where is {a} located?",
            f"Answer with exactly one word: {c} or {e}.",
            "Answer:",
        ]

    elif variant == "weak_branch":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located in {c}.",
            f"Weak branch: some unrelated descriptions mention {e}.",
            f"Question: Where is {a} located?",
            f"Answer with exactly one word: {c} or {e}.",
            "Answer:",
        ]

    elif variant == "ambiguous_branch":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located in {c}.",
            f"Ambiguous branch: some descriptions connect {a} with {e}.",
            f"Question: Where is {a} located?",
            f"Answer with exactly one word: {c} or {e}.",
            "Answer:",
        ]

    elif variant == "direct_conflict":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located in {c}.",
            f"Fact 3: {a} is located in {e}.",
            f"Question: Where is {a} located?",
            f"Answer with exactly one word: {c} or {e}.",
            "Answer:",
        ]

    elif variant == "closure_negation":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located in {c}.",
            f"Correction: the relation from {b} to {c} is invalid in this case.",
            f"Fact 3: {b} is located in {e}.",
            f"Question: Where is {a} located?",
            f"Answer with exactly one word: {c} or {e}.",
            "Answer:",
        ]

    elif variant == "closure_update":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Initial rule: {b} is located in {c}.",
            f"Update: in this graph, {b} is now located in {e}.",
            f"Fact: {a} belongs to {b}.",
            f"Question: Where is {a} located after the update?",
            f"Answer with exactly one word: {c} or {e}.",
            "Answer:",
        ]

    elif variant == "closure_exception":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"General rule: items belonging to {b} are located in {c}.",
            f"Exception: {a} is a special case located in {e}.",
            f"Fact: {a} belongs to {b}.",
            f"Question: Where is {a} located?",
            f"Answer with exactly one word: {c} or {e}.",
            "Answer:",
        ]

    else:
        raise ValueError(variant)

    return "\n".join(lines)

# ============================================================
# BUILD DATASET
# ============================================================

rows = []
graph_id = 0

for rep in range(N_REPEAT_PER_TRIPLE):
    for a, b, c, e in triples:
        static_feat = closure_features(a, b, c, e)

        for cond, eps, mech in CONDITIONS:
            rows.append({
                "graph_id": graph_id,
                "rep": rep,
                "A": a,
                "B": b,
                "C_clean": c,
                "E_conflict": e,
                "condition": cond,
                "epsilon": eps,
                "mechanism": mech,
                "prompt": make_prompt(a, b, c, e, cond),
                **static_feat,
            })

        graph_id += 1

df = pd.DataFrame(rows)

print("Dataset rows:", len(df))
print("Graphs:", df["graph_id"].nunique())

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

    hidden_states = out.hidden_states
    attn = enc["attention_mask"]
    last_idx = attn.sum(dim=1) - 1

    batch_rows = []

    for bi in range(len(prompts)):
        c_id = single_token_id(clean_targets[bi])
        e_id = single_token_id(conflict_targets[bi])

        if c_id is None or e_id is None:
            raise RuntimeError("Target tokenization failed.")

        item = {}

        for layer in TRACK_LAYERS:
            h = hidden_states[layer + 1][bi, last_idx[bi], :]
            logits = model.lm_head(h).float()
            r = float(logits[c_id].detach().cpu() - logits[e_id].detach().cpu())
            item[f"R{layer}"] = r

        batch_rows.append(item)

    return batch_rows

all_R = []

for start in range(0, len(df), BATCH_SIZE):
    sub = df.iloc[start:start + BATCH_SIZE]

    batch_R = compute_R_batch(
        sub["prompt"].tolist(),
        sub["C_clean"].tolist(),
        sub["E_conflict"].tolist(),
    )

    all_R.extend(batch_R)

    print(f"Processed {min(start + BATCH_SIZE, len(df))}/{len(df)}")

R_df = pd.DataFrame(all_R)
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

df["is_closure"] = (df["mechanism"] == "closure").astype(int)
df["is_nonstable"] = (df["mechanism"] != "stable").astype(int)

# ============================================================
# CORRELATIONS
# ============================================================

W_closure_cols = [
    "W_AB",
    "W_BC",
    "W_AC",
    "W_BE",
    "W_AE",
    "W_CE",
    "W_clean_closure",
    "W_conflict_closure",
    "W_closure_gap_C_minus_E",
    "W_C_density50",
    "W_E_density50",
    "W_density_gap_C_minus_E",
]

C_cols = dR_cols + [
    "deltaR_l2",
    "deltaR_mean",
    "deltaR_min",
    "deltaR_final",
]

corr_rows = []

for wc in W_closure_cols:
    for cc in C_cols:
        x = df[wc].values.astype(float)
        y = df[cc].values.astype(float)

        if np.std(x) < 1e-9 or np.std(y) < 1e-9:
            r = np.nan
        else:
            r = float(np.corrcoef(x, y)[0, 1])

        corr_rows.append({
            "W_feature": wc,
            "C_feature": cc,
            "pearson": r,
            "abs_corr": abs(r) if not np.isnan(r) else np.nan,
        })

corr_df = pd.DataFrame(corr_rows).sort_values("abs_corr", ascending=False)

# ============================================================
# PREDICTION TESTS
# ============================================================

def binary_cv_metrics(X, y, groups=None, group=False):
    y = np.asarray(y).astype(int)

    if group:
        cv = GroupKFold(n_splits=5)
        splits = cv.split(X, y, groups)
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        splits = cv.split(X, y)

    prob = np.zeros(len(y), dtype=float)

    for tr, te in splits:
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
        "auc": float(roc_auc_score(y, prob)),
        "acc": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
    }

def regression_cv_r2(X, y, groups=None, group=False):
    y = np.asarray(y).astype(float)

    if group:
        cv = GroupKFold(n_splits=5)
        splits = cv.split(X, y, groups)
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        # bin y for stratification
        bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
        splits = cv.split(X, bins)

    pred = np.zeros(len(y), dtype=float)

    for tr, te in splits:
        reg = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ])
        reg.fit(X[tr], y[tr])
        pred[te] = reg.predict(X[te])

    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-9
    r2 = 1.0 - ss_res / ss_tot

    if np.std(pred) < 1e-9:
        corr = np.nan
    else:
        corr = float(np.corrcoef(y, pred)[0, 1])

    return {
        "r2": float(r2),
        "corr": corr,
    }

X_closure = df[W_closure_cols].values.astype(float)
groups = df["graph_id"].values

summary = {
    "n_rows": int(len(df)),
    "n_graphs": int(df["graph_id"].nunique()),
    "n_triples": int(len(triples)),
    "n_repeat_per_triple": int(N_REPEAT_PER_TRIPLE),
    "W_closure_features": W_closure_cols,
    "C_features": C_cols,

    "W_closure_only_predict_closure_stratified": binary_cv_metrics(
        X_closure,
        df["is_closure"].values,
        group=False,
    ),
    "W_closure_only_predict_closure_group": binary_cv_metrics(
        X_closure,
        df["is_closure"].values,
        groups=groups,
        group=True,
    ),

    "W_closure_only_predict_nonstable_stratified": binary_cv_metrics(
        X_closure,
        df["is_nonstable"].values,
        group=False,
    ),
    "W_closure_only_predict_nonstable_group": binary_cv_metrics(
        X_closure,
        df["is_nonstable"].values,
        groups=groups,
        group=True,
    ),

    "W_closure_to_deltaR_l2_regression_stratified": regression_cv_r2(
        X_closure,
        df["deltaR_l2"].values,
        group=False,
    ),
    "W_closure_to_deltaR_l2_regression_group": regression_cv_r2(
        X_closure,
        df["deltaR_l2"].values,
        groups=groups,
        group=True,
    ),

    "top_abs_correlations": corr_df.head(30).to_dict(orient="records"),
}

# ============================================================
# SAVE
# ============================================================

df.to_csv(SAVE_DIR / "oa2b_dataset.csv", index=False, encoding="utf-8-sig")
corr_df.to_csv(SAVE_DIR / "oa2b_correlations.csv", index=False, encoding="utf-8-sig")

with open(SAVE_DIR / "oa2b_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\nOA-2B finished.")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("Saved to:", SAVE_DIR.resolve())