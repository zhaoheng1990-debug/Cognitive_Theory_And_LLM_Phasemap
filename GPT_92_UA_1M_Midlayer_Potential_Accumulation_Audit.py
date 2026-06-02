# ============================================================
# UA-1M: Mid-layer Potential Accumulation Audit
#
# Goal:
#   Test the existence of a measurable potential-gap-like quantity
#   in the middle layers L7-L19.
#
# Core hypothesis:
#
#   L0-L6:
#       broad relevance / task manifold
#
#   L7-L19:
#       potential gap accumulation local phase space
#
#   L20-L22:
#       basin selection transition / critical entry
#
#   L23-L26:
#       answer basin formation
#
#   L27:
#       output commitment
#
# Observable:
#
#   Gap_l = logit_l(C) - logit_l(E)
#
# Interpretation:
#
#   Gap_l > 0:
#       clean basin C has lower effective potential than conflict basin E.
#
#   Gap_l ≈ 0:
#       competing basins near critical balance.
#
#   Gap_l < 0:
#       conflict basin E dominates.
#
# This is not claiming U = -log P is fully proven.
# It only tests whether a continuous layerwise potential-gap-like
# order parameter exists.
#
# Outputs:
#   ua1m_dataset.csv
#   ua1m_gap_trajectory.csv
#   ua1m_feature_table.csv
#   ua1m_prediction_summary.csv
#   ua1m_condition_profile.csv
#   ua1m_summary.json
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
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    r2_score,
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

SAVE_DIR = Path("./ua1m_midlayer_potential_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

N_GRAPHS = 72
BATCH_SIZE = 8
MAX_LEN = 260

# Track all relevant layers. For Qwen2.5-1.5B, layers are 0..27.
# hidden_states[0] = embedding output
# hidden_states[layer + 1] = after transformer layer `layer`
TRACK_LAYERS = list(range(0, 28))

MID_LAYERS = list(range(7, 20))
PRE_CRITICAL_LAYERS = list(range(16, 20))
CRITICAL_LAYERS = list(range(20, 23))
BASIN_LAYERS = list(range(23, 27))
COMMIT_LAYER = 27

RUN_GENERATION = True
GEN_MAX_NEW_TOKENS = 4
GEN_BATCH_SIZE = 8

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

N_LAYERS = len(model.model.layers)
print("Detected layers:", N_LAYERS)

if max(TRACK_LAYERS) >= N_LAYERS:
    raise RuntimeError(
        f"TRACK_LAYERS contains {max(TRACK_LAYERS)}, but model has only {N_LAYERS} layers."
    )

# ============================================================
# TOKEN HELPERS
# ============================================================

def token_ids_no_special(text: str):
    return tokenizer(text, add_special_tokens=False)["input_ids"]

def continuation_ids(label: str):
    # For causal LM continuation after "Answer:", labels usually appear with leading space.
    return token_ids_no_special(" " + label)

def single_token_id(label: str):
    candidates = [
        " " + label,
        label,
        " " + label.lower(),
        label.lower(),
    ]
    for cand in candidates:
        ids = token_ids_no_special(cand)
        if len(ids) == 1:
            return ids[0]
    return None

def decode_ids(ids):
    return tokenizer.decode(ids, skip_special_tokens=True)

def select_single_token_labels():
    candidates = [
        "Red", "Blue", "Green", "Yellow",
        "North", "South", "East", "West",
        "Alpha", "Beta", "Gamma", "Delta",
        "Circle", "Square", "Triangle", "Star",
        "Copper", "Silver", "Gold", "Iron",
        "Apple", "Orange", "Lemon", "Pear",
        "River", "Mountain", "Forest", "Ocean",
        "Sun", "Moon", "Cloud", "Stone",
    ]

    valid = []
    print("\nLabel tokenization audit:")
    for lab in candidates:
        tid = single_token_id(lab)
        ids_space = continuation_ids(lab)
        print(f"  {lab:<10} single_id={tid} continuation_ids={ids_space}")
        if tid is not None:
            valid.append(lab)

    if len(valid) < 16:
        raise RuntimeError("Too few single-token labels. Adjust LABEL candidates.")

    return valid

LABEL_POOL = select_single_token_labels()

# ============================================================
# DATASET
# ============================================================

ENTITY_POOL = [
    ("Paris", "France"),
    ("Berlin", "Germany"),
    ("Tokyo", "Japan"),
    ("Beijing", "China"),
    ("doctor", "hospital"),
    ("judge", "court"),
    ("teacher", "school"),
    ("dog", "animal"),
    ("cat", "animal"),
    ("river", "water"),
    ("tree", "plant"),
    ("chef", "kitchen"),
    ("pilot", "airport"),
    ("student", "classroom"),
    ("painter", "studio"),
    ("nurse", "clinic"),
    ("shark", "ocean"),
    ("camel", "desert"),
]

CONDITIONS = [
    {
        "condition": "clean",
        "mechanism": "stable",
        "epsilon": 0.0,
        "expected_sign": +1,
    },
    {
        "condition": "weak_distractor",
        "mechanism": "stable_shift",
        "epsilon": 1.0,
        "expected_sign": +1,
    },
    {
        "condition": "ambiguous_branch",
        "mechanism": "competition",
        "epsilon": 2.0,
        "expected_sign": 0,
    },
    {
        "condition": "direct_conflict",
        "mechanism": "competition",
        "epsilon": 3.0,
        "expected_sign": -1,
    },
    {
        "condition": "rule_update",
        "mechanism": "closure",
        "epsilon": 4.0,
        "expected_sign": -1,
    },
    {
        "condition": "exception_override",
        "mechanism": "closure",
        "epsilon": 5.0,
        "expected_sign": -1,
    },
]

def make_prompt(A, B, C, E, condition):
    options = f"{C} or {E}"

    if condition == "clean":
        lines = [
            "You are given a small relation graph.",
            f"Possible labels: {options}.",
            f"Fact 1: {A} belongs to {B}.",
            f"Fact 2: {B} is associated with {C}.",
            f"Question: Which label is {A} associated with?",
            f"Answer with exactly one word: {options}.",
            "Answer:",
        ]

    elif condition == "weak_distractor":
        lines = [
            "You are given a small relation graph.",
            f"Possible labels: {options}.",
            f"Fact 1: {A} belongs to {B}.",
            f"Fact 2: {B} is associated with {C}.",
            f"Unrelated note: another object is sometimes associated with {E}.",
            f"Question: Which label is {A} associated with?",
            f"Answer with exactly one word: {options}.",
            "Answer:",
        ]

    elif condition == "ambiguous_branch":
        lines = [
            "You are given a small relation graph.",
            f"Possible labels: {options}.",
            f"Fact 1: {A} belongs to {B}.",
            f"Fact 2: {B} is associated with {C}.",
            f"Ambiguous branch: some descriptions also connect {A} with {E}.",
            f"Question: Which label is {A} associated with?",
            f"Answer with exactly one word: {options}.",
            "Answer:",
        ]

    elif condition == "direct_conflict":
        lines = [
            "You are given a small relation graph.",
            f"Possible labels: {options}.",
            f"Fact 1: {A} belongs to {B}.",
            f"Fact 2: {B} is associated with {C}.",
            f"Fact 3: {A} is associated with {E}.",
            f"Question: Which label is {A} associated with?",
            f"Answer with exactly one word: {options}.",
            "Answer:",
        ]

    elif condition == "rule_update":
        lines = [
            "You are given a small relation graph.",
            f"Possible labels: {options}.",
            f"Old rule: {B} is associated with {C}.",
            f"New rule: in this graph, {B} is associated with {E}.",
            f"Fact: {A} belongs to {B}.",
            "Use the new rule.",
            f"Question: Which label is {A} associated with?",
            f"Answer with exactly one word: {options}.",
            "Answer:",
        ]

    elif condition == "exception_override":
        lines = [
            "You are given a small relation graph.",
            f"Possible labels: {options}.",
            f"General rule: objects connected through {B} use label {C}.",
            f"Exception: {A} is a special case using label {E}.",
            f"Question: Which label is {A} associated with?",
            f"Answer with exactly one word: {options}.",
            "Answer:",
        ]

    else:
        raise ValueError(condition)

    return "\n".join(lines)

def build_dataset():
    rows = []
    label_pairs = []

    labels = LABEL_POOL.copy()
    # deterministic pairing
    for i in range(0, len(labels) - 1, 2):
        label_pairs.append((labels[i], labels[i + 1]))

    gid = 0
    for i in range(N_GRAPHS):
        A, B = ENTITY_POOL[i % len(ENTITY_POOL)]
        C, E = label_pairs[i % len(label_pairs)]

        for cmeta in CONDITIONS:
            condition = cmeta["condition"]

            rows.append({
                "graph_id": gid,
                "A": A,
                "B": B,
                "C": C,
                "E": E,
                "condition": condition,
                "mechanism": cmeta["mechanism"],
                "epsilon": cmeta["epsilon"],
                "expected_sign": cmeta["expected_sign"],
                "prompt": make_prompt(A, B, C, E, condition),
            })

        gid += 1

    return pd.DataFrame(rows)

df = build_dataset()

print("\nDataset:")
print(df.head())
print("Rows:", len(df))
print("Graphs:", df["graph_id"].nunique())
print("Conditions:", df["condition"].unique().tolist())

# ============================================================
# FORWARD: GAP TRAJECTORY
# ============================================================

@torch.no_grad()
def compute_gap_batch(prompts, clean_labels, conflict_labels):
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
        c_id = single_token_id(clean_labels[bi])
        e_id = single_token_id(conflict_labels[bi])

        if c_id is None or e_id is None:
            raise RuntimeError(f"Label not single-token: {clean_labels[bi]} / {conflict_labels[bi]}")

        item = {}

        for layer in TRACK_LAYERS:
            # hidden_states[layer + 1] is after transformer layer `layer`.
            h = hs[layer + 1][bi, last_idx[bi], :]
            h = h.to(model.lm_head.weight.dtype)

            logits = model.lm_head(h)

            c_logit = float(logits[c_id].detach().cpu())
            e_logit = float(logits[e_id].detach().cpu())
            gap = c_logit - e_logit

            # rank: smaller is better
            rank_c = int((logits > logits[c_id]).sum().detach().cpu().item() + 1)
            rank_e = int((logits > logits[e_id]).sum().detach().cpu().item() + 1)

            item[f"C_logit_L{layer}"] = c_logit
            item[f"E_logit_L{layer}"] = e_logit
            item[f"gap_L{layer}"] = gap
            item[f"rank_C_L{layer}"] = rank_c
            item[f"rank_E_L{layer}"] = rank_e
            item[f"rank_gap_EminusC_L{layer}"] = rank_e - rank_c

        batch_rows.append(item)

    return batch_rows

all_gap_rows = []

for start in range(0, len(df), BATCH_SIZE):
    sub = df.iloc[start:start + BATCH_SIZE]

    gap_rows = compute_gap_batch(
        sub["prompt"].tolist(),
        sub["C"].tolist(),
        sub["E"].tolist(),
    )

    all_gap_rows.extend(gap_rows)

    print(f"Gap forward {min(start + BATCH_SIZE, len(df))}/{len(df)}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

gap_df = pd.DataFrame(all_gap_rows)
df = pd.concat([df.reset_index(drop=True), gap_df], axis=1)

# ============================================================
# OPTIONAL GENERATION
# ============================================================

@torch.no_grad()
def generate_batch(prompts):
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    out = model.generate(
        **enc,
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    input_lens = enc["attention_mask"].sum(dim=1).detach().cpu().numpy().tolist()

    gens = []
    for i in range(out.shape[0]):
        gen_ids = out[i, input_lens[i]:].detach().cpu().tolist()
        gens.append(decode_ids(gen_ids).strip())

    return gens

if RUN_GENERATION:
    generations = []
    for start in range(0, len(df), GEN_BATCH_SIZE):
        sub = df.iloc[start:start + GEN_BATCH_SIZE]
        gens = generate_batch(sub["prompt"].tolist())
        generations.extend(gens)
        print(f"Generation {min(start + GEN_BATCH_SIZE, len(df))}/{len(df)}")

    df["generation"] = generations

    gen_c = []
    gen_e = []
    gen_other = []

    for _, row in df.iterrows():
        g = str(row["generation"]).strip().lower()
        c = str(row["C"]).strip().lower()
        e = str(row["E"]).strip().lower()

        is_c = int(g.startswith(c) or c in g.split())
        is_e = int(g.startswith(e) or e in g.split())

        gen_c.append(is_c)
        gen_e.append(is_e)
        gen_other.append(int((is_c == 0) and (is_e == 0)))

    df["Gen_C"] = gen_c
    df["Gen_E"] = gen_e
    df["Gen_Other"] = gen_other
else:
    df["generation"] = ""
    df["Gen_C"] = 0
    df["Gen_E"] = (df[f"gap_L{COMMIT_LAYER}"] < 0).astype(int)
    df["Gen_Other"] = 0

# ============================================================
# FEATURE EXTRACTION: POTENTIAL GAP ACCUMULATION
# ============================================================

def layer_values(row, prefix, layers):
    return np.array([float(row[f"{prefix}_L{l}"]) for l in layers], dtype=float)

feature_rows = []

for idx, row in df.iterrows():
    gaps_all = layer_values(row, "gap", TRACK_LAYERS)
    gaps_mid = layer_values(row, "gap", MID_LAYERS)
    gaps_precrit = layer_values(row, "gap", PRE_CRITICAL_LAYERS)
    gaps_crit = layer_values(row, "gap", CRITICAL_LAYERS)
    gaps_basin = layer_values(row, "gap", BASIN_LAYERS)

    # First derivative and second derivative in middle phase
    d_mid = np.diff(gaps_mid)
    dd_mid = np.diff(gaps_mid, n=2)

    # Basic mid-layer potential accumulation features
    mid_slope = (gaps_mid[-1] - gaps_mid[0]) / max(len(MID_LAYERS) - 1, 1)
    mid_area = float(np.sum(gaps_mid))
    mid_mean = float(np.mean(gaps_mid))
    mid_min = float(np.min(gaps_mid))
    mid_max = float(np.max(gaps_mid))
    mid_range = float(mid_max - mid_min)
    mid_abs_area = float(np.sum(np.abs(gaps_mid)))
    mid_d_mean = float(np.mean(d_mid))
    mid_d_abs_mean = float(np.mean(np.abs(d_mid)))
    mid_dd_mean = float(np.mean(dd_mid)) if len(dd_mid) else 0.0
    mid_dd_abs_mean = float(np.mean(np.abs(dd_mid))) if len(dd_mid) else 0.0

    # Directional consistency relative to final/expected sign
    expected_sign = int(row["expected_sign"])
    if expected_sign == 0:
        directional_consistency = float(np.mean(np.abs(gaps_mid) < np.std(gaps_all) + 1e-9))
    else:
        directional_consistency = float(np.mean(np.sign(gaps_mid) == expected_sign))

    # Critical and basin summaries
    precrit_mean = float(np.mean(gaps_precrit))
    precrit_final = float(gaps_precrit[-1])
    crit_mean = float(np.mean(gaps_crit))
    crit_min = float(np.min(gaps_crit))
    crit_max = float(np.max(gaps_crit))
    basin_mean = float(np.mean(gaps_basin))
    basin_min = float(np.min(gaps_basin))
    basin_max = float(np.max(gaps_basin))
    commit_gap = float(row[f"gap_L{COMMIT_LAYER}"])

    # Transition-like quantities
    crit_jump_from_pre = crit_mean - precrit_final
    basin_jump_from_crit = basin_mean - crit_mean
    commit_jump_from_basin = commit_gap - basin_mean

    # Candidate threshold-crossing layer.
    # Uses a per-row adaptive threshold based on mid-layer variability.
    threshold = max(0.25, float(np.std(gaps_mid)))
    first_abs_cross = None
    first_sign_cross = None

    for l in TRACK_LAYERS:
        g = float(row[f"gap_L{l}"])
        if first_abs_cross is None and abs(g) > threshold:
            first_abs_cross = l

        if expected_sign != 0:
            if first_sign_cross is None and np.sign(g) == expected_sign and abs(g) > threshold:
                first_sign_cross = l

    feature_rows.append({
        "row_id": idx,
        "graph_id": row["graph_id"],
        "condition": row["condition"],
        "mechanism": row["mechanism"],
        "epsilon": row["epsilon"],
        "expected_sign": expected_sign,
        "Gen_C": int(row["Gen_C"]),
        "Gen_E": int(row["Gen_E"]),

        "mid_slope_7_19": mid_slope,
        "mid_area_7_19": mid_area,
        "mid_mean_7_19": mid_mean,
        "mid_min_7_19": mid_min,
        "mid_max_7_19": mid_max,
        "mid_range_7_19": mid_range,
        "mid_abs_area_7_19": mid_abs_area,
        "mid_d_mean_7_19": mid_d_mean,
        "mid_d_abs_mean_7_19": mid_d_abs_mean,
        "mid_dd_mean_7_19": mid_dd_mean,
        "mid_dd_abs_mean_7_19": mid_dd_abs_mean,
        "mid_directional_consistency": directional_consistency,

        "precrit_mean_16_19": precrit_mean,
        "precrit_final_19": precrit_final,
        "crit_mean_20_22": crit_mean,
        "crit_min_20_22": crit_min,
        "crit_max_20_22": crit_max,
        "basin_mean_23_26": basin_mean,
        "basin_min_23_26": basin_min,
        "basin_max_23_26": basin_max,
        "commit_gap_27": commit_gap,

        "crit_jump_from_pre": crit_jump_from_pre,
        "basin_jump_from_crit": basin_jump_from_crit,
        "commit_jump_from_basin": commit_jump_from_basin,

        "adaptive_threshold": threshold,
        "first_abs_cross_layer": -1 if first_abs_cross is None else int(first_abs_cross),
        "first_expected_sign_cross_layer": -1 if first_sign_cross is None else int(first_sign_cross),
    })

feat = pd.DataFrame(feature_rows)

# Merge features back.
# IMPORTANT:
#   `feat` contains metadata columns already present in `df`
#   such as graph_id / condition / mechanism / epsilon.
#   If we concatenate them directly, pandas creates duplicate column names.
#   Later row["condition"] returns a Series instead of a scalar, causing:
#       TypeError: unhashable type: 'Series'
#
#   Therefore we only merge genuinely new feature columns.
meta_cols_already_in_df = {
    "row_id",
    "graph_id",
    "condition",
    "mechanism",
    "epsilon",
    "expected_sign",
    "Gen_C",
    "Gen_E",
}

feat_new_cols = [
    c for c in feat.columns
    if c not in meta_cols_already_in_df
]

df = pd.concat(
    [
        df.reset_index(drop=True),
        feat[feat_new_cols].reset_index(drop=True),
    ],
    axis=1,
)

# Defensive check: no duplicate columns allowed.
dupes = df.columns[df.columns.duplicated()].tolist()
if dupes:
    raise RuntimeError(f"Duplicate columns after feature merge: {dupes}")

# ============================================================
# GAP TRAJECTORY LONG TABLE
# ============================================================

traj_rows = []

for i, row in df.iterrows():
    for l in TRACK_LAYERS:
        traj_rows.append({
            "row_id": i,
            "graph_id": row["graph_id"],
            "condition": row["condition"],
            "mechanism": row["mechanism"],
            "epsilon": row["epsilon"],
            "expected_sign": row["expected_sign"],
            "layer": l,
            "gap": float(row[f"gap_L{l}"]),
            "rank_gap_EminusC": float(row[f"rank_gap_EminusC_L{l}"]),
        })

traj = pd.DataFrame(traj_rows)

# ============================================================
# CONDITION PROFILE
# ============================================================

condition_profile = (
    traj.groupby(["condition", "mechanism", "epsilon", "layer"])[["gap", "rank_gap_EminusC"]]
    .agg(["mean", "std", "median"])
    .reset_index()
)

# flatten columns
condition_profile.columns = [
    "_".join([str(x) for x in col if str(x) != ""]).rstrip("_")
    for col in condition_profile.columns.values
]

# Feature profile by condition
condition_feature_profile = (
    feat.groupby(["condition", "mechanism", "epsilon"])[[
        "mid_slope_7_19",
        "mid_area_7_19",
        "mid_mean_7_19",
        "precrit_final_19",
        "crit_mean_20_22",
        "basin_mean_23_26",
        "commit_gap_27",
        "crit_jump_from_pre",
        "basin_jump_from_crit",
        "mid_directional_consistency",
        "Gen_C",
        "Gen_E",
    ]]
    .mean()
    .reset_index()
)

# ============================================================
# PREDICTION TESTS
# ============================================================

MID_FEATURES = [
    "mid_slope_7_19",
    "mid_area_7_19",
    "mid_mean_7_19",
    "mid_min_7_19",
    "mid_max_7_19",
    "mid_range_7_19",
    "mid_abs_area_7_19",
    "mid_d_mean_7_19",
    "mid_d_abs_mean_7_19",
    "mid_dd_mean_7_19",
    "mid_dd_abs_mean_7_19",
    "mid_directional_consistency",
    "precrit_mean_16_19",
    "precrit_final_19",
]

MID_PLUS_CRIT_FEATURES = MID_FEATURES + [
    "crit_mean_20_22",
    "crit_min_20_22",
    "crit_max_20_22",
    "crit_jump_from_pre",
]

ALL_FEATURES = MID_PLUS_CRIT_FEATURES + [
    "basin_mean_23_26",
    "basin_min_23_26",
    "basin_max_23_26",
    "commit_gap_27",
]

def group_binary_eval(feature_cols, target_col="Gen_E", group_col="graph_id"):
    X = feat[feature_cols].values.astype(float)
    y = feat[target_col].values.astype(int)
    groups = feat[group_col].values

    if len(np.unique(y)) < 2:
        return {"error": f"Target {target_col} has one class only."}

    n_splits = min(5, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)

    prob = np.zeros(len(y), dtype=float)

    for tr, te in cv.split(X, y, groups):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=4000, class_weight="balanced")),
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

def group_multiclass_eval(feature_cols, target_col="mechanism", group_col="graph_id"):
    X = feat[feature_cols].values.astype(float)
    y = feat[target_col].astype(str).values
    groups = feat[group_col].values

    n_splits = min(5, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)

    pred = np.empty(len(y), dtype=object)

    for tr, te in cv.split(X, y, groups):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ])
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])

    return {
        "target": target_col,
        "group_col": group_col,
        "n_splits": int(n_splits),
        "acc": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }

def group_regression_eval(feature_cols, target_col="commit_gap_27", group_col="graph_id"):
    X = feat[feature_cols].values.astype(float)
    y = feat[target_col].values.astype(float)
    groups = feat[group_col].values

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

    corr = float(np.corrcoef(y, pred)[0, 1]) if np.std(pred) > 1e-9 else np.nan

    return {
        "target": target_col,
        "group_col": group_col,
        "n_splits": int(n_splits),
        "r2": float(r2_score(y, pred)),
        "corr": corr,
    }

prediction_summary = {
    "binary_GenE": {
        "mid_only": group_binary_eval(MID_FEATURES, "Gen_E", "graph_id"),
        "mid_plus_critical": group_binary_eval(MID_PLUS_CRIT_FEATURES, "Gen_E", "graph_id"),
        "all_with_basin_commit": group_binary_eval(ALL_FEATURES, "Gen_E", "graph_id"),
    },
    "mechanism_classification": {
        "mid_only": group_multiclass_eval(MID_FEATURES, "mechanism", "graph_id"),
        "mid_plus_critical": group_multiclass_eval(MID_PLUS_CRIT_FEATURES, "mechanism", "graph_id"),
        "all_with_basin_commit": group_multiclass_eval(ALL_FEATURES, "mechanism", "graph_id"),
    },
    "regression_commit_gap": {
        "mid_only": group_regression_eval(MID_FEATURES, "commit_gap_27", "graph_id"),
        "mid_plus_critical": group_regression_eval(MID_PLUS_CRIT_FEATURES, "commit_gap_27", "graph_id"),
        "all_with_basin_commit": group_regression_eval(ALL_FEATURES, "commit_gap_27", "graph_id"),
    },
    "regression_critical_mean": {
        "mid_only": group_regression_eval(MID_FEATURES, "crit_mean_20_22", "graph_id"),
    },
    "regression_basin_mean": {
        "mid_only": group_regression_eval(MID_FEATURES, "basin_mean_23_26", "graph_id"),
    },
}

# ============================================================
# SIMPLE EXISTENCE DIAGNOSTICS
# ============================================================

# Does mid-layer accumulation separate conditions monotonically by epsilon?
eps_corrs = {}
for col in [
    "mid_slope_7_19",
    "mid_area_7_19",
    "mid_mean_7_19",
    "precrit_final_19",
    "crit_mean_20_22",
    "basin_mean_23_26",
    "commit_gap_27",
]:
    x = feat["epsilon"].values.astype(float)
    y = feat[col].values.astype(float)
    eps_corrs[col] = float(np.corrcoef(x, y)[0, 1]) if np.std(y) > 1e-9 else np.nan

# How often does first threshold crossing occur in critical zone?
cross_valid = feat[feat["first_expected_sign_cross_layer"] >= 0]
if len(cross_valid) > 0:
    cross_counts = cross_valid["first_expected_sign_cross_layer"].value_counts().sort_index().to_dict()
    cross_in_mid = float(np.mean(cross_valid["first_expected_sign_cross_layer"].isin(MID_LAYERS)))
    cross_in_critical = float(np.mean(cross_valid["first_expected_sign_cross_layer"].isin(CRITICAL_LAYERS)))
    cross_in_basin = float(np.mean(cross_valid["first_expected_sign_cross_layer"].isin(BASIN_LAYERS)))
else:
    cross_counts = {}
    cross_in_mid = np.nan
    cross_in_critical = np.nan
    cross_in_basin = np.nan

existence_diagnostics = {
    "epsilon_correlations": eps_corrs,
    "threshold_cross_counts": {str(k): int(v) for k, v in cross_counts.items()},
    "fraction_first_expected_cross_in_mid_7_19": cross_in_mid,
    "fraction_first_expected_cross_in_critical_20_22": cross_in_critical,
    "fraction_first_expected_cross_in_basin_23_26": cross_in_basin,
}

# ============================================================
# SAVE
# ============================================================

df.to_csv(SAVE_DIR / "ua1m_dataset.csv", index=False, encoding="utf-8-sig")
traj.to_csv(SAVE_DIR / "ua1m_gap_trajectory.csv", index=False, encoding="utf-8-sig")
feat.to_csv(SAVE_DIR / "ua1m_feature_table.csv", index=False, encoding="utf-8-sig")
condition_profile.to_csv(SAVE_DIR / "ua1m_condition_profile.csv", index=False, encoding="utf-8-sig")
condition_feature_profile.to_csv(SAVE_DIR / "ua1m_condition_feature_profile.csv", index=False, encoding="utf-8-sig")

with open(SAVE_DIR / "ua1m_prediction_summary.json", "w", encoding="utf-8") as f:
    json.dump(prediction_summary, f, ensure_ascii=False, indent=2)

summary = {
    "experiment": "UA-1M Mid-layer Potential Accumulation Audit",
    "n_rows": int(len(df)),
    "n_graphs": int(df["graph_id"].nunique()),
    "n_conditions": int(df["condition"].nunique()),
    "track_layers": TRACK_LAYERS,
    "mid_layers": MID_LAYERS,
    "precritical_layers": PRE_CRITICAL_LAYERS,
    "critical_layers": CRITICAL_LAYERS,
    "basin_layers": BASIN_LAYERS,
    "commit_layer": COMMIT_LAYER,
    "observable": "Gap_l = logit_l(C) - logit_l(E), interpreted as potential-gap proxy U_E - U_C.",
    "prediction_summary": prediction_summary,
    "existence_diagnostics": existence_diagnostics,
    "condition_feature_profile": condition_feature_profile.to_dict(orient="records"),
    "interpretation_guide": {
        "PASS_strong": [
            "mid_only features predict Gen_E or commit_gap strongly",
            "mid_only features predict critical_mean_20_22 / basin_mean_23_26",
            "condition profile shows clean positive, ambiguous near zero, closure/conflict negative",
            "first_expected_sign_cross concentrates around L20-L22 or just before it"
        ],
        "PASS_mixed": [
            "mid features predict mechanism or later gap but not final generation",
            "critical/basin features are needed for Gen_E"
        ],
        "FAIL": [
            "mid features do not predict critical, basin, mechanism, or final gap",
            "gap trajectories are random or dominated only by L26-L27"
        ]
    }
}

with open(SAVE_DIR / "ua1m_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\nSaved to:", SAVE_DIR.resolve())
