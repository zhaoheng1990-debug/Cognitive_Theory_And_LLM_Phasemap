# ============================================================
# OA-4I: Prompt Initialization Audit
#
# Goal:
#   Test how Prompt is transformed into an initial state.
#
# Core question:
#   Prompt -> H_init -> DeltaR20:25
#
# Input:
#   oa4a2_lite_dataset.csv
#
# Outputs:
#   oa4i_dataset.csv
#   oa4i_summary.json
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

CSV_PATH = r"C:\Windows\System32\oa4a2_lite_outputs\oa4a2_lite_dataset.csv"

SAVE_DIR = Path("./oa4i_outputs")
SAVE_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 8
MAX_LEN = 256

# hidden_states[0] = embedding output
# hidden_states[1] = after transformer layer 0
# ...
INIT_STATE_INDICES = list(range(0, 8))   # embedding + L0-L6
DECISION_LAYERS = [20, 21, 22, 23, 24, 25]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

if df.columns.duplicated().any():
    df = df.loc[:, ~df.columns.duplicated()].copy()

keep_cols = [
    "graph_id",
    "condition",
    "surface_id",
    "A",
    "B",
    "C",
    "E",
    "prompt",
]

df = df[keep_cols].copy()

required = ["prompt", "C", "E", "graph_id", "condition", "surface_id"]
for c in required:
    if c not in df.columns:
        raise RuntimeError(f"Missing column: {c}")

print("Rows:", len(df))
print("Conditions:", df["condition"].nunique())
print("Surfaces:", df["surface_id"].nunique())

# ============================================================
# LOAD MODEL
# ============================================================

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

# ============================================================
# TOKEN HELPERS
# ============================================================

def encode_no_special(text):
    return tokenizer(text, add_special_tokens=False)["input_ids"]

def single_token_id(text):
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

# ============================================================
# FORWARD
# ============================================================

@torch.no_grad()
def compute_batch(prompts, clean_targets, conflict_targets):
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

    rows = []

    for bi in range(len(prompts)):
        c_id = single_token_id(clean_targets[bi])
        e_id = single_token_id(conflict_targets[bi])

        if c_id is None or e_id is None:
            raise RuntimeError(
                f"Non-single token label: {clean_targets[bi]}, {conflict_targets[bi]}"
            )

        item = {}

        # ---------- initialization window ----------
        init_vecs = []

        for si in INIT_STATE_INDICES:
            h = hs[si][bi, last_idx[bi], :]
            h = h.to(model.lm_head.weight.dtype)
            logits = model.lm_head(h)

            r = float((logits[c_id] - logits[e_id]).detach().cpu())

            if si == 0:
                name = "emb"
            else:
                name = f"L{si-1}"

            item[f"Rinit_{name}"] = r
            init_vecs.append(r)

        init_vecs = np.array(init_vecs, dtype=float)

        item["Rinit_mean"] = float(np.mean(init_vecs))
        item["Rinit_min"] = float(np.min(init_vecs))
        item["Rinit_max"] = float(np.max(init_vecs))
        item["Rinit_range"] = float(np.max(init_vecs) - np.min(init_vecs))
        item["Rinit_l2"] = float(np.sqrt(np.sum(init_vecs ** 2)))

        # ---------- decision window recompute ----------
        for layer in DECISION_LAYERS:
            # existing convention:
            # layer 20 -> hidden_states[21]
            h = hs[layer + 1][bi, last_idx[bi], :]
            h = h.to(model.lm_head.weight.dtype)
            logits = model.lm_head(h)

            r = float((logits[c_id] - logits[e_id]).detach().cpu())
            item[f"R{layer}"] = r

        rows.append(item)

    return rows

# ============================================================
# RUN FORWARD
# ============================================================

all_rows = []

for start in range(0, len(df), BATCH_SIZE):
    sub = df.iloc[start:start + BATCH_SIZE]

    out_rows = compute_batch(
        sub["prompt"].tolist(),
        sub["C"].tolist(),
        sub["E"].tolist(),
    )

    all_rows.extend(out_rows)

    print(f"{min(start+BATCH_SIZE, len(df))}/{len(df)}")

feat_df = pd.DataFrame(all_rows)
df = pd.concat([df.reset_index(drop=True), feat_df], axis=1)

# ============================================================
# DeltaR20:25 relative to clean per graph + surface
# Clear old columns first to avoid _x/_y merge suffix bugs.
# ============================================================

old_cols = []

for layer in DECISION_LAYERS:
    old_cols.extend([
        f"R{layer}_clean",
        f"R{layer}_clean_x",
        f"R{layer}_clean_y",
        f"dR{layer}",
    ])

old_cols.extend([
    "deltaR_l2",
    "deltaR_mean",
    "deltaR_min",
])

old_cols = [c for c in old_cols if c in df.columns]

if old_cols:
    print("Dropping old decision columns:", old_cols)
    df = df.drop(columns=old_cols)

for layer in DECISION_LAYERS:
    clean_base = (
        df[df["condition"] == "clean"]
        [["graph_id", "surface_id", f"R{layer}"]]
        .copy()
    )

    clean_base = clean_base.rename(
        columns={f"R{layer}": f"R{layer}_clean"}
    )

    df = df.merge(
        clean_base,
        on=["graph_id", "surface_id"],
        how="left",
        validate="many_to_one"
    )

    if f"R{layer}_clean" not in df.columns:
        raise RuntimeError(
            f"Failed to create R{layer}_clean. Columns now: {list(df.columns)}"
        )

    df[f"dR{layer}"] = df[f"R{layer}"] - df[f"R{layer}_clean"]

dR_cols = [f"dR{x}" for x in DECISION_LAYERS]

df["deltaR_l2"] = np.sqrt(np.sum(np.square(df[dR_cols].values), axis=1))
df["deltaR_mean"] = np.mean(df[dR_cols].values, axis=1)
df["deltaR_min"] = np.min(df[dR_cols].values, axis=1)
dR_cols = [f"dR{x}" for x in DECISION_LAYERS]

df["deltaR_l2"] = np.sqrt(np.sum(np.square(df[dR_cols].values), axis=1))
df["deltaR_mean"] = np.mean(df[dR_cols].values, axis=1)
df["deltaR_min"] = np.min(df[dR_cols].values, axis=1)

# ============================================================
# Initialization Delta relative to clean per graph + surface
# Clear old init baseline columns first.
# ============================================================

init_cols = [
    c for c in df.columns
    if c.startswith("Rinit_")
    and c not in [
        "Rinit_mean",
        "Rinit_min",
        "Rinit_max",
        "Rinit_range",
        "Rinit_l2",
    ]
]

old_init_cols = []

for col in init_cols:
    old_init_cols.extend([
        f"{col}_clean",
        f"{col}_clean_x",
        f"{col}_clean_y",
        f"d{col}",
    ])

old_init_cols.extend([
    "deltaInit_l2",
    "deltaInit_mean",
    "deltaInit_min",
])

old_init_cols = [c for c in old_init_cols if c in df.columns]

if old_init_cols:
    print("Dropping old init columns:", old_init_cols)
    df = df.drop(columns=old_init_cols)

for col in init_cols:
    clean_base = (
        df[df["condition"] == "clean"]
        [["graph_id", "surface_id", col]]
        .copy()
    )

    clean_base = clean_base.rename(
        columns={col: f"{col}_clean"}
    )

    df = df.merge(
        clean_base,
        on=["graph_id", "surface_id"],
        how="left",
        validate="many_to_one"
    )

    df[f"d{col}"] = df[col] - df[f"{col}_clean"]

dinit_cols = [f"d{c}" for c in init_cols]

df["deltaInit_l2"] = np.sqrt(np.sum(np.square(df[dinit_cols].values), axis=1))
df["deltaInit_mean"] = np.mean(df[dinit_cols].values, axis=1)
df["deltaInit_min"] = np.min(df[dinit_cols].values, axis=1)

# ============================================================
# Variance audit
# ============================================================

surface_var_init = (
    df.groupby(["graph_id", "condition"])[dinit_cols]
    .var()
    .mean()
    .mean()
)

condition_var_init = (
    df.groupby(["graph_id"])[dinit_cols]
    .var()
    .mean()
    .mean()
)

surface_var_decision = (
    df.groupby(["graph_id", "condition"])[dR_cols]
    .var()
    .mean()
    .mean()
)

condition_var_decision = (
    df.groupby(["graph_id"])[dR_cols]
    .var()
    .mean()
    .mean()
)

# ============================================================
# Regression: Can initialization predict decision trajectory?
# ============================================================

def group_regression(feature_cols, target="deltaR_l2"):
    X = df[feature_cols].values.astype(float)
    y = df[target].values.astype(float)
    groups = df["graph_id"].values

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

    return {
        "r2": float(r2_score(y, pred)),
        "corr": float(np.corrcoef(y, pred)[0, 1]),
    }

init_feature_sets = {
    "raw_Rinit": init_cols,
    "delta_Rinit": dinit_cols,
    "summary_Rinit": [
        "Rinit_mean",
        "Rinit_min",
        "Rinit_max",
        "Rinit_range",
        "Rinit_l2",
    ],
    "summary_deltaInit": [
        "deltaInit_l2",
        "deltaInit_mean",
        "deltaInit_min",
    ],
    "all_init": init_cols + dinit_cols + [
        "Rinit_mean",
        "Rinit_min",
        "Rinit_max",
        "Rinit_range",
        "Rinit_l2",
        "deltaInit_l2",
        "deltaInit_mean",
        "deltaInit_min",
    ],
}

regression_results = {
    name: group_regression(cols, target="deltaR_l2")
    for name, cols in init_feature_sets.items()
}

# ============================================================
# Summary
# ============================================================

summary = {
    "experiment": "OA-4I Prompt Initialization Audit",
    "n_rows": int(len(df)),
    "n_graphs": int(df["graph_id"].nunique()),
    "n_conditions": int(df["condition"].nunique()),
    "n_surfaces": int(df["surface_id"].nunique()),

    "init_surface_variance": float(surface_var_init),
    "init_condition_variance": float(condition_var_init),
    "init_surface_over_condition": float(surface_var_init / condition_var_init),

    "decision_surface_variance": float(surface_var_decision),
    "decision_condition_variance": float(condition_var_decision),
    "decision_surface_over_condition": float(surface_var_decision / condition_var_decision),

    "regression_init_to_deltaR": regression_results,

    "condition_means": (
        df.groupby("condition")
        [["deltaInit_l2", "deltaR_l2", "deltaR_mean", "deltaR_min"]]
        .mean()
        .reset_index()
        .to_dict(orient="records")
    )
}

# ============================================================
# SAVE
# ============================================================

df.to_csv(
    SAVE_DIR / "oa4i_dataset.csv",
    index=False,
    encoding="utf-8-sig"
)

with open(SAVE_DIR / "oa4i_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\nSaved to:", SAVE_DIR.resolve())