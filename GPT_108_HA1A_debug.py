# ============================================================
# HA-1A-debug: Mechanism-Split Hallucination Basin Audit
#
# Fixes from HA-1:
#   1. Do NOT mix competition-collapse and closure-collapse in one PCA.
#   2. Fit separate order parameters:
#
#        ΔU_K = PC1_K(ΔR20:25)  for competition mechanism
#        ΔU_Q = PC1_Q(ΔR20:25)  for closure mechanism
#
#   3. Weak distractor is weakened.
#   4. Closure conditions are analyzed separately.
#
# Outputs:
#   ha1a_debug_outputs/
#       dataset.csv
#       layer_R.csv
#       generation.csv
#       deltaR.csv
#       deltaU_competition.csv
#       deltaU_closure.csv
#       summary.json
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
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

SAVE_DIR = Path("./ha1a_debug_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SEED = 42
N_GRAPHS = 96
BATCH_SIZE = 8
MAX_LEN = 256
GEN_MAX_NEW_TOKENS = 4

TRACK_LAYERS = list(range(20, 28))
DELTA_LAYERS = list(range(20, 26))

COMPETITION_CONDITIONS = [
    ("clean", 0.0, "positive"),
    ("weak_distractor_v2", 0.5, "positive"),
    ("ambiguous_branch", 2.0, "critical"),
    ("strong_conflict", 3.0, "negative"),
]

CLOSURE_CONDITIONS = [
    ("clean", 0.0, "positive"),
    ("closure_negation", 4.0, "negative"),
    ("closure_rewrite", 5.0, "negative"),
    ("closure_override", 5.5, "negative"),
    ("closure_exception", 6.0, "negative"),
]

ALL_CONDITIONS = []
for c in COMPETITION_CONDITIONS + CLOSURE_CONDITIONS:
    if c not in ALL_CONDITIONS:
        ALL_CONDITIONS.append(c)

# ============================================================
# SEED
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ============================================================
# LOAD MODEL
# ============================================================

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

num_layers = len(model.model.layers)
print("Num layers:", num_layers)

if max(TRACK_LAYERS) >= num_layers:
    raise RuntimeError(
        f"Model has {num_layers} layers, but TRACK_LAYERS includes {max(TRACK_LAYERS)}."
    )

# ============================================================
# LABELS
# ============================================================

def token_ids_no_special(text):
    return tokenizer(text, add_special_tokens=False)["input_ids"]

def continuation_ids(label):
    return token_ids_no_special(" " + label)

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

    single = []
    print("\nLabel tokenization audit:")
    for lab in candidates:
        ids = continuation_ids(lab)
        print(f"{lab:<10} ids={ids} len={len(ids)}")
        if len(ids) == 1:
            single.append(lab)

    if len(single) < 12:
        raise RuntimeError("Need at least 12 single-token labels.")

    return single[:12]

LABEL_POOL = select_single_token_labels()
print("Using labels:", LABEL_POOL)

# ============================================================
# DATASET
# ============================================================

def make_entity(prefix, i):
    return f"{prefix}{i:03d}"

def make_prompt(a, b, c_label, e_label, condition):
    option = f"Answer with exactly one word: {c_label} or {e_label}."

    if condition == "clean":
        return "\n".join([
            "You are given a small relation graph.",
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is associated with {c_label}.",
            f"Question: Which label is {a} associated with?",
            option,
            "Answer:"
        ])

    # Weakened: E is mentioned as a formatting distractor, not as semantic evidence.
    if condition == "weak_distractor_v2":
        return "\n".join([
            "You are given a small relation graph.",
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is associated with {c_label}.",
            f"Ignore any option that is not supported by the facts.",
            f"Question: Which label is {a} associated with?",
            option,
            "Answer:"
        ])

    if condition == "ambiguous_branch":
        return "\n".join([
            "You are given a small relation graph.",
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is usually associated with {c_label}.",
            f"Fact 3: In a separate possible branch, {a} may be associated with {e_label}.",
            f"Question: Which label is {a} associated with in the main graph?",
            option,
            "Answer:"
        ])

    if condition == "strong_conflict":
        return "\n".join([
            "You are given a small relation graph.",
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is associated with {c_label}.",
            f"Conflicting fact: {a} is associated with {e_label}.",
            f"Question: Which label is {a} associated with?",
            option,
            "Answer:"
        ])

    if condition == "closure_negation":
        return "\n".join([
            "You are given a small relation graph.",
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is associated with {c_label}.",
            f"Correction: {a} is not associated with {c_label}; {a} is associated with {e_label}.",
            f"Question: Which label is {a} associated with?",
            option,
            "Answer:"
        ])

    if condition == "closure_rewrite":
        return "\n".join([
            "You are given a small relation graph.",
            f"Initial rule: if an item belongs to {b}, it is associated with {c_label}.",
            f"Updated rule: for this item, the association is rewritten to {e_label}.",
            f"Fact: {a} belongs to {b}.",
            f"Question: Which label is {a} associated with after the update?",
            option,
            "Answer:"
        ])

    if condition == "closure_override":
        return "\n".join([
            "You are given a small relation graph.",
            f"Default rule: members of {b} are associated with {c_label}.",
            f"Override rule: {a} is an exception and must be associated with {e_label}.",
            f"Fact: {a} belongs to {b}.",
            f"Question: Which label is {a} associated with under the override rule?",
            option,
            "Answer:"
        ])

    if condition == "closure_exception":
        return "\n".join([
            "You are given a small relation graph.",
            f"General rule: anything belonging to {b} is associated with {c_label}.",
            f"Exception: {a} belongs to {b}, but this specific item is associated with {e_label}.",
            f"Question: Which label is {a} associated with?",
            option,
            "Answer:"
        ])

    raise ValueError(condition)

def build_dataset():
    rows = []

    for gid in range(N_GRAPHS):
        a = make_entity("A", gid)
        b = make_entity("B", gid)

        c_label = LABEL_POOL[(2 * gid) % len(LABEL_POOL)]
        e_label = LABEL_POOL[(2 * gid + 1) % len(LABEL_POOL)]

        if c_label == e_label:
            e_label = LABEL_POOL[(2 * gid + 2) % len(LABEL_POOL)]

        for condition, eps, phase in ALL_CONDITIONS:
            mechanism = (
                "competition"
                if condition in [x[0] for x in COMPETITION_CONDITIONS]
                else "closure"
            )

            prompt = make_prompt(a, b, c_label, e_label, condition)

            rows.append({
                "graph_id": gid,
                "condition": condition,
                "epsilon": eps,
                "phase_target": phase,
                "mechanism": mechanism if condition != "clean" else "shared_clean",
                "A": a,
                "B": b,
                "C_label": c_label,
                "E_label": e_label,
                "prompt": prompt,
            })

    return pd.DataFrame(rows)

df = build_dataset()
df.to_csv(SAVE_DIR / "dataset.csv", index=False)
print("Dataset:", df.shape)

# ============================================================
# READOUT
# ============================================================

@torch.no_grad()
def batch_layer_logits(prompts, c_labels, e_labels):
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
    ).to(model.device)

    outputs = model(
        **enc,
        output_hidden_states=True,
        use_cache=False,
    )

    hidden_states = outputs.hidden_states
    attention_mask = enc["attention_mask"]
    last_indices = attention_mask.sum(dim=1) - 1

    rows = []

    for bi in range(len(prompts)):
        c_id = continuation_ids(c_labels[bi])[0]
        e_id = continuation_ids(e_labels[bi])[0]

        for layer in TRACK_LAYERS:
            hs = hidden_states[layer + 1][bi, last_indices[bi], :]
            logits = model.lm_head(hs).float()

            c_logit = logits[c_id].item()
            e_logit = logits[e_id].item()

            rows.append({
                "batch_index": bi,
                "layer": layer,
                "C_id": c_id,
                "E_id": e_id,
                "C_logit": c_logit,
                "E_logit": e_logit,
                "R": c_logit - e_logit,
            })

    return rows

@torch.no_grad()
def batch_generate(prompts):
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
    ).to(model.device)

    out = model.generate(
        **enc,
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    gen_texts = []
    for i in range(out.shape[0]):
        new_ids = out[i, enc["input_ids"].shape[1]:]
        gen_texts.append(tokenizer.decode(new_ids, skip_special_tokens=True).strip())

    return gen_texts

# ============================================================
# RUN
# ============================================================

layer_rows = []
gen_rows = []

print("\nRunning layer readout and generation...")

for start in range(0, len(df), BATCH_SIZE):
    batch = df.iloc[start:start + BATCH_SIZE].reset_index(drop=True)

    prompts = batch["prompt"].tolist()
    c_labels = batch["C_label"].tolist()
    e_labels = batch["E_label"].tolist()

    read_rows = batch_layer_logits(prompts, c_labels, e_labels)

    for r in read_rows:
        source = batch.iloc[r["batch_index"]]
        r.update({
            "row_id": int(start + r["batch_index"]),
            "graph_id": int(source["graph_id"]),
            "condition": source["condition"],
            "epsilon": float(source["epsilon"]),
            "phase_target": source["phase_target"],
            "mechanism": source["mechanism"],
            "C_label": source["C_label"],
            "E_label": source["E_label"],
        })
        layer_rows.append(r)

    gens = batch_generate(prompts)

    for i, g in enumerate(gens):
        source = batch.iloc[i]
        c = source["C_label"]
        e = source["E_label"]

        gen_c = int(g.lower().startswith(c.lower()))
        gen_e = int(g.lower().startswith(e.lower()))

        gen_rows.append({
            "row_id": int(start + i),
            "graph_id": int(source["graph_id"]),
            "condition": source["condition"],
            "epsilon": float(source["epsilon"]),
            "phase_target": source["phase_target"],
            "mechanism": source["mechanism"],
            "C_label": c,
            "E_label": e,
            "generation": g,
            "Gen_C": gen_c,
            "Gen_E": gen_e,
        })

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

layer_df = pd.DataFrame(layer_rows)
gen_df = pd.DataFrame(gen_rows)

layer_df.to_csv(SAVE_DIR / "layer_R.csv", index=False)
gen_df.to_csv(SAVE_DIR / "generation.csv", index=False)

print("Layer R:", layer_df.shape)
print("Generation:", gen_df.shape)

# ============================================================
# DELTA R
# ============================================================

wide = layer_df.pivot_table(
    index=[
        "row_id", "graph_id", "condition", "epsilon",
        "phase_target", "mechanism", "C_label", "E_label"
    ],
    columns="layer",
    values="R",
).reset_index()

wide.columns = [str(c) if isinstance(c, int) else c for c in wide.columns]

clean = wide[wide["condition"] == "clean"].copy()
clean_cols = ["graph_id"] + [str(l) for l in TRACK_LAYERS]
clean = clean[clean_cols].rename(columns={str(l): f"clean_R_{l}" for l in TRACK_LAYERS})

delta = wide.merge(clean, on="graph_id", how="left")

for l in TRACK_LAYERS:
    delta[f"R_{l}"] = delta[str(l)]
    delta[f"dR_{l}"] = delta[str(l)] - delta[f"clean_R_{l}"]

delta = delta.merge(
    gen_df[["row_id", "generation", "Gen_C", "Gen_E"]],
    on="row_id",
    how="left"
)

delta_cols = [
    "row_id", "graph_id", "condition", "epsilon", "phase_target",
    "mechanism", "C_label", "E_label", "generation", "Gen_C", "Gen_E"
] + [f"R_{l}" for l in TRACK_LAYERS] + [f"dR_{l}" for l in TRACK_LAYERS]

delta = delta[delta_cols]
delta.to_csv(SAVE_DIR / "deltaR.csv", index=False)

# ============================================================
# MECHANISM-SPLIT PCA
# ============================================================

def fit_mechanism_deltaU(delta_df, mechanism_name, condition_names, output_name):
    sub = delta_df[delta_df["condition"].isin(condition_names)].copy()

    X = sub[[f"dR_{l}" for l in DELTA_LAYERS]].values.astype(np.float32)

    pca = PCA(n_components=3, random_state=SEED)
    Z = pca.fit_transform(X)

    sub["DeltaU_raw"] = Z[:, 0]
    sub["DeltaU_PC2"] = Z[:, 1]
    sub["DeltaU_PC3"] = Z[:, 2]

    phase_mean_raw = sub.groupby("phase_target")["DeltaU_raw"].mean().to_dict()

    # Orient so positive > negative when both exist.
    if phase_mean_raw.get("positive", 0) < phase_mean_raw.get("negative", 0):
        sub["DeltaU"] = -sub["DeltaU_raw"]
        pc1_loading = (-pca.components_[0]).tolist()
    else:
        sub["DeltaU"] = sub["DeltaU_raw"]
        pc1_loading = pca.components_[0].tolist()

    phase_mean = sub.groupby("phase_target")["DeltaU"].mean().to_dict()
    condition_mean = sub.groupby("condition")["DeltaU"].mean().to_dict()
    gen_rate = sub.groupby("condition")["Gen_E"].mean().to_dict()

    # Critical band only meaningful if critical samples exist.
    crit = sub[sub["phase_target"] == "critical"]["DeltaU"].values
    if len(crit) >= 4:
        U_minus = float(np.percentile(crit, 10))
        U_center = float(np.percentile(crit, 50))
        U_plus = float(np.percentile(crit, 90))
    else:
        U_minus = None
        U_center = None
        U_plus = None

    if U_minus is not None:
        sub["in_critical_band"] = (
            (sub["DeltaU"] >= U_minus) & (sub["DeltaU"] <= U_plus)
        ).astype(int)
    else:
        sub["in_critical_band"] = 0

    # ΔU -> Gen_E predictive check
    y = sub["Gen_E"].values.astype(int)
    u = sub[["DeltaU"]].values.astype(np.float32)

    metrics = {}
    if len(np.unique(y)) > 1:
        # Direction-free: use LR CV.
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        preds = np.zeros(len(sub))

        for train_idx, test_idx in skf.split(u, y):
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000)),
            ])
            clf.fit(u[train_idx], y[train_idx])
            preds[test_idx] = clf.predict_proba(u[test_idx])[:, 1]

        pred_label = (preds >= 0.5).astype(int)

        metrics = {
            "cv_auc": float(roc_auc_score(y, preds)),
            "cv_acc": float(accuracy_score(y, pred_label)),
            "cv_f1": float(f1_score(y, pred_label, zero_division=0)),
        }

        sub["Pred_GenE_prob_from_DeltaU"] = preds
    else:
        metrics = {
            "cv_auc": None,
            "cv_acc": None,
            "cv_f1": None,
        }
        sub["Pred_GenE_prob_from_DeltaU"] = np.nan

    sub.to_csv(SAVE_DIR / output_name, index=False)

    return {
        "mechanism": mechanism_name,
        "conditions": condition_names,
        "n_rows": int(len(sub)),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pc1_loading_oriented": pc1_loading,
        "phase_mean_DeltaU": phase_mean,
        "condition_mean_DeltaU": condition_mean,
        "condition_GenE_rate": gen_rate,
        "critical_band": {
            "U_minus": U_minus,
            "U_center": U_center,
            "U_plus": U_plus,
            "band_width": None if U_minus is None else U_plus - U_minus,
        },
        "metrics": metrics,
    }

competition_condition_names = [x[0] for x in COMPETITION_CONDITIONS]
closure_condition_names = [x[0] for x in CLOSURE_CONDITIONS]

summary_comp = fit_mechanism_deltaU(
    delta,
    "competition",
    competition_condition_names,
    "deltaU_competition.csv",
)

summary_closure = fit_mechanism_deltaU(
    delta,
    "closure",
    closure_condition_names,
    "deltaU_closure.csv",
)

# ============================================================
# CROSS-MECHANISM SANITY CHECK
# ============================================================

summary = {
    "experiment": "HA-1A-debug Mechanism-Split Hallucination Basin Audit",
    "model_path": MODEL_PATH,
    "n_graphs": N_GRAPHS,
    "track_layers": TRACK_LAYERS,
    "delta_layers": DELTA_LAYERS,
    "competition": summary_comp,
    "closure": summary_closure,
    "global_condition_GenE_rate": delta.groupby("condition")["Gen_E"].mean().to_dict(),
    "global_condition_mean_dR_l2": {
        cond: float(
            np.linalg.norm(
                delta[delta["condition"] == cond][[f"dR_{l}" for l in DELTA_LAYERS]]
                .mean()
                .values
            )
        )
        for cond in sorted(delta["condition"].unique())
    },
}

with open(SAVE_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\nDone.")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nSaved to: {SAVE_DIR.resolve()}")