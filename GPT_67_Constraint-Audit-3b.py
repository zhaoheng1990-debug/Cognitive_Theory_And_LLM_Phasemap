# ============================================================
# Constraint-Audit-3B
# Delta-C and Closure-Chain Invariant Test
#
# Goal:
#   Move from feature accumulation to invariant mining.
#
# Inputs:
#   Prefer:
#     ./constraint_audit3a_outputs/constraint_audit3a_augmented_feature_table.csv
#   Fallback:
#     ./constraint_audit2c_outputs/constraint_audit2c_feature_table.csv
#     ./constraint_audit2b_outputs/constraint_audit2b_feature_table.csv
#
# New candidate structural invariants:
#   1. Delta-C:
#        C(x_family) - C(x_clean_basic) within same graph idx
#
#   2. Closure-chain probes:
#        B_current_location: B -> C/E
#        B_old_location:     old/default B -> C/E
#        A_final_location:   A -> C/E
#
#   3. Chain consistency:
#        A_final_margin - B_current_margin
#        A_final_margin + B_current_margin
#        sign agreement between A and B margins
#
#   4. Evidence-binding:
#        B_current_margin - B_old_margin
#
# Evaluation:
#   - GroupKFold by graph idx
#   - Leave-one-family-out
#   - binary Gen_E
#   - submechanism
#   - risk_regime
#   - invariant feature ranking
#
# Outputs:
#   ./constraint_audit3b_outputs/
# ============================================================

import re
import gc
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

INPUT_3A = Path("./constraint_audit3a_outputs/constraint_audit3a_augmented_feature_table.csv")
INPUT_2C = Path("./constraint_audit2c_outputs/constraint_audit2c_feature_table.csv")
INPUT_2B = Path("./constraint_audit2b_outputs/constraint_audit2b_feature_table.csv")

SAVE_DIR = Path("./constraint_audit3b_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

MAX_LEN = 360
BATCH_SIZE = 8

TRACK_LAYERS = [20, 21, 22, 23, 24, 25]

N_GROUP_SPLITS = 5

PROBE_TYPES = [
    "B_current_location",
    "B_old_location",
    "A_final_location",
]

# ============================================================
# LABEL MAPS
# ============================================================

FAMILY_ORDER = [
    "stable_clean_basic",
    "stable_paraphrase",
    "stable_redundant",
    "stable_irrelevant",
    "stable_weak_note",

    "competition_ambiguous",
    "competition_branch",
    "competition_direct",
    "competition_source_claim",
    "competition_equal_evidence",

    "closure_negation",
    "closure_update",
    "closure_override",
    "closure_temporal",
    "closure_authority",
    "closure_exception",
]

COARSE_MECHANISM_MAP = {
    "stable_clean_basic": "stable_or_preserved",
    "stable_paraphrase": "stable_or_preserved",
    "stable_redundant": "stable_or_preserved",
    "stable_irrelevant": "stable_or_preserved",
    "stable_weak_note": "stable_or_preserved",

    "competition_ambiguous": "competition_conflict",
    "competition_branch": "competition_conflict",
    "competition_direct": "competition_conflict",
    "competition_source_claim": "competition_conflict",
    "competition_equal_evidence": "competition_conflict",

    "closure_negation": "closure_rewrite",
    "closure_update": "closure_rewrite",
    "closure_override": "closure_rewrite",
    "closure_temporal": "closure_rewrite",
    "closure_authority": "closure_rewrite",
    "closure_exception": "closure_rewrite",
}

SUBMECHANISM_MAP = {
    "stable_clean_basic": "stable_core",
    "stable_redundant": "stable_core",
    "stable_weak_note": "stable_core",
    "stable_paraphrase": "stable_surface_shift",
    "stable_irrelevant": "stable_surface_shift",

    "competition_branch": "competition_low_commit",
    "competition_direct": "competition_low_commit",
    "competition_ambiguous": "competition_ambiguous",
    "competition_source_claim": "competition_high_commit",
    "competition_equal_evidence": "competition_high_commit",

    "closure_negation": "closure_canonical_rewrite",
    "closure_update": "closure_canonical_rewrite",
    "closure_temporal": "closure_canonical_rewrite",
    "closure_authority": "closure_canonical_rewrite",
    "closure_override": "closure_rule_override",
    "closure_exception": "closure_rule_override",
}

RISK_REGIME_MAP = {
    "stable_clean_basic": "clean_basin",
    "stable_redundant": "clean_basin",
    "stable_weak_note": "clean_basin",

    "stable_paraphrase": "low_risk_shift",
    "stable_irrelevant": "low_risk_shift",

    "competition_branch": "competition_low_risk",
    "competition_direct": "competition_low_risk",

    "competition_ambiguous": "competition_mid_risk",

    "competition_source_claim": "competition_high_risk",
    "competition_equal_evidence": "competition_high_risk",

    "closure_negation": "closure_collapse",
    "closure_update": "closure_collapse",
    "closure_override": "closure_collapse",
    "closure_temporal": "closure_collapse",
    "closure_authority": "closure_collapse",
    "closure_exception": "closure_collapse",
}

SUBMECHANISM_CLASSES = [
    "stable_core",
    "stable_surface_shift",
    "competition_low_commit",
    "competition_ambiguous",
    "competition_high_commit",
    "closure_canonical_rewrite",
    "closure_rule_override",
]

RISK_REGIME_CLASSES = [
    "clean_basin",
    "low_risk_shift",
    "competition_low_risk",
    "competition_mid_risk",
    "competition_high_risk",
    "closure_collapse",
]

COARSE_CLASSES = [
    "stable_or_preserved",
    "competition_conflict",
    "closure_rewrite",
]

# ============================================================
# BASIC HELPERS
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)


def parse_bool_series(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def sanitize_prob(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(p, 1e-6, 1.0 - 1e-6)


def safe_auc(y, p):
    y = np.asarray(y).astype(int)
    p = sanitize_prob(p)

    if len(np.unique(y)) < 2:
        return np.nan

    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return np.nan


def safe_log_loss(y, p):
    y = np.asarray(y).astype(int)
    p = sanitize_prob(p)

    try:
        return float(log_loss(y, p, labels=[0, 1]))
    except Exception:
        return np.nan


def calc_binary_metrics(y, p, threshold=0.5):
    y = np.asarray(y).astype(int)
    p = sanitize_prob(p)

    pred = (p >= threshold).astype(int)

    out = {
        "auc": safe_auc(y, p),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "nll": safe_log_loss(y, p),
        "pred_pos_rate": float(pred.mean()),
        "mean_prob": float(p.mean()),
        "threshold": float(threshold),
    }

    try:
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        out["tn"] = int(tn)
        out["fp"] = int(fp)
        out["fn"] = int(fn)
        out["tp"] = int(tp)
    except Exception:
        out["tn"] = 0
        out["fp"] = 0
        out["fn"] = 0
        out["tp"] = 0

    return out


def binary_entropy_from_margin(m):
    p = 1.0 / (1.0 + np.exp(-np.asarray(m, dtype=np.float64)))
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    q = 1.0 - p
    h = -(p * np.log(p) + q * np.log(q)) / np.log(2.0)
    return h.astype(np.float32)


def entity_a(idx):
    return f"A{int(idx):03d}"


def entity_b(idx):
    return f"B{int(idx):03d}"


def strip_final_answer_line(prompt):
    lines = prompt.rstrip().splitlines()
    while lines and lines[-1].strip().lower() in ["answer:", "diagnostic answer:"]:
        lines = lines[:-1]
    return "\n".join(lines).rstrip()


# ============================================================
# SKLEARN MODELS
# ============================================================

class ConstantBinaryModel:
    def __init__(self, p):
        self.p = float(np.clip(p, 1e-6, 1.0 - 1e-6))

    def predict_proba(self, X):
        n = len(X)
        p1 = np.full(n, self.p, dtype=np.float64)
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)


class ConstantMulticlassModel:
    def __init__(self, classes):
        self.classes_ = np.asarray(classes)

    def predict(self, X):
        return np.full(len(X), self.classes_[0], dtype=object)

    def predict_proba(self, X):
        out = np.zeros((len(X), len(self.classes_)), dtype=np.float64)
        out[:, 0] = 1.0
        return out


def make_binary_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            max_iter=4000,
        )),
    ])


def make_multiclass_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            max_iter=6000,
        )),
    ])


def fit_binary_or_constant(X, y):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return ConstantBinaryModel(float(y.mean()))
    model = make_binary_model()
    model.fit(X, y)
    return model


def fit_multiclass_or_constant(X, y):
    y = np.asarray(y)
    vals = np.unique(y)
    if len(vals) < 2:
        return ConstantMulticlassModel(vals)
    model = make_multiclass_model()
    model.fit(X, y)
    return model


def aligned_proba(model, X, classes):
    p_raw = model.predict_proba(X)

    if hasattr(model, "classes_"):
        model_classes = list(model.classes_)
    else:
        model_classes = list(model.named_steps["clf"].classes_)

    out = np.zeros((len(X), len(classes)), dtype=np.float64)

    for j, cls in enumerate(model_classes):
        if cls in classes:
            out[:, classes.index(cls)] = p_raw[:, j]

    row_sum = out.sum(axis=1, keepdims=True)
    bad = row_sum[:, 0] <= 1e-12

    if bad.any():
        out[bad, :] = 1.0 / len(classes)
        row_sum = out.sum(axis=1, keepdims=True)

    return out / (row_sum + 1e-12)


# ============================================================
# LOAD DATA
# ============================================================

if INPUT_3A.exists():
    INPUT_FILE = INPUT_3A
elif INPUT_2C.exists():
    INPUT_FILE = INPUT_2C
elif INPUT_2B.exists():
    INPUT_FILE = INPUT_2B
else:
    raise FileNotFoundError(
        "Cannot find 3A / 2C / 2B feature table."
    )

df = pd.read_csv(INPUT_FILE)

for col in ["is_clean", "is_conflict", "is_other"]:
    if col in df.columns:
        df[col] = parse_bool_series(df[col])

if "y_conflict" not in df.columns:
    if "is_conflict" not in df.columns:
        raise RuntimeError("Need y_conflict or is_conflict column.")
    df["y_conflict"] = df["is_conflict"].astype(int)

df["y_conflict"] = df["y_conflict"].astype(int)

if "is_clean" in df.columns and "is_conflict" in df.columns:
    df = df[(df["is_clean"] == True) | (df["is_conflict"] == True)].copy()

required_cols = [
    "prompt",
    "family",
    "idx",
    "clean_label",
    "conflict_label",
    "clean_token_id",
    "conflict_token_id",
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise RuntimeError(
        "Input table lacks required columns: "
        + ", ".join(missing)
    )

df["mechanism"] = df["family"].map(COARSE_MECHANISM_MAP)
df["submechanism"] = df["family"].map(SUBMECHANISM_MAP)
df["risk_regime"] = df["family"].map(RISK_REGIME_MAP)

if df["submechanism"].isna().any():
    bad = df[df["submechanism"].isna()]["family"].unique().tolist()
    raise RuntimeError(f"Unknown family values: {bad}")

df = df.reset_index(drop=True)

print("\n============================================================")
print("Constraint-Audit-3B Input Summary")
print("============================================================\n")
print("Input:", INPUT_FILE)
print("Rows:", len(df))
print("Families:", df["family"].nunique())
print("Device:", DEVICE)

# ============================================================
# DELTA-C FEATURES
# ============================================================

def add_delta_c_features(df_in):
    out = df_in.copy()

    base_family = "stable_clean_basic"

    base_df = out[out["family"] == base_family].copy()
    if len(base_df) == 0:
        raise RuntimeError("No stable_clean_basic baseline rows found.")

    base_df = base_df.set_index("idx")

    raw_base_cols = []
    for l in TRACK_LAYERS:
        for name in ["R", "H", "pC"]:
            c = f"{name}_L{l}"
            if c in out.columns:
                raw_base_cols.append(c)

    if len(raw_base_cols) == 0:
        raise RuntimeError("No R/H/pC layer columns found for Delta-C.")

    for c in raw_base_cols:
        base_map = base_df[c].to_dict()
        aligned_base = out["idx"].map(base_map).astype(float)
        out[f"DELTA_{c}"] = out[c].astype(float) - aligned_base

    # Aggregate delta trajectories.
    for name in ["R", "H", "pC"]:
        cols = [f"DELTA_{name}_L{l}" for l in TRACK_LAYERS if f"DELTA_{name}_L{l}" in out.columns]
        if len(cols) == 0:
            continue

        X = out[cols].values.astype(np.float32)

        out[f"DELTA_{name}_mean"] = X.mean(axis=1)
        out[f"DELTA_{name}_min"] = X.min(axis=1)
        out[f"DELTA_{name}_max"] = X.max(axis=1)
        out[f"DELTA_{name}_last"] = X[:, -1]
        out[f"DELTA_{name}_slope"] = X[:, -1] - X[:, 0]
        out[f"DELTA_{name}_range"] = X.max(axis=1) - X.min(axis=1)
        out[f"DELTA_{name}_area"] = X.sum(axis=1)

    # Trajectory distance/cosine to clean.
    r_cols = [f"R_L{l}" for l in TRACK_LAYERS if f"R_L{l}" in out.columns]
    if len(r_cols) >= 2:
        R = out[r_cols].values.astype(np.float32)

        clean_R = []
        for _, row in out.iterrows():
            idx = row["idx"]
            clean_R.append(base_df.loc[idx, r_cols].values.astype(np.float32))
        clean_R = np.vstack(clean_R)

        diff = R - clean_R
        out["DELTA_R_traj_l2"] = np.linalg.norm(diff, axis=1)

        dot = (R * clean_R).sum(axis=1)
        denom = np.linalg.norm(R, axis=1) * np.linalg.norm(clean_R, axis=1) + 1e-8
        out["DELTA_R_traj_cos_to_clean"] = dot / denom

        R_centered = R - R.mean(axis=1, keepdims=True)
        C_centered = clean_R - clean_R.mean(axis=1, keepdims=True)
        dot2 = (R_centered * C_centered).sum(axis=1)
        denom2 = np.linalg.norm(R_centered, axis=1) * np.linalg.norm(C_centered, axis=1) + 1e-8
        out["DELTA_R_shape_cos_to_clean"] = dot2 / denom2

    return out


df = add_delta_c_features(df)

# ============================================================
# PROBE PROMPTS
# ============================================================

def make_probe_prompt(row, probe_type):
    base = strip_final_answer_line(str(row["prompt"]))

    idx = int(row["idx"])
    a = entity_a(idx)
    b = entity_b(idx)

    clean = str(row["clean_label"])
    conflict = str(row["conflict_label"])

    option_line = f"Answer with exactly one word: {clean} or {conflict}."

    if probe_type == "B_current_location":
        q = (
            f"Diagnostic question: After applying any update, override, exception, "
            f"temporal, or verified registry statement above, which location label "
            f"is {b} currently assigned to?"
        )

    elif probe_type == "B_old_location":
        q = (
            f"Diagnostic question: According to the old, default, earlier, or general "
            f"record above, which location label was {b} assigned to before any update "
            f"or exception? If no such old/default record appears, use the original "
            f"non-conflicting record."
        )

    elif probe_type == "A_final_location":
        q = (
            f"Diagnostic question: After applying the valid records and rules above, "
            f"which location label is {a} located at?"
        )

    else:
        raise ValueError(f"Unknown probe type: {probe_type}")

    return "\n".join([
        base,
        "",
        q,
        option_line,
        "Diagnostic answer:",
    ])


# ============================================================
# LOAD MODEL FOR PROBES
# ============================================================

print("\nLoading model for closure-chain probes...")

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
if max(TRACK_LAYERS) >= num_layers:
    raise RuntimeError(
        f"TRACK_LAYERS includes {max(TRACK_LAYERS)}, but model has {num_layers} layers."
    )

print("Num layers:", num_layers)
print("LM head:", tuple(model.lm_head.weight.shape))
print("LM head dtype:", model.lm_head.weight.dtype)
print("LM head device:", model.lm_head.weight.device)


def lm_head_forward(hidden):
    w = model.lm_head.weight
    hidden = hidden.to(device=w.device, dtype=w.dtype)
    return model.lm_head(hidden).float()


# ============================================================
# PROBE FEATURE EXTRACTION
# ============================================================

def validate_probe_cache(path, n_expected):
    if not path.exists():
        return False

    try:
        tmp = pd.read_csv(path, nrows=5)
        full_rows = sum(1 for _ in open(path, "r", encoding="utf-8")) - 1
    except Exception:
        return False

    if full_rows != n_expected:
        print(f"Probe cache row mismatch: got {full_rows}, expected {n_expected}. Regenerating.")
        return False

    required = [
        "P3B_B_current_location_margin_L20",
        "P3B_B_old_location_margin_L20",
        "P3B_A_final_location_margin_L20",
        "CHAIN_A_minus_B_current_L20",
        "BIND_current_minus_old_B_L20",
    ]

    for c in required:
        if c not in tmp.columns:
            print(f"Probe cache missing {c}. Regenerating.")
            return False

    return True


def extract_probe_features(df_in):
    n = len(df_in)

    probe_prompts = {}
    for probe in PROBE_TYPES:
        probe_prompts[probe] = [
            make_probe_prompt(row, probe)
            for _, row in df_in.iterrows()
        ]

    out = pd.DataFrame(index=np.arange(n))

    clean_ids = df_in["clean_token_id"].astype(int).values
    conflict_ids = df_in["conflict_token_id"].astype(int).values

    for probe in PROBE_TYPES:
        for l in TRACK_LAYERS:
            out[f"P3B_{probe}_margin_L{l}"] = 0.0
            out[f"P3B_{probe}_pC_L{l}"] = 0.0
            out[f"P3B_{probe}_H_L{l}"] = 0.0

    with torch.no_grad():
        for probe in PROBE_TYPES:
            prompts = probe_prompts[probe]
            print(f"\nExtracting probe: {probe}")

            for start in range(0, n, BATCH_SIZE):
                end = min(start + BATCH_SIZE, n)
                batch_prompts = prompts[start:end]

                inputs = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_LEN,
                ).to(DEVICE)

                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=False,
                )

                hidden_states = outputs.hidden_states
                row_indices = np.arange(start, end)

                clean_t = torch.tensor(
                    clean_ids[row_indices],
                    dtype=torch.long,
                    device=model.lm_head.weight.device,
                )

                conflict_t = torch.tensor(
                    conflict_ids[row_indices],
                    dtype=torch.long,
                    device=model.lm_head.weight.device,
                )

                for l in TRACK_LAYERS:
                    h = hidden_states[l + 1][:, -1, :]
                    logits = lm_head_forward(h)

                    c_logit = logits.gather(1, clean_t.view(-1, 1)).squeeze(1)
                    e_logit = logits.gather(1, conflict_t.view(-1, 1)).squeeze(1)

                    margin = (c_logit - e_logit).detach().cpu().numpy().astype(np.float32)
                    pC = 1.0 / (1.0 + np.exp(-margin))
                    H = binary_entropy_from_margin(margin)

                    out.loc[start:end - 1, f"P3B_{probe}_margin_L{l}"] = margin
                    out.loc[start:end - 1, f"P3B_{probe}_pC_L{l}"] = pC
                    out.loc[start:end - 1, f"P3B_{probe}_H_L{l}"] = H

                if start % (BATCH_SIZE * 10) == 0:
                    print(f"  {probe}: {end}/{n}")

                del outputs, hidden_states, inputs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Chain and binding features.
    for l in TRACK_LAYERS:
        A = out[f"P3B_A_final_location_margin_L{l}"].values.astype(np.float32)
        Bc = out[f"P3B_B_current_location_margin_L{l}"].values.astype(np.float32)
        Bo = out[f"P3B_B_old_location_margin_L{l}"].values.astype(np.float32)

        out[f"CHAIN_A_minus_B_current_L{l}"] = A - Bc
        out[f"CHAIN_A_plus_B_current_L{l}"] = A + Bc
        out[f"CHAIN_A_times_B_current_L{l}"] = A * Bc
        out[f"CHAIN_abs_A_minus_B_current_L{l}"] = np.abs(A - Bc)
        out[f"CHAIN_sign_agree_A_B_current_L{l}"] = (np.sign(A) == np.sign(Bc)).astype(np.float32)

        out[f"BIND_current_minus_old_B_L{l}"] = Bc - Bo
        out[f"BIND_abs_current_minus_old_B_L{l}"] = np.abs(Bc - Bo)

    # Aggregates.
    layer_prefixes = []
    for c in out.columns:
        if "_L" in c:
            suffix = c.rsplit("_L", 1)[-1]
            if suffix.isdigit():
                layer_prefixes.append(c.rsplit("_L", 1)[0])
    layer_prefixes = sorted(set(layer_prefixes))

    for prefix in layer_prefixes:
        cols = [f"{prefix}_L{l}" for l in TRACK_LAYERS if f"{prefix}_L{l}" in out.columns]
        if len(cols) == 0:
            continue

        X = out[cols].values.astype(np.float32)

        out[f"{prefix}_mean"] = X.mean(axis=1)
        out[f"{prefix}_min"] = X.min(axis=1)
        out[f"{prefix}_max"] = X.max(axis=1)
        out[f"{prefix}_last"] = X[:, -1]
        out[f"{prefix}_slope"] = X[:, -1] - X[:, 0]
        out[f"{prefix}_range"] = X.max(axis=1) - X.min(axis=1)
        out[f"{prefix}_area"] = X.sum(axis=1)

    return out


probe_cache = SAVE_DIR / "constraint_audit3b_probe_features.csv"

if validate_probe_cache(probe_cache, len(df)):
    print("\nLoading cached probe features:", probe_cache)
    probe_df = pd.read_csv(probe_cache)
else:
    print("\nExtracting closure-chain probe features...")
    probe_df = extract_probe_features(df)
    probe_df.to_csv(probe_cache, index=False)
    print("Saved probe features:", probe_cache)

df3b = pd.concat(
    [df.reset_index(drop=True), probe_df.reset_index(drop=True)],
    axis=1,
)

# ============================================================
# FEATURE SETS
# ============================================================

def is_numeric_col(c):
    return c in df3b.columns and pd.api.types.is_numeric_dtype(df3b[c])


def clean_cols(cols):
    seen = set()
    out = []

    for c in cols:
        if c in seen:
            continue
        seen.add(c)

        if is_numeric_col(c):
            out.append(c)

    return out


baseline_cols = []
for l in TRACK_LAYERS:
    for name in ["R", "H", "pC"]:
        c = f"{name}_L{l}"
        if c in df3b.columns:
            baseline_cols.append(c)

baseline_cols += [
    c for c in df3b.columns
    if c.startswith("ENTRY20_22_BASE")
    or c.startswith("BASIN23_25_BASE")
    or c.startswith("ALL20_25_BASE")
]

delta_cols = [c for c in df3b.columns if c.startswith("DELTA_")]

probe_margin_cols = [
    c for c in df3b.columns
    if c.startswith("P3B_") and "_margin" in c
]

probe_all_cols = [
    c for c in df3b.columns
    if c.startswith("P3B_")
]

chain_cols = [
    c for c in df3b.columns
    if c.startswith("CHAIN_")
]

binding_cols = [
    c for c in df3b.columns
    if c.startswith("BIND_")
]

structural_probe_cols = clean_cols(probe_all_cols + chain_cols + binding_cols)

feature_sets = {
    "baseline_RHpC": clean_cols(baseline_cols),

    "delta_C_only": clean_cols(delta_cols),

    "probe_margin_only": clean_cols(probe_margin_cols),

    "chain_binding_only": clean_cols(chain_cols + binding_cols),

    "delta_plus_chain": clean_cols(delta_cols + structural_probe_cols),

    "minimal_structural_C": clean_cols(baseline_cols + delta_cols + structural_probe_cols),
}

feature_sets = {k: v for k, v in feature_sets.items() if len(v) > 0}

all_used_cols = sorted(set(sum(feature_sets.values(), [])))

for c in all_used_cols:
    df3b[c] = pd.to_numeric(df3b[c], errors="coerce")

df3b[all_used_cols] = (
    df3b[all_used_cols]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)

print("\nFeature sets:")
for name, cols in feature_sets.items():
    print(f"  {name:<28} n={len(cols)}")

# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def eval_binary_groupcv(df_in, cols, feature_set):
    X = df_in[cols].values.astype(np.float32)
    y = df_in["y_conflict"].values.astype(int)
    groups = df_in["idx"].values

    gkf = GroupKFold(n_splits=N_GROUP_SPLITS)
    p = np.zeros(len(df_in), dtype=np.float64)

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        model = fit_binary_or_constant(X[train_idx], y[train_idx])
        p[test_idx] = sanitize_prob(model.predict_proba(X[test_idx])[:, 1])

    row = {"feature_set": feature_set}
    row.update(calc_binary_metrics(y, p, threshold=0.5))
    return row


def eval_multiclass_groupcv(df_in, cols, target_col, classes, feature_set):
    X = df_in[cols].values.astype(np.float32)
    y = df_in[target_col].values
    groups = df_in["idx"].values

    gkf = GroupKFold(n_splits=N_GROUP_SPLITS)
    pred = np.empty(len(df_in), dtype=object)

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        model = fit_multiclass_or_constant(X[train_idx], y[train_idx])
        pred[test_idx] = model.predict(X[test_idx])

    return {
        "feature_set": feature_set,
        "target": target_col,
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def eval_binary_lofo(df_in, cols, feature_set):
    X = df_in[cols].values.astype(np.float32)
    y = df_in["y_conflict"].values.astype(int)
    families = df_in["family"].values

    logo = LeaveOneGroupOut()

    rows = []

    for train_idx, test_idx in logo.split(X, y, groups=families):
        heldout = families[test_idx][0]

        model = fit_binary_or_constant(X[train_idx], y[train_idx])
        p_test = sanitize_prob(model.predict_proba(X[test_idx])[:, 1])

        m = calc_binary_metrics(y[test_idx], p_test, threshold=0.5)

        row = {
            "feature_set": feature_set,
            "heldout_family": heldout,
            "heldout_submechanism": SUBMECHANISM_MAP[heldout],
            "heldout_risk_regime": RISK_REGIME_MAP[heldout],
            "positive_rate": float(y[test_idx].mean()),
        }
        row.update(m)
        rows.append(row)

    return pd.DataFrame(rows)


def eval_multiclass_lofo(df_in, cols, target_col, classes, feature_set):
    X = df_in[cols].values.astype(np.float32)
    y = df_in[target_col].values
    families = df_in["family"].values

    logo = LeaveOneGroupOut()

    rows = []

    for train_idx, test_idx in logo.split(X, y, groups=families):
        heldout = families[test_idx][0]
        true_label = y[test_idx][0]
        train_has_true = true_label in set(y[train_idx])

        model = fit_multiclass_or_constant(X[train_idx], y[train_idx])
        p = aligned_proba(model, X[test_idx], classes)
        pred = np.array(classes)[np.argmax(p, axis=1)]

        row = {
            "feature_set": feature_set,
            "target": target_col,
            "heldout_family": heldout,
            "true_label": true_label,
            "train_has_true_label": bool(train_has_true),
            "accuracy": float(accuracy_score(y[test_idx], pred)),
            "macro_f1": float(f1_score(y[test_idx], pred, average="macro", zero_division=0)),
        }

        for j, cls in enumerate(classes):
            row[f"mean_p_{cls}"] = float(p[:, j].mean())

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# INVARIANT FEATURE RANKING
# ============================================================

def invariant_f_score(df_in, feature_cols, label_col):
    rows = []

    y = df_in[label_col].values
    labels = sorted(pd.unique(y).tolist())

    for feat in feature_cols:
        x = df_in[feat].values.astype(np.float64)
        overall = np.nanmean(x)

        between = 0.0
        within = 0.0
        n_total = 0

        for lab in labels:
            mask = y == lab
            vals = x[mask]
            vals = vals[np.isfinite(vals)]

            if len(vals) == 0:
                continue

            n = len(vals)
            mu = vals.mean()

            between += n * (mu - overall) ** 2
            within += ((vals - mu) ** 2).sum()
            n_total += n

        k = len(labels)
        if k <= 1 or n_total <= k:
            continue

        between = between / max(1, k - 1)
        within = within / max(1, n_total - k)

        f = between / (within + 1e-8)

        rows.append({
            "label_col": label_col,
            "feature": feat,
            "f_score": float(f),
            "between_var": float(between),
            "within_var": float(within),
        })

    return pd.DataFrame(rows).sort_values(by="f_score", ascending=False)


# ============================================================
# RUN EVALUATIONS
# ============================================================

binary_group_rows = []
multi_group_rows = []
binary_lofo_all = []
multi_lofo_all = []

for fs_name, cols in feature_sets.items():
    print(f"\nEvaluating feature set: {fs_name} | n={len(cols)}")

    binary_group_rows.append(
        eval_binary_groupcv(df3b, cols, fs_name)
    )

    multi_group_rows.append(
        eval_multiclass_groupcv(
            df3b,
            cols,
            target_col="submechanism",
            classes=SUBMECHANISM_CLASSES,
            feature_set=fs_name,
        )
    )

    multi_group_rows.append(
        eval_multiclass_groupcv(
            df3b,
            cols,
            target_col="risk_regime",
            classes=RISK_REGIME_CLASSES,
            feature_set=fs_name,
        )
    )

    binary_lofo_all.append(
        eval_binary_lofo(df3b, cols, fs_name)
    )

    multi_lofo_all.append(
        eval_multiclass_lofo(
            df3b,
            cols,
            target_col="submechanism",
            classes=SUBMECHANISM_CLASSES,
            feature_set=fs_name,
        )
    )

    multi_lofo_all.append(
        eval_multiclass_lofo(
            df3b,
            cols,
            target_col="risk_regime",
            classes=RISK_REGIME_CLASSES,
            feature_set=fs_name,
        )
    )

binary_group_df = pd.DataFrame(binary_group_rows)
multi_group_df = pd.DataFrame(multi_group_rows)

binary_lofo_df = pd.concat(binary_lofo_all, axis=0, ignore_index=True)
multi_lofo_df = pd.concat(multi_lofo_all, axis=0, ignore_index=True)

binary_lofo_summary_rows = []
for fs, sub in binary_lofo_df.groupby("feature_set"):
    binary_lofo_summary_rows.append({
        "feature_set": fs,
        "mean_auc": float(np.nanmean(sub["auc"])),
        "mean_accuracy": float(np.nanmean(sub["accuracy"])),
        "mean_f1": float(np.nanmean(sub["f1"])),
        "mean_brier": float(np.nanmean(sub["brier"])),
        "mean_pred_pos_rate": float(np.nanmean(sub["pred_pos_rate"])),
    })

binary_lofo_summary = pd.DataFrame(binary_lofo_summary_rows)

multi_lofo_summary_rows = []
for keys, sub in multi_lofo_df.groupby(["feature_set", "target"]):
    fs, target = keys
    multi_lofo_summary_rows.append({
        "feature_set": fs,
        "target": target,
        "mean_accuracy": float(np.nanmean(sub["accuracy"])),
        "mean_macro_f1": float(np.nanmean(sub["macro_f1"])),
    })

multi_lofo_summary = pd.DataFrame(multi_lofo_summary_rows)

# Invariant ranking only on structural candidates.
invariant_candidate_cols = clean_cols(delta_cols + structural_probe_cols)

inv_sub = invariant_f_score(df3b, invariant_candidate_cols, "submechanism")
inv_risk = invariant_f_score(df3b, invariant_candidate_cols, "risk_regime")
inv_mech = invariant_f_score(df3b, invariant_candidate_cols, "mechanism")

invariant_ranking = pd.concat([inv_mech, inv_sub, inv_risk], axis=0, ignore_index=True)

# Family top prediction summary.
top_class_rows = []

for _, r in multi_lofo_df.iterrows():
    prob_cols = [c for c in multi_lofo_df.columns if c.startswith("mean_p_")]
    vals = {}

    for c in prob_cols:
        val = r.get(c, np.nan)
        if pd.notna(val):
            vals[c.replace("mean_p_", "")] = val

    if len(vals) == 0:
        continue

    top_cls = max(vals, key=vals.get)

    top_class_rows.append({
        "feature_set": r["feature_set"],
        "target": r["target"],
        "heldout_family": r["heldout_family"],
        "true_label": r["true_label"],
        "top_pred_class": top_cls,
        "top_pred_prob": float(vals[top_cls]),
        "accuracy": float(r["accuracy"]),
        "macro_f1": float(r["macro_f1"]),
        "train_has_true_label": bool(r["train_has_true_label"]),
    })

top_class_df = pd.DataFrame(top_class_rows)

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n\n============================================================")
print("3B GroupKFold Binary")
print("============================================================\n")
print(
    binary_group_df[[
        "feature_set", "auc", "accuracy", "f1", "brier", "pred_pos_rate"
    ]]
    .sort_values(by="brier")
    .to_string(index=False)
)

print("\n\n============================================================")
print("3B GroupKFold Multiclass")
print("============================================================\n")
print(
    multi_group_df
    .sort_values(by=["target", "macro_f1"], ascending=[True, False])
    .to_string(index=False)
)

print("\n\n============================================================")
print("3B LOFO Binary Summary")
print("============================================================\n")
print(
    binary_lofo_summary
    .sort_values(by="mean_brier")
    .to_string(index=False)
)

print("\n\n============================================================")
print("3B LOFO Multiclass Summary")
print("============================================================\n")
print(
    multi_lofo_summary
    .sort_values(by=["target", "mean_macro_f1"], ascending=[True, False])
    .to_string(index=False)
)

print("\n\n============================================================")
print("3B Top Invariant Candidates: submechanism")
print("============================================================\n")
print(inv_sub.head(25).to_string(index=False))

print("\n\n============================================================")
print("3B Top Invariant Candidates: risk_regime")
print("============================================================\n")
print(inv_risk.head(25).to_string(index=False))

# ============================================================
# SAVE OUTPUTS
# ============================================================

feature_table_path = SAVE_DIR / "constraint_audit3b_feature_table.csv"

binary_group_path = SAVE_DIR / "constraint_audit3b_group_binary.csv"
multi_group_path = SAVE_DIR / "constraint_audit3b_group_multiclass.csv"

binary_lofo_path = SAVE_DIR / "constraint_audit3b_lofo_binary.csv"
binary_lofo_summary_path = SAVE_DIR / "constraint_audit3b_lofo_binary_summary.csv"

multi_lofo_path = SAVE_DIR / "constraint_audit3b_lofo_multiclass.csv"
multi_lofo_summary_path = SAVE_DIR / "constraint_audit3b_lofo_multiclass_summary.csv"

top_class_path = SAVE_DIR / "constraint_audit3b_lofo_top_class_summary.csv"

invariant_path = SAVE_DIR / "constraint_audit3b_invariant_feature_ranking.csv"

feature_sets_path = SAVE_DIR / "constraint_audit3b_feature_sets.csv"

df3b.to_csv(feature_table_path, index=False)

binary_group_df.to_csv(binary_group_path, index=False)
multi_group_df.to_csv(multi_group_path, index=False)

binary_lofo_df.to_csv(binary_lofo_path, index=False)
binary_lofo_summary.to_csv(binary_lofo_summary_path, index=False)

multi_lofo_df.to_csv(multi_lofo_path, index=False)
multi_lofo_summary.to_csv(multi_lofo_summary_path, index=False)

top_class_df.to_csv(top_class_path, index=False)

invariant_ranking.to_csv(invariant_path, index=False)

feature_set_rows = []
for name, cols in feature_sets.items():
    for c in cols:
        feature_set_rows.append({
            "feature_set": name,
            "feature": c,
        })

pd.DataFrame(feature_set_rows).to_csv(feature_sets_path, index=False)

print("\nSaved outputs:")
print(" ", feature_table_path)
print(" ", binary_group_path)
print(" ", multi_group_path)
print(" ", binary_lofo_path)
print(" ", binary_lofo_summary_path)
print(" ", multi_lofo_path)
print(" ", multi_lofo_summary_path)
print(" ", top_class_path)
print(" ", invariant_path)
print(" ", feature_sets_path)

# ============================================================
# INTERPRETATION GUIDE
# ============================================================

print("\n\n============================================================")
print("Constraint-Audit-3B Interpretation Guide")
print("============================================================\n")

print("Main success conditions:")
print("  1. delta_C_only improves LOFO submechanism or risk_regime over baseline.")
print("  2. probe_margin_only or chain_binding_only improves closure / override separation.")
print("  3. minimal_structural_C improves LOFO multiclass without relying on L26-L27 commit.")
print("  4. invariant ranking surfaces interpretable features such as:")
print("       DELTA_R_*")
print("       P3B_B_current_location_margin_*")
print("       P3B_A_final_location_margin_*")
print("       CHAIN_A_minus_B_current_*")
print("       BIND_current_minus_old_B_*")
print()
print("Interpretation:")
print("  If B_current and A_final margins move together:")
print("    closure-chain propagation is being represented.")
print()
print("  If B_current shifts but A_final does not:")
print("    model reads update at B but does not propagate to A.")
print()
print("  If A_final shifts but B_current does not:")
print("    answer basin is moving without relation-chain consistency.")
print()
print("  If B_current - B_old separates closure templates:")
print("    evidence-binding / update-binding is being captured.")
print()
print("If 3B fails:")
print("  The probe formulation may be too linguistic or too late-stage.")
print("  Next step should use contrastive hidden-state geometry rather than text probes.")
print()
print("Done.")