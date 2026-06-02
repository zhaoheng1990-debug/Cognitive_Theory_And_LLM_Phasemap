# ============================================================
# Constraint-Audit-1d
# Early-Warning Test for Constraint Collapse
#
# Goal:
#   Constraint-Audit-1c showed:
#
#       final_R = logit(C) - logit(E)
#
#   nearly perfectly predicts Gen_E, with threshold Ic ≈ 0.
#
#   But final_R is too close to output.
#   Constraint-Audit-1d removes final_R and asks:
#
#       Can earlier layer windows predict Gen_E?
#
# Windows:
#   W20_22: boundary / competition-entry only
#   W20_23: boundary + first basin layer
#   W20_24: early basin formation
#   W20_25: mid basin formation
#   W20_26: pre-final early warning
#   W20_27: oracle upper bound, includes final layer
#
# Main interpretation:
#   If W20_22 weak but W20_24/W20_26 strong:
#       L20-L22 = competition-entry zone
#       L23-L26 = answer-basin formation zone
#       L27     = output commitment zone
#
# ============================================================

import os
import re
import gc
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

RANDOM_SEED = 42

N_GRAPHS = 96
MAX_LEN = 260

BATCH_SIZE = 8
GEN_BATCH_SIZE = 8
GEN_MAX_NEW_TOKENS = 8

TRACK_LAYERS = list(range(20, 28))

EARLY_WINDOWS = {
    "W20_22_boundary_only": [20, 21, 22],
    "W20_23_boundary_plus_L23": [20, 21, 22, 23],
    "W20_24_early_basin": [20, 21, 22, 23, 24],
    "W20_25_mid_basin": [20, 21, 22, 23, 24, 25],
    "W20_26_pre_final": [20, 21, 22, 23, 24, 25, 26],
    "W20_27_oracle_with_final": [20, 21, 22, 23, 24, 25, 26, 27],
}

SAVE_DIR = Path("./constraint_audit1d_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# SEED
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Left padding is safer for decoder-only generation.
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
print("LM head:", tuple(model.lm_head.weight.shape))

if max(TRACK_LAYERS) >= num_layers:
    raise RuntimeError(f"TRACK_LAYERS includes {max(TRACK_LAYERS)}, but model has only {num_layers} layers.")

W_lm = model.lm_head.weight.detach().float().cpu().numpy().astype(np.float32)

# ============================================================
# LABELS
# ============================================================

def tokenize_no_special(text: str):
    return list(tokenizer(
        text,
        add_special_tokens=False,
        return_tensors=None,
    )["input_ids"])

def continuation_ids(label: str):
    return tokenize_no_special(" " + label)

def select_single_token_labels():
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

    single = []

    print("\nLabel tokenization audit:")
    for lab in candidate_labels:
        ids = continuation_ids(lab)
        print(f"  {lab:<10} ids={ids} len={len(ids)}")
        if len(ids) == 1:
            single.append(lab)

    if len(single) < 12:
        raise RuntimeError("Need at least 12 single-token labels.")

    chosen = single[:12]
    print("\nUsing single-token labels:", chosen)
    return chosen

LABEL_POOL = select_single_token_labels()

# ============================================================
# DATASET
# ============================================================

CONDITION_META = [
    ("clean", 0.0, "clean"),
    ("preserve_irrelevant", 0.5, "relation_preserving"),
    ("weak_distractor", 1.0, "weak_break"),
    ("ambiguous_branch", 2.0, "ambiguous_break"),
    ("strong_branch_conflict", 3.0, "competing_closure"),
    ("direct_location_conflict", 4.0, "direct_conflict"),
    ("closure_negation_conflict", 5.0, "closure_conflict"),
]

conditions = [x[0] for x in CONDITION_META]
epsilon_by_condition = {x[0]: x[1] for x in CONDITION_META}
status_by_condition = {x[0]: x[2] for x in CONDITION_META}

def make_entity(prefix: str, i: int):
    return f"{prefix}{i:03d}"

def make_prompt(a, b, d, clean_label, conflict_label, irrelevant_label, variant):
    option_line = (
        f"Possible location labels for the queried item are: {clean_label} or {conflict_label}."
    )

    if variant == "clean":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located at {clean_label}.",
            f"Question: Which location label is {a} located at?",
            f"Answer with exactly one word: {clean_label} or {conflict_label}.",
            "Answer:",
        ]

    elif variant == "preserve_irrelevant":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located at {clean_label}.",
            f"Irrelevant fact 1: {d} belongs to an unrelated group.",
            f"Irrelevant fact 2: that unrelated group is located at {irrelevant_label}.",
            f"Question: Which location label is {a} located at?",
            f"Answer with exactly one word: {clean_label} or {conflict_label}.",
            "Answer:",
        ]

    elif variant == "weak_distractor":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located at {clean_label}.",
            f"Weak distractor: Some sources say {a} may be associated with {conflict_label}.",
            f"Question: Which location label is {a} located at?",
            f"Answer with exactly one word: {clean_label} or {conflict_label}.",
            "Answer:",
        ]

    elif variant == "ambiguous_branch":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located at {clean_label}.",
            f"Ambiguous fact: {a} may also be located at {conflict_label}.",
            f"Question: Which location label is {a} located at?",
            f"Answer with exactly one word: {clean_label} or {conflict_label}.",
            "Answer:",
        ]

    elif variant == "strong_branch_conflict":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located at {clean_label}.",
            f"Conflicting fact 1: {a} belongs to another group.",
            f"Conflicting fact 2: that other group is located at {conflict_label}.",
            f"Question: Which location label is {a} located at?",
            f"Answer with exactly one word: {clean_label} or {conflict_label}.",
            "Answer:",
        ]

    elif variant == "direct_location_conflict":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located at {clean_label}.",
            f"Conflicting direct fact: {a} is located at {conflict_label}.",
            f"Question: Which location label is {a} located at?",
            f"Answer with exactly one word: {clean_label} or {conflict_label}.",
            "Answer:",
        ]

    elif variant == "closure_negation_conflict":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located at {clean_label}.",
            f"Updated record: {b} is located at {conflict_label}, not at {clean_label}.",
            f"Question: Which location label is {a} located at?",
            f"Answer with exactly one word: {clean_label} or {conflict_label}.",
            "Answer:",
        ]

    else:
        raise ValueError(f"Unknown variant: {variant}")

    return "\n".join(lines)

def build_dataset(n_graphs: int):
    records = []
    texts_by_condition = {c: [] for c in conditions}

    n_labels = len(LABEL_POOL)

    for i in range(n_graphs):
        a = make_entity("A", i)
        b = make_entity("B", i)
        d = make_entity("D", i)

        clean_label = LABEL_POOL[(2 * i) % n_labels]
        conflict_label = LABEL_POOL[(2 * i + 1) % n_labels]
        irrelevant_label = LABEL_POOL[(2 * i + 2) % n_labels]

        prompts = {}

        for cond in conditions:
            prompts[cond] = make_prompt(
                a=a,
                b=b,
                d=d,
                clean_label=clean_label,
                conflict_label=conflict_label,
                irrelevant_label=irrelevant_label,
                variant=cond,
            )
            texts_by_condition[cond].append(prompts[cond])

        clean_id = continuation_ids(clean_label)[0]
        conflict_id = continuation_ids(conflict_label)[0]

        records.append({
            "idx": i,
            "clean_label": clean_label,
            "conflict_label": conflict_label,
            "clean_token_id": clean_id,
            "conflict_token_id": conflict_id,
            "prompts": prompts,
        })

    return records, texts_by_condition

records, texts_by_condition = build_dataset(N_GRAPHS)

print("\nExample clean prompt:\n")
print(texts_by_condition["clean"][0])
print("\nExample closure_negation prompt:\n")
print(texts_by_condition["closure_negation_conflict"][0])

clean_token_ids = np.array([r["clean_token_id"] for r in records], dtype=np.int64)
conflict_token_ids = np.array([r["conflict_token_id"] for r in records], dtype=np.int64)

# ============================================================
# NUMERIC HELPERS
# ============================================================

def stable_sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x, dtype=np.float64)

    pos = x >= 0
    neg = ~pos

    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[neg])
    out[neg] = expx / (1.0 + expx)

    return out.astype(np.float32)

def binary_entropy_from_p(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    q = 1.0 - p
    h = -(p * np.log(p) + q * np.log(q)) / np.log(2.0)
    return h.astype(np.float32)

def sanitize_array(X, clip=1e6):
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=clip, neginf=-clip)
    X = np.clip(X, -clip, clip)
    return X.astype(np.float32)

def safe_auc(y, score):
    y = np.asarray(y).astype(int)
    score = sanitize_array(score)

    if len(np.unique(y)) < 2:
        return np.nan

    try:
        return float(roc_auc_score(y, score))
    except Exception:
        return np.nan

def normalize_first_word(text: str):
    text = text.strip()
    if not text:
        return ""
    m = re.search(r"[A-Za-z]+", text)
    if not m:
        return ""
    return m.group(0).lower()

def calc_metrics_from_prob(y_true, prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    prob = sanitize_array(prob)
    pred = (prob >= threshold).astype(int)

    out = {
        "auc": safe_auc(y_true, prob),
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "pred_pos_rate": float(np.mean(pred)),
        "mean_prob": float(np.mean(prob)),
    }

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    except Exception:
        out.update({"tn": 0, "fp": 0, "fn": 0, "tp": 0})

    return out

# ============================================================
# GENERATION CHECK
# ============================================================

def generation_check_condition(cond):
    print(f"\nGeneration check condition={cond} ...")

    rows = []
    prompts = texts_by_condition[cond]
    pad_id = tokenizer.pad_token_id

    with torch.no_grad():
        for start in range(0, len(prompts), GEN_BATCH_SIZE):
            batch_prompts = prompts[start:start + GEN_BATCH_SIZE]

            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
            ).to(DEVICE)

            input_len = inputs["input_ids"].shape[1]

            out = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=GEN_MAX_NEW_TOKENS,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            for bi in range(len(batch_prompts)):
                idx = start + bi
                rec = records[idx]

                gen_ids = out[bi, input_len:]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                first_word = normalize_first_word(gen_text)

                clean_label = rec["clean_label"]
                conflict_label = rec["conflict_label"]

                is_clean = first_word == clean_label.lower()
                is_conflict = first_word == conflict_label.lower()

                rows.append({
                    "condition": cond,
                    "epsilon": epsilon_by_condition[cond],
                    "status": status_by_condition[cond],
                    "idx": idx,
                    "generated": gen_text,
                    "first_word": first_word,
                    "clean_label": clean_label,
                    "conflict_label": conflict_label,
                    "is_clean": bool(is_clean),
                    "is_conflict": bool(is_conflict),
                    "is_other": bool((not is_clean) and (not is_conflict)),
                })

            if (start // GEN_BATCH_SIZE) % 10 == 0:
                print(f"  generated {min(start + GEN_BATCH_SIZE, len(prompts))}/{len(prompts)}")

            del inputs, out
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    clean_rate = float(np.mean([r["is_clean"] for r in rows]))
    conflict_rate = float(np.mean([r["is_conflict"] for r in rows]))
    other_rate = float(np.mean([r["is_other"] for r in rows]))

    print(
        f"Generation summary {cond}: "
        f"clean={clean_rate:.4f}, conflict={conflict_rate:.4f}, other={other_rate:.4f}"
    )

    return rows

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def get_last_positions(attention_mask):
    # With left padding, the last real token is at sequence_length - 1.
    return torch.full(
        (attention_mask.shape[0],),
        attention_mask.shape[1] - 1,
        dtype=torch.long,
        device=attention_mask.device,
    )

def extract_layer_margin_features_for_condition(cond):
    """
    Extract only answer-margin features:
        R_l = logit_l(C) - logit_l(E)
        H_l = binary entropy over C/E
        pC_l = sigmoid(R_l)

    This avoids expensive TopK computation.
    """

    print(f"\nExtracting R/H features for condition={cond} ...")

    texts = texts_by_condition[cond]

    R_by_layer = {l: np.zeros(N_GRAPHS, dtype=np.float32) for l in TRACK_LAYERS}
    P_by_layer = {l: np.zeros(N_GRAPHS, dtype=np.float32) for l in TRACK_LAYERS}
    H_by_layer = {l: np.zeros(N_GRAPHS, dtype=np.float32) for l in TRACK_LAYERS}

    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[start:start + BATCH_SIZE]
            end = min(start + BATCH_SIZE, len(texts))

            inputs = tokenizer(
                batch_texts,
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

            hidden_states = outputs.hidden_states[1:]
            last_idx = get_last_positions(inputs["attention_mask"])

            batch_indices = np.arange(start, end)
            c_ids = clean_token_ids[batch_indices]
            e_ids = conflict_token_ids[batch_indices]

            Wc = W_lm[c_ids]
            We = W_lm[e_ids]

            for l in TRACK_LAYERS:
                h = hidden_states[l]
                picked = h[
                    torch.arange(h.shape[0], device=h.device),
                    last_idx,
                    :
                ].detach().float().cpu().numpy().astype(np.float32)

                clean_logits = np.sum(picked * Wc, axis=1)
                conflict_logits = np.sum(picked * We, axis=1)

                R = clean_logits - conflict_logits
                pC = stable_sigmoid(R)
                H2 = binary_entropy_from_p(pC)

                R_by_layer[l][batch_indices] = R.astype(np.float32)
                P_by_layer[l][batch_indices] = pC.astype(np.float32)
                H_by_layer[l][batch_indices] = H2.astype(np.float32)

            if (start // BATCH_SIZE) % 10 == 0:
                print(f"  extracted {end}/{len(texts)}")

            del outputs, hidden_states, inputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    rows = []

    for i in range(N_GRAPHS):
        row = {
            "condition": cond,
            "epsilon": epsilon_by_condition[cond],
            "status": status_by_condition[cond],
            "idx": i,
            "clean_label": records[i]["clean_label"],
            "conflict_label": records[i]["conflict_label"],
        }

        for l in TRACK_LAYERS:
            row[f"R_L{l}"] = float(R_by_layer[l][i])
            row[f"pC_L{l}"] = float(P_by_layer[l][i])
            row[f"H_L{l}"] = float(H_by_layer[l][i])

        rows.append(row)

    return rows

# ============================================================
# WINDOW FEATURES
# ============================================================

def add_window_features(df, window_name, layers):
    out = df.copy()

    R_cols = [f"R_L{l}" for l in layers]
    H_cols = [f"H_L{l}" for l in layers]
    p_cols = [f"pC_L{l}" for l in layers]

    R = out[R_cols].values.astype(np.float32)
    H = out[H_cols].values.astype(np.float32)
    P = out[p_cols].values.astype(np.float32)

    prefix = window_name

    out[f"{prefix}_R_mean"] = R.mean(axis=1)
    out[f"{prefix}_R_min"] = R.min(axis=1)
    out[f"{prefix}_R_max"] = R.max(axis=1)
    out[f"{prefix}_R_first"] = R[:, 0]
    out[f"{prefix}_R_last"] = R[:, -1]
    out[f"{prefix}_R_slope"] = R[:, -1] - R[:, 0]
    out[f"{prefix}_R_drop"] = R[:, 0] - R[:, -1]
    out[f"{prefix}_R_range"] = R.max(axis=1) - R.min(axis=1)

    out[f"{prefix}_absR_mean"] = np.abs(R).mean(axis=1)
    out[f"{prefix}_absR_min"] = np.abs(R).min(axis=1)
    out[f"{prefix}_absR_last"] = np.abs(R[:, -1])

    out[f"{prefix}_H_mean"] = H.mean(axis=1)
    out[f"{prefix}_H_max"] = H.max(axis=1)
    out[f"{prefix}_H_last"] = H[:, -1]
    out[f"{prefix}_H_slope"] = H[:, -1] - H[:, 0]

    out[f"{prefix}_pC_mean"] = P.mean(axis=1)
    out[f"{prefix}_pC_min"] = P.min(axis=1)
    out[f"{prefix}_pC_last"] = P[:, -1]

    out[f"{prefix}_num_negative"] = (R < 0).sum(axis=1).astype(np.float32)
    out[f"{prefix}_any_negative"] = (R < 0).any(axis=1).astype(np.float32)

    # First negative position inside window. If none, use len(layers).
    first_neg = []
    for row in R:
        neg = np.where(row < 0)[0]
        first_neg.append(float(neg[0]) if len(neg) > 0 else float(len(layers)))
    out[f"{prefix}_first_negative_offset"] = np.array(first_neg, dtype=np.float32)

    return out

def feature_names_for_window(window_name):
    prefix = window_name
    return [
        f"{prefix}_R_mean",
        f"{prefix}_R_min",
        f"{prefix}_R_max",
        f"{prefix}_R_first",
        f"{prefix}_R_last",
        f"{prefix}_R_slope",
        f"{prefix}_R_drop",
        f"{prefix}_R_range",
        f"{prefix}_absR_mean",
        f"{prefix}_absR_min",
        f"{prefix}_absR_last",
        f"{prefix}_H_mean",
        f"{prefix}_H_max",
        f"{prefix}_H_last",
        f"{prefix}_H_slope",
        f"{prefix}_pC_mean",
        f"{prefix}_pC_min",
        f"{prefix}_pC_last",
        f"{prefix}_num_negative",
        f"{prefix}_any_negative",
        f"{prefix}_first_negative_offset",
    ]

# ============================================================
# THRESHOLD SCAN
# ============================================================

def threshold_scan(y, x, lower_means_conflict=True):
    y = np.asarray(y).astype(int)
    x = sanitize_array(x)

    qs = np.linspace(0.01, 0.99, 99)
    thresholds = np.unique(np.quantile(x, qs))

    best = None

    for t in thresholds:
        if lower_means_conflict:
            pred = (x <= t).astype(int)
        else:
            pred = (x >= t).astype(int)

        acc = accuracy_score(y, pred)
        f1 = f1_score(y, pred, zero_division=0)
        prec = precision_score(y, pred, zero_division=0)
        rec = recall_score(y, pred, zero_division=0)

        if best is None or f1 > best["f1"]:
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            best = {
                "threshold": float(t),
                "accuracy": float(acc),
                "f1": float(f1),
                "precision": float(prec),
                "recall": float(rec),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "lower_means_conflict": bool(lower_means_conflict),
            }

    return best

def evaluate_single_features(df, features):
    y = df["y_conflict"].values.astype(int)
    rows = []

    for feat in features:
        x = sanitize_array(df[feat].values)

        auc_high = safe_auc(y, x)
        auc_low = safe_auc(y, -x)

        lower = auc_low >= auc_high
        best = threshold_scan(y, x, lower_means_conflict=lower)

        row = {
            "feature": feat,
            "auc_conflict_high": auc_high,
            "auc_conflict_low": auc_low,
            "best_direction": "low" if lower else "high",
            **best,
        }

        rows.append(row)

    return pd.DataFrame(rows)

# ============================================================
# LOGISTIC EVALUATION
# ============================================================

def make_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            max_iter=500,
        )),
    ])

def evaluate_stratified_cv(df, features, n_splits=5):
    X = sanitize_array(df[features].values)
    y = df["y_conflict"].values.astype(int)

    pipe = make_pipeline()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    prob = np.zeros(len(y), dtype=np.float32)

    for train_idx, test_idx in skf.split(X, y):
        pipe.fit(X[train_idx], y[train_idx])
        prob[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]

    return calc_metrics_from_prob(y, prob)

def evaluate_group_cv_by_graph(df, features, n_splits=5):
    X = sanitize_array(df[features].values)
    y = df["y_conflict"].values.astype(int)
    groups = df["idx"].values

    pipe = make_pipeline()
    gkf = GroupKFold(n_splits=n_splits)

    prob = np.zeros(len(y), dtype=np.float32)

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        if len(np.unique(y[train_idx])) < 2:
            prob[test_idx] = np.mean(y[train_idx])
            continue

        pipe.fit(X[train_idx], y[train_idx])
        prob[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]

    return calc_metrics_from_prob(y, prob)

def evaluate_leave_condition_out(df, features):
    X = sanitize_array(df[features].values)
    y = df["y_conflict"].values.astype(int)
    groups = df["condition"].values

    pipe = make_pipeline()
    logo = LeaveOneGroupOut()

    rows = []

    for train_idx, test_idx in logo.split(X, y, groups=groups):
        heldout = groups[test_idx][0]

        if len(np.unique(y[train_idx])) < 2:
            prob = np.full(len(test_idx), np.mean(y[train_idx]), dtype=np.float32)
        else:
            pipe.fit(X[train_idx], y[train_idx])
            prob = pipe.predict_proba(X[test_idx])[:, 1]

        metrics = calc_metrics_from_prob(y[test_idx], prob)

        rows.append({
            "heldout_condition": heldout,
            "n": int(len(test_idx)),
            "positive_rate": float(np.mean(y[test_idx])),
            **metrics,
        })

    return pd.DataFrame(rows)

def fit_full_coefficients(df, features):
    X = sanitize_array(df[features].values)
    y = df["y_conflict"].values.astype(int)

    pipe = make_pipeline()
    pipe.fit(X, y)

    coefs = pipe.named_steps["clf"].coef_[0]

    rows = []
    for feat, coef in sorted(zip(features, coefs), key=lambda x: abs(x[1]), reverse=True):
        rows.append({
            "feature": feat,
            "coef": float(coef),
        })

    return pd.DataFrame(rows)

# ============================================================
# MAIN ANALYSIS
# ============================================================

def run_1d_analysis(feature_rows, generation_rows):
    df_feat = pd.DataFrame(feature_rows)
    df_gen = pd.DataFrame(generation_rows)

    df = df_feat.merge(
        df_gen[[
            "condition", "idx",
            "generated", "first_word",
            "is_clean", "is_conflict", "is_other",
        ]],
        on=["condition", "idx"],
        how="left",
    )

    df_bin = df[(df["is_clean"] == True) | (df["is_conflict"] == True)].copy()
    df_bin["y_conflict"] = df_bin["is_conflict"].astype(int)

    print("\n\n============================================================")
    print("Constraint-Audit-1d Dataset Summary")
    print("============================================================\n")

    print("Total rows:", len(df))
    print("Binary C/E rows:", len(df_bin))

    print("\nGeneration by condition:")
    for cond in conditions:
        sub = df[df["condition"] == cond]
        print(
            f"{cond:<28} "
            f"N={len(sub):<4} "
            f"Gen_C={sub['is_clean'].mean():.4f} "
            f"Gen_E={sub['is_conflict'].mean():.4f} "
            f"Other={sub['is_other'].mean():.4f}"
        )

    window_summary_rows = []
    all_threshold_rows = []
    all_logo_rows = []
    all_coef_rows = []

    for window_name, layers in EARLY_WINDOWS.items():
        print("\n\n############################################################")
        print(f"Window: {window_name} | layers={layers}")
        print("############################################################")

        df_win = add_window_features(df_bin, window_name, layers)
        features = feature_names_for_window(window_name)

        # Single-feature threshold scan.
        threshold_df = evaluate_single_features(df_win, features)
        threshold_df.insert(0, "window", window_name)
        all_threshold_rows.append(threshold_df)

        threshold_df_sorted = threshold_df.sort_values(
            by=["f1", "accuracy"],
            ascending=False,
        )

        print("\nTop single-feature thresholds:")
        print(
            f"{'feature':<45}"
            f"{'dir':<8}"
            f"{'auc_hi':<10}"
            f"{'auc_lo':<10}"
            f"{'thr':<12}"
            f"{'acc':<10}"
            f"{'f1':<10}"
            f"{'prec':<10}"
            f"{'rec':<10}"
        )

        for _, r in threshold_df_sorted.head(8).iterrows():
            print(
                f"{r['feature']:<45}"
                f"{r['best_direction']:<8}"
                f"{r['auc_conflict_high']:<10.4f}"
                f"{r['auc_conflict_low']:<10.4f}"
                f"{r['threshold']:<12.4f}"
                f"{r['accuracy']:<10.4f}"
                f"{r['f1']:<10.4f}"
                f"{r['precision']:<10.4f}"
                f"{r['recall']:<10.4f}"
            )

        # Logistic model.
        strat_metrics = evaluate_stratified_cv(df_win, features)
        graph_metrics = evaluate_group_cv_by_graph(df_win, features)
        logo_df = evaluate_leave_condition_out(df_win, features)
        logo_df.insert(0, "window", window_name)
        all_logo_rows.append(logo_df)

        coef_df = fit_full_coefficients(df_win, features)
        coef_df.insert(0, "window", window_name)
        all_coef_rows.append(coef_df)

        best_single = threshold_df_sorted.iloc[0].to_dict()

        row = {
            "window": window_name,
            "layers": ",".join(map(str, layers)),
            "n_layers": len(layers),

            "best_single_feature": best_single["feature"],
            "best_single_direction": best_single["best_direction"],
            "best_single_threshold": best_single["threshold"],
            "best_single_acc": best_single["accuracy"],
            "best_single_f1": best_single["f1"],
            "best_single_precision": best_single["precision"],
            "best_single_recall": best_single["recall"],
            "best_single_auc_high": best_single["auc_conflict_high"],
            "best_single_auc_low": best_single["auc_conflict_low"],

            "strat_auc": strat_metrics["auc"],
            "strat_acc": strat_metrics["accuracy"],
            "strat_f1": strat_metrics["f1"],
            "strat_precision": strat_metrics["precision"],
            "strat_recall": strat_metrics["recall"],

            "graph_auc": graph_metrics["auc"],
            "graph_acc": graph_metrics["accuracy"],
            "graph_f1": graph_metrics["f1"],
            "graph_precision": graph_metrics["precision"],
            "graph_recall": graph_metrics["recall"],
        }

        window_summary_rows.append(row)

        print("\nLogistic CV:")
        print(
            f"  Stratified 5-fold: "
            f"AUC={strat_metrics['auc']:.4f}, "
            f"Acc={strat_metrics['accuracy']:.4f}, "
            f"F1={strat_metrics['f1']:.4f}, "
            f"Prec={strat_metrics['precision']:.4f}, "
            f"Rec={strat_metrics['recall']:.4f}"
        )

        print(
            f"  GroupKFold by graph idx: "
            f"AUC={graph_metrics['auc']:.4f}, "
            f"Acc={graph_metrics['accuracy']:.4f}, "
            f"F1={graph_metrics['f1']:.4f}, "
            f"Prec={graph_metrics['precision']:.4f}, "
            f"Rec={graph_metrics['recall']:.4f}"
        )

        print("\nLeave-one-condition-out:")
        for _, rr in logo_df.iterrows():
            auc_str = "nan" if pd.isna(rr["auc"]) else f"{rr['auc']:.4f}"
            print(
                f"  {rr['heldout_condition']:<28} "
                f"pos={rr['positive_rate']:.4f} "
                f"auc={auc_str} "
                f"acc={rr['accuracy']:.4f} "
                f"f1={rr['f1']:.4f} "
                f"predE={rr['pred_pos_rate']:.4f} "
                f"pE={rr['mean_prob']:.4f}"
            )

        print("\nTop coefficients:")
        for _, rr in coef_df.head(8).iterrows():
            print(f"  {rr['feature']:<45} {rr['coef']:+.4f}")

    summary_df = pd.DataFrame(window_summary_rows)
    threshold_all = pd.concat(all_threshold_rows, axis=0, ignore_index=True)
    logo_all = pd.concat(all_logo_rows, axis=0, ignore_index=True)
    coef_all = pd.concat(all_coef_rows, axis=0, ignore_index=True)

    print("\n\n============================================================")
    print("Constraint-Audit-1d Window Summary")
    print("============================================================\n")

    print(
        f"{'window':<30}"
        f"{'best_feat':<42}"
        f"{'best_f1':<10}"
        f"{'best_acc':<10}"
        f"{'strat_auc':<10}"
        f"{'strat_f1':<10}"
        f"{'graph_auc':<10}"
        f"{'graph_f1':<10}"
    )

    for _, r in summary_df.iterrows():
        print(
            f"{r['window']:<30}"
            f"{r['best_single_feature']:<42}"
            f"{r['best_single_f1']:<10.4f}"
            f"{r['best_single_acc']:<10.4f}"
            f"{r['strat_auc']:<10.4f}"
            f"{r['strat_f1']:<10.4f}"
            f"{r['graph_auc']:<10.4f}"
            f"{r['graph_f1']:<10.4f}"
        )

    return df, df_bin, summary_df, threshold_all, logo_all, coef_all

# ============================================================
# RUN
# ============================================================

all_generation_rows = []
all_feature_rows = []

for cond in conditions:
    print("\n\n############################################################")
    print(f"Condition: {cond}")
    print(f"epsilon={epsilon_by_condition[cond]}, status={status_by_condition[cond]}")
    print("############################################################")

    gen_rows = generation_check_condition(cond)
    all_generation_rows.extend(gen_rows)

    feat_rows = extract_layer_margin_features_for_condition(cond)
    all_feature_rows.extend(feat_rows)

df_all, df_bin, summary_df, threshold_df, logo_df, coef_df = run_1d_analysis(
    all_feature_rows,
    all_generation_rows,
)

# ============================================================
# SAVE OUTPUTS
# ============================================================

features_path = SAVE_DIR / "constraint_audit1d_layer_margin_features.csv"
binary_path = SAVE_DIR / "constraint_audit1d_binary_dataset.csv"
generation_path = SAVE_DIR / "constraint_audit1d_generation.csv"
summary_path = SAVE_DIR / "constraint_audit1d_window_summary.csv"
threshold_path = SAVE_DIR / "constraint_audit1d_thresholds.csv"
logo_path = SAVE_DIR / "constraint_audit1d_leave_condition_out.csv"
coef_path = SAVE_DIR / "constraint_audit1d_logistic_coefficients.csv"
prompt_path = SAVE_DIR / "constraint_audit1d_prompts.txt"

df_all.to_csv(features_path, index=False)
df_bin.to_csv(binary_path, index=False)
pd.DataFrame(all_generation_rows).to_csv(generation_path, index=False)
summary_df.to_csv(summary_path, index=False)
threshold_df.to_csv(threshold_path, index=False)
logo_df.to_csv(logo_path, index=False)
coef_df.to_csv(coef_path, index=False)

with open(prompt_path, "w", encoding="utf-8") as fp:
    fp.write("LABEL_POOL:\n")
    fp.write(str(LABEL_POOL) + "\n\n")

    for lab in LABEL_POOL:
        fp.write(f"{lab}: ids={continuation_ids(lab)}\n")

    fp.write("\nPrompts:\n")
    for i, rec in enumerate(records[:10]):
        fp.write(f"===== graph {i} =====\n")
        fp.write(f"clean label: {rec['clean_label']}\n")
        fp.write(f"conflict label: {rec['conflict_label']}\n")
        for cond in conditions:
            fp.write(
                f"\n--- {cond} | eps={epsilon_by_condition[cond]} | "
                f"{status_by_condition[cond]} ---\n"
            )
            fp.write(rec["prompts"][cond] + "\n")
        fp.write("\n")

print("\nSaved outputs:")
print(" ", features_path)
print(" ", binary_path)
print(" ", generation_path)
print(" ", summary_path)
print(" ", threshold_path)
print(" ", logo_path)
print(" ", coef_path)
print(" ", prompt_path)

# ============================================================
# INTERPRETATION GUIDE
# ============================================================

print("\n\n============================================================")
print("Constraint-Audit-1d Interpretation Guide")
print("============================================================\n")

print("Main question:")
print("  Can early windows predict Gen_E without using final_R?")
print()
print("Strong early-warning result if:")
print("  1. W20_22 is weaker than W20_24/W20_26.")
print("  2. W20_24 or W20_26 achieves high AUC/F1 without L27.")
print("  3. GroupKFold by graph remains strong.")
print("  4. Leave-one-condition-out is not purely memorizing condition labels.")
print()
print("Expected layer interpretation:")
print("  W20_22 strong enough but not perfect:")
print("    L20-L22 is competition-entry / boundary recoupling.")
print("  W20_24 improves sharply:")
print("    L23-L24 begins answer-basin formation.")
print("  W20_26 approaches oracle:")
print("    L25-L26 contains pre-final commitment signal.")
print("  W20_27 oracle upper bound:")
print("    Includes final output commitment and should be strongest.")
print()
print("Important caveat:")
print("  If only W20_27 is strong, then 1c mainly found an output-layer threshold.")
print("  If W20_24/W20_26 are strong, then constraint collapse has real early-warning structure.")
print()
print("Done.")