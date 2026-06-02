# ============================================================
# Constraint-Audit-1c
# Critical Threshold Estimation for Constraint Collapse
#
# Goal:
#   Constraint-Audit-1b validated logit-margin constraint state:
#
#       R_l = logit_l(clean_label) - logit_l(conflict_label)
#
#   and showed:
#       clean:
#           Final_R > 0, Gen_C = 1
#       closure_negation:
#           Final_R < 0, Gen_E = 1
#
#   Constraint-Audit-1c asks:
#
#       Can C_l features predict generation flip?
#       Is there a critical threshold I_c?
#
# Core target:
#       y = 1 if generated conflict answer E
#       y = 0 if generated clean answer C
#
# Candidate threshold variables:
#       boundary_R_mean
#       boundary_entropy_mean
#       late_R_mean
#       final_R
#       final_entropy
#       min_R_20_27
#       R_drop_boundary_to_final
#
# Optional F features:
#       TopK spread / TopK gap at boundary and final layers
#
# Main outputs:
#       1. feature AUC
#       2. best threshold per feature
#       3. logistic prediction performance
#       4. critical zone estimate
#
# ============================================================

import os
import re
import gc
import random
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

device = "cuda"
dtype = torch.float16

RANDOM_SEED = 42

N_GRAPHS = 96
MAX_LEN = 260
BATCH_SIZE = 4

# R/B are cheap. F is expensive because it computes TopK over full vocab.
# Keep True for full 1c; set False if you only want R/B threshold estimation.
COMPUTE_TOPK_FEATURES = True
TOPK = 5000
CENTER_HIDDEN_FOR_TOPK = True

BOUNDARY_LAYERS = [20, 21, 22]
LATE_LAYERS = [23, 24, 25, 26, 27]
TRACK_LAYERS = sorted(set(BOUNDARY_LAYERS + LATE_LAYERS))

DO_GENERATION_CHECK = True
GEN_BATCH_SIZE = 4
GEN_MAX_NEW_TOKENS = 8

SAVE_DIR = "./constraint_audit1c_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# SEED
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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

# Right padding is fine for hidden capture; generation slicing uses max input length.
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=dtype,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True,
)

model.eval()

num_layers = len(model.model.layers)
print("Num layers:", num_layers)
print("LM head:", tuple(model.lm_head.weight.shape))

W_lm = model.lm_head.weight.detach().float().cpu()
W_np = W_lm.numpy().astype(np.float32)
W_norm = W_np / (np.linalg.norm(W_np, axis=1, keepdims=True) + 1e-12)

vocab_size, d_model = W_np.shape

# ============================================================
# LABELS
# ============================================================

def tokenize_no_special(text):
    return list(tokenizer(
        text,
        add_special_tokens=False,
        return_tensors=None,
    )["input_ids"])

def continuation_ids(label):
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

    if len(single) < 8:
        raise RuntimeError("Need at least 8 single-token labels.")

    chosen = single[:12]
    print("\nUsing single-token labels:", chosen)
    return chosen

LABEL_POOL = select_single_token_labels()

# ============================================================
# DATASET
# ============================================================

def make_entity(prefix, i):
    return f"{prefix}{i:03d}"

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

def build_dataset(n_graphs):
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

# ============================================================
# GENERATION CHECK
# ============================================================

def normalize_first_word(text):
    text = text.strip()
    if not text:
        return ""
    m = re.search(r"[A-Za-z]+", text)
    if not m:
        return ""
    return m.group(0).lower()

def generation_check_condition(cond):
    if not DO_GENERATION_CHECK:
        return []

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
            ).to(device)

            # Generation appends after the padded batch length.
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

                rows.append({
                    "condition": cond,
                    "epsilon": epsilon_by_condition[cond],
                    "status": status_by_condition[cond],
                    "idx": idx,
                    "generated": gen_text,
                    "first_word": first_word,
                    "clean_label": clean_label,
                    "conflict_label": conflict_label,
                    "is_clean": first_word == clean_label.lower(),
                    "is_conflict": first_word == conflict_label.lower(),
                    "is_other": (first_word != clean_label.lower()) and (first_word != conflict_label.lower()),
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
# TOPK FEATURES
# ============================================================

def compute_topk_ids_and_spread(H, k):
    logits = H @ W_np.T
    ids = np.argpartition(logits, -k, axis=1)[:, -k:]

    vals = np.take_along_axis(logits, ids, axis=1)
    order = np.argsort(vals, axis=1)[:, ::-1]
    ids = np.take_along_axis(ids, order, axis=1).astype(np.int32)

    N = ids.shape[0]
    spread = np.zeros(N, dtype=np.float32)

    batch_rows = 4
    for s in range(0, N, batch_rows):
        e = min(s + batch_rows, N)
        batch_ids = ids[s:e]
        E = W_norm[batch_ids]

        C = E.mean(axis=1)
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)

        sims = np.einsum("bkd,bd->bk", E, C)
        spr = np.mean(1.0 - sims, axis=1)
        spread[s:e] = spr.astype(np.float32)

    return ids, spread

def membership_rate_rows(ids, token_ids):
    out = np.zeros(ids.shape[0], dtype=np.float32)
    for i in range(ids.shape[0]):
        s = set(ids[i].tolist())
        out[i] = 1.0 if token_ids[i] in s else 0.0
    return out

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_constraint_features_for_condition(cond):
    """
    Returns sample-level feature rows:
      one row per graph/sample for a given condition.
    """

    print(f"\nExtracting constraint features for condition={cond} ...")

    texts = texts_by_condition[cond]

    # Store per-sample per-layer R/B values.
    R_by_layer = {l: np.zeros(N_GRAPHS, dtype=np.float32) for l in TRACK_LAYERS}
    P_by_layer = {l: np.zeros(N_GRAPHS, dtype=np.float32) for l in TRACK_LAYERS}
    H_by_layer = {l: np.zeros(N_GRAPHS, dtype=np.float32) for l in TRACK_LAYERS}

    if COMPUTE_TOPK_FEATURES:
        Spread_by_layer = {l: np.zeros(N_GRAPHS, dtype=np.float32) for l in TRACK_LAYERS}
        TopKGap_by_layer = {l: np.zeros(N_GRAPHS, dtype=np.float32) for l in TRACK_LAYERS}
    else:
        Spread_by_layer = None
        TopKGap_by_layer = None

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
            ).to(device)

            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )

            hidden_states = outputs.hidden_states[1:]
            attn = inputs["attention_mask"]
            last_idx = attn.sum(dim=1) - 1

            batch_indices = np.arange(start, end)

            for l in TRACK_LAYERS:
                h = hidden_states[l]
                picked = h[
                    torch.arange(h.shape[0], device=h.device),
                    last_idx,
                    :
                ].detach().float().cpu().numpy().astype(np.float32)

                logits = picked @ W_np.T

                c_ids = clean_token_ids[batch_indices]
                e_ids = conflict_token_ids[batch_indices]

                clean_logits = logits[np.arange(len(batch_indices)), c_ids]
                conflict_logits = logits[np.arange(len(batch_indices)), e_ids]

                R = clean_logits - conflict_logits
                P = stable_sigmoid(R)
                H2 = binary_entropy_from_p(P)

                R_by_layer[l][batch_indices] = R.astype(np.float32)
                P_by_layer[l][batch_indices] = P.astype(np.float32)
                H_by_layer[l][batch_indices] = H2.astype(np.float32)

                if COMPUTE_TOPK_FEATURES:
                    if CENTER_HIDDEN_FOR_TOPK:
                        H_topk = picked - picked.mean(axis=0, keepdims=True)
                    else:
                        H_topk = picked

                    ids, spread = compute_topk_ids_and_spread(H_topk, TOPK)

                    c_in = membership_rate_rows(ids, c_ids)
                    e_in = membership_rate_rows(ids, e_ids)
                    gap = c_in - e_in

                    Spread_by_layer[l][batch_indices] = spread.astype(np.float32)
                    TopKGap_by_layer[l][batch_indices] = gap.astype(np.float32)

            if (start // BATCH_SIZE) % 10 == 0:
                print(f"  extracted {end}/{len(texts)}")

            del outputs, hidden_states, inputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    rows = []

    for i in range(N_GRAPHS):
        boundary_R = np.array([R_by_layer[l][i] for l in BOUNDARY_LAYERS], dtype=np.float32)
        late_R = np.array([R_by_layer[l][i] for l in LATE_LAYERS], dtype=np.float32)
        all_R = np.array([R_by_layer[l][i] for l in TRACK_LAYERS], dtype=np.float32)

        boundary_H = np.array([H_by_layer[l][i] for l in BOUNDARY_LAYERS], dtype=np.float32)
        late_H = np.array([H_by_layer[l][i] for l in LATE_LAYERS], dtype=np.float32)

        row = {
            "condition": cond,
            "epsilon": epsilon_by_condition[cond],
            "status": status_by_condition[cond],
            "idx": i,
            "clean_label": records[i]["clean_label"],
            "conflict_label": records[i]["conflict_label"],

            "boundary_R_mean": float(np.mean(boundary_R)),
            "boundary_R_min": float(np.min(boundary_R)),
            "boundary_R_max": float(np.max(boundary_R)),
            "boundary_R_L20": float(R_by_layer[20][i]),
            "boundary_R_L21": float(R_by_layer[21][i]),
            "boundary_R_L22": float(R_by_layer[22][i]),

            "boundary_entropy_mean": float(np.mean(boundary_H)),
            "boundary_entropy_max": float(np.max(boundary_H)),
            "boundary_distance_mean": float(np.mean(np.abs(boundary_R))),

            "late_R_mean": float(np.mean(late_R)),
            "late_R_min": float(np.min(late_R)),
            "late_R_max": float(np.max(late_R)),
            "late_entropy_mean": float(np.mean(late_H)),
            "late_entropy_max": float(np.max(late_H)),
            "late_distance_mean": float(np.mean(np.abs(late_R))),

            "final_R": float(R_by_layer[27][i]),
            "final_p_clean": float(P_by_layer[27][i]),
            "final_entropy": float(H_by_layer[27][i]),
            "final_distance": float(abs(R_by_layer[27][i])),

            "min_R_20_27": float(np.min(all_R)),
            "max_R_20_27": float(np.max(all_R)),
            "mean_R_20_27": float(np.mean(all_R)),

            "R_slope_20_to_27": float(R_by_layer[27][i] - R_by_layer[20][i]),
            "R_drop_boundary_to_final": float(np.mean(boundary_R) - R_by_layer[27][i]),
            "R_drop_late_to_final": float(np.mean(late_R) - R_by_layer[27][i]),

            "first_negative_layer_20_27": -1,
            "num_negative_layers_20_27": int(np.sum(all_R < 0)),
        }

        for l in TRACK_LAYERS:
            if R_by_layer[l][i] < 0:
                row["first_negative_layer_20_27"] = int(l)
                break

        if COMPUTE_TOPK_FEATURES:
            boundary_spread = np.array([Spread_by_layer[l][i] for l in BOUNDARY_LAYERS], dtype=np.float32)
            late_spread = np.array([Spread_by_layer[l][i] for l in LATE_LAYERS], dtype=np.float32)
            boundary_gap = np.array([TopKGap_by_layer[l][i] for l in BOUNDARY_LAYERS], dtype=np.float32)
            late_gap = np.array([TopKGap_by_layer[l][i] for l in LATE_LAYERS], dtype=np.float32)

            row.update({
                "boundary_spread_mean": float(np.mean(boundary_spread)),
                "late_spread_mean": float(np.mean(late_spread)),
                "final_spread": float(Spread_by_layer[27][i]),

                "boundary_topk_gap_mean": float(np.mean(boundary_gap)),
                "late_topk_gap_mean": float(np.mean(late_gap)),
                "final_topk_gap": float(TopKGap_by_layer[27][i]),
            })
        else:
            row.update({
                "boundary_spread_mean": np.nan,
                "late_spread_mean": np.nan,
                "final_spread": np.nan,

                "boundary_topk_gap_mean": np.nan,
                "late_topk_gap_mean": np.nan,
                "final_topk_gap": np.nan,
            })

        rows.append(row)

    return rows

# ============================================================
# THRESHOLD / PREDICTION ANALYSIS
# ============================================================

def safe_auc(y, score):
    y = np.asarray(y)
    score = np.asarray(score)

    if len(np.unique(y)) < 2:
        return np.nan

    try:
        return float(roc_auc_score(y, score))
    except Exception:
        return np.nan

def threshold_scan(y, x, lower_means_conflict=True):
    """
    Find threshold t for predicting conflict.
      if lower_means_conflict:
          pred = x <= t
      else:
          pred = x >= t
    """

    y = np.asarray(y).astype(int)
    x = np.asarray(x).astype(np.float32)

    xs = np.unique(np.quantile(x, np.linspace(0.01, 0.99, 99)))

    best = None

    for t in xs:
        if lower_means_conflict:
            pred = (x <= t).astype(int)
        else:
            pred = (x >= t).astype(int)

        acc = accuracy_score(y, pred)
        f1 = f1_score(y, pred, zero_division=0)
        prec = precision_score(y, pred, zero_division=0)
        rec = recall_score(y, pred, zero_division=0)

        score = f1

        if best is None or score > best["f1"]:
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

def run_1c_analysis(feature_rows, generation_rows):
    import pandas as pd

    df_feat = pd.DataFrame(feature_rows)
    df_gen = pd.DataFrame(generation_rows)

    # Merge generation labels.
    df = df_feat.merge(
        df_gen[["condition", "idx", "generated", "first_word", "is_clean", "is_conflict", "is_other"]],
        on=["condition", "idx"],
        how="left",
    )

    # Keep binary C/E samples only.
    df_bin = df[(df["is_clean"] == True) | (df["is_conflict"] == True)].copy()
    df_bin["y_conflict"] = df_bin["is_conflict"].astype(int)

    print("\n\n============================================================")
    print("Constraint-Audit-1c Dataset Summary")
    print("============================================================\n")

    print("Total rows:", len(df))
    print("Binary C/E rows:", len(df_bin))

    print("\nGeneration by condition:")
    for cond in conditions:
        sub = df[df["condition"] == cond]
        if len(sub) == 0:
            continue
        print(
            f"{cond:<28} "
            f"N={len(sub):<4} "
            f"Gen_C={sub['is_clean'].mean():.4f} "
            f"Gen_E={sub['is_conflict'].mean():.4f} "
            f"Other={sub['is_other'].mean():.4f}"
        )

    # Candidate features.
    features = [
        "boundary_R_mean",
        "boundary_R_min",
        "boundary_R_L20",
        "boundary_R_L21",
        "boundary_R_L22",
        "boundary_entropy_mean",
        "boundary_entropy_max",
        "boundary_distance_mean",

        "late_R_mean",
        "late_R_min",
        "late_entropy_mean",
        "late_entropy_max",

        "final_R",
        "final_p_clean",
        "final_entropy",
        "final_distance",

        "min_R_20_27",
        "mean_R_20_27",
        "R_slope_20_to_27",
        "R_drop_boundary_to_final",
        "R_drop_late_to_final",
        "num_negative_layers_20_27",
    ]

    if COMPUTE_TOPK_FEATURES:
        features += [
            "boundary_spread_mean",
            "late_spread_mean",
            "final_spread",
            "boundary_topk_gap_mean",
            "late_topk_gap_mean",
            "final_topk_gap",
        ]

    feature_rows_out = []

    y = df_bin["y_conflict"].values.astype(int)

    print("\n\n============================================================")
    print("Feature AUC and Best Thresholds")
    print("============================================================\n")

    print(
        f"{'feature':<32}"
        f"{'auc_conflict_high':<18}"
        f"{'auc_conflict_low':<18}"
        f"{'best_dir':<10}"
        f"{'thr':<12}"
        f"{'acc':<10}"
        f"{'f1':<10}"
        f"{'prec':<10}"
        f"{'rec':<10}"
    )

    for feat in features:
        x = sanitize_array(df_bin[feat].values)

        # score high means conflict
        auc_high = safe_auc(y, x)
        # score low means conflict
        auc_low = safe_auc(y, -x)

        # Choose direction by AUC.
        lower_means_conflict = auc_low >= auc_high
        best = threshold_scan(y, x, lower_means_conflict=lower_means_conflict)

        row = {
            "feature": feat,
            "auc_conflict_high": auc_high,
            "auc_conflict_low": auc_low,
            "best_direction": "low" if lower_means_conflict else "high",
            **best,
        }

        feature_rows_out.append(row)

        print(
            f"{feat:<32}"
            f"{auc_high:<18.4f}"
            f"{auc_low:<18.4f}"
            f"{row['best_direction']:<10}"
            f"{best['threshold']:<12.4f}"
            f"{best['accuracy']:<10.4f}"
            f"{best['f1']:<10.4f}"
            f"{best['precision']:<10.4f}"
            f"{best['recall']:<10.4f}"
        )

    # Logistic model.
    model_features = [
        "boundary_R_mean",
        "boundary_entropy_mean",
        "late_R_mean",
        "final_R",
        "final_entropy",
        "min_R_20_27",
        "R_drop_boundary_to_final",
        "num_negative_layers_20_27",
    ]

    if COMPUTE_TOPK_FEATURES:
        model_features += [
            "boundary_spread_mean",
            "late_spread_mean",
            "boundary_topk_gap_mean",
            "final_topk_gap",
        ]

    X = sanitize_array(df_bin[model_features].values)
    y = df_bin["y_conflict"].values.astype(int)
    groups = df_bin["condition"].values

    print("\n\n============================================================")
    print("Logistic Prediction")
    print("============================================================\n")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            max_iter=500,
        )),
    ])

    # Stratified random CV.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    probs = np.zeros(len(y), dtype=np.float32)
    preds = np.zeros(len(y), dtype=int)

    for train_idx, test_idx in skf.split(X, y):
        pipe.fit(X[train_idx], y[train_idx])
        p = pipe.predict_proba(X[test_idx])[:, 1]
        probs[test_idx] = p
        preds[test_idx] = (p >= 0.5).astype(int)

    cv_auc = safe_auc(y, probs)
    cv_acc = accuracy_score(y, preds)
    cv_f1 = f1_score(y, preds, zero_division=0)
    cv_prec = precision_score(y, preds, zero_division=0)
    cv_rec = recall_score(y, preds, zero_division=0)

    print("5-fold stratified CV:")
    print(f"  AUC      = {cv_auc:.4f}")
    print(f"  Accuracy = {cv_acc:.4f}")
    print(f"  F1       = {cv_f1:.4f}")
    print(f"  Precision= {cv_prec:.4f}")
    print(f"  Recall   = {cv_rec:.4f}")

    # Leave-one-condition-out diagnostic.
    logo = LeaveOneGroupOut()
    logo_rows = []

    print("\nLeave-one-condition-out:")

    for train_idx, test_idx in logo.split(X, y, groups=groups):
        test_group = groups[test_idx][0]

        if len(np.unique(y[train_idx])) < 2:
            continue

        pipe.fit(X[train_idx], y[train_idx])
        p = pipe.predict_proba(X[test_idx])[:, 1]
        pred = (p >= 0.5).astype(int)

        acc = accuracy_score(y[test_idx], pred)
        f1 = f1_score(y[test_idx], pred, zero_division=0)

        auc = safe_auc(y[test_idx], p)

        row = {
            "heldout_condition": test_group,
            "n": int(len(test_idx)),
            "positive_rate": float(np.mean(y[test_idx])),
            "auc": auc,
            "accuracy": float(acc),
            "f1": float(f1),
            "pred_conflict_rate": float(np.mean(pred)),
            "mean_prob": float(np.mean(p)),
        }

        logo_rows.append(row)

        print(
            f"  {test_group:<28} "
            f"N={row['n']:<4} "
            f"pos={row['positive_rate']:.4f} "
            f"auc={row['auc']} "
            f"acc={row['accuracy']:.4f} "
            f"f1={row['f1']:.4f} "
            f"predE={row['pred_conflict_rate']:.4f} "
            f"pE={row['mean_prob']:.4f}"
        )

    # Fit final model for coefficients.
    pipe.fit(X, y)

    clf = pipe.named_steps["clf"]
    coefs = clf.coef_[0]

    coef_rows = []
    for feat, coef in sorted(zip(model_features, coefs), key=lambda x: abs(x[1]), reverse=True):
        coef_rows.append({"feature": feat, "coef": float(coef)})
        print(f"coef {feat:<32} {coef:+.4f}")

    # Critical zone around final_R.
    print("\n\n============================================================")
    print("Critical Zone Estimate")
    print("============================================================\n")

    critical_features = ["final_R", "late_R_mean", "boundary_R_mean", "final_entropy"]

    critical_rows = []

    for feat in critical_features:
        if feat not in df_bin.columns:
            continue

        x = sanitize_array(df_bin[feat].values)
        y = df_bin["y_conflict"].values.astype(int)

        auc_high = safe_auc(y, x)
        auc_low = safe_auc(y, -x)

        lower = auc_low >= auc_high
        best = threshold_scan(y, x, lower_means_conflict=lower)

        row = {
            "feature": feat,
            "direction": "low_conflict" if lower else "high_conflict",
            **best,
        }

        critical_rows.append(row)

        print(
            f"{feat:<20} "
            f"direction={row['direction']:<14} "
            f"threshold={row['threshold']:+.4f} "
            f"acc={row['accuracy']:.4f} "
            f"f1={row['f1']:.4f} "
            f"precision={row['precision']:.4f} "
            f"recall={row['recall']:.4f}"
        )

    return df, df_bin, feature_rows_out, logo_rows, coef_rows, critical_rows

# ============================================================
# MAIN
# ============================================================

all_feature_rows = []
all_generation_rows = []

for cond in conditions:
    print("\n\n############################################################")
    print(f"Condition: {cond}")
    print(f"epsilon={epsilon_by_condition[cond]}, status={status_by_condition[cond]}")
    print("############################################################")

    gen_rows = generation_check_condition(cond)
    all_generation_rows.extend(gen_rows)

    feat_rows = extract_constraint_features_for_condition(cond)
    all_feature_rows.extend(feat_rows)

# Analysis and save.
df_all, df_bin, feature_auc_rows, logo_rows, coef_rows, critical_rows = run_1c_analysis(
    all_feature_rows,
    all_generation_rows,
)

# ============================================================
# SAVE OUTPUTS
# ============================================================

try:
    import pandas as pd

    feature_path = os.path.join(SAVE_DIR, "constraint_audit1c_sample_features.csv")
    df_all.to_csv(feature_path, index=False)

    binary_path = os.path.join(SAVE_DIR, "constraint_audit1c_binary_dataset.csv")
    df_bin.to_csv(binary_path, index=False)

    gen_path = os.path.join(SAVE_DIR, "constraint_audit1c_generation.csv")
    pd.DataFrame(all_generation_rows).to_csv(gen_path, index=False)

    auc_path = os.path.join(SAVE_DIR, "constraint_audit1c_feature_auc_thresholds.csv")
    pd.DataFrame(feature_auc_rows).to_csv(auc_path, index=False)

    logo_path = os.path.join(SAVE_DIR, "constraint_audit1c_leave_condition_out.csv")
    pd.DataFrame(logo_rows).to_csv(logo_path, index=False)

    coef_path = os.path.join(SAVE_DIR, "constraint_audit1c_logistic_coefficients.csv")
    pd.DataFrame(coef_rows).to_csv(coef_path, index=False)

    crit_path = os.path.join(SAVE_DIR, "constraint_audit1c_critical_thresholds.csv")
    pd.DataFrame(critical_rows).to_csv(crit_path, index=False)

    prompt_path = os.path.join(SAVE_DIR, "constraint_audit1c_prompts.txt")
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
    print(" ", feature_path)
    print(" ", binary_path)
    print(" ", gen_path)
    print(" ", auc_path)
    print(" ", logo_path)
    print(" ", coef_path)
    print(" ", crit_path)
    print(" ", prompt_path)

except Exception as e:
    print("\nCould not save outputs:", repr(e))

# ============================================================
# INTERPRETATION GUIDE
# ============================================================

print("\n\n============================================================")
print("Constraint-Audit-1c Interpretation Guide")
print("============================================================\n")

print("Strong result if:")
print("  1. final_R or late_R_mean has high AUC for Gen_E.")
print("  2. best threshold is near final_R ≈ 0.")
print("  3. ambiguous_branch lies near the threshold.")
print("  4. closure_negation lies clearly beyond the conflict side.")
print("  5. logistic model predicts Gen_E with high CV performance.")
print()
print("Expected critical picture:")
print("  final_R > positive threshold:")
print("    clean basin, Gen_C likely.")
print("  final_R near 0:")
print("    critical region, ambiguous outputs.")
print("  final_R < 0:")
print("    conflict basin, Gen_E likely.")
print()
print("Important caveat:")
print("  This is a task-local critical threshold, not a universal model constant.")
print("  To generalize it, repeat with different label pools, relation templates, and models.")
print()
print("Done.")