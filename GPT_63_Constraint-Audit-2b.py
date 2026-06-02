# ============================================================
# Constraint-Audit-2B
# Multi-template Mechanism Generalization
#
# Goal:
#   Constraint-Audit-2A showed mechanism coordinates exist:
#       stable / competition / closure
#
#   But closure failed in LOCO because closure_rewrite had only
#   one condition. 2B fixes this by creating multiple template
#   families per mechanism and evaluating:
#
#       leave-one-template-family-out
#
# Mechanisms:
#   stable_or_preserved
#   competition_conflict
#   closure_rewrite
#
# Main tests:
#   1. Can W20-W25 classify mechanism across held-out templates?
#   2. Can closure_rewrite generalize to unseen closure templates?
#   3. Does mechanism-aware Gen_E prediction improve calibration?
#   4. Does full-data head activate cleanly by mechanism family?
#
# Output:
#   ./constraint_audit2b_outputs/
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

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    brier_score_loss,
    log_loss,
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

RANDOM_SEED = 42

N_GRAPHS = 96
MAX_LEN = 300

BATCH_SIZE = 8
GEN_BATCH_SIZE = 8
GEN_MAX_NEW_TOKENS = 8

TRACK_LAYERS = [20, 21, 22, 23, 24, 25]

SEGMENTS = {
    "ENTRY20_22": [20, 21, 22],
    "BASIN23_25": [23, 24, 25],
    "ALL20_25": [20, 21, 22, 23, 24, 25],
}

MECHANISM_CLASSES = [
    "stable_or_preserved",
    "competition_conflict",
    "closure_rewrite",
]

CAL_SIZE = 0.30
N_GROUP_SPLITS = 5

SAVE_DIR = Path("./constraint_audit2b_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# SEED
# ============================================================

def set_seed(seed):
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
    raise RuntimeError(
        f"TRACK_LAYERS includes L{max(TRACK_LAYERS)}, "
        f"but model only has {num_layers} layers."
    )

W_lm = model.lm_head.weight.detach().float().cpu().numpy().astype(np.float32)

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

    if len(single) < 12:
        raise RuntimeError("Need at least 12 single-token labels.")

    chosen = single[:12]
    print("\nUsing labels:", chosen)
    return chosen

LABEL_POOL = select_single_token_labels()

# ============================================================
# TEMPLATE FAMILIES
# ============================================================

TEMPLATE_FAMILIES = [
    # ---------------- stable / preserved ----------------
    {
        "family": "stable_clean_basic",
        "mechanism": "stable_or_preserved",
        "epsilon": 0.0,
    },
    {
        "family": "stable_paraphrase",
        "mechanism": "stable_or_preserved",
        "epsilon": 0.1,
    },
    {
        "family": "stable_redundant",
        "mechanism": "stable_or_preserved",
        "epsilon": 0.2,
    },
    {
        "family": "stable_irrelevant",
        "mechanism": "stable_or_preserved",
        "epsilon": 0.3,
    },
    {
        "family": "stable_weak_note",
        "mechanism": "stable_or_preserved",
        "epsilon": 0.5,
    },

    # ---------------- competition conflict ----------------
    {
        "family": "competition_ambiguous",
        "mechanism": "competition_conflict",
        "epsilon": 2.0,
    },
    {
        "family": "competition_branch",
        "mechanism": "competition_conflict",
        "epsilon": 2.5,
    },
    {
        "family": "competition_direct",
        "mechanism": "competition_conflict",
        "epsilon": 3.0,
    },
    {
        "family": "competition_source_claim",
        "mechanism": "competition_conflict",
        "epsilon": 3.5,
    },
    {
        "family": "competition_equal_evidence",
        "mechanism": "competition_conflict",
        "epsilon": 4.0,
    },

    # ---------------- closure rewrite ----------------
    {
        "family": "closure_negation",
        "mechanism": "closure_rewrite",
        "epsilon": 5.0,
    },
    {
        "family": "closure_update",
        "mechanism": "closure_rewrite",
        "epsilon": 5.1,
    },
    {
        "family": "closure_override",
        "mechanism": "closure_rewrite",
        "epsilon": 5.2,
    },
    {
        "family": "closure_temporal",
        "mechanism": "closure_rewrite",
        "epsilon": 5.3,
    },
    {
        "family": "closure_authority",
        "mechanism": "closure_rewrite",
        "epsilon": 5.4,
    },
    {
        "family": "closure_exception",
        "mechanism": "closure_rewrite",
        "epsilon": 5.5,
    },
]

FAMILY_ORDER = [x["family"] for x in TEMPLATE_FAMILIES]
FAMILY_META = {x["family"]: x for x in TEMPLATE_FAMILIES}

# ============================================================
# PROMPT CONSTRUCTION
# ============================================================

def make_entity(prefix, i):
    return f"{prefix}{i:03d}"

def make_prompt(
    a,
    b,
    d,
    clean_label,
    conflict_label,
    irrelevant_label,
    family,
):
    option_line = (
        f"Possible location labels for the queried item are: "
        f"{clean_label} or {conflict_label}."
    )

    answer_line = (
        f"Answer with exactly one word: {clean_label} or {conflict_label}."
    )

    if family == "stable_clean_basic":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located at {clean_label}.",
            f"Question: Which location label is {a} located at?",
            answer_line,
            "Answer:",
        ]

    elif family == "stable_paraphrase":
        lines = [
            "Read the following relational record.",
            option_line,
            f"Record 1: {a} is a member of group {b}.",
            f"Record 2: the location label assigned to {b} is {clean_label}.",
            f"Question: Which location label should be assigned to {a}?",
            answer_line,
            "Answer:",
        ]

    elif family == "stable_redundant":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located at {clean_label}.",
            f"Redundant fact: every item belonging to {b} uses the location label {clean_label}.",
            f"Question: Which location label is {a} located at?",
            answer_line,
            "Answer:",
        ]

    elif family == "stable_irrelevant":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located at {clean_label}.",
            f"Irrelevant fact 1: {d} belongs to an unrelated group.",
            f"Irrelevant fact 2: that unrelated group is located at {irrelevant_label}.",
            f"Question: Which location label is {a} located at?",
            answer_line,
            "Answer:",
        ]

    elif family == "stable_weak_note":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Verified fact 1: {a} belongs to {b}.",
            f"Verified fact 2: {b} is located at {clean_label}.",
            f"Unverified note: someone mentioned {conflict_label}, but the verified facts above are authoritative.",
            f"Question: Which location label is {a} located at?",
            answer_line,
            "Answer:",
        ]

    elif family == "competition_ambiguous":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located at {clean_label}.",
            f"Ambiguous fact: {a} may also be associated with {conflict_label}.",
            f"Question: Which location label is {a} located at?",
            answer_line,
            "Answer:",
        ]

    elif family == "competition_branch":
        other_group = f"D{i_from_entity(a):03d}"
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located at {clean_label}.",
            f"Conflicting branch fact 1: {a} also belongs to {other_group}.",
            f"Conflicting branch fact 2: {other_group} is located at {conflict_label}.",
            f"Question: Which location label is {a} located at?",
            answer_line,
            "Answer:",
        ]

    elif family == "competition_direct":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located at {clean_label}.",
            f"Conflicting direct fact: {a} is located at {conflict_label}.",
            f"Question: Which location label is {a} located at?",
            answer_line,
            "Answer:",
        ]

    elif family == "competition_source_claim":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Primary record: {a} belongs to {b}.",
            f"Primary record: {b} is located at {clean_label}.",
            f"Secondary source claim: {a} is located at {conflict_label}.",
            f"Question: Which location label is {a} located at?",
            answer_line,
            "Answer:",
        ]

    elif family == "competition_equal_evidence":
        other_group = f"D{i_from_entity(a):03d}"
        lines = [
            "You are given two records about the same queried item.",
            option_line,
            f"Record A1: {a} belongs to {b}.",
            f"Record A2: {b} is located at {clean_label}.",
            f"Record B1: {a} belongs to {other_group}.",
            f"Record B2: {other_group} is located at {conflict_label}.",
            f"Question: Which location label is {a} located at?",
            answer_line,
            "Answer:",
        ]

    elif family == "closure_negation":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Old fact: {b} is located at {clean_label}.",
            f"Updated record: {b} is located at {conflict_label}, not at {clean_label}.",
            f"Question: Which location label is {a} located at?",
            answer_line,
            "Answer:",
        ]

    elif family == "closure_update":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Earlier record: {b} was located at {clean_label}.",
            f"Latest update: the location label for {b} changed from {clean_label} to {conflict_label}.",
            f"Question: Which location label is {a} located at now?",
            answer_line,
            "Answer:",
        ]

    elif family == "closure_override":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Default record: {b} is located at {clean_label}.",
            f"Override rule for this query: use {conflict_label} as the valid location label for {b}.",
            f"Question: Which location label is {a} located at under the override rule?",
            answer_line,
            "Answer:",
        ]

    elif family == "closure_temporal":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Past record: {b} was located at {clean_label} before.",
            f"Current record: {b} is now located at {conflict_label}.",
            f"Question: Which current location label is {a} located at?",
            answer_line,
            "Answer:",
        ]

    elif family == "closure_authority":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Unverified old note: {b} is located at {clean_label}.",
            f"Verified registry: {b} is located at {conflict_label}.",
            f"Question: According to the verified registry, which location label is {a} located at?",
            answer_line,
            "Answer:",
        ]

    elif family == "closure_exception":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"General rule: items in {b} usually use location label {clean_label}.",
            f"Exception for this case: {b} uses location label {conflict_label}.",
            f"Question: Which location label is {a} located at in this case?",
            answer_line,
            "Answer:",
        ]

    else:
        raise ValueError(f"Unknown family: {family}")

    return "\n".join(lines)

def i_from_entity(a):
    m = re.search(r"(\d+)$", a)
    if m:
        return int(m.group(1))
    return 0

def build_dataset(n_graphs):
    records = []
    texts = []

    n_labels = len(LABEL_POOL)

    for i in range(n_graphs):
        a = make_entity("A", i)
        b = make_entity("B", i)
        d = make_entity("X", i)

        clean_label = LABEL_POOL[(2 * i) % n_labels]
        conflict_label = LABEL_POOL[(2 * i + 1) % n_labels]
        irrelevant_label = LABEL_POOL[(2 * i + 2) % n_labels]

        clean_id = continuation_ids(clean_label)[0]
        conflict_id = continuation_ids(conflict_label)[0]

        for fam_meta in TEMPLATE_FAMILIES:
            family = fam_meta["family"]
            mechanism = fam_meta["mechanism"]
            epsilon = fam_meta["epsilon"]

            prompt = make_prompt(
                a=a,
                b=b,
                d=d,
                clean_label=clean_label,
                conflict_label=conflict_label,
                irrelevant_label=irrelevant_label,
                family=family,
            )

            row = {
                "idx": i,
                "family": family,
                "mechanism": mechanism,
                "epsilon": epsilon,
                "clean_label": clean_label,
                "conflict_label": conflict_label,
                "clean_token_id": clean_id,
                "conflict_token_id": conflict_id,
                "prompt": prompt,
            }

            records.append(row)
            texts.append(prompt)

    return pd.DataFrame(records), texts

df_records, all_texts = build_dataset(N_GRAPHS)

print("\nDataset rows:", len(df_records))
print("Families:", len(FAMILY_ORDER))
print("\nExample prompts:")
for fam in ["stable_clean_basic", "competition_direct", "closure_update"]:
    ex = df_records[df_records["family"] == fam].iloc[0]
    print("\n---", fam, "---")
    print(ex["prompt"])

clean_token_ids_all = df_records["clean_token_id"].values.astype(np.int64)
conflict_token_ids_all = df_records["conflict_token_id"].values.astype(np.int64)

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

def best_threshold_on_calibration(y_cal, p_cal, objective="f1"):
    y_cal = np.asarray(y_cal).astype(int)
    p_cal = sanitize_prob(p_cal)

    thresholds = np.unique(np.quantile(p_cal, np.linspace(0.01, 0.99, 99)))

    if len(thresholds) == 0:
        return 0.5, calc_binary_metrics(y_cal, p_cal, threshold=0.5)

    best_t = 0.5
    best_score = -1.0
    best_metrics = None

    for t in thresholds:
        m = calc_binary_metrics(y_cal, p_cal, threshold=float(t))
        score = m.get(objective, 0.0)

        if score > best_score:
            best_t = float(t)
            best_score = score
            best_metrics = m

    return best_t, best_metrics

def split_core_calibration(y_trainval):
    idx = np.arange(len(y_trainval))
    y_trainval = np.asarray(y_trainval).astype(int)

    try:
        core_idx, cal_idx = train_test_split(
            idx,
            test_size=CAL_SIZE,
            random_state=RANDOM_SEED,
            stratify=y_trainval,
        )
    except Exception:
        core_idx, cal_idx = train_test_split(
            idx,
            test_size=CAL_SIZE,
            random_state=RANDOM_SEED,
            shuffle=True,
        )

    return core_idx, cal_idx

# ============================================================
# GENERATION
# ============================================================

def normalize_first_word(text):
    text = text.strip()
    if not text:
        return ""
    m = re.search(r"[A-Za-z]+", text)
    if not m:
        return ""
    return m.group(0).lower()

def run_generation(df):
    print("\nRunning generation...")

    rows = []
    prompts = df["prompt"].tolist()
    pad_id = tokenizer.pad_token_id

    with torch.no_grad():
        for start in range(0, len(prompts), GEN_BATCH_SIZE):
            end = min(start + GEN_BATCH_SIZE, len(prompts))
            batch_prompts = prompts[start:end]

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

            for bi in range(end - start):
                ridx = start + bi
                rec = df.iloc[ridx]

                gen_ids = out[bi, input_len:]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                first = normalize_first_word(gen_text)

                clean = str(rec["clean_label"]).lower()
                conflict = str(rec["conflict_label"]).lower()

                is_clean = first == clean
                is_conflict = first == conflict

                rows.append({
                    "row_id": ridx,
                    "generated": gen_text,
                    "first_word": first,
                    "is_clean": bool(is_clean),
                    "is_conflict": bool(is_conflict),
                    "is_other": bool((not is_clean) and (not is_conflict)),
                    "y_conflict": int(is_conflict),
                })

            if start % (GEN_BATCH_SIZE * 10) == 0:
                print(f"  generated {end}/{len(prompts)}")

            del inputs, out
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return pd.DataFrame(rows)

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_layer_margin_features(df):
    print("\nExtracting W20-W25 R/H/pC features...")

    prompts = df["prompt"].tolist()
    n = len(df)

    R_by_layer = {l: np.zeros(n, dtype=np.float32) for l in TRACK_LAYERS}
    P_by_layer = {l: np.zeros(n, dtype=np.float32) for l in TRACK_LAYERS}
    H_by_layer = {l: np.zeros(n, dtype=np.float32) for l in TRACK_LAYERS}

    with torch.no_grad():
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

            hidden_states = outputs.hidden_states[1:]

            # left padding -> last real token is final sequence position
            last_idx = torch.full(
                (end - start,),
                inputs["input_ids"].shape[1] - 1,
                dtype=torch.long,
                device=inputs["input_ids"].device,
            )

            batch_indices = np.arange(start, end)
            c_ids = clean_token_ids_all[batch_indices]
            e_ids = conflict_token_ids_all[batch_indices]

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

            if start % (BATCH_SIZE * 10) == 0:
                print(f"  extracted {end}/{n}")

            del outputs, hidden_states, inputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    feature_df = df.copy()

    for l in TRACK_LAYERS:
        feature_df[f"R_L{l}"] = R_by_layer[l]
        feature_df[f"pC_L{l}"] = P_by_layer[l]
        feature_df[f"H_L{l}"] = H_by_layer[l]

    return feature_df

# ============================================================
# SEGMENT FEATURES
# ============================================================

def add_segment_features(df, segment_name, layers):
    out = df.copy()

    r_cols = [f"R_L{l}" for l in layers]
    h_cols = [f"H_L{l}" for l in layers]
    p_cols = [f"pC_L{l}" for l in layers]

    R = out[r_cols].values.astype(np.float32)
    H = out[h_cols].values.astype(np.float32)
    P = out[p_cols].values.astype(np.float32)

    prefix = segment_name

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

    first_neg = []
    for row in R:
        neg = np.where(row < 0)[0]
        if len(neg) == 0:
            first_neg.append(float(len(layers)))
        else:
            first_neg.append(float(neg[0]))

    out[f"{prefix}_first_negative_offset"] = np.array(first_neg, dtype=np.float32)

    return out

def segment_feature_names(segment_name):
    prefix = segment_name

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

def build_feature_table(df):
    out = df.copy()

    for seg_name, layers in SEGMENTS.items():
        out = add_segment_features(out, seg_name, layers)

    raw_cols = []
    for l in TRACK_LAYERS:
        raw_cols.extend([f"R_L{l}", f"H_L{l}", f"pC_L{l}"])

    feature_cols = []
    feature_cols.extend(raw_cols)

    for seg_name in SEGMENTS:
        feature_cols.extend(segment_feature_names(seg_name))

    seen = set()
    unique_cols = []
    for c in feature_cols:
        if c not in seen:
            seen.add(c)
            unique_cols.append(c)

    return out, unique_cols

# ============================================================
# MODELS
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
    def __init__(self, classes, probs=None):
        self.classes_ = np.asarray(classes)
        if probs is None:
            probs = np.ones(len(classes), dtype=np.float64)
        probs = np.asarray(probs, dtype=np.float64)
        probs = probs / (probs.sum() + 1e-12)
        self.probs = probs

    def predict_proba(self, X):
        n = len(X)
        return np.tile(self.probs.reshape(1, -1), (n, 1))

    def predict(self, X):
        p = self.predict_proba(X)
        idx = np.argmax(p, axis=1)
        return self.classes_[idx]

def make_binary_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            max_iter=2000,
        )),
    ])

def make_mechanism_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            max_iter=4000,
        )),
    ])

def fit_binary_or_constant(X, y):
    y = np.asarray(y).astype(int)

    if len(np.unique(y)) < 2:
        return ConstantBinaryModel(float(y.mean()))

    model = make_binary_model()
    model.fit(X, y)
    return model

def fit_mechanism_or_constant(X, y):
    y = np.asarray(y)

    vals, counts = np.unique(y, return_counts=True)

    if len(vals) < 2:
        return ConstantMulticlassModel(vals)

    model = make_mechanism_model()
    model.fit(X, y)
    return model

def aligned_mechanism_proba(model, X, all_classes):
    p_raw = model.predict_proba(X)

    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    else:
        classes = list(model.named_steps["clf"].classes_)

    out = np.zeros((len(X), len(all_classes)), dtype=np.float64)

    for j, cls in enumerate(classes):
        if cls in all_classes:
            idx = all_classes.index(cls)
            out[:, idx] = p_raw[:, j]

    row_sum = out.sum(axis=1, keepdims=True)

    bad = row_sum[:, 0] <= 1e-12
    if bad.any():
        out[bad, :] = 1.0 / len(all_classes)
        row_sum = out.sum(axis=1, keepdims=True)

    out = out / (row_sum + 1e-12)
    return out

# ============================================================
# GROUP CV BY GRAPH
# ============================================================

def evaluate_group_cv(df, feature_cols):
    X = df[feature_cols].values.astype(np.float32)
    y = df["y_conflict"].values.astype(int)
    y_mech = df["mechanism"].values
    groups = df["idx"].values

    gkf = GroupKFold(n_splits=N_GROUP_SPLITS)

    p_base = np.zeros(len(df), dtype=np.float64)
    p_mechaware = np.zeros(len(df), dtype=np.float64)
    mech_pred = np.empty(len(df), dtype=object)

    for fold, (trainval_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_trainval = X[trainval_idx]
        y_trainval = y[trainval_idx]
        m_trainval = y_mech[trainval_idx]

        X_test = X[test_idx]

        core_rel, cal_rel = split_core_calibration(y_trainval)

        X_core = X_trainval[core_rel]
        y_core = y_trainval[core_rel]
        m_core = m_trainval[core_rel]

        X_cal = X_trainval[cal_rel]
        y_cal = y_trainval[cal_rel]

        # Baseline binary
        base_bin = fit_binary_or_constant(X_core, y_core)
        p_base[test_idx] = sanitize_prob(base_bin.predict_proba(X_test)[:, 1])

        # Mechanism head
        mech_model = fit_mechanism_or_constant(X_core, m_core)

        p_mech_core = aligned_mechanism_proba(mech_model, X_core, MECHANISM_CLASSES)
        p_mech_test = aligned_mechanism_proba(mech_model, X_test, MECHANISM_CLASSES)

        mech_pred[test_idx] = np.array(MECHANISM_CLASSES)[np.argmax(p_mech_test, axis=1)]

        # Mechanism-aware binary
        X_core_aug = np.concatenate([X_core, p_mech_core], axis=1)
        X_test_aug = np.concatenate([X_test, p_mech_test], axis=1)

        me_bin = fit_binary_or_constant(X_core_aug, y_core)
        p_mechaware[test_idx] = sanitize_prob(me_bin.predict_proba(X_test_aug)[:, 1])

    base_metrics = calc_binary_metrics(y, p_base, threshold=0.5)
    me_metrics = calc_binary_metrics(y, p_mechaware, threshold=0.5)

    mech_acc = float(accuracy_score(y_mech, mech_pred))
    mech_macro_f1 = float(f1_score(y_mech, mech_pred, average="macro", zero_division=0))

    summary_rows = [
        {"model": "baseline_binary", **base_metrics},
        {"model": "mechanism_aware_binary", **me_metrics},
    ]

    pred_df = df[["idx", "family", "mechanism", "y_conflict"]].copy()
    pred_df["p_base"] = p_base
    pred_df["p_mechaware"] = p_mechaware
    pred_df["mech_pred"] = mech_pred

    return pd.DataFrame(summary_rows), {
        "mechanism_accuracy": mech_acc,
        "mechanism_macro_f1": mech_macro_f1,
    }, pred_df

# ============================================================
# LEAVE-ONE-TEMPLATE-FAMILY-OUT
# ============================================================

def evaluate_leave_family_out(df, feature_cols):
    X = df[feature_cols].values.astype(np.float32)
    y = df["y_conflict"].values.astype(int)
    y_mech = df["mechanism"].values
    families = df["family"].values

    rows = []
    mech_rows = []
    pred_rows = []

    logo = LeaveOneGroupOut()

    for trainval_idx, test_idx in logo.split(X, y, groups=families):
        heldout = families[test_idx][0]

        X_trainval = X[trainval_idx]
        y_trainval = y[trainval_idx]
        m_trainval = y_mech[trainval_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]
        m_test = y_mech[test_idx]

        core_rel, cal_rel = split_core_calibration(y_trainval)

        X_core = X_trainval[core_rel]
        y_core = y_trainval[core_rel]
        m_core = m_trainval[core_rel]

        X_cal = X_trainval[cal_rel]
        y_cal = y_trainval[cal_rel]

        # Baseline binary
        base_bin = fit_binary_or_constant(X_core, y_core)
        p_cal_base = sanitize_prob(base_bin.predict_proba(X_cal)[:, 1])
        p_test_base = sanitize_prob(base_bin.predict_proba(X_test)[:, 1])

        t_base, _ = best_threshold_on_calibration(y_cal, p_cal_base, objective="f1")

        for rule, thr in [("fixed_0.5", 0.5), ("calibrated_threshold", t_base)]:
            metrics = calc_binary_metrics(y_test, p_test_base, threshold=thr)
            row = {
                "heldout_family": heldout,
                "heldout_mechanism": FAMILY_META[heldout]["mechanism"],
                "model": "baseline_binary",
                "decision_rule": rule,
                "n_test": int(len(y_test)),
                "positive_rate": float(y_test.mean()),
                "cal_threshold": float(thr),
            }
            row.update(metrics)
            rows.append(row)

        # Mechanism head
        mech_model = fit_mechanism_or_constant(X_core, m_core)

        p_mech_core = aligned_mechanism_proba(mech_model, X_core, MECHANISM_CLASSES)
        p_mech_cal = aligned_mechanism_proba(mech_model, X_cal, MECHANISM_CLASSES)
        p_mech_test = aligned_mechanism_proba(mech_model, X_test, MECHANISM_CLASSES)

        mech_pred = np.array(MECHANISM_CLASSES)[np.argmax(p_mech_test, axis=1)]

        mech_acc = float(accuracy_score(m_test, mech_pred))
        mech_f1 = float(f1_score(m_test, mech_pred, average="macro", zero_division=0))

        mech_row = {
            "heldout_family": heldout,
            "heldout_mechanism": FAMILY_META[heldout]["mechanism"],
            "mechanism_accuracy": mech_acc,
            "mechanism_macro_f1": mech_f1,
        }

        for j, cls in enumerate(MECHANISM_CLASSES):
            mech_row[f"mean_p_{cls}"] = float(p_mech_test[:, j].mean())

        mech_rows.append(mech_row)

        # Mechanism-aware binary
        X_core_aug = np.concatenate([X_core, p_mech_core], axis=1)
        X_cal_aug = np.concatenate([X_cal, p_mech_cal], axis=1)
        X_test_aug = np.concatenate([X_test, p_mech_test], axis=1)

        me_bin = fit_binary_or_constant(X_core_aug, y_core)

        p_cal_me = sanitize_prob(me_bin.predict_proba(X_cal_aug)[:, 1])
        p_test_me = sanitize_prob(me_bin.predict_proba(X_test_aug)[:, 1])

        t_me, _ = best_threshold_on_calibration(y_cal, p_cal_me, objective="f1")

        for rule, thr in [("fixed_0.5", 0.5), ("calibrated_threshold", t_me)]:
            metrics = calc_binary_metrics(y_test, p_test_me, threshold=thr)
            row = {
                "heldout_family": heldout,
                "heldout_mechanism": FAMILY_META[heldout]["mechanism"],
                "model": "mechanism_aware_binary",
                "decision_rule": rule,
                "n_test": int(len(y_test)),
                "positive_rate": float(y_test.mean()),
                "cal_threshold": float(thr),
            }
            row.update(metrics)
            rows.append(row)

        # Save predictions
        temp = df.iloc[test_idx][["idx", "family", "mechanism", "y_conflict"]].copy()
        temp["p_base"] = p_test_base
        temp["p_mechaware"] = p_test_me
        temp["mech_pred"] = mech_pred

        for j, cls in enumerate(MECHANISM_CLASSES):
            temp[f"p_mech_{cls}"] = p_mech_test[:, j]

        pred_rows.append(temp)

    metrics_df = pd.DataFrame(rows)
    mech_df = pd.DataFrame(mech_rows)
    pred_df = pd.concat(pred_rows, axis=0, ignore_index=True)

    return metrics_df, mech_df, pred_df

# ============================================================
# FULL-DATA HEAD ACTIVATION
# ============================================================

def full_data_mechanism_activations(df, feature_cols):
    X = df[feature_cols].values.astype(np.float32)
    y_mech = df["mechanism"].values

    mech_model = fit_mechanism_or_constant(X, y_mech)
    p_mech = aligned_mechanism_proba(mech_model, X, MECHANISM_CLASSES)

    out = df[["idx", "family", "mechanism", "y_conflict"]].copy()

    for j, cls in enumerate(MECHANISM_CLASSES):
        out[f"p_{cls}"] = p_mech[:, j]

    rows = []
    for fam in FAMILY_ORDER:
        sub = out[out["family"] == fam]
        if len(sub) == 0:
            continue

        row = {
            "family": fam,
            "mechanism": FAMILY_META[fam]["mechanism"],
            "n": int(len(sub)),
            "genE_rate": float(sub["y_conflict"].mean()),
        }

        for cls in MECHANISM_CLASSES:
            row[f"mean_p_{cls}"] = float(sub[f"p_{cls}"].mean())

        rows.append(row)

    return out, pd.DataFrame(rows)

def full_data_binary_coefficients(df, feature_cols):
    X = df[feature_cols].values.astype(np.float32)
    y = df["y_conflict"].values.astype(int)

    model = fit_binary_or_constant(X, y)

    if not isinstance(model, Pipeline):
        return pd.DataFrame()

    clf = model.named_steps["clf"]
    coefs = clf.coef_[0]

    rows = []
    for feat, coef in sorted(zip(feature_cols, coefs), key=lambda x: abs(x[1]), reverse=True):
        rows.append({
            "feature": feat,
            "coef": float(coef),
            "abs_coef": float(abs(coef)),
        })

    return pd.DataFrame(rows)

# ============================================================
# SUMMARY HELPERS
# ============================================================

def summarize_leave_family(metrics_df):
    rows = []

    for keys, sub in metrics_df.groupby(["model", "decision_rule"]):
        model_name, rule = keys

        rows.append({
            "model": model_name,
            "decision_rule": rule,
            "mean_auc": float(np.nanmean(sub["auc"])),
            "mean_accuracy": float(np.nanmean(sub["accuracy"])),
            "mean_f1": float(np.nanmean(sub["f1"])),
            "mean_precision": float(np.nanmean(sub["precision"])),
            "mean_recall": float(np.nanmean(sub["recall"])),
            "mean_brier": float(np.nanmean(sub["brier"])),
            "mean_nll": float(np.nanmean(sub["nll"])),
            "mean_pred_pos_rate": float(np.nanmean(sub["pred_pos_rate"])),
            "std_pred_pos_rate": float(np.nanstd(sub["pred_pos_rate"])),
        })

    return pd.DataFrame(rows)

def summarize_leave_family_by_mechanism(metrics_df):
    rows = []

    for keys, sub in metrics_df.groupby(["heldout_mechanism", "model", "decision_rule"]):
        mech, model_name, rule = keys

        rows.append({
            "heldout_mechanism": mech,
            "model": model_name,
            "decision_rule": rule,
            "mean_auc": float(np.nanmean(sub["auc"])),
            "mean_accuracy": float(np.nanmean(sub["accuracy"])),
            "mean_f1": float(np.nanmean(sub["f1"])),
            "mean_brier": float(np.nanmean(sub["brier"])),
            "mean_pred_pos_rate": float(np.nanmean(sub["pred_pos_rate"])),
            "mean_positive_rate": float(np.nanmean(sub["positive_rate"])),
        })

    return pd.DataFrame(rows)

# ============================================================
# RUN
# ============================================================

gen_df = run_generation(df_records)

df_with_gen = df_records.merge(
    gen_df,
    left_index=True,
    right_on="row_id",
    how="left",
)

df_with_gen = df_with_gen.drop(columns=["row_id"])

# Keep only binary clean/conflict generations.
df_binary = df_with_gen[
    (df_with_gen["is_clean"] == True) | (df_with_gen["is_conflict"] == True)
].copy()

print("\n============================================================")
print("Generation Summary")
print("============================================================\n")

print("Total rows:", len(df_with_gen))
print("Binary rows:", len(df_binary))
print("Other rows:", int(df_with_gen["is_other"].sum()))

print(
    f"{'family':<32}"
    f"{'mech':<24}"
    f"{'N':<6}"
    f"{'GenE':<10}"
    f"{'Other':<10}"
)

for fam in FAMILY_ORDER:
    sub_all = df_with_gen[df_with_gen["family"] == fam]
    sub_bin = df_binary[df_binary["family"] == fam]

    print(
        f"{fam:<32}"
        f"{FAMILY_META[fam]['mechanism']:<24}"
        f"{len(sub_all):<6}"
        f"{sub_bin['y_conflict'].mean() if len(sub_bin) else np.nan:<10.4f}"
        f"{sub_all['is_other'].mean():<10.4f}"
    )

feature_base = extract_layer_margin_features(df_binary)
feature_df, feature_cols = build_feature_table(feature_base)

print("\nFeature count:", len(feature_cols))

# ---------------- GroupKFold by graph ----------------

print("\n\n============================================================")
print("GroupKFold by graph idx")
print("============================================================\n")

group_summary, group_mech_summary, group_pred_df = evaluate_group_cv(feature_df, feature_cols)

print("Binary summary:")
print(
    f"{'model':<28}"
    f"{'AUC':<10}"
    f"{'Acc':<10}"
    f"{'F1':<10}"
    f"{'Brier':<10}"
    f"{'PredE':<10}"
)

for _, r in group_summary.iterrows():
    print(
        f"{r['model']:<28}"
        f"{r['auc']:<10.4f}"
        f"{r['accuracy']:<10.4f}"
        f"{r['f1']:<10.4f}"
        f"{r['brier']:<10.4f}"
        f"{r['pred_pos_rate']:<10.4f}"
    )

print("\nMechanism head:")
print(f"  mechanism_accuracy = {group_mech_summary['mechanism_accuracy']:.4f}")
print(f"  mechanism_macro_f1 = {group_mech_summary['mechanism_macro_f1']:.4f}")

# ---------------- Leave-one-template-family-out ----------------

print("\n\n============================================================")
print("Leave-One-Template-Family-Out")
print("============================================================\n")

lofo_metrics, lofo_mech, lofo_pred = evaluate_leave_family_out(feature_df, feature_cols)

lofo_summary = summarize_leave_family(lofo_metrics)
lofo_by_mech = summarize_leave_family_by_mechanism(lofo_metrics)

print("LOFO binary summary:")
print(
    f"{'model':<28}"
    f"{'rule':<24}"
    f"{'AUC':<10}"
    f"{'Acc':<10}"
    f"{'F1':<10}"
    f"{'Brier':<10}"
    f"{'PredE':<10}"
)

for _, r in lofo_summary.iterrows():
    print(
        f"{r['model']:<28}"
        f"{r['decision_rule']:<24}"
        f"{r['mean_auc']:<10.4f}"
        f"{r['mean_accuracy']:<10.4f}"
        f"{r['mean_f1']:<10.4f}"
        f"{r['mean_brier']:<10.4f}"
        f"{r['mean_pred_pos_rate']:<10.4f}"
    )

print("\nLOFO mechanism diagnostics:")
print(
    f"{'heldout_family':<32}"
    f"{'mech':<24}"
    f"{'acc':<8}"
    f"{'f1':<8}"
    f"{'p_stable':<10}"
    f"{'p_comp':<10}"
    f"{'p_closure':<10}"
)

for _, r in lofo_mech.iterrows():
    print(
        f"{r['heldout_family']:<32}"
        f"{r['heldout_mechanism']:<24}"
        f"{r['mechanism_accuracy']:<8.4f}"
        f"{r['mechanism_macro_f1']:<8.4f}"
        f"{r['mean_p_stable_or_preserved']:<10.4f}"
        f"{r['mean_p_competition_conflict']:<10.4f}"
        f"{r['mean_p_closure_rewrite']:<10.4f}"
    )

print("\nLOFO binary by mechanism:")
print(
    f"{'heldout_mech':<24}"
    f"{'model':<28}"
    f"{'rule':<24}"
    f"{'pos':<8}"
    f"{'predE':<8}"
    f"{'acc':<8}"
    f"{'f1':<8}"
    f"{'brier':<10}"
)

for _, r in lofo_by_mech.iterrows():
    print(
        f"{r['heldout_mechanism']:<24}"
        f"{r['model']:<28}"
        f"{r['decision_rule']:<24}"
        f"{r['mean_positive_rate']:<8.4f}"
        f"{r['mean_pred_pos_rate']:<8.4f}"
        f"{r['mean_accuracy']:<8.4f}"
        f"{r['mean_f1']:<8.4f}"
        f"{r['mean_brier']:<10.4f}"
    )

# ---------------- Full-data mechanism activations ----------------

print("\n\n============================================================")
print("Full-data Mechanism Head Activations")
print("============================================================\n")

head_pred_df, head_activation_df = full_data_mechanism_activations(feature_df, feature_cols)

print(
    f"{'family':<32}"
    f"{'mech':<24}"
    f"{'GenE':<10}"
    f"{'p_stable':<10}"
    f"{'p_comp':<10}"
    f"{'p_closure':<10}"
)

for _, r in head_activation_df.iterrows():
    print(
        f"{r['family']:<32}"
        f"{r['mechanism']:<24}"
        f"{r['genE_rate']:<10.4f}"
        f"{r['mean_p_stable_or_preserved']:<10.4f}"
        f"{r['mean_p_competition_conflict']:<10.4f}"
        f"{r['mean_p_closure_rewrite']:<10.4f}"
    )

coef_df = full_data_binary_coefficients(feature_df, feature_cols)

# ============================================================
# SAVE
# ============================================================

records_path = SAVE_DIR / "constraint_audit2b_records.csv"
generation_path = SAVE_DIR / "constraint_audit2b_generation.csv"
feature_path = SAVE_DIR / "constraint_audit2b_feature_table.csv"

group_summary_path = SAVE_DIR / "constraint_audit2b_groupcv_binary_summary.csv"
group_pred_path = SAVE_DIR / "constraint_audit2b_groupcv_predictions.csv"

lofo_metrics_path = SAVE_DIR / "constraint_audit2b_lofo_metrics.csv"
lofo_summary_path = SAVE_DIR / "constraint_audit2b_lofo_summary.csv"
lofo_by_mech_path = SAVE_DIR / "constraint_audit2b_lofo_by_mechanism.csv"
lofo_mech_path = SAVE_DIR / "constraint_audit2b_lofo_mechanism_diagnostics.csv"
lofo_pred_path = SAVE_DIR / "constraint_audit2b_lofo_predictions.csv"

head_pred_path = SAVE_DIR / "constraint_audit2b_mechanism_head_predictions.csv"
head_activation_path = SAVE_DIR / "constraint_audit2b_mechanism_head_activations.csv"
coef_path = SAVE_DIR / "constraint_audit2b_binary_feature_coefficients.csv"

df_records.to_csv(records_path, index=False)
df_with_gen.to_csv(generation_path, index=False)
feature_df.to_csv(feature_path, index=False)

group_summary.to_csv(group_summary_path, index=False)
group_pred_df.to_csv(group_pred_path, index=False)

lofo_metrics.to_csv(lofo_metrics_path, index=False)
lofo_summary.to_csv(lofo_summary_path, index=False)
lofo_by_mech.to_csv(lofo_by_mech_path, index=False)
lofo_mech.to_csv(lofo_mech_path, index=False)
lofo_pred.to_csv(lofo_pred_path, index=False)

head_pred_df.to_csv(head_pred_path, index=False)
head_activation_df.to_csv(head_activation_path, index=False)
coef_df.to_csv(coef_path, index=False)

prompt_path = SAVE_DIR / "constraint_audit2b_prompts.txt"
with open(prompt_path, "w", encoding="utf-8") as f:
    f.write("LABEL_POOL:\n")
    f.write(str(LABEL_POOL) + "\n\n")

    for fam in FAMILY_ORDER:
        f.write(f"\n\n================ {fam} ================\n")
        ex = df_records[df_records["family"] == fam].iloc[0]
        f.write(f"mechanism: {ex['mechanism']}\n")
        f.write(f"epsilon: {ex['epsilon']}\n\n")
        f.write(ex["prompt"] + "\n")

print("\nSaved outputs:")
print(" ", records_path)
print(" ", generation_path)
print(" ", feature_path)
print(" ", group_summary_path)
print(" ", group_pred_path)
print(" ", lofo_metrics_path)
print(" ", lofo_summary_path)
print(" ", lofo_by_mech_path)
print(" ", lofo_mech_path)
print(" ", lofo_pred_path)
print(" ", head_pred_path)
print(" ", head_activation_path)
print(" ", coef_path)
print(" ", prompt_path)

# ============================================================
# INTERPRETATION GUIDE
# ============================================================

print("\n\n============================================================")
print("Constraint-Audit-2B Interpretation Guide")
print("============================================================\n")

print("Strong positive result if:")
print("  1. GroupKFold mechanism accuracy / macro-F1 remains high.")
print("  2. LOFO mechanism diagnostics correctly classify held-out closure templates.")
print("  3. Full-data activation shows:")
print("       stable templates    -> high p_stable")
print("       competition templates -> high p_comp")
print("       closure templates   -> high p_closure")
print("  4. LOFO closure_rewrite no longer collapses into competition_conflict.")
print()
print("Key comparison against 2A:")
print("  In 2A, held-out closure had no closure examples in training.")
print("  In 2B, each held-out closure family still leaves other closure families in training.")
print("  Therefore LOFO closure performance is now a real closure-template generalization test.")
print()
print("If closure still fails:")
print("  The closure coordinate is not template-general yet.")
print("  Next step: add richer closure support features, e.g. closure-chain scoring.")
print()
print("If mechanism-aware binary does not improve Gen_E:")
print("  That is not fatal if baseline is near ceiling.")
print("  Mechanism coordinates may be explanatory rather than performance-improving.")
print()
print("Done.")