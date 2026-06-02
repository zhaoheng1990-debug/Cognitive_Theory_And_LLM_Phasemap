# ============================================================
# Constraint-Audit-1b
# Logit-margin Constraint State Test
#
# Fix from Constraint-Audit-1:
#   Old R_l:
#       cos(TopK center, clean_token) - cos(TopK center, conflict_token)
#
#   Problem:
#       clean prompts had negative R_bias despite 100% clean generation.
#
#   New R_l:
#       logit_l(clean_token) - logit_l(conflict_token)
#
# Constraint state:
#   C_l = (R_l, B_l, F_l)
#
#   R_l:
#       answer logit margin
#
#   B_l:
#       decision boundary distance
#       binary entropy
#       boundary instability
#
#   F_l:
#       TopK spread
#       clean/conflict TopK membership
#       TopK membership gap
#
# Tests:
#   1. Does clean have positive R in late/final layers?
#   2. Does closure-negation flip R negative?
#   3. Does C_l -> C_{l+1} have stable layerwise dynamics?
#   4. Does constraint-operator path recover L20-L22 better than 1a?
#   5. Does R/B/F predict generation-level C/E flip?
# ============================================================

import os
import re
import gc
import random
import warnings
from collections import Counter

import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

TOPK = 5000
CENTER_HIDDEN_FOR_TOPK = True

BOUNDARY_LAYERS = [20, 21, 22]
LATE_LAYERS = [23, 24, 25, 26, 27]
EXPECTED_TAU_ZONE = {20, 21, 22}

TEST_SIZE = 0.25
RIDGE_ALPHA = 1e-2

K_LIST = [2, 3, 4, 5]
OUTPUT_WINDOW = 5
TAU_SEARCH_START_RATIO = 0.45
MIN_OUTPUT_RUN = 2

DO_GENERATION_CHECK = True
GEN_BATCH_SIZE = 4
GEN_MAX_NEW_TOKENS = 8

SAVE_DIR = "./constraint_audit1b_outputs"
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
# LABEL SELECTION
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

clean_token_vecs = W_norm[clean_token_ids]
conflict_token_vecs = W_norm[conflict_token_ids]

# ============================================================
# HIDDEN CAPTURE
# ============================================================

def capture_hidden_states(texts, condition_name=""):
    all_layer_states = [[] for _ in range(num_layers)]

    print(f"\nCapturing hidden states for condition={condition_name} ...")

    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[start:start + BATCH_SIZE]

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

            for l in range(num_layers):
                h = hidden_states[l]
                picked = h[
                    torch.arange(h.shape[0], device=h.device),
                    last_idx,
                    :
                ]
                all_layer_states[l].append(picked.detach().float().cpu())

            if (start // BATCH_SIZE) % 10 == 0:
                print(f"  captured {min(start + BATCH_SIZE, len(texts))}/{len(texts)}")

    H_raws = []
    for l in range(num_layers):
        H = torch.cat(all_layer_states[l], dim=0).numpy().astype(np.float32)
        H_raws.append(H)

    print(f"Done condition={condition_name}. Shape L00={H_raws[0].shape}")
    return H_raws

# ============================================================
# TOPK + CONSTRAINT STATE
# ============================================================

def compute_topk_ids_and_center(H, k):
    logits = H @ W_np.T
    ids = np.argpartition(logits, -k, axis=1)[:, -k:]

    vals = np.take_along_axis(logits, ids, axis=1)
    order = np.argsort(vals, axis=1)[:, ::-1]
    ids = np.take_along_axis(ids, order, axis=1).astype(np.int32)

    N = ids.shape[0]
    Ck = np.zeros((N, d_model), dtype=np.float32)
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

        Ck[s:e] = C.astype(np.float32)
        spread[s:e] = spr.astype(np.float32)

    return ids, Ck, spread

def membership_rate(ids, token_ids):
    out = np.zeros(ids.shape[0], dtype=np.float32)

    for i in range(ids.shape[0]):
        out[i] = 1.0 if token_ids[i] in set(ids[i].tolist()) else 0.0

    return out

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
    p = np.clip(p, 1e-8, 1.0 - 1e-8)
    q = 1.0 - p
    h = -(p * np.log(p) + q * np.log(q)) / np.log(2.0)
    return h.astype(np.float32)

def build_constraint_states(H_raws, condition_name=""):
    """
    Per sample, per layer constraint vector C_l:

    Core dimensions:
      0  R_margin              = logit(clean) - logit(conflict)
      1  R_p_clean             = sigmoid(R_margin)
      2  R_p_conflict          = 1 - p_clean

      3  B_distance            = abs(R_margin)
      4  B_entropy             = binary entropy over clean/conflict
      5  B_instability         = entropy, repeated as explicit boundary variable

      6  F_spread              = TopK neighborhood spread
      7  F_clean_in_topk
      8  F_conflict_in_topk
      9  F_topk_gap

      10 VIM_center_bias       = cos(Ck, clean_w) - cos(Ck, conflict_w)
                                  diagnostic only, not core R
    """

    print(f"\nBuilding logit-margin constraint states for {condition_name} ...")

    C_states = []
    layer_summaries = []

    for l, H_raw in enumerate(H_raws):
        # ---------- R from raw layer logits ----------
        logits = H_raw @ W_np.T

        clean_logits = logits[np.arange(N_GRAPHS), clean_token_ids]
        conflict_logits = logits[np.arange(N_GRAPHS), conflict_token_ids]

        R_margin = clean_logits - conflict_logits
        p_clean = stable_sigmoid(R_margin)
        p_conflict = 1.0 - p_clean

        B_distance = np.abs(R_margin).astype(np.float32)
        B_entropy = binary_entropy_from_p(p_clean)
        B_instability = B_entropy.copy()

        # ---------- F from TopK/VIM neighborhood ----------
        if CENTER_HIDDEN_FOR_TOPK:
            H_for_topk = H_raw - H_raw.mean(axis=0, keepdims=True)
        else:
            H_for_topk = H_raw

        ids, Ck, spread = compute_topk_ids_and_center(H_for_topk, TOPK)

        clean_in = membership_rate(ids, clean_token_ids)
        conflict_in = membership_rate(ids, conflict_token_ids)
        topk_gap = clean_in - conflict_in

        clean_cos = np.sum(Ck * clean_token_vecs, axis=1)
        conflict_cos = np.sum(Ck * conflict_token_vecs, axis=1)
        vim_center_bias = clean_cos - conflict_cos

        C_l = np.stack([
            R_margin.astype(np.float32),
            p_clean.astype(np.float32),
            p_conflict.astype(np.float32),

            B_distance.astype(np.float32),
            B_entropy.astype(np.float32),
            B_instability.astype(np.float32),

            spread.astype(np.float32),
            clean_in.astype(np.float32),
            conflict_in.astype(np.float32),
            topk_gap.astype(np.float32),

            vim_center_bias.astype(np.float32),
        ], axis=1).astype(np.float32)

        C_states.append(C_l)

        summary = {
            "condition": condition_name,
            "epsilon": epsilon_by_condition[condition_name],
            "status": status_by_condition[condition_name],
            "layer": l,

            "R_margin_mean": float(np.mean(R_margin)),
            "R_margin_median": float(np.median(R_margin)),
            "R_clean_win_rate": float(np.mean(R_margin > 0)),
            "R_conflict_win_rate": float(np.mean(R_margin < 0)),
            "R_p_clean_mean": float(np.mean(p_clean)),
            "R_p_conflict_mean": float(np.mean(p_conflict)),

            "B_distance_mean": float(np.mean(B_distance)),
            "B_entropy_mean": float(np.mean(B_entropy)),
            "B_instability_mean": float(np.mean(B_instability)),

            "F_spread_mean": float(np.mean(spread)),
            "F_spread_std": float(np.std(spread)),
            "F_clean_in_topk": float(np.mean(clean_in)),
            "F_conflict_in_topk": float(np.mean(conflict_in)),
            "F_topk_gap": float(np.mean(topk_gap)),

            "VIM_center_bias_mean": float(np.mean(vim_center_bias)),
            "VIM_center_bias_clean_win": float(np.mean(vim_center_bias > 0)),
        }

        layer_summaries.append(summary)

        print(
            f"L{l:02d} "
            f"R={summary['R_margin_mean']:+.4f} "
            f"Cwin={summary['R_clean_win_rate']:.3f} "
            f"Ewin={summary['R_conflict_win_rate']:.3f} "
            f"H2={summary['B_entropy_mean']:.3f} "
            f"TopKGap={summary['F_topk_gap']:+.3f} "
            f"Spread={summary['F_spread_mean']:.5f} "
            f"VIMbias={summary['VIM_center_bias_mean']:+.5f}"
        )

    return C_states, layer_summaries

# ============================================================
# CONSTRAINT OPERATOR FIT
# ============================================================

def r2_score_global(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))

def mean_row_cosine(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return float(np.mean(np.sum(A * B, axis=1)))

def sanitize_array(X, clip=1e6):
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=clip, neginf=-clip)
    X = np.clip(X, -clip, clip)
    return X.astype(np.float32)


def fit_ridge_operator(X_train, Y_train, alpha=1e-2):
    """
    Numerically safer ridge fit:
        Y ≈ X B + intercept

    Uses feature centering + scaling to prevent raw logit-margin dimensions
    from dominating small variables such as entropy / spread / TopK membership.
    """

    X_train = sanitize_array(X_train)
    Y_train = sanitize_array(Y_train)

    mx = X_train.mean(axis=0, keepdims=True)
    my = Y_train.mean(axis=0, keepdims=True)

    sx = X_train.std(axis=0, keepdims=True) + 1e-6
    sy = Y_train.std(axis=0, keepdims=True) + 1e-6

    Xc = (X_train - mx) / sx
    Yc = (Y_train - my) / sy

    D = Xc.shape[1]

    A = Xc.T @ Xc + alpha * np.eye(D, dtype=np.float32)
    RHS = Xc.T @ Yc

    try:
        B_scaled = np.linalg.solve(A, RHS)
    except np.linalg.LinAlgError:
        B_scaled = np.linalg.lstsq(A, RHS, rcond=1e-4)[0]

    B_scaled = sanitize_array(B_scaled)

    # Prediction will be done in scaled space, so we store sx/sy too.
    return {
        "B_scaled": B_scaled.astype(np.float32),
        "mx": mx.astype(np.float32),
        "my": my.astype(np.float32),
        "sx": sx.astype(np.float32),
        "sy": sy.astype(np.float32),
    }

def predict_operator(X, op):
    X = sanitize_array(X)

    B_scaled = op["B_scaled"]
    mx = op["mx"]
    my = op["my"]
    sx = op["sx"]
    sy = op["sy"]

    Xc = (X - mx) / sx
    Yc_hat = Xc @ B_scaled
    Y_hat = Yc_hat * sy + my

    return sanitize_array(Y_hat)

def safe_svd_values(M):
    M = sanitize_array(M)

    try:
        S = np.linalg.svd(M, compute_uv=False)
        S = sanitize_array(S)
        return S.astype(np.float32)
    except np.linalg.LinAlgError:
        # Fallback: eigenvalues of B^T B
        G = M.T @ M
        G = sanitize_array(G)

        try:
            eigvals = np.linalg.eigvalsh(G)
            eigvals = np.maximum(eigvals, 0.0)
            S = np.sqrt(eigvals)[::-1]
            S = sanitize_array(S)
            return S.astype(np.float32)
        except Exception:
            return np.zeros(min(M.shape), dtype=np.float32)


def fit_constraint_operators(C_states, train_idx, test_idx):
    operators = []
    metrics = []

    for l in range(num_layers - 1):
        X = sanitize_array(C_states[l])
        Y = sanitize_array(C_states[l + 1])

        X_train = X[train_idx]
        Y_train = Y[train_idx]
        X_test = X[test_idx]
        Y_test = Y[test_idx]

        op = fit_ridge_operator(X_train, Y_train, alpha=RIDGE_ALPHA)

        Yhat_train = predict_operator(X_train, op)
        Yhat_test = predict_operator(X_test, op)

        train_r2 = r2_score_global(Y_train, Yhat_train)
        test_r2 = r2_score_global(Y_test, Yhat_test)
        train_cos = mean_row_cosine(Y_train, Yhat_train)
        test_cos = mean_row_cosine(Y_test, Yhat_test)

        B_scaled = op["B_scaled"]
        S = safe_svd_values(B_scaled)

        operators.append({
            "layer": l,
            "B": B_scaled,
            "op": op,
            "singular": S.astype(np.float32),
        })

        metrics.append({
            "layer": l,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_cos": train_cos,
            "test_cos": test_cos,
            "s1": float(S[0]) if len(S) > 0 else 0.0,
            "s2": float(S[1]) if len(S) > 1 else 0.0,
            "s_mean": float(np.mean(S)) if len(S) > 0 else 0.0,
            "s_std": float(np.std(S)) if len(S) > 0 else 0.0,
            "fro": float(np.linalg.norm(B_scaled, ord="fro")),
        })

    return operators, metrics

def build_operator_signatures(operators):
    sigs = []

    for op in operators:
        B = sanitize_array(op["B"])
        S = sanitize_array(op["singular"])

        flat = B.reshape(-1)

        stats = np.array([
            np.mean(S),
            np.std(S),
            np.max(S),
            np.min(S),
            np.linalg.norm(B, ord="fro"),
            np.trace(B),
            np.linalg.cond(B + 1e-3 * np.eye(B.shape[0], dtype=np.float32)),
        ], dtype=np.float32)

        stats = sanitize_array(stats)

        sig = np.concatenate([flat, S.astype(np.float32), stats], axis=0)
        sig = sanitize_array(sig)
        sigs.append(sig)

    return np.stack(sigs).astype(np.float32)

# ============================================================
# PATH HELPERS
# ============================================================

def remap_by_first_appearance(labels):
    mapping = {}
    next_id = 0
    out = []

    for x in labels:
        x = int(x)
        if x not in mapping:
            mapping[x] = next_id
            next_id += 1
        out.append(mapping[x])

    return np.array(out), mapping

def apply_mapping(raw_labels, mapping):
    return np.array([mapping[int(x)] for x in raw_labels], dtype=np.int32)

def format_path(labels):
    return " ".join([f"O{int(x)}" for x in labels])

def estimate_tau(labels, output_window=5, start_ratio=0.45, min_run=2):
    labels = np.asarray(labels)

    final_window = labels[-output_window:]
    output_cluster = Counter(final_window).most_common(1)[0][0]

    start_search = int(len(labels) * start_ratio)
    tau = None

    for i in range(start_search, len(labels)):
        if labels[i] == output_cluster:
            run = labels[i:i + min_run]
            if len(run) >= min_run and np.all(run == output_cluster):
                tau = i
                break

    return tau, int(output_cluster)

def path_agreement(a, b):
    return float(np.mean(np.asarray(a) == np.asarray(b)))

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

            prompt_lens = inputs["attention_mask"].sum(dim=1).detach().cpu().numpy().tolist()

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

                gen_ids = out[bi, prompt_lens[bi]:]
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
                })

            if (start // GEN_BATCH_SIZE) % 10 == 0:
                print(f"  generated {min(start + GEN_BATCH_SIZE, len(prompts))}/{len(prompts)}")

            del inputs, out
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    clean_rate = float(np.mean([r["is_clean"] for r in rows]))
    conflict_rate = float(np.mean([r["is_conflict"] for r in rows]))

    print(f"Generation summary {cond}: clean={clean_rate:.4f}, conflict={conflict_rate:.4f}")

    return rows

# ============================================================
# SUMMARY HELPERS
# ============================================================

def mean_key(rows, key):
    return float(np.mean([r[key] for r in rows]))

def summarize_condition(cond, layer_rows, metrics, gen_rows, best_k4_row=None):
    boundary = [r for r in layer_rows if r["layer"] in BOUNDARY_LAYERS]
    late = [r for r in layer_rows if r["layer"] in LATE_LAYERS]
    final = layer_rows[-1]

    gen_clean_rate = float(np.mean([r["is_clean"] for r in gen_rows])) if gen_rows else np.nan
    gen_conflict_rate = float(np.mean([r["is_conflict"] for r in gen_rows])) if gen_rows else np.nan

    summary = {
        "condition": cond,
        "epsilon": epsilon_by_condition[cond],
        "status": status_by_condition[cond],

        "boundary_R_margin": mean_key(boundary, "R_margin_mean"),
        "boundary_R_clean_win": mean_key(boundary, "R_clean_win_rate"),
        "boundary_R_conflict_win": mean_key(boundary, "R_conflict_win_rate"),
        "boundary_p_clean": mean_key(boundary, "R_p_clean_mean"),
        "boundary_entropy": mean_key(boundary, "B_entropy_mean"),
        "boundary_distance": mean_key(boundary, "B_distance_mean"),

        "boundary_F_spread": mean_key(boundary, "F_spread_mean"),
        "boundary_F_topk_gap": mean_key(boundary, "F_topk_gap"),
        "boundary_VIM_bias": mean_key(boundary, "VIM_center_bias_mean"),

        "late_R_margin": mean_key(late, "R_margin_mean"),
        "late_R_clean_win": mean_key(late, "R_clean_win_rate"),
        "late_R_conflict_win": mean_key(late, "R_conflict_win_rate"),
        "late_p_clean": mean_key(late, "R_p_clean_mean"),
        "late_entropy": mean_key(late, "B_entropy_mean"),
        "late_distance": mean_key(late, "B_distance_mean"),
        "late_F_spread": mean_key(late, "F_spread_mean"),

        "final_R_margin": final["R_margin_mean"],
        "final_R_clean_win": final["R_clean_win_rate"],
        "final_R_conflict_win": final["R_conflict_win_rate"],
        "final_p_clean": final["R_p_clean_mean"],
        "final_entropy": final["B_entropy_mean"],
        "final_distance": final["B_distance_mean"],
        "final_F_spread": final["F_spread_mean"],

        "operator_test_r2": float(np.mean([m["test_r2"] for m in metrics])),
        "operator_test_cos": float(np.mean([m["test_cos"] for m in metrics])),

        "gen_clean_rate": gen_clean_rate,
        "gen_conflict_rate": gen_conflict_rate,
    }

    if best_k4_row is not None:
        summary["k4_path_agreement"] = best_k4_row["path_agreement"]
        summary["k4_tau"] = best_k4_row["tau"]
        summary["k4_tau_shift"] = best_k4_row["tau_shift"]

    return summary

# ============================================================
# MAIN
# ============================================================

indices = np.arange(N_GRAPHS)
train_idx, test_idx = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    shuffle=True,
)

all_layer_rows = []
all_operator_metric_rows = []
all_path_rows = []
all_generation_rows = []
condition_summaries = []

sigs_by_condition = {}
metrics_by_condition = {}
clean_library = {}

# ---------------- Clean baseline ----------------

print("\n\n============================================================")
print("Constraint-Audit-1b Step 1: Clean baseline")
print("============================================================\n")

H_clean = capture_hidden_states(texts_by_condition["clean"], "clean")
C_clean, clean_layer_rows = build_constraint_states(H_clean, "clean")

clean_ops, clean_metrics = fit_constraint_operators(C_clean, train_idx, test_idx)
clean_sigs = build_operator_signatures(clean_ops)

sigs_by_condition["clean"] = clean_sigs
metrics_by_condition["clean"] = clean_metrics
all_layer_rows.extend(clean_layer_rows)

for m in clean_metrics:
    row = dict(m)
    row["condition"] = "clean"
    row["epsilon"] = 0.0
    row["status"] = "clean"
    all_operator_metric_rows.append(row)

clean_scaler = StandardScaler()
clean_X = clean_scaler.fit_transform(clean_sigs)

print("\nClean logit-margin constraint operator fit:")
print(f"  mean TestR2  = {np.mean([m['test_r2'] for m in clean_metrics]):.4f}")
print(f"  mean TestCos = {np.mean([m['test_cos'] for m in clean_metrics]):.4f}")

print("\nClean constraint operator libraries:")

for k in K_LIST:
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=50)
    raw = km.fit_predict(clean_X)
    labels, raw_to_ordered = remap_by_first_appearance(raw)

    sil = silhouette_score(clean_X, raw) if len(set(raw)) > 1 else -1.0
    tau, out_c = estimate_tau(
        labels,
        output_window=OUTPUT_WINDOW,
        start_ratio=TAU_SEARCH_START_RATIO,
        min_run=MIN_OUTPUT_RUN,
    )

    clean_library[k] = {
        "model": km,
        "mapping": raw_to_ordered,
        "labels": labels,
        "silhouette": sil,
        "tau": tau,
        "output_cluster": out_c,
    }

    print(f"\nK={k}")
    print(f"  silhouette: {sil:.4f}")
    print(f"  path      : {format_path(labels)}")
    print(f"  tau       : {tau}")

clean_gen_rows = generation_check_condition("clean")
all_generation_rows.extend(clean_gen_rows)

clean_summary = summarize_condition(
    "clean",
    clean_layer_rows,
    clean_metrics,
    clean_gen_rows,
    best_k4_row={
        "path_agreement": 1.0,
        "tau": clean_library[4]["tau"] if 4 in clean_library else None,
        "tau_shift": 0,
    }
)
condition_summaries.append(clean_summary)

del H_clean, C_clean, clean_ops
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ---------------- Other conditions ----------------

print("\n\n============================================================")
print("Constraint-Audit-1b Step 2: Conditions")
print("============================================================\n")

for cond in conditions:
    if cond == "clean":
        continue

    print("\n\n############################################################")
    print(f"Condition: {cond}")
    print(f"epsilon={epsilon_by_condition[cond]}, status={status_by_condition[cond]}")
    print("############################################################\n")

    H = capture_hidden_states(texts_by_condition[cond], cond)
    C_states, layer_rows = build_constraint_states(H, cond)

    ops, metrics = fit_constraint_operators(C_states, train_idx, test_idx)
    sigs = build_operator_signatures(ops)

    sigs_by_condition[cond] = sigs
    metrics_by_condition[cond] = metrics
    all_layer_rows.extend(layer_rows)

    for m in metrics:
        row = dict(m)
        row["condition"] = cond
        row["epsilon"] = epsilon_by_condition[cond]
        row["status"] = status_by_condition[cond]
        all_operator_metric_rows.append(row)

    mean_test_r2 = float(np.mean([m["test_r2"] for m in metrics]))
    mean_test_cos = float(np.mean([m["test_cos"] for m in metrics]))

    print(f"\nConstraint operator fit: mean TestR2={mean_test_r2:.4f}, mean TestCos={mean_test_cos:.4f}")

    # assign to clean libraries
    X = clean_scaler.transform(sigs)

    best_k4_row = None

    print("\nClean library assignment:")

    for k in K_LIST:
        lib = clean_library[k]
        raw_pred = lib["model"].predict(X)
        labels = apply_mapping(raw_pred, lib["mapping"])

        clean_labels = lib["labels"]
        clean_tau = lib["tau"]

        tau, out_c = estimate_tau(
            labels,
            output_window=OUTPUT_WINDOW,
            start_ratio=TAU_SEARCH_START_RATIO,
            min_run=MIN_OUTPUT_RUN,
        )

        agree = path_agreement(clean_labels, labels)
        tau_shift = None if tau is None or clean_tau is None else tau - clean_tau
        tau_in_zone = tau in EXPECTED_TAU_ZONE if tau is not None else False

        row = {
            "condition": cond,
            "epsilon": epsilon_by_condition[cond],
            "status": status_by_condition[cond],
            "k": k,
            "clean_tau": clean_tau,
            "tau": tau,
            "tau_shift": tau_shift,
            "tau_in_expected_zone": tau_in_zone,
            "path_agreement": agree,
            "clean_path": format_path(clean_labels),
            "condition_path": format_path(labels),
        }

        all_path_rows.append(row)

        print(f"\n  K={k}")
        print(f"    agreement: {agree:.4f}")
        print(f"    clean_tau: {clean_tau}")
        print(f"    tau      : {tau}")
        print(f"    shift    : {tau_shift}")
        print(f"    in zone  : {tau_in_zone}")
        print(f"    path     : {format_path(labels)}")

        if k == 4:
            best_k4_row = row

    gen_rows = generation_check_condition(cond)
    all_generation_rows.extend(gen_rows)

    summary = summarize_condition(
        cond,
        layer_rows,
        metrics,
        gen_rows,
        best_k4_row=best_k4_row,
    )
    condition_summaries.append(summary)

    print("\nCondition logit-margin constraint summary:")
    print(f"  boundary R_margin      : {summary['boundary_R_margin']:+.4f}")
    print(f"  boundary clean win     : {summary['boundary_R_clean_win']:.4f}")
    print(f"  boundary entropy       : {summary['boundary_entropy']:.4f}")
    print(f"  late R_margin          : {summary['late_R_margin']:+.4f}")
    print(f"  final R_margin         : {summary['final_R_margin']:+.4f}")
    print(f"  Gen_C                  : {summary['gen_clean_rate']:.4f}")
    print(f"  Gen_E                  : {summary['gen_conflict_rate']:.4f}")

    del H, C_states, ops
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================
# FINAL TABLE
# ============================================================

print("\n\n============================================================")
print("Constraint-Audit-1b Final Logit-Margin Constraint Curve")
print("============================================================\n")

print(
    f"{'Cond':<28}"
    f"{'eps':<6}"
    f"{'B_R':<10}"
    f"{'B_Cwin':<10}"
    f"{'B_H2':<10}"
    f"{'Late_R':<10}"
    f"{'Final_R':<10}"
    f"{'K4Tau':<8}"
    f"{'K4Shift':<9}"
    f"{'Gen_C':<10}"
    f"{'Gen_E':<10}"
)

for s in sorted(condition_summaries, key=lambda x: x["epsilon"]):
    print(
        f"{s['condition']:<28}"
        f"{s['epsilon']:<6.1f}"
        f"{s['boundary_R_margin']:<10.4f}"
        f"{s['boundary_R_clean_win']:<10.4f}"
        f"{s['boundary_entropy']:<10.4f}"
        f"{s['late_R_margin']:<10.4f}"
        f"{s['final_R_margin']:<10.4f}"
        f"{str(s.get('k4_tau', None)):<8}"
        f"{str(s.get('k4_tau_shift', None)):<9}"
        f"{s['gen_clean_rate']:<10.4f}"
        f"{s['gen_conflict_rate']:<10.4f}"
    )

# ============================================================
# SAVE OUTPUTS
# ============================================================

try:
    import pandas as pd

    df_layers = pd.DataFrame(all_layer_rows)
    layer_path = os.path.join(SAVE_DIR, "constraint_audit1b_layerwise_constraint_states.csv")
    df_layers.to_csv(layer_path, index=False)

    df_ops = pd.DataFrame(all_operator_metric_rows)
    op_path = os.path.join(SAVE_DIR, "constraint_audit1b_operator_metrics.csv")
    df_ops.to_csv(op_path, index=False)

    df_paths = pd.DataFrame(all_path_rows)
    path_path = os.path.join(SAVE_DIR, "constraint_audit1b_path_tau.csv")
    df_paths.to_csv(path_path, index=False)

    df_summary = pd.DataFrame(condition_summaries)
    summary_path = os.path.join(SAVE_DIR, "constraint_audit1b_summary.csv")
    df_summary.to_csv(summary_path, index=False)

    df_gen = pd.DataFrame(all_generation_rows)
    gen_path = os.path.join(SAVE_DIR, "constraint_audit1b_generation_check.csv")
    df_gen.to_csv(gen_path, index=False)

    prompt_path = os.path.join(SAVE_DIR, "constraint_audit1b_prompts.txt")
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
    print(" ", layer_path)
    print(" ", op_path)
    print(" ", path_path)
    print(" ", summary_path)
    print(" ", gen_path)
    print(" ", prompt_path)

except Exception as e:
    print("\nCould not save outputs:", repr(e))

# ============================================================
# INTERPRETATION GUIDE
# ============================================================

print("\n\n============================================================")
print("Constraint-Audit-1b Interpretation Guide")
print("============================================================\n")

print("Primary success pattern:")
print("  clean / weak conditions:")
print("    boundary_R_margin positive")
print("    late_R_margin positive")
print("    final_R_margin positive")
print("    Gen_C high")
print()
print("  ambiguous / closure conflicts:")
print("    boundary_R_margin decreases")
print("    late/final_R_margin approaches zero or turns negative")
print("    Gen_E increases")
print()
print("Key comparison with Constraint-Audit-1:")
print("  If clean R is now positive, then the old TopK-center R definition was confounded.")
print("  If closure_negation still flips late/final R and Gen_E=1,")
print("  then logit-margin C_l is a better constraint-state candidate.")
print()
print("Operator interpretation:")
print("  If C_l operator path recovers L20-L22, this supports a dynamical constraint-state model.")
print("  If path still gives spurious tau, use C_l for collapse diagnostics,")
print("  but do not yet treat its operator library as the true macro phase structure.")
print()
print("Important distinction:")
print("  R_margin is the answer constraint.")
print("  VIM_center_bias is only diagnostic, not the core relation constraint.")
print()
print("Done.")