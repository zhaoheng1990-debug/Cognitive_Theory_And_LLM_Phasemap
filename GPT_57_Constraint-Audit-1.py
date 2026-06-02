# ============================================================
# Constraint-Audit-1
# Can TopK-derived constraint states reproduce boundary dynamics?
#
# Goal:
#   Convert TopK neighborhood states into a minimal constraint state:
#
#       C_l = (B_l, R_l, F_l)
#
#   where:
#       R_l = relation/rule bias toward clean vs conflict label
#       B_l = distance to clean/conflict decision boundary
#       F_l = neighborhood freedom / spread
#
#   Then test:
#       1. Does C_l show clean -> conflict degradation?
#       2. Does closure-negation collapse constraint state?
#       3. Does C_l -> C_{l+1} form stable operators?
#       4. Does constraint-operator path recover tau near L20-L22?
#
# This builds directly on Audit-6C.2b single-token labels.
# ============================================================

import os
import re
import gc
import random
import warnings
from collections import Counter, defaultdict

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

K_LIST = [2, 3, 4]
OUTPUT_WINDOW = 5
TAU_SEARCH_START_RATIO = 0.45
MIN_OUTPUT_RUN = 2

DO_GENERATION_CHECK = True
GEN_BATCH_SIZE = 4
GEN_MAX_NEW_TOKENS = 8

SAVE_DIR = "./constraint_audit1_outputs"
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
    for lab in candidate_labels:
        ids = continuation_ids(lab)
        if len(ids) == 1:
            single.append(lab)

    print("\nLabel tokenization audit:")
    for lab in candidate_labels[:20]:
        ids = continuation_ids(lab)
        print(f"  {lab:<10} ids={ids} len={len(ids)}")

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

def build_constraint_states(H_raws, condition_name=""):
    """
    Per sample, per layer constraint state vector:

      R components:
        clean_cos
        conflict_cos
        relation_bias = clean_cos - conflict_cos
        clean_in_topk
        conflict_in_topk
        topk_membership_gap

      B components:
        boundary_distance = abs(relation_bias)
        boundary_instability = 1 - abs(tanh(4 * relation_bias))

      F components:
        spread
        spread_z proxy left as raw spread

    Final C_l dimension = 9.
    """

    print(f"\nBuilding constraint states for {condition_name} ...")

    C_states = []
    layer_summaries = []

    for l, H_raw in enumerate(H_raws):
        if CENTER_HIDDEN_FOR_TOPK:
            H_for_topk = H_raw - H_raw.mean(axis=0, keepdims=True)
        else:
            H_for_topk = H_raw

        ids, Ck, spread = compute_topk_ids_and_center(H_for_topk, TOPK)

        clean_cos = np.sum(Ck * clean_token_vecs, axis=1)
        conflict_cos = np.sum(Ck * conflict_token_vecs, axis=1)

        relation_bias = clean_cos - conflict_cos

        clean_in = membership_rate(ids, clean_token_ids)
        conflict_in = membership_rate(ids, conflict_token_ids)
        topk_gap = clean_in - conflict_in

        boundary_distance = np.abs(relation_bias)
        boundary_instability = 1.0 - np.abs(np.tanh(4.0 * relation_bias))

        # Constraint state vector C_l = [R, B, F]
        C_l = np.stack([
            clean_cos,
            conflict_cos,
            relation_bias,
            clean_in,
            conflict_in,
            topk_gap,
            boundary_distance,
            boundary_instability,
            spread,
        ], axis=1).astype(np.float32)

        C_states.append(C_l)

        summary = {
            "condition": condition_name,
            "epsilon": epsilon_by_condition[condition_name],
            "status": status_by_condition[condition_name],
            "layer": l,

            "R_bias_mean": float(np.mean(relation_bias)),
            "R_bias_median": float(np.median(relation_bias)),
            "R_clean_win_rate": float(np.mean(relation_bias > 0)),
            "R_conflict_win_rate": float(np.mean(relation_bias < 0)),

            "R_clean_in_topk": float(np.mean(clean_in)),
            "R_conflict_in_topk": float(np.mean(conflict_in)),
            "R_topk_gap": float(np.mean(topk_gap)),

            "B_distance_mean": float(np.mean(boundary_distance)),
            "B_instability_mean": float(np.mean(boundary_instability)),

            "F_spread_mean": float(np.mean(spread)),
            "F_spread_std": float(np.std(spread)),
        }

        layer_summaries.append(summary)

        print(
            f"L{l:02d} "
            f"R_bias={summary['R_bias_mean']:+.5f} "
            f"Cwin={summary['R_clean_win_rate']:.3f} "
            f"Ewin={summary['R_conflict_win_rate']:.3f} "
            f"TopKGap={summary['R_topk_gap']:+.3f} "
            f"Bdist={summary['B_distance_mean']:.5f} "
            f"F={summary['F_spread_mean']:.5f}"
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

def fit_ridge_operator(X_train, Y_train, alpha=1e-2):
    D = X_train.shape[1]

    mx = X_train.mean(axis=0, keepdims=True)
    my = Y_train.mean(axis=0, keepdims=True)

    Xc = X_train - mx
    Yc = Y_train - my

    A = Xc.T @ Xc + alpha * np.eye(D, dtype=np.float32)
    B = np.linalg.solve(A, Xc.T @ Yc)

    return B.astype(np.float32), mx.astype(np.float32), my.astype(np.float32)

def predict_operator(X, B, mx, my):
    return (X - mx) @ B + my

def fit_constraint_operators(C_states, train_idx, test_idx):
    operators = []
    metrics = []

    for l in range(num_layers - 1):
        X = C_states[l]
        Y = C_states[l + 1]

        X_train = X[train_idx]
        Y_train = Y[train_idx]
        X_test = X[test_idx]
        Y_test = Y[test_idx]

        B, mx, my = fit_ridge_operator(X_train, Y_train, alpha=RIDGE_ALPHA)

        Yhat_train = predict_operator(X_train, B, mx, my)
        Yhat_test = predict_operator(X_test, B, mx, my)

        train_r2 = r2_score_global(Y_train, Yhat_train)
        test_r2 = r2_score_global(Y_test, Yhat_test)
        train_cos = mean_row_cosine(Y_train, Yhat_train)
        test_cos = mean_row_cosine(Y_test, Yhat_test)

        U, S, Vt = np.linalg.svd(B, full_matrices=False)

        operators.append({
            "layer": l,
            "B": B,
            "mx": mx,
            "my": my,
            "singular": S.astype(np.float32),
        })

        metrics.append({
            "layer": l,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_cos": train_cos,
            "test_cos": test_cos,
            "s1": float(S[0]),
            "s2": float(S[1]) if len(S) > 1 else 0.0,
            "s_mean": float(np.mean(S)),
            "s_std": float(np.std(S)),
            "fro": float(np.linalg.norm(B, ord="fro")),
        })

    return operators, metrics

def build_operator_signatures(operators):
    sigs = []

    for op in operators:
        B = op["B"]
        S = op["singular"]

        flat = B.reshape(-1)

        stats = np.array([
            np.mean(S),
            np.std(S),
            np.max(S),
            np.min(S),
            np.linalg.norm(B, ord="fro"),
            np.linalg.det(B + 1e-3 * np.eye(B.shape[0], dtype=np.float32)),
        ], dtype=np.float32)

        sig = np.concatenate([flat, S.astype(np.float32), stats], axis=0)
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

C_states_by_condition = {}
sigs_by_condition = {}
metrics_by_condition = {}

clean_library = {}

# ---------------- Clean baseline ----------------

print("\n\n============================================================")
print("Constraint-Audit-1 Step 1: Clean baseline")
print("============================================================\n")

H_clean = capture_hidden_states(texts_by_condition["clean"], "clean")
C_clean, clean_layer_rows = build_constraint_states(H_clean, "clean")

clean_ops, clean_metrics = fit_constraint_operators(C_clean, train_idx, test_idx)
clean_sigs = build_operator_signatures(clean_ops)

C_states_by_condition["clean"] = C_clean
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

print("\nClean constraint operator fit:")
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

gen_rows = generation_check_condition("clean")
all_generation_rows.extend(gen_rows)

del H_clean
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ---------------- Other conditions ----------------

print("\n\n============================================================")
print("Constraint-Audit-1 Step 2: Conditions")
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

    C_states_by_condition[cond] = C_states
    sigs_by_condition[cond] = sigs
    metrics_by_condition[cond] = metrics

    all_layer_rows.extend(layer_rows)

    for m in metrics:
        row = dict(m)
        row["condition"] = cond
        row["epsilon"] = epsilon_by_condition[cond]
        row["status"] = status_by_condition[cond]
        all_operator_metric_rows.append(row)

    mean_test_r2 = np.mean([m["test_r2"] for m in metrics])
    mean_test_cos = np.mean([m["test_cos"] for m in metrics])

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
        print(f"    path     : {format_path(labels)}")

        if k == 4:
            best_k4_row = row

    gen_rows = generation_check_condition(cond)
    all_generation_rows.extend(gen_rows)

    # Condition summary
    boundary = [r for r in layer_rows if r["layer"] in BOUNDARY_LAYERS]
    late = [r for r in layer_rows if r["layer"] in LATE_LAYERS]
    final = layer_rows[-1]

    gen_clean_rate = float(np.mean([r["is_clean"] for r in gen_rows])) if gen_rows else np.nan
    gen_conflict_rate = float(np.mean([r["is_conflict"] for r in gen_rows])) if gen_rows else np.nan

    def mean_key(rows, key):
        return float(np.mean([r[key] for r in rows]))

    summary = {
        "condition": cond,
        "epsilon": epsilon_by_condition[cond],
        "status": status_by_condition[cond],

        "boundary_R_bias": mean_key(boundary, "R_bias_mean"),
        "boundary_R_clean_win": mean_key(boundary, "R_clean_win_rate"),
        "boundary_R_conflict_win": mean_key(boundary, "R_conflict_win_rate"),
        "boundary_R_topk_gap": mean_key(boundary, "R_topk_gap"),
        "boundary_B_distance": mean_key(boundary, "B_distance_mean"),
        "boundary_B_instability": mean_key(boundary, "B_instability_mean"),
        "boundary_F_spread": mean_key(boundary, "F_spread_mean"),

        "late_R_bias": mean_key(late, "R_bias_mean"),
        "late_R_clean_win": mean_key(late, "R_clean_win_rate"),
        "late_R_conflict_win": mean_key(late, "R_conflict_win_rate"),
        "late_B_distance": mean_key(late, "B_distance_mean"),
        "late_B_instability": mean_key(late, "B_instability_mean"),
        "late_F_spread": mean_key(late, "F_spread_mean"),

        "final_R_bias": final["R_bias_mean"],
        "final_R_clean_win": final["R_clean_win_rate"],
        "final_R_conflict_win": final["R_conflict_win_rate"],
        "final_B_distance": final["B_distance_mean"],
        "final_B_instability": final["B_instability_mean"],
        "final_F_spread": final["F_spread_mean"],

        "operator_test_r2": mean_test_r2,
        "operator_test_cos": mean_test_cos,

        "gen_clean_rate": gen_clean_rate,
        "gen_conflict_rate": gen_conflict_rate,
    }

    if best_k4_row is not None:
        summary["k4_path_agreement"] = best_k4_row["path_agreement"]
        summary["k4_tau"] = best_k4_row["tau"]
        summary["k4_tau_shift"] = best_k4_row["tau_shift"]

    condition_summaries.append(summary)

    print("\nCondition constraint summary:")
    print(f"  boundary R_bias       : {summary['boundary_R_bias']:+.5f}")
    print(f"  boundary conflict win : {summary['boundary_R_conflict_win']:.4f}")
    print(f"  boundary B_instability: {summary['boundary_B_instability']:.5f}")
    print(f"  late R_bias           : {summary['late_R_bias']:+.5f}")
    print(f"  final R_bias          : {summary['final_R_bias']:+.5f}")
    print(f"  Gen_C                 : {summary['gen_clean_rate']:.4f}")
    print(f"  Gen_E                 : {summary['gen_conflict_rate']:.4f}")

    del H
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Add clean summary
clean_boundary = [r for r in clean_layer_rows if r["layer"] in BOUNDARY_LAYERS]
clean_late = [r for r in clean_layer_rows if r["layer"] in LATE_LAYERS]
clean_final = clean_layer_rows[-1]
clean_gen = [r for r in all_generation_rows if r["condition"] == "clean"]

def mean_key(rows, key):
    return float(np.mean([r[key] for r in rows]))

clean_summary = {
    "condition": "clean",
    "epsilon": 0.0,
    "status": "clean",

    "boundary_R_bias": mean_key(clean_boundary, "R_bias_mean"),
    "boundary_R_clean_win": mean_key(clean_boundary, "R_clean_win_rate"),
    "boundary_R_conflict_win": mean_key(clean_boundary, "R_conflict_win_rate"),
    "boundary_R_topk_gap": mean_key(clean_boundary, "R_topk_gap"),
    "boundary_B_distance": mean_key(clean_boundary, "B_distance_mean"),
    "boundary_B_instability": mean_key(clean_boundary, "B_instability_mean"),
    "boundary_F_spread": mean_key(clean_boundary, "F_spread_mean"),

    "late_R_bias": mean_key(clean_late, "R_bias_mean"),
    "late_R_clean_win": mean_key(clean_late, "R_clean_win_rate"),
    "late_R_conflict_win": mean_key(clean_late, "R_conflict_win_rate"),
    "late_B_distance": mean_key(clean_late, "B_distance_mean"),
    "late_B_instability": mean_key(clean_late, "B_instability_mean"),
    "late_F_spread": mean_key(clean_late, "F_spread_mean"),

    "final_R_bias": clean_final["R_bias_mean"],
    "final_R_clean_win": clean_final["R_clean_win_rate"],
    "final_R_conflict_win": clean_final["R_conflict_win_rate"],
    "final_B_distance": clean_final["B_distance_mean"],
    "final_B_instability": clean_final["B_instability_mean"],
    "final_F_spread": clean_final["F_spread_mean"],

    "operator_test_r2": np.mean([m["test_r2"] for m in clean_metrics]),
    "operator_test_cos": np.mean([m["test_cos"] for m in clean_metrics]),

    "gen_clean_rate": float(np.mean([r["is_clean"] for r in clean_gen])),
    "gen_conflict_rate": float(np.mean([r["is_conflict"] for r in clean_gen])),

    "k4_path_agreement": 1.0,
    "k4_tau": clean_library[4]["tau"] if 4 in clean_library else None,
    "k4_tau_shift": 0,
}

condition_summaries.insert(0, clean_summary)

# ============================================================
# FINAL TABLE
# ============================================================

print("\n\n============================================================")
print("Constraint-Audit-1 Final Constraint Breakdown Curve")
print("============================================================\n")

print(
    f"{'Cond':<28}"
    f"{'eps':<6}"
    f"{'B_Rbias':<10}"
    f"{'B_Ewin':<10}"
    f"{'B_Instab':<10}"
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
        f"{s['boundary_R_bias']:<10.5f}"
        f"{s['boundary_R_conflict_win']:<10.4f}"
        f"{s['boundary_B_instability']:<10.5f}"
        f"{s['late_R_bias']:<10.5f}"
        f"{s['final_R_bias']:<10.5f}"
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
    layer_path = os.path.join(SAVE_DIR, "constraint_audit1_layerwise_constraint_states.csv")
    df_layers.to_csv(layer_path, index=False)

    df_ops = pd.DataFrame(all_operator_metric_rows)
    op_path = os.path.join(SAVE_DIR, "constraint_audit1_operator_metrics.csv")
    df_ops.to_csv(op_path, index=False)

    df_paths = pd.DataFrame(all_path_rows)
    path_path = os.path.join(SAVE_DIR, "constraint_audit1_path_tau.csv")
    df_paths.to_csv(path_path, index=False)

    df_summary = pd.DataFrame(condition_summaries)
    summary_path = os.path.join(SAVE_DIR, "constraint_audit1_summary.csv")
    df_summary.to_csv(summary_path, index=False)

    df_gen = pd.DataFrame(all_generation_rows)
    gen_path = os.path.join(SAVE_DIR, "constraint_audit1_generation_check.csv")
    df_gen.to_csv(gen_path, index=False)

    prompt_path = os.path.join(SAVE_DIR, "constraint_audit1_prompts.txt")
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
print("Constraint-Audit-1 Interpretation Guide")
print("============================================================\n")

print("Primary success pattern:")
print("  clean / weak conditions:")
print("    boundary_R_bias positive")
print("    late_R_bias positive")
print("    final_R_bias positive")
print("    Gen_C high")
print()
print("  closure_negation_conflict:")
print("    boundary conflict win rises")
print("    late_R_bias drops or turns negative")
print("    final_R_bias turns negative")
print("    Gen_E high")
print()
print("Constraint interpretation:")
print("  R_bias:")
print("    clean-label neighborhood preference minus conflict-label preference.")
print("  B_instability:")
print("    near 1 means close to C/E boundary; near 0 means confidently away from boundary.")
print("  F_spread:")
print("    TopK neighborhood spread; higher can mean larger admissible freedom.")
print()
print("Operator interpretation:")
print("  If clean constraint operators recover tau near L20-L22,")
print("  then C_l=(B_l,R_l,F_l) is not just a diagnostic summary,")
print("  but has layerwise dynamical structure.")
print()
print("If K4 tau remains stable while R_bias flips:")
print("  This supports the 6C conclusion:")
print("    macro tau is stable processing schedule,")
print("    while constraint collapse happens inside the answer basin.")
print()
print("Done.")