# ============================================================
# Audit-6C
# Controlled Symmetry-Breaking / Conservation Breakdown Curve
#
# Goal:
#   After Audit-6B established approximate symmetry preservation,
#   Audit-6C introduces controlled relation-breaking perturbations:
#
#       clean
#       preserve_irrelevant
#       weak_distractor
#       ambiguous_branch
#       strong_branch_conflict
#       direct_location_conflict
#       closure_negation_conflict
#
#   and measures:
#       1. VIM pairwise degradation vs clean
#       2. TopK identity degradation
#       3. operator path agreement degradation
#       4. tau shift
#       5. whether L20-L22 boundary remains stable or breaks
#
# Core curve:
#
#       epsilon
#       -> I_topo / I_rel proxy
#       -> I_closure proxy
#       -> tau shift
#
# This is NOT yet hallucination-output verification.
# It is controlled internal symmetry-breaking.
# ============================================================

import os
import random
import warnings
import gc
from collections import Counter

import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

device = "cuda"
dtype = torch.float16

MAX_LEN = 220
BATCH_SIZE = 4

RANDOM_SEED = 42

TOPK = 5000
CENTER_HIDDEN_FOR_TOPK = True

# Synthetic relation graph count.
# 64 is faster. 96 matches your Audit-6B scale.
N_GRAPHS = 96

# Operator fit
RIDGE_ALPHA = 1e-2
TEST_SIZE = 0.25

# Operator signature
RANDOM_PROJ_DIM = 32
SPECTRUM_K = 64

# Clean library K values
K_LIST = [2, 3, 4]

# Tau
TAU_SEARCH_START_RATIO = 0.45
OUTPUT_WINDOW = 5
MIN_OUTPUT_RUN = 2
EXPECTED_TAU_ZONE = {20, 21, 22}

# Store TopK ids for Jaccard.
# If memory is tight, set False.
STORE_TOPK_IDS = True

SAVE_DIR = "./audit6c_outputs"
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
# SYNTHETIC RELATION GRAPH DATA
# ============================================================

def make_entity(prefix, i):
    return f"{prefix}{i:03d}"

def make_prompt(a, b, c, d, e, f, variant):
    """
    Base closure:
        A belongs_to B
        B located_in C
        Therefore A located_in C
    """

    if variant == "clean":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located in {c}.",
            f"Question: Where is {a} located?",
            "Answer:",
        ]

    elif variant == "preserve_irrelevant":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located in {c}.",
            f"Irrelevant fact 1: {d} belongs to {e}.",
            f"Irrelevant fact 2: {e} is located in {f}.",
            f"Question: Where is {a} located?",
            "Answer:",
        ]

    elif variant == "weak_distractor":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located in {c}.",
            f"Weak distractor: Some sources say {a} belongs to {d}.",
            f"Distractor fact: {d} is located in {e}.",
            f"Question: Where is {a} located?",
            "Answer:",
        ]

    elif variant == "ambiguous_branch":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located in {c}.",
            f"Ambiguous fact: {a} may also belong to {d}.",
            f"Fact 4: {d} is located in {e}.",
            f"Question: Where is {a} located?",
            "Answer:",
        ]

    elif variant == "strong_branch_conflict":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located in {c}.",
            f"Conflicting fact 1: {a} belongs to {d}.",
            f"Conflicting fact 2: {d} is located in {e}.",
            f"Question: Where is {a} located?",
            "Answer:",
        ]

    elif variant == "direct_location_conflict":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located in {c}.",
            f"Conflicting direct fact: {a} is located in {e}.",
            f"Question: Where is {a} located?",
            "Answer:",
        ]

    elif variant == "closure_negation_conflict":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located in {c}.",
            f"Updated record: {b} is located in {e}, not in {c}.",
            f"Question: Where is {a} located?",
            "Answer:",
        ]

    else:
        raise ValueError(f"Unknown variant: {variant}")

    return "\n".join(lines)

CONDITION_META = [
    # name, epsilon, theoretical relation status
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

def build_dataset(n_graphs):
    records = []
    texts_by_condition = {c: [] for c in conditions}

    for i in range(n_graphs):
        a = make_entity("A", i)
        b = make_entity("B", i)
        c = make_entity("C", i)

        d = make_entity("D", i)
        e = make_entity("E", i)
        f = make_entity("F", i)

        prompts = {}
        for cond in conditions:
            prompts[cond] = make_prompt(a, b, c, d, e, f, variant=cond)
            texts_by_condition[cond].append(prompts[cond])

        records.append({
            "idx": i,
            "core_entities": (a, b, c),
            "distractor_entities": (d, e, f),
            "answer_clean": c,
            "answer_conflict": e,
            "prompts": prompts,
        })

    return records, texts_by_condition

records, texts_by_condition = build_dataset(N_GRAPHS)

print("Conditions:")
for cond in conditions:
    print(f"  {cond:<28} epsilon={epsilon_by_condition[cond]} status={status_by_condition[cond]}")

print("\nExample clean prompt:\n")
print(texts_by_condition["clean"][0])

print("\nExample strong_branch_conflict prompt:\n")
print(texts_by_condition["strong_branch_conflict"][0])

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

W_lm = model.lm_head.weight.detach().float().cpu()
vocab_size, d_model = W_lm.shape
print("LM head:", tuple(W_lm.shape))

W_np = W_lm.numpy().astype(np.float32)
W_norm = W_np / (np.linalg.norm(W_np, axis=1, keepdims=True) + 1e-12)

# Fixed random projection for all operator signatures
D_Z = d_model + 1
rng_sig = np.random.default_rng(RANDOM_SEED)

U_SIG = rng_sig.normal(size=(D_Z, RANDOM_PROJ_DIM)).astype(np.float32)
V_SIG = rng_sig.normal(size=(D_Z, RANDOM_PROJ_DIM)).astype(np.float32)

U_SIG = U_SIG / (np.linalg.norm(U_SIG, axis=0, keepdims=True) + 1e-12)
V_SIG = V_SIG / (np.linalg.norm(V_SIG, axis=0, keepdims=True) + 1e-12)

# Fixed train/test split for all conditions
indices = np.arange(N_GRAPHS)
train_idx, test_idx = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    shuffle=True,
)

# Fixed shuffled baseline permutation
rng = np.random.default_rng(RANDOM_SEED)
shuf_perm = rng.permutation(N_GRAPHS)

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
# VIM STATE Z_l = [C_k, r_k]
# ============================================================

def compute_topk_ids(H, k):
    logits = H @ W_np.T
    ids = np.argpartition(logits, -k, axis=1)[:, -k:]

    vals = np.take_along_axis(logits, ids, axis=1)
    order = np.argsort(vals, axis=1)[:, ::-1]
    ids = np.take_along_axis(ids, order, axis=1)

    return ids.astype(np.int32)

def compute_center_spread_from_ids(ids, batch_rows=4):
    n, k = ids.shape
    Z = np.zeros((n, d_model + 1), dtype=np.float32)

    for s in range(0, n, batch_rows):
        e = min(s + batch_rows, n)
        batch_ids = ids[s:e]

        E = W_norm[batch_ids]  # [B, K, D]

        C = E.mean(axis=1)
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)

        sims = np.einsum("bkd,bd->bk", E, C)
        spread = np.mean(1.0 - sims, axis=1, keepdims=True)

        Z[s:e, :d_model] = C
        Z[s:e, d_model:] = spread.astype(np.float32)

    return Z

def build_vim_states(H_raws, condition_name=""):
    Zs = []
    ids_by_layer = [] if STORE_TOPK_IDS else None

    print(f"\nBuilding VIM states for condition={condition_name} ...")
    print(f"TOPK={TOPK}, CENTER_HIDDEN_FOR_TOPK={CENTER_HIDDEN_FOR_TOPK}")

    for l, H_raw in enumerate(H_raws):
        if CENTER_HIDDEN_FOR_TOPK:
            H_for_topk = H_raw - H_raw.mean(axis=0, keepdims=True)
        else:
            H_for_topk = H_raw

        ids = compute_topk_ids(H_for_topk, TOPK)
        Z = compute_center_spread_from_ids(ids)

        Zs.append(Z.astype(np.float32))

        if STORE_TOPK_IDS:
            ids_by_layer.append(ids)

        print(
            f"L{l:02d}: Z={Z.shape}, "
            f"spread_mean={Z[:, -1].mean():.4f}, "
            f"spread_std={Z[:, -1].std():.4f}"
        )

    return Zs, ids_by_layer

def get_vim_for_condition(cond):
    H = capture_hidden_states(texts_by_condition[cond], condition_name=cond)
    Zs, ids = build_vim_states(H, condition_name=cond)
    del H
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return Zs, ids

# ============================================================
# METRICS
# ============================================================

def row_cosine(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return np.sum(A * B, axis=1)

def mean_row_cosine(A, B):
    return float(np.mean(row_cosine(A, B)))

def mean_l2(A, B):
    return float(np.mean(np.linalg.norm(A - B, axis=1)))

def jaccard_rows(A, B):
    N = A.shape[0]
    out = np.zeros(N, dtype=np.float32)

    for i in range(N):
        ai = np.sort(A[i])
        bi = np.sort(B[i])
        inter = np.intersect1d(ai, bi, assume_unique=False).size
        union = A.shape[1] + B.shape[1] - inter
        out[i] = inter / max(union, 1)

    return out

def r2_score_global(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))

def effective_rank_from_singular_values(s):
    s = np.asarray(s, dtype=np.float64)
    if s.sum() <= 1e-12:
        return 0.0
    p = s / (s.sum() + 1e-12)
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))

def r_energy_rank(s, threshold=0.90):
    s2 = np.asarray(s, dtype=np.float64) ** 2
    if s2.sum() <= 1e-12:
        return 0
    cs = np.cumsum(s2) / s2.sum()
    return int(np.searchsorted(cs, threshold) + 1)

def safe_randomized_svd(M, n_components):
    k = min(n_components, min(M.shape) - 1)

    if k <= 1:
        return np.array([0.0], dtype=np.float32)

    try:
        _, S, _ = randomized_svd(
            M,
            n_components=k,
            random_state=RANDOM_SEED,
            n_iter=5,
        )
        return S.astype(np.float32)
    except Exception:
        try:
            S = np.linalg.svd(M, compute_uv=False)
            return S[:k].astype(np.float32)
        except Exception:
            return np.zeros(k, dtype=np.float32)

# ============================================================
# OPERATOR FIT
# ============================================================

def fit_dual_ridge_operator(X_train, Y_train, alpha=1e-2):
    mx = X_train.mean(axis=0, keepdims=True)
    my = Y_train.mean(axis=0, keepdims=True)

    Xc = X_train - mx
    Yc = Y_train - my

    K = Xc @ Xc.T
    K = K + alpha * np.eye(K.shape[0], dtype=np.float32)

    A = np.linalg.solve(K, Yc)
    B = Xc.T @ A

    return B.astype(np.float32), mx.astype(np.float32), my.astype(np.float32)

def predict_operator(X, B, mx, my):
    return (X - mx) @ B + my

def fit_vim_operators(Zs):
    operators = []
    metrics = []

    for l in range(num_layers - 1):
        X = Zs[l]
        Y = Zs[l + 1]

        X_train = X[train_idx]
        Y_train = Y[train_idx]
        X_test = X[test_idx]
        Y_test = Y[test_idx]

        B, mx, my = fit_dual_ridge_operator(X_train, Y_train, alpha=RIDGE_ALPHA)

        Yhat_train = predict_operator(X_train, B, mx, my)
        Yhat_test = predict_operator(X_test, B, mx, my)

        train_r2 = r2_score_global(Y_train, Yhat_train)
        test_r2 = r2_score_global(Y_test, Yhat_test)
        train_cos = mean_row_cosine(Y_train, Yhat_train)
        test_cos = mean_row_cosine(Y_test, Yhat_test)

        S = safe_randomized_svd(B, SPECTRUM_K)

        operators.append({
            "layer": l,
            "B": B,
            "mx": mx,
            "my": my,
            "singular": S,
        })

        metrics.append({
            "layer": l,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_cos": train_cos,
            "test_cos": test_cos,
            "eff_rank": effective_rank_from_singular_values(S),
            "r90": r_energy_rank(S, 0.90),
            "r95": r_energy_rank(S, 0.95),
        })

    return operators, metrics

def build_operator_signatures(operators, spectrum_k=64):
    sigs = []

    for op in operators:
        B = op["B"]
        S = op["singular"]

        sketch = U_SIG.T @ (B @ V_SIG)
        sketch = sketch.reshape(-1)

        s = np.zeros(spectrum_k, dtype=np.float32)
        m = min(len(S), spectrum_k)
        s[:m] = S[:m]
        s_norm = s / (np.linalg.norm(s) + 1e-12)

        stats = np.array([
            float(np.mean(S)),
            float(np.std(S)),
            float(np.max(S)),
            float(effective_rank_from_singular_values(S)),
            float(r_energy_rank(S, 0.90)),
            float(r_energy_rank(S, 0.95)),
        ], dtype=np.float32)

        sig = np.concatenate([sketch, s_norm, stats], axis=0)
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
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.mean(a == b))

def segments(labels):
    segs = []
    start = 0

    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            segs.append((start, i - 1, int(labels[i - 1])))
            start = i

    segs.append((start, len(labels) - 1, int(labels[-1])))
    return segs

def print_segments(labels):
    for s, e, c in segments(labels):
        if s == e:
            print(f"  O{c}: op {s}, layer L{s}->L{s+1}")
        else:
            print(f"  O{c}: ops {s}-{e}, layers L{s}->L{e+1}")

# ============================================================
# CLEAN BASELINE
# ============================================================

print("\n\n============================================================")
print("Audit-6C Step 1: Clean Baseline")
print("============================================================\n")

clean_Zs, clean_IDS = get_vim_for_condition("clean")

clean_ops, clean_metrics = fit_vim_operators(clean_Zs)
clean_sigs = build_operator_signatures(clean_ops, spectrum_k=SPECTRUM_K)

mean_clean_r2 = np.mean([m["test_r2"] for m in clean_metrics])
mean_clean_cos = np.mean([m["test_cos"] for m in clean_metrics])

print(f"\nClean operator fit: mean TestR2={mean_clean_r2:.4f}, mean TestCos={mean_clean_cos:.4f}")

clean_scaler = StandardScaler()
clean_X = clean_scaler.fit_transform(clean_sigs)

clean_libraries = {}

print("\n================ Clean VIM Libraries ================\n")

for k in K_LIST:
    km = KMeans(
        n_clusters=k,
        random_state=RANDOM_SEED,
        n_init=50,
    )

    clean_raw = km.fit_predict(clean_X)
    clean_labels, raw_to_ordered = remap_by_first_appearance(clean_raw)

    clean_sil = silhouette_score(clean_X, clean_raw) if len(set(clean_raw)) > 1 else -1.0
    clean_tau, clean_out = estimate_tau(
        clean_labels,
        output_window=OUTPUT_WINDOW,
        start_ratio=TAU_SEARCH_START_RATIO,
        min_run=MIN_OUTPUT_RUN,
    )

    clean_libraries[k] = {
        "k": k,
        "model": km,
        "raw_to_ordered": raw_to_ordered,
        "labels": clean_labels,
        "silhouette": clean_sil,
        "tau": clean_tau,
        "output_cluster": clean_out,
    }

    print(f"\nK={k}")
    print(f"  silhouette: {clean_sil:.4f}")
    print(f"  path      : {format_path(clean_labels)}")
    print(f"  tau       : {clean_tau}")
    print("  segments:")
    print_segments(clean_labels)

# ============================================================
# PAIRWISE BREAKDOWN METRICS
# ============================================================

def compute_pairwise_breakdown(cond, cond_Zs, cond_IDS):
    rows = []

    print(f"\n---------------- Pairwise breakdown: {cond} vs clean ----------------\n")

    print(
        f"{'Layer':<8}"
        f"{'CenterCos':<12}"
        f"{'ShufCos':<12}"
        f"{'CosLift':<12}"
        f"{'SpreadΔ':<12}"
        f"{'Jaccard':<12}"
        f"{'ShufJac':<12}"
        f"{'JacLift':<12}"
        f"{'Z_L2':<12}"
    )

    for l in range(num_layers):
        Zc = clean_Zs[l]
        Zt = cond_Zs[l]

        Cc = Zc[:, :d_model]
        Ct = Zt[:, :d_model]

        spread_c = Zc[:, -1]
        spread_t = Zt[:, -1]

        center_cos = float(np.mean(row_cosine(Cc, Ct)))
        shuf_cos = float(np.mean(row_cosine(Cc, Ct[shuf_perm])))
        cos_lift = center_cos - shuf_cos

        spread_delta = float(np.mean(np.abs(spread_c - spread_t)))
        z_l2 = mean_l2(Zc, Zt)

        if STORE_TOPK_IDS:
            jac = float(np.mean(jaccard_rows(clean_IDS[l], cond_IDS[l])))
            shuf_jac = float(np.mean(jaccard_rows(clean_IDS[l], cond_IDS[l][shuf_perm])))
            jac_lift = jac - shuf_jac
        else:
            jac = np.nan
            shuf_jac = np.nan
            jac_lift = np.nan

        row = {
            "condition": cond,
            "epsilon": epsilon_by_condition[cond],
            "status": status_by_condition[cond],
            "layer": l,
            "center_cos": center_cos,
            "shuf_cos": shuf_cos,
            "cos_lift": cos_lift,
            "spread_delta": spread_delta,
            "jaccard": jac,
            "shuf_jaccard": shuf_jac,
            "jaccard_lift": jac_lift,
            "z_l2": z_l2,
        }

        rows.append(row)

        print(
            f"L{l:02d}    "
            f"{center_cos:<12.4f}"
            f"{shuf_cos:<12.4f}"
            f"{cos_lift:<12.4f}"
            f"{spread_delta:<12.5f}"
            f"{jac:<12.4f}"
            f"{shuf_jac:<12.4f}"
            f"{jac_lift:<12.4f}"
            f"{z_l2:<12.4f}"
        )

    return rows

# ============================================================
# CONDITION LOOP
# ============================================================

pair_rows = []
path_rows = []
operator_metric_rows = []

condition_summaries = []

print("\n\n============================================================")
print("Audit-6C Step 2: Perturbation Conditions")
print("============================================================\n")

for cond in conditions:
    if cond == "clean":
        continue

    print("\n\n############################################################")
    print(f"Condition: {cond}")
    print(f"epsilon={epsilon_by_condition[cond]}, status={status_by_condition[cond]}")
    print("############################################################\n")

    cond_Zs, cond_IDS = get_vim_for_condition(cond)

    # Pairwise VIM / TopK breakdown
    rows = compute_pairwise_breakdown(cond, cond_Zs, cond_IDS)
    pair_rows.extend(rows)

    # Operator fit and signature
    cond_ops, cond_metrics = fit_vim_operators(cond_Zs)
    cond_sigs = build_operator_signatures(cond_ops, spectrum_k=SPECTRUM_K)

    mean_test_r2 = np.mean([m["test_r2"] for m in cond_metrics])
    mean_test_cos = np.mean([m["test_cos"] for m in cond_metrics])

    for m in cond_metrics:
        mm = dict(m)
        mm["condition"] = cond
        mm["epsilon"] = epsilon_by_condition[cond]
        mm["status"] = status_by_condition[cond]
        operator_metric_rows.append(mm)

    print(f"\nOperator fit condition={cond}: mean TestR2={mean_test_r2:.4f}, mean TestCos={mean_test_cos:.4f}")

    # Assign to clean libraries
    X = clean_scaler.transform(cond_sigs)

    best_k4_row = None

    print("\nClean library assignment:")

    for k in K_LIST:
        lib = clean_libraries[k]
        km = lib["model"]

        raw_pred = km.predict(X)
        labels = apply_mapping(raw_pred, lib["raw_to_ordered"])

        clean_labels = lib["labels"]
        clean_tau = lib["tau"]

        tau, out_c = estimate_tau(
            labels,
            output_window=OUTPUT_WINDOW,
            start_ratio=TAU_SEARCH_START_RATIO,
            min_run=MIN_OUTPUT_RUN,
        )

        agree = path_agreement(clean_labels, labels)
        tau_shift = None if (tau is None or clean_tau is None) else tau - clean_tau
        tau_in_zone = tau in EXPECTED_TAU_ZONE if tau is not None else False

        row = {
            "k": k,
            "condition": cond,
            "epsilon": epsilon_by_condition[cond],
            "status": status_by_condition[cond],
            "clean_tau": clean_tau,
            "tau": tau,
            "tau_shift": tau_shift,
            "tau_in_expected_zone": tau_in_zone,
            "path_agreement": agree,
            "clean_path": format_path(clean_labels),
            "condition_path": format_path(labels),
        }

        path_rows.append(row)

        print(f"\n  K={k}")
        print(f"    path agreement: {agree:.4f}")
        print(f"    clean tau     : {clean_tau}")
        print(f"    cond tau      : {tau}")
        print(f"    tau shift     : {tau_shift}")
        print(f"    tau in zone   : {tau_in_zone}")
        print(f"    path          : {format_path(labels)}")

        if k == 4:
            best_k4_row = row

    # Condition-level summary
    mean_center_cos = np.mean([r["center_cos"] for r in rows])
    mean_cos_lift = np.mean([r["cos_lift"] for r in rows])
    mean_jaccard = np.nanmean([r["jaccard"] for r in rows])
    mean_jaccard_lift = np.nanmean([r["jaccard_lift"] for r in rows])
    mean_spread_delta = np.mean([r["spread_delta"] for r in rows])
    mean_z_l2 = np.mean([r["z_l2"] for r in rows])

    boundary_rows = [r for r in rows if r["layer"] in EXPECTED_TAU_ZONE]
    boundary_center_cos = np.mean([r["center_cos"] for r in boundary_rows])
    boundary_cos_lift = np.mean([r["cos_lift"] for r in boundary_rows])
    boundary_jaccard = np.nanmean([r["jaccard"] for r in boundary_rows])
    boundary_jaccard_lift = np.nanmean([r["jaccard_lift"] for r in boundary_rows])

    summary = {
        "condition": cond,
        "epsilon": epsilon_by_condition[cond],
        "status": status_by_condition[cond],
        "mean_center_cos": mean_center_cos,
        "mean_cos_lift": mean_cos_lift,
        "mean_jaccard": mean_jaccard,
        "mean_jaccard_lift": mean_jaccard_lift,
        "mean_spread_delta": mean_spread_delta,
        "mean_z_l2": mean_z_l2,
        "boundary_center_cos": boundary_center_cos,
        "boundary_cos_lift": boundary_cos_lift,
        "boundary_jaccard": boundary_jaccard,
        "boundary_jaccard_lift": boundary_jaccard_lift,
        "mean_test_r2": mean_test_r2,
        "mean_test_cos": mean_test_cos,
    }

    if best_k4_row is not None:
        summary["k4_path_agreement"] = best_k4_row["path_agreement"]
        summary["k4_tau"] = best_k4_row["tau"]
        summary["k4_tau_shift"] = best_k4_row["tau_shift"]
        summary["k4_tau_in_expected_zone"] = best_k4_row["tau_in_expected_zone"]

    condition_summaries.append(summary)

    print("\nCondition summary:")
    print(f"  mean center_cos       : {mean_center_cos:.4f}")
    print(f"  mean center_cos lift  : {mean_cos_lift:.4f}")
    print(f"  mean jaccard          : {mean_jaccard:.4f}")
    print(f"  mean jaccard lift     : {mean_jaccard_lift:.4f}")
    print(f"  mean spread_delta     : {mean_spread_delta:.5f}")
    print(f"  boundary L20-L22 cos  : {boundary_center_cos:.4f}")
    print(f"  boundary L20-L22 lift : {boundary_cos_lift:.4f}")
    if best_k4_row is not None:
        print(f"  K=4 path agreement    : {best_k4_row['path_agreement']:.4f}")
        print(f"  K=4 tau               : {best_k4_row['tau']}")
        print(f"  K=4 tau_shift         : {best_k4_row['tau_shift']}")

    # Free condition memory
    del cond_Zs, cond_IDS, cond_ops, cond_metrics, cond_sigs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================
# FINAL BREAKDOWN CURVE
# ============================================================

print("\n\n============================================================")
print("Audit-6C Final Conservation Breakdown Curve")
print("============================================================\n")

print(
    f"{'Cond':<28}"
    f"{'eps':<6}"
    f"{'status':<22}"
    f"{'cos':<9}"
    f"{'jac':<9}"
    f"{'Bcos':<9}"
    f"{'Bjac':<9}"
    f"{'K4Agree':<9}"
    f"{'K4Tau':<8}"
    f"{'K4Shift':<9}"
    f"{'R2':<9}"
)

for s in sorted(condition_summaries, key=lambda x: x["epsilon"]):
    print(
        f"{s['condition']:<28}"
        f"{s['epsilon']:<6.1f}"
        f"{s['status']:<22}"
        f"{s['mean_center_cos']:<9.4f}"
        f"{s['mean_jaccard']:<9.4f}"
        f"{s['boundary_center_cos']:<9.4f}"
        f"{s['boundary_jaccard']:<9.4f}"
        f"{s.get('k4_path_agreement', np.nan):<9.4f}"
        f"{str(s.get('k4_tau', None)):<8}"
        f"{str(s.get('k4_tau_shift', None)):<9}"
        f"{s['mean_test_r2']:<9.4f}"
    )

# ============================================================
# SAVE OUTPUTS
# ============================================================

try:
    import pandas as pd

    df_pair = pd.DataFrame(pair_rows)
    pair_path = os.path.join(SAVE_DIR, "audit6c_pairwise_breakdown_metrics.csv")
    df_pair.to_csv(pair_path, index=False)

    df_path = pd.DataFrame(path_rows)
    path_summary = os.path.join(SAVE_DIR, "audit6c_path_tau_breakdown.csv")
    df_path.to_csv(path_summary, index=False)

    df_metric = pd.DataFrame(operator_metric_rows)
    metric_path = os.path.join(SAVE_DIR, "audit6c_operator_metrics.csv")
    df_metric.to_csv(metric_path, index=False)

    df_summary = pd.DataFrame(condition_summaries)
    summary_path = os.path.join(SAVE_DIR, "audit6c_conservation_breakdown_curve.csv")
    df_summary.to_csv(summary_path, index=False)

    prompt_path = os.path.join(SAVE_DIR, "audit6c_prompts.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        for i, rec in enumerate(records[:10]):
            f.write(f"===== graph {i} =====\n")
            for cond in conditions:
                f.write(f"\n--- {cond} | eps={epsilon_by_condition[cond]} | {status_by_condition[cond]} ---\n")
                f.write(rec["prompts"][cond] + "\n")
            f.write("\n")

    print("\nSaved outputs:")
    print(" ", pair_path)
    print(" ", path_summary)
    print(" ", metric_path)
    print(" ", summary_path)
    print(" ", prompt_path)

except Exception as e:
    print("\nCould not save outputs:", repr(e))

# ============================================================
# INTERPRETATION GUIDE
# ============================================================

print("\n\n============================================================")
print("Audit-6C Interpretation Guide")
print("============================================================\n")

print("Primary success pattern:")
print("  As epsilon increases:")
print("    center_cos decreases")
print("    Jaccard decreases")
print("    boundary L20-L22 metrics degrade")
print("    path agreement drops")
print("    tau shifts earlier/later or leaves L20-L22")
print()

print("If preserve_irrelevant remains close to clean:")
print("  This reproduces Audit-6B and validates the 6C input construction.")
print()

print("If weak_distractor changes TopK identity but tau remains stable:")
print("  TopK neighborhood is sensitive, but macro phase boundary is still robust.")
print()

print("If strong_branch_conflict or direct_location_conflict shifts tau:")
print("  Relation/closure conflict is coupled to VIM macro phase organization.")
print()

print("If closure_negation_conflict produces the largest tau/path disruption:")
print("  Closure preservation is a stronger invariant candidate than raw relation identity.")
print()

print("If path/tau remain stable even under strong conflicts:")
print("  6C does NOT yet connect relation breaking to tau; stronger output-level or hidden perturbation tests are needed.")
print()

print("Done.")