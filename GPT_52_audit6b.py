# ============================================================
# Audit-6B
# Symmetry Preservation Test for VIM / TopK Neighborhood Space
#
# Goal:
#   Test whether relation-preserving input transformations preserve:
#     1. VIM neighborhood states Z_l = [C_k(H_l), r_k(H_l)]
#     2. TopK neighborhood invariants
#     3. Coarse VIM operator path
#     4. Macro phase boundary tau near L20-L22
#
# Transformations:
#   clean
#   rename      : entity relabeling; relation graph is isomorphic
#   redundant   : add redundant facts; relation closure unchanged
#   irrelevant  : add unrelated facts; queried relation unchanged
#   permuted    : reorder facts; relation graph unchanged
#
# This is 6B, not 6C:
#   All transformations are intended to preserve the underlying relation graph.
#   Symmetry breaking / controlled conflict perturbation belongs to Audit-6C.
# ============================================================

import os
import random
import warnings
from collections import Counter, defaultdict

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

MAX_LEN = 180
BATCH_SIZE = 4

RANDOM_SEED = 42

TOPK = 5000
CENTER_HIDDEN_FOR_TOPK = True

# Synthetic graph size.
# 64 or 96 is usually enough for 6B.
# If GPU / CPU memory allows, you may raise to 128.
N_GRAPHS = 96

# Operator fit
RIDGE_ALPHA = 1e-2
TEST_SIZE = 0.25

# Operator signature
RANDOM_PROJ_DIM = 32
SPECTRUM_K = 64

# Coarse cluster tests
K_LIST = [2, 3, 4]

# Tau
TAU_SEARCH_START_RATIO = 0.45
OUTPUT_WINDOW = 5
MIN_OUTPUT_RUN = 2
EXPECTED_TAU_ZONE = {20, 21, 22}

# Store TopK ids for Jaccard.
# This may consume memory. Keep True for 6B.
STORE_TOPK_IDS = True

SAVE_DIR = "./audit6b_outputs"
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
    # Use pseudo-symbolic names to reduce prior factual knowledge.
    return f"{prefix}{i:03d}"

def make_prompt(a, b, c, d=None, e=None, f=None, variant="clean"):
    """
    Relation graph:
      a belongs_to b
      b located_in c
    Query:
      Where is a located?
    Expected answer:
      c

    Symmetry-preserving transformations:
      rename      : same structure with different labels
      redundant   : add a logically redundant closure hint
      irrelevant  : add unrelated graph d -> e -> f
      permuted    : reverse fact order
    """

    if variant == "clean":
        return (
            "You are given a small relation graph.\n"
            f"Fact 1: {a} belongs to {b}.\n"
            f"Fact 2: {b} is located in {c}.\n"
            f"Question: Where is {a} located?\n"
            "Answer:"
        )

    if variant == "redundant":
        return (
            "You are given a small relation graph.\n"
            f"Fact 1: {a} belongs to {b}.\n"
            f"Fact 2: {b} is located in {c}.\n"
            f"Redundant note: If an entity belongs to {b}, then it is associated with the location {c}.\n"
            f"Question: Where is {a} located?\n"
            "Answer:"
        )

    if variant == "irrelevant":
        return (
            "You are given a small relation graph.\n"
            f"Fact 1: {a} belongs to {b}.\n"
            f"Fact 2: {b} is located in {c}.\n"
            f"Irrelevant fact 1: {d} belongs to {e}.\n"
            f"Irrelevant fact 2: {e} is located in {f}.\n"
            f"Question: Where is {a} located?\n"
            "Answer:"
        )

    if variant == "permuted":
        return (
            "You are given a small relation graph.\n"
            f"Fact 1: {b} is located in {c}.\n"
            f"Fact 2: {a} belongs to {b}.\n"
            f"Question: Where is {a} located?\n"
            "Answer:"
        )

    raise ValueError(f"Unknown variant: {variant}")

def build_dataset(n_graphs):
    """
    Returns:
      records: list of dicts.
      texts_by_condition: dict condition -> list[str]
    """

    conditions = ["clean", "rename", "redundant", "irrelevant", "permuted"]
    texts_by_condition = {c: [] for c in conditions}
    records = []

    for i in range(n_graphs):
        # Clean entity triple
        a = make_entity("A", i)
        b = make_entity("B", i)
        c = make_entity("C", i)

        # Renamed isomorphic triple
        ar = make_entity("X", i)
        br = make_entity("Y", i)
        cr = make_entity("Z", i)

        # Irrelevant graph
        d = make_entity("D", i)
        e = make_entity("E", i)
        f = make_entity("F", i)

        clean = make_prompt(a, b, c, variant="clean")
        rename = make_prompt(ar, br, cr, variant="clean")
        redundant = make_prompt(a, b, c, variant="redundant")
        irrelevant = make_prompt(a, b, c, d=d, e=e, f=f, variant="irrelevant")
        permuted = make_prompt(a, b, c, variant="permuted")

        item = {
            "idx": i,
            "clean_entities": (a, b, c),
            "rename_entities": (ar, br, cr),
            "irrelevant_entities": (d, e, f),
            "answer_clean": c,
            "answer_rename": cr,
            "prompts": {
                "clean": clean,
                "rename": rename,
                "redundant": redundant,
                "irrelevant": irrelevant,
                "permuted": permuted,
            },
        }

        records.append(item)

        for cond in conditions:
            texts_by_condition[cond].append(item["prompts"][cond])

    return records, texts_by_condition

records, texts_by_condition = build_dataset(N_GRAPHS)
conditions = list(texts_by_condition.keys())

print("Conditions:", conditions)
print("Graphs:", len(records))
print("\nExample clean prompt:\n")
print(texts_by_condition["clean"][0])
print("\nExample irrelevant prompt:\n")
print(texts_by_condition["irrelevant"][0])

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

# ============================================================
# CAPTURE HIDDEN STATES
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

            hidden_states = outputs.hidden_states[1:]  # remove embedding state

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

# ============================================================
# BASIC METRICS
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
    """
    A, B: [N, K] int arrays.
    Returns rowwise Jaccard.
    """
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
# RUN CAPTURE + VIM BUILD
# ============================================================

H_by_condition = {}
Z_by_condition = {}
IDS_by_condition = {}

for cond in conditions:
    H_raws = capture_hidden_states(texts_by_condition[cond], condition_name=cond)
    Zs, ids_by_layer = build_vim_states(H_raws, condition_name=cond)

    H_by_condition[cond] = H_raws
    Z_by_condition[cond] = Zs
    IDS_by_condition[cond] = ids_by_layer

    # Free hidden states early if memory pressure is high.
    # Keep H_by_condition only if you want hidden-space diagnostics.
    # del H_raws

# ============================================================
# 6B PART 1: PAIRWISE SYMMETRY PRESERVATION METRICS
# ============================================================

print("\n\n============================================================")
print("Audit-6B Part 1: Pairwise Symmetry Preservation")
print("============================================================\n")

rng = np.random.default_rng(RANDOM_SEED)
perm = rng.permutation(N_GRAPHS)

pair_rows = []

clean_Zs = Z_by_condition["clean"]
clean_IDS = IDS_by_condition["clean"]

for cond in conditions:
    if cond == "clean":
        continue

    print(f"\n---------------- condition={cond} vs clean ----------------\n")

    cond_Zs = Z_by_condition[cond]
    cond_IDS = IDS_by_condition[cond]

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
        shuf_cos = float(np.mean(row_cosine(Cc, Ct[perm])))
        cos_lift = center_cos - shuf_cos

        spread_delta = float(np.mean(np.abs(spread_c - spread_t)))

        z_l2 = mean_l2(Zc, Zt)

        if STORE_TOPK_IDS:
            jac = float(np.mean(jaccard_rows(clean_IDS[l], cond_IDS[l])))
            shuf_jac = float(np.mean(jaccard_rows(clean_IDS[l], cond_IDS[l][perm])))
            jac_lift = jac - shuf_jac
        else:
            jac = np.nan
            shuf_jac = np.nan
            jac_lift = np.nan

        row = {
            "condition": cond,
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

        pair_rows.append(row)

        # Print all layers for visibility.
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

    # Condition-level summary
    rows = [r for r in pair_rows if r["condition"] == cond]
    mean_cos = np.mean([r["center_cos"] for r in rows])
    mean_cos_lift = np.mean([r["cos_lift"] for r in rows])
    mean_jac = np.mean([r["jaccard"] for r in rows])
    mean_jac_lift = np.mean([r["jaccard_lift"] for r in rows])
    mean_spread_delta = np.mean([r["spread_delta"] for r in rows])

    boundary_rows = [r for r in rows if r["layer"] in EXPECTED_TAU_ZONE]
    b_cos = np.mean([r["center_cos"] for r in boundary_rows])
    b_lift = np.mean([r["cos_lift"] for r in boundary_rows])

    print("\nCondition summary:")
    print(f"  mean center_cos       : {mean_cos:.4f}")
    print(f"  mean center_cos lift  : {mean_cos_lift:.4f}")
    print(f"  mean jaccard          : {mean_jac:.4f}")
    print(f"  mean jaccard lift     : {mean_jac_lift:.4f}")
    print(f"  mean spread_delta     : {mean_spread_delta:.5f}")
    print(f"  boundary L20-L22 cos  : {b_cos:.4f}")
    print(f"  boundary L20-L22 lift : {b_lift:.4f}")

# ============================================================
# OPERATOR FIT HELPERS
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

def fit_vim_operators(Zs, train_idx, test_idx):
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

# Fixed random projection for all conditions
D_Z = d_model + 1
rng_sig = np.random.default_rng(RANDOM_SEED)

U_SIG = rng_sig.normal(size=(D_Z, RANDOM_PROJ_DIM)).astype(np.float32)
V_SIG = rng_sig.normal(size=(D_Z, RANDOM_PROJ_DIM)).astype(np.float32)

U_SIG = U_SIG / (np.linalg.norm(U_SIG, axis=0, keepdims=True) + 1e-12)
V_SIG = V_SIG / (np.linalg.norm(V_SIG, axis=0, keepdims=True) + 1e-12)

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
    """
    Use clean raw-label -> ordered-label mapping.
    Unknown raw labels should not happen for KMeans predict.
    """
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
# 6B PART 2: CONDITION-WISE VIM OPERATOR FIT
# ============================================================

print("\n\n============================================================")
print("Audit-6B Part 2: Condition-wise VIM Operator Fit")
print("============================================================\n")

indices = np.arange(N_GRAPHS)
train_idx, test_idx = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    shuffle=True,
)

operators_by_condition = {}
metrics_by_condition = {}
sigs_by_condition = {}

for cond in conditions:
    ops, mets = fit_vim_operators(Z_by_condition[cond], train_idx, test_idx)
    sigs = build_operator_signatures(ops, spectrum_k=SPECTRUM_K)

    operators_by_condition[cond] = ops
    metrics_by_condition[cond] = mets
    sigs_by_condition[cond] = sigs

    mean_test_r2 = np.mean([m["test_r2"] for m in mets])
    mean_test_cos = np.mean([m["test_cos"] for m in mets])

    print(f"{cond:<12} mean TestR2={mean_test_r2:.4f}, mean TestCos={mean_test_cos:.4f}")

# ============================================================
# 6B PART 3: CLEAN LIBRARY ASSIGNMENT + PATH AGREEMENT
# ============================================================

print("\n\n============================================================")
print("Audit-6B Part 3: Clean Library Assignment and Path Agreement")
print("============================================================\n")

clean_sigs = sigs_by_condition["clean"]
clean_scaler = StandardScaler()
clean_X = clean_scaler.fit_transform(clean_sigs)

path_rows = []

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

    print(f"\n================ Clean Library K={k} ================\n")
    print(f"Clean silhouette: {clean_sil:.4f}")
    print(f"Clean path: {format_path(clean_labels)}")
    print("Clean segments:")
    print_segments(clean_labels)
    print(f"Clean tau: {clean_tau}, output cluster=O{clean_out}")

    for cond in conditions:
        X = clean_scaler.transform(sigs_by_condition[cond])
        raw_pred = km.predict(X)
        labels = apply_mapping(raw_pred, raw_to_ordered)

        tau, out_c = estimate_tau(
            labels,
            output_window=OUTPUT_WINDOW,
            start_ratio=TAU_SEARCH_START_RATIO,
            min_run=MIN_OUTPUT_RUN,
        )

        agree = path_agreement(clean_labels, labels)

        tau_shift = None if (tau is None or clean_tau is None) else tau - clean_tau
        tau_in_zone = tau in EXPECTED_TAU_ZONE if tau is not None else False

        print(f"\nCondition={cond}")
        print(f"  path      : {format_path(labels)}")
        print(f"  agreement : {agree:.4f}")
        print(f"  tau       : {tau}")
        print(f"  tau_shift : {tau_shift}")
        print(f"  tau_in_expected_zone: {tau_in_zone}")

        path_rows.append({
            "k": k,
            "condition": cond,
            "clean_silhouette": clean_sil,
            "clean_tau": clean_tau,
            "tau": tau,
            "tau_shift": tau_shift,
            "tau_in_expected_zone": tau_in_zone,
            "path_agreement": agree,
            "clean_path": format_path(clean_labels),
            "condition_path": format_path(labels),
        })

# ============================================================
# 6B PART 4: SUMMARY
# ============================================================

print("\n\n============================================================")
print("Audit-6B Summary")
print("============================================================\n")

print("Pairwise symmetry summary:")
for cond in conditions:
    if cond == "clean":
        continue

    rows = [r for r in pair_rows if r["condition"] == cond]

    mean_cos = np.mean([r["center_cos"] for r in rows])
    mean_cos_lift = np.mean([r["cos_lift"] for r in rows])
    mean_jac = np.mean([r["jaccard"] for r in rows])
    mean_jac_lift = np.mean([r["jaccard_lift"] for r in rows])
    mean_spread_delta = np.mean([r["spread_delta"] for r in rows])

    boundary_rows = [r for r in rows if r["layer"] in EXPECTED_TAU_ZONE]
    boundary_cos = np.mean([r["center_cos"] for r in boundary_rows])
    boundary_lift = np.mean([r["cos_lift"] for r in boundary_rows])

    print(
        f"{cond:<12} "
        f"mean_cos={mean_cos:.4f} "
        f"cos_lift={mean_cos_lift:.4f} "
        f"jac={mean_jac:.4f} "
        f"jac_lift={mean_jac_lift:.4f} "
        f"spreadΔ={mean_spread_delta:.5f} "
        f"L20-22_cos={boundary_cos:.4f} "
        f"L20-22_lift={boundary_lift:.4f}"
    )

print("\nPath agreement summary:")
for row in path_rows:
    if row["condition"] == "clean":
        continue
    print(
        f"K={row['k']:<2} {row['condition']:<12} "
        f"agree={row['path_agreement']:.4f} "
        f"clean_tau={row['clean_tau']} "
        f"tau={row['tau']} "
        f"shift={row['tau_shift']} "
        f"in_zone={row['tau_in_expected_zone']}"
    )

# ============================================================
# SAVE OUTPUTS
# ============================================================

try:
    import pandas as pd

    df_pair = pd.DataFrame(pair_rows)
    pair_path = os.path.join(SAVE_DIR, "audit6b_pairwise_symmetry_metrics.csv")
    df_pair.to_csv(pair_path, index=False)

    df_path = pd.DataFrame(path_rows)
    path_summary = os.path.join(SAVE_DIR, "audit6b_path_agreement.csv")
    df_path.to_csv(path_summary, index=False)

    # Condition operator metrics
    metric_rows = []
    for cond in conditions:
        for m in metrics_by_condition[cond]:
            row = dict(m)
            row["condition"] = cond
            metric_rows.append(row)

    df_metric = pd.DataFrame(metric_rows)
    metric_path = os.path.join(SAVE_DIR, "audit6b_operator_metrics.csv")
    df_metric.to_csv(metric_path, index=False)

    text_path = os.path.join(SAVE_DIR, "audit6b_prompts.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        for i, rec in enumerate(records[:10]):
            f.write(f"===== graph {i} =====\n")
            for cond in conditions:
                f.write(f"\n--- {cond} ---\n")
                f.write(rec["prompts"][cond] + "\n")
            f.write("\n")

    print("\nSaved outputs:")
    print(" ", pair_path)
    print(" ", path_summary)
    print(" ", metric_path)
    print(" ", text_path)

except Exception as e:
    print("\nCould not save outputs:", repr(e))

# ============================================================
# INTERPRETATION GUIDE
# ============================================================

print("\n\n============================================================")
print("Audit-6B Interpretation Guide")
print("============================================================\n")

print("Strong symmetry preservation if:")
print("  1. paired center_cos is consistently higher than shuffled baseline.")
print("  2. cos_lift is positive across most layers.")
print("  3. L20-L22 boundary metrics remain high.")
print("  4. transformed paths have high agreement with clean path.")
print("  5. tau remains in L20-L22 and tau_shift is small.")
print()

print("Expected caveat:")
print("  TopK identity Jaccard may drop under entity renaming because entity labels change.")
print("  For rename, center_cos and path/tau preservation are more important than raw Jaccard.")
print()

print("Weak / failed symmetry preservation if:")
print("  1. paired metrics are no better than shuffled baseline.")
print("  2. tau shifts far away from L20-L22.")
print("  3. clean-to-transform path agreement collapses.")
print("  4. irrelevant or redundant facts cause stronger disruption than rename/permutation.")
print()

print("Audit-6C should only start after 6B establishes which transformations are truly preserving.")
print("Done.")