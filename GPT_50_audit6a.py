# ============================================================
# Audit-6A
# TopK Neighborhood Operator Program Test
#
# Goal:
#   Replace hidden operator:
#       H_l -> H_{l+1}
#
#   with vocabulary-induced neighborhood operator:
#       Z_l -> Z_{l+1}
#
#   where:
#       Z_l = [C_k(H_l), r_k(H_l)]
#
#   C_k = center of TopK token embeddings
#   r_k = spread radius of TopK token neighborhood
#
# Main tests:
#   1. Out-of-sample fit of Z_l -> Z_{l+1}
#   2. Operator clustering
#   3. Boundary/Bulk path
#   4. τ estimate
#   5. Whether VIM operators reproduce PMD/Audit-1 structure
# ============================================================

import os
import math
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
from scipy.spatial.distance import cosine

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

device = "cuda"
dtype = torch.float16

MAX_LEN = 160
BATCH_SIZE = 4

RANDOM_SEED = 42

# Audit-5B already showed TopK=5000 + center+spread is strongest.
TOPK = 5000

# Keep consistent with previous Audit-5A / Audit-5B code.
# If True, TopK is computed from layer-centered hidden states.
# If False, TopK is computed from raw hidden states.
CENTER_HIDDEN_FOR_TOPK = True

# For operator fitting
RIDGE_ALPHA = 1e-2
TEST_SIZE = 0.25

# For clustering operator signatures
K_LIST = [2, 3, 4, 5, 6, 8]
RANDOM_PROJ_DIM = 32
SPECTRUM_K = 64

# For τ estimate
TAU_SEARCH_START_RATIO = 0.45
OUTPUT_WINDOW = 5
MIN_OUTPUT_RUN = 2

# Output
SAVE_DIR = "./audit6a_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# TEXT DATA
# ============================================================
# Replace this with your 200+ natural semantic texts.
# Keep each item short / medium length. One sample = one hidden trajectory.

texts = [
    "Photosynthesis converts sunlight, water, and carbon dioxide into chemical energy.",
    "A proof by contradiction assumes the opposite statement and derives an impossibility.",
    "A compiler transforms source code into instructions that hardware can execute.",
    "The central bank raised interest rates after inflation remained above expectations.",
    "The old clock stopped during the storm.",
    "A protein folds into a shape determined by physical interactions.",
    "The database query became faster after an index was added.",
    "The storm damaged the harbor but spared most homes inland.",
    "A graph can represent roads, friendships, dependencies, or molecules.",
    "The expedition turned back when supplies ran low.",
    "A neuron integrates signals before producing an electrical spike.",
    "The server rejected the request because authentication failed.",
    "The doctor ordered additional tests because the first result was ambiguous.",
    "The child remembered the story but changed the ending.",
    "The algorithm converged after several thousand iterations.",
    "A market can react strongly to expectations rather than facts.",
    "A glacier records climate information in layers of ice.",
    "The programmer refactored the code to make future changes safer.",
    "The policy change caused prices to rise.",
    "The violinist paused before the final note.",
    "The proof used induction on the number of vertices.",
    "A battery stores chemical energy and releases it as electrical current.",
    "The economy slowed after exports declined.",
    "The village changed after the bridge was built.",
    "A classifier draws a decision boundary in feature space.",
    "A network router forwards packets based on destination addresses.",
    "The athlete changed training plans after a minor knee injury.",
    "The forest became quiet just before sunrise.",
    "An enzyme speeds up a reaction without being consumed.",
    "The server returned an error because authentication failed.",
    "A mathematical model simplifies reality by keeping only important variables.",
    "Astronomers infer stellar composition from absorption lines in spectra.",
    "The court ruling changed how agencies interpreted the rule.",
    "The traveler opened the door and found a hidden room.",
    "The theorem is stronger than the lemma.",
    "The processor executed instructions out of order for efficiency.",
    "Low temperature caused the battery to drain faster.",
    "The child built a tower and then knocked it down.",
    "The doctor compared the new scan with the earlier result.",
    "A volcano can remain dormant for centuries.",
    "A child found an old map inside a wooden box and tried to understand the symbols.",
    "The central bank raised interest rates after inflation remained above expectations.",
    "In a distributed database, consistency and availability often require trade-offs.",
    "The patient showed mild symptoms at first, but later developed a persistent fever.",
    "A proof by contradiction assumes the opposite statement and derives an impossibility.",
    "The concert began quietly with a solo violin before the orchestra entered.",
    "Engineers redesigned the cooling system after the prototype overheated.",
    "Ancient trade routes connected coastal cities with inland markets.",
    "A neural network may compress information into a small number of active directions.",
    "The spacecraft corrected its orbit using a precisely timed engine burn.",
    "Farmers changed irrigation schedules after weeks of unusually dry weather.",
    "The novel begins with a family argument at breakfast.",
    "A compiler transforms source code into instructions that hardware can execute.",
    "The immune system responds differently to bacteria, viruses, and allergens.",
    "The mountain trail became dangerous after heavy rain loosened the rocks.",
    "A legal precedent can influence how judges interpret later cases.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into chemical energy.",
    "The software update fixed several bugs but introduced a new performance issue.",
    "A historian compared letters, tax records, and maps to reconstruct the event.",
    "The robot used camera images and joint sensors to adjust its grip.",
    "A small change in initial conditions can alter the long-term weather forecast.",
    "The poet used repetition to create a sense of grief and memory.",
    "Quantum measurements reveal probabilities rather than fixed hidden values.",
    "The database query became faster after an index was added.",
    "The museum displayed tools, pottery, and fragments of woven fabric.",
    "A language model predicts the next token from context.",
    "The doctor ordered additional tests because the first result was ambiguous.",
    "A satellite can monitor forest fires from orbit.",
    "The orchestra struggled with timing during the final movement.",
    "Economic growth depends on productivity, investment, institutions, and trust.",
    "The river flooded after snow melted rapidly in the mountains.",
    "A camera sensor converts incoming photons into electrical signals.",
    "The teacher asked students to explain their reasoning rather than memorize formulas.",
    "In chess, a sacrifice can create long-term positional compensation.",
    "A bridge must distribute force across beams, cables, and foundations.",
    "The recipe failed because the dough was left too cold overnight.",
    "Astronomers infer stellar composition from absorption lines in spectra.",
    "The company delayed the launch after discovering a security vulnerability.",
    "The athlete changed training plans after a minor knee injury.",
    "A theorem can be elegant even when its proof is technically difficult.",
    "Machine translation often struggles with idioms and cultural context.",
    "The old clock stopped during the storm.",
    "A researcher measured reaction time across several experimental conditions.",
    "The city expanded public transit to reduce traffic congestion.",
    "A battery stores chemical energy and releases it as electrical current.",
    "The painting uses contrast to guide attention toward the central figure.",
    "A network router forwards packets based on destination addresses.",
    "The debate shifted after new evidence became public.",
    "A glacier moves slowly but can reshape an entire valley.",
    "The child learned the pattern after seeing only a few examples.",
    "A judge must distinguish relevant evidence from emotional appeal.",
    "The machine overheated because dust blocked the ventilation path.",
    "The melody returns in a different key near the end of the piece.",
    "A model can generalize poorly if trained on a narrow distribution.",
    "The expedition turned back when supplies ran low.",
    "The parliament passed the bill after several amendments.",
    "A neuron integrates signals before producing an electrical spike.",
    "The customer returned the device because the screen flickered.",
    "The forest became quiet just before sunrise.",
    "A mathematical model simplifies reality by keeping only important variables.",
    "The aircraft adjusted altitude to avoid turbulence.",
    "An enzyme speeds up a reaction without being consumed.",
    "The detective noticed that the witness avoided mentioning the time.",
    "A compression algorithm removes redundancy from data.",
    "The philosopher asked whether knowledge requires certainty.",
    "The storm damaged the harbor but spared most homes inland.",
    "The programmer refactored the code to make future changes safer.",
    "The experiment failed because the control group was not properly matched.",
    "A market can react strongly to expectations rather than facts.",
    "The violinist paused before the final note.",
    "The telescope detected faint light from a distant galaxy.",
    "The warehouse used robots to sort packages overnight.",
    "A constitution defines powers, limits, and procedures of government.",
    "The patient recovered slowly after surgery.",
    "The proof used induction on the number of vertices.",
    "A transformer model uses attention to mix information across tokens.",
    "The chef balanced sweetness with acidity.",
    "The student solved the problem by drawing a diagram.",
    "The shoreline changed after years of erosion.",
    "A social network spreads information through repeated local interactions.",
    "The engineer tested the material under heat and pressure.",
    "The bird changed direction when the wind shifted.",
    "A financial report can hide risk behind aggregate numbers.",
    "The scientist repeated the measurement after calibration.",
    "The actor delivered the line with unexpected restraint.",
    "A protein folds into a shape determined by physical interactions.",
    "The court considered intent, evidence, and statutory language.",
    "The classroom became silent when the lights went out.",
    "The algorithm converged after several thousand iterations.",
    "The historian warned against reading modern assumptions into ancient texts.",
    "The sensor failed because moisture entered the casing.",
    "The pianist practiced the difficult passage slowly.",
    "A species may adapt when environmental pressure changes.",
    "The server rejected the request because authentication failed.",
    "The mountain village relied on narrow roads during winter.",
    "A good explanation reduces confusion without hiding uncertainty.",
    "The telescope image was blurred by atmospheric turbulence.",
    "The child built a tower and then knocked it down.",
    "A medical trial must separate treatment effects from placebo effects.",
    "The river carried sediment toward the delta.",
    "The lawyer challenged the reliability of the witness.",
    "The model assigned high probability to a plausible but false answer.",
    "The spacecraft antenna unfolded after reaching orbit.",
    "The gardener moved the plant to a sunnier window.",
    "A low-rank update can change behavior using few parameters.",
    "The economy slowed after exports declined.",
    "The museum guide explained how pigments were made.",
    "The storm warning arrived too late for some boats.",
    "A graph can represent roads, friendships, dependencies, or molecules.",
    "The battery drained faster in cold weather.",
    "The poet revised one line many times.",
    "A doctor must sometimes act before all information is available.",
    "The student confused correlation with causation.",
    "The satellite image showed changes in vegetation.",
    "A language can preserve old grammar while changing vocabulary.",
    "The laboratory stored samples at very low temperature.",
    "The old bridge creaked under the weight of trucks.",
    "A classifier draws a decision boundary in feature space.",
    "The singer missed one note but recovered quickly.",
    "The court ruling changed how agencies interpreted the rule.",
    "A volcano can remain dormant for centuries.",
    "The robot hesitated when the object slipped.",
    "The researcher compared multiple baselines before drawing conclusions.",
    "The network became unstable after a routing loop formed.",
    "A child can infer a rule from surprisingly little data.",
    "The contract included a clause about unexpected delays.",
    "The desert temperature dropped sharply after sunset.",
    "The processor executed instructions out of order for efficiency.",
    "The team postponed the meeting after the outage.",
    "A poem can create meaning through rhythm as much as vocabulary.",
    "The fish migrated when water temperature changed.",
    "The diagnosis depended on both imaging and blood tests.",
    "The judge asked whether the argument applied to future cases.",
    "The model's hidden states may follow a low-dimensional trajectory.",
    "The engineer reduced vibration by changing the mounting structure.",
    "The historical record was incomplete and partly contradictory.",
    "A storm surge can push seawater far inland.",
    "The child remembered the story but changed the ending.",
    "The software crashed only when memory usage became high.",
    "A proof may reveal why a statement is true, not just that it is true.",
    "The company improved logistics by forecasting demand earlier.",
    "The immune response can damage healthy tissue if misdirected.",
    "The telescope needed long exposure to capture faint objects.",
    "A market bubble can grow when expectations reinforce themselves.",
    "The robot mapped the room before planning a path.",
    "The researcher removed outliers and reran the analysis.",
    "A glacier records climate information in layers of ice.",
    "The musician changed tempo to match the dancers.",
    "The lawyer prepared several alternative arguments.",
    "The model confused two entities with similar names.",
    "The bridge design changed after wind tunnel testing.",
    "A neural representation can be stable even while residual updates continue.",
    "The farmer repaired the fence after the storm.",
    "The philosopher distinguished belief from justified belief.",
    "The server scaled automatically when traffic increased.",
    "The student learned more from the mistake than from the correct answer.",

]

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

print("Loading model...")

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

def capture_hidden_states(texts):
    """
    Returns:
        H_raws: list length num_layers
                each array shape [N, d_model]
                raw last-token hidden states from every transformer block.
    """
    all_layer_states = [[] for _ in range(num_layers)]

    print("Capturing hidden states...")

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

            # hidden_states length = embedding + each block output
            hidden_states = outputs.hidden_states[1:]

            attn = inputs["attention_mask"]
            last_idx = attn.sum(dim=1) - 1

            for l in range(num_layers):
                h = hidden_states[l]  # [B, T, D]
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
        print(f"L{l:02d}: {H.shape}")

    return H_raws

H_raws = capture_hidden_states(texts)

N = H_raws[0].shape[0]
D = H_raws[0].shape[1]
print("Samples:", N)
print("Hidden dim:", D)

# ============================================================
# TOPK NEIGHBORHOOD STATE Z_l = [C_k, r_k]
# ============================================================

def compute_topk_ids(H, k):
    """
    Compute TopK token ids from H @ W^T.
    H shape: [N, D]
    Returns ids shape [N, k].
    """
    logits = H @ W_np.T  # [N, vocab]
    ids = np.argpartition(logits, -k, axis=1)[:, -k:]

    # Stable descending order, only for reproducibility.
    vals = np.take_along_axis(logits, ids, axis=1)
    order = np.argsort(vals, axis=1)[:, ::-1]
    ids = np.take_along_axis(ids, order, axis=1)

    return ids.astype(np.int64)

def compute_center_spread_from_ids(ids, batch_rows=4):
    """
    Identity-only representation:
      - no logit values
      - no probability
      - no rank score

    For each sample:
      C = normalized mean of TopK token embeddings
      r = mean cosine distance from C
    """
    n, k = ids.shape
    Z = np.zeros((n, d_model + 1), dtype=np.float32)

    for s in range(0, n, batch_rows):
        e = min(s + batch_rows, n)
        batch_ids = ids[s:e]

        # shape [B, K, D]
        E = W_norm[batch_ids]

        C = E.mean(axis=1)
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)

        # spread = mean(1 - cos(w_i, C))
        sims = np.einsum("bkd,bd->bk", E, C)
        spread = np.mean(1.0 - sims, axis=1, keepdims=True)

        Z[s:e, :d_model] = C
        Z[s:e, d_model:] = spread.astype(np.float32)

    return Z

def build_vim_states(H_raws):
    """
    Build Z_l = [C_k(H_l), r_k(H_l)] for all layers.
    """
    Zs = []

    print("\nBuilding VIM states...")
    print(f"TOPK={TOPK}, CENTER_HIDDEN_FOR_TOPK={CENTER_HIDDEN_FOR_TOPK}")

    for l, H_raw in enumerate(H_raws):
        if CENTER_HIDDEN_FOR_TOPK:
            H_for_topk = H_raw - H_raw.mean(axis=0, keepdims=True)
        else:
            H_for_topk = H_raw

        ids = compute_topk_ids(H_for_topk, TOPK)
        Z = compute_center_spread_from_ids(ids)

        # Center Z for geometry / fitting later will also use intercept.
        Zs.append(Z.astype(np.float32))

        print(
            f"L{l:02d}: Z={Z.shape}, "
            f"spread_mean={Z[:, -1].mean():.4f}, "
            f"spread_std={Z[:, -1].std():.4f}"
        )

    return Zs

Zs = build_vim_states(H_raws)

# ============================================================
# METRICS
# ============================================================

def r2_score_global(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))

def mean_row_cosine(y_true, y_pred):
    yt = y_true / (np.linalg.norm(y_true, axis=1, keepdims=True) + 1e-12)
    yp = y_pred / (np.linalg.norm(y_pred, axis=1, keepdims=True) + 1e-12)
    return float(np.mean(np.sum(yt * yp, axis=1)))

def effective_rank_from_singular_values(s):
    s = np.asarray(s, dtype=np.float64)
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
# DUAL RIDGE OPERATOR FIT
# ============================================================

def fit_dual_ridge_operator(X_train, Y_train, alpha=1e-2):
    """
    Fit Y ≈ X B + intercept using dual ridge.

    X_train: [n, d]
    Y_train: [n, d]

    Center X, Y:
        Xc = X - mx
        Yc = Y - my

    B = Xc.T @ inv(Xc Xc.T + alpha I) @ Yc

    Returns:
        B, mx, my
    """
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

# ============================================================
# FIT VIM OPERATORS
# ============================================================

indices = np.arange(N)
train_idx, test_idx = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    shuffle=True,
)

print("\nTrain samples:", len(train_idx))
print("Test samples :", len(test_idx))

operators = []
op_metrics = []

print("\n================ Audit-6A VIM Operator Fit ================\n")

print(
    f"{'Layer':<10}"
    f"{'TrainR2':<10}"
    f"{'TestR2':<10}"
    f"{'TrainCos':<10}"
    f"{'TestCos':<10}"
    f"{'EffRank':<10}"
    f"{'R90':<8}"
    f"{'R95':<8}"
)

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
    eff_rank = effective_rank_from_singular_values(S)
    r90 = r_energy_rank(S, 0.90)
    r95 = r_energy_rank(S, 0.95)

    operators.append({
        "layer": l,
        "B": B,
        "mx": mx,
        "my": my,
        "singular": S,
    })

    op_metrics.append({
        "layer": l,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_cos": train_cos,
        "test_cos": test_cos,
        "eff_rank": eff_rank,
        "r90": r90,
        "r95": r95,
    })

    print(
        f"L{l:02d}->{l+1:<5}"
        f"{train_r2:<10.4f}"
        f"{test_r2:<10.4f}"
        f"{train_cos:<10.4f}"
        f"{test_cos:<10.4f}"
        f"{eff_rank:<10.2f}"
        f"{r90:<8}"
        f"{r95:<8}"
    )

# ============================================================
# OPERATOR SIGNATURE
# ============================================================

def build_operator_signatures(operators, d, q=32, spectrum_k=64):
    """
    Signature includes:
      1. Random bilinear projection U^T B V
      2. Top singular spectrum
      3. Simple spectral statistics
    """
    rng = np.random.default_rng(RANDOM_SEED)

    U = rng.normal(size=(d, q)).astype(np.float32)
    V = rng.normal(size=(d, q)).astype(np.float32)

    U = U / (np.linalg.norm(U, axis=0, keepdims=True) + 1e-12)
    V = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12)

    sigs = []

    for op in operators:
        B = op["B"]
        S = op["singular"]

        # Random operator sketch
        sketch = U.T @ (B @ V)
        sketch = sketch.reshape(-1)

        # Spectrum features
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

D_Z = Zs[0].shape[1]
signatures = build_operator_signatures(
    operators,
    d=D_Z,
    q=RANDOM_PROJ_DIM,
    spectrum_k=SPECTRUM_K,
)

scaler = StandardScaler()
Xsig = scaler.fit_transform(signatures)

# ============================================================
# CLUSTERING
# ============================================================

print("\n================ Audit-6A VIM Operator Library ================\n")
print(f"{'K':<8}{'Inertia':<16}{'Silhouette':<12}")

cluster_results = []

for k in K_LIST:
    if k >= len(operators):
        continue

    km = KMeans(
        n_clusters=k,
        random_state=RANDOM_SEED,
        n_init=30,
    )

    labels = km.fit_predict(Xsig)

    if len(set(labels)) > 1:
        sil = silhouette_score(Xsig, labels)
    else:
        sil = -1.0

    cluster_results.append({
        "k": k,
        "inertia": float(km.inertia_),
        "silhouette": float(sil),
        "labels": labels,
        "model": km,
    })

    print(f"{k:<8}{km.inertia_:<16.4f}{sil:<12.4f}")

best = max(cluster_results, key=lambda x: x["silhouette"])
best_k = best["k"]
raw_labels = best["labels"]

print("\nBest VIM operator library:")
print(f"K={best_k}, silhouette={best['silhouette']:.4f}")

# ============================================================
# LABEL REMAP BY FIRST APPEARANCE
# ============================================================

def remap_by_first_appearance(labels):
    mapping = {}
    next_id = 0
    new_labels = []
    for x in labels:
        x = int(x)
        if x not in mapping:
            mapping[x] = next_id
            next_id += 1
        new_labels.append(mapping[x])
    return np.array(new_labels), mapping

labels, label_mapping = remap_by_first_appearance(raw_labels)

# ============================================================
# FAMILY SUMMARY
# ============================================================

print("\n================ VIM Operator Family Summary ================\n")

for c in sorted(set(labels)):
    idxs = np.where(labels == c)[0]
    mets = [op_metrics[i] for i in idxs]

    print(f"O{c:02d}")
    print(f"  size       : {len(idxs)}")
    print(f"  layer span : {int(idxs.min())} -> {int(idxs.max())}")
    print(f"  layers     : {list(map(int, idxs))}")
    print(f"  mean TestR2: {np.mean([m['test_r2'] for m in mets]):.4f}")
    print(f"  mean TestCos: {np.mean([m['test_cos'] for m in mets]):.4f}")
    print(f"  mean EffRank: {np.mean([m['eff_rank'] for m in mets]):.2f}")
    print()

# ============================================================
# PATH AND BOUNDARY / TAU
# ============================================================

def format_path(labels):
    return " ".join([f"O{x}" for x in labels])

print("\n================ VIM Operator Path ================\n")
print(format_path(labels))

changes = []
for i in range(1, len(labels)):
    if labels[i] != labels[i - 1]:
        changes.append(i)

print("\nLayer/operator changes:")
if len(changes) == 0:
    print("No label changes.")
else:
    for i in changes:
        print(
            f"  op index {i}: "
            f"L{i-1}->L{i} was O{labels[i-1]}, "
            f"L{i}->L{i+1} is O{labels[i]}"
        )

# Estimate output regime as dominant cluster in final window.
final_window = labels[-OUTPUT_WINDOW:]
output_cluster = Counter(final_window).most_common(1)[0][0]

start_search = int(len(labels) * TAU_SEARCH_START_RATIO)
tau = None

for i in range(start_search, len(labels)):
    if labels[i] == output_cluster:
        run = labels[i:i + MIN_OUTPUT_RUN]
        if len(run) >= MIN_OUTPUT_RUN and np.all(run == output_cluster):
            tau = i
            break

print("\n================ τ Estimate ================\n")
print(f"Output-regime cluster estimated from last {OUTPUT_WINDOW} ops: O{output_cluster}")
print(f"τ search starts at op index: {start_search}")

if tau is None:
    print("τ not detected with current persistence criterion.")
else:
    print(f"Estimated τ = {tau}")
    print(f"Meaning: first persistent entry into output regime at operator L{tau}->L{tau+1}")

# ============================================================
# ROLE INTERPRETATION
# ============================================================

def mode_of(arr):
    return Counter(arr).most_common(1)[0][0]

input_cluster = mode_of(labels[:min(3, len(labels))])
middle_start = max(3, int(len(labels) * 0.25))
middle_end = min(len(labels), int(len(labels) * 0.70))
bulk_cluster = mode_of(labels[middle_start:middle_end])
output_cluster = mode_of(labels[-OUTPUT_WINDOW:])

role = {}
role[input_cluster] = "InputBoundary"
role[bulk_cluster] = "Bulk"
role[output_cluster] = "OutputBoundary"

print("\n================ Boundary / Bulk Role Guess ================\n")
for c in sorted(set(labels)):
    print(f"O{c}: {role.get(c, 'Other')}")

role_path = " ".join([role.get(x, f"O{x}") for x in labels])
print("\nRole path:")
print(role_path)

# ============================================================
# SIMPLE DECISION GUIDE
# ============================================================

print("\n================ Audit-6A Decision Guide ================\n")

print("Strong support for Axiom-2 if:")
print("  1. VIM operator fit has non-trivial held-out TestCos / TestR2.")
print("  2. VIM operator clustering prefers small K, especially K=2 or K=3.")
print("  3. Path resembles Boundary/Bulk/Boundary.")
print("  4. τ appears near the previously observed L20-L22 zone.")
print("  5. VIM path qualitatively matches hidden-space PMD/Audit-1.")

print("\nWeak support / caution if:")
print("  1. TestR2 and TestCos collapse to random baseline.")
print("  2. KMeans silhouette is near zero for all K.")
print("  3. Path is noisy with no interpretable boundary.")
print("  4. τ cannot be detected or appears at arbitrary layers.")

# ============================================================
# SAVE RESULTS
# ============================================================

try:
    import pandas as pd

    df_metrics = pd.DataFrame(op_metrics)
    df_metrics["cluster"] = labels

    metrics_path = os.path.join(SAVE_DIR, "audit6a_operator_metrics.csv")
    df_metrics.to_csv(metrics_path, index=False)

    df_cluster = pd.DataFrame([
        {
            "k": r["k"],
            "inertia": r["inertia"],
            "silhouette": r["silhouette"],
        }
        for r in cluster_results
    ])

    cluster_path = os.path.join(SAVE_DIR, "audit6a_cluster_results.csv")
    df_cluster.to_csv(cluster_path, index=False)

    path_path = os.path.join(SAVE_DIR, "audit6a_path.txt")
    with open(path_path, "w", encoding="utf-8") as f:
        f.write("Audit-6A VIM Operator Path\n")
        f.write(format_path(labels) + "\n\n")
        f.write(f"Best K={best_k}, silhouette={best['silhouette']:.4f}\n")
        f.write(f"tau={tau}\n")
        f.write(f"output_cluster=O{output_cluster}\n")
        f.write("\nRole path:\n")
        f.write(role_path + "\n")

    print("\nSaved outputs:")
    print(" ", metrics_path)
    print(" ", cluster_path)
    print(" ", path_path)

except Exception as e:
    print("\nCould not save CSV outputs:", repr(e))

print("\nDone.")