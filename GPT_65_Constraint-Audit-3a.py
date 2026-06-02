# ============================================================
# Constraint-Audit-3A
# Support-Mass and Closure-Chain Feature Augmentation
#
# Fixed version:
#   - fixes lm_head fp16/fp32 dtype mismatch
#   - handles device_map="auto"
#   - validates augmented cache
#
# Input:
#   Prefer:
#     ./constraint_audit2c_outputs/constraint_audit2c_feature_table.csv
#   Fallback:
#     ./constraint_audit2b_outputs/constraint_audit2b_feature_table.csv
#
# Goal:
#   2C showed:
#     submechanism coordinates exist,
#     but LOFO generalization is unstable.
#
#   3A asks:
#     Does adding support-mass / marker-mass / TopK persistence
#     improve cross-template mechanism generalization?
#
# New feature families:
#   1. support mass:
#        clean label embedding-neighborhood mass
#        conflict label embedding-neighborhood mass
#
#   2. closure / competition / stability marker mass:
#        hidden readout mass toward marker-token clusters
#
#   3. freedom:
#        TopK entropy
#        TopK spread
#        clean/conflict support diversity
#
#   4. topology persistence:
#        Jaccard(TopK_l, TopK_{l+1})
#
#   5. phase-window aggregation:
#        ENTRY20_22
#        BASIN23_25
#        COMMIT26_27
#        ALL20_25
#
# Outputs:
#   ./constraint_audit3a_outputs/
# ============================================================

import gc
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

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

INPUT_2C = Path("./constraint_audit2c_outputs/constraint_audit2c_feature_table.csv")
INPUT_2B = Path("./constraint_audit2b_outputs/constraint_audit2b_feature_table.csv")

SAVE_DIR = Path("./constraint_audit3a_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

MAX_LEN = 300
BATCH_SIZE = 4

# If memory/time is tight, use 500 or 1000.
# Later you can raise to 2000 or 5000.
TOPK_FEATURE = 1000

LABEL_CLUSTER_K = 128
MARKER_CLUSTER_K = 128

TRACK_LAYERS = [20, 21, 22, 23, 24, 25, 26, 27]

WINDOWS = {
    "ENTRY20_22": [20, 21, 22],
    "BASIN23_25": [23, 24, 25],
    "COMMIT26_27": [26, 27],
    "ALL20_25": [20, 21, 22, 23, 24, 25],
    "ALL20_27": [20, 21, 22, 23, 24, 25, 26, 27],
}

N_GROUP_SPLITS = 5

# ============================================================
# LABEL MAPS FROM 2C
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

MARKER_GROUPS = {
    "stable_marker": [
        "fact", "verified", "record", "redundant", "valid", "authoritative"
    ],
    "competition_marker": [
        "conflicting", "ambiguous", "also", "may", "claim", "source", "another"
    ],
    "closure_marker": [
        "updated", "changed", "current", "now", "not", "latest", "registry"
    ],
    "override_marker": [
        "override", "exception", "rule", "case", "use", "valid"
    ],
    "authority_marker": [
        "verified", "registry", "authority", "official", "confirmed"
    ],
}

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


def best_threshold(y_cal, p_cal):
    y_cal = np.asarray(y_cal).astype(int)
    p_cal = sanitize_prob(p_cal)

    thresholds = np.unique(np.quantile(p_cal, np.linspace(0.01, 0.99, 99)))

    best_t = 0.5
    best_f1 = -1.0

    for t in thresholds:
        f1 = calc_binary_metrics(y_cal, p_cal, threshold=float(t))["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    return best_t


def jaccard_np(a, b):
    a = set(map(int, a))
    b = set(map(int, b))
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


# ============================================================
# LOAD DATA
# ============================================================

if INPUT_2C.exists():
    INPUT_FILE = INPUT_2C
elif INPUT_2B.exists():
    INPUT_FILE = INPUT_2B
else:
    raise FileNotFoundError(
        "Cannot find 2C or 2B feature table. Run Constraint-Audit-2B/2C first."
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

required_cols = ["prompt", "family", "idx", "clean_token_id", "conflict_token_id"]
missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise RuntimeError(
        "Input feature table lacks required columns: "
        + ", ".join(missing)
        + "\nUse the full 2B/2C feature table generated by earlier scripts."
    )

df["mechanism"] = df["family"].map(COARSE_MECHANISM_MAP)
df["submechanism"] = df["family"].map(SUBMECHANISM_MAP)
df["risk_regime"] = df["family"].map(RISK_REGIME_MAP)

if df["submechanism"].isna().any():
    bad = df[df["submechanism"].isna()]["family"].unique().tolist()
    raise RuntimeError(f"Unknown family values: {bad}")

df = df.reset_index(drop=True)

print("\n============================================================")
print("Constraint-Audit-3A Input Summary")
print("============================================================\n")
print("Input:", INPUT_FILE)
print("Rows:", len(df))
print("Families:", df["family"].nunique())
print("TopK feature:", TOPK_FEATURE)
print("Device:", DEVICE)

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

if max(TRACK_LAYERS) >= num_layers:
    raise RuntimeError(
        f"TRACK_LAYERS includes L{max(TRACK_LAYERS)}, model has {num_layers} layers."
    )

print("Num layers:", num_layers)
print("LM head:", tuple(model.lm_head.weight.shape))
print("LM head dtype:", model.lm_head.weight.dtype)
print("LM head device:", model.lm_head.weight.device)

# CPU copy of LM head for cluster construction / spread.
W_cpu = model.lm_head.weight.detach().float().cpu()
W_norm_cpu = F.normalize(W_cpu, p=2, dim=1)

vocab_size = W_cpu.shape[0]
hidden_dim = W_cpu.shape[1]


def lm_head_forward(hidden):
    """
    Robust lm_head forward:
    hidden state may be fp32/fp16 and may sit on a different device
    when device_map='auto' is used.
    """
    w = model.lm_head.weight
    hidden = hidden.to(device=w.device, dtype=w.dtype)
    return model.lm_head(hidden).float()


# ============================================================
# TOKEN CLUSTERS
# ============================================================

def tokenize_word_ids(word):
    ids = tokenizer(
        " " + word,
        add_special_tokens=False,
        return_tensors=None,
    )["input_ids"]
    return [int(x) for x in ids]


@torch.no_grad()
def nearest_cluster_from_token_ids(token_ids, k):
    token_ids = [int(x) for x in token_ids if 0 <= int(x) < vocab_size]

    if len(token_ids) == 0:
        return np.array([], dtype=np.int64)

    vec = W_norm_cpu[token_ids].mean(dim=0)
    vec = F.normalize(vec, dim=0)

    sims = torch.mv(W_norm_cpu, vec)
    top = torch.topk(
        sims,
        k=min(k, vocab_size),
        largest=True,
    ).indices.cpu().numpy()

    merged = list(dict.fromkeys(list(map(int, token_ids)) + list(map(int, top))))
    return np.array(merged, dtype=np.int64)


unique_label_token_ids = sorted(
    set(df["clean_token_id"].astype(int).tolist())
    | set(df["conflict_token_id"].astype(int).tolist())
)

label_cluster = {}
for tid in unique_label_token_ids:
    label_cluster[int(tid)] = nearest_cluster_from_token_ids([tid], LABEL_CLUSTER_K)

marker_clusters = {}
for group, words in MARKER_GROUPS.items():
    seed_ids = []
    for w in words:
        seed_ids.extend(tokenize_word_ids(w))
    marker_clusters[group] = nearest_cluster_from_token_ids(seed_ids, MARKER_CLUSTER_K)

print("\nCluster summary:")
print("  label token clusters:", len(label_cluster))
for group, ids in marker_clusters.items():
    print(f"  {group:<20} size={len(ids)}")

# ============================================================
# AUGMENTED FEATURE EXTRACTION
# ============================================================

def log_mass_for_ids(logits_1d, ids, log_denom):
    if len(ids) == 0:
        return -30.0, 0.0

    idx = torch.tensor(
        ids,
        dtype=torch.long,
        device=logits_1d.device,
    )

    val = torch.logsumexp(logits_1d[idx], dim=0) - log_denom
    mass = float(torch.exp(val).detach().cpu().item())

    return float(val.detach().cpu().item()), mass


def normalized_topk_entropy(topk_logits):
    p = torch.softmax(topk_logits.float(), dim=-1)
    h = -(p * torch.log(p + 1e-12)).sum(dim=-1)
    h = h / np.log(topk_logits.shape[-1])
    return h.detach().cpu().numpy().astype(np.float32)


def topk_spread_from_ids(top_ids_np):
    emb = W_norm_cpu[top_ids_np]
    center = emb.mean(dim=0)
    center = F.normalize(center, dim=0)
    sims = torch.mv(emb, center)
    spread = (1.0 - sims).mean().item()
    return float(spread)


def validate_augmented_cache(path, n_expected):
    if not path.exists():
        return False

    try:
        tmp = pd.read_csv(path, nrows=5)
        full_rows = sum(1 for _ in open(path, "r", encoding="utf-8")) - 1
    except Exception:
        return False

    if full_rows != n_expected:
        print(f"Cache row mismatch: got {full_rows}, expected {n_expected}. Regenerating.")
        return False

    required = [
        "AUG_ans_margin_L20",
        "AUG_clean_mass_L20",
        "AUG_conflict_mass_L20",
        "AUG_support_logratio_L20",
        "AUG_topk_entropy_L20",
        "AUG_topk_spread_L20",
        "AUG_topk_jaccard_L20_21",
    ]

    for c in required:
        if c not in tmp.columns:
            print(f"Cache missing column {c}. Regenerating.")
            return False

    return True


def extract_augmented_features(df_in):
    n = len(df_in)
    prompts = df_in["prompt"].tolist()

    aug = pd.DataFrame(index=np.arange(n))

    topk_by_layer = {
        l: [None for _ in range(n)]
        for l in TRACK_LAYERS
    }

    clean_ids_all = df_in["clean_token_id"].astype(int).values
    conflict_ids_all = df_in["conflict_token_id"].astype(int).values

    for l in TRACK_LAYERS:
        for name in [
            "ans_margin",
            "clean_mass",
            "conflict_mass",
            "support_logratio",
            "clean_support_count",
            "conflict_support_count",
            "support_count_margin",
            "clean_rank",
            "conflict_rank",
            "rank_gap_E_minus_C",
            "topk_entropy",
            "topk_spread",
        ]:
            aug[f"AUG_{name}_L{l}"] = 0.0

        for group in MARKER_GROUPS:
            aug[f"AUG_mass_{group}_L{l}"] = 0.0
            aug[f"AUG_logmass_{group}_L{l}"] = 0.0

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

            hidden_states = outputs.hidden_states

            bsz = end - start
            row_indices = np.arange(start, end)

            for l in TRACK_LAYERS:
                # hidden_states[0] = embedding output
                # hidden_states[l + 1] = output after transformer layer l
                h = hidden_states[l + 1][:, -1, :]

                # Important: dtype/device-safe lm_head.
                logits = lm_head_forward(h)

                clean_t = torch.tensor(
                    clean_ids_all[row_indices],
                    dtype=torch.long,
                    device=logits.device,
                )
                conflict_t = torch.tensor(
                    conflict_ids_all[row_indices],
                    dtype=torch.long,
                    device=logits.device,
                )

                clean_logit = logits.gather(1, clean_t.view(-1, 1)).squeeze(1)
                conflict_logit = logits.gather(1, conflict_t.view(-1, 1)).squeeze(1)

                ans_margin = (clean_logit - conflict_logit).detach().cpu().numpy()

                clean_rank = (logits > clean_logit.view(-1, 1)).sum(dim=1) + 1
                conflict_rank = (logits > conflict_logit.view(-1, 1)).sum(dim=1) + 1

                clean_rank_np = clean_rank.detach().cpu().numpy().astype(np.float32)
                conflict_rank_np = conflict_rank.detach().cpu().numpy().astype(np.float32)

                # positive means conflict token ranks better than clean token
                rank_gap = clean_rank_np - conflict_rank_np

                topk_vals, topk_ids = torch.topk(
                    logits,
                    k=min(TOPK_FEATURE, logits.shape[-1]),
                    dim=-1,
                    largest=True,
                )

                topk_entropy = normalized_topk_entropy(topk_vals)
                log_denom = torch.logsumexp(logits, dim=-1)

                for bi in range(bsz):
                    ridx = start + bi

                    ids_np = topk_ids[bi].detach().cpu().numpy().astype(np.int64)
                    topk_by_layer[l][ridx] = ids_np

                    clean_tid = int(clean_ids_all[ridx])
                    conflict_tid = int(conflict_ids_all[ridx])

                    clean_cluster = label_cluster[clean_tid]
                    conflict_cluster = label_cluster[conflict_tid]

                    clean_logmass, clean_mass = log_mass_for_ids(
                        logits[bi],
                        clean_cluster,
                        log_denom[bi],
                    )
                    conflict_logmass, conflict_mass = log_mass_for_ids(
                        logits[bi],
                        conflict_cluster,
                        log_denom[bi],
                    )

                    top_set = set(map(int, ids_np))
                    clean_count = len(top_set & set(map(int, clean_cluster)))
                    conflict_count = len(top_set & set(map(int, conflict_cluster)))

                    aug.loc[ridx, f"AUG_ans_margin_L{l}"] = float(ans_margin[bi])
                    aug.loc[ridx, f"AUG_clean_mass_L{l}"] = clean_mass
                    aug.loc[ridx, f"AUG_conflict_mass_L{l}"] = conflict_mass
                    aug.loc[ridx, f"AUG_support_logratio_L{l}"] = clean_logmass - conflict_logmass
                    aug.loc[ridx, f"AUG_clean_support_count_L{l}"] = float(clean_count)
                    aug.loc[ridx, f"AUG_conflict_support_count_L{l}"] = float(conflict_count)
                    aug.loc[ridx, f"AUG_support_count_margin_L{l}"] = float(clean_count - conflict_count)
                    aug.loc[ridx, f"AUG_clean_rank_L{l}"] = float(clean_rank_np[bi])
                    aug.loc[ridx, f"AUG_conflict_rank_L{l}"] = float(conflict_rank_np[bi])
                    aug.loc[ridx, f"AUG_rank_gap_E_minus_C_L{l}"] = float(rank_gap[bi])
                    aug.loc[ridx, f"AUG_topk_entropy_L{l}"] = float(topk_entropy[bi])
                    aug.loc[ridx, f"AUG_topk_spread_L{l}"] = topk_spread_from_ids(ids_np)

                    for group, mids in marker_clusters.items():
                        logm, mass = log_mass_for_ids(
                            logits[bi],
                            mids,
                            log_denom[bi],
                        )
                        aug.loc[ridx, f"AUG_mass_{group}_L{l}"] = mass
                        aug.loc[ridx, f"AUG_logmass_{group}_L{l}"] = logm

            if start % (BATCH_SIZE * 10) == 0:
                print(f"  extracted {end}/{n}")

            del outputs, hidden_states, inputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for i in range(len(TRACK_LAYERS) - 1):
        l1 = TRACK_LAYERS[i]
        l2 = TRACK_LAYERS[i + 1]

        vals = []
        for ridx in range(n):
            vals.append(
                jaccard_np(
                    topk_by_layer[l1][ridx],
                    topk_by_layer[l2][ridx],
                )
            )

        aug[f"AUG_topk_jaccard_L{l1}_{l2}"] = np.array(vals, dtype=np.float32)

    return aug


aug_cache = SAVE_DIR / "constraint_audit3a_augmented_only.csv"

if validate_augmented_cache(aug_cache, len(df)):
    print("\nLoading cached augmented features:", aug_cache)
    aug_df = pd.read_csv(aug_cache)
else:
    print("\nExtracting augmented C features...")
    aug_df = extract_augmented_features(df)
    aug_df.to_csv(aug_cache, index=False)
    print("Saved augmented features:", aug_cache)

df_aug = pd.concat(
    [df.reset_index(drop=True), aug_df.reset_index(drop=True)],
    axis=1,
)

# ============================================================
# WINDOW AGGREGATION
# ============================================================

def add_window_aggregates(df_in):
    out = df_in.copy()

    layer_feature_prefixes = []

    for c in out.columns:
        if c.startswith("AUG_") and "_L" in c:
            last = c.rsplit("_L", 1)[-1]
            if last.isdigit():
                prefix = c.rsplit("_L", 1)[0]
                layer_feature_prefixes.append(prefix)

    layer_feature_prefixes = sorted(set(layer_feature_prefixes))

    for win_name, layers in WINDOWS.items():
        for prefix in layer_feature_prefixes:
            cols = [
                f"{prefix}_L{l}"
                for l in layers
                if f"{prefix}_L{l}" in out.columns
            ]

            if len(cols) == 0:
                continue

            X = out[cols].values.astype(np.float32)

            out[f"{prefix}_{win_name}_mean"] = X.mean(axis=1)
            out[f"{prefix}_{win_name}_min"] = X.min(axis=1)
            out[f"{prefix}_{win_name}_max"] = X.max(axis=1)
            out[f"{prefix}_{win_name}_last"] = X[:, -1]
            out[f"{prefix}_{win_name}_slope"] = X[:, -1] - X[:, 0]
            out[f"{prefix}_{win_name}_range"] = X.max(axis=1) - X.min(axis=1)

    j_cols = [c for c in out.columns if c.startswith("AUG_topk_jaccard_L")]

    if j_cols:
        j20_25 = [
            c for c in j_cols
            if any(x in c for x in ["L20_21", "L21_22", "L22_23", "L23_24", "L24_25"])
        ]

        if len(j20_25) > 0:
            out["AUG_topk_jaccard_20_25_mean"] = out[j20_25].mean(axis=1)

        out["AUG_topk_jaccard_20_27_mean"] = out[j_cols].mean(axis=1)

    return out


df_aug = add_window_aggregates(df_aug)

# ============================================================
# BASELINE FEATURE CONSTRUCTION
# ============================================================

def ensure_baseline_segment_features(df_in):
    out = df_in.copy()

    segs = {
        "ENTRY20_22_BASE": [20, 21, 22],
        "BASIN23_25_BASE": [23, 24, 25],
        "ALL20_25_BASE": [20, 21, 22, 23, 24, 25],
    }

    for seg, layers in segs.items():
        r_cols = [f"R_L{l}" for l in layers if f"R_L{l}" in out.columns]
        h_cols = [f"H_L{l}" for l in layers if f"H_L{l}" in out.columns]
        p_cols = [f"pC_L{l}" for l in layers if f"pC_L{l}" in out.columns]

        for name, cols in [("R", r_cols), ("H", h_cols), ("pC", p_cols)]:
            if len(cols) == 0:
                continue

            X = out[cols].values.astype(np.float32)

            out[f"{seg}_{name}_mean"] = X.mean(axis=1)
            out[f"{seg}_{name}_min"] = X.min(axis=1)
            out[f"{seg}_{name}_max"] = X.max(axis=1)
            out[f"{seg}_{name}_last"] = X[:, -1]
            out[f"{seg}_{name}_slope"] = X[:, -1] - X[:, 0]
            out[f"{seg}_{name}_range"] = X.max(axis=1) - X.min(axis=1)

    return out


df_aug = ensure_baseline_segment_features(df_aug)

baseline_cols = []

for l in [20, 21, 22, 23, 24, 25]:
    for name in ["R", "H", "pC"]:
        c = f"{name}_L{l}"
        if c in df_aug.columns:
            baseline_cols.append(c)

baseline_cols += [
    c for c in df_aug.columns
    if c.startswith("ENTRY20_22_BASE")
    or c.startswith("BASIN23_25_BASE")
    or c.startswith("ALL20_25_BASE")
]

augmented_cols = baseline_cols + [
    c for c in df_aug.columns
    if c.startswith("AUG_")
]


def clean_feature_cols(df_in, cols):
    seen = set()
    out = []

    for c in cols:
        if c in seen:
            continue

        seen.add(c)

        if c not in df_in.columns:
            continue

        if pd.api.types.is_numeric_dtype(df_in[c]):
            out.append(c)

    return out


baseline_cols = clean_feature_cols(df_aug, baseline_cols)
augmented_cols = clean_feature_cols(df_aug, augmented_cols)

for c in augmented_cols:
    df_aug[c] = pd.to_numeric(df_aug[c], errors="coerce")

df_aug[augmented_cols] = (
    df_aug[augmented_cols]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)

print("\nFeature sets:")
print("  baseline_cols:", len(baseline_cols))
print("  augmented_cols:", len(augmented_cols))

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
            max_iter=3000,
        )),
    ])


def make_multiclass_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            max_iter=5000,
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


def aligned_proba(model, X, all_classes):
    p_raw = model.predict_proba(X)

    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    else:
        classes = list(model.named_steps["clf"].classes_)

    out = np.zeros((len(X), len(all_classes)), dtype=np.float64)

    for j, cls in enumerate(classes):
        if cls in all_classes:
            out[:, all_classes.index(cls)] = p_raw[:, j]

    row_sum = out.sum(axis=1, keepdims=True)
    bad = row_sum[:, 0] <= 1e-12

    if bad.any():
        out[bad, :] = 1.0 / len(all_classes)
        row_sum = out.sum(axis=1, keepdims=True)

    return out / (row_sum + 1e-12)

# ============================================================
# EVALUATION
# ============================================================

def eval_binary_groupcv(df_in, cols, name):
    X = df_in[cols].values.astype(np.float32)
    y = df_in["y_conflict"].values.astype(int)
    groups = df_in["idx"].values

    gkf = GroupKFold(n_splits=N_GROUP_SPLITS)
    p = np.zeros(len(df_in), dtype=np.float64)

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        model_sklearn = fit_binary_or_constant(X[train_idx], y[train_idx])
        p[test_idx] = sanitize_prob(model_sklearn.predict_proba(X[test_idx])[:, 1])

    row = {
        "feature_set": name,
        "eval": "GroupKFold_binary",
    }
    row.update(calc_binary_metrics(y, p, threshold=0.5))

    return row


def eval_multiclass_groupcv(df_in, cols, target_col, classes, name):
    X = df_in[cols].values.astype(np.float32)
    y = df_in[target_col].values
    groups = df_in["idx"].values

    gkf = GroupKFold(n_splits=N_GROUP_SPLITS)
    pred = np.empty(len(df_in), dtype=object)

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        model_sklearn = fit_multiclass_or_constant(X[train_idx], y[train_idx])
        pred[test_idx] = model_sklearn.predict(X[test_idx])

    return {
        "feature_set": name,
        "target": target_col,
        "eval": "GroupKFold_multiclass",
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def eval_binary_lofo(df_in, cols, name):
    X = df_in[cols].values.astype(np.float32)
    y = df_in["y_conflict"].values.astype(int)
    families = df_in["family"].values

    logo = LeaveOneGroupOut()

    rows = []
    pred_rows = []

    for train_idx, test_idx in logo.split(X, y, groups=families):
        heldout = families[test_idx][0]

        model_sklearn = fit_binary_or_constant(X[train_idx], y[train_idx])
        p_test = sanitize_prob(model_sklearn.predict_proba(X[test_idx])[:, 1])

        m = calc_binary_metrics(y[test_idx], p_test, threshold=0.5)

        row = {
            "feature_set": name,
            "heldout_family": heldout,
            "heldout_submechanism": SUBMECHANISM_MAP[heldout],
            "heldout_risk_regime": RISK_REGIME_MAP[heldout],
            "positive_rate": float(y[test_idx].mean()),
        }
        row.update(m)
        rows.append(row)

        temp = df_in.iloc[test_idx][[
            "idx",
            "family",
            "submechanism",
            "risk_regime",
            "y_conflict",
        ]].copy()

        temp["feature_set"] = name
        temp["p_conflict"] = p_test
        pred_rows.append(temp)

    return pd.DataFrame(rows), pd.concat(pred_rows, axis=0, ignore_index=True)


def eval_multiclass_lofo(df_in, cols, target_col, classes, name):
    X = df_in[cols].values.astype(np.float32)
    y = df_in[target_col].values
    families = df_in["family"].values

    logo = LeaveOneGroupOut()

    rows = []
    pred_rows = []

    for train_idx, test_idx in logo.split(X, y, groups=families):
        heldout = families[test_idx][0]
        true_label = y[test_idx][0]
        train_has = true_label in set(y[train_idx])

        model_sklearn = fit_multiclass_or_constant(X[train_idx], y[train_idx])
        p = aligned_proba(model_sklearn, X[test_idx], classes)
        pred = np.array(classes)[np.argmax(p, axis=1)]

        row = {
            "feature_set": name,
            "target": target_col,
            "heldout_family": heldout,
            "true_label": true_label,
            "train_has_true_label": bool(train_has),
            "accuracy": float(accuracy_score(y[test_idx], pred)),
            "macro_f1": float(f1_score(y[test_idx], pred, average="macro", zero_division=0)),
        }

        for j, cls in enumerate(classes):
            row[f"mean_p_{cls}"] = float(p[:, j].mean())

        rows.append(row)

        temp = df_in.iloc[test_idx][[
            "idx",
            "family",
            "submechanism",
            "risk_regime",
            "y_conflict",
        ]].copy()

        temp["feature_set"] = name
        temp["target"] = target_col
        temp["pred"] = pred

        for j, cls in enumerate(classes):
            temp[f"p_{cls}"] = p[:, j]

        pred_rows.append(temp)

    return pd.DataFrame(rows), pd.concat(pred_rows, axis=0, ignore_index=True)


# ============================================================
# RUN EVALUATIONS
# ============================================================

feature_sets = {
    "baseline_RHpC": baseline_cols,
    "augmented_C3A": augmented_cols,
}

binary_group_rows = []
multi_group_rows = []

binary_lofo_all = []
binary_lofo_pred_all = []
multi_lofo_all = []
multi_lofo_pred_all = []

for fs_name, cols in feature_sets.items():
    print(f"\nEvaluating feature set: {fs_name} | n_features={len(cols)}")

    binary_group_rows.append(
        eval_binary_groupcv(df_aug, cols, fs_name)
    )

    multi_group_rows.append(
        eval_multiclass_groupcv(
            df_aug,
            cols,
            target_col="submechanism",
            classes=SUBMECHANISM_CLASSES,
            name=fs_name,
        )
    )

    multi_group_rows.append(
        eval_multiclass_groupcv(
            df_aug,
            cols,
            target_col="risk_regime",
            classes=RISK_REGIME_CLASSES,
            name=fs_name,
        )
    )

    blofo, blofo_pred = eval_binary_lofo(df_aug, cols, fs_name)
    binary_lofo_all.append(blofo)
    binary_lofo_pred_all.append(blofo_pred)

    slofo, slofo_pred = eval_multiclass_lofo(
        df_aug,
        cols,
        target_col="submechanism",
        classes=SUBMECHANISM_CLASSES,
        name=fs_name,
    )
    multi_lofo_all.append(slofo)
    multi_lofo_pred_all.append(slofo_pred)

    rlofo, rlofo_pred = eval_multiclass_lofo(
        df_aug,
        cols,
        target_col="risk_regime",
        classes=RISK_REGIME_CLASSES,
        name=fs_name,
    )
    multi_lofo_all.append(rlofo)
    multi_lofo_pred_all.append(rlofo_pred)

binary_group_df = pd.DataFrame(binary_group_rows)
multi_group_df = pd.DataFrame(multi_group_rows)

binary_lofo_df = pd.concat(binary_lofo_all, axis=0, ignore_index=True)
binary_lofo_pred_df = pd.concat(binary_lofo_pred_all, axis=0, ignore_index=True)

multi_lofo_df = pd.concat(multi_lofo_all, axis=0, ignore_index=True)
multi_lofo_pred_df = pd.concat(multi_lofo_pred_all, axis=0, ignore_index=True)

binary_lofo_summary = []

for fs, sub in binary_lofo_df.groupby("feature_set"):
    binary_lofo_summary.append({
        "feature_set": fs,
        "mean_auc": float(np.nanmean(sub["auc"])),
        "mean_accuracy": float(np.nanmean(sub["accuracy"])),
        "mean_f1": float(np.nanmean(sub["f1"])),
        "mean_brier": float(np.nanmean(sub["brier"])),
        "mean_pred_pos_rate": float(np.nanmean(sub["pred_pos_rate"])),
    })

binary_lofo_summary = pd.DataFrame(binary_lofo_summary)

multi_lofo_summary = []

for keys, sub in multi_lofo_df.groupby(["feature_set", "target"]):
    fs, target = keys
    multi_lofo_summary.append({
        "feature_set": fs,
        "target": target,
        "mean_accuracy": float(np.nanmean(sub["accuracy"])),
        "mean_macro_f1": float(np.nanmean(sub["macro_f1"])),
    })

multi_lofo_summary = pd.DataFrame(multi_lofo_summary)


def fit_full_binary_coefficients(df_in, cols, name):
    X = df_in[cols].values.astype(np.float32)
    y = df_in["y_conflict"].values.astype(int)

    model_sklearn = fit_binary_or_constant(X, y)

    if not isinstance(model_sklearn, Pipeline):
        return pd.DataFrame()

    coef = model_sklearn.named_steps["clf"].coef_[0]

    rows = []
    for feat, val in sorted(zip(cols, coef), key=lambda x: abs(x[1]), reverse=True):
        rows.append({
            "feature_set": name,
            "feature": feat,
            "coef": float(val),
            "abs_coef": float(abs(val)),
        })

    return pd.DataFrame(rows)


coef_df = fit_full_binary_coefficients(df_aug, augmented_cols, "augmented_C3A")

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n\n============================================================")
print("Constraint-Audit-3A GroupKFold Binary")
print("============================================================\n")

print(
    binary_group_df[[
        "feature_set",
        "auc",
        "accuracy",
        "f1",
        "brier",
        "pred_pos_rate",
    ]].to_string(index=False)
)

print("\n\n============================================================")
print("Constraint-Audit-3A GroupKFold Multiclass")
print("============================================================\n")

print(multi_group_df.to_string(index=False))

print("\n\n============================================================")
print("Constraint-Audit-3A LOFO Binary Summary")
print("============================================================\n")

print(binary_lofo_summary.to_string(index=False))

print("\n\n============================================================")
print("Constraint-Audit-3A LOFO Multiclass Summary")
print("============================================================\n")

print(multi_lofo_summary.to_string(index=False))

print("\n\n============================================================")
print("Most Important Augmented Binary Features")
print("============================================================\n")

print(coef_df.head(30).to_string(index=False))

# ============================================================
# SAVE
# ============================================================

augmented_table_path = SAVE_DIR / "constraint_audit3a_augmented_feature_table.csv"

binary_group_path = SAVE_DIR / "constraint_audit3a_group_binary.csv"
multi_group_path = SAVE_DIR / "constraint_audit3a_group_multiclass.csv"

binary_lofo_path = SAVE_DIR / "constraint_audit3a_lofo_binary.csv"
binary_lofo_summary_path = SAVE_DIR / "constraint_audit3a_lofo_binary_summary.csv"
binary_lofo_pred_path = SAVE_DIR / "constraint_audit3a_lofo_binary_predictions.csv"

multi_lofo_path = SAVE_DIR / "constraint_audit3a_lofo_multiclass.csv"
multi_lofo_summary_path = SAVE_DIR / "constraint_audit3a_lofo_multiclass_summary.csv"
multi_lofo_pred_path = SAVE_DIR / "constraint_audit3a_lofo_multiclass_predictions.csv"

coef_path = SAVE_DIR / "constraint_audit3a_augmented_binary_coefficients.csv"

df_aug.to_csv(augmented_table_path, index=False)

binary_group_df.to_csv(binary_group_path, index=False)
multi_group_df.to_csv(multi_group_path, index=False)

binary_lofo_df.to_csv(binary_lofo_path, index=False)
binary_lofo_summary.to_csv(binary_lofo_summary_path, index=False)
binary_lofo_pred_df.to_csv(binary_lofo_pred_path, index=False)

multi_lofo_df.to_csv(multi_lofo_path, index=False)
multi_lofo_summary.to_csv(multi_lofo_summary_path, index=False)
multi_lofo_pred_df.to_csv(multi_lofo_pred_path, index=False)

coef_df.to_csv(coef_path, index=False)

print("\nSaved outputs:")
print(" ", augmented_table_path)
print(" ", binary_group_path)
print(" ", multi_group_path)
print(" ", binary_lofo_path)
print(" ", binary_lofo_summary_path)
print(" ", binary_lofo_pred_path)
print(" ", multi_lofo_path)
print(" ", multi_lofo_summary_path)
print(" ", multi_lofo_pred_path)
print(" ", coef_path)

# ============================================================
# INTERPRETATION GUIDE
# ============================================================

print("\n\n============================================================")
print("Constraint-Audit-3A Interpretation Guide")
print("============================================================\n")

print("Strong positive result if:")
print("  1. augmented_C3A improves LOFO submechanism macro-F1 over baseline_RHpC.")
print("  2. augmented_C3A improves LOFO risk_regime macro-F1.")
print("  3. stable_surface_shift no longer collapses into stable_core.")
print("  4. competition_high_commit vs competition_low_commit improves.")
print("  5. closure_rule_override vs closure_canonical_rewrite improves.")
print()
print("If GroupKFold improves but LOFO does not:")
print("  Added features help in-distribution explanation but not template generalization.")
print()
print("If support_logratio / support_mass features dominate coefficients:")
print("  Answer-basin support mass is the missing component.")
print()
print("If marker-mass features dominate:")
print("  Closure / source / override semantic markers are being encoded in W20-W25.")
print()
print("If topk_jaccard / entropy / spread dominate:")
print("  Freedom/topology persistence is more important than answer-token margins.")
print()
print("If no improvement:")
print("  Current TopK readout features are insufficient; next step should add explicit")
print("  relation-triple scoring or contrastive clean-vs-perturbed Delta-C features.")
print()
print("Done.")