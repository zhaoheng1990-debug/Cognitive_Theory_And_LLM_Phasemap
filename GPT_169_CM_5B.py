# ============================================================
# CM-5B: Non-Tautological Early/Mid O-flow Audit
#
# Goal:
#   Fix CM-5A's telescoping caveat.
#
# CM-5A used:
#       O_l = dR_{l+1} - dR_l
#   and allowed windows that overlap the decision ΔU target.
#
# CM-5B forbids decision-window leakage:
#
#   Target:
#       ΔU_decision = PC1(dR over decision window)
#
#   Predictors:
#       O_profile from windows strictly before decision_start.
#
# Tests:
#   1. O_early / O_mid / O_predecision predict later ΔU.
#   2. O_predecision classifies mechanism.
#   3. Layerwise predecision O correlation peaks before decision window.
#   4. Cross-model early/mid O profiles are compared after depth normalization.
#
# Inputs:
#   It can run standalone.
#   If cm4d_outputs/<model>_best_summary.json exists, model-specific init windows are used.
#
# Run:
#   python cm5b_early_mid_oflow_non_tautological_audit.py
#
# Outputs:
#   cm5b_outputs/
#       cm5b_model_summary.csv
#       cm5b_cross_model_profile_corr.csv
#       cm5b_overall_summary.json
#       <model>_cm5b_summary.json
#       <model>_predecision_layer_corr.csv
#       <model>_nonleak_window_summary.csv
#       <model>_R_dataset.csv
#
# ============================================================

import os
import re
import gc
import json
import math
import random
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

QWEN_PATH = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"
LLAMA_PATH = r"D:\model\Llama-3.2-1B-Instruct"
GEMMA_PATH = r"D:\model\gemma-2-2b-it"

MODEL_SPECS = {
    "qwen": QWEN_PATH,
    "llama": LLAMA_PATH,
    "gemma": GEMMA_PATH,
}

CM4D_DIR = Path("cm4d_outputs")
SAVE_DIR = Path("cm5b_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 4
MAX_LEN = 320
N_GRAPHS = 36

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Decision target window. Leakage starts at decision_start.
DECISION_FRAC = (0.70, 0.92)

REL_PRESERVE_CONDS = ["rename", "permuted", "redundant", "irrelevant", "paraphrase"]
STRUCTURE_CHANGE_CONDS = [
    "weak_distractor",
    "competition_balanced",
    "direct_conflict",
    "closure_update",
    "closure_override",
    "exception_override",
]
ALL_CONDS = ["clean"] + REL_PRESERVE_CONDS + STRUCTURE_CHANGE_CONDS

MECHANISM = {
    "clean": "stable",
    "rename": "stable",
    "permuted": "stable",
    "redundant": "stable",
    "irrelevant": "stable",
    "paraphrase": "stable",
    "weak_distractor": "stable_shift",
    "competition_balanced": "competition",
    "direct_conflict": "competition",
    "closure_update": "closure",
    "closure_override": "closure",
    "exception_override": "closure",
}

COND_CLASS = {
    "clean": "clean",
    **{c: "relation_preserving" for c in REL_PRESERVE_CONDS},
    **{c: "structure_changing" for c in STRUCTURE_CHANGE_CONDS},
}

LABEL_CANDIDATES = [
    "Red", "Blue", "Green", "Yellow", "North", "South", "East", "West",
    "Copper", "Silver", "Gold", "Iron", "Circle", "Square", "Triangle", "Star",
    "River", "Mountain", "Forest", "Ocean", "Sun", "Moon", "Cloud", "Stone",
    "Alpha", "Beta", "Gamma", "Delta", "Apple", "Orange", "Lemon", "Pear",
]

ENTITIES = [
    ("Ava", "Bela", "Cora"),
    ("Darin", "Elo", "Faye"),
    ("Galen", "Hera", "Ivo"),
    ("Juno", "Kira", "Lio"),
    ("Mira", "Nero", "Orin"),
    ("Pia", "Quin", "Rhea"),
    ("Sola", "Taro", "Una"),
    ("Vera", "Wen", "Xio"),
    ("Yara", "Zeno", "Nia"),
    ("Orla", "Pavel", "Rin"),
    ("Nora", "Silas", "Tess"),
    ("Uma", "Vito", "Willa"),
]

REL_WORDS = [
    ("belongs to", "is located at"),
    ("is assigned to", "maps to"),
    ("is part of", "points to"),
    ("is grouped under", "has label"),
]

MECH_MAP3 = {"stable": 0, "stable_shift": 0, "competition": 1, "closure": 2}

# ============================================================
# SEED
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ============================================================
# DATASET
# ============================================================

def continuation_ids(tokenizer, text):
    return tokenizer(" " + text, add_special_tokens=False)["input_ids"]

def audit_common_single_token_labels(model_paths):
    tokenizers = {}
    rows = []
    common = []
    for name, path in model_paths.items():
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
        tokenizers[name] = tok

    for lab in LABEL_CANDIDATES:
        ok = True
        row = {"label": lab}
        for name, tok in tokenizers.items():
            ids = continuation_ids(tok, lab)
            row[f"{name}_ids"] = str(ids)
            row[f"{name}_len"] = len(ids)
            if len(ids) != 1:
                ok = False
        row["is_common_single_token"] = ok
        rows.append(row)
        if ok:
            common.append(lab)

    pd.DataFrame(rows).to_csv(SAVE_DIR / "common_label_tokenization_audit.csv", index=False, encoding="utf-8-sig")

    if len(common) < 12:
        raise RuntimeError(f"Need at least 12 common single-token labels. Found {len(common)}: {common}")

    return common[:12]

def make_prompt(cond, a, b, d, clean_label, conflict_label, aux_label, relation1, relation2):
    if cond == "clean":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} {relation1} {b}.",
            f"Fact 2: {b} {relation2} {clean_label}.",
            f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
            "Answer with exactly one word:",
        ]
    elif cond == "rename":
        lines = [
            "Consider a tiny mapping network.",
            f"Link one: {a} {relation1} {b}.",
            f"Link two: {b} {relation2} {clean_label}.",
            f"Choose the label reached from {a}: {clean_label} or {conflict_label}.",
            "One word answer:",
        ]
    elif cond == "permuted":
        lines = [
            "You are given a small relation graph.",
            f"Fact 2: {b} {relation2} {clean_label}.",
            f"Fact 1: {a} {relation1} {b}.",
            f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
            "Answer with exactly one word:",
        ]
    elif cond == "redundant":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} {relation1} {b}.",
            f"Fact 2: {b} {relation2} {clean_label}.",
            f"Repeated confirmation: {a} still goes through {b}.",
            f"Repeated confirmation: {b} still points to {clean_label}.",
            f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
            "Answer with exactly one word:",
        ]
    elif cond == "irrelevant":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} {relation1} {b}.",
            f"Fact 2: {b} {relation2} {clean_label}.",
            f"Irrelevant fact: {d} is associated with {aux_label}.",
            f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
            "Answer with exactly one word:",
        ]
    elif cond == "paraphrase":
        lines = [
            f"In this example, {a} reaches {b}.",
            f"The destination connected to {b} is {clean_label}.",
            f"Based on those two links, select the label for {a}: {clean_label} or {conflict_label}.",
            "Answer:",
        ]
    elif cond == "weak_distractor":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} {relation1} {b}.",
            f"Fact 2: {b} {relation2} {clean_label}.",
            f"Weak note: some unrelated source mentions {conflict_label}, but does not update the graph.",
            f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
            "Answer with exactly one word:",
        ]
    elif cond == "competition_balanced":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} {relation1} {b}.",
            f"Fact 2: {b} {relation2} {clean_label}.",
            f"Competing fact: {a} is also associated with {conflict_label}.",
            f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
            "Answer with exactly one word:",
        ]
    elif cond == "direct_conflict":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} {relation1} {b}.",
            f"Fact 2: {b} {relation2} {clean_label}.",
            f"Direct conflicting fact: {a} {relation2} {conflict_label}.",
            f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
            "Answer with exactly one word:",
        ]
    elif cond == "closure_update":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} {relation1} {b}.",
            f"Old record: {b} {relation2} {clean_label}.",
            f"Updated record: {b} {relation2} {conflict_label}, not {clean_label}.",
            f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
            "Answer with exactly one word:",
        ]
    elif cond == "closure_override":
        lines = [
            "You are given a rule system.",
            f"General rule: items that {relation1} {b} receive label {clean_label}.",
            f"Override rule: in this case, items that {relation1} {b} receive label {conflict_label}.",
            f"Fact: {a} {relation1} {b}.",
            f"Question: Which label applies to {a}: {clean_label} or {conflict_label}?",
            "Answer with exactly one word:",
        ]
    elif cond == "exception_override":
        lines = [
            "You are given a rule system.",
            f"General rule: items connected to {b} use label {clean_label}.",
            f"Exception: {a} is a special case and uses label {conflict_label}.",
            f"Fact: {a} is connected to {b}.",
            f"Question: Which label applies to {a}: {clean_label} or {conflict_label}?",
            "Answer with exactly one word:",
        ]
    else:
        raise ValueError(cond)

    return "\n".join(lines)

def build_dataset(common_labels):
    rows = []
    graph_count = 0
    n_labels = len(common_labels)
    for ent_i, (a, b, d) in enumerate(ENTITIES):
        for rel_i, (r1, r2) in enumerate(REL_WORDS):
            if graph_count >= N_GRAPHS:
                break
            clean_label = common_labels[(2 * graph_count) % n_labels]
            conflict_label = common_labels[(2 * graph_count + 1) % n_labels]
            aux_label = common_labels[(2 * graph_count + 2) % n_labels]
            for cond in ALL_CONDS:
                rows.append({
                    "prompt_id": f"g{graph_count:03d}_{cond}",
                    "graph_id": graph_count,
                    "entity_id": ent_i,
                    "relation_id": rel_i,
                    "condition": cond,
                    "condition_class": COND_CLASS[cond],
                    "mechanism": MECHANISM[cond],
                    "clean_label": clean_label,
                    "conflict_label": conflict_label,
                    "aux_label": aux_label,
                    "text": make_prompt(cond, a, b, d, clean_label, conflict_label, aux_label, r1, r2),
                })
            graph_count += 1
        if graph_count >= N_GRAPHS:
            break
    return pd.DataFrame(rows)

# ============================================================
# MODEL UTILS
# ============================================================

def load_tokenizer(path):
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok

def load_model(path):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    if DEVICE == "cpu":
        model.to(DEVICE)
    model.eval()
    return model

def get_num_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "layers"):
        return len(model.layers)
    raise RuntimeError("Cannot locate layers")

def get_lm_head_weight(model):
    if hasattr(model, "lm_head"):
        return model.lm_head.weight.detach()
    raise RuntimeError("Cannot locate lm_head")

def last_positions(attention_mask):
    return torch.full((attention_mask.shape[0],), attention_mask.shape[1] - 1, dtype=torch.long, device=attention_mask.device)

def decision_layers(num_layers):
    a, b = DECISION_FRAC
    l0 = max(0, min(num_layers - 1, int(math.floor(num_layers * a))))
    l1 = max(l0, min(num_layers - 1, int(math.floor(num_layers * b))))
    return list(range(l0, l1 + 1))

def parse_layers(s):
    nums = re.findall(r"\d+", str(s))
    return [int(x) for x in nums]

def load_cm4d_windows(model_key):
    path = CM4D_DIR / f"{model_key}_best_summary.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for kind in ["topology", "mechanism", "downstream", "overall"]:
        key = f"{kind}_best_layers"
        if key in data:
            out[kind] = parse_layers(data[key])
    return out

def safe_auc(y, score):
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y, score))
    except Exception:
        return np.nan

def corr_safe(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def ridge_cv(X, y, groups):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    if X.shape[1] == 0 or len(np.unique(groups)) < 3:
        return np.nan, np.nan
    pred = np.zeros_like(y)
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for tr, te in gkf.split(X, y, groups):
        reg = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0)),
        ])
        reg.fit(X[tr], y[tr])
        pred[te] = reg.predict(X[te])
    return float(r2_score(y, pred)), corr_safe(y, pred)

def logistic_cv(X, y, groups):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    if X.shape[1] == 0 or len(np.unique(y)) < 2 or len(np.unique(groups)) < 3:
        return np.nan, np.nan
    pred = np.zeros_like(y)
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for tr, te in gkf.split(X, y, groups):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    return float(accuracy_score(y, pred)), float(f1_score(y, pred, average="macro", zero_division=0))

def interpolate_profile(values, n_points=50):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return np.full(n_points, values[0], dtype=float)
    x = np.linspace(0, 1, len(values))
    xi = np.linspace(0, 1, n_points)
    return np.interp(xi, x, values)

# ============================================================
# EXTRACTION
# ============================================================

def extract_R_all_layers(model_key, model_path, df):
    print(f"\n========== {model_key} ==========")
    tok = load_tokenizer(model_path)
    model = load_model(model_path)
    num_layers = get_num_layers(model)
    W = get_lm_head_weight(model).detach().float().to(model.device)
    print("num_layers:", num_layers)

    clean_ids, conflict_ids = [], []
    token_rows = []
    for _, row in df.iterrows():
        cids = continuation_ids(tok, row["clean_label"])
        eids = continuation_ids(tok, row["conflict_label"])
        token_rows.append({
            "prompt_id": row["prompt_id"],
            "clean_label": row["clean_label"],
            "conflict_label": row["conflict_label"],
            "clean_ids": str(cids),
            "conflict_ids": str(eids),
            "both_single": int(len(cids) == 1 and len(eids) == 1),
        })
        if len(cids) != 1 or len(eids) != 1:
            raise RuntimeError(f"{model_key}: non-single labels {row['clean_label']} {cids}, {row['conflict_label']} {eids}")
        clean_ids.append(cids[0])
        conflict_ids.append(eids[0])
    pd.DataFrame(token_rows).to_csv(SAVE_DIR / f"{model_key}_tokenization_audit.csv", index=False, encoding="utf-8-sig")

    n = len(df)
    R = np.zeros((n, num_layers), dtype=np.float32)
    texts = df["text"].tolist()

    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            end = min(n, start + BATCH_SIZE)
            inputs = tok(texts[start:end], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(model.device)
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            hstates = outputs.hidden_states
            pos = last_positions(inputs["attention_mask"])
            bsz = end - start
            cids_t = torch.tensor(clean_ids[start:end], dtype=torch.long, device=model.device)
            eids_t = torch.tensor(conflict_ids[start:end], dtype=torch.long, device=model.device)

            for l in range(num_layers):
                h = hstates[l + 1][torch.arange(bsz, device=model.device), pos, :].detach().float()
                rc = torch.sum(h * W[cids_t].float(), dim=1)
                re = torch.sum(h * W[eids_t].float(), dim=1)
                R[start:end, l] = (rc - re).detach().cpu().numpy().astype(np.float32)

            print(f"  processed {end}/{n}")

            del outputs, hstates, inputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    del model, tok, W
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return R, num_layers

# ============================================================
# PROCESS MODEL
# ============================================================

def process_model(model_key, model_path, df):
    R, L = extract_R_all_layers(model_key, model_path, df)
    dec_layers = decision_layers(L)
    dec_start = min(dec_layers)
    dec_end = max(dec_layers)
    cm4d = load_cm4d_windows(model_key)

    model_df = df.copy()
    for l in range(L):
        model_df[f"R_L{l}"] = R[:, l]

    # clean-relative dR
    clean_indices = {int(row.graph_id): idx for idx, row in model_df.reset_index(drop=True).iterrows() if row["condition"] == "clean"}
    dR = np.zeros_like(R)
    for i, row in model_df.reset_index(drop=True).iterrows():
        ci = clean_indices[int(row["graph_id"])]
        dR[i] = R[i] - R[ci]

    for l in range(L):
        model_df[f"dR_L{l}"] = dR[:, l]

    # Target ΔU strictly in decision window.
    X_dec = dR[:, dec_layers]
    pca = PCA(n_components=min(3, X_dec.shape[1]))
    pcs = pca.fit_transform(X_dec)
    deltaU = pcs[:, 0]
    if np.mean(deltaU[model_df["mechanism"] == "stable"]) < np.mean(deltaU[model_df["mechanism"] == "closure"]):
        deltaU = -deltaU
    model_df["DeltaU_decision"] = deltaU.astype(np.float32)

    O = dR[:, 1:] - dR[:, :-1]  # n x L-1
    y3 = model_df["mechanism"].map(MECH_MAP3).values.astype(int)
    groups = model_df["graph_id"].values.astype(int)

    # strictly non-leaking transition layers must satisfy l+1 < dec_start, i.e. l <= dec_start-2
    pre_max = dec_start - 2
    if pre_max < 0:
        pre_max = 0

    def trans_range(a, b):
        a = max(0, min(L - 2, a))
        b = max(a, min(L - 2, b))
        # enforce no leakage
        b = min(b, pre_max)
        if b < a:
            return []
        return list(range(a, b + 1))

    # CM4D model-specific windows.
    downstream = cm4d.get("downstream", list(range(0, min(3, L))))
    mechanism = cm4d.get("mechanism", list(range(0, min(6, L))))
    topology = cm4d.get("topology", list(range(0, min(2, L))))
    overall = cm4d.get("overall", downstream)

    d_end = max(downstream) if downstream else 0
    m_end = max(mechanism) if mechanism else 0
    t_end = max(topology) if topology else 0
    o_end = max(overall) if overall else 0

    # Define non-leaking windows.
    early_end = max(1, int(math.floor((L - 1) * 0.25)))
    mid_end = max(early_end + 1, int(math.floor((L - 1) * 0.55)))

    window_defs = {
        "O_early_0_25depth": trans_range(0, early_end),
        "O_mid_25_55depth": trans_range(early_end + 1, mid_end),
        "O_predecision_all": trans_range(0, dec_start - 2),
        "O_after_topology_until_predecision": trans_range(t_end, dec_start - 2),
        "O_after_mechanism_until_predecision": trans_range(m_end, dec_start - 2),
        "O_after_downstream_until_predecision": trans_range(d_end, dec_start - 2),
        "O_after_overall_until_predecision": trans_range(o_end, dec_start - 2),
        "O_cm4d_mechanism_only": trans_range(min(mechanism) if mechanism else 0, max(mechanism)-1 if len(mechanism) > 1 else max(mechanism) if mechanism else 0),
        "O_cm4d_downstream_only": trans_range(min(downstream) if downstream else 0, max(downstream)-1 if len(downstream) > 1 else max(downstream) if downstream else 0),
    }

    rows = []
    for win_name, layers in window_defs.items():
        layers = [l for l in layers if 0 <= l < L - 1 and l <= pre_max]
        if len(layers) == 0:
            continue
        X = O[:, layers]
        O_integral = X.sum(axis=1)
        O_abs_integral = np.abs(X).sum(axis=1)
        O_energy = np.sqrt((X ** 2).sum(axis=1))
        O_mean = X.mean(axis=1)
        O_slope = X[:, -1] - X[:, 0] if X.shape[1] > 1 else np.zeros(X.shape[0])
        X_aug = np.column_stack([X, O_integral, O_abs_integral, O_energy, O_mean, O_slope]).astype(np.float32)

        r2, corr = ridge_cv(X_aug, deltaU, groups)
        acc, f1 = logistic_cv(X_aug, y3, groups)

        rows.append({
            "model_key": model_key,
            "window": win_name,
            "transition_layers": str(layers),
            "n_transitions": len(layers),
            "decision_start": dec_start,
            "decision_layers": str(dec_layers),
            "leakage_free": True,
            "integral_corr_deltaU": corr_safe(O_integral, deltaU),
            "abs_integral_corr_deltaU": corr_safe(O_abs_integral, deltaU),
            "energy_corr_deltaU": corr_safe(O_energy, deltaU),
            "mean_corr_deltaU": corr_safe(O_mean, deltaU),
            "slope_corr_deltaU": corr_safe(O_slope, deltaU),
            "ridge_deltaU_r2": r2,
            "ridge_deltaU_corr": corr,
            "mechanism_acc": acc,
            "mechanism_macro_f1": f1,
        })

    win_df = pd.DataFrame(rows)
    win_df.to_csv(SAVE_DIR / f"{model_key}_nonleak_window_summary.csv", index=False, encoding="utf-8-sig")

    # Layerwise predecision correlations.
    layer_rows = []
    for l in range(0, pre_max + 1):
        ol = O[:, l]
        layer_rows.append({
            "model_key": model_key,
            "transition_layer": l,
            "transition": f"L{l}->L{l+1}",
            "layer_frac": l / max(1, L - 2),
            "decision_start": dec_start,
            "corr_O_deltaU_decision": corr_safe(ol, deltaU),
            "abs_corr_O_deltaU_decision": abs(corr_safe(ol, deltaU)),
            "mean_O_stable": float(np.mean(ol[y3 == 0])),
            "mean_O_competition": float(np.mean(ol[y3 == 1])),
            "mean_O_closure": float(np.mean(ol[y3 == 2])),
            "auc_closure_by_O": safe_auc((y3 == 2).astype(int), ol),
            "auc_competition_by_O": safe_auc((y3 == 1).astype(int), ol),
        })
    layer_df = pd.DataFrame(layer_rows)
    layer_df.to_csv(SAVE_DIR / f"{model_key}_predecision_layer_corr.csv", index=False, encoding="utf-8-sig")

    # profiles for cross-model comparison.
    profile_abs = interpolate_profile(layer_df["abs_corr_O_deltaU_decision"].values, 50)
    profile_signed = interpolate_profile(layer_df["corr_O_deltaU_decision"].values, 50)
    np.savez_compressed(
        SAVE_DIR / f"{model_key}_predecision_profiles.npz",
        profile_abs=profile_abs,
        profile_signed=profile_signed,
        raw_abs=layer_df["abs_corr_O_deltaU_decision"].values,
        raw_signed=layer_df["corr_O_deltaU_decision"].values,
        layer_frac=layer_df["layer_frac"].values,
    )

    best_delta = win_df.sort_values("ridge_deltaU_corr", ascending=False).iloc[0].to_dict()
    best_mech = win_df.sort_values("mechanism_macro_f1", ascending=False).iloc[0].to_dict()
    peak_layer = layer_df.sort_values("abs_corr_O_deltaU_decision", ascending=False).iloc[0].to_dict() if len(layer_df) else {}

    # Strict controls:
    # - compare predecision with cm4d-only window.
    # - no decision transitions are used.
    summary = {
        "model_key": model_key,
        "num_layers": L,
        "decision_layers": dec_layers,
        "decision_start": dec_start,
        "cm4d_windows": cm4d,
        "deltaU_pc_variance": pca.explained_variance_ratio_.tolist(),
        "deltaU_pc1_variance": float(pca.explained_variance_ratio_[0]),
        "predecision_max_transition": pre_max,
        "best_nonleak_deltaU_window": best_delta,
        "best_nonleak_mechanism_window": best_mech,
        "peak_predecision_layer_corr": peak_layer,
        "pass_nonleak_deltaU_corr_gt_0p4": bool(best_delta["ridge_deltaU_corr"] > 0.4),
        "pass_nonleak_mechanism_f1_gt_0p65": bool(best_mech["mechanism_macro_f1"] > 0.65),
    }

    with open(SAVE_DIR / f"{model_key}_cm5b_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    model_df.to_csv(SAVE_DIR / f"{model_key}_R_dataset.csv", index=False, encoding="utf-8-sig")

    return summary

def compute_cross_model_profiles():
    profiles = {}
    for mk in MODEL_SPECS:
        data = np.load(SAVE_DIR / f"{mk}_predecision_profiles.npz", allow_pickle=True)
        profiles[mk] = {
            "abs": data["profile_abs"],
            "signed": data["profile_signed"],
        }

    rows = []
    for a, b in combinations(MODEL_SPECS.keys(), 2):
        for typ in ["abs", "signed"]:
            rows.append({
                "profile_type": typ,
                "model_a": a,
                "model_b": b,
                "pearson_profile_corr": corr_safe(profiles[a][typ], profiles[b][typ]),
            })
    out = pd.DataFrame(rows)
    out.to_csv(SAVE_DIR / "cm5b_cross_model_profile_corr.csv", index=False, encoding="utf-8-sig")
    return out

# ============================================================
# MAIN
# ============================================================

def main():
    print("Auditing common single-token labels...")
    common_labels = audit_common_single_token_labels(MODEL_SPECS)
    print("Common labels:", common_labels)

    df = build_dataset(common_labels)
    df.to_csv(SAVE_DIR / "cm5b_prompt_dataset.csv", index=False, encoding="utf-8-sig")
    print("Dataset:", len(df), "prompts")

    summaries = []
    for mk, path in MODEL_SPECS.items():
        summaries.append(process_model(mk, path, df))

    flat = []
    for s in summaries:
        bd = s["best_nonleak_deltaU_window"]
        bm = s["best_nonleak_mechanism_window"]
        peak = s["peak_predecision_layer_corr"]
        flat.append({
            "model_key": s["model_key"],
            "num_layers": s["num_layers"],
            "decision_layers": str(s["decision_layers"]),
            "decision_start": s["decision_start"],
            "predecision_max_transition": s["predecision_max_transition"],
            "deltaU_pc1_variance": s["deltaU_pc1_variance"],
            "best_nonleak_deltaU_window": bd["window"],
            "best_nonleak_deltaU_layers": bd["transition_layers"],
            "best_nonleak_deltaU_corr": bd["ridge_deltaU_corr"],
            "best_nonleak_deltaU_r2": bd["ridge_deltaU_r2"],
            "best_nonleak_integral_corr": bd["integral_corr_deltaU"],
            "best_nonleak_mechanism_window": bm["window"],
            "best_nonleak_mechanism_layers": bm["transition_layers"],
            "best_nonleak_mechanism_f1": bm["mechanism_macro_f1"],
            "best_nonleak_mechanism_acc": bm["mechanism_acc"],
            "peak_predecision_transition": peak.get("transition", None),
            "peak_predecision_layer_frac": peak.get("layer_frac", None),
            "peak_predecision_abs_corr": peak.get("abs_corr_O_deltaU_decision", None),
            "pass_nonleak_deltaU_corr_gt_0p4": s["pass_nonleak_deltaU_corr_gt_0p4"],
            "pass_nonleak_mechanism_f1_gt_0p65": s["pass_nonleak_mechanism_f1_gt_0p65"],
        })

    model_summary = pd.DataFrame(flat)
    model_summary.to_csv(SAVE_DIR / "cm5b_model_summary.csv", index=False, encoding="utf-8-sig")

    cross = compute_cross_model_profiles()

    overall = {
        "audit": "CM-5B Non-Tautological Early/Mid O-flow Audit",
        "definition": (
            "Uses only pre-decision O_adv transitions to predict later decision-window DeltaU. "
            "This forbids telescoping overlap with the target window."
        ),
        "models": summaries,
        "cross_model_profile_corr": cross.to_dict(orient="records"),
    }
    with open(SAVE_DIR / "cm5b_overall_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    print("\nCM-5B complete.")
    print(model_summary)
    print(cross)

if __name__ == "__main__":
    main()
