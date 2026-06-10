# ============================================================
# CM-5A: Cross-Model O_cont / Advantage-Flow Audit
#
# Goal:
#   Validate the cross-model analogue of:
#
#       O_l^cont ≈ d(λ1 - λ2)/dl
#       ∫ O_l^cont dl → ΔU
#
# Minimal observable proxy:
#
#       R_l = logit_l(C) - logit_l(E)
#       dR_l = R_l(condition) - R_l(clean_anchor)
#       O_l^adv = dR_{l+1} - dR_l
#
# This is not the final continuous operator from the Qwen-only
# PhaseMap-7E/8A line, but a cross-model observable proxy for
# layerwise path-advantage growth.
#
# It tests:
#   1. O-integral predicts model-specific DeltaU.
#   2. O-profile classifies mechanism.
#   3. O-layer correlations identify accumulation / decision bands.
#   4. O-correlation profiles are compared across models after depth normalization.
#
# Uses CM-4D outputs if available to select model-specific init/downstream windows.
#
# Run:
#   python cm5a_cross_model_ocont_advantage_flow_audit.py
#
# Outputs:
#   cm5a_outputs/
#       cm5a_model_summary.csv
#       cm5a_cross_model_profile_corr.csv
#       cm5a_overall_summary.json
#       <model>_ocont_summary.json
#       <model>_layer_o_corr.csv
#       <model>_window_prediction_summary.csv
#       <model>_transition_dataset.csv
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
SAVE_DIR = Path("cm5a_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 4
MAX_LEN = 320
N_GRAPHS = 36

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

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
MECH_NAMES = {0: "stable_like", 1: "competition", 2: "closure"}

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

def load_cm4d_windows(model_key, num_layers):
    """
    Use CM-4D model-specific windows if present.
    Fallback to relative defaults.
    """
    path = CM4D_DIR / f"{model_key}_best_summary.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        out = {}
        for kind in ["topology", "mechanism", "downstream", "overall"]:
            key = f"{kind}_best_layers"
            if key in data:
                out[kind] = parse_layers(data[key])
        if out:
            return out

    # fallback
    dec = decision_layers(num_layers)
    return {
        "topology": list(range(0, min(2, num_layers - 1) + 1)),
        "mechanism": list(range(0, min(6, num_layers - 1) + 1)),
        "downstream": list(range(0, min(6, num_layers - 1) + 1)),
        "overall": list(range(0, min(6, num_layers - 1) + 1)),
    }

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

# ============================================================
# EVAL HELPERS
# ============================================================

def ridge_cv(X, y, groups):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    if len(np.unique(groups)) < 3:
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
    if len(np.unique(y)) < 2 or len(np.unique(groups)) < 3:
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
# MAIN MODEL EXTRACTION
# ============================================================

def extract_R_all_layers(model_key, model_path, df):
    print(f"\n========== {model_key} ==========")
    tok = load_tokenizer(model_path)
    model = load_model(model_path)
    num_layers = get_num_layers(model)
    W = get_lm_head_weight(model).detach().float().to(model.device)

    print("num_layers:", num_layers)

    # Label IDs
    clean_ids, conflict_ids = [], []
    rows = []
    for _, row in df.iterrows():
        cids = continuation_ids(tok, row["clean_label"])
        eids = continuation_ids(tok, row["conflict_label"])
        rows.append({
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
    pd.DataFrame(rows).to_csv(SAVE_DIR / f"{model_key}_tokenization_audit.csv", index=False, encoding="utf-8-sig")

    n = len(df)
    texts = df["text"].tolist()
    R = np.zeros((n, num_layers), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            end = min(n, start + BATCH_SIZE)
            batch = texts[start:end]
            inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(model.device)
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

def process_model(model_key, model_path, df):
    R, L = extract_R_all_layers(model_key, model_path, df)
    dec_layers = decision_layers(L)
    cm4d_windows = load_cm4d_windows(model_key, L)

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

    # DeltaU = PC1 over decision dR window.
    X_dec = dR[:, dec_layers]
    pca = PCA(n_components=min(3, X_dec.shape[1]))
    pcs = pca.fit_transform(X_dec)
    deltaU = pcs[:, 0]
    if np.mean(deltaU[model_df["mechanism"] == "stable"]) < np.mean(deltaU[model_df["mechanism"] == "closure"]):
        deltaU = -deltaU
    model_df["DeltaU"] = deltaU.astype(np.float32)

    # O_adv transitions
    O = dR[:, 1:] - dR[:, :-1]  # shape n x (L-1)

    # Conditions metadata.
    y3 = model_df["mechanism"].map(MECH_MAP3).values.astype(int)
    groups = model_df["graph_id"].values.astype(int)

    # O windows.
    init_downstream = cm4d_windows.get("downstream", list(range(0, min(2, L))))
    init_mechanism = cm4d_windows.get("mechanism", list(range(0, min(6, L))))
    init_overall = cm4d_windows.get("overall", init_downstream)

    init_down_end = max(init_downstream)
    init_mech_end = max(init_mechanism)
    dec_start, dec_end = min(dec_layers), max(dec_layers)

    def transition_range(a, b):
        # transitions l means dR_{l+1}-dR_l, valid l=0..L-2
        a = max(0, min(L - 2, a))
        b = max(a, min(L - 2, b))
        return list(range(a, b + 1))

    window_defs = {
        "O_all_transitions": list(range(0, L - 1)),
        "O_predecision": transition_range(0, dec_start - 1),
        "O_decision": transition_range(dec_start, dec_end - 1),
        "O_initDown_to_decision": transition_range(init_down_end, dec_end - 1),
        "O_initMech_to_decision": transition_range(init_mech_end, dec_end - 1),
        "O_accum_after_initDown": transition_range(init_down_end, dec_start - 1),
        "O_accum_after_initMech": transition_range(init_mech_end, dec_start - 1),
        "O_cm4d_downstream_window": transition_range(min(init_downstream), max(init_downstream) - 1 if len(init_downstream) > 1 else max(init_downstream)),
        "O_cm4d_mechanism_window": transition_range(min(init_mechanism), max(init_mechanism) - 1 if len(init_mechanism) > 1 else max(init_mechanism)),
    }

    # Window prediction summary.
    rows = []
    for win_name, trans_layers in window_defs.items():
        trans_layers = [l for l in trans_layers if 0 <= l < L - 1]
        if len(trans_layers) == 0:
            continue

        X = O[:, trans_layers]
        O_integral = X.sum(axis=1)
        O_abs_integral = np.abs(X).sum(axis=1)
        O_energy = np.sqrt((X ** 2).sum(axis=1))
        X_aug = np.column_stack([X, O_integral, O_abs_integral, O_energy]).astype(np.float32)

        r2, corr = ridge_cv(X_aug, deltaU, groups)
        acc, f1 = logistic_cv(X_aug, y3, groups)

        rows.append({
            "model_key": model_key,
            "window": win_name,
            "transition_layers": str(trans_layers),
            "n_transitions": len(trans_layers),
            "integral_corr_deltaU": corr_safe(O_integral, deltaU),
            "abs_integral_corr_deltaU": corr_safe(O_abs_integral, deltaU),
            "energy_corr_deltaU": corr_safe(O_energy, deltaU),
            "ridge_deltaU_r2": r2,
            "ridge_deltaU_corr": corr,
            "mechanism_acc": acc,
            "mechanism_macro_f1": f1,
        })

    win_df = pd.DataFrame(rows)
    win_df.to_csv(SAVE_DIR / f"{model_key}_window_prediction_summary.csv", index=False, encoding="utf-8-sig")

    # Layer-wise O correlation with DeltaU and mechanism separation.
    layer_rows = []
    for l in range(L - 1):
        ol = O[:, l]
        closure_auc = safe_auc((y3 == 2).astype(int), ol)
        comp_auc = safe_auc((y3 == 1).astype(int), ol)
        layer_rows.append({
            "model_key": model_key,
            "transition_layer": l,
            "transition": f"L{l}->L{l+1}",
            "layer_frac": l / max(1, L - 2),
            "corr_O_deltaU": corr_safe(ol, deltaU),
            "mean_O_stable": float(np.mean(ol[y3 == 0])),
            "mean_O_competition": float(np.mean(ol[y3 == 1])),
            "mean_O_closure": float(np.mean(ol[y3 == 2])),
            "auc_closure_by_O": closure_auc,
            "auc_competition_by_O": comp_auc,
            "abs_corr_O_deltaU": abs(corr_safe(ol, deltaU)),
        })
    layer_df = pd.DataFrame(layer_rows)
    layer_df.to_csv(SAVE_DIR / f"{model_key}_layer_o_corr.csv", index=False, encoding="utf-8-sig")

    # Transition dataset.
    trans_rows = []
    for i, row in model_df.reset_index(drop=True).iterrows():
        base = {
            "prompt_id": row["prompt_id"],
            "graph_id": int(row["graph_id"]),
            "condition": row["condition"],
            "mechanism": row["mechanism"],
            "DeltaU": float(row["DeltaU"]),
        }
        for l in range(L - 1):
            trans_rows.append({
                **base,
                "transition_layer": l,
                "transition": f"L{l}->L{l+1}",
                "O_adv": float(O[i, l]),
                "dR_l": float(dR[i, l]),
                "dR_next": float(dR[i, l + 1]),
            })
    pd.DataFrame(trans_rows).to_csv(SAVE_DIR / f"{model_key}_transition_dataset.csv", index=False, encoding="utf-8-sig")

    # Peaks and summaries.
    best_win_delta = win_df.sort_values("ridge_deltaU_corr", ascending=False).iloc[0].to_dict()
    best_win_mech = win_df.sort_values("mechanism_macro_f1", ascending=False).iloc[0].to_dict()
    peak_corr_row = layer_df.sort_values("abs_corr_O_deltaU", ascending=False).iloc[0].to_dict()

    # Profiles for cross-model comparison.
    profile_corr = interpolate_profile(layer_df["corr_O_deltaU"].values, n_points=50)
    profile_abs = interpolate_profile(layer_df["abs_corr_O_deltaU"].values, n_points=50)
    np.savez_compressed(
        SAVE_DIR / f"{model_key}_ocont_profiles.npz",
        profile_corr=profile_corr,
        profile_abs=profile_abs,
        raw_corr=layer_df["corr_O_deltaU"].values,
        layer_frac=layer_df["layer_frac"].values,
    )

    summary = {
        "model_key": model_key,
        "num_layers": L,
        "decision_layers": dec_layers,
        "cm4d_windows": cm4d_windows,
        "deltaU_pc_variance": pca.explained_variance_ratio_.tolist(),
        "deltaU_pc1_variance": float(pca.explained_variance_ratio_[0]),
        "best_deltaU_window": best_win_delta,
        "best_mechanism_window": best_win_mech,
        "peak_layer_corr": peak_corr_row,
        "pass_integral_deltaU_corr_gt_0p5": bool(best_win_delta["ridge_deltaU_corr"] > 0.5),
        "pass_mechanism_f1_gt_0p65": bool(best_win_mech["mechanism_macro_f1"] > 0.65),
    }

    with open(SAVE_DIR / f"{model_key}_ocont_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    model_df.to_csv(SAVE_DIR / f"{model_key}_R_dataset.csv", index=False, encoding="utf-8-sig")

    return summary

def compute_cross_model_profiles():
    profiles = {}
    for mk in MODEL_SPECS:
        data = np.load(SAVE_DIR / f"{mk}_ocont_profiles.npz", allow_pickle=True)
        profiles[mk] = {
            "corr": data["profile_corr"],
            "abs": data["profile_abs"],
        }

    rows = []
    for a, b in combinations(MODEL_SPECS.keys(), 2):
        for typ in ["corr", "abs"]:
            xa = profiles[a][typ]
            xb = profiles[b][typ]
            rows.append({
                "profile_type": typ,
                "model_a": a,
                "model_b": b,
                "pearson_profile_corr": corr_safe(xa, xb),
            })
    out = pd.DataFrame(rows)
    out.to_csv(SAVE_DIR / "cm5a_cross_model_profile_corr.csv", index=False, encoding="utf-8-sig")
    return out

# ============================================================
# MAIN
# ============================================================

def main():
    print("Auditing common single-token labels...")
    common_labels = audit_common_single_token_labels(MODEL_SPECS)
    print("Common labels:", common_labels)

    df = build_dataset(common_labels)
    df.to_csv(SAVE_DIR / "cm5a_prompt_dataset.csv", index=False, encoding="utf-8-sig")
    print("Dataset:", len(df), "prompts")

    summaries = []
    for mk, path in MODEL_SPECS.items():
        summaries.append(process_model(mk, path, df))

    # Flatten model summary for CSV.
    flat_rows = []
    for s in summaries:
        row = {
            "model_key": s["model_key"],
            "num_layers": s["num_layers"],
            "decision_layers": str(s["decision_layers"]),
            "deltaU_pc1_variance": s["deltaU_pc1_variance"],
            "best_deltaU_window": s["best_deltaU_window"]["window"],
            "best_deltaU_corr": s["best_deltaU_window"]["ridge_deltaU_corr"],
            "best_deltaU_r2": s["best_deltaU_window"]["ridge_deltaU_r2"],
            "best_deltaU_integral_corr": s["best_deltaU_window"]["integral_corr_deltaU"],
            "best_mechanism_window": s["best_mechanism_window"]["window"],
            "best_mechanism_f1": s["best_mechanism_window"]["mechanism_macro_f1"],
            "best_mechanism_acc": s["best_mechanism_window"]["mechanism_acc"],
            "peak_O_layer": s["peak_layer_corr"]["transition"],
            "peak_O_layer_frac": s["peak_layer_corr"]["layer_frac"],
            "peak_abs_corr_O_deltaU": s["peak_layer_corr"]["abs_corr_O_deltaU"],
            "pass_integral_deltaU_corr_gt_0p5": s["pass_integral_deltaU_corr_gt_0p5"],
            "pass_mechanism_f1_gt_0p65": s["pass_mechanism_f1_gt_0p65"],
        }
        flat_rows.append(row)

    model_summary = pd.DataFrame(flat_rows)
    model_summary.to_csv(SAVE_DIR / "cm5a_model_summary.csv", index=False, encoding="utf-8-sig")

    cross = compute_cross_model_profiles()

    overall = {
        "audit": "CM-5A Cross-Model O_cont / Advantage-Flow Audit",
        "observable_proxy": "O_adv_l = dR_{l+1} - dR_l, with dR_l = R_l(condition)-R_l(clean_anchor)",
        "models": summaries,
        "cross_model_profile_corr": cross.to_dict(orient="records"),
    }
    with open(SAVE_DIR / "cm5a_overall_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    print("\nCM-5A complete.")
    print(model_summary)
    print(cross)

if __name__ == "__main__":
    main()
