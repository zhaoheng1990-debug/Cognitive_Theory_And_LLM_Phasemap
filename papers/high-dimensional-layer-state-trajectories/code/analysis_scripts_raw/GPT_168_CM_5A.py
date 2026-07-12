# ============================================================
# CM-5A: Cross-Model O_cont / Advantage-Flow Audit
#
# Goal:
#   Validate the cross-model analogue of:
#
#       O_l^cont â‰ˆ d(Î»1 - Î»2)/dl
#       âˆ« O_l^cont dl â†’ Î”U
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
import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ModuleNotFoundError:
    AutoTokenizer = None
    AutoModelForCausalLM = None

try:
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
except ModuleNotFoundError:
    PCA = None
    accuracy_score = f1_score = roc_auc_score = r2_score = None
    GroupKFold = LogisticRegression = Ridge = StandardScaler = Pipeline = None

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

QWEN_PATH = "<local-qwen-checkpoint-path>"
LLAMA_PATH = "<local-llama-checkpoint-path>"
GEMMA_PATH = "<local-gemma-checkpoint-path>"

MODEL_SPECS = {
    "qwen": QWEN_PATH,
    "llama": LLAMA_PATH,
    "gemma": GEMMA_PATH,
}

CM4D_DIR = Path("cm4d_outputs")
SAVE_DIR = Path("cm5a_outputs")

SEED = 42
BATCH_SIZE = 4
MAX_LEN = 320
N_GRAPHS = 36

DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch is not None and DEVICE == "cuda" else (torch.float32 if torch is not None else None)

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

def parse_args():
    parser = argparse.ArgumentParser(description="CM-5A cross-model O_cont / advantage-flow audit")
    parser.add_argument("--cm4d-dir", type=Path, default=CM4D_DIR)
    parser.add_argument("--save-dir", type=Path, default=SAVE_DIR)
    parser.add_argument("--qwen-path", default=QWEN_PATH)
    parser.add_argument("--llama-path", default=LLAMA_PATH)
    parser.add_argument("--gemma-path", default=GEMMA_PATH)
    parser.add_argument("--device", default=DEVICE, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--n-graphs", type=int, default=N_GRAPHS)
    parser.add_argument("--check-inputs-only", action="store_true")
    return parser.parse_args()

def dependency_status():
    return {
        "torch": torch is not None,
        "transformers": AutoTokenizer is not None and AutoModelForCausalLM is not None,
        "scikit_learn": Pipeline is not None,
    }

def path_status(path):
    p = Path(path)
    return {
        "path": p.as_posix(),
        "exists": p.exists(),
        "is_dir": p.is_dir() if p.exists() else False,
    }

def configure_from_args(args):
    global CM4D_DIR, SAVE_DIR, MODEL_SPECS, DEVICE, DTYPE
    global SEED, BATCH_SIZE, MAX_LEN, N_GRAPHS

    CM4D_DIR = args.cm4d_dir
    SAVE_DIR = args.save_dir
    MODEL_SPECS = {
        "qwen": args.qwen_path,
        "llama": args.llama_path,
        "gemma": args.gemma_path,
    }
    if args.device == "auto":
        DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    else:
        DEVICE = args.device
    DTYPE = torch.float16 if torch is not None and DEVICE == "cuda" else (torch.float32 if torch is not None else None)

    SEED = args.seed
    BATCH_SIZE = args.batch_size
    MAX_LEN = args.max_len
    N_GRAPHS = args.n_graphs

def write_input_check_report():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "script": "GPT_168_CM_5A.py",
        "audit": "CM-5A Cross-Model O_cont / Advantage-Flow Audit",
        "dependencies": dependency_status(),
        "configuration": {
            "device": DEVICE,
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "max_len": MAX_LEN,
            "n_graphs": N_GRAPHS,
        },
        "inputs": {
            "cm4d_dir_optional": path_status(CM4D_DIR),
            "model_paths": {name: path_status(path) for name, path in MODEL_SPECS.items()},
        },
        "outputs": {"save_dir": path_status(SAVE_DIR)},
    }
    report_path = SAVE_DIR / "cm5a_input_check_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"CM-5A input check report written to: {report_path}")

def ensure_runtime_dependencies():
    missing = [name for name, ok in dependency_status().items() if not ok]
    if missing:
        raise RuntimeError(
            "Missing required runtime dependencies for full CM-5A run: "
            + ", ".join(missing)
            + ". Use --check-inputs-only for portability checks without loading models."
        )

# ============================================================
# SEED
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================
# DATASET
# ============================================================

def continuation_ids(tokenizer, text):
    return tokenizer(" " + text, add_special_tokens=False)["input_ids"]

def audit_common_single_token_labels(model_paths):
    if AutoTokenizer is None:
        raise RuntimeError("transformers is required for full CM-5A runs.")
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
    rows×¾v¶‰žËkºwµçHÉ}Í½É”¡ä°ÁÉ•¤¤°½ÉÉ}Í…™”¡ä°ÁÉ•¤()‘•˜±½¥ÍÑ¥}Ø¡`°ä°É½ÕÁÌ¤è(€€€¥˜A¥Á•±¥¹”¥Ì9½¹”è(€€€€€€€É…¥Í”IÕ¹Ñ¥µ•ÉÉ½È ‰Í¥­¥Ðµ±•…É¸¥ÌÉ•ÅÕ¥É•™½È™Õ±°4´ÕÉÕ¹Ì¸ˆ¤(€€€`€ô¹À¹…Í…ÉÉ…ä¡`°‘ÑåÁ”õ¹À¹™±½…ÐÌÈ¤(€€€ä€ô¹À¹…Í…ÉÉ…ä¡ä°‘ÑåÁ”õ¥¹Ð¤(€€€É½ÕÁÌ€ô¹À¹…Í…ÉÉ…ä¡É½ÕÁÌ¤(€€€¥˜±•¸¡¹À¹Õ¹¥ÅÕ”¡ä¤¤€ð€È½È±•¸¡¹À¹Õ¹¥ÅÕ”¡É½ÕÁÌ¤¤€ð€Ìè(€€€€€€€É•ÑÕÉ¸¹À¹¹…¸°¹À¹¹…¸(€€€ÁÉ•€ô¹À¹é•É½Í}±¥­”¡ä¤(€€€­˜€ôÉ½ÕÁ-½±¡¹}ÍÁ±¥ÑÌõµ¥¸ Ô°±•¸¡¹À¹Õ¹¥ÅÕ”¡É½ÕÁÌ¤¤¤¤(€€€™½ÈÑÈ°Ñ”¥¸­˜¹ÍÁ±¥Ð¡`°ä°É½ÕÁÌ¤è(€€€€€€€±˜€ôA¥Á•±¥¹”¡l(€€€€€€€€€€€€ ‰Í…±•Èˆ°MÑ…¹‘…É‘M…±•È ¤¤°(€€€€€€€€€€€€ ‰±Èˆ°1½¥ÍÑ¥I•É•ÍÍ¥½¸¡µ…á}¥Ñ•ÈôÄÀÀÀ°±…ÍÍ}Ý•¥¡Ðô‰‰…±…¹•ˆ¤¤°(€€€€€€€t¤(€€€€€€€±˜¹™¥Ð¡amÑÉt°åmÑÉt¤(€€€€€€€ÁÉ•‘mÑ•t€ô±˜¹ÁÉ•‘¥Ð¡amÑ•t¤(€€€É•ÑÕÉ¸™±½…Ð¡…ÕÉ…å}Í½É”¡ä°ÁÉ•¤¤°™±½…Ð¡˜Å}Í½É”¡ä°ÁÉ•°…Ù•É…”ô‰µ…É¼ˆ°é•É½}‘¥Ù¥Í¥½¸ôÀ¤¤()‘•˜¥¹Ñ•ÉÁ½±…Ñ•}ÁÉ½™¥±”¡Ù…±Õ•Ì°¹}Á½¥¹ÑÌôÔÀ¤è(€€€Ù…±Õ•Ì€ô¹À¹…Í…ÉÉ…ä¡Ù…±Õ•Ì°‘ÑåÁ”õ™±½…Ð¤(€€€¥˜±•¸¡Ù…±Õ•Ì¤€ôô€Äè(€€€€€€€É•ÑÕÉ¸¹À¹™Õ±°¡¹}Á½¥¹ÑÌ°Ù…±Õ•ÍlÁt°‘ÑåÁ”õ™±½…Ð¤(€€€à€ô¹À¹±¥¹ÍÁ…” À°€Ä°±•¸¡Ù…±Õ•Ì¤¤(€€€á¤€ô¹À¹±¥¹ÍÁ…” À°€Ä°¹}Á½¥¹ÑÌ¤(€€€É•ÑÕÉ¸¹À¹¥¹Ñ•ÉÀ¡á¤°à°Ù…±Õ•Ì¤((Œ€ôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôô(Œ5%85=0aQIQ%=8(Œ€ôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôô()‘•˜•áÑÉ…Ñ}I}…±±}±…å•ÉÌ¡µ½‘•±}­•ä°µ½‘•±}Á…Ñ °‘˜¤è(€€€ÁÉ¥¹Ð¡˜‰q¸ôôôôôôôôôôíµ½‘•±}­•åô€ôôôôôôôôôôˆ¤(€€€Ñ½¬€ô±½…‘}Ñ½­•¹¥é•È¡µ½‘•±}Á…Ñ ¤(€€€µ½‘•°€ô±½…‘}µ½‘•°¡µ½‘•±}Á…Ñ ¤(€€€¹Õµ}±…å•ÉÌ€ô•Ñ}¹Õµ}±…å•ÉÌ¡µ½‘•°¤(€€€\€ô•Ñ}±µ}¡•…‘}Ý•¥¡Ð¡µ½‘•°¤¹‘•Ñ…  ¤¹™±½…Ð ¤¹Ñ¼¡µ½‘•°¹‘•Ù¥”¤((€€€ÁÉ¥¹Ð ‰¹Õµ}±…å•ÉÌèˆ°¹Õµ}±…å•ÉÌ¤((€€€€Œ1…‰•°%Ì(€€€±•…¹}¥‘Ì°½¹™±¥Ñ}¥‘Ì€ômt°mt(€€€É½ÝÌ€ômt(€€€™½È|°É½Ü¥¸‘˜¹¥Ñ•ÉÉ½ÝÌ ¤è(€€€€€€€¥‘Ì€ô½¹Ñ¥¹Õ…Ñ¥½¹}¥‘Ì¡Ñ½¬°É½Ýl‰±•…¹}±…‰•°‰t¤(€€€€€€€•¥‘Ì€ô½¹Ñ¥¹Õ…Ñ¥½¹}¥‘Ì¡Ñ½¬°É½Ýl‰½¹™±¥Ñ}±…‰•°‰t¤(€€€€€€€É½ÝÌ¹…ÁÁ•¹¡ì(€€€€€€€€€€€€‰ÁÉ½µÁÑ}¥ˆèÉ½Ýl‰ÁÉ½µÁÑ}¥‰t°(€€€€€€€€€€€€‰±•…¹}±…‰•°ˆèÉ½Ýl‰±•…¹}±…‰•°‰t°(€€€€€€€€€€€€‰½¹™±¥Ñ}±…‰•°ˆèÉ½Ýl‰½¹™±¥Ñ}±…‰•°‰t°(€€€€€€€€€€€€‰±•…¹}¥‘ÌˆèÍÑÈ¡¥‘Ì¤°(€€€€€€€€€€€€‰½¹™±¥Ñ}¥‘ÌˆèÍÑÈ¡•¥‘Ì¤°(€€€€€€€€€€€€‰‰½Ñ¡}Í¥¹±”ˆè¥¹Ð¡±•¸¡¥‘Ì¤€ôô€Ä…¹±•¸¡•¥‘Ì¤€ôô€Ä¤°(€€€€€€€ô¤(€€€€€€€¥˜±•¸¡¥‘Ì¤€„ô€Ä½È±•¸¡•¥‘Ì¤€„ô€Äè(€€€€€€€€€€€É…¥Í”IÕ¹Ñ¥µ•ÉÉ½È¡˜‰íµ½‘•±}­•åôè¹½¸µÍ¥¹±”±…‰•±ÌíÉ½Ýl±•…¹}±…‰•°uôí¥‘Íô°íÉ½Ýl½¹™±¥Ñ}±…‰•°uôí•¥‘Íôˆ¤(€€€€€€€±•…¹}¥‘Ì¹…ÁÁ•¹¡¥‘ÍlÁt¤(€€€€€€€½¹™±¥Ñ}¥‘Ì¹…ÁÁ•¹¡•¥‘ÍlÁt¤(€€€Á¹…Ñ…É…µ”¡É½ÝÌ¤¹Ñ½}ÍØ¡MY}%H€¼˜‰íµ½‘•±}­•åõ}Ñ½­•¹¥é…Ñ¥½¹}…Õ‘¥Ð¹ÍØˆ°¥¹‘•àõ…±Í”°•¹½‘¥¹œô‰ÕÑ˜´àµÍ¥œˆ¤((€€€¸€ô±•¸¡‘˜¤(€€€Ñ•áÑÌ€ô‘™l‰Ñ•áÐ‰t¹Ñ½±¥ÍÐ ¤(€€€H€ô¹À¹é•É½Ì ¡¸°¹Õµ}±…å•ÉÌ¤°‘ÑåÁ”õ¹À¹™±½…ÐÌÈ¤((€€€Ý¥Ñ Ñ½É ¹¹½}É… ¤è(€€€€€€€™½ÈÍÑ…ÉÐ¥¸É…¹” À°¸°	Q!}M%i¤è(€€€€€€€€€€€•¹€ôµ¥¸¡¸°ÍÑ…ÉÐ€¬	Q!}M%i¤(€€€€€€€€€€€‰…Ñ €ôÑ•áÑÍmÍÑ…ÉÐé•¹‘t(€€€€€€€€€€€¥¹ÁÕÑÌ€ôÑ½¬¡‰…Ñ °É•ÑÕÉ¹}Ñ•¹Í½ÉÌô‰ÁÐˆ°Á…‘‘¥¹œõQÉÕ”°ÑÉÕ¹…Ñ¥½¸õQÉÕ”°µ…á}±•¹Ñ õ5a}18¤¹Ñ¼¡µ½‘•°¹‘•Ù¥”¤(€€€€€€€€€€€½ÕÑÁÕÑÌ€ôµ½‘•° ¨©¥¹ÁÕÑÌ°½ÕÑÁÕÑ}¡¥‘‘•¹}ÍÑ…Ñ•ÌõQÉÕ”°ÕÍ•}…¡”õ…±Í”¤(€€€€€€€€€€€¡ÍÑ…Ñ•Ì€ô½ÕÑÁÕÑÌ¹¡¥‘‘•¹}ÍÑ…Ñ•Ì(€€€€€€€€€€€Á½Ì€ô±…ÍÑ}Á½Í¥Ñ¥½¹Ì¡¥¹ÁÕÑÍl‰…ÑÑ•¹Ñ¥½¹}µ…Í¬‰t¤(€€€€€€€€€€€‰Íè€ô•¹€´ÍÑ…ÉÐ(€€€€€€€€€€€¥‘Í}Ð€ôÑ½É ¹Ñ•¹Í½È¡±•…¹}¥‘ÍmÍÑ…ÉÐé•¹‘t°‘ÑåÁ”õÑ½É ¹±½¹œ°‘•Ù¥”õµ½‘•°¹‘•Ù¥”¤(€€€€€€€€€€€•¥‘Í}Ð€ôÑ½É ¹Ñ•¹Í½È¡½¹™±¥Ñ}¥‘ÍmÍÑ…ÉÐé•¹‘t°‘ÑåÁ”õÑ½É ¹±½¹œ°‘•Ù¥”õµ½‘•°¹‘•Ù¥”¤((€€€€€€€€€€€™½È°¥¸É…¹”¡¹Õµ}±…å•ÉÌ¤è(€€€€€€€€€€€€€€€ €ô¡ÍÑ…Ñ•Ím°€¬€ÅumÑ½É ¹…É…¹”¡‰Íè°‘•Ù¥”õµ½‘•°¹‘•Ù¥”¤°Á½Ì°€ét¹‘•Ñ…  ¤¹™±½…Ð ¤(€€€€€€€€€€€€€€€ÉŒ€ôÑ½É ¹ÍÕ´¡ €¨]m¥‘Í}Ñt¹™±½…Ð ¤°‘¥´ôÄ¤(€€€€€€€€€€€€€€€É”€ôÑ½É ¹ÍÕ´¡ €¨]m•¥‘Í}Ñt¹™±½…Ð ¤°‘¥´ôÄ¤(€€€€€€€€€€€€€€€ImÍÑ…ÉÐé•¹°±t€ô€¡ÉŒ€´É”¤¹‘•Ñ…  ¤¹ÁÔ ¤¹¹ÕµÁä ¤¹…ÍÑåÁ”¡¹À¹™±½…ÐÌÈ¤((€€€€€€€€€€€ÁÉ¥¹Ð¡˜ˆ€ÁÉ½•ÍÍ•í•¹‘ô½í¹ôˆ¤((€€€€€€€€€€€‘•°½ÕÑÁÕÑÌ°¡ÍÑ…Ñ•Ì°¥¹ÁÕÑÌ(€€€€€€€€€€€Œ¹½±±•Ð ¤(€€€€€€€€€€€¥˜Ñ½É ¥Ì¹½Ð9½¹”…¹Ñ½É ¹Õ‘„¹¥Í}…Ù…¥±…‰±” ¤è(€€€€€€€€€€€€€€€Ñ½É ¹Õ‘„¹•µÁÑå}…¡” ¤((€€€‘•°µ½‘•°°Ñ½¬°\(€€€Œ¹½±±•Ð ¤(€€€¥˜Ñ½É ¥Ì¹½Ð9½¹”…¹Ñ½É ¹Õ‘„¹¥Í}…Ù…¥±…‰±” ¤è(€€€€€€€Ñ½É ¹Õ‘„¹•µÁÑå}…¡” ¤((€€€É•ÑÕÉ¸H°¹Õµ}±…å•ÉÌ()‘•˜ÁÉ½•ÍÍ}µ½‘•°¡µ½‘•±}­•ä°µ½‘•±}Á…Ñ °‘˜¤è(€€€H°0€ô•áÑÉ…Ñ}I}…±±}±…å•ÉÌ¡µ½‘•±}­•ä°µ½‘•±}Á…Ñ °‘˜¤(€€€‘•}±…å•ÉÌ€ô‘•¥Í¥½¹}±…å•ÉÌ¡0¤(€€€´Ñ‘}Ý¥¹‘½ÝÌ€ô±½…‘}´Ñ‘}Ý¥¹‘½ÝÌ¡µ½‘•±}­•ä°0¤((€€€µ½‘•±}‘˜€ô‘˜¹½Áä ¤(€€€™½È°¥¸É…¹”¡0¤è(€€€€€€€µ½‘•±}‘™m˜‰I}1í±ô‰t€ôIlè°±t((€€€€Œ±•…¸µÉ•±…Ñ¥Ù”‘H(€€€±•…¹}¥¹‘¥•Ì€ôí¥¹Ð¡É½Ü¹É…Á¡}¥¤è¥‘à™½È¥‘à°É½Ü¥¸µ½‘•±}‘˜¹É•Í•Ñ}¥¹‘•à¡‘É½ÀõQÉÕ”¤¹¥Ñ•ÉÉ½ÝÌ ¤¥˜É½Ýl‰½¹‘¥Ñ¥½¸‰t€ôô€‰±•…¸‰ô(€€€‘H€ô¹À¹é•É½Í}±¥­”¡H¤(€€€™½È¤°É½Ü¥¸µ½‘•±}‘˜¹É•Í•Ñ}¥¹‘•à¡‘É½ÀõQÉÕ”¤¹¥Ñ•ÉÉ½ÝÌ ¤è(€€€€€€€¤€ô±•…¹}¥¹‘¥•Ím¥¹Ð¡É½Ýl‰É…Á¡}¥‰t¥t(€€€€€€€‘Im¥t€ôIm¥t€´Im¥t((€€€™½È°¥¸É…¹”¡0¤è(€€€€€€€µ½‘•±}‘™m˜‰‘I}1í±ô‰t€ô‘Ilè°±t((€€€€Œ•±Ñ…T€ôAÄ½Ù•È‘•¥Í¥½¸‘HÝ¥¹‘½Ü¸(€€€a}‘•Œ€ô‘Ilè°‘•}±…å•ÉÍt(€€€Á„€ôA¡¹}½µÁ½¹•¹ÑÌõµ¥¸ Ì°a}‘•Œ¹Í¡…Á•lÅt¤¤(€€€ÁÌ€ôÁ„¹™¥Ñ}ÑÉ…¹Í™½É´¡a}‘•Œ¤(€€€‘•±Ñ…T€ôÁÍlè°€Át(€€€¥˜¹À¹µ•…¸¡‘•±Ñ…Umµ½‘•±}‘™l‰µ•¡…¹¥Í´‰t€ôô€‰ÍÑ…‰±”‰t¤€ð¹À¹µ•…¸¡‘•±Ñ…Umµ½‘•±}‘™l‰µ•¡…¹¥Í´‰t€ôô€‰±½ÍÕÉ”‰t¤è(€€€€€€€‘•±Ñ…T€ô€µ‘•±Ñ…T(€€€µ½‘•±}‘™l‰•±Ñ…T‰t€ô‘•±Ñ…T¹…ÍÑåÁ”¡¹À¹™±½…ÐÌÈ¤((€€€€Œ=}…‘ØÑÉ…¹Í¥Ñ¥½¹Ì(€€€<€ô‘Ilè°€Äét€´‘Ilè°€è´Åt€€ŒÍ¡…Á”¸à€¡0´Ä¤((€€€€Œ½¹‘¥Ñ¥½¹Ìµ•Ñ…‘…Ñ„¸(€€€äÌ€ôµ½‘•±}‘™l‰µ•¡…¹¥Í´‰t¹µ…À¡5!}5@Ì¤¹Ù…±Õ•Ì¹…ÍÑåÁ”¡¥¹Ð¤(€€€É½ÕÁÌ€ôµ½‘•±}‘™l‰É…Á¡}¥‰t¹Ù…±Õ•Ì¹…ÍÑåÁ”¡¥¹Ð¤((€€€€Œ<Ý¥¹‘½ÝÌ¸(€€€¥¹¥Ñ}‘½Ý¹ÍÑÉ•…´€ô´Ñ‘}Ý¥¹‘½ÝÌ¹•Ð ‰‘½Ý¹ÍÑÉ•…´ˆ°±¥ÍÐ¡É…¹” À°µ¥¸ È°0¤¤¤¤(€€€¥¹¥Ñ}µ•¡…¹¥Í´€ô´Ñ‘}Ý¥¹‘½ÝÌ¹•Ð ‰µ•¡…¹¥Í´ˆ°±¥ÍÐ¡É…¹” À°µ¥¸ Ø°0¤¤¤¤(€€€¥¹¥Ñ}½Ù•É…±°€ô´Ñ‘}Ý¥¹‘½ÝÌ¹•Ð ‰½Ù•É…±°ˆ°¥¹¥Ñ}‘½Ý¹ÍÑÉ•…´¤((€€€¥¹¥Ñ}‘½Ý¹}•¹€ôµ…à¡¥¹¥Ñ}‘½Ý¹ÍÑÉ•…´¤(€€€¥¹¥Ñ}µ•¡}•¹€ôµ…à¡¥¹¥Ñ}µ•¡…¹¥Í´¤(€€€‘•}ÍÑ…ÉÐ°‘•}•¹€ôµ¥¸¡‘•}±…å•ÉÌ¤°µ…à¡‘•}±…å•ÉÌ¤((€€€‘•˜ÑÉ…¹Í¥Ñ¥½¹}É…¹”¡„°ˆ¤è(€€€€€€€€ŒÑÉ…¹Í¥Ñ¥½¹Ì°µ•…¹Ì‘I}í°¬Åôµ‘I}°°Ù…±¥°ôÀ¸¹0´È(€€€€€€€„€ôµ…à À°µ¥¸¡0€´€È°„¤¤(€€€€€€€ˆ€ôµ…à¡„°µ¥¸¡0€´€È°ˆ¤¤(€€€€€€€É•ÑÕÉ¸±¥ÍÐ¡É…¹”¡„°ˆ€¬€Ä¤¤((€€€Ý¥¹‘½Ý}‘•™Ì€ôì(€€€€€€€€‰=}…±±}ÑÉ…¹Í¥Ñ¥½¹Ìˆè±¥ÍÐ¡É…¹” À°0€´€Ä¤¤°(€€€€€€€€‰=}ÁÉ•‘•¥Í¥½¸ˆèÑÉ…¹Í¥Ñ¥½¹}É…¹” À°‘•}ÍÑ…ÉÐ€´€Ä¤°(€€€€€€€€‰=}‘•¥Í¥½¸ˆèÑÉ…¹Í¥Ñ¥½¹}É…¹”¡‘•}ÍÑ…ÉÐ°‘•}•¹€´€Ä¤°(€€€€€€€€‰=}¥¹¥Ñ½Ý¹}Ñ½}‘•¥Í¥½¸ˆèÑÉ…¹Í¥Ñ¥½¹}É…¹”¡¥¹¥Ñ}‘½Ý¹}•¹°‘•}•¹€´€Ä¤°(€€€€€€€€‰=}¥¹¥Ñ5•¡}Ñ½}‘•¥Í¥½¸ˆèÑÉ…¹Í¥Ñ¥½¹}É…¹”¡¥¹¥Ñ}µ•¡}•¹°‘•}•¹€´€Ä¤°(€€€€€€€€‰=}…Õµ}…™Ñ•É}¥¹¥Ñ½Ý¸ˆèÑÉ…¹Í¥Ñ¥½¹}É…¹”¡¥¹¥Ñ}‘½Ý¹}•¹°‘•}ÍÑ…ÉÐ€´€Ä¤°(€€€€€€€€‰=}…Õµ}…™Ñ•É}¥¹¥Ñ5• ˆèÑÉ…¹Í¥Ñ¥½¹}É…¹”¡¥¹¥Ñ}µ•¡}•¹°‘•}ÍÑ…ÉÐ€´€Ä¤°(€€€€€€€€‰=}´Ñ‘}‘½Ý¹ÍÑÉ•…µ}Ý¥¹‘½ÜˆèÑÉ…¹Í¥Ñ¥½¹}É…¹”¡µ¥¸¡¥¹¥Ñ}‘½Ý¹ÍÑÉ•…´¤°µ…à¡¥¹¥Ñ}‘½Ý¹ÍÑÉ•…´¤€´€Ä¥˜±•¸¡¥¹¥Ñ}‘½Ý¹ÍÑÉ•…´¤€ø€Ä•±Í”µ…à¡¥¹¥Ñ}‘½Ý¹ÍÑÉ•…´¤¤°(€€€€€€€€‰=}´Ñ‘}µ•¡…¹¥Íµ}Ý¥¹‘½ÜˆèÑÉ…¹Í¥Ñ¥½¹}É…¹”¡µ¥¸¡¥¹¥Ñ}µ•¡…¹¥Í´¤°µ…à¡¥¹¥Ñ}µ•¡…¹¥Í´¤€´€Ä¥˜±•¸¡¥¹¥Ñ}µ•¡…¹¥Í´¤€ø€Ä•±Í”µ…à¡¥¹¥Ñ}µ•¡…¹¥Í´¤¤°(€€€ô((€€€€Œ]¥¹‘½ÜÁÉ•‘¥Ñ¥½¸ÍÕµµ…Éä¸(€€€É½ÝÌ€ômt(€€€™½ÈÝ¥¹}¹…µ”°ÑÉ…¹Í}±…å•ÉÌ¥¸Ý¥¹‘½Ý}‘•™Ì¹¥Ñ•µÌ ¤è(€€€€€€€ÑÉ…¹Í}±…å•ÉÌ€ôm°™½È°¥¸ÑÉ…¹Í}±…å•ÉÌ¥˜€À€ðô°€ð0€´€Åt(€€€€€€€¥˜±•¸¡ÑÉ…¹Í}±…å•ÉÌ¤€ôô€Àè(€€€€€€€€€€€½¹Ñ¥¹Õ”((€€€€€€€`€ô=lè°ÑÉ…¹Í}±…å•ÉÍt(€€€€€€€=}¥¹Ñ•É…°€ô`¹ÍÕ´¡…á¥ÌôÄ¤(€€€€€€€=}…‰Í}¥¹Ñ•É…°€ô¹À¹…‰Ì¡`¤¹ÍÕ´¡…á¥ÌôÄ¤(€€€€€€€=}•¹•Éä€ô¹À¹ÍÅÉÐ ¡`€¨¨€È¤¹ÍÕ´¡…á¥ÌôÄ¤¤(€€€€€€€a}…Õœ€ô¹À¹½±Õµ¹}ÍÑ…¬¡m`°=}¥¹Ñ•É…°°=}…‰Í}¥¹Ñ•É…°°=}•¹•Éåt¤¹…ÍÑåÁ”¡¹À¹™±½…ÐÌÈ¤((€€€€€€€ÈÈ°½ÉÈ€ôÉ¥‘•}Ø¡a}…Õœ°‘•±Ñ…T°É½ÕÁÌ¤(€€€€€€€…Œ°˜Ä€ô±½¥ÍÑ¥}Ø¡a}…Õœ°äÌ°É½ÕÁÌ¤((€€€€€€€É½ÝÌ¹…ÁÁ•¹¡ì(€€€€€€€€€€€€‰µ½‘•±}­•äˆèµ½‘•±}­•ä°(€€€€€€€€€€€€‰Ý¥¹‘½ÜˆèÝ¥¹}¹…µ”°(€€€€€€€€€€€€‰ÑÉ…¹Í¥Ñ¥½¹}±…å•ÉÌˆèÍÑÈ¡ÑÉ…¹Í}±…å•ÉÌ¤°(€€€€€€€€€€€€‰¹}ÑÉ…¹Í¥Ñ¥½¹Ìˆè±•¸¡ÑÉ…¹Í}±…å•ÉÌ¤°(€€€€€€€€€€€€‰¥¹Ñ•É…±}½ÉÉ}‘•±Ñ…Tˆè½ÉÉ}Í…™”¡=}¥¹Ñ•É…°°‘•±Ñ…T¤°(€€€€€€€€€€€€‰…‰Í}¥¹Ñ•É…±}½ÉÉ}‘•±Ñ…Tˆè½ÉÉ}Í…™”¡=}…‰Í}¥¹Ñ•É…°°‘•±Ñ…T¤°(€€€€€€€€€€€€‰•¹•Éå}½ÉÉ}‘•±Ñ…Tˆè½ÉÉ}Í…™”¡=}•¹•Éä°‘•±Ñ…T¤°(€€€€€€€€€€€€‰É¥‘•}‘•±Ñ…U}ÈÈˆèÈÈ°(€€€€€€€€€€€€‰É¥‘•}‘•±Ñ…U}½ÉÈˆè½ÉÈ°(€€€€€€€€€€€€‰µ•¡…¹¥Íµ}…Œˆè…Œ°(€€€€€€€€€€€€‰µ•¡…¹¥Íµ}µ…É½}˜Äˆè˜Ä°(€€€€€€€ô¤((€€€Ý¥¹}‘˜€ôÁ¹…Ñ…É…µ”¡É½ÝÌ¤(€€€Ý¥¹}‘˜¹Ñ½}ÍØ¡MY}%H€¼˜‰íµ½‘•±}­•åõ}Ý¥¹‘½Ý}ÁÉ•‘¥Ñ¥½¹}ÍÕµµ…Éä¹ÍØˆ°¥¹‘•àõ…±Í”°•¹½‘¥¹œô‰ÕÑ˜´àµÍ¥œˆ¤((€€€€Œ1…å•ÈµÝ¥Í”<½ÉÉ•±…Ñ¥½¸Ý¥Ñ •±Ñ…T…¹µ•¡…¹¥Í´Í•Á…É…Ñ¥½¸¸(€€€±…å•É}É½ÝÌ€ômt(€€€™½È°¥¸É…¹”¡0€´€Ä¤è(€€€€€€€½°€ô=lè°±t(€€€€€€€±½ÍÕÉ•}…ÕŒ€ôÍ…™•}…ÕŒ ¡äÌ€ôô€È¤¹…ÍÑåÁ”¡¥¹Ð¤°½°¤(€€€€€€€½µÁ}…ÕŒ€ôÍ…™•}…ÕŒ ¡äÌ€ôô€Ä¤¹…ÍÑåÁ”¡¥¹Ð¤°½°¤(€€€€€€€±…å•É}É½ÝÌ¹…ÁÁ•¹¡ì(€€€€€€€€€€€€‰µ½‘•±}­•äˆèµ½‘•±}­•ä°(€€€€€€€€€€€€‰ÑÉ…¹Í¥Ñ¥½¹}±…å•Èˆè°°(€€€€€€€€€€€€‰ÑÉ…¹Í¥Ñ¥½¸ˆè˜‰1í±ô´ù1í°¬Åôˆ°(€€€€€€€€€€€€‰±…å•É}™É…Œˆè°€¼µ…à Ä°0€´€È¤°(€€€€€€€€€€€€‰½ÉÉ}=}‘•±Ñ…Tˆè½ÉÉ}Í…™”¡½°°‘•±Ñ…T¤°(€€€€€€€€€€€€‰µ•…¹}=}ÍÑ…‰±”ˆè™±½…Ð¡¹À¹µ•…¸¡½±mäÌ€ôô€Át¤¤°(€€€€€€€€€€€€‰µ•…¹}=}½µÁ•Ñ¥Ñ¥½¸ˆè™±½…Ð¡¹À¹µ•…¸¡½±mäÌ€ôô€Åt¤¤°(€€€€€€€€€€€€‰µ•…¹}=}±½ÍÕÉ”ˆè™±½…Ð¡¹À¹µ•…¸¡½±mäÌ€ôô€Ét¤¤°(€€€€€€€€€€€€‰…Õ}±½ÍÕÉ•}‰å}<ˆè±½ÍÕÉ•}…ÕŒ°(€€€€€€€€€€€€‰…Õ}½µÁ•Ñ¥Ñ¥½¹}‰å}<ˆè½µÁ}…ÕŒ°(€€€€€€€€€€€€‰…‰Í}½ÉÉ}=}‘•±Ñ…Tˆè…‰Ì¡½ÉÉ}Í…™”¡½°°‘•±Ñ…T¤¤°(€€€€€€€ô¤(€€€±…å•É}‘˜€ôÁ¹…Ñ…É…µ”¡±…å•É}É½ÝÌ¤(€€€±…å•É}‘˜¹Ñ½}ÍØ¡MY}%H€¼˜‰íµ½‘•±}­•åõ}±…å•É}½}½ÉÈ¹ÍØˆ°¥¹‘•àõ…±Í”°•¹½‘¥¹œô‰ÕÑ˜´àµÍ¥œˆ¤((€€€€ŒQÉ…¹Í¥Ñ¥½¸‘…Ñ…Í•Ð¸(€€€ÑÉ…¹Í}É½ÝÌ€ômt(€€€™½È¤°É½Ü¥¸µ½‘•±}‘˜¹É•Í•Ñ}¥¹‘•à¡‘É½ÀõQÉÕ”¤¹¥Ñ•ÉÉ½ÝÌ ¤è(€€€€€€€‰…Í”€ôì(€€€€€€€€€€€€‰ÁÉ½µÁÑ}¥ˆèÉ½Ýl‰ÁÉ½µÁÑ}¥‰t°(€€€€€€€€€€€€‰É…Á¡}¥ˆè¥¹Ð¡É½Ýl‰É…Á¡}¥‰t¤°(€€€€€€€€€€€€‰½¹‘¥Ñ¥½¸ˆèÉ½Ýl‰½¹‘¥Ñ¥½¸‰t°(€€€€€€€€€€€€‰µ•¡…¹¥Í´ˆèÉ½Ýl‰µ•¡…¹¥Í´‰t°(€€€€€€€€€€€€‰•±Ñ…Tˆè™±½…Ð¡É½Ýl‰•±Ñ…T‰t¤°(€€€€€€€ô(€€€€€€€™½È°¥¸É…¹”¡0€´€Ä¤è(€€€€€€€€€€€ÑÉ…¹Í}É½ÝÌ¹…ÁÁ•¹¡ì(€€€€€€€€€€€€€€€€¨©‰…Í”°(€€€€€€€€€€€€€€€€‰ÑÉ…¹Í¥Ñ¥½¹}±…å•Èˆè°°(€€€€€€€€€€€€€€€€‰ÑÉ…¹Í¥Ñ¥½¸ˆè˜‰1í±ô´ù1í°¬Åôˆ°(€€€€€€€€€€€€€€€€‰=}…‘Øˆè™±½…Ð¡=m¤°±t¤°(€€€€€€€€€€€€€€€€‰‘I}°ˆè™±½…Ð¡‘Im¤°±t¤°(€€€€€€€€€€€€€€€€‰‘I}¹•áÐˆè™±½…Ð¡‘Im¤°°€¬€Åt¤°(€€€€€€€€€€€ô¤(€€€Á¹…Ñ…É…µ”¡ÑÉ…¹Í}É½ÝÌ¤¹Ñ½}ÍØ¡MY}%H€¼˜‰íµ½‘•±}­•åõ}ÑÉ…¹Í¥Ñ¥½¹}‘…Ñ…Í•Ð¹ÍØˆ°¥¹‘•àõ…±Í”°•¹½‘¥¹œô‰ÕÑ˜´àµÍ¥œˆ¤((€€€€ŒA•…­Ì…¹ÍÕµµ…É¥•Ì¸(€€€‰•ÍÑ}Ý¥¹}‘•±Ñ„€ôÝ¥¹}‘˜¹Í½ÉÑ}Ù…±Õ•Ì ‰É¥‘•}‘•±Ñ…U}½ÉÈˆ°…Í•¹‘¥¹œõ…±Í”¤¹¥±½lÁt¹Ñ½}‘¥Ð ¤(€€€‰•ÍÑ}Ý¥¹}µ• €ôÝ¥¹}‘˜¹Í½ÉÑ}Ù…±Õ•Ì ‰µ•¡…¹¥Íµ}µ…É½}˜Äˆ°…Í•¹‘¥¹œõ…±Í”¤¹¥±½lÁt¹Ñ½}‘¥Ð ¤(€€€Á•…­}½ÉÉ}É½Ü€ô±…å•É}‘˜¹Í½ÉÑ}Ù…±Õ•Ì ‰…‰Í}½ÉÉ}=}‘•±Ñ…Tˆ°…Í•¹‘¥¹œõ…±Í”¤¹¥±½lÁt¹Ñ½}‘¥Ð ¤((€€€€ŒAÉ½™¥±•Ì™½ÈÉ½ÍÌµµ½‘•°½µÁ…É¥Í½¸¸(€€€ÁÉ½™¥±•}½ÉÈ€ô¥¹Ñ•ÉÁ½±…Ñ•}ÁÉ½™¥±”¡±…å•É}‘™l‰½ÉÉ}=}‘•±Ñ…T‰t¹Ù…±Õ•Ì°¹}Á½¥¹ÑÌôÔÀ¤(€€€ÁÉ½™¥±•}…‰Ì€ô¥¹Ñ•ÉÁ½±…Ñ•}ÁÉ½™¥±”¡±…å•É}‘™l‰…‰Í}½ÉÉ}=}‘•±Ñ…T‰t¹Ù…±Õ•Ì°¹}Á½¥¹ÑÌôÔÀ¤(€€€¹À¹Í…Ù•é}½µÁÉ•ÍÍ• (€€€€€€€MY}%H€¼˜‰íµ½‘•±}­•åõ}½½¹Ñ}ÁÉ½™¥±•Ì¹¹Áèˆ°(€€€€€€€ÁÉ½™¥±•}½ÉÈõÁÉ½™¥±•}½ÉÈ°(€€€€€€€ÁÉ½™¥±•}…‰ÌõÁÉ½™¥±•}…‰Ì°(€€€€€€€É…Ý}½ÉÈõ±…å•É}‘™l‰½ÉÉ}=}‘•±Ñ…T‰t¹Ù…±Õ•Ì°(€€€€€€€±…å•É}™É…Œõ±…å•É}‘™l‰±…å•É}™É…Œ‰t¹Ù…±Õ•Ì°(€€€€¤((€€€ÍÕµµ…Éä€ôì(€€€€€€€€‰µ½‘•±}­•äˆèµ½‘•±}­•ä°(€€€€€€€€‰¹Õµ}±…å•ÉÌˆè0°(€€€€€€€€‰‘•¥Í¥½¹}±…å•ÉÌˆè‘•}±…å•ÉÌ°(€€€€€€€€‰´Ñ‘}Ý¥¹‘½ÝÌˆè´Ñ‘}Ý¥¹‘½ÝÌ°(€€€€€€€€‰‘•±Ñ…U}Á}Ù…É¥…¹”ˆèÁ„¹•áÁ±…¥¹•‘}Ù…É¥…¹•}É…Ñ¥½|¹Ñ½±¥ÍÐ ¤°(€€€€€€€€‰‘•±Ñ…U}ÁŒÅ}Ù…É¥…¹”ˆè™±½…Ð¡Á„¹•áÁ±…¥¹•‘}Ù…É¥…¹•}É…Ñ¥½}lÁt¤°(€€€€€€€€‰‰•ÍÑ}‘•±Ñ…U}Ý¥¹‘½Üˆè‰•ÍÑ}Ý¥¹}‘•±Ñ„°(€€€€€€€€‰‰•ÍÑ}µ•¡…¹¥Íµ}Ý¥¹‘½Üˆè‰•ÍÑ}Ý¥¹}µ• °(€€€€€€€€‰Á•…­}±…å•É}½ÉÈˆèÁ•…­}½ÉÉ}É½Ü°(€€€€€€€€‰Á…ÍÍ}¥¹Ñ•É…±}‘•±Ñ…U}½ÉÉ}Ñ|ÁÀÔˆè‰½½°¡‰•ÍÑ}Ý¥¹}‘•±Ñ…l‰É¥‘•}‘•±Ñ…U}½ÉÈ‰t€ø€À¸Ô¤°(€€€€€€€€‰Á…ÍÍ}µ•¡…¹¥Íµ}˜Å}Ñ|ÁÀØÔˆè‰½½°¡‰•ÍÑ}Ý¥¹}µ•¡l‰µ•¡…¹¥Íµ}µ…É½}˜Ä‰t€ø€À¸ØÔ¤°(€€€ô((€€€Ý¥Ñ ½Á•¸¡MY}%H€¼˜‰íµ½‘•±}­•åõ}½½¹Ñ}ÍÕµµ…Éä¹©Í½¸ˆ°€‰Üˆ°•¹½‘¥¹œô‰ÕÑ˜´àˆ¤…Ì˜è(€€€€€€€©Í½¸¹‘ÕµÀ¡ÍÕµµ…Éä°˜°¥¹‘•¹ÐôÈ°•¹ÍÕÉ•}…Í¥¤õ…±Í”¤((€€€µ½‘•±}‘˜¹Ñ½}ÍØ¡MY}%H€¼˜‰íµ½‘•±}­•åõ}I}‘…Ñ…Í•Ð¹ÍØˆ°¥¹‘•àõ…±Í”°•¹½‘¥¹œô‰ÕÑ˜´àµÍ¥œˆ¤((€€€É•ÑÕÉ¸ÍÕµµ…Éä()‘•˜½µÁÕÑ•}É½ÍÍ}µ½‘•±}ÁÉ½™¥±•Ì ¤è(€€€ÁÉ½™¥±•Ì€ôíô(€€€™½Èµ¬¥¸5=1}MALè(€€€€€€€‘…Ñ„€ô¹À¹±½…¡MY}%H€¼˜‰íµ­õ}½½¹Ñ}ÁÉ½™¥±•Ì¹¹Áèˆ°…±±½Ý}Á¥­±”õQÉÕ”¤(€€€€€€€ÁÉ½™¥±•Ímµ­t€ôì(€€€€€€€€€€€€‰½ÉÈˆè‘…Ñ…l‰ÁÉ½™¥±•}½ÉÈ‰t°(€€€€€€€€€€€€‰…‰Ìˆè‘…Ñ…l‰ÁÉ½™¥±•}…‰Ì‰t°(€€€€€€€ô((€€€É½ÝÌ€ômt(€€€™½È„°ˆ¥¸½µ‰¥¹…Ñ¥½¹Ì¡5=1}MAL¹­•åÌ ¤°€È¤è(€€€€€€€™½ÈÑåÀ¥¸l‰½ÉÈˆ°€‰…‰Ì‰tè(€€€€€€€€€€€á„€ôÁÉ½™¥±•Ím…umÑåÁt(€€€€€€€€€€€áˆ€ôÁÉ½™¥±•Ím‰umÑåÁt(€€€€€€€€€€€É½ÝÌ¹…ÁÁ•¹¡ì(€€€€€€€€€€€€€€€€‰ÁÉ½™¥±•}ÑåÁ”ˆèÑåÀ°(€€€€€€€€€€€€€€€€‰µ½‘•±}„ˆè„°(€€€€€€€€€€€€€€€€‰µ½‘•±}ˆˆèˆ°(€€€€€€€€€€€€€€€€‰Á•…ÉÍ½¹}ÁÉ½™¥±•}½ÉÈˆè½ÉÉ}Í…™”¡á„°áˆ¤°(€€€€€€€€€€€ô¤(€€€½ÕÐ€ôÁ¹…Ñ…É…µ”¡É½ÝÌ¤(€€€½ÕÐ¹Ñ½}ÍØ¡MY}%H€¼€‰´Õ…}É½ÍÍ}µ½‘•±}ÁÉ½™¥±•}½ÉÈ¹ÍØˆ°¥¹‘•àõ…±Í”°•¹½‘¥¹œô‰ÕÑ˜´àµÍ¥œˆ¤(€€€É•ÑÕÉ¸½ÕÐ((Œ€ôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôô(Œ5%8(Œ€ôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôô()‘•˜µ…¥¸ ¤è(€€€…ÉÌ€ôÁ…ÉÍ•}…ÉÌ ¤(€€€½¹™¥ÕÉ•}™É½µ}…ÉÌ¡…ÉÌ¤(€€€MY}%H¹µ­‘¥È¡Á…É•¹ÑÌõQÉÕ”°•á¥ÍÑ}½¬õQÉÕ”¤((€€€¥˜…ÉÌ¹¡•­}¥¹ÁÕÑÍ}½¹±äè(€€€€€€€ÝÉ¥Ñ•}¥¹ÁÕÑ}¡•­}É•Á½ÉÐ ¤(€€€€€€€É•ÑÕÉ¸((€€€•¹ÍÕÉ•}ÉÕ¹Ñ¥µ•}‘•Á•¹‘•¹¥•Ì ¤(€€€Í•Ñ}Í••¡M¤((€€€ÁÉ¥¹Ð ‰Õ‘¥Ñ¥¹œ½µµ½¸Í¥¹±”µÑ½­•¸±…‰•±Ì¸¸¸ˆ¤(€€€½µµ½¹}±…‰•±Ì€ô…Õ‘¥Ñ}½µµ½¹}Í¥¹±•}Ñ½­•¹}±…‰•±Ì¡5=1}MAL¤(€€€ÁÉ¥¹Ð ‰½µµ½¸±…‰•±Ìèˆ°½µµ½¹}±…‰•±Ì¤((€€€‘˜€ô‰Õ¥±‘}‘…Ñ…Í•Ð¡½µµ½¹}±…‰•±Ì¤(€€€‘˜¹Ñ½}ÍØ¡MY}%H€¼€‰´Õ…}ÁÉ½µÁÑ}‘…Ñ…Í•Ð¹ÍØˆ°¥¹‘•àõ…±Í”°•¹½‘¥¹œô‰ÕÑ˜´àµÍ¥œˆ¤(€€€ÁÉ¥¹Ð ‰…Ñ…Í•Ðèˆ°±•¸¡‘˜¤°€‰ÁÉ½µÁÑÌˆ¤((€€€ÍÕµµ…É¥•Ì€ômt(€€€™½Èµ¬°Á…Ñ ¥¸5=1}MAL¹¥Ñ•µÌ ¤è(€€€€€€€ÍÕµµ…É¥•Ì¹…ÁÁ•¹¡ÁÉ½•ÍÍ}µ½‘•°¡µ¬°Á…Ñ °‘˜¤¤((€€€€Œ±…ÑÑ•¸µ½‘•°ÍÕµµ…Éä™½ÈMX¸(€€€™±…Ñ}É½ÝÌ€ômt(€€€™½ÈÌ¥¸ÍÕµµ…É¥•Ìè(€€€€€€€É½Ü€ôì(€€€€€€€€€€€€‰µ½‘•±}­•äˆèÍl‰µ½‘•±}­•ä‰t°(€€€€€€€€€€€€‰¹Õµ}±…å•ÉÌˆèÍl‰¹Õµ}±…å•ÉÌ‰t°(€€€€€€€€€€€€‰‘•¥Í¥½¹}±…å•ÉÌˆèÍÑÈ¡Íl‰‘•¥Í¥½¹}±…å•ÉÌ‰t¤°(€€€€€€€€€€€€‰‘•±Ñ…U}ÁŒÅ}Ù…É¥…¹”ˆèÍl‰‘•±Ñ…U}ÁŒÅ}Ù…É¥…¹”‰t°(€€€€€€€€€€€€‰‰•ÍÑ}‘•±Ñ…U}Ý¥¹‘½ÜˆèÍl‰‰•ÍÑ}‘•±Ñ…U}Ý¥¹‘½Ü‰ul‰Ý¥¹‘½Ü‰t°(€€€€€€€€€€€€‰‰•ÍÑ}‘•±Ñ…U}½ÉÈˆèÍl‰‰•ÍÑ}‘•±Ñ…U}Ý¥¹‘½Ü‰ul‰É¥‘•}‘•±Ñ…U}½ÉÈ‰t°(€€€€€€€€€€€€‰‰•ÍÑ}‘•±Ñ…U}ÈÈˆèÍl‰‰•ÍÑ}‘•±Ñ…U}Ý¥¹‘½Ü‰ul‰É¥‘•}‘•±Ñ…U}ÈÈ‰t°(€€€€€€€€€€€€‰‰•ÍÑ}‘•±Ñ…U}¥¹Ñ•É…±}½ÉÈˆèÍl‰‰•ÍÑ}‘•±Ñ…U}Ý¥¹‘½Ü‰ul‰¥¹Ñ•É…±}½ÉÉ}‘•±Ñ…T‰t°(€€€€€€€€€€€€‰‰•ÍÑ}µ•¡…¹¥Íµ}Ý¥¹‘½ÜˆèÍl‰‰•ÍÑ}µ•¡…¹¥Íµ}Ý¥¹‘½Ü‰ul‰Ý¥¹‘½Ü‰t°(€€€€€€€€€€€€‰‰•ÍÑ}µ•¡…¹¥Íµ}˜ÄˆèÍl‰‰•ÍÑ}µ•¡…¹¥Íµ}Ý¥¹‘½Ü‰ul‰µ•¡…¹¥Íµ}µ…É½}˜Ä‰t°(€€€€€€€€€€€€‰‰•ÍÑ}µ•¡…¹¥Íµ}…ŒˆèÍl‰‰•ÍÑ}µ•¡…¹¥Íµ}Ý¥¹‘½Ü‰ul‰µ•¡…¹¥Íµ}…Œ‰t°(€€€€€€€€€€€€‰Á•…­}=}±…å•ÈˆèÍl‰Á•…­}±…å•É}½ÉÈ‰ul‰ÑÉ…¹Í¥Ñ¥½¸‰t°(€€€€€€€€€€€€‰Á•…­}=}±…å•É}™É…ŒˆèÍl‰Á•…­}±…å•É}½ÉÈ‰ul‰±…å•É}™É…Œ‰t°(€€€€€€€€€€€€‰Á•…­}…‰Í}½ÉÉ}=}‘•±Ñ…TˆèÍl‰Á•…­}±…å•É}½ÉÈ‰ul‰…‰Í}½ÉÉ}=}‘•±Ñ…T‰t°(€€€€€€€€€€€€‰Á…ÍÍ}¥¹Ñ•É…±}‘•±Ñ…U}½ÉÉ}Ñ|ÁÀÔˆèÍl‰Á…ÍÍ}¥¹Ñ•É…±}‘•±Ñ…U}½ÉÉ}Ñ|ÁÀÔ‰t°(€€€€€€€€€€€€‰Á…ÍÍ}µ•¡…¹¥Íµ}˜Å}Ñ|ÁÀØÔˆèÍl‰Á…ÍÍ}µ•¡…¹¥Íµ}˜Å}Ñ|ÁÀØÔ‰t°(€€€€€€€ô(€€€€€€€™±…Ñ}É½ÝÌ¹…ÁÁ•¹¡É½Ü¤((€€€µ½‘•±}ÍÕµµ…Éä€ôÁ¹…Ñ…É…µ”¡™±…Ñ}É½ÝÌ¤(€€€µ½‘•±}ÍÕµµ…Éä¹Ñ½}ÍØ¡MY}%H€¼€‰´Õ…}µ½‘•±}ÍÕµµ…Éä¹ÍØˆ°¥¹‘•àõ…±Í”°•¹½‘¥¹œô‰ÕÑ˜´àµÍ¥œˆ¤((€€€É½ÍÌ€ô½µÁÕÑ•}É½ÍÍ}µ½‘•±}ÁÉ½™¥±•Ì ¤((€€€½Ù•É…±°€ôì(€€€€€€€€‰…Õ‘¥Ðˆè€‰4´ÕÉ½ÍÌµ5½‘•°=}½¹Ð€¼‘Ù…¹Ñ…”µ±½ÜÕ‘¥Ðˆ°(€€€€€€€€‰½‰Í•ÉÙ…‰±•}ÁÉ½áäˆè€‰=}…‘Ù}°€ô‘I}í°¬Åô€´‘I}°°Ý¥Ñ ‘I}°€ôI}°¡½¹‘¥Ñ¥½¸¤µI}°¡±•…¹}…¹¡½È¤ˆ°(€€€€€€€€‰µ½‘•±ÌˆèÍÕµµ…É¥•Ì°(€€€€€€€€‰É½ÍÍ}µ½‘•±}ÁÉ½™¥±•}½ÉÈˆèÉ½ÍÌ¹Ñ½}‘¥Ð¡½É¥•¹Ðô‰É•½É‘Ìˆ¤°(€€€ô(€€€Ý¥Ñ ½Á•¸¡MY}%H€¼€‰´Õ…}½Ù•É…±±}ÍÕµµ…Éä¹©Í½¸ˆ°€‰Üˆ°•¹½‘¥¹œô‰ÕÑ˜´àˆ¤…Ì˜è(€€€€€€€©Í½¸¹‘ÕµÀ¡½Ù•É…±°°˜°¥¹‘•¹ÐôÈ°•¹ÍÕÉ•}…Í¥¤õ…±Í”¤((€€€ÁÉ¥¹Ð ‰q¹4´Õ½µÁ±•Ñ”¸ˆ¤(€€€ÁÉ¥¹Ð¡µ½‘•±}ÍÕµµ…Éä¤(€€€ÁÉ¥¹Ð¡É½ÍÌ¤()¥˜}}¹…µ•}|€ôô€‰}}µ…¥¹}|ˆè(€€€µ…¥¸ ¤(