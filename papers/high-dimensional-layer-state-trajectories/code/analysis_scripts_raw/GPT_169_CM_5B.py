# ============================================================
# CM-5B: Non-Tautological Early/Mid O-flow Audit
#
# Goal:
#   Fix CM-5A's telescoping caveat.
#
# CM-5A used:
#       O_l = dR_{l+1} - dR_l
#   and allowed windows that overlap the decision Î”U target.
#
# CM-5B forbids decision-window leakage:
#
#   Target:
#       Î”U_decision = PC1(dR over decision window)
#
#   Predictors:
#       O_profile from windows strictly before decision_start.
#
# Tests:
#   1. O_early / O_mid / O_predecision predict later Î”U.
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
SAVE_DIR = Path("cm5b_outputs")

SEED = 42
BATCH_SIZE = 4
MAX_LEN = 320
N_GRAPHS = 36

DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch is not None and DEVICE == "cuda" else (torch.float32 if torch is not None else None)

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

def parse_args():
    parser = argparse.ArgumentParser(description="CM-5B non-tautological early/mid O-flow audit")
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
        "script": "GPT_169_CM_5B.py",
        "audit": "CM-5B Non-Tautological Early/Mid O-flow Audit",
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
    report_path = SAVE_DIR / "cm5b_input_check_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"CM-5B input check report written to: {report_path}")

def ensure_runtime_dependencies():
    missing = [name for name, ok in dependency_status().items() if not ok]
    if missing:
        raise RuntimeError(
            "Missing required runtime dependencies for full CM-5B run: "
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
        raise RuntimeError("transformers is required for full CM-5B runs.")
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
    n_labels = len(common_l×¾ù¶‰žËkºwµçQmÑ•t€ô±˜¹ÁÉ•‘¥Ð¡amÑ•t¤(€€€É•ÑÕÉ¸™±½…Ð¡…ÕÉ…å}Í½É”¡ä°ÁÉ•¤¤°™±½…Ð¡˜Å}Í½É”¡ä°ÁÉ•°…Ù•É…”ô‰µ…É¼ˆ°é•É½}‘¥Ù¥Í¥½¸ôÀ¤¤()‘•˜¥¹Ñ•ÉÁ½±…Ñ•}ÁÉ½™¥±”¡Ù…±Õ•Ì°¹}Á½¥¹ÑÌôÔÀ¤è(€€€Ù…±Õ•Ì€ô¹À¹…Í…ÉÉ…ä¡Ù…±Õ•Ì°‘ÑåÁ”õ™±½…Ð¤(€€€¥˜±•¸¡Ù…±Õ•Ì¤€ôô€Äè(€€€€€€€É•ÑÕÉ¸¹À¹™Õ±°¡¹}Á½¥¹ÑÌ°Ù…±Õ•ÍlÁt°‘ÑåÁ”õ™±½…Ð¤(€€€à€ô¹À¹±¥¹ÍÁ…” À°€Ä°±•¸¡Ù…±Õ•Ì¤¤(€€€á¤€ô¹À¹±¥¹ÍÁ…” À°€Ä°¹}Á½¥¹ÑÌ¤(€€€É•ÑÕÉ¸¹À¹¥¹Ñ•ÉÀ¡á¤°à°Ù…±Õ•Ì¤((Œ€ôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôô(ŒaQIQ%=8(Œ€ôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôô()‘•˜•áÑÉ…Ñ}I}…±±}±…å•ÉÌ¡µ½‘•±}­•ä°µ½‘•±}Á…Ñ °‘˜¤è(€€€ÁÉ¥¹Ð¡˜‰q¸ôôôôôôôôôôíµ½‘•±}­•åô€ôôôôôôôôôôˆ¤(€€€Ñ½¬€ô±½…‘}Ñ½­•¹¥é•È¡µ½‘•±}Á…Ñ ¤(€€€µ½‘•°€ô±½…‘}µ½‘•°¡µ½‘•±}Á…Ñ ¤(€€€¹Õµ}±…å•ÉÌ€ô•Ñ}¹Õµ}±…å•ÉÌ¡µ½‘•°¤(€€€\€ô•Ñ}±µ}¡•…‘}Ý•¥¡Ð¡µ½‘•°¤¹‘•Ñ…  ¤¹™±½…Ð ¤¹Ñ¼¡µ½‘•°¹‘•Ù¥”¤(€€€ÁÉ¥¹Ð ‰¹Õµ}±…å•ÉÌèˆ°¹Õµ}±…å•ÉÌ¤((€€€±•…¹}¥‘Ì°½¹™±¥Ñ}¥‘Ì€ômt°mt(€€€Ñ½­•¹}É½ÝÌ€ômt(€€€™½È|°É½Ü¥¸‘˜¹¥Ñ•ÉÉ½ÝÌ ¤è(€€€€€€€¥‘Ì€ô½¹Ñ¥¹Õ…Ñ¥½¹}¥‘Ì¡Ñ½¬°É½Ýl‰±•…¹}±…‰•°‰t¤(€€€€€€€•¥‘Ì€ô½¹Ñ¥¹Õ…Ñ¥½¹}¥‘Ì¡Ñ½¬°É½Ýl‰½¹™±¥Ñ}±…‰•°‰t¤(€€€€€€€Ñ½­•¹}É½ÝÌ¹…ÁÁ•¹¡ì(€€€€€€€€€€€€‰ÁÉ½µÁÑ}¥ˆèÉ½Ýl‰ÁÉ½µÁÑ}¥‰t°(€€€€€€€€€€€€‰±•…¹}±…‰•°ˆèÉ½Ýl‰±•…¹}±…‰•°‰t°(€€€€€€€€€€€€‰½¹™±¥Ñ}±…‰•°ˆèÉ½Ýl‰½¹™±¥Ñ}±…‰•°‰t°(€€€€€€€€€€€€‰±•…¹}¥‘ÌˆèÍÑÈ¡¥‘Ì¤°(€€€€€€€€€€€€‰½¹™±¥Ñ}¥‘ÌˆèÍÑÈ¡•¥‘Ì¤°(€€€€€€€€€€€€‰‰½Ñ¡}Í¥¹±”ˆè¥¹Ð¡±•¸¡¥‘Ì¤€ôô€Ä…¹±•¸¡•¥‘Ì¤€ôô€Ä¤°(€€€€€€€ô¤(€€€€€€€¥˜±•¸¡¥‘Ì¤€„ô€Ä½È±•¸¡•¥‘Ì¤€„ô€Äè(€€€€€€€€€€€É…¥Í”IÕ¹Ñ¥µ•ÉÉ½È¡˜‰íµ½‘•±}­•åôè¹½¸µÍ¥¹±”±…‰•±ÌíÉ½Ýl±•…¹}±…‰•°uôí¥‘Íô°íÉ½Ýl½¹™±¥Ñ}±…‰•°uôí•¥‘Íôˆ¤(€€€€€€€±•…¹}¥‘Ì¹…ÁÁ•¹¡¥‘ÍlÁt¤(€€€€€€€½¹™±¥Ñ}¥‘Ì¹…ÁÁ•¹¡•¥‘ÍlÁt¤(€€€Á¹…Ñ…É…µ”¡Ñ½­•¹}É½ÝÌ¤¹Ñ½}ÍØ¡MY}%H€¼˜‰íµ½‘•±}­•åõ}Ñ½­•¹¥é…Ñ¥½¹}…Õ‘¥Ð¹ÍØˆ°¥¹‘•àõ…±Í”°•¹½‘¥¹œô‰ÕÑ˜´àµÍ¥œˆ¤((€€€¸€ô±•¸¡‘˜¤(€€€H€ô¹À¹é•É½Ì ¡¸°¹Õµ}±…å•ÉÌ¤°‘ÑåÁ”õ¹À¹™±½…ÐÌÈ¤(€€€Ñ•áÑÌ€ô‘™l‰Ñ•áÐ‰t¹Ñ½±¥ÍÐ ¤((€€€Ý¥Ñ Ñ½É ¹¹½}É… ¤è(€€€€€€€™½ÈÍÑ…ÉÐ¥¸É…¹” À°¸°	Q!}M%i¤è(€€€€€€€€€€€•¹€ôµ¥¸¡¸°ÍÑ…ÉÐ€¬	Q!}M%i¤(€€€€€€€€€€€¥¹ÁÕÑÌ€ôÑ½¬¡Ñ•áÑÍmÍÑ…ÉÐé•¹‘t°É•ÑÕÉ¹}Ñ•¹Í½ÉÌô‰ÁÐˆ°Á…‘‘¥¹œõQÉÕ”°ÑÉÕ¹…Ñ¥½¸õQÉÕ”°µ…á}±•¹Ñ õ5a}18¤¹Ñ¼¡µ½‘•°¹‘•Ù¥”¤(€€€€€€€€€€€½ÕÑÁÕÑÌ€ôµ½‘•° ¨©¥¹ÁÕÑÌ°½ÕÑÁÕÑ}¡¥‘‘•¹}ÍÑ…Ñ•ÌõQÉÕ”°ÕÍ•}…¡”õ…±Í”¤(€€€€€€€€€€€¡ÍÑ…Ñ•Ì€ô½ÕÑÁÕÑÌ¹¡¥‘‘•¹}ÍÑ…Ñ•Ì(€€€€€€€€€€€Á½Ì€ô±…ÍÑ}Á½Í¥Ñ¥½¹Ì¡¥¹ÁÕÑÍl‰…ÑÑ•¹Ñ¥½¹}µ…Í¬‰t¤(€€€€€€€€€€€‰Íè€ô•¹€´ÍÑ…ÉÐ(€€€€€€€€€€€¥‘Í}Ð€ôÑ½É ¹Ñ•¹Í½È¡±•…¹}¥‘ÍmÍÑ…ÉÐé•¹‘t°‘ÑåÁ”õÑ½É ¹±½¹œ°‘•Ù¥”õµ½‘•°¹‘•Ù¥”¤(€€€€€€€€€€€•¥‘Í}Ð€ôÑ½É ¹Ñ•¹Í½È¡½¹™±¥Ñ}¥‘ÍmÍÑ…ÉÐé•¹‘t°‘ÑåÁ”õÑ½É ¹±½¹œ°‘•Ù¥”õµ½‘•°¹‘•Ù¥”¤((€€€€€€€€€€€™½È°¥¸É…¹”¡¹Õµ}±…å•ÉÌ¤è(€€€€€€€€€€€€€€€ €ô¡ÍÑ…Ñ•Ím°€¬€ÅumÑ½É ¹…É…¹”¡‰Íè°‘•Ù¥”õµ½‘•°¹‘•Ù¥”¤°Á½Ì°€ét¹‘•Ñ…  ¤¹™±½…Ð ¤(€€€€€€€€€€€€€€€ÉŒ€ôÑ½É ¹ÍÕ´¡ €¨]m¥‘Í}Ñt¹™±½…Ð ¤°‘¥´ôÄ¤(€€€€€€€€€€€€€€€É”€ôÑ½É ¹ÍÕ´¡ €¨]m•¥‘Í}Ñt¹™±½…Ð ¤°‘¥´ôÄ¤(€€€€€€€€€€€€€€€ImÍÑ…ÉÐé•¹°±t€ô€¡ÉŒ€´É”¤¹‘•Ñ…  ¤¹ÁÔ ¤¹¹ÕµÁä ¤¹…ÍÑåÁ”¡¹À¹™±½…ÐÌÈ¤((€€€€€€€€€€€ÁÉ¥¹Ð¡˜ˆ€ÁÉ½•ÍÍ•í•¹‘ô½í¹ôˆ¤((€€€€€€€€€€€‘•°½ÕÑÁÕÑÌ°¡ÍÑ…Ñ•Ì°¥¹ÁÕÑÌ(€€€€€€€€€€€Œ¹½±±•Ð ¤(€€€€€€€€€€€¥˜Ñ½É ¥Ì¹½Ð9½¹”…¹Ñ½É ¹Õ‘„¹¥Í}…Ù…¥±…‰±” ¤è(€€€€€€€€€€€€€€€Ñ½É ¹Õ‘„¹•µÁÑå}…¡” ¤((€€€‘•°µ½‘•°°Ñ½¬°\(€€€Œ¹½±±•Ð ¤(€€€¥˜Ñ½É ¥Ì¹½Ð9½¹”…¹Ñ½É ¹Õ‘„¹¥Í}…Ù…¥±…‰±” ¤è(€€€€€€€Ñ½É ¹Õ‘„¹•µÁÑå}…¡” ¤((€€€É•ÑÕÉ¸H°¹Õµ}±…å•ÉÌ((Œ€ôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôô(ŒAI=ML5=0(Œ€ôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôô()‘•˜ÁÉ½•ÍÍ}µ½‘•°¡µ½‘•±}­•ä°µ½‘•±}Á…Ñ °‘˜¤è(€€€H°0€ô•áÑÉ…Ñ}I}…±±}±…å•ÉÌ¡µ½‘•±}­•ä°µ½‘•±}Á…Ñ °‘˜¤(€€€‘•}±…å•ÉÌ€ô‘•¥Í¥½¹}±…å•ÉÌ¡0¤(€€€‘•}ÍÑ…ÉÐ€ôµ¥¸¡‘•}±…å•ÉÌ¤(€€€‘•}•¹€ôµ…à¡‘•}±…å•ÉÌ¤(€€€´Ñ€ô±½…‘}´Ñ‘}Ý¥¹‘½ÝÌ¡µ½‘•±}­•ä¤((€€€µ½‘•±}‘˜€ô‘˜¹½Áä ¤(€€€™½È°¥¸É…¹”¡0¤è(€€€€€€€µ½‘•±}‘™m˜‰I}1í±ô‰t€ôIlè°±t((€€€€Œ±•…¸µÉ•±…Ñ¥Ù”‘H(€€€±•…¹}¥¹‘¥•Ì€ôí¥¹Ð¡É½Ü¹É…Á¡}¥¤è¥‘à™½È¥‘à°É½Ü¥¸µ½‘•±}‘˜¹É•Í•Ñ}¥¹‘•à¡‘É½ÀõQÉÕ”¤¹¥Ñ•ÉÉ½ÝÌ ¤¥˜É½Ýl‰½¹‘¥Ñ¥½¸‰t€ôô€‰±•…¸‰ô(€€€‘H€ô¹À¹é•É½Í}±¥­”¡H¤(€€€™½È¤°É½Ü¥¸µ½‘•±}‘˜¹É•Í•Ñ}¥¹‘•à¡‘É½ÀõQÉÕ”¤¹¥Ñ•ÉÉ½ÝÌ ¤è(€€€€€€€¤€ô±•…¹}¥¹‘¥•Ím¥¹Ð¡É½Ýl‰É…Á¡}¥‰t¥t(€€€€€€€‘Im¥t€ôIm¥t€´Im¥t((€€€™½È°¥¸É…¹”¡0¤è(€€€€€€€µ½‘•±}‘™m˜‰‘I}1í±ô‰t€ô‘Ilè°±t((€€€€ŒQ…É•Ðƒ:QTÍÑÉ¥Ñ±ä¥¸‘•¥Í¥½¸Ý¥¹‘½Ü¸(€€€a}‘•Œ€ô‘Ilè°‘•}±…å•ÉÍt(€€€Á„€ôA¡¹}½µÁ½¹•¹ÑÌõµ¥¸ Ì°a}‘•Œ¹Í¡…Á•lÅt¤¤(€€€ÁÌ€ôÁ„¹™¥Ñ}ÑÉ…¹Í™½É´¡a}‘•Œ¤(€€€‘•±Ñ…T€ôÁÍlè°€Át(€€€¥˜¹À¹µ•…¸¡‘•±Ñ…Umµ½‘•±}‘™l‰µ•¡…¹¥Í´‰t€ôô€‰ÍÑ…‰±”‰t¤€ð¹À¹µ•…¸¡‘•±Ñ…Umµ½‘•±}‘™l‰µ•¡…¹¥Í´‰t€ôô€‰±½ÍÕÉ”‰t¤è(€€€€€€€‘•±Ñ…T€ô€µ‘•±Ñ…T(€€€µ½‘•±}‘™l‰•±Ñ…U}‘•¥Í¥½¸‰t€ô‘•±Ñ…T¹…ÍÑåÁ”¡¹À¹™±½…ÐÌÈ¤((€€€<€ô‘Ilè°€Äét€´‘Ilè°€è´Åt€€Œ¸à0´Ä(€€€äÌ€ôµ½‘•±}‘™l‰µ•¡…¹¥Í´‰t¹µ…À¡5!}5@Ì¤¹Ù…±Õ•Ì¹…ÍÑåÁ”¡¥¹Ð¤(€€€É½ÕÁÌ€ôµ½‘•±}‘™l‰É…Á¡}¥‰t¹Ù…±Õ•Ì¹…ÍÑåÁ”¡¥¹Ð¤((€€€€ŒÍÑÉ¥Ñ±ä¹½¸µ±•…­¥¹œÑÉ…¹Í¥Ñ¥½¸±…å•ÉÌµÕÍÐÍ…Ñ¥Í™ä°¬Ä€ð‘•}ÍÑ…ÉÐ°¤¹”¸°€ðô‘•}ÍÑ…ÉÐ´È(€€€ÁÉ•}µ…à€ô‘•}ÍÑ…ÉÐ€´€È(€€€¥˜ÁÉ•}µ…à€ð€Àè(€€€€€€€ÁÉ•}µ…à€ô€À((€€€‘•˜ÑÉ…¹Í}É…¹”¡„°ˆ¤è(€€€€€€€„€ôµ…à À°µ¥¸¡0€´€È°„¤¤(€€€€€€€ˆ€ôµ…à¡„°µ¥¸¡0€´€È°ˆ¤¤(€€€€€€€€Œ•¹™½É”¹¼±•…­…”(€€€€€€€ˆ€ôµ¥¸¡ˆ°ÁÉ•}µ…à¤(€€€€€€€¥˜ˆ€ð„è(€€€€€€€€€€€É•ÑÕÉ¸mt(€€€€€€€É•ÑÕÉ¸±¥ÍÐ¡É…¹”¡„°ˆ€¬€Ä¤¤((€€€€Œ4Ñµ½‘•°µÍÁ•¥™¥ŒÝ¥¹‘½ÝÌ¸(€€€‘½Ý¹ÍÑÉ•…´€ô´Ñ¹•Ð ‰‘½Ý¹ÍÑÉ•…´ˆ°±¥ÍÐ¡É…¹” À°µ¥¸ Ì°0¤¤¤¤(€€€µ•¡…¹¥Í´€ô´Ñ¹•Ð ‰µ•¡…¹¥Í´ˆ°±¥ÍÐ¡É…¹” À°µ¥¸ Ø°0¤¤¤¤(€€€Ñ½Á½±½ä€ô´Ñ¹•Ð ‰Ñ½Á½±½äˆ°±¥ÍÐ¡É…¹” À°µ¥¸ È°0¤¤¤¤(€€€½Ù•É…±°€ô´Ñ¹•Ð ‰½Ù•É…±°ˆ°‘½Ý¹ÍÑÉ•…´¤((€€€‘}•¹€ôµ…à¡‘½Ý¹ÍÑÉ•…´¤¥˜‘½Ý¹ÍÑÉ•…´•±Í”€À(€€€µ}•¹€ôµ…à¡µ•¡…¹¥Í´¤¥˜µ•¡…¹¥Í´•±Í”€À(€€€Ñ}•¹€ôµ…à¡Ñ½Á½±½ä¤¥˜Ñ½Á½±½ä•±Í”€À(€€€½}•¹€ôµ…à¡½Ù•É…±°¤¥˜½Ù•É…±°•±Í”€À((€€€€Œ•™¥¹”¹½¸µ±•…­¥¹œÝ¥¹‘½ÝÌ¸(€€€•…É±å}•¹€ôµ…à Ä°¥¹Ð¡µ…Ñ ¹™±½½È ¡0€´€Ä¤€¨€À¸ÈÔ¤¤¤(€€€µ¥‘}•¹€ôµ…à¡•…É±å}•¹€¬€Ä°¥¹Ð¡µ…Ñ ¹™±½½È ¡0€´€Ä¤€¨€À¸ÔÔ¤¤¤((€€€Ý¥¹‘½Ý}‘•™Ì€ôì(€€€€€€€€‰=}•…É±å|Á|ÈÕ‘•ÁÑ ˆèÑÉ…¹Í}É…¹” À°•…É±å}•¹¤°(€€€€€€€€‰=}µ¥‘|ÈÕ|ÔÕ‘•ÁÑ ˆèÑÉ…¹Í}É…¹”¡•…É±å}•¹€¬€Ä°µ¥‘}•¹¤°(€€€€€€€€‰=}ÁÉ•‘•¥Í¥½¹}…±°ˆèÑÉ…¹Í}É…¹” À°‘•}ÍÑ…ÉÐ€´€È¤°(€€€€€€€€‰=}…™Ñ•É}Ñ½Á½±½å}Õ¹Ñ¥±}ÁÉ•‘•¥Í¥½¸ˆèÑÉ…¹Í}É…¹”¡Ñ}•¹°‘•}ÍÑ…ÉÐ€´€È¤°(€€€€€€€€‰=}…™Ñ•É}µ•¡…¹¥Íµ}Õ¹Ñ¥±}ÁÉ•‘•¥Í¥½¸ˆèÑÉ…¹Í}É…¹”¡µ}•¹°‘•}ÍÑ…ÉÐ€´€È¤°(€€€€€€€€‰=}…™Ñ•É}‘½Ý¹ÍÑÉ•…µ}Õ¹Ñ¥±}ÁÉ•‘•¥Í¥½¸ˆèÑÉ…¹Í}É…¹”¡‘}•¹°‘•}ÍÑ…ÉÐ€´€È¤°(€€€€€€€€‰=}…™Ñ•É}½Ù•É…±±}Õ¹Ñ¥±}ÁÉ•‘•¥Í¥½¸ˆèÑÉ…¹Í}É…¹”¡½}•¹°‘•}ÍÑ…ÉÐ€´€È¤°(€€€€€€€€‰=}´Ñ‘}µ•¡…¹¥Íµ}½¹±äˆèÑÉ…¹Í}É…¹”¡µ¥¸¡µ•¡…¹¥Í´¤¥˜µ•¡…¹¥Í´•±Í”€À°µ…à¡µ•¡…¹¥Í´¤´Ä¥˜±•¸¡µ•¡…¹¥Í´¤€ø€Ä•±Í”µ…à¡µ•¡…¹¥Í´¤¥˜µ•¡…¹¥Í´•±Í”€À¤°(€€€€€€€€‰=}´Ñ‘}‘½Ý¹ÍÑÉ•…µ}½¹±äˆèÑÉ…¹Í}É…¹”¡µ¥¸¡‘½Ý¹ÍÑÉ•…´¤¥˜‘½Ý¹ÍÑÉ•…´•±Í”€À°µ…à¡‘½Ý¹ÍÑÉ•…´¤´Ä¥˜±•¸¡‘½Ý¹ÍÑÉ•…´¤€ø€Ä•±Í”µ…à¡‘½Ý¹ÍÑÉ•…´¤¥˜‘½Ý¹ÍÑÉ•…´•±Í”€À¤°(€€€ô((€€€É½ÝÌ€ômt(€€€™½ÈÝ¥¹}¹…µ”°±…å•ÉÌ¥¸Ý¥¹‘½Ý}‘•™Ì¹¥Ñ•µÌ ¤è(€€€€€€€±…å•ÉÌ€ôm°™½È°¥¸±…å•ÉÌ¥˜€À€ðô°€ð0€´€Ä…¹°€ðôÁÉ•}µ…át(€€€€€€€¥˜±•¸¡±…å•ÉÌ¤€ôô€Àè(€€€€€€€€€€€½¹Ñ¥¹Õ”(€€€€€€€`€ô=lè°±…å•ÉÍt(€€€€€€€=}¥¹Ñ•É…°€ô`¹ÍÕ´¡…á¥ÌôÄ¤(€€€€€€€=}…‰Í}¥¹Ñ•É…°€ô¹À¹…‰Ì¡`¤¹ÍÕ´¡…á¥ÌôÄ¤(€€€€€€€=}•¹•Éä€ô¹À¹ÍÅÉÐ ¡`€¨¨€È¤¹ÍÕ´¡…á¥ÌôÄ¤¤(€€€€€€€=}µ•…¸€ô`¹µ•…¸¡…á¥ÌôÄ¤(€€€€€€€=}Í±½Á”€ôalè°€´Åt€´alè°€Át¥˜`¹Í¡…Á•lÅt€ø€Ä•±Í”¹À¹é•É½Ì¡`¹Í¡…Á•lÁt¤(€€€€€€€a}…Õœ€ô¹À¹½±Õµ¹}ÍÑ…¬¡m`°=}¥¹Ñ•É…°°=}…‰Í}¥¹Ñ•É…°°=}•¹•Éä°=}µ•…¸°=}Í±½Á•t¤¹…ÍÑåÁ”¡¹À¹™±½…ÐÌÈ¤((€€€€€€€ÈÈ°½ÉÈ€ôÉ¥‘•}Ø¡a}…Õœ°‘•±Ñ…T°É½ÕÁÌ¤(€€€€€€€…Œ°˜Ä€ô±½¥ÍÑ¥}Ø¡a}…Õœ°äÌ°É½ÕÁÌ¤((€€€€€€€É½ÝÌ¹…ÁÁ•¹¡ì(€€€€€€€€€€€€‰µ½‘•±}­•äˆèµ½‘•±}­•ä°(€€€€€€€€€€€€‰Ý¥¹‘½ÜˆèÝ¥¹}¹…µ”°(€€€€€€€€€€€€‰ÑÉ…¹Í¥Ñ¥½¹}±…å•ÉÌˆèÍÑÈ¡±…å•ÉÌ¤°(€€€€€€€€€€€€‰¹}ÑÉ…¹Í¥Ñ¥½¹Ìˆè±•¸¡±…å•ÉÌ¤°(€€€€€€€€€€€€‰‘•¥Í¥½¹}ÍÑ…ÉÐˆè‘•}ÍÑ…ÉÐ°(€€€€€€€€€€€€‰‘•¥Í¥½¹}±…å•ÉÌˆèÍÑÈ¡‘•}±…å•ÉÌ¤°(€€€€€€€€€€€€‰±•…­…•}™É•”ˆèQÉÕ”°(€€€€€€€€€€€€‰¥¹Ñ•É…±}½ÉÉ}‘•±Ñ…Tˆè½ÉÉ}Í…™”¡=}¥¹Ñ•É…°°‘•±Ñ…T¤°(€€€€€€€€€€€€‰…‰Í}¥¹Ñ•É…±}½ÉÉ}‘•±Ñ…Tˆè½ÉÉ}Í…™”¡=}…‰Í}¥¹Ñ•É…°°‘•±Ñ…T¤°(€€€€€€€€€€€€‰•¹•Éå}½ÉÉ}‘•±Ñ…Tˆè½ÉÉ}Í…™”¡=}•¹•Éä°‘•±Ñ…T¤°(€€€€€€€€€€€€‰µ•…¹}½ÉÉ}‘•±Ñ…Tˆè½ÉÉ}Í…™”¡=}µ•…¸°‘•±Ñ…T¤°(€€€€€€€€€€€€‰Í±½Á•}½ÉÉ}‘•±Ñ…Tˆè½ÉÉ}Í…™”¡=}Í±½Á”°‘•±Ñ…T¤°(€€€€€€€€€€€€‰É¥‘•}‘•±Ñ…U}ÈÈˆèÈÈ°(€€€€€€€€€€€€‰É¥‘•}‘•±Ñ…U}½ÉÈˆè½ÉÈ°(€€€€€€€€€€€€‰µ•¡…¹¥Íµ}…Œˆè…Œ°(€€€€€€€€€€€€‰µ•¡…¹¥Íµ}µ…É½}˜Äˆè˜Ä°(€€€€€€€ô¤((€€€Ý¥¹}‘˜€ôÁ¹…Ñ…É…µ”¡É½ÝÌ¤(€€€Ý¥¹}‘˜¹Ñ½}ÍØ¡MY}%H€¼˜‰íµ½‘•±}­•åõ}¹½¹±•…­}Ý¥¹‘½Ý}ÍÕµµ…Éä¹ÍØˆ°¥¹‘•àõ…±Í”°•¹½‘¥¹œô‰ÕÑ˜´àµÍ¥œˆ¤((€€€€Œ1…å•ÉÝ¥Í”ÁÉ•‘•¥Í¥½¸½ÉÉ•±…Ñ¥½¹Ì¸(€€€±…å•É}É½ÝÌ€ômt(€€€™½È°¥¸É…¹” À°ÁÉ•}µ…à€¬€Ä¤è(€€€€€€€½°€ô=lè°±t(€€€€€€€±…å•É}É½ÝÌ¹…ÁÁ•¹¡ì(€€€€€€€€€€€€‰µ½‘•±}­•äˆèµ½‘•±}­•ä°(€€€€€€€€€€€€‰ÑÉ…¹Í¥Ñ¥½¹}±…å•Èˆè°°(€€€€€€€€€€€€‰ÑÉ…¹Í¥Ñ¥½¸ˆè˜‰1í±ô´ù1í°¬Åôˆ°(€€€€€€€€€€€€‰±…å•É}™É…Œˆè°€¼µ…à Ä°0€´€È¤°(€€€€€€€€€€€€‰‘•¥Í¥½¹}ÍÑ…ÉÐˆè‘•}ÍÑ…ÉÐ°(€€€€€€€€€€€€‰½ÉÉ}=}‘•±Ñ…U}‘•¥Í¥½¸ˆè½ÉÉ}Í…™”¡½°°‘•±Ñ…T¤°(€€€€€€€€€€€€‰…‰Í}½ÉÉ}=}‘•±Ñ…U}‘•¥Í¥½¸ˆè…‰Ì¡½ÉÉ}Í…™”¡½°°‘•±Ñ…T¤¤°(€€€€€€€€€€€€‰µ•…¹}=}ÍÑ…‰±”ˆè™±½…Ð¡¹À¹µ•…¸¡½±mäÌ€ôô€Át¤¤°(€€€€€€€€€€€€‰µ•…¹}=}½µÁ•Ñ¥Ñ¥½¸ˆè™±½…Ð¡¹À¹µ•…¸¡½±mäÌ€ôô€Åt¤¤°(€€€€€€€€€€€€‰µ•…¹}=}±½ÍÕÉ”ˆè™±½…Ð¡¹À¹µ•…¸¡½±mäÌ€ôô€Ét¤¤°(€€€€€€€€€€€€‰…Õ}±½ÍÕÉ•}‰å}<ˆèÍ…™•}…ÕŒ ¡äÌ€ôô€È¤¹…ÍÑåÁ”¡¥¹Ð¤°½°¤°(€€€€€€€€€€€€‰…Õ}½µÁ•Ñ¥Ñ¥½¹}‰å}<ˆèÍ…™•}…ÕŒ ¡äÌ€ôô€Ä¤¹…ÍÑåÁ”¡¥¹Ð¤°½°¤°(€€€€€€€ô¤(€€€±…å•É}‘˜€ôÁ¹…Ñ…É…µ”¡±…å•É}É½ÝÌ¤(€€€±…å•É}‘˜¹Ñ½}ÍØ¡MY}%H€¼˜‰íµ½‘•±}­•åõ}ÁÉ•‘•¥Í¥½¹}±…å•É}½ÉÈ¹ÍØˆ°¥¹‘•àõ…±Í”°•¹½‘¥¹œô‰ÕÑ˜´àµÍ¥œˆ¤((€€€€ŒÁÉ½™¥±•Ì™½ÈÉ½ÍÌµµ½‘•°½µÁ…É¥Í½¸¸(€€€ÁÉ½™¥±•}…‰Ì€ô¥¹Ñ•ÉÁ½±…Ñ•}ÁÉ½™¥±”¡±…å•É}‘™l‰…‰Í}½ÉÉ}=}‘•±Ñ…U}‘•¥Í¥½¸‰t¹Ù…±Õ•Ì°€ÔÀ¤(€€€ÁÉ½™¥±•}Í¥¹•€ô¥¹Ñ•ÉÁ½±…Ñ•}ÁÉ½™¥±”¡±…å•É}‘™l‰½ÉÉ}=}‘•±Ñ…U}‘•¥Í¥½¸‰t¹Ù…±Õ•Ì°€ÔÀ¤(€€€¹À¹Í…Ù•é}½µÁÉ•ÍÍ• (€€€€€€€MY}%H€¼˜‰íµ½‘•±}­•åõ}ÁÉ•‘•¥Í¥½¹}ÁÉ½™¥±•Ì¹¹Áèˆ°(€€€€€€€ÁÉ½™¥±•}…‰ÌõÁÉ½™¥±•}…‰Ì°(€€€€€€€ÁÉ½™¥±•}Í¥¹•õÁÉ½™¥±•}Í¥¹•°(€€€€€€€É…Ý}…‰Ìõ±…å•É}‘™l‰…‰Í}½ÉÉ}=}‘•±Ñ…U}‘•¥Í¥½¸‰t¹Ù…±Õ•Ì°(€€€€€€€É…Ý}Í¥¹•õ±…å•É}‘™l‰½ÉÉ}=}‘•±Ñ…U}‘•¥Í¥½¸‰t¹Ù…±Õ•Ì°(€€€€€€€±…å•É}™É…Œõ±…å•É}‘™l‰±…å•É}™É…Œ‰t¹Ù…±Õ•Ì°(€€€€¤((€€€‰•ÍÑ}‘•±Ñ„€ôÝ¥¹}‘˜¹Í½ÉÑ}Ù…±Õ•Ì ‰É¥‘•}‘•±Ñ…U}½ÉÈˆ°…Í•¹‘¥¹œõ…±Í”¤¹¥±½lÁt¹Ñ½}‘¥Ð ¤(€€€‰•ÍÑ}µ• €ôÝ¥¹}‘˜¹Í½ÉÑ}Ù…±Õ•Ì ‰µ•¡…¹¥Íµ}µ…É½}˜Äˆ°…Í•¹‘¥¹œõ…±Í”¤¹¥±½lÁt¹Ñ½}‘¥Ð ¤(€€€Á•…­}±…å•È€ô±…å•É}‘˜¹Í½ÉÑ}Ù…±Õ•Ì ‰…‰Í}½ÉÉ}=}‘•±Ñ…U}‘•¥Í¥½¸ˆ°…Í•¹‘¥¹œõ…±Í”¤¹¥±½lÁt¹Ñ½}‘¥Ð ¤¥˜±•¸¡±…å•É}‘˜¤•±Í”íô((€€€€ŒMÑÉ¥Ð½¹ÑÉ½±Ìè(€€€€Œ€´½µÁ…É”ÁÉ•‘•¥Í¥½¸Ý¥Ñ ´Ñµ½¹±äÝ¥¹‘½Ü¸(€€€€Œ€´¹¼‘•¥Í¥½¸ÑÉ…¹Í¥Ñ¥½¹Ì…É”ÕÍ•¸(€€€ÍÕµµ…Éä€ôì(€€€€€€€€‰µ½‘•±}­•äˆèµ½‘•±}­•ä°(€€€€€€€€‰¹Õµ}±…å•ÉÌˆè0°(€€€€€€€€‰‘•¥Í¥½¹}±…å•ÉÌˆè‘•}±…å•ÉÌ°(€€€€€€€€‰‘•¥Í¥½¹}ÍÑ…ÉÐˆè‘•}ÍÑ…ÉÐ°(€€€€€€€€‰´Ñ‘}Ý¥¹‘½ÝÌˆè´Ñ°(€€€€€€€€‰‘•±Ñ…U}Á}Ù…É¥…¹”ˆèÁ„¹•áÁ±…¥¹•‘}Ù…É¥…¹•}É…Ñ¥½|¹Ñ½±¥ÍÐ ¤°(€€€€€€€€‰‘•±Ñ…U}ÁŒÅ}Ù…É¥…¹”ˆè™±½…Ð¡Á„¹•áÁ±…¥¹•‘}Ù…É¥…¹•}É…Ñ¥½}lÁt¤°(€€€€€€€€‰ÁÉ•‘•¥Í¥½¹}µ…á}ÑÉ…¹Í¥Ñ¥½¸ˆèÁÉ•}µ…à°(€€€€€€€€‰‰•ÍÑ}¹½¹±•…­}‘•±Ñ…U}Ý¥¹‘½Üˆè‰•ÍÑ}‘•±Ñ„°(€€€€€€€€‰‰•ÍÑ}¹½¹±•…­}µ•¡…¹¥Íµ}Ý¥¹‘½Üˆè‰•ÍÑ}µ• °(€€€€€€€€‰Á•…­}ÁÉ•‘•¥Í¥½¹}±…å•É}½ÉÈˆèÁ•…­}±…å•È°(€€€€€€€€‰Á…ÍÍ}¹½¹±•…­}‘•±Ñ…U}½ÉÉ}Ñ|ÁÀÐˆè‰½½°¡‰•ÍÑ}‘•±Ñ…l‰É¥‘•}‘•±Ñ…U}½ÉÈ‰t€ø€À¸Ð¤°(€€€€€€€€‰Á…ÍÍ}¹½¹±•…­}µ•¡…¹¥Íµ}˜Å}Ñ|ÁÀØÔˆè‰½½°¡‰•ÍÑ}µ•¡l‰µ•¡…¹¥Íµ}µ…É½}˜Ä‰t€ø€À¸ØÔ¤°(€€€ô((€€€Ý¥Ñ ½Á•¸¡MY}%H€¼˜‰íµ½‘•±}­•åõ}´Õ‰}ÍÕµµ…Éä¹©Í½¸ˆ°€‰Üˆ°•¹½‘¥¹œô‰ÕÑ˜´àˆ¤…Ì˜è(€€€€€€€©Í½¸¹‘ÕµÀ¡ÍÕµµ…Éä°˜°¥¹‘•¹ÐôÈ°•¹ÍÕÉ•}…Í¥¤õ…±Í”¤((€€€µ½‘•±}‘˜¹Ñ½}ÍØ¡MY}%H€¼˜‰íµ½‘•±}­•åõ}I}‘…Ñ…Í•Ð¹ÍØˆ°¥¹‘•àõ…±Í”°•¹½‘¥¹œô‰ÕÑ˜´àµÍ¥œˆ¤((€€€É•ÑÕÉ¸ÍÕµµ…Éä()‘•˜½µÁÕÑ•}É½ÍÍ}µ½‘•±}ÁÉ½™¥±•Ì ¤è(€€€ÁÉ½™¥±•Ì€ôíô(€€€™½Èµ¬¥¸5=1}MALè(€€€€€€€‘…Ñ„€ô¹À¹±½…¡MY}%H€¼˜‰íµ­õ}ÁÉ•‘•¥Í¥½¹}ÁÉ½™¥±•Ì¹¹Áèˆ°…±±½Ý}Á¥­±”õQÉÕ”¤(€€€€€€€ÁÉ½™¥±•Ímµ­t€ôì(€€€€€€€€€€€€‰…‰Ìˆè‘…Ñ…l‰ÁÉ½™¥±•}…‰Ì‰t°(€€€€€€€€€€€€‰Í¥¹•ˆè‘…Ñ…l‰ÁÉ½™¥±•}Í¥¹•‰t°(€€€€€€€ô((€€€É½ÝÌ€ômt(€€€™½È„°ˆ¥¸½µ‰¥¹…Ñ¥½¹Ì¡5=1}MAL¹­•åÌ ¤°€È¤è(€€€€€€€™½ÈÑåÀ¥¸l‰…‰Ìˆ°€‰Í¥¹•‰tè(€€€€€€€€€€€É½ÝÌ¹…ÁÁ•¹¡ì(€€€€€€€€€€€€€€€€‰ÁÉ½™¥±•}ÑåÁ”ˆèÑåÀ°(€€€€€€€€€€€€€€€€‰µ½‘•±}„ˆè„°(€€€€€€€€€€€€€€€€‰µ½‘•±}ˆˆèˆ°(€€€€€€€€€€€€€€€€‰Á•…ÉÍ½¹}ÁÉ½™¥±•}½ÉÈˆè½ÉÉ}Í…™”¡ÁÉ½™¥±•Ím…umÑåÁt°ÁÉ½™¥±•Ím‰umÑåÁt¤°(€€€€€€€€€€€ô¤(€€€½ÕÐ€ôÁ¹…Ñ…É…µ”¡É½ÝÌ¤(€€€½ÕÐ¹Ñ½}ÍØ¡MY}%H€¼€‰´Õ‰}É½ÍÍ}µ½‘•±}ÁÉ½™¥±•}½ÉÈ¹ÍØˆ°¥¹‘•àõ…±Í”°•¹½‘¥¹œô‰ÕÑ˜´àµÍ¥œˆ¤(€€€É•ÑÕÉ¸½ÕÐ((Œ€ôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôô(Œ5%8(Œ€ôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôô()‘•˜µ…¥¸ ¤è(€€€…ÉÌ€ôÁ…ÉÍ•}…ÉÌ ¤(€€€½¹™¥ÕÉ•}™É½µ}…ÉÌ¡…ÉÌ¤(€€€MY}%H¹µ­‘¥È¡Á…É•¹ÑÌõQÉÕ”°•á¥ÍÑ}½¬õQÉÕ”¤((€€€¥˜…ÉÌ¹¡•­}¥¹ÁÕÑÍ}½¹±äè(€€€€€€€ÝÉ¥Ñ•}¥¹ÁÕÑ}¡•­}É•Á½ÉÐ ¤(€€€€€€€É•ÑÕÉ¸((€€€•¹ÍÕÉ•}ÉÕ¹Ñ¥µ•}‘•Á•¹‘•¹¥•Ì ¤(€€€Í•Ñ}Í••¡M¤((€€€ÁÉ¥¹Ð ‰Õ‘¥Ñ¥¹œ½µµ½¸Í¥¹±”µÑ½­•¸±…‰•±Ì¸¸¸ˆ¤(€€€½µµ½¹}±…‰•±Ì€ô…Õ‘¥Ñ}½µµ½¹}Í¥¹±•}Ñ½­•¹}±…‰•±Ì¡5=1}MAL¤(€€€ÁÉ¥¹Ð ‰½µµ½¸±…‰•±Ìèˆ°½µµ½¹}±…‰•±Ì¤((€€€‘˜€ô‰Õ¥±‘}‘…Ñ…Í•Ð¡½µµ½¹}±…‰•±Ì¤(€€€‘˜¹Ñ½}ÍØ¡MY}%H€¼€‰´Õ‰}ÁÉ½µÁÑ}‘…Ñ…Í•Ð¹ÍØˆ°¥¹‘•àõ…±Í”°•¹½‘¥¹œô‰ÕÑ˜´àµÍ¥œˆ¤(€€€ÁÉ¥¹Ð ‰…Ñ…Í•Ðèˆ°±•¸¡‘˜¤°€‰ÁÉ½µÁÑÌˆ¤((€€€ÍÕµµ…É¥•Ì€ômt(€€€™½Èµ¬°Á…Ñ ¥¸5=1}MAL¹¥Ñ•µÌ ¤è(€€€€€€€ÍÕµµ…É¥•Ì¹…ÁÁ•¹¡ÁÉ½•ÍÍ}µ½‘•°¡µ¬°Á…Ñ °‘˜¤¤((€€€™±…Ð€ômt(€€€™½ÈÌ¥¸ÍÕµµ…É¥•Ìè(€€€€€€€‰€ôÍl‰‰•ÍÑ}¹½¹±•…­}‘•±Ñ…U}Ý¥¹‘½Ü‰t(€€€€€€€‰´€ôÍl‰‰•ÍÑ}¹½¹±•…­}µ•¡…¹¥Íµ}Ý¥¹‘½Ü‰t(€€€€€€€Á•…¬€ôÍl‰Á•…­}ÁÉ•‘•¥Í¥½¹}±…å•É}½ÉÈ‰t(€€€€€€€™±…Ð¹…ÁÁ•¹¡ì(€€€€€€€€€€€€‰µ½‘•±}­•äˆèÍl‰µ½‘•±}­•ä‰t°(€€€€€€€€€€€€‰¹Õµ}±…å•ÉÌˆèÍl‰¹Õµ}±…å•ÉÌ‰t°(€€€€€€€€€€€€‰‘•¥Í¥½¹}±…å•ÉÌˆèÍÑÈ¡Íl‰‘•¥Í¥½¹}±…å•ÉÌ‰t¤°(€€€€€€€€€€€€‰‘•¥Í¥½¹}ÍÑ…ÉÐˆèÍl‰‘•¥Í¥½¹}ÍÑ…ÉÐ‰t°(€€€€€€€€€€€€‰ÁÉ•‘•¥Í¥½¹}µ…á}ÑÉ…¹Í¥Ñ¥½¸ˆèÍl‰ÁÉ•‘•¥Í¥½¹}µ…á}ÑÉ…¹Í¥Ñ¥½¸‰t°(€€€€€€€€€€€€‰‘•±Ñ…U}ÁŒÅ}Ù…É¥…¹”ˆèÍl‰‘•±Ñ…U}ÁŒÅ}Ù…É¥…¹”‰t°(€€€€€€€€€€€€‰‰•ÍÑ}¹½¹±•…­}‘•±Ñ…U}Ý¥¹‘½Üˆè‰‘l‰Ý¥¹‘½Ü‰t°(€€€€€€€€€€€€‰‰•ÍÑ}¹½¹±•…­}‘•±Ñ…U}±…å•ÉÌˆè‰‘l‰ÑÉ…¹Í¥Ñ¥½¹}±…å•ÉÌ‰t°(€€€€€€€€€€€€‰‰•ÍÑ}¹½¹±•…­}‘•±Ñ…U}½ÉÈˆè‰‘l‰É¥‘•}‘•±Ñ…U}½ÉÈ‰t°(€€€€€€€€€€€€‰‰•ÍÑ}¹½¹±•…­}‘•±Ñ…U}ÈÈˆè‰‘l‰É¥‘•}‘•±Ñ…U}ÈÈ‰t°(€€€€€€€€€€€€‰‰•ÍÑ}¹½¹±•…­}¥¹Ñ•É…±}½ÉÈˆè‰‘l‰¥¹Ñ•É…±}½ÉÉ}‘•±Ñ…T‰t°(€€€€€€€€€€€€‰‰•ÍÑ}¹½¹±•…­}µ•¡…¹¥Íµ}Ý¥¹‘½Üˆè‰µl‰Ý¥¹‘½Ü‰t°(€€€€€€€€€€€€‰‰•ÍÑ}¹½¹±•…­}µ•¡…¹¥Íµ}±…å•ÉÌˆè‰µl‰ÑÉ…¹Í¥Ñ¥½¹}±…å•ÉÌ‰t°(€€€€€€€€€€€€‰‰•ÍÑ}¹½¹±•…­}µ•¡…¹¥Íµ}˜Äˆè‰µl‰µ•¡…¹¥Íµ}µ…É½}˜Ä‰t°(€€€€€€€€€€€€‰‰•ÍÑ}¹½¹±•…­}µ•¡…¹¥Íµ}…Œˆè‰µl‰µ•¡…¹¥Íµ}…Œ‰t°(€€€€€€€€€€€€‰Á•…­}ÁÉ•‘•¥Í¥½¹}ÑÉ…¹Í¥Ñ¥½¸ˆèÁ•…¬¹•Ð ‰ÑÉ…¹Í¥Ñ¥½¸ˆ°9½¹”¤°(€€€€€€€€€€€€‰Á•…­}ÁÉ•‘•¥Í¥½¹}±…å•É}™É…ŒˆèÁ•…¬¹•Ð ‰±…å•É}™É…Œˆ°9½¹”¤°(€€€€€€€€€€€€‰Á•…­}ÁÉ•‘•¥Í¥½¹}…‰Í}½ÉÈˆèÁ•…¬¹•Ð ‰…‰Í}½ÉÉ}=}‘•±Ñ…U}‘•¥Í¥½¸ˆ°9½¹”¤°(€€€€€€€€€€€€‰Á…ÍÍ}¹½¹±•…­}‘•±Ñ…U}½ÉÉ}Ñ|ÁÀÐˆèÍl‰Á…ÍÍ}¹½¹±•…­}‘•±Ñ…U}½ÉÉ}Ñ|ÁÀÐ‰t°(€€€€€€€€€€€€‰Á…ÍÍ}¹½¹±•…­}µ•¡…¹¥Íµ}˜Å}Ñ|ÁÀØÔˆèÍl‰Á…ÍÍ}¹½¹±•…­}µ•¡…¹¥Íµ}˜Å}Ñ|ÁÀØÔ‰t°(€€€€€€€ô¤((€€€µ½‘•±}ÍÕµµ…Éä€ôÁ¹…Ñ…É…µ”¡™±…Ð¤(€€€µ½‘•±}ÍÕµµ…Éä¹Ñ½}ÍØ¡MY}%H€¼€‰´Õ‰}µ½‘•±}ÍÕµµ…Éä¹ÍØˆ°¥¹‘•àõ…±Í”°•¹½‘¥¹œô‰ÕÑ˜´àµÍ¥œˆ¤((€€€É½ÍÌ€ô½µÁÕÑ•}É½ÍÍ}µ½‘•±}ÁÉ½™¥±•Ì ¤((€€€½Ù•É…±°€ôì(€€€€€€€€‰…Õ‘¥Ðˆè€‰4´Õ9½¸µQ…ÕÑ½±½¥…°…É±ä½5¥<µ™±½ÜÕ‘¥Ðˆ°(€€€€€€€€‰‘•™¥¹¥Ñ¥½¸ˆè€ (€€€€€€€€€€€€‰UÍ•Ì½¹±äÁÉ”µ‘•¥Í¥½¸=}…‘ØÑÉ…¹Í¥Ñ¥½¹ÌÑ¼ÁÉ•‘¥Ð±…Ñ•È‘•¥Í¥½¸µÝ¥¹‘½Ü•±Ñ…T¸€ˆ(€€€€€€€€€€€€‰Q¡¥Ì™½É‰¥‘ÌÑ•±•Í½Á¥¹œ½Ù•É±…ÀÝ¥Ñ Ñ¡”Ñ…É•ÐÝ¥¹‘½Ü¸ˆ(€€€€€€€€¤°(€€€€€€€€‰µ½‘•±ÌˆèÍÕµµ…É¥•Ì°(€€€€€€€€‰É½ÍÍ}µ½‘•±}ÁÉ½™¥±•}½ÉÈˆèÉ½ÍÌ¹Ñ½}‘¥Ð¡½É¥•¹Ðô‰É•½É‘Ìˆ¤°(€€€ô(€€€Ý¥Ñ ½Á•¸¡MY}%H€¼€‰´Õ‰}½Ù•É…±±}ÍÕµµ…Éä¹©Í½¸ˆ°€‰Üˆ°•¹½‘¥¹œô‰ÕÑ˜´àˆ¤…Ì˜è(€€€€€€€©Í½¸¹‘ÕµÀ¡½Ù•É…±°°˜°¥¹‘•¹ÐôÈ°•¹ÍÕÉ•}…Í¥¤õ…±Í”¤((€€€ÁÉ¥¹Ð ‰q¹4´Õ½µÁ±•Ñ”¸ˆ¤(€€€ÁÉ¥¹Ð¡µ½‘•±}ÍÕµµ…Éä¤(€€€ÁÉ¥¹Ð¡É½ÍÌ¤()¥˜}}¹…µ•}|€ôô€‰}}µ…¥¹}|ˆè(€€€µ…¥¸ ¤