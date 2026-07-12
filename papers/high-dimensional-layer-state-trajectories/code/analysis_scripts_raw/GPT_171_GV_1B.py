# ============================================================
# GV-1B: Geodesic-Biased Trajectory Audit
#
# Revised after theoretical correction:
#
#   Real Transformer trajectory is NOT expected to be identical
#   to a strict geodesic / shortest path.
#
#   Instead, it should be geodesic-biased:
#
#       trajectory direction has significant projection onto
#       a low-cost / endpoint direction,
#
#   while residual components encode:
#       correction
#       rotation
#       competition
#       boundary reconfiguration
#       layerwise local connection effects
#
# Core tests:
#
#   1. Direction alignment:
#        cos(Î”C_l, C_end - C_start)
#
#   2. Projection dominance:
#        ||Proj_g(Î”C_l)|| / ||Î”C_l||
#
#   3. Residual structure:
#        ||Î”C_l - Proj_g(Î”C_l)||
#
#   4. Path efficiency:
#        endpoint_distance / path_length
#
#   5. Controls:
#        shuffled-layer trajectory
#        random endpoint control
#        reversed trajectory control
#
#   6. Downstream prediction:
#        geodesic-bias metrics -> DeltaU_decision
#        geodesic-bias metrics -> stable / competition / closure
#
# Important:
#   This script does NOT claim:
#       trajectory == geodesic
#
#   It tests:
#       trajectory is biased toward a dominant geodesic direction,
#       with meaningful residuals.
#
# Run:
#   python gv1b_geodesic_biased_trajectory_audit.py
#
# Outputs:
#   gv1b_outputs/
#       gv1b_model_summary.csv
#       gv1b_cross_model_profile_corr.csv
#       gv1b_overall_summary.json
#       <model>_gv1b_summary.json
#       <model>_trajectory_metrics.csv
#       <model>_layer_profile.csv
#       <model>_window_eval.csv
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
SAVE_DIR = Path("gv1b_outputs")

SEED = 42
BATCH_SIZE = 4
MAX_LEN = 320
N_GRAPHS = 36
K_LIST = [100, 500]

DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch is not None and DEVICE == "cuda" else (torch.float32 if torch is not None else None)

# Decision target window.
DECISION_FRAC = (0.70, 0.92)

# Windows to evaluate.
BASE_WINDOWS = {
    "full": "full",
    "early_half": "early_half",
    "predecision": "predecision",
    "decision": "decision",
}

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
    parser = argparse.ArgumentParser(description="GV-1B geodesic-biased trajectory audit")
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
    parser.add_argument("--k-list", default="100,500", help="Comma-separated TopK values, e.g. 100,500")
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
    global SEED, BATCH_SIZE, MAX_LEN, N_GRAPHS, K_LIST

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
    K_LIST = [int(x.strip()) for x in str(args.k_list).split(",") if x.strip()]

def write_input_check_report():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "script": "GPT_171_GV_1B.py",
        "audit": "GV-1B Geodesic-Biased Trajectory Audit",
        "dependencies": dependency_status(),
        "configuration": {
            "device": DEVICE,
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "max_len": MAX_LEN,
            "n_graphs": N_GRAPHS,
            "k_list": K_LIST,
        },
        "inputs": {
            "cm4d_dir": path_status(CM4D_DIR),
            "model_paths": {name: path_status(path) for name, path in MODEL_SPECS.items()},
        },
        "outputs": {
            "save_dir": path_status(SAVE_DIR),
        },
    }
    report_path = SAVE_DIR / "gv1b_input_check_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"GV-1B input check report written to: {report_path}")

def ensure_runtime_dependencies():
    missing = [name for name, ok in dependency_status().items() if not ok]
    if missing:
        raise RuntimeError(
            "Missing required runtime dependencies for full GV-1B run: "
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
        raise RuntimeError("transformers is required for full GV-1B runs.")
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
            f"Override rule: in this case, items that {relation1} {b} receive label ëOy¶‰žËkºwµçA¥˜±•¸¡±…å•ÉÌ¤€ð€Èè(€€€€€€€€€€€€€€€½¹Ñ¥¹Õ”((€€€€€€€€€€€€ô¹À¹ÍÑ…¬¡m‘…Ñ…l‰•¹Ñ•ÉÌ‰um­um±t™½È°¥¸±…å•ÉÍt°…á¥ÌôÄ¤((€€€€€€€€€€€½¹ÑÉ½±Ì€ôì(€€€€€€€€€€€€€€€€‰É•…°ˆè°(€€€€€€€€€€€€€€€€‰Í¡Õ™™±”ˆèÍ¡Õ™™±•‘}±…å•É}½¹ÑÉ½°¡¤°(€€€€€€€€€€€€€€€€‰É…¹‘•¹ˆèÉ…¹‘½µ}•¹‘Á½¥¹Ñ}½¹ÑÉ½°¡¤°(€€€€€€€€€€€€€€€€‰É•Ù•ÉÍ”ˆèÉ•Ù•ÉÍ•‘}½¹ÑÉ½°¡¤°(€€€€€€€€€€€ô((€€€€€€€€€€€½¹ÑÉ½±}µ•ÑÉ¥Ì€ôíô(€€€€€€€€€€€½¹ÑÉ½±}ÁÉ½™¥±•Ì€ôíô((€€€€€€€€€€€™½È¹…µ”°Œ¥¸½¹ÑÉ½±Ì¹¥Ñ•µÌ ¤è(€€€€€€€€€€€€€€€´°±À€ô½µÁÕÑ•}•½‘•Í¥}‰¥…Í}µ•ÑÉ¥Ì¡Œ¤(€€€€€€€€€€€€€€€½¹ÑÉ½±}µ•ÑÉ¥Ím¹…µ•t€ô´(€€€€€€€€€€€€€€€½¹ÑÉ½±}ÁÉ½™¥±•Ím¹…µ•t€ô±À((€€€€€€€€€€€É•…±}´€ô½¹ÑÉ½±}µ•ÑÉ¥Íl‰É•…°‰t(€€€€€€€€€€€Í¡Õ™}´€ô½¹ÑÉ½±}µ•ÑÉ¥Íl‰Í¡Õ™™±”‰t(€€€€€€€€€€€É…¹‘}´€ô½¹ÑÉ½±}µ•ÑÉ¥Íl‰É…¹‘•¹‰t(€€€€€€€€€€€É•Ù}´€ô½¹ÑÉ½±}µ•ÑÉ¥Íl‰É•Ù•ÉÍ”‰t((€€€€€€€€€€€€Œ1…å•ÈÁÉ½™¥±”‰äµ•¡…¹¥Í´¸(€€€€€€€€€€€É•…±}±À€ô½¹ÑÉ½±}ÁÉ½™¥±•Íl‰É•…°‰t(€€€€€€€€€€€™½ÈÑ¤°°¥¸•¹Õµ•É…Ñ”¡±…å•ÉÍlè´Åt¤è(€€€€€€€€€€€€€€€™½Èµ•¡}¥°µ•¡}¹…µ”¥¸l À°€‰ÍÑ…‰±•}±¥­”ˆ¤°€ Ä°€‰½µÁ•Ñ¥Ñ¥½¸ˆ¤°€ È°€‰±½ÍÕÉ”ˆ¥tè(€€€€€€€€€€€€€€€€€€€µ…Í¬€ôäÌ€ôôµ•¡}¥(€€€€€€€€€€€€€€€€€€€¥˜µ…Í¬¹ÍÕ´ ¤€ôô€Àè(€€€€€€€€€€€€€€€€€€€€€€€½¹Ñ¥¹Õ”(€€€€€€€€€€€€€€€€€€€±…å•É}É½ÝÌ¹…ÁÁ•¹¡ì(€€€€€€€€€€€€€€€€€€€€€€€€‰µ½‘•±}­•äˆèµ½‘•±}­•ä°(€€€€€€€€€€€€€€€€€€€€€€€€‰¬ˆè¬°(€€€€€€€€€€€€€€€€€€€€€€€€‰Ý¥¹‘½ÜˆèÝ¥¹}¹…µ”°(€€€€€€€€€€€€€€€€€€€€€€€€‰ÑÉ…¹Í¥Ñ¥½¹}±…å•Èˆè°°(€€€€€€€€€€€€€€€€€€€€€€€€‰ÑÉ…¹Í¥Ñ¥½¹}¥¹‘•àˆèÑ¤°(€€€€€€€€€€€€€€€€€€€€€€€€‰±…å•É}™É…Œˆè°€¼µ…à Ä°0€´€Ä¤°(€€€€€€€€€€€€€€€€€€€€€€€€‰µ•¡…¹¥Íµ}¥ˆèµ•¡}¥°(€€€€€€€€€€€€€€€€€€€€€€€€‰µ•¡…¹¥Í´ˆèµ•¡}¹…µ”°(€€€€€€€€€€€€€€€€€€€€€€€€‰±½…±}…±¥¹}µ•…¸ˆè™±½…Ð¡¹À¹µ•…¸¡É•…±}±Ál‰±½…±}…±¥¸‰umµ…Í¬°Ñ¥t¤¤°(€€€€€€€€€€€€€€€€€€€€€€€€‰ÁÉ½©}Á½Í}É…Ñ¥½}µ•…¸ˆè™±½…Ð¡¹À¹µ•…¸¡É•…±}±Ál‰ÁÉ½©}Á½Í}É…Ñ¥¼‰umµ…Í¬°Ñ¥t¤¤°(€€€€€€€€€€€€€€€€€€€€€€€€‰É•Í¥‘Õ…±}É…Ñ¥½}µ•…¸ˆè™±½…Ð¡¹À¹µ•…¸¡É•…±}±Ál‰É•Í¥‘Õ…±}É…Ñ¥¼‰umµ…Í¬°Ñ¥t¤¤°(€€€€€€€€€€€€€€€€€€€€€€€€‰ÍÑ•Á}¹½Éµ}µ•…¸ˆè™±½…Ð¡¹À¹µ•…¸¡É•…±}±Ál‰ÍÑ•Á}¹½É´‰umµ…Í¬°Ñ¥t¤¤°(€€€€€€€€€€€€€€€€€€€€€€€€‰ÍÑ•Á}½Í}‘¥ÍÑ}µ•…¸ˆè™±½…Ð¡¹À¹µ•…¸¡É•…±}±Ál‰ÍÑ•Á}½Í}‘¥ÍÐ‰umµ…Í¬°Ñ¥t¤¤°(€€€€€€€€€€€€€€€€€€€ô¤((€€€€€€€€€€€™½È¤°É½Ü¥¸µ½‘•±}‘˜¹É•Í•Ñ}¥¹‘•à¡‘É½ÀõQÉÕ”¤¹¥Ñ•ÉÉ½ÝÌ ¤è(€€€€€€€€€€€€€€€É•Œ€ôì(€€€€€€€€€€€€€€€€€€€€‰µ½‘•±}­•äˆèµ½‘•±}­•ä°(€€€€€€€€€€€€€€€€€€€€‰¬ˆè¬°(€€€€€€€€€€€€€€€€€€€€‰Ý¥¹‘½ÜˆèÝ¥¹}¹…µ”°(€€€€€€€€€€€€€€€€€€€€‰±…å•ÉÌˆèÍÑÈ¡±…å•ÉÌ¤°(€€€€€€€€€€€€€€€€€€€€‰ÁÉ½µÁÑ}¥ˆèÉ½Ýl‰ÁÉ½µÁÑ}¥‰t°(€€€€€€€€€€€€€€€€€€€€‰É…Á¡}¥ˆè¥¹Ð¡É½Ýl‰É…Á¡}¥‰t¤°(€€€€€€€€€€€€€€€€€€€€‰½¹‘¥Ñ¥½¸ˆèÉ½Ýl‰½¹‘¥Ñ¥½¸‰t°(€€€€€€€€€€€€€€€€€€€€‰½¹‘¥Ñ¥½¹}±…ÍÌˆèÉ½Ýl‰½¹‘¥Ñ¥½¹}±…ÍÌ‰t°(€€€€€€€€€€€€€€€€€€€€‰µ•¡…¹¥Í´ˆèÉ½Ýl‰µ•¡…¹¥Í´‰t°(€€€€€€€€€€€€€€€€€€€€‰µ•¡…¹¥Íµ}¥ˆè¥¹Ð¡äÍm¥t¤°(€€€€€€€€€€€€€€€€€€€€‰•±Ñ…U}‘•¥Í¥½¸ˆè™±½…Ð¡‘•±Ñ…Um¥t¤°(€€€€€€€€€€€€€€€€€€€€‰¥¹Í¥‘•}‰…¹‘}ÁÉ½áäˆè¥¹Ð¡¥¹Í¥‘•}‰…¹‘m¥t¤°(€€€€€€€€€€€€€€€€€€€€‰‰…¹‘}±½Üˆè™±½…Ð¡‰…¹‘}±½Ü¤°(€€€€€€€€€€€€€€€€€€€€‰‰…¹‘}¡¥ ˆè™±½…Ð¡‰…¹‘}¡¥ ¤°(€€€€€€€€€€€€€€€ô((€€€€€€€€€€€€€€€™½È¹…µ”°´¥¸½¹ÑÉ½±}µ•ÑÉ¥Ì¹¥Ñ•µÌ ¤è(€€€€€€€€€€€€€€€€€€€™½Èµ¹…µ”°…ÉÈ¥¸´¹¥Ñ•µÌ ¤è(€€€€€€€€€€€€€€€€€€€€€€€É•m˜‰í¹…µ•õ}íµ¹…µ•ô‰t€ô™±½…Ð¡…ÉÉm¥t¤((€€€€€€€€€€€€€€€€Œ½¹ÑÉ½°µÉ•±…Ñ¥Ù”•Ù¥‘•¹”¸(€€€€€€€€€€€€€€€É•l‰É•…±}µ¥¹ÕÍ}Í¡Õ™™±•}…±¥¸‰t€ôÉ•l‰É•…±}…±¥¹}µ•…¸‰t€´É•l‰Í¡Õ™™±•}…±¥¹}µ•…¸‰t(€€€€€€€€€€€€€€€É•l‰É•…±}µ¥¹ÕÍ}Í¡Õ™™±•}ÁÉ½¨‰t€ôÉ•l‰É•…±}ÁÉ½©•Ñ¥½¹}‘½µ¥¹…¹•}Á½Ì‰t€´É•l‰Í¡Õ™™±•}ÁÉ½©•Ñ¥½¹}‘½µ¥¹…¹•}Á½Ì‰t(€€€€€€€€€€€€€€€É•l‰É•…±}µ¥¹ÕÍ}Í¡Õ™™±•}É•Í¥‘Õ…°‰t€ôÉ•l‰É•…±}É•Í¥‘Õ…±}É…Ñ¥½}µ•…¸‰t€´É•l‰Í¡Õ™™±•}É•Í¥‘Õ…±}É…Ñ¥½}µ•…¸‰t(€€€€€€€€€€€€€€€É•l‰É•…±}µ¥¹ÕÍ}Í¡Õ™™±•}‘•Ñ½ÕÈ‰t€ôÉ•l‰É•…±}‘•Ñ½ÕÉ}É…Ñ¥¼‰t€´É•l‰Í¡Õ™™±•}‘•Ñ½ÕÉ}É…Ñ¥¼‰t(€€€€€€€€€€€€€€€É•l‰É•…±}µ¥¹ÕÍ}Í¡Õ™™±•}ÕÉÙ…ÑÕÉ”‰t€ôÉ•l‰É•…±}ÕÉÙ…ÑÕÉ•}ÍÕ´‰t€´É•l‰Í¡Õ™™±•}ÕÉÙ…ÑÕÉ•}ÍÕ´‰t((€€€€€€€€€€€€€€€É•l‰É•…±}ÙÍ}É…¹‘•¹‘}•½‘•Í¥}Í½É”‰t€ôÉ•l‰É•…±}•½‘•Í¥}Í½É”‰t€¼€¡É•l‰É…¹‘•¹‘}•½‘•Í¥}Í½É”‰t€¬€Å”´à¤(€€€€€€€€€€€€€€€É•l‰É•…±}ÙÍ}É•Ù•ÉÍ•}…±¥¸‰t€ôÉ•l‰É•…±}…±¥¹}µ•…¸‰t€´É•l‰É•Ù•ÉÍ•}…±¥¹}µ•…¸‰t((€€€€€€€€€€€€€€€µ•ÑÉ¥}É½ÝÌ¹…ÁÁ•¹¡É•Œ¤((€€€µ•ÑÉ¥Í}‘˜€ôÁ¹…Ñ…É…µ”¡µ•ÑÉ¥}É½ÝÌ¤(€€€±…å•É}‘˜€ôÁ¹…Ñ…É…µ”¡±…å•É}É½ÝÌ¤((€€€µ•ÑÉ¥Í}‘˜¹Ñ½}ÍØ¡MY}%H€¼˜‰íµ½‘•±}­•åõ}ÑÉ…©•Ñ½Éå}µ•ÑÉ¥Ì¹ÍØˆ°¥¹‘•àõ…±Í”°•¹½‘¥¹œô‰ÕÑ˜´àµÍ¥œˆ¤(€€€±…å•É}‘˜¹Ñ½}ÍØ¡MY}%H€¼˜‰íµ½‘•±}­•åõ}±…å•É}ÁÉ½™¥±”¹ÍØˆ°¥¹‘•àõ…±Í”°•¹½‘¥¹œô‰ÕÑ˜´àµÍ¥œˆ¤(€€€µ½‘•±}‘˜¹Ñ½}ÍØ¡MY}%H€¼˜‰íµ½‘•±}­•åõ}I}‘…Ñ…Í•Ð¹ÍØˆ°¥¹‘•àõ…±Í”°•¹½‘¥¹œô‰ÕÑ˜´àµÍ¥œˆ¤((€€€€ŒÙ…±Õ…Ñ”•… Ý¥¹‘½Ü¸(€€€•Ù…±}É½ÝÌ€ômt(€€€™•…ÑÕÉ•}½±Ì€ôl(€€€€€€€€‰É•…±}Á…Ñ¡}±•¹Ñ ˆ°(€€€€€€€€‰É•…±}•¹‘Á½¥¹Ñ}‘¥ÍÑ…¹”ˆ°(€€€€€€€€‰É•…±}•½‘•Í¥}Í½É”ˆ°(€€€€€€€€‰É•…±}‘•Ñ½ÕÉ}É…Ñ¥¼ˆ°(€€€€€€€€‰É•…±}…±¥¹}µ•…¸ˆ°(€€€€€€€€‰É•…±}…±¥¹}µ¥¸ˆ°(€€€€€€€€‰É•…±}…±¥¹}™¥¹…°ˆ°(€€€€€€€€‰É•…±}ÁÉ½©}Á½Í}É…Ñ¥½}µ•…¸ˆ°(€€€€€€€€‰É•…±}ÁÉ½©}…‰Í}É…Ñ¥½}µ•…¸ˆ°(€€€€€€€€‰É•…±}ÁÉ½©•Ñ¥½¹}‘½µ¥¹…¹•}Á½Ìˆ°(€€€€€€€€‰É•…±}ÁÉ½©•Ñ¥½¹}‘½µ¥¹…¹•}…‰Ìˆ°(€€€€€€€€‰É•…±}É•Í¥‘Õ…±}É…Ñ¥½}µ•…¸ˆ°(€€€€€€€€‰É•…±}É•Í¥‘Õ…±}É…Ñ¥½}µ…àˆ°(€€€€€€€€‰É•…±}ÕÉÙ…ÑÕÉ•}ÍÕ´ˆ°(€€€€€€€€‰É•…±}ÕÉÙ…ÑÕÉ•}µ•…¸ˆ°(€€€€€€€€‰É•…±}ÍÁ••‘}ÍÑˆ°(€€€€€€€€‰É•…±}ÍÁ••‘}µ…àˆ°(€€€€€€€€‰É•…±}µ¥¹ÕÍ}Í¡Õ™™±•}…±¥¸ˆ°(€€€€€€€€‰É•…±}µ¥¹ÕÍ}Í¡Õ™™±•}ÁÉ½¨ˆ°(€€€€€€€€‰É•…±}µ¥¹ÕÍ}Í¡Õ™™±•}É•Í¥‘Õ…°ˆ°(€€€€€€€€‰É•…±}µ¥¹ÕÍ}Í¡Õ™™±•}‘•Ñ½ÕÈˆ°(€€€€€€€€‰É•…±}µ¥¹ÕÍ}Í¡Õ™™±•}ÕÉÙ…ÑÕÉ”ˆ°(€€€€€€€€‰É•…±}ÙÍ}É…¹‘•¹‘}•½‘•Í¥}Í½É”ˆ°(€€€€€€€€‰É•…±}ÙÍ}É•Ù•ÉÍ•}…±¥¸ˆ°(€€€t((€€€™½È€¡¬°Ý¥¸¤°ÍÕˆ¥¸µ•ÑÉ¥Í}‘˜¹É½ÕÁ‰ä¡l‰¬ˆ°€‰Ý¥¹‘½Ü‰t¤è(€€€€€€€`€ôÍÕ‰m™•…ÑÕÉ•}½±Ít¹Ù…±Õ•Ì¹…ÍÑåÁ”¡¹À¹™±½…ÐÌÈ¤(€€€€€€€å}‘•±Ñ„€ôÍÕ‰l‰•±Ñ…U}‘•¥Í¥½¸‰t¹Ù…±Õ•Ì¹…ÍÑåÁ”¡™±½…Ð¤(€€€€€€€å}µ• €ôÍÕ‰l‰µ•¡…¹¥Íµ}¥‰t¹Ù…±Õ•Ì¹…ÍÑåÁ”¡¥¹Ð¤(€€€€€€€å}‰…¹€ôÍÕ‰l‰¥¹Í¥‘•}‰…¹‘}ÁÉ½áä‰t¹Ù…±Õ•Ì¹…ÍÑåÁ”¡¥¹Ð¤(€€€€€€€œ€ôÍÕ‰l‰É…Á¡}¥‰t¹Ù…±Õ•Ì¹…ÍÑåÁ”¡¥¹Ð¤((€€€€€€€ÈÈ°½ÉÈ€ôÉ¥‘•}Ø¡`°å}‘•±Ñ„°œ¤(€€€€€€€…Œ°˜Ä€ô±½¥ÍÑ¥}Ø¡`°å}µ• °œ¤((€€€€€€€€Œ	…¹‘•Ñ•Ñ¥½¸Ý¥Ñ É•Í¥‘Õ…°€¼ÕÉÙ…ÑÕÉ”€¼‘•Ñ½ÕÈ¸(€€€€€€€…Õ}‰…¹‘}É•Í¥€ôÍ…™•}…ÕŒ¡å}‰…¹°ÍÕ‰l‰É•…±}É•Í¥‘Õ…±}É…Ñ¥½}µ•…¸‰t¹Ù…±Õ•Ì¤(€€€€€€€…Õ}‰…¹‘}ÕÉØ€ôÍ…™•}…ÕŒ¡å}‰…¹°ÍÕ‰l‰É•…±}ÕÉÙ…ÑÕÉ•}ÍÕ´‰t¹Ù…±Õ•Ì¤(€€€€€€€…Õ}‰…¹‘}‘•Ñ½ÕÈ€ôÍ…™•}…ÕŒ¡å}‰…¹°ÍÕ‰l‰É•…±}‘•Ñ½ÕÉ}É…Ñ¥¼‰t¹Ù…±Õ•Ì¤(€€€€€€€…Õ}‰…¹‘}±½Ý}•½}Í½É”€ôÍ…™•}…ÕŒ¡å}‰…¹°€µÍÕ‰l‰É•…±}•½‘•Í¥}Í½É”‰t¹Ù…±Õ•Ì¤((€€€€€€€€ŒI•…°ÙÌ½¹ÑÉ½±Ì¸(€€€€€€€É•…±}…±¥¸€ô™±½…Ð¡ÍÕ‰l‰É•…±}…±¥¹}µ•…¸‰t¹µ•…¸ ¤¤(€€€€€€€Í¡Õ™}…±¥¸€ô™±½…Ð¡ÍÕ‰l‰Í¡Õ™™±•}…±¥¹}µ•…¸‰t¹µ•…¸ ¤¤(€€€€€€€É•Ù}…±¥¸€ô™±½…Ð¡ÍÕ‰l‰É•Ù•ÉÍ•}…±¥¹}µ•…¸‰t¹µ•…¸ ¤¤((€€€€€€€É•…±}ÁÉ½¨€ô™±½…Ð¡ÍÕ‰l‰É•…±}ÁÉ½©•Ñ¥½¹}‘½µ¥¹…¹•}Á½Ì‰t¹µ•…¸ ¤¤(€€€€€€€Í¡Õ™}ÁÉ½¨€ô™±½…Ð¡ÍÕ‰l‰Í¡Õ™™±•}ÁÉ½©•Ñ¥½¹}‘½µ¥¹…¹•}Á½Ì‰t¹µ•…¸ ¤¤((€€€€€€€É•…±}‘•Ñ½ÕÈ€ô™±½…Ð¡ÍÕ‰l‰É•…±}‘•Ñ½ÕÉ}É…Ñ¥¼‰t¹µ•…¸ ¤¤(€€€€€€€Í¡Õ™}‘•Ñ½ÕÈ€ô™±½…Ð¡ÍÕ‰l‰Í¡Õ™™±•}‘•Ñ½ÕÉ}É…Ñ¥¼‰t¹µ•…¸ ¤¤((€€€€€€€É•…±}ÕÉØ€ô™±½…Ð¡ÍÕ‰l‰É•…±}ÕÉÙ…ÑÕÉ•}ÍÕ´‰t¹µ•…¸ ¤¤(€€€€€€€Í¡Õ™}ÕÉØ€ô™±½…Ð¡ÍÕ‰l‰Í¡Õ™™±•}ÕÉÙ…ÑÕÉ•}ÍÕ´‰t¹µ•…¸ ¤¤((€€€€€€€É•…±}É•Í¥€ô™±½…Ð¡ÍÕ‰l‰É•…±}É•Í¥‘Õ…±}É…Ñ¥½}µ•…¸‰t¹µ•…¸ ¤¤(€€€€€€€Í¡Õ™}É•Í¥€ô™±½…Ð¡ÍÕ‰l‰Í¡Õ™™±•}É•Í¥‘Õ…±}É…Ñ¥½}µ•…¸‰t¹µ•…¸ ¤¤((€€€€€€€•Ù…±}É½ÝÌ¹…ÁÁ•¹¡ì(€€€€€€€€€€€€‰µ½‘•±}­•äˆèµ½‘•±}­•ä°(€€€€€€€€€€€€‰¬ˆè¥¹Ð¡¬¤°(€€€€€€€€€€€€‰Ý¥¹‘½ÜˆèÝ¥¸°(€€€€€€€€€€€€‰¸ˆè¥¹Ð¡±•¸¡ÍÕˆ¤¤°(€€€€€€€€€€€€‰±…å•ÉÌˆèÍÑÈ¡ÍÕ‰l‰±…å•ÉÌ‰t¹¥±½lÁt¤°(€€€€€€€€€€€€‰‘•±Ñ…U}É¥‘•}ÈÈˆèÈÈ°(€€€€€€€€€€€€‰‘•±Ñ…U}É¥‘•}½ÉÈˆè½ÉÈ°(€€€€€€€€€€€€‰µ•¡…¹¥Íµ}…Œˆè…Œ°(€€€€€€€€€€€€‰µ•¡…¹¥Íµ}µ…É½}˜Äˆè˜Ä°(€€€€€€€€€€€€‰…Õ}¥¹Í¥‘•}‰…¹‘}‰å}É•Í¥‘Õ…°ˆè…Õ}‰…¹‘}É•Í¥°(€€€€€€€€€€€€‰…Õ}¥¹Í¥‘•}‰…¹‘}‰å}ÕÉÙ…ÑÕÉ”ˆè…Õ}‰…¹‘}ÕÉØ°(€€€€€€€€€€€€‰…Õ}¥¹Í¥‘•}‰…¹‘}‰å}‘•Ñ½ÕÈˆè…Õ}‰…¹‘}‘•Ñ½ÕÈ°(€€€€€€€€€€€€‰…Õ}¥¹Í¥‘•}‰…¹‘}‰å}±½Ý}•½‘•Í¥}Í½É”ˆè…Õ}‰…¹‘}±½Ý}•½}Í½É”°(€€€€€€€€€€€€‰É•…±}…±¥¹}µ•…¸ˆèÉ•…±}…±¥¸°(€€€€€€€€€€€€‰Í¡Õ™™±•}…±¥¹}µ•…¸ˆèÍ¡Õ™}…±¥¸°(€€€€€€€€€€€€‰É•Ù•ÉÍ•}…±¥¹}µ•…¸ˆèÉ•Ù}…±¥¸°(€€€€€€€€€€€€‰É•…±}…±¥¹}Ñ}Í¡Õ™™±”ˆè‰½½°¡É•…±}…±¥¸€øÍ¡Õ™}…±¥¸¤°(€€€€€€€€€€€€‰É•…±}…±¥¹}Ñ}É•Ù•ÉÍ”ˆè‰½½°¡É•…±}…±¥¸€øÉ•Ù}…±¥¸¤°(€€€€€€€€€€€€‰É•…±}ÁÉ½©}Á½Í}µ•…¸ˆèÉ•…±}ÁÉ½¨°(€€€€€€€€€€€€‰Í¡Õ™™±•}ÁÉ½©}Á½Í}µ•…¸ˆèÍ¡Õ™}ÁÉ½¨°(€€€€€€€€€€€€‰É•…±}ÁÉ½©}Ñ}Í¡Õ™™±”ˆè‰½½°¡É•…±}ÁÉ½¨€øÍ¡Õ™}ÁÉ½¨¤°(€€€€€€€€€€€€‰É•…±}‘•Ñ½ÕÉ}µ•…¸ˆèÉ•…±}‘•Ñ½ÕÈ°(€€€€€€€€€€€€‰Í¡Õ™™±•}‘•Ñ½ÕÉ}µ•…¸ˆèÍ¡Õ™}‘•Ñ½ÕÈ°(€€€€€€€€€€€€‰É•…±}‘•Ñ½ÕÉ}±Ñ}Í¡Õ™™±”ˆè‰½½°¡É•…±}‘•Ñ½ÕÈ€ðÍ¡Õ™}‘•Ñ½ÕÈ¤°(€€€€€€€€€€€€‰É•…±}ÕÉÙ…ÑÕÉ•}µ•…¸ˆèÉ•…±}ÕÉØ°(€€€€€€€€€€€€‰Í¡Õ™™±•}ÕÉÙ…ÑÕÉ•}µ•…¸ˆèÍ¡Õ™}ÕÉØ°(€€€€€€€€€€€€‰É•…±}ÕÉÙ…ÑÕÉ•}±Ñ}Í¡Õ™™±”ˆè‰½½°¡É•…±}ÕÉØ€ðÍ¡Õ™}ÕÉØ¤°(€€€€€€€€€€€€‰É•…±}É•Í¥‘Õ…±}µ•…¸ˆèÉ•…±}É•Í¥°(€€€€€€€€€€€€‰Í¡Õ™™±•}É•Í¥‘Õ…±}µ•…¸ˆèÍ¡Õ™}É•Í¥°(€€€€€€€ô¤((€€€•Ù…±}‘˜€ôÁ¹…Ñ…É…µ”¡•Ù…±}É½ÝÌ¤(€€€•Ù…±}‘˜¹Ñ½}ÍØ¡MY}%H€¼˜‰íµ½‘•±}­•åõ}Ý¥¹‘½Ý}•Ù…°¹ÍØˆ°¥¹‘•àõ…±Í”°•¹½‘¥¹œô‰ÕÑ˜´àµÍ¥œˆ¤((€€€€ŒA¥¬‰•ÍÐ‰ä‘¥™™•É•¹ÐÉ¥Ñ•É¥„¸(€€€‰•ÍÑ}‘•±Ñ„€ô•Ù…±}‘˜¹Í½ÉÑ}Ù…±Õ•Ì ‰‘•±Ñ…U}É¥‘•}½ÉÈˆ°…Í•¹‘¥¹œõ…±Í”¤¹¥±½lÁt¹Ñ½}‘¥Ð ¤(€€€‰•ÍÑ}µ• €ô•Ù…±}‘˜¹Í½ÉÑ}Ù…±Õ•Ì ‰µ•¡…¹¥Íµ}µ…É½}˜Äˆ°…Í•¹‘¥¹œõ…±Í”¤¹¥±½lÁt¹Ñ½}‘¥Ð ¤((€€€€Œ•½‘•Í¥Œµ‰¥…Ì½¹ÑÉ½°èÁÉ¥½É¥Ñ¥é”…±¥¹µ•¹Ð½ÁÉ½©•Ñ¥½¸½Ù•ÈÍÑÉ¥ÐÕÉÙ…ÑÕÉ”¸(€€€•Ù…±}‘™l‰‰¥…Í}½¹ÑÉ½±}Í½É”‰t€ô€ (€€€€€€€•Ù…±}‘™l‰É•…±}…±¥¹}Ñ}Í¡Õ™™±”‰t¹…ÍÑåÁ”¡™±½…Ð¤(€€€€€€€€¬•Ù…±}‘™l‰É•…±}ÁÉ½©}Ñ}Í¡Õ™™±”‰t¹…ÍÑåÁ”¡™±½…Ð¤(€€€€€€€€¬•Ù…±}‘™l‰É•…±}…±¥¹}Ñ}É•Ù•ÉÍ”‰t¹…ÍÑåÁ”¡™±½…Ð¤(€€€€€€€€¬¹À¹µ…á¥µÕ´¡•Ù…±}‘™l‰‘•±Ñ…U}É¥‘•}½ÉÈ‰t¹™¥±±¹„ À¤°€À¤(€€€€€€€€¬¹À¹µ…á¥µÕ´¡•Ù…±}‘™l‰µ•¡…¹¥Íµ}µ…É½}˜Ä‰t¹™¥±±¹„ À¤°€À¤(€€€€¤(€€€‰•ÍÑ}‰¥…Ì€ô•Ù…±}‘˜¹Í½ÉÑ}Ù…±Õ•Ì ‰‰¥…Í}½¹ÑÉ½±}Í½É”ˆ°…Í•¹‘¥¹œõ…±Í”¤¹¥±½lÁt¹Ñ½}‘¥Ð ¤((€€€ÍÕµµ…Éä€ôì(€€€€€€€€‰µ½‘•±}­•äˆèµ½‘•±}­•ä°(€€€€€€€€‰¹Õµ}±…å•ÉÌˆè0°(€€€€€€€€‰‘•¥Í¥½¹}±…å•ÉÌˆè‘•}±…å•ÉÌ°(€€€€€€€€‰‘•±Ñ…U}Á}Ù…É¥…¹”ˆèÁ„¹•áÁ±…¥¹•‘}Ù…É¥…¹•}É…Ñ¥½|¹Ñ½±¥ÍÐ ¤°(€€€€€€€€‰‘•±Ñ…U}ÁŒÅ}Ù…É¥…¹”ˆè™±½…Ð¡Á„¹•áÁ±…¥¹•‘}Ù…É¥…¹•}É…Ñ¥½}lÁt¤°(€€€€€€€€‰É¥Ñ¥…±}‰…¹‘}ÁÉ½áäˆèì‰±½Üˆè™±½…Ð¡‰…¹‘}±½Ü¤°€‰¡¥ ˆè™±½…Ð¡‰…¹‘}¡¥ ¥ô°(€€€€€€€€‰‰•ÍÑ}‘•±Ñ…U}Ý¥¹‘½Üˆè‰•ÍÑ}‘•±Ñ„°(€€€€€€€€‰‰•ÍÑ}µ•¡…¹¥Íµ}Ý¥¹‘½Üˆè‰•ÍÑ}µ• °(€€€€€€€€‰‰•ÍÑ}•½‘•Í¥}‰¥…Í}Ý¥¹‘½Üˆè‰•ÍÑ}‰¥…Ì°(€€€€€€€€‰Á…ÍÍ}…±¥¹µ•¹Ñ}Ñ}Í¡Õ™™±”ˆè‰½½°¡‰•ÍÑ}‰¥…Íl‰É•…±}…±¥¹}Ñ}Í¡Õ™™±”‰t¤°(€€€€€€€€‰Á…ÍÍ}ÁÉ½©•Ñ¥½¹}Ñ}Í¡Õ™™±”ˆè‰½½°¡‰•ÍÑ}‰¥…Íl‰É•…±}ÁÉ½©}Ñ}Í¡Õ™™±”‰t¤°(€€€€€€€€‰Á…ÍÍ}…±¥¹µ•¹Ñ}Ñ}É•Ù•ÉÍ”ˆè‰½½°¡‰•ÍÑ}‰¥…Íl‰É•…±}…±¥¹}Ñ}É•Ù•ÉÍ”‰t¤°(€€€€€€€€‰Á…ÍÍ}‘•±Ñ…U}½ÉÉ}Ñ|ÁÀÌˆè‰½½°¡‰•ÍÑ}‘•±Ñ…l‰‘•±Ñ…U}É¥‘•}½ÉÈ‰t€ø€À¸Ì¤°(€€€€€€€€‰Á…ÍÍ}µ•¡…¹¥Íµ}˜Å}Ñ|ÁÀØˆè‰½½°¡‰•ÍÑ}µ•¡l‰µ•¡…¹¥Íµ}µ…É½}˜Ä‰t€ø€À¸Ø¤°(€€€ô((€€€Ý¥Ñ ½Á•¸¡MY}%H€¼˜‰íµ½‘•±}­•åõ}ØÅ‰}ÍÕµµ…Éä¹©Í½¸ˆ°€‰Üˆ°•¹½‘¥¹œô‰ÕÑ˜´àˆ¤…Ì˜è(€€€€€€€©Í½¸¹‘ÕµÀ¡ÍÕµµ…Éä°˜°¥¹‘•¹ÐôÈ°•¹ÍÕÉ•}…Í¥¤õ…±Í”¤((€€€É•ÑÕÉ¸ÍÕµµ…Éä()‘•˜½µÁÕÑ•}É½ÍÍ}µ½‘•±}ÁÉ½™¥±•Ì¡ÍÕµµ…É¥•Ì¤è(€€€ÁÉ½™¥±•Ì€ôíô((€€€™½ÈÌ¥¸ÍÕµµ…É¥•Ìè(€€€€€€€µ¬€ôÍl‰µ½‘•±}­•ä‰t(€€€€€€€Á…Ñ €ôMY}%H€¼˜‰íµ­õ}±…å•É}ÁÉ½™¥±”¹ÍØˆ(€€€€€€€‘˜€ôÁ¹É•…‘}ÍØ¡Á…Ñ ¤(€€€€€€€ÍÕˆ€ô‘™l¡‘™l‰Ý¥¹‘½Ü‰t€ôô€‰™Õ±°ˆ¤€˜€¡‘™l‰¬‰t€ôô€ÔÀÀ¥t(€€€€€€€¥˜±•¸¡ÍÕˆ¤€ôô€Àè(€€€€€€€€€€€ÍÕˆ€ô‘™m‘™l‰Ý¥¹‘½Ü‰t€ôô€‰™Õ±°‰t((€€€€€€€ÁÉ½˜€ôÍÕˆ¹É½ÕÁ‰ä ‰±…å•É}™É…Œˆ¥ml(€€€€€€€€€€€€‰±½…±}…±¥¹}µ•…¸ˆ°(€€€€€€€€€€€€‰ÁÉ½©}Á½Í}É…Ñ¥½}µ•…¸ˆ°(€€€€€€€€€€€€‰É•Í¥‘Õ…±}É…Ñ¥½}µ•…¸ˆ°(€€€€€€€€€€€€‰ÍÑ•Á}¹½Éµ}µ•…¸ˆ°(€€€€€€€ut¹µ•…¸ ¤¹É•Í•Ñ}¥¹‘•à ¤¹Í½ÉÑ}Ù…±Õ•Ì ‰±…å•É}™É…Œˆ¤((€€€€€€€à€ôÁÉ½™l‰±…å•É}™É…Œ‰t¹Ù…±Õ•Ì(€€€€€€€á¤€ô¹À¹±¥¹ÍÁ…” À°€Ä°€ÔÀ¤(€€€€€€€ÁÉ½™¥±•Ímµ­t€ôíô(€€€€€€€™½È½°¥¸l‰±½…±}…±¥¹}µ•…¸ˆ°€‰ÁÉ½©}Á½Í}É…Ñ¥½}µ•…¸ˆ°€‰É•Í¥‘Õ…±}É…Ñ¥½}µ•…¸ˆ°€‰ÍÑ•Á}¹½Éµ}µ•…¸‰tè(€€€€€€€€€€€ÁÉ½™¥±•Ímµ­um½±t€ô¹À¹¥¹Ñ•ÉÀ¡á¤°à°ÁÉ½™m½±t¹Ù…±Õ•Ì¤((€€€É½ÝÌ€ômt(€€€™½È„°ˆ¥¸½µ‰¥¹…Ñ¥½¹Ì¡ÁÉ½™¥±•Ì¹­•åÌ ¤°€È¤è(€€€€€€€™½È½°¥¸l‰±½…±}…±¥¹}µ•…¸ˆ°€‰ÁÉ½©}Á½Í}É…Ñ¥½}µ•…¸ˆ°€‰É•Í¥‘Õ…±}É…Ñ¥½}µ•…¸ˆ°€‰ÍÑ•Á}¹½Éµ}µ•…¸‰tè(€€€€€€€€€€€É½ÝÌ¹…ÁÁ•¹¡ì(€€€€€€€€€€€€€€€€‰ÁÉ½™¥±•}ÑåÁ”ˆè½°°(€€€€€€€€€€€€€€€€‰µ½‘•±}„ˆè„°(€€€€€€€€€€€€€€€€‰µ½‘•±}ˆˆèˆ°(€€€€€€€€€€€€€€€€‰Á•…ÉÍ½¹}ÁÉ½™¥±•}½ÉÈˆè½ÉÉ}Í…™”¡ÁÉ½™¥±•Ím…um½±t°ÁÉ½™¥±•Ím‰um½±t¤°(€€€€€€€€€€€ô¤((€€€½ÕÐ€ôÁ¹…Ñ…É…µ”¡É½ÝÌ¤(€€€½ÕÐ¹Ñ½}ÍØ¡MY}%H€¼€‰ØÅ‰}É½ÍÍ}µ½‘•±}ÁÉ½™¥±•}½ÉÈ¹ÍØˆ°¥¹‘•àõ…±Í”°•¹½‘¥¹œô‰ÕÑ˜´àµÍ¥œˆ¤(€€€É•ÑÕÉ¸½ÕÐ((Œ€ôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôô(Œ5%8(Œ€ôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôô()‘•˜µ…¥¸ ¤è(€€€…ÉÌ€ôÁ…ÉÍ•}…ÉÌ ¤(€€€½¹™¥ÕÉ•}™É½µ}…ÉÌ¡…ÉÌ¤(€€€MY}%H¹µ­‘¥È¡Á…É•¹ÑÌõQÉÕ”°•á¥ÍÑ}½¬õQÉÕ”¤((€€€¥˜…ÉÌ¹¡•­}¥¹ÁÕÑÍ}½¹±äè(€€€€€€€ÝÉ¥Ñ•}¥¹ÁÕÑ}¡•­}É•Á½ÉÐ ¤(€€€€€€€É•ÑÕÉ¸((€€€•¹ÍÕÉ•}ÉÕ¹Ñ¥µ•}‘•Á•¹‘•¹¥•Ì ¤(€€€Í•Ñ}Í••¡M¤((€€€ÁÉ¥¹Ð ‰Õ‘¥Ñ¥¹œ½µµ½¸Í¥¹±”µÑ½­•¸±…‰•±Ì¸¸¸ˆ¤(€€€±…‰•±Ì€ô…Õ‘¥Ñ}½µµ½¹}Í¥¹±•}Ñ½­•¹}±…‰•±Ì¡5=1}MAL¤(€€€ÁÉ¥¹Ð ‰½µµ½¸±…‰•±Ìèˆ°±…‰•±Ì¤((€€€‘˜€ô‰Õ¥±‘}‘…Ñ…Í•Ð¡±…‰•±Ì¤(€€€‘˜¹Ñ½}ÍØ¡MY}%H€¼€‰ØÅ‰}ÁÉ½µÁÑ}‘…Ñ…Í•Ð¹ÍØˆ°¥¹‘•àõ…±Í”°•¹½‘¥¹œô‰ÕÑ˜´àµÍ¥œˆ¤(€€€ÁÉ¥¹Ð ‰…Ñ…Í•Ðèˆ°±•¸¡‘˜¤°€‰ÁÉ½µÁÑÌˆ¤((€€€ÍÕµµ…É¥•Ì€ômt(€€€™½Èµ¬°Á…Ñ ¥¸5=1}MAL¹¥Ñ•µÌ ¤è(€€€€€€€ÍÕµµ…É¥•Ì¹…ÁÁ•¹¡ÁÉ½•ÍÍ}µ½‘•°¡µ¬°Á…Ñ °‘˜¤¤((€€€™±…Ð€ômt(€€€™½ÈÌ¥¸ÍÕµµ…É¥•Ìè(€€€€€€€‰€ôÍl‰‰•ÍÑ}‘•±Ñ…U}Ý¥¹‘½Ü‰t(€€€€€€€‰´€ôÍl‰‰•ÍÑ}µ•¡…¹¥Íµ}Ý¥¹‘½Ü‰t(€€€€€€€‰ˆ€ôÍl‰‰•ÍÑ}•½‘•Í¥}‰¥…Í}Ý¥¹‘½Ü‰t(€€€€€€€™±…Ð¹…ÁÁ•¹¡ì(€€€€€€€€€€€€‰µ½‘•±}­•äˆèÍl‰µ½‘•±}­•ä‰t°(€€€€€€€€€€€€‰¹Õµ}±…å•ÉÌˆèÍl‰¹Õµ}±…å•ÉÌ‰t°(€€€€€€€€€€€€‰‘•¥Í¥½¹}±…å•ÉÌˆèÍÑÈ¡Íl‰‘•¥Í¥½¹}±…å•ÉÌ‰t¤°(€€€€€€€€€€€€‰‘•±Ñ…U}ÁŒÅ}Ù…É¥…¹”ˆèÍl‰‘•±Ñ…U}ÁŒÅ}Ù…É¥…¹”‰t°(€€€€€€€€€€€€‰‰•ÍÑ}‘•±Ñ…U}Ý¥¹‘½Üˆè‰‘l‰Ý¥¹‘½Ü‰t°(€€€€€€€€€€€€‰‰•ÍÑ}‘•±Ñ…U}¬ˆè‰‘l‰¬‰t°(€€€€€€€€€€€€‰‰•ÍÑ}‘•±Ñ…U}½ÉÈˆè‰‘l‰‘•±Ñ…U}É¥‘•}½ÉÈ‰t°(€€€€€€€€€€€€‰‰•ÍÑ}‘•±Ñ…U}ÈÈˆè‰‘l‰‘•±Ñ…U}É¥‘•}ÈÈ‰t°(€€€€€€€€€€€€‰‰•ÍÑ}µ•¡…¹¥Íµ}Ý¥¹‘½Üˆè‰µl‰Ý¥¹‘½Ü‰t°(€€€€€€€€€€€€‰‰•ÍÑ}µ•¡…¹¥Íµ}¬ˆè‰µl‰¬‰t°(€€€€€€€€€€€€‰‰•ÍÑ}µ•¡…¹¥Íµ}˜Äˆè‰µl‰µ•¡…¹¥Íµ}µ…É½}˜Ä‰t°(€€€€€€€€€€€€‰‰•ÍÑ}‰¥…Í}Ý¥¹‘½Üˆè‰‰l‰Ý¥¹‘½Ü‰t°(€€€€€€€€€€€€‰‰•ÍÑ}‰¥…Í}¬ˆè‰‰l‰¬‰t°(€€€€€€€€€€€€‰‰¥…Í}É•…±}…±¥¹}µ•…¸ˆè‰‰l‰É•…±}…±¥¹}µ•…¸‰t°(€€€€€€€€€€€€‰‰¥…Í}Í¡Õ™™±•}…±¥¹}µ•…¸ˆè‰‰l‰Í¡Õ™™±•}…±¥¹}µ•…¸‰t°(€€€€€€€€€€€€‰‰¥…Í}É•Ù•ÉÍ•}…±¥¹}µ•…¸ˆè‰‰l‰É•Ù•ÉÍ•}…±¥¹}µ•…¸‰t°(€€€€€€€€€€€€‰‰¥…Í}É•…±}ÁÉ½©}Á½Í}µ•…¸ˆè‰‰l‰É•…±}ÁÉ½©}Á½Í}µ•…¸‰t°(€€€€€€€€€€€€‰‰¥…Í}Í¡Õ™™±•}ÁÉ½©}Á½Í}µ•…¸ˆè‰‰l‰Í¡Õ™™±•}ÁÉ½©}Á½Í}µ•…¸‰t°(€€€€€€€€€€€€‰‰¥…Í}É•…±}‘•Ñ½ÕÉ}µ•…¸ˆè‰‰l‰É•…±}‘•Ñ½ÕÉ}µ•…¸‰t°(€€€€€€€€€€€€‰‰¥…Í}Í¡Õ™™±•}‘•Ñ½ÕÉ}µ•…¸ˆè‰‰l‰Í¡Õ™™±•}‘•Ñ½ÕÉ}µ•…¸‰t°(€€€€€€€€€€€€‰‰¥…Í}É•…±}ÕÉÙ…ÑÕÉ•}µ•…¸ˆè‰‰l‰É•…±}ÕÉÙ…ÑÕÉ•}µ•…¸‰t°(€€€€€€€€€€€€‰‰¥…Í}Í¡Õ™™±•}ÕÉÙ…ÑÕÉ•}µ•…¸ˆè‰‰l‰Í¡Õ™™±•}ÕÉÙ…ÑÕÉ•}µ•…¸‰t°(€€€€€€€€€€€€‰Á…ÍÍ}…±¥¹µ•¹Ñ}Ñ}Í¡Õ™™±”ˆèÍl‰Á…ÍÍ}…±¥¹µ•¹Ñ}Ñ}Í¡Õ™™±”‰t°(€€€€€€€€€€€€‰Á…ÍÍ}ÁÉ½©•Ñ¥½¹}Ñ}Í¡Õ™™±”ˆèÍl‰Á…ÍÍ}ÁÉ½©•Ñ¥½¹}Ñ}Í¡Õ™™±”‰t°(€€€€€€€€€€€€‰Á…ÍÍ}…±¥¹µ•¹Ñ}Ñ}É•Ù•ÉÍ”ˆèÍl‰Á…ÍÍ}…±¥¹µ•¹Ñ}Ñ}É•Ù•ÉÍ”‰t°(€€€€€€€€€€€€‰Á…ÍÍ}‘•±Ñ…U}½ÉÉ}Ñ|ÁÀÌˆèÍl‰Á…ÍÍ}‘•±Ñ…U}½ÉÉ}Ñ|ÁÀÌ‰t°(€€€€€€€€€€€€‰Á…ÍÍ}µ•¡…¹¥Íµ}˜Å}Ñ|ÁÀØˆèÍl‰Á…ÍÍ}µ•¡…¹¥Íµ}˜Å}Ñ|ÁÀØ‰t°(€€€€€€€ô¤((€€€µ½‘•±}ÍÕµµ…Éä€ôÁ¹…Ñ…É…µ”¡™±…Ð¤(€€€µ½‘•±}ÍÕµµ…Éä¹Ñ½}ÍØ¡MY}%H€¼€‰ØÅ‰}µ½‘•±}ÍÕµµ…Éä¹ÍØˆ°¥¹‘•àõ…±Í”°•¹½‘¥¹œô‰ÕÑ˜´àµÍ¥œˆ¤((€€€É½ÍÌ€ô½µÁÕÑ•}É½ÍÍ}µ½‘•±}ÁÉ½™¥±•Ì¡ÍÕµµ…É¥•Ì¤((€€€½Ù•É…±°€ôì(€€€€€€€€‰…Õ‘¥Ðˆè€‰X´Å•½‘•Í¥Œµ	¥…Í•QÉ…©•Ñ½ÉäÕ‘¥Ðˆ°(€€€€€€€€‰‘•™¥¹¥Ñ¥½¸ˆè€ (€€€€€€€€€€€€‰Q•ÍÑÌÝ¡•Ñ¡•ÈÉ•…°Q½Á,½Y%4ÑÉ…©•Ñ½É¥•Ì…É”‰¥…Í•Ñ½Ý…É„‘½µ¥¹…¹Ð•¹‘Á½¥¹Ð½•½‘•Í¥Œ‘¥É•Ñ¥½¸°€ˆ(€€€€€€€€€€€€‰Ý¥Ñ¡½ÕÐÉ•ÅÕ¥É¥¹œÍÑÉ¥Ð•ÅÕ…±¥ÑäÑ¼Í¡½ÉÑ•ÍÐÁ…Ñ¡Ì¸I•Í¥‘Õ…±Ì…É”ÑÉ•…Ñ•…Ìµ•…¹¥¹™Õ°±½…°½¹¹•Ñ¥½¸•™™•ÑÌ¸ˆ(€€€€€€€€¤°(€€€€€€€€‰µ½‘•±ÌˆèÍÕµµ…É¥•Ì°(€€€€€€€€‰É½ÍÍ}µ½‘•±}ÁÉ½™¥±•}½ÉÈˆèÉ½ÍÌ¹Ñ½}‘¥Ð¡½É¥•¹Ðô‰É•½É‘Ìˆ¤°(€€€ô((€€€Ý¥Ñ ½Á•¸¡MY}%H€¼€‰ØÅ‰}½Ù•É…±±}ÍÕµµ…Éä¹©Í½¸ˆ°€‰Üˆ°•¹½‘¥¹œô‰ÕÑ˜´àˆ¤…Ì˜è(€€€€€€€©Í½¸¹‘ÕµÀ¡½Ù•É…±°°˜°¥¹‘•¹ÐôÈ°•¹ÍÕÉ•}…Í¥¤õ…±Í”¤((€€€ÁÉ¥¹Ð ‰q¹X´Å½µÁ±•Ñ”¸ˆ¤(€€€ÁÉ¥¹Ð¡µ½‘•±}ÍÕµµ…Éä¤(€€€ÁÉ¥¹Ð¡É½ÍÌ¤()¥˜}}¹…µ•}|€ôô€‰}}µ…¥¹}|ˆè(€€€µ…¥¸ ¤(