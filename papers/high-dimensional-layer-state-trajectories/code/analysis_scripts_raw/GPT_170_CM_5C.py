# ============================================================
# CM-5C: Non-R O-flow Audit
#
# Goal:
#   Move beyond answer-margin O_flow:
#
#       O_adv = dR_{l+1} - dR_l
#
#   and test whether non-R neighborhood dynamics predict later
#   geodesic dominance:
#
#       O_topo_predecision -> DeltaU_decision
#
# Non-R O_topo features:
#   For each transition l -> l+1:
#       center_shift      = 1 - cos(C_{l+1}, C_l)
#       jaccard_turnover  = 1 - Jaccard(TopK_{l+1}, TopK_l)
#       wjaccard_turnover = 1 - weighted Jaccard(TopK logits)
#       spread_delta      = spread_{l+1} - spread_l
#       entropy_delta     = entropy_{l+1} - entropy_l
#
# Target:
#   DeltaU_decision = PC1(dR over decision window)
#
# Leakage control:
#   O_topo transitions are restricted to l+1 < decision_start.
#
# Run:
#   python cm5c_nonR_topo_oflow_audit.py
#
# Outputs:
#   cm5c_outputs/
#       cm5c_model_summary.csv
#       cm5c_cross_model_profile_corr.csv
#       cm5c_overall_summary.json
#       <model>_cm5c_summary.json
#       <model>_nonR_window_summary.csv
#       <model>_predecision_layer_corr.csv
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
SAVE_DIR = Path("cm5c_outputs")

SEED = 42
BATCH_SIZE = 4
MAX_LEN = 320
N_GRAPHS = 36

DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch is not None and DEVICE == "cuda" else (torch.float32 if torch is not None else None)

K_LIST = [100, 500]
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
    parser = argparse.ArgumentParser(description="CM-5C non-R TopK/VIM O-flow audit")
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
        "script": "GPT_170_CM_5C.py",
        "audit": "CM-5C Non-R TopK/VIM O-flow Audit",
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
            "cm4d_dir_optional": path_status(CM4D_DIR),
            "model_paths": {name: path_status(path) for name, path in MODEL_SPECS.items()},
        },
        "outputs": {"save_dir": path_status(SAVE_DIR)},
    }
    report_path = SAVE_DIR / "cm5c_input_check_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"CM-5C input check report written to: {report_path}")

def ensure_runtime_dependencies():
    missing = [name for name, ok in dependency_status().items() if not ok]
    if missing:
        raise RuntimeError(
            "Missing required runtime dependencies for full CM-5C run: "
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
        raise RuntimeError("transformers is required for full CM-5C runs.")
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

def build_daßŸ<¶‰žËkºwµçh€€€€€€€€€€€€€€€€€€€€€€€¥˜°¹½Ð¥¸Ñ½Á½m­ul‰•¹Ñ•ÉÌ‰tè(€€€€€€€€€€€€€€€€€€€€€€€€€€€‘¥´€ô•¹Ñ•È¹Í¡…Á•lÅt(€€€€€€€€€€€€€€€€€€€€€€€€€€€Ñ½Á½m­ul‰•¹Ñ•ÉÌ‰um±t€ô¹À¹é•É½Ì ¡¸°‘¥´¤°‘ÑåÁ”õ¹À¹™±½…ÐÌÈ¤(€€€€€€€€€€€€€€€€€€€€€€€€€€€Ñ½Á½m­ul‰¥‘Ì‰um±t€ô¹À¹é•É½Ì ¡¸°¬¤°‘ÑåÁ”õ¹À¹¥¹ÐÌÈ¤(€€€€€€€€€€€€€€€€€€€€€€€€€€€Ñ½Á½m­ul‰Ù…±Ì‰um±t€ô¹À¹é•É½Ì ¡¸°¬¤°‘ÑåÁ”õ¹À¹™±½…ÐÌÈ¤(€€€€€€€€€€€€€€€€€€€€€€€€€€€Ñ½Á½m­ul‰ÍÁÉ•…‰um±t€ô¹À¹é•É½Ì¡¸°‘ÑåÁ”õ¹À¹™±½…ÐÌÈ¤(€€€€€€€€€€€€€€€€€€€€€€€€€€€Ñ½Á½m­ul‰•¹ÑÉ½Áä‰um±t€ô¹À¹é•É½Ì¡¸°‘ÑåÁ”õ¹À¹™±½…ÐÌÈ¤((€€€€€€€€€€€€€€€€€€€€€€€Ñ½Á½m­ul‰•¹Ñ•ÉÌ‰um±umÍÑ…ÉÐé•¹‘t€ô•¹Ñ•È¹‘•Ñ…  ¤¹ÁÔ ¤¹¹ÕµÁä ¤¹…ÍÑåÁ”¡¹À¹™±½…ÐÌÈ¤(€€€€€€€€€€€€€€€€€€€€€€€Ñ½Á½m­ul‰¥‘Ì‰um±umÍÑ…ÉÐé•¹‘t€ô¥‘Í}¬¹‘•Ñ…  ¤¹ÁÔ ¤¹¹ÕµÁä ¤¹…ÍÑåÁ”¡¹À¹¥¹ÐÌÈ¤(€€€€€€€€€€€€€€€€€€€€€€€Ñ½Á½m­ul‰Ù…±Ì‰um±umÍÑ…ÉÐé•¹‘t€ôÙ…±Í}¬¹‘•Ñ…  ¤¹ÁÔ ¤¹¹ÕµÁä ¤¹…ÍÑåÁ”¡¹À¹™±½…ÐÌÈ¤(€€€€€€€€€€€€€€€€€€€€€€€Ñ½Á½m­ul‰ÍÁÉ•…‰um±umÍÑ…ÉÐé•¹‘t€ôÍÁÉ•…¹‘•Ñ…  ¤¹ÁÔ ¤¹¹ÕµÁä ¤¹…ÍÑåÁ”¡¹À¹™±½…ÐÌÈ¤(€€€€€€€€€€€€€€€€€€€€€€€Ñ½Á½m­ul‰•¹ÑÉ½Áä‰um±umÍÑ…ÉÐé•¹‘t€ô•¹ÑÉ½Áä¹‘•Ñ…  ¤¹ÁÔ ¤¹¹ÕµÁä ¤¹…ÍÑåÁ”¡¹À¹™±½…ÐÌÈ¤((€€€€€€€€€€€ÁÉ¥¹Ð¡˜ˆ€ÁÉ½•ÍÍ•í•¹‘ô½í¹ôˆ¤((€€€€€€€€€€€‘•°½ÕÑÁÕÑÌ°¡ÍÑ…Ñ•Ì°¥¹ÁÕÑÌ(€€€€€€€€€€€Œ¹½±±•Ð ¤(€€€€€€€€€€€¥˜Ñ½É ¥Ì¹½Ð9½¹”…¹Ñ½É ¹Õ‘„¹¥Í}…Ù…¥±…‰±” ¤è(€€€€€€€€€€€€€€€Ñ½É ¹Õ‘„¹•µÁÑå}…¡” ¤((€€€‘•°µ½‘•°°Ñ½¬°\°]}¹½É´(€€€Œ¹½±±•Ð ¤(€€€¥˜Ñ½É ¥Ì¹½Ð9½¹”…¹Ñ½É ¹Õ‘„¹¥Í}…Ù…¥±…‰±” ¤è(€€€€€€€Ñ½É ¹Õ‘„¹•µÁÑå}…¡” ¤((€€€É•ÑÕÉ¸H°Ñ½Á¼°0°‘•}±…å•ÉÌ°ÁÉ•}µ…à((Œ€ôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôô(ŒAI=ML5=0(Œ€ôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôô()‘•˜ÁÉ½•ÍÍ}µ½‘•°¡µ½‘•±}­•ä°µ½‘•±}Á…Ñ °‘˜¤è(€€€H°Ñ½Á¼°0°‘•}±…å•ÉÌ°ÁÉ•}µ…à€ô•áÑÉ…Ñ}…±°¡µ½‘•±}­•ä°µ½‘•±}Á…Ñ °‘˜¤(€€€‘•}ÍÑ…ÉÐ€ôµ¥¸¡‘•}±…å•ÉÌ¤(€€€´Ñ€ô±½…‘}´Ñ‘}Ý¥¹‘½ÝÌ¡µ½‘•±}­•ä¤((€€€µ½‘•±}‘˜€ô‘˜¹½Áä ¤(€€€™½È°¥¸É…¹”¡0¤è(€€€€€€€µ½‘•±}‘™m˜‰I}1í±ô‰t€ôIlè°±t((€€€€Œ±•…¸µÉ•±…Ñ¥Ù”‘H™½ÈÑ…É•Ð½¹±ä¸(€€€±•…¹}¥‘à€ôí¥¹Ð¡É½Ü¹É…Á¡}¥¤è¥‘à™½È¥‘à°É½Ü¥¸µ½‘•±}‘˜¹É•Í•Ñ}¥¹‘•à¡‘É½ÀõQÉÕ”¤¹¥Ñ•ÉÉ½ÝÌ ¤¥˜É½Ýl‰½¹‘¥Ñ¥½¸‰t€ôô€‰±•…¸‰ô(€€€‘H€ô¹À¹é•É½Í}±¥­”¡H¤(€€€™½È¤°É½Ü¥¸µ½‘•±}‘˜¹É•Í•Ñ}¥¹‘•à¡‘É½ÀõQÉÕ”¤¹¥Ñ•ÉÉ½ÝÌ ¤è(€€€€€€€¤€ô±•…¹}¥‘ám¥¹Ð¡É½Ýl‰É…Á¡}¥‰t¥t(€€€€€€€‘Im¥t€ôIm¥t€´Im¥t(€€€™½È°¥¸É…¹”¡0¤è(€€€€€€€µ½‘•±}‘™m˜‰‘I}1í±ô‰t€ô‘Ilè°±t((€€€a}‘•Œ€ô‘Ilè°‘•}±…å•ÉÍt(€€€Á„€ôA¡¹}½µÁ½¹•¹ÑÌõµ¥¸ Ì°a}‘•Œ¹Í¡…Á•lÅt¤¤(€€€ÁÌ€ôÁ„¹™¥Ñ}ÑÉ…¹Í™½É´¡a}‘•Œ¤(€€€‘•±Ñ…T€ôÁÍlè°€Át(€€€¥˜¹À¹µ•…¸¡‘•±Ñ…Umµ½‘•±}‘™l‰µ•¡…¹¥Í´‰t€ôô€‰ÍÑ…‰±”‰t¤€ð¹À¹µ•…¸¡‘•±Ñ…Umµ½‘•±}‘™l‰µ•¡…¹¥Í´‰t€ôô€‰±½ÍÕÉ”‰t¤è(€€€€€€€‘•±Ñ…T€ô€µ‘•±Ñ…T(€€€µ½‘•±}‘™l‰•±Ñ…U}‘•¥Í¥½¸‰t€ô‘•±Ñ…T¹…ÍÑåÁ”¡¹À¹™±½…ÐÌÈ¤((€€€äÌ€ôµ½‘•±}‘™l‰µ•¡…¹¥Í´‰t¹µ…À¡5!}5@Ì¤¹Ù…±Õ•Ì¹…ÍÑåÁ”¡¥¹Ð¤(€€€É½ÕÁÌ€ôµ½‘•±}‘™l‰É…Á¡}¥‰t¹Ù…±Õ•Ì¹…ÍÑåÁ”¡¥¹Ð¤((€€€€Œ]¥¹‘½Ü‘•™Ì°ÍÑÉ¥Ñ±äÁÉ•‘•¥Í¥½¸¸(€€€‘•˜ÑÉ…¹Í}É…¹”¡„°ˆ¤è(€€€€€€€„€ôµ…à À°µ¥¸¡ÁÉ•}µ…à°„¤¤(€€€€€€€ˆ€ôµ…à¡„°µ¥¸¡ÁÉ•}µ…à°ˆ¤¤(€€€€€€€¥˜ˆ€ð„è(€€€€€€€€€€€É•ÑÕÉ¸mt(€€€€€€€É•ÑÕÉ¸±¥ÍÐ¡É…¹”¡„°ˆ€¬€Ä¤¤((€€€•…É±å}•¹€ôµ…à Ä°¥¹Ð¡µ…Ñ ¹™±½½È ¡0€´€Ä¤€¨€À¸ÈÔ¤¤¤(€€€µ¥‘}•¹€ôµ…à¡•…É±å}•¹€¬€Ä°¥¹Ð¡µ…Ñ ¹™±½½È ¡0€´€Ä¤€¨€À¸ÔÔ¤¤¤((€€€Ñ½Á½±½ä€ô´Ñ¹•Ð ‰Ñ½Á½±½äˆ°±¥ÍÐ¡É…¹” À°µ¥¸ È°0¤¤¤¤(€€€µ•¡…¹¥Í´€ô´Ñ¹•Ð ‰µ•¡…¹¥Í´ˆ°±¥ÍÐ¡É…¹” À°µ¥¸ Ø°0¤¤¤¤(€€€‘½Ý¹ÍÑÉ•…´€ô´Ñ¹•Ð ‰‘½Ý¹ÍÑÉ•…´ˆ°±¥ÍÐ¡É…¹” À°µ¥¸ Ì°0¤¤¤¤(€€€½Ù•É…±°€ô´Ñ¹•Ð ‰½Ù•É…±°ˆ°‘½Ý¹ÍÑÉ•…´¤((€€€Ñ}•¹€ôµ…à¡Ñ½Á½±½ä¤¥˜Ñ½Á½±½ä•±Í”€À(€€€µ}•¹€ôµ…à¡µ•¡…¹¥Í´¤¥˜µ•¡…¹¥Í´•±Í”€À(€€€‘}•¹€ôµ…à¡‘½Ý¹ÍÑÉ•…´¤¥˜‘½Ý¹ÍÑÉ•…´•±Í”€À(€€€½}•¹€ôµ…à¡½Ù•É…±°¤¥˜½Ù•É…±°•±Í”€À((€€€Ý¥¹‘½Ý}‘•™Ì€ôì(€€€€€€€€‰Ñ½Á½}•…É±å|Á|ÈÕ‘•ÁÑ ˆèÑÉ…¹Í}É…¹” À°•…É±å}•¹¤°(€€€€€€€€‰Ñ½Á½}µ¥‘|ÈÕ|ÔÕ‘•ÁÑ ˆèÑÉ…¹Í}É…¹”¡•…É±å}•¹€¬€Ä°µ¥‘}•¹¤°(€€€€€€€€‰Ñ½Á½}ÁÉ•‘•¥Í¥½¹}…±°ˆèÑÉ…¹Í}É…¹” À°ÁÉ•}µ…à¤°(€€€€€€€€‰Ñ½Á½}…™Ñ•É}Ñ½Á½±½å}Õ¹Ñ¥±}ÁÉ•‘•¥Í¥½¸ˆèÑÉ…¹Í}É…¹”¡Ñ}•¹°ÁÉ•}µ…à¤°(€€€€€€€€‰Ñ½Á½}…™Ñ•É}µ•¡…¹¥Íµ}Õ¹Ñ¥±}ÁÉ•‘•¥Í¥½¸ˆèÑÉ…¹Í}É…¹”¡µ}•¹°ÁÉ•}µ…à¤°(€€€€€€€€‰Ñ½Á½}…™Ñ•É}‘½Ý¹ÍÑÉ•…µ}Õ¹Ñ¥±}ÁÉ•‘•¥Í¥½¸ˆèÑÉ…¹Í}É…¹”¡‘}•¹°ÁÉ•}µ…à¤°(€€€€€€€€‰Ñ½Á½}…™Ñ•É}½Ù•É…±±}Õ¹Ñ¥±}ÁÉ•‘•¥Í¥½¸ˆèÑÉ…¹Í}É…¹”¡½}•¹°ÁÉ•}µ…à¤°(€€€€€€€€‰Ñ½Á½}´Ñ‘}Ñ½Á½±½å}½¹±äˆèÑÉ…¹Í}É…¹”¡µ¥¸¡Ñ½Á½±½ä¤¥˜Ñ½Á½±½ä•±Í”€À°µ…à¡Ñ½Á½±½ä¤´Ä¥˜±•¸¡Ñ½Á½±½ä¤€ø€Ä•±Í”µ…à¡Ñ½Á½±½ä¤¥˜Ñ½Á½±½ä•±Í”€À¤°(€€€€€€€€‰Ñ½Á½}´Ñ‘}µ•¡…¹¥Íµ}½¹±äˆèÑÉ…¹Í}É…¹”¡µ¥¸¡µ•¡…¹¥Í´¤¥˜µ•¡…¹¥Í´•±Í”€À°µ…à¡µ•¡…¹¥Í´¤´Ä¥˜±•¸¡µ•¡…¹¥Í´¤€ø€Ä•±Í”µ…à¡µ•¡…¹¥Í´¤¥˜µ•¡…¹¥Í´•±Í”€À¤°(€€€€€€€€‰Ñ½Á½}´Ñ‘}‘½Ý¹ÍÑÉ•…µ}½¹±äˆèÑÉ…¹Í}É…¹”¡µ¥¸¡‘½Ý¹ÍÑÉ•…´¤¥˜‘½Ý¹ÍÑÉ•…´•±Í”€À°µ…à¡‘½Ý¹ÍÑÉ•…´¤´Ä¥˜±•¸¡‘½Ý¹ÍÑÉ•…´¤€ø€Ä•±Í”µ…à¡‘½Ý¹ÍÑÉ•…´¤¥˜‘½Ý¹ÍÑÉ•…´•±Í”€À¤°(€€€ô((€€€€Œ	Õ¥±Á•ÈµÑÉ…¹Í¥Ñ¥½¸=}Ñ½Á¼™½È•… ¬¸(€€€=}‰å}¬€ôíô(€€€±…å•É}½ÉÉ}É½ÝÌ€ômt((€€€™½È¬¥¸-}1%MPè(€€€€€€€€Œ™•…ÑÕÉ”Ñ•¹Í½È¸àÑÉ…¹Í¥Ñ¥½¹Ìà˜(€€€€€€€™•…ÑÕÉ•Ì€ômt(€€€€€€€¹…µ•Ì€ômt(€€€€€€€™½È°¥¸É…¹” À°ÁÉ•}µ…à€¬€Ä¤è(€€€€€€€€€€€À€ôÑ½Á½m­ul‰•¹Ñ•ÉÌ‰um±t(€€€€€€€€€€€Ä€ôÑ½Á½m­ul‰•¹Ñ•ÉÌ‰um°€¬€Åt(€€€€€€€€€€€•¹Ñ•É}Í¡¥™Ð€ô€Ä¸À€´É½Ý}½Í¥¹”¡À°Ä¤((€€€€€€€€€€€¥‘ÌÀ€ôÑ½Á½m­ul‰¥‘Ì‰um±t(€€€€€€€€€€€¥‘ÌÄ€ôÑ½Á½m­ul‰¥‘Ì‰um°€¬€Åt(€€€€€€€€€€€Ù…±ÌÀ€ôÑ½Á½m­ul‰Ù…±Ì‰um±t(€€€€€€€€€€€Ù…±ÌÄ€ôÑ½Á½m­ul‰Ù…±Ì‰um°€¬€Åt((€€€€€€€€€€€©…Œ€ô©……É‘}…ÉÉ…åÌ¡¥‘ÌÀ°¥‘ÌÄ¤(€€€€€€€€€€€Ý©…Œ€ôÝ•¥¡Ñ•‘}©……É‘}…ÉÉ…åÌ¡¥‘ÌÀ°Ù…±ÌÀ°¥‘ÌÄ°Ù…±ÌÄ¤((€€€€€€€€€€€ÍÁÉ•…‘}‘•±Ñ„€ôÑ½Á½m­ul‰ÍÁÉ•…‰um°€¬€Åt€´Ñ½Á½m­ul‰ÍÁÉ•…‰um±t(€€€€€€€€€€€•¹ÑÉ½Áå}‘•±Ñ„€ôÑ½Á½m­ul‰•¹ÑÉ½Áä‰um°€¬€Åt€´Ñ½Á½m­ul‰•¹ÑÉ½Áä‰um±t(€€€€€€€€€€€ÍÁÉ•…‘}…‰Í}‘•±Ñ„€ô¹À¹…‰Ì¡ÍÁÉ•…‘}‘•±Ñ„¤(€€€€€€€€€€€•¹ÑÉ½Áå}…‰Í}‘•±Ñ„€ô¹À¹…‰Ì¡•¹ÑÉ½Áå}‘•±Ñ„¤((€€€€€€€€€€€€ô¹À¹½±Õµ¹}ÍÑ…¬¡l(€€€€€€€€€€€€€€€•¹Ñ•É}Í¡¥™Ð°(€€€€€€€€€€€€€€€€Ä¸À€´©…Œ°(€€€€€€€€€€€€€€€€Ä¸À€´Ý©…Œ°(€€€€€€€€€€€€€€€ÍÁÉ•…‘}‘•±Ñ„°(€€€€€€€€€€€€€€€•¹ÑÉ½Áå}‘•±Ñ„°(€€€€€€€€€€€€€€€ÍÁÉ•…‘}…‰Í}‘•±Ñ„°(€€€€€€€€€€€€€€€•¹ÑÉ½Áå}…‰Í}‘•±Ñ„°(€€€€€€€€€€€t¤¹…ÍÑåÁ”¡¹À¹™±½…ÐÌÈ¤(€€€€€€€€€€€™•…ÑÕÉ•Ì¹…ÁÁ•¹¡¤(€€€€€€€€€€€¹…µ•Ì€ôl(€€€€€€€€€€€€€€€€‰•¹Ñ•É}Í¡¥™Ðˆ°(€€€€€€€€€€€€€€€€‰©……É‘}ÑÕÉ¹½Ù•Èˆ°(€€€€€€€€€€€€€€€€‰Ý•¥¡Ñ•‘}©……É‘}ÑÕÉ¹½Ù•Èˆ°(€€€€€€€€€€€€€€€€‰ÍÁÉ•…‘}‘•±Ñ„ˆ°(€€€€€€€€€€€€€€€€‰•¹ÑÉ½Áå}‘•±Ñ„ˆ°(€€€€€€€€€€€€€€€€‰ÍÁÉ•…‘}…‰Í}‘•±Ñ„ˆ°(€€€€€€€€€€€€€€€€‰•¹ÑÉ½Áå}…‰Í}‘•±Ñ„ˆ°(€€€€€€€€€€€t((€€€€€€€€€€€€Œ±…å•ÉÝ¥Í”ÍÕµµ…ÉäèÕÍ”…±°™•…ÑÕÉ•ÌÑ¼É¥‘”•±Ñ…T™½È•… Í¥¹±”ÑÉ…¹Í¥Ñ¥½¸¸(€€€€€€€€€€€ÈÉ}°°½ÉÉ}°€ôÉ¥‘•}Ø¡°‘•±Ñ…T°É½ÕÁÌ¤(€€€€€€€€€€€…}°°˜Å}°€ô±½¥ÍÑ¥}Ø¡°äÌ°É½ÕÁÌ¤(€€€€€€€€€€€±…å•É}½ÉÉ}É½ÝÌ¹…ÁÁ•¹¡ì(€€€€€€€€€€€€€€€€‰µ½‘•±}­•äˆèµ½‘•±}­•ä°(€€€€€€€€€€€€€€€€‰¬ˆè¬°(€€€€€€€€€€€€€€€€‰ÑÉ…¹Í¥Ñ¥½¹}±…å•Èˆè°°(€€€€€€€€€€€€€€€€‰ÑÉ…¹Í¥Ñ¥½¸ˆè˜‰1í±ô´ù1í°¬Åôˆ°(€€€€€€€€€€€€€€€€‰±…å•É}™É…Œˆè°€¼µ…à Ä°0€´€È¤°(€€€€€€€€€€€€€€€€‰‘•¥Í¥½¹}ÍÑ…ÉÐˆè‘•}ÍÑ…ÉÐ°(€€€€€€€€€€€€€€€€‰É¥‘•}‘•±Ñ…U}ÈÈˆèÈÉ}°°(€€€€€€€€€€€€€€€€‰É¥‘•}‘•±Ñ…U}½ÉÈˆè½ÉÉ}°°(€€€€€€€€€€€€€€€€‰µ•¡…¹¥Íµ}…Œˆè…}°°(€€€€€€€€€€€€€€€€‰µ•¡…¹¥Íµ}µ…É½}˜Äˆè˜Å}°°(€€€€€€€€€€€ô¤((€€€€€€€=}‰å}­m­t€ô¹À¹ÍÑ…¬¡™•…ÑÕÉ•Ì°…á¥ÌôÄ¤€€Œ¸àPà˜((€€€±…å•É}½ÉÉ}‘˜€ôÁ¹…Ñ…É…µ”¡±…å•É}½ÉÉ}É½ÝÌ¤(€€€±…å•É}½ÉÉ}‘˜¹Ñ½}ÍØ¡MY}%H€¼˜‰íµ½‘•±}­•åõ}ÁÉ•‘•¥Í¥½¹}±…å•É}½ÉÈ¹ÍØˆ°¥¹‘•àõ…±Í”°•¹½‘¥¹œô‰ÕÑ˜´àµÍ¥œˆ¤((€€€€Œ]¥¹‘½Ü•Ù…±Õ…Ñ¥½¹Ì¸(€€€É½ÝÌ€ômt(€€€‰•ÍÑ}…ÉÑ¥™…ÑÌ€ôíô((€€€™½È¬¥¸-}1%MPè(€€€€€€€<€ô=}‰å}­m­t(€€€€€€€™½ÈÝ¥¹}¹…µ”°±…å•ÉÌ¥¸Ý¥¹‘½Ý}‘•™Ì¹¥Ñ•µÌ ¤è(€€€€€€€€€€€±…å•ÉÌ€ôm°™½È°¥¸±…å•ÉÌ¥˜€À€ðô°€ðôÁÉ•}µ…át(€€€€€€€€€€€¥˜±•¸¡±…å•ÉÌ¤€ôô€Àè(€€€€€€€€€€€€€€€½¹Ñ¥¹Õ”((€€€€€€€€€€€a}Í•Ä€ô=lè°±…å•ÉÌ°€ét€€Œ¸àÐà˜((€€€€€€€€€€€€ŒÉ•…Ñ•Ì…¹™±…ÑÑ•¹•ÁÉ½™¥±”¸(€€€€€€€€€€€a}™±…Ð€ôa}Í•Ä¹É•Í¡…Á”¡a}Í•Ä¹Í¡…Á•lÁt°€´Ä¤(€€€€€€€€€€€a}µ•…¸€ôa}Í•Ä¹µ•…¸¡…á¥ÌôÄ¤(€€€€€€€€€€€a}ÍÕ´€ôa}Í•Ä¹ÍÕ´¡…á¥ÌôÄ¤(€€€€€€€€€€€a}…‰Í}ÍÕ´€ô¹À¹…‰Ì¡a}Í•Ä¤¹ÍÕ´¡…á¥ÌôÄ¤(€€€€€€€€€€€a}µ…à€ôa}Í•Ä¹µ…à¡…á¥ÌôÄ¤(€€€€€€€€€€€a}µ¥¸€ôa}Í•Ä¹µ¥¸¡…á¥ÌôÄ¤((€€€€€€€€€€€a}…Õœ€ô¹À¹½±Õµ¹}ÍÑ…¬¡ma}™±…Ð°a}µ•…¸°a}ÍÕ´°a}…‰Í}ÍÕ´°a}µ…à°a}µ¥¹t¤¹…ÍÑåÁ”¡¹À¹™±½…ÐÌÈ¤((€€€€€€€€€€€ÈÈ°½ÉÈ€ôÉ¥‘•}Ø¡a}…Õœ°‘•±Ñ…T°É½ÕÁÌ¤(€€€€€€€€€€€…Œ°˜Ä€ô±½¥ÍÑ¥}Ø¡a}…Õœ°äÌ°É½ÕÁÌ¤((€€€€€€€€€€€€ŒM¥µÁ±”Í…±…È‘¥…¹½ÍÑ¥Ìè•¹Ñ•ÈÍ¡¥™ÐÍÕ´°©……ÉÑÕÉ¹½Ù•ÈÍÕ´¸(€€€€€€€€€€€•¹Ñ•É}ÍÕ´€ôa}Í•Ålè°€è°€Át¹ÍÕ´¡…á¥ÌôÄ¤(€€€€€€€€€€€©…}ÍÕ´€ôa}Í•Ålè°€è°€Åt¹ÍÕ´¡…á¥ÌôÄ¤(€€€€€€€€€€€Ý©…}ÍÕ´€ôa}Í•Ålè°€è°€Ét¹ÍÕ´¡…á¥ÌôÄ¤(€€€€€€€€€€€•¹ÑÉ½Áå}…‰Í}ÍÕ´€ôa}Í•Ålè°€è°€Ùt¹ÍÕ´¡…á¥ÌôÄ¤((€€€€€€€€€€€É½Ü€ôì(€€€€€€€€€€€€€€€€‰µ½‘•±}­•äˆèµ½‘•±}­•ä°(€€€€€€€€€€€€€€€€‰¬ˆè¬°(€€€€€€€€€€€€€€€€‰Ý¥¹‘½ÜˆèÝ¥¹}¹…µ”°(€€€€€€€€€€€€€€€€‰ÑÉ…¹Í¥Ñ¥½¹}±…å•ÉÌˆèÍÑÈ¡±…å•ÉÌ¤°(€€€€€€€€€€€€€€€€‰¹}ÑÉ…¹Í¥Ñ¥½¹Ìˆè±•¸¡±…å•ÉÌ¤°(€€€€€€€€€€€€€€€€‰‘•¥Í¥½¹}ÍÑ…ÉÐˆè‘•}ÍÑ…ÉÐ°(€€€€€€€€€€€€€€€€‰‘•¥Í¥½¹}±…å•ÉÌˆèÍÑÈ¡‘•}±…å•ÉÌ¤°(€€€€€€€€€€€€€€€€‰±•…­…•}™É•”ˆèQÉÕ”°(€€€€€€€€€€€€€€€€‰É¥‘•}‘•±Ñ…U}ÈÈˆèÈÈ°(€€€€€€€€€€€€€€€€‰É¥‘•}‘•±Ñ…U}½ÉÈˆè½ÉÈ°(€€€€€€€€€€€€€€€€‰µ•¡…¹¥Íµ}…Œˆè…Œ°(€€€€€€€€€€€€€€€€‰µ•¡…¹¥Íµ}µ…É½}˜Äˆè˜Ä°(€€€€€€€€€€€€€€€€‰•¹Ñ•É}ÍÕµ}½ÉÉ}‘•±Ñ…Tˆè½ÉÉ}Í…™”¡•¹Ñ•É}ÍÕ´°‘•±Ñ…T¤°(€€€€€€€€€€€€€€€€‰©……É‘}ÍÕµ}½ÉÉ}‘•±Ñ…Tˆè½ÉÉ}Í…™”¡©…}ÍÕ´°‘•±Ñ…T¤°(€€€€€€€€€€€€€€€€‰Ý•¥¡Ñ•‘}©……É‘}ÍÕµ}½ÉÉ}‘•±Ñ…Tˆè½ÉÉ}Í…™”¡Ý©…}ÍÕ´°‘•±Ñ…T¤°(€€€€€€€€€€€€€€€€‰•¹ÑÉ½Áå}…‰Í}ÍÕµ}½ÉÉ}‘•±Ñ…Tˆè½ÉÉ}Í…™”¡•¹ÑÉ½Áå}…‰Í}ÍÕ´°‘•±Ñ…T¤°(€€€€€€€€€€€ô(€€€€€€€€€€€É½ÝÌ¹…ÁÁ•¹¡É½Ü¤((€€€€€€€€€€€‰•ÍÑ}…ÉÑ¥™…ÑÍl¡¬°Ý¥¹}¹…µ”¥t€ôì(€€€€€€€€€€€€€€€€‰a}…Õœˆèa}…Õœ°(€€€€€€€€€€€€€€€€‰±…å•ÉÌˆè±…å•ÉÌ°(€€€€€€€€€€€€€€€€‰É½ÜˆèÉ½Ü°(€€€€€€€€€€€ô((€€€Ý¥¹}‘˜€ôÁ¹…Ñ…É…µ”¡É½ÝÌ¤(€€€Ý¥¹}‘˜¹Ñ½}ÍØ¡MY}%H€¼˜‰íµ½‘•±}­•åõ}¹½¹I}Ý¥¹‘½Ý}ÍÕµµ…Éä¹ÍØˆ°¥¹‘•àõ…±Í”°•¹½‘¥¹œô‰ÕÑ˜´àµÍ¥œˆ¤((€€€‰•ÍÑ}‘•±Ñ„€ôÝ¥¹}‘˜¹Í½ÉÑ}Ù…±Õ•Ì ‰É¥‘•}‘•±Ñ…U}½ÉÈˆ°…Í•¹‘¥¹œõ…±Í”¤¹¥±½lÁt¹Ñ½}‘¥Ð ¤(€€€‰•ÍÑ}µ• €ôÝ¥¹}‘˜¹Í½ÉÑ}Ù…±Õ•Ì ‰µ•¡…¹¥Íµ}µ…É½}˜Äˆ°…Í•¹‘¥¹œõ…±Í”¤¹¥±½lÁt¹Ñ½}‘¥Ð ¤(€€€Á•…­}±…å•È€ô±…å•É}½ÉÉ}‘˜¹Í½ÉÑ}Ù…±Õ•Ì ‰É¥‘•}‘•±Ñ…U}½ÉÈˆ°…Í•¹‘¥¹œõ…±Í”¤¹¥±½lÁt¹Ñ½}‘¥Ð ¤((€€€€ŒM…Ù”ÁÉ½™¥±”™½ÈÉ½ÍÌµµ½‘•°è‰•ÍÐ‘•±Ñ„Ý¥¹‘½ÜÍ¥¹±”µÑÉ…¹Í¥Ñ¥½¸½ÉÈÕÉÙ”‰ä¬‰•ÍÐ¸(€€€‰•ÍÑ}¬€ô¥¹Ð¡‰•ÍÑ}‘•±Ñ…l‰¬‰t¤(€€€ÕÉÙ”€ô±…å•É}½ÉÉ}‘™m±…å•É}½ÉÉ}‘™l‰¬‰t€ôô‰•ÍÑ}­t¹Í½ÉÑ}Ù…±Õ•Ì ‰ÑÉ…¹Í¥Ñ¥½¹}±…å•Èˆ¤(€€€ÁÉ½™¥±•}½ÉÈ€ô¥¹Ñ•ÉÁ½±…Ñ•}ÁÉ½™¥±”¡ÕÉÙ•l‰É¥‘•}‘•±Ñ…U}½ÉÈ‰t¹Ù…±Õ•Ì°€ÔÀ¤(€€€ÁÉ½™¥±•}˜Ä€ô¥¹Ñ•ÉÁ½±…Ñ•}ÁÉ½™¥±”¡ÕÉÙ•l‰µ•¡…¹¥Íµ}µ…É½}˜Ä‰t¹™¥±±¹„ À¤¹Ù…±Õ•Ì°€ÔÀ¤(€€€¹À¹Í…Ù•é}½µÁÉ•ÍÍ• (€€€€€€€MY}%H€¼˜‰íµ½‘•±}­•åõ}¹½¹I}ÁÉ½™¥±•Ì¹¹Áèˆ°(€€€€€€€ÁÉ½™¥±•}½ÉÈõÁÉ½™¥±•}½ÉÈ°(€€€€€€€ÁÉ½™¥±•}˜ÄõÁÉ½™¥±•}˜Ä°(€€€€€€€É…Ý}½ÉÈõÕÉÙ•l‰É¥‘•}‘•±Ñ…U}½ÉÈ‰t¹Ù…±Õ•Ì°(€€€€€€€É…Ý}˜ÄõÕÉÙ•l‰µ•¡…¹¥Íµ}µ…É½}˜Ä‰t¹™¥±±¹„ À¤¹Ù…±Õ•Ì°(€€€€€€€±…å•É}™É…ŒõÕÉÙ•l‰±…å•É}™É…Œ‰t¹Ù…±Õ•Ì°(€€€€¤((€€€µ½‘•±}‘˜¹Ñ½}ÍØ¡MY}%H€¼˜‰íµ½‘•±}­•åõ}I}‘…Ñ…Í•Ð¹ÍØˆ°¥¹‘•àõ…±Í”°•¹½‘¥¹œô‰ÕÑ˜´àµÍ¥œˆ¤((€€€ÍÕµµ…Éä€ôì(€€€€€€€€‰µ½‘•±}­•äˆèµ½‘•±}­•ä°(€€€€€€€€‰¹Õµ}±…å•ÉÌˆè0°(€€€€€€€€‰‘•¥Í¥½¹}±…å•ÉÌˆè‘•}±…å•ÉÌ°(€€€€€€€€‰‘•¥Í¥½¹}ÍÑ…ÉÐˆè‘•}ÍÑ…ÉÐ°(€€€€€€€€‰ÁÉ•‘•¥Í¥½¹}µ…á}ÑÉ…¹Í¥Ñ¥½¸ˆèÁÉ•}µ…à°(€€€€€€€€‰´Ñ‘}Ý¥¹‘½ÝÌˆè´Ñ°(€€€€€€€€‰‘•±Ñ…U}Á}Ù…É¥…¹”ˆèÁ„¹•áÁ±…¥¹•‘}Ù…É¥…¹•}É…Ñ¥½|¹Ñ½±¥ÍÐ ¤°(€€€€€€€€‰‘•±Ñ…U}ÁŒÅ}Ù…É¥…¹”ˆè™±½…Ð¡Á„¹•áÁ±…¥¹•‘}Ù…É¥…¹•}É…Ñ¥½}lÁt¤°(€€€€€€€€‰‰•ÍÑ}¹½¹I}‘•±Ñ…U}Ý¥¹‘½Üˆè‰•ÍÑ}‘•±Ñ„°(€€€€€€€€‰‰•ÍÑ}¹½¹I}µ•¡…¹¥Íµ}Ý¥¹‘½Üˆè‰•ÍÑ}µ• °(€€€€€€€€‰Á•…­}¹½¹I}±…å•É}½ÉÈˆèÁ•…­}±…å•È°(€€€€€€€€‰Á…ÍÍ}¹½¹I}‘•±Ñ…U}½ÉÉ}Ñ|ÁÀÌˆè‰½½°¡‰•ÍÑ}‘•±Ñ…l‰É¥‘•}‘•±Ñ…U}½ÉÈ‰t€ø€À¸Ì¤°(€€€€€€€€‰Á…ÍÍ}¹½¹I}µ•¡…¹¥Íµ}˜Å}Ñ|ÁÀØˆè‰½½°¡‰•ÍÑ}µ•¡l‰µ•¡…¹¥Íµ}µ…É½}˜Ä‰t€ø€À¸Ø¤°(€€€ô((€€€Ý¥Ñ ½Á•¸¡MY}%H€¼˜‰íµ½‘•±}­•åõ}´Õ}ÍÕµµ…Éä¹©Í½¸ˆ°€‰Üˆ°•¹½‘¥¹œô‰ÕÑ˜´àˆ¤…Ì˜è(€€€€€€€©Í½¸¹‘ÕµÀ¡ÍÕµµ…Éä°˜°¥¹‘•¹ÐôÈ°•¹ÍÕÉ•}…Í¥¤õ…±Í”¤((€€€É•ÑÕÉ¸ÍÕµµ…Éä()‘•˜½µÁÕÑ•}É½ÍÍ}µ½‘•±}ÁÉ½™¥±•Ì ¤è(€€€ÁÉ½™¥±•Ì€ôíô(€€€™½Èµ¬¥¸5=1}MALè(€€€€€€€‘…Ñ„€ô¹À¹±½…¡MY}%H€¼˜‰íµ­õ}¹½¹I}ÁÉ½™¥±•Ì¹¹Áèˆ°…±±½Ý}Á¥­±”õQÉÕ”¤(€€€€€€€ÁÉ½™¥±•Ímµ­t€ôì(€€€€€€€€€€€€‰½ÉÈˆè‘…Ñ…l‰ÁÉ½™¥±•}½ÉÈ‰t°(€€€€€€€€€€€€‰˜Äˆè‘…Ñ…l‰ÁÉ½™¥±•}˜Ä‰t°(€€€€€€€ô((€€€É½ÝÌ€ômt(€€€™½È„°ˆ¥¸½µ‰¥¹…Ñ¥½¹Ì¡5=1}MAL¹­•åÌ ¤°€È¤è(€€€€€€€™½ÈÑåÀ¥¸l‰½ÉÈˆ°€‰˜Ä‰tè(€€€€€€€€€€€É½ÝÌ¹…ÁÁ•¹¡ì(€€€€€€€€€€€€€€€€‰ÁÉ½™¥±•}ÑåÁ”ˆèÑåÀ°(€€€€€€€€€€€€€€€€‰µ½‘•±}„ˆè„°(€€€€€€€€€€€€€€€€‰µ½‘•±}ˆˆèˆ°(€€€€€€€€€€€€€€€€‰Á•…ÉÍ½¹}ÁÉ½™¥±•}½ÉÈˆè½ÉÉ}Í…™”¡ÁÉ½™¥±•Ím…umÑåÁt°ÁÉ½™¥±•Ím‰umÑåÁt¤°(€€€€€€€€€€€ô¤(€€€½ÕÐ€ôÁ¹…Ñ…É…µ”¡É½ÝÌ¤(€€€½ÕÐ¹Ñ½}ÍØ¡MY}%H€¼€‰´Õ}É½ÍÍ}µ½‘•±}ÁÉ½™¥±•}½ÉÈ¹ÍØˆ°¥¹‘•àõ…±Í”°•¹½‘¥¹œô‰ÕÑ˜´àµÍ¥œˆ¤(€€€É•ÑÕÉ¸½ÕÐ((Œ€ôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôô(Œ5%8(Œ€ôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôô()‘•˜µ…¥¸ ¤è(€€€…ÉÌ€ôÁ…ÉÍ•}…ÉÌ ¤(€€€½¹™¥ÕÉ•}™É½µ}…ÉÌ¡…ÉÌ¤(€€€MY}%H¹µ­‘¥È¡Á…É•¹ÑÌõQÉÕ”°•á¥ÍÑ}½¬õQÉÕ”¤((€€€¥˜…ÉÌ¹¡•­}¥¹ÁÕÑÍ}½¹±äè(€€€€€€€ÝÉ¥Ñ•}¥¹ÁÕÑ}¡•­}É•Á½ÉÐ ¤(€€€€€€€É•ÑÕÉ¸((€€€•¹ÍÕÉ•}ÉÕ¹Ñ¥µ•}‘•Á•¹‘•¹¥•Ì ¤(€€€Í•Ñ}Í••¡M¤((€€€ÁÉ¥¹Ð ‰Õ‘¥Ñ¥¹œ½µµ½¸Í¥¹±”µÑ½­•¸±…‰•±Ì¸¸¸ˆ¤(€€€½µµ½¹}±…‰•±Ì€ô…Õ‘¥Ñ}½µµ½¹}Í¥¹±•}Ñ½­•¹}±…‰•±Ì¡5=1}MAL¤(€€€ÁÉ¥¹Ð ‰½µµ½¸±…‰•±Ìèˆ°½µµ½¹}±…‰•±Ì¤((€€€‘˜€ô‰Õ¥±‘}‘…Ñ…Í•Ð¡½µµ½¹}±…‰•±Ì¤(€€€‘˜¹Ñ½}ÍØ¡MY}%H€¼€‰´Õ}ÁÉ½µÁÑ}‘…Ñ…Í•Ð¹ÍØˆ°¥¹‘•àõ…±Í”°•¹½‘¥¹œô‰ÕÑ˜´àµÍ¥œˆ¤(€€€ÁÉ¥¹Ð ‰…Ñ…Í•Ðèˆ°±•¸¡‘˜¤°€‰ÁÉ½µÁÑÌˆ¤((€€€ÍÕµµ…É¥•Ì€ômt(€€€™½Èµ¬°Á…Ñ ¥¸5=1}MAL¹¥Ñ•µÌ ¤è(€€€€€€€ÍÕµµ…É¥•Ì¹…ÁÁ•¹¡ÁÉ½•ÍÍ}µ½‘•°¡µ¬°Á…Ñ °‘˜¤¤((€€€™±…Ð€ômt(€€€™½ÈÌ¥¸ÍÕµµ…É¥•Ìè(€€€€€€€‰€ôÍl‰‰•ÍÑ}¹½¹I}‘•±Ñ…U}Ý¥¹‘½Ü‰t(€€€€€€€‰´€ôÍl‰‰•ÍÑ}¹½¹I}µ•¡…¹¥Íµ}Ý¥¹‘½Ü‰t(€€€€€€€Á•…¬€ôÍl‰Á•…­}¹½¹I}±…å•É}½ÉÈ‰t(€€€€€€€™±…Ð¹…ÁÁ•¹¡ì(€€€€€€€€€€€€‰µ½‘•±}­•äˆèÍl‰µ½‘•±}­•ä‰t°(€€€€€€€€€€€€‰¹Õµ}±…å•ÉÌˆèÍl‰¹Õµ}±…å•ÉÌ‰t°(€€€€€€€€€€€€‰‘•¥Í¥½¹}±…å•ÉÌˆèÍÑÈ¡Íl‰‘•¥Í¥½¹}±…å•ÉÌ‰t¤°(€€€€€€€€€€€€‰‘•¥Í¥½¹}ÍÑ…ÉÐˆèÍl‰‘•¥Í¥½¹}ÍÑ…ÉÐ‰t°(€€€€€€€€€€€€‰ÁÉ•‘•¥Í¥½¹}µ…á}ÑÉ…¹Í¥Ñ¥½¸ˆèÍl‰ÁÉ•‘•¥Í¥½¹}µ…á}ÑÉ…¹Í¥Ñ¥½¸‰t°(€€€€€€€€€€€€‰‘•±Ñ…U}ÁŒÅ}Ù…É¥…¹”ˆèÍl‰‘•±Ñ…U}ÁŒÅ}Ù…É¥…¹”‰t°(€€€€€€€€€€€€‰‰•ÍÑ}¹½¹I}‘•±Ñ…U}Ý¥¹‘½Üˆè‰‘l‰Ý¥¹‘½Ü‰t°(€€€€€€€€€€€€‰‰•ÍÑ}¹½¹I}‘•±Ñ…U}¬ˆè‰‘l‰¬‰t°(€€€€€€€€€€€€‰‰•ÍÑ}¹½¹I}‘•±Ñ…U}±…å•ÉÌˆè‰‘l‰ÑÉ…¹Í¥Ñ¥½¹}±…å•ÉÌ‰t°(€€€€€€€€€€€€‰‰•ÍÑ}¹½¹I}‘•±Ñ…U}½ÉÈˆè‰‘l‰É¥‘•}‘•±Ñ…U}½ÉÈ‰t°(€€€€€€€€€€€€‰‰•ÍÑ}¹½¹I}‘•±Ñ…U}ÈÈˆè‰‘l‰É¥‘•}‘•±Ñ…U}ÈÈ‰t°(€€€€€€€€€€€€‰‰•ÍÑ}¹½¹I}µ•¡…¹¥Íµ}Ý¥¹‘½Üˆè‰µl‰Ý¥¹‘½Ü‰t°(€€€€€€€€€€€€‰‰•ÍÑ}¹½¹I}µ•¡…¹¥Íµ}¬ˆè‰µl‰¬‰t°(€€€€€€€€€€€€‰‰•ÍÑ}¹½¹I}µ•¡…¹¥Íµ}±…å•ÉÌˆè‰µl‰ÑÉ…¹Í¥Ñ¥½¹}±…å•ÉÌ‰t°(€€€€€€€€€€€€‰‰•ÍÑ}¹½¹I}µ•¡…¹¥Íµ}˜Äˆè‰µl‰µ•¡…¹¥Íµ}µ…É½}˜Ä‰t°(€€€€€€€€€€€€‰‰•ÍÑ}¹½¹I}µ•¡…¹¥Íµ}…Œˆè‰µl‰µ•¡…¹¥Íµ}…Œ‰t°(€€€€€€€€€€€€‰Á•…­}¹½¹I}ÑÉ…¹Í¥Ñ¥½¸ˆèÁ•…¬¹•Ð ‰ÑÉ…¹Í¥Ñ¥½¸ˆ°9½¹”¤°(€€€€€€€€€€€€‰Á•…­}¹½¹I}±…å•É}™É…ŒˆèÁ•…¬¹•Ð ‰±…å•É}™É…Œˆ°9½¹”¤°(€€€€€€€€€€€€‰Á•…­}¹½¹I}‘•±Ñ…U}½ÉÈˆèÁ•…¬¹•Ð ‰É¥‘•}‘•±Ñ…U}½ÉÈˆ°9½¹”¤°(€€€€€€€€€€€€‰Á…ÍÍ}¹½¹I}‘•±Ñ…U}½ÉÉ}Ñ|ÁÀÌˆèÍl‰Á…ÍÍ}¹½¹I}‘•±Ñ…U}½ÉÉ}Ñ|ÁÀÌ‰t°(€€€€€€€€€€€€‰Á…ÍÍ}¹½¹I}µ•¡…¹¥Íµ}˜Å}Ñ|ÁÀØˆèÍl‰Á…ÍÍ}¹½¹I}µ•¡…¹¥Íµ}˜Å}Ñ|ÁÀØ‰t°(€€€€€€€ô¤(€€€µ½‘•±}ÍÕµµ…Éä€ôÁ¹…Ñ…É…µ”¡™±…Ð¤(€€€µ½‘•±}ÍÕµµ…Éä¹Ñ½}ÍØ¡MY}%H€¼€‰´Õ}µ½‘•±}ÍÕµµ…Éä¹ÍØˆ°¥¹‘•àõ…±Í”°•¹½‘¥¹œô‰ÕÑ˜´àµÍ¥œˆ¤((€€€É½ÍÌ€ô½µÁÕÑ•}É½ÍÍ}µ½‘•±}ÁÉ½™¥±•Ì ¤((€€€½Ù•É…±°€ôì(€€€€€€€€‰…Õ‘¥Ðˆè€‰4´Õ9½¸µHQ½Á,½Y%4<µ™±½ÜÕ‘¥Ðˆ°(€€€€€€€€‰‘•™¥¹¥Ñ¥½¸ˆè€ (€€€€€€€€€€€€‰UÍ•ÌÁÉ•‘•¥Í¥½¸Q½Á,½Y%4¹•¥¡‰½É¡½½ÑÉ…¹Í¥Ñ¥½¸™•…ÑÕÉ•Ì°¹½Ð‘H‘¥™™•É•¹•Ì°€ˆ(€€€€€€€€€€€€‰Ñ¼ÁÉ•‘¥Ð±…Ñ•È‘•¥Í¥½¸µÝ¥¹‘½Ü•±Ñ…T¸ˆ(€€€€€€€€¤°(€€€€€€€€‰µ½‘•±ÌˆèÍÕµµ…É¥•Ì°(€€€€€€€€‰É½ÍÍ}µ½‘•±}ÁÉ½™¥±•}½ÉÈˆèÉ½ÍÌ¹Ñ½}‘¥Ð¡½É¥•¹Ðô‰É•½É‘Ìˆ¤°(€€€ô((€€€Ý¥Ñ ½Á•¸¡MY}%H€¼€‰´Õ}½Ù•É…±±}ÍÕµµ…Éä¹©Í½¸ˆ°€‰Üˆ°•¹½‘¥¹œô‰ÕÑ˜´àˆ¤…Ì˜è(€€€€€€€©Í½¸¹‘ÕµÀ¡½Ù•É…±°°˜°¥¹‘•¹ÐôÈ°•¹ÍÕÉ•}…Í¥¤õ…±Í”¤((€€€ÁÉ¥¹Ð ‰q¹4´Õ½µÁ±•Ñ”¸ˆ¤(€€€ÁÉ¥¹Ð¡µ½‘•±}ÍÕµµ…Éä¤(€€€ÁÉ¥¹Ð¡É½ÍÌ¤()¥˜}}¹…µ•}|€ôô€‰}}µ…¥¹}|ˆè(€€€µ…¥¸ ¤(