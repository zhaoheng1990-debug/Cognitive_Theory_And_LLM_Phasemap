# -*- coding: utf-8 -*-
"""
GPT_201_SEM_DSTA2_policyid_lowrank_compression_across_layers.py

SEM-DSTA-2: PolicyID Low-Rank Compression Across Layers
-------------------------------------------------------

Goal:
    Test whether PolicyID signatures are high-dimensional in shallow initialization
    and progressively compress into a lower-rank policy structure across layers.

Motivation from SEM-DSTA-1:
    - PolicyID -> Sigma_init is strongly recoverable across held-out Concepts.
    - But shallow Sigma_init is not rank~6; rank90 was around 21-22 for full center/VIM features.
    - SEM-2A.3 full trajectory showed rank90 ~= 6.
    Therefore:
        PolicyID may be high-dimensional at initialization and become low-rank after
        transport/competition.

Core hypothesis:
    PolicyID
      -> Sigma_init^{high-dim}
      -> Sigma_l=(D_l,W_l)
      -> PolicyID^{low-rank}
      -> DeltaU / trajectory advantage

This script:
    1. Builds or loads the 24 concepts x 10 PolicyIDs x 8 surface families dataset.
    2. Extracts TopK/VIM features across L0-L25.
    3. Runs PolicyID recovery per layer window.
    4. Runs rank audit per layer window and feature block.
    5. Evaluates Concept-vs-PolicyID separation.
    6. Tests whether rank90 decreases from init to mid/decision windows.

Default:
    Qwen only, hard-coded local path.

Outputs:
    sem_dsta2_outputs/
      sem_dsta2_config.json
      sem_dsta2_dataset.csv
      sem_dsta2_features_raw.csv
      sem_dsta2_window_recovery.csv
      sem_dsta2_rank_by_window.csv
      sem_dsta2_compression_summary.csv
      sem_dsta2_concept_vs_policyid.csv
      sem_dsta2_surface_baseline.csv
      sem_dsta2_feature_importance.csv
      sem_dsta2_results_summary.json

Run:
    python GPT_201_SEM_DSTA2_policyid_lowrank_compression_across_layers.py

Notes:
    - This script is heavier than SEM-DSTA-1 because it extracts L0-L25.
    - Set MAX_ROWS to e.g. 240 for a smoke test.
"""

import json
import random
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# ============================================================
# CONFIG
# ============================================================

@dataclass
class CFG:
    MODEL_PATH: str = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

    # Prefer user's local prompt file if present.
    INPUT_CSV: str = r"C:\Users\ZH\Desktop\AGI\python_script\sem_dsta_prompts.csv"
    FALLBACK_INPUTS: Tuple[str, ...] = (
        r"./sem_dsta1_outputs/sem_dsta1_dataset.csv",
        r"./sem2a3_outputs/sem2a3_dataset.csv",
        r"./PROMPT_sem2a3.csv",
    )

    OUTPUT_DIR: str = "sem_dsta2_outputs"

    TOPK_LIST: Tuple[int, ...] = (100, 500)
    BATCH_SIZE: int = 4
    MAX_LENGTH: int = 192
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE: str = "auto"

    # Qwen layers: hidden_states[0] = embedding; hidden_states[L+1] = block L.
    LAYERS: Tuple[int, ...] = tuple(range(0, 26))

    # Window definitions.
    WINDOWS: Dict[str, Tuple[int, ...]] = None

    CENTER_PCA_DIM: int = 32
    N_SPLITS: int = 5
    RANDOM_SEED: int = 42

    MAX_ROWS: int = 0  # 0 = all. Use 240 for smoke test.

    REUSE_FEATURES_IF_EXISTS: bool = True
    USE_RANDOM_FOREST: bool = True
    SAVE_FEATURE_IMPORTANCE: bool = True

    # Label column preference.
    POLICY_COL_CANDIDATES: Tuple[str, ...] = ("policy_id", "constraint")
    CONCEPT_COL: str = "concept"
    STYLE_COL: str = "style_id"
    PROMPT_COL: str = "prompt"


def make_cfg() -> CFG:
    cfg = CFG()
    cfg.WINDOWS = {
        "init_L0_6": tuple(range(0, 7)),
        "qwen_mech_L5_6": (5, 6),
        "early_L0_11": tuple(range(0, 12)),
        "mid_L7_19": tuple(range(7, 20)),
        "predecision_L13_19": tuple(range(13, 20)),
        "decision_L20_25": tuple(range(20, 26)),
        "full_L0_25": tuple(range(0, 26)),
    }
    return cfg


cfg = make_cfg()


# ============================================================
# DATASET
# ============================================================

CONCEPTS = [
    "Paris", "Tokyo", "Newton", "Einstein", "Python", "Transformer",
    "Apple", "Hospital", "Democracy", "Evolution", "Gravity", "Market",
    "Language", "Memory", "Cancer", "Court", "School", "Ocean",
    "Desert", "Music", "Painting", "Energy", "Robot", "Internet",
]

POLICY_IDS = [
    "definition",
    "relation",
    "property",
    "list",
    "cause",
    "compare",
    "mechanism",
    "risk",
    "plan",
    "counterfactual",
]

TEMPLATES = {
    "definition": [
        "What is {concept}?",
        "Give a concise definition of {concept}.",
        "For {concept}, produce the answer using the 'what it is' operation.",
        "Apply the identity-description operation to {concept}.",
        "Handle {concept} by saying what kind of thing it is.",
        "For {concept}, return a short identity statement.",
        "Use operation code A: describe the basic identity of {concept}.",
        "In the shared analysis frame, resolve {concept} by identifying it.",
    ],
    "relation": [
        "What is {concept} related to?",
        "Identify the broader category or domain associated with {concept}.",
        "For {concept}, produce the answer using the 'connection' operation.",
        "Apply the relation-mapping operation to {concept}.",
        "Handle {concept} by placing it in a broader context.",
        "For {concept}, return its main association.",
        "Use operation code B: connect {concept} to its domain.",
        "In the shared analysis frame, resolve {concept} by mapping its relation.",
    ],
    "property": [
        "What are notable features of {concept}?",
        "Describe key characteristics of {concept}.",
        "For {concept}, produce the answer using the 'attribute' operation.",
        "Apply the feature-extraction operation to {concept}.",
        "Handle {concept} by giving traits rather than examples.",
        "For {concept}, return characteristic attributes.",
        "Use operation code C: extract properties of {concept}.",
        "In the shared analysis frame, resolve {concept} by listing attributes.",
    ],
    "list": [
        "List examples related to {concept}.",
        "Name several well-known items associated with {concept}.",
        "For {concept}, produce the answer using the 'enumeration' operation.",
        "Apply the example-generation operation to {concept}.",
        "Handle {concept} by giving multiple related items.",
        "For {concept}, return a compact set of examples.",
        "Use operation code D: enumerate related instances for {concept}.",
        "In the shared analysis frame, resolve {concept} by expanding examples.",
    ],
    "cause": [
        "What causes or explains {concept}?",
        "Explain why {concept} occurs or matters.",
        "For {concept}, produce the answer using the 'explanation' operation.",
        "Apply the causal-account operation to {concept}.",
        "Handle {concept} by giving reasons or drivers.",
        "For {concept}, return a short causal account.",
        "Use operation code E: explain drivers of {concept}.",
        "In the shared analysis frame, resolve {concept} by tracing causes.",
    ],
    "compare": [
        "Compare {concept} with a similar concept.",
        "Contrast {concept} with another relevant example.",
        "For {concept}, produce the answer using the 'contrast' operation.",
        "Apply the comparison operation to {concept}.",
        "Handle {concept} by separating it from a related case.",
        "For {concept}, return similarities and differences.",
        "Use operation code F: contrast {concept} with a neighbor.",
        "In the shared analysis frame, resolve {concept} by comparing it.",
    ],
    "mechanism": [
        "How does {concept} work?",
        "Describe the internal process behind {concept}.",
        "For {concept}, produce the answer using the 'process' operation.",
        "Apply the mechanism-tracing operation to {concept}.",
        "Handle {concept} by describing components and steps.",
        "For {concept}, return the working process.",
        "Use operation code G: trace how {concept} functions.",
        "In the shared analysis frame, resolve {concept} by explaining mechanism.",
    ],
    "risk": [
        "What risks are associated with {concept}?",
        "Identify vulnerabilities or dangers related to {concept}.",
        "For {concept}, produce the answer using the 'failure' operation.",
        "Apply the risk-audit operation to {concept}.",
        "Handle {concept} by looking for failure modes.",
        "For {concept}, return likely problems or hazards.",
        "Use operation code H: audit risks of {concept}.",
        "In the shared analysis frame, resolve {concept} by checking failure modes.",
    ],
    "plan": [
        "Make a plan involving {concept}.",
        "Outline steps to use or study {concept}.",
        "For {concept}, produce the answer using the 'procedure' operation.",
        "Apply the planning operation to {concept}.",
        "Handle {concept} by giving an ordered course of action.",
        "For {concept}, return practical steps.",
        "Use operation code I: plan around {concept}.",
        "In the shared analysis frame, resolve {concept} by sequencing actions.",
    ],
    "counterfactual": [
        "What if {concept} were different?",
        "Imagine an alternative version of {concept} and analyze it.",
        "For {concept}, produce the answer using the 'alternative' operation.",
        "Apply the counterfactual operation to {concept}.",
        "Handle {concept} by changing an assumption.",
        "For {concept}, return a hypothetical variation.",
        "Use operation code J: alter a condition involving {concept}.",
        "In the shared analysis frame, resolve {concept} by considering an alternative.",
    ],
}


def build_dataset() -> pd.DataFrame:
    rows = []
    for concept in CONCEPTS:
        for policy_id in POLICY_IDS:
            for style_id, template in enumerate(TEMPLATES[policy_id]):
                rows.append({
                    "prompt_id": len(rows),
                    "concept": concept,
                    "policy_id": policy_id,
                    "constraint": policy_id,
                    "seed_id": f"{concept}__{policy_id}",
                    "style_id": style_id,
                    "prompt": template.format(concept=concept),
                })
    return pd.DataFrame(rows)


def load_dataset(cfg: CFG) -> pd.DataFrame:
    candidates = [cfg.INPUT_CSV] + list(cfg.FALLBACK_INPUTS)
    for p in candidates:
        path = Path(p)
        if path.exists():
            print(f"[SEM-DSTA-2] Loading dataset: {path}")
            df = pd.read_csv(path)
            break
    else:
        print("[SEM-DSTA-2] No dataset found; building default dataset.")
        df = build_dataset()

    # Normalize columns.
    if "policy_id" not in df.columns:
        if "constraint" in df.columns:
            df["policy_id"] = df["constraint"]
        else:
            raise ValueError("Dataset must contain either policy_id or constraint.")
    if "constraint" not in df.columns:
        df["constraint"] = df["policy_id"]
    if "seed_id" not in df.columns:
        df["seed_id"] = df["concept"].astype(str) + "__" + df["policy_id"].astype(str)
    if "prompt_id" not in df.columns:
        df["prompt_id"] = np.arange(len(df))
    if "style_id" not in df.columns:
        raise ValueError("Dataset must contain style_id for surface-family heldout.")
    if "prompt" not in df.columns:
        raise ValueError("Dataset must contain prompt column.")

    if cfg.MAX_ROWS and cfg.MAX_ROWS > 0:
        df = df.sample(n=min(cfg.MAX_ROWS, len(df)), random_state=cfg.RANDOM_SEED).reset_index(drop=True)
        df["prompt_id"] = np.arange(len(df))
    return df


# ============================================================
# UTILITIES
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def json_dump(obj: Any, path: Path):
    def convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, float) and np.isnan(o):
            return None
        return str(o)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=convert)


def get_torch_dtype(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    return "auto"


def safe_mean(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def safe_std(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.std(xs)) if xs else float("nan")


def cosine_np(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


# ============================================================
# MODEL + FEATURE EXTRACTION
# ============================================================

def load_model_and_tokenizer(cfg: CFG):
    print(f"[SEM-DSTA-2] Loading model: {cfg.MODEL_PATH}")
    dtype = get_torch_dtype(cfg.DTYPE)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
        device_map="auto" if cfg.DEVICE == "cuda" else None,
    )
    if cfg.DEVICE != "cuda":
        model.to(cfg.DEVICE)
    model.eval()
    return model, tokenizer


def get_lm_head_weight(model):
    if hasattr(model, "get_output_embeddings") and model.get_output_embeddings() is not None:
        return model.get_output_embeddings().weight
    raise RuntimeError("Could not find output embedding / lm_head weight.")


@torch.no_grad()
def extract_batch_features(prompts: List[str], model, tokenizer, cfg: CFG) -> List[Dict[str, Any]]:
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg.MAX_LENGTH,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    out = model(
        **enc,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    hidden_states = out.hidden_states
    W = get_lm_head_weight(model).to(model.device)
    attention_mask = enc["attention_mask"]
    last_pos = attention_mask.sum(dim=1) - 1

    batch_rows = []

    for bi in range(len(prompts)):
        row = {}
        prev_center_by_k = {}
        prev_set_by_k = {}

        # Store per-layer/per-k centers internally for window path metrics.
        centers_by_k = {k: {} for k in cfg.TOPK_LIST}
        sets_by_k = {k: {} for k in cfg.TOPK_LIST}
        scalars_by_k = {k: {} for k in cfg.TOPK_LIST}

        for layer in cfg.LAYERS:
            hs_index = layer + 1
            if hs_index >= len(hidden_states):
                continue

            h = hidden_states[hs_index][bi, last_pos[bi], :]
            logits = torch.matmul(h, W.T)

            for k in cfg.TOPK_LIST:
                vals, ids = torch.topk(logits, k=k, dim=-1)
                vals_np = vals.detach().float().cpu().numpy()
                ids_np = ids.detach().cpu().numpy().astype(np.int64)

                emb = W[ids].detach().float()
                center = emb.mean(dim=0)
                center_np = center.cpu().numpy()

                emb_norm = torch.nn.functional.normalize(emb, dim=-1)
                c_norm = torch.nn.functional.normalize(center.unsqueeze(0), dim=-1)
                cos = (emb_norm * c_norm).sum(dim=-1)
                spread = float((1.0 - cos).mean().detach().cpu())

                probs = torch.softmax(vals.float(), dim=-1)
                entropy = float((-(probs * torch.log(probs + 1e-12)).sum()).detach().cpu())

                id_set = set(ids_np.tolist())

                prefix = f"k{k}_L{layer}"
                row[f"{prefix}_spread"] = spread
                row[f"{prefix}_entropy"] = entropy
                row[f"{prefix}_gap"] = float(vals_np[0] - vals_np[-1])
                row[f"{prefix}_meanlogit"] = float(vals_np.mean())
                row[f"{prefix}_center_norm"] = float(np.linalg.norm(center_np))

                if k in prev_center_by_k:
                    row[f"{prefix}_center_shift_cosdist"] = 1.0 - cosine_np(prev_center_by_k[k], center_np)
                    row[f"{prefix}_center_shift_l2"] = float(np.linalg.norm(center_np - prev_center_by_k[k]))
                else:
                    row[f"{prefix}_center_shift_cosdist"] = 0.0
                    row[f"{prefix}_center_shift_l2"] = 0.0

                if k in prev_set_by_k:
                    jac = jaccard(prev_set_by_k[k], id_set)
                    row[f"{prefix}_jaccard_prev"] = jac
                    row[f"{prefix}_jaccard_dist_prev"] = 1.0 - jac
                else:
                    row[f"{prefix}_jaccard_prev"] = 1.0
                    row[f"{prefix}_jaccard_dist_prev"] = 0.0

                prev_center_by_k[k] = center_np
                prev_set_by_k[k] = id_set
                centers_by_k[k][layer] = center_np
                sets_by_k[k][layer] = id_set
                scalars_by_k[k][layer] = {
                    "spread": spread,
                    "entropy": entropy,
                    "gap": float(vals_np[0] - vals_np[-1]),
                    "meanlogit": float(vals_np.mean()),
                    "center_norm": float(np.linalg.norm(center_np)),
                    "shift_cosdist": row[f"{prefix}_center_shift_cosdist"],
                    "shift_l2": row[f"{prefix}_center_shift_l2"],
                    "jaccard_dist": row[f"{prefix}_jaccard_dist_prev"],
                }

        # Window summaries for each k.
        for k in cfg.TOPK_LIST:
            centers = centers_by_k[k]
            scalars = scalars_by_k[k]

            for wname, layers in cfg.WINDOWS.items():
                valid = [l for l in layers if l in centers]
                if not valid:
                    continue
                p = f"k{k}_{wname}"

                for sname in ["spread", "entropy", "gap", "meanlogit", "center_norm", "shift_cosdist", "shift_l2", "jaccard_dist"]:
                    vals = [scalars[l][sname] for l in valid]
                    row[f"{p}_{sname}_mean"] = safe_mean(vals)
                    row[f"{p}_{sname}_std"] = safe_std(vals)
                    row[f"{p}_{sname}_min"] = float(np.min(vals)) if vals else np.nan
                    row[f"{p}_{sname}_max"] = float(np.max(vals)) if vals else np.nan

                if len(valid) >= 2:
                    start = centers[valid[0]]
                    end = centers[valid[-1]]
                    row[f"{p}_start_end_cosdist"] = 1.0 - cosine_np(start, end)
                    row[f"{p}_start_end_l2"] = float(np.linalg.norm(end - start))
                    path_len = 0.0
                    for a, b in zip(valid[:-1], valid[1:]):
                        path_len += float(np.linalg.norm(centers[b] - centers[a]))
                    chord = float(np.linalg.norm(end - start)) + 1e-9
                    row[f"{p}_path_len"] = path_len
                    row[f"{p}_detour_ratio"] = path_len / chord
                else:
                    row[f"{p}_start_end_cosdist"] = 0.0
                    row[f"{p}_start_end_l2"] = 0.0
                    row[f"{p}_path_len"] = 0.0
                    row[f"{p}_detour_ratio"] = 0.0

        batch_rows.append(row)

    return batch_rows


def extract_all_features(df: pd.DataFrame, model, tokenizer, cfg: CFG) -> pd.DataFrame:
    prompts = df[cfg.PROMPT_COL].astype(str).tolist()
    rows = []
    for start in range(0, len(prompts), cfg.BATCH_SIZE):
        end = min(start + cfg.BATCH_SIZE, len(prompts))
        print(f"[SEM-DSTA-2] Extracting {start}:{end}/{len(prompts)}")
        rows.extend(extract_batch_features(prompts[start:end], model, tokenizer, cfg))
    feat_df = pd.DataFrame(rows)
    return pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)


# ============================================================
# FEATURE SELECTION / TRANSFORMS
# ============================================================

def feature_cols(df: pd.DataFrame, block: str) -> List[str]:
    exclude = {"prompt_id", "concept", "policy_id", "constraint", "seed_id", "style_id", "prompt"}
    numeric = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    if block == "all":
        return numeric
    if block == "scalar_only":
        keys = ["spread", "entropy", "gap", "meanlogit", "center_norm"]
        return [c for c in numeric if any(k in c for k in keys) and not ("center_shift" in c or "jaccard" in c or "start_end" in c or "path_len" in c or "detour" in c)]
    if block == "transport_only":
        keys = ["center_shift", "jaccard", "start_end", "path_len", "detour"]
        return [c for c in numeric if any(k in c for k in keys)]
    if block.startswith("window:"):
        w = block.split(":", 1)[1]
        return [c for c in numeric if f"_{w}_" in c]
    if block.startswith("k:"):
        kk = block.split(":", 1)[1]
        return [c for c in numeric if c.startswith(f"k{kk}_")]
    raise ValueError(block)


def make_X(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def residualize_group(Xtr, Xte, gtr, gte):
    Xtr = Xtr.copy()
    Xte = Xte.copy()
    gtr = np.asarray(gtr)
    gte = np.asarray(gte)
    global_mean = Xtr.mean(axis=0)
    means = {g: Xtr.loc[gtr == g, :].mean(axis=0) for g in np.unique(gtr)}

    for i, g in enumerate(gtr):
        Xtr.iloc[i, :] = Xtr.iloc[i, :] - means.get(g, global_mean)
    for i, g in enumerate(gte):
        Xte.iloc[i, :] = Xte.iloc[i, :] - means.get(g, global_mean)
    return Xtr, Xte


def additive_residual(Xtr, Xte, dftr, dfte):
    Xtr = Xtr.copy()
    Xte = Xte.copy()
    global_mean = Xtr.mean(axis=0)

    c_tr = dftr["concept"].values
    s_tr = dftr["style_id"].values
    c_te = dfte["concept"].values
    s_te = dfte["style_id"].values

    c_means = {c: Xtr.loc[c_tr == c, :].mean(axis=0) for c in np.unique(c_tr)}
    s_means = {s: Xtr.loc[s_tr == s, :].mean(axis=0) for s in np.unique(s_tr)}

    for i, (c, s) in enumerate(zip(c_tr, s_tr)):
        Xtr.iloc[i, :] = Xtr.iloc[i, :] - c_means.get(c, global_mean) - s_means.get(s, global_mean) + global_mean
    for i, (c, s) in enumerate(zip(c_te, s_te)):
        Xte.iloc[i, :] = Xte.iloc[i, :] - c_means.get(c, global_mean) - s_means.get(s, global_mean) + global_mean
    return Xtr, Xte


def transform_fold(name: str, Xtr, Xte, dftr, dfte):
    if name == "raw":
        return Xtr, Xte
    if name == "global_centered":
        mu = Xtr.mean(axis=0)
        return Xtr - mu, Xte - mu
    if name == "concept_quotient":
        return residualize_group(Xtr, Xte, dftr["concept"].values, dfte["concept"].values)
    if name == "style_quotient":
        return residualize_group(Xtr, Xte, dftr["style_id"].values, dfte["style_id"].values)
    if name == "concept_style_additive_residual":
        return additive_residual(Xtr, Xte, dftr, dfte)
    raise ValueError(name)


# ============================================================
# EVALUATION
# ============================================================

def make_classifier(name: str, cfg: CFG):
    if name == "ridge":
        return Pipeline([("scaler", StandardScaler()), ("clf", RidgeClassifier(class_weight="balanced"))])
    if name == "logreg":
        return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, solver="lbfgs", class_weight="balanced"))])
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=cfg.RANDOM_SEED,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
    raise ValueError(name)


def get_splits(df: pd.DataFrame, y: np.ndarray, mode: str, group_col: str, cfg: CFG):
    dummy = np.zeros((len(df), 1))
    if mode == "stratified":
        min_count = np.min(np.bincount(y))
        n_splits = max(2, min(cfg.N_SPLITS, int(min_count)))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.RANDOM_SEED)
        return list(cv.split(dummy, y))
    if mode == "group":
        groups = df[group_col].values
        n_groups = len(np.unique(groups))
        cv = GroupKFold(n_splits=min(cfg.N_SPLITS, n_groups))
        return list(cv.split(dummy, y, groups=groups))
    raise ValueError(mode)


def evaluate_cv(df, cols, target, model_name, cv_mode, group_col, transform, block, cfg):
    Xall = make_X(df, cols)
    le = LabelEncoder()
    y = le.fit_transform(df[target].astype(str).values)
    splits = get_splits(df, y, cv_mode, group_col, cfg)

    y_true, y_pred = [], []
    for tr, te in splits:
        if set(y[te]) - set(y[tr]):
            return {
                "target": target, "block": block, "model": model_name, "cv_mode": cv_mode,
                "group_col": group_col, "transform": transform, "accuracy": np.nan,
                "macro_f1": np.nan, "weighted_f1": np.nan, "skipped": True,
                "skip_reason": "unseen target class in test", "n_features": len(cols),
            }
        Xtr = Xall.iloc[tr, :].reset_index(drop=True)
        Xte = Xall.iloc[te, :].reset_index(drop=True)
        dftr = df.iloc[tr, :].reset_index(drop=True)
        dfte = df.iloc[te, :].reset_index(drop=True)
        Xtr_t, Xte_t = transform_fold(transform, Xtr, Xte, dftr, dfte)

        clf = make_classifier(model_name, cfg)
        clf.fit(Xtr_t.values.astype(np.float32), y[tr])
        pred = clf.predict(Xte_t.values.astype(np.float32))
        y_true.extend(y[te].tolist())
        y_pred.extend(pred.tolist())

    return {
        "target": target,
        "block": block,
        "model": model_name,
        "cv_mode": cv_mode,
        "group_col": group_col,
        "transform": transform,
        "n": int(len(df)),
        "n_classes": int(len(le.classes_)),
        "n_features": int(len(cols)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "skipped": False,
    }


def evaluate_surface(df, target, cv_mode, group_col, cfg):
    le = LabelEncoder()
    y = le.fit_transform(df[target].astype(str).values)
    prompts = df["prompt"].astype(str).values
    splits = get_splits(df, y, cv_mode, group_col, cfg)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
        ("clf", LinearSVC(class_weight="balanced", max_iter=12000)),
    ])

    y_true, y_pred = [], []
    for tr, te in splits:
        if set(y[te]) - set(y[tr]):
            return {
                "target": target, "model": "surface_tfidf_linearsvc", "cv_mode": cv_mode,
                "group_col": group_col, "accuracy": np.nan, "macro_f1": np.nan,
                "weighted_f1": np.nan, "skipped": True,
                "skip_reason": "unseen target class in test",
            }
        pipe.fit(prompts[tr], y[tr])
        pred = pipe.predict(prompts[te])
        y_true.extend(y[te].tolist())
        y_pred.extend(pred.tolist())

    return {
        "target": target,
        "model": "surface_tfidf_linearsvc",
        "cv_mode": cv_mode,
        "group_col": group_col,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "skipped": False,
    }


def run_eval(df: pd.DataFrame, cfg: CFG):
    targets = ["policy_id", "concept", "seed_id"]
    blocks = (
        ["scalar_only", "transport_only", "all", "k:100", "k:500"] +
        [f"window:{w}" for w in cfg.WINDOWS.keys()]
    )
    models = ["ridge", "logreg"] + (["rf"] if cfg.USE_RANDOM_FOREST else [])
    transforms = ["raw", "global_centered", "concept_quotient", "style_quotient", "concept_style_additive_residual"]
    cvs = [("stratified", None), ("group", "concept"), ("group", "style_id")]

    rows = []
    for target in targets:
        for block in blocks:
            cols = feature_cols(df, block)
            if not cols:
                continue
            for model in models:
                for transform in transforms:
                    for cv_mode, group_col in cvs:
                        if target == "concept" and group_col == "concept":
                            continue
                        print(f"[SEM-DSTA-2] Eval target={target}, block={block}, model={model}, transform={transform}, cv={cv_mode}, group={group_col}")
                        rows.append(evaluate_cv(df, cols, target, model, cv_mode, group_col, transform, block, cfg))

    baselines = []
    for target in targets:
        for cv_mode, group_col in cvs:
            if target == "concept" and group_col == "concept":
                continue
            print(f"[SEM-DSTA-2] Surface target={target}, cv={cv_mode}, group={group_col}")
            baselines.append(evaluate_surface(df, target, cv_mode, group_col, cfg))

    return pd.DataFrame(rows), pd.DataFrame(baselines)


# ============================================================
# RANK AUDIT
# ============================================================

def offline_transform_all(df, cols, transform):
    X0 = make_X(df, cols)
    X = pd.DataFrame(StandardScaler().fit_transform(X0), columns=cols)

    if transform == "raw":
        return X
    if transform == "global_centered":
        return X - X.mean(axis=0)
    if transform == "concept_quotient":
        Y = X.copy()
        for c in df["concept"].unique():
            idx = df["concept"] == c
            Y.loc[idx, :] = X.loc[idx, :] - X.loc[idx, :].mean(axis=0)
        return Y
    if transform == "style_quotient":
        Y = X.copy()
        for s in df["style_id"].unique():
            idx = df["style_id"] == s
            Y.loc[idx, :] = X.loc[idx, :] - X.loc[idx, :].mean(axis=0)
        return Y
    if transform == "concept_style_additive_residual":
        Y = X.copy()
        global_mean = X.mean(axis=0)
        c_means = {c: X.loc[df["concept"] == c, :].mean(axis=0) for c in df["concept"].unique()}
        s_means = {s: X.loc[df["style_id"] == s, :].mean(axis=0) for s in df["style_id"].unique()}
        for i, row in df.reset_index(drop=True).iterrows():
            Y.iloc[i, :] = X.iloc[i, :] - c_means[row["concept"]] - s_means[row["style_id"]] + global_mean
        return Y
    raise ValueError(transform)


def rank_audit(df: pd.DataFrame, cfg: CFG) -> pd.DataFrame:
    blocks = ["scalar_only", "transport_only", "all"] + [f"window:{w}" for w in cfg.WINDOWS.keys()]
    transforms = ["raw", "concept_quotient", "style_quotient", "concept_style_additive_residual"]

    rows = []
    for block in blocks:
        cols = feature_cols(df, block)
        if not cols:
            continue
        for transform in transforms:
            X = offline_transform_all(df, cols, transform)

            centroids = []
            labels = []
            for pid in sorted(df["policy_id"].unique()):
                idx = df["policy_id"] == pid
                centroids.append(X.loc[idx, :].mean(axis=0).values)
                labels.append(pid)

            C = np.vstack(centroids)
            n_comp = min(C.shape[0], C.shape[1])
            pca = PCA(n_components=n_comp, random_state=cfg.RANDOM_SEED)
            pca.fit(C)
            cumsum = np.cumsum(pca.explained_variance_ratio_)

            for i, ev in enumerate(pca.explained_variance_ratio_):
                rows.append({
                    "block": block,
                    "transform": transform,
                    "component": i + 1,
                    "explained_variance_ratio": float(ev),
                    "cumulative_explained_variance": float(cumsum[i]),
                    "rank_estimate": np.nan,
                    "n_policy": len(labels),
                    "n_features": len(cols),
                })

            for thr in [0.8, 0.9, 0.95]:
                k = int(np.searchsorted(cumsum, thr) + 1)
                rows.append({
                    "block": block,
                    "transform": transform,
                    "component": f"rank_for_{thr}",
                    "explained_variance_ratio": np.nan,
                    "cumulative_explained_variance": thr,
                    "rank_estimate": k,
                    "n_policy": len(labels),
                    "n_features": len(cols),
                })

    return pd.DataFrame(rows)


def feature_importance(df: pd.DataFrame, cfg: CFG) -> pd.DataFrame:
    cols = feature_cols(df, "all")
    X = make_X(df, cols)
    y = LabelEncoder().fit_transform(df["policy_id"].astype(str).values)
    rf = RandomForestClassifier(n_estimators=500, random_state=cfg.RANDOM_SEED, class_weight="balanced_subsample", n_jobs=-1)
    rf.fit(X.values.astype(np.float32), y)
    return pd.DataFrame({"target": "policy_id", "feature": cols, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)


# ============================================================
# SUMMARY
# ============================================================

def best(df, mask):
    sub = df[mask & (df.get("skipped", False) != True)]
    if len(sub) == 0:
        return None
    return sub.sort_values(["accuracy", "macro_f1"], ascending=False).iloc[0].to_dict()


def summarize(eval_df, surf_df, rank_df, cfg: CFG):
    summary = {
        "config": asdict(cfg),
        "best_policyid_by_block": {},
        "best_concept_by_block": {},
        "key_tests": {},
        "rank90_by_block": {},
        "compression_trend": {},
        "verdict": "UNDETERMINED",
        "interpretation": [],
    }

    blocks = sorted(eval_df["block"].dropna().unique())
    for block in blocks:
        b = best(eval_df, (eval_df["target"] == "policy_id") & (eval_df["block"] == block))
        if b:
            summary["best_policyid_by_block"][block] = b
        c = best(eval_df, (eval_df["target"] == "concept") & (eval_df["block"] == block))
        if c:
            summary["best_concept_by_block"][block] = c

    # Key: group concept and group style for main windows.
    for block in ["window:init_L0_6", "window:qwen_mech_L5_6", "window:mid_L7_19", "window:decision_L20_25", "window:full_L0_25", "all"]:
        for transform in ["raw", "concept_quotient", "style_quotient", "concept_style_additive_residual"]:
            for group_col in ["concept", "style_id"]:
                b = best(
                    eval_df,
                    (eval_df["target"] == "policy_id") &
                    (eval_df["block"] == block) &
                    (eval_df["transform"] == transform) &
                    (eval_df["group_col"] == group_col)
                )
                if b:
                    key = f"{block}__{transform}__group_{group_col}"
                    summary["key_tests"][key] = {
                        "acc": float(b["accuracy"]),
                        "macro_f1": float(b["macro_f1"]),
                        "model": b["model"],
                        "n_features": int(b["n_features"]),
                    }

    # Rank trend under raw and concept_style residual.
    for block in ["window:init_L0_6", "window:qwen_mech_L5_6", "window:mid_L7_19", "window:decision_L20_25", "window:full_L0_25", "all", "scalar_only", "transport_only"]:
        for transform in ["raw", "concept_style_additive_residual"]:
            sub = rank_df[
                (rank_df["block"] == block) &
                (rank_df["transform"] == transform) &
                (rank_df["component"].astype(str).str.startswith("rank_for_"))
            ]
            if len(sub):
                summary["rank90_by_block"][f"{block}__{transform}"] = {
                    str(row["component"]): int(row["rank_estimate"]) for _, row in sub.iterrows()
                }

    # Compression trend.
    def rank90(block, transform="raw"):
        return summary["rank90_by_block"].get(f"{block}__{transform}", {}).get("rank_for_0.9", None)

    r_init = rank90("window:init_L0_6")
    r_mid = rank90("window:mid_L7_19")
    r_dec = rank90("window:decision_L20_25")
    r_full = rank90("window:full_L0_25")
    summary["compression_trend"] = {
        "rank90_init": r_init,
        "rank90_mid": r_mid,
        "rank90_decision": r_dec,
        "rank90_full": r_full,
        "init_minus_decision": (r_init - r_dec) if r_init is not None and r_dec is not None else None,
        "init_minus_full": (r_init - r_full) if r_init is not None and r_full is not None else None,
    }

    # Verdict logic.
    policy_concept_init = summary["key_tests"].get("window:init_L0_6__raw__group_concept", {}).get("acc", 0.0)
    policy_concept_mid = summary["key_tests"].get("window:mid_L7_19__raw__group_concept", {}).get("acc", 0.0)
    policy_concept_dec = summary["key_tests"].get("window:decision_L20_25__raw__group_concept", {}).get("acc", 0.0)

    rank_compression = False
    if r_init is not None and r_dec is not None:
        rank_compression = (r_init - r_dec) >= 4

    if policy_concept_init >= 0.85 and rank_compression and r_dec is not None and r_dec <= 10:
        summary["verdict"] = "PASS_STRONG_POLICYID_LOW_RANK_COMPRESSION"
    elif policy_concept_init >= 0.85 and rank_compression:
        summary["verdict"] = "PASS_LITE_POLICYID_LOW_RANK_COMPRESSION"
    elif policy_concept_init >= 0.85 and policy_concept_mid >= 0.85:
        summary["verdict"] = "PARTIAL_PASS_POLICYID_RECOVERY_NO_RANK_COMPRESSION"
    else:
        summary["verdict"] = "NO_PASS_YET"

    summary["interpretation"].append("If rank90 decreases from init to decision/full windows, PolicyID compresses across layers.")
    summary["interpretation"].append("If PolicyID recovery stays high while rank decreases, high-dimensional Sigma_init becomes low-rank policy flow.")
    summary["interpretation"].append("If rank does not decrease, low-rank policy structure may only appear in complete trajectory summaries or different features.")
    summary["interpretation"].append("Style-heldout remains a separate SurfaceForm robustness test.")

    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    warnings.filterwarnings("ignore")
    set_seed(cfg.RANDOM_SEED)
    outdir = Path(cfg.OUTPUT_DIR)
    ensure_dir(str(outdir))

    print("=" * 100)
    print("SEM-DSTA-2: PolicyID Low-Rank Compression Across Layers")
    print("=" * 100)
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    json_dump(asdict(cfg), outdir / "sem_dsta2_config.json")

    df = load_dataset(cfg)
    df.to_csv(outdir / "sem_dsta2_dataset.csv", index=False, encoding="utf-8-sig")
    print(f"[SEM-DSTA-2] Dataset n={len(df)} saved.")

    feat_path = outdir / "sem_dsta2_features_raw.csv"
    if cfg.REUSE_FEATURES_IF_EXISTS and feat_path.exists():
        print(f"[SEM-DSTA-2] Reusing features: {feat_path}")
        feat_df = pd.read_csv(feat_path)
    else:
        model, tokenizer = load_model_and_tokenizer(cfg)
        feat_df = extract_all_features(df, model, tokenizer, cfg)
        feat_df.to_csv(feat_path, index=False, encoding="utf-8-sig")
        print(f"[SEM-DSTA-2] Features saved: {feat_path} shape={feat_df.shape}")

    eval_df, surf_df = run_eval(feat_df, cfg)
    eval_df.to_csv(outdir / "sem_dsta2_window_recovery.csv", index=False, encoding="utf-8-sig")
    surf_df.to_csv(outdir / "sem_dsta2_surface_baseline.csv", index=False, encoding="utf-8-sig")

    # Convenience subsets.
    eval_df[eval_df["target"].isin(["concept", "policy_id"])].to_csv(
        outdir / "sem_dsta2_concept_vs_policyid.csv", index=False, encoding="utf-8-sig"
    )

    rank_df = rank_audit(feat_df, cfg)
    rank_df.to_csv(outdir / "sem_dsta2_rank_by_window.csv", index=False, encoding="utf-8-sig")

    if cfg.SAVE_FEATURE_IMPORTANCE:
        imp_df = feature_importance(feat_df, cfg).head(300)
        imp_df.to_csv(outdir / "sem_dsta2_feature_importance.csv", index=False, encoding="utf-8-sig")

    summary = summarize(eval_df, surf_df, rank_df, cfg)
    json_dump(summary, outdir / "sem_dsta2_results_summary.json")

    # Compact CSV summary for quick reading.
    comp = pd.DataFrame([{
        "verdict": summary["verdict"],
        **summary["compression_trend"],
    }])
    comp.to_csv(outdir / "sem_dsta2_compression_summary.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("[SEM-DSTA-2] SUMMARY")
    print("=" * 100)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n[SEM-DSTA-2] Output files:")
    for p in [
        "sem_dsta2_config.json",
        "sem_dsta2_dataset.csv",
        "sem_dsta2_features_raw.csv",
        "sem_dsta2_window_recovery.csv",
        "sem_dsta2_rank_by_window.csv",
        "sem_dsta2_compression_summary.csv",
        "sem_dsta2_concept_vs_policyid.csv",
        "sem_dsta2_surface_baseline.csv",
        "sem_dsta2_feature_importance.csv",
        "sem_dsta2_results_summary.json",
    ]:
        print("  -", outdir / p)


if __name__ == "__main__":
    main()
