# -*- coding: utf-8 -*-
"""
SEM-5C.3: Component-Swap Memory Control Audit

Purpose
-------
SEM-5C.2 showed a strong correction:
    qC / commit geometry is the primary MemoryUnit address.
    qO / shape geometry is likely the trajectory adapter.

SEM-5C.3 asks a sharper causal question:
    After commit-based retrieval finds a MemoryUnit, which component actually changes
    the live trajectory?

Hypothesis
----------
    mu_commit : address / identity lookup
    mu_shape  : trajectory guidance / control prior

Therefore:
    commit-only retrieval should improve hit rate;
    shape injection should carry most trajectory-improving effect;
    commit injection alone should not explain the full improvement;
    oracle > retrieved > shuffle should remain under component-swap controls.

Outputs
-------
sem5c3_outputs/
    sem5c3_addressing_results.csv
    sem5c3_component_injection_results.csv
    sem5c3_component_summary.csv
    sem5c3_pass_checks.csv
    sem5c3_report.json

Default inputs
--------------
sem5a2_outputs/sem5a2_dataset.csv
Optional external memory:
sem4e_outputs/memory_units.csv
sem4e_outputs/memory_units.npz

Notes
-----
This script does not require clean/conflict answer columns.
It is designed to be robust to missing SEM-4E memory by building a leave-one-out
MemoryBank from live trajectories.
"""

import os
import json
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# Config
# =========================

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

MODEL_PATH = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"
DATASET_CSV = r"sem5a2_outputs\sem5a2_dataset.csv"

MEMORY_UNITS_CSV = r"sem4e_outputs\memory_units.csv"
MEMORY_UNITS_NPZ = r"sem4e_outputs\memory_units.npz"

OUT_DIR = Path("sem5c3_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SHAPE_LAYERS = list(range(7, 20))
COMMIT_LAYERS = list(range(23, 26))
MAX_LENGTH = 512
MAX_ROWS = None        # set to e.g. 96 for a smoke test

# commit-only is intentionally the default address after SEM-5C.2.
RETRIEVAL_WEIGHTS = [
    {"name": "commit_only", "wo": 0.0, "wc": 1.0},
    {"name": "shape_commit_30", "wo": 0.3, "wc": 0.7},
    {"name": "shape_commit_50", "wo": 0.5, "wc": 0.5},
    {"name": "shape_only", "wo": 1.0, "wc": 0.0},
]
BEST_RETRIEVAL_NAME = "commit_only"

ALPHAS = [0.10, 0.20, 0.30]

# Component injections.
# alpha_shape_mul / alpha_commit_mul are multiplied by alpha.
COMPONENT_CONDITIONS = [
    {"condition": "raw", "source": "none", "shape_source": "none", "commit_source": "none", "alpha_shape_mul": 0.0, "alpha_commit_mul": 0.0},

    {"condition": "retrieved_shape_only", "source": "retrieved", "shape_source": "retrieved", "commit_source": "none", "alpha_shape_mul": 1.0, "alpha_commit_mul": 0.0},
    {"condition": "retrieved_commit_only", "source": "retrieved", "shape_source": "none", "commit_source": "retrieved", "alpha_shape_mul": 0.0, "alpha_commit_mul": 1.0},
    {"condition": "retrieved_full", "source": "retrieved", "shape_source": "retrieved", "commit_source": "retrieved", "alpha_shape_mul": 1.0, "alpha_commit_mul": 1.0},

    {"condition": "oracle_shape_only", "source": "oracle", "shape_source": "oracle", "commit_source": "none", "alpha_shape_mul": 1.0, "alpha_commit_mul": 0.0},
    {"condition": "oracle_commit_only", "source": "oracle", "shape_source": "none", "commit_source": "oracle", "alpha_shape_mul": 0.0, "alpha_commit_mul": 1.0},
    {"condition": "oracle_full", "source": "oracle", "shape_source": "oracle", "commit_source": "oracle", "alpha_shape_mul": 1.0, "alpha_commit_mul": 1.0},

    {"condition": "retrieved_shape_oracle_commit", "source": "swap", "shape_source": "retrieved", "commit_source": "oracle", "alpha_shape_mul": 1.0, "alpha_commit_mul": 1.0},
    {"condition": "oracle_shape_retrieved_commit", "source": "swap", "shape_source": "oracle", "commit_source": "retrieved", "alpha_shape_mul": 1.0, "alpha_commit_mul": 1.0},

    {"condition": "shuffle_shape_only", "source": "shuffle", "shape_source": "shuffle", "commit_source": "none", "alpha_shape_mul": 1.0, "alpha_commit_mul": 0.0},
    {"condition": "shuffle_full", "source": "shuffle", "shape_source": "shuffle", "commit_source": "shuffle", "alpha_shape_mul": 1.0, "alpha_commit_mul": 1.0},
]

# Optional semantic controls, enabled when family names parse as Concept__Operator.
ADD_FAMILY_FACTOR_CONTROLS = True

# =========================
# Utilities
# =========================

def safe_read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(p)


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def get_prompt_col(df: pd.DataFrame) -> str:
    col = find_col(df, ["prompt", "text", "input", "query"])
    if col is None:
        raise ValueError(f"Cannot find prompt column. Columns={list(df.columns)}")
    return col


def get_family_col(df: pd.DataFrame) -> str:
    col = find_col(df, ["seed_family", "family_id", "SeedFamily", "family", "Family", "memory_family"])
    if col is None:
        raise ValueError("Cannot find seed family column. Expected seed_family/family_id/family.")
    return col


def get_row_id_col(df: pd.DataFrame) -> Optional[str]:
    return find_col(df, ["row_id", "id", "idx", "index"])


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < eps:
        return v
    return v / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(normalize(a), normalize(b)))


def mean_pool_layers(layer_vecs: Dict[int, np.ndarray], layers: List[int]) -> np.ndarray:
    xs = [layer_vecs[l] for l in layers if l in layer_vecs]
    if not xs:
        raise ValueError(f"No requested layers found: {layers}")
    return np.mean(np.stack(xs, axis=0), axis=0).astype(np.float32)


def layer_hidden_dict(outputs) -> Dict[int, torch.Tensor]:
    hs = outputs.hidden_states
    return {layer_idx: hs[layer_idx + 1] for layer_idx in range(len(hs) - 1)}


def parse_family(fam: str) -> Tuple[str, str]:
    fam = str(fam)
    if "__" in fam:
        a, b = fam.split("__", 1)
        return a, b
    if "::" in fam:
        a, b = fam.split("::", 1)
        return a, b
    return fam, ""

# =========================
# Hook
# =========================

class ComponentInjector:
    def __init__(self, model, mu_shape, mu_commit, alpha_shape: float, alpha_commit: float):
        self.model = model
        self.mu_shape = None if mu_shape is None else torch.tensor(mu_shape, dtype=torch.float32)
        self.mu_commit = None if mu_commit is None else torch.tensor(mu_commit, dtype=torch.float32)
        self.alpha_shape = float(alpha_shape)
        self.alpha_commit = float(alpha_commit)
        self.handles = []

    def _hook(self, layer_idx):
        def fn(module, inputs, output):
            if isinstance(output, tuple):
                h = output[0]
                rest = output[1:]
            else:
                h = output
                rest = None
            h2 = h.clone()
            if layer_idx in SHAPE_LAYERS and self.alpha_shape != 0 and self.mu_shape is not None:
                v = self.mu_shape.to(h2.device, h2.dtype)
                h2[:, -1, :] = h2[:, -1, :] + self.alpha_shape * v
            if layer_idx in COMMIT_LAYERS and self.alpha_commit != 0 and self.mu_commit is not None:
                v = self.mu_commit.to(h2.device, h2.dtype)
                h2[:, -1, :] = h2[:, -1, :] + self.alpha_commit * v
            if rest is not None:
                return (h2,) + rest
            return h2
        return fn

    def __enter__(self):
        layers = self.model.model.layers
        for idx in sorted(set(SHAPE_LAYERS + COMMIT_LAYERS)):
            if 0 <= idx < len(layers):
                self.handles.append(layers[idx].register_forward_hook(self._hook(idx)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles = []

# =========================
# Forward
# =========================

@torch.no_grad()
def forward_collect(model, tokenizer, prompt: str) -> Dict[int, np.ndarray]:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(DEVICE)
    outputs = model(**enc, output_hidden_states=True, use_cache=False)
    hdict = layer_hidden_dict(outputs)
    return {l: h[:, -1, :].detach()[0].float().cpu().numpy().astype(np.float32) for l, h in hdict.items()}


def extract_shape_commit(hidden_by_layer: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    shape = mean_pool_layers(hidden_by_layer, SHAPE_LAYERS)
    commit = mean_pool_layers(hidden_by_layer, COMMIT_LAYERS)
    return normalize(shape), normalize(commit)

# =========================
# MemoryBank
# =========================

def load_sem4e_memory_if_available(hidden_dim: int) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    csv_p = Path(MEMORY_UNITS_CSV)
    npz_p = Path(MEMORY_UNITS_NPZ)
    if not (csv_p.exists() and npz_p.exists()):
        return None
    meta = pd.read_csv(csv_p)
    fam_col = find_col(meta, ["seed_family", "family_id", "SeedFamily", "family", "Family", "memory_family"])
    if fam_col is None:
        print("[MemoryBank] SEM-4E CSV found but family column missing; skip external memory.")
        return None
    arr = np.load(npz_p)
    shape_key = next((k for k in ["mu_shape", "mu_F_shape", "shape", "memory_shape"] if k in arr), None)
    commit_key = next((k for k in ["mu_commit", "mu_F_commit", "commit", "memory_commit"] if k in arr), None)
    if shape_key is None:
        print(f"[MemoryBank] SEM-4E NPZ has no shape key. Keys={list(arr.keys())}; skip.")
        return None
    shapes = arr[shape_key]
    if shapes.shape[-1] != hidden_dim:
        print(f"[MemoryBank] SEM-4E dim mismatch {shapes.shape[-1]} != {hidden_dim}; skip.")
        return None
    commits = arr[commit_key] if commit_key is not None and arr[commit_key].shape[-1] == hidden_dim else None
    bank = {}
    for i, fam in enumerate(meta[fam_col].astype(str).tolist()):
        bank[str(fam)] = {
            "shape": normalize(shapes[i].astype(np.float32)),
            "commit": normalize(commits[i].astype(np.float32)) if commits is not None else normalize(shapes[i].astype(np.float32)),
        }
    print(f"[MemoryBank] Loaded external SEM-4E memory: {len(bank)} families")
    return bank


def build_live_memory_bank(raw_records: List[Dict], exclude_row_idx: Optional[int] = None) -> Dict[str, Dict[str, np.ndarray]]:
    by_family: Dict[str, List[Dict]] = {}
    for rec in raw_records:
        if exclude_row_idx is not None and rec["row_idx"] == exclude_row_idx:
            continue
        by_family.setdefault(str(rec["family"]), []).append(rec)
    # fallback if leave-one-out removes singleton family
    all_by_family: Dict[str, List[Dict]] = {}
    for rec in raw_records:
        all_by_family.setdefault(str(rec["family"]), []).append(rec)
    for fam, xs in all_by_family.items():
        if fam not in by_family or not by_family[fam]:
            by_family[fam] = xs
    bank = {}
    for fam, xs in by_family.items():
        shape = np.mean(np.stack([x["shape_raw"] for x in xs], axis=0), axis=0)
        commit = np.mean(np.stack([x["commit_raw"] for x in xs], axis=0), axis=0)
        bank[fam] = {"shape": normalize(shape), "commit": normalize(commit)}
    return bank

# =========================
# Retrieval / scoring
# =========================

def ranked_retrieve(q_shape, q_commit, bank, wo: float, wc: float):
    rows = []
    for fam, mem in bank.items():
        s_shape = cosine(q_shape, mem["shape"])
        s_commit = cosine(q_commit, mem["commit"])
        s = wo * s_shape + wc * s_commit
        rows.append((fam, mem, float(s), float(s_shape), float(s_commit)))
    rows.sort(key=lambda x: x[2], reverse=True)
    return rows


def rank_of_family(scores: Dict[str, float], oracle_family: str) -> int:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    for i, (fam, _) in enumerate(ordered, start=1):
        if fam == oracle_family:
            return i
    return len(ordered) + 1


def margin_to_oracle(scores: Dict[str, float], oracle_family: str) -> float:
    oracle = scores.get(oracle_family, np.nan)
    others = [v for fam, v in scores.items() if fam != oracle_family]
    if not others:
        return np.nan
    return float(oracle - max(others))


def family_scores(vec: np.ndarray, bank: Dict[str, Dict[str, np.ndarray]], key: str):
    return {fam: cosine(vec, mem[key]) for fam, mem in bank.items()}


def choose_family_control(bank: Dict[str, Dict[str, np.ndarray]], oracle_family: str, mode: str, rng: random.Random) -> Optional[str]:
    fams = list(bank.keys())
    if mode == "oracle":
        return oracle_family if oracle_family in bank else None
    if mode == "shuffle":
        pool = [f for f in fams if f != oracle_family]
        return rng.choice(pool) if pool else oracle_family
    if mode == "same_concept_wrong_operator":
        c0, o0 = parse_family(oracle_family)
        pool = [f for f in fams if parse_family(f)[0] == c0 and parse_family(f)[1] != o0]
        return rng.choice(pool) if pool else None
    if mode == "same_operator_wrong_concept":
        c0, o0 = parse_family(oracle_family)
        pool = [f for f in fams if parse_family(f)[1] == o0 and parse_family(f)[0] != c0]
        return rng.choice(pool) if pool else None
    return None


def vector_from_source(source_name: str, oracle_family: str, retrieved_family: Optional[str], bank, key: str, rng: random.Random):
    if source_name == "none":
        return None, None
    if source_name == "oracle":
        fam = oracle_family if oracle_family in bank else None
    elif source_name == "retrieved":
        fam = retrieved_family if retrieved_family in bank else None
    elif source_name == "shuffle":
        fam = choose_family_control(bank, oracle_family, "shuffle", rng)
    elif source_name in ["same_concept_wrong_operator", "same_operator_wrong_concept"]:
        fam = choose_family_control(bank, oracle_family, source_name, rng)
    else:
        fam = None
    if fam is None:
        return None, None
    return bank[fam][key], fam

# =========================
# Main
# =========================

def run():
    print("=" * 90)
    print("SEM-5C.3 Component-Swap Memory Control Audit")
    print("=" * 90)
    print(f"DEVICE={DEVICE}, DTYPE={DTYPE}")
    print(f"MODEL_PATH={MODEL_PATH}")

    dataset = safe_read_csv(DATASET_CSV)
    if MAX_ROWS is not None:
        dataset = dataset.head(MAX_ROWS).copy()
    prompt_col = get_prompt_col(dataset)
    family_col = get_family_col(dataset)
    row_id_col = get_row_id_col(dataset)
    print(f"[Data] n={len(dataset)} prompt_col={prompt_col} family_col={family_col}")

    print("[Model] Loading...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    if DEVICE != "cuda":
        model.to(DEVICE)
    hidden_dim = int(model.config.hidden_size)
    print(f"[Model] hidden_dim={hidden_dim}, n_layers={model.config.num_hidden_layers}")

    print("[Stage 1] Collect raw trajectories...")
    raw_records = []
    for pos, (idx, row) in enumerate(dataset.iterrows()):
        prompt = str(row[prompt_col])
        fam = str(row[family_col])
        row_id = row[row_id_col] if row_id_col is not None else idx
        h = forward_collect(model, tokenizer, prompt)
        shape, commit = extract_shape_commit(h)
        concept, operator = parse_family(fam)
        raw_records.append({
            "row_idx": int(idx), "row_pos": int(pos), "row_id": row_id,
            "family": fam, "concept": concept, "operator": operator, "prompt": prompt,
            "shape_raw": shape, "commit_raw": commit,
        })
        if (pos + 1) % 10 == 0 or (pos + 1) == len(dataset):
            print(f"[Raw] {pos+1}/{len(dataset)}")

    external_bank = load_sem4e_memory_if_available(hidden_dim)

    print("[Stage 2] Addressing sweep...")
    addr_rows = []
    weight_hits = {}
    best_weight = None
    best_hit = -1.0
    retrieval_cache = {}

    for weight in RETRIEVAL_WEIGHTS:
        hits = []
        ranks = []
        for rec in raw_records:
            oracle_family = rec["family"]
            bank = external_bank if (external_bank is not None and oracle_family in external_bank) else build_live_memory_bank(raw_records, rec["row_idx"])
            ranked = ranked_retrieve(rec["shape_raw"], rec["commit_raw"], bank, weight["wo"], weight["wc"])
            fams = [x[0] for x in ranked]
            top = ranked[0]
            hit = int(top[0] == oracle_family)
            rank = fams.index(oracle_family) + 1 if oracle_family in fams else len(fams) + 1
            hits.append(hit)
            ranks.append(rank)
            addr_rows.append({
                "row_idx": rec["row_idx"], "row_id": rec["row_id"], "family": oracle_family,
                "retrieval_weight": weight["name"], "wo": weight["wo"], "wc": weight["wc"],
                "top1_family": top[0], "top1_hit": hit, "oracle_rank": rank,
                "top1_score": top[2], "top1_shape_sim": top[3], "top1_commit_sim": top[4],
            })
            if weight["name"] == BEST_RETRIEVAL_NAME:
                retrieval_cache[rec["row_idx"]] = {"retrieved_family": top[0], "oracle_rank": rank, "top1_hit": hit}
        hit_rate = float(np.mean(hits))
        weight_hits[weight["name"]] = {"top1_hit": hit_rate, "mean_oracle_rank": float(np.mean(ranks))}
        print(f"[Address] {weight['name']}: top1_hit={hit_rate:.4f}, mean_rank={np.mean(ranks):.2f}")
        if hit_rate > best_hit:
            best_hit = hit_rate
            best_weight = weight

    pd.DataFrame(addr_rows).to_csv(OUT_DIR / "sem5c3_addressing_results.csv", index=False, encoding="utf-8-sig")

    print(f"[Stage 3] Component-swap injection using {BEST_RETRIEVAL_NAME}...")
    inj_rows = []
    rng = random.Random(RANDOM_SEED)

    # Add family-factor controls only if family names are parseable.
    component_conditions = list(COMPONENT_CONDITIONS)
    if ADD_FAMILY_FACTOR_CONTROLS and any(parse_family(r["family"])[1] for r in raw_records):
        component_conditions += [
            {"condition": "same_concept_wrong_operator_shape", "source": "same_concept_wrong_operator", "shape_source": "same_concept_wrong_operator", "commit_source": "none", "alpha_shape_mul": 1.0, "alpha_commit_mul": 0.0},
            {"condition": "same_operator_wrong_concept_shape", "source": "same_operator_wrong_concept", "shape_source": "same_operator_wrong_concept", "commit_source": "none", "alpha_shape_mul": 1.0, "alpha_commit_mul": 0.0},
        ]

    for pos, rec in enumerate(raw_records):
        oracle_family = rec["family"]
        prompt = rec["prompt"]
        bank = external_bank if (external_bank is not None and oracle_family in external_bank) else build_live_memory_bank(raw_records, rec["row_idx"])
        retrieved_family = retrieval_cache.get(rec["row_idx"], {}).get("retrieved_family")
        if retrieved_family is None:
            w = next(x for x in RETRIEVAL_WEIGHTS if x["name"] == BEST_RETRIEVAL_NAME)
            retrieved_family = ranked_retrieve(rec["shape_raw"], rec["commit_raw"], bank, w["wo"], w["wc"])[0][0]

        raw_shape_scores = family_scores(rec["shape_raw"], bank, "shape")
        raw_commit_scores = family_scores(rec["commit_raw"], bank, "commit")
        raw_shape_rank = rank_of_family(raw_shape_scores, oracle_family)
        raw_commit_rank = rank_of_family(raw_commit_scores, oracle_family)
        raw_shape_margin = margin_to_oracle(raw_shape_scores, oracle_family)
        raw_commit_margin = margin_to_oracle(raw_commit_scores, oracle_family)
        raw_shape_sim = cosine(rec["shape_raw"], bank[oracle_family]["shape"])
        raw_commit_sim = cosine(rec["commit_raw"], bank[oracle_family]["commit"])

        for cond in component_conditions:
            alphas = [0.0] if cond["condition"] == "raw" else ALPHAS
            for alpha in alphas:
                shape_vec, shape_mem_family = vector_from_source(cond["shape_source"], oracle_family, retrieved_family, bank, "shape", rng)
                commit_vec, commit_mem_family = vector_from_source(cond["commit_source"], oracle_family, retrieved_family, bank, "commit", rng)
                alpha_shape = float(alpha) * cond["alpha_shape_mul"]
                alpha_commit = float(alpha) * cond["alpha_commit_mul"]

                if cond["condition"] == "raw":
                    shape_new, commit_new = rec["shape_raw"], rec["commit_raw"]
                else:
                    with ComponentInjector(model, shape_vec, commit_vec, alpha_shape, alpha_commit):
                        h2 = forward_collect(model, tokenizer, prompt)
                    shape_new, commit_new = extract_shape_commit(h2)

                shape_scores = family_scores(shape_new, bank, "shape")
                commit_scores = family_scores(commit_new, bank, "commit")
                shape_rank = rank_of_family(shape_scores, oracle_family)
                commit_rank = rank_of_family(commit_scores, oracle_family)
                shape_margin = margin_to_oracle(shape_scores, oracle_family)
                commit_margin = margin_to_oracle(commit_scores, oracle_family)
                shape_sim = cosine(shape_new, bank[oracle_family]["shape"])
                commit_sim = cosine(commit_new, bank[oracle_family]["commit"])

                inj_rows.append({
                    "row_idx": rec["row_idx"], "row_id": rec["row_id"], "family": oracle_family,
                    "concept": rec["concept"], "operator": rec["operator"],
                    "retrieval_weight": BEST_RETRIEVAL_NAME,
                    "retrieved_family": retrieved_family,
                    "retrieval_hit": int(retrieved_family == oracle_family),
                    "condition": cond["condition"], "source": cond["source"], "alpha": alpha,
                    "alpha_shape": alpha_shape, "alpha_commit": alpha_commit,
                    "shape_memory_family": shape_mem_family,
                    "commit_memory_family": commit_mem_family,
                    "shape_sim_to_oracle": shape_sim,
                    "commit_sim_to_oracle": commit_sim,
                    "family_margin_shape": shape_margin,
                    "family_margin_commit": commit_margin,
                    "family_rank_shape": shape_rank,
                    "family_rank_commit": commit_rank,
                    "gain_shape_sim": shape_sim - raw_shape_sim,
                    "gain_commit_sim": commit_sim - raw_commit_sim,
                    "gain_margin_shape": shape_margin - raw_shape_margin,
                    "gain_margin_commit": commit_margin - raw_commit_margin,
                    "rank_improvement_shape": raw_shape_rank - shape_rank,
                    "rank_improvement_commit": raw_commit_rank - commit_rank,
                })

        if (pos + 1) % 10 == 0 or (pos + 1) == len(raw_records):
            print(f"[Inject] {pos+1}/{len(raw_records)}")

    inj_df = pd.DataFrame(inj_rows)
    inj_df.to_csv(OUT_DIR / "sem5c3_component_injection_results.csv", index=False, encoding="utf-8-sig")

    print("[Stage 4] Summarize...")
    nonraw = inj_df[inj_df["condition"] != "raw"].copy()
    summary = nonraw.groupby(["condition", "alpha"], as_index=False).agg(
        n=("row_idx", "count"),
        retrieval_hit=("retrieval_hit", "mean"),
        gain_shape_sim=("gain_shape_sim", "mean"),
        gain_commit_sim=("gain_commit_sim", "mean"),
        gain_margin_shape=("gain_margin_shape", "mean"),
        gain_margin_commit=("gain_margin_commit", "mean"),
        rank_improvement_shape=("rank_improvement_shape", "mean"),
        rank_improvement_commit=("rank_improvement_commit", "mean"),
    )
    summary.to_csv(OUT_DIR / "sem5c3_component_summary.csv", index=False, encoding="utf-8-sig")

    def metric(cond, alpha, col):
        sub = summary[(summary["condition"] == cond) & (np.isclose(summary["alpha"], alpha))]
        if sub.empty:
            return np.nan
        return float(sub.iloc[0][col])

    checks = []
    for alpha in ALPHAS:
        retrieved_shape = metric("retrieved_shape_only", alpha, "rank_improvement_shape")
        retrieved_commit = metric("retrieved_commit_only", alpha, "rank_improvement_shape")
        retrieved_full = metric("retrieved_full", alpha, "rank_improvement_shape")
        oracle_shape = metric("oracle_shape_only", alpha, "rank_improvement_shape")
        oracle_full = metric("oracle_full", alpha, "rank_improvement_shape")
        shuffle_shape = metric("shuffle_shape_only", alpha, "rank_improvement_shape")
        shuffle_full = metric("shuffle_full", alpha, "rank_improvement_shape")

        checks.append({
            "check": f"shape_guidance_alpha_{alpha}",
            "value": retrieved_shape,
            "baseline": shuffle_shape,
            "oracle": oracle_shape,
            "pass_lite": bool(retrieved_shape > 0),
            "pass_strong": bool(retrieved_shape > shuffle_shape and oracle_shape >= retrieved_shape),
            "note": "Retrieved shape-only should improve trajectory-shape rank over raw and shuffle."
        })
        checks.append({
            "check": f"component_specificity_alpha_{alpha}",
            "value": retrieved_shape - retrieved_commit,
            "baseline": 0.0,
            "oracle": oracle_shape - metric("oracle_commit_only", alpha, "rank_improvement_shape"),
            "pass_lite": bool(retrieved_shape >= retrieved_commit),
            "pass_strong": bool((retrieved_shape > retrieved_commit) and (oracle_shape >= metric("oracle_commit_only", alpha, "rank_improvement_shape"))),
            "note": "Shape component should explain more trajectory-shape gain than commit-only component."
        })
        checks.append({
            "check": f"full_memory_alpha_{alpha}",
            "value": retrieved_full,
            "baseline": shuffle_full,
            "oracle": oracle_full,
            "pass_lite": bool(retrieved_full > 0),
            "pass_strong": bool(retrieved_full > shuffle_full and oracle_full >= retrieved_full),
            "note": "Full retrieved MemoryUnit should beat shuffle; oracle should upper-bound retrieved."
        })

    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(OUT_DIR / "sem5c3_pass_checks.csv", index=False, encoding="utf-8-sig")

    verdict = "PASS-Strong" if bool(checks_df["pass_strong"].all()) else ("PASS-Lite" if bool(checks_df["pass_lite"].any()) else "NO-PASS")
    report = {
        "experiment": "SEM-5C.3 Component-Swap Memory Control Audit",
        "verdict": verdict,
        "motivation": [
            "SEM-5C.2 showed commit-only retrieval is the best address.",
            "SEM-5C.3 tests whether shape is the trajectory-control component after commit-based addressing.",
        ],
        "best_retrieval_fixed_from_5c2": BEST_RETRIEVAL_NAME,
        "addressing_summary": weight_hits,
        "interpretation_rules": [
            "If retrieved_shape_only > shuffle_shape_only, retrieved MemoryUnit carries live trajectory-guidance signal.",
            "If retrieved_shape_only > retrieved_commit_only on shape-rank/margin, mu_shape is control/guidance and mu_commit is address/identity.",
            "If oracle_full >= retrieved_full > shuffle_full, component-swapped memory intervention supports MemoryUnit -> BetterTrajectory beyond addressing.",
        ],
        "outputs": {
            "addressing": str(OUT_DIR / "sem5c3_addressing_results.csv"),
            "injection": str(OUT_DIR / "sem5c3_component_injection_results.csv"),
            "summary": str(OUT_DIR / "sem5c3_component_summary.csv"),
            "checks": str(OUT_DIR / "sem5c3_pass_checks.csv"),
        },
    }
    with open(OUT_DIR / "sem5c3_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== SEM-5C.3 SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== PASS CHECKS ===")
    print(checks_df.to_string(index=False))
    print(f"\nVerdict: {verdict}")
    print(f"Outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    run()
