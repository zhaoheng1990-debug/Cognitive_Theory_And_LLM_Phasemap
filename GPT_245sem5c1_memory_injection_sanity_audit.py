# -*- coding: utf-8 -*-
"""
SEM-5C.1 v2: Label-Free Memory Injection Sanity Audit

Why v2:
    sem5a2_dataset.csv may only contain:
        ['row_id', 'row_type', 'seed_family', 'concept_star', 'concept_phrase',
         'concept_description', 'operator_star', 'operator_id',
         'operator_phrase', 'surface_id', 'prompt']

    Therefore there may be no clean/conflict answer columns.
    This version does NOT require clean/conflict labels.

Goal:
    Verify whether retrieved MemoryUnit geometry can causally change live trajectory.

Core comparison:
    raw
    retrieved_memory_injection
    shuffle_memory_injection
    oracle_memory_injection

Main label-free metrics:
    shape_sim_to_oracle
    commit_sim_to_oracle
    family_margin_shape  = sim(to oracle family) - max sim(to non-oracle family)
    family_rank_shape    = rank of oracle family among all family memories
    retrieval_hit        = retrieved family == oracle family

Interpretation:
    If retrieved/oracle injection increases:
        shape_sim_to_oracle
        commit_sim_to_oracle
        family_margin_shape
        and improves family_rank_shape,
    then MemoryUnit has live trajectory-guiding effect.

Default model:
    D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main

Inputs:
    sem5a2_outputs/sem5a2_dataset.csv

Optional:
    sem5a2_outputs/sem5a2_features.csv
    sem4e_outputs/memory_units.npz

Important:
    If SEM-4E hidden-dim memory units are absent, this script builds a live
    MemoryBank from raw hidden trajectories grouped by seed_family.
"""

import os
import json
import math
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
FEATURES_CSV = r"sem5a2_outputs\sem5a2_features.csv"

MEMORY_UNITS_CSV = r"sem4e_outputs\memory_units.csv"
MEMORY_UNITS_NPZ = r"sem4e_outputs\memory_units.npz"

OUT_DIR = Path("sem5c1_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Qwen2.5-1.5B-Instruct usually has 28 layers indexed 0..27
SHAPE_LAYERS = list(range(7, 20))      # L7-L19
COMMIT_LAYERS = list(range(23, 26))    # L23-L25

# injection strength sweep
ALPHAS = [0.05, 0.10, 0.20, 0.30]

# set a number for quick smoke test, e.g. 24
MAX_ROWS = None

MAX_LENGTH = 512


# =========================
# Utilities
# =========================

def safe_read_csv(path: str, required: bool = True) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        if required:
            raise FileNotFoundError(f"Missing file: {path}")
        return None
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
        raise ValueError(
            "Cannot find seed family column. Expected one of: "
            "seed_family / family_id / SeedFamily / family."
        )
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
    a = normalize(a)
    b = normalize(b)
    return float(np.dot(a, b))


def mean_pool_layers(layer_vecs: Dict[int, np.ndarray], layers: List[int]) -> np.ndarray:
    xs = [layer_vecs[l] for l in layers if l in layer_vecs]
    if not xs:
        raise ValueError(f"No requested layers found: {layers}")
    return np.mean(np.stack(xs, axis=0), axis=0).astype(np.float32)


def layer_hidden_dict(outputs) -> Dict[int, torch.Tensor]:
    """
    transformers hidden_states:
      hidden_states[0] = embedding output
      hidden_states[i+1] = layer i output
    Return layer-indexed dict 0..n_layers-1.
    """
    hs = outputs.hidden_states
    d = {}
    for layer_idx in range(len(hs) - 1):
        d[layer_idx] = hs[layer_idx + 1]
    return d


# =========================
# Model hook
# =========================

class ShapeInjector:
    """
    Add memory vector to target layer outputs at last token position.

        h_l' = h_l + alpha * memory_shape_vector

    This is intentionally minimal and attribution-clean.
    """

    def __init__(self, model, target_layers: List[int], memory_vec: torch.Tensor, alpha: float):
        self.model = model
        self.target_layers = set(target_layers)
        self.memory_vec = memory_vec
        self.alpha = alpha
        self.handles = []

    def _hook(self, layer_idx):
        def fn(module, inputs, output):
            if isinstance(output, tuple):
                h = output[0]
                rest = output[1:]
                h2 = h.clone()
                h2[:, -1, :] = h2[:, -1, :] + self.alpha * self.memory_vec.to(h2.device, h2.dtype)
                return (h2,) + rest
            else:
                h = output
                h2 = h.clone()
                h2[:, -1, :] = h2[:, -1, :] + self.alpha * self.memory_vec.to(h2.device, h2.dtype)
                return h2
        return fn

    def __enter__(self):
        layers = self.model.model.layers
        for idx in self.target_layers:
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
def forward_collect(model, tokenizer, prompt: str, max_length: int = MAX_LENGTH) -> Dict[int, np.ndarray]:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(DEVICE)
    outputs = model(**enc, output_hidden_states=True, use_cache=False)
    hdict = layer_hidden_dict(outputs)

    hidden_by_layer = {}
    for l, h in hdict.items():
        hv = h[:, -1, :].detach()[0].float().cpu().numpy()
        hidden_by_layer[l] = hv.astype(np.float32)

    return hidden_by_layer


def extract_shape_commit(hidden_by_layer: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    shape = mean_pool_layers(hidden_by_layer, SHAPE_LAYERS)
    commit = mean_pool_layers(hidden_by_layer, COMMIT_LAYERS)
    return shape, commit


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
        print("[MemoryBank] SEM-4E CSV found but family column missing; skip SEM-4E memory.")
        return None

    arr = np.load(npz_p)
    shape_key = None
    for k in ["mu_shape", "mu_F_shape", "shape", "memory_shape"]:
        if k in arr:
            shape_key = k
            break

    commit_key = None
    for k in ["mu_commit", "mu_F_commit", "commit", "memory_commit"]:
        if k in arr:
            commit_key = k
            break

    if shape_key is None:
        print(f"[MemoryBank] SEM-4E NPZ has no shape key. Keys={list(arr.keys())}; skip.")
        return None

    shapes = arr[shape_key]
    if shapes.shape[-1] != hidden_dim:
        print(
            f"[MemoryBank] SEM-4E shape dim={shapes.shape[-1]} != hidden_dim={hidden_dim}; "
            "skip SEM-4E memory and build live hidden MemoryBank."
        )
        return None

    commits = None
    if commit_key is not None:
        commits = arr[commit_key]
        if commits.shape[-1] != hidden_dim:
            commits = None

    bank = {}
    for i, fam in enumerate(meta[fam_col].astype(str).tolist()):
        shape = normalize(shapes[i].astype(np.float32))
        commit = normalize(commits[i].astype(np.float32)) if commits is not None else shape.copy()
        bank[str(fam)] = {"shape": shape, "commit": commit}

    print(f"[MemoryBank] Loaded SEM-4E hidden-dim memory units: {len(bank)} families")
    return bank


def build_live_memory_bank(
    raw_records: List[Dict],
    exclude_row_idx: Optional[int] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Build family centers from raw hidden trajectories.

    For row-level oracle evaluation, exclude current row to avoid self leakage.
    If a family only has one sample and it is excluded, fallback to all samples.
    """
    by_family: Dict[str, List[Dict]] = {}
    for rec in raw_records:
        if exclude_row_idx is not None and rec["row_idx"] == exclude_row_idx:
            continue
        by_family.setdefault(str(rec["family"]), []).append(rec)

    # fallback families that became empty
    if exclude_row_idx is not None:
        all_by_family: Dict[str, List[Dict]] = {}
        for rec in raw_records:
            all_by_family.setdefault(str(rec["family"]), []).append(rec)
        for fam, xs in all_by_family.items():
            if fam not in by_family or len(by_family[fam]) == 0:
                by_family[fam] = xs

    bank = {}
    for fam, xs in by_family.items():
        shape = np.mean(np.stack([r["shape_raw"] for r in xs], axis=0), axis=0).astype(np.float32)
        commit = np.mean(np.stack([r["commit_raw"] for r in xs], axis=0), axis=0).astype(np.float32)
        bank[fam] = {"shape": normalize(shape), "commit": normalize(commit)}

    return bank


def retrieve_memory(
    query_vec: np.ndarray,
    memory_bank: Dict[str, Dict[str, np.ndarray]],
    oracle_family: Optional[str] = None,
    mode: str = "retrieved"
) -> Tuple[str, Dict[str, np.ndarray], float]:
    fams = list(memory_bank.keys())

    if mode == "oracle":
        if oracle_family is None or oracle_family not in memory_bank:
            raise ValueError(f"Oracle family missing from MemoryBank: {oracle_family}")
        return oracle_family, memory_bank[oracle_family], 1.0

    if mode == "shuffle":
        fam = random.choice(fams)
        if oracle_family is not None and len(fams) > 1:
            tries = 0
            while fam == oracle_family and tries < 50:
                fam = random.choice(fams)
                tries += 1
        sim = cosine(query_vec, memory_bank[fam]["shape"])
        return fam, memory_bank[fam], sim

    best_fam, best_sim = None, -1e9
    for fam, mem in memory_bank.items():
        sim = cosine(query_vec, mem["shape"])
        if sim > best_sim:
            best_sim = sim
            best_fam = fam
    return best_fam, memory_bank[best_fam], best_sim


def family_scores(vec: np.ndarray, memory_bank: Dict[str, Dict[str, np.ndarray]], key: str = "shape") -> Dict[str, float]:
    return {fam: cosine(vec, mem[key]) for fam, mem in memory_bank.items()}


def rank_of_family(scores: Dict[str, float], oracle_family: str) -> int:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    for i, (fam, _) in enumerate(ordered, start=1):
        if fam == oracle_family:
            return i
    return len(ordered) + 1


def margin_to_oracle(scores: Dict[str, float], oracle_family: str) -> float:
    oracle = scores.get(oracle_family, float("nan"))
    others = [v for fam, v in scores.items() if fam != oracle_family]
    if not others:
        return float("nan")
    return float(oracle - max(others))


# =========================
# Main
# =========================

def run():
    print("=" * 90)
    print("SEM-5C.1 v2 Label-Free Memory Injection Sanity Audit")
    print("=" * 90)
    print(f"DEVICE={DEVICE}, DTYPE={DTYPE}")
    print(f"MODEL_PATH={MODEL_PATH}")

    dataset = safe_read_csv(DATASET_CSV, required=True)
    if MAX_ROWS is not None:
        dataset = dataset.head(MAX_ROWS).copy()

    prompt_col = get_prompt_col(dataset)
    family_col = get_family_col(dataset)
    row_id_col = get_row_id_col(dataset)

    print(f"[Data] n={len(dataset)}")
    print(f"[Data] prompt_col={prompt_col}, family_col={family_col}, row_id_col={row_id_col}")
    print(f"[Data] columns={list(dataset.columns)}")

    print("[Model] Loading tokenizer/model...")
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
    n_layers = int(model.config.num_hidden_layers)
    print(f"[Model] hidden_dim={hidden_dim}, n_layers={n_layers}")

    # 1) Raw pass for all rows.
    print("[Stage 1] Collecting raw trajectories...")
    raw_records = []
    for pos, (idx, row) in enumerate(dataset.iterrows()):
        prompt = str(row[prompt_col])
        fam = str(row[family_col])
        row_id = row[row_id_col] if row_id_col is not None else idx

        h = forward_collect(model, tokenizer, prompt)
        shape, commit = extract_shape_commit(h)

        raw_records.append({
            "row_idx": int(idx),
            "row_pos": int(pos),
            "row_id": row_id,
            "family": fam,
            "prompt": prompt,
            "shape_raw": normalize(shape),
            "commit_raw": normalize(commit),
        })

        if (pos + 1) % 10 == 0 or (pos + 1) == len(dataset):
            print(f"[Raw] {pos+1}/{len(dataset)}")

    # Try external SEM-4E memory. If unavailable, live bank will be built leave-one-out per row.
    external_bank = load_sem4e_memory_if_available(hidden_dim)

    rows = []

    print("[Stage 2] Running memory injection conditions...")
    for rec_i, rec in enumerate(raw_records):
        oracle_family = str(rec["family"])

        if external_bank is not None and oracle_family in external_bank:
            bank = external_bank
        else:
            # leave-one-out live memory bank to avoid exact self leakage
            bank = build_live_memory_bank(raw_records, exclude_row_idx=rec["row_idx"])

        if oracle_family not in bank:
            # should not happen with fallback, but skip cleanly
            print(f"[WARN] oracle family {oracle_family} not in bank for row {rec['row_idx']}; skipped.")
            continue

        raw_shape = rec["shape_raw"]
        raw_commit = rec["commit_raw"]

        raw_shape_scores = family_scores(raw_shape, bank, key="shape")
        raw_commit_scores = family_scores(raw_commit, bank, key="commit")

        raw_shape_sim_oracle = raw_shape_scores[oracle_family]
        raw_commit_sim_oracle = raw_commit_scores[oracle_family]
        raw_margin_shape = margin_to_oracle(raw_shape_scores, oracle_family)
        raw_margin_commit = margin_to_oracle(raw_commit_scores, oracle_family)
        raw_rank_shape = rank_of_family(raw_shape_scores, oracle_family)
        raw_rank_commit = rank_of_family(raw_commit_scores, oracle_family)

        # retrieve memories using raw shape query
        conds = [("raw", None, None, None)]
        for mode in ["retrieved", "shuffle", "oracle"]:
            fam, mem, sim = retrieve_memory(raw_shape, bank, oracle_family=oracle_family, mode=mode)
            conds.append((mode, fam, mem, sim))

        for cond, mem_fam, mem, retrieval_sim in conds:
            alphas = [0.0] if cond == "raw" else ALPHAS

            for alpha in alphas:
                if cond == "raw":
                    out_shape = raw_shape
                    out_commit = raw_commit
                else:
                    prompt = rec["prompt"]
                    mem_t = torch.tensor(mem["shape"], dtype=torch.float32)
                    with ShapeInjector(model, SHAPE_LAYERS, mem_t, alpha):
                        h2 = forward_collect(model, tokenizer, prompt)
                    out_shape, out_commit = extract_shape_commit(h2)
                    out_shape = normalize(out_shape)
                    out_commit = normalize(out_commit)

                shape_scores = family_scores(out_shape, bank, key="shape")
                commit_scores = family_scores(out_commit, bank, key="commit")

                shape_sim_oracle = shape_scores[oracle_family]
                commit_sim_oracle = commit_scores[oracle_family]
                margin_shape = margin_to_oracle(shape_scores, oracle_family)
                margin_commit = margin_to_oracle(commit_scores, oracle_family)
                rank_shape = rank_of_family(shape_scores, oracle_family)
                rank_commit = rank_of_family(commit_scores, oracle_family)

                rows.append({
                    "row_idx": rec["row_idx"],
                    "row_pos": rec["row_pos"],
                    "row_id": rec["row_id"],
                    "family": oracle_family,
                    "condition": cond,
                    "alpha": alpha,
                    "memory_family": mem_fam,
                    "retrieval_sim": retrieval_sim,
                    "retrieval_hit": None if cond == "raw" else int(mem_fam == oracle_family),

                    "shape_sim_to_oracle": shape_sim_oracle,
                    "commit_sim_to_oracle": commit_sim_oracle,
                    "family_margin_shape": margin_shape,
                    "family_margin_commit": margin_commit,
                    "family_rank_shape": rank_shape,
                    "family_rank_commit": rank_commit,

                    "raw_shape_sim_to_oracle": raw_shape_sim_oracle,
                    "raw_commit_sim_to_oracle": raw_commit_sim_oracle,
                    "raw_family_margin_shape": raw_margin_shape,
                    "raw_family_margin_commit": raw_margin_commit,
                    "raw_family_rank_shape": raw_rank_shape,
                    "raw_family_rank_commit": raw_rank_commit,

                    "gain_shape_sim": shape_sim_oracle - raw_shape_sim_oracle,
                    "gain_commit_sim": commit_sim_oracle - raw_commit_sim_oracle,
                    "gain_margin_shape": margin_shape - raw_margin_shape,
                    "gain_margin_commit": margin_commit - raw_margin_commit,
                    "rank_improvement_shape": raw_rank_shape - rank_shape,
                    "rank_improvement_commit": raw_rank_commit - rank_commit,
                })

        if (rec_i + 1) % 10 == 0 or (rec_i + 1) == len(raw_records):
            print(f"[Inject] {rec_i+1}/{len(raw_records)}")

    out = pd.DataFrame(rows)
    out_csv = OUT_DIR / "sem5c1_v2_label_free_injection_results.csv"
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    summary = (
        out.groupby(["condition", "alpha"], dropna=False)
        .agg(
            n=("row_idx", "count"),
            retrieval_hit_rate=("retrieval_hit", "mean"),
            mean_shape_sim=("shape_sim_to_oracle", "mean"),
            mean_commit_sim=("commit_sim_to_oracle", "mean"),
            mean_margin_shape=("family_margin_shape", "mean"),
            mean_margin_commit=("family_margin_commit", "mean"),
            mean_rank_shape=("family_rank_shape", "mean"),
            mean_rank_commit=("family_rank_commit", "mean"),
            mean_gain_shape_sim=("gain_shape_sim", "mean"),
            mean_gain_commit_sim=("gain_commit_sim", "mean"),
            mean_gain_margin_shape=("gain_margin_shape", "mean"),
            mean_gain_margin_commit=("gain_margin_commit", "mean"),
            mean_rank_improvement_shape=("rank_improvement_shape", "mean"),
            mean_rank_improvement_commit=("rank_improvement_commit", "mean"),
        )
        .reset_index()
    )

    summary_csv = OUT_DIR / "sem5c1_v2_label_free_summary.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    checks = []
    for alpha in ALPHAS:
        sub = summary[summary["alpha"] == alpha]

        def val(cond, col):
            v = sub[sub["condition"] == cond][col]
            return float(v.iloc[0]) if len(v) else float("nan")

        retrieved_gain = val("retrieved", "mean_gain_margin_shape")
        shuffle_gain = val("shuffle", "mean_gain_margin_shape")
        oracle_gain = val("oracle", "mean_gain_margin_shape")

        retrieved_rank = val("retrieved", "mean_rank_improvement_shape")
        shuffle_rank = val("shuffle", "mean_rank_improvement_shape")
        oracle_rank = val("oracle", "mean_rank_improvement_shape")

        checks.append({
            "alpha": alpha,
            "retrieved_gain_margin_shape": retrieved_gain,
            "shuffle_gain_margin_shape": shuffle_gain,
            "oracle_gain_margin_shape": oracle_gain,
            "retrieved_rank_improvement_shape": retrieved_rank,
            "shuffle_rank_improvement_shape": shuffle_rank,
            "oracle_rank_improvement_shape": oracle_rank,
            "pass_lite_margin": bool(retrieved_gain > 0),
            "pass_lite_rank": bool(retrieved_rank > 0),
            "pass_strong_margin": bool((retrieved_gain > shuffle_gain) and (oracle_gain >= retrieved_gain)),
            "pass_strong_rank": bool((retrieved_rank > shuffle_rank) and (oracle_rank >= retrieved_rank)),
        })

    checks_df = pd.DataFrame(checks)
    checks_csv = OUT_DIR / "sem5c1_v2_label_free_pass_checks.csv"
    checks_df.to_csv(checks_csv, index=False, encoding="utf-8-sig")

    verdict = "FAIL"
    if checks_df["pass_lite_margin"].any() or checks_df["pass_lite_rank"].any():
        verdict = "PASS-Lite"
    if checks_df["pass_strong_margin"].any() or checks_df["pass_strong_rank"].any():
        verdict = "PASS-Strong-Candidate"

    report = {
        "experiment": "SEM-5C.1 v2 Label-Free Memory Injection Sanity Audit",
        "verdict": verdict,
        "mode": "label_free_trajectory_geometry",
        "why_v2": "dataset has no clean/conflict answer columns; use family geometry metrics instead.",
        "main_metrics": [
            "shape_sim_to_oracle",
            "commit_sim_to_oracle",
            "family_margin_shape",
            "family_rank_shape",
            "gain_margin_shape",
            "rank_improvement_shape",
        ],
        "pass_logic": [
            "PASS-Lite: retrieved injection improves oracle-family margin/rank over raw.",
            "PASS-Strong-Candidate: retrieved > shuffle and oracle >= retrieved."
        ],
        "outputs": {
            "results": str(out_csv),
            "summary": str(summary_csv),
            "checks": str(checks_csv),
        },
        "notes": [
            "This version does not require answer labels.",
            "If SEM-4E hidden-dim memory units are absent, a leave-one-out live MemoryBank is built from raw hidden trajectories.",
            "Injection currently adds μF_shape to L7-L19 last-token hidden state.",
            "The cleanest first signal is mean_gain_margin_shape and rank_improvement_shape."
        ],
    }

    report_json = OUT_DIR / "sem5c1_v2_label_free_report.json"
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("SEM-5C.1 v2 finished")
    print("=" * 90)
    print(summary)
    print("\nPASS checks:")
    print(checks_df)
    print(f"\nVerdict: {verdict}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {checks_csv}")
    print(f"Saved: {report_json}")


if __name__ == "__main__":
    run()
