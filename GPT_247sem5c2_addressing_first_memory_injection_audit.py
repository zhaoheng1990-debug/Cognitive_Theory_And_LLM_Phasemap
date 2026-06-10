# -*- coding: utf-8 -*-
"""
SEM-5C.2: Addressing-First Memory Injection Audit

Purpose
-------
SEM-5C.1 v2 showed:
    1) oracle MemoryUnit injection improves trajectory geometry;
    2) retrieved memory does not clearly beat shuffle;
    3) retrieval_hit_rate is extremely low.

Therefore SEM-5C.2 fixes the real bottleneck:
    Prompt -> (qC, qO) -> MemoryUnit_F

Core change vs 5C.1:
    Do NOT retrieve using raw shape query alone.
    Build a two-channel address:
        qO = shape window L7-L19      operator / trajectory-shape address
        qC = commit window L23-L25    concept / family identity address

Retrieval score:
    score(F) =
        wo * cos(qO, mu_F_shape)
      + wc * cos(qC, mu_F_commit)
      + wf * optional metadata family-prior score

Then run live injection only after retrieval is improved.

Outputs
-------
sem5c2_outputs/
    sem5c2_addressing_results.csv
    sem5c2_injection_results.csv
    sem5c2_summary.csv
    sem5c2_pass_checks.csv
    sem5c2_report.json

Default inputs
--------------
sem5a2_outputs/sem5a2_dataset.csv

Optional external SEM-4E memory:
sem4e_outputs/memory_units.csv
sem4e_outputs/memory_units.npz

If external MemoryUnit is absent or dimension-incompatible,
the script builds a leave-one-out live MemoryBank from raw trajectories.

Important
---------
This is still label-free.
It does not require clean/conflict answer columns.
"""

import os
import json
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

MEMORY_UNITS_CSV = r"sem4e_outputs\memory_units.csv"
MEMORY_UNITS_NPZ = r"sem4e_outputs\memory_units.npz"

OUT_DIR = Path("sem5c2_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SHAPE_LAYERS = list(range(7, 20))      # qO / mu_shape
COMMIT_LAYERS = list(range(23, 26))    # qC / mu_commit

MAX_LENGTH = 512
MAX_ROWS = None

# Retrieval weight sweep.
# 5C.1 effectively used (shape=1, commit=0).
RETRIEVAL_WEIGHTS = [
    {"name": "shape_only",       "wo": 1.00, "wc": 0.00},
    {"name": "commit_only",      "wo": 0.00, "wc": 1.00},
    {"name": "shape_commit_50",  "wo": 0.50, "wc": 0.50},
    {"name": "shape_commit_70",  "wo": 0.70, "wc": 0.30},
    {"name": "shape_commit_30",  "wo": 0.30, "wc": 0.70},
]

# Injection sweep. Use smaller set first to avoid huge runtime.
ALPHAS = [0.10, 0.20, 0.30]

# Top-N retrieval diagnostic.
TOPN = [1, 3, 5, 10]


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
        raise ValueError(
            "Cannot find seed family column. Expected seed_family/family_id/family."
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
    return float(np.dot(normalize(a), normalize(b)))


def mean_pool_layers(layer_vecs: Dict[int, np.ndarray], layers: List[int]) -> np.ndarray:
    xs = [layer_vecs[l] for l in layers if l in layer_vecs]
    if not xs:
        raise ValueError(f"No requested layers found: {layers}")
    return np.mean(np.stack(xs, axis=0), axis=0).astype(np.float32)


def layer_hidden_dict(outputs) -> Dict[int, torch.Tensor]:
    hs = outputs.hidden_states
    d = {}
    for layer_idx in range(len(hs) - 1):
        d[layer_idx] = hs[layer_idx + 1]
    return d


# =========================
# Hook
# =========================

class ShapeCommitInjector:
    """
    Inject memory shape into shape layers and optionally memory commit into commit layers.

    h_l' = h_l + alpha_shape  * mu_shape   for L7-L19
    h_l' = h_l + alpha_commit * mu_commit  for L23-L25

    For SEM-5C.2 default:
      alpha_shape = alpha
      alpha_commit = 0
    because we first test trajectory guidance.
    """

    def __init__(
        self,
        model,
        shape_layers: List[int],
        commit_layers: List[int],
        mu_shape: torch.Tensor,
        mu_commit: Optional[torch.Tensor],
        alpha_shape: float,
        alpha_commit: float = 0.0,
    ):
        self.model = model
        self.shape_layers = set(shape_layers)
        self.commit_layers = set(commit_layers)
        self.mu_shape = mu_shape
        self.mu_commit = mu_commit
        self.alpha_shape = alpha_shape
        self.alpha_commit = alpha_commit
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

            if layer_idx in self.shape_layers and self.alpha_shape != 0:
                v = self.mu_shape.to(h2.device, h2.dtype)
                h2[:, -1, :] = h2[:, -1, :] + self.alpha_shape * v

            if (
                layer_idx in self.commit_layers
                and self.alpha_commit != 0
                and self.mu_commit is not None
            ):
                v = self.mu_commit.to(h2.device, h2.dtype)
                h2[:, -1, :] = h2[:, -1, :] + self.alpha_commit * v

            if rest is not None:
                return (h2,) + rest
            return h2

        return fn

    def __enter__(self):
        layers = self.model.model.layers
        target_layers = self.shape_layers.union(self.commit_layers)
        for idx in target_layers:
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

    hidden_by_layer = {}
    for l, h in hdict.items():
        hidden_by_layer[l] = h[:, -1, :].detach()[0].float().cpu().numpy().astype(np.float32)
    return hidden_by_layer


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
        print("[MemoryBank] SEM-4E CSV found but family column missing; skip.")
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
            "skip external memory."
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

    print(f"[MemoryBank] Loaded SEM-4E memory: {len(bank)} families")
    return bank


def build_live_memory_bank(raw_records: List[Dict], exclude_row_idx: Optional[int] = None):
    by_family = {}
    for rec in raw_records:
        if exclude_row_idx is not None and rec["row_idx"] == exclude_row_idx:
            continue
        by_family.setdefault(str(rec["family"]), []).append(rec)

    # fallback if excluded family becomes empty
    if exclude_row_idx is not None:
        all_by_family = {}
        for rec in raw_records:
            all_by_family.setdefault(str(rec["family"]), []).append(rec)
        for fam, xs in all_by_family.items():
            if fam not in by_family or len(by_family[fam]) == 0:
                by_family[fam] = xs

    bank = {}
    for fam, xs in by_family.items():
        shape = np.mean(np.stack([r["shape_raw"] for r in xs], axis=0), axis=0)
        commit = np.mean(np.stack([r["commit_raw"] for r in xs], axis=0), axis=0)
        bank[fam] = {"shape": normalize(shape), "commit": normalize(commit)}
    return bank


# =========================
# Retrieval
# =========================

def score_memory(q_shape, q_commit, mem, wo: float, wc: float) -> float:
    return float(wo * cosine(q_shape, mem["shape"]) + wc * cosine(q_commit, mem["commit"]))


def ranked_retrieve(q_shape, q_commit, bank, wo: float, wc: float):
    rows = []
    for fam, mem in bank.items():
        s_shape = cosine(q_shape, mem["shape"])
        s_commit = cosine(q_commit, mem["commit"])
        s = wo * s_shape + wc * s_commit
        rows.append((fam, mem, float(s), float(s_shape), float(s_commit)))
    rows.sort(key=lambda x: x[2], reverse=True)
    return rows


def family_scores(vec: np.ndarray, bank: Dict[str, Dict[str, np.ndarray]], key: str = "shape"):
    return {fam: cosine(vec, mem[key]) for fam, mem in bank.items()}


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


# =========================
# Main
# =========================

def run():
    print("=" * 90)
    print("SEM-5C.2 Addressing-First Memory Injection Audit")
    print("=" * 90)
    print(f"DEVICE={DEVICE}, DTYPE={DTYPE}")
    print(f"MODEL_PATH={MODEL_PATH}")

    dataset = safe_read_csv(DATASET_CSV)
    if MAX_ROWS is not None:
        dataset = dataset.head(MAX_ROWS).copy()

    prompt_col = get_prompt_col(dataset)
    family_col = get_family_col(dataset)
    row_id_col = get_row_id_col(dataset)

    print(f"[Data] n={len(dataset)}")
    print(f"[Data] prompt_col={prompt_col}, family_col={family_col}, row_id_col={row_id_col}")

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

    # Stage 1: raw trajectories
    print("[Stage 1] Collect raw shape/commit queries...")
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
            "shape_raw": shape,
            "commit_raw": commit,
        })
        if (pos + 1) % 10 == 0 or (pos + 1) == len(dataset):
            print(f"[Raw] {pos+1}/{len(dataset)}")

    external_bank = load_sem4e_memory_if_available(hidden_dim)

    # Stage 2: addressing diagnostic
    print("[Stage 2] Addressing diagnostic...")
    addr_rows = []
    best_weight_by_hit = None
    best_hit = -1.0

    for weight in RETRIEVAL_WEIGHTS:
        hits = []
        topn_hits = {n: [] for n in TOPN}

        for rec in raw_records:
            oracle_family = str(rec["family"])
            if external_bank is not None and oracle_family in external_bank:
                bank = external_bank
            else:
                bank = build_live_memory_bank(raw_records, exclude_row_idx=rec["row_idx"])

            ranked = ranked_retrieve(
                rec["shape_raw"],
                rec["commit_raw"],
                bank,
                wo=weight["wo"],
                wc=weight["wc"],
            )

            fams = [x[0] for x in ranked]
            hit1 = int(fams[0] == oracle_family)
            hits.append(hit1)

            for n in TOPN:
                topn_hits[n].append(int(oracle_family in fams[:n]))

            oracle_rank = fams.index(oracle_family) + 1 if oracle_family in fams else len(fams) + 1
            best = ranked[0]

            addr_rows.append({
                "row_idx": rec["row_idx"],
                "row_id": rec["row_id"],
                "family": oracle_family,
                "retrieval_weight": weight["name"],
                "wo": weight["wo"],
                "wc": weight["wc"],
                "top1_family": best[0],
                "top1_hit": hit1,
                "oracle_rank": oracle_rank,
                "top1_score": best[2],
                "top1_shape_sim": best[3],
                "top1_commit_sim": best[4],
            })

        hit_rate = float(np.mean(hits))
        if hit_rate > best_hit:
            best_hit = hit_rate
            best_weight_by_hit = weight

        print(f"[Address] {weight['name']}: top1_hit={hit_rate:.4f} " +
              " ".join([f"top{n}={np.mean(topn_hits[n]):.4f}" for n in TOPN]))

    addr_df = pd.DataFrame(addr_rows)
    addr_csv = OUT_DIR / "sem5c2_addressing_results.csv"
    addr_df.to_csv(addr_csv, index=False, encoding="utf-8-sig")

    if best_weight_by_hit is None:
        best_weight_by_hit = RETRIEVAL_WEIGHTS[0]

    print(f"[Address] Best weight by top1 hit: {best_weight_by_hit}")

    # Stage 3: live injection using best retrieval, plus oracle/shuffle controls
    print("[Stage 3] Live injection with best address...")
    inj_rows = []
    wo = best_weight_by_hit["wo"]
    wc = best_weight_by_hit["wc"]
    wname = best_weight_by_hit["name"]

    for rec_i, rec in enumerate(raw_records):
        oracle_family = str(rec["family"])

        if external_bank is not None and oracle_family in external_bank:
            bank = external_bank
        else:
            bank = build_live_memory_bank(raw_records, exclude_row_idx=rec["row_idx"])

        if oracle_family not in bank:
            print(f"[WARN] oracle family missing: {oracle_family}; skip row {rec['row_idx']}")
            continue

        ranked = ranked_retrieve(rec["shape_raw"], rec["commit_raw"], bank, wo=wo, wc=wc)
        retrieved_family, retrieved_mem = ranked[0][0], ranked[0][1]

        # choose shuffle family != oracle if possible
        fams = list(bank.keys())
        shuffle_family = random.choice(fams)
        if len(fams) > 1:
            tries = 0
            while shuffle_family == oracle_family and tries < 100:
                shuffle_family = random.choice(fams)
                tries += 1
        shuffle_mem = bank[shuffle_family]
        oracle_mem = bank[oracle_family]

        conditions = [
            ("raw", None, None),
            ("retrieved", retrieved_family, retrieved_mem),
            ("shuffle", shuffle_family, shuffle_mem),
            ("oracle", oracle_family, oracle_mem),
        ]

        raw_shape = rec["shape_raw"]
        raw_commit = rec["commit_raw"]

        raw_shape_scores = family_scores(raw_shape, bank, key="shape")
        raw_commit_scores = family_scores(raw_commit, bank, key="commit")
        raw_shape_sim = raw_shape_scores[oracle_family]
        raw_commit_sim = raw_commit_scores[oracle_family]
        raw_margin_shape = margin_to_oracle(raw_shape_scores, oracle_family)
        raw_margin_commit = margin_to_oracle(raw_commit_scores, oracle_family)
        raw_rank_shape = rank_of_family(raw_shape_scores, oracle_family)
        raw_rank_commit = rank_of_family(raw_commit_scores, oracle_family)

        for cond, mem_fam, mem in conditions:
            alphas = [0.0] if cond == "raw" else ALPHAS

            for alpha in alphas:
                if cond == "raw":
                    out_shape = raw_shape
                    out_commit = raw_commit
                else:
                    mu_shape = torch.tensor(mem["shape"], dtype=torch.float32)
                    mu_commit = torch.tensor(mem["commit"], dtype=torch.float32)
                    with ShapeCommitInjector(
                        model=model,
                        shape_layers=SHAPE_LAYERS,
                        commit_layers=COMMIT_LAYERS,
                        mu_shape=mu_shape,
                        mu_commit=mu_commit,
                        alpha_shape=alpha,
                        alpha_commit=0.0,
                    ):
                        h2 = forward_collect(model, tokenizer, rec["prompt"])
                    out_shape, out_commit = extract_shape_commit(h2)

                shape_scores = family_scores(out_shape, bank, key="shape")
                commit_scores = family_scores(out_commit, bank, key="commit")

                shape_sim = shape_scores[oracle_family]
                commit_sim = commit_scores[oracle_family]
                margin_shape = margin_to_oracle(shape_scores, oracle_family)
                margin_commit = margin_to_oracle(commit_scores, oracle_family)
                rank_shape = rank_of_family(shape_scores, oracle_family)
                rank_commit = rank_of_family(commit_scores, oracle_family)

                inj_rows.append({
                    "row_idx": rec["row_idx"],
                    "row_id": rec["row_id"],
                    "family": oracle_family,
                    "retrieval_weight": wname,
                    "condition": cond,
                    "alpha": alpha,
                    "memory_family": mem_fam,
                    "retrieval_hit": None if cond == "raw" else int(mem_fam == oracle_family),

                    "shape_sim_to_oracle": shape_sim,
                    "commit_sim_to_oracle": commit_sim,
                    "family_margin_shape": margin_shape,
                    "family_margin_commit": margin_commit,
                    "family_rank_shape": rank_shape,
                    "family_rank_commit": rank_commit,

                    "gain_shape_sim": shape_sim - raw_shape_sim,
                    "gain_commit_sim": commit_sim - raw_commit_sim,
                    "gain_margin_shape": margin_shape - raw_margin_shape,
                    "gain_margin_commit": margin_commit - raw_margin_commit,
                    "rank_improvement_shape": raw_rank_shape - rank_shape,
                    "rank_improvement_commit": raw_rank_commit - rank_commit,
                })

        if (rec_i + 1) % 10 == 0 or (rec_i + 1) == len(raw_records):
            print(f"[Inject] {rec_i+1}/{len(raw_records)}")

    inj_df = pd.DataFrame(inj_rows)
    inj_csv = OUT_DIR / "sem5c2_injection_results.csv"
    inj_df.to_csv(inj_csv, index=False, encoding="utf-8-sig")

    # Summaries
    addr_summary = (
        addr_df.groupby("retrieval_weight")
        .agg(
            n=("row_idx", "count"),
            top1_hit_rate=("top1_hit", "mean"),
            mean_oracle_rank=("oracle_rank", "mean"),
            median_oracle_rank=("oracle_rank", "median"),
        )
        .reset_index()
    )

    inj_summary = (
        inj_df.groupby(["condition", "alpha"], dropna=False)
        .agg(
            n=("row_idx", "count"),
            retrieval_hit_rate=("retrieval_hit", "mean"),
            mean_gain_shape_sim=("gain_shape_sim", "mean"),
            mean_gain_commit_sim=("gain_commit_sim", "mean"),
            mean_gain_margin_shape=("gain_margin_shape", "mean"),
            mean_gain_margin_commit=("gain_margin_commit", "mean"),
            mean_rank_improvement_shape=("rank_improvement_shape", "mean"),
            mean_rank_improvement_commit=("rank_improvement_commit", "mean"),
        )
        .reset_index()
    )

    summary_csv = OUT_DIR / "sem5c2_summary.csv"
    with open(summary_csv, "w", encoding="utf-8-sig") as f:
        f.write("# Addressing Summary\n")
        addr_summary.to_csv(f, index=False)
        f.write("\n# Injection Summary\n")
        inj_summary.to_csv(f, index=False)

    checks = []

    # Addressing pass checks
    best_addr = addr_summary.sort_values("top1_hit_rate", ascending=False).iloc[0]
    checks.append({
        "check": "addressing_top1_hit",
        "value": float(best_addr["top1_hit_rate"]),
        "pass_lite": bool(best_addr["top1_hit_rate"] >= 0.10),
        "pass_strong": bool(best_addr["top1_hit_rate"] >= 0.30),
        "note": "Top1 hit threshold is deliberately conservative; previous 5C.1 was about 0.008.",
    })

    # Injection pass checks
    for alpha in ALPHAS:
        sub = inj_summary[inj_summary["alpha"] == alpha]

        def val(cond, col):
            v = sub[sub["condition"] == cond][col]
            return float(v.iloc[0]) if len(v) else np.nan

        r_gain = val("retrieved", "mean_rank_improvement_shape")
        s_gain = val("shuffle", "mean_rank_improvement_shape")
        o_gain = val("oracle", "mean_rank_improvement_shape")

        r_margin = val("retrieved", "mean_gain_margin_shape")
        s_margin = val("shuffle", "mean_gain_margin_shape")
        o_margin = val("oracle", "mean_gain_margin_shape")

        checks.append({
            "check": f"injection_rank_alpha_{alpha}",
            "value": r_gain,
            "shuffle": s_gain,
            "oracle": o_gain,
            "pass_lite": bool(r_gain > 0),
            "pass_strong": bool((r_gain > s_gain) and (o_gain >= r_gain)),
            "note": "Retrieved should improve raw; strong requires retrieved > shuffle and oracle >= retrieved.",
        })

        checks.append({
            "check": f"injection_margin_alpha_{alpha}",
            "value": r_margin,
            "shuffle": s_margin,
            "oracle": o_margin,
            "pass_lite": bool(r_margin > 0),
            "pass_strong": bool((r_margin > s_margin) and (o_margin >= r_margin)),
            "note": "Margin check is stricter and was failed in 5C.1.",
        })

    checks_df = pd.DataFrame(checks)
    checks_csv = OUT_DIR / "sem5c2_pass_checks.csv"
    checks_df.to_csv(checks_csv, index=False, encoding="utf-8-sig")

    verdict = "FAIL"
    if checks_df["pass_lite"].any():
        verdict = "PASS-Lite"
    # Strong requires addressing strong plus at least one strong injection check.
    addr_strong = bool(checks_df[checks_df["check"] == "addressing_top1_hit"]["pass_strong"].iloc[0])
    inj_strong = bool(checks_df[checks_df["check"].str.startswith("injection_")]["pass_strong"].any())
    if addr_strong and inj_strong:
        verdict = "PASS-Strong"

    report = {
        "experiment": "SEM-5C.2 Addressing-First Memory Injection Audit",
        "verdict": verdict,
        "motivation": [
            "SEM-5C.1 showed oracle memory is useful but retrieval hit rate was extremely low.",
            "SEM-5C.2 explicitly tests qO=shape and qC=commit retrieval before injection."
        ],
        "best_retrieval_weight": dict(best_weight_by_hit),
        "outputs": {
            "addressing": str(addr_csv),
            "injection": str(inj_csv),
            "summary": str(summary_csv),
            "checks": str(checks_csv),
        },
        "interpretation_rules": [
            "If addressing improves but retrieved still does not beat shuffle, retrieval is better but injection mapping is not family-specific.",
            "If oracle beats retrieved by a large margin, MemoryUnit utility remains positive but addressing remains bottleneck.",
            "If retrieved > shuffle and oracle >= retrieved, SEM-5C live memory intervention is supported.",
        ],
    }

    report_json = OUT_DIR / "sem5c2_report.json"
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("SEM-5C.2 finished")
    print("=" * 90)
    print("\nAddressing summary:")
    print(addr_summary)
    print("\nInjection summary:")
    print(inj_summary)
    print("\nPass checks:")
    print(checks_df)
    print(f"\nVerdict: {verdict}")
    print(f"Saved: {addr_csv}")
    print(f"Saved: {inj_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {checks_csv}")
    print(f"Saved: {report_json}")


if __name__ == "__main__":
    run()
