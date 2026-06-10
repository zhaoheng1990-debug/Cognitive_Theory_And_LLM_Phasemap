# -*- coding: utf-8 -*-
"""
SEM-5C.4: Operator-Transfer Memory Control Audit

Motivation
----------
SEM-5C.2:
    commit-only retrieval is the best address.
SEM-5C.3:
    mu_shape drives trajectory-shape improvement;
    mu_commit mostly acts as identity/address;
    same_operator_wrong_concept_shape can be close to / stronger than retrieved_shape.

SEM-5C.4 asks:
    Is mu_shape primarily an operator-level transferable control prior?

Core tests
----------
For each prompt row, build memory candidates:

1. oracle_family
2. retrieved_by_commit
3. same_operator_wrong_concept
4. same_concept_wrong_operator
5. different_concept_different_operator
6. shuffle

Inject only shape vectors into L7-L19 and compare:

    same_operator_wrong_concept > same_concept_wrong_operator
    same_operator_wrong_concept > shuffle
    oracle >= same_operator_wrong_concept

If true:
    mu_shape ~= OperatorPrior / trajectory-control prior
    mu_commit ~= IdentityAddress / concept-family locator

Outputs
-------
sem5c4_outputs/
    sem5c4_addressing_results.csv
    sem5c4_operator_transfer_results.csv
    sem5c4_summary.csv
    sem5c4_pass_checks.csv
    sem5c4_report.json

Inputs
------
sem5a2_outputs/sem5a2_dataset.csv

Optional:
sem4e_outputs/memory_units.csv
sem4e_outputs/memory_units.npz

Fallback:
Build leave-one-out live MemoryBank from raw hidden trajectories.

Default model:
D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main
"""

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

OUT_DIR = Path("sem5c4_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SHAPE_LAYERS = list(range(7, 20))      # operator/trajectory control
COMMIT_LAYERS = list(range(23, 26))    # identity/address

MAX_LENGTH = 512
MAX_ROWS = None

ALPHAS = [0.10, 0.20, 0.30]

# retrieval fixed from SEM-5C.2/5C.3
RETRIEVAL_WO = 0.0
RETRIEVAL_WC = 1.0
RETRIEVAL_NAME = "commit_only"

# candidate selection strictness
REQUIRE_DIFFERENT_FAMILY = True


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
        raise ValueError("Cannot find seed family column.")
    return col


def get_concept_col(df: pd.DataFrame) -> str:
    col = find_col(df, ["concept_star", "concept", "concept_id", "concept_phrase"])
    if col is None:
        raise ValueError("Cannot find concept column. Expected concept_star/concept/concept_id/concept_phrase.")
    return col


def get_operator_col(df: pd.DataFrame) -> str:
    col = find_col(df, ["operator_star", "operator_id", "operator", "operator_phrase"])
    if col is None:
        raise ValueError("Cannot find operator column. Expected operator_star/operator_id/operator/operator_phrase.")
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


# =========================
# Hook
# =========================

class ShapeInjector:
    """Inject one shape vector into L7-L19 last-token hidden state."""

    def __init__(self, model, target_layers: List[int], mu_shape: torch.Tensor, alpha: float):
        self.model = model
        self.target_layers = set(target_layers)
        self.mu_shape = mu_shape
        self.alpha = alpha
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
            v = self.mu_shape.to(h2.device, h2.dtype)
            h2[:, -1, :] = h2[:, -1, :] + self.alpha * v

            if rest is not None:
                return (h2,) + rest
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
def forward_collect(model, tokenizer, prompt: str) -> Dict[int, np.ndarray]:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(DEVICE)
    outputs = model(**enc, output_hidden_states=True, use_cache=False)
    hdict = layer_hidden_dict(outputs)
    return {l: h[:, -1, :].detach()[0].float().cpu().numpy().astype(np.float32) for l, h in hdict.items()}


def extract_shape_commit(hidden_by_layer: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    shape = normalize(mean_pool_layers(hidden_by_layer, SHAPE_LAYERS))
    commit = normalize(mean_pool_layers(hidden_by_layer, COMMIT_LAYERS))
    return shape, commit


# =========================
# MemoryBank
# =========================

def load_sem4e_memory_if_available(hidden_dim: int, family_meta: pd.DataFrame, family_col: str, concept_col: str, operator_col: str):
    csv_p = Path(MEMORY_UNITS_CSV)
    npz_p = Path(MEMORY_UNITS_NPZ)
    if not (csv_p.exists() and npz_p.exists()):
        return None

    meta = pd.read_csv(csv_p)
    fam_col_meta = find_col(meta, ["seed_family", "family_id", "SeedFamily", "family", "Family", "memory_family"])
    if fam_col_meta is None:
        print("[MemoryBank] SEM-4E CSV found but no family column; fallback live bank.")
        return None

    arr = np.load(npz_p)
    shape_key = next((k for k in ["mu_shape", "mu_F_shape", "shape", "memory_shape"] if k in arr), None)
    commit_key = next((k for k in ["mu_commit", "mu_F_commit", "commit", "memory_commit"] if k in arr), None)
    if shape_key is None:
        print(f"[MemoryBank] SEM-4E NPZ has no shape key. Keys={list(arr.keys())}; fallback.")
        return None

    shapes = arr[shape_key]
    if shapes.shape[-1] != hidden_dim:
        print(f"[MemoryBank] external shape dim={shapes.shape[-1]} != hidden_dim={hidden_dim}; fallback.")
        return None

    commits = arr[commit_key] if commit_key is not None and arr[commit_key].shape[-1] == hidden_dim else None

    # map family -> concept/operator from dataset metadata
    meta_map = (
        family_meta[[family_col, concept_col, operator_col]]
        .drop_duplicates(subset=[family_col])
        .assign(**{family_col: lambda x: x[family_col].astype(str)})
        .set_index(family_col)
        .to_dict("index")
    )

    bank = {}
    for i, fam in enumerate(meta[fam_col_meta].astype(str).tolist()):
        if fam not in meta_map:
            # external memory may include extra family, keep but concept/operator unknown
            concept = None
            operator = None
        else:
            concept = str(meta_map[fam][concept_col])
            operator = str(meta_map[fam][operator_col])

        bank[fam] = {
            "shape": normalize(shapes[i].astype(np.float32)),
            "commit": normalize(commits[i].astype(np.float32)) if commits is not None else normalize(shapes[i].astype(np.float32)),
            "concept": concept,
            "operator": operator,
        }

    print(f"[MemoryBank] Loaded external SEM-4E memory: {len(bank)} families")
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
        shape = normalize(np.mean(np.stack([r["shape_raw"] for r in xs], axis=0), axis=0))
        commit = normalize(np.mean(np.stack([r["commit_raw"] for r in xs], axis=0), axis=0))
        # family metadata should be constant
        concept = str(xs[0]["concept"])
        operator = str(xs[0]["operator"])
        bank[fam] = {
            "shape": shape,
            "commit": commit,
            "concept": concept,
            "operator": operator,
        }
    return bank


# =========================
# Retrieval / candidate selection
# =========================

def score_retrieval(q_shape, q_commit, mem, wo=RETRIEVAL_WO, wc=RETRIEVAL_WC) -> float:
    return float(wo * cosine(q_shape, mem["shape"]) + wc * cosine(q_commit, mem["commit"]))


def ranked_retrieve(q_shape, q_commit, bank):
    rows = []
    for fam, mem in bank.items():
        s = score_retrieval(q_shape, q_commit, mem)
        rows.append((fam, mem, s, cosine(q_shape, mem["shape"]), cosine(q_commit, mem["commit"])))
    rows.sort(key=lambda x: x[2], reverse=True)
    return rows


def select_candidate(bank, oracle_family, concept, operator, kind):
    """
    Select deterministic candidate family according to kind.
    Prefer candidates with highest commit score? Candidate selection here is structural.
    The actual memory vector comes from bank.
    """
    fams = list(bank.keys())

    def ok(fam):
        mem = bank[fam]
        c = str(mem.get("concept"))
        o = str(mem.get("operator"))
        if REQUIRE_DIFFERENT_FAMILY and fam == oracle_family:
            return False
        if kind == "same_operator_wrong_concept":
            return (o == str(operator)) and (c != str(concept))
        if kind == "same_concept_wrong_operator":
            return (c == str(concept)) and (o != str(operator))
        if kind == "diff_concept_diff_operator":
            return (c != str(concept)) and (o != str(operator))
        if kind == "shuffle":
            return fam != oracle_family
        raise ValueError(f"Unknown kind: {kind}")

    candidates = [fam for fam in fams if ok(fam)]
    if not candidates:
        # fallback to any non-oracle candidate
        candidates = [fam for fam in fams if fam != oracle_family] or fams

    return random.choice(candidates)


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
    print("SEM-5C.4 Operator-Transfer Memory Control Audit")
    print("=" * 90)
    print(f"DEVICE={DEVICE}, DTYPE={DTYPE}")

    dataset = safe_read_csv(DATASET_CSV)
    if MAX_ROWS is not None:
        dataset = dataset.head(MAX_ROWS).copy()

    prompt_col = get_prompt_col(dataset)
    family_col = get_family_col(dataset)
    concept_col = get_concept_col(dataset)
    operator_col = get_operator_col(dataset)
    row_id_col = get_row_id_col(dataset)

    print(f"[Data] n={len(dataset)}")
    print(f"[Data] prompt_col={prompt_col}, family_col={family_col}, concept_col={concept_col}, operator_col={operator_col}")

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

    # Stage 1: collect raw
    print("[Stage 1] Collect raw shape/commit...")
    raw_records = []
    for pos, (idx, row) in enumerate(dataset.iterrows()):
        prompt = str(row[prompt_col])
        fam = str(row[family_col])
        concept = str(row[concept_col])
        operator = str(row[operator_col])
        row_id = row[row_id_col] if row_id_col is not None else idx

        h = forward_collect(model, tokenizer, prompt)
        shape, commit = extract_shape_commit(h)

        raw_records.append({
            "row_idx": int(idx),
            "row_pos": int(pos),
            "row_id": row_id,
            "family": fam,
            "concept": concept,
            "operator": operator,
            "prompt": prompt,
            "shape_raw": shape,
            "commit_raw": commit,
        })

        if (pos + 1) % 10 == 0 or (pos + 1) == len(dataset):
            print(f"[Raw] {pos+1}/{len(dataset)}")

    external_bank = load_sem4e_memory_if_available(
        hidden_dim=hidden_dim,
        family_meta=dataset,
        family_col=family_col,
        concept_col=concept_col,
        operator_col=operator_col,
    )

    # Stage 2: addressing diagnostic
    print("[Stage 2] Commit-only addressing diagnostic...")
    addr_rows = []
    for rec in raw_records:
        oracle_family = str(rec["family"])
        if external_bank is not None and oracle_family in external_bank:
            bank = external_bank
        else:
            bank = build_live_memory_bank(raw_records, exclude_row_idx=rec["row_idx"])

        ranked = ranked_retrieve(rec["shape_raw"], rec["commit_raw"], bank)
        fams = [r[0] for r in ranked]
        oracle_rank = fams.index(oracle_family) + 1 if oracle_family in fams else len(fams) + 1
        top1 = ranked[0]

        addr_rows.append({
            "row_idx": rec["row_idx"],
            "row_id": rec["row_id"],
            "family": oracle_family,
            "concept": rec["concept"],
            "operator": rec["operator"],
            "retrieval_weight": RETRIEVAL_NAME,
            "top1_family": top1[0],
            "top1_concept": bank[top1[0]].get("concept"),
            "top1_operator": bank[top1[0]].get("operator"),
            "top1_hit": int(top1[0] == oracle_family),
            "top1_same_concept": int(str(bank[top1[0]].get("concept")) == str(rec["concept"])),
            "top1_same_operator": int(str(bank[top1[0]].get("operator")) == str(rec["operator"])),
            "oracle_rank": oracle_rank,
            "top1_score": top1[2],
            "top1_shape_sim": top1[3],
            "top1_commit_sim": top1[4],
        })

    addr_df = pd.DataFrame(addr_rows)
    addr_csv = OUT_DIR / "sem5c4_addressing_results.csv"
    addr_df.to_csv(addr_csv, index=False, encoding="utf-8-sig")

    # Stage 3: component/operator transfer injection
    print("[Stage 3] Operator-transfer shape injection...")
    rows = []

    for rec_i, rec in enumerate(raw_records):
        oracle_family = str(rec["family"])
        concept = str(rec["concept"])
        operator = str(rec["operator"])

        if external_bank is not None and oracle_family in external_bank:
            bank = external_bank
        else:
            bank = build_live_memory_bank(raw_records, exclude_row_idx=rec["row_idx"])

        if oracle_family not in bank:
            print(f"[WARN] oracle family missing: {oracle_family}; skip")
            continue

        ranked = ranked_retrieve(rec["shape_raw"], rec["commit_raw"], bank)
        retrieved_family, retrieved_mem = ranked[0][0], ranked[0][1]

        cand_map = {
            "raw": None,
            "oracle_shape": oracle_family,
            "retrieved_shape": retrieved_family,
            "same_operator_wrong_concept_shape": select_candidate(bank, oracle_family, concept, operator, "same_operator_wrong_concept"),
            "same_concept_wrong_operator_shape": select_candidate(bank, oracle_family, concept, operator, "same_concept_wrong_operator"),
            "diff_concept_diff_operator_shape": select_candidate(bank, oracle_family, concept, operator, "diff_concept_diff_operator"),
            "shuffle_shape": select_candidate(bank, oracle_family, concept, operator, "shuffle"),
        }

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

        for cond, fam in cand_map.items():
            alphas = [0.0] if cond == "raw" else ALPHAS

            for alpha in alphas:
                if cond == "raw":
                    out_shape, out_commit = raw_shape, raw_commit
                    mem = None
                else:
                    mem = bank[fam]
                    mu_shape = torch.tensor(mem["shape"], dtype=torch.float32)
                    with ShapeInjector(model, SHAPE_LAYERS, mu_shape, alpha):
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

                rows.append({
                    "row_idx": rec["row_idx"],
                    "row_id": rec["row_id"],
                    "family": oracle_family,
                    "concept": concept,
                    "operator": operator,
                    "condition": cond,
                    "alpha": alpha,
                    "memory_family": fam,
                    "memory_concept": None if mem is None else mem.get("concept"),
                    "memory_operator": None if mem is None else mem.get("operator"),
                    "retrieved_family": retrieved_family,
                    "retrieval_hit": int(retrieved_family == oracle_family),

                    "same_concept_memory": None if mem is None else int(str(mem.get("concept")) == concept),
                    "same_operator_memory": None if mem is None else int(str(mem.get("operator")) == operator),

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

    res_df = pd.DataFrame(rows)
    res_csv = OUT_DIR / "sem5c4_operator_transfer_results.csv"
    res_df.to_csv(res_csv, index=False, encoding="utf-8-sig")

    summary = (
        res_df.groupby(["condition", "alpha"], dropna=False)
        .agg(
            n=("row_idx", "count"),
            retrieval_hit=("retrieval_hit", "mean"),
            same_concept_memory=("same_concept_memory", "mean"),
            same_operator_memory=("same_operator_memory", "mean"),
            gain_shape_sim=("gain_shape_sim", "mean"),
            gain_commit_sim=("gain_commit_sim", "mean"),
            gain_margin_shape=("gain_margin_shape", "mean"),
            gain_margin_commit=("gain_margin_commit", "mean"),
            rank_improvement_shape=("rank_improvement_shape", "mean"),
            rank_improvement_commit=("rank_improvement_commit", "mean"),
        )
        .reset_index()
    )
    summary_csv = OUT_DIR / "sem5c4_summary.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    # Pass checks
    checks = []

    addr_summary = {
        "top1_hit": float(addr_df["top1_hit"].mean()),
        "top1_same_concept": float(addr_df["top1_same_concept"].mean()),
        "top1_same_operator": float(addr_df["top1_same_operator"].mean()),
        "mean_oracle_rank": float(addr_df["oracle_rank"].mean()),
    }

    for alpha in ALPHAS:
        sub = summary[summary["alpha"] == alpha]

        def val(cond, col):
            v = sub[sub["condition"] == cond][col]
            return float(v.iloc[0]) if len(v) else np.nan

        same_op = val("same_operator_wrong_concept_shape", "rank_improvement_shape")
        same_con = val("same_concept_wrong_operator_shape", "rank_improvement_shape")
        diffdiff = val("diff_concept_diff_operator_shape", "rank_improvement_shape")
        shuffle = val("shuffle_shape", "rank_improvement_shape")
        retrieved = val("retrieved_shape", "rank_improvement_shape")
        oracle = val("oracle_shape", "rank_improvement_shape")

        checks.append({
            "check": f"operator_transfer_alpha_{alpha}",
            "value": same_op,
            "baseline_same_concept_wrong_operator": same_con,
            "baseline_shuffle": shuffle,
            "oracle": oracle,
            "pass_lite": bool(same_op > shuffle),
            "pass_strong": bool((same_op > shuffle) and (same_op > same_con) and (oracle >= same_op)),
            "note": "same-operator wrong-concept shape should beat shuffle and same-concept wrong-operator if shape is OperatorPrior.",
        })

        checks.append({
            "check": f"retrieved_vs_shuffle_alpha_{alpha}",
            "value": retrieved,
            "baseline_shuffle": shuffle,
            "oracle": oracle,
            "pass_lite": bool(retrieved > 0),
            "pass_strong": bool((retrieved > shuffle) and (oracle >= retrieved)),
            "note": "retrieved shape should beat shuffle; oracle should upper-bound retrieved.",
        })

        checks.append({
            "check": f"same_operator_vs_diffdiff_alpha_{alpha}",
            "value": same_op,
            "baseline_diff_concept_diff_operator": diffdiff,
            "oracle": oracle,
            "pass_lite": bool(same_op > diffdiff),
            "pass_strong": bool((same_op > diffdiff) and (oracle >= same_op)),
            "note": "same operator should transfer better than fully mismatched memory.",
        })

    checks_df = pd.DataFrame(checks)
    checks_csv = OUT_DIR / "sem5c4_pass_checks.csv"
    checks_df.to_csv(checks_csv, index=False, encoding="utf-8-sig")

    verdict = "FAIL"
    if checks_df["pass_lite"].any():
        verdict = "PASS-Lite"
    # Strong requires all three alpha operator_transfer strong OR two out of three + retrieved strong
    op_strong_count = checks_df[checks_df["check"].str.startswith("operator_transfer")]["pass_strong"].sum()
    ret_strong_count = checks_df[checks_df["check"].str.startswith("retrieved_vs_shuffle")]["pass_strong"].sum()
    if op_strong_count >= 2 and ret_strong_count >= 2:
        verdict = "PASS-Strong"
    elif op_strong_count >= 2:
        verdict = "PASS-OperatorTransfer"

    report = {
        "experiment": "SEM-5C.4 Operator-Transfer Memory Control Audit",
        "verdict": verdict,
        "motivation": [
            "SEM-5C.3 suggested mu_shape may act as an operator-level control prior.",
            "This audit tests whether same-operator wrong-concept shape transfers better than same-concept wrong-operator shape and shuffle."
        ],
        "addressing_fixed": {
            "retrieval": RETRIEVAL_NAME,
            "wo": RETRIEVAL_WO,
            "wc": RETRIEVAL_WC,
            "summary": addr_summary,
        },
        "interpretation_rules": [
            "If same_operator_wrong_concept_shape > same_concept_wrong_operator_shape, mu_shape is more operator-like than concept-like.",
            "If same_operator_wrong_concept_shape > shuffle/diffdiff, operator transfer is nontrivial.",
            "If oracle > same_operator_wrong_concept_shape, family-specific shape still contains additional information.",
            "If retrieved > shuffle, commit-addressed memory is still useful in live intervention."
        ],
        "outputs": {
            "addressing": str(addr_csv),
            "results": str(res_csv),
            "summary": str(summary_csv),
            "checks": str(checks_csv),
        },
    }

    report_json = OUT_DIR / "sem5c4_report.json"
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("SEM-5C.4 finished")
    print("=" * 90)
    print("Addressing summary:")
    print(addr_summary)
    print("\nComponent/operator-transfer summary:")
    print(summary)
    print("\nPass checks:")
    print(checks_df)
    print(f"\nVerdict: {verdict}")
    print(f"Saved: {addr_csv}")
    print(f"Saved: {res_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {checks_csv}")
    print(f"Saved: {report_json}")


if __name__ == "__main__":
    run()
