# -*- coding: utf-8 -*-
"""
SEM-5C.5: Factorized MemoryUnit Composition Audit

Motivation
----------
SEM-5C.2:
    commit-only retrieval is the best address.
SEM-5C.3:
    mu_commit ~= Identity/Address; mu_shape ~= Trajectory Control.
SEM-5C.4:
    same_operator_wrong_concept_shape transfers strongly;
    mu_shape behaves like an operator-level prior.

SEM-5C.5 asks:
    Can a synthetic MemoryUnit be composed from:
        commit vector of same-concept/wrong-operator memory
        +
        shape vector of same-operator/wrong-concept memory

If this works, MemoryUnit_F factorizes approximately as:

    MemoryUnit_F ~= ConceptIdentity_F + OperatorPrior_F

Core conditions
---------------
raw

oracle_full:
    shape = oracle family
    commit = oracle family

retrieved_full:
    shape = commit-retrieved family
    commit = commit-retrieved family

synthetic_concept_commit_operator_shape:
    commit = same_concept_wrong_operator memory
    shape  = same_operator_wrong_concept memory

anti_synthetic_operator_commit_concept_shape:
    commit = same_operator_wrong_concept memory
    shape  = same_concept_wrong_operator memory

same_operator_shape_only:
    shape = same_operator_wrong_concept memory

same_concept_commit_only:
    commit = same_concept_wrong_operator memory

shuffle_full:
    shape = random non-oracle
    commit = random non-oracle

Main hypothesis
---------------
synthetic_concept_commit_operator_shape
    > anti_synthetic_operator_commit_concept_shape
    > shuffle_full

and ideally:

oracle_full >= synthetic_concept_commit_operator_shape >= retrieved_full

Interpretation
--------------
If synthetic works:
    commit carries concept / identity address;
    shape carries transferable operator prior;
    MemoryUnit can be assembled compositionally.
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

OUT_DIR = Path("sem5c5_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SHAPE_LAYERS = list(range(7, 20))
COMMIT_LAYERS = list(range(23, 26))

MAX_LENGTH = 512
MAX_ROWS = None

ALPHAS = [0.10, 0.20, 0.30]

# commit-only retrieval fixed from SEM-5C.2/5C.3/5C.4
RETRIEVAL_WO = 0.0
RETRIEVAL_WC = 1.0
RETRIEVAL_NAME = "commit_only"


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
        raise ValueError("Cannot find concept column.")
    return col


def get_operator_col(df: pd.DataFrame) -> str:
    col = find_col(df, ["operator_star", "operator_id", "operator", "operator_phrase"])
    if col is None:
        raise ValueError("Cannot find operator column.")
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

class FactorizedInjector:
    """
    Inject shape vector into L7-L19 and commit vector into L23-L25.

    This tests whether concept identity and operator trajectory control can be
    separated and recombined.
    """

    def __init__(
        self,
        model,
        mu_shape: Optional[torch.Tensor],
        mu_commit: Optional[torch.Tensor],
        alpha_shape: float,
        alpha_commit: float,
    ):
        self.model = model
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

            if layer_idx in SHAPE_LAYERS and self.mu_shape is not None and self.alpha_shape != 0:
                v = self.mu_shape.to(h2.device, h2.dtype)
                h2[:, -1, :] = h2[:, -1, :] + self.alpha_shape * v

            if layer_idx in COMMIT_LAYERS and self.mu_commit is not None and self.alpha_commit != 0:
                v = self.mu_commit.to(h2.device, h2.dtype)
                h2[:, -1, :] = h2[:, -1, :] + self.alpha_commit * v

            if rest is not None:
                return (h2,) + rest
            return h2
        return fn

    def __enter__(self):
        layers = self.model.model.layers
        for idx in set(SHAPE_LAYERS + COMMIT_LAYERS):
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
        return None

    arr = np.load(npz_p)
    shape_key = next((k for k in ["mu_shape", "mu_F_shape", "shape", "memory_shape"] if k in arr), None)
    commit_key = next((k for k in ["mu_commit", "mu_F_commit", "commit", "memory_commit"] if k in arr), None)
    if shape_key is None:
        return None

    shapes = arr[shape_key]
    if shapes.shape[-1] != hidden_dim:
        return None

    commits = arr[commit_key] if commit_key is not None and arr[commit_key].shape[-1] == hidden_dim else None

    meta_map = (
        family_meta[[family_col, concept_col, operator_col]]
        .drop_duplicates(subset=[family_col])
        .assign(**{family_col: lambda x: x[family_col].astype(str)})
        .set_index(family_col)
        .to_dict("index")
    )

    bank = {}
    for i, fam in enumerate(meta[fam_col_meta].astype(str).tolist()):
        if fam in meta_map:
            concept = str(meta_map[fam][concept_col])
            operator = str(meta_map[fam][operator_col])
        else:
            concept, operator = None, None

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

    if exclude_row_idx is not None:
        all_by_family = {}
        for rec in raw_records:
            all_by_family.setdefault(str(rec["family"]), []).append(rec)
        for fam, xs in all_by_family.items():
            if fam not in by_family or len(by_family[fam]) == 0:
                by_family[fam] = xs

    bank = {}
    for fam, xs in by_family.items():
        bank[fam] = {
            "shape": normalize(np.mean(np.stack([r["shape_raw"] for r in xs], axis=0), axis=0)),
            "commit": normalize(np.mean(np.stack([r["commit_raw"] for r in xs], axis=0), axis=0)),
            "concept": str(xs[0]["concept"]),
            "operator": str(xs[0]["operator"]),
        }
    return bank


# =========================
# Retrieval / Candidate selection
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
    fams = list(bank.keys())

    def ok(fam):
        if fam == oracle_family:
            return False
        mem = bank[fam]
        c = str(mem.get("concept"))
        o = str(mem.get("operator"))
        if kind == "same_operator_wrong_concept":
            return o == str(operator) and c != str(concept)
        if kind == "same_concept_wrong_operator":
            return c == str(concept) and o != str(operator)
        if kind == "diff_concept_diff_operator":
            return c != str(concept) and o != str(operator)
        if kind == "shuffle":
            return fam != oracle_family
        raise ValueError(kind)

    candidates = [fam for fam in fams if ok(fam)]
    if not candidates:
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
    print("SEM-5C.5 Factorized MemoryUnit Composition Audit")
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

    # Stage 1: raw representations
    print("[Stage 1] Collect raw trajectories...")
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

    # Stage 2: factorized injection
    print("[Stage 2] Factorized composition injection...")
    rows = []
    addr_rows = []

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
        fams = [r[0] for r in ranked]
        retrieved_family = fams[0]
        oracle_rank = fams.index(oracle_family) + 1 if oracle_family in fams else len(fams) + 1

        same_op_fam = select_candidate(bank, oracle_family, concept, operator, "same_operator_wrong_concept")
        same_con_fam = select_candidate(bank, oracle_family, concept, operator, "same_concept_wrong_operator")
        diffdiff_fam = select_candidate(bank, oracle_family, concept, operator, "diff_concept_diff_operator")
        shuffle_fam = select_candidate(bank, oracle_family, concept, operator, "shuffle")

        addr_rows.append({
            "row_idx": rec["row_idx"],
            "row_id": rec["row_id"],
            "family": oracle_family,
            "concept": concept,
            "operator": operator,
            "retrieved_family": retrieved_family,
            "retrieval_hit": int(retrieved_family == oracle_family),
            "retrieved_same_concept": int(str(bank[retrieved_family].get("concept")) == concept),
            "retrieved_same_operator": int(str(bank[retrieved_family].get("operator")) == operator),
            "oracle_rank": oracle_rank,
            "same_operator_family": same_op_fam,
            "same_concept_family": same_con_fam,
            "diffdiff_family": diffdiff_fam,
            "shuffle_family": shuffle_fam,
        })

        # baseline raw metrics
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

        # condition -> (shape_family, commit_family, alpha_shape_multiplier, alpha_commit_multiplier)
        conditions = {
            "raw": (None, None, 0.0, 0.0),

            "oracle_full": (oracle_family, oracle_family, 1.0, 1.0),
            "retrieved_full": (retrieved_family, retrieved_family, 1.0, 1.0),

            # Main synthetic hypothesis:
            # concept identity from same-concept commit, operator control from same-operator shape
            "synthetic_concept_commit_operator_shape": (same_op_fam, same_con_fam, 1.0, 1.0),

            # Anti-synthetic: opposite wiring
            "anti_synthetic_operator_commit_concept_shape": (same_con_fam, same_op_fam, 1.0, 1.0),

            # One-channel controls
            "same_operator_shape_only": (same_op_fam, None, 1.0, 0.0),
            "same_concept_commit_only": (None, same_con_fam, 0.0, 1.0),

            "diffdiff_full": (diffdiff_fam, diffdiff_fam, 1.0, 1.0),
            "shuffle_full": (shuffle_fam, shuffle_fam, 1.0, 1.0),
        }

        for cond, (shape_fam, commit_fam, smul, cmul) in conditions.items():
            alphas = [0.0] if cond == "raw" else ALPHAS

            for alpha in alphas:
                if cond == "raw":
                    out_shape, out_commit = raw_shape, raw_commit
                else:
                    mu_shape = None if shape_fam is None else torch.tensor(bank[shape_fam]["shape"], dtype=torch.float32)
                    mu_commit = None if commit_fam is None else torch.tensor(bank[commit_fam]["commit"], dtype=torch.float32)

                    with FactorizedInjector(
                        model=model,
                        mu_shape=mu_shape,
                        mu_commit=mu_commit,
                        alpha_shape=alpha * smul,
                        alpha_commit=alpha * cmul,
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

                rows.append({
                    "row_idx": rec["row_idx"],
                    "row_id": rec["row_id"],
                    "family": oracle_family,
                    "concept": concept,
                    "operator": operator,
                    "condition": cond,
                    "alpha": alpha,
                    "shape_family": shape_fam,
                    "commit_family": commit_fam,
                    "shape_family_concept": None if shape_fam is None else bank[shape_fam].get("concept"),
                    "shape_family_operator": None if shape_fam is None else bank[shape_fam].get("operator"),
                    "commit_family_concept": None if commit_fam is None else bank[commit_fam].get("concept"),
                    "commit_family_operator": None if commit_fam is None else bank[commit_fam].get("operator"),
                    "retrieved_family": retrieved_family,
                    "retrieval_hit": int(retrieved_family == oracle_family),

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

    addr_df = pd.DataFrame(addr_rows)
    addr_csv = OUT_DIR / "sem5c5_addressing_results.csv"
    addr_df.to_csv(addr_csv, index=False, encoding="utf-8-sig")

    res_df = pd.DataFrame(rows)
    res_csv = OUT_DIR / "sem5c5_factorized_composition_results.csv"
    res_df.to_csv(res_csv, index=False, encoding="utf-8-sig")

    summary = (
        res_df.groupby(["condition", "alpha"], dropna=False)
        .agg(
            n=("row_idx", "count"),
            retrieval_hit=("retrieval_hit", "mean"),
            gain_shape_sim=("gain_shape_sim", "mean"),
            gain_commit_sim=("gain_commit_sim", "mean"),
            gain_margin_shape=("gain_margin_shape", "mean"),
            gain_margin_commit=("gain_margin_commit", "mean"),
            rank_improvement_shape=("rank_improvement_shape", "mean"),
            rank_improvement_commit=("rank_improvement_commit", "mean"),
        )
        .reset_index()
    )

    summary_csv = OUT_DIR / "sem5c5_summary.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    # Pass checks
    checks = []
    for alpha in ALPHAS:
        sub = summary[summary["alpha"] == alpha]

        def val(cond, col):
            v = sub[sub["condition"] == cond][col]
            return float(v.iloc[0]) if len(v) else np.nan

        synthetic = val("synthetic_concept_commit_operator_shape", "rank_improvement_shape")
        anti = val("anti_synthetic_operator_commit_concept_shape", "rank_improvement_shape")
        oracle = val("oracle_full", "rank_improvement_shape")
        retrieved = val("retrieved_full", "rank_improvement_shape")
        shuffle = val("shuffle_full", "rank_improvement_shape")
        diffdiff = val("diffdiff_full", "rank_improvement_shape")
        sameop_shape = val("same_operator_shape_only", "rank_improvement_shape")
        samecon_commit = val("same_concept_commit_only", "rank_improvement_shape")

        checks.append({
            "check": f"synthetic_vs_anti_alpha_{alpha}",
            "value": synthetic,
            "anti": anti,
            "oracle": oracle,
            "pass_lite": bool(synthetic > anti),
            "pass_strong": bool((synthetic > anti) and (oracle >= synthetic)),
            "note": "Correct wiring should beat anti-synthetic wiring."
        })

        checks.append({
            "check": f"synthetic_vs_shuffle_alpha_{alpha}",
            "value": synthetic,
            "shuffle": shuffle,
            "diffdiff": diffdiff,
            "oracle": oracle,
            "pass_lite": bool(synthetic > shuffle),
            "pass_strong": bool((synthetic > shuffle) and (synthetic > diffdiff) and (oracle >= synthetic)),
            "note": "Synthetic memory should beat random/mismatched memory."
        })

        checks.append({
            "check": f"synthetic_vs_one_channel_alpha_{alpha}",
            "value": synthetic,
            "same_operator_shape_only": sameop_shape,
            "same_concept_commit_only": samecon_commit,
            "oracle": oracle,
            "pass_lite": bool(synthetic > samecon_commit),
            "pass_strong": bool((synthetic >= sameop_shape) and (synthetic > samecon_commit) and (oracle >= synthetic)),
            "note": "If composition works, synthetic should exceed commit-only and approach shape-only/operator control."
        })

        checks.append({
            "check": f"retrieved_full_vs_shuffle_alpha_{alpha}",
            "value": retrieved,
            "shuffle": shuffle,
            "oracle": oracle,
            "pass_lite": bool(retrieved > shuffle),
            "pass_strong": bool((retrieved > shuffle) and (oracle >= retrieved)),
            "note": "Commit-retrieved full memory should remain better than shuffle."
        })

    checks_df = pd.DataFrame(checks)
    checks_csv = OUT_DIR / "sem5c5_pass_checks.csv"
    checks_df.to_csv(checks_csv, index=False, encoding="utf-8-sig")

    addr_summary = {
        "top1_hit": float(addr_df["retrieval_hit"].mean()),
        "top1_same_concept": float(addr_df["retrieved_same_concept"].mean()),
        "top1_same_operator": float(addr_df["retrieved_same_operator"].mean()),
        "mean_oracle_rank": float(addr_df["oracle_rank"].mean()),
    }

    verdict = "FAIL"
    if checks_df["pass_lite"].any():
        verdict = "PASS-Lite"

    syn_shuffle_strong = checks_df[checks_df["check"].str.startswith("synthetic_vs_shuffle")]["pass_strong"].sum()
    syn_anti_strong = checks_df[checks_df["check"].str.startswith("synthetic_vs_anti")]["pass_strong"].sum()
    retrieved_strong = checks_df[checks_df["check"].str.startswith("retrieved_full")]["pass_strong"].sum()

    if syn_shuffle_strong >= 2 and syn_anti_strong >= 2:
        verdict = "PASS-FactorizedComposition"
    if syn_shuffle_strong >= 2 and syn_anti_strong >= 2 and retrieved_strong >= 2:
        verdict = "PASS-Strong"

    report = {
        "experiment": "SEM-5C.5 Factorized MemoryUnit Composition Audit",
        "verdict": verdict,
        "motivation": [
            "SEM-5C.4 suggests mu_shape is operator-transferable.",
            "SEM-5C.5 tests whether commit(identity) and shape(operator) can be recombined into synthetic MemoryUnits."
        ],
        "addressing_fixed": {
            "retrieval": RETRIEVAL_NAME,
            "wo": RETRIEVAL_WO,
            "wc": RETRIEVAL_WC,
            "summary": addr_summary,
        },
        "interpretation_rules": [
            "If synthetic_concept_commit_operator_shape > anti_synthetic, correct factor wiring is supported.",
            "If synthetic > shuffle/diffdiff, factorized composition is nontrivial.",
            "If oracle > synthetic, family-specific memory still contains additional information beyond factorization.",
            "If retrieved_full > shuffle_full, commit-retrieved memory remains useful."
        ],
        "outputs": {
            "addressing": str(addr_csv),
            "results": str(res_csv),
            "summary": str(summary_csv),
            "checks": str(checks_csv),
        }
    }

    report_json = OUT_DIR / "sem5c5_report.json"
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("SEM-5C.5 finished")
    print("=" * 90)
    print("Addressing summary:")
    print(addr_summary)
    print("\nSummary:")
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
