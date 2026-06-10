# -*- coding: utf-8 -*-
"""
SEM-5C.8: Residual Mode Low-Rank Compression Audit

Motivation
----------
SEM-5C.7 showed:
    FamilyResidual has utility.
    commit_retrieved_residual and top-k commit average residual beat shuffle.
    But the extra residual gain over operator prototype is modest.

SEM-5C.8 asks:
    Can FamilyResidual be compressed into low-rank residual modes R_F^{1:k}?

Core decomposition
------------------
    mu_F_shape = OperatorPrototype_O(F) + FamilyResidual_F

SEM-5C.8 tests:
    FamilyResidual_F ≈ PCA_k(FamilyResiduals)

Two types of evidence:
1. Compression upper bound:
       oracle residual projected into residual PCA basis, then reconstructed.
       If low k recovers most oracle residual utility, residual modes are low-rank.

2. Addressable compressed residual:
       commit-retrieved / top-k residual average projected into residual PCA basis.
       If this beats shuffle-projected residual, residual modes are addressable.

Conditions
----------
raw
operator_proto
oracle_residual
oracle_residual_pca_k{2,4,8,16}
retrieved_residual_pca_k{2,4,8,16}
topk_residual_pca_k{2,4,8,16}
shuffle_residual_pca_k{2,4,8,16}
oracle_shape
shuffle_shape

Expected
--------
oracle_residual_pca_k increases with k and approaches oracle_residual / oracle_shape.
retrieved_residual_pca_k > shuffle_residual_pca_k.
topk_residual_pca_k > shuffle_residual_pca_k.

If true:
    R_F^{1:k} is a compact, useful, and partially addressable residual memory.
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

from sklearn.decomposition import PCA
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

OUT_DIR = Path("sem5c8_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SHAPE_LAYERS = list(range(7, 20))
COMMIT_LAYERS = list(range(23, 26))

MAX_LENGTH = 512
MAX_ROWS = None

ALPHAS = [0.10, 0.20, 0.30]
RESIDUAL_BETA = 0.5
PCA_KS = [2, 4, 8, 16]
TOPK_RESIDUAL = 3

# commit-only retrieval
RETRIEVAL_WO = 0.0
RETRIEVAL_WC = 1.0


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

class ShapeInjector:
    def __init__(self, model, mu_shape: Optional[torch.Tensor], alpha: float):
        self.model = model
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
            if self.mu_shape is not None and self.alpha != 0:
                v = self.mu_shape.to(h2.device, h2.dtype)
                h2[:, -1, :] = h2[:, -1, :] + self.alpha * v

            if rest is not None:
                return (h2,) + rest
            return h2
        return fn

    def __enter__(self):
        layers = self.model.model.layers
        for idx in SHAPE_LAYERS:
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
# Residual helpers
# =========================

def make_operator_prototype(bank, operator, exclude_family):
    candidates = [
        mem["shape"] for fam, mem in bank.items()
        if fam != exclude_family and str(mem.get("operator")) == str(operator)
    ]
    if not candidates:
        candidates = [mem["shape"] for fam, mem in bank.items() if fam != exclude_family]
    return normalize(np.mean(np.stack(candidates, axis=0), axis=0))


def residual_for_family(bank, family):
    mem = bank[family]
    proto = make_operator_prototype(bank, str(mem.get("operator")), family)
    return (mem["shape"] - proto).astype(np.float32)


def compose(proto, residual, beta=RESIDUAL_BETA):
    return normalize(proto + beta * residual)


def fit_residual_pca(bank, exclude_family, max_k):
    fams = [fam for fam in bank.keys() if fam != exclude_family]
    residuals = np.stack([residual_for_family(bank, fam) for fam in fams], axis=0)
    n_components = min(max_k, residuals.shape[0], residuals.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    pca.fit(residuals)
    return pca, residuals, fams


def project_reconstruct(pca, residual, k):
    k_eff = min(k, pca.components_.shape[0])
    centered = residual - pca.mean_
    coeff = centered @ pca.components_[:k_eff].T
    rec = pca.mean_ + coeff @ pca.components_[:k_eff]
    return rec.astype(np.float32)


def score_retrieval(q_shape, q_commit, mem):
    return float(RETRIEVAL_WO * cosine(q_shape, mem["shape"]) + RETRIEVAL_WC * cosine(q_commit, mem["commit"]))


def ranked_retrieve(q_shape, q_commit, bank):
    rows = []
    for fam, mem in bank.items():
        rows.append((fam, mem, score_retrieval(q_shape, q_commit, mem)))
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
    print("SEM-5C.8 Residual Mode Low-Rank Compression Audit")
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

    raw_records = []
    print("[Stage 1] Collect raw trajectories...")
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

    rows = []
    pca_rows = []

    print("[Stage 2] Residual PCA compression injection...")

    for rec_i, rec in enumerate(raw_records):
        oracle_family = str(rec["family"])
        operator = str(rec["operator"])

        if external_bank is not None and oracle_family in external_bank:
            bank = external_bank
        else:
            bank = build_live_memory_bank(raw_records, exclude_row_idx=rec["row_idx"])

        if oracle_family not in bank:
            continue

        operator_proto = make_operator_prototype(bank, operator, oracle_family)
        oracle_residual = residual_for_family(bank, oracle_family)

        ranked = ranked_retrieve(rec["shape_raw"], rec["commit_raw"], bank)
        retrieved_family = ranked[0][0]
        topk_fams = [fam for fam, _, _ in ranked[:TOPK_RESIDUAL]]
        topk_residual = np.mean(np.stack([residual_for_family(bank, fam) for fam in topk_fams], axis=0), axis=0)

        non_oracle = [fam for fam in bank.keys() if fam != oracle_family] or list(bank.keys())
        shuffle_family = random.choice(non_oracle)
        shuffle_residual = residual_for_family(bank, shuffle_family)

        max_k = max(PCA_KS)
        pca, residual_matrix, pca_fams = fit_residual_pca(bank, oracle_family, max_k=max_k)

        for k in PCA_KS:
            k_eff = min(k, pca.components_.shape[0])
            evr = float(np.sum(pca.explained_variance_ratio_[:k_eff]))
            pca_rows.append({
                "row_idx": rec["row_idx"],
                "family": oracle_family,
                "k": k,
                "k_eff": k_eff,
                "explained_variance_ratio": evr,
                "oracle_residual_recon_cos": cosine(oracle_residual, project_reconstruct(pca, oracle_residual, k)),
                "retrieved_residual_recon_cos": cosine(residual_for_family(bank, retrieved_family), project_reconstruct(pca, residual_for_family(bank, retrieved_family), k)),
                "topk_residual_recon_cos": cosine(topk_residual, project_reconstruct(pca, topk_residual, k)),
                "shuffle_residual_recon_cos": cosine(shuffle_residual, project_reconstruct(pca, shuffle_residual, k)),
            })

        condition_vecs = {
            "raw": None,
            "operator_proto": operator_proto,
            "oracle_residual": compose(operator_proto, oracle_residual),
            "retrieved_residual": compose(operator_proto, residual_for_family(bank, retrieved_family)),
            "topk_residual": compose(operator_proto, topk_residual),
            "shuffle_residual": compose(operator_proto, shuffle_residual),
            "oracle_shape": bank[oracle_family]["shape"],
            "shuffle_shape": bank[shuffle_family]["shape"],
        }

        for k in PCA_KS:
            condition_vecs[f"oracle_residual_pca{k}"] = compose(operator_proto, project_reconstruct(pca, oracle_residual, k))
            condition_vecs[f"retrieved_residual_pca{k}"] = compose(operator_proto, project_reconstruct(pca, residual_for_family(bank, retrieved_family), k))
            condition_vecs[f"topk_residual_pca{k}"] = compose(operator_proto, project_reconstruct(pca, topk_residual, k))
            condition_vecs[f"shuffle_residual_pca{k}"] = compose(operator_proto, project_reconstruct(pca, shuffle_residual, k))

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

        for cond, vec in condition_vecs.items():
            alphas = [0.0] if cond == "raw" else ALPHAS
            for alpha in alphas:
                if cond == "raw":
                    out_shape, out_commit = raw_shape, raw_commit
                else:
                    mu_shape = torch.tensor(vec, dtype=torch.float32)
                    with ShapeInjector(model, mu_shape, alpha):
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
                    "concept": rec["concept"],
                    "operator": operator,
                    "condition": cond,
                    "alpha": alpha,
                    "retrieved_family": retrieved_family,
                    "retrieval_hit": int(retrieved_family == oracle_family),
                    "topk_fams": "|".join(topk_fams),
                    "shuffle_family": shuffle_family,

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
    res_csv = OUT_DIR / "sem5c8_residual_mode_compression_results.csv"
    res_df.to_csv(res_csv, index=False, encoding="utf-8-sig")

    pca_df = pd.DataFrame(pca_rows)
    pca_csv = OUT_DIR / "sem5c8_residual_pca_diagnostics.csv"
    pca_df.to_csv(pca_csv, index=False, encoding="utf-8-sig")

    summary = (
        res_df.groupby(["condition", "alpha"], dropna=False)
        .agg(
            n=("row_idx", "count"),
            retrieval_hit=("retrieval_hit", "mean"),
            gain_shape_sim=("gain_shape_sim", "mean"),
            gain_margin_shape=("gain_margin_shape", "mean"),
            rank_improvement_shape=("rank_improvement_shape", "mean"),
            gain_commit_sim=("gain_commit_sim", "mean"),
            gain_margin_commit=("gain_margin_commit", "mean"),
            rank_improvement_commit=("rank_improvement_commit", "mean"),
        )
        .reset_index()
    )

    summary_csv = OUT_DIR / "sem5c8_summary.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    pca_summary = (
        pca_df.groupby("k")
        .agg(
            n=("row_idx", "count"),
            explained_variance_ratio=("explained_variance_ratio", "mean"),
            oracle_residual_recon_cos=("oracle_residual_recon_cos", "mean"),
            retrieved_residual_recon_cos=("retrieved_residual_recon_cos", "mean"),
            topk_residual_recon_cos=("topk_residual_recon_cos", "mean"),
            shuffle_residual_recon_cos=("shuffle_residual_recon_cos", "mean"),
        )
        .reset_index()
    )
    pca_summary_csv = OUT_DIR / "sem5c8_pca_summary.csv"
    pca_summary.to_csv(pca_summary_csv, index=False, encoding="utf-8-sig")

    # Pass checks
    checks = []
    for alpha in ALPHAS:
        sub = summary[summary["alpha"] == alpha]

        def val(cond, col):
            v = sub[sub["condition"] == cond][col]
            return float(v.iloc[0]) if len(v) else np.nan

        oracle_shape = val("oracle_shape", "rank_improvement_shape")
        operator_proto = val("operator_proto", "rank_improvement_shape")
        shuffle_res = val("shuffle_residual", "rank_improvement_shape")
        oracle_res = val("oracle_residual", "rank_improvement_shape")

        for k in PCA_KS:
            oracle_pca = val(f"oracle_residual_pca{k}", "rank_improvement_shape")
            retrieved_pca = val(f"retrieved_residual_pca{k}", "rank_improvement_shape")
            topk_pca = val(f"topk_residual_pca{k}", "rank_improvement_shape")
            shuffle_pca = val(f"shuffle_residual_pca{k}", "rank_improvement_shape")

            evr = float(pca_summary[pca_summary["k"] == k]["explained_variance_ratio"].iloc[0])

            checks.append({
                "check": f"oracle_pca_k{k}_alpha_{alpha}",
                "k": k,
                "alpha": alpha,
                "value": oracle_pca,
                "operator_proto": operator_proto,
                "oracle_residual": oracle_res,
                "oracle_shape": oracle_shape,
                "evr": evr,
                "pass_lite": bool(oracle_pca > operator_proto),
                "pass_strong": bool((oracle_pca > operator_proto) and (oracle_shape >= oracle_pca)),
                "note": "Compressed oracle residual should improve over operator prototype."
            })

            checks.append({
                "check": f"retrieved_pca_k{k}_alpha_{alpha}",
                "k": k,
                "alpha": alpha,
                "value": retrieved_pca,
                "shuffle_pca": shuffle_pca,
                "operator_proto": operator_proto,
                "oracle_shape": oracle_shape,
                "evr": evr,
                "pass_lite": bool(retrieved_pca > shuffle_pca),
                "pass_strong": bool((retrieved_pca > shuffle_pca) and (retrieved_pca >= operator_proto) and (oracle_shape >= retrieved_pca)),
                "note": "Compressed retrieved residual should beat compressed shuffle residual."
            })

            checks.append({
                "check": f"topk_pca_k{k}_alpha_{alpha}",
                "k": k,
                "alpha": alpha,
                "value": topk_pca,
                "shuffle_pca": shuffle_pca,
                "operator_proto": operator_proto,
                "oracle_shape": oracle_shape,
                "evr": evr,
                "pass_lite": bool(topk_pca > shuffle_pca),
                "pass_strong": bool((topk_pca > shuffle_pca) and (topk_pca >= operator_proto) and (oracle_shape >= topk_pca)),
                "note": "Compressed top-k residual should beat compressed shuffle residual."
            })

    checks_df = pd.DataFrame(checks)
    checks_csv = OUT_DIR / "sem5c8_pass_checks.csv"
    checks_df.to_csv(checks_csv, index=False, encoding="utf-8-sig")

    verdict = "FAIL"
    if checks_df["pass_lite"].any():
        verdict = "PASS-Lite"

    oracle_pca_strong = checks_df[checks_df["check"].str.startswith("oracle_pca")]["pass_strong"].sum()
    retrieved_pca_strong = checks_df[checks_df["check"].str.startswith("retrieved_pca")]["pass_strong"].sum()
    topk_pca_strong = checks_df[checks_df["check"].str.startswith("topk_pca")]["pass_strong"].sum()

    if oracle_pca_strong >= 4 and (retrieved_pca_strong >= 4 or topk_pca_strong >= 4):
        verdict = "PASS-Strong"
    elif oracle_pca_strong >= 4:
        verdict = "PASS-CompressedOracleResidual"

    report = {
        "experiment": "SEM-5C.8 Residual Mode Low-Rank Compression Audit",
        "verdict": verdict,
        "motivation": [
            "SEM-5C.7 showed residuals can be partially retrieved/averaged.",
            "SEM-5C.8 tests whether family residuals can be represented by low-rank residual modes R_F^{1:k}."
        ],
        "config": {
            "residual_beta": RESIDUAL_BETA,
            "pca_ks": PCA_KS,
            "topk_residual": TOPK_RESIDUAL,
        },
        "interpretation_rules": [
            "If oracle_residual_pca_k improves over operator_proto, residual modes compress useful family information.",
            "If retrieved/topk residual PCA beats shuffle PCA, compressed residual modes are addressable.",
            "If only oracle PCA works, residuals are compressible but retrieval remains weak."
        ],
        "outputs": {
            "results": str(res_csv),
            "pca_diagnostics": str(pca_csv),
            "summary": str(summary_csv),
            "pca_summary": str(pca_summary_csv),
            "checks": str(checks_csv),
        }
    }

    report_json = OUT_DIR / "sem5c8_report.json"
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("SEM-5C.8 finished")
    print("=" * 90)
    print("PCA summary:")
    print(pca_summary)
    print("\nMain summary:")
    print(summary)
    print("\nPass checks:")
    print(checks_df)
    print(f"\nVerdict: {verdict}")
    print(f"Saved: {res_csv}")
    print(f"Saved: {pca_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {pca_summary_csv}")
    print(f"Saved: {checks_csv}")
    print(f"Saved: {report_json}")


if __name__ == "__main__":
    run()
