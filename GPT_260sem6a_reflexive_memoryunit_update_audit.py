# -*- coding: utf-8 -*-
"""
SEM-6A: Reflexive MemoryUnit Update Audit

Motivation
----------
SEM-5C.9 selected the minimal non-oracle MemoryUnit form:

    MemoryUnit_lite =
        IdentityAddress:   mu_commit
        OperatorPrior:     operator_proto_shape
        ResidualModes:     topk_residual_pca, k=2
        Policy:            deferred

SEM-6A tests the first reflexive-memory claim:

    R: MemoryUnit_t -> MemoryUnit_{t+1}

Question:
    If a new successful experience is added to the memory bank,
    does the updated MemoryUnit improve later held-out trajectories?

Experimental split
------------------
For each SeedFamily with multiple surface prompts:
    init rows    -> build M_t
    update row   -> update M_t into M_{t+1}
    test rows    -> evaluate trajectory guidance

Conditions:
    raw
    M0_operator_proto
    M0_minimal_topk_pca2
    M1_operator_proto
    M1_minimal_topk_pca2
    oracle_shape
    shuffle_shape

Main hypothesis:
    M1_minimal_topk_pca2 > M0_minimal_topk_pca2
    M1_operator_proto >= M0_operator_proto
    oracle_shape >= M1_minimal_topk_pca2 > shuffle_shape

If true:
    New experience can be compressed into MemoryUnit and improves future trajectory.

Default paths:
    model:   D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main
    dataset: sem5a2_outputs/sem5a2_dataset.csv
"""

import json
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

MODEL_PATH = r"D:\model\models--Qwen--Qwen2.5B-Instruct\main"
# Auto fallback to the known correct project path if above typo path absent.
MODEL_PATH_FALLBACK = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

DATASET_CSV = r"sem5a2_outputs\sem5a2_dataset.csv"
OUT_DIR = Path("sem6a_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SHAPE_LAYERS = list(range(7, 20))
COMMIT_LAYERS = list(range(23, 26))

MAX_LENGTH = 512
MAX_ROWS = None

ALPHAS = [0.10, 0.20, 0.30]
RESIDUAL_BETA = 0.5
RESIDUAL_K = 2
TOPK_RESIDUAL = 3

# Minimum rows per family for init/update/test.
MIN_ROWS_PER_FAMILY = 3


# =========================
# Utilities
# =========================

def resolve_model_path() -> str:
    if Path(MODEL_PATH).exists():
        return MODEL_PATH
    return MODEL_PATH_FALLBACK


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


def layer_hidden_dict(outputs):
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
def forward_collect(model, tokenizer, prompt: str):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(DEVICE)
    outputs = model(**enc, output_hidden_states=True, use_cache=False)
    hdict = layer_hidden_dict(outputs)
    return {l: h[:, -1, :].detach()[0].float().cpu().numpy().astype(np.float32) for l, h in hdict.items()}


def extract_shape_commit(hidden_by_layer):
    shape = normalize(mean_pool_layers(hidden_by_layer, SHAPE_LAYERS))
    commit = normalize(mean_pool_layers(hidden_by_layer, COMMIT_LAYERS))
    return shape, commit


# =========================
# Memory construction
# =========================

def build_records(dataset: pd.DataFrame, prompt_col: str, family_col: str, concept_col: str, operator_col: str, row_id_col: Optional[str], model, tokenizer):
    records = []
    for pos, (idx, row) in enumerate(dataset.iterrows()):
        h = forward_collect(model, tokenizer, str(row[prompt_col]))
        shape, commit = extract_shape_commit(h)
        records.append({
            "row_idx": int(idx),
            "row_pos": int(pos),
            "row_id": row[row_id_col] if row_id_col is not None else idx,
            "family": str(row[family_col]),
            "concept": str(row[concept_col]),
            "operator": str(row[operator_col]),
            "prompt": str(row[prompt_col]),
            "shape": shape,
            "commit": commit,
        })
        if (pos + 1) % 10 == 0 or (pos + 1) == len(dataset):
            print(f"[Forward] {pos+1}/{len(dataset)}")
    return records


def make_family_splits(records):
    by_fam = {}
    for r in records:
        by_fam.setdefault(r["family"], []).append(r)

    splits = {}
    for fam, xs in by_fam.items():
        xs = sorted(xs, key=lambda r: r["row_pos"])
        if len(xs) < MIN_ROWS_PER_FAMILY:
            continue
        splits[fam] = {
            "init": xs[:1],
            "update": xs[1:2],
            "test": xs[2:],
            "all": xs,
        }
    return splits


def build_bank_from_rows(rows_by_family: Dict[str, List[dict]]):
    bank = {}
    for fam, rows in rows_by_family.items():
        if not rows:
            continue
        bank[fam] = {
            "shape": normalize(np.mean(np.stack([r["shape"] for r in rows], axis=0), axis=0)),
            "commit": normalize(np.mean(np.stack([r["commit"] for r in rows], axis=0), axis=0)),
            "concept": rows[0]["concept"],
            "operator": rows[0]["operator"],
        }
    return bank


def build_stage_banks(splits):
    init_rows = {fam: sp["init"] for fam, sp in splits.items()}
    updated_rows = {fam: sp["init"] + sp["update"] for fam, sp in splits.items()}
    oracle_rows = {fam: sp["all"] for fam, sp in splits.items()}
    return (
        build_bank_from_rows(init_rows),
        build_bank_from_rows(updated_rows),
        build_bank_from_rows(oracle_rows),
    )


def make_operator_prototype(bank, operator, exclude_family):
    candidates = [
        mem["shape"] for fam, mem in bank.items()
        if fam != exclude_family and str(mem.get("operator")) == str(operator)
    ]
    if not candidates:
        candidates = [mem["shape"] for fam, mem in bank.items() if fam != exclude_family]
    if not candidates:
        candidates = [mem["shape"] for fam, mem in bank.items()]
    return normalize(np.mean(np.stack(candidates, axis=0), axis=0))


def residual_for_family(bank, family):
    mem = bank[family]
    proto = make_operator_prototype(bank, str(mem.get("operator")), family)
    return (mem["shape"] - proto).astype(np.float32)


def compose(proto, residual):
    return normalize(proto + RESIDUAL_BETA * residual)


def fit_residual_pca(bank, exclude_family, max_k):
    fams = [fam for fam in bank.keys() if fam != exclude_family]
    if len(fams) < 2:
        return None
    residuals = np.stack([residual_for_family(bank, fam) for fam in fams], axis=0)
    n_components = min(max_k, residuals.shape[0], residuals.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    pca.fit(residuals)
    return pca


def project_reconstruct(pca, residual, k):
    if pca is None:
        return residual
    k_eff = min(k, pca.components_.shape[0])
    centered = residual - pca.mean_
    coeff = centered @ pca.components_[:k_eff].T
    return (pca.mean_ + coeff @ pca.components_[:k_eff]).astype(np.float32)


def score_retrieval(q_shape, q_commit, mem):
    # commit-only
    return cosine(q_commit, mem["commit"])


def ranked_retrieve(q_shape, q_commit, bank):
    rows = [(fam, mem, score_retrieval(q_shape, q_commit, mem)) for fam, mem in bank.items()]
    rows.sort(key=lambda x: x[2], reverse=True)
    return rows


def minimal_memory_shape(test_rec, bank, family, mode):
    """
    mode:
        operator_proto
        minimal_topk_pca2
        oracle_shape
    """
    operator = test_rec["operator"]
    proto = make_operator_prototype(bank, operator, family)

    if mode == "operator_proto":
        return proto

    if mode == "oracle_shape":
        return bank[family]["shape"]

    if mode == "minimal_topk_pca2":
        ranked = ranked_retrieve(test_rec["shape"], test_rec["commit"], bank)
        topk_fams = [fam for fam, _, _ in ranked[:TOPK_RESIDUAL]]
        pca = fit_residual_pca(bank, exclude_family=family, max_k=RESIDUAL_K)
        topk_residual = np.mean(np.stack([residual_for_family(bank, fam) for fam in topk_fams], axis=0), axis=0)
        rec_residual = project_reconstruct(pca, topk_residual, RESIDUAL_K)
        return compose(proto, rec_residual)

    raise ValueError(mode)


def family_scores(vec, bank, key="shape"):
    return {fam: cosine(vec, mem[key]) for fam, mem in bank.items()}


def rank_of_family(scores, oracle_family):
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    for i, (fam, _) in enumerate(ordered, start=1):
        if fam == oracle_family:
            return i
    return len(ordered) + 1


def margin_to_oracle(scores, oracle_family):
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
    print("SEM-6A Reflexive MemoryUnit Update Audit")
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

    model_path = resolve_model_path()
    print(f"[Model] path={model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    if DEVICE != "cuda":
        model.to(DEVICE)

    print("[Stage 1] Forward all rows...")
    records = build_records(dataset, prompt_col, family_col, concept_col, operator_col, row_id_col, model, tokenizer)
    splits = make_family_splits(records)
    print(f"[Splits] families={len(splits)}")

    bank_M0, bank_M1, bank_oracle = build_stage_banks(splits)

    rows = []

    print("[Stage 2] Evaluate M_t vs M_{t+1} on held-out test rows...")
    for fam, sp in splits.items():
        if fam not in bank_M0 or fam not in bank_M1 or fam not in bank_oracle:
            continue

        test_rows = sp["test"]
        for test_rec in test_rows:
            raw_shape = test_rec["shape"]
            raw_commit = test_rec["commit"]

            # raw baseline scored against oracle bank to keep target geometry stable
            raw_scores_shape = family_scores(raw_shape, bank_oracle, key="shape")
            raw_scores_commit = family_scores(raw_commit, bank_oracle, key="commit")
            raw_rank_shape = rank_of_family(raw_scores_shape, fam)
            raw_rank_commit = rank_of_family(raw_scores_commit, fam)
            raw_margin_shape = margin_to_oracle(raw_scores_shape, fam)
            raw_margin_commit = margin_to_oracle(raw_scores_commit, fam)
            raw_sim_shape = raw_scores_shape[fam]
            raw_sim_commit = raw_scores_commit[fam]

            condition_specs = {
                "raw": None,
                "M0_operator_proto": ("M0", "operator_proto"),
                "M0_minimal_topk_pca2": ("M0", "minimal_topk_pca2"),
                "M1_operator_proto": ("M1", "operator_proto"),
                "M1_minimal_topk_pca2": ("M1", "minimal_topk_pca2"),
                "oracle_shape": ("oracle", "oracle_shape"),
            }

            # shuffle from oracle bank
            non_oracle = [x for x in bank_oracle.keys() if x != fam] or list(bank_oracle.keys())
            shuffle_fam = random.choice(non_oracle)
            shuffle_vec = bank_oracle[shuffle_fam]["shape"]

            for cond, spec in condition_specs.items():
                alphas = [0.0] if cond == "raw" else ALPHAS
                for alpha in alphas:
                    if cond == "raw":
                        out_shape, out_commit = raw_shape, raw_commit
                        mem_vec = None
                    else:
                        bank_name, mode = spec
                        if bank_name == "M0":
                            bank = bank_M0
                        elif bank_name == "M1":
                            bank = bank_M1
                        else:
                            bank = bank_oracle
                        mem_vec = minimal_memory_shape(test_rec, bank, fam, mode)

                        with ShapeInjector(model, torch.tensor(mem_vec, dtype=torch.float32), alpha):
                            h2 = forward_collect(model, tokenizer, test_rec["prompt"])
                        out_shape, out_commit = extract_shape_commit(h2)

                    scores_shape = family_scores(out_shape, bank_oracle, key="shape")
                    scores_commit = family_scores(out_commit, bank_oracle, key="commit")

                    rows.append({
                        "family": fam,
                        "row_idx": test_rec["row_idx"],
                        "row_id": test_rec["row_id"],
                        "concept": test_rec["concept"],
                        "operator": test_rec["operator"],
                        "condition": cond,
                        "alpha": alpha,
                        "shuffle_family": None,

                        "shape_sim_to_oracle": scores_shape[fam],
                        "commit_sim_to_oracle": scores_commit[fam],
                        "family_margin_shape": margin_to_oracle(scores_shape, fam),
                        "family_margin_commit": margin_to_oracle(scores_commit, fam),
                        "family_rank_shape": rank_of_family(scores_shape, fam),
                        "family_rank_commit": rank_of_family(scores_commit, fam),

                        "gain_shape_sim": scores_shape[fam] - raw_sim_shape,
                        "gain_commit_sim": scores_commit[fam] - raw_sim_commit,
                        "gain_margin_shape": margin_to_oracle(scores_shape, fam) - raw_margin_shape,
                        "gain_margin_commit": margin_to_oracle(scores_commit, fam) - raw_margin_commit,
                        "rank_improvement_shape": raw_rank_shape - rank_of_family(scores_shape, fam),
                        "rank_improvement_commit": raw_rank_commit - rank_of_family(scores_commit, fam),
                    })

            for alpha in ALPHAS:
                with ShapeInjector(model, torch.tensor(shuffle_vec, dtype=torch.float32), alpha):
                    h2 = forward_collect(model, tokenizer, test_rec["prompt"])
                out_shape, out_commit = extract_shape_commit(h2)
                scores_shape = family_scores(out_shape, bank_oracle, key="shape")
                scores_commit = family_scores(out_commit, bank_oracle, key="commit")

                rows.append({
                    "family": fam,
                    "row_idx": test_rec["row_idx"],
                    "row_id": test_rec["row_id"],
                    "concept": test_rec["concept"],
                    "operator": test_rec["operator"],
                    "condition": "shuffle_shape",
                    "alpha": alpha,
                    "shuffle_family": shuffle_fam,

                    "shape_sim_to_oracle": scores_shape[fam],
                    "commit_sim_to_oracle": scores_commit[fam],
                    "family_margin_shape": margin_to_oracle(scores_shape, fam),
                    "family_margin_commit": margin_to_oracle(scores_commit, fam),
                    "family_rank_shape": rank_of_family(scores_shape, fam),
                    "family_rank_commit": rank_of_family(scores_commit, fam),

                    "gain_shape_sim": scores_shape[fam] - raw_sim_shape,
                    "gain_commit_sim": scores_commit[fam] - raw_sim_commit,
                    "gain_margin_shape": margin_to_oracle(scores_shape, fam) - raw_margin_shape,
                    "gain_margin_commit": margin_to_oracle(scores_commit, fam) - raw_margin_commit,
                    "rank_improvement_shape": raw_rank_shape - rank_of_family(scores_shape, fam),
                    "rank_improvement_commit": raw_rank_commit - rank_of_family(scores_commit, fam),
                })

    res_df = pd.DataFrame(rows)
    res_csv = OUT_DIR / "sem6a_reflexive_update_results.csv"
    res_df.to_csv(res_csv, index=False, encoding="utf-8-sig")

    summary = (
        res_df.groupby(["condition", "alpha"], dropna=False)
        .agg(
            n=("row_idx", "count"),
            gain_shape_sim=("gain_shape_sim", "mean"),
            gain_margin_shape=("gain_margin_shape", "mean"),
            rank_improvement_shape=("rank_improvement_shape", "mean"),
            gain_commit_sim=("gain_commit_sim", "mean"),
            gain_margin_commit=("gain_margin_commit", "mean"),
            rank_improvement_commit=("rank_improvement_commit", "mean"),
        )
        .reset_index()
    )
    summary_csv = OUT_DIR / "sem6a_summary.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    checks = []
    for alpha in ALPHAS:
        sub = summary[summary["alpha"] == alpha]

        def val(cond, col):
            v = sub[sub["condition"] == cond][col]
            return float(v.iloc[0]) if len(v) else np.nan

        m0 = val("M0_minimal_topk_pca2", "rank_improvement_shape")
        m1 = val("M1_minimal_topk_pca2", "rank_improvement_shape")
        m0_proto = val("M0_operator_proto", "rank_improvement_shape")
        m1_proto = val("M1_operator_proto", "rank_improvement_shape")
        shuffle = val("shuffle_shape", "rank_improvement_shape")
        oracle = val("oracle_shape", "rank_improvement_shape")

        checks.append({
            "check": f"reflexive_update_minimal_alpha_{alpha}",
            "alpha": alpha,
            "M0": m0,
            "M1": m1,
            "shuffle": shuffle,
            "oracle": oracle,
            "delta_M1_minus_M0": m1 - m0,
            "pass_lite": bool(m1 > m0),
            "pass_strong": bool((m1 > m0) and (m1 > shuffle) and (oracle >= m1)),
            "note": "Updated MemoryUnit should improve over initial MemoryUnit on held-out rows.",
        })

        checks.append({
            "check": f"reflexive_update_operator_proto_alpha_{alpha}",
            "alpha": alpha,
            "M0_proto": m0_proto,
            "M1_proto": m1_proto,
            "shuffle": shuffle,
            "oracle": oracle,
            "delta_M1_minus_M0": m1_proto - m0_proto,
            "pass_lite": bool(m1_proto >= m0_proto),
            "pass_strong": bool((m1_proto >= m0_proto) and (m1_proto > shuffle) and (oracle >= m1_proto)),
            "note": "Updated operator prototype should not degrade and should beat shuffle.",
        })

        checks.append({
            "check": f"minimal_vs_proto_after_update_alpha_{alpha}",
            "alpha": alpha,
            "M1_minimal": m1,
            "M1_proto": m1_proto,
            "shuffle": shuffle,
            "oracle": oracle,
            "delta_minimal_minus_proto": m1 - m1_proto,
            "pass_lite": bool(m1 >= m1_proto),
            "pass_strong": bool((m1 >= m1_proto) and (oracle >= m1)),
            "note": "Residual modes should add value beyond operator prototype after update.",
        })

    checks_df = pd.DataFrame(checks)
    checks_csv = OUT_DIR / "sem6a_pass_checks.csv"
    checks_df.to_csv(checks_csv, index=False, encoding="utf-8-sig")

    verdict = "FAIL"
    if checks_df["pass_lite"].any():
        verdict = "PASS-Lite"
    strong_update = checks_df[checks_df["check"].str.startswith("reflexive_update_minimal")]["pass_strong"].sum()
    strong_residual = checks_df[checks_df["check"].str.startswith("minimal_vs_proto")]["pass_strong"].sum()
    if strong_update >= 2 and strong_residual >= 2:
        verdict = "PASS-Strong"
    elif strong_update >= 2:
        verdict = "PASS-ReflexiveUpdate"

    report = {
        "experiment": "SEM-6A Reflexive MemoryUnit Update Audit",
        "verdict": verdict,
        "motivation": [
            "SEM-5C.9 recommended MemoryUnit_lite=(mu_commit, operator_proto_shape, topk_residual_pca2).",
            "SEM-6A tests whether adding one update experience improves held-out trajectory guidance."
        ],
        "split": {
            "per_family": "init=first row, update=second row, test=remaining rows",
            "families": len(splits),
            "test_rows": int(len(res_df[res_df["condition"] == "raw"])),
        },
        "interpretation_rules": [
            "If M1_minimal_topk_pca2 > M0_minimal_topk_pca2, MemoryUnit update improves future trajectory.",
            "If M1 > shuffle and oracle >= M1, improvement is nontrivial and bounded.",
            "If minimal > operator_proto, residual modes add value beyond operator prior."
        ],
        "outputs": {
            "results": str(res_csv),
            "summary": str(summary_csv),
            "checks": str(checks_csv),
        }
    }

    report_json = OUT_DIR / "sem6a_report.json"
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("SEM-6A finished")
    print("=" * 90)
    print(summary)
    print("\nPass checks:")
    print(checks_df)
    print(f"\nVerdict: {verdict}")
    print(f"Saved: {res_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {checks_csv}")
    print(f"Saved: {report_json}")


if __name__ == "__main__":
    run()
