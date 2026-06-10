# -*- coding: utf-8 -*-
"""
SEM-5C.6: Family Residual Decomposition Audit

Motivation
----------
SEM-5C.4/5C.5 showed:
    mu_shape is strongly operator-transferable.
    synthetic(concept-commit + operator-shape) works.
    But oracle_shape still beats synthetic/same_operator_shape.

Therefore SEM-5C.6 tests:
    mu_F_shape = OperatorPrototype_operator + FamilyResidual_F

Core question:
    Is the oracle > operator-transfer gap explained by a family-specific residual?

Method
------
For each row/family F with operator O:

1. Build operator prototype P_O:
       mean shape of families with same operator, excluding current family.

2. Define family residual:
       r_F = mu_F_shape - P_O
       normalized as optional direction.

3. Inject conditions:
       raw
       operator_proto_shape
       operator_proto_plus_oracle_residual
       operator_proto_plus_shuffle_residual
       oracle_shape
       shuffle_shape

Expected:
    oracle_shape >= operator_proto_plus_oracle_residual > operator_proto_shape > shuffle
    operator_proto_plus_oracle_residual > operator_proto_plus_shuffle_residual

If true:
    mu_shape decomposes into:
        OperatorPrior + FamilySpecificResidual

Outputs:
sem5c6_outputs/
    sem5c6_residual_decomposition_results.csv
    sem5c6_summary.csv
    sem5c6_pass_checks.csv
    sem5c6_report.json
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

OUT_DIR = Path("sem5c6_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SHAPE_LAYERS = list(range(7, 20))
COMMIT_LAYERS = list(range(23, 26))

MAX_LENGTH = 512
MAX_ROWS = None

ALPHAS = [0.10, 0.20, 0.30]

# residual mixing coefficient
RESIDUAL_BETAS = [0.5, 1.0]


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

def make_operator_prototype(bank: Dict[str, Dict], operator: str, exclude_family: str):
    candidates = [
        mem["shape"] for fam, mem in bank.items()
        if fam != exclude_family and str(mem.get("operator")) == str(operator)
    ]
    if not candidates:
        candidates = [
            mem["shape"] for fam, mem in bank.items()
            if fam != exclude_family
        ]
    return normalize(np.mean(np.stack(candidates, axis=0), axis=0))


def make_random_residual(bank: Dict[str, Dict], operator: str, exclude_family: str):
    fams = [fam for fam in bank.keys() if fam != exclude_family]
    if not fams:
        fams = list(bank.keys())
    fam = random.choice(fams)
    mem = bank[fam]
    proto = make_operator_prototype(bank, str(mem.get("operator")), fam)
    residual = mem["shape"] - proto
    return residual.astype(np.float32), fam


def compose_shape(proto: np.ndarray, residual: np.ndarray, beta: float):
    return normalize(proto + beta * residual)


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
    print("SEM-5C.6 Family Residual Decomposition Audit")
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

    # Stage 1: raw
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

    # Stage 2: residual decomposition injection
    print("[Stage 2] Residual decomposition injection...")
    rows = []

    for rec_i, rec in enumerate(raw_records):
        oracle_family = str(rec["family"])
        operator = str(rec["operator"])

        if external_bank is not None and oracle_family in external_bank:
            bank = external_bank
        else:
            bank = build_live_memory_bank(raw_records, exclude_row_idx=rec["row_idx"])

        if oracle_family not in bank:
            print(f"[WARN] oracle family missing: {oracle_family}; skip")
            continue

        oracle_shape = bank[oracle_family]["shape"]
        operator_proto = make_operator_prototype(bank, operator, oracle_family)
        oracle_residual = (oracle_shape - operator_proto).astype(np.float32)
        shuffle_residual, shuffle_resid_family = make_random_residual(bank, operator, oracle_family)

        # random shuffle shape
        non_oracle_fams = [fam for fam in bank.keys() if fam != oracle_family] or list(bank.keys())
        shuffle_shape_family = random.choice(non_oracle_fams)
        shuffle_shape = bank[shuffle_shape_family]["shape"]

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

        base_conditions = {
            "raw": None,
            "operator_proto_shape": operator_proto,
            "oracle_shape": oracle_shape,
            "shuffle_shape": shuffle_shape,
        }

        composed_conditions = {}
        for beta in RESIDUAL_BETAS:
            composed_conditions[f"proto_plus_oracle_residual_b{beta}"] = compose_shape(operator_proto, oracle_residual, beta)
            composed_conditions[f"proto_plus_shuffle_residual_b{beta}"] = compose_shape(operator_proto, shuffle_residual, beta)

        all_conditions = {**base_conditions, **composed_conditions}

        for cond, vec in all_conditions.items():
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
                    "shuffle_residual_family": shuffle_resid_family if "shuffle_residual" in cond else None,
                    "shuffle_shape_family": shuffle_shape_family if cond == "shuffle_shape" else None,

                    "residual_norm_oracle": float(np.linalg.norm(oracle_residual)),
                    "residual_norm_shuffle": float(np.linalg.norm(shuffle_residual)),
                    "operator_proto_to_oracle_cos": cosine(operator_proto, oracle_shape),

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
    res_csv = OUT_DIR / "sem5c6_residual_decomposition_results.csv"
    res_df.to_csv(res_csv, index=False, encoding="utf-8-sig")

    summary = (
        res_df.groupby(["condition", "alpha"], dropna=False)
        .agg(
            n=("row_idx", "count"),
            mean_operator_proto_to_oracle_cos=("operator_proto_to_oracle_cos", "mean"),
            mean_residual_norm_oracle=("residual_norm_oracle", "mean"),
            mean_residual_norm_shuffle=("residual_norm_shuffle", "mean"),
            gain_shape_sim=("gain_shape_sim", "mean"),
            gain_margin_shape=("gain_margin_shape", "mean"),
            rank_improvement_shape=("rank_improvement_shape", "mean"),
            gain_commit_sim=("gain_commit_sim", "mean"),
            gain_margin_commit=("gain_margin_commit", "mean"),
            rank_improvement_commit=("rank_improvement_commit", "mean"),
        )
        .reset_index()
    )

    summary_csv = OUT_DIR / "sem5c6_summary.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    # Pass checks
    checks = []
    for alpha in ALPHAS:
        sub = summary[summary["alpha"] == alpha]

        def val(cond, col):
            v = sub[sub["condition"] == cond][col]
            return float(v.iloc[0]) if len(v) else np.nan

        proto = val("operator_proto_shape", "rank_improvement_shape")
        oracle = val("oracle_shape", "rank_improvement_shape")
        shuffle = val("shuffle_shape", "rank_improvement_shape")

        for beta in RESIDUAL_BETAS:
            oracle_res = val(f"proto_plus_oracle_residual_b{beta}", "rank_improvement_shape")
            shuffle_res = val(f"proto_plus_shuffle_residual_b{beta}", "rank_improvement_shape")

            checks.append({
                "check": f"oracle_residual_gain_alpha_{alpha}_beta_{beta}",
                "value": oracle_res,
                "operator_proto": proto,
                "shuffle_residual": shuffle_res,
                "oracle_shape": oracle,
                "pass_lite": bool(oracle_res > proto),
                "pass_strong": bool((oracle_res > proto) and (oracle_res > shuffle_res) and (oracle >= oracle_res)),
                "note": "Oracle family residual should improve operator prototype more than shuffled residual."
            })

        checks.append({
            "check": f"operator_proto_vs_shuffle_alpha_{alpha}",
            "value": proto,
            "shuffle_shape": shuffle,
            "oracle_shape": oracle,
            "pass_lite": bool(proto > shuffle),
            "pass_strong": bool((proto > shuffle) and (oracle >= proto)),
            "note": "Operator prototype should beat random shape and remain below oracle."
        })

    checks_df = pd.DataFrame(checks)
    checks_csv = OUT_DIR / "sem5c6_pass_checks.csv"
    checks_df.to_csv(checks_csv, index=False, encoding="utf-8-sig")

    verdict = "FAIL"
    if checks_df["pass_lite"].any():
        verdict = "PASS-Lite"

    residual_strong = checks_df[checks_df["check"].str.startswith("oracle_residual_gain")]["pass_strong"].sum()
    proto_strong = checks_df[checks_df["check"].str.startswith("operator_proto_vs_shuffle")]["pass_strong"].sum()

    if residual_strong >= 2 and proto_strong >= 2:
        verdict = "PASS-Strong"
    elif residual_strong >= 2:
        verdict = "PASS-FamilyResidual"

    report = {
        "experiment": "SEM-5C.6 Family Residual Decomposition Audit",
        "verdict": verdict,
        "motivation": [
            "SEM-5C.4/5C.5 suggest mu_shape contains an operator-transferable prior plus family-specific excess.",
            "This audit decomposes shape into operator prototype and family residual."
        ],
        "interpretation_rules": [
            "If operator_proto_shape > shuffle_shape, operator-level prior is confirmed.",
            "If proto_plus_oracle_residual > operator_proto_shape, family residual has additional utility.",
            "If proto_plus_oracle_residual > proto_plus_shuffle_residual, residual is family-specific rather than generic noise.",
            "If oracle_shape >= proto_plus_oracle_residual, oracle memory remains the upper bound."
        ],
        "outputs": {
            "results": str(res_csv),
            "summary": str(summary_csv),
            "checks": str(checks_csv),
        }
    }

    report_json = OUT_DIR / "sem5c6_report.json"
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("SEM-5C.6 finished")
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
