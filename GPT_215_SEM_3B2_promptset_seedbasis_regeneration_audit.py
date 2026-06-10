# -*- coding: utf-8 -*-
"""
GPT_206_SEM_3B2_promptset_seedbasis_regeneration_audit.py

SEM-3B.2: PromptSet / SeedBasis Regeneration Audit
--------------------------------------------------

Goal:
    Test whether a SeedFamily center μ_F is better realized by a small PromptSet*
    rather than a single Prompt*.

Motivation:
    SEM-3B and SEM-3B.1 showed:
      - single generated Prompt* does not regenerate center_pca trajectory well
      - μF-nearest exemplar has scalar signal but still fails center geometry
    Hypothesis:
      μ_F is a distribution center, not necessarily realizable by one prompt.
      It may require a small PromptSet / SeedBasis.

Candidate PromptSet strategies:
    1. basis_nearest_natural_structured:
       {nearest_exemplar, natural_prompt_star, structured_prompt_star}

    2. basis_nearest_natural_compressed:
       {nearest_exemplar, natural_prompt_star, compressed_prompt_star}

    3. basis_nearest_plus_structure_natural:
       {nearest_plus_structure, natural_prompt_star, structured_prompt_star}

    4. basis_all_promptstars:
       {natural_prompt_star, structured_prompt_star, compressed_prompt_star}

    5. basis_nearest_only:
       {nearest_exemplar}

    6. basis_canonical_only:
       {canonical_minimal_seed}

For each family:
    PromptSet trajectory representation = mean(feature vectors of prompts in set)

For each original prompt:
    compare T(original) to:
      - same-family PromptSet center
      - random-family PromptSet centers

Outputs:
    sem3b2_outputs/
      sem3b2_seedbasis_dataset.csv
      sem3b2_seedbasis_features.csv
      sem3b2_promptset_centers.csv
      sem3b2_pairwise_regeneration.csv
      sem3b2_strategy_summary.csv
      sem3b2_family_summary.csv
      sem3b2_statistical_tests.csv
      sem3b2_results_summary.json

Run:
    python GPT_206_SEM_3B2_promptset_seedbasis_regeneration_audit.py
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import ttest_rel, wilcoxon


# ============================================================
# CONFIG
# ============================================================

@dataclass
class CFG:
    MODEL_PATH: str = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

    SEM3A_DIR: str = "./sem3a_outputs"
    SEM3C0B_DIR: str = "./sem3c0b_outputs"
    SEM3B1_DIR: str = "./sem3b1_outputs"
    OUTPUT_DIR: str = "./sem3b2_outputs"

    SEM3A_DATASET: str = "sem3a_dataset.csv"
    SEM3A_FEATURES: str = "sem3a_features.csv"
    PROMPTSTAR_CANDIDATES: str = "sem3c0b_prompt_star_candidates.csv"
    MEMBER_RESIDUALS: str = "sem3c0b_family_member_residuals.csv"
    NEAREST_EXEMPLARS: str = "sem3b1_nearest_exemplars.csv"

    TOPK_LIST: Tuple[int, ...] = (100, 500)
    LAYERS: Tuple[int, ...] = tuple(range(0, 26))
    BATCH_SIZE: int = 4
    MAX_LENGTH: int = 192
    CENTER_PCA_DIM: int = 16
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE: str = "auto"
    RANDOM_SEED: int = 42

    MAIN_BLOCK: str = "center_pca"
    REUSE_SEEDBASIS_FEATURES_IF_EXISTS: bool = True

    # Include candidates from SEM-3B.1 if already extracted.
    REUSE_SEM3B1_CANDIDATE_FEATURES_IF_EXISTS: bool = True
    SEM3B1_CANDIDATE_FEATURES: str = "sem3b1_candidate_features.csv"
    SEM3B1_CANDIDATE_DATASET: str = "sem3b1_candidate_dataset.csv"


cfg = CFG()


# ============================================================
# UTILS
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
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=convert)


def get_torch_dtype(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    return "auto"


def cosine_np(a, b, eps=1e-9):
    a = np.asarray(a)
    b = np.asarray(b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def jaccard(a: set, b: set):
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def load_model_and_tokenizer(cfg: CFG):
    print(f"[SEM-3B.2] Loading model: {cfg.MODEL_PATH}")
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
    raise RuntimeError("Cannot find output embedding / lm_head weight.")


@torch.no_grad()
def extract_batch(prompts: List[str], model, tokenizer, cfg: CFG):
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
    last_pos = enc["attention_mask"].sum(dim=1) - 1

    scalar_rows = []
    center_records = []

    for bi in range(len(prompts)):
        row = {}
        prev_center = {}
        prev_set = {}

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
                p = f"k{k}_L{layer}"

                row[f"{p}_spread"] = spread
                row[f"{p}_entropy"] = entropy
                row[f"{p}_gap"] = float(vals_np[0] - vals_np[-1])
                row[f"{p}_meanlogit"] = float(vals_np.mean())
                row[f"{p}_center_norm"] = float(np.linalg.norm(center_np))

                if k in prev_center:
                    row[f"{p}_center_shift_cosdist"] = 1.0 - cosine_np(prev_center[k], center_np)
                    row[f"{p}_center_shift_l2"] = float(np.linalg.norm(center_np - prev_center[k]))
                else:
                    row[f"{p}_center_shift_cosdist"] = 0.0
                    row[f"{p}_center_shift_l2"] = 0.0

                if k in prev_set:
                    jac = jaccard(prev_set[k], id_set)
                    row[f"{p}_jaccard_prev"] = jac
                    row[f"{p}_jaccard_dist_prev"] = 1.0 - jac
                else:
                    row[f"{p}_jaccard_prev"] = 1.0
                    row[f"{p}_jaccard_dist_prev"] = 0.0

                prev_center[k] = center_np
                prev_set[k] = id_set

                center_records.append({
                    "batch_index": bi,
                    "layer": layer,
                    "k": k,
                    "center": center_np,
                })

        scalar_rows.append(row)

    return scalar_rows, center_records


def extract_all_features(df: pd.DataFrame, model, tokenizer, cfg: CFG):
    prompts = df["prompt"].astype(str).tolist()
    all_scalar = []
    all_center_records = []

    for start in range(0, len(prompts), cfg.BATCH_SIZE):
        end = min(start + cfg.BATCH_SIZE, len(prompts))
        print(f"[SEM-3B.2] Extracting {start}:{end}/{len(prompts)}")
        scalar_rows, center_records = extract_batch(prompts[start:end], model, tokenizer, cfg)

        for i, r in enumerate(scalar_rows):
            r["row_id"] = int(df.iloc[start + i]["row_id"])
            all_scalar.append(r)

        for rec in center_records:
            rec["row_id"] = int(df.iloc[start + rec["batch_index"]]["row_id"])
            del rec["batch_index"]
            all_center_records.append(rec)

    scalar_df = pd.DataFrame(all_scalar)

    center_feature_rows = {int(rid): {"row_id": int(rid)} for rid in df["row_id"].values}
    for k in cfg.TOPK_LIST:
        for layer in cfg.LAYERS:
            recs = [r for r in all_center_records if r["k"] == k and r["layer"] == layer]
            if not recs:
                continue
            X = np.vstack([r["center"] for r in recs])
            n_comp = min(cfg.CENTER_PCA_DIM, X.shape[0], X.shape[1])
            pca = PCA(n_components=n_comp, random_state=cfg.RANDOM_SEED)
            Z = pca.fit_transform(StandardScaler().fit_transform(X))
            for rec, z in zip(recs, Z):
                rid = int(rec["row_id"])
                for j in range(n_comp):
                    center_feature_rows[rid][f"k{k}_L{layer}_centerPC{j+1}"] = float(z[j])

    center_df = pd.DataFrame(list(center_feature_rows.values())).fillna(0.0)
    return df.merge(scalar_df, on="row_id", how="left").merge(center_df, on="row_id", how="left")


# ============================================================
# INPUTS / SEEDBASIS DATASET
# ============================================================

def load_inputs(cfg: CFG):
    sem3a_dir = Path(cfg.SEM3A_DIR)
    sem3c_dir = Path(cfg.SEM3C0B_DIR)
    sem3b1_dir = Path(cfg.SEM3B1_DIR)

    sem3a_dataset = pd.read_csv(sem3a_dir / cfg.SEM3A_DATASET)
    sem3a_features = pd.read_csv(sem3a_dir / cfg.SEM3A_FEATURES)
    promptstars = pd.read_csv(sem3c_dir / cfg.PROMPTSTAR_CANDIDATES)

    nearest_path = sem3b1_dir / cfg.NEAREST_EXEMPLARS
    if nearest_path.exists():
        nearest = pd.read_csv(nearest_path)
    else:
        # fallback from 3C0b residuals
        mr = pd.read_csv(sem3c_dir / cfg.MEMBER_RESIDUALS)
        mr = mr[mr["block"] == cfg.MAIN_BLOCK]
        rows = []
        for fam, g in mr.groupby("seed_family"):
            best = g.sort_values("residual_cos_to_center", ascending=False).iloc[0]
            rows.append({
                "seed_family": fam,
                "source_row_id": int(best["row_id"]),
                "nearest_exemplar_prompt": best["prompt"],
            })
        nearest = pd.DataFrame(rows)

    sem3b1_candidate_dataset_path = sem3b1_dir / cfg.SEM3B1_CANDIDATE_DATASET
    sem3b1_candidate_features_path = sem3b1_dir / cfg.SEM3B1_CANDIDATE_FEATURES

    sem3b1_candidate_dataset = None
    sem3b1_candidate_features = None
    if sem3b1_candidate_dataset_path.exists() and sem3b1_candidate_features_path.exists():
        sem3b1_candidate_dataset = pd.read_csv(sem3b1_candidate_dataset_path)
        sem3b1_candidate_features = pd.read_csv(sem3b1_candidate_features_path)

    return sem3a_dataset, sem3a_features, promptstars, nearest, sem3b1_candidate_dataset, sem3b1_candidate_features


def get_promptstar_prompt(promptstars: pd.DataFrame, family: str, template_type: str):
    r = promptstars[(promptstars["seed_family"] == family) & (promptstars["template_type"] == template_type)]
    if len(r):
        return str(r.iloc[0]["prompt_star_candidate"])
    return ""


def get_promptstar_meta(promptstars: pd.DataFrame, family: str):
    r = promptstars[promptstars["seed_family"] == family]
    if len(r):
        r0 = r.iloc[0]
        return {
            "concept_star": r0.get("concept_star", ""),
            "operator_id": r0.get("operator_id", ""),
            "common_structure": r0.get("common_structure", ""),
        }
    return {"concept_star": "", "operator_id": "", "common_structure": ""}


def build_seedbasis_dataset(sem3a_dataset, promptstars, nearest):
    """
    Build unique prompt rows needed for all promptsets.
    """
    rows = []
    row_id = 0

    families = sorted(sem3a_dataset["seed_family"].unique())

    for fam in families:
        meta = get_promptstar_meta(promptstars, fam)

        # Canonical minimal seed
        canon = sem3a_dataset[(sem3a_dataset["seed_family"] == fam) & (sem3a_dataset["row_type"] == "canonical_minimal_seed")]
        if len(canon):
            rows.append({
                "row_id": row_id,
                "row_type": "basis_prompt",
                "seed_family": fam,
                "basis_prompt_type": "canonical_minimal_seed",
                "source_row_id": int(canon.iloc[0]["row_id"]),
                "prompt": canon.iloc[0]["prompt"],
                **meta,
            })
            row_id += 1

        # Promptstar variants
        for ttype in ["natural_prompt_star", "structured_prompt_star", "compressed_prompt_star"]:
            prompt = get_promptstar_prompt(promptstars, fam, ttype)
            if prompt:
                rows.append({
                    "row_id": row_id,
                    "row_type": "basis_prompt",
                    "seed_family": fam,
                    "basis_prompt_type": ttype,
                    "source_row_id": -1,
                    "prompt": prompt,
                    **meta,
                })
                row_id += 1

        # nearest exemplar
        nr = nearest[nearest["seed_family"] == fam]
        if len(nr):
            nearest_prompt = nr.iloc[0].get("nearest_exemplar_prompt", "")
            source_id = int(nr.iloc[0].get("source_row_id", -1))
            rows.append({
                "row_id": row_id,
                "row_type": "basis_prompt",
                "seed_family": fam,
                "basis_prompt_type": "nearest_exemplar",
                "source_row_id": source_id,
                "prompt": nearest_prompt,
                **meta,
            })
            row_id += 1

            # nearest plus structure
            common = meta.get("common_structure", "")
            operator = meta.get("operator_id", "")
            distilled = f"{nearest_prompt}\n\nPreserve the shared structure: {common}.\nUse operator: {operator}."
            rows.append({
                "row_id": row_id,
                "row_type": "basis_prompt",
                "seed_family": fam,
                "basis_prompt_type": "nearest_plus_structure",
                "source_row_id": source_id,
                "prompt": distilled,
                **meta,
            })
            row_id += 1

    return pd.DataFrame(rows)


def define_promptsets():
    return {
        "basis_nearest_only": ["nearest_exemplar"],
        "basis_canonical_only": ["canonical_minimal_seed"],
        "basis_all_promptstars": ["natural_prompt_star", "structured_prompt_star", "compressed_prompt_star"],
        "basis_nearest_natural_structured": ["nearest_exemplar", "natural_prompt_star", "structured_prompt_star"],
        "basis_nearest_natural_compressed": ["nearest_exemplar", "natural_prompt_star", "compressed_prompt_star"],
        "basis_nearest_plus_structure_natural": ["nearest_plus_structure", "natural_prompt_star", "structured_prompt_star"],
        "basis_full_six": ["nearest_exemplar", "nearest_plus_structure", "natural_prompt_star", "structured_prompt_star", "compressed_prompt_star", "canonical_minimal_seed"],
    }


# ============================================================
# FEATURE BLOCKS
# ============================================================

def get_feature_cols(df: pd.DataFrame, block: str):
    exclude = {
        "row_id", "row_type", "seed_family", "concept_star", "operator_star",
        "surface_id", "prompt", "basis_prompt_type", "source_row_id",
        "operator_id", "common_structure"
    }
    numeric = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    if block == "all":
        return numeric
    if block == "scalar":
        return [c for c in numeric if "centerPC" not in c]
    if block == "center_pca":
        return [c for c in numeric if "centerPC" in c]
    if block == "transport":
        return [c for c in numeric if any(k in c for k in ["shift", "jaccard"])]
    if block == "init":
        return [c for c in numeric if any(f"_L{i}_" in c for i in range(0, 7))]
    if block == "mid":
        return [c for c in numeric if any(f"_L{i}_" in c for i in range(7, 20))]
    if block == "decision":
        return [c for c in numeric if any(f"_L{i}_" in c for i in range(20, 26))]
    raise ValueError(block)


def align_spaces(original_features, basis_features, block: str):
    cols = sorted(set(get_feature_cols(original_features, block)) & set(get_feature_cols(basis_features, block)))
    if not cols:
        return None, None, []

    combined = pd.concat([
        original_features[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0),
        basis_features[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0),
    ], axis=0)

    scaler = StandardScaler()
    scaler.fit(combined.values.astype(np.float32))

    Xo = scaler.transform(original_features[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32))
    Xb = scaler.transform(basis_features[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32))
    return Xo, Xb, cols


def build_promptset_centers(basis_features: pd.DataFrame, Xb: np.ndarray, promptsets: Dict[str, List[str]]):
    basis = basis_features.reset_index(drop=True)
    rows = []
    vectors = {}

    for fam in sorted(basis["seed_family"].unique()):
        for set_name, types in promptsets.items():
            idxs = [
                i for i, r in basis.iterrows()
                if r["seed_family"] == fam and r["basis_prompt_type"] in types
            ]
            if not idxs:
                continue
            center = Xb[idxs, :].mean(axis=0)
            vectors[(fam, set_name)] = center
            rows.append({
                "seed_family": fam,
                "promptset_name": set_name,
                "basis_types": "|".join(types),
                "n_basis": int(len(idxs)),
                "basis_row_ids": json.dumps([int(basis.iloc[i]["row_id"]) for i in idxs]),
                "basis_prompts": json.dumps([basis.iloc[i]["prompt"] for i in idxs], ensure_ascii=False),
                "center_vector_json": json.dumps(center.tolist()),
            })

    return pd.DataFrame(rows), vectors


# ============================================================
# COMPARISON
# ============================================================

def compare_promptsets(original_features, basis_features, block: str):
    Xo, Xb, cols = align_spaces(original_features, basis_features, block)
    if Xo is None:
        return pd.DataFrame(), pd.DataFrame()

    orig = original_features.reset_index(drop=True)
    promptsets = define_promptsets()
    centers_df, vectors = build_promptset_centers(basis_features, Xb, promptsets)

    rows = []
    all_keys = list(vectors.keys())

    for oi, row in orig.iterrows():
        if row.get("row_type") != "original_prompt":
            continue

        true_family = row["seed_family"]
        x = Xo[oi]

        for set_name in promptsets.keys():
            key = (true_family, set_name)
            if key not in vectors:
                continue
            same_vec = vectors[key]
            same_cos = cosine_np(x, same_vec)
            same_l2 = float(np.linalg.norm(x - same_vec))

            rand_keys = [(fam, sn) for (fam, sn) in all_keys if sn == set_name and fam != true_family]
            if rand_keys:
                rand_cos = np.array([cosine_np(x, vectors[k]) for k in rand_keys])
                rand_l2 = np.array([np.linalg.norm(x - vectors[k]) for k in rand_keys])
                rand_mean_cos = float(np.mean(rand_cos))
                rand_max_cos = float(np.max(rand_cos))
                rand_mean_l2 = float(np.mean(rand_l2))
                rand_min_l2 = float(np.min(rand_l2))
            else:
                rand_mean_cos = rand_max_cos = rand_mean_l2 = rand_min_l2 = np.nan

            # nearest promptset within same set_name
            type_keys = [(fam, sn) for (fam, sn) in all_keys if sn == set_name]
            type_cos = np.array([cosine_np(x, vectors[k]) for k in type_keys])
            nn_idx = int(np.argmax(type_cos))
            nn_family = type_keys[nn_idx][0]

            rows.append({
                "row_id": int(row["row_id"]),
                "block": block,
                "promptset_name": set_name,
                "true_family": true_family,
                "original_prompt": row["prompt"],
                "same_family_cos": same_cos,
                "same_family_l2": same_l2,
                "random_family_mean_cos": rand_mean_cos,
                "random_family_max_cos": rand_max_cos,
                "random_family_mean_l2": rand_mean_l2,
                "random_family_min_l2": rand_min_l2,
                "cos_margin_vs_random_mean": same_cos - rand_mean_cos if np.isfinite(rand_mean_cos) else np.nan,
                "cos_margin_vs_random_max": same_cos - rand_max_cos if np.isfinite(rand_max_cos) else np.nan,
                "l2_gain_vs_random_mean": rand_mean_l2 - same_l2 if np.isfinite(rand_mean_l2) else np.nan,
                "l2_gain_vs_random_min": rand_min_l2 - same_l2 if np.isfinite(rand_min_l2) else np.nan,
                "nearest_promptset_family": nn_family,
                "nearest_promptset_correct_family": int(nn_family == true_family),
                "n_features": len(cols),
            })

    return pd.DataFrame(rows), centers_df


def summarize_pairwise(pair_df: pd.DataFrame):
    rows = []
    for (block, set_name), g in pair_df.groupby(["block", "promptset_name"]):
        rows.append({
            "block": block,
            "promptset_name": set_name,
            "n": int(len(g)),
            "mean_same_family_cos": float(g["same_family_cos"].mean()),
            "mean_random_family_mean_cos": float(g["random_family_mean_cos"].mean()),
            "mean_cos_margin_vs_random_mean": float(g["cos_margin_vs_random_mean"].mean()),
            "frac_cos_gt_random_mean": float((g["cos_margin_vs_random_mean"] > 0).mean()),
            "frac_cos_gt_random_max": float((g["cos_margin_vs_random_max"] > 0).mean()),
            "mean_same_family_l2": float(g["same_family_l2"].mean()),
            "mean_l2_gain_vs_random_mean": float(g["l2_gain_vs_random_mean"].mean()),
            "frac_l2_better_than_random_mean": float((g["l2_gain_vs_random_mean"] > 0).mean()),
            "nearest_promptset_correct_family_rate": float(g["nearest_promptset_correct_family"].mean()),
        })
    return pd.DataFrame(rows)


def family_summary(pair_df: pd.DataFrame):
    rows = []
    for (block, set_name, fam), g in pair_df.groupby(["block", "promptset_name", "true_family"]):
        rows.append({
            "block": block,
            "promptset_name": set_name,
            "seed_family": fam,
            "n": int(len(g)),
            "mean_cos_margin_vs_random_mean": float(g["cos_margin_vs_random_mean"].mean()),
            "frac_cos_gt_random_mean": float((g["cos_margin_vs_random_mean"] > 0).mean()),
            "mean_l2_gain_vs_random_mean": float(g["l2_gain_vs_random_mean"].mean()),
            "nearest_promptset_correct_family_rate": float(g["nearest_promptset_correct_family"].mean()),
        })
    return pd.DataFrame(rows)


def stats_tests(pair_df: pd.DataFrame):
    rows = []
    for (block, set_name), g in pair_df.groupby(["block", "promptset_name"]):
        cos_same = g["same_family_cos"].values
        cos_rand = g["random_family_mean_cos"].values
        l2_same = g["same_family_l2"].values
        l2_rand = g["random_family_mean_l2"].values

        try:
            tt = ttest_rel(cos_same, cos_rand, nan_policy="omit")
            t_stat, t_p = float(tt.statistic), float(tt.pvalue)
        except Exception:
            t_stat, t_p = np.nan, np.nan
        try:
            ww = wilcoxon(cos_same - cos_rand)
            w_stat, w_p = float(ww.statistic), float(ww.pvalue)
        except Exception:
            w_stat, w_p = np.nan, np.nan
        try:
            tt2 = ttest_rel(l2_rand, l2_same, nan_policy="omit")
            t2_stat, t2_p = float(tt2.statistic), float(tt2.pvalue)
        except Exception:
            t2_stat, t2_p = np.nan, np.nan

        rows.append({
            "block": block,
            "promptset_name": set_name,
            "n": int(len(g)),
            "cos_margin_mean": float(np.nanmean(cos_same - cos_rand)),
            "cos_paired_t_stat": t_stat,
            "cos_paired_t_p": t_p,
            "cos_wilcoxon_stat": w_stat,
            "cos_wilcoxon_p": w_p,
            "l2_gain_mean": float(np.nanmean(l2_rand - l2_same)),
            "l2_paired_t_stat": t2_stat,
            "l2_paired_t_p": t2_p,
        })
    return pd.DataFrame(rows)


def build_summary(strategy_df: pd.DataFrame):
    summary = {
        "config": asdict(cfg),
        "verdict": "UNDETERMINED",
        "main_block": cfg.MAIN_BLOCK,
        "best_overall": {},
        "best_main_block": {},
        "diagnosis": "",
        "interpretation": [],
    }

    if len(strategy_df):
        df = strategy_df.copy()
        df["score"] = (
            df["mean_cos_margin_vs_random_mean"].fillna(-999)
            + df["frac_cos_gt_random_mean"].fillna(0)
            + df["nearest_promptset_correct_family_rate"].fillna(0)
            + 0.1 * df["mean_l2_gain_vs_random_mean"].fillna(0)
        )
        best = df.sort_values("score", ascending=False).iloc[0].to_dict()
        summary["best_overall"] = best

        main = df[df["block"] == cfg.MAIN_BLOCK]
        if len(main):
            main_best = main.sort_values("score", ascending=False).iloc[0].to_dict()
            summary["best_main_block"] = main_best
        else:
            main_best = best

        # Diagnosis
        def get(block, set_name, field):
            sub = df[(df["block"] == block) & (df["promptset_name"] == set_name)]
            if len(sub):
                return float(sub.iloc[0].get(field, np.nan))
            return np.nan

        mb = cfg.MAIN_BLOCK
        nearest = get(mb, "basis_nearest_only", "mean_cos_margin_vs_random_mean")
        canonical = get(mb, "basis_canonical_only", "mean_cos_margin_vs_random_mean")
        all_promptstars = get(mb, "basis_all_promptstars", "mean_cos_margin_vs_random_mean")
        nearest_mix = get(mb, "basis_nearest_natural_structured", "mean_cos_margin_vs_random_mean")
        full_six = get(mb, "basis_full_six", "mean_cos_margin_vs_random_mean")

        vals = {
            "nearest": nearest,
            "canonical": canonical,
            "all_promptstars": all_promptstars,
            "nearest_mix": nearest_mix,
            "full_six": full_six,
        }
        finite_vals = {k: v for k, v in vals.items() if np.isfinite(v)}
        if finite_vals:
            winner = max(finite_vals, key=finite_vals.get)
            if winner == "nearest":
                summary["diagnosis"] = "SINGLE_NEAREST_EXEMPLAR_BEST"
            elif winner in ["nearest_mix", "full_six"]:
                summary["diagnosis"] = "PROMPTSET_BASIS_IMPROVES_OVER_SINGLE_PROMPT"
            elif winner == "all_promptstars":
                summary["diagnosis"] = "GENERATED_PROMPTSTAR_SET_BEST"
            elif winner == "canonical":
                summary["diagnosis"] = "CANONICAL_SEED_BEST"
            else:
                summary["diagnosis"] = "NO_CLEAR_WINNER"
        else:
            summary["diagnosis"] = "NO_CLEAR_WINNER"

        margin = main_best.get("mean_cos_margin_vs_random_mean", -999)
        frac = main_best.get("frac_cos_gt_random_mean", 0)
        nn = main_best.get("nearest_promptset_correct_family_rate", 0)
        l2_gain = main_best.get("mean_l2_gain_vs_random_mean", -999)

        if margin > 0 and frac >= 0.85 and nn >= 0.70 and l2_gain > 0:
            summary["verdict"] = "PASS_STRONG_PROMPTSET_REGENERATION"
        elif margin > 0 and frac >= 0.70 and nn >= 0.50:
            summary["verdict"] = "PASS_LITE_PROMPTSET_REGENERATION"
        elif margin > 0 and frac >= 0.60:
            summary["verdict"] = "PARTIAL_PASS_PROMPTSET_SIGNAL"
        else:
            summary["verdict"] = "NO_PASS_YET"

    summary["interpretation"] = [
        "PromptSet center is the mean of feature vectors from multiple basis prompts.",
        "If promptset basis beats nearest-only, μF is better realized by a set rather than a single prompt.",
        "If nearest-only remains best, the best available executable seed is a real exemplar prompt.",
        "If all promptsets fail in center_pca, μF may not be directly executable by small prompt bases under current encoding.",
    ]
    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(cfg.RANDOM_SEED)
    outdir = Path(cfg.OUTPUT_DIR)
    ensure_dir(str(outdir))
    json_dump(asdict(cfg), outdir / "sem3b2_config.json")

    sem3a_dataset, sem3a_features, promptstars, nearest, sem3b1_dataset, sem3b1_features = load_inputs(cfg)

    seedbasis_dataset = build_seedbasis_dataset(sem3a_dataset, promptstars, nearest)
    seedbasis_dataset.to_csv(outdir / "sem3b2_seedbasis_dataset.csv", index=False, encoding="utf-8-sig")

    feat_path = outdir / "sem3b2_seedbasis_features.csv"
    if cfg.REUSE_SEEDBASIS_FEATURES_IF_EXISTS and feat_path.exists():
        print(f"[SEM-3B.2] Reusing seedbasis features: {feat_path}")
        seedbasis_features = pd.read_csv(feat_path)
    else:
        model, tokenizer = load_model_and_tokenizer(cfg)
        seedbasis_features = extract_all_features(seedbasis_dataset, model, tokenizer, cfg)
        seedbasis_features.to_csv(feat_path, index=False, encoding="utf-8-sig")

    original_features = sem3a_features[sem3a_features["row_type"] == "original_prompt"].copy()

    blocks = ["center_pca", "decision", "mid", "all", "scalar", "transport", "init"]
    all_pair = []
    all_centers = []

    for block in blocks:
        print(f"[SEM-3B.2] Comparing promptsets block={block}")
        pair_df, centers_df = compare_promptsets(original_features, seedbasis_features, block)
        if len(pair_df):
            all_pair.append(pair_df)
        if len(centers_df):
            centers_df["block"] = block
            all_centers.append(centers_df)

    pair = pd.concat(all_pair, ignore_index=True) if all_pair else pd.DataFrame()
    centers = pd.concat(all_centers, ignore_index=True) if all_centers else pd.DataFrame()

    pair.to_csv(outdir / "sem3b2_pairwise_regeneration.csv", index=False, encoding="utf-8-sig")
    centers.to_csv(outdir / "sem3b2_promptset_centers.csv", index=False, encoding="utf-8-sig")

    strategy = summarize_pairwise(pair) if len(pair) else pd.DataFrame()
    strategy.to_csv(outdir / "sem3b2_strategy_summary.csv", index=False, encoding="utf-8-sig")

    fam = family_summary(pair) if len(pair) else pd.DataFrame()
    fam.to_csv(outdir / "sem3b2_family_summary.csv", index=False, encoding="utf-8-sig")

    stats = stats_tests(pair) if len(pair) else pd.DataFrame()
    stats.to_csv(outdir / "sem3b2_statistical_tests.csv", index=False, encoding="utf-8-sig")

    summary = build_summary(strategy)
    json_dump(summary, outdir / "sem3b2_results_summary.json")

    print("=" * 100)
    print("SEM-3B.2: PromptSet / SeedBasis Regeneration Audit")
    print("=" * 100)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nOutput files:")
    for p in [
        "sem3b2_config.json",
        "sem3b2_seedbasis_dataset.csv",
        "sem3b2_seedbasis_features.csv",
        "sem3b2_promptset_centers.csv",
        "sem3b2_pairwise_regeneration.csv",
        "sem3b2_strategy_summary.csv",
        "sem3b2_family_summary.csv",
        "sem3b2_statistical_tests.csv",
        "sem3b2_results_summary.json",
    ]:
        print("  -", outdir / p)


if __name__ == "__main__":
    main()
