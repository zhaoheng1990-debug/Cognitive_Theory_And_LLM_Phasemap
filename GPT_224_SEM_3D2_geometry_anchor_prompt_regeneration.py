# -*- coding: utf-8 -*-
"""
GPT_210_SEM_3D2_geometry_anchor_prompt_regeneration.py

SEM-3D.2: GeometryAnchor Prompt Regeneration Audit
--------------------------------------------------

Goal:
    Test whether adding GeometryAnchor to Concept+Operator improves trajectory
    regeneration, especially center_pca / TopK center geometry.

Theory:
    MemoryUnit_F = Concept_F + OperatorID_F + GeometryAnchor_F

SEM-3D.1 produced:
    - sem3d1_anchor_prompt_candidates.csv
    - sem3d1_anchor_ablation_prompts.csv

This script forwards those anchor prompts and compares them to original prompts.

Main tests:
    1. Concept+Operator+Anchor > Concept+Operator
    2. SameFamilyAnchor > RandomFamilyAnchor
    3. Anchor-only / Concept+Anchor / Operator+Anchor ablations
    4. Which prompt form regenerates original trajectory best?

Inputs:
    sem3a_outputs/
      sem3a_features.csv
      sem3a_dataset.csv

    sem3d1_outputs/
      sem3d1_anchor_prompt_candidates.csv
      sem3d1_anchor_ablation_prompts.csv

Outputs:
    sem3d2_outputs/
      sem3d2_config.json
      sem3d2_anchor_prompt_dataset.csv
      sem3d2_anchor_prompt_features.csv
      sem3d2_pairwise_regeneration.csv
      sem3d2_candidate_type_summary.csv
      sem3d2_family_summary.csv
      sem3d2_anchor_ablation_summary.csv
      sem3d2_statistical_tests.csv
      sem3d2_results_summary.json

Run:
    python GPT_210_SEM_3D2_geometry_anchor_prompt_regeneration.py
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Any

import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_rel, wilcoxon


# ============================================================
# CONFIG
# ============================================================

@dataclass
class CFG:
    MODEL_PATH: str = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

    SEM3A_DIR: str = "./sem3a_outputs"
    SEM3D1_DIR: str = "./sem3d1_outputs"
    OUTPUT_DIR: str = "./sem3d2_outputs"

    SEM3A_FEATURES: str = "sem3a_features.csv"
    SEM3A_DATASET: str = "sem3a_dataset.csv"
    ANCHOR_PROMPT_CANDIDATES: str = "sem3d1_anchor_prompt_candidates.csv"
    ANCHOR_ABLATION_PROMPTS: str = "sem3d1_anchor_ablation_prompts.csv"

    TOPK_LIST: Tuple[int, ...] = (100, 500)
    LAYERS: Tuple[int, ...] = tuple(range(0, 26))
    BATCH_SIZE: int = 4
    MAX_LENGTH: int = 192
    CENTER_PCA_DIM: int = 16
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE: str = "auto"
    RANDOM_SEED: int = 42

    MAIN_BLOCK: str = "center_pca"
    REUSE_ANCHOR_FEATURES_IF_EXISTS: bool = True


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
# MODEL + FEATURE EXTRACTION
# ============================================================

def load_model_and_tokenizer(cfg: CFG):
    print(f"[SEM-3D.2] Loading model: {cfg.MODEL_PATH}")
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
        print(f"[SEM-3D.2] Extracting {start}:{end}/{len(prompts)}")
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
# INPUTS / DATASET
# ============================================================

def load_inputs(cfg: CFG):
    sem3a_dir = Path(cfg.SEM3A_DIR)
    sem3d1_dir = Path(cfg.SEM3D1_DIR)

    original_features = pd.read_csv(sem3a_dir / cfg.SEM3A_FEATURES)
    original_dataset = pd.read_csv(sem3a_dir / cfg.SEM3A_DATASET)

    cand = pd.read_csv(sem3d1_dir / cfg.ANCHOR_PROMPT_CANDIDATES)
    abl = pd.read_csv(sem3d1_dir / cfg.ANCHOR_ABLATION_PROMPTS)

    return original_dataset, original_features, cand, abl


def build_anchor_prompt_dataset(cand: pd.DataFrame, abl: pd.DataFrame):
    rows = []
    row_id = 0

    # Candidate prompts from sem3d1_anchor_prompt_candidates.csv
    for _, r in cand.iterrows():
        rows.append({
            "row_id": row_id,
            "row_type": "anchor_candidate",
            "seed_family": r["seed_family"],
            "candidate_type": r["candidate_type"],
            "anchor_family": r.get("anchor_family", r["seed_family"]),
            "ablation_type": "",
            "concept_star": r.get("concept_star", ""),
            "operator_id": r.get("operator_id", ""),
            "anchor_terms": r.get("anchor_terms", ""),
            "common_structure": r.get("common_structure", ""),
            "prompt": r["prompt"],
        })
        row_id += 1

    # Ablation prompts
    for _, r in abl.iterrows():
        rows.append({
            "row_id": row_id,
            "row_type": "anchor_ablation",
            "seed_family": r["seed_family"],
            "candidate_type": f"ablation_{r['ablation_type']}",
            "anchor_family": r["seed_family"] if "anchor" in str(r["ablation_type"]) else "",
            "ablation_type": r["ablation_type"],
            "concept_star": r.get("concept_star", ""),
            "operator_id": r.get("operator_id", ""),
            "anchor_terms": r.get("anchor_terms", ""),
            "common_structure": r.get("common_structure", ""),
            "prompt": r["prompt"],
        })
        row_id += 1

    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["seed_family", "candidate_type", "prompt"]).reset_index(drop=True).assign(row_id=lambda x: np.arange(len(x)))


# ============================================================
# FEATURE BLOCKS
# ============================================================

def get_feature_cols(df: pd.DataFrame, block: str):
    exclude = {
        "row_id", "row_type", "seed_family", "concept_star", "operator_star",
        "surface_id", "prompt", "candidate_type", "anchor_family", "ablation_type",
        "operator_id", "anchor_terms", "common_structure"
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


def align_spaces(original_features, anchor_features, block: str):
    cols = sorted(set(get_feature_cols(original_features, block)) & set(get_feature_cols(anchor_features, block)))
    if not cols:
        return None, None, []
    combined = pd.concat([
        original_features[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0),
        anchor_features[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0),
    ], axis=0)

    scaler = StandardScaler()
    scaler.fit(combined.values.astype(np.float32))

    Xo = scaler.transform(original_features[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32))
    Xa = scaler.transform(anchor_features[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32))
    return Xo, Xa, cols


# ============================================================
# COMPARISON
# ============================================================

def compare_anchor_prompts(original_features, anchor_features, block: str):
    Xo, Xa, cols = align_spaces(original_features, anchor_features, block)
    if Xo is None:
        return pd.DataFrame()

    orig = original_features.reset_index(drop=True)
    anch = anchor_features.reset_index(drop=True)

    idx_by_family_type = {}
    for i, r in anch.iterrows():
        idx_by_family_type[(r["seed_family"], r["candidate_type"])] = i

    all_indices = list(range(len(anch)))
    candidate_types = sorted(anch["candidate_type"].unique())

    rows = []
    for oi, row in orig.iterrows():
        if row.get("row_type") != "original_prompt":
            continue
        true_family = row["seed_family"]
        x = Xo[oi]

        for ctype in candidate_types:
            key = (true_family, ctype)
            if key not in idx_by_family_type:
                continue

            same_idx = idx_by_family_type[key]
            same_vec = Xa[same_idx]
            same_cos = cosine_np(x, same_vec)
            same_l2 = float(np.linalg.norm(x - same_vec))

            # Random family with same candidate_type.
            rand_idxs = [
                i for i, r in anch.iterrows()
                if r["candidate_type"] == ctype and r["seed_family"] != true_family
            ]
            if rand_idxs:
                rand_cos = np.array([cosine_np(x, Xa[i]) for i in rand_idxs])
                rand_l2 = np.array([np.linalg.norm(x - Xa[i]) for i in rand_idxs])
                rand_mean_cos = float(np.mean(rand_cos))
                rand_max_cos = float(np.max(rand_cos))
                rand_mean_l2 = float(np.mean(rand_l2))
                rand_min_l2 = float(np.min(rand_l2))
            else:
                rand_mean_cos = rand_max_cos = rand_mean_l2 = rand_min_l2 = np.nan

            # nearest among same type.
            type_idxs = [i for i, r in anch.iterrows() if r["candidate_type"] == ctype]
            type_cos = np.array([cosine_np(x, Xa[i]) for i in type_idxs])
            nn_idx = type_idxs[int(np.argmax(type_cos))]
            nn_family = anch.iloc[nn_idx]["seed_family"]

            rows.append({
                "row_id": int(row["row_id"]),
                "block": block,
                "candidate_type": ctype,
                "true_family": true_family,
                "original_prompt": row["prompt"],
                "candidate_prompt": anch.iloc[same_idx]["prompt"],
                "anchor_family": anch.iloc[same_idx].get("anchor_family", ""),
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
                "nearest_same_type_family": nn_family,
                "nearest_same_type_correct_family": int(nn_family == true_family),
                "n_features": len(cols),
            })

    return pd.DataFrame(rows)


def summarize_pairwise(pair_df):
    rows = []
    for (block, ctype), g in pair_df.groupby(["block", "candidate_type"]):
        rows.append({
            "block": block,
            "candidate_type": ctype,
            "n": int(len(g)),
            "mean_same_family_cos": float(g["same_family_cos"].mean()),
            "mean_random_family_mean_cos": float(g["random_family_mean_cos"].mean()),
            "mean_cos_margin_vs_random_mean": float(g["cos_margin_vs_random_mean"].mean()),
            "frac_cos_gt_random_mean": float((g["cos_margin_vs_random_mean"] > 0).mean()),
            "frac_cos_gt_random_max": float((g["cos_margin_vs_random_max"] > 0).mean()),
            "mean_same_family_l2": float(g["same_family_l2"].mean()),
            "mean_l2_gain_vs_random_mean": float(g["l2_gain_vs_random_mean"].mean()),
            "frac_l2_better_than_random_mean": float((g["l2_gain_vs_random_mean"] > 0).mean()),
            "nearest_same_type_correct_family_rate": float(g["nearest_same_type_correct_family"].mean()),
        })
    return pd.DataFrame(rows)


def family_summary(pair_df):
    rows = []
    for (block, ctype, fam), g in pair_df.groupby(["block", "candidate_type", "true_family"]):
        rows.append({
            "block": block,
            "candidate_type": ctype,
            "seed_family": fam,
            "n": int(len(g)),
            "mean_cos_margin_vs_random_mean": float(g["cos_margin_vs_random_mean"].mean()),
            "frac_cos_gt_random_mean": float((g["cos_margin_vs_random_mean"] > 0).mean()),
            "mean_l2_gain_vs_random_mean": float(g["l2_gain_vs_random_mean"].mean()),
            "nearest_same_type_correct_family_rate": float(g["nearest_same_type_correct_family"].mean()),
        })
    return pd.DataFrame(rows)


def stats_tests(pair_df):
    rows = []
    for (block, ctype), g in pair_df.groupby(["block", "candidate_type"]):
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
            "candidate_type": ctype,
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


def build_anchor_effect_summary(summary_df):
    """
    Compare key conditions:
      concept_operator_anchor vs ablation_concept_operator
      concept_operator_anchor vs concept_operator_random_anchor
    """
    rows = []
    for block in sorted(summary_df["block"].unique()):
        sub = summary_df[summary_df["block"] == block]
        def val(ctype, field):
            r = sub[sub["candidate_type"] == ctype]
            if len(r):
                return float(r.iloc[0][field])
            return np.nan

        same_anchor = val("concept_operator_anchor", "mean_cos_margin_vs_random_mean")
        no_anchor = val("ablation_concept_operator", "mean_cos_margin_vs_random_mean")
        random_anchor = val("concept_operator_random_anchor", "mean_cos_margin_vs_random_mean")
        anchor_only = val("ablation_anchor_only", "mean_cos_margin_vs_random_mean")
        concept_anchor = val("ablation_concept_anchor", "mean_cos_margin_vs_random_mean")

        rows.append({
            "block": block,
            "same_anchor_margin": same_anchor,
            "no_anchor_margin": no_anchor,
            "random_anchor_margin": random_anchor,
            "anchor_only_margin": anchor_only,
            "concept_anchor_margin": concept_anchor,
            "gain_same_anchor_over_no_anchor": same_anchor - no_anchor if np.isfinite(same_anchor) and np.isfinite(no_anchor) else np.nan,
            "gain_same_anchor_over_random_anchor": same_anchor - random_anchor if np.isfinite(same_anchor) and np.isfinite(random_anchor) else np.nan,
            "gain_concept_anchor_over_no_anchor": concept_anchor - no_anchor if np.isfinite(concept_anchor) and np.isfinite(no_anchor) else np.nan,
        })
    return pd.DataFrame(rows)


def build_results_summary(type_summary, anchor_effect):
    summary = {
        "config": asdict(cfg),
        "main_block": cfg.MAIN_BLOCK,
        "verdict": "UNDETERMINED",
        "best_overall": {},
        "best_main_block": {},
        "anchor_effect_main_block": {},
        "diagnosis": "",
        "interpretation": [],
    }

    if len(type_summary):
        df = type_summary.copy()
        df["score"] = (
            df["mean_cos_margin_vs_random_mean"].fillna(-999)
            + df["frac_cos_gt_random_mean"].fillna(0)
            + df["nearest_same_type_correct_family_rate"].fillna(0)
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

        ae = anchor_effect[anchor_effect["block"] == cfg.MAIN_BLOCK]
        if len(ae):
            summary["anchor_effect_main_block"] = ae.iloc[0].to_dict()

        gain_no = summary["anchor_effect_main_block"].get("gain_same_anchor_over_no_anchor", np.nan)
        gain_rand = summary["anchor_effect_main_block"].get("gain_same_anchor_over_random_anchor", np.nan)
        best_type = main_best.get("candidate_type", "")

        if np.isfinite(gain_no) and gain_no > 0 and np.isfinite(gain_rand) and gain_rand > 0:
            summary["diagnosis"] = "GEOMETRY_ANCHOR_IMPROVES_OVER_CONTROLS"
        elif "anchor" in best_type:
            summary["diagnosis"] = "ANCHOR_TYPE_BEST_BUT_CONTROLS_MIXED"
        else:
            summary["diagnosis"] = "NO_ANCHOR_ADVANTAGE"

        margin = main_best.get("mean_cos_margin_vs_random_mean", -999)
        frac = main_best.get("frac_cos_gt_random_mean", 0)
        nn = main_best.get("nearest_same_type_correct_family_rate", 0)
        l2_gain = main_best.get("mean_l2_gain_vs_random_mean", -999)

        if summary["diagnosis"] == "GEOMETRY_ANCHOR_IMPROVES_OVER_CONTROLS" and margin > 0 and frac >= 0.80 and nn >= 0.50 and l2_gain > 0:
            summary["verdict"] = "PASS_STRONG_GEOMETRY_ANCHOR"
        elif margin > 0 and frac >= 0.70 and gain_no > 0:
            summary["verdict"] = "PASS_LITE_GEOMETRY_ANCHOR"
        elif margin > 0 and frac >= 0.60:
            summary["verdict"] = "PARTIAL_PASS_ANCHOR_SIGNAL"
        else:
            summary["verdict"] = "NO_PASS_YET"

    summary["interpretation"] = [
        "Primary test: Concept+Operator+GeometryAnchor should beat Concept+Operator and random-family anchor.",
        "If same-family anchor beats random anchor, GeometryAnchor carries family-specific geometry.",
        "If anchor-only is strong, GeometryAnchor may be more important than Concept/Operator for this feature block.",
        "If no anchor advantage appears, anchor extraction may be lexical rather than geometric, requiring TopK-token-derived anchors."
    ]
    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(cfg.RANDOM_SEED)
    outdir = Path(cfg.OUTPUT_DIR)
    ensure_dir(str(outdir))
    json_dump(asdict(cfg), outdir / "sem3d2_config.json")

    original_dataset, original_features, cand, abl = load_inputs(cfg)
    anchor_dataset = build_anchor_prompt_dataset(cand, abl)
    anchor_dataset.to_csv(outdir / "sem3d2_anchor_prompt_dataset.csv", index=False, encoding="utf-8-sig")

    feat_path = outdir / "sem3d2_anchor_prompt_features.csv"
    if cfg.REUSE_ANCHOR_FEATURES_IF_EXISTS and feat_path.exists():
        print(f"[SEM-3D.2] Reusing anchor prompt features: {feat_path}")
        anchor_features = pd.read_csv(feat_path)
    else:
        model, tokenizer = load_model_and_tokenizer(cfg)
        anchor_features = extract_all_features(anchor_dataset, model, tokenizer, cfg)
        anchor_features.to_csv(feat_path, index=False, encoding="utf-8-sig")

    original_features = original_features[original_features["row_type"] == "original_prompt"].copy()

    blocks = ["center_pca", "decision", "mid", "all", "scalar", "transport", "init"]
    all_pair = []
    for block in blocks:
        print(f"[SEM-3D.2] Comparing block={block}")
        p = compare_anchor_prompts(original_features, anchor_features, block)
        if len(p):
            all_pair.append(p)

    pair_df = pd.concat(all_pair, ignore_index=True) if all_pair else pd.DataFrame()
    pair_df.to_csv(outdir / "sem3d2_pairwise_regeneration.csv", index=False, encoding="utf-8-sig")

    type_summary = summarize_pairwise(pair_df) if len(pair_df) else pd.DataFrame()
    type_summary.to_csv(outdir / "sem3d2_candidate_type_summary.csv", index=False, encoding="utf-8-sig")

    fam = family_summary(pair_df) if len(pair_df) else pd.DataFrame()
    fam.to_csv(outdir / "sem3d2_family_summary.csv", index=False, encoding="utf-8-sig")

    anchor_effect = build_anchor_effect_summary(type_summary) if len(type_summary) else pd.DataFrame()
    anchor_effect.to_csv(outdir / "sem3d2_anchor_ablation_summary.csv", index=False, encoding="utf-8-sig")

    stats = stats_tests(pair_df) if len(pair_df) else pd.DataFrame()
    stats.to_csv(outdir / "sem3d2_statistical_tests.csv", index=False, encoding="utf-8-sig")

    summary = build_results_summary(type_summary, anchor_effect)
    json_dump(summary, outdir / "sem3d2_results_summary.json")

    print("=" * 100)
    print("SEM-3D.2: GeometryAnchor Prompt Regeneration Audit")
    print("=" * 100)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nOutput files:")
    for p in [
        "sem3d2_config.json",
        "sem3d2_anchor_prompt_dataset.csv",
        "sem3d2_anchor_prompt_features.csv",
        "sem3d2_pairwise_regeneration.csv",
        "sem3d2_candidate_type_summary.csv",
        "sem3d2_family_summary.csv",
        "sem3d2_anchor_ablation_summary.csv",
        "sem3d2_statistical_tests.csv",
        "sem3d2_results_summary.json",
    ]:
        print("  -", outdir / p)


if __name__ == "__main__":
    main()
