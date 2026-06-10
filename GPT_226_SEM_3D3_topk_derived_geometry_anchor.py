# -*- coding: utf-8 -*-
"""
GPT_211_SEM_3D3_topk_derived_geometry_anchor.py

SEM-3D.3: TopK-derived GeometryAnchor Extraction
------------------------------------------------

Goal:
    Extract GeometryAnchor from the model's own TopK/VIM semantic neighborhood,
    not from prompt text.

Motivation:
    SEM-3D.2 showed lexical anchor terms do NOT improve center_pca regeneration.
    Therefore:
        GeometryAnchor_text != GeometryAnchor_geo

Hypothesis:
    Real GeometryAnchor should come from:
        TopK(H_l W^T)
    i.e. model-induced vocabulary-neighborhood prototypes.

This script:
    1. For SEM-3A prompts, forward Qwen and save TopK token identities.
    2. For each SeedFamily, compute token frequency and specificity.
    3. Extract family-specific TopK-derived anchor tokens.
    4. Generate anchor prompt candidates for SEM-3D.4.
    5. Audit whether TopK anchors are layer-specific, family-specific, and not just lexical echoes.

Inputs:
    sem3a_outputs/
      sem3a_dataset.csv

Outputs:
    sem3d3_outputs/
      sem3d3_config.json
      sem3d3_topk_tokens_long.csv
      sem3d3_topk_anchor_scores.csv
      sem3d3_family_topk_anchor_summary.csv
      sem3d3_topk_anchor_prompt_candidates.csv
      sem3d3_topk_anchor_ablation_prompts.csv
      sem3d3_layer_anchor_diagnostics.csv
      sem3d3_results_summary.json

Run:
    python GPT_211_SEM_3D3_topk_derived_geometry_anchor.py

Next:
    SEM-3D.4 should forward TopKAnchor prompts and test:
      Concept+Operator+TopKAnchor > Concept+Operator
      SameFamilyTopKAnchor > RandomFamilyTopKAnchor
      TopKAnchor > LexicalAnchor
"""

import json
import math
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# CONFIG
# ============================================================

@dataclass
class CFG:
    MODEL_PATH: str = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

    SEM3A_DIR: str = "./sem3a_outputs"
    OUTPUT_DIR: str = "./sem3d3_outputs"

    SEM3A_DATASET: str = "sem3a_dataset.csv"

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE: str = "auto"
    BATCH_SIZE: int = 4
    MAX_LENGTH: int = 192

    # We save TopK identities from multiple windows.
    LAYERS: Tuple[int, ...] = tuple(range(0, 26))
    TOPK_LIST: Tuple[int, ...] = (50, 100, 500)

    # Anchor extraction settings.
    TOP_N_ANCHORS: int = 48
    TOP_N_PROMPT_ANCHORS: int = 16
    EPS: float = 1e-6
    ALPHA_OTHER_FREQ: float = 0.35

    # Windows for layer diagnostics.
    WINDOWS: Dict[str, Tuple[int, ...]] = None

    # Prefer original prompts for family anchors; canonical rows can be included optionally.
    USE_CANONICAL_FOR_ANCHOR: bool = False

    RANDOM_SEED: int = 42
    REUSE_TOPK_IF_EXISTS: bool = True


def make_cfg():
    cfg = CFG()
    cfg.WINDOWS = {
        "init_L0_6": tuple(range(0, 7)),
        "mid_L7_19": tuple(range(7, 20)),
        "decision_L20_25": tuple(range(20, 26)),
        "full_L0_25": tuple(range(0, 26)),
    }
    return cfg


cfg = make_cfg()


# ============================================================
# STRUCTURAL PRIOR MAP
# ============================================================

def seed_family_structure_map() -> Dict[str, Dict[str, str]]:
    return {
        "inertia_mechanism": {
            "concept_star": "inertia / persistent motion",
            "operator_id": "mechanism_explanation",
            "domain_frame": "classical_physics",
            "common_structure": "state of motion persists unless acted on by a net external force",
        },
        "photosynthesis_mechanism": {
            "concept_star": "photosynthesis",
            "operator_id": "mechanism_explanation",
            "domain_frame": "biology",
            "common_structure": "plants convert light, carbon dioxide, and water into chemical energy",
        },
        "democracy_definition": {
            "concept_star": "democracy",
            "operator_id": "definition",
            "domain_frame": "politics",
            "common_structure": "political system involving citizen participation, representation, and collective rule",
        },
        "gravity_causal": {
            "concept_star": "gravity",
            "operator_id": "causal_explanation",
            "domain_frame": "physics",
            "common_structure": "mass attracts mass and shapes falling and orbital motion",
        },
        "python_use_cases": {
            "concept_star": "Python programming language",
            "operator_id": "list_use_cases",
            "domain_frame": "software",
            "common_structure": "common practical applications and domains of Python",
        },
        "internet_risk": {
            "concept_star": "internet",
            "operator_id": "risk_audit",
            "domain_frame": "technology_society",
            "common_structure": "risks, vulnerabilities, and failure modes of internet systems and use",
        },
        "climate_plan": {
            "concept_star": "climate change",
            "operator_id": "planning",
            "domain_frame": "environment_policy",
            "common_structure": "practical mitigation and adaptation strategy",
        },
        "transformer_mechanism": {
            "concept_star": "Transformer model",
            "operator_id": "mechanism_explanation",
            "domain_frame": "machine_learning",
            "common_structure": "attention, token representations, and layered computation",
        },
        "market_compare": {
            "concept_star": "market coordination",
            "operator_id": "comparison",
            "domain_frame": "economics",
            "common_structure": "compare decentralized market coordination with centralized planning",
        },
        "memory_counterfactual": {
            "concept_star": "memory",
            "operator_id": "counterfactual",
            "domain_frame": "cognition",
            "common_structure": "consequences of absent, altered, or unreliable memory",
        },
        "ocean_property": {
            "concept_star": "ocean",
            "operator_id": "property_description",
            "domain_frame": "earth_science",
            "common_structure": "major physical and ecological properties of oceans",
        },
        "robot_relation": {
            "concept_star": "robot",
            "operator_id": "relation_mapping",
            "domain_frame": "robotics",
            "common_structure": "relation among robots, automation, sensors, control, and machines",
        },
    }


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


def clean_token(tok: str) -> str:
    """
    Keep token readable but remove common Qwen/SentencePiece artifacts.
    """
    tok = str(tok)
    tok = tok.replace("Ġ", " ")
    tok = tok.replace("▁", " ")
    tok = tok.replace("Ċ", "\\n")
    tok = tok.strip()
    # Avoid very noisy empty strings.
    if not tok:
        return ""
    return tok


def is_bad_anchor_token(tok: str) -> bool:
    t = tok.strip().lower()
    if not t:
        return True
    # Pure punctuation / markup / control-like pieces.
    if all(ch in ".,;:!?()[]{}<>-_=+*/\\|`'\" \n\t" for ch in t):
        return True
    if len(t) == 1 and not t.isalnum():
        return True
    # Common instruction/filler fragments likely not useful as geometry anchors.
    bad = {
        "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "be",
        "what", "why", "how", "explain", "describe", "give", "list", "define",
        "concept", "operator", "answer", "prompt", "task", "using", "use"
    }
    if t in bad:
        return True
    return False


# ============================================================
# LOAD MODEL / DATASET
# ============================================================

def load_dataset(cfg: CFG):
    path = Path(cfg.SEM3A_DIR) / cfg.SEM3A_DATASET
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    df = pd.read_csv(path)
    return df


def load_model_and_tokenizer(cfg: CFG):
    print(f"[SEM-3D.3] Loading model: {cfg.MODEL_PATH}")
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


# ============================================================
# TOPK EXTRACTION
# ============================================================

@torch.no_grad()
def extract_batch_topk(batch_df: pd.DataFrame, model, tokenizer, cfg: CFG):
    prompts = batch_df["prompt"].astype(str).tolist()
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

    rows = []

    for bi in range(len(prompts)):
        meta = batch_df.iloc[bi]
        for layer in cfg.LAYERS:
            hs_index = layer + 1
            if hs_index >= len(hidden_states):
                continue

            h = hidden_states[hs_index][bi, last_pos[bi], :]
            logits = torch.matmul(h, W.T)

            max_k = max(cfg.TOPK_LIST)
            vals, ids = torch.topk(logits, k=max_k, dim=-1)
            vals_np = vals.detach().float().cpu().numpy()
            ids_np = ids.detach().cpu().numpy().astype(int)

            decoded = [clean_token(tokenizer.decode([int(tid)])) for tid in ids_np]

            for k in cfg.TOPK_LIST:
                for rank in range(k):
                    tok_id = int(ids_np[rank])
                    tok = decoded[rank]
                    if is_bad_anchor_token(tok):
                        continue
                    rows.append({
                        "row_id": int(meta["row_id"]),
                        "row_type": meta.get("row_type", ""),
                        "seed_family": meta["seed_family"],
                        "concept_star": meta.get("concept_star", ""),
                        "operator_star": meta.get("operator_star", ""),
                        "surface_id": int(meta.get("surface_id", -999)),
                        "layer": int(layer),
                        "topk": int(k),
                        "rank": int(rank + 1),
                        "token_id": tok_id,
                        "token": tok,
                        "logit": float(vals_np[rank]),
                    })

    return rows


def extract_all_topk(dataset: pd.DataFrame, model, tokenizer, cfg: CFG):
    # Optionally only original prompts.
    if cfg.USE_CANONICAL_FOR_ANCHOR:
        df = dataset.copy()
    else:
        df = dataset[dataset["row_type"] == "original_prompt"].copy()

    rows = []
    for start in range(0, len(df), cfg.BATCH_SIZE):
        end = min(start + cfg.BATCH_SIZE, len(df))
        print(f"[SEM-3D.3] Extracting TopK {start}:{end}/{len(df)}")
        rows.extend(extract_batch_topk(df.iloc[start:end].reset_index(drop=True), model, tokenizer, cfg))

    return pd.DataFrame(rows)


# ============================================================
# ANCHOR SCORING
# ============================================================

def compute_topk_anchor_scores(topk_df: pd.DataFrame, cfg: CFG):
    """
    Score token anchors by family-specific frequency and rank/logit weighting.

    Each token occurrence weight:
        rank_weight = 1 / log2(rank+1)
        logit_weight normalized implicitly by occurrence only; keep mean logit as diagnostic.
    """
    if topk_df.empty:
        return pd.DataFrame()

    # Add windows.
    def layer_to_window(layer: int):
        wins = []
        for w, layers in cfg.WINDOWS.items():
            if layer in layers:
                wins.append(w)
        return wins

    # Expanded rows by window.
    rows = []
    for _, r in topk_df.iterrows():
        rank_weight = 1.0 / math.log2(float(r["rank"]) + 1.0)
        for wname in layer_to_window(int(r["layer"])):
            rows.append({
                "seed_family": r["seed_family"],
                "window": wname,
                "topk": int(r["topk"]),
                "token_id": int(r["token_id"]),
                "token": r["token"],
                "rank_weight": rank_weight,
                "logit": float(r["logit"]),
            })

    x = pd.DataFrame(rows)
    families = sorted(x["seed_family"].unique())
    all_rows = []

    # Compute by window and k.
    for (window, k), sub in x.groupby(["window", "topk"]):
        # weighted counts
        fam_token_weight = defaultdict(lambda: defaultdict(float))
        fam_token_count = defaultdict(lambda: defaultdict(int))
        fam_token_logit = defaultdict(lambda: defaultdict(list))
        global_token_weight = defaultdict(float)
        global_token_count = defaultdict(int)

        for _, r in sub.iterrows():
            fam = r["seed_family"]
            key = (int(r["token_id"]), str(r["token"]))
            w = float(r["rank_weight"])
            fam_token_weight[fam][key] += w
            fam_token_count[fam][key] += 1
            fam_token_logit[fam][key].append(float(r["logit"]))
            global_token_weight[key] += w
            global_token_count[key] += 1

        global_total_weight = sum(global_token_weight.values()) + cfg.EPS

        for fam in families:
            fam_total_weight = sum(fam_token_weight[fam].values()) + cfg.EPS
            other_total_weight = global_total_weight - fam_total_weight + cfg.EPS

            for key, fw in fam_token_weight[fam].items():
                ow = global_token_weight[key] - fw
                freq_f = fw / fam_total_weight
                freq_other = ow / other_total_weight
                ratio = (freq_f + cfg.EPS) / (freq_other + cfg.EPS)
                log_ratio = math.log(ratio)
                diff = freq_f - cfg.ALPHA_OTHER_FREQ * freq_other
                anchor_score = diff * (1.0 + max(0.0, log_ratio))

                token_id, token = key
                all_rows.append({
                    "seed_family": fam,
                    "window": window,
                    "topk": int(k),
                    "token_id": int(token_id),
                    "token": token,
                    "weighted_count_family": float(fw),
                    "raw_count_family": int(fam_token_count[fam][key]),
                    "freq_family": float(freq_f),
                    "weighted_count_other": float(ow),
                    "freq_other": float(freq_other),
                    "specificity_ratio": float(ratio),
                    "log_specificity_ratio": float(log_ratio),
                    "anchor_score": float(anchor_score),
                    "mean_logit_family": float(np.mean(fam_token_logit[fam][key])),
                })

    scores = pd.DataFrame(all_rows)
    if len(scores):
        scores = scores.sort_values(
            ["seed_family", "window", "topk", "anchor_score"],
            ascending=[True, True, True, False]
        )
    return scores


def select_anchor_terms(scores: pd.DataFrame, cfg: CFG):
    rows = []
    if scores.empty:
        return pd.DataFrame()

    for (fam, window, k), g in scores.groupby(["seed_family", "window", "topk"]):
        g = g.sort_values("anchor_score", ascending=False).head(cfg.TOP_N_ANCHORS)
        for rank, (_, r) in enumerate(g.iterrows(), start=1):
            rows.append({**r.to_dict(), "anchor_rank": rank})
    return pd.DataFrame(rows)


def build_layer_anchor_diagnostics(scores: pd.DataFrame):
    if scores.empty:
        return pd.DataFrame()

    rows = []
    for (window, topk), g in scores.groupby(["window", "topk"]):
        rows.append({
            "window": window,
            "topk": int(topk),
            "n_anchor_candidates": int(len(g)),
            "mean_anchor_score": float(g["anchor_score"].mean()),
            "median_anchor_score": float(g["anchor_score"].median()),
            "mean_specificity_ratio": float(g["specificity_ratio"].mean()),
            "p95_anchor_score": float(g["anchor_score"].quantile(0.95)),
        })
    return pd.DataFrame(rows)


# ============================================================
# PROMPT CANDIDATES
# ============================================================

def build_family_summary(anchor_terms: pd.DataFrame, cfg: CFG):
    fmap = seed_family_structure_map()
    rows = []
    # Prefer full window k100 for prompt anchors; fallback to largest available.
    for fam in sorted(anchor_terms["seed_family"].unique()):
        subset = anchor_terms[
            (anchor_terms["seed_family"] == fam) &
            (anchor_terms["window"] == "full_L0_25") &
            (anchor_terms["topk"] == 100)
        ]
        if len(subset) == 0:
            subset = anchor_terms[anchor_terms["seed_family"] == fam].sort_values("anchor_score", ascending=False)

        top = subset.sort_values("anchor_score", ascending=False).head(cfg.TOP_N_PROMPT_ANCHORS)
        terms = top["token"].astype(str).tolist()
        token_ids = top["token_id"].astype(int).tolist()

        meta = fmap.get(fam, {})
        rows.append({
            "seed_family": fam,
            "concept_star": meta.get("concept_star", fam),
            "operator_id": meta.get("operator_id", ""),
            "domain_frame": meta.get("domain_frame", ""),
            "common_structure": meta.get("common_structure", ""),
            "topk_anchor_terms": ", ".join(terms),
            "topk_anchor_token_ids": ",".join(map(str, token_ids)),
            "topk_anchor_terms_json": json.dumps(terms, ensure_ascii=False),
            "topk_anchor_token_ids_json": json.dumps(token_ids),
            "mean_anchor_score": float(top["anchor_score"].mean()) if len(top) else np.nan,
            "mean_specificity_ratio": float(top["specificity_ratio"].mean()) if len(top) else np.nan,
        })

    return pd.DataFrame(rows)


def build_topk_anchor_prompt_candidates(summary: pd.DataFrame):
    rows = []
    families = summary["seed_family"].tolist()
    fam_to_anchor = {r["seed_family"]: r["topk_anchor_terms"] for _, r in summary.iterrows()}

    for _, r in summary.iterrows():
        fam = r["seed_family"]
        concept = r.get("concept_star", fam)
        operator = r.get("operator_id", "")
        common = r.get("common_structure", "")
        anchors = r.get("topk_anchor_terms", "")

        rows.append({
            "seed_family": fam,
            "candidate_type": "concept_operator_topk_anchor",
            "anchor_family": fam,
            "prompt": (
                f"Concept: {concept}\n"
                f"OperatorID: {operator}\n"
                f"TopKGeometryAnchor: {anchors}\n"
                f"Use this model-derived semantic neighborhood to answer. Preserve: {common}."
            ),
            "concept_star": concept,
            "operator_id": operator,
            "anchor_terms": anchors,
            "common_structure": common,
        })

        rows.append({
            "seed_family": fam,
            "candidate_type": "natural_topk_anchor_prompt",
            "anchor_family": fam,
            "prompt": (
                f"Use the semantic neighborhood suggested by these model-derived anchors: {anchors}. "
                f"Explain {concept} with operation {operator}. Shared structure: {common}."
            ),
            "concept_star": concept,
            "operator_id": operator,
            "anchor_terms": anchors,
            "common_structure": common,
        })

        rows.append({
            "seed_family": fam,
            "candidate_type": "topk_anchor_only",
            "anchor_family": fam,
            "prompt": f"TopKGeometryAnchor: {anchors}\nExplain the shared structure represented by this semantic neighborhood.",
            "concept_star": concept,
            "operator_id": operator,
            "anchor_terms": anchors,
            "common_structure": common,
        })

        rows.append({
            "seed_family": fam,
            "candidate_type": "concept_operator_no_topk_anchor",
            "anchor_family": "",
            "prompt": f"Concept: {concept}\nOperatorID: {operator}\nAnswer according to this concept and operation.",
            "concept_star": concept,
            "operator_id": operator,
            "anchor_terms": "",
            "common_structure": common,
        })

        # Random-family anchor control.
        if len(families) > 1:
            idx = families.index(fam)
            rand_fam = families[(idx + 1) % len(families)]
            rand_anchor = fam_to_anchor[rand_fam]
            rows.append({
                "seed_family": fam,
                "candidate_type": "concept_operator_random_topk_anchor",
                "anchor_family": rand_fam,
                "prompt": (
                    f"Concept: {concept}\n"
                    f"OperatorID: {operator}\n"
                    f"TopKGeometryAnchor: {rand_anchor}\n"
                    f"Use this model-derived semantic neighborhood to answer. Preserve: {common}."
                ),
                "concept_star": concept,
                "operator_id": operator,
                "anchor_terms": rand_anchor,
                "common_structure": common,
            })

    return pd.DataFrame(rows)


def build_topk_anchor_ablation_prompts(summary: pd.DataFrame):
    rows = []
    for _, r in summary.iterrows():
        fam = r["seed_family"]
        concept = r.get("concept_star", fam)
        operator = r.get("operator_id", "")
        common = r.get("common_structure", "")
        anchors = r.get("topk_anchor_terms", "")

        ablations = {
            "concept_only": f"Concept: {concept}\nAnswer about this concept.",
            "operator_only": f"OperatorID: {operator}\nApply this operation to an appropriate concept.",
            "topk_anchor_only": f"TopKGeometryAnchor: {anchors}\nAnswer using this model-derived semantic neighborhood.",
            "concept_operator": f"Concept: {concept}\nOperatorID: {operator}\nAnswer accordingly.",
            "concept_topk_anchor": f"Concept: {concept}\nTopKGeometryAnchor: {anchors}\nAnswer using this model-derived semantic neighborhood.",
            "operator_topk_anchor": f"OperatorID: {operator}\nTopKGeometryAnchor: {anchors}\nAnswer using this model-derived semantic neighborhood.",
            "concept_operator_topk_anchor": (
                f"Concept: {concept}\nOperatorID: {operator}\nTopKGeometryAnchor: {anchors}\n"
                f"Preserve the shared structure: {common}."
            ),
        }
        for name, prompt in ablations.items():
            rows.append({
                "seed_family": fam,
                "ablation_type": name,
                "prompt": prompt,
                "concept_star": concept,
                "operator_id": operator,
                "anchor_terms": anchors,
                "common_structure": common,
            })
    return pd.DataFrame(rows)


# ============================================================
# SUMMARY
# ============================================================

def build_results_summary(anchor_terms, scores, family_summary, prompt_candidates, ablations, diagnostics):
    result = {
        "config": asdict(cfg),
        "n_families": int(family_summary["seed_family"].nunique()) if len(family_summary) else 0,
        "n_topk_anchor_terms": int(len(anchor_terms)),
        "n_score_rows": int(len(scores)),
        "n_prompt_candidates": int(len(prompt_candidates)),
        "n_ablation_prompts": int(len(ablations)),
        "verdict": "TOPK_ANCHOR_CANDIDATES_READY",
        "interpretation": [
            "TopK-derived anchors are extracted from model-induced vocabulary neighborhoods TopK(H_l W^T), not from prompt text.",
            "SEM-3D.4 should forward these TopK anchor prompts and compare them against lexical anchors and random-family anchors.",
            "If TopK anchors improve center_pca regeneration while lexical anchors did not, GeometryAnchor_geo is supported.",
        ],
        "top_anchor_preview": {},
        "diagnostics": {},
    }

    if len(family_summary):
        for _, r in family_summary.iterrows():
            result["top_anchor_preview"][r["seed_family"]] = json.loads(r["topk_anchor_terms_json"])

        result["diagnostics"]["mean_anchor_score"] = float(family_summary["mean_anchor_score"].mean())
        result["diagnostics"]["mean_specificity_ratio"] = float(family_summary["mean_specificity_ratio"].mean())

    if len(diagnostics):
        result["diagnostics"]["window_summary"] = diagnostics.to_dict(orient="records")

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(cfg.RANDOM_SEED)
    outdir = Path(cfg.OUTPUT_DIR)
    ensure_dir(str(outdir))
    json_dump(asdict(cfg), outdir / "sem3d3_config.json")

    dataset = load_dataset(cfg)
    topk_path = outdir / "sem3d3_topk_tokens_long.csv"

    if cfg.REUSE_TOPK_IF_EXISTS and topk_path.exists():
        print(f"[SEM-3D.3] Reusing TopK tokens: {topk_path}")
        topk_df = pd.read_csv(topk_path)
    else:
        model, tokenizer = load_model_and_tokenizer(cfg)
        topk_df = extract_all_topk(dataset, model, tokenizer, cfg)
        topk_df.to_csv(topk_path, index=False, encoding="utf-8-sig")

    scores = compute_topk_anchor_scores(topk_df, cfg)
    anchors = select_anchor_terms(scores, cfg)
    diagnostics = build_layer_anchor_diagnostics(scores)
    family_summary = build_family_summary(anchors, cfg)
    prompt_candidates = build_topk_anchor_prompt_candidates(family_summary)
    ablations = build_topk_anchor_ablation_prompts(family_summary)

    scores.to_csv(outdir / "sem3d3_topk_anchor_scores.csv", index=False, encoding="utf-8-sig")
    anchors.to_csv(outdir / "sem3d3_anchor_terms.csv", index=False, encoding="utf-8-sig")
    family_summary.to_csv(outdir / "sem3d3_family_topk_anchor_summary.csv", index=False, encoding="utf-8-sig")
    prompt_candidates.to_csv(outdir / "sem3d3_topk_anchor_prompt_candidates.csv", index=False, encoding="utf-8-sig")
    ablations.to_csv(outdir / "sem3d3_topk_anchor_ablation_prompts.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(outdir / "sem3d3_layer_anchor_diagnostics.csv", index=False, encoding="utf-8-sig")

    summary = build_results_summary(anchors, scores, family_summary, prompt_candidates, ablations, diagnostics)
    json_dump(summary, outdir / "sem3d3_results_summary.json")

    print("=" * 100)
    print("SEM-3D.3: TopK-derived GeometryAnchor Extraction")
    print("=" * 100)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nOutput files:")
    for p in [
        "sem3d3_config.json",
        "sem3d3_topk_tokens_long.csv",
        "sem3d3_topk_anchor_scores.csv",
        "sem3d3_anchor_terms.csv",
        "sem3d3_family_topk_anchor_summary.csv",
        "sem3d3_topk_anchor_prompt_candidates.csv",
        "sem3d3_topk_anchor_ablation_prompts.csv",
        "sem3d3_layer_anchor_diagnostics.csv",
        "sem3d3_results_summary.json",
    ]:
        print("  -", outdir / p)


if __name__ == "__main__":
    main()
