# -*- coding: utf-8 -*-
"""
GPT_213_SEM_3D3C_semantic_projection_filtering.py

SEM-3D.3c: Semantic Projection Filtering
----------------------------------------

Goal:
    Convert CleanTopKAnchor into SemanticGeometryAnchor.

Why:
    SEM-3D.3b removed explicit bad tokens, but top anchors were still polluted by:
      - completion artifacts
      - instruction tokens
      - corpus/template fragments
      - subword artifacts
      - high-specificity but low-semantic relevance tokens

Core idea:
    GeometryAnchor_geo =
        TopKSpecificity
        ∩ SemanticProjection
        ∩ ArtifactQuotient

This script:
    1. Loads clean TopK anchor scores from SEM-3D.3b.
    2. Loads local Qwen tokenizer/model embedding W.
    3. Builds family prior semantic centroids in W-space.
    4. Scores each candidate token by:
        clean_anchor_score
        semantic cosine to family prior centroid
        prior lexical overlap / subword match
        content quality
        artifact penalty
    5. Selects SemanticGeometryAnchor candidates.
    6. Generates semantic-anchor prompt candidates for SEM-3D.4.

Inputs:
    sem3d3b_outputs/
      sem3d3b_clean_anchor_scores.csv
      sem3d3b_clean_anchor_terms.csv
      sem3d3b_family_clean_anchor_summary.csv

Outputs:
    sem3d3c_outputs/
      sem3d3c_config.json
      sem3d3c_semantic_anchor_scores.csv
      sem3d3c_semantic_anchor_terms.csv
      sem3d3c_family_semantic_anchor_summary.csv
      sem3d3c_semantic_anchor_prompt_candidates.csv
      sem3d3c_semantic_anchor_ablation_prompts.csv
      sem3d3c_filter_audit.csv
      sem3d3c_results_summary.json

Run:
    python GPT_213_SEM_3D3C_semantic_projection_filtering.py

Next:
    SEM-3D.4 should compare:
      semantic_topk_anchor
      clean_topk_anchor
      lexical_anchor
      no_anchor
      random_anchor
"""

import json
import re
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple

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

    INPUT_DIR: str = "./sem3d3b_outputs"
    OUTPUT_DIR: str = "./sem3d3c_outputs"

    CLEAN_SCORES: str = "sem3d3b_clean_anchor_scores.csv"
    CLEAN_ANCHORS: str = "sem3d3b_clean_anchor_terms.csv"
    CLEAN_FAMILY_SUMMARY: str = "sem3d3b_family_clean_anchor_summary.csv"

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE: str = "auto"

    # Candidate windows/k
    PREFERRED_WINDOWS: Tuple[str, ...] = ("init_L0_6", "mid_L7_19", "decision_L20_25")
    PREFERRED_TOPK: Tuple[int, ...] = (50, 100)

    TOP_N_SEMANTIC_ANCHORS: int = 24
    TOP_N_PROMPT_ANCHORS: int = 12

    # Semantic projection thresholds. Loose by default because subword tokenization is noisy.
    MIN_SEMANTIC_COS: float = 0.03
    MIN_FINAL_SCORE: float = 0.0

    # Weighting
    W_ANCHOR: float = 1.0
    W_SEMANTIC: float = 2.5
    W_PRIOR: float = 1.5
    W_CONTENT: float = 0.4
    W_SPECIFICITY: float = 0.05

    # If true, require at least this fraction of selected prompt anchors to match prior.
    TARGET_PRIOR_MATCH_FRAC: float = 0.45

    RANDOM_SEED: int = 42


cfg = CFG()


# ============================================================
# STRUCTURAL PRIOR
# ============================================================

def seed_family_structure_map() -> Dict[str, Dict[str, str]]:
    return {
        "inertia_mechanism": {
            "concept_star": "inertia / persistent motion",
            "operator_id": "mechanism_explanation",
            "domain_frame": "classical_physics",
            "common_structure": "state of motion persists unless acted on by a net external force",
            "prior_terms": "inertia inertial motion force rest uniform external object state mechanics Newton law acceleration velocity momentum body bodies net force",
        },
        "photosynthesis_mechanism": {
            "concept_star": "photosynthesis",
            "operator_id": "mechanism_explanation",
            "domain_frame": "biology",
            "common_structure": "plants convert light, carbon dioxide, and water into chemical energy",
            "prior_terms": "photosynthesis plant plants light sunlight carbon dioxide water chlorophyll sugar glucose chemical energy leaves oxygen solar green",
        },
        "democracy_definition": {
            "concept_star": "democracy",
            "operator_id": "definition",
            "domain_frame": "politics",
            "common_structure": "political system involving citizen participation, representation, and collective rule",
            "prior_terms": "democracy democratic government citizen citizens participation representation rule voting people society political system rights majority collective",
        },
        "gravity_causal": {
            "concept_star": "gravity",
            "operator_id": "causal_explanation",
            "domain_frame": "physics",
            "common_structure": "mass attracts mass and shapes falling and orbital motion",
            "prior_terms": "gravity gravitational gravitation mass attraction attracts force fall falling earth orbit planets stars motion acceleration spacetime object objects",
        },
        "python_use_cases": {
            "concept_star": "Python programming language",
            "operator_id": "list_use_cases",
            "domain_frame": "software",
            "common_structure": "common practical applications and domains of Python",
            "prior_terms": "Python programming language software data analysis automation scripting machine learning web development code libraries script applications uses",
        },
        "internet_risk": {
            "concept_star": "internet",
            "operator_id": "risk_audit",
            "domain_frame": "technology_society",
            "common_structure": "risks, vulnerabilities, and failure modes of internet systems and use",
            "prior_terms": "internet online network cybersecurity privacy security vulnerability vulnerabilities risk risks malware phishing connectivity misinformation systems users data",
        },
        "climate_plan": {
            "concept_star": "climate change",
            "operator_id": "planning",
            "domain_frame": "environment_policy",
            "common_structure": "practical mitigation and adaptation strategy",
            "prior_terms": "climate change mitigation adaptation emissions carbon energy policy renewable resilience warming greenhouse communities governments strategy action plan",
        },
        "transformer_mechanism": {
            "concept_star": "Transformer model",
            "operator_id": "mechanism_explanation",
            "domain_frame": "machine_learning",
            "common_structure": "attention, token representations, and layered computation",
            "prior_terms": "Transformer attention self attention token tokens representations embeddings layers computation neural model context sequence architecture heads keys queries values",
        },
        "market_compare": {
            "concept_star": "market coordination",
            "operator_id": "comparison",
            "domain_frame": "economics",
            "common_structure": "compare decentralized market coordination with centralized planning",
            "prior_terms": "market markets economy price prices coordination decentralized centralized planning allocation command supply demand exchange competition signals",
        },
        "memory_counterfactual": {
            "concept_star": "memory",
            "operator_id": "counterfactual",
            "domain_frame": "cognition",
            "common_structure": "consequences of absent, altered, or unreliable memory",
            "prior_terms": "memory memories cognition mind recall remember experience learning long term short term forgetting identity absent altered unreliable counterfactual",
        },
        "ocean_property": {
            "concept_star": "ocean",
            "operator_id": "property_description",
            "domain_frame": "earth_science",
            "common_structure": "major physical and ecological properties of oceans",
            "prior_terms": "ocean oceans sea marine water salinity currents tides ecosystem ecosystems climate waves depth earth ecological physical properties",
        },
        "robot_relation": {
            "concept_star": "robot",
            "operator_id": "relation_mapping",
            "domain_frame": "robotics",
            "common_structure": "relation among robots, automation, sensors, control, and machines",
            "prior_terms": "robot robots robotics automation machine machines sensors control actuators autonomous systems technology mechanical relation function",
        },
    }


# ============================================================
# ARTIFACT FILTERS
# ============================================================

HARD_ARTIFACTS = {
    "asked", "iss", "restr", "spdx", "conf", "below", "tow", "purs", "based",
    "true", "provide", "provided", "write", "please", "help", "section",
    "@section", "apache", "https", "http", "www", "sites", "topics", "wh",
    ".wh", "/sites", "/topics", "_bl", ".desc", "pub", "public", "options",
    "given", "begin", "your", "all", "one", "some", "ms", "dec", "symbol",
    "def", "polit", "illustr", "illustrate", "example", "examples", "asked",
    "contin", "inue", "reperc", "aeros", "inh", "enc", "nd", "tf", "br",
    "bec", "don", "let", "he", "so", "high", "factors", "name", "set",
    "limits", "this", "that", "these", "those", "here", "there", "has",
    "have", "had", "would", "could", "should", "will", "shall", "may",
    "might", "must", "not", "yes", "no", "if", "then", "else", "when",
    "while", "because", "therefore", "however", "also", "which", "what",
    "why", "how", "answer", "question", "prompt", "task", "concept",
    "operator", "instruction", "common", "structure", "explain", "describe",
    "define", "list", "compare", "analyze", "analysis"
}

BAD_PATTERNS = [
    r"^https?",
    r"^www",
    r"\.com",
    r"^/[\w\-]+",
    r"^\.[A-Za-z]",
    r"^_",
    r"^[\W_]+$",
    r"^\d+$",
    r"^[A-Za-z]$",
    r" ",
    r"^<\|.*\|>$",
    r"^[A-Z][a-z]?$",   # many single fragments like Ms, Gr; semantic words still rescued by prior/cos
]


def norm_text(s: str) -> str:
    s = str(s).replace("▁", " ").replace("Ġ", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def key_text(s: str) -> str:
    return norm_text(s).lower().strip()


def ascii_ratio(s: str) -> float:
    if not s:
        return 0.0
    return sum(ord(ch) < 128 for ch in s) / max(1, len(s))


def is_artifact_token(tok: str) -> bool:
    t = norm_text(tok)
    k = key_text(t)
    if not k or len(k) < 2:
        return True
    if ascii_ratio(t) < 0.85:
        return True
    if k in HARD_ARTIFACTS:
        return True
    for pat in BAD_PATTERNS:
        if re.search(pat, k):
            return True
    # Too little alphabetic content
    alpha = sum(ch.isalpha() for ch in k)
    alnum = sum(ch.isalnum() for ch in k)
    if alnum / max(1, len(k)) < 0.55:
        return True
    if alpha == 0:
        return True
    return False


def content_quality(tok: str) -> float:
    k = key_text(tok)
    q = 0.0
    if len(k) >= 4:
        q += 0.2
    if len(k) >= 7:
        q += 0.2
    if "_" in k or " " in k:
        q += 0.25
    if any(ch.isalpha() for ch in k):
        q += 0.25
    if k[0].isupper() if k else False:
        q += 0.05
    return q


def prior_terms_for_family(fam: str) -> List[str]:
    meta = seed_family_structure_map().get(fam, {})
    text = " ".join([
        meta.get("concept_star", ""),
        meta.get("operator_id", ""),
        meta.get("domain_frame", ""),
        meta.get("common_structure", ""),
        meta.get("prior_terms", ""),
    ])
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text).lower()
    toks = [t for t in text.split() if len(t) >= 2 and t not in HARD_ARTIFACTS]
    # Preserve order
    out = []
    seen = set()
    for t in toks:
        if t not in seen:
            out.append(t); seen.add(t)
    return out


def prior_lexical_match(tok: str, fam: str) -> bool:
    k = key_text(tok).replace(" ", "_")
    priors = prior_terms_for_family(fam)
    prior_set = set(priors)
    if k in prior_set:
        return True
    k_plain = k.replace("_", "")
    for p in priors:
        p2 = p.replace("_", "")
        if len(k_plain) >= 4 and len(p2) >= 4 and (k_plain in p2 or p2 in k_plain):
            return True
    return False


# ============================================================
# MODEL EMBEDDING / SEMANTIC PROJECTION
# ============================================================

def get_torch_dtype(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    return "auto"


def load_tokenizer_and_W(cfg: CFG):
    print(f"[SEM-3D.3c] Loading tokenizer/model embeddings: {cfg.MODEL_PATH}")
    dtype = get_torch_dtype(cfg.DTYPE)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
        device_map="auto" if cfg.DEVICE == "cuda" else None,
    )
    W = model.get_output_embeddings().weight.detach().float().cpu().numpy()
    return tokenizer, W


def encode_term_ids(tokenizer, term: str) -> List[int]:
    try:
        ids = tokenizer.encode(term, add_special_tokens=False)
    except Exception:
        ids = []
    return [int(i) for i in ids if isinstance(i, int)]


def vector_for_text(tokenizer, W: np.ndarray, text: str):
    ids = encode_term_ids(tokenizer, text)
    ids = [i for i in ids if 0 <= i < W.shape[0]]
    if not ids:
        return None
    return W[ids].mean(axis=0)


def normalize_vec(v: np.ndarray, eps=1e-9):
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


def family_prior_centroids(tokenizer, W: np.ndarray):
    centroids = {}
    prior_vecs = {}
    for fam in seed_family_structure_map().keys():
        terms = prior_terms_for_family(fam)
        vecs = []
        for t in terms:
            v = vector_for_text(tokenizer, W, t)
            if v is not None:
                vecs.append(normalize_vec(v))
        if vecs:
            c = normalize_vec(np.mean(np.vstack(vecs), axis=0))
            centroids[fam] = c
            prior_vecs[fam] = vecs
    return centroids, prior_vecs


def token_vector(W: np.ndarray, token_id: int):
    if 0 <= int(token_id) < W.shape[0]:
        return normalize_vec(W[int(token_id)])
    return None


def cosine(a, b, eps=1e-9):
    if a is None or b is None:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ============================================================
# IO
# ============================================================

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


def load_inputs(cfg: CFG):
    indir = Path(cfg.INPUT_DIR)
    scores_path = indir / cfg.CLEAN_SCORES
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing {scores_path}")
    scores = pd.read_csv(scores_path)
    anchors = pd.read_csv(indir / cfg.CLEAN_ANCHORS) if (indir / cfg.CLEAN_ANCHORS).exists() else pd.DataFrame()
    family_summary = pd.read_csv(indir / cfg.CLEAN_FAMILY_SUMMARY) if (indir / cfg.CLEAN_FAMILY_SUMMARY).exists() else pd.DataFrame()
    return scores, anchors, family_summary


# ============================================================
# SCORING
# ============================================================

def semantic_projection_filter(scores: pd.DataFrame, tokenizer, W: np.ndarray, cfg: CFG):
    centroids, prior_vecs = family_prior_centroids(tokenizer, W)

    rows = []
    audit = {
        "total_input": int(len(scores)),
        "removed_window_topk": 0,
        "removed_artifact": 0,
        "removed_missing_centroid": 0,
        "removed_low_semantic": 0,
        "kept": 0,
        "prior_match": 0,
    }

    for _, r in scores.iterrows():
        fam = r["seed_family"]
        tok = norm_text(r.get("token_clean", r.get("token", "")))

        if r.get("window", "") not in cfg.PREFERRED_WINDOWS or int(r.get("topk", -1)) not in cfg.PREFERRED_TOPK:
            audit["removed_window_topk"] += 1
            continue

        if is_artifact_token(tok):
            # Rescue if it is explicit prior, e.g. "Mass", "Python", "Newton".
            if not prior_lexical_match(tok, fam):
                audit["removed_artifact"] += 1
                continue

        if fam not in centroids:
            audit["removed_missing_centroid"] += 1
            continue

        tv = token_vector(W, int(r["token_id"]))
        sem_cos = cosine(tv, centroids[fam])
        prior_match = prior_lexical_match(tok, fam)

        # Allow either semantic embedding proximity or lexical prior match.
        if sem_cos < cfg.MIN_SEMANTIC_COS and not prior_match:
            audit["removed_low_semantic"] += 1
            continue

        if prior_match:
            audit["prior_match"] += 1

        # Anchor score may be tiny; use log specificity to avoid huge raw ratios dominating too much.
        clean_anchor = float(r.get("clean_anchor_score", r.get("anchor_score", 0.0)))
        log_spec = float(r.get("log_specificity_ratio", 0.0))
        spec_component = max(0.0, log_spec)

        cqual = content_quality(tok)
        prior_component = 1.0 if prior_match else 0.0

        final_score = (
            cfg.W_ANCHOR * clean_anchor
            + cfg.W_SEMANTIC * max(0.0, sem_cos)
            + cfg.W_PRIOR * prior_component
            + cfg.W_CONTENT * cqual
            + cfg.W_SPECIFICITY * spec_component
        )

        if final_score < cfg.MIN_FINAL_SCORE:
            continue

        row = r.to_dict()
        row.update({
            "token_semantic": tok,
            "semantic_cos_to_prior_centroid": sem_cos,
            "prior_match": int(prior_match),
            "content_quality": cqual,
            "semantic_final_score": float(final_score),
        })
        rows.append(row)
        audit["kept"] += 1

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            ["seed_family", "window", "topk", "semantic_final_score"],
            ascending=[True, True, True, False]
        )
    return out, audit


def select_semantic_anchors(semantic_scores: pd.DataFrame, cfg: CFG):
    rows = []
    if semantic_scores.empty:
        return pd.DataFrame()
    for (fam, window, topk), g in semantic_scores.groupby(["seed_family", "window", "topk"]):
        g = g.sort_values("semantic_final_score", ascending=False)
        g = g.drop_duplicates(subset=["token_key"] if "token_key" in g.columns else ["token_semantic"], keep="first")
        top = g.head(cfg.TOP_N_SEMANTIC_ANCHORS)
        for rank, (_, r) in enumerate(top.iterrows(), start=1):
            rows.append({**r.to_dict(), "semantic_anchor_rank": rank})
    return pd.DataFrame(rows)


# ============================================================
# FAMILY SUMMARY AND PROMPTS
# ============================================================

def build_family_summary(semantic_anchors: pd.DataFrame, cfg: CFG):
    fmap = seed_family_structure_map()
    rows = []

    for fam in sorted(semantic_anchors["seed_family"].unique()):
        sub = semantic_anchors[semantic_anchors["seed_family"] == fam].copy()
        if sub.empty:
            continue

        # Prefer prior-matched, semantically close, topk50, but preserve model-discovered terms too.
        sub["prompt_rank_score"] = (
            sub["semantic_final_score"]
            + 0.25 * sub["prior_match"].astype(float)
            + 0.15 * (sub["topk"] == 50).astype(float)
            + 0.25 * sub["semantic_cos_to_prior_centroid"].astype(float)
        )
        sub = sub.sort_values("prompt_rank_score", ascending=False).drop_duplicates("token_semantic")

        # Enforce minimum prior fraction if possible by mixing prior-matched terms first.
        prior = sub[sub["prior_match"] == 1]
        nonprior = sub[sub["prior_match"] == 0]
        n = cfg.TOP_N_PROMPT_ANCHORS
        need_prior = int(np.ceil(cfg.TARGET_PRIOR_MATCH_FRAC * n))
        selected = pd.concat([prior.head(need_prior), nonprior.head(n)], axis=0)
        selected = selected.drop_duplicates("token_semantic").head(n)
        # If not enough, backfill
        if len(selected) < n:
            selected = pd.concat([selected, sub], axis=0).drop_duplicates("token_semantic").head(n)

        terms = selected["token_semantic"].astype(str).tolist()
        ids = selected["token_id"].astype(int).tolist()

        meta = fmap.get(fam, {})
        rows.append({
            "seed_family": fam,
            "concept_star": meta.get("concept_star", fam),
            "operator_id": meta.get("operator_id", ""),
            "domain_frame": meta.get("domain_frame", ""),
            "common_structure": meta.get("common_structure", ""),
            "semantic_anchor_terms": ", ".join(terms),
            "semantic_anchor_token_ids": ",".join(map(str, ids)),
            "semantic_anchor_terms_json": json.dumps(terms, ensure_ascii=False),
            "semantic_anchor_token_ids_json": json.dumps(ids),
            "n_semantic_anchors": int(len(terms)),
            "mean_semantic_final_score": float(selected["semantic_final_score"].mean()) if len(selected) else np.nan,
            "mean_semantic_cos": float(selected["semantic_cos_to_prior_centroid"].mean()) if len(selected) else np.nan,
            "mean_specificity_ratio": float(selected["specificity_ratio"].mean()) if "specificity_ratio" in selected and len(selected) else np.nan,
            "prior_match_frac": float(selected["prior_match"].mean()) if len(selected) else np.nan,
            "top_windows": json.dumps(selected["window"].value_counts().to_dict(), ensure_ascii=False) if len(selected) else "{}",
            "top_topk": json.dumps(selected["topk"].value_counts().to_dict(), ensure_ascii=False) if len(selected) else "{}",
        })

    return pd.DataFrame(rows)


def build_prompt_candidates(summary: pd.DataFrame):
    rows = []
    families = summary["seed_family"].tolist()
    fam_to_anchor = {r["seed_family"]: r["semantic_anchor_terms"] for _, r in summary.iterrows()}

    for _, r in summary.iterrows():
        fam = r["seed_family"]
        concept = r.get("concept_star", fam)
        operator = r.get("operator_id", "")
        common = r.get("common_structure", "")
        anchors = r.get("semantic_anchor_terms", "")

        rows.append({
            "seed_family": fam,
            "candidate_type": "concept_operator_semantic_topk_anchor",
            "anchor_family": fam,
            "prompt": (
                f"Concept: {concept}\n"
                f"OperatorID: {operator}\n"
                f"SemanticGeometryAnchor: {anchors}\n"
                f"Use this model-derived semantic neighborhood to answer. Preserve: {common}."
            ),
            "concept_star": concept,
            "operator_id": operator,
            "anchor_terms": anchors,
            "common_structure": common,
        })

        rows.append({
            "seed_family": fam,
            "candidate_type": "natural_semantic_topk_anchor_prompt",
            "anchor_family": fam,
            "prompt": (
                f"Use these semantic geometry anchors: {anchors}. "
                f"Explain {concept} with operation {operator}. Shared structure: {common}."
            ),
            "concept_star": concept,
            "operator_id": operator,
            "anchor_terms": anchors,
            "common_structure": common,
        })

        rows.append({
            "seed_family": fam,
            "candidate_type": "semantic_topk_anchor_only",
            "anchor_family": fam,
            "prompt": f"SemanticGeometryAnchor: {anchors}\nExplain the shared structure represented by this model-derived semantic neighborhood.",
            "concept_star": concept,
            "operator_id": operator,
            "anchor_terms": anchors,
            "common_structure": common,
        })

        rows.append({
            "seed_family": fam,
            "candidate_type": "concept_operator_no_semantic_anchor",
            "anchor_family": "",
            "prompt": f"Concept: {concept}\nOperatorID: {operator}\nAnswer according to this concept and operation.",
            "concept_star": concept,
            "operator_id": operator,
            "anchor_terms": "",
            "common_structure": common,
        })

        # random-family anchor control
        if len(families) > 1:
            idx = families.index(fam)
            rand_fam = families[(idx + 1) % len(families)]
            rand_anchor = fam_to_anchor[rand_fam]
            rows.append({
                "seed_family": fam,
                "candidate_type": "concept_operator_random_semantic_topk_anchor",
                "anchor_family": rand_fam,
                "prompt": (
                    f"Concept: {concept}\n"
                    f"OperatorID: {operator}\n"
                    f"SemanticGeometryAnchor: {rand_anchor}\n"
                    f"Use this model-derived semantic neighborhood to answer. Preserve: {common}."
                ),
                "concept_star": concept,
                "operator_id": operator,
                "anchor_terms": rand_anchor,
                "common_structure": common,
            })

    return pd.DataFrame(rows)


def build_ablation_prompts(summary: pd.DataFrame):
    rows = []
    for _, r in summary.iterrows():
        fam = r["seed_family"]
        concept = r.get("concept_star", fam)
        operator = r.get("operator_id", "")
        common = r.get("common_structure", "")
        anchors = r.get("semantic_anchor_terms", "")

        ablations = {
            "concept_only": f"Concept: {concept}\nAnswer about this concept.",
            "operator_only": f"OperatorID: {operator}\nApply this operation to an appropriate concept.",
            "semantic_topk_anchor_only": f"SemanticGeometryAnchor: {anchors}\nAnswer using this model-derived semantic neighborhood.",
            "concept_operator": f"Concept: {concept}\nOperatorID: {operator}\nAnswer accordingly.",
            "concept_semantic_topk_anchor": f"Concept: {concept}\nSemanticGeometryAnchor: {anchors}\nAnswer using this model-derived semantic neighborhood.",
            "operator_semantic_topk_anchor": f"OperatorID: {operator}\nSemanticGeometryAnchor: {anchors}\nAnswer using this model-derived semantic neighborhood.",
            "concept_operator_semantic_topk_anchor": (
                f"Concept: {concept}\nOperatorID: {operator}\nSemanticGeometryAnchor: {anchors}\n"
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
# AUDIT / SUMMARY
# ============================================================

def build_filter_audit(input_scores: pd.DataFrame, semantic_scores: pd.DataFrame, audit: Dict[str, int]):
    rows = []
    total = max(1, audit["total_input"])
    for k, v in audit.items():
        rows.append({"metric": k, "value": v, "ratio": v / total})
    if len(input_scores):
        rows.append({
            "metric": "input_prior_match_ratio",
            "value": float(pd.to_numeric(input_scores.get("prior_match", pd.Series([0]*len(input_scores))), errors="coerce").fillna(0).mean()),
            "ratio": float(pd.to_numeric(input_scores.get("prior_match", pd.Series([0]*len(input_scores))), errors="coerce").fillna(0).mean()),
        })
    if len(semantic_scores):
        rows.append({
            "metric": "semantic_prior_match_ratio",
            "value": float(semantic_scores["prior_match"].mean()),
            "ratio": float(semantic_scores["prior_match"].mean()),
        })
        rows.append({
            "metric": "semantic_mean_cos",
            "value": float(semantic_scores["semantic_cos_to_prior_centroid"].mean()),
            "ratio": np.nan,
        })
    return pd.DataFrame(rows)


def build_results_summary(cfg: CFG, audit_df, semantic_scores, anchors, fam_summary, prompt_candidates):
    result = {
        "config": asdict(cfg),
        "verdict": "SEMANTIC_GEOMETRY_ANCHOR_CANDIDATES_READY",
        "n_semantic_score_rows": int(len(semantic_scores)),
        "n_semantic_anchor_terms": int(len(anchors)),
        "n_families": int(fam_summary["seed_family"].nunique()) if len(fam_summary) else 0,
        "n_prompt_candidates": int(len(prompt_candidates)),
        "quality": {},
        "top_anchor_preview": {},
        "interpretation": [
            "SemanticGeometryAnchor candidates are selected by intersection of TopK specificity, semantic projection to family prior centroid, and artifact quotient.",
            "SEM-3D.4 should compare semantic TopK anchors against clean TopK anchors, lexical anchors, no-anchor, and random-anchor controls.",
            "If semantic anchors improve center_pca regeneration, GeometryAnchor_geo is supported."
        ],
    }
    if len(audit_df):
        result["filter_audit"] = audit_df.to_dict(orient="records")
    if len(fam_summary):
        result["quality"] = {
            "mean_semantic_final_score": float(fam_summary["mean_semantic_final_score"].mean()),
            "mean_semantic_cos": float(fam_summary["mean_semantic_cos"].mean()),
            "mean_specificity_ratio": float(fam_summary["mean_specificity_ratio"].mean()),
            "mean_prior_match_frac": float(fam_summary["prior_match_frac"].mean()),
        }
        for _, r in fam_summary.iterrows():
            result["top_anchor_preview"][r["seed_family"]] = json.loads(r["semantic_anchor_terms_json"])
    return result


# ============================================================
# MAIN
# ============================================================

def main():
    np.random.seed(cfg.RANDOM_SEED)
    ensure_dir(cfg.OUTPUT_DIR)
    outdir = Path(cfg.OUTPUT_DIR)
    json_dump(asdict(cfg), outdir / "sem3d3c_config.json")

    clean_scores, clean_anchors, clean_family_summary = load_inputs(cfg)
    tokenizer, W = load_tokenizer_and_W(cfg)

    semantic_scores, audit = semantic_projection_filter(clean_scores, tokenizer, W, cfg)
    semantic_anchors = select_semantic_anchors(semantic_scores, cfg)
    fam_summary = build_family_summary(semantic_anchors, cfg)
    prompt_candidates = build_prompt_candidates(fam_summary)
    ablations = build_ablation_prompts(fam_summary)
    audit_df = build_filter_audit(clean_scores, semantic_scores, audit)

    semantic_scores.to_csv(outdir / "sem3d3c_semantic_anchor_scores.csv", index=False, encoding="utf-8-sig")
    semantic_anchors.to_csv(outdir / "sem3d3c_semantic_anchor_terms.csv", index=False, encoding="utf-8-sig")
    fam_summary.to_csv(outdir / "sem3d3c_family_semantic_anchor_summary.csv", index=False, encoding="utf-8-sig")
    prompt_candidates.to_csv(outdir / "sem3d3c_semantic_anchor_prompt_candidates.csv", index=False, encoding="utf-8-sig")
    ablations.to_csv(outdir / "sem3d3c_semantic_anchor_ablation_prompts.csv", index=False, encoding="utf-8-sig")
    audit_df.to_csv(outdir / "sem3d3c_filter_audit.csv", index=False, encoding="utf-8-sig")

    summary = build_results_summary(cfg, audit_df, semantic_scores, semantic_anchors, fam_summary, prompt_candidates)
    json_dump(summary, outdir / "sem3d3c_results_summary.json")

    print("=" * 100)
    print("SEM-3D.3c: Semantic Projection Filtering")
    print("=" * 100)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nOutput files:")
    for p in [
        "sem3d3c_config.json",
        "sem3d3c_semantic_anchor_scores.csv",
        "sem3d3c_semantic_anchor_terms.csv",
        "sem3d3c_family_semantic_anchor_summary.csv",
        "sem3d3c_semantic_anchor_prompt_candidates.csv",
        "sem3d3c_semantic_anchor_ablation_prompts.csv",
        "sem3d3c_filter_audit.csv",
        "sem3d3c_results_summary.json",
    ]:
        print("  -", outdir / p)


if __name__ == "__main__":
    main()
