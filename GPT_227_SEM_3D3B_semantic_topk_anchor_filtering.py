# -*- coding: utf-8 -*-
"""
GPT_212_SEM_3D3B_semantic_topk_anchor_filtering.py

SEM-3D.3b: Semantic TopK Anchor Filtering
-----------------------------------------

Goal:
    Clean raw TopK-derived anchors from SEM-3D.3 and extract semantic GeometryAnchor candidates.

Motivation:
    SEM-3D.3 showed family-specific TopK signal exists, but raw anchors are polluted by:
        - numeric tokens
        - punctuation / malformed tokens
        - generic format tokens
        - template tokens
        - cross-family frequent tokens
        - url/code fragments

    We need:
        RawTopKAnchor -> SemanticGeometryAnchor

Inputs:
    sem3d3_outputs/
      sem3d3_topk_anchor_scores.csv
      sem3d3_anchor_terms.csv
      sem3d3_family_topk_anchor_summary.csv
      sem3d3_layer_anchor_diagnostics.csv

Outputs:
    sem3d3b_outputs/
      sem3d3b_config.json
      sem3d3b_clean_anchor_scores.csv
      sem3d3b_clean_anchor_terms.csv
      sem3d3b_family_clean_anchor_summary.csv
      sem3d3b_clean_anchor_prompt_candidates.csv
      sem3d3b_clean_anchor_ablation_prompts.csv
      sem3d3b_filter_audit.csv
      sem3d3b_results_summary.json

Run:
    python GPT_212_SEM_3D3B_semantic_topk_anchor_filtering.py

Next:
    SEM-3D.4 should forward sem3d3b_clean_anchor_prompt_candidates.csv and
    compare:
        clean TopK anchors vs lexical anchors vs raw TopK anchors vs random anchors.
"""

import json
import re
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

@dataclass
class CFG:
    INPUT_DIR: str = "./sem3d3_outputs"
    OUTPUT_DIR: str = "./sem3d3b_outputs"

    RAW_SCORES: str = "sem3d3_topk_anchor_scores.csv"
    RAW_ANCHORS: str = "sem3d3_anchor_terms.csv"
    RAW_FAMILY_SUMMARY: str = "sem3d3_family_topk_anchor_summary.csv"
    RAW_DIAGNOSTICS: str = "sem3d3_layer_anchor_diagnostics.csv"

    # Preferred windows/k for clean prompt anchors.
    PREFERRED_WINDOWS: Tuple[str, ...] = ("init_L0_6", "mid_L7_19", "decision_L20_25")
    PREFERRED_TOPK: Tuple[int, ...] = (50, 100)

    TOP_N_CLEAN_ANCHORS: int = 32
    TOP_N_PROMPT_ANCHORS: int = 14

    # Filtering thresholds
    MIN_TOKEN_LEN: int = 2
    MIN_ALPHA_RATIO: float = 0.0

    # Score blend
    PRIOR_BOOST: float = 1.25
    CONTENT_BOOST: float = 1.0
    BAD_PENALTY: float = 10.0

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
            "prior_terms": "inertia inertial motion force rest uniform external object state mechanics Newton law acceleration velocity momentum",
        },
        "photosynthesis_mechanism": {
            "concept_star": "photosynthesis",
            "operator_id": "mechanism_explanation",
            "domain_frame": "biology",
            "common_structure": "plants convert light, carbon dioxide, and water into chemical energy",
            "prior_terms": "photosynthesis plant plants light sunlight carbon dioxide water chlorophyll sugar glucose chemical energy leaves oxygen",
        },
        "democracy_definition": {
            "concept_star": "democracy",
            "operator_id": "definition",
            "domain_frame": "politics",
            "common_structure": "political system involving citizen participation, representation, and collective rule",
            "prior_terms": "democracy democratic government citizen participation representation rule voting people society political system rights majority",
        },
        "gravity_causal": {
            "concept_star": "gravity",
            "operator_id": "causal_explanation",
            "domain_frame": "physics",
            "common_structure": "mass attracts mass and shapes falling and orbital motion",
            "prior_terms": "gravity gravitational mass attraction attracts force fall falling earth orbit planets stars motion acceleration spacetime",
        },
        "python_use_cases": {
            "concept_star": "Python programming language",
            "operator_id": "list_use_cases",
            "domain_frame": "software",
            "common_structure": "common practical applications and domains of Python",
            "prior_terms": "Python programming language software data analysis automation scripting machine learning web development code libraries",
        },
        "internet_risk": {
            "concept_star": "internet",
            "operator_id": "risk_audit",
            "domain_frame": "technology_society",
            "common_structure": "risks, vulnerabilities, and failure modes of internet systems and use",
            "prior_terms": "internet online network cybersecurity privacy security vulnerability vulnerabilities risk risks malware phishing connectivity misinformation",
        },
        "climate_plan": {
            "concept_star": "climate change",
            "operator_id": "planning",
            "domain_frame": "environment_policy",
            "common_structure": "practical mitigation and adaptation strategy",
            "prior_terms": "climate change mitigation adaptation emissions carbon energy policy renewable resilience warming greenhouse communities",
        },
        "transformer_mechanism": {
            "concept_star": "Transformer model",
            "operator_id": "mechanism_explanation",
            "domain_frame": "machine_learning",
            "common_structure": "attention, token representations, and layered computation",
            "prior_terms": "Transformer attention self attention token tokens representations embeddings layers computation neural model context sequence",
        },
        "market_compare": {
            "concept_star": "market coordination",
            "operator_id": "comparison",
            "domain_frame": "economics",
            "common_structure": "compare decentralized market coordination with centralized planning",
            "prior_terms": "market markets economy price prices coordination decentralized centralized planning allocation command supply demand exchange",
        },
        "memory_counterfactual": {
            "concept_star": "memory",
            "operator_id": "counterfactual",
            "domain_frame": "cognition",
            "common_structure": "consequences of absent, altered, or unreliable memory",
            "prior_terms": "memory memories cognition mind recall remember experience learning long term short term forgetting identity",
        },
        "ocean_property": {
            "concept_star": "ocean",
            "operator_id": "property_description",
            "domain_frame": "earth_science",
            "common_structure": "major physical and ecological properties of oceans",
            "prior_terms": "ocean oceans sea marine water salinity currents tides ecosystem ecosystems climate waves depth earth",
        },
        "robot_relation": {
            "concept_star": "robot",
            "operator_id": "relation_mapping",
            "domain_frame": "robotics",
            "common_structure": "relation among robots, automation, sensors, control, and machines",
            "prior_terms": "robot robots robotics automation machine machines sensors control actuators autonomous systems technology mechanical",
        },
    }


# ============================================================
# STOPLIST / FILTERS
# ============================================================

GENERIC_BAD = {
    # numbers / steps / formatting
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "step", "steps",
    "a", "b", "c", "d", "e", "r", "s", "i", "o",
    "this", "that", "these", "those", "there", "here", "it", "its", "they",
    "if", "then", "else", "when", "while", "will", "would", "could", "should",
    "has", "have", "had", "was", "were", "is", "are", "be", "been", "being",
    "the", "and", "or", "of", "to", "in", "on", "for", "with", "by", "as", "at",
    "from", "about", "into", "through", "between", "among", "which", "what", "why",
    "how", "can", "may", "might", "must", "not", "no", "yes",
    "according", "because", "therefore", "however", "also", "such", "than",
    "answer", "question", "prompt", "concept", "operator", "definition", "explain",
    "describe", "list", "compare", "contrast", "analyze", "analysis",
    "scient", "sovere", "contin", "enh", "deep", "aside", "new", "my", "we",
    "wouldn", "sup", "that", "提", "哪", "媲", "https", "http", "www",
    "<|endoftext|>", "endoftext",
}

GENERIC_PATTERNS = [
    r"^#+$",
    r"^[\W_]+$",
    r"^\d+$",
    r"^https?$",
    r"^www$",
    r"^<\|.*\|>$",
    r"^[a-z]$",
    r"^[A-Z]$",
    r" ",
    r"  ",
]


def normalize_token(tok: str) -> str:
    tok = str(tok).strip()
    tok = tok.replace("▁", " ").replace("Ġ", " ")
    tok = tok.replace("\n", " ")
    tok = re.sub(r"\s+", " ", tok).strip()
    return tok


def token_key(tok: str) -> str:
    return normalize_token(tok).lower().strip()


def is_asciiish(tok: str) -> bool:
    # Allow normal punctuation inside terms but require most chars ascii.
    if not tok:
        return False
    ascii_count = sum(1 for ch in tok if ord(ch) < 128)
    return ascii_count / max(1, len(tok)) >= 0.85


def is_bad_token(tok: str, cfg: CFG) -> bool:
    tok = normalize_token(tok)
    key = token_key(tok)
    if len(key) < cfg.MIN_TOKEN_LEN:
        return True
    if key in GENERIC_BAD:
        return True
    if not is_asciiish(tok):
        return True
    for pat in GENERIC_PATTERNS:
        if re.search(pat, key):
            return True
    # Remove mostly non-alphanumeric tokens.
    alnum = sum(ch.isalnum() for ch in key)
    if alnum / max(1, len(key)) < 0.55:
        return True
    # Remove URL/code fragments.
    if "http" in key or "://" in key or ".com" in key or ".enumer" in key:
        return True
    return False


def prior_terms_for_family(fam: str) -> set:
    meta = seed_family_structure_map().get(fam, {})
    text = " ".join([
        meta.get("concept_star", ""),
        meta.get("operator_id", ""),
        meta.get("domain_frame", ""),
        meta.get("common_structure", ""),
        meta.get("prior_terms", ""),
    ])
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text).lower()
    toks = [t for t in text.split() if len(t) >= 2 and t not in GENERIC_BAD]
    out = set(toks)
    # crude bigrams
    for a, b in zip(toks[:-1], toks[1:]):
        out.add(f"{a}_{b}")
    return out


def token_matches_prior(tok: str, fam: str) -> bool:
    key = token_key(tok)
    priors = prior_terms_for_family(fam)
    if key in priors:
        return True
    # allow substring relation for longer content words
    for p in priors:
        if len(key) >= 5 and len(p) >= 5 and (key in p or p in key):
            return True
    return False


def content_score(tok: str) -> float:
    key = token_key(tok)
    score = 0.0
    if len(key) >= 4:
        score += 0.2
    if len(key) >= 7:
        score += 0.2
    if "_" in key:
        score += 0.3
    if any(ch.isalpha() for ch in key):
        score += 0.3
    if any(ch.isdigit() for ch in key):
        score -= 0.2
    return score


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
    raw_scores_path = indir / cfg.RAW_SCORES
    raw_anchors_path = indir / cfg.RAW_ANCHORS
    if not raw_scores_path.exists():
        raise FileNotFoundError(f"Missing {raw_scores_path}")
    raw_scores = pd.read_csv(raw_scores_path)
    raw_anchors = pd.read_csv(raw_anchors_path) if raw_anchors_path.exists() else pd.DataFrame()
    raw_family_summary = pd.read_csv(indir / cfg.RAW_FAMILY_SUMMARY) if (indir / cfg.RAW_FAMILY_SUMMARY).exists() else pd.DataFrame()
    raw_diag = pd.read_csv(indir / cfg.RAW_DIAGNOSTICS) if (indir / cfg.RAW_DIAGNOSTICS).exists() else pd.DataFrame()
    return raw_scores, raw_anchors, raw_family_summary, raw_diag


# ============================================================
# CLEANING
# ============================================================

def clean_scores(raw_scores: pd.DataFrame, cfg: CFG):
    rows = []
    audit = {
        "total_raw": int(len(raw_scores)),
        "removed_bad_token": 0,
        "removed_window_topk": 0,
        "kept": 0,
        "prior_match": 0,
    }

    for _, r in raw_scores.iterrows():
        fam = r["seed_family"]
        tok = normalize_token(r["token"])
        key = token_key(tok)

        if r.get("window", "") not in cfg.PREFERRED_WINDOWS or int(r.get("topk", -1)) not in cfg.PREFERRED_TOPK:
            audit["removed_window_topk"] += 1
            continue

        if is_bad_token(tok, cfg):
            audit["removed_bad_token"] += 1
            continue

        prior_match = token_matches_prior(tok, fam)
        if prior_match:
            audit["prior_match"] += 1

        cscore = content_score(tok)
        prior_boost = cfg.PRIOR_BOOST if prior_match else 0.0
        clean_score = float(r["anchor_score"]) * (1.0 + cscore + prior_boost)

        # If not prior-matched but strong TopK specificity, still allow.
        # This lets us discover model-native anchors not in manual prior.
        if (not prior_match) and clean_score <= cfg.MIN_ALPHA_RATIO:
            # keep if original anchor score is genuinely positive and content-like
            if float(r["anchor_score"]) <= 0:
                continue

        row = r.to_dict()
        row.update({
            "token_clean": tok,
            "token_key": key,
            "prior_match": int(prior_match),
            "content_score": float(cscore),
            "clean_anchor_score": clean_score,
        })
        rows.append(row)
        audit["kept"] += 1

    clean = pd.DataFrame(rows)
    if len(clean):
        clean = clean.sort_values(
            ["seed_family", "window", "topk", "clean_anchor_score"],
            ascending=[True, True, True, False]
        )
    return clean, audit


def select_clean_anchors(clean_scores: pd.DataFrame, cfg: CFG):
    rows = []
    if clean_scores.empty:
        return pd.DataFrame()

    for (fam, window, topk), g in clean_scores.groupby(["seed_family", "window", "topk"]):
        # Deduplicate by token_key, keep max score.
        g = g.sort_values("clean_anchor_score", ascending=False)
        g = g.drop_duplicates(subset=["token_key"], keep="first")
        top = g.head(cfg.TOP_N_CLEAN_ANCHORS)
        for rank, (_, r) in enumerate(top.iterrows(), start=1):
            rows.append({**r.to_dict(), "clean_anchor_rank": rank})
    return pd.DataFrame(rows)


# ============================================================
# FAMILY SUMMARY AND PROMPTS
# ============================================================

def build_family_summary(clean_anchors: pd.DataFrame, cfg: CFG):
    fmap = seed_family_structure_map()
    rows = []
    for fam in sorted(clean_anchors["seed_family"].unique()):
        # Prefer init/mid/decision topk50, then topk100, combined.
        sub = clean_anchors[clean_anchors["seed_family"] == fam].copy()
        if sub.empty:
            continue

        # Additional preference for topk50 and prior matched anchors.
        sub["prompt_rank_score"] = (
            sub["clean_anchor_score"]
            + 0.002 * (sub["topk"] == 50).astype(float)
            + 0.003 * sub["prior_match"].astype(float)
        )
        sub = sub.sort_values("prompt_rank_score", ascending=False).drop_duplicates("token_key")

        top = sub.head(cfg.TOP_N_PROMPT_ANCHORS)
        terms = top["token_clean"].astype(str).tolist()
        ids = top["token_id"].astype(int).tolist()

        meta = fmap.get(fam, {})
        rows.append({
            "seed_family": fam,
            "concept_star": meta.get("concept_star", fam),
            "operator_id": meta.get("operator_id", ""),
            "domain_frame": meta.get("domain_frame", ""),
            "common_structure": meta.get("common_structure", ""),
            "clean_anchor_terms": ", ".join(terms),
            "clean_anchor_token_ids": ",".join(map(str, ids)),
            "clean_anchor_terms_json": json.dumps(terms, ensure_ascii=False),
            "clean_anchor_token_ids_json": json.dumps(ids),
            "n_clean_anchors": int(len(terms)),
            "mean_clean_anchor_score": float(top["clean_anchor_score"].mean()) if len(top) else np.nan,
            "mean_specificity_ratio": float(top["specificity_ratio"].mean()) if len(top) else np.nan,
            "prior_match_frac": float(top["prior_match"].mean()) if len(top) else np.nan,
            "top_windows": json.dumps(top["window"].value_counts().to_dict(), ensure_ascii=False),
            "top_topk": json.dumps(top["topk"].value_counts().to_dict(), ensure_ascii=False),
        })
    return pd.DataFrame(rows)


def build_clean_anchor_prompt_candidates(summary: pd.DataFrame):
    rows = []
    families = summary["seed_family"].tolist()
    fam_to_anchor = {r["seed_family"]: r["clean_anchor_terms"] for _, r in summary.iterrows()}

    for _, r in summary.iterrows():
        fam = r["seed_family"]
        concept = r.get("concept_star", fam)
        operator = r.get("operator_id", "")
        common = r.get("common_structure", "")
        anchors = r.get("clean_anchor_terms", "")

        rows.append({
            "seed_family": fam,
            "candidate_type": "concept_operator_clean_topk_anchor",
            "anchor_family": fam,
            "prompt": (
                f"Concept: {concept}\n"
                f"OperatorID: {operator}\n"
                f"CleanTopKGeometryAnchor: {anchors}\n"
                f"Use this model-derived semantic neighborhood to answer. Preserve: {common}."
            ),
            "concept_star": concept,
            "operator_id": operator,
            "anchor_terms": anchors,
            "common_structure": common,
        })

        rows.append({
            "seed_family": fam,
            "candidate_type": "natural_clean_topk_anchor_prompt",
            "anchor_family": fam,
            "prompt": (
                f"Use these model-derived semantic anchors: {anchors}. "
                f"Explain {concept} with operation {operator}. Shared structure: {common}."
            ),
            "concept_star": concept,
            "operator_id": operator,
            "anchor_terms": anchors,
            "common_structure": common,
        })

        rows.append({
            "seed_family": fam,
            "candidate_type": "clean_topk_anchor_only",
            "anchor_family": fam,
            "prompt": f"CleanTopKGeometryAnchor: {anchors}\nExplain the shared structure represented by this semantic neighborhood.",
            "concept_star": concept,
            "operator_id": operator,
            "anchor_terms": anchors,
            "common_structure": common,
        })

        rows.append({
            "seed_family": fam,
            "candidate_type": "concept_operator_no_clean_anchor",
            "anchor_family": "",
            "prompt": f"Concept: {concept}\nOperatorID: {operator}\nAnswer according to this concept and operation.",
            "concept_star": concept,
            "operator_id": operator,
            "anchor_terms": "",
            "common_structure": common,
        })

        # deterministic random anchor control
        if len(families) > 1:
            idx = families.index(fam)
            rand_fam = families[(idx + 1) % len(families)]
            rand_anchor = fam_to_anchor[rand_fam]
            rows.append({
                "seed_family": fam,
                "candidate_type": "concept_operator_random_clean_topk_anchor",
                "anchor_family": rand_fam,
                "prompt": (
                    f"Concept: {concept}\n"
                    f"OperatorID: {operator}\n"
                    f"CleanTopKGeometryAnchor: {rand_anchor}\n"
                    f"Use this model-derived semantic neighborhood to answer. Preserve: {common}."
                ),
                "concept_star": concept,
                "operator_id": operator,
                "anchor_terms": rand_anchor,
                "common_structure": common,
            })

    return pd.DataFrame(rows)


def build_clean_anchor_ablation_prompts(summary: pd.DataFrame):
    rows = []
    for _, r in summary.iterrows():
        fam = r["seed_family"]
        concept = r.get("concept_star", fam)
        operator = r.get("operator_id", "")
        common = r.get("common_structure", "")
        anchors = r.get("clean_anchor_terms", "")

        ablations = {
            "concept_only": f"Concept: {concept}\nAnswer about this concept.",
            "operator_only": f"OperatorID: {operator}\nApply this operation to an appropriate concept.",
            "clean_topk_anchor_only": f"CleanTopKGeometryAnchor: {anchors}\nAnswer using this model-derived semantic neighborhood.",
            "concept_operator": f"Concept: {concept}\nOperatorID: {operator}\nAnswer accordingly.",
            "concept_clean_topk_anchor": f"Concept: {concept}\nCleanTopKGeometryAnchor: {anchors}\nAnswer using this model-derived semantic neighborhood.",
            "operator_clean_topk_anchor": f"OperatorID: {operator}\nCleanTopKGeometryAnchor: {anchors}\nAnswer using this model-derived semantic neighborhood.",
            "concept_operator_clean_topk_anchor": (
                f"Concept: {concept}\nOperatorID: {operator}\nCleanTopKGeometryAnchor: {anchors}\n"
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

def build_filter_audit(raw_scores: pd.DataFrame, clean_scores: pd.DataFrame, audit: Dict[str, int]):
    rows = []
    total = max(1, audit["total_raw"])
    rows.append({
        "metric": "total_raw",
        "value": audit["total_raw"],
        "ratio": 1.0,
    })
    for k in ["removed_window_topk", "removed_bad_token", "kept", "prior_match"]:
        rows.append({
            "metric": k,
            "value": audit[k],
            "ratio": audit[k] / total,
        })

    if len(raw_scores):
        raw_numeric = raw_scores["token"].astype(str).str.match(r"^\d+$").mean()
        rows.append({"metric": "raw_numeric_token_ratio", "value": float(raw_numeric), "ratio": float(raw_numeric)})
    if len(clean_scores):
        clean_numeric = clean_scores["token_clean"].astype(str).str.match(r"^\d+$").mean()
        rows.append({"metric": "clean_numeric_token_ratio", "value": float(clean_numeric), "ratio": float(clean_numeric)})
        rows.append({"metric": "clean_prior_match_ratio", "value": float(clean_scores["prior_match"].mean()), "ratio": float(clean_scores["prior_match"].mean())})
    return pd.DataFrame(rows)


def build_results_summary(cfg: CFG, audit_df: pd.DataFrame, clean_scores: pd.DataFrame, anchors: pd.DataFrame, fam_summary: pd.DataFrame, prompt_candidates: pd.DataFrame):
    result = {
        "config": asdict(cfg),
        "verdict": "CLEAN_TOPK_ANCHOR_CANDIDATES_READY",
        "n_clean_score_rows": int(len(clean_scores)),
        "n_clean_anchor_terms": int(len(anchors)),
        "n_families": int(fam_summary["seed_family"].nunique()) if len(fam_summary) else 0,
        "n_prompt_candidates": int(len(prompt_candidates)),
        "quality": {},
        "top_anchor_preview": {},
        "interpretation": [
            "Clean TopK anchors remove generic/numeric/malformed tokens from raw TopK-derived anchors.",
            "SEM-3D.4 should compare CleanTopKAnchor prompts against lexical anchors, raw TopK anchors, no-anchor, and random-anchor controls.",
            "If clean TopK anchors improve center_pca regeneration, GeometryAnchor_geo is supported.",
        ],
    }
    if len(audit_df):
        result["filter_audit"] = audit_df.to_dict(orient="records")
    if len(fam_summary):
        result["quality"] = {
            "mean_clean_anchor_score": float(fam_summary["mean_clean_anchor_score"].mean()),
            "mean_specificity_ratio": float(fam_summary["mean_specificity_ratio"].mean()),
            "mean_prior_match_frac": float(fam_summary["prior_match_frac"].mean()),
        }
        for _, r in fam_summary.iterrows():
            result["top_anchor_preview"][r["seed_family"]] = json.loads(r["clean_anchor_terms_json"])
    return result


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_dir(cfg.OUTPUT_DIR)
    outdir = Path(cfg.OUTPUT_DIR)
    json_dump(asdict(cfg), outdir / "sem3d3b_config.json")

    raw_scores, raw_anchors, raw_family_summary, raw_diag = load_inputs(cfg)
    clean_scores, audit = clean_scores_func(raw_scores, cfg) if False else (None, None)

    # Use separate name to avoid shadowing.
    clean_scores_df, audit_dict = clean_scores_from_raw(raw_scores, cfg)

    anchors = select_clean_anchors(clean_scores_df, cfg)
    fam_summary = build_family_summary(anchors, cfg)
    prompt_candidates = build_clean_anchor_prompt_candidates(fam_summary)
    ablations = build_clean_anchor_ablation_prompts(fam_summary)
    audit_df = build_filter_audit(raw_scores, clean_scores_df, audit_dict)

    clean_scores_df.to_csv(outdir / "sem3d3b_clean_anchor_scores.csv", index=False, encoding="utf-8-sig")
    anchors.to_csv(outdir / "sem3d3b_clean_anchor_terms.csv", index=False, encoding="utf-8-sig")
    fam_summary.to_csv(outdir / "sem3d3b_family_clean_anchor_summary.csv", index=False, encoding="utf-8-sig")
    prompt_candidates.to_csv(outdir / "sem3d3b_clean_anchor_prompt_candidates.csv", index=False, encoding="utf-8-sig")
    ablations.to_csv(outdir / "sem3d3b_clean_anchor_ablation_prompts.csv", index=False, encoding="utf-8-sig")
    audit_df.to_csv(outdir / "sem3d3b_filter_audit.csv", index=False, encoding="utf-8-sig")

    summary = build_results_summary(cfg, audit_df, clean_scores_df, anchors, fam_summary, prompt_candidates)
    json_dump(summary, outdir / "sem3d3b_results_summary.json")

    print("=" * 100)
    print("SEM-3D.3b: Semantic TopK Anchor Filtering")
    print("=" * 100)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nOutput files:")
    for p in [
        "sem3d3b_config.json",
        "sem3d3b_clean_anchor_scores.csv",
        "sem3d3b_clean_anchor_terms.csv",
        "sem3d3b_family_clean_anchor_summary.csv",
        "sem3d3b_clean_anchor_prompt_candidates.csv",
        "sem3d3b_clean_anchor_ablation_prompts.csv",
        "sem3d3b_filter_audit.csv",
        "sem3d3b_results_summary.json",
    ]:
        print("  -", outdir / p)


# alias because Python allows shadowing issue above avoided here
def clean_scores_from_raw(raw_scores: pd.DataFrame, cfg: CFG):
    return clean_scores(raw_scores, cfg)


if __name__ == "__main__":
    main()
