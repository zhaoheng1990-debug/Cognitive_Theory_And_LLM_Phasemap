# -*- coding: utf-8 -*-
"""
GPT_209_SEM_3D1_geometry_anchor_extraction.py

SEM-3D.1: GeometryAnchor Extraction Audit
-----------------------------------------

Goal:
    Extract GeometryAnchor_F for each SeedFamily.

Motivation:
    SEM-3B.4 showed:
      - Concept/Operator/SeedBasis can regenerate scalar trajectory profile.
      - But they fail to regenerate TopK center / center_pca geometry.
    Hypothesis:
      MemoryUnit_F = Concept_F + OperatorID_F + GeometryAnchor_F

GeometryAnchor:
    A family-specific semantic-neighborhood anchor extracted from:
      1. prompt text tokens / terms
      2. candidate Prompt* structure text
      3. optional tokenizer-level lexical units
      4. family-specificity against other families

This script does NOT run the model.
It builds candidate anchors and anchor prompts for SEM-3D.2.

Inputs:
    sem3a_outputs/
      sem3a_dataset.csv
      sem3a_features.csv                    [optional, for metadata only]

    sem3c0b_outputs/
      sem3c0b_prompt_star_candidates.csv
      sem3c0b_family_invariant_scores.csv   [optional]
      sem3c0b_center_vs_canonical.csv       [optional]

Outputs:
    sem3d1_outputs/
      sem3d1_config.json
      sem3d1_anchor_terms.csv
      sem3d1_anchor_scores.csv
      sem3d1_family_anchor_summary.csv
      sem3d1_anchor_prompt_candidates.csv
      sem3d1_anchor_ablation_prompts.csv
      sem3d1_results_summary.json

Run:
    python GPT_209_SEM_3D1_geometry_anchor_extraction.py

Next:
    SEM-3D.2 should forward:
      Concept+Operator
      Concept+Operator+GeometryAnchor
      random-family anchor
      anchor-only
    and compare center_pca regeneration.
"""

import json
import re
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

@dataclass
class CFG:
    SEM3A_DIR: str = "./sem3a_outputs"
    SEM3C0B_DIR: str = "./sem3c0b_outputs"
    OUTPUT_DIR: str = "./sem3d1_outputs"

    SEM3A_DATASET: str = "sem3a_dataset.csv"
    SEM3C0B_PROMPTSTAR: str = "sem3c0b_prompt_star_candidates.csv"
    SEM3C0B_INVARIANT: str = "sem3c0b_family_invariant_scores.csv"
    SEM3C0B_CENTER_VS_CANONICAL: str = "sem3c0b_center_vs_canonical.csv"

    # Anchor extraction
    MIN_TOKEN_LEN: int = 3
    TOP_N_ANCHORS: int = 24
    TOP_N_PROMPT_ANCHORS: int = 12
    ALPHA_OTHER_FREQ: float = 0.35
    EPS: float = 1e-5

    # Include unigrams and bigrams from text.
    USE_BIGRAMS: bool = True

    # Main block used for optional family quality metadata.
    MAIN_BLOCK: str = "center_pca"

    RANDOM_SEED: int = 42


cfg = CFG()


# ============================================================
# STRUCTURAL PRIOR MAP
# ============================================================

def seed_family_structure_map() -> Dict[str, Dict[str, str]]:
    """
    Same family ontology as 3C.0b, but used here only as a structural prior.
    Anchor extraction remains text/frequency driven.
    """
    return {
        "inertia_mechanism": {
            "concept_star": "inertia / persistent motion",
            "operator_id": "mechanism_explanation",
            "domain_frame": "classical_physics",
            "common_structure": "state of motion persists unless acted on by a net external force",
            "manual_anchor_terms": "inertia motion force rest uniform motion external force object state mechanics Newton first law",
        },
        "photosynthesis_mechanism": {
            "concept_star": "photosynthesis",
            "operator_id": "mechanism_explanation",
            "domain_frame": "biology",
            "common_structure": "plants convert light, carbon dioxide, and water into chemical energy",
            "manual_anchor_terms": "photosynthesis plants light sunlight carbon dioxide water chlorophyll sugar chemical energy green plants",
        },
        "democracy_definition": {
            "concept_star": "democracy",
            "operator_id": "definition",
            "domain_frame": "politics",
            "common_structure": "political system involving citizen participation, representation, and collective rule",
            "manual_anchor_terms": "democracy democratic government citizen participation representation collective rule society political system",
        },
        "gravity_causal": {
            "concept_star": "gravity",
            "operator_id": "causal_explanation",
            "domain_frame": "physics",
            "common_structure": "mass attracts mass and shapes falling and orbital motion",
            "manual_anchor_terms": "gravity mass attraction objects fall earth planets orbit stars gravitational motion force",
        },
        "python_use_cases": {
            "concept_star": "Python programming language",
            "operator_id": "list_use_cases",
            "domain_frame": "software",
            "common_structure": "common practical applications and domains of Python",
            "manual_anchor_terms": "Python programming language software development data analysis automation scripting machine learning web applications",
        },
        "internet_risk": {
            "concept_star": "internet",
            "operator_id": "risk_audit",
            "domain_frame": "technology_society",
            "common_structure": "risks, vulnerabilities, and failure modes of internet systems and use",
            "manual_anchor_terms": "internet online systems risks vulnerabilities cybersecurity privacy connectivity social dangers failure modes",
        },
        "climate_plan": {
            "concept_star": "climate change",
            "operator_id": "planning",
            "domain_frame": "environment_policy",
            "common_structure": "practical mitigation and adaptation strategy",
            "manual_anchor_terms": "climate change mitigation adaptation emissions energy policy communities governments risks strategy action plan",
        },
        "transformer_mechanism": {
            "concept_star": "Transformer model",
            "operator_id": "mechanism_explanation",
            "domain_frame": "machine_learning",
            "common_structure": "attention, token representations, and layered computation",
            "manual_anchor_terms": "Transformer model attention self-attention tokens context layers representations neural network architecture",
        },
        "market_compare": {
            "concept_star": "market coordination",
            "operator_id": "comparison",
            "domain_frame": "economics",
            "common_structure": "compare decentralized market coordination with centralized planning",
            "manual_anchor_terms": "market economy central planning price signals allocation coordination decentralized command system compare contrast",
        },
        "memory_counterfactual": {
            "concept_star": "memory",
            "operator_id": "counterfactual",
            "domain_frame": "cognition",
            "common_structure": "consequences of absent, altered, or unreliable memory",
            "manual_anchor_terms": "memory cognition mind long-term memory experience unreliable absent altered consequences counterfactual humans",
        },
        "ocean_property": {
            "concept_star": "ocean",
            "operator_id": "property_description",
            "domain_frame": "earth_science",
            "common_structure": "major physical and ecological properties of oceans",
            "manual_anchor_terms": "ocean oceans earth water marine ecological physical properties climate currents salinity ecosystems",
        },
        "robot_relation": {
            "concept_star": "robot",
            "operator_id": "relation_mapping",
            "domain_frame": "robotics",
            "common_structure": "relation among robots, automation, sensors, control, and machines",
            "manual_anchor_terms": "robot robotics automation machines sensors control actuators technology functional relations autonomous systems",
        },
    }


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
    sem3a_dir = Path(cfg.SEM3A_DIR)
    sem3c_dir = Path(cfg.SEM3C0B_DIR)

    dataset_path = sem3a_dir / cfg.SEM3A_DATASET
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing {dataset_path}")

    dataset = pd.read_csv(dataset_path)

    promptstar_path = sem3c_dir / cfg.SEM3C0B_PROMPTSTAR
    promptstars = pd.read_csv(promptstar_path) if promptstar_path.exists() else pd.DataFrame()

    inv_path = sem3c_dir / cfg.SEM3C0B_INVARIANT
    invariant = pd.read_csv(inv_path) if inv_path.exists() else pd.DataFrame()

    cvc_path = sem3c_dir / cfg.SEM3C0B_CENTER_VS_CANONICAL
    center_vs_canonical = pd.read_csv(cvc_path) if cvc_path.exists() else pd.DataFrame()

    return dataset, promptstars, invariant, center_vs_canonical


# ============================================================
# TEXT ANCHOR EXTRACTION
# ============================================================

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "what", "why", "how", "are", "is", "was",
    "were", "will", "would", "could", "should", "into", "from", "about", "over", "under",
    "between", "among", "through", "using", "use", "used", "does", "do", "did", "give",
    "explain", "describe", "define", "list", "name", "identify", "make", "create",
    "analyze", "analysis", "concept", "operator", "function", "common", "structure",
    "instruction", "answer", "prompt", "task", "thing", "things", "major", "main",
    "several", "short", "clear", "concise", "practical", "associated", "related",
    "relevant", "important", "key", "focus", "focusing", "including", "without",
    "unless", "acted", "upon", "one", "two", "three", "their", "its", "itself",
}


def normalize_text(text: str) -> str:
    text = str(text)
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def tokenize_terms(text: str, min_len: int = 3, use_bigrams: bool = True) -> List[str]:
    text = normalize_text(text)
    toks = [t for t in text.split() if len(t) >= min_len and t not in STOPWORDS and not t.isdigit()]
    out = list(toks)
    if use_bigrams:
        for a, b in zip(toks[:-1], toks[1:]):
            if a not in STOPWORDS and b not in STOPWORDS:
                out.append(f"{a}_{b}")
    return out


def build_family_corpus(dataset: pd.DataFrame, promptstars: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Text corpus per family:
      - original prompts
      - canonical prompt
      - promptstar candidates
      - manual structure priors
    """
    fmap = seed_family_structure_map()
    corpus = defaultdict(list)

    for _, r in dataset.iterrows():
        fam = r["seed_family"]
        corpus[fam].append(str(r["prompt"]))

    if len(promptstars):
        for _, r in promptstars.iterrows():
            fam = r["seed_family"]
            if "prompt_star_candidate" in r:
                corpus[fam].append(str(r["prompt_star_candidate"]))
            for col in ["concept_star", "operator_id", "common_structure", "avoid_surface"]:
                if col in r and pd.notna(r[col]):
                    corpus[fam].append(str(r[col]))

    for fam, meta in fmap.items():
        for key in ["concept_star", "operator_id", "domain_frame", "common_structure", "manual_anchor_terms"]:
            if key in meta:
                corpus[fam].append(str(meta[key]))

    return corpus


def compute_anchor_scores(corpus: Dict[str, List[str]], cfg: CFG):
    fam_term_counts = {}
    fam_total = {}
    all_fams = sorted(corpus.keys())

    for fam, docs in corpus.items():
        counter = Counter()
        for doc in docs:
            counter.update(tokenize_terms(doc, cfg.MIN_TOKEN_LEN, cfg.USE_BIGRAMS))
        fam_term_counts[fam] = counter
        fam_total[fam] = sum(counter.values())

    global_counts = Counter()
    for c in fam_term_counts.values():
        global_counts.update(c)
    global_total = sum(global_counts.values())

    rows = []
    for fam in all_fams:
        counter = fam_term_counts[fam]
        total = fam_total[fam] + cfg.EPS
        other_counts = global_counts - counter
        other_total = (global_total - fam_total[fam]) + cfg.EPS

        for term, count in counter.items():
            freq_f = count / total
            freq_other = other_counts.get(term, 0) / other_total
            ratio = (freq_f + cfg.EPS) / (freq_other + cfg.EPS)
            log_ratio = math.log(ratio)
            diff = freq_f - cfg.ALPHA_OTHER_FREQ * freq_other

            # Combined score rewards within-family frequency and family specificity.
            score = diff * (1.0 + max(0.0, log_ratio))

            rows.append({
                "seed_family": fam,
                "term": term,
                "count_family": int(count),
                "freq_family": float(freq_f),
                "count_other": int(other_counts.get(term, 0)),
                "freq_other": float(freq_other),
                "specificity_ratio": float(ratio),
                "log_specificity_ratio": float(log_ratio),
                "anchor_score": float(score),
            })

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["seed_family", "anchor_score"], ascending=[True, False])
    return df


def select_anchor_terms(anchor_scores: pd.DataFrame, cfg: CFG):
    rows = []
    for fam, g in anchor_scores.groupby("seed_family"):
        g = g.sort_values("anchor_score", ascending=False).copy()
        top = g.head(cfg.TOP_N_ANCHORS)
        rank = 1
        for _, r in top.iterrows():
            rows.append({**r.to_dict(), "anchor_rank": rank})
            rank += 1
    return pd.DataFrame(rows)


# ============================================================
# FAMILY SUMMARY / PROMPT CANDIDATES
# ============================================================

def get_family_quality(invariant: pd.DataFrame, center_vs_canonical: pd.DataFrame, cfg: CFG):
    quality = {}
    if len(invariant):
        inv = invariant[invariant["block"] == cfg.MAIN_BLOCK] if "block" in invariant.columns else invariant
        for _, r in inv.iterrows():
            quality.setdefault(r["seed_family"], {}).update({
                "invariant_ratio": r.get("invariant_ratio", np.nan),
                "separation_ratio": r.get("separation_ratio", np.nan),
                "within_l2_mean": r.get("within_l2_mean", np.nan),
            })
    if len(center_vs_canonical):
        cvc = center_vs_canonical[center_vs_canonical["block"] == cfg.MAIN_BLOCK] if "block" in center_vs_canonical.columns else center_vs_canonical
        for _, r in cvc.iterrows():
            quality.setdefault(r["seed_family"], {}).update({
                "center_cos_gain": r.get("center_cos_gain", np.nan),
                "center_l2_gain": r.get("center_l2_gain", np.nan),
                "center_better_cos_frac": r.get("center_better_cos_frac", np.nan),
            })
    return quality


def build_family_anchor_summary(anchor_terms: pd.DataFrame, invariant: pd.DataFrame, cvc: pd.DataFrame, cfg: CFG):
    fmap = seed_family_structure_map()
    quality = get_family_quality(invariant, cvc, cfg)

    rows = []
    for fam in sorted(anchor_terms["seed_family"].unique()):
        top = anchor_terms[anchor_terms["seed_family"] == fam].sort_values("anchor_rank")
        meta = fmap.get(fam, {})
        terms = top["term"].head(cfg.TOP_N_PROMPT_ANCHORS).tolist()

        row = {
            "seed_family": fam,
            "concept_star": meta.get("concept_star", ""),
            "operator_id": meta.get("operator_id", ""),
            "domain_frame": meta.get("domain_frame", ""),
            "common_structure": meta.get("common_structure", ""),
            "anchor_terms_top": ", ".join(terms),
            "anchor_terms_json": json.dumps(terms, ensure_ascii=False),
            "n_anchor_terms": int(len(terms)),
        }
        row.update(quality.get(fam, {}))
        rows.append(row)

    return pd.DataFrame(rows)


def build_anchor_prompt_candidates(summary: pd.DataFrame, cfg: CFG):
    rows = []
    families = summary["seed_family"].tolist()
    fam_to_terms = {
        r["seed_family"]: json.loads(r["anchor_terms_json"])
        for _, r in summary.iterrows()
    }

    for _, r in summary.iterrows():
        fam = r["seed_family"]
        concept = r.get("concept_star", fam)
        operator = r.get("operator_id", "")
        common = r.get("common_structure", "")
        anchors = json.loads(r.get("anchor_terms_json", "[]"))
        anchor_text = ", ".join(anchors)

        # Same-family anchors
        rows.append({
            "seed_family": fam,
            "candidate_type": "concept_operator_anchor",
            "anchor_family": fam,
            "prompt": (
                f"Concept: {concept}\n"
                f"OperatorID: {operator}\n"
                f"GeometryAnchor: {anchor_text}\n"
                f"Use the semantic neighborhood indicated by the GeometryAnchor to answer."
            ),
            "concept_star": concept,
            "operator_id": operator,
            "anchor_terms": anchor_text,
            "common_structure": common,
        })

        rows.append({
            "seed_family": fam,
            "candidate_type": "natural_anchor_prompt",
            "anchor_family": fam,
            "prompt": (
                f"Using the concepts {anchor_text}, explain {concept} with the operation {operator}. "
                f"Preserve the shared structure: {common}."
            ),
            "concept_star": concept,
            "operator_id": operator,
            "anchor_terms": anchor_text,
            "common_structure": common,
        })

        rows.append({
            "seed_family": fam,
            "candidate_type": "anchor_only",
            "anchor_family": fam,
            "prompt": f"GeometryAnchor terms: {anchor_text}. Explain the shared semantic structure.",
            "concept_star": concept,
            "operator_id": operator,
            "anchor_terms": anchor_text,
            "common_structure": common,
        })

        rows.append({
            "seed_family": fam,
            "candidate_type": "concept_operator_no_anchor",
            "anchor_family": "",
            "prompt": f"Concept: {concept}\nOperatorID: {operator}\nAnswer according to this concept and operation.",
            "concept_star": concept,
            "operator_id": operator,
            "anchor_terms": "",
            "common_structure": common,
        })

        # Deterministic random-family anchor control: choose next family in sorted list.
        if len(families) > 1:
            idx = families.index(fam)
            rand_fam = families[(idx + 1) % len(families)]
            rand_anchor_text = ", ".join(fam_to_terms.get(rand_fam, []))
            rows.append({
                "seed_family": fam,
                "candidate_type": "concept_operator_random_anchor",
                "anchor_family": rand_fam,
                "prompt": (
                    f"Concept: {concept}\n"
                    f"OperatorID: {operator}\n"
                    f"GeometryAnchor: {rand_anchor_text}\n"
                    f"Use the semantic neighborhood indicated by the GeometryAnchor to answer."
                ),
                "concept_star": concept,
                "operator_id": operator,
                "anchor_terms": rand_anchor_text,
                "common_structure": common,
            })

    return pd.DataFrame(rows)


def build_anchor_ablation_prompts(summary: pd.DataFrame):
    rows = []
    for _, r in summary.iterrows():
        fam = r["seed_family"]
        concept = r.get("concept_star", fam)
        operator = r.get("operator_id", "")
        common = r.get("common_structure", "")
        anchors = ", ".join(json.loads(r.get("anchor_terms_json", "[]")))

        ablations = {
            "concept_only": f"Concept: {concept}\nAnswer about this concept.",
            "operator_only": f"OperatorID: {operator}\nApply this operation to an appropriate concept.",
            "anchor_only": f"GeometryAnchor: {anchors}\nAnswer using this semantic neighborhood.",
            "concept_operator": f"Concept: {concept}\nOperatorID: {operator}\nAnswer accordingly.",
            "concept_anchor": f"Concept: {concept}\nGeometryAnchor: {anchors}\nAnswer using this semantic neighborhood.",
            "operator_anchor": f"OperatorID: {operator}\nGeometryAnchor: {anchors}\nAnswer using this semantic neighborhood.",
            "concept_operator_anchor": (
                f"Concept: {concept}\nOperatorID: {operator}\nGeometryAnchor: {anchors}\n"
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

def build_results_summary(anchor_terms, family_summary, prompt_candidates, ablation_prompts):
    result = {
        "config": asdict(cfg),
        "n_families": int(family_summary["seed_family"].nunique()) if len(family_summary) else 0,
        "n_anchor_terms": int(len(anchor_terms)),
        "n_prompt_candidates": int(len(prompt_candidates)),
        "n_ablation_prompts": int(len(ablation_prompts)),
        "verdict": "ANCHOR_CANDIDATES_READY",
        "main_block": cfg.MAIN_BLOCK,
        "quality_summary": {},
        "interpretation": [
            "GeometryAnchor terms are family-specific lexical/semantic anchors extracted from original prompts, Prompt* candidates, and structural priors.",
            "SEM-3D.2 must forward these anchor prompts and test whether Concept+Operator+GeometryAnchor improves center_pca regeneration.",
            "Random-anchor and ablation prompts are included for control.",
        ],
    }

    if len(family_summary):
        for col in ["invariant_ratio", "separation_ratio", "center_cos_gain", "center_l2_gain"]:
            if col in family_summary.columns:
                result["quality_summary"][f"mean_{col}"] = float(pd.to_numeric(family_summary[col], errors="coerce").mean())

    # Top anchors per family.
    top_preview = {}
    for fam, g in anchor_terms.groupby("seed_family"):
        top_preview[fam] = g.sort_values("anchor_rank")["term"].head(8).tolist()
    result["top_anchor_preview"] = top_preview

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_dir(cfg.OUTPUT_DIR)
    outdir = Path(cfg.OUTPUT_DIR)

    dataset, promptstars, invariant, cvc = load_inputs(cfg)
    corpus = build_family_corpus(dataset, promptstars)
    scores = compute_anchor_scores(corpus, cfg)
    anchors = select_anchor_terms(scores, cfg)

    family_summary = build_family_anchor_summary(anchors, invariant, cvc, cfg)
    prompt_candidates = build_anchor_prompt_candidates(family_summary, cfg)
    ablation_prompts = build_anchor_ablation_prompts(family_summary)

    json_dump(asdict(cfg), outdir / "sem3d1_config.json")
    anchors.to_csv(outdir / "sem3d1_anchor_terms.csv", index=False, encoding="utf-8-sig")
    scores.to_csv(outdir / "sem3d1_anchor_scores.csv", index=False, encoding="utf-8-sig")
    family_summary.to_csv(outdir / "sem3d1_family_anchor_summary.csv", index=False, encoding="utf-8-sig")
    prompt_candidates.to_csv(outdir / "sem3d1_anchor_prompt_candidates.csv", index=False, encoding="utf-8-sig")
    ablation_prompts.to_csv(outdir / "sem3d1_anchor_ablation_prompts.csv", index=False, encoding="utf-8-sig")

    summary = build_results_summary(anchors, family_summary, prompt_candidates, ablation_prompts)
    json_dump(summary, outdir / "sem3d1_results_summary.json")

    print("=" * 100)
    print("SEM-3D.1: GeometryAnchor Extraction Audit")
    print("=" * 100)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nOutput files:")
    for p in [
        "sem3d1_config.json",
        "sem3d1_anchor_terms.csv",
        "sem3d1_anchor_scores.csv",
        "sem3d1_family_anchor_summary.csv",
        "sem3d1_anchor_prompt_candidates.csv",
        "sem3d1_anchor_ablation_prompts.csv",
        "sem3d1_results_summary.json",
    ]:
        print("  -", outdir / p)


if __name__ == "__main__":
    main()
