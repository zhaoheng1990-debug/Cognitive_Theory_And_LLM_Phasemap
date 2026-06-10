# -*- coding: utf-8 -*-
"""
GPT_203B_SEM_3C0_maximal_common_structure_extraction.py

SEM-3C.0b: Maximal Common Structure Extraction
----------------------------------------------

Goal:
    Extract maximal common structure from each SeedFamily to construct Prompt*.

Theory:
    We should NOT recover the exact original prompt.
    We should NOT rely only on a hand-written canonical prompt.
    We should NOT only cluster.

    Instead:
        SeedFamily
          -> trajectory-space family center μ_F
          -> invariant/common structure
          -> PromptStructure
          -> Prompt*

Inputs from SEM-3A:
    sem3a_outputs/
      sem3a_dataset.csv
      sem3a_features.csv
      sem3a_prototype_recovery.csv
      sem3a_similarity_audit.csv
      sem3a_results_summary.json

Main operations:
    1. Build trajectory feature blocks.
    2. For each block, compute family centers μ_F from original prompts.
    3. Audit compactness, separation, invariant ratio.
    4. Compare family center μ_F with hand-written canonical seed trajectory.
    5. Compute member residuals ε_i = T_i - μ_F.
    6. Run small family-internal subcluster audit.
    7. Generate Prompt* candidates from family center + structural prior.

Outputs:
    sem3c0b_outputs/
      sem3c0b_config.json
      sem3c0b_family_center_features.csv
      sem3c0b_family_invariant_scores.csv
      sem3c0b_family_member_residuals.csv
      sem3c0b_family_subcluster_audit.csv
      sem3c0b_center_vs_canonical.csv
      sem3c0b_prompt_star_candidates.csv
      sem3c0b_results_summary.json

Run:
    python GPT_203B_SEM_3C0_maximal_common_structure_extraction.py
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIG
# ============================================================

@dataclass
class CFG:
    INPUT_DIR: str = "./sem3a_outputs"
    OUTPUT_DIR: str = "./sem3c0b_outputs"

    DATASET_CSV: str = "sem3a_dataset.csv"
    FEATURES_CSV: str = "sem3a_features.csv"
    PROTOTYPE_CSV: str = "sem3a_prototype_recovery.csv"
    SIMILARITY_CSV: str = "sem3a_similarity_audit.csv"
    SUMMARY_JSON: str = "sem3a_results_summary.json"

    # Main block from SEM-3A result. center_pca had best prototype recovery.
    MAIN_BLOCK: str = "center_pca"

    # Blocks to audit.
    BLOCKS: Tuple[str, ...] = ("center_pca", "decision", "mid", "all", "scalar", "transport", "init")

    # Family center built from original prompts, not canonical rows.
    CENTER_SOURCE_ROW_TYPE: str = "original_prompt"

    # Subcluster settings.
    MAX_SUBCLUSTERS: int = 3
    RANDOM_SEED: int = 42


cfg = CFG()


# ============================================================
# STRUCTURAL PRIOR MAP
# ============================================================

def seed_family_structure_map() -> Dict[str, Dict[str, str]]:
    return {
        "inertia_mechanism": {
            "concept_star": "inertia / persistent motion",
            "concept_family": "physics_motion",
            "operator_id": "mechanism_explanation",
            "operator_family": "mechanism",
            "question_form_core": "why_how_explain",
            "specificity_level": "physics_law",
            "domain_frame": "classical_physics",
            "constraint_polarity": "explanatory",
            "common_structure": "state of motion persists unless acted on by a net external force",
            "avoid_surface": "avoid tying the prompt only to the phrase Newton's first law",
            "prompt_star": "Explain the mechanism of inertia: why an object tends to preserve its state of rest or uniform motion unless acted on by a net external force.",
        },
        "photosynthesis_mechanism": {
            "concept_star": "photosynthesis",
            "concept_family": "biology_process",
            "operator_id": "mechanism_explanation",
            "operator_family": "mechanism",
            "question_form_core": "how_explain",
            "specificity_level": "biological_process",
            "domain_frame": "biology",
            "constraint_polarity": "explanatory",
            "common_structure": "plants convert light, carbon dioxide, and water into chemical energy",
            "avoid_surface": "avoid limiting to only chlorophyll wording",
            "prompt_star": "Explain how photosynthesis works as a mechanism for converting light, carbon dioxide, and water into chemical energy.",
        },
        "democracy_definition": {
            "concept_star": "democracy",
            "concept_family": "political_concept",
            "operator_id": "definition",
            "operator_family": "definition",
            "question_form_core": "what_define",
            "specificity_level": "general_concept",
            "domain_frame": "politics",
            "constraint_polarity": "identity",
            "common_structure": "political system involving citizen participation, representation, and collective rule",
            "avoid_surface": "avoid committing to one governmental form only",
            "prompt_star": "Define democracy as a political system based on citizen participation, representation, and collective rule.",
        },
        "gravity_causal": {
            "concept_star": "gravity",
            "concept_family": "physics_force",
            "operator_id": "causal_explanation",
            "operator_family": "causal",
            "question_form_core": "why_cause",
            "specificity_level": "physics_law",
            "domain_frame": "physics",
            "constraint_polarity": "causal",
            "common_structure": "mass attracts mass and shapes falling and orbital motion",
            "avoid_surface": "avoid only Earth-fall framing",
            "prompt_star": "Give a causal explanation of gravity, focusing on how mass attracts mass and shapes falling and orbital motion.",
        },
        "python_use_cases": {
            "concept_star": "Python programming language",
            "concept_family": "technology_tool",
            "operator_id": "list_use_cases",
            "operator_family": "enumeration",
            "question_form_core": "list_examples",
            "specificity_level": "practical_domains",
            "domain_frame": "software",
            "constraint_polarity": "expansive",
            "common_structure": "common practical applications and domains of Python",
            "avoid_surface": "avoid only software development wording",
            "prompt_star": "List common practical use cases and application areas for the Python programming language.",
        },
        "internet_risk": {
            "concept_star": "internet",
            "concept_family": "technology_infrastructure",
            "operator_id": "risk_audit",
            "operator_family": "audit",
            "question_form_core": "risk_audit",
            "specificity_level": "systemic_risk",
            "domain_frame": "technology_society",
            "constraint_polarity": "risk_oriented",
            "common_structure": "risks, vulnerabilities, and failure modes of internet systems and use",
            "avoid_surface": "avoid only cybersecurity or only social framing",
            "prompt_star": "Audit the major risks, vulnerabilities, and failure modes associated with the internet.",
        },
        "climate_plan": {
            "concept_star": "climate change",
            "concept_family": "environmental_system",
            "operator_id": "planning",
            "operator_family": "planning",
            "question_form_core": "plan_steps",
            "specificity_level": "multi_step_strategy",
            "domain_frame": "environment_policy",
            "constraint_polarity": "action_oriented",
            "common_structure": "practical mitigation and adaptation strategy",
            "avoid_surface": "avoid only government or only community framing",
            "prompt_star": "Propose a practical multi-step plan for climate change mitigation and adaptation.",
        },
        "transformer_mechanism": {
            "concept_star": "Transformer model",
            "concept_family": "ai_model_architecture",
            "operator_id": "mechanism_explanation",
            "operator_family": "mechanism",
            "question_form_core": "how_explain",
            "specificity_level": "technical_architecture",
            "domain_frame": "machine_learning",
            "constraint_polarity": "mechanistic",
            "common_structure": "attention, token representations, and layered computation",
            "avoid_surface": "avoid only self-attention wording",
            "prompt_star": "Explain how a Transformer model works, focusing on attention, token representations, and layered computation.",
        },
        "market_compare": {
            "concept_star": "market coordination",
            "concept_family": "economic_system",
            "operator_id": "comparison",
            "operator_family": "comparison",
            "question_form_core": "compare_contrast",
            "specificity_level": "conceptual_comparison",
            "domain_frame": "economics",
            "constraint_polarity": "contrastive",
            "common_structure": "compare decentralized market coordination with centralized planning",
            "avoid_surface": "avoid only price signals or only command systems",
            "prompt_star": "Compare market coordination with centralized planning, focusing on similarities and differences.",
        },
        "memory_counterfactual": {
            "concept_star": "memory",
            "concept_family": "cognitive_function",
            "operator_id": "counterfactual",
            "operator_family": "counterfactual",
            "question_form_core": "what_if_imagine",
            "specificity_level": "hypothetical_system",
            "domain_frame": "cognition",
            "constraint_polarity": "counterfactual",
            "common_structure": "consequences of absent, altered, or unreliable memory",
            "avoid_surface": "avoid only no-memory framing",
            "prompt_star": "Analyze a counterfactual scenario in which memory is absent, altered, or unreliable.",
        },
        "ocean_property": {
            "concept_star": "ocean",
            "concept_family": "natural_system",
            "operator_id": "property_description",
            "operator_family": "property",
            "question_form_core": "describe_features",
            "specificity_level": "property_summary",
            "domain_frame": "earth_science",
            "constraint_polarity": "descriptive",
            "common_structure": "major physical and ecological properties of oceans",
            "avoid_surface": "avoid only physical or only ecological framing",
            "prompt_star": "Describe the major physical and ecological properties that characterize Earth's oceans.",
        },
        "robot_relation": {
            "concept_star": "robot",
            "concept_family": "technology_agent",
            "operator_id": "relation_mapping",
            "operator_family": "relation",
            "question_form_core": "relation_domain_mapping",
            "specificity_level": "domain_mapping",
            "domain_frame": "robotics",
            "constraint_polarity": "relational",
            "common_structure": "relation among robots, automation, sensors, control, and machines",
            "avoid_surface": "avoid only automation or only machine framing",
            "prompt_star": "Map the concept of a robot to its broader technological relations, including automation, sensors, control, and machines.",
        },
    }


# ============================================================
# IO
# ============================================================

def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
    dataset_path = indir / cfg.DATASET_CSV
    features_path = indir / cfg.FEATURES_CSV
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing {dataset_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}")

    dataset = pd.read_csv(dataset_path)
    features = pd.read_csv(features_path)

    proto = pd.read_csv(indir / cfg.PROTOTYPE_CSV) if (indir / cfg.PROTOTYPE_CSV).exists() else pd.DataFrame()
    sim = pd.read_csv(indir / cfg.SIMILARITY_CSV) if (indir / cfg.SIMILARITY_CSV).exists() else pd.DataFrame()
    summary = read_json(indir / cfg.SUMMARY_JSON)

    # Avoid duplicate metadata columns if features already contains dataset columns.
    if "row_id" in features.columns and "prompt" in features.columns:
        merged = features.copy()
    else:
        merged = dataset.merge(features, on="row_id", how="left")

    return dataset, merged, proto, sim, summary


# ============================================================
# FEATURE BLOCKS
# ============================================================

def get_feature_cols(df: pd.DataFrame, block: str) -> List[str]:
    exclude = {
        "row_id", "row_type", "seed_family", "concept_star",
        "operator_star", "surface_id", "prompt"
    }
    numeric = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    if block == "all":
        return numeric
    if block == "center_pca":
        return [c for c in numeric if "centerPC" in c]
    if block == "scalar":
        return [c for c in numeric if "centerPC" not in c]
    if block == "transport":
        return [c for c in numeric if any(k in c for k in ["shift", "jaccard"])]
    if block == "init":
        return [c for c in numeric if any(f"_L{i}_" in c for i in range(0, 7))]
    if block == "mid":
        return [c for c in numeric if any(f"_L{i}_" in c for i in range(7, 20))]
    if block == "decision":
        return [c for c in numeric if any(f"_L{i}_" in c for i in range(20, 26))]
    raise ValueError(block)


def make_X(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def standardize_block(df: pd.DataFrame, cols: List[str]):
    X = make_X(df, cols).values
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    return Xz, scaler


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ============================================================
# FAMILY CENTER EXTRACTION
# ============================================================

def compute_family_centers(features: pd.DataFrame, block: str):
    cols = get_feature_cols(features, block)
    if not cols:
        return pd.DataFrame(), pd.DataFrame(), None

    Xz, scaler = standardize_block(features, cols)
    Xdf = pd.DataFrame(Xz, columns=cols)
    meta = features.reset_index(drop=True)

    original_mask = meta["row_type"] == "original_prompt"
    canonical_mask = meta["row_type"] == "canonical_minimal_seed"

    center_rows = []
    residual_rows = []
    canonical_rows = []

    for fam in sorted(meta["seed_family"].unique()):
        idx_orig = np.where((meta["seed_family"].values == fam) & original_mask.values)[0]
        idx_can = np.where((meta["seed_family"].values == fam) & canonical_mask.values)[0]
        if len(idx_orig) == 0:
            continue

        Xfam = Xz[idx_orig, :]
        mu = Xfam.mean(axis=0)
        compact_cos = np.mean([cosine(x, mu) for x in Xfam])
        compact_l2 = np.mean(np.linalg.norm(Xfam - mu.reshape(1, -1), axis=1))

        row = {
            "seed_family": fam,
            "block": block,
            "n_members": int(len(idx_orig)),
            "compact_cos_mean": float(compact_cos),
            "compact_l2_mean": float(compact_l2),
        }
        # Store center vector in compact json form for downstream scripts.
        row["center_vector_json"] = json.dumps(mu.tolist())
        center_rows.append(row)

        for idx in idx_orig:
            x = Xz[idx, :]
            eps = x - mu
            residual_rows.append({
                "row_id": int(meta.iloc[idx]["row_id"]),
                "seed_family": fam,
                "block": block,
                "prompt": meta.iloc[idx]["prompt"],
                "surface_id": int(meta.iloc[idx]["surface_id"]),
                "residual_l2": float(np.linalg.norm(eps)),
                "residual_cos_to_center": float(cosine(x, mu)),
                "residual_vector_json": json.dumps(eps.tolist()),
            })

        if len(idx_can):
            can_idx = idx_can[0]
            xcan = Xz[can_idx, :]
            canonical_rows.append({
                "seed_family": fam,
                "block": block,
                "canonical_row_id": int(meta.iloc[can_idx]["row_id"]),
                "canonical_prompt": meta.iloc[can_idx]["prompt"],
                "canonical_cos_to_family_center": float(cosine(xcan, mu)),
                "canonical_l2_to_family_center": float(np.linalg.norm(xcan - mu)),
            })

    centers = pd.DataFrame(center_rows)
    residuals = pd.DataFrame(residual_rows)
    canonicals = pd.DataFrame(canonical_rows)

    return centers, residuals, canonicals


def invariant_scores(features: pd.DataFrame, centers: pd.DataFrame, block: str):
    cols = get_feature_cols(features, block)
    if centers.empty or not cols:
        return pd.DataFrame()

    Xz, _ = standardize_block(features, cols)
    meta = features.reset_index(drop=True)
    original_mask = meta["row_type"] == "original_prompt"

    fams = sorted(meta["seed_family"].unique())
    center_vectors = {}
    for _, r in centers.iterrows():
        center_vectors[r["seed_family"]] = np.array(json.loads(r["center_vector_json"]), dtype=np.float32)

    # Global original variance.
    idx_orig_all = np.where(original_mask.values)[0]
    Xorig = Xz[idx_orig_all, :]
    global_var = float(np.mean(np.var(Xorig, axis=0)))

    rows = []
    for fam in fams:
        idx = np.where((meta["seed_family"].values == fam) & original_mask.values)[0]
        if len(idx) == 0 or fam not in center_vectors:
            continue

        Xfam = Xz[idx, :]
        mu = center_vectors[fam]
        within_var = float(np.mean(np.var(Xfam, axis=0)))
        within_l2 = float(np.mean(np.linalg.norm(Xfam - mu.reshape(1, -1), axis=1)))

        other_centers = [v for k, v in center_vectors.items() if k != fam]
        if other_centers:
            cos_to_others = np.array([cosine(mu, v) for v in other_centers])
            l2_to_others = np.array([np.linalg.norm(mu - v) for v in other_centers])
            nearest_other_cos = float(np.max(cos_to_others))
            nearest_other_l2 = float(np.min(l2_to_others))
            mean_other_l2 = float(np.mean(l2_to_others))
        else:
            nearest_other_cos = np.nan
            nearest_other_l2 = np.nan
            mean_other_l2 = np.nan

        # Higher is better: between center distance relative to within spread.
        separation_ratio = nearest_other_l2 / (within_l2 + 1e-9) if np.isfinite(nearest_other_l2) else np.nan

        # Between-family vs within-family variance.
        invariant_ratio = (global_var - within_var) / (within_var + 1e-9)

        rows.append({
            "seed_family": fam,
            "block": block,
            "n_members": int(len(idx)),
            "within_var": within_var,
            "global_var": global_var,
            "invariant_ratio": float(invariant_ratio),
            "within_l2_mean": within_l2,
            "nearest_other_center_cos": nearest_other_cos,
            "nearest_other_center_l2": nearest_other_l2,
            "mean_other_center_l2": mean_other_l2,
            "separation_ratio": float(separation_ratio),
        })

    return pd.DataFrame(rows)


def center_vs_canonical(features: pd.DataFrame, centers: pd.DataFrame, canonicals: pd.DataFrame, block: str):
    if centers.empty or canonicals.empty:
        return pd.DataFrame()

    cols = get_feature_cols(features, block)
    Xz, _ = standardize_block(features, cols)
    meta = features.reset_index(drop=True)

    center_map = {r["seed_family"]: np.array(json.loads(r["center_vector_json"]), dtype=np.float32) for _, r in centers.iterrows()}
    canonical_idx_map = {}
    for _, r in canonicals.iterrows():
        canonical_idx_map[r["seed_family"]] = int(np.where(meta["row_id"].values == r["canonical_row_id"])[0][0])

    rows = []
    for fam, mu in center_map.items():
        idx_orig = np.where((meta["seed_family"].values == fam) & (meta["row_type"].values == "original_prompt"))[0]
        if fam not in canonical_idx_map:
            continue
        xcan = Xz[canonical_idx_map[fam], :]

        cos_to_center = []
        cos_to_canonical = []
        l2_to_center = []
        l2_to_canonical = []
        for idx in idx_orig:
            x = Xz[idx, :]
            cos_to_center.append(cosine(x, mu))
            cos_to_canonical.append(cosine(x, xcan))
            l2_to_center.append(np.linalg.norm(x - mu))
            l2_to_canonical.append(np.linalg.norm(x - xcan))

        rows.append({
            "seed_family": fam,
            "block": block,
            "mean_cos_to_family_center": float(np.mean(cos_to_center)),
            "mean_cos_to_canonical": float(np.mean(cos_to_canonical)),
            "center_cos_gain": float(np.mean(cos_to_center) - np.mean(cos_to_canonical)),
            "mean_l2_to_family_center": float(np.mean(l2_to_center)),
            "mean_l2_to_canonical": float(np.mean(l2_to_canonical)),
            "center_l2_gain": float(np.mean(l2_to_canonical) - np.mean(l2_to_center)),
            "center_better_cos_frac": float(np.mean(np.array(cos_to_center) > np.array(cos_to_canonical))),
            "center_better_l2_frac": float(np.mean(np.array(l2_to_center) < np.array(l2_to_canonical))),
        })

    return pd.DataFrame(rows)


# ============================================================
# SUBCLUSTER AUDIT
# ============================================================

def infer_surface_form(prompt: str) -> str:
    p = str(prompt).lower()
    if p.startswith("concept:"):
        return "structured_seed"
    if "first principles" in p:
        return "first_principles"
    if "what if" in p or "imagine" in p or "counterfactual" in p:
        return "counterfactual"
    if p.startswith("what is") or "define" in p:
        return "definition"
    if p.startswith("why") or "cause" in p:
        return "why_causal"
    if p.startswith("how") or "how do" in p or "mechanism" in p or "works" in p:
        return "how_mechanism"
    if p.startswith("list") or p.startswith("name") or "examples" in p:
        return "enumeration"
    if p.startswith("compare") or "contrast" in p or "different" in p:
        return "comparison"
    if "risk" in p or "audit" in p or "danger" in p or "vulnerab" in p:
        return "risk_audit"
    if "plan" in p or "strategy" in p or "steps" in p:
        return "planning"
    if "related" in p or "relation" in p or "domain" in p:
        return "relation"
    if p.startswith("explain") or "describe" in p:
        return "explain_describe"
    return "general"


def subcluster_audit(features: pd.DataFrame, block: str, max_k: int, random_seed: int):
    cols = get_feature_cols(features, block)
    if not cols:
        return pd.DataFrame()

    Xz, _ = standardize_block(features, cols)
    meta = features.reset_index(drop=True)
    rows = []

    for fam in sorted(meta["seed_family"].unique()):
        idx = np.where((meta["seed_family"].values == fam) & (meta["row_type"].values == "original_prompt"))[0]
        if len(idx) < 4:
            continue

        Xfam = Xz[idx, :]
        prompts = meta.iloc[idx]["prompt"].astype(str).tolist()
        surface_forms = np.array([infer_surface_form(p) for p in prompts])

        best = {"k": 1, "silhouette": np.nan, "ari_surface": np.nan}
        for k in range(2, min(max_k, len(idx) - 1) + 1):
            km = KMeans(n_clusters=k, random_state=random_seed, n_init=20)
            labels = km.fit_predict(Xfam)
            try:
                sil = silhouette_score(Xfam, labels)
            except Exception:
                sil = np.nan

            # ARI only meaningful if more than one surface class.
            if len(np.unique(surface_forms)) > 1:
                surface_le = pd.factorize(surface_forms)[0]
                ari = adjusted_rand_score(surface_le, labels)
            else:
                ari = np.nan

            if best["k"] == 1 or (np.nan_to_num(sil, nan=-999) > np.nan_to_num(best["silhouette"], nan=-999)):
                best = {"k": k, "silhouette": sil, "ari_surface": ari}

        rows.append({
            "seed_family": fam,
            "block": block,
            "n_members": int(len(idx)),
            "best_k": int(best["k"]),
            "best_silhouette": float(best["silhouette"]) if np.isfinite(best["silhouette"]) else np.nan,
            "ari_with_surface_form": float(best["ari_surface"]) if np.isfinite(best["ari_surface"]) else np.nan,
            "surface_forms_json": json.dumps(pd.Series(surface_forms).value_counts().to_dict(), ensure_ascii=False),
        })

    return pd.DataFrame(rows)


# ============================================================
# PROMPT* CANDIDATES
# ============================================================

def build_prompt_star_candidates(family_structure: Dict[str, Dict[str, str]], inv_scores: pd.DataFrame, center_vs_can: pd.DataFrame):
    rows = []
    score_map = {}
    if not inv_scores.empty:
        main = inv_scores[inv_scores["block"] == cfg.MAIN_BLOCK] if "block" in inv_scores.columns else inv_scores
        for _, r in main.iterrows():
            score_map.setdefault(r["seed_family"], {}).update({
                "invariant_ratio": r.get("invariant_ratio", np.nan),
                "separation_ratio": r.get("separation_ratio", np.nan),
                "within_l2_mean": r.get("within_l2_mean", np.nan),
            })

    gain_map = {}
    if not center_vs_can.empty:
        main = center_vs_can[center_vs_can["block"] == cfg.MAIN_BLOCK] if "block" in center_vs_can.columns else center_vs_can
        for _, r in main.iterrows():
            gain_map.setdefault(r["seed_family"], {}).update({
                "center_cos_gain": r.get("center_cos_gain", np.nan),
                "center_l2_gain": r.get("center_l2_gain", np.nan),
                "center_better_cos_frac": r.get("center_better_cos_frac", np.nan),
            })

    for fam, s in family_structure.items():
        concept = s.get("concept_star", fam)
        operator = s.get("operator_id", "")
        common = s.get("common_structure", "")
        avoid = s.get("avoid_surface", "")
        prompt_star = s.get("prompt_star", f"Concept: {concept}. Operator: {operator}.")
        structured = (
            f"Concept: {concept}\n"
            f"OperatorID: {operator}\n"
            f"Common structure: {common}\n"
            f"Instruction: generate an answer that preserves the common structure while avoiding family-specific surface wording.\n"
        )
        compressed = f"{concept} + {operator}: {common}"

        for template_type, text in [
            ("natural_prompt_star", prompt_star),
            ("structured_prompt_star", structured),
            ("compressed_prompt_star", compressed),
        ]:
            row = {
                "seed_family": fam,
                "template_type": template_type,
                "prompt_star_candidate": text,
                "concept_star": concept,
                "operator_id": operator,
                "common_structure": common,
                "avoid_surface": avoid,
            }
            row.update(score_map.get(fam, {}))
            row.update(gain_map.get(fam, {}))
            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# SUMMARY
# ============================================================

def summarize(inv_scores, center_vs_can, subclusters, prompt_candidates, sem3a_summary):
    summary = {
        "config": asdict(cfg),
        "sem3a_verdict": sem3a_summary.get("verdict", None) if isinstance(sem3a_summary, dict) else None,
        "sem3a_best": sem3a_summary.get("best", {}) if isinstance(sem3a_summary, dict) else {},
        "main_block": cfg.MAIN_BLOCK,
        "verdict": "UNDETERMINED",
        "key_results": {},
        "interpretation": [],
    }

    main_inv = inv_scores[inv_scores["block"] == cfg.MAIN_BLOCK] if len(inv_scores) else pd.DataFrame()
    main_cvc = center_vs_can[center_vs_can["block"] == cfg.MAIN_BLOCK] if len(center_vs_can) else pd.DataFrame()
    main_sub = subclusters[subclusters["block"] == cfg.MAIN_BLOCK] if len(subclusters) else pd.DataFrame()

    if len(main_inv):
        summary["key_results"]["mean_invariant_ratio"] = float(main_inv["invariant_ratio"].mean())
        summary["key_results"]["mean_separation_ratio"] = float(main_inv["separation_ratio"].mean())
        summary["key_results"]["families_with_separation_gt_1"] = int((main_inv["separation_ratio"] > 1.0).sum())
        summary["key_results"]["n_families"] = int(main_inv["seed_family"].nunique())

    if len(main_cvc):
        summary["key_results"]["mean_center_cos_gain"] = float(main_cvc["center_cos_gain"].mean())
        summary["key_results"]["mean_center_l2_gain"] = float(main_cvc["center_l2_gain"].mean())
        summary["key_results"]["mean_center_better_cos_frac"] = float(main_cvc["center_better_cos_frac"].mean())
        summary["key_results"]["families_center_better_cos_positive"] = int((main_cvc["center_cos_gain"] > 0).sum())

    if len(main_sub):
        summary["key_results"]["mean_subcluster_silhouette"] = float(main_sub["best_silhouette"].mean())
        summary["key_results"]["mean_ari_with_surface_form"] = float(main_sub["ari_with_surface_form"].fillna(0).mean())
        summary["key_results"]["families_with_subclusters"] = int((main_sub["best_k"] > 1).sum())

    # Simple verdict logic.
    sep = summary["key_results"].get("mean_separation_ratio", 0)
    cos_gain = summary["key_results"].get("mean_center_cos_gain", -999)
    better_frac = summary["key_results"].get("mean_center_better_cos_frac", 0)

    if sep > 1.0 and cos_gain > 0 and better_frac > 0.55:
        summary["verdict"] = "PASS_STRONG_COMMON_STRUCTURE_CENTER"
    elif sep > 0.7 and (cos_gain > 0 or better_frac > 0.5):
        summary["verdict"] = "PASS_LITE_COMMON_STRUCTURE_CENTER"
    elif sep > 0.7:
        summary["verdict"] = "PARTIAL_PASS_FAMILY_SEPARATION"
    else:
        summary["verdict"] = "NO_PASS_YET"

    summary["interpretation"] = [
        "Family center μ_F estimates the maximal common trajectory structure of a SeedFamily.",
        "center_vs_canonical compares μ_F against the hand-written canonical prompt; if μ_F is better, Prompt* should come from the family center rather than the canonical sentence.",
        "Subcluster audit is auxiliary; high ARI with surface form means family-internal clusters may be surface-driven.",
        "Prompt* candidates are generated from structural prior plus family-center diagnostics; SEM-3B should forward these candidates.",
    ]

    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    outdir = Path(cfg.OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset, features, proto, sim, sem3a_summary = load_inputs(cfg)
    family_structure = seed_family_structure_map()

    all_centers = []
    all_residuals = []
    all_canonicals = []
    all_inv = []
    all_cvc = []
    all_sub = []

    for block in cfg.BLOCKS:
        print(f"[SEM-3C.0b] Processing block={block}")
        centers, residuals, canonicals = compute_family_centers(features, block)
        if len(centers):
            all_centers.append(centers)
        if len(residuals):
            all_residuals.append(residuals)
        if len(canonicals):
            all_canonicals.append(canonicals)

        inv = invariant_scores(features, centers, block)
        if len(inv):
            all_inv.append(inv)

        cvc = center_vs_canonical(features, centers, canonicals, block)
        if len(cvc):
            all_cvc.append(cvc)

        sub = subcluster_audit(features, block, cfg.MAX_SUBCLUSTERS, cfg.RANDOM_SEED)
        if len(sub):
            all_sub.append(sub)

    centers_df = pd.concat(all_centers, ignore_index=True) if all_centers else pd.DataFrame()
    residuals_df = pd.concat(all_residuals, ignore_index=True) if all_residuals else pd.DataFrame()
    canonicals_df = pd.concat(all_canonicals, ignore_index=True) if all_canonicals else pd.DataFrame()
    inv_df = pd.concat(all_inv, ignore_index=True) if all_inv else pd.DataFrame()
    cvc_df = pd.concat(all_cvc, ignore_index=True) if all_cvc else pd.DataFrame()
    sub_df = pd.concat(all_sub, ignore_index=True) if all_sub else pd.DataFrame()

    prompt_candidates = build_prompt_star_candidates(family_structure, inv_df, cvc_df)
    summary = summarize(inv_df, cvc_df, sub_df, prompt_candidates, sem3a_summary)

    centers_df.to_csv(outdir / "sem3c0b_family_center_features.csv", index=False, encoding="utf-8-sig")
    inv_df.to_csv(outdir / "sem3c0b_family_invariant_scores.csv", index=False, encoding="utf-8-sig")
    residuals_df.to_csv(outdir / "sem3c0b_family_member_residuals.csv", index=False, encoding="utf-8-sig")
    sub_df.to_csv(outdir / "sem3c0b_family_subcluster_audit.csv", index=False, encoding="utf-8-sig")
    cvc_df.to_csv(outdir / "sem3c0b_center_vs_canonical.csv", index=False, encoding="utf-8-sig")
    prompt_candidates.to_csv(outdir / "sem3c0b_prompt_star_candidates.csv", index=False, encoding="utf-8-sig")
    json_dump(asdict(cfg), outdir / "sem3c0b_config.json")
    json_dump(summary, outdir / "sem3c0b_results_summary.json")

    print("=" * 100)
    print("SEM-3C.0b: Maximal Common Structure Extraction")
    print("=" * 100)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nOutput files:")
    for p in [
        "sem3c0b_config.json",
        "sem3c0b_family_center_features.csv",
        "sem3c0b_family_invariant_scores.csv",
        "sem3c0b_family_member_residuals.csv",
        "sem3c0b_family_subcluster_audit.csv",
        "sem3c0b_center_vs_canonical.csv",
        "sem3c0b_prompt_star_candidates.csv",
        "sem3c0b_results_summary.json",
    ]:
        print("  -", outdir / p)


if __name__ == "__main__":
    main()
