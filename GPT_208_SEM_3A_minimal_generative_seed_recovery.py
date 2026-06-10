# -*- coding: utf-8 -*-
"""
GPT_202_SEM_3A_minimal_generative_seed_recovery.py

SEM-3A: Minimal Generative Seed Recovery
----------------------------------------

Core theoretical correction:
    Seed inversion is NOT unique inversion.

Old impossible target:
    Output / Trajectory -> original Prompt / original Seed

New target:
    Output / Trajectory -> SeedFamily / Minimal Generative Seed

We test:
    Multiple different surface prompts / seed expressions can converge to a
    shared minimal generative seed family.

    Original Prompt_i
        -> trajectory signature T_i

    Minimal Seed_j
        -> trajectory signature C_j

    Recovery succeeds if:
        nearest/minimal C_j has same seed_family as T_i,
        and T_i is closer to its recovered minimal seed than random alternatives.

This is a generative-compression test:
    Recoverability(Trajectory), not exact Seed accuracy.

Default:
    Qwen2.5-1.5B-Instruct, hard-coded local path.

Outputs:
    sem3a_outputs/
      sem3a_config.json
      sem3a_dataset.csv
      sem3a_features.csv
      sem3a_center_features.csv
      sem3a_prototype_recovery.csv
      sem3a_similarity_audit.csv
      sem3a_family_cv.csv
      sem3a_results_summary.json

Run:
    python GPT_202_SEM_3A_minimal_generative_seed_recovery.py
"""

import json
import random
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GroupKFold


# ============================================================
# CONFIG
# ============================================================

@dataclass
class CFG:
    MODEL_PATH: str = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"
    OUTPUT_DIR: str = "sem3a_outputs"

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE: str = "auto"

    TOPK_LIST: Tuple[int, ...] = (100, 500)
    LAYERS: Tuple[int, ...] = tuple(range(0, 26))
    BATCH_SIZE: int = 4
    MAX_LENGTH: int = 192
    CENTER_PCA_DIM: int = 16

    N_SPLITS: int = 4
    RANDOM_SEED: int = 42

    REUSE_FEATURES_IF_EXISTS: bool = True


cfg = CFG()


# ============================================================
# DATASET
# ============================================================

def build_sem3a_dataset() -> pd.DataFrame:
    """
    Each family contains:
      - one canonical minimal seed prompt
      - several original prompts that are different but expected to produce
        a similar trajectory / structure.

    The family label is the target; the original prompt text is NOT expected
    to be recoverable.
    """
    families = [
        {
            "seed_family": "inertia_mechanism",
            "concept_star": "Inertia",
            "operator_star": "MechanismExplanation",
            "canonical": "Concept: inertia. Operator: explain the mechanism by which objects maintain their state of motion unless acted on by a net external force.",
            "prompts": [
                "Explain Newton's first law.",
                "From first principles, explain inertia.",
                "Why do objects tend to keep their current state of motion?",
                "Describe why a body remains at rest or in uniform motion unless a force acts on it.",
                "What is the mechanism behind inertial motion?",
                "Explain the idea that motion persists unless disturbed.",
            ],
        },
        {
            "seed_family": "photosynthesis_mechanism",
            "concept_star": "Photosynthesis",
            "operator_star": "MechanismExplanation",
            "canonical": "Concept: photosynthesis. Operator: explain the mechanism by which plants convert light, carbon dioxide, and water into chemical energy.",
            "prompts": [
                "Explain how photosynthesis works.",
                "Describe how plants turn sunlight into energy.",
                "What mechanism lets green plants make sugar from light?",
                "Explain the process by which plants use carbon dioxide and water to make food.",
                "How do chlorophyll and sunlight help plants produce energy?",
                "Give a mechanism-level explanation of photosynthesis.",
            ],
        },
        {
            "seed_family": "democracy_definition",
            "concept_star": "Democracy",
            "operator_star": "Definition",
            "canonical": "Concept: democracy. Operator: define the political system in terms of citizen participation, representation, and collective rule.",
            "prompts": [
                "What is democracy?",
                "Define democratic government.",
                "Explain the meaning of democracy in politics.",
                "Describe democracy in one concise paragraph.",
                "What does it mean for a society to be democratic?",
                "Give a clear definition of a democracy.",
            ],
        },
        {
            "seed_family": "gravity_causal",
            "concept_star": "Gravity",
            "operator_star": "CausalExplanation",
            "canonical": "Concept: gravity. Operator: explain the cause-effect relation by which mass attracts mass and shapes motion.",
            "prompts": [
                "Why do objects fall toward Earth?",
                "Explain the cause of gravitational attraction.",
                "What causes planets to orbit stars?",
                "Why does mass pull other mass?",
                "Give a causal explanation of gravity.",
                "Explain why things with mass attract each other.",
            ],
        },
        {
            "seed_family": "python_use_cases",
            "concept_star": "Python",
            "operator_star": "ListUseCases",
            "canonical": "Concept: Python programming language. Operator: list common practical use cases and application areas.",
            "prompts": [
                "What is Python used for?",
                "List common applications of the Python programming language.",
                "Give examples of tasks people do with Python.",
                "Where is Python commonly used in software development?",
                "Name major use cases for Python.",
                "List practical domains where Python is useful.",
            ],
        },
        {
            "seed_family": "internet_risk",
            "concept_star": "Internet",
            "operator_star": "RiskAudit",
            "canonical": "Concept: internet. Operator: audit major risks, vulnerabilities, and failure modes.",
            "prompts": [
                "What are the risks of the internet?",
                "Audit major dangers associated with internet use.",
                "Identify vulnerabilities created by global internet connectivity.",
                "What can go wrong with online systems?",
                "List key security and social risks of the internet.",
                "Give a risk analysis of the internet.",
            ],
        },
        {
            "seed_family": "climate_plan",
            "concept_star": "ClimateChange",
            "operator_star": "Planning",
            "canonical": "Concept: climate change. Operator: propose a practical multi-step mitigation and adaptation plan.",
            "prompts": [
                "Make a plan to address climate change.",
                "Outline practical steps for climate change mitigation.",
                "Design a strategy to reduce climate risks.",
                "What should governments and communities do about climate change?",
                "Create an action plan for climate adaptation and mitigation.",
                "Give a practical plan for responding to climate change.",
            ],
        },
        {
            "seed_family": "transformer_mechanism",
            "concept_star": "TransformerModel",
            "operator_star": "MechanismExplanation",
            "canonical": "Concept: Transformer model. Operator: explain the mechanism of attention, token representations, and layered computation.",
            "prompts": [
                "Explain how a Transformer model works.",
                "Describe the mechanism of self-attention.",
                "How do Transformers process token sequences?",
                "Explain attention and layers in Transformer architecture.",
                "What mechanism allows Transformers to model context?",
                "Give a mechanism-level explanation of Transformer neural networks.",
            ],
        },
        {
            "seed_family": "market_compare",
            "concept_star": "Market",
            "operator_star": "Comparison",
            "canonical": "Concept: market. Operator: compare market coordination with centralized planning.",
            "prompts": [
                "Compare markets with central planning.",
                "How is a market economy different from a planned economy?",
                "Contrast decentralized market coordination with centralized allocation.",
                "Explain similarities and differences between markets and planning.",
                "Compare price signals with administrative allocation.",
                "How do markets differ from command systems?",
            ],
        },
        {
            "seed_family": "memory_counterfactual",
            "concept_star": "Memory",
            "operator_star": "Counterfactual",
            "canonical": "Concept: memory. Operator: analyze a counterfactual scenario where memory is absent or altered.",
            "prompts": [
                "What if humans had no memory?",
                "Imagine a mind without long-term memory and analyze the consequences.",
                "How would cognition change if memory were unreliable?",
                "Consider a world where people could not store past experience.",
                "What would happen if memory disappeared?",
                "Analyze a counterfactual scenario involving loss of memory.",
            ],
        },
        {
            "seed_family": "ocean_property",
            "concept_star": "Ocean",
            "operator_star": "PropertyDescription",
            "canonical": "Concept: ocean. Operator: describe major properties and characteristic features.",
            "prompts": [
                "What are the main features of oceans?",
                "Describe important properties of the ocean.",
                "What characteristics define Earth's oceans?",
                "List and explain key attributes of oceans.",
                "What is the ocean known for physically and ecologically?",
                "Give a property-focused description of oceans.",
            ],
        },
        {
            "seed_family": "robot_relation",
            "concept_star": "Robot",
            "operator_star": "RelationMapping",
            "canonical": "Concept: robot. Operator: map the concept to its broader technological and functional relations.",
            "prompts": [
                "What is a robot related to?",
                "Place robots in their broader technological context.",
                "How are robots related to automation and machines?",
                "Identify the main domains connected to robotics.",
                "Explain the relation between robots, sensors, control, and automation.",
                "What broader field does the concept of a robot belong to?",
            ],
        },
    ]

    rows = []
    for fam in families:
        # Canonical minimal seed row.
        rows.append({
            "row_id": len(rows),
            "row_type": "canonical_minimal_seed",
            "seed_family": fam["seed_family"],
            "concept_star": fam["concept_star"],
            "operator_star": fam["operator_star"],
            "surface_id": -1,
            "prompt": fam["canonical"],
        })
        # Original prompts.
        for sid, p in enumerate(fam["prompts"]):
            rows.append({
                "row_id": len(rows),
                "row_type": "original_prompt",
                "seed_family": fam["seed_family"],
                "concept_star": fam["concept_star"],
                "operator_star": fam["operator_star"],
                "surface_id": sid,
                "prompt": p,
            })

    return pd.DataFrame(rows)


# ============================================================
# UTILITIES
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=convert)


def get_torch_dtype(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    return "auto"


def safe_mean(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def safe_std(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.std(xs)) if xs else float("nan")


def cosine_np(a, b, eps=1e-9):
    a = np.asarray(a)
    b = np.asarray(b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


# ============================================================
# MODEL
# ============================================================

def load_model_and_tokenizer(cfg: CFG):
    print(f"[SEM-3A] Loading model: {cfg.MODEL_PATH}")
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
# FEATURE EXTRACTION
# ============================================================

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

                key = (k)
                if key in prev_center:
                    row[f"{p}_center_shift_cosdist"] = 1.0 - cosine_np(prev_center[key], center_np)
                    row[f"{p}_center_shift_l2"] = float(np.linalg.norm(center_np - prev_center[key]))
                else:
                    row[f"{p}_center_shift_cosdist"] = 0.0
                    row[f"{p}_center_shift_l2"] = 0.0

                if key in prev_set:
                    jac = jaccard(prev_set[key], id_set)
                    row[f"{p}_jaccard_prev"] = jac
                    row[f"{p}_jaccard_dist_prev"] = 1.0 - jac
                else:
                    row[f"{p}_jaccard_prev"] = 1.0
                    row[f"{p}_jaccard_dist_prev"] = 0.0

                prev_center[key] = center_np
                prev_set[key] = id_set

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
        print(f"[SEM-3A] Extracting {start}:{end}/{len(prompts)}")
        scalar_rows, center_records = extract_batch(prompts[start:end], model, tokenizer, cfg)

        for i, r in enumerate(scalar_rows):
            r["row_id"] = int(df.iloc[start + i]["row_id"])
            all_scalar.append(r)

        for rec in center_records:
            rec["row_id"] = int(df.iloc[start + rec["batch_index"]]["row_id"])
            del rec["batch_index"]
            all_center_records.append(rec)

    scalar_df = pd.DataFrame(all_scalar)

    # Center PCA features per layer/k, fitted over all rows for geometry audit.
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

    feature_df = df.merge(scalar_df, on="row_id", how="left").merge(center_df, on="row_id", how="left")
    return feature_df, center_df


# ============================================================
# FEATURE SELECTION
# ============================================================

def get_feature_cols(df: pd.DataFrame, block: str):
    exclude = {
        "row_id", "row_type", "seed_family", "concept_star",
        "operator_star", "surface_id", "prompt"
    }
    numeric = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    if block == "all":
        return numeric
    if block == "scalar":
        return [c for c in numeric if "centerPC" not in c]
    if block == "center_pca":
        return [c for c in numeric if "centerPC" in c]
    if block == "init":
        return [c for c in numeric if any(f"_L{i}_" in c for i in range(0, 7))]
    if block == "mid":
        return [c for c in numeric if any(f"_L{i}_" in c for i in range(7, 20))]
    if block == "decision":
        return [c for c in numeric if any(f"_L{i}_" in c for i in range(20, 26))]
    if block == "transport":
        keys = ["shift", "jaccard"]
        return [c for c in numeric if any(k in c for k in keys)]
    raise ValueError(block)


def make_X(df, cols):
    return df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


# ============================================================
# PROTOTYPE RECOVERY
# ============================================================

def nearest_canonical_recovery(feature_df: pd.DataFrame, block: str, metric: str):
    cols = get_feature_cols(feature_df, block)
    if not cols:
        return pd.DataFrame()

    canonical = feature_df[feature_df["row_type"] == "canonical_minimal_seed"].copy()
    originals = feature_df[feature_df["row_type"] == "original_prompt"].copy()

    X_all = make_X(pd.concat([canonical, originals], axis=0), cols)
    scaler = StandardScaler()
    scaler.fit(X_all.values)

    Xc = scaler.transform(make_X(canonical, cols).values)
    Xo = scaler.transform(make_X(originals, cols).values)

    fams_c = canonical["seed_family"].astype(str).values
    prompts_c = canonical["prompt"].astype(str).values

    rows = []
    for i, (_, row) in enumerate(originals.iterrows()):
        x = Xo[i]
        if metric == "cosine":
            sims = []
            for c in Xc:
                sims.append(cosine_np(x, c))
            best_idx = int(np.argmax(sims))
            score = float(sims[best_idx])
            # random baseline mean
            random_score_mean = float(np.mean(sims))
        elif metric == "euclidean":
            dists = np.linalg.norm(Xc - x.reshape(1, -1), axis=1)
            best_idx = int(np.argmin(dists))
            score = float(-dists[best_idx])
            random_score_mean = float(-np.mean(dists))
        else:
            raise ValueError(metric)

        recovered_family = fams_c[best_idx]
        true_family = str(row["seed_family"])
        rows.append({
            "row_id": int(row["row_id"]),
            "block": block,
            "metric": metric,
            "true_family": true_family,
            "recovered_family": recovered_family,
            "correct_family": int(recovered_family == true_family),
            "similarity_or_negdist": score,
            "random_baseline_mean": random_score_mean,
            "lift_over_random": score - random_score_mean,
            "original_prompt": row["prompt"],
            "recovered_minimal_seed_prompt": prompts_c[best_idx],
        })

    return pd.DataFrame(rows)


def similarity_audit(feature_df: pd.DataFrame, block: str):
    cols = get_feature_cols(feature_df, block)
    if not cols:
        return pd.DataFrame()

    canonical = feature_df[feature_df["row_type"] == "canonical_minimal_seed"].copy()
    originals = feature_df[feature_df["row_type"] == "original_prompt"].copy()

    X_all = make_X(pd.concat([canonical, originals], axis=0), cols)
    scaler = StandardScaler()
    scaler.fit(X_all.values)

    Xc = scaler.transform(make_X(canonical, cols).values)
    Xo = scaler.transform(make_X(originals, cols).values)

    fam_to_idx = {fam: i for i, fam in enumerate(canonical["seed_family"].astype(str).values)}

    rows = []
    for i, (_, row) in enumerate(originals.iterrows()):
        true_fam = str(row["seed_family"])
        true_idx = fam_to_idx[true_fam]
        x = Xo[i]
        true_sim = cosine_np(x, Xc[true_idx])
        all_sims = np.array([cosine_np(x, c) for c in Xc])
        rand_mean = float(np.mean(np.delete(all_sims, true_idx)))
        rand_max = float(np.max(np.delete(all_sims, true_idx)))
        margin_vs_random_mean = float(true_sim - rand_mean)
        margin_vs_random_max = float(true_sim - rand_max)

        rows.append({
            "row_id": int(row["row_id"]),
            "block": block,
            "seed_family": true_fam,
            "true_canonical_cosine": float(true_sim),
            "random_canonical_mean_cosine": rand_mean,
            "random_canonical_max_cosine": rand_max,
            "margin_vs_random_mean": margin_vs_random_mean,
            "margin_vs_random_max": margin_vs_random_max,
        })

    return pd.DataFrame(rows)


# ============================================================
# CV FAMILY CLASSIFICATION
# ============================================================

def family_cv(feature_df: pd.DataFrame, block: str):
    """
    A secondary diagnostic:
      Can trajectory features classify SeedFamily across held-out surface_id?

    This is NOT exact seed inversion. It is family recovery.
    """
    df = feature_df[feature_df["row_type"] == "original_prompt"].copy()
    cols = get_feature_cols(df, block)
    if not cols:
        return []

    X = make_X(df, cols).values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df["seed_family"].astype(str).values)

    results = []
    for model_name, clf in [
        ("ridge", Pipeline([("scaler", StandardScaler()), ("clf", RidgeClassifier(class_weight="balanced"))])),
        ("logreg", Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, solver="lbfgs", class_weight="balanced"))])),
    ]:
        for cv_name, splitter in [
            ("stratified", StratifiedKFold(n_splits=cfg.N_SPLITS, shuffle=True, random_state=cfg.RANDOM_SEED)),
            ("leave_surface_out", GroupKFold(n_splits=len(np.unique(df["surface_id"].values)))),
        ]:
            y_true, y_pred = [], []
            groups = df["surface_id"].values
            splits = splitter.split(X, y, groups) if cv_name == "leave_surface_out" else splitter.split(X, y)

            for tr, te in splits:
                clf.fit(X[tr], y[tr])
                pred = clf.predict(X[te])
                y_true.extend(y[te].tolist())
                y_pred.extend(pred.tolist())

            results.append({
                "block": block,
                "model": model_name,
                "cv": cv_name,
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            })

    return results


# ============================================================
# SUMMARY
# ============================================================

def summarize(proto_df, sim_df, cv_df):
    summary = {
        "config": asdict(cfg),
        "prototype_recovery": {},
        "similarity_audit": {},
        "family_cv": {},
        "verdict": "UNDETERMINED",
        "interpretation": [],
    }

    if len(proto_df):
        for block in sorted(proto_df["block"].unique()):
            sub = proto_df[proto_df["block"] == block]
            summary["prototype_recovery"][block] = {
                "best_cosine_acc": float(sub[sub["metric"] == "cosine"]["correct_family"].mean()) if len(sub[sub["metric"] == "cosine"]) else None,
                "best_euclidean_acc": float(sub[sub["metric"] == "euclidean"]["correct_family"].mean()) if len(sub[sub["metric"] == "euclidean"]) else None,
                "mean_lift_cosine": float(sub[sub["metric"] == "cosine"]["lift_over_random"].mean()) if len(sub[sub["metric"] == "cosine"]) else None,
                "mean_lift_euclidean": float(sub[sub["metric"] == "euclidean"]["lift_over_random"].mean()) if len(sub[sub["metric"] == "euclidean"]) else None,
            }

    if len(sim_df):
        for block in sorted(sim_df["block"].unique()):
            sub = sim_df[sim_df["block"] == block]
            summary["similarity_audit"][block] = {
                "mean_true_canonical_cosine": float(sub["true_canonical_cosine"].mean()),
                "mean_random_canonical_cosine": float(sub["random_canonical_mean_cosine"].mean()),
                "mean_margin_vs_random_mean": float(sub["margin_vs_random_mean"].mean()),
                "frac_true_gt_random_mean": float((sub["margin_vs_random_mean"] > 0).mean()),
                "frac_true_gt_random_max": float((sub["margin_vs_random_max"] > 0).mean()),
            }

    if len(cv_df):
        for block in sorted(cv_df["block"].unique()):
            sub = cv_df[cv_df["block"] == block]
            best = sub.sort_values(["accuracy", "macro_f1"], ascending=False).iloc[0].to_dict()
            summary["family_cv"][block] = best

    # Main verdict logic.
    best_proto_acc = 0.0
    best_proto_block = None
    for block, vals in summary["prototype_recovery"].items():
        for key in ["best_cosine_acc", "best_euclidean_acc"]:
            val = vals.get(key)
            if val is not None and val > best_proto_acc:
                best_proto_acc = val
                best_proto_block = block

    best_sim_margin = -999
    best_sim_block = None
    for block, vals in summary["similarity_audit"].items():
        val = vals.get("mean_margin_vs_random_mean", -999)
        if val > best_sim_margin:
            best_sim_margin = val
            best_sim_block = block

    best_cv_acc = 0.0
    for block, vals in summary["family_cv"].items():
        acc = vals.get("accuracy", 0.0)
        if acc > best_cv_acc:
            best_cv_acc = acc

    summary["best"] = {
        "best_proto_acc": best_proto_acc,
        "best_proto_block": best_proto_block,
        "best_sim_margin": best_sim_margin,
        "best_sim_block": best_sim_block,
        "best_cv_acc": best_cv_acc,
    }

    if best_proto_acc >= 0.75 and best_sim_margin > 0 and best_cv_acc >= 0.85:
        summary["verdict"] = "PASS_STRONG_MINIMAL_GENERATIVE_SEED_RECOVERY"
    elif best_proto_acc >= 0.50 and best_sim_margin > 0:
        summary["verdict"] = "PASS_LITE_SEED_FAMILY_RECOVERY"
    elif best_sim_margin > 0:
        summary["verdict"] = "PARTIAL_PASS_GENERATIVE_EQUIVALENCE_SIGNAL"
    else:
        summary["verdict"] = "NO_PASS_YET"

    summary["interpretation"].append(
        "Prototype recovery tests whether original prompts are closer to their canonical minimal seed than to other canonical seeds."
    )
    summary["interpretation"].append(
        "Family CV is secondary; exact original prompt recovery is not the goal."
    )
    summary["interpretation"].append(
        "A positive result supports SeedFamily / Minimal Generative Seed recovery rather than unique prompt inversion."
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    warnings.filterwarnings("ignore")
    set_seed(cfg.RANDOM_SEED)

    outdir = Path(cfg.OUTPUT_DIR)
    ensure_dir(str(outdir))
    json_dump(asdict(cfg), outdir / "sem3a_config.json")

    print("=" * 100)
    print("SEM-3A: Minimal Generative Seed Recovery")
    print("=" * 100)

    df = build_sem3a_dataset()
    df.to_csv(outdir / "sem3a_dataset.csv", index=False, encoding="utf-8-sig")
    print(f"[SEM-3A] Dataset rows={len(df)} families={df['seed_family'].nunique()}")

    features_path = outdir / "sem3a_features.csv"
    center_path = outdir / "sem3a_center_features.csv"

    if cfg.REUSE_FEATURES_IF_EXISTS and features_path.exists():
        print(f"[SEM-3A] Reusing features: {features_path}")
        feature_df = pd.read_csv(features_path)
    else:
        model, tokenizer = load_model_and_tokenizer(cfg)
        feature_df, center_df = extract_all_features(df, model, tokenizer, cfg)
        feature_df.to_csv(features_path, index=False, encoding="utf-8-sig")
        center_df.to_csv(center_path, index=False, encoding="utf-8-sig")
        print(f"[SEM-3A] Features saved: {features_path}")

    blocks = ["scalar", "center_pca", "transport", "init", "mid", "decision", "all"]

    proto_rows = []
    sim_rows = []
    cv_rows = []

    for block in blocks:
        for metric in ["cosine", "euclidean"]:
            pr = nearest_canonical_recovery(feature_df, block, metric)
            if len(pr):
                proto_rows.append(pr)

        sr = similarity_audit(feature_df, block)
        if len(sr):
            sim_rows.append(sr)

        cv_rows.extend(family_cv(feature_df, block))

    proto_df = pd.concat(proto_rows, ignore_index=True) if proto_rows else pd.DataFrame()
    sim_df = pd.concat(sim_rows, ignore_index=True) if sim_rows else pd.DataFrame()
    cv_df = pd.DataFrame(cv_rows)

    proto_df.to_csv(outdir / "sem3a_prototype_recovery.csv", index=False, encoding="utf-8-sig")
    sim_df.to_csv(outdir / "sem3a_similarity_audit.csv", index=False, encoding="utf-8-sig")
    cv_df.to_csv(outdir / "sem3a_family_cv.csv", index=False, encoding="utf-8-sig")

    summary = summarize(proto_df, sim_df, cv_df)
    json_dump(summary, outdir / "sem3a_results_summary.json")

    print("\n" + "=" * 100)
    print("[SEM-3A] SUMMARY")
    print("=" * 100)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n[SEM-3A] Output files:")
    for p in [
        "sem3a_config.json",
        "sem3a_dataset.csv",
        "sem3a_features.csv",
        "sem3a_center_features.csv",
        "sem3a_prototype_recovery.csv",
        "sem3a_similarity_audit.csv",
        "sem3a_family_cv.csv",
        "sem3a_results_summary.json",
    ]:
        print("  -", outdir / p)


if __name__ == "__main__":
    main()
