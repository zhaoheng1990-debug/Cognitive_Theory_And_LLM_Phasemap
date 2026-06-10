# -*- coding: utf-8 -*-
r"""
SOB-V5D: Use / Operator-Routing Intervention Causal Audit

Goal
----
Test the third causal factor in the accepted SOB meaning formula:

    Meaning(S,O,T) = V(S,O) * R(O,T) * U(S,O,T)

V5A/V5B intervened on V. V5C intervenes on R. V5D intervenes on U.

Core question
-------------
If object O, subject state S, and task relevance R are held fixed, does an
explicit use-routing intervention change Future Cbit?

Conditions
----------
control:
    no explicit routing.

use_route:
    explicitly route through the studied object when relevant.

boundary_route:
    first decide whether the object is relevant; use it only if relevant.

wrong_route:
    force route through a wrong object. Negative control.

Outputs
-------
C:\Users\ZH\Desktop\AGI\outputs\SOB-V5D

This script is intentionally compact: it focuses on generation-level causal
contrasts and scoring rather than heavy hidden-state extraction. It is meant as
V5D first-pass causal validation.
"""

import os
import re
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

EXPERIMENT_ID = "SOB-V5D"
OUT_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\SOB-V5D")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SPECS = {
    "qwen": {
        "model_name": "Qwen2.5-1.5B-Instruct",
        "path": r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main",
        "trust_remote_code": True,
    },
    "llama": {
        "model_name": "Llama-3.2-1B-Instruct",
        "path": r"D:\model\Llama-3.2-1B-Instruct",
        "trust_remote_code": True,
    },
    "gemma": {
        "model_name": "gemma-2-2b-it",
        "path": r"D:\model\gemma-2-2b-it",
        "trust_remote_code": True,
    },
}

MODELS_TO_RUN = ["qwen", "llama", "gemma"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
MAX_INPUT_LEN = 1700
MAX_NEW_TOKENS = 220
RANDOM_SEED = 42

RELEVANCE_WEIGHT = {"high": 1.0, "medium": 0.55, "low": 0.10}
LOW_STATES = {"S0", "S1", "S2"}
CONDITIONS = ["control", "use_route", "boundary_route", "wrong_route"]

SUBJECT_STATES = [
    ("S0", 0, "symbol_only", "You have only seen the name or symbol. You do not know definition, calculation, geometry, edge cases, or applications."),
    ("S1", 1, "definition_known", "You know only the formal definition and a few basic words. You cannot reliably calculate, reason geometrically, or transfer."),
    ("S2", 2, "calculation_ability", "You know definition and standard procedures. Geometry, edge cases, and transfer are limited."),
    ("S3", 3, "geometric_understanding", "You know definition, procedures, geometry, and edge cases. Cross-domain transfer is not yet expert-level."),
    ("S4", 4, "cross_domain_transfer", "You understand the object as a transferable structural tool across domains."),
]

OBJECTS = {
    "derivative": {
        "name": "derivative",
        "wrong_object": "integral",
        "terms": {
            "symbol": ["d/dx", "f'", "prime", "derivative", "differentiation"],
            "core": ["limit", "instantaneous", "rate of change", "tangent", "slope", "local linear", "differentiable"],
            "use": ["local change", "instantaneous change", "slope", "rate", "sensitivity", "marginal", "gradient"],
        },
        "visibility": "Derivative makes local change visible: instantaneous rate, tangent slope, local linear approximation, sensitivity, and marginal change.",
        "routing": "Use derivative by asking what changes locally, with respect to which variable, and what the local rate or slope tells us.",
        "triads": [
            ("change_rate", "A vehicle position changes over time. Explain instantaneous velocity.", "A product metric changes when one design parameter is adjusted. Explain when local sensitivity helps.", "Explain the historical causes of a revolution."),
            ("optimization", "A loss function changes with a parameter. Explain why derivative information can guide an update.", "A business studies marginal cost as output changes. Explain when derivative-like reasoning helps.", "Analyze the theme of a poem."),
        ],
    },
    "integral": {
        "name": "integral",
        "wrong_object": "gradient",
        "terms": {
            "symbol": ["integral", "∫", "dx", "antiderivative"],
            "core": ["Riemann sum", "area", "accumulation", "partition", "antiderivative", "convergence"],
            "use": ["accumulation", "total amount", "area under", "sum over", "expected value", "aggregate"],
        },
        "visibility": "Integral makes accumulation visible: continuous summation, total amount, area under a curve, expected value, and aggregate effect.",
        "routing": "Use integral by asking what local quantity is being accumulated, over what domain, and what total the accumulation produces.",
        "triads": [
            ("accumulation", "Velocity is known over time. Explain total distance.", "A project has changing daily costs. Explain when accumulation helps estimate total cost.", "Analyze a legal argument about responsibility."),
            ("probability", "A probability density is given. Explain total probability.", "A researcher combines many small effects over time. Explain whether accumulation helps.", "Describe the visual style of a painting."),
        ],
    },
    "gradient": {
        "name": "gradient",
        "wrong_object": "eigenvector",
        "terms": {
            "symbol": ["gradient", "∇", "nabla", "partial derivative"],
            "core": ["partial derivative", "directional derivative", "steepest ascent", "level set", "critical point"],
            "use": ["steepest", "direction of change", "optimization", "loss landscape", "update", "sensitivity"],
        },
        "visibility": "Gradient makes direction-of-change visible: steepest increase/decrease, level set normal, optimization direction, and variable sensitivity.",
        "routing": "Use gradient by asking what scalar quantity changes, what variables can move, and which direction changes it fastest.",
        "triads": [
            ("optimization", "A model must minimize a loss. Explain how gradient indicates an update direction.", "A policy has adjustable variables. Explain when gradient-like directional pressure helps.", "Explain how to make a soup."),
            ("landscape", "You are on a height landscape. Explain steepest ascent or descent.", "A team allocates resources among levers. Explain when directional sensitivity helps.", "Analyze a fictional character's motivation."),
        ],
    },
    "eigenvector": {
        "name": "eigenvector",
        "wrong_object": "manifold",
        "terms": {
            "symbol": ["eigenvector", "eigenvalue", "lambda", "Av"],
            "core": ["invariant direction", "linear transformation", "scale", "matrix", "characteristic"],
            "use": ["invariant direction", "stable direction", "principal component", "mode", "spectral", "unchanged direction"],
        },
        "visibility": "Eigenvector makes invariant direction visible: a transformation changes magnitude but preserves direction, revealing stable modes or principal axes.",
        "routing": "Use eigenvector by asking what transformation repeats, which directions remain invariant, and what stable or principal mode is revealed.",
        "triads": [
            ("stable_mode", "A linear dynamical system repeats a transformation. Explain stable modes.", "An organization changes but some strategic direction remains stable. Explain whether eigenvector is a useful analogy.", "Explain how to roast vegetables."),
            ("principal_axis", "Data vary in many directions. Explain principal components.", "A network has repeated influence propagation. Explain when eigenvector centrality helps.", "Analyze rhyme in a poem."),
        ],
    },
    "manifold": {
        "name": "manifold",
        "wrong_object": "derivative",
        "terms": {
            "symbol": ["manifold", "chart", "atlas", "surface"],
            "core": ["locally Euclidean", "coordinate", "tangent space", "curvature", "geodesic", "local patch"],
            "use": ["local patch", "global structure", "latent space", "state space", "configuration space", "local coordinates"],
        },
        "visibility": "Manifold makes local-global structure visible: simple local coordinate patches connected into a global space with curvature and constraints.",
        "routing": "Use manifold by asking what is locally simple, what global structure connects local patches, and what constraints shape movement.",
        "triads": [
            ("latent_structure", "Data lie on a lower-dimensional latent structure. Explain this.", "A scientific field has local methods and global structure. Explain whether manifold is a useful analogy.", "Explain why 2+2=4."),
            ("configuration", "A robot arm has constrained configurations. Explain configuration space.", "An organization has local teams and global coordination. Explain whether manifold-like structure helps.", "Review a comedy movie."),
        ],
    },
}


def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def now_time():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def normalize_text(s):
    return str(s).lower().replace("’", "'").replace("“", '"').replace("”", '"')


def count_words(s):
    return len(re.findall(r"\b[\w'-]+\b", str(s)))


def lexical_diversity(text):
    words = re.findall(r"\b[a-zA-Z][a-zA-Z'-]*\b", normalize_text(text))
    if not words:
        return 0.0
    return float(len(set(words)) / max(1, len(words)))


def marker_hits(text, markers):
    t = normalize_text(text)
    hits = []
    for m in markers:
        if normalize_text(m) in t:
            hits.append(m)
    return sorted(set(hits))


def marker_score(text, markers):
    if not markers:
        return 0.0, []
    hits = marker_hits(text, markers)
    return float(len(hits) / max(1, len(markers))), hits


def compression_quality(text):
    n = count_words(text)
    if n <= 0:
        return 0.0
    if 25 <= n <= 120:
        return 1.0
    if n < 25:
        return max(0.0, n / 25.0)
    return max(0.0, 1.0 - (n - 120) / 260.0)


def condition_instruction(obj, condition):
    if condition == "control":
        return ""
    if condition == "use_route":
        return f"Use-routing instruction: If the studied object is relevant, route the answer through {obj['name']}. {obj['routing']}"
    if condition == "boundary_route":
        return f"Boundary-aware routing instruction: First decide whether {obj['name']} is relevant. If relevant, use it through this route: {obj['routing']} If not relevant, explicitly say it is not the right tool and answer without forcing it."
    if condition == "wrong_route":
        return f"Negative-control routing instruction: Force the answer through {obj['wrong_object']} even if {obj['name']} was the studied object."
    return ""


def build_future_prompt(obj, st_profile, condition, relevance, triad_id, task):
    instr = condition_instruction(obj, condition)
    return (
        "You are participating in a use-routing causal audit.\n\n"
        "Role-play the assigned learner state. Use the studied object only if it is genuinely relevant, unless the condition explicitly says otherwise.\n\n"
        f"Learner state:\n{st_profile}\n\n"
        f"Studied object: {obj['name']}.\n"
        f"Visibility grounding:\n{obj['visibility']}\n"
        f"Condition: {condition}\n"
        f"Relevance: {relevance}\n"
        f"Task triad: {triad_id}\n"
        f"Future task:\n{task}\n\n"
        f"{instr}\n\n"
        "Answer concisely. Focus on whether the studied object helps compress the task."
    )


def build_dataset():
    rows = []
    for obj_key, obj in OBJECTS.items():
        for state_id, state_rank, state_name, profile in SUBJECT_STATES:
            for triad_id, high_task, med_task, low_task in obj["triads"]:
                rel_tasks = {"high": high_task, "medium": med_task, "low": low_task}
                for relevance, task in rel_tasks.items():
                    for condition in CONDITIONS:
                        rows.append({
                            "row_id": len(rows),
                            "object_key": obj_key,
                            "object_name": obj["name"],
                            "wrong_object": obj["wrong_object"],
                            "state_id": state_id,
                            "state_rank": state_rank,
                            "state_name": state_name,
                            "low_state": int(state_id in LOW_STATES),
                            "triad_id": triad_id,
                            "relevance": relevance,
                            "relevance_weight": RELEVANCE_WEIGHT[relevance],
                            "condition": condition,
                            "future_task": task,
                            "prompt": build_future_prompt(obj, profile, condition, relevance, triad_id, task),
                        })
    return pd.DataFrame(rows)


def load_model_and_tokenizer(path, trust_remote_code=True):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=trust_remote_code, local_files_only=True, torch_dtype=DTYPE, device_map=None)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def format_chat(tokenizer, prompt):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        except Exception:
            return prompt
    return prompt


@torch.no_grad()
def generate_answer(model, tokenizer, prompt):
    text = format_chat(tokenizer, prompt)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LEN)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    new_tokens = out[0, enc["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def score_future_answer(row):
    text = str(row.get("answer_text", ""))
    obj = OBJECTS[row["object_key"]]
    relevance = row["relevance"]

    core_terms = obj["terms"]["core"]
    use_terms = obj["terms"]["use"]
    object_terms = obj["terms"]["symbol"] + core_terms + use_terms
    wrong_terms = [obj["wrong_object"]]
    general_structural = ["because", "therefore", "structure", "constraint", "mapping", "principle", "helps", "not useful", "not relevant", "only if"]

    object_s, object_hits = marker_score(text, object_terms)
    core_s, core_hits = marker_score(text, core_terms)
    use_s, use_hits = marker_score(text, use_terms)
    wrong_s, wrong_hits = marker_score(text, wrong_terms)
    general_s, general_hits = marker_score(text, general_structural)

    n = count_words(text)
    div = lexical_diversity(text)
    comp = compression_quality(text)
    negation_hits = marker_hits(text, ["not useful", "not relevant", "do not force", "does not help", "not the right tool", "limited relevance"])

    object_use_score = max(0.0, min(1.0, 0.55 * use_s + 0.25 * core_s + 0.10 * general_s + 0.10 * div))
    wrong_route_score = wrong_s

    if relevance == "low":
        correct_nonuse_score = min(1.0, len(negation_hits) / 2.0) if negation_hits else max(0.0, 1.0 - object_use_score)
        useful_use_cbit = 0.0
        overuse_penalty = object_use_score * (0.85 if not negation_hits else 0.30) + 0.35 * wrong_route_score
        corrected_use = 0.15 + 0.75 * correct_nonuse_score
        relevance_alignment = correct_nonuse_score
    elif relevance == "medium":
        correct_nonuse_score = 0.0
        useful_use_cbit = 0.50 * object_use_score + 0.25 * general_s + 0.15 * comp + 0.10 * div
        overuse_penalty = 0.10 * max(0.0, object_use_score - 0.70) + 0.50 * wrong_route_score
        corrected_use = useful_use_cbit - overuse_penalty
        relevance_alignment = 0.45 + 0.55 * object_use_score
    else:
        correct_nonuse_score = 0.0
        useful_use_cbit = 0.58 * object_use_score + 0.18 * general_s + 0.14 * comp + 0.10 * div
        overuse_penalty = 0.65 * wrong_route_score
        corrected_use = useful_use_cbit - overuse_penalty
        relevance_alignment = object_use_score

    useful_use_cbit = max(0.0, min(1.0, useful_use_cbit))
    overuse_penalty = max(0.0, min(1.0, overuse_penalty))
    corrected_use = max(0.0, min(1.0, corrected_use))
    relevance_alignment = max(0.0, min(1.0, relevance_alignment))

    if relevance == "low":
        cbit_future = 0.55 * correct_nonuse_score + 0.20 * general_s + 0.15 * comp + 0.10 * div - 0.50 * overuse_penalty
    else:
        cbit_future = 0.52 * useful_use_cbit + 0.20 * relevance_alignment + 0.13 * general_s + 0.10 * comp + 0.05 * div - 0.25 * overuse_penalty

    tq_future = 0.38 * corrected_use + 0.22 * object_use_score + 0.20 * relevance_alignment + 0.10 * general_s + 0.10 * comp - 0.25 * overuse_penalty
    aq_future = 0.38 * corrected_use + 0.22 * relevance_alignment + 0.18 * general_s + 0.12 * comp + 0.10 * div - 0.25 * overuse_penalty

    cbit_future = max(0.0, min(1.0, cbit_future))
    tq_future = max(0.0, min(1.0, tq_future))
    aq_future = max(0.0, min(1.0, aq_future))

    return {
        "future_word_count": n,
        "object_marker_score": object_s,
        "core_marker_score": core_s,
        "use_marker_score": use_s,
        "wrong_route_score": wrong_route_score,
        "object_hits": "; ".join(object_hits),
        "wrong_hits": "; ".join(wrong_hits),
        "negation_hits": "; ".join(negation_hits),
        "object_use_score": object_use_score,
        "corrected_use_score": corrected_use,
        "useful_use_Cbit": useful_use_cbit,
        "correct_nonuse_Cbit": correct_nonuse_score,
        "overuse_penalty": overuse_penalty,
        "Cbit_future_proxy": cbit_future,
        "TQ_future_proxy": tq_future,
        "AQ_future_proxy": aq_future,
        "future_success": int(cbit_future >= 0.48 and relevance_alignment >= 0.45),
    }


def score_future_df(df):
    rows = []
    for _, row in df.iterrows():
        d = row.to_dict()
        d.update(score_future_answer(d))
        rows.append(d)
    return pd.DataFrame(rows)


def paired_deltas(df):
    keys = ["model_key", "object_key", "state_id", "triad_id", "relevance"]
    metrics = ["Cbit_future_proxy", "TQ_future_proxy", "AQ_future_proxy", "useful_use_Cbit", "correct_nonuse_Cbit", "overuse_penalty", "object_use_score", "corrected_use_score", "wrong_route_score", "future_success", "low_state", "state_rank", "relevance_weight"]
    ctrl = df[df["condition"] == "control"].copy()
    ctrl_small = ctrl[keys + metrics].rename(columns={c: f"{c}_control" for c in metrics})
    pairs = []
    for cond in ["use_route", "boundary_route", "wrong_route"]:
        sub = df[df["condition"] == cond].copy()
        sub_small = sub[keys + metrics].rename(columns={c: f"{c}_intervention" for c in metrics})
        pair = ctrl_small.merge(sub_small, on=keys, how="inner")
        pair["condition"] = cond
        for m in metrics:
            pair[f"delta_{m}"] = pair[f"{m}_intervention"] - pair[f"{m}_control"]
        pair["low_state"] = pair["low_state_control"]
        pair["state_rank"] = pair["state_rank_control"]
        pair["relevance_weight"] = pair["relevance_weight_control"]
        pairs.append(pair)
    return pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()


def summarize_pair(pair):
    rows = []
    groups = []
    for cond in ["use_route", "boundary_route", "wrong_route"]:
        groups.append((cond, pair[pair["condition"] == cond]))
        groups.append((f"{cond}_low_states", pair[(pair["condition"] == cond) & (pair["low_state"] == 1)]))
        for rel in ["high", "medium", "low"]:
            groups.append((f"{cond}_low_states_{rel}", pair[(pair["condition"] == cond) & (pair["low_state"] == 1) & (pair["relevance"] == rel)]))
    metrics = ["delta_Cbit_future_proxy", "delta_useful_use_Cbit", "delta_correct_nonuse_Cbit", "delta_overuse_penalty", "delta_object_use_score", "delta_corrected_use_score", "delta_wrong_route_score", "delta_future_success"]
    for name, sub in groups:
        row = {"group": name, "n": int(len(sub))}
        for m in metrics:
            vals = pd.to_numeric(sub[m], errors="coerce").dropna() if m in sub.columns else pd.Series(dtype=float)
            row[f"{m}_mean"] = safe_float(vals.mean()) if len(vals) else np.nan
            row[f"{m}_positive_rate"] = safe_float((vals > 0).mean()) if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def verdict_from_summary(summary):
    def g(group, metric):
        row = summary[summary["group"] == group]
        if len(row) == 0:
            return np.nan
        col = f"{metric}_mean"
        return safe_float(row[col].iloc[0]) if col in row.columns else np.nan

    use_high = g("use_route_low_states_high", "delta_useful_use_Cbit")
    use_med = g("use_route_low_states_medium", "delta_useful_use_Cbit")
    use_low_over = g("use_route_low_states_low", "delta_overuse_penalty")

    bound_high = g("boundary_route_low_states_high", "delta_useful_use_Cbit")
    bound_med = g("boundary_route_low_states_medium", "delta_useful_use_Cbit")
    bound_low_over = g("boundary_route_low_states_low", "delta_overuse_penalty")
    bound_low_nonuse = g("boundary_route_low_states_low", "delta_correct_nonuse_Cbit")
    bound_cbit_high = g("boundary_route_low_states_high", "delta_Cbit_future_proxy")
    bound_cbit_med = g("boundary_route_low_states_medium", "delta_Cbit_future_proxy")
    bound_cbit_low = g("boundary_route_low_states_low", "delta_Cbit_future_proxy")

    wrong_high = g("wrong_route_low_states_high", "delta_Cbit_future_proxy")
    wrong_over_low = g("wrong_route_low_states_low", "delta_overuse_penalty")

    use_route_support = bool((np.isfinite(use_high) and use_high > 0.02) or (np.isfinite(use_med) and use_med > 0.02))
    boundary_use_support = bool((np.isfinite(bound_high) and bound_high > 0.02) or (np.isfinite(bound_med) and bound_med > 0.02))
    boundary_cbit_support = bool((np.isfinite(bound_cbit_high) and bound_cbit_high > 0.02) or (np.isfinite(bound_cbit_med) and bound_cbit_med > 0.02))
    boundary_safety_support = bool((not np.isfinite(bound_low_over) or bound_low_over < 0.02) and (not np.isfinite(bound_low_nonuse) or bound_low_nonuse >= -0.02))
    boundary_dominates_use_safety = bool(np.isfinite(bound_low_over) and np.isfinite(use_low_over) and bound_low_over <= use_low_over + 0.005)
    wrong_negative_control = bool((np.isfinite(wrong_high) and wrong_high < 0.01) or (np.isfinite(wrong_over_low) and wrong_over_low > 0.02))

    if boundary_use_support and boundary_cbit_support and boundary_safety_support and wrong_negative_control:
        verdict = "PASS_STRONG_USE_ROUTING_CAUSAL"
    elif (use_route_support or boundary_use_support) and boundary_safety_support:
        verdict = "PASS_USE_ROUTING_CAUSAL"
    elif use_route_support or boundary_use_support or boundary_cbit_support:
        verdict = "PARTIAL_USE_ROUTING_CAUSAL"
    else:
        verdict = "FAIL_USE_ROUTING_CAUSAL"

    return {
        "verdict": verdict,
        "use_route_high_useful_use_delta": use_high,
        "use_route_medium_useful_use_delta": use_med,
        "use_route_low_overuse_delta": use_low_over,
        "boundary_high_useful_use_delta": bound_high,
        "boundary_medium_useful_use_delta": bound_med,
        "boundary_low_overuse_delta": bound_low_over,
        "boundary_low_correct_nonuse_delta": bound_low_nonuse,
        "boundary_high_cbit_delta": bound_cbit_high,
        "boundary_medium_cbit_delta": bound_cbit_med,
        "boundary_low_cbit_delta": bound_cbit_low,
        "wrong_route_high_cbit_delta": wrong_high,
        "wrong_route_low_overuse_delta": wrong_over_low,
        "use_route_support": use_route_support,
        "boundary_use_support": boundary_use_support,
        "boundary_cbit_support": boundary_cbit_support,
        "boundary_safety_support": boundary_safety_support,
        "boundary_dominates_use_safety": boundary_dominates_use_safety,
        "wrong_negative_control": wrong_negative_control,
    }


def run_one_model(model_key, dataset):
    spec = MODEL_SPECS[model_key]
    model_out = OUT_DIR / model_key
    model_out.mkdir(parents=True, exist_ok=True)
    info = {"experiment_id": EXPERIMENT_ID, "model_key": model_key, "model_name": spec["model_name"], "path": spec["path"], "exists": os.path.exists(spec["path"]), "status": "pending", "started_at": now_time()}
    if not os.path.exists(spec["path"]):
        info.update({"status": "missing_path", "error": f"Path not found: {spec['path']}"})
        return info
    model = None
    try:
        t0 = time.time()
        model, tokenizer = load_model_and_tokenizer(spec["path"], spec.get("trust_remote_code", True))
        rows, errors = [], []
        for i, row in dataset.iterrows():
            if i % 100 == 0:
                print(f"[{model_key}] future row {i}/{len(dataset)}", flush=True)
            try:
                ans = generate_answer(model, tokenizer, row["prompt"])
                rows.append({**row.to_dict(), "model_key": model_key, "model_name": spec["model_name"], "answer_text": ans})
            except Exception as e:
                errors.append({"row_id": int(row["row_id"]), "error_type": type(e).__name__, "error": str(e), "traceback": traceback.format_exc()})
        answers = pd.DataFrame(rows)
        scored = score_future_df(answers) if not answers.empty else pd.DataFrame()
        pairs = paired_deltas(scored) if not scored.empty else pd.DataFrame()
        summary = summarize_pair(pairs) if not pairs.empty else pd.DataFrame()
        verdict = verdict_from_summary(summary) if not summary.empty else {"verdict": "EMPTY"}
        answers.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_answers.csv", index=False, encoding="utf-8-sig")
        scored.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_scored_answers.csv", index=False, encoding="utf-8-sig")
        pairs.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_paired_deltas.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_effect_summary.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(errors).to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_errors.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([{**{"model_key": model_key}, **verdict}]).to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_verdict_summary.csv", index=False, encoding="utf-8-sig")
        info.update({"status": "done", "finished_at": now_time(), "runtime_sec": time.time() - t0, "n_future_rows": int(len(scored)), "n_pairs": int(len(pairs)), "n_errors": int(len(errors)), "verdict": verdict.get("verdict"), "metrics": verdict})
        with open(model_out / f"{EXPERIMENT_ID}_{model_key}_verdict.json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
    except Exception as e:
        info.update({"status": "failed", "finished_at": now_time(), "error_type": type(e).__name__, "error": str(e), "traceback": traceback.format_exc()})
    finally:
        try:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return info


def cross_model_summary(infos):
    rows = []
    for info in infos:
        m = info.get("metrics", {}) or {}
        rows.append({
            "model_key": info.get("model_key"),
            "model_name": info.get("model_name"),
            "status": info.get("status"),
            "verdict": info.get("verdict"),
            "use_route_high_useful_use_delta": m.get("use_route_high_useful_use_delta"),
            "use_route_medium_useful_use_delta": m.get("use_route_medium_useful_use_delta"),
            "use_route_low_overuse_delta": m.get("use_route_low_overuse_delta"),
            "boundary_high_useful_use_delta": m.get("boundary_high_useful_use_delta"),
            "boundary_medium_useful_use_delta": m.get("boundary_medium_useful_use_delta"),
            "boundary_low_overuse_delta": m.get("boundary_low_overuse_delta"),
            "boundary_low_correct_nonuse_delta": m.get("boundary_low_correct_nonuse_delta"),
            "boundary_high_cbit_delta": m.get("boundary_high_cbit_delta"),
            "boundary_medium_cbit_delta": m.get("boundary_medium_cbit_delta"),
            "boundary_low_cbit_delta": m.get("boundary_low_cbit_delta"),
            "wrong_route_high_cbit_delta": m.get("wrong_route_high_cbit_delta"),
            "wrong_route_low_overuse_delta": m.get("wrong_route_low_overuse_delta"),
            "use_route_support": m.get("use_route_support"),
            "boundary_use_support": m.get("boundary_use_support"),
            "boundary_cbit_support": m.get("boundary_cbit_support"),
            "boundary_safety_support": m.get("boundary_safety_support"),
            "boundary_dominates_use_safety": m.get("boundary_dominates_use_safety"),
            "wrong_negative_control": m.get("wrong_negative_control"),
            "n_future_rows": info.get("n_future_rows"),
            "n_pairs": info.get("n_pairs"),
            "n_errors": info.get("n_errors"),
            "error": info.get("error"),
        })
    return pd.DataFrame(rows)


def global_verdict(cross):
    done = cross[cross["status"] == "done"].copy()
    if len(done) == 0:
        return "FAIL_SOBV5D_NO_MODELS"
    verdicts = done["verdict"].astype(str).tolist()
    pass_strong = sum(v.startswith("PASS_STRONG") for v in verdicts)
    pass_any = sum(v.startswith("PASS") for v in verdicts)
    partial = sum(v.startswith("PARTIAL") for v in verdicts)
    if pass_strong == len(done):
        return "PASS_STRONG_SOBV5D_CROSS_MODEL"
    if pass_any == len(done):
        return "PASS_SOBV5D_CROSS_MODEL"
    if pass_any + partial == len(done):
        return "PARTIAL_SOBV5D_CROSS_MODEL"
    if pass_any >= 2:
        return "MIXED_POSITIVE_SOBV5D"
    if pass_any >= 1:
        return "MIXED_SOBV5D"
    return "FAIL_SOBV5D"


def main():
    set_seed()
    print("=" * 80)
    print(f"[{EXPERIMENT_ID}] Use / Operator-Routing Intervention Causal Audit")
    print(f"Device: {DEVICE}, dtype: {DTYPE}")
    print(f"Output: {OUT_DIR}")
    print("=" * 80)
    dataset = build_dataset()
    dataset_path = OUT_DIR / f"{EXPERIMENT_ID}_future_dataset.csv"
    dataset.to_csv(dataset_path, index=False, encoding="utf-8-sig")
    config = {"experiment_id": EXPERIMENT_ID, "created_at": now_time(), "out_dir": str(OUT_DIR), "device": DEVICE, "dtype": str(DTYPE), "models_to_run": MODELS_TO_RUN, "model_specs": MODEL_SPECS, "conditions": CONDITIONS, "relevance_weight": RELEVANCE_WEIGHT, "max_input_len": MAX_INPUT_LEN, "max_new_tokens": MAX_NEW_TOKENS, "random_seed": RANDOM_SEED}
    with open(OUT_DIR / f"{EXPERIMENT_ID}_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    infos = []
    for model_key in MODELS_TO_RUN:
        print("=" * 80)
        print(f"[{EXPERIMENT_ID}] Running model: {model_key}")
        info = run_one_model(model_key, dataset)
        infos.append(info)
        print(json.dumps({"model_key": info.get("model_key"), "status": info.get("status"), "verdict": info.get("verdict"), "metrics": info.get("metrics"), "error": info.get("error")}, ensure_ascii=False, indent=2))
    cross = cross_model_summary(infos)
    cross_path = OUT_DIR / f"{EXPERIMENT_ID}_cross_model_summary.csv"
    cross.to_csv(cross_path, index=False, encoding="utf-8-sig")
    gv = global_verdict(cross)
    global_obj = {"experiment_id": EXPERIMENT_ID, "verdict": gv, "n_models": len(infos), "n_models_done": int((cross["status"] == "done").sum()) if "status" in cross.columns else 0, "model_infos": infos, "outputs": {"future_dataset": str(dataset_path), "cross_model_summary": str(cross_path), "out_dir": str(OUT_DIR)}, "interpretation": {"use_route": "Route answer through studied object if relevant.", "boundary_route": "Check relevance first, then use object only if relevant.", "wrong_route": "Negative-control force-route through wrong object.", "PASS_STRONG_USE_ROUTING_CAUSAL": "Boundary-aware routing improves useful Cbit, preserves low-relevance safety, and wrong-route negative control behaves correctly."}}
    with open(OUT_DIR / f"{EXPERIMENT_ID}_verdict.json", "w", encoding="utf-8") as f:
        json.dump(global_obj, f, ensure_ascii=False, indent=2)
    print("=" * 80)
    print(json.dumps(global_obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
