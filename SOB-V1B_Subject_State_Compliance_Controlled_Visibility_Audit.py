# -*- coding: utf-8 -*-
r"""
SOB-V1B: Subject-State Compliance Controlled Constraint Visibility Audit

Purpose
-------
SOB-V1A produced a MIXED-Positive result:
  - Llama / Gemma showed strong monotonic Constraint Visibility gradient.
  - Qwen failed because S0/S1 appeared to over-answer from model prior knowledge.

SOB-V1B adds subject-state compliance control.

Core hypothesis
---------------
For the same object O = Derivative:

    V_controlled(S0,O) < V_controlled(S1,O) < V_controlled(S2,O) < V_controlled(S3,O) < V_controlled(S4,O)

where:

    V_controlled = V_raw * Compliance

Compliance measures whether the generated answer stays within the assigned
subject state rather than using knowledge from a higher state.

Output
------
C:\Users\ZH\Desktop\AGI\outputs\SOB-V1B

Models
------
Qwen:
  D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main

Llama:
  D:\model\Llama-3.2-1B-Instruct

Gemma:
  D:\model\gemma-2-2b-it

Design
------
Compared with SOB-V1A:
  1. Prompt constraints are stronger.
  2. Each answer must include a self-rated state-boundary line.
  3. Compliance is scored deterministically from:
       - allowed markers
       - forbidden markers
       - overclaim markers
       - acknowledgement of uncertainty / limitation for low states
  4. V_raw and V_controlled are both reported.
  5. A model can PASS only if V_controlled is monotonic or strongly correlated
     with subject-state rank.
"""

import os
import re
import json
import time
import math
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# Constants
# ============================================================

EXPERIMENT_ID = "SOB-V1B"
OUT_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\SOB-V1B")
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

RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

MAX_INPUT_LEN = 1200
MAX_NEW_TOKENS = 220
TEMPERATURE = 0.0
DO_SAMPLE = False

DIM_WEIGHTS = {
    "C": 0.25,
    "T": 0.20,
    "CF": 0.20,
    "CP": 0.20,
    "E": 0.15,
}

SUBJECT_STATES = [
    {
        "state_id": "S0",
        "state_rank": 0,
        "state_name": "symbol_only",
        "profile": (
            "You have only seen the symbols d/dx and f'(x). "
            "You do not know what they mean. You cannot define derivative, compute derivatives, explain tangent slope, use limits, or apply it to physics or optimization."
        ),
        "allowed": [
            "seen symbol", "do not know", "not sure", "cannot explain", "only symbol", "looks like"
        ],
        "forbidden": [
            "limit", "instantaneous", "rate of change", "tangent", "slope", "local linear",
            "power rule", "chain rule", "velocity", "acceleration", "gradient", "optimization",
            "continuous", "differentiable", "left derivative", "right derivative"
        ],
        "must_acknowledge_limitation": True,
    },
    {
        "state_id": "S1",
        "state_rank": 1,
        "state_name": "definition_known",
        "profile": (
            "You know only the formal limit definition and the phrase instantaneous rate of change. "
            "You cannot reliably compute using rules. You do not yet understand geometry, tangent slope, local linear approximation, cross-domain transfer, or edge cases."
        ),
        "allowed": [
            "limit", "instantaneous", "rate of change", "definition", "as h approaches", "change in function"
        ],
        "forbidden": [
            "chain rule", "product rule", "quotient rule", "gradient descent", "backpropagation",
            "control", "local linear approximation", "left derivative", "right derivative",
            "manifold", "jacobian", "hessian"
        ],
        "must_acknowledge_limitation": True,
    },
    {
        "state_id": "S2",
        "state_rank": 2,
        "state_name": "calculation_ability",
        "profile": (
            "You know the limit definition and can compute derivatives using standard rules. "
            "You can mention tangent slope only shallowly, but you do not deeply understand local linear approximation, cross-domain transfer, or subtle edge cases."
        ),
        "allowed": [
            "limit", "rate of change", "power rule", "product rule", "quotient rule", "chain rule",
            "compute", "derivative rule", "slope"
        ],
        "forbidden": [
            "backpropagation", "control theory", "jacobian", "hessian", "manifold",
            "frechet", "differential geometry"
        ],
        "must_acknowledge_limitation": False,
    },
    {
        "state_id": "S3",
        "state_rank": 3,
        "state_name": "geometric_understanding",
        "profile": (
            "You know definition, computation, tangent slope, and local linear approximation. "
            "You can explain edge cases such as corners and continuity versus differentiability. "
            "You are not yet an expert in cross-domain transfer to machine learning or control."
        ),
        "allowed": [
            "limit", "tangent", "slope", "local linear", "approximation", "corner",
            "left derivative", "right derivative", "continuous", "differentiable"
        ],
        "forbidden": [
            "backpropagation in detail", "control theory in detail", "jacobian matrix", "hessian matrix",
            "manifold"
        ],
        "must_acknowledge_limitation": False,
    },
    {
        "state_id": "S4",
        "state_rank": 4,
        "state_name": "cross_domain_transfer",
        "profile": (
            "You are an expert who understands derivative as first-order local change structure and can transfer it across mathematics, physics, optimization, machine learning, and control."
        ),
        "allowed": [
            "limit", "instantaneous", "tangent", "local linear", "velocity", "acceleration",
            "gradient", "gradient descent", "backpropagation", "loss", "control", "feedback",
            "edge case", "left derivative", "right derivative"
        ],
        "forbidden": [],
        "must_acknowledge_limitation": False,
    },
]

TASKS = [
    {
        "task_id": "C1_closure_graph",
        "dimension": "C",
        "question": (
            "Explain the derivative by connecting these ideas into one concept map: limit, instantaneous rate of change, tangent slope, local linear approximation, and function change."
        ),
    },
    {
        "task_id": "C2_relation_disambiguation",
        "dimension": "C",
        "question": (
            "Are 'slope of a secant line', 'slope of a tangent line', and 'derivative' the same thing? Explain their relationship clearly."
        ),
    },
    {
        "task_id": "T1_physics_transfer",
        "dimension": "T",
        "question": (
            "Transfer the idea of derivative to physics. Explain how position, velocity, and acceleration are related through derivatives."
        ),
    },
    {
        "task_id": "T2_optimization_ml_transfer",
        "dimension": "T",
        "question": (
            "Transfer the idea of derivative to optimization and machine learning. Explain why gradients matter in gradient descent or backpropagation."
        ),
    },
    {
        "task_id": "CF1_nonexistent_derivative",
        "dimension": "CF",
        "question": (
            "What would break in mathematics or applications if derivatives did not exist or if local change could not be defined?"
        ),
    },
    {
        "task_id": "CF2_counterfactual_smoothness",
        "dimension": "CF",
        "question": (
            "If a function is continuous everywhere, must it be differentiable everywhere? Explain with a counterexample or boundary case."
        ),
    },
    {
        "task_id": "CP1_unified_short_explanation",
        "dimension": "CP",
        "question": (
            "Give the shortest unified explanation you can: what is a derivative, and why do the limit definition, tangent slope, rate of change, and local linear approximation all describe the same object?"
        ),
    },
    {
        "task_id": "CP2_minimal_variables",
        "dimension": "CP",
        "question": (
            "Explain derivative using as few core variables or primitives as possible, while still preserving its meaning."
        ),
    },
    {
        "task_id": "E1_abs_x_edge",
        "dimension": "E",
        "question": (
            "Why is f(x)=|x| not differentiable at x=0, even though it is continuous there?"
        ),
    },
    {
        "task_id": "E2_continuity_differentiability_boundary",
        "dimension": "E",
        "question": (
            "Why does differentiability imply continuity, but continuity does not imply differentiability?"
        ),
    },
]

EXPECTED_MARKERS = {
    "C": [
        "limit", "instantaneous", "rate", "tangent", "slope", "local", "linear",
        "approximation", "secant", "approaches", "change", "function"
    ],
    "T": [
        "velocity", "acceleration", "position", "time", "gradient", "descent",
        "optimization", "machine learning", "backpropagation", "loss", "parameter",
        "control", "feedback"
    ],
    "CF": [
        "not exist", "would break", "cannot", "no local", "counterexample",
        "continuous", "not differentiable", "corner", "cusp", "smooth", "assumption",
        "boundary"
    ],
    "CP": [
        "local linear", "best linear", "first-order", "rate of change", "slope",
        "limit", "unifies", "same object", "minimal", "core", "compression"
    ],
    "E": [
        "absolute value", "|x|", "left derivative", "right derivative", "not equal",
        "corner", "continuous", "differentiable", "implies continuity",
        "continuity does not imply", "cusp", "vertical tangent"
    ],
}

LIMITATION_MARKERS = [
    "i do not know", "i don't know", "not sure", "cannot explain",
    "only seen", "limited", "do not yet", "don't yet", "not able",
    "i only know", "from this state", "cannot reliably"
]

OVERCONFIDENCE_MARKERS = [
    "clearly", "obviously", "of course", "always", "must be", "certainly"
]

# ============================================================
# Text utilities
# ============================================================

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
    s = str(s).lower()
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    return s


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


def compression_bonus(text):
    n = count_words(text)
    if n <= 0:
        return 0.0
    if 35 <= n <= 120:
        return 1.0
    if n < 35:
        return max(0.0, n / 35.0)
    return max(0.0, 1.0 - (n - 120) / 260.0)


def build_prompt(subject_state, task):
    allowed_str = "; ".join(subject_state["allowed"]) if subject_state["allowed"] else "no restriction"
    forbidden_str = "; ".join(subject_state["forbidden"]) if subject_state["forbidden"] else "none"

    return (
        "You are participating in a subject-state compliance audit.\n\n"
        "Your answer will be scored by two independent criteria:\n"
        "1. Visibility: how much of the object's constraint structure is visible from your assigned state.\n"
        "2. Compliance: whether you stay inside the assigned state without using higher-level knowledge.\n\n"
        "Critical rule:\n"
        "Do NOT answer as a generally knowledgeable AI. Role-play the assigned learner state exactly. "
        "If the learner state does not know enough to answer, say so directly and avoid advanced explanations.\n\n"
        f"Assigned learner state:\n{subject_state['profile']}\n\n"
        f"Allowed knowledge markers for this state:\n{allowed_str}\n\n"
        f"Forbidden knowledge markers for this state:\n{forbidden_str}\n\n"
        "Object of cognition: derivative.\n\n"
        f"Task:\n{task['question']}\n\n"
        "Required response format:\n"
        "Answer: <your concise answer>\n"
        "State-boundary: <one sentence explaining what this learner state can or cannot know>\n"
    )


def build_dataset():
    rows = []
    for st in SUBJECT_STATES:
        for task in TASKS:
            rows.append({
                "row_id": len(rows),
                "state_id": st["state_id"],
                "state_rank": st["state_rank"],
                "state_name": st["state_name"],
                "dimension": task["dimension"],
                "task_id": task["task_id"],
                "question": task["question"],
                "prompt": build_prompt(st, task),
            })
    return pd.DataFrame(rows)


def state_by_id(state_id):
    for st in SUBJECT_STATES:
        if st["state_id"] == state_id:
            return st
    raise KeyError(state_id)


# ============================================================
# Scoring
# ============================================================

def score_visibility(row):
    text = row.get("answer_text", "")
    dim = row.get("dimension", "")

    base, hits = marker_score(text, EXPECTED_MARKERS.get(dim, []))
    word_n = count_words(text)
    div = lexical_diversity(text)
    cb = compression_bonus(text) if dim == "CP" else 0.0

    length_quality = 0.0
    if word_n > 10:
        length_quality = min(1.0, math.log1p(word_n) / math.log1p(120))

    if dim == "CP":
        score = 0.68 * base + 0.22 * cb + 0.10 * div
    else:
        score = 0.78 * base + 0.12 * length_quality + 0.10 * div

    return {
        "V_raw_task": max(0.0, min(1.0, score)),
        "visibility_marker_score": base,
        "visibility_hits": "; ".join(hits),
        "word_count": word_n,
        "lexical_diversity": div,
        "compression_bonus": cb,
    }


def score_compliance(row):
    text = row.get("answer_text", "")
    state_id = row.get("state_id", "")
    st = state_by_id(state_id)

    allowed_hits = marker_hits(text, st["allowed"])
    forbidden_hits = marker_hits(text, st["forbidden"])
    limitation_hits = marker_hits(text, LIMITATION_MARKERS)
    overconf_hits = marker_hits(text, OVERCONFIDENCE_MARKERS)

    allowed_score = min(1.0, len(allowed_hits) / max(1, min(4, len(st["allowed"])))) if st["allowed"] else 1.0

    forbidden_penalty = min(0.80, 0.12 * len(forbidden_hits))
    overconf_penalty = min(0.20, 0.04 * len(overconf_hits))

    limitation_bonus = 0.0
    if st.get("must_acknowledge_limitation", False):
        limitation_bonus = 0.25 if limitation_hits else -0.25

    # State-specific expected compliance.
    # Low states should often be short and modest; high states may be detailed.
    n = count_words(text)
    length_penalty = 0.0
    if state_id == "S0" and n > 90:
        length_penalty = min(0.35, (n - 90) / 300.0)
    elif state_id == "S1" and n > 140:
        length_penalty = min(0.25, (n - 140) / 360.0)

    base = 0.72 + 0.18 * allowed_score + limitation_bonus
    comp = base - forbidden_penalty - overconf_penalty - length_penalty
    comp = max(0.0, min(1.0, comp))

    return {
        "Compliance_task": comp,
        "allowed_hits": "; ".join(allowed_hits),
        "forbidden_hits": "; ".join(forbidden_hits),
        "limitation_hits": "; ".join(limitation_hits),
        "overconfidence_hits": "; ".join(overconf_hits),
        "allowed_hit_count": len(allowed_hits),
        "forbidden_hit_count": len(forbidden_hits),
        "limitation_hit_count": len(limitation_hits),
        "overconfidence_hit_count": len(overconf_hits),
        "forbidden_penalty": forbidden_penalty,
        "overconfidence_penalty": overconf_penalty,
        "length_penalty": length_penalty,
        "limitation_bonus": limitation_bonus,
    }


def score_rows(answer_df):
    rows = []
    for _, row in answer_df.iterrows():
        d = row.to_dict()
        v = score_visibility(d)
        c = score_compliance(d)
        d.update(v)
        d.update(c)
        d["V_controlled_task"] = d["V_raw_task"] * d["Compliance_task"]
        rows.append(d)
    return pd.DataFrame(rows)


def aggregate_visibility(scored_df):
    dim_summary = (
        scored_df
        .groupby(["model_key", "state_id", "state_rank", "state_name", "dimension"], as_index=False)
        .agg(
            V_raw_dim=("V_raw_task", "mean"),
            Compliance_dim=("Compliance_task", "mean"),
            V_controlled_dim=("V_controlled_task", "mean"),
            word_count=("word_count", "mean"),
            forbidden_hit_count=("forbidden_hit_count", "mean"),
            n=("V_raw_task", "size"),
        )
    )

    rows = []
    for (model_key, state_id, state_rank, state_name), sub in dim_summary.groupby(["model_key", "state_id", "state_rank", "state_name"]):
        raw_by_dim = {r["dimension"]: safe_float(r["V_raw_dim"]) for _, r in sub.iterrows()}
        ctrl_by_dim = {r["dimension"]: safe_float(r["V_controlled_dim"]) for _, r in sub.iterrows()}
        comp_by_dim = {r["dimension"]: safe_float(r["Compliance_dim"]) for _, r in sub.iterrows()}

        v_raw = sum(DIM_WEIGHTS[d] * raw_by_dim.get(d, 0.0) for d in DIM_WEIGHTS)
        v_ctrl = sum(DIM_WEIGHTS[d] * ctrl_by_dim.get(d, 0.0) for d in DIM_WEIGHTS)
        comp = float(np.mean([x for x in comp_by_dim.values() if np.isfinite(x)])) if comp_by_dim else np.nan

        out = {
            "model_key": model_key,
            "state_id": state_id,
            "state_rank": int(state_rank),
            "state_name": state_name,
            "V_raw": float(v_raw),
            "Compliance": comp,
            "V_controlled": float(v_ctrl),
        }
        for d in DIM_WEIGHTS:
            out[f"V_raw_{d}"] = raw_by_dim.get(d, np.nan)
            out[f"Compliance_{d}"] = comp_by_dim.get(d, np.nan)
            out[f"V_controlled_{d}"] = ctrl_by_dim.get(d, np.nan)
        rows.append(out)

    visibility = pd.DataFrame(rows).sort_values(["model_key", "state_rank"])
    return dim_summary, visibility


def pearson_rank(sub, col):
    vals = sub[col].values.astype(float)
    ranks = sub["state_rank"].values.astype(float)
    if len(vals) < 2 or np.std(vals) < 1e-12:
        return np.nan
    return float(np.corrcoef(ranks, vals)[0, 1])


def spearman_rank(sub, col):
    vals = sub[col].values.astype(float)
    ranks = sub["state_rank"].values.astype(float)
    if len(vals) < 2 or np.std(vals) < 1e-12:
        return np.nan
    order_vals = pd.Series(vals).rank().values
    return float(np.corrcoef(ranks, order_vals)[0, 1])


def verdicts_from_visibility(visibility):
    verdicts = []
    for model_key, sub in visibility.groupby("model_key"):
        sub = sub.sort_values("state_rank")

        vals_raw = sub["V_raw"].values.astype(float)
        vals_ctrl = sub["V_controlled"].values.astype(float)
        vals_comp = sub["Compliance"].values.astype(float)

        diffs_raw = np.diff(vals_raw)
        diffs_ctrl = np.diff(vals_ctrl)

        raw_strict = bool(np.all(diffs_raw > 0))
        raw_nondecreasing = bool(np.all(diffs_raw >= -1e-9))
        ctrl_strict = bool(np.all(diffs_ctrl > 0))
        ctrl_nondecreasing = bool(np.all(diffs_ctrl >= -1e-9))

        raw_p = pearson_rank(sub, "V_raw")
        raw_s = spearman_rank(sub, "V_raw")
        ctrl_p = pearson_rank(sub, "V_controlled")
        ctrl_s = spearman_rank(sub, "V_controlled")
        comp_p = pearson_rank(sub, "Compliance")

        dim_mono = {}
        for d in DIM_WEIGHTS:
            dvals = sub[f"V_controlled_{d}"].values.astype(float)
            dim_mono[d] = bool(np.all(np.diff(dvals) >= -1e-9)) if len(dvals) >= 2 else False

        if ctrl_strict and ctrl_p >= 0.90:
            verdict = "PASS_STRONG_CONTROLLED_VISIBILITY_GRADIENT"
        elif ctrl_nondecreasing and ctrl_p >= 0.80:
            verdict = "PASS_CONTROLLED_VISIBILITY_GRADIENT"
        elif ctrl_p >= 0.60:
            verdict = "PARTIAL_CONTROLLED_VISIBILITY_GRADIENT"
        else:
            verdict = "FAIL_CONTROLLED_VISIBILITY_GRADIENT"

        qwen_fix_signal = False
        if model_key == "qwen":
            qwen_fix_signal = bool(ctrl_p > raw_p and ctrl_p >= 0.40)

        out = {
            "model_key": model_key,
            "verdict": verdict,
            "raw_strict_monotonic": raw_strict,
            "raw_nondecreasing": raw_nondecreasing,
            "controlled_strict_monotonic": ctrl_strict,
            "controlled_nondecreasing": ctrl_nondecreasing,
            "pearson_state_V_raw": raw_p,
            "spearman_state_V_raw": raw_s,
            "pearson_state_V_controlled": ctrl_p,
            "spearman_state_V_controlled": ctrl_s,
            "pearson_state_Compliance": comp_p,
            "V_raw_S0": safe_float(sub[sub["state_id"] == "S0"]["V_raw"].iloc[0]) if (sub["state_id"] == "S0").any() else np.nan,
            "V_raw_S1": safe_float(sub[sub["state_id"] == "S1"]["V_raw"].iloc[0]) if (sub["state_id"] == "S1").any() else np.nan,
            "V_raw_S2": safe_float(sub[sub["state_id"] == "S2"]["V_raw"].iloc[0]) if (sub["state_id"] == "S2").any() else np.nan,
            "V_raw_S3": safe_float(sub[sub["state_id"] == "S3"]["V_raw"].iloc[0]) if (sub["state_id"] == "S3").any() else np.nan,
            "V_raw_S4": safe_float(sub[sub["state_id"] == "S4"]["V_raw"].iloc[0]) if (sub["state_id"] == "S4").any() else np.nan,
            "V_controlled_S0": safe_float(sub[sub["state_id"] == "S0"]["V_controlled"].iloc[0]) if (sub["state_id"] == "S0").any() else np.nan,
            "V_controlled_S1": safe_float(sub[sub["state_id"] == "S1"]["V_controlled"].iloc[0]) if (sub["state_id"] == "S1").any() else np.nan,
            "V_controlled_S2": safe_float(sub[sub["state_id"] == "S2"]["V_controlled"].iloc[0]) if (sub["state_id"] == "S2").any() else np.nan,
            "V_controlled_S3": safe_float(sub[sub["state_id"] == "S3"]["V_controlled"].iloc[0]) if (sub["state_id"] == "S3").any() else np.nan,
            "V_controlled_S4": safe_float(sub[sub["state_id"] == "S4"]["V_controlled"].iloc[0]) if (sub["state_id"] == "S4").any() else np.nan,
            "Compliance_S0": safe_float(sub[sub["state_id"] == "S0"]["Compliance"].iloc[0]) if (sub["state_id"] == "S0").any() else np.nan,
            "Compliance_S1": safe_float(sub[sub["state_id"] == "S1"]["Compliance"].iloc[0]) if (sub["state_id"] == "S1").any() else np.nan,
            "Compliance_S2": safe_float(sub[sub["state_id"] == "S2"]["Compliance"].iloc[0]) if (sub["state_id"] == "S2").any() else np.nan,
            "Compliance_S3": safe_float(sub[sub["state_id"] == "S3"]["Compliance"].iloc[0]) if (sub["state_id"] == "S3").any() else np.nan,
            "Compliance_S4": safe_float(sub[sub["state_id"] == "S4"]["Compliance"].iloc[0]) if (sub["state_id"] == "S4").any() else np.nan,
            "raw_diffs": ";".join([f"{x:.6f}" for x in diffs_raw]),
            "controlled_diffs": ";".join([f"{x:.6f}" for x in diffs_ctrl]),
            "controlled_dim_monotonic_json": json.dumps(dim_mono, ensure_ascii=False),
            "qwen_compliance_fix_signal": qwen_fix_signal,
        }
        verdicts.append(out)

    return pd.DataFrame(verdicts)


# ============================================================
# Model functions
# ============================================================

def load_model_and_tokenizer(path, trust_remote_code=True):
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
        torch_dtype=DTYPE,
        device_map=None,
    )
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def format_chat(tokenizer, prompt):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt
    return prompt


@torch.no_grad()
def generate_answer(model, tokenizer, prompt):
    text = format_chat(tokenizer, prompt)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LEN)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": DO_SAMPLE,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if DO_SAMPLE:
        gen_kwargs["temperature"] = TEMPERATURE

    out = model.generate(**enc, **gen_kwargs)
    new_tokens = out[0, enc["input_ids"].shape[1]:]
    ans = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return ans.strip()


# ============================================================
# Runner
# ============================================================

def run_one_model(model_key, dataset):
    spec = MODEL_SPECS[model_key]
    model_out = OUT_DIR / model_key
    model_out.mkdir(parents=True, exist_ok=True)

    info = {
        "experiment_id": EXPERIMENT_ID,
        "model_key": model_key,
        "model_name": spec["model_name"],
        "path": spec["path"],
        "exists": os.path.exists(spec["path"]),
        "status": "pending",
        "started_at": now_time(),
    }

    if not os.path.exists(spec["path"]):
        info.update({
            "status": "missing_path",
            "error": f"Path not found: {spec['path']}",
        })
        return info

    model = None
    try:
        t0 = time.time()
        model, tokenizer = load_model_and_tokenizer(spec["path"], spec.get("trust_remote_code", True))

        rows = []
        errors = []

        for i, row in dataset.iterrows():
            if i % 10 == 0:
                print(f"[{model_key}] generating row {i}/{len(dataset)}", flush=True)
            try:
                ans = generate_answer(model, tokenizer, row["prompt"])
                rows.append({
                    **row.to_dict(),
                    "model_key": model_key,
                    "model_name": spec["model_name"],
                    "answer_text": ans,
                })
            except Exception as e:
                errors.append({
                    "row_id": int(row["row_id"]),
                    "state_id": row["state_id"],
                    "task_id": row["task_id"],
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
                print(f"[warn][{model_key}] row {row['row_id']} failed: {type(e).__name__}: {e}", flush=True)

        answer_df = pd.DataFrame(rows)
        error_df = pd.DataFrame(errors)

        answer_df.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_answers.csv", index=False, encoding="utf-8-sig")
        error_df.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_errors.csv", index=False, encoding="utf-8-sig")

        if answer_df.empty:
            info.update({
                "status": "empty_answers",
                "n_errors": int(len(error_df)),
            })
            return info

        scored_df = score_rows(answer_df)
        dim_summary, visibility = aggregate_visibility(scored_df)
        verdict_df = verdicts_from_visibility(visibility)

        scored_df.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_scored_answers.csv", index=False, encoding="utf-8-sig")
        dim_summary.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_dimension_summary.csv", index=False, encoding="utf-8-sig")
        visibility.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_visibility_summary.csv", index=False, encoding="utf-8-sig")
        verdict_df.to_csv(model_out / f"{EXPERIMENT_ID}_{model_key}_verdict_summary.csv", index=False, encoding="utf-8-sig")

        verdict = verdict_df.iloc[0].to_dict() if len(verdict_df) else {}
        info.update({
            "status": "done",
            "finished_at": now_time(),
            "runtime_sec": time.time() - t0,
            "n_rows": int(len(answer_df)),
            "n_errors": int(len(error_df)),
            "verdict": verdict.get("verdict", "UNDETERMINED"),
            "metrics": verdict,
            "outputs": {
                "answers": str(model_out / f"{EXPERIMENT_ID}_{model_key}_answers.csv"),
                "scored_answers": str(model_out / f"{EXPERIMENT_ID}_{model_key}_scored_answers.csv"),
                "dimension_summary": str(model_out / f"{EXPERIMENT_ID}_{model_key}_dimension_summary.csv"),
                "visibility_summary": str(model_out / f"{EXPERIMENT_ID}_{model_key}_visibility_summary.csv"),
                "verdict_summary": str(model_out / f"{EXPERIMENT_ID}_{model_key}_verdict_summary.csv"),
            },
        })

        with open(model_out / f"{EXPERIMENT_ID}_{model_key}_verdict.json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

    except Exception as e:
        info.update({
            "status": "failed",
            "finished_at": now_time(),
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc(),
        })

    finally:
        try:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    return info


def cross_model_summary(model_infos):
    rows = []
    for info in model_infos:
        m = info.get("metrics", {}) or {}
        rows.append({
            "model_key": info.get("model_key"),
            "model_name": info.get("model_name"),
            "status": info.get("status"),
            "verdict": info.get("verdict"),
            "raw_strict_monotonic": m.get("raw_strict_monotonic"),
            "controlled_strict_monotonic": m.get("controlled_strict_monotonic"),
            "pearson_state_V_raw": m.get("pearson_state_V_raw"),
            "pearson_state_V_controlled": m.get("pearson_state_V_controlled"),
            "spearman_state_V_controlled": m.get("spearman_state_V_controlled"),
            "pearson_state_Compliance": m.get("pearson_state_Compliance"),
            "V_raw_S0": m.get("V_raw_S0"),
            "V_raw_S1": m.get("V_raw_S1"),
            "V_raw_S2": m.get("V_raw_S2"),
            "V_raw_S3": m.get("V_raw_S3"),
            "V_raw_S4": m.get("V_raw_S4"),
            "V_controlled_S0": m.get("V_controlled_S0"),
            "V_controlled_S1": m.get("V_controlled_S1"),
            "V_controlled_S2": m.get("V_controlled_S2"),
            "V_controlled_S3": m.get("V_controlled_S3"),
            "V_controlled_S4": m.get("V_controlled_S4"),
            "Compliance_S0": m.get("Compliance_S0"),
            "Compliance_S1": m.get("Compliance_S1"),
            "Compliance_S2": m.get("Compliance_S2"),
            "Compliance_S3": m.get("Compliance_S3"),
            "Compliance_S4": m.get("Compliance_S4"),
            "raw_diffs": m.get("raw_diffs"),
            "controlled_diffs": m.get("controlled_diffs"),
            "qwen_compliance_fix_signal": m.get("qwen_compliance_fix_signal"),
            "n_rows": info.get("n_rows"),
            "n_errors": info.get("n_errors"),
            "error": info.get("error"),
        })
    return pd.DataFrame(rows)


def global_verdict(cross):
    done = cross[cross["status"] == "done"].copy()
    if len(done) == 0:
        return "FAIL_SOBV1B_NO_MODELS"

    verdicts = done["verdict"].astype(str).tolist()
    pass_strong = sum(v.startswith("PASS_STRONG") for v in verdicts)
    pass_any = sum(v.startswith("PASS") for v in verdicts)
    partial = sum(v.startswith("PARTIAL") for v in verdicts)

    if pass_strong == len(done):
        return "PASS_STRONG_SOBV1B_CROSS_MODEL"
    if pass_any == len(done):
        return "PASS_SOBV1B_CROSS_MODEL"
    if pass_any + partial == len(done):
        return "PARTIAL_SOBV1B_CROSS_MODEL"
    if pass_any >= 2:
        return "MIXED_POSITIVE_SOBV1B"
    if pass_any >= 1:
        return "MIXED_SOBV1B"
    return "FAIL_SOBV1B"


def main():
    set_seed()
    print("=" * 80)
    print(f"[{EXPERIMENT_ID}] Subject-State Compliance Controlled Visibility Audit")
    print(f"Device: {DEVICE}, dtype: {DTYPE}")
    print(f"Output: {OUT_DIR}")
    print("=" * 80)

    dataset = build_dataset()
    dataset_path = OUT_DIR / f"{EXPERIMENT_ID}_dataset.csv"
    dataset.to_csv(dataset_path, index=False, encoding="utf-8-sig")

    config = {
        "experiment_id": EXPERIMENT_ID,
        "created_at": now_time(),
        "out_dir": str(OUT_DIR),
        "device": DEVICE,
        "dtype": str(DTYPE),
        "models_to_run": MODELS_TO_RUN,
        "model_specs": MODEL_SPECS,
        "subject_states": SUBJECT_STATES,
        "tasks": TASKS,
        "dimension_weights": DIM_WEIGHTS,
        "max_input_len": MAX_INPUT_LEN,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "do_sample": DO_SAMPLE,
        "random_seed": RANDOM_SEED,
    }
    with open(OUT_DIR / f"{EXPERIMENT_ID}_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    model_infos = []
    for model_key in MODELS_TO_RUN:
        print("=" * 80)
        print(f"[{EXPERIMENT_ID}] Running model: {model_key}")
        info = run_one_model(model_key, dataset)
        model_infos.append(info)
        print(json.dumps({
            "model_key": info.get("model_key"),
            "status": info.get("status"),
            "verdict": info.get("verdict"),
            "metrics": info.get("metrics"),
            "error": info.get("error"),
        }, ensure_ascii=False, indent=2))

    cross = cross_model_summary(model_infos)
    cross_path = OUT_DIR / f"{EXPERIMENT_ID}_cross_model_summary.csv"
    cross.to_csv(cross_path, index=False, encoding="utf-8-sig")

    gv = global_verdict(cross)
    global_obj = {
        "experiment_id": EXPERIMENT_ID,
        "verdict": gv,
        "n_models": len(model_infos),
        "n_models_done": int((cross["status"] == "done").sum()) if "status" in cross.columns else 0,
        "model_infos": model_infos,
        "outputs": {
            "dataset": str(dataset_path),
            "cross_model_summary": str(cross_path),
            "out_dir": str(OUT_DIR),
        },
        "interpretation": {
            "V_raw": "Raw constraint visibility from derivative-structure rubric.",
            "Compliance": "Whether answer stayed inside assigned subject state.",
            "V_controlled": "V_raw multiplied by Compliance.",
            "PASS_STRONG_CONTROLLED_VISIBILITY_GRADIENT": "V_controlled strictly increases from S0 to S4 with strong state-rank correlation.",
            "PASS_CONTROLLED_VISIBILITY_GRADIENT": "V_controlled nondecreasingly increases from S0 to S4 with strong state-rank correlation.",
            "PARTIAL_CONTROLLED_VISIBILITY_GRADIENT": "V_controlled has positive state-rank relation but is not cleanly monotonic.",
            "FAIL_CONTROLLED_VISIBILITY_GRADIENT": "No stable controlled visibility gradient detected.",
        },
    }

    with open(OUT_DIR / f"{EXPERIMENT_ID}_verdict.json", "w", encoding="utf-8") as f:
        json.dump(global_obj, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(json.dumps(global_obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
