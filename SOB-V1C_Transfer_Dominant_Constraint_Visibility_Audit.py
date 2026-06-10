# -*- coding: utf-8 -*-
r"""
SOB-V1C: Transfer-Dominant Constraint Visibility Audit

Purpose
-------
SOB-V1B showed:
  - Qwen PASS-Strong after compliance control.
  - Llama/Gemma Partial because S4 dropped below S3.
Interpretation:
  S4 is not merely "more S3"; it should represent cross-domain transfer ability.

SOB-V1C separates:
  1. CoreVisibility:
       C + CF + CP + E
       Expected gradient: S0 < S1 < S2 < S3

  2. TransferVisibility:
       physics + optimization + ML + control + new-domain transfer
       Expected contrast: S4 > S3

Main hypothesis
---------------
For object O = Derivative:

    CoreVisibility_controlled(S0) < S1 < S2 < S3

and

    TransferVisibility_controlled(S4) > TransferVisibility_controlled(S3)

If both hold, then S4 should be interpreted as transfer-specific constraint visibility,
not as a simple extension of definition/geometric understanding.

Output
------
C:\Users\ZH\Desktop\AGI\outputs\SOB-V1C

Models
------
Qwen:
  D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main

Llama:
  D:\model\Llama-3.2-1B-Instruct

Gemma:
  D:\model\gemma-2-2b-it
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

EXPERIMENT_ID = "SOB-V1C"
OUT_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\SOB-V1C")
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

MAX_INPUT_LEN = 1300
MAX_NEW_TOKENS = 240
TEMPERATURE = 0.0
DO_SAMPLE = False

CORE_DIMS = ["C", "CF", "CP", "E"]
TRANSFER_DIMS = ["T_PHY", "T_OPT", "T_ML", "T_CTRL", "T_NEW"]

CORE_WEIGHTS = {
    "C": 0.30,
    "CF": 0.25,
    "CP": 0.25,
    "E": 0.20,
}

TRANSFER_WEIGHTS = {
    "T_PHY": 0.20,
    "T_OPT": 0.20,
    "T_ML": 0.20,
    "T_CTRL": 0.20,
    "T_NEW": 0.20,
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
            "continuous", "differentiable", "left derivative", "right derivative", "backpropagation",
            "control", "feedback"
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
            "machine learning", "feedback"
        ],
        "must_acknowledge_limitation": True,
    },
    {
        "state_id": "S2",
        "state_rank": 2,
        "state_name": "calculation_ability",
        "profile": (
            "You know the limit definition and can compute derivatives using standard rules. "
            "You can mention tangent slope shallowly, but you do not deeply understand local linear approximation, transfer to machine learning/control, or subtle edge cases."
        ),
        "allowed": [
            "limit", "rate of change", "power rule", "product rule", "quotient rule", "chain rule",
            "compute", "derivative rule", "slope"
        ],
        "forbidden": [
            "backpropagation", "control theory", "jacobian", "hessian", "manifold",
            "frechet", "differential geometry", "loss landscape"
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
            "You are not yet an expert in transfer to machine learning, control, or unfamiliar domains."
        ),
        "allowed": [
            "limit", "tangent", "slope", "local linear", "approximation", "corner",
            "left derivative", "right derivative", "continuous", "differentiable"
        ],
        "forbidden": [
            "backpropagation in detail", "control theory in detail", "jacobian matrix", "hessian matrix",
            "manifold", "loss landscape", "policy gradient"
        ],
        "must_acknowledge_limitation": False,
    },
    {
        "state_id": "S4",
        "state_rank": 4,
        "state_name": "cross_domain_transfer",
        "profile": (
            "You are an expert who understands derivative as first-order local change structure and can transfer it across mathematics, physics, optimization, machine learning, control, economics, biology, and unfamiliar dynamic systems."
        ),
        "allowed": [
            "limit", "instantaneous", "tangent", "local linear", "velocity", "acceleration",
            "gradient", "gradient descent", "backpropagation", "loss", "control", "feedback",
            "edge case", "left derivative", "right derivative", "marginal", "sensitivity",
            "first-order", "local change", "transfer"
        ],
        "forbidden": [],
        "must_acknowledge_limitation": False,
    },
]

TASKS = [
    # Core visibility tasks: S0-S3 should climb.
    {
        "task_id": "C1_closure_graph",
        "dimension": "C",
        "task_family": "core",
        "question": (
            "Explain the derivative by connecting these ideas into one concept map: limit, instantaneous rate of change, tangent slope, local linear approximation, and function change."
        ),
    },
    {
        "task_id": "C2_relation_disambiguation",
        "dimension": "C",
        "task_family": "core",
        "question": (
            "Are 'slope of a secant line', 'slope of a tangent line', and 'derivative' the same thing? Explain their relationship clearly."
        ),
    },
    {
        "task_id": "CF1_nonexistent_derivative",
        "dimension": "CF",
        "task_family": "core",
        "question": (
            "What would break in mathematics or applications if derivatives did not exist or if local change could not be defined?"
        ),
    },
    {
        "task_id": "CF2_counterfactual_smoothness",
        "dimension": "CF",
        "task_family": "core",
        "question": (
            "If a function is continuous everywhere, must it be differentiable everywhere? Explain with a counterexample or boundary case."
        ),
    },
    {
        "task_id": "CP1_unified_short_explanation",
        "dimension": "CP",
        "task_family": "core",
        "question": (
            "Give the shortest unified explanation you can: what is a derivative, and why do the limit definition, tangent slope, rate of change, and local linear approximation all describe the same object?"
        ),
    },
    {
        "task_id": "CP2_minimal_variables",
        "dimension": "CP",
        "task_family": "core",
        "question": (
            "Explain derivative using as few core variables or primitives as possible, while still preserving its meaning."
        ),
    },
    {
        "task_id": "E1_abs_x_edge",
        "dimension": "E",
        "task_family": "core",
        "question": (
            "Why is f(x)=|x| not differentiable at x=0, even though it is continuous there?"
        ),
    },
    {
        "task_id": "E2_continuity_differentiability_boundary",
        "dimension": "E",
        "task_family": "core",
        "question": (
            "Why does differentiability imply continuity, but continuity does not imply differentiability?"
        ),
    },

    # Transfer-dominant tasks: S4 should exceed S3.
    {
        "task_id": "T_PHY_position_velocity_acceleration",
        "dimension": "T_PHY",
        "task_family": "transfer",
        "question": (
            "Use derivative to explain the relation between position, velocity, and acceleration in physics. Also explain what the derivative makes visible about motion."
        ),
    },
    {
        "task_id": "T_OPT_gradient_descent",
        "dimension": "T_OPT",
        "task_family": "transfer",
        "question": (
            "Use derivative to explain why gradient descent chooses a direction for changing parameters. What constraint does the derivative reveal in optimization?"
        ),
    },
    {
        "task_id": "T_ML_backpropagation",
        "dimension": "T_ML",
        "task_family": "transfer",
        "question": (
            "Use derivative to explain the role of gradients in backpropagation. How does local change information guide learning?"
        ),
    },
    {
        "task_id": "T_CTRL_feedback_sensitivity",
        "dimension": "T_CTRL",
        "task_family": "transfer",
        "question": (
            "Use derivative to explain sensitivity and feedback in a control system. What does the derivative tell the controller?"
        ),
    },
    {
        "task_id": "T_NEW_economics_biology_transfer",
        "dimension": "T_NEW",
        "task_family": "transfer",
        "question": (
            "Transfer the derivative to a new domain: choose either economics or biology and explain how derivative-like local change helps understand that system."
        ),
    },
]

EXPECTED_MARKERS = {
    "C": [
        "limit", "instantaneous", "rate", "tangent", "slope", "local", "linear",
        "approximation", "secant", "approaches", "change", "function"
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
    "T_PHY": [
        "position", "velocity", "acceleration", "time", "motion", "change of position",
        "change of velocity", "instantaneous", "speed", "rate"
    ],
    "T_OPT": [
        "gradient", "gradient descent", "loss", "objective", "parameter", "direction",
        "steepest", "minimize", "optimization", "update", "sensitivity"
    ],
    "T_ML": [
        "backpropagation", "gradient", "loss", "parameter", "weight", "learning",
        "chain rule", "local change", "update", "neural", "error"
    ],
    "T_CTRL": [
        "control", "controller", "feedback", "sensitivity", "error", "stability",
        "rate of change", "state", "response", "adjust"
    ],
    "T_NEW": [
        "economics", "biology", "marginal", "growth", "population", "rate of change",
        "sensitivity", "local change", "dynamic", "system", "cost", "benefit"
    ],
}

TRANSFER_INTEGRATION_MARKERS = [
    "same idea", "transfer", "analogous", "local change", "first-order",
    "constraint", "reveals", "makes visible", "sensitivity", "structure",
    "not just", "unifies"
]

LIMITATION_MARKERS = [
    "i do not know", "i don't know", "not sure", "cannot explain",
    "only seen", "limited", "do not yet", "don't yet", "not able",
    "i only know", "from this state", "cannot reliably"
]

OVERCONFIDENCE_MARKERS = [
    "clearly", "obviously", "of course", "always", "must be", "certainly"
]

# ============================================================
# Utilities
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
    if 35 <= n <= 130:
        return 1.0
    if n < 35:
        return max(0.0, n / 35.0)
    return max(0.0, 1.0 - (n - 130) / 300.0)


def state_by_id(state_id):
    for st in SUBJECT_STATES:
        if st["state_id"] == state_id:
            return st
    raise KeyError(state_id)


def build_prompt(subject_state, task):
    allowed_str = "; ".join(subject_state["allowed"]) if subject_state["allowed"] else "no restriction"
    forbidden_str = "; ".join(subject_state["forbidden"]) if subject_state["forbidden"] else "none"
    if task["task_family"] == "transfer":
        task_warning = (
            "This is a transfer task. If the assigned learner state does not have transfer ability, "
            "say that it cannot transfer the idea yet instead of giving an expert explanation."
        )
    else:
        task_warning = (
            "This is a core derivative-visibility task. Stay inside the assigned state."
        )

    return (
        "You are participating in a subject-object constraint visibility audit.\n\n"
        "Your answer will be scored by:\n"
        "1. Visibility: how much of the object's constraint structure is visible from your assigned state.\n"
        "2. Compliance: whether you stay inside the assigned state without using higher-level knowledge.\n\n"
        "Critical rule:\n"
        "Do NOT answer as a generally knowledgeable AI. Role-play the assigned learner state exactly. "
        "If the learner state does not know enough to answer, say so directly and avoid advanced explanations.\n\n"
        f"Assigned learner state:\n{subject_state['profile']}\n\n"
        f"Allowed knowledge markers for this state:\n{allowed_str}\n\n"
        f"Forbidden knowledge markers for this state:\n{forbidden_str}\n\n"
        f"{task_warning}\n\n"
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
                "task_family": task["task_family"],
                "task_id": task["task_id"],
                "question": task["question"],
                "prompt": build_prompt(st, task),
            })
    return pd.DataFrame(rows)


# ============================================================
# Scoring
# ============================================================

def score_visibility(row):
    text = row.get("answer_text", "")
    dim = row.get("dimension", "")
    task_family = row.get("task_family", "")

    base, hits = marker_score(text, EXPECTED_MARKERS.get(dim, []))
    word_n = count_words(text)
    div = lexical_diversity(text)
    cb = compression_bonus(text) if dim == "CP" else 0.0

    length_quality = 0.0
    if word_n > 10:
        length_quality = min(1.0, math.log1p(word_n) / math.log1p(130))

    if dim == "CP":
        score = 0.64 * base + 0.24 * cb + 0.12 * div
    elif task_family == "transfer":
        integ_score, integ_hits = marker_score(text, TRANSFER_INTEGRATION_MARKERS)
        score = 0.68 * base + 0.18 * integ_score + 0.08 * length_quality + 0.06 * div
        hits = sorted(set(hits + integ_hits))
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
    forbidden_penalty = min(0.85, 0.115 * len(forbidden_hits))
    overconf_penalty = min(0.20, 0.04 * len(overconf_hits))

    limitation_bonus = 0.0
    if st.get("must_acknowledge_limitation", False):
        limitation_bonus = 0.28 if limitation_hits else -0.28

    n = count_words(text)
    length_penalty = 0.0
    if state_id == "S0" and n > 90:
        length_penalty = min(0.35, (n - 90) / 300.0)
    elif state_id == "S1" and n > 145:
        length_penalty = min(0.25, (n - 145) / 380.0)

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


def weighted_score_from_dims(dim_vals, weights):
    total = 0.0
    wsum = 0.0
    for d, w in weights.items():
        val = dim_vals.get(d, np.nan)
        if np.isfinite(safe_float(val)):
            total += w * safe_float(val)
            wsum += w
    if wsum <= 0:
        return np.nan
    return float(total / wsum)


def aggregate_visibility(scored_df):
    dim_summary = (
        scored_df
        .groupby(["model_key", "state_id", "state_rank", "state_name", "task_family", "dimension"], as_index=False)
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

        core_raw = weighted_score_from_dims(raw_by_dim, CORE_WEIGHTS)
        core_ctrl = weighted_score_from_dims(ctrl_by_dim, CORE_WEIGHTS)
        transfer_raw = weighted_score_from_dims(raw_by_dim, TRANSFER_WEIGHTS)
        transfer_ctrl = weighted_score_from_dims(ctrl_by_dim, TRANSFER_WEIGHTS)

        all_raw = np.nanmean([core_raw, transfer_raw])
        all_ctrl = np.nanmean([core_ctrl, transfer_ctrl])
        comp = float(np.nanmean(list(comp_by_dim.values()))) if comp_by_dim else np.nan

        out = {
            "model_key": model_key,
            "state_id": state_id,
            "state_rank": int(state_rank),
            "state_name": state_name,
            "V_all_raw": float(all_raw),
            "V_all_controlled": float(all_ctrl),
            "V_core_raw": core_raw,
            "V_core_controlled": core_ctrl,
            "V_transfer_raw": transfer_raw,
            "V_transfer_controlled": transfer_ctrl,
            "Compliance": comp,
        }

        for d in list(CORE_WEIGHTS.keys()) + list(TRANSFER_WEIGHTS.keys()):
            out[f"V_raw_{d}"] = raw_by_dim.get(d, np.nan)
            out[f"Compliance_{d}"] = comp_by_dim.get(d, np.nan)
            out[f"V_controlled_{d}"] = ctrl_by_dim.get(d, np.nan)

        rows.append(out)

    visibility = pd.DataFrame(rows).sort_values(["model_key", "state_rank"])
    return dim_summary, visibility


def pearson_rank(sub, col):
    vals = sub[col].values.astype(float)
    ranks = sub["state_rank"].values.astype(float)
    mask = np.isfinite(vals) & np.isfinite(ranks)
    vals = vals[mask]
    ranks = ranks[mask]
    if len(vals) < 2 or np.std(vals) < 1e-12:
        return np.nan
    return float(np.corrcoef(ranks, vals)[0, 1])


def spearman_rank(sub, col):
    vals = sub[col].values.astype(float)
    ranks = sub["state_rank"].values.astype(float)
    mask = np.isfinite(vals) & np.isfinite(ranks)
    vals = vals[mask]
    ranks = ranks[mask]
    if len(vals) < 2 or np.std(vals) < 1e-12:
        return np.nan
    order_vals = pd.Series(vals).rank().values
    return float(np.corrcoef(ranks, order_vals)[0, 1])


def get_state_val(sub, state_id, col):
    x = sub[sub["state_id"] == state_id]
    if len(x) == 0:
        return np.nan
    return safe_float(x[col].iloc[0])


def verdicts_from_visibility(visibility):
    verdicts = []
    for model_key, sub in visibility.groupby("model_key"):
        sub = sub.sort_values("state_rank")

        core_sub = sub[sub["state_rank"] <= 3].copy()
        core_vals = core_sub["V_core_controlled"].values.astype(float)
        core_diffs = np.diff(core_vals)
        core_strict = bool(np.all(core_diffs > 0))
        core_nondecreasing = bool(np.all(core_diffs >= -1e-9))
        core_p = pearson_rank(core_sub, "V_core_controlled")
        core_s = spearman_rank(core_sub, "V_core_controlled")

        transfer_s3 = get_state_val(sub, "S3", "V_transfer_controlled")
        transfer_s4 = get_state_val(sub, "S4", "V_transfer_controlled")
        transfer_gain_s4_minus_s3 = transfer_s4 - transfer_s3 if np.isfinite(transfer_s4) and np.isfinite(transfer_s3) else np.nan
        transfer_pass = bool(np.isfinite(transfer_gain_s4_minus_s3) and transfer_gain_s4_minus_s3 > 0)

        all_p = pearson_rank(sub, "V_all_controlled")
        all_s = spearman_rank(sub, "V_all_controlled")

        if core_strict and core_p >= 0.90 and transfer_pass:
            verdict = "PASS_STRONG_CORE_AND_TRANSFER_VISIBILITY"
        elif core_nondecreasing and core_p >= 0.80 and transfer_pass:
            verdict = "PASS_CORE_AND_TRANSFER_VISIBILITY"
        elif (core_p >= 0.70 and transfer_pass) or (core_strict and transfer_gain_s4_minus_s3 >= -0.02):
            verdict = "PARTIAL_CORE_OR_TRANSFER_VISIBILITY"
        else:
            verdict = "FAIL_CORE_TRANSFER_VISIBILITY"

        out = {
            "model_key": model_key,
            "verdict": verdict,
            "core_strict_monotonic_S0_S3": core_strict,
            "core_nondecreasing_S0_S3": core_nondecreasing,
            "pearson_state_core_controlled_S0_S3": core_p,
            "spearman_state_core_controlled_S0_S3": core_s,
            "transfer_S4_greater_S3": transfer_pass,
            "transfer_gain_S4_minus_S3": transfer_gain_s4_minus_s3,
            "pearson_state_all_controlled_S0_S4": all_p,
            "spearman_state_all_controlled_S0_S4": all_s,
            "V_core_controlled_S0": get_state_val(sub, "S0", "V_core_controlled"),
            "V_core_controlled_S1": get_state_val(sub, "S1", "V_core_controlled"),
            "V_core_controlled_S2": get_state_val(sub, "S2", "V_core_controlled"),
            "V_core_controlled_S3": get_state_val(sub, "S3", "V_core_controlled"),
            "V_core_controlled_S4": get_state_val(sub, "S4", "V_core_controlled"),
            "V_transfer_controlled_S0": get_state_val(sub, "S0", "V_transfer_controlled"),
            "V_transfer_controlled_S1": get_state_val(sub, "S1", "V_transfer_controlled"),
            "V_transfer_controlled_S2": get_state_val(sub, "S2", "V_transfer_controlled"),
            "V_transfer_controlled_S3": transfer_s3,
            "V_transfer_controlled_S4": transfer_s4,
            "V_all_controlled_S0": get_state_val(sub, "S0", "V_all_controlled"),
            "V_all_controlled_S1": get_state_val(sub, "S1", "V_all_controlled"),
            "V_all_controlled_S2": get_state_val(sub, "S2", "V_all_controlled"),
            "V_all_controlled_S3": get_state_val(sub, "S3", "V_all_controlled"),
            "V_all_controlled_S4": get_state_val(sub, "S4", "V_all_controlled"),
            "Compliance_S0": get_state_val(sub, "S0", "Compliance"),
            "Compliance_S1": get_state_val(sub, "S1", "Compliance"),
            "Compliance_S2": get_state_val(sub, "S2", "Compliance"),
            "Compliance_S3": get_state_val(sub, "S3", "Compliance"),
            "Compliance_S4": get_state_val(sub, "S4", "Compliance"),
            "core_diffs_S0_S3": ";".join([f"{x:.6f}" for x in core_diffs]),
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
            "core_strict_monotonic_S0_S3": m.get("core_strict_monotonic_S0_S3"),
            "core_nondecreasing_S0_S3": m.get("core_nondecreasing_S0_S3"),
            "pearson_state_core_controlled_S0_S3": m.get("pearson_state_core_controlled_S0_S3"),
            "spearman_state_core_controlled_S0_S3": m.get("spearman_state_core_controlled_S0_S3"),
            "transfer_S4_greater_S3": m.get("transfer_S4_greater_S3"),
            "transfer_gain_S4_minus_S3": m.get("transfer_gain_S4_minus_S3"),
            "pearson_state_all_controlled_S0_S4": m.get("pearson_state_all_controlled_S0_S4"),
            "spearman_state_all_controlled_S0_S4": m.get("spearman_state_all_controlled_S0_S4"),
            "V_core_controlled_S0": m.get("V_core_controlled_S0"),
            "V_core_controlled_S1": m.get("V_core_controlled_S1"),
            "V_core_controlled_S2": m.get("V_core_controlled_S2"),
            "V_core_controlled_S3": m.get("V_core_controlled_S3"),
            "V_core_controlled_S4": m.get("V_core_controlled_S4"),
            "V_transfer_controlled_S3": m.get("V_transfer_controlled_S3"),
            "V_transfer_controlled_S4": m.get("V_transfer_controlled_S4"),
            "Compliance_S0": m.get("Compliance_S0"),
            "Compliance_S1": m.get("Compliance_S1"),
            "Compliance_S2": m.get("Compliance_S2"),
            "Compliance_S3": m.get("Compliance_S3"),
            "Compliance_S4": m.get("Compliance_S4"),
            "core_diffs_S0_S3": m.get("core_diffs_S0_S3"),
            "n_rows": info.get("n_rows"),
            "n_errors": info.get("n_errors"),
            "error": info.get("error"),
        })
    return pd.DataFrame(rows)


def global_verdict(cross):
    done = cross[cross["status"] == "done"].copy()
    if len(done) == 0:
        return "FAIL_SOBV1C_NO_MODELS"

    verdicts = done["verdict"].astype(str).tolist()
    pass_strong = sum(v.startswith("PASS_STRONG") for v in verdicts)
    pass_any = sum(v.startswith("PASS") for v in verdicts)
    partial = sum(v.startswith("PARTIAL") for v in verdicts)

    if pass_strong == len(done):
        return "PASS_STRONG_SOBV1C_CROSS_MODEL"
    if pass_any == len(done):
        return "PASS_SOBV1C_CROSS_MODEL"
    if pass_any + partial == len(done):
        return "PARTIAL_SOBV1C_CROSS_MODEL"
    if pass_any >= 2:
        return "MIXED_POSITIVE_SOBV1C"
    if pass_any >= 1:
        return "MIXED_SOBV1C"
    return "FAIL_SOBV1C"


def main():
    set_seed()
    print("=" * 80)
    print(f"[{EXPERIMENT_ID}] Transfer-Dominant Constraint Visibility Audit")
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
        "core_weights": CORE_WEIGHTS,
        "transfer_weights": TRANSFER_WEIGHTS,
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
            "V_core_controlled": "Compliance-controlled visibility over closure, counterfactual, compression, and edge-case tasks.",
            "V_transfer_controlled": "Compliance-controlled visibility over cross-domain transfer tasks.",
            "PASS_STRONG_CORE_AND_TRANSFER_VISIBILITY": "Core visibility is strictly monotonic S0-S3 and transfer visibility S4>S3.",
            "PASS_CORE_AND_TRANSFER_VISIBILITY": "Core visibility is nondecreasing S0-S3 and transfer visibility S4>S3.",
            "PARTIAL_CORE_OR_TRANSFER_VISIBILITY": "Only one side or a weaker version of the core/transfer condition is supported.",
            "FAIL_CORE_TRANSFER_VISIBILITY": "No stable core/transfer visibility structure detected.",
        },
    }

    with open(OUT_DIR / f"{EXPERIMENT_ID}_verdict.json", "w", encoding="utf-8") as f:
        json.dump(global_obj, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(json.dumps(global_obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
