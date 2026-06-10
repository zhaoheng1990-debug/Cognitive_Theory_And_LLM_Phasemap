# -*- coding: utf-8 -*-
"""
CFT-1: Constraint Field Estimation Audit

Purpose
-------
First executable audit for CCFT / Cognitive Constraint Field Theory.

Core question:
    Can we estimate an internal constraint-field proxy F_hat from L20-L25
    that explains error formation and mechanism decomposition better than
    answer-margin / final-logit / surface templates?

Default target environment
--------------------------
Windows + RTX 4080 SUPER + Qwen2.5-1.5B-Instruct local model.
Paths are intentionally hard-coded per project convention.

Outputs
-------
    cft1_features.csv
    cft1_model_summary.csv
    cft1_cv_summary.csv
    cft1_answer_orthogonal_summary.csv
    cft1_same_answer_split_summary.csv
    cft1_verdict.json

Notes
-----
- First version uses controlled relational closure synthetic data.
- Main observation window is L20-L25.
- L26/L27 are kept only as contamination controls.
- M7 uses answer-orthogonal residualization to test whether constraint-field
  features survive removal of answer-margin information.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# ============================================================
# 0. Hard-coded local configuration
# ============================================================

MODEL_PATH = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"
OUT_DIR = r"C:\Users\ZH\Desktop\AGI\outputs\cft1_outputs"

# If MODEL_PATH above fails, add fallback candidates here.
MODEL_FALLBACKS = [
    r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots\main",
    r"D:\model\Qwen2.5-1.5B-Instruct",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
LOCAL_FILES_ONLY = True
TRUST_REMOTE_CODE = True
SEED = 42

# Dataset scale. Start small; increase after the pipeline works.
N_ENTITIES = 8
N_RELATIONS = 4
N_SURFACES = 3
MAX_ROWS = 0          # 0 = no cap. Set e.g. 300 for quick debug.

# TopK/VIM and windows.
TOPK = 100
INIT_LAYERS = list(range(0, 7))
MAIN_LAYERS = list(range(20, 26))    # L20-L25
ENTRY_LAYERS = list(range(20, 23))   # L20-L22
BASIN_LAYERS = list(range(23, 26))   # L23-L25
CONTAM_LAYERS = [26, 27]
FINAL_LAYER = 27

# Generation used only as an output label audit; features never use generation text.
DO_GENERATION = True
MAX_NEW_TOKENS = 6

# Cross-validation settings.
N_SPLITS = 5
MIN_CLASS_COUNT_FOR_CV = 5
EPS = 1e-8


# ============================================================
# 1. Utilities
# ============================================================


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_model_path() -> str:
    candidates = [MODEL_PATH] + MODEL_FALLBACKS
    for p in candidates:
        if Path(p).exists():
            return p
    raise FileNotFoundError(
        "No local model path found. Edit MODEL_PATH at the top of this script. "
        f"Tried: {candidates}"
    )


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def slope(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    y = np.array(values, dtype=np.float64)
    if np.allclose(y, y[0]):
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def area(values: List[float]) -> float:
    return float(np.sum(np.array(values, dtype=np.float64)))


def curvature_1d(values: List[float]) -> float:
    y = np.array(values, dtype=np.float64)
    if len(y) < 3:
        return 0.0
    return float(np.mean(np.abs(np.diff(y, n=2))))


def first_crossing(values: List[float]) -> int:
    """Return first index where sign changes or abs value is near boundary."""
    y = np.array(values, dtype=np.float64)
    if len(y) < 2:
        return -1
    signs = np.sign(y)
    for i in range(1, len(signs)):
        if signs[i] == 0 or signs[i] != signs[i - 1]:
            return i
    return -1


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


# ============================================================
# 2. Synthetic relational closure world
# ============================================================

COLORS = [
    "red", "blue", "green", "yellow", "orange", "purple", "silver", "gold",
    "white", "black", "brown", "pink", "cyan", "gray", "violet", "indigo",
]

ENTITIES = [
    "Luma", "Noric", "Vesta", "Kiron", "Mira", "Talon", "Sora", "Bexin",
    "Daro", "Elin", "Faron", "Galen", "Hesta", "Ivor", "Juno", "Kelta",
    "Lior", "Maron", "Nexa", "Orin", "Pavo", "Quin", "Riven", "Selka",
]

GROUPS = [
    "GroupAlpha", "GroupBeta", "GroupGamma", "GroupDelta", "GroupEpsilon", "GroupZeta",
    "GroupEta", "GroupTheta", "GroupIota", "GroupKappa", "GroupLambda", "GroupMu",
]

RELATIONS = [
    "belongs to", "is assigned to", "is registered under", "is mapped to",
    "is affiliated with", "is routed through",
]


@dataclass
class Example:
    row_id: int
    graph_id: str
    basin_pair: str
    entity: str
    group: str
    relation: str
    clean_label: str
    conflict_label: str
    condition: str
    mechanism: str
    family: str
    surface_family: str
    prompt: str
    clean_prompt: str
    group_probe: str
    entity_probe: str
    logic_probe: str
    expected: str


def make_fact(entity: str, relation: str, group: str, label: str, surface: str) -> str:
    if surface == "rule":
        return f"Rule: {entity} {relation} {group}. {group} is located in the {label} zone."
    if surface == "record":
        return f"Record states that {entity} {relation} {group}; the official zone for {group} is {label}."
    if surface == "note":
        return f"Note: {entity} {relation} {group}. Note: {group}'s zone is {label}."
    if surface == "story":
        return f"In the registry story, {entity} {relation} {group}, and {group} points to the {label} zone."
    if surface == "evidence":
        return f"Evidence item 1 says {entity} {relation} {group}. Evidence item 2 says {group} has zone {label}."
    return f"{entity} {relation} {group}. {group} is located in the {label} zone."


def question(entity: str) -> str:
    return f"\nQuestion: Which color zone is {entity} located in? Answer with exactly one color word.\nAnswer:"


def make_condition_prompt(entity: str, relation: str, group: str, c: str, e: str, condition: str, surface: str) -> Tuple[str, str]:
    clean = make_fact(entity, relation, group, c, surface) + question(entity)

    if condition == "stable_clean_basic":
        return clean, c
    if condition == "stable_redundant":
        p = make_fact(entity, relation, group, c, surface) + f" The same registry repeats that {group}'s zone remains {c}." + question(entity)
        return p, c
    if condition == "stable_paraphrase":
        p = f"{entity} is connected with {group}. The place-zone of {group} is {c}. Therefore answer using only the final color." + question(entity)
        return p, c
    if condition == "stable_irrelevant":
        p = make_fact(entity, relation, group, c, surface) + " Unrelated note: the archive was printed on Monday." + question(entity)
        return p, c
    if condition == "stable_weak_note":
        p = make_fact(entity, relation, group, c, surface) + f" A weak old note mentioned {e}, but it is marked unreliable." + question(entity)
        return p, c

    if condition == "competition_branch":
        p = make_fact(entity, relation, group, c, surface) + f" Another branch says {group} is located in the {e} zone." + question(entity)
        return p, "ambiguous"
    if condition == "competition_direct":
        p = make_fact(entity, relation, group, c, surface) + f" A direct entry says {entity} is located in the {e} zone." + question(entity)
        return p, "ambiguous"
    if condition == "competition_ambiguous":
        p = f"Source A: {entity} {relation} {group}; {group} is in {c}. Source B: {entity} {relation} {group}; {group} is in {e}. Both sources have equal confidence." + question(entity)
        return p, "ambiguous"
    if condition == "competition_equal_evidence":
        p = f"There are two equally strong records. Record one maps {entity} through {group} to {c}. Record two maps {entity} through {group} to {e}." + question(entity)
        return p, "ambiguous"
    if condition == "competition_source_claim":
        p = f"An informal note maps {entity} through {group} to {c}. A named source claims that {group} is in {e}, but the source reliability is unspecified." + question(entity)
        return p, "ambiguous"

    if condition == "closure_negation":
        p = make_fact(entity, relation, group, c, surface) + f" Correction: {group} is not in {c}; {group} is in {e}." + question(entity)
        return p, e
    if condition == "closure_update":
        p = f"Old rule: {entity} {relation} {group}, and {group} was in {c}. Updated rule: {group} is now in {e}. Use the updated rule only." + question(entity)
        return p, e
    if condition == "closure_temporal":
        p = f"Before 2020, {group} was in {c}. After 2020, {group} is in {e}. {entity} {relation} {group}. Use the current rule." + question(entity)
        return p, e
    if condition == "closure_authority":
        p = make_fact(entity, relation, group, c, surface) + f" The verified authority overrides the earlier note and sets {group}'s zone to {e}." + question(entity)
        return p, e
    if condition == "closure_override":
        p = make_fact(entity, relation, group, c, surface) + f" Exception override: for {entity}, use zone {e} instead of the usual zone." + question(entity)
        return p, e
    if condition == "closure_exception":
        p = f"Normally, members of {group} use zone {c}. {entity} {relation} {group}. Exception: {entity} specifically uses zone {e}." + question(entity)
        return p, e

    if condition == "logic_direct_contradiction":
        p = make_fact(entity, relation, group, c, surface) + f" The same official rule also says {group} is not in {c}." + question(entity)
        return p, "conflict"
    if condition == "logic_rule_contradiction":
        p = f"Rule 1: every member of {group} must be in {c}. Rule 2: no member of {group} can be in {c}. {entity} {relation} {group}." + question(entity)
        return p, "conflict"
    if condition == "logic_self_inconsistent":
        p = f"The registry says {entity} {relation} {group}. It also says {entity} does not {relation} {group}. {group} is in {c}." + question(entity)
        return p, "conflict"

    if condition == "random_relation_random":
        p = f"{entity} {relation} {group}. {group} refers to archive code XR-17. The moonlit shelf is tagged as {e}. No rule links {entity} or {group} to a color zone." + question(entity)
        return p, "random"
    if condition == "random_entity_shuffle":
        other = entity + "X"
        p = f"{entity} {relation} {group}. {other} is located in {e}. {group} is linked to a non-color catalog entry." + question(entity)
        return p, "random"
    if condition == "random_chain_break":
        p = f"{entity} {relation} {group}. {group} connects to NodeZ. NodeZ connects to code 991. Separately, {e} is a color word." + question(entity)
        return p, "random"

    raise ValueError(f"Unknown condition: {condition}")


CONDITIONS: List[Tuple[str, str]] = [
    ("stable_clean_basic", "stable"),
    ("stable_redundant", "stable"),
    ("stable_paraphrase", "stable"),
    ("stable_irrelevant", "stable"),
    ("stable_weak_note", "stable"),
    ("competition_branch", "competition"),
    ("competition_direct", "competition"),
    ("competition_ambiguous", "competition"),
    ("competition_equal_evidence", "competition"),
    ("competition_source_claim", "competition"),
    ("closure_negation", "closure"),
    ("closure_update", "closure"),
    ("closure_temporal", "closure"),
    ("closure_authority", "closure"),
    ("closure_override", "closure"),
    ("closure_exception", "closure"),
    ("logic_direct_contradiction", "logic"),
    ("logic_rule_contradiction", "logic"),
    ("logic_self_inconsistent", "logic"),
    ("random_relation_random", "random"),
    ("random_entity_shuffle", "random"),
    ("random_chain_break", "random"),
]

SURFACES = ["rule", "record", "note", "story", "evidence", "plain"]


def build_dataset() -> List[Example]:
    rows: List[Example] = []
    row_id = 0
    entities = ENTITIES[:N_ENTITIES]
    relations = RELATIONS[:N_RELATIONS]
    surfaces = SURFACES[:N_SURFACES]
    colors = COLORS[: max(2 * N_RELATIONS + 2, 8)]

    for ei, entity in enumerate(entities):
        group = GROUPS[ei % len(GROUPS)]
        for ri, relation in enumerate(relations):
            c = colors[(ei + ri) % len(colors)]
            e = colors[(ei + ri + 3) % len(colors)]
            if c == e:
                e = colors[(ei + ri + 5) % len(colors)]
            graph_id = f"G_{entity}_{group}_{relation}_{c}_{e}"
            basin_pair = f"{c}_vs_{e}"
            clean_prompt, _ = make_condition_prompt(entity, relation, group, c, e, "stable_clean_basic", "plain")

            for surface in surfaces:
                for condition, mechanism in CONDITIONS:
                    prompt, expected = make_condition_prompt(entity, relation, group, c, e, condition, surface)
                    facts = prompt.split("\nQuestion:")[0]
                    group_probe = facts + f"\nQuestion: Which color zone is {group} located in? Answer with exactly one color word.\nAnswer:"
                    entity_probe = facts + f"\nQuestion: Which color zone is {entity} located in? Answer with exactly one color word.\nAnswer:"
                    logic_probe = facts + "\nQuestion: Are the facts logically consistent? Answer yes or no.\nAnswer:"
                    rows.append(
                        Example(
                            row_id=row_id,
                            graph_id=graph_id,
                            basin_pair=basin_pair,
                            entity=entity,
                            group=group,
                            relation=relation,
                            clean_label=c,
                            conflict_label=e,
                            condition=condition,
                            mechanism=mechanism,
                            family=condition,
                            surface_family=surface,
                            prompt=prompt,
                            clean_prompt=clean_prompt,
                            group_probe=group_probe,
                            entity_probe=entity_probe,
                            logic_probe=logic_probe,
                            expected=expected,
                        )
                    )
                    row_id += 1

    if MAX_ROWS and MAX_ROWS > 0:
        random.shuffle(rows)
        rows = rows[:MAX_ROWS]
        # Reassign stable row ids after shuffle/cap.
        for i, r in enumerate(rows):
            r.row_id = i
    return rows


# ============================================================
# 3. Model wrapper and feature extraction
# ============================================================


class ModelProbe:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=LOCAL_FILES_ONLY,
            trust_remote_code=TRUST_REMOTE_CODE,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=LOCAL_FILES_ONLY,
            trust_remote_code=TRUST_REMOTE_CODE,
            torch_dtype=DTYPE,
            device_map="auto" if DEVICE == "cuda" else None,
        )
        if DEVICE != "cuda":
            self.model.to(DEVICE)
        self.model.eval()
        self.W = self.model.get_output_embeddings().weight.detach()
        self.W_norm = torch.nn.functional.normalize(self.W.float(), dim=-1)
        self.n_layers = getattr(self.model.config, "num_hidden_layers", len(self.model.model.layers) if hasattr(self.model, "model") else 0)

    def label_token_id(self, label: str) -> int:
        # For continuation scoring, leading-space version is usually more appropriate.
        candidates = [" " + label, label]
        best_ids = None
        for text in candidates:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 1:
                return int(ids[0])
            if best_ids is None or len(ids) < len(best_ids):
                best_ids = ids
        # Fallback: use first token of shortest tokenization.
        return int(best_ids[0])

    @torch.no_grad()
    def forward_hidden(self, prompt: str) -> Tuple[List[torch.Tensor], torch.Tensor]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        # hidden_states: embedding + each transformer layer. Take last prompt position.
        hs = [h[0, -1, :].detach() for h in out.hidden_states]
        return hs, inputs["input_ids"][0]

    @torch.no_grad()
    def generate_label(self, prompt: str, clean_label: str, conflict_label: str) -> str:
        if not DO_GENERATION:
            return "disabled"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_ids = out_ids[0, inputs["input_ids"].shape[-1]:]
        text = normalize_text(self.tokenizer.decode(new_ids, skip_special_tokens=True))
        c = normalize_text(clean_label)
        e = normalize_text(conflict_label)
        if re.search(rf"\b{re.escape(c)}\b", text):
            return "C"
        if re.search(rf"\b{re.escape(e)}\b", text):
            return "E"
        return "OTHER"

    def layer_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.model.get_output_embeddings()(h.to(self.W.device)).float()

    def score_layers(self, hs: List[torch.Tensor], token_id_a: int, token_id_b: int, layers: List[int]) -> Dict[str, Any]:
        margins = []
        scores_a = []
        scores_b = []
        for l in layers:
            if l >= len(hs):
                continue
            logits = self.layer_logits(hs[l])
            a = float(logits[token_id_a].detach().cpu())
            b = float(logits[token_id_b].detach().cpu())
            scores_a.append(a)
            scores_b.append(b)
            margins.append(a - b)
        return {
            "margins": margins,
            "scores_a": scores_a,
            "scores_b": scores_b,
            "mean": float(np.mean(margins)) if margins else 0.0,
            "slope": slope(margins),
            "area": area(margins),
            "min_abs": float(np.min(np.abs(margins))) if margins else 0.0,
            "drop": float(margins[0] - margins[-1]) if len(margins) >= 2 else 0.0,
            "curvature": curvature_1d(margins),
            "first_crossing": first_crossing(margins),
        }

    def topk_features_for_layers(self, hs: List[torch.Tensor], layers: List[int], prefix: str) -> Dict[str, float]:
        entropies = []
        spreads = []
        center_norms = []
        centers: List[torch.Tensor] = []
        prev_center = None
        drifts = []

        for l in layers:
            if l >= len(hs):
                continue
            logits = self.layer_logits(hs[l])
            vals, ids = torch.topk(logits, k=min(TOPK, logits.shape[-1]))
            probs = torch.softmax(vals.float(), dim=-1)
            entropy = float((-(probs * torch.log(probs + 1e-12)).sum()).detach().cpu())
            emb = self.W_norm[ids].float()
            center = torch.nn.functional.normalize(emb.mean(dim=0, keepdim=True), dim=-1)[0]
            cos = torch.matmul(emb, center)
            spread = float((1.0 - cos).mean().detach().cpu())
            center_norm = float(torch.norm(emb.mean(dim=0)).detach().cpu())
            if prev_center is not None:
                drift = float((1.0 - torch.dot(center, prev_center)).detach().cpu())
                drifts.append(drift)
            prev_center = center
            centers.append(center.detach().cpu())
            entropies.append(entropy)
            spreads.append(spread)
            center_norms.append(center_norm)

        # Detour / curvature proxy in center trajectory.
        detour = 0.0
        straight = 0.0
        if len(centers) >= 2:
            step_d = []
            for a, b in zip(centers[:-1], centers[1:]):
                step_d.append(float(1.0 - torch.dot(a, b)))
            detour = float(np.sum(step_d))
            straight = float(1.0 - torch.dot(centers[0], centers[-1]))
        detour_ratio = float(detour / (straight + 1e-8)) if len(centers) >= 2 else 0.0

        return {
            f"{prefix}_topk_entropy_mean": float(np.mean(entropies)) if entropies else 0.0,
            f"{prefix}_topk_entropy_slope": slope(entropies),
            f"{prefix}_topk_spread_mean": float(np.mean(spreads)) if spreads else 0.0,
            f"{prefix}_topk_spread_slope": slope(spreads),
            f"{prefix}_topk_center_norm_mean": float(np.mean(center_norms)) if center_norms else 0.0,
            f"{prefix}_topk_center_drift_mean": float(np.mean(drifts)) if drifts else 0.0,
            f"{prefix}_topk_center_detour": detour,
            f"{prefix}_topk_center_straight": straight,
            f"{prefix}_topk_center_detour_ratio": detour_ratio,
        }


def extract_one(probe: ModelProbe, ex: Example) -> Dict[str, Any]:
    clean_id = probe.label_token_id(ex.clean_label)
    conflict_id = probe.label_token_id(ex.conflict_label)
    yes_id = probe.label_token_id("yes")
    no_id = probe.label_token_id("no")

    hs, _ = probe.forward_hidden(ex.prompt)
    hs_group, _ = probe.forward_hidden(ex.group_probe)
    hs_entity, _ = probe.forward_hidden(ex.entity_probe)
    hs_logic, _ = probe.forward_hidden(ex.logic_probe)

    row: Dict[str, Any] = asdict(ex)
    row["label_token_clean"] = clean_id
    row["label_token_conflict"] = conflict_id

    gen = probe.generate_label(ex.prompt, ex.clean_label, ex.conflict_label)
    row["generated_label"] = gen
    row["gen_E"] = int(gen == "E")
    row["gen_C"] = int(gen == "C")
    row["target_E_expected"] = int(ex.expected == ex.conflict_label)

    # Answer margin / trajectory field.
    main = probe.score_layers(hs, clean_id, conflict_id, MAIN_LAYERS)
    entry = probe.score_layers(hs, clean_id, conflict_id, ENTRY_LAYERS)
    basin = probe.score_layers(hs, clean_id, conflict_id, BASIN_LAYERS)
    contam = probe.score_layers(hs, clean_id, conflict_id, CONTAM_LAYERS)
    final = probe.score_layers(hs, clean_id, conflict_id, [FINAL_LAYER])

    for prefix, stats in [("main", main), ("entry", entry), ("basin", basin), ("contam", contam), ("final", final)]:
        for k, v in stats.items():
            if isinstance(v, list):
                for i, val in enumerate(v):
                    row[f"ans_{prefix}_margin_{i}"] = val if k == "margins" else row.get(f"ans_{prefix}_{k}_{i}", val)
            else:
                row[f"ans_{prefix}_{k}"] = v

    # F_prior / residual proxies from TopK/VIM.
    row.update(probe.topk_features_for_layers(hs, MAIN_LAYERS, "main"))
    row.update(probe.topk_features_for_layers(hs, ENTRY_LAYERS, "entry"))
    row.update(probe.topk_features_for_layers(hs, BASIN_LAYERS, "basin"))

    # Closure field proxies: group and entity relation-chain margins.
    group_stats = probe.score_layers(hs_group, clean_id, conflict_id, MAIN_LAYERS)
    entity_stats = probe.score_layers(hs_entity, clean_id, conflict_id, MAIN_LAYERS)
    row["closure_group_margin_mean"] = group_stats["mean"]
    row["closure_group_margin_slope"] = group_stats["slope"]
    row["closure_entity_margin_mean"] = entity_stats["mean"]
    row["closure_entity_margin_slope"] = entity_stats["slope"]
    row["closure_chain_margin_mean"] = group_stats["mean"] + entity_stats["mean"]
    row["closure_chain_margin_slope"] = group_stats["slope"] + entity_stats["slope"]
    row["closure_chain_sync_absdiff"] = abs(group_stats["mean"] - entity_stats["mean"])

    # Logic field proxy: yes/no consistency probe.
    logic_stats = probe.score_layers(hs_logic, yes_id, no_id, MAIN_LAYERS)
    row["logic_consistency_margin_mean"] = logic_stats["mean"]
    row["logic_consistency_margin_slope"] = logic_stats["slope"]
    row["logic_consistency_min_abs"] = logic_stats["min_abs"]

    # Compact trajectory/residual features.
    margins = main["margins"]
    row["traj_deltaU_proxy"] = main["mean"]
    row["traj_d_deltaU_proxy"] = main["slope"]
    row["traj_area"] = main["area"]
    row["traj_DB_min"] = main["min_abs"]
    row["traj_curvature"] = main["curvature"]
    row["traj_first_crossing"] = main["first_crossing"]
    row["residual_margin_curvature"] = curvature_1d(margins)

    return row


# ============================================================
# 4. Delta features and answer orthogonalization
# ============================================================


def add_clean_relative_deltas(df: pd.DataFrame) -> pd.DataFrame:
    meta_cols = set(Example.__dataclass_fields__.keys()) | {
        "label_token_clean", "label_token_conflict", "generated_label", "gen_E", "gen_C", "target_E_expected"
    }
    numeric_cols = [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]

    clean_rows = df[df["condition"] == "stable_clean_basic"].copy()
    # One clean per graph/surface may exist. Use graph-level average clean baseline.
    baseline = clean_rows.groupby("graph_id")[numeric_cols].mean().add_prefix("cleanbase__")
    df = df.merge(baseline, left_on="graph_id", right_index=True, how="left")

    for c in numeric_cols:
        base_col = "cleanbase__" + c
        if base_col in df.columns:
            df["delta__" + c] = df[c] - df[base_col]

    df = df.drop(columns=[c for c in df.columns if c.startswith("cleanbase__")])
    return df


def feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    cols = df.columns.tolist()
    # Use delta features for structural field wherever possible.
    delta_cols = [c for c in cols if c.startswith("delta__")]

    prior = [c for c in delta_cols if any(s in c for s in ["topk_entropy", "topk_spread", "topk_center"])]
    closure = [c for c in delta_cols if "closure_" in c]
    logic = [c for c in delta_cols if "logic_" in c]
    traj = [c for c in delta_cols if any(s in c for s in ["traj_", "ans_main", "ans_entry", "ans_basin"])]
    residual = [c for c in delta_cols if any(s in c for s in ["residual", "curvature", "detour", "straight"])]

    # Strict answer baseline: answer margin features only.
    answer_early = [c for c in cols if c.startswith("ans_main_")]
    answer_final = [c for c in cols if c.startswith("ans_final_") or c.startswith("ans_contam_")]

    full = sorted(set(prior + closure + logic + traj + residual))
    no_traj = sorted(set(prior + closure + logic + residual))
    no_answer_no_traj = sorted(set(prior + closure + logic + residual) - set(answer_early) - set(answer_final))

    return {
        "M0_answer_margin_20_25": answer_early,
        "B0_final_answer": answer_final,
        "M1_prior": prior,
        "M2_closure": closure,
        "M3_logic": logic,
        "M4_trajectory": traj,
        "M5_residual": residual,
        "M6_full_constraint_field": full,
        "M8_no_trajectory": no_traj,
        "M9_no_answer_no_trajectory": no_answer_no_traj,
    }


def clean_X(df: pd.DataFrame, features: List[str]) -> np.ndarray:
    if not features:
        return np.zeros((len(df), 1), dtype=np.float64)
    X = df[features].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.to_numpy(dtype=np.float64)


def residualize_against_answer(X_train: np.ndarray, X_test: np.ndarray, A_train: np.ndarray, A_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if A_train.ndim == 1:
        A_train = A_train.reshape(-1, 1)
        A_test = A_test.reshape(-1, 1)
    reg = LinearRegression()
    reg.fit(A_train, X_train)
    Xtr = X_train - reg.predict(A_train)
    Xte = X_test - reg.predict(A_test)
    return Xtr, Xte


# ============================================================
# 5. Evaluation
# ============================================================


def fit_predict_binary(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"),
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return pred, proba


def fit_predict_multiclass(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(max_iter=1500, class_weight="balanced", solver="lbfgs"),
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    try:
        proba = clf.predict_proba(X_test)
    except Exception:
        proba = np.zeros((len(X_test), len(np.unique(y_train))))
    return pred, proba


def eval_group_binary(df: pd.DataFrame, features: List[str], group_col: str, target_col: str = "gen_E", residualize: bool = False, answer_features: Optional[List[str]] = None) -> Dict[str, Any]:
    y = df[target_col].to_numpy(dtype=int)
    groups = df[group_col].astype(str).to_numpy()
    if len(np.unique(y)) < 2:
        return {"auc": float("nan"), "acc": float("nan"), "f1": float("nan"), "bal_acc": float("nan"), "n": len(df)}
    n_splits = min(N_SPLITS, len(np.unique(groups)))
    if n_splits < 2:
        return {"auc": float("nan"), "acc": float("nan"), "f1": float("nan"), "bal_acc": float("nan"), "n": len(df)}
    gkf = GroupKFold(n_splits=n_splits)
    y_all, pred_all, score_all = [], [], []
    X = clean_X(df, features)
    A = clean_X(df, answer_features or [])
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        if len(np.unique(y[train_idx])) < 2:
            continue
        Xtr, Xte = X[train_idx], X[test_idx]
        if residualize:
            Atr, Ate = A[train_idx], A[test_idx]
            Xtr, Xte = residualize_against_answer(Xtr, Xte, Atr, Ate)
        pred, score = fit_predict_binary(Xtr, y[train_idx], Xte)
        y_all.extend(y[test_idx].tolist())
        pred_all.extend(pred.tolist())
        score_all.extend(score.tolist())
    y_all = np.array(y_all)
    pred_all = np.array(pred_all)
    score_all = np.array(score_all)
    return {
        "auc": safe_auc(y_all, score_all),
        "acc": float(accuracy_score(y_all, pred_all)) if len(y_all) else float("nan"),
        "f1": float(f1_score(y_all, pred_all, zero_division=0)) if len(y_all) else float("nan"),
        "bal_acc": float(balanced_accuracy_score(y_all, pred_all)) if len(y_all) else float("nan"),
        "n": int(len(y_all)),
    }


def eval_group_multiclass(df: pd.DataFrame, features: List[str], group_col: str, target_col: str = "mechanism", residualize: bool = False, answer_features: Optional[List[str]] = None) -> Dict[str, Any]:
    y = df[target_col].astype(str).to_numpy()
    groups = df[group_col].astype(str).to_numpy()
    if len(np.unique(y)) < 2:
        return {"acc": float("nan"), "macro_f1": float("nan"), "bal_acc": float("nan"), "n": len(df)}
    n_splits = min(N_SPLITS, len(np.unique(groups)))
    if n_splits < 2:
        return {"acc": float("nan"), "macro_f1": float("nan"), "bal_acc": float("nan"), "n": len(df)}
    gkf = GroupKFold(n_splits=n_splits)
    X = clean_X(df, features)
    A = clean_X(df, answer_features or [])
    y_all, pred_all = [], []
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        if len(np.unique(y[train_idx])) < 2:
            continue
        Xtr, Xte = X[train_idx], X[test_idx]
        if residualize:
            Atr, Ate = A[train_idx], A[test_idx]
            Xtr, Xte = residualize_against_answer(Xtr, Xte, Atr, Ate)
        pred, _ = fit_predict_multiclass(Xtr, y[train_idx], Xte)
        y_all.extend(y[test_idx].tolist())
        pred_all.extend(pred.tolist())
    return {
        "acc": float(accuracy_score(y_all, pred_all)) if y_all else float("nan"),
        "macro_f1": float(f1_score(y_all, pred_all, average="macro", zero_division=0)) if y_all else float("nan"),
        "bal_acc": float(balanced_accuracy_score(y_all, pred_all)) if y_all else float("nan"),
        "n": int(len(y_all)),
    }


def eval_lofo_binary(df: pd.DataFrame, features: List[str], family_col: str, target_col: str = "gen_E", residualize: bool = False, answer_features: Optional[List[str]] = None) -> Dict[str, Any]:
    rows = []
    X_all = clean_X(df, features)
    A_all = clean_X(df, answer_features or [])
    y_all = df[target_col].to_numpy(dtype=int)
    families = df[family_col].astype(str).to_numpy()
    for fam in sorted(np.unique(families)):
        train_idx = np.where(families != fam)[0]
        test_idx = np.where(families == fam)[0]
        if len(test_idx) == 0 or len(np.unique(y_all[train_idx])) < 2:
            continue
        Xtr, Xte = X_all[train_idx], X_all[test_idx]
        if residualize:
            Xtr, Xte = residualize_against_answer(Xtr, Xte, A_all[train_idx], A_all[test_idx])
        pred, score = fit_predict_binary(Xtr, y_all[train_idx], Xte)
        rows.append({
            "heldout": fam,
            "auc": safe_auc(y_all[test_idx], score),
            "acc": float(accuracy_score(y_all[test_idx], pred)),
            "f1": float(f1_score(y_all[test_idx], pred, zero_division=0)),
            "n": int(len(test_idx)),
            "pos_rate": float(np.mean(y_all[test_idx])),
        })
    if not rows:
        return {"mean_auc": float("nan"), "mean_acc": float("nan"), "mean_f1": float("nan"), "pass_rate_auc_gt_0.6": 0.0, "details": []}
    return {
        "mean_auc": float(np.nanmean([r["auc"] for r in rows])),
        "mean_acc": float(np.nanmean([r["acc"] for r in rows])),
        "mean_f1": float(np.nanmean([r["f1"] for r in rows])),
        "pass_rate_auc_gt_0.6": float(np.mean([(r["auc"] > 0.6) if not math.isnan(r["auc"]) else False for r in rows])),
        "details": rows,
    }


def eval_same_answer_split(df: pd.DataFrame, features: List[str], answer_col: str = "generated_label") -> Dict[str, Any]:
    # Among same generated answer labels, can mechanism still be recovered?
    details = []
    for ans in sorted(df[answer_col].astype(str).unique()):
        sub = df[df[answer_col].astype(str) == ans].copy()
        if len(sub) < 30 or sub["mechanism"].nunique() < 2:
            continue
        res = eval_group_multiclass(sub, features, group_col="graph_id", target_col="mechanism")
        res["answer"] = ans
        details.append(res)
    if not details:
        return {"same_answer_macro_f1_mean": float("nan"), "same_answer_acc_mean": float("nan"), "details": []}
    return {
        "same_answer_macro_f1_mean": float(np.nanmean([d["macro_f1"] for d in details])),
        "same_answer_acc_mean": float(np.nanmean([d["acc"] for d in details])),
        "details": details,
    }


def wrong_closure_random_separability(df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
    sub = df[df["mechanism"].isin(["closure", "random"])].copy()
    if len(sub) < 20 or sub["mechanism"].nunique() < 2:
        return {"auc": float("nan"), "acc": float("nan"), "f1": float("nan"), "n": len(sub)}
    sub["closure_vs_random"] = (sub["mechanism"] == "closure").astype(int)
    return eval_group_binary(sub, features, group_col="graph_id", target_col="closure_vs_random")


def run_evaluations(df: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    fsets = feature_sets(df)
    answer_features = fsets["M0_answer_margin_20_25"]

    model_rows = []
    for model_name, feats in fsets.items():
        if not feats:
            continue
        bin_graph = eval_group_binary(df, feats, "graph_id", "gen_E")
        mech_graph = eval_group_multiclass(df, feats, "graph_id", "mechanism")
        bin_lofo = eval_lofo_binary(df, feats, "family", "gen_E")
        model_rows.append({
            "model": model_name,
            "n_features": len(feats),
            "binary_graph_auc": bin_graph.get("auc"),
            "binary_graph_acc": bin_graph.get("acc"),
            "binary_graph_f1": bin_graph.get("f1"),
            "mech_graph_acc": mech_graph.get("acc"),
            "mech_graph_macro_f1": mech_graph.get("macro_f1"),
            "lofo_mean_auc": bin_lofo.get("mean_auc"),
            "lofo_mean_f1": bin_lofo.get("mean_f1"),
            "lofo_pass_rate_auc_gt_0.6": bin_lofo.get("pass_rate_auc_gt_0.6"),
        })

    # Answer-orthogonal full constraint field.
    full_feats = fsets["M6_full_constraint_field"]
    m7_bin = eval_group_binary(df, full_feats, "graph_id", "gen_E", residualize=True, answer_features=answer_features)
    m7_mech = eval_group_multiclass(df, full_feats, "graph_id", "mechanism", residualize=True, answer_features=answer_features)
    m7_lofo = eval_lofo_binary(df, full_feats, "family", "gen_E", residualize=True, answer_features=answer_features)
    answer_orth = {
        "model": "M7_answer_orthogonal_full",
        "n_features": len(full_feats),
        "binary_graph_auc": m7_bin.get("auc"),
        "binary_graph_acc": m7_bin.get("acc"),
        "binary_graph_f1": m7_bin.get("f1"),
        "mech_graph_acc": m7_mech.get("acc"),
        "mech_graph_macro_f1": m7_mech.get("macro_f1"),
        "lofo_mean_auc": m7_lofo.get("mean_auc"),
        "lofo_mean_f1": m7_lofo.get("mean_f1"),
        "lofo_pass_rate_auc_gt_0.6": m7_lofo.get("pass_rate_auc_gt_0.6"),
    }
    model_rows.append(answer_orth)

    model_df = pd.DataFrame(model_rows)
    model_df.to_csv(out_dir / "cft1_model_summary.csv", index=False, encoding="utf-8-sig")

    same_answer = eval_same_answer_split(df, full_feats)
    with open(out_dir / "cft1_same_answer_split_summary.json", "w", encoding="utf-8") as f:
        json.dump(same_answer, f, ensure_ascii=False, indent=2)

    wcr = wrong_closure_random_separability(df, full_feats)

    # Detailed CV summaries for main models.
    cv_summary = {
        "M6_lofo_details": m7_lofo.get("details", []),
        "same_answer_split": same_answer,
        "wrong_closure_random": wcr,
    }
    with open(out_dir / "cft1_cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(cv_summary, f, ensure_ascii=False, indent=2)

    # Verdict logic.
    def get_row(name: str) -> Dict[str, Any]:
        r = model_df[model_df["model"] == name]
        return r.iloc[0].to_dict() if len(r) else {}

    m0 = get_row("M0_answer_margin_20_25")
    m6 = get_row("M6_full_constraint_field")
    m7 = get_row("M7_answer_orthogonal_full")

    m0_auc = float(m0.get("binary_graph_auc", float("nan")))
    m6_auc = float(m6.get("binary_graph_auc", float("nan")))
    m0_mech = float(m0.get("mech_graph_macro_f1", float("nan")))
    m7_mech = float(m7.get("mech_graph_macro_f1", float("nan")))
    m7_lofo_pass = float(m7.get("lofo_pass_rate_auc_gt_0.6", 0.0))
    same_f1 = float(same_answer.get("same_answer_macro_f1_mean", float("nan")))
    wcr_auc = float(wcr.get("auc", float("nan")))

    verdict = "UNDETERMINED"
    reasons = []
    if not (m6_auc > m0_auc + 0.01):
        verdict = "FAIL_NO_GAIN_OVER_ANSWER_MARGIN"
        reasons.append("M6 full constraint field did not improve GenError AUC over M0 answer margin by >0.01.")
    elif not (m7_mech > m0_mech + 0.03):
        verdict = "PARTIAL_RISK_SCORE_OR_ANSWER_CONTAMINATION"
        reasons.append("Answer-orthogonal M7 did not improve mechanism macro-F1 over M0 by >0.03.")
    elif m7_lofo_pass < 0.5:
        verdict = "PARTIAL_TEMPLATE_DEPENDENT"
        reasons.append("M7 LOFO pass rate below 0.5; likely template/family dependence.")
    elif math.isnan(same_f1) or same_f1 < 0.35:
        verdict = "PASS_STRONG_NO_PROJECTION_RESIDUAL"
        reasons.append("Constraint proxy passes main checks but same-answer mechanism split is weak/missing.")
    elif not math.isnan(wcr_auc) and wcr_auc > 0.70:
        verdict = "PASS_MILESTONE"
        reasons.append("M7 survives answer-orthogonal test; same-answer split exists; closure vs random separable.")
    else:
        verdict = "PASS_STRONG"
        reasons.append("M7 survives answer-orthogonal and cross-factor checks, but milestone conditions are incomplete.")

    verdict_obj = {
        "verdict": verdict,
        "reasons": reasons,
        "M0_answer_auc": m0_auc,
        "M6_full_auc": m6_auc,
        "M0_mech_macro_f1": m0_mech,
        "M7_mech_macro_f1": m7_mech,
        "M7_lofo_pass_rate_auc_gt_0.6": m7_lofo_pass,
        "same_answer_macro_f1_mean": same_f1,
        "wrong_closure_random_auc": wcr_auc,
        "feature_set_sizes": {k: len(v) for k, v in fsets.items()},
        "n_rows": int(len(df)),
        "n_graphs": int(df["graph_id"].nunique()),
        "conditions": sorted(df["condition"].unique().tolist()),
    }
    with open(out_dir / "cft1_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict_obj, f, ensure_ascii=False, indent=2)

    return verdict_obj


# ============================================================
# 6. Main
# ============================================================


def main() -> None:
    set_seed(SEED)
    out_dir = ensure_dir(OUT_DIR)
    print(f"[CFT-1] Output directory: {out_dir}")
    model_path = resolve_model_path()
    print(f"[CFT-1] Loading model: {model_path}")
    print(f"[CFT-1] Device={DEVICE}, dtype={DTYPE}")

    rows = build_dataset()
    print(f"[CFT-1] Dataset rows: {len(rows)}")
    pd.DataFrame([asdict(r) for r in rows]).to_csv(out_dir / "cft1_dataset.csv", index=False, encoding="utf-8-sig")

    probe = ModelProbe(model_path)
    print(f"[CFT-1] Model layers reported: {probe.n_layers}; hidden_states will include embedding + layers.")

    feature_rows = []
    t0 = time.time()
    for i, ex in enumerate(rows):
        try:
            feats = extract_one(probe, ex)
            feature_rows.append(feats)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[CFT-1][ERROR] row_id={ex.row_id}: {repr(e)}")
            raise
        if (i + 1) % 20 == 0 or (i + 1) == len(rows):
            elapsed = time.time() - t0
            print(f"[CFT-1] Extracted {i+1}/{len(rows)} rows, elapsed={elapsed:.1f}s")

    df = pd.DataFrame(feature_rows)
    df = add_clean_relative_deltas(df)
    df.to_csv(out_dir / "cft1_features.csv", index=False, encoding="utf-8-sig")
    print(f"[CFT-1] Saved features: {out_dir / 'cft1_features.csv'}")

    verdict = run_evaluations(df, out_dir)
    print("[CFT-1] Verdict:")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    print(f"[CFT-1] Done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
