# -*- coding: utf-8 -*-
r"""
SEM-6H.2b: True MemoryUnit Tensor Live Write Replay

Purpose
-------
SEM-6H.2 failed/inconclusive because the first live script used a synthetic latent
memory prompt:

    candidate_concept + candidate_operator -> forward -> hidden vector

That is not the real SEM MemoryUnit:
    MemoryUnit_F = (mu_shape, mu_commit, residual_modes, Policy_F)

SEM-6H.2b fixes the write side:
    - load real MemoryUnit tensors if available
    - write mu_shape to L7-L19
    - write mu_commit to L23-L25
    - use small alpha values: 0.01 / 0.02 / 0.05

If no real MemoryUnit bank exists, the script fails loudly with the exact expected
schema instead of silently falling back to synthetic prompt vectors.

Default paths
-------------
Root:
    C:\Users\ZH\Desktop\AGI\python_script

Replay plan:
    C:\Users\ZH\Desktop\AGI\python_script\sem6h2_outputs\sem6h2_replay_plan.csv

Output:
    C:\Users\ZH\Desktop\AGI\python_script\sem6h2_outputs\sem6h2b_live_results.csv

Model:
    D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main

Run
---
conda activate dhrf_4080s
cd C:\Users\ZH\Desktop\AGI\python_script

# First inspect / auto-find bank:
python GPT_sem6h2b_true_memoryunit_live_write.py --inspect_only

# Smoke test:
python GPT_sem6h2b_true_memoryunit_live_write.py --max_rows 120 --max_null_reps 2

# Full selected families only, small null:
python GPT_sem6h2b_true_memoryunit_live_write.py --max_rows 0 --max_null_reps 10

Then evaluate:
python GPT_sem6h2_noleak_live_write_replay.py ^
  --candidates sem6h1b_outputs\sem6h1b_scored_candidates.csv ^
  --live_results sem6h2_outputs\sem6h2b_live_results.csv ^
  --out_dir sem6h2_outputs

Supported MemoryUnit bank schemas
---------------------------------

A) NPZ preferred:

Required:
    family_ids or families or family

Option 1:
    mu_shape:  np.ndarray [n_family, dim] or [n_family, n_shape_layers, dim]
    mu_commit: np.ndarray [n_family, dim] or [n_family, n_commit_layers, dim]

Option 2:
    shape_l7, shape_l8, ..., shape_l19
    commit_l23, commit_l24, commit_l25
    each array [n_family, dim]

B) CSV:

Required:
    family

Option 1:
    mu_shape_0 ... mu_shape_D
    mu_commit_0 ... mu_commit_D

Option 2:
    shape_l7_d0 ... shape_l7_dD
    ...
    commit_l23_d0 ... commit_l23_dD

Important:
    Vector dimension must equal model hidden_size.
    PCA vectors cannot be injected unless you also provide a decoder; this script does
    not silently project PCA back into hidden space.
"""

import argparse
import gc
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================
# Hardcoded local configuration
# =============================

DEFAULT_ROOT = r"C:\Users\ZH\Desktop\AGI\python_script"
DEFAULT_MODEL_PATH = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"
DEFAULT_SEM6H2_DIR = rf"{DEFAULT_ROOT}\sem6h2_outputs"

DEFAULT_REPLAY_PLAN = rf"{DEFAULT_SEM6H2_DIR}\sem6h2_replay_plan.csv"
DEFAULT_OUT_CSV = rf"{DEFAULT_SEM6H2_DIR}\sem6h2b_live_results.csv"
DEFAULT_PROGRESS_JSON = rf"{DEFAULT_SEM6H2_DIR}\sem6h2b_live_progress.json"

DEFAULT_SHAPE_LAYERS = "7,8,9,10,11,12,13,14,15,16,17,18,19"
DEFAULT_COMMIT_LAYERS = "23,24,25"
DEFAULT_ALPHAS = "0.01,0.02,0.05"

SEED = 20260606


BANK_CANDIDATES = [
    rf"{DEFAULT_ROOT}\sem5c_outputs\sem5c_memory_bank.npz",
    rf"{DEFAULT_ROOT}\sem5c_outputs\sem5c_memory_units.npz",
    rf"{DEFAULT_ROOT}\sem5c_outputs\memory_units.npz",
    rf"{DEFAULT_ROOT}\sem5c2_outputs\sem5c2_memory_bank.npz",
    rf"{DEFAULT_ROOT}\sem5c2_outputs\memory_units.npz",
    rf"{DEFAULT_ROOT}\sem5c_outputs\sem5c_memory_bank.csv",
    rf"{DEFAULT_ROOT}\sem5c_outputs\sem5c_memory_units.csv",
    rf"{DEFAULT_ROOT}\sem5c2_outputs\sem5c2_memory_bank.csv",
    rf"{DEFAULT_ROOT}\sem5c2_outputs\memory_units.csv",
    rf"{DEFAULT_ROOT}\sem6_outputs\memory_units.npz",
    rf"{DEFAULT_ROOT}\sem6_outputs\memory_units.csv",
]


OPERATOR_TEMPLATES = {
    "Definition": "What is {concept}? Give a concise definition.",
    "MechanismExplanation": "Explain the mechanism behind {concept}.",
    "CausalExplanation": "Explain what causes {concept} and what effects it produces.",
    "RelationMapping": "Map the key relationships involving {concept}.",
    "PropertyDescription": "Describe the key properties of {concept}.",
    "Comparison": "Compare {concept} with closely related concepts.",
    "CounterExample": "Find a counterexample or limitation related to {concept}.",
    "InvariantSearch": "Find the invariant structure behind {concept}.",
    "ClosureCheck": "Check whether the explanation of {concept} is internally closed and consistent.",
    "PolicySelection": "Select the best strategy for reasoning about {concept}.",
}


# =============================
# Utility
# =============================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_csv_list(s: str, cast=float):
    return [cast(x.strip()) for x in str(s).split(",") if x.strip()]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("Could not locate transformer layers on this model.")


def operator_description(op: str) -> str:
    desc = {
        "Definition": "define the concept and state its core meaning",
        "MechanismExplanation": "explain internal mechanism and process",
        "CausalExplanation": "explain causes, effects, and causal chain",
        "RelationMapping": "map relations between concepts, entities, and roles",
        "PropertyDescription": "describe attributes, properties, and characteristic features",
        "Comparison": "compare similarities and differences",
        "CounterExample": "test the claim through counterexamples",
        "InvariantSearch": "search for invariant structure across cases",
        "ClosureCheck": "check relation closure and internal consistency",
        "PolicySelection": "select an appropriate reasoning/control policy",
    }
    return desc.get(str(op), f"use the reasoning operator named {op}")


def build_task_request(concept: str, operator: str) -> str:
    tmpl = OPERATOR_TEMPLATES.get(str(operator))
    if tmpl:
        return tmpl.format(concept=concept)
    return f"Reason about {concept} using the operator {operator}."


def build_classifier_prompt(concept: str, target_operator: str, operator_to_letter: Dict[str, str]) -> str:
    request = build_task_request(concept, target_operator)
    options = "\n".join(
        f"{letter}. {op}: {operator_description(op)}"
        for op, letter in operator_to_letter.items()
    )
    return (
        "You are classifying the reasoning operator required by a request.\n"
        "Choose exactly one option letter.\n\n"
        f"Options:\n{options}\n\n"
        f"Request:\n{request}\n\n"
        "Answer with one option letter only:"
    )


def apply_chat(tokenizer, user_text: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return user_text + "\nAssistant:"


def first_token_id_for_label(tokenizer, letter: str) -> int:
    variants = [letter, " " + letter, "\n" + letter]
    best = None
    for v in variants:
        ids = tokenizer(v, add_special_tokens=False).input_ids
        if ids and (best is None or len(ids) < len(best)):
            best = ids
    if not best:
        raise ValueError(f"Could not tokenize label {letter!r}")
    return int(best[0])


def downsample_null_reps(df: pd.DataFrame, max_null_reps: int) -> pd.DataFrame:
    if max_null_reps <= 0:
        return df
    parts = []
    for fam, sub in df.groupby("_selection_family", sort=False):
        if fam not in {"random_pool", "shuffle_learned_Q"}:
            parts.append(sub)
        else:
            vals = sorted(sub["_selection"].dropna().unique().tolist())
            keep = set(vals[:max_null_reps])
            parts.append(sub[sub["_selection"].isin(keep)])
    return pd.concat(parts, ignore_index=True)


# =============================
# MemoryUnit bank loader
# =============================

class MemoryUnitBank:
    """
    family -> shape layer vectors / commit layer vectors
    """

    def __init__(
        self,
        shape: Dict[str, Dict[int, np.ndarray]],
        commit: Dict[str, Dict[int, np.ndarray]],
        path: str,
        source_info: Dict[str, Any],
    ):
        self.shape = shape
        self.commit = commit
        self.path = path
        self.source_info = source_info

    @property
    def families(self):
        return sorted(set(self.shape.keys()) | set(self.commit.keys()))

    def has_family(self, fam: str) -> bool:
        return fam in self.shape or fam in self.commit

    def get_vectors(
        self,
        fam: str,
        shape_layers: List[int],
        commit_layers: List[int],
        device: str,
        dtype: torch.dtype,
    ) -> Dict[int, torch.Tensor]:
        if fam not in self.shape and fam not in self.commit:
            raise KeyError(f"Family {fam!r} not found in memory bank.")

        out = {}

        if fam in self.shape:
            for l in shape_layers:
                vec = self._get_layer_vec(self.shape[fam], l, default_kind="shape")
                if vec is not None:
                    out[l] = torch.tensor(vec, device=device, dtype=dtype)

        if fam in self.commit:
            for l in commit_layers:
                vec = self._get_layer_vec(self.commit[fam], l, default_kind="commit")
                if vec is not None:
                    out[l] = torch.tensor(vec, device=device, dtype=dtype)

        if not out:
            raise KeyError(f"No shape/commit vectors available for family {fam!r}.")
        return out

    @staticmethod
    def _get_layer_vec(layer_map: Dict[int, np.ndarray], layer: int, default_kind: str):
        if layer in layer_map:
            return layer_map[layer]
        # layer -1 means shared vector across all layers.
        if -1 in layer_map:
            return layer_map[-1]
        return None

    def infer_dim(self) -> Optional[int]:
        for d in [self.shape, self.commit]:
            for fam, layer_map in d.items():
                for layer, vec in layer_map.items():
                    return int(np.asarray(vec).shape[-1])
        return None


def _decode_family_ids(arr) -> List[str]:
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def _find_family_key(keys: List[str]) -> Optional[str]:
    for k in ["family_ids", "families", "family", "seed_family", "family_id"]:
        if k in keys:
            return k
    for k in keys:
        if "family" in norm(k):
            return k
    return None


def _extract_npz_layer_arrays(npz, families: List[str], prefix_patterns: List[str]) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Extract keys like shape_l7, mu_shape_l7, commit_l23, etc.
    """
    result = {fam: {} for fam in families}
    keys = list(npz.keys())
    for key in keys:
        nk = norm(key)
        matched = False
        kind_layer = None
        for pat in prefix_patterns:
            # pat examples: shape, mu_shape
            # accept shape_l7 / shape_layer_7 / mu_shape_l7
            m = re.search(rf"{pat}.*(?:l|layer)_?(\d+)", nk)
            if m:
                kind_layer = int(m.group(1))
                matched = True
                break
        if not matched:
            continue
        arr = np.asarray(npz[key])
        if arr.ndim != 2 or arr.shape[0] != len(families):
            continue
        for i, fam in enumerate(families):
            result[fam][kind_layer] = arr[i].astype(np.float32)
    return result


def load_npz_bank(path: Path) -> MemoryUnitBank:
    npz = np.load(path, allow_pickle=True)
    keys = list(npz.keys())
    fam_key = _find_family_key(keys)
    if fam_key is None:
        raise ValueError(f"NPZ memory bank missing family ids. Keys={keys}")
    families = _decode_family_ids(npz[fam_key])

    shape: Dict[str, Dict[int, np.ndarray]] = {fam: {} for fam in families}
    commit: Dict[str, Dict[int, np.ndarray]] = {fam: {} for fam in families}

    # Direct mu_shape / mu_commit arrays.
    shape_key = None
    commit_key = None
    for k in keys:
        nk = norm(k)
        if nk in {"mu_shape", "shape_mu", "shape", "shape_center", "mu_f_shape"}:
            shape_key = k
        if nk in {"mu_commit", "commit_mu", "commit", "commit_center", "mu_f_commit"}:
            commit_key = k

    def assign_direct(arr_key: str, dest: Dict[str, Dict[int, np.ndarray]], kind: str):
        arr = np.asarray(npz[arr_key])
        if arr.shape[0] != len(families):
            raise ValueError(f"{arr_key} first dimension {arr.shape[0]} != n_families {len(families)}")
        if arr.ndim == 2:
            for i, fam in enumerate(families):
                dest[fam][-1] = arr[i].astype(np.float32)
        elif arr.ndim == 3:
            # Assign pseudo layer order: caller layer list order will use nearest if exact absent impossible.
            # Store as -1000-indexed; later not exact. Instead duplicate mean vector as fallback.
            for i, fam in enumerate(families):
                # Store shared mean as fallback, and also index by 0..n_layer-1 for diagnostics.
                dest[fam][-1] = arr[i].mean(axis=0).astype(np.float32)
                for j in range(arr.shape[1]):
                    dest[fam][j] = arr[i, j].astype(np.float32)
        else:
            raise ValueError(f"{arr_key} must be 2D or 3D, got shape={arr.shape}")

    if shape_key:
        assign_direct(shape_key, shape, "shape")
    if commit_key:
        assign_direct(commit_key, commit, "commit")

    # Layer-specific arrays override / complement.
    layer_shape = _extract_npz_layer_arrays(npz, families, ["shape", "mu_shape"])
    layer_commit = _extract_npz_layer_arrays(npz, families, ["commit", "mu_commit"])
    for fam in families:
        shape[fam].update(layer_shape.get(fam, {}))
        commit[fam].update(layer_commit.get(fam, {}))

    # Remove empty families from each dict.
    shape = {k: v for k, v in shape.items() if v}
    commit = {k: v for k, v in commit.items() if v}

    if not shape and not commit:
        raise ValueError(
            "Could not find mu_shape/mu_commit or layer-specific shape/commit arrays in NPZ. "
            f"Keys={keys}"
        )

    return MemoryUnitBank(shape, commit, str(path), {"type": "npz", "keys": keys, "family_key": fam_key})


def _vector_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = []
    p = norm(prefix)
    for c in df.columns:
        nc = norm(c)
        # mu_shape_0, mu_shape_d0, shape_0
        if re.match(rf"^{p}_(?:d)?\d+$", nc):
            cols.append(c)
    def order_key(c):
        m = re.search(r"(\d+)$", norm(c))
        return int(m.group(1)) if m else 0
    return sorted(cols, key=order_key)


def _layer_vector_cols(df: pd.DataFrame, kind: str, layer: int) -> List[str]:
    cols = []
    for c in df.columns:
        nc = norm(c)
        # shape_l7_d0 / mu_shape_l7_d0 / commit_layer_23_d0
        if kind in nc and re.search(rf"(?:l|layer)_?{layer}(?:_|$)", nc) and re.search(r"(?:d)?\d+$", nc):
            cols.append(c)
    def order_key(c):
        m = re.search(r"(\d+)$", norm(c))
        return int(m.group(1)) if m else 0
    return sorted(cols, key=order_key)


def load_csv_bank(path: Path, shape_layers: List[int], commit_layers: List[int]) -> MemoryUnitBank:
    df = pd.read_csv(path)
    fam_col = None
    for c in df.columns:
        if norm(c) in {"family", "family_id", "seed_family"}:
            fam_col = c
            break
    if fam_col is None:
        raise ValueError(f"CSV bank missing family column. Columns={list(df.columns)}")

    shape: Dict[str, Dict[int, np.ndarray]] = {}
    commit: Dict[str, Dict[int, np.ndarray]] = {}

    shared_shape_cols = []
    for p in ["mu_shape", "shape_mu", "shape_center", "shape"]:
        shared_shape_cols = _vector_cols(df, p)
        if shared_shape_cols:
            break

    shared_commit_cols = []
    for p in ["mu_commit", "commit_mu", "commit_center", "commit"]:
        shared_commit_cols = _vector_cols(df, p)
        if shared_commit_cols:
            break

    for _, row in df.iterrows():
        fam = str(row[fam_col])
        s_map = {}
        c_map = {}

        if shared_shape_cols:
            s_map[-1] = row[shared_shape_cols].to_numpy(dtype=np.float32)
        if shared_commit_cols:
            c_map[-1] = row[shared_commit_cols].to_numpy(dtype=np.float32)

        for l in shape_layers:
            cols = _layer_vector_cols(df, "shape", l)
            if cols:
                s_map[l] = row[cols].to_numpy(dtype=np.float32)

        for l in commit_layers:
            cols = _layer_vector_cols(df, "commit", l)
            if cols:
                c_map[l] = row[cols].to_numpy(dtype=np.float32)

        if s_map:
            shape[fam] = s_map
        if c_map:
            commit[fam] = c_map

    if not shape and not commit:
        raise ValueError(
            "CSV bank has family column but no supported vector columns. "
            "Expected mu_shape_* / mu_commit_* or shape_l7_d* / commit_l23_d*."
        )

    return MemoryUnitBank(shape, commit, str(path), {"type": "csv", "columns": list(df.columns), "family_col": fam_col})


def find_memory_bank(user_path: Optional[str], shape_layers: List[int], commit_layers: List[int]) -> MemoryUnitBank:
    candidates = []
    if user_path:
        candidates.append(user_path)
    candidates += BANK_CANDIDATES

    tried = []
    for p in candidates:
        path = Path(p)
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".npz":
                bank = load_npz_bank(path)
            elif path.suffix.lower() == ".csv":
                bank = load_csv_bank(path, shape_layers, commit_layers)
            else:
                continue
            print(f"[bank] loaded: {path}")
            print(f"[bank] families={len(bank.families)} dim={bank.infer_dim()} source={bank.source_info.get('type')}")
            return bank
        except Exception as e:
            tried.append({"path": str(path), "error": str(e)})

    msg = {
        "error": "No usable MemoryUnit bank found.",
        "searched_existing_or_user_paths": [p for p in candidates if Path(p).exists()],
        "tried_errors": tried,
        "expected_schema": {
            "npz": {
                "required": ["family_ids/families/family"],
                "option_1": ["mu_shape [n, hidden] or [n, layers, hidden]", "mu_commit [n, hidden] or [n, layers, hidden]"],
                "option_2": ["shape_l7...shape_l19 arrays [n, hidden]", "commit_l23...commit_l25 arrays [n, hidden]"],
            },
            "csv": {
                "required": ["family"],
                "option_1": ["mu_shape_0...mu_shape_D", "mu_commit_0...mu_commit_D"],
                "option_2": ["shape_l7_d0...shape_l19_dD", "commit_l23_d0...commit_l25_dD"],
            },
            "note": "Vector dimension must equal model hidden_size. PCA vectors are not enough unless decoded back to hidden dimension.",
        },
    }
    raise FileNotFoundError(json.dumps(msg, ensure_ascii=False, indent=2))


# =============================
# Model scoring + hooks
# =============================

def register_write_hooks(
    model,
    memory_vecs: Dict[int, torch.Tensor],
    alpha: float,
):
    layers = get_layers(model)
    handles = []

    def make_hook(layer_idx: int):
        def hook(module, inputs, output):
            if layer_idx not in memory_vecs:
                return output
            mem = memory_vecs[layer_idx]

            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None

            h = hidden.clone()
            mem2 = mem.to(device=h.device, dtype=h.dtype).view(1, 1, -1)
            if mem2.shape[-1] != h.shape[-1]:
                raise ValueError(
                    f"Memory vector dim {mem2.shape[-1]} != hidden dim {h.shape[-1]} at layer {layer_idx}. "
                    "Your MemoryUnit bank may contain PCA vectors rather than full hidden vectors."
                )
            h[:, -1:, :] = (1.0 - alpha) * h[:, -1:, :] + alpha * mem2

            if rest is None:
                return h
            return (h,) + rest
        return hook

    for layer_idx in memory_vecs:
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"Layer index {layer_idx} out of range 0..{len(layers)-1}")
        handles.append(layers[layer_idx].register_forward_hook(make_hook(layer_idx)))
    return handles


@torch.no_grad()
def score_prompt_letters(
    model,
    tokenizer,
    prompt_text: str,
    letter_to_token_id: Dict[str, int],
    device: str,
    max_len: int,
    memory_vecs: Optional[Dict[int, torch.Tensor]] = None,
    alpha: float = 0.0,
) -> Dict[str, float]:
    prompt = apply_chat(tokenizer, prompt_text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(device)

    handles = []
    try:
        if memory_vecs is not None and alpha > 0:
            handles = register_write_hooks(model, memory_vecs, alpha)
        outputs = model(**inputs, use_cache=False)
        logits = outputs.logits[0, -1, :].float()
        logp = torch.log_softmax(logits, dim=-1)
        return {letter: float(logp[token_id].detach().cpu()) for letter, token_id in letter_to_token_id.items()}
    finally:
        for h in handles:
            h.remove()
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def compute_margin_and_rank(scores: Dict[str, float], correct_letter: str) -> Tuple[float, int, float]:
    correct = scores[correct_letter]
    others = [v for k, v in scores.items() if k != correct_letter]
    best_other = max(others) if others else float("-inf")
    margin = correct - best_other
    sorted_letters = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    rank = sorted_letters.index(correct_letter) + 1
    gap_to_top = correct - scores[sorted_letters[0]]
    return float(margin), int(rank), float(gap_to_top)


def print_bank_inspection(bank: MemoryUnitBank, plan: pd.DataFrame):
    print("\n========== MemoryUnit Bank Inspection ==========")
    print("path:", bank.path)
    print("families:", len(bank.families))
    print("dim:", bank.infer_dim())
    print("source:", bank.source_info)

    plan_fams = set(plan["candidate_family"].astype(str).unique())
    bank_fams = set(bank.families)
    missing = sorted(plan_fams - bank_fams)
    present = sorted(plan_fams & bank_fams)
    print("candidate families in plan:", len(plan_fams))
    print("present in bank:", len(present))
    print("missing from bank:", len(missing))
    if missing[:20]:
        print("missing examples:", missing[:20])

    # Layer availability.
    for fam in present[:3]:
        print(f"family={fam}: shape_layers={sorted(bank.shape.get(fam, {}).keys())[:20]} commit_layers={sorted(bank.commit.get(fam, {}).keys())[:20]}")


# =============================
# Main
# =============================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--replay_plan", default=DEFAULT_REPLAY_PLAN)
    parser.add_argument("--memory_bank", default="", help="Optional explicit .npz/.csv MemoryUnit bank path.")
    parser.add_argument("--out_csv", default=DEFAULT_OUT_CSV)
    parser.add_argument("--progress_json", default=DEFAULT_PROGRESS_JSON)

    parser.add_argument("--shape_layers", default=DEFAULT_SHAPE_LAYERS)
    parser.add_argument("--commit_layers", default=DEFAULT_COMMIT_LAYERS)
    parser.add_argument("--alphas", default=DEFAULT_ALPHAS)
    parser.add_argument("--max_len", type=int, default=384)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)

    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--max_null_reps", type=int, default=10)
    parser.add_argument("--selection_families", default="")
    parser.add_argument("--write_mode", default="shape_commit", choices=["shape_commit", "shape_only", "commit_only"])
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--inspect_only", action="store_true", default=False)
    parser.add_argument("--flush_every", type=int, default=20)
    args = parser.parse_args()

    set_seed(SEED)

    shape_layers = [int(x) for x in parse_csv_list(args.shape_layers, int)]
    commit_layers = [int(x) for x in parse_csv_list(args.commit_layers, int)]
    alphas = [float(x) for x in parse_csv_list(args.alphas, float)]

    replay_path = Path(args.replay_plan)
    if not replay_path.exists():
        raise FileNotFoundError(f"Replay plan not found: {replay_path}")

    plan = pd.read_csv(replay_path)
    required = [
        "_selection", "_selection_family", "_group_id",
        "target_concept", "target_operator",
        "candidate_family", "candidate_concept", "candidate_operator",
    ]
    missing = [c for c in required if c not in plan.columns]
    if missing:
        raise ValueError(f"Replay plan missing required columns: {missing}")

    plan = downsample_null_reps(plan, args.max_null_reps)

    if args.selection_families.strip():
        keep = {x.strip() for x in args.selection_families.split(",") if x.strip()}
        plan = plan[plan["_selection_family"].isin(keep)].copy()

    if args.max_rows and args.max_rows > 0:
        plan = plan.head(args.max_rows).copy()

    if "_live_row_id" not in plan.columns:
        plan["_live_row_id"] = np.arange(len(plan), dtype=int)

    # Load / inspect real bank before model.
    bank = find_memory_bank(args.memory_bank.strip() or None, shape_layers, commit_layers)
    print_bank_inspection(bank, plan)

    if args.inspect_only:
        print("\n[inspect_only] Done. If present/missing counts look right, run without --inspect_only.")
        return

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = Path(args.progress_json)
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model.
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; using CPU.")
        device = "cpu"
    torch_dtype = torch.float16 if args.dtype == "float16" and device == "cuda" else torch.float32

    print(f"Loading tokenizer/model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if device != "cuda":
        model.to(device)
    model.eval()

    hidden_size = int(model.config.hidden_size)
    bank_dim = bank.infer_dim()
    if bank_dim != hidden_size:
        raise ValueError(
            f"MemoryUnit vector dim={bank_dim}, model hidden_size={hidden_size}. "
            "This usually means the bank contains PCA/compressed vectors rather than full hidden vectors. "
            "For SEM-6H.2b, export full hidden-dim mu_shape/mu_commit tensors or provide a decoder."
        )

    # Operator label mapping.
    operator_list = sorted(set(plan["target_operator"].astype(str).tolist()) | set(plan["candidate_operator"].astype(str).tolist()))
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if len(operator_list) > len(letters):
        raise ValueError(f"Too many operators for label set: {len(operator_list)}")
    operator_to_letter = {op: letters[i] for i, op in enumerate(operator_list)}
    letter_to_token_id = {
        letter: first_token_id_for_label(tokenizer, letter)
        for letter in operator_to_letter.values()
    }

    print("Operators:")
    for op, letter in operator_to_letter.items():
        print(f"  {letter}: {op}")

    # Resume.
    done_ids = set()
    results = []
    if args.resume and out_path.exists():
        old = pd.read_csv(out_path)
        if "_live_row_id" in old.columns:
            done_ids = set(pd.to_numeric(old["_live_row_id"], errors="coerce").dropna().astype(int).tolist())
            results = old.to_dict("records")
            print(f"[resume] loaded {len(done_ids)} completed rows from {out_path}")

    baseline_cache: Dict[Tuple[str, str], Tuple[Dict[str, float], float, int, float]] = {}
    mem_cache: Dict[str, Dict[int, torch.Tensor]] = {}

    rows = plan.to_dict("records")
    total = len(rows)
    print(f"Running rows={total}, already_done={len(done_ids)} write_mode={args.write_mode}")
    print(f"shape_layers={shape_layers}; commit_layers={commit_layers}; alphas={alphas}")

    for idx, row in enumerate(rows):
        live_id = int(row["_live_row_id"])
        if live_id in done_ids:
            continue

        target_concept = str(row["target_concept"])
        target_operator = str(row["target_operator"])
        candidate_family = str(row["candidate_family"])

        correct_letter = operator_to_letter[target_operator]
        target_key = (target_concept, target_operator)

        if candidate_family not in mem_cache:
            use_shape_layers = shape_layers if args.write_mode in {"shape_commit", "shape_only"} else []
            use_commit_layers = commit_layers if args.write_mode in {"shape_commit", "commit_only"} else []
            mem_cache[candidate_family] = bank.get_vectors(
                candidate_family,
                use_shape_layers,
                use_commit_layers,
                device=device,
                dtype=torch_dtype,
            )

        mem_vecs = mem_cache[candidate_family]

        if target_key not in baseline_cache:
            prompt = build_classifier_prompt(target_concept, target_operator, operator_to_letter)
            base_scores = score_prompt_letters(
                model, tokenizer, prompt, letter_to_token_id,
                device=device, max_len=args.max_len,
            )
            base_margin, base_rank, base_gap = compute_margin_and_rank(base_scores, correct_letter)
            baseline_cache[target_key] = (base_scores, base_margin, base_rank, base_gap)
        else:
            base_scores, base_margin, base_rank, base_gap = baseline_cache[target_key]

        prompt = build_classifier_prompt(target_concept, target_operator, operator_to_letter)
        out = dict(row)
        out["write_mode"] = args.write_mode
        out["correct_letter"] = correct_letter
        out["base_margin"] = base_margin
        out["base_rank"] = base_rank
        out["base_logprob_gap_to_top"] = base_gap
        out["operator_options_json"] = json.dumps(operator_to_letter, ensure_ascii=False)
        out["memory_bank_path"] = bank.path

        for alpha in alphas:
            scores = score_prompt_letters(
                model, tokenizer, prompt, letter_to_token_id,
                device=device, max_len=args.max_len,
                memory_vecs=mem_vecs,
                alpha=alpha,
            )
            margin, rank, gap = compute_margin_and_rank(scores, correct_letter)
            tag = str(alpha).replace(".", "_")

            out[f"live_margin_alpha_{tag}"] = margin
            out[f"live_margin_gain_alpha_{tag}"] = margin - base_margin
            out[f"live_rank_alpha_{tag}"] = rank
            out[f"live_rank_improvement_alpha_{tag}"] = base_rank - rank
            out[f"live_logprob_gap_to_top_alpha_{tag}"] = gap
            out[f"live_logprob_gap_gain_alpha_{tag}"] = gap - base_gap

            # Evaluator-compatible aliases.
            out[f"live_gain_alpha_{alpha}"] = margin - base_margin
            out[f"live_rank_improvement_alpha_{alpha}"] = base_rank - rank

        main_alpha = 0.05 if 0.05 in alphas else alphas[-1]
        main_tag = str(main_alpha).replace(".", "_")
        out["live_margin_gain"] = out[f"live_margin_gain_alpha_{main_tag}"]
        out["live_rank_improvement"] = out[f"live_rank_improvement_alpha_{main_tag}"]
        out["trajectory_improvement"] = out["live_margin_gain"]

        results.append(out)

        if (len(results) % args.flush_every == 0) or (idx == total - 1):
            pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")
            progress = {
                "completed_rows": len(results),
                "total_planned_rows_after_filter": total,
                "output": str(out_path),
                "last_live_row_id": live_id,
                "shape_layers": shape_layers,
                "commit_layers": commit_layers,
                "alphas": alphas,
                "write_mode": args.write_mode,
                "memory_bank": bank.path,
                "note": "SEM-6H.2b true MemoryUnit tensor live write replay.",
            }
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
            print(f"[progress] {len(results)}/{total} rows -> {out_path}")

        if idx % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[DONE] SEM-6H.2b live results written to:\n{out_path}")
    print("\nEvaluate:")
    print(
        "python GPT_sem6h2_noleak_live_write_replay.py ^\n"
        "  --candidates sem6h1b_outputs\\sem6h1b_scored_candidates.csv ^\n"
        f"  --live_results {out_path} ^\n"
        "  --out_dir sem6h2_outputs"
    )


if __name__ == "__main__":
    main()
