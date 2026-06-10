# -*- coding: utf-8 -*-
"""
PSA-3A.3: Pair-Delta Mechanism Unfolding/Selection Audit
======================================================

Why this version exists
-----------------------
PSA-3A.0 failed because mechanism labels leaked through surface text.
PSA-3A.1 removed explicit labels, but still used absolute hidden states. It showed:

    H_init  ~ 0.325
    H_mid   ~ 0.403
    H_selection ~ 0.396

So mechanism possibility space exists, but it does not show a simple init-to-selection_compression in absolute-H space.
This suggests the mechanism variable is not an absolute state, but a structural delta:

    condition trajectory - clean trajectory

PSA-3A.3 therefore measures:

    DeltaH_l = H_l(condition) - H_l(clean-matched)

and audits mechanism entropy / Cbit in this pair-delta space.

Default output:
    C:\\Users\\ZH\\Desktop\\AGI\\outputs\\psa3a3_pairdelta_selection_outputs

Run:
    python psa3a3_pairdelta_mechanism_audit.py --n_base 24 --topk 30

Formal run:
    python psa3a3_pairdelta_mechanism_audit.py --n_base 60 --topk 50
"""

import os
import json
import math
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr


MODEL_PATHS = {
    "qwen": r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main",
    "llama": r"D:\model\Llama-3.2-1B-Instruct",
    "gemma": r"D:\model\gemma-2-2b-it",
}

DEFAULT_MODEL_KEY = "qwen"
DEFAULT_OUT_DIR = r"C:\\Users\\ZH\\Desktop\\AGI\\outputs\\psa3a3_pairdelta_selection_outputs"
SEED = 20260607

# We do not expose these labels inside prompts.
MECHANISMS = ["stable", "competition", "closure", "logic", "random"]
MECH_TO_ID = {m: i for i, m in enumerate(MECHANISMS)}


BASE_NAMES = [
    ("Aster", "Boreal", "Cobalt", "Dune"),
    ("Lumen", "Harbor", "Ivory", "Quartz"),
    ("Orion", "Maple", "Cedar", "Nimbus"),
    ("Atlas", "River", "Copper", "Falcon"),
    ("Nova", "Garden", "Silver", "Echo"),
    ("Pixel", "Forest", "Amber", "Delta"),
    ("Sora", "Valley", "Jade", "Mosaic"),
    ("Kite", "Meadow", "Ruby", "Zenith"),
    ("Vega", "Island", "Opal", "Canyon"),
    ("Mira", "Station", "Onyx", "Aurora"),
    ("Raven", "Library", "Topaz", "Lagoon"),
    ("Cairo", "Museum", "Marble", "Ember"),
    ("Sol", "Depot", "Pearl", "Glade"),
    ("Iris", "Temple", "Bronze", "Field"),
    ("Lyra", "Archive", "Coral", "Ridge"),
    ("Echo", "Forum", "Granite", "Selection"),
]

SURFACE_PREFIX = [
    "Study the record and infer the licensed endpoint.",
    "Use the listed relations to determine the permitted endpoint.",
    "Read the symbolic case and decide which endpoint remains valid.",
    "Follow the relation constraints and identify the licensed endpoint.",
    "Inspect the record. Use only the local constraints.",
    "Resolve the endpoint using the relation rules below.",
]


def record_lines(mech: str, a: str, b: str, c: str, d: str) -> List[str]:
    """
    Every condition uses similar vocabulary and clause count.
    The mechanism is hidden in the structural pattern, not in labels.
    """
    if mech == "clean":
        return [
            f"R1: {a} -> {b}.",
            f"R2: {b} -> {c}.",
            f"R3: default composition is active.",
            f"R4: no exception is recorded.",
        ]

    if mech == "stable":
        return [
            f"R1: {a} -> {b}.",
            f"R2: {b} -> {c}.",
            f"R3: default composition is active.",
            f"R4: a redundant note repeats {b} -> {c}.",
        ]

    if mech == "competition":
        return [
            f"R1: {a} -> {b}.",
            f"R2: {b} -> {c}.",
            f"R3: {b} -> {d} is also recorded.",
            f"R4: both endpoint records have equal status.",
        ]

    if mech == "closure":
        return [
            f"R1: {a} -> {b}.",
            f"R2: {b} -> {c}.",
            f"R3: revised rule says {b} -> {d}.",
            f"R4: revised rule overrides the older endpoint.",
        ]

    if mech == "logic":
        return [
            f"R1: if {a} -> {b}, then {b} must be checked.",
            f"R2: if {b} -> {c}, then {a} may reach {c}.",
            f"R3: if contradiction appears, composition is invalid.",
            f"R4: no contradiction is present in this record.",
        ]

    if mech == "random":
        return [
            f"R1: {a} -> {b}.",
            f"R2: {c} mentions {d}.",
            f"R3: the middle rule is absent.",
            f"R4: no valid composition can be licensed.",
        ]

    raise ValueError(f"unknown mechanism {mech}")


def make_prompt(kind: str, base_id: int, surf_id: int, names: Tuple[str, str, str, str]) -> str:
    a, b, c, d = names
    prefix = SURFACE_PREFIX[surf_id % len(SURFACE_PREFIX)]
    lines = record_lines(kind, a, b, c, d)
    # Do not include mechanism labels or options.
    # Ask for endpoint, not mechanism.
    return (
        f"{prefix}\n"
        f"Record: {base_id}-{surf_id}\n"
        + "\n".join(lines)
        + "\nQuestion: Which endpoint is licensed by this record?\n"
        + "Answer with the endpoint name only."
    )


def normalize_prompt_value(x: Any) -> str:
    if isinstance(x, str):
        return x
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    if isinstance(x, (list, tuple)):
        return "\n".join(str(v) for v in x)
    return str(x)


def build_pair_dataset(n_base: int = 24) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        prompt_df: one row per actual prompt, including clean and conditions.
        pair_df: one row per condition-clean pair.
    """
    rows = []
    pairs = []
    rid = 0
    rng = random.Random(SEED)

    for base_id in range(n_base):
        names = BASE_NAMES[base_id % len(BASE_NAMES)]
        surf_id = base_id % len(SURFACE_PREFIX)

        clean_prompt = make_prompt("clean", base_id, surf_id, names)
        clean_row_id = rid
        rows.append({
            "prompt_row_id": rid,
            "base_id": base_id,
            "surface_id": surf_id,
            "kind": "clean",
            "mechanism": "clean",
            "mechanism_id": -1,
            "prompt": clean_prompt,
        })
        rid += 1

        for mech in MECHANISMS:
            cond_prompt = make_prompt(mech, base_id, surf_id, names)
            cond_row_id = rid
            rows.append({
                "prompt_row_id": rid,
                "base_id": base_id,
                "surface_id": surf_id,
                "kind": "condition",
                "mechanism": mech,
                "mechanism_id": MECH_TO_ID[mech],
                "prompt": cond_prompt,
            })
            pairs.append({
                "pair_id": len(pairs),
                "base_id": base_id,
                "surface_id": surf_id,
                "clean_row_id": clean_row_id,
                "cond_row_id": cond_row_id,
                "mechanism": mech,
                "mechanism_id": MECH_TO_ID[mech],
            })
            rid += 1

    prompt_df = pd.DataFrame(rows)
    pair_df = pd.DataFrame(pairs)

    prompt_df["prompt"] = prompt_df["prompt"].map(normalize_prompt_value)
    bad = prompt_df[~prompt_df["prompt"].map(lambda x: isinstance(x, str))]
    if len(bad):
        print("[WARN] non-string prompts after normalization")
        print(bad.head(10).to_string())

    return prompt_df, pair_df


def load_model(model_key: str, device: str, dtype: str, local_files_only: bool = True):
    model_path = MODEL_PATHS[model_key]
    if dtype == "float32":
        torch_dtype = torch.float32
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=local_files_only,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=local_files_only,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=None,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def extract_hidden_topk_features(
    tokenizer,
    model,
    df: pd.DataFrame,
    device: str,
    max_len: int = 256,
    topk: int = 50,
    batch_size: int = 1,
):
    all_hidden = []
    all_entropy = []
    all_spread = []
    all_center = []

    emb_weight = model.get_output_embeddings().weight.detach()

    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        texts = [normalize_prompt_value(x) for x in batch["prompt"].tolist()]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(device)

        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states[1:]
        attention = inputs["attention_mask"]
        last_pos = attention.sum(dim=1) - 1

        layer_hidden = []
        layer_ent = []
        layer_spread = []
        layer_center = []

        for h in hidden_states:
            h_last = h[torch.arange(h.shape[0], device=device), last_pos]
            layer_hidden.append(h_last.detach().float().cpu())

            logits = torch.matmul(h_last, emb_weight.t())
            vals, ids = torch.topk(logits, k=min(topk, logits.shape[-1]), dim=-1)

            probs = torch.softmax(vals.float(), dim=-1)
            entropy = -(probs * torch.log2(probs + 1e-12)).sum(dim=-1)

            tok_emb = emb_weight[ids].detach().float()
            center = tok_emb.mean(dim=1)
            center_norm = F.normalize(center, dim=-1)
            tok_norm = F.normalize(tok_emb, dim=-1)
            cos = (tok_norm * center_norm.unsqueeze(1)).sum(dim=-1)
            spread = (1.0 - cos).mean(dim=-1)

            layer_ent.append(entropy.detach().cpu())
            layer_spread.append(spread.detach().cpu())
            layer_center.append(center.detach().cpu())

        all_hidden.append(torch.stack(layer_hidden, dim=0).transpose(0, 1))
        all_entropy.append(torch.stack(layer_ent, dim=0).transpose(0, 1))
        all_spread.append(torch.stack(layer_spread, dim=0).transpose(0, 1))
        all_center.append(torch.stack(layer_center, dim=0).transpose(0, 1))

        print(f"[extract] {start + len(batch)}/{len(df)} done")

    return {
        "hidden": torch.cat(all_hidden, dim=0).numpy(),
        "topk_entropy": torch.cat(all_entropy, dim=0).numpy(),
        "topk_spread": torch.cat(all_spread, dim=0).numpy(),
        "topk_center": torch.cat(all_center, dim=0).numpy(),
    }


def build_delta_features(feats: Dict[str, np.ndarray], pair_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    clean_idx = pair_df["clean_row_id"].values.astype(int)
    cond_idx = pair_df["cond_row_id"].values.astype(int)

    hidden_delta = feats["hidden"][cond_idx] - feats["hidden"][clean_idx]
    topk_entropy_delta = feats["topk_entropy"][cond_idx] - feats["topk_entropy"][clean_idx]
    topk_spread_delta = feats["topk_spread"][cond_idx] - feats["topk_spread"][clean_idx]
    topk_center_delta = feats["topk_center"][cond_idx] - feats["topk_center"][clean_idx]

    # Also keep condition absolute values for fallback diagnostics.
    return {
        "hidden_delta": hidden_delta,
        "topk_entropy_delta": topk_entropy_delta,
        "topk_spread_delta": topk_spread_delta,
        "topk_center_delta": topk_center_delta,
        "hidden_cond": feats["hidden"][cond_idx],
        "topk_entropy_cond": feats["topk_entropy"][cond_idx],
        "topk_spread_cond": feats["topk_spread"][cond_idx],
        "topk_center_cond": feats["topk_center"][cond_idx],
    }


def entropy_bits(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(prob, 1e-12, 1.0)
    return -(prob * np.log2(prob)).sum(axis=-1)


def soften_probs(p: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 1.0:
        return p
    logits = np.log(np.clip(p, 1e-12, 1.0))
    logits = logits / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    q = np.exp(logits)
    q = q / q.sum(axis=1, keepdims=True)
    return q


def layerwise_oof_mech_probs(
    X_by_layer: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    prob_temperature: float = 1.0,
):
    n, L, d = X_by_layer.shape
    C = len(np.unique(labels))
    probs = np.zeros((n, L, C), dtype=np.float32)
    metrics = []

    n_splits = min(5, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)

    for l in range(L):
        X = X_by_layer[:, l, :]
        clf = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                class_weight="balanced",
                C=0.5,
            )
        )

        p = cross_val_predict(
            clf,
            X,
            labels,
            cv=cv.split(X, labels, groups),
            method="predict_proba",
            n_jobs=None,
        )
        p = soften_probs(p, prob_temperature)

        probs[:, l, :] = p.astype(np.float32)
        pred = p.argmax(axis=1)

        metrics.append({
            "layer": l,
            "mech_acc": float(accuracy_score(labels, pred)),
            "mech_macro_f1": float(f1_score(labels, pred, average="macro")),
            "H_mech_mean": float(entropy_bits(p).mean()),
            "H_mech_std": float(entropy_bits(p).std()),
            "N_eff_mean": float((2.0 ** entropy_bits(p)).mean()),
        })
        print(f"[probe-delta] layer {l:02d}: acc={metrics[-1]['mech_acc']:.3f}, H={metrics[-1]['H_mech_mean']:.3f}")

    return probs, pd.DataFrame(metrics)


def clamp_window(start: int, end: int, L: int) -> List[int]:
    return [i for i in range(start, min(end + 1, L)) if 0 <= i < L]


def window_mean(arr: np.ndarray, idx: List[int]):
    return arr[:, idx].mean(axis=1)


def window_center_mean(center: np.ndarray, idx: List[int]):
    return center[:, idx, :].mean(axis=1)


def cos_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return 1.0 - (an * bn).sum(axis=-1)


def slope_per_sample(series: np.ndarray, idx: List[int]) -> np.ndarray:
    if len(idx) < 2:
        return np.zeros(series.shape[0])
    x = np.asarray(idx, dtype=np.float64)
    x = x - x.mean()
    denom = (x ** 2).sum() + 1e-12
    y = series[:, idx].astype(np.float64)
    y_centered = y - y.mean(axis=1, keepdims=True)
    return (y_centered @ x) / denom


def safe_corr(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None
    return float(pearsonr(x, y)[0])


def build_sample_metrics(pair_df, mech_probs, delta_feats):
    labels = pair_df["mechanism_id"].values.astype(int)
    n, L, C = mech_probs.shape

    init_w = clamp_window(0, 6, L)
    mid_w = clamp_window(7, 19, L)
    selection_w = clamp_window(20, 25, L)
    if len(selection_w) == 0:
        selection_w = clamp_window(max(0, L - 6), L - 2, L)

    H_mech = entropy_bits(mech_probs.reshape(-1, C)).reshape(n, L)

    H_init = window_mean(H_mech, init_w)
    H_mid = window_mean(H_mech, mid_w) if len(mid_w) else H_init
    H_selection = window_mean(H_mech, selection_w)

    # For PSA-3A.3, both directions are informative:
    # selection_compression: H_init - H_selection
    # expansion: H_selection - H_init
    raw_cbit_init_selection = H_init - H_selection
    raw_expansion_init_selection = H_selection - H_init
    raw_cbit_mid_selection = H_mid - H_selection

    true_prob_layer = np.zeros((n, L), dtype=np.float32)
    for i in range(n):
        true_prob_layer[i, :] = mech_probs[i, :, labels[i]]

    Q_mech_selection = window_mean(true_prob_layer, selection_w)
    Q_mech_init = window_mean(true_prob_layer, init_w)

    cbit_eff_init_selection = raw_cbit_init_selection * Q_mech_selection
    expansion_eff_init_selection = raw_expansion_init_selection * Q_mech_selection
    cbit_eff_mid_selection = raw_cbit_mid_selection * Q_mech_selection

    selection_pred = window_mean(mech_probs, selection_w).argmax(axis=1)
    selection_correct = (selection_pred == labels).astype(int)

    te = delta_feats["topk_entropy_delta"]
    ts = delta_feats["topk_spread_delta"]
    tc = delta_feats["topk_center_delta"]
    hd = delta_feats["hidden_delta"]

    init_center = window_center_mean(tc, init_w)
    mid_center = window_center_mean(tc, mid_w) if len(mid_w) else init_center
    selection_center = window_center_mean(tc, selection_w)

    hidden_vel = np.linalg.norm(np.diff(hd, axis=1), axis=-1)
    vel_mid_idx = [i for i in mid_w if i < hidden_vel.shape[1]]
    vel_selection_idx = [i for i in selection_w if i < hidden_vel.shape[1]]

    out = pd.DataFrame({
        "pair_id": pair_df["pair_id"].values,
        "base_id": pair_df["base_id"].values,
        "surface_id": pair_df["surface_id"].values,
        "mechanism": pair_df["mechanism"].values,
        "mechanism_id": labels,

        "H_mech_init": H_init,
        "H_mech_mid": H_mid,
        "H_mech_selection": H_selection,
        "N_eff_init": 2.0 ** H_init,
        "N_eff_mid": 2.0 ** H_mid,
        "N_eff_selection": 2.0 ** H_selection,

        "raw_Cbit_mech_init_selection": raw_cbit_init_selection,
        "raw_Expansion_mech_init_selection": raw_expansion_init_selection,
        "raw_Cbit_mech_mid_selection": raw_cbit_mid_selection,
        "Q_mech_selection": Q_mech_selection,
        "Q_mech_init": Q_mech_init,
        "Cbit_eff_mech_init_selection": cbit_eff_init_selection,
        "Expansion_eff_mech_init_selection": expansion_eff_init_selection,
        "Cbit_eff_mech_mid_selection": cbit_eff_mid_selection,
        "selection_pred_mechanism_id": selection_pred,
        "selection_correct": selection_correct,

        "topk_entropy_delta_init": window_mean(te, init_w),
        "topk_entropy_delta_mid": window_mean(te, mid_w) if len(mid_w) else window_mean(te, init_w),
        "topk_entropy_delta_selection": window_mean(te, selection_w),
        "topk_entropy_delta_drop_init_selection": window_mean(te, init_w) - window_mean(te, selection_w),
        "topk_entropy_delta_slope_mid": slope_per_sample(te, mid_w),
        "topk_entropy_delta_slope_selection": slope_per_sample(te, selection_w),

        "topk_spread_delta_init": window_mean(ts, init_w),
        "topk_spread_delta_mid": window_mean(ts, mid_w) if len(mid_w) else window_mean(ts, init_w),
        "topk_spread_delta_selection": window_mean(ts, selection_w),
        "topk_spread_delta_change_init_selection": window_mean(ts, selection_w) - window_mean(ts, init_w),

        "center_delta_drift_init_selection": cos_distance(init_center, selection_center),
        "center_delta_drift_mid_selection": cos_distance(mid_center, selection_center),

        "hidden_delta_vel_mid": hidden_vel[:, vel_mid_idx].mean(axis=1) if len(vel_mid_idx) else hidden_vel.mean(axis=1),
        "hidden_delta_vel_selection": hidden_vel[:, vel_selection_idx].mean(axis=1) if len(vel_selection_idx) else hidden_vel.mean(axis=1),
    })
    return out


def regression_cv_eval(X, y, groups):
    n_splits = min(5, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)

    pred = np.zeros_like(y, dtype=np.float64)
    for tr, te in cv.split(X, y, groups):
        model = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            Ridge(alpha=1.0)
        )
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])

    spr = None
    if np.std(pred) > 1e-9 and np.std(y) > 1e-9:
        spr = float(spearmanr(pred, y).correlation)

    return {
        "corr": safe_corr(pred, y),
        "spearman": spr,
        "r2": float(r2_score(y, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y, pred))),
    }


def run_cft_regressions(sample_metrics):
    groups = sample_metrics["base_id"].values

    topk_cols = [
        "topk_entropy_delta_init",
        "topk_entropy_delta_mid",
        "topk_entropy_delta_selection",
        "topk_entropy_delta_drop_init_selection",
        "topk_entropy_delta_slope_mid",
        "topk_entropy_delta_slope_selection",
        "topk_spread_delta_init",
        "topk_spread_delta_mid",
        "topk_spread_delta_selection",
        "topk_spread_delta_change_init_selection",
        "center_delta_drift_init_selection",
        "center_delta_drift_mid_selection",
    ]
    dyn_cols = [
        "hidden_delta_vel_mid",
        "hidden_delta_vel_selection",
    ]
    cft_cols = topk_cols + dyn_cols

    targets = [
        "Cbit_eff_mech_init_selection",
        "Expansion_eff_mech_init_selection",
        "Cbit_eff_mech_mid_selection",
    ]

    out = {"feature_columns": {"topk_only": topk_cols, "cft_proxy": cft_cols}}
    for target in targets:
        y = sample_metrics[target].values.astype(float)
        out[target] = {
            "topk_only_group": regression_cv_eval(sample_metrics[topk_cols].values, y, groups),
            "cft_proxy_group": regression_cv_eval(sample_metrics[cft_cols].values, y, groups),
        }
    return out


def build_verdict(layer_metrics, sample_metrics, reg, prob_temperature):
    H_init = float(sample_metrics["H_mech_init"].mean())
    H_mid = float(sample_metrics["H_mech_mid"].mean())
    H_selection = float(sample_metrics["H_mech_selection"].mean())
    N_init = float(sample_metrics["N_eff_init"].mean())
    N_mid = float(sample_metrics["N_eff_mid"].mean())
    N_selection = float(sample_metrics["N_eff_selection"].mean())

    first_layer_acc = float(layer_metrics.iloc[0]["mech_acc"])
    first_layer_H = float(layer_metrics.iloc[0]["H_mech_mean"])

    selection_acc = float(sample_metrics["selection_correct"].mean())

    raw = sample_metrics["raw_Cbit_mech_init_selection"]
    eff = sample_metrics["Cbit_eff_mech_init_selection"]
    exp_eff = sample_metrics["Expansion_eff_mech_init_selection"]
    correct = sample_metrics["selection_correct"] == 1

    raw_gap = float(raw[correct].mean() - raw[~correct].mean()) if (~correct).sum() else None
    eff_gap = float(eff[correct].mean() - eff[~correct].mean()) if (~correct).sum() else None

    cft_cbit = reg["Cbit_eff_mech_init_selection"]["cft_proxy_group"]
    topk_cbit = reg["Cbit_eff_mech_init_selection"]["topk_only_group"]
    cft_exp = reg["Expansion_eff_mech_init_selection"]["cft_proxy_group"]
    topk_exp = reg["Expansion_eff_mech_init_selection"]["topk_only_group"]

    # Hard saturation threshold: if first layer is still too high, mark caveat.
    saturation = (first_layer_acc > 0.90 and first_layer_H < 0.45 and prob_temperature <= 1.0)

    pass1 = (N_init > 1.15) and (N_selection > 1.15)
    compression = H_init > H_selection
    expansion_then_selection = (H_mid > H_init) and (H_mid >= H_selection)
    pass2 = compression or expansion_then_selection
    pass3 = (eff_gap is not None and raw_gap is not None and eff_gap > raw_gap)
    pass4_cbit = (
        cft_cbit["corr"] is not None and
        cft_cbit["corr"] > 0.30 and
        cft_cbit["r2"] > 0.10 and
        cft_cbit["r2"] > topk_cbit["r2"]
    )
    pass4_exp = (
        cft_exp["corr"] is not None and
        cft_exp["corr"] > 0.30 and
        cft_exp["r2"] > 0.10 and
        cft_exp["r2"] > topk_exp["r2"]
    )

    if saturation:
        verdict = "FAIL_OR_CAVEAT_FIRST_LAYER_DELTA_SATURATION"
    elif pass1 and compression and pass3 and pass4_cbit:
        verdict = "PASS_PSA3A3_PAIRDELTA_MECHANISM_SELECTION_STRONG"
    elif pass1 and expansion_then_selection:
        verdict = "PARTIAL_PAIRDELTA_MECHANISM_UNFOLDING_SELECTION"
    elif pass1 and pass2:
        verdict = "PARTIAL_PAIRDELTA_MECHANISM_SPACE_MEASURABLE"
    else:
        verdict = "FAIL_PAIRDELTA_MECHANISM_SPACE_NOT_MEASURABLE"

    return {
        "verdict": verdict,
        "pass_flags": {
            "PASS1_structure_space_measurable": bool(pass1),
            "PASS2_selection_compression_or_unfolding_selection": bool(pass2),
            "PASS3_effective_gt_raw": bool(pass3),
            "PASS4_cft_predicts_cbiteff": bool(pass4_cbit),
            "PASS4_alt_cft_predicts_expansion_eff": bool(pass4_exp),
            "first_layer_saturation_caveat": bool(saturation),
        },
        "summary_metrics": {
            "H_mech_init_mean": H_init,
            "H_mech_mid_mean": H_mid,
            "H_mech_selection_mean": H_selection,
            "N_eff_init_mean": N_init,
            "N_eff_mid_mean": N_mid,
            "N_eff_selection_mean": N_selection,
            "raw_Cbit_mean": float(sample_metrics["raw_Cbit_mech_init_selection"].mean()),
            "raw_Expansion_mean": float(sample_metrics["raw_Expansion_mech_init_selection"].mean()),
            "Cbit_eff_mean": float(sample_metrics["Cbit_eff_mech_init_selection"].mean()),
            "Expansion_eff_mean": float(sample_metrics["Expansion_eff_mech_init_selection"].mean()),
            "selection_probe_acc": selection_acc,
            "first_layer_acc": first_layer_acc,
            "first_layer_H": first_layer_H,
            "raw_correct_wrong_gap": raw_gap,
            "effective_correct_wrong_gap": eff_gap,
            "cft_cbit_group_corr": cft_cbit["corr"],
            "cft_cbit_group_r2": cft_cbit["r2"],
            "topk_cbit_group_corr": topk_cbit["corr"],
            "topk_cbit_group_r2": topk_cbit["r2"],
            "cft_expansion_group_corr": cft_exp["corr"],
            "cft_expansion_group_r2": cft_exp["r2"],
            "topk_expansion_group_corr": topk_exp["corr"],
            "topk_expansion_group_r2": topk_exp["r2"],
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", default=DEFAULT_MODEL_KEY, choices=list(MODEL_PATHS.keys()))
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--n_base", type=int, default=24)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--prob_temperature", type=float, default=1.0)
    parser.add_argument("--local_files_only", action="store_true", default=True)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PSA-3A.3: Pair-Delta Mechanism Unfolding/Selection Audit")
    print("=" * 80)
    print(f"model_key={args.model_key}")
    print(f"model_path={MODEL_PATHS[args.model_key]}")
    print(f"device={args.device}")
    print(f"dtype={args.dtype}")
    print(f"n_base={args.n_base}")
    print(f"topk={args.topk}")
    print(f"prob_temperature={args.prob_temperature}")
    print(f"out_dir={out_dir.resolve()}")

    prompt_df, pair_df = build_pair_dataset(args.n_base)
    prompt_df.to_csv(out_dir / "psa3a3_prompts.csv", index=False, encoding="utf-8-sig")
    pair_df.to_csv(out_dir / "psa3a3_pairs.csv", index=False, encoding="utf-8-sig")
    print(f"[data] prompts={len(prompt_df)}, pairs={len(pair_df)}")

    tokenizer, model = load_model(args.model_key, args.device, args.dtype, args.local_files_only)

    feats = extract_hidden_topk_features(
        tokenizer=tokenizer,
        model=model,
        df=prompt_df,
        device=args.device,
        max_len=args.max_len,
        topk=args.topk,
        batch_size=1,
    )

    delta_feats = build_delta_features(feats, pair_df)
    labels = pair_df["mechanism_id"].values.astype(int)
    groups = pair_df["base_id"].values.astype(int)

    mech_probs, layer_metrics = layerwise_oof_mech_probs(
        X_by_layer=delta_feats["hidden_delta"],
        labels=labels,
        groups=groups,
        prob_temperature=args.prob_temperature,
    )
    layer_metrics.to_csv(out_dir / "psa3a3_layer_metrics.csv", index=False, encoding="utf-8-sig")

    sample_metrics = build_sample_metrics(pair_df, mech_probs, delta_feats)
    sample_metrics.to_csv(out_dir / "psa3a3_sample_metrics.csv", index=False, encoding="utf-8-sig")

    reg = run_cft_regressions(sample_metrics)
    with open(out_dir / "psa3a3_cft_regression.json", "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)

    verdict = build_verdict(layer_metrics, sample_metrics, reg, args.prob_temperature)
    verdict["config"] = vars(args)
    verdict["model_path"] = MODEL_PATHS[args.model_key]
    verdict["mechanisms"] = MECHANISMS
    verdict["note"] = (
        "PSA-3A.3 measures mechanism possibility in condition-clean delta trajectory space. "
        "If verdict is expansion-not-compression, interpret as mechanism possibilities being unfolded "
        "rather than compressed in this window."
    )

    with open(out_dir / "psa3a3_summary.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)

    with open(out_dir / "psa3a3_verdict.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(verdict, ensure_ascii=False, indent=2))

    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()
