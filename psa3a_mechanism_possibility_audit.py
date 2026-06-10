# -*- coding: utf-8 -*-
"""
PSA-3A: Mechanism Possibility Space Audit
========================================

Goal
----
Validate whether a *mechanism-level* possibility space exists and undergoes Cbit
compression during LLM inference.

This is the first PSA-3 experiment:
    Omega_mech = {stable, competition, closure, logic, random}

Core measurements:
    H_mech(l)              = entropy of mechanism posterior at layer l
    raw_Cbit_mech          = H_mech(init_window) - H_mech(basin_window)
    Q_mech                 = out-of-fold p(true_mechanism | basin hidden state)
    Cbit_eff_mech          = raw_Cbit_mech * Q_mech
    CFT proxy -> Cbit_eff  = regression from TopK/Dynamics proxy to Cbit_eff

Default:
    Qwen smoke test with hardcoded local path.

Expected environment:
    Windows + CUDA + transformers + torch + sklearn + pandas + numpy

Outputs:
    psa3a_outputs/
        psa3a_samples.csv
        psa3a_layer_metrics.csv
        psa3a_summary.json
        psa3a_sample_metrics.csv
        psa3a_cft_regression.json
        psa3a_verdict.txt
"""

import os
import json
import math
import random
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr


# ============================================================
# 0. Hardcoded paths
# ============================================================

MODEL_PATHS = {
    "qwen": r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main",
    "llama": r"D:\model\Llama-3.2-1B-Instruct",
    "gemma": r"D:\model\gemma-2-2b-it",
}

DEFAULT_MODEL_KEY = "qwen"
DEFAULT_OUT_DIR = r"C:\Users\ZH\Desktop\AGI\outputs\psa3a_outputs"

SEED = 20260607

MECHANISMS = ["stable", "competition", "closure", "logic", "random"]
MECH_TO_ID = {m: i for i, m in enumerate(MECHANISMS)}


# ============================================================
# 1. Synthetic mechanism dataset
# ============================================================

ENTITIES = [
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
]

SURFACES = [
    "Read the facts carefully and identify the underlying reasoning pattern.",
    "Consider the following mini-world. Focus on the relation structure, not the names.",
    "Analyze the constraint pattern in this short case.",
    "Classify the mechanism behind the following facts.",
    "Inspect the local rule system and choose the mechanism type.",
    "Determine what kind of reasoning situation this example represents.",
]


def make_prompt(mech: str, entity_tuple: Tuple[str, str, str, str], surface: str, idx: int) -> str:
    a, b, c, d = entity_tuple

    if mech == "stable":
        body = f"""
Facts:
1. {a} is linked to {b}.
2. {b} is linked to {c}.
3. The rule is consistent: links compose normally.
Question:
What is the reasoning mechanism here?
Options: stable, competition, closure, logic, random.
"""
    elif mech == "competition":
        body = f"""
Facts:
1. {a} is linked to {b}.
2. {a} is also linked to {c}.
3. Both links have similar support, so two interpretations remain plausible.
Question:
What is the reasoning mechanism here?
Options: stable, competition, closure, logic, random.
"""
    elif mech == "closure":
        body = f"""
Facts:
1. {a} is linked to {b}.
2. {b} is linked to {c}.
3. New exception: when {a} passes through {b}, the final link should be {d}, not {c}.
Question:
What is the reasoning mechanism here?
Options: stable, competition, closure, logic, random.
"""
    elif mech == "logic":
        body = f"""
Facts:
1. If {a} implies {b}, and {b} implies {c}, then {a} implies {c}.
2. If the implication is contradicted, the chain is invalid.
3. Use formal consistency rather than surface association.
Question:
What is the reasoning mechanism here?
Options: stable, competition, closure, logic, random.
"""
    elif mech == "random":
        body = f"""
Facts:
1. {a} is blue because seven clocks whisper.
2. {b} contains the idea of sideways Thursday.
3. {c} and {d} are unrelated fragments with no stable rule.
Question:
What is the reasoning mechanism here?
Options: stable, competition, closure, logic, random.
"""
    else:
        raise ValueError(f"Unknown mechanism: {mech}")

    return (
        f"{surface}\n"
        f"Case ID: {idx}\n"
        f"{body.strip()}\n"
        "Answer with the mechanism name only."
    )


def build_dataset(n_per_mech: int = 48) -> pd.DataFrame:
    rows = []
    rng = random.Random(SEED)

    for mech in MECHANISMS:
        for i in range(n_per_mech):
            ent = ENTITIES[i % len(ENTITIES)]
            surf = SURFACES[(i // len(ENTITIES)) % len(SURFACES)]
            prompt = make_prompt(mech, ent, surf, i)
            rows.append({
                "row_id": len(rows),
                "mechanism": mech,
                "mechanism_id": MECH_TO_ID[mech],
                "entity_group": i % len(ENTITIES),
                "surface_id": (i // len(ENTITIES)) % len(SURFACES),
                "group_id": f"{mech}_{i % len(ENTITIES)}",
                "prompt": prompt,
            })

    rng.shuffle(rows)
    for j, r in enumerate(rows):
        r["row_id"] = j
    return pd.DataFrame(rows)


# ============================================================
# 2. Model loading and feature extraction
# ============================================================

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
) -> Dict[str, Any]:
    """
    Extract:
        hidden_last_token: N x L x D
        topk_entropy: N x L
        topk_spread: N x L
        topk_center: N x L x D
    Layer L excludes embedding state: hidden_states[1:].
    """

    all_hidden = []
    all_entropy = []
    all_spread = []
    all_center = []

    emb_weight = model.get_output_embeddings().weight.detach()

    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        texts = batch["prompt"].tolist()

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

        # exclude embedding state, keep transformer layers
        hidden_states = outputs.hidden_states[1:]

        # last non-pad token position per sample
        attention = inputs["attention_mask"]
        last_pos = attention.sum(dim=1) - 1

        layer_hidden = []
        layer_ent = []
        layer_spread = []
        layer_center = []

        for h in hidden_states:
            # B x D
            h_last = h[torch.arange(h.shape[0], device=device), last_pos]
            layer_hidden.append(h_last.detach().float().cpu())

            logits = torch.matmul(h_last, emb_weight.t())
            vals, ids = torch.topk(logits, k=min(topk, logits.shape[-1]), dim=-1)

            probs = torch.softmax(vals.float(), dim=-1)
            entropy = -(probs * torch.log2(probs + 1e-12)).sum(dim=-1)

            tok_emb = emb_weight[ids].detach().float()       # B x K x D
            center = tok_emb.mean(dim=1)                    # B x D
            center_norm = F.normalize(center, dim=-1)
            tok_norm = F.normalize(tok_emb, dim=-1)
            cos = (tok_norm * center_norm.unsqueeze(1)).sum(dim=-1)
            spread = (1.0 - cos).mean(dim=-1)

            layer_ent.append(entropy.detach().cpu())
            layer_spread.append(spread.detach().cpu())
            layer_center.append(center.detach().cpu())

        # L x B x D -> B x L x D
        all_hidden.append(torch.stack(layer_hidden, dim=0).transpose(0, 1))
        all_entropy.append(torch.stack(layer_ent, dim=0).transpose(0, 1))
        all_spread.append(torch.stack(layer_spread, dim=0).transpose(0, 1))
        all_center.append(torch.stack(layer_center, dim=0).transpose(0, 1))

        print(f"[extract] {start + len(batch)}/{len(df)} done")

    hidden = torch.cat(all_hidden, dim=0).numpy()
    topk_entropy = torch.cat(all_entropy, dim=0).numpy()
    topk_spread = torch.cat(all_spread, dim=0).numpy()
    topk_center = torch.cat(all_center, dim=0).numpy()

    return {
        "hidden": hidden,
        "topk_entropy": topk_entropy,
        "topk_spread": topk_spread,
        "topk_center": topk_center,
    }


# ============================================================
# 3. Mechanism posterior probes
# ============================================================

def entropy_bits(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(prob, 1e-12, 1.0)
    return -(prob * np.log2(prob)).sum(axis=-1)


def safe_corr(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return np.nan
    return float(pearsonr(x, y)[0])


def layerwise_oof_mech_probs(hidden: np.ndarray, labels: np.ndarray, n_splits: int = 5):
    """
    hidden: N x L x D
    returns:
        probs: N x L x C
        layer_metrics: DataFrame
    """
    n, L, d = hidden.shape
    C = len(np.unique(labels))
    probs = np.zeros((n, L, C), dtype=np.float32)
    metrics = []

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for l in range(L):
        X = hidden[:, l, :]
        clf = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                class_weight="balanced",
                C=1.0,
            )
        )

        p = cross_val_predict(
            clf,
            X,
            labels,
            cv=cv,
            method="predict_proba",
            n_jobs=None,
        )

        # sklearn class order should be sorted unique labels
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

        print(f"[probe] layer {l:02d}: acc={metrics[-1]['mech_acc']:.3f}, H={metrics[-1]['H_mech_mean']:.3f}")

    return probs, pd.DataFrame(metrics)


# ============================================================
# 4. Cbit and CFT proxy
# ============================================================

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
    # series: N x L
    x = np.asarray(idx, dtype=np.float64)
    x = x - x.mean()
    denom = (x ** 2).sum() + 1e-12
    y = series[:, idx].astype(np.float64)
    y_centered = y - y.mean(axis=1, keepdims=True)
    return (y_centered @ x) / denom


def build_sample_metrics(
    df: pd.DataFrame,
    mech_probs: np.ndarray,
    topk_entropy: np.ndarray,
    topk_spread: np.ndarray,
    topk_center: np.ndarray,
    hidden: np.ndarray,
) -> pd.DataFrame:
    labels = df["mechanism_id"].values
    n, L, C = mech_probs.shape

    init_w = clamp_window(0, 6, L)
    mid_w = clamp_window(7, 19, L)
    basin_w = clamp_window(20, 25, L)
    if len(basin_w) == 0:
        basin_w = clamp_window(max(0, L - 6), L - 2, L)

    H_mech = entropy_bits(mech_probs.reshape(-1, C)).reshape(n, L)

    H_init = window_mean(H_mech, init_w)
    H_mid = window_mean(H_mech, mid_w) if len(mid_w) else H_init
    H_basin = window_mean(H_mech, basin_w)

    raw_cbit_init_basin = H_init - H_basin
    raw_cbit_mid_basin = H_mid - H_basin

    # Q_mech = mean OOF probability assigned to true mechanism in basin window
    true_prob_layer = np.zeros((n, L), dtype=np.float32)
    for i in range(n):
        true_prob_layer[i, :] = mech_probs[i, :, labels[i]]
    Q_mech_basin = window_mean(true_prob_layer, basin_w)
    Q_mech_init = window_mean(true_prob_layer, init_w)

    cbit_eff_init_basin = raw_cbit_init_basin * Q_mech_basin
    cbit_eff_mid_basin = raw_cbit_mid_basin * Q_mech_basin

    basin_pred = window_mean(mech_probs, basin_w).argmax(axis=1)
    basin_correct = (basin_pred == labels).astype(int)

    init_center = window_center_mean(topk_center, init_w)
    mid_center = window_center_mean(topk_center, mid_w) if len(mid_w) else init_center
    basin_center = window_center_mean(topk_center, basin_w)

    # hidden dynamics
    hidden_vel = np.linalg.norm(np.diff(hidden, axis=1), axis=-1)  # N x (L-1)
    vel_mid_idx = [i for i in mid_w if i < hidden_vel.shape[1]]
    vel_basin_idx = [i for i in basin_w if i < hidden_vel.shape[1]]

    out = pd.DataFrame({
        "row_id": df["row_id"].values,
        "mechanism": df["mechanism"].values,
        "mechanism_id": labels,
        "group_id": df["group_id"].values,
        "surface_id": df["surface_id"].values,

        "H_mech_init": H_init,
        "H_mech_mid": H_mid,
        "H_mech_basin": H_basin,
        "N_eff_init": 2.0 ** H_init,
        "N_eff_basin": 2.0 ** H_basin,

        "raw_Cbit_mech_init_basin": raw_cbit_init_basin,
        "raw_Cbit_mech_mid_basin": raw_cbit_mid_basin,
        "Q_mech_basin": Q_mech_basin,
        "Q_mech_init": Q_mech_init,
        "Cbit_eff_mech_init_basin": cbit_eff_init_basin,
        "Cbit_eff_mech_mid_basin": cbit_eff_mid_basin,
        "basin_pred_mechanism_id": basin_pred,
        "basin_correct": basin_correct,

        # CFT/TopK proxy features
        "topk_entropy_init": window_mean(topk_entropy, init_w),
        "topk_entropy_mid": window_mean(topk_entropy, mid_w) if len(mid_w) else window_mean(topk_entropy, init_w),
        "topk_entropy_basin": window_mean(topk_entropy, basin_w),
        "topk_entropy_drop_init_basin": window_mean(topk_entropy, init_w) - window_mean(topk_entropy, basin_w),
        "topk_entropy_slope_mid": slope_per_sample(topk_entropy, mid_w) if len(mid_w) >= 2 else 0.0,
        "topk_entropy_slope_basin": slope_per_sample(topk_entropy, basin_w) if len(basin_w) >= 2 else 0.0,

        "topk_spread_init": window_mean(topk_spread, init_w),
        "topk_spread_mid": window_mean(topk_spread, mid_w) if len(mid_w) else window_mean(topk_spread, init_w),
        "topk_spread_basin": window_mean(topk_spread, basin_w),
        "topk_spread_delta_init_basin": window_mean(topk_spread, basin_w) - window_mean(topk_spread, init_w),

        "center_drift_init_basin": cos_distance(init_center, basin_center),
        "center_drift_mid_basin": cos_distance(mid_center, basin_center),

        "hidden_vel_mid": hidden_vel[:, vel_mid_idx].mean(axis=1) if len(vel_mid_idx) else hidden_vel.mean(axis=1),
        "hidden_vel_basin": hidden_vel[:, vel_basin_idx].mean(axis=1) if len(vel_basin_idx) else hidden_vel.mean(axis=1),
    })

    return out


def regression_cv_eval(X: np.ndarray, y: np.ndarray, groups: np.ndarray = None, n_splits: int = 5) -> Dict[str, float]:
    if groups is None:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        # stratify by quantile bins of y
        bins = pd.qcut(y, q=min(5, len(np.unique(y))), labels=False, duplicates="drop")
        split_iter = cv.split(X, bins)
    else:
        cv = GroupKFold(n_splits=n_splits)
        split_iter = cv.split(X, y, groups)

    pred = np.zeros_like(y, dtype=np.float64)
    for tr, te in split_iter:
        model = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            Ridge(alpha=1.0)
        )
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])

    corr = safe_corr(pred, y)
    spr = spearmanr(pred, y).correlation if np.std(pred) > 1e-9 and np.std(y) > 1e-9 else np.nan
    return {
        "corr": float(corr) if corr == corr else None,
        "spearman": float(spr) if spr == spr else None,
        "r2": float(r2_score(y, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y, pred))),
    }


def run_cft_regressions(sample_metrics: pd.DataFrame) -> Dict[str, Any]:
    y = sample_metrics["Cbit_eff_mech_init_basin"].values.astype(float)
    groups = sample_metrics["group_id"].astype(str).values

    topk_cols = [
        "topk_entropy_init",
        "topk_entropy_mid",
        "topk_entropy_basin",
        "topk_entropy_drop_init_basin",
        "topk_entropy_slope_mid",
        "topk_entropy_slope_basin",
        "topk_spread_init",
        "topk_spread_mid",
        "topk_spread_basin",
        "topk_spread_delta_init_basin",
        "center_drift_init_basin",
        "center_drift_mid_basin",
    ]

    dyn_cols = [
        "hidden_vel_mid",
        "hidden_vel_basin",
    ]

    cft_cols = topk_cols + dyn_cols

    results = {
        "target": "Cbit_eff_mech_init_basin",
        "topk_only_stratified": regression_cv_eval(sample_metrics[topk_cols].values, y, None),
        "cft_proxy_stratified": regression_cv_eval(sample_metrics[cft_cols].values, y, None),
        "topk_only_group": regression_cv_eval(sample_metrics[topk_cols].values, y, groups),
        "cft_proxy_group": regression_cv_eval(sample_metrics[cft_cols].values, y, groups),
        "feature_columns": {
            "topk_only": topk_cols,
            "cft_proxy": cft_cols,
        }
    }
    return results


# ============================================================
# 5. Verdict
# ============================================================

def build_verdict(layer_metrics: pd.DataFrame, sample_metrics: pd.DataFrame, reg: Dict[str, Any]) -> Dict[str, Any]:
    H_init = sample_metrics["H_mech_init"].mean()
    H_mid = sample_metrics["H_mech_mid"].mean()
    H_basin = sample_metrics["H_mech_basin"].mean()
    N_init = sample_metrics["N_eff_init"].mean()
    N_basin = sample_metrics["N_eff_basin"].mean()

    raw = sample_metrics["raw_Cbit_mech_init_basin"]
    eff = sample_metrics["Cbit_eff_mech_init_basin"]
    correct = sample_metrics["basin_correct"] == 1

    raw_gap = float(raw[correct].mean() - raw[~correct].mean()) if (~correct).sum() > 0 else None
    eff_gap = float(eff[correct].mean() - eff[~correct].mean()) if (~correct).sum() > 0 else None

    cft_group = reg["cft_proxy_group"]
    topk_group = reg["topk_only_group"]

    pass1 = (N_init > 1.1) and (N_basin > 1.1)
    pass2 = (H_init > H_basin) or (H_mid > H_basin)
    pass3 = (eff_gap is not None and raw_gap is not None and eff_gap > raw_gap)
    pass4 = (
        cft_group["corr"] is not None and
        cft_group["corr"] > 0.30 and
        cft_group["r2"] > 0.20 and
        cft_group["r2"] > topk_group["r2"]
    )

    if pass1 and pass2 and pass3 and pass4:
        verdict = "PASS_PSA3A_MECHANISM_CBIT_STRONG"
    elif pass1 and pass2 and (pass3 or pass4):
        verdict = "PARTIAL_MECHANISM_CBIT"
    elif pass1 and not pass2:
        verdict = "FAIL_NO_MECHANISM_COMPRESSION"
    else:
        verdict = "FAIL_MECHANISM_SPACE_NOT_MEASURABLE"

    return {
        "verdict": verdict,
        "pass_flags": {
            "PASS1_structure_space_measurable": bool(pass1),
            "PASS2_positive_structural_cbit": bool(pass2),
            "PASS3_effective_gt_raw": bool(pass3),
            "PASS4_cft_predicts_structural_cbiteff": bool(pass4),
        },
        "summary_metrics": {
            "H_mech_init_mean": float(H_init),
            "H_mech_mid_mean": float(H_mid),
            "H_mech_basin_mean": float(H_basin),
            "N_eff_init_mean": float(N_init),
            "N_eff_basin_mean": float(N_basin),
            "raw_Cbit_mean": float(raw.mean()),
            "Cbit_eff_mean": float(eff.mean()),
            "basin_probe_acc": float(sample_metrics["basin_correct"].mean()),
            "raw_correct_wrong_gap": raw_gap,
            "effective_correct_wrong_gap": eff_gap,
            "cft_proxy_group_corr": cft_group["corr"],
            "cft_proxy_group_r2": cft_group["r2"],
            "topk_only_group_corr": topk_group["corr"],
            "topk_only_group_r2": topk_group["r2"],
        }
    }


# ============================================================
# 6. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", default=DEFAULT_MODEL_KEY, choices=list(MODEL_PATHS.keys()))
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--n_per_mech", type=int, default=48)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--local_files_only", action="store_true", default=True)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PSA-3A: Mechanism Possibility Space Audit")
    print("=" * 80)
    print(f"model_key={args.model_key}")
    print(f"model_path={MODEL_PATHS[args.model_key]}")
    print(f"device={args.device}")
    print(f"dtype={args.dtype}")
    print(f"n_per_mech={args.n_per_mech}")
    print(f"out_dir={out_dir.resolve()}")

    df = build_dataset(n_per_mech=args.n_per_mech)
    df.to_csv(out_dir / "psa3a_samples.csv", index=False, encoding="utf-8-sig")
    print(f"[data] n={len(df)}, labels={df['mechanism'].value_counts().to_dict()}")

    tokenizer, model = load_model(args.model_key, args.device, args.dtype, args.local_files_only)

    feats = extract_hidden_topk_features(
        tokenizer=tokenizer,
        model=model,
        df=df,
        device=args.device,
        max_len=args.max_len,
        topk=args.topk,
        batch_size=1,
    )

    labels = df["mechanism_id"].values.astype(int)

    mech_probs, layer_metrics = layerwise_oof_mech_probs(
        hidden=feats["hidden"],
        labels=labels,
        n_splits=5,
    )
    layer_metrics.to_csv(out_dir / "psa3a_layer_metrics.csv", index=False, encoding="utf-8-sig")

    sample_metrics = build_sample_metrics(
        df=df,
        mech_probs=mech_probs,
        topk_entropy=feats["topk_entropy"],
        topk_spread=feats["topk_spread"],
        topk_center=feats["topk_center"],
        hidden=feats["hidden"],
    )
    sample_metrics.to_csv(out_dir / "psa3a_sample_metrics.csv", index=False, encoding="utf-8-sig")

    reg = run_cft_regressions(sample_metrics)
    with open(out_dir / "psa3a_cft_regression.json", "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)

    verdict = build_verdict(layer_metrics, sample_metrics, reg)
    verdict["config"] = vars(args)
    verdict["model_path"] = MODEL_PATHS[args.model_key]
    verdict["mechanisms"] = MECHANISMS

    with open(out_dir / "psa3a_summary.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)

    verdict_text = json.dumps(verdict, ensure_ascii=False, indent=2)
    with open(out_dir / "psa3a_verdict.txt", "w", encoding="utf-8") as f:
        f.write(verdict_text)

    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    print(verdict_text)
    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
