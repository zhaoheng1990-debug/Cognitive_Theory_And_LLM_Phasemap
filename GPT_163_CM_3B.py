# -*- coding: utf-8 -*-
"""
CM-3B: Real Perturbation Response Audit

Goal:
  CM-3 used a logistic proxy for response:
      chi = |dP(Gen_E)/dDeltaU|
  CM-3B performs REAL prompt perturbation and asks:
      Are samples inside each model-specific critical band more responsive
      to small conflict-oriented perturbations than samples outside the band?

Inputs expected from prior runs:
  cm1_outputs/{qwen,llama,gemma}_deltaR_dataset.csv
  cm2_outputs/{qwen,llama,gemma}_critical_band_summary.json

Outputs:
  cm3b_outputs/cm3b_model_summary.csv
  cm3b_outputs/cm3b_overall_summary.json
  cm3b_outputs/{model}_perturb_response_summary.json
  cm3b_outputs/{model}_perturb_response_dataset.csv

Local model paths are hard-coded below.

Run:
  python cm3b_real_perturbation_response_audit.py
"""

import argparse
import gc
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------
# Local Model Paths
# -----------------------------
QWEN_PATH = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"
LLAMA_PATH = r"D:\model\Llama-3.2-1B-Instruct"
GEMMA_PATH = r"D:\model\gemma-2-2b-it"

MODEL_PATHS = {
    "qwen": QWEN_PATH,
    "llama": LLAMA_PATH,
    "gemma": GEMMA_PATH,
}

# -----------------------------
# Config
# -----------------------------
SEED = 42
CM1_DIR = Path("./cm1_outputs")
CM2_DIR = Path("./cm2_outputs")
SAVE_DIR = Path("./cm3b_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MAX_LEN = 260
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "float16" if DEVICE == "cuda" else "float32"

LABEL_POOL = [
    "Red", "Blue", "Green", "Yellow", "North", "South", "East", "West",
    "Alpha", "Beta", "Gamma", "Delta", "Circle", "Square", "Triangle", "Star",
    "Copper", "Silver", "Gold", "Iron", "River", "Forest", "Ocean", "Mountain",
]

CONDITIONS = [
    ("stable_positive", "positive"),
    ("weak_positive", "positive"),
    ("true_compete_balanced", "critical"),
    ("direct_negative", "negative"),
    ("update_negative", "negative"),
    ("exception_negative", "negative"),
]

PERTURBATIONS = [
    "weak_conflict_note",
    "direct_conflict_note",
    "latest_update_note",
]

# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def auc_safe(y, score):
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y, score))
    except Exception:
        return np.nan


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))


def pE_from_R(R):
    # R = logit(C) - logit(E); pE over {C,E} = sigmoid(-R)
    return sigmoid(-np.asarray(R, dtype=np.float64))


def continuation_ids(tokenizer, label: str) -> List[int]:
    return tokenizer(" " + label, add_special_tokens=False)["input_ids"]


def select_single_token_labels(tokenizer, min_needed: int = 12) -> List[str]:
    singles = []
    for lab in LABEL_POOL:
        ids = continuation_ids(tokenizer, lab)
        if len(ids) == 1:
            singles.append(lab)
    if len(singles) < min_needed:
        print("[WARN] Fewer than required single-token labels. Falling back to first-token labels.")
        return LABEL_POOL[:min_needed]
    return singles[:min_needed]

# -----------------------------
# Dataset reconstruction, identical to CM-1
# -----------------------------

def make_prompt(a: str, b: str, clean_label: str, conflict_label: str, condition: str) -> str:
    options = f"Answer with exactly one word: {clean_label} or {conflict_label}."
    base = [
        "You are given a small relation graph.",
        f"Possible labels are: {clean_label} or {conflict_label}.",
    ]

    if condition == "stable_positive":
        facts = [
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located at {clean_label}.",
        ]
    elif condition == "weak_positive":
        facts = [
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located at {clean_label}.",
            f"Weak note: another unrelated item may be associated with {conflict_label}.",
        ]
    elif condition == "true_compete_balanced":
        facts = [
            f"Source one says: {a} belongs to {b}, and {b} is located at {clean_label}.",
            f"Source two says: {a} is located at {conflict_label}.",
            "Both sources are presented without priority.",
        ]
    elif condition == "direct_negative":
        facts = [
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located at {clean_label}.",
            f"Direct override: {a} is located at {conflict_label}.",
        ]
    elif condition == "update_negative":
        facts = [
            f"Old record: {b} is located at {clean_label}.",
            f"Updated record: {b} is now located at {conflict_label}.",
            f"Fact: {a} belongs to {b}.",
            "Use the updated record.",
        ]
    elif condition == "exception_negative":
        facts = [
            f"General rule: items belonging to {b} are located at {clean_label}.",
            f"Exception: {a} is a special case located at {conflict_label}.",
            f"Fact: {a} belongs to {b}.",
            "Use the exception when it applies.",
        ]
    else:
        raise ValueError(condition)

    q = [f"Question: Which label is {a} located at?", options, "Answer:"]
    return "\n".join(base + facts + q)


def perturb_prompt(prompt: str, a: str, b: str, clean_label: str, conflict_label: str, perturb_type: str) -> str:
    """Insert a small conflict-oriented perturbation before Answer:."""
    if perturb_type == "weak_conflict_note":
        note = f"Additional weak note: some sources loosely associate {a} with {conflict_label}."
    elif perturb_type == "direct_conflict_note":
        note = f"Additional direct note: {a} is located at {conflict_label}."
    elif perturb_type == "latest_update_note":
        note = f"Latest update: {b} is now located at {conflict_label}. Prefer the latest update when records disagree."
    else:
        raise ValueError(perturb_type)

    if "\nAnswer:" in prompt:
        return prompt.replace("\nAnswer:", f"\n{note}\nAnswer:")
    return prompt + "\n" + note + "\nAnswer:"


def build_prompt_table(tokenizer, n_graphs: int) -> pd.DataFrame:
    labels = select_single_token_labels(tokenizer, min_needed=12)
    rows = []
    for i in range(n_graphs):
        a = f"A{i:03d}"
        b = f"B{i:03d}"
        clean_label = labels[(2 * i) % len(labels)]
        conflict_label = labels[(2 * i + 1) % len(labels)]
        c_id = continuation_ids(tokenizer, clean_label)[0]
        e_id = continuation_ids(tokenizer, conflict_label)[0]
        for cond, phase in CONDITIONS:
            prompt = make_prompt(a, b, clean_label, conflict_label, cond)
            rows.append({
                "graph_id": i,
                "condition": cond,
                "phase": phase,
                "a": a,
                "b": b,
                "clean_label": clean_label,
                "conflict_label": conflict_label,
                "clean_token_id": c_id,
                "conflict_token_id": e_id,
                "base_prompt": prompt,
            })
    return pd.DataFrame(rows)

# -----------------------------
# DeltaU + band membership
# -----------------------------

def orient_delta_u(df: pd.DataFrame, dR_cols: List[str]):
    X = df[dR_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    pca = PCA(n_components=min(3, X.shape[1]))
    scores = pca.fit_transform(X)
    du = scores[:, 0].astype(np.float32)

    phases = df["phase"].astype(str).values
    pos_mean = np.nanmean(du[phases == "positive"])
    neg_mean = np.nanmean(du[phases == "negative"])
    if np.isfinite(pos_mean) and np.isfinite(neg_mean) and pos_mean < neg_mean:
        du = -du
        scores[:, 0] = -scores[:, 0]

    return du, pca.explained_variance_ratio_.tolist()


def band_distance(x, lo, hi):
    x = np.asarray(x, dtype=np.float64)
    return np.maximum(lo - x, 0.0) + np.maximum(x - hi, 0.0)

# -----------------------------
# Model scoring
# -----------------------------

@torch.no_grad()
def score_prompts(model, tokenizer, prompts: List[str], c_ids: np.ndarray, e_ids: np.ndarray, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return R=logitC-logitE, pE, predE(first-token among C/E)."""
    R_all = []
    pE_all = []
    predE_all = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        c_batch = torch.tensor(c_ids[start:start + batch_size], dtype=torch.long, device=model.device)
        e_batch = torch.tensor(e_ids[start:start + batch_size], dtype=torch.long, device=model.device)

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs, use_cache=False)
        logits = outputs.logits[:, -1, :].float()
        lc = logits.gather(1, c_batch.view(-1, 1)).squeeze(1)
        le = logits.gather(1, e_batch.view(-1, 1)).squeeze(1)
        R = (lc - le).detach().float().cpu().numpy()
        pE = pE_from_R(R)
        predE = (le > lc).detach().cpu().numpy().astype(int)

        R_all.append(R)
        pE_all.append(pE)
        predE_all.append(predE)

        del inputs, outputs, logits, lc, le
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.concatenate(R_all), np.concatenate(pE_all), np.concatenate(predE_all)

# -----------------------------
# Analysis
# -----------------------------

def summarize_response(df: pd.DataFrame, model_key: str, lo: float, hi: float):
    inside = df["inside_band"].astype(bool).values
    y_inside = inside.astype(int)

    metrics = {
        "model_key": model_key,
        "band_lo": float(lo),
        "band_hi": float(hi),
        "n": int(len(df)),
        "inside_n": int(inside.sum()),
        "outside_n": int((~inside).sum()),
    }

    response_cols = [
        "response_abs_dP_mean",
        "response_abs_dR_mean",
        "response_to_conflict_dP_mean",
        "response_flip_any",
        "response_cross_boundary_any",
        "response_abs_dP_max",
        "response_to_conflict_dP_max",
    ]

    for col in response_cols:
        inside_mean = float(df.loc[inside, col].mean()) if inside.any() else np.nan
        outside_mean = float(df.loc[~inside, col].mean()) if (~inside).any() else np.nan
        metrics[f"{col}_inside"] = inside_mean
        metrics[f"{col}_outside"] = outside_mean
        metrics[f"{col}_ratio_inside_outside"] = float(inside_mean / (outside_mean + 1e-12)) if np.isfinite(inside_mean) and np.isfinite(outside_mean) else np.nan
        metrics[f"auc_inside_by_{col}"] = auc_safe(y_inside, df[col].values)

    # Phase summaries
    by_phase = {}
    for phase, sub in df.groupby("phase"):
        by_phase[str(phase)] = {
            "n": int(len(sub)),
            "inside_frac": float(sub["inside_band"].mean()),
            "DeltaU_mean": float(sub["DeltaU_CM3B"].mean()),
            "dist_to_band_mean": float(sub["dist_to_band"].mean()),
            "response_abs_dP_mean": float(sub["response_abs_dP_mean"].mean()),
            "response_to_conflict_dP_mean": float(sub["response_to_conflict_dP_mean"].mean()),
            "response_flip_any": float(sub["response_flip_any"].mean()),
            "response_cross_boundary_any": float(sub["response_cross_boundary_any"].mean()),
        }
    metrics["by_phase"] = by_phase

    # Region summaries
    region = np.where(df["DeltaU_CM3B"].values < lo, "below_band", np.where(df["DeltaU_CM3B"].values > hi, "above_band", "inside_band"))
    df_tmp = df.copy()
    df_tmp["region"] = region
    by_region = {}
    for reg, sub in df_tmp.groupby("region"):
        by_region[str(reg)] = {
            "n": int(len(sub)),
            "phase_critical_rate": float((sub["phase"] == "critical").mean()),
            "phase_negative_rate": float((sub["phase"] == "negative").mean()),
            "response_abs_dP_mean": float(sub["response_abs_dP_mean"].mean()),
            "response_to_conflict_dP_mean": float(sub["response_to_conflict_dP_mean"].mean()),
            "response_flip_any": float(sub["response_flip_any"].mean()),
            "response_cross_boundary_any": float(sub["response_cross_boundary_any"].mean()),
        }
    metrics["by_region"] = by_region

    # Pass criteria: require response lift in at least 2 of 3 robust measures.
    pass_abs = metrics["response_abs_dP_mean_ratio_inside_outside"] > 1.10
    pass_dir = metrics["response_to_conflict_dP_mean_ratio_inside_outside"] > 1.10
    pass_flip = metrics["response_cross_boundary_any_ratio_inside_outside"] > 1.10
    metrics["pass_abs_response_lift_1p10"] = bool(pass_abs)
    metrics["pass_directional_response_lift_1p10"] = bool(pass_dir)
    metrics["pass_cross_boundary_lift_1p10"] = bool(pass_flip)
    metrics["pass_cm3b_real_response"] = bool(sum([pass_abs, pass_dir, pass_flip]) >= 2)

    return metrics


def run_one_model(model_key: str, model_path: str, args):
    print(f"\n==== CM-3B running model={model_key} ====")
    df_delta_path = CM1_DIR / f"{model_key}_deltaR_dataset.csv"
    band_path = CM2_DIR / f"{model_key}_critical_band_summary.json"
    if not df_delta_path.exists():
        raise FileNotFoundError(df_delta_path)
    if not band_path.exists():
        raise FileNotFoundError(band_path)

    df_delta = pd.read_csv(df_delta_path)
    band_summary = read_json(band_path)
    opt_band = band_summary["critical_band"]["optimized_band"]
    lo, hi = float(opt_band["lo"]), float(opt_band["hi"])
    decision_cols = band_summary["decision_cols"]

    du, pc_var = orient_delta_u(df_delta, decision_cols)
    df_delta["DeltaU_CM3B"] = du
    df_delta["inside_band"] = (df_delta["DeltaU_CM3B"] >= lo) & (df_delta["DeltaU_CM3B"] <= hi)
    df_delta["dist_to_band"] = band_distance(df_delta["DeltaU_CM3B"].values, lo, hi)

    # Optional cap for fast debugging.
    if args.max_rows_per_model and len(df_delta) > args.max_rows_per_model:
        # Stratified sample by inside/outside and phase.
        df_delta = (
            df_delta.groupby(["inside_band", "phase"], group_keys=False)
            .apply(lambda x: x.sample(min(len(x), max(1, args.max_rows_per_model // 8)), random_state=SEED))
            .reset_index(drop=True)
        )
        print(f"[{model_key}] sampled rows={len(df_delta)}")

    torch_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    if DEVICE == "cpu":
        model.to("cpu")
    model.eval()

    n_graphs = int(df_delta["graph_id"].max()) + 1
    prompt_table = build_prompt_table(tokenizer, n_graphs)
    df = df_delta.merge(prompt_table, on=["graph_id", "condition", "phase"], how="left")
    if df["base_prompt"].isna().any():
        raise RuntimeError("Prompt reconstruction failed. Check CM-1 dataset compatibility.")

    # Score base prompts.
    base_prompts = df["base_prompt"].tolist()
    c_ids = df["clean_token_id"].values.astype(np.int64)
    e_ids = df["conflict_token_id"].values.astype(np.int64)

    print(f"[{model_key}] scoring base prompts n={len(base_prompts)}")
    R_base, pE_base, predE_base = score_prompts(model, tokenizer, base_prompts, c_ids, e_ids, args.batch_size)
    df["R_base_actual"] = R_base
    df["pE_base_actual"] = pE_base
    df["predE_base_actual"] = predE_base

    # Score perturbation prompts.
    dP_cols = []
    dR_cols = []
    flip_cols = []
    cross_cols = []

    for ptype in PERTURBATIONS:
        pert_prompts = [
            perturb_prompt(row.base_prompt, row.a, row.b, row.clean_label, row.conflict_label, ptype)
            for row in df.itertuples(index=False)
        ]
        print(f"[{model_key}] scoring perturbation={ptype} n={len(pert_prompts)}")
        R_pert, pE_pert, predE_pert = score_prompts(model, tokenizer, pert_prompts, c_ids, e_ids, args.batch_size)

        df[f"R_pert_{ptype}"] = R_pert
        df[f"pE_pert_{ptype}"] = pE_pert
        df[f"predE_pert_{ptype}"] = predE_pert
        df[f"dP_to_conflict_{ptype}"] = pE_pert - pE_base
        df[f"abs_dP_{ptype}"] = np.abs(pE_pert - pE_base)
        df[f"dR_{ptype}"] = R_pert - R_base
        df[f"abs_dR_{ptype}"] = np.abs(R_pert - R_base)
        df[f"flip_{ptype}"] = (predE_pert != predE_base).astype(int)
        df[f"cross_boundary_{ptype}"] = ((pE_base < 0.5) & (pE_pert >= 0.5)).astype(int)

        dP_cols.append(f"abs_dP_{ptype}")
        dR_cols.append(f"abs_dR_{ptype}")
        flip_cols.append(f"flip_{ptype}")
        cross_cols.append(f"cross_boundary_{ptype}")

    dir_cols = [f"dP_to_conflict_{p}" for p in PERTURBATIONS]

    df["response_abs_dP_mean"] = df[dP_cols].mean(axis=1)
    df["response_abs_dP_max"] = df[dP_cols].max(axis=1)
    df["response_abs_dR_mean"] = df[dR_cols].mean(axis=1)
    df["response_abs_dR_max"] = df[dR_cols].max(axis=1)
    df["response_to_conflict_dP_mean"] = df[dir_cols].mean(axis=1)
    df["response_to_conflict_dP_max"] = df[dir_cols].max(axis=1)
    df["response_flip_any"] = (df[flip_cols].max(axis=1) > 0).astype(int)
    df["response_cross_boundary_any"] = (df[cross_cols].max(axis=1) > 0).astype(int)

    summary = summarize_response(df, model_key, lo, hi)
    summary["pc_variance_recomputed"] = pc_var

    # Save.
    df.to_csv(SAVE_DIR / f"{model_key}_perturb_response_dataset.csv", index=False, encoding="utf-8-sig")
    write_json(SAVE_DIR / f"{model_key}_perturb_response_summary.json", summary)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen", "llama", "gemma"], choices=["qwen", "llama", "gemma"])
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dtype", type=str, default=DTYPE, choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-rows-per-model", type=int, default=0, help="0 = use all rows; positive value for quick debugging.")
    args = parser.parse_args()

    set_seed(SEED)
    print(f"Device={DEVICE}, dtype={args.dtype}, batch_size={args.batch_size}")

    summaries = []
    for m in args.models:
        summaries.append(run_one_model(m, MODEL_PATHS[m], args))

    # Flatten key metrics for model summary CSV.
    rows = []
    for s in summaries:
        rows.append({
            "model_key": s["model_key"],
            "band_lo": s["band_lo"],
            "band_hi": s["band_hi"],
            "n": s["n"],
            "inside_n": s["inside_n"],
            "outside_n": s["outside_n"],
            "abs_dP_inside": s["response_abs_dP_mean_inside"],
            "abs_dP_outside": s["response_abs_dP_mean_outside"],
            "abs_dP_ratio": s["response_abs_dP_mean_ratio_inside_outside"],
            "dir_dP_inside": s["response_to_conflict_dP_mean_inside"],
            "dir_dP_outside": s["response_to_conflict_dP_mean_outside"],
            "dir_dP_ratio": s["response_to_conflict_dP_mean_ratio_inside_outside"],
            "cross_inside": s["response_cross_boundary_any_inside"],
            "cross_outside": s["response_cross_boundary_any_outside"],
            "cross_ratio": s["response_cross_boundary_any_ratio_inside_outside"],
            "auc_inside_by_abs_dP": s["auc_inside_by_response_abs_dP_mean"],
            "auc_inside_by_dir_dP": s["auc_inside_by_response_to_conflict_dP_mean"],
            "pass_abs_response_lift_1p10": s["pass_abs_response_lift_1p10"],
            "pass_directional_response_lift_1p10": s["pass_directional_response_lift_1p10"],
            "pass_cross_boundary_lift_1p10": s["pass_cross_boundary_lift_1p10"],
            "pass_cm3b_real_response": s["pass_cm3b_real_response"],
        })
    df_sum = pd.DataFrame(rows)
    df_sum.to_csv(SAVE_DIR / "cm3b_model_summary.csv", index=False, encoding="utf-8-sig")
    write_json(SAVE_DIR / "cm3b_overall_summary.json", summaries)

    print("\n==== CM-3B Summary ====")
    print(df_sum.to_string(index=False))
    print(f"\nSaved to: {SAVE_DIR.resolve()}")


if __name__ == "__main__":
    main()
