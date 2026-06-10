# -*- coding: utf-8 -*-
"""
CM-PSG-1/2 Cross-Model PSG Correction Audit

Purpose
-------
Supplementary experiment for Paper2 cross-model section.

This script audits whether cross-model TopK topology preservation is explained by:
1. real W
2. row-permuted W
3. Gaussian projection-selection null

It computes:
    TopoReal_m
    TopoRowPerm_m
    TopoGaussian_m
    ResidualGaussian_m = TopoReal_m - TopoGaussian_m
    RowIdentityEffect_m = TopoReal_m - TopoRowPerm_m

Interpretation
--------------
If ResidualGaussian≈0 and RowIdentityEffect≈0:
    topology preservation is mostly PSG readback, not learned semantic row identity.

If ResidualGaussian>0.05 consistently:
    real W has positive topology residual beyond PSG.

Default local model paths
-------------------------
Qwen:
D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main

Llama:
D:\model\Llama-3.2-1B-Instruct

Gemma:
D:\model\gemma-2-2b-it

Run
---
python cm_psg_1_2_crossmodel_topology_correction.py
"""

import os
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import pairwise_distances
from transformers import AutoTokenizer, AutoModelForCausalLM


EXPERIMENT_ID = "CM-PSG-1_2_CrossModel_Topology_Correction_v1_0"

OUT_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\cm_psg_1_2_outputs")

MODEL_CONFIGS = {
    "qwen": {
        "path": r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main",
        "layers": [6, 12, 18, 22, 26],
    },
    "llama": {
        "path": r"D:\model\Llama-3.2-1B-Instruct",
        "layers": [1, 3, 6, 10, 14],
    },
    "gemma": {
        "path": r"D:\model\gemma-2-2b-it",
        "layers": [1, 4, 8, 16, 22],
    },
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SEED = 42
MAX_PROMPTS = 32
MAX_LENGTH = 256
K_LIST = [1000, 5000]
N_GAUSSIAN_REPEATS = 3

MATRIX_VARIANTS = ["real_W", "row_permuted_W", "gaussian_iid"]
SELECTION_MODES = ["topk", "randomk"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_prompts():
    concepts = [
        "Paris", "Newton", "Einstein", "photosynthesis", "democracy", "inflation",
        "semiconductor", "gradient descent", "black hole", "evolution", "Shakespeare",
        "supply chain", "quantum mechanics", "climate change", "neural network", "volcano",
        "antibiotics", "electric field", "Renaissance", "machine translation", "compiler",
        "Bayesian inference", "game theory", "protein folding", "memory retrieval", "operator routing",
        "trajectory control", "constraint field", "semantic manifold", "hallucination", "Cbit", "phase transition",
    ]
    operators = [
        "define the core idea of",
        "explain the causal mechanism behind",
        "give a simple example of",
        "compare two competing interpretations of",
    ]
    prompts = []
    for c in concepts:
        for op in operators:
            prompts.append(f"{op} {c}. Answer in one concise paragraph.")
    return prompts[:MAX_PROMPTS]


def l2_normalize(x, axis=-1, eps=1e-9):
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(denom, eps)


def cosine_distance_matrix(x):
    return pairwise_distances(np.asarray(x, dtype=np.float32), metric="cosine")


def upper_tri_values(mat):
    idx = np.triu_indices(mat.shape[0], k=1)
    return mat[idx]


def safe_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4:
        return np.nan, np.nan
    if np.std(x[mask]) < 1e-12 or np.std(y[mask]) < 1e-12:
        return np.nan, np.nan
    return float(pearsonr(x[mask], y[mask]).statistic), float(spearmanr(x[mask], y[mask]).statistic)


def load_model(model_key, model_path):
    print(f"[INFO] Loading {model_key}: {model_path}")
    tokenizer = None
    errors = []
    for use_fast in [True, False]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True,
                use_fast=use_fast,
            )
            print(f"[INFO] tokenizer loaded use_fast={use_fast}")
            break
        except Exception as e:
            errors.append(f"use_fast={use_fast}: {repr(e)}")
            tokenizer = None
    if tokenizer is None:
        raise RuntimeError("Tokenizer failed:\n" + "\n".join(errors))

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        device_map=None,
    ).to(DEVICE)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_output_W(model):
    emb = model.get_output_embeddings()
    if emb is None:
        raise RuntimeError("No output embedding found.")
    return emb.weight.detach().float().cpu().numpy().astype(np.float32)


@torch.no_grad()
def extract_hidden(model, tokenizer, prompts, requested_layers):
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(DEVICE)

    out = model(**encoded, output_hidden_states=True, use_cache=False, return_dict=True)
    last_idx = encoded["attention_mask"].sum(dim=1) - 1
    batch_idx = torch.arange(len(prompts), device=DEVICE)

    n_layers_total = len(out.hidden_states) - 1
    layers = [l for l in requested_layers if 0 <= l <= n_layers_total]
    if not layers:
        layers = sorted(set([1, n_layers_total // 4, n_layers_total // 2, int(n_layers_total * 0.75), n_layers_total - 1]))

    hidden_by_layer = {}
    for l in layers:
        H = out.hidden_states[l][batch_idx, last_idx, :].detach().float().cpu().numpy().astype(np.float32)
        hidden_by_layer[l] = H

    print(f"[INFO] layers total={n_layers_total}, using={layers}")
    return hidden_by_layer, layers


def make_matrix(W, variant, rng):
    V, d = W.shape
    if variant == "real_W":
        return W.copy()
    if variant == "row_permuted_W":
        return W[rng.permutation(V)].copy()
    if variant == "gaussian_iid":
        X = rng.standard_normal((V, d)).astype(np.float32)
        target_norm = np.linalg.norm(W, axis=1).mean()
        return l2_normalize(X) * target_norm
    raise ValueError(variant)


def select_indices(scores, k, mode, rng):
    n, V = scores.shape
    k = min(k, V)
    if mode == "topk":
        idx = np.argpartition(-scores, kth=k-1, axis=1)[:, :k]
        sc = np.take_along_axis(scores, idx, axis=1)
        order = np.argsort(-sc, axis=1)
        return np.take_along_axis(idx, order, axis=1)
    if mode == "randomk":
        return np.stack([rng.choice(V, size=k, replace=False) for _ in range(n)], axis=0)
    raise ValueError(mode)


def run_condition(model_key, H, R, layer, k, variant, selection, repeat_id):
    rng = np.random.default_rng(SEED + 1000 * repeat_id + 17 * layer + k + hash((model_key, variant, selection)) % 997)

    t0 = time.time()
    scores = H @ R.T
    idx = select_indices(scores, k, selection, rng)
    Z = R[idx].mean(axis=1)

    D_H = cosine_distance_matrix(H)
    D_Z = cosine_distance_matrix(Z)
    pear, spear = safe_corr(upper_tri_values(D_H), upper_tri_values(D_Z))

    Hn = l2_normalize(H)
    Zn = l2_normalize(Z)
    sim = Zn @ Hn.T
    self_cos = np.diag(sim)
    off_mask = ~np.eye(sim.shape[0], dtype=bool)
    off_mean = float(sim[off_mask].mean()) if off_mask.sum() else np.nan

    return {
        "experiment_id": EXPERIMENT_ID,
        "model": model_key,
        "layer": layer,
        "k": k,
        "matrix_variant": variant,
        "selection": selection,
        "repeat_id": repeat_id,
        "topo_pearson": pear,
        "topo_spearman": spear,
        "self_cos_mean": float(self_cos.mean()),
        "off_cos_mean": off_mean,
        "self_vs_off_margin": float(self_cos.mean() - off_mean),
        "z_norm_mean": float(np.linalg.norm(Z, axis=1).mean()),
        "elapsed_sec": time.time() - t0,
    }


def build_verdict(summary):
    lines = []
    lines.append("=" * 80)
    lines.append("CM-PSG-1/2 Cross-Model Topology Correction Verdict")
    lines.append("=" * 80)

    if summary.empty:
        lines.append("NO_VALID_ROWS")
        return "\n".join(lines)

    lines.append("\nPer-model topk means:")
    rows = []
    for model in sorted(summary["model"].unique()):
        sub = summary[(summary["model"] == model) & (summary["selection"] == "topk")]
        def m(v):
            ss = sub[sub["matrix_variant"] == v]
            return float(ss["topo_spearman"].mean()) if not ss.empty else np.nan
        real = m("real_W")
        rowp = m("row_permuted_W")
        gaus = m("gaussian_iid")
        rows.append({
            "model": model,
            "TopoReal": real,
            "TopoRowPerm": rowp,
            "TopoGaussian": gaus,
            "ResidualGaussian": real - gaus,
            "RowIdentityEffect": real - rowp,
        })
        lines.append(
            f"{model}: real={real:.4f}, rowperm={rowp:.4f}, gaussian={gaus:.4f}, "
            f"residual(real-gaussian)={real-gaus:.4f}, row_identity={real-rowp:.4f}"
        )

    df = pd.DataFrame(rows)
    lines.append("\nInterpretation:")
    mean_resid = float(df["ResidualGaussian"].mean())
    mean_row = float(df["RowIdentityEffect"].mean())
    lines.append(f"mean_residual_gaussian={mean_resid:.4f}")
    lines.append(f"mean_row_identity_effect={mean_row:.4f}")

    if mean_resid > 0.05:
        lines.append("PASS-RESIDUAL: Cross-model real W exceeds Gaussian PSG null.")
    elif mean_resid > -0.03:
        lines.append("NO-STRONG-RESIDUAL: Cross-model real W is comparable to Gaussian PSG null.")
    else:
        lines.append("NEGATIVE-RESIDUAL: Gaussian PSG null exceeds real W on average.")

    if abs(mean_row) < 0.03:
        lines.append("NO-ROW-IDENTITY: Row identity does not materially affect topology preservation.")
    elif mean_row > 0.05:
        lines.append("ROW-IDENTITY-SUPPORT: Row permutation hurts topology preservation.")
    else:
        lines.append("MIXED-ROW-IDENTITY: Row identity effect is nonzero but weak/mixed.")

    lines.append("\nPaper2 use:")
    lines.append("Use these residuals to replace raw cross-model TopK topology claims with PSG-corrected claims.")
    lines.append("=" * 80)
    return "\n".join(lines)


def main():
    warnings.filterwarnings("ignore")
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    prompts = build_prompts()
    pd.DataFrame({"prompt": prompts}).to_csv(OUT_DIR / "cm_psg_prompts.csv", index=False, encoding="utf-8-sig")

    all_rows = []

    for model_key, cfg in MODEL_CONFIGS.items():
        model_path = cfg["path"]
        if not Path(model_path).exists():
            print(f"[WARN] skip {model_key}, path not found: {model_path}")
            continue

        model, tokenizer = load_model(model_key, model_path)
        W = get_output_W(model)
        hidden_by_layer, layers = extract_hidden(model, tokenizer, prompts, cfg["layers"])

        # Free model after hidden extraction.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        matrix_cache = {}
        for variant in ["real_W", "row_permuted_W"]:
            rng = np.random.default_rng(SEED + hash((model_key, variant)) % 997)
            matrix_cache[(variant, 0)] = make_matrix(W, variant, rng)

        for layer in layers:
            H = hidden_by_layer[layer]
            for k in K_LIST:
                for variant in MATRIX_VARIANTS:
                    repeats = N_GAUSSIAN_REPEATS if variant == "gaussian_iid" else 1
                    for rep in range(repeats):
                        if (variant, rep) in matrix_cache:
                            R = matrix_cache[(variant, rep)]
                        else:
                            rng = np.random.default_rng(SEED + 911 * rep + hash((model_key, variant)) % 997)
                            R = make_matrix(W, variant, rng)

                        for selection in SELECTION_MODES:
                            try:
                                all_rows.append(run_condition(model_key, H, R, layer, k, variant, selection, rep))
                            except Exception as e:
                                all_rows.append({
                                    "experiment_id": EXPERIMENT_ID,
                                    "model": model_key,
                                    "layer": layer,
                                    "k": k,
                                    "matrix_variant": variant,
                                    "selection": selection,
                                    "repeat_id": rep,
                                    "error": repr(e),
                                })
                            print(f"[INFO] rows={len(all_rows)} model={model_key} layer={layer} k={k} variant={variant} sel={selection}")

        del tokenizer, W, hidden_by_layer, matrix_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    raw = pd.DataFrame(all_rows)
    raw_path = OUT_DIR / "cm_psg_1_2_raw.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")

    ok = raw[raw.get("topo_spearman", pd.Series(index=raw.index)).notna()].copy()
    group_cols = ["model", "layer", "k", "matrix_variant", "selection"]
    numeric_cols = ["topo_pearson", "topo_spearman", "self_cos_mean", "off_cos_mean", "self_vs_off_margin", "z_norm_mean", "elapsed_sec"]
    summary = ok.groupby(group_cols, dropna=False)[numeric_cols].mean().reset_index()
    summary["n_repeats"] = ok.groupby(group_cols, dropna=False).size().values

    summary_path = OUT_DIR / "cm_psg_1_2_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    for col in ["model", "matrix_variant", "selection", "k", "layer"]:
        view = summary.groupby(col)["topo_spearman"].agg(["mean", "median", "max", "std", "count"]).reset_index()
        view = view.sort_values("mean", ascending=False)
        view.to_csv(OUT_DIR / f"cm_psg_1_2_factor_view_{col}.csv", index=False, encoding="utf-8-sig")

    correction_rows = []
    for model in sorted(summary["model"].unique()):
        sub = summary[(summary["model"] == model) & (summary["selection"] == "topk")]
        for k in sorted(sub["k"].unique()):
            sk = sub[sub["k"] == k]
            real = sk[sk["matrix_variant"] == "real_W"]["topo_spearman"].mean()
            rowp = sk[sk["matrix_variant"] == "row_permuted_W"]["topo_spearman"].mean()
            gaus = sk[sk["matrix_variant"] == "gaussian_iid"]["topo_spearman"].mean()
            correction_rows.append({
                "model": model,
                "k": k,
                "TopoReal": real,
                "TopoRowPerm": rowp,
                "TopoGaussian": gaus,
                "ResidualGaussian": real - gaus,
                "RowIdentityEffect": real - rowp,
            })
    pd.DataFrame(correction_rows).to_csv(OUT_DIR / "cm_psg_1_2_residual_table.csv", index=False, encoding="utf-8-sig")

    verdict = build_verdict(summary)
    verdict_path = OUT_DIR / "cm_psg_1_2_verdict.txt"
    verdict_path.write_text(verdict, encoding="utf-8")

    print("\n" + verdict)
    print(f"\n[OK] raw: {raw_path}")
    print(f"[OK] summary: {summary_path}")
    print(f"[OK] verdict: {verdict_path}")


if __name__ == "__main__":
    main()
