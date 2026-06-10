# -*- coding: utf-8 -*-
"""
PSG-1C Real-W Residual / Anisotropy Audit v1.0

Small validation experiment.

Goal:
ObservedTopKPreservation = PSG_GeometricReadback + RealWResidual ?

Compares real W variants against Gaussian / norm-matched / covariance-diagonal nulls.

Run:
python psg1c_realw_residual_anisotropy_audit_v1_0.py
"""

import time
import random
import warnings
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import pairwise_distances
from transformers import AutoModelForCausalLM, AutoTokenizer


EXPERIMENT_ID = "PSG-1C_RealW_Residual_Anisotropy_Audit_v1_0"

MODEL_PATH = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"
OUT_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\psg1c_realw_residual_outputs")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SEED = 42
MAX_PROMPTS = 32
MAX_LENGTH = 256

LAYER_IDS = [12, 18, 22, 26]
K_LIST = [5000]
N_GAUSSIAN_REPEATS = 3

SELECTION_MODES = ["topk", "randomk"]

MATRIX_VARIANTS = [
    "real_W_raw",
    "real_W_row_norm",
    "real_W_center_row_norm",
    "real_W_whiten_row_norm",
    "real_W_row_permuted",
    "gaussian_iid",
    "gaussian_norm_matched",
    "gaussian_cov_matched_diag",
]


def package_available(name):
    return importlib.util.find_spec(name) is not None


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
        "trajectory control", "constraint field", "semantic manifold", "hallucination", "Cbit", "phase transition"
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


def cosine_distance_matrix(x):
    x = np.asarray(x, dtype=np.float32)
    return pairwise_distances(x, metric="cosine")


def mean_jaccard(idx):
    vals = []
    for i in range(idx.shape[0]):
        si = set(idx[i].tolist())
        for j in range(i + 1, idx.shape[0]):
            sj = set(idx[j].tolist())
            vals.append(len(si & sj) / max(1, len(si | sj)))
    return float(np.mean(vals)) if vals else np.nan


def select_indices(scores, k, selection, rng):
    n, v = scores.shape
    k = min(k, v)
    if selection == "topk":
        return np.argpartition(-scores, kth=k-1, axis=1)[:, :k]
    if selection == "randomk":
        return np.stack([rng.choice(v, size=k, replace=False) for _ in range(n)], axis=0)
    raise ValueError(f"Unknown selection: {selection}")


def sort_idx_by_score(idx, scores):
    selected_scores = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-selected_scores, axis=1)
    return np.take_along_axis(idx, order, axis=1)


def whiten_rows_svd(W, eps=1e-5):
    X = W.astype(np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    cov = (X.T @ X) / max(1, X.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, eps)
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return (X @ inv_sqrt).astype(np.float32)


def load_model_and_tokenizer():
    print(f"[INFO] MODEL_PATH={MODEL_PATH}")
    for dep in ["tiktoken", "sentencepiece", "protobuf", "tokenizers"]:
        print(f"[INFO] {dep}: {'OK' if package_available(dep) else 'MISSING'}")

    errors = []
    tokenizer = None
    for use_fast in [True, False]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATH,
                local_files_only=True,
                trust_remote_code=True,
                use_fast=use_fast,
            )
            print(f"[INFO] tokenizer loaded use_fast={use_fast}: {type(tokenizer)}")
            break
        except Exception as e:
            errors.append(f"use_fast={use_fast}: {repr(e)}")
            tokenizer = None

    if tokenizer is None:
        raise RuntimeError("Tokenizer load failed:\n" + "\n".join(errors))

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
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
        raise RuntimeError("No output embedding / lm_head found.")
    return emb.weight.detach().float().cpu().numpy().astype(np.float32)


@torch.no_grad()
def extract_hidden_states(model, tokenizer, prompts):
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(DEVICE)

    out = model(
        **encoded,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    last_idx = encoded["attention_mask"].sum(dim=1) - 1
    batch_idx = torch.arange(len(prompts), device=DEVICE)

    n_layers_total = len(out.hidden_states) - 1
    valid_layers = [l for l in LAYER_IDS if 0 <= l <= n_layers_total]
    print(f"[INFO] model layers={n_layers_total}; valid_layers={valid_layers}")

    hidden_by_layer = {}
    for l in valid_layers:
        H = out.hidden_states[l][batch_idx, last_idx, :].detach().float().cpu().numpy().astype(np.float32)
        hidden_by_layer[l] = H

    return hidden_by_layer, valid_layers


def build_matrix_variant(W_raw, variant, repeat_id, rng):
    vocab, d = W_raw.shape

    if variant == "real_W_raw":
        return W_raw.copy()

    if variant == "real_W_row_norm":
        return l2_normalize(W_raw).astype(np.float32)

    if variant == "real_W_center_row_norm":
        X = W_raw - W_raw.mean(axis=0, keepdims=True)
        return l2_normalize(X).astype(np.float32)

    if variant == "real_W_whiten_row_norm":
        Xw = whiten_rows_svd(W_raw)
        return l2_normalize(Xw).astype(np.float32)

    if variant == "real_W_row_permuted":
        perm = rng.permutation(vocab)
        return W_raw[perm].copy()

    if variant == "gaussian_iid":
        X = rng.standard_normal((vocab, d)).astype(np.float32)
        target_norm = np.linalg.norm(W_raw, axis=1).mean()
        return l2_normalize(X) * target_norm

    if variant == "gaussian_norm_matched":
        X = rng.standard_normal((vocab, d)).astype(np.float32)
        X = l2_normalize(X)
        norms = np.linalg.norm(W_raw, axis=1).astype(np.float32)
        sampled_norms = rng.choice(norms, size=vocab, replace=True).reshape(-1, 1)
        return X * sampled_norms

    if variant == "gaussian_cov_matched_diag":
        mu = W_raw.mean(axis=0, keepdims=True).astype(np.float32)
        sigma = W_raw.std(axis=0, keepdims=True).astype(np.float32)
        sigma = np.maximum(sigma, 1e-6)
        X = rng.standard_normal((vocab, d)).astype(np.float32) * sigma + mu
        return X.astype(np.float32)

    raise ValueError(f"Unknown matrix variant: {variant}")


def run_condition(H, R, layer, k, variant, selection, repeat_id):
    rng = np.random.default_rng(SEED + 1000 * repeat_id + 17 * layer + k + hash((variant, selection)) % 997)

    t0 = time.time()

    scores = H @ R.T
    idx = select_indices(scores, k, selection, rng)
    idx = sort_idx_by_score(idx, scores)

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
    self_cos_mean = float(self_cos.mean())
    self_vs_off_margin = self_cos_mean - off_mean

    return {
        "experiment_id": EXPERIMENT_ID,
        "layer": layer,
        "k": k,
        "matrix_variant": variant,
        "selection": selection,
        "repeat_id": repeat_id,
        "topo_pearson": pear,
        "topo_spearman": spear,
        "abs_topo_spearman": abs(spear) if np.isfinite(spear) else np.nan,
        "self_cos_mean": self_cos_mean,
        "off_cos_mean": off_mean,
        "self_vs_off_margin": self_vs_off_margin,
        "z_norm_mean": float(np.linalg.norm(Z, axis=1).mean()),
        "mean_selected_jaccard": mean_jaccard(idx),
        "elapsed_sec": time.time() - t0,
    }


def build_verdict(summary):
    lines = []
    lines.append("=" * 80)
    lines.append("PSG-1C Real-W Residual / Anisotropy Audit Verdict")
    lines.append("=" * 80)

    if summary.empty:
        lines.append("NO_VALID_ROWS")
        return "\n".join(lines)

    variants = MATRIX_VARIANTS

    def mean_variant(v, selection="topk"):
        sub = summary[(summary["matrix_variant"] == v) & (summary["selection"] == selection)]
        if sub.empty:
            return np.nan
        return float(sub["topo_spearman"].mean())

    means = {v: mean_variant(v, "topk") for v in variants}

    lines.append("\nVariant means, selection=topk:")
    for v in variants:
        lines.append(f"{v}: mean_topo_spearman={means[v]:.4f}")

    lines.append("\nRandomK controls:")
    for v in variants:
        lines.append(f"{v}: randomk_mean={mean_variant(v, 'randomk'):.4f}")

    real_best = np.nanmax([
        means["real_W_raw"],
        means["real_W_row_norm"],
        means["real_W_center_row_norm"],
        means["real_W_whiten_row_norm"],
    ])
    null_best = np.nanmax([
        means["real_W_row_permuted"],
        means["gaussian_iid"],
        means["gaussian_norm_matched"],
        means["gaussian_cov_matched_diag"],
    ])

    lines.append("\nMain contrasts:")
    lines.append(f"real_best={real_best:.4f}")
    lines.append(f"null_best={null_best:.4f}")
    lines.append(f"real_best_minus_null_best={real_best - null_best:.4f}")
    lines.append(f"real_raw_minus_row_permuted={means['real_W_raw'] - means['real_W_row_permuted']:.4f}")
    lines.append(f"real_raw_minus_gaussian_iid={means['real_W_raw'] - means['gaussian_iid']:.4f}")
    lines.append(f"real_raw_minus_gaussian_cov_diag={means['real_W_raw'] - means['gaussian_cov_matched_diag']:.4f}")
    lines.append(f"whitened_real_minus_gaussian_iid={means['real_W_whiten_row_norm'] - means['gaussian_iid']:.4f}")

    lines.append("\nTop 20 conditions:")
    top = summary.sort_values("topo_spearman", ascending=False).head(20)
    for _, r in top.iterrows():
        lines.append(
            f"layer={int(r['layer'])} k={int(r['k'])} variant={r['matrix_variant']} "
            f"select={r['selection']} spearman={r['topo_spearman']:.4f} "
            f"margin={r['self_vs_off_margin']:.4f}"
        )

    lines.append("\nInterpretation:")
    diff = real_best - null_best
    if diff > 0.10:
        lines.append("PASS-RESIDUAL: Real W variants exceed matched PSG nulls by a meaningful margin.")
    elif diff > 0.03:
        lines.append("WEAK-RESIDUAL: Real W variants slightly exceed matched PSG nulls.")
    elif diff > -0.03:
        lines.append("NO-STRONG-RESIDUAL: Real W is comparable to matched PSG nulls.")
    else:
        lines.append("NEGATIVE-RESIDUAL: Matched PSG nulls exceed real W variants.")

    if means["real_W_raw"] > means["real_W_row_permuted"] + 0.05:
        lines.append("SUPPORT-SEMANTIC-ORDER: Row permutation hurts real W readback.")
    else:
        lines.append("NO-ROW-ORDER-SUPPORT: Row permutation does not materially hurt readback.")

    if means["real_W_whiten_row_norm"] > means["real_W_raw"] + 0.05:
        lines.append("SUPPORT-ANISOTROPY-CONFOUND: Whitening improves real W readback.")
    elif means["real_W_raw"] > means["real_W_whiten_row_norm"] + 0.05:
        lines.append("SUPPORT-RAW-STRUCTURE: Raw W contributes more than whitened geometry.")
    else:
        lines.append("WHITENING-NEUTRAL: Whitening does not materially change real W readback.")

    lines.append("\nSuggested theory update:")
    lines.append("ObservedTopKPreservation = PSG_GeometricReadback + RealWResidual.")
    lines.append("=" * 80)
    return "\n".join(lines)


def main():
    warnings.filterwarnings("ignore")
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    prompts = build_prompts()
    pd.DataFrame({"prompt": prompts}).to_csv(OUT_DIR / "psg1c_prompts.csv", index=False, encoding="utf-8-sig")

    model, tokenizer = load_model_and_tokenizer()
    hidden_by_layer, valid_layers = extract_hidden_states(model, tokenizer, prompts)

    print("[INFO] extracting output W...")
    W_raw = get_output_W(model)
    print(f"[INFO] W_raw shape={W_raw.shape}")

    rows = []
    start = time.time()

    variant_cache = {}
    rng_base = np.random.default_rng(SEED)

    for variant in MATRIX_VARIANTS:
        if variant.startswith("real_W"):
            print(f"[INFO] building deterministic variant: {variant}")
            variant_cache[(variant, 0)] = build_matrix_variant(W_raw, variant, 0, rng_base)

    total = 0
    for variant in MATRIX_VARIANTS:
        repeats = N_GAUSSIAN_REPEATS if variant.startswith("gaussian") else 1
        total += len(valid_layers) * len(K_LIST) * len(SELECTION_MODES) * repeats

    done = 0

    for layer in valid_layers:
        H = hidden_by_layer[layer]
        pd.DataFrame(cosine_distance_matrix(H)).to_csv(OUT_DIR / f"hidden_distance_layer_{layer}.csv", index=False)

        for k in K_LIST:
            for variant in MATRIX_VARIANTS:
                repeats = N_GAUSSIAN_REPEATS if variant.startswith("gaussian") else 1

                for repeat_id in range(repeats):
                    if (variant, repeat_id) in variant_cache:
                        R = variant_cache[(variant, repeat_id)]
                    else:
                        rng = np.random.default_rng(SEED + 911 * repeat_id + hash(variant) % 997)
                        print(f"[INFO] building variant: {variant}, repeat={repeat_id}")
                        R = build_matrix_variant(W_raw, variant, repeat_id, rng)

                    for selection in SELECTION_MODES:
                        try:
                            rows.append(run_condition(H, R, layer, k, variant, selection, repeat_id))
                        except Exception as e:
                            rows.append({
                                "experiment_id": EXPERIMENT_ID,
                                "layer": layer,
                                "k": k,
                                "matrix_variant": variant,
                                "selection": selection,
                                "repeat_id": repeat_id,
                                "error": repr(e),
                            })

                        done += 1
                        print(f"[INFO] progress {done}/{total}; rows={len(rows)}; elapsed={time.time() - start:.1f}s")

    raw = pd.DataFrame(rows)
    raw_path = OUT_DIR / "psg1c_realw_residual_raw.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")

    ok = raw[raw.get("topo_spearman", pd.Series(index=raw.index)).notna()].copy()
    group_cols = ["layer", "k", "matrix_variant", "selection"]
    numeric_cols = [
        "topo_pearson", "topo_spearman", "abs_topo_spearman",
        "self_cos_mean", "off_cos_mean", "self_vs_off_margin",
        "z_norm_mean", "mean_selected_jaccard", "elapsed_sec"
    ]
    summary = ok.groupby(group_cols, dropna=False)[numeric_cols].mean().reset_index()
    summary["n_repeats"] = ok.groupby(group_cols, dropna=False).size().values

    summary_path = OUT_DIR / "psg1c_realw_residual_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    for col in ["matrix_variant", "selection", "layer", "k"]:
        view = summary.groupby(col)["topo_spearman"].agg(["mean", "median", "max", "std", "count"]).reset_index()
        view = view.sort_values("mean", ascending=False)
        view.to_csv(OUT_DIR / f"psg1c_factor_view_{col}.csv", index=False, encoding="utf-8-sig")

    summary.sort_values("topo_spearman", ascending=False).head(100).to_csv(
        OUT_DIR / "psg1c_top100_conditions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    verdict = build_verdict(summary)
    verdict_path = OUT_DIR / "psg1c_verdict.txt"
    verdict_path.write_text(verdict, encoding="utf-8")

    print("\n" + verdict)
    print(f"\n[OK] raw: {raw_path}")
    print(f"[OK] summary: {summary_path}")
    print(f"[OK] verdict: {verdict_path}")


if __name__ == "__main__":
    main()
