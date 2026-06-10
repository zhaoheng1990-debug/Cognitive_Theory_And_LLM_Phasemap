# -*- coding: utf-8 -*-
"""
CM-PSG-3 Semantic Content Residual Audit

Purpose
-------
Supplementary experiment for Paper2.

PSG-1/2 showed topology preservation can be explained by projection-selection
geometry. This script tests a different claim:

    learned W may still provide semantic content / interpretability.

It compares whether real_W TopK token texts are more predictive of prompt-family
labels than row-permuted W and random token controls.

This does NOT test topology preservation.
It tests semantic content residual.

Default local model paths
-------------------------
Same as CM-PSG-1/2.

Run
---
python cm_psg_3_semantic_content_residual.py
"""

import re
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM


EXPERIMENT_ID = "CM-PSG-3_Semantic_Content_Residual_v1_0"

OUT_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\cm_psg_3_semantic_outputs")

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
MAX_LENGTH = 256

K_TOKENS_FOR_TEXT = 80
K_FOR_SELECTION = 500

MATRIX_VARIANTS = ["real_W", "row_permuted_W", "random_tokens"]

# Prompt family labels. Keep labels semantically distinct but not too easy.
FAMILIES = {
    "geography": ["Paris", "France", "river delta", "mountain range", "capital city", "island nation"],
    "science": ["photosynthesis", "gravity", "electric field", "protein folding", "evolution", "black hole"],
    "computing": ["compiler", "gradient descent", "neural network", "database index", "operating system", "encryption"],
    "economics": ["inflation", "supply chain", "market liquidity", "interest rate", "trade deficit", "monopoly"],
    "literature": ["Shakespeare", "metaphor", "narrative voice", "Renaissance poetry", "tragedy", "literary realism"],
    "cognition": ["memory retrieval", "operator routing", "trajectory control", "constraint field", "semantic manifold", "hallucination"],
}

TASKS = [
    "define the core idea of",
    "explain the mechanism behind",
    "give a simple example of",
    "compare two interpretations of",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_labeled_prompts():
    rows = []
    group_id = 0
    for family, concepts in FAMILIES.items():
        for concept in concepts:
            for task in TASKS:
                rows.append({
                    "prompt": f"{task} {concept}. Answer in one concise paragraph.",
                    "family": family,
                    "concept": concept,
                    "task": task,
                    "group_id": group_id,
                })
            group_id += 1
    return pd.DataFrame(rows)


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


def decode_token(tokenizer, token_id):
    try:
        txt = tokenizer.decode([int(token_id)], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    except Exception:
        txt = ""
    txt = txt.replace("\n", " ").replace("\r", " ").strip()
    # Remove extreme whitespace.
    txt = re.sub(r"\s+", " ", txt)
    return txt


def topk_token_texts(H, W, tokenizer, variant, rng):
    n, d = H.shape
    V = W.shape[0]

    if variant == "real_W":
        R = W
        scores = H @ R.T
        idx = np.argpartition(-scores, kth=K_FOR_SELECTION-1, axis=1)[:, :K_FOR_SELECTION]
        # Sort and keep first K_TOKENS_FOR_TEXT.
        sc = np.take_along_axis(scores, idx, axis=1)
        order = np.argsort(-sc, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)[:, :K_TOKENS_FOR_TEXT]

    elif variant == "row_permuted_W":
        perm = rng.permutation(V)
        R = W[perm]
        scores = H @ R.T
        idx_local = np.argpartition(-scores, kth=K_FOR_SELECTION-1, axis=1)[:, :K_FOR_SELECTION]
        sc = np.take_along_axis(scores, idx_local, axis=1)
        order = np.argsort(-sc, axis=1)
        idx_local = np.take_along_axis(idx_local, order, axis=1)[:, :K_TOKENS_FOR_TEXT]
        # Decode the permuted token ids actually selected in original vocabulary index.
        idx = perm[idx_local]

    elif variant == "random_tokens":
        idx = np.stack([rng.choice(V, size=K_TOKENS_FOR_TEXT, replace=False) for _ in range(n)], axis=0)

    else:
        raise ValueError(variant)

    docs = []
    for row in idx:
        toks = [decode_token(tokenizer, int(t)) for t in row]
        toks = [t for t in toks if t and len(t) <= 40]
        docs.append(" ".join(toks))
    return docs, idx


def evaluate_docs_cv(df_docs):
    # Need enough samples and labels.
    y = df_docs["family"].values
    groups = df_docs["concept"].values
    docs = df_docs["topk_text"].fillna("").astype(str).values

    if len(set(y)) < 2:
        return {"cv_acc": np.nan, "cv_macro_f1": np.nan, "note": "not enough labels"}

    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        min_df=1,
        max_features=5000,
        ngram_range=(1, 2),
    )
    X = vectorizer.fit_transform(docs)

    clf = LogisticRegression(max_iter=2000, solver="liblinear")
    n_splits = min(5, len(set(groups)))
    if n_splits < 2:
        return {"cv_acc": np.nan, "cv_macro_f1": np.nan, "note": "not enough groups"}

    gkf = GroupKFold(n_splits=n_splits)
    accs = []
    f1s = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], pred))
        f1s.append(f1_score(y[test_idx], pred, average="macro"))

    return {
        "cv_acc": float(np.mean(accs)),
        "cv_macro_f1": float(np.mean(f1s)),
        "n_docs": int(len(docs)),
        "n_labels": int(len(set(y))),
        "n_groups": int(len(set(groups))),
    }


def build_verdict(metrics):
    lines = []
    lines.append("=" * 80)
    lines.append("CM-PSG-3 Semantic Content Residual Verdict")
    lines.append("=" * 80)

    if metrics.empty:
        lines.append("NO_VALID_ROWS")
        return "\n".join(lines)

    lines.append("\nPer-model / variant / layer CV:")
    top = metrics.sort_values("cv_macro_f1", ascending=False).head(30)
    for _, r in top.iterrows():
        lines.append(
            f"model={r['model']} layer={int(r['layer'])} variant={r['matrix_variant']} "
            f"acc={r['cv_acc']:.4f} macroF1={r['cv_macro_f1']:.4f}"
        )

    lines.append("\nPer-model best contrasts:")
    rows = []
    for model in sorted(metrics["model"].unique()):
        sub = metrics[metrics["model"] == model]
        def best(v):
            ss = sub[sub["matrix_variant"] == v]
            return float(ss["cv_macro_f1"].max()) if not ss.empty else np.nan
        real = best("real_W")
        rowp = best("row_permuted_W")
        rand = best("random_tokens")
        rows.append({
            "model": model,
            "real_best": real,
            "rowperm_best": rowp,
            "random_best": rand,
            "semantic_residual_vs_rowperm": real - rowp,
            "semantic_residual_vs_random": real - rand,
        })
        lines.append(
            f"{model}: real_best={real:.4f}, rowperm_best={rowp:.4f}, random_best={rand:.4f}, "
            f"real-rowperm={real-rowp:.4f}, real-random={real-rand:.4f}"
        )

    df = pd.DataFrame(rows)
    mean_row = float(df["semantic_residual_vs_rowperm"].mean())
    mean_rand = float(df["semantic_residual_vs_random"].mean())
    lines.append("\nInterpretation:")
    lines.append(f"mean_real_minus_rowperm={mean_row:.4f}")
    lines.append(f"mean_real_minus_random={mean_rand:.4f}")

    if mean_row > 0.10 and mean_rand > 0.10:
        lines.append("PASS-SEMANTIC-RESIDUAL: real W TopK content carries semantic label information beyond controls.")
    elif mean_rand > 0.10:
        lines.append("MIXED: real W exceeds random tokens but not clearly row permutation; semantic content may be partly token-frequency/surface.")
    else:
        lines.append("NO-SEMANTIC-RESIDUAL: current token-text content audit does not support strong semantic residual.")

    lines.append("\nPaper2 use:")
    lines.append("Use this to separate geometry readback from semantic interpretability/content.")
    lines.append("=" * 80)
    return "\n".join(lines)


def main():
    warnings.filterwarnings("ignore")
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    prompt_df = build_labeled_prompts()
    prompt_df.to_csv(OUT_DIR / "cm_psg_3_labeled_prompts.csv", index=False, encoding="utf-8-sig")

    all_doc_rows = []
    all_metric_rows = []

    for model_key, cfg in MODEL_CONFIGS.items():
        model_path = cfg["path"]
        if not Path(model_path).exists():
            print(f"[WARN] skip {model_key}, path not found: {model_path}")
            continue

        model, tokenizer = load_model(model_key, model_path)
        W = get_output_W(model)
        hidden_by_layer, layers = extract_hidden(model, tokenizer, prompt_df["prompt"].tolist(), cfg["layers"])

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for layer in layers:
            H = hidden_by_layer[layer]
            for variant in MATRIX_VARIANTS:
                rng = np.random.default_rng(SEED + hash((model_key, layer, variant)) % 10007)
                try:
                    docs, idx = topk_token_texts(H, W, tokenizer, variant, rng)
                    df_docs = prompt_df.copy()
                    df_docs["model"] = model_key
                    df_docs["layer"] = layer
                    df_docs["matrix_variant"] = variant
                    df_docs["topk_text"] = docs
                    all_doc_rows.append(df_docs)

                    met = evaluate_docs_cv(df_docs)
                    met.update({
                        "experiment_id": EXPERIMENT_ID,
                        "model": model_key,
                        "layer": layer,
                        "matrix_variant": variant,
                    })
                    all_metric_rows.append(met)
                    print(f"[INFO] metric model={model_key} layer={layer} variant={variant} f1={met.get('cv_macro_f1')}")
                except Exception as e:
                    all_metric_rows.append({
                        "experiment_id": EXPERIMENT_ID,
                        "model": model_key,
                        "layer": layer,
                        "matrix_variant": variant,
                        "error": repr(e),
                    })
                    print(f"[WARN] failed model={model_key} layer={layer} variant={variant}: {e}")

        del tokenizer, W, hidden_by_layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    docs_all = pd.concat(all_doc_rows, ignore_index=True) if all_doc_rows else pd.DataFrame()
    docs_path = OUT_DIR / "cm_psg_3_topk_text_docs.csv"
    docs_all.to_csv(docs_path, index=False, encoding="utf-8-sig")

    metrics = pd.DataFrame(all_metric_rows)
    metrics_path = OUT_DIR / "cm_psg_3_semantic_metrics.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    if not metrics.empty and "cv_macro_f1" in metrics.columns:
        for col in ["model", "matrix_variant", "layer"]:
            view = metrics.groupby(col)["cv_macro_f1"].agg(["mean", "median", "max", "std", "count"]).reset_index()
            view = view.sort_values("mean", ascending=False)
            view.to_csv(OUT_DIR / f"cm_psg_3_factor_view_{col}.csv", index=False, encoding="utf-8-sig")

        residual_rows = []
        for model in sorted(metrics["model"].dropna().unique()):
            sub = metrics[metrics["model"] == model]
            for layer in sorted(sub["layer"].dropna().unique()):
                sl = sub[sub["layer"] == layer]
                def val(v):
                    ss = sl[sl["matrix_variant"] == v]
                    return float(ss["cv_macro_f1"].mean()) if not ss.empty else np.nan
                real = val("real_W")
                rowp = val("row_permuted_W")
                rand = val("random_tokens")
                residual_rows.append({
                    "model": model,
                    "layer": layer,
                    "real_W_macroF1": real,
                    "row_permuted_macroF1": rowp,
                    "random_tokens_macroF1": rand,
                    "real_minus_rowperm": real - rowp,
                    "real_minus_random": real - rand,
                })
        pd.DataFrame(residual_rows).to_csv(OUT_DIR / "cm_psg_3_semantic_residual_table.csv", index=False, encoding="utf-8-sig")

    verdict = build_verdict(metrics)
    verdict_path = OUT_DIR / "cm_psg_3_verdict.txt"
    verdict_path.write_text(verdict, encoding="utf-8")

    print("\n" + verdict)
    print(f"\n[OK] docs: {docs_path}")
    print(f"[OK] metrics: {metrics_path}")
    print(f"[OK] verdict: {verdict_path}")


if __name__ == "__main__":
    main()
