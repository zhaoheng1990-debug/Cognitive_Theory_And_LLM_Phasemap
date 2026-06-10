# -*- coding: utf-8 -*-
"""
CM-PSG-4 Prompt-Relevance Semantic Residual Audit

Purpose
-------
CM-PSG-3 showed:
    real_W TopK token text > random tokens,
    but real_W == row_permuted_W under family text classification.

CM-PSG-4 tests a stricter semantic claim:

    Does real_W TopK select token IDs that are more relevant to the prompt
    concept / family / task than row-permuted W or random token controls?

This is a prompt-conditioned relevance audit, not a topology audit.

Core comparisons
----------------
For each model/layer/prompt, select TopK token IDs using:

1. real_W:
    scores = H W^T
    selected ids = topk(scores)

2. row_permuted_W:
    scores = H W_perm^T
    selected local ids -> original token ids via perm mapping

3. random_tokens:
    random token ids

Then compute semantic relevance using the model's own W embeddings:
    concept relevance: selected token embeddings vs concept token embedding centroid
    family relevance:  selected token embeddings vs family prototype embedding centroid
    task relevance:    selected token embeddings vs task phrase embedding centroid

If real_W > row_permuted_W:
    supports prompt-conditioned semantic residual beyond PSG / token distribution.

If real_W == row_permuted_W:
    no learned row-identity semantic residual under this audit.

Run
---
python cm_psg_4_prompt_relevance_semantic_residual.py

Outputs
-------
C:\\Users\\ZH\\Desktop\\AGI\\outputs\\cm_psg_4_prompt_relevance_outputs
"""

import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


EXPERIMENT_ID = "CM-PSG-4_Prompt_Relevance_Semantic_Residual_v1_0"

OUT_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\cm_psg_4_prompt_relevance_outputs")

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

K_SELECT = 500
K_EVAL = 80

MATRIX_VARIANTS = ["real_W", "row_permuted_W", "random_tokens"]

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


def l2_normalize(x, axis=-1, eps=1e-9):
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(denom, eps)


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
    W = emb.weight.detach().float().cpu().numpy().astype(np.float32)
    return W


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


def encode_token_ids(tokenizer, text):
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        ids = tokenizer.encode(text)
    # Remove invalid / duplicate ids.
    out = []
    for x in ids:
        try:
            xi = int(x)
            if xi >= 0 and xi not in out:
                out.append(xi)
        except Exception:
            pass
    return out


def vector_from_text(tokenizer, Wn, text):
    ids = encode_token_ids(tokenizer, text)
    ids = [i for i in ids if 0 <= i < Wn.shape[0]]
    if not ids:
        return np.zeros((Wn.shape[1],), dtype=np.float32), 0
    v = Wn[ids].mean(axis=0)
    v = l2_normalize(v.reshape(1, -1))[0].astype(np.float32)
    return v, len(ids)


def family_vector(tokenizer, Wn, family):
    concepts = FAMILIES[family]
    vecs = []
    count = 0
    for c in concepts:
        v, n = vector_from_text(tokenizer, Wn, c)
        if n > 0 and np.linalg.norm(v) > 0:
            vecs.append(v)
            count += n
    if not vecs:
        return np.zeros((Wn.shape[1],), dtype=np.float32), 0
    v = np.stack(vecs).mean(axis=0)
    v = l2_normalize(v.reshape(1, -1))[0].astype(np.float32)
    return v, count


def select_token_ids(H, W, variant, rng):
    n, d = H.shape
    V = W.shape[0]

    if variant == "real_W":
        scores = H @ W.T
        idx = np.argpartition(-scores, kth=K_SELECT-1, axis=1)[:, :K_SELECT]
        sc = np.take_along_axis(scores, idx, axis=1)
        order = np.argsort(-sc, axis=1)
        return np.take_along_axis(idx, order, axis=1)[:, :K_EVAL]

    if variant == "row_permuted_W":
        perm = rng.permutation(V)
        Wp = W[perm]
        scores = H @ Wp.T
        idx_local = np.argpartition(-scores, kth=K_SELECT-1, axis=1)[:, :K_SELECT]
        sc = np.take_along_axis(scores, idx_local, axis=1)
        order = np.argsort(-sc, axis=1)
        idx_local = np.take_along_axis(idx_local, order, axis=1)[:, :K_EVAL]
        return perm[idx_local]

    if variant == "random_tokens":
        return np.stack([rng.choice(V, size=K_EVAL, replace=False) for _ in range(n)], axis=0)

    raise ValueError(variant)


def relevance_scores_for_prompt(selected_ids, Wn, concept_vec, family_vec, task_vec):
    X = Wn[selected_ids]  # [K,d], already normalized

    def score_to(v):
        if np.linalg.norm(v) < 1e-9:
            return {
                "mean": np.nan,
                "max": np.nan,
                "top10_mean": np.nan,
                "top20_mean": np.nan,
            }
        sims = X @ v
        sims_sorted = np.sort(sims)[::-1]
        return {
            "mean": float(np.mean(sims)),
            "max": float(np.max(sims)),
            "top10_mean": float(np.mean(sims_sorted[:min(10, len(sims_sorted))])),
            "top20_mean": float(np.mean(sims_sorted[:min(20, len(sims_sorted))])),
        }

    c = score_to(concept_vec)
    f = score_to(family_vec)
    t = score_to(task_vec)

    out = {}
    for prefix, vals in [("concept", c), ("family", f), ("task", t)]:
        for k, v in vals.items():
            out[f"{prefix}_{k}"] = v

    # Combined prompt relevance: prioritize concept and family, then task.
    out["prompt_relevance_mean"] = np.nanmean([
        out["concept_top20_mean"],
        out["family_top20_mean"],
        out["task_top20_mean"],
    ])
    out["concept_family_mean"] = np.nanmean([
        out["concept_top20_mean"],
        out["family_top20_mean"],
    ])

    return out


def decode_preview(tokenizer, ids, n=20):
    toks = []
    for i in ids[:n]:
        try:
            txt = tokenizer.decode([int(i)], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            txt = txt.replace("\n", " ").replace("\r", " ").strip()
            if txt:
                toks.append(txt)
        except Exception:
            pass
    return " ".join(toks)


def build_verdict(summary, residual_table):
    lines = []
    lines.append("=" * 80)
    lines.append("CM-PSG-4 Prompt-Relevance Semantic Residual Verdict")
    lines.append("=" * 80)

    if summary.empty:
        lines.append("NO_VALID_ROWS")
        return "\n".join(lines)

    lines.append("\nTop aggregate rows:")
    top = summary.sort_values("prompt_relevance_mean", ascending=False).head(30)
    for _, r in top.iterrows():
        lines.append(
            f"model={r['model']} layer={int(r['layer'])} variant={r['matrix_variant']} "
            f"promptRel={r['prompt_relevance_mean']:.4f} "
            f"concept20={r['concept_top20_mean']:.4f} "
            f"family20={r['family_top20_mean']:.4f} "
            f"task20={r['task_top20_mean']:.4f}"
        )

    lines.append("\nBest per-model contrasts:")
    rows = []
    for model in sorted(summary["model"].unique()):
        sub = summary[summary["model"] == model]

        def best(v, metric):
            ss = sub[sub["matrix_variant"] == v]
            return float(ss[metric].max()) if not ss.empty else np.nan

        real = best("real_W", "prompt_relevance_mean")
        rowp = best("row_permuted_W", "prompt_relevance_mean")
        rand = best("random_tokens", "prompt_relevance_mean")
        real_c = best("real_W", "concept_top20_mean")
        rowp_c = best("row_permuted_W", "concept_top20_mean")
        rand_c = best("random_tokens", "concept_top20_mean")

        rows.append({
            "model": model,
            "real_prompt_best": real,
            "rowperm_prompt_best": rowp,
            "random_prompt_best": rand,
            "real_minus_rowperm": real - rowp,
            "real_minus_random": real - rand,
            "real_concept_best": real_c,
            "rowperm_concept_best": rowp_c,
            "random_concept_best": rand_c,
            "concept_real_minus_rowperm": real_c - rowp_c,
            "concept_real_minus_random": real_c - rand_c,
        })

        lines.append(
            f"{model}: prompt real={real:.4f}, rowperm={rowp:.4f}, random={rand:.4f}, "
            f"real-rowperm={real-rowp:.4f}, real-random={real-rand:.4f}; "
            f"concept20 real={real_c:.4f}, rowperm={rowp_c:.4f}, random={rand_c:.4f}"
        )

    df = pd.DataFrame(rows)
    mean_row = float(df["real_minus_rowperm"].mean())
    mean_rand = float(df["real_minus_random"].mean())
    mean_crow = float(df["concept_real_minus_rowperm"].mean())
    mean_crand = float(df["concept_real_minus_random"].mean())

    lines.append("\nResidual means:")
    lines.append(f"mean_prompt_real_minus_rowperm={mean_row:.4f}")
    lines.append(f"mean_prompt_real_minus_random={mean_rand:.4f}")
    lines.append(f"mean_concept_real_minus_rowperm={mean_crow:.4f}")
    lines.append(f"mean_concept_real_minus_random={mean_crand:.4f}")

    lines.append("\nInterpretation:")
    if mean_row > 0.03 and mean_rand > 0.03:
        lines.append("PASS-PROMPT-RELEVANCE-RESIDUAL: real W TopK is more prompt-relevant than rowperm and random controls.")
    elif mean_rand > 0.03 and mean_row <= 0.03:
        lines.append("MIXED: real W exceeds random but not rowperm; prompt relevance is not separable from row permutation.")
    elif mean_row > 0.03:
        lines.append("MIXED: real W exceeds rowperm but not random; inspect token-frequency/random baseline.")
    else:
        lines.append("NO-PROMPT-RELEVANCE-RESIDUAL: current audit does not support learned row-identity semantic residual.")

    if mean_crow > 0.03 and mean_crand > 0.03:
        lines.append("PASS-CONCEPT-RESIDUAL: concept-specific relevance survives controls.")
    else:
        lines.append("NO-STRONG-CONCEPT-RESIDUAL: concept-specific relevance is weak or control-matched.")

    lines.append("\nPaper2 use:")
    lines.append("If PASS, state that learned W supplies prompt-conditioned semantic relevance after PSG correction.")
    lines.append("If MIXED/NO, keep semantic interpretability as descriptive, not causal learned-W residual.")
    lines.append("=" * 80)
    return "\n".join(lines)


def main():
    warnings.filterwarnings("ignore")
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    prompt_df = build_labeled_prompts()
    prompt_df.to_csv(OUT_DIR / "cm_psg_4_labeled_prompts.csv", index=False, encoding="utf-8-sig")

    all_rows = []
    preview_rows = []

    for model_key, cfg in MODEL_CONFIGS.items():
        model_path = cfg["path"]
        if not Path(model_path).exists():
            print(f"[WARN] skip {model_key}, path not found: {model_path}")
            continue

        model, tokenizer = load_model(model_key, model_path)
        W = get_output_W(model)
        Wn = l2_normalize(W)

        hidden_by_layer, layers = extract_hidden(model, tokenizer, prompt_df["prompt"].tolist(), cfg["layers"])

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Precompute text vectors.
        concept_vec_cache = {}
        family_vec_cache = {}
        task_vec_cache = {}

        for concept in prompt_df["concept"].unique():
            concept_vec_cache[concept] = vector_from_text(tokenizer, Wn, concept)

        for family in prompt_df["family"].unique():
            family_vec_cache[family] = family_vector(tokenizer, Wn, family)

        for task in prompt_df["task"].unique():
            task_vec_cache[task] = vector_from_text(tokenizer, Wn, task)

        for layer in layers:
            H = hidden_by_layer[layer]

            for variant in MATRIX_VARIANTS:
                rng = np.random.default_rng(SEED + hash((model_key, layer, variant)) % 10007)
                try:
                    selected = select_token_ids(H, W, variant, rng)
                except Exception as e:
                    print(f"[WARN] selection failed model={model_key} layer={layer} variant={variant}: {e}")
                    continue

                for i, meta in prompt_df.iterrows():
                    concept = meta["concept"]
                    family = meta["family"]
                    task = meta["task"]

                    concept_vec, concept_tok_count = concept_vec_cache[concept]
                    family_vec, family_tok_count = family_vec_cache[family]
                    task_vec, task_tok_count = task_vec_cache[task]

                    ids = selected[i]
                    scores = relevance_scores_for_prompt(ids, Wn, concept_vec, family_vec, task_vec)

                    row = {
                        "experiment_id": EXPERIMENT_ID,
                        "model": model_key,
                        "layer": layer,
                        "matrix_variant": variant,
                        "prompt_index": int(i),
                        "family": family,
                        "concept": concept,
                        "task": task,
                        "group_id": int(meta["group_id"]),
                        "concept_token_count": concept_tok_count,
                        "family_token_count": family_tok_count,
                        "task_token_count": task_tok_count,
                    }
                    row.update(scores)
                    all_rows.append(row)

                    if i < 10:
                        preview_rows.append({
                            "model": model_key,
                            "layer": layer,
                            "matrix_variant": variant,
                            "prompt_index": int(i),
                            "family": family,
                            "concept": concept,
                            "task": task,
                            "preview_tokens": decode_preview(tokenizer, ids, n=20),
                        })

                print(f"[INFO] done model={model_key} layer={layer} variant={variant} rows={len(all_rows)}")

        del tokenizer, W, Wn, hidden_by_layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    raw = pd.DataFrame(all_rows)
    raw_path = OUT_DIR / "cm_psg_4_prompt_relevance_raw.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")

    previews = pd.DataFrame(preview_rows)
    previews.to_csv(OUT_DIR / "cm_psg_4_token_previews.csv", index=False, encoding="utf-8-sig")

    if raw.empty:
        verdict = "NO_VALID_ROWS"
        (OUT_DIR / "cm_psg_4_verdict.txt").write_text(verdict, encoding="utf-8")
        print(verdict)
        return

    group_cols = ["model", "layer", "matrix_variant"]
    metric_cols = [
        "concept_mean", "concept_max", "concept_top10_mean", "concept_top20_mean",
        "family_mean", "family_max", "family_top10_mean", "family_top20_mean",
        "task_mean", "task_max", "task_top10_mean", "task_top20_mean",
        "prompt_relevance_mean", "concept_family_mean",
    ]

    summary = raw.groupby(group_cols, dropna=False)[metric_cols].mean().reset_index()
    summary_path = OUT_DIR / "cm_psg_4_prompt_relevance_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    for col in ["model", "layer", "matrix_variant"]:
        view = summary.groupby(col)[["prompt_relevance_mean", "concept_top20_mean", "family_top20_mean", "task_top20_mean"]].agg(["mean", "median", "max", "std", "count"])
        view.columns = ["_".join([str(x) for x in c if x]) for c in view.columns.to_flat_index()]
        view = view.reset_index()
        view.to_csv(OUT_DIR / f"cm_psg_4_factor_view_{col}.csv", index=False, encoding="utf-8-sig")

    # Residual table per model/layer.
    residual_rows = []
    for model in sorted(summary["model"].unique()):
        sm = summary[summary["model"] == model]
        for layer in sorted(sm["layer"].unique()):
            sl = sm[sm["layer"] == layer]
            def row(v):
                ss = sl[sl["matrix_variant"] == v]
                return ss.iloc[0] if not ss.empty else None
            rr = row("real_W")
            rp = row("row_permuted_W")
            rn = row("random_tokens")
            if rr is None or rp is None or rn is None:
                continue
            residual_rows.append({
                "model": model,
                "layer": layer,
                "real_prompt": rr["prompt_relevance_mean"],
                "rowperm_prompt": rp["prompt_relevance_mean"],
                "random_prompt": rn["prompt_relevance_mean"],
                "real_minus_rowperm_prompt": rr["prompt_relevance_mean"] - rp["prompt_relevance_mean"],
                "real_minus_random_prompt": rr["prompt_relevance_mean"] - rn["prompt_relevance_mean"],
                "real_concept20": rr["concept_top20_mean"],
                "rowperm_concept20": rp["concept_top20_mean"],
                "random_concept20": rn["concept_top20_mean"],
                "real_minus_rowperm_concept20": rr["concept_top20_mean"] - rp["concept_top20_mean"],
                "real_minus_random_concept20": rr["concept_top20_mean"] - rn["concept_top20_mean"],
                "real_family20": rr["family_top20_mean"],
                "rowperm_family20": rp["family_top20_mean"],
                "random_family20": rn["family_top20_mean"],
                "real_minus_rowperm_family20": rr["family_top20_mean"] - rp["family_top20_mean"],
                "real_minus_random_family20": rr["family_top20_mean"] - rn["family_top20_mean"],
            })

    residual = pd.DataFrame(residual_rows)
    residual_path = OUT_DIR / "cm_psg_4_prompt_relevance_residual_table.csv"
    residual.to_csv(residual_path, index=False, encoding="utf-8-sig")

    verdict = build_verdict(summary, residual)
    verdict_path = OUT_DIR / "cm_psg_4_verdict.txt"
    verdict_path.write_text(verdict, encoding="utf-8")

    print("\n" + verdict)
    print(f"\n[OK] raw: {raw_path}")
    print(f"[OK] summary: {summary_path}")
    print(f"[OK] residual: {residual_path}")
    print(f"[OK] verdict: {verdict_path}")


if __name__ == "__main__":
    main()
