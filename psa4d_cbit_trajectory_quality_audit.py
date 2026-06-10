# -*- coding: utf-8 -*-
"""
PSA-4D: Cbit -> TrajectoryQuality -> AnswerQuality Audit
========================================================

Why this version exists
-----------------------
PSA-4B/4C showed:
    Qwen: Cbit -> AnswerQuality works.
    Llama/Gemma: direct Cbit -> endpoint AnswerQuality does not close.

This suggests the quality object is too narrow. In the cognitive ontology,
Q_c is not only answer correctness. It includes closure, consistency,
trajectory stability, transferability and cost.

PSA-4D therefore tests a two-stage main-chain model:

    Cbit_structure -> TrajectoryQuality -> AnswerQuality

Main tests
----------
1. Does stagewise/effective Cbit predict trajectory quality?
2. Does trajectory quality predict answer quality?
3. Does trajectory quality mediate Cbit -> answer quality?
4. Is Cbit -> TrajectoryQuality more cross-model stable than
   Cbit -> AnswerQuality?

TrajectoryQuality proxies
-------------------------
- transition_stability
- commit_final_conditional_prob
- final_family_sharpness
- stagewise_Cbit
- effective_Cbit
- TopK entropy/drop/spread features
- family transition consistency
- correct-rank quality as optional answer-adjacent proxy

Terminology
-----------
Uses Cbit / trajectory quality / answer quality / family transition.
No basin terminology.

Run smoke:
    python psa4d_cbit_trajectory_quality_audit.py --models qwen --n_base 40 --topk 30

Formal:
    python psa4d_cbit_trajectory_quality_audit.py --models qwen llama gemma --n_base 120 --topk 50 --window_mode scan
"""

import json
import math
import random
import argparse
from pathlib import Path
from typing import Any, List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer

from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    r2_score,
    mean_squared_error,
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import pearsonr, spearmanr


MODEL_PATHS = {
    "qwen": r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main",
    "llama": r"D:\model\Llama-3.2-1B-Instruct",
    "gemma": r"D:\model\gemma-2-2b-it",
}

DEFAULT_OUT_DIR = r"C:\Users\ZH\Desktop\AGI\outputs\psa4d_trajectory_quality_outputs"
SEED = 20260607

TASK_FAMILIES = [
    "chain_transitive",
    "override_update",
    "exception_binding",
    "majority_evidence",
    "source_priority",
]

NAME_POOL = [
    "Aster", "Boreal", "Cobalt", "Dune", "Ember", "Fjord", "Glade", "Harbor",
    "Ivory", "Jade", "Kite", "Lumen", "Mosaic", "Nimbus", "Opal", "Quartz",
    "Raven", "Sol", "Topaz", "Umber", "Vega", "Willow", "Xylo", "Yarrow",
    "Zenith", "Atlas", "Copper", "Falcon", "Garden", "Island", "Lagoon",
    "Marble", "Onyx", "Pearl", "Ridge", "Station", "Temple", "Valley"
]

SURFACES = [
    "Read the symbolic record and answer the target.",
    "Use only the directed rules below to resolve the target.",
    "Inspect the local relation record and identify the licensed endpoint.",
    "Follow the rule system and choose the correct endpoint.",
    "Resolve the endpoint using the stated constraints only.",
    "Determine which target is licensed by the record.",
]


def norm_text(x: Any) -> str:
    if isinstance(x, str):
        return x
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    if isinstance(x, (list, tuple)):
        return "\n".join(str(v) for v in x)
    return str(x)


def sanitize_array(x, name="array"):
    arr = np.asarray(x)
    if not np.isfinite(arr).all():
        n_nan = int(np.isnan(arr).sum())
        n_posinf = int(np.isposinf(arr).sum())
        n_neginf = int(np.isneginf(arr).sum())
        print(f"[WARN] non-finite values in {name}: nan={n_nan}, +inf={n_posinf}, -inf={n_neginf}; applying nan_to_num")
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
    return arr


def sanitize_dataframe_numeric(df, name="dataframe"):
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        arr = out[num_cols].values
        if not np.isfinite(arr).all():
            n_nan = int(np.isnan(arr).sum())
            n_posinf = int(np.isposinf(arr).sum())
            n_neginf = int(np.isneginf(arr).sum())
            print(f"[WARN] non-finite numeric values in {name}: nan={n_nan}, +inf={n_posinf}, -inf={n_neginf}; applying nan_to_num")
            out[num_cols] = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
    return out


def make_case(base_id: int, family: str, rng: random.Random) -> Dict[str, Any]:
    names = rng.sample(NAME_POOL, 6)
    a, b, c, d, e, f = names
    surface = SURFACES[base_id % len(SURFACES)]

    if family == "chain_transitive":
        correct = e
        wrongs = [c, d, f]
        lines = [
            "Rule: follow the directed chain until it ends.",
            f"E1: {a} -> {b}.",
            f"E2: {b} -> {c}.",
            f"E3: {c} -> {d}.",
            f"E4: {d} -> {e}.",
            f"Start: {a}.",
        ]
    elif family == "override_update":
        correct = d
        wrongs = [c, e, f]
        lines = [
            "Rule: if a revised endpoint is present, the revised endpoint overrides the old endpoint.",
            f"E1: {a} -> {b}.",
            f"E2: old endpoint from {b} is {c}.",
            f"E3: revised endpoint from {b} is {d}.",
            f"E4: use the revised endpoint.",
            f"Start: {a}.",
        ]
    elif family == "exception_binding":
        correct = e
        wrongs = [c, d, f]
        lines = [
            "Rule: use the default endpoint unless the exception marker is active; if active, use the exception endpoint.",
            f"E1: {a} -> {b}.",
            f"E2: default endpoint from {b} is {c}.",
            f"E3: exception endpoint from {b} is {e}.",
            f"E4: exception marker is active.",
            f"Start: {a}.",
        ]
    elif family == "majority_evidence":
        correct = d
        wrongs = [c, e, f]
        lines = [
            "Rule: choose the endpoint with the strongest support count.",
            f"E1: support for {d}.",
            f"E2: support for {d}.",
            f"E3: support for {c}.",
            f"E4: support for {e}.",
            f"Start: {a}.",
        ]
    elif family == "source_priority":
        correct = f
        wrongs = [c, d, e]
        lines = [
            "Rule: expert source overrides ordinary source; use the expert endpoint.",
            f"E1: ordinary source says endpoint is {c}.",
            f"E2: secondary source says endpoint is {d}.",
            f"E3: expert source says endpoint is {f}.",
            f"E4: source priority is expert > secondary > ordinary.",
            f"Start: {a}.",
        ]
    else:
        raise ValueError(family)

    middle = lines[1:-1]
    rng.shuffle(middle)
    lines = [lines[0]] + middle + [lines[-1]]

    prompt = (
        f"{surface}\n"
        f"Record ID: {base_id}-{family}\n"
        + "\n".join(lines)
        + "\nQuestion: Which endpoint is licensed by this record?\n"
        + "Answer with one endpoint name only."
    )
    return {"prompt": prompt, "correct": correct, "all_candidates": [correct] + wrongs}


def make_clean_case(base_id: int, rng: random.Random) -> Dict[str, Any]:
    names = rng.sample(NAME_POOL, 6)
    a, b, c, d, e, f = names
    surface = SURFACES[base_id % len(SURFACES)]
    correct = e
    wrongs = [c, d, f]
    lines = [
        "Rule: follow the directed chain until it ends.",
        f"E1: {a} -> {b}.",
        f"E2: {b} -> {c}.",
        f"E3: {c} -> {d}.",
        f"E4: {d} -> {e}.",
        f"Start: {a}.",
    ]
    prompt = (
        f"{surface}\n"
        f"Record ID: {base_id}-clean\n"
        + "\n".join(lines)
        + "\nQuestion: Which endpoint is licensed by this record?\n"
        + "Answer with one endpoint name only."
    )
    return {"prompt": prompt, "correct": correct, "all_candidates": [correct] + wrongs}


def build_dataset(n_base: int = 80):
    rows = []
    pairs = []
    rid = 0
    for base_id in range(n_base):
        clean_rng = random.Random(SEED * 100000 + base_id * 100 + 777)
        clean = make_clean_case(base_id, clean_rng)
        clean_row_id = rid
        rows.append({
            "prompt_row_id": rid,
            "base_id": base_id,
            "task_family": "clean",
            "task_family_id": -1,
            "kind": "clean",
            "correct": clean["correct"],
            "all_candidates": json.dumps(clean["all_candidates"], ensure_ascii=False),
            "prompt": clean["prompt"],
        })
        rid += 1

        for fid, fam in enumerate(TASK_FAMILIES):
            rng = random.Random(SEED * 100000 + base_id * 1000 + fid)
            case = make_case(base_id, fam, rng)
            cond_row_id = rid
            rows.append({
                "prompt_row_id": rid,
                "base_id": base_id,
                "task_family": fam,
                "task_family_id": fid,
                "kind": "condition",
                "correct": case["correct"],
                "all_candidates": json.dumps(case["all_candidates"], ensure_ascii=False),
                "prompt": case["prompt"],
            })
            pairs.append({
                "pair_id": len(pairs),
                "base_id": base_id,
                "clean_row_id": clean_row_id,
                "cond_row_id": cond_row_id,
                "task_family": fam,
                "task_family_id": fid,
                "correct": case["correct"],
                "all_candidates": json.dumps(case["all_candidates"], ensure_ascii=False),
            })
            rid += 1

    prompt_df = pd.DataFrame(rows)
    pair_df = pd.DataFrame(pairs)
    prompt_df["prompt"] = prompt_df["prompt"].map(norm_text)
    return prompt_df, pair_df


def load_model(model_key: str, device: str, dtype: str, local_files_only: bool = True):
    torch_dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    model_path = MODEL_PATHS[model_key]

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
def extract_hidden_topk(tokenizer, model, df, device, max_len=256, topk=50, batch_size=1):
    all_hidden, all_ent, all_spread, all_center = [], [], [], []
    emb_weight = model.get_output_embeddings().weight.detach()

    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        texts = [norm_text(x) for x in batch["prompt"].tolist()]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(device)

        out = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        hidden_states = out.hidden_states[1:]
        attention = inputs["attention_mask"]
        last_pos = attention.sum(dim=1) - 1

        lh, le, ls, lc = [], [], [], []
        for h in hidden_states:
            h_last = h[torch.arange(h.shape[0], device=device), last_pos]
            lh.append(h_last.detach().float().cpu())

            logits = h_last @ emb_weight.t()
            vals, ids = torch.topk(logits, k=min(topk, logits.shape[-1]), dim=-1)

            probs = torch.softmax(vals.float(), dim=-1)
            ent = -(probs * torch.log2(probs + 1e-12)).sum(dim=-1)

            tok_emb = emb_weight[ids].detach().float()
            center = tok_emb.mean(dim=1)
            center_n = F.normalize(center, dim=-1)
            tok_n = F.normalize(tok_emb, dim=-1)
            spread = (1.0 - (tok_n * center_n.unsqueeze(1)).sum(dim=-1)).mean(dim=-1)

            le.append(ent.detach().cpu())
            ls.append(spread.detach().cpu())
            lc.append(center.detach().cpu())

        all_hidden.append(torch.stack(lh, dim=0).transpose(0, 1))
        all_ent.append(torch.stack(le, dim=0).transpose(0, 1))
        all_spread.append(torch.stack(ls, dim=0).transpose(0, 1))
        all_center.append(torch.stack(lc, dim=0).transpose(0, 1))
        print(f"[extract] {start + len(batch)}/{len(df)}")

    feats = {
        "hidden": torch.cat(all_hidden, dim=0).numpy(),
        "topk_entropy": torch.cat(all_ent, dim=0).numpy(),
        "topk_spread": torch.cat(all_spread, dim=0).numpy(),
        "topk_center": torch.cat(all_center, dim=0).numpy(),
    }
    return {k: sanitize_array(v, f"extract_hidden_topk.{k}") for k, v in feats.items()}


@torch.no_grad()
def score_candidates(tokenizer, model, pair_df, prompt_df, device, max_len=256, score_mode="mean"):
    prompt_map = prompt_df.set_index("prompt_row_id")["prompt"].to_dict()
    out_rows = []

    for i, row in pair_df.iterrows():
        prompt = prompt_map[int(row["cond_row_id"])]
        candidates = json.loads(row["all_candidates"])
        correct = row["correct"]

        scores = {}
        for cand in candidates:
            full = prompt + " " + cand
            enc_full = tokenizer(full, return_tensors="pt", truncation=True, max_length=max_len).to(device)
            enc_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(device)

            ids = enc_full["input_ids"]
            prompt_len = enc_prompt["input_ids"].shape[1]
            cand_start = max(1, prompt_len)

            logits = model(**enc_full, use_cache=False).logits
            logp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
            target = ids[:, 1:]
            start_t = max(0, cand_start - 1)

            if start_t >= target.shape[1]:
                score = -1e9
            else:
                lp = logp[:, start_t:, :].gather(-1, target[:, start_t:].unsqueeze(-1)).squeeze(-1)
                if score_mode == "sum":
                    score = float(lp.sum().detach().cpu().item())
                elif score_mode == "first":
                    score = float(lp.flatten()[0].detach().cpu().item())
                else:
                    score = float(lp.mean().detach().cpu().item())

            if not math.isfinite(score):
                score = -1e9
            scores[cand] = score

        correct_score = scores[correct]
        best_wrong = max(v for k, v in scores.items() if k != correct)
        margin = correct_score - best_wrong
        pred = max(scores.items(), key=lambda kv: kv[1])[0]
        rank = 1 + sum(1 for k, v in scores.items() if v > correct_score)

        out_rows.append({
            "pair_id": int(row["pair_id"]),
            "correct": correct,
            "pred_candidate": pred,
            "answer_correct": int(pred == correct),
            "answer_margin": float(margin),
            "correct_rank": int(rank),
            "rank_quality": float(-rank),
            "correct_score": float(correct_score),
            "best_wrong_score": float(best_wrong),
            "candidate_scores": json.dumps(scores, ensure_ascii=False),
        })
        if (i + 1) % 20 == 0:
            print(f"[score-{score_mode}] {i + 1}/{len(pair_df)}")

    return pd.DataFrame(out_rows)


def build_delta_features(feats, pair_df):
    clean = pair_df["clean_row_id"].values.astype(int)
    cond = pair_df["cond_row_id"].values.astype(int)
    out = {
        "hidden_delta": feats["hidden"][cond] - feats["hidden"][clean],
        "topk_entropy_delta": feats["topk_entropy"][cond] - feats["topk_entropy"][clean],
        "topk_spread_delta": feats["topk_spread"][cond] - feats["topk_spread"][clean],
        "topk_center_delta": feats["topk_center"][cond] - feats["topk_center"][clean],
    }
    return {k: sanitize_array(v, k) for k, v in out.items()}


def clamp_window(start, end, L):
    return [i for i in range(max(0, start), min(end + 1, L)) if 0 <= i < L]


def relative_window(L, a, b):
    start = int(math.floor(a * L))
    end = int(math.ceil(b * L)) - 1
    return clamp_window(start, end, L)


def make_windows(L, profile="relative_default"):
    profiles = {
        "relative_default": {
            "shape": (0.30, 0.65),
            "entry": (0.65, 0.78),
            "commit": (0.78, 0.90),
            "final": (0.90, 1.00),
        },
        "relative_early": {
            "shape": (0.25, 0.58),
            "entry": (0.58, 0.72),
            "commit": (0.72, 0.86),
            "final": (0.86, 1.00),
        },
        "relative_late": {
            "shape": (0.35, 0.68),
            "entry": (0.68, 0.80),
            "commit": (0.80, 0.92),
            "final": (0.92, 1.00),
        },
        "relative_wide_entry": {
            "shape": (0.25, 0.60),
            "entry": (0.60, 0.80),
            "commit": (0.80, 0.92),
            "final": (0.92, 1.00),
        },
    }

    spec = profiles[profile]
    windows = {k: relative_window(L, *v) for k, v in spec.items()}
    if any(len(v) == 0 for v in windows.values()):
        q1 = max(1, int(round(0.60 * L)))
        q2 = max(q1 + 1, int(round(0.75 * L)))
        q3 = max(q2 + 1, int(round(0.88 * L)))
        windows = {
            "shape": clamp_window(max(0, int(round(0.25 * L))), q1 - 1, L),
            "entry": clamp_window(q1, q2 - 1, L),
            "commit": clamp_window(q2, q3 - 1, L),
            "final": clamp_window(q3, L - 1, L),
        }
    return windows


def scalar_window(arr, idx):
    return arr[:, idx].mean(axis=1)


def center_window(arr, idx):
    return arr[:, idx, :].mean(axis=1)


def standard_pca(X, dim, seed=SEED):
    X = sanitize_array(X, "standard_pca.input")
    Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    Xs = sanitize_array(Xs, "standard_pca.scaled")
    d = min(dim, Xs.shape[0] - 1, Xs.shape[1])
    if d >= 2:
        Xp = PCA(n_components=d, random_state=seed).fit_transform(Xs)
        return sanitize_array(Xp, "standard_pca.output")
    return Xs


def window_features(delta, window):
    hd = delta["hidden_delta"]
    tc = delta["topk_center_delta"]
    te = delta["topk_entropy_delta"]
    ts = delta["topk_spread_delta"]
    X = np.concatenate([
        center_window(hd, window),
        center_window(tc, window),
        scalar_window(te, window)[:, None],
        scalar_window(ts, window)[:, None],
    ], axis=1)
    return sanitize_array(X, "window_features")


def make_family(delta, window, n_clusters=5, pca_dim=32, seed=SEED):
    X = window_features(delta, window)
    Xp = standard_pca(X, pca_dim, seed=seed)
    k = min(n_clusters, max(2, Xp.shape[0] // 10))
    labels = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(Xp)
    return labels.astype(int)


def transition_oof(src, dst, groups, alpha=1.0):
    src = np.asarray(src).astype(int)
    dst = np.asarray(dst).astype(int)
    groups = np.asarray(groups)
    n = len(src)

    prior_surprisal = np.zeros(n, dtype=np.float64)
    conditional_surprisal = np.zeros(n, dtype=np.float64)
    conditional_prob = np.zeros(n, dtype=np.float64)

    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    k_dst = int(dst.max()) + 1
    k_src = int(src.max()) + 1

    for tr, te in cv.split(np.zeros(n), dst, groups):
        prior_counts = np.bincount(dst[tr], minlength=k_dst).astype(float) + alpha
        prior_probs = prior_counts / prior_counts.sum()

        table = np.zeros((k_src, k_dst), dtype=float) + alpha
        for s, d in zip(src[tr], dst[tr]):
            table[s, d] += 1.0
        table = table / table.sum(axis=1, keepdims=True)

        for j in te:
            p_prior = prior_probs[dst[j]]
            p_cond = table[src[j], dst[j]]
            prior_surprisal[j] = -math.log2(max(p_prior, 1e-12))
            conditional_surprisal[j] = -math.log2(max(p_cond, 1e-12))
            conditional_prob[j] = p_cond

    cbit = prior_surprisal - conditional_surprisal
    return {
        "cbit": cbit,
        "conditional_prob": conditional_prob,
        "surprisal_reduction": cbit,
        "mean_cbit": float(np.mean(cbit)),
        "mean_conditional_prob": float(np.mean(conditional_prob)),
    }


def entropy_bits_from_labels(labels):
    counts = np.bincount(np.asarray(labels).astype(int))
    p = counts / (counts.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def family_transition_consistency(src, dst):
    src = np.asarray(src).astype(int)
    dst = np.asarray(dst).astype(int)
    vals = []
    for s in np.unique(src):
        mask = src == s
        counts = np.bincount(dst[mask].astype(int), minlength=int(dst.max()) + 1)
        vals.append(counts.max() / max(1, counts.sum()))
    return float(np.mean(vals))


def safe_corr(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(pearsonr(x, y)[0])


def reg_eval(X, y, groups):
    X = sanitize_array(X, "reg_eval.X")
    y = sanitize_array(np.asarray(y).astype(float), "reg_eval.y")
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    pred = np.zeros_like(y, dtype=np.float64)
    for tr, te in cv.split(X, y, groups):
        model = make_pipeline(StandardScaler(with_mean=True, with_std=True), Ridge(alpha=1.0))
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
    spr = None
    if np.std(pred) > 1e-12 and np.std(y) > 1e-12:
        spr = float(spearmanr(pred, y).correlation)
    return {
        "corr": safe_corr(pred, y),
        "spearman": spr,
        "r2": float(r2_score(y, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y, pred))),
    }


def clf_eval(X, y, groups):
    X = sanitize_array(X, "clf_eval.X")
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return {"acc": None, "macro_f1": None, "auc": None, "note": "degenerate_target"}
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced", C=1.0)
    )
    prob = cross_val_predict(clf, X, y, cv=cv.split(X, y, groups), method="predict_proba")[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "acc": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "auc": float(roc_auc_score(y, prob)),
    }


def quality_features_from_pairs(pair_local):
    """
    Build trajectory-quality proxy columns. These are internal quality proxies,
    not direct answer labels.
    """
    out = pair_local.copy()

    out["trajectory_stability"] = out["transition_stability"]
    out["commit_final_quality"] = out["prob_commit_final"]
    out["entry_commit_quality"] = out["prob_entry_commit"]
    out["stagewise_surprisal_reduction"] = out["stagewise_cbit"]

    # Direct internal quality aggregate.
    out["trajectory_quality_proxy"] = (
        0.35 * out["stagewise_cbit"]
        + 0.25 * out["transition_stability"]
        + 0.20 * out["prob_entry_commit"]
        + 0.20 * out["prob_commit_final"]
    )

    # Normalized version per run.
    x = out["trajectory_quality_proxy"].values.astype(float)
    out["trajectory_quality_z"] = (x - x.mean()) / (x.std() + 1e-12)

    # Binary high-quality trajectory proxy based on above-median trajectory quality.
    med = float(np.median(out["trajectory_quality_proxy"].values))
    out["trajectory_quality_high"] = (out["trajectory_quality_proxy"] >= med).astype(int)
    return out


def evaluate_mapping(pair_local, groups):
    """
    Test:
      Cbit -> TrajectoryQuality
      TrajectoryQuality -> AnswerQuality
      Cbit + TrajectoryQuality -> AnswerQuality
    """
    cbit_cols = [
        "cbit_shape_entry",
        "cbit_entry_commit",
        "cbit_commit_final",
        "stagewise_cbit",
        "transition_stability",
        "effective_cbit_proxy",
    ]
    direct_cols = ["cbit_shape_final_direct", "cbit_shape_commit_direct"]
    tq_cols = [
        "trajectory_stability",
        "entry_commit_quality",
        "commit_final_quality",
        "trajectory_quality_proxy",
        "trajectory_quality_z",
    ]
    cbit_tq_cols = cbit_cols + tq_cols

    y_tq = pair_local["trajectory_quality_z"].values.astype(float)
    y_tq_high = pair_local["trajectory_quality_high"].values.astype(int)
    y_margin = pair_local["answer_margin"].values.astype(float)
    y_correct = pair_local["answer_correct"].values.astype(int)

    result = {
        "cbit_to_trajectory_quality": {
            "regression": reg_eval(pair_local[cbit_cols].values, y_tq, groups),
            "classification": clf_eval(pair_local[cbit_cols].values, y_tq_high, groups),
        },
        "direct_cbit_to_trajectory_quality": {
            "regression": reg_eval(pair_local[direct_cols].values, y_tq, groups),
            "classification": clf_eval(pair_local[direct_cols].values, y_tq_high, groups),
        },
        "trajectory_quality_to_answer_margin": reg_eval(pair_local[tq_cols].values, y_margin, groups),
        "trajectory_quality_to_answer_correct": clf_eval(pair_local[tq_cols].values, y_correct, groups),
        "cbit_to_answer_margin": reg_eval(pair_local[cbit_cols].values, y_margin, groups),
        "cbit_to_answer_correct": clf_eval(pair_local[cbit_cols].values, y_correct, groups),
        "cbit_plus_tq_to_answer_margin": reg_eval(pair_local[cbit_tq_cols].values, y_margin, groups),
        "cbit_plus_tq_to_answer_correct": clf_eval(pair_local[cbit_tq_cols].values, y_correct, groups),
    }

    cbit_tq_r2 = result["cbit_to_trajectory_quality"]["regression"]["r2"]
    cbit_tq_auc = result["cbit_to_trajectory_quality"]["classification"]["auc"]
    direct_tq_r2 = result["direct_cbit_to_trajectory_quality"]["regression"]["r2"]
    direct_tq_auc = result["direct_cbit_to_trajectory_quality"]["classification"]["auc"]

    tq_ans_auc = result["trajectory_quality_to_answer_correct"]["auc"]
    tq_ans_r2 = result["trajectory_quality_to_answer_margin"]["r2"]

    cbit_ans_auc = result["cbit_to_answer_correct"]["auc"]
    cbit_ans_r2 = result["cbit_to_answer_margin"]["r2"]
    cbit_tq_ans_auc = result["cbit_plus_tq_to_answer_correct"]["auc"]
    cbit_tq_ans_r2 = result["cbit_plus_tq_to_answer_margin"]["r2"]

    flags = {
        "cbit_predicts_trajectory_quality": bool(
            (cbit_tq_r2 > 0.05) or (cbit_tq_auc is not None and cbit_tq_auc > 0.62)
        ),
        "stagewise_beats_direct_for_trajectory_quality": bool(
            (cbit_tq_r2 > direct_tq_r2 + 0.03) or
            (cbit_tq_auc is not None and direct_tq_auc is not None and cbit_tq_auc > direct_tq_auc + 0.03)
        ),
        "trajectory_quality_predicts_answer_quality": bool(
            (tq_ans_r2 > 0.03) or (tq_ans_auc is not None and tq_ans_auc > 0.58)
        ),
        "trajectory_quality_improves_answer_model": bool(
            (cbit_tq_ans_r2 > cbit_ans_r2 + 0.02) or
            (cbit_tq_ans_auc is not None and cbit_ans_auc is not None and cbit_tq_ans_auc > cbit_ans_auc + 0.03)
        ),
    }

    summary = {
        "cbit_to_tq_r2": cbit_tq_r2,
        "cbit_to_tq_auc": cbit_tq_auc,
        "direct_to_tq_r2": direct_tq_r2,
        "direct_to_tq_auc": direct_tq_auc,
        "tq_to_answer_margin_r2": tq_ans_r2,
        "tq_to_answer_correct_auc": tq_ans_auc,
        "cbit_to_answer_margin_r2": cbit_ans_r2,
        "cbit_to_answer_correct_auc": cbit_ans_auc,
        "cbit_plus_tq_to_answer_margin_r2": cbit_tq_ans_r2,
        "cbit_plus_tq_to_answer_correct_auc": cbit_tq_ans_auc,
    }

    return result, flags, summary


def score_window_profile(delta, pair_df, quality, windows, args, profile_name):
    pair_local = pair_df.copy()
    groups = pair_local["base_id"].values.astype(int)

    families = {}
    for name in ["shape", "entry", "commit", "final"]:
        families[name] = make_family(delta, windows[name], n_clusters=args.n_clusters, pca_dim=args.pca_dim, seed=SEED + len(name) + abs(hash(profile_name)) % 1000)
        pair_local[f"{name}_family"] = families[name]

    t_se = transition_oof(families["shape"], families["entry"], groups)
    t_ec = transition_oof(families["entry"], families["commit"], groups)
    t_cf = transition_oof(families["commit"], families["final"], groups)
    t_sf = transition_oof(families["shape"], families["final"], groups)
    t_sc = transition_oof(families["shape"], families["commit"], groups)

    pair_local = pair_local.merge(quality, on="pair_id", how="left")
    pair_local = sanitize_dataframe_numeric(pair_local, f"{profile_name}.pair_local_after_quality")

    pair_local["cbit_shape_entry"] = t_se["cbit"]
    pair_local["cbit_entry_commit"] = t_ec["cbit"]
    pair_local["cbit_commit_final"] = t_cf["cbit"]
    pair_local["cbit_shape_final_direct"] = t_sf["cbit"]
    pair_local["cbit_shape_commit_direct"] = t_sc["cbit"]

    pair_local["prob_shape_entry"] = t_se["conditional_prob"]
    pair_local["prob_entry_commit"] = t_ec["conditional_prob"]
    pair_local["prob_commit_final"] = t_cf["conditional_prob"]
    pair_local["prob_shape_final_direct"] = t_sf["conditional_prob"]
    pair_local["prob_shape_commit_direct"] = t_sc["conditional_prob"]

    pair_local["stagewise_cbit"] = pair_local["cbit_shape_entry"] + pair_local["cbit_entry_commit"] + pair_local["cbit_commit_final"]
    pair_local["transition_stability"] = t_se["conditional_prob"] * t_ec["conditional_prob"] * t_cf["conditional_prob"]
    pair_local["effective_cbit_proxy"] = pair_local["stagewise_cbit"] * pair_local["transition_stability"]

    # Transition consistency global summaries.
    cons_se = family_transition_consistency(families["shape"], families["entry"])
    cons_ec = family_transition_consistency(families["entry"], families["commit"])
    cons_cf = family_transition_consistency(families["commit"], families["final"])

    pair_local["consistency_shape_entry_global"] = cons_se
    pair_local["consistency_entry_commit_global"] = cons_ec
    pair_local["consistency_commit_final_global"] = cons_cf
    pair_local["stagewise_consistency_global"] = (cons_se + cons_ec + cons_cf) / 3.0

    pair_local = quality_features_from_pairs(pair_local)
    pair_local = sanitize_dataframe_numeric(pair_local, f"{profile_name}.pair_local_final")

    evals, flags, summary = evaluate_mapping(pair_local, groups)

    sel_score = 0.0
    sel_score += max(0.0, summary["cbit_to_tq_r2"]) * 3.0
    if summary["cbit_to_tq_auc"] is not None:
        sel_score += max(0.0, summary["cbit_to_tq_auc"] - 0.5) * 3.0
    if summary["tq_to_answer_correct_auc"] is not None:
        sel_score += max(0.0, summary["tq_to_answer_correct_auc"] - 0.5) * 2.0
    sel_score += max(0.0, summary["tq_to_answer_margin_r2"]) * 2.0
    sel_score += 0.5 if flags["stagewise_beats_direct_for_trajectory_quality"] else 0.0

    result = {
        "profile_name": profile_name,
        "windows": windows,
        "selection_score": float(sel_score),
        "transition_summary": {
            "mean_cbit_shape_entry": t_se["mean_cbit"],
            "mean_cbit_entry_commit": t_ec["mean_cbit"],
            "mean_cbit_commit_final": t_cf["mean_cbit"],
            "mean_cbit_shape_final_direct": t_sf["mean_cbit"],
            "mean_stagewise_cbit": float(pair_local["stagewise_cbit"].mean()),
            "mean_transition_stability": float(pair_local["transition_stability"].mean()),
            "mean_effective_cbit_proxy": float(pair_local["effective_cbit_proxy"].mean()),
            "consistency_shape_entry": cons_se,
            "consistency_entry_commit": cons_ec,
            "consistency_commit_final": cons_cf,
        },
        "mapping_results": evals,
        "pass_flags": flags,
        "summary_metrics": summary,
    }
    return result, pair_local


def run_one_model(model_key, args, prompt_df, pair_df, model_out_dir):
    model_out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"MODEL: {model_key}")
    print("=" * 80)

    tokenizer, model = load_model(model_key, args.device, args.dtype, args.local_files_only)

    feats = extract_hidden_topk(tokenizer, model, prompt_df, args.device, args.max_len, args.topk, batch_size=1)
    delta = build_delta_features(feats, pair_df)

    quality = score_candidates(
        tokenizer, model, pair_df, prompt_df, args.device, args.max_len, score_mode=args.score_mode
    )
    quality.to_csv(model_out_dir / f"psa4d_{model_key}_answer_quality.csv", index=False, encoding="utf-8-sig")

    L = delta["hidden_delta"].shape[1]
    profiles = ["relative_default", "relative_early", "relative_late", "relative_wide_entry"] if args.window_mode == "scan" else ["relative_default"]

    profile_results = []
    profile_pairs = {}
    for profile in profiles:
        windows = make_windows(L, profile)
        if any(len(v) == 0 for v in windows.values()):
            print(f"[WARN] skipping empty profile {profile}: {windows}")
            continue
        res, pairs_scored = score_window_profile(delta, pair_df, quality, windows, args, profile)
        profile_results.append(res)
        profile_pairs[profile] = pairs_scored

    if not profile_results:
        raise RuntimeError(f"No valid window profiles for {model_key}, L={L}")

    best = max(profile_results, key=lambda r: r["selection_score"])
    best_profile = best["profile_name"]
    pair_best = profile_pairs[best_profile]
    pair_best.to_csv(model_out_dir / f"psa4d_{model_key}_pairs_with_cbit_tq_quality.csv", index=False, encoding="utf-8-sig")

    profile_table = []
    for r in profile_results:
        sm = r["summary_metrics"]
        pf = r["pass_flags"]
        ts = r["transition_summary"]
        profile_table.append({
            "model_key": model_key,
            "profile_name": r["profile_name"],
            "selection_score": r["selection_score"],
            "cbit_to_tq_r2": sm["cbit_to_tq_r2"],
            "cbit_to_tq_auc": sm["cbit_to_tq_auc"],
            "direct_to_tq_r2": sm["direct_to_tq_r2"],
            "direct_to_tq_auc": sm["direct_to_tq_auc"],
            "tq_to_answer_correct_auc": sm["tq_to_answer_correct_auc"],
            "tq_to_answer_margin_r2": sm["tq_to_answer_margin_r2"],
            "cbit_to_answer_correct_auc": sm["cbit_to_answer_correct_auc"],
            "cbit_plus_tq_to_answer_correct_auc": sm["cbit_plus_tq_to_answer_correct_auc"],
            "cbit_predicts_tq": pf["cbit_predicts_trajectory_quality"],
            "stagewise_beats_direct_tq": pf["stagewise_beats_direct_for_trajectory_quality"],
            "tq_predicts_answer": pf["trajectory_quality_predicts_answer_quality"],
            "tq_improves_answer_model": pf["trajectory_quality_improves_answer_model"],
            "mean_stagewise_cbit": ts["mean_stagewise_cbit"],
            "mean_effective_cbit_proxy": ts["mean_effective_cbit_proxy"],
            "windows": json.dumps(r["windows"], ensure_ascii=False),
        })
    pd.DataFrame(profile_table).to_csv(model_out_dir / f"psa4d_{model_key}_profile_scan.csv", index=False, encoding="utf-8-sig")

    pf = best["pass_flags"]
    if pf["cbit_predicts_trajectory_quality"] and pf["stagewise_beats_direct_for_trajectory_quality"] and pf["trajectory_quality_predicts_answer_quality"]:
        verdict = "PASS_PSA4D_CBIT_TO_TQ_TO_ANSWER"
    elif pf["cbit_predicts_trajectory_quality"] and pf["stagewise_beats_direct_for_trajectory_quality"]:
        verdict = "PARTIAL_PSA4D_CBIT_TO_TRAJECTORY_QUALITY"
    elif pf["cbit_predicts_trajectory_quality"]:
        verdict = "PARTIAL_PSA4D_CBIT_TQ_SIGNAL"
    else:
        verdict = "FAIL_OR_CAVEAT_PSA4D"

    summary = {
        "model_key": model_key,
        "verdict": verdict,
        "selected_profile": best_profile,
        "selected_profile_score": best["selection_score"],
        "pass_flags": pf,
        "summary_metrics": {
            **best["summary_metrics"],
            **best["transition_summary"],
        },
        "mapping_results": best["mapping_results"],
        "windows": best["windows"],
        "all_profile_results": profile_table,
        "model_path": MODEL_PATHS[model_key],
    }

    with open(model_out_dir / f"psa4d_{model_key}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(model_out_dir / f"psa4d_{model_key}_verdict.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def aggregate_results(model_summaries):
    rows = []
    for s in model_summaries:
        m = s["summary_metrics"]
        rows.append({
            "model_key": s["model_key"],
            "verdict": s["verdict"],
            "selected_profile": s["selected_profile"],
            "cbit_predicts_tq": s["pass_flags"]["cbit_predicts_trajectory_quality"],
            "stagewise_beats_direct_tq": s["pass_flags"]["stagewise_beats_direct_for_trajectory_quality"],
            "tq_predicts_answer": s["pass_flags"]["trajectory_quality_predicts_answer_quality"],
            "tq_improves_answer_model": s["pass_flags"]["trajectory_quality_improves_answer_model"],
            "cbit_to_tq_r2": m["cbit_to_tq_r2"],
            "cbit_to_tq_auc": m["cbit_to_tq_auc"],
            "direct_to_tq_r2": m["direct_to_tq_r2"],
            "direct_to_tq_auc": m["direct_to_tq_auc"],
            "tq_to_answer_correct_auc": m["tq_to_answer_correct_auc"],
            "tq_to_answer_margin_r2": m["tq_to_answer_margin_r2"],
            "cbit_to_answer_correct_auc": m["cbit_to_answer_correct_auc"],
            "cbit_plus_tq_to_answer_correct_auc": m["cbit_plus_tq_to_answer_correct_auc"],
            "mean_stagewise_cbit": m["mean_stagewise_cbit"],
            "mean_effective_cbit_proxy": m["mean_effective_cbit_proxy"],
        })
    df = pd.DataFrame(rows)

    n = len(df)
    n_cbit_tq = int(df["cbit_predicts_tq"].sum())
    n_stage_beats = int(df["stagewise_beats_direct_tq"].sum())
    n_tq_answer = int(df["tq_predicts_answer"].sum())
    n_pass_or_partial = int(df["verdict"].str.contains("PASS|PARTIAL", regex=True).sum())

    if n_cbit_tq >= max(1, n - 1) and n_stage_beats >= max(1, n - 1) and n_tq_answer >= max(1, n - 1):
        verdict = "PASS_PSA4D_CROSSMODEL_CBIT_TRAJECTORY_QUALITY_CHAIN"
    elif n_cbit_tq >= max(1, n - 1) and n_stage_beats >= max(1, n - 1):
        verdict = "PARTIAL_PSA4D_CROSSMODEL_CBIT_TO_TRAJECTORY_QUALITY"
    elif n_cbit_tq >= max(1, n - 1):
        verdict = "PARTIAL_PSA4D_CROSSMODEL_CBIT_TQ_SIGNAL"
    else:
        verdict = "FAIL_OR_CAVEAT_PSA4D_CROSSMODEL"

    return df, {
        "verdict": verdict,
        "n_models": int(n),
        "n_pass_or_partial": n_pass_or_partial,
        "n_cbit_predicts_trajectory_quality": n_cbit_tq,
        "n_stagewise_beats_direct_for_tq": n_stage_beats,
        "n_trajectory_quality_predicts_answer": n_tq_answer,
        "models": rows,
        "interpretation": (
            "PSA-4D tests the corrected main-chain model after 4B/4C: Cbit may first predict "
            "trajectory quality, and trajectory quality may then predict answer quality. This separates "
            "internal cognitive quality from final endpoint scoring."
        ),
        "terminology": "Cbit / trajectory quality / answer quality / stagewise transition; no basin terminology",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen", "llama", "gemma"], choices=list(MODEL_PATHS.keys()))
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--n_base", type=int, default=80)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--n_clusters", type=int, default=5)
    parser.add_argument("--pca_dim", type=int, default=32)
    parser.add_argument("--window_mode", default="scan", choices=["relative", "scan"])
    parser.add_argument("--score_mode", default="mean", choices=["mean", "sum", "first"])
    parser.add_argument("--local_files_only", action="store_true", default=True)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_df, pair_df = build_dataset(args.n_base)
    prompt_df.to_csv(out_dir / "psa4d_prompts.csv", index=False, encoding="utf-8-sig")
    pair_df.to_csv(out_dir / "psa4d_pairs.csv", index=False, encoding="utf-8-sig")

    model_summaries = []
    for model_key in args.models:
        model_dir = out_dir / model_key
        result = run_one_model(model_key, args, prompt_df, pair_df, model_dir)
        model_summaries.append(result)

    agg_df, agg = aggregate_results(model_summaries)
    agg_df.to_csv(out_dir / "psa4d_crossmodel_summary_table.csv", index=False, encoding="utf-8-sig")

    agg["config"] = vars(args)
    with open(out_dir / "psa4d_crossmodel_summary.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    with open(out_dir / "psa4d_crossmodel_verdict.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(agg, ensure_ascii=False, indent=2))

    print("=" * 80)
    print("CROSS-MODEL VERDICT")
    print(json.dumps(agg, ensure_ascii=False, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()
