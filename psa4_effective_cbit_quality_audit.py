# -*- coding: utf-8 -*-
"""
PSA-4: Effective Cbit -> Cognitive Quality Audit
================================================

Purpose
-------
PSA-3 established that structure-level possibility spaces are measurable.
PSA-4 tests the next main-chain claim:

    structure-level Cbit predicts cognitive quality.

This script audits whether stagewise family-transition Cbit predicts answer
quality on synthetic graph-reasoning tasks with known ground-truth endpoints.

Main objects
------------
1. F_shape -> F_entry -> F_commit -> F_final family transitions.
2. Per-sample transition Cbit from OOF transition tables:
       log2 P(dst|src) - log2 P(dst)
3. Stagewise structure Cbit:
       Cbit(shape->entry) + Cbit(entry->commit) + Cbit(commit->final)
4. Answer quality:
       candidate log-prob margin(correct endpoint vs strongest wrong endpoint)
       answer_correct = margin > 0
5. Effective Cbit proxy:
       stagewise_cbit * transition_stability

Terminology
-----------
Uses shape / entry / commit / final / trajectory / commitment.
No basin terminology.

Run smoke:
    python psa4_effective_cbit_quality_audit.py --n_base 40 --topk 30

Formal:
    python psa4_effective_cbit_quality_audit.py --n_base 120 --topk 50
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

DEFAULT_MODEL_KEY = "qwen"
DEFAULT_OUT_DIR = r"C:\Users\ZH\Desktop\AGI\outputs\psa4_effective_cbit_outputs"
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

    rng.shuffle(lines[1:-1])
    prompt = (
        f"{surface}\n"
        f"Record ID: {base_id}-{family}\n"
        + "\n".join(lines)
        + "\nQuestion: Which endpoint is licensed by this record?\n"
        + "Answer with one endpoint name only."
    )
    candidates = [correct] + wrongs
    return {
        "prompt": prompt,
        "correct": correct,
        "wrong_candidates": wrongs,
        "all_candidates": candidates,
    }


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
    return {"prompt": prompt, "correct": correct, "wrong_candidates": wrongs, "all_candidates": [correct] + wrongs}


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

        out = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

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

    return {
        "hidden": torch.cat(all_hidden, dim=0).numpy(),
        "topk_entropy": torch.cat(all_ent, dim=0).numpy(),
        "topk_spread": torch.cat(all_spread, dim=0).numpy(),
        "topk_center": torch.cat(all_center, dim=0).numpy(),
    }


@torch.no_grad()
def score_candidates(tokenizer, model, pair_df, prompt_df, device, max_len=256):
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

            # token positions in target corresponding to candidate suffix
            start_t = max(0, cand_start - 1)
            if start_t >= target.shape[1]:
                score = -1e9
            else:
                lp = logp[:, start_t:, :].gather(-1, target[:, start_t:].unsqueeze(-1)).squeeze(-1)
                score = float(lp.mean().detach().cpu().item())
            scores[cand] = score

        correct_score = scores[correct]
        wrong_scores = [v for k, v in scores.items() if k != correct]
        best_wrong = max(wrong_scores)
        margin = correct_score - best_wrong
        pred = max(scores.items(), key=lambda kv: kv[1])[0]

        out_rows.append({
            "pair_id": int(row["pair_id"]),
            "correct": correct,
            "pred_candidate": pred,
            "answer_correct": int(pred == correct),
            "answer_margin": float(margin),
            "correct_score": float(correct_score),
            "best_wrong_score": float(best_wrong),
            "candidate_scores": json.dumps(scores, ensure_ascii=False),
        })
        if (i + 1) % 20 == 0:
            print(f"[score] {i + 1}/{len(pair_df)}")

    return pd.DataFrame(out_rows)


def build_delta_features(feats, pair_df):
    clean = pair_df["clean_row_id"].values.astype(int)
    cond = pair_df["cond_row_id"].values.astype(int)
    return {
        "hidden_delta": feats["hidden"][cond] - feats["hidden"][clean],
        "topk_entropy_delta": feats["topk_entropy"][cond] - feats["topk_entropy"][clean],
        "topk_spread_delta": feats["topk_spread"][cond] - feats["topk_spread"][clean],
        "topk_center_delta": feats["topk_center"][cond] - feats["topk_center"][clean],
    }


def clamp_window(start, end, L):
    return [i for i in range(start, min(end + 1, L)) if 0 <= i < L]


def scalar_window(arr, idx):
    return arr[:, idx].mean(axis=1)


def center_window(arr, idx):
    return arr[:, idx, :].mean(axis=1)


def standard_pca(X, dim, seed=SEED):
    Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    d = min(dim, Xs.shape[0] - 1, Xs.shape[1])
    if d >= 2:
        return PCA(n_components=d, random_state=seed).fit_transform(Xs)
    return Xs


def window_features(delta, window):
    hd = delta["hidden_delta"]
    tc = delta["topk_center_delta"]
    te = delta["topk_entropy_delta"]
    ts = delta["topk_spread_delta"]
    return np.concatenate([
        center_window(hd, window),
        center_window(tc, window),
        scalar_window(te, window)[:, None],
        scalar_window(ts, window)[:, None],
    ], axis=1)


def make_family(delta, window, n_clusters=5, pca_dim=32, seed=SEED):
    X = window_features(delta, window)
    Xp = standard_pca(X, pca_dim, seed=seed)
    k = min(n_clusters, max(2, Xp.shape[0] // 10))
    labels = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(Xp)
    return labels.astype(int)


def transition_oof_cbit(src, dst, groups, alpha=1.0):
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
        "prior_surprisal": prior_surprisal,
        "conditional_surprisal": conditional_surprisal,
        "conditional_prob": conditional_prob,
        "mean_cbit": float(np.mean(cbit)),
        "mean_conditional_prob": float(np.mean(conditional_prob)),
    }


def safe_corr(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(pearsonr(x, y)[0])


def reg_eval(X, y, groups):
    y = np.asarray(y).astype(float)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", default=DEFAULT_MODEL_KEY, choices=list(MODEL_PATHS.keys()))
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--n_base", type=int, default=80)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--n_clusters", type=int, default=5)
    parser.add_argument("--pca_dim", type=int, default=32)
    parser.add_argument("--local_files_only", action="store_true", default=True)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PSA-4: Effective Cbit -> Cognitive Quality Audit")
    print("=" * 80)
    print(f"model_key={args.model_key}")
    print(f"model_path={MODEL_PATHS[args.model_key]}")
    print(f"n_base={args.n_base}")
    print(f"out_dir={out_dir.resolve()}")

    prompt_df, pair_df = build_dataset(args.n_base)
    prompt_df.to_csv(out_dir / "psa4_prompts.csv", index=False, encoding="utf-8-sig")
    pair_df.to_csv(out_dir / "psa4_pairs.csv", index=False, encoding="utf-8-sig")

    tokenizer, model = load_model(args.model_key, args.device, args.dtype, args.local_files_only)

    feats = extract_hidden_topk(tokenizer, model, prompt_df, args.device, args.max_len, args.topk, batch_size=1)
    delta = build_delta_features(feats, pair_df)

    quality = score_candidates(tokenizer, model, pair_df, prompt_df, args.device, args.max_len)
    quality.to_csv(out_dir / "psa4_answer_quality.csv", index=False, encoding="utf-8-sig")

    L = delta["hidden_delta"].shape[1]
    windows = {
        "shape": clamp_window(7, 19, L),
        "entry": clamp_window(20, 22, L),
        "commit": clamp_window(23, 25, L),
        "final": clamp_window(26, 27, L),
    }
    if not windows["commit"]:
        windows["commit"] = clamp_window(max(0, L - 5), L - 2, L)
    if not windows["final"]:
        windows["final"] = clamp_window(max(0, L - 2), L - 1, L)

    families = {}
    for name in ["shape", "entry", "commit", "final"]:
        families[name] = make_family(delta, windows[name], n_clusters=args.n_clusters, pca_dim=args.pca_dim, seed=SEED + len(name))
        pair_df[f"{name}_family"] = families[name]

    groups = pair_df["base_id"].values.astype(int)

    t_se = transition_oof_cbit(families["shape"], families["entry"], groups)
    t_ec = transition_oof_cbit(families["entry"], families["commit"], groups)
    t_cf = transition_oof_cbit(families["commit"], families["final"], groups)
    t_sf = transition_oof_cbit(families["shape"], families["final"], groups)
    t_sc = transition_oof_cbit(families["shape"], families["commit"], groups)

    pair_df = pair_df.merge(quality, on="pair_id", how="left")

    pair_df["cbit_shape_entry"] = t_se["cbit"]
    pair_df["cbit_entry_commit"] = t_ec["cbit"]
    pair_df["cbit_commit_final"] = t_cf["cbit"]
    pair_df["cbit_shape_final_direct"] = t_sf["cbit"]
    pair_df["cbit_shape_commit_direct"] = t_sc["cbit"]
    pair_df["stagewise_cbit"] = pair_df["cbit_shape_entry"] + pair_df["cbit_entry_commit"] + pair_df["cbit_commit_final"]
    pair_df["transition_stability"] = t_se["conditional_prob"] * t_ec["conditional_prob"] * t_cf["conditional_prob"]
    pair_df["effective_cbit_proxy"] = pair_df["stagewise_cbit"] * pair_df["transition_stability"]

    pair_df.to_csv(out_dir / "psa4_pairs_with_cbit_quality.csv", index=False, encoding="utf-8-sig")

    y_margin = pair_df["answer_margin"].values.astype(float)
    y_correct = pair_df["answer_correct"].values.astype(int)

    feature_sets = {
        "direct_cbit": ["cbit_shape_final_direct", "cbit_shape_commit_direct"],
        "stagewise_cbit": ["cbit_shape_entry", "cbit_entry_commit", "cbit_commit_final", "stagewise_cbit"],
        "effective_cbit": ["stagewise_cbit", "transition_stability", "effective_cbit_proxy"],
        "all_cbit": [
            "cbit_shape_entry", "cbit_entry_commit", "cbit_commit_final",
            "cbit_shape_final_direct", "cbit_shape_commit_direct",
            "stagewise_cbit", "transition_stability", "effective_cbit_proxy"
        ],
    }

    regression = {}
    classification = {}
    for name, cols in feature_sets.items():
        X = pair_df[cols].values
        regression[name + "_to_answer_margin"] = reg_eval(X, y_margin, groups)
        classification[name + "_to_answer_correct"] = clf_eval(X, y_correct, groups)

    corr_stage_margin = safe_corr(pair_df["stagewise_cbit"], y_margin)
    corr_eff_margin = safe_corr(pair_df["effective_cbit_proxy"], y_margin)
    corr_direct_margin = safe_corr(pair_df["cbit_shape_final_direct"], y_margin)

    correct_rate = float(pair_df["answer_correct"].mean())
    quality_degenerate = correct_rate < 0.05 or correct_rate > 0.95

    stage_r2 = regression["stagewise_cbit_to_answer_margin"]["r2"]
    direct_r2 = regression["direct_cbit_to_answer_margin"]["r2"]
    eff_r2 = regression["effective_cbit_to_answer_margin"]["r2"]

    stage_auc = classification["stagewise_cbit_to_answer_correct"]["auc"]
    direct_auc = classification["direct_cbit_to_answer_correct"]["auc"]
    eff_auc = classification["effective_cbit_to_answer_correct"]["auc"]

    stage_predicts = (
        (corr_stage_margin is not None and corr_stage_margin > 0.20)
        or (stage_auc is not None and stage_auc > 0.60)
        or stage_r2 > 0.05
    )
    effective_improves = (
        eff_r2 > stage_r2 + 0.02
        or (eff_auc is not None and stage_auc is not None and eff_auc > stage_auc + 0.03)
        or (corr_eff_margin is not None and corr_stage_margin is not None and corr_eff_margin > corr_stage_margin + 0.03)
    )
    stage_beats_direct = (
        stage_r2 > direct_r2 + 0.03
        or (stage_auc is not None and direct_auc is not None and stage_auc > direct_auc + 0.03)
    )

    if quality_degenerate:
        verdict = "FAIL_DEGENERATE_ANSWER_QUALITY"
    elif stage_predicts and effective_improves and stage_beats_direct:
        verdict = "PASS_PSA4_EFFECTIVE_CBIT_PREDICTS_QUALITY"
    elif stage_predicts and stage_beats_direct:
        verdict = "PARTIAL_PSA4_STAGEWISE_CBIT_PREDICTS_QUALITY"
    elif stage_predicts:
        verdict = "PARTIAL_PSA4_CBIT_QUALITY_SIGNAL"
    else:
        verdict = "FAIL_OR_CAVEAT_PSA4"

    out = {
        "verdict": verdict,
        "pass_flags": {
            "quality_non_degenerate": not bool(quality_degenerate),
            "stagewise_cbit_predicts_quality": bool(stage_predicts),
            "effective_cbit_improves": bool(effective_improves),
            "stagewise_beats_direct": bool(stage_beats_direct),
        },
        "summary_metrics": {
            "answer_correct_rate": correct_rate,
            "answer_margin_mean": float(pair_df["answer_margin"].mean()),
            "answer_margin_std": float(pair_df["answer_margin"].std()),
            "mean_cbit_shape_entry": t_se["mean_cbit"],
            "mean_cbit_entry_commit": t_ec["mean_cbit"],
            "mean_cbit_commit_final": t_cf["mean_cbit"],
            "mean_cbit_shape_final_direct": t_sf["mean_cbit"],
            "mean_stagewise_cbit": float(pair_df["stagewise_cbit"].mean()),
            "mean_transition_stability": float(pair_df["transition_stability"].mean()),
            "mean_effective_cbit_proxy": float(pair_df["effective_cbit_proxy"].mean()),
            "corr_stagewise_cbit_answer_margin": corr_stage_margin,
            "corr_effective_cbit_answer_margin": corr_eff_margin,
            "corr_direct_cbit_answer_margin": corr_direct_margin,
            "stagewise_margin_r2": stage_r2,
            "direct_margin_r2": direct_r2,
            "effective_margin_r2": eff_r2,
            "stagewise_correct_auc": stage_auc,
            "direct_correct_auc": direct_auc,
            "effective_correct_auc": eff_auc,
        },
        "regression_results": regression,
        "classification_results": classification,
        "windows": windows,
        "config": vars(args),
        "model_path": MODEL_PATHS[args.model_key],
        "interpretation": (
            "PSA-4 tests the main cognitive-ontology claim after PSA-3: structural Cbit must not only exist; "
            "it should predict cognitive quality. Stagewise Cbit is compared against direct Cbit, and effective "
            "Cbit is approximated by multiplying stagewise Cbit by transition stability."
        ),
        "terminology": "Cbit / effective Cbit / stagewise transition / cognitive quality; no basin terminology",
    }

    with open(out_dir / "psa4_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(out_dir / "psa4_verdict.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False, indent=2))

    print("=" * 80)
    print("VERDICT")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()
