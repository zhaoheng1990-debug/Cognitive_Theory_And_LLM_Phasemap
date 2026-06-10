# -*- coding: utf-8 -*-
"""
PSA-3C.1: Stagewise FamilyTransition Cbit Audit
===============================================

Purpose
-------
PSA-3C rejected the direct F_shape -> F_commit hypothesis, but revealed a
stronger staged structure:

    F_shape -> F_entry -> F_commit -> F_final

This script re-runs the same family construction but evaluates the stagewise
transition chain explicitly.

Core observables
----------------
1. Cbit(shape -> entry)
2. Cbit(entry -> commit)
3. Cbit(commit -> final)
4. Cbit(shape -> commit)
5. Cbit(shape -> final)
6. Stagewise total Cbit = sum of adjacent transition Cbits
7. Direct-vs-stagewise gain
8. Markov compression:
       H(final | shape)
       H(final | entry)
       H(final | commit)
9. Transition bottleneck:
       weakest adjacent stage by Cbit/consistency
10. Feature support:
       entry features -> commit family
       commit features -> final family

Terminology
-----------
Uses shape / entry / commit / final families.
Uses selection / commitment language. No basin terminology.

Run smoke:
    python psa3c1_stagewise_family_transition_cbit_audit.py --n_base 40 --topk 30

Formal:
    python psa3c1_stagewise_family_transition_cbit_audit.py --n_base 120 --topk 50
"""

import json
import math
import random
import argparse
from pathlib import Path
from typing import Any, List, Tuple

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
    adjusted_rand_score,
    normalized_mutual_info_score,
    r2_score,
    mean_squared_error,
    mutual_info_score,
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import pearsonr, spearmanr


MODEL_PATHS = {
    "qwen": r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main",
    "llama": r"D:\model\Llama-3.2-1B-Instruct",
    "gemma": r"D:\model\gemma-2-2b-it",
}

DEFAULT_MODEL_KEY = "qwen"
DEFAULT_OUT_DIR = r"C:\Users\ZH\Desktop\AGI\outputs\psa3c1_stagewise_outputs"
SEED = 20260607

DIAGNOSTIC_MECHANISMS = ["chain", "fork", "triangle", "cycle", "fragmented"]
MECH_TO_ID = {m: i for i, m in enumerate(DIAGNOSTIC_MECHANISMS)}

SURFACES = [
    "Use the directed record below to infer the licensed target.",
    "Read the symbolic relation record and choose the licensed target.",
    "Inspect the directed edges and report the licensed target.",
    "Resolve the target from the relation record.",
    "Use only the listed directed edges to infer the target.",
    "Determine the licensed target from the symbolic record.",
]

NAME_POOL = [
    "Aster", "Boreal", "Cobalt", "Dune", "Ember", "Fjord", "Glade", "Harbor",
    "Ivory", "Jade", "Kite", "Lumen", "Mosaic", "Nimbus", "Opal", "Quartz",
    "Raven", "Sol", "Topaz", "Umber", "Vega", "Willow", "Xylo", "Yarrow",
    "Zenith", "Atlas", "Copper", "Falcon", "Garden", "Island", "Lagoon",
    "Marble", "Onyx", "Pearl", "Ridge", "Station", "Temple", "Valley"
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


def topology_edges(mech: str, nodes: List[str]) -> List[Tuple[str, str]]:
    a, b, c, d, e = nodes[:5]
    if mech == "chain":
        return [(a, b), (b, c), (c, d), (d, e)]
    if mech == "fork":
        return [(a, b), (a, c), (b, d), (c, e)]
    if mech == "triangle":
        return [(a, b), (b, c), (a, c), (c, d)]
    if mech == "cycle":
        return [(a, b), (b, c), (c, a), (c, d)]
    if mech == "fragmented":
        return [(a, b), (c, d), (d, e), (b, a)]
    raise ValueError(mech)


def make_prompt(base_id: int, surface_id: int, mech: str, rng: random.Random) -> str:
    names = rng.sample(NAME_POOL, 5)
    edges = topology_edges(mech, names)
    rng.shuffle(edges)
    surface = SURFACES[surface_id % len(SURFACES)]
    edge_lines = [f"E{i+1}: {u} -> {v}." for i, (u, v) in enumerate(edges)]
    start = names[0]
    return (
        f"{surface}\n"
        f"Record ID: {base_id}-{surface_id}\n"
        + "\n".join(edge_lines)
        + f"\nStart: {start}.\n"
        + "Question: Which target is licensed by this record?\n"
        + "Answer with one target name only."
    )


def make_clean_prompt(base_id: int, surface_id: int, rng: random.Random) -> str:
    names = rng.sample(NAME_POOL, 5)
    a, b, c, d, e = names
    edges = [(a, b), (b, c), (c, d), (d, e)]
    rng.shuffle(edges)
    surface = SURFACES[surface_id % len(SURFACES)]
    edge_lines = [f"E{i+1}: {u} -> {v}." for i, (u, v) in enumerate(edges)]
    return (
        f"{surface}\n"
        f"Record ID: {base_id}-{surface_id}\n"
        + "\n".join(edge_lines)
        + f"\nStart: {a}.\n"
        + "Question: Which target is licensed by this record?\n"
        + "Answer with one target name only."
    )


def build_dataset(n_base: int = 80):
    rows = []
    pairs = []
    rid = 0
    for base_id in range(n_base):
        surface_id = base_id % len(SURFACES)
        clean_rng = random.Random(SEED * 100000 + base_id * 100 + 777)
        clean_row_id = rid
        rows.append({
            "prompt_row_id": rid,
            "base_id": base_id,
            "surface_id": surface_id,
            "kind": "clean",
            "mechanism": "clean",
            "mechanism_id": -1,
            "prompt": make_clean_prompt(base_id, surface_id, clean_rng),
        })
        rid += 1

        for mech in DIAGNOSTIC_MECHANISMS:
            rng = random.Random(SEED * 100000 + base_id * 1000 + MECH_TO_ID[mech])
            cond_row_id = rid
            rows.append({
                "prompt_row_id": rid,
                "base_id": base_id,
                "surface_id": surface_id,
                "kind": "condition",
                "mechanism": mech,
                "mechanism_id": MECH_TO_ID[mech],
                "prompt": make_prompt(base_id, surface_id, mech, rng),
            })
            pairs.append({
                "pair_id": len(pairs),
                "base_id": base_id,
                "surface_id": surface_id,
                "clean_row_id": clean_row_id,
                "cond_row_id": cond_row_id,
                "diagnostic_mechanism": mech,
                "diagnostic_mechanism_id": MECH_TO_ID[mech],
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


def flatten_window(arr, idx):
    return arr[:, idx, :].reshape(arr.shape[0], -1)


def flow_window(arr, idx):
    if len(idx) < 2:
        return np.zeros((arr.shape[0], arr.shape[2]), dtype=np.float32)
    sub = arr[:, idx, :]
    return np.diff(sub, axis=1).reshape(arr.shape[0], -1)


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
    return labels.astype(int), Xp


def entropy_from_counts(counts):
    counts = np.asarray(counts, dtype=float)
    probs = counts / (counts.sum() + 1e-12)
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def conditional_entropy(x, y):
    x = np.asarray(x).astype(int)
    y = np.asarray(y).astype(int)
    out = 0.0
    for xv in np.unique(x):
        mask = x == xv
        out += mask.mean() * entropy_from_counts(np.bincount(y[mask]))
    return float(out)


def transition_stats(src, dst, src_name, dst_name):
    src = np.asarray(src).astype(int)
    dst = np.asarray(dst).astype(int)
    H_dst = entropy_from_counts(np.bincount(dst))
    H_dst_given_src = conditional_entropy(src, dst)
    cbit = H_dst - H_dst_given_src
    mi = float(mutual_info_score(src, dst) / math.log(2.0))
    nmi = float(normalized_mutual_info_score(src, dst))

    rows = []
    consistency_vals = []
    for s in sorted(np.unique(src)):
        mask = src == s
        counts = np.bincount(dst[mask], minlength=int(dst.max()) + 1)
        total = int(counts.sum())
        top = int(counts.argmax())
        top_count = int(counts.max())
        consistency = top_count / max(total, 1)
        consistency_vals.append(consistency)
        rows.append({
            src_name: int(s),
            "n": total,
            f"major_{dst_name}": top,
            "major_count": top_count,
            "consistency": float(consistency),
            "entropy": entropy_from_counts(counts),
            "distribution": {str(i): int(c) for i, c in enumerate(counts) if c > 0},
        })

    return {
        "src": src_name,
        "dst": dst_name,
        "H_dst": H_dst,
        "H_dst_given_src": H_dst_given_src,
        "transition_Cbit": cbit,
        "MI_bits": mi,
        "NMI": nmi,
        "mean_consistency": float(np.mean(consistency_vals)),
        "min_consistency": float(np.min(consistency_vals)),
        "transition_rows": rows,
    }


def chain_transition_cbit(labels_dict):
    stages = ["shape", "entry", "commit", "final"]
    stats = {}
    for a, b in zip(stages[:-1], stages[1:]):
        stats[f"{a}_to_{b}"] = transition_stats(labels_dict[a], labels_dict[b], f"{a}_family", f"{b}_family")

    direct = {
        "shape_to_commit": transition_stats(labels_dict["shape"], labels_dict["commit"], "shape_family", "commit_family"),
        "shape_to_final": transition_stats(labels_dict["shape"], labels_dict["final"], "shape_family", "final_family"),
        "entry_to_final": transition_stats(labels_dict["entry"], labels_dict["final"], "entry_family", "final_family"),
    }

    adjacent_total = sum(stats[k]["transition_Cbit"] for k in stats)
    direct_shape_final = direct["shape_to_final"]["transition_Cbit"]
    stagewise_gain_over_direct = adjacent_total - direct_shape_final

    return stats, direct, {
        "adjacent_total_Cbit": float(adjacent_total),
        "direct_shape_to_final_Cbit": float(direct_shape_final),
        "stagewise_gain_over_direct": float(stagewise_gain_over_direct),
        "weakest_adjacent_by_Cbit": min(stats.items(), key=lambda kv: kv[1]["transition_Cbit"])[0],
        "weakest_adjacent_Cbit": float(min(v["transition_Cbit"] for v in stats.values())),
        "weakest_adjacent_by_consistency": min(stats.items(), key=lambda kv: kv[1]["mean_consistency"])[0],
        "weakest_adjacent_consistency": float(min(v["mean_consistency"] for v in stats.values())),
    }


def onehot(labels):
    labels = np.asarray(labels).astype(int)
    k = int(labels.max()) + 1
    return np.eye(k, dtype=np.float32)[labels]


def group_cv_classifier(X, y, groups):
    y = np.asarray(y).astype(int)
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced", C=1.0)
    )
    pred = cross_val_predict(clf, X, y, cv=cv.split(X, y, groups))
    return {
        "acc": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }


def text_leakage(prompt_df, pair_df, labels):
    texts = prompt_df.set_index("prompt_row_id").loc[pair_df["cond_row_id"].values, "prompt"].tolist()
    groups = pair_df["base_id"].values.astype(int)
    y = labels.astype(int)
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    pipe = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2),
        LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced", C=1.0)
    )
    pred = cross_val_predict(pipe, texts, y, cv=cv.split(texts, y, groups))
    return {
        "acc": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }


def build_blocks(delta, windows):
    hd = delta["hidden_delta"]
    tc = delta["topk_center_delta"]
    te = delta["topk_entropy_delta"]
    ts = delta["topk_spread_delta"]
    n = hd.shape[0]

    blocks = {}
    for name, win in windows.items():
        blocks[name + "_mean"] = window_features(delta, win)
        blocks[name + "_flat"] = standard_pca(
            np.concatenate([flatten_window(hd, win), flatten_window(tc, win)], axis=1),
            64
        )

    shape_w = windows["shape"]
    entry_w = windows["entry"]
    commit_w = windows["commit"]

    blocks["flow_shape"] = standard_pca(
        np.concatenate([
            flow_window(hd, shape_w),
            flow_window(tc, shape_w),
            np.diff(te[:, shape_w], axis=1) if len(shape_w) >= 2 else np.zeros((n, 1)),
            np.diff(ts[:, shape_w], axis=1) if len(shape_w) >= 2 else np.zeros((n, 1)),
        ], axis=1),
        64
    )

    blocks["flow_entry"] = standard_pca(
        np.concatenate([
            flow_window(hd, entry_w),
            flow_window(tc, entry_w),
            np.diff(te[:, entry_w], axis=1) if len(entry_w) >= 2 else np.zeros((n, 1)),
            np.diff(ts[:, entry_w], axis=1) if len(entry_w) >= 2 else np.zeros((n, 1)),
        ], axis=1),
        64
    )

    blocks["flow_commit"] = standard_pca(
        np.concatenate([
            flow_window(hd, commit_w),
            flow_window(tc, commit_w),
            np.diff(te[:, commit_w], axis=1) if len(commit_w) >= 2 else np.zeros((n, 1)),
            np.diff(ts[:, commit_w], axis=1) if len(commit_w) >= 2 else np.zeros((n, 1)),
        ], axis=1),
        64
    )

    blocks["shape_plus_entry"] = np.concatenate([blocks["shape_mean"], blocks["entry_mean"]], axis=1)
    blocks["entry_plus_commit"] = np.concatenate([blocks["entry_mean"], blocks["commit_mean"]], axis=1)
    blocks["commit_plus_final"] = np.concatenate([blocks["commit_mean"], blocks["final_mean"]], axis=1)
    return blocks


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
    print("PSA-3C.1: Stagewise FamilyTransition Cbit Audit")
    print("=" * 80)
    print(f"model_key={args.model_key}")
    print(f"model_path={MODEL_PATHS[args.model_key]}")
    print(f"n_base={args.n_base}")
    print(f"out_dir={out_dir.resolve()}")

    prompt_df, pair_df = build_dataset(args.n_base)
    prompt_df.to_csv(out_dir / "psa3c1_prompts.csv", index=False, encoding="utf-8-sig")
    pair_df.to_csv(out_dir / "psa3c1_pairs.csv", index=False, encoding="utf-8-sig")

    tokenizer, model = load_model(args.model_key, args.device, args.dtype, args.local_files_only)
    feats = extract_hidden_topk(tokenizer, model, prompt_df, args.device, args.max_len, args.topk, batch_size=1)
    delta = build_delta_features(feats, pair_df)

    L = delta["hidden_delta"].shape[1]
    windows = {
        "init": clamp_window(0, 6, L),
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
        labels, _ = make_family(delta, windows[name], n_clusters=args.n_clusters, pca_dim=args.pca_dim, seed=SEED + len(name))
        families[name] = labels
        pair_df[f"{name}_family"] = labels

    pair_df.to_csv(out_dir / "psa3c1_pairs_with_families.csv", index=False, encoding="utf-8-sig")

    groups = pair_df["base_id"].values.astype(int)
    diag = pair_df["diagnostic_mechanism_id"].values.astype(int)

    adjacent_stats, direct_stats, chain_summary = chain_transition_cbit(families)

    text = {name: text_leakage(prompt_df, pair_df, families[name]) for name in ["shape", "entry", "commit", "final"]}
    text["diagnostic_mechanism"] = text_leakage(prompt_df, pair_df, diag)

    label_free = {
        "shape_vs_diag_ARI": float(adjusted_rand_score(diag, families["shape"])),
        "entry_vs_diag_ARI": float(adjusted_rand_score(diag, families["entry"])),
        "commit_vs_diag_ARI": float(adjusted_rand_score(diag, families["commit"])),
        "final_vs_diag_ARI": float(adjusted_rand_score(diag, families["final"])),
        "shape_vs_diag_NMI": float(normalized_mutual_info_score(diag, families["shape"])),
        "entry_vs_diag_NMI": float(normalized_mutual_info_score(diag, families["entry"])),
        "commit_vs_diag_NMI": float(normalized_mutual_info_score(diag, families["commit"])),
        "final_vs_diag_NMI": float(normalized_mutual_info_score(diag, families["final"])),
    }

    blocks = build_blocks(delta, windows)

    classification = {}
    targets = {
        "shape_family": families["shape"],
        "entry_family": families["entry"],
        "commit_family": families["commit"],
        "final_family": families["final"],
        "diagnostic_mechanism": diag,
    }
    for block_name, X in blocks.items():
        for target_name, y in targets.items():
            classification[f"{block_name}_to_{target_name}"] = group_cv_classifier(X, y, groups)

    # Family label transition classifiers.
    classification["shape_family_onehot_to_entry_family"] = group_cv_classifier(onehot(families["shape"]), families["entry"], groups)
    classification["entry_family_onehot_to_commit_family"] = group_cv_classifier(onehot(families["entry"]), families["commit"], groups)
    classification["commit_family_onehot_to_final_family"] = group_cv_classifier(onehot(families["commit"]), families["final"], groups)
    classification["shape_family_onehot_to_final_family"] = group_cv_classifier(onehot(families["shape"]), families["final"], groups)

    # Direct and staged compression.
    shape_entry_cbit = adjacent_stats["shape_to_entry"]["transition_Cbit"]
    entry_commit_cbit = adjacent_stats["entry_to_commit"]["transition_Cbit"]
    commit_final_cbit = adjacent_stats["commit_to_final"]["transition_Cbit"]

    text_ok = max(text[name]["macro_f1"] for name in ["shape", "entry", "commit", "final"]) < 0.45
    label_free_ok = max(abs(v) for k, v in label_free.items() if k.endswith("_ARI")) < 0.05
    adjacent_cbit_ok = (
        shape_entry_cbit > 0.15 and
        entry_commit_cbit > 0.25 and
        commit_final_cbit > 0.25
    )
    stagewise_gain_ok = chain_summary["stagewise_gain_over_direct"] > 0.50
    bottleneck_expected = chain_summary["weakest_adjacent_by_Cbit"] == "shape_to_entry"

    entry_commit_feature_ok = (
        classification["entry_mean_to_commit_family"]["macro_f1"] > text["commit"]["macro_f1"] + 0.15 and
        classification["entry_mean_to_commit_family"]["macro_f1"] >= 0.50
    )
    commit_final_feature_ok = (
        classification["commit_mean_to_final_family"]["macro_f1"] > text["final"]["macro_f1"] + 0.15 and
        classification["commit_mean_to_final_family"]["macro_f1"] >= 0.50
    )

    if text_ok and label_free_ok and adjacent_cbit_ok and stagewise_gain_ok and entry_commit_feature_ok and commit_final_feature_ok:
        verdict = "PASS_PSA3C1_STAGEWISE_STRUCTURE_CBIT"
    elif text_ok and label_free_ok and adjacent_cbit_ok and stagewise_gain_ok:
        verdict = "PARTIAL_PSA3C1_STAGEWISE_CBIT_FEATURE_WEAK"
    elif text_ok and label_free_ok and (entry_commit_cbit > 0.25 or commit_final_cbit > 0.25):
        verdict = "PARTIAL_PSA3C1_MAIN_TRANSITIONS_ONLY"
    else:
        verdict = "FAIL_OR_CAVEAT_PSA3C1"

    out = {
        "verdict": verdict,
        "pass_flags": {
            "text_leakage_ok": bool(text_ok),
            "label_free_ok": bool(label_free_ok),
            "adjacent_cbit_ok": bool(adjacent_cbit_ok),
            "stagewise_gain_ok": bool(stagewise_gain_ok),
            "bottleneck_is_shape_to_entry": bool(bottleneck_expected),
            "entry_commit_feature_ok": bool(entry_commit_feature_ok),
            "commit_final_feature_ok": bool(commit_final_feature_ok),
        },
        "summary_metrics": {
            "Cbit_shape_to_entry": float(shape_entry_cbit),
            "Cbit_entry_to_commit": float(entry_commit_cbit),
            "Cbit_commit_to_final": float(commit_final_cbit),
            "Cbit_shape_to_commit_direct": float(direct_stats["shape_to_commit"]["transition_Cbit"]),
            "Cbit_shape_to_final_direct": float(direct_stats["shape_to_final"]["transition_Cbit"]),
            "Cbit_entry_to_final_direct": float(direct_stats["entry_to_final"]["transition_Cbit"]),
            "adjacent_total_Cbit": chain_summary["adjacent_total_Cbit"],
            "stagewise_gain_over_direct_shape_to_final": chain_summary["stagewise_gain_over_direct"],
            "weakest_adjacent_by_Cbit": chain_summary["weakest_adjacent_by_Cbit"],
            "weakest_adjacent_Cbit": chain_summary["weakest_adjacent_Cbit"],
            "weakest_adjacent_by_consistency": chain_summary["weakest_adjacent_by_consistency"],
            "weakest_adjacent_consistency": chain_summary["weakest_adjacent_consistency"],
            "text_shape_macro_f1": text["shape"]["macro_f1"],
            "text_entry_macro_f1": text["entry"]["macro_f1"],
            "text_commit_macro_f1": text["commit"]["macro_f1"],
            "text_final_macro_f1": text["final"]["macro_f1"],
            "shape_mean_to_entry_family_macro_f1": classification["shape_mean_to_entry_family"]["macro_f1"],
            "entry_mean_to_commit_family_macro_f1": classification["entry_mean_to_commit_family"]["macro_f1"],
            "commit_mean_to_final_family_macro_f1": classification["commit_mean_to_final_family"]["macro_f1"],
            "shape_family_onehot_to_entry_family_macro_f1": classification["shape_family_onehot_to_entry_family"]["macro_f1"],
            "entry_family_onehot_to_commit_family_macro_f1": classification["entry_family_onehot_to_commit_family"]["macro_f1"],
            "commit_family_onehot_to_final_family_macro_f1": classification["commit_family_onehot_to_final_family"]["macro_f1"],
        },
        "windows": windows,
        "text_leakage": text,
        "label_free_diagnostics": label_free,
        "adjacent_transitions": adjacent_stats,
        "direct_transitions": direct_stats,
        "chain_summary": chain_summary,
        "classification_results": classification,
        "config": vars(args),
        "model_path": MODEL_PATHS[args.model_key],
        "terminology": "stagewise family transition / selection / commitment / final readout; no basin terminology",
        "interpretation": (
            "PSA-3C.1 tests whether structure-level Cbit is stagewise. The expected pattern is "
            "weak shape->entry pre-organization, strong entry->commit compression, and strong "
            "commit->final stabilization."
        ),
    }

    with open(out_dir / "psa3c1_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(out_dir / "psa3c1_verdict.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False, indent=2))

    print("=" * 80)
    print("VERDICT")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()
