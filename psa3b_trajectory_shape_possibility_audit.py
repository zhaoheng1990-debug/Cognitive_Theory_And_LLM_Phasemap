# -*- coding: utf-8 -*-
"""
PSA-3B: Trajectory-Shape Possibility Space Audit
================================================

Purpose
-------
PSA-3A.5B established label-free latent trajectory families in selection /
commitment geometry. PSA-3B tests whether these latent families are already
encoded in trajectory-shape space and layerwise flow.

Main hypotheses
---------------
H1. H_shape features from L7-L19 recover latent trajectory families better than
    init-only features and text-only leakage.
H2. O-flow / TopK-VIM flow improves or matches H_shape recovery.
H3. H_shape / flow features predict the later trajectory dominance observable
    derived from selection-window latent-family geometry.
H4. Diagnostic mechanism labels are weaker / different from label-free latent
    trajectory families.

Terminology
-----------
Uses selection / commitment / trajectory family. No basin terminology.

Run smoke:
    python psa3b_trajectory_shape_possibility_audit.py --n_base 40 --topk 30

Formal:
    python psa3b_trajectory_shape_possibility_audit.py --n_base 120 --topk 50
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
    roc_auc_score,
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
DEFAULT_OUT_DIR = r"C:\Users\ZH\Desktop\AGI\outputs\psa3b_trajectory_shape_outputs"
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


def make_latent_families(delta, n_clusters=5, pca_dim=32):
    hd = delta["hidden_delta"]
    tc = delta["topk_center_delta"]
    te = delta["topk_entropy_delta"]
    ts = delta["topk_spread_delta"]
    n, L, d = hd.shape
    selection_w = clamp_window(20, 25, L)
    if not selection_w:
        selection_w = clamp_window(max(0, L - 6), L - 2, L)

    X = np.concatenate([
        center_window(hd, selection_w),
        center_window(tc, selection_w),
        scalar_window(te, selection_w)[:, None],
        scalar_window(ts, selection_w)[:, None],
    ], axis=1)

    Xp = standard_pca(X, pca_dim)
    k = min(n_clusters, max(2, Xp.shape[0] // 10))
    labels = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(Xp)

    # trajectory dominance observable:
    # margin between nearest and second-nearest cluster centroids in selection geometry.
    centroids = np.stack([Xp[labels == c].mean(axis=0) for c in range(k)], axis=0)
    dist = np.linalg.norm(Xp[:, None, :] - centroids[None, :, :], axis=-1)
    sorted_dist = np.sort(dist, axis=1)
    dominance_margin = sorted_dist[:, 1] - sorted_dist[:, 0]
    dominance_margin = (dominance_margin - dominance_margin.mean()) / (dominance_margin.std() + 1e-12)

    return labels.astype(int), dominance_margin.astype(np.float32), {
        "n_clusters": int(k),
        "pca_dim": int(Xp.shape[1]),
        "selection_window": selection_w,
    }


def safe_corr(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(pearsonr(x, y)[0])


def group_cv_classifier(X, y, groups):
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


def group_cv_regressor(X, y, groups):
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


def build_feature_blocks(delta):
    hd = delta["hidden_delta"]
    tc = delta["topk_center_delta"]
    te = delta["topk_entropy_delta"]
    ts = delta["topk_spread_delta"]

    n, L, d = hd.shape
    init_w = clamp_window(0, 6, L)
    shape_w = clamp_window(7, 19, L)
    selection_w = clamp_window(20, 25, L)
    if not selection_w:
        selection_w = clamp_window(max(0, L - 6), L - 2, L)

    # Compact H_shape: mean, PCA-like trajectory summary through flatten window.
    X_init = np.concatenate([
        center_window(hd, init_w),
        center_window(tc, init_w),
        scalar_window(te, init_w)[:, None],
        scalar_window(ts, init_w)[:, None],
    ], axis=1)

    X_shape = np.concatenate([
        center_window(hd, shape_w),
        center_window(tc, shape_w),
        scalar_window(te, shape_w)[:, None],
        scalar_window(ts, shape_w)[:, None],
    ], axis=1)

    X_selection = np.concatenate([
        center_window(hd, selection_w),
        center_window(tc, selection_w),
        scalar_window(te, selection_w)[:, None],
        scalar_window(ts, selection_w)[:, None],
    ], axis=1)

    X_hshape_flat = np.concatenate([
        flatten_window(hd, shape_w),
        flatten_window(tc, shape_w),
    ], axis=1)

    X_oflow = np.concatenate([
        flow_window(hd, shape_w),
        flow_window(tc, shape_w),
        np.diff(te[:, shape_w], axis=1) if len(shape_w) >= 2 else np.zeros((n, 1)),
        np.diff(ts[:, shape_w], axis=1) if len(shape_w) >= 2 else np.zeros((n, 1)),
    ], axis=1)

    X_topo_flow = np.concatenate([
        flow_window(tc, shape_w),
        np.diff(te[:, shape_w], axis=1) if len(shape_w) >= 2 else np.zeros((n, 1)),
        np.diff(ts[:, shape_w], axis=1) if len(shape_w) >= 2 else np.zeros((n, 1)),
    ], axis=1)

    return {
        "init": X_init,
        "shape_mean": X_shape,
        "shape_flat": standard_pca(X_hshape_flat, 64),
        "o_flow": standard_pca(X_oflow, 64),
        "topo_flow": standard_pca(X_topo_flow, 64),
        "selection": X_selection,
        "shape_plus_oflow": None,
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
    print("PSA-3B: Trajectory-Shape Possibility Space Audit")
    print("=" * 80)
    print(f"model_key={args.model_key}")
    print(f"model_path={MODEL_PATHS[args.model_key]}")
    print(f"n_base={args.n_base}")
    print(f"out_dir={out_dir.resolve()}")

    prompt_df, pair_df = build_dataset(args.n_base)
    prompt_df.to_csv(out_dir / "psa3b_prompts.csv", index=False, encoding="utf-8-sig")
    pair_df.to_csv(out_dir / "psa3b_pairs.csv", index=False, encoding="utf-8-sig")

    tokenizer, model = load_model(args.model_key, args.device, args.dtype, args.local_files_only)
    feats = extract_hidden_topk(tokenizer, model, prompt_df, args.device, args.max_len, args.topk, batch_size=1)
    delta = build_delta_features(feats, pair_df)

    latent_labels, dominance_margin, cluster_info = make_latent_families(
        delta,
        n_clusters=args.n_clusters,
        pca_dim=args.pca_dim,
    )
    pair_df["latent_family_id"] = latent_labels
    pair_df["trajectory_dominance_margin"] = dominance_margin
    pair_df.to_csv(out_dir / "psa3b_pairs_with_latent_family.csv", index=False, encoding="utf-8-sig")

    groups = pair_df["base_id"].values.astype(int)
    diagnostic_labels = pair_df["diagnostic_mechanism_id"].values.astype(int)

    text_latent = text_leakage(prompt_df, pair_df, latent_labels)
    text_diag = text_leakage(prompt_df, pair_df, diagnostic_labels)

    blocks = build_feature_blocks(delta)
    blocks["shape_plus_oflow"] = np.concatenate([blocks["shape_mean"], blocks["o_flow"]], axis=1)

    cls_results = {}
    reg_results = {}

    for name, X in blocks.items():
        cls_results[name + "_to_latent_family"] = group_cv_classifier(X, latent_labels, groups)
        cls_results[name + "_to_diagnostic_mechanism"] = group_cv_classifier(X, diagnostic_labels, groups)
        reg_results[name + "_to_dominance_margin"] = group_cv_regressor(X, dominance_margin, groups)

    ari = float(adjusted_rand_score(diagnostic_labels, latent_labels))
    nmi = float(normalized_mutual_info_score(diagnostic_labels, latent_labels))

    best_latent = max(
        ((k, v["macro_f1"]) for k, v in cls_results.items() if k.endswith("_to_latent_family")),
        key=lambda kv: kv[1],
    )
    best_diag = max(
        ((k, v["macro_f1"]) for k, v in cls_results.items() if k.endswith("_to_diagnostic_mechanism")),
        key=lambda kv: kv[1],
    )
    best_dom = max(
        ((k, v["r2"]) for k, v in reg_results.items()),
        key=lambda kv: kv[1],
    )

    hshape_f1 = cls_results["shape_mean_to_latent_family"]["macro_f1"]
    hshape_flat_f1 = cls_results["shape_flat_to_latent_family"]["macro_f1"]
    init_f1 = cls_results["init_to_latent_family"]["macro_f1"]
    text_f1 = text_latent["macro_f1"]
    oflow_f1 = cls_results["o_flow_to_latent_family"]["macro_f1"]
    shape_plus_f1 = cls_results["shape_plus_oflow_to_latent_family"]["macro_f1"]

    anti_leak_ok = text_f1 < 0.45
    label_free_ok = ari < 0.05 and nmi < 0.05
    hshape_recovers = (max(hshape_f1, hshape_flat_f1, shape_plus_f1) > text_f1 + 0.15
                       and max(hshape_f1, hshape_flat_f1, shape_plus_f1) > init_f1 + 0.10
                       and max(hshape_f1, hshape_flat_f1, shape_plus_f1) >= 0.50)
    oflow_adds = shape_plus_f1 >= max(hshape_f1, hshape_flat_f1) - 0.02
    dominance_pred = best_dom[1] > 0.10

    if anti_leak_ok and label_free_ok and hshape_recovers and dominance_pred:
        verdict = "PASS_PSA3B_TRAJECTORY_SHAPE_POSSIBILITY_SPACE"
    elif anti_leak_ok and label_free_ok and hshape_recovers:
        verdict = "PARTIAL_PSA3B_HSHAPE_RECOVERS_LATENT_FAMILY"
    elif anti_leak_ok and label_free_ok:
        verdict = "PARTIAL_PSA3B_LABEL_FREE_BUT_HSHAPE_WEAK"
    else:
        verdict = "FAIL_OR_CAVEAT_PSA3B"

    out = {
        "verdict": verdict,
        "pass_flags": {
            "anti_leak_ok": bool(anti_leak_ok),
            "label_free_ok": bool(label_free_ok),
            "hshape_recovers_latent_family": bool(hshape_recovers),
            "oflow_preserves_or_adds": bool(oflow_adds),
            "dominance_margin_predicted": bool(dominance_pred),
        },
        "summary_metrics": {
            "text_to_latent_family_macro_f1": text_f1,
            "text_to_diagnostic_mechanism_macro_f1": text_diag["macro_f1"],
            "latent_vs_diagnostic_mechanism_ARI": ari,
            "latent_vs_diagnostic_mechanism_NMI": nmi,
            "init_to_latent_family_macro_f1": init_f1,
            "shape_mean_to_latent_family_macro_f1": hshape_f1,
            "shape_flat_to_latent_family_macro_f1": hshape_flat_f1,
            "o_flow_to_latent_family_macro_f1": oflow_f1,
            "shape_plus_oflow_to_latent_family_macro_f1": shape_plus_f1,
            "best_latent_classifier": best_latent[0],
            "best_latent_macro_f1": float(best_latent[1]),
            "best_diagnostic_classifier": best_diag[0],
            "best_diagnostic_macro_f1": float(best_diag[1]),
            "best_dominance_regressor": best_dom[0],
            "best_dominance_r2": float(best_dom[1]),
        },
        "cluster_info": cluster_info,
        "text_leakage": {
            "latent_family": text_latent,
            "diagnostic_mechanism": text_diag,
        },
        "classification_results": cls_results,
        "dominance_regression_results": reg_results,
        "config": vars(args),
        "model_path": MODEL_PATHS[args.model_key],
        "terminology": "selection / commitment / latent trajectory family; no basin terminology",
        "interpretation": (
            "PSA-3B tests whether label-free latent trajectory families discovered in selection geometry "
            "are recoverable from trajectory-shape and flow features. A PASS supports structure-level "
            "possibility space beyond answer-level candidates."
        ),
    }

    with open(out_dir / "psa3b_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(out_dir / "psa3b_verdict.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False, indent=2))

    print("=" * 80)
    print("VERDICT")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()
