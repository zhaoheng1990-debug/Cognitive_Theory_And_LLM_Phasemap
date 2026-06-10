# -*- coding: utf-8 -*-
"""
PSA-3A.5: Label-Free Latent Trajectory Family Audit
===================================================

Why this version exists
-----------------------
PSA-3A.0 / 3A.3 showed that supervised mechanism labels can be linearly
recoverable from surface or graph-template structure too early. That makes the
probe measure label leakage rather than mechanism unfolding.

This version removes supervised mechanism labels from the main target.

Main idea
---------
1. Build neutral symbolic prompts.
2. Extract condition-clean delta trajectories.
3. Define latent trajectory families from selection-window internal geometry.
4. Audit when those latent families become decodable across layers.
5. Use text-only leakage and first-layer saturation as hard caveats.

This tests whether the model forms non-trivial structure families in its own
trajectory geometry, without forcing stable/competition/closure labels as the
primary target.

Run smoke:
    python psa3a5_label_free_latent_trajectory_family_audit.py --n_base 40 --topk 30

Formal:
    python psa3a5_label_free_latent_trajectory_family_audit.py --n_base 120 --topk 50
"""

import os
import json
import math
import random
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    r2_score,
    mean_squared_error,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
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
DEFAULT_OUT_DIR = r"C:\Users\ZH\Desktop\AGI\outputs\psa3a5_label_free_outputs"
SEED = 20260607

# These labels are only diagnostic. They are not used to train the main target.
MECHANISMS = ["chain", "fork", "triangle", "cycle", "fragmented"]
MECH_TO_ID = {m: i for i, m in enumerate(MECHANISMS)}

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

        for mech in MECHANISMS:
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
                "mechanism": mech,
                "mechanism_id": MECH_TO_ID[mech],
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


def center_window(arr, idx):
    return arr[:, idx, :].mean(axis=1)


def scalar_window(arr, idx):
    return arr[:, idx].mean(axis=1)


def entropy_bits(prob):
    prob = np.clip(prob, 1e-12, 1.0)
    return -(prob * np.log2(prob)).sum(axis=-1)


def safe_corr(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None
    return float(pearsonr(x, y)[0])


def make_selection_family_labels(delta, n_clusters=5, pca_dim=32, random_state=SEED):
    hd = delta["hidden_delta"]
    tc = delta["topk_center_delta"]
    te = delta["topk_entropy_delta"]
    ts = delta["topk_spread_delta"]

    n, L, d = hd.shape
    selection_w = clamp_window(20, 25, L)
    if not selection_w:
        selection_w = clamp_window(max(0, L - 6), L - 2, L)

    X_hidden = center_window(hd, selection_w)
    X_topk = center_window(tc, selection_w)
    X_scalar = np.stack([
        scalar_window(te, selection_w),
        scalar_window(ts, selection_w),
    ], axis=1)

    X = np.concatenate([X_hidden, X_topk, X_scalar], axis=1)
    Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    dim = min(pca_dim, Xs.shape[0] - 1, Xs.shape[1])
    if dim >= 2:
        Xp = PCA(n_components=dim, random_state=random_state).fit_transform(Xs)
    else:
        Xp = Xs

    k = min(n_clusters, max(2, Xp.shape[0] // 10))
    labels = KMeans(n_clusters=k, n_init=20, random_state=random_state).fit_predict(Xp)

    sil = None
    if len(np.unique(labels)) > 1 and Xp.shape[0] > len(np.unique(labels)):
        sil = float(silhouette_score(Xp, labels))

    return labels.astype(int), {
        "n_clusters": int(k),
        "pca_dim": int(dim),
        "silhouette": sil,
        "selection_window": selection_w,
    }


def eval_text_leakage(prompt_df, pair_df, target_labels):
    groups = pair_df["base_id"].values.astype(int)
    texts = prompt_df.set_index("prompt_row_id").loc[pair_df["cond_row_id"].values, "prompt"].tolist()
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))

    pipe = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2),
        LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced", C=1.0)
    )
    pred = cross_val_predict(pipe, texts, target_labels, cv=cv.split(texts, target_labels, groups))
    return {
        "text_tfidf_acc_group": float(accuracy_score(target_labels, pred)),
        "text_tfidf_macro_f1_group": float(f1_score(target_labels, pred, average="macro")),
    }


def layerwise_latent_probe(X_by_layer, target_labels, groups, prob_temperature=1.0):
    n, L, d = X_by_layer.shape
    C = len(np.unique(target_labels))
    probs = np.zeros((n, L, C), dtype=np.float32)
    metrics = []
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))

    for l in range(L):
        X = X_by_layer[:, l, :]
        clf = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced", C=0.25)
        )
        p = cross_val_predict(
            clf,
            X,
            target_labels,
            cv=cv.split(X, target_labels, groups),
            method="predict_proba",
        )
        if prob_temperature > 1.0:
            logits = np.log(np.clip(p, 1e-12, 1.0)) / prob_temperature
            logits = logits - logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            p = p / p.sum(axis=1, keepdims=True)

        probs[:, l, :] = p
        pred = p.argmax(axis=1)
        H = entropy_bits(p)
        metrics.append({
            "layer": l,
            "latent_acc": float(accuracy_score(target_labels, pred)),
            "latent_macro_f1": float(f1_score(target_labels, pred, average="macro")),
            "H_latent_mean": float(H.mean()),
            "H_latent_std": float(H.std()),
            "N_eff_latent_mean": float((2 ** H).mean()),
        })
        print(f"[latent-probe] L{l:02d} acc={metrics[-1]['latent_acc']:.3f} H={metrics[-1]['H_latent_mean']:.3f}")

    return probs, pd.DataFrame(metrics)


def slope(series, idx):
    if len(idx) < 2:
        return np.zeros(series.shape[0])
    x = np.asarray(idx, dtype=np.float64)
    x = x - x.mean()
    y = series[:, idx].astype(np.float64)
    y = y - y.mean(axis=1, keepdims=True)
    return y @ x / ((x ** 2).sum() + 1e-12)


def cos_distance(a, b):
    an = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return 1.0 - (an * bn).sum(axis=-1)


def build_sample_metrics(pair_df, probs, delta, latent_labels):
    n, L, C = probs.shape
    init_w = clamp_window(0, 6, L)
    mid_w = clamp_window(7, 19, L)
    selection_w = clamp_window(20, 25, L)
    if not selection_w:
        selection_w = clamp_window(max(0, L - 6), L - 2, L)

    H = entropy_bits(probs.reshape(-1, C)).reshape(n, L)
    H_init = scalar_window(H, init_w)
    H_mid = scalar_window(H, mid_w) if mid_w else H_init
    H_selection = scalar_window(H, selection_w)

    true_p = np.zeros((n, L), dtype=np.float32)
    for i in range(n):
        true_p[i, :] = probs[i, :, latent_labels[i]]
    Q_selection = scalar_window(true_p, selection_w)

    te = delta["topk_entropy_delta"]
    ts = delta["topk_spread_delta"]
    tc = delta["topk_center_delta"]
    hd = delta["hidden_delta"]

    init_c = center_window(tc, init_w)
    mid_c = center_window(tc, mid_w) if mid_w else init_c
    sel_c = center_window(tc, selection_w)

    hv = np.linalg.norm(np.diff(hd, axis=1), axis=-1)
    vel_mid_idx = [i for i in mid_w if i < hv.shape[1]]
    vel_sel_idx = [i for i in selection_w if i < hv.shape[1]]

    raw_selection = H_init - H_selection
    raw_unfolding = H_selection - H_init

    return pd.DataFrame({
        "pair_id": pair_df["pair_id"].values,
        "base_id": pair_df["base_id"].values,
        "surface_id": pair_df["surface_id"].values,
        "diagnostic_mechanism": pair_df["mechanism"].values,
        "diagnostic_mechanism_id": pair_df["mechanism_id"].values,
        "latent_family_id": latent_labels,
        "H_latent_init": H_init,
        "H_latent_mid": H_mid,
        "H_latent_selection": H_selection,
        "N_eff_init": 2 ** H_init,
        "N_eff_mid": 2 ** H_mid,
        "N_eff_selection": 2 ** H_selection,
        "raw_selection_init_selection": raw_selection,
        "raw_unfolding_init_selection": raw_unfolding,
        "Q_latent_selection": Q_selection,
        "Selection_eff_init_selection": raw_selection * Q_selection,
        "Unfolding_eff_init_selection": raw_unfolding * Q_selection,
        "latent_selection_correct": (scalar_window(probs, selection_w).argmax(axis=1) == latent_labels).astype(int),

        "topk_entropy_delta_init": scalar_window(te, init_w),
        "topk_entropy_delta_mid": scalar_window(te, mid_w) if mid_w else scalar_window(te, init_w),
        "topk_entropy_delta_selection": scalar_window(te, selection_w),
        "topk_entropy_delta_change_init_selection": scalar_window(te, selection_w) - scalar_window(te, init_w),
        "topk_entropy_delta_slope_mid": slope(te, mid_w),
        "topk_entropy_delta_slope_selection": slope(te, selection_w),

        "topk_spread_delta_init": scalar_window(ts, init_w),
        "topk_spread_delta_mid": scalar_window(ts, mid_w) if mid_w else scalar_window(ts, init_w),
        "topk_spread_delta_selection": scalar_window(ts, selection_w),
        "topk_spread_delta_change_init_selection": scalar_window(ts, selection_w) - scalar_window(ts, init_w),

        "center_delta_drift_init_selection": cos_distance(init_c, sel_c),
        "center_delta_drift_mid_selection": cos_distance(mid_c, sel_c),

        "hidden_delta_vel_mid": hv[:, vel_mid_idx].mean(axis=1) if vel_mid_idx else hv.mean(axis=1),
        "hidden_delta_vel_selection": hv[:, vel_sel_idx].mean(axis=1) if vel_sel_idx else hv.mean(axis=1),
    })


def reg_eval(X, y, groups):
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    pred = np.zeros_like(y, dtype=np.float64)
    for tr, te in cv.split(X, y, groups):
        model = make_pipeline(StandardScaler(with_mean=True, with_std=True), Ridge(alpha=1.0))
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
    spr = None
    if np.std(pred) > 1e-9 and np.std(y) > 1e-9:
        spr = float(spearmanr(pred, y).correlation)
    return {
        "corr": safe_corr(pred, y),
        "spearman": spr,
        "r2": float(r2_score(y, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y, pred))),
    }


def run_regressions(sm):
    groups = sm["base_id"].values
    topk_cols = [
        "topk_entropy_delta_init",
        "topk_entropy_delta_mid",
        "topk_entropy_delta_selection",
        "topk_entropy_delta_change_init_selection",
        "topk_entropy_delta_slope_mid",
        "topk_entropy_delta_slope_selection",
        "topk_spread_delta_init",
        "topk_spread_delta_mid",
        "topk_spread_delta_selection",
        "topk_spread_delta_change_init_selection",
        "center_delta_drift_init_selection",
        "center_delta_drift_mid_selection",
    ]
    cft_cols = topk_cols + ["hidden_delta_vel_mid", "hidden_delta_vel_selection"]

    out = {"feature_columns": {"topk_only": topk_cols, "cft_proxy": cft_cols}}
    for target in ["Selection_eff_init_selection", "Unfolding_eff_init_selection"]:
        y = sm[target].values.astype(float)
        out[target] = {
            "topk_only_group": reg_eval(sm[topk_cols].values, y, groups),
            "cft_proxy_group": reg_eval(sm[cft_cols].values, y, groups),
        }
    return out


def build_verdict(layer_metrics, sm, regressions, text_leak, cluster_info, pair_df, latent_labels, prob_temperature):
    H_init = float(sm["H_latent_init"].mean())
    H_mid = float(sm["H_latent_mid"].mean())
    H_sel = float(sm["H_latent_selection"].mean())
    N_init = float(sm["N_eff_init"].mean())
    N_mid = float(sm["N_eff_mid"].mean())
    N_sel = float(sm["N_eff_selection"].mean())

    first_acc = float(layer_metrics.iloc[0]["latent_acc"])
    first_H = float(layer_metrics.iloc[0]["H_latent_mean"])
    sel_acc = float(sm["latent_selection_correct"].mean())

    ari = float(adjusted_rand_score(pair_df["mechanism_id"].values, latent_labels))
    nmi = float(normalized_mutual_info_score(pair_df["mechanism_id"].values, latent_labels))

    text_f1 = text_leak["text_tfidf_macro_f1_group"]
    text_acc = text_leak["text_tfidf_acc_group"]

    text_leak_caveat = text_f1 > 0.70
    first_layer_caveat = (first_acc > 0.90 and first_H < 0.45 and prob_temperature <= 1.0)

    space_measurable = (N_init > 1.10 or N_mid > 1.10 or N_sel > 1.10)
    latent_emergence = (sel_acc - first_acc) > 0.15
    unfolding = H_mid > H_init and H_sel >= H_init
    selection_from_mid = H_mid > H_sel
    specialization = (H_mid > H_init) or (H_sel > H_init)

    cft_unfold = regressions["Unfolding_eff_init_selection"]["cft_proxy_group"]
    topk_unfold = regressions["Unfolding_eff_init_selection"]["topk_only_group"]
    cft_predicts = (
        cft_unfold["corr"] is not None and
        cft_unfold["corr"] > 0.30 and
        cft_unfold["r2"] > 0.10 and
        cft_unfold["r2"] > topk_unfold["r2"]
    )

    if text_leak_caveat:
        v = "FAIL_TEXT_LEAKAGE_FOR_LATENT_FAMILIES"
    elif first_layer_caveat:
        v = "FAIL_FIRST_LAYER_LATENT_SATURATION"
    elif space_measurable and latent_emergence and (unfolding or selection_from_mid or specialization) and cft_predicts:
        v = "PASS_PSA3A5_LABEL_FREE_LATENT_TRAJECTORY_FAMILIES"
    elif space_measurable and latent_emergence and (unfolding or selection_from_mid or specialization):
        v = "PARTIAL_LABEL_FREE_LATENT_FAMILIES_NO_CFT"
    elif space_measurable:
        v = "PARTIAL_LATENT_FAMILIES_MEASURABLE_BUT_NO_EMERGENCE"
    else:
        v = "FAIL_LATENT_FAMILIES_NOT_MEASURABLE"

    return {
        "verdict": v,
        "anti_leak_flags": {
            "text_leak_caveat": bool(text_leak_caveat),
            "first_layer_caveat": bool(first_layer_caveat),
        },
        "pass_flags": {
            "latent_space_measurable": bool(space_measurable),
            "latent_emergence_over_layers": bool(latent_emergence),
            "unfolding_pattern": bool(unfolding),
            "selection_from_mid_pattern": bool(selection_from_mid),
            "specialization_pattern": bool(specialization),
            "cft_predicts_unfolding_eff": bool(cft_predicts),
        },
        "summary_metrics": {
            "H_latent_init_mean": H_init,
            "H_latent_mid_mean": H_mid,
            "H_latent_selection_mean": H_sel,
            "N_eff_init_mean": N_init,
            "N_eff_mid_mean": N_mid,
            "N_eff_selection_mean": N_sel,
            "first_layer_acc": first_acc,
            "first_layer_H": first_H,
            "selection_probe_acc": sel_acc,
            "text_tfidf_acc_group": text_acc,
            "text_tfidf_macro_f1_group": text_f1,
            "latent_vs_diagnostic_mechanism_ARI": ari,
            "latent_vs_diagnostic_mechanism_NMI": nmi,
            "raw_selection_mean": float(sm["raw_selection_init_selection"].mean()),
            "raw_unfolding_mean": float(sm["raw_unfolding_init_selection"].mean()),
            "Selection_eff_mean": float(sm["Selection_eff_init_selection"].mean()),
            "Unfolding_eff_mean": float(sm["Unfolding_eff_init_selection"].mean()),
            "cft_unfolding_group_corr": cft_unfold["corr"],
            "cft_unfolding_group_r2": cft_unfold["r2"],
            "topk_unfolding_group_corr": topk_unfold["corr"],
            "topk_unfolding_group_r2": topk_unfold["r2"],
        },
        "cluster_info": cluster_info,
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
    parser.add_argument("--prob_temperature", type=float, default=1.0)
    parser.add_argument("--local_files_only", action="store_true", default=True)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PSA-3A.5: Label-Free Latent Trajectory Family Audit")
    print("=" * 80)
    print(f"model_key={args.model_key}")
    print(f"model_path={MODEL_PATHS[args.model_key]}")
    print(f"n_base={args.n_base}")
    print(f"out_dir={out_dir.resolve()}")

    prompt_df, pair_df = build_dataset(args.n_base)
    prompt_df.to_csv(out_dir / "psa3a5_prompts.csv", index=False, encoding="utf-8-sig")
    pair_df.to_csv(out_dir / "psa3a5_pairs.csv", index=False, encoding="utf-8-sig")

    tokenizer, model = load_model(args.model_key, args.device, args.dtype, args.local_files_only)
    feats = extract_hidden_topk(tokenizer, model, prompt_df, args.device, args.max_len, args.topk, batch_size=1)
    delta = build_delta_features(feats, pair_df)

    latent_labels, cluster_info = make_selection_family_labels(
        delta,
        n_clusters=args.n_clusters,
        pca_dim=args.pca_dim,
        random_state=SEED,
    )
    pair_df["latent_family_id"] = latent_labels
    pair_df.to_csv(out_dir / "psa3a5_pairs_with_latent_family.csv", index=False, encoding="utf-8-sig")

    text_leak = eval_text_leakage(prompt_df, pair_df, latent_labels)
    with open(out_dir / "psa3a5_text_leakage.json", "w", encoding="utf-8") as f:
        json.dump(text_leak, f, ensure_ascii=False, indent=2)
    print("[text-leak]", text_leak)

    groups = pair_df["base_id"].values.astype(int)
    probs, layer_metrics = layerwise_latent_probe(
        delta["hidden_delta"],
        latent_labels,
        groups,
        prob_temperature=args.prob_temperature,
    )
    layer_metrics.to_csv(out_dir / "psa3a5_layer_metrics.csv", index=False, encoding="utf-8-sig")

    sm = build_sample_metrics(pair_df, probs, delta, latent_labels)
    sm.to_csv(out_dir / "psa3a5_sample_metrics.csv", index=False, encoding="utf-8-sig")

    regs = run_regressions(sm)
    with open(out_dir / "psa3a5_cft_regression.json", "w", encoding="utf-8") as f:
        json.dump(regs, f, ensure_ascii=False, indent=2)

    vd = build_verdict(
        layer_metrics=layer_metrics,
        sm=sm,
        regressions=regs,
        text_leak=text_leak,
        cluster_info=cluster_info,
        pair_df=pair_df,
        latent_labels=latent_labels,
        prob_temperature=args.prob_temperature,
    )
    vd["config"] = vars(args)
    vd["model_path"] = MODEL_PATHS[args.model_key]
    vd["note"] = (
        "This is a label-free audit. Diagnostic mechanism labels are only used "
        "for ARI/NMI diagnostics, not as training targets. A PASS requires low "
        "text leakage, no first-layer saturation, emergence across layers, and "
        "CFT/topology proxy support."
    )

    with open(out_dir / "psa3a5_summary.json", "w", encoding="utf-8") as f:
        json.dump(vd, f, ensure_ascii=False, indent=2)
    with open(out_dir / "psa3a5_verdict.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(vd, ensure_ascii=False, indent=2))

    print("=" * 80)
    print("VERDICT")
    print(json.dumps(vd, ensure_ascii=False, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()
