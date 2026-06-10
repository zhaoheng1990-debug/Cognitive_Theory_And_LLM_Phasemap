# -*- coding: utf-8 -*-
"""
EDA-1C / P4-PSG-ProxyNull
PSG Proxy-Leakage Audit for Paper4 non-R Vocabulary-Neighborhood Flow

Goal
----
Paper4 currently uses non-R vocabulary-neighborhood flow features:

    1 - cos(C_{l+1}, C_l)
    1 - J(S_{l+1}, S_l)
    1 - WJ(S_{l+1}, S_l)
    Δspread
    Δentropy

These features avoid R-coordinate leakage, but they are still TopK/VIM-derived.
After PSG correction, we must test whether they are merely projection-selection
readback proxies of hidden-state motion.

This script compares five feature families:

    real_W_vim
    rowperm_W_vim
    gaussian_R_vim
    randomK_vim
    direct_hidden_delta

Targets:
    model-specific mechanism_label
    model-specific deltaU proxy from decision-window hidden trajectory PC1

Important
---------
This is a minimal runnable proxy-null audit. It does not require existing Paper4
raw files. It constructs a controlled prompt family, extracts hidden states, builds
model-specific downstream deltaU_proxy from decision-window hidden deltas, and tests
whether pre-decision VIM flow predicts it under different PSG controls.

If gaussian_R_vim / rowperm_W_vim ≈ real_W_vim, then Paper4 non-R VIM flow should be
downgraded to PSG readback proxy.

If real_W_vim > controls, then there is residual support for learned-W / real-vocabulary
specific information in the non-R flow channel.

Run
---
python eda1c_p4_psg_proxy_null.py

Output
------
C:\\Users\\ZH\\Desktop\\AGI\\outputs\\eda1c_p4_psg_proxy_null_outputs
"""

import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from transformers import AutoTokenizer, AutoModelForCausalLM


EXPERIMENT_ID = "EDA-1C_P4_PSG_ProxyNull_v1_0"

OUT_DIR = Path(r"C:\Users\ZH\Desktop\AGI\outputs\eda1c_p4_psg_proxy_null_outputs")

MODEL_CONFIGS = {
    "qwen": {
        "path": r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main",
        "layers": list(range(0, 28)),
        "decision_window": [19, 20, 21, 22, 23, 24, 25],
        "predecision_transitions": [(13, 14), (14, 15), (15, 16), (16, 17), (17, 18)],
    },
    "llama": {
        "path": r"D:\model\Llama-3.2-1B-Instruct",
        "layers": list(range(0, 16)),
        "decision_window": [11, 12, 13, 14],
        "predecision_transitions": [(3, 4), (4, 5), (5, 6), (6, 7), (7, 8)],
    },
    "gemma": {
        "path": r"D:\model\gemma-2-2b-it",
        "layers": list(range(0, 26)),
        "decision_window": [18, 19, 20, 21, 22, 23],
        "predecision_transitions": [(7, 8), (8, 9), (9, 10), (10, 11), (11, 12)],
    },
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SEED = 42
MAX_LENGTH = 192
K_SELECT = 1000
K_EVAL = 400

FEATURE_FAMILIES = [
    "real_W_vim",
    "rowperm_W_vim",
    "gaussian_R_vim",
    "randomK_vim",
    "direct_hidden_delta",
]

MECHANISM_LABELS = ["stable", "weak_shift", "competition", "conflict", "closure", "override"]

BASE_FACTS = [
    ("Alice", "red", "Paris", "violin"),
    ("Bob", "blue", "London", "piano"),
    ("Clara", "green", "Tokyo", "guitar"),
    ("David", "yellow", "Rome", "flute"),
    ("Eva", "purple", "Berlin", "drum"),
    ("Frank", "orange", "Madrid", "cello"),
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


def safe_entropy(vals, eps=1e-12):
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals - np.nanmin(vals)
    vals = vals + eps
    p = vals / np.sum(vals)
    return float(-(p * np.log(p + eps)).sum())


def jaccard(a, b):
    sa = set(map(int, a))
    sb = set(map(int, b))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def weighted_jaccard(ids_a, scores_a, ids_b, scores_b):
    # Positive weights from local score ranks. Robust to arbitrary score scale.
    wa = {}
    wb = {}
    ra = np.argsort(-np.asarray(scores_a))
    rb = np.argsort(-np.asarray(scores_b))
    for rank, pos in enumerate(ra):
        wa[int(ids_a[pos])] = 1.0 / (1.0 + rank)
    for rank, pos in enumerate(rb):
        wb[int(ids_b[pos])] = 1.0 / (1.0 + rank)

    keys = set(wa) | set(wb)
    if not keys:
        return 1.0
    num = sum(min(wa.get(k, 0.0), wb.get(k, 0.0)) for k in keys)
    den = sum(max(wa.get(k, 0.0), wb.get(k, 0.0)) for k in keys)
    return float(num / max(den, 1e-12))


def build_controlled_prompts(n_graphs=36):
    rows = []
    rng = random.Random(SEED)

    for gid in range(n_graphs):
        a, color, city, instrument = BASE_FACTS[gid % len(BASE_FACTS)]
        b, color2, city2, instrument2 = BASE_FACTS[(gid + 2) % len(BASE_FACTS)]

        base = (
            f"In a small record, {a} is associated with {color}, {city}, and {instrument}. "
            f"{b} is associated with {color2}, {city2}, and {instrument2}. "
        )

        variants = [
            ("stable", base + f"Question: Which city is associated with {a}? Answer with the city only."),
            ("stable", base + f"Question: Which instrument is associated with {a}? Answer with the instrument only."),
            ("weak_shift", base + f"A later note says {a} also recently visited {city2}, but the original association remains important. Question: Which city was originally associated with {a}?"),
            ("weak_shift", base + f"A distractor says {b} borrowed a {instrument}. Question: Which instrument is associated with {a}?"),
            ("competition", base + f"A competing note says {a} may be associated with either {city} or {city2}. Question: Which city should be selected for {a}?"),
            ("competition", base + f"A competing note says {a} may play either {instrument} or {instrument2}. Question: Which instrument should be selected for {a}?"),
            ("conflict", base + f"Contradictory update: {a} is not associated with {city}; {a} is associated with {city2}. Question: Which city is now associated with {a}?"),
            ("conflict", base + f"Contradictory update: {a} no longer plays {instrument}; {a} plays {instrument2}. Question: Which instrument is now associated with {a}?"),
            ("closure", base + f"If a person is associated with {color}, then the city relation closes to {city}. {a} is associated with {color}. Question: Which city follows for {a}?"),
            ("closure", base + f"If a person is associated with {color}, then the instrument relation closes to {instrument}. {a} is associated with {color}. Question: Which instrument follows for {a}?"),
            ("override", base + f"Override rule: recent city updates override original city records. Recent update: {a} -> {city2}. Question: Which city should be used for {a}?"),
            ("override", base + f"Override rule: recent instrument updates override original records. Recent update: {a} -> {instrument2}. Question: Which instrument should be used for {a}?"),
        ]

        rng.shuffle(variants)
        for vid, (label, prompt) in enumerate(variants):
            rows.append({
                "graph_id": gid,
                "variant_id": vid,
                "mechanism_label": label,
                "prompt": prompt,
            })

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
def extract_all_hidden(model, tokenizer, prompts, layers):
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
    use_layers = [l for l in layers if 0 <= l <= n_layers_total]

    H = {}
    for l in use_layers:
        H[l] = out.hidden_states[l][batch_idx, last_idx, :].detach().float().cpu().numpy().astype(np.float32)

    print(f"[INFO] total_layers={n_layers_total}, extracted={len(use_layers)}")
    return H, use_layers


def topk_ids_scores(H, W, k_select=K_SELECT, k_eval=K_EVAL):
    scores = H @ W.T
    idx = np.argpartition(-scores, kth=k_select-1, axis=1)[:, :k_select]
    sc = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-sc, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)[:, :k_eval]
    sc = np.take_along_axis(sc, order, axis=1)[:, :k_eval]
    return idx, sc


def random_ids_scores(n, V, rng, k_eval=K_EVAL):
    idx = np.stack([rng.choice(V, size=k_eval, replace=False) for _ in range(n)], axis=0)
    sc = rng.normal(size=(n, k_eval)).astype(np.float32)
    return idx, sc


def center_spread_entropy(ids, scores, Wn):
    X = Wn[ids]  # [N,K,d]
    C = X.mean(axis=1)
    Cn = l2_normalize(C)
    sims = np.einsum("nkd,nd->nk", X, Cn)
    spread = np.mean(1.0 - sims, axis=1)
    ent = np.array([safe_entropy(s) for s in scores], dtype=np.float32)
    return Cn.astype(np.float32), spread.astype(np.float32), ent.astype(np.float32)


def build_vim_transition_features(H_a, H_b, W_variant, Wn_eval, variant_name, rng):
    n = H_a.shape[0]
    V = W_variant.shape[0]

    if variant_name == "randomK_vim":
        ids_a, sc_a = random_ids_scores(n, V, rng)
        ids_b, sc_b = random_ids_scores(n, V, rng)
    else:
        ids_a, sc_a = topk_ids_scores(H_a, W_variant)
        ids_b, sc_b = topk_ids_scores(H_b, W_variant)

    # If scoring W is row-permuted or gaussian, ids refer to local rows.
    # For rowperm, caller passes W_variant with original row order already permuted and Wn_eval aligned.
    C_a, sp_a, en_a = center_spread_entropy(ids_a, sc_a, Wn_eval)
    C_b, sp_b, en_b = center_spread_entropy(ids_b, sc_b, Wn_eval)

    center_delta = 1.0 - np.sum(C_a * C_b, axis=1)
    j = np.array([jaccard(ids_a[i], ids_b[i]) for i in range(n)], dtype=np.float32)
    wj = np.array([weighted_jaccard(ids_a[i], sc_a[i], ids_b[i], sc_b[i]) for i in range(n)], dtype=np.float32)

    feats = np.stack([
        center_delta.astype(np.float32),
        1.0 - j,
        1.0 - wj,
        (sp_b - sp_a).astype(np.float32),
        (en_b - en_a).astype(np.float32),
        sp_a.astype(np.float32),
        sp_b.astype(np.float32),
        en_a.astype(np.float32),
        en_b.astype(np.float32),
    ], axis=1)

    return feats


def build_direct_hidden_delta_features(H_a, H_b):
    Ha = l2_normalize(H_a)
    Hb = l2_normalize(H_b)
    cos_delta = 1.0 - np.sum(Ha * Hb, axis=1)
    norm_a = np.linalg.norm(H_a, axis=1)
    norm_b = np.linalg.norm(H_b, axis=1)
    delta_norm = norm_b - norm_a
    step_norm = np.linalg.norm(H_b - H_a, axis=1)
    return np.stack([cos_delta, delta_norm, step_norm, norm_a, norm_b], axis=1).astype(np.float32)


def build_deltaU_proxy(H, decision_window):
    # Build downstream hidden-delta matrix from decision window transitions.
    layers = [l for l in decision_window if l in H]
    layers = sorted(layers)
    pairs = [(layers[i], layers[i+1]) for i in range(len(layers)-1)]
    if not pairs:
        raise RuntimeError("Not enough decision-window layers extracted.")

    blocks = []
    for a, b in pairs:
        d = H[b] - H[a]
        # compress very high-dimensional deltas by samplewise summary + random projection
        rng = np.random.default_rng(SEED + a * 100 + b)
        proj_dim = min(64, d.shape[1])
        R = rng.normal(size=(d.shape[1], proj_dim)).astype(np.float32)
        R = R / np.sqrt(d.shape[1])
        blocks.append(d @ R)
        blocks.append(build_direct_hidden_delta_features(H[a], H[b]))
    X = np.concatenate(blocks, axis=1)
    X = StandardScaler().fit_transform(X)
    pc = PCA(n_components=3, random_state=SEED).fit(X)
    deltaU = pc.transform(X)[:, 0]
    evr = pc.explained_variance_ratio_
    # sign convention: competition/conflict/override often should differ from stable; keep deterministic by skew
    if np.mean(deltaU[:len(deltaU)//2]) < np.mean(deltaU[len(deltaU)//2:]):
        deltaU = -deltaU
    return deltaU.astype(np.float32), evr.astype(float)


def evaluate_feature_family(X, y_reg, y_cls, groups):
    rows = []
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))

    reg_preds = np.zeros_like(y_reg, dtype=np.float32)
    cls_preds = np.empty_like(y_cls, dtype=object)

    for train_idx, test_idx in gkf.split(X, y_cls, groups):
        reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        clf = make_pipeline(
            StandardScaler(),
            OneVsRestClassifier(LogisticRegression(max_iter=3000, solver="liblinear"))
        )
        reg.fit(X[train_idx], y_reg[train_idx])
        clf.fit(X[train_idx], y_cls[train_idx])
        reg_preds[test_idx] = reg.predict(X[test_idx]).astype(np.float32)
        cls_preds[test_idx] = clf.predict(X[test_idx])

    r2 = float(r2_score(y_reg, reg_preds))
    corr = float(np.corrcoef(y_reg, reg_preds)[0, 1]) if np.std(reg_preds) > 1e-9 else np.nan
    acc = float(accuracy_score(y_cls, cls_preds))
    macro_f1 = float(f1_score(y_cls, cls_preds, average="macro", zero_division=0))
    return {
        "cv_r2": r2,
        "cv_corr": corr,
        "cv_acc": acc,
        "cv_macro_f1": macro_f1,
    }


def main():
    warnings.filterwarnings("ignore")
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    prompt_df = build_controlled_prompts(n_graphs=36)
    prompt_df.to_csv(OUT_DIR / "eda1c_controlled_prompts.csv", index=False, encoding="utf-8-sig")

    all_metric_rows = []
    all_feature_rows = []

    for model_key, cfg in MODEL_CONFIGS.items():
        model_path = cfg["path"]
        if not Path(model_path).exists():
            print(f"[WARN] skip {model_key}, path not found: {model_path}")
            continue

        model, tokenizer = load_model(model_key, model_path)
        W = get_output_W(model)
        Wn = l2_normalize(W)
        V, d = W.shape

        H, layers = extract_all_hidden(model, tokenizer, prompt_df["prompt"].tolist(), cfg["layers"])

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        deltaU, evr = build_deltaU_proxy(H, cfg["decision_window"])
        y_cls = prompt_df["mechanism_label"].values
        groups = prompt_df["graph_id"].values

        # Prepare matrix variants.
        rng_global = np.random.default_rng(SEED + hash(model_key) % 9999)
        perm = rng_global.permutation(V)
        W_perm = W[perm]
        Wn_perm = l2_normalize(W_perm)

        R = rng_global.normal(size=W.shape).astype(np.float32)
        R = l2_normalize(R)
        Rn = R.copy()

        transition_list = []
        for a, b in cfg["predecision_transitions"]:
            if a in H and b in H:
                transition_list.append((a, b))
        if not transition_list:
            print(f"[WARN] no valid predecision transitions for {model_key}")
            continue

        family_feature_blocks = {name: [] for name in FEATURE_FAMILIES}

        for a, b in transition_list:
            print(f"[INFO] {model_key} transition {a}->{b}")

            rng = np.random.default_rng(SEED + a * 101 + b * 307 + hash(model_key) % 10007)

            family_feature_blocks["real_W_vim"].append(
                build_vim_transition_features(H[a], H[b], W, Wn, "real_W_vim", rng)
            )
            family_feature_blocks["rowperm_W_vim"].append(
                build_vim_transition_features(H[a], H[b], W_perm, Wn_perm, "rowperm_W_vim", rng)
            )
            family_feature_blocks["gaussian_R_vim"].append(
                build_vim_transition_features(H[a], H[b], R, Rn, "gaussian_R_vim", rng)
            )
            family_feature_blocks["randomK_vim"].append(
                build_vim_transition_features(H[a], H[b], W, Wn, "randomK_vim", rng)
            )
            family_feature_blocks["direct_hidden_delta"].append(
                build_direct_hidden_delta_features(H[a], H[b])
            )

        for family, blocks in family_feature_blocks.items():
            X = np.concatenate(blocks, axis=1)
            metrics = evaluate_feature_family(X, deltaU, y_cls, groups)
            row = {
                "experiment_id": EXPERIMENT_ID,
                "model": model_key,
                "feature_family": family,
                "n_prompts": int(len(prompt_df)),
                "n_groups": int(len(np.unique(groups))),
                "n_features": int(X.shape[1]),
                "deltaU_pc1_var": float(evr[0]) if len(evr) > 0 else np.nan,
                "deltaU_pc2_var": float(evr[1]) if len(evr) > 1 else np.nan,
                "deltaU_pc3_var": float(evr[2]) if len(evr) > 2 else np.nan,
            }
            row.update(metrics)
            all_metric_rows.append(row)

            # Save compact feature rows for future bootstrap / analysis.
            feat_df = pd.DataFrame(X, columns=[f"f{i:03d}" for i in range(X.shape[1])])
            feat_df.insert(0, "model", model_key)
            feat_df.insert(1, "feature_family", family)
            feat_df.insert(2, "graph_id", groups)
            feat_df.insert(3, "mechanism_label", y_cls)
            feat_df.insert(4, "deltaU_proxy", deltaU)
            feat_path = OUT_DIR / f"eda1c_features_{model_key}_{family}.csv"
            feat_df.to_csv(feat_path, index=False, encoding="utf-8-sig")

            print(f"[RESULT] {model_key} {family}: {metrics}")

        del W, Wn, H
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics_df = pd.DataFrame(all_metric_rows)
    metrics_path = OUT_DIR / "eda1c_p4_psg_proxy_null_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    residual_rows = []
    for model in sorted(metrics_df["model"].unique()) if not metrics_df.empty else []:
        sub = metrics_df[metrics_df["model"] == model].copy()
        def get(f, metric):
            ss = sub[sub["feature_family"] == f]
            return float(ss.iloc[0][metric]) if not ss.empty else np.nan

        for metric in ["cv_r2", "cv_corr", "cv_macro_f1"]:
            real = get("real_W_vim", metric)
            rowp = get("rowperm_W_vim", metric)
            gaus = get("gaussian_R_vim", metric)
            rand = get("randomK_vim", metric)
            hid = get("direct_hidden_delta", metric)
            residual_rows.append({
                "model": model,
                "metric": metric,
                "real_W_vim": real,
                "rowperm_W_vim": rowp,
                "gaussian_R_vim": gaus,
                "randomK_vim": rand,
                "direct_hidden_delta": hid,
                "real_minus_rowperm": real - rowp,
                "real_minus_gaussian": real - gaus,
                "real_minus_randomK": real - rand,
                "real_minus_direct_hidden": real - hid,
            })

    residual_df = pd.DataFrame(residual_rows)
    residual_path = OUT_DIR / "eda1c_p4_psg_proxy_null_residual_table.csv"
    residual_df.to_csv(residual_path, index=False, encoding="utf-8-sig")

    verdict = build_verdict(metrics_df, residual_df)
    verdict_path = OUT_DIR / "eda1c_p4_psg_proxy_null_verdict.txt"
    verdict_path.write_text(verdict, encoding="utf-8")

    print("\n" + verdict)
    print(f"\n[OK] metrics: {metrics_path}")
    print(f"[OK] residual: {residual_path}")
    print(f"[OK] verdict: {verdict_path}")


def build_verdict(metrics_df, residual_df):
    lines = []
    lines.append("=" * 88)
    lines.append("EDA-1C / P4-PSG-ProxyNull Verdict")
    lines.append("=" * 88)

    if metrics_df.empty:
        lines.append("NO_VALID_ROWS")
        return "\n".join(lines)

    lines.append("\nMetric table:")
    for _, r in metrics_df.sort_values(["model", "feature_family"]).iterrows():
        lines.append(
            f"model={r['model']} family={r['feature_family']} "
            f"R2={r['cv_r2']:.4f} corr={r['cv_corr']:.4f} "
            f"F1={r['cv_macro_f1']:.4f} PC1={r['deltaU_pc1_var']:.4f}"
        )

    lines.append("\nResidual contrasts:")
    for _, r in residual_df.iterrows():
        lines.append(
            f"model={r['model']} metric={r['metric']} "
            f"real-rowperm={r['real_minus_rowperm']:.4f} "
            f"real-gaussian={r['real_minus_gaussian']:.4f} "
            f"real-randomK={r['real_minus_randomK']:.4f} "
            f"real-hidden={r['real_minus_direct_hidden']:.4f}"
        )

    # Interpret primarily on cv_corr and cv_r2.
    cont = residual_df[residual_df["metric"].isin(["cv_r2", "cv_corr"])].copy()
    mean_real_gauss = float(cont["real_minus_gaussian"].mean())
    mean_real_rowp = float(cont["real_minus_rowperm"].mean())
    mean_real_rand = float(cont["real_minus_randomK"].mean())
    mean_real_hidden = float(cont["real_minus_direct_hidden"].mean())

    lines.append("\nMean continuous-target residuals:")
    lines.append(f"mean_real_minus_rowperm={mean_real_rowp:.4f}")
    lines.append(f"mean_real_minus_gaussian={mean_real_gauss:.4f}")
    lines.append(f"mean_real_minus_randomK={mean_real_rand:.4f}")
    lines.append(f"mean_real_minus_direct_hidden={mean_real_hidden:.4f}")

    lines.append("\nInterpretation:")
    if mean_real_gauss > 0.05 and mean_real_rowp > 0.05 and mean_real_rand > 0.05:
        lines.append("PASS-REALW-RESIDUAL: real_W non-R VIM flow exceeds PSG controls.")
        lines.append("Paper4 may retain a real-vocabulary residual claim, with cautious wording.")
    elif mean_real_rand > 0.05 and (mean_real_gauss <= 0.05 or mean_real_rowp <= 0.05):
        lines.append("MIXED-READBACK: real_W exceeds randomK but not PSG controls.")
        lines.append("Paper4 non-R VIM flow should be described as a strong readback predictor, not learned-W semantic residual.")
    else:
        lines.append("NO-REALW-RESIDUAL: real_W does not clearly exceed controls.")
        lines.append("Paper4 should move non-R VIM flow to PSG-corrected readback evidence only.")

    if mean_real_hidden < 0:
        lines.append("DIRECT-HIDDEN-BASELINE-WARNING: direct hidden delta outperforms real_W VIM on average; VIM may be a lossy proxy.")
    else:
        lines.append("VIM-VS-HIDDEN: real_W VIM is competitive with or stronger than direct hidden delta under this proxy audit.")

    lines.append("\nPaper4 action:")
    lines.append("Add a PSG Proxy-Leakage subsection before claiming non-R vocabulary-neighborhood flow as anti-artifact evidence.")
    lines.append("If this audit returns MIXED/NO, rewrite non-R VIM flow as PSG-corrected readback predictor.")
    lines.append("=" * 88)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
