# -*- coding: utf-8 -*-
"""
SEM-6D: Utility-Weighted Policy Update Audit

Why SEM-6D
----------
SEM-6A / 6A.1 / 6B / 6C showed that naive MemoryUnit updating fails.

Key finding from SEM-6C:
    ReflexiveUpdate != MemoryCenterUpdate

The failure is not that MemoryUnit is invalid. SEM-5C established:
    MemoryUnit = IdentityAddress + OperatorPrior + ResidualModes + Policy

The missing part is:
    Q(E_t): which experience should update future control?

SEM-6D tests a utility-weighted update:

    M_{t+1} = M_t + eta * Q(E_t) * DeltaM_t

where Q(E_t) is not similarity, but estimated future utility.

Because true future utility is unavailable at update time, SEM-6D runs two levels:

1) oracle_utility_update:
       Q(E_t) is measured by actual held-out utility of using E_t-derived memory.
       This is an upper-bound feasibility test.

2) proxy_utility_update:
       Q_hat(E_t) is learned from update-row internal features to predict oracle utility.
       This tests whether a usable Q proxy exists.

If oracle utility update works but proxy fails:
    Q exists but not yet learnable from current features.

If proxy utility update works:
    first operational reflexive update candidate.

Outputs:
    sem6d_outputs/
"""

import json
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================
# Config
# =========================

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

MODEL_PATH = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"
DATASET_CSV = r"sem5a2_outputs\sem5a2_dataset.csv"

OUT_DIR = Path("sem6d_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SHAPE_LAYERS = list(range(7, 20))
COMMIT_LAYERS = list(range(23, 26))

MAX_LENGTH = 512
MAX_ROWS = None

ALPHAS = [0.10, 0.20, 0.30]
UTILITY_ALPHA = 0.30  # use strongest alpha to estimate utility signal
ETA_VALUES = [0.25, 0.50, 1.00]

MIN_ROWS_PER_FAMILY = 4
N_INIT = 1
N_UPDATE_CANDIDATES = 2


# =========================
# Basic utilities
# =========================

def find_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def normalize(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v if n < eps else v / n


def cosine(a, b):
    return float(np.dot(normalize(a), normalize(b)))


def layer_hidden_dict(outputs):
    hs = outputs.hidden_states
    return {i: hs[i + 1] for i in range(len(hs) - 1)}


def mean_pool_layers(layer_vecs, layers):
    xs = [layer_vecs[l] for l in layers if l in layer_vecs]
    return np.mean(np.stack(xs, axis=0), axis=0).astype(np.float32)


class ShapeInjector:
    def __init__(self, model, vec, alpha):
        self.model = model
        self.vec = vec
        self.alpha = alpha
        self.handles = []

    def _hook(self, layer_idx):
        def fn(module, inputs, output):
            if isinstance(output, tuple):
                h = output[0]
                rest = output[1:]
            else:
                h = output
                rest = None
            h2 = h.clone()
            if self.vec is not None and self.alpha != 0:
                v = self.vec.to(h2.device, h2.dtype)
                h2[:, -1, :] = h2[:, -1, :] + self.alpha * v
            return (h2,) + rest if rest is not None else h2
        return fn

    def __enter__(self):
        for idx in SHAPE_LAYERS:
            self.handles.append(self.model.model.layers[idx].register_forward_hook(self._hook(idx)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def forward_collect(model, tokenizer, prompt):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(DEVICE)
    out = model(**enc, output_hidden_states=True, use_cache=False)
    hdict = layer_hidden_dict(out)
    return {l: h[:, -1, :].detach()[0].float().cpu().numpy().astype(np.float32) for l, h in hdict.items()}


def extract_shape_commit(hdict):
    return normalize(mean_pool_layers(hdict, SHAPE_LAYERS)), normalize(mean_pool_layers(hdict, COMMIT_LAYERS))


# =========================
# Memory helpers
# =========================

def build_records(df, prompt_col, family_col, concept_col, operator_col, row_id_col, model, tokenizer):
    records = []
    for pos, (idx, row) in enumerate(df.iterrows()):
        h = forward_collect(model, tokenizer, str(row[prompt_col]))
        shape, commit = extract_shape_commit(h)
        records.append({
            "row_idx": int(idx),
            "row_pos": int(pos),
            "row_id": row[row_id_col] if row_id_col else idx,
            "family": str(row[family_col]),
            "concept": str(row[concept_col]),
            "operator": str(row[operator_col]),
            "prompt": str(row[prompt_col]),
            "shape": shape,
            "commit": commit,
        })
        if (pos + 1) % 10 == 0 or pos + 1 == len(df):
            print(f"[Forward] {pos+1}/{len(df)}")
    return records


def make_splits(records):
    by = {}
    for r in records:
        by.setdefault(r["family"], []).append(r)
    splits = {}
    for fam, xs in by.items():
        xs = sorted(xs, key=lambda r: r["row_pos"])
        if len(xs) >= MIN_ROWS_PER_FAMILY:
            splits[fam] = {
                "init": xs[:N_INIT],
                "updates": xs[N_INIT:N_INIT + N_UPDATE_CANDIDATES],
                "test": xs[N_INIT + N_UPDATE_CANDIDATES:],
                "all": xs,
            }
    return splits


def build_bank(rows_by_family):
    bank = {}
    for fam, rows in rows_by_family.items():
        if not rows:
            continue
        bank[fam] = {
            "shape": normalize(np.mean(np.stack([r["shape"] for r in rows], axis=0), axis=0)),
            "commit": normalize(np.mean(np.stack([r["commit"] for r in rows], axis=0), axis=0)),
            "concept": rows[0]["concept"],
            "operator": rows[0]["operator"],
        }
    return bank


def rows_to_shape(rows):
    return normalize(np.mean(np.stack([r["shape"] for r in rows], axis=0), axis=0))


def update_delta(init_rows, update_row):
    init_shape = rows_to_shape(init_rows)
    return (update_row["shape"] - init_shape).astype(np.float32)


def utility_weighted_shape(init_rows, update_rows, weights, eta):
    init_shape = rows_to_shape(init_rows)
    if not update_rows:
        return init_shape
    delta = np.zeros_like(init_shape)
    for u, w in zip(update_rows, weights):
        delta += float(w) * update_delta(init_rows, u)
    if np.sum(np.abs(weights)) > 1e-8:
        delta = delta / (np.sum(np.abs(weights)) + 1e-8)
    return normalize(init_shape + eta * delta)


def family_scores(vec, bank, key="shape"):
    return {fam: cosine(vec, mem[key]) for fam, mem in bank.items()}


def rank_of_family(scores, fam):
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    for i, (f, _) in enumerate(ordered, 1):
        if f == fam:
            return i
    return len(ordered) + 1


def margin_to_family(scores, fam):
    oracle = scores.get(fam, np.nan)
    others = [v for f, v in scores.items() if f != fam]
    return float(oracle - max(others)) if others else np.nan


def eval_memory_vec(model, tokenizer, test_rows, fam, vec, oracle_bank, alpha):
    gains = []
    for tr in test_rows:
        raw_scores = family_scores(tr["shape"], oracle_bank, "shape")
        raw_sim = raw_scores[fam]
        raw_margin = margin_to_family(raw_scores, fam)
        raw_rank = rank_of_family(raw_scores, fam)

        with ShapeInjector(model, torch.tensor(vec, dtype=torch.float32), alpha):
            h2 = forward_collect(model, tokenizer, tr["prompt"])
        out_shape, _ = extract_shape_commit(h2)
        scores = family_scores(out_shape, oracle_bank, "shape")

        gains.append({
            "gain_shape_sim": scores[fam] - raw_sim,
            "gain_margin_shape": margin_to_family(scores, fam) - raw_margin,
            "rank_improvement_shape": raw_rank - rank_of_family(scores, fam),
        })
    if not gains:
        return {"gain_shape_sim": np.nan, "gain_margin_shape": np.nan, "rank_improvement_shape": np.nan}
    return {
        "gain_shape_sim": float(np.mean([g["gain_shape_sim"] for g in gains])),
        "gain_margin_shape": float(np.mean([g["gain_margin_shape"] for g in gains])),
        "rank_improvement_shape": float(np.mean([g["rank_improvement_shape"] for g in gains])),
    }


# =========================
# Utility features
# =========================

def update_features(init_rows, update_row, oracle_bank, fam):
    init_shape = rows_to_shape(init_rows)
    init_commit = normalize(np.mean(np.stack([r["commit"] for r in init_rows], axis=0), axis=0))
    u_shape = update_row["shape"]
    u_commit = update_row["commit"]

    # Basic intrinsic features
    shape_fit = cosine(u_shape, init_shape)
    commit_fit = cosine(u_commit, init_commit)
    delta_norm = float(np.linalg.norm(u_shape - init_shape))
    commit_delta_norm = float(np.linalg.norm(u_commit - init_commit))

    # family geometry features under oracle bank
    shape_scores = family_scores(u_shape, oracle_bank, "shape")
    commit_scores = family_scores(u_commit, oracle_bank, "commit")
    self_shape_rank = rank_of_family(shape_scores, fam)
    self_commit_rank = rank_of_family(commit_scores, fam)
    self_shape_margin = margin_to_family(shape_scores, fam)
    self_commit_margin = margin_to_family(commit_scores, fam)

    # novelty but identity-preserving proxy
    novelty_identity_product = delta_norm * max(0.0, commit_fit)

    return np.array([
        shape_fit,
        commit_fit,
        delta_norm,
        commit_delta_norm,
        self_shape_rank,
        self_commit_rank,
        self_shape_margin,
        self_commit_margin,
        novelty_identity_product,
    ], dtype=np.float32)


# =========================
# Main
# =========================

def main():
    df = pd.read_csv(DATASET_CSV)
    if MAX_ROWS is not None:
        df = df.head(MAX_ROWS).copy()

    prompt_col = find_col(df, ["prompt", "text", "input", "query"])
    family_col = find_col(df, ["seed_family", "family_id", "family"])
    concept_col = find_col(df, ["concept_star", "concept", "concept_phrase"])
    operator_col = find_col(df, ["operator_star", "operator_id", "operator", "operator_phrase"])
    row_id_col = find_col(df, ["row_id", "id", "idx"])

    print("=" * 90)
    print("SEM-6D Utility-Weighted Policy Update Audit")
    print("=" * 90)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    if DEVICE != "cuda":
        model.to(DEVICE)

    records = build_records(df, prompt_col, family_col, concept_col, operator_col, row_id_col, model, tokenizer)
    splits = make_splits(records)
    print(f"[Splits] valid families={len(splits)}")

    all_rows_by_family = {fam: sp["all"] for fam, sp in splits.items()}
    oracle_bank = build_bank(all_rows_by_family)

    # Stage 1: compute oracle utility for each update candidate
    utility_rows = []
    print("[Stage 1] Estimate oracle update utility per candidate...")
    for fam, sp in splits.items():
        init_rows = sp["init"]
        test_rows = sp["test"]
        if not test_rows:
            continue

        init_vec = rows_to_shape(init_rows)
        init_gain = eval_memory_vec(model, tokenizer, test_rows, fam, init_vec, oracle_bank, UTILITY_ALPHA)["gain_shape_sim"]

        for j, u in enumerate(sp["updates"]):
            cand_vec = utility_weighted_shape(init_rows, [u], [1.0], eta=1.0)
            cand_gain = eval_memory_vec(model, tokenizer, test_rows, fam, cand_vec, oracle_bank, UTILITY_ALPHA)["gain_shape_sim"]
            utility = cand_gain - init_gain
            feats = update_features(init_rows, u, oracle_bank, fam)

            utility_rows.append({
                "family": fam,
                "update_idx": j,
                "update_row_idx": u["row_idx"],
                "oracle_utility": float(utility),
                "init_gain": float(init_gain),
                "candidate_gain": float(cand_gain),
                **{f"feat_{i}": float(v) for i, v in enumerate(feats)}
            })

    util_df = pd.DataFrame(utility_rows)
    util_path = OUT_DIR / "sem6d_update_utilities.csv"
    util_df.to_csv(util_path, index=False, encoding="utf-8-sig")

    # Stage 2: learn proxy utility with GroupKFold by family
    feature_cols = [c for c in util_df.columns if c.startswith("feat_")]
    X = util_df[feature_cols].values.astype(np.float32)
    y = util_df["oracle_utility"].values.astype(np.float32)
    groups = util_df["family"].values

    preds = np.zeros_like(y, dtype=np.float32)
    n_splits = min(5, len(np.unique(groups)))
    if n_splits >= 2:
        gkf = GroupKFold(n_splits=n_splits)
        for train_idx, test_idx in gkf.split(X, y, groups):
            model_q = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model_q.fit(X[train_idx], y[train_idx])
            preds[test_idx] = model_q.predict(X[test_idx])
        proxy_r2 = float(r2_score(y, preds))
        proxy_corr = float(np.corrcoef(y, preds)[0, 1]) if np.std(preds) > 1e-8 and np.std(y) > 1e-8 else 0.0
    else:
        proxy_r2, proxy_corr = np.nan, np.nan

    util_df["proxy_utility"] = preds
    util_df.to_csv(util_path, index=False, encoding="utf-8-sig")

    # Stage 3: evaluate update policies
    print("[Stage 2] Evaluate utility-weighted policies...")
    result_rows = []

    util_by_family = {
        fam: sub.sort_values("update_idx").to_dict("records")
        for fam, sub in util_df.groupby("family")
    }

    for fam, sp in splits.items():
        init_rows = sp["init"]
        update_rows = sp["updates"]
        test_rows = sp["test"]
        if not test_rows or fam not in util_by_family:
            continue

        oracle_utils = [r["oracle_utility"] for r in util_by_family[fam]]
        proxy_utils = [r["proxy_utility"] for r in util_by_family[fam]]

        # normalize positive utility weights
        def pos_weights(vals):
            arr = np.array(vals, dtype=np.float32)
            arr = np.maximum(arr, 0.0)
            if np.sum(arr) < 1e-8:
                return np.zeros_like(arr)
            return arr / (np.sum(arr) + 1e-8)

        policies = {
            "init_only": ([], []),
            "naive_equal": (update_rows, np.ones(len(update_rows), dtype=np.float32) / max(1, len(update_rows))),
            "oracle_utility_weighted": (update_rows, pos_weights(oracle_utils)),
            "proxy_utility_weighted": (update_rows, pos_weights(proxy_utils)),
            "anti_oracle_weighted": (update_rows, pos_weights([-u for u in oracle_utils])),
            "oracle_all_shape": (None, None),
            "shuffle_shape": (None, None),
        }

        non = [x for x in oracle_bank if x != fam]
        shuffle_fam = random.choice(non)

        for policy, (ups, weights) in policies.items():
            for eta in ETA_VALUES:
                if policy == "init_only":
                    vec = rows_to_shape(init_rows)
                elif policy == "oracle_all_shape":
                    vec = oracle_bank[fam]["shape"]
                elif policy == "shuffle_shape":
                    vec = oracle_bank[shuffle_fam]["shape"]
                else:
                    vec = utility_weighted_shape(init_rows, ups, weights, eta=eta)

                for alpha in ALPHAS:
                    gains = eval_memory_vec(model, tokenizer, test_rows, fam, vec, oracle_bank, alpha)
                    result_rows.append({
                        "family": fam,
                        "policy": policy,
                        "eta": eta,
                        "alpha": alpha,
                        "n_updates": 0 if ups is None else len(ups),
                        "oracle_utility_sum_pos": float(np.sum(np.maximum(oracle_utils, 0.0))),
                        "proxy_utility_sum_pos": float(np.sum(np.maximum(proxy_utils, 0.0))),
                        **gains,
                    })

    res_df = pd.DataFrame(result_rows)
    res_path = OUT_DIR / "sem6d_policy_results.csv"
    res_df.to_csv(res_path, index=False, encoding="utf-8-sig")

    summary = (
        res_df.groupby(["policy", "eta", "alpha"], dropna=False)
        .agg(
            n=("family", "count"),
            gain_shape_sim=("gain_shape_sim", "mean"),
            gain_margin_shape=("gain_margin_shape", "mean"),
            rank_improvement_shape=("rank_improvement_shape", "mean"),
        )
        .reset_index()
    )
    summary_path = OUT_DIR / "sem6d_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    checks = []
    for eta in ETA_VALUES:
        for alpha in ALPHAS:
            sub = summary[(summary["eta"] == eta) & (summary["alpha"] == alpha)]

            def val(policy, col):
                v = sub[sub["policy"] == policy][col]
                return float(v.iloc[0]) if len(v) else np.nan

            init = val("init_only", "gain_shape_sim")
            naive = val("naive_equal", "gain_shape_sim")
            oracle_u = val("oracle_utility_weighted", "gain_shape_sim")
            proxy_u = val("proxy_utility_weighted", "gain_shape_sim")
            anti = val("anti_oracle_weighted", "gain_shape_sim")
            shuffle = val("shuffle_shape", "gain_shape_sim")
            oracle_shape = val("oracle_all_shape", "gain_shape_sim")

            checks.append({
                "check": f"oracle_utility_eta_{eta}_alpha_{alpha}",
                "eta": eta,
                "alpha": alpha,
                "value": oracle_u,
                "init": init,
                "naive": naive,
                "anti": anti,
                "shuffle": shuffle,
                "oracle_shape": oracle_shape,
                "pass_lite": bool(oracle_u > init),
                "pass_strong": bool((oracle_u > init) and (oracle_u >= naive) and (oracle_u > anti) and (oracle_u > shuffle)),
                "note": "Oracle utility weighting should beat init/naive/anti/shuffle."
            })

            checks.append({
                "check": f"proxy_utility_eta_{eta}_alpha_{alpha}",
                "eta": eta,
                "alpha": alpha,
                "value": proxy_u,
                "init": init,
                "naive": naive,
                "anti": anti,
                "shuffle": shuffle,
                "oracle_shape": oracle_shape,
                "pass_lite": bool(proxy_u > init),
                "pass_strong": bool((proxy_u > init) and (proxy_u >= naive) and (proxy_u > anti) and (proxy_u > shuffle)),
                "note": "Proxy utility weighting should beat init/naive/anti/shuffle."
            })

    checks_df = pd.DataFrame(checks)
    checks_path = OUT_DIR / "sem6d_pass_checks.csv"
    checks_df.to_csv(checks_path, index=False, encoding="utf-8-sig")

    oracle_strong = int(checks_df[checks_df["check"].str.startswith("oracle_utility")]["pass_strong"].sum())
    proxy_strong = int(checks_df[checks_df["check"].str.startswith("proxy_utility")]["pass_strong"].sum())

    verdict = "FAIL"
    if oracle_strong >= 3:
        verdict = "PASS-OracleUtility"
    if proxy_strong >= 3:
        verdict = "PASS-Strong"

    best = summary.sort_values("gain_shape_sim", ascending=False).head(10).to_dict("records")

    report = {
        "experiment": "SEM-6D Utility-Weighted Policy Update Audit",
        "verdict": verdict,
        "proxy_utility": {
            "groupkfold_r2": proxy_r2,
            "groupkfold_corr": proxy_corr,
            "n_update_candidates": int(len(util_df)),
        },
        "best_conditions": best,
        "interpretation_rules": [
            "If oracle utility weighting works but proxy fails, Q exists but current proxy features are insufficient.",
            "If proxy utility weighting works, this is an operational reflexive update candidate.",
            "If neither works, even future-utility weighting is not sufficient under this setup."
        ],
        "outputs": {
            "update_utilities": str(util_path),
            "policy_results": str(res_path),
            "summary": str(summary_path),
            "checks": str(checks_path),
        }
    }
    report_path = OUT_DIR / "sem6d_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("SEM-6D finished")
    print("=" * 90)
    print(f"Proxy utility R2={proxy_r2:.4f}, corr={proxy_corr:.4f}")
    print(summary)
    print("\nChecks:")
    print(checks_df)
    print(f"\nVerdict: {verdict}")
    print(f"Saved: {util_path}")
    print(f"Saved: {res_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {checks_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
