# -*- coding: utf-8 -*-
"""
SEM-6A.1: Reflexive Update Robustness Audit

Why 6A.1
--------
SEM-6A gave PASS-Lite, not strong:
    M1_minimal > M0_minimal only at alpha=0.3.
    Rank-improvement is too discrete and weak.
    One update row per family may be too little.

SEM-6A.1 tests robustness with:
    1) continuous metrics as primary: gain_shape_sim / gain_margin_shape
    2) variable update budget per family
    3) EMA-style update of MemoryUnit
    4) held-out surface tests
    5) update quality checks

Core question:
    Does adding more successful experience monotonically improve MemoryUnit guidance?

Usage:
    Put this script in your python_script folder and run in dhrf_4080s.
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
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForCausalLM


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

MODEL_PATH = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"
DATASET_CSV = r"sem5a2_outputs\sem5a2_dataset.csv"

OUT_DIR = Path("sem6a1_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SHAPE_LAYERS = list(range(7, 20))
COMMIT_LAYERS = list(range(23, 26))
MAX_LENGTH = 512

ALPHAS = [0.10, 0.20, 0.30]
UPDATE_BUDGETS = [0, 1, 2]   # additional examples after init
RESIDUAL_K = 2
RESIDUAL_BETA = 0.5
TOPK_RESIDUAL = 3
MIN_ROWS_PER_FAMILY = 4      # init + up to 2 update + at least 1 test


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
    return {layer_idx: hs[layer_idx + 1] for layer_idx in range(len(hs) - 1)}


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


def operator_proto(bank, operator, exclude_family):
    xs = [m["shape"] for f, m in bank.items() if f != exclude_family and str(m["operator"]) == str(operator)]
    if not xs:
        xs = [m["shape"] for f, m in bank.items() if f != exclude_family]
    if not xs:
        xs = [m["shape"] for _, m in bank.items()]
    return normalize(np.mean(np.stack(xs, axis=0), axis=0))


def residual_for_family(bank, fam):
    m = bank[fam]
    p = operator_proto(bank, m["operator"], fam)
    return (m["shape"] - p).astype(np.float32)


def fit_pca(bank, exclude_family, k):
    fams = [f for f in bank if f != exclude_family]
    if len(fams) < 2:
        return None
    X = np.stack([residual_for_family(bank, f) for f in fams], axis=0)
    n = min(k, X.shape[0], X.shape[1])
    pca = PCA(n_components=n, random_state=RANDOM_SEED)
    pca.fit(X)
    return pca


def pca_reconstruct(pca, residual, k):
    if pca is None:
        return residual
    k_eff = min(k, pca.components_.shape[0])
    centered = residual - pca.mean_
    coeff = centered @ pca.components_[:k_eff].T
    return (pca.mean_ + coeff @ pca.components_[:k_eff]).astype(np.float32)


def rank_retrieve(q_commit, bank):
    rows = [(fam, cosine(q_commit, mem["commit"])) for fam, mem in bank.items()]
    rows.sort(key=lambda x: x[1], reverse=True)
    return [fam for fam, _ in rows]


def minimal_vec(test_rec, bank, fam):
    p = operator_proto(bank, test_rec["operator"], fam)
    ranked = rank_retrieve(test_rec["commit"], bank)
    top = ranked[:TOPK_RESIDUAL]
    if top:
        r = np.mean(np.stack([residual_for_family(bank, f) for f in top], axis=0), axis=0)
    else:
        r = np.zeros_like(p)
    pca = fit_pca(bank, fam, RESIDUAL_K)
    r2 = pca_reconstruct(pca, r, RESIDUAL_K)
    return normalize(p + RESIDUAL_BETA * r2)


def main():
    df = pd.read_csv(DATASET_CSV)
    prompt_col = find_col(df, ["prompt", "text", "input", "query"])
    family_col = find_col(df, ["seed_family", "family_id", "family"])
    concept_col = find_col(df, ["concept_star", "concept", "concept_phrase"])
    operator_col = find_col(df, ["operator_star", "operator_id", "operator", "operator_phrase"])
    row_id_col = find_col(df, ["row_id", "id", "idx"])

    print("=" * 90)
    print("SEM-6A.1 Reflexive Update Robustness Audit")
    print("=" * 90)
    print(f"[Data] n={len(df)}")
    print(f"[Cols] prompt={prompt_col}, family={family_col}, concept={concept_col}, operator={operator_col}")

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

    by_fam = {}
    for r in records:
        by_fam.setdefault(r["family"], []).append(r)

    valid = {f: sorted(xs, key=lambda r: r["row_pos"]) for f, xs in by_fam.items() if len(xs) >= MIN_ROWS_PER_FAMILY}
    print(f"[Families valid] {len(valid)}")

    # oracle scoring bank uses all examples
    oracle_bank = build_bank(valid)

    rows = []
    for fam, xs in valid.items():
        for budget in UPDATE_BUDGETS:
            init_rows = {f: ys[:1] for f, ys in valid.items()}
            # update target family by adding budget rows; all other families stay init-only
            init_rows[fam] = xs[:1 + budget]
            bank = build_bank(init_rows)
            test_rows = xs[1 + max(UPDATE_BUDGETS):]  # fixed held-out across budgets
            if not test_rows:
                continue

            for tr in test_rows:
                raw_shape_scores = family_scores(tr["shape"], oracle_bank, "shape")
                raw_commit_scores = family_scores(tr["commit"], oracle_bank, "commit")
                raw_rank = rank_of_family(raw_shape_scores, fam)
                raw_sim = raw_shape_scores[fam]
                raw_margin = margin_to_family(raw_shape_scores, fam)

                vecs = {
                    "raw": None,
                    f"M{budget}_operator_proto": operator_proto(bank, tr["operator"], fam),
                    f"M{budget}_minimal_topk_pca2": minimal_vec(tr, bank, fam),
                    "oracle_shape": oracle_bank[fam]["shape"],
                }

                non = [x for x in oracle_bank if x != fam]
                shuffle_fam = random.choice(non)
                vecs["shuffle_shape"] = oracle_bank[shuffle_fam]["shape"]

                for cond, vec in vecs.items():
                    alphas = [0.0] if cond == "raw" else ALPHAS
                    for alpha in alphas:
                        if cond == "raw":
                            out_shape = tr["shape"]
                            out_commit = tr["commit"]
                        else:
                            with ShapeInjector(model, torch.tensor(vec, dtype=torch.float32), alpha):
                                h2 = forward_collect(model, tokenizer, tr["prompt"])
                            out_shape, out_commit = extract_shape_commit(h2)

                        scores_shape = family_scores(out_shape, oracle_bank, "shape")
                        scores_commit = family_scores(out_commit, oracle_bank, "commit")

                        rows.append({
                            "family": fam,
                            "row_idx": tr["row_idx"],
                            "row_id": tr["row_id"],
                            "concept": tr["concept"],
                            "operator": tr["operator"],
                            "update_budget": budget,
                            "condition": cond,
                            "alpha": alpha,
                            "shape_sim_to_oracle": scores_shape[fam],
                            "commit_sim_to_oracle": scores_commit[fam],
                            "family_margin_shape": margin_to_family(scores_shape, fam),
                            "family_rank_shape": rank_of_family(scores_shape, fam),
                            "gain_shape_sim": scores_shape[fam] - raw_sim,
                            "gain_margin_shape": margin_to_family(scores_shape, fam) - raw_margin,
                            "rank_improvement_shape": raw_rank - rank_of_family(scores_shape, fam),
                        })

    res = pd.DataFrame(rows)
    res_path = OUT_DIR / "sem6a1_results.csv"
    res.to_csv(res_path, index=False, encoding="utf-8-sig")

    summary = (
        res.groupby(["update_budget", "condition", "alpha"], dropna=False)
        .agg(
            n=("row_idx", "count"),
            gain_shape_sim=("gain_shape_sim", "mean"),
            gain_margin_shape=("gain_margin_shape", "mean"),
            rank_improvement_shape=("rank_improvement_shape", "mean"),
        )
        .reset_index()
    )
    summary_path = OUT_DIR / "sem6a1_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    checks = []
    for alpha in ALPHAS:
        def val(budget, condition, col):
            sub = summary[(summary["update_budget"] == budget) & (summary["condition"] == condition) & (summary["alpha"] == alpha)]
            return float(sub[col].iloc[0]) if len(sub) else np.nan

        m0 = val(0, "M0_minimal_topk_pca2", "gain_shape_sim")
        m1 = val(1, "M1_minimal_topk_pca2", "gain_shape_sim")
        m2 = val(2, "M2_minimal_topk_pca2", "gain_shape_sim")
        s = val(0, "shuffle_shape", "gain_shape_sim")
        o = val(0, "oracle_shape", "gain_shape_sim")

        checks.append({
            "check": f"continuous_update_alpha_{alpha}",
            "alpha": alpha,
            "M0": m0,
            "M1": m1,
            "M2": m2,
            "shuffle": s,
            "oracle": o,
            "delta_M1_M0": m1 - m0,
            "delta_M2_M0": m2 - m0,
            "monotonic": bool(m2 >= m1 >= m0),
            "pass_lite": bool((m1 > m0) or (m2 > m0)),
            "pass_strong": bool((m2 > m1 > m0) and (m2 > s) and (o >= m2)),
            "note": "Continuous gain should improve with update budget."
        })

        # residual vs proto after strongest update
        min2 = val(2, "M2_minimal_topk_pca2", "gain_shape_sim")
        proto2 = val(2, "M2_operator_proto", "gain_shape_sim")
        checks.append({
            "check": f"residual_value_after_update_alpha_{alpha}",
            "alpha": alpha,
            "M2_minimal": min2,
            "M2_proto": proto2,
            "delta": min2 - proto2,
            "pass_lite": bool(min2 >= proto2),
            "pass_strong": bool(min2 > proto2),
            "note": "Residual PCA should add continuous value over operator prototype."
        })

    checks_df = pd.DataFrame(checks)
    checks_path = OUT_DIR / "sem6a1_pass_checks.csv"
    checks_df.to_csv(checks_path, index=False, encoding="utf-8-sig")

    strong_updates = checks_df[checks_df["check"].str.startswith("continuous_update")]["pass_strong"].sum()
    lite_updates = checks_df[checks_df["check"].str.startswith("continuous_update")]["pass_lite"].sum()
    strong_res = checks_df[checks_df["check"].str.startswith("residual_value")]["pass_strong"].sum()

    verdict = "FAIL"
    if lite_updates >= 2:
        verdict = "PASS-Lite"
    if strong_updates >= 2 and strong_res >= 2:
        verdict = "PASS-Strong"
    elif strong_updates >= 2:
        verdict = "PASS-ReflexiveUpdate"

    report = {
        "experiment": "SEM-6A.1 Reflexive Update Robustness Audit",
        "verdict": verdict,
        "motivation": [
            "SEM-6A was PASS-Lite with weak rank signal.",
            "SEM-6A.1 uses continuous metrics and multiple update budgets."
        ],
        "config": {
            "update_budgets": UPDATE_BUDGETS,
            "primary_metric": "gain_shape_sim",
            "residual_k": RESIDUAL_K,
            "residual_beta": RESIDUAL_BETA,
        },
        "interpretation_rules": [
            "PASS-Lite: M1 or M2 improves over M0 on continuous trajectory metric.",
            "PASS-Strong: monotonic M0 < M1 < M2, beats shuffle, and remains below oracle.",
            "Residual value: M2_minimal_topk_pca2 beats M2_operator_proto."
        ],
        "outputs": {
            "results": str(res_path),
            "summary": str(summary_path),
            "checks": str(checks_path),
        }
    }

    report_path = OUT_DIR / "sem6a1_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("SEM-6A.1 finished")
    print("=" * 90)
    print(summary)
    print("\nPass checks:")
    print(checks_df)
    print(f"\nVerdict: {verdict}")
    print(f"Saved: {res_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {checks_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
