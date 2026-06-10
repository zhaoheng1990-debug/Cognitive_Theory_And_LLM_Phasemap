# -*- coding: utf-8 -*-
"""
SEM-6C: Reflexive Update Failure Decomposition Audit

Why SEM-6C
----------
SEM-6A.1 failed.
SEM-6B quality-gated update also failed.

The failures are informative:
    - gate scores are saturated/high;
    - update rows are very similar to init rows;
    - oracle/all-family shape can even underperform operator prototype on gain_shape_sim;
    - naive and gated updates barely change the vector.

Therefore SEM-6C is a diagnostic audit, not a positive-control experiment.

It asks:
    Why did MemoryUnit update fail?

Hypotheses:
H1 Redundancy:
    update rows add almost no new trajectory information.
    prediction: ||M1-M0|| and ||M2-M0|| are tiny; update residual novelty is low.

H2 Metric mismatch:
    gain_shape_sim / rank may not capture update benefit.
    prediction: update changes commit/identity or residual coverage but not shape-sim.

H3 Update operator too weak:
    averaging same-family rows is too conservative.
    prediction: stronger update operators (replace / residual-add / novelty-weighted)
    change MemoryUnit more and may improve test trajectory.

SEM-6C compares update operators:
    M0_init
    M1_avg
    M2_avg
    M1_replace
    M2_replace
    M1_residual_add
    M2_residual_add
    M1_novelty_weighted
    M2_novelty_weighted
    oracle_all
    shuffle

Primary diagnostics:
    vector_delta_norm: ||M_policy - M0||
    residual_delta_norm
    test gain_shape_sim
    test gain_margin_shape
    rank_improvement_shape

Outputs:
    sem6c_outputs/
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

OUT_DIR = Path("sem6c_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SHAPE_LAYERS = list(range(7, 20))
COMMIT_LAYERS = list(range(23, 26))

MAX_LENGTH = 512
MAX_ROWS = None

ALPHAS = [0.10, 0.20, 0.30]
RESIDUAL_K = 2
RESIDUAL_BETA = 0.5
TOPK_RESIDUAL = 3
MIN_ROWS_PER_FAMILY = 4


# =========================
# utilities
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
                "init": xs[:1],
                "updates": xs[1:3],
                "test": xs[3:],
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


def operator_proto(bank, operator, exclude_family):
    xs = [m["shape"] for f, m in bank.items() if f != exclude_family and str(m["operator"]) == str(operator)]
    if not xs:
        xs = [m["shape"] for f, m in bank.items() if f != exclude_family]
    if not xs:
        xs = [m["shape"] for _, m in bank.items()]
    return normalize(np.mean(np.stack(xs, axis=0), axis=0))


def residual_for_family(bank, fam):
    mem = bank[fam]
    p = operator_proto(bank, mem["operator"], fam)
    return (mem["shape"] - p).astype(np.float32)


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


def rows_to_shape(rows):
    return normalize(np.mean(np.stack([r["shape"] for r in rows], axis=0), axis=0))


def update_shape_operator(policy, init_rows, update_rows):
    """
    Returns memory rows or direct shape override depending on policy.
    Policies:
      avg_k0 / avg_k1 / avg_k2: average rows
      replace_k1 / replace_k2: use only update rows
      residual_add_k1/k2: init + sum residuals relative to init
      novelty_weighted_k1/k2: weighted average favoring farthest update
    """
    init_shape = rows_to_shape(init_rows)
    if policy == "avg_k0":
        return init_shape

    if policy.endswith("k1"):
        ups = update_rows[:1]
    elif policy.endswith("k2"):
        ups = update_rows[:2]
    else:
        ups = update_rows

    if not ups:
        return init_shape

    if policy.startswith("avg"):
        return rows_to_shape(init_rows + ups)

    if policy.startswith("replace"):
        return rows_to_shape(ups)

    if policy.startswith("residual_add"):
        delta = np.mean(np.stack([u["shape"] - init_shape for u in ups], axis=0), axis=0)
        return normalize(init_shape + delta)

    if policy.startswith("novelty_weighted"):
        dists = np.array([1.0 - cosine(u["shape"], init_shape) for u in ups], dtype=np.float32)
        if float(np.sum(dists)) < 1e-8:
            weights = np.ones(len(ups), dtype=np.float32) / len(ups)
        else:
            weights = dists / np.sum(dists)
        upd_shape = normalize(np.sum(np.stack([w * u["shape"] for w, u in zip(weights, ups)], axis=0), axis=0))
        return normalize(0.5 * init_shape + 0.5 * upd_shape)

    raise ValueError(policy)


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
    print("SEM-6C Reflexive Update Failure Decomposition Audit")
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

    policies = [
        "avg_k0",
        "avg_k1",
        "avg_k2",
        "replace_k1",
        "replace_k2",
        "residual_add_k1",
        "residual_add_k2",
        "novelty_weighted_k1",
        "novelty_weighted_k2",
        "oracle_all",
        "shuffle_shape",
    ]

    rows = []
    diag_rows = []

    for fam, sp in splits.items():
        init = sp["init"]
        ups = sp["updates"]
        tests = sp["test"]

        init_shape = rows_to_shape(init)
        oracle_shape = oracle_bank[fam]["shape"]

        for policy in policies:
            if policy == "oracle_all":
                mem_shape = oracle_shape
            elif policy == "shuffle_shape":
                non = [x for x in oracle_bank if x != fam]
                sfam = random.choice(non)
                mem_shape = oracle_bank[sfam]["shape"]
            else:
                mem_shape = update_shape_operator(policy, init, ups)

            diag_rows.append({
                "family": fam,
                "policy": policy,
                "vector_delta_from_init": float(np.linalg.norm(mem_shape - init_shape)),
                "cos_to_init": cosine(mem_shape, init_shape),
                "cos_to_oracle": cosine(mem_shape, oracle_shape),
                "delta_cos_oracle_minus_init": cosine(mem_shape, oracle_shape) - cosine(init_shape, oracle_shape),
                "n_updates_available": len(ups),
            })

            for tr in tests:
                raw_scores = family_scores(tr["shape"], oracle_bank, "shape")
                raw_rank = rank_of_family(raw_scores, fam)
                raw_sim = raw_scores[fam]
                raw_margin = margin_to_family(raw_scores, fam)

                for alpha in ALPHAS:
                    with ShapeInjector(model, torch.tensor(mem_shape, dtype=torch.float32), alpha):
                        h2 = forward_collect(model, tokenizer, tr["prompt"])
                    out_shape, _ = extract_shape_commit(h2)
                    scores = family_scores(out_shape, oracle_bank, "shape")
                    rows.append({
                        "family": fam,
                        "row_idx": tr["row_idx"],
                        "row_id": tr["row_id"],
                        "concept": tr["concept"],
                        "operator": tr["operator"],
                        "policy": policy,
                        "alpha": alpha,
                        "shape_sim_to_oracle": scores[fam],
                        "family_margin_shape": margin_to_family(scores, fam),
                        "family_rank_shape": rank_of_family(scores, fam),
                        "gain_shape_sim": scores[fam] - raw_sim,
                        "gain_margin_shape": margin_to_family(scores, fam) - raw_margin,
                        "rank_improvement_shape": raw_rank - rank_of_family(scores, fam),
                    })

    res = pd.DataFrame(rows)
    res_path = OUT_DIR / "sem6c_results.csv"
    res.to_csv(res_path, index=False, encoding="utf-8-sig")

    diag = pd.DataFrame(diag_rows)
    diag_path = OUT_DIR / "sem6c_update_vector_diagnostics.csv"
    diag.to_csv(diag_path, index=False, encoding="utf-8-sig")

    summary = (
        res.groupby(["policy", "alpha"], dropna=False)
        .agg(
            n=("row_idx", "count"),
            gain_shape_sim=("gain_shape_sim", "mean"),
            gain_margin_shape=("gain_margin_shape", "mean"),
            rank_improvement_shape=("rank_improvement_shape", "mean"),
        )
        .reset_index()
    )
    summary_path = OUT_DIR / "sem6c_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    diag_summary = (
        diag.groupby("policy")
        .agg(
            n=("family", "count"),
            vector_delta_from_init=("vector_delta_from_init", "mean"),
            cos_to_init=("cos_to_init", "mean"),
            cos_to_oracle=("cos_to_oracle", "mean"),
            delta_cos_oracle_minus_init=("delta_cos_oracle_minus_init", "mean"),
        )
        .reset_index()
    )
    diag_summary_path = OUT_DIR / "sem6c_diagnostic_summary.csv"
    diag_summary.to_csv(diag_summary_path, index=False, encoding="utf-8-sig")

    checks = []
    for alpha in ALPHAS:
        sub = summary[summary["alpha"] == alpha]

        def val(policy, col):
            v = sub[sub["policy"] == policy][col]
            return float(v.iloc[0]) if len(v) else np.nan

        init = val("avg_k0", "gain_shape_sim")
        oracle = val("oracle_all", "gain_shape_sim")
        shuffle = val("shuffle_shape", "gain_shape_sim")

        for policy in policies:
            if policy in ["avg_k0", "oracle_all", "shuffle_shape"]:
                continue
            v = val(policy, "gain_shape_sim")
            checks.append({
                "check": f"{policy}_alpha_{alpha}",
                "policy": policy,
                "alpha": alpha,
                "value": v,
                "init": init,
                "shuffle": shuffle,
                "oracle": oracle,
                "gain_over_init": v - init,
                "pass_lite": bool(v > init),
                "pass_strong": bool((v > init) and (v > shuffle) and (oracle >= v)),
                "note": "Update operator should improve over init, beat shuffle, and remain bounded by oracle."
            })

    checks_df = pd.DataFrame(checks)
    checks_path = OUT_DIR / "sem6c_pass_checks.csv"
    checks_df.to_csv(checks_path, index=False, encoding="utf-8-sig")

    # identify best update operator
    best_by_alpha = []
    for alpha in ALPHAS:
        sub = summary[(summary["alpha"] == alpha) & (~summary["policy"].isin(["oracle_all", "shuffle_shape"]))]
        best = sub.sort_values("gain_shape_sim", ascending=False).iloc[0].to_dict()
        best_by_alpha.append(best)

    strong = int(checks_df["pass_strong"].sum())
    lite = int(checks_df["pass_lite"].sum())

    verdict = "FAIL"
    if lite >= 2:
        verdict = "PASS-Lite"
    if strong >= 2:
        verdict = "PASS-Strong"

    report = {
        "experiment": "SEM-6C Reflexive Update Failure Decomposition Audit",
        "verdict": verdict,
        "motivation": [
            "SEM-6B failed; gate scores were saturated and updates did not improve MemoryUnit.",
            "SEM-6C decomposes failure into redundancy, metric mismatch, and update-operator weakness."
        ],
        "best_by_alpha": best_by_alpha,
        "interpretation_rules": [
            "If all update operators have tiny vector_delta_from_init, failure is data redundancy.",
            "If stronger update operators change vectors but do not improve test gain, update direction is not aligned with future trajectory.",
            "If replace/residual-add improves while average fails, the update operator was too weak.",
            "If no operator improves, current SEM-6 update setup lacks a meaningful Q/update objective."
        ],
        "outputs": {
            "results": str(res_path),
            "update_vector_diagnostics": str(diag_path),
            "summary": str(summary_path),
            "diagnostic_summary": str(diag_summary_path),
            "checks": str(checks_path),
        }
    }
    report_path = OUT_DIR / "sem6c_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("SEM-6C finished")
    print("=" * 90)
    print("Diagnostic summary:")
    print(diag_summary)
    print("\nSummary:")
    print(summary)
    print("\nChecks:")
    print(checks_df)
    print(f"\nVerdict: {verdict}")
    print(f"Saved: {res_path}")
    print(f"Saved: {diag_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {diag_summary_path}")
    print(f"Saved: {checks_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
