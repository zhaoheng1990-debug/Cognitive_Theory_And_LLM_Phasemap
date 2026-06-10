# -*- coding: utf-8 -*-
"""
SEM-6B: Quality-Gated Reflexive MemoryUnit Update Audit

Why SEM-6B
----------
SEM-6A.1 failed:
    M0, M1, M2 were almost identical.
This suggests that naive "add more same-family rows" is not a meaningful
reflexive update rule.

SEM-6B tests a better update operator:

    R(M_t, E_t, Q_t) = M_{t+1}

where Q_t is an internal quality gate.

Instead of blindly averaging update examples, SEM-6B only admits an update row
if it improves a MemoryUnit quality proxy.

Quality proxies:
    1) update_shape_fit:
        cos(update_shape, current_family_shape)
    2) update_commit_fit:
        cos(update_commit, current_family_commit)
    3) operator_consistency:
        cos(update_shape, operator_proto)
    4) residual_nontriviality:
        norm(update_shape - operator_proto)
    5) retrieval_self_hit:
        whether update query retrieves its own family in current bank

Update policies:
    naive_update:
        always add update rows
    quality_gated_update:
        add update rows only if gate_score >= threshold
    anti_gated_update:
        add update rows only if gate_score < threshold
    oracle_all:
        use all rows as upper bound

Main hypothesis:
    quality_gated_update > naive_update > anti_gated_update
    quality_gated_update > shuffle
    oracle_all >= quality_gated_update

This is a better test of reflexive update because it includes Q_t.

Outputs:
    sem6b_outputs/
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


# =========================
# Config
# =========================

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

MODEL_PATH = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"
DATASET_CSV = r"sem5a2_outputs\sem5a2_dataset.csv"

OUT_DIR = Path("sem6b_outputs")
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

# Update rows available after init and before test.
# For sem5a2 384 rows = 4 rows/family if 96 families, this gives:
# init 1, update candidates 2, test 1.
N_INIT = 1
N_UPDATE_CANDIDATES = 2

# Quality gate threshold mode:
# Use family-wise median of gate_score among update candidates.
GATE_MODE = "within_family_median"


# =========================
# Utilities
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
    shape = normalize(mean_pool_layers(hdict, SHAPE_LAYERS))
    commit = normalize(mean_pool_layers(hdict, COMMIT_LAYERS))
    return shape, commit


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


def make_family_splits(records):
    by_fam = {}
    for r in records:
        by_fam.setdefault(r["family"], []).append(r)
    valid = {}
    for fam, xs in by_fam.items():
        xs = sorted(xs, key=lambda r: r["row_pos"])
        if len(xs) >= MIN_ROWS_PER_FAMILY:
            valid[fam] = {
                "init": xs[:N_INIT],
                "updates": xs[N_INIT:N_INIT + N_UPDATE_CANDIDATES],
                "test": xs[N_INIT + N_UPDATE_CANDIDATES:],
                "all": xs,
            }
    return valid


def quality_score(update_row, current_bank, family):
    mem = current_bank[family]
    p = operator_proto(current_bank, update_row["operator"], family)

    update_shape_fit = cosine(update_row["shape"], mem["shape"])
    update_commit_fit = cosine(update_row["commit"], mem["commit"])
    operator_consistency = cosine(update_row["shape"], p)
    residual_nontriviality = float(np.linalg.norm(update_row["shape"] - p))

    ranked = rank_retrieve(update_row["commit"], current_bank)
    retrieval_self_hit = 1.0 if ranked and ranked[0] == family else 0.0

    # Normalize residual contribution softly; avoid letting norm dominate.
    residual_score = np.tanh(residual_nontriviality)

    # Gate favors update that is identity-consistent, operator-consistent, and not pure duplicate.
    score = (
        0.30 * update_shape_fit
        + 0.30 * update_commit_fit
        + 0.25 * operator_consistency
        + 0.10 * retrieval_self_hit
        + 0.05 * residual_score
    )

    return {
        "gate_score": float(score),
        "update_shape_fit": float(update_shape_fit),
        "update_commit_fit": float(update_commit_fit),
        "operator_consistency": float(operator_consistency),
        "residual_nontriviality": float(residual_nontriviality),
        "retrieval_self_hit": float(retrieval_self_hit),
    }


def select_updates(policy, init_rows_by_family, splits, family):
    init_rows = init_rows_by_family[family]
    updates = splits[family]["updates"]
    current_bank = build_bank(init_rows_by_family)

    scored = []
    for u in updates:
        q = quality_score(u, current_bank, family)
        scored.append((u, q))

    if policy == "init_only":
        return init_rows, []

    if policy == "naive_update":
        return init_rows + updates, scored

    if not scored:
        return init_rows, []

    scores = np.array([q["gate_score"] for _, q in scored], dtype=float)
    threshold = float(np.median(scores))

    if policy == "quality_gated_update":
        selected = [u for u, q in scored if q["gate_score"] >= threshold]
    elif policy == "anti_gated_update":
        selected = [u for u, q in scored if q["gate_score"] < threshold]
    else:
        raise ValueError(policy)

    return init_rows + selected, scored


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
    print("SEM-6B Quality-Gated Reflexive MemoryUnit Update Audit")
    print("=" * 90)
    print(f"[Data] n={len(df)}")

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
    splits = make_family_splits(records)
    print(f"[Splits] valid families={len(splits)}")

    init_rows_by_family = {fam: sp["init"] for fam, sp in splits.items()}
    all_rows_by_family = {fam: sp["all"] for fam, sp in splits.items()}
    oracle_bank = build_bank(all_rows_by_family)

    policies = ["init_only", "naive_update", "quality_gated_update", "anti_gated_update", "oracle_all"]

    result_rows = []
    gate_rows = []

    for fam, sp in splits.items():
        if not sp["test"]:
            continue

        for policy in policies:
            if policy == "oracle_all":
                rows_for_family = sp["all"]
                scored = []
            else:
                rows_for_family, scored = select_updates(policy, init_rows_by_family, splits, fam)

            # collect gate rows once per non-oracle policy
            if policy in ["naive_update", "quality_gated_update", "anti_gated_update"]:
                for u, q in scored:
                    gate_rows.append({
                        "family": fam,
                        "policy_context": policy,
                        "update_row_idx": u["row_idx"],
                        "concept": u["concept"],
                        "operator": u["operator"],
                        **q,
                    })

            rows_by_family = dict(init_rows_by_family)
            rows_by_family[fam] = rows_for_family
            bank = build_bank(rows_by_family)

            for tr in sp["test"]:
                raw_shape_scores = family_scores(tr["shape"], oracle_bank, "shape")
                raw_rank = rank_of_family(raw_shape_scores, fam)
                raw_sim = raw_shape_scores[fam]
                raw_margin = margin_to_family(raw_shape_scores, fam)

                if policy == "oracle_all":
                    vec = oracle_bank[fam]["shape"]
                else:
                    vec = minimal_vec(tr, bank, fam)

                for alpha in ALPHAS:
                    with ShapeInjector(model, torch.tensor(vec, dtype=torch.float32), alpha):
                        h2 = forward_collect(model, tokenizer, tr["prompt"])
                    out_shape, out_commit = extract_shape_commit(h2)

                    scores_shape = family_scores(out_shape, oracle_bank, "shape")
                    result_rows.append({
                        "family": fam,
                        "row_idx": tr["row_idx"],
                        "row_id": tr["row_id"],
                        "concept": tr["concept"],
                        "operator": tr["operator"],
                        "policy": policy,
                        "alpha": alpha,
                        "n_memory_rows_for_family": len(rows_for_family),
                        "shape_sim_to_oracle": scores_shape[fam],
                        "family_margin_shape": margin_to_family(scores_shape, fam),
                        "family_rank_shape": rank_of_family(scores_shape, fam),
                        "gain_shape_sim": scores_shape[fam] - raw_sim,
                        "gain_margin_shape": margin_to_family(scores_shape, fam) - raw_margin,
                        "rank_improvement_shape": raw_rank - rank_of_family(scores_shape, fam),
                    })

                # raw and shuffle only once per family/test
                if policy == "init_only":
                    for alpha in ALPHAS:
                        non = [x for x in oracle_bank if x != fam]
                        sfam = random.choice(non)
                        with ShapeInjector(model, torch.tensor(oracle_bank[sfam]["shape"], dtype=torch.float32), alpha):
                            h2 = forward_collect(model, tokenizer, tr["prompt"])
                        out_shape, _ = extract_shape_commit(h2)
                        scores_shape = family_scores(out_shape, oracle_bank, "shape")
                        result_rows.append({
                            "family": fam,
                            "row_idx": tr["row_idx"],
                            "row_id": tr["row_id"],
                            "concept": tr["concept"],
                            "operator": tr["operator"],
                            "policy": "shuffle_shape",
                            "alpha": alpha,
                            "n_memory_rows_for_family": 0,
                            "shape_sim_to_oracle": scores_shape[fam],
                            "family_margin_shape": margin_to_family(scores_shape, fam),
                            "family_rank_shape": rank_of_family(scores_shape, fam),
                            "gain_shape_sim": scores_shape[fam] - raw_sim,
                            "gain_margin_shape": margin_to_family(scores_shape, fam) - raw_margin,
                            "rank_improvement_shape": raw_rank - rank_of_family(scores_shape, fam),
                        })

    res = pd.DataFrame(result_rows)
    res_path = OUT_DIR / "sem6b_results.csv"
    res.to_csv(res_path, index=False, encoding="utf-8-sig")

    gates = pd.DataFrame(gate_rows)
    gates_path = OUT_DIR / "sem6b_gate_scores.csv"
    gates.to_csv(gates_path, index=False, encoding="utf-8-sig")

    summary = (
        res.groupby(["policy", "alpha"], dropna=False)
        .agg(
            n=("row_idx", "count"),
            mean_rows=("n_memory_rows_for_family", "mean"),
            gain_shape_sim=("gain_shape_sim", "mean"),
            gain_margin_shape=("gain_margin_shape", "mean"),
            rank_improvement_shape=("rank_improvement_shape", "mean"),
        )
        .reset_index()
    )
    summary_path = OUT_DIR / "sem6b_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    checks = []
    for alpha in ALPHAS:
        def val(policy, col):
            sub = summary[(summary["policy"] == policy) & (summary["alpha"] == alpha)]
            return float(sub[col].iloc[0]) if len(sub) else np.nan

        qg = val("quality_gated_update", "gain_shape_sim")
        naive = val("naive_update", "gain_shape_sim")
        anti = val("anti_gated_update", "gain_shape_sim")
        init = val("init_only", "gain_shape_sim")
        shuffle = val("shuffle_shape", "gain_shape_sim")
        oracle = val("oracle_all", "gain_shape_sim")

        checks.append({
            "check": f"quality_gate_continuous_alpha_{alpha}",
            "alpha": alpha,
            "quality_gated": qg,
            "naive": naive,
            "anti": anti,
            "init": init,
            "shuffle": shuffle,
            "oracle": oracle,
            "qg_minus_init": qg - init,
            "qg_minus_naive": qg - naive,
            "qg_minus_anti": qg - anti,
            "pass_lite": bool(qg > init),
            "pass_strong": bool((qg > init) and (qg >= naive) and (qg > anti) and (qg > shuffle) and (oracle >= qg)),
            "note": "Quality-gated update should beat init, anti-gate, shuffle, and be bounded by oracle."
        })

    checks_df = pd.DataFrame(checks)
    checks_path = OUT_DIR / "sem6b_pass_checks.csv"
    checks_df.to_csv(checks_path, index=False, encoding="utf-8-sig")

    strong = int(checks_df["pass_strong"].sum())
    lite = int(checks_df["pass_lite"].sum())
    verdict = "FAIL"
    if lite >= 2:
        verdict = "PASS-Lite"
    if strong >= 2:
        verdict = "PASS-Strong"

    gate_summary = {}
    if len(gates):
        gate_summary = {
            "mean_gate_score": float(gates["gate_score"].mean()),
            "mean_update_shape_fit": float(gates["update_shape_fit"].mean()),
            "mean_update_commit_fit": float(gates["update_commit_fit"].mean()),
            "mean_operator_consistency": float(gates["operator_consistency"].mean()),
            "mean_retrieval_self_hit": float(gates["retrieval_self_hit"].mean()),
        }

    report = {
        "experiment": "SEM-6B Quality-Gated Reflexive MemoryUnit Update Audit",
        "verdict": verdict,
        "motivation": [
            "SEM-6A.1 failed because naive update budgets did not improve MemoryUnit.",
            "SEM-6B tests R(M,E,Q) with an explicit internal quality gate Q_t."
        ],
        "gate_summary": gate_summary,
        "interpretation_rules": [
            "PASS-Lite: quality-gated update improves over init-only.",
            "PASS-Strong: quality-gated update beats init, anti-gate, shuffle, and is bounded by oracle.",
            "If naive does not beat init but quality-gated does, Q_t is necessary for reflexive update."
        ],
        "outputs": {
            "results": str(res_path),
            "gate_scores": str(gates_path),
            "summary": str(summary_path),
            "checks": str(checks_path),
        }
    }
    report_path = OUT_DIR / "sem6b_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("SEM-6B finished")
    print("=" * 90)
    print(summary)
    print("\nChecks:")
    print(checks_df)
    print("\nGate summary:")
    print(gate_summary)
    print(f"\nVerdict: {verdict}")
    print(f"Saved: {res_path}")
    print(f"Saved: {gates_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {checks_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
