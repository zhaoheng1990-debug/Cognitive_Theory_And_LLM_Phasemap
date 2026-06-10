# -*- coding: utf-8 -*-
"""
SEM-6G: Online Proxy-Selected Reflexive Write Audit

Why SEM-6G
----------
SEM-6F showed:
    proxy-selected utility > same-family best
    proxy-selected utility > random best
    proxy-selected utility > 0
but:
    proxy-selected utility < commit-topk best

Therefore SEM-6G asks:
    In a live write pipeline, can Q_proxy-selected write beat:
        init_only
        same_family_update
        random_update
    and compete with:
        commit_topk_rule

SEM-6G differs from SEM-6F:
    SEM-6F used SEM-6E candidate utility table only.
    SEM-6G trains Q_proxy on SEM-6E table, then runs a fresh live intervention
    by selecting candidates and injecting the resulting write vector.

Inputs:
    sem6e_outputs/sem6e_candidate_utilities.csv
    sem5a2_outputs/sem5a2_dataset.csv

Candidate pools for live write:
    same_family_update
    same_operator_other_concept
    same_concept_other_operator
    commit_topk_neighbors
    random_other_family

Policies:
    init_only
    same_family_best_by_proxy
    proxy_selected_global
    commit_topk_rule
    random_best_by_proxy
    oracle_shape

Expected:
    proxy_selected_global > init_only
    proxy_selected_global > same_family_best_by_proxy
    proxy_selected_global > random_best_by_proxy
    oracle_shape >= proxy_selected_global

If proxy_selected_global >= commit_topk_rule:
    Q_proxy outperforms hand-coded retrieval heuristic.

If proxy_selected_global < commit_topk_rule but positive:
    Q_proxy is useful but commit-topk remains strong baseline.
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

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
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
CANDIDATE_UTILITY_CSV_CANDIDATES = [
    r"sem6e_outputs\sem6e_candidate_utilities.csv",
    r"sem6e_candidate_utilities.csv",
]

OUT_DIR = Path("sem6g_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SHAPE_LAYERS = list(range(7, 20))
COMMIT_LAYERS = list(range(23, 26))

MAX_LENGTH = 512
MAX_ROWS = None

ALPHAS = [0.10, 0.20, 0.30]
TOPK_COMMIT = 5
RANDOM_CANDIDATES_PER_FAMILY = 5
MIN_ROWS_PER_FAMILY = 4
N_INIT = 1
N_UPDATE_CANDIDATES = 2

ETA = 1.0


# =========================
# Helpers
# =========================

def find_candidate_utility_csv():
    for p in CANDIDATE_UTILITY_CSV_CANDIDATES:
        pp = Path(p)
        if pp.exists():
            return pp
    raise FileNotFoundError("Missing sem6e_candidate_utilities.csv. Run SEM-6E first.")


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
                "init": xs[:N_INIT],
                "updates": xs[N_INIT:N_INIT + N_UPDATE_CANDIDATES],
                "test": xs[N_INIT + N_UPDATE_CANDIDATES:],
                "all": xs,
            }
    return splits


def rows_to_shape(rows):
    return normalize(np.mean(np.stack([r["shape"] for r in rows], axis=0), axis=0))


def rows_to_commit(rows):
    return normalize(np.mean(np.stack([r["commit"] for r in rows], axis=0), axis=0))


def build_bank(rows_by_family):
    bank = {}
    for fam, rows in rows_by_family.items():
        if not rows:
            continue
        bank[fam] = {
            "shape": rows_to_shape(rows),
            "commit": rows_to_commit(rows),
            "concept": rows[0]["concept"],
            "operator": rows[0]["operator"],
        }
    return bank


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


def candidate_update_vec(init_rows, candidate_row, eta=ETA):
    init = rows_to_shape(init_rows)
    delta = candidate_row["shape"] - init
    return normalize(init + eta * delta)


def commit_topk_candidates(target_init_commit, all_family_rows, target_family, k=TOPK_COMMIT):
    family_commits = []
    for fam, rows in all_family_rows.items():
        if fam == target_family:
            continue
        family_commits.append((fam, rows_to_commit(rows), rows))
    family_commits.sort(key=lambda x: cosine(target_init_commit, x[1]), reverse=True)
    cands = []
    for fam, _, rows in family_commits[:k]:
        cands.extend(rows[:1])
    return cands


def candidate_features(target, candidate, init_rows, oracle_bank, target_family):
    init_shape = rows_to_shape(init_rows)
    init_commit = rows_to_commit(init_rows)
    cand_shape = candidate["shape"]
    cand_commit = candidate["commit"]

    return np.array([
        cosine(cand_shape, init_shape),
        cosine(cand_commit, init_commit),
        float(np.linalg.norm(cand_shape - init_shape)),
        float(np.linalg.norm(cand_commit - init_commit)),
        float(candidate["concept"] == target["concept"]),
        float(candidate["operator"] == target["operator"]),
        float(candidate["family"] == target_family),
        cosine(cand_shape, oracle_bank[target_family]["shape"]),
        cosine(cand_commit, oracle_bank[target_family]["commit"]),
    ], dtype=np.float32)


def eval_vec(model, tokenizer, test_rows, fam, vec, oracle_bank, alpha):
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

    return {
        "gain_shape_sim": float(np.mean([g["gain_shape_sim"] for g in gains])),
        "gain_margin_shape": float(np.mean([g["gain_margin_shape"] for g in gains])),
        "rank_improvement_shape": float(np.mean([g["rank_improvement_shape"] for g in gains])),
    }


def train_q_proxy():
    cand_path = find_candidate_utility_csv()
    cdf = pd.read_csv(cand_path)
    feat_cols = [c for c in cdf.columns if c.startswith("feat_")]
    X = cdf[feat_cols].values.astype(np.float32)
    y = cdf["oracle_utility"].values.astype(np.float32)
    model_q = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model_q.fit(X, y)
    return model_q, feat_cols, str(cand_path)


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
    print("SEM-6G Online Proxy-Selected Reflexive Write Audit")
    print("=" * 90)

    q_model, feat_cols, q_input = train_q_proxy()
    print(f"[Q proxy] trained from {q_input}, features={len(feat_cols)}")

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
    all_records = [r for rows in all_rows_by_family.values() for r in rows]

    rows = []
    selection_rows = []

    for fam, sp in splits.items():
        init_rows = sp["init"]
        test_rows = sp["test"]
        if not test_rows:
            continue
        target = init_rows[0]

        pools = {}
        pools["same_family_update"] = sp["updates"]

        pools["same_operator_other_concept"] = [
            r for r in all_records
            if r["family"] != fam and r["operator"] == target["operator"] and r["concept"] != target["concept"]
        ][:TOPK_COMMIT]

        pools["same_concept_other_operator"] = [
            r for r in all_records
            if r["family"] != fam and r["concept"] == target["concept"] and r["operator"] != target["operator"]
        ][:TOPK_COMMIT]

        pools["commit_topk_neighbors"] = commit_topk_candidates(rows_to_commit(init_rows), all_rows_by_family, fam, k=TOPK_COMMIT)

        non = [r for r in all_records if r["family"] != fam]
        pools["random_other_family"] = random.sample(non, min(RANDOM_CANDIDATES_PER_FAMILY, len(non)))

        # score candidates
        scored = []
        for pool, cands in pools.items():
            for cand in cands:
                feats = candidate_features(target, cand, init_rows, oracle_bank, fam)
                pred = float(q_model.predict(feats.reshape(1, -1))[0])
                scored.append((pool, cand, pred))

        if not scored:
            continue

        proxy_pool, proxy_cand, proxy_pred = sorted(scored, key=lambda x: x[2], reverse=True)[0]

        # Simple baseline: best predicted within same-family pool
        sf_scored = [x for x in scored if x[0] == "same_family_update"]
        if sf_scored:
            same_family_pool, same_family_cand, same_family_pred = sorted(sf_scored, key=lambda x: x[2], reverse=True)[0]
        else:
            same_family_pool, same_family_cand, same_family_pred = proxy_pool, proxy_cand, proxy_pred

        # Simple baseline: commit-topk highest predicted
        ct_scored = [x for x in scored if x[0] == "commit_topk_neighbors"]
        if ct_scored:
            ct_pool, ct_cand, ct_pred = sorted(ct_scored, key=lambda x: x[2], reverse=True)[0]
        else:
            ct_pool, ct_cand, ct_pred = proxy_pool, proxy_cand, proxy_pred

        # Random baseline: random-other highest predicted
        rnd_scored = [x for x in scored if x[0] == "random_other_family"]
        if rnd_scored:
            rnd_pool, rnd_cand, rnd_pred = sorted(rnd_scored, key=lambda x: x[2], reverse=True)[0]
        else:
            rnd_pool, rnd_cand, rnd_pred = proxy_pool, proxy_cand, proxy_pred

        policies = {
            "init_only": rows_to_shape(init_rows),
            "proxy_selected_global": candidate_update_vec(init_rows, proxy_cand),
            "same_family_best_by_proxy": candidate_update_vec(init_rows, same_family_cand),
            "commit_topk_rule": candidate_update_vec(init_rows, ct_cand),
            "random_best_by_proxy": candidate_update_vec(init_rows, rnd_cand),
            "oracle_shape": oracle_bank[fam]["shape"],
        }

        selection_rows.append({
            "family": fam,
            "proxy_pool": proxy_pool,
            "proxy_candidate_family": proxy_cand["family"],
            "proxy_pred": proxy_pred,
            "same_family_pred": same_family_pred,
            "commit_topk_pred": ct_pred,
            "random_pred": rnd_pred,
        })

        for policy, vec in policies.items():
            for alpha in ALPHAS:
                ev = eval_vec(model, tokenizer, test_rows, fam, vec, oracle_bank, alpha)
                rows.append({
                    "family": fam,
                    "policy": policy,
                    "alpha": alpha,
                    "proxy_selected_pool": proxy_pool,
                    "gain_shape_sim": ev["gain_shape_sim"],
                    "gain_margin_shape": ev["gain_margin_shape"],
                    "rank_improvement_shape": ev["rank_improvement_shape"],
                })

    res_df = pd.DataFrame(rows)
    res_path = OUT_DIR / "sem6g_live_write_results.csv"
    res_df.to_csv(res_path, index=False, encoding="utf-8-sig")

    sel_df = pd.DataFrame(selection_rows)
    sel_path = OUT_DIR / "sem6g_selection_log.csv"
    sel_df.to_csv(sel_path, index=False, encoding="utf-8-sig")

    summary = (
        res_df.groupby(["policy", "alpha"], dropna=False)
        .agg(
            n=("family", "count"),
            gain_shape_sim=("gain_shape_sim", "mean"),
            gain_margin_shape=("gain_margin_shape", "mean"),
            rank_improvement_shape=("rank_improvement_shape", "mean"),
        )
        .reset_index()
    )
    summary_path = OUT_DIR / "sem6g_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    pool_dist = (
        sel_df.groupby("proxy_pool")
        .agg(n=("family", "count"), mean_pred=("proxy_pred", "mean"))
        .reset_index()
        .sort_values("n", ascending=False)
    )
    pool_path = OUT_DIR / "sem6g_proxy_pool_distribution.csv"
    pool_dist.to_csv(pool_path, index=False, encoding="utf-8-sig")

    checks = []
    for alpha in ALPHAS:
        sub = summary[summary["alpha"] == alpha]

        def val(policy, col):
            v = sub[sub["policy"] == policy][col]
            return float(v.iloc[0]) if len(v) else np.nan

        proxy = val("proxy_selected_global", "gain_shape_sim")
        init = val("init_only", "gain_shape_sim")
        sf = val("same_family_best_by_proxy", "gain_shape_sim")
        ct = val("commit_topk_rule", "gain_shape_sim")
        rnd = val("random_best_by_proxy", "gain_shape_sim")
        oracle = val("oracle_shape", "gain_shape_sim")

        checks.append({
            "check": f"proxy_live_alpha_{alpha}",
            "alpha": alpha,
            "proxy": proxy,
            "init": init,
            "same_family": sf,
            "commit_topk": ct,
            "random": rnd,
            "oracle": oracle,
            "pass_lite": bool(proxy > init and proxy > sf and proxy > rnd),
            "pass_strong": bool((proxy > init) and (proxy > sf) and (proxy > rnd) and (oracle >= proxy)),
            "beats_commit_topk": bool(proxy >= ct),
            "note": "Proxy selected live write should beat init, same-family, random, and be bounded by oracle."
        })

    checks_df = pd.DataFrame(checks)
    checks_path = OUT_DIR / "sem6g_pass_checks.csv"
    checks_df.to_csv(checks_path, index=False, encoding="utf-8-sig")

    strong = int(checks_df["pass_strong"].sum())
    lite = int(checks_df["pass_lite"].sum())
    beats_ct = int(checks_df["beats_commit_topk"].sum())

    verdict = "FAIL"
    if lite >= 2:
        verdict = "PASS-Lite"
    if strong >= 2:
        verdict = "PASS-Strong"
    if strong >= 2 and beats_ct >= 2:
        verdict = "PASS-Strong-BeatsCommitTopK"

    report = {
        "experiment": "SEM-6G Online Proxy-Selected Reflexive Write Audit",
        "verdict": verdict,
        "q_proxy_source": q_input,
        "interpretation_rules": [
            "PASS-Lite: proxy selected live write beats init, same-family, and random.",
            "PASS-Strong: additionally bounded by oracle.",
            "PASS-Strong-BeatsCommitTopK: proxy also beats hand-coded commit-topk rule."
        ],
        "outputs": {
            "live_write_results": str(res_path),
            "selection_log": str(sel_path),
            "summary": str(summary_path),
            "proxy_pool_distribution": str(pool_path),
            "checks": str(checks_path),
        }
    }

    report_path = OUT_DIR / "sem6g_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 90)
    print("SEM-6G finished")
    print("=" * 90)
    print(summary)
    print("\nPool distribution:")
    print(pool_dist)
    print("\nChecks:")
    print(checks_df)
    print("\nVerdict:", verdict)
    print(f"Saved: {res_path}")
    print(f"Saved: {sel_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {pool_path}")
    print(f"Saved: {checks_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
