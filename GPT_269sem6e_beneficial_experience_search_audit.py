# -*- coding: utf-8 -*-
"""
SEM-6E: Beneficial Experience Search Audit

Why SEM-6E
----------
SEM-6D found that same-family update candidates have negative oracle utility:
    oracle_utility mean < 0
    init_only is best
Therefore the failure is not only Q-proxy failure; the candidate pool itself is bad.

SEM-6E asks:
    Are beneficial update experiences available elsewhere in the memory space?

Candidate pools:
    1. same_family_update        -- original SEM-6D candidate pool
    2. same_operator_other_concept
    3. same_concept_other_operator
    4. commit_topk_neighbors
    5. random_other_family

For each target family/test row group:
    Evaluate candidate-derived memory update vectors on held-out tests.
    Search for best candidate by oracle utility.

Main questions:
    - Does best_same_operator > best_same_family?
    - Does best_commit_topk > same_family?
    - Are there positive-utility update experiences at all?
    - If yes, can candidate features predict utility?

Interpretation:
    If beneficial updates come from same_operator_other_concept:
        Reflexive update should update OperatorPrior, not family center.

    If beneficial updates are rare but detectable:
        Q must be a selection/search function, not a smoothing gate.

    If no pool has positive utility:
        current injection metric/setup cannot support SEM-6 reflexive updates.

Outputs:
    sem6e_outputs/
"""

import json
import random
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

MODEL_PATH = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"
DATASET_CSV = r"sem5a2_outputs\sem5a2_dataset.csv"

OUT_DIR = Path("sem6e_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SHAPE_LAYERS = list(range(7, 20))
COMMIT_LAYERS = list(range(23, 26))

MAX_LENGTH = 512
MAX_ROWS = None

ALPHAS = [0.30]  # use strongest alpha for utility search
ETA = 1.0
MIN_ROWS_PER_FAMILY = 4
TOPK_COMMIT = 5
RANDOM_CANDIDATES_PER_FAMILY = 5


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


def eval_vec(model, tokenizer, test_rows, fam, vec, oracle_bank, alpha=0.30):
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


def candidate_update_vec(init_rows, candidate_row, eta=1.0):
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
    print("SEM-6E Beneficial Experience Search Audit")
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

    all_records = [r for rows in all_rows_by_family.values() for r in rows]

    candidate_rows = []
    policy_rows = []

    for fam, sp in splits.items():
        init_rows = sp["init"]
        test_rows = sp["test"]
        if not test_rows:
            continue

        target = init_rows[0]
        init_vec = rows_to_shape(init_rows)
        init_gain = eval_vec(model, tokenizer, test_rows, fam, init_vec, oracle_bank, alpha=0.30)["gain_shape_sim"]

        # candidate pools
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

        for pool_name, cands in pools.items():
            for ci, cand in enumerate(cands):
                vec = candidate_update_vec(init_rows, cand, eta=ETA)
                evals = eval_vec(model, tokenizer, test_rows, fam, vec, oracle_bank, alpha=0.30)
                utility = evals["gain_shape_sim"] - init_gain
                feats = candidate_features(target, cand, init_rows, oracle_bank, fam)

                candidate_rows.append({
                    "family": fam,
                    "target_concept": target["concept"],
                    "target_operator": target["operator"],
                    "pool": pool_name,
                    "candidate_i": ci,
                    "candidate_family": cand["family"],
                    "candidate_concept": cand["concept"],
                    "candidate_operator": cand["operator"],
                    "init_gain": init_gain,
                    "candidate_gain": evals["gain_shape_sim"],
                    "oracle_utility": utility,
                    "candidate_margin_gain": evals["gain_margin_shape"],
                    "candidate_rank_improvement": evals["rank_improvement_shape"],
                    **{f"feat_{i}": float(v) for i, v in enumerate(feats)}
                })

        # policy-level best from each pool
        for pool_name, cands in pools.items():
            if not cands:
                continue
            eval_list = []
            for cand in cands:
                vec = candidate_update_vec(init_rows, cand, eta=ETA)
                ev = eval_vec(model, tokenizer, test_rows, fam, vec, oracle_bank, alpha=0.30)
                eval_list.append((cand, ev["gain_shape_sim"], ev))
            best_cand, best_gain, best_ev = sorted(eval_list, key=lambda x: x[1], reverse=True)[0]
            policy_rows.append({
                "family": fam,
                "pool": pool_name,
                "best_candidate_family": best_cand["family"],
                "best_candidate_concept": best_cand["concept"],
                "best_candidate_operator": best_cand["operator"],
                "init_gain": init_gain,
                "best_gain": best_gain,
                "best_utility": best_gain - init_gain,
                "best_margin_gain": best_ev["gain_margin_shape"],
                "best_rank_improvement": best_ev["rank_improvement_shape"],
            })

    cand_df = pd.DataFrame(candidate_rows)
    cand_path = OUT_DIR / "sem6e_candidate_utilities.csv"
    cand_df.to_csv(cand_path, index=False, encoding="utf-8-sig")

    policy_df = pd.DataFrame(policy_rows)
    policy_path = OUT_DIR / "sem6e_pool_best_policies.csv"
    policy_df.to_csv(policy_path, index=False, encoding="utf-8-sig")

    pool_summary = (
        cand_df.groupby("pool")
        .agg(
            n=("family", "count"),
            mean_utility=("oracle_utility", "mean"),
            median_utility=("oracle_utility", "median"),
            positive_rate=("oracle_utility", lambda s: float(np.mean(np.array(s) > 0))),
            mean_candidate_gain=("candidate_gain", "mean"),
            mean_init_gain=("init_gain", "mean"),
        )
        .reset_index()
    )
    pool_summary_path = OUT_DIR / "sem6e_pool_summary.csv"
    pool_summary.to_csv(pool_summary_path, index=False, encoding="utf-8-sig")

    policy_summary = (
        policy_df.groupby("pool")
        .agg(
            n=("family", "count"),
            mean_best_utility=("best_utility", "mean"),
            median_best_utility=("best_utility", "median"),
            positive_best_rate=("best_utility", lambda s: float(np.mean(np.array(s) > 0))),
            mean_best_gain=("best_gain", "mean"),
            mean_init_gain=("init_gain", "mean"),
        )
        .reset_index()
    )
    policy_summary_path = OUT_DIR / "sem6e_policy_summary.csv"
    policy_summary.to_csv(policy_summary_path, index=False, encoding="utf-8-sig")

    # Learn utility predictor
    feat_cols = [c for c in cand_df.columns if c.startswith("feat_")]
    X = cand_df[feat_cols].values.astype(np.float32)
    y = cand_df["oracle_utility"].values.astype(np.float32)
    groups = cand_df["family"].values

    pred = np.zeros_like(y)
    r2, corr, auc = np.nan, np.nan, np.nan
    if len(np.unique(groups)) >= 2 and len(cand_df) > 10:
        n_splits = min(5, len(np.unique(groups)))
        gkf = GroupKFold(n_splits=n_splits)
        for tr, te in gkf.split(X, y, groups):
            reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            reg.fit(X[tr], y[tr])
            pred[te] = reg.predict(X[te])
        r2 = float(r2_score(y, pred))
        corr = float(np.corrcoef(y, pred)[0,1]) if np.std(pred) > 1e-8 and np.std(y) > 1e-8 else 0.0
        labels = (y > 0).astype(int)
        if len(set(labels)) == 2:
            auc = float(roc_auc_score(labels, pred))

    cand_df["proxy_utility"] = pred
    cand_df.to_csv(cand_path, index=False, encoding="utf-8-sig")

    # Checks
    checks = []
    def summary_val(df, pool, col):
        v = df[df["pool"] == pool][col]
        return float(v.iloc[0]) if len(v) else np.nan

    samefam_mean = summary_val(pool_summary, "same_family_update", "mean_utility")
    sameop_mean = summary_val(pool_summary, "same_operator_other_concept", "mean_utility")
    topk_mean = summary_val(pool_summary, "commit_topk_neighbors", "mean_utility")
    rand_mean = summary_val(pool_summary, "random_other_family", "mean_utility")

    samefam_best = summary_val(policy_summary, "same_family_update", "mean_best_utility")
    sameop_best = summary_val(policy_summary, "same_operator_other_concept", "mean_best_utility")
    topk_best = summary_val(policy_summary, "commit_topk_neighbors", "mean_best_utility")
    rand_best = summary_val(policy_summary, "random_other_family", "mean_best_utility")

    checks.append({
        "check": "beneficial_pool_mean_same_operator",
        "value": sameop_mean,
        "same_family_mean": samefam_mean,
        "random_mean": rand_mean,
        "pass_lite": bool(sameop_mean > samefam_mean),
        "pass_strong": bool((sameop_mean > samefam_mean) and (sameop_mean > rand_mean) and (sameop_mean > 0)),
        "note": "Same-operator external experiences should have better mean utility than same-family updates."
    })

    checks.append({
        "check": "beneficial_pool_best_same_operator",
        "value": sameop_best,
        "same_family_best": samefam_best,
        "random_best": rand_best,
        "pass_lite": bool(sameop_best > samefam_best),
        "pass_strong": bool((sameop_best > samefam_best) and (sameop_best > rand_best) and (sameop_best > 0)),
        "note": "Best same-operator external experience should improve over best same-family update."
    })

    checks.append({
        "check": "beneficial_pool_best_commit_topk",
        "value": topk_best,
        "same_family_best": samefam_best,
        "random_best": rand_best,
        "pass_lite": bool(topk_best > samefam_best),
        "pass_strong": bool((topk_best > samefam_best) and (topk_best > rand_best) and (topk_best > 0)),
        "note": "Best commit-topk neighbor should improve over best same-family update."
    })

    checks.append({
        "check": "proxy_utility_predictability",
        "value_corr": corr,
        "value_r2": r2,
        "value_auc": auc,
        "pass_lite": bool((not np.isnan(corr)) and corr > 0.2),
        "pass_strong": bool((not np.isnan(corr)) and corr > 0.4 and (np.isnan(auc) or auc > 0.65)),
        "note": "A proxy utility predictor should recover held-out update utility."
    })

    checks_df = pd.DataFrame(checks)
    checks_path = OUT_DIR / "sem6e_pass_checks.csv"
    checks_df.to_csv(checks_path, index=False, encoding="utf-8-sig")

    strong = int(checks_df["pass_strong"].sum())
    lite = int(checks_df["pass_lite"].sum())
    verdict = "FAIL"
    if lite >= 1:
        verdict = "PASS-Lite"
    if strong >= 2:
        verdict = "PASS-Strong"

    report = {
        "experiment": "SEM-6E Beneficial Experience Search Audit",
        "verdict": verdict,
        "proxy_utility": {
            "groupkfold_r2": r2,
            "groupkfold_corr": corr,
            "groupkfold_auc_positive": auc,
            "n_candidates": int(len(cand_df)),
        },
        "interpretation_rules": [
            "If same-family utilities are negative but same-operator utilities are positive, reflexive update should target OperatorPrior rather than family mean.",
            "If beneficial candidates exist but proxy is weak, Q must be improved.",
            "If no candidate pool has positive utility, current SEM-6 setup lacks useful update candidates."
        ],
        "outputs": {
            "candidate_utilities": str(cand_path),
            "pool_best_policies": str(policy_path),
            "pool_summary": str(pool_summary_path),
            "policy_summary": str(policy_summary_path),
            "checks": str(checks_path),
        }
    }
    report_path = OUT_DIR / "sem6e_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("SEM-6E finished")
    print("=" * 90)
    print("Pool summary:")
    print(pool_summary)
    print("\nPolicy summary:")
    print(policy_summary)
    print("\nChecks:")
    print(checks_df)
    print(f"\nProxy utility R2={r2:.4f}, corr={corr:.4f}, auc={auc}")
    print(f"Verdict: {verdict}")
    print(f"Saved: {cand_path}")
    print(f"Saved: {policy_path}")
    print(f"Saved: {pool_summary_path}")
    print(f"Saved: {policy_summary_path}")
    print(f"Saved: {checks_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
