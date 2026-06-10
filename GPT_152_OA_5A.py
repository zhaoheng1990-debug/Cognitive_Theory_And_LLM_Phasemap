# ============================================================
# Tangent-Audit-1 / OA-5A
# Verify whether TopK(H_l W^T) behaves like local tangent space
#
# Core hypothesis:
#   Same topic + same task + different paraphrases should form
#   a compact local TopK trajectory cluster in shallow layers L0-L6.
#
# Expected:
#   D_sameTask < D_diffEntitySameTask < D_sameEntityDiffTask
#
# Model default:
#   Qwen2.5-1.5B-Instruct local HF path
# ============================================================

import os
import gc
import math
import random
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"
# If you run on Linux/macOS, replace MODEL_PATH with your local snapshot path.

SAVE_DIR = Path("./tangent_audit_oa5a_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SEED = 42
MAX_LEN = 128
BATCH_SIZE = 8

# Shallow layers. hidden_states[0] is embedding output; hidden_states[1] is layer0 output.
# We name emb=-1, transformer layers 0..6.
TRACK_HIDDEN_INDICES = list(range(0, 8))
LAYER_NAMES = ["emb"] + [f"L{i}" for i in range(7)]

TOPK_LIST = [100, 500, 1000]

# -----------------------------
# SEED
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# -----------------------------
# DATASET
# -----------------------------
# 6 topics × 4 tasks × 6 paraphrases = 144 prompts
# Tasks are intentionally reused across topics to test:
#   same entity diff task
#   diff entity same task
#   same entity same task paraphrase convergence

TOPICS = {
    "Paris": {
        "definition": [
            "What is Paris?",
            "Explain what Paris is.",
            "Give a brief definition of Paris.",
            "Describe Paris in one or two sentences.",
            "What kind of place is Paris?",
            "Tell me the basic identity of Paris.",
        ],
        "landmark": [
            "What are famous landmarks in Paris?",
            "List notable attractions in Paris.",
            "Which places should visitors see in Paris?",
            "Name important historical sites in Paris.",
            "What are the best-known tourist sights in Paris?",
            "Give examples of iconic places in Paris.",
        ],
        "relation": [
            "What country is Paris in?",
            "Paris belongs to which country?",
            "Which nation contains Paris?",
            "Identify the country associated with Paris.",
            "Where is Paris located politically?",
            "Paris is the capital of which country?",
        ],
        "property": [
            "What is Paris known for?",
            "Describe some characteristics of Paris.",
            "What features make Paris distinctive?",
            "What cultural qualities are associated with Paris?",
            "What are typical attributes of Paris?",
            "What makes Paris notable?",
        ],
    },
    "Tokyo": {
        "definition": [
            "What is Tokyo?",
            "Explain what Tokyo is.",
            "Give a brief definition of Tokyo.",
            "Describe Tokyo in one or two sentences.",
            "What kind of place is Tokyo?",
            "Tell me the basic identity of Tokyo.",
        ],
        "landmark": [
            "What are famous landmarks in Tokyo?",
            "List notable attractions in Tokyo.",
            "Which places should visitors see in Tokyo?",
            "Name important historical sites in Tokyo.",
            "What are the best-known tourist sights in Tokyo?",
            "Give examples of iconic places in Tokyo.",
        ],
        "relation": [
            "What country is Tokyo in?",
            "Tokyo belongs to which country?",
            "Which nation contains Tokyo?",
            "Identify the country associated with Tokyo.",
            "Where is Tokyo located politically?",
            "Tokyo is the capital of which country?",
        ],
        "property": [
            "What is Tokyo known for?",
            "Describe some characteristics of Tokyo.",
            "What features make Tokyo distinctive?",
            "What cultural qualities are associated with Tokyo?",
            "What are typical attributes of Tokyo?",
            "What makes Tokyo notable?",
        ],
    },
    "Apple fruit": {
        "definition": [
            "What is an apple fruit?",
            "Explain what an apple is as a fruit.",
            "Give a brief definition of an apple fruit.",
            "Describe the fruit called apple.",
            "What kind of fruit is an apple?",
            "Tell me the basic identity of an apple fruit.",
        ],
        "landmark": [
            "List common varieties of apple fruit.",
            "What are well-known kinds of apples?",
            "Name notable apple cultivars.",
            "Give examples of apple varieties.",
            "Which apple types are commonly recognized?",
            "What are famous varieties of apples?",
        ],
        "relation": [
            "What plant family does the apple fruit belong to?",
            "Apple fruit belongs to which plant group?",
            "Which botanical category contains apples?",
            "Identify the biological group associated with apples.",
            "Where does apple fit in plant classification?",
            "What type of plant product is an apple?",
        ],
        "property": [
            "What are apples known for nutritionally?",
            "Describe some characteristics of apple fruit.",
            "What features make apples distinctive?",
            "What qualities are associated with apples?",
            "What are typical attributes of apple fruit?",
            "What makes apples notable as food?",
        ],
    },
    "Apple company": {
        "definition": [
            "What is Apple Inc.?",
            "Explain what Apple the company is.",
            "Give a brief definition of Apple Inc.",
            "Describe Apple as a technology company.",
            "What kind of company is Apple?",
            "Tell me the basic identity of Apple Inc.",
        ],
        "landmark": [
            "List notable products made by Apple.",
            "What are famous Apple products?",
            "Which devices is Apple known for?",
            "Name important product lines from Apple.",
            "What are the best-known Apple devices?",
            "Give examples of iconic Apple products.",
        ],
        "relation": [
            "What industry is Apple Inc. in?",
            "Apple Inc. belongs to which industry?",
            "Which sector contains Apple the company?",
            "Identify the business category associated with Apple.",
            "Where does Apple fit in the technology sector?",
            "Apple is known as a company in what field?",
        ],
        "property": [
            "What is Apple Inc. known for?",
            "Describe some characteristics of Apple the company.",
            "What features make Apple distinctive as a company?",
            "What business qualities are associated with Apple?",
            "What are typical attributes of Apple Inc.?",
            "What makes Apple notable in technology?",
        ],
    },
    "Newton": {
        "definition": [
            "Who was Isaac Newton?",
            "Explain who Newton was.",
            "Give a brief definition of Isaac Newton.",
            "Describe Newton in one or two sentences.",
            "What kind of historical figure was Newton?",
            "Tell me the basic identity of Isaac Newton.",
        ],
        "landmark": [
            "List notable discoveries associated with Newton.",
            "What are famous contributions by Newton?",
            "Which ideas is Newton known for?",
            "Name important achievements of Isaac Newton.",
            "What are Newton's best-known scientific contributions?",
            "Give examples of iconic Newtonian ideas.",
        ],
        "relation": [
            "What field is Newton associated with?",
            "Newton belongs to which scientific tradition?",
            "Which discipline contains Newton's work?",
            "Identify the field associated with Isaac Newton.",
            "Where does Newton fit in science history?",
            "Newton is known as a figure in what field?",
        ],
        "property": [
            "What is Newton known for?",
            "Describe some characteristics of Newton's work.",
            "What features make Newton distinctive?",
            "What intellectual qualities are associated with Newton?",
            "What are typical attributes of Newton's legacy?",
            "What makes Isaac Newton notable?",
        ],
    },
    "Python language": {
        "definition": [
            "What is Python programming language?",
            "Explain what Python is in computing.",
            "Give a brief definition of Python language.",
            "Describe Python as a programming language.",
            "What kind of programming language is Python?",
            "Tell me the basic identity of Python in software.",
        ],
        "landmark": [
            "List common uses of Python programming language.",
            "What are well-known applications of Python?",
            "Which tasks is Python commonly used for?",
            "Name important use cases for Python.",
            "What are the best-known domains for Python?",
            "Give examples of typical Python applications.",
        ],
        "relation": [
            "What programming paradigm is Python associated with?",
            "Python belongs to which programming language category?",
            "Which software ecosystem contains Python?",
            "Identify the computing field associated with Python.",
            "Where does Python fit in programming languages?",
            "Python is known as a language in what field?",
        ],
        "property": [
            "What is Python known for?",
            "Describe some characteristics of Python language.",
            "What features make Python distinctive?",
            "What qualities are associated with Python programming?",
            "What are typical attributes of Python language?",
            "What makes Python notable in software?",
        ],
    },
}

TASKS = ["definition", "landmark", "relation", "property"]


def build_dataset():
    rows = []
    for topic, task_map in TOPICS.items():
        for task in TASKS:
            for pidx, prompt in enumerate(task_map[task]):
                rows.append({
                    "prompt_id": f"{topic}::{task}::{pidx}",
                    "topic": topic,
                    "task": task,
                    "paraphrase_id": pidx,
                    "prompt": prompt,
                })
    return pd.DataFrame(rows)

# -----------------------------
# MODEL
# -----------------------------
def load_model():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None,
        local_files_only=True,
        trust_remote_code=True,
    )
    if DEVICE == "cpu":
        model.to(DEVICE)
    model.eval()
    return tokenizer, model

# -----------------------------
# HELPERS
# -----------------------------
def l2_normalize(x, eps=1e-8):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def cosine_distance(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(1.0 - np.dot(a, b))


def jaccard_distance(set_a, set_b):
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / max(1, len(a | b))


def get_last_positions(attention_mask):
    # left padding: last real token is at sequence_length - 1
    return torch.full(
        (attention_mask.shape[0],),
        attention_mask.shape[1] - 1,
        dtype=torch.long,
        device=attention_mask.device,
    )


def extract_topk_states(df, tokenizer, model):
    """Extract TopK ids, centers, spreads for emb/L0-L6."""
    W = model.lm_head.weight.detach().float()
    W_norm = torch.nn.functional.normalize(W, dim=1)

    records = []
    prompts = df["prompt"].tolist()

    for start in range(0, len(prompts), BATCH_SIZE):
        batch_prompts = prompts[start:start+BATCH_SIZE]
        batch_df = df.iloc[start:start+BATCH_SIZE].reset_index(drop=True)
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        ).to(DEVICE)

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, use_cache=False)
            hs_all = out.hidden_states  # embedding + layer outputs
            last_idx = get_last_positions(inputs["attention_mask"])

            for local_i in range(len(batch_prompts)):
                meta = batch_df.iloc[local_i].to_dict()
                for hidx, lname in zip(TRACK_HIDDEN_INDICES, LAYER_NAMES):
                    h = hs_all[hidx][local_i, last_idx[local_i], :].detach().float()
                    logits = torch.matmul(W, h)

                    max_k = max(TOPK_LIST)
                    top_vals, top_ids = torch.topk(logits, k=max_k, dim=0)
                    top_ids_cpu = top_ids.detach().cpu().numpy().astype(np.int64)
                    top_vals_cpu = top_vals.detach().cpu().numpy().astype(np.float32)

                    for k in TOPK_LIST:
                        ids_k = top_ids[:k]
                        emb_k = W[ids_k]
                        center = emb_k.mean(dim=0)
                        center_norm = torch.nn.functional.normalize(center, dim=0)
                        cos_to_center = torch.matmul(W_norm[ids_k], center_norm)
                        spread = torch.mean(1.0 - cos_to_center).item()

                        # TopK logit shape descriptors for extra diagnostics
                        vals = top_vals_cpu[:k]
                        logit_mean = float(np.mean(vals))
                        logit_std = float(np.std(vals))
                        logit_gap = float(vals[0] - vals[-1])

                        rec = dict(meta)
                        rec.update({
                            "layer_name": lname,
                            "layer_index": hidx - 1,  # emb=-1, L0=0
                            "k": k,
                            "topk_ids": " ".join(map(str, top_ids_cpu[:k].tolist())),
                            "center": center.detach().cpu().numpy().astype(np.float32),
                            "spread": spread,
                            "logit_mean": logit_mean,
                            "logit_std": logit_std,
                            "logit_gap": logit_gap,
                        })
                        records.append(rec)

        del inputs, out, hs_all
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Extracted {min(start+BATCH_SIZE, len(prompts))}/{len(prompts)} prompts")

    return records


def make_feature_matrix(records, k):
    """Prompt-level feature vector = concatenated shallow centers + spreads/logit stats."""
    df_rec = pd.DataFrame([{kk: vv for kk, vv in r.items() if kk != "center"} for r in records])
    centers = [r["center"] for r in records]
    df_rec["center_obj"] = centers

    rows = []
    for prompt_id, g in df_rec[df_rec["k"] == k].groupby("prompt_id"):
        g = g.sort_values("layer_index")
        meta = g.iloc[0][["prompt_id", "topic", "task", "paraphrase_id", "prompt"]].to_dict()
        vec_parts = []
        for _, row in g.iterrows():
            vec_parts.append(row["center_obj"])
            vec_parts.append(np.array([row["spread"], row["logit_mean"], row["logit_std"], row["logit_gap"]], dtype=np.float32))
        vec = np.concatenate(vec_parts).astype(np.float32)
        meta["feature"] = vec
        rows.append(meta)
    return rows


def pairwise_distance_report(feature_rows, records, k):
    """Compute D_sameTask, D_diffEntitySameTask, D_sameEntityDiffTask."""
    feat_df = pd.DataFrame([{kk: vv for kk, vv in r.items() if kk != "feature"} for r in feature_rows])
    X = np.stack([r["feature"] for r in feature_rows])
    Xn = l2_normalize(X)

    # Build id -> TopK sets per layer for jaccard trajectory metric
    rec_df = pd.DataFrame([{kk: vv for kk, vv in r.items() if kk != "center"} for r in records if r["k"] == k])

    set_map = {}
    for _, row in rec_df.iterrows():
        set_map[(row["prompt_id"], row["layer_name"])] = list(map(int, row["topk_ids"].split()))

    pair_rows = []
    for i, j in combinations(range(len(feat_df)), 2):
        a = feat_df.iloc[i]
        b = feat_df.iloc[j]
        same_topic = a["topic"] == b["topic"]
        same_task = a["task"] == b["task"]

        if same_topic and same_task:
            pair_type = "sameTask"
        elif same_topic and not same_task:
            pair_type = "sameEntityDiffTask"
        elif (not same_topic) and same_task:
            pair_type = "diffEntitySameTask"
        else:
            pair_type = "diffEntityDiffTask"

        center_dist = float(1.0 - np.dot(Xn[i], Xn[j]))

        jac_dists = []
        for lname in LAYER_NAMES:
            jac_dists.append(jaccard_distance(
                set_map[(a["prompt_id"], lname)],
                set_map[(b["prompt_id"], lname)]
            ))
        jac_traj = float(np.mean(jac_dists))

        pair_rows.append({
            "k": k,
            "pair_type": pair_type,
            "prompt_a": a["prompt_id"],
            "prompt_b": b["prompt_id"],
            "topic_a": a["topic"],
            "topic_b": b["topic"],
            "task_a": a["task"],
            "task_b": b["task"],
            "center_traj_cosine_distance": center_dist,
            "topk_jaccard_trajectory_distance": jac_traj,
        })

    pair_df = pd.DataFrame(pair_rows)
    summary = pair_df.groupby("pair_type").agg(
        n=("pair_type", "size"),
        center_dist_mean=("center_traj_cosine_distance", "mean"),
        center_dist_std=("center_traj_cosine_distance", "std"),
        jaccard_dist_mean=("topk_jaccard_trajectory_distance", "mean"),
        jaccard_dist_std=("topk_jaccard_trajectory_distance", "std"),
    ).reset_index()

    return pair_df, summary


def tangent_low_dim_report(feature_rows, k):
    """Within each topic+task, estimate low-dimensionality of paraphrase cluster."""
    rows = []
    df = pd.DataFrame([{kk: vv for kk, vv in r.items() if kk != "feature"} for r in feature_rows])
    X_all = np.stack([r["feature"] for r in feature_rows])

    for (topic, task), idxs in df.groupby(["topic", "task"]).groups.items():
        idxs = list(idxs)
        X = X_all[idxs]
        X = X - X.mean(axis=0, keepdims=True)
        n = X.shape[0]
        if n < 3:
            continue
        pca = PCA(n_components=min(n, 5), random_state=SEED)
        pca.fit(X)
        evr = pca.explained_variance_ratio_
        rows.append({
            "k": k,
            "topic": topic,
            "task": task,
            "n": n,
            "pc1": float(evr[0]),
            "pc2": float(evr[1]) if len(evr) > 1 else 0.0,
            "pc3": float(evr[2]) if len(evr) > 2 else 0.0,
            "pc1_2": float(np.sum(evr[:2])),
            "pc1_3": float(np.sum(evr[:3])),
            "effective_dim_90": int(np.searchsorted(np.cumsum(evr), 0.90) + 1),
        })
    return pd.DataFrame(rows)


def classification_report(feature_rows, k):
    """Can shallow TopK trajectory classify task beyond topic leakage?"""
    df = pd.DataFrame([{kk: vv for kk, vv in r.items() if kk != "feature"} for r in feature_rows])
    X = np.stack([r["feature"] for r in feature_rows])
    y_task = df["task"].astype("category").cat.codes.values
    y_topic = df["topic"].astype("category").cat.codes.values
    groups_topic = df["topic"].astype("category").cat.codes.values

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)),
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    task_cv = cross_val_score(pipe, X, y_task, cv=skf, scoring="accuracy")
    topic_cv = cross_val_score(pipe, X, y_topic, cv=skf, scoring="accuracy")

    # Leave-topic-out task classification: if task remains predictable when topic held out,
    # then task/tangent structure is not purely entity identity.
    gkf = GroupKFold(n_splits=len(np.unique(groups_topic)))
    task_g = cross_val_score(pipe, X, y_task, cv=gkf, groups=groups_topic, scoring="accuracy")

    return pd.DataFrame([{
        "k": k,
        "task_strat_acc_mean": float(np.mean(task_cv)),
        "task_strat_acc_std": float(np.std(task_cv)),
        "topic_strat_acc_mean": float(np.mean(topic_cv)),
        "topic_strat_acc_std": float(np.std(topic_cv)),
        "task_leave_topic_out_acc_mean": float(np.mean(task_g)),
        "task_leave_topic_out_acc_std": float(np.std(task_g)),
    }])


def evaluate_pass(summary_all, lowdim_all, cls_all):
    rows = []
    for k in sorted(summary_all["k"].unique()):
        s = summary_all[summary_all["k"] == k].set_index("pair_type")
        try:
            D_same = float(s.loc["sameTask", "center_dist_mean"])
            D_diff_task = float(s.loc["sameEntityDiffTask", "center_dist_mean"])
            D_diff_entity_same_task = float(s.loc["diffEntitySameTask", "center_dist_mean"])
        except Exception:
            continue

        order_ok = D_same < D_diff_entity_same_task < D_diff_task
        task_sep_ratio = D_diff_task / max(D_same, 1e-8)
        lowdim = lowdim_all[lowdim_all["k"] == k]
        mean_pc12 = float(lowdim["pc1_2"].mean())
        mean_eff90 = float(lowdim["effective_dim_90"].mean())
        cls = cls_all[cls_all["k"] == k].iloc[0]
        leave_topic_task_acc = float(cls["task_leave_topic_out_acc_mean"])

        if order_ok and task_sep_ratio > 1.20 and mean_pc12 > 0.75 and leave_topic_task_acc > 0.45:
            verdict = "PASS-Strong"
        elif D_same < D_diff_task and mean_pc12 > 0.60:
            verdict = "PASS-Lite"
        else:
            verdict = "FAIL/Mixed"

        rows.append({
            "k": k,
            "D_sameTask": D_same,
            "D_diffEntitySameTask": D_diff_task,
            "D_diffEntitySameTask_over_sameTask": task_sep_ratio,
            "D_diffEntitySameTask_middle_expected": D_diff_entity_same_task,
            "expected_order_D_same_lt_diffEntitySameTask_lt_sameEntityDiffTask": bool(order_ok),
            "mean_pc12_within_sameTask": mean_pc12,
            "mean_effective_dim_90": mean_eff90,
            "leave_topic_out_task_acc": leave_topic_task_acc,
            "verdict": verdict,
        })
    return pd.DataFrame(rows)


def main():
    df = build_dataset()
    df.to_csv(SAVE_DIR / "oa5a_prompt_dataset.csv", index=False, encoding="utf-8-sig")
    print("Dataset:", df.shape)
    print(df.head())

    tokenizer, model = load_model()
    records = extract_topk_states(df, tokenizer, model)

    # Save record table without huge center arrays.
    rec_save = pd.DataFrame([{kk: vv for kk, vv in r.items() if kk != "center"} for r in records])
    rec_save.to_csv(SAVE_DIR / "oa5a_topk_records.csv", index=False, encoding="utf-8-sig")

    pair_dfs = []
    summary_dfs = []
    lowdim_dfs = []
    cls_dfs = []

    for k in TOPK_LIST:
        print(f"\nAnalyzing k={k} ...")
        feature_rows = make_feature_matrix(records, k)
        pair_df, summary_df = pairwise_distance_report(feature_rows, records, k)
        lowdim_df = tangent_low_dim_report(feature_rows, k)
        cls_df = classification_report(feature_rows, k)

        pair_dfs.append(pair_df)
        summary_dfs.append(summary_df.assign(k=k))
        lowdim_dfs.append(lowdim_df)
        cls_dfs.append(cls_df)

    pair_all = pd.concat(pair_dfs, ignore_index=True)
    summary_all = pd.concat(summary_dfs, ignore_index=True)
    lowdim_all = pd.concat(lowdim_dfs, ignore_index=True)
    cls_all = pd.concat(cls_dfs, ignore_index=True)
    verdict = evaluate_pass(summary_all, lowdim_all, cls_all)

    pair_all.to_csv(SAVE_DIR / "oa5a_pairwise_distances.csv", index=False, encoding="utf-8-sig")
    summary_all.to_csv(SAVE_DIR / "oa5a_distance_summary.csv", index=False, encoding="utf-8-sig")
    lowdim_all.to_csv(SAVE_DIR / "oa5a_lowdim_tangent_summary.csv", index=False, encoding="utf-8-sig")
    cls_all.to_csv(SAVE_DIR / "oa5a_classification_summary.csv", index=False, encoding="utf-8-sig")
    verdict.to_csv(SAVE_DIR / "oa5a_verdict_summary.csv", index=False, encoding="utf-8-sig")

    print("\n==== Distance Summary ====")
    print(summary_all)
    print("\n==== Low-dimensional Tangent Summary ====")
    print(lowdim_all.groupby("k")[["pc1", "pc1_2", "pc1_3", "effective_dim_90"]].mean().reset_index())
    print("\n==== Classification Summary ====")
    print(cls_all)
    print("\n==== Verdict ====")
    print(verdict)
    print(f"\nSaved outputs to: {SAVE_DIR.resolve()}")


if __name__ == "__main__":
    main()
