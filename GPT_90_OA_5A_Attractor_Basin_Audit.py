# ============================================================
# OA-5A: Attractor Basin Audit
#
# Goal:
#   Test whether prompts with the same semantic task but different
#   surface forms converge to a stable shallow initialization basin:
#
#       Prompt -> TopK(H_emb..H6) -> Omega_init
#
# Core tests:
#   H1 same topic + same task + different paraphrase should be close.
#   H2 same topic + different task should be farther.
#   H3 different topic + same task should be intermediate / separable.
#
# Outputs:
#   oa5a_dataset.csv
#   oa5a_pairwise_distances.csv
#   oa5a_layer_profile.csv
#   oa5a_summary.json
#
# Notes:
#   - This script does not require any previous OA CSV.
#   - It only needs a local HuggingFace causal LM path.
#   - Tested design target: Qwen2.5-1.5B-Instruct-like decoder-only model.
# ============================================================

import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

SAVE_DIR = Path("./oa5a_attractor_basin_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

BATCH_SIZE = 4
MAX_LEN = 256

# hidden_states[0] = embedding output
# hidden_states[1] = after transformer layer 0
# ...
INIT_STATE_INDICES = list(range(0, 8))  # embedding + L0-L6

TOPK_LIST = [100, 500, 1000]
MAX_TOPK = max(TOPK_LIST)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# DATASET: topics x tasks x paraphrases
# ============================================================

TOPICS = {
    "paris": {
        "surface_names": ["巴黎", "法国巴黎", "Paris"],
        "definition": [
            "巴黎是什么？",
            "请简单解释巴黎这个地方。",
            "巴黎指的是什么城市？",
            "介绍一下巴黎。",
            "Paris 是什么地方？",
            "用一句话说明巴黎是什么。",
        ],
        "landmark_list": [
            "巴黎有哪些名胜古迹？",
            "巴黎著名景点有哪些？",
            "去巴黎旅游最值得看的地方有哪些？",
            "请列举巴黎的代表性地标。",
            "法国巴黎有什么有名的景点？",
            "巴黎有哪些值得参观的历史建筑或景点？",
        ],
        "relation": [
            "巴黎属于哪个国家？",
            "巴黎和法国是什么关系？",
            "巴黎是哪国的首都？",
            "巴黎位于哪个国家？",
            "法国和巴黎之间是什么关系？",
            "请判断巴黎对应的国家。",
        ],
        "property": [
            "巴黎的气候怎么样？",
            "巴黎这座城市有什么特点？",
            "巴黎的城市风格是什么？",
            "巴黎给人的典型印象是什么？",
            "巴黎有哪些文化特征？",
            "巴黎作为城市有什么主要属性？",
        ],
    },
    "tokyo": {
        "surface_names": ["东京", "日本东京", "Tokyo"],
        "definition": [
            "东京是什么？",
            "请简单解释东京这个地方。",
            "东京指的是什么城市？",
            "介绍一下东京。",
            "Tokyo 是什么地方？",
            "用一句话说明东京是什么。",
        ],
        "landmark_list": [
            "东京有哪些著名景点？",
            "去东京旅游最值得看的地方有哪些？",
            "请列举东京的代表性地标。",
            "日本东京有什么有名的景点？",
            "东京有哪些值得参观的地方？",
            "东京的热门旅游地点有哪些？",
        ],
        "relation": [
            "东京属于哪个国家？",
            "东京和日本是什么关系？",
            "东京是哪国的首都？",
            "东京位于哪个国家？",
            "日本和东京之间是什么关系？",
            "请判断东京对应的国家。",
        ],
        "property": [
            "东京的城市特点是什么？",
            "东京的生活节奏怎么样？",
            "东京给人的典型印象是什么？",
            "东京有哪些文化特征？",
            "东京作为城市有什么主要属性？",
            "东京的城市风格是什么？",
        ],
    },
    "apple_fruit": {
        "surface_names": ["苹果这种水果", "水果苹果", "apple fruit"],
        "definition": [
            "苹果是什么水果？",
            "请解释苹果这种水果。",
            "苹果作为水果是什么？",
            "介绍一下水果苹果。",
            "apple fruit 是什么？",
            "用一句话说明苹果这种水果。",
        ],
        "landmark_list": [
            "苹果常见品种有哪些？",
            "苹果有哪些常见种类？",
            "请列举几种苹果品种。",
            "常见的苹果类型有哪些？",
            "水果苹果可以分成哪些品种？",
            "苹果有哪些代表性品类？",
        ],
        "relation": [
            "苹果和水果是什么关系？",
            "苹果属于什么植物类别？",
            "苹果属于水果吗？",
            "苹果和植物有什么关系？",
            "苹果在食物分类中属于什么？",
            "请判断苹果这种水果的类别关系。",
        ],
        "property": [
            "苹果的营养特点是什么？",
            "苹果这种水果有什么特征？",
            "苹果通常是什么味道？",
            "苹果有哪些常见属性？",
            "苹果的口感和用途有什么特点？",
            "苹果作为水果有什么主要特性？",
        ],
    },
    "apple_company": {
        "surface_names": ["苹果公司", "Apple 公司", "Apple Inc."],
        "definition": [
            "苹果公司是什么？",
            "请解释 Apple 公司。",
            "Apple Inc. 是什么公司？",
            "介绍一下苹果公司。",
            "苹果公司是做什么的？",
            "用一句话说明 Apple Inc.。",
        ],
        "landmark_list": [
            "苹果公司的代表性产品有哪些？",
            "Apple 有哪些知名产品？",
            "请列举苹果公司的重要产品。",
            "苹果公司最有代表性的设备有哪些？",
            "Apple 的核心产品线有哪些？",
            "苹果公司有哪些著名硬件或服务？",
        ],
        "relation": [
            "苹果公司和 iPhone 是什么关系？",
            "Apple 和智能手机产业有什么关系？",
            "苹果公司属于什么行业？",
            "Apple 与消费电子是什么关系？",
            "苹果公司和科技产业有什么关系？",
            "请判断 Apple Inc. 的产业归属。",
        ],
        "property": [
            "苹果公司的品牌特点是什么？",
            "Apple 的设计风格有什么特点？",
            "苹果公司给人的典型印象是什么？",
            "苹果公司的商业特点是什么？",
            "Apple 产品生态有什么特征？",
            "苹果公司作为企业有什么主要属性？",
        ],
    },
    "newton": {
        "surface_names": ["牛顿", "Isaac Newton", "艾萨克·牛顿"],
        "definition": [
            "牛顿是谁？",
            "请简单介绍牛顿。",
            "Isaac Newton 是什么人物？",
            "牛顿在科学史上是谁？",
            "用一句话说明牛顿是谁。",
            "艾萨克·牛顿是什么人？",
        ],
        "landmark_list": [
            "牛顿有哪些重要贡献？",
            "请列举牛顿的代表性成就。",
            "牛顿在科学上提出了哪些重要思想？",
            "牛顿的主要科学贡献有哪些？",
            "Isaac Newton 有哪些著名成果？",
            "牛顿最有名的发现或理论有哪些？",
        ],
        "relation": [
            "牛顿和万有引力是什么关系？",
            "牛顿和经典力学有什么关系？",
            "牛顿属于哪个科学领域？",
            "牛顿和物理学是什么关系？",
            "牛顿与微积分有什么关系？",
            "请判断牛顿和自然科学的关系。",
        ],
        "property": [
            "牛顿的研究特点是什么？",
            "牛顿的科学风格有什么特征？",
            "牛顿给人的典型历史印象是什么？",
            "牛顿作为科学家有什么主要属性？",
            "牛顿的学术影响有什么特点？",
            "牛顿的思想贡献有什么特征？",
        ],
    },
    "python": {
        "surface_names": ["Python", "Python 编程语言", "python language"],
        "definition": [
            "Python 是什么？",
            "请解释 Python 编程语言。",
            "Python 指的是什么语言？",
            "介绍一下 Python。",
            "Python language 是什么？",
            "用一句话说明 Python。",
        ],
        "landmark_list": [
            "Python 常用库有哪些？",
            "请列举 Python 的代表性库。",
            "Python 生态中有哪些常见工具？",
            "Python 有哪些常用框架？",
            "Python 开发常见库包括哪些？",
            "Python 的重要软件包有哪些？",
        ],
        "relation": [
            "Python 和编程是什么关系？",
            "Python 属于什么类型的语言？",
            "Python 和人工智能开发有什么关系？",
            "Python 与数据科学是什么关系？",
            "Python 和软件开发有什么关系？",
            "请判断 Python 的技术领域归属。",
        ],
        "property": [
            "Python 的语言特点是什么？",
            "Python 为什么常被认为易学？",
            "Python 的语法风格有什么特点？",
            "Python 作为编程语言有什么属性？",
            "Python 的优势和特征是什么？",
            "Python 的典型使用特点是什么？",
        ],
    },
}

TASKS = ["definition", "landmark_list", "relation", "property"]

def build_dataset():
    rows = []
    for topic_id, topic_data in TOPICS.items():
        for task_id in TASKS:
            prompts = topic_data[task_id]
            for para_id, prompt in enumerate(prompts):
                rows.append({
                    "topic_id": topic_id,
                    "task_id": task_id,
                    "paraphrase_id": para_id,
                    "prompt_id": f"{topic_id}__{task_id}__p{para_id}",
                    "prompt": prompt,
                })
    return pd.DataFrame(rows)

df = build_dataset()
print("Dataset rows:", len(df))
print(df.groupby(["topic_id", "task_id"]).size().head())

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model/tokenizer...")

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

num_layers = len(model.model.layers)
print("Num layers:", num_layers)

W_cpu = model.lm_head.weight.detach().float().cpu().numpy()
W_norm = W_cpu / (np.linalg.norm(W_cpu, axis=1, keepdims=True) + 1e-9)
print("W shape:", W_norm.shape)

# ============================================================
# FORWARD + TOPK FEATURES
# ============================================================

def state_name(si):
    return "emb" if si == 0 else f"L{si-1}"

def entropy_np(vals):
    vals = vals.astype(np.float64)
    vals = vals - np.max(vals)
    p = np.exp(vals)
    p = p / (np.sum(p) + 1e-12)
    return float(-np.sum(p * np.log(p + 1e-12)))

@torch.no_grad()
def compute_batch(prompts):
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    out = model(
        **enc,
        output_hidden_states=True,
        use_cache=False,
    )

    hs = out.hidden_states
    attn = enc["attention_mask"]
    last_idx = attn.sum(dim=1) - 1

    rows = []
    centers_payload = {}
    topids_payload = {}

    for si in INIT_STATE_INDICES:
        lname = state_name(si)
        for k in TOPK_LIST:
            key = f"{lname}_k{k}"
            centers_payload[key] = []
            topids_payload[key] = []

    for bi in range(len(prompts)):
        item = {}

        for si in INIT_STATE_INDICES:
            lname = state_name(si)
            h = hs[si][bi, last_idx[bi], :]
            h = h.to(model.lm_head.weight.dtype)
            logits = model.lm_head(h)

            top_vals, top_idx = torch.topk(logits, k=MAX_TOPK)
            vals = top_vals.detach().float().cpu().numpy()
            idx = top_idx.detach().cpu().numpy().astype(np.int32)

            for k in TOPK_LIST:
                vals_k = vals[:k]
                idx_k = idx[:k]

                item[f"{lname}_k{k}_top1"] = float(vals_k[0])
                item[f"{lname}_k{k}_kth"] = float(vals_k[-1])
                item[f"{lname}_k{k}_meanlogit"] = float(np.mean(vals_k))
                item[f"{lname}_k{k}_stdlogit"] = float(np.std(vals_k))
                item[f"{lname}_k{k}_gap_top1_kth"] = float(vals_k[0] - vals_k[-1])
                item[f"{lname}_k{k}_entropy"] = entropy_np(vals_k)

                center = np.mean(W_norm[idx_k], axis=0)
                center = center / (np.linalg.norm(center) + 1e-9)

                key = f"{lname}_k{k}"
                centers_payload[key].append(center.astype(np.float32))
                topids_payload[key].append(idx_k.copy())

        rows.append(item)

    return rows, centers_payload, topids_payload

all_rows = []
center_store = {}
topid_store = {}

for si in INIT_STATE_INDICES:
    lname = state_name(si)
    for k in TOPK_LIST:
        key = f"{lname}_k{k}"
        center_store[key] = []
        topid_store[key] = []

for start in range(0, len(df), BATCH_SIZE):
    sub = df.iloc[start:start+BATCH_SIZE]
    rows, centers_payload, topids_payload = compute_batch(sub["prompt"].tolist())
    all_rows.extend(rows)

    for key in center_store:
        center_store[key].extend(centers_payload[key])
        topid_store[key].extend(topids_payload[key])

    print(f"Processed {min(start+BATCH_SIZE, len(df))}/{len(df)}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

feat_df = pd.DataFrame(all_rows)
df = pd.concat([df.reset_index(drop=True), feat_df], axis=1)

for key in center_store:
    center_store[key] = np.stack(center_store[key], axis=0)
    topid_store[key] = np.stack(topid_store[key], axis=0)

# ============================================================
# PAIRWISE DISTANCES
# ============================================================

def cosine_distance_matrix(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    sim = Xn @ Xn.T
    return 1.0 - sim

def jaccard_distance(a, b):
    s1 = set(a.tolist())
    s2 = set(b.tolist())
    inter = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return 1.0 - inter / max(union, 1)

pair_rows = []

meta = df[["prompt_id", "topic_id", "task_id", "paraphrase_id", "prompt"]].reset_index(drop=True)

for key in center_store:
    centers = center_store[key]
    topids = topid_store[key]
    center_dist = cosine_distance_matrix(centers)

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            same_topic = meta.loc[i, "topic_id"] == meta.loc[j, "topic_id"]
            same_task = meta.loc[i, "task_id"] == meta.loc[j, "task_id"]

            if same_topic and same_task:
                pair_type = "same_topic_same_task"
            elif same_topic and not same_task:
                pair_type = "same_topic_diff_task"
            elif (not same_topic) and same_task:
                pair_type = "diff_topic_same_task"
            else:
                pair_type = "diff_topic_diff_task"

            pair_rows.append({
                "feature_key": key,
                "layer": key.split("_k")[0],
                "topk": int(key.split("_k")[-1]),
                "i": i,
                "j": j,
                "prompt_i": meta.loc[i, "prompt"],
                "prompt_j": meta.loc[j, "prompt"],
                "topic_i": meta.loc[i, "topic_id"],
                "topic_j": meta.loc[j, "topic_id"],
                "task_i": meta.loc[i, "task_id"],
                "task_j": meta.loc[j, "task_id"],
                "pair_type": pair_type,
                "center_cos_dist": float(center_dist[i, j]),
                "topk_jaccard_dist": float(jaccard_distance(topids[i], topids[j])),
            })

pair_df = pd.DataFrame(pair_rows)

# combined distance
pair_df["combined_dist"] = 0.5 * pair_df["center_cos_dist"] + 0.5 * pair_df["topk_jaccard_dist"]

# ============================================================
# LAYER PROFILE
# ============================================================

profile_rows = []

for (feature_key, layer, topk), sub in pair_df.groupby(["feature_key", "layer", "topk"]):
    means = sub.groupby("pair_type")["combined_dist"].mean().to_dict()

    same_task = means.get("same_topic_same_task", np.nan)
    same_entity_diff_task = means.get("same_topic_diff_task", np.nan)
    diff_entity_same_task = means.get("diff_topic_same_task", np.nan)
    diff_all = means.get("diff_topic_diff_task", np.nan)

    attractor_ratio = same_task / (same_entity_diff_task + 1e-12)
    task_generalization_ratio = same_task / (diff_entity_same_task + 1e-12)

    # Silhouette by topic_task cluster
    labels_topic_task = (df["topic_id"] + "__" + df["task_id"]).values
    labels_task = df["task_id"].values
    labels_topic = df["topic_id"].values

    centers = center_store[feature_key]
    try:
        sil_topic_task = float(silhouette_score(centers, labels_topic_task, metric="cosine"))
    except Exception:
        sil_topic_task = np.nan
    try:
        sil_task = float(silhouette_score(centers, labels_task, metric="cosine"))
    except Exception:
        sil_task = np.nan
    try:
        sil_topic = float(silhouette_score(centers, labels_topic, metric="cosine"))
    except Exception:
        sil_topic = np.nan

    profile_rows.append({
        "feature_key": feature_key,
        "layer": layer,
        "topk": int(topk),
        "same_task_dist": float(same_task),
        "same_entity_diff_task_dist": float(same_entity_diff_task),
        "diff_entity_same_task_dist": float(diff_entity_same_task),
        "diff_entity_diff_task_dist": float(diff_all),
        "attractor_ratio_same_over_same_entity_diff_task": float(attractor_ratio),
        "same_over_diff_entity_same_task": float(task_generalization_ratio),
        "separation_margin_entity_task": float(same_entity_diff_task - same_task),
        "separation_margin_diff_topic_same_task": float(diff_entity_same_task - same_task),
        "silhouette_topic_task": sil_topic_task,
        "silhouette_task": sil_task,
        "silhouette_topic": sil_topic,
    })

profile_df = pd.DataFrame(profile_rows)
profile_df = profile_df.sort_values(
    ["attractor_ratio_same_over_same_entity_diff_task", "same_task_dist"],
    ascending=[True, True],
).reset_index(drop=True)

best_row = profile_df.iloc[0].to_dict()

# ============================================================
# CLASSIFICATION sanity check
#   Can shallow basin centers predict task across held-out topics?
#   This is not the main result, but useful diagnostic.
# ============================================================

def group_task_classification(feature_key, group_col="topic_id"):
    X = center_store[feature_key].astype(float)
    y = df["task_id"].values
    groups = df[group_col].values

    cv = GroupKFold(n_splits=min(6, len(np.unique(groups))))
    preds = np.empty(len(y), dtype=object)

    for tr, te in cv.split(X, y, groups):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ])
        clf.fit(X[tr], y[tr])
        preds[te] = clf.predict(X[te])

    return {
        "feature_key": feature_key,
        "group_col": group_col,
        "acc": float(accuracy_score(y, preds)),
        "macro_f1": float(f1_score(y, preds, average="macro")),
    }

classification_results = []
for feature_key in center_store:
    classification_results.append(group_task_classification(feature_key, "topic_id"))

class_df = pd.DataFrame(classification_results).sort_values("macro_f1", ascending=False)

# ============================================================
# SUMMARY
# ============================================================

def interpret_ratio(r):
    if r < 0.5:
        return "PASS-Strong"
    if r < 0.75:
        return "PASS-Mixed"
    if r < 0.85:
        return "Weak"
    return "FAIL"

summary = {
    "experiment": "OA-5A Attractor Basin Audit",
    "n_prompts": int(len(df)),
    "n_topics": int(df["topic_id"].nunique()),
    "n_tasks": int(df["task_id"].nunique()),
    "n_paraphrases_per_topic_task": int(df.groupby(["topic_id", "task_id"]).size().min()),
    "init_state_indices": INIT_STATE_INDICES,
    "topk_list": TOPK_LIST,
    "best_attractor_row": best_row,
    "best_interpretation": interpret_ratio(best_row["attractor_ratio_same_over_same_entity_diff_task"]),
    "top_10_layer_profile": profile_df.head(10).to_dict(orient="records"),
    "top_10_task_classification": class_df.head(10).to_dict(orient="records"),
    "main_metric_definition": {
        "same_task_dist": "same topic + same task + different paraphrase",
        "same_entity_diff_task_dist": "same topic + different task",
        "attractor_ratio": "same_task_dist / same_entity_diff_task_dist",
        "criterion": "<0.5 strong, 0.5-0.75 mixed, >0.85 fail",
    }
}

# ============================================================
# SAVE
# ============================================================

df.to_csv(SAVE_DIR / "oa5a_dataset.csv", index=False, encoding="utf-8-sig")
pair_df.to_csv(SAVE_DIR / "oa5a_pairwise_distances.csv", index=False, encoding="utf-8-sig")
profile_df.to_csv(SAVE_DIR / "oa5a_layer_profile.csv", index=False, encoding="utf-8-sig")
class_df.to_csv(SAVE_DIR / "oa5a_task_classification.csv", index=False, encoding="utf-8-sig")

with open(SAVE_DIR / "oa5a_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\nSaved to:", SAVE_DIR.resolve())
