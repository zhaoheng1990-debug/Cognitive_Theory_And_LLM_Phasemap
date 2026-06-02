# ============================================================
# OA-5B
# Task -> Entity Basin Transition Audit
#
# Hypothesis:
#
# L0-L6
#   task manifold
#
# L7-L19
#   relation fitting
#
# L20-L22
#   constraint entry
#
# L23-L26
#   entity / answer basin formation
#
# L27
#   output commitment
#
# ============================================================

import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    silhouette_score,
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

SAVE_DIR = Path("./oa5b_outputs")
SAVE_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

BATCH_SIZE = 4
MAX_LEN = 256

TOPK_LIST = [100, 500, 1000]
MAX_TOPK = max(TOPK_LIST)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# DATASET
# ============================================================

TOPICS = {

    "paris": {

        "definition": [
            "巴黎是什么？",
            "请简单解释巴黎这个地方。",
            "巴黎指的是什么城市？",
            "介绍一下巴黎。",
            "Paris是什么地方？",
            "用一句话说明巴黎是什么。"
        ],

        "landmark": [
            "巴黎有哪些名胜古迹？",
            "巴黎著名景点有哪些？",
            "巴黎最值得参观的地方有哪些？",
            "请列举巴黎的代表性地标。",
            "法国巴黎有哪些著名景点？",
            "巴黎有哪些历史建筑？"
        ],

        "relation": [
            "巴黎属于哪个国家？",
            "巴黎和法国是什么关系？",
            "巴黎是哪国首都？",
            "巴黎位于哪个国家？",
            "法国和巴黎之间是什么关系？",
            "请判断巴黎对应国家。"
        ],

        "property": [
            "巴黎有什么特点？",
            "巴黎的城市风格是什么？",
            "巴黎有哪些文化特征？",
            "巴黎给人的典型印象是什么？",
            "巴黎作为城市有什么主要属性？",
            "巴黎气候如何？"
        ],
    },

    "tokyo": {

        "definition": [
            "东京是什么？",
            "请简单解释东京。",
            "东京指的是什么城市？",
            "介绍一下东京。",
            "Tokyo是什么地方？",
            "用一句话说明东京。"
        ],

        "landmark": [
            "东京有哪些著名景点？",
            "东京最值得参观的地方有哪些？",
            "东京有哪些地标？",
            "请列举东京代表性景点。",
            "东京有哪些热门旅游地点？",
            "东京有哪些历史景观？"
        ],

        "relation": [
            "东京属于哪个国家？",
            "东京和日本是什么关系？",
            "东京是哪国首都？",
            "东京位于哪个国家？",
            "日本和东京之间是什么关系？",
            "请判断东京对应国家。"
        ],

        "property": [
            "东京有什么特点？",
            "东京城市风格如何？",
            "东京有哪些文化特征？",
            "东京给人的典型印象是什么？",
            "东京有哪些主要属性？",
            "东京生活节奏如何？"
        ],
    },

    "newton": {

        "definition": [
            "牛顿是谁？",
            "请介绍牛顿。",
            "Isaac Newton是谁？",
            "牛顿是什么人物？",
            "用一句话说明牛顿。",
            "艾萨克牛顿是谁？"
        ],

        "landmark": [
            "牛顿有哪些重要贡献？",
            "请列举牛顿主要成就。",
            "牛顿最著名发现有哪些？",
            "牛顿有哪些重要理论？",
            "Isaac Newton贡献有哪些？",
            "牛顿科学成果有哪些？"
        ],

        "relation": [
            "牛顿和万有引力是什么关系？",
            "牛顿和经典力学是什么关系？",
            "牛顿属于哪个领域？",
            "牛顿和物理学是什么关系？",
            "牛顿和微积分是什么关系？",
            "请判断牛顿所属领域。"
        ],

        "property": [
            "牛顿有哪些特点？",
            "牛顿研究风格如何？",
            "牛顿历史影响是什么？",
            "牛顿主要属性是什么？",
            "牛顿有哪些学术特征？",
            "牛顿典型形象是什么？"
        ],
    },

    "python": {

        "definition": [
            "Python是什么？",
            "请解释Python。",
            "Python是什么语言？",
            "介绍一下Python。",
            "Python language是什么？",
            "用一句话说明Python。"
        ],

        "landmark": [
            "Python常用库有哪些？",
            "请列举Python代表性库。",
            "Python有哪些常见框架？",
            "Python生态有哪些工具？",
            "Python常见软件包有哪些？",
            "Python开发常用库有哪些？"
        ],

        "relation": [
            "Python和编程是什么关系？",
            "Python属于什么语言？",
            "Python与AI开发是什么关系？",
            "Python和数据科学是什么关系？",
            "Python与软件开发是什么关系？",
            "请判断Python所属领域。"
        ],

        "property": [
            "Python有什么特点？",
            "Python为什么易学？",
            "Python语法风格如何？",
            "Python有哪些优势？",
            "Python主要属性是什么？",
            "Python典型使用特点是什么？"
        ],
    }
}

TASKS = [
    "definition",
    "landmark",
    "relation",
    "property"
]

def build_dataset():

    rows = []

    for topic_id, topic in TOPICS.items():

        for task_id in TASKS:

            prompts = topic[task_id]

            for pid, prompt in enumerate(prompts):

                rows.append({

                    "topic_id": topic_id,
                    "task_id": task_id,
                    "topic_task_id": f"{topic_id}__{task_id}",
                    "paraphrase_id": pid,
                    "prompt": prompt,
                })

    return pd.DataFrame(rows)

df = build_dataset()

print(df.shape)

# ============================================================
# LOAD MODEL
# ============================================================

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

NUM_LAYERS = len(model.model.layers)

print("Transformer layers:", NUM_LAYERS)

STATE_INDICES = list(range(0, NUM_LAYERS + 1))

# ============================================================
# LM HEAD GEOMETRY
# ============================================================

W_cpu = (
    model.lm_head.weight
    .detach()
    .float()
    .cpu()
    .numpy()
)

W_norm = (
    W_cpu /
    (np.linalg.norm(
        W_cpu,
        axis=1,
        keepdims=True
    ) + 1e-9)
)

print("W shape:", W_norm.shape)

# ============================================================
# HELPERS
# ============================================================

def state_name(idx):

    if idx == 0:
        return "emb"

    return f"L{idx-1}"

def entropy_np(vals):

    vals = vals.astype(np.float64)

    vals = vals - np.max(vals)

    p = np.exp(vals)

    p = p / (np.sum(p) + 1e-12)

    return float(
        -np.sum(
            p * np.log(p + 1e-12)
        )
    )

# ============================================================
# FORWARD
# ============================================================

@torch.no_grad()
def compute_batch(prompts):

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
    )

    enc = {
        k: v.to(model.device)
        for k, v in enc.items()
    }

    out = model(
        **enc,
        output_hidden_states=True,
        use_cache=False,
    )

    hs = out.hidden_states

    attn = enc["attention_mask"]

    last_idx = attn.sum(dim=1) - 1

    rows = []

    center_payload = {}

    topid_payload = {}

    for si in STATE_INDICES:

        lname = state_name(si)

        for k in TOPK_LIST:

            key = f"{lname}_k{k}"

            center_payload[key] = []

            topid_payload[key] = []

    for bi in range(len(prompts)):

        item = {}

        for si in STATE_INDICES:

            lname = state_name(si)

            h = hs[si][bi, last_idx[bi], :]

            h = h.to(
                model.lm_head.weight.dtype
            )

            logits = model.lm_head(h)

            top_vals, top_idx = torch.topk(
                logits,
                k=MAX_TOPK
            )

            vals = (
                top_vals
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            idxs = (
                top_idx
                .detach()
                .cpu()
                .numpy()
                .astype(np.int32)
            )

            for k in TOPK_LIST:

                vals_k = vals[:k]

                idx_k = idxs[:k]

                item[
                    f"{lname}_k{k}_top1"
                ] = float(vals_k[0])

                item[
                    f"{lname}_k{k}_kth"
                ] = float(vals_k[-1])

                item[
                    f"{lname}_k{k}_mean"
                ] = float(np.mean(vals_k))

                item[
                    f"{lname}_k{k}_std"
                ] = float(np.std(vals_k))

                item[
                    f"{lname}_k{k}_gap"
                ] = float(
                    vals_k[0] - vals_k[-1]
                )

                item[
                    f"{lname}_k{k}_entropy"
                ] = entropy_np(vals_k)

                center = np.mean(
                    W_norm[idx_k],
                    axis=0
                )

                center = (
                    center /
                    (np.linalg.norm(center)+1e-9)
                )

                key = f"{lname}_k{k}"

                center_payload[key].append(
                    center.astype(np.float32)
                )

                topid_payload[key].append(
                    idx_k.copy()
                )

        rows.append(item)

    return (
        rows,
        center_payload,
        topid_payload
    )

# ============================================================
# RUN ALL PROMPTS
# ============================================================

all_rows = []

center_store = {}

topid_store = {}

for si in STATE_INDICES:

    lname = state_name(si)

    for k in TOPK_LIST:

        key = f"{lname}_k{k}"

        center_store[key] = []

        topid_store[key] = []

for start in range(
    0,
    len(df),
    BATCH_SIZE
):

    sub = df.iloc[
        start:start+BATCH_SIZE
    ]

    rows, centers, topids = compute_batch(
        sub["prompt"].tolist()
    )

    all_rows.extend(rows)

    for key in center_store:

        center_store[key].extend(
            centers[key]
        )

        topid_store[key].extend(
            topids[key]
        )

    print(
        f"{min(start+BATCH_SIZE,len(df))}"
        f"/{len(df)}"
    )

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

feat_df = pd.DataFrame(all_rows)

df = pd.concat(
    [
        df.reset_index(drop=True),
        feat_df
    ],
    axis=1
)

for key in center_store:

    center_store[key] = np.stack(
        center_store[key],
        axis=0
    )

    topid_store[key] = np.stack(
        topid_store[key],
        axis=0
    )

print("Forward complete.")

# ============================================================
# CLASSIFICATION
# ============================================================

def group_classification(
    feature_key,
    target_col,
    group_col,
):

    X = center_store[
        feature_key
    ].astype(float)

    y = (
        df[target_col]
        .astype(str)
        .values
    )

    groups = (
        df[group_col]
        .astype(str)
        .values
    )

    n_splits = min(
        6,
        len(np.unique(groups))
    )

    cv = GroupKFold(
        n_splits=n_splits
    )

    preds = np.empty(
        len(y),
        dtype=object
    )

    for tr, te in cv.split(
        X,
        y,
        groups
    ):

        clf = Pipeline([

            (
                "scaler",
                StandardScaler()
            ),

            (
                "lr",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced"
                )
            )

        ])

        clf.fit(
            X[tr],
            y[tr]
        )

        preds[te] = clf.predict(
            X[te]
        )

    return {

        "feature_key":
            feature_key,

        "target":
            target_col,

        "group_col":
            group_col,

        "acc":
            float(
                accuracy_score(
                    y,
                    preds
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y,
                    preds,
                    average="macro"
                )
            )
    }

# ============================================================
# RUN CLASSIFIERS
# ============================================================

rows = []

for feature_key in center_store:

    layer = feature_key.split("_k")[0]

    topk = int(
        feature_key.split("_k")[-1]
    )

    # --------------------------
    # task
    # held-out topic
    # --------------------------

    r = group_classification(
        feature_key,
        "task_id",
        "topic_id"
    )

    r["layer"] = layer
    r["topk"] = topk

    rows.append(r)

    # --------------------------
    # topic
    # held-out task
    # --------------------------

    r = group_classification(
        feature_key,
        "topic_id",
        "task_id"
    )

    r["layer"] = layer
    r["topk"] = topk

    rows.append(r)

    # --------------------------
    # topic_task
    # held-out paraphrase
    # --------------------------

    r = group_classification(
        feature_key,
        "topic_task_id",
        "paraphrase_id"
    )

    r["layer"] = layer
    r["topk"] = topk

    rows.append(r)

class_df = pd.DataFrame(rows)

# ============================================================
# SILHOUETTE ANALYSIS
# ============================================================

sil_rows = []

for feature_key in center_store:

    layer = feature_key.split("_k")[0]

    topk = int(
        feature_key.split("_k")[-1]
    )

    X = center_store[
        feature_key
    ]

    try:

        sil_task = silhouette_score(
            X,
            df["task_id"],
            metric="cosine"
        )

    except:
        sil_task = np.nan

    try:

        sil_topic = silhouette_score(
            X,
            df["topic_id"],
            metric="cosine"
        )

    except:
        sil_topic = np.nan

    try:

        sil_topic_task = silhouette_score(
            X,
            df["topic_task_id"],
            metric="cosine"
        )

    except:
        sil_topic_task = np.nan

    sil_rows.append({

        "feature_key":
            feature_key,

        "layer":
            layer,

        "topk":
            topk,

        "sil_task":
            sil_task,

        "sil_topic":
            sil_topic,

        "sil_topic_task":
            sil_topic_task,
    })

sil_df = pd.DataFrame(sil_rows)

# ============================================================
# TRANSITION TABLE
# ============================================================

transition_rows = []

for feature_key in center_store:

    layer = feature_key.split("_k")[0]

    topk = int(
        feature_key.split("_k")[-1]
    )

    task_f1 = class_df[
        (class_df.feature_key==feature_key)
        &
        (class_df.target=="task_id")
    ].iloc[0]["macro_f1"]

    topic_f1 = class_df[
        (class_df.feature_key==feature_key)
        &
        (class_df.target=="topic_id")
    ].iloc[0]["macro_f1"]

    topic_task_f1 = class_df[
        (class_df.feature_key==feature_key)
        &
        (class_df.target=="topic_task_id")
    ].iloc[0]["macro_f1"]

    transition_rows.append({

        "feature_key":
            feature_key,

        "layer":
            layer,

        "topk":
            topk,

        "task_f1":
            float(task_f1),

        "topic_f1":
            float(topic_f1),

        "topic_task_f1":
            float(topic_task_f1),

        "topic_minus_task":
            float(
                topic_f1
                -
                task_f1
            ),

        "topic_task_minus_task":
            float(
                topic_task_f1
                -
                task_f1
            ),

        "abs_gap":
            float(
                abs(
                    topic_task_f1
                    -
                    task_f1
                )
            )
    })

transition_df = pd.DataFrame(
    transition_rows
)

# ============================================================
# LAYER ORDER
# ============================================================

def layer_order(x):

    if x == "emb":
        return -1

    return int(
        x.replace("L","")
    )

transition_df[
    "layer_order"
] = transition_df[
    "layer"
].apply(
    layer_order
)

transition_df = transition_df.sort_values(
    [
        "topk",
        "layer_order"
    ]
)

# ============================================================
# FIND TRANSITION
# ============================================================

best_by_topk = {}

for topk in TOPK_LIST:

    sub = transition_df[
        transition_df.topk == topk
    ]

    closest = sub.loc[
        sub.abs_gap.idxmin()
    ]

    best_task = sub.loc[
        sub.task_f1.idxmax()
    ]

    best_topic_task = sub.loc[
        sub.topic_task_f1.idxmax()
    ]

    crossover = sub[
        sub.topic_task_f1
        >=
        sub.task_f1
    ]

    if len(crossover) > 0:

        first_cross = crossover.iloc[0]

    else:

        first_cross = None

    best_by_topk[str(topk)] = {

        "closest_gap":
            closest.to_dict(),

        "best_task":
            best_task.to_dict(),

        "best_topic_task":
            best_topic_task.to_dict(),

        "first_cross":
            None
            if first_cross is None
            else
            first_cross.to_dict()
    }

# ============================================================
# GLOBAL RESULT
# ============================================================

global_closest = transition_df.loc[
    transition_df.abs_gap.idxmin()
]

global_best_task = transition_df.loc[
    transition_df.task_f1.idxmax()
]

global_best_topic_task = transition_df.loc[
    transition_df.topic_task_f1.idxmax()
]

summary = {

    "experiment":
        "OA-5B Task-to-Entity Basin Transition Audit",

    "n_prompts":
        int(len(df)),

    "n_topics":
        int(
            df.topic_id.nunique()
        ),

    "n_tasks":
        int(
            df.task_id.nunique()
        ),

    "best_by_topk":
        best_by_topk,

    "global_closest":
        global_closest.to_dict(),

    "global_best_task":
        global_best_task.to_dict(),

    "global_best_topic_task":
        global_best_topic_task.to_dict(),

    "hypothesis":

        "Task manifold dominates early. "
        "Topic-task basin emerges later. "
        "Transition expected around L18-L22."
}

# ============================================================
# SAVE
# ============================================================

df.to_csv(
    SAVE_DIR /
    "oa5b_dataset.csv",
    index=False,
    encoding="utf-8-sig"
)

class_df.to_csv(
    SAVE_DIR /
    "oa5b_layer_classification.csv",
    index=False,
    encoding="utf-8-sig"
)

sil_df.to_csv(
    SAVE_DIR /
    "oa5b_silhouette.csv",
    index=False,
    encoding="utf-8-sig"
)

transition_df.to_csv(
    SAVE_DIR /
    "oa5b_transition_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

with open(
    SAVE_DIR /
    "oa5b_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2
    )

print(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2
    )
)

print(
    "\nSaved:",
    SAVE_DIR.resolve()
)