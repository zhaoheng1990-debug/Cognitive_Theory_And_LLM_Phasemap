# ============================================================
# OA-4A.2 Lite
#
# Surface Invariance Audit
#
# Test:
#
#   Surface wording
#       vs
#   Constraint structure
#
# Which determines DeltaR20:25 ?
#
# ============================================================

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from sklearn.metrics import (
    pairwise_distances,
)

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

SAVE_DIR = Path("./oa4a2_lite_outputs")
SAVE_DIR.mkdir(exist_ok=True)

TRACK_LAYERS = [20,21,22,23,24,25]

BATCH_SIZE = 8

SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# LOAD
# ============================================================

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

# ============================================================
# HELPERS
# ============================================================

def encode_no_special(text):
    return tokenizer(
        text,
        add_special_tokens=False
    )["input_ids"]

def single_token_id(text):

    candidates = [
        text,
        " " + text,
        text.lower(),
        " " + text.lower(),
    ]

    for c in candidates:

        ids = encode_no_special(c)

        if len(ids) == 1:
            return ids[0]

    return None

# ============================================================
# DATA
# ============================================================

GRAPHS = [
    ("Paris","France","Europe","Asia"),
    ("Berlin","Germany","Europe","Asia"),
    ("Tokyo","Japan","Asia","Europe"),
    ("Beijing","China","Asia","Europe"),
    ("doctor","hospital","medicine","law"),
    ("judge","court","law","medicine"),
    ("teacher","school","education","finance"),
    ("pilot","airport","aviation","medicine"),
    ("dog","animal","life","machine"),
    ("cat","animal","life","machine"),
    ("river","water","nature","finance"),
    ("sun","star","space","court"),
]

CONDITIONS = [
    "clean",
    "weak_distractor",
    "ambiguous_branch",
    "direct_conflict",
    "exception_override",
    "rule_update",
    "meta_override",
]

# ============================================================
# 5 SURFACES
# ============================================================

def prompt_surface(
    a,b,c,e,
    condition,
    surface_id
):

    if surface_id == 0:

        style = lambda s: s

    elif surface_id == 1:

        style = lambda s: (
            "Please consider:\n" + s
        )

    elif surface_id == 2:

        style = lambda s: (
            "Given the following facts:\n" + s
        )

    elif surface_id == 3:

        style = lambda s: (
            "Knowledge base:\n" + s
        )

    else:

        style = lambda s: (
            "Reason carefully.\n" + s
        )

    if condition == "clean":

        core = f"""
Fact 1: {a} belongs to {b}.
Fact 2: {b} is associated with {c}.
Question:
Which label is {a} associated with?
Answer: {c} or {e}
"""

    elif condition == "weak_distractor":

        core = f"""
Fact 1: {a} belongs to {b}.
Fact 2: {b} is associated with {c}.
Unrelated note: another object mentions {e}.
Question:
Which label is {a} associated with?
Answer: {c} or {e}
"""

    elif condition == "ambiguous_branch":

        core = f"""
Fact 1: {a} belongs to {b}.
Fact 2: {b} is associated with {c}.
Some descriptions also connect {a} with {e}.
Question:
Which label is {a} associated with?
Answer: {c} or {e}
"""

    elif condition == "direct_conflict":

        core = f"""
Fact 1: {a} belongs to {b}.
Fact 2: {b} is associated with {c}.
Fact 3: {a} is associated with {e}.
Question:
Which label is {a} associated with?
Answer: {c} or {e}
"""

    elif condition == "exception_override":

        core = f"""
General rule:
items belonging to {b}
are associated with {c}.

Exception:
{a} is associated with {e}.

Question:
Which label is {a} associated with?
Answer: {c} or {e}
"""

    elif condition == "rule_update":

        core = f"""
Old rule:
{b} -> {c}

New rule:
{b} -> {e}

{a} belongs to {b}

Question:
Answer: {c} or {e}
"""

    elif condition == "meta_override":

        core = f"""
Ignore old rules.

Use only:

{b} -> {e}

{a} belongs to {b}

Question:
Answer: {c} or {e}
"""

    return style(core)

# ============================================================
# BUILD DATASET
# ============================================================

rows = []

gid = 0

for graph_id,(a,b,c,e) in enumerate(GRAPHS):

    for condition in CONDITIONS:

        for surface_id in range(5):

            rows.append({
                "graph_id": graph_id,
                "condition": condition,
                "surface_id": surface_id,
                "A": a,
                "B": b,
                "C": c,
                "E": e,
                "prompt":
                    prompt_surface(
                        a,b,c,e,
                        condition,
                        surface_id
                    )
            })

            gid += 1

df = pd.DataFrame(rows)

print("Rows:",len(df))

# ============================================================
# FORWARD
# ============================================================

@torch.no_grad()
def compute_batch(
    prompts,
    clean_targets,
    conflict_targets,
):

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )

    enc = {
        k:v.to(model.device)
        for k,v in enc.items()
    }

    out = model(
        **enc,
        output_hidden_states=True,
        use_cache=False,
    )

    hs = out.hidden_states

    attn = enc["attention_mask"]

    last_idx = attn.sum(dim=1)-1

    rows = []

    for bi in range(len(prompts)):

        c_id = single_token_id(
            clean_targets[bi]
        )

        e_id = single_token_id(
            conflict_targets[bi]
        )

        item = {}

        for layer in TRACK_LAYERS:

            h = hs[layer+1][
                bi,
                last_idx[bi],
                :
            ]

            logits = model.lm_head(h)

            r = float(
                logits[c_id]
                -
                logits[e_id]
            )

            item[f"R{layer}"] = r

        rows.append(item)

    return rows

all_rows = []

for start in range(
    0,
    len(df),
    BATCH_SIZE
):

    sub = df.iloc[
        start:start+BATCH_SIZE
    ]

    batch = compute_batch(
        sub["prompt"].tolist(),
        sub["C"].tolist(),
        sub["E"].tolist(),
    )

    all_rows.extend(batch)

    print(
        f"{min(start+BATCH_SIZE,len(df))}/{len(df)}"
    )

R_df = pd.DataFrame(all_rows)

df = pd.concat(
    [df.reset_index(drop=True),R_df],
    axis=1
)

# ============================================================
# DELTA R
# ============================================================

for layer in TRACK_LAYERS:

    base = (
        df[df["condition"]=="clean"]
        [["graph_id","surface_id",f"R{layer}"]]
        .rename(
            columns={
                f"R{layer}":
                f"R{layer}_clean"
            }
        )
    )

    df = df.merge(
        base,
        on=[
            "graph_id",
            "surface_id"
        ],
        how="left"
    )

    df[f"dR{layer}"] = (
        df[f"R{layer}"]
        -
        df[f"R{layer}_clean"]
    )

dR_cols = [
    f"dR{x}"
    for x in TRACK_LAYERS
]

# ============================================================
# AUDIT 1
# Surface variance
# ============================================================

surface_var = (
    df.groupby(
        ["graph_id","condition"]
    )[dR_cols]
    .var()
    .mean()
    .mean()
)

condition_var = (
    df.groupby(
        ["graph_id"]
    )[dR_cols]
    .var()
    .mean()
    .mean()
)

# ============================================================
# AUDIT 2
# Trajectory consistency
# ============================================================

cond_proto = (
    df.groupby("condition")[dR_cols]
    .mean()
)

within = []

between = []

for _,row in df.iterrows():

    vec = row[dR_cols].values

    own = cond_proto.loc[
        row["condition"]
    ].values

    within.append(
        np.linalg.norm(
            vec-own
        )
    )

    for cond in cond_proto.index:

        if cond == row["condition"]:
            continue

        other = cond_proto.loc[
            cond
        ].values

        between.append(
            np.linalg.norm(
                vec-other
            )
        )

summary = {

    "surface_variance":
        float(surface_var),

    "condition_variance":
        float(condition_var),

    "within_condition_distance":
        float(np.mean(within)),

    "between_condition_distance":
        float(np.mean(between)),

    "ratio":
        float(
            np.mean(within)
            /
            np.mean(between)
        )
}

print(
    json.dumps(
        summary,
        indent=2
    )
)

with open(
    SAVE_DIR /
    "oa4a2_lite_summary.json",
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )

df.to_csv(
    SAVE_DIR /
    "oa4a2_lite_dataset.csv",
    index=False
)

print("\nDONE.")