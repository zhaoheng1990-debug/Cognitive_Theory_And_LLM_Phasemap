import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    r2_score,
    roc_auc_score,
    accuracy_score,
    f1_score,
)

# ============================================================
# CONFIG
# ============================================================

CSV_PATH = r"C:\Windows\System32\oa4a_prompt_initial_condition_outputs\oa4a_dataset.csv"

SAVE_DIR = Path("./oa4a1_outputs")
SAVE_DIR.mkdir(exist_ok=True)

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(CSV_PATH)

print("Rows:", len(df))
print("Cols:", len(df.columns))

# ============================================================
# PROMPT FEATURE EXTRACTION
# ============================================================

AMBIG_WORDS = [
    "some",
    "maybe",
    "possibly",
    "ambiguous",
    "might",
]

OVERRIDE_WORDS = [
    "exception",
    "special case",
    "ignore",
    "latest rule",
    "override",
]

UPDATE_WORDS = [
    "update",
    "new rule",
    "old rule",
    "now associated",
]

NEGATION_WORDS = [
    "not",
    "invalid",
    "ignore",
]

def extract_prompt_features(prompt):

    p = str(prompt).lower()

    fact_count = len(re.findall(r"fact\s*\d*", p))

    rule_count = (
        p.count("rule")
        + p.count("general rule")
    )

    conflict_count = (
        p.count("exception")
        + p.count("special case")
        + p.count("conflict")
    )

    ambiguity_count = sum(
        p.count(w)
        for w in AMBIG_WORDS
    )

    update_count = sum(
        p.count(w)
        for w in UPDATE_WORDS
    )

    override_count = sum(
        p.count(w)
        for w in OVERRIDE_WORDS
    )

    negation_count = sum(
        p.count(w)
        for w in NEGATION_WORDS
    )

    sentence_count = len(
        re.findall(r"[.!?]", p)
    )

    return {
        "pf_fact_count": fact_count,
        "pf_rule_count": rule_count,
        "pf_conflict_count": conflict_count,
        "pf_ambiguity_count": ambiguity_count,
        "pf_update_count": update_count,
        "pf_override_count": override_count,
        "pf_negation_count": negation_count,
        "pf_sentence_count": sentence_count,
    }

prompt_feat_df = pd.DataFrame(
    [extract_prompt_features(x) for x in df["prompt"]]
)

df = pd.concat(
    [df.reset_index(drop=True), prompt_feat_df],
    axis=1
)

# ============================================================
# AUTO xi0
# ============================================================

df["R0_auto"] = (
    df["pf_rule_count"]
    - df["pf_conflict_count"]
    - df["pf_override_count"]
)

df["B0_auto"] = (
    df["pf_conflict_count"]
    + df["pf_negation_count"]
    + df["pf_update_count"]
)

df["F0_auto"] = (
    df["pf_ambiguity_count"]
    + 0.5 * df["pf_sentence_count"]
)

df["xi0_auto_energy"] = np.sqrt(
    df["R0_auto"]**2
    +
    df["B0_auto"]**2
    +
    df["F0_auto"]**2
)

# ============================================================
# FEATURE SETS
# ============================================================

PROMPT_FEATURES = [
    "pf_fact_count",
    "pf_rule_count",
    "pf_conflict_count",
    "pf_ambiguity_count",
    "pf_update_count",
    "pf_override_count",
    "pf_negation_count",
    "pf_sentence_count",
]

AUTO_XI0 = [
    "R0_auto",
    "B0_auto",
    "F0_auto",
    "xi0_auto_energy",
]

MANUAL_XI0 = [
    "R0_prompt_bias",
    "B0_boundary_pressure",
    "F0_freedom",
    "xi0_energy",
]

W_FEATURES = [
    c
    for c in df.columns
    if c.startswith("W_")
]

# ============================================================
# GROUP REGRESSION
# ============================================================

def group_regression(
    feature_cols,
    target="deltaR_l2"
):

    X = df[feature_cols].values
    y = df[target].values

    groups = df["base_graph_id"].values

    cv = GroupKFold(n_splits=5)

    pred = np.zeros(len(y))

    for tr, te in cv.split(
        X,
        y,
        groups
    ):
        reg = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0))
        ])

        reg.fit(
            X[tr],
            y[tr]
        )

        pred[te] = reg.predict(
            X[te]
        )

    return {
        "r2":
            float(
                r2_score(y, pred)
            ),
        "corr":
            float(
                np.corrcoef(
                    y,
                    pred
                )[0,1]
            )
    }

# ============================================================
# GROUP CLASSIFICATION
# ============================================================

def group_binary(
    feature_cols,
    target_col
):

    X = df[feature_cols].values

    y = df[target_col].values

    groups = df["base_graph_id"].values

    cv = GroupKFold(
        n_splits=5
    )

    prob = np.zeros(len(y))

    for tr, te in cv.split(
        X,
        y,
        groups
    ):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                max_iter=5000
            ))
        ])

        clf.fit(
            X[tr],
            y[tr]
        )

        prob[te] = clf.predict_proba(
            X[te]
        )[:,1]

    pred = (prob > 0.5).astype(int)

    return {
        "auc":
            float(
                roc_auc_score(
                    y,
                    prob
                )
            ),
        "acc":
            float(
                accuracy_score(
                    y,
                    pred
                )
            ),
        "f1":
            float(
                f1_score(
                    y,
                    pred
                )
            )
    }

# ============================================================
# RUN
# ============================================================

summary = {

    "prompt_features_to_deltaR":

        group_regression(
            PROMPT_FEATURES,
            "deltaR_l2"
        ),

    "auto_xi0_to_deltaR":

        group_regression(
            AUTO_XI0,
            "deltaR_l2"
        ),

    "manual_xi0_to_deltaR":

        group_regression(
            MANUAL_XI0,
            "deltaR_l2"
        ),

    "W_to_deltaR":

        group_regression(
            W_FEATURES,
            "deltaR_l2"
        ),

    "prompt_features_to_closure":

        group_binary(
            PROMPT_FEATURES,
            "is_closure"
        ),

    "auto_xi0_to_closure":

        group_binary(
            AUTO_XI0,
            "is_closure"
        ),

    "manual_xi0_to_closure":

        group_binary(
            MANUAL_XI0,
            "is_closure"
        ),

    "W_to_closure":

        group_binary(
            W_FEATURES,
            "is_closure"
        )
}

print(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False
    )
)

with open(
    SAVE_DIR /
    "oa4a1_summary.json",
    "w",
    encoding="utf8"
) as f:
    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False
    )

df.to_csv(
    SAVE_DIR /
    "oa4a1_dataset.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved.")