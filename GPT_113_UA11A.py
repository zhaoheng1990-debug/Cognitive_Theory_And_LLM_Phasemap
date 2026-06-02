# ============================================================
# UA-11A: State Space Audit
#
# Goal:
#   Test whether the minimal state space:
#
#       X = (DeltaU, U_K, D_B)
#
#   explains Gen_E / phase / condition better than any single variable.
#
# Inputs:
#   ha1a_debug_outputs/deltaU_competition.csv
#
# Outputs:
#   ua11a_outputs/
#       ua11a_eval.csv
#       ua11a_feature_eval.csv
#       ua11a_state_space_summary.csv
#       ua11a_condition_summary.csv
#       ua11a_phase_summary.csv
#       ua11a_summary.json
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier

# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path("C:\Windows\System32\ha1a_debug_outputs\deltaU_competition.csv")
SAVE_DIR = Path("./ua11a_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

ENTRY_LAYERS = [20, 21, 22]
BASIN_LAYERS = [23, 24, 25, 26]
COMMIT_LAYER = 27

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_PATH)

required = [
    "DeltaU",
    "Gen_E",
    "condition",
    "phase_target",
] + [f"R_{l}" for l in ENTRY_LAYERS + BASIN_LAYERS + [COMMIT_LAYER]]

for c in required:
    if c not in df.columns:
        raise RuntimeError(f"Missing column: {c}")

df = df.copy()
df["Gen_E"] = df["Gen_E"].astype(int)

# ============================================================
# BUILD STATE VARIABLES
# ============================================================

# U_K: competition order/intensity from HA-1A-debug
df["U_K"] = df["DeltaU"]

# DeltaU_global: PC1 of raw R trajectory, oriented so clean/positive higher than negative if possible.
R_cols = [f"R_{l}" for l in range(20, 26)]
X_R = df[R_cols].values.astype(float)

pca = PCA(n_components=3, random_state=SEED)
Z = pca.fit_transform(X_R)

df["DeltaU_raw_PC1"] = Z[:, 0]
df["DeltaU_PC2"] = Z[:, 1]
df["DeltaU_PC3"] = Z[:, 2]

phase_mean_raw = df.groupby("phase_target")["DeltaU_raw_PC1"].mean().to_dict()
if phase_mean_raw.get("positive", 0) < phase_mean_raw.get("negative", 0):
    df["DeltaU_global"] = -df["DeltaU_raw_PC1"]
    pc1_loading = (-pca.components_[0]).tolist()
else:
    df["DeltaU_global"] = df["DeltaU_raw_PC1"]
    pc1_loading = pca.components_[0].tolist()

# D_B: boundary distance candidates
df["S_entry_mean_R20_22"] = df[[f"R_{l}" for l in ENTRY_LAYERS]].mean(axis=1)
df["S_basin_mean_R23_26"] = df[[f"R_{l}" for l in BASIN_LAYERS]].mean(axis=1)
df["S_commit_R27"] = df[f"R_{COMMIT_LAYER}"]

df["D_entry_minabs_R20_22"] = df[[f"R_{l}" for l in ENTRY_LAYERS]].abs().min(axis=1)
df["D_basin_minabs_R23_26"] = df[[f"R_{l}" for l in BASIN_LAYERS]].abs().min(axis=1)
df["D_basin_absmean_R23_26"] = df[[f"R_{l}" for l in BASIN_LAYERS]].mean(axis=1).abs()
df["D_commit_abs_R27"] = df[f"R_{COMMIT_LAYER}"].abs()

# Main chosen D_B: final boundary distance, plus non-final control.
df["D_B"] = df["D_commit_abs_R27"]
df["D_B_nonfinal"] = df["D_basin_minabs_R23_26"]

# Mechanism/mixed labels
df["is_ambiguous"] = (df["condition"] == "ambiguous_branch").astype(int)
df["is_mixed_condition"] = df["condition"].isin(
    ["ambiguous_branch", "weak_distractor_v2"]
).astype(int)

df.to_csv(SAVE_DIR / "ua11a_eval.csv", index=False)

# ============================================================
# EVALUATION HELPERS
# ============================================================

def binary_cv(df, features, target):
    X = df[features].values.astype(float)
    y = df[target].values.astype(int)

    if len(np.unique(y)) < 2:
        return None

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pred = np.zeros(len(df))

    for tr, te in skf.split(X, y):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=3000)),
        ])
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict_proba(X[te])[:, 1]

    label = (pred >= 0.5).astype(int)

    return {
        "auc": float(roc_auc_score(y, pred)),
        "acc": float(accuracy_score(y, label)),
        "f1": float(f1_score(y, label, zero_division=0)),
    }

def multiclass_cv(df, features, target):
    X = df[features].values.astype(float)
    y = df[target].astype(str).values

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    pred = np.empty(len(df), dtype=object)

    for tr, te in skf.split(X, y):

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                max_iter=5000
            )),
        ])

        clf.fit(X[tr], y[tr])

        pred[te] = clf.predict(X[te])

    return {
        "acc": float(accuracy_score(y, pred)),
        "macro_f1": float(
            f1_score(
                y,
                pred,
                average="macro",
                zero_division=0,
            )
        ),
    }
# ============================================================
# FEATURE SETS
# ============================================================

FEATURE_SETS = {
    # Single variables
    "DeltaU_global_only": ["DeltaU_global"],
    "U_K_only": ["U_K"],
    "D_B_only": ["D_B"],
    "D_B_nonfinal_only": ["D_B_nonfinal"],

    # Pairs
    "DeltaU_plus_UK": ["DeltaU_global", "U_K"],
    "DeltaU_plus_DB": ["DeltaU_global", "D_B"],
    "UK_plus_DB": ["U_K", "D_B"],

    # Proposed minimal state
    "STATE_DeltaU_UK_DB": ["DeltaU_global", "U_K", "D_B"],

    # Non-final state, stricter
    "STATE_nonfinal": ["DeltaU_global", "U_K", "D_B_nonfinal"],

    # Oracle signed readout baseline
    "signed_all": [
        "S_entry_mean_R20_22",
        "S_basin_mean_R23_26",
        "S_commit_R27",
    ],

    # Raw trajectory baseline
    "raw_R_20_25": R_cols,

    # Full state + raw control
    "STATE_plus_rawR": ["DeltaU_global", "U_K", "D_B"] + R_cols,
}

# ============================================================
# RUN EVALS
# ============================================================

rows = []

binary_targets = ["Gen_E", "is_ambiguous", "is_mixed_condition"]
multi_targets = ["phase_target", "condition"]

for target in binary_targets:
    for name, feats in FEATURE_SETS.items():
        res = binary_cv(df, feats, target)
        if res is None:
            continue
        rows.append({
            "target": target,
            "task": "binary",
            "feature_set": name,
            "features": ",".join(feats),
            **res,
        })

for target in multi_targets:
    for name, feats in FEATURE_SETS.items():
        res = multiclass_cv(df, feats, target)
        rows.append({
            "target": target,
            "task": "multiclass",
            "feature_set": name,
            "features": ",".join(feats),
            "auc": None,
            "acc": res["acc"],
            "f1": res["macro_f1"],
        })

feature_eval = pd.DataFrame(rows)
feature_eval.to_csv(SAVE_DIR / "ua11a_feature_eval.csv", index=False)

# ============================================================
# STATE SPACE GEOMETRY
# ============================================================

state_cols = ["DeltaU_global", "U_K", "D_B"]
state_nonfinal_cols = ["DeltaU_global", "U_K", "D_B_nonfinal"]

def centroid_table(group_col, cols):
    out = []
    for key, sub in df.groupby(group_col):
        row = {group_col: key, "n": int(len(sub))}
        for c in cols:
            row[f"{c}_mean"] = float(sub[c].mean())
            row[f"{c}_std"] = float(sub[c].std())
        out.append(row)
    return pd.DataFrame(out)

condition_summary = centroid_table("condition", state_cols + state_nonfinal_cols)
phase_summary = centroid_table("phase_target", state_cols + state_nonfinal_cols)

condition_summary["GenE_rate"] = df.groupby("condition")["Gen_E"].mean().values
phase_summary["GenE_rate"] = df.groupby("phase_target")["Gen_E"].mean().values

condition_summary.to_csv(SAVE_DIR / "ua11a_condition_summary.csv", index=False)
phase_summary.to_csv(SAVE_DIR / "ua11a_phase_summary.csv", index=False)

# Correlation matrix
corr_cols = [
    "DeltaU_global",
    "U_K",
    "D_B",
    "D_B_nonfinal",
    "S_commit_R27",
    "Gen_E",
    "is_ambiguous",
]
corr_matrix = df[corr_cols].corr(numeric_only=True)
corr_matrix.to_csv(SAVE_DIR / "ua11a_state_corr.csv")

# ============================================================
# STATE SPACE PASS/FAIL SUMMARY
# ============================================================

def best_for(target):
    sub = feature_eval[feature_eval["target"] == target].copy()
    if target in binary_targets:
        sub = sub.sort_values("auc", ascending=False)
    else:
        sub = sub.sort_values("f1", ascending=False)
    return sub.head(8).to_dict(orient="records")

def get_metric(target, feature_set, metric):
    sub = feature_eval[
        (feature_eval["target"] == target)
        & (feature_eval["feature_set"] == feature_set)
    ]
    if len(sub) == 0:
        return None
    val = sub.iloc[0][metric]
    return None if pd.isna(val) else float(val)

summary_state = {
    "pca_R20_25_explained_variance": pca.explained_variance_ratio_.tolist(),
    "pca_R20_25_pc1_loading_oriented": pc1_loading,

    "GenE_auc": {
        "DeltaU_global_only": get_metric("Gen_E", "DeltaU_global_only", "auc"),
        "U_K_only": get_metric("Gen_E", "U_K_only", "auc"),
        "D_B_only": get_metric("Gen_E", "D_B_only", "auc"),
        "D_B_nonfinal_only": get_metric("Gen_E", "D_B_nonfinal_only", "auc"),
        "STATE": get_metric("Gen_E", "STATE_DeltaU_UK_DB", "auc"),
        "STATE_nonfinal": get_metric("Gen_E", "STATE_nonfinal", "auc"),
        "raw_R_20_25": get_metric("Gen_E", "raw_R_20_25", "auc"),
        "signed_all": get_metric("Gen_E", "signed_all", "auc"),
    },

    "ambiguous_auc": {
        "DeltaU_global_only": get_metric("is_ambiguous", "DeltaU_global_only", "auc"),
        "U_K_only": get_metric("is_ambiguous", "U_K_only", "auc"),
        "D_B_only": get_metric("is_ambiguous", "D_B_only", "auc"),
        "D_B_nonfinal_only": get_metric("is_ambiguous", "D_B_nonfinal_only", "auc"),
        "STATE": get_metric("is_ambiguous", "STATE_DeltaU_UK_DB", "auc"),
        "STATE_nonfinal": get_metric("is_ambiguous", "STATE_nonfinal", "auc"),
        "raw_R_20_25": get_metric("is_ambiguous", "raw_R_20_25", "auc"),
    },

    "phase_macro_f1": {
        "DeltaU_global_only": get_metric("phase_target", "DeltaU_global_only", "f1"),
        "U_K_only": get_metric("phase_target", "U_K_only", "f1"),
        "D_B_only": get_metric("phase_target", "D_B_only", "f1"),
        "D_B_nonfinal_only": get_metric("phase_target", "D_B_nonfinal_only", "f1"),
        "STATE": get_metric("phase_target", "STATE_DeltaU_UK_DB", "f1"),
        "STATE_nonfinal": get_metric("phase_target", "STATE_nonfinal", "f1"),
        "raw_R_20_25": get_metric("phase_target", "raw_R_20_25", "f1"),
    },
}

with open(SAVE_DIR / "ua11a_state_space_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_state, f, ensure_ascii=False, indent=2)

# ============================================================
# FINAL SUMMARY
# ============================================================

summary = {
    "experiment": "UA-11A State Space Audit",
    "input_path": str(INPUT_PATH),
    "n_rows": int(len(df)),
    "state_definition": {
        "DeltaU_global": "PC1 of R20:25 trajectory",
        "U_K": "competition order/intensity from HA-1A-debug DeltaU",
        "D_B": "commit boundary distance |R27|",
        "D_B_nonfinal": "basin boundary distance min |R23:26|"
    },
    "state_space_summary": summary_state,
    "top_features": {
        "Gen_E": best_for("Gen_E"),
        "is_ambiguous": best_for("is_ambiguous"),
        "is_mixed_condition": best_for("is_mixed_condition"),
        "phase_target": best_for("phase_target"),
        "condition": best_for("condition"),
    },
    "condition_summary": condition_summary.to_dict(orient="records"),
    "phase_summary": phase_summary.to_dict(orient="records"),
    "correlation_matrix": corr_matrix.to_dict(),
    "interpretation_rules": {
        "PASS_minimal_state": [
            "STATE_DeltaU_UK_DB beats all single variables across Gen_E, ambiguous, and phase tasks.",
            "STATE_nonfinal remains competitive, showing not only final-layer readout.",
            "State centroids separate positive / critical / negative phases."
        ],
        "PASS_partial_state": [
            "STATE improves over U_K and DeltaU, but D_B or signed final still dominates Gen_E.",
            "This means state space is useful, but D_B is near-output and not fully causal."
        ],
        "FAIL_minimal_state": [
            "raw_R_20_25 or signed_all dominates all state variables.",
            "STATE does not improve over single variables.",
            "The proposed 3D state is only a lossy summary."
        ]
    }
}

with open(SAVE_DIR / "ua11a_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nSaved to: {SAVE_DIR.resolve()}")