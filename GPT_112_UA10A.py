# ============================================================
# UA-10A: Competition -> Boundary Compression -> Hallucination
#
# Goal:
#   Test the mediation chain:
#
#       U_K  ->  D_B  ->  Gen_E
#
#   where:
#       U_K = competition intensity / competition order parameter
#       D_B = distance to basin boundary
#       Gen_E = generated conflict / hallucination answer
#
# Inputs:
#   ha1a_debug_outputs/deltaU_competition.csv
#
# Outputs:
#   ua10a_outputs/
#       ua10a_eval.csv
#       ua10a_feature_eval.csv
#       ua10a_mediation_summary.csv
#       ua10a_condition_summary.csv
#       ua10a_phase_summary.csv
#       ua10a_summary.json
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, r2_score
from sklearn.model_selection import StratifiedKFold, KFold

# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path("C:\Windows\System32\ha1a_debug_outputs\deltaU_competition.csv")
SAVE_DIR = Path("./ua10a_outputs")
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
    "DeltaU", "Gen_E", "condition", "phase_target"
] + [f"R_{l}" for l in ENTRY_LAYERS + BASIN_LAYERS + [COMMIT_LAYER]]

for c in required:
    if c not in df.columns:
        raise RuntimeError(f"Missing column: {c}")

df = df.copy()
df["Gen_E"] = df["Gen_E"].astype(int)

# ============================================================
# BUILD VARIABLES
# ============================================================

df["U_K"] = df["DeltaU"]

df["S_entry_mean_R20_22"] = df[[f"R_{l}" for l in ENTRY_LAYERS]].mean(axis=1)
df["S_basin_mean_R23_26"] = df[[f"R_{l}" for l in BASIN_LAYERS]].mean(axis=1)
df["S_commit_R27"] = df[f"R_{COMMIT_LAYER}"]

df["D_entry_minabs_R20_22"] = df[[f"R_{l}" for l in ENTRY_LAYERS]].abs().min(axis=1)
df["D_basin_minabs_R23_26"] = df[[f"R_{l}" for l in BASIN_LAYERS]].abs().min(axis=1)
df["D_basin_absmean_R23_26"] = df[[f"R_{l}" for l in BASIN_LAYERS]].mean(axis=1).abs()
df["D_commit_abs_R27"] = df[f"R_{COMMIT_LAYER}"].abs()

# Boundary compression = closer to boundary.
# Larger B means smaller distance.
eps = 1e-6
df["B_entry_close"] = 1.0 / (df["D_entry_minabs_R20_22"] + eps)
df["B_basin_close"] = 1.0 / (df["D_basin_minabs_R23_26"] + eps)
df["B_commit_close"] = 1.0 / (df["D_commit_abs_R27"] + eps)

# Main mediator candidate
MAIN_D = "D_commit_abs_R27"
MAIN_B = "B_commit_close"

# Non-final mediator candidate
NONFINAL_D = "D_basin_minabs_R23_26"
NONFINAL_B = "B_basin_close"

# ============================================================
# HELPERS
# ============================================================

def cv_logit_auc(df, features, target="Gen_E"):
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
        "pred": pred,
    }

def cv_linear_r2_corr(df, x_features, y_col):
    X = df[x_features].values.astype(float)
    y = df[y_col].values.astype(float)

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    pred = np.zeros(len(df))

    for tr, te in kf.split(X):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LinearRegression()),
        ])
        pipe.fit(X[tr], y[tr])
        pred[te] = pipe.predict(X[te])

    r2 = r2_score(y, pred)
    corr = np.corrcoef(y, pred)[0, 1]

    return {
        "r2": float(r2),
        "corr": float(corr),
        "pred": pred,
    }

def corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])

# ============================================================
# STEP 1: U_K -> D_B / B_CLOSE
# ============================================================

mediators = [
    "D_entry_minabs_R20_22",
    "D_basin_minabs_R23_26",
    "D_basin_absmean_R23_26",
    "D_commit_abs_R27",
    "B_entry_close",
    "B_basin_close",
    "B_commit_close",
]

uk_to_mediator_rows = []

for m in mediators:
    res = cv_linear_r2_corr(df, ["U_K"], m)
    uk_to_mediator_rows.append({
        "path": f"U_K -> {m}",
        "mediator": m,
        "r2": res["r2"],
        "corr": res["corr"],
        "raw_corr_UK_mediator": corr(df["U_K"], df[m]),
    })
    df[f"Pred_{m}_from_UK"] = res["pred"]

uk_to_mediator_df = pd.DataFrame(uk_to_mediator_rows)
uk_to_mediator_df.to_csv(SAVE_DIR / "ua10a_uk_to_mediator.csv", index=False)

# ============================================================
# STEP 2: D_B / B_CLOSE -> Gen_E
# ============================================================

feature_sets = {
    "U_K_only": ["U_K"],

    "D_entry_only": ["D_entry_minabs_R20_22"],
    "D_basin_only": ["D_basin_minabs_R23_26"],
    "D_basin_absmean_only": ["D_basin_absmean_R23_26"],
    "D_commit_only": ["D_commit_abs_R27"],

    "B_entry_only": ["B_entry_close"],
    "B_basin_only": ["B_basin_close"],
    "B_commit_only": ["B_commit_close"],

    "U_K_plus_D_basin": ["U_K", "D_basin_minabs_R23_26"],
    "U_K_plus_D_commit": ["U_K", "D_commit_abs_R27"],
    "U_K_plus_B_basin": ["U_K", "B_basin_close"],
    "U_K_plus_B_commit": ["U_K", "B_commit_close"],

    "D_entry_basin_commit": [
        "D_entry_minabs_R20_22",
        "D_basin_minabs_R23_26",
        "D_commit_abs_R27",
    ],

    "B_entry_basin_commit": [
        "B_entry_close",
        "B_basin_close",
        "B_commit_close",
    ],

    "signed_R_all": [
        "S_entry_mean_R20_22",
        "S_basin_mean_R23_26",
        "S_commit_R27",
    ],
}

feature_rows = []

for name, feats in feature_sets.items():
    res = cv_logit_auc(df, feats)
    if res is None:
        continue

    feature_rows.append({
        "feature_set": name,
        "features": ",".join(feats),
        "auc": res["auc"],
        "acc": res["acc"],
        "f1": res["f1"],
    })
    df[f"Pred_GenE_{name}"] = res["pred"]

feature_eval = pd.DataFrame(feature_rows).sort_values("auc", ascending=False)
feature_eval.to_csv(SAVE_DIR / "ua10a_feature_eval.csv", index=False)

# ============================================================
# STEP 3: MEDIATION-LIKE TEST
# ============================================================

# We evaluate whether adding boundary mediator improves over U_K.
# Not formal causal mediation, but a predictive mediation audit.

def get_auc(feature_set_name):
    row = feature_eval[feature_eval["feature_set"] == feature_set_name]
    if len(row) == 0:
        return None
    return float(row.iloc[0]["auc"])

mediation_summary = {
    "U_K_only_auc": get_auc("U_K_only"),
    "D_basin_only_auc": get_auc("D_basin_only"),
    "D_commit_only_auc": get_auc("D_commit_only"),
    "B_basin_only_auc": get_auc("B_basin_only"),
    "B_commit_only_auc": get_auc("B_commit_only"),
    "U_K_plus_D_basin_auc": get_auc("U_K_plus_D_basin"),
    "U_K_plus_D_commit_auc": get_auc("U_K_plus_D_commit"),
    "U_K_plus_B_basin_auc": get_auc("U_K_plus_B_basin"),
    "U_K_plus_B_commit_auc": get_auc("U_K_plus_B_commit"),
}

for k in [
    "D_basin_only_auc",
    "D_commit_only_auc",
    "B_basin_only_auc",
    "B_commit_only_auc",
    "U_K_plus_D_basin_auc",
    "U_K_plus_D_commit_auc",
    "U_K_plus_B_basin_auc",
    "U_K_plus_B_commit_auc",
]:
    if mediation_summary[k] is not None and mediation_summary["U_K_only_auc"] is not None:
        mediation_summary[f"{k}_minus_UK"] = (
            mediation_summary[k] - mediation_summary["U_K_only_auc"]
        )

# Stepwise chain score:
# U_K should predict boundary compression;
# boundary compression should predict Gen_E;
# U_K + boundary should improve over U_K alone.

chain_tests = {
    "UK_predicts_B_basin_corr": corr(df["U_K"], df["B_basin_close"]),
    "UK_predicts_B_commit_corr": corr(df["U_K"], df["B_commit_close"]),
    "UK_predicts_D_basin_corr": corr(df["U_K"], df["D_basin_minabs_R23_26"]),
    "UK_predicts_D_commit_corr": corr(df["U_K"], df["D_commit_abs_R27"]),

    "B_basin_beats_UK": (
        mediation_summary["B_basin_only_auc"] is not None
        and mediation_summary["U_K_only_auc"] is not None
        and mediation_summary["B_basin_only_auc"] > mediation_summary["U_K_only_auc"]
    ),
    "B_commit_beats_UK": (
        mediation_summary["B_commit_only_auc"] is not None
        and mediation_summary["U_K_only_auc"] is not None
        and mediation_summary["B_commit_only_auc"] > mediation_summary["U_K_only_auc"]
    ),
    "UK_plus_B_basin_improves": (
        mediation_summary["U_K_plus_B_basin_auc"] is not None
        and mediation_summary["U_K_only_auc"] is not None
        and mediation_summary["U_K_plus_B_basin_auc"] > mediation_summary["U_K_only_auc"]
    ),
    "UK_plus_B_commit_improves": (
        mediation_summary["U_K_plus_B_commit_auc"] is not None
        and mediation_summary["U_K_only_auc"] is not None
        and mediation_summary["U_K_plus_B_commit_auc"] > mediation_summary["U_K_only_auc"]
    ),
}

mediation_summary["chain_tests"] = chain_tests

with open(SAVE_DIR / "ua10a_mediation_summary.json", "w", encoding="utf-8") as f:
    json.dump(mediation_summary, f, ensure_ascii=False, indent=2)

# ============================================================
# CONDITION / PHASE SUMMARY
# ============================================================

condition_summary = df.groupby("condition").agg(
    n=("Gen_E", "count"),
    GenE_rate=("Gen_E", "mean"),

    U_K_mean=("U_K", "mean"),
    U_K_std=("U_K", "std"),

    D_entry_mean=("D_entry_minabs_R20_22", "mean"),
    D_basin_mean=("D_basin_minabs_R23_26", "mean"),
    D_commit_mean=("D_commit_abs_R27", "mean"),

    B_entry_mean=("B_entry_close", "mean"),
    B_basin_mean=("B_basin_close", "mean"),
    B_commit_mean=("B_commit_close", "mean"),

    S_entry_mean=("S_entry_mean_R20_22", "mean"),
    S_basin_mean=("S_basin_mean_R23_26", "mean"),
    S_commit_mean=("S_commit_R27", "mean"),
).reset_index()

phase_summary = df.groupby("phase_target").agg(
    n=("Gen_E", "count"),
    GenE_rate=("Gen_E", "mean"),

    U_K_mean=("U_K", "mean"),
    U_K_std=("U_K", "std"),

    D_entry_mean=("D_entry_minabs_R20_22", "mean"),
    D_basin_mean=("D_basin_minabs_R23_26", "mean"),
    D_commit_mean=("D_commit_abs_R27", "mean"),

    B_entry_mean=("B_entry_close", "mean"),
    B_basin_mean=("B_basin_close", "mean"),
    B_commit_mean=("B_commit_close", "mean"),

    S_entry_mean=("S_entry_mean_R20_22", "mean"),
    S_basin_mean=("S_basin_mean_R23_26", "mean"),
    S_commit_mean=("S_commit_R27", "mean"),
).reset_index()

condition_summary.to_csv(SAVE_DIR / "ua10a_condition_summary.csv", index=False)
phase_summary.to_csv(SAVE_DIR / "ua10a_phase_summary.csv", index=False)

# ============================================================
# SAVE EVAL
# ============================================================

df.to_csv(SAVE_DIR / "ua10a_eval.csv", index=False)

# ============================================================
# FINAL SUMMARY
# ============================================================

summary = {
    "experiment": "UA-10A Competition Boundary Mediation Audit",
    "input_path": str(INPUT_PATH),
    "n_rows": int(len(df)),

    "hypothesis": "U_K -> D_B/B_close -> Gen_E",

    "uk_to_mediator": uk_to_mediator_df.to_dict(orient="records"),
    "feature_eval_top": feature_eval.head(12).to_dict(orient="records"),
    "mediation_summary": mediation_summary,

    "condition_summary": condition_summary.to_dict(orient="records"),
    "phase_summary": phase_summary.to_dict(orient="records"),

    "interpretation_rules": {
        "PASS_FULL_CHAIN": [
            "U_K predicts B_basin or B_commit with nontrivial correlation",
            "B_basin or B_commit beats U_K_only in Gen_E prediction",
            "U_K_plus_B improves over U_K_only",
            "Non-final B_basin works, not only B_commit"
        ],
        "PASS_PARTIAL_CHAIN": [
            "Boundary distance predicts Gen_E better than U_K",
            "But U_K weakly predicts boundary compression",
            "This means U_K and D_B are related but not a simple causal chain"
        ],
        "FAIL_CHAIN": [
            "U_K does not predict boundary variables",
            "Boundary variables do not improve Gen_E prediction",
            "Only final signed R predicts Gen_E"
        ],
        "IMPORTANT_CAVEAT": [
            "B_commit is close to output and may be an oracle readout.",
            "B_basin is the stronger nontrivial mediator if it works."
        ]
    }
}

with open(SAVE_DIR / "ua10a_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nSaved to: {SAVE_DIR.resolve()}")