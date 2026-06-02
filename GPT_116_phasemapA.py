import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path(r"C:\Windows\System32\phasemap1_outputs\phasemap1_eval.csv")
SAVE_DIR = Path("./phasemap1a_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

R_LAYERS = [20, 21, 22, 23, 24, 25]

# ============================================================
# LOAD
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Cannot find: {INPUT_PATH.resolve()}")

df = pd.read_csv(INPUT_PATH)

required = ["phase_target", "condition", "Gen_E", "U_K", "D_B"] + [f"R_{l}" for l in R_LAYERS]
for c in required:
    if c not in df.columns:
        raise RuntimeError(f"Missing column: {c}")

df = df.copy()
df["phase_target"] = df["phase_target"].astype(str)
df["Gen_E"] = df["Gen_E"].astype(int)

# ============================================================
# CONSTRUCT MOMENTUM FEATURES
# ============================================================

# Basic trajectory momentum
df["M_R25_minus_R20"] = df["R_25"] - df["R_20"]
df["M_R24_minus_R20"] = df["R_24"] - df["R_20"]
df["M_R25_minus_R22"] = df["R_25"] - df["R_22"]

# Mean slope over L20-L25
df["M_slope_20_25"] = (df["R_25"] - df["R_20"]) / 5.0

# Area / accumulation
df["M_area_R20_25"] = df[[f"R_{l}" for l in R_LAYERS]].sum(axis=1)
df["M_mean_R20_25"] = df[[f"R_{l}" for l in R_LAYERS]].mean(axis=1)

# Curvature-like terms
df["M_front_slope_20_22"] = (df["R_22"] - df["R_20"]) / 2.0
df["M_back_slope_23_25"] = (df["R_25"] - df["R_23"]) / 2.0
df["M_slope_shift"] = df["M_back_slope_23_25"] - df["M_front_slope_20_22"]

# Boundary crossing / sign instability
r_values = df[[f"R_{l}" for l in R_LAYERS]]
df["M_min_R20_25"] = r_values.min(axis=1)
df["M_max_R20_25"] = r_values.max(axis=1)
df["M_range_R20_25"] = df["M_max_R20_25"] - df["M_min_R20_25"]
df["M_cross_zero"] = ((df["M_min_R20_25"] <= 0) & (df["M_max_R20_25"] >= 0)).astype(int)

# Directional commitment score
df["M_commit_direction"] = np.sign(df["R_25"]) * df["D_B"]

# ============================================================
# EVAL HELPERS
# ============================================================

def binary_cv_auc(data, features, positive_label):
    sub = data[data["phase_target"].isin(["critical", "negative"])].copy()
    sub["target"] = (sub["phase_target"] == positive_label).astype(int)

    X = sub[features].values.astype(float)
    y = sub["target"].values.astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pred = np.zeros(len(sub))

    for tr, te in skf.split(X, y):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=5000)),
        ])
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict_proba(X[te])[:, 1]

    label = (pred >= 0.5).astype(int)

    return {
        "auc": float(roc_auc_score(y, pred)),
        "acc": float(accuracy_score(y, label)),
        "f1": float(f1_score(y, label, zero_division=0)),
    }

# ============================================================
# FEATURE SETS
# ============================================================

FEATURE_SETS = {
    "UK_DB_baseline": ["U_K", "D_B"],

    "M_simple": ["M_R25_minus_R20"],
    "M_slope": ["M_slope_20_25"],
    "M_area": ["M_area_R20_25"],
    "M_mean": ["M_mean_R20_25"],
    "M_range": ["M_range_R20_25"],
    "M_cross_zero": ["M_cross_zero"],
    "M_slope_shift": ["M_slope_shift"],
    "M_commit_direction": ["M_commit_direction"],

    "M_all": [
        "M_R25_minus_R20",
        "M_slope_20_25",
        "M_area_R20_25",
        "M_mean_R20_25",
        "M_range_R20_25",
        "M_cross_zero",
        "M_slope_shift",
        "M_commit_direction",
    ],

    "UK_DB_plus_M_simple": ["U_K", "D_B", "M_R25_minus_R20"],
    "UK_DB_plus_M_area": ["U_K", "D_B", "M_area_R20_25"],
    "UK_DB_plus_M_commit": ["U_K", "D_B", "M_commit_direction"],

    "STATE_UK_DB_M_all": [
        "U_K",
        "D_B",
        "M_R25_minus_R20",
        "M_slope_20_25",
        "M_area_R20_25",
        "M_mean_R20_25",
        "M_range_R20_25",
        "M_cross_zero",
        "M_slope_shift",
        "M_commit_direction",
    ],
}

# ============================================================
# RUN CRITICAL VS NEGATIVE TEST
# ============================================================

rows = []

for name, feats in FEATURE_SETS.items():
    res = binary_cv_auc(df, feats, positive_label="critical")
    rows.append({
        "task": "critical_vs_negative",
        "positive_label": "critical",
        "feature_set": name,
        "features": ",".join(feats),
        **res,
    })

eval_df = pd.DataFrame(rows).sort_values("auc", ascending=False)
eval_df.to_csv(SAVE_DIR / "phasemap1a_feature_eval.csv", index=False)

# ============================================================
# SUMMARY TABLES
# ============================================================

momentum_cols = [
    "U_K",
    "D_B",
    "M_R25_minus_R20",
    "M_slope_20_25",
    "M_area_R20_25",
    "M_mean_R20_25",
    "M_range_R20_25",
    "M_cross_zero",
    "M_slope_shift",
    "M_commit_direction",
]

phase_summary = df.groupby("phase_target").agg(
    n=("Gen_E", "count"),
    GenE_rate=("Gen_E", "mean"),
    **{f"{c}_mean": (c, "mean") for c in momentum_cols},
    **{f"{c}_std": (c, "std") for c in momentum_cols},
).reset_index()

condition_summary = df.groupby("condition").agg(
    n=("Gen_E", "count"),
    GenE_rate=("Gen_E", "mean"),
    phase=("phase_target", lambda x: x.mode().iloc[0]),
    **{f"{c}_mean": (c, "mean") for c in momentum_cols},
).reset_index()

phase_summary.to_csv(SAVE_DIR / "phasemap1a_phase_summary.csv", index=False)
condition_summary.to_csv(SAVE_DIR / "phasemap1a_condition_summary.csv", index=False)

df.to_csv(SAVE_DIR / "phasemap1a_eval.csv", index=False)

# ============================================================
# SUMMARY JSON
# ============================================================

summary = {
    "experiment": "PhaseMap-1A Momentum Axis Audit",
    "input_path": str(INPUT_PATH),
    "n_rows": int(len(df)),
    "core_question": "Can momentum M separate critical from negative when U_K and D_B overlap?",
    "top_feature_eval": eval_df.head(12).to_dict(orient="records"),
    "phase_summary": phase_summary.to_dict(orient="records"),
    "condition_summary": condition_summary.to_dict(orient="records"),
    "interpretation_rules": {
        "PASS_M_axis": [
            "M feature sets beat UK_DB_baseline on critical_vs_negative.",
            "UK_DB_plus_M improves clearly over UK_DB_baseline.",
            "Critical and negative have distinct M means or sign patterns."
        ],
        "FAIL_M_axis": [
            "M does not improve over UK_DB.",
            "Critical and negative remain overlapped in all M variables."
        ],
        "main_caveat": [
            "M based on R20:25 may still be close to answer-basin readout; later test should use L20:23 or L7:19 momentum."
        ]
    }
}

with open(SAVE_DIR / "phasemap1a_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nSaved to: {SAVE_DIR.resolve()}")