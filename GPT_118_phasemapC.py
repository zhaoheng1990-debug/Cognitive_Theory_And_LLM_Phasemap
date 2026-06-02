import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score
from sklearn.model_selection import StratifiedKFold, KFold

# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path(r"C:\Windows\System32\phasemap1b_outputs\phasemap1b_eval.csv")
SAVE_DIR = Path("./phasemap1c_outputs")
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
# CONSTRUCT D_B(l), M(l), AND BOUNDARY VELOCITY
# ============================================================

for l in R_LAYERS:
    df[f"D_{l}"] = df[f"R_{l}"].abs()

# Boundary distance velocity:
# negative means moving toward boundary;
# positive means moving away from boundary.
df["V_DB_20_22"] = df["D_22"] - df["D_20"]
df["V_DB_23_25"] = df["D_25"] - df["D_23"]
df["V_DB_20_25"] = df["D_25"] - df["D_20"]

df["V_DB_slope_20_22"] = df["V_DB_20_22"] / 2.0
df["V_DB_slope_23_25"] = df["V_DB_23_25"] / 2.0
df["V_DB_slope_20_25"] = df["V_DB_20_25"] / 5.0

# Signed momentum from R itself
df["M_R_20_22"] = df["R_22"] - df["R_20"]
df["M_R_23_25"] = df["R_25"] - df["R_23"]
df["M_R_20_25"] = df["R_25"] - df["R_20"]

df["M_R_slope_20_22"] = df["M_R_20_22"] / 2.0
df["M_R_slope_23_25"] = df["M_R_23_25"] / 2.0
df["M_R_slope_20_25"] = df["M_R_20_25"] / 5.0

# Curvature / acceleration
df["A_R_entry_to_basin"] = df["M_R_slope_23_25"] - df["M_R_slope_20_22"]
df["A_DB_entry_to_basin"] = df["V_DB_slope_23_25"] - df["V_DB_slope_20_22"]

# Directional decomposition
df["sign_R25"] = np.sign(df["R_25"])
df["M_directional"] = df["sign_R25"] * df["M_R_20_25"]
df["V_directional"] = df["sign_R25"] * df["V_DB_20_25"]

# ============================================================
# HELPERS
# ============================================================

def binary_cv(data, features, positive_label="critical"):
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

def cv_linear_predict(data, x_cols, y_col):
    X = data[x_cols].values.astype(float)
    y = data[y_col].values.astype(float)

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    pred = np.zeros(len(data))

    for tr, te in kf.split(X):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LinearRegression()),
        ])
        pipe.fit(X[tr], y[tr])
        pred[te] = pipe.predict(X[te])

    corr = np.corrcoef(y, pred)[0, 1]
    return {
        "r2": float(r2_score(y, pred)),
        "corr": float(corr),
    }

def corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])

# ============================================================
# FEATURE SETS
# ============================================================

FEATURE_SETS = {
    "UK_DB_baseline": ["U_K", "D_B"],

    # M only
    "M_R_entry": ["M_R_20_22", "M_R_slope_20_22"],
    "M_R_basin": ["M_R_23_25", "M_R_slope_23_25"],
    "M_R_full": ["M_R_20_25", "M_R_slope_20_25"],
    "M_R_all": [
        "M_R_20_22", "M_R_23_25", "M_R_20_25",
        "M_R_slope_20_22", "M_R_slope_23_25", "M_R_slope_20_25",
        "A_R_entry_to_basin",
    ],

    # boundary velocity only
    "V_DB_entry": ["V_DB_20_22", "V_DB_slope_20_22"],
    "V_DB_basin": ["V_DB_23_25", "V_DB_slope_23_25"],
    "V_DB_full": ["V_DB_20_25", "V_DB_slope_20_25"],
    "V_DB_all": [
        "V_DB_20_22", "V_DB_23_25", "V_DB_20_25",
        "V_DB_slope_20_22", "V_DB_slope_23_25", "V_DB_slope_20_25",
        "A_DB_entry_to_basin",
    ],

    # compare
    "UK_DB_plus_M_R_all": [
        "U_K", "D_B",
        "M_R_20_22", "M_R_23_25", "M_R_20_25",
        "A_R_entry_to_basin",
    ],
    "UK_DB_plus_V_DB_all": [
        "U_K", "D_B",
        "V_DB_20_22", "V_DB_23_25", "V_DB_20_25",
        "A_DB_entry_to_basin",
    ],
    "UK_DB_plus_M_and_V": [
        "U_K", "D_B",
        "M_R_20_22", "M_R_23_25", "M_R_20_25",
        "V_DB_20_22", "V_DB_23_25", "V_DB_20_25",
        "A_R_entry_to_basin", "A_DB_entry_to_basin",
    ],

    # directional terms
    "directional_MV": ["M_directional", "V_directional", "sign_R25"],
}

# ============================================================
# RUN CLASSIFICATION
# ============================================================

rows = []

for name, feats in FEATURE_SETS.items():
    res = binary_cv(df, feats, positive_label="critical")
    rows.append({
        "task": "critical_vs_negative",
        "positive_label": "critical",
        "feature_set": name,
        "features": ",".join(feats),
        **res,
    })

eval_df = pd.DataFrame(rows).sort_values("auc", ascending=False)
eval_df.to_csv(SAVE_DIR / "phasemap1c_feature_eval.csv", index=False)

# ============================================================
# IS M EXPLAINED BY V_DB?
# ============================================================

reg_rows = []

reg_tests = [
    ("M_R_20_22", ["V_DB_20_22", "D_20", "D_22"]),
    ("M_R_23_25", ["V_DB_23_25", "D_23", "D_25"]),
    ("M_R_20_25", ["V_DB_20_25", "D_20", "D_25"]),
    ("M_R_20_25", ["V_DB_20_22", "V_DB_23_25", "D_20", "D_25"]),
]

for y_col, x_cols in reg_tests:
    res = cv_linear_predict(df, x_cols, y_col)
    reg_rows.append({
        "target_M": y_col,
        "predictors": ",".join(x_cols),
        **res,
        "raw_corr_first_predictor": corr(df[y_col], df[x_cols[0]]),
    })

reg_df = pd.DataFrame(reg_rows)
reg_df.to_csv(SAVE_DIR / "phasemap1c_M_from_boundary_velocity.csv", index=False)

# ============================================================
# SUMMARY TABLES
# ============================================================

summary_cols = [
    "U_K", "D_B",
    "M_R_20_22", "M_R_23_25", "M_R_20_25",
    "V_DB_20_22", "V_DB_23_25", "V_DB_20_25",
    "A_R_entry_to_basin", "A_DB_entry_to_basin",
    "M_directional", "V_directional",
]

phase_summary = df.groupby("phase_target").agg(
    n=("Gen_E", "count"),
    GenE_rate=("Gen_E", "mean"),
    **{f"{c}_mean": (c, "mean") for c in summary_cols},
    **{f"{c}_std": (c, "std") for c in summary_cols},
).reset_index()

condition_summary = df.groupby("condition").agg(
    n=("Gen_E", "count"),
    GenE_rate=("Gen_E", "mean"),
    phase=("phase_target", lambda x: x.mode().iloc[0]),
    **{f"{c}_mean": (c, "mean") for c in summary_cols},
).reset_index()

phase_summary.to_csv(SAVE_DIR / "phasemap1c_phase_summary.csv", index=False)
condition_summary.to_csv(SAVE_DIR / "phasemap1c_condition_summary.csv", index=False)

df.to_csv(SAVE_DIR / "phasemap1c_eval.csv", index=False)

# ============================================================
# SUMMARY JSON
# ============================================================

def metric_for(feature_set, metric):
    r = eval_df[eval_df["feature_set"] == feature_set]
    if len(r) == 0:
        return None
    return float(r.iloc[0][metric])

summary = {
    "experiment": "PhaseMap-1C Momentum vs Boundary-Velocity Audit",
    "input_path": str(INPUT_PATH),
    "n_rows": int(len(df)),
    "core_question": "Is basin momentum M reducible to dD_B/dl, or is it an independent state variable?",
    "top_feature_eval": eval_df.head(12).to_dict(orient="records"),
    "key_metrics": {
        "UK_DB_baseline_auc": metric_for("UK_DB_baseline", "auc"),
        "M_R_all_auc": metric_for("M_R_all", "auc"),
        "V_DB_all_auc": metric_for("V_DB_all", "auc"),
        "UK_DB_plus_M_R_all_auc": metric_for("UK_DB_plus_M_R_all", "auc"),
        "UK_DB_plus_V_DB_all_auc": metric_for("UK_DB_plus_V_DB_all", "auc"),
        "UK_DB_plus_M_and_V_auc": metric_for("UK_DB_plus_M_and_V", "auc"),
    },
    "M_from_boundary_velocity": reg_df.to_dict(orient="records"),
    "phase_summary": phase_summary.to_dict(orient="records"),
    "condition_summary": condition_summary.to_dict(orient="records"),
    "interpretation_rules": {
        "M_reducible_to_boundary_velocity": [
            "V_DB_all matches M_R_all in critical_vs_negative AUC.",
            "M_R is well predicted by V_DB with high R2/corr.",
            "Adding M to V_DB gives little improvement."
        ],
        "M_independent_state_variable": [
            "M_R_all clearly beats V_DB_all.",
            "V_DB poorly predicts M_R.",
            "UK_DB_plus_M beats UK_DB_plus_V_DB."
        ],
        "mixed_result": [
            "V_DB explains part of M, but M retains significant predictive advantage.",
            "Interpret M as signed basin-flow velocity, not merely d|R|/dl."
        ]
    }
}

with open(SAVE_DIR / "phasemap1c_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nSaved to: {SAVE_DIR.resolve()}")