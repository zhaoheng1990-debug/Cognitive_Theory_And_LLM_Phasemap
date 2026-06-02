import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    r2_score,
    mean_absolute_error,
)
from sklearn.model_selection import StratifiedKFold, KFold

# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path(r"C:\Windows\System32\phasemap1c_outputs\phasemap1c_eval.csv")
SAVE_DIR = Path("./phasemap2a_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

D_LAYERS = [20, 21, 22, 23, 24, 25]
STATE_COLS = ["U_K", "D_B"]

# ============================================================
# LOAD
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Cannot find input file: {INPUT_PATH.resolve()}")

df = pd.read_csv(INPUT_PATH)

required = ["phase_target", "condition", "Gen_E", "U_K", "D_B"] + [f"D_{l}" for l in D_LAYERS]

for c in required:
    if c not in df.columns:
        raise RuntimeError(f"Missing column: {c}")

df = df.copy()
df["phase_target"] = df["phase_target"].astype(str)
df["Gen_E"] = df["Gen_E"].astype(int)

# ============================================================
# BUILD FLOW VARIABLES
# ============================================================

for a, b in zip(D_LAYERS[:-1], D_LAYERS[1:]):
    df[f"dD_{a}_{b}"] = df[f"D_{b}"] - df[f"D_{a}"]

df["dD_entry_20_22"] = df["D_22"] - df["D_20"]
df["dD_basin_23_25"] = df["D_25"] - df["D_23"]
df["dD_full_20_25"] = df["D_25"] - df["D_20"]

df["abs_dD_entry_20_22"] = df["dD_entry_20_22"].abs()
df["abs_dD_basin_23_25"] = df["dD_basin_23_25"].abs()
df["abs_dD_full_20_25"] = df["dD_full_20_25"].abs()

# Main flow target
MAIN_TARGET = "dD_basin_23_25"

df["flow_outward"] = (df[MAIN_TARGET] > 0).astype(int)
df["flow_inward"] = (df[MAIN_TARGET] < 0).astype(int)
df["near_boundary"] = (df["D_B"] < 1.5).astype(int)

# ============================================================
# HELPERS
# ============================================================

def cv_regression(data, features, target):
    X = data[features].values.astype(float)
    y = data[target].values.astype(float)

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    pred = np.zeros(len(data))

    for tr, te in kf.split(X):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ])
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])

    corr = np.corrcoef(y, pred)[0, 1] if np.std(pred) > 0 else 0.0

    return {
        "r2": float(r2_score(y, pred)),
        "corr": float(corr),
        "mae": float(mean_absolute_error(y, pred)),
    }

def cv_binary(data, features, target):
    X = data[features].values.astype(float)
    y = data[target].values.astype(int)

    if len(np.unique(y)) < 2:
        return None

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    pred = np.zeros(len(data))

    for tr, te in skf.split(X, y):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=5000)),
        ])
        model.fit(X[tr], y[tr])
        pred[te] = model.predict_proba(X[te])[:, 1]

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
    "UK_only": ["U_K"],
    "DB_only": ["D_B"],
    "UK_DB_state": ["U_K", "D_B"],

    "UK_DB_near_boundary": ["U_K", "D_B", "near_boundary"],

    "D_trajectory_entry": ["D_20", "D_21", "D_22"],
    "D_trajectory_basin": ["D_23", "D_24", "D_25"],
    "D_trajectory_full": ["D_20", "D_21", "D_22", "D_23", "D_24", "D_25"],

    "UK_DB_plus_entry_flow": ["U_K", "D_B", "dD_entry_20_22"],
    "UK_DB_plus_Dtraj": ["U_K", "D_B", "D_20", "D_21", "D_22", "D_23", "D_24", "D_25"],
}

FLOW_TARGETS = [
    "dD_entry_20_22",
    "dD_basin_23_25",
    "dD_full_20_25",
]

# ============================================================
# REGRESSION: predict dD
# ============================================================

reg_rows = []

for target in FLOW_TARGETS:
    for name, feats in FEATURE_SETS.items():
        res = cv_regression(df, feats, target)
        reg_rows.append({
            "task": "flow_regression",
            "target": target,
            "feature_set": name,
            "features": ",".join(feats),
            **res,
        })

reg_df = pd.DataFrame(reg_rows).sort_values(["target", "r2"], ascending=[True, False])
reg_df.to_csv(SAVE_DIR / "phasemap2a_flow_regression.csv", index=False)

# ============================================================
# CLASSIFICATION: predict flow sign
# ============================================================

clf_rows = []

for sign_target in ["flow_outward", "flow_inward"]:
    for name, feats in FEATURE_SETS.items():
        res = cv_binary(df, feats, sign_target)
        if res is None:
            continue
        clf_rows.append({
            "task": "flow_direction_classification",
            "target": sign_target,
            "feature_set": name,
            "features": ",".join(feats),
            **res,
        })

clf_df = pd.DataFrame(clf_rows).sort_values(["target", "auc"], ascending=[True, False])
clf_df.to_csv(SAVE_DIR / "phasemap2a_flow_direction.csv", index=False)

# ============================================================
# PHASE FLOW SUMMARY
# ============================================================

flow_cols = [
    "U_K",
    "D_B",
    "dD_entry_20_22",
    "dD_basin_23_25",
    "dD_full_20_25",
    "abs_dD_entry_20_22",
    "abs_dD_basin_23_25",
    "abs_dD_full_20_25",
    "flow_outward",
    "flow_inward",
    "near_boundary",
]

phase_summary = df.groupby("phase_target").agg(
    n=("Gen_E", "count"),
    GenE_rate=("Gen_E", "mean"),
    **{f"{c}_mean": (c, "mean") for c in flow_cols},
    **{f"{c}_std": (c, "std") for c in flow_cols if c not in ["flow_outward", "flow_inward", "near_boundary"]},
).reset_index()

condition_summary = df.groupby("condition").agg(
    n=("Gen_E", "count"),
    phase=("phase_target", lambda x: x.mode().iloc[0]),
    GenE_rate=("Gen_E", "mean"),
    **{f"{c}_mean": (c, "mean") for c in flow_cols},
).reset_index()

phase_summary.to_csv(SAVE_DIR / "phasemap2a_phase_flow_summary.csv", index=False)
condition_summary.to_csv(SAVE_DIR / "phasemap2a_condition_flow_summary.csv", index=False)

# ============================================================
# 2D FLOW FIELD MAP
# ============================================================

# Bin U_K x D_B and compute mean flow.
N_BINS_UK = 8
N_BINS_DB = 8

df["UK_bin"] = pd.qcut(df["U_K"], q=N_BINS_UK, duplicates="drop")
df["DB_bin"] = pd.qcut(df["D_B"], q=N_BINS_DB, duplicates="drop")

grid_rows = []

for (uk_bin, db_bin), sub in df.groupby(["UK_bin", "DB_bin"], observed=True):
    if len(sub) < 3:
        continue

    row = {
        "UK_bin": str(uk_bin),
        "DB_bin": str(db_bin),
        "n": int(len(sub)),
        "U_K_center": float(sub["U_K"].mean()),
        "D_B_center": float(sub["D_B"].mean()),
        "mean_dD_entry_20_22": float(sub["dD_entry_20_22"].mean()),
        "mean_dD_basin_23_25": float(sub["dD_basin_23_25"].mean()),
        "mean_dD_full_20_25": float(sub["dD_full_20_25"].mean()),
        "mean_abs_dD_basin_23_25": float(sub["abs_dD_basin_23_25"].mean()),
        "outward_rate": float(sub["flow_outward"].mean()),
        "GenE_rate": float(sub["Gen_E"].mean()),
        "dominant_phase": sub["phase_target"].mode().iloc[0],
    }
    grid_rows.append(row)

grid_df = pd.DataFrame(grid_rows)
grid_df.to_csv(SAVE_DIR / "phasemap2a_flow_grid.csv", index=False)

# ============================================================
# ATTRACTOR / SADDLE HEURISTIC
# ============================================================

# Simple diagnostic:
# attractor-like: low |mean_dD| and high sample density
# saddle/critical-like: high |mean_dD| near boundary
if len(grid_df) > 0:
    grid_df["flow_magnitude"] = grid_df["mean_dD_basin_23_25"].abs()
    grid_df["attractor_score"] = grid_df["n"] / (1.0 + grid_df["flow_magnitude"])
    grid_df["critical_flow_score"] = grid_df["mean_abs_dD_basin_23_25"] / (1.0 + grid_df["D_B_center"])

    attractor_candidates = grid_df.sort_values("attractor_score", ascending=False).head(8)
    critical_candidates = grid_df.sort_values("critical_flow_score", ascending=False).head(8)
else:
    attractor_candidates = pd.DataFrame()
    critical_candidates = pd.DataFrame()

attractor_candidates.to_csv(SAVE_DIR / "phasemap2a_attractor_candidates.csv", index=False)
critical_candidates.to_csv(SAVE_DIR / "phasemap2a_critical_flow_candidates.csv", index=False)

# ============================================================
# SAVE EVAL
# ============================================================

df.to_csv(SAVE_DIR / "phasemap2a_eval.csv", index=False)

# ============================================================
# SUMMARY JSON
# ============================================================

def best_reg(target):
    sub = reg_df[reg_df["target"] == target].sort_values("r2", ascending=False)
    return sub.head(8).to_dict(orient="records")

def best_clf(target):
    sub = clf_df[clf_df["target"] == target].sort_values("auc", ascending=False)
    return sub.head(8).to_dict(orient="records")

def metric_row(table, target, feature_set, metric):
    sub = table[
        (table["target"] == target)
        & (table["feature_set"] == feature_set)
    ]
    if len(sub) == 0:
        return None
    val = sub.iloc[0][metric]
    return None if pd.isna(val) else float(val)

summary = {
    "experiment": "PhaseMap-2A Dynamic Flow Audit",
    "input_path": str(INPUT_PATH),
    "n_rows": int(len(df)),
    "state_definition": {
        "U_K": "competition intensity",
        "D_B": "boundary distance",
        "flow": "dD/dl over D_l = |R_l|"
    },
    "main_target": MAIN_TARGET,
    "key_metrics": {
        "UK_DB_to_dD_basin_r2": metric_row(reg_df, "dD_basin_23_25", "UK_DB_state", "r2"),
        "UK_DB_to_dD_basin_corr": metric_row(reg_df, "dD_basin_23_25", "UK_DB_state", "corr"),
        "UK_DB_to_flow_outward_auc": metric_row(clf_df, "flow_outward", "UK_DB_state", "auc"),
        "UK_DB_to_flow_outward_f1": metric_row(clf_df, "flow_outward", "UK_DB_state", "f1"),
        "UK_DB_plus_entry_to_flow_outward_auc": metric_row(clf_df, "flow_outward", "UK_DB_plus_entry_flow", "auc"),
        "D_traj_full_to_dD_basin_r2": metric_row(reg_df, "dD_basin_23_25", "D_trajectory_full", "r2"),
    },
    "best_regression": {
        "dD_entry_20_22": best_reg("dD_entry_20_22"),
        "dD_basin_23_25": best_reg("dD_basin_23_25"),
        "dD_full_20_25": best_reg("dD_full_20_25"),
    },
    "best_direction_classification": {
        "flow_outward": best_clf("flow_outward"),
        "flow_inward": best_clf("flow_inward"),
    },
    "phase_flow_summary": phase_summary.to_dict(orient="records"),
    "condition_flow_summary": condition_summary.to_dict(orient="records"),
    "flow_grid_top_attractors": attractor_candidates.to_dict(orient="records"),
    "flow_grid_top_critical": critical_candidates.to_dict(orient="records"),
    "interpretation_rules": {
        "PASS_dynamic_flow": [
            "UK_DB_state predicts flow_outward with AUC > 0.75.",
            "UK_DB_state predicts dD_basin_23_25 with positive R2 and corr > 0.4.",
            "Grid flow field shows phase-dependent flow direction."
        ],
        "PASS_partial": [
            "UK_DB weakly predicts continuous dD but predicts sign(flow) well.",
            "D_B dynamics exist but may require nonlinear model or trajectory history."
        ],
        "FAIL_state_flow": [
            "UK_DB_state cannot predict flow sign or dD.",
            "Only full D trajectory predicts flow, meaning current 2D state is insufficient."
        ],
        "important_caveat": [
            "If D_trajectory_full dominates UK_DB_state, then (U_K,D_B) is not Markov-sufficient.",
            "A future PhaseMap-2B should test nonlinear flow models and lagged state variables."
        ]
    }
}

with open(SAVE_DIR / "phasemap2a_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nSaved to: {SAVE_DIR.resolve()}")