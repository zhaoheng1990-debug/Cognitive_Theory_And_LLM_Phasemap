# ============================================================
# UA-9A: Basin Distance Response Audit
#
# Goal:
#   Test whether response sensitivity χ is governed by basin
#   distance D_B rather than competition order parameter U_K.
#
# Inputs:
#   ha1a_debug_outputs/deltaU_competition.csv
#
# Main candidates:
#   U_K        = DeltaU
#   D_B_final  = |R_27|
#   D_B_basin  = |mean(R_23:26)|
#   D_B_min    = min(|R_23|, |R_24|, |R_25|, |R_26|)
#   D_B_prob   = |P_E_final - 0.5| if P is available, else skipped
#
# Interpretation:
#   If D_B is the true distance-to-boundary variable:
#       smaller D_B -> higher response sensitivity / uncertainty
#       D_B predicts Gen_E boundary better than U_K
#       near-boundary samples have higher mixed outcomes
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = r"C:\Windows\System32\ua8a_outputs\deltaU_competition.csv"
SAVE_DIR = Path("./ua9a_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_BINS = 12

BASIN_LAYERS = [23, 24, 25, 26]
FINAL_LAYER = 27

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_PATH)

required = ["DeltaU", "Gen_E", "condition", "phase_target"]
for col in required:
    if col not in df.columns:
        raise RuntimeError(f"Missing required column: {col}")

for l in BASIN_LAYERS + [FINAL_LAYER]:
    col = f"R_{l}"
    if col not in df.columns:
        raise RuntimeError(f"Missing required column: {col}")

df = df.copy()
df["Gen_E"] = df["Gen_E"].astype(int)

# ============================================================
# CONSTRUCT BASIN DISTANCE CANDIDATES
# ============================================================

df["U_K"] = df["DeltaU"]

df["R_basin_mean_23_26"] = df[[f"R_{l}" for l in BASIN_LAYERS]].mean(axis=1)
df["R_basin_minabs_23_26"] = df[[f"R_{l}" for l in BASIN_LAYERS]].abs().min(axis=1)
df["R_basin_final_27"] = df[f"R_{FINAL_LAYER}"]

# Distance-to-boundary candidates
df["D_B_final_absR27"] = df["R_basin_final_27"].abs()
df["D_B_basin_absMeanR23_26"] = df["R_basin_mean_23_26"].abs()
df["D_B_min_absR23_26"] = df["R_basin_minabs_23_26"]

# Signed basin indicators, not distances
df["S_final_R27"] = df["R_basin_final_27"]
df["S_basin_meanR23_26"] = df["R_basin_mean_23_26"]

FEATURES = {
    "U_K_only": ["U_K"],

    "D_B_final_absR27": ["D_B_final_absR27"],
    "D_B_basin_absMeanR23_26": ["D_B_basin_absMeanR23_26"],
    "D_B_min_absR23_26": ["D_B_min_absR23_26"],

    "S_final_R27": ["S_final_R27"],
    "S_basin_meanR23_26": ["S_basin_meanR23_26"],

    "U_K_plus_D_B_final": ["U_K", "D_B_final_absR27"],
    "U_K_plus_D_B_basin": ["U_K", "D_B_basin_absMeanR23_26"],
    "U_K_plus_signed_final": ["U_K", "S_final_R27"],
    "U_K_plus_signed_basin": ["U_K", "S_basin_meanR23_26"],
}

# ============================================================
# CV EVALUATION
# ============================================================

def cv_eval_feature_set(df, feature_cols):
    X = df[feature_cols].values.astype(float)
    y = df["Gen_E"].values.astype(int)

    if len(np.unique(y)) < 2:
        return {
            "auc": None,
            "acc": None,
            "f1": None,
            "pred": [None] * len(df),
        }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pred = np.zeros(len(df))

    for tr, te in skf.split(X, y):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=3000)),
        ])
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict_proba(X[te])[:, 1]

    pred_label = (pred >= 0.5).astype(int)

    return {
        "auc": float(roc_auc_score(y, pred)),
        "acc": float(accuracy_score(y, pred_label)),
        "f1": float(f1_score(y, pred_label, zero_division=0)),
        "pred": pred.tolist(),
    }

eval_rows = []
pred_store = {}

for name, cols in FEATURES.items():
    res = cv_eval_feature_set(df, cols)
    eval_rows.append({
        "feature_set": name,
        "features": ",".join(cols),
        "auc": res["auc"],
        "acc": res["acc"],
        "f1": res["f1"],
    })
    pred_store[name] = res["pred"]

eval_df = pd.DataFrame(eval_rows).sort_values(
    by="auc",
    ascending=False,
    na_position="last",
)

eval_df.to_csv(SAVE_DIR / "ua9a_feature_eval.csv", index=False)

for name, pred in pred_store.items():
    df[f"Pred_{name}"] = pred

# ============================================================
# DISTANCE RESPONSE ANALYSIS
# ============================================================

def bin_response(df, variable, q=N_BINS):
    out = df.copy()
    out = out[np.isfinite(out[variable])].copy()

    out[f"{variable}_bin"] = pd.qcut(
        out[variable],
        q=q,
        duplicates="drop",
    )

    b = out.groupby(f"{variable}_bin").agg(
        var_mean=(variable, "mean"),
        var_min=(variable, "min"),
        var_max=(variable, "max"),
        GenE_rate=("Gen_E", "mean"),
        count=("Gen_E", "count"),
        U_K_mean=("U_K", "mean"),
    ).reset_index()

    b["empirical_chi"] = (
        b["GenE_rate"].diff().abs()
        /
        b["var_mean"].diff().abs()
    )

    b["variable"] = variable
    return b

distance_vars = [
    "U_K",
    "D_B_final_absR27",
    "D_B_basin_absMeanR23_26",
    "D_B_min_absR23_26",
    "S_final_R27",
    "S_basin_meanR23_26",
]

bin_tables = []
for v in distance_vars:
    b = bin_response(df, v)
    b.to_csv(SAVE_DIR / f"ua9a_bins_{v}.csv", index=False)
    bin_tables.append(b)

all_bins = pd.concat(bin_tables, ignore_index=True)
all_bins.to_csv(SAVE_DIR / "ua9a_all_bins.csv", index=False)

# ============================================================
# NEAR-BOUNDARY TEST
# ============================================================

boundary_rows = []

for v in [
    "D_B_final_absR27",
    "D_B_basin_absMeanR23_26",
    "D_B_min_absR23_26",
]:
    # Near boundary = lowest 25% distance
    threshold = float(df[v].quantile(0.25))
    near = df[df[v] <= threshold]
    far = df[df[v] > threshold]

    boundary_rows.append({
        "distance_var": v,
        "near_threshold_q25": threshold,
        "near_count": int(len(near)),
        "far_count": int(len(far)),
        "near_GenE_rate": float(near["Gen_E"].mean()),
        "far_GenE_rate": float(far["Gen_E"].mean()),
        "near_condition_mix": near["condition"].value_counts(normalize=True).to_dict(),
        "far_condition_mix": far["condition"].value_counts(normalize=True).to_dict(),
        "near_phase_mix": near["phase_target"].value_counts(normalize=True).to_dict(),
        "far_phase_mix": far["phase_target"].value_counts(normalize=True).to_dict(),
    })

boundary_df = pd.DataFrame(boundary_rows)
boundary_df.to_csv(SAVE_DIR / "ua9a_boundary_test.csv", index=False)

# ============================================================
# CORRELATION WITH UNCERTAINTY-LIKE RESPONSE
# ============================================================

# Binary uncertainty proxy:
# max at predicted probability near 0.5.
# This is model-based, using best signed predictor if available.
best_feature_set = eval_df.iloc[0]["feature_set"]
best_pred_col = f"Pred_{best_feature_set}"

if best_pred_col in df.columns:
    df["uncertainty_proxy"] = 0.5 - (df[best_pred_col] - 0.5).abs()
else:
    df["uncertainty_proxy"] = np.nan

corr_rows = []
for v in distance_vars:
    if df["uncertainty_proxy"].notna().any():
        corr = np.corrcoef(df[v], df["uncertainty_proxy"])[0, 1]
    else:
        corr = np.nan

    corr_rows.append({
        "variable": v,
        "corr_with_uncertainty_proxy": float(corr) if np.isfinite(corr) else None,
        "mean": float(df[v].mean()),
        "std": float(df[v].std()),
    })

corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(SAVE_DIR / "ua9a_uncertainty_corr.csv", index=False)

# ============================================================
# CONDITION / PHASE SUMMARY
# ============================================================

condition_summary = df.groupby("condition").agg(
    n=("Gen_E", "count"),
    GenE_rate=("Gen_E", "mean"),
    U_K_mean=("U_K", "mean"),
    U_K_std=("U_K", "std"),
    D_B_final_mean=("D_B_final_absR27", "mean"),
    D_B_basin_mean=("D_B_basin_absMeanR23_26", "mean"),
    D_B_min_mean=("D_B_min_absR23_26", "mean"),
    S_final_R27_mean=("S_final_R27", "mean"),
    S_basin_mean_mean=("S_basin_meanR23_26", "mean"),
).reset_index()

phase_summary = df.groupby("phase_target").agg(
    n=("Gen_E", "count"),
    GenE_rate=("Gen_E", "mean"),
    U_K_mean=("U_K", "mean"),
    U_K_std=("U_K", "std"),
    D_B_final_mean=("D_B_final_absR27", "mean"),
    D_B_basin_mean=("D_B_basin_absMeanR23_26", "mean"),
    D_B_min_mean=("D_B_min_absR23_26", "mean"),
    S_final_R27_mean=("S_final_R27", "mean"),
    S_basin_mean_mean=("S_basin_meanR23_26", "mean"),
).reset_index()

condition_summary.to_csv(SAVE_DIR / "ua9a_condition_summary.csv", index=False)
phase_summary.to_csv(SAVE_DIR / "ua9a_phase_summary.csv", index=False)

# ============================================================
# SAVE EVAL
# ============================================================

df.to_csv(SAVE_DIR / "ua9a_eval.csv", index=False)

# ============================================================
# SUMMARY
# ============================================================

best_rows = eval_df.head(5).to_dict(orient="records")

summary = {
    "experiment": "UA-9A Basin Distance Response Audit",
    "input_path": str(INPUT_PATH),
    "n_rows": int(len(df)),
    "feature_eval_top5": best_rows,
    "u_k_baseline": eval_df[eval_df["feature_set"] == "U_K_only"].to_dict(orient="records"),
    "boundary_test": boundary_rows,
    "uncertainty_corr": corr_rows,
    "condition_summary": condition_summary.to_dict(orient="records"),
    "phase_summary": phase_summary.to_dict(orient="records"),
    "interpretation_rules": {
        "PASS_D_B": [
            "A D_B feature beats U_K_only by clear margin",
            "near-boundary samples show higher uncertainty / mixed outcomes",
            "D_B correlates negatively with uncertainty distance or positively with response proxy"
        ],
        "FAIL_D_B": [
            "D_B features do not outperform U_K",
            "near-boundary samples are not more mixed or sensitive",
            "signed R features dominate entirely, meaning this is just answer readout"
        ],
        "IMPORTANT_CAVEAT": [
            "Signed final R may trivially predict Gen_E because it is close to output.",
            "The strongest nontrivial result would come from D_B_basin_absMeanR23_26 or D_B_min_absR23_26, not S_final_R27."
        ]
    }
}

with open(SAVE_DIR / "ua9a_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nSaved to: {SAVE_DIR.resolve()}")