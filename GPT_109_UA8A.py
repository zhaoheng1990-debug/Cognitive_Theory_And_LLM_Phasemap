# ============================================================
# UA-8A: Competition Critical Response Audit
#
# Goal:
#   Test whether competition order parameter ΔU_K explains
#   response sensitivity:
#
#       χ = | d P(Gen_E) / d ΔU_K |
#
# Input:
#   ha1a_debug_outputs/deltaU_competition.csv
#
# Outputs:
#   ua8a_outputs/
#       ua8a_response_curve.csv
#       ua8a_band_summary.csv
#       ua8a_summary.json
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

INPUT_PATH = Path("C:\Windows\System32\ha1a_debug_outputs\deltaU_competition.csv")
SAVE_DIR = Path("./ua8a_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_BINS = 12

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_PATH)

required = ["DeltaU", "Gen_E", "condition", "phase_target"]
for col in required:
    if col not in df.columns:
        raise RuntimeError(f"Missing column: {col}")

df = df.copy()
df["Gen_E"] = df["Gen_E"].astype(int)

# ============================================================
# 1. FIT P(Gen_E | DeltaU_K)
# ============================================================

X = df[["DeltaU"]].values.astype(float)
y = df["Gen_E"].values.astype(int)

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=2000)),
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

pred = np.zeros(len(df))

for tr, te in skf.split(X, y):
    clf.fit(X[tr], y[tr])
    pred[te] = clf.predict_proba(X[te])[:, 1]

df["P_GenE_pred"] = pred
df["Pred_GenE"] = (pred >= 0.5).astype(int)

metrics = {
    "cv_auc": float(roc_auc_score(y, pred)),
    "cv_acc": float(accuracy_score(y, df["Pred_GenE"])),
    "cv_f1": float(f1_score(y, df["Pred_GenE"], zero_division=0)),
}

# Fit final model on all data for smooth response curve
clf.fit(X, y)

# ============================================================
# 2. RESPONSE CURVE
# ============================================================

u_min, u_max = df["DeltaU"].min(), df["DeltaU"].max()
grid = np.linspace(u_min, u_max, 300).reshape(-1, 1)

p_grid = clf.predict_proba(grid)[:, 1]

# numerical derivative chi = |dp/dU|
chi = np.abs(np.gradient(p_grid, grid[:, 0]))

curve = pd.DataFrame({
    "DeltaU_grid": grid[:, 0],
    "P_GenE": p_grid,
    "chi": chi,
})

curve.to_csv(SAVE_DIR / "ua8a_response_curve.csv", index=False)

# ============================================================
# 3. EMPIRICAL BIN RESPONSE
# ============================================================

df["U_bin"] = pd.qcut(df["DeltaU"], q=N_BINS, duplicates="drop")

bin_df = df.groupby("U_bin").agg(
    DeltaU_mean=("DeltaU", "mean"),
    DeltaU_min=("DeltaU", "min"),
    DeltaU_max=("DeltaU", "max"),
    GenE_rate=("Gen_E", "mean"),
    count=("Gen_E", "count"),
).reset_index()

bin_df["empirical_chi"] = (
    bin_df["GenE_rate"].diff().abs()
    /
    bin_df["DeltaU_mean"].diff().abs()
)

bin_df.to_csv(SAVE_DIR / "ua8a_empirical_bins.csv", index=False)

# ============================================================
# 4. CRITICAL BAND ESTIMATION
# ============================================================

# Define critical band as top 20% chi region on smooth curve
chi_threshold = np.quantile(curve["chi"], 0.80)
crit_curve = curve[curve["chi"] >= chi_threshold]

U_minus = float(crit_curve["DeltaU_grid"].min())
U_plus = float(crit_curve["DeltaU_grid"].max())
U_center = float(curve.loc[curve["chi"].idxmax(), "DeltaU_grid"])
chi_max = float(curve["chi"].max())

df["inside_chi_band"] = (
    (df["DeltaU"] >= U_minus) & (df["DeltaU"] <= U_plus)
).astype(int)

band_summary = df.groupby(["inside_chi_band", "phase_target", "condition"]).agg(
    n=("Gen_E", "count"),
    GenE_rate=("Gen_E", "mean"),
    DeltaU_mean=("DeltaU", "mean"),
).reset_index()

band_summary.to_csv(SAVE_DIR / "ua8a_band_summary.csv", index=False)

# ============================================================
# 5. PHASE / CONDITION SUMMARY
# ============================================================

phase_summary = df.groupby("phase_target").agg(
    n=("Gen_E", "count"),
    DeltaU_mean=("DeltaU", "mean"),
    DeltaU_std=("DeltaU", "std"),
    GenE_rate=("Gen_E", "mean"),
    inside_band_frac=("inside_chi_band", "mean"),
).reset_index()

condition_summary = df.groupby("condition").agg(
    n=("Gen_E", "count"),
    DeltaU_mean=("DeltaU", "mean"),
    DeltaU_std=("DeltaU", "std"),
    GenE_rate=("Gen_E", "mean"),
    inside_band_frac=("inside_chi_band", "mean"),
).reset_index()

phase_summary.to_csv(SAVE_DIR / "ua8a_phase_summary.csv", index=False)
condition_summary.to_csv(SAVE_DIR / "ua8a_condition_summary.csv", index=False)

# ============================================================
# 6. MAIN TESTS
# ============================================================

inside = df[df["inside_chi_band"] == 1]
outside = df[df["inside_chi_band"] == 0]

tests = {
    "chi_max": chi_max,
    "U_minus": U_minus,
    "U_center": U_center,
    "U_plus": U_plus,
    "band_width": U_plus - U_minus,
    "inside_band_GenE_rate": float(inside["Gen_E"].mean()) if len(inside) else None,
    "outside_band_GenE_rate": float(outside["Gen_E"].mean()) if len(outside) else None,
    "inside_band_count": int(len(inside)),
    "outside_band_count": int(len(outside)),
    "critical_phase_inside_frac": float(
        df[df["phase_target"] == "critical"]["inside_chi_band"].mean()
    ) if "critical" in set(df["phase_target"]) else None,
    "positive_phase_inside_frac": float(
        df[df["phase_target"] == "positive"]["inside_chi_band"].mean()
    ) if "positive" in set(df["phase_target"]) else None,
    "negative_phase_inside_frac": float(
        df[df["phase_target"] == "negative"]["inside_chi_band"].mean()
    ) if "negative" in set(df["phase_target"]) else None,
}

tests["critical_has_highest_inside_frac"] = (
    tests["critical_phase_inside_frac"] is not None
    and tests["critical_phase_inside_frac"] >= max(
        tests["positive_phase_inside_frac"],
        tests["negative_phase_inside_frac"],
    )
)

# ============================================================
# SAVE EVAL DATA
# ============================================================

df.to_csv(SAVE_DIR / "ua8a_eval.csv", index=False)

summary = {
    "experiment": "UA-8A Competition Critical Response Audit",
    "input_path": str(INPUT_PATH),
    "n_rows": int(len(df)),
    "metrics_DeltaU_to_GenE": metrics,
    "critical_response_band": tests,
    "phase_summary": phase_summary.to_dict(orient="records"),
    "condition_summary": condition_summary.to_dict(orient="records"),
    "interpretation_rules": {
        "PASS_Strong": [
            "cv_auc is clearly above chance",
            "critical_has_highest_inside_frac is true",
            "inside_band response is concentrated around ambiguous/critical samples",
            "chi curve has a clear peak rather than flat response"
        ],
        "FAIL": [
            "cv_auc near 0.5",
            "critical samples are not enriched in chi band",
            "chi curve is flat or dominated by endpoint artifacts"
        ]
    }
}

with open(SAVE_DIR / "ua8a_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nSaved to: {SAVE_DIR.resolve()}")