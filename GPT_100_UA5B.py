import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

INPUT_CSV = r"C:\Windows\System32\ua5a_outputs\ua5a_dataset_with_critical_phenomena.csv"
CONFIG_JSON = "./ua5a_outputs/ua5a_config.json"

SAVE_DIR = Path("./ua5b_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

U_COL = "DeltaU_global"
GROUP_COL = "graph_id"

df = pd.read_csv(INPUT_CSV)

with open(CONFIG_JSON, "r", encoding="utf-8") as f:
    cfg = json.load(f)

U_MINUS = cfg["critical_band"]["U_minus"]
U_CENTER = cfg["critical_band"]["U_center"]
U_PLUS = cfg["critical_band"]["U_plus"]

if "gen_negative_proxy" not in df.columns:
    df["gen_negative_proxy"] = (df["target_final_R"] < 0).astype(int)

# ============================================================
# 1. Fit logistic response P(Gen_E | DeltaU)
# ============================================================

X = df[[U_COL]].values.astype(float)
y = df["gen_negative_proxy"].values.astype(int)
groups = df[GROUP_COL].values

gkf = GroupKFold(n_splits=6)

probs = np.zeros(len(df))
coef_list = []
intercept_list = []

for tr, te in gkf.split(X, y, groups):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ])
    model.fit(X[tr], y[tr])
    probs[te] = model.predict_proba(X[te])[:, 1]

    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]

    # Convert coefficient back to raw DeltaU scale
    beta_scaled = clf.coef_[0][0]
    beta_raw = beta_scaled / scaler.scale_[0]
    alpha_raw = clf.intercept_[0] - beta_scaled * scaler.mean_[0] / scaler.scale_[0]

    coef_list.append(beta_raw)
    intercept_list.append(alpha_raw)

beta = float(np.mean(coef_list))
alpha = float(np.mean(intercept_list))

df["P_GenE_DeltaU"] = probs

# Logistic susceptibility:
# P = sigmoid(alpha + beta U)
# dP/dU = beta * P * (1-P)
# use absolute susceptibility
df["chi_DeltaU"] = np.abs(beta * df["P_GenE_DeltaU"] * (1 - df["P_GenE_DeltaU"]))

# theoretical max at P=0.5
chi_max = abs(beta) * 0.25

# ============================================================
# 2. Local finite-difference susceptibility by U bins
# ============================================================

df["U_bin"] = pd.qcut(df[U_COL], q=12, duplicates="drop")

bin_summary = (
    df.groupby("U_bin", observed=False)
    .agg(
        n=(U_COL, "count"),
        U_mean=(U_COL, "mean"),
        U_min=(U_COL, "min"),
        U_max=(U_COL, "max"),
        GenE_rate=("gen_negative_proxy", "mean"),
        P_mean=("P_GenE_DeltaU", "mean"),
        chi_mean=("chi_DeltaU", "mean"),
        inside_band=("inside_band", "mean"),
    )
    .reset_index()
)

# finite difference dP/dU across bins
bin_summary["finite_chi"] = np.nan
for i in range(1, len(bin_summary) - 1):
    dp = bin_summary.loc[i + 1, "GenE_rate"] - bin_summary.loc[i - 1, "GenE_rate"]
    du = bin_summary.loc[i + 1, "U_mean"] - bin_summary.loc[i - 1, "U_mean"]
    bin_summary.loc[i, "finite_chi"] = abs(dp / du) if abs(du) > 1e-12 else np.nan

# ============================================================
# 3. Band susceptibility: inside vs outside
# ============================================================

df["band_region"] = np.where(
    df[U_COL] < U_MINUS, "below_band",
    np.where(df[U_COL] > U_PLUS, "above_band", "inside_band")
)

region_summary = (
    df.groupby("band_region")
    .agg(
        n=(U_COL, "count"),
        U_mean=(U_COL, "mean"),
        U_std=(U_COL, "std"),
        GenE_rate=("gen_negative_proxy", "mean"),
        P_mean=("P_GenE_DeltaU", "mean"),
        chi_mean=("chi_DeltaU", "mean"),
        chi_max=("chi_DeltaU", "max"),
        slope_abs_mean=("slope_abs_mean", "mean"),
        band_residence=("band_residence", "mean"),
    )
    .reset_index()
)

phase_summary = (
    df.groupby("phase")
    .agg(
        n=(U_COL, "count"),
        U_mean=(U_COL, "mean"),
        U_std=(U_COL, "std"),
        GenE_rate=("gen_negative_proxy", "mean"),
        P_mean=("P_GenE_DeltaU", "mean"),
        chi_mean=("chi_DeltaU", "mean"),
        chi_max=("chi_DeltaU", "max"),
        inside_band_frac=("inside_band", "mean"),
        band_residence=("band_residence", "mean"),
        slope_abs_mean=("slope_abs_mean", "mean"),
    )
    .reset_index()
)

condition_summary = (
    df.groupby(["condition", "phase"])
    .agg(
        n=(U_COL, "count"),
        U_mean=(U_COL, "mean"),
        U_std=(U_COL, "std"),
        GenE_rate=("gen_negative_proxy", "mean"),
        P_mean=("P_GenE_DeltaU", "mean"),
        chi_mean=("chi_DeltaU", "mean"),
        chi_max=("chi_DeltaU", "max"),
        inside_band_frac=("inside_band", "mean"),
        band_residence=("band_residence", "mean"),
        slope_abs_mean=("slope_abs_mean", "mean"),
    )
    .reset_index()
)

# ============================================================
# 4. Multivariate susceptibility:
# test whether band/scaling features sharpen response
# ============================================================

feature_sets = {
    "DeltaU_only": [U_COL],
    "DeltaU_band": [U_COL, "inside_band", "dist_to_center", "dist_to_nearest_edge"],
    "DeltaU_band_scaling": [
        U_COL,
        "inside_band",
        "dist_to_center",
        "dist_to_nearest_edge",
        "band_residence",
        "slope_abs_mean",
        "slope_var",
        "acc_abs_mean",
    ],
}

pred_rows = []

for fs_name, cols in feature_sets.items():
    cols = [c for c in cols if c in df.columns]
    Xf = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(float)

    preds = np.zeros(len(df), dtype=int)
    ps = np.zeros(len(df), dtype=float)

    for tr, te in gkf.split(Xf, y, groups):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ])
        model.fit(Xf[tr], y[tr])
        preds[te] = model.predict(Xf[te])
        ps[te] = model.predict_proba(Xf[te])[:, 1]

    pred_rows.append({
        "feature_set": fs_name,
        "auc": roc_auc_score(y, ps),
        "acc": accuracy_score(y, preds),
        "macro_f1": f1_score(y, preds, average="macro"),
        "mean_pred_entropy": float(np.mean(-(ps*np.log(ps+1e-12)+(1-ps)*np.log(1-ps+1e-12)))),
    })

prediction_summary = pd.DataFrame(pred_rows)

# ============================================================
# 5. Diagnostics
# ============================================================

crit_chi = phase_summary.loc[phase_summary["phase"] == "critical", "chi_mean"].iloc[0]
pos_chi = phase_summary.loc[phase_summary["phase"] == "positive", "chi_mean"].iloc[0]
neg_chi = phase_summary.loc[phase_summary["phase"] == "negative", "chi_mean"].iloc[0]

inside_chi = region_summary.loc[region_summary["band_region"] == "inside_band", "chi_mean"].iloc[0]
outside_chi = region_summary.loc[region_summary["band_region"] != "inside_band", "chi_mean"].mean()

diagnostics = pd.DataFrame([{
    "beta_response": beta,
    "alpha_response": alpha,
    "chi_theoretical_max": chi_max,
    "critical_has_max_chi": bool(crit_chi > pos_chi and crit_chi > neg_chi),
    "inside_band_chi_gt_outside": bool(inside_chi > outside_chi),
    "inside_band_chi": inside_chi,
    "outside_band_chi_mean": outside_chi,
    "auc_deltaU_only": float(prediction_summary.loc[prediction_summary["feature_set"] == "DeltaU_only", "auc"].iloc[0]),
    "auc_band_scaling": float(prediction_summary.loc[prediction_summary["feature_set"] == "DeltaU_band_scaling", "auc"].iloc[0]),
}])

# ============================================================
# SAVE
# ============================================================

df.to_csv(SAVE_DIR / "ua5b_dataset_with_susceptibility.csv", index=False, encoding="utf-8-sig")
bin_summary.to_csv(SAVE_DIR / "ua5b_bin_susceptibility.csv", index=False, encoding="utf-8-sig")
region_summary.to_csv(SAVE_DIR / "ua5b_region_summary.csv", index=False, encoding="utf-8-sig")
phase_summary.to_csv(SAVE_DIR / "ua5b_phase_summary.csv", index=False, encoding="utf-8-sig")
condition_summary.to_csv(SAVE_DIR / "ua5b_condition_summary.csv", index=False, encoding="utf-8-sig")
prediction_summary.to_csv(SAVE_DIR / "ua5b_prediction_summary.csv", index=False, encoding="utf-8-sig")
diagnostics.to_csv(SAVE_DIR / "ua5b_diagnostics.csv", index=False, encoding="utf-8-sig")

with open(SAVE_DIR / "ua5b_config.json", "w", encoding="utf-8") as f:
    json.dump({
        "input_csv": INPUT_CSV,
        "susceptibility": "chi = |d P(Gen_E) / d DeltaU|",
        "critical_band": {
            "U_minus": U_MINUS,
            "U_center": U_CENTER,
            "U_plus": U_PLUS,
        },
        "success_criteria": {
            "PASS_Lite": "inside band has higher chi than outside band",
            "PASS": "critical phase has highest chi",
            "PASS_Strong": "susceptibility features improve prediction over DeltaU alone"
        }
    }, f, ensure_ascii=False, indent=2)

print("\n========== UA-5B BIN SUSCEPTIBILITY ==========")
print(bin_summary.to_string(index=False))

print("\n========== UA-5B REGION SUMMARY ==========")
print(region_summary.to_string(index=False))

print("\n========== UA-5B PHASE SUMMARY ==========")
print(phase_summary.to_string(index=False))

print("\n========== UA-5B CONDITION SUMMARY ==========")
print(condition_summary.to_string(index=False))

print("\n========== UA-5B PREDICTION SUMMARY ==========")
print(prediction_summary.to_string(index=False))

print("\n========== UA-5B DIAGNOSTICS ==========")
print(diagnostics.to_string(index=False))

print("\n[DONE] Saved to:", SAVE_DIR)