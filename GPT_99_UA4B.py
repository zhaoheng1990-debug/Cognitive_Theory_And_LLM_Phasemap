import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

INPUT_CSV = r"C:\Windows\System32\ua4a_outputs\ua4a_dataset_with_critical_scaling.csv"
SAVE_DIR = Path("./ua4b_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

LAYER_RANGE = list(range(20, 26))
U_COL = "DeltaU_global"
GROUP_COL = "graph_id"

df = pd.read_csv(INPUT_CSV)

# ============================================================
# 1. Estimate critical band [U-, U+]
# ============================================================

phase_mean = df.groupby("phase")[U_COL].mean()

u_pos = phase_mean.get("positive")
u_crit = phase_mean.get("critical")
u_neg = phase_mean.get("negative")

# band edges: midpoints between negative-critical and critical-positive
U_minus = (u_neg + u_crit) / 2
U_plus = (u_crit + u_pos) / 2

df["in_band_global"] = ((df[U_COL] >= U_minus) & (df[U_COL] <= U_plus)).astype(int)

# phase prediction by band
def band_phase(u):
    if u > U_plus:
        return "positive"
    elif u < U_minus:
        return "negative"
    else:
        return "critical"

df["phase_pred_band"] = df[U_COL].apply(band_phase)

# ============================================================
# 2. Layerwise band escape tau
# ============================================================

for l in LAYER_RANGE:
    col = f"DeltaU_cum_scaled_{l}"
    df[f"in_band_{l}"] = ((df[col] >= U_minus) & (df[col] <= U_plus)).astype(int)
    df[f"above_band_{l}"] = (df[col] > U_plus).astype(int)
    df[f"below_band_{l}"] = (df[col] < U_minus).astype(int)

def tau_escape(row):
    for l in LAYER_RANGE:
        if row[f"in_band_{l}"] == 0:
            return l
    return np.nan

def tau_enter_single_basin(row):
    for l in LAYER_RANGE:
        if row[f"above_band_{l}"] == 1:
            return l
        if row[f"below_band_{l}"] == 1:
            return l
    return np.nan

df["tau_band_escape"] = df.apply(tau_escape, axis=1)
df["tau_single_basin"] = df.apply(tau_enter_single_basin, axis=1)

# ============================================================
# 3. Critical-band properties
# ============================================================

# sensitivity proxy: variance / slope / distance to band edge
slope_cols = [f"dDeltaU_{a}_{b}" for a, b in zip(LAYER_RANGE[:-1], LAYER_RANGE[1:]) if f"dDeltaU_{a}_{b}" in df.columns]

df["dist_to_band"] = df[U_COL].apply(lambda u: 0 if U_minus <= u <= U_plus else min(abs(u-U_minus), abs(u-U_plus)))
df["signed_band_position"] = (df[U_COL] - u_crit) / (U_plus - U_minus + 1e-12)

if slope_cols:
    df["slope_abs_mean"] = df[slope_cols].abs().mean(axis=1)
    df["slope_abs_max"] = df[slope_cols].abs().max(axis=1)
else:
    df["slope_abs_mean"] = np.nan
    df["slope_abs_max"] = np.nan

# local uncertainty / mixedness
if "gen_negative_proxy" in df.columns:
    y = df["gen_negative_proxy"].astype(int)
else:
    y = (df["target_final_R"] < 0).astype(int)
    df["gen_negative_proxy"] = y

band_summary = pd.DataFrame([{
    "U_minus": U_minus,
    "U_critical_center": u_crit,
    "U_plus": U_plus,
    "band_width": U_plus - U_minus,
    "positive_mean": u_pos,
    "critical_mean": u_crit,
    "negative_mean": u_neg,
}])

phase_band_summary = (
    df.groupby("phase")
    .agg(
        n=(U_COL, "count"),
        DeltaU_mean=(U_COL, "mean"),
        DeltaU_std=(U_COL, "std"),
        in_band_frac=("in_band_global", "mean"),
        gen_negative_rate=("gen_negative_proxy", "mean"),
        tau_band_escape_mean=("tau_band_escape", "mean"),
        tau_band_escape_frac=("tau_band_escape", lambda x: x.notna().mean()),
        slope_abs_mean=("slope_abs_mean", "mean"),
        dist_to_band=("dist_to_band", "mean"),
    )
    .reset_index()
)

condition_band_summary = (
    df.groupby(["condition", "phase"])
    .agg(
        n=(U_COL, "count"),
        DeltaU_mean=(U_COL, "mean"),
        DeltaU_std=(U_COL, "std"),
        in_band_frac=("in_band_global", "mean"),
        gen_negative_rate=("gen_negative_proxy", "mean"),
        tau_band_escape_mean=("tau_band_escape", "mean"),
        tau_band_escape_frac=("tau_band_escape", lambda x: x.notna().mean()),
        slope_abs_mean=("slope_abs_mean", "mean"),
        dist_to_band=("dist_to_band", "mean"),
    )
    .reset_index()
)

trajectory_band_rows = []
for phase, sub in df.groupby("phase"):
    for l in LAYER_RANGE:
        trajectory_band_rows.append({
            "phase": phase,
            "layer": l,
            "n": len(sub),
            "DeltaU_mean": sub[f"DeltaU_cum_scaled_{l}"].mean(),
            "DeltaU_std": sub[f"DeltaU_cum_scaled_{l}"].std(),
            "in_band_frac": sub[f"in_band_{l}"].mean(),
            "above_band_frac": sub[f"above_band_{l}"].mean(),
            "below_band_frac": sub[f"below_band_{l}"].mean(),
        })

trajectory_band_summary = pd.DataFrame(trajectory_band_rows)

# ============================================================
# 4. Compare single threshold vs band classifier
# ============================================================

phase_true = df["phase"].astype(str)
phase_pred = df["phase_pred_band"].astype(str)

band_classification_summary = pd.DataFrame([{
    "method": "global_band_midpoint",
    "phase_acc": accuracy_score(phase_true, phase_pred),
    "phase_macro_f1": f1_score(phase_true, phase_pred, average="macro"),
    "U_minus": U_minus,
    "U_plus": U_plus,
}])

# Binary prediction: inside band vs outside band; Gen negative is expected highest below band, mixed inside
# Use features to see if band variables predict gen flip.
feature_sets = {
    "DeltaU_only": [U_COL],
    "band_features": [U_COL, "in_band_global", "dist_to_band", "signed_band_position"],
    "band_plus_scaling": [U_COL, "in_band_global", "dist_to_band", "signed_band_position", "slope_abs_mean", "slope_abs_max"],
}

pred_rows = []
groups = df[GROUP_COL].values

for fs_name, cols in feature_sets.items():
    cols = [c for c in cols if c in df.columns and df[c].notna().all()]
    X = df[cols].values.astype(float)
    y = df["gen_negative_proxy"].values.astype(int)

    preds = np.zeros(len(df), dtype=int)
    probs = np.zeros(len(df), dtype=float)

    gkf = GroupKFold(n_splits=6)
    for tr, te in gkf.split(X, y, groups):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])
        model.fit(X[tr], y[tr])
        preds[te] = model.predict(X[te])
        probs[te] = model.predict_proba(X[te])[:, 1]

    pred_rows.append({
        "feature_set": fs_name,
        "target": "gen_negative_proxy",
        "acc": accuracy_score(y, preds),
        "macro_f1": f1_score(y, preds, average="macro"),
        "auc": roc_auc_score(y, probs),
    })

prediction_summary = pd.DataFrame(pred_rows)

# ============================================================
# 5. Save
# ============================================================

df.to_csv(SAVE_DIR / "ua4b_dataset_with_critical_band.csv", index=False, encoding="utf-8-sig")
band_summary.to_csv(SAVE_DIR / "ua4b_band_summary.csv", index=False, encoding="utf-8-sig")
phase_band_summary.to_csv(SAVE_DIR / "ua4b_phase_band_summary.csv", index=False, encoding="utf-8-sig")
condition_band_summary.to_csv(SAVE_DIR / "ua4b_condition_band_summary.csv", index=False, encoding="utf-8-sig")
trajectory_band_summary.to_csv(SAVE_DIR / "ua4b_trajectory_band_summary.csv", index=False, encoding="utf-8-sig")
band_classification_summary.to_csv(SAVE_DIR / "ua4b_band_classification_summary.csv", index=False, encoding="utf-8-sig")
prediction_summary.to_csv(SAVE_DIR / "ua4b_prediction_summary.csv", index=False, encoding="utf-8-sig")

with open(SAVE_DIR / "ua4b_config.json", "w", encoding="utf-8") as f:
    json.dump({
        "input_csv": INPUT_CSV,
        "order_parameter": U_COL,
        "critical_band": {
            "U_minus": float(U_minus),
            "U_center": float(u_crit),
            "U_plus": float(U_plus),
            "width": float(U_plus - U_minus),
        },
        "hypothesis": "Critical phase is a band [U-, U+] rather than a single threshold.",
        "success_criteria": {
            "PASS_Lite": "critical phase has highest in_band_frac",
            "PASS": "band classifier separates positive/critical/negative better than single threshold intuition",
            "PASS_Strong": "band membership and escape tau explain basin flip / phase transition"
        }
    }, f, ensure_ascii=False, indent=2)

print("\n========== UA-4B BAND SUMMARY ==========")
print(band_summary.to_string(index=False))

print("\n========== UA-4B PHASE BAND SUMMARY ==========")
print(phase_band_summary.to_string(index=False))

print("\n========== UA-4B CONDITION BAND SUMMARY ==========")
print(condition_band_summary.to_string(index=False))

print("\n========== UA-4B TRAJECTORY BAND SUMMARY ==========")
print(trajectory_band_summary.to_string(index=False))

print("\n========== UA-4B BAND CLASSIFICATION ==========")
print(band_classification_summary.to_string(index=False))

print("\n========== UA-4B PREDICTION SUMMARY ==========")
print(prediction_summary.to_string(index=False))

print("\n[DONE] Saved to:", SAVE_DIR)