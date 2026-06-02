import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, r2_score
)

INPUT_CSV = r"C:\Windows\System32\ua4b_outputs\ua4b_dataset_with_critical_band.csv"
SAVE_DIR = Path("./ua5a_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

LAYER_RANGE = list(range(20, 26))
U_COL = "DeltaU_global"
GROUP_COL = "graph_id"

df = pd.read_csv(INPUT_CSV)

# Critical band from UA-4B
with open("./ua4b_outputs/ua4b_config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

U_MINUS = cfg["critical_band"]["U_minus"]
U_CENTER = cfg["critical_band"]["U_center"]
U_PLUS = cfg["critical_band"]["U_plus"]
BAND_WIDTH = cfg["critical_band"]["width"]

# ============================================================
# 1. Critical distance / susceptibility proxies
# ============================================================

df["dist_to_center"] = (df[U_COL] - U_CENTER).abs()
df["dist_to_lower"] = (df[U_COL] - U_MINUS).abs()
df["dist_to_upper"] = (df[U_COL] - U_PLUS).abs()
df["dist_to_nearest_edge"] = df.apply(
    lambda r: 0.0 if U_MINUS <= r[U_COL] <= U_PLUS
    else min(abs(r[U_COL] - U_MINUS), abs(r[U_COL] - U_PLUS)),
    axis=1
)

df["inside_band"] = ((df[U_COL] >= U_MINUS) & (df[U_COL] <= U_PLUS)).astype(int)

# susceptibility proxy: high when close to center / band
eps = 1e-6
df["chi_center_proxy"] = 1.0 / (df["dist_to_center"] + eps)
df["chi_edge_proxy"] = 1.0 / (df["dist_to_nearest_edge"] + eps)

# layerwise slopes
slope_cols = []
for a, b in zip(LAYER_RANGE[:-1], LAYER_RANGE[1:]):
    col = f"dDeltaU_{a}_{b}"
    if col not in df.columns:
        df[col] = df[f"DeltaU_cum_scaled_{b}"] - df[f"DeltaU_cum_scaled_{a}"]
    slope_cols.append(col)

df["slope_mean"] = df[slope_cols].mean(axis=1)
df["slope_abs_mean"] = df[slope_cols].abs().mean(axis=1)
df["slope_abs_max"] = df[slope_cols].abs().max(axis=1)
df["slope_var"] = df[slope_cols].var(axis=1)

# acceleration / curvature
acc_cols = []
for a, b, c in zip(LAYER_RANGE[:-2], LAYER_RANGE[1:-1], LAYER_RANGE[2:]):
    col = f"ddDeltaU_{a}_{c}"
    df[col] = df[f"dDeltaU_{b}_{c}"] - df[f"dDeltaU_{a}_{b}"]
    acc_cols.append(col)

df["acc_abs_mean"] = df[acc_cols].abs().mean(axis=1)
df["acc_abs_max"] = df[acc_cols].abs().max(axis=1)

# ============================================================
# 2. Critical slowing down proxy
# If near critical band, escape should be delayed / trajectory should linger.
# ============================================================

def last_in_band(row):
    last = np.nan
    for l in LAYER_RANGE:
        col = f"in_band_{l}"
        if col in df.columns and row[col] == 1:
            last = l
    return last

def band_residence(row):
    return sum(row.get(f"in_band_{l}", 0) for l in LAYER_RANGE)

df["last_in_band_layer"] = df.apply(last_in_band, axis=1)
df["band_residence"] = df.apply(band_residence, axis=1)

# ============================================================
# 3. Layerwise variance / fluctuation peak
# ============================================================

var_rows = []
for phase, sub in df.groupby("phase"):
    for l in LAYER_RANGE:
        vals = sub[f"DeltaU_cum_scaled_{l}"].values
        var_rows.append({
            "phase": phase,
            "layer": l,
            "n": len(sub),
            "U_mean": np.mean(vals),
            "U_std": np.std(vals),
            "U_var": np.var(vals),
            "in_band_frac": sub[f"in_band_{l}"].mean() if f"in_band_{l}" in sub.columns else np.nan,
            "above_band_frac": sub[f"above_band_{l}"].mean() if f"above_band_{l}" in sub.columns else np.nan,
            "below_band_frac": sub[f"below_band_{l}"].mean() if f"below_band_{l}" in sub.columns else np.nan,
        })

variance_by_layer = pd.DataFrame(var_rows)

# ============================================================
# 4. Phase-level critical phenomena summary
# ============================================================

phase_summary = (
    df.groupby("phase")
    .agg(
        n=(U_COL, "count"),
        DeltaU_mean=(U_COL, "mean"),
        DeltaU_std=(U_COL, "std"),
        inside_band_frac=("inside_band", "mean"),
        band_residence_mean=("band_residence", "mean"),
        last_in_band_mean=("last_in_band_layer", "mean"),
        slope_abs_mean=("slope_abs_mean", "mean"),
        slope_var_mean=("slope_var", "mean"),
        acc_abs_mean=("acc_abs_mean", "mean"),
        chi_center_proxy_mean=("chi_center_proxy", "mean"),
        dist_to_center_mean=("dist_to_center", "mean"),
        dist_to_nearest_edge_mean=("dist_to_nearest_edge", "mean"),
        gen_negative_rate=("gen_negative_proxy", "mean"),
    )
    .reset_index()
)

condition_summary = (
    df.groupby(["condition", "phase"])
    .agg(
        n=(U_COL, "count"),
        DeltaU_mean=(U_COL, "mean"),
        DeltaU_std=(U_COL, "std"),
        inside_band_frac=("inside_band", "mean"),
        band_residence_mean=("band_residence", "mean"),
        last_in_band_mean=("last_in_band_layer", "mean"),
        slope_abs_mean=("slope_abs_mean", "mean"),
        slope_var_mean=("slope_var", "mean"),
        acc_abs_mean=("acc_abs_mean", "mean"),
        chi_center_proxy_mean=("chi_center_proxy", "mean"),
        dist_to_center_mean=("dist_to_center", "mean"),
        dist_to_nearest_edge_mean=("dist_to_nearest_edge", "mean"),
        gen_negative_rate=("gen_negative_proxy", "mean"),
    )
    .reset_index()
)

# ============================================================
# 5. Predictive test:
# Does adding critical phenomena features improve over DeltaU alone?
# ============================================================

feature_sets = {
    "DeltaU_only": [U_COL],
    "band_only": [U_COL, "inside_band", "dist_to_center", "dist_to_nearest_edge"],
    "critical_phenomena": [
        U_COL,
        "inside_band",
        "dist_to_center",
        "dist_to_nearest_edge",
        "band_residence",
        "last_in_band_layer",
        "slope_abs_mean",
        "slope_var",
        "acc_abs_mean",
        "chi_center_proxy",
    ],
}

targets_cls = ["gen_negative_proxy", "phase_id"] if "phase_id" in df.columns else ["gen_negative_proxy"]
targets_reg = ["target_final_R", "target_final_dltR", "target_basin_dltR_mean"]

pred_rows = []
groups = df[GROUP_COL].values

def cls_cv(cols, target):
    X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(float)
    y = df[target].values.astype(int)
    gkf = GroupKFold(n_splits=6)

    preds = np.zeros(len(df), dtype=int)
    binary = len(np.unique(y)) == 2
    probs = np.zeros(len(df)) if binary else None

    for tr, te in gkf.split(X, y, groups):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ])
        model.fit(X[tr], y[tr])
        preds[te] = model.predict(X[te])
        if binary:
            probs[te] = model.predict_proba(X[te])[:, 1]

    out = {
        "acc": accuracy_score(y, preds),
        "macro_f1": f1_score(y, preds, average="macro"),
        "auc": np.nan,
    }
    if binary:
        out["auc"] = roc_auc_score(y, probs)
    return out

def reg_cv(cols, target):
    X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(float)
    y = df[target].values.astype(float)
    gkf = GroupKFold(n_splits=6)

    preds = np.zeros(len(df))
    for tr, te in gkf.split(X, y, groups):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ])
        model.fit(X[tr], y[tr])
        preds[te] = model.predict(X[te])

    corr = np.corrcoef(y, preds)[0, 1] if np.std(preds) > 1e-8 else 0
    return {
        "r2": r2_score(y, preds),
        "corr": corr,
    }

for fs_name, cols in feature_sets.items():
    cols = [c for c in cols if c in df.columns]

    for target in targets_cls:
        out = cls_cv(cols, target)
        pred_rows.append({
            "task": "classification",
            "feature_set": fs_name,
            "target": target,
            "r2": np.nan,
            "corr": np.nan,
            **out,
        })

    for target in targets_reg:
        if target in df.columns:
            out = reg_cv(cols, target)
            pred_rows.append({
                "task": "regression",
                "feature_set": fs_name,
                "target": target,
                **out,
                "acc": np.nan,
                "macro_f1": np.nan,
                "auc": np.nan,
            })

prediction_summary = pd.DataFrame(pred_rows)

# ============================================================
# 6. Simple PASS diagnostics
# ============================================================

crit = phase_summary[phase_summary["phase"] == "critical"].iloc[0]
pos = phase_summary[phase_summary["phase"] == "positive"].iloc[0]
neg = phase_summary[phase_summary["phase"] == "negative"].iloc[0]

diagnostics = pd.DataFrame([{
    "critical_has_max_inside_band": bool(
        crit["inside_band_frac"] > pos["inside_band_frac"]
        and crit["inside_band_frac"] > neg["inside_band_frac"]
    ),
    "critical_has_max_residence": bool(
        crit["band_residence_mean"] > pos["band_residence_mean"]
        and crit["band_residence_mean"] > neg["band_residence_mean"]
    ),
    "critical_has_max_variance": bool(
        crit["DeltaU_std"] > pos["DeltaU_std"]
        and crit["DeltaU_std"] > neg["DeltaU_std"]
    ),
    "critical_mixed_generation": float(crit["gen_negative_rate"]),
    "band_width": BAND_WIDTH,
    "U_minus": U_MINUS,
    "U_center": U_CENTER,
    "U_plus": U_PLUS,
}])

# ============================================================
# SAVE
# ============================================================

df.to_csv(SAVE_DIR / "ua5a_dataset_with_critical_phenomena.csv", index=False, encoding="utf-8-sig")
phase_summary.to_csv(SAVE_DIR / "ua5a_phase_summary.csv", index=False, encoding="utf-8-sig")
condition_summary.to_csv(SAVE_DIR / "ua5a_condition_summary.csv", index=False, encoding="utf-8-sig")
variance_by_layer.to_csv(SAVE_DIR / "ua5a_variance_by_layer.csv", index=False, encoding="utf-8-sig")
prediction_summary.to_csv(SAVE_DIR / "ua5a_prediction_summary.csv", index=False, encoding="utf-8-sig")
diagnostics.to_csv(SAVE_DIR / "ua5a_diagnostics.csv", index=False, encoding="utf-8-sig")

with open(SAVE_DIR / "ua5a_config.json", "w", encoding="utf-8") as f:
    json.dump({
        "input_csv": INPUT_CSV,
        "critical_band": {
            "U_minus": U_MINUS,
            "U_center": U_CENTER,
            "U_plus": U_PLUS,
            "width": BAND_WIDTH,
        },
        "hypothesis": "Critical band should show critical phenomena: high residence, high variance, mixed generation, and improved prediction from scaling features.",
        "success_criteria": {
            "PASS_Lite": "critical phase has highest band residence or inside-band occupancy",
            "PASS": "critical phase also shows mixed generation and stronger variance/fluctuation",
            "PASS_Strong": "critical phenomena features improve prediction over DeltaU alone"
        }
    }, f, ensure_ascii=False, indent=2)

print("\n========== UA-5A PHASE SUMMARY ==========")
print(phase_summary.to_string(index=False))

print("\n========== UA-5A CONDITION SUMMARY ==========")
print(condition_summary.to_string(index=False))

print("\n========== UA-5A VARIANCE BY LAYER ==========")
print(variance_by_layer.to_string(index=False))

print("\n========== UA-5A PREDICTION SUMMARY ==========")
print(prediction_summary.to_string(index=False))

print("\n========== UA-5A DIAGNOSTICS ==========")
print(diagnostics.to_string(index=False))

print("\n[DONE] Saved to:", SAVE_DIR)