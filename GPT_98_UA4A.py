import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score

# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = r"C:\Windows\System32\ua3a_outputs\ua3a_dataset_with_DeltaU.csv"
SAVE_DIR = Path("./ua4a_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

LAYER_RANGE = list(range(20, 26))
DELTA_FEATURES = [f"dltR_{l}" for l in LAYER_RANGE]

GROUP_COL = "graph_id"
PHASE_ID = {"positive": 0, "critical": 1, "negative": 2}

# From UA-3A
DELTA_U_CRITICAL = 0.399504835673922

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_CSV)

if "phase_id" not in df.columns:
    df["phase_id"] = df["phase"].map(PHASE_ID)

groups = df[GROUP_COL].values

# ============================================================
# RECONSTRUCT LAYERWISE ORDER PARAMETER ΔU(l)
# ============================================================

# Use UA-3A PC1 loadings as global order-parameter direction.
X = df[DELTA_FEATURES].values.astype(float)
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

pca = PCA(n_components=6, random_state=42)
Z = pca.fit_transform(Xs)

pc1 = Z[:, 0]
if df.loc[df["condition"] == "stable_positive", :].assign(tmp_pc1=pc1[df["condition"] == "stable_positive"])["tmp_pc1"].mean() < 0:
    pca.components_[0] *= -1
    pc1 = -pc1

df["DeltaU_global"] = pc1

pc1_loadings = pca.components_[0]

# Layerwise contribution:
# ΔU_l = standardized dltR_l * PC1_loading_l
for i, l in enumerate(LAYER_RANGE):
    df[f"DeltaU_contrib_{l}"] = Xs[:, i] * pc1_loadings[i]

# Cumulative order parameter trajectory
running = np.zeros(len(df))
for l in LAYER_RANGE:
    running = running + df[f"DeltaU_contrib_{l}"].values
    df[f"DeltaU_cum_{l}"] = running

# Normalize cumulative trajectory to same scale as global PC1 by linear fit
A = df[[f"DeltaU_cum_{l}" for l in LAYER_RANGE]].values
final_cum = df[f"DeltaU_cum_{LAYER_RANGE[-1]}"].values
global_u = df["DeltaU_global"].values

scale = np.dot(final_cum, global_u) / (np.dot(final_cum, final_cum) + 1e-12)
for l in LAYER_RANGE:
    df[f"DeltaU_cum_scaled_{l}"] = df[f"DeltaU_cum_{l}"] * scale

# Distance to critical threshold
for l in LAYER_RANGE:
    df[f"dist_to_Uc_{l}"] = df[f"DeltaU_cum_scaled_{l}"] - DELTA_U_CRITICAL
    df[f"abs_dist_to_Uc_{l}"] = np.abs(df[f"dist_to_Uc_{l}"])

# Estimate tau: first layer where DeltaU_cum_scaled crosses Uc.
def estimate_tau(row):
    prev = None
    for l in LAYER_RANGE:
        val = row[f"DeltaU_cum_scaled_{l}"]
        if val >= DELTA_U_CRITICAL:
            return l
        prev = val
    return np.nan

df["tau_hat"] = df.apply(estimate_tau, axis=1)

# Also define signed phase crossing:
# For negative/conflict, lower ΔU means conflict basin.
# So use critical threshold for clean basin entry, and sign zero as basin boundary.
def estimate_zero_cross(row):
    for l in LAYER_RANGE:
        if row[f"DeltaU_cum_scaled_{l}"] < 0:
            return l
    return np.nan

df["tau_zero_cross"] = df.apply(estimate_zero_cross, axis=1)

# ============================================================
# SUMMARY TABLES
# ============================================================

trajectory_rows = []

for group_cols in [["phase"], ["condition", "phase"]]:
    grouped = df.groupby(group_cols)
    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        base = dict(zip(group_cols, keys))
        for l in LAYER_RANGE:
            row = dict(base)
            row.update({
                "layer": l,
                "n": len(sub),
                "DeltaU_mean": sub[f"DeltaU_cum_scaled_{l}"].mean(),
                "DeltaU_std": sub[f"DeltaU_cum_scaled_{l}"].std(),
                "dist_to_Uc_mean": sub[f"dist_to_Uc_{l}"].mean(),
                "abs_dist_to_Uc_mean": sub[f"abs_dist_to_Uc_{l}"].mean(),
                "cross_Uc_frac": (sub[f"DeltaU_cum_scaled_{l}"] >= DELTA_U_CRITICAL).mean(),
                "cross_zero_frac": (sub[f"DeltaU_cum_scaled_{l}"] < 0).mean(),
            })
            trajectory_rows.append(row)

trajectory_summary = pd.DataFrame(trajectory_rows)

tau_summary = (
    df.groupby(["condition", "phase"])
    .agg(
        n=("graph_id", "count"),
        DeltaU_global_mean=("DeltaU_global", "mean"),
        DeltaU_global_std=("DeltaU_global", "std"),
        tau_hat_mean=("tau_hat", "mean"),
        tau_hat_frac=("tau_hat", lambda x: x.notna().mean()),
        tau_zero_cross_mean=("tau_zero_cross", "mean"),
        tau_zero_cross_frac=("tau_zero_cross", lambda x: x.notna().mean()),
        final_R=("target_final_R", "mean"),
        gen_neg=("gen_negative_proxy", "mean"),
    )
    .reset_index()
)

phase_tau_summary = (
    df.groupby("phase")
    .agg(
        n=("graph_id", "count"),
        DeltaU_global_mean=("DeltaU_global", "mean"),
        DeltaU_global_std=("DeltaU_global", "std"),
        tau_hat_mean=("tau_hat", "mean"),
        tau_hat_frac=("tau_hat", lambda x: x.notna().mean()),
        tau_zero_cross_mean=("tau_zero_cross", "mean"),
        tau_zero_cross_frac=("tau_zero_cross", lambda x: x.notna().mean()),
        final_R=("target_final_R", "mean"),
        gen_neg=("gen_negative_proxy", "mean"),
    )
    .reset_index()
)

pc1_summary = pd.DataFrame({
    "feature": DELTA_FEATURES,
    "layer": LAYER_RANGE,
    "pc1_loading": pc1_loadings,
    "abs_loading": np.abs(pc1_loadings),
    "explained_variance_ratio_PC1": [pca.explained_variance_ratio_[0]] * len(LAYER_RANGE),
})

variance_summary = pd.DataFrame({
    "component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
})

# ============================================================
# CRITICAL SCALING FEATURES
# ============================================================

# Approximate slope and acceleration of ΔU(l)
for l1, l2 in zip(LAYER_RANGE[:-1], LAYER_RANGE[1:]):
    df[f"dDeltaU_{l1}_{l2}"] = df[f"DeltaU_cum_scaled_{l2}"] - df[f"DeltaU_cum_scaled_{l1}"]

for l1, l2, l3 in zip(LAYER_RANGE[:-2], LAYER_RANGE[1:-1], LAYER_RANGE[2:]):
    df[f"ddDeltaU_{l1}_{l3}"] = df[f"dDeltaU_{l2}_{l3}"] - df[f"dDeltaU_{l1}_{l2}"]

df["max_abs_slope"] = df[[f"dDeltaU_{l1}_{l2}" for l1, l2 in zip(LAYER_RANGE[:-1], LAYER_RANGE[1:])]].abs().max(axis=1)
df["mean_slope"] = df[[f"dDeltaU_{l1}_{l2}" for l1, l2 in zip(LAYER_RANGE[:-1], LAYER_RANGE[1:])]].mean(axis=1)
df["min_abs_dist_to_Uc"] = df[[f"abs_dist_to_Uc_{l}" for l in LAYER_RANGE]].min(axis=1)

scaling_summary = (
    df.groupby(["condition", "phase"])
    .agg(
        n=("graph_id", "count"),
        max_abs_slope=("max_abs_slope", "mean"),
        mean_slope=("mean_slope", "mean"),
        min_abs_dist_to_Uc=("min_abs_dist_to_Uc", "mean"),
        DeltaU_global=("DeltaU_global", "mean"),
        final_R=("target_final_R", "mean"),
        gen_neg=("gen_negative_proxy", "mean"),
    )
    .reset_index()
)

# ============================================================
# PREDICTION TESTS
# ============================================================

prediction_rows = []

FEATURE_SETS = {
    "DeltaU_global": ["DeltaU_global"],
    "DeltaU_cum_20_25": [f"DeltaU_cum_scaled_{l}" for l in LAYER_RANGE],
    "critical_scaling": [
        "DeltaU_global",
        "max_abs_slope",
        "mean_slope",
        "min_abs_dist_to_Uc",
    ],
    "raw_dltR_20_25": DELTA_FEATURES,
}

REG_TARGETS = [
    "target_final_R",
    "target_final_dltR",
    "target_basin_dltR_mean",
]

CLS_TARGETS = [
    "phase_id",
    "gen_negative_proxy",
]

def reg_cv(cols, target):
    Xf = df[cols].values.astype(float)
    y = df[target].values.astype(float)
    preds = np.zeros(len(df))

    gkf = GroupKFold(n_splits=6)
    for tr, te in gkf.split(Xf, y, groups):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ])
        model.fit(Xf[tr], y[tr])
        preds[te] = model.predict(Xf[te])

    return {
        "r2": r2_score(y, preds),
        "corr": np.corrcoef(y, preds)[0, 1] if np.std(preds) > 1e-8 else 0,
    }

def cls_cv(cols, target):
    Xf = df[cols].values.astype(float)
    y = df[target].values.astype(int)
    preds = np.zeros(len(df), dtype=int)

    binary = len(np.unique(y)) == 2
    probs = np.zeros(len(df)) if binary else None

    gkf = GroupKFold(n_splits=6)
    for tr, te in gkf.split(Xf, y, groups):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])
        model.fit(Xf[tr], y[tr])
        preds[te] = model.predict(Xf[te])
        if binary:
            probs[te] = model.predict_proba(Xf[te])[:, 1]

    out = {
        "acc": accuracy_score(y, preds),
        "macro_f1": f1_score(y, preds, average="macro"),
        "auc": np.nan,
    }

    if binary:
        try:
            out["auc"] = roc_auc_score(y, probs)
        except Exception:
            pass

    return out

for fs_name, cols in FEATURE_SETS.items():
    for target in REG_TARGETS:
        out = reg_cv(cols, target)
        prediction_rows.append({
            "task": "regression",
            "feature_set": fs_name,
            "target": target,
            **out,
            "acc": np.nan,
            "macro_f1": np.nan,
            "auc": np.nan,
        })

    for target in CLS_TARGETS:
        out = cls_cv(cols, target)
        prediction_rows.append({
            "task": "classification",
            "feature_set": fs_name,
            "target": target,
            "r2": np.nan,
            "corr": np.nan,
            **out,
        })

prediction_summary = pd.DataFrame(prediction_rows)

# ============================================================
# SAVE
# ============================================================

df.to_csv(SAVE_DIR / "ua4a_dataset_with_critical_scaling.csv", index=False, encoding="utf-8-sig")
trajectory_summary.to_csv(SAVE_DIR / "ua4a_trajectory_summary.csv", index=False, encoding="utf-8-sig")
tau_summary.to_csv(SAVE_DIR / "ua4a_tau_summary.csv", index=False, encoding="utf-8-sig")
phase_tau_summary.to_csv(SAVE_DIR / "ua4a_phase_tau_summary.csv", index=False, encoding="utf-8-sig")
pc1_summary.to_csv(SAVE_DIR / "ua4a_pc1_summary.csv", index=False, encoding="utf-8-sig")
variance_summary.to_csv(SAVE_DIR / "ua4a_variance_summary.csv", index=False, encoding="utf-8-sig")
scaling_summary.to_csv(SAVE_DIR / "ua4a_scaling_summary.csv", index=False, encoding="utf-8-sig")
prediction_summary.to_csv(SAVE_DIR / "ua4a_prediction_summary.csv", index=False, encoding="utf-8-sig")

with open(SAVE_DIR / "ua4a_config.json", "w", encoding="utf-8") as f:
    json.dump({
        "input_csv": INPUT_CSV,
        "delta_features": DELTA_FEATURES,
        "DeltaU_critical": DELTA_U_CRITICAL,
        "hypothesis": "DeltaU behaves like an order parameter with critical crossing around tau.",
        "success_criteria": {
            "PASS_Lite": "DeltaU trajectories show phase-ordered scaling.",
            "PASS": "critical phase approaches Uc while positive/negative remain separated.",
            "PASS_Strong": "tau_hat / distance-to-Uc predicts basin flip and phase."
        }
    }, f, ensure_ascii=False, indent=2)

# ============================================================
# PRINT
# ============================================================

print("\n========== UA-4A VARIANCE ==========")
print(variance_summary.to_string(index=False))

print("\n========== UA-4A PC1 SUMMARY ==========")
print(pc1_summary.to_string(index=False))

print("\n========== UA-4A PHASE TAU SUMMARY ==========")
print(phase_tau_summary.to_string(index=False))

print("\n========== UA-4A TAU SUMMARY ==========")
print(tau_summary.to_string(index=False))

print("\n========== UA-4A SCALING SUMMARY ==========")
print(scaling_summary.to_string(index=False))

print("\n========== UA-4A PREDICTION SUMMARY ==========")
print(prediction_summary.to_string(index=False))

print("\n[DONE] Saved to:", SAVE_DIR)