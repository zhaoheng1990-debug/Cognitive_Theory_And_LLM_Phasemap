import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, accuracy_score, f1_score, roc_auc_score

# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = r"C:\Windows\System32\ua2c_outputs\ua2c_dataset_with_basis.csv"
SAVE_DIR = Path("./ua3a_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DELTA_FEATURES = [f"dltR_{l}" for l in range(20, 26)]
GROUP_COL = "graph_id"

PHASE_ID = {"positive": 0, "critical": 1, "negative": 2}

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_CSV)

if "phase_id" not in df.columns:
    df["phase_id"] = df["phase"].map(PHASE_ID)

X_raw = df[DELTA_FEATURES].values.astype(float)
groups = df[GROUP_COL].values

# Orient PC1 so stable_positive is positive
X_scaled = StandardScaler().fit_transform(X_raw)
pca = PCA(n_components=6, random_state=42)
Z = pca.fit_transform(X_scaled)

pc1 = Z[:, 0]

# 先写入全表，再用 stable_positive 子集判断方向
df["DeltaU_PC1"] = pc1

if df.loc[df["condition"] == "stable_positive", "DeltaU_PC1"].mean() < 0:
    df["DeltaU_PC1"] = -df["DeltaU_PC1"]
    pca.components_[0] *= -1

# ============================================================
# ORDER PARAMETER SUMMARIES
# ============================================================

condition_summary = (
    df.groupby(["condition", "phase"])
    .agg(
        n=("DeltaU_PC1", "count"),
        DeltaU_mean=("DeltaU_PC1", "mean"),
        DeltaU_std=("DeltaU_PC1", "std"),
        crit_dltR=("target_crit_dltR_mean", "mean"),
        basin_dltR=("target_basin_dltR_mean", "mean"),
        final_R=("target_final_R", "mean"),
        gen_neg=("gen_negative_proxy", "mean"),
    )
    .reset_index()
)

phase_summary = (
    df.groupby("phase")
    .agg(
        n=("DeltaU_PC1", "count"),
        DeltaU_mean=("DeltaU_PC1", "mean"),
        DeltaU_std=("DeltaU_PC1", "std"),
        crit_dltR=("target_crit_dltR_mean", "mean"),
        basin_dltR=("target_basin_dltR_mean", "mean"),
        final_R=("target_final_R", "mean"),
        gen_neg=("gen_negative_proxy", "mean"),
    )
    .reset_index()
)

variance_summary = pd.DataFrame({
    "component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
})

loadings = pd.DataFrame({
    "feature": DELTA_FEATURES,
    "PC1_loading": pca.components_[0],
    "abs_loading": np.abs(pca.components_[0]),
})

# ============================================================
# PREDICTION TESTS
# ============================================================

REG_TARGETS = [
    "target_crit_dltR_mean",
    "target_basin_dltR_mean",
    "target_final_dltR",
    "target_final_R",
]

CLS_TARGETS = [
    "phase_id",
    "gen_negative_proxy",
]

def reg_cv(feature_cols, target):
    X = df[feature_cols].values.astype(float)
    y = df[target].values.astype(float)

    preds = np.zeros(len(df))
    gkf = GroupKFold(n_splits=6)

    for tr, te in gkf.split(X, y, groups):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ])
        model.fit(X[tr], y[tr])
        preds[te] = model.predict(X[te])

    r2 = r2_score(y, preds)
    corr = np.corrcoef(y, preds)[0, 1] if np.std(preds) > 1e-8 else 0
    return r2, corr

def cls_cv(feature_cols, target):
    X = df[feature_cols].values.astype(float)
    y = df[target].values.astype(int)

    preds = np.zeros(len(df), dtype=int)
    binary = len(np.unique(y)) == 2
    probs = np.zeros(len(df)) if binary else None

    gkf = GroupKFold(n_splits=6)

    for tr, te in gkf.split(X, y, groups):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])
        model.fit(X[tr], y[tr])
        preds[te] = model.predict(X[te])
        if binary:
            probs[te] = model.predict_proba(X[te])[:, 1]

    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="macro")

    auc = np.nan
    if binary:
        try:
            auc = roc_auc_score(y, probs)
        except Exception:
            pass

    return acc, f1, auc

prediction_rows = []

FEATURE_SETS = {
    "DeltaU_PC1_only": ["DeltaU_PC1"],
    "raw_dltR_20_25": DELTA_FEATURES,
    "late_only_dltR_23_25": ["dltR_23", "dltR_24", "dltR_25"],
    "boundary_only_dltR_20_22": ["dltR_20", "dltR_21", "dltR_22"],
}

for fs_name, cols in FEATURE_SETS.items():
    for target in REG_TARGETS:
        r2, corr = reg_cv(cols, target)
        prediction_rows.append({
            "task": "regression",
            "feature_set": fs_name,
            "target": target,
            "r2": r2,
            "corr": corr,
            "acc": np.nan,
            "macro_f1": np.nan,
            "auc": np.nan,
        })

    for target in CLS_TARGETS:
        acc, f1, auc = cls_cv(cols, target)
        prediction_rows.append({
            "task": "classification",
            "feature_set": fs_name,
            "target": target,
            "r2": np.nan,
            "corr": np.nan,
            "acc": acc,
            "macro_f1": f1,
            "auc": auc,
        })

prediction_summary = pd.DataFrame(prediction_rows)

# ============================================================
# CRITICAL THRESHOLD TEST
# ============================================================

# Test whether DeltaU_PC1 has threshold-like behavior for gen_negative_proxy.
threshold_rows = []
y = df["gen_negative_proxy"].values.astype(int)
u = df["DeltaU_PC1"].values.astype(float)

candidate_thresholds = np.quantile(u, np.linspace(0.05, 0.95, 181))

best = None
for th in candidate_thresholds:
    # Lower DeltaU means more negative/conflict
    pred = (u < th).astype(int)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    row = {"threshold": th, "acc": acc, "f1": f1}
    threshold_rows.append(row)
    if best is None or f1 > best["f1"]:
        best = row

threshold_summary = pd.DataFrame(threshold_rows)

# ============================================================
# MONOTONICITY / ORDER TEST
# ============================================================

phase_order = ["positive", "critical", "negative"]
phase_means = phase_summary.set_index("phase").loc[phase_order]["DeltaU_mean"].values

# Expected: positive > critical > negative
monotonic_order_pass = bool(phase_means[0] > phase_means[1] > phase_means[2])

order_summary = pd.DataFrame([{
    "positive_mean": phase_means[0],
    "critical_mean": phase_means[1],
    "negative_mean": phase_means[2],
    "monotonic_positive_gt_critical_gt_negative": monotonic_order_pass,
    "gap_positive_critical": phase_means[0] - phase_means[1],
    "gap_critical_negative": phase_means[1] - phase_means[2],
}])

# ============================================================
# SAVE
# ============================================================

df.to_csv(SAVE_DIR / "ua3a_dataset_with_DeltaU.csv", index=False, encoding="utf-8-sig")
variance_summary.to_csv(SAVE_DIR / "ua3a_variance_summary.csv", index=False, encoding="utf-8-sig")
loadings.to_csv(SAVE_DIR / "ua3a_pc1_loadings.csv", index=False, encoding="utf-8-sig")
condition_summary.to_csv(SAVE_DIR / "ua3a_condition_summary.csv", index=False, encoding="utf-8-sig")
phase_summary.to_csv(SAVE_DIR / "ua3a_phase_summary.csv", index=False, encoding="utf-8-sig")
prediction_summary.to_csv(SAVE_DIR / "ua3a_prediction_summary.csv", index=False, encoding="utf-8-sig")
threshold_summary.to_csv(SAVE_DIR / "ua3a_threshold_sweep.csv", index=False, encoding="utf-8-sig")
order_summary.to_csv(SAVE_DIR / "ua3a_order_summary.csv", index=False, encoding="utf-8-sig")

with open(SAVE_DIR / "ua3a_config.json", "w", encoding="utf-8") as f:
    json.dump({
        "input_csv": INPUT_CSV,
        "delta_features": DELTA_FEATURES,
        "order_parameter": "DeltaU_PC1 = oriented PC1(dltR_20...dltR_25)",
        "hypothesis": "DeltaU_PC1 is the potential order parameter.",
        "success_criteria": {
            "PASS_Lite": "PC1 explains >70% variance and phase means are monotonic.",
            "PASS": "PC1 explains >80%, phase means monotonic, DeltaU predicts critical/final targets.",
            "PASS_Strong": "PC1 explains >85%, phase means monotonic, threshold separates basin flip."
        },
        "best_threshold": best,
    }, f, ensure_ascii=False, indent=2)

# ============================================================
# PRINT
# ============================================================

print("\n========== UA-3A VARIANCE ==========")
print(variance_summary.to_string(index=False))

print("\n========== UA-3A PC1 LOADINGS ==========")
print(loadings.to_string(index=False))

print("\n========== UA-3A PHASE SUMMARY ==========")
print(phase_summary.to_string(index=False))

print("\n========== UA-3A CONDITION SUMMARY ==========")
print(condition_summary.to_string(index=False))

print("\n========== UA-3A PREDICTION SUMMARY ==========")
print(
    prediction_summary.sort_values(
        ["task", "target", "r2", "macro_f1"],
        ascending=[True, True, False, False],
    ).to_string(index=False)
)

print("\n========== UA-3A ORDER SUMMARY ==========")
print(order_summary.to_string(index=False))

print("\n========== UA-3A BEST THRESHOLD ==========")
print(best)

print("\n[DONE] Saved to:", SAVE_DIR)