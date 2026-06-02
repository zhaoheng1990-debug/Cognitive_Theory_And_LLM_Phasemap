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

INPUT_PATH = Path(r"C:\Windows\System32\ua11a_outputs\ua11a_eval.csv")
SAVE_DIR = Path("./phasemap1b_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

WINDOWS = {
    "M_20_22_entry": [20, 21, 22],
    "M_23_25_basin": [23, 24, 25],
    "M_20_25_full": [20, 21, 22, 23, 24, 25],
}

# ============================================================
# LOAD
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Cannot find: {INPUT_PATH.resolve()}")

df = pd.read_csv(INPUT_PATH)

base_required = ["phase_target", "condition", "Gen_E", "U_K", "D_B"]
for c in base_required:
    if c not in df.columns:
        raise RuntimeError(f"Missing column: {c}")

all_needed_layers = sorted(set(sum(WINDOWS.values(), [])))
for l in all_needed_layers:
    col = f"R_{l}"
    if col not in df.columns:
        raise RuntimeError(f"Missing column: {col}")

df = df.copy()
df["phase_target"] = df["phase_target"].astype(str)
df["Gen_E"] = df["Gen_E"].astype(int)

# ============================================================
# BUILD MOMENTUM FEATURES BY WINDOW
# ============================================================

def add_window_features(df, name, layers):
    rcols = [f"R_{l}" for l in layers]
    start = layers[0]
    end = layers[-1]

    df[f"{name}_delta"] = df[f"R_{end}"] - df[f"R_{start}"]
    df[f"{name}_slope"] = df[f"{name}_delta"] / max(1, (end - start))
    df[f"{name}_area"] = df[rcols].sum(axis=1)
    df[f"{name}_mean"] = df[rcols].mean(axis=1)
    df[f"{name}_min"] = df[rcols].min(axis=1)
    df[f"{name}_max"] = df[rcols].max(axis=1)
    df[f"{name}_range"] = df[f"{name}_max"] - df[f"{name}_min"]
    df[f"{name}_cross_zero"] = (
        (df[f"{name}_min"] <= 0) & (df[f"{name}_max"] >= 0)
    ).astype(int)

    return [
        f"{name}_delta",
        f"{name}_slope",
        f"{name}_area",
        f"{name}_mean",
        f"{name}_range",
        f"{name}_cross_zero",
    ]

window_feature_cols = {}

for name, layers in WINDOWS.items():
    window_feature_cols[name] = add_window_features(df, name, layers)

# ============================================================
# CV HELPER
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

# ============================================================
# FEATURE SETS
# ============================================================

feature_sets = {
    "UK_DB_baseline": ["U_K", "D_B"],
}

for name, cols in window_feature_cols.items():
    feature_sets[f"{name}_only"] = cols
    feature_sets[f"UK_DB_plus_{name}"] = ["U_K", "D_B"] + cols

feature_sets["all_early_mid_momentum"] = (
        window_feature_cols["M_20_22_entry"]
)

feature_sets["all_momentum_no_full"] = (
       window_feature_cols["M_20_22_entry"]
    + window_feature_cols["M_23_25_basin"]
)

feature_sets["STATE_UK_DB_all_momentum"] = (
    ["U_K", "D_B"]
    + window_feature_cols["M_20_22_entry"]
    + window_feature_cols["M_23_25_basin"]
    + window_feature_cols["M_20_25_full"]
)

# ============================================================
# RUN EVAL
# ============================================================

rows = []

for name, feats in feature_sets.items():
    res = binary_cv(df, feats, positive_label="critical")
    rows.append({
        "task": "critical_vs_negative",
        "positive_label": "critical",
        "feature_set": name,
        "features": ",".join(feats),
        **res,
    })

eval_df = pd.DataFrame(rows).sort_values("auc", ascending=False)
eval_df.to_csv(SAVE_DIR / "phasemap1b_feature_eval.csv", index=False)

# ============================================================
# SUMMARY TABLES
# ============================================================

all_m_cols = sorted(set(sum(window_feature_cols.values(), [])))

phase_summary = df.groupby("phase_target").agg(
    n=("Gen_E", "count"),
    GenE_rate=("Gen_E", "mean"),
    U_K_mean=("U_K", "mean"),
    D_B_mean=("D_B", "mean"),
    **{f"{c}_mean": (c, "mean") for c in all_m_cols},
    **{f"{c}_std": (c, "std") for c in all_m_cols},
).reset_index()

condition_summary = df.groupby("condition").agg(
    n=("Gen_E", "count"),
    GenE_rate=("Gen_E", "mean"),
    phase=("phase_target", lambda x: x.mode().iloc[0]),
    U_K_mean=("U_K", "mean"),
    D_B_mean=("D_B", "mean"),
    **{f"{c}_mean": (c, "mean") for c in all_m_cols},
).reset_index()

phase_summary.to_csv(SAVE_DIR / "phasemap1b_phase_summary.csv", index=False)
condition_summary.to_csv(SAVE_DIR / "phasemap1b_condition_summary.csv", index=False)

df.to_csv(SAVE_DIR / "phasemap1b_eval.csv", index=False)

# ============================================================
# SUMMARY JSON
# ============================================================

def metric_for(feature_set, metric):
    r = eval_df[eval_df["feature_set"] == feature_set]
    if len(r) == 0:
        return None
    return float(r.iloc[0][metric])

summary = {
    "experiment": "PhaseMap-1B Early Momentum Audit",
    "input_path": str(INPUT_PATH),
    "n_rows": int(len(df)),
    "core_question": "Does basin momentum M appear before near-output layers?",
    "top_feature_eval": eval_df.head(15).to_dict(orient="records"),
   "key_metrics": {
    "UK_DB_baseline_auc": metric_for("UK_DB_baseline", "auc"),
    "M_20_22_entry_only_auc": metric_for("M_20_22_entry_only", "auc"),
    "M_23_25_basin_only_auc": metric_for("M_23_25_basin_only", "auc"),
    "M_20_25_full_only_auc": metric_for("M_20_25_full_only", "auc"),
    "UK_DB_plus_M_20_22_entry_auc": metric_for("UK_DB_plus_M_20_22_entry", "auc"),
    "UK_DB_plus_M_23_25_basin_auc": metric_for("UK_DB_plus_M_23_25_basin", "auc"),
},
    "phase_summary": phase_summary.to_dict(orient="records"),
    "condition_summary": condition_summary.to_dict(orient="records"),
    "interpretation_rules": {
        "PASS_strong_early_M": [
            "M_7_19_mid_only beats UK_DB_baseline clearly.",
            "UK_DB_plus_M_7_19_mid improves over UK_DB_baseline.",
            "Early M separates critical vs negative before L20."
        ],
        "PASS_entry_M": [
            "M_20_22_entry beats UK_DB_baseline clearly.",
            "Momentum appears at boundary-entry zone, before basin formation."
        ],
        "PASS_late_only": [
            "Only M_23_25 or M_20_25 works strongly.",
            "M may be basin-formation readout rather than early causal state."
        ],
        "FAIL_M": [
            "No early or entry momentum feature improves over UK_DB_baseline."
        ]
    }
}

with open(SAVE_DIR / "phasemap1b_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nSaved to: {SAVE_DIR.resolve()}")