# ============================================================
# UA-9A-v2: Boundary Distance Audit
#
# Goal:
#   Test whether response sensitivity / mixed outcome is governed
#   by distance to answer-boundary D_B, rather than competition
#   intensity U_K.
#
# Input:
#   ha1a_debug_outputs/deltaR.csv
#
# Main idea:
#   R_l = logit(C) - logit(E)
#
#   D_entry = min |R_20:22|
#   D_basin = min |R_23:26|
#   D_commit = |R_27|
#
#   If D_B is true boundary distance:
#       smaller D_B -> more mixed / uncertain outcome
#       ambiguous samples should concentrate near boundary
#
# Outputs:
#   ua9a_v2_outputs/
#       ua9a_v2_boundary_distance.csv
#       ua9a_v2_feature_eval.csv
#       ua9a_v2_condition_summary.csv
#       ua9a_v2_phase_summary.csv
#       ua9a_v2_boundary_test.csv
#       ua9a_v2_summary.json
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

INPUT_PATH = Path("C:\Windows\System32\ha1a_debug_outputs\deltaR.csv")
SAVE_DIR = Path("./ua9a_v2_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

COMPETITION_CONDITIONS = [
    "clean",
    "weak_distractor_v2",
    "ambiguous_branch",
    "strong_conflict",
]

ENTRY_LAYERS = [20, 21, 22]
BASIN_LAYERS = [23, 24, 25, 26]
COMMIT_LAYER = 27

N_BINS = 10

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_PATH)

required_cols = [
    "condition",
    "phase_target",
    "Gen_E",
] + [f"R_{l}" for l in ENTRY_LAYERS + BASIN_LAYERS + [COMMIT_LAYER]]

for col in required_cols:
    if col not in df.columns:
        raise RuntimeError(f"Missing column: {col}")

df = df[df["condition"].isin(COMPETITION_CONDITIONS)].copy()
df["Gen_E"] = df["Gen_E"].astype(int)

if len(df) == 0:
    raise RuntimeError("No competition rows found. Check condition names.")

# ============================================================
# CONSTRUCT BOUNDARY DISTANCE FEATURES
# ============================================================

def min_abs(row, layers):
    return min(abs(row[f"R_{l}"]) for l in layers)

def mean_abs(row, layers):
    return np.mean([abs(row[f"R_{l}"]) for l in layers])

def abs_mean(row, layers):
    return abs(np.mean([row[f"R_{l}"] for l in layers]))

df["D_entry_minabs_R20_22"] = df.apply(lambda r: min_abs(r, ENTRY_LAYERS), axis=1)
df["D_entry_meanabs_R20_22"] = df.apply(lambda r: mean_abs(r, ENTRY_LAYERS), axis=1)
df["D_entry_absmean_R20_22"] = df.apply(lambda r: abs_mean(r, ENTRY_LAYERS), axis=1)

df["D_basin_minabs_R23_26"] = df.apply(lambda r: min_abs(r, BASIN_LAYERS), axis=1)
df["D_basin_meanabs_R23_26"] = df.apply(lambda r: mean_abs(r, BASIN_LAYERS), axis=1)
df["D_basin_absmean_R23_26"] = df.apply(lambda r: abs_mean(r, BASIN_LAYERS), axis=1)

df["D_commit_abs_R27"] = df[f"R_{COMMIT_LAYER}"].abs()

# signed variables for comparison
df["S_entry_mean_R20_22"] = df[[f"R_{l}" for l in ENTRY_LAYERS]].mean(axis=1)
df["S_basin_mean_R23_26"] = df[[f"R_{l}" for l in BASIN_LAYERS]].mean(axis=1)
df["S_commit_R27"] = df[f"R_{COMMIT_LAYER}"]

# boundary closeness score: larger = closer
eps = 1e-6
df["B_entry_close"] = 1.0 / (df["D_entry_minabs_R20_22"] + eps)
df["B_basin_close"] = 1.0 / (df["D_basin_minabs_R23_26"] + eps)
df["B_commit_close"] = 1.0 / (df["D_commit_abs_R27"] + eps)

df.to_csv(SAVE_DIR / "ua9a_v2_boundary_distance.csv", index=False)

# ============================================================
# MIXED OUTCOME TARGET
# ============================================================

# We define "mixed / near-boundary condition" at condition level.
# ambiguous_branch is expected to be mixed.
# weak_distractor_v2 may be weakly mixed.
# clean = stable C, strong_conflict = stable E.
df["Target_mixed_condition"] = df["condition"].isin([
    "ambiguous_branch",
    "weak_distractor_v2",
]).astype(int)

df["Target_ambiguous_only"] = (df["condition"] == "ambiguous_branch").astype(int)

# ============================================================
# FEATURE EVALUATION
# ============================================================

FEATURE_SETS = {
    # pure distance variables
    "D_entry_minabs": ["D_entry_minabs_R20_22"],
    "D_basin_minabs": ["D_basin_minabs_R23_26"],
    "D_commit_abs": ["D_commit_abs_R27"],

    "D_entry_meanabs": ["D_entry_meanabs_R20_22"],
    "D_basin_meanabs": ["D_basin_meanabs_R23_26"],

    "D_entry_absmean": ["D_entry_absmean_R20_22"],
    "D_basin_absmean": ["D_basin_absmean_R23_26"],

    # closeness variables, larger means closer to boundary
    "B_entry_close": ["B_entry_close"],
    "B_basin_close": ["B_basin_close"],
    "B_commit_close": ["B_commit_close"],

    # signed readouts, included as caveat baseline
    "S_entry_mean": ["S_entry_mean_R20_22"],
    "S_basin_mean": ["S_basin_mean_R23_26"],
    "S_commit_R27": ["S_commit_R27"],

    # combined early/basin distance only
    "D_entry_plus_basin": [
        "D_entry_minabs_R20_22",
        "D_basin_minabs_R23_26",
    ],
    "B_entry_plus_basin": [
        "B_entry_close",
        "B_basin_close",
    ],

    # all non-final distance variables
    "D_nonfinal_all": [
        "D_entry_minabs_R20_22",
        "D_entry_meanabs_R20_22",
        "D_entry_absmean_R20_22",
        "D_basin_minabs_R23_26",
        "D_basin_meanabs_R23_26",
        "D_basin_absmean_R23_26",
    ],

    # final oracle baseline
    "D_all_with_commit": [
        "D_entry_minabs_R20_22",
        "D_basin_minabs_R23_26",
        "D_commit_abs_R27",
    ],
    "S_all_with_commit": [
        "S_entry_mean_R20_22",
        "S_basin_mean_R23_26",
        "S_commit_R27",
    ],
}

def cv_eval(df, feature_cols, target_col):
    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(int)

    if len(np.unique(y)) < 2:
        return None

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
    }

eval_rows = []

for target_col in ["Gen_E", "Target_mixed_condition", "Target_ambiguous_only"]:
    for name, cols in FEATURE_SETS.items():
        res = cv_eval(df, cols, target_col)

        if res is None:
            continue

        eval_rows.append({
            "target": target_col,
            "feature_set": name,
            "features": ",".join(cols),
            "auc": res["auc"],
            "acc": res["acc"],
            "f1": res["f1"],
        })

feature_eval = pd.DataFrame(eval_rows).sort_values(
    ["target", "auc"],
    ascending=[True, False],
)

feature_eval.to_csv(SAVE_DIR / "ua9a_v2_feature_eval.csv", index=False)

# ============================================================
# NEAR-BOUNDARY TEST
# ============================================================

boundary_vars = [
    "D_entry_minabs_R20_22",
    "D_basin_minabs_R23_26",
    "D_commit_abs_R27",
    "D_basin_absmean_R23_26",
]

boundary_rows = []

for v in boundary_vars:
    q25 = float(df[v].quantile(0.25))
    q50 = float(df[v].quantile(0.50))
    q75 = float(df[v].quantile(0.75))

    near = df[df[v] <= q25]
    mid = df[(df[v] > q25) & (df[v] <= q75)]
    far = df[df[v] > q75]

    def summarize_group(g):
        return {
            "n": int(len(g)),
            "GenE_rate": float(g["Gen_E"].mean()) if len(g) else None,
            "mixed_condition_rate": float(g["Target_mixed_condition"].mean()) if len(g) else None,
            "ambiguous_rate": float(g["Target_ambiguous_only"].mean()) if len(g) else None,
            "condition_mix": g["condition"].value_counts(normalize=True).to_dict(),
            "phase_mix": g["phase_target"].value_counts(normalize=True).to_dict(),
        }

    boundary_rows.append({
        "distance_var": v,
        "q25": q25,
        "q50": q50,
        "q75": q75,
        "near_q25": summarize_group(near),
        "mid_q25_q75": summarize_group(mid),
        "far_q75": summarize_group(far),
    })

boundary_test = pd.json_normalize(boundary_rows)
boundary_test.to_csv(SAVE_DIR / "ua9a_v2_boundary_test.csv", index=False)

# ============================================================
# BIN CURVES
# ============================================================

def make_bin_table(var):
    temp = df.copy()

    temp[f"{var}_bin"] = pd.qcut(
        temp[var],
        q=N_BINS,
        duplicates="drop",
    )

    b = temp.groupby(f"{var}_bin").agg(
        var_mean=(var, "mean"),
        var_min=(var, "min"),
        var_max=(var, "max"),
        n=("Gen_E", "count"),
        GenE_rate=("Gen_E", "mean"),
        mixed_condition_rate=("Target_mixed_condition", "mean"),
        ambiguous_rate=("Target_ambiguous_only", "mean"),
    ).reset_index()

    b["variable"] = var
    return b

bin_tables = []
for v in boundary_vars:
    b = make_bin_table(v)
    b.to_csv(SAVE_DIR / f"ua9a_v2_bins_{v}.csv", index=False)
    bin_tables.append(b)

all_bins = pd.concat(bin_tables, ignore_index=True)
all_bins.to_csv(SAVE_DIR / "ua9a_v2_all_bins.csv", index=False)

# ============================================================
# CONDITION / PHASE SUMMARY
# ============================================================

condition_summary = df.groupby("condition").agg(
    n=("Gen_E", "count"),
    GenE_rate=("Gen_E", "mean"),
    mixed_condition_rate=("Target_mixed_condition", "mean"),
    ambiguous_rate=("Target_ambiguous_only", "mean"),

    D_entry_minabs_mean=("D_entry_minabs_R20_22", "mean"),
    D_basin_minabs_mean=("D_basin_minabs_R23_26", "mean"),
    D_commit_abs_mean=("D_commit_abs_R27", "mean"),

    D_entry_absmean_mean=("D_entry_absmean_R20_22", "mean"),
    D_basin_absmean_mean=("D_basin_absmean_R23_26", "mean"),

    S_entry_mean=("S_entry_mean_R20_22", "mean"),
    S_basin_mean=("S_basin_mean_R23_26", "mean"),
    S_commit_mean=("S_commit_R27", "mean"),
).reset_index()

phase_summary = df.groupby("phase_target").agg(
    n=("Gen_E", "count"),
    GenE_rate=("Gen_E", "mean"),
    mixed_condition_rate=("Target_mixed_condition", "mean"),
    ambiguous_rate=("Target_ambiguous_only", "mean"),

    D_entry_minabs_mean=("D_entry_minabs_R20_22", "mean"),
    D_basin_minabs_mean=("D_basin_minabs_R23_26", "mean"),
    D_commit_abs_mean=("D_commit_abs_R27", "mean"),

    D_entry_absmean_mean=("D_entry_absmean_R20_22", "mean"),
    D_basin_absmean_mean=("D_basin_absmean_R23_26", "mean"),

    S_entry_mean=("S_entry_mean_R20_22", "mean"),
    S_basin_mean=("S_basin_mean_R23_26", "mean"),
    S_commit_mean=("S_commit_R27", "mean"),
).reset_index()

condition_summary.to_csv(SAVE_DIR / "ua9a_v2_condition_summary.csv", index=False)
phase_summary.to_csv(SAVE_DIR / "ua9a_v2_phase_summary.csv", index=False)

# ============================================================
# SUMMARY
# ============================================================

def top_features(target, k=8):
    return (
        feature_eval[feature_eval["target"] == target]
        .head(k)
        .to_dict(orient="records")
    )

summary = {
    "experiment": "UA-9A-v2 Boundary Distance Audit",
    "input_path": str(INPUT_PATH),
    "n_rows": int(len(df)),
    "competition_conditions": COMPETITION_CONDITIONS,

    "top_features_Gen_E": top_features("Gen_E"),
    "top_features_mixed_condition": top_features("Target_mixed_condition"),
    "top_features_ambiguous_only": top_features("Target_ambiguous_only"),

    "boundary_test": boundary_rows,

    "condition_summary": condition_summary.to_dict(orient="records"),
    "phase_summary": phase_summary.to_dict(orient="records"),

    "interpretation_rules": {
        "PASS_Boundary_Distance": [
            "Non-final D_basin or D_entry features predict mixed/ambiguous better than signed final readout.",
            "Near-boundary groups are enriched for ambiguous_branch or mixed conditions.",
            "Near-boundary GenE_rate is closer to mixed rather than saturated 0 or 1."
        ],
        "FAIL_Boundary_Distance": [
            "Only S_commit_R27 predicts Gen_E.",
            "Near-boundary groups are not enriched for ambiguous/mixed samples.",
            "D_B behaves merely as final answer readout."
        ],
        "Strongest_nontrivial_positive_result": [
            "D_basin_minabs_R23_26 or D_basin_absmean_R23_26 predicts Target_ambiguous_only or Target_mixed_condition."
        ]
    }
}

with open(SAVE_DIR / "ua9a_v2_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nSaved to: {SAVE_DIR.resolve()}")