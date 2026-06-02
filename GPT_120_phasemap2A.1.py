import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, r2_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.decomposition import PCA

INPUT_PATH = Path(r"C:\Windows\System32\phasemap2a_outputs\phasemap2a_eval.csv")
SAVE_DIR = Path("./phasemap2a1_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

D_LAYERS = [20, 21, 22, 23, 24, 25]
R_LAYERS = [20, 21, 22, 23, 24, 25]

df = pd.read_csv(INPUT_PATH)

required = ["phase_target", "condition", "Gen_E", "U_K", "D_B"] + [f"D_{l}" for l in D_LAYERS] + [f"R_{l}" for l in R_LAYERS]
for c in required:
    if c not in df.columns:
        raise RuntimeError(f"Missing column: {c}")

df = df.copy()
df["phase_target"] = df["phase_target"].astype(str)
df["Gen_E"] = df["Gen_E"].astype(int)

# ============================================================
# BUILD FLOW TARGETS
# ============================================================

for a, b in zip(D_LAYERS[:-1], D_LAYERS[1:]):
    df[f"dD_{a}_{b}"] = df[f"D_{b}"] - df[f"D_{a}"]

df["dD_entry_20_22"] = df["D_22"] - df["D_20"]
df["dD_basin_23_25"] = df["D_25"] - df["D_23"]
df["dD_full_20_25"] = df["D_25"] - df["D_20"]

MAIN_TARGET = "dD_basin_23_25"
df["flow_outward"] = (df[MAIN_TARGET] > 0).astype(int)
df["flow_inward"] = (df[MAIN_TARGET] < 0).astype(int)

# ============================================================
# BUILD TRAJECTORY MEMORY FEATURES
# ============================================================

R_COLS = [f"R_{l}" for l in R_LAYERS]
D_COLS = [f"D_{l}" for l in D_LAYERS]

# If clean baseline is available by graph_id, construct clean-relative deltaR.
# Otherwise use raw R trajectory as memory.
if "graph_id" in df.columns:
    clean = df[df["condition"].astype(str).str.contains("clean", case=False, na=False)]
    if len(clean) > 0:
        clean_ref = clean.groupby("graph_id")[R_COLS].mean()
        for c in R_COLS:
            df[f"clean_{c}"] = df["graph_id"].map(clean_ref[c])
            df[f"Delta{c}"] = df[c] - df[f"clean_{c}"]
        DELTA_R_COLS = [f"DeltaR_{l}" for l in R_LAYERS]
    else:
        DELTA_R_COLS = R_COLS
else:
    DELTA_R_COLS = R_COLS

# Safety: if mapped names are absent because original c was R_20 -> DeltaR_20
DELTA_R_COLS = [c for c in [f"DeltaR_{l}" for l in R_LAYERS] if c in df.columns]
if len(DELTA_R_COLS) == 0:
    DELTA_R_COLS = R_COLS

# trajectory shape summaries
df["R_delta_20_25"] = df["R_25"] - df["R_20"]
df["R_slope_20_25"] = df["R_delta_20_25"] / 5.0
df["R_area_20_25"] = df[R_COLS].sum(axis=1)
df["R_mean_20_25"] = df[R_COLS].mean(axis=1)
df["R_min_20_25"] = df[R_COLS].min(axis=1)
df["R_max_20_25"] = df[R_COLS].max(axis=1)
df["R_range_20_25"] = df["R_max_20_25"] - df["R_min_20_25"]
df["R_cross_zero_20_25"] = ((df["R_min_20_25"] <= 0) & (df["R_max_20_25"] >= 0)).astype(int)

MEM_SHAPE = [
    "R_delta_20_25",
    "R_slope_20_25",
    "R_area_20_25",
    "R_mean_20_25",
    "R_range_20_25",
    "R_cross_zero_20_25",
]

# PCA compressed memory
pca = PCA(n_components=3, random_state=SEED)
Z = pca.fit_transform(StandardScaler().fit_transform(df[DELTA_R_COLS].values.astype(float)))
df["Mem_PC1"] = Z[:, 0]
df["Mem_PC2"] = Z[:, 1]
df["Mem_PC3"] = Z[:, 2]

MEM_PC = ["Mem_PC1", "Mem_PC2", "Mem_PC3"]

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
    "UK_DB_state": ["U_K", "D_B"],

    "memory_raw_R20_25": R_COLS,
    "memory_deltaR20_25": DELTA_R_COLS,
    "memory_shape": MEM_SHAPE,
    "memory_pc123": MEM_PC,

    "STATE_plus_raw_R_memory": ["U_K", "D_B"] + R_COLS,
    "STATE_plus_deltaR_memory": ["U_K", "D_B"] + DELTA_R_COLS,
    "STATE_plus_shape_memory": ["U_K", "D_B"] + MEM_SHAPE,
    "STATE_plus_pc_memory": ["U_K", "D_B"] + MEM_PC,

    "D_trajectory_full_oracle": D_COLS,
}

FLOW_TARGETS = [
    "dD_entry_20_22",
    "dD_basin_23_25",
    "dD_full_20_25",
]

# ============================================================
# RUN REGRESSION
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
reg_df.to_csv(SAVE_DIR / "phasemap2a1_flow_regression.csv", index=False)

# ============================================================
# RUN FLOW SIGN CLASSIFICATION
# ============================================================

clf_rows = []

for target in ["flow_outward", "flow_inward"]:
    for name, feats in FEATURE_SETS.items():
        res = cv_binary(df, feats, target)
        if res is None:
            continue
        clf_rows.append({
            "task": "flow_direction_classification",
            "target": target,
            "feature_set": name,
            "features": ",".join(feats),
            **res,
        })

clf_df = pd.DataFrame(clf_rows).sort_values(["target", "auc"], ascending=[True, False])
clf_df.to_csv(SAVE_DIR / "phasemap2a1_flow_direction.csv", index=False)

# ============================================================
# MEMORY GEOMETRY SUMMARY
# ============================================================

memory_cols = ["U_K", "D_B"] + R_COLS + DELTA_R_COLS + MEM_SHAPE + MEM_PC + [
    "dD_entry_20_22",
    "dD_basin_23_25",
    "dD_full_20_25",
    "flow_outward",
    "flow_inward",
]

memory_cols = list(dict.fromkeys(memory_cols))

phase_summary = df.groupby("phase_target").agg(
    n=("Gen_E", "count"),
    GenE_rate=("Gen_E", "mean"),
    **{f"{c}_mean": (c, "mean") for c in memory_cols},
    **{f"{c}_std": (c, "std") for c in memory_cols if c not in ["flow_outward", "flow_inward"]},
).reset_index()

condition_summary = df.groupby("condition").agg(
    n=("Gen_E", "count"),
    phase=("phase_target", lambda x: x.mode().iloc[0]),
    GenE_rate=("Gen_E", "mean"),
    **{f"{c}_mean": (c, "mean") for c in memory_cols},
).reset_index()

phase_summary.to_csv(SAVE_DIR / "phasemap2a1_phase_memory_summary.csv", index=False)
condition_summary.to_csv(SAVE_DIR / "phasemap2a1_condition_memory_summary.csv", index=False)

pca_loadings = pd.DataFrame(
    pca.components_.T,
    index=DELTA_R_COLS,
    columns=["Mem_PC1", "Mem_PC2", "Mem_PC3"],
).reset_index().rename(columns={"index": "variable"})

pca_loadings.to_csv(SAVE_DIR / "phasemap2a1_memory_pca_loadings.csv", index=False)

df.to_csv(SAVE_DIR / "phasemap2a1_eval.csv", index=False)

# ============================================================
# SUMMARY JSON
# ============================================================

def metric_row(table, target, feature_set, metric):
    sub = table[
        (table["target"] == target)
        & (table["feature_set"] == feature_set)
    ]
    if len(sub) == 0:
        return None
    val = sub.iloc[0][metric]
    return None if pd.isna(val) else float(val)

def best_reg(target):
    return (
        reg_df[reg_df["target"] == target]
        .sort_values("r2", ascending=False)
        .head(10)
        .to_dict(orient="records")
    )

def best_clf(target):
    return (
        clf_df[clf_df["target"] == target]
        .sort_values("auc", ascending=False)
        .head(10)
        .to_dict(orient="records")
    )

summary = {
    "experiment": "PhaseMap-2A.1 Trajectory Memory Closure Audit",
    "input_path": str(INPUT_PATH),
    "n_rows": int(len(df)),
    "core_question": "Is DeltaR/R20:25 the missing history variable that closes the flow dynamics?",
    "memory_definition": {
        "raw_R20_25": R_COLS,
        "deltaR20_25": DELTA_R_COLS,
        "shape_memory": MEM_SHAPE,
        "pc_memory": MEM_PC,
    },
    "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
    "key_metrics": {
        "UK_DB_to_dD_basin_r2": metric_row(reg_df, "dD_basin_23_25", "UK_DB_state", "r2"),
        "UK_DB_to_dD_basin_corr": metric_row(reg_df, "dD_basin_23_25", "UK_DB_state", "corr"),

        "memory_raw_to_dD_basin_r2": metric_row(reg_df, "dD_basin_23_25", "memory_raw_R20_25", "r2"),
        "memory_raw_to_dD_basin_corr": metric_row(reg_df, "dD_basin_23_25", "memory_raw_R20_25", "corr"),

        "memory_deltaR_to_dD_basin_r2": metric_row(reg_df, "dD_basin_23_25", "memory_deltaR20_25", "r2"),
        "memory_deltaR_to_dD_basin_corr": metric_row(reg_df, "dD_basin_23_25", "memory_deltaR20_25", "corr"),

        "state_plus_raw_memory_to_dD_basin_r2": metric_row(reg_df, "dD_basin_23_25", "STATE_plus_raw_R_memory", "r2"),
        "state_plus_deltaR_memory_to_dD_basin_r2": metric_row(reg_df, "dD_basin_23_25", "STATE_plus_deltaR_memory", "r2"),

        "UK_DB_to_flow_outward_auc": metric_row(clf_df, "flow_outward", "UK_DB_state", "auc"),
        "memory_raw_to_flow_outward_auc": metric_row(clf_df, "flow_outward", "memory_raw_R20_25", "auc"),
        "state_plus_raw_memory_to_flow_outward_auc": metric_row(clf_df, "flow_outward", "STATE_plus_raw_R_memory", "auc"),
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
    "phase_memory_summary": phase_summary.to_dict(orient="records"),
    "condition_memory_summary": condition_summary.to_dict(orient="records"),
    "interpretation_rules": {
        "PASS_memory_closure": [
            "memory_raw_R20_25 or memory_deltaR20_25 strongly improves over UK_DB_state.",
            "STATE_plus_memory predicts dD_basin_23_25 with R2 > 0.5 or flow sign AUC > 0.85.",
            "Memory PC1/PC2 explain most variance and are interpretable."
        ],
        "PASS_partial": [
            "Memory improves flow sign but not continuous dD.",
            "This supports history dependence but not a fully closed low-dimensional flow equation."
        ],
        "FAIL_memory": [
            "Memory features do not improve over UK_DB_state.",
            "Only D_trajectory_full_oracle works, meaning R-memory is not the missing variable."
        ],
        "important_caveat": [
            "Raw R20:25 may include information close to D trajectory; deltaR is cleaner if graph_id clean baseline exists.",
            "If raw_R works but deltaR fails, memory may be answer-readout rather than structural memory."
        ]
    }
}

with open(SAVE_DIR / "phasemap2a1_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nSaved to: {SAVE_DIR.resolve()}")