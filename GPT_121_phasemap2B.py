import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    r2_score,
    mean_absolute_error,
)
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path(r"C:\Windows\System32\phasemap2a1_outputs\phasemap2a1_eval.csv")
SAVE_DIR = Path("./phasemap2b_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

STATE_BASE = ["U_K", "D_B"]

H_SHAPE = [
    "R_delta_20_25",
    "R_slope_20_25",
    "R_area_20_25",
    "R_mean_20_25",
    "R_range_20_25",
    "R_cross_zero_20_25",
]

R_COLS = [f"R_{l}" for l in range(20, 26)]
D_COLS = [f"D_{l}" for l in range(20, 26)]

MAIN_TARGET = "dD_basin_23_25"

# ============================================================
# LOAD
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Cannot find input file: {INPUT_PATH.resolve()}")

df = pd.read_csv(INPUT_PATH)

required = [
    "phase_target",
    "condition",
    "Gen_E",
    "U_K",
    "D_B",
    MAIN_TARGET,
    "flow_outward",
    "flow_inward",
] + H_SHAPE + R_COLS + D_COLS

for c in required:
    if c not in df.columns:
        raise RuntimeError(f"Missing column: {c}")

df = df.copy()
df["phase_target"] = df["phase_target"].astype(str)
df["condition"] = df["condition"].astype(str)
df["Gen_E"] = df["Gen_E"].astype(int)

# Group column
if "graph_id" in df.columns:
    GROUP_COL = "graph_id"
else:
    # fallback: condition grouping is harsher but coarse
    GROUP_COL = "condition"

# ============================================================
# FEATURE SETS
# ============================================================

FEATURE_SETS = {
    # Base state
    "S_base_UK_DB": STATE_BASE,

    # Memory only
    "H_shape_only": H_SHAPE,
    "H_rawR_only": R_COLS,

    # Closed state candidates
    "S_plus_H_shape": STATE_BASE + H_SHAPE,
    "S_plus_H_rawR": STATE_BASE + R_COLS,

    # Controls / oracles
    "D_traj_oracle": D_COLS,
    "S_plus_Dtraj_oracle": STATE_BASE + D_COLS,
}

# deltaR if available
DELTA_R_COLS = [f"DeltaR_{l}" for l in range(20, 26) if f"DeltaR_{l}" in df.columns]
if len(DELTA_R_COLS) == 6:
    FEATURE_SETS["H_deltaR_only"] = DELTA_R_COLS
    FEATURE_SETS["S_plus_H_deltaR"] = STATE_BASE + DELTA_R_COLS

# ============================================================
# CV HELPERS
# ============================================================

def make_splits(data, y=None, group_mode=False, classification=False):
    if group_mode:
        groups = data[GROUP_COL].values
        n_groups = len(np.unique(groups))
        if n_groups >= 5:
            gkf = GroupKFold(n_splits=5)
            return list(gkf.split(data, y, groups))
        # fallback
    if classification:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        return list(skf.split(data, y))
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    return list(kf.split(data))

def cv_regression(data, features, target, model_type="ridge", group_mode=False):
    X = data[features].values.astype(float)
    y = data[target].values.astype(float)

    splits = make_splits(data, y=y, group_mode=group_mode, classification=False)
    pred = np.zeros(len(data))

    for tr, te in splits:
        if model_type == "ridge":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ])
        elif model_type == "rf":
            model = RandomForestRegressor(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=5,
                random_state=SEED,
            )
        else:
            raise ValueError(model_type)

        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])

    corr = np.corrcoef(y, pred)[0, 1] if np.std(pred) > 0 else 0.0

    return {
        "r2": float(r2_score(y, pred)),
        "corr": float(corr),
        "mae": float(mean_absolute_error(y, pred)),
    }

def cv_binary(data, features, target, model_type="lr", group_mode=False):
    X = data[features].values.astype(float)
    y = data[target].values.astype(int)

    if len(np.unique(y)) < 2:
        return None

    splits = make_splits(data, y=y, group_mode=group_mode, classification=True)
    pred = np.zeros(len(data))

    for tr, te in splits:
        if model_type == "lr":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=5000)),
            ])
        elif model_type == "rf":
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=SEED,
            )
        else:
            raise ValueError(model_type)

        model.fit(X[tr], y[tr])
        pred[te] = model.predict_proba(X[te])[:, 1]

    label = (pred >= 0.5).astype(int)

    return {
        "auc": float(roc_auc_score(y, pred)),
        "acc": float(accuracy_score(y, label)),
        "f1": float(f1_score(y, label, zero_division=0)),
    }

# ============================================================
# RUN REGRESSION AND CLASSIFICATION
# ============================================================

reg_rows = []
clf_rows = []

for group_mode in [False, True]:
    cv_name = "GroupKFold" if group_mode else "KFold"

    for model_type in ["ridge", "rf"]:
        for name, feats in FEATURE_SETS.items():
            res = cv_regression(
                df,
                feats,
                MAIN_TARGET,
                model_type=model_type,
                group_mode=group_mode,
            )
            reg_rows.append({
                "cv": cv_name,
                "model": model_type,
                "target": MAIN_TARGET,
                "feature_set": name,
                "features": ",".join(feats),
                **res,
            })

    for model_type in ["lr", "rf"]:
        for target in ["flow_outward", "flow_inward"]:
            for name, feats in FEATURE_SETS.items():
                res = cv_binary(
                    df,
                    feats,
                    target,
                    model_type=model_type,
                    group_mode=group_mode,
                )
                if res is None:
                    continue
                clf_rows.append({
                    "cv": cv_name,
                    "model": model_type,
                    "target": target,
                    "feature_set": name,
                    "features": ",".join(feats),
                    **res,
                })

reg_df = pd.DataFrame(reg_rows).sort_values(
    ["cv", "model", "r2"],
    ascending=[True, True, False],
)
clf_df = pd.DataFrame(clf_rows).sort_values(
    ["cv", "model", "target", "auc"],
    ascending=[True, True, True, False],
)

reg_df.to_csv(SAVE_DIR / "phasemap2b_regression_eval.csv", index=False)
clf_df.to_csv(SAVE_DIR / "phasemap2b_direction_eval.csv", index=False)

# ============================================================
# ABLATION / SYNERGY
# ============================================================

def get_metric(table, cv, model, feature_set, metric, target=None):
    sub = table[
        (table["cv"] == cv)
        & (table["model"] == model)
        & (table["feature_set"] == feature_set)
    ]
    if target is not None:
        sub = sub[sub["target"] == target]
    if len(sub) == 0:
        return None
    val = sub.iloc[0][metric]
    return None if pd.isna(val) else float(val)

ablation_rows = []

for cv in ["KFold", "GroupKFold"]:
    for model in ["ridge", "rf"]:
        base = get_metric(reg_df, cv, model, "S_base_UK_DB", "r2", MAIN_TARGET)
        h = get_metric(reg_df, cv, model, "H_shape_only", "r2", MAIN_TARGET)
        closed = get_metric(reg_df, cv, model, "S_plus_H_shape", "r2", MAIN_TARGET)

        if base is not None and h is not None and closed is not None:
            ablation_rows.append({
                "cv": cv,
                "model": model,
                "task": "regression",
                "target": MAIN_TARGET,
                "base_UK_DB": base,
                "H_shape_only": h,
                "closed_S_plus_H_shape": closed,
                "gain_over_base": closed - base,
                "gain_over_H_only": closed - h,
                "synergy": closed - max(base, h),
            })

for cv in ["KFold", "GroupKFold"]:
    for model in ["lr", "rf"]:
        base = get_metric(clf_df, cv, model, "S_base_UK_DB", "auc", "flow_outward")
        h = get_metric(clf_df, cv, model, "H_shape_only", "auc", "flow_outward")
        closed = get_metric(clf_df, cv, model, "S_plus_H_shape", "auc", "flow_outward")

        if base is not None and h is not None and closed is not None:
            ablation_rows.append({
                "cv": cv,
                "model": model,
                "task": "classification",
                "target": "flow_outward",
                "base_UK_DB": base,
                "H_shape_only": h,
                "closed_S_plus_H_shape": closed,
                "gain_over_base": closed - base,
                "gain_over_H_only": closed - h,
                "synergy": closed - max(base, h),
            })

ablation_df = pd.DataFrame(ablation_rows)
ablation_df.to_csv(SAVE_DIR / "phasemap2b_ablation_synergy.csv", index=False)

# ============================================================
# PHASE / CONDITION CLOSED-STATE SUMMARY
# ============================================================

summary_cols = STATE_BASE + H_SHAPE + R_COLS + [MAIN_TARGET, "flow_outward", "flow_inward"]

phase_summary = df.groupby("phase_target").agg(
    n=("Gen_E", "count"),
    GenE_rate=("Gen_E", "mean"),
    **{f"{c}_mean": (c, "mean") for c in summary_cols},
    **{f"{c}_std": (c, "std") for c in summary_cols if c not in ["flow_outward", "flow_inward"]},
).reset_index()

condition_summary = df.groupby("condition").agg(
    n=("Gen_E", "count"),
    phase=("phase_target", lambda x: x.mode().iloc[0]),
    GenE_rate=("Gen_E", "mean"),
    **{f"{c}_mean": (c, "mean") for c in summary_cols},
).reset_index()

phase_summary.to_csv(SAVE_DIR / "phasemap2b_phase_closed_state_summary.csv", index=False)
condition_summary.to_csv(SAVE_DIR / "phasemap2b_condition_closed_state_summary.csv", index=False)

# ============================================================
# FEATURE IMPORTANCE FOR RF CLOSED MODEL
# ============================================================

importance_rows = []

for feature_set in ["S_plus_H_shape", "S_plus_H_rawR"]:
    feats = FEATURE_SETS[feature_set]
    X = df[feats].values.astype(float)
    y_reg = df[MAIN_TARGET].values.astype(float)

    rf_reg = RandomForestRegressor(
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=5,
        random_state=SEED,
    )
    rf_reg.fit(X, y_reg)

    for f, imp in zip(feats, rf_reg.feature_importances_):
        importance_rows.append({
            "model": "rf_regression",
            "feature_set": feature_set,
            "feature": f,
            "importance": float(imp),
        })

    y_clf = df["flow_outward"].values.astype(int)
    rf_clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=SEED,
    )
    rf_clf.fit(X, y_clf)

    for f, imp in zip(feats, rf_clf.feature_importances_):
        importance_rows.append({
            "model": "rf_classification",
            "feature_set": feature_set,
            "feature": f,
            "importance": float(imp),
        })

importance_df = pd.DataFrame(importance_rows).sort_values(
    ["model", "feature_set", "importance"],
    ascending=[True, True, False],
)
importance_df.to_csv(SAVE_DIR / "phasemap2b_feature_importance.csv", index=False)

# ============================================================
# SAVE FULL EVAL
# ============================================================

df.to_csv(SAVE_DIR / "phasemap2b_eval.csv", index=False)

# ============================================================
# SUMMARY JSON
# ============================================================

def top_rows(table, cv=None, model=None, target=None, metric="r2", n=10):
    sub = table.copy()
    if cv is not None:
        sub = sub[sub["cv"] == cv]
    if model is not None:
        sub = sub[sub["model"] == model]
    if target is not None:
        sub = sub[sub["target"] == target]
    return sub.sort_values(metric, ascending=False).head(n).to_dict(orient="records")

summary = {
    "experiment": "PhaseMap-2B Minimal Closed State Audit",
    "input_path": str(INPUT_PATH),
    "n_rows": int(len(df)),
    "group_col": GROUP_COL,
    "core_question": "Is (U_K, D_B, H_shape) a minimal closed state for basin-boundary flow?",
    "feature_sets": FEATURE_SETS,
    "key_metrics": {
        "KFold_ridge_UK_DB_r2": get_metric(reg_df, "KFold", "ridge", "S_base_UK_DB", "r2", MAIN_TARGET),
        "KFold_ridge_H_shape_r2": get_metric(reg_df, "KFold", "ridge", "H_shape_only", "r2", MAIN_TARGET),
        "KFold_ridge_closed_r2": get_metric(reg_df, "KFold", "ridge", "S_plus_H_shape", "r2", MAIN_TARGET),

        "Group_ridge_UK_DB_r2": get_metric(reg_df, "GroupKFold", "ridge", "S_base_UK_DB", "r2", MAIN_TARGET),
        "Group_ridge_H_shape_r2": get_metric(reg_df, "GroupKFold", "ridge", "H_shape_only", "r2", MAIN_TARGET),
        "Group_ridge_closed_r2": get_metric(reg_df, "GroupKFold", "ridge", "S_plus_H_shape", "r2", MAIN_TARGET),

        "KFold_lr_UK_DB_auc": get_metric(clf_df, "KFold", "lr", "S_base_UK_DB", "auc", "flow_outward"),
        "KFold_lr_H_shape_auc": get_metric(clf_df, "KFold", "lr", "H_shape_only", "auc", "flow_outward"),
        "KFold_lr_closed_auc": get_metric(clf_df, "KFold", "lr", "S_plus_H_shape", "auc", "flow_outward"),

        "Group_lr_UK_DB_auc": get_metric(clf_df, "GroupKFold", "lr", "S_base_UK_DB", "auc", "flow_outward"),
        "Group_lr_H_shape_auc": get_metric(clf_df, "GroupKFold", "lr", "H_shape_only", "auc", "flow_outward"),
        "Group_lr_closed_auc": get_metric(clf_df, "GroupKFold", "lr", "S_plus_H_shape", "auc", "flow_outward"),
    },
    "top_regression": {
        "KFold_ridge": top_rows(reg_df, cv="KFold", model="ridge", target=MAIN_TARGET, metric="r2"),
        "Group_ridge": top_rows(reg_df, cv="GroupKFold", model="ridge", target=MAIN_TARGET, metric="r2"),
        "KFold_rf": top_rows(reg_df, cv="KFold", model="rf", target=MAIN_TARGET, metric="r2"),
        "Group_rf": top_rows(reg_df, cv="GroupKFold", model="rf", target=MAIN_TARGET, metric="r2"),
    },
    "top_direction": {
        "KFold_lr": top_rows(clf_df, cv="KFold", model="lr", target="flow_outward", metric="auc"),
        "Group_lr": top_rows(clf_df, cv="GroupKFold", model="lr", target="flow_outward", metric="auc"),
        "KFold_rf": top_rows(clf_df, cv="KFold", model="rf", target="flow_outward", metric="auc"),
        "Group_rf": top_rows(clf_df, cv="GroupKFold", model="rf", target="flow_outward", metric="auc"),
    },
    "ablation_synergy": ablation_df.to_dict(orient="records"),
    "feature_importance": importance_df.head(30).to_dict(orient="records"),
    "phase_summary": phase_summary.to_dict(orient="records"),
    "condition_summary": condition_summary.to_dict(orient="records"),
    "interpretation_rules": {
        "PASS_closed_state": [
            "S_plus_H_shape beats S_base_UK_DB clearly in KFold and GroupKFold.",
            "S_plus_H_shape reaches regression R2 > 0.5 or flow_outward AUC > 0.85.",
            "It performs close to rawR memory but with fewer interpretable variables."
        ],
        "PASS_memory_dominant": [
            "H_shape_only performs almost as well as S_plus_H_shape.",
            "This means trajectory memory dominates current coordinates."
        ],
        "FAIL_minimal_closed_state": [
            "Only D_trajectory_oracle or rawR works.",
            "H_shape does not improve over UK_DB.",
            "GroupKFold collapses, suggesting leakage / non-generalization."
        ],
        "next_if_pass": "PhaseMap-2C: fit explicit low-dimensional flow equation dD = F(U_K,D_B,H_shape).",
        "next_if_fail": "Return to state variable discovery; current H_shape is not the missing state."
    }
}

with open(SAVE_DIR / "phasemap2b_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nSaved to: {SAVE_DIR.resolve()}")