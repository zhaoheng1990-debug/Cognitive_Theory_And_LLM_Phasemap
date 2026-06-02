import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold, GroupKFold
from sklearn.ensemble import RandomForestRegressor

# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path(r"C:\Windows\System32\phasemap2b_outputs\phasemap2b_eval.csv")
SAVE_DIR = Path("./phasemap2c_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
MAIN_TARGET = "dD_basin_23_25"

H_SHAPE = [
    "R_delta_20_25",
    "R_slope_20_25",
    "R_area_20_25",
    "R_mean_20_25",
    "R_range_20_25",
    "R_cross_zero_20_25",
]

STATE_PROJ = ["U_K", "D_B"]

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_PATH)

required = [
    "phase_target",
    "condition",
    "Gen_E",
    MAIN_TARGET,
] + H_SHAPE + STATE_PROJ

for c in required:
    if c not in df.columns:
        raise RuntimeError(f"Missing column: {c}")

df = df.copy()
df["phase_target"] = df["phase_target"].astype(str)
df["condition"] = df["condition"].astype(str)
df["Gen_E"] = df["Gen_E"].astype(int)

GROUP_COL = "graph_id" if "graph_id" in df.columns else "condition"

# ============================================================
# DERIVED LOW-DIMENSIONAL COORDINATES
# ============================================================

# Minimal interpretable H candidates
df["H_velocity"] = df["R_slope_20_25"]
df["H_position"] = df["R_mean_20_25"]
df["H_energy"] = df["R_area_20_25"]
df["H_span"] = df["R_range_20_25"]

# Main compact state
H_COMPACT = [
    "H_velocity",
    "H_position",
    "H_energy",
    "H_span",
    "R_cross_zero_20_25",
]

# ============================================================
# FEATURE EQUATION SETS
# ============================================================

FEATURE_SETS = {
    "H_shape_linear": H_SHAPE,
    "H_compact_linear": H_COMPACT,
    "H_velocity_only": ["H_velocity"],
    "H_position_velocity": ["H_position", "H_velocity"],
    "H_energy_velocity": ["H_energy", "H_velocity"],
    "H_span_velocity": ["H_span", "H_velocity"],
    "H_shape_plus_projection": H_SHAPE + STATE_PROJ,
}

# ============================================================
# CV HELPERS
# ============================================================

def get_splits(data, group_mode=False):
    if group_mode:
        groups = data[GROUP_COL].values
        if len(np.unique(groups)) >= 5:
            return list(GroupKFold(n_splits=5).split(data, groups=groups))
    return list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(data))

def cv_regression(data, features, target, model_kind, group_mode=False):
    X = data[features].values.astype(float)
    y = data[target].values.astype(float)

    splits = get_splits(data, group_mode=group_mode)
    pred = np.zeros(len(data))

    for tr, te in splits:
        if model_kind == "ridge_linear":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ])
        elif model_kind == "ridge_poly2":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("ridge", Ridge(alpha=1.0)),
            ])
        elif model_kind == "lasso_poly2":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("lasso", Lasso(alpha=0.002, max_iter=20000)),
            ])
        elif model_kind == "rf":
            model = RandomForestRegressor(
                n_estimators=500,
                max_depth=5,
                min_samples_leaf=5,
                random_state=SEED,
            )
        else:
            raise ValueError(model_kind)

        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])

    corr = np.corrcoef(y, pred)[0, 1] if np.std(pred) > 0 else 0.0

    return {
        "r2": float(r2_score(y, pred)),
        "corr": float(corr),
        "mae": float(mean_absolute_error(y, pred)),
    }

# ============================================================
# RUN MODELS
# ============================================================

rows = []

for group_mode in [False, True]:
    cv_name = "GroupKFold" if group_mode else "KFold"

    for feature_set, feats in FEATURE_SETS.items():
        for model_kind in ["ridge_linear", "ridge_poly2", "lasso_poly2", "rf"]:
            res = cv_regression(
                df,
                feats,
                MAIN_TARGET,
                model_kind=model_kind,
                group_mode=group_mode,
            )
            rows.append({
                "cv": cv_name,
                "model": model_kind,
                "feature_set": feature_set,
                "features": ",".join(feats),
                **res,
            })

eval_df = pd.DataFrame(rows).sort_values(
    ["cv", "r2"],
    ascending=[True, False],
)
eval_df.to_csv(SAVE_DIR / "phasemap2c_flow_equation_eval.csv", index=False)

# ============================================================
# FIT INTERPRETABLE FINAL EQUATIONS
# ============================================================

equation_rows = []

def fit_ridge_equation(data, features, target, name):
    X = data[features].values.astype(float)
    y = data[target].values.astype(float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    ridge = Ridge(alpha=1.0)
    ridge.fit(Xs, y)

    # Convert standardized coefficients approximately to raw coordinates
    coef_raw = ridge.coef_ / scaler.scale_
    intercept_raw = ridge.intercept_ - np.sum(ridge.coef_ * scaler.mean_ / scaler.scale_)

    equation = f"{intercept_raw:+.6f}"
    for f, c in zip(features, coef_raw):
        equation += f" {c:+.6f}*{f}"

    pred = ridge.predict(Xs)

    return {
        "equation_name": name,
        "model": "ridge_linear_raw_units",
        "features": ",".join(features),
        "intercept": float(intercept_raw),
        "equation": equation,
        "train_r2": float(r2_score(y, pred)),
        "train_corr": float(np.corrcoef(y, pred)[0, 1]),
        "train_mae": float(mean_absolute_error(y, pred)),
        **{f"coef_{f}": float(c) for f, c in zip(features, coef_raw)},
    }

equation_rows.append(
    fit_ridge_equation(df, H_SHAPE, MAIN_TARGET, "dD = F(H_shape)")
)

equation_rows.append(
    fit_ridge_equation(df, H_COMPACT, MAIN_TARGET, "dD = F(H_compact)")
)

equation_rows.append(
    fit_ridge_equation(df, ["H_position", "H_velocity"], MAIN_TARGET, "dD = F(position, velocity)")
)

equation_rows.append(
    fit_ridge_equation(df, H_SHAPE + STATE_PROJ, MAIN_TARGET, "dD = F(H_shape, U_K, D_B)")
)

eq_df = pd.DataFrame(equation_rows)
eq_df.to_csv(SAVE_DIR / "phasemap2c_explicit_equations.csv", index=False)

# ============================================================
# RESIDUAL ANALYSIS
# ============================================================

best_linear_feats = H_SHAPE

X = df[best_linear_feats].values.astype(float)
y = df[MAIN_TARGET].values.astype(float)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0)),
])
model.fit(X, y)
df["pred_dD_H_shape_linear"] = model.predict(X)
df["resid_dD_H_shape_linear"] = df[MAIN_TARGET] - df["pred_dD_H_shape_linear"]
df["abs_resid_dD_H_shape_linear"] = df["resid_dD_H_shape_linear"].abs()

resid_phase = df.groupby("phase_target").agg(
    n=("Gen_E", "count"),
    target_mean=(MAIN_TARGET, "mean"),
    pred_mean=("pred_dD_H_shape_linear", "mean"),
    resid_mean=("resid_dD_H_shape_linear", "mean"),
    abs_resid_mean=("abs_resid_dD_H_shape_linear", "mean"),
    abs_resid_std=("abs_resid_dD_H_shape_linear", "std"),
).reset_index()

resid_condition = df.groupby("condition").agg(
    n=("Gen_E", "count"),
    phase=("phase_target", lambda x: x.mode().iloc[0]),
    target_mean=(MAIN_TARGET, "mean"),
    pred_mean=("pred_dD_H_shape_linear", "mean"),
    resid_mean=("resid_dD_H_shape_linear", "mean"),
    abs_resid_mean=("abs_resid_dD_H_shape_linear", "mean"),
).reset_index()

resid_phase.to_csv(SAVE_DIR / "phasemap2c_residual_by_phase.csv", index=False)
resid_condition.to_csv(SAVE_DIR / "phasemap2c_residual_by_condition.csv", index=False)

# ============================================================
# VECTOR FIELD / PHASE FLOW TABLE
# ============================================================

# Bin H_position x H_velocity to create low-dimensional flow map
df["H_position_bin"] = pd.qcut(df["H_position"], q=8, duplicates="drop")
df["H_velocity_bin"] = pd.qcut(df["H_velocity"], q=8, duplicates="drop")

grid_rows = []

for (pb, vb), sub in df.groupby(["H_position_bin", "H_velocity_bin"], observed=True):
    if len(sub) < 3:
        continue

    grid_rows.append({
        "H_position_bin": str(pb),
        "H_velocity_bin": str(vb),
        "n": int(len(sub)),
        "H_position_center": float(sub["H_position"].mean()),
        "H_velocity_center": float(sub["H_velocity"].mean()),
        "H_energy_center": float(sub["H_energy"].mean()),
        "H_span_center": float(sub["H_span"].mean()),
        "mean_dD": float(sub[MAIN_TARGET].mean()),
        "mean_abs_dD": float(sub[MAIN_TARGET].abs().mean()),
        "mean_pred_dD": float(sub["pred_dD_H_shape_linear"].mean()),
        "GenE_rate": float(sub["Gen_E"].mean()),
        "dominant_phase": sub["phase_target"].mode().iloc[0],
    })

grid_df = pd.DataFrame(grid_rows)
grid_df.to_csv(SAVE_DIR / "phasemap2c_H_flow_grid.csv", index=False)

# ============================================================
# PHASE SUMMARY
# ============================================================

summary_cols = H_SHAPE + H_COMPACT + STATE_PROJ + [
    MAIN_TARGET,
    "pred_dD_H_shape_linear",
    "resid_dD_H_shape_linear",
]

phase_summary = df.groupby("phase_target").agg(
    n=("Gen_E", "count"),
    GenE_rate=("Gen_E", "mean"),
    **{f"{c}_mean": (c, "mean") for c in summary_cols},
    **{f"{c}_std": (c, "std") for c in summary_cols},
).reset_index()

condition_summary = df.groupby("condition").agg(
    n=("Gen_E", "count"),
    phase=("phase_target", lambda x: x.mode().iloc[0]),
    GenE_rate=("Gen_E", "mean"),
    **{f"{c}_mean": (c, "mean") for c in summary_cols},
).reset_index()

phase_summary.to_csv(SAVE_DIR / "phasemap2c_phase_summary.csv", index=False)
condition_summary.to_csv(SAVE_DIR / "phasemap2c_condition_summary.csv", index=False)

df.to_csv(SAVE_DIR / "phasemap2c_eval.csv", index=False)

# ============================================================
# SUMMARY
# ============================================================

def top_rows(cv=None, model=None, n=12):
    sub = eval_df.copy()
    if cv is not None:
        sub = sub[sub["cv"] == cv]
    if model is not None:
        sub = sub[sub["model"] == model]
    return sub.sort_values("r2", ascending=False).head(n).to_dict(orient="records")

def metric(cv, model, feature_set, m):
    sub = eval_df[
        (eval_df["cv"] == cv)
        & (eval_df["model"] == model)
        & (eval_df["feature_set"] == feature_set)
    ]
    if len(sub) == 0:
        return None
    return float(sub.iloc[0][m])

summary = {
    "experiment": "PhaseMap-2C Explicit Flow Equation Audit",
    "input_path": str(INPUT_PATH),
    "n_rows": int(len(df)),
    "group_col": GROUP_COL,
    "main_target": MAIN_TARGET,
    "core_question": "Can H_shape define an explicit low-dimensional flow equation for basin-boundary dynamics?",
    "key_metrics": {
        "KFold_linear_H_shape_r2": metric("KFold", "ridge_linear", "H_shape_linear", "r2"),
        "Group_linear_H_shape_r2": metric("GroupKFold", "ridge_linear", "H_shape_linear", "r2"),
        "KFold_poly2_H_shape_r2": metric("KFold", "ridge_poly2", "H_shape_linear", "r2"),
        "Group_poly2_H_shape_r2": metric("GroupKFold", "ridge_poly2", "H_shape_linear", "r2"),
        "KFold_linear_H_compact_r2": metric("KFold", "ridge_linear", "H_compact_linear", "r2"),
        "Group_linear_H_compact_r2": metric("GroupKFold", "ridge_linear", "H_compact_linear", "r2"),
        "KFold_linear_position_velocity_r2": metric("KFold", "ridge_linear", "H_position_velocity", "r2"),
        "Group_linear_position_velocity_r2": metric("GroupKFold", "ridge_linear", "H_position_velocity", "r2"),
    },
    "top_models": {
        "KFold": top_rows(cv="KFold"),
        "GroupKFold": top_rows(cv="GroupKFold"),
        "linear_only": top_rows(model="ridge_linear"),
    },
    "explicit_equations": eq_df.to_dict(orient="records"),
    "residual_by_phase": resid_phase.to_dict(orient="records"),
    "residual_by_condition": resid_condition.to_dict(orient="records"),
    "phase_summary": phase_summary.to_dict(orient="records"),
    "condition_summary": condition_summary.to_dict(orient="records"),
    "interpretation_rules": {
        "PASS_explicit_flow": [
            "Linear H_shape equation reaches GroupKFold R2 > 0.6.",
            "Compact H equation remains close to full H_shape.",
            "Residuals are not dominated by a single phase or condition."
        ],
        "PASS_nonlinear_flow": [
            "Polynomial/RF strongly beats linear, indicating nonlinear basin flow.",
            "Need PhaseMap-2D nonlinear phase portrait."
        ],
        "FAIL_equation": [
            "Explicit H_shape equations collapse under GroupKFold.",
            "2B success was model/feature-set predictive but not equation-like."
        ],
        "main_theory_update_if_pass": "H_shape is not merely memory; it is a low-dimensional dynamical state with an explicit flow equation."
    }
}

with open(SAVE_DIR / "phasemap2c_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nSaved to: {SAVE_DIR.resolve()}")