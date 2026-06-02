import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold, GroupKFold

# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path(r"C:\Windows\System32\phasemap2c_outputs\phasemap2c_eval.csv")
SAVE_DIR = Path("./phasemap3a1b_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
R_LAYERS = list(range(20, 27))

# ============================================================
# LOAD
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Cannot find input file: {INPUT_PATH.resolve()}")

df = pd.read_csv(INPUT_PATH)

required = ["phase_target", "condition"] + [f"R_{l}" for l in R_LAYERS]

missing = [c for c in required if c not in df.columns]
if missing:
    print("Available columns:")
    print(df.columns.tolist())
    raise RuntimeError(f"Missing columns: {missing}")

df = df.copy()
df["phase_target"] = df["phase_target"].astype(str)
df["condition"] = df["condition"].astype(str)

GROUP_COL = "graph_id" if "graph_id" in df.columns else "condition"

# ============================================================
# BUILD STEP DATASET
#
# H_l uses local trajectory information around l:
#
#   R_l
#   v_l      = R_{l+1} - R_l
#   a_l      = v_l - v_{l-1}
#   energy   = local R^2 + v^2
#   span     = local max(R)-min(R)
#   phase    = layer / coarse phase
#
# Predict:
#   R_next = R_{l+1}
#   v_next = R_{l+2} - R_{l+1}
#
# Requires R_{l-1}, R_l, R_{l+1}, R_{l+2}
# Therefore l = 21..24.
# ============================================================

rows = []

for idx, row in df.iterrows():
    for l in range(21, 25):
        R_prev = float(row[f"R_{l-1}"])
        R_l = float(row[f"R_{l}"])
        R_next = float(row[f"R_{l+1}"])
        R_next2 = float(row[f"R_{l+2}"])

        v_prev = R_l - R_prev
        v_l = R_next - R_l
        v_next = R_next2 - R_next

        a_l = v_l - v_prev

        local_vals = np.array([R_prev, R_l, R_next], dtype=float)
        local_span = float(local_vals.max() - local_vals.min())
        local_mean = float(local_vals.mean())
        local_energy = float(R_l ** 2 + v_l ** 2)
        local_abs_energy = float(abs(R_l) + abs(v_l))
        local_curvature_energy = float(a_l ** 2)

        # coarse phase clock
        if l <= 22:
            phase_clock = 0  # entry / critical gateway
        else:
            phase_clock = 1  # basin formation

        rows.append({
            "source_row": int(idx),
            "group_id": row[GROUP_COL],
            "condition": row["condition"],
            "phase_target": row["phase_target"],
            "layer": int(l),
            "phase_clock": int(phase_clock),

            # current state
            "R_l": R_l,
            "v_l": v_l,
            "v_prev": v_prev,
            "a_l": a_l,
            "abs_R_l": abs(R_l),
            "abs_v_l": abs(v_l),
            "abs_a_l": abs(a_l),

            "local_mean": local_mean,
            "local_span": local_span,
            "local_energy": local_energy,
            "local_abs_energy": local_abs_energy,
            "curvature_energy": local_curvature_energy,

            # next state
            "R_next": R_next,
            "v_next": v_next,
            "abs_R_next": abs(R_next),
            "abs_v_next": abs(v_next),
        })

if len(rows) == 0:
    raise RuntimeError("No step rows created. Check R_20~R_26 columns.")

step_df = pd.DataFrame(rows)

step_df["layer_norm"] = (
    (step_df["layer"] - step_df["layer"].mean())
    / max(1e-9, step_df["layer"].std())
)

print("Original df shape:", df.shape)
print("Step df shape:", step_df.shape)
print(step_df.head())

# ============================================================
# FEATURE SETS
# ============================================================

BASE = ["R_l", "v_l"]

FEATURE_SETS = {
    # 3A.1 baseline
    "H_pos_vel": ["R_l", "v_l"],

    # Add local acceleration / curvature
    "H_pos_vel_acc": ["R_l", "v_l", "a_l"],
    "H_pos_vel_abs_acc": ["R_l", "v_l", "a_l", "abs_R_l", "abs_v_l", "abs_a_l"],

    # Add local shape variables
    "H_shape_local": [
        "R_l", "v_l", "a_l",
        "local_mean", "local_span",
        "local_energy", "local_abs_energy",
        "curvature_energy",
    ],

    # Add explicit layer clock
    "H_pos_vel_layer": ["R_l", "v_l", "layer_norm"],
    "H_pos_vel_acc_layer": ["R_l", "v_l", "a_l", "layer_norm"],

    # Add coarse phase clock
    "H_pos_vel_phaseclock": ["R_l", "v_l", "phase_clock"],
    "H_pos_vel_acc_phaseclock": ["R_l", "v_l", "a_l", "phase_clock"],

    # Full candidates
    "H_shape_plus_layer": [
        "R_l", "v_l", "a_l",
        "local_mean", "local_span",
        "local_energy", "local_abs_energy",
        "curvature_energy",
        "layer_norm",
    ],

    "H_shape_plus_phaseclock": [
        "R_l", "v_l", "a_l",
        "local_mean", "local_span",
        "local_energy", "local_abs_energy",
        "curvature_energy",
        "phase_clock",
    ],

    "H_shape_plus_both_clocks": [
        "R_l", "v_l", "a_l",
        "local_mean", "local_span",
        "local_energy", "local_abs_energy",
        "curvature_energy",
        "layer_norm",
        "phase_clock",
    ],
}

TARGETS = ["R_next", "v_next"]

# ============================================================
# HELPERS
# ============================================================

def get_splits(data, group_mode=False):
    if group_mode:
        groups = data["group_id"].values
        if len(np.unique(groups)) >= 5:
            return list(GroupKFold(n_splits=5).split(data, groups=groups))

    return list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(data))


def make_model(kind):
    if kind == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ])

    if kind == "poly2":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("ridge", Ridge(alpha=1.0)),
        ])

    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=5,
            random_state=SEED,
        )

    raise ValueError(kind)


def cv_regression(data, features, target, model_kind, group_mode=False):
    X = data[features].values.astype(float)
    y = data[target].values.astype(float)

    splits = get_splits(data, group_mode=group_mode)
    pred = np.zeros(len(data), dtype=float)

    for tr, te in splits:
        model = make_model(model_kind)
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])

    corr = 0.0
    if np.std(pred) > 0 and np.std(y) > 0:
        corr = float(np.corrcoef(y, pred)[0, 1])

    return {
        "r2": float(r2_score(y, pred)),
        "corr": corr,
        "mae": float(mean_absolute_error(y, pred)),
    }


def cv_multioutput(data, features, targets, model_kind, group_mode=False):
    out = {}
    r2s, corrs, maes = [], [], []

    for target in targets:
        res = cv_regression(data, features, target, model_kind, group_mode)
        out[f"{target}_r2"] = res["r2"]
        out[f"{target}_corr"] = res["corr"]
        out[f"{target}_mae"] = res["mae"]

        r2s.append(res["r2"])
        corrs.append(res["corr"])
        maes.append(res["mae"])

    out["mean_r2"] = float(np.mean(r2s))
    out["mean_corr"] = float(np.mean(corrs))
    out["mean_mae"] = float(np.mean(maes))

    return out

# ============================================================
# GLOBAL EVAL
# ============================================================

eval_rows = []

for group_mode in [False, True]:
    cv_name = "GroupKFold" if group_mode else "KFold"

    for model_kind in ["ridge", "poly2", "rf"]:
        for feature_set, features in FEATURE_SETS.items():
            res = cv_multioutput(
                step_df,
                features,
                TARGETS,
                model_kind,
                group_mode,
            )

            eval_rows.append({
                "scope": "global",
                "cv": cv_name,
                "model": model_kind,
                "feature_set": feature_set,
                "features": ",".join(features),
                **res,
            })

eval_df = pd.DataFrame(eval_rows).sort_values(
    ["cv", "mean_r2"],
    ascending=[True, False],
)

eval_df.to_csv(SAVE_DIR / "phasemap3a1b_state_eval.csv", index=False)

# ============================================================
# ABLATION / CLOCK DIAGNOSTIC
# ============================================================

def get_metric(table, cv, model, feature_set, metric):
    sub = table[
        (table["cv"] == cv)
        & (table["model"] == model)
        & (table["feature_set"] == feature_set)
    ]
    if len(sub) == 0:
        return None
    return float(sub.iloc[0][metric])

ablation_rows = []

for cv in ["KFold", "GroupKFold"]:
    for model in ["ridge", "poly2", "rf"]:
        base = get_metric(eval_df, cv, model, "H_pos_vel", "mean_r2")
        acc = get_metric(eval_df, cv, model, "H_pos_vel_acc", "mean_r2")
        shape = get_metric(eval_df, cv, model, "H_shape_local", "mean_r2")
        layer = get_metric(eval_df, cv, model, "H_pos_vel_layer", "mean_r2")
        phase = get_metric(eval_df, cv, model, "H_pos_vel_phaseclock", "mean_r2")
        shape_layer = get_metric(eval_df, cv, model, "H_shape_plus_layer", "mean_r2")
        shape_phase = get_metric(eval_df, cv, model, "H_shape_plus_phaseclock", "mean_r2")

        ablation_rows.append({
            "cv": cv,
            "model": model,
            "base_H_pos_vel": base,
            "plus_acc": acc,
            "plus_shape": shape,
            "plus_layer": layer,
            "plus_phaseclock": phase,
            "shape_plus_layer": shape_layer,
            "shape_plus_phaseclock": shape_phase,
            "gain_acc_over_base": None if base is None or acc is None else acc - base,
            "gain_shape_over_base": None if base is None or shape is None else shape - base,
            "gain_layer_over_base": None if base is None or layer is None else layer - base,
            "gain_phaseclock_over_base": None if base is None or phase is None else phase - base,
            "gain_layer_over_shape": None if shape is None or shape_layer is None else shape_layer - shape,
            "gain_phase_over_shape": None if shape is None or shape_phase is None else shape_phase - shape,
        })

ablation_df = pd.DataFrame(ablation_rows)
ablation_df.to_csv(SAVE_DIR / "phasemap3a1b_ablation_clock.csv", index=False)

# ============================================================
# PHASE-SPECIFIC EVAL
# ============================================================

phase_rows = []

for phase, sub in step_df.groupby("phase_target"):
    if len(sub) < 30:
        continue

    for model_kind in ["ridge", "poly2", "rf"]:
        for feature_set, features in FEATURE_SETS.items():
            res = cv_multioutput(
                sub,
                features,
                TARGETS,
                model_kind,
                group_mode=False,
            )

            phase_rows.append({
                "phase_target": phase,
                "n": int(len(sub)),
                "cv": "KFold",
                "model": model_kind,
                "feature_set": feature_set,
                "features": ",".join(features),
                **res,
            })

phase_eval = pd.DataFrame(phase_rows).sort_values(
    ["phase_target", "mean_r2"],
    ascending=[True, False],
)

phase_eval.to_csv(SAVE_DIR / "phasemap3a1b_phase_eval.csv", index=False)

# ============================================================
# LAYER-SPECIFIC EVAL
# ============================================================

layer_rows = []

for layer, sub in step_df.groupby("layer"):
    if len(sub) < 30:
        continue

    for model_kind in ["ridge", "poly2", "rf"]:
        for feature_set, features in FEATURE_SETS.items():
            if "layer" in feature_set or "phaseclock" in feature_set:
                continue

            res = cv_multioutput(
                sub,
                features,
                TARGETS,
                model_kind,
                group_mode=False,
            )

            layer_rows.append({
                "layer_transition": f"L{layer}->L{layer+1}",
                "n": int(len(sub)),
                "cv": "KFold",
                "model": model_kind,
                "feature_set": feature_set,
                "features": ",".join(features),
                **res,
            })

layer_eval = pd.DataFrame(layer_rows).sort_values(
    ["layer_transition", "mean_r2"],
    ascending=[True, False],
)

layer_eval.to_csv(SAVE_DIR / "phasemap3a1b_layer_eval.csv", index=False)

# ============================================================
# EXPLICIT EQUATIONS
# ============================================================

def fit_raw_ridge_equation(data, features, target):
    X = data[features].values.astype(float)
    y = data[target].values.astype(float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = Ridge(alpha=1.0)
    model.fit(Xs, y)

    coef_raw = model.coef_ / scaler.scale_
    intercept_raw = model.intercept_ - np.sum(
        model.coef_ * scaler.mean_ / scaler.scale_
    )

    pred = model.predict(Xs)

    equation = f"{target} = {intercept_raw:+.6f}"
    for f, c in zip(features, coef_raw):
        equation += f" {c:+.6f}*{f}"

    return {
        "target": target,
        "features": ",".join(features),
        "equation": equation,
        "train_r2": float(r2_score(y, pred)),
        "train_corr": float(np.corrcoef(y, pred)[0, 1]),
        "train_mae": float(mean_absolute_error(y, pred)),
        "intercept": float(intercept_raw),
        **{f"coef_{f}": float(c) for f, c in zip(features, coef_raw)},
    }

equation_rows = []

EQUATION_FEATURE_SETS = {
    "pos_vel": ["R_l", "v_l"],
    "pos_vel_acc": ["R_l", "v_l", "a_l"],
    "shape_local": [
        "R_l", "v_l", "a_l",
        "local_mean", "local_span",
        "local_energy", "local_abs_energy",
        "curvature_energy",
    ],
    "shape_layer": [
        "R_l", "v_l", "a_l",
        "local_mean", "local_span",
        "local_energy", "local_abs_energy",
        "curvature_energy",
        "layer_norm",
    ],
}

for name, feats in EQUATION_FEATURE_SETS.items():
    for target in TARGETS:
        row = fit_raw_ridge_equation(step_df, feats, target)
        row["equation_feature_set"] = name
        equation_rows.append(row)

eq_df = pd.DataFrame(equation_rows)
eq_df.to_csv(SAVE_DIR / "phasemap3a1b_explicit_equations.csv", index=False)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_rows = []

for feature_set in [
    "H_shape_local",
    "H_shape_plus_layer",
    "H_shape_plus_phaseclock",
    "H_shape_plus_both_clocks",
]:
    feats = FEATURE_SETS[feature_set]
    X = step_df[feats].values.astype(float)

    for target in TARGETS:
        y = step_df[target].values.astype(float)

        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=5,
            random_state=SEED,
        )
        model.fit(X, y)

        for f, imp in zip(feats, model.feature_importances_):
            importance_rows.append({
                "target": target,
                "feature_set": feature_set,
                "feature": f,
                "importance": float(imp),
            })

importance_df = pd.DataFrame(importance_rows).sort_values(
    ["target", "feature_set", "importance"],
    ascending=[True, True, False],
)

importance_df.to_csv(SAVE_DIR / "phasemap3a1b_feature_importance.csv", index=False)

# ============================================================
# SAVE STEP DATASET
# ============================================================

step_df.to_csv(SAVE_DIR / "phasemap3a1b_step_dataset.csv", index=False)

# ============================================================
# SUMMARY
# ============================================================

def metric(cv, model, feature_set, key):
    return get_metric(eval_df, cv, model, feature_set, key)

def top_rows(table, cv=None, model=None, n=12):
    sub = table.copy()
    if cv is not None:
        sub = sub[sub["cv"] == cv]
    if model is not None:
        sub = sub[sub["model"] == model]

    return (
        sub.sort_values("mean_r2", ascending=False)
        .head(n)
        .to_dict(orient="records")
    )

summary = {
    "experiment": "PhaseMap-3A.1b Missing State Variable Audit",
    "input_path": str(INPUT_PATH),
    "n_original_rows": int(len(df)),
    "n_step_rows": int(len(step_df)),
    "group_col": GROUP_COL,
    "state_definition": {
        "base": "H_l=(R_l, v_l)",
        "added_candidates": [
            "a_l = v_l - v_prev",
            "local_mean",
            "local_span",
            "local_energy",
            "curvature_energy",
            "layer_norm",
            "phase_clock"
        ],
        "target": "H_{l+1}=(R_next, v_next)"
    },
    "key_metrics": {
        "Group_rf_H_pos_vel_mean_r2": metric("GroupKFold", "rf", "H_pos_vel", "mean_r2"),
        "Group_rf_H_pos_vel_acc_mean_r2": metric("GroupKFold", "rf", "H_pos_vel_acc", "mean_r2"),
        "Group_rf_H_shape_local_mean_r2": metric("GroupKFold", "rf", "H_shape_local", "mean_r2"),
        "Group_rf_H_pos_vel_layer_mean_r2": metric("GroupKFold", "rf", "H_pos_vel_layer", "mean_r2"),
        "Group_rf_H_shape_plus_layer_mean_r2": metric("GroupKFold", "rf", "H_shape_plus_layer", "mean_r2"),
        "Group_rf_H_shape_plus_phaseclock_mean_r2": metric("GroupKFold", "rf", "H_shape_plus_phaseclock", "mean_r2"),
        "Group_rf_H_shape_plus_both_clocks_mean_r2": metric("GroupKFold", "rf", "H_shape_plus_both_clocks", "mean_r2"),

        "Group_poly2_H_shape_local_mean_r2": metric("GroupKFold", "poly2", "H_shape_local", "mean_r2"),
        "Group_poly2_H_shape_plus_layer_mean_r2": metric("GroupKFold", "poly2", "H_shape_plus_layer", "mean_r2"),
    },
    "top_global": {
        "KFold": top_rows(eval_df, cv="KFold"),
        "GroupKFold": top_rows(eval_df, cv="GroupKFold"),
        "rf": top_rows(eval_df, model="rf"),
        "poly2": top_rows(eval_df, model="poly2"),
        "ridge": top_rows(eval_df, model="ridge"),
    },
    "ablation_clock": ablation_df.to_dict(orient="records"),
    "feature_importance_top": importance_df.head(40).to_dict(orient="records"),
    "explicit_equations": eq_df.to_dict(orient="records"),
    "interpretation_rules": {
        "PASS_minimal_state": [
            "H_shape_local reaches GroupKFold mean_r2 > 0.90.",
            "Adding layer_norm / phase_clock gives little extra gain.",
            "This supports a closed autonomous state using local shape variables."
        ],
        "PASS_phase_clock_state": [
            "H_shape_local improves, but clocks still add large gain.",
            "State is non-autonomous: H plus phase/layer clock is required."
        ],
        "PASS_acceleration_state": [
            "Adding a_l gives most of the gain over H_pos_vel.",
            "Missing variable was curvature/acceleration."
        ],
        "FAIL_state_closure": [
            "Even H_shape_plus_clock stays below mean_r2 ~0.85.",
            "Need broader history window or prompt-conditioned variables."
        ],
        "next_if_pass_minimal": "PhaseMap-3A.2 Multi-Step Rollout without layer clock.",
        "next_if_phase_clock": "PhaseMap-3A.2b Multi-Step Rollout with phase/layer clock.",
        "next_if_fail": "PhaseMap-3B broader history-state discovery."
    }
}

with open(SAVE_DIR / "phasemap3a1b_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nSaved to: {SAVE_DIR.resolve()}")