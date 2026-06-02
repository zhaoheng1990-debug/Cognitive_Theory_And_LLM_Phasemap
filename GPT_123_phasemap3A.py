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
SAVE_DIR = Path("./phasemap3a1_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
R_LAYERS = list(range(20, 27))  # R_20 ... R_26

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
# BUILD ONE-STEP STATE DATASET
#
# H_l = (R_l, v_l)
# v_l = R_{l+1} - R_l
#
# Predict:
# H_{l+1} = (R_{l+1}, v_{l+1})
# v_{l+1} = R_{l+2} - R_{l+1}
#
# Requires R_l, R_{l+1}, R_{l+2}
# Therefore transitions:
# L20 -> L21
# L21 -> L22
# L22 -> L23
# L23 -> L24
# L24 -> L25
# ============================================================

rows = []

for idx, row in df.iterrows():
    for l in range(20, 25):
        R_l = float(row[f"R_{l}"])
        R_next = float(row[f"R_{l+1}"])
        R_next2 = float(row[f"R_{l+2}"])

        v_l = R_next - R_l
        v_next = R_next2 - R_next

        rows.append({
            "source_row": int(idx),
            "group_id": row[GROUP_COL],
            "condition": row["condition"],
            "phase_target": row["phase_target"],
            "layer": int(l),

            "R_l": R_l,
            "v_l": v_l,
            "abs_R_l": abs(R_l),
            "abs_v_l": abs(v_l),

            "R_next": R_next,
            "v_next": v_next,
            "abs_R_next": abs(R_next),
            "abs_v_next": abs(v_next),
        })

if len(rows) == 0:
    raise RuntimeError("No step rows were created. Check R_20~R_26 columns.")

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

FEATURE_SETS = {
    "R_only": ["R_l"],
    "v_only": ["v_l"],
    "H_pos_vel": ["R_l", "v_l"],
    "H_pos_vel_abs": ["R_l", "v_l", "abs_R_l", "abs_v_l"],

    # If this is much stronger, dynamics are layer-clock dependent.
    "H_pos_vel_layer": ["R_l", "v_l", "layer_norm"],
    "H_pos_vel_abs_layer": ["R_l", "v_l", "abs_R_l", "abs_v_l", "layer_norm"],
}

TARGETS = ["R_next", "v_next"]

# ============================================================
# HELPERS
# ============================================================

def get_splits(data, group_mode=False):
    if group_mode:
        groups = data["group_id"].values
        n_groups = len(np.unique(groups))

        if n_groups >= 5:
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

    raise ValueError(f"Unknown model kind: {kind}")


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

    r2s = []
    corrs = []
    maes = []

    for target in targets:
        res = cv_regression(
            data=data,
            features=features,
            target=target,
            model_kind=model_kind,
            group_mode=group_mode,
        )

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
# GLOBAL ONE-STEP AUDIT
# ============================================================

eval_rows = []

for group_mode in [False, True]:
    cv_name = "GroupKFold" if group_mode else "KFold"

    for model_kind in ["ridge", "poly2", "rf"]:
        for feature_set, features in FEATURE_SETS.items():
            res = cv_multioutput(
                data=step_df,
                features=features,
                targets=TARGETS,
                model_kind=model_kind,
                group_mode=group_mode,
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

eval_df.to_csv(SAVE_DIR / "phasemap3a1_one_step_eval.csv", index=False)

# ============================================================
# PHASE-SPECIFIC AUDIT
# ============================================================

phase_rows = []

for phase, sub in step_df.groupby("phase_target"):
    if len(sub) < 30:
        continue

    for model_kind in ["ridge", "poly2", "rf"]:
        for feature_set, features in FEATURE_SETS.items():
            res = cv_multioutput(
                data=sub,
                features=features,
                targets=TARGETS,
                model_kind=model_kind,
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

phase_eval.to_csv(SAVE_DIR / "phasemap3a1_phase_eval.csv", index=False)

# ============================================================
# LAYER-SPECIFIC AUDIT
# ============================================================

layer_rows = []

for layer, sub in step_df.groupby("layer"):
    if len(sub) < 30:
        continue

    for model_kind in ["ridge", "poly2", "rf"]:
        for feature_set, features in FEATURE_SETS.items():
            if "layer" in feature_set:
                continue

            res = cv_multioutput(
                data=sub,
                features=features,
                targets=TARGETS,
                model_kind=model_kind,
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

layer_eval.to_csv(SAVE_DIR / "phasemap3a1_layer_eval.csv", index=False)

# ============================================================
# EXPLICIT LINEAR EQUATIONS
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

for target in TARGETS:
    equation_rows.append(
        fit_raw_ridge_equation(step_df, ["R_l", "v_l"], target)
    )
    equation_rows.append(
        fit_raw_ridge_equation(step_df, ["R_l", "v_l", "layer_norm"], target)
    )

eq_df = pd.DataFrame(equation_rows)
eq_df.to_csv(SAVE_DIR / "phasemap3a1_explicit_equations.csv", index=False)

# ============================================================
# SAVE STEP DATASET
# ============================================================

step_df.to_csv(SAVE_DIR / "phasemap3a1_step_dataset.csv", index=False)

# ============================================================
# SUMMARY
# ============================================================

def metric(cv, model, feature_set, key):
    sub = eval_df[
        (eval_df["cv"] == cv)
        & (eval_df["model"] == model)
        & (eval_df["feature_set"] == feature_set)
    ]
    if len(sub) == 0:
        return None

    val = sub.iloc[0][key]
    return None if pd.isna(val) else float(val)


def top_rows(table, cv=None, model=None, n=10):
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
    "experiment": "PhaseMap-3A.1 One-Step Closure Audit FIXED",
    "input_path": str(INPUT_PATH),
    "n_original_rows": int(len(df)),
    "n_step_rows": int(len(step_df)),
    "group_col": GROUP_COL,
    "state_definition": {
        "H_l": [
            "R_l",
            "v_l = R_{l+1} - R_l"
        ],
        "H_next": [
            "R_next = R_{l+1}",
            "v_next = R_{l+2} - R_{l+1}"
        ],
        "note": "No R_19 required."
    },
    "key_metrics": {
        "KFold_ridge_H_pos_vel_mean_r2": metric("KFold", "ridge", "H_pos_vel", "mean_r2"),
        "KFold_poly2_H_pos_vel_mean_r2": metric("KFold", "poly2", "H_pos_vel", "mean_r2"),
        "KFold_rf_H_pos_vel_mean_r2": metric("KFold", "rf", "H_pos_vel", "mean_r2"),

        "Group_ridge_H_pos_vel_mean_r2": metric("GroupKFold", "ridge", "H_pos_vel", "mean_r2"),
        "Group_poly2_H_pos_vel_mean_r2": metric("GroupKFold", "poly2", "H_pos_vel", "mean_r2"),
        "Group_rf_H_pos_vel_mean_r2": metric("GroupKFold", "rf", "H_pos_vel", "mean_r2"),

        "Group_poly2_H_pos_vel_layer_mean_r2": metric("GroupKFold", "poly2", "H_pos_vel_layer", "mean_r2"),
        "Group_rf_H_pos_vel_layer_mean_r2": metric("GroupKFold", "rf", "H_pos_vel_layer", "mean_r2"),
    },
    "top_global": {
        "KFold": top_rows(eval_df, cv="KFold"),
        "GroupKFold": top_rows(eval_df, cv="GroupKFold"),
        "ridge": top_rows(eval_df, model="ridge"),
        "poly2": top_rows(eval_df, model="poly2"),
        "rf": top_rows(eval_df, model="rf"),
    },
    "explicit_equations": eq_df.to_dict(orient="records"),
    "interpretation_rules": {
        "PASS_autonomous_H": [
            "H_pos_vel reaches mean_r2 > 0.80 under GroupKFold.",
            "Adding layer_norm gives little improvement.",
            "This supports H=(position,velocity) as an approximately closed state."
        ],
        "PASS_layer_clock": [
            "H_pos_vel is moderate but H_pos_vel_layer greatly improves.",
            "Dynamics require explicit layer/time variable."
        ],
        "FAIL_closure": [
            "H_pos_vel cannot predict H_next under GroupKFold.",
            "H_shape from 2C may be predictive but not a closed dynamical state."
        ],
        "next_if_pass": "PhaseMap-3A.2 Multi-Step Rollout Audit.",
        "next_if_layer_clock": "PhaseMap-3A.1b add phase/layer-clock as state variable.",
        "next_if_fail": "Return to state discovery; include H_energy/H_span or prompt-conditioned variables."
    }
}

with open(SAVE_DIR / "phasemap3a1_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nSaved to: {SAVE_DIR.resolve()}")