import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold, GroupKFold

# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path(r"C:\Windows\System32\phasemap3b_outputs\phasemap3b_enriched_energy_error.csv")
SAVE_DIR = Path("./phasemap3d1_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
LAYERS = [23, 24, 25]  # need v_l and v_next up to 26

EPS = 1e-6

# ============================================================
# LOAD
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Missing input: {INPUT_PATH.resolve()}")

df = pd.read_csv(INPUT_PATH)

if "model" not in df.columns:
    df["model"] = "unknown"

required = [
    "cv", "model", "feature_set", "condition", "phase_target",
] + [f"true_R_{l}" for l in range(20, 27)] + [f"pred_R_{l}" for l in range(20, 27)]

missing = [c for c in required if c not in df.columns]
if missing:
    print(df.columns.tolist())
    raise RuntimeError(f"Missing columns: {missing}")

GROUP_COL = "group_id" if "group_id" in df.columns else "condition"

# ============================================================
# DERIVE VELOCITY / DAMPING TARGETS
# ============================================================

for prefix in ["true", "pred"]:
    for a, b in zip(range(20, 26), range(21, 27)):
        df[f"{prefix}_v_{a}_{b}"] = df[f"{prefix}_R_{b}"] - df[f"{prefix}_R_{a}"]

    for a, b, c in zip(range(20, 25), range(21, 26), range(22, 27)):
        df[f"{prefix}_a_{a}_{b}_{c}"] = df[f"{prefix}_v_{b}_{c}"] - df[f"{prefix}_v_{a}_{b}"]

for l in range(20, 27):
    df[f"err_R_{l}"] = df[f"pred_R_{l}"] - df[f"true_R_{l}"]
    df[f"abs_err_R_{l}"] = df[f"err_R_{l}"].abs()

ENERGY_COLS = [
    c for c in [
        "E_v_init",
        "E_a_init",
        "E_va_init",
        "E_abs_init",
        "E_span_init",
        "E_mixed_init",
        "E_quad_mixed_init",
        "E_mixed_L22",
        "E_quad_L22",
        "true_E_decay_22_to_24",
        "pred_E_decay_22_to_24",
        "energy_decay_gap",
    ] if c in df.columns
]

if len(ENERGY_COLS) == 0:
    raise RuntimeError("No energy columns found. Run PhaseMap-3B first.")

# ============================================================
# STEP DATASET
#
# Damping model:
#
#   v_true_next ≈ v_pred_current - lambda(E,state) * v_pred_current
#
# Therefore:
#
#   lambda_target = (v_pred_current - v_true_next) / v_pred_current
#
# But only valid when |v_pred_current| is not too small.
#
# Also test direct velocity correction:
#
#   delta_v = v_pred_next - v_true_next
#
# ============================================================

rows = []

for idx, row in df.iterrows():
    for l in LAYERS:
        v_cur = float(row[f"pred_v_{l-1}_{l}"])
        v_next_pred = float(row[f"pred_v_{l}_{l+1}"])
        v_next_true = float(row[f"true_v_{l}_{l+1}"])

        delta_v_error = v_next_pred - v_next_true

        lambda_target = np.nan
        if abs(v_cur) > EPS:
            # correction term: subtract lambda * v_cur from pred next velocity
            lambda_target = delta_v_error / v_cur

        item = {
            "source_row": idx,
            "candidate_cv": row["cv"],
            "candidate_model": row["model"],
            "candidate_feature_set": row["feature_set"],
            "condition": row["condition"],
            "phase_target": row["phase_target"],
            "group_id": row[GROUP_COL],

            "layer": l,
            "layer_norm": (l - np.mean(LAYERS)) / np.std(LAYERS),

            "R_pred_l": float(row[f"pred_R_{l}"]),
            "R_true_l": float(row[f"true_R_{l}"]),
            "err_R_l": float(row[f"err_R_{l}"]),

            "v_cur": v_cur,
            "v_next_pred": v_next_pred,
            "v_next_true": v_next_true,
            "delta_v_error": delta_v_error,
            "lambda_target": lambda_target,

            "a_pred": float(row.get(f"pred_a_{l-2}_{l-1}_{l}", 0.0)),
            "abs_R_pred": abs(float(row[f"pred_R_{l}"])),
            "abs_v_cur": abs(v_cur),
            "abs_a_pred": abs(float(row.get(f"pred_a_{l-2}_{l-1}_{l}", 0.0))),
            "R_times_v": float(row[f"pred_R_{l}"]) * v_cur,
        }

        for c in ENERGY_COLS:
            item[c] = row[c]

        rows.append(item)

step = pd.DataFrame(rows)

# avoid unstable lambda outliers
lambda_step = step[np.isfinite(step["lambda_target"])].copy()
lambda_step = lambda_step[lambda_step["lambda_target"].abs() < 20].copy()

# ============================================================
# FEATURE SETS
# ============================================================

FEATURE_SETS = {
    "energy_only": ENERGY_COLS,

    "state_only": [
        "R_pred_l", "v_cur", "a_pred",
        "abs_R_pred", "abs_v_cur", "abs_a_pred",
        "layer_norm",
    ],

    "energy_plus_state": [
        "R_pred_l", "v_cur", "a_pred",
        "abs_R_pred", "abs_v_cur", "abs_a_pred",
        "R_times_v", "layer_norm",
    ] + ENERGY_COLS,

    "minimal_damping": [
        "v_cur", "abs_v_cur",
        "E_mixed_L22" if "E_mixed_L22" in ENERGY_COLS else ENERGY_COLS[0],
        "E_abs_init" if "E_abs_init" in ENERGY_COLS else ENERGY_COLS[0],
        "layer_norm",
    ],
}

# remove duplicate features
FEATURE_SETS = {k: list(dict.fromkeys(v)) for k, v in FEATURE_SETS.items()}

# ============================================================
# HELPERS
# ============================================================

def get_splits(data, group_mode=True):
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

    if kind == "poly2_ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("ridge", Ridge(alpha=1.0)),
        ])

    raise ValueError(kind)


def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def eval_velocity_correction(data, pred_lambda):
    out = data.copy()
    out["pred_lambda"] = pred_lambda
    out["v_next_corrected"] = out["v_next_pred"] - out["pred_lambda"] * out["v_cur"]

    out["base_v_abs_error"] = (out["v_next_pred"] - out["v_next_true"]).abs()
    out["corr_v_abs_error"] = (out["v_next_corrected"] - out["v_next_true"]).abs()

    return {
        "lambda_r2": float(r2_score(out["lambda_target"], out["pred_lambda"])),
        "lambda_corr": safe_corr(out["lambda_target"], out["pred_lambda"]),
        "lambda_mae": float(mean_absolute_error(out["lambda_target"], out["pred_lambda"])),

        "baseline_v_mae": float(out["base_v_abs_error"].mean()),
        "corrected_v_mae": float(out["corr_v_abs_error"].mean()),
        "delta_v_mae": float(out["corr_v_abs_error"].mean() - out["base_v_abs_error"].mean()),

        "baseline_v_r2": float(r2_score(out["v_next_true"], out["v_next_pred"])),
        "corrected_v_r2": float(r2_score(out["v_next_true"], out["v_next_corrected"])),
        "delta_v_r2": float(
            r2_score(out["v_next_true"], out["v_next_corrected"])
            - r2_score(out["v_next_true"], out["v_next_pred"])
        ),
    }, out


def eval_direct_delta_v(data, pred_delta):
    out = data.copy()
    out["pred_delta_v_error"] = pred_delta
    out["v_next_corrected"] = out["v_next_pred"] - out["pred_delta_v_error"]

    out["base_v_abs_error"] = (out["v_next_pred"] - out["v_next_true"]).abs()
    out["corr_v_abs_error"] = (out["v_next_corrected"] - out["v_next_true"]).abs()

    return {
        "delta_v_error_r2": float(r2_score(out["delta_v_error"], out["pred_delta_v_error"])),
        "delta_v_error_corr": safe_corr(out["delta_v_error"], out["pred_delta_v_error"]),
        "delta_v_error_mae": float(mean_absolute_error(out["delta_v_error"], out["pred_delta_v_error"])),

        "baseline_v_mae": float(out["base_v_abs_error"].mean()),
        "corrected_v_mae": float(out["corr_v_abs_error"].mean()),
        "delta_v_mae": float(out["corr_v_abs_error"].mean() - out["base_v_abs_error"].mean()),

        "baseline_v_r2": float(r2_score(out["v_next_true"], out["v_next_pred"])),
        "corrected_v_r2": float(r2_score(out["v_next_true"], out["v_next_corrected"])),
        "delta_v_r2": float(
            r2_score(out["v_next_true"], out["v_next_corrected"])
            - r2_score(out["v_next_true"], out["v_next_pred"])
        ),
    }, out

# ============================================================
# MAIN CV AUDIT
# ============================================================

lambda_rows = []
direct_rows = []
prediction_rows = []

for cand, sub0 in lambda_step.groupby(["candidate_cv", "candidate_model", "candidate_feature_set"]):
    cand_cv, cand_model, cand_fs = cand
    sub0 = sub0.reset_index(drop=True)

    if len(sub0) < 100:
        continue

    for group_mode in [False, True]:
        audit_cv = "GroupKFold" if group_mode else "KFold"
        splits = get_splits(sub0, group_mode=group_mode)

        for fs_name, feats in FEATURE_SETS.items():
            X = sub0[feats].values.astype(float)

            for model_kind in ["ridge", "poly2_ridge"]:
                # -------------------------------
                # A. Explicit lambda(E) model
                # -------------------------------
                y_lambda = sub0["lambda_target"].values.astype(float)
                pred_lambda = np.zeros(len(sub0), dtype=float)

                for tr, te in splits:
                    model = make_model(model_kind)
                    model.fit(X[tr], y_lambda[tr])
                    pred_lambda[te] = model.predict(X[te])

                lambda_metrics, lambda_out = eval_velocity_correction(sub0, pred_lambda)

                lambda_rows.append({
                    "candidate_cv": cand_cv,
                    "candidate_model": cand_model,
                    "candidate_feature_set": cand_fs,
                    "audit_cv": audit_cv,
                    "model": model_kind,
                    "feature_set": fs_name,
                    "features": ",".join(feats),
                    **lambda_metrics,
                })

                keep = lambda_out[[
                    "source_row", "condition", "phase_target", "group_id",
                    "candidate_cv", "candidate_model", "candidate_feature_set",
                    "layer", "v_cur", "v_next_pred", "v_next_true",
                    "lambda_target", "pred_lambda",
                    "v_next_corrected",
                    "base_v_abs_error", "corr_v_abs_error",
                ]].copy()
                keep["audit_cv"] = audit_cv
                keep["model"] = model_kind
                keep["feature_set"] = fs_name
                keep["mode"] = "lambda"
                prediction_rows.append(keep)

                # -------------------------------
                # B. Direct delta-v residual model
                # -------------------------------
                y_delta = sub0["delta_v_error"].values.astype(float)
                pred_delta = np.zeros(len(sub0), dtype=float)

                for tr, te in splits:
                    model = make_model(model_kind)
                    model.fit(X[tr], y_delta[tr])
                    pred_delta[te] = model.predict(X[te])

                direct_metrics, direct_out = eval_direct_delta_v(sub0, pred_delta)

                direct_rows.append({
                    "candidate_cv": cand_cv,
                    "candidate_model": cand_model,
                    "candidate_feature_set": cand_fs,
                    "audit_cv": audit_cv,
                    "model": model_kind,
                    "feature_set": fs_name,
                    "features": ",".join(feats),
                    **direct_metrics,
                })

lambda_summary = pd.DataFrame(lambda_rows).sort_values(
    ["audit_cv", "delta_v_mae", "corrected_v_r2"],
    ascending=[True, True, False],
)

direct_summary = pd.DataFrame(direct_rows).sort_values(
    ["audit_cv", "delta_v_mae", "corrected_v_r2"],
    ascending=[True, True, False],
)

predictions = pd.concat(prediction_rows, ignore_index=True)

lambda_summary.to_csv(SAVE_DIR / "phasemap3d1_lambda_summary.csv", index=False)
direct_summary.to_csv(SAVE_DIR / "phasemap3d1_direct_delta_v_summary.csv", index=False)
predictions.to_csv(SAVE_DIR / "phasemap3d1_lambda_predictions.csv", index=False)
step.to_csv(SAVE_DIR / "phasemap3d1_step_dataset.csv", index=False)
lambda_step.to_csv(SAVE_DIR / "phasemap3d1_lambda_step_dataset.csv", index=False)

# ============================================================
# BEST EXPLICIT LAMBDA EQUATION
# ============================================================

group_lambda = lambda_summary[lambda_summary["audit_cv"] == "GroupKFold"].copy()
best_lambda = group_lambda.iloc[0]

best_sub = lambda_step[
    (lambda_step["candidate_cv"] == best_lambda["candidate_cv"])
    & (lambda_step["candidate_model"] == best_lambda["candidate_model"])
    & (lambda_step["candidate_feature_set"] == best_lambda["candidate_feature_set"])
].copy()

best_feats = best_lambda["features"].split(",")

X = best_sub[best_feats].values.astype(float)
y = best_sub["lambda_target"].values.astype(float)

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

lin = Ridge(alpha=1.0)
lin.fit(Xs, y)

coef_raw = lin.coef_ / scaler.scale_
intercept_raw = lin.intercept_ - np.sum(lin.coef_ * scaler.mean_ / scaler.scale_)

coef_df = pd.DataFrame({
    "feature": best_feats,
    "coef": coef_raw,
})
coef_df["abs_coef"] = coef_df["coef"].abs()
coef_df = coef_df.sort_values("abs_coef", ascending=False)
coef_df.to_csv(SAVE_DIR / "phasemap3d1_lambda_equation_coefficients.csv", index=False)

lambda_equation = f"lambda_l = {intercept_raw:+.6f}"
for f, c in zip(best_feats, coef_raw):
    lambda_equation += f" {c:+.6f}*{f}"

# ============================================================
# PHASE / LAYER SUMMARY FOR BEST LAMBDA MODEL
# ============================================================

best_pred = predictions[
    (predictions["audit_cv"] == best_lambda["audit_cv"])
    & (predictions["model"] == best_lambda["model"])
    & (predictions["feature_set"] == best_lambda["feature_set"])
    & (predictions["candidate_cv"] == best_lambda["candidate_cv"])
    & (predictions["candidate_model"] == best_lambda["candidate_model"])
    & (predictions["candidate_feature_set"] == best_lambda["candidate_feature_set"])
    & (predictions["mode"] == "lambda")
].copy()

phase_summary = best_pred.groupby("phase_target").agg(
    n=("phase_target", "count"),
    lambda_true=("lambda_target", "mean"),
    lambda_pred=("pred_lambda", "mean"),
    base_v_mae=("base_v_abs_error", "mean"),
    corr_v_mae=("corr_v_abs_error", "mean"),
).reset_index()
phase_summary["delta_v_mae"] = phase_summary["corr_v_mae"] - phase_summary["base_v_mae"]

layer_summary = best_pred.groupby("layer").agg(
    n=("layer", "count"),
    lambda_true=("lambda_target", "mean"),
    lambda_pred=("pred_lambda", "mean"),
    base_v_mae=("base_v_abs_error", "mean"),
    corr_v_mae=("corr_v_abs_error", "mean"),
).reset_index()
layer_summary["delta_v_mae"] = layer_summary["corr_v_mae"] - layer_summary["base_v_mae"]

phase_summary.to_csv(SAVE_DIR / "phasemap3d1_best_lambda_by_phase.csv", index=False)
layer_summary.to_csv(SAVE_DIR / "phasemap3d1_best_lambda_by_layer.csv", index=False)

# ============================================================
# SUMMARY JSON
# ============================================================

summary = {
    "experiment": "PhaseMap-3D.1 Energy-Conditioned Damping Coefficient Audit",
    "input_path": str(INPUT_PATH),
    "n_step_rows": int(len(step)),
    "n_lambda_rows": int(len(lambda_step)),
    "core_question": "Does an explicit lambda(E,state) damping coefficient explain velocity correction?",
    "best_groupkfold_lambda": best_lambda.to_dict(),
    "best_groupkfold_direct_delta_v": (
        direct_summary[direct_summary["audit_cv"] == "GroupKFold"].iloc[0].to_dict()
        if len(direct_summary[direct_summary["audit_cv"] == "GroupKFold"]) > 0 else None
    ),
    "lambda_equation": lambda_equation,
    "lambda_top_coefficients": coef_df.head(20).to_dict(orient="records"),
    "top_groupkfold_lambda": group_lambda.head(12).to_dict(orient="records"),
    "interpretation_rules": {
        "PASS_strong": [
            "lambda_r2 > 0.5 under GroupKFold.",
            "lambda correction reduces velocity MAE substantially.",
            "Energy terms dominate lambda equation.",
            "Supports v_next = v_pred - lambda(E,state) * v_cur."
        ],
        "PASS_partial": [
            "Direct delta-v correction works, but lambda model is weak.",
            "Damping exists but is not simply multiplicative in v."
        ],
        "FAIL": [
            "Neither lambda nor direct delta-v correction generalizes.",
            "3D correction is not reducible to velocity damping."
        ],
        "next_if_pass_strong": "PhaseMap-3E: integrate lambda(E) into full recursive rollout.",
        "next_if_pass_partial": "PhaseMap-3D.2: additive attractor pull + damping hybrid.",
        "next_if_fail": "Return to hidden residual state discovery."
    }
}

with open(SAVE_DIR / "phasemap3d1_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nSaved to: {SAVE_DIR.resolve()}")