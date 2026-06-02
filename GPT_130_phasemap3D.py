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
SAVE_DIR = Path("./phasemap3d_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
LAYERS = [23, 24, 25, 26]

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
# BUILD ERROR / DAMPING FEATURES
# ============================================================

for l in range(20, 27):
    df[f"err_R_{l}"] = df[f"pred_R_{l}"] - df[f"true_R_{l}"]
    df[f"abs_err_R_{l}"] = df[f"err_R_{l}"].abs()

for a, b in zip(range(20, 26), range(21, 27)):
    df[f"pred_v_{a}_{b}"] = df[f"pred_R_{b}"] - df[f"pred_R_{a}"]
    df[f"true_v_{a}_{b}"] = df[f"true_R_{b}"] - df[f"true_R_{a}"]

for a, b, c in zip(range(20, 25), range(21, 26), range(22, 27)):
    df[f"pred_a_{a}_{b}_{c}"] = df[f"pred_v_{b}_{c}"] - df[f"pred_v_{a}_{b}"]
    df[f"true_a_{a}_{b}_{c}"] = df[f"true_v_{b}_{c}"] - df[f"true_v_{a}_{b}"]

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
    raise RuntimeError("No energy columns found. Run 3B first.")

# ============================================================
# STEPWISE DATASET
# target:
#   correction_l = pred_R_l - true_R_l
#
# Explicit damping equation candidates:
#
#   correction_l ≈ b0 + λ v_pred + κ R_pred + η S + ...
#
# If λ/κ/S terms reduce error under GroupKFold,
# then 3C correction can be interpreted as damping/attractor pull.
# ============================================================

rows = []

for idx, row in df.iterrows():
    for l in LAYERS:
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

            "R_pred": row[f"pred_R_{l}"],
            "R_true": row[f"true_R_{l}"],
            "err": row[f"err_R_{l}"],
            "abs_err": row[f"abs_err_R_{l}"],
        }

        if l > 20:
            item["v_pred"] = row.get(f"pred_v_{l-1}_{l}", 0.0)
            item["v_true"] = row.get(f"true_v_{l-1}_{l}", 0.0)
        else:
            item["v_pred"] = 0.0
            item["v_true"] = 0.0

        if l > 21:
            item["a_pred"] = row.get(f"pred_a_{l-2}_{l-1}_{l}", 0.0)
            item["a_true"] = row.get(f"true_a_{l-2}_{l-1}_{l}", 0.0)
        else:
            item["a_pred"] = 0.0
            item["a_true"] = 0.0

        item["abs_R_pred"] = abs(item["R_pred"])
        item["abs_v_pred"] = abs(item["v_pred"])
        item["abs_a_pred"] = abs(item["a_pred"])

        # Signed damping candidates
        item["R_times_v"] = item["R_pred"] * item["v_pred"]
        item["sign_R"] = np.sign(item["R_pred"])
        item["sign_v"] = np.sign(item["v_pred"])
        item["pull_to_zero"] = item["R_pred"]
        item["damping_v"] = item["v_pred"]

        for c in ENERGY_COLS:
            item[c] = row[c]

        rows.append(item)

step = pd.DataFrame(rows)

# ============================================================
# FEATURE SETS
# ============================================================

FEATURE_SETS = {
    "state_only": [
        "R_pred", "v_pred", "a_pred", "layer_norm",
    ],
    "damping_linear": [
        "R_pred", "v_pred", "a_pred",
        "pull_to_zero", "damping_v",
        "layer_norm",
    ],
    "energy_only": ENERGY_COLS,
    "state_plus_energy": [
        "R_pred", "v_pred", "a_pred",
        "abs_R_pred", "abs_v_pred", "abs_a_pred",
        "layer_norm",
    ] + ENERGY_COLS,
    "explicit_damped": [
        "R_pred", "v_pred", "a_pred",
        "R_times_v", "pull_to_zero", "damping_v",
        "abs_R_pred", "abs_v_pred", "layer_norm",
    ] + ENERGY_COLS,
}

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


def evaluate_correction(data, pred_corr):
    out = data.copy()
    out["pred_corr"] = pred_corr
    out["R_corrected"] = out["R_pred"] - out["pred_corr"]
    out["err_corrected"] = out["R_corrected"] - out["R_true"]
    out["abs_err_corrected"] = out["err_corrected"].abs()

    return {
        "correction_r2": float(r2_score(out["err"], out["pred_corr"])),
        "correction_corr": safe_corr(out["err"], out["pred_corr"]),
        "correction_mae": float(mean_absolute_error(out["err"], out["pred_corr"])),

        "baseline_mae": float(out["abs_err"].mean()),
        "corrected_mae": float(out["abs_err_corrected"].mean()),
        "delta_mae": float(out["abs_err_corrected"].mean() - out["abs_err"].mean()),

        "baseline_r2": float(r2_score(out["R_true"], out["R_pred"])),
        "corrected_r2": float(r2_score(out["R_true"], out["R_corrected"])),
        "delta_r2": float(r2_score(out["R_true"], out["R_corrected"]) - r2_score(out["R_true"], out["R_pred"])),
    }, out

# ============================================================
# CV EVAL
# ============================================================

result_rows = []
prediction_rows = []

for cand, sub in step.groupby(["candidate_cv", "candidate_model", "candidate_feature_set"]):
    cand_cv, cand_model, cand_fs = cand
    sub = sub.reset_index(drop=True)

    if len(sub) < 100:
        continue

    for audit_group in [False, True]:
        audit_cv = "GroupKFold" if audit_group else "KFold"
        splits = get_splits(sub, group_mode=audit_group)

        for fs_name, feats in FEATURE_SETS.items():
            X = sub[feats].values.astype(float)
            y = sub["err"].values.astype(float)

            for model_kind in ["ridge", "poly2_ridge"]:
                pred = np.zeros(len(sub), dtype=float)

                for tr, te in splits:
                    model = make_model(model_kind)
                    model.fit(X[tr], y[tr])
                    pred[te] = model.predict(X[te])

                metrics, out = evaluate_correction(sub, pred)

                result_rows.append({
                    "candidate_cv": cand_cv,
                    "candidate_model": cand_model,
                    "candidate_feature_set": cand_fs,
                    "audit_cv": audit_cv,
                    "equation_model": model_kind,
                    "feature_set": fs_name,
                    "features": ",".join(feats),
                    **metrics,
                })

                keep = out[[
                    "source_row", "condition", "phase_target", "group_id",
                    "candidate_cv", "candidate_model", "candidate_feature_set",
                    "layer", "R_true", "R_pred", "R_corrected",
                    "err", "pred_corr", "err_corrected",
                    "abs_err", "abs_err_corrected",
                ]].copy()

                keep["audit_cv"] = audit_cv
                keep["equation_model"] = model_kind
                keep["feature_set"] = fs_name

                prediction_rows.append(keep)

results = pd.DataFrame(result_rows).sort_values(
    ["audit_cv", "delta_mae", "corrected_r2"],
    ascending=[True, True, False],
)

predictions = pd.concat(prediction_rows, ignore_index=True)

results.to_csv(SAVE_DIR / "phasemap3d_equation_summary.csv", index=False)
predictions.to_csv(SAVE_DIR / "phasemap3d_layerwise_predictions.csv", index=False)
step.to_csv(SAVE_DIR / "phasemap3d_step_dataset.csv", index=False)

# ============================================================
# FIT EXPLICIT EQUATION ON BEST GROUP MODEL
# ============================================================

group_results = results[results["audit_cv"] == "GroupKFold"].copy()
best = group_results.sort_values(["delta_mae", "corrected_r2"], ascending=[True, False]).iloc[0]

best_sub = step[
    (step["candidate_cv"] == best["candidate_cv"])
    & (step["candidate_model"] == best["candidate_model"])
    & (step["candidate_feature_set"] == best["candidate_feature_set"])
].copy()

best_feats = best["features"].split(",")

# Raw linear equation for interpretability
X = best_sub[best_feats].values.astype(float)
y = best_sub["err"].values.astype(float)

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

lin = Ridge(alpha=1.0)
lin.fit(Xs, y)

coef_raw = lin.coef_ / scaler.scale_
intercept_raw = lin.intercept_ - np.sum(lin.coef_ * scaler.mean_ / scaler.scale_)

equation_rows = []
for f, c in zip(best_feats, coef_raw):
    equation_rows.append({
        "feature": f,
        "coef": float(c),
        "abs_coef": float(abs(c)),
    })

equation_df = pd.DataFrame(equation_rows).sort_values("abs_coef", ascending=False)
equation_df.to_csv(SAVE_DIR / "phasemap3d_explicit_equation_coefficients.csv", index=False)

equation = f"err_l = {intercept_raw:+.6f}"
for f, c in zip(best_feats, coef_raw):
    equation += f" {c:+.6f}*{f}"

# ============================================================
# PHASE / LAYER SUMMARY FOR BEST MODEL
# ============================================================

best_pred = predictions[
    (predictions["audit_cv"] == best["audit_cv"])
    & (predictions["equation_model"] == best["equation_model"])
    & (predictions["feature_set"] == best["feature_set"])
    & (predictions["candidate_cv"] == best["candidate_cv"])
    & (predictions["candidate_model"] == best["candidate_model"])
    & (predictions["candidate_feature_set"] == best["candidate_feature_set"])
].copy()

phase_summary = best_pred.groupby("phase_target").agg(
    n=("phase_target", "count"),
    base_mae=("abs_err", "mean"),
    corr_mae=("abs_err_corrected", "mean"),
    mean_pred_corr=("pred_corr", "mean"),
).reset_index()
phase_summary["delta_mae"] = phase_summary["corr_mae"] - phase_summary["base_mae"]

layer_summary = best_pred.groupby("layer").agg(
    n=("layer", "count"),
    base_mae=("abs_err", "mean"),
    corr_mae=("abs_err_corrected", "mean"),
    mean_pred_corr=("pred_corr", "mean"),
).reset_index()
layer_summary["delta_mae"] = layer_summary["corr_mae"] - layer_summary["base_mae"]

phase_summary.to_csv(SAVE_DIR / "phasemap3d_best_by_phase.csv", index=False)
layer_summary.to_csv(SAVE_DIR / "phasemap3d_best_by_layer.csv", index=False)

# ============================================================
# SUMMARY
# ============================================================

summary = {
    "experiment": "PhaseMap-3D Explicit Damped State Equation Audit",
    "input_path": str(INPUT_PATH),
    "n_step_rows": int(len(step)),
    "core_question": "Can 3C black-box correction be expressed as layerwise damped attractor equation?",
    "best_groupkfold": best.to_dict(),
    "explicit_equation": equation,
    "top_coefficients": equation_df.head(20).to_dict(orient="records"),
    "top_groupkfold": group_results.head(15).to_dict(orient="records"),
    "interpretation_rules": {
        "PASS_strong": [
            "GroupKFold corrected MAE improves substantially.",
            "Explicit ridge/poly equation approaches 3C correction quality.",
            "Coefficients show meaningful damping terms: v_pred, R_pred, energy, decay_gap."
        ],
        "PASS_partial": [
            "Layerwise correction improves but remains well below 3C black-box correction.",
            "Damping exists but is nonlinear or requires hidden variables."
        ],
        "FAIL": [
            "Explicit equation fails under GroupKFold.",
            "3C correction is predictive but not reducible to simple damping state equation."
        ],
        "next_if_pass": "Write PhaseMap dynamical equation section.",
        "next_if_partial": "PhaseMap-3D.1 nonlinear damping law / piecewise phase-conditioned equation.",
        "next_if_fail": "Return to residual hidden state discovery."
    }
}

with open(SAVE_DIR / "phasemap3d_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nSaved to: {SAVE_DIR.resolve()}")