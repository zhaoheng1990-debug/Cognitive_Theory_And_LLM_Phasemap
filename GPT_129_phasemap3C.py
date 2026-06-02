import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold, GroupKFold

# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path(r"C:\Windows\System32\phasemap3b_outputs\phasemap3b_enriched_energy_error.csv")
SAVE_DIR = Path("./phasemap3c_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
R_LAYERS = list(range(20, 27))
TARGET_LAYERS = [23, 24, 25, 26]

# ============================================================
# LOAD
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Cannot find input file: {INPUT_PATH.resolve()}")

df = pd.read_csv(INPUT_PATH)

if "model" not in df.columns:
    df["model"] = "unknown"

required = [
    "cv", "model", "feature_set", "condition", "phase_target",
] + [f"true_R_{l}" for l in R_LAYERS] + [f"pred_R_{l}" for l in R_LAYERS]

missing = [c for c in required if c not in df.columns]
if missing:
    print(df.columns.tolist())
    raise RuntimeError(f"Missing columns: {missing}")

GROUP_COL = "group_id" if "group_id" in df.columns else "condition"

# ============================================================
# ERROR / RESIDUAL TARGETS
# ============================================================

for l in R_LAYERS:
    df[f"err_R_{l}"] = df[f"pred_R_{l}"] - df[f"true_R_{l}"]
    df[f"abs_err_R_{l}"] = df[f"err_R_{l}"].abs()

df["target_correction_R26"] = df["err_R_26"]
df["target_correction_R25"] = df["err_R_25"]
df["target_correction_tail_mean"] = df[[f"err_R_{l}" for l in TARGET_LAYERS]].mean(axis=1)

df["baseline_tail_mae"] = df[[f"abs_err_R_{l}" for l in TARGET_LAYERS]].mean(axis=1)
df["baseline_final_abs_error_R26"] = df["abs_err_R_26"]

# ============================================================
# FEATURE SETS
# ============================================================

ENERGY_FEATURES = [
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

PRED_STATE_FEATURES = []
for l in [23, 24, 25, 26]:
    PRED_STATE_FEATURES.append(f"pred_R_{l}")

for a, b in [(22, 23), (23, 24), (24, 25), (25, 26)]:
    c = f"pred_v_{a}_{b}"
    if c in df.columns:
        PRED_STATE_FEATURES.append(c)

FEATURE_SETS = {
    "energy_only": ENERGY_FEATURES,
    "pred_state_only": PRED_STATE_FEATURES,
    "energy_plus_pred_state": ENERGY_FEATURES + PRED_STATE_FEATURES,
}

FEATURE_SETS = {k: v for k, v in FEATURE_SETS.items() if len(v) > 0}

# ============================================================
# HELPERS
# ============================================================

def get_splits(data, group_mode=True):
    if group_mode:
        groups = data[GROUP_COL].values
        if len(np.unique(groups)) >= 5:
            return list(GroupKFold(n_splits=5).split(data, groups=groups))

    return list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(data))


def make_model(kind):
    if kind == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ])

    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=8,
            random_state=SEED,
            n_jobs=-1,
        )

    raise ValueError(kind)


def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def eval_corrected(sub, correction_pred, correction_target_name):
    out = sub.copy()

    if correction_target_name == "R26":
        out["corr_R_26"] = out["pred_R_26"] - correction_pred

        for l in [23, 24, 25]:
            out[f"corr_R_{l}"] = out[f"pred_R_{l}"]

    elif correction_target_name == "tail_mean":
        for l in TARGET_LAYERS:
            out[f"corr_R_{l}"] = out[f"pred_R_{l}"] - correction_pred

    else:
        raise ValueError(correction_target_name)

    for l in TARGET_LAYERS:
        out[f"corr_err_R_{l}"] = out[f"corr_R_{l}"] - out[f"true_R_{l}"]
        out[f"corr_abs_err_R_{l}"] = out[f"corr_err_R_{l}"].abs()

    base_y, base_p = [], []
    corr_y, corr_p = [], []

    for _, row in out.iterrows():
        for l in TARGET_LAYERS:
            base_y.append(row[f"true_R_{l}"])
            base_p.append(row[f"pred_R_{l}"])
            corr_y.append(row[f"true_R_{l}"])
            corr_p.append(row[f"corr_R_{l}"])

    base_y = np.array(base_y)
    base_p = np.array(base_p)
    corr_y = np.array(corr_y)
    corr_p = np.array(corr_p)

    return {
        "baseline_rollout_r2": float(r2_score(base_y, base_p)),
        "corrected_rollout_r2": float(r2_score(corr_y, corr_p)),
        "delta_rollout_r2": float(r2_score(corr_y, corr_p) - r2_score(base_y, base_p)),

        "baseline_tail_mae": float(np.mean(np.abs(base_p - base_y))),
        "corrected_tail_mae": float(np.mean(np.abs(corr_p - corr_y))),
        "delta_tail_mae": float(np.mean(np.abs(corr_p - corr_y)) - np.mean(np.abs(base_p - base_y))),

        "baseline_final_abs_error_R26": float(out["abs_err_R_26"].mean()),
        "corrected_final_abs_error_R26": float(out["corr_abs_err_R_26"].mean()),
        "delta_final_abs_error_R26": float(out["corr_abs_err_R_26"].mean() - out["abs_err_R_26"].mean()),

        "baseline_R26_corr": safe_corr(out["true_R_26"], out["pred_R_26"]),
        "corrected_R26_corr": safe_corr(out["true_R_26"], out["corr_R_26"]),
    }, out

# ============================================================
# MAIN CV CORRECTION
# ============================================================

result_rows = []
prediction_rows = []

for candidate, sub in df.groupby(["cv", "model", "feature_set"]):
    cv0, model0, fs0 = candidate
    sub = sub.reset_index(drop=True)

    if len(sub) < 50:
        continue

    for group_mode in [False, True]:
        audit_cv = "GroupKFold" if group_mode else "KFold"
        splits = get_splits(sub, group_mode=group_mode)

        for feat_name, feats in FEATURE_SETS.items():
            X = sub[feats].values.astype(float)

            for target_name, target_col, correction_mode in [
                ("R26", "target_correction_R26", "R26"),
                ("tail_mean", "target_correction_tail_mean", "tail_mean"),
            ]:
                y = sub[target_col].values.astype(float)

                for model_kind in ["ridge", "rf"]:
                    pred_corr = np.zeros(len(sub), dtype=float)

                    for tr, te in splits:
                        m = make_model(model_kind)
                        m.fit(X[tr], y[tr])
                        pred_corr[te] = m.predict(X[te])

                    correction_r2 = float(r2_score(y, pred_corr))
                    correction_corr = safe_corr(y, pred_corr)
                    correction_mae = float(mean_absolute_error(y, pred_corr))

                    metrics, out = eval_corrected(
                        sub,
                        correction_pred=pred_corr,
                        correction_target_name=correction_mode,
                    )

                    result_rows.append({
                        "candidate_cv": cv0,
                        "candidate_model": model0,
                        "candidate_feature_set": fs0,
                        "audit_cv": audit_cv,
                        "correction_model": model_kind,
                        "correction_feature_set": feat_name,
                        "correction_target": target_name,
                        "features": ",".join(feats),
                        "correction_r2": correction_r2,
                        "correction_corr": correction_corr,
                        "correction_mae": correction_mae,
                        **metrics,
                    })

                    keep = out[[
                        "cv", "model", "feature_set",
                        "condition", "phase_target",
                        "source_row" if "source_row" in out.columns else "condition",
                    ]].copy()

                    keep["audit_cv"] = audit_cv
                    keep["correction_model"] = model_kind
                    keep["correction_feature_set"] = feat_name
                    keep["correction_target"] = target_name
                    keep["predicted_correction"] = pred_corr

                    for l in TARGET_LAYERS:
                        keep[f"true_R_{l}"] = out[f"true_R_{l}"]
                        keep[f"pred_R_{l}"] = out[f"pred_R_{l}"]
                        keep[f"corr_R_{l}"] = out[f"corr_R_{l}"]
                        keep[f"base_abs_err_R_{l}"] = out[f"abs_err_R_{l}"]
                        keep[f"corr_abs_err_R_{l}"] = out[f"corr_abs_err_R_{l}"]

                    prediction_rows.append(keep)

results = pd.DataFrame(result_rows).sort_values(
    ["audit_cv", "delta_final_abs_error_R26", "corrected_rollout_r2"],
    ascending=[True, True, False],
)

predictions = pd.concat(prediction_rows, ignore_index=True)

results.to_csv(SAVE_DIR / "phasemap3c_correction_summary.csv", index=False)
predictions.to_csv(SAVE_DIR / "phasemap3c_corrected_predictions.csv", index=False)

# ============================================================
# PHASE / CONDITION ANALYSIS FOR BEST GROUP MODEL
# ============================================================

group_results = results[results["audit_cv"] == "GroupKFold"].copy()
best = group_results.sort_values(
    ["delta_final_abs_error_R26", "corrected_rollout_r2"],
    ascending=[True, False],
).iloc[0]

mask = (
    (predictions["audit_cv"] == best["audit_cv"])
    & (predictions["correction_model"] == best["correction_model"])
    & (predictions["correction_feature_set"] == best["correction_feature_set"])
    & (predictions["correction_target"] == best["correction_target"])
    & (predictions["cv"] == best["candidate_cv"])
    & (predictions["model"] == best["candidate_model"])
    & (predictions["feature_set"] == best["candidate_feature_set"])
)

best_pred = predictions[mask].copy()

phase_summary = best_pred.groupby("phase_target").agg(
    n=("phase_target", "count"),
    base_R26_err=("base_abs_err_R_26", "mean"),
    corr_R26_err=("corr_abs_err_R_26", "mean"),
    base_tail_err=("base_abs_err_R_23", "mean"),
).reset_index()

condition_summary = best_pred.groupby("condition").agg(
    n=("condition", "count"),
    phase=("phase_target", lambda x: x.mode().iloc[0]),
    base_R26_err=("base_abs_err_R_26", "mean"),
    corr_R26_err=("corr_abs_err_R_26", "mean"),
    correction=("predicted_correction", "mean"),
).reset_index()

phase_summary["delta_R26_err"] = phase_summary["corr_R26_err"] - phase_summary["base_R26_err"]
condition_summary["delta_R26_err"] = condition_summary["corr_R26_err"] - condition_summary["base_R26_err"]

phase_summary.to_csv(SAVE_DIR / "phasemap3c_best_by_phase.csv", index=False)
condition_summary.to_csv(SAVE_DIR / "phasemap3c_best_by_condition.csv", index=False)

# ============================================================
# SUMMARY JSON
# ============================================================

summary = {
    "experiment": "PhaseMap-3C Damped Attractor Correction Audit",
    "input_path": str(INPUT_PATH),
    "n_rows": int(len(df)),
    "core_question": "Can residual energy / stability variables correct rollout overshoot and reduce R26 error?",
    "best_groupkfold": best.to_dict(),
    "top_groupkfold": group_results.head(12).to_dict(orient="records"),
    "interpretation_rules": {
        "PASS_strong": [
            "corrected_final_abs_error_R26 decreases by > 1.0",
            "corrected_rollout_r2 improves substantially",
            "correction generalizes under GroupKFold"
        ],
        "PASS_partial": [
            "R26 error decreases but rollout_r2 only modestly improves",
            "energy variables act as residual stabilizer but not full attractor equation"
        ],
        "FAIL": [
            "correction does not reduce R26 error under GroupKFold",
            "3B explained error but cannot causally correct rollout"
        ],
        "next_if_pass": "PhaseMap-3D: explicit damped state equation with layerwise correction",
        "next_if_partial": "PhaseMap-3C.1: layerwise damping correction R23-R26",
        "next_if_fail": "Return to hidden state discovery beyond energy variables"
    }
}

with open(SAVE_DIR / "phasemap3c_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nSaved to: {SAVE_DIR.resolve()}")