import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold, GroupKFold

# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path(r"C:\Windows\System32\phasemap3b_outputs\phasemap3b_enriched_energy_error.csv")
SAVE_DIR = Path("./phasemap3e_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
LAYERS = [23, 24, 25]
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
# DERIVED VARIABLES
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
            "sign_R": np.sign(float(row[f"pred_R_{l}"])),
            "sign_v": np.sign(v_cur),
        }

        # attractor-pull candidates
        item["pull_zero"] = item["R_pred_l"]
        item["pull_signed"] = item["sign_R"] * item["abs_R_pred"]
        item["velocity_signed_pull"] = item["sign_v"] * item["abs_R_pred"]

        for c in ENERGY_COLS:
            item[c] = row[c]

        rows.append(item)

step = pd.DataFrame(rows)

lambda_step = step[np.isfinite(step["lambda_target"])].copy()
lambda_step = lambda_step[lambda_step["lambda_target"].abs() < 20].copy()

# ============================================================
# FEATURE SETS
# ============================================================

STATE_COLS = [
    "R_pred_l", "v_cur", "a_pred",
    "abs_R_pred", "abs_v_cur", "abs_a_pred",
    "R_times_v", "layer_norm",
]

LAMBDA_FEATURES = STATE_COLS + ENERGY_COLS

ATTRACTOR_FEATURES = [
    "R_pred_l", "abs_R_pred", "pull_zero", "pull_signed",
    "velocity_signed_pull", "layer_norm",
] + ENERGY_COLS

FULL_FEATURES = list(dict.fromkeys(STATE_COLS + ATTRACTOR_FEATURES + ENERGY_COLS))

FEATURE_SETS = {
    "lambda_only": LAMBDA_FEATURES,
    "attractor_only": ATTRACTOR_FEATURES,
    "lambda_plus_attractor": FULL_FEATURES,
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


def velocity_metrics(out):
    out = out.copy()
    out["base_v_abs_error"] = (out["v_next_pred"] - out["v_next_true"]).abs()
    out["corr_v_abs_error"] = (out["v_next_corrected"] - out["v_next_true"]).abs()

    return {
        "baseline_v_mae": float(out["base_v_abs_error"].mean()),
        "corrected_v_mae": float(out["corr_v_abs_error"].mean()),
        "delta_v_mae": float(out["corr_v_abs_error"].mean() - out["base_v_abs_error"].mean()),

        "baseline_v_r2": float(r2_score(out["v_next_true"], out["v_next_pred"])),
        "corrected_v_r2": float(r2_score(out["v_next_true"], out["v_next_corrected"])),
        "delta_v_r2": float(
            r2_score(out["v_next_true"], out["v_next_corrected"])
            - r2_score(out["v_next_true"], out["v_next_pred"])
        ),
        "corr_true_corrected_v": safe_corr(out["v_next_true"], out["v_next_corrected"]),
    }


# ============================================================
# MAIN AUDIT
#
# Three models:
#
# A. lambda_only:
#    learn lambda, correction = lambda * v_cur
#
# B. attractor_only:
#    learn A directly, correction = A
#
# C. lambda_plus_attractor:
#    stage 1 learn lambda
#    residual r = delta_v_error - lambda*v_cur
#    stage 2 learn A(R,E)
#    correction = lambda*v_cur + A
# ============================================================

summary_rows = []
prediction_rows = []

for cand, sub0 in lambda_step.groupby(["candidate_cv", "candidate_model", "candidate_feature_set"]):
    cand_cv, cand_model, cand_fs = cand
    sub0 = sub0.reset_index(drop=True)

    if len(sub0) < 100:
        continue

    for group_mode in [False, True]:
        audit_cv = "GroupKFold" if group_mode else "KFold"
        splits = get_splits(sub0, group_mode=group_mode)

        for model_kind in ["ridge", "poly2_ridge"]:

            # -------------------------------
            # A. lambda-only
            # -------------------------------
            feats = FEATURE_SETS["lambda_only"]
            X = sub0[feats].values.astype(float)
            y_lambda = sub0["lambda_target"].values.astype(float)

            pred_lambda = np.zeros(len(sub0), dtype=float)

            for tr, te in splits:
                m = make_model(model_kind)
                m.fit(X[tr], y_lambda[tr])
                pred_lambda[te] = m.predict(X[te])

            out = sub0.copy()
            out["pred_lambda"] = pred_lambda
            out["pred_attractor"] = 0.0
            out["pred_total_correction"] = out["pred_lambda"] * out["v_cur"]
            out["v_next_corrected"] = out["v_next_pred"] - out["pred_total_correction"]

            metrics = velocity_metrics(out)

            summary_rows.append({
                "candidate_cv": cand_cv,
                "candidate_model": cand_model,
                "candidate_feature_set": cand_fs,
                "audit_cv": audit_cv,
                "equation_model": model_kind,
                "equation_type": "lambda_only",
                "features_lambda": ",".join(feats),
                "features_attractor": "",
                "lambda_r2": float(r2_score(y_lambda, pred_lambda)),
                "lambda_corr": safe_corr(y_lambda, pred_lambda),
                "attractor_r2": None,
                "attractor_corr": None,
                "total_correction_r2": float(r2_score(sub0["delta_v_error"], out["pred_total_correction"])),
                "total_correction_corr": safe_corr(sub0["delta_v_error"], out["pred_total_correction"]),
                **metrics,
            })

            # -------------------------------
            # B. attractor-only direct pull
            # -------------------------------
            feats_A = FEATURE_SETS["attractor_only"]
            X_A = sub0[feats_A].values.astype(float)
            y_delta = sub0["delta_v_error"].values.astype(float)

            pred_A = np.zeros(len(sub0), dtype=float)

            for tr, te in splits:
                m = make_model(model_kind)
                m.fit(X_A[tr], y_delta[tr])
                pred_A[te] = m.predict(X_A[te])

            out = sub0.copy()
            out["pred_lambda"] = 0.0
            out["pred_attractor"] = pred_A
            out["pred_total_correction"] = pred_A
            out["v_next_corrected"] = out["v_next_pred"] - out["pred_total_correction"]

            metrics = velocity_metrics(out)

            summary_rows.append({
                "candidate_cv": cand_cv,
                "candidate_model": cand_model,
                "candidate_feature_set": cand_fs,
                "audit_cv": audit_cv,
                "equation_model": model_kind,
                "equation_type": "attractor_only",
                "features_lambda": "",
                "features_attractor": ",".join(feats_A),
                "lambda_r2": None,
                "lambda_corr": None,
                "attractor_r2": float(r2_score(y_delta, pred_A)),
                "attractor_corr": safe_corr(y_delta, pred_A),
                "total_correction_r2": float(r2_score(y_delta, pred_A)),
                "total_correction_corr": safe_corr(y_delta, pred_A),
                **metrics,
            })

            # -------------------------------
            # C. lambda + attractor residual
            # -------------------------------
            feats_L = FEATURE_SETS["lambda_only"]
            feats_A = FEATURE_SETS["attractor_only"]

            X_L = sub0[feats_L].values.astype(float)
            X_A = sub0[feats_A].values.astype(float)

            pred_lambda = np.zeros(len(sub0), dtype=float)
            pred_A = np.zeros(len(sub0), dtype=float)

            for tr, te in splits:
                # stage 1: lambda
                mL = make_model(model_kind)
                mL.fit(X_L[tr], y_lambda[tr])
                pred_lambda[te] = mL.predict(X_L[te])

                train_lambda = mL.predict(X_L[tr])
                train_lambda_correction = train_lambda * sub0.iloc[tr]["v_cur"].values

                # stage 2: residual attractor
                residual_A_train = (
                    sub0.iloc[tr]["delta_v_error"].values
                    - train_lambda_correction
                )

                mA = make_model(model_kind)
                mA.fit(X_A[tr], residual_A_train)
                pred_A[te] = mA.predict(X_A[te])

            out = sub0.copy()
            out["pred_lambda"] = pred_lambda
            out["pred_attractor"] = pred_A
            out["pred_total_correction"] = out["pred_lambda"] * out["v_cur"] + out["pred_attractor"]
            out["v_next_corrected"] = out["v_next_pred"] - out["pred_total_correction"]

            metrics = velocity_metrics(out)

            residual_target = sub0["delta_v_error"].values - pred_lambda * sub0["v_cur"].values

            summary_rows.append({
                "candidate_cv": cand_cv,
                "candidate_model": cand_model,
                "candidate_feature_set": cand_fs,
                "audit_cv": audit_cv,
                "equation_model": model_kind,
                "equation_type": "lambda_plus_attractor",
                "features_lambda": ",".join(feats_L),
                "features_attractor": ",".join(feats_A),
                "lambda_r2": float(r2_score(y_lambda, pred_lambda)),
                "lambda_corr": safe_corr(y_lambda, pred_lambda),
                "attractor_r2": float(r2_score(residual_target, pred_A)),
                "attractor_corr": safe_corr(residual_target, pred_A),
                "total_correction_r2": float(r2_score(sub0["delta_v_error"], out["pred_total_correction"])),
                "total_correction_corr": safe_corr(sub0["delta_v_error"], out["pred_total_correction"]),
                **metrics,
            })

            keep = out[[
                "source_row", "condition", "phase_target", "group_id",
                "candidate_cv", "candidate_model", "candidate_feature_set",
                "layer", "v_cur", "v_next_pred", "v_next_true",
                "delta_v_error", "pred_lambda", "pred_attractor",
                "pred_total_correction", "v_next_corrected",
            ]].copy()
            keep["audit_cv"] = audit_cv
            keep["equation_model"] = model_kind
            keep["equation_type"] = "lambda_plus_attractor"
            prediction_rows.append(keep)

summary_df = pd.DataFrame(summary_rows).sort_values(
    ["audit_cv", "delta_v_mae", "corrected_v_r2"],
    ascending=[True, True, False],
)

pred_df = pd.concat(prediction_rows, ignore_index=True)

summary_df.to_csv(SAVE_DIR / "phasemap3e_equation_summary.csv", index=False)
pred_df.to_csv(SAVE_DIR / "phasemap3e_predictions.csv", index=False)
step.to_csv(SAVE_DIR / "phasemap3e_step_dataset.csv", index=False)

# ============================================================
# BEST MODEL INTERPRETATION
# ============================================================

group_summary = summary_df[summary_df["audit_cv"] == "GroupKFold"].copy()
best = group_summary.iloc[0].to_dict()

best_pred = pred_df[
    (pred_df["candidate_cv"] == best["candidate_cv"])
    & (pred_df["candidate_model"] == best["candidate_model"])
    & (pred_df["candidate_feature_set"] == best["candidate_feature_set"])
    & (pred_df["audit_cv"] == best["audit_cv"])
    & (pred_df["equation_model"] == best["equation_model"])
    & (pred_df["equation_type"] == "lambda_plus_attractor")
].copy()

phase_summary = best_pred.groupby("phase_target").agg(
    n=("phase_target", "count"),
    delta_v_error=("delta_v_error", "mean"),
    pred_lambda=("pred_lambda", "mean"),
    pred_attractor=("pred_attractor", "mean"),
    pred_total_correction=("pred_total_correction", "mean"),
).reset_index()

layer_summary = best_pred.groupby("layer").agg(
    n=("layer", "count"),
    delta_v_error=("delta_v_error", "mean"),
    pred_lambda=("pred_lambda", "mean"),
    pred_attractor=("pred_attractor", "mean"),
    pred_total_correction=("pred_total_correction", "mean"),
).reset_index()

phase_summary.to_csv(SAVE_DIR / "phasemap3e_best_by_phase.csv", index=False)
layer_summary.to_csv(SAVE_DIR / "phasemap3e_best_by_layer.csv", index=False)

# ============================================================
# SUMMARY JSON
# ============================================================

summary = {
    "experiment": "PhaseMap-3E Damping + Attractor Pull State Equation Audit",
    "input_path": str(INPUT_PATH),
    "n_step_rows": int(len(step)),
    "n_lambda_rows": int(len(lambda_step)),
    "core_question": "Does v_next require both multiplicative damping lambda*v and additive attractor pull A(R,E)?",
    "best_groupkfold": best,
    "top_groupkfold": group_summary.head(15).to_dict(orient="records"),
    "interpretation_rules": {
        "PASS_strong": [
            "lambda_plus_attractor beats lambda_only under GroupKFold.",
            "total_correction_r2 approaches direct delta-v model.",
            "attractor_r2 is positive and nontrivial.",
            "Supports v_next = v_pred - lambda(E,state)v + A(R,E)."
        ],
        "PASS_lambda_only": [
            "lambda_only is near best; attractor term adds little.",
            "Pure multiplicative damping is sufficient."
        ],
        "PASS_attractor_only": [
            "attractor_only is best; multiplicative damping is not essential.",
            "Correction is more like additive basin pull."
        ],
        "FAIL": [
            "Hybrid does not improve generalization.",
            "3E equation does not explain residual correction."
        ],
        "next_if_pass_strong": "PhaseMap-3F: recursive rollout using explicit damping+attractor equation.",
        "next_if_attractor_only": "PhaseMap-3F: additive attractor field equation.",
        "next_if_fail": "Return to residual hidden state."
    }
}

with open(SAVE_DIR / "phasemap3e_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nSaved to: {SAVE_DIR.resolve()}")