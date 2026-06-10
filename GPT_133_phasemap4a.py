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
SAVE_DIR = Path("./phasemap4a_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
R_LAYERS = list(range(20, 27))
INIT_LAYERS = [20, 21, 22]
ROLLOUT_LAYERS = [23, 24, 25, 26]
STEP_LAYERS = [22, 23, 24, 25]

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
] + [f"true_R_{l}" for l in R_LAYERS] + [f"pred_R_{l}" for l in R_LAYERS]

missing = [c for c in required if c not in df.columns]
if missing:
    print(df.columns.tolist())
    raise RuntimeError(f"Missing columns: {missing}")

GROUP_COL = "group_id" if "group_id" in df.columns else "condition"

# ============================================================
# BASE VELOCITY / ENERGY
# ============================================================

for prefix in ["true", "pred"]:
    for a, b in zip(range(20, 26), range(21, 27)):
        df[f"{prefix}_v_{a}_{b}"] = df[f"{prefix}_R_{b}"] - df[f"{prefix}_R_{a}"]

    for a, b, c in zip(range(20, 25), range(21, 26), range(22, 27)):
        df[f"{prefix}_a_{a}_{b}_{c}"] = df[f"{prefix}_v_{b}_{c}"] - df[f"{prefix}_v_{a}_{b}"]

for l in R_LAYERS:
    df[f"base_err_R_{l}"] = df[f"pred_R_{l}"] - df[f"true_R_{l}"]
    df[f"base_abs_err_R_{l}"] = df[f"base_err_R_{l}"].abs()

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
# TRAINING STEP DATASET
#
# We train lambda from existing predicted trajectory:
#
#   delta_v_error = pred_v_next - true_v_next
#   lambda_target = delta_v_error / v_cur
#
# Then rollout uses:
#
#   v_next_base = one-step base velocity estimate from original pred trajectory
#   v_next_corr = v_next_base - lambda * v_cur
#
# For pure explicit rollout, we do not use true future R.
# ============================================================

def make_step_dataset(data):
    rows = []

    for idx, row in data.iterrows():
        for l in [23, 24, 25]:
            v_cur = float(row[f"pred_v_{l-1}_{l}"])
            v_next_pred = float(row[f"pred_v_{l}_{l+1}"])
            v_next_true = float(row[f"true_v_{l}_{l+1}"])

            delta_v_error = v_next_pred - v_next_true

            if abs(v_cur) <= EPS:
                continue

            lambda_target = delta_v_error / v_cur

            if not np.isfinite(lambda_target) or abs(lambda_target) >= 20:
                continue

            item = {
                "source_row": idx,
                "group_id": row[GROUP_COL],
                "condition": row["condition"],
                "phase_target": row["phase_target"],
                "candidate_cv": row["cv"],
                "candidate_model": row["model"],
                "candidate_feature_set": row["feature_set"],

                "layer": l,
                "layer_norm": (l - 24.0) / 0.816496580927726,

                "R_l": float(row[f"pred_R_{l}"]),
                "v_cur": v_cur,
                "a_cur": float(row.get(f"pred_a_{l-2}_{l-1}_{l}", 0.0)),

                "abs_R_l": abs(float(row[f"pred_R_{l}"])),
                "abs_v_cur": abs(v_cur),
                "abs_a_cur": abs(float(row.get(f"pred_a_{l-2}_{l-1}_{l}", 0.0))),
                "R_times_v": float(row[f"pred_R_{l}"]) * v_cur,

                "v_next_pred": v_next_pred,
                "v_next_true": v_next_true,
                "delta_v_error": delta_v_error,
                "lambda_target": lambda_target,
            }

            for c in ENERGY_COLS:
                item[c] = row[c]

            rows.append(item)

    return pd.DataFrame(rows)


step_df = make_step_dataset(df)
step_df.to_csv(SAVE_DIR / "phasemap4a_step_dataset.csv", index=False)

# ============================================================
# FEATURE SETS
# ============================================================

FEATURE_SETS = {
    "state_only": [
        "R_l", "v_cur", "a_cur",
        "abs_R_l", "abs_v_cur", "abs_a_cur",
        "R_times_v", "layer_norm",
    ],

    "energy_only": ENERGY_COLS,

    "state_plus_energy": [
        "R_l", "v_cur", "a_cur",
        "abs_R_l", "abs_v_cur", "abs_a_cur",
        "R_times_v", "layer_norm",
    ] + ENERGY_COLS,

    "minimal_lambda": [
        "v_cur", "abs_v_cur",
        "E_span_init" if "E_span_init" in ENERGY_COLS else ENERGY_COLS[0],
        "E_mixed_L22" if "E_mixed_L22" in ENERGY_COLS else ENERGY_COLS[0],
        "layer_norm",
    ],
}

FEATURE_SETS = {k: list(dict.fromkeys(v)) for k, v in FEATURE_SETS.items()}

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


def compute_energy_from_rollout(pred_R):
    # Local rollout-derived energy proxies.
    # These are deliberately simple because 3D.1 showed v and |v| dominate.
    vals = {}

    for l in [22, 23, 24, 25]:
        if l - 1 in pred_R and l in pred_R:
            v_cur = pred_R[l] - pred_R[l - 1]
        else:
            v_cur = 0.0

        if l - 2 in pred_R and l - 1 in pred_R and l in pred_R:
            v_prev = pred_R[l - 1] - pred_R[l - 2]
            a_cur = v_cur - v_prev
        else:
            a_cur = 0.0

        vals[f"roll_E_v_L{l}"] = v_cur ** 2
        vals[f"roll_E_abs_L{l}"] = abs(v_cur) + abs(a_cur)

    return vals


def make_feature_vector(row, pred_R, layer, feats):
    v_cur = pred_R[layer] - pred_R[layer - 1]
    v_prev = pred_R[layer - 1] - pred_R[layer - 2]
    a_cur = v_cur - v_prev

    vals = {
        "R_l": pred_R[layer],
        "v_cur": v_cur,
        "a_cur": a_cur,
        "abs_R_l": abs(pred_R[layer]),
        "abs_v_cur": abs(v_cur),
        "abs_a_cur": abs(a_cur),
        "R_times_v": pred_R[layer] * v_cur,
        "layer_norm": (layer - 24.0) / 0.816496580927726,
    }

    for c in ENERGY_COLS:
        vals[c] = float(row[c])

    X = np.array([vals[f] for f in feats], dtype=float)

    # 防止递推爆炸传入 sklearn
    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=1e6,
        neginf=-1e6,
    )

    X = np.clip(X, -1e6, 1e6)

    return X


def rollout_one(row, model, feats):
    pred_R = {
        20: float(row["true_R_20"]),
        21: float(row["true_R_21"]),
        22: float(row["true_R_22"]),
    }

    for layer in [22, 23, 24, 25]:
        if layer == 22:
            v_next_base = float(row["pred_v_22_23"])
        elif layer == 23:
            v_next_base = float(row["pred_v_23_24"])
        elif layer == 24:
            v_next_base = float(row["pred_v_24_25"])
        else:
            v_next_base = float(row["pred_v_25_26"])

        X = make_feature_vector(row, pred_R, layer, feats)

        lam = float(model.predict(X.reshape(1, -1))[0])

        # 关键：lambda 必须裁剪，否则递推会发散
        lam = np.nan_to_num(lam, nan=0.0, posinf=3.0, neginf=-3.0)
        lam = float(np.clip(lam, -3.0, 3.0))

        v_cur = pred_R[layer] - pred_R[layer - 1]

        # 速度也裁剪，避免一步爆炸
        v_cur = float(np.clip(v_cur, -50.0, 50.0))
        v_next_base = float(np.clip(v_next_base, -50.0, 50.0))

        v_next_corr = v_next_base - lam * v_cur
        v_next_corr = float(np.clip(v_next_corr, -50.0, 50.0))

        pred_R[layer + 1] = pred_R[layer] + v_next_corr
        pred_R[layer + 1] = float(np.clip(pred_R[layer + 1], -100.0, 100.0))

    return pred_R

    # base predicted velocities from original 3A/3B prediction path
    # used as one-step drift proposal; correction is explicit lambda damping.
    for layer in [22, 23, 24, 25]:
        if layer == 22:
            v_next_base = float(row["pred_v_22_23"])
        elif layer == 23:
            v_next_base = float(row["pred_v_23_24"])
        elif layer == 24:
            v_next_base = float(row["pred_v_24_25"])
        else:
            v_next_base = float(row["pred_v_25_26"])

        X = make_feature_vector(row, pred_R, layer, feats)
        lam = float(model.predict(X.reshape(1, -1))[0])

        v_cur = pred_R[layer] - pred_R[layer - 1]
        v_next_corr = v_next_base - lam * v_cur

        pred_R[layer + 1] = pred_R[layer] + v_next_corr

    return pred_R


def rollout_metrics(out_df):
    y_true, y_base, y_corr = [], [], []

    for _, r in out_df.iterrows():
        for l in ROLLOUT_LAYERS:
            y_true.append(r[f"true_R_{l}"])
            y_base.append(r[f"base_pred_R_{l}"])
            y_corr.append(r[f"roll_R_{l}"])

    y_true = np.array(y_true, dtype=float)
    y_base = np.array(y_base, dtype=float)
    y_corr = np.array(y_corr, dtype=float)

    return {
        "baseline_rollout_r2": float(r2_score(y_true, y_base)),
        "corrected_rollout_r2": float(r2_score(y_true, y_corr)),
        "delta_rollout_r2": float(r2_score(y_true, y_corr) - r2_score(y_true, y_base)),

        "baseline_tail_mae": float(mean_absolute_error(y_true, y_base)),
        "corrected_tail_mae": float(mean_absolute_error(y_true, y_corr)),
        "delta_tail_mae": float(mean_absolute_error(y_true, y_corr) - mean_absolute_error(y_true, y_base)),

        "baseline_R26_mae": float(np.mean(np.abs(out_df["base_pred_R_26"] - out_df["true_R_26"]))),
        "corrected_R26_mae": float(np.mean(np.abs(out_df["roll_R_26"] - out_df["true_R_26"]))),
        "delta_R26_mae": float(
            np.mean(np.abs(out_df["roll_R_26"] - out_df["true_R_26"]))
            - np.mean(np.abs(out_df["base_pred_R_26"] - out_df["true_R_26"]))
        ),

        "baseline_R26_corr": safe_corr(out_df["true_R_26"], out_df["base_pred_R_26"]),
        "corrected_R26_corr": safe_corr(out_df["true_R_26"], out_df["roll_R_26"]),
    }

# ============================================================
# MAIN CV ROLLOUT
# ============================================================

result_rows = []
prediction_rows = []

for cand, sub in df.groupby(["cv", "model", "feature_set"]):
    cand_cv, cand_model, cand_fs = cand
    sub = sub.reset_index(drop=True)

    if len(sub) < 50:
        continue

    candidate_step = step_df[
        (step_df["candidate_cv"] == cand_cv)
        & (step_df["candidate_model"] == cand_model)
        & (step_df["candidate_feature_set"] == cand_fs)
    ].copy()

    if len(candidate_step) < 100:
        continue

    for group_mode in [False, True]:
        audit_cv = "GroupKFold" if group_mode else "KFold"
        splits = get_splits(sub, group_mode=group_mode)

        for fs_name, feats in FEATURE_SETS.items():
            for model_kind in ["ridge", "poly2_ridge"]:
                fold_outs = []

                for fold, (tr_idx, te_idx) in enumerate(splits):
                    train_rows = set(tr_idx.tolist())

                    train_step = candidate_step[
                        candidate_step["source_row"].isin(train_rows)
                    ].copy()

                    X_train = train_step[feats].values.astype(float)
                    y_train = train_step["lambda_target"].values.astype(float)

                    if len(train_step) < 50:
                        continue

                    model = make_model(model_kind)
                    model.fit(X_train, y_train)

                    for idx in te_idx:
                        row = sub.iloc[idx]
                        pred_R = rollout_one(row, model, feats)

                        out = {
                            "candidate_cv": cand_cv,
                            "candidate_model": cand_model,
                            "candidate_feature_set": cand_fs,
                            "audit_cv": audit_cv,
                            "fold": fold,
                            "lambda_model": model_kind,
                            "lambda_feature_set": fs_name,
                            "source_row": int(idx),
                            "condition": row["condition"],
                            "phase_target": row["phase_target"],
                            "group_id": row[GROUP_COL],
                        }

                        for l in R_LAYERS:
                            out[f"true_R_{l}"] = float(row[f"true_R_{l}"])
                            out[f"base_pred_R_{l}"] = float(row[f"pred_R_{l}"])

                            if l in pred_R:
                                out[f"roll_R_{l}"] = float(pred_R[l])
                            else:
                                out[f"roll_R_{l}"] = float(row[f"pred_R_{l}"])

                            out[f"base_abs_err_R_{l}"] = abs(out[f"base_pred_R_{l}"] - out[f"true_R_{l}"])
                            out[f"roll_abs_err_R_{l}"] = abs(out[f"roll_R_{l}"] - out[f"true_R_{l}"])

                        fold_outs.append(out)
                        prediction_rows.append(out)

                if len(fold_outs) == 0:
                    continue

                pred_df = pd.DataFrame(fold_outs)
                metrics = rollout_metrics(pred_df)

                result_rows.append({
                    "candidate_cv": cand_cv,
                    "candidate_model": cand_model,
                    "candidate_feature_set": cand_fs,
                    "audit_cv": audit_cv,
                    "lambda_model": model_kind,
                    "lambda_feature_set": fs_name,
                    "lambda_features": ",".join(feats),
                    **metrics,
                })

results = pd.DataFrame(result_rows).sort_values(
    ["audit_cv", "corrected_R26_mae", "corrected_rollout_r2"],
    ascending=[True, True, False],
)

predictions = pd.DataFrame(prediction_rows)

results.to_csv(SAVE_DIR / "phasemap4a_rollout_summary.csv", index=False)
predictions.to_csv(SAVE_DIR / "phasemap4a_rollout_predictions.csv", index=False)

# ============================================================
# BEST MODEL PHASE / LAYER
# ============================================================

group_results = results[results["audit_cv"] == "GroupKFold"].copy()
best = group_results.iloc[0].to_dict()

best_pred = predictions[
    (predictions["candidate_cv"] == best["candidate_cv"])
    & (predictions["candidate_model"] == best["candidate_model"])
    & (predictions["candidate_feature_set"] == best["candidate_feature_set"])
    & (predictions["audit_cv"] == best["audit_cv"])
    & (predictions["lambda_model"] == best["lambda_model"])
    & (predictions["lambda_feature_set"] == best["lambda_feature_set"])
].copy()

phase_summary = best_pred.groupby("phase_target").agg(
    n=("phase_target", "count"),
    base_R26_mae=("base_abs_err_R_26", "mean"),
    roll_R26_mae=("roll_abs_err_R_26", "mean"),
    base_R23_mae=("base_abs_err_R_23", "mean"),
    roll_R23_mae=("roll_abs_err_R_23", "mean"),
    base_R24_mae=("base_abs_err_R_24", "mean"),
    roll_R24_mae=("roll_abs_err_R_24", "mean"),
    base_R25_mae=("base_abs_err_R_25", "mean"),
    roll_R25_mae=("roll_abs_err_R_25", "mean"),
).reset_index()

phase_summary["delta_R26_mae"] = phase_summary["roll_R26_mae"] - phase_summary["base_R26_mae"]

layer_rows = []
for l in ROLLOUT_LAYERS:
    layer_rows.append({
        "layer": l,
        "base_mae": float(best_pred[f"base_abs_err_R_{l}"].mean()),
        "roll_mae": float(best_pred[f"roll_abs_err_R_{l}"].mean()),
        "delta_mae": float(best_pred[f"roll_abs_err_R_{l}"].mean() - best_pred[f"base_abs_err_R_{l}"].mean()),
    })

layer_summary = pd.DataFrame(layer_rows)

phase_summary.to_csv(SAVE_DIR / "phasemap4a_best_by_phase.csv", index=False)
layer_summary.to_csv(SAVE_DIR / "phasemap4a_best_by_layer.csv", index=False)

# ============================================================
# SUMMARY
# ============================================================

summary = {
    "experiment": "PhaseMap-4A Explicit Damped Rollout Audit",
    "input_path": str(INPUT_PATH),
    "n_rows": int(len(df)),
    "n_step_rows": int(len(step_df)),
    "core_question": "Can explicit v_next = v_pred - lambda(R,v,E)*v recursively roll out R23-R26?",
    "best_groupkfold": best,
    "top_groupkfold": group_results.head(12).to_dict(orient="records"),
    "interpretation_rules": {
        "PASS_strong": [
            "Corrected R26 MAE approaches 3C level around 1.0.",
            "Corrected rollout R2 improves substantially over baseline.",
            "Recursive explicit lambda rollout remains stable through R26."
        ],
        "PASS_partial": [
            "R26 MAE improves but remains above 3C.",
            "Explicit damping captures part of correction but rollout still accumulates error."
        ],
        "FAIL": [
            "Recursive rollout diverges or worsens baseline.",
            "Lambda works only for one-step correction, not generative rollout."
        ],
        "next_if_pass": "Freeze PhaseMap state equation draft.",
        "next_if_partial": "Add rollout-stabilized lambda clipping / phase-conditioned lambda.",
        "next_if_fail": "Return to 3C correction model; explicit recursive equation insufficient."
    }
}

with open(SAVE_DIR / "phasemap4a_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nSaved to: {SAVE_DIR.resolve()}")