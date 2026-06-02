import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold

# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path(r"C:\Windows\System32\phasemap2c_outputs\phasemap2c_eval.csv")
SAVE_DIR = Path("./phasemap3a2_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
R_LAYERS = list(range(20, 27))

# rollout starts from R20,R21,R22 and predicts R23-R26
INIT_LAYERS = [20, 21, 22]
ROLLOUT_TARGET_LAYERS = [23, 24, 25, 26]

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
# STEP DATASET
#
# We learn:
#
#   state_l = (R_l, v_l, a_l, optional clock)
#
# where:
#   v_l = R_{l+1} - R_l
#   a_l = v_l - v_prev
#   v_prev = R_l - R_{l-1}
#
# Target:
#   v_next = R_{l+2} - R_{l+1}
#
# Then rollout:
#   R_{l+2} = R_{l+1} + v_next
#
# For rollout from R20,R21,R22:
#   initial at l=21:
#       R_l = R21
#       v_prev = R21-R20
#       v_l = R22-R21
#       a_l = v_l-v_prev
#   predict v22 = R23-R22
#   then advance.
# ============================================================

def build_state_features(R_prev, R_l, R_next, layer, feature_set):
    v_prev = R_l - R_prev
    v_l = R_next - R_l
    a_l = v_l - v_prev

    local_vals = np.array([R_prev, R_l, R_next], dtype=float)

    local_mean = float(local_vals.mean())
    local_span = float(local_vals.max() - local_vals.min())
    local_energy = float(R_l ** 2 + v_l ** 2)
    local_abs_energy = float(abs(R_l) + abs(v_l))
    curvature_energy = float(a_l ** 2)

    phase_clock = 0 if layer <= 22 else 1

    layer_norm = (layer - 22.5) / 1.11803398875  # approx std for layers 21..24

    vals = {
        "R_l": R_l,
        "v_l": v_l,
        "a_l": a_l,
        "local_mean": local_mean,
        "local_span": local_span,
        "local_energy": local_energy,
        "local_abs_energy": local_abs_energy,
        "curvature_energy": curvature_energy,
        "layer_norm": layer_norm,
        "phase_clock": phase_clock,
    }

    return np.array([vals[f] for f in feature_set], dtype=float), vals


rows = []

for idx, row in df.iterrows():
    for l in range(21, 25):
        R_prev = float(row[f"R_{l-1}"])
        R_l = float(row[f"R_{l}"])
        R_next = float(row[f"R_{l+1}"])
        R_next2 = float(row[f"R_{l+2}"])

        v_prev = R_l - R_prev
        v_l = R_next - R_l
        a_l = v_l - v_prev
        v_next = R_next2 - R_next

        local_vals = np.array([R_prev, R_l, R_next], dtype=float)

        rows.append({
            "source_row": int(idx),
            "group_id": row[GROUP_COL],
            "condition": row["condition"],
            "phase_target": row["phase_target"],
            "layer": int(l),
            "phase_clock": 0 if l <= 22 else 1,

            "R_l": R_l,
            "v_l": v_l,
            "a_l": a_l,

            "local_mean": float(local_vals.mean()),
            "local_span": float(local_vals.max() - local_vals.min()),
            "local_energy": float(R_l ** 2 + v_l ** 2),
            "local_abs_energy": float(abs(R_l) + abs(v_l)),
            "curvature_energy": float(a_l ** 2),

            "layer_norm": (l - 22.5) / 1.11803398875,

            "v_next": v_next,
        })

step_df = pd.DataFrame(rows)

# ============================================================
# FEATURE SETS
# ============================================================

FEATURE_SETS = {
    "RVA": ["R_l", "v_l", "a_l"],

    "RVA_layer": ["R_l", "v_l", "a_l", "layer_norm"],

    "RVA_phaseclock": ["R_l", "v_l", "a_l", "phase_clock"],

    "H_shape_local": [
        "R_l", "v_l", "a_l",
        "local_mean", "local_span",
        "local_energy", "local_abs_energy",
        "curvature_energy",
    ],

    "H_shape_layer": [
        "R_l", "v_l", "a_l",
        "local_mean", "local_span",
        "local_energy", "local_abs_energy",
        "curvature_energy",
        "layer_norm",
    ],
}

# ============================================================
# MODEL / SPLITS
# ============================================================

def make_model():
    return RandomForestRegressor(
        n_estimators=600,
        max_depth=6,
        min_samples_leaf=5,
        random_state=SEED,
    )


def get_splits(data, group_mode=True):
    if group_mode:
        if "group_id" in data.columns:
            groups = data["group_id"].values
        elif GROUP_COL in data.columns:
            groups = data[GROUP_COL].values
        else:
            groups = data.index.values

        if len(np.unique(groups)) >= 5:
            return list(GroupKFold(n_splits=5).split(data, groups=groups))

    return list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(data))

# ============================================================
# ONE-STEP V_NEXT EVAL
# ============================================================

one_step_rows = []

for cv_name, group_mode in [("KFold", False), ("GroupKFold", True)]:
    splits = get_splits(step_df, group_mode=group_mode)

    for fs_name, feats in FEATURE_SETS.items():
        y = step_df["v_next"].values.astype(float)
        X = step_df[feats].values.astype(float)

        pred = np.zeros(len(step_df), dtype=float)

        for tr, te in splits:
            model = make_model()
            model.fit(X[tr], y[tr])
            pred[te] = model.predict(X[te])

        corr = float(np.corrcoef(y, pred)[0, 1]) if np.std(pred) > 0 else 0.0

        one_step_rows.append({
            "cv": cv_name,
            "feature_set": fs_name,
            "features": ",".join(feats),
            "target": "v_next",
            "r2": float(r2_score(y, pred)),
            "corr": corr,
            "mae": float(mean_absolute_error(y, pred)),
        })

one_step_df = pd.DataFrame(one_step_rows).sort_values(
    ["cv", "r2"],
    ascending=[True, False],
)

one_step_df.to_csv(SAVE_DIR / "phasemap3a2_one_step_v_eval.csv", index=False)

# ============================================================
# MULTI-STEP ROLLOUT
# ============================================================

def rollout_one_row(row, model, feature_set):
    # Known initial values
    R20 = float(row["R_20"])
    R21 = float(row["R_21"])
    R22 = float(row["R_22"])

    pred_R = {
        20: R20,
        21: R21,
        22: R22,
    }

    # Start at l=21:
    # state uses R20,R21,R22 and predicts v_next = R23-R22
    for l in range(21, 25):
        R_prev = pred_R[l - 1]
        R_l = pred_R[l]
        R_next = pred_R[l + 1]

        X, _ = build_state_features(
            R_prev=R_prev,
            R_l=R_l,
            R_next=R_next,
            layer=l,
            feature_set=feature_set,
        )

        v_next_pred = float(model.predict(X.reshape(1, -1))[0])

        pred_R[l + 2] = R_next + v_next_pred

    return pred_R


rollout_rows = []
summary_rows = []

for cv_name, group_mode in [("KFold", False), ("GroupKFold", True)]:
    splits = get_splits(df, group_mode=group_mode)

    for fs_name, feats in FEATURE_SETS.items():
        fold_preds = []

        for fold, (tr_idx, te_idx) in enumerate(splits):
            # train step model only on train original rows
            train_source_rows = set(tr_idx.tolist())
            train_step = step_df[step_df["source_row"].isin(train_source_rows)].copy()

            X_train = train_step[feats].values.astype(float)
            y_train = train_step["v_next"].values.astype(float)

            model = make_model()
            model.fit(X_train, y_train)

            for idx in te_idx:
                row = df.iloc[idx]
                pred_R = rollout_one_row(row, model, feats)

                true_vals = []
                pred_vals = []

                out = {
                    "cv": cv_name,
                    "fold": int(fold),
                    "feature_set": fs_name,
                    "source_row": int(idx),
                    "group_id": row[GROUP_COL],
                    "condition": row["condition"],
                    "phase_target": row["phase_target"],
                }

                for l in R_LAYERS:
                    true_R = float(row[f"R_{l}"])
                    pred_val = float(pred_R[l])

                    out[f"true_R_{l}"] = true_R
                    out[f"pred_R_{l}"] = pred_val
                    out[f"err_R_{l}"] = pred_val - true_R

                    true_vals.append(true_R)
                    pred_vals.append(pred_val)

                true_tail = np.array([float(row[f"R_{l}"]) for l in ROLLOUT_TARGET_LAYERS])
                pred_tail = np.array([float(pred_R[l]) for l in ROLLOUT_TARGET_LAYERS])

                full_true = np.array(true_vals, dtype=float)
                full_pred = np.array(pred_vals, dtype=float)

                out["tail_mae_R23_26"] = float(np.mean(np.abs(pred_tail - true_tail)))
                out["tail_rmse_R23_26"] = float(np.sqrt(np.mean((pred_tail - true_tail) ** 2)))
                out["final_abs_error_R26"] = float(abs(pred_R[26] - float(row["R_26"])))

                if np.std(full_true) > 0 and np.std(full_pred) > 0:
                    out["traj_corr_R20_26"] = float(np.corrcoef(full_true, full_pred)[0, 1])
                else:
                    out["traj_corr_R20_26"] = 0.0

                if np.std(true_tail) > 0 and np.std(pred_tail) > 0:
                    out["tail_corr_R23_26"] = float(np.corrcoef(true_tail, pred_tail)[0, 1])
                else:
                    out["tail_corr_R23_26"] = 0.0

                fold_preds.append(out)
                rollout_rows.append(out)

        pred_df = pd.DataFrame(fold_preds)

        # global metrics over all rollout rows
        y_true = []
        y_pred = []

        for _, r in pred_df.iterrows():
            for l in ROLLOUT_TARGET_LAYERS:
                y_true.append(r[f"true_R_{l}"])
                y_pred.append(r[f"pred_R_{l}"])

        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)

        summary_rows.append({
            "cv": cv_name,
            "feature_set": fs_name,
            "features": ",".join(feats),
            "rollout_target_layers": ",".join(map(str, ROLLOUT_TARGET_LAYERS)),
            "rollout_r2_R23_26": float(r2_score(y_true, y_pred)),
            "rollout_corr_R23_26": float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_pred) > 0 else 0.0,
            "rollout_mae_R23_26": float(mean_absolute_error(y_true, y_pred)),
            "mean_tail_mae": float(pred_df["tail_mae_R23_26"].mean()),
            "mean_tail_rmse": float(pred_df["tail_rmse_R23_26"].mean()),
            "mean_final_abs_error_R26": float(pred_df["final_abs_error_R26"].mean()),
            "mean_traj_corr_R20_26": float(pred_df["traj_corr_R20_26"].mean()),
            "mean_tail_corr_R23_26": float(pred_df["tail_corr_R23_26"].mean()),
        })

rollout_df = pd.DataFrame(rollout_rows)
rollout_summary = pd.DataFrame(summary_rows).sort_values(
    ["cv", "rollout_r2_R23_26"],
    ascending=[True, False],
)

rollout_df.to_csv(SAVE_DIR / "phasemap3a2_rollout_predictions.csv", index=False)
rollout_summary.to_csv(SAVE_DIR / "phasemap3a2_rollout_summary.csv", index=False)

# ============================================================
# PHASE / CONDITION ROLLOUT SUMMARY
# ============================================================

phase_summary = rollout_df.groupby(["cv", "feature_set", "phase_target"]).agg(
    n=("source_row", "count"),
    tail_mae=("tail_mae_R23_26", "mean"),
    final_abs_error_R26=("final_abs_error_R26", "mean"),
    traj_corr=("traj_corr_R20_26", "mean"),
    tail_corr=("tail_corr_R23_26", "mean"),
).reset_index()

condition_summary = rollout_df.groupby(["cv", "feature_set", "condition"]).agg(
    n=("source_row", "count"),
    phase=("phase_target", lambda x: x.mode().iloc[0]),
    tail_mae=("tail_mae_R23_26", "mean"),
    final_abs_error_R26=("final_abs_error_R26", "mean"),
    traj_corr=("traj_corr_R20_26", "mean"),
    tail_corr=("tail_corr_R23_26", "mean"),
).reset_index()

phase_summary.to_csv(SAVE_DIR / "phasemap3a2_rollout_by_phase.csv", index=False)
condition_summary.to_csv(SAVE_DIR / "phasemap3a2_rollout_by_condition.csv", index=False)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_rows = []

for fs_name, feats in FEATURE_SETS.items():
    X = step_df[feats].values.astype(float)
    y = step_df["v_next"].values.astype(float)

    model = make_model()
    model.fit(X, y)

    for f, imp in zip(feats, model.feature_importances_):
        importance_rows.append({
            "feature_set": fs_name,
            "feature": f,
            "importance": float(imp),
        })

importance_df = pd.DataFrame(importance_rows).sort_values(
    ["feature_set", "importance"],
    ascending=[True, False],
)

importance_df.to_csv(SAVE_DIR / "phasemap3a2_feature_importance.csv", index=False)

# ============================================================
# SAVE STEP DATASET
# ============================================================

step_df.to_csv(SAVE_DIR / "phasemap3a2_step_dataset.csv", index=False)

# ============================================================
# SUMMARY JSON
# ============================================================

def best_rows(table, cv=None, n=8):
    sub = table.copy()
    if cv is not None:
        sub = sub[sub["cv"] == cv]
    return sub.head(n).to_dict(orient="records")


def metric(table, cv, feature_set, key):
    sub = table[(table["cv"] == cv) & (table["feature_set"] == feature_set)]
    if len(sub) == 0:
        return None
    return float(sub.iloc[0][key])


summary = {
    "experiment": "PhaseMap-3A.2 Multi-Step Rollout Audit",
    "input_path": str(INPUT_PATH),
    "n_original_rows": int(len(df)),
    "n_step_rows": int(len(step_df)),
    "group_col": GROUP_COL,
    "core_question": "Can one-step dynamics learned from H=(R,v,a,...) roll out R23-R26 from initial R20-R22?",
    "rollout_protocol": {
        "given": ["R20", "R21", "R22"],
        "predict": ["R23", "R24", "R25", "R26"],
        "recursive_state": "R_{l-1}, R_l, R_{l+1} -> predict v_next = R_{l+2}-R_{l+1}"
    },
    "key_metrics": {
        "Group_RVA_rollout_r2": metric(rollout_summary, "GroupKFold", "RVA", "rollout_r2_R23_26"),
        "Group_RVA_rollout_corr": metric(rollout_summary, "GroupKFold", "RVA", "rollout_corr_R23_26"),
        "Group_RVA_layer_rollout_r2": metric(rollout_summary, "GroupKFold", "RVA_layer", "rollout_r2_R23_26"),
        "Group_H_shape_layer_rollout_r2": metric(rollout_summary, "GroupKFold", "H_shape_layer", "rollout_r2_R23_26"),
        "Group_H_shape_layer_traj_corr": metric(rollout_summary, "GroupKFold", "H_shape_layer", "mean_traj_corr_R20_26"),
        "Group_H_shape_layer_final_abs_error": metric(rollout_summary, "GroupKFold", "H_shape_layer", "mean_final_abs_error_R26"),
    },
    "best_one_step": {
        "KFold": best_rows(one_step_df[one_step_df["cv"] == "KFold"].sort_values("r2", ascending=False)),
        "GroupKFold": best_rows(one_step_df[one_step_df["cv"] == "GroupKFold"].sort_values("r2", ascending=False)),
    },
    "best_rollout": {
        "KFold": best_rows(rollout_summary[rollout_summary["cv"] == "KFold"].sort_values("rollout_r2_R23_26", ascending=False)),
        "GroupKFold": best_rows(rollout_summary[rollout_summary["cv"] == "GroupKFold"].sort_values("rollout_r2_R23_26", ascending=False)),
    },
    "feature_importance": importance_df.to_dict(orient="records"),
    "interpretation_rules": {
        "PASS_state_equation": [
            "Group rollout_r2_R23_26 > 0.75.",
            "Mean trajectory correlation > 0.80.",
            "Final R26 error remains moderate.",
            "This supports a genuine low-dimensional dynamical state equation."
        ],
        "PASS_one_step_only": [
            "One-step v_next is strong but rollout collapses.",
            "Dynamics are locally predictable but errors compound; need stabilizing latent variable."
        ],
        "PASS_layer_clock_rollout": [
            "Layer/phase clock greatly improves rollout.",
            "Dynamics are non-autonomous; state equation requires clock variable."
        ],
        "FAIL_rollout": [
            "Rollout R2 <= 0.5 or trajectory correlation low.",
            "One-step closure does not imply generative trajectory dynamics."
        ],
        "next_if_pass": "PhaseMap-3B: fit explicit interpretable dynamical equation and phase portrait.",
        "next_if_fail": "PhaseMap-3B: broader hidden state/history discovery."
    }
}

with open(SAVE_DIR / "phasemap3a2_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nSaved to: {SAVE_DIR.resolve()}")