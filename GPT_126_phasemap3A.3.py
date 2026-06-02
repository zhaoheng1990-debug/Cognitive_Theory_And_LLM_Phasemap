import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path(r"C:\Windows\System32\phasemap2c_outputs\phasemap2c_eval.csv")
SAVE_DIR = Path("./phasemap3a3_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
R_LAYERS = list(range(20, 27))
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
# LOCAL STATE FEATURES
# ============================================================

def local_state_values(R_prev, R_l, R_next, layer):
    v_prev = R_l - R_prev
    v_l = R_next - R_l
    a_l = v_l - v_prev

    local_vals = np.array([R_prev, R_l, R_next], dtype=float)

    local_mean = float(local_vals.mean())
    local_span = float(local_vals.max() - local_vals.min())
    local_energy = float(R_l ** 2 + v_l ** 2)
    local_abs_energy = float(abs(R_l) + abs(v_l))
    curvature_energy = float(a_l ** 2)

    # slow-energy hand-crafted candidates
    E_sum = local_abs_energy + local_span
    E_quad = local_energy + curvature_energy
    E_mixed = local_abs_energy + local_span + np.sqrt(curvature_energy + 1e-9)

    phase_clock = 0 if layer <= 22 else 1
    layer_norm = (layer - 22.5) / 1.11803398875

    return {
        "R_l": float(R_l),
        "v_l": float(v_l),
        "a_l": float(a_l),
        "local_mean": local_mean,
        "local_span": local_span,
        "local_energy": local_energy,
        "local_abs_energy": local_abs_energy,
        "curvature_energy": curvature_energy,
        "E_sum": float(E_sum),
        "E_quad": float(E_quad),
        "E_mixed": float(E_mixed),
        "layer_norm": float(layer_norm),
        "phase_clock": int(phase_clock),
    }


rows = []

for idx, row in df.iterrows():
    for l in range(21, 25):
        R_prev = float(row[f"R_{l-1}"])
        R_l = float(row[f"R_{l}"])
        R_next = float(row[f"R_{l+1}"])
        R_next2 = float(row[f"R_{l+2}"])

        vals = local_state_values(R_prev, R_l, R_next, l)
        v_next = R_next2 - R_next

        rows.append({
            "source_row": int(idx),
            "group_id": row[GROUP_COL],
            "condition": row["condition"],
            "phase_target": row["phase_target"],
            "layer": int(l),
            **vals,
            "v_next": float(v_next),
        })

step_df = pd.DataFrame(rows)

# ============================================================
# LEARN E_PC1 FROM SLOW VARIABLES
# ============================================================

SLOW_COLS = [
    "local_abs_energy",
    "local_span",
    "local_energy",
    "curvature_energy",
    "E_sum",
    "E_quad",
    "E_mixed",
]

scaler = StandardScaler()
Z = scaler.fit_transform(step_df[SLOW_COLS].values.astype(float))

pca = PCA(n_components=3, random_state=SEED)
PC = pca.fit_transform(Z)

step_df["E_PC1"] = PC[:, 0]
step_df["E_PC2"] = PC[:, 1]
step_df["E_PC3"] = PC[:, 2]

pca_loadings = pd.DataFrame(
    pca.components_.T,
    index=SLOW_COLS,
    columns=["E_PC1", "E_PC2", "E_PC3"],
).reset_index().rename(columns={"index": "variable"})

pca_loadings.to_csv(SAVE_DIR / "phasemap3a3_E_pca_loadings.csv", index=False)

# ============================================================
# FEATURE SETS
# ============================================================

FEATURE_SETS = {
    "RVA": ["R_l", "v_l", "a_l"],
    "RVA_E_PC1": ["R_l", "v_l", "a_l", "E_PC1"],
    "RVA_slow_all": ["R_l", "v_l", "a_l"] + SLOW_COLS,
    "H_shape_layer": [
        "R_l", "v_l", "a_l",
        "local_mean", "local_span",
        "local_energy", "local_abs_energy",
        "curvature_energy",
        "layer_norm",
    ],
}

# ============================================================
# MODELS / SPLITS
# ============================================================

def make_model(kind):
    if kind == "rf":
          return RandomForestRegressor(
             n_estimators=200,
             max_depth=5,
             min_samples_leaf=8,
             random_state=SEED,
             n_jobs=-1,
)
    if kind == "gbr":
        return GradientBoostingRegressor(
            n_estimators=350,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=5,
            random_state=SEED,
        )

    raise ValueError(kind)


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
# ONE-STEP EVAL
# ============================================================

one_step_rows = []

for cv_name, group_mode in [("KFold", False), ("GroupKFold", True)]:
    splits = get_splits(step_df, group_mode=group_mode)

    for model_kind in ["rf", "gbr"]:
        for fs_name, feats in FEATURE_SETS.items():
            X = step_df[feats].values.astype(float)
            y = step_df["v_next"].values.astype(float)

            pred = np.zeros(len(step_df), dtype=float)

            for tr, te in splits:
                model = make_model(model_kind)
                model.fit(X[tr], y[tr])
                pred[te] = model.predict(X[te])

            one_step_rows.append({
                "cv": cv_name,
                "model": model_kind,
                "feature_set": fs_name,
                "features": ",".join(feats),
                "target": "v_next",
                "r2": float(r2_score(y, pred)),
                "corr": float(np.corrcoef(y, pred)[0, 1]) if np.std(pred) > 0 else 0.0,
                "mae": float(mean_absolute_error(y, pred)),
            })

one_step_df = pd.DataFrame(one_step_rows).sort_values(
    ["cv", "model", "r2"],
    ascending=[True, True, False],
)

one_step_df.to_csv(SAVE_DIR / "phasemap3a3_one_step_eval.csv", index=False)

# ============================================================
# ROLLOUT HELPERS
# ============================================================

def make_feature_vector_from_pred(pred_R, layer, feature_set):
    R_prev = pred_R[layer - 1]
    R_l = pred_R[layer]
    R_next = pred_R[layer + 1]

    vals = local_state_values(R_prev, R_l, R_next, layer)

    # compute E_PC from local slow cols using fitted scaler/pca
    slow_vec = np.array([[vals[c] for c in SLOW_COLS]], dtype=float)
    pc = pca.transform(scaler.transform(slow_vec))[0]
    vals["E_PC1"] = float(pc[0])
    vals["E_PC2"] = float(pc[1])
    vals["E_PC3"] = float(pc[2])

    return np.array([vals[f] for f in feature_set], dtype=float)


def rollout_one_row(row, model, feature_set):
    pred_R = {
        20: float(row["R_20"]),
        21: float(row["R_21"]),
        22: float(row["R_22"]),
    }

    for layer in range(21, 25):
        X = make_feature_vector_from_pred(pred_R, layer, feature_set)
        v_next_pred = float(model.predict(X.reshape(1, -1))[0])
        pred_R[layer + 2] = pred_R[layer + 1] + v_next_pred

    return pred_R


# ============================================================
# MULTI-STEP ROLLOUT
# ============================================================

rollout_rows = []
summary_rows = []

for cv_name, group_mode in [("KFold", False), ("GroupKFold", True)]:
    splits = get_splits(df, group_mode=group_mode)

    for model_kind in ["rf", "gbr"]:
        for fs_name, feats in FEATURE_SETS.items():
            fold_preds = []

            for fold, (tr_idx, te_idx) in enumerate(splits):
                train_rows = set(tr_idx.tolist())
                train_step = step_df[step_df["source_row"].isin(train_rows)].copy()

                X_train = train_step[feats].values.astype(float)
                y_train = train_step["v_next"].values.astype(float)

                model = make_model(model_kind)
                model.fit(X_train, y_train)

                for idx in te_idx:
                    row = df.iloc[idx]
                    pred_R = rollout_one_row(row, model, feats)

                    out = {
                        "cv": cv_name,
                        "model": model_kind,
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

                    true_tail = np.array([float(row[f"R_{l}"]) for l in ROLLOUT_TARGET_LAYERS])
                    pred_tail = np.array([float(pred_R[l]) for l in ROLLOUT_TARGET_LAYERS])

                    full_true = np.array([float(row[f"R_{l}"]) for l in R_LAYERS])
                    full_pred = np.array([float(pred_R[l]) for l in R_LAYERS])

                    out["tail_mae_R23_26"] = float(np.mean(np.abs(pred_tail - true_tail)))
                    out["tail_rmse_R23_26"] = float(np.sqrt(np.mean((pred_tail - true_tail) ** 2)))
                    out["final_abs_error_R26"] = float(abs(pred_R[26] - float(row["R_26"])))

                    out["traj_corr_R20_26"] = (
                        float(np.corrcoef(full_true, full_pred)[0, 1])
                        if np.std(full_true) > 0 and np.std(full_pred) > 0
                        else 0.0
                    )
                    out["tail_corr_R23_26"] = (
                        float(np.corrcoef(true_tail, pred_tail)[0, 1])
                        if np.std(true_tail) > 0 and np.std(pred_tail) > 0
                        else 0.0
                    )

                    fold_preds.append(out)
                    rollout_rows.append(out)

            pred_df = pd.DataFrame(fold_preds)

            y_true, y_pred = [], []
            for _, r in pred_df.iterrows():
                for l in ROLLOUT_TARGET_LAYERS:
                    y_true.append(r[f"true_R_{l}"])
                    y_pred.append(r[f"pred_R_{l}"])

            y_true = np.array(y_true, dtype=float)
            y_pred = np.array(y_pred, dtype=float)

            summary_rows.append({
                "cv": cv_name,
                "model": model_kind,
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
    ["cv", "model", "rollout_r2_R23_26"],
    ascending=[True, True, False],
)

rollout_df.to_csv(SAVE_DIR / "phasemap3a3_rollout_predictions.csv", index=False)
rollout_summary.to_csv(SAVE_DIR / "phasemap3a3_rollout_summary.csv", index=False)

# ============================================================
# PHASE / CONDITION SUMMARY
# ============================================================

phase_summary = rollout_df.groupby(["cv", "model", "feature_set", "phase_target"]).agg(
    n=("source_row", "count"),
    tail_mae=("tail_mae_R23_26", "mean"),
    final_abs_error_R26=("final_abs_error_R26", "mean"),
    traj_corr=("traj_corr_R20_26", "mean"),
    tail_corr=("tail_corr_R23_26", "mean"),
).reset_index()

condition_summary = rollout_df.groupby(["cv", "model", "feature_set", "condition"]).agg(
    n=("source_row", "count"),
    phase=("phase_target", lambda x: x.mode().iloc[0]),
    tail_mae=("tail_mae_R23_26", "mean"),
    final_abs_error_R26=("final_abs_error_R26", "mean"),
    traj_corr=("traj_corr_R20_26", "mean"),
    tail_corr=("tail_corr_R23_26", "mean"),
).reset_index()

phase_summary.to_csv(SAVE_DIR / "phasemap3a3_rollout_by_phase.csv", index=False)
condition_summary.to_csv(SAVE_DIR / "phasemap3a3_rollout_by_condition.csv", index=False)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_rows = []

for model_kind in ["rf", "gbr"]:
    for fs_name, feats in FEATURE_SETS.items():
        X = step_df[feats].values.astype(float)
        y = step_df["v_next"].values.astype(float)

        model = make_model(model_kind)
        model.fit(X, y)

        if hasattr(model, "feature_importances_"):
            for f, imp in zip(feats, model.feature_importances_):
                importance_rows.append({
                    "model": model_kind,
                    "feature_set": fs_name,
                    "feature": f,
                    "importance": float(imp),
                })

importance_df = pd.DataFrame(importance_rows).sort_values(
    ["model", "feature_set", "importance"],
    ascending=[True, True, False],
)

importance_df.to_csv(SAVE_DIR / "phasemap3a3_feature_importance.csv", index=False)

# ============================================================
# SAVE STEP DATASET
# ============================================================

step_df.to_csv(SAVE_DIR / "phasemap3a3_step_dataset.csv", index=False)

# ============================================================
# SUMMARY JSON
# ============================================================

def metric(table, cv, model, feature_set, key):
    sub = table[
        (table["cv"] == cv)
        & (table["model"] == model)
        & (table["feature_set"] == feature_set)
    ]
    if len(sub) == 0:
        return None
    return float(sub.iloc[0][key])


def best_rows(table, cv=None, model=None, n=10, metric_col="rollout_r2_R23_26"):
    sub = table.copy()
    if cv is not None:
        sub = sub[sub["cv"] == cv]
    if model is not None:
        sub = sub[sub["model"] == model]
    return sub.sort_values(metric_col, ascending=False).head(n).to_dict(orient="records")


summary = {
    "experiment": "PhaseMap-3A.3 Slow Variable Energy Audit",
    "input_path": str(INPUT_PATH),
    "n_original_rows": int(len(df)),
    "n_step_rows": int(len(step_df)),
    "group_col": GROUP_COL,
    "core_question": "Can a slow energy variable E improve multi-step rollout beyond RVA / H_shape?",
    "slow_variables": SLOW_COLS,
    "E_pca_explained_variance": pca.explained_variance_ratio_.tolist(),
    "key_metrics": {
        "Group_rf_RVA_rollout_r2": metric(rollout_summary, "GroupKFold", "rf", "RVA", "rollout_r2_R23_26"),
        "Group_rf_RVA_E_PC1_rollout_r2": metric(rollout_summary, "GroupKFold", "rf", "RVA_E_PC1", "rollout_r2_R23_26"),
        "Group_rf_RVA_E_PC123_rollout_r2": metric(rollout_summary, "GroupKFold", "rf", "RVA_E_PC123", "rollout_r2_R23_26"),
        "Group_rf_RVA_slow_all_rollout_r2": metric(rollout_summary, "GroupKFold", "rf", "RVA_slow_all", "rollout_r2_R23_26"),
        "Group_rf_RVA_slow_all_layer_rollout_r2": metric(rollout_summary, "GroupKFold", "rf", "RVA_slow_all_layer", "rollout_r2_R23_26"),
        "Group_rf_H_shape_layer_rollout_r2": metric(rollout_summary, "GroupKFold", "rf", "H_shape_layer", "rollout_r2_R23_26"),
    },
    "best_one_step": {
        "GroupKFold_rf": best_rows(one_step_df, cv="GroupKFold", model="rf", metric_col="r2"),
        "GroupKFold_gbr": best_rows(one_step_df, cv="GroupKFold", model="gbr", metric_col="r2"),
    },
    "best_rollout": {
        "GroupKFold_rf": best_rows(rollout_summary, cv="GroupKFold", model="rf"),
        "GroupKFold_gbr": best_rows(rollout_summary, cv="GroupKFold", model="gbr"),
    },
    "feature_importance": importance_df.head(50).to_dict(orient="records"),
    "interpretation_rules": {
        "PASS_slow_variable": [
            "RVA_E_PC1 or RVA_slow_all improves rollout_r2 over RVA by > 0.08.",
            "Rollout R2 approaches or exceeds 0.75.",
            "Slow energy variables reduce final R26 error."
        ],
        "PASS_partial": [
            "Slow variables improve modestly but rollout remains below 0.75.",
            "Slow variables exist but not sufficient for closed rollout."
        ],
        "FAIL_slow_variable": [
            "Slow variables fail to improve over RVA / H_shape baselines.",
            "Rollout failure comes from another missing latent state."
        ],
        "next_if_pass": "PhaseMap-3B: explicit fast-slow dynamical equation.",
        "next_if_partial": "PhaseMap-3B: search broader memory latent variables.",
        "next_if_fail": "Return to history-state discovery beyond local R trajectory."
    }
}

with open(SAVE_DIR / "phasemap3a3_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nSaved to: {SAVE_DIR.resolve()}")