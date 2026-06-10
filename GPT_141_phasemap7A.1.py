# ============================================================
# PhaseMap-7A.1 v2
# State Variable Ablation / State-Space Closure Audit
#
# This version matches phasemap6b1_correction_dataset.csv.
# No command-line path is required. Put this script in the same
# folder as phasemap6b1_correction_dataset.csv and run:
#
#     python phasemap7a1_state_variable_ablation_v2.py
#
# Goal:
#   Test whether a minimal state Z_l closes the transition:
#
#       Z_l -> Z_{l+1}
#
#   and whether adding operator O_l improves the transition map:
#
#       Z_{l+1} = F(Z_l, O_l)
#
# Input schema expected from PhaseMap-6B.1 correction dataset:
#   sample_id, graph_id, condition, phase, layer, transition,
#   R_l, R_l1, spread_l, spread_l1,
#   boundary_dist_l, boundary_dist_l1,
#   rank_gap_l, rank_gap_l1,
#   heuristic_operator, operator_cluster,
#   optional op_pc1..op_pc10 and diagnostic transition columns.
#
# Important:
#   center_cos / jaccard / center_step are transition-observed metrics
#   because they already compare layer l with l+1. They are useful as
#   diagnostic/upper-bound features, but not leakage-safe state variables.
#   Therefore the main readout uses leakage-safe state variables only.
# ============================================================

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# Fixed paths: edit only here if your filenames change.
# ============================================================

INPUT_CSV = r"C:\Windows\System32\phasemap6b1_outputs\phasemap6b1_correction_dataset.csv"
OUTPUT_DIR = r"phasemap7a1_outputs_v2"
RANDOM_SEED = 42
N_SPLITS = 5
RIDGE_ALPHA = 1.0

# Use layers where l -> l+1 exists in the dataset.
# If None, all observed layers are used.
TRACK_LAYERS = None

# ============================================================
# Utilities
# ============================================================

def find_input_path() -> Path:
    """Find input CSV with minimal friction.

    Priority:
      1. Current working directory / INPUT_CSV
      2. Script directory / INPUT_CSV
      3. /mnt/data / INPUT_CSV, useful inside ChatGPT sandbox
    """
    candidates = [
        Path.cwd() / INPUT_CSV,
        Path(__file__).resolve().parent / INPUT_CSV,
        Path("/mnt/data") / INPUT_CSV,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot find input CSV. Put phasemap6b1_correction_dataset.csv "
        "in the same folder as this script, or edit INPUT_CSV at the top.\n"
        + "Tried:\n"
        + "\n".join(str(x) for x in candidates)
    )


def ensure_cols(df: pd.DataFrame, cols, label: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {label} columns: {missing}\nAvailable columns: {df.columns.tolist()}")


def numeric_existing(df: pd.DataFrame, cols):
    return [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def categorical_existing(df: pd.DataFrame, cols):
    return [c for c in cols if c in df.columns]


def safe_corr(x, y):
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)
    m = x.notna() & y.notna()
    if m.sum() < 3:
        return np.nan
    if x[m].std() == 0 or y[m].std() == 0:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def make_preprocess(X: pd.DataFrame):
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            num_cols,
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cat_cols,
        ))

    return ColumnTransformer(transformers, remainder="drop")


def fit_predict_cv(df, feature_cols, target_cols, group_col="graph_id", n_splits=5):
    data = df[feature_cols + target_cols + ([group_col] if group_col in df.columns else [])].copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=target_cols)

    if len(data) < 20:
        return None, None

    X = data[feature_cols]
    Y = data[target_cols].astype(float)

    if group_col in data.columns and data[group_col].nunique() >= 2:
        groups = data[group_col]
        n = min(n_splits, groups.nunique())
        cv = GroupKFold(n_splits=n).split(X, Y, groups)
        cv_name = "GroupKFold"
    else:
        n = min(n_splits, len(data))
        cv = KFold(n_splits=n, shuffle=True, random_state=RANDOM_SEED).split(X, Y)
        cv_name = "KFold"

    pred = np.zeros_like(Y.values, dtype=float)
    fold_ids = np.full(len(data), -1, dtype=int)

    for fold, (tr, te) in enumerate(cv):
        pipe = Pipeline([
            ("prep", make_preprocess(X.iloc[tr])),
            ("model", Ridge(alpha=RIDGE_ALPHA)),
        ])
        pipe.fit(X.iloc[tr], Y.iloc[tr])
        pred[te] = pipe.predict(X.iloc[te])
        fold_ids[te] = fold

    pred_df = data[[c for c in [group_col] if c in data.columns]].copy()
    pred_df["cv_fold"] = fold_ids
    for i, t in enumerate(target_cols):
        pred_df[f"true_{t}"] = Y.values[:, i]
        pred_df[f"pred_{t}"] = pred[:, i]
        pred_df[f"resid_{t}"] = Y.values[:, i] - pred[:, i]

    # Aggregate metrics.
    metrics = []
    y_true = Y.values
    y_pred = pred

    # multi-target global metrics
    metrics.append({
        "target": "__all__",
        "r2": float(r2_score(y_true, y_pred, multioutput="variance_weighted")),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "corr": safe_corr(y_true.ravel(), y_pred.ravel()),
        "n_rows": int(len(data)),
        "cv": cv_name,
    })

    for i, t in enumerate(target_cols):
        metrics.append({
            "target": t,
            "r2": float(r2_score(y_true[:, i], y_pred[:, i])),
            "rmse": float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))),
            "mae": float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
            "corr": safe_corr(y_true[:, i], y_pred[:, i]),
            "n_rows": int(len(data)),
            "cv": cv_name,
        })

    return pd.DataFrame(metrics), pred_df


def summarize_residual_by_operator(df_base, residuals, target_cols):
    join_cols = ["sample_id", "graph_id", "condition", "phase", "layer", "transition",
                 "heuristic_operator", "operator_cluster"]
    join_cols = [c for c in join_cols if c in df_base.columns]
    tmp = df_base.loc[residuals.index if residuals.index.equals(df_base.index) else df_base.index[:len(residuals)], join_cols].reset_index(drop=True)
    res = pd.concat([tmp, residuals.reset_index(drop=True)], axis=1)

    resid_cols = [f"resid_{t}" for t in target_cols if f"resid_{t}" in res.columns]
    if not resid_cols:
        return pd.DataFrame(), res

    res["resid_l2"] = np.sqrt(np.sum(np.square(res[resid_cols].values), axis=1))
    group_cols = [c for c in ["heuristic_operator", "operator_cluster", "phase", "condition", "layer"] if c in res.columns]

    rows = []
    for col in group_cols:
        g = res.groupby(col, dropna=False)["resid_l2"].agg(["count", "mean", "median", "std"]).reset_index()
        g.insert(0, "group_by", col)
        g = g.rename(columns={col: "group_value", "count": "n", "mean": "resid_l2_mean", "median": "resid_l2_median", "std": "resid_l2_std"})
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(), res

# ============================================================
# Main
# ============================================================

def main():
    input_path = find_input_path()
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df = df.replace([np.inf, -np.inf], np.nan)

    required = ["layer", "R_l", "R_l1", "spread_l", "spread_l1"]
    ensure_cols(df, required, "required")

    if TRACK_LAYERS is not None:
        df = df[df["layer"].isin(TRACK_LAYERS)].copy()

    # Targets: next-layer state. Keep leakage-safe and available.
    target_cols = numeric_existing(df, [
        "R_l1",
        "boundary_dist_l1",
        "spread_l1",
        "rank_gap_l1",
    ])

    # Core leakage-safe current-state variables.
    R_features = numeric_existing(df, ["R_l", "rank_gap_l"])
    B_features = numeric_existing(df, ["boundary_dist_l"])
    TopK_state_features = numeric_existing(df, ["spread_l"])

    # Previous-step dynamics are still current-known if the row was built from a trajectory.
    PrevDyn_features = numeric_existing(df, [
        "R_prev_delta", "rank_gap_prev_delta", "center_backtrack_init",
        "center_backtrack_prevprev", "R_velocity_reversal", "rank_gap_reversal",
    ])

    # Operator labels.
    O_cat_features = categorical_existing(df, ["heuristic_operator", "operator_cluster"])
    O_pc_features = numeric_existing(df, [f"op_pc{i}" for i in range(1, 11)])

    # Transition-observed metrics: useful diagnostic upper bound, not main closure.
    Transition_observed_features = numeric_existing(df, [
        "jaccard", "center_cos", "center_step", "spread_delta",
        "R_delta", "rank_gap_delta", "boundary_dist_delta",
        "correction_candidate", "is_correction", "unstable_reversal", "reconstructive_reversal",
        "weak_confidence",
    ])

    # Deduplicate while preserving order.
    def uniq(cols):
        seen, out = set(), []
        for c in cols:
            if c not in seen:
                out.append(c); seen.add(c)
        return out

    feature_sets = {
        "R_only": uniq(R_features),
        "B_only": uniq(B_features),
        "TopK_state_only": uniq(TopK_state_features),
        "R_plus_B": uniq(R_features + B_features),
        "R_plus_TopK": uniq(R_features + TopK_state_features),
        "B_plus_TopK": uniq(B_features + TopK_state_features),
        "Z_base_R_B_TopK": uniq(R_features + B_features + TopK_state_features),
        "Z_plus_prev_dynamics": uniq(R_features + B_features + TopK_state_features + PrevDyn_features),
        "Z_plus_operator_label": uniq(R_features + B_features + TopK_state_features + O_cat_features),
        "Z_plus_operator_pc": uniq(R_features + B_features + TopK_state_features + O_pc_features),
        "Z_plus_operator_all": uniq(R_features + B_features + TopK_state_features + O_cat_features + O_pc_features),
        "Z_plus_prev_plus_operator": uniq(R_features + B_features + TopK_state_features + PrevDyn_features + O_cat_features + O_pc_features),
        "DIAGNOSTIC_transition_observed_upper_bound": uniq(R_features + B_features + TopK_state_features + Transition_observed_features),
        "DIAGNOSTIC_all_available_no_future_targets": uniq(R_features + B_features + TopK_state_features + PrevDyn_features + O_cat_features + O_pc_features + Transition_observed_features),
    }

    # Remove empty feature sets.
    feature_sets = {k: v for k, v in feature_sets.items() if len(v) > 0}

    inventory_rows = []
    for name, cols in feature_sets.items():
        inventory_rows.append({
            "feature_set": name,
            "n_features": len(cols),
            "features": "|".join(cols),
        })
    inventory = pd.DataFrame(inventory_rows)
    inventory.to_csv(outdir / "phasemap7a1_feature_inventory.csv", index=False, encoding="utf-8-sig")

    # Redundancy audit: boundary_dist_l may duplicate R_l.
    redundancy_rows = []
    pairs = [
        ("R_l", "boundary_dist_l"),
        ("R_l1", "boundary_dist_l1"),
        ("R_delta", "boundary_dist_delta"),
    ]
    for a, b in pairs:
        if a in df.columns and b in df.columns:
            diff = (df[a].astype(float) - df[b].astype(float)).abs()
            redundancy_rows.append({
                "col_a": a,
                "col_b": b,
                "corr": safe_corr(df[a], df[b]),
                "max_abs_diff": float(diff.max()),
                "mean_abs_diff": float(diff.mean()),
                "effectively_identical": bool(diff.max() < 1e-9),
            })
    redundancy = pd.DataFrame(redundancy_rows)
    redundancy.to_csv(outdir / "phasemap7a1_redundancy_audit.csv", index=False, encoding="utf-8-sig")

    # Run ablations.
    all_metrics = []
    residual_frames = {}
    target_summary_rows = []

    for fs_name, cols in feature_sets.items():
        metrics, preds = fit_predict_cv(df, cols, target_cols, group_col="graph_id", n_splits=N_SPLITS)
        if metrics is None:
            continue
        metrics.insert(0, "feature_set", fs_name)
        metrics.insert(1, "n_features", len(cols))
        all_metrics.append(metrics)
        residual_frames[fs_name] = preds

    if not all_metrics:
        raise RuntimeError("No valid feature set could be evaluated.")

    ablation = pd.concat(all_metrics, ignore_index=True)

    # Compute deltas vs best single and vs Z_base.
    main = ablation[ablation["target"] == "__all__"].copy()
    single_names = ["R_only", "B_only", "TopK_state_only"]
    best_single_r2 = main[main["feature_set"].isin(single_names)]["r2"].max()
    z_base_r2 = main.loc[main["feature_set"] == "Z_base_R_B_TopK", "r2"]
    z_base_r2 = float(z_base_r2.iloc[0]) if len(z_base_r2) else np.nan

    main_r2_map = dict(zip(main["feature_set"], main["r2"]))
    delta_o_label = main_r2_map.get("Z_plus_operator_label", np.nan) - z_base_r2
    delta_o_pc = main_r2_map.get("Z_plus_operator_pc", np.nan) - z_base_r2
    delta_o_all = main_r2_map.get("Z_plus_operator_all", np.nan) - z_base_r2
    delta_z = z_base_r2 - best_single_r2 if np.isfinite(best_single_r2) else np.nan

    ablation["delta_vs_best_single"] = ablation["r2"] - best_single_r2
    ablation["delta_vs_Z_base"] = ablation["r2"] - z_base_r2
    ablation.to_csv(outdir / "phasemap7a1_ablation_summary.csv", index=False, encoding="utf-8-sig")

    # Per-target compact summary.
    for t in target_cols:
        sub = ablation[ablation["target"] == t].sort_values("r2", ascending=False)
        if len(sub):
            best = sub.iloc[0].to_dict()
            target_summary_rows.append({
                "target": t,
                "best_feature_set": best["feature_set"],
                "best_r2": best["r2"],
                "best_corr": best["corr"],
                "best_rmse": best["rmse"],
                "best_mae": best["mae"],
            })
    target_summary = pd.DataFrame(target_summary_rows)
    target_summary.to_csv(outdir / "phasemap7a1_target_summary.csv", index=False, encoding="utf-8-sig")

    # Residual analysis for key models.
    key_models = [
        "Z_base_R_B_TopK",
        "Z_plus_operator_label",
        "Z_plus_operator_pc",
        "Z_plus_operator_all",
        "Z_plus_prev_plus_operator",
        "DIAGNOSTIC_transition_observed_upper_bound",
    ]
    key_models = [m for m in key_models if m in residual_frames]

    residual_all = []
    residual_by_group_all = []
    for m in key_models:
        pred = residual_frames[m].copy()
        pred.insert(0, "feature_set", m)
        # Attach metadata by positional index. fit_predict_cv drops only rows with missing target,
        # target columns are complete in this dataset, so this is safe. If future data has missing
        # targets, this still remains an approximate diagnostic.
        meta_cols = [c for c in ["sample_id", "graph_id", "condition", "phase", "layer", "transition", "heuristic_operator", "operator_cluster"] if c in df.columns]
        meta = df[meta_cols].iloc[:len(pred)].reset_index(drop=True)
        pred = pd.concat([meta, pred.reset_index(drop=True)], axis=1)
        resid_cols = [f"resid_{t}" for t in target_cols if f"resid_{t}" in pred.columns]
        pred["resid_l2"] = np.sqrt(np.sum(np.square(pred[resid_cols].values), axis=1)) if resid_cols else np.nan
        residual_all.append(pred)

        for col in ["heuristic_operator", "operator_cluster", "phase", "condition", "layer"]:
            if col in pred.columns:
                g = pred.groupby(col, dropna=False)["resid_l2"].agg(["count", "mean", "median", "std"]).reset_index()
                g.insert(0, "feature_set", m)
                g.insert(1, "group_by", col)
                g = g.rename(columns={col: "group_value", "count": "n", "mean": "resid_l2_mean", "median": "resid_l2_median", "std": "resid_l2_std"})
                residual_by_group_all.append(g)

    if residual_all:
        pd.concat(residual_all, ignore_index=True).to_csv(outdir / "phasemap7a1_residuals.csv", index=False, encoding="utf-8-sig")
    if residual_by_group_all:
        pd.concat(residual_by_group_all, ignore_index=True).to_csv(outdir / "phasemap7a1_residual_by_group.csv", index=False, encoding="utf-8-sig")

    # Operator-local models: fit separate local F_O for each heuristic_operator.
    local_rows = []
    if "heuristic_operator" in df.columns:
        base_cols = feature_sets.get("Z_base_R_B_TopK", [])
        for op, subdf in df.groupby("heuristic_operator", dropna=False):
            if len(subdf) < 40 or not base_cols:
                continue
            metrics, _ = fit_predict_cv(subdf, base_cols, target_cols, group_col="graph_id", n_splits=min(N_SPLITS, 5))
            if metrics is None:
                continue
            row = metrics[metrics["target"] == "__all__"].iloc[0].to_dict()
            row["heuristic_operator"] = op
            row["n_rows_operator"] = int(len(subdf))
            local_rows.append(row)
    local_summary = pd.DataFrame(local_rows)
    if len(local_summary):
        local_summary = local_summary[["heuristic_operator", "n_rows_operator", "r2", "rmse", "mae", "corr", "cv", "n_rows", "target"]]
    local_summary.to_csv(outdir / "phasemap7a1_operator_local_summary.csv", index=False, encoding="utf-8-sig")

    # Main readout.
    best_overall = main.sort_values("r2", ascending=False).iloc[0].to_dict()
    summary = {
        "experiment": "PhaseMap-7A.1 v2 State Variable Ablation / State-Space Closure Audit",
        "input": str(input_path),
        "outdir": str(outdir),
        "n_rows": int(len(df)),
        "layers": sorted([int(x) for x in df["layer"].dropna().unique().tolist()]),
        "targets": target_cols,
        "group_col": "graph_id" if "graph_id" in df.columns else None,
        "model": f"Ridge(alpha={RIDGE_ALPHA})",
        "feature_sets": feature_sets,
        "main_readout": {
            "best_overall_feature_set": best_overall.get("feature_set"),
            "best_overall_r2": best_overall.get("r2"),
            "best_overall_corr": best_overall.get("corr"),
            "best_single_r2": None if not np.isfinite(best_single_r2) else float(best_single_r2),
            "Z_base_r2": None if not np.isfinite(z_base_r2) else float(z_base_r2),
            "delta_Z_base_minus_best_single": None if not np.isfinite(delta_z) else float(delta_z),
            "delta_O_label_vs_Z_base": None if not np.isfinite(delta_o_label) else float(delta_o_label),
            "delta_O_pc_vs_Z_base": None if not np.isfinite(delta_o_pc) else float(delta_o_pc),
            "delta_O_all_vs_Z_base": None if not np.isfinite(delta_o_all) else float(delta_o_all),
        },
        "interpretation_rules": {
            "closure": "Z_base_R_B_TopK > best(R_only, B_only, TopK_state_only) supports a combined closed state.",
            "operator_modulation": "Z_plus_operator_* > Z_base_R_B_TopK supports F_O rather than a single F.",
            "boundary_independence": "Check phasemap7a1_redundancy_audit.csv; if B_l is identical to R_l, B is not independent in this dataset.",
            "transition_observed_warning": "DIAGNOSTIC_transition_observed_upper_bound uses center_cos/jaccard/center_step, which compare l and l+1 and should be treated as diagnostic upper bound, not leakage-safe state closure.",
        },
        "outputs": {
            "feature_inventory": str(outdir / "phasemap7a1_feature_inventory.csv"),
            "redundancy_audit": str(outdir / "phasemap7a1_redundancy_audit.csv"),
            "ablation_summary": str(outdir / "phasemap7a1_ablation_summary.csv"),
            "target_summary": str(outdir / "phasemap7a1_target_summary.csv"),
            "residuals": str(outdir / "phasemap7a1_residuals.csv"),
            "residual_by_group": str(outdir / "phasemap7a1_residual_by_group.csv"),
            "operator_local_summary": str(outdir / "phasemap7a1_operator_local_summary.csv"),
        },
    }

    with open(outdir / "phasemap7a1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== PhaseMap-7A.1 v2 finished ===")
    print("Input:", input_path)
    print("Output dir:", outdir)
    print("Rows:", len(df))
    print("Targets:", target_cols)
    print("\nMain readout:")
    print(json.dumps(summary["main_readout"], indent=2, ensure_ascii=False))
    print("\nTop overall models:")
    print(main.sort_values("r2", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
