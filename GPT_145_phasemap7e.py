# ============================================================
# PhaseMap-7E: State Equation Ablation Audit
#
# Goal
#   Audit the minimal closed state space for the layer transition:
#
#       Z_l -> Z_{l+1}
#
#   and test whether continuous operator coordinates O_cont are
#   necessary for the state-transition equation.
#
# Input
#   Fixed local path:
#       phasemap6b1_correction_dataset.csv
#
# Output
#   phasemap7e_outputs/
#       phasemap7e_model_summary.csv
#       phasemap7e_fold_summary.csv
#       phasemap7e_target_summary.csv
#       phasemap7e_component_ablation.csv
#       phasemap7e_residual_by_operator.csv
#       phasemap7e_residual_by_phase.csv
#       phasemap7e_feature_inventory.csv
#       phasemap7e_summary.json
#
# Notes
#   This script separates STRICT pre-transition state features from
#   OBSERVED transition/operator features.
#
#   STRICT features avoid columns such as center_cos/jaccard/center_step,
#   because those summarize l -> l+1 and can leak future information.
#
#   O_cont = op_pc1..op_pc10 is treated as a latent transition/operator
#   coordinate. It is not a pure pre-transition state, but PhaseMap-7B/7C
#   suggested it is the correct intermediate variable for F_O.
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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures

warnings.filterwarnings("ignore")

# --------------------------
# Fixed paths
# --------------------------
INPUT_CSV = r"C:\Windows\System32\phasemap6b1_outputs\phasemap6b1_correction_dataset.csv"
OUTPUT_DIR = Path(r"phasemap7e_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_SPLITS = 5
RIDGE_ALPHA = 10.0

# --------------------------
# Utilities
# --------------------------

def safe_cols(df, cols):
    return [c for c in cols if c in df.columns]


def make_preprocessor(df, features):
    num_cols = []
    cat_cols = []
    for c in features:
        if c not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

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

    if not transformers:
        raise ValueError("No valid features supplied.")

    return ColumnTransformer(transformers, remainder="drop"), num_cols, cat_cols


def build_pipeline(df, features, alpha=RIDGE_ALPHA):
    pre, num_cols, cat_cols = make_preprocessor(df, features)
    model = Ridge(alpha=alpha, random_state=RANDOM_SEED)
    pipe = Pipeline([
        ("pre", pre),
        ("model", model),
    ])
    return pipe, num_cols, cat_cols


def regression_metrics(y_true, y_pred, target_cols):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    out = {}
    out["macro_r2"] = float(r2_score(y_true, y_pred, multioutput="uniform_average"))
    out["variance_weighted_r2"] = float(r2_score(y_true, y_pred, multioutput="variance_weighted"))
    out["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    out["mae"] = float(mean_absolute_error(y_true, y_pred))

    per_target = []
    for j, t in enumerate(target_cols):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        if np.std(yt) < 1e-12:
            r2 = np.nan
            corr = np.nan
        else:
            r2 = float(r2_score(yt, yp))
            corr = float(np.corrcoef(yt, yp)[0, 1]) if np.std(yp) > 1e-12 else np.nan
        per_target.append({
            "target": t,
            "r2": r2,
            "corr": corr,
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "mae": float(mean_absolute_error(yt, yp)),
        })
    return out, per_target


def add_bilinear_features(df, left_cols, right_cols, prefix="bilin"):
    out = df.copy()
    left_cols = safe_cols(out, left_cols)
    right_cols = safe_cols(out, right_cols)
    new_cols = []
    for a in left_cols:
        for b in right_cols:
            nc = f"{prefix}__{a}__x__{b}"
            out[nc] = pd.to_numeric(out[a], errors="coerce") * pd.to_numeric(out[b], errors="coerce")
            new_cols.append(nc)
    return out, new_cols


def fit_cv(df, features, target_cols, model_name, groups=None):
    X = df[features].copy()
    y = df[target_cols].astype(float).values

    if groups is not None and len(np.unique(groups)) >= N_SPLITS:
        splitter = GroupKFold(n_splits=N_SPLITS)
        splits = splitter.split(X, y, groups=groups)
        split_type = "GroupKFold_graph"
    else:
        splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        splits = splitter.split(X, y)
        split_type = "KFold"

    oof = np.zeros_like(y, dtype=float)
    fold_rows = []

    for fold, (tr, te) in enumerate(splits):
        pipe, num_cols, cat_cols = build_pipeline(df, features)
        pipe.fit(X.iloc[tr], y[tr])
        pred = pipe.predict(X.iloc[te])
        oof[te] = pred
        metrics, per_t = regression_metrics(y[te], pred, target_cols)
        row = {
            "model": model_name,
            "fold": fold,
            "split_type": split_type,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            **metrics,
            "n_features_requested": len(features),
            "n_num_features": len(num_cols),
            "n_cat_features": len(cat_cols),
        }
        fold_rows.append(row)

    metrics, per_target = regression_metrics(y, oof, target_cols)
    summary = {
        "model": model_name,
        "split_type": split_type,
        "n": int(len(df)),
        "n_features_requested": int(len(features)),
        "features": ";".join(features),
        **metrics,
    }

    target_rows = []
    for r in per_target:
        target_rows.append({"model": model_name, **r})

    resid = y - oof
    resid_df = df[[c for c in ["sample_id", "graph_id", "condition", "phase", "layer", "transition", "heuristic_operator", "operator_cluster"] if c in df.columns]].copy()
    for j, t in enumerate(target_cols):
        resid_df[f"true__{t}"] = y[:, j]
        resid_df[f"pred__{t}"] = oof[:, j]
        resid_df[f"resid__{t}"] = resid[:, j]
    resid_df["resid_l2"] = np.sqrt(np.mean(resid ** 2, axis=1))
    resid_df["model"] = model_name

    return summary, fold_rows, target_rows, resid_df


def summarize_residuals(resid_df, group_col):
    if group_col not in resid_df.columns:
        return pd.DataFrame()
    return (
        resid_df.groupby(["model", group_col], dropna=False)
        .agg(
            n=("resid_l2", "size"),
            resid_l2_mean=("resid_l2", "mean"),
            resid_l2_median=("resid_l2", "median"),
            resid_l2_std=("resid_l2", "std"),
        )
        .reset_index()
    )

# --------------------------
# Main
# --------------------------

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"Could not find {INPUT_CSV}. Put this script in the same directory as phasemap6b1_correction_dataset.csv."
        )

    df0 = pd.read_csv(INPUT_CSV)
    df = df0.copy()

    # Core targets: next-layer state observables.
    target_cols = safe_cols(df, [
        "R_l1",
        "boundary_dist_l1",
        "spread_l1",
        "rank_gap_l1",
    ])
    if not target_cols:
        raise RuntimeError("No target columns found.")

    # Strict pre-transition components.
    R_cols = safe_cols(df, ["R_l", "rank_gap_l"])
    B_cols = safe_cols(df, ["boundary_dist_l"])
    T_cols = safe_cols(df, ["spread_l"])

    # History / shape features available before the current transition.
    H_cols = safe_cols(df, [
        "R_prev_delta",
        "rank_gap_prev_delta",
        "R_velocity_reversal",
        "rank_gap_reversal",
        "center_backtrack_prevprev",
        "center_backtrack_init",
    ])

    # Latent/current operator coordinates.
    O_pc_cols = safe_cols(df, [f"op_pc{i}" for i in range(1, 11)])
    O_label_cols = safe_cols(df, ["heuristic_operator", "operator_cluster", "is_correction", "unstable_reversal", "reconstructive_reversal"])

    context_cols = safe_cols(df, ["layer", "phase", "condition"])

    # Observed transition signature: this is a diagnostic upper bound, not a strict state model.
    observed_transition_cols = safe_cols(df, [
        "jaccard",
        "center_cos",
        "center_step",
        "spread_delta",
        "R_delta",
        "rank_gap_delta",
        "boundary_dist_delta",
        "correction_candidate",
        "weak_confidence",
    ])

    # Build bilinear state x operator interactions for F_O(Z).
    strict_state_cols = R_cols + B_cols + T_cols + H_cols
    df_bilin, bilin_cols = add_bilinear_features(df, strict_state_cols, O_pc_cols, prefix="ZxO")
    df = df_bilin

    # Model definitions.
    models = []
    def add(name, cols):
        cols = safe_cols(df, list(dict.fromkeys(cols)))
        if cols:
            models.append((name, cols))

    add("M0_R_only", R_cols)
    add("M1_R_plus_B", R_cols + B_cols)
    add("M2_RBT_state", R_cols + B_cols + T_cols)
    add("M3_RBT_plus_Hshape", R_cols + B_cols + T_cols + H_cols)
    add("M4_Hshape_only", H_cols)
    add("M5_state_plus_Ocont", R_cols + B_cols + T_cols + H_cols + O_pc_cols)
    add("M6_state_plus_Olabel", R_cols + B_cols + T_cols + H_cols + O_label_cols)
    add("M7_state_plus_Ocont_label", R_cols + B_cols + T_cols + H_cols + O_pc_cols + O_label_cols)
    add("M8_state_Ocont_bilinear", R_cols + B_cols + T_cols + H_cols + O_pc_cols + bilin_cols)
    add("M9_state_plus_context", R_cols + B_cols + T_cols + H_cols + context_cols)
    add("M10_state_Ocont_context", R_cols + B_cols + T_cols + H_cols + O_pc_cols + context_cols)
    add("D0_observed_transition_upper_bound", R_cols + B_cols + T_cols + H_cols + O_pc_cols + observed_transition_cols)
    add("D1_all_available_diagnostic", R_cols + B_cols + T_cols + H_cols + O_pc_cols + O_label_cols + context_cols + observed_transition_cols)

    # Leave-one-component-out from the main full equation.
    full_components = {
        "R": R_cols,
        "B": B_cols,
        "T": T_cols,
        "Hshape": H_cols,
        "Ocont": O_pc_cols,
    }
    full_cols = []
    for cols in full_components.values():
        full_cols.extend(cols)
    full_cols = list(dict.fromkeys(full_cols))
    add("A0_full_RBTHO", full_cols)
    for comp, cols in full_components.items():
        keep = [c for c in full_cols if c not in cols]
        add(f"Ablate_without_{comp}", keep)

    # Feature inventory.
    inv_rows = []
    for group, cols in {
        "R": R_cols,
        "B": B_cols,
        "T": T_cols,
        "Hshape": H_cols,
        "Ocont": O_pc_cols,
        "Olabel": O_label_cols,
        "context": context_cols,
        "observed_transition_diagnostic": observed_transition_cols,
        "bilinear_ZxO": bilin_cols,
        "targets": target_cols,
    }.items():
        inv_rows.append({"group": group, "n_cols": len(cols), "cols": ";".join(cols)})
    pd.DataFrame(inv_rows).to_csv(OUTPUT_DIR / "phasemap7e_feature_inventory.csv", index=False)

    groups = df["graph_id"].values if "graph_id" in df.columns else None

    all_summary = []
    all_folds = []
    all_targets = []
    all_resid = []

    print("\nPhaseMap-7E starting")
    print(f"Input rows: {len(df):,}")
    print(f"Targets: {target_cols}")
    print(f"Models: {len(models)}")

    for i, (name, cols) in enumerate(models, 1):
        print(f"[{i}/{len(models)}] {name} | features={len(cols)}")
        try:
            summary, folds, targets, resid_df = fit_cv(df, cols, target_cols, name, groups=groups)
            all_summary.append(summary)
            all_folds.extend(folds)
            all_targets.extend(targets)
            all_resid.append(resid_df)
        except Exception as e:
            all_summary.append({
                "model": name,
                "status": "failed",
                "error": repr(e),
                "n_features_requested": len(cols),
                "features": ";".join(cols),
            })
            print(f"  FAILED: {e}")

    summary_df = pd.DataFrame(all_summary)
    fold_df = pd.DataFrame(all_folds)
    target_df = pd.DataFrame(all_targets)
    resid_all = pd.concat(all_resid, ignore_index=True) if all_resid else pd.DataFrame()

    summary_df.to_csv(OUTPUT_DIR / "phasemap7e_model_summary.csv", index=False)
    fold_df.to_csv(OUTPUT_DIR / "phasemap7e_fold_summary.csv", index=False)
    target_df.to_csv(OUTPUT_DIR / "phasemap7e_target_summary.csv", index=False)
    if not resid_all.empty:
        # Save a compact residual table for all models. It can be large but useful.
        resid_all.to_csv(OUTPUT_DIR / "phasemap7e_residuals.csv", index=False)
        summarize_residuals(resid_all, "heuristic_operator").to_csv(
            OUTPUT_DIR / "phasemap7e_residual_by_operator.csv", index=False
        )
        summarize_residuals(resid_all, "phase").to_csv(
            OUTPUT_DIR / "phasemap7e_residual_by_phase.csv", index=False
        )
        summarize_residuals(resid_all, "condition").to_csv(
            OUTPUT_DIR / "phasemap7e_residual_by_condition.csv", index=False
        )

    # Component ablation relative to A0_full_RBTHO and M3/M5.
    comp_rows = []
    def get_r2(model):
        s = summary_df[summary_df["model"] == model]
        if len(s) == 0 or "macro_r2" not in s:
            return np.nan
        return float(s.iloc[0]["macro_r2"])

    full_r2 = get_r2("A0_full_RBTHO")
    m3_r2 = get_r2("M3_RBT_plus_Hshape")
    m5_r2 = get_r2("M5_state_plus_Ocont")
    m8_r2 = get_r2("M8_state_Ocont_bilinear")

    for comp in full_components:
        r2_abl = get_r2(f"Ablate_without_{comp}")
        comp_rows.append({
            "component": comp,
            "full_model": "A0_full_RBTHO",
            "full_macro_r2": full_r2,
            "ablated_model": f"Ablate_without_{comp}",
            "ablated_macro_r2": r2_abl,
            "drop_when_removed": full_r2 - r2_abl if np.isfinite(full_r2) and np.isfinite(r2_abl) else np.nan,
            "interpretation": "larger positive drop => more necessary for closure",
        })

    comp_rows.extend([
        {
            "component": "Ocont_gain",
            "full_model": "M5_state_plus_Ocont",
            "full_macro_r2": m5_r2,
            "ablated_model": "M3_RBT_plus_Hshape",
            "ablated_macro_r2": m3_r2,
            "drop_when_removed": m5_r2 - m3_r2 if np.isfinite(m5_r2) and np.isfinite(m3_r2) else np.nan,
            "interpretation": "positive => continuous operator coordinates modulate transition",
        },
        {
            "component": "Bilinear_gain",
            "full_model": "M8_state_Ocont_bilinear",
            "full_macro_r2": m8_r2,
            "ablated_model": "M5_state_plus_Ocont",
            "ablated_macro_r2": m5_r2,
            "drop_when_removed": m8_r2 - m5_r2 if np.isfinite(m8_r2) and np.isfinite(m5_r2) else np.nan,
            "interpretation": "positive => F_O is state-dependent, not just additive operator bias",
        },
    ])
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(OUTPUT_DIR / "phasemap7e_component_ablation.csv", index=False)

    # Main readout.
    def row_for(model):
        s = summary_df[summary_df["model"] == model]
        return s.iloc[0].to_dict() if len(s) else {}

    main = {
        "M0_R_only_r2": get_r2("M0_R_only"),
        "M2_RBT_state_r2": get_r2("M2_RBT_state"),
        "M3_RBT_plus_Hshape_r2": m3_r2,
        "M4_Hshape_only_r2": get_r2("M4_Hshape_only"),
        "M5_state_plus_Ocont_r2": m5_r2,
        "M8_state_Ocont_bilinear_r2": m8_r2,
        "delta_Hshape_vs_RBT": m3_r2 - get_r2("M2_RBT_state") if np.isfinite(m3_r2) and np.isfinite(get_r2("M2_RBT_state")) else np.nan,
        "delta_Ocont_vs_state": m5_r2 - m3_r2 if np.isfinite(m5_r2) and np.isfinite(m3_r2) else np.nan,
        "delta_bilinear_vs_additive_O": m8_r2 - m5_r2 if np.isfinite(m8_r2) and np.isfinite(m5_r2) else np.nan,
        "diagnostic_observed_transition_upper_bound_r2": get_r2("D0_observed_transition_upper_bound"),
    }

    readout = {
        "input_csv": INPUT_CSV,
        "n_rows": int(len(df)),
        "target_cols": target_cols,
        "feature_groups": {r["group"]: r["cols"] for r in inv_rows},
        "main_readout": main,
        "caveats": [
            "op_pc columns are latent/operator coordinates derived from transition signatures; they validate F_O but are not pure pre-transition observables.",
            "center_cos/jaccard/center_step are treated only as diagnostic upper-bound features because they summarize the observed l->l+1 transition.",
            "boundary_dist_l may be redundant with R_l in this dataset; check feature_inventory and ablation drop.",
        ],
    }
    with open(OUTPUT_DIR / "phasemap7e_summary.json", "w", encoding="utf-8") as f:
        json.dump(readout, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print("Main readout:")
    for k, v in main.items():
        print(f"  {k}: {v}")
    print(f"\nOutputs saved to: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
