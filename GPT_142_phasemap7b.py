# ============================================================
# PhaseMap-7B v2-fixed
# State Transition Equation Audit
#
# Goal:
#   Fit and compare candidate transition equations:
#
#       Z_{l+1} = F(Z_l)
#       Z_{l+1} = F(Z_l, O_l^{label})
#       Z_{l+1} = F(Z_l, O_l^{cont})
#       Z_{l+1} = F(Z_l, O_l^{cont}, Z_l \otimes O_l^{cont})
#
# This version fixes the previous categorical preprocessing bug:
#   - numeric columns -> median imputation + scaling
#   - categorical columns -> most-frequent imputation + one-hot encoding
#
# IMPORTANT:
#   The input path is hardcoded for the user's workflow.
#   Put this script in the same folder as phasemap6b1_correction_dataset.csv,
#   then run:
#       python phasemap7b_state_transition_equation_v2_fixed.py
# ============================================================

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG: edit here if needed, no command-line args required
# ============================================================

INPUT_CSV = r"C:\Windows\System32\phasemap6b1_outputs\phasemap6b1_correction_dataset.csv"
OUTPUT_DIR = r"phasemap7b_outputs_v2_fixed"

RANDOM_SEED = 42
RIDGE_ALPHA = 1.0
N_SPLITS = 5

TARGET_COLS = [
    "R_l1",
    "boundary_dist_l1",
    "spread_l1",
    "rank_gap_l1",
]

STATE_NUMERIC = [
    "R_l",
    "boundary_dist_l",
    "spread_l",
    "rank_gap_l",
]

PREV_DYNAMICS_NUMERIC = [
    "R_prev_delta",
    "rank_gap_prev_delta",
    "center_backtrack_init",
    "center_backtrack_prevprev",
    "R_velocity_reversal",
    "rank_gap_reversal",
]

OPERATOR_PC_NUMERIC = [f"op_pc{i}" for i in range(1, 11)]

OPERATOR_LABEL_CATEGORICAL = [
    "heuristic_operator",
    "operator_cluster",
]

# Diagnostic features that compare l and l+1. Useful only as an observed-transition upper bound.
# They are NOT leakage-safe if the goal is pre-transition prediction.
OBSERVED_TRANSITION_NUMERIC = [
    "jaccard",
    "center_cos",
    "center_step",
    "spread_delta",
    "R_delta",
    "rank_gap_delta",
    "boundary_dist_delta",
    "correction_candidate",
    "is_correction",
    "unstable_reversal",
    "reconstructive_reversal",
    "weak_confidence",
]

GROUP_COL_CANDIDATES = ["graph_id", "sample_id"]
LAYER_COL = "layer"

# ============================================================
# Utilities
# ============================================================

def existing_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def add_bilinear_terms(df: pd.DataFrame, state_cols: List[str], op_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    new_cols = []
    for s in state_cols:
        if s not in out.columns:
            continue
        for o in op_cols:
            if o not in out.columns:
                continue
            name = f"bilinear__{s}__x__{o}"
            out[name] = pd.to_numeric(out[s], errors="coerce") * pd.to_numeric(out[o], errors="coerce")
            new_cols.append(name)
    return out, new_cols



def make_onehot_encoder():
    """Return a dense OneHotEncoder compatible with old and new sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def make_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    transformers = []
    if numeric_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, numeric_cols))
    if categorical_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot_encoder()),
        ])
        transformers.append(("cat", cat_pipe, categorical_cols))
    if not transformers:
        raise ValueError("No usable features for preprocessor.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_model(numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    return Pipeline([
        ("prep", make_preprocessor(numeric_cols, categorical_cols)),
        ("ridge", Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_SEED)),
    ])


def get_cv(df: pd.DataFrame, group_col: str | None):
    if group_col and group_col in df.columns:
        groups = df[group_col].values
        n_groups = len(pd.unique(groups))
        if n_groups >= N_SPLITS:
            return GroupKFold(n_splits=N_SPLITS), groups, f"GroupKFold({group_col})"
    return KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED), None, "KFold(shuffled)"


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    if np.nanstd(a[ok]) == 0 or np.nanstd(b[ok]) == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def multi_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_cols: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["r2_macro"] = float(r2_score(y_true, y_pred, multioutput="uniform_average"))
    out["r2_variance_weighted"] = float(r2_score(y_true, y_pred, multioutput="variance_weighted"))
    out["rmse_macro"] = float(np.mean(np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))))
    out["mae_macro"] = float(mean_absolute_error(y_true, y_pred, multioutput="uniform_average"))
    cors = []
    for i, t in enumerate(target_cols):
        r2_i = r2_score(y_true[:, i], y_pred[:, i])
        rmse_i = float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
        mae_i = mean_absolute_error(y_true[:, i], y_pred[:, i])
        corr_i = safe_corr(y_true[:, i], y_pred[:, i])
        out[f"r2__{t}"] = float(r2_i)
        out[f"rmse__{t}"] = float(rmse_i)
        out[f"mae__{t}"] = float(mae_i)
        out[f"corr__{t}"] = float(corr_i) if np.isfinite(corr_i) else np.nan
        if np.isfinite(corr_i):
            cors.append(corr_i)
    out["corr_macro"] = float(np.mean(cors)) if cors else np.nan
    return out


def fit_predict_cv(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    target_cols: List[str],
    group_col: str | None,
    model_name: str,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict[str, Any]]:
    cols = numeric_cols + categorical_cols
    use = df[cols + target_cols + ([group_col] if group_col else [])].copy()
    use = use.dropna(subset=target_cols).reset_index(drop=True)

    X = use[cols]
    y = use[target_cols].values.astype(float)

    cv, groups, cv_kind = get_cv(use, group_col)
    preds = np.full_like(y, fill_value=np.nan, dtype=float)
    fold_rows = []

    split_iter = cv.split(X, y, groups=groups) if groups is not None else cv.split(X, y)
    for fold, (tr, te) in enumerate(split_iter):
        pipe = make_model(numeric_cols, categorical_cols)
        X_tr = X.iloc[tr]
        X_te = X.iloc[te]
        y_tr = y[tr]
        y_te = y[te]
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        preds[te] = pred
        m = multi_metrics(y_te, pred, target_cols)
        m.update({"model": model_name, "fold": fold, "n_train": len(tr), "n_test": len(te), "cv_kind": cv_kind})
        fold_rows.append(m)

    metrics = multi_metrics(y, preds, target_cols)
    metrics.update({
        "model": model_name,
        "n_rows": int(len(use)),
        "n_numeric_features": int(len(numeric_cols)),
        "n_categorical_features": int(len(categorical_cols)),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "cv_kind": cv_kind,
    })
    residuals = y - preds
    fold_df = pd.DataFrame(fold_rows)
    return preds, residuals, fold_df, metrics


def fit_full_and_export_coefficients(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    target_cols: List[str],
    out_path: Path,
    model_name: str,
):
    cols = numeric_cols + categorical_cols
    use = df[cols + target_cols].dropna(subset=target_cols).copy()
    X = use[cols]
    y = use[target_cols].values.astype(float)

    pipe = make_model(numeric_cols, categorical_cols)
    pipe.fit(X, y)
    prep = pipe.named_steps["prep"]
    ridge = pipe.named_steps["ridge"]

    try:
        feat_names = prep.get_feature_names_out().tolist()
    except Exception:
        feat_names = [f"feature_{i}" for i in range(ridge.coef_.shape[1])]

    rows = []
    coefs = np.asarray(ridge.coef_)
    # For multi-target Ridge: shape (n_targets, n_features)
    if coefs.ndim == 1:
        coefs = coefs.reshape(1, -1)
    for ti, target in enumerate(target_cols):
        for fi, fname in enumerate(feat_names):
            rows.append({
                "model": model_name,
                "target": target,
                "feature": fname,
                "coef": float(coefs[ti, fi]),
                "abs_coef": float(abs(coefs[ti, fi])),
            })
    coef_df = pd.DataFrame(rows).sort_values(["target", "abs_coef"], ascending=[True, False])
    coef_df.to_csv(out_path, index=False, encoding="utf-8-sig")


def audit_redundancy(df: pd.DataFrame, out_path: Path):
    pairs = [
        ("R_l", "boundary_dist_l"),
        ("R_l1", "boundary_dist_l1"),
        ("R_delta", "boundary_dist_delta"),
        ("rank_gap_l", "R_l"),
        ("rank_gap_l1", "R_l1"),
    ]
    rows = []
    for a, b in pairs:
        if a in df.columns and b in df.columns:
            aa = pd.to_numeric(df[a], errors="coerce").values
            bb = pd.to_numeric(df[b], errors="coerce").values
            diff = aa - bb
            rows.append({
                "a": a,
                "b": b,
                "corr": safe_corr(aa, bb),
                "max_abs_diff": float(np.nanmax(np.abs(diff))),
                "mean_abs_diff": float(np.nanmean(np.abs(diff))),
                "n": int(np.isfinite(diff).sum()),
            })
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


def summarize_by_operator(df: pd.DataFrame, residual: np.ndarray, target_cols: List[str], out_path: Path, model_name: str):
    if "heuristic_operator" not in df.columns:
        return
    rnorm = np.sqrt(np.nanmean(residual ** 2, axis=1))
    tmp = df[["heuristic_operator"]].copy().reset_index(drop=True)
    tmp["residual_rmse_row"] = rnorm
    for i, t in enumerate(target_cols):
        tmp[f"residual__{t}"] = residual[:, i]
    g = tmp.groupby("heuristic_operator", dropna=False).agg(
        n=("residual_rmse_row", "size"),
        residual_rmse_mean=("residual_rmse_row", "mean"),
        residual_rmse_median=("residual_rmse_row", "median"),
        residual_rmse_std=("residual_rmse_row", "std"),
    ).reset_index()
    g.insert(0, "model", model_name)
    g.to_csv(out_path, index=False, encoding="utf-8-sig")

# ============================================================
# Main
# ============================================================

def main():
    in_path = Path(INPUT_CSV)
    if not in_path.exists():
        alt = Path.cwd() / INPUT_CSV
        if alt.exists():
            in_path = alt
        else:
            raise FileNotFoundError(
                f"Cannot find {INPUT_CSV}. Put this script in the same folder as the CSV, "
                f"or edit INPUT_CSV at the top of the script. Current folder: {Path.cwd()}"
            )

    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # Choose group column
    group_col = None
    for c in GROUP_COL_CANDIDATES:
        if c in df.columns:
            group_col = c
            break

    # Ensure numeric columns are numeric
    all_numeric_candidates = list(dict.fromkeys(
        TARGET_COLS + STATE_NUMERIC + PREV_DYNAMICS_NUMERIC + OPERATOR_PC_NUMERIC + OBSERVED_TRANSITION_NUMERIC
    ))
    df = ensure_numeric(df, existing_cols(df, all_numeric_candidates))

    target_cols = existing_cols(df, TARGET_COLS)
    if not target_cols:
        raise RuntimeError(f"No target columns found. Expected one of: {TARGET_COLS}")

    state_num = existing_cols(df, STATE_NUMERIC)
    prev_num = existing_cols(df, PREV_DYNAMICS_NUMERIC)
    op_pc_num = existing_cols(df, OPERATOR_PC_NUMERIC)
    op_label_cat = existing_cols(df, OPERATOR_LABEL_CATEGORICAL)
    obs_num = existing_cols(df, OBSERVED_TRANSITION_NUMERIC)

    # Add bilinear interaction terms for state x continuous operator PCs
    df_bilin, bilinear_cols = add_bilinear_terms(df, state_num, op_pc_num)

    feature_sets: Dict[str, Dict[str, List[str]]] = {
        "F0_state_only": {
            "num": state_num,
            "cat": [],
        },
        "F1_state_plus_operator_pc": {
            "num": state_num + op_pc_num,
            "cat": [],
        },
        "F2_state_plus_operator_label": {
            "num": state_num,
            "cat": op_label_cat,
        },
        "F3_state_plus_prev_dynamics": {
            "num": state_num + prev_num,
            "cat": [],
        },
        "F4_state_plus_prev_plus_operator_pc": {
            "num": state_num + prev_num + op_pc_num,
            "cat": [],
        },
        "F5_state_plus_operator_label_and_pc": {
            "num": state_num + op_pc_num,
            "cat": op_label_cat,
        },
        "F6_bilinear_state_x_operator_pc": {
            "num": state_num + op_pc_num + bilinear_cols,
            "cat": [],
        },
        "F7_bilinear_plus_label": {
            "num": state_num + op_pc_num + bilinear_cols,
            "cat": op_label_cat,
        },
        "DIAGNOSTIC_observed_transition_upper_bound": {
            "num": state_num + obs_num,
            "cat": [],
        },
        "DIAGNOSTIC_all_available_no_future_targets": {
            "num": state_num + prev_num + op_pc_num + bilinear_cols + obs_num,
            "cat": op_label_cat,
        },
    }

    inventory_rows = []
    for name, spec in feature_sets.items():
        inventory_rows.append({
            "model": name,
            "n_numeric": len(spec["num"]),
            "n_categorical": len(spec["cat"]),
            "numeric_features": ";".join(spec["num"]),
            "categorical_features": ";".join(spec["cat"]),
        })
    pd.DataFrame(inventory_rows).to_csv(outdir / "phasemap7b_feature_inventory.csv", index=False, encoding="utf-8-sig")

    audit_redundancy(df_bilin, outdir / "phasemap7b_redundancy_audit.csv")

    summary_rows = []
    fold_dfs = []
    residual_frames = []
    operator_residual_summaries = []

    for model_name, spec in feature_sets.items():
        num_cols = spec["num"]
        cat_cols = spec["cat"]
        if not num_cols and not cat_cols:
            continue
        print(f"\nRunning {model_name}: num={len(num_cols)}, cat={len(cat_cols)}")
        preds, residual, fold_df, metrics = fit_predict_cv(
            df_bilin,
            num_cols,
            cat_cols,
            target_cols,
            group_col,
            model_name,
        )
        summary_rows.append(metrics)
        fold_dfs.append(fold_df)

        # Residual export, limited to useful identifiers plus targets/preds/residuals
        ids = [c for c in ["sample_id", "graph_id", "condition", "phase", "layer", "transition", "heuristic_operator"] if c in df_bilin.columns]
        res_df = df_bilin[ids].copy().reset_index(drop=True)
        for i, t in enumerate(target_cols):
            res_df[f"true__{t}"] = pd.to_numeric(df_bilin[t], errors="coerce").values
            res_df[f"pred__{t}"] = preds[:, i]
            res_df[f"resid__{t}"] = residual[:, i]
        res_df["row_rmse"] = np.sqrt(np.nanmean(residual ** 2, axis=1))
        res_df.insert(0, "model", model_name)
        residual_frames.append(res_df)

        op_path = outdir / f"phasemap7b_operator_residual__{model_name}.csv"
        summarize_by_operator(df_bilin, residual, target_cols, op_path, model_name)
        if op_path.exists():
            operator_residual_summaries.append(pd.read_csv(op_path))

        # Coefficients only for the main interpretable candidates to avoid huge files
        if model_name in [
            "F0_state_only",
            "F1_state_plus_operator_pc",
            "F2_state_plus_operator_label",
            "F6_bilinear_state_x_operator_pc",
        ]:
            fit_full_and_export_coefficients(
                df_bilin,
                num_cols,
                cat_cols,
                target_cols,
                outdir / f"phasemap7b_coefficients__{model_name}.csv",
                model_name,
            )

    summary_df = pd.DataFrame(summary_rows).sort_values("r2_macro", ascending=False)
    summary_df.to_csv(outdir / "phasemap7b_model_summary.csv", index=False, encoding="utf-8-sig")

    if fold_dfs:
        pd.concat(fold_dfs, ignore_index=True).to_csv(outdir / "phasemap7b_fold_summary.csv", index=False, encoding="utf-8-sig")
    if residual_frames:
        pd.concat(residual_frames, ignore_index=True).to_csv(outdir / "phasemap7b_residuals.csv", index=False, encoding="utf-8-sig")
    if operator_residual_summaries:
        pd.concat(operator_residual_summaries, ignore_index=True).to_csv(outdir / "phasemap7b_operator_residual_summary.csv", index=False, encoding="utf-8-sig")

    # Main readout
    def r2_of(name: str) -> float | None:
        row = summary_df[summary_df["model"] == name]
        if row.empty:
            return None
        return float(row.iloc[0]["r2_macro"])

    f0 = r2_of("F0_state_only")
    f1 = r2_of("F1_state_plus_operator_pc")
    f2 = r2_of("F2_state_plus_operator_label")
    f6 = r2_of("F6_bilinear_state_x_operator_pc")
    f7 = r2_of("F7_bilinear_plus_label")

    main_readout = {
        "F0_state_only_r2": f0,
        "F1_state_plus_operator_pc_r2": f1,
        "F2_state_plus_operator_label_r2": f2,
        "F6_bilinear_state_x_operator_pc_r2": f6,
        "F7_bilinear_plus_label_r2": f7,
        "delta_operator_pc_minus_state": None if f0 is None or f1 is None else f1 - f0,
        "delta_operator_label_minus_state": None if f0 is None or f2 is None else f2 - f0,
        "delta_bilinear_minus_operator_pc": None if f1 is None or f6 is None else f6 - f1,
        "delta_label_on_top_of_bilinear": None if f6 is None or f7 is None else f7 - f6,
        "best_model": str(summary_df.iloc[0]["model"]) if not summary_df.empty else None,
        "best_r2_macro": float(summary_df.iloc[0]["r2_macro"]) if not summary_df.empty else None,
    }

    report = {
        "experiment": "PhaseMap-7B v2-fixed State Transition Equation Audit",
        "input": str(in_path),
        "outdir": str(outdir),
        "n_rows": int(len(df_bilin)),
        "targets": target_cols,
        "group_col": group_col,
        "model": f"Ridge(alpha={RIDGE_ALPHA})",
        "feature_sets": feature_sets,
        "main_readout": main_readout,
        "interpretation_rules": {
            "operator_modulation": "F1 > F0 supports continuous operator modulation in Z_{l+1}=F(Z_l,O_l).",
            "label_modulation": "F2 > F0 tests whether discrete heuristic operator labels add predictive power.",
            "bilinear_local_equation": "F6 > F1 supports a local equation with state-operator interaction, Z_l \u2297 O_l.",
            "categorical_bug_fixed": "heuristic_operator/operator_cluster are one-hot encoded, never median-imputed.",
            "diagnostic_warning": "DIAGNOSTIC models use observed transition features such as R_delta/center_cos and should not be treated as leakage-safe pre-transition equations.",
        },
        "outputs": {
            "model_summary": str(outdir / "phasemap7b_model_summary.csv"),
            "fold_summary": str(outdir / "phasemap7b_fold_summary.csv"),
            "feature_inventory": str(outdir / "phasemap7b_feature_inventory.csv"),
            "redundancy_audit": str(outdir / "phasemap7b_redundancy_audit.csv"),
            "residuals": str(outdir / "phasemap7b_residuals.csv"),
            "operator_residual_summary": str(outdir / "phasemap7b_operator_residual_summary.csv"),
        },
    }
    with open(outdir / "phasemap7b_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== PhaseMap-7B v2-fixed main readout ===")
    for k, v in main_readout.items():
        print(f"{k}: {v}")
    print(f"\nSaved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
