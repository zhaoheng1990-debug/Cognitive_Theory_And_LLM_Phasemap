# ============================================================
# PhaseMap-7C v2
# Operator Coordinate Origin Audit
#
# Fixed paths. No command-line arguments needed.
# Run:
#     python phasemap7c_operator_coordinate_origin_audit_v2.py
#
# Core question:
#   7B proved that O_l^{cont}=op_pc1..op_pc10 strongly modulates
#   Z_l -> Z_{l+1}. 7C asks where O_l^{cont} comes from:
#
#       O_l^{cont} = G(Z_l, history, layer, phase/context) ?
#
# Leakage discipline:
#   Main models use only pre-transition variables.
#   DIAGNOSTIC models use observed transition variables and are upper bounds.
# ============================================================

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

INPUT_CSV = r"C:\Windows\System32\phasemap6b1_outputs\phasemap6b1_correction_dataset.csv"
OUTPUT_DIR = r"phasemap7c_outputs_v2"
RANDOM_SEED = 42
N_SPLITS = 5
RIDGE_ALPHA = 1.0
TARGETS = [f"op_pc{i}" for i in range(1, 11)]


def resolve_input_path():
    candidates = [
        Path(INPUT_CSV),
        Path.cwd() / INPUT_CSV,
        Path(__file__).resolve().parent / INPUT_CSV,
        Path("/mnt/data") / INPUT_CSV,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Cannot find phasemap6b1_correction_dataset.csv")


def safe_cols(df, cols):
    return [c for c in cols if c in df.columns]


def corr_1d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    a = a[m]
    b = b[m]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def macro_corr(y, yp):
    vals = [corr_1d(y[:, j], yp[:, j]) for j in range(y.shape[1])]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def make_preprocessor(num_cols, cat_cols):
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
        raise ValueError("No available features")
    return ColumnTransformer(transformers, remainder="drop")


def cv_splits(df):
    if "graph_id" in df.columns and df["graph_id"].nunique() >= N_SPLITS:
        return "GroupKFold", GroupKFold(n_splits=N_SPLITS).split(df, groups=df["graph_id"])
    return "KFold", KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED).split(df)


def fit_model(df, name, num_cols, cat_cols, targets):
    num_cols = safe_cols(df, num_cols)
    cat_cols = safe_cols(df, cat_cols)
    targets = safe_cols(df, targets)
    valid = df[targets].notna().all(axis=1)
    sub = df.loc[valid].reset_index(drop=True)
    y = sub[targets].to_numpy(dtype=float)
    X_cols = num_cols + cat_cols

    pipe = Pipeline([
        ("pre", make_preprocessor(num_cols, cat_cols)),
        ("ridge", Ridge(alpha=RIDGE_ALPHA)),
    ])

    preds = np.zeros_like(y)
    fold_rows = []
    cv_kind, splits = cv_splits(sub)
    for fold, (tr, te) in enumerate(splits):
        pipe.fit(sub.iloc[tr][X_cols], y[tr])
        yp = pipe.predict(sub.iloc[te][X_cols])
        preds[te] = yp
        fold_rows.append({
            "model": name,
            "fold": fold,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "r2_macro": float(r2_score(y[te], yp, multioutput="uniform_average")),
            "corr_macro": macro_corr(y[te], yp),
            "cv_kind": cv_kind,
        })

    summary = {
        "model": name,
        "n_rows": int(len(sub)),
        "n_num_features": int(len(num_cols)),
        "n_cat_features": int(len(cat_cols)),
        "num_features": ";".join(num_cols),
        "cat_features": ";".join(cat_cols),
        "r2_macro": float(r2_score(y, preds, multioutput="uniform_average")),
        "corr_macro": macro_corr(y, preds),
        "mae_macro": float(np.mean(np.abs(y - preds))),
        "rmse_macro": float(np.sqrt(np.mean((y - preds) ** 2))),
        "is_diagnostic": bool(name.startswith("DIAGNOSTIC") or "label" in name.lower()),
        "cv_kind": cv_kind,
    }

    target_rows = []
    for j, t in enumerate(targets):
        target_rows.append({
            "model": name,
            "target": t,
            "r2": float(r2_score(y[:, j], preds[:, j])),
            "corr": corr_1d(y[:, j], preds[:, j]),
            "mae": float(np.mean(np.abs(y[:, j] - preds[:, j]))),
            "rmse": float(np.sqrt(np.mean((y[:, j] - preds[:, j]) ** 2))),
            "target_std": float(np.std(y[:, j])),
        })

    residual_cols = [c for c in ["sample_id", "graph_id", "condition", "phase", "layer", "transition", "heuristic_operator", "operator_cluster"] if c in sub.columns]
    res = sub[residual_cols].copy()
    res["model"] = name
    resid = y - preds
    res["resid_l2"] = np.sqrt(np.sum(resid ** 2, axis=1))
    for j, t in enumerate(targets):
        res[f"true__{t}"] = y[:, j]
        res[f"pred__{t}"] = preds[:, j]
        res[f"resid__{t}"] = resid[:, j]

    # coefficient extraction on full data for interpretability
    coef_df = None
    try:
        pipe.fit(sub[X_cols], y)
        names = pipe.named_steps["pre"].get_feature_names_out()
        coefs = pipe.named_steps["ridge"].coef_
        rows = []
        for j, t in enumerate(targets):
            for k, f in enumerate(names):
                rows.append({"model": name, "target": t, "feature": f, "coef": float(coefs[j, k])})
        coef_df = pd.DataFrame(rows)
    except Exception as e:
        coef_df = pd.DataFrame([{"model": name, "error": str(e)}])

    return summary, pd.DataFrame(fold_rows), pd.DataFrame(target_rows), res, coef_df


def group_residual_summary(residuals, group_cols):
    out = []
    for c in group_cols:
        if c not in residuals.columns:
            continue
        g = residuals.groupby(["model", c], dropna=False)["resid_l2"].agg(["count", "mean", "median", "std", "max"]).reset_index()
        g = g.rename(columns={c: "group_value"})
        g.insert(1, "group_col", c)
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    input_path = resolve_input_path()
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    missing = [c for c in TARGETS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing target op_pc columns: {missing}")

    state = ["R_l", "boundary_dist_l", "spread_l", "rank_gap_l"]
    prev = ["R_prev_delta", "rank_gap_prev_delta", "center_backtrack_init", "center_backtrack_prevprev", "R_velocity_reversal", "rank_gap_reversal"]
    layer = ["layer"]
    phase_condition = ["phase", "condition"]
    label = ["heuristic_operator", "operator_cluster"]
    observed = ["jaccard", "center_cos", "center_step", "spread_delta", "R_delta", "rank_gap_delta", "boundary_dist_delta", "correction_candidate", "is_correction", "unstable_reversal", "reconstructive_reversal", "weak_confidence"]

    feature_sets = {
        # Strict pre-transition origin candidates
        "G0_layer_only": {"num": layer, "cat": []},
        "G1_state_only": {"num": state, "cat": []},
        "G2_prev_dynamics_only": {"num": prev, "cat": []},
        "G3_state_plus_prev": {"num": state + prev, "cat": []},
        "G4_state_plus_layer": {"num": state + layer, "cat": []},
        "G5_state_prev_layer": {"num": state + prev + layer, "cat": []},
        "G6_phase_condition_only": {"num": [], "cat": phase_condition},
        "G7_state_prev_layer_phase_condition": {"num": state + prev + layer, "cat": phase_condition},
        # Not an independent origin: labels are coarse operator summaries.
        "G8_operator_label_only_diagnostic": {"num": [], "cat": label},
        "G9_state_prev_layer_plus_label_diagnostic": {"num": state + prev + layer, "cat": label},
        # Transition-observed upper bounds: leakage for pre-transition origin.
        "DIAGNOSTIC_observed_transition_upper_bound": {"num": state + observed, "cat": []},
        "DIAGNOSTIC_all_available": {"num": state + prev + layer + observed, "cat": phase_condition + label},
    }

    summaries, folds, targets, residuals = [], [], [], []
    coef_paths = {}
    for name, spec in feature_sets.items():
        num = safe_cols(df, spec["num"])
        cat = safe_cols(df, spec["cat"])
        if not num and not cat:
            summaries.append({"model": name, "status": "skipped", "reason": "no features"})
            continue
        print(f"[7C] {name}: num={len(num)} cat={len(cat)}")
        s, f, t, r, coef = fit_model(df, name, num, cat, TARGETS)
        s["status"] = "ok"
        summaries.append(s)
        folds.append(f)
        targets.append(t)
        residuals.append(r)
        cp = outdir / f"phasemap7c_coefficients__{name}.csv"
        coef.to_csv(cp, index=False, encoding="utf-8-sig")
        coef_paths[name] = str(cp)

    model_summary = pd.DataFrame(summaries).sort_values("r2_macro", ascending=False, na_position="last")
    fold_summary = pd.concat(folds, ignore_index=True) if folds else pd.DataFrame()
    target_summary = pd.concat(targets, ignore_index=True) if targets else pd.DataFrame()
    residual_df = pd.concat(residuals, ignore_index=True) if residuals else pd.DataFrame()

    model_summary.to_csv(outdir / "phasemap7c_model_summary.csv", index=False, encoding="utf-8-sig")
    fold_summary.to_csv(outdir / "phasemap7c_fold_summary.csv", index=False, encoding="utf-8-sig")
    target_summary.to_csv(outdir / "phasemap7c_target_summary.csv", index=False, encoding="utf-8-sig")
    residual_df.to_csv(outdir / "phasemap7c_residuals.csv", index=False, encoding="utf-8-sig")
    group_residual_summary(residual_df, ["heuristic_operator", "operator_cluster", "phase", "condition", "layer"]).to_csv(outdir / "phasemap7c_residual_by_group.csv", index=False, encoding="utf-8-sig")

    inventory = []
    for name, spec in feature_sets.items():
        inventory.append({
            "model": name,
            "num_requested": ";".join(spec["num"]),
            "cat_requested": ";".join(spec["cat"]),
            "num_available": ";".join(safe_cols(df, spec["num"])),
            "cat_available": ";".join(safe_cols(df, spec["cat"])),
            "is_diagnostic": bool(name.startswith("DIAGNOSTIC") or "label" in name.lower()),
        })
    pd.DataFrame(inventory).to_csv(outdir / "phasemap7c_feature_inventory.csv", index=False, encoding="utf-8-sig")

    corr_cols = safe_cols(df, state + prev + layer + observed + TARGETS)
    df[corr_cols].corr(numeric_only=True).to_csv(outdir / "phasemap7c_correlation_audit.csv", encoding="utf-8-sig")

    def r2(name):
        m = model_summary[model_summary["model"] == name]
        if len(m) == 0 or "r2_macro" not in m:
            return None
        val = m.iloc[0]["r2_macro"]
        return None if pd.isna(val) else float(val)

    main = {
        "G0_layer_only_r2": r2("G0_layer_only"),
        "G1_state_only_r2": r2("G1_state_only"),
        "G2_prev_dynamics_only_r2": r2("G2_prev_dynamics_only"),
        "G3_state_plus_prev_r2": r2("G3_state_plus_prev"),
        "G5_state_prev_layer_r2": r2("G5_state_prev_layer"),
        "G7_state_prev_layer_phase_condition_r2": r2("G7_state_prev_layer_phase_condition"),
        "G8_operator_label_only_diagnostic_r2": r2("G8_operator_label_only_diagnostic"),
        "DIAGNOSTIC_observed_transition_upper_bound_r2": r2("DIAGNOSTIC_observed_transition_upper_bound"),
    }
    if main["G1_state_only_r2"] is not None and main["G5_state_prev_layer_r2"] is not None:
        main["delta_state_prev_layer_minus_state"] = main["G5_state_prev_layer_r2"] - main["G1_state_only_r2"]
    if main["G5_state_prev_layer_r2"] is not None and main["G7_state_prev_layer_phase_condition_r2"] is not None:
        main["delta_phase_condition_minus_state_prev_layer"] = main["G7_state_prev_layer_phase_condition_r2"] - main["G5_state_prev_layer_r2"]

    summary = {
        "experiment": "PhaseMap-7C v2 Operator Coordinate Origin Audit",
        "input": str(input_path),
        "outdir": str(outdir),
        "n_rows": int(len(df)),
        "targets": TARGETS,
        "group_col": "graph_id" if "graph_id" in df.columns else None,
        "model": f"Ridge(alpha={RIDGE_ALPHA})",
        "main_readout": main,
        "feature_sets": feature_sets,
        "coefficient_files": coef_paths,
        "interpretation_rules": {
            "strict_pre_transition_origin": "G1/G3/G5/G7 test whether O_l^{cont} can be predicted before seeing l->l+1 transition features.",
            "layer_clock": "If G0/G4/G5 are strong, operator coordinates may contain layer-clock or phase-clock structure.",
            "history_effect": "G3-G1 or G5-G1 estimates whether previous motion/momentum explains O_l^{cont} beyond current state.",
            "context_effect": "G7-G5 estimates whether phase/condition adds prompt-level operator-origin information.",
            "label_diagnostic": "G8/G9 test whether coarse operator labels summarize op_pc; they are not independent causal origins.",
            "observed_transition_warning": "DIAGNOSTIC models use actual transition features and are upper bounds, not leakage-safe origin models.",
        },
        "outputs": {
            "model_summary": str(outdir / "phasemap7c_model_summary.csv"),
            "fold_summary": str(outdir / "phasemap7c_fold_summary.csv"),
            "target_summary": str(outdir / "phasemap7c_target_summary.csv"),
            "residuals": str(outdir / "phasemap7c_residuals.csv"),
            "residual_by_group": str(outdir / "phasemap7c_residual_by_group.csv"),
            "feature_inventory": str(outdir / "phasemap7c_feature_inventory.csv"),
            "correlation_audit": str(outdir / "phasemap7c_correlation_audit.csv"),
        },
    }
    with open(outdir / "phasemap7c_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== PhaseMap-7C v2 main readout ===")
    for k, v in main.items():
        print(f"{k}: {v}")
    print(f"Saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
