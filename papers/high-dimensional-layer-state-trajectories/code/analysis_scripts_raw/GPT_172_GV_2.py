# ============================================================
# GV-2: Residual Structure Audit
#
# Purpose:
#   Continue after GV-1B.
#
# GV-1B showed:
#   real TopK/VIM trajectories are geodesic-biased,
#   but the residual component is large and meaningful.
#
# GV-2 asks:
#
#   Trajectory = Projection_to_main_geodesic_direction + Residual
#
#   Is Residual structured?
#
# Core tests:
#   1. Residual-only features -> mechanism
#   2. Residual-only features -> DeltaU_decision
#   3. Residual metrics detect critical-band / competition-band proxy
#   4. Residual layer profiles are non-flat and mechanism-specific
#   5. Cross-model residual profiles show structural isomorphism
#
# Inputs:
#   gv1b_outputs/
#       <model>_trajectory_metrics.csv
#       <model>_layer_profile.csv
#       <model>_gv1b_summary.json
#
# Run:
#   python gv2_residual_structure_audit.py
#
# Outputs:
#   gv2_outputs/
#       gv2_model_summary.csv
#       gv2_cross_model_residual_profile_corr.csv
#       gv2_overall_summary.json
#       <model>_residual_window_eval.csv
#       <model>_residual_layer_mechanism_summary.csv
#       <model>_gv2_summary.json
#
# ============================================================

import json
import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
except ModuleNotFoundError:
    accuracy_score = f1_score = roc_auc_score = r2_score = None
    GroupKFold = LogisticRegression = Ridge = StandardScaler = Pipeline = PCA = None

IN_DIR = Path("gv1b_outputs")
OUT_DIR = Path("gv2_outputs")

MODEL_KEYS = ["qwen", "llama", "gemma"]

def parse_args():
    parser = argparse.ArgumentParser(description="GV-2 residual structure audit")
    parser.add_argument("--in-dir", type=Path, default=IN_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--models", default=",".join(MODEL_KEYS), help="Comma-separated model keys")
    parser.add_argument("--check-inputs-only", action="store_true")
    return parser.parse_args()

def configure_from_args(args):
    global IN_DIR, OUT_DIR, MODEL_KEYS
    IN_DIR = args.in_dir
    OUT_DIR = args.out_dir
    MODEL_KEYS = [x.strip() for x in str(args.models).split(",") if x.strip()]

def dependency_status():
    return {"scikit_learn": Pipeline is not None}

def path_status(path):
    p = Path(path)
    return {
        "path": p.as_posix(),
        "exists": p.exists(),
        "is_dir": p.is_dir() if p.exists() else False,
    }

def expected_input_paths(model_key):
    return {
        "trajectory_metrics": IN_DIR / f"{model_key}_trajectory_metrics.csv",
        "layer_profile": IN_DIR / f"{model_key}_layer_profile.csv",
        "gv1b_summary": IN_DIR / f"{model_key}_gv1b_summary.json",
    }

def write_input_check_report():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "script": "GPT_172_GV_2.py",
        "audit": "GV-2 Residual Structure Audit",
        "dependencies": dependency_status(),
        "inputs": {
            "in_dir": path_status(IN_DIR),
            "models": {
                model: {name: path_status(path) for name, path in expected_input_paths(model).items()}
                for model in MODEL_KEYS
            },
        },
        "outputs": {"out_dir": path_status(OUT_DIR)},
    }
    report_path = OUT_DIR / "gv2_input_check_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"GV-2 input check report written to: {report_path}")

def ensure_runtime_dependencies():
    if Pipeline is None:
        raise RuntimeError(
            "Missing required runtime dependency for full GV-2 run: scikit-learn. "
            "Use --check-inputs-only for portability checks."
        )

# Residual-related feature columns from GV-1B trajectory_metrics.csv.
RESIDUAL_COLS = [
    "real_residual_ratio_mean",
    "real_residual_ratio_max",
    "real_curvature_sum",
    "real_curvature_mean",
    "real_max_turn",
    "real_speed_std",
    "real_speed_max",
    "real_minus_shuffle_residual",
    "real_minus_shuffle_curvature",
    "real_minus_shuffle_detour",
]

# Projection/geodesic-bias cols for comparison.
PROJECTION_COLS = [
    "real_align_mean",
    "real_align_min",
    "real_align_final",
    "real_proj_pos_ratio_mean",
    "real_proj_abs_ratio_mean",
    "real_projection_dominance_pos",
    "real_projection_dominance_abs",
    "real_geodesic_score",
    "real_detour_ratio",
    "real_minus_shuffle_align",
    "real_minus_shuffle_proj",
    "real_vs_reverse_align",
]

# Mixed features.
MIXED_COLS = RESIDUAL_COLS + PROJECTION_COLS

MECH_MAP3 = {
    "stable": 0,
    "stable_shift": 0,
    "competition": 1,
    "closure": 2,
}

MECH_NAMES = {
    0: "stable_like",
    1: "competition",
    2: "closure",
}

def corr_safe(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def safe_auc(y, score):
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y, score))
    except Exception:
        return np.nan

def ridge_cv(X, y, groups):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    if X.shape[1] == 0 or len(np.unique(groups)) < 3:
        return np.nan, np.nan

    pred = np.zeros_like(y)
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))

    for tr, te in gkf.split(X, y, groups):
        reg = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0)),
        ])
        reg.fit(X[tr], y[tr])
        pred[te] = reg.predict(X[te])

    return float(r2_score(y, pred)), corr_safe(y, pred)

def logistic_cv(X, y, groups):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)

    if X.shape[1] == 0 or len(np.unique(y)) < 2 or len(np.unique(groups)) < 3:
        return np.nan, np.nan

    pred = np.zeros_like(y)
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))

    for tr, te in gkf.split(X, y, groups):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])

    return float(accuracy_score(y, pred)), float(f1_score(y, pred, average="macro", zero_division=0))

def interpolate_profile(x, y, n_points=50):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if len(x) == 0:
        return np.zeros(n_points)
    if len(x) == 1:
        return np.full(n_points, y[0])
    xi = np.linspace(0, 1, n_points)
    return np.interp(xi, x, y)

def load_inputs(model_key):
    metrics_path = IN_DIR / f"{model_key}_trajectory_metrics.csv"
    layer_path = IN_DIR / f"{model_key}_layer_profile.csv"
    summary_path = IN_DIR / f"{model_key}_gv1b_summary.json"

    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    if not layer_path.exists():
        raise FileNotFoundError(layer_path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    metrics = pd.read_csv(metrics_path)
    layer = pd.read_csv(layer_path)
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    return metrics, layer, summary

def sanitize_features(df, cols):
    available = [c for c in cols if c in df.columns]
    if not available:
        raise RuntimeError(f"No feature columns found. Need one of: {cols}")
    X = df[available].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    X = np.clip(X, -1e6, 1e6)
    return X, available

def evaluate_feature_group(sub, cols, y_delta, y_mech, y_band, groups, group_name):
    X, used_cols = sanitize_features(sub, cols)

    r2, corr = ridge_cv(X, y_delta, groups)
    acc, f1 = logistic_cv(X, y_mech, groups)

    # Single scalar diagnostics.
    scalar_rows = []
    for c in used_cols:
        x = np.nan_to_num(sub[c].values.astype(float), nan=0.0)
        scalar_rows.append({
            "feature": c,
            "corr_deltaU": corr_safe(x, y_delta),
            "auc_band_high": safe_auc(y_band, x),
            "auc_band_low": safe_auc(y_band, -x),
            "auc_closure_high": safe_auc((y_mech == 2).astype(int), x),
            "auc_competition_high": safe_auc((y_mech == 1).astype(int), x),
        })

    # PCA geometry of feature group.
    if X.shape[1] >= 2 and X.shape[0] >= 3:
        Xs = StandardScaler().fit_transform(X)
        pca = PCA(n_components=min(3, Xs.shape[1]))
        pcs = pca.fit_transform(Xs)
        pc1 = pcs[:, 0]
        pc1_corr_delta = corr_safe(pc1, y_delta)
        pc1_auc_band = max(safe_auc(y_band, pc1), safe_auc(y_band, -pc1))
        pc_var = pca.explained_variance_ratio_.tolist()
    else:
        pc1_corr_delta = np.nan
        pc1_auc_band = np.nan
        pc_var = []

    return {
        "feature_group": group_name,
        "n_features": len(used_cols),
        "features": used_cols,
        "ridge_deltaU_r2": r2,
        "ridge_deltaU_corr": corr,
        "mechanism_acc": acc,
        "mechanism_macro_f1": f1,
        "pc1_corr_deltaU": pc1_corr_delta,
        "pc1_auc_band": pc1_auc_band,
        "pc_variance": pc_var,
        "scalar_rows": scalar_rows,
    }

def process_model(model_key):
    metrics, layer, gv1b_summary = load_inputs(model_key)

    # Prepare labels.
    metrics = metrics.copy()
    metrics["mechanism_id"] = metrics["mechanism"].map(MECH_MAP3).astype(int)

    y_delta_all = metrics["DeltaU_decision"].values.astype(float)
    y_mech_all = metrics["mechanism_id"].values.astype(int)
    y_band_all = metrics["inside_band_proxy"].values.astype(int)
    groups_all = metrics["graph_id"].values.astype(int)

    rows = []
    scalar_rows_all = []

    for (k, window), sub in metrics.groupby(["k", "window"]):
        sub = sub.reset_index(drop=True)
        y_delta = sub["DeltaU_decision"].values.astype(float)
        y_mech = sub["mechanism_id"].values.astype(int)
        y_band = sub["inside_band_proxy"].values.astype(int)
        groups = sub["graph_id"].values.astype(int)

        for group_name, cols in [
            ("residual_only", RESIDUAL_COLS),
            ("projection_only", PROJECTION_COLS),
            ("mixed_projection_residual", MIXED_COLS),
        ]:
            result = evaluate_feature_group(sub, cols, y_delta, y_mech, y_band, groups, group_name)

            row = {
                "model_key": model_key,
                "k": int(k),
                "window": window,
                "feature_group": group_name,
                "n": int(len(sub)),
                "n_features": result["n_features"],
                "features": json.dumps(result["features"], ensure_ascii=False),
                "ridge_deltaU_r2": result["ridge_deltaU_r2"],
                "ridge_deltaU_corr": result["ridge_deltaU_corr"],
                "mechanism_acc": result["mechanism_acc"],
                "mechanism_macro_f1": result["mechanism_macro_f1"],
                "pc1_corr_deltaU": result["pc1_corr_deltaU"],
                "pc1_auc_band": result["pc1_auc_band"],
                "pc_variance": json.dumps(result["pc_variance"]),
            }
            rows.append(row)

            for sr in result["scalar_rows"]:
                scalar_rows_all.append({
                    "model_key": model_key,
                    "k": int(k),
                    "window": window,
                    "feature_group": group_name,
                    **sr,
                })

    eval_df = pd.DataFrame(rows)
    scalar_df = pd.DataFrame(scalar_rows_all)

    eval_df.to_csv(OUT_DIR / f"{model_key}_residual_window_eval.csv", index=False, encoding="utf-8-sig")
    scalar_df.to_csv(OUT_DIR / f"{model_key}_residual_scalar_feature_eval.csv", index=False, encoding="utf-8-sig")

    # Layer residual mechanism summary.
    # GV1B layer_profile has mechanism-level means per transition.
    layer_summary = []
    for (k, window), sub in layer.groupby(["k", "window"]):
        # Pivot each metric by mechanism and compute separations.
        for metric in [
            "local_align_mean",
            "proj_pos_ratio_mean",
            "residual_ratio_mean",
            "step_norm_mean",
            "step_cos_dist_mean",
        ]:
            if metric not in sub.columns:
                continue
            pivot = sub.pivot_table(
                index="transition_layer",
                columns="mechanism",
                values=metric,
                aggfunc="mean",
            ).reset_index()

            cols = pivot.columns.tolist()
            stable_col = "stable_like" if "stable_like" in cols else None
            comp_col = "competition" if "competition" in cols else None
            closure_col = "closure" if "closure" in cols else None

            if stable_col and comp_col:
                pivot["competition_minus_stable"] = pivot[comp_col] - pivot[stable_col]
            else:
                pivot["competition_minus_stable"] = np.nan

            if stable_col and closure_col:
                pivot["closure_minus_stable"] = pivot[closure_col] - pivot[stable_col]
            else:
                pivot["closure_minus_stable"] = np.nan

            if comp_col and closure_col:
                pivot["closure_minus_competition"] = pivot[closure_col] - pivot[comp_col]
            else:
                pivot["closure_minus_competition"] = np.nan

            for _, r in pivot.iterrows():
                layer_summary.append({
                    "model_key": model_key,
                    "k": int(k),
                    "window": window,
                    "transition_layer": int(r["transition_layer"]),
                    "metric": metric,
                    "stable_like": float(r[stable_col]) if stable_col else np.nan,
                    "competition": float(r[comp_col]) if comp_col else np.nan,
                    "closure": float(r[closure_col]) if closure_col else np.nan,
                    "competition_minus_stable": float(r["competition_minus_stable"]),
                    "closure_minus_stable": float(r["closure_minus_stable"]),
                    "closure_minus_competition": float(r["closure_minus_competition"]),
                })

    layer_summary_df = pd.DataFrame(layer_summary)
    layer_summary_df.to_csv(OUT_DIR / f"{model_key}_residual_layer_mechanism_summary.csv", index=False, encoding="utf-8-sig")

    # Best rows.
    residual_df = eval_df[eval_df["feature_group"] == "residual_only"].copy()
    projection_df = eval_df[eval_df["feature_group"] == "projection_only"].copy()
    mixed_df = eval_df[eval_df["feature_group"] == "mixed_projection_residual"].copy()

    best_resid_delta = residual_df.sort_values("ridge_deltaU_corr", ascending=False).iloc[0].to_dict()
    best_resid_mech = residual_df.sort_values("mechanism_macro_f1", ascending=False).iloc[0].to_dict()
    best_proj_delta = projection_df.sort_values("ridge_deltaU_corr", ascending=False).iloc[0].to_dict()
    best_mixed_delta = mixed_df.sort_values("ridge_deltaU_corr", ascending=False).iloc[0].to_dict()
    best_mixed_mech = mixed_df.sort_values("mechanism_macro_f1", ascending=False).iloc[0].to_dict()

    # Residual signature around critical band:
    # choose best residual window, inspect band AUC from scalar residual features.
    scalar_resid = scalar_df[scalar_df["feature_group"] == "residual_only"].copy()
    best_band_high = scalar_resid.sort_values("auc_band_high", ascending=False).head(1).to_dict(orient="records")
    best_band_low = scalar_resid.sort_values("auc_band_low", ascending=False).head(1).to_dict(orient="records")

    summary = {
        "model_key": model_key,
        "gv1b_deltaU_pc1_variance": gv1b_summary.get("deltaU_pc1_variance", None),
        "best_residual_deltaU": best_resid_delta,
        "best_residual_mechanism": best_resid_mech,
        "best_projection_deltaU": best_proj_delta,
        "best_mixed_deltaU": best_mixed_delta,
        "best_mixed_mechanism": best_mixed_mech,
        "best_band_by_residual_high": best_band_high,
        "best_band_by_residual_low": best_band_low,
        "residual_deltaU_beats_projection": bool(best_resid_delta["ridge_deltaU_corr"] > best_proj_delta["ridge_deltaU_corr"]),
        "mixed_deltaU_beats_projection": bool(best_mixed_delta["ridge_deltaU_corr"] > best_proj_delta["ridge_deltaU_corr"]),
        "pass_residual_deltaU_corr_gt_0p3": bool(best_resid_delta["ridge_deltaU_corr"] > 0.3),
        "pass_residual_mechanism_f1_gt_0p6": bool(best_resid_mech["mechanism_macro_f1"] > 0.6),
    }

    with open(OUT_DIR / f"{model_key}_gv2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary, layer_summary_df

def cross_model_profiles(layer_summaries):
    profiles = {}

    for mk, df in layer_summaries.items():
        # Use full/k=500 if present, otherwise all full.
        sub = df[(df["window"] == "full") & (df["k"] == 500) & (df["metric"] == "residual_ratio_mean")]
        if len(sub) == 0:
            sub = df[(df["window"] == "full") & (df["metric"] == "residual_ratio_mean")]
        if len(sub) == 0:
            continue

        # Average over duplicate rows if any.
        prof = sub.groupby("transition_layer")[[
            "stable_like",
            "competition",
            "closure",
            "competition_minus_stable",
            "closure_minus_stable",
        ]].mean().reset_index()

        max_layer = max(1, prof["transition_layer"].max())
        x = prof["transition_layer"].values / max_layer

        profiles[mk] = {
            "stable_like": interpolate_profile(x, prof["stable_like"].values),
            "competition": interpolate_profile(x, prof["competition"].values),
            "closure": interpolate_profile(x, prof["closure"].values),
            "competition_minus_stable": interpolate_profile(x, prof["competition_minus_stable"].values),
            "closure_minus_stable": interpolate_profile(x, prof["closure_minus_stable"].values),
        }

    rows = []
    for a, b in combinations(profiles.keys(), 2):
        for typ in ["stable_like", "competition", "closure", "competition_minus_stable", "closure_minus_stable"]:
            rows.append({
                "profile_type": typ,
                "model_a": a,
                "model_b": b,
                "pearson_profile_corr": corr_safe(profiles[a][typ], profiles[b][typ]),
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "gv2_cross_model_residual_profile_corr.csv", index=False, encoding="utf-8-sig")
    return out

def main():
    args = parse_args()
    configure_from_args(args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.check_inputs_only:
        write_input_check_report()
        return

    ensure_runtime_dependencies()

    summaries = []
    layer_summaries = {}

    for mk in MODEL_KEYS:
        summary, layer_summary_df = process_model(mk)
        summaries.append(summary)
        layer_summaries[mk] = layer_summary_df

    flat = []
    for s in summaries:
        brd = s["best_residual_deltaU"]
        brm = s["best_residual_mechanism"]
        bpd = s["best_projection_deltaU"]
        bmd = s["best_mixed_deltaU"]
        bmm = s["best_mixed_mechanism"]

        flat.append({
            "model_key": s["model_key"],
            "gv1b_deltaU_pc1_variance": s["gv1b_deltaU_pc1_variance"],
            "best_resid_deltaU_window": brd["window"],
            "best_resid_deltaU_k": brd["k"],
            "best_resid_deltaU_corr": brd["ridge_deltaU_corr"],
            "best_resid_deltaU_r2": brd["ridge_deltaU_r2"],
            "best_resid_mechanism_window": brm["window"],
            "best_resid_mechanism_k": brm["k"],
            "best_resid_mechanism_f1": brm["mechanism_macro_f1"],
            "best_projection_deltaU_corr": bpd["ridge_deltaU_corr"],
            "best_mixed_deltaU_corr": bmd["ridge_deltaU_corr"],
            "best_mixed_mechanism_f1": bmm["mechanism_macro_f1"],
            "residual_deltaU_beats_projection": s["residual_deltaU_beats_projection"],
            "mixed_deltaU_beats_projection": s["mixed_deltaU_beats_projection"],
            "pass_residual_deltaU_corr_gt_0p3": s["pass_residual_deltaU_corr_gt_0p3"],
            "pass_residual_mechanism_f1_gt_0p6": s["pass_residual_mechanism_f1_gt_0p6"],
        })

    model_summary = pd.DataFrame(flat)
    model_summary.to_csv(OUT_DIR / "gv2_model_summary.csv", index=False, encoding="utf-8-sig")

    cross = cross_model_profiles(layer_summaries)

    overall = {
        "audit": "GV-2 Residual Structure Audit",
        "definition": (
            "Tests whether the residual component after projection to the dominant geodesic direction "
            "is structured and predictive of DeltaU, mechanism, and critical-band behavior."
        ),
        "models": summaries,
        "cross_model_residual_profile_corr": cross.to_dict(orient="records"),
    }

    with open(OUT_DIR / "gv2_overall_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    print("\nGV-2 complete.")
    print(model_summary)
    print(cross)

if __name__ == "__main__":
    main()
