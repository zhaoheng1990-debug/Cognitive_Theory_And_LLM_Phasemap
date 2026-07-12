# ============================================================
# GV-3: O-Residual Coupling Audit
#
# Goal:
#   Test the hypothesis:
#
#       Residual is not independent noise.
#       Residual is the orthogonal / non-geodesic component
#       of the TopK/VIM layerwise operator O_l.
#
# Conceptual decomposition:
#
#       O_l = O_l_parallel + O_l_perp
#
#       O_l_parallel  -> geodesic-aligned dominance growth
#       O_l_perp      -> correction / rotation / competition / boundary reconfiguration
#
# Inputs:
#   gv1b_outputs/
#       <model>_layer_profile.csv
#       <model>_trajectory_metrics.csv
#       <model>_R_dataset.csv
#
#   gv2_outputs/
#       <model>_residual_layer_mechanism_summary.csv
#       <model>_residual_window_eval.csv
#
#   cm5c_outputs/   optional but strongly preferred
#       <model>_predecision_layer_corr.csv
#       <model>_nonR_window_summary.csv
#
# Tests:
#   1. Layerwise coupling:
#       O_topo single-transition strength <-> GV residual profile
#
#   2. Mechanism coupling:
#       O_topo mechanism F1 / DeltaU corr <-> residual mechanism separation
#
#   3. Window-level coupling:
#       CM5C non-R TopK O-flow performance <-> GV2 residual-only performance
#
#   4. Parallel/perp proxy:
#       O_parallel proxy = center_shift / geodesic alignment
#       O_perp proxy     = turnover / curvature / residual ratio
#
#   5. Cross-model consistency.
#
# Run:
#   python gv3_o_residual_coupling_audit.py
#
# Outputs:
#   gv3_outputs/
#       gv3_model_summary.csv
#       gv3_cross_model_summary.csv
#       gv3_overall_summary.json
#       <model>_gv3_summary.json
#       <model>_layer_coupling.csv
#       <model>_window_coupling.csv
#
# ============================================================

import json
import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import r2_score, accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
except ModuleNotFoundError:
    r2_score = accuracy_score = f1_score = roc_auc_score = None
    GroupKFold = Ridge = LogisticRegression = Pipeline = StandardScaler = PCA = None

GV1B_DIR = Path("gv1b_outputs")
GV2_DIR = Path("gv2_outputs")
CM5C_DIR = Path("cm5c_outputs")
OUT_DIR = Path("gv3_outputs")

MODEL_KEYS = ["qwen", "llama", "gemma"]

def parse_args():
    parser = argparse.ArgumentParser(description="GV-3 O-residual coupling audit")
    parser.add_argument("--gv1b-dir", type=Path, default=GV1B_DIR)
    parser.add_argument("--gv2-dir", type=Path, default=GV2_DIR)
    parser.add_argument("--cm5c-dir", type=Path, default=CM5C_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--models", default=",".join(MODEL_KEYS), help="Comma-separated model keys")
    parser.add_argument("--check-inputs-only", action="store_true")
    return parser.parse_args()

def configure_from_args(args):
    global GV1B_DIR, GV2_DIR, CM5C_DIR, OUT_DIR, MODEL_KEYS
    GV1B_DIR = args.gv1b_dir
    GV2_DIR = args.gv2_dir
    CM5C_DIR = args.cm5c_dir
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
        "required": {
            "gv1b_layer_profile": GV1B_DIR / f"{model_key}_layer_profile.csv",
            "gv1b_trajectory_metrics": GV1B_DIR / f"{model_key}_trajectory_metrics.csv",
            "gv1b_summary": GV1B_DIR / f"{model_key}_gv1b_summary.json",
            "gv2_residual_layer_mechanism_summary": GV2_DIR / f"{model_key}_residual_layer_mechanism_summary.csv",
            "gv2_residual_window_eval": GV2_DIR / f"{model_key}_residual_window_eval.csv",
            "gv2_summary": GV2_DIR / f"{model_key}_gv2_summary.json",
        },
        "optional": {
            "cm5c_predecision_layer_corr": CM5C_DIR / f"{model_key}_predecision_layer_corr.csv",
            "cm5c_nonR_window_summary": CM5C_DIR / f"{model_key}_nonR_window_summary.csv",
        },
    }

def write_input_check_report():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "script": "GPT_173_GV_3.py",
        "audit": "GV-3 O-Residual Coupling Audit",
        "dependencies": dependency_status(),
        "inputs": {
            "gv1b_dir": path_status(GV1B_DIR),
            "gv2_dir": path_status(GV2_DIR),
            "cm5c_dir_optional": path_status(CM5C_DIR),
            "models": {
                model: {
                    group: {name: path_status(path) for name, path in paths.items()}
                    for group, paths in expected_input_paths(model).items()
                }
                for model in MODEL_KEYS
            },
        },
        "outputs": {"out_dir": path_status(OUT_DIR)},
    }
    report_path = OUT_DIR / "gv3_input_check_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"GV-3 input check report written to: {report_path}")

def ensure_runtime_dependencies():
    if Pipeline is None:
        raise RuntimeError(
            "Missing required runtime dependency for full GV-3 run: scikit-learn. "
            "Use --check-inputs-only for portability checks."
        )

MECH_MAP3 = {
    "stable": 0,
    "stable_shift": 0,
    "competition": 1,
    "closure": 2,
}

def corr_safe(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
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

def sanitize(X):
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(X, -1e6, 1e6)

def interp_by_frac(frac, values, n=50):
    frac = np.asarray(frac, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(frac) == 0:
        return np.zeros(n)
    order = np.argsort(frac)
    frac = frac[order]
    values = values[order]
    xi = np.linspace(0, 1, n)
    if len(frac) == 1:
        return np.full(n, values[0])
    return np.interp(xi, frac, values)

def load_required(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def load_json(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def best_available_cm5c_layer(model_key):
    path = CM5C_DIR / f"{model_key}_predecision_layer_corr.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

def best_available_cm5c_window(model_key):
    path = CM5C_DIR / f"{model_key}_nonR_window_summary.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

def process_model(model_key):
    print(f"\n========== {model_key} ==========")

    gv1_layer = load_required(GV1B_DIR / f"{model_key}_layer_profile.csv")
    gv1_metrics = load_required(GV1B_DIR / f"{model_key}_trajectory_metrics.csv")
    gv2_layer = load_required(GV2_DIR / f"{model_key}_residual_layer_mechanism_summary.csv")
    gv2_eval = load_required(GV2_DIR / f"{model_key}_residual_window_eval.csv")
    gv2_summary = load_json(GV2_DIR / f"{model_key}_gv2_summary.json")
    gv1_summary = load_json(GV1B_DIR / f"{model_key}_gv1b_summary.json")

    cm5c_layer = best_available_cm5c_layer(model_key)
    cm5c_window = best_available_cm5c_window(model_key)

    # ------------------------------------------------------------
    # 1. Layerwise coupling: GV1B residual profiles vs CM5C O_topo profiles
    # ------------------------------------------------------------

    layer_rows = []

    # GV1B layer profile has mechanism-level rows.
    # Collapse to all-mechanism mean per k/window/layer.
    gv1_mean = gv1_layer.groupby(["k", "window", "transition_layer", "layer_frac"])[
        ["local_align_mean", "proj_pos_ratio_mean", "residual_ratio_mean", "step_norm_mean", "step_cos_dist_mean"]
    ].mean().reset_index()

    # Mechanism separations from GV2 residual layer summary.
    gv2_resid = gv2_layer[gv2_layer["metric"] == "residual_ratio_mean"].copy()
    gv2_resid = gv2_resid.rename(columns={
        "competition_minus_stable": "resid_comp_minus_stable",
        "closure_minus_stable": "resid_closure_minus_stable",
        "closure_minus_competition": "resid_closure_minus_comp",
    })

    # Merge GV1 and GV2 residual layer summaries.
    merged_gv = pd.merge(
        gv1_mean,
        gv2_resid[[
            "k", "window", "transition_layer",
            "resid_comp_minus_stable",
            "resid_closure_minus_stable",
            "resid_closure_minus_comp"
        ]],
        on=["k", "window", "transition_layer"],
        how="left"
    )

    # If CM5C layer info is available, merge by closest k/layer.
    if cm5c_layer is not None:
        cm5 = cm5c_layer.copy()
        # Columns expected:
        # model_key,k,transition_layer,transition,layer_frac,ridge_deltaU_r2,ridge_deltaU_corr,mechanism_acc,mechanism_macro_f1
        cm5_cols = [
            c for c in [
                "k", "transition_layer", "ridge_deltaU_corr", "ridge_deltaU_r2",
                "mechanism_acc", "mechanism_macro_f1"
            ] if c in cm5.columns
        ]
        cm5 = cm5[cm5_cols].copy()
        merged = pd.merge(
            merged_gv,
            cm5,
            on=["k", "transition_layer"],
            how="left",
            suffixes=("", "_cm5c")
        )
    else:
        merged = merged_gv.copy()
        for c in ["ridge_deltaU_corr", "ridge_deltaU_r2", "mechanism_acc", "mechanism_macro_f1"]:
            merged[c] = np.nan

    # Derived parallel/perp proxies:
    # parallel proxy: alignment and positive projection
    # perp proxy: residual, step_norm, step_cos_dist, curvature-like residual separations.
    merged["O_parallel_proxy"] = (
        merged["local_align_mean"].fillna(0)
        + merged["proj_pos_ratio_mean"].fillna(0)
    ) / 2.0

    merged["O_perp_proxy"] = (
        merged["residual_ratio_mean"].fillna(0)
        + merged["step_cos_dist_mean"].fillna(0)
        + merged["step_norm_mean"].fillna(0)
    ) / 3.0

    merged["residual_mech_separation_abs"] = (
        merged["resid_comp_minus_stable"].abs().fillna(0)
        + merged["resid_closure_minus_stable"].abs().fillna(0)
        + merged["resid_closure_minus_comp"].abs().fillna(0)
    ) / 3.0

    # Layer coupling summary by k/window.
    for (k, win), sub in merged.groupby(["k", "window"]):
        row = {
            "model_key": model_key,
            "k": int(k),
            "window": win,
            "n_layers": int(len(sub)),
            "corr_O_perp_with_residual": corr_safe(sub["O_perp_proxy"], sub["residual_ratio_mean"]),
            "corr_O_parallel_with_align": corr_safe(sub["O_parallel_proxy"], sub["local_align_mean"]),
            "corr_O_perp_with_mech_sep": corr_safe(sub["O_perp_proxy"], sub["residual_mech_separation_abs"]),
            "corr_residual_with_cm5c_deltaU_corr": corr_safe(sub["residual_ratio_mean"], sub["ridge_deltaU_corr"]),
            "corr_residual_mechsep_with_cm5c_mech_f1": corr_safe(sub["residual_mech_separation_abs"], sub["mechanism_macro_f1"]),
            "corr_parallel_with_cm5c_deltaU_corr": corr_safe(sub["O_parallel_proxy"], sub["ridge_deltaU_corr"]),
            "corr_perp_with_cm5c_mech_f1": corr_safe(sub["O_perp_proxy"], sub["mechanism_macro_f1"]),
        }
        layer_rows.append(row)

    layer_coupling_df = pd.DataFrame(layer_rows)
    merged.to_csv(OUT_DIR / f"{model_key}_layer_coupling_raw.csv", index=False, encoding="utf-8-sig")
    layer_coupling_df.to_csv(OUT_DIR / f"{model_key}_layer_coupling.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # 2. Window-level coupling: CM5C window O-flow vs GV2 residual performance
    # ------------------------------------------------------------

    window_rows = []

    gv2_resid_win = gv2_eval[gv2_eval["feature_group"] == "residual_only"].copy()
    gv2_proj_win = gv2_eval[gv2_eval["feature_group"] == "projection_only"].copy()
    gv2_mix_win = gv2_eval[gv2_eval["feature_group"] == "mixed_projection_residual"].copy()

    # Normalize naming:
    # GV2 windows and CM5C windows differ, but common keywords allow loose matching.
    def canonical_window(w):
        w = str(w)
        if "predecision" in w:
            return "predecision"
        if "early_half" in w or "early" in w:
            return "early"
        if "mid" in w:
            return "mid"
        if "decision" in w:
            return "decision"
        if "cm4d_topology" in w or "topology" in w:
            return "topology"
        if "cm4d_mechanism" in w or "mechanism" in w:
            return "mechanism"
        if "cm4d_downstream" in w or "downstream" in w:
            return "downstream"
        if "overall" in w:
            return "overall"
        if "full" in w or "all" in w:
            return "full"
        return w

    for df_ in [gv2_resid_win, gv2_proj_win, gv2_mix_win]:
        df_["canon_window"] = df_["window"].apply(canonical_window)

    if cm5c_window is not None:
        cmw = cm5c_window.copy()
        cmw["canon_window"] = cmw["window"].apply(canonical_window)

        # Merge residual windows to cm5c windows by k + canonical window where possible.
        for group_name, gdf in [
            ("residual_only", gv2_resid_win),
            ("projection_only", gv2_proj_win),
            ("mixed", gv2_mix_win),
        ]:
            m = pd.merge(
                gdf,
                cmw,
                on=["k", "canon_window"],
                how="inner",
                suffixes=("_gv2", "_cm5c")
            )

            for _, r in m.iterrows():
                window_rows.append({
                    "model_key": model_key,
                    "feature_group": group_name,
                    "k": int(r["k"]),
                    "canon_window": r["canon_window"],
                    "gv2_window": r["window_gv2"],
                    "cm5c_window": r["window_cm5c"],
                    "gv2_deltaU_corr": float(r.get("ridge_deltaU_corr_gv2", np.nan)),
                    "gv2_mech_f1": float(r.get("mechanism_macro_f1_gv2", np.nan)),
                    "cm5c_deltaU_corr": float(r.get("ridge_deltaU_corr_cm5c", np.nan)),
                    "cm5c_mech_f1": float(r.get("mechanism_macro_f1_cm5c", np.nan)),
                    "cm5c_center_sum_corr_deltaU": float(r.get("center_sum_corr_deltaU", np.nan)),
                    "cm5c_jaccard_sum_corr_deltaU": float(r.get("jaccard_sum_corr_deltaU", np.nan)),
                    "cm5c_entropy_abs_sum_corr_deltaU": float(r.get("entropy_abs_sum_corr_deltaU", np.nan)),
                })

    window_coupling_df = pd.DataFrame(window_rows)
    if len(window_coupling_df) > 0:
        window_coupling_df.to_csv(OUT_DIR / f"{model_key}_window_coupling.csv", index=False, encoding="utf-8-sig")
    else:
        # Still create empty file
        window_coupling_df.to_csv(OUT_DIR / f"{model_key}_window_coupling.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # 3. Sample-level coupling: use GV1B trajectory metrics to predict residual-only
    #    performance proxies and mechanism. This is not O-labeled but tests
    #    whether parallel/perp decomposition improves mechanism representation.
    # ------------------------------------------------------------

    # Use GV1B trajectory metrics.
    traj = gv1_metrics.copy()
    traj["mechanism_id"] = traj["mechanism"].map(MECH_MAP3).astype(int)

    sample_rows = []

    for (k, win), sub in traj.groupby(["k", "window"]):
        sub = sub.reset_index(drop=True)
        y_delta = sub["DeltaU_decision"].values.astype(float)
        y_mech = sub["mechanism_id"].values.astype(int)
        groups = sub["graph_id"].values.astype(int)

        parallel_cols = [
            c for c in [
                "real_align_mean",
                "real_align_min",
                "real_align_final",
                "real_proj_pos_ratio_mean",
                "real_projection_dominance_pos",
                "real_geodesic_score",
                "real_minus_shuffle_align",
                "real_minus_shuffle_proj",
            ] if c in sub.columns
        ]

        perp_cols = [
            c for c in [
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
            ] if c in sub.columns
        ]

        both_cols = parallel_cols + perp_cols

        for name, cols in [
            ("parallel_proxy", parallel_cols),
            ("perp_proxy", perp_cols),
            ("parallel_plus_perp", both_cols),
        ]:
            X = sub[cols].values.astype(np.float32)
            X = sanitize(X)

            r2, cc = ridge_cv(X, y_delta, groups)
            acc, f1 = logistic_cv(X, y_mech, groups)

            sample_rows.append({
                "model_key": model_key,
                "k": int(k),
                "window": win,
                "feature_group": name,
                "n_features": len(cols),
                "ridge_deltaU_r2": r2,
                "ridge_deltaU_corr": cc,
                "mechanism_acc": acc,
                "mechanism_macro_f1": f1,
            })

    sample_coupling_df = pd.DataFrame(sample_rows)
    sample_coupling_df.to_csv(OUT_DIR / f"{model_key}_sample_parallel_perp_eval.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------

    # Best coupling measures.
    best_layer_perp_mech = layer_coupling_df.sort_values("corr_O_perp_with_mech_sep", ascending=False).head(1).to_dict(orient="records")
    best_layer_resid_cm5_delta = layer_coupling_df.sort_values("corr_residual_with_cm5c_deltaU_corr", ascending=False).head(1).to_dict(orient="records")
    best_layer_perp_cm5_mech = layer_coupling_df.sort_values("corr_perp_with_cm5c_mech_f1", ascending=False).head(1).to_dict(orient="records")

    if len(window_coupling_df) > 0:
        # Correlation across matched windows.
        wcorr_resid_delta = corr_safe(window_coupling_df["gv2_deltaU_corr"], window_coupling_df["cm5c_deltaU_corr"])
        wcorr_resid_mech = corr_safe(window_coupling_df["gv2_mech_f1"], window_coupling_df["cm5c_mech_f1"])
    else:
        wcorr_resid_delta = np.nan
        wcorr_resid_mech = np.nan

    best_sample_parallel = sample_coupling_df[sample_coupling_df["feature_group"] == "parallel_proxy"].sort_values("ridge_deltaU_corr", ascending=False).iloc[0].to_dict()
    best_sample_perp = sample_coupling_df[sample_coupling_df["feature_group"] == "perp_proxy"].sort_values("ridge_deltaU_corr", ascending=False).iloc[0].to_dict()
    best_sample_both = sample_coupling_df[sample_coupling_df["feature_group"] == "parallel_plus_perp"].sort_values("ridge_deltaU_corr", ascending=False).iloc[0].to_dict()

    summary = {
        "model_key": model_key,
        "gv2_summary_key_results": {
            "best_residual_deltaU_corr": gv2_summary.get("best_residual_deltaU", {}).get("ridge_deltaU_corr", None),
            "best_residual_mechanism_f1": gv2_summary.get("best_residual_mechanism", {}).get("mechanism_macro_f1", None),
            "residual_deltaU_beats_projection": gv2_summary.get("residual_deltaU_beats_projection", None),
        },
        "best_layer_perp_with_residual_mechanism_separation": best_layer_perp_mech,
        "best_layer_residual_with_cm5c_deltaU_corr": best_layer_resid_cm5_delta,
        "best_layer_perp_with_cm5c_mechanism_f1": best_layer_perp_cm5_mech,
        "window_corr_gv2_residual_deltaU_vs_cm5c_O_deltaU": wcorr_resid_delta,
        "window_corr_gv2_residual_mech_vs_cm5c_O_mech": wcorr_resid_mech,
        "best_sample_parallel_deltaU": best_sample_parallel,
        "best_sample_perp_deltaU": best_sample_perp,
        "best_sample_parallel_plus_perp_deltaU": best_sample_both,
        "perp_beats_parallel_deltaU": bool(best_sample_perp["ridge_deltaU_corr"] > best_sample_parallel["ridge_deltaU_corr"]),
        "parallel_plus_perp_beats_both": bool(
            best_sample_both["ridge_deltaU_corr"] > max(
                best_sample_perp["ridge_deltaU_corr"],
                best_sample_parallel["ridge_deltaU_corr"]
            )
        ),
    }

    with open(OUT_DIR / f"{model_key}_gv3_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary, merged, sample_coupling_df

def cross_model_summary(model_raws):
    # Compare layerwise O_perp_proxy and residual_ratio profiles across models.
    profiles = {}
    for mk, merged in model_raws.items():
        sub = merged[(merged["window"] == "full") & (merged["k"] == 500)].copy()
        if len(sub) == 0:
            sub = merged[merged["window"] == "full"].copy()
        if len(sub) == 0:
            sub = merged.copy()

        # normalize layer_frac if available
        if "layer_frac" in sub.columns:
            frac = sub["layer_frac"].values
        else:
            max_l = max(1, sub["transition_layer"].max())
            frac = sub["transition_layer"].values / max_l

        prof = sub.groupby(frac).mean(numeric_only=True).reset_index()
        # The groupby with array may create weird column name; easier:
        temp = pd.DataFrame({
            "frac": frac,
            "O_perp_proxy": sub["O_perp_proxy"].values,
            "O_parallel_proxy": sub["O_parallel_proxy"].values,
            "residual_ratio_mean": sub["residual_ratio_mean"].values,
            "residual_mech_separation_abs": sub["residual_mech_separation_abs"].values,
        })
        temp = temp.groupby("frac").mean().reset_index().sort_values("frac")

        profiles[mk] = {
            "O_perp_proxy": interp_by_frac(temp["frac"].values, temp["O_perp_proxy"].values),
            "O_parallel_proxy": interp_by_frac(temp["frac"].values, temp["O_parallel_proxy"].values),
            "residual_ratio_mean": interp_by_frac(temp["frac"].values, temp["residual_ratio_mean"].values),
            "residual_mech_separation_abs": interp_by_frac(temp["frac"].values, temp["residual_mech_separation_abs"].values),
        }

    rows = []
    for a, b in combinations(profiles.keys(), 2):
        for typ in ["O_perp_proxy", "O_parallel_proxy", "residual_ratio_mean", "residual_mech_separation_abs"]:
            rows.append({
                "model_a": a,
                "model_b": b,
                "profile_type": typ,
                "pearson_profile_corr": corr_safe(profiles[a][typ], profiles[b][typ]),
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "gv3_cross_model_summary.csv", index=False, encoding="utf-8-sig")
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
    model_raws = {}

    for mk in MODEL_KEYS:
        summary, merged, sample_eval = process_model(mk)
        summaries.append(summary)
        model_raws[mk] = merged

    flat = []
    for s in summaries:
        bpar = s["best_sample_parallel_deltaU"]
        bperp = s["best_sample_perp_deltaU"]
        bboth = s["best_sample_parallel_plus_perp_deltaU"]

        flat.append({
            "model_key": s["model_key"],
            "gv2_best_residual_deltaU_corr": s["gv2_summary_key_results"]["best_residual_deltaU_corr"],
            "gv2_best_residual_mechanism_f1": s["gv2_summary_key_results"]["best_residual_mechanism_f1"],
            "window_corr_gv2_resid_deltaU_vs_cm5c_O_deltaU": s["window_corr_gv2_residual_deltaU_vs_cm5c_O_deltaU"],
            "window_corr_gv2_resid_mech_vs_cm5c_O_mech": s["window_corr_gv2_residual_mech_vs_cm5c_O_mech"],
            "best_parallel_deltaU_corr": bpar["ridge_deltaU_corr"],
            "best_perp_deltaU_corr": bperp["ridge_deltaU_corr"],
            "best_both_deltaU_corr": bboth["ridge_deltaU_corr"],
            "best_parallel_mech_f1": bpar["mechanism_macro_f1"],
            "best_perp_mech_f1": bperp["mechanism_macro_f1"],
            "best_both_mech_f1": bboth["mechanism_macro_f1"],
            "perp_beats_parallel_deltaU": s["perp_beats_parallel_deltaU"],
            "parallel_plus_perp_beats_both": s["parallel_plus_perp_beats_both"],
        })

    model_summary = pd.DataFrame(flat)
    model_summary.to_csv(OUT_DIR / "gv3_model_summary.csv", index=False, encoding="utf-8-sig")

    cross = cross_model_summary(model_raws)

    overall = {
        "audit": "GV-3 O-Residual Coupling Audit",
        "definition": (
            "Tests whether GV residual components are coupled with TopK/VIM layerwise operator O_topo, "
            "and whether O decomposes into geodesic-parallel and structured-perpendicular components."
        ),
        "models": summaries,
        "cross_model_summary": cross.to_dict(orient="records"),
    }

    with open(OUT_DIR / "gv3_overall_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    print("\nGV-3 complete.")
    print(model_summary)
    print(cross)

if __name__ == "__main__":
    main()
