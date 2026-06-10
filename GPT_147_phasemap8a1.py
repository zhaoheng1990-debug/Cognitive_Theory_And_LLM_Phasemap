# ============================================================
# PhaseMap-8A.1: Operator Integral Audit
#
# Goal:
#   Test whether continuous operator coordinates O_l^cont behave like
#   a differential / local connection whose layer-integral predicts:
#       DeltaU = PC1(DeltaR_{20:25})
#
# Fixed input:
#   phasemap6b1_correction_dataset.csv
#
# Output dir:
#   phasemap8a1_outputs/
# ============================================================

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# -----------------------------
# CONFIG
# -----------------------------
INPUT_CSV = r"C:\Windows\System32\phasemap6b1_outputs\phasemap6b1_correction_dataset.csv"
OUTPUT_DIR = Path("phasemap8a1_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_SPLITS = 5
N_SHUFFLES = 64
RIDGE_ALPHA = 1.0

WINDOWS = {
    "I_0_6": list(range(0, 7)),
    "I_7_19": list(range(7, 20)),
    "I_20_22": list(range(20, 23)),
    "I_23_26": list(range(23, 27)),
    "I_20_25": list(range(20, 26)),
    "I_7_26": list(range(7, 27)),
    "I_0_26": list(range(0, 27)),
}
TARGET_DELTA_R_LAYERS = list(range(20, 26))


def first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def safe_cols(df, cols):
    return [c for c in cols if c in df.columns]


def dedup(cols):
    seen = set(); out = []
    for c in cols:
        if c not in seen:
            out.append(c); seen.add(c)
    return out


def pearsonr_safe(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    a = a[m]; b = b[m]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "corr": pearsonr_safe(y_true, y_pred),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def get_cv(df):
    group_col = first_existing(df, ["sample_id", "idx", "graph_id", "prompt_id", "case_id"])
    if group_col is not None:
        groups = df[group_col].astype(str).values
        n_unique = len(np.unique(groups))
        n_splits = min(N_SPLITS, n_unique)
        if n_splits >= 2:
            return GroupKFold(n_splits=n_splits), groups, group_col
    return KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED), None, None


def cv_predict(df, feature_cols, target_col, model_kind="ridge"):
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)
    if model_kind == "ridge":
        model = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=RIDGE_ALPHA))])
    elif model_kind == "linear":
        model = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])
    else:
        raise ValueError(model_kind)
    cv, groups, group_col = get_cv(df)
    if groups is None:
        pred = cross_val_predict(model, X, y, cv=cv)
    else:
        pred = cross_val_predict(model, X, y, cv=cv, groups=groups)
    return pred, group_col


def fit_delta_u_pc1(wide, cols):
    X = wide[cols].values.astype(np.float32)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=min(len(cols), Xs.shape[1]), random_state=RANDOM_SEED)
    Z = pca.fit_transform(Xs)
    u = Z[:, 0]
    # orient so larger DeltaU roughly follows larger mean DeltaR
    if pearsonr_safe(u, X.mean(axis=1)) < 0:
        u = -u
        pca.components_[0, :] *= -1
    return u.astype(np.float32), pca, scaler


def main():
    if not Path(INPUT_CSV).exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_CSV}. Put this script in the same folder as the CSV and run again."
        )

    df = pd.read_csv(INPUT_CSV)
    df.columns = [str(c).strip() for c in df.columns]
    if "layer" not in df.columns:
        raise RuntimeError("Input must contain a 'layer' column.")
    df["layer"] = df["layer"].astype(int)

    op_cols = [c for c in df.columns if c.startswith("op_pc")]
    op_cols = sorted(op_cols, key=lambda x: int(x.replace("op_pc", "")) if x.replace("op_pc", "").isdigit() else x)
    if not op_cols:
        raise RuntimeError("No op_pc* columns found. 8A.1 requires continuous operator coordinates.")

    id_col = first_existing(df, ["sample_id", "idx", "graph_id", "prompt_id", "case_id"])
    if id_col is None:
        df["_pseudo_id"] = np.arange(len(df)) // max(1, df["layer"].nunique())
        id_col = "_pseudo_id"

    r_curr_col = first_existing(df, ["R_l", "R", "R_current", "r_l"])
    r_next_col = first_existing(df, ["R_l1", "R_next", "R_l+1", "r_l1", "r_next"])
    if r_curr_col is None and r_next_col is None:
        raise RuntimeError("No R columns found. Need R_l or R_l1/R_next to build DeltaU.")

    meta_cols = safe_cols(df, [id_col, "condition", "phase", "phase_target", "heuristic_operator", "operator_cluster"])
    agg_cols = op_cols[:]
    if r_curr_col: agg_cols.append(r_curr_col)
    if r_next_col: agg_cols.append(r_next_col)

    num = df.groupby([id_col, "layer"], as_index=False)[agg_cols].mean()
    meta = df.groupby(id_col, as_index=False).first()[meta_cols]
    wide = meta.copy()
    available_layers = sorted(num["layer"].unique().tolist())

    for c in op_cols:
        piv = num.pivot(index=id_col, columns="layer", values=c)
        piv.columns = [f"{c}_L{int(l)}" for l in piv.columns]
        wide = wide.merge(piv.reset_index(), on=id_col, how="left")

    if r_curr_col:
        piv = num.pivot(index=id_col, columns="layer", values=r_curr_col)
        piv.columns = [f"R_L{int(l)}" for l in piv.columns]
        wide = wide.merge(piv.reset_index(), on=id_col, how="left")
    if r_next_col:
        piv = num.pivot(index=id_col, columns="layer", values=r_next_col)
        piv.columns = [f"Rnext_from_L{int(l)}" for l in piv.columns]
        wide = wide.merge(piv.reset_index(), on=id_col, how="left")

    # Fill R_L20..25 from previous layer's next target when needed.
    for l in TARGET_DELTA_R_LAYERS:
        if f"R_L{l}" not in wide.columns and f"Rnext_from_L{l-1}" in wide.columns:
            wide[f"R_L{l}"] = wide[f"Rnext_from_L{l-1}"]

    r_target_cols = [f"R_L{l}" for l in TARGET_DELTA_R_LAYERS if f"R_L{l}" in wide.columns]
    if len(r_target_cols) < 4:
        raise RuntimeError(f"Too few R_L20..25 columns available. Found: {r_target_cols}")

    wide = wide.dropna(subset=r_target_cols).reset_index(drop=True)
    base_col = "R_L20" if "R_L20" in wide.columns else r_target_cols[0]
    dlt_cols = []
    for c in r_target_cols:
        l = c.split("_L")[-1]
        dc = f"dltR_L{l}"
        wide[dc] = wide[c] - wide[base_col]
        dlt_cols.append(dc)
    dlt_cols_for_pca = [c for c in dlt_cols if wide[c].std() > 1e-12]
    if len(dlt_cols_for_pca) < 3:
        dlt_cols_for_pca = r_target_cols
    wide["DeltaU_PC1"], pca, _ = fit_delta_u_pc1(wide, dlt_cols_for_pca)

    # Integral features.
    window_feature_sets = {}
    for wname, layers in WINDOWS.items():
        valid_layers = [l for l in layers if l in available_layers]
        feat_cols = []
        for op in op_cols:
            cols = [f"{op}_L{l}" for l in valid_layers if f"{op}_L{l}" in wide.columns]
            if cols:
                oc = f"{wname}_sum_{op}"
                wide[oc] = wide[cols].sum(axis=1)
                feat_cols.append(oc)
        if feat_cols:
            wide[f"{wname}_norm"] = np.linalg.norm(wide[feat_cols].values.astype(float), axis=1)
            wide[f"{wname}_mean"] = wide[feat_cols].mean(axis=1)
            wide[f"{wname}_absmean"] = np.abs(wide[feat_cols].values.astype(float)).mean(axis=1)
            feat_cols += [f"{wname}_norm", f"{wname}_mean", f"{wname}_absmean"]
            window_feature_sets[wname] = feat_cols

    single_feature_sets = {}
    for l in available_layers:
        cols = [f"{op}_L{l}" for op in op_cols if f"{op}_L{l}" in wide.columns]
        if cols:
            single_feature_sets[f"single_L{l}"] = cols

    feature_sets = {}
    if "I_7_26" in window_feature_sets:
        feature_sets["Integral_I_7_26"] = window_feature_sets["I_7_26"]
    phase_feats = []
    for k in ["I_7_19", "I_20_22", "I_23_26"]:
        phase_feats += window_feature_sets.get(k, [])
    if phase_feats:
        feature_sets["Integral_phase_windows_7_19_20_22_23_26"] = dedup(phase_feats)
    for k, v in window_feature_sets.items():
        feature_sets[f"Window_{k}"] = v
    for k, v in single_feature_sets.items():
        feature_sets[k] = v

    summary_cols = []
    for w in WINDOWS:
        summary_cols += safe_cols(wide, [f"{w}_norm", f"{w}_mean", f"{w}_absmean"])
    if summary_cols:
        feature_sets["summary_only_all_windows"] = dedup(summary_cols)

    pred_table = wide[[id_col] + [c for c in ["condition", "phase", "phase_target"] if c in wide.columns]].copy()
    pred_table["DeltaU_PC1"] = wide["DeltaU_PC1"]
    rows = []
    for name, feats in feature_sets.items():
        for kind in ["ridge", "linear"]:
            try:
                pred, group_col = cv_predict(wide, feats, "DeltaU_PC1", model_kind=kind)
                m = metrics(wide["DeltaU_PC1"], pred)
                rows.append({"model": name, "model_kind": kind, "n_features": len(feats), "group_col": group_col, "features": "|".join(feats), **m})
                pred_table[f"pred_{name}_{kind}"] = pred
            except Exception as e:
                rows.append({"model": name, "model_kind": kind, "status": "error", "error": str(e), "n_features": len(feats), "features": "|".join(feats)})
    model_summary = pd.DataFrame(rows)
    valid = model_summary[np.isfinite(model_summary.get("r2", np.nan))].copy()

    # Layer shuffle control.
    rng = np.random.default_rng(RANDOM_SEED)
    shuffle_rows = []
    base_layers = [l for l in range(7, 27) if l in available_layers]
    if len(base_layers) >= 5:
        for s in range(N_SHUFFLES):
            shuf = wide[[id_col, "DeltaU_PC1"] + [c for c in ["condition", "phase", "phase_target"] if c in wide.columns]].copy()
            perm = base_layers.copy(); rng.shuffle(perm)
            lm = dict(zip(base_layers, perm))
            shuf_feats = []
            for wname, layers in {"SHUF_I_7_19": range(7,20), "SHUF_I_20_22": range(20,23), "SHUF_I_23_26": range(23,27), "SHUF_I_7_26": range(7,27)}.items():
                mapped = [lm[l] for l in layers if l in lm]
                for op in op_cols:
                    cols = [f"{op}_L{ml}" for ml in mapped if f"{op}_L{ml}" in wide.columns]
                    if cols:
                        oc = f"{wname}_sum_{op}"
                        shuf[oc] = wide[cols].sum(axis=1)
                        shuf_feats.append(oc)
            shuf_feats = dedup(shuf_feats)
            if shuf_feats:
                try:
                    pred, group_col = cv_predict(shuf, shuf_feats, "DeltaU_PC1", model_kind="ridge")
                    shuffle_rows.append({"shuffle_id": s, "n_features": len(shuf_feats), "group_col": group_col, **metrics(shuf["DeltaU_PC1"], pred)})
                except Exception as e:
                    shuffle_rows.append({"shuffle_id": s, "status": "error", "error": str(e)})
    shuffle_summary = pd.DataFrame(shuffle_rows)

    # Random feature control with same dimensionality as main integral.
    random_rows = []
    main_feats = feature_sets.get("Integral_I_7_26", feature_sets.get("Integral_phase_windows_7_19_20_22_23_26", []))
    if main_feats:
        for s in range(N_SHUFFLES):
            rand_df = wide[[id_col, "DeltaU_PC1"] + [c for c in ["condition", "phase", "phase_target"] if c in wide.columns]].copy()
            rand_cols = []
            Xrand = rng.normal(0, 1, size=(len(wide), len(main_feats))).astype(np.float32)
            for j in range(Xrand.shape[1]):
                c = f"rand_{j}"
                rand_df[c] = Xrand[:, j]
                rand_cols.append(c)
            try:
                pred, group_col = cv_predict(rand_df, rand_cols, "DeltaU_PC1", model_kind="ridge")
                random_rows.append({"random_id": s, "n_features": len(rand_cols), "group_col": group_col, **metrics(rand_df["DeltaU_PC1"], pred)})
            except Exception as e:
                random_rows.append({"random_id": s, "status": "error", "error": str(e)})
    random_summary = pd.DataFrame(random_rows)

    # Leave-one-window-out ablation.
    ablation_rows = []
    phase_keys = ["I_7_19", "I_20_22", "I_23_26"]
    if all(k in window_feature_sets for k in phase_keys):
        full_feats = dedup(sum([window_feature_sets[k] for k in phase_keys], []))
        pred, group_col = cv_predict(wide, full_feats, "DeltaU_PC1", model_kind="ridge")
        ablation_rows.append({"setting": "full_phase_windows", "removed": "none", "n_features": len(full_feats), "group_col": group_col, **metrics(wide["DeltaU_PC1"], pred)})
        for rem in phase_keys:
            feats = dedup(sum([window_feature_sets[k] for k in phase_keys if k != rem], []))
            pred, group_col = cv_predict(wide, feats, "DeltaU_PC1", model_kind="ridge")
            ablation_rows.append({"setting": "leave_one_window_out", "removed": rem, "n_features": len(feats), "group_col": group_col, **metrics(wide["DeltaU_PC1"], pred)})
    window_ablation = pd.DataFrame(ablation_rows)

    # Best feature importance.
    importance_df = pd.DataFrame()
    best_row = None
    if len(valid):
        best_row = valid.sort_values("r2", ascending=False).iloc[0].to_dict()
        best_feats = best_row["features"].split("|")
        X = wide[best_feats].values.astype(np.float32)
        y = wide["DeltaU_PC1"].values.astype(np.float32)
        model = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=RIDGE_ALPHA))])
        model.fit(X, y)
        try:
            perm = permutation_importance(model, X, y, n_repeats=32, random_state=RANDOM_SEED, scoring="r2")
            importance_df = pd.DataFrame({"feature": best_feats, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)
        except Exception:
            pass

    best_integral = None
    for nm in ["Integral_I_7_26", "Integral_phase_windows_7_19_20_22_23_26"]:
        sub = valid[(valid["model"] == nm) & (valid["model_kind"] == "ridge")]
        if len(sub):
            r = sub.sort_values("r2", ascending=False).iloc[0].to_dict()
            if best_integral is None or r["r2"] > best_integral["r2"]:
                best_integral = r
    best_single = None
    singles = valid[valid["model"].str.startswith("single_L", na=False)]
    if len(singles):
        best_single = singles.sort_values("r2", ascending=False).iloc[0].to_dict()

    shuffle_r2_mean = float(shuffle_summary["r2"].mean()) if "r2" in shuffle_summary and len(shuffle_summary) else np.nan
    shuffle_r2_max = float(shuffle_summary["r2"].max()) if "r2" in shuffle_summary and len(shuffle_summary) else np.nan
    random_r2_mean = float(random_summary["r2"].mean()) if "r2" in random_summary and len(random_summary) else np.nan
    random_r2_max = float(random_summary["r2"].max()) if "r2" in random_summary and len(random_summary) else np.nan
    integral_r2 = float(best_integral["r2"]) if best_integral else np.nan
    integral_corr = float(best_integral["corr"]) if best_integral else np.nan
    single_r2 = float(best_single["r2"]) if best_single else np.nan
    gain_vs_shuffle = integral_r2 - shuffle_r2_mean if np.isfinite(integral_r2) and np.isfinite(shuffle_r2_mean) else np.nan
    gain_vs_single = integral_r2 - single_r2 if np.isfinite(integral_r2) and np.isfinite(single_r2) else np.nan

    if np.isfinite(integral_r2) and np.isfinite(integral_corr):
        if integral_r2 >= 0.60 and abs(integral_corr) >= 0.80 and (not np.isfinite(gain_vs_shuffle) or gain_vs_shuffle > 0.10):
            verdict = "PASS-Strong"
        elif integral_r2 >= 0.35 and abs(integral_corr) >= 0.60:
            verdict = "PASS-Lite"
        else:
            verdict = "INCONCLUSIVE-or-FAIL"
    else:
        verdict = "INCONCLUSIVE-or-FAIL"

    summary = {
        "input_csv": INPUT_CSV,
        "n_rows_raw": int(len(df)),
        "n_trajectories": int(len(wide)),
        "id_col": id_col,
        "available_layers": available_layers,
        "op_cols": op_cols,
        "r_curr_col": r_curr_col,
        "r_next_col": r_next_col,
        "r_target_cols": r_target_cols,
        "deltaR_cols_for_pca": dlt_cols_for_pca,
        "DeltaU_PC1_explained_variance": float(pca.explained_variance_ratio_[0]),
        "DeltaU_PC123_explained_variance": float(np.sum(pca.explained_variance_ratio_[:min(3, len(pca.explained_variance_ratio_))])),
        "best_overall_model": best_row,
        "best_integral_model": best_integral,
        "best_single_layer_model": best_single,
        "best_integral_r2": integral_r2,
        "best_integral_corr": integral_corr,
        "best_single_r2": single_r2,
        "gain_vs_best_single": gain_vs_single,
        "shuffle_r2_mean": shuffle_r2_mean,
        "shuffle_r2_max": shuffle_r2_max,
        "random_r2_mean": random_r2_mean,
        "random_r2_max": random_r2_max,
        "gain_vs_shuffle_mean": gain_vs_shuffle,
        "verdict": verdict,
        "verdict_rule": "PASS-Strong if integral R2>=0.60, |corr|>=0.80, and layer-shuffle gain >0.10; PASS-Lite if R2>=0.35 and |corr|>=0.60.",
    }

    wide.to_csv(OUTPUT_DIR / "phasemap8a1_integral_features.csv", index=False)
    model_summary.sort_values(["r2", "corr"], ascending=False, na_position="last").to_csv(OUTPUT_DIR / "phasemap8a1_model_summary.csv", index=False)
    window_ablation.to_csv(OUTPUT_DIR / "phasemap8a1_window_ablation.csv", index=False)
    shuffle_summary.to_csv(OUTPUT_DIR / "phasemap8a1_shuffle_control.csv", index=False)
    random_summary.to_csv(OUTPUT_DIR / "phasemap8a1_random_control.csv", index=False)
    pred_table.to_csv(OUTPUT_DIR / "phasemap8a1_predictions.csv", index=False)
    importance_df.to_csv(OUTPUT_DIR / "phasemap8a1_best_feature_importance.csv", index=False)
    with open(OUTPUT_DIR / "phasemap8a1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n================ PhaseMap-8A.1 Summary ================")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nTop models:")
    if len(valid):
        print(valid.sort_values(["r2", "corr"], ascending=False).head(20)[["model", "model_kind", "n_features", "r2", "corr", "rmse", "mae"]].to_string(index=False))
    else:
        print("No valid models.")
    print("\nOutputs saved to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
