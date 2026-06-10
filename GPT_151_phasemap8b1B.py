# ============================================================
# PhaseMap-8B STRICT
# True Multi-Path Geodesic Dominance Audit
#
# This version deliberately REMOVES pseudo-multipath fallback.
#
# Required inputs:
#   1) phasemap6b1_correction_dataset.csv
#   2) phasemap8b_path_scores.csv
#
# phasemap8b_path_scores.csv must contain:
#   sample_id, layer, path_name, path_score
#
# Optional columns:
#   condition, phase
#
# Interpretation:
#   path_score should be a real logit/log-score/probe score for a
#   candidate explanation path / basin, not derived from R_l alone.
#
# Output:
#   phasemap8b_strict_outputs/
# ============================================================

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

PRIMARY_CSV = r"C:\Windows\System32\phasemap6b1_outputs\phasemap6b1_correction_dataset.csv"
PATH_SCORE_CSV = r"C:\Windows\System32\phasemap8b1_outputs\phasemap8b_path_scores.csv"
OUTPUT_DIR = Path("phasemap8b_strict_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_SPLITS = 5
RIDGE_ALPHA = 1.0
N_SHUFFLES = 64

DELTAU_LAYERS = list(range(20, 26))
WINDOWS = {
    "G_7_19": list(range(7, 20)),
    "G_20_22": list(range(20, 23)),
    "G_23_25": list(range(23, 26)),
    "G_20_25": list(range(20, 26)),
    "G_7_25": list(range(7, 26)),
}

def first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def pearsonr_safe(a, b):
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

def reg_metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "corr": pearsonr_safe(y_true, y_pred),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }

def get_cv(df, group_col):
    if group_col in df.columns:
        groups = df[group_col].astype(str).values
        n_unique = len(np.unique(groups))
        if n_unique >= 2:
            return GroupKFold(n_splits=min(N_SPLITS, n_unique)), groups, group_col
    return KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED), None, None

def cv_predict(df, features, target_col, group_col):
    X = df[features].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=RIDGE_ALPHA)),
    ])
    cv, groups, used_group = get_cv(df, group_col)
    if groups is None:
        pred = cross_val_predict(model, X, y, cv=cv)
    else:
        pred = cross_val_predict(model, X, y, cv=cv, groups=groups)
    return pred, used_group

def softmax(scores):
    s = np.asarray(scores, dtype=float)
    s = s - np.nanmax(s)
    p = np.exp(np.clip(s, -60, 60))
    z = np.nansum(p)
    if z <= 0 or not np.isfinite(z):
        return np.ones_like(p) / len(p)
    return p / z

def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if len(p) == 0:
        return np.nan
    return float(-np.sum(p * np.log(p)))

def fit_deltaU(primary, id_col):
    df = primary.copy()
    df["layer"] = df["layer"].astype(int)
    r_curr = first_existing(df, ["R_l", "R", "R_current", "r_l"])
    r_next = first_existing(df, ["R_l1", "R_next", "R_l+1", "r_l1", "r_next"])
    if r_curr is None and r_next is None:
        raise RuntimeError("Primary CSV must contain R_l or R_l1/R_next.")

    agg_cols = []
    if r_curr is not None:
        agg_cols.append(r_curr)
    if r_next is not None:
        agg_cols.append(r_next)

    num = df.groupby([id_col, "layer"], as_index=False)[agg_cols].mean()
    meta_cols = [id_col]
    for c in ["phase_target", "phase", "condition"]:
        if c in df.columns and c not in meta_cols:
            meta_cols.append(c)
    wide = df.groupby(id_col, as_index=False).first()[meta_cols]

    if r_curr is not None:
        piv = num.pivot(index=id_col, columns="layer", values=r_curr)
        piv.columns = [f"R_L{int(l)}" for l in piv.columns]
        wide = wide.merge(piv.reset_index(), on=id_col, how="left")

    if r_next is not None:
        piv = num.pivot(index=id_col, columns="layer", values=r_next)
        piv.columns = [f"Rnext_from_L{int(l)}" for l in piv.columns]
        wide = wide.merge(piv.reset_index(), on=id_col, how="left")

    for l in DELTAU_LAYERS:
        if f"R_L{l}" not in wide.columns:
            prev = f"Rnext_from_L{l-1}"
            if prev in wide.columns:
                wide[f"R_L{l}"] = wide[prev]

    r_cols = [f"R_L{l}" for l in DELTAU_LAYERS if f"R_L{l}" in wide.columns]
    if len(r_cols) < 4:
        raise RuntimeError(f"Too few R columns for DeltaU: {r_cols}")

    wide = wide.dropna(subset=r_cols).reset_index(drop=True)
    X = wide[r_cols].values.astype(np.float32)
    Xd = X - X[:, [0]]
    keep = np.std(Xd, axis=0) > 1e-12
    if keep.sum() >= 3:
        Xp = Xd[:, keep]
        used_cols = [f"dlt_{c}" for c, k in zip(r_cols, keep) if k]
    else:
        Xp = X
        used_cols = r_cols

    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xp)
    pca = PCA(n_components=min(Xs.shape[1], len(r_cols)), random_state=RANDOM_SEED)
    Z = pca.fit_transform(Xs)
    u = Z[:, 0]
    if pearsonr_safe(u, X.mean(axis=1)) < 0:
        u = -u
        pca.components_[0, :] *= -1

    wide["DeltaU_PC1"] = u.astype(np.float32)
    return wide, pca, used_cols, r_cols

def build_path_gap(path_df, id_col, allowed_ids):
    path_df = path_df.copy()
    path_df["layer"] = path_df["layer"].astype(int)
    path_df["path_score"] = path_df["path_score"].astype(float)
    path_df = path_df[path_df[id_col].isin(allowed_ids)]

    rows = []
    for (sid, layer), g in path_df.groupby([id_col, "layer"]):
        g = g.dropna(subset=["path_score"])
        if g["path_name"].nunique() < 2:
            continue

        scores = g["path_score"].values.astype(float)
        names = g["path_name"].astype(str).values
        order = np.argsort(scores)[::-1]
        probs = softmax(scores)
        porder = probs[order]

        rows.append({
            id_col: sid,
            "layer": int(layer),
            "winner": names[order[0]],
            "runner_up": names[order[1]],
            "gap_log": float(scores[order[0]] - scores[order[1]]),
            "gap_prob": float(porder[0] - porder[1]),
            "top1_score": float(scores[order[0]]),
            "top2_score": float(scores[order[1]]),
            "top1_prob": float(porder[0]),
            "top2_prob": float(porder[1]),
            "path_entropy": entropy(probs),
            "n_paths": int(len(scores)),
        })
    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise RuntimeError("No valid multi-path rows. Need at least 2 paths per sample_id/layer.")
    return out

def add_window_features(base, gap_long, id_col):
    out = base.copy()
    value_cols = ["gap_log", "gap_prob", "top1_score", "top2_score", "top1_prob", "top2_prob", "path_entropy"]
    for vc in value_cols:
        piv = gap_long.pivot(index=id_col, columns="layer", values=vc)
        piv.columns = [f"{vc}_L{int(l)}" for l in piv.columns]
        out = out.merge(piv.reset_index(), on=id_col, how="left")

    wpiv = gap_long.pivot(index=id_col, columns="layer", values="winner")
    wpiv.columns = [f"winner_L{int(l)}" for l in wpiv.columns]
    out = out.merge(wpiv.reset_index(), on=id_col, how="left")

    feature_sets = {}
    for wname, layers in WINDOWS.items():
        feats = []
        for vc in value_cols:
            cols = [f"{vc}_L{l}" for l in layers if f"{vc}_L{l}" in out.columns]
            if not cols:
                continue
            X = out[cols].values.astype(float)
            stats = {
                "mean": np.nanmean(X, axis=1),
                "max": np.nanmax(X, axis=1),
                "min": np.nanmin(X, axis=1),
                "last": X[:, -1],
                "first": X[:, 0],
                "slope": X[:, -1] - X[:, 0],
                "range": np.nanmax(X, axis=1) - np.nanmin(X, axis=1),
                "std": np.nanstd(X, axis=1),
            }
            for s, arr in stats.items():
                c = f"{wname}_{vc}_{s}"
                out[c] = arr
                feats.append(c)

        wcols = [f"winner_L{l}" for l in layers if f"winner_L{l}" in out.columns]
        if wcols:
            switches, domfrac, wentropy, firstw, lastw = [], [], [], [], []
            for _, row in out[wcols].iterrows():
                vals = [v for v in row.values.tolist() if isinstance(v, str) and v]
                if not vals:
                    switches.append(np.nan); domfrac.append(np.nan); wentropy.append(np.nan); firstw.append(""); lastw.append("")
                    continue
                vc = pd.Series(vals).value_counts(normalize=True)
                switches.append(float(sum(vals[i] != vals[i-1] for i in range(1, len(vals)))))
                domfrac.append(float(vc.iloc[0]))
                wentropy.append(entropy(vc.values))
                firstw.append(vals[0]); lastw.append(vals[-1])
            for name, arr in [("winner_switches", switches), ("winner_dominant_frac", domfrac), ("winner_entropy", wentropy)]:
                c = f"{wname}_{name}"
                out[c] = arr
                feats.append(c)
            out[f"{wname}_winner_first"] = firstw
            out[f"{wname}_winner_last"] = lastw

        feature_sets[wname] = feats

    for c in set(sum(feature_sets.values(), [])):
        med = out[c].median()
        out[c] = out[c].fillna(0.0 if not np.isfinite(med) else med)
    return out, feature_sets

def main():
    if not Path(PRIMARY_CSV).exists():
        raise FileNotFoundError(f"Missing {PRIMARY_CSV}")
    if not Path(PATH_SCORE_CSV).exists():
        raise FileNotFoundError(
            f"Missing {PATH_SCORE_CSV}. Strict 8B requires true multi-path scores. "
            "Run phasemap8b_path_score_builder_template.py or create the file manually."
        )

    primary = pd.read_csv(PRIMARY_CSV)
    primary.columns = [str(c).strip() for c in primary.columns]
    id_col = first_existing(primary, ["sample_id", "idx", "graph_id", "prompt_id", "case_id"])
    if id_col is None:
        raise RuntimeError("Primary CSV needs sample_id/idx/graph_id/prompt_id/case_id.")

    wide, pca, delta_used_cols, r_cols = fit_deltaU(primary, id_col)
    phase_col = first_existing(wide, ["phase_target", "phase"])
    condition_col = first_existing(wide, ["condition"])

    path_df = pd.read_csv(PATH_SCORE_CSV)
    path_df.columns = [str(c).strip() for c in path_df.columns]
    if id_col not in path_df.columns and "sample_id" in path_df.columns:
        path_df = path_df.rename(columns={"sample_id": id_col})

    required = {id_col, "layer", "path_name", "path_score"}
    if not required.issubset(path_df.columns):
        raise RuntimeError(f"{PATH_SCORE_CSV} must contain columns: {sorted(required)}")

    gap_long = build_path_gap(path_df, id_col, set(wide[id_col]))
    base_cols = [id_col, "DeltaU_PC1"]
    for c in [phase_col, condition_col]:
        if c and c in wide.columns and c not in base_cols:
            base_cols.append(c)
    gap_wide, feature_sets = add_window_features(wide[base_cols], gap_long, id_col)

    all_feature_sets = {}
    for w, feats in feature_sets.items():
        all_feature_sets[f"window_{w}"] = feats
    combo = []
    for w in ["G_20_22", "G_23_25"]:
        combo += feature_sets.get(w, [])
    all_feature_sets["combo_G_20_22_plus_23_25"] = list(dict.fromkeys(combo))
    combo_all = []
    for w in ["G_7_19", "G_20_22", "G_23_25"]:
        combo_all += feature_sets.get(w, [])
    all_feature_sets["combo_G_7_19_20_22_23_25"] = list(dict.fromkeys(combo_all))

    model_rows = []
    pred_table = gap_wide[[id_col, "DeltaU_PC1"] + [c for c in [phase_col, condition_col] if c and c in gap_wide.columns]].copy()
    for name, feats in all_feature_sets.items():
        feats = [f for f in feats if f in gap_wide.columns]
        if not feats:
            continue
        pred, used_group = cv_predict(gap_wide, feats, "DeltaU_PC1", id_col)
        m = reg_metrics(gap_wide["DeltaU_PC1"], pred)
        model_rows.append({"model": name, "n_features": len(feats), "group_col": used_group, "features": "|".join(feats), **m})
        pred_table[f"pred_{name}"] = pred
    model_summary = pd.DataFrame(model_rows).sort_values("r2", ascending=False)

    direct_rows = []
    for c in set(sum(feature_sets.values(), [])):
        if c in gap_wide.columns and gap_wide[c].std() > 1e-12:
            corr = pearsonr_safe(gap_wide[c], gap_wide["DeltaU_PC1"])
            coef = np.polyfit(gap_wide[c].values, gap_wide["DeltaU_PC1"].values, 1)
            pred = np.poly1d(coef)(gap_wide[c].values)
            direct_rows.append({"feature": c, "corr_with_DeltaU": corr, "r2_linear_fit": float(r2_score(gap_wide["DeltaU_PC1"], pred))})
    direct_corr = pd.DataFrame(direct_rows)
    if len(direct_corr):
        direct_corr = direct_corr.sort_values("corr_with_DeltaU", key=lambda s: np.abs(s), ascending=False)

    phase_profile = pd.DataFrame()
    if phase_col and phase_col in gap_wide.columns:
        cols = ["DeltaU_PC1"]
        for c in ["G_20_25_gap_log_mean", "G_20_25_gap_prob_mean", "G_20_25_path_entropy_mean", "G_20_25_winner_switches", "G_20_25_winner_dominant_frac"]:
            if c in gap_wide.columns:
                cols.append(c)
        phase_profile = gap_wide.groupby(phase_col)[cols].agg(["mean", "std", "count"])
        phase_profile.columns = ["__".join(x) for x in phase_profile.columns]
        phase_profile = phase_profile.reset_index()

    winner_rows = []
    for w in ["G_20_22", "G_23_25", "G_20_25"]:
        for pos in ["first", "last"]:
            c = f"{w}_winner_{pos}"
            if c in gap_wide.columns:
                vc = gap_wide[c].value_counts(normalize=True)
                for winner, frac in vc.items():
                    winner_rows.append({"window": w, "position": pos, "winner": winner, "frac": float(frac), "count": int((gap_wide[c] == winner).sum())})
    winner_profile = pd.DataFrame(winner_rows)

    layer_rows = []
    for l, sub in gap_long.groupby("layer"):
        vc = sub["winner"].value_counts(normalize=True)
        layer_rows.append({
            "layer": int(l), "n": int(len(sub)),
            "gap_log_mean": float(sub["gap_log"].mean()),
            "gap_prob_mean": float(sub["gap_prob"].mean()),
            "path_entropy_mean": float(sub["path_entropy"].mean()),
            "n_paths_mean": float(sub["n_paths"].mean()),
            "winner_entropy": entropy(vc.values),
            "top_winner": str(vc.index[0]) if len(vc) else "",
            "top_winner_frac": float(vc.iloc[0]) if len(vc) else np.nan,
        })
    layer_profile = pd.DataFrame(layer_rows)

    # Layer shuffle control.
    rng = np.random.default_rng(RANDOM_SEED)
    shuffle_rows = []
    for s in range(N_SHUFFLES):
        sh = gap_long.copy()
        for sid, idxs in sh.groupby(id_col).groups.items():
            idxs = list(idxs)
            for col in ["gap_log", "gap_prob", "path_entropy"]:
                vals = sh.loc[idxs, col].values.copy()
                rng.shuffle(vals)
                sh.loc[idxs, f"{col}_shuf"] = vals

        swide = wide[[id_col, "DeltaU_PC1"]].copy()
        feats = []
        for col in ["gap_log_shuf", "gap_prob_shuf", "path_entropy_shuf"]:
            piv = sh.pivot(index=id_col, columns="layer", values=col)
            piv.columns = [f"{col}_L{int(l)}" for l in piv.columns]
            swide = swide.merge(piv.reset_index(), on=id_col, how="left")
            layers = WINDOWS["G_20_25"]
            cols = [f"{col}_L{l}" for l in layers if f"{col}_L{l}" in swide.columns]
            if cols:
                X = swide[cols].values.astype(float)
                for stat, arr in {
                    "mean": np.nanmean(X, axis=1),
                    "last": X[:, -1],
                    "slope": X[:, -1] - X[:, 0],
                    "range": np.nanmax(X, axis=1) - np.nanmin(X, axis=1),
                }.items():
                    fc = f"SHUF_{col}_{stat}"
                    swide[fc] = arr
                    swide[fc] = swide[fc].fillna(swide[fc].median())
                    feats.append(fc)
        if feats:
            pred, used_group = cv_predict(swide, feats, "DeltaU_PC1", id_col)
            shuffle_rows.append({"shuffle_id": s, "n_features": len(feats), "group_col": used_group, **reg_metrics(swide["DeltaU_PC1"], pred)})
    shuffle_summary = pd.DataFrame(shuffle_rows)

    best = model_summary.iloc[0].to_dict() if len(model_summary) else None
    main = model_summary[model_summary["model"] == "window_G_20_25"].iloc[0].to_dict() if "window_G_20_25" in set(model_summary["model"]) else None
    shuf_mean = float(shuffle_summary["r2"].mean()) if len(shuffle_summary) else np.nan
    main_r2 = float(main["r2"]) if main else np.nan
    main_corr = float(main["corr"]) if main else np.nan
    gain = main_r2 - shuf_mean if np.isfinite(shuf_mean) and np.isfinite(main_r2) else np.nan

    if main and main_r2 >= 0.60 and abs(main_corr) >= 0.80 and gain > 0.10:
        verdict = "PASS-Strong"
    elif main and main_r2 >= 0.35 and abs(main_corr) >= 0.60:
        verdict = "PASS-Lite"
    else:
        verdict = "INCONCLUSIVE-or-FAIL"

    summary = {
        "mode": "true_multipath_strict",
        "primary_csv": PRIMARY_CSV,
        "path_score_csv": PATH_SCORE_CSV,
        "n_primary_rows": int(len(primary)),
        "n_trajectories": int(len(gap_wide)),
        "n_path_score_rows": int(len(path_df)),
        "n_gap_long_rows": int(len(gap_long)),
        "id_col": id_col,
        "phase_col": phase_col,
        "condition_col": condition_col,
        "r_cols_for_DeltaU": r_cols,
        "delta_used_cols": delta_used_cols,
        "DeltaU_PC1_explained_variance": float(pca.explained_variance_ratio_[0]),
        "DeltaU_PC123_explained_variance": float(np.sum(pca.explained_variance_ratio_[:min(3, len(pca.explained_variance_ratio_))])),
        "mean_n_paths": float(gap_long["n_paths"].mean()),
        "unique_path_names": sorted(path_df["path_name"].astype(str).unique().tolist()),
        "best_model": best,
        "main_G20_25_model": main,
        "best_direct_feature": direct_corr.iloc[0].to_dict() if len(direct_corr) else None,
        "shuffle_r2_mean": shuf_mean,
        "gain_vs_shuffle_mean": gain,
        "verdict": verdict,
    }

    gap_long.to_csv(OUTPUT_DIR / "phasemap8b_strict_path_gap_long.csv", index=False)
    gap_wide.to_csv(OUTPUT_DIR / "phasemap8b_strict_gap_features.csv", index=False)
    model_summary.to_csv(OUTPUT_DIR / "phasemap8b_strict_model_summary.csv", index=False)
    direct_corr.to_csv(OUTPUT_DIR / "phasemap8b_strict_direct_correlations.csv", index=False)
    phase_profile.to_csv(OUTPUT_DIR / "phasemap8b_strict_phase_profile.csv", index=False)
    winner_profile.to_csv(OUTPUT_DIR / "phasemap8b_strict_winner_profile.csv", index=False)
    layer_profile.to_csv(OUTPUT_DIR / "phasemap8b_strict_layer_profile.csv", index=False)
    shuffle_summary.to_csv(OUTPUT_DIR / "phasemap8b_strict_shuffle_control.csv", index=False)
    pred_table.to_csv(OUTPUT_DIR / "phasemap8b_strict_predictions.csv", index=False)

    with open(OUTPUT_DIR / "phasemap8b_strict_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n================ PhaseMap-8B STRICT Summary ================")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nTop models:")
    print(model_summary[["model", "n_features", "r2", "corr", "rmse", "mae"]].head(20).to_string(index=False))
    print("\nOutputs saved to:", OUTPUT_DIR.resolve())

if __name__ == "__main__":
    main()
