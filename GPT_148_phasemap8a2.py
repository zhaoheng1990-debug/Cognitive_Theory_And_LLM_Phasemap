# ============================================================
# PhaseMap-8A.2
# Geodesic Gap Audit
#
# Goal:
#   Test whether the PhaseMap order observable:
#
#       DeltaU = PC1(DeltaR_{20:25})
#
#   is better interpreted as:
#
#       dominant geodesic/path advantage
#
#   i.e.
#
#       DeltaU ≈ lambda_1 - lambda_2
#
#   or, in log-space:
#
#       DeltaU ≈ log(lambda_1 / lambda_2)
#
# Fixed primary input:
#   phasemap6b1_correction_dataset.csv
#
# Optional multi-path input:
#   phasemap8a2_path_scores.csv
#
#   If present, this file should contain:
#       sample_id, layer, path_name, path_score
#
#   where path_score can be a logit/log-score for each candidate
#   explanation path / basin.
#
# Modes:
#   1. Multi-path mode:
#      Uses path_score across candidate paths.
#
#   2. Binary fallback mode:
#      Uses R_l = logit(clean) - logit(conflict)
#      as a two-path log advantage.
#
# Main outputs:
#   phasemap8a2_outputs/
#     phasemap8a2_summary.json
#     phasemap8a2_gap_features.csv
#     phasemap8a2_model_summary.csv
#     phasemap8a2_phase_profile.csv
#     phasemap8a2_layer_profile.csv
#     phasemap8a2_shuffle_control.csv
# ============================================================

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    roc_auc_score,
    accuracy_score,
    f1_score,
    classification_report,
)

warnings.filterwarnings("ignore")

# -----------------------------
# CONFIG
# -----------------------------

PRIMARY_CSV = r"C:\Windows\System32\phasemap6b1_outputs\phasemap6b1_correction_dataset.csv"
OPTIONAL_PATH_SCORE_CSV = r"phasemap8a2_path_scores.csv"

OUTPUT_DIR = Path("phasemap8a2_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_SPLITS = 5
N_SHUFFLES = 64
RIDGE_ALPHA = 1.0

DELTAU_LAYERS = list(range(20, 26))
GAP_WINDOWS = {
    "G_7_19": list(range(7, 20)),
    "G_20_22": list(range(20, 23)),
    "G_23_25": list(range(23, 26)),
    "G_20_25": list(range(20, 26)),
    "G_7_25": list(range(7, 26)),
}

# -----------------------------
# HELPERS
# -----------------------------

def first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def safe_cols(df, candidates):
    return [c for c in candidates if c in df.columns]

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

def metrics_reg(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "corr": pearsonr_safe(y_true, y_pred),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }

def get_group_cv(df, group_col=None):
    if group_col is not None and group_col in df.columns:
        groups = df[group_col].astype(str).values
        n_unique = len(np.unique(groups))
        n_splits = min(N_SPLITS, n_unique)
        if n_splits >= 2:
            return GroupKFold(n_splits=n_splits), groups, group_col
    return KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED), None, None

def cv_predict_reg(df, features, target_col, group_col=None):
    X = df[features].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=RIDGE_ALPHA)),
    ])

    cv, groups, used_group = get_group_cv(df, group_col)
    if groups is None:
        pred = cross_val_predict(model, X, y, cv=cv)
    else:
        pred = cross_val_predict(model, X, y, cv=cv, groups=groups)

    return pred, used_group

def cv_predict_clf(df, features, target_col, group_col=None):
    X = df[features].values.astype(np.float32)
    y = df[target_col].values.astype(int)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])

    # For classification, if grouped CV is impossible or class split bad, fallback stratified.
    if group_col is not None and group_col in df.columns:
        groups = df[group_col].astype(str).values
        n_unique = len(np.unique(groups))
        if n_unique >= N_SPLITS:
            cv = GroupKFold(n_splits=N_SPLITS)
            try:
                prob = cross_val_predict(model, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
                return prob, group_col
            except Exception:
                pass

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    return prob, None

def fit_deltaU_from_R(wide_df, r_cols):
    X = wide_df[r_cols].values.astype(np.float32)

    # Convert to DeltaR relative to first available layer.
    Xd = X - X[:, [0]]

    # Drop constant first delta column if needed.
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

    # Orient DeltaU so higher means larger mean R / clean advantage.
    if pearsonr_safe(u, X.mean(axis=1)) < 0:
        u = -u
        pca.components_[0, :] *= -1

    return u.astype(np.float32), pca, scaler, used_cols

def phase_order_check(df, value_col, phase_col="phase_target"):
    if phase_col not in df.columns:
        return {}

    prof = df.groupby(phase_col)[value_col].agg(["mean", "std", "count"]).reset_index()
    means = dict(zip(prof[phase_col], prof["mean"]))

    out = {
        "phase_profile": prof.to_dict(orient="records"),
        "has_positive": "positive" in means,
        "has_critical": "critical" in means,
        "has_negative": "negative" in means,
    }

    if all(k in means for k in ["positive", "critical", "negative"]):
        out["positive_gt_critical_gt_negative"] = bool(
            means["positive"] > means["critical"] > means["negative"]
        )
        out["abs_gap_critical_min"] = bool(
            abs(means["critical"]) <= max(abs(means["positive"]), abs(means["negative"]))
        )

    return out

# -----------------------------
# LOAD PRIMARY
# -----------------------------

if not Path(PRIMARY_CSV).exists():
    raise FileNotFoundError(
        f"Cannot find {PRIMARY_CSV}. Put this script in the same folder as "
        f"{PRIMARY_CSV} and run again."
    )

df = pd.read_csv(PRIMARY_CSV)
df.columns = [str(c).strip() for c in df.columns]

if "layer" not in df.columns:
    raise RuntimeError("Primary CSV must contain a layer column.")

df["layer"] = df["layer"].astype(int)

id_col = first_existing(df, ["sample_id", "idx", "graph_id", "prompt_id", "case_id"])
if id_col is None:
    df["_pseudo_id"] = np.arange(len(df)) // max(1, df["layer"].nunique())
    id_col = "_pseudo_id"

phase_col = first_existing(df, ["phase_target", "phase"])
condition_col = first_existing(df, ["condition", "family", "condition_name"])

r_curr_col = first_existing(df, ["R_l", "R", "R_current", "r_l"])
r_next_col = first_existing(df, ["R_l1", "R_next", "R_l+1", "r_l1", "r_next"])

if r_curr_col is None and r_next_col is None:
    raise RuntimeError("Need R_l or R_l1/R_next in primary CSV.")

# Meta table.
meta_cols = [id_col]
for c in [phase_col, condition_col, "heuristic_operator", "operator_cluster"]:
    if c is not None and c in df.columns and c not in meta_cols:
        meta_cols.append(c)

meta = df.groupby(id_col, as_index=False).first()[meta_cols]

# Aggregate numeric by id/layer.
agg_cols = []
if r_curr_col is not None:
    agg_cols.append(r_curr_col)
if r_next_col is not None:
    agg_cols.append(r_next_col)

num = df.groupby([id_col, "layer"], as_index=False)[agg_cols].mean()
available_layers = sorted(num["layer"].unique().tolist())

wide = meta.copy()

# Pivot R_l.
if r_curr_col is not None:
    piv = num.pivot(index=id_col, columns="layer", values=r_curr_col)
    piv.columns = [f"R_L{int(l)}" for l in piv.columns]
    wide = wide.merge(piv.reset_index(), on=id_col, how="left")

if r_next_col is not None:
    piv = num.pivot(index=id_col, columns="layer", values=r_next_col)
    piv.columns = [f"Rnext_from_L{int(l)}" for l in piv.columns]
    wide = wide.merge(piv.reset_index(), on=id_col, how="left")

# Fill R_Ll using next-from previous if needed.
for l in DELTAU_LAYERS:
    if f"R_L{l}" not in wide.columns:
        prev = f"Rnext_from_L{l-1}"
        if prev in wide.columns:
            wide[f"R_L{l}"] = wide[prev]

r_cols = [f"R_L{l}" for l in DELTAU_LAYERS if f"R_L{l}" in wide.columns]
if len(r_cols) < 4:
    raise RuntimeError(f"Too few R columns for DeltaU. Found: {r_cols}")

wide = wide.dropna(subset=r_cols).reset_index(drop=True)
wide["DeltaU_PC1"], pca, scaler, delta_used_cols = fit_deltaU_from_R(wide, r_cols)

# -----------------------------
# GEODESIC GAP CONSTRUCTION
# -----------------------------

path_score_file = Path(OPTIONAL_PATH_SCORE_CSV)
mode = "binary_R_fallback"

layer_gap_long = []

if path_score_file.exists():
    ps = pd.read_csv(path_score_file)
    ps.columns = [str(c).strip() for c in ps.columns]

    required = {id_col, "layer", "path_name", "path_score"}
    # Accept sample_id even if primary id_col differs.
    if id_col not in ps.columns and "sample_id" in ps.columns:
        ps = ps.rename(columns={"sample_id": id_col})

    if required.issubset(set(ps.columns)):
        mode = "multi_path_scores"
        ps["layer"] = ps["layer"].astype(int)

        # For each sample/layer, compute sorted path scores.
        for (sid, layer), g in ps.groupby([id_col, "layer"]):
            scores = g["path_score"].values.astype(float)
            names = g["path_name"].astype(str).values

            if len(scores) < 2:
                continue

            order = np.argsort(scores)[::-1]
            top1 = scores[order[0]]
            top2 = scores[order[1]]
            name1 = names[order[0]]
            name2 = names[order[1]]

            # Log-score gap.
            gap_log = top1 - top2

            # Probability gap under softmax.
            s = scores - np.max(scores)
            p = np.exp(s)
            p = p / p.sum()
            p_order = p[order]
            gap_prob = p_order[0] - p_order[1]

            layer_gap_long.append({
                id_col: sid,
                "layer": int(layer),
                "gap_log": float(gap_log),
                "gap_prob": float(gap_prob),
                "winner": name1,
                "runner_up": name2,
                "top1_score": float(top1),
                "top2_score": float(top2),
                "n_paths": int(len(scores)),
            })

# Binary fallback using R as clean/conflict log-score difference.
if not layer_gap_long:
    mode = "binary_R_fallback"

    # In binary case:
    #   R = logit(clean)-logit(conflict)
    #   signed_gap = R
    #   advantage magnitude = |R| = top1-top2
    #
    # For DeltaU orientation, signed_gap is often more relevant than abs gap.
    # For geodesic dominance, abs gap measures lambda1-lambda2 irrespective winner.
    for _, row in num.iterrows():
        sid = row[id_col]
        layer = int(row["layer"])

        Rval = np.nan
        if r_curr_col is not None and r_curr_col in row:
            Rval = row[r_curr_col]
        elif r_next_col is not None and r_next_col in row:
            Rval = row[r_next_col]

        if not np.isfinite(Rval):
            continue

        signed_gap = float(Rval)
        abs_gap = float(abs(Rval))
        winner = "clean" if signed_gap >= 0 else "conflict"
        runner = "conflict" if signed_gap >= 0 else "clean"

        # Convert log-odds to probability gap:
        # p_clean = sigmoid(R), p_conflict = 1-p_clean
        p_clean = 1.0 / (1.0 + np.exp(-np.clip(signed_gap, -60, 60)))
        gap_prob_signed = 2.0 * p_clean - 1.0
        gap_prob_abs = abs(gap_prob_signed)

        layer_gap_long.append({
            id_col: sid,
            "layer": layer,
            "gap_log_signed": signed_gap,
            "gap_log": abs_gap,
            "gap_prob_signed": float(gap_prob_signed),
            "gap_prob": float(gap_prob_abs),
            "winner": winner,
            "runner_up": runner,
            "top1_score": abs_gap,
            "top2_score": 0.0,
            "n_paths": 2,
        })

gap_long = pd.DataFrame(layer_gap_long)

# Merge only sample ids present in wide.
gap_long = gap_long[gap_long[id_col].isin(set(wide[id_col]))].copy()

# -----------------------------
# WIDE GAP FEATURES
# -----------------------------

gap_wide = wide[[id_col, "DeltaU_PC1"] + [c for c in [phase_col, condition_col] if c is not None and c in wide.columns]].copy()

# Add layer-level gap columns.
for value_col in ["gap_log", "gap_prob", "gap_log_signed", "gap_prob_signed"]:
    if value_col not in gap_long.columns:
        continue
    piv = gap_long.pivot(index=id_col, columns="layer", values=value_col)
    piv.columns = [f"{value_col}_L{int(l)}" for l in piv.columns]
    gap_wide = gap_wide.merge(piv.reset_index(), on=id_col, how="left")

# Winner columns by layer.
winner_piv = gap_long.pivot(index=id_col, columns="layer", values="winner")
winner_piv.columns = [f"winner_L{int(l)}" for l in winner_piv.columns]
gap_wide = gap_wide.merge(winner_piv.reset_index(), on=id_col, how="left")

# Window summaries.
feature_cols_by_window = {}

base_gap_cols = ["gap_log", "gap_prob"]
if "gap_log_signed" in gap_long.columns:
    base_gap_cols += ["gap_log_signed", "gap_prob_signed"]

for wname, layers in GAP_WINDOWS.items():
    feats = []

    for gcol in base_gap_cols:
        cols = [f"{gcol}_L{l}" for l in layers if f"{gcol}_L{l}" in gap_wide.columns]
        if not cols:
            continue

        X = gap_wide[cols].values.astype(float)
        gap_wide[f"{wname}_{gcol}_mean"] = np.nanmean(X, axis=1)
        gap_wide[f"{wname}_{gcol}_max"] = np.nanmax(X, axis=1)
        gap_wide[f"{wname}_{gcol}_min"] = np.nanmin(X, axis=1)
        gap_wide[f"{wname}_{gcol}_last"] = X[:, -1]
        gap_wide[f"{wname}_{gcol}_slope"] = X[:, -1] - X[:, 0]
        gap_wide[f"{wname}_{gcol}_range"] = np.nanmax(X, axis=1) - np.nanmin(X, axis=1)

        feats += [
            f"{wname}_{gcol}_mean",
            f"{wname}_{gcol}_max",
            f"{wname}_{gcol}_min",
            f"{wname}_{gcol}_last",
            f"{wname}_{gcol}_slope",
            f"{wname}_{gcol}_range",
        ]

    # Winner stability / switches.
    wcols = [f"winner_L{l}" for l in layers if f"winner_L{l}" in gap_wide.columns]
    if wcols:
        switch_count = []
        dominant_frac = []
        first_winner = []
        last_winner = []

        for _, row in gap_wide[wcols].iterrows():
            vals = [x for x in row.values.tolist() if isinstance(x, str)]
            if len(vals) == 0:
                switch_count.append(np.nan)
                dominant_frac.append(np.nan)
                first_winner.append("")
                last_winner.append("")
                continue

            sw = sum(vals[i] != vals[i-1] for i in range(1, len(vals)))
            vc = pd.Series(vals).value_counts(normalize=True)

            switch_count.append(float(sw))
            dominant_frac.append(float(vc.iloc[0]))
            first_winner.append(vals[0])
            last_winner.append(vals[-1])

        gap_wide[f"{wname}_winner_switches"] = switch_count
        gap_wide[f"{wname}_winner_dominant_frac"] = dominant_frac
        gap_wide[f"{wname}_winner_first"] = first_winner
        gap_wide[f"{wname}_winner_last"] = last_winner

        feats += [f"{wname}_winner_switches", f"{wname}_winner_dominant_frac"]

    feature_cols_by_window[wname] = feats

# Drop rows with missing core gap features for main window.
main_window = "G_20_25"
main_feats = feature_cols_by_window.get(main_window, [])
if not main_feats:
    raise RuntimeError("No gap features built for main window G_20_25.")

# Fill missing numeric features.
for c in set(sum(feature_cols_by_window.values(), [])):
    if c in gap_wide.columns:
        med = gap_wide[c].median()
        gap_wide[c] = gap_wide[c].fillna(0.0 if not np.isfinite(med) else med)

# -----------------------------
# REGRESSION: GAP -> DeltaU
# -----------------------------

model_rows = []
pred_table = gap_wide[[id_col, "DeltaU_PC1"] + [c for c in [phase_col, condition_col] if c is not None and c in gap_wide.columns]].copy()

feature_sets = {}
for wname, feats in feature_cols_by_window.items():
    if feats:
        feature_sets[f"window_{wname}"] = feats

# Combined layered windows.
combo_20_22_23_25 = []
for k in ["G_20_22", "G_23_25"]:
    combo_20_22_23_25 += feature_cols_by_window.get(k, [])
if combo_20_22_23_25:
    feature_sets["combo_G_20_22_plus_23_25"] = list(dict.fromkeys(combo_20_22_23_25))

combo_all = []
for k in ["G_7_19", "G_20_22", "G_23_25"]:
    combo_all += feature_cols_by_window.get(k, [])
if combo_all:
    feature_sets["combo_G_7_19_20_22_23_25"] = list(dict.fromkeys(combo_all))

for name, feats in feature_sets.items():
    feats = [c for c in feats if c in gap_wide.columns]
    if not feats:
        continue

    try:
        pred, used_group = cv_predict_reg(gap_wide, feats, "DeltaU_PC1", group_col=id_col)
        m = metrics_reg(gap_wide["DeltaU_PC1"], pred)
        row = {
            "model": name,
            "n_features": len(feats),
            "group_col": used_group,
            "features": "|".join(feats),
            **m,
        }
        model_rows.append(row)
        pred_table[f"pred_{name}"] = pred
    except Exception as e:
        model_rows.append({
            "model": name,
            "status": "error",
            "error": str(e),
            "n_features": len(feats),
            "features": "|".join(feats),
        })

model_summary = pd.DataFrame(model_rows)
valid_models = model_summary[np.isfinite(model_summary.get("r2", np.nan))].copy()

# Direct correlations of key scalar candidates.
direct_rows = []
key_direct_cols = []
for w in ["G_20_22", "G_23_25", "G_20_25", "G_7_25"]:
    for g in ["gap_log", "gap_prob", "gap_log_signed", "gap_prob_signed"]:
        for stat in ["mean", "last", "max", "min", "slope"]:
            c = f"{w}_{g}_{stat}"
            if c in gap_wide.columns:
                key_direct_cols.append(c)

for c in key_direct_cols:
    direct_rows.append({
        "feature": c,
        "corr_with_DeltaU": pearsonr_safe(gap_wide[c], gap_wide["DeltaU_PC1"]),
        "r2_linear_fit": float(r2_score(
            gap_wide["DeltaU_PC1"],
            np.poly1d(np.polyfit(gap_wide[c].values, gap_wide["DeltaU_PC1"].values, 1))(gap_wide[c].values)
        )) if np.std(gap_wide[c].values) > 1e-12 else np.nan,
    })

direct_corr = pd.DataFrame(direct_rows).sort_values("corr_with_DeltaU", key=lambda s: np.abs(s), ascending=False)

# -----------------------------
# PHASE / LAYER PROFILES
# -----------------------------

phase_profile = pd.DataFrame()

if phase_col is not None and phase_col in gap_wide.columns:
    phase_metrics = ["DeltaU_PC1"]
    for c in [
        "G_20_25_gap_log_mean",
        "G_20_25_gap_log_signed_mean",
        "G_20_25_gap_prob_mean",
        "G_20_25_winner_switches",
        "G_20_25_winner_dominant_frac",
    ]:
        if c in gap_wide.columns:
            phase_metrics.append(c)

    phase_profile = gap_wide.groupby(phase_col)[phase_metrics].agg(["mean", "std", "count"])
    phase_profile.columns = ["__".join(col).strip() for col in phase_profile.columns.values]
    phase_profile = phase_profile.reset_index()

# Layer profile.
layer_profile_rows = []
for l in sorted(gap_long["layer"].unique()):
    sub = gap_long[gap_long["layer"] == l]
    row = {
        "layer": int(l),
        "n": int(len(sub)),
        "gap_log_mean": float(sub["gap_log"].mean()) if "gap_log" in sub.columns else np.nan,
        "gap_prob_mean": float(sub["gap_prob"].mean()) if "gap_prob" in sub.columns else np.nan,
    }

    if "gap_log_signed" in sub.columns:
        row["gap_log_signed_mean"] = float(sub["gap_log_signed"].mean())
        row["gap_prob_signed_mean"] = float(sub["gap_prob_signed"].mean())

    # Winner entropy.
    vc = sub["winner"].value_counts(normalize=True)
    ent = -float(np.sum(vc.values * np.log(np.clip(vc.values, 1e-12, 1.0)))) if len(vc) else np.nan
    row["winner_entropy"] = ent

    layer_profile_rows.append(row)

layer_profile = pd.DataFrame(layer_profile_rows)

# -----------------------------
# CLASSIFICATION: phase / sign
# -----------------------------

clf_rows = []

if phase_col is not None and phase_col in gap_wide.columns:
    # Binary: critical vs non-critical if available.
    if "critical" in set(gap_wide[phase_col].astype(str)):
        gap_wide["is_critical"] = (gap_wide[phase_col].astype(str) == "critical").astype(int)

        for name, feats in feature_sets.items():
            feats = [c for c in feats if c in gap_wide.columns]
            if not feats:
                continue
            try:
                prob, used_group = cv_predict_clf(gap_wide, feats, "is_critical", group_col=id_col)
                y = gap_wide["is_critical"].values.astype(int)
                pred = (prob >= 0.5).astype(int)
                clf_rows.append({
                    "target": "is_critical",
                    "model": name,
                    "n_features": len(feats),
                    "group_col": used_group,
                    "auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else np.nan,
                    "accuracy": float(accuracy_score(y, pred)),
                    "f1": float(f1_score(y, pred, zero_division=0)),
                    "mean_prob": float(np.mean(prob)),
                })
            except Exception as e:
                clf_rows.append({
                    "target": "is_critical",
                    "model": name,
                    "status": "error",
                    "error": str(e),
                })

clf_summary = pd.DataFrame(clf_rows)

# -----------------------------
# SHUFFLE CONTROL
# -----------------------------

rng = np.random.default_rng(RANDOM_SEED)
shuffle_rows = []

# Shuffle gap values across layers within each trajectory, then rebuild G20_25-like features.
available_gap_layers = sorted(gap_long["layer"].unique().tolist())

if len(available_gap_layers) >= 5:
    for s in range(N_SHUFFLES):
        shuf = gap_long.copy()

        for sid, idxs in shuf.groupby(id_col).groups.items():
            idxs = list(idxs)
            vals = shuf.loc[idxs, "gap_log"].values.copy()
            rng.shuffle(vals)
            shuf.loc[idxs, "gap_log_shuf"] = vals

            if "gap_log_signed" in shuf.columns:
                vals2 = shuf.loc[idxs, "gap_log_signed"].values.copy()
                rng.shuffle(vals2)
                shuf.loc[idxs, "gap_log_signed_shuf"] = vals2

        swide = gap_wide[[id_col, "DeltaU_PC1"]].copy()

        for gcol in ["gap_log_shuf", "gap_log_signed_shuf"]:
            if gcol not in shuf.columns:
                continue
            piv = shuf.pivot(index=id_col, columns="layer", values=gcol)
            piv.columns = [f"{gcol}_L{int(l)}" for l in piv.columns]
            swide = swide.merge(piv.reset_index(), on=id_col, how="left")

        feats = []
        layers = GAP_WINDOWS["G_20_25"]
        for gcol in ["gap_log_shuf", "gap_log_signed_shuf"]:
            cols = [f"{gcol}_L{l}" for l in layers if f"{gcol}_L{l}" in swide.columns]
            if not cols:
                continue
            X = swide[cols].values.astype(float)
            for stat_name, arr in [
                ("mean", np.nanmean(X, axis=1)),
                ("max", np.nanmax(X, axis=1)),
                ("min", np.nanmin(X, axis=1)),
                ("last", X[:, -1]),
                ("slope", X[:, -1] - X[:, 0]),
            ]:
                c = f"SHUF_G20_25_{gcol}_{stat_name}"
                swide[c] = arr
                swide[c] = swide[c].fillna(swide[c].median())
                feats.append(c)

        if feats:
            try:
                pred, used_group = cv_predict_reg(swide, feats, "DeltaU_PC1", group_col=id_col)
                m = metrics_reg(swide["DeltaU_PC1"], pred)
                shuffle_rows.append({
                    "shuffle_id": s,
                    "n_features": len(feats),
                    "group_col": used_group,
                    **m,
                })
            except Exception as e:
                shuffle_rows.append({
                    "shuffle_id": s,
                    "status": "error",
                    "error": str(e),
                })

shuffle_summary = pd.DataFrame(shuffle_rows)

# -----------------------------
# SUMMARY / VERDICT
# -----------------------------

best_model = valid_models.sort_values("r2", ascending=False).iloc[0].to_dict() if len(valid_models) else None

best_direct = direct_corr.iloc[0].to_dict() if len(direct_corr) else None

main_model_row = None
sub = valid_models[valid_models["model"] == "window_G_20_25"]
if len(sub):
    main_model_row = sub.sort_values("r2", ascending=False).iloc[0].to_dict()

main_r2 = float(main_model_row["r2"]) if main_model_row is not None else np.nan
main_corr = float(main_model_row["corr"]) if main_model_row is not None else np.nan

best_r2 = float(best_model["r2"]) if best_model is not None else np.nan
best_corr = float(best_model["corr"]) if best_model is not None else np.nan

shuffle_mean_r2 = float(shuffle_summary["r2"].mean()) if "r2" in shuffle_summary.columns and len(shuffle_summary) else np.nan
gain_vs_shuffle = main_r2 - shuffle_mean_r2 if np.isfinite(main_r2) and np.isfinite(shuffle_mean_r2) else np.nan

phase_check_signed = {}
if "G_20_25_gap_log_signed_mean" in gap_wide.columns:
    phase_check_signed = phase_order_check(gap_wide, "G_20_25_gap_log_signed_mean", phase_col=phase_col)

phase_check_abs = {}
if "G_20_25_gap_log_mean" in gap_wide.columns:
    phase_check_abs = phase_order_check(gap_wide, "G_20_25_gap_log_mean", phase_col=phase_col)

# Verdict logic:
# Multi-path mode wants gap magnitude / winner structure. Binary fallback is weaker.
if np.isfinite(main_r2) and abs(main_corr) >= 0.80 and main_r2 >= 0.60:
    if mode == "multi_path_scores":
        verdict = "PASS-Strong"
    else:
        verdict = "PASS-BinaryStrong"
elif np.isfinite(main_r2) and abs(main_corr) >= 0.60 and main_r2 >= 0.35:
    verdict = "PASS-Lite"
else:
    verdict = "INCONCLUSIVE-or-FAIL"

summary = {
    "primary_csv": PRIMARY_CSV,
    "optional_path_score_csv": OPTIONAL_PATH_SCORE_CSV,
    "mode": mode,
    "n_rows_raw": int(len(df)),
    "n_trajectories": int(len(gap_wide)),
    "n_gap_long_rows": int(len(gap_long)),
    "id_col": id_col,
    "phase_col": phase_col,
    "condition_col": condition_col,
    "available_layers": available_layers,
    "r_curr_col": r_curr_col,
    "r_next_col": r_next_col,
    "r_cols_for_DeltaU": r_cols,
    "delta_used_cols": delta_used_cols,
    "DeltaU_PC1_explained_variance": float(pca.explained_variance_ratio_[0]),
    "DeltaU_PC123_explained_variance": float(np.sum(pca.explained_variance_ratio_[:min(3, len(pca.explained_variance_ratio_))])),
    "best_model": best_model,
    "main_G20_25_model": main_model_row,
    "best_direct_feature": best_direct,
    "main_G20_25_r2": main_r2,
    "main_G20_25_corr": main_corr,
    "best_r2": best_r2,
    "best_corr": best_corr,
    "shuffle_r2_mean": shuffle_mean_r2,
    "gain_vs_shuffle_mean": gain_vs_shuffle,
    "phase_check_signed_gap": phase_check_signed,
    "phase_check_abs_gap": phase_check_abs,
    "verdict": verdict,
    "verdict_note": (
        "Binary fallback can only test two-path clean/conflict advantage. "
        "For full geodesic competition, provide phasemap8a2_path_scores.csv with sample_id, layer, path_name, path_score."
    ),
}

# -----------------------------
# SAVE
# -----------------------------

gap_long.to_csv(OUTPUT_DIR / "phasemap8a2_layer_gap_long.csv", index=False)
gap_wide.to_csv(OUTPUT_DIR / "phasemap8a2_gap_features.csv", index=False)
model_summary.sort_values("r2", ascending=False, na_position="last").to_csv(
    OUTPUT_DIR / "phasemap8a2_model_summary.csv", index=False
)
direct_corr.to_csv(OUTPUT_DIR / "phasemap8a2_direct_correlations.csv", index=False)
phase_profile.to_csv(OUTPUT_DIR / "phasemap8a2_phase_profile.csv", index=False)
layer_profile.to_csv(OUTPUT_DIR / "phasemap8a2_layer_profile.csv", index=False)
clf_summary.to_csv(OUTPUT_DIR / "phasemap8a2_classification_summary.csv", index=False)
shuffle_summary.to_csv(OUTPUT_DIR / "phasemap8a2_shuffle_control.csv", index=False)
pred_table.to_csv(OUTPUT_DIR / "phasemap8a2_predictions.csv", index=False)

with open(OUTPUT_DIR / "phasemap8a2_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n================ PhaseMap-8A.2 Summary ================")
print(json.dumps(summary, ensure_ascii=False, indent=2))

print("\nTop regression models:")
if len(valid_models):
    print(valid_models.sort_values("r2", ascending=False)[["model", "n_features", "r2", "corr", "rmse", "mae"]].head(20).to_string(index=False))
else:
    print("No valid regression models.")

print("\nTop direct correlations:")
if len(direct_corr):
    print(direct_corr.head(20).to_string(index=False))
else:
    print("No direct correlations.")

print("\nPhase profile:")
if len(phase_profile):
    print(phase_profile.to_string(index=False))
else:
    print("No phase profile.")

print("\nLayer profile:")
print(layer_profile.to_string(index=False))

print("\nShuffle control:")
if len(shuffle_summary) and "r2" in shuffle_summary.columns:
    print(shuffle_summary[["r2", "corr", "rmse", "mae"]].describe().to_string())
else:
    print("No shuffle control.")

print("\nOutputs saved to:", OUTPUT_DIR.resolve())
