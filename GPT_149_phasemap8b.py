# ============================================================
# PhaseMap-8B
# Multi-Path Geodesic Dominance Audit
#
# Goal:
#   Verify whether the strong 8A.2 binary result:
#
#       DeltaU ≈ Gap(clean, conflict)
#
#   generalizes to a true multi-path/geodesic setting:
#
#       DeltaU ≈ lambda_1 - lambda_2
#
#   where lambda_i are candidate explanation-path scores.
#
# Why 8B:
#   8A.2 used binary fallback from R_l. That is strong but may only
#   prove clean-vs-conflict projection. 8B constructs multiple
#   candidate path/basin scores from available token/probe columns
#   or from an optional user-provided path score file.
#
# Fixed primary input:
#   phasemap6b1_correction_dataset.csv
#
# Optional stronger input:
#   phasemap8b_path_scores.csv
#
#   Required columns:
#     sample_id, layer, path_name, path_score
#
#   Recommended path_name examples:
#     clean_closure
#     conflict_direct
#     override_rule
#     weak_distractor
#     alternative_basin
#
# If optional file is absent:
#   Script tries to infer multi-path scores from columns such as:
#     clean_score, conflict_score, closure_score, override_score,
#     stable_score, competition_score, ...
#
# If no true multi-path score columns exist:
#   It falls back to a pseudo-multipath construction from:
#     R_l, boundary distance, operator cluster/phase if available.
#   This fallback is weaker and should be treated as exploratory.
#
# Outputs:
#   phasemap8b_outputs/
#     phasemap8b_summary.json
#     phasemap8b_path_gap_long.csv
#     phasemap8b_gap_features.csv
#     phasemap8b_model_summary.csv
#     phasemap8b_direct_correlations.csv
#     phasemap8b_phase_profile.csv
#     phasemap8b_winner_profile.csv
#     phasemap8b_shuffle_control.csv
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
)

warnings.filterwarnings("ignore")

# -----------------------------
# CONFIG
# -----------------------------

PRIMARY_CSV = r"C:\Windows\System32\phasemap6b1_outputs\phasemap6b1_correction_dataset.csv"
OPTIONAL_PATH_SCORE_CSV = r"phasemap8b_path_scores.csv"

OUTPUT_DIR = Path("phasemap8b_outputs")
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

# Candidate score columns if optional path score CSV is absent.
PATH_SCORE_KEYWORDS = [
    "clean", "conflict", "closure", "override", "update", "exception",
    "stable", "competition", "positive", "negative", "critical",
    "evidence", "rule", "source", "answer", "basin", "path"
]

EXCLUDE_SCORE_COLS = [
    "sample_id", "idx", "graph_id", "prompt_id", "case_id",
    "layer", "condition", "phase", "phase_target",
]

# -----------------------------
# HELPERS
# -----------------------------

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

def fit_deltaU_from_R(wide_df, r_cols):
    X = wide_df[r_cols].values.astype(np.float32)
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

    return u.astype(np.float32), pca, used_cols

def softmax_np(scores):
    s = np.asarray(scores, dtype=float)
    s = s - np.nanmax(s)
    p = np.exp(np.clip(s, -60, 60))
    denom = np.nansum(p)
    if denom <= 0 or not np.isfinite(denom):
        return np.ones_like(p) / len(p)
    return p / denom

def entropy_np(p):
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p)]
    p = p[p > 0]
    if len(p) == 0:
        return np.nan
    return float(-np.sum(p * np.log(p)))

def find_path_score_columns(df):
    cols = []
    for c in df.columns:
        cl = c.lower()
        if c in EXCLUDE_SCORE_COLS:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if c.startswith("op_pc"):
            continue
        if c in ["R_l", "R_l1", "R", "R_next", "boundary_dist_l", "boundary_dist_l1"]:
            continue
        if any(k in cl for k in PATH_SCORE_KEYWORDS) and any(s in cl for s in ["score", "logit", "margin", "prob"]):
            cols.append(c)
    return cols

def add_window_features(gap_wide, gap_long, id_col):
    feature_cols_by_window = {}

    value_cols = [
        "gap_log", "gap_prob", "gap_signed", "gap_prob_signed",
        "top1_score", "top2_score", "path_entropy", "top1_prob", "top2_prob",
    ]
    value_cols = [c for c in value_cols if c in gap_long.columns]

    for value_col in value_cols:
        piv = gap_long.pivot(index=id_col, columns="layer", values=value_col)
        piv.columns = [f"{value_col}_L{int(l)}" for l in piv.columns]
        gap_wide = gap_wide.merge(piv.reset_index(), on=id_col, how="left")

    winner_piv = gap_long.pivot(index=id_col, columns="layer", values="winner")
    winner_piv.columns = [f"winner_L{int(l)}" for l in winner_piv.columns]
    gap_wide = gap_wide.merge(winner_piv.reset_index(), on=id_col, how="left")

    for wname, layers in GAP_WINDOWS.items():
        feats = []

        for value_col in value_cols:
            cols = [f"{value_col}_L{l}" for l in layers if f"{value_col}_L{l}" in gap_wide.columns]
            if not cols:
                continue
            X = gap_wide[cols].values.astype(float)

            stat_defs = {
                "mean": np.nanmean(X, axis=1),
                "max": np.nanmax(X, axis=1),
                "min": np.nanmin(X, axis=1),
                "last": X[:, -1],
                "first": X[:, 0],
                "slope": X[:, -1] - X[:, 0],
                "range": np.nanmax(X, axis=1) - np.nanmin(X, axis=1),
                "std": np.nanstd(X, axis=1),
            }
            for stat, arr in stat_defs.items():
                outc = f"{wname}_{value_col}_{stat}"
                gap_wide[outc] = arr
                feats.append(outc)

        wcols = [f"winner_L{l}" for l in layers if f"winner_L{l}" in gap_wide.columns]
        if wcols:
            switch_count = []
            dominant_frac = []
            winner_entropy = []
            first_winner = []
            last_winner = []

            for _, row in gap_wide[wcols].iterrows():
                vals = [x for x in row.values.tolist() if isinstance(x, str) and x != ""]
                if not vals:
                    switch_count.append(np.nan)
                    dominant_frac.append(np.nan)
                    winner_entropy.append(np.nan)
                    first_winner.append("")
                    last_winner.append("")
                    continue
                sw = sum(vals[i] != vals[i - 1] for i in range(1, len(vals)))
                vc = pd.Series(vals).value_counts(normalize=True)
                switch_count.append(float(sw))
                dominant_frac.append(float(vc.iloc[0]))
                winner_entropy.append(entropy_np(vc.values))
                first_winner.append(vals[0])
                last_winner.append(vals[-1])

            for name, arr in [
                ("winner_switches", switch_count),
                ("winner_dominant_frac", dominant_frac),
                ("winner_entropy", winner_entropy),
            ]:
                outc = f"{wname}_{name}"
                gap_wide[outc] = arr
                feats.append(outc)

            gap_wide[f"{wname}_winner_first"] = first_winner
            gap_wide[f"{wname}_winner_last"] = last_winner

        feature_cols_by_window[wname] = feats

    for c in set(sum(feature_cols_by_window.values(), [])):
        if c in gap_wide.columns:
            med = gap_wide[c].median()
            gap_wide[c] = gap_wide[c].fillna(0.0 if not np.isfinite(med) else med)

    return gap_wide, feature_cols_by_window

# -----------------------------
# LOAD PRIMARY
# -----------------------------

if not Path(PRIMARY_CSV).exists():
    raise FileNotFoundError(
        f"Cannot find {PRIMARY_CSV}. Put this script in the same folder as {PRIMARY_CSV}."
    )

df = pd.read_csv(PRIMARY_CSV)
df.columns = [str(c).strip() for c in df.columns]

if "layer" not in df.columns:
    raise RuntimeError("Primary CSV must contain layer column.")

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
    raise RuntimeError("Need R_l or R_next/R_l1 to define DeltaU.")

meta_cols = [id_col]
for c in [phase_col, condition_col, "heuristic_operator", "operator_cluster"]:
    if c is not None and c in df.columns and c not in meta_cols:
        meta_cols.append(c)

meta = df.groupby(id_col, as_index=False).first()[meta_cols]

agg_cols = []
if r_curr_col is not None:
    agg_cols.append(r_curr_col)
if r_next_col is not None:
    agg_cols.append(r_next_col)

# Include possible score columns for aggregation.
candidate_score_cols = find_path_score_columns(df)
agg_cols += candidate_score_cols
agg_cols = list(dict.fromkeys([c for c in agg_cols if c in df.columns]))

num = df.groupby([id_col, "layer"], as_index=False)[agg_cols].mean()
available_layers = sorted(num["layer"].unique().tolist())

wide = meta.copy()

if r_curr_col is not None:
    piv = num.pivot(index=id_col, columns="layer", values=r_curr_col)
    piv.columns = [f"R_L{int(l)}" for l in piv.columns]
    wide = wide.merge(piv.reset_index(), on=id_col, how="left")

if r_next_col is not None:
    piv = num.pivot(index=id_col, columns="layer", values=r_next_col)
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
wide["DeltaU_PC1"], pca, delta_used_cols = fit_deltaU_from_R(wide, r_cols)

# -----------------------------
# BUILD MULTI-PATH SCORES
# -----------------------------

mode = None
gap_long_rows = []
path_score_cols_used = []

optional_path = Path(OPTIONAL_PATH_SCORE_CSV)

if optional_path.exists():
    ps = pd.read_csv(optional_path)
    ps.columns = [str(c).strip() for c in ps.columns]

    if id_col not in ps.columns and "sample_id" in ps.columns:
        ps = ps.rename(columns={"sample_id": id_col})

    required = {id_col, "layer", "path_name", "path_score"}
    if required.issubset(set(ps.columns)):
        mode = "true_multipath_file"
        ps["layer"] = ps["layer"].astype(int)
        ps = ps[ps[id_col].isin(set(wide[id_col]))].copy()

        for (sid, layer), g in ps.groupby([id_col, "layer"]):
            scores = g["path_score"].astype(float).values
            names = g["path_name"].astype(str).values
            if len(scores) < 2:
                continue

            order = np.argsort(scores)[::-1]
            probs = softmax_np(scores)
            p_order = probs[order]

            top1 = scores[order[0]]
            top2 = scores[order[1]]
            top1p = p_order[0]
            top2p = p_order[1]

            gap_long_rows.append({
                id_col: sid,
                "layer": int(layer),
                "winner": names[order[0]],
                "runner_up": names[order[1]],
                "gap_log": float(top1 - top2),
                "gap_prob": float(top1p - top2p),
                "gap_signed": float(top1 - top2),
                "gap_prob_signed": float(top1p - top2p),
                "top1_score": float(top1),
                "top2_score": float(top2),
                "top1_prob": float(top1p),
                "top2_prob": float(top2p),
                "path_entropy": entropy_np(probs),
                "n_paths": int(len(scores)),
                "path_source": "file",
            })

if not gap_long_rows and candidate_score_cols:
    mode = "inferred_score_columns"
    path_score_cols_used = candidate_score_cols

    # Long format from score columns.
    for _, row in num.iterrows():
        sid = row[id_col]
        layer = int(row["layer"])
        scores = []
        names = []
        for c in candidate_score_cols:
            val = row.get(c, np.nan)
            if np.isfinite(val):
                scores.append(float(val))
                names.append(c)
        if len(scores) < 2:
            continue

        scores = np.asarray(scores, dtype=float)
        names = np.asarray(names, dtype=str)
        order = np.argsort(scores)[::-1]
        probs = softmax_np(scores)
        p_order = probs[order]

        top1 = scores[order[0]]
        top2 = scores[order[1]]

        gap_long_rows.append({
            id_col: sid,
            "layer": layer,
            "winner": names[order[0]],
            "runner_up": names[order[1]],
            "gap_log": float(top1 - top2),
            "gap_prob": float(p_order[0] - p_order[1]),
            "gap_signed": float(top1 - top2),
            "gap_prob_signed": float(p_order[0] - p_order[1]),
            "top1_score": float(top1),
            "top2_score": float(top2),
            "top1_prob": float(p_order[0]),
            "top2_prob": float(p_order[1]),
            "path_entropy": entropy_np(probs),
            "n_paths": int(len(scores)),
            "path_source": "columns",
        })

if not gap_long_rows:
    # Exploratory pseudo-multipath fallback.
    # Build three pseudo path scores:
    #   clean_path    = +R
    #   conflict_path = -R
    #   boundary_path = -|R|  (near-boundary / ambiguous path)
    #
    # This is NOT a true multi-path audit, but it tests whether adding
    # a third near-boundary basin changes the binary interpretation.
    mode = "pseudo_multipath_from_R"

    for _, row in num.iterrows():
        sid = row[id_col]
        layer = int(row["layer"])

        if r_curr_col is not None and r_curr_col in row:
            R = row[r_curr_col]
        elif r_next_col is not None and r_next_col in row:
            R = row[r_next_col]
        else:
            continue

        if not np.isfinite(R):
            continue

        scores = np.asarray([R, -R, -abs(R)], dtype=float)
        names = np.asarray(["clean_path", "conflict_path", "boundary_path"], dtype=str)
        order = np.argsort(scores)[::-1]
        probs = softmax_np(scores)
        p_order = probs[order]

        gap_long_rows.append({
            id_col: sid,
            "layer": layer,
            "winner": names[order[0]],
            "runner_up": names[order[1]],
            "gap_log": float(scores[order[0]] - scores[order[1]]),
            "gap_prob": float(p_order[0] - p_order[1]),
            # Signed with clean-positive orientation.
            "gap_signed": float(R),
            "gap_prob_signed": float(2 * (1 / (1 + np.exp(-np.clip(R, -60, 60)))) - 1),
            "top1_score": float(scores[order[0]]),
            "top2_score": float(scores[order[1]]),
            "top1_prob": float(p_order[0]),
            "top2_prob": float(p_order[1]),
            "path_entropy": entropy_np(probs),
            "n_paths": 3,
            "path_source": "pseudo_R",
        })

gap_long = pd.DataFrame(gap_long_rows)
gap_long = gap_long[gap_long[id_col].isin(set(wide[id_col]))].copy()

if len(gap_long) == 0:
    raise RuntimeError("Could not construct any path-gap rows.")

# -----------------------------
# FEATURES
# -----------------------------

base_cols = [id_col, "DeltaU_PC1"]
for c in [phase_col, condition_col]:
    if c is not None and c in wide.columns and c not in base_cols:
        base_cols.append(c)

gap_wide = wide[base_cols].copy()
gap_wide, feature_cols_by_window = add_window_features(gap_wide, gap_long, id_col)

# -----------------------------
# MODELS
# -----------------------------

feature_sets = {}
for w, feats in feature_cols_by_window.items():
    if feats:
        feature_sets[f"window_{w}"] = feats

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

model_rows = []
pred_table = gap_wide[[id_col, "DeltaU_PC1"] + [c for c in [phase_col, condition_col] if c is not None and c in gap_wide.columns]].copy()

for name, feats in feature_sets.items():
    feats = [c for c in feats if c in gap_wide.columns]
    if not feats:
        continue
    try:
        pred, used_group = cv_predict_reg(gap_wide, feats, "DeltaU_PC1", group_col=id_col)
        m = reg_metrics(gap_wide["DeltaU_PC1"], pred)
        model_rows.append({
            "model": name,
            "n_features": len(feats),
            "group_col": used_group,
            "features": "|".join(feats),
            **m,
        })
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

# Direct correlations.
direct_rows = []
for c in sum(feature_cols_by_window.values(), []):
    if c not in gap_wide.columns:
        continue
    vals = gap_wide[c].values
    if np.std(vals) < 1e-12:
        continue
    try:
        corr = pearsonr_safe(vals, gap_wide["DeltaU_PC1"])
        coef = np.polyfit(vals, gap_wide["DeltaU_PC1"], 1)
        pred = np.poly1d(coef)(vals)
        direct_rows.append({
            "feature": c,
            "corr_with_DeltaU": corr,
            "r2_linear_fit": float(r2_score(gap_wide["DeltaU_PC1"], pred)),
        })
    except Exception:
        pass

direct_corr = pd.DataFrame(direct_rows)
if len(direct_corr):
    direct_corr = direct_corr.sort_values("corr_with_DeltaU", key=lambda s: np.abs(s), ascending=False)

# -----------------------------
# PROFILES
# -----------------------------

phase_profile = pd.DataFrame()
if phase_col is not None and phase_col in gap_wide.columns:
    prof_cols = ["DeltaU_PC1"]
    for c in [
        "G_20_25_gap_log_mean",
        "G_20_25_gap_signed_mean",
        "G_20_25_gap_prob_mean",
        "G_20_25_path_entropy_mean",
        "G_20_25_winner_switches",
        "G_20_25_winner_dominant_frac",
    ]:
        if c in gap_wide.columns:
            prof_cols.append(c)

    phase_profile = gap_wide.groupby(phase_col)[prof_cols].agg(["mean", "std", "count"])
    phase_profile.columns = ["__".join(col).strip() for col in phase_profile.columns.values]
    phase_profile = phase_profile.reset_index()

winner_profile_rows = []
for w in ["G_20_22", "G_23_25", "G_20_25"]:
    last_col = f"{w}_winner_last"
    first_col = f"{w}_winner_first"
    if last_col in gap_wide.columns:
        vc = gap_wide[last_col].value_counts(normalize=True)
        for winner, frac in vc.items():
            winner_profile_rows.append({
                "window": w,
                "position": "last",
                "winner": winner,
                "frac": float(frac),
                "count": int((gap_wide[last_col] == winner).sum()),
            })
    if first_col in gap_wide.columns:
        vc = gap_wide[first_col].value_counts(normalize=True)
        for winner, frac in vc.items():
            winner_profile_rows.append({
                "window": w,
                "position": "first",
                "winner": winner,
                "frac": float(frac),
                "count": int((gap_wide[first_col] == winner).sum()),
            })

winner_profile = pd.DataFrame(winner_profile_rows)

layer_profile_rows = []
for l in sorted(gap_long["layer"].unique()):
    sub = gap_long[gap_long["layer"] == l]
    row = {
        "layer": int(l),
        "n": int(len(sub)),
        "gap_log_mean": float(sub["gap_log"].mean()),
        "gap_log_std": float(sub["gap_log"].std()),
        "gap_prob_mean": float(sub["gap_prob"].mean()),
        "path_entropy_mean": float(sub["path_entropy"].mean()),
        "n_paths_mean": float(sub["n_paths"].mean()),
    }
    vc = sub["winner"].value_counts(normalize=True)
    row["winner_entropy"] = entropy_np(vc.values)
    if len(vc):
        row["top_winner"] = str(vc.index[0])
        row["top_winner_frac"] = float(vc.iloc[0])
    layer_profile_rows.append(row)

layer_profile = pd.DataFrame(layer_profile_rows)

# -----------------------------
# SHUFFLE CONTROL
# -----------------------------

rng = np.random.default_rng(RANDOM_SEED)
shuffle_rows = []

if len(gap_long["layer"].unique()) >= 5:
    for s in range(N_SHUFFLES):
        sh = gap_long.copy()
        for sid, idxs in sh.groupby(id_col).groups.items():
            idxs = list(idxs)
            for col in ["gap_log", "gap_prob", "gap_signed", "path_entropy"]:
                if col in sh.columns:
                    vals = sh.loc[idxs, col].values.copy()
                    rng.shuffle(vals)
                    sh.loc[idxs, f"{col}_shuf"] = vals

        swide = gap_wide[[id_col, "DeltaU_PC1"]].copy()

        for col in ["gap_log_shuf", "gap_prob_shuf", "gap_signed_shuf", "path_entropy_shuf"]:
            if col not in sh.columns:
                continue
            piv = sh.pivot(index=id_col, columns="layer", values=col)
            piv.columns = [f"{col}_L{int(l)}" for l in piv.columns]
            swide = swide.merge(piv.reset_index(), on=id_col, how="left")

        feats = []
        layers = GAP_WINDOWS["G_20_25"]
        for col in ["gap_log_shuf", "gap_prob_shuf", "gap_signed_shuf", "path_entropy_shuf"]:
            cols = [f"{col}_L{l}" for l in layers if f"{col}_L{l}" in swide.columns]
            if not cols:
                continue
            X = swide[cols].values.astype(float)
            for stat, arr in {
                "mean": np.nanmean(X, axis=1),
                "max": np.nanmax(X, axis=1),
                "min": np.nanmin(X, axis=1),
                "last": X[:, -1],
                "slope": X[:, -1] - X[:, 0],
                "range": np.nanmax(X, axis=1) - np.nanmin(X, axis=1),
            }.items():
                outc = f"SHUF_G20_25_{col}_{stat}"
                swide[outc] = arr
                med = swide[outc].median()
                swide[outc] = swide[outc].fillna(0.0 if not np.isfinite(med) else med)
                feats.append(outc)

        if feats:
            try:
                pred, used_group = cv_predict_reg(swide, feats, "DeltaU_PC1", group_col=id_col)
                m = reg_metrics(swide["DeltaU_PC1"], pred)
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
# VERDICT
# -----------------------------

best_model = valid_models.sort_values("r2", ascending=False).iloc[0].to_dict() if len(valid_models) else None
main_model = None
sub = valid_models[valid_models["model"] == "window_G_20_25"]
if len(sub):
    main_model = sub.sort_values("r2", ascending=False).iloc[0].to_dict()

best_direct = direct_corr.iloc[0].to_dict() if len(direct_corr) else None

main_r2 = float(main_model["r2"]) if main_model is not None else np.nan
main_corr = float(main_model["corr"]) if main_model is not None else np.nan
best_r2 = float(best_model["r2"]) if best_model is not None else np.nan
best_corr = float(best_model["corr"]) if best_model is not None else np.nan

shuffle_r2_mean = float(shuffle_summary["r2"].mean()) if len(shuffle_summary) and "r2" in shuffle_summary.columns else np.nan
gain_vs_shuffle = main_r2 - shuffle_r2_mean if np.isfinite(main_r2) and np.isfinite(shuffle_r2_mean) else np.nan

if mode == "true_multipath_file":
    if main_r2 >= 0.60 and abs(main_corr) >= 0.80 and gain_vs_shuffle > 0.10:
        verdict = "PASS-Strong"
    elif main_r2 >= 0.35 and abs(main_corr) >= 0.60:
        verdict = "PASS-Lite"
    else:
        verdict = "INCONCLUSIVE-or-FAIL"
elif mode == "inferred_score_columns":
    if main_r2 >= 0.60 and abs(main_corr) >= 0.80:
        verdict = "PASS-InferredStrong"
    elif main_r2 >= 0.35 and abs(main_corr) >= 0.60:
        verdict = "PASS-InferredLite"
    else:
        verdict = "INCONCLUSIVE-or-FAIL"
else:
    if main_r2 >= 0.60 and abs(main_corr) >= 0.80:
        verdict = "PASS-PseudoOnly"
    elif main_r2 >= 0.35 and abs(main_corr) >= 0.60:
        verdict = "PASS-PseudoLite"
    else:
        verdict = "INCONCLUSIVE-or-FAIL"

summary = {
    "primary_csv": PRIMARY_CSV,
    "optional_path_score_csv": OPTIONAL_PATH_SCORE_CSV,
    "mode": mode,
    "mode_warning": (
        "Only true_multipath_file is a decisive 8B test. "
        "inferred_score_columns is useful if score columns are real candidate path scores. "
        "pseudo_multipath_from_R is exploratory and mostly checks whether a third boundary path changes the binary picture."
    ),
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
    "path_score_cols_used": path_score_cols_used,
    "candidate_score_cols_detected": candidate_score_cols,
    "mean_n_paths": float(gap_long["n_paths"].mean()),
    "best_model": best_model,
    "main_G20_25_model": main_model,
    "best_direct_feature": best_direct,
    "main_G20_25_r2": main_r2,
    "main_G20_25_corr": main_corr,
    "best_r2": best_r2,
    "best_corr": best_corr,
    "shuffle_r2_mean": shuffle_r2_mean,
    "gain_vs_shuffle_mean": gain_vs_shuffle,
    "verdict": verdict,
}

# -----------------------------
# SAVE
# -----------------------------

gap_long.to_csv(OUTPUT_DIR / "phasemap8b_path_gap_long.csv", index=False)
gap_wide.to_csv(OUTPUT_DIR / "phasemap8b_gap_features.csv", index=False)
model_summary.sort_values("r2", ascending=False, na_position="last").to_csv(
    OUTPUT_DIR / "phasemap8b_model_summary.csv", index=False
)
direct_corr.to_csv(OUTPUT_DIR / "phasemap8b_direct_correlations.csv", index=False)
phase_profile.to_csv(OUTPUT_DIR / "phasemap8b_phase_profile.csv", index=False)
winner_profile.to_csv(OUTPUT_DIR / "phasemap8b_winner_profile.csv", index=False)
layer_profile.to_csv(OUTPUT_DIR / "phasemap8b_layer_profile.csv", index=False)
shuffle_summary.to_csv(OUTPUT_DIR / "phasemap8b_shuffle_control.csv", index=False)
pred_table.to_csv(OUTPUT_DIR / "phasemap8b_predictions.csv", index=False)

with open(OUTPUT_DIR / "phasemap8b_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n================ PhaseMap-8B Summary ================")
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

print("\nWinner profile:")
if len(winner_profile):
    print(winner_profile.head(40).to_string(index=False))
else:
    print("No winner profile.")

print("\nLayer profile:")
print(layer_profile.to_string(index=False))

print("\nShuffle control:")
if len(shuffle_summary) and "r2" in shuffle_summary.columns:
    print(shuffle_summary[["r2", "corr", "rmse", "mae"]].describe().to_string())
else:
    print("No shuffle control.")

print("\nOutputs saved to:", OUTPUT_DIR.resolve())
