# ============================================================
# PhaseMap-7F
# O-cont Origin and Causality Audit
#
# Goal:
#   PhaseMap-7E showed O_cont is crucial for:
#
#       Z_{l+1} = F(Z_l, O_l^cont, Z_l ⊗ O_l^cont)
#
#   But O_cont may be a posterior transition signature leaking
#   information from Z_{l+1}.
#
#   7F tests whether O_cont can be generated from pre-transition
#   information only:
#
#       O_hat = G(Z_l, dZ_l/history, Context)
#
#   and whether substituting O_hat into F still improves over
#   the state-only baseline.
#
# Fixed input:
#   phasemap6b1_correction_dataset.csv
#
# Outputs:
#   phasemap7f_outputs/
# ============================================================

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import clone

warnings.filterwarnings("ignore")

# -----------------------------
# CONFIG
# -----------------------------

INPUT_CSV = r"C:\Windows\System32\phasemap6b1_outputs\phasemap6b1_correction_dataset.csv"
OUTPUT_DIR = Path("phasemap7f_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_SPLITS = 5
RIDGE_ALPHA = 1.0

# -----------------------------
# HELPERS
# -----------------------------

def safe_cols(df, candidates):
    return [c for c in candidates if c in df.columns]

def macro_r2(y_true, y_pred):
    vals = []
    for j in range(y_true.shape[1]):
        try:
            vals.append(r2_score(y_true[:, j], y_pred[:, j]))
        except Exception:
            vals.append(np.nan)
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan

def macro_corr(y_true, y_pred):
    vals = []
    for j in range(y_true.shape[1]):
        a = y_true[:, j]
        b = y_pred[:, j]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        vals.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(vals)) if vals else np.nan

def metrics(y_true, y_pred, prefix=""):
    out = {
        prefix + "r2_macro": macro_r2(y_true, y_pred),
        prefix + "corr_macro": macro_corr(y_true, y_pred),
        prefix + "mse": float(mean_squared_error(y_true, y_pred)),
        prefix + "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        prefix + "mae": float(mean_absolute_error(y_true, y_pred)),
    }
    return out

def make_preprocessor(df, feature_cols):
    num_cols = []
    cat_cols = []
    for c in feature_cols:
        if c not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", numeric_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", categorical_pipe, cat_cols))

    if not transformers:
        raise ValueError("No valid feature columns found.")

    return ColumnTransformer(transformers), num_cols, cat_cols

def make_model(df, feature_cols):
    pre, num_cols, cat_cols = make_preprocessor(df, feature_cols)
    model = Pipeline([
        ("pre", pre),
        ("reg", MultiOutputRegressor(Ridge(alpha=RIDGE_ALPHA))),
    ])
    return model, num_cols, cat_cols

def get_cv(df):
    # Prefer sample/graph grouping when available.
    group_candidates = ["sample_id", "idx", "graph_id", "prompt_id", "case_id"]
    group_cols = safe_cols(df, group_candidates)
    if group_cols:
        g = df[group_cols[0]].astype(str).values
        n_unique = len(np.unique(g))
        n_splits = min(N_SPLITS, n_unique)
        if n_splits >= 2:
            return GroupKFold(n_splits=n_splits), g, group_cols[0]
    return KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED), None, None

def cv_predict(df, feature_cols, target_cols):
    X = df[feature_cols].copy()
    y = df[target_cols].values.astype(np.float32)

    cv, groups, group_col = get_cv(df)
    model, num_cols, cat_cols = make_model(df, feature_cols)

    if groups is None:
        pred = cross_val_predict(model, X, y, cv=cv)
    else:
        pred = cross_val_predict(model, X, y, cv=cv, groups=groups)

    return pred.astype(np.float32), {
        "group_col": group_col,
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "n_features": len(feature_cols),
        "n_targets": len(target_cols),
    }

def fit_predict_infold_two_stage(df, o_feature_cols, z_feature_builder, op_cols, z_target_cols):
    """
    Leakage-safe nested-ish two-stage CV:
      For each outer fold:
        1) Fit G: pre-state -> O_cont on train only.
        2) Predict O_hat for train and test.
        3) Fit F: Z + O_hat (+ optional bilinear features) -> Z_next on train only.
        4) Predict Z_next for test.
    """
    cv, groups, group_col = get_cv(df)
    splits = list(cv.split(df, groups=groups)) if groups is not None else list(cv.split(df))

    y_all = df[z_target_cols].values.astype(np.float32)
    pred_all = np.zeros_like(y_all, dtype=np.float32)
    o_pred_all = np.zeros((len(df), len(op_cols)), dtype=np.float32)

    fold_rows = []

    for fold_idx, (tr, te) in enumerate(splits):
        train_df = df.iloc[tr].reset_index(drop=True)
        test_df = df.iloc[te].reset_index(drop=True)

        # Stage G: predict O from pre-state only.
        G, _, _ = make_model(train_df, o_feature_cols)
        G.fit(train_df[o_feature_cols], train_df[op_cols].values.astype(np.float32))

        Ohat_train = G.predict(train_df[o_feature_cols]).astype(np.float32)
        Ohat_test = G.predict(test_df[o_feature_cols]).astype(np.float32)

        # Store out-of-fold O predictions.
        o_pred_all[te, :] = Ohat_test

        # Stage F: use predicted O only.
        train_aug = train_df.copy()
        test_aug = test_df.copy()

        for j, c in enumerate(op_cols):
            train_aug[f"ohat_{c}"] = Ohat_train[:, j]
            test_aug[f"ohat_{c}"] = Ohat_test[:, j]

        z_features = z_feature_builder(train_aug, prefix="ohat_")

        F, _, _ = make_model(train_aug, z_features)
        F.fit(train_aug[z_features], train_aug[z_target_cols].values.astype(np.float32))

        pred = F.predict(test_aug[z_features]).astype(np.float32)
        pred_all[te, :] = pred

        fold_metric = metrics(y_all[te], pred, prefix="")
        fold_metric.update({
            "fold": fold_idx,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
        })
        fold_rows.append(fold_metric)

    return pred_all, o_pred_all, pd.DataFrame(fold_rows), group_col

def add_bilinear_features(df, state_cols, op_cols_with_prefix):
    out = df.copy()
    new_cols = []
    for s in state_cols:
        for o in op_cols_with_prefix:
            name = f"bilin__{s}__x__{o}"
            out[name] = out[s].astype(float) * out[o].astype(float)
            new_cols.append(name)
    return out, new_cols

# -----------------------------
# LOAD
# -----------------------------

if not Path(INPUT_CSV).exists():
    raise FileNotFoundError(
        f"Cannot find {INPUT_CSV}. Put this script in the same folder as "
        f"phasemap6b1_correction_dataset.csv and run again."
    )

df = pd.read_csv(INPUT_CSV)
df = df.copy()

# Normalize column names lightly.
df.columns = [str(c).strip() for c in df.columns]

# -----------------------------
# COLUMN DISCOVERY
# -----------------------------

# Targets Z_{l+1}
z_target_candidates = [
    "R_l1",
    "boundary_dist_l1",
    "spread_l1",
    "rank_gap_l1",
]
z_target_cols = safe_cols(df, z_target_candidates)

# Some datasets may use alternative names.
if not z_target_cols:
    alt_targets = [
        "R_next", "B_next", "spread_next", "rank_gap_next",
        "R_l+1", "B_l+1",
    ]
    z_target_cols = safe_cols(df, alt_targets)

if len(z_target_cols) < 2:
    raise RuntimeError(
        f"Too few Z_next target columns found. Found: {z_target_cols}. "
        f"Available columns: {df.columns.tolist()}"
    )

# State Z_l features.
state_candidates = [
    "R_l",
    "boundary_dist_l",
    "spread_l",
    "rank_gap_l",
    "center_cos",
    "jaccard",
]
state_cols = safe_cols(df, state_candidates)

if len(state_cols) < 2:
    raise RuntimeError(
        f"Too few Z_l state columns found. Found: {state_cols}. "
        f"Available columns: {df.columns.tolist()}"
    )

# Context / phase / layer features available before transition.
context_candidates = [
    "layer",
    "condition",
    "phase",
    "phase_target",
    "condition_id",
    "epsilon",
    "heuristic_operator",
    "operator_cluster",
]
context_cols = safe_cols(df, context_candidates)

# History / momentum / previous dynamics candidates.
history_candidates = [
    "R_prev", "R_lm1", "R_delta_prev", "dR_prev", "dR_l",
    "boundary_dist_prev", "boundary_dist_lm1", "dB_prev", "dB_l",
    "spread_prev", "spread_lm1", "dspread_prev", "dspread_l",
    "rank_gap_prev", "rank_gap_lm1", "drank_gap_prev", "drank_gap_l",
]
history_cols = safe_cols(df, history_candidates)

# If explicit history is absent, create within-sample/layer lag features.
# This is still pre-transition if sample_id exists.
if not history_cols:
    sort_cols = []
    group_col_for_lag = None
    for gc in ["sample_id", "idx", "graph_id", "prompt_id", "case_id"]:
        if gc in df.columns:
            group_col_for_lag = gc
            break
    if "layer" in df.columns and group_col_for_lag is not None:
        df = df.sort_values([group_col_for_lag, "layer"]).reset_index(drop=True)
        for c in state_cols:
            prev = df.groupby(group_col_for_lag)[c].shift(1)
            dcol = f"lag_delta_{c}"
            pcol = f"lag_prev_{c}"
            df[pcol] = prev
            df[dcol] = df[c] - prev
            history_cols.extend([pcol, dcol])
    # if still empty, no history.

# Continuous operator coordinates.
op_cols = [c for c in df.columns if c.startswith("op_pc")]
op_cols = sorted(op_cols, key=lambda x: int(x.replace("op_pc", "")) if x.replace("op_pc", "").isdigit() else x)

if not op_cols:
    raise RuntimeError(
        "No op_pc* columns found. 7F requires continuous operator coordinates."
    )

# Clean rows for core columns.
needed = list(set(z_target_cols + state_cols + op_cols))
work = df.dropna(subset=[c for c in needed if c in df.columns]).reset_index(drop=True)

# Recompute after possible sorting/drop.
df = work

# -----------------------------
# FEATURE SETS
# -----------------------------

# G features: must be pre-transition only.
G1_state = state_cols
G2_state_history = state_cols + history_cols
G3_state_context = state_cols + context_cols
G4_state_history_context = state_cols + history_cols + context_cols

G_feature_sets = {
    "G1_state_only": G1_state,
    "G2_state_plus_history": G2_state_history,
    "G3_state_plus_context": G3_state_context,
    "G4_state_history_context": G4_state_history_context,
}

# Remove duplicate cols while preserving order.
for k, cols in list(G_feature_sets.items()):
    seen = set()
    dedup = []
    for c in cols:
        if c in df.columns and c not in seen:
            dedup.append(c)
            seen.add(c)
    G_feature_sets[k] = dedup

# F feature builders.
def build_F_state_only(frame, prefix=""):
    return state_cols

def build_F_state_plus_op(frame, prefix=""):
    ohat_cols = [f"{prefix}{c}" for c in op_cols]
    return state_cols + [c for c in ohat_cols if c in frame.columns]

def build_F_state_plus_op_bilinear(frame, prefix=""):
    ohat_cols = [f"{prefix}{c}" for c in op_cols if f"{prefix}{c}" in frame.columns]
    # Bilinear features are added outside where needed.
    return state_cols + ohat_cols

# -----------------------------
# BASELINES: STATE ONLY / TRUE O / TRUE O + BILINEAR
# -----------------------------

summary = {
    "input_csv": INPUT_CSV,
    "n_rows": int(len(df)),
    "z_target_cols": z_target_cols,
    "state_cols": state_cols,
    "history_cols": history_cols,
    "context_cols": context_cols,
    "op_cols": op_cols,
}

# F0: state only
pred_F0, info_F0 = cv_predict(df, state_cols, z_target_cols)
summary.update({f"F0_state_only_{k}": v for k, v in metrics(df[z_target_cols].values, pred_F0).items()})

# F_true: state + true O
true_feature_cols = state_cols + op_cols
pred_Ftrue, info_Ftrue = cv_predict(df, true_feature_cols, z_target_cols)
summary.update({f"Ftrue_state_Otrue_{k}": v for k, v in metrics(df[z_target_cols].values, pred_Ftrue).items()})

# F_true_bilinear
true_bilin_df, true_bilin_cols = add_bilinear_features(df, state_cols, op_cols)
true_bilin_features = state_cols + op_cols + true_bilin_cols
pred_Ftrue_bilin, info_Ftrue_bilin = cv_predict(true_bilin_df, true_bilin_features, z_target_cols)
summary.update({f"Ftrue_bilinear_{k}": v for k, v in metrics(true_bilin_df[z_target_cols].values, pred_Ftrue_bilin).items()})

# Save baseline predictions.
pred_rows = df[["layer"] + [c for c in ["sample_id", "idx", "graph_id", "condition", "phase", "heuristic_operator"] if c in df.columns]].copy() if "layer" in df.columns else pd.DataFrame(index=df.index)
for j, c in enumerate(z_target_cols):
    pred_rows[f"true_{c}"] = df[c].values
    pred_rows[f"pred_F0_{c}"] = pred_F0[:, j]
    pred_rows[f"pred_Ftrue_{c}"] = pred_Ftrue[:, j]
    pred_rows[f"pred_Ftrue_bilin_{c}"] = pred_Ftrue_bilin[:, j]

# -----------------------------
# STAGE G: O origin audit
# -----------------------------

g_summary_rows = []
oof_o_predictions = {}

for name, feats in G_feature_sets.items():
    if not feats:
        continue
    try:
        op_pred, info = cv_predict(df, feats, op_cols)
        oof_o_predictions[name] = op_pred

        row = {
            "model": name,
            "n_features": len(feats),
            "features": "|".join(feats),
            "group_col": info.get("group_col"),
        }
        row.update(metrics(df[op_cols].values.astype(np.float32), op_pred, prefix="Ohat_"))
        g_summary_rows.append(row)

        for j, c in enumerate(op_cols):
            pred_rows[f"pred_{name}_{c}"] = op_pred[:, j]

    except Exception as e:
        g_summary_rows.append({
            "model": name,
            "status": "error",
            "error": str(e),
            "n_features": len(feats),
            "features": "|".join(feats),
        })

g_summary = pd.DataFrame(g_summary_rows)

# Select best G by macro R2.
valid_g = g_summary[np.isfinite(g_summary.get("Ohat_r2_macro", np.nan))]
if len(valid_g) == 0:
    raise RuntimeError("No valid G model for O prediction.")
best_g_name = valid_g.sort_values("Ohat_r2_macro", ascending=False).iloc[0]["model"]
summary["best_G_model"] = str(best_g_name)
summary["best_G_Ohat_r2_macro"] = float(valid_g.sort_values("Ohat_r2_macro", ascending=False).iloc[0]["Ohat_r2_macro"])
summary["best_G_Ohat_corr_macro"] = float(valid_g.sort_values("Ohat_r2_macro", ascending=False).iloc[0]["Ohat_corr_macro"])

# -----------------------------
# TWO-STAGE F WITH O_HAT
# -----------------------------

two_stage_rows = []
two_stage_fold_tables = []

for g_name, g_feats in G_feature_sets.items():
    if g_name not in oof_o_predictions and g_feats:
        # It may still work in leakage-safe two-stage; try anyway.
        pass
    if not g_feats:
        continue

    # Stage F with predicted O, no bilinear.
    def z_builder_no_bilin(frame, prefix="ohat_"):
        return state_cols + [f"{prefix}{c}" for c in op_cols if f"{prefix}{c}" in frame.columns]

    try:
        pred_Z, pred_O, fold_df, group_col = fit_predict_infold_two_stage(
            df=df,
            o_feature_cols=g_feats,
            z_feature_builder=z_builder_no_bilin,
            op_cols=op_cols,
            z_target_cols=z_target_cols,
        )

        row = {
            "model": f"F_state_Ohat_from_{g_name}",
            "G_model": g_name,
            "bilinear": False,
            "group_col": group_col,
            "G_features": "|".join(g_feats),
        }
        row.update(metrics(df[z_target_cols].values.astype(np.float32), pred_Z, prefix="Znext_"))
        two_stage_rows.append(row)

        fold_df["model"] = row["model"]
        two_stage_fold_tables.append(fold_df)

        for j, c in enumerate(z_target_cols):
            pred_rows[f"pred_F_Ohat_{g_name}_{c}"] = pred_Z[:, j]

    except Exception as e:
        two_stage_rows.append({
            "model": f"F_state_Ohat_from_{g_name}",
            "G_model": g_name,
            "bilinear": False,
            "status": "error",
            "error": str(e),
            "G_features": "|".join(g_feats),
        })

    # Stage F with predicted O + bilinear.
    def z_builder_bilin(frame, prefix="ohat_"):
        ohat_cols = [f"{prefix}{c}" for c in op_cols if f"{prefix}{c}" in frame.columns]
        # Add bilinear columns in-place if absent.
        for s in state_cols:
            for o in ohat_cols:
                name = f"bilin__{s}__x__{o}"
                if name not in frame.columns:
                    frame[name] = frame[s].astype(float) * frame[o].astype(float)
        bilin_cols = [f"bilin__{s}__x__{o}" for s in state_cols for o in ohat_cols]
        return state_cols + ohat_cols + bilin_cols

    try:
        pred_Z, pred_O, fold_df, group_col = fit_predict_infold_two_stage(
            df=df,
            o_feature_cols=g_feats,
            z_feature_builder=z_builder_bilin,
            op_cols=op_cols,
            z_target_cols=z_target_cols,
        )

        row = {
            "model": f"F_state_Ohat_bilin_from_{g_name}",
            "G_model": g_name,
            "bilinear": True,
            "group_col": group_col,
            "G_features": "|".join(g_feats),
        }
        row.update(metrics(df[z_target_cols].values.astype(np.float32), pred_Z, prefix="Znext_"))
        two_stage_rows.append(row)

        fold_df["model"] = row["model"]
        two_stage_fold_tables.append(fold_df)

        for j, c in enumerate(z_target_cols):
            pred_rows[f"pred_F_Ohat_bilin_{g_name}_{c}"] = pred_Z[:, j]

    except Exception as e:
        two_stage_rows.append({
            "model": f"F_state_Ohat_bilin_from_{g_name}",
            "G_model": g_name,
            "bilinear": True,
            "status": "error",
            "error": str(e),
            "G_features": "|".join(g_feats),
        })

two_stage_summary = pd.DataFrame(two_stage_rows)
fold_summary = pd.concat(two_stage_fold_tables, ignore_index=True) if two_stage_fold_tables else pd.DataFrame()

# -----------------------------
# READOUTS
# -----------------------------

valid_two = two_stage_summary[np.isfinite(two_stage_summary.get("Znext_r2_macro", np.nan))].copy()
best_two = valid_two.sort_values("Znext_r2_macro", ascending=False).iloc[0] if len(valid_two) else None

F0_r2 = summary.get("F0_state_only_r2_macro")
Ftrue_r2 = summary.get("Ftrue_state_Otrue_r2_macro")
Ftrue_bilin_r2 = summary.get("Ftrue_bilinear_r2_macro")
best_Ohat_r2 = float(best_two["Znext_r2_macro"]) if best_two is not None else np.nan

summary["best_two_stage_model"] = str(best_two["model"]) if best_two is not None else None
summary["best_two_stage_Znext_r2_macro"] = best_Ohat_r2
summary["delta_trueO_minus_state"] = float(Ftrue_r2 - F0_r2)
summary["delta_trueO_bilin_minus_trueO"] = float(Ftrue_bilin_r2 - Ftrue_r2)
summary["delta_Ohat_minus_state"] = float(best_Ohat_r2 - F0_r2) if np.isfinite(best_Ohat_r2) else np.nan
summary["Ohat_retention_vs_trueO"] = float((best_Ohat_r2 - F0_r2) / (Ftrue_r2 - F0_r2)) if (Ftrue_r2 - F0_r2) != 0 and np.isfinite(best_Ohat_r2) else np.nan

# Verdict
ret = summary["Ohat_retention_vs_trueO"]
delta = summary["delta_Ohat_minus_state"]

if np.isfinite(ret) and ret >= 0.50 and delta > 0.05:
    verdict = "PASS-Strong"
elif np.isfinite(ret) and ret >= 0.25 and delta > 0.02:
    verdict = "PASS-Lite"
else:
    verdict = "INCONCLUSIVE-or-FAIL"

summary["verdict"] = verdict
summary["verdict_rule"] = (
    "PASS-Strong if predicted O_hat keeps >=50% of true-O gain and improves state-only by >0.05; "
    "PASS-Lite if >=25% and >0.02."
)

# -----------------------------
# SAVE
# -----------------------------

g_summary.to_csv(OUTPUT_DIR / "phasemap7f_operator_origin_G_summary.csv", index=False)
two_stage_summary.to_csv(OUTPUT_DIR / "phasemap7f_two_stage_causality_summary.csv", index=False)
fold_summary.to_csv(OUTPUT_DIR / "phasemap7f_fold_summary.csv", index=False)
pred_rows.to_csv(OUTPUT_DIR / "phasemap7f_predictions.csv", index=False)

with open(OUTPUT_DIR / "phasemap7f_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n================ PhaseMap-7F Summary ================")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\nG summary:")
print(g_summary.sort_values("Ohat_r2_macro", ascending=False).head(10).to_string(index=False))
print("\nTwo-stage summary:")
print(two_stage_summary.sort_values("Znext_r2_macro", ascending=False).head(10).to_string(index=False))
print("\nOutputs saved to:", OUTPUT_DIR.resolve())
