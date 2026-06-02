# ============================================================
# Constraint-Audit-3B.1
# Delta-R Minimal Invariant Test
#
# Input:
#   ./constraint_audit3b_outputs/constraint_audit3b_feature_table.csv
#
# Goal:
#   3B showed that delta_C_only improves LOFO mechanism generalization,
#   and invariant ranking is dominated by DELTA_R trajectory features.
#
#   3B.1 asks:
#       Is Delta-R trajectory alone sufficient as a minimal
#       structural invariant candidate?
#
# Main feature sets:
#   1. baseline_RHpC_20_25
#   2. absolute_R_raw_20_25
#   3. absolute_R_shape_20_25
#   4. delta_R_raw_20_25
#   5. delta_R_shape_20_25
#   6. delta_R_minimal_top
#   7. delta_R_full_20_25
#   8. delta_R_plus_abs_R
#   9. delta_R_plus_delta_HpC
#
# Evaluations:
#   - GroupKFold binary Gen_E
#   - GroupKFold submechanism / risk_regime
#   - LOFO binary Gen_E
#   - LOFO submechanism / risk_regime
#   - invariant F-score ranking
#   - full-data coefficient inspection
#
# Outputs:
#   ./constraint_audit3b1_outputs/
# ============================================================

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

RANDOM_SEED = 42

INPUT_FILE = Path("./constraint_audit3b_outputs/constraint_audit3b_feature_table.csv")

SAVE_DIR = Path("./constraint_audit3b1_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

N_GROUP_SPLITS = 5

TRACK_LAYERS = [20, 21, 22, 23, 24, 25]

# Top candidates from 3B invariant ranking. If some do not exist,
# the script will silently skip them.
DELTA_R_TOP_CANDIDATES = [
    "DELTA_R_slope",
    "DELTA_R_L24",
    "DELTA_R_min",
    "DELTA_R_range",
    "DELTA_R_L25",
    "DELTA_R_last",
    "DELTA_R_traj_l2",
    "DELTA_R_area",
    "DELTA_R_mean",
    "DELTA_R_L23",
    "DELTA_R_L22",
    "DELTA_R_shape_cos_to_clean",
    "DELTA_R_traj_cos_to_clean",
]

# ============================================================
# LABEL MAPS
# ============================================================

FAMILY_ORDER = [
    "stable_clean_basic",
    "stable_paraphrase",
    "stable_redundant",
    "stable_irrelevant",
    "stable_weak_note",

    "competition_ambiguous",
    "competition_branch",
    "competition_direct",
    "competition_source_claim",
    "competition_equal_evidence",

    "closure_negation",
    "closure_update",
    "closure_override",
    "closure_temporal",
    "closure_authority",
    "closure_exception",
]

COARSE_MECHANISM_MAP = {
    "stable_clean_basic": "stable_or_preserved",
    "stable_paraphrase": "stable_or_preserved",
    "stable_redundant": "stable_or_preserved",
    "stable_irrelevant": "stable_or_preserved",
    "stable_weak_note": "stable_or_preserved",

    "competition_ambiguous": "competition_conflict",
    "competition_branch": "competition_conflict",
    "competition_direct": "competition_conflict",
    "competition_source_claim": "competition_conflict",
    "competition_equal_evidence": "competition_conflict",

    "closure_negation": "closure_rewrite",
    "closure_update": "closure_rewrite",
    "closure_override": "closure_rewrite",
    "closure_temporal": "closure_rewrite",
    "closure_authority": "closure_rewrite",
    "closure_exception": "closure_rewrite",
}

SUBMECHANISM_MAP = {
    "stable_clean_basic": "stable_core",
    "stable_redundant": "stable_core",
    "stable_weak_note": "stable_core",
    "stable_paraphrase": "stable_surface_shift",
    "stable_irrelevant": "stable_surface_shift",

    "competition_branch": "competition_low_commit",
    "competition_direct": "competition_low_commit",
    "competition_ambiguous": "competition_ambiguous",
    "competition_source_claim": "competition_high_commit",
    "competition_equal_evidence": "competition_high_commit",

    "closure_negation": "closure_canonical_rewrite",
    "closure_update": "closure_canonical_rewrite",
    "closure_temporal": "closure_canonical_rewrite",
    "closure_authority": "closure_canonical_rewrite",
    "closure_override": "closure_rule_override",
    "closure_exception": "closure_rule_override",
}

RISK_REGIME_MAP = {
    "stable_clean_basic": "clean_basin",
    "stable_redundant": "clean_basin",
    "stable_weak_note": "clean_basin",

    "stable_paraphrase": "low_risk_shift",
    "stable_irrelevant": "low_risk_shift",

    "competition_branch": "competition_low_risk",
    "competition_direct": "competition_low_risk",

    "competition_ambiguous": "competition_mid_risk",

    "competition_source_claim": "competition_high_risk",
    "competition_equal_evidence": "competition_high_risk",

    "closure_negation": "closure_collapse",
    "closure_update": "closure_collapse",
    "closure_override": "closure_collapse",
    "closure_temporal": "closure_collapse",
    "closure_authority": "closure_collapse",
    "closure_exception": "closure_collapse",
}

SUBMECHANISM_CLASSES = [
    "stable_core",
    "stable_surface_shift",
    "competition_low_commit",
    "competition_ambiguous",
    "competition_high_commit",
    "closure_canonical_rewrite",
    "closure_rule_override",
]

RISK_REGIME_CLASSES = [
    "clean_basin",
    "low_risk_shift",
    "competition_low_risk",
    "competition_mid_risk",
    "competition_high_risk",
    "closure_collapse",
]

MECHANISM_CLASSES = [
    "stable_or_preserved",
    "competition_conflict",
    "closure_rewrite",
]

# ============================================================
# HELPERS
# ============================================================

def parse_bool_series(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def sanitize_prob(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(p, 1e-6, 1.0 - 1e-6)


def safe_auc(y, p):
    y = np.asarray(y).astype(int)
    p = sanitize_prob(p)

    if len(np.unique(y)) < 2:
        return np.nan

    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return np.nan


def safe_log_loss(y, p):
    y = np.asarray(y).astype(int)
    p = sanitize_prob(p)

    try:
        return float(log_loss(y, p, labels=[0, 1]))
    except Exception:
        return np.nan


def calc_binary_metrics(y, p, threshold=0.5):
    y = np.asarray(y).astype(int)
    p = sanitize_prob(p)
    pred = (p >= threshold).astype(int)

    out = {
        "auc": safe_auc(y, p),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "nll": safe_log_loss(y, p),
        "pred_pos_rate": float(pred.mean()),
        "mean_prob": float(p.mean()),
        "threshold": float(threshold),
    }

    try:
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        out["tn"] = int(tn)
        out["fp"] = int(fp)
        out["fn"] = int(fn)
        out["tp"] = int(tp)
    except Exception:
        out["tn"] = out["fp"] = out["fn"] = out["tp"] = 0

    return out


def clean_cols(df, cols):
    seen = set()
    out = []

    for c in cols:
        if c in seen:
            continue
        seen.add(c)

        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)

    return out


def add_shape_features(df, prefix, layers):
    """
    Adds shape features for columns like:
        R_L20...
        DELTA_R_L20...
    prefix examples:
        "R"
        "DELTA_R"
        "DELTA_H"
        "DELTA_pC"
    """
    out = df.copy()
    cols = [f"{prefix}_L{l}" for l in layers if f"{prefix}_L{l}" in out.columns]

    if len(cols) == 0:
        return out

    X = out[cols].values.astype(np.float32)

    out[f"{prefix}_mean_20_25"] = X.mean(axis=1)
    out[f"{prefix}_min_20_25"] = X.min(axis=1)
    out[f"{prefix}_max_20_25"] = X.max(axis=1)
    out[f"{prefix}_last_20_25"] = X[:, -1]
    out[f"{prefix}_first_20_25"] = X[:, 0]
    out[f"{prefix}_slope_20_25"] = X[:, -1] - X[:, 0]
    out[f"{prefix}_drop_20_25"] = X[:, 0] - X[:, -1]
    out[f"{prefix}_range_20_25"] = X.max(axis=1) - X.min(axis=1)
    out[f"{prefix}_area_20_25"] = X.sum(axis=1)

    out[f"{prefix}_abs_mean_20_25"] = np.abs(X).mean(axis=1)
    out[f"{prefix}_abs_min_20_25"] = np.abs(X).min(axis=1)
    out[f"{prefix}_abs_max_20_25"] = np.abs(X).max(axis=1)

    out[f"{prefix}_num_negative_20_25"] = (X < 0).sum(axis=1).astype(np.float32)
    out[f"{prefix}_any_negative_20_25"] = (X < 0).any(axis=1).astype(np.float32)

    first_neg = []
    for row in X:
        neg = np.where(row < 0)[0]
        if len(neg) == 0:
            first_neg.append(float(len(layers)))
        else:
            first_neg.append(float(neg[0]))
    out[f"{prefix}_first_negative_offset_20_25"] = np.array(first_neg, dtype=np.float32)

    # Curvature-like finite difference features.
    if X.shape[1] >= 3:
        d1 = np.diff(X, axis=1)
        d2 = np.diff(d1, axis=1)

        out[f"{prefix}_d1_mean_20_25"] = d1.mean(axis=1)
        out[f"{prefix}_d1_min_20_25"] = d1.min(axis=1)
        out[f"{prefix}_d1_max_20_25"] = d1.max(axis=1)
        out[f"{prefix}_d1_range_20_25"] = d1.max(axis=1) - d1.min(axis=1)

        out[f"{prefix}_d2_mean_20_25"] = d2.mean(axis=1)
        out[f"{prefix}_d2_min_20_25"] = d2.min(axis=1)
        out[f"{prefix}_d2_max_20_25"] = d2.max(axis=1)
        out[f"{prefix}_d2_abs_mean_20_25"] = np.abs(d2).mean(axis=1)

    return out


# ============================================================
# MODELS
# ============================================================

class ConstantBinaryModel:
    def __init__(self, p):
        self.p = float(np.clip(p, 1e-6, 1.0 - 1e-6))

    def predict_proba(self, X):
        n = len(X)
        p1 = np.full(n, self.p, dtype=np.float64)
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)


class ConstantMulticlassModel:
    def __init__(self, classes):
        self.classes_ = np.asarray(classes)

    def predict(self, X):
        return np.full(len(X), self.classes_[0], dtype=object)

    def predict_proba(self, X):
        out = np.zeros((len(X), len(self.classes_)), dtype=np.float64)
        out[:, 0] = 1.0
        return out


def make_binary_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            max_iter=5000,
        )),
    ])


def make_multiclass_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            max_iter=8000,
        )),
    ])


def fit_binary_or_constant(X, y):
    y = np.asarray(y).astype(int)

    if len(np.unique(y)) < 2:
        return ConstantBinaryModel(float(y.mean()))

    model = make_binary_model()
    model.fit(X, y)
    return model


def fit_multiclass_or_constant(X, y):
    y = np.asarray(y)
    vals = np.unique(y)

    if len(vals) < 2:
        return ConstantMulticlassModel(vals)

    model = make_multiclass_model()
    model.fit(X, y)
    return model


def aligned_proba(model, X, classes):
    p_raw = model.predict_proba(X)

    if hasattr(model, "classes_"):
        model_classes = list(model.classes_)
    else:
        model_classes = list(model.named_steps["clf"].classes_)

    out = np.zeros((len(X), len(classes)), dtype=np.float64)

    for j, cls in enumerate(model_classes):
        if cls in classes:
            out[:, classes.index(cls)] = p_raw[:, j]

    row_sum = out.sum(axis=1, keepdims=True)
    bad = row_sum[:, 0] <= 1e-12

    if bad.any():
        out[bad, :] = 1.0 / len(classes)
        row_sum = out.sum(axis=1, keepdims=True)

    return out / (row_sum + 1e-12)


# ============================================================
# LOAD DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find input file: {INPUT_FILE}\n"
        f"Run Constraint-Audit-3B first."
    )

df = pd.read_csv(INPUT_FILE)

for col in ["is_clean", "is_conflict", "is_other"]:
    if col in df.columns:
        df[col] = parse_bool_series(df[col])

if "y_conflict" not in df.columns:
    if "is_conflict" not in df.columns:
        raise RuntimeError("Need y_conflict or is_conflict column.")
    df["y_conflict"] = df["is_conflict"].astype(int)

df["y_conflict"] = df["y_conflict"].astype(int)

if "is_clean" in df.columns and "is_conflict" in df.columns:
    df = df[(df["is_clean"] == True) | (df["is_conflict"] == True)].copy()

if "family" not in df.columns:
    raise RuntimeError("Input table must contain family column.")

if "idx" not in df.columns:
    raise RuntimeError("Input table must contain idx column for GroupKFold.")

df["mechanism"] = df["family"].map(COARSE_MECHANISM_MAP)
df["submechanism"] = df["family"].map(SUBMECHANISM_MAP)
df["risk_regime"] = df["family"].map(RISK_REGIME_MAP)

if df["submechanism"].isna().any():
    bad = df[df["submechanism"].isna()]["family"].unique().tolist()
    raise RuntimeError(f"Unknown family values: {bad}")

df = df.reset_index(drop=True)

# Add robust shape features, even if 3B already has some.
for prefix in ["R", "H", "pC", "DELTA_R", "DELTA_H", "DELTA_pC"]:
    df = add_shape_features(df, prefix, TRACK_LAYERS)

print("\n============================================================")
print("Constraint-Audit-3B.1 Input Summary")
print("============================================================\n")

print("Input:", INPUT_FILE)
print("Rows:", len(df))
print("Families:", df["family"].nunique())
print("Positive rate:", df["y_conflict"].mean())

print("\nFamily summary:")
print(
    f"{'family':<32}"
    f"{'mechanism':<24}"
    f"{'submechanism':<28}"
    f"{'risk_regime':<24}"
    f"{'N':<6}"
    f"{'GenE':<10}"
)

for fam in FAMILY_ORDER:
    sub = df[df["family"] == fam]
    if len(sub) == 0:
        continue

    print(
        f"{fam:<32}"
        f"{COARSE_MECHANISM_MAP[fam]:<24}"
        f"{SUBMECHANISM_MAP[fam]:<28}"
        f"{RISK_REGIME_MAP[fam]:<24}"
        f"{len(sub):<6}"
        f"{sub['y_conflict'].mean():<10.4f}"
    )

# ============================================================
# FEATURE SETS
# ============================================================

# Raw baseline R/H/pC.
baseline_raw_cols = []
for l in TRACK_LAYERS:
    for name in ["R", "H", "pC"]:
        c = f"{name}_L{l}"
        if c in df.columns:
            baseline_raw_cols.append(c)

# Include existing baseline segment features if present.
baseline_segment_cols = [
    c for c in df.columns
    if (
        c.startswith("ENTRY20_22")
        or c.startswith("BASIN23_25")
        or c.startswith("ALL20_25")
        or c.startswith("ENTRY20_22_BASE")
        or c.startswith("BASIN23_25_BASE")
        or c.startswith("ALL20_25_BASE")
    )
    and pd.api.types.is_numeric_dtype(df[c])
]

baseline_cols = clean_cols(df, baseline_raw_cols + baseline_segment_cols)

absolute_R_raw_cols = clean_cols(df, [f"R_L{l}" for l in TRACK_LAYERS])

absolute_R_shape_cols = clean_cols(df, [
    c for c in df.columns
    if c.startswith("R_") and c.endswith("_20_25")
])

absolute_R_full_cols = clean_cols(df, absolute_R_raw_cols + absolute_R_shape_cols)

delta_R_raw_cols = clean_cols(df, [f"DELTA_R_L{l}" for l in TRACK_LAYERS])

delta_R_shape_cols = clean_cols(df, [
    c for c in df.columns
    if c.startswith("DELTA_R_")
    and (
        c.endswith("_20_25")
        or c in [
            "DELTA_R_mean",
            "DELTA_R_min",
            "DELTA_R_max",
            "DELTA_R_last",
            "DELTA_R_slope",
            "DELTA_R_range",
            "DELTA_R_area",
            "DELTA_R_traj_l2",
            "DELTA_R_traj_cos_to_clean",
            "DELTA_R_shape_cos_to_clean",
        ]
    )
])

delta_R_top_cols = clean_cols(df, DELTA_R_TOP_CANDIDATES)

delta_R_full_cols = clean_cols(df, delta_R_raw_cols + delta_R_shape_cols + delta_R_top_cols)

delta_HpC_cols = clean_cols(df, [
    c for c in df.columns
    if (
        c.startswith("DELTA_H_")
        or c.startswith("DELTA_pC_")
    )
    and (
        any(f"_L{l}" in c for l in TRACK_LAYERS)
        or c.endswith("_20_25")
        or c in [
            "DELTA_H_mean",
            "DELTA_H_min",
            "DELTA_H_max",
            "DELTA_H_last",
            "DELTA_H_slope",
            "DELTA_H_range",
            "DELTA_H_area",
            "DELTA_pC_mean",
            "DELTA_pC_min",
            "DELTA_pC_max",
            "DELTA_pC_last",
            "DELTA_pC_slope",
            "DELTA_pC_range",
            "DELTA_pC_area",
        ]
    )
])

feature_sets = {
    "baseline_RHpC_20_25": baseline_cols,

    "absolute_R_raw_20_25": absolute_R_raw_cols,
    "absolute_R_shape_20_25": absolute_R_shape_cols,
    "absolute_R_full_20_25": absolute_R_full_cols,

    "delta_R_raw_20_25": delta_R_raw_cols,
    "delta_R_shape_20_25": delta_R_shape_cols,
    "delta_R_minimal_top": delta_R_top_cols,
    "delta_R_full_20_25": delta_R_full_cols,

    "delta_R_plus_abs_R": clean_cols(df, delta_R_full_cols + absolute_R_full_cols),
    "delta_R_plus_delta_HpC": clean_cols(df, delta_R_full_cols + delta_HpC_cols),
}

# Drop empty sets.
feature_sets = {k: v for k, v in feature_sets.items() if len(v) > 0}

# Sanitize numeric values.
all_used_cols = sorted(set(sum(feature_sets.values(), [])))

for c in all_used_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df[all_used_cols] = (
    df[all_used_cols]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)

print("\nFeature sets:")
for name, cols in feature_sets.items():
    print(f"  {name:<32} n={len(cols)}")

# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def eval_binary_groupcv(df_in, cols, feature_set):
    X = df_in[cols].values.astype(np.float32)
    y = df_in["y_conflict"].values.astype(int)
    groups = df_in["idx"].values

    n_splits = min(N_GROUP_SPLITS, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    p = np.zeros(len(df_in), dtype=np.float64)

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        model = fit_binary_or_constant(X[train_idx], y[train_idx])
        p[test_idx] = sanitize_prob(model.predict_proba(X[test_idx])[:, 1])

    row = {
        "feature_set": feature_set,
        "eval": "GroupKFold_binary",
        "n_features": len(cols),
    }
    row.update(calc_binary_metrics(y, p, threshold=0.5))
    return row


def eval_multiclass_groupcv(df_in, cols, target_col, classes, feature_set):
    X = df_in[cols].values.astype(np.float32)
    y = df_in[target_col].values
    groups = df_in["idx"].values

    n_splits = min(N_GROUP_SPLITS, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    pred = np.empty(len(df_in), dtype=object)

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        model = fit_multiclass_or_constant(X[train_idx], y[train_idx])
        pred[test_idx] = model.predict(X[test_idx])

    return {
        "feature_set": feature_set,
        "target": target_col,
        "eval": "GroupKFold_multiclass",
        "n_features": len(cols),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def eval_binary_lofo(df_in, cols, feature_set):
    X = df_in[cols].values.astype(np.float32)
    y = df_in["y_conflict"].values.astype(int)
    families = df_in["family"].values

    logo = LeaveOneGroupOut()

    rows = []
    pred_rows = []

    for train_idx, test_idx in logo.split(X, y, groups=families):
        heldout = families[test_idx][0]

        model = fit_binary_or_constant(X[train_idx], y[train_idx])
        p_test = sanitize_prob(model.predict_proba(X[test_idx])[:, 1])

        metrics = calc_binary_metrics(y[test_idx], p_test, threshold=0.5)

        row = {
            "feature_set": feature_set,
            "heldout_family": heldout,
            "heldout_mechanism": COARSE_MECHANISM_MAP[heldout],
            "heldout_submechanism": SUBMECHANISM_MAP[heldout],
            "heldout_risk_regime": RISK_REGIME_MAP[heldout],
            "positive_rate": float(y[test_idx].mean()),
            "n_test": int(len(test_idx)),
        }
        row.update(metrics)
        rows.append(row)

        temp = df_in.iloc[test_idx][[
            "idx",
            "family",
            "mechanism",
            "submechanism",
            "risk_regime",
            "y_conflict",
        ]].copy()
        temp["feature_set"] = feature_set
        temp["p_conflict"] = p_test
        pred_rows.append(temp)

    return pd.DataFrame(rows), pd.concat(pred_rows, axis=0, ignore_index=True)


def eval_multiclass_lofo(df_in, cols, target_col, classes, feature_set):
    X = df_in[cols].values.astype(np.float32)
    y = df_in[target_col].values
    families = df_in["family"].values

    logo = LeaveOneGroupOut()

    rows = []
    pred_rows = []

    for train_idx, test_idx in logo.split(X, y, groups=families):
        heldout = families[test_idx][0]
        true_label = y[test_idx][0]
        train_has_true_label = true_label in set(y[train_idx])

        model = fit_multiclass_or_constant(X[train_idx], y[train_idx])
        p = aligned_proba(model, X[test_idx], classes)
        pred = np.array(classes)[np.argmax(p, axis=1)]

        row = {
            "feature_set": feature_set,
            "target": target_col,
            "heldout_family": heldout,
            "true_label": true_label,
            "train_has_true_label": bool(train_has_true_label),
            "accuracy": float(accuracy_score(y[test_idx], pred)),
            "macro_f1": float(f1_score(y[test_idx], pred, average="macro", zero_division=0)),
            "n_test": int(len(test_idx)),
        }

        for j, cls in enumerate(classes):
            row[f"mean_p_{cls}"] = float(p[:, j].mean())

        rows.append(row)

        temp = df_in.iloc[test_idx][[
            "idx",
            "family",
            "mechanism",
            "submechanism",
            "risk_regime",
            "y_conflict",
        ]].copy()

        temp["feature_set"] = feature_set
        temp["target"] = target_col
        temp["pred"] = pred

        for j, cls in enumerate(classes):
            temp[f"p_{cls}"] = p[:, j]

        pred_rows.append(temp)

    return pd.DataFrame(rows), pd.concat(pred_rows, axis=0, ignore_index=True)


# ============================================================
# INVARIANT RANKING
# ============================================================

def invariant_f_score(df_in, feature_cols, label_col):
    rows = []

    y = df_in[label_col].values
    labels = sorted(pd.unique(y).tolist())

    for feat in feature_cols:
        x = df_in[feat].values.astype(np.float64)
        valid = np.isfinite(x)

        if valid.sum() == 0:
            continue

        x = x[valid]
        y_valid = y[valid]

        overall = np.nanmean(x)

        between = 0.0
        within = 0.0
        n_total = 0

        for lab in labels:
            vals = x[y_valid == lab]
            vals = vals[np.isfinite(vals)]

            if len(vals) == 0:
                continue

            n = len(vals)
            mu = vals.mean()

            between += n * (mu - overall) ** 2
            within += ((vals - mu) ** 2).sum()
            n_total += n

        k = len(labels)

        if k <= 1 or n_total <= k:
            continue

        between_norm = between / max(1, k - 1)
        within_norm = within / max(1, n_total - k)

        f = between_norm / (within_norm + 1e-8)

        rows.append({
            "label_col": label_col,
            "feature": feat,
            "f_score": float(f),
            "between_var": float(between_norm),
            "within_var": float(within_norm),
        })

    return pd.DataFrame(rows).sort_values(by="f_score", ascending=False)


def fit_full_binary_coefficients(df_in, cols, feature_set):
    X = df_in[cols].values.astype(np.float32)
    y = df_in["y_conflict"].values.astype(int)

    model = fit_binary_or_constant(X, y)

    if not isinstance(model, Pipeline):
        return pd.DataFrame()

    coef = model.named_steps["clf"].coef_[0]

    rows = []
    for feat, val in sorted(zip(cols, coef), key=lambda x: abs(x[1]), reverse=True):
        rows.append({
            "feature_set": feature_set,
            "feature": feat,
            "coef": float(val),
            "abs_coef": float(abs(val)),
        })

    return pd.DataFrame(rows)


def fit_full_multiclass_coefficients(df_in, cols, target_col, feature_set):
    X = df_in[cols].values.astype(np.float32)
    y = df_in[target_col].values

    model = fit_multiclass_or_constant(X, y)

    if not isinstance(model, Pipeline):
        return pd.DataFrame()

    clf = model.named_steps["clf"]
    classes = list(clf.classes_)
    coef = clf.coef_

    rows = []
    for i, cls in enumerate(classes):
        for feat, val in sorted(zip(cols, coef[i]), key=lambda x: abs(x[1]), reverse=True):
            rows.append({
                "feature_set": feature_set,
                "target": target_col,
                "class": cls,
                "feature": feat,
                "coef": float(val),
                "abs_coef": float(abs(val)),
            })

    return pd.DataFrame(rows)


# ============================================================
# RUN EVALUATIONS
# ============================================================

binary_group_rows = []
multi_group_rows = []

binary_lofo_all = []
binary_lofo_pred_all = []

multi_lofo_all = []
multi_lofo_pred_all = []

coef_binary_all = []
coef_multi_all = []

for fs_name, cols in feature_sets.items():
    print(f"\nEvaluating feature set: {fs_name} | n_features={len(cols)}")

    binary_group_rows.append(
        eval_binary_groupcv(df, cols, fs_name)
    )

    for target_col, classes in [
        ("mechanism", MECHANISM_CLASSES),
        ("submechanism", SUBMECHANISM_CLASSES),
        ("risk_regime", RISK_REGIME_CLASSES),
    ]:
        multi_group_rows.append(
            eval_multiclass_groupcv(
                df,
                cols,
                target_col=target_col,
                classes=classes,
                feature_set=fs_name,
            )
        )

    blofo, blofo_pred = eval_binary_lofo(df, cols, fs_name)
    binary_lofo_all.append(blofo)
    binary_lofo_pred_all.append(blofo_pred)

    for target_col, classes in [
        ("mechanism", MECHANISM_CLASSES),
        ("submechanism", SUBMECHANISM_CLASSES),
        ("risk_regime", RISK_REGIME_CLASSES),
    ]:
        mlofo, mlofo_pred = eval_multiclass_lofo(
            df,
            cols,
            target_col=target_col,
            classes=classes,
            feature_set=fs_name,
        )
        multi_lofo_all.append(mlofo)
        multi_lofo_pred_all.append(mlofo_pred)

    # Coefficients are most useful for minimal Delta-R sets, but compute all.
    coef_binary_all.append(
        fit_full_binary_coefficients(df, cols, fs_name)
    )

    coef_multi_all.append(
        fit_full_multiclass_coefficients(
            df,
            cols,
            target_col="submechanism",
            feature_set=fs_name,
        )
    )

binary_group_df = pd.DataFrame(binary_group_rows)
multi_group_df = pd.DataFrame(multi_group_rows)

binary_lofo_df = pd.concat(binary_lofo_all, axis=0, ignore_index=True)
binary_lofo_pred_df = pd.concat(binary_lofo_pred_all, axis=0, ignore_index=True)

multi_lofo_df = pd.concat(multi_lofo_all, axis=0, ignore_index=True)
multi_lofo_pred_df = pd.concat(multi_lofo_pred_all, axis=0, ignore_index=True)

coef_binary_df = pd.concat(coef_binary_all, axis=0, ignore_index=True)
coef_multi_df = pd.concat(coef_multi_all, axis=0, ignore_index=True)

# ============================================================
# SUMMARIES
# ============================================================

binary_lofo_summary_rows = []

for fs, sub in binary_lofo_df.groupby("feature_set"):
    binary_lofo_summary_rows.append({
        "feature_set": fs,
        "mean_auc": float(np.nanmean(sub["auc"])),
        "mean_accuracy": float(np.nanmean(sub["accuracy"])),
        "mean_f1": float(np.nanmean(sub["f1"])),
        "mean_brier": float(np.nanmean(sub["brier"])),
        "mean_pred_pos_rate": float(np.nanmean(sub["pred_pos_rate"])),
    })

binary_lofo_summary = pd.DataFrame(binary_lofo_summary_rows)

multi_lofo_summary_rows = []

for keys, sub in multi_lofo_df.groupby(["feature_set", "target"]):
    fs, target = keys
    multi_lofo_summary_rows.append({
        "feature_set": fs,
        "target": target,
        "mean_accuracy": float(np.nanmean(sub["accuracy"])),
        "mean_macro_f1": float(np.nanmean(sub["macro_f1"])),
    })

multi_lofo_summary = pd.DataFrame(multi_lofo_summary_rows)

# Top class summary for LOFO multiclass.
top_class_rows = []

for _, r in multi_lofo_df.iterrows():
    prob_cols = [c for c in multi_lofo_df.columns if c.startswith("mean_p_")]

    vals = {}
    for c in prob_cols:
        val = r.get(c, np.nan)
        if pd.notna(val):
            vals[c.replace("mean_p_", "")] = val

    if len(vals) == 0:
        continue

    top_cls = max(vals, key=vals.get)
    top_p = vals[top_cls]

    top_class_rows.append({
        "feature_set": r["feature_set"],
        "target": r["target"],
        "heldout_family": r["heldout_family"],
        "true_label": r["true_label"],
        "train_has_true_label": bool(r["train_has_true_label"]),
        "top_pred_class": top_cls,
        "top_pred_prob": float(top_p),
        "accuracy": float(r["accuracy"]),
        "macro_f1": float(r["macro_f1"]),
    })

top_class_df = pd.DataFrame(top_class_rows)

# Invariant ranking only over Delta-R candidate cols.
delta_r_candidate_cols = clean_cols(df, delta_R_full_cols)

inv_mech = invariant_f_score(df, delta_r_candidate_cols, "mechanism")
inv_sub = invariant_f_score(df, delta_r_candidate_cols, "submechanism")
inv_risk = invariant_f_score(df, delta_r_candidate_cols, "risk_regime")

invariant_ranking = pd.concat([inv_mech, inv_sub, inv_risk], axis=0, ignore_index=True)

# Family means for minimal Delta-R features.
family_delta_rows = []

family_mean_cols = clean_cols(df, delta_R_top_cols + delta_R_raw_cols)

for fam in FAMILY_ORDER:
    sub = df[df["family"] == fam]
    if len(sub) == 0:
        continue

    row = {
        "family": fam,
        "mechanism": COARSE_MECHANISM_MAP[fam],
        "submechanism": SUBMECHANISM_MAP[fam],
        "risk_regime": RISK_REGIME_MAP[fam],
        "n": int(len(sub)),
        "genE_rate": float(sub["y_conflict"].mean()),
    }

    for c in family_mean_cols:
        row[f"mean_{c}"] = float(sub[c].mean())
        row[f"std_{c}"] = float(sub[c].std())

    family_delta_rows.append(row)

family_delta_summary = pd.DataFrame(family_delta_rows)

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n\n============================================================")
print("3B.1 GroupKFold Binary")
print("============================================================\n")

print(
    binary_group_df[[
        "feature_set",
        "n_features",
        "auc",
        "accuracy",
        "f1",
        "brier",
        "pred_pos_rate",
    ]]
    .sort_values(by="brier")
    .to_string(index=False)
)

print("\n\n============================================================")
print("3B.1 GroupKFold Multiclass")
print("============================================================\n")

print(
    multi_group_df
    .sort_values(by=["target", "macro_f1"], ascending=[True, False])
    .to_string(index=False)
)

print("\n\n============================================================")
print("3B.1 LOFO Binary Summary")
print("============================================================\n")

print(
    binary_lofo_summary
    .sort_values(by="mean_brier")
    .to_string(index=False)
)

print("\n\n============================================================")
print("3B.1 LOFO Multiclass Summary")
print("============================================================\n")

print(
    multi_lofo_summary
    .sort_values(by=["target", "mean_macro_f1"], ascending=[True, False])
    .to_string(index=False)
)

print("\n\n============================================================")
print("3B.1 Top Delta-R Invariant Candidates: submechanism")
print("============================================================\n")

print(inv_sub.head(30).to_string(index=False))

print("\n\n============================================================")
print("3B.1 Top Delta-R Invariant Candidates: risk_regime")
print("============================================================\n")

print(inv_risk.head(30).to_string(index=False))

print("\n\n============================================================")
print("3B.1 Binary Coefficients: delta_R_minimal_top")
print("============================================================\n")

print(
    coef_binary_df[coef_binary_df["feature_set"] == "delta_R_minimal_top"]
    .head(30)
    .to_string(index=False)
)

print("\n\n============================================================")
print("3B.1 Key LOFO Top Predictions")
print("============================================================\n")

interesting_families = [
    "stable_paraphrase",
    "stable_irrelevant",
    "competition_branch",
    "competition_direct",
    "competition_source_claim",
    "competition_equal_evidence",
    "closure_override",
    "closure_exception",
]

view = top_class_df[
    top_class_df["heldout_family"].isin(interesting_families)
    & top_class_df["target"].isin(["submechanism", "risk_regime"])
].copy()

print(
    view.sort_values(by=["target", "heldout_family", "feature_set"])
    .to_string(index=False)
)

# ============================================================
# SAVE OUTPUTS
# ============================================================

feature_table_path = SAVE_DIR / "constraint_audit3b1_feature_table.csv"

feature_sets_path = SAVE_DIR / "constraint_audit3b1_feature_sets.csv"

binary_group_path = SAVE_DIR / "constraint_audit3b1_group_binary.csv"
multi_group_path = SAVE_DIR / "constraint_audit3b1_group_multiclass.csv"

binary_lofo_path = SAVE_DIR / "constraint_audit3b1_lofo_binary.csv"
binary_lofo_summary_path = SAVE_DIR / "constraint_audit3b1_lofo_binary_summary.csv"
binary_lofo_pred_path = SAVE_DIR / "constraint_audit3b1_lofo_binary_predictions.csv"

multi_lofo_path = SAVE_DIR / "constraint_audit3b1_lofo_multiclass.csv"
multi_lofo_summary_path = SAVE_DIR / "constraint_audit3b1_lofo_multiclass_summary.csv"
multi_lofo_pred_path = SAVE_DIR / "constraint_audit3b1_lofo_multiclass_predictions.csv"

top_class_path = SAVE_DIR / "constraint_audit3b1_lofo_top_class_summary.csv"

invariant_path = SAVE_DIR / "constraint_audit3b1_deltaR_invariant_ranking.csv"

coef_binary_path = SAVE_DIR / "constraint_audit3b1_binary_coefficients.csv"
coef_multi_path = SAVE_DIR / "constraint_audit3b1_submechanism_coefficients.csv"

family_delta_path = SAVE_DIR / "constraint_audit3b1_family_deltaR_summary.csv"

df.to_csv(feature_table_path, index=False)

feature_set_rows = []
for name, cols in feature_sets.items():
    for c in cols:
        feature_set_rows.append({
            "feature_set": name,
            "feature": c,
        })

pd.DataFrame(feature_set_rows).to_csv(feature_sets_path, index=False)

binary_group_df.to_csv(binary_group_path, index=False)
multi_group_df.to_csv(multi_group_path, index=False)

binary_lofo_df.to_csv(binary_lofo_path, index=False)
binary_lofo_summary.to_csv(binary_lofo_summary_path, index=False)
binary_lofo_pred_df.to_csv(binary_lofo_pred_path, index=False)

multi_lofo_df.to_csv(multi_lofo_path, index=False)
multi_lofo_summary.to_csv(multi_lofo_summary_path, index=False)
multi_lofo_pred_df.to_csv(multi_lofo_pred_path, index=False)

top_class_df.to_csv(top_class_path, index=False)

invariant_ranking.to_csv(invariant_path, index=False)

coef_binary_df.to_csv(coef_binary_path, index=False)
coef_multi_df.to_csv(coef_multi_path, index=False)

family_delta_summary.to_csv(family_delta_path, index=False)

print("\nSaved outputs:")
print(" ", feature_table_path)
print(" ", feature_sets_path)
print(" ", binary_group_path)
print(" ", multi_group_path)
print(" ", binary_lofo_path)
print(" ", binary_lofo_summary_path)
print(" ", binary_lofo_pred_path)
print(" ", multi_lofo_path)
print(" ", multi_lofo_summary_path)
print(" ", multi_lofo_pred_path)
print(" ", top_class_path)
print(" ", invariant_path)
print(" ", coef_binary_path)
print(" ", coef_multi_path)
print(" ", family_delta_path)

# ============================================================
# INTERPRETATION GUIDE
# ============================================================

print("\n\n============================================================")
print("Constraint-Audit-3B.1 Interpretation Guide")
print("============================================================\n")

print("Strong positive result if:")
print("  1. delta_R_minimal_top or delta_R_full_20_25 improves LOFO submechanism macro-F1 over baseline_RHpC_20_25.")
print("  2. delta_R_minimal_top or delta_R_full_20_25 improves LOFO risk_regime macro-F1 over baseline_RHpC_20_25.")
print("  3. delta_R_shape_20_25 performs close to delta_R_full_20_25.")
print("  4. invariant ranking is dominated by DELTA_R_slope / L24 / min / range / L25 / traj_l2.")
print()
print("Interpretation:")
print("  If delta_R_shape_20_25 beats delta_R_raw_20_25:")
print("    The structural signal is trajectory shape, not individual layer value.")
print()
print("  If delta_R_minimal_top performs close to delta_R_full_20_25:")
print("    A very small invariant basis may be enough.")
print()
print("  If absolute_R_full_20_25 beats delta_R_full_20_25:")
print("    Clean-relative differencing is not necessary; structure is already absolute.")
print()
print("  If delta_R_plus_delta_HpC improves over delta_R_full_20_25:")
print("    Delta-H / Delta-pC adds useful uncertainty or boundary information.")
print()
print("  If baseline remains stronger than all Delta-R sets:")
print("    3B's delta_C gain came from non-R components or from feature accumulation.")
print()
print("Target theory update if positive:")
print("  C_struct_20:25 = Shape(Delta R_20:25)")
print()
print("Done.")