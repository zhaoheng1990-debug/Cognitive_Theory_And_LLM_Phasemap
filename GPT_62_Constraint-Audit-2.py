# ============================================================
# Constraint-Audit-2A
# Mechanism-aware Constraint State Decomposition
#
# Input:
#   Reuse Constraint-Audit-1d output:
#   ./constraint_audit1d_outputs/constraint_audit1d_binary_dataset.csv
#
# Goal:
#   1e showed:
#     C_20:25 is a strong early-warning ranking diagnostic,
#     but unified cross-condition calibration fails.
#
#   Constraint-Audit-2A asks:
#     Can we decompose C_20:25 into mechanism coordinates?
#
# Mechanism heads:
#   stable_or_preserved:
#       clean / preserve_irrelevant / weak_distractor
#
#   competition_conflict:
#       ambiguous_branch / strong_branch_conflict / direct_location_conflict
#
#   closure_rewrite:
#       closure_negation_conflict
#
# Prediction heads:
#   Head-M:
#       mechanism classifier
#
#   Head-E:
#       Gen_E predictor without mechanism coordinates
#
#   Head-ME:
#       Gen_E predictor with predicted mechanism probabilities appended
#
# Outputs:
#   ./constraint_audit2_outputs/
#
# ============================================================

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    brier_score_loss,
    log_loss,
    classification_report,
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

RANDOM_SEED = 42

INPUT_FILE = Path("./constraint_audit1d_outputs/constraint_audit1d_binary_dataset.csv")

SAVE_DIR = Path("./constraint_audit2_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

CAL_SIZE = 0.30
N_GROUP_SPLITS = 5

# Main early-warning window from Constraint-Audit-1d.
PRIMARY_LAYERS = [20, 21, 22, 23, 24, 25]

SEGMENTS = {
    "ENTRY20_22": [20, 21, 22],
    "BASIN23_25": [23, 24, 25],
    "ALL20_25": [20, 21, 22, 23, 24, 25],
}

CONDITION_ORDER = [
    "clean",
    "preserve_irrelevant",
    "weak_distractor",
    "ambiguous_branch",
    "strong_branch_conflict",
    "direct_location_conflict",
    "closure_negation_conflict",
]

# Fine-grained mechanism labels.
FINE_MECHANISM_MAP = {
    "clean": "stable_clean",
    "preserve_irrelevant": "stable_irrelevant",
    "weak_distractor": "stable_weak",
    "ambiguous_branch": "ambiguous_competition",
    "strong_branch_conflict": "branch_conflict",
    "direct_location_conflict": "direct_conflict",
    "closure_negation_conflict": "closure_rewrite",
}

# Coarse labels are used for the main mechanism head.
# This reduces sparsity but still explicitly separates closure rewrite.
COARSE_MECHANISM_MAP = {
    "clean": "stable_or_preserved",
    "preserve_irrelevant": "stable_or_preserved",
    "weak_distractor": "stable_or_preserved",
    "ambiguous_branch": "competition_conflict",
    "strong_branch_conflict": "competition_conflict",
    "direct_location_conflict": "competition_conflict",
    "closure_negation_conflict": "closure_rewrite",
}

COARSE_MECH_CLASSES = [
    "stable_or_preserved",
    "competition_conflict",
    "closure_rewrite",
]

# ============================================================
# BASIC HELPERS
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
        out["tn"] = 0
        out["fp"] = 0
        out["fn"] = 0
        out["tp"] = 0

    return out


def best_threshold_on_calibration(y_cal, p_cal, objective="f1"):
    y_cal = np.asarray(y_cal).astype(int)
    p_cal = sanitize_prob(p_cal)

    qs = np.linspace(0.01, 0.99, 99)
    thresholds = np.unique(np.quantile(p_cal, qs))

    if len(thresholds) == 0:
        return 0.5, calc_binary_metrics(y_cal, p_cal, threshold=0.5)

    best_t = 0.5
    best_score = -1.0
    best_metrics = None

    for t in thresholds:
        m = calc_binary_metrics(y_cal, p_cal, threshold=float(t))
        score = m[objective]

        if score > best_score:
            best_score = score
            best_t = float(t)
            best_metrics = m

    return best_t, best_metrics


def split_core_calibration(y_trainval):
    idx = np.arange(len(y_trainval))
    y_trainval = np.asarray(y_trainval).astype(int)

    try:
        core_idx, cal_idx = train_test_split(
            idx,
            test_size=CAL_SIZE,
            random_state=RANDOM_SEED,
            stratify=y_trainval,
        )
    except Exception:
        core_idx, cal_idx = train_test_split(
            idx,
            test_size=CAL_SIZE,
            random_state=RANDOM_SEED,
            shuffle=True,
        )

    return core_idx, cal_idx


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_segment_features(df, segment_name, layers):
    out = df.copy()

    r_cols = [f"R_L{l}" for l in layers]
    h_cols = [f"H_L{l}" for l in layers]
    p_cols = [f"pC_L{l}" for l in layers]

    missing = [c for c in r_cols + h_cols + p_cols if c not in out.columns]
    if missing:
        raise RuntimeError(
            "Missing required columns from 1d binary dataset: "
            + ", ".join(missing[:10])
        )

    R = out[r_cols].values.astype(np.float32)
    H = out[h_cols].values.astype(np.float32)
    P = out[p_cols].values.astype(np.float32)

    prefix = segment_name

    out[f"{prefix}_R_mean"] = R.mean(axis=1)
    out[f"{prefix}_R_min"] = R.min(axis=1)
    out[f"{prefix}_R_max"] = R.max(axis=1)
    out[f"{prefix}_R_first"] = R[:, 0]
    out[f"{prefix}_R_last"] = R[:, -1]
    out[f"{prefix}_R_slope"] = R[:, -1] - R[:, 0]
    out[f"{prefix}_R_drop"] = R[:, 0] - R[:, -1]
    out[f"{prefix}_R_range"] = R.max(axis=1) - R.min(axis=1)

    out[f"{prefix}_absR_mean"] = np.abs(R).mean(axis=1)
    out[f"{prefix}_absR_min"] = np.abs(R).min(axis=1)
    out[f"{prefix}_absR_last"] = np.abs(R[:, -1])

    out[f"{prefix}_H_mean"] = H.mean(axis=1)
    out[f"{prefix}_H_max"] = H.max(axis=1)
    out[f"{prefix}_H_last"] = H[:, -1]
    out[f"{prefix}_H_slope"] = H[:, -1] - H[:, 0]

    out[f"{prefix}_pC_mean"] = P.mean(axis=1)
    out[f"{prefix}_pC_min"] = P.min(axis=1)
    out[f"{prefix}_pC_last"] = P[:, -1]

    out[f"{prefix}_num_negative"] = (R < 0).sum(axis=1).astype(np.float32)
    out[f"{prefix}_any_negative"] = (R < 0).any(axis=1).astype(np.float32)

    first_neg = []
    for row in R:
        neg = np.where(row < 0)[0]
        if len(neg) == 0:
            first_neg.append(float(len(layers)))
        else:
            first_neg.append(float(neg[0]))

    out[f"{prefix}_first_negative_offset"] = np.array(first_neg, dtype=np.float32)

    return out


def segment_feature_names(segment_name):
    prefix = segment_name
    return [
        f"{prefix}_R_mean",
        f"{prefix}_R_min",
        f"{prefix}_R_max",
        f"{prefix}_R_first",
        f"{prefix}_R_last",
        f"{prefix}_R_slope",
        f"{prefix}_R_drop",
        f"{prefix}_R_range",

        f"{prefix}_absR_mean",
        f"{prefix}_absR_min",
        f"{prefix}_absR_last",

        f"{prefix}_H_mean",
        f"{prefix}_H_max",
        f"{prefix}_H_last",
        f"{prefix}_H_slope",

        f"{prefix}_pC_mean",
        f"{prefix}_pC_min",
        f"{prefix}_pC_last",

        f"{prefix}_num_negative",
        f"{prefix}_any_negative",
        f"{prefix}_first_negative_offset",
    ]


def build_feature_table(df):
    out = df.copy()

    for seg_name, layers in SEGMENTS.items():
        out = add_segment_features(out, seg_name, layers)

    raw_cols = []
    for l in PRIMARY_LAYERS:
        raw_cols.extend([f"R_L{l}", f"H_L{l}", f"pC_L{l}"])

    feature_cols = []
    feature_cols.extend(raw_cols)

    for seg_name in SEGMENTS:
        feature_cols.extend(segment_feature_names(seg_name))

    # Remove accidental duplicates while preserving order.
    seen = set()
    feature_cols_unique = []
    for c in feature_cols:
        if c not in seen:
            seen.add(c)
            feature_cols_unique.append(c)

    return out, feature_cols_unique


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
    def __init__(self, classes, probs):
        self.classes_ = np.asarray(classes)
        probs = np.asarray(probs, dtype=np.float64)
        probs = probs / (probs.sum() + 1e-12)
        self.probs = probs

    def predict_proba(self, X):
        n = len(X)
        return np.tile(self.probs.reshape(1, -1), (n, 1))

    def predict(self, X):
        p = self.predict_proba(X)
        idx = np.argmax(p, axis=1)
        return self.classes_[idx]


def make_binary_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            max_iter=2000,
        )),
    ])


def make_mechanism_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            max_iter=3000,
        )),
    ])


def fit_binary_or_constant(X, y):
    y = np.asarray(y).astype(int)

    if len(np.unique(y)) < 2:
        return ConstantBinaryModel(float(y.mean()))

    model = make_binary_model()
    model.fit(X, y)
    return model


def fit_mechanism_or_constant(X, y_mech):
    y_mech = np.asarray(y_mech)

    values, counts = np.unique(y_mech, return_counts=True)

    if len(values) < 2:
        return ConstantMulticlassModel(values, np.ones(len(values)))

    model = make_mechanism_model()
    model.fit(X, y_mech)
    return model


def aligned_mechanism_proba(model, X, all_classes):
    """
    Return probabilities aligned to COARSE_MECH_CLASSES.
    Missing classes get probability 0.
    """
    p_raw = model.predict_proba(X)

    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    else:
        classes = list(model.named_steps["clf"].classes_)

    out = np.zeros((len(X), len(all_classes)), dtype=np.float64)

    for j, cls in enumerate(classes):
        if cls in all_classes:
            idx = all_classes.index(cls)
            out[:, idx] = p_raw[:, j]

    row_sum = out.sum(axis=1, keepdims=True)

    # If no known class is aligned, use uniform.
    bad = row_sum[:, 0] <= 1e-12
    if bad.any():
        out[bad, :] = 1.0 / len(all_classes)
        row_sum = out.sum(axis=1, keepdims=True)

    out = out / (row_sum + 1e-12)
    return out


# ============================================================
# EVALUATION: GROUP CV
# ============================================================

def evaluate_group_cv(df, feature_cols):
    X = df[feature_cols].values.astype(np.float32)
    y = df["y_conflict"].values.astype(int)
    y_mech = df["coarse_mechanism"].values
    groups = df["idx"].values

    gkf = GroupKFold(n_splits=N_GROUP_SPLITS)

    p_base = np.zeros(len(df), dtype=np.float64)
    p_mechaware = np.zeros(len(df), dtype=np.float64)
    mech_pred = np.empty(len(df), dtype=object)

    for fold, (trainval_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_trainval = X[trainval_idx]
        y_trainval = y[trainval_idx]
        m_trainval = y_mech[trainval_idx]

        X_test = X[test_idx]

        core_rel, cal_rel = split_core_calibration(y_trainval)

        X_core = X_trainval[core_rel]
        y_core = y_trainval[core_rel]
        m_core = m_trainval[core_rel]

        X_cal = X_trainval[cal_rel]
        y_cal = y_trainval[cal_rel]

        # ---------------- Baseline binary head ----------------
        base_bin = fit_binary_or_constant(X_core, y_core)

        p_cal_base = sanitize_prob(base_bin.predict_proba(X_cal)[:, 1])
        p_test_base = sanitize_prob(base_bin.predict_proba(X_test)[:, 1])

        t_base, _ = best_threshold_on_calibration(y_cal, p_cal_base, objective="f1")

        # Store probabilities, threshold applied later only in summaries.
        # For fair global metrics, use 0.5; threshold tuned metrics are computed separately in LOCO.
        p_base[test_idx] = p_test_base

        # ---------------- Mechanism head ----------------
        mech_model = fit_mechanism_or_constant(X_core, m_core)

        p_mech_core = aligned_mechanism_proba(mech_model, X_core, COARSE_MECH_CLASSES)
        p_mech_cal = aligned_mechanism_proba(mech_model, X_cal, COARSE_MECH_CLASSES)
        p_mech_test = aligned_mechanism_proba(mech_model, X_test, COARSE_MECH_CLASSES)

        mech_pred[test_idx] = np.array(COARSE_MECH_CLASSES)[np.argmax(p_mech_test, axis=1)]

        # ---------------- Mechanism-aware binary head ----------------
        X_core_aug = np.concatenate([X_core, p_mech_core], axis=1)
        X_cal_aug = np.concatenate([X_cal, p_mech_cal], axis=1)
        X_test_aug = np.concatenate([X_test, p_mech_test], axis=1)

        mechaware_bin = fit_binary_or_constant(X_core_aug, y_core)

        p_cal_me = sanitize_prob(mechaware_bin.predict_proba(X_cal_aug)[:, 1])
        p_test_me = sanitize_prob(mechaware_bin.predict_proba(X_test_aug)[:, 1])

        t_me, _ = best_threshold_on_calibration(y_cal, p_cal_me, objective="f1")

        p_mechaware[test_idx] = p_test_me

    base_metrics = calc_binary_metrics(y, p_base, threshold=0.5)
    me_metrics = calc_binary_metrics(y, p_mechaware, threshold=0.5)

    mech_acc = float(accuracy_score(y_mech, mech_pred))
    mech_macro_f1 = float(f1_score(y_mech, mech_pred, average="macro", zero_division=0))

    summary_rows = [
        {"model": "baseline_binary", **base_metrics},
        {"model": "mechanism_aware_binary", **me_metrics},
    ]

    mech_summary = {
        "mechanism_accuracy": mech_acc,
        "mechanism_macro_f1": mech_macro_f1,
    }

    pred_df = df[["condition", "idx", "y_conflict", "fine_mechanism", "coarse_mechanism"]].copy()
    pred_df["p_base"] = p_base
    pred_df["p_mechaware"] = p_mechaware
    pred_df["mech_pred"] = mech_pred

    return pd.DataFrame(summary_rows), mech_summary, pred_df


# ============================================================
# EVALUATION: LEAVE-ONE-CONDITION-OUT
# ============================================================

def evaluate_loco(df, feature_cols):
    X = df[feature_cols].values.astype(np.float32)
    y = df["y_conflict"].values.astype(int)
    y_mech = df["coarse_mechanism"].values
    conditions = df["condition"].values

    loco = LeaveOneGroupOut()

    rows = []
    pred_rows = []
    mech_rows = []

    for trainval_idx, test_idx in loco.split(X, y, groups=conditions):
        heldout = conditions[test_idx][0]

        X_trainval = X[trainval_idx]
        y_trainval = y[trainval_idx]
        m_trainval = y_mech[trainval_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]
        m_test = y_mech[test_idx]

        core_rel, cal_rel = split_core_calibration(y_trainval)

        X_core = X_trainval[core_rel]
        y_core = y_trainval[core_rel]
        m_core = m_trainval[core_rel]

        X_cal = X_trainval[cal_rel]
        y_cal = y_trainval[cal_rel]

        # ---------------- Baseline binary ----------------
        base_bin = fit_binary_or_constant(X_core, y_core)
        p_cal_base = sanitize_prob(base_bin.predict_proba(X_cal)[:, 1])
        p_test_base = sanitize_prob(base_bin.predict_proba(X_test)[:, 1])

        t_base, cal_base_metrics = best_threshold_on_calibration(
            y_cal,
            p_cal_base,
            objective="f1",
        )

        base_fixed = calc_binary_metrics(y_test, p_test_base, threshold=0.5)
        base_tuned = calc_binary_metrics(y_test, p_test_base, threshold=t_base)

        row = {
            "heldout_condition": heldout,
            "model": "baseline_binary",
            "decision_rule": "fixed_0.5",
            "n_test": int(len(y_test)),
            "positive_rate": float(y_test.mean()),
            "cal_threshold": 0.5,
        }
        row.update(base_fixed)
        rows.append(row)

        row = {
            "heldout_condition": heldout,
            "model": "baseline_binary",
            "decision_rule": "calibrated_threshold",
            "n_test": int(len(y_test)),
            "positive_rate": float(y_test.mean()),
            "cal_threshold": float(t_base),
        }
        row.update(base_tuned)
        rows.append(row)

        # ---------------- Mechanism head ----------------
        mech_model = fit_mechanism_or_constant(X_core, m_core)

        p_mech_core = aligned_mechanism_proba(mech_model, X_core, COARSE_MECH_CLASSES)
        p_mech_cal = aligned_mechanism_proba(mech_model, X_cal, COARSE_MECH_CLASSES)
        p_mech_test = aligned_mechanism_proba(mech_model, X_test, COARSE_MECH_CLASSES)

        mech_pred = np.array(COARSE_MECH_CLASSES)[np.argmax(p_mech_test, axis=1)]

        mech_acc = float(accuracy_score(m_test, mech_pred))
        mech_f1 = float(f1_score(m_test, mech_pred, average="macro", zero_division=0))

        mech_row = {
            "heldout_condition": heldout,
            "true_mechanism": str(pd.Series(m_test).mode().iloc[0]),
            "mech_accuracy": mech_acc,
            "mech_macro_f1": mech_f1,
        }

        for j, cls in enumerate(COARSE_MECH_CLASSES):
            mech_row[f"mean_p_{cls}"] = float(p_mech_test[:, j].mean())

        mech_rows.append(mech_row)

        # ---------------- Mechanism-aware binary ----------------
        X_core_aug = np.concatenate([X_core, p_mech_core], axis=1)
        X_cal_aug = np.concatenate([X_cal, p_mech_cal], axis=1)
        X_test_aug = np.concatenate([X_test, p_mech_test], axis=1)

        mechaware_bin = fit_binary_or_constant(X_core_aug, y_core)

        p_cal_me = sanitize_prob(mechaware_bin.predict_proba(X_cal_aug)[:, 1])
        p_test_me = sanitize_prob(mechaware_bin.predict_proba(X_test_aug)[:, 1])

        t_me, cal_me_metrics = best_threshold_on_calibration(
            y_cal,
            p_cal_me,
            objective="f1",
        )

        me_fixed = calc_binary_metrics(y_test, p_test_me, threshold=0.5)
        me_tuned = calc_binary_metrics(y_test, p_test_me, threshold=t_me)

        row = {
            "heldout_condition": heldout,
            "model": "mechanism_aware_binary",
            "decision_rule": "fixed_0.5",
            "n_test": int(len(y_test)),
            "positive_rate": float(y_test.mean()),
            "cal_threshold": 0.5,
        }
        row.update(me_fixed)
        rows.append(row)

        row = {
            "heldout_condition": heldout,
            "model": "mechanism_aware_binary",
            "decision_rule": "calibrated_threshold",
            "n_test": int(len(y_test)),
            "positive_rate": float(y_test.mean()),
            "cal_threshold": float(t_me),
        }
        row.update(me_tuned)
        rows.append(row)

        # Save sample-level predictions.
        temp = df.iloc[test_idx][[
            "condition",
            "idx",
            "y_conflict",
            "fine_mechanism",
            "coarse_mechanism",
        ]].copy()

        temp["p_base"] = p_test_base
        temp["p_mechaware"] = p_test_me
        temp["mech_pred"] = mech_pred

        for j, cls in enumerate(COARSE_MECH_CLASSES):
            temp[f"p_mech_{cls}"] = p_mech_test[:, j]

        pred_rows.append(temp)

    loco_df = pd.DataFrame(rows)
    pred_df = pd.concat(pred_rows, axis=0, ignore_index=True)
    mech_df = pd.DataFrame(mech_rows)

    return loco_df, pred_df, mech_df


# ============================================================
# HEAD ACTIVATION ANALYSIS
# ============================================================

def fit_full_mechanism_head(df, feature_cols):
    X = df[feature_cols].values.astype(np.float32)
    y_mech = df["coarse_mechanism"].values

    model = fit_mechanism_or_constant(X, y_mech)
    p = aligned_mechanism_proba(model, X, COARSE_MECH_CLASSES)

    out = df[["condition", "idx", "y_conflict", "fine_mechanism", "coarse_mechanism"]].copy()

    for j, cls in enumerate(COARSE_MECH_CLASSES):
        out[f"p_{cls}"] = p[:, j]

    rows = []
    for cond in CONDITION_ORDER:
        sub = out[out["condition"] == cond]
        if len(sub) == 0:
            continue

        row = {
            "condition": cond,
            "n": int(len(sub)),
            "genE_rate": float(sub["y_conflict"].mean()),
            "true_coarse_mechanism": str(sub["coarse_mechanism"].mode().iloc[0]),
        }

        for cls in COARSE_MECH_CLASSES:
            row[f"mean_p_{cls}"] = float(sub[f"p_{cls}"].mean())

        rows.append(row)

    return out, pd.DataFrame(rows)


def fit_full_binary_coefficients(df, feature_cols):
    X = df[feature_cols].values.astype(np.float32)
    y = df["y_conflict"].values.astype(int)

    model = fit_binary_or_constant(X, y)

    if not isinstance(model, Pipeline):
        return pd.DataFrame()

    clf = model.named_steps["clf"]
    coefs = clf.coef_[0]

    rows = []
    for feat, coef in sorted(zip(feature_cols, coefs), key=lambda x: abs(x[1]), reverse=True):
        rows.append({
            "feature": feat,
            "coef": float(coef),
            "abs_coef": float(abs(coef)),
        })

    return pd.DataFrame(rows)


# ============================================================
# SUMMARY HELPERS
# ============================================================

def summarize_loco(loco_df):
    rows = []

    for keys, sub in loco_df.groupby(["model", "decision_rule"]):
        model_name, rule = keys

        rows.append({
            "model": model_name,
            "decision_rule": rule,

            "mean_auc": float(np.nanmean(sub["auc"])),
            "mean_accuracy": float(np.nanmean(sub["accuracy"])),
            "mean_f1": float(np.nanmean(sub["f1"])),
            "mean_precision": float(np.nanmean(sub["precision"])),
            "mean_recall": float(np.nanmean(sub["recall"])),
            "mean_brier": float(np.nanmean(sub["brier"])),
            "mean_nll": float(np.nanmean(sub["nll"])),
            "mean_pred_pos_rate": float(np.nanmean(sub["pred_pos_rate"])),
            "std_pred_pos_rate": float(np.nanstd(sub["pred_pos_rate"])),
        })

    return pd.DataFrame(rows)


def summarize_loco_by_condition(loco_df):
    cols = [
        "heldout_condition",
        "model",
        "decision_rule",
        "positive_rate",
        "pred_pos_rate",
        "mean_prob",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "brier",
        "auc",
        "cal_threshold",
    ]

    return loco_df[cols].sort_values(
        by=["model", "decision_rule", "heldout_condition"]
    )


# ============================================================
# MAIN
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find input file: {INPUT_FILE}\n"
        f"Run Constraint-Audit-1d first, or change INPUT_FILE."
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

# Keep binary C/E rows.
if "is_clean" in df.columns and "is_conflict" in df.columns:
    df = df[(df["is_clean"] == True) | (df["is_conflict"] == True)].copy()

# Add mechanism labels.
df["fine_mechanism"] = df["condition"].map(FINE_MECHANISM_MAP)
df["coarse_mechanism"] = df["condition"].map(COARSE_MECHANISM_MAP)

if df["fine_mechanism"].isna().any() or df["coarse_mechanism"].isna().any():
    bad = df[df["coarse_mechanism"].isna()]["condition"].unique().tolist()
    raise RuntimeError(f"Unknown condition(s) in input: {bad}")

df_feat, feature_cols = build_feature_table(df)

print("\n============================================================")
print("Constraint-Audit-2A Input Summary")
print("============================================================\n")

print("Input file:", INPUT_FILE)
print("Rows:", len(df_feat))
print("Feature count:", len(feature_cols))

print("\nCondition summary:")
print(
    f"{'condition':<30}"
    f"{'N':<6}"
    f"{'GenE':<10}"
    f"{'coarse_mechanism':<24}"
)

for cond in CONDITION_ORDER:
    sub = df_feat[df_feat["condition"] == cond]
    if len(sub) == 0:
        continue

    mech = sub["coarse_mechanism"].mode().iloc[0]

    print(
        f"{cond:<30}"
        f"{len(sub):<6}"
        f"{sub['y_conflict'].mean():<10.4f}"
        f"{mech:<24}"
    )

# ---------------- Group CV ----------------

print("\n\n============================================================")
print("GroupKFold by graph idx")
print("============================================================\n")

group_summary, mech_summary, group_pred_df = evaluate_group_cv(df_feat, feature_cols)

print("Binary prediction summary:")
print(
    f"{'model':<28}"
    f"{'AUC':<10}"
    f"{'Acc':<10}"
    f"{'F1':<10}"
    f"{'Brier':<10}"
    f"{'PredE':<10}"
)

for _, r in group_summary.iterrows():
    print(
        f"{r['model']:<28}"
        f"{r['auc']:<10.4f}"
        f"{r['accuracy']:<10.4f}"
        f"{r['f1']:<10.4f}"
        f"{r['brier']:<10.4f}"
        f"{r['pred_pos_rate']:<10.4f}"
    )

print("\nMechanism head summary:")
print(f"  mechanism_accuracy = {mech_summary['mechanism_accuracy']:.4f}")
print(f"  mechanism_macro_f1 = {mech_summary['mechanism_macro_f1']:.4f}")

# ---------------- LOCO ----------------

print("\n\n============================================================")
print("Leave-One-Condition-Out")
print("============================================================\n")

loco_df, loco_pred_df, loco_mech_df = evaluate_loco(df_feat, feature_cols)

loco_summary = summarize_loco(loco_df)
loco_condition = summarize_loco_by_condition(loco_df)

print("LOCO summary:")
print(
    f"{'model':<28}"
    f"{'rule':<24}"
    f"{'AUC':<10}"
    f"{'Acc':<10}"
    f"{'F1':<10}"
    f"{'Brier':<10}"
    f"{'PredE':<10}"
)

for _, r in loco_summary.iterrows():
    print(
        f"{r['model']:<28}"
        f"{r['decision_rule']:<24}"
        f"{r['mean_auc']:<10.4f}"
        f"{r['mean_accuracy']:<10.4f}"
        f"{r['mean_f1']:<10.4f}"
        f"{r['mean_brier']:<10.4f}"
        f"{r['mean_pred_pos_rate']:<10.4f}"
    )

print("\nCondition-wise LOCO, calibrated threshold:")
view = loco_condition[loco_condition["decision_rule"] == "calibrated_threshold"]

print(
    f"{'condition':<30}"
    f"{'model':<28}"
    f"{'pos':<8}"
    f"{'predE':<8}"
    f"{'pE':<8}"
    f"{'acc':<8}"
    f"{'f1':<8}"
    f"{'brier':<10}"
)

for _, r in view.iterrows():
    print(
        f"{r['heldout_condition']:<30}"
        f"{r['model']:<28}"
        f"{r['positive_rate']:<8.4f}"
        f"{r['pred_pos_rate']:<8.4f}"
        f"{r['mean_prob']:<8.4f}"
        f"{r['accuracy']:<8.4f}"
        f"{r['f1']:<8.4f}"
        f"{r['brier']:<10.4f}"
    )

print("\nLOCO mechanism head diagnostics:")
print(
    f"{'heldout_condition':<30}"
    f"{'true_mech':<24}"
    f"{'acc':<8}"
    f"{'f1':<8}"
    f"{'p_stable':<10}"
    f"{'p_comp':<10}"
    f"{'p_closure':<10}"
)

for _, r in loco_mech_df.iterrows():
    print(
        f"{r['heldout_condition']:<30}"
        f"{r['true_mechanism']:<24}"
        f"{r['mech_accuracy']:<8.4f}"
        f"{r['mech_macro_f1']:<8.4f}"
        f"{r['mean_p_stable_or_preserved']:<10.4f}"
        f"{r['mean_p_competition_conflict']:<10.4f}"
        f"{r['mean_p_closure_rewrite']:<10.4f}"
    )

# ---------------- Full head activations ----------------

print("\n\n============================================================")
print("Full-data Mechanism Head Activations")
print("============================================================\n")

head_pred_df, head_activation_df = fit_full_mechanism_head(df_feat, feature_cols)

print(
    f"{'condition':<30}"
    f"{'GenE':<10}"
    f"{'true_mech':<24}"
    f"{'p_stable':<10}"
    f"{'p_comp':<10}"
    f"{'p_closure':<10}"
)

for _, r in head_activation_df.iterrows():
    print(
        f"{r['condition']:<30}"
        f"{r['genE_rate']:<10.4f}"
        f"{r['true_coarse_mechanism']:<24}"
        f"{r['mean_p_stable_or_preserved']:<10.4f}"
        f"{r['mean_p_competition_conflict']:<10.4f}"
        f"{r['mean_p_closure_rewrite']:<10.4f}"
    )

coef_df = fit_full_binary_coefficients(df_feat, feature_cols)

# ============================================================
# SAVE OUTPUTS
# ============================================================

feature_path = SAVE_DIR / "constraint_audit2_feature_table.csv"
group_summary_path = SAVE_DIR / "constraint_audit2_groupcv_binary_summary.csv"
group_pred_path = SAVE_DIR / "constraint_audit2_groupcv_predictions.csv"
loco_path = SAVE_DIR / "constraint_audit2_loco_metrics.csv"
loco_summary_path = SAVE_DIR / "constraint_audit2_loco_summary.csv"
loco_condition_path = SAVE_DIR / "constraint_audit2_loco_condition_summary.csv"
loco_pred_path = SAVE_DIR / "constraint_audit2_loco_predictions.csv"
loco_mech_path = SAVE_DIR / "constraint_audit2_loco_mechanism_diagnostics.csv"
head_pred_path = SAVE_DIR / "constraint_audit2_mechanism_head_predictions.csv"
head_activation_path = SAVE_DIR / "constraint_audit2_mechanism_head_activations.csv"
coef_path = SAVE_DIR / "constraint_audit2_binary_feature_coefficients.csv"

df_feat.to_csv(feature_path, index=False)
group_summary.to_csv(group_summary_path, index=False)
group_pred_df.to_csv(group_pred_path, index=False)
loco_df.to_csv(loco_path, index=False)
loco_summary.to_csv(loco_summary_path, index=False)
loco_condition.to_csv(loco_condition_path, index=False)
loco_pred_df.to_csv(loco_pred_path, index=False)
loco_mech_df.to_csv(loco_mech_path, index=False)
head_pred_df.to_csv(head_pred_path, index=False)
head_activation_df.to_csv(head_activation_path, index=False)
coef_df.to_csv(coef_path, index=False)

print("\nSaved outputs:")
print(" ", feature_path)
print(" ", group_summary_path)
print(" ", group_pred_path)
print(" ", loco_path)
print(" ", loco_summary_path)
print(" ", loco_condition_path)
print(" ", loco_pred_path)
print(" ", loco_mech_path)
print(" ", head_pred_path)
print(" ", head_activation_path)
print(" ", coef_path)

# ============================================================
# INTERPRETATION GUIDE
# ============================================================

print("\n\n============================================================")
print("Constraint-Audit-2A Interpretation Guide")
print("============================================================\n")

print("Strong positive result if:")
print("  1. Mechanism head has high GroupKFold accuracy / macro-F1.")
print("  2. Mechanism-aware binary improves Brier / F1 / condition-wise predE.")
print("  3. clean false positive decreases.")
print("  4. closure false negative decreases, if closure mechanism can be inferred.")
print()
print("Important caveat:")
print("  closure_negation_conflict is the only closure_rewrite condition.")
print("  In leave-one-condition-out, when closure is held out, the training set has no closure_rewrite examples.")
print("  Therefore failure on held-out closure does not falsify mechanism decomposition.")
print("  It means Constraint-Audit-2B must add multiple closure rewrite templates.")
print()
print("Read the results as follows:")
print("  GroupKFold strong + LOCO closure weak:")
print("    mechanism decomposition works in-distribution, but closure mechanism needs template diversity.")
print()
print("  Mechanism head weak:")
print("    W20-W25 does not cleanly separate mechanisms; need richer C_l features.")
print()
print("  Mechanism head strong but Gen_E not improved:")
print("    mechanism is separable, but the binary risk head still lacks the right closure/support variables.")
print()
print("Suggested next step if 2A succeeds:")
print("  Constraint-Audit-2B:")
print("    Generate multiple templates per mechanism class.")
print("    Especially add several closure_rewrite variants.")
print("    Then repeat LOCO by template-family instead of by single condition.")
print()
print("Done.")