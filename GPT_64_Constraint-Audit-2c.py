# ============================================================
# Constraint-Audit-2C
# Submechanism Decomposition
#
# Input:
#   Reuse Constraint-Audit-2B output:
#   ./constraint_audit2b_outputs/constraint_audit2b_feature_table.csv
#
# Goal:
#   Constraint-Audit-2B showed that coarse mechanisms exist:
#       stable_or_preserved / competition_conflict / closure_rewrite
#
#   But 2B also showed:
#       stable_paraphrase and stable_irrelevant partly activate competition;
#       competition families have very different GenE rates;
#       closure_override is weaker / more mixed than canonical closure.
#
#   2C asks whether C_20:25 contains submechanism coordinates:
#
#       stable_core
#       stable_surface_shift
#       competition_low_commit
#       competition_ambiguous
#       competition_high_commit
#       closure_canonical_rewrite
#       closure_rule_override
#
# Main tests:
#   1. Can W20-W25 classify submechanisms under GroupKFold by graph?
#   2. Can submechanism-aware GenE prediction improve calibration?
#   3. Can leave-one-family-out recover held-out family submechanism?
#   4. Which submechanisms explain stable/competition boundary mixing?
#
# Outputs:
#   ./constraint_audit2c_outputs/
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
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

RANDOM_SEED = 42

INPUT_FILE = Path("./constraint_audit2b_outputs/constraint_audit2b_feature_table.csv")

SAVE_DIR = Path("./constraint_audit2c_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

CAL_SIZE = 0.30
N_GROUP_SPLITS = 5

TRACK_LAYERS = [20, 21, 22, 23, 24, 25]

SEGMENTS = {
    "ENTRY20_22": [20, 21, 22],
    "BASIN23_25": [23, 24, 25],
    "ALL20_25": [20, 21, 22, 23, 24, 25],
}

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
    # Stable submechanisms.
    # stable_paraphrase / stable_irrelevant showed partial competition activation in 2B,
    # so they are separated from stable_core.
    "stable_clean_basic": "stable_core",
    "stable_redundant": "stable_core",
    "stable_weak_note": "stable_core",
    "stable_paraphrase": "stable_surface_shift",
    "stable_irrelevant": "stable_surface_shift",

    # Competition submechanisms.
    # branch/direct had low GenE in 2B; source/equal had high GenE.
    "competition_branch": "competition_low_commit",
    "competition_direct": "competition_low_commit",
    "competition_ambiguous": "competition_ambiguous",
    "competition_source_claim": "competition_high_commit",
    "competition_equal_evidence": "competition_high_commit",

    # Closure submechanisms.
    # override/exception were weaker/more mixed than canonical rewrite templates.
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

COARSE_CLASSES = [
    "stable_or_preserved",
    "competition_conflict",
    "closure_rewrite",
]

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

# For LOFO, these submechanisms have more than one template family remaining
# after holding one family out. competition_ambiguous has only one family, so
# LOFO cannot truly test submechanism generalization for that class.
SINGLE_FAMILY_SUBMECHANISMS = {"competition_ambiguous"}

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

    thresholds = np.unique(np.quantile(p_cal, np.linspace(0.01, 0.99, 99)))

    if len(thresholds) == 0:
        return 0.5, calc_binary_metrics(y_cal, p_cal, threshold=0.5)

    best_t = 0.5
    best_score = -1.0
    best_metrics = None

    for t in thresholds:
        m = calc_binary_metrics(y_cal, p_cal, threshold=float(t))
        score = m.get(objective, 0.0)

        if score > best_score:
            best_t = float(t)
            best_score = score
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
            "Missing required columns from 2B feature table: "
            + ", ".join(missing[:10])
        )

    R = out[r_cols].values.astype(np.float32)
    H = out[h_cols].values.astype(np.float32)
    P = out[p_cols].values.astype(np.float32)

    prefix = segment_name

    # If already exists, overwrite deliberately for reproducibility.
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
    for l in TRACK_LAYERS:
        raw_cols.extend([f"R_L{l}", f"H_L{l}", f"pC_L{l}"])

    feature_cols = []
    feature_cols.extend(raw_cols)

    for seg_name in SEGMENTS:
        feature_cols.extend(segment_feature_names(seg_name))

    # Remove duplicates while preserving order.
    seen = set()
    unique_cols = []
    for c in feature_cols:
        if c not in seen:
            seen.add(c)
            unique_cols.append(c)

    return out, unique_cols


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
    def __init__(self, classes, probs=None):
        self.classes_ = np.asarray(classes)

        if probs is None:
            probs = np.ones(len(classes), dtype=np.float64)

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
            max_iter=3000,
        )),
    ])


def make_multiclass_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            max_iter=5000,
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
    vals, counts = np.unique(y, return_counts=True)

    if len(vals) < 2:
        return ConstantMulticlassModel(vals)

    model = make_multiclass_model()
    model.fit(X, y)
    return model


def aligned_proba(model, X, all_classes):
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

    bad = row_sum[:, 0] <= 1e-12
    if bad.any():
        out[bad, :] = 1.0 / len(all_classes)
        row_sum = out.sum(axis=1, keepdims=True)

    out = out / (row_sum + 1e-12)
    return out


# ============================================================
# GROUP CV
# ============================================================

def evaluate_group_cv(df, feature_cols):
    X = df[feature_cols].values.astype(np.float32)
    y = df["y_conflict"].values.astype(int)
    sub = df["submechanism"].values
    risk = df["risk_regime"].values
    groups = df["idx"].values

    gkf = GroupKFold(n_splits=N_GROUP_SPLITS)

    p_base = np.zeros(len(df), dtype=np.float64)
    p_subaware = np.zeros(len(df), dtype=np.float64)
    p_riskaware = np.zeros(len(df), dtype=np.float64)

    sub_pred = np.empty(len(df), dtype=object)
    risk_pred = np.empty(len(df), dtype=object)

    for fold, (trainval_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_trainval = X[trainval_idx]
        y_trainval = y[trainval_idx]
        sub_trainval = sub[trainval_idx]
        risk_trainval = risk[trainval_idx]

        X_test = X[test_idx]

        core_rel, cal_rel = split_core_calibration(y_trainval)

        X_core = X_trainval[core_rel]
        y_core = y_trainval[core_rel]
        sub_core = sub_trainval[core_rel]
        risk_core = risk_trainval[core_rel]

        # ---------------- Baseline GenE head ----------------
        base_bin = fit_binary_or_constant(X_core, y_core)
        p_base[test_idx] = sanitize_prob(base_bin.predict_proba(X_test)[:, 1])

        # ---------------- Submechanism head ----------------
        sub_model = fit_multiclass_or_constant(X_core, sub_core)

        p_sub_core = aligned_proba(sub_model, X_core, SUBMECHANISM_CLASSES)
        p_sub_test = aligned_proba(sub_model, X_test, SUBMECHANISM_CLASSES)

        sub_pred[test_idx] = np.array(SUBMECHANISM_CLASSES)[np.argmax(p_sub_test, axis=1)]

        X_core_sub_aug = np.concatenate([X_core, p_sub_core], axis=1)
        X_test_sub_aug = np.concatenate([X_test, p_sub_test], axis=1)

        subaware_bin = fit_binary_or_constant(X_core_sub_aug, y_core)
        p_subaware[test_idx] = sanitize_prob(subaware_bin.predict_proba(X_test_sub_aug)[:, 1])

        # ---------------- Risk-regime head ----------------
        risk_model = fit_multiclass_or_constant(X_core, risk_core)

        p_risk_core = aligned_proba(risk_model, X_core, RISK_REGIME_CLASSES)
        p_risk_test = aligned_proba(risk_model, X_test, RISK_REGIME_CLASSES)

        risk_pred[test_idx] = np.array(RISK_REGIME_CLASSES)[np.argmax(p_risk_test, axis=1)]

        X_core_risk_aug = np.concatenate([X_core, p_risk_core], axis=1)
        X_test_risk_aug = np.concatenate([X_test, p_risk_test], axis=1)

        riskaware_bin = fit_binary_or_constant(X_core_risk_aug, y_core)
        p_riskaware[test_idx] = sanitize_prob(riskaware_bin.predict_proba(X_test_risk_aug)[:, 1])

    summary_rows = [
        {"model": "baseline_binary", **calc_binary_metrics(y, p_base, threshold=0.5)},
        {"model": "submechanism_aware_binary", **calc_binary_metrics(y, p_subaware, threshold=0.5)},
        {"model": "risk_regime_aware_binary", **calc_binary_metrics(y, p_riskaware, threshold=0.5)},
    ]

    sub_acc = float(accuracy_score(sub, sub_pred))
    sub_macro_f1 = float(f1_score(sub, sub_pred, average="macro", zero_division=0))

    risk_acc = float(accuracy_score(risk, risk_pred))
    risk_macro_f1 = float(f1_score(risk, risk_pred, average="macro", zero_division=0))

    pred_df = df[[
        "idx",
        "family",
        "mechanism",
        "submechanism",
        "risk_regime",
        "y_conflict",
    ]].copy()

    pred_df["p_base"] = p_base
    pred_df["p_subaware"] = p_subaware
    pred_df["p_riskaware"] = p_riskaware
    pred_df["sub_pred"] = sub_pred
    pred_df["risk_pred"] = risk_pred

    aux_summary = {
        "submechanism_accuracy": sub_acc,
        "submechanism_macro_f1": sub_macro_f1,
        "risk_regime_accuracy": risk_acc,
        "risk_regime_macro_f1": risk_macro_f1,
    }

    return pd.DataFrame(summary_rows), aux_summary, pred_df


# ============================================================
# LEAVE-ONE-FAMILY-OUT
# ============================================================

def evaluate_leave_family_out(df, feature_cols):
    X = df[feature_cols].values.astype(np.float32)
    y = df["y_conflict"].values.astype(int)
    sub = df["submechanism"].values
    risk = df["risk_regime"].values
    families = df["family"].values

    logo = LeaveOneGroupOut()

    metric_rows = []
    sub_rows = []
    risk_rows = []
    pred_rows = []

    for trainval_idx, test_idx in logo.split(X, y, groups=families):
        heldout = families[test_idx][0]
        heldout_sub = SUBMECHANISM_MAP[heldout]
        heldout_risk = RISK_REGIME_MAP[heldout]

        X_trainval = X[trainval_idx]
        y_trainval = y[trainval_idx]
        sub_trainval = sub[trainval_idx]
        risk_trainval = risk[trainval_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]
        sub_test = sub[test_idx]
        risk_test = risk[test_idx]

        core_rel, cal_rel = split_core_calibration(y_trainval)

        X_core = X_trainval[core_rel]
        y_core = y_trainval[core_rel]
        sub_core = sub_trainval[core_rel]
        risk_core = risk_trainval[core_rel]

        X_cal = X_trainval[cal_rel]
        y_cal = y_trainval[cal_rel]

        # ---------------- Baseline binary ----------------
        base_bin = fit_binary_or_constant(X_core, y_core)

        p_cal_base = sanitize_prob(base_bin.predict_proba(X_cal)[:, 1])
        p_test_base = sanitize_prob(base_bin.predict_proba(X_test)[:, 1])

        t_base, _ = best_threshold_on_calibration(y_cal, p_cal_base, objective="f1")

        for rule, thr in [("fixed_0.5", 0.5), ("calibrated_threshold", t_base)]:
            metrics = calc_binary_metrics(y_test, p_test_base, threshold=thr)
            row = {
                "heldout_family": heldout,
                "heldout_mechanism": COARSE_MECHANISM_MAP[heldout],
                "heldout_submechanism": heldout_sub,
                "heldout_risk_regime": heldout_risk,
                "model": "baseline_binary",
                "decision_rule": rule,
                "n_test": int(len(y_test)),
                "positive_rate": float(y_test.mean()),
                "cal_threshold": float(thr),
            }
            row.update(metrics)
            metric_rows.append(row)

        # ---------------- Submechanism head ----------------
        sub_model = fit_multiclass_or_constant(X_core, sub_core)

        p_sub_core = aligned_proba(sub_model, X_core, SUBMECHANISM_CLASSES)
        p_sub_cal = aligned_proba(sub_model, X_cal, SUBMECHANISM_CLASSES)
        p_sub_test = aligned_proba(sub_model, X_test, SUBMECHANISM_CLASSES)

        sub_pred = np.array(SUBMECHANISM_CLASSES)[np.argmax(p_sub_test, axis=1)]

        train_has_sub = heldout_sub in set(sub_core)

        sub_acc = float(accuracy_score(sub_test, sub_pred))
        sub_macro_f1 = float(f1_score(sub_test, sub_pred, average="macro", zero_division=0))

        sub_row = {
            "heldout_family": heldout,
            "heldout_mechanism": COARSE_MECHANISM_MAP[heldout],
            "heldout_submechanism": heldout_sub,
            "train_has_heldout_submechanism": bool(train_has_sub),
            "submechanism_accuracy": sub_acc,
            "submechanism_macro_f1": sub_macro_f1,
        }

        for j, cls in enumerate(SUBMECHANISM_CLASSES):
            sub_row[f"mean_p_{cls}"] = float(p_sub_test[:, j].mean())

        sub_rows.append(sub_row)

        # Submechanism-aware binary.
        X_core_sub_aug = np.concatenate([X_core, p_sub_core], axis=1)
        X_cal_sub_aug = np.concatenate([X_cal, p_sub_cal], axis=1)
        X_test_sub_aug = np.concatenate([X_test, p_sub_test], axis=1)

        subaware_bin = fit_binary_or_constant(X_core_sub_aug, y_core)

        p_cal_subaware = sanitize_prob(subaware_bin.predict_proba(X_cal_sub_aug)[:, 1])
        p_test_subaware = sanitize_prob(subaware_bin.predict_proba(X_test_sub_aug)[:, 1])

        t_sub, _ = best_threshold_on_calibration(y_cal, p_cal_subaware, objective="f1")

        for rule, thr in [("fixed_0.5", 0.5), ("calibrated_threshold", t_sub)]:
            metrics = calc_binary_metrics(y_test, p_test_subaware, threshold=thr)
            row = {
                "heldout_family": heldout,
                "heldout_mechanism": COARSE_MECHANISM_MAP[heldout],
                "heldout_submechanism": heldout_sub,
                "heldout_risk_regime": heldout_risk,
                "model": "submechanism_aware_binary",
                "decision_rule": rule,
                "n_test": int(len(y_test)),
                "positive_rate": float(y_test.mean()),
                "cal_threshold": float(thr),
            }
            row.update(metrics)
            metric_rows.append(row)

        # ---------------- Risk-regime head ----------------
        risk_model = fit_multiclass_or_constant(X_core, risk_core)

        p_risk_core = aligned_proba(risk_model, X_core, RISK_REGIME_CLASSES)
        p_risk_cal = aligned_proba(risk_model, X_cal, RISK_REGIME_CLASSES)
        p_risk_test = aligned_proba(risk_model, X_test, RISK_REGIME_CLASSES)

        risk_pred = np.array(RISK_REGIME_CLASSES)[np.argmax(p_risk_test, axis=1)]

        train_has_risk = heldout_risk in set(risk_core)

        risk_acc = float(accuracy_score(risk_test, risk_pred))
        risk_macro_f1 = float(f1_score(risk_test, risk_pred, average="macro", zero_division=0))

        risk_row = {
            "heldout_family": heldout,
            "heldout_mechanism": COARSE_MECHANISM_MAP[heldout],
            "heldout_submechanism": heldout_sub,
            "heldout_risk_regime": heldout_risk,
            "train_has_heldout_risk_regime": bool(train_has_risk),
            "risk_regime_accuracy": risk_acc,
            "risk_regime_macro_f1": risk_macro_f1,
        }

        for j, cls in enumerate(RISK_REGIME_CLASSES):
            risk_row[f"mean_p_{cls}"] = float(p_risk_test[:, j].mean())

        risk_rows.append(risk_row)

        # Risk-regime-aware binary.
        X_core_risk_aug = np.concatenate([X_core, p_risk_core], axis=1)
        X_cal_risk_aug = np.concatenate([X_cal, p_risk_cal], axis=1)
        X_test_risk_aug = np.concatenate([X_test, p_risk_test], axis=1)

        riskaware_bin = fit_binary_or_constant(X_core_risk_aug, y_core)

        p_cal_riskaware = sanitize_prob(riskaware_bin.predict_proba(X_cal_risk_aug)[:, 1])
        p_test_riskaware = sanitize_prob(riskaware_bin.predict_proba(X_test_risk_aug)[:, 1])

        t_risk, _ = best_threshold_on_calibration(y_cal, p_cal_riskaware, objective="f1")

        for rule, thr in [("fixed_0.5", 0.5), ("calibrated_threshold", t_risk)]:
            metrics = calc_binary_metrics(y_test, p_test_riskaware, threshold=thr)
            row = {
                "heldout_family": heldout,
                "heldout_mechanism": COARSE_MECHANISM_MAP[heldout],
                "heldout_submechanism": heldout_sub,
                "heldout_risk_regime": heldout_risk,
                "model": "risk_regime_aware_binary",
                "decision_rule": rule,
                "n_test": int(len(y_test)),
                "positive_rate": float(y_test.mean()),
                "cal_threshold": float(thr),
            }
            row.update(metrics)
            metric_rows.append(row)

        # Sample predictions.
        temp = df.iloc[test_idx][[
            "idx",
            "family",
            "mechanism",
            "submechanism",
            "risk_regime",
            "y_conflict",
        ]].copy()

        temp["p_base"] = p_test_base
        temp["p_subaware"] = p_test_subaware
        temp["p_riskaware"] = p_test_riskaware
        temp["sub_pred"] = sub_pred
        temp["risk_pred"] = risk_pred

        for j, cls in enumerate(SUBMECHANISM_CLASSES):
            temp[f"p_sub_{cls}"] = p_sub_test[:, j]

        for j, cls in enumerate(RISK_REGIME_CLASSES):
            temp[f"p_risk_{cls}"] = p_risk_test[:, j]

        pred_rows.append(temp)

    metrics_df = pd.DataFrame(metric_rows)
    sub_df = pd.DataFrame(sub_rows)
    risk_df = pd.DataFrame(risk_rows)
    pred_df = pd.concat(pred_rows, axis=0, ignore_index=True)

    return metrics_df, sub_df, risk_df, pred_df


# ============================================================
# FULL-DATA ACTIVATION ANALYSIS
# ============================================================

def full_data_head_activations(df, feature_cols):
    X = df[feature_cols].values.astype(np.float32)

    sub_y = df["submechanism"].values
    risk_y = df["risk_regime"].values

    sub_model = fit_multiclass_or_constant(X, sub_y)
    risk_model = fit_multiclass_or_constant(X, risk_y)

    p_sub = aligned_proba(sub_model, X, SUBMECHANISM_CLASSES)
    p_risk = aligned_proba(risk_model, X, RISK_REGIME_CLASSES)

    out = df[[
        "idx",
        "family",
        "mechanism",
        "submechanism",
        "risk_regime",
        "y_conflict",
    ]].copy()

    for j, cls in enumerate(SUBMECHANISM_CLASSES):
        out[f"p_sub_{cls}"] = p_sub[:, j]

    for j, cls in enumerate(RISK_REGIME_CLASSES):
        out[f"p_risk_{cls}"] = p_risk[:, j]

    family_rows = []

    for fam in FAMILY_ORDER:
        sub = out[out["family"] == fam]
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

        for cls in SUBMECHANISM_CLASSES:
            row[f"mean_p_sub_{cls}"] = float(sub[f"p_sub_{cls}"].mean())

        for cls in RISK_REGIME_CLASSES:
            row[f"mean_p_risk_{cls}"] = float(sub[f"p_risk_{cls}"].mean())

        family_rows.append(row)

    return out, pd.DataFrame(family_rows)


def full_data_binary_coefficients(df, feature_cols):
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

def summarize_binary_metrics(metrics_df):
    rows = []

    for keys, sub in metrics_df.groupby(["model", "decision_rule"]):
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


def summarize_binary_by_family(metrics_df):
    cols = [
        "heldout_family",
        "heldout_mechanism",
        "heldout_submechanism",
        "heldout_risk_regime",
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

    return metrics_df[cols].sort_values(
        by=["heldout_mechanism", "heldout_submechanism", "heldout_family", "model", "decision_rule"]
    )


def summarize_binary_by_submechanism(metrics_df):
    rows = []

    for keys, sub in metrics_df.groupby(["heldout_submechanism", "model", "decision_rule"]):
        submech, model_name, rule = keys

        rows.append({
            "heldout_submechanism": submech,
            "model": model_name,
            "decision_rule": rule,
            "mean_positive_rate": float(np.nanmean(sub["positive_rate"])),
            "mean_pred_pos_rate": float(np.nanmean(sub["pred_pos_rate"])),
            "mean_accuracy": float(np.nanmean(sub["accuracy"])),
            "mean_f1": float(np.nanmean(sub["f1"])),
            "mean_brier": float(np.nanmean(sub["brier"])),
            "mean_auc": float(np.nanmean(sub["auc"])),
        })

    return pd.DataFrame(rows).sort_values(
        by=["heldout_submechanism", "model", "decision_rule"]
    )


# ============================================================
# MAIN
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find input file: {INPUT_FILE}\n"
        f"Run Constraint-Audit-2B first, or update INPUT_FILE."
    )

df = pd.read_csv(INPUT_FILE)

# Normalize booleans if they exist.
for col in ["is_clean", "is_conflict", "is_other"]:
    if col in df.columns:
        df[col] = parse_bool_series(df[col])

# Build target if missing.
if "y_conflict" not in df.columns:
    if "is_conflict" not in df.columns:
        raise RuntimeError("Need y_conflict or is_conflict column.")
    df["y_conflict"] = df["is_conflict"].astype(int)

df["y_conflict"] = df["y_conflict"].astype(int)

# Keep binary clean/conflict rows if possible.
if "is_clean" in df.columns and "is_conflict" in df.columns:
    df = df[(df["is_clean"] == True) | (df["is_conflict"] == True)].copy()

# Add mechanism labels.
if "family" not in df.columns:
    raise RuntimeError("Input file must contain family column from Constraint-Audit-2B.")

df["mechanism"] = df["family"].map(COARSE_MECHANISM_MAP)
df["submechanism"] = df["family"].map(SUBMECHANISM_MAP)
df["risk_regime"] = df["family"].map(RISK_REGIME_MAP)

if df["mechanism"].isna().any() or df["submechanism"].isna().any() or df["risk_regime"].isna().any():
    bad = df[df["submechanism"].isna()]["family"].unique().tolist()
    raise RuntimeError(f"Unknown family in input: {bad}")

df_feat, feature_cols = build_feature_table(df)

print("\n============================================================")
print("Constraint-Audit-2C Input Summary")
print("============================================================\n")

print("Input file:", INPUT_FILE)
print("Rows:", len(df_feat))
print("Feature count:", len(feature_cols))

print("\nFamily summary:")
print(
    f"{'family':<32}"
    f"{'mech':<24}"
    f"{'submechanism':<28}"
    f"{'risk':<24}"
    f"{'N':<6}"
    f"{'GenE':<10}"
)

for fam in FAMILY_ORDER:
    subdf = df_feat[df_feat["family"] == fam]
    if len(subdf) == 0:
        continue

    print(
        f"{fam:<32}"
        f"{COARSE_MECHANISM_MAP[fam]:<24}"
        f"{SUBMECHANISM_MAP[fam]:<28}"
        f"{RISK_REGIME_MAP[fam]:<24}"
        f"{len(subdf):<6}"
        f"{subdf['y_conflict'].mean():<10.4f}"
    )

# ---------------- GroupKFold ----------------

print("\n\n============================================================")
print("GroupKFold by graph idx")
print("============================================================\n")

group_summary, group_aux, group_pred = evaluate_group_cv(df_feat, feature_cols)

print("Binary prediction summary:")
print(
    f"{'model':<32}"
    f"{'AUC':<10}"
    f"{'Acc':<10}"
    f"{'F1':<10}"
    f"{'Brier':<10}"
    f"{'PredE':<10}"
)

for _, r in group_summary.iterrows():
    print(
        f"{r['model']:<32}"
        f"{r['auc']:<10.4f}"
        f"{r['accuracy']:<10.4f}"
        f"{r['f1']:<10.4f}"
        f"{r['brier']:<10.4f}"
        f"{r['pred_pos_rate']:<10.4f}"
    )

print("\nAuxiliary head summary:")
print(f"  submechanism_accuracy = {group_aux['submechanism_accuracy']:.4f}")
print(f"  submechanism_macro_f1 = {group_aux['submechanism_macro_f1']:.4f}")
print(f"  risk_regime_accuracy  = {group_aux['risk_regime_accuracy']:.4f}")
print(f"  risk_regime_macro_f1  = {group_aux['risk_regime_macro_f1']:.4f}")

# ---------------- Leave-one-family-out ----------------

print("\n\n============================================================")
print("Leave-One-Family-Out")
print("============================================================\n")

lofo_metrics, lofo_sub, lofo_risk, lofo_pred = evaluate_leave_family_out(df_feat, feature_cols)

lofo_summary = summarize_binary_metrics(lofo_metrics)
lofo_by_family = summarize_binary_by_family(lofo_metrics)
lofo_by_sub = summarize_binary_by_submechanism(lofo_metrics)

print("LOFO binary summary:")
print(
    f"{'model':<32}"
    f"{'rule':<24}"
    f"{'AUC':<10}"
    f"{'Acc':<10}"
    f"{'F1':<10}"
    f"{'Brier':<10}"
    f"{'PredE':<10}"
)

for _, r in lofo_summary.iterrows():
    print(
        f"{r['model']:<32}"
        f"{r['decision_rule']:<24}"
        f"{r['mean_auc']:<10.4f}"
        f"{r['mean_accuracy']:<10.4f}"
        f"{r['mean_f1']:<10.4f}"
        f"{r['mean_brier']:<10.4f}"
        f"{r['mean_pred_pos_rate']:<10.4f}"
    )

print("\nLOFO submechanism diagnostics:")
print(
    f"{'heldout_family':<32}"
    f"{'submechanism':<28}"
    f"{'train_has':<10}"
    f"{'acc':<8}"
    f"{'f1':<8}"
    f"{'top_prob_class':<30}"
    f"{'top_prob':<10}"
)

for _, r in lofo_sub.iterrows():
    prob_cols = [c for c in lofo_sub.columns if c.startswith("mean_p_")]
    vals = {c.replace("mean_p_", ""): r[c] for c in prob_cols}
    top_cls = max(vals, key=vals.get)
    top_prob = vals[top_cls]

    print(
        f"{r['heldout_family']:<32}"
        f"{r['heldout_submechanism']:<28}"
        f"{str(r['train_has_heldout_submechanism']):<10}"
        f"{r['submechanism_accuracy']:<8.4f}"
        f"{r['submechanism_macro_f1']:<8.4f}"
        f"{top_cls:<30}"
        f"{top_prob:<10.4f}"
    )

print("\nLOFO risk-regime diagnostics:")
print(
    f"{'heldout_family':<32}"
    f"{'risk_regime':<24}"
    f"{'train_has':<10}"
    f"{'acc':<8}"
    f"{'f1':<8}"
    f"{'top_prob_class':<30}"
    f"{'top_prob':<10}"
)

for _, r in lofo_risk.iterrows():
    prob_cols = [c for c in lofo_risk.columns if c.startswith("mean_p_")]
    vals = {c.replace("mean_p_", ""): r[c] for c in prob_cols}
    top_cls = max(vals, key=vals.get)
    top_prob = vals[top_cls]

    print(
        f"{r['heldout_family']:<32}"
        f"{r['heldout_risk_regime']:<24}"
        f"{str(r['train_has_heldout_risk_regime']):<10}"
        f"{r['risk_regime_accuracy']:<8.4f}"
        f"{r['risk_regime_macro_f1']:<8.4f}"
        f"{top_cls:<30}"
        f"{top_prob:<10.4f}"
    )

print("\nLOFO binary by submechanism, calibrated threshold:")
view = lofo_by_sub[lofo_by_sub["decision_rule"] == "calibrated_threshold"]

print(
    f"{'submechanism':<28}"
    f"{'model':<32}"
    f"{'pos':<8}"
    f"{'predE':<8}"
    f"{'acc':<8}"
    f"{'f1':<8}"
    f"{'brier':<10}"
)

for _, r in view.iterrows():
    print(
        f"{r['heldout_submechanism']:<28}"
        f"{r['model']:<32}"
        f"{r['mean_positive_rate']:<8.4f}"
        f"{r['mean_pred_pos_rate']:<8.4f}"
        f"{r['mean_accuracy']:<8.4f}"
        f"{r['mean_f1']:<8.4f}"
        f"{r['mean_brier']:<10.4f}"
    )

# ---------------- Full-data heads ----------------

print("\n\n============================================================")
print("Full-data Submechanism / Risk Head Activations")
print("============================================================\n")

head_pred, family_activation = full_data_head_activations(df_feat, feature_cols)

print("Family activations: submechanism head")
print(
    f"{'family':<32}"
    f"{'true_sub':<28}"
    f"{'GenE':<10}"
    f"{'top_sub_class':<30}"
    f"{'top_sub_prob':<10}"
)

for _, r in family_activation.iterrows():
    prob_cols = [c for c in family_activation.columns if c.startswith("mean_p_sub_")]
    vals = {c.replace("mean_p_sub_", ""): r[c] for c in prob_cols}
    top_cls = max(vals, key=vals.get)
    top_prob = vals[top_cls]

    print(
        f"{r['family']:<32}"
        f"{r['submechanism']:<28}"
        f"{r['genE_rate']:<10.4f}"
        f"{top_cls:<30}"
        f"{top_prob:<10.4f}"
    )

print("\nFamily activations: risk-regime head")
print(
    f"{'family':<32}"
    f"{'true_risk':<24}"
    f"{'GenE':<10}"
    f"{'top_risk_class':<30}"
    f"{'top_risk_prob':<10}"
)

for _, r in family_activation.iterrows():
    prob_cols = [c for c in family_activation.columns if c.startswith("mean_p_risk_")]
    vals = {c.replace("mean_p_risk_", ""): r[c] for c in prob_cols}
    top_cls = max(vals, key=vals.get)
    top_prob = vals[top_cls]

    print(
        f"{r['family']:<32}"
        f"{r['risk_regime']:<24}"
        f"{r['genE_rate']:<10.4f}"
        f"{top_cls:<30}"
        f"{top_prob:<10.4f}"
    )

coef_df = full_data_binary_coefficients(df_feat, feature_cols)

# ============================================================
# SAVE OUTPUTS
# ============================================================

feature_path = SAVE_DIR / "constraint_audit2c_feature_table.csv"

group_summary_path = SAVE_DIR / "constraint_audit2c_groupcv_binary_summary.csv"
group_pred_path = SAVE_DIR / "constraint_audit2c_groupcv_predictions.csv"

lofo_metrics_path = SAVE_DIR / "constraint_audit2c_lofo_binary_metrics.csv"
lofo_summary_path = SAVE_DIR / "constraint_audit2c_lofo_binary_summary.csv"
lofo_by_family_path = SAVE_DIR / "constraint_audit2c_lofo_binary_by_family.csv"
lofo_by_sub_path = SAVE_DIR / "constraint_audit2c_lofo_binary_by_submechanism.csv"
lofo_sub_path = SAVE_DIR / "constraint_audit2c_lofo_submechanism_diagnostics.csv"
lofo_risk_path = SAVE_DIR / "constraint_audit2c_lofo_risk_regime_diagnostics.csv"
lofo_pred_path = SAVE_DIR / "constraint_audit2c_lofo_predictions.csv"

head_pred_path = SAVE_DIR / "constraint_audit2c_head_predictions.csv"
activation_path = SAVE_DIR / "constraint_audit2c_family_activations.csv"
coef_path = SAVE_DIR / "constraint_audit2c_binary_feature_coefficients.csv"

df_feat.to_csv(feature_path, index=False)

group_summary.to_csv(group_summary_path, index=False)
group_pred.to_csv(group_pred_path, index=False)

lofo_metrics.to_csv(lofo_metrics_path, index=False)
lofo_summary.to_csv(lofo_summary_path, index=False)
lofo_by_family.to_csv(lofo_by_family_path, index=False)
lofo_by_sub.to_csv(lofo_by_sub_path, index=False)
lofo_sub.to_csv(lofo_sub_path, index=False)
lofo_risk.to_csv(lofo_risk_path, index=False)
lofo_pred.to_csv(lofo_pred_path, index=False)

head_pred.to_csv(head_pred_path, index=False)
family_activation.to_csv(activation_path, index=False)
coef_df.to_csv(coef_path, index=False)

print("\nSaved outputs:")
print(" ", feature_path)
print(" ", group_summary_path)
print(" ", group_pred_path)
print(" ", lofo_metrics_path)
print(" ", lofo_summary_path)
print(" ", lofo_by_family_path)
print(" ", lofo_by_sub_path)
print(" ", lofo_sub_path)
print(" ", lofo_risk_path)
print(" ", lofo_pred_path)
print(" ", head_pred_path)
print(" ", activation_path)
print(" ", coef_path)

# ============================================================
# INTERPRETATION GUIDE
# ============================================================

print("\n\n============================================================")
print("Constraint-Audit-2C Interpretation Guide")
print("============================================================\n")

print("Strong positive result if:")
print("  1. GroupKFold submechanism accuracy / macro-F1 is high.")
print("  2. stable_core separates from stable_surface_shift.")
print("  3. competition_low_commit separates from competition_high_commit.")
print("  4. closure_canonical_rewrite separates from closure_rule_override.")
print("  5. risk_regime head explains GenE differences better than coarse mechanism head.")
print()
print("Expected caveat:")
print("  competition_ambiguous has only one family.")
print("  In leave-one-family-out, when competition_ambiguous is held out,")
print("  the training set has no direct competition_ambiguous template.")
print("  Therefore failure on that family does not falsify submechanism decomposition.")
print()
print("Key interpretation:")
print("  If submechanism_aware_binary does not strongly improve GenE,")
print("  that is not fatal if baseline_binary is already near ceiling.")
print("  The main contribution is whether C_20:25 decomposes the mixed mechanisms")
print("  that 2B exposed.")
print()
print("If 2C succeeds:")
print("  C_20:25 should be rewritten as:")
print("    C_20:25 = (S_core, S_shift, K_low, K_mid, K_high, Q_canonical, Q_override, R_answer)")
print()
print("If 2C fails:")
print("  The current R/H/pC feature set is insufficient for submechanisms.")
print("  Next step should add closure-chain scoring and support-mass features.")
print()
print("Done.")