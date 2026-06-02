# ============================================================
# Constraint-Audit-1e
# Early-window Calibration & Critical-band Validation
#
# Purpose:
#   Reuse Constraint-Audit-1d saved binary dataset.
#   Do NOT rerun the model.
#
# Input:
#   ./constraint_audit1d_outputs/constraint_audit1d_binary_dataset.csv
#
# Main questions:
#   1. Does W20-W25 remain strong under leave-one-condition-out?
#   2. Can probability calibration reduce condition-wise miscalibration?
#   3. Can we build a stable critical band:
#        low risk  -> clean basin
#        high risk -> conflict basin
#        middle    -> critical / uncertain zone
#
# Outputs:
#   ./constraint_audit1e_outputs/
# ============================================================

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

INPUT_FILE = Path("./constraint_audit1d_outputs/constraint_audit1d_binary_dataset.csv")

SAVE_DIR = Path("./constraint_audit1e_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

CAL_SIZE = 0.30
N_BINS_ECE = 10

TARGET_DECISION_PRECISION = 0.95
MIN_COVERAGE_FOR_BAND = 0.30

PRIMARY_WINDOW = "W20_25_mid_basin"

WINDOWS = {
    "W20_22_boundary_only": [20, 21, 22],
    "W20_24_early_basin": [20, 21, 22, 23, 24],
    "W20_25_mid_basin": [20, 21, 22, 23, 24, 25],
    "W20_26_pre_final": [20, 21, 22, 23, 24, 25, 26],
    "W20_27_oracle_with_final": [20, 21, 22, 23, 24, 25, 26, 27],
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

CALIBRATION_METHODS = [
    "none",
    "platt",
    "isotonic",
    "temperature",
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
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return p


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x, dtype=np.float64)

    pos = x >= 0
    neg = ~pos

    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[neg])
    out[neg] = expx / (1.0 + expx)

    return sanitize_prob(out)


def logit(p):
    p = sanitize_prob(p)
    return np.log(p / (1.0 - p))


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


def ece_score(y, p, n_bins=10):
    y = np.asarray(y).astype(int)
    p = sanitize_prob(p)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]

        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)

        if mask.sum() == 0:
            continue

        mean_conf = float(p[mask].mean())
        empirical = float(y[mask].mean())
        weight = float(mask.mean())

        ece += weight * abs(mean_conf - empirical)

    return float(ece)


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
        "ece": ece_score(y, p, n_bins=N_BINS_ECE),
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


# ============================================================
# WINDOW FEATURE CONSTRUCTION
# ============================================================

def add_window_features(df, window_name, layers):
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

    prefix = window_name

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


def feature_names_for_window(window_name):
    prefix = window_name

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


# ============================================================
# BASE MODEL + CALIBRATORS
# ============================================================

def make_base_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            max_iter=1000,
        )),
    ])


class ConstantModel:
    def __init__(self, p):
        self.p = float(np.clip(p, 1e-6, 1.0 - 1e-6))

    def predict_proba(self, X):
        n = len(X)
        p1 = np.full(n, self.p, dtype=np.float64)
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)


class NoCalibrator:
    def fit(self, p_cal, y_cal):
        return self

    def transform(self, p):
        return sanitize_prob(p)


class PlattCalibrator:
    def __init__(self):
        self.model = None
        self.constant = None

    def fit(self, p_cal, y_cal):
        p_cal = sanitize_prob(p_cal)
        y_cal = np.asarray(y_cal).astype(int)

        if len(np.unique(y_cal)) < 2:
            self.constant = float(y_cal.mean())
            self.model = None
            return self

        z = logit(p_cal).reshape(-1, 1)

        self.model = LogisticRegression(
            solver="liblinear",
            random_state=RANDOM_SEED,
            max_iter=1000,
        )
        self.model.fit(z, y_cal)

        return self

    def transform(self, p):
        p = sanitize_prob(p)

        if self.model is None:
            return sanitize_prob(np.full(len(p), self.constant, dtype=np.float64))

        z = logit(p).reshape(-1, 1)
        return sanitize_prob(self.model.predict_proba(z)[:, 1])


class IsotonicCalibrator:
    def __init__(self):
        self.model = None
        self.constant = None

    def fit(self, p_cal, y_cal):
        p_cal = sanitize_prob(p_cal)
        y_cal = np.asarray(y_cal).astype(int)

        if len(np.unique(y_cal)) < 2:
            self.constant = float(y_cal.mean())
            self.model = None
            return self

        self.model = IsotonicRegression(out_of_bounds="clip")
        self.model.fit(p_cal, y_cal)

        return self

    def transform(self, p):
        p = sanitize_prob(p)

        if self.model is None:
            return sanitize_prob(np.full(len(p), self.constant, dtype=np.float64))

        return sanitize_prob(self.model.predict(p))


class TemperatureCalibrator:
    def __init__(self):
        self.T = 1.0

    def fit(self, p_cal, y_cal):
        p_cal = sanitize_prob(p_cal)
        y_cal = np.asarray(y_cal).astype(int)

        z = logit(p_cal)

        T_grid = np.concatenate([
            np.linspace(0.25, 1.0, 31),
            np.linspace(1.05, 5.0, 80),
            np.linspace(5.5, 15.0, 20),
        ])

        best_T = 1.0
        best_nll = float("inf")

        for T in T_grid:
            p = sigmoid(z / T)
            nll = safe_log_loss(y_cal, p)

            if np.isfinite(nll) and nll < best_nll:
                best_nll = nll
                best_T = float(T)

        self.T = best_T
        return self

    def transform(self, p):
        p = sanitize_prob(p)
        z = logit(p)
        return sanitize_prob(sigmoid(z / self.T))


def make_calibrator(method):
    if method == "none":
        return NoCalibrator()
    if method == "platt":
        return PlattCalibrator()
    if method == "isotonic":
        return IsotonicCalibrator()
    if method == "temperature":
        return TemperatureCalibrator()

    raise ValueError(f"Unknown calibration method: {method}")


# ============================================================
# THRESHOLD + CRITICAL BAND
# ============================================================

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
            best_score = score
            best_t = float(t)
            best_metrics = m

    return best_t, best_metrics


def fit_critical_band(y_cal, p_cal, target_precision=0.95):
    """
    Find alpha and beta:
        p <= alpha -> clean
        p >= beta  -> conflict
        else       -> critical / abstain
    """

    y = np.asarray(y_cal).astype(int)
    p = sanitize_prob(p_cal)

    candidates = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 49)))

    best_strict = None
    best_fallback = None

    for alpha in candidates:
        for beta in candidates:
            if alpha >= beta:
                continue

            pred = np.full(len(y), -1, dtype=int)
            pred[p <= alpha] = 0
            pred[p >= beta] = 1

            decided = pred != -1

            if decided.sum() == 0:
                continue

            coverage = float(decided.mean())
            decided_acc = float(np.mean(pred[decided] == y[decided]))

            clean_mask = pred == 0
            conflict_mask = pred == 1

            if clean_mask.sum() > 0:
                clean_precision = float(np.mean(y[clean_mask] == 0))
            else:
                clean_precision = np.nan

            if conflict_mask.sum() > 0:
                conflict_precision = float(np.mean(y[conflict_mask] == 1))
            else:
                conflict_precision = np.nan

            fallback_score = decided_acc * coverage

            fallback_row = {
                "alpha": float(alpha),
                "beta": float(beta),
                "coverage": coverage,
                "abstention_rate": float(1.0 - coverage),
                "decided_accuracy": decided_acc,
                "clean_precision": clean_precision,
                "conflict_precision": conflict_precision,
                "score": fallback_score,
                "strict": False,
            }

            if best_fallback is None or fallback_score > best_fallback["score"]:
                best_fallback = fallback_row

            has_both_sides = clean_mask.sum() > 0 and conflict_mask.sum() > 0

            if not has_both_sides:
                continue

            if coverage < MIN_COVERAGE_FOR_BAND:
                continue

            if clean_precision >= target_precision and conflict_precision >= target_precision:
                strict_row = dict(fallback_row)
                strict_row["score"] = coverage
                strict_row["strict"] = True

                if best_strict is None or strict_row["score"] > best_strict["score"]:
                    best_strict = strict_row

    if best_strict is not None:
        return best_strict

    if best_fallback is not None:
        return best_fallback

    return {
        "alpha": 0.33,
        "beta": 0.67,
        "coverage": 0.0,
        "abstention_rate": 1.0,
        "decided_accuracy": np.nan,
        "clean_precision": np.nan,
        "conflict_precision": np.nan,
        "score": 0.0,
        "strict": False,
    }


def apply_critical_band(y, p, alpha, beta):
    y = np.asarray(y).astype(int)
    p = sanitize_prob(p)

    pred = np.full(len(y), -1, dtype=int)
    pred[p <= alpha] = 0
    pred[p >= beta] = 1

    decided = pred != -1

    if decided.sum() > 0:
        decided_accuracy = float(np.mean(pred[decided] == y[decided]))
        decided_f1 = float(f1_score(y[decided], pred[decided], zero_division=0))
        decided_precision = float(precision_score(y[decided], pred[decided], zero_division=0))
        decided_recall = float(recall_score(y[decided], pred[decided], zero_division=0))
    else:
        decided_accuracy = np.nan
        decided_f1 = np.nan
        decided_precision = np.nan
        decided_recall = np.nan

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "coverage": float(decided.mean()),
        "abstention_rate": float(1.0 - decided.mean()),
        "decided_accuracy": decided_accuracy,
        "decided_f1": decided_f1,
        "decided_precision": decided_precision,
        "decided_recall": decided_recall,
        "pred_clean_rate": float(np.mean(pred == 0)),
        "pred_conflict_rate": float(np.mean(pred == 1)),
        "critical_rate": float(np.mean(pred == -1)),
    }


# ============================================================
# LOCO CALIBRATION
# ============================================================

def fit_base_or_constant(X_core, y_core):
    y_core = np.asarray(y_core).astype(int)

    if len(np.unique(y_core)) < 2:
        return ConstantModel(float(y_core.mean()))

    model = make_base_model()
    model.fit(X_core, y_core)
    return model


def run_loco_for_window(df, window_name, layers):
    dfw = add_window_features(df, window_name, layers)
    features = feature_names_for_window(window_name)

    X_all = dfw[features].values.astype(np.float32)
    y_all = dfw["y_conflict"].values.astype(int)
    cond_all = dfw["condition"].values

    loco_rows = []
    band_rows = []
    reliability_rows = []

    unique_conditions = [c for c in CONDITION_ORDER if c in set(cond_all)]

    for heldout in unique_conditions:
        test_mask = cond_all == heldout
        trainval_mask = ~test_mask

        X_trainval = X_all[trainval_mask]
        y_trainval = y_all[trainval_mask]

        X_test = X_all[test_mask]
        y_test = y_all[test_mask]

        if len(X_test) == 0:
            continue

        # Internal core/cal split on non-heldout data.
        idx = np.arange(len(y_trainval))

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

        X_core = X_trainval[core_idx]
        y_core = y_trainval[core_idx]

        X_cal = X_trainval[cal_idx]
        y_cal = y_trainval[cal_idx]

        base_model = fit_base_or_constant(X_core, y_core)

        p_cal_raw = sanitize_prob(base_model.predict_proba(X_cal)[:, 1])
        p_test_raw = sanitize_prob(base_model.predict_proba(X_test)[:, 1])

        for method in CALIBRATION_METHODS:
            calibrator = make_calibrator(method)
            calibrator.fit(p_cal_raw, y_cal)

            p_cal = sanitize_prob(calibrator.transform(p_cal_raw))
            p_test = sanitize_prob(calibrator.transform(p_test_raw))

            # Rule 1: fixed threshold 0.5.
            fixed_test = calc_binary_metrics(y_test, p_test, threshold=0.5)
            fixed_cal = calc_binary_metrics(y_cal, p_cal, threshold=0.5)

            # Rule 2: threshold tuned on calibration set.
            best_t, best_cal = best_threshold_on_calibration(y_cal, p_cal, objective="f1")
            tuned_test = calc_binary_metrics(y_test, p_test, threshold=best_t)

            base_info = {
                "window": window_name,
                "layers": ",".join(map(str, layers)),
                "heldout_condition": heldout,
                "method": method,
                "n_test": int(len(y_test)),
                "test_positive_rate": float(y_test.mean()),
                "cal_positive_rate": float(y_cal.mean()),

                "cal_fixed_acc": fixed_cal["accuracy"],
                "cal_fixed_f1": fixed_cal["f1"],
                "cal_fixed_brier": fixed_cal["brier"],
                "cal_fixed_ece": fixed_cal["ece"],

                "cal_best_threshold": float(best_t),
                "cal_best_acc": best_cal["accuracy"],
                "cal_best_f1": best_cal["f1"],
                "cal_best_precision": best_cal["precision"],
                "cal_best_recall": best_cal["recall"],
            }

            fixed_row = dict(base_info)
            fixed_row["decision_rule"] = "fixed_0.5"
            for k, v in fixed_test.items():
                fixed_row[f"test_{k}"] = v
            loco_rows.append(fixed_row)

            tuned_row = dict(base_info)
            tuned_row["decision_rule"] = "calibrated_threshold"
            for k, v in tuned_test.items():
                tuned_row[f"test_{k}"] = v
            loco_rows.append(tuned_row)

            # Critical band.
            band = fit_critical_band(
                y_cal,
                p_cal,
                target_precision=TARGET_DECISION_PRECISION,
            )

            band_test = apply_critical_band(
                y_test,
                p_test,
                alpha=band["alpha"],
                beta=band["beta"],
            )

            band_row = {
                "window": window_name,
                "layers": ",".join(map(str, layers)),
                "heldout_condition": heldout,
                "method": method,
                "n_test": int(len(y_test)),
                "test_positive_rate": float(y_test.mean()),
                "cal_positive_rate": float(y_cal.mean()),

                "band_alpha": band["alpha"],
                "band_beta": band["beta"],
                "band_strict": band["strict"],

                "cal_band_coverage": band["coverage"],
                "cal_band_decided_accuracy": band["decided_accuracy"],
                "cal_band_clean_precision": band["clean_precision"],
                "cal_band_conflict_precision": band["conflict_precision"],
            }

            for k, v in band_test.items():
                band_row[f"test_band_{k}"] = v

            band_rows.append(band_row)

            # Reliability bins on held-out condition.
            bin_edges = np.linspace(0.0, 1.0, N_BINS_ECE + 1)

            for bi in range(N_BINS_ECE):
                lo = bin_edges[bi]
                hi = bin_edges[bi + 1]

                if bi == N_BINS_ECE - 1:
                    mask = (p_test >= lo) & (p_test <= hi)
                else:
                    mask = (p_test >= lo) & (p_test < hi)

                if mask.sum() == 0:
                    continue

                reliability_rows.append({
                    "window": window_name,
                    "heldout_condition": heldout,
                    "method": method,
                    "bin": int(bi),
                    "bin_lo": float(lo),
                    "bin_hi": float(hi),
                    "n": int(mask.sum()),
                    "mean_prob": float(p_test[mask].mean()),
                    "empirical_positive_rate": float(y_test[mask].mean()),
                })

    return (
        pd.DataFrame(loco_rows),
        pd.DataFrame(band_rows),
        pd.DataFrame(reliability_rows),
    )


# ============================================================
# SUMMARIES
# ============================================================

def summarize_loco(loco_df):
    rows = []

    group_cols = ["window", "method", "decision_rule"]

    for keys, sub in loco_df.groupby(group_cols):
        window, method, rule = keys

        rows.append({
            "window": window,
            "method": method,
            "decision_rule": rule,

            "mean_auc": float(np.nanmean(sub["test_auc"])),
            "mean_accuracy": float(np.nanmean(sub["test_accuracy"])),
            "mean_f1": float(np.nanmean(sub["test_f1"])),
            "mean_precision": float(np.nanmean(sub["test_precision"])),
            "mean_recall": float(np.nanmean(sub["test_recall"])),

            "mean_brier": float(np.nanmean(sub["test_brier"])),
            "mean_nll": float(np.nanmean(sub["test_nll"])),
            "mean_ece": float(np.nanmean(sub["test_ece"])),

            "mean_pred_pos_rate": float(np.nanmean(sub["test_pred_pos_rate"])),
            "std_pred_pos_rate": float(np.nanstd(sub["test_pred_pos_rate"])),
        })

    return pd.DataFrame(rows)


def summarize_conditions(loco_df):
    rows = []

    group_cols = ["window", "method", "decision_rule", "heldout_condition"]

    for keys, sub in loco_df.groupby(group_cols):
        window, method, rule, cond = keys

        rows.append({
            "window": window,
            "method": method,
            "decision_rule": rule,
            "heldout_condition": cond,

            "positive_rate": float(sub["test_positive_rate"].iloc[0]),
            "pred_pos_rate": float(np.nanmean(sub["test_pred_pos_rate"])),
            "mean_prob": float(np.nanmean(sub["test_mean_prob"])),

            "accuracy": float(np.nanmean(sub["test_accuracy"])),
            "f1": float(np.nanmean(sub["test_f1"])),
            "brier": float(np.nanmean(sub["test_brier"])),
            "ece": float(np.nanmean(sub["test_ece"])),
        })

    return pd.DataFrame(rows)


def summarize_bands(band_df):
    rows = []

    for keys, sub in band_df.groupby(["window", "method"]):
        window, method = keys

        rows.append({
            "window": window,
            "method": method,

            "mean_coverage": float(np.nanmean(sub["test_band_coverage"])),
            "mean_abstention_rate": float(np.nanmean(sub["test_band_abstention_rate"])),
            "mean_decided_accuracy": float(np.nanmean(sub["test_band_decided_accuracy"])),
            "mean_decided_f1": float(np.nanmean(sub["test_band_decided_f1"])),

            "mean_pred_clean_rate": float(np.nanmean(sub["test_band_pred_clean_rate"])),
            "mean_pred_conflict_rate": float(np.nanmean(sub["test_band_pred_conflict_rate"])),
            "mean_critical_rate": float(np.nanmean(sub["test_band_critical_rate"])),

            "strict_band_rate": float(np.mean(sub["band_strict"].astype(bool))),
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find input file: {INPUT_FILE}\n"
        f"Please run Constraint-Audit-1d first, or change INPUT_FILE."
    )

df = pd.read_csv(INPUT_FILE)

# Normalize bool columns if present.
for col in ["is_clean", "is_conflict", "is_other"]:
    if col in df.columns:
        df[col] = parse_bool_series(df[col])

# Build target if missing.
if "y_conflict" not in df.columns:
    if "is_conflict" not in df.columns:
        raise RuntimeError("Need either y_conflict or is_conflict column.")
    df["y_conflict"] = df["is_conflict"].astype(int)

df["y_conflict"] = df["y_conflict"].astype(int)

# Keep only binary clean/conflict rows if columns are available.
if "is_clean" in df.columns and "is_conflict" in df.columns:
    df = df[(df["is_clean"] == True) | (df["is_conflict"] == True)].copy()

print("\n============================================================")
print("Constraint-Audit-1e Input Summary")
print("============================================================\n")

print("Input file:", INPUT_FILE)
print("Rows:", len(df))
print("Conditions:", sorted(df["condition"].unique().tolist()))

print("\nCondition positive rates:")
for cond in CONDITION_ORDER:
    sub = df[df["condition"] == cond]
    if len(sub) == 0:
        continue
    print(
        f"{cond:<28} "
        f"N={len(sub):<5} "
        f"Gen_E={sub['y_conflict'].mean():.4f}"
    )

all_loco = []
all_bands = []
all_reliability = []

for window_name, layers in WINDOWS.items():
    print("\n\n############################################################")
    print(f"Running LOCO calibration: {window_name} | layers={layers}")
    print("############################################################\n")

    loco_df, band_df, rel_df = run_loco_for_window(df, window_name, layers)

    all_loco.append(loco_df)
    all_bands.append(band_df)
    all_reliability.append(rel_df)

loco_all = pd.concat(all_loco, axis=0, ignore_index=True)
band_all = pd.concat(all_bands, axis=0, ignore_index=True)
reliability_all = pd.concat(all_reliability, axis=0, ignore_index=True)

summary_df = summarize_loco(loco_all)
condition_df = summarize_conditions(loco_all)
band_summary_df = summarize_bands(band_all)

# ============================================================
# PRINT PRIMARY RESULTS
# ============================================================

print("\n\n============================================================")
print("Constraint-Audit-1e Primary Window Summary")
print("============================================================\n")

primary_summary = summary_df[
    (summary_df["window"] == PRIMARY_WINDOW)
    & (summary_df["decision_rule"] == "calibrated_threshold")
].copy()

primary_summary = primary_summary.sort_values(
    by=["mean_brier", "mean_ece"],
    ascending=[True, True],
)

print(f"Primary window: {PRIMARY_WINDOW}\n")

print(
    f"{'method':<14}"
    f"{'AUC':<10}"
    f"{'Acc':<10}"
    f"{'F1':<10}"
    f"{'Brier':<10}"
    f"{'NLL':<10}"
    f"{'ECE':<10}"
    f"{'PredE':<10}"
)

for _, r in primary_summary.iterrows():
    print(
        f"{r['method']:<14}"
        f"{r['mean_auc']:<10.4f}"
        f"{r['mean_accuracy']:<10.4f}"
        f"{r['mean_f1']:<10.4f}"
        f"{r['mean_brier']:<10.4f}"
        f"{r['mean_nll']:<10.4f}"
        f"{r['mean_ece']:<10.4f}"
        f"{r['mean_pred_pos_rate']:<10.4f}"
    )

print("\n\nCondition-wise calibration for primary window:\n")

primary_condition = condition_df[
    (condition_df["window"] == PRIMARY_WINDOW)
    & (condition_df["decision_rule"] == "calibrated_threshold")
].copy()

primary_condition = primary_condition.sort_values(
    by=["method", "heldout_condition"]
)

for method in CALIBRATION_METHODS:
    sub = primary_condition[primary_condition["method"] == method]

    if len(sub) == 0:
        continue

    print(f"\nMethod = {method}")
    print(
        f"{'condition':<28}"
        f"{'pos':<8}"
        f"{'predE':<8}"
        f"{'pE':<8}"
        f"{'acc':<8}"
        f"{'f1':<8}"
        f"{'brier':<10}"
        f"{'ece':<10}"
    )

    for _, r in sub.iterrows():
        print(
            f"{r['heldout_condition']:<28}"
            f"{r['positive_rate']:<8.4f}"
            f"{r['pred_pos_rate']:<8.4f}"
            f"{r['mean_prob']:<8.4f}"
            f"{r['accuracy']:<8.4f}"
            f"{r['f1']:<8.4f}"
            f"{r['brier']:<10.4f}"
            f"{r['ece']:<10.4f}"
        )

print("\n\nCritical-band summary for primary window:\n")

primary_band = band_summary_df[
    band_summary_df["window"] == PRIMARY_WINDOW
].copy()

primary_band = primary_band.sort_values(
    by=["mean_decided_accuracy", "mean_coverage"],
    ascending=[False, False],
)

print(
    f"{'method':<14}"
    f"{'coverage':<10}"
    f"{'abstain':<10}"
    f"{'decAcc':<10}"
    f"{'decF1':<10}"
    f"{'clean':<10}"
    f"{'conflict':<10}"
    f"{'critical':<10}"
)

for _, r in primary_band.iterrows():
    print(
        f"{r['method']:<14}"
        f"{r['mean_coverage']:<10.4f}"
        f"{r['mean_abstention_rate']:<10.4f}"
        f"{r['mean_decided_accuracy']:<10.4f}"
        f"{r['mean_decided_f1']:<10.4f}"
        f"{r['mean_pred_clean_rate']:<10.4f}"
        f"{r['mean_pred_conflict_rate']:<10.4f}"
        f"{r['mean_critical_rate']:<10.4f}"
    )

# ============================================================
# SAVE OUTPUTS
# ============================================================

loco_path = SAVE_DIR / "constraint_audit1e_loco_calibration.csv"
summary_path = SAVE_DIR / "constraint_audit1e_window_method_summary.csv"
condition_path = SAVE_DIR / "constraint_audit1e_condition_calibration.csv"
band_path = SAVE_DIR / "constraint_audit1e_critical_band.csv"
band_summary_path = SAVE_DIR / "constraint_audit1e_critical_band_summary.csv"
reliability_path = SAVE_DIR / "constraint_audit1e_reliability_bins.csv"

loco_all.to_csv(loco_path, index=False)
summary_df.to_csv(summary_path, index=False)
condition_df.to_csv(condition_path, index=False)
band_all.to_csv(band_path, index=False)
band_summary_df.to_csv(band_summary_path, index=False)
reliability_all.to_csv(reliability_path, index=False)

print("\nSaved outputs:")
print(" ", loco_path)
print(" ", summary_path)
print(" ", condition_path)
print(" ", band_path)
print(" ", band_summary_path)
print(" ", reliability_path)

# ============================================================
# INTERPRETATION GUIDE
# ============================================================

print("\n\n============================================================")
print("Constraint-Audit-1e Interpretation Guide")
print("============================================================\n")

print("Primary success pattern:")
print("  1. W20_25 remains high-AUC / high-F1 under LOCO.")
print("  2. Calibration reduces Brier / NLL / ECE versus method='none'.")
print("  3. clean and weak_distractor predE move closer to 0.")
print("  4. closure_negation_conflict predE moves closer to 1.")
print("  5. ambiguous_branch should remain partly critical rather than being over-forced.")
print()
print("If AUC stays high but Brier/ECE remain poor:")
print("  C_20:25 is a strong ranking diagnostic, but not yet calibrated probability.")
print()
print("If critical-band decided accuracy is high with moderate coverage:")
print("  The stable basin / critical band / collapse basin decomposition is supported.")
print()
print("If some heldout conditions remain miscalibrated after all calibration methods:")
print("  This suggests mechanism heterogeneity.")
print("  Next step would be mechanism-aware C_l decomposition.")
print()
print("Done.")