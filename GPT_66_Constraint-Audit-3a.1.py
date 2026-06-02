# ============================================================
# Constraint-Audit-3A.1
# Feature-family and Early-only Ablation
#
# Input:
#   ./constraint_audit3a_outputs/constraint_audit3a_augmented_feature_table.csv
#
# Goal:
#   3A showed augmented_C3A improves Gen_E prediction strongly,
#   but LOFO submechanism / risk-regime generalization did not improve.
#
#   3A.1 asks:
#     Which feature family caused the improvement?
#     Was it mainly near-output L26-L27 rank/answer-basin readout?
#     Or do early-only L20-L25 augmented features add real signal?
#
# Feature sets:
#   1. baseline_RHpC_20_25
#   2. ans_margin_only_20_25
#   3. rank_only_20_25
#   4. support_only_20_25
#   5. marker_only_20_25
#   6. topology_only_20_25
#   7. all_aug_20_25_no_commit
#   8. commit_only_26_27
#   9. all_aug_20_27_with_commit
#
# Evaluations:
#   - GroupKFold binary Gen_E
#   - GroupKFold submechanism
#   - GroupKFold risk_regime
#   - LOFO binary Gen_E
#   - LOFO submechanism
#   - LOFO risk_regime
#
# Outputs:
#   ./constraint_audit3a1_outputs/
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

INPUT_FILE = Path("./constraint_audit3a_outputs/constraint_audit3a_augmented_feature_table.csv")

SAVE_DIR = Path("./constraint_audit3a1_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

N_GROUP_SPLITS = 5

EARLY_LAYERS = [20, 21, 22, 23, 24, 25]
COMMIT_LAYERS = [26, 27]
ALL_LAYERS = [20, 21, 22, 23, 24, 25, 26, 27]

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
        out["tn"] = 0
        out["fp"] = 0
        out["fn"] = 0
        out["tp"] = 0

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
            max_iter=4000,
        )),
    ])


def make_multiclass_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            max_iter=6000,
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
        f"Cannot find {INPUT_FILE}. Run Constraint-Audit-3A first."
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

df["mechanism"] = df["family"].map(COARSE_MECHANISM_MAP)
df["submechanism"] = df["family"].map(SUBMECHANISM_MAP)
df["risk_regime"] = df["family"].map(RISK_REGIME_MAP)

if df["submechanism"].isna().any():
    bad = df[df["submechanism"].isna()]["family"].unique().tolist()
    raise RuntimeError(f"Unknown family values: {bad}")

df = df.reset_index(drop=True)

print("\n============================================================")
print("Constraint-Audit-3A.1 Input Summary")
print("============================================================\n")
print("Input:", INPUT_FILE)
print("Rows:", len(df))
print("Families:", df["family"].nunique())

print("\nFamily summary:")
print(
    f"{'family':<32}"
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
        f"{SUBMECHANISM_MAP[fam]:<28}"
        f"{RISK_REGIME_MAP[fam]:<24}"
        f"{len(sub):<6}"
        f"{sub['y_conflict'].mean():<10.4f}"
    )

# ============================================================
# FEATURE SELECTION
# ============================================================

def is_numeric_col(c):
    return c in df.columns and pd.api.types.is_numeric_dtype(df[c])


def clean_cols(cols):
    seen = set()
    out = []

    for c in cols:
        if c in seen:
            continue
        seen.add(c)

        if is_numeric_col(c):
            out.append(c)

    return out


def has_layer_suffix(c, layers):
    for l in layers:
        if c.endswith(f"_L{l}") or f"_L{l}_" in c:
            return True
    return False


def is_early_col(c):
    # Include L20-L25 layer features and early window aggregates.
    if has_layer_suffix(c, EARLY_LAYERS):
        return True

    early_tokens = [
        "ENTRY20_22",
        "BASIN23_25",
        "ALL20_25",
        "20_25",
        "L20_21",
        "L21_22",
        "L22_23",
        "L23_24",
        "L24_25",
    ]

    return any(tok in c for tok in early_tokens)


def is_commit_col(c):
    if has_layer_suffix(c, COMMIT_LAYERS):
        return True

    commit_tokens = [
        "COMMIT26_27",
        "ALL20_27",
        "20_27",
        "L25_26",
        "L26_27",
    ]

    return any(tok in c for tok in commit_tokens)


def is_aug(c):
    return c.startswith("AUG_")


def match_any(c, patterns):
    return any(p in c for p in patterns)


# ---------------- baseline ----------------

baseline_cols = []

for l in EARLY_LAYERS:
    for name in ["R", "H", "pC"]:
        col = f"{name}_L{l}"
        if col in df.columns:
            baseline_cols.append(col)

baseline_cols += [
    c for c in df.columns
    if c.startswith("ENTRY20_22_BASE")
    or c.startswith("BASIN23_25_BASE")
    or c.startswith("ALL20_25_BASE")
]

baseline_cols = clean_cols(baseline_cols)

# ---------------- augmented families ----------------

aug_cols_all = clean_cols([c for c in df.columns if is_aug(c)])

ans_margin_cols_early = clean_cols([
    c for c in aug_cols_all
    if "AUG_ans_margin" in c and is_early_col(c) and not is_commit_col(c)
])

rank_cols_early = clean_cols([
    c for c in aug_cols_all
    if match_any(c, [
        "AUG_clean_rank",
        "AUG_conflict_rank",
        "AUG_rank_gap_E_minus_C",
    ])
    and is_early_col(c)
    and not is_commit_col(c)
])

support_cols_early = clean_cols([
    c for c in aug_cols_all
    if match_any(c, [
        "AUG_clean_mass",
        "AUG_conflict_mass",
        "AUG_support_logratio",
        "AUG_clean_support_count",
        "AUG_conflict_support_count",
        "AUG_support_count_margin",
    ])
    and is_early_col(c)
    and not is_commit_col(c)
])

marker_cols_early = clean_cols([
    c for c in aug_cols_all
    if match_any(c, [
        "stable_marker",
        "competition_marker",
        "closure_marker",
        "override_marker",
        "authority_marker",
    ])
    and is_early_col(c)
    and not is_commit_col(c)
])

topology_cols_early = clean_cols([
    c for c in aug_cols_all
    if match_any(c, [
        "AUG_topk_entropy",
        "AUG_topk_spread",
        "AUG_topk_jaccard",
    ])
    and is_early_col(c)
    and not is_commit_col(c)
])

all_aug_early_no_commit = clean_cols([
    c for c in aug_cols_all
    if is_early_col(c) and not is_commit_col(c)
])

commit_only_cols = clean_cols([
    c for c in aug_cols_all
    if is_commit_col(c)
])

all_aug_with_commit = clean_cols(baseline_cols + aug_cols_all)

feature_sets = {
    "baseline_RHpC_20_25": baseline_cols,

    "ans_margin_only_20_25": ans_margin_cols_early,
    "rank_only_20_25": rank_cols_early,
    "support_only_20_25": support_cols_early,
    "marker_only_20_25": marker_cols_early,
    "topology_only_20_25": topology_cols_early,

    "all_aug_20_25_no_commit": clean_cols(baseline_cols + all_aug_early_no_commit),
    "commit_only_26_27": commit_only_cols,
    "all_aug_20_27_with_commit": all_aug_with_commit,
}

# Remove empty feature sets.
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

    gkf = GroupKFold(n_splits=N_GROUP_SPLITS)
    p = np.zeros(len(df_in), dtype=np.float64)

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        model = fit_binary_or_constant(X[train_idx], y[train_idx])
        p[test_idx] = sanitize_prob(model.predict_proba(X[test_idx])[:, 1])

    row = {
        "feature_set": feature_set,
        "eval": "GroupKFold_binary",
    }
    row.update(calc_binary_metrics(y, p, threshold=0.5))
    return row


def eval_multiclass_groupcv(df_in, cols, target_col, classes, feature_set):
    X = df_in[cols].values.astype(np.float32)
    y = df_in[target_col].values
    groups = df_in["idx"].values

    gkf = GroupKFold(n_splits=N_GROUP_SPLITS)
    pred = np.empty(len(df_in), dtype=object)

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        model = fit_multiclass_or_constant(X[train_idx], y[train_idx])
        pred[test_idx] = model.predict(X[test_idx])

    return {
        "feature_set": feature_set,
        "target": target_col,
        "eval": "GroupKFold_multiclass",
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

        m = calc_binary_metrics(y[test_idx], p_test, threshold=0.5)

        row = {
            "feature_set": feature_set,
            "heldout_family": heldout,
            "heldout_submechanism": SUBMECHANISM_MAP[heldout],
            "heldout_risk_regime": RISK_REGIME_MAP[heldout],
            "positive_rate": float(y[test_idx].mean()),
        }
        row.update(m)
        rows.append(row)

        temp = df_in.iloc[test_idx][[
            "idx",
            "family",
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
        }

        for j, cls in enumerate(classes):
            row[f"mean_p_{cls}"] = float(p[:, j].mean())

        rows.append(row)

        temp = df_in.iloc[test_idx][[
            "idx",
            "family",
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
# RUN EVALUATIONS
# ============================================================

binary_group_rows = []
multi_group_rows = []

binary_lofo_all = []
binary_lofo_pred_all = []

multi_lofo_all = []
multi_lofo_pred_all = []

for fs_name, cols in feature_sets.items():
    print(f"\nEvaluating: {fs_name} | n_features={len(cols)}")

    binary_group_rows.append(
        eval_binary_groupcv(df, cols, fs_name)
    )

    multi_group_rows.append(
        eval_multiclass_groupcv(
            df,
            cols,
            target_col="submechanism",
            classes=SUBMECHANISM_CLASSES,
            feature_set=fs_name,
        )
    )

    multi_group_rows.append(
        eval_multiclass_groupcv(
            df,
            cols,
            target_col="risk_regime",
            classes=RISK_REGIME_CLASSES,
            feature_set=fs_name,
        )
    )

    blofo, blofo_pred = eval_binary_lofo(df, cols, fs_name)
    binary_lofo_all.append(blofo)
    binary_lofo_pred_all.append(blofo_pred)

    slofo, slofo_pred = eval_multiclass_lofo(
        df,
        cols,
        target_col="submechanism",
        classes=SUBMECHANISM_CLASSES,
        feature_set=fs_name,
    )
    multi_lofo_all.append(slofo)
    multi_lofo_pred_all.append(slofo_pred)

    rlofo, rlofo_pred = eval_multiclass_lofo(
        df,
        cols,
        target_col="risk_regime",
        classes=RISK_REGIME_CLASSES,
        feature_set=fs_name,
    )
    multi_lofo_all.append(rlofo)
    multi_lofo_pred_all.append(rlofo_pred)

binary_group_df = pd.DataFrame(binary_group_rows)
multi_group_df = pd.DataFrame(multi_group_rows)

binary_lofo_df = pd.concat(binary_lofo_all, axis=0, ignore_index=True)
binary_lofo_pred_df = pd.concat(binary_lofo_pred_all, axis=0, ignore_index=True)

multi_lofo_df = pd.concat(multi_lofo_all, axis=0, ignore_index=True)
multi_lofo_pred_df = pd.concat(multi_lofo_pred_all, axis=0, ignore_index=True)

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

# family-level top prediction summary for multiclass
top_class_rows = []

for _, r in multi_lofo_df.iterrows():
    prob_cols = [c for c in multi_lofo_df.columns if c.startswith("mean_p_")]
    vals = {}

    for c in prob_cols:
        if pd.notna(r.get(c, np.nan)):
            vals[c.replace("mean_p_", "")] = r[c]

    if len(vals) == 0:
        continue

    top_cls = max(vals, key=vals.get)
    top_p = vals[top_cls]

    top_class_rows.append({
        "feature_set": r["feature_set"],
        "target": r["target"],
        "heldout_family": r["heldout_family"],
        "true_label": r["true_label"],
        "top_pred_class": top_cls,
        "top_pred_prob": float(top_p),
        "accuracy": r["accuracy"],
        "macro_f1": r["macro_f1"],
        "train_has_true_label": r["train_has_true_label"],
    })

top_class_df = pd.DataFrame(top_class_rows)

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n\n============================================================")
print("3A.1 GroupKFold Binary")
print("============================================================\n")

print(
    binary_group_df[[
        "feature_set",
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
print("3A.1 GroupKFold Multiclass")
print("============================================================\n")

print(
    multi_group_df
    .sort_values(by=["target", "macro_f1"], ascending=[True, False])
    .to_string(index=False)
)

print("\n\n============================================================")
print("3A.1 LOFO Binary Summary")
print("============================================================\n")

print(
    binary_lofo_summary
    .sort_values(by="mean_brier")
    .to_string(index=False)
)

print("\n\n============================================================")
print("3A.1 LOFO Multiclass Summary")
print("============================================================\n")

print(
    multi_lofo_summary
    .sort_values(by=["target", "mean_macro_f1"], ascending=[True, False])
    .to_string(index=False)
)

print("\n\n============================================================")
print("3A.1 Key LOFO Family Top Predictions")
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

view = top_class_df[top_class_df["heldout_family"].isin(interesting_families)].copy()

print(
    view.sort_values(
        by=["target", "heldout_family", "feature_set"]
    ).to_string(index=False)
)

# ============================================================
# SAVE OUTPUTS
# ============================================================

feature_sets_path = SAVE_DIR / "constraint_audit3a1_feature_sets.csv"
binary_group_path = SAVE_DIR / "constraint_audit3a1_group_binary.csv"
multi_group_path = SAVE_DIR / "constraint_audit3a1_group_multiclass.csv"

binary_lofo_path = SAVE_DIR / "constraint_audit3a1_lofo_binary.csv"
binary_lofo_summary_path = SAVE_DIR / "constraint_audit3a1_lofo_binary_summary.csv"
binary_lofo_pred_path = SAVE_DIR / "constraint_audit3a1_lofo_binary_predictions.csv"

multi_lofo_path = SAVE_DIR / "constraint_audit3a1_lofo_multiclass.csv"
multi_lofo_summary_path = SAVE_DIR / "constraint_audit3a1_lofo_multiclass_summary.csv"
multi_lofo_pred_path = SAVE_DIR / "constraint_audit3a1_lofo_multiclass_predictions.csv"
top_class_path = SAVE_DIR / "constraint_audit3a1_lofo_top_class_summary.csv"

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

print("\nSaved outputs:")
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

# ============================================================
# INTERPRETATION GUIDE
# ============================================================

print("\n\n============================================================")
print("Constraint-Audit-3A.1 Interpretation Guide")
print("============================================================\n")

print("Main questions:")
print("  1. Is 3A binary improvement mainly from commit_only_26_27?")
print("  2. Does all_aug_20_25_no_commit improve LOFO binary over baseline?")
print("  3. Do marker_only_20_25 or topology_only_20_25 improve LOFO multiclass?")
print("  4. Does support_only_20_25 help GenE without hurting mechanism generalization?")
print()
print("Readout:")
print("  If commit_only_26_27 is strongest for binary:")
print("    3A mostly added near-output answer-basin readout.")
print()
print("  If all_aug_20_25_no_commit improves binary but not multiclass:")
print("    early answer-basin readout improved, but mechanism abstraction still missing.")
print()
print("  If marker_only_20_25 improves closure/override LOFO:")
print("    marker mass is a genuine mechanism feature.")
print()
print("  If topology_only_20_25 improves risk_regime LOFO:")
print("    freedom/topology persistence is a genuine mechanism feature.")
print()
print("  If no early-only family improves multiclass:")
print("    next step should be explicit relation-triple / closure-chain scoring,")
print("    not more TopK readout features.")
print()
print("Done.")