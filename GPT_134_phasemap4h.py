import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import KFold, GroupKFold

# ============================================================
# CONFIG
# ============================================================

PRED_PATH = Path(r"C:\Windows\System32\phasemap4a_outputs\phasemap4a_rollout_predictions.csv")
BASE_PATH = Path(r"C:\Windows\System32\phasemap3b_outputs\phasemap3b_enriched_energy_error.csv")
SAVE_DIR = Path("./phasemap4h_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
R_LAYERS = list(range(20, 27))
TAIL_LAYERS = [23, 24, 25, 26]

# ============================================================
# LOAD
# ============================================================

if not PRED_PATH.exists():
    raise FileNotFoundError(f"Missing 4A predictions: {PRED_PATH.resolve()}")

if not BASE_PATH.exists():
    raise FileNotFoundError(f"Missing 3B enriched data: {BASE_PATH.resolve()}")

pred = pd.read_csv(PRED_PATH)
base = pd.read_csv(BASE_PATH)

if "model" not in base.columns:
    base["model"] = "unknown"

GROUP_COL = "group_id" if "group_id" in pred.columns else "condition"

# ============================================================
# SELECT BEST / MAIN 4A CANDIDATE
# ============================================================

# Use all rows, but candidate identity is preserved.
pred["candidate_id"] = (
    pred["candidate_cv"].astype(str)
    + "|"
    + pred["candidate_model"].astype(str)
    + "|"
    + pred["candidate_feature_set"].astype(str)
    + "|"
    + pred["audit_cv"].astype(str)
    + "|"
    + pred["lambda_model"].astype(str)
    + "|"
    + pred["lambda_feature_set"].astype(str)
)

# ============================================================
# DERIVED ERROR TARGETS
# ============================================================

for l in R_LAYERS:
    if f"true_R_{l}" in pred.columns:
        pred[f"base_err_R_{l}"] = pred[f"base_pred_R_{l}"] - pred[f"true_R_{l}"]
        pred[f"roll_err_R_{l}"] = pred[f"roll_R_{l}"] - pred[f"true_R_{l}"]

        pred[f"base_abs_err_R_{l}"] = pred[f"base_err_R_{l}"].abs()
        pred[f"roll_abs_err_R_{l}"] = pred[f"roll_err_R_{l}"].abs()

for a, b in zip(R_LAYERS[:-1], R_LAYERS[1:]):
    pred[f"base_v_{a}_{b}"] = pred[f"base_pred_R_{b}"] - pred[f"base_pred_R_{a}"]
    pred[f"roll_v_{a}_{b}"] = pred[f"roll_R_{b}"] - pred[f"roll_R_{a}"]
    pred[f"true_v_{a}_{b}"] = pred[f"true_R_{b}"] - pred[f"true_R_{a}"]

pred["target_roll_R26_error"] = pred["roll_err_R_26"]
pred["target_roll_R26_abs_error"] = pred["roll_abs_err_R_26"]
pred["target_roll_minus_base_R26_abs"] = pred["roll_abs_err_R_26"] - pred["base_abs_err_R_26"]

pred["target_roll_tail_mae"] = pred[[f"roll_abs_err_R_{l}" for l in TAIL_LAYERS]].mean(axis=1)
pred["target_base_tail_mae"] = pred[[f"base_abs_err_R_{l}" for l in TAIL_LAYERS]].mean(axis=1)
pred["target_roll_minus_base_tail_mae"] = pred["target_roll_tail_mae"] - pred["target_base_tail_mae"]

pred["roll_worse_R26"] = (pred["target_roll_minus_base_R26_abs"] > 0).astype(int)
pred["roll_worse_tail"] = (pred["target_roll_minus_base_tail_mae"] > 0).astype(int)

# ============================================================
# HIDDEN STATE CANDIDATES
# ============================================================

# Candidate Z1: initial observed state
INIT_FEATURES = [
    "true_R_20", "true_R_21", "true_R_22",
]

# Candidate Z2: base proposal velocity field
# If this explains 4A failure, hidden Z is the future proposal field / velocity schedule.
BASE_V_FEATURES = [
    "base_v_20_21", "base_v_21_22", "base_v_22_23",
    "base_v_23_24", "base_v_24_25", "base_v_25_26",
]

# Candidate Z3: rollout velocity field
ROLL_V_FEATURES = [
    "roll_v_20_21", "roll_v_21_22", "roll_v_22_23",
    "roll_v_23_24", "roll_v_24_25", "roll_v_25_26",
]

# Candidate Z4: divergence signature
DIVERGENCE_FEATURES = []
for l in TAIL_LAYERS:
    pred[f"div_R_{l}"] = pred[f"roll_R_{l}"] - pred[f"base_pred_R_{l}"]
    pred[f"abs_div_R_{l}"] = pred[f"div_R_{l}"].abs()
    DIVERGENCE_FEATURES += [f"div_R_{l}", f"abs_div_R_{l}"]

# Candidate Z5: shape of original base proposal path
pred["base_tail_span"] = pred[[f"base_pred_R_{l}" for l in TAIL_LAYERS]].max(axis=1) - pred[[f"base_pred_R_{l}" for l in TAIL_LAYERS]].min(axis=1)
pred["base_tail_mean"] = pred[[f"base_pred_R_{l}" for l in TAIL_LAYERS]].mean(axis=1)
pred["base_tail_slope"] = pred["base_pred_R_26"] - pred["base_pred_R_23"]

pred["roll_tail_span"] = pred[[f"roll_R_{l}" for l in TAIL_LAYERS]].max(axis=1) - pred[[f"roll_R_{l}" for l in TAIL_LAYERS]].min(axis=1)
pred["roll_tail_mean"] = pred[[f"roll_R_{l}" for l in TAIL_LAYERS]].mean(axis=1)
pred["roll_tail_slope"] = pred["roll_R_26"] - pred["roll_R_23"]

SHAPE_FEATURES = [
    "base_tail_span", "base_tail_mean", "base_tail_slope",
    "roll_tail_span", "roll_tail_mean", "roll_tail_slope",
]

# Energy columns from 4A prediction may not exist; merge from base if possible is difficult because source_row is candidate-local.
# Therefore use whatever is present.
ENERGY_FEATURES = [
    c for c in pred.columns
    if c.startswith("E_") or c in [
        "true_E_decay_22_to_24",
        "pred_E_decay_22_to_24",
        "energy_decay_gap",
    ]
]

FEATURE_SETS = {
    "init_only": INIT_FEATURES,
    "base_velocity_field": BASE_V_FEATURES,
    "roll_velocity_field": ROLL_V_FEATURES,
    "divergence_signature": DIVERGENCE_FEATURES,
    "base_roll_shape": SHAPE_FEATURES,
    "proposal_plus_shape": BASE_V_FEATURES + SHAPE_FEATURES,
    "all_available_Z": INIT_FEATURES + BASE_V_FEATURES + ROLL_V_FEATURES + DIVERGENCE_FEATURES + SHAPE_FEATURES + ENERGY_FEATURES,
}

FEATURE_SETS = {
    k: [f for f in dict.fromkeys(v) if f in pred.columns]
    for k, v in FEATURE_SETS.items()
}
FEATURE_SETS = {k: v for k, v in FEATURE_SETS.items() if len(v) > 0}

# ============================================================
# HELPERS
# ============================================================

def clean_xy(data, feats, target):
    cols = feats + [target]
    sub = data[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    X = sub[feats].values.astype(float)
    y = sub[target].values
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    X = np.clip(X, -1e6, 1e6)
    return sub, X, y


def get_splits(data, group_mode=True):
    if group_mode:
        groups = data[GROUP_COL].values if GROUP_COL in data.columns else data["condition"].values
        if len(np.unique(groups)) >= 5:
            return list(GroupKFold(n_splits=5).split(data, groups=groups))
    return list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(data))


def make_reg(kind):
    if kind == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ])
    if kind == "poly2_ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("ridge", Ridge(alpha=1.0)),
        ])
    raise ValueError(kind)


def make_cls():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=5000, class_weight="balanced")),
    ])


def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def cv_reg(data, feats, target, model_kind, group_mode):
    sub, X, y = clean_xy(data, feats, target)
    y = y.astype(float)

    splits = get_splits(sub, group_mode=group_mode)
    pred_y = np.zeros(len(sub), dtype=float)

    for tr, te in splits:
        model = make_reg(model_kind)
        model.fit(X[tr], y[tr])
        pred_y[te] = model.predict(X[te])

    return {
        "n": int(len(sub)),
        "r2": float(r2_score(y, pred_y)),
        "corr": safe_corr(y, pred_y),
        "mae": float(mean_absolute_error(y, pred_y)),
    }


def cv_cls(data, feats, target, group_mode):
    sub, X, y = clean_xy(data, feats, target)
    y = y.astype(int)

    if len(np.unique(y)) < 2:
        return None

    splits = get_splits(sub, group_mode=group_mode)
    prob = np.zeros(len(sub), dtype=float)

    for tr, te in splits:
        model = make_cls()
        model.fit(X[tr], y[tr])
        prob[te] = model.predict_proba(X[te])[:, 1]

    label = (prob >= 0.5).astype(int)

    return {
        "n": int(len(sub)),
        "auc": float(roc_auc_score(y, prob)),
        "acc": float(accuracy_score(y, label)),
        "f1": float(f1_score(y, label, zero_division=0)),
    }

# ============================================================
# MAIN AUDIT
# ============================================================

REG_TARGETS = [
    "target_roll_R26_abs_error",
    "target_roll_minus_base_R26_abs",
    "target_roll_tail_mae",
    "target_roll_minus_base_tail_mae",
]

CLS_TARGETS = [
    "roll_worse_R26",
    "roll_worse_tail",
]

reg_rows = []
cls_rows = []

for candidate_id, sub0 in pred.groupby("candidate_id"):
    if len(sub0) < 50:
        continue

    cand_parts = candidate_id.split("|")

    for group_mode in [False, True]:
        audit_cv = "GroupKFold" if group_mode else "KFold"

        for fs_name, feats in FEATURE_SETS.items():
            for target in REG_TARGETS:
                for model_kind in ["ridge", "poly2_ridge"]:
                    try:
                        res = cv_reg(sub0, feats, target, model_kind, group_mode)
                    except Exception as e:
                        continue

                    reg_rows.append({
                        "candidate_id": candidate_id,
                        "audit_cv": audit_cv,
                        "model": model_kind,
                        "feature_set": fs_name,
                        "target": target,
                        "features": ",".join(feats),
                        **res,
                    })

            for target in CLS_TARGETS:
                try:
                    res = cv_cls(sub0, feats, target, group_mode)
                except Exception:
                    res = None

                if res is None:
                    continue

                cls_rows.append({
                    "candidate_id": candidate_id,
                    "audit_cv": audit_cv,
                    "model": "logreg",
                    "feature_set": fs_name,
                    "target": target,
                    "features": ",".join(feats),
                    **res,
                })

reg_df = pd.DataFrame(reg_rows).sort_values(
    ["audit_cv", "target", "r2"],
    ascending=[True, True, False],
)

cls_df = pd.DataFrame(cls_rows).sort_values(
    ["audit_cv", "target", "auc"],
    ascending=[True, True, False],
)

reg_df.to_csv(SAVE_DIR / "phasemap4h_hidden_regression.csv", index=False)
cls_df.to_csv(SAVE_DIR / "phasemap4h_hidden_classification.csv", index=False)

# ============================================================
# FEATURE FAMILY COMPARISON
# ============================================================

family_summary = []

for target in REG_TARGETS:
    for audit_cv in ["KFold", "GroupKFold"]:
        sub = reg_df[(reg_df["target"] == target) & (reg_df["audit_cv"] == audit_cv)]
        if len(sub) == 0:
            continue

        for fs_name in FEATURE_SETS:
            ss = sub[sub["feature_set"] == fs_name]
            if len(ss) == 0:
                continue
            best = ss.sort_values("r2", ascending=False).iloc[0]
            family_summary.append({
                "task_type": "regression",
                "target": target,
                "audit_cv": audit_cv,
                "feature_set": fs_name,
                "best_score": float(best["r2"]),
                "best_corr": float(best["corr"]),
                "best_model": best["model"],
            })

for target in CLS_TARGETS:
    for audit_cv in ["KFold", "GroupKFold"]:
        sub = cls_df[(cls_df["target"] == target) & (cls_df["audit_cv"] == audit_cv)]
        if len(sub) == 0:
            continue

        for fs_name in FEATURE_SETS:
            ss = sub[sub["feature_set"] == fs_name]
            if len(ss) == 0:
                continue
            best = ss.sort_values("auc", ascending=False).iloc[0]
            family_summary.append({
                "task_type": "classification",
                "target": target,
                "audit_cv": audit_cv,
                "feature_set": fs_name,
                "best_score": float(best["auc"]),
                "best_corr": None,
                "best_model": best["model"],
            })

family_df = pd.DataFrame(family_summary)
family_df.to_csv(SAVE_DIR / "phasemap4h_feature_family_summary.csv", index=False)

# ============================================================
# SUMMARY
# ============================================================

def top_reg(target, n=8):
    return reg_df[
        (reg_df["audit_cv"] == "GroupKFold")
        & (reg_df["target"] == target)
    ].head(n).to_dict(orient="records")


def top_cls(target, n=8):
    return cls_df[
        (cls_df["audit_cv"] == "GroupKFold")
        & (cls_df["target"] == target)
    ].head(n).to_dict(orient="records")


summary = {
    "experiment": "PhaseMap-4H Hidden State Candidate Audit",
    "input_path": str(PRED_PATH),
    "n_rows": int(len(pred)),
    "core_question": "Which hidden-state proxy explains why explicit recursive rollout fails?",
    "feature_sets": FEATURE_SETS,
    "top_regression": {
        target: top_reg(target)
        for target in REG_TARGETS
    },
    "top_classification": {
        target: top_cls(target)
        for target in CLS_TARGETS
    },
    "interpretation_rules": {
        "Z_is_future_proposal_field": [
            "base_velocity_field or proposal_plus_shape dominates.",
            "Missing state is not local R,v,a,E but future velocity schedule / internal plan."
        ],
        "Z_is_divergence_signature": [
            "divergence_signature dominates.",
            "Failure is caused by self-induced rollout drift; needs closed-loop stabilization."
        ],
        "Z_is_initial_state": [
            "init_only dominates.",
            "Initial R20-R22 still contains hidden predictive structure."
        ],
        "Z_not_found": [
            "All feature families weak under GroupKFold.",
            "Need new hidden probes from actual hidden states / TopK trajectory, not R-space only."
        ],
        "next_if_found": "PhaseMap-4I: construct closed state using best Z proxy.",
        "next_if_not_found": "PhaseMap-4I: extract hidden/TopK latent Z from layers L7-L19 or L20-L22."
    }
}

with open(SAVE_DIR / "phasemap4h_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nSaved to: {SAVE_DIR.resolve()}")