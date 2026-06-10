# ============================================================
# PhaseMap-5C.1
# Boundary Condition Residual Audit
#
# Goal:
#   Test whether static W geometry explains the residual error
#   left by Hybrid(R_space + TopK_state).
#
# Hypothesis:
#   W is not a direct state variable.
#   W behaves more like a boundary condition / landscape geometry.
#
# Core test:
#   Hybrid(R+TopK) -> target
#   residual = target - pred_hybrid
#   W_geometry -> |residual| or residual^2
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    roc_auc_score,
    accuracy_score,
    f1_score,
)

# =========================
# CONFIG
# =========================

SAVE_DIR = Path(r"C:\Windows\System32\phasemap5a_outputs")

DATASET_PATH = SAVE_DIR / "phasemap5a_dataset.csv"
R_PATH = SAVE_DIR / "phasemap5a_r_features.csv"
TOPK_PATH = SAVE_DIR / "phasemap5a_topk_features.csv"
W_PATH = SAVE_DIR / "phasemap5c_w_features.csv"

OUT_DIR = Path("./phasemap5c1_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5

# =========================
# LOAD
# =========================

df = pd.read_csv(DATASET_PATH)
r_df = pd.read_csv(R_PATH)
topk_df = pd.read_csv(TOPK_PATH)
w_df = pd.read_csv(W_PATH)

full = pd.concat(
    [
        df.reset_index(drop=True),
        r_df.reset_index(drop=True),
        topk_df.reset_index(drop=True),
        w_df.reset_index(drop=True),
    ],
    axis=1,
)

r_feature_cols = list(r_df.columns)
topk_feature_cols = list(topk_df.columns)
w_feature_cols = list(w_df.columns)

hybrid_cols = r_feature_cols + topk_feature_cols

groups = full["graph_id"].values

# =========================
# TARGETS
# =========================

classification_targets = [
    "target_GenE_proxy",
    "target_overshoot_proxy",
    "target_oscillation_proxy",
]

regression_targets = [
    "target_instability_energy",
]

# =========================
# HELPERS
# =========================

def safe_corr(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def eval_regression_residual(target_col):
    """
    Step 1:
        Hybrid(R+TopK) predicts y.
    Step 2:
        W predicts abs residual / squared residual.
    """

    X_h = full[hybrid_cols].values
    X_w = full[w_feature_cols].values
    y = full[target_col].values.astype(float)

    gkf = GroupKFold(n_splits=N_SPLITS)

    rows = []
    residual_rows = []

    for fold, (tr, te) in enumerate(gkf.split(X_h, y, groups)):
        hybrid_model = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0)),
        ])

        hybrid_model.fit(X_h[tr], y[tr])

        pred_h_te = hybrid_model.predict(X_h[te])
        pred_h_tr = hybrid_model.predict(X_h[tr])

        resid_tr = y[tr] - pred_h_tr
        resid_te = y[te] - pred_h_te

        abs_resid_tr = np.abs(resid_tr)
        abs_resid_te = np.abs(resid_te)

        sq_resid_tr = resid_tr ** 2
        sq_resid_te = resid_te ** 2

        for resid_name, y_res_tr, y_res_te in [
            ("abs_residual", abs_resid_tr, abs_resid_te),
            ("sq_residual", sq_resid_tr, sq_resid_te),
        ]:
            w_model = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=1.0)),
            ])

            w_model.fit(X_w[tr], y_res_tr)
            pred_res = w_model.predict(X_w[te])

            rows.append({
                "target": target_col,
                "residual_target": resid_name,
                "fold": fold,
                "hybrid_target_r2": r2_score(y[te], pred_h_te),
                "hybrid_target_mae": mean_absolute_error(y[te], pred_h_te),
                "w_residual_r2": r2_score(y_res_te, pred_res),
                "w_residual_mae": mean_absolute_error(y_res_te, pred_res),
                "w_residual_corr": safe_corr(y_res_te, pred_res),
                "true_residual_mean": float(np.mean(y_res_te)),
                "pred_residual_mean": float(np.mean(pred_res)),
            })

        for idx, true_y, pred_y, resid in zip(te, y[te], pred_h_te, resid_te):
            residual_rows.append({
                "row_index": int(idx),
                "target": target_col,
                "fold": fold,
                "y_true": float(true_y),
                "hybrid_pred": float(pred_y),
                "residual": float(resid),
                "abs_residual": float(abs(resid)),
                "sq_residual": float(resid ** 2),
            })

    return pd.DataFrame(rows), pd.DataFrame(residual_rows)


def eval_classification_residual(target_col):
    """
    Step 1:
        Hybrid(R+TopK) predicts probability p.
    Step 2:
        W predicts whether Hybrid made an error.
    """

    X_h = full[hybrid_cols].values
    X_w = full[w_feature_cols].values
    y = full[target_col].values.astype(int)

    gkf = GroupKFold(n_splits=N_SPLITS)

    rows = []
    residual_rows = []

    for fold, (tr, te) in enumerate(gkf.split(X_h, y, groups)):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue

        hybrid_model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
            )),
        ])

        hybrid_model.fit(X_h[tr], y[tr])

        prob_te = hybrid_model.predict_proba(X_h[te])[:, 1]
        pred_te = (prob_te >= 0.5).astype(int)

        prob_tr = hybrid_model.predict_proba(X_h[tr])[:, 1]
        pred_tr = (prob_tr >= 0.5).astype(int)

        err_tr = (pred_tr != y[tr]).astype(int)
        err_te = (pred_te != y[te]).astype(int)

        # If no errors in train/test, AUC cannot be computed.
        if len(np.unique(err_tr)) < 2 or len(np.unique(err_te)) < 2:
            rows.append({
                "target": target_col,
                "fold": fold,
                "hybrid_auc": roc_auc_score(y[te], prob_te),
                "hybrid_acc": accuracy_score(y[te], pred_te),
                "hybrid_f1": f1_score(y[te], pred_te),
                "n_test_errors": int(err_te.sum()),
                "w_error_auc": np.nan,
                "w_error_acc": np.nan,
                "w_error_f1": np.nan,
                "note": "insufficient error class variation",
            })
        else:
            w_model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                )),
            ])

            w_model.fit(X_w[tr], err_tr)
            err_prob = w_model.predict_proba(X_w[te])[:, 1]
            err_pred = (err_prob >= 0.5).astype(int)

            rows.append({
                "target": target_col,
                "fold": fold,
                "hybrid_auc": roc_auc_score(y[te], prob_te),
                "hybrid_acc": accuracy_score(y[te], pred_te),
                "hybrid_f1": f1_score(y[te], pred_te),
                "n_test_errors": int(err_te.sum()),
                "w_error_auc": roc_auc_score(err_te, err_prob),
                "w_error_acc": accuracy_score(err_te, err_pred),
                "w_error_f1": f1_score(err_te, err_pred),
                "note": "",
            })

        for idx, yy, pp, pred, err in zip(te, y[te], prob_te, pred_te, err_te):
            residual_rows.append({
                "row_index": int(idx),
                "target": target_col,
                "fold": fold,
                "y_true": int(yy),
                "hybrid_prob": float(pp),
                "hybrid_pred": int(pred),
                "hybrid_error": int(err),
                "margin_uncertainty": float(abs(pp - 0.5)),
            })

    return pd.DataFrame(rows), pd.DataFrame(residual_rows)


# =========================
# RUN
# =========================

reg_cv = []
reg_residual_detail = []

for target in regression_targets:
    cv, detail = eval_regression_residual(target)
    reg_cv.append(cv)
    reg_residual_detail.append(detail)

reg_cv = pd.concat(reg_cv, ignore_index=True)
reg_residual_detail = pd.concat(reg_residual_detail, ignore_index=True)

cls_cv = []
cls_residual_detail = []

for target in classification_targets:
    cv, detail = eval_classification_residual(target)
    cls_cv.append(cv)
    cls_residual_detail.append(detail)

cls_cv = pd.concat(cls_cv, ignore_index=True)
cls_residual_detail = pd.concat(cls_residual_detail, ignore_index=True)

# =========================
# SUMMARIES
# =========================

reg_summary = reg_cv.groupby(
    ["target", "residual_target"]
).agg(
    hybrid_target_r2_mean=("hybrid_target_r2", "mean"),
    hybrid_target_r2_std=("hybrid_target_r2", "std"),
    hybrid_target_mae_mean=("hybrid_target_mae", "mean"),
    hybrid_target_mae_std=("hybrid_target_mae", "std"),
    w_residual_r2_mean=("w_residual_r2", "mean"),
    w_residual_r2_std=("w_residual_r2", "std"),
    w_residual_mae_mean=("w_residual_mae", "mean"),
    w_residual_mae_std=("w_residual_mae", "std"),
    w_residual_corr_mean=("w_residual_corr", "mean"),
    w_residual_corr_std=("w_residual_corr", "std"),
    true_residual_mean=("true_residual_mean", "mean"),
    pred_residual_mean=("pred_residual_mean", "mean"),
).reset_index()

cls_summary = cls_cv.groupby(
    ["target"]
).agg(
    hybrid_auc_mean=("hybrid_auc", "mean"),
    hybrid_auc_std=("hybrid_auc", "std"),
    hybrid_acc_mean=("hybrid_acc", "mean"),
    hybrid_acc_std=("hybrid_acc", "std"),
    hybrid_f1_mean=("hybrid_f1", "mean"),
    hybrid_f1_std=("hybrid_f1", "std"),
    n_test_errors_mean=("n_test_errors", "mean"),
    w_error_auc_mean=("w_error_auc", "mean"),
    w_error_auc_std=("w_error_auc", "std"),
    w_error_acc_mean=("w_error_acc", "mean"),
    w_error_acc_std=("w_error_acc", "std"),
    w_error_f1_mean=("w_error_f1", "mean"),
    w_error_f1_std=("w_error_f1", "std"),
).reset_index()

# =========================
# SAVE
# =========================

reg_cv.to_csv(OUT_DIR / "phasemap5c1_regression_residual_cv.csv", index=False)
reg_summary.to_csv(OUT_DIR / "phasemap5c1_regression_residual_summary.csv", index=False)
reg_residual_detail.to_csv(OUT_DIR / "phasemap5c1_regression_residual_detail.csv", index=False)

cls_cv.to_csv(OUT_DIR / "phasemap5c1_classification_error_cv.csv", index=False)
cls_summary.to_csv(OUT_DIR / "phasemap5c1_classification_error_summary.csv", index=False)
cls_residual_detail.to_csv(OUT_DIR / "phasemap5c1_classification_error_detail.csv", index=False)

summary = {
    "n_samples": int(len(full)),
    "n_graphs": int(full["graph_id"].nunique()),
    "regression_summary": reg_summary.to_dict(orient="records"),
    "classification_error_summary": cls_summary.to_dict(orient="records"),
}

with open(OUT_DIR / "phasemap5c1_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\nDone.")
print("Saved to:", OUT_DIR)
print("\nRegression residual summary:")
print(reg_summary)

print("\nClassification error summary:")
print(cls_summary)