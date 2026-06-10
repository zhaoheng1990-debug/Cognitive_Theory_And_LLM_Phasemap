import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

SAVE_DIR = Path(r"C:\Windows\System32\phasemap5a_outputs")
C1_DIR = Path(r"C:\Windows\System32\phasemap5c1_outputs")
OUT_DIR = Path("./phasemap6a_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SAVE_DIR / "phasemap5a_dataset.csv")
w_df = pd.read_csv(SAVE_DIR / "phasemap5c_w_features.csv")
resid = pd.read_csv(C1_DIR / "phasemap5c1_regression_residual_detail.csv")

# 只分析 instability_energy 的 residual
resid = resid[resid["target"] == "target_instability_energy"].copy()

full = pd.concat(
    [
        df.reset_index(drop=True),
        w_df.reset_index(drop=True),
    ],
    axis=1,
)

# remove duplicated columns caused by 5C dataset already containing W features
full = full.loc[:, ~full.columns.duplicated()].copy()

# 按 row_index 对齐 residual
resid = resid.sort_values("row_index")
full = full.iloc[resid["row_index"].values].reset_index(drop=True)
resid = resid.reset_index(drop=True)

for col in ["residual", "abs_residual", "sq_residual"]:
    full[col] = resid[col].values

w_cols = list(w_df.columns)
groups = full["graph_id"].values

def safe_corr(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)

    if len(a) != len(b):
        return np.nan

    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])

# =========================
# 1. 单变量相关审计
# =========================

corr_rows = []

for feat in w_cols:
    x = full[feat].values

    for target in ["abs_residual", "sq_residual"]:
        y = full[target].values
        corr_rows.append({
            "feature": feat,
            "target": target,
            "corr": safe_corr(x, y),
            "abs_corr": abs(safe_corr(x, y)) if not np.isnan(safe_corr(x, y)) else np.nan,
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
        })

corr_df = pd.DataFrame(corr_rows)
corr_df = corr_df.sort_values(["target", "abs_corr"], ascending=[True, False])

# =========================
# 2. GroupKFold 预测 residual
# =========================

def eval_model(target, model_name):
    X = full[w_cols].values
    y = full[target].values

    gkf = GroupKFold(n_splits=5)
    rows = []
    pred_rows = []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        if model_name == "ridge":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=1.0)),
            ])
        elif model_name == "lasso":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", LassoCV(cv=3, random_state=42, max_iter=5000)),
            ])
        elif model_name == "rf":
            model = RandomForestRegressor(
                n_estimators=300,
                max_depth=5,
                random_state=42,
                min_samples_leaf=5,
            )
        else:
            raise ValueError(model_name)

        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])

        rows.append({
            "target": target,
            "model": model_name,
            "fold": fold,
            "r2": r2_score(y[te], pred),
            "mae": mean_absolute_error(y[te], pred),
            "corr": safe_corr(y[te], pred),
        })

        for idx, yy, pp in zip(te, y[te], pred):
            pred_rows.append({
                "row_index": int(idx),
                "target": target,
                "model": model_name,
                "fold": fold,
                "y_true": float(yy),
                "y_pred": float(pp),
                "error": float(yy - pp),
            })

    return pd.DataFrame(rows), pd.DataFrame(pred_rows)

cv_all = []
pred_all = []

for target in ["abs_residual", "sq_residual"]:
    for model_name in ["ridge", "lasso", "rf"]:
        cv, pred = eval_model(target, model_name)
        cv_all.append(cv)
        pred_all.append(pred)

cv_df = pd.concat(cv_all, ignore_index=True)
pred_df = pd.concat(pred_all, ignore_index=True)

cv_summary = cv_df.groupby(["target", "model"]).agg(
    r2_mean=("r2", "mean"),
    r2_std=("r2", "std"),
    mae_mean=("mae", "mean"),
    mae_std=("mae", "std"),
    corr_mean=("corr", "mean"),
    corr_std=("corr", "std"),
).reset_index()

# =========================
# 3. W 几何分组：找高风险边界
# =========================

risk_rows = []

for feat in w_cols:
    values = full[feat].values

    try:
        q_low = np.quantile(values, 0.25)
        q_high = np.quantile(values, 0.75)
    except Exception:
        continue

    low = full[values <= q_low]
    high = full[values >= q_high]

    if len(low) < 10 or len(high) < 10:
        continue

    risk_rows.append({
        "feature": feat,
        "low_mean_abs_residual": float(low["abs_residual"].mean()),
        "high_mean_abs_residual": float(high["abs_residual"].mean()),
        "high_minus_low_abs_residual": float(
            high["abs_residual"].mean() - low["abs_residual"].mean()
        ),
        "low_mean_sq_residual": float(low["sq_residual"].mean()),
        "high_mean_sq_residual": float(high["sq_residual"].mean()),
        "high_minus_low_sq_residual": float(
            high["sq_residual"].mean() - low["sq_residual"].mean()
        ),
    })

risk_df = pd.DataFrame(risk_rows)
risk_df["abs_effect"] = risk_df["high_minus_low_abs_residual"].abs()
risk_df = risk_df.sort_values("abs_effect", ascending=False)

# =========================
# SAVE
# =========================

corr_df.to_csv(OUT_DIR / "phasemap6a_w_feature_correlations.csv", index=False)
cv_df.to_csv(OUT_DIR / "phasemap6a_boundary_prediction_cv.csv", index=False)
cv_summary.to_csv(OUT_DIR / "phasemap6a_boundary_prediction_summary.csv", index=False)
pred_df.to_csv(OUT_DIR / "phasemap6a_boundary_predictions.csv", index=False)
risk_df.to_csv(OUT_DIR / "phasemap6a_boundary_risk_effects.csv", index=False)

summary = {
    "n_samples": int(len(full)),
    "n_graphs": int(full["graph_id"].nunique()),
    "top_correlations": corr_df.head(20).to_dict(orient="records"),
    "prediction_summary": cv_summary.to_dict(orient="records"),
    "top_boundary_effects": risk_df.head(20).to_dict(orient="records"),
}

with open(OUT_DIR / "phasemap6a_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("Done.")
print("Saved to:", OUT_DIR)
print("\nPrediction summary:")
print(cv_summary)
print("\nTop W boundary correlations:")
print(corr_df.head(20))
print("\nTop W boundary risk effects:")
print(risk_df.head(20))