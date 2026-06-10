import json
from pathlib import Path

import pandas as pd

SAVE_DIR = Path(r"C:\Windows\System32\phasemap5a_outputs")

cls_out = pd.read_csv(SAVE_DIR / "phasemap5a_classification_cv.csv")
reg_out = pd.read_csv(SAVE_DIR / "phasemap5a_regression_cv.csv")

cls_summary = cls_out.groupby(
    ["target", "feature_set"]
).agg(
    auc_mean=("auc", "mean"),
    auc_std=("auc", "std"),
    acc_mean=("acc", "mean"),
    acc_std=("acc", "std"),
    f1_mean=("f1", "mean"),
    f1_std=("f1", "std"),
).reset_index()

reg_summary = reg_out.groupby(
    ["target", "feature_set"]
).agg(
    r2_mean=("r2", "mean"),
    r2_std=("r2", "std"),
    mae_mean=("mae", "mean"),
    mae_std=("mae", "std"),
    corr_mean=("corr", "mean"),
    corr_std=("corr", "std"),
).reset_index()

cls_summary.to_csv(
    SAVE_DIR / "phasemap5a_classification_summary.csv",
    index=False,
)

reg_summary.to_csv(
    SAVE_DIR / "phasemap5a_regression_summary.csv",
    index=False,
)

summary = {
    "classification_summary": cls_summary.to_dict(orient="records"),
    "regression_summary": reg_summary.to_dict(orient="records"),
}

with open(
    SAVE_DIR / "phasemap5a_summary.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("Done.")
print("Saved:")
print(SAVE_DIR / "phasemap5a_classification_summary.csv")
print(SAVE_DIR / "phasemap5a_regression_summary.csv")
print(SAVE_DIR / "phasemap5a_summary.json")