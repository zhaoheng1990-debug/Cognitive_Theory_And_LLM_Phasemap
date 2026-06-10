import json
from pathlib import Path

import numpy as np
import pandas as pd

IN_DIR = Path(r"C:\Windows\System32\phasemap6b_full_outputs")
OUT_DIR = Path("./phasemap6b1_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN_DIR / "phasemap6b_full_transition_dataset.csv")

# 基础量
df["boundary_dist_l"] = df["R_l"].abs()
df["boundary_dist_l1"] = df["R_l1"].abs()
df["boundary_dist_delta"] = df["boundary_dist_l1"] - df["boundary_dist_l"]

# correction候选：R方向或rank_gap方向反转
df["correction_candidate"] = (
    (df["R_velocity_reversal"] == 1) |
    (df["rank_gap_reversal"] == 1)
).astype(int)

# 真 correction：反转后远离边界，且不是大重构
df["is_correction"] = (
    (df["correction_candidate"] == 1) &
    (df["boundary_dist_delta"] > 0) &
    (df["jaccard"] > 0.45) &
    (df["center_step"] < 0.12)
).astype(int)

# 伪回退 / 不稳定反转：反转但更接近边界
df["unstable_reversal"] = (
    (df["correction_candidate"] == 1) &
    (df["boundary_dist_delta"] <= 0)
).astype(int)

# 大重构反转：反转伴随低邻域保留
df["reconstructive_reversal"] = (
    (df["correction_candidate"] == 1) &
    ((df["jaccard"] <= 0.45) | (df["center_step"] >= 0.12))
).astype(int)

summary = {}

summary["counts"] = {
    "n_transitions": int(len(df)),
    "correction_candidate": int(df["correction_candidate"].sum()),
    "is_correction": int(df["is_correction"].sum()),
    "unstable_reversal": int(df["unstable_reversal"].sum()),
    "reconstructive_reversal": int(df["reconstructive_reversal"].sum()),
}

summary["fractions"] = {
    k: v / len(df)
    for k, v in summary["counts"].items()
    if k != "n_transitions"
}

# 分组统计
group_cols = ["correction_candidate", "is_correction", "unstable_reversal", "reconstructive_reversal"]

rows = []
for col in group_cols:
    sub = df[df[col] == 1]
    if len(sub) == 0:
        continue
    rows.append({
        "group": col,
        "n": int(len(sub)),
        "jaccard_mean": float(sub["jaccard"].mean()),
        "center_step_mean": float(sub["center_step"].mean()),
        "boundary_dist_delta_mean": float(sub["boundary_dist_delta"].mean()),
        "R_delta_mean": float(sub["R_delta"].mean()),
        "spread_delta_mean": float(sub["spread_delta"].mean()),
        "backtrack_init_mean": float(sub["center_backtrack_init"].mean()),
        "backtrack_prevprev_mean": float(sub["center_backtrack_prevprev"].mean()),
    })

group_summary = pd.DataFrame(rows)

# layer profile
layer_profile = df.groupby("layer").agg(
    n=("layer", "size"),
    correction_candidate_frac=("correction_candidate", "mean"),
    is_correction_frac=("is_correction", "mean"),
    unstable_reversal_frac=("unstable_reversal", "mean"),
    reconstructive_reversal_frac=("reconstructive_reversal", "mean"),
    boundary_dist_delta_mean=("boundary_dist_delta", "mean"),
).reset_index()

# condition profile
condition_profile = df.groupby("condition").agg(
    n=("condition", "size"),
    correction_candidate_frac=("correction_candidate", "mean"),
    is_correction_frac=("is_correction", "mean"),
    unstable_reversal_frac=("unstable_reversal", "mean"),
    reconstructive_reversal_frac=("reconstructive_reversal", "mean"),
    boundary_dist_delta_mean=("boundary_dist_delta", "mean"),
).reset_index()

# phase profile
phase_profile = df.groupby("phase").agg(
    n=("phase", "size"),
    correction_candidate_frac=("correction_candidate", "mean"),
    is_correction_frac=("is_correction", "mean"),
    unstable_reversal_frac=("unstable_reversal", "mean"),
    reconstructive_reversal_frac=("reconstructive_reversal", "mean"),
    boundary_dist_delta_mean=("boundary_dist_delta", "mean"),
).reset_index()

# 关键：如果 correction 在 L20-L22 / L23-L26 比例高，说明它参与盆地选择/形成
boundary_layers = df[df["layer"].between(20, 22)]
basin_layers = df[df["layer"].between(23, 25)]
fit_layers = df[df["layer"].between(7, 19)]

summary["regime_profile"] = {
    "fit_L7_19": {
        "n": int(len(fit_layers)),
        "is_correction_frac": float(fit_layers["is_correction"].mean()),
        "unstable_reversal_frac": float(fit_layers["unstable_reversal"].mean()),
    },
    "critical_L20_22": {
        "n": int(len(boundary_layers)),
        "is_correction_frac": float(boundary_layers["is_correction"].mean()),
        "unstable_reversal_frac": float(boundary_layers["unstable_reversal"].mean()),
    },
    "basin_L23_25": {
        "n": int(len(basin_layers)),
        "is_correction_frac": float(basin_layers["is_correction"].mean()),
        "unstable_reversal_frac": float(basin_layers["unstable_reversal"].mean()),
    },
}

df.to_csv(OUT_DIR / "phasemap6b1_correction_dataset.csv", index=False)
group_summary.to_csv(OUT_DIR / "phasemap6b1_correction_group_summary.csv", index=False)
layer_profile.to_csv(OUT_DIR / "phasemap6b1_correction_layer_profile.csv", index=False)
condition_profile.to_csv(OUT_DIR / "phasemap6b1_correction_condition_profile.csv", index=False)
phase_profile.to_csv(OUT_DIR / "phasemap6b1_correction_phase_profile.csv", index=False)

summary["group_summary"] = group_summary.to_dict(orient="records")
summary["layer_profile_head"] = layer_profile.head(20).to_dict(orient="records")
summary["condition_profile"] = condition_profile.to_dict(orient="records")
summary["phase_profile"] = phase_profile.to_dict(orient="records")

with open(OUT_DIR / "phasemap6b1_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("Done.")
print(json.dumps(summary, ensure_ascii=False, indent=2))