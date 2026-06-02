import json
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

INPUT_CANDIDATES = [
    Path(r"C:\Windows\System32\phasemap3a3_outputs\phasemap3a3_rollout_predictions.csv"),
    Path(r"C:\Windows\System32\phasemap3a2_outputs\phasemap3a2_rollout_predictions.csv"),
]

SAVE_DIR = Path("./phasemap3a4_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

R_LAYERS = list(range(20, 27))
PRED_LAYERS = [23, 24, 25, 26]

# ============================================================
# LOAD
# ============================================================

INPUT_PATH = None
for p in INPUT_CANDIDATES:
    if p.exists():
        INPUT_PATH = p
        break

if INPUT_PATH is None:
    raise FileNotFoundError(
        "Cannot find rollout predictions from 3A.3 or 3A.2."
    )

df = pd.read_csv(INPUT_PATH)

required_base = ["cv", "feature_set", "condition", "phase_target"]
missing = [c for c in required_base if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing base columns: {missing}")

for l in R_LAYERS:
    for prefix in ["true_R", "pred_R", "err_R"]:
        c = f"{prefix}_{l}"
        if c not in df.columns:
            raise RuntimeError(f"Missing column: {c}")

df = df.copy()
df["phase_target"] = df["phase_target"].astype(str)
df["condition"] = df["condition"].astype(str)
df["feature_set"] = df["feature_set"].astype(str)
df["cv"] = df["cv"].astype(str)

if "model" not in df.columns:
    df["model"] = "unknown"

df["model"] = df["model"].astype(str)

# ============================================================
# BUILD ERROR FEATURES
# ============================================================

for l in R_LAYERS:
    df[f"abs_err_R_{l}"] = df[f"err_R_{l}"].abs()

for a, b in zip(R_LAYERS[:-1], R_LAYERS[1:]):
    df[f"true_dR_{a}_{b}"] = df[f"true_R_{b}"] - df[f"true_R_{a}"]
    df[f"pred_dR_{a}_{b}"] = df[f"pred_R_{b}"] - df[f"pred_R_{a}"]
    df[f"err_dR_{a}_{b}"] = df[f"pred_dR_{a}_{b}"] - df[f"true_dR_{a}_{b}"]
    df[f"abs_err_dR_{a}_{b}"] = df[f"err_dR_{a}_{b}"].abs()

# Tail aggregate errors
df["tail_mae_23_26_recalc"] = df[[f"abs_err_R_{l}" for l in PRED_LAYERS]].mean(axis=1)
df["final_abs_error_26_recalc"] = df["abs_err_R_26"]

# Error growth
df["err_growth_23_to_26"] = df["abs_err_R_26"] - df["abs_err_R_23"]
df["err_growth_ratio_26_over_23"] = df["abs_err_R_26"] / (df["abs_err_R_23"] + 1e-9)

# Where max error appears
abs_cols = [f"abs_err_R_{l}" for l in PRED_LAYERS]
df["max_error_layer"] = df[abs_cols].idxmax(axis=1).str.extract(r"(\d+)").astype(int)
df["max_error_value"] = df[abs_cols].max(axis=1)

# First explosion layer: first layer where abs error exceeds thresholds
for threshold in [1.0, 2.0, 3.0]:
    out_col = f"first_layer_abs_err_gt_{threshold}"
    vals = []
    for _, row in df.iterrows():
        hit = None
        for l in PRED_LAYERS:
            if row[f"abs_err_R_{l}"] > threshold:
                hit = l
                break
        vals.append(hit if hit is not None else np.nan)
    df[out_col] = vals

# Direction / sign preservation
for l in PRED_LAYERS:
    df[f"true_sign_R_{l}"] = np.sign(df[f"true_R_{l}"])
    df[f"pred_sign_R_{l}"] = np.sign(df[f"pred_R_{l}"])
    df[f"sign_match_R_{l}"] = (df[f"true_sign_R_{l}"] == df[f"pred_sign_R_{l}"]).astype(int)

df["tail_sign_match_rate"] = df[[f"sign_match_R_{l}" for l in PRED_LAYERS]].mean(axis=1)

# Basin commitment preservation at final layer
df["final_sign_match_R26"] = df["sign_match_R_26"]

# ============================================================
# SUMMARY TABLES
# ============================================================

group_cols = ["cv", "model", "feature_set"]

overall_rows = []

for keys, sub in df.groupby(group_cols):
    row = dict(zip(group_cols, keys))
    row.update({
        "n": int(len(sub)),
        "mean_tail_mae": float(sub["tail_mae_23_26_recalc"].mean()),
        "mean_final_abs_error_R26": float(sub["final_abs_error_26_recalc"].mean()),
        "mean_err_growth_23_to_26": float(sub["err_growth_23_to_26"].mean()),
        "median_err_growth_23_to_26": float(sub["err_growth_23_to_26"].median()),
        "mean_tail_sign_match": float(sub["tail_sign_match_rate"].mean()),
        "final_sign_match_rate_R26": float(sub["final_sign_match_R26"].mean()),
    })

    for l in PRED_LAYERS:
        row[f"mean_abs_err_R{l}"] = float(sub[f"abs_err_R_{l}"].mean())
        row[f"median_abs_err_R{l}"] = float(sub[f"abs_err_R_{l}"].median())
        row[f"sign_match_R{l}"] = float(sub[f"sign_match_R_{l}"].mean())

    # distribution of max error layer
    counts = sub["max_error_layer"].value_counts(normalize=True).to_dict()
    for l in PRED_LAYERS:
        row[f"max_error_layer_frac_R{l}"] = float(counts.get(l, 0.0))

    overall_rows.append(row)

overall_summary = pd.DataFrame(overall_rows).sort_values(
    ["cv", "model", "mean_tail_mae"],
    ascending=[True, True, True],
)

overall_summary.to_csv(SAVE_DIR / "phasemap3a4_error_overall.csv", index=False)

# Phase summary
phase_summary = df.groupby(["cv", "model", "feature_set", "phase_target"]).agg(
    n=("phase_target", "count"),
    mean_tail_mae=("tail_mae_23_26_recalc", "mean"),
    mean_final_abs_error_R26=("final_abs_error_26_recalc", "mean"),
    mean_err_growth=("err_growth_23_to_26", "mean"),
    final_sign_match_R26=("final_sign_match_R26", "mean"),
    mean_tail_sign_match=("tail_sign_match_rate", "mean"),
    max_error_layer_mean=("max_error_layer", "mean"),
).reset_index()

phase_summary.to_csv(SAVE_DIR / "phasemap3a4_error_by_phase.csv", index=False)

# Condition summary
condition_summary = df.groupby(["cv", "model", "feature_set", "condition"]).agg(
    n=("condition", "count"),
    phase=("phase_target", lambda x: x.mode().iloc[0]),
    mean_tail_mae=("tail_mae_23_26_recalc", "mean"),
    mean_final_abs_error_R26=("final_abs_error_26_recalc", "mean"),
    mean_err_growth=("err_growth_23_to_26", "mean"),
    final_sign_match_R26=("final_sign_match_R26", "mean"),
    mean_tail_sign_match=("tail_sign_match_rate", "mean"),
    max_error_layer_mean=("max_error_layer", "mean"),
).reset_index()

condition_summary.to_csv(SAVE_DIR / "phasemap3a4_error_by_condition.csv", index=False)

# Layer error table
layer_rows = []

for keys, sub in df.groupby(group_cols):
    keydict = dict(zip(group_cols, keys))

    for l in PRED_LAYERS:
        layer_rows.append({
            **keydict,
            "layer": l,
            "n": int(len(sub)),
            "mean_abs_err": float(sub[f"abs_err_R_{l}"].mean()),
            "median_abs_err": float(sub[f"abs_err_R_{l}"].median()),
            "p90_abs_err": float(sub[f"abs_err_R_{l}"].quantile(0.90)),
            "sign_match_rate": float(sub[f"sign_match_R_{l}"].mean()),
        })

layer_error = pd.DataFrame(layer_rows)
layer_error.to_csv(SAVE_DIR / "phasemap3a4_error_by_layer.csv", index=False)

# Transition velocity error
transition_rows = []

for keys, sub in df.groupby(group_cols):
    keydict = dict(zip(group_cols, keys))

    for a, b in zip(R_LAYERS[:-1], R_LAYERS[1:]):
        if b < 23:
            continue

        transition_rows.append({
            **keydict,
            "transition": f"L{a}->L{b}",
            "n": int(len(sub)),
            "mean_abs_dR_error": float(sub[f"abs_err_dR_{a}_{b}"].mean()),
            "median_abs_dR_error": float(sub[f"abs_err_dR_{a}_{b}"].median()),
            "p90_abs_dR_error": float(sub[f"abs_err_dR_{a}_{b}"].quantile(0.90)),
        })

transition_error = pd.DataFrame(transition_rows)
transition_error.to_csv(SAVE_DIR / "phasemap3a4_transition_error.csv", index=False)

# Explosion layer distribution
explosion_rows = []

for keys, sub in df.groupby(group_cols):
    keydict = dict(zip(group_cols, keys))

    for threshold in [1.0, 2.0, 3.0]:
        col = f"first_layer_abs_err_gt_{threshold}"
        vc = sub[col].value_counts(dropna=False, normalize=True)

        for layer_val, frac in vc.items():
            explosion_rows.append({
                **keydict,
                "threshold": threshold,
                "first_explosion_layer": "none" if pd.isna(layer_val) else int(layer_val),
                "fraction": float(frac),
            })

explosion_summary = pd.DataFrame(explosion_rows)
explosion_summary.to_csv(SAVE_DIR / "phasemap3a4_explosion_layer_distribution.csv", index=False)

# Save enriched predictions
df.to_csv(SAVE_DIR / "phasemap3a4_enriched_predictions.csv", index=False)

# ============================================================
# DIAGNOSTIC DECISION
# ============================================================

best = overall_summary.iloc[0].to_dict()

# Find best GroupKFold if available
group_best_df = overall_summary[overall_summary["cv"] == "GroupKFold"]
if len(group_best_df) > 0:
    best_group = group_best_df.iloc[0].to_dict()
else:
    best_group = best

# Determine likely failure mode
layer_cols = [f"mean_abs_err_R{l}" for l in PRED_LAYERS]
layer_errs = {l: best_group.get(f"mean_abs_err_R{l}", None) for l in PRED_LAYERS}

max_layer = None
if all(v is not None for v in layer_errs.values()):
    max_layer = max(layer_errs, key=lambda l: layer_errs[l])

growth = best_group.get("mean_err_growth_23_to_26", None)
sign_match = best_group.get("final_sign_match_rate_R26", None)

if max_layer in [23, 24]:
    failure_mode = "early_basin_entry_error"
elif max_layer in [25, 26] and growth is not None and growth > 0:
    failure_mode = "progressive_error_accumulation"
elif sign_match is not None and sign_match < 0.80:
    failure_mode = "basin_sign_flip_error"
else:
    failure_mode = "mixed_or_stable_error"

summary = {
    "experiment": "PhaseMap-3A.4 Error Accumulation Audit",
    "input_path": str(INPUT_PATH),
    "n_rows": int(len(df)),
    "core_question": "Where does multi-step rollout error accumulate, and what failure mode blocks global closure?",
    "best_overall": best,
    "best_groupkfold": best_group,
    "best_groupkfold_layer_errors": layer_errs,
    "diagnosis": {
        "likely_failure_mode": failure_mode,
        "max_mean_error_layer": max_layer,
        "mean_error_growth_23_to_26": growth,
        "final_sign_match_rate_R26": sign_match,
    },
    "interpretation_rules": {
        "early_basin_entry_error": [
            "Errors peak at R23/R24.",
            "Missing variable likely controls basin-entry or phase transition."
        ],
        "progressive_error_accumulation": [
            "Errors grow toward R26.",
            "Missing mechanism likely damping / stabilizing attractor correction."
        ],
        "basin_sign_flip_error": [
            "Final sign match is poor.",
            "Model fails to preserve basin commitment."
        ],
        "mixed_or_stable_error": [
            "No single layer dominates.",
            "Need residual modeling or broader history state."
        ],
        "next_steps": [
            "If early error: add transition-gate variable.",
            "If progressive: add damping / attractor correction.",
            "If sign flip: add basin commitment classifier.",
            "If mixed: do residual correction model."
        ]
    }
}

with open(SAVE_DIR / "phasemap3a4_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nSaved to: {SAVE_DIR.resolve()}")