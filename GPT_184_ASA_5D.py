
# ============================================================
# ASA-5D: Acceleration Control Decomposition Audit
#
# Purpose:
#   Offline decomposition audit for ASA-5 outputs.
#
#   ASA-5 showed:
#     acceleration_residual = weak positive / PASS-Lite
#     acceleration_delta    = negative or weak
#     velocity/position     = weaker
#     shuffle z < 2
#
#   ASA-5D asks:
#     1) Is acceleration_residual really separable from raw acceleration_delta?
#     2) Is the positive signal driven by target closure samples or by non-target drift?
#     3) Is the signal stronger in critical / low-inertia / low-boundary-distance samples?
#     4) Does acceleration_residual improve over velocity/position/random consistently by alpha?
#     5) Is there evidence that ASA-5 is a distinct curvature-control channel,
#        or only a weak projection / numerical residual artifact?
#
# Inputs expected in ./asa5_outputs:
#   asa5_summary.json
#   asa5_baseline_features.csv
#   asa5_steering_results.csv
#   asa5_specificity_summary.csv
#   asa5_shuffle_specificity_significance.csv
#   asa5_dose_response.csv
#   asa5_direction_audit.csv
#   asa5_verdict.csv
#
# Outputs:
#   asa5d_outputs/
#     asa5d_control_ranking.csv
#     asa5d_alpha_profile.csv
#     asa5d_target_nontarget_profile.csv
#     asa5d_sample_shift_table.csv
#     asa5d_mechanism_shift_profile.csv
#     asa5d_inertia_proxy_profile.csv
#     asa5d_inertia_shift_correlation.csv
#     asa5d_delta_vs_residual_gap.csv
#     asa5d_shuffle_percentile.csv
#     asa5d_verdict.csv
#     asa5d_summary.json
# ============================================================

import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# ============================================================
# CONFIG
# ============================================================

ASA5_DIR = Path("./asa5_outputs")
SAVE_DIR = Path("./asa5d_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MAIN_DIRECTION = "Q_to_S"
MAIN_CONTROL = "acceleration_residual"
RAW_ACCEL_CONTROL = "acceleration_delta"
STABLE_ACCEL_CONTROL = "acceleration_stable"
VELOCITY_CONTROL = "velocity_residual"
POSITION_CONTROL = "position_residual"
RANDOM_CONTROL = "random"
ANSWER_CONTROL = "answer"

MAIN_ALPHA = 0.30
SHUFFLE_PREFIX = "shuffle_acceleration_residual_"

TARGET_MECHANISM_BY_DIRECTION = {
    "Q_to_S": "closure",
    "K_to_S": "competition",
    "S_to_Q": "stable",
}

# In ASA-5, pair choice target for Q_to_S is clean.
TARGET_PAIR_BY_DIRECTION = {
    "Q_to_S": "clean",
    "K_to_S": "clean",
    "S_to_Q": "conflict",
}

# ============================================================
# HELPERS
# ============================================================

def read_csv_required(name):
    path = ASA5_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)

def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) < 1e-12 or np.std(y[m]) < 1e-12:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])

def safe_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else np.nan

def qcut3(s):
    s = pd.Series(s)
    try:
        return pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"])
    except Exception:
        return pd.Series(["unknown"] * len(s), index=s.index)

def linear_slope_r2(x, y):
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x[:, 0]) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m, 0]) < 1e-12:
        return np.nan, np.nan
    reg = LinearRegression().fit(x[m], y[m])
    pred = reg.predict(x[m])
    return float(reg.coef_[0]), float(r2_score(y[m], pred))

def infer_condition_order(df):
    if "condition" in df.columns:
        return sorted(df["condition"].unique().tolist())
    return []

# ============================================================
# LOAD
# ============================================================

summary_path = ASA5_DIR / "asa5_summary.json"
summary = {}
if summary_path.exists():
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

baseline = read_csv_required("asa5_baseline_features.csv")
steer = read_csv_required("asa5_steering_results.csv")
spec = read_csv_required("asa5_specificity_summary.csv")
sig = read_csv_required("asa5_shuffle_specificity_significance.csv")
dose = read_csv_required("asa5_dose_response.csv")
audit = read_csv_required("asa5_direction_audit.csv")
verdict_in = read_csv_required("asa5_verdict.csv")

# ============================================================
# BASIC MERGED SHIFT TABLE
# ============================================================

base_cols = ["prompt_id", "DeltaU", "pair_choice", "mechanism", "condition", "graph_id"]
base_cols = [c for c in base_cols if c in baseline.columns]
base = baseline[base_cols].copy()
base = base.rename(columns={"DeltaU": "DeltaU_base", "pair_choice": "pair_choice_base"})

merged = steer.merge(base, on="prompt_id", how="left", suffixes=("", "_base_extra"))

# handle duplicated mechanism/condition if present
if "mechanism_base_extra" in merged.columns and "mechanism" not in merged.columns:
    merged["mechanism"] = merged["mechanism_base_extra"]
if "condition_base_extra" in merged.columns and "condition" not in merged.columns:
    merged["condition"] = merged["condition_base_extra"]

merged["shift"] = merged["DeltaU"] - merged["DeltaU_base"]
merged["abs_shift"] = merged["shift"].abs()
merged["target_mechanism"] = merged["direction"].map(TARGET_MECHANISM_BY_DIRECTION)
merged["is_target_mech"] = merged["mechanism"] == merged["target_mechanism"]
merged["target_pair_choice"] = merged["direction"].map(TARGET_PAIR_BY_DIRECTION)
merged["target_hit"] = (merged["pair_choice"] == merged["target_pair_choice"]).astype(int)

merged.to_csv(SAVE_DIR / "asa5d_sample_shift_table.csv", index=False, encoding="utf-8-sig")

# ============================================================
# CONTROL RANKING AT MAIN ALPHA
# ============================================================

main_spec = spec[(spec["direction"] == MAIN_DIRECTION) & (np.isclose(spec["alpha"], MAIN_ALPHA))].copy()
main_spec = main_spec.sort_values("specificity", ascending=False)
main_spec["rank_by_specificity"] = np.arange(1, len(main_spec) + 1)
main_spec.to_csv(SAVE_DIR / "asa5d_control_ranking.csv", index=False, encoding="utf-8-sig")

# ============================================================
# ALPHA PROFILE FOR KEY CONTROLS
# ============================================================

key_controls = [
    MAIN_CONTROL,
    RAW_ACCEL_CONTROL,
    STABLE_ACCEL_CONTROL,
    VELOCITY_CONTROL,
    POSITION_CONTROL,
    RANDOM_CONTROL,
    ANSWER_CONTROL,
]

alpha_prof = spec[
    (spec["direction"] == MAIN_DIRECTION) &
    (spec["control"].isin(key_controls))
].copy()

# Add deltas vs controls at same alpha
wide = alpha_prof.pivot_table(
    index=["direction", "alpha"],
    columns="control",
    values="specificity",
    aggfunc="mean"
).reset_index()

for ctrl in [RAW_ACCEL_CONTROL, STABLE_ACCEL_CONTROL, VELOCITY_CONTROL, POSITION_CONTROL, RANDOM_CONTROL]:
    if MAIN_CONTROL in wide.columns and ctrl in wide.columns:
        wide[f"{MAIN_CONTROL}_minus_{ctrl}"] = wide[MAIN_CONTROL] - wide[ctrl]

wide.to_csv(SAVE_DIR / "asa5d_alpha_profile.csv", index=False, encoding="utf-8-sig")

# ============================================================
# TARGET / NON-TARGET PROFILE
# ============================================================

target_profile = spec[
    (spec["direction"] == MAIN_DIRECTION) &
    (spec["control"].isin(key_controls))
].copy()

target_profile = target_profile.sort_values(["control", "alpha"])
target_profile.to_csv(SAVE_DIR / "asa5d_target_nontarget_profile.csv", index=False, encoding="utf-8-sig")

# ============================================================
# MECHANISM SHIFT PROFILE
# ============================================================

mech_rows = []
for (direction, control, alpha, mech), g in merged.groupby(["direction", "control", "alpha", "mechanism"]):
    if direction != MAIN_DIRECTION or control not in key_controls:
        continue
    mech_rows.append({
        "direction": direction,
        "control": control,
        "alpha": float(alpha),
        "mechanism": mech,
        "mean_shift": safe_mean(g["shift"]),
        "mean_abs_shift": safe_mean(g["abs_shift"]),
        "target_hit_rate": safe_mean(g["target_hit"]),
        "n": int(len(g)),
    })

mech_profile = pd.DataFrame(mech_rows).sort_values(["control", "alpha", "mechanism"])
mech_profile.to_csv(SAVE_DIR / "asa5d_mechanism_shift_profile.csv", index=False, encoding="utf-8-sig")

# ============================================================
# DIRECTION AUDIT DECOMPOSITION
# ============================================================

audit_out = audit.copy()
eps = 1e-12
if "norm_acc_residual" in audit_out.columns and "norm_acc_delta" in audit_out.columns:
    audit_out["residual_to_delta_norm_ratio"] = audit_out["norm_acc_residual"] / (audit_out["norm_acc_delta"] + eps)
if "cos_delta_residual" in audit_out.columns:
    audit_out["abs_cos_delta_residual"] = audit_out["cos_delta_residual"].abs()

audit_out.to_csv(SAVE_DIR / "asa5d_delta_vs_residual_gap.csv", index=False, encoding="utf-8-sig")

# ============================================================
# SHUFFLE PERCENTILE / Z AUDIT
# ============================================================

shuffle_rows = []
for alpha in sorted(sig[sig["direction"] == MAIN_DIRECTION]["alpha"].unique()):
    srow = sig[(sig["direction"] == MAIN_DIRECTION) & (np.isclose(sig["alpha"], alpha))]
    if len(srow) == 0:
        continue
    row = srow.iloc[0].to_dict()

    sh_specs = spec[
        (spec["direction"] == MAIN_DIRECTION) &
        (np.isclose(spec["alpha"], alpha)) &
        (spec["control"].str.startswith(SHUFFLE_PREFIX))
    ]["specificity"].astype(float).values

    proto = float(row.get("acceleration_specificity", np.nan))
    if len(sh_specs):
        percentile = float(np.mean(sh_specs <= proto))
        max_sh = float(np.max(sh_specs))
        min_sh = float(np.min(sh_specs))
        q95 = float(np.quantile(sh_specs, 0.95))
    else:
        percentile, max_sh, min_sh, q95 = np.nan, np.nan, np.nan, np.nan

    row.update({
        "shuffle_percentile": percentile,
        "shuffle_max": max_sh,
        "shuffle_min": min_sh,
        "shuffle_q95": q95,
        "proto_exceeds_shuffle_q95": bool(np.isfinite(proto) and np.isfinite(q95) and proto > q95),
    })
    shuffle_rows.append(row)

shuffle_audit = pd.DataFrame(shuffle_rows)
shuffle_audit.to_csv(SAVE_DIR / "asa5d_shuffle_percentile.csv", index=False, encoding="utf-8-sig")

# ============================================================
# INERTIA PROXY PROFILE
# ============================================================
# We do not have hidden vectors here, so use available DeltaU / pair-choice / baseline variables
# as proxy strata:
#   boundary_proxy = |DeltaU_base| low => nearer decision boundary
#   phase_proxy by mechanism and condition
#   baseline_pair_choice
#
# This is not full inertia. It is a cheap proxy audit:
#   lower |DeltaU_base| may be lower basin commitment / more steerable.

main_rows = merged[
    (merged["direction"] == MAIN_DIRECTION) &
    (merged["control"].isin(key_controls)) &
    (np.isclose(merged["alpha"], MAIN_ALPHA))
].copy()

main_rows["boundary_proxy_abs_DeltaU"] = main_rows["DeltaU_base"].abs()
main_rows["boundary_bin"] = qcut3(main_rows["boundary_proxy_abs_DeltaU"])

# target-only and all-sample bins
bin_rows = []
for (control, b), g in main_rows.groupby(["control", "boundary_bin"]):
    bin_rows.append({
        "control": control,
        "boundary_bin": str(b),
        "mean_abs_DeltaU_base": safe_mean(g["boundary_proxy_abs_DeltaU"]),
        "mean_shift": safe_mean(g["shift"]),
        "mean_abs_shift": safe_mean(g["abs_shift"]),
        "specificity_like_target_abs_minus_nontarget_abs": (
            safe_mean(g[g["is_target_mech"]]["abs_shift"]) -
            safe_mean(g[~g["is_target_mech"]]["abs_shift"])
        ),
        "target_abs_shift": safe_mean(g[g["is_target_mech"]]["abs_shift"]),
        "nontarget_abs_shift": safe_mean(g[~g["is_target_mech"]]["abs_shift"]),
        "target_hit_rate": safe_mean(g["target_hit"]),
        "n": int(len(g)),
    })

bin_profile = pd.DataFrame(bin_rows)
bin_profile.to_csv(SAVE_DIR / "asa5d_inertia_proxy_profile.csv", index=False, encoding="utf-8-sig")

# Correlation between baseline boundary proxy and shift
corr_rows = []
for control, g in main_rows.groupby("control"):
    corr_rows.append({
        "control": control,
        "corr_absDeltaUbase_abs_shift": safe_corr(g["boundary_proxy_abs_DeltaU"], g["abs_shift"]),
        "corr_absDeltaUbase_signed_shift": safe_corr(g["boundary_proxy_abs_DeltaU"], g["shift"]),
        "corr_DeltaUbase_signed_shift": safe_corr(g["DeltaU_base"], g["shift"]),
        "n": int(len(g)),
    })

corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(SAVE_DIR / "asa5d_inertia_shift_correlation.csv", index=False, encoding="utf-8-sig")

# ============================================================
# VERDICT
# ============================================================

# Pull main data
def get_spec(control, alpha=MAIN_ALPHA):
    r = spec[
        (spec["direction"] == MAIN_DIRECTION) &
        (spec["control"] == control) &
        (np.isclose(spec["alpha"], alpha))
    ]
    if len(r) == 0:
        return np.nan
    return float(r.iloc[0]["specificity"])

main_acc = get_spec(MAIN_CONTROL)
raw_acc = get_spec(RAW_ACCEL_CONTROL)
stable_acc = get_spec(STABLE_ACCEL_CONTROL)
vel = get_spec(VELOCITY_CONTROL)
pos = get_spec(POSITION_CONTROL)
rand = get_spec(RANDOM_CONTROL)

sig_main = sig[
    (sig["direction"] == MAIN_DIRECTION) &
    (np.isclose(sig["alpha"], MAIN_ALPHA))
]
if len(sig_main):
    sig_main = sig_main.iloc[0]
    z = float(sig_main.get("shuffle_z_specificity", np.nan))
    sh_mean = float(sig_main.get("shuffle_specificity_mean", np.nan))
    pass_lite = bool(sig_main.get("pass_lite", False))
    pass_strong = bool(sig_main.get("pass_strong_z2", False))
else:
    z, sh_mean, pass_lite, pass_strong = np.nan, np.nan, False, False

# residual norm caveat
if "residual_to_delta_norm_ratio" in audit_out.columns:
    residual_ratio_mean = safe_mean(audit_out["residual_to_delta_norm_ratio"])
    residual_ratio_min = float(np.nanmin(audit_out["residual_to_delta_norm_ratio"]))
    residual_ratio_max = float(np.nanmax(audit_out["residual_to_delta_norm_ratio"]))
else:
    residual_ratio_mean = residual_ratio_min = residual_ratio_max = np.nan

residual_norm_caveat = bool(np.isfinite(residual_ratio_mean) and residual_ratio_mean < 1e-4)

# Dose monotonicity for main control
main_curve = spec[
    (spec["direction"] == MAIN_DIRECTION) &
    (spec["control"] == MAIN_CONTROL)
].sort_values("alpha")
y = main_curve["specificity"].values.astype(float)
x = main_curve["alpha"].values.astype(float)
monotonic_nondec = bool(np.all(np.diff(y) >= -1e-9)) if len(y) > 1 else False
slope, curve_r2 = linear_slope_r2(x, y)

# boundary proxy result
acc_boundary_corr = corr_df[corr_df["control"] == MAIN_CONTROL]
if len(acc_boundary_corr):
    corr_abs = float(acc_boundary_corr.iloc[0]["corr_absDeltaUbase_abs_shift"])
else:
    corr_abs = np.nan

# interpret
if pass_strong and not residual_norm_caveat:
    verdict_label = "PASS-Strong"
elif pass_lite or (np.isfinite(main_acc) and np.isfinite(rand) and main_acc > 0 and main_acc > rand):
    verdict_label = "PASS-Lite"
else:
    verdict_label = "FAIL"

distinct_from_velocity_position = bool(
    np.isfinite(main_acc) and
    np.isfinite(vel) and
    np.isfinite(pos) and
    main_acc > vel and
    main_acc > pos
)

raw_delta_failure = bool(np.isfinite(raw_acc) and raw_acc < 0 and np.isfinite(main_acc) and main_acc > 0)

verdict_rows = [{
    "audit": "ASA-5D Acceleration Control Decomposition",
    "direction": MAIN_DIRECTION,
    "alpha": MAIN_ALPHA,
    "verdict": verdict_label,
    "acceleration_residual_specificity": main_acc,
    "acceleration_delta_specificity": raw_acc,
    "acceleration_stable_specificity": stable_acc,
    "velocity_residual_specificity": vel,
    "position_residual_specificity": pos,
    "random_specificity": rand,
    "shuffle_mean": sh_mean,
    "shuffle_z": z,
    "pass_lite_from_asa5": pass_lite,
    "pass_strong_from_asa5": pass_strong,
    "main_gt_velocity_and_position": distinct_from_velocity_position,
    "raw_delta_failure": raw_delta_failure,
    "dose_monotonic_nondec": monotonic_nondec,
    "dose_slope": slope,
    "dose_r2": curve_r2,
    "residual_to_delta_norm_ratio_mean": residual_ratio_mean,
    "residual_to_delta_norm_ratio_min": residual_ratio_min,
    "residual_to_delta_norm_ratio_max": residual_ratio_max,
    "residual_norm_caveat": residual_norm_caveat,
    "corr_absDeltaUbase_abs_shift_for_main": corr_abs,
    "interpretation": (
        "Acceleration residual shows weak positive separability, but raw acceleration delta fails. "
        "Residual norm caveat indicates possible numerical amplification after shared-subspace removal. "
        "Treat as weak curvature-control candidate, not a confirmed independent control channel."
        if verdict_label == "PASS-Lite" else
        "Strong acceleration channel if z>2 and no residual norm caveat." if verdict_label == "PASS-Strong" else
        "No reliable acceleration steering evidence."
    )
}]

verdict_df = pd.DataFrame(verdict_rows)
verdict_df.to_csv(SAVE_DIR / "asa5d_verdict.csv", index=False, encoding="utf-8-sig")

# ============================================================
# SUMMARY
# ============================================================

summary_out = {
    "audit": "ASA-5D Acceleration Control Decomposition",
    "input_dir": str(ASA5_DIR),
    "main_direction": MAIN_DIRECTION,
    "main_alpha": MAIN_ALPHA,
    "verdict": verdict_label,
    "main_findings": {
        "acceleration_residual_specificity": main_acc,
        "acceleration_delta_specificity": raw_acc,
        "velocity_residual_specificity": vel,
        "position_residual_specificity": pos,
        "random_specificity": rand,
        "shuffle_z": z,
        "dose_monotonic_nondec": monotonic_nondec,
        "dose_slope": slope,
        "dose_r2": curve_r2,
        "residual_norm_caveat": residual_norm_caveat,
        "residual_to_delta_norm_ratio_mean": residual_ratio_mean,
        "corr_absDeltaUbase_abs_shift_for_main": corr_abs,
    },
    "output_files": {
        "control_ranking": "asa5d_control_ranking.csv",
        "alpha_profile": "asa5d_alpha_profile.csv",
        "target_nontarget_profile": "asa5d_target_nontarget_profile.csv",
        "sample_shift_table": "asa5d_sample_shift_table.csv",
        "mechanism_shift_profile": "asa5d_mechanism_shift_profile.csv",
        "inertia_proxy_profile": "asa5d_inertia_proxy_profile.csv",
        "inertia_shift_correlation": "asa5d_inertia_shift_correlation.csv",
        "delta_vs_residual_gap": "asa5d_delta_vs_residual_gap.csv",
        "shuffle_percentile": "asa5d_shuffle_percentile.csv",
        "verdict": "asa5d_verdict.csv",
    },
    "recommended_next": (
        "If ASA-5D remains PASS-Lite with residual-norm caveat, rerun ASA-5E with a safer residualization scheme: "
        "do not normalize near-zero residual vectors; instead use ridge/PLS acceleration directions or layerwise low-rank curvature templates. "
        "Also stratify by true inertia proxies from hidden trajectory if available."
    )
}

with open(SAVE_DIR / "asa5d_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_out, f, indent=2, ensure_ascii=False)

print("ASA-5D complete.")
print(json.dumps(summary_out["main_findings"], indent=2, ensure_ascii=False))
print(f"Outputs written to: {SAVE_DIR.resolve()}")
