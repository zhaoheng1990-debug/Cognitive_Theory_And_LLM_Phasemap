import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    roc_auc_score,
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold

# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path(r"C:\Windows\System32\phasemap3a4_outputs\phasemap3a4_enriched_predictions.csv")
SAVE_DIR = Path("./phasemap3b_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

R_LAYERS = list(range(20, 27))
PRED_LAYERS = [23, 24, 25, 26]

# ============================================================
# LOAD
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Cannot find input file: {INPUT_PATH.resolve()}")

df = pd.read_csv(INPUT_PATH)

required = [
    "cv", "feature_set", "condition", "phase_target",
] + [f"true_R_{l}" for l in R_LAYERS] + [f"pred_R_{l}" for l in R_LAYERS]

missing = [c for c in required if c not in df.columns]
if missing:
    print("Available columns:")
    print(df.columns.tolist())
    raise RuntimeError(f"Missing columns: {missing}")

df = df.copy()
df["phase_target"] = df["phase_target"].astype(str)
df["condition"] = df["condition"].astype(str)
df["feature_set"] = df["feature_set"].astype(str)
df["cv"] = df["cv"].astype(str)

if "model" not in df.columns:
    df["model"] = "unknown"
df["model"] = df["model"].astype(str)

GROUP_COL = "group_id" if "group_id" in df.columns else "condition"

# ============================================================
# DERIVED TRUE / PRED DYNAMICS
# ============================================================

for prefix in ["true", "pred"]:
    for a, b in zip(R_LAYERS[:-1], R_LAYERS[1:]):
        df[f"{prefix}_v_{a}_{b}"] = df[f"{prefix}_R_{b}"] - df[f"{prefix}_R_{a}"]

    # acceleration over consecutive velocity intervals
    for a, b, c in zip(R_LAYERS[:-2], R_LAYERS[1:-1], R_LAYERS[2:]):
        df[f"{prefix}_a_{a}_{b}_{c}"] = (
            df[f"{prefix}_v_{b}_{c}"] - df[f"{prefix}_v_{a}_{b}"]
        )

# local states at L22/L23/L24
# L22 state uses R20,R21,R22 or R21,R22,R23 depending what we want.
# We focus on initial rollout energy using true R20-R22.
df["true_v_20_21"] = df["true_R_21"] - df["true_R_20"]
df["true_v_21_22"] = df["true_R_22"] - df["true_R_21"]
df["true_a_20_21_22"] = df["true_v_21_22"] - df["true_v_20_21"]

df["E_v_init"] = df["true_v_21_22"] ** 2
df["E_a_init"] = df["true_a_20_21_22"] ** 2
df["E_va_init"] = df["E_v_init"] + df["E_a_init"]
df["E_abs_init"] = df["true_v_21_22"].abs() + df["true_a_20_21_22"].abs()
df["E_span_init"] = df[["true_R_20", "true_R_21", "true_R_22"]].max(axis=1) - df[["true_R_20", "true_R_21", "true_R_22"]].min(axis=1)
df["E_mixed_init"] = df["E_abs_init"] + df["E_span_init"]
df["E_quad_mixed_init"] = df["E_v_init"] + df["E_a_init"] + df["E_span_init"] ** 2

# Energies over true trajectory windows
for l in [22, 23, 24]:
    # window l-1,l,l+1
    r_prev = f"true_R_{l-1}"
    r_l = f"true_R_{l}"
    r_next = f"true_R_{l+1}"

    v_prev = df[r_l] - df[r_prev]
    v_l = df[r_next] - df[r_l]
    a_l = v_l - v_prev
    span_l = df[[r_prev, r_l, r_next]].max(axis=1) - df[[r_prev, r_l, r_next]].min(axis=1)

    df[f"E_v_L{l}"] = v_l ** 2
    df[f"E_a_L{l}"] = a_l ** 2
    df[f"E_span_L{l}"] = span_l
    df[f"E_abs_L{l}"] = v_l.abs() + a_l.abs()
    df[f"E_mixed_L{l}"] = v_l.abs() + a_l.abs() + span_l
    df[f"E_quad_L{l}"] = v_l ** 2 + a_l ** 2 + span_l ** 2

# Predicted trajectory energies
for l in [22, 23, 24]:
    r_prev = f"pred_R_{l-1}"
    r_l = f"pred_R_{l}"
    r_next = f"pred_R_{l+1}"

    v_prev = df[r_l] - df[r_prev]
    v_l = df[r_next] - df[r_l]
    a_l = v_l - v_prev
    span_l = df[[r_prev, r_l, r_next]].max(axis=1) - df[[r_prev, r_l, r_next]].min(axis=1)

    df[f"pred_E_mixed_L{l}"] = v_l.abs() + a_l.abs() + span_l
    df[f"pred_E_quad_L{l}"] = v_l ** 2 + a_l ** 2 + span_l ** 2

# Energy decay
df["true_E_decay_22_to_24"] = df["E_mixed_L22"] - df["E_mixed_L24"]
df["pred_E_decay_22_to_24"] = df["pred_E_mixed_L22"] - df["pred_E_mixed_L24"]
df["energy_decay_gap"] = df["pred_E_decay_22_to_24"] - df["true_E_decay_22_to_24"]

# Error variables
for l in R_LAYERS:
    df[f"err_R_{l}"] = df[f"pred_R_{l}"] - df[f"true_R_{l}"]
    df[f"abs_err_R_{l}"] = df[f"err_R_{l}"].abs()

df["final_error_R26"] = df["err_R_26"]
df["final_abs_error_R26"] = df["abs_err_R_26"]

df["overshoot_R26"] = df["pred_R_26"].abs() - df["true_R_26"].abs()
df["abs_overshoot_R26"] = df["overshoot_R26"].abs()
df["is_overshoot_R26"] = (df["overshoot_R26"] > 0).astype(int)

# Basin sign preservation
df["sign_match_R26"] = (np.sign(df["pred_R_26"]) == np.sign(df["true_R_26"])).astype(int)

# Oscillation indicators
for prefix in ["true", "pred"]:
    df[f"{prefix}_osc_23_25"] = (
        np.sign(df[f"{prefix}_v_22_23"]) != np.sign(df[f"{prefix}_v_23_24"])
    ).astype(int) | (
        np.sign(df[f"{prefix}_v_23_24"]) != np.sign(df[f"{prefix}_v_24_25"])
    ).astype(int)

    df[f"{prefix}_osc_24_26"] = (
        np.sign(df[f"{prefix}_v_23_24"]) != np.sign(df[f"{prefix}_v_24_25"])
    ).astype(int) | (
        np.sign(df[f"{prefix}_v_24_25"]) != np.sign(df[f"{prefix}_v_25_26"])
    ).astype(int)

df["oscillation_mismatch_23_25"] = (df["true_osc_23_25"] != df["pred_osc_23_25"]).astype(int)
df["oscillation_mismatch_24_26"] = (df["true_osc_24_26"] != df["pred_osc_24_26"]).astype(int)

# ============================================================
# FOCUS ON BEST EXISTING MODEL BY DEFAULT
# ============================================================

# If many model/feature/cv combinations exist, analyze all,
# but also mark best-known main candidate.
df["candidate_id"] = df["cv"] + "|" + df["model"] + "|" + df["feature_set"]

# ============================================================
# HELPERS
# ============================================================

ENERGY_FEATURES = [
    "E_v_init",
    "E_a_init",
    "E_va_init",
    "E_abs_init",
    "E_span_init",
    "E_mixed_init",
    "E_quad_mixed_init",
    "E_mixed_L22",
    "E_quad_L22",
    "true_E_decay_22_to_24",
    "pred_E_decay_22_to_24",
    "energy_decay_gap",
]

REG_TARGETS = [
    "final_abs_error_R26",
    "abs_overshoot_R26",
    "overshoot_R26",
]

CLS_TARGETS = [
    "is_overshoot_R26",
    "oscillation_mismatch_23_25",
    "oscillation_mismatch_24_26",
    "sign_match_R26",
]

def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def get_splits(data, y=None, group_mode=True, classification=False):
    if group_mode:
        groups = data[GROUP_COL].values
        if len(np.unique(groups)) >= 5:
            return list(GroupKFold(n_splits=5).split(data, groups=groups))

    if classification and y is not None and len(np.unique(y)) >= 2:
        return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(data, y))

    return list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(data))


def cv_reg(data, features, target, model_kind="ridge", group_mode=True):
    X = data[features].values.astype(float)
    y = data[target].values.astype(float)

    splits = get_splits(data, y=None, group_mode=group_mode, classification=False)

    pred = np.zeros(len(data), dtype=float)

    for tr, te in splits:
        if model_kind == "ridge":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ])
        elif model_kind == "rf":
            model = RandomForestRegressor(
                n_estimators=400,
                max_depth=5,
                min_samples_leaf=8,
                random_state=SEED,
                n_jobs=-1,
            )
        else:
            raise ValueError(model_kind)

        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])

    return {
        "r2": float(r2_score(y, pred)),
        "corr": safe_corr(y, pred),
        "mae": float(mean_absolute_error(y, pred)),
    }


def cv_cls(data, features, target, model_kind="lr", group_mode=True):
    X = data[features].values.astype(float)
    y = data[target].values.astype(int)

    if len(np.unique(y)) < 2:
        return None

    splits = get_splits(data, y=y, group_mode=group_mode, classification=True)
    pred = np.zeros(len(data), dtype=float)

    for tr, te in splits:
        if model_kind == "lr":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=5000)),
            ])
        elif model_kind == "rf":
            model = RandomForestClassifier(
                n_estimators=400,
                max_depth=5,
                min_samples_leaf=8,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
            )
        else:
            raise ValueError(model_kind)

        model.fit(X[tr], y[tr])
        pred[te] = model.predict_proba(X[te])[:, 1]

    label = (pred >= 0.5).astype(int)

    return {
        "auc": float(roc_auc_score(y, pred)),
        "acc": float(accuracy_score(y, label)),
        "f1": float(f1_score(y, label, zero_division=0)),
    }

# ============================================================
# CORRELATION AUDIT
# ============================================================

corr_rows = []

for candidate, sub in df.groupby("candidate_id"):
    cv, model, feature_set = candidate.split("|", 2)

    for e in ENERGY_FEATURES:
        for t in REG_TARGETS:
            corr_rows.append({
                "cv": cv,
                "model": model,
                "feature_set": feature_set,
                "energy_feature": e,
                "target": t,
                "corr": safe_corr(sub[e], sub[t]),
                "abs_corr": abs(safe_corr(sub[e], sub[t])),
            })

corr_df = pd.DataFrame(corr_rows).sort_values(
    ["target", "abs_corr"],
    ascending=[True, False],
)

corr_df.to_csv(SAVE_DIR / "phasemap3b_energy_error_correlations.csv", index=False)

# ============================================================
# HIGH/LOW ENERGY GROUP AUDIT
# ============================================================

group_rows = []

for candidate, sub in df.groupby("candidate_id"):
    cv, model, feature_set = candidate.split("|", 2)

    for e in ["E_mixed_init", "E_quad_mixed_init", "E_mixed_L22", "E_quad_L22"]:
        q_low = sub[e].quantile(0.33)
        q_high = sub[e].quantile(0.67)

        low = sub[sub[e] <= q_low]
        high = sub[sub[e] >= q_high]

        group_rows.append({
            "cv": cv,
            "model": model,
            "feature_set": feature_set,
            "energy_feature": e,
            "low_n": int(len(low)),
            "high_n": int(len(high)),
            "low_final_abs_error": float(low["final_abs_error_R26"].mean()),
            "high_final_abs_error": float(high["final_abs_error_R26"].mean()),
            "high_minus_low_final_abs_error": float(high["final_abs_error_R26"].mean() - low["final_abs_error_R26"].mean()),
            "low_overshoot_rate": float(low["is_overshoot_R26"].mean()),
            "high_overshoot_rate": float(high["is_overshoot_R26"].mean()),
            "high_minus_low_overshoot_rate": float(high["is_overshoot_R26"].mean() - low["is_overshoot_R26"].mean()),
            "low_true_osc_23_25": float(low["true_osc_23_25"].mean()),
            "high_true_osc_23_25": float(high["true_osc_23_25"].mean()),
            "high_minus_low_true_osc": float(high["true_osc_23_25"].mean() - low["true_osc_23_25"].mean()),
            "low_sign_match_R26": float(low["sign_match_R26"].mean()),
            "high_sign_match_R26": float(high["sign_match_R26"].mean()),
        })

energy_group_df = pd.DataFrame(group_rows).sort_values(
    "high_minus_low_final_abs_error",
    ascending=False,
)

energy_group_df.to_csv(SAVE_DIR / "phasemap3b_high_low_energy_groups.csv", index=False)

# ============================================================
# PREDICTIVE AUDIT
# ============================================================

FEATURE_SETS = {
    "E_init_basic": [
        "E_v_init",
        "E_a_init",
        "E_span_init",
    ],
    "E_init_mixed": [
        "E_v_init",
        "E_a_init",
        "E_abs_init",
        "E_span_init",
        "E_mixed_init",
        "E_quad_mixed_init",
    ],
    "E_L22": [
        "E_mixed_L22",
        "E_quad_L22",
    ],
    "E_decay": [
        "true_E_decay_22_to_24",
        "pred_E_decay_22_to_24",
        "energy_decay_gap",
    ],
    "E_all": ENERGY_FEATURES,
}

reg_rows = []
cls_rows = []

for candidate, sub in df.groupby("candidate_id"):
    cv0, model0, feature_set0 = candidate.split("|", 2)

    # avoid tiny subsets
    if len(sub) < 50:
        continue

    for group_mode in [False, True]:
        cv_name = "GroupKFold" if group_mode else "KFold"

        for fs_name, feats in FEATURE_SETS.items():
            for target in REG_TARGETS:
                for model_kind in ["ridge", "rf"]:
                    res = cv_reg(
                        sub,
                        feats,
                        target,
                        model_kind=model_kind,
                        group_mode=group_mode,
                    )

                    reg_rows.append({
                        "candidate_cv": cv0,
                        "candidate_model": model0,
                        "candidate_feature_set": feature_set0,
                        "audit_cv": cv_name,
                        "audit_model": model_kind,
                        "energy_feature_set": fs_name,
                        "target": target,
                        "features": ",".join(feats),
                        **res,
                    })

            for target in CLS_TARGETS:
                for model_kind in ["lr", "rf"]:
                    res = cv_cls(
                        sub,
                        feats,
                        target,
                        model_kind=model_kind,
                        group_mode=group_mode,
                    )
                    if res is None:
                        continue

                    cls_rows.append({
                        "candidate_cv": cv0,
                        "candidate_model": model0,
                        "candidate_feature_set": feature_set0,
                        "audit_cv": cv_name,
                        "audit_model": model_kind,
                        "energy_feature_set": fs_name,
                        "target": target,
                        "features": ",".join(feats),
                        **res,
                    })

reg_df = pd.DataFrame(reg_rows).sort_values(
    ["target", "r2"],
    ascending=[True, False],
)
cls_df = pd.DataFrame(cls_rows).sort_values(
    ["target", "auc"],
    ascending=[True, False],
)

reg_df.to_csv(SAVE_DIR / "phasemap3b_energy_predicts_error.csv", index=False)
cls_df.to_csv(SAVE_DIR / "phasemap3b_energy_predicts_oscillation.csv", index=False)

# ============================================================
# ENERGY DECAY SUMMARY
# ============================================================

decay_rows = []

for candidate, sub in df.groupby("candidate_id"):
    cv, model, feature_set = candidate.split("|", 2)

    decay_rows.append({
        "cv": cv,
        "model": model,
        "feature_set": feature_set,
        "n": int(len(sub)),
        "true_E_mixed_L22_mean": float(sub["E_mixed_L22"].mean()),
        "true_E_mixed_L23_mean": float(sub["E_mixed_L23"].mean()),
        "true_E_mixed_L24_mean": float(sub["E_mixed_L24"].mean()),
        "pred_E_mixed_L22_mean": float(sub["pred_E_mixed_L22"].mean()),
        "pred_E_mixed_L23_mean": float(sub["pred_E_mixed_L23"].mean()),
        "pred_E_mixed_L24_mean": float(sub["pred_E_mixed_L24"].mean()),
        "true_decay_22_to_24_mean": float(sub["true_E_decay_22_to_24"].mean()),
        "pred_decay_22_to_24_mean": float(sub["pred_E_decay_22_to_24"].mean()),
        "energy_decay_gap_mean": float(sub["energy_decay_gap"].mean()),
        "energy_decay_gap_corr_final_error": safe_corr(sub["energy_decay_gap"], sub["final_abs_error_R26"]),
    })

decay_df = pd.DataFrame(decay_rows).sort_values(
    "energy_decay_gap_corr_final_error",
    ascending=False,
)

decay_df.to_csv(SAVE_DIR / "phasemap3b_energy_decay_summary.csv", index=False)

# ============================================================
# PHASE / CONDITION SUMMARY
# ============================================================

phase_summary = df.groupby(["cv", "model", "feature_set", "phase_target"]).agg(
    n=("phase_target", "count"),
    final_abs_error_R26=("final_abs_error_R26", "mean"),
    overshoot_R26=("overshoot_R26", "mean"),
    overshoot_rate=("is_overshoot_R26", "mean"),
    sign_match_R26=("sign_match_R26", "mean"),
    true_osc_23_25=("true_osc_23_25", "mean"),
    pred_osc_23_25=("pred_osc_23_25", "mean"),
    oscillation_mismatch=("oscillation_mismatch_23_25", "mean"),
    E_mixed_init=("E_mixed_init", "mean"),
    E_mixed_L22=("E_mixed_L22", "mean"),
    true_decay=("true_E_decay_22_to_24", "mean"),
    pred_decay=("pred_E_decay_22_to_24", "mean"),
    decay_gap=("energy_decay_gap", "mean"),
).reset_index()

condition_summary = df.groupby(["cv", "model", "feature_set", "condition"]).agg(
    n=("condition", "count"),
    phase=("phase_target", lambda x: x.mode().iloc[0]),
    final_abs_error_R26=("final_abs_error_R26", "mean"),
    overshoot_R26=("overshoot_R26", "mean"),
    overshoot_rate=("is_overshoot_R26", "mean"),
    sign_match_R26=("sign_match_R26", "mean"),
    true_osc_23_25=("true_osc_23_25", "mean"),
    pred_osc_23_25=("pred_osc_23_25", "mean"),
    oscillation_mismatch=("oscillation_mismatch_23_25", "mean"),
    E_mixed_init=("E_mixed_init", "mean"),
    E_mixed_L22=("E_mixed_L22", "mean"),
    true_decay=("true_E_decay_22_to_24", "mean"),
    pred_decay=("pred_E_decay_22_to_24", "mean"),
    decay_gap=("energy_decay_gap", "mean"),
).reset_index()

phase_summary.to_csv(SAVE_DIR / "phasemap3b_energy_by_phase.csv", index=False)
condition_summary.to_csv(SAVE_DIR / "phasemap3b_energy_by_condition.csv", index=False)

# Save enriched table
df.to_csv(SAVE_DIR / "phasemap3b_enriched_energy_error.csv", index=False)

# ============================================================
# SUMMARY
# ============================================================

# focus on best candidate if it exists
preferred = df[
    (df["cv"] == "GroupKFold")
    & (df["model"].astype(str).str.contains("gbr", case=False, na=False))
    & (df["feature_set"] == "H_shape_layer")
]

if len(preferred) == 0:
    preferred = df.copy()

best_corrs = corr_df.head(20).to_dict(orient="records")
best_groups = energy_group_df.head(20).to_dict(orient="records")

def top_reg(target):
    return reg_df[reg_df["target"] == target].head(10).to_dict(orient="records")

def top_cls(target):
    return cls_df[cls_df["target"] == target].head(10).to_dict(orient="records")

summary = {
    "experiment": "PhaseMap-3B Residual Energy Oscillation Audit",
    "input_path": str(INPUT_PATH),
    "n_rows": int(len(df)),
    "core_question": "Does residual energy explain rollout over-shoot / oscillation / final error?",
    "preferred_candidate_n": int(len(preferred)),
    "preferred_candidate": {
        "cv": "GroupKFold",
        "model": "gbr",
        "feature_set": "H_shape_layer",
    },
    "preferred_candidate_metrics": {
        "corr_E_mixed_init_final_abs_error": safe_corr(preferred["E_mixed_init"], preferred["final_abs_error_R26"]),
        "corr_E_quad_L22_final_abs_error": safe_corr(preferred["E_quad_L22"], preferred["final_abs_error_R26"]),
        "corr_energy_decay_gap_final_abs_error": safe_corr(preferred["energy_decay_gap"], preferred["final_abs_error_R26"]),
        "corr_E_mixed_init_overshoot": safe_corr(preferred["E_mixed_init"], preferred["overshoot_R26"]),
        "high_low_E_mixed_init_final_error_gap": None,
    },
    "top_energy_error_correlations": best_corrs,
    "top_high_low_energy_group_effects": best_groups,
    "top_error_prediction": {
        "final_abs_error_R26": top_reg("final_abs_error_R26"),
        "abs_overshoot_R26": top_reg("abs_overshoot_R26"),
        "overshoot_R26": top_reg("overshoot_R26"),
    },
    "top_oscillation_prediction": {
        "is_overshoot_R26": top_cls("is_overshoot_R26"),
        "oscillation_mismatch_23_25": top_cls("oscillation_mismatch_23_25"),
        "oscillation_mismatch_24_26": top_cls("oscillation_mismatch_24_26"),
        "sign_match_R26": top_cls("sign_match_R26"),
    },
    "energy_decay_summary_top": decay_df.head(20).to_dict(orient="records"),
    "interpretation_rules": {
        "PASS_residual_energy": [
            "Residual energy correlates with final_abs_error or overshoot.",
            "High-energy group has substantially higher final error or overshoot rate.",
            "Energy decay gap predicts final error.",
            "Oscillation/overshoot classifiers achieve AUC > 0.75."
        ],
        "PASS_partial": [
            "Energy predicts final error but not oscillation.",
            "Interpret as residual energy drift, not full damping oscillation."
        ],
        "FAIL_energy": [
            "Energy variables do not predict error, overshoot, or oscillation.",
            "Missing mechanism is not residual energy."
        ],
        "next_if_pass": "PhaseMap-3C: add damping/attractor correction using E_res.",
        "next_if_partial": "PhaseMap-3C: residual correction model.",
        "next_if_fail": "Return to latent-state discovery."
    }
}

# fill high_low gap for preferred candidate
pref_key = (
    (energy_group_df["cv"] == "GroupKFold")
    & (energy_group_df["model"] == "gbr")
    & (energy_group_df["feature_set"] == "H_shape_layer")
    & (energy_group_df["energy_feature"] == "E_mixed_init")
)

if pref_key.any():
    summary["preferred_candidate_metrics"]["high_low_E_mixed_init_final_error_gap"] = float(
        energy_group_df[pref_key].iloc[0]["high_minus_low_final_abs_error"]
    )

with open(SAVE_DIR / "phasemap3b_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nSaved to: {SAVE_DIR.resolve()}")