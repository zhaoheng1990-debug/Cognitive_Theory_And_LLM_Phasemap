# ============================================================
# Constraint-Audit-4B
# Evidence–Rule Bipolar Axis Validation
#
# Input:
#   Prefer:
#     ./constraint_audit4a_outputs/constraint_audit4a_feature_table_with_residual_subspace.csv
#   Fallback:
#     ./constraint_audit3e_outputs/constraint_audit3e_feature_table_with_binding_semantics.csv
#     ./constraint_audit3d_outputs/constraint_audit3d_feature_table_with_skqo_geometry.csv
#     ./constraint_audit3b1_outputs/constraint_audit3b1_feature_table.csv
#     ./constraint_audit3b_outputs/constraint_audit3b_feature_table.csv
#
# Goal:
#   4A showed:
#       residual PC1 ≈ rule-binding positive pole
#       residual PC1 ≈ evidence-binding negative pole
#
#   4B asks:
#       Is this a stable train-only bipolar axis?
#
# Main construction:
#   1. Standardize raw Delta-R_20:25 using train split only.
#   2. Fit SKQ coarse affine plane using train split only.
#   3. Compute residuals from SKQ plane.
#   4. Construct B_ER axis:
#
#          B_ER = mean(residual_rule_binding)
#                 - mean(residual_evidence_binding)
#
#      Positive side = rule / override / exception.
#      Negative side = evidence / source / equal-evidence.
#
#   5. Project held-out samples:
#
#          z_ER = residual · B_ER
#
# Validation:
#   - LOFO family sign test:
#       evidence families should have z_ER < 0
#       rule families should have z_ER > 0
#
#   - E/R binary classification:
#       evidence = 0
#       rule = 1
#
#   - Compare direct z_ER against:
#       PC1
#       raw Delta-R logistic
#       SKQ coords logistic
#       SKQ + z_ER
#
# Outputs:
#   ./constraint_audit4b_outputs/
# ============================================================

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    precision_score,
    recall_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

RANDOM_SEED = 42

INPUT_4A = Path("./constraint_audit4a_outputs/constraint_audit4a_feature_table_with_residual_subspace.csv")
INPUT_3E = Path("./constraint_audit3e_outputs/constraint_audit3e_feature_table_with_binding_semantics.csv")
INPUT_3D = Path("./constraint_audit3d_outputs/constraint_audit3d_feature_table_with_skqo_geometry.csv")
INPUT_3B1 = Path("./constraint_audit3b1_outputs/constraint_audit3b1_feature_table.csv")
INPUT_3B = Path("./constraint_audit3b_outputs/constraint_audit3b_feature_table.csv")

SAVE_DIR = Path("./constraint_audit4b_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

TRACK_LAYERS = [20, 21, 22, 23, 24, 25]
DELTA_R_COLS = [f"DELTA_R_L{l}" for l in TRACK_LAYERS]

N_GROUP_SPLITS = 5

EVIDENCE_FAMILIES = [
    "competition_source_claim",
    "competition_equal_evidence",
]

RULE_FAMILIES = [
    "closure_override",
    "closure_exception",
]

BINDING_FAMILIES = EVIDENCE_FAMILIES + RULE_FAMILIES

# ============================================================
# MAPS
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

SEMANTIC6_MAP = {
    "stable_clean_basic": "S_core",
    "stable_redundant": "S_core",
    "stable_weak_note": "S_core",

    "stable_paraphrase": "S_shift",
    "stable_irrelevant": "S_shift",

    "competition_ambiguous": "K_competition",
    "competition_branch": "K_competition",
    "competition_direct": "K_competition",

    "competition_source_claim": "E_evidence_binding",
    "competition_equal_evidence": "E_evidence_binding",

    "closure_negation": "Q_canonical_closure",
    "closure_update": "Q_canonical_closure",
    "closure_temporal": "Q_canonical_closure",
    "closure_authority": "Q_canonical_closure",

    "closure_override": "R_rule_binding",
    "closure_exception": "R_rule_binding",
}

DIAGNOSTIC_GROUP_MAP = {
    "stable_clean_basic": "S_core",
    "stable_redundant": "S_core",
    "stable_weak_note": "S_core",

    "stable_paraphrase": "S_shift_boundary",
    "stable_irrelevant": "S_shift_boundary",

    "competition_branch": "K_low",
    "competition_direct": "K_low",
    "competition_ambiguous": "K_mid",

    "competition_source_claim": "E_source_binding",
    "competition_equal_evidence": "E_equal_binding",

    "closure_negation": "Q_negation",
    "closure_update": "Q_update",
    "closure_temporal": "Q_temporal",
    "closure_authority": "Q_authority",

    "closure_override": "R_override",
    "closure_exception": "R_exception",
}

SKQ_ANCHOR_MAP = {
    "stable_clean_basic": "S_stable",
    "stable_redundant": "S_stable",
    "stable_weak_note": "S_stable",

    "competition_ambiguous": "K_competition",
    "competition_branch": "K_competition",
    "competition_direct": "K_competition",

    "closure_negation": "Q_closure",
    "closure_update": "Q_closure",
    "closure_temporal": "Q_closure",
    "closure_authority": "Q_closure",
}

SKQ_CLASSES = [
    "S_stable",
    "K_competition",
    "Q_closure",
]

# ============================================================
# HELPERS
# ============================================================

def parse_bool_series(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def safe_name(x):
    return str(x).replace("/", "_").replace(" ", "_").replace("-", "_")


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def build_anchor_prototypes(X, families, anchor_map, classes):
    protos = []
    rows = []

    for cls in classes:
        anchor_fams = [fam for fam, lab in anchor_map.items() if lab == cls]
        mask = np.isin(families, anchor_fams)

        if mask.sum() == 0:
            protos.append(np.zeros(X.shape[1], dtype=np.float64))
            rows.append({
                "class": cls,
                "n_anchor": 0,
                "anchor_families": ",".join(anchor_fams),
            })
        else:
            protos.append(X[mask].mean(axis=0))
            rows.append({
                "class": cls,
                "n_anchor": int(mask.sum()),
                "anchor_families": ",".join(anchor_fams),
            })

    return np.vstack(protos).astype(np.float64), pd.DataFrame(rows)


def affine_projection_to_prototypes(X, proto):
    p0 = proto[0]
    B = (proto[1:] - p0).T

    x_hat_rows = []
    residual_rows = []
    residual_norm_rows = []
    coord_rows = []

    for x in X:
        y = x - p0

        if B.shape[1] == 0:
            coef = np.array([], dtype=np.float64)
            x_hat = p0.copy()
        else:
            coef, *_ = np.linalg.lstsq(B, y, rcond=None)
            x_hat = p0 + B @ coef

        coords = np.zeros(proto.shape[0], dtype=np.float64)
        coords[0] = 1.0 - coef.sum() if len(coef) else 1.0
        if len(coef):
            coords[1:] = coef

        r = x - x_hat

        x_hat_rows.append(x_hat)
        residual_rows.append(r)
        residual_norm_rows.append(np.linalg.norm(r))
        coord_rows.append(coords)

    return (
        np.vstack(x_hat_rows),
        np.vstack(residual_rows),
        np.asarray(residual_norm_rows, dtype=np.float64),
        np.vstack(coord_rows),
    )


def construct_er_axis(residual_vec, families):
    evidence_mask = np.isin(families, EVIDENCE_FAMILIES)
    rule_mask = np.isin(families, RULE_FAMILIES)

    if evidence_mask.sum() == 0 or rule_mask.sum() == 0:
        return np.zeros(residual_vec.shape[1], dtype=np.float64), {
            "n_evidence": int(evidence_mask.sum()),
            "n_rule": int(rule_mask.sum()),
            "axis_norm": 0.0,
        }

    evidence_mean = residual_vec[evidence_mask].mean(axis=0)
    rule_mean = residual_vec[rule_mask].mean(axis=0)

    axis_raw = rule_mean - evidence_mean
    axis = normalize(axis_raw)

    return axis, {
        "n_evidence": int(evidence_mask.sum()),
        "n_rule": int(rule_mask.sum()),
        "axis_norm": float(np.linalg.norm(axis_raw)),
    }


def construct_pc1_axis(residual_vec, er_axis=None):
    pca = PCA(n_components=1, random_state=RANDOM_SEED)
    pc = pca.fit(residual_vec).components_[0].astype(np.float64)

    if er_axis is not None and np.linalg.norm(er_axis) > 1e-12:
        if np.dot(pc, er_axis) < 0:
            pc = -pc

    return normalize(pc), float(pca.explained_variance_ratio_[0])


def project(X, axis):
    axis = normalize(axis)
    if np.linalg.norm(axis) < 1e-12:
        return np.zeros(len(X), dtype=np.float64)
    return X @ axis


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def safe_auc(y, score):
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=np.float64)
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y, score))
    except Exception:
        return np.nan


def binary_metrics_from_score(y, score, threshold=0.0):
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=np.float64)

    pred = (score >= threshold).astype(int)

    # For Brier, map axis score to probability with a train-free sigmoid.
    p = sigmoid(score)

    return {
        "auc": safe_auc(y, score),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "brier_sigmoid_score": float(brier_score_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "mean_score_rule": float(np.mean(score[y == 1])) if np.any(y == 1) else np.nan,
        "mean_score_evidence": float(np.mean(score[y == 0])) if np.any(y == 0) else np.nan,
        "score_gap_rule_minus_evidence": (
            float(np.mean(score[y == 1]) - np.mean(score[y == 0]))
            if np.any(y == 1) and np.any(y == 0)
            else np.nan
        ),
        "pred_rule_rate": float(pred.mean()),
    }


def make_lr_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            max_iter=8000,
        )),
    ])


def logistic_metrics(y, prob):
    y = np.asarray(y).astype(int)
    prob = np.clip(np.asarray(prob, dtype=np.float64), 1e-6, 1 - 1e-6)
    pred = (prob >= 0.5).astype(int)

    return {
        "auc": safe_auc(y, prob),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, prob)),
        "pred_rule_rate": float(pred.mean()),
        "mean_prob": float(prob.mean()),
    }


def fit_predict_logistic(X_train, y_train, X_test):
    y_train = np.asarray(y_train).astype(int)

    if len(np.unique(y_train)) < 2:
        return np.full(len(X_test), float(y_train.mean()), dtype=np.float64)

    model = make_lr_model()
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def build_fold_geometry(X_train_raw, X_test_raw, train_families):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train_raw)
    Xte = scaler.transform(X_test_raw)

    proto, proto_info = build_anchor_prototypes(
        Xtr,
        train_families,
        SKQ_ANCHOR_MAP,
        SKQ_CLASSES,
    )

    _, res_tr, norm_tr, coords_tr = affine_projection_to_prototypes(Xtr, proto)
    _, res_te, norm_te, coords_te = affine_projection_to_prototypes(Xte, proto)

    er_axis, er_info = construct_er_axis(res_tr, train_families)
    pc1_axis, pc1_var = construct_pc1_axis(res_tr, er_axis)

    z_er_tr = project(res_tr, er_axis)
    z_er_te = project(res_te, er_axis)

    z_pc1_tr = project(res_tr, pc1_axis)
    z_pc1_te = project(res_te, pc1_axis)

    return {
        "Xtr_std": Xtr,
        "Xte_std": Xte,

        "res_tr": res_tr,
        "res_te": res_te,

        "norm_tr": norm_tr,
        "norm_te": norm_te,

        "coords_tr": coords_tr,
        "coords_te": coords_te,

        "er_axis": er_axis,
        "er_info": er_info,

        "pc1_axis": pc1_axis,
        "pc1_var": pc1_var,

        "z_er_tr": z_er_tr,
        "z_er_te": z_er_te,

        "z_pc1_tr": z_pc1_tr,
        "z_pc1_te": z_pc1_te,

        "proto_info": proto_info,
    }


def feature_matrix_from_geometry(geom, mode, split):
    if split == "train":
        Xstd = geom["Xtr_std"]
        res = geom["res_tr"]
        norm = geom["norm_tr"]
        coords = geom["coords_tr"]
        z_er = geom["z_er_tr"]
        z_pc1 = geom["z_pc1_tr"]
    else:
        Xstd = geom["Xte_std"]
        res = geom["res_te"]
        norm = geom["norm_te"]
        coords = geom["coords_te"]
        z_er = geom["z_er_te"]
        z_pc1 = geom["z_pc1_te"]

    if mode == "z_ER_only":
        return z_er.reshape(-1, 1)

    if mode == "z_ER_abs":
        return np.abs(z_er).reshape(-1, 1)

    if mode == "z_ER_plus_norm":
        return np.column_stack([z_er, norm])

    if mode == "pc1_only":
        return z_pc1.reshape(-1, 1)

    if mode == "pc1_plus_norm":
        return np.column_stack([z_pc1, norm])

    if mode == "raw_deltaR":
        return Xstd

    if mode == "SKQ_coords_norm":
        return np.column_stack([coords, norm])

    if mode == "SKQ_plus_z_ER":
        return np.column_stack([coords, norm, z_er])

    if mode == "SKQ_plus_pc1":
        return np.column_stack([coords, norm, z_pc1])

    if mode == "raw_plus_z_ER":
        return np.column_stack([Xstd, z_er])

    if mode == "raw_plus_pc1":
        return np.column_stack([Xstd, z_pc1])

    if mode == "residual_vector":
        return res

    raise ValueError(f"Unknown feature mode: {mode}")


FEATURE_MODES = [
    "z_ER_only",
    "z_ER_plus_norm",
    "pc1_only",
    "pc1_plus_norm",
    "raw_deltaR",
    "SKQ_coords_norm",
    "SKQ_plus_z_ER",
    "SKQ_plus_pc1",
    "raw_plus_z_ER",
    "raw_plus_pc1",
    "residual_vector",
]

# ============================================================
# LOAD DATA
# ============================================================

if INPUT_4A.exists():
    INPUT_FILE = INPUT_4A
elif INPUT_3E.exists():
    INPUT_FILE = INPUT_3E
elif INPUT_3D.exists():
    INPUT_FILE = INPUT_3D
elif INPUT_3B1.exists():
    INPUT_FILE = INPUT_3B1
elif INPUT_3B.exists():
    INPUT_FILE = INPUT_3B
else:
    raise FileNotFoundError("Cannot find 4A / 3E / 3D / 3B.1 / 3B input table.")

df = pd.read_csv(INPUT_FILE)
df = df.loc[:, ~df.columns.duplicated()].copy()

for col in ["is_clean", "is_conflict", "is_other"]:
    if col in df.columns:
        df[col] = parse_bool_series(df[col])

if "y_conflict" not in df.columns:
    if "is_conflict" in df.columns:
        df["y_conflict"] = df["is_conflict"].astype(int)
    else:
        raise RuntimeError("Need y_conflict or is_conflict column.")

df["y_conflict"] = df["y_conflict"].astype(int)

if "is_clean" in df.columns and "is_conflict" in df.columns:
    df = df[(df["is_clean"] == True) | (df["is_conflict"] == True)].copy()

required = ["idx", "family"] + DELTA_R_COLS
missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError("Missing required columns: " + ", ".join(missing))

df["mechanism"] = df["family"].map(COARSE_MECHANISM_MAP)
df["submechanism"] = df["family"].map(SUBMECHANISM_MAP)
df["risk_regime"] = df["family"].map(RISK_REGIME_MAP)
df["semantic6_label"] = df["family"].map(SEMANTIC6_MAP)
df["diagnostic_group"] = df["family"].map(DIAGNOSTIC_GROUP_MAP)

df["er_label"] = np.nan
df.loc[df["family"].isin(EVIDENCE_FAMILIES), "er_label"] = 0
df.loc[df["family"].isin(RULE_FAMILIES), "er_label"] = 1

df["er_label_name"] = "other"
df.loc[df["family"].isin(EVIDENCE_FAMILIES), "er_label_name"] = "evidence"
df.loc[df["family"].isin(RULE_FAMILIES), "er_label_name"] = "rule"

if df["semantic6_label"].isna().any():
    bad = df[df["semantic6_label"].isna()]["family"].unique().tolist()
    raise RuntimeError(f"Unknown family values: {bad}")

df = df.reset_index(drop=True)

for c in DELTA_R_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df[DELTA_R_COLS] = (
    df[DELTA_R_COLS]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)

X_raw = df[DELTA_R_COLS].values.astype(np.float64)
families = df["family"].values

binding_mask = df["family"].isin(BINDING_FAMILIES).values
df_bind = df[binding_mask].copy().reset_index(drop=True)
X_bind_raw = df_bind[DELTA_R_COLS].values.astype(np.float64)
families_bind = df_bind["family"].values
y_er = df_bind["er_label"].astype(int).values
groups_bind = df_bind["idx"].values

print("\n============================================================")
print("Constraint-Audit-4B Input Summary")
print("============================================================\n")

print("Input:", INPUT_FILE)
print("Rows:", len(df))
print("Binding rows:", len(df_bind))
print("Families:", df["family"].nunique())
print("Binding families:", BINDING_FAMILIES)
print("Delta-R cols:", DELTA_R_COLS)

print("\nBinding family summary:")
print(
    f"{'family':<32}"
    f"{'er_label':<12}"
    f"{'diagnostic_group':<22}"
    f"{'N':<6}"
    f"{'GenE':<10}"
)

for fam in BINDING_FAMILIES:
    sub = df[df["family"] == fam]
    print(
        f"{fam:<32}"
        f"{df[df['family'] == fam]['er_label_name'].iloc[0]:<12}"
        f"{DIAGNOSTIC_GROUP_MAP[fam]:<22}"
        f"{len(sub):<6}"
        f"{sub['y_conflict'].mean():<10.4f}"
    )

# ============================================================
# FULL-DATA GEOMETRY FOR INTERPRETATION ONLY
# ============================================================

full_geom = build_fold_geometry(
    X_raw,
    X_raw,
    families,
)

z_er_full = full_geom["z_er_te"]
z_pc1_full = full_geom["z_pc1_te"]
res_norm_full = full_geom["norm_te"]
coords_full = full_geom["coords_te"]

df4b = df[[
    "idx",
    "family",
    "mechanism",
    "submechanism",
    "risk_regime",
    "semantic6_label",
    "diagnostic_group",
    "er_label",
    "er_label_name",
    "y_conflict",
]].copy()

for j, c in enumerate(DELTA_R_COLS):
    df4b[c] = X_raw[:, j]

df4b["z_ER_full"] = z_er_full
df4b["z_PC1_full"] = z_pc1_full
df4b["SKQ_residual_norm_full"] = res_norm_full

for j, cls in enumerate(SKQ_CLASSES):
    df4b[f"SKQ_coord_{safe_name(cls)}_full"] = coords_full[:, j]

# Family summary.
family_rows = []

for fam in FAMILY_ORDER:
    sub = df4b[df4b["family"] == fam]
    if len(sub) == 0:
        continue

    row = {
        "family": fam,
        "mechanism": COARSE_MECHANISM_MAP[fam],
        "submechanism": SUBMECHANISM_MAP[fam],
        "risk_regime": RISK_REGIME_MAP[fam],
        "semantic6_label": SEMANTIC6_MAP[fam],
        "diagnostic_group": DIAGNOSTIC_GROUP_MAP[fam],
        "er_label_name": sub["er_label_name"].iloc[0],
        "n": int(len(sub)),
        "genE_rate": float(sub["y_conflict"].mean()),
        "mean_z_ER": float(sub["z_ER_full"].mean()),
        "std_z_ER": float(sub["z_ER_full"].std()),
        "mean_z_PC1": float(sub["z_PC1_full"].mean()),
        "std_z_PC1": float(sub["z_PC1_full"].std()),
        "mean_residual_norm": float(sub["SKQ_residual_norm_full"].mean()),
        "std_residual_norm": float(sub["SKQ_residual_norm_full"].std()),
    }

    if fam in EVIDENCE_FAMILIES:
        row["expected_z_sign"] = "negative"
        row["sign_accuracy"] = float((sub["z_ER_full"] < 0).mean())
    elif fam in RULE_FAMILIES:
        row["expected_z_sign"] = "positive"
        row["sign_accuracy"] = float((sub["z_ER_full"] > 0).mean())
    else:
        row["expected_z_sign"] = "none"
        row["sign_accuracy"] = np.nan

    family_rows.append(row)

family_summary_df = pd.DataFrame(family_rows)

family_summary_df["family_order"] = family_summary_df["family"].map({
    fam: i for i, fam in enumerate(FAMILY_ORDER)
})
family_summary_df = family_summary_df.sort_values("family_order").drop(columns=["family_order"])

# Full-data axis metrics on E/R subset.
bind_full = df4b[df4b["family"].isin(BINDING_FAMILIES)].copy()
y_full_er = bind_full["er_label"].astype(int).values

full_axis_metrics = pd.DataFrame([
    {
        "eval": "full_data_interpretive_only",
        "score": "z_ER_full",
        **binary_metrics_from_score(y_full_er, bind_full["z_ER_full"].values),
    },
    {
        "eval": "full_data_interpretive_only",
        "score": "z_PC1_full",
        **binary_metrics_from_score(y_full_er, bind_full["z_PC1_full"].values),
    },
])

# ============================================================
# GROUPKFOLD E/R BINARY VALIDATION
# ============================================================

def run_groupkfold_er():
    rows = []
    pred_rows = []
    axis_rows = []

    n_splits = min(N_GROUP_SPLITS, len(np.unique(groups_bind)))
    gkf = GroupKFold(n_splits=n_splits)

    for mode in FEATURE_MODES:
        p_all = np.zeros(len(df_bind), dtype=np.float64)

        # For direct axis scores.
        score_all = np.zeros(len(df_bind), dtype=np.float64)

        for fold_id, (train_idx, test_idx) in enumerate(gkf.split(X_bind_raw, y_er, groups_bind)):
            X_train_raw = X_bind_raw[train_idx]
            X_test_raw = X_bind_raw[test_idx]
            train_fams = families_bind[train_idx]

            geom = build_fold_geometry(
                X_train_raw,
                X_test_raw,
                train_fams,
            )

            Xtr = feature_matrix_from_geometry(geom, mode, "train")
            Xte = feature_matrix_from_geometry(geom, mode, "test")

            # Direct axis mode can be judged by score sign without logistic.
            if mode == "z_ER_only":
                score = geom["z_er_te"]
                score_all[test_idx] = score
                p_all[test_idx] = sigmoid(score)
            elif mode == "pc1_only":
                score = geom["z_pc1_te"]
                score_all[test_idx] = score
                p_all[test_idx] = sigmoid(score)
            else:
                p = fit_predict_logistic(Xtr, y_er[train_idx], Xte)
                p_all[test_idx] = p
                score_all[test_idx] = p - 0.5

            train_z = geom["z_er_tr"]
            train_y = y_er[train_idx]

            axis_rows.append({
                "eval": "GroupKFold",
                "fold_id": fold_id,
                "feature_mode": mode,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "train_axis_n_evidence": geom["er_info"]["n_evidence"],
                "train_axis_n_rule": geom["er_info"]["n_rule"],
                "train_axis_norm": geom["er_info"]["axis_norm"],
                "train_pc1_variance_ratio": geom["pc1_var"],
                "train_mean_z_rule": float(train_z[train_y == 1].mean()),
                "train_mean_z_evidence": float(train_z[train_y == 0].mean()),
                "train_gap_rule_minus_evidence": float(train_z[train_y == 1].mean() - train_z[train_y == 0].mean()),
            })

        if mode in ["z_ER_only", "pc1_only"]:
            m = binary_metrics_from_score(y_er, score_all)
            row = {
                "eval": "GroupKFold",
                "feature_mode": mode,
                "decision": "direct_sign",
            }
            row.update(m)
        else:
            m = logistic_metrics(y_er, p_all)
            row = {
                "eval": "GroupKFold",
                "feature_mode": mode,
                "decision": "logistic",
            }
            row.update(m)

        rows.append(row)

        temp = df_bind[[
            "idx",
            "family",
            "diagnostic_group",
            "semantic6_label",
            "er_label",
            "er_label_name",
            "y_conflict",
        ]].copy()

        temp["eval"] = "GroupKFold"
        temp["feature_mode"] = mode
        temp["score"] = score_all
        temp["prob_rule"] = p_all
        pred_rows.append(temp)

    return (
        pd.DataFrame(rows),
        pd.concat(pred_rows, axis=0, ignore_index=True),
        pd.DataFrame(axis_rows),
    )


# ============================================================
# LOFO E/R BINARY VALIDATION
# ============================================================

def run_lofo_er():
    rows = []
    pred_rows = []
    family_rows = []
    axis_rows = []

    logo = LeaveOneGroupOut()

    for mode in FEATURE_MODES:
        p_all = np.zeros(len(df_bind), dtype=np.float64)
        score_all = np.zeros(len(df_bind), dtype=np.float64)

        for fold_id, (train_idx, test_idx) in enumerate(logo.split(X_bind_raw, y_er, groups=families_bind)):
            heldout_family = families_bind[test_idx][0]

            X_train_raw = X_bind_raw[train_idx]
            X_test_raw = X_bind_raw[test_idx]
            train_fams = families_bind[train_idx]

            geom = build_fold_geometry(
                X_train_raw,
                X_test_raw,
                train_fams,
            )

            Xtr = feature_matrix_from_geometry(geom, mode, "train")
            Xte = feature_matrix_from_geometry(geom, mode, "test")

            if mode == "z_ER_only":
                score = geom["z_er_te"]
                score_all[test_idx] = score
                p_all[test_idx] = sigmoid(score)
            elif mode == "pc1_only":
                score = geom["z_pc1_te"]
                score_all[test_idx] = score
                p_all[test_idx] = sigmoid(score)
            else:
                p = fit_predict_logistic(Xtr, y_er[train_idx], Xte)
                p_all[test_idx] = p
                score_all[test_idx] = p - 0.5

            train_z = geom["z_er_tr"]
            train_y = y_er[train_idx]

            axis_rows.append({
                "eval": "LOFO",
                "fold_id": fold_id,
                "feature_mode": mode,
                "heldout_family": heldout_family,
                "heldout_label": df_bind.iloc[test_idx]["er_label_name"].iloc[0],
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "train_axis_n_evidence": geom["er_info"]["n_evidence"],
                "train_axis_n_rule": geom["er_info"]["n_rule"],
                "train_axis_norm": geom["er_info"]["axis_norm"],
                "train_pc1_variance_ratio": geom["pc1_var"],
                "train_mean_z_rule": float(train_z[train_y == 1].mean()),
                "train_mean_z_evidence": float(train_z[train_y == 0].mean()),
                "train_gap_rule_minus_evidence": float(train_z[train_y == 1].mean() - train_z[train_y == 0].mean()),
            })

            # Family-level metrics.
            if mode in ["z_ER_only", "pc1_only"]:
                fam_metric = binary_metrics_from_score(y_er[test_idx], score_all[test_idx])
                family_decision = "direct_sign"
            else:
                fam_metric = logistic_metrics(y_er[test_idx], p_all[test_idx])
                family_decision = "logistic"

            family_row = {
                "eval": "LOFO_family",
                "feature_mode": mode,
                "decision": family_decision,
                "heldout_family": heldout_family,
                "heldout_label": df_bind.iloc[test_idx]["er_label_name"].iloc[0],
                "diagnostic_group": DIAGNOSTIC_GROUP_MAP[heldout_family],
                "n_test": int(len(test_idx)),
                "mean_score": float(score_all[test_idx].mean()),
                "std_score": float(score_all[test_idx].std()),
                "mean_prob_rule": float(p_all[test_idx].mean()),
            }
            family_row.update(fam_metric)
            family_rows.append(family_row)

        if mode in ["z_ER_only", "pc1_only"]:
            m = binary_metrics_from_score(y_er, score_all)
            row = {
                "eval": "LOFO",
                "feature_mode": mode,
                "decision": "direct_sign",
            }
            row.update(m)
        else:
            m = logistic_metrics(y_er, p_all)
            row = {
                "eval": "LOFO",
                "feature_mode": mode,
                "decision": "logistic",
            }
            row.update(m)

        rows.append(row)

        temp = df_bind[[
            "idx",
            "family",
            "diagnostic_group",
            "semantic6_label",
            "er_label",
            "er_label_name",
            "y_conflict",
        ]].copy()

        temp["eval"] = "LOFO"
        temp["feature_mode"] = mode
        temp["score"] = score_all
        temp["prob_rule"] = p_all
        pred_rows.append(temp)

    return (
        pd.DataFrame(rows),
        pd.concat(pred_rows, axis=0, ignore_index=True),
        pd.DataFrame(family_rows),
        pd.DataFrame(axis_rows),
    )


# ============================================================
# LOFO ALL-FAMILY PROJECTION DIAGNOSTICS
# ============================================================

def run_lofo_all_family_projection():
    """
    Train B_ER axis on all non-heldout families, then project every heldout family.
    This is not a classifier for all families; it shows where each family lies
    on the evidence-rule polarity axis.
    """
    rows = []
    pred_rows = []

    logo = LeaveOneGroupOut()

    dummy_y = np.zeros(len(df), dtype=int)

    for fold_id, (train_idx, test_idx) in enumerate(logo.split(X_raw, dummy_y, groups=families)):
        heldout_family = families[test_idx][0]

        geom = build_fold_geometry(
            X_raw[train_idx],
            X_raw[test_idx],
            families[train_idx],
        )

        z = geom["z_er_te"]
        pc1 = geom["z_pc1_te"]
        norm = geom["norm_te"]

        temp = df.iloc[test_idx][[
            "idx",
            "family",
            "mechanism",
            "submechanism",
            "risk_regime",
            "semantic6_label",
            "diagnostic_group",
            "er_label",
            "er_label_name",
            "y_conflict",
        ]].copy()

        temp["eval"] = "LOFO_all_family_projection"
        temp["z_ER_trainonly"] = z
        temp["z_PC1_trainonly"] = pc1
        temp["residual_norm_trainonly"] = norm

        pred_rows.append(temp)

        row = {
            "heldout_family": heldout_family,
            "mechanism": COARSE_MECHANISM_MAP[heldout_family],
            "submechanism": SUBMECHANISM_MAP[heldout_family],
            "risk_regime": RISK_REGIME_MAP[heldout_family],
            "semantic6_label": SEMANTIC6_MAP[heldout_family],
            "diagnostic_group": DIAGNOSTIC_GROUP_MAP[heldout_family],
            "er_label_name": temp["er_label_name"].iloc[0],
            "n_test": int(len(test_idx)),
            "mean_z_ER": float(z.mean()),
            "std_z_ER": float(z.std()),
            "median_z_ER": float(np.median(z)),
            "mean_z_PC1": float(pc1.mean()),
            "std_z_PC1": float(pc1.std()),
            "mean_residual_norm": float(norm.mean()),
            "train_axis_n_evidence": geom["er_info"]["n_evidence"],
            "train_axis_n_rule": geom["er_info"]["n_rule"],
            "train_axis_norm": geom["er_info"]["axis_norm"],
            "train_pc1_variance_ratio": geom["pc1_var"],
        }

        if heldout_family in EVIDENCE_FAMILIES:
            row["expected_sign"] = "negative"
            row["sign_accuracy"] = float((z < 0).mean())
        elif heldout_family in RULE_FAMILIES:
            row["expected_sign"] = "positive"
            row["sign_accuracy"] = float((z > 0).mean())
        else:
            row["expected_sign"] = "none"
            row["sign_accuracy"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows), pd.concat(pred_rows, axis=0, ignore_index=True)


print("\nRunning GroupKFold E/R validation...")
gkf_summary_df, gkf_pred_df, gkf_axis_df = run_groupkfold_er()

print("Running LOFO E/R validation...")
lofo_summary_df, lofo_pred_df, lofo_family_df, lofo_axis_df = run_lofo_er()

print("Running LOFO all-family projection diagnostics...")
all_family_proj_summary_df, all_family_proj_pred_df = run_lofo_all_family_projection()

# ============================================================
# AXIS STABILITY SUMMARY
# ============================================================

axis_stability_rows = []

combined_axis_df = pd.concat([gkf_axis_df, lofo_axis_df], axis=0, ignore_index=True)

for keys, sub in combined_axis_df.groupby(["eval", "feature_mode"]):
    eval_name, mode = keys

    axis_stability_rows.append({
        "eval": eval_name,
        "feature_mode": mode,
        "n_folds": int(len(sub)),
        "mean_axis_norm": float(sub["train_axis_norm"].mean()),
        "std_axis_norm": float(sub["train_axis_norm"].std()),
        "mean_train_gap_rule_minus_evidence": float(sub["train_gap_rule_minus_evidence"].mean()),
        "std_train_gap_rule_minus_evidence": float(sub["train_gap_rule_minus_evidence"].std()),
        "mean_pc1_variance_ratio": float(sub["train_pc1_variance_ratio"].mean()),
        "std_pc1_variance_ratio": float(sub["train_pc1_variance_ratio"].std()),
        "min_train_axis_n_evidence": int(sub["train_axis_n_evidence"].min()),
        "min_train_axis_n_rule": int(sub["train_axis_n_rule"].min()),
    })

axis_stability_df = pd.DataFrame(axis_stability_rows)

# Family sign summary for binding families only.
binding_family_sign_rows = []

for fam in BINDING_FAMILIES:
    sub = all_family_proj_summary_df[all_family_proj_summary_df["heldout_family"] == fam]
    if len(sub) == 0:
        continue

    row = sub.iloc[0].to_dict()
    binding_family_sign_rows.append(row)

binding_family_sign_df = pd.DataFrame(binding_family_sign_rows)

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n\n============================================================")
print("4B Full-Data Interpretive Axis Metrics")
print("============================================================\n")
print(full_axis_metrics.to_string(index=False))

print("\n\n============================================================")
print("4B Family Projection Summary: Full-Data Axis")
print("============================================================\n")

view_cols = [
    "family",
    "semantic6_label",
    "diagnostic_group",
    "er_label_name",
    "genE_rate",
    "mean_z_ER",
    "std_z_ER",
    "mean_z_PC1",
    "mean_residual_norm",
    "expected_z_sign",
    "sign_accuracy",
]

print(
    family_summary_df[view_cols]
    .to_string(index=False)
)

print("\n\n============================================================")
print("4B GroupKFold E/R Binary Summary")
print("============================================================\n")
print(
    gkf_summary_df
    .sort_values(by=["auc", "accuracy"], ascending=[False, False])
    .to_string(index=False)
)

print("\n\n============================================================")
print("4B LOFO E/R Binary Summary")
print("============================================================\n")
print(
    lofo_summary_df
    .sort_values(by=["auc", "accuracy"], ascending=[False, False])
    .to_string(index=False)
)

print("\n\n============================================================")
print("4B LOFO Binding-Family Sign Summary")
print("============================================================\n")
print(
    binding_family_sign_df[[
        "heldout_family",
        "er_label_name",
        "diagnostic_group",
        "mean_z_ER",
        "std_z_ER",
        "mean_z_PC1",
        "mean_residual_norm",
        "expected_sign",
        "sign_accuracy",
        "train_axis_norm",
        "train_pc1_variance_ratio",
    ]]
    .to_string(index=False)
)

print("\n\n============================================================")
print("4B All-Family Train-Only Projection Summary")
print("============================================================\n")

print(
    all_family_proj_summary_df[[
        "heldout_family",
        "semantic6_label",
        "diagnostic_group",
        "er_label_name",
        "mean_z_ER",
        "std_z_ER",
        "mean_z_PC1",
        "mean_residual_norm",
        "expected_sign",
        "sign_accuracy",
    ]]
    .to_string(index=False)
)

print("\n\n============================================================")
print("4B Axis Stability Summary")
print("============================================================\n")
print(axis_stability_df.to_string(index=False))

print("\n\n============================================================")
print("4B LOFO Family Metrics by Feature Mode")
print("============================================================\n")
print(
    lofo_family_df
    .sort_values(by=["feature_mode", "heldout_family"])
    .to_string(index=False)
)

# ============================================================
# SAVE OUTPUTS
# ============================================================

feature_table_path = SAVE_DIR / "constraint_audit4b_feature_table_with_ER_axis.csv"

full_axis_metrics_path = SAVE_DIR / "constraint_audit4b_full_axis_metrics.csv"
family_summary_path = SAVE_DIR / "constraint_audit4b_family_projection_summary.csv"

gkf_summary_path = SAVE_DIR / "constraint_audit4b_groupkfold_er_summary.csv"
gkf_pred_path = SAVE_DIR / "constraint_audit4b_groupkfold_er_predictions.csv"
gkf_axis_path = SAVE_DIR / "constraint_audit4b_groupkfold_axis_diagnostics.csv"

lofo_summary_path = SAVE_DIR / "constraint_audit4b_lofo_er_summary.csv"
lofo_pred_path = SAVE_DIR / "constraint_audit4b_lofo_er_predictions.csv"
lofo_family_path = SAVE_DIR / "constraint_audit4b_lofo_er_family_summary.csv"
lofo_axis_path = SAVE_DIR / "constraint_audit4b_lofo_axis_diagnostics.csv"

all_family_proj_summary_path = SAVE_DIR / "constraint_audit4b_lofo_all_family_projection_summary.csv"
all_family_proj_pred_path = SAVE_DIR / "constraint_audit4b_lofo_all_family_projection_predictions.csv"

axis_stability_path = SAVE_DIR / "constraint_audit4b_axis_stability_summary.csv"
binding_family_sign_path = SAVE_DIR / "constraint_audit4b_binding_family_sign_summary.csv"

df4b.to_csv(feature_table_path, index=False)

full_axis_metrics.to_csv(full_axis_metrics_path, index=False)
family_summary_df.to_csv(family_summary_path, index=False)

gkf_summary_df.to_csv(gkf_summary_path, index=False)
gkf_pred_df.to_csv(gkf_pred_path, index=False)
gkf_axis_df.to_csv(gkf_axis_path, index=False)

lofo_summary_df.to_csv(lofo_summary_path, index=False)
lofo_pred_df.to_csv(lofo_pred_path, index=False)
lofo_family_df.to_csv(lofo_family_path, index=False)
lofo_axis_df.to_csv(lofo_axis_path, index=False)

all_family_proj_summary_df.to_csv(all_family_proj_summary_path, index=False)
all_family_proj_pred_df.to_csv(all_family_proj_pred_path, index=False)

axis_stability_df.to_csv(axis_stability_path, index=False)
binding_family_sign_df.to_csv(binding_family_sign_path, index=False)

print("\nSaved outputs:")
print(" ", feature_table_path)
print(" ", full_axis_metrics_path)
print(" ", family_summary_path)
print(" ", gkf_summary_path)
print(" ", gkf_pred_path)
print(" ", gkf_axis_path)
print(" ", lofo_summary_path)
print(" ", lofo_pred_path)
print(" ", lofo_family_path)
print(" ", lofo_axis_path)
print(" ", all_family_proj_summary_path)
print(" ", all_family_proj_pred_path)
print(" ", axis_stability_path)
print(" ", binding_family_sign_path)

# ============================================================
# INTERPRETATION GUIDE
# ============================================================

print("\n\n============================================================")
print("Constraint-Audit-4B Interpretation Guide")
print("============================================================\n")

print("Core hypothesis:")
print("  B_ER is a bipolar residual axis:")
print("      negative side = evidence / source / equal-evidence binding")
print("      positive side = rule / override / exception binding")
print()
print("Strong positive result if:")
print("  1. z_ER_only has high LOFO AUC / accuracy on E/R binary classification.")
print("  2. held-out evidence families have mean_z_ER < 0.")
print("  3. held-out rule families have mean_z_ER > 0.")
print("  4. train_gap_rule_minus_evidence remains positive in every fold.")
print("  5. z_ER_only is close to or better than pc1_only.")
print()
print("Interpretation details:")
print("  If z_ER_only works better than pc1_only:")
print("    supervised semantic bipolar axis is more stable than unsupervised PC1.")
print()
print("  If pc1_only works similarly:")
print("    PC1 genuinely recovers the evidence-rule polarity without labels.")
print()
print("  If raw_deltaR beats z_ER on E/R but z_ER sign is stable:")
print("    raw trajectory remains the best predictive representation,")
print("    while B_ER remains a valid explanatory coordinate.")
print()
print("  If source_claim and equal_evidence both fall negative:")
print("    evidence-binding pole is stable.")
print()
print("  If override and exception both fall positive:")
print("    rule-binding pole is stable.")
print()
print("  If exception is weak or near zero:")
print("    exception is not pure rule-binding; it has Q_modality contribution.")
print()
print("Theory update if positive:")
print("  epsilon_perp_SKQ ≈ z_ER * B_ER + z_Qm * Q_modality + residual")
print("  B_ER is the dominant evidence-rule binding polarity coordinate.")
print()
print("Done.")