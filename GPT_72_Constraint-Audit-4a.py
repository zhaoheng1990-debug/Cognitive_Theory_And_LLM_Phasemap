# ============================================================
# Constraint-Audit-4A
# Residual Subspace Decomposition
#
# Input:
#   Prefer:
#     ./constraint_audit3e_outputs/constraint_audit3e_feature_table_with_binding_semantics.csv
#   Fallback:
#     ./constraint_audit3d_outputs/constraint_audit3d_feature_table_with_skqo_geometry.csv
#     ./constraint_audit3c_outputs/constraint_audit3c_feature_table_with_geometry.csv
#     ./constraint_audit3b1_outputs/constraint_audit3b1_feature_table.csv
#     ./constraint_audit3b_outputs/constraint_audit3b_feature_table.csv
#
# Goal:
#   3B.1:
#       raw Delta-R_20:25 is the current minimal structural invariant.
#
#   3C:
#       Delta-R space forms SKQ prototype geometry:
#           S = stable
#           K = competition
#           Q = closure
#
#   3E:
#       the fourth O region is not one broad-binding axis;
#       it decomposes into at least:
#           E = evidence/source/equal-evidence binding
#           R = rule/override/exception binding
#           Q_modality = temporal/authority/update closure modality
#
#   4A asks:
#       If we subtract the coarse SKQ affine plane,
#       does the residual subspace reveal stable semantic directions?
#
# Core outputs:
#   ./constraint_audit4a_outputs/
#
# Main files:
#   constraint_audit4a_residual_pca_summary.csv
#   constraint_audit4a_family_residual_summary.csv
#   constraint_audit4a_diagnostic_group_summary.csv
#   constraint_audit4a_residual_axis_similarity.csv
#   constraint_audit4a_groupkfold_summary.csv
#   constraint_audit4a_lofo_summary.csv
# ============================================================

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

RANDOM_SEED = 42

INPUT_3E = Path("./constraint_audit3e_outputs/constraint_audit3e_feature_table_with_binding_semantics.csv")
INPUT_3D = Path("./constraint_audit3d_outputs/constraint_audit3d_feature_table_with_skqo_geometry.csv")
INPUT_3C = Path("./constraint_audit3c_outputs/constraint_audit3c_feature_table_with_geometry.csv")
INPUT_3B1 = Path("./constraint_audit3b1_outputs/constraint_audit3b1_feature_table.csv")
INPUT_3B = Path("./constraint_audit3b_outputs/constraint_audit3b_feature_table.csv")

SAVE_DIR = Path("./constraint_audit4a_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

TRACK_LAYERS = [20, 21, 22, 23, 24, 25]
DELTA_R_COLS = [f"DELTA_R_L{l}" for l in TRACK_LAYERS]

N_GROUP_SPLITS = 5
MAX_RESIDUAL_PCS = 4

# ============================================================
# FAMILY MAPS
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

MECHANISM_CLASSES = [
    "stable_or_preserved",
    "competition_conflict",
    "closure_rewrite",
]

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

SEMANTIC6_CLASSES = [
    "S_core",
    "S_shift",
    "K_competition",
    "E_evidence_binding",
    "Q_canonical_closure",
    "R_rule_binding",
]

# SKQ anchors define the coarse affine plane.
# We intentionally exclude:
#   - stable surface-shift
#   - high-risk evidence competition
#   - override / exception
# because these are what residual subspace should explain.
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

# Residual semantic groups to test.
RESIDUAL_AXIS_GROUPS = {
    "S_shift": ["stable_paraphrase", "stable_irrelevant"],

    "E_source": ["competition_source_claim"],
    "E_equal": ["competition_equal_evidence"],
    "E_evidence_binding": ["competition_source_claim", "competition_equal_evidence"],

    "R_override": ["closure_override"],
    "R_exception": ["closure_exception"],
    "R_rule_binding": ["closure_override", "closure_exception"],

    "Q_negation_update": ["closure_negation", "closure_update"],
    "Q_temporal": ["closure_temporal"],
    "Q_authority": ["closure_authority"],
    "Q_temporal_authority": ["closure_temporal", "closure_authority"],

    "K_low": ["competition_branch", "competition_direct"],
    "K_mid": ["competition_ambiguous"],

    "binding_candidate_all": [
        "competition_source_claim",
        "competition_equal_evidence",
        "closure_override",
        "closure_exception",
    ],
}

# ============================================================
# HELPERS
# ============================================================

def parse_bool_series(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def safe_name(x):
    return str(x).replace("/", "_").replace(" ", "_").replace("-", "_")


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
    """
    Project X onto affine subspace spanned by prototypes.

    For SKQ:
        proto[0] = S
        proto[1] = K
        proto[2] = Q

    x_hat = S + a(K-S) + b(Q-S)

    Returns:
        x_hat
        residual_vec
        residual_norm
        affine_coords over prototype vertices
    """
    p0 = proto[0]
    B = (proto[1:] - p0).T  # d x (k-1)

    x_hat_rows = []
    res_rows = []
    coord_rows = []
    norm_rows = []

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
        res_rows.append(r)
        coord_rows.append(coords)
        norm_rows.append(np.linalg.norm(r))

    return (
        np.vstack(x_hat_rows),
        np.vstack(res_rows),
        np.asarray(norm_rows, dtype=np.float64),
        np.vstack(coord_rows),
    )


def normalize_axis(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def axis_from_group(residual_vec, families, group_fams):
    mask = np.isin(families, group_fams)
    if mask.sum() == 0:
        return np.zeros(residual_vec.shape[1], dtype=np.float64)
    return normalize_axis(residual_vec[mask].mean(axis=0))


def projection_to_axis(X, axis):
    axis = normalize_axis(axis)
    if np.linalg.norm(axis) < 1e-12:
        return np.zeros(len(X), dtype=np.float64)
    return X @ axis


def safe_auc_binary(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return np.nan


def make_lr_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            max_iter=8000,
        )),
    ])


def fit_predict_multiclass(X_train, y_train, X_test):
    y_train = np.asarray(y_train)

    if len(np.unique(y_train)) < 2:
        return np.full(len(X_test), np.unique(y_train)[0], dtype=object)

    model = make_lr_model()
    model.fit(X_train, y_train)
    return model.predict(X_test)


def fit_predict_binary_prob(X_train, y_train, X_test):
    y_train = np.asarray(y_train).astype(int)

    if len(np.unique(y_train)) < 2:
        return np.full(len(X_test), float(y_train.mean()), dtype=np.float64)

    model = make_lr_model()
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def multiclass_metrics(y, pred, classes):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=classes, zero_division=0)),
    }


def binary_metrics(y, p):
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    pred = (p >= 0.5).astype(int)

    return {
        "auc": safe_auc_binary(y, p),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "pred_pos_rate": float(pred.mean()),
        "mean_prob": float(p.mean()),
    }


def f_score_by_label(df_in, feature_cols, label_col):
    rows = []
    y = df_in[label_col].values
    labels = sorted(pd.unique(y).tolist())

    for feat in feature_cols:
        x = df_in[feat].values.astype(np.float64)
        valid = np.isfinite(x)

        if valid.sum() == 0:
            continue

        x = x[valid]
        yv = y[valid]

        overall = x.mean()
        between = 0.0
        within = 0.0
        n_total = 0

        for lab in labels:
            vals = x[yv == lab]
            if len(vals) == 0:
                continue
            n = len(vals)
            mu = vals.mean()
            between += n * (mu - overall) ** 2
            within += ((vals - mu) ** 2).sum()
            n_total += n

        k = len(labels)
        if n_total <= k or k <= 1:
            continue

        f = (between / max(1, k - 1)) / ((within / max(1, n_total - k)) + 1e-12)

        rows.append({
            "label_col": label_col,
            "feature": feat,
            "f_score": float(f),
        })

    return pd.DataFrame(rows).sort_values("f_score", ascending=False)


# ============================================================
# LOAD DATA
# ============================================================

if INPUT_3E.exists():
    INPUT_FILE = INPUT_3E
elif INPUT_3D.exists():
    INPUT_FILE = INPUT_3D
elif INPUT_3C.exists():
    INPUT_FILE = INPUT_3C
elif INPUT_3B1.exists():
    INPUT_FILE = INPUT_3B1
elif INPUT_3B.exists():
    INPUT_FILE = INPUT_3B
else:
    raise FileNotFoundError("Cannot find 3E / 3D / 3C / 3B.1 / 3B feature table.")

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

print("\n============================================================")
print("Constraint-Audit-4A Input Summary")
print("============================================================\n")
print("Input:", INPUT_FILE)
print("Rows:", len(df))
print("Families:", df["family"].nunique())
print("Delta-R cols:", DELTA_R_COLS)

print("\nFamily summary:")
print(
    f"{'family':<32}"
    f"{'semantic6':<24}"
    f"{'diagnostic_group':<22}"
    f"{'mechanism':<24}"
    f"{'N':<6}"
    f"{'GenE':<10}"
)

for fam in FAMILY_ORDER:
    sub = df[df["family"] == fam]
    if len(sub) == 0:
        continue
    print(
        f"{fam:<32}"
        f"{SEMANTIC6_MAP[fam]:<24}"
        f"{DIAGNOSTIC_GROUP_MAP[fam]:<22}"
        f"{COARSE_MECHANISM_MAP[fam]:<24}"
        f"{len(sub):<6}"
        f"{sub['y_conflict'].mean():<10.4f}"
    )

# ============================================================
# FULL-DATA SKQ RESIDUAL DECOMPOSITION
# ============================================================

full_scaler = StandardScaler()
X_std = full_scaler.fit_transform(X_raw)

proto_skq, proto_info = build_anchor_prototypes(
    X_std,
    families,
    SKQ_ANCHOR_MAP,
    SKQ_CLASSES,
)

X_hat, residual_vec, residual_norm, skq_coords = affine_projection_to_prototypes(
    X_std,
    proto_skq,
)

df4 = df[[
    "idx",
    "family",
    "mechanism",
    "submechanism",
    "risk_regime",
    "semantic6_label",
    "diagnostic_group",
    "y_conflict",
]].copy()

for j, c in enumerate(DELTA_R_COLS):
    df4[c] = X_raw[:, j]
    df4[f"STD_{c}"] = X_std[:, j]
    df4[f"SKQ_HAT_{c}"] = X_hat[:, j]
    df4[f"SKQ_RESID_{c}"] = residual_vec[:, j]

df4["SKQ_residual_norm"] = residual_norm

for j, cls in enumerate(SKQ_CLASSES):
    df4[f"SKQ_affine_coord_{safe_name(cls)}"] = skq_coords[:, j]

# PCA on residual vectors.
n_pcs = min(MAX_RESIDUAL_PCS, residual_vec.shape[1], len(df4) - 1)
pca = PCA(n_components=n_pcs, random_state=RANDOM_SEED)
resid_pc = pca.fit_transform(residual_vec)

for i in range(n_pcs):
    df4[f"resid_pc{i+1}"] = resid_pc[:, i]

pca_summary_rows = []
for i in range(n_pcs):
    row = {
        "pc": f"resid_pc{i+1}",
        "explained_variance": float(pca.explained_variance_[i]),
        "explained_variance_ratio": float(pca.explained_variance_ratio_[i]),
    }
    for j, c in enumerate(DELTA_R_COLS):
        row[f"loading_{c}"] = float(pca.components_[i, j])
    pca_summary_rows.append(row)

pca_summary_df = pd.DataFrame(pca_summary_rows)

# Residual semantic axes.
axis_vectors = {}
axis_info_rows = []

for axis_name, fams in RESIDUAL_AXIS_GROUPS.items():
    axis = axis_from_group(residual_vec, families, fams)
    axis_vectors[axis_name] = axis

    row = {
        "axis_name": axis_name,
        "families": ",".join(fams),
        "n": int(np.isin(families, fams).sum()),
        "axis_norm": float(np.linalg.norm(axis)),
    }
    for j, c in enumerate(DELTA_R_COLS):
        row[f"axis_loading_{c}"] = float(axis[j])
    axis_info_rows.append(row)

axis_info_df = pd.DataFrame(axis_info_rows)

for axis_name, axis in axis_vectors.items():
    df4[f"proj_{axis_name}"] = projection_to_axis(residual_vec, axis)
    df4[f"abs_proj_{axis_name}"] = np.abs(df4[f"proj_{axis_name}"])

# Axis similarity matrix.
axis_names = list(axis_vectors.keys())
axis_pair_rows = []

for i, a in enumerate(axis_names):
    for j, b in enumerate(axis_names):
        if i >= j:
            continue
        va = axis_vectors[a]
        vb = axis_vectors[b]
        cos = float(
            np.dot(va, vb)
            / ((np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-12)
        )
        axis_pair_rows.append({
            "axis_1": a,
            "axis_2": b,
            "cosine_similarity": cos,
            "angle_degrees": float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))),
        })

axis_similarity_df = pd.DataFrame(axis_pair_rows).sort_values(
    "cosine_similarity",
    ascending=False,
)

# PC alignment with semantic axes.
pc_axis_rows = []
for i in range(n_pcs):
    pc_vec = pca.components_[i]
    for axis_name, axis in axis_vectors.items():
        cos = float(
            np.dot(pc_vec, axis)
            / ((np.linalg.norm(pc_vec) * np.linalg.norm(axis)) + 1e-12)
        )
        pc_axis_rows.append({
            "pc": f"resid_pc{i+1}",
            "axis_name": axis_name,
            "cosine_similarity": cos,
            "abs_cosine_similarity": abs(cos),
            "angle_degrees": float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))),
        })

pc_axis_alignment_df = pd.DataFrame(pc_axis_rows).sort_values(
    ["pc", "abs_cosine_similarity"],
    ascending=[True, False],
)

# ============================================================
# FAMILY AND GROUP SUMMARIES
# ============================================================

family_rows = []

for fam in FAMILY_ORDER:
    sub = df4[df4["family"] == fam]
    if len(sub) == 0:
        continue

    row = {
        "family": fam,
        "mechanism": COARSE_MECHANISM_MAP[fam],
        "submechanism": SUBMECHANISM_MAP[fam],
        "risk_regime": RISK_REGIME_MAP[fam],
        "semantic6_label": SEMANTIC6_MAP[fam],
        "diagnostic_group": DIAGNOSTIC_GROUP_MAP[fam],
        "n": int(len(sub)),
        "genE_rate": float(sub["y_conflict"].mean()),
        "mean_SKQ_residual_norm": float(sub["SKQ_residual_norm"].mean()),
        "std_SKQ_residual_norm": float(sub["SKQ_residual_norm"].std()),
    }

    for cls in SKQ_CLASSES:
        s = safe_name(cls)
        row[f"mean_SKQ_coord_{s}"] = float(sub[f"SKQ_affine_coord_{s}"].mean())

    for i in range(n_pcs):
        row[f"mean_resid_pc{i+1}"] = float(sub[f"resid_pc{i+1}"].mean())
        row[f"std_resid_pc{i+1}"] = float(sub[f"resid_pc{i+1}"].std())

    for axis_name in axis_names:
        row[f"mean_proj_{axis_name}"] = float(sub[f"proj_{axis_name}"].mean())
        row[f"mean_abs_proj_{axis_name}"] = float(sub[f"abs_proj_{axis_name}"].mean())

    family_rows.append(row)

family_summary_df = pd.DataFrame(family_rows)

family_summary_df["family_order"] = family_summary_df["family"].map({
    fam: i for i, fam in enumerate(FAMILY_ORDER)
})
family_summary_df = family_summary_df.sort_values("family_order").drop(columns=["family_order"])

group_rows = []

for group, sub in df4.groupby("diagnostic_group"):
    row = {
        "diagnostic_group": group,
        "n_samples": int(len(sub)),
        "n_families": int(sub["family"].nunique()),
        "genE_rate": float(sub["y_conflict"].mean()),
        "mean_SKQ_residual_norm": float(sub["SKQ_residual_norm"].mean()),
    }

    for i in range(n_pcs):
        row[f"mean_resid_pc{i+1}"] = float(sub[f"resid_pc{i+1}"].mean())
        row[f"std_resid_pc{i+1}"] = float(sub[f"resid_pc{i+1}"].std())

    for axis_name in axis_names:
        row[f"mean_proj_{axis_name}"] = float(sub[f"proj_{axis_name}"].mean())
        row[f"mean_abs_proj_{axis_name}"] = float(sub[f"abs_proj_{axis_name}"].mean())

    group_rows.append(row)

diagnostic_group_summary_df = pd.DataFrame(group_rows).sort_values(
    "mean_SKQ_residual_norm",
    ascending=False,
)

# F-score ranking: which residual PCs / projections separate labels?
candidate_cols = (
    ["SKQ_residual_norm"]
    + [f"resid_pc{i+1}" for i in range(n_pcs)]
    + [f"proj_{axis_name}" for axis_name in axis_names]
    + [f"abs_proj_{axis_name}" for axis_name in axis_names]
)

invariant_rankings = []
for label_col in ["mechanism", "submechanism", "risk_regime", "semantic6_label", "diagnostic_group"]:
    invariant_rankings.append(f_score_by_label(df4, candidate_cols, label_col))

invariant_ranking_df = pd.concat(invariant_rankings, axis=0, ignore_index=True)

# ============================================================
# TRAIN-ONLY RESIDUAL FEATURE BUILDERS FOR CV
# ============================================================

def build_fold_features(
    X_train_raw,
    X_test_raw,
    train_families,
    feature_mode,
    n_pc=4,
):
    """
    Avoid leakage:
      - StandardScaler fit on train only.
      - SKQ prototypes fit on train only.
      - residual PCA fit on train only.
      - transform test using train geometry.
    """
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train_raw)
    Xte = scaler.transform(X_test_raw)

    proto, _ = build_anchor_prototypes(
        Xtr,
        train_families,
        SKQ_ANCHOR_MAP,
        SKQ_CLASSES,
    )

    _, res_tr, norm_tr, coords_tr = affine_projection_to_prototypes(Xtr, proto)
    _, res_te, norm_te, coords_te = affine_projection_to_prototypes(Xte, proto)

    n_pc_eff = min(n_pc, res_tr.shape[1], len(Xtr) - 1)
    pca_fold = PCA(n_components=n_pc_eff, random_state=RANDOM_SEED)
    pc_tr = pca_fold.fit_transform(res_tr)
    pc_te = pca_fold.transform(res_te)

    # Semantic residual axes fit on train only.
    axis_train = {}
    for axis_name, fams in RESIDUAL_AXIS_GROUPS.items():
        axis_train[axis_name] = axis_from_group(res_tr, train_families, fams)

    proj_tr = []
    proj_te = []
    for axis_name in sorted(axis_train.keys()):
        axis = axis_train[axis_name]
        proj_tr.append(projection_to_axis(res_tr, axis))
        proj_te.append(projection_to_axis(res_te, axis))

    proj_tr = np.vstack(proj_tr).T if proj_tr else np.zeros((len(Xtr), 0))
    proj_te = np.vstack(proj_te).T if proj_te else np.zeros((len(Xte), 0))

    raw_tr = Xtr
    raw_te = Xte

    skq_tr = np.column_stack([coords_tr, norm_tr])
    skq_te = np.column_stack([coords_te, norm_te])

    resid_pc_tr = np.column_stack([norm_tr, pc_tr])
    resid_pc_te = np.column_stack([norm_te, pc_te])

    if feature_mode == "raw_deltaR":
        return raw_tr, raw_te

    if feature_mode == "SKQ_coords_only":
        return skq_tr, skq_te

    if feature_mode == "residual_pc_only":
        return resid_pc_tr, resid_pc_te

    if feature_mode == "SKQ_plus_residual_pc":
        return np.column_stack([skq_tr, pc_tr]), np.column_stack([skq_te, pc_te])

    if feature_mode == "semantic_axis_proj_only":
        return proj_tr, proj_te

    if feature_mode == "SKQ_plus_axis_proj":
        return np.column_stack([skq_tr, proj_tr]), np.column_stack([skq_te, proj_te])

    if feature_mode == "raw_plus_residual_pc":
        return np.column_stack([raw_tr, norm_tr, pc_tr]), np.column_stack([raw_te, norm_te, pc_te])

    if feature_mode == "raw_plus_SKQ_plus_residual_pc":
        return (
            np.column_stack([raw_tr, skq_tr, pc_tr]),
            np.column_stack([raw_te, skq_te, pc_te]),
        )

    raise ValueError(f"Unknown feature_mode: {feature_mode}")


FEATURE_MODES = [
    "raw_deltaR",
    "SKQ_coords_only",
    "residual_pc_only",
    "SKQ_plus_residual_pc",
    "semantic_axis_proj_only",
    "SKQ_plus_axis_proj",
    "raw_plus_residual_pc",
    "raw_plus_SKQ_plus_residual_pc",
]

# ============================================================
# GROUPKFOLD EVALUATION
# ============================================================

def run_groupkfold_eval(target_col, classes=None, is_binary=False):
    rows = []
    pred_rows = []

    y = df[target_col].values
    groups = df["idx"].values

    n_splits = min(N_GROUP_SPLITS, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    for feature_mode in FEATURE_MODES:
        if is_binary:
            p_all = np.zeros(len(df), dtype=np.float64)
        else:
            pred_all = np.empty(len(df), dtype=object)

        for train_idx, test_idx in gkf.split(X_raw, y, groups):
            X_train_raw = X_raw[train_idx]
            X_test_raw = X_raw[test_idx]
            train_fams = families[train_idx]

            Xtr, Xte = build_fold_features(
                X_train_raw,
                X_test_raw,
                train_fams,
                feature_mode=feature_mode,
                n_pc=MAX_RESIDUAL_PCS,
            )

            if is_binary:
                p = fit_predict_binary_prob(Xtr, y[train_idx], Xte)
                p_all[test_idx] = p
            else:
                pred = fit_predict_multiclass(Xtr, y[train_idx], Xte)
                pred_all[test_idx] = pred

        if is_binary:
            m = binary_metrics(y, p_all)
            row = {
                "eval": "GroupKFold",
                "target": target_col,
                "feature_mode": feature_mode,
            }
            row.update(m)
            rows.append(row)

            temp = df[["idx", "family", "mechanism", "submechanism", "risk_regime", "semantic6_label", "y_conflict"]].copy()
            temp["eval"] = "GroupKFold"
            temp["target"] = target_col
            temp["feature_mode"] = feature_mode
            temp["p_conflict"] = p_all
            pred_rows.append(temp)

        else:
            m = multiclass_metrics(y, pred_all, classes)
            row = {
                "eval": "GroupKFold",
                "target": target_col,
                "feature_mode": feature_mode,
            }
            row.update(m)
            rows.append(row)

            temp = df[["idx", "family", "mechanism", "submechanism", "risk_regime", "semantic6_label", "y_conflict"]].copy()
            temp["eval"] = "GroupKFold"
            temp["target"] = target_col
            temp["feature_mode"] = feature_mode
            temp["pred"] = pred_all
            pred_rows.append(temp)

    return pd.DataFrame(rows), pd.concat(pred_rows, axis=0, ignore_index=True)


# ============================================================
# LOFO EVALUATION
# ============================================================

def run_lofo_eval(target_col, classes=None, is_binary=False):
    rows = []
    pred_rows = []

    y = df[target_col].values
    logo = LeaveOneGroupOut()

    for feature_mode in FEATURE_MODES:
        if is_binary:
            p_all = np.zeros(len(df), dtype=np.float64)
        else:
            pred_all = np.empty(len(df), dtype=object)

        family_rows = []

        for train_idx, test_idx in logo.split(X_raw, y, groups=families):
            heldout = families[test_idx][0]

            X_train_raw = X_raw[train_idx]
            X_test_raw = X_raw[test_idx]
            train_fams = families[train_idx]

            Xtr, Xte = build_fold_features(
                X_train_raw,
                X_test_raw,
                train_fams,
                feature_mode=feature_mode,
                n_pc=MAX_RESIDUAL_PCS,
            )

            if is_binary:
                p = fit_predict_binary_prob(Xtr, y[train_idx], Xte)
                p_all[test_idx] = p
                fam_metric = binary_metrics(y[test_idx], p)

                fam_row = {
                    "eval": "LOFO_family",
                    "target": target_col,
                    "feature_mode": feature_mode,
                    "heldout_family": heldout,
                    "semantic6_label": SEMANTIC6_MAP[heldout],
                    "diagnostic_group": DIAGNOSTIC_GROUP_MAP[heldout],
                    "n_test": int(len(test_idx)),
                    "positive_rate": float(y[test_idx].mean()),
                }
                fam_row.update(fam_metric)
                family_rows.append(fam_row)

            else:
                pred = fit_predict_multiclass(Xtr, y[train_idx], Xte)
                pred_all[test_idx] = pred
                fam_metric = multiclass_metrics(y[test_idx], pred, classes)

                fam_row = {
                    "eval": "LOFO_family",
                    "target": target_col,
                    "feature_mode": feature_mode,
                    "heldout_family": heldout,
                    "true_label": y[test_idx][0],
                    "semantic6_label": SEMANTIC6_MAP[heldout],
                    "diagnostic_group": DIAGNOSTIC_GROUP_MAP[heldout],
                    "n_test": int(len(test_idx)),
                }
                fam_row.update(fam_metric)
                family_rows.append(fam_row)

        if is_binary:
            m = binary_metrics(y, p_all)
            row = {
                "eval": "LOFO",
                "target": target_col,
                "feature_mode": feature_mode,
            }
            row.update(m)
            rows.append(row)

            temp = df[["idx", "family", "mechanism", "submechanism", "risk_regime", "semantic6_label", "y_conflict"]].copy()
            temp["eval"] = "LOFO"
            temp["target"] = target_col
            temp["feature_mode"] = feature_mode
            temp["p_conflict"] = p_all
            pred_rows.append(temp)

        else:
            m = multiclass_metrics(y, pred_all, classes)
            row = {
                "eval": "LOFO",
                "target": target_col,
                "feature_mode": feature_mode,
            }
            row.update(m)
            rows.append(row)

            temp = df[["idx", "family", "mechanism", "submechanism", "risk_regime", "semantic6_label", "y_conflict"]].copy()
            temp["eval"] = "LOFO"
            temp["target"] = target_col
            temp["feature_mode"] = feature_mode
            temp["pred"] = pred_all
            pred_rows.append(temp)

        rows.extend(family_rows)

    return pd.DataFrame(rows), pd.concat(pred_rows, axis=0, ignore_index=True)


print("\nRunning GroupKFold evaluations...")

gkf_binary_df, gkf_binary_pred_df = run_groupkfold_eval(
    target_col="y_conflict",
    is_binary=True,
)

gkf_mech_df, gkf_mech_pred_df = run_groupkfold_eval(
    target_col="mechanism",
    classes=MECHANISM_CLASSES,
)

gkf_sub_df, gkf_sub_pred_df = run_groupkfold_eval(
    target_col="submechanism",
    classes=SUBMECHANISM_CLASSES,
)

gkf_risk_df, gkf_risk_pred_df = run_groupkfold_eval(
    target_col="risk_regime",
    classes=RISK_REGIME_CLASSES,
)

gkf_sem_df, gkf_sem_pred_df = run_groupkfold_eval(
    target_col="semantic6_label",
    classes=SEMANTIC6_CLASSES,
)

groupkfold_summary_df = pd.concat(
    [gkf_binary_df, gkf_mech_df, gkf_sub_df, gkf_risk_df, gkf_sem_df],
    axis=0,
    ignore_index=True,
)

groupkfold_pred_df = pd.concat(
    [gkf_binary_pred_df, gkf_mech_pred_df, gkf_sub_pred_df, gkf_risk_pred_df, gkf_sem_pred_df],
    axis=0,
    ignore_index=True,
)

print("Running LOFO evaluations...")

lofo_binary_df, lofo_binary_pred_df = run_lofo_eval(
    target_col="y_conflict",
    is_binary=True,
)

lofo_mech_df, lofo_mech_pred_df = run_lofo_eval(
    target_col="mechanism",
    classes=MECHANISM_CLASSES,
)

lofo_sub_df, lofo_sub_pred_df = run_lofo_eval(
    target_col="submechanism",
    classes=SUBMECHANISM_CLASSES,
)

lofo_risk_df, lofo_risk_pred_df = run_lofo_eval(
    target_col="risk_regime",
    classes=RISK_REGIME_CLASSES,
)

lofo_sem_df, lofo_sem_pred_df = run_lofo_eval(
    target_col="semantic6_label",
    classes=SEMANTIC6_CLASSES,
)

lofo_summary_df = pd.concat(
    [lofo_binary_df, lofo_mech_df, lofo_sub_df, lofo_risk_df, lofo_sem_df],
    axis=0,
    ignore_index=True,
)

lofo_pred_df = pd.concat(
    [lofo_binary_pred_df, lofo_mech_pred_df, lofo_sub_pred_df, lofo_risk_pred_df, lofo_sem_pred_df],
    axis=0,
    ignore_index=True,
)

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n\n============================================================")
print("4A Residual PCA Summary")
print("============================================================\n")
print(pca_summary_df.to_string(index=False))

print("\n\n============================================================")
print("4A PC ↔ Semantic Residual Axis Alignment")
print("============================================================\n")
print(pc_axis_alignment_df.head(60).to_string(index=False))

print("\n\n============================================================")
print("4A Residual Axis Similarity")
print("============================================================\n")
print(axis_similarity_df.head(60).to_string(index=False))

print("\n\n============================================================")
print("4A Family Residual Summary")
print("============================================================\n")

view_cols = [
    "family",
    "semantic6_label",
    "diagnostic_group",
    "genE_rate",
    "mean_SKQ_residual_norm",
    "mean_resid_pc1",
    "mean_resid_pc2",
    "mean_resid_pc3",
    "mean_resid_pc4",
    "mean_proj_E_evidence_binding",
    "mean_proj_R_rule_binding",
    "mean_proj_Q_temporal_authority",
    "mean_proj_S_shift",
]

existing_view_cols = [c for c in view_cols if c in family_summary_df.columns]
print(family_summary_df[existing_view_cols].to_string(index=False))

print("\n\n============================================================")
print("4A Diagnostic Group Summary")
print("============================================================\n")

existing_group_cols = [c for c in view_cols if c in diagnostic_group_summary_df.columns]
print(
    diagnostic_group_summary_df[
        ["diagnostic_group", "n_samples", "n_families", "genE_rate", "mean_SKQ_residual_norm"]
        + [c for c in existing_group_cols if c.startswith("mean_resid_pc") or c.startswith("mean_proj")]
    ].to_string(index=False)
)

print("\n\n============================================================")
print("4A Invariant Ranking: semantic6_label")
print("============================================================\n")
print(
    invariant_ranking_df[invariant_ranking_df["label_col"] == "semantic6_label"]
    .head(30)
    .to_string(index=False)
)

print("\n\n============================================================")
print("4A GroupKFold Summary")
print("============================================================\n")
print(
    groupkfold_summary_df
    .sort_values(by=["target", "macro_f1" if "macro_f1" in groupkfold_summary_df.columns else "accuracy"], ascending=[True, False])
    .to_string(index=False)
)

print("\n\n============================================================")
print("4A LOFO Summary")
print("============================================================\n")

lofo_overall_df = lofo_summary_df[lofo_summary_df["eval"] == "LOFO"].copy()

print(
    lofo_overall_df
    .sort_values(by=["target", "macro_f1" if "macro_f1" in lofo_overall_df.columns else "accuracy"], ascending=[True, False])
    .to_string(index=False)
)

print("\n\n============================================================")
print("4A LOFO Family Diagnostics: semantic6_label")
print("============================================================\n")

lofo_family_sem = lofo_summary_df[
    (lofo_summary_df["eval"] == "LOFO_family")
    & (lofo_summary_df["target"] == "semantic6_label")
].copy()

print(
    lofo_family_sem
    .sort_values(by=["feature_mode", "heldout_family"])
    .to_string(index=False)
)

# ============================================================
# SAVE OUTPUTS
# ============================================================

feature_table_path = SAVE_DIR / "constraint_audit4a_feature_table_with_residual_subspace.csv"
proto_info_path = SAVE_DIR / "constraint_audit4a_skq_anchor_info.csv"

pca_summary_path = SAVE_DIR / "constraint_audit4a_residual_pca_summary.csv"
axis_info_path = SAVE_DIR / "constraint_audit4a_residual_axis_info.csv"
axis_similarity_path = SAVE_DIR / "constraint_audit4a_residual_axis_similarity.csv"
pc_axis_alignment_path = SAVE_DIR / "constraint_audit4a_pc_axis_alignment.csv"

family_summary_path = SAVE_DIR / "constraint_audit4a_family_residual_summary.csv"
group_summary_path = SAVE_DIR / "constraint_audit4a_diagnostic_group_summary.csv"
invariant_ranking_path = SAVE_DIR / "constraint_audit4a_residual_invariant_ranking.csv"

groupkfold_summary_path = SAVE_DIR / "constraint_audit4a_groupkfold_summary.csv"
groupkfold_pred_path = SAVE_DIR / "constraint_audit4a_groupkfold_predictions.csv"

lofo_summary_path = SAVE_DIR / "constraint_audit4a_lofo_summary.csv"
lofo_pred_path = SAVE_DIR / "constraint_audit4a_lofo_predictions.csv"

df4.to_csv(feature_table_path, index=False)
proto_info.to_csv(proto_info_path, index=False)

pca_summary_df.to_csv(pca_summary_path, index=False)
axis_info_df.to_csv(axis_info_path, index=False)
axis_similarity_df.to_csv(axis_similarity_path, index=False)
pc_axis_alignment_df.to_csv(pc_axis_alignment_path, index=False)

family_summary_df.to_csv(family_summary_path, index=False)
diagnostic_group_summary_df.to_csv(group_summary_path, index=False)
invariant_ranking_df.to_csv(invariant_ranking_path, index=False)

groupkfold_summary_df.to_csv(groupkfold_summary_path, index=False)
groupkfold_pred_df.to_csv(groupkfold_pred_path, index=False)

lofo_summary_df.to_csv(lofo_summary_path, index=False)
lofo_pred_df.to_csv(lofo_pred_path, index=False)

print("\nSaved outputs:")
print(" ", feature_table_path)
print(" ", proto_info_path)
print(" ", pca_summary_path)
print(" ", axis_info_path)
print(" ", axis_similarity_path)
print(" ", pc_axis_alignment_path)
print(" ", family_summary_path)
print(" ", group_summary_path)
print(" ", invariant_ranking_path)
print(" ", groupkfold_summary_path)
print(" ", groupkfold_pred_path)
print(" ", lofo_summary_path)
print(" ", lofo_pred_path)

# ============================================================
# INTERPRETATION GUIDE
# ============================================================

print("\n\n============================================================")
print("Constraint-Audit-4A Interpretation Guide")
print("============================================================\n")

print("Core question:")
print("  After subtracting the coarse SKQ affine plane,")
print("  does the residual subspace reveal stable semantic directions?")
print()
print("Read first:")
print("  constraint_audit4a_residual_pca_summary.csv")
print("  constraint_audit4a_pc_axis_alignment.csv")
print("  constraint_audit4a_family_residual_summary.csv")
print("  constraint_audit4a_residual_axis_similarity.csv")
print("  constraint_audit4a_lofo_summary.csv")
print()
print("Strong positive result if:")
print("  1. resid_pc1/pc2 align strongly with E_evidence_binding or R_rule_binding axes.")
print("  2. E_source / E_equal have consistent PC/projection signs.")
print("  3. R_override / R_exception have consistent but distinct PC/projection signs.")
print("  4. residual_pc_only or SKQ_plus_residual_pc improves LOFO semantic6/submechanism/risk over SKQ_coords_only.")
print("  5. semantic_axis_proj_only is competitive with residual_pc_only.")
print()
print("Evidence that E/R are separable:")
print("  - E_evidence_binding and R_rule_binding projections separate E_* from R_* families.")
print("  - PC alignment shows E and R occupy different residual directions.")
print("  - SKQ_plus_residual_pc improves semantic6_label over SKQ_coords_only.")
print()
print("Evidence against residual-subspace theory:")
print("  - residual PCs explain little variance and do not align with semantic axes.")
print("  - residual_pc_only performs no better than SKQ_coords_only under LOFO.")
print("  - family residual summaries do not separate E/R/Q_modality/S_shift.")
print()
print("Theory update if positive:")
print("  DeltaR_20:25 = SKQ coarse geometry + residual semantic subspace.")
print("  C_struct = (SKQ coordinates, residual PCs).")
print("  E and R should be treated as residual directions, not merely prototype labels.")
print()
print("Done.")