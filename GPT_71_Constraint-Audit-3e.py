# ============================================================
# Constraint-Audit-3E
# Binding-Axis Semantic Dissection
#
# Input:
#   Prefer:
#     ./constraint_audit3d_outputs/constraint_audit3d_feature_table_with_skqo_geometry.csv
#   Fallback:
#     ./constraint_audit3c_outputs/constraint_audit3c_feature_table_with_geometry.csv
#     ./constraint_audit3b1_outputs/constraint_audit3b1_feature_table.csv
#     ./constraint_audit3b_outputs/constraint_audit3b_feature_table.csv
#
# Goal:
#   3D showed that the fourth prototype O attracts:
#       - closure_override
#       - closure_exception
#       - competition_equal_evidence
#       - competition_source_claim
#       - part of temporal / authority / canonical closure
#
#   Therefore 3E asks:
#
#       Is O an override-specific axis?
#       Or is it a broader evidence/rule/source/authority binding axis?
#
# Core tests:
#   1. Build semantic prototypes:
#        S_core
#        S_shift
#        K_competition
#        E_evidence_binding
#        Q_canonical_closure
#        R_rule_binding
#
#   2. Compare prototype geometries:
#        SKQ
#        SKQ + R_rule
#        SKQ + E_evidence
#        SKQ + R_rule + E_evidence
#        SKQ + B_broad
#        Semantic6
#
#   3. Residual-axis dissection:
#        Remove SKQ plane.
#        Inspect residual directions:
#           R_rule_binding residual axis
#           E_evidence_binding residual axis
#           Q_temporal_authority residual axis
#           S_shift residual axis
#
#   4. Decide whether:
#        R_rule and E_evidence are the same broad binding axis
#        or separable semantic subaxes.
#
# Outputs:
#   ./constraint_audit3e_outputs/
# ============================================================

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

RANDOM_SEED = 42

INPUT_3D = Path("./constraint_audit3d_outputs/constraint_audit3d_feature_table_with_skqo_geometry.csv")
INPUT_3C = Path("./constraint_audit3c_outputs/constraint_audit3c_feature_table_with_geometry.csv")
INPUT_3B1 = Path("./constraint_audit3b1_outputs/constraint_audit3b1_feature_table.csv")
INPUT_3B = Path("./constraint_audit3b_outputs/constraint_audit3b_feature_table.csv")

SAVE_DIR = Path("./constraint_audit3e_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

TRACK_LAYERS = [20, 21, 22, 23, 24, 25]
DELTA_R_COLS = [f"DELTA_R_L{l}" for l in TRACK_LAYERS]

N_GROUP_SPLITS = 5
SOFTMAX_TEMPERATURE = 1.0

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

AXIS3_MAP = {
    "stable_clean_basic": "S_stable",
    "stable_paraphrase": "S_stable",
    "stable_redundant": "S_stable",
    "stable_irrelevant": "S_stable",
    "stable_weak_note": "S_stable",

    "competition_ambiguous": "K_competition",
    "competition_branch": "K_competition",
    "competition_direct": "K_competition",
    "competition_source_claim": "K_competition",
    "competition_equal_evidence": "K_competition",

    "closure_negation": "Q_closure",
    "closure_update": "Q_closure",
    "closure_override": "Q_closure",
    "closure_temporal": "Q_closure",
    "closure_authority": "Q_closure",
    "closure_exception": "Q_closure",
}

AXIS4_MAP = {
    "stable_clean_basic": "S_stable",
    "stable_paraphrase": "S_stable",
    "stable_redundant": "S_stable",
    "stable_irrelevant": "S_stable",
    "stable_weak_note": "S_stable",

    "competition_ambiguous": "K_competition",
    "competition_branch": "K_competition",
    "competition_direct": "K_competition",
    "competition_source_claim": "K_competition",
    "competition_equal_evidence": "K_competition",

    "closure_negation": "Q_closure",
    "closure_update": "Q_closure",
    "closure_temporal": "Q_closure",
    "closure_authority": "Q_closure",

    "closure_override": "O_rule_binding",
    "closure_exception": "O_rule_binding",
}

# 3E semantic dissection labels.
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

SEMANTIC6_CLASSES = [
    "S_core",
    "S_shift",
    "K_competition",
    "E_evidence_binding",
    "Q_canonical_closure",
    "R_rule_binding",
]

AXIS3_CLASSES = [
    "S_stable",
    "K_competition",
    "Q_closure",
]

AXIS4_CLASSES = [
    "S_stable",
    "K_competition",
    "Q_closure",
    "O_rule_binding",
]

# Fine diagnostic groups used only for interpretation.
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

# Anchor maps for prototype-set comparison.
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

SKQR_ANCHOR_MAP = {
    **SKQ_ANCHOR_MAP,
    "closure_override": "R_rule_binding",
    "closure_exception": "R_rule_binding",
}

SKQR_CLASSES = [
    "S_stable",
    "K_competition",
    "Q_closure",
    "R_rule_binding",
]

SKQE_ANCHOR_MAP = {
    **SKQ_ANCHOR_MAP,
    "competition_source_claim": "E_evidence_binding",
    "competition_equal_evidence": "E_evidence_binding",
}

SKQE_CLASSES = [
    "S_stable",
    "K_competition",
    "Q_closure",
    "E_evidence_binding",
]

SKQRE_ANCHOR_MAP = {
    **SKQ_ANCHOR_MAP,
    "competition_source_claim": "E_evidence_binding",
    "competition_equal_evidence": "E_evidence_binding",
    "closure_override": "R_rule_binding",
    "closure_exception": "R_rule_binding",
}

SKQRE_CLASSES = [
    "S_stable",
    "K_competition",
    "Q_closure",
    "E_evidence_binding",
    "R_rule_binding",
]

# Broad binding prototype: evidence + rule binding together.
SKQB_ANCHOR_MAP = {
    **SKQ_ANCHOR_MAP,
    "competition_source_claim": "B_broad_binding",
    "competition_equal_evidence": "B_broad_binding",
    "closure_override": "B_broad_binding",
    "closure_exception": "B_broad_binding",
}

SKQB_CLASSES = [
    "S_stable",
    "K_competition",
    "Q_closure",
    "B_broad_binding",
]

SEMANTIC6_ANCHOR_MAP = {
    fam: SEMANTIC6_MAP[fam]
    for fam in FAMILY_ORDER
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


def softmax_neg_distance(dist, temperature=1.0):
    dist = np.asarray(dist, dtype=np.float64)
    z = -dist / max(temperature, 1e-8)
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / (ez.sum(axis=1, keepdims=True) + 1e-12)


def entropy_rows(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    h = -(p * np.log(p)).sum(axis=1)
    return h / np.log(p.shape[1])


def euclidean_dist_matrix(X, P):
    diff = X[:, None, :] - P[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def cosine_sim_matrix(X, P):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    return Xn @ Pn.T


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

    return np.vstack(protos), pd.DataFrame(rows)


def nearest_proto_predict(X, proto, classes, metric="euclidean"):
    if metric == "euclidean":
        dist = euclidean_dist_matrix(X, proto)
        idx = np.argmin(dist, axis=1)
        weights = softmax_neg_distance(dist, temperature=SOFTMAX_TEMPERATURE)
        score = -dist
    elif metric == "cosine":
        sim = cosine_sim_matrix(X, proto)
        idx = np.argmax(sim, axis=1)
        dist = 1.0 - sim
        weights = softmax_neg_distance(dist, temperature=SOFTMAX_TEMPERATURE)
        score = sim
    else:
        raise ValueError(f"Unknown metric: {metric}")

    pred = np.array(classes, dtype=object)[idx]
    return pred, dist, weights, score


def prototype_feature_frame(X, proto, classes, prefix):
    dist = euclidean_dist_matrix(X, proto)
    sim = cosine_sim_matrix(X, proto)
    weights = softmax_neg_distance(dist, temperature=SOFTMAX_TEMPERATURE)

    out = pd.DataFrame(index=np.arange(len(X)))

    for j, cls in enumerate(classes):
        s = safe_name(cls)
        out[f"{prefix}_dist_to_{s}"] = dist[:, j]
        out[f"{prefix}_cos_to_{s}"] = sim[:, j]
        out[f"{prefix}_mix_w_{s}"] = weights[:, j]

    order = np.argsort(dist, axis=1)
    best = order[:, 0]
    second = order[:, 1]

    out[f"{prefix}_nearest_proto"] = np.array(classes, dtype=object)[best]
    out[f"{prefix}_nearest_dist"] = dist[np.arange(len(X)), best]
    out[f"{prefix}_second_dist"] = dist[np.arange(len(X)), second]
    out[f"{prefix}_dist_margin_second_minus_first"] = (
        out[f"{prefix}_second_dist"] - out[f"{prefix}_nearest_dist"]
    )

    w_order = np.argsort(-weights, axis=1)
    out[f"{prefix}_top_mix_class"] = np.array(classes, dtype=object)[w_order[:, 0]]
    out[f"{prefix}_top_mix_weight"] = weights[np.arange(len(X)), w_order[:, 0]]
    out[f"{prefix}_second_mix_weight"] = weights[np.arange(len(X)), w_order[:, 1]]
    out[f"{prefix}_mix_margin_top_minus_second"] = (
        out[f"{prefix}_top_mix_weight"] - out[f"{prefix}_second_mix_weight"]
    )
    out[f"{prefix}_mix_entropy"] = entropy_rows(weights)

    return out


def affine_projection_residuals(X, proto):
    """
    Project X onto the affine subspace spanned by prototypes.
    Returns:
        residual_norm: n
        residual_vec: n x d
        coords: n x k affine coordinates
    """
    p0 = proto[0]
    B = (proto[1:] - p0).T

    residuals = []
    residual_vecs = []
    coords = []

    for x in X:
        y = x - p0

        if B.shape[1] == 0:
            coef = np.array([], dtype=np.float64)
            x_hat = p0.copy()
        else:
            coef, *_ = np.linalg.lstsq(B, y, rcond=None)
            x_hat = p0 + B @ coef

        w = np.zeros(proto.shape[0], dtype=np.float64)
        w[0] = 1.0 - coef.sum() if len(coef) else 1.0
        if len(coef):
            w[1:] = coef

        r = x - x_hat

        residuals.append(np.linalg.norm(r))
        residual_vecs.append(r)
        coords.append(w)

    return (
        np.asarray(residuals, dtype=np.float64),
        np.vstack(residual_vecs).astype(np.float64),
        np.vstack(coords).astype(np.float64),
    )


def residual_axis_from_group(residual_vecs, mask):
    if mask.sum() == 0:
        return np.zeros(residual_vecs.shape[1], dtype=np.float64)

    v = residual_vecs[mask].mean(axis=0)
    n = np.linalg.norm(v)

    if n < 1e-12:
        return v

    return v / n


def projection_to_axis(X, axis):
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)

    if n < 1e-12:
        return np.zeros(len(X), dtype=np.float64)

    a = axis / n
    return X @ a


def summarize_nearest(y_true, y_pred, classes, target_name, proto_set, metric):
    return {
        "target": target_name,
        "prototype_set": proto_set,
        "metric": metric,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=classes, zero_division=0)),
    }


def make_lr_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            max_iter=6000,
        )),
    ])


def groupkfold_logistic_eval(df, X, target_col, classes, feature_set):
    y = df[target_col].values
    groups = df["idx"].values

    n_splits = min(N_GROUP_SPLITS, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    pred = np.empty(len(df), dtype=object)

    for train_idx, test_idx in gkf.split(X, y, groups):
        model = make_lr_model()
        model.fit(X[train_idx], y[train_idx])
        pred[test_idx] = model.predict(X[test_idx])

    return {
        "feature_set": feature_set,
        "target": target_col,
        "metric": "GroupKFold_logistic",
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=classes, zero_division=0)),
    }


# ============================================================
# LOAD DATA
# ============================================================

if INPUT_3D.exists():
    INPUT_FILE = INPUT_3D
elif INPUT_3C.exists():
    INPUT_FILE = INPUT_3C
elif INPUT_3B1.exists():
    INPUT_FILE = INPUT_3B1
elif INPUT_3B.exists():
    INPUT_FILE = INPUT_3B
else:
    raise FileNotFoundError(
        "Cannot find 3D / 3C / 3B.1 / 3B feature table."
    )

df = pd.read_csv(INPUT_FILE)

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
df["axis3_label"] = df["family"].map(AXIS3_MAP)
df["axis4_label"] = df["family"].map(AXIS4_MAP)
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

scaler = StandardScaler()
X_std = scaler.fit_transform(X_raw)

print("\n============================================================")
print("Constraint-Audit-3E Input Summary")
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
# BUILD FULL-DATA PROTOTYPE SETS
# ============================================================

PROTO_SPECS = [
    ("SKQ", SKQ_ANCHOR_MAP, SKQ_CLASSES, "axis3_label"),
    ("SKQR_rule", SKQR_ANCHOR_MAP, SKQR_CLASSES, None),
    ("SKQE_evidence", SKQE_ANCHOR_MAP, SKQE_CLASSES, None),
    ("SKQRE_rule_evidence", SKQRE_ANCHOR_MAP, SKQRE_CLASSES, None),
    ("SKQB_broad_binding", SKQB_ANCHOR_MAP, SKQB_CLASSES, None),
    ("SEMANTIC6", SEMANTIC6_ANCHOR_MAP, SEMANTIC6_CLASSES, "semantic6_label"),
]

proto_dict = {}
proto_info_rows = []
feature_frames = []
residual_frames = []

for proto_set, anchor_map, classes, target_col in PROTO_SPECS:
    proto, info = build_anchor_prototypes(
        X_std,
        families,
        anchor_map,
        classes,
    )

    proto_dict[proto_set] = {
        "proto": proto,
        "classes": classes,
        "anchor_map": anchor_map,
        "target_col": target_col,
    }

    info["prototype_set"] = proto_set
    proto_info_rows.append(info)

    feat = prototype_feature_frame(
        X_std,
        proto,
        classes,
        prefix=proto_set,
    )

    feature_frames.append(feat)

    residual_norm, residual_vec, coords = affine_projection_residuals(X_std, proto)

    res_df = pd.DataFrame({
        f"{proto_set}_affine_residual": residual_norm,
    })

    for j, cls in enumerate(classes):
        res_df[f"{proto_set}_affine_coord_{safe_name(cls)}"] = coords[:, j]

    residual_frames.append(res_df)

proto_info_df = pd.concat(proto_info_rows, axis=0, ignore_index=True)

# ------------------------------------------------------------
# Avoid duplicate geometry columns when input comes from 3D.
# 3D feature table already contains SKQ / SKQO geometry columns.
# 3E recomputes SKQ / SKQR / SKQE / SKQRE / SKQB / SEMANTIC6,
# so old geometry columns must be removed before concat.
# ------------------------------------------------------------

generated_prefixes = (
    "SKQ_",
    "SKQR_rule_",
    "SKQE_evidence_",
    "SKQRE_rule_evidence_",
    "SKQB_broad_binding_",
    "SEMANTIC6_",
    "AXIS4_SKQO_",
)

generated_other_prefixes = (
    "proj_axis_",
    "abs_proj_axis_",
)

generated_suffixes = (
    "_residual_reduction_from_SKQ",
    "_residual_reduction_ratio",
)

base_df = df.reset_index(drop=True).copy()

# First remove duplicated columns already present in the input table.
base_df = base_df.loc[:, ~base_df.columns.duplicated()].copy()

drop_existing_cols = []

for c in base_df.columns:
    if c.startswith(generated_prefixes):
        drop_existing_cols.append(c)
    elif c.startswith(generated_other_prefixes):
        drop_existing_cols.append(c)
    elif c.endswith(generated_suffixes):
        drop_existing_cols.append(c)

if drop_existing_cols:
    print(f"Dropping {len(drop_existing_cols)} pre-existing geometry columns from input table.")
    base_df = base_df.drop(columns=drop_existing_cols)

df_geo = pd.concat(
    [base_df]
    + [x.reset_index(drop=True) for x in feature_frames]
    + [x.reset_index(drop=True) for x in residual_frames],
    axis=1,
)

# Final safety check.
df_geo = df_geo.loc[:, ~df_geo.columns.duplicated()].copy()

# Residual reductions relative to SKQ.
for proto_set, _, _, _ in PROTO_SPECS:
    if proto_set == "SKQ":
        continue

    df_geo[f"{proto_set}_residual_reduction_from_SKQ"] = (
        df_geo["SKQ_affine_residual"] - df_geo[f"{proto_set}_affine_residual"]
    )

    df_geo[f"{proto_set}_residual_reduction_ratio"] = (
        df_geo[f"{proto_set}_residual_reduction_from_SKQ"]
        / (df_geo["SKQ_affine_residual"].abs() + 1e-12)
    )

# ============================================================
# SKQ RESIDUAL AXIS DISSECTION
# ============================================================

skq_proto = proto_dict["SKQ"]["proto"]
skq_residual_norm, skq_residual_vec, skq_coords = affine_projection_residuals(
    X_std,
    skq_proto,
)

df_geo["SKQ_residual_norm_recomputed"] = skq_residual_norm

AXIS_GROUPS = {
    "axis_S_shift": ["stable_paraphrase", "stable_irrelevant"],
    "axis_E_source": ["competition_source_claim"],
    "axis_E_equal": ["competition_equal_evidence"],
    "axis_E_evidence_binding": ["competition_source_claim", "competition_equal_evidence"],
    "axis_R_override": ["closure_override"],
    "axis_R_exception": ["closure_exception"],
    "axis_R_rule_binding": ["closure_override", "closure_exception"],
    "axis_Q_temporal": ["closure_temporal"],
    "axis_Q_authority": ["closure_authority"],
    "axis_Q_temporal_authority": ["closure_temporal", "closure_authority"],
    "axis_Q_negation_update": ["closure_negation", "closure_update"],
    "axis_binding_broad": [
        "competition_source_claim",
        "competition_equal_evidence",
        "closure_override",
        "closure_exception",
    ],
}

axis_vectors = {}
axis_rows = []

for axis_name, fams in AXIS_GROUPS.items():
    mask = df["family"].isin(fams).values
    axis = residual_axis_from_group(skq_residual_vec, mask)
    axis_vectors[axis_name] = axis

    axis_rows.append({
        "axis_name": axis_name,
        "families": ",".join(fams),
        "n": int(mask.sum()),
        "axis_norm": float(np.linalg.norm(axis)),
    })

axis_info_df = pd.DataFrame(axis_rows)

# Projection of every sample to each residual axis.
for axis_name, axis in axis_vectors.items():
    df_geo[f"proj_{axis_name}"] = projection_to_axis(skq_residual_vec, axis)
    df_geo[f"abs_proj_{axis_name}"] = np.abs(df_geo[f"proj_{axis_name}"])

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
    by="cosine_similarity",
    ascending=False,
)

# ============================================================
# LOFO NEAREST-PROTOTYPE EVALUATION
# ============================================================

logo = LeaveOneGroupOut()

lofo_summary_rows = []
lofo_pred_rows = []
lofo_family_rows = []

LOFO_SPECS = [
    ("SKQ", SKQ_ANCHOR_MAP, SKQ_CLASSES, "axis3_label"),
    ("AXIS4_SKQO", {
        "stable_clean_basic": "S_stable",
        "stable_redundant": "S_stable",
        "stable_weak_note": "S_stable",

        "competition_ambiguous": "K_competition",
        "competition_branch": "K_competition",
        "competition_direct": "K_competition",
        "competition_source_claim": "K_competition",
        "competition_equal_evidence": "K_competition",

        "closure_negation": "Q_closure",
        "closure_update": "Q_closure",
        "closure_temporal": "Q_closure",
        "closure_authority": "Q_closure",

        "closure_override": "O_rule_binding",
        "closure_exception": "O_rule_binding",
    }, AXIS4_CLASSES, "axis4_label"),
    ("SEMANTIC6", SEMANTIC6_ANCHOR_MAP, SEMANTIC6_CLASSES, "semantic6_label"),
]

for proto_set, anchor_map, classes, target_col in LOFO_SPECS:
    y = df[target_col].values

    for metric in ["euclidean", "cosine"]:
        pred_all = np.empty(len(df), dtype=object)

        for train_idx, test_idx in logo.split(X_raw, y, groups=families):
            heldout = families[test_idx][0]

            fold_scaler = StandardScaler()
            X_train = fold_scaler.fit_transform(X_raw[train_idx])
            X_test = fold_scaler.transform(X_raw[test_idx])

            train_families = families[train_idx]

            proto, proto_info = build_anchor_prototypes(
                X_train,
                train_families,
                anchor_map,
                classes,
            )

            pred, dist, weights, score = nearest_proto_predict(
                X_test,
                proto,
                classes,
                metric=metric,
            )

            pred_all[test_idx] = pred

            temp = df.iloc[test_idx][[
                "idx",
                "family",
                "mechanism",
                "submechanism",
                "risk_regime",
                "axis3_label",
                "axis4_label",
                "semantic6_label",
                "diagnostic_group",
                "y_conflict",
            ]].copy()

            temp["prototype_set"] = proto_set
            temp["metric"] = metric
            temp["target_col"] = target_col
            temp["true_label"] = y[test_idx]
            temp["pred_label"] = pred
            temp["mix_entropy"] = entropy_rows(weights)
            temp["nearest_dist"] = np.min(dist, axis=1)
            temp["dist_margin_second_minus_first"] = (
                np.sort(dist, axis=1)[:, 1] - np.sort(dist, axis=1)[:, 0]
            )

            for j, cls in enumerate(classes):
                s = safe_name(cls)
                temp[f"dist_to_{s}"] = dist[:, j]
                temp[f"mix_w_{s}"] = weights[:, j]
                temp[f"score_to_{s}"] = score[:, j]

            lofo_pred_rows.append(temp)

            fam_row = {
                "prototype_set": proto_set,
                "metric": metric,
                "target_col": target_col,
                "heldout_family": heldout,
                "true_label": y[test_idx][0],
                "semantic6_label": SEMANTIC6_MAP[heldout],
                "diagnostic_group": DIAGNOSTIC_GROUP_MAP[heldout],
                "train_has_true_label": bool(y[test_idx][0] in set(df.iloc[train_idx][target_col].values)),
                "accuracy": float(accuracy_score(y[test_idx], pred)),
                "macro_f1": float(f1_score(y[test_idx], pred, average="macro", labels=classes, zero_division=0)),
                "mean_mix_entropy": float(entropy_rows(weights).mean()),
                "mean_nearest_dist": float(np.min(dist, axis=1).mean()),
                "mean_dist_margin": float(
                    (np.sort(dist, axis=1)[:, 1] - np.sort(dist, axis=1)[:, 0]).mean()
                ),
            }

            for j, cls in enumerate(classes):
                s = safe_name(cls)
                fam_row[f"mean_dist_to_{s}"] = float(dist[:, j].mean())
                fam_row[f"mean_mix_w_{s}"] = float(weights[:, j].mean())

            lofo_family_rows.append(fam_row)

        lofo_summary_rows.append(
            summarize_nearest(
                y_true=y,
                y_pred=pred_all,
                classes=classes,
                target_name=target_col,
                proto_set=proto_set,
                metric=f"LOFO_nearest_proto_{metric}",
            )
        )

lofo_summary_df = pd.DataFrame(lofo_summary_rows)
lofo_pred_df = pd.concat(lofo_pred_rows, axis=0, ignore_index=True)
lofo_family_df = pd.DataFrame(lofo_family_rows)

# ============================================================
# GROUPKFOLD LOGISTIC BASELINES
# ============================================================

lr_rows = []

for target_col, classes in [
    ("axis3_label", AXIS3_CLASSES),
    ("axis4_label", AXIS4_CLASSES),
    ("semantic6_label", SEMANTIC6_CLASSES),
    ("mechanism", sorted(df["mechanism"].unique().tolist())),
    ("submechanism", sorted(df["submechanism"].unique().tolist())),
    ("risk_regime", sorted(df["risk_regime"].unique().tolist())),
]:
    lr_rows.append(
        groupkfold_logistic_eval(
            df,
            X_raw,
            target_col=target_col,
            classes=classes,
            feature_set="delta_R_raw_20_25",
        )
    )

group_lr_df = pd.DataFrame(lr_rows)

# ============================================================
# FAMILY AND GROUP SUMMARIES
# ============================================================

family_rows = []

for fam in FAMILY_ORDER:
    sub = df_geo[df_geo["family"] == fam]
    if len(sub) == 0:
        continue

    row = {
        "family": fam,
        "semantic6_label": SEMANTIC6_MAP[fam],
        "diagnostic_group": DIAGNOSTIC_GROUP_MAP[fam],
        "mechanism": COARSE_MECHANISM_MAP[fam],
        "submechanism": SUBMECHANISM_MAP[fam],
        "risk_regime": RISK_REGIME_MAP[fam],
        "n": int(len(sub)),
        "genE_rate": float(sub["y_conflict"].mean()),
    }

    for c in DELTA_R_COLS:
        row[f"mean_{c}"] = float(sub[c].mean())
        row[f"std_{c}"] = float(sub[c].std())

    # Residuals and residual reductions.
    row["SKQ_mean_residual"] = float(sub["SKQ_affine_residual"].mean())

    for proto_set, _, classes, _ in PROTO_SPECS:
        row[f"{proto_set}_mean_residual"] = float(sub[f"{proto_set}_affine_residual"].mean())

        if proto_set != "SKQ":
            row[f"{proto_set}_mean_reduction_from_SKQ"] = float(
                sub[f"{proto_set}_residual_reduction_from_SKQ"].mean()
            )
            row[f"{proto_set}_mean_reduction_ratio"] = float(
                sub[f"{proto_set}_residual_reduction_ratio"].mean()
            )

    # Semantic6 mixture.
    row["SEMANTIC6_centroid_nearest_proto"] = sub["SEMANTIC6_nearest_proto"].mode().iloc[0]
    row["SEMANTIC6_mean_mix_entropy"] = float(sub["SEMANTIC6_mix_entropy"].mean())
    row["SEMANTIC6_mean_mix_margin"] = float(sub["SEMANTIC6_mix_margin_top_minus_second"].mean())

    for cls in SEMANTIC6_CLASSES:
        s = safe_name(cls)
        row[f"SEMANTIC6_mean_mix_w_{s}"] = float(sub[f"SEMANTIC6_mix_w_{s}"].mean())
        row[f"SEMANTIC6_mean_dist_to_{s}"] = float(sub[f"SEMANTIC6_dist_to_{s}"].mean())

    # Residual-axis projections.
    for axis_name in axis_names:
        row[f"mean_proj_{axis_name}"] = float(sub[f"proj_{axis_name}"].mean())
        row[f"mean_abs_proj_{axis_name}"] = float(sub[f"abs_proj_{axis_name}"].mean())

    family_rows.append(row)

family_summary_df = pd.DataFrame(family_rows)

family_summary_df["family_order"] = family_summary_df["family"].map({
    fam: i for i, fam in enumerate(FAMILY_ORDER)
})
family_summary_df = family_summary_df.sort_values("family_order").drop(columns=["family_order"])

# Diagnostic group summary.
group_rows = []

for group, sub in df_geo.groupby("diagnostic_group"):
    row = {
        "diagnostic_group": group,
        "n_samples": int(len(sub)),
        "n_families": int(sub["family"].nunique()),
        "genE_rate": float(sub["y_conflict"].mean()),
        "SKQ_mean_residual": float(sub["SKQ_affine_residual"].mean()),
    }

    for proto_set, _, classes, _ in PROTO_SPECS:
        row[f"{proto_set}_mean_residual"] = float(sub[f"{proto_set}_affine_residual"].mean())

        if proto_set != "SKQ":
            row[f"{proto_set}_mean_reduction_from_SKQ"] = float(
                sub[f"{proto_set}_residual_reduction_from_SKQ"].mean()
            )
            row[f"{proto_set}_mean_reduction_ratio"] = float(
                sub[f"{proto_set}_residual_reduction_ratio"].mean()
            )

    for cls in SEMANTIC6_CLASSES:
        s = safe_name(cls)
        row[f"SEMANTIC6_mean_mix_w_{s}"] = float(sub[f"SEMANTIC6_mix_w_{s}"].mean())

    row["SEMANTIC6_mean_mix_entropy"] = float(sub["SEMANTIC6_mix_entropy"].mean())
    row["SEMANTIC6_mean_mix_margin"] = float(sub["SEMANTIC6_mix_margin_top_minus_second"].mean())

    for axis_name in axis_names:
        row[f"mean_proj_{axis_name}"] = float(sub[f"proj_{axis_name}"].mean())
        row[f"mean_abs_proj_{axis_name}"] = float(sub[f"abs_proj_{axis_name}"].mean())

    group_rows.append(row)

group_summary_df = pd.DataFrame(group_rows)

# Which prototype set reduces residual most for each family?
best_reduction_rows = []

for _, row in family_summary_df.iterrows():
    fam = row["family"]

    candidates = []
    for proto_set, _, _, _ in PROTO_SPECS:
        if proto_set == "SKQ":
            continue
        candidates.append((
            proto_set,
            row[f"{proto_set}_mean_reduction_from_SKQ"],
            row[f"{proto_set}_mean_reduction_ratio"],
        ))

    best = sorted(candidates, key=lambda x: x[1], reverse=True)[0]

    best_reduction_rows.append({
        "family": fam,
        "semantic6_label": row["semantic6_label"],
        "diagnostic_group": row["diagnostic_group"],
        "best_residual_reduction_proto_set": best[0],
        "best_reduction": float(best[1]),
        "best_reduction_ratio": float(best[2]),
    })

best_reduction_df = pd.DataFrame(best_reduction_rows)

# Pairwise family centroid distances.
family_centroids = []
family_names = []

for fam in FAMILY_ORDER:
    sub = df[df["family"] == fam]
    if len(sub) == 0:
        continue

    family_names.append(fam)
    family_centroids.append(X_std[sub.index].mean(axis=0))

family_centroids = np.vstack(family_centroids)

fam_dist = euclidean_dist_matrix(family_centroids, family_centroids)
fam_cos = cosine_sim_matrix(family_centroids, family_centroids)

pair_rows = []

for i, f1 in enumerate(family_names):
    for j, f2 in enumerate(family_names):
        if i >= j:
            continue

        pair_rows.append({
            "family_1": f1,
            "family_2": f2,
            "semantic6_1": SEMANTIC6_MAP[f1],
            "semantic6_2": SEMANTIC6_MAP[f2],
            "diagnostic_group_1": DIAGNOSTIC_GROUP_MAP[f1],
            "diagnostic_group_2": DIAGNOSTIC_GROUP_MAP[f2],
            "euclidean_distance": float(fam_dist[i, j]),
            "cosine_similarity": float(fam_cos[i, j]),
            "same_semantic6": bool(SEMANTIC6_MAP[f1] == SEMANTIC6_MAP[f2]),
            "same_diagnostic_group": bool(DIAGNOSTIC_GROUP_MAP[f1] == DIAGNOSTIC_GROUP_MAP[f2]),
            "same_mechanism": bool(COARSE_MECHANISM_MAP[f1] == COARSE_MECHANISM_MAP[f2]),
            "same_submechanism": bool(SUBMECHANISM_MAP[f1] == SUBMECHANISM_MAP[f2]),
        })

family_pairwise_df = pd.DataFrame(pair_rows)

# PCA coordinates.
pca = PCA(n_components=2, random_state=RANDOM_SEED)
XY = pca.fit_transform(X_std)

pca_df = df[[
    "idx",
    "family",
    "mechanism",
    "submechanism",
    "risk_regime",
    "axis3_label",
    "axis4_label",
    "semantic6_label",
    "diagnostic_group",
    "y_conflict",
]].copy()

pca_df["pc1"] = XY[:, 0]
pca_df["pc2"] = XY[:, 1]

# Prototype coordinates.
proto_coord_rows = []

for proto_set, info in proto_dict.items():
    proto = info["proto"]
    classes = info["classes"]
    proto_xy = pca.transform(proto)

    for i, cls in enumerate(classes):
        row = {
            "prototype_set": proto_set,
            "prototype_class": cls,
            "pc1": float(proto_xy[i, 0]),
            "pc2": float(proto_xy[i, 1]),
        }

        for j, c in enumerate(DELTA_R_COLS):
            row[f"proto_std_{c}"] = float(proto[i, j])

        proto_coord_rows.append(row)

prototype_coordinates_df = pd.DataFrame(proto_coord_rows)

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n\n============================================================")
print("3E LOFO Nearest-Prototype Summary")
print("============================================================\n")
print(
    lofo_summary_df
    .sort_values(by=["prototype_set", "metric"])
    .to_string(index=False)
)

print("\n\n============================================================")
print("3E GroupKFold Logistic Baseline on Raw Delta-R")
print("============================================================\n")
print(
    group_lr_df
    .sort_values(by=["target", "macro_f1"], ascending=[True, False])
    .to_string(index=False)
)

print("\n\n============================================================")
print("3E Axis Similarity: SKQ Residual Semantic Axes")
print("============================================================\n")
print(axis_similarity_df.head(40).to_string(index=False))

print("\n\n============================================================")
print("3E Family Semantic Summary")
print("============================================================\n")

view_cols = [
    "family",
    "semantic6_label",
    "diagnostic_group",
    "genE_rate",

    "SEMANTIC6_centroid_nearest_proto",
    "SEMANTIC6_mean_mix_entropy",
    "SEMANTIC6_mean_mix_margin",

    "SEMANTIC6_mean_mix_w_S_core",
    "SEMANTIC6_mean_mix_w_S_shift",
    "SEMANTIC6_mean_mix_w_K_competition",
    "SEMANTIC6_mean_mix_w_E_evidence_binding",
    "SEMANTIC6_mean_mix_w_Q_canonical_closure",
    "SEMANTIC6_mean_mix_w_R_rule_binding",

    "SKQ_mean_residual",
    "SKQRE_rule_evidence_mean_residual",
    "SKQB_broad_binding_mean_residual",
    "SKQRE_rule_evidence_mean_reduction_from_SKQ",
    "SKQB_broad_binding_mean_reduction_from_SKQ",

    "mean_proj_axis_E_evidence_binding",
    "mean_proj_axis_R_rule_binding",
    "mean_proj_axis_binding_broad",
    "mean_proj_axis_Q_temporal_authority",
]

existing_view_cols = [c for c in view_cols if c in family_summary_df.columns]

print(
    family_summary_df[existing_view_cols]
    .to_string(index=False)
)

print("\n\n============================================================")
print("3E Diagnostic Group Summary")
print("============================================================\n")

group_view_cols = [
    "diagnostic_group",
    "n_samples",
    "n_families",
    "genE_rate",

    "SKQ_mean_residual",
    "SKQR_rule_mean_reduction_from_SKQ",
    "SKQE_evidence_mean_reduction_from_SKQ",
    "SKQRE_rule_evidence_mean_reduction_from_SKQ",
    "SKQB_broad_binding_mean_reduction_from_SKQ",
    "SEMANTIC6_mean_residual",

    "SEMANTIC6_mean_mix_w_E_evidence_binding",
    "SEMANTIC6_mean_mix_w_R_rule_binding",
    "SEMANTIC6_mean_mix_w_Q_canonical_closure",
    "SEMANTIC6_mean_mix_w_K_competition",

    "mean_proj_axis_E_evidence_binding",
    "mean_proj_axis_R_rule_binding",
    "mean_proj_axis_binding_broad",
]

existing_group_view_cols = [c for c in group_view_cols if c in group_summary_df.columns]

print(
    group_summary_df[existing_group_view_cols]
    .sort_values(by="SKQ_mean_residual", ascending=False)
    .to_string(index=False)
)

print("\n\n============================================================")
print("3E Best Residual-Reduction Prototype Set by Family")
print("============================================================\n")
print(best_reduction_df.to_string(index=False))

print("\n\n============================================================")
print("3E Closest Family Pairs in Delta-R Space")
print("============================================================\n")
print(
    family_pairwise_df
    .sort_values(by="euclidean_distance")
    .head(30)
    .to_string(index=False)
)

# ============================================================
# SAVE OUTPUTS
# ============================================================

feature_table_path = SAVE_DIR / "constraint_audit3e_feature_table_with_binding_semantics.csv"
proto_info_path = SAVE_DIR / "constraint_audit3e_prototype_anchor_info.csv"
axis_info_path = SAVE_DIR / "constraint_audit3e_residual_axis_info.csv"
axis_similarity_path = SAVE_DIR / "constraint_audit3e_residual_axis_similarity.csv"

lofo_summary_path = SAVE_DIR / "constraint_audit3e_lofo_summary.csv"
lofo_pred_path = SAVE_DIR / "constraint_audit3e_lofo_predictions.csv"
lofo_family_path = SAVE_DIR / "constraint_audit3e_lofo_family_summary.csv"

group_lr_path = SAVE_DIR / "constraint_audit3e_groupkfold_logistic_summary.csv"

family_summary_path = SAVE_DIR / "constraint_audit3e_family_semantic_summary.csv"
group_summary_path = SAVE_DIR / "constraint_audit3e_diagnostic_group_summary.csv"
best_reduction_path = SAVE_DIR / "constraint_audit3e_best_residual_reduction_by_family.csv"
pairwise_path = SAVE_DIR / "constraint_audit3e_family_pairwise_distances.csv"

pca_path = SAVE_DIR / "constraint_audit3e_deltaR_pca_coordinates.csv"
prototype_coordinates_path = SAVE_DIR / "constraint_audit3e_prototype_coordinates.csv"

df_geo.to_csv(feature_table_path, index=False)
proto_info_df.to_csv(proto_info_path, index=False)
axis_info_df.to_csv(axis_info_path, index=False)
axis_similarity_df.to_csv(axis_similarity_path, index=False)

lofo_summary_df.to_csv(lofo_summary_path, index=False)
lofo_pred_df.to_csv(lofo_pred_path, index=False)
lofo_family_df.to_csv(lofo_family_path, index=False)

group_lr_df.to_csv(group_lr_path, index=False)

family_summary_df.to_csv(family_summary_path, index=False)
group_summary_df.to_csv(group_summary_path, index=False)
best_reduction_df.to_csv(best_reduction_path, index=False)
family_pairwise_df.to_csv(pairwise_path, index=False)

pca_df.to_csv(pca_path, index=False)
prototype_coordinates_df.to_csv(prototype_coordinates_path, index=False)

print("\nSaved outputs:")
print(" ", feature_table_path)
print(" ", proto_info_path)
print(" ", axis_info_path)
print(" ", axis_similarity_path)
print(" ", lofo_summary_path)
print(" ", lofo_pred_path)
print(" ", lofo_family_path)
print(" ", group_lr_path)
print(" ", family_summary_path)
print(" ", group_summary_path)
print(" ", best_reduction_path)
print(" ", pairwise_path)
print(" ", pca_path)
print(" ", prototype_coordinates_path)

# ============================================================
# INTERPRETATION GUIDE
# ============================================================

print("\n\n============================================================")
print("Constraint-Audit-3E Interpretation Guide")
print("============================================================\n")

print("Main question:")
print("  Is O an override-only axis, or a broader binding axis?")
print()
print("Read these files first:")
print("  constraint_audit3e_family_semantic_summary.csv")
print("  constraint_audit3e_diagnostic_group_summary.csv")
print("  constraint_audit3e_residual_axis_similarity.csv")
print("  constraint_audit3e_best_residual_reduction_by_family.csv")
print("  constraint_audit3e_lofo_summary.csv")
print()
print("Positive evidence for broad binding axis if:")
print("  1. axis_E_evidence_binding is highly cosine-aligned with axis_R_rule_binding.")
print("  2. SKQB_broad_binding reduces residual for both E_* and R_* families.")
print("  3. competition_equal_evidence / competition_source_claim receive large binding projection.")
print("  4. closure_override / closure_exception receive both R_rule and broad-binding projection.")
print()
print("Evidence for separable subaxes if:")
print("  1. axis_E_evidence_binding and axis_R_rule_binding are low-cosine or orthogonal.")
print("  2. SKQRE_rule_evidence beats SKQB_broad_binding clearly.")
print("  3. E_source/equal and R_override/exception choose different nearest semantic prototypes.")
print()
print("Important caveat:")
print("  SEMANTIC6 is harder than SKQ/SKQO; lower LOFO macro-F1 does not mean failure.")
print("  The key evidence is residual reduction and residual-axis alignment.")
print()
print("Theory update if broad binding wins:")
print("  Replace O with B:")
print("      C_struct_20:25 = DeltaR_20:25 in G_{S,K,Q,B}")
print("      B = evidence/rule/source/authority binding")
print()
print("Theory update if subaxes separate:")
print("  Keep multiple binding modes:")
print("      B_E = evidence/source binding")
print("      B_R = rule/exception binding")
print("      B_A = authority/temporal binding")
print()
print("Done.")