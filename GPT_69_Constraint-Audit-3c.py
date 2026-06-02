# ============================================================
# Constraint-Audit-3C
# Delta-R Prototype Geometry
#
# Input:
#   Prefer:
#     ./constraint_audit3b1_outputs/constraint_audit3b1_feature_table.csv
#   Fallback:
#     ./constraint_audit3b_outputs/constraint_audit3b_feature_table.csv
#
# Goal:
#   3B.1 showed that raw Delta-R trajectory:
#
#       [DELTA_R_L20, ..., DELTA_R_L25]
#
#   is currently the strongest minimal structural invariant.
#
#   3C asks:
#       Does Delta-R space form interpretable mechanism geometry?
#
# Core idea:
#   Build prototypes in Delta-R space:
#
#       S = stable prototype
#       K = competition prototype
#       Q = closure prototype
#
#   Then analyze:
#       1. nearest prototype classification
#       2. prototype mixture weights
#       3. family centroid geometry
#       4. ambiguous / mixed families
#       5. whether misclassified families lie between prototypes
#
# Outputs:
#   ./constraint_audit3c_outputs/
#
# Main files:
#   constraint_audit3c_lofo_nearest_prototype.csv
#   constraint_audit3c_lofo_summary.csv
#   constraint_audit3c_family_geometry.csv
#   constraint_audit3c_family_mixture_summary.csv
#   constraint_audit3c_full_prototype_coordinates.csv
#   constraint_audit3c_deltaR_pca_coordinates.csv
# ============================================================

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

RANDOM_SEED = 42

INPUT_3B1 = Path("./constraint_audit3b1_outputs/constraint_audit3b1_feature_table.csv")
INPUT_3B = Path("./constraint_audit3b_outputs/constraint_audit3b_feature_table.csv")

SAVE_DIR = Path("./constraint_audit3c_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

TRACK_LAYERS = [20, 21, 22, 23, 24, 25]

DELTA_R_COLS = [f"DELTA_R_L{l}" for l in TRACK_LAYERS]

N_GROUP_SPLITS = 5

SOFTMAX_TEMPERATURE = 1.0

# ============================================================
# LABEL MAPS
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

# ============================================================
# HELPERS
# ============================================================

def parse_bool_series(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


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


def cosine_sim_matrix(X, P):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    return Xn @ Pn.T


def euclidean_dist_matrix(X, P):
    # X: n x d, P: k x d
    diff = X[:, None, :] - P[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def build_prototypes(X_train, y_train, classes):
    rows = []
    proto = []

    for cls in classes:
        mask = y_train == cls
        if mask.sum() == 0:
            proto.append(np.zeros(X_train.shape[1], dtype=np.float64))
            rows.append({
                "class": cls,
                "n_train": 0,
            })
        else:
            p = X_train[mask].mean(axis=0)
            proto.append(p)
            rows.append({
                "class": cls,
                "n_train": int(mask.sum()),
            })

    proto = np.vstack(proto).astype(np.float64)
    info = pd.DataFrame(rows)

    return proto, info


def nearest_prototype_predict(X_test, proto, classes, metric="euclidean"):
    if metric == "euclidean":
        dist = euclidean_dist_matrix(X_test, proto)
        pred_idx = np.argmin(dist, axis=1)
        weights = softmax_neg_distance(dist, temperature=SOFTMAX_TEMPERATURE)
        score_mat = -dist

    elif metric == "cosine":
        sim = cosine_sim_matrix(X_test, proto)
        pred_idx = np.argmax(sim, axis=1)
        # Convert cosine similarity to pseudo-distance.
        dist = 1.0 - sim
        weights = softmax_neg_distance(dist, temperature=SOFTMAX_TEMPERATURE)
        score_mat = sim

    else:
        raise ValueError(f"Unknown metric: {metric}")

    pred = np.array(classes, dtype=object)[pred_idx]

    return pred, dist, weights, score_mat


def prototype_features(X, proto, classes):
    dist = euclidean_dist_matrix(X, proto)
    sim = cosine_sim_matrix(X, proto)
    weights = softmax_neg_distance(dist, temperature=SOFTMAX_TEMPERATURE)

    out = pd.DataFrame(index=np.arange(len(X)))

    for j, cls in enumerate(classes):
        safe = cls.replace("/", "_")
        out[f"dist_to_{safe}"] = dist[:, j]
        out[f"cos_to_{safe}"] = sim[:, j]
        out[f"mix_w_{safe}"] = weights[:, j]

    order = np.argsort(dist, axis=1)
    best = order[:, 0]
    second = order[:, 1]

    out["nearest_proto"] = np.array(classes, dtype=object)[best]
    out["nearest_dist"] = dist[np.arange(len(X)), best]
    out["second_dist"] = dist[np.arange(len(X)), second]
    out["dist_margin_second_minus_first"] = out["second_dist"] - out["nearest_dist"]

    w_order = np.argsort(-weights, axis=1)
    out["top_mix_class"] = np.array(classes, dtype=object)[w_order[:, 0]]
    out["top_mix_weight"] = weights[np.arange(len(X)), w_order[:, 0]]
    out["second_mix_weight"] = weights[np.arange(len(X)), w_order[:, 1]]
    out["mix_margin_top_minus_second"] = out["top_mix_weight"] - out["second_mix_weight"]
    out["mix_entropy"] = entropy_rows(weights)

    return out


def affine_residual_to_prototype_plane(X, proto):
    """
    For three coarse prototypes, this measures whether a point lies near
    the affine plane spanned by S/K/Q. In 6D, large residual means the family
    contains structure not captured by the three coarse prototypes.
    """
    if proto.shape[0] < 2:
        return np.zeros(len(X), dtype=np.float64)

    p0 = proto[0]
    B = (proto[1:] - p0).T  # d x (k-1)

    residuals = []

    for x in X:
        y = x - p0
        coef, *_ = np.linalg.lstsq(B, y, rcond=None)
        x_hat = p0 + B @ coef
        residuals.append(np.linalg.norm(x - x_hat))

    return np.array(residuals, dtype=np.float64)


def summarize_classification(y_true, y_pred, classes, feature_set, target, metric):
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", labels=classes, zero_division=0)

    return {
        "feature_set": feature_set,
        "target": target,
        "metric": metric,
        "accuracy": float(acc),
        "macro_f1": float(macro),
    }


def make_lr_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            max_iter=5000,
        )),
    ])


def groupkfold_lr_eval(df, X, target_col, classes, feature_set):
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

if INPUT_3B1.exists():
    INPUT_FILE = INPUT_3B1
elif INPUT_3B.exists():
    INPUT_FILE = INPUT_3B
else:
    raise FileNotFoundError(
        "Cannot find 3B.1 or 3B feature table. Run Constraint-Audit-3B.1 first."
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
    raise RuntimeError(
        "Missing required columns: "
        + ", ".join(missing)
        + "\nNeed raw DELTA_R_L20...DELTA_R_L25 columns from 3B/3B.1."
    )

df["mechanism"] = df["family"].map(COARSE_MECHANISM_MAP)
df["submechanism"] = df["family"].map(SUBMECHANISM_MAP)
df["risk_regime"] = df["family"].map(RISK_REGIME_MAP)

if df["mechanism"].isna().any():
    bad = df[df["mechanism"].isna()]["family"].unique().tolist()
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

print("\n============================================================")
print("Constraint-Audit-3C Input Summary")
print("============================================================\n")

print("Input:", INPUT_FILE)
print("Rows:", len(df))
print("Families:", df["family"].nunique())
print("Delta-R cols:", DELTA_R_COLS)

print("\nFamily summary:")
print(
    f"{'family':<32}"
    f"{'mechanism':<24}"
    f"{'submechanism':<28}"
    f"{'risk_regime':<24}"
    f"{'N':<6}"
    f"{'GenE':<10}"
)

for fam in FAMILY_ORDER:
    sub = df[df["family"] == fam]
    if len(sub) == 0:
        continue

    print(
        f"{fam:<32}"
        f"{COARSE_MECHANISM_MAP[fam]:<24}"
        f"{SUBMECHANISM_MAP[fam]:<28}"
        f"{RISK_REGIME_MAP[fam]:<24}"
        f"{len(sub):<6}"
        f"{sub['y_conflict'].mean():<10.4f}"
    )

# ============================================================
# FULL-DATA PROTOTYPES FOR GEOMETRY INSPECTION
# ============================================================

full_scaler = StandardScaler()
X_std = full_scaler.fit_transform(X_raw)

full_proto, full_proto_info = build_prototypes(
    X_std,
    df["mechanism"].values,
    MECHANISM_CLASSES,
)

full_proto_features = prototype_features(
    X_std,
    full_proto,
    MECHANISM_CLASSES,
)

full_proto_features["affine_residual_to_SKQ_plane"] = affine_residual_to_prototype_plane(
    X_std,
    full_proto,
)

df_geo = pd.concat(
    [df.reset_index(drop=True), full_proto_features.reset_index(drop=True)],
    axis=1,
)

# PCA for inspection.
pca = PCA(n_components=2, random_state=RANDOM_SEED)
XY = pca.fit_transform(X_std)
proto_XY = pca.transform(full_proto)

pca_df = df[["idx", "family", "mechanism", "submechanism", "risk_regime", "y_conflict"]].copy()
pca_df["pc1"] = XY[:, 0]
pca_df["pc2"] = XY[:, 1]

proto_coord_rows = []
for i, cls in enumerate(MECHANISM_CLASSES):
    row = {
        "prototype_class": cls,
        "pc1": float(proto_XY[i, 0]),
        "pc2": float(proto_XY[i, 1]),
    }
    for j, c in enumerate(DELTA_R_COLS):
        row[f"proto_std_{c}"] = float(full_proto[i, j])
    proto_coord_rows.append(row)

proto_coord_df = pd.DataFrame(proto_coord_rows)

# ============================================================
# LOFO NEAREST-PROTOTYPE EVALUATION
# ============================================================

logo = LeaveOneGroupOut()

lofo_rows = []
lofo_pred_rows = []
lofo_family_geo_rows = []

for target_col, classes in [
    ("mechanism", MECHANISM_CLASSES),
    ("submechanism", SUBMECHANISM_CLASSES),
    ("risk_regime", RISK_REGIME_CLASSES),
]:
    y = df[target_col].values
    families = df["family"].values

    for metric in ["euclidean", "cosine"]:
        pred_all = np.empty(len(df), dtype=object)

        for train_idx, test_idx in logo.split(X_raw, y, groups=families):
            heldout = families[test_idx][0]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_raw[train_idx])
            X_test = scaler.transform(X_raw[test_idx])

            y_train = y[train_idx]

            proto, proto_info = build_prototypes(X_train, y_train, classes)

            pred, dist, weights, score_mat = nearest_prototype_predict(
                X_test,
                proto,
                classes,
                metric=metric,
            )

            pred_all[test_idx] = pred

            # Per-row predictions.
            temp = df.iloc[test_idx][[
                "idx",
                "family",
                "mechanism",
                "submechanism",
                "risk_regime",
                "y_conflict",
            ]].copy()

            temp["target"] = target_col
            temp["metric"] = metric
            temp["true_label"] = y[test_idx]
            temp["pred_label"] = pred

            for j, cls in enumerate(classes):
                safe = cls.replace("/", "_")
                temp[f"dist_to_{safe}"] = dist[:, j]
                temp[f"mix_w_{safe}"] = weights[:, j]
                temp[f"score_to_{safe}"] = score_mat[:, j]

            temp["mix_entropy"] = entropy_rows(weights)
            temp["nearest_dist"] = np.min(dist, axis=1)
            temp["dist_margin_second_minus_first"] = (
                np.sort(dist, axis=1)[:, 1] - np.sort(dist, axis=1)[:, 0]
            )

            lofo_pred_rows.append(temp)

            # Family-level geometry under this LOFO fold.
            family_row = {
                "target": target_col,
                "metric": metric,
                "heldout_family": heldout,
                "true_label": y[test_idx][0],
                "train_has_true_label": bool(y[test_idx][0] in set(y_train)),
                "accuracy": float(accuracy_score(y[test_idx], pred)),
                "macro_f1": float(f1_score(y[test_idx], pred, average="macro", labels=classes, zero_division=0)),
                "mean_mix_entropy": float(entropy_rows(weights).mean()),
                "mean_nearest_dist": float(np.min(dist, axis=1).mean()),
                "mean_dist_margin": float(
                    (np.sort(dist, axis=1)[:, 1] - np.sort(dist, axis=1)[:, 0]).mean()
                ),
            }

            for j, cls in enumerate(classes):
                safe = cls.replace("/", "_")
                family_row[f"mean_dist_to_{safe}"] = float(dist[:, j].mean())
                family_row[f"mean_mix_w_{safe}"] = float(weights[:, j].mean())

            lofo_family_geo_rows.append(family_row)

        summary = summarize_classification(
            y_true=y,
            y_pred=pred_all,
            classes=classes,
            feature_set="delta_R_raw_20_25",
            target=target_col,
            metric=f"LOFO_nearest_proto_{metric}",
        )

        lofo_rows.append(summary)

lofo_summary_df = pd.DataFrame(lofo_rows)
lofo_pred_df = pd.concat(lofo_pred_rows, axis=0, ignore_index=True)
lofo_family_geo_df = pd.DataFrame(lofo_family_geo_rows)

# ============================================================
# GROUPKFOLD LOGISTIC BASELINE ON SAME 6D SPACE
# ============================================================

lr_rows = []

for target_col, classes in [
    ("mechanism", MECHANISM_CLASSES),
    ("submechanism", SUBMECHANISM_CLASSES),
    ("risk_regime", RISK_REGIME_CLASSES),
]:
    lr_rows.append(
        groupkfold_lr_eval(
            df,
            X_raw,
            target_col=target_col,
            classes=classes,
            feature_set="delta_R_raw_20_25",
        )
    )

group_lr_df = pd.DataFrame(lr_rows)

# ============================================================
# FAMILY CENTROIDS AND MIXTURE GEOMETRY
# ============================================================

family_rows = []

for fam in FAMILY_ORDER:
    sub = df_geo[df_geo["family"] == fam].copy()
    if len(sub) == 0:
        continue

    row = {
        "family": fam,
        "mechanism": COARSE_MECHANISM_MAP[fam],
        "submechanism": SUBMECHANISM_MAP[fam],
        "risk_regime": RISK_REGIME_MAP[fam],
        "n": int(len(sub)),
        "genE_rate": float(sub["y_conflict"].mean()),
        "pc1_mean": float(pca_df.loc[sub.index, "pc1"].mean()),
        "pc2_mean": float(pca_df.loc[sub.index, "pc2"].mean()),
    }

    for c in DELTA_R_COLS:
        row[f"mean_{c}"] = float(sub[c].mean())
        row[f"std_{c}"] = float(sub[c].std())

    for cls in MECHANISM_CLASSES:
        safe = cls.replace("/", "_")
        row[f"mean_dist_to_{safe}"] = float(sub[f"dist_to_{safe}"].mean())
        row[f"mean_cos_to_{safe}"] = float(sub[f"cos_to_{safe}"].mean())
        row[f"mean_mix_w_{safe}"] = float(sub[f"mix_w_{safe}"].mean())

    row["mean_mix_entropy"] = float(sub["mix_entropy"].mean())
    row["mean_mix_margin"] = float(sub["mix_margin_top_minus_second"].mean())
    row["mean_dist_margin"] = float(sub["dist_margin_second_minus_first"].mean())
    row["mean_affine_residual_to_SKQ_plane"] = float(sub["affine_residual_to_SKQ_plane"].mean())

    # Family centroid classification in full-data standardized space.
    fam_centroid = X_std[sub.index].mean(axis=0, keepdims=True)
    fam_feat = prototype_features(fam_centroid, full_proto, MECHANISM_CLASSES)
    row["centroid_nearest_proto"] = fam_feat["nearest_proto"].iloc[0]
    row["centroid_top_mix_class"] = fam_feat["top_mix_class"].iloc[0]
    row["centroid_mix_entropy"] = float(fam_feat["mix_entropy"].iloc[0])
    row["centroid_mix_margin"] = float(fam_feat["mix_margin_top_minus_second"].iloc[0])

    family_rows.append(row)

family_geometry_df = pd.DataFrame(family_rows)

# Sort for readability.
family_geometry_df["family_order"] = family_geometry_df["family"].map({
    fam: i for i, fam in enumerate(FAMILY_ORDER)
})
family_geometry_df = family_geometry_df.sort_values("family_order").drop(columns=["family_order"])

# Mixed-family candidates.
# High entropy / low margin means geometrically mixed between prototypes.
mixed_df = family_geometry_df.sort_values(
    by=["mean_mix_entropy", "mean_mix_margin"],
    ascending=[False, True],
).copy()

# ============================================================
# PAIRWISE FAMILY DISTANCES IN DELTA-R SPACE
# ============================================================

centroid_rows = []
centroids = []
fam_names = []

for fam in FAMILY_ORDER:
    sub = df[df["family"] == fam]
    if len(sub) == 0:
        continue

    fam_names.append(fam)
    centroids.append(X_std[sub.index].mean(axis=0))

centroids = np.vstack(centroids)

fam_dist = euclidean_dist_matrix(centroids, centroids)
fam_cos = cosine_sim_matrix(centroids, centroids)

pair_rows = []

for i, f1 in enumerate(fam_names):
    for j, f2 in enumerate(fam_names):
        if i >= j:
            continue

        pair_rows.append({
            "family_1": f1,
            "family_2": f2,
            "mechanism_1": COARSE_MECHANISM_MAP[f1],
            "mechanism_2": COARSE_MECHANISM_MAP[f2],
            "submechanism_1": SUBMECHANISM_MAP[f1],
            "submechanism_2": SUBMECHANISM_MAP[f2],
            "risk_regime_1": RISK_REGIME_MAP[f1],
            "risk_regime_2": RISK_REGIME_MAP[f2],
            "euclidean_distance": float(fam_dist[i, j]),
            "cosine_similarity": float(fam_cos[i, j]),
            "same_mechanism": bool(COARSE_MECHANISM_MAP[f1] == COARSE_MECHANISM_MAP[f2]),
            "same_submechanism": bool(SUBMECHANISM_MAP[f1] == SUBMECHANISM_MAP[f2]),
            "same_risk_regime": bool(RISK_REGIME_MAP[f1] == RISK_REGIME_MAP[f2]),
        })

family_pairwise_df = pd.DataFrame(pair_rows)

# ============================================================
# PROTOTYPE BASIS DECOMPOSITION
# ============================================================

# For coarse S/K/Q prototypes:
# Try to approximate each sample as an affine combination of prototypes.
# This is not constrained to be non-negative; it is a diagnostic of mixture direction.

def affine_coordinates_in_prototype_basis(X, proto, classes):
    p0 = proto[0]
    B = (proto[1:] - p0).T

    rows = []

    for i, x in enumerate(X):
        y = x - p0
        coef, *_ = np.linalg.lstsq(B, y, rcond=None)

        # affine coordinates:
        # x ≈ w0*p0 + w1*p1 + w2*p2
        # with w0 = 1 - sum(coef), w1=coef[0], w2=coef[1]
        weights = np.zeros(len(classes), dtype=np.float64)
        weights[0] = 1.0 - coef.sum()
        weights[1:] = coef

        x_hat = p0 + B @ coef
        residual = np.linalg.norm(x - x_hat)

        row = {
            "affine_residual": float(residual),
        }

        for j, cls in enumerate(classes):
            safe = cls.replace("/", "_")
            row[f"affine_coord_{safe}"] = float(weights[j])

        rows.append(row)

    return pd.DataFrame(rows)


affine_df = affine_coordinates_in_prototype_basis(
    X_std,
    full_proto,
    MECHANISM_CLASSES,
)

affine_aug_df = pd.concat(
    [
        df[["idx", "family", "mechanism", "submechanism", "risk_regime", "y_conflict"]].reset_index(drop=True),
        affine_df.reset_index(drop=True),
    ],
    axis=1,
)

affine_family_rows = []

for fam in FAMILY_ORDER:
    sub = affine_aug_df[affine_aug_df["family"] == fam]
    if len(sub) == 0:
        continue

    row = {
        "family": fam,
        "mechanism": COARSE_MECHANISM_MAP[fam],
        "submechanism": SUBMECHANISM_MAP[fam],
        "risk_regime": RISK_REGIME_MAP[fam],
        "n": int(len(sub)),
        "genE_rate": float(sub["y_conflict"].mean()),
        "mean_affine_residual": float(sub["affine_residual"].mean()),
    }

    for cls in MECHANISM_CLASSES:
        safe = cls.replace("/", "_")
        row[f"mean_affine_coord_{safe}"] = float(sub[f"affine_coord_{safe}"].mean())
        row[f"std_affine_coord_{safe}"] = float(sub[f"affine_coord_{safe}"].std())

    affine_family_rows.append(row)

affine_family_df = pd.DataFrame(affine_family_rows)

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n\n============================================================")
print("3C LOFO Nearest-Prototype Summary")
print("============================================================\n")

print(
    lofo_summary_df
    .sort_values(by=["target", "macro_f1"], ascending=[True, False])
    .to_string(index=False)
)

print("\n\n============================================================")
print("3C GroupKFold Logistic Baseline on Same 6D Delta-R")
print("============================================================\n")

print(
    group_lr_df
    .sort_values(by=["target", "macro_f1"], ascending=[True, False])
    .to_string(index=False)
)

print("\n\n============================================================")
print("3C Family Geometry: Coarse Prototype Mixture")
print("============================================================\n")

view_cols = [
    "family",
    "mechanism",
    "submechanism",
    "risk_regime",
    "genE_rate",
    "centroid_nearest_proto",
    "mean_mix_entropy",
    "mean_mix_margin",
    "mean_mix_w_stable_or_preserved",
    "mean_mix_w_competition_conflict",
    "mean_mix_w_closure_rewrite",
    "mean_affine_residual_to_SKQ_plane",
]

existing_view_cols = [c for c in view_cols if c in family_geometry_df.columns]

print(
    family_geometry_df[existing_view_cols]
    .to_string(index=False)
)

print("\n\n============================================================")
print("3C Most Mixed Families")
print("============================================================\n")

print(
    mixed_df[existing_view_cols]
    .head(16)
    .to_string(index=False)
)

print("\n\n============================================================")
print("3C Affine Prototype Coordinates by Family")
print("============================================================\n")

affine_view_cols = [
    "family",
    "mechanism",
    "submechanism",
    "risk_regime",
    "genE_rate",
    "mean_affine_coord_stable_or_preserved",
    "mean_affine_coord_competition_conflict",
    "mean_affine_coord_closure_rewrite",
    "mean_affine_residual",
]

existing_affine_view_cols = [c for c in affine_view_cols if c in affine_family_df.columns]

print(
    affine_family_df[existing_affine_view_cols]
    .to_string(index=False)
)

print("\n\n============================================================")
print("3C Closest Family Pairs in Delta-R Space")
print("============================================================\n")

print(
    family_pairwise_df
    .sort_values(by="euclidean_distance")
    .head(30)
    .to_string(index=False)
)

print("\n\n============================================================")
print("3C Farthest Family Pairs in Delta-R Space")
print("============================================================\n")

print(
    family_pairwise_df
    .sort_values(by="euclidean_distance", ascending=False)
    .head(20)
    .to_string(index=False)
)

# ============================================================
# SAVE OUTPUTS
# ============================================================

feature_table_path = SAVE_DIR / "constraint_audit3c_feature_table_with_geometry.csv"
lofo_summary_path = SAVE_DIR / "constraint_audit3c_lofo_summary.csv"
lofo_pred_path = SAVE_DIR / "constraint_audit3c_lofo_nearest_prototype_predictions.csv"
lofo_family_geo_path = SAVE_DIR / "constraint_audit3c_lofo_family_geometry.csv"
group_lr_path = SAVE_DIR / "constraint_audit3c_groupkfold_logistic_summary.csv"
family_geometry_path = SAVE_DIR / "constraint_audit3c_family_geometry.csv"
mixed_family_path = SAVE_DIR / "constraint_audit3c_family_mixture_summary.csv"
proto_coord_path = SAVE_DIR / "constraint_audit3c_full_prototype_coordinates.csv"
pca_coord_path = SAVE_DIR / "constraint_audit3c_deltaR_pca_coordinates.csv"
pairwise_path = SAVE_DIR / "constraint_audit3c_family_pairwise_distances.csv"
affine_sample_path = SAVE_DIR / "constraint_audit3c_affine_coordinates.csv"
affine_family_path = SAVE_DIR / "constraint_audit3c_affine_family_summary.csv"

df_geo.to_csv(feature_table_path, index=False)
lofo_summary_df.to_csv(lofo_summary_path, index=False)
lofo_pred_df.to_csv(lofo_pred_path, index=False)
lofo_family_geo_df.to_csv(lofo_family_geo_path, index=False)
group_lr_df.to_csv(group_lr_path, index=False)
family_geometry_df.to_csv(family_geometry_path, index=False)
mixed_df.to_csv(mixed_family_path, index=False)
proto_coord_df.to_csv(proto_coord_path, index=False)
pca_df.to_csv(pca_coord_path, index=False)
family_pairwise_df.to_csv(pairwise_path, index=False)
affine_aug_df.to_csv(affine_sample_path, index=False)
affine_family_df.to_csv(affine_family_path, index=False)

print("\nSaved outputs:")
print(" ", feature_table_path)
print(" ", lofo_summary_path)
print(" ", lofo_pred_path)
print(" ", lofo_family_geo_path)
print(" ", group_lr_path)
print(" ", family_geometry_path)
print(" ", mixed_family_path)
print(" ", proto_coord_path)
print(" ", pca_coord_path)
print(" ", pairwise_path)
print(" ", affine_sample_path)
print(" ", affine_family_path)

# ============================================================
# INTERPRETATION GUIDE
# ============================================================

print("\n\n============================================================")
print("Constraint-Audit-3C Interpretation Guide")
print("============================================================\n")

print("Main questions:")
print("  1. Does nearest-prototype geometry classify mechanism well under LOFO?")
print("  2. Are stable_surface_shift, source_claim, override, exception mixed families?")
print("  3. Do misclassified families lie between S/K/Q prototypes rather than randomly?")
print("  4. Are raw Delta-R failures geometric mixtures rather than classifier failures?")
print()
print("Strong positive result if:")
print("  - LOFO nearest-prototype mechanism macro-F1 is close to or above baseline.")
print("  - closure families cluster near closure prototype.")
print("  - competition families cluster near competition prototype.")
print("  - stable_core clusters near stable prototype.")
print("  - ambiguous families show high mixture entropy or low distance margin.")
print()
print("Important mixed-family expectations:")
print("  stable_paraphrase / stable_irrelevant:")
print("    may lie between stable and competition.")
print()
print("  competition_source_claim / competition_equal_evidence:")
print("    may lie between competition and closure or near high-risk competition.")
print()
print("  closure_override / closure_exception:")
print("    may lie between closure and competition because rule-binding is mixed.")
print()
print("Theory update if positive:")
print("  Delta-R space is not only predictive; it has prototype geometry:")
print("      DeltaR ≈ alpha*S + beta*K + gamma*Q + residual")
print()
print("If affine residual is high for override/source families:")
print("  S/K/Q coarse prototypes are insufficient; need a fourth rule-binding axis.")
print()
print("If stable_surface_shift has high mixture entropy:")
print("  surface shift is not pure stable; it occupies stable-competition boundary.")
print()
print("Done.")