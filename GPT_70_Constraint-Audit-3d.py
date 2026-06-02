# ============================================================
# Constraint-Audit-3D
# Delta-R Four-Prototype Geometry
#
# Input:
#   Prefer:
#     ./constraint_audit3c_outputs/constraint_audit3c_feature_table_with_geometry.csv
#   Fallback:
#     ./constraint_audit3b1_outputs/constraint_audit3b1_feature_table.csv
#     ./constraint_audit3b_outputs/constraint_audit3b_feature_table.csv
#
# Goal:
#   3C showed that Delta-R_20:25 forms an interpretable S/K/Q
#   prototype geometry:
#
#       S = stable
#       K = competition
#       Q = closure
#
#   But closure_override / closure_exception showed high mixture
#   entropy and high affine residual, suggesting a fourth axis:
#
#       O = rule-binding / override / exception
#
#   3D tests:
#
#       DeltaR ≈ alpha*S + beta*K + gamma*Q + delta*O + residual
#
# Main questions:
#   1. Does SKQO improve LOFO axis classification over SKQ?
#   2. Does adding O reduce affine residual for override/exception?
#   3. Does closure_override move from K/Q mixture to O?
#   4. Do source_claim / equal_evidence remain K-Q boundary cases?
#   5. Do stable_surface_shift families remain S-K boundary cases?
#
# Outputs:
#   ./constraint_audit3d_outputs/
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

INPUT_3C = Path("./constraint_audit3c_outputs/constraint_audit3c_feature_table_with_geometry.csv")
INPUT_3B1 = Path("./constraint_audit3b1_outputs/constraint_audit3b1_feature_table.csv")
INPUT_3B = Path("./constraint_audit3b_outputs/constraint_audit3b_feature_table.csv")

SAVE_DIR = Path("./constraint_audit3d_outputs")
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

# Intended three-axis geometry.
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

AXIS3_CLASSES = [
    "S_stable",
    "K_competition",
    "Q_closure",
]

# Four-axis geometry separates override/exception.
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

AXIS4_CLASSES = [
    "S_stable",
    "K_competition",
    "Q_closure",
    "O_rule_binding",
]

# Anchor families define prototypes.
# Non-anchor families are projected into this geometry.
AXIS4_ANCHOR_MAP = {
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

    "closure_override": "O_rule_binding",
    "closure_exception": "O_rule_binding",
}

AXIS3_ANCHOR_MAP = {
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
    "closure_override": "Q_closure",
    "closure_exception": "Q_closure",
}

# Diagnostic family groups.
DIAGNOSTIC_GROUP_MAP = {
    "stable_clean_basic": "S_core",
    "stable_redundant": "S_core",
    "stable_weak_note": "S_core",
    "stable_paraphrase": "S_shift_boundary",
    "stable_irrelevant": "S_shift_boundary",

    "competition_branch": "K_low",
    "competition_direct": "K_low",
    "competition_ambiguous": "K_mid",
    "competition_source_claim": "K_high_KQ",
    "competition_equal_evidence": "K_high_KQ",

    "closure_negation": "Q_canonical",
    "closure_update": "Q_canonical",
    "closure_temporal": "Q_canonical",
    "closure_authority": "Q_canonical",

    "closure_override": "O_rule_binding",
    "closure_exception": "O_rule_binding",
}

# ============================================================
# HELPERS
# ============================================================

def parse_bool_series(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def safe_name(x):
    return str(x).replace("/", "_").replace(" ", "_")


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
    proto_rows = []
    protos = []

    for cls in classes:
        anchor_fams = [fam for fam, lab in anchor_map.items() if lab == cls]
        mask = np.isin(families, anchor_fams)

        if mask.sum() == 0:
            protos.append(np.zeros(X.shape[1], dtype=np.float64))
            proto_rows.append({
                "class": cls,
                "n_anchor": 0,
                "anchor_families": ",".join(anchor_fams),
            })
        else:
            protos.append(X[mask].mean(axis=0))
            proto_rows.append({
                "class": cls,
                "n_anchor": int(mask.sum()),
                "anchor_families": ",".join(anchor_fams),
            })

    return np.vstack(protos), pd.DataFrame(proto_rows)


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
    out[f"{prefix}_dist_margin_second_minus_first"] = out[f"{prefix}_second_dist"] - out[f"{prefix}_nearest_dist"]

    w_order = np.argsort(-weights, axis=1)
    out[f"{prefix}_top_mix_class"] = np.array(classes, dtype=object)[w_order[:, 0]]
    out[f"{prefix}_top_mix_weight"] = weights[np.arange(len(X)), w_order[:, 0]]
    out[f"{prefix}_second_mix_weight"] = weights[np.arange(len(X)), w_order[:, 1]]
    out[f"{prefix}_mix_margin_top_minus_second"] = out[f"{prefix}_top_mix_weight"] - out[f"{prefix}_second_mix_weight"]
    out[f"{prefix}_mix_entropy"] = entropy_rows(weights)

    return out


def affine_coordinates_and_residual(X, proto, classes, prefix):
    p0 = proto[0]
    B = (proto[1:] - p0).T

    rows = []

    for x in X:
        y = x - p0

        if B.shape[1] == 0:
            coef = np.array([], dtype=np.float64)
            x_hat = p0.copy()
        else:
            coef, *_ = np.linalg.lstsq(B, y, rcond=None)
            x_hat = p0 + B @ coef

        weights = np.zeros(len(classes), dtype=np.float64)
        weights[0] = 1.0 - coef.sum() if len(coef) else 1.0
        if len(coef):
            weights[1:] = coef

        residual = np.linalg.norm(x - x_hat)

        row = {
            f"{prefix}_affine_residual": float(residual),
        }

        for j, cls in enumerate(classes):
            row[f"{prefix}_affine_coord_{safe_name(cls)}"] = float(weights[j])

        rows.append(row)

    return pd.DataFrame(rows)


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
            max_iter=5000,
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

if INPUT_3C.exists():
    INPUT_FILE = INPUT_3C
elif INPUT_3B1.exists():
    INPUT_FILE = INPUT_3B1
elif INPUT_3B.exists():
    INPUT_FILE = INPUT_3B
else:
    raise FileNotFoundError("Cannot find 3C / 3B.1 / 3B feature table.")

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
df["diagnostic_group"] = df["family"].map(DIAGNOSTIC_GROUP_MAP)

if df["axis4_label"].isna().any():
    bad = df[df["axis4_label"].isna()]["family"].unique().tolist()
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
print("Constraint-Audit-3D Input Summary")
print("============================================================\n")

print("Input:", INPUT_FILE)
print("Rows:", len(df))
print("Families:", df["family"].nunique())
print("Delta-R cols:", DELTA_R_COLS)

print("\nFamily summary:")
print(
    f"{'family':<32}"
    f"{'axis4':<18}"
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
        f"{AXIS4_MAP[fam]:<18}"
        f"{DIAGNOSTIC_GROUP_MAP[fam]:<22}"
        f"{COARSE_MECHANISM_MAP[fam]:<24}"
        f"{len(sub):<6}"
        f"{sub['y_conflict'].mean():<10.4f}"
    )

# ============================================================
# FULL-DATA PROTOTYPES
# ============================================================

scaler_full = StandardScaler()
X_std = scaler_full.fit_transform(X_raw)

proto3, proto3_info = build_anchor_prototypes(
    X_std,
    families,
    AXIS3_ANCHOR_MAP,
    AXIS3_CLASSES,
)

proto4, proto4_info = build_anchor_prototypes(
    X_std,
    families,
    AXIS4_ANCHOR_MAP,
    AXIS4_CLASSES,
)

feat3 = prototype_feature_frame(X_std, proto3, AXIS3_CLASSES, "SKQ")
feat4 = prototype_feature_frame(X_std, proto4, AXIS4_CLASSES, "SKQO")

aff3 = affine_coordinates_and_residual(X_std, proto3, AXIS3_CLASSES, "SKQ")
aff4 = affine_coordinates_and_residual(X_std, proto4, AXIS4_CLASSES, "SKQO")

df_geo = pd.concat(
    [
        df.reset_index(drop=True),
        feat3.reset_index(drop=True),
        feat4.reset_index(drop=True),
        aff3.reset_index(drop=True),
        aff4.reset_index(drop=True),
    ],
    axis=1,
)

df_geo["SKQO_residual_reduction_from_SKQ"] = (
    df_geo["SKQ_affine_residual"] - df_geo["SKQO_affine_residual"]
)

df_geo["SKQO_residual_reduction_ratio"] = (
    df_geo["SKQO_residual_reduction_from_SKQ"]
    / (df_geo["SKQ_affine_residual"].abs() + 1e-12)
)

# PCA for visualization coordinates.
pca = PCA(n_components=2, random_state=RANDOM_SEED)
XY = pca.fit_transform(X_std)
proto3_xy = pca.transform(proto3)
proto4_xy = pca.transform(proto4)

pca_df = df[[
    "idx",
    "family",
    "mechanism",
    "submechanism",
    "risk_regime",
    "axis3_label",
    "axis4_label",
    "diagnostic_group",
    "y_conflict",
]].copy()

pca_df["pc1"] = XY[:, 0]
pca_df["pc2"] = XY[:, 1]

proto_rows = []

for i, cls in enumerate(AXIS3_CLASSES):
    row = {
        "prototype_set": "SKQ",
        "prototype_class": cls,
        "pc1": float(proto3_xy[i, 0]),
        "pc2": float(proto3_xy[i, 1]),
    }
    for j, c in enumerate(DELTA_R_COLS):
        row[f"proto_std_{c}"] = float(proto3[i, j])
    proto_rows.append(row)

for i, cls in enumerate(AXIS4_CLASSES):
    row = {
        "prototype_set": "SKQO",
        "prototype_class": cls,
        "pc1": float(proto4_xy[i, 0]),
        "pc2": float(proto4_xy[i, 1]),
    }
    for j, c in enumerate(DELTA_R_COLS):
        row[f"proto_std_{c}"] = float(proto4[i, j])
    proto_rows.append(row)

prototype_coordinates_df = pd.DataFrame(proto_rows)

# ============================================================
# LOFO NEAREST-PROTOTYPE EVALUATION
# ============================================================

logo = LeaveOneGroupOut()

lofo_summary_rows = []
lofo_pred_rows = []
lofo_family_rows = []

eval_specs = [
    {
        "proto_set": "SKQ",
        "classes": AXIS3_CLASSES,
        "anchor_map": AXIS3_ANCHOR_MAP,
        "target_col": "axis3_label",
    },
    {
        "proto_set": "SKQO",
        "classes": AXIS4_CLASSES,
        "anchor_map": AXIS4_ANCHOR_MAP,
        "target_col": "axis4_label",
    },
]

for spec in eval_specs:
    proto_set = spec["proto_set"]
    classes = spec["classes"]
    anchor_map = spec["anchor_map"]
    target_col = spec["target_col"]

    y = df[target_col].values

    for metric in ["euclidean", "cosine"]:
        pred_all = np.empty(len(df), dtype=object)

        for train_idx, test_idx in logo.split(X_raw, y, groups=families):
            heldout = families[test_idx][0]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_raw[train_idx])
            X_test = scaler.transform(X_raw[test_idx])

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
                "diagnostic_group",
                "y_conflict",
            ]].copy()

            temp["prototype_set"] = proto_set
            temp["metric"] = metric
            temp["target_col"] = target_col
            temp["true_label"] = y[test_idx]
            temp["pred_label"] = pred

            for j, cls in enumerate(classes):
                s = safe_name(cls)
                temp[f"dist_to_{s}"] = dist[:, j]
                temp[f"mix_w_{s}"] = weights[:, j]
                temp[f"score_to_{s}"] = score[:, j]

            temp["mix_entropy"] = entropy_rows(weights)
            temp["nearest_dist"] = np.min(dist, axis=1)
            temp["dist_margin_second_minus_first"] = (
                np.sort(dist, axis=1)[:, 1] - np.sort(dist, axis=1)[:, 0]
            )

            lofo_pred_rows.append(temp)

            fam_row = {
                "prototype_set": proto_set,
                "metric": metric,
                "target_col": target_col,
                "heldout_family": heldout,
                "true_label": y[test_idx][0],
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
    ("mechanism", ["stable_or_preserved", "competition_conflict", "closure_rewrite"]),
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
# FAMILY GEOMETRY SUMMARY
# ============================================================

family_rows = []

for fam in FAMILY_ORDER:
    sub = df_geo[df_geo["family"] == fam]
    if len(sub) == 0:
        continue

    row = {
        "family": fam,
        "axis3_label": AXIS3_MAP[fam],
        "axis4_label": AXIS4_MAP[fam],
        "diagnostic_group": DIAGNOSTIC_GROUP_MAP[fam],
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

    for cls in AXIS3_CLASSES:
        s = safe_name(cls)
        row[f"SKQ_mean_dist_to_{s}"] = float(sub[f"SKQ_dist_to_{s}"].mean())
        row[f"SKQ_mean_mix_w_{s}"] = float(sub[f"SKQ_mix_w_{s}"].mean())

    for cls in AXIS4_CLASSES:
        s = safe_name(cls)
        row[f"SKQO_mean_dist_to_{s}"] = float(sub[f"SKQO_dist_to_{s}"].mean())
        row[f"SKQO_mean_mix_w_{s}"] = float(sub[f"SKQO_mix_w_{s}"].mean())

    row["SKQ_mean_mix_entropy"] = float(sub["SKQ_mix_entropy"].mean())
    row["SKQ_mean_mix_margin"] = float(sub["SKQ_mix_margin_top_minus_second"].mean())
    row["SKQ_mean_dist_margin"] = float(sub["SKQ_dist_margin_second_minus_first"].mean())
    row["SKQ_mean_affine_residual"] = float(sub["SKQ_affine_residual"].mean())

    row["SKQO_mean_mix_entropy"] = float(sub["SKQO_mix_entropy"].mean())
    row["SKQO_mean_mix_margin"] = float(sub["SKQO_mix_margin_top_minus_second"].mean())
    row["SKQO_mean_dist_margin"] = float(sub["SKQO_dist_margin_second_minus_first"].mean())
    row["SKQO_mean_affine_residual"] = float(sub["SKQO_affine_residual"].mean())

    row["mean_residual_reduction_from_SKQ_to_SKQO"] = float(sub["SKQO_residual_reduction_from_SKQ"].mean())
    row["mean_residual_reduction_ratio"] = float(sub["SKQO_residual_reduction_ratio"].mean())

    # Centroid prototype assignment.
    fam_centroid = X_std[sub.index].mean(axis=0, keepdims=True)

    c3 = prototype_feature_frame(fam_centroid, proto3, AXIS3_CLASSES, "SKQ_centroid")
    c4 = prototype_feature_frame(fam_centroid, proto4, AXIS4_CLASSES, "SKQO_centroid")

    row["SKQ_centroid_nearest_proto"] = c3["SKQ_centroid_nearest_proto"].iloc[0]
    row["SKQ_centroid_mix_entropy"] = float(c3["SKQ_centroid_mix_entropy"].iloc[0])
    row["SKQ_centroid_mix_margin"] = float(c3["SKQ_centroid_mix_margin_top_minus_second"].iloc[0])

    row["SKQO_centroid_nearest_proto"] = c4["SKQO_centroid_nearest_proto"].iloc[0]
    row["SKQO_centroid_mix_entropy"] = float(c4["SKQO_centroid_mix_entropy"].iloc[0])
    row["SKQO_centroid_mix_margin"] = float(c4["SKQO_centroid_mix_margin_top_minus_second"].iloc[0])

    family_rows.append(row)

family_geometry_df = pd.DataFrame(family_rows)

family_geometry_df["family_order"] = family_geometry_df["family"].map({
    fam: i for i, fam in enumerate(FAMILY_ORDER)
})
family_geometry_df = family_geometry_df.sort_values("family_order").drop(columns=["family_order"])

# Residual reduction summary by diagnostic group.
group_residual_rows = []

for group, sub in family_geometry_df.groupby("diagnostic_group"):
    group_residual_rows.append({
        "diagnostic_group": group,
        "n_families": int(len(sub)),
        "mean_SKQ_residual": float(sub["SKQ_mean_affine_residual"].mean()),
        "mean_SKQO_residual": float(sub["SKQO_mean_affine_residual"].mean()),
        "mean_residual_reduction": float(sub["mean_residual_reduction_from_SKQ_to_SKQO"].mean()),
        "mean_residual_reduction_ratio": float(sub["mean_residual_reduction_ratio"].mean()),
        "mean_SKQO_mix_entropy": float(sub["SKQO_mean_mix_entropy"].mean()),
        "mean_SKQO_mix_margin": float(sub["SKQO_mean_mix_margin"].mean()),
    })

group_residual_df = pd.DataFrame(group_residual_rows).sort_values(
    by="mean_residual_reduction",
    ascending=False,
)

# Families most affected by adding O.
residual_reduction_df = family_geometry_df.sort_values(
    by="mean_residual_reduction_from_SKQ_to_SKQO",
    ascending=False,
).copy()

# Most mixed in SKQO.
mixed_skqo_df = family_geometry_df.sort_values(
    by=["SKQO_mean_mix_entropy", "SKQO_mean_mix_margin"],
    ascending=[False, True],
).copy()

# ============================================================
# PAIRWISE DISTANCES INCLUDING PROTOTYPES
# ============================================================

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
            "axis4_1": AXIS4_MAP[f1],
            "axis4_2": AXIS4_MAP[f2],
            "diagnostic_group_1": DIAGNOSTIC_GROUP_MAP[f1],
            "diagnostic_group_2": DIAGNOSTIC_GROUP_MAP[f2],
            "mechanism_1": COARSE_MECHANISM_MAP[f1],
            "mechanism_2": COARSE_MECHANISM_MAP[f2],
            "submechanism_1": SUBMECHANISM_MAP[f1],
            "submechanism_2": SUBMECHANISM_MAP[f2],
            "risk_regime_1": RISK_REGIME_MAP[f1],
            "risk_regime_2": RISK_REGIME_MAP[f2],
            "euclidean_distance": float(fam_dist[i, j]),
            "cosine_similarity": float(fam_cos[i, j]),
            "same_axis4": bool(AXIS4_MAP[f1] == AXIS4_MAP[f2]),
            "same_diagnostic_group": bool(DIAGNOSTIC_GROUP_MAP[f1] == DIAGNOSTIC_GROUP_MAP[f2]),
            "same_mechanism": bool(COARSE_MECHANISM_MAP[f1] == COARSE_MECHANISM_MAP[f2]),
            "same_submechanism": bool(SUBMECHANISM_MAP[f1] == SUBMECHANISM_MAP[f2]),
            "same_risk_regime": bool(RISK_REGIME_MAP[f1] == RISK_REGIME_MAP[f2]),
        })

family_pairwise_df = pd.DataFrame(pair_rows)

# Prototype pair distances.
proto_pair_rows = []

for proto_set, proto, classes in [
    ("SKQ", proto3, AXIS3_CLASSES),
    ("SKQO", proto4, AXIS4_CLASSES),
]:
    dist = euclidean_dist_matrix(proto, proto)
    cos = cosine_sim_matrix(proto, proto)

    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            if i >= j:
                continue
            proto_pair_rows.append({
                "prototype_set": proto_set,
                "class_1": c1,
                "class_2": c2,
                "euclidean_distance": float(dist[i, j]),
                "cosine_similarity": float(cos[i, j]),
            })

prototype_pairwise_df = pd.DataFrame(proto_pair_rows)

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n\n============================================================")
print("3D LOFO Nearest-Prototype Summary")
print("============================================================\n")
print(
    lofo_summary_df
    .sort_values(by=["prototype_set", "metric"])
    .to_string(index=False)
)

print("\n\n============================================================")
print("3D GroupKFold Logistic Baseline on Raw Delta-R")
print("============================================================\n")
print(
    group_lr_df
    .sort_values(by=["target", "macro_f1"], ascending=[True, False])
    .to_string(index=False)
)

print("\n\n============================================================")
print("3D Family Geometry: SKQO Mixture")
print("============================================================\n")

view_cols = [
    "family",
    "axis4_label",
    "diagnostic_group",
    "genE_rate",
    "SKQO_centroid_nearest_proto",
    "SKQO_mean_mix_entropy",
    "SKQO_mean_mix_margin",
    "SKQO_mean_mix_w_S_stable",
    "SKQO_mean_mix_w_K_competition",
    "SKQO_mean_mix_w_Q_closure",
    "SKQO_mean_mix_w_O_rule_binding",
    "SKQ_mean_affine_residual",
    "SKQO_mean_affine_residual",
    "mean_residual_reduction_from_SKQ_to_SKQO",
]

existing_view_cols = [c for c in view_cols if c in family_geometry_df.columns]

print(
    family_geometry_df[existing_view_cols]
    .to_string(index=False)
)

print("\n\n============================================================")
print("3D Residual Reduction by Diagnostic Group")
print("============================================================\n")
print(group_residual_df.to_string(index=False))

print("\n\n============================================================")
print("3D Families Most Helped by Adding O")
print("============================================================\n")
print(
    residual_reduction_df[existing_view_cols]
    .head(16)
    .to_string(index=False)
)

print("\n\n============================================================")
print("3D Most Mixed Families Under SKQO")
print("============================================================\n")
print(
    mixed_skqo_df[existing_view_cols]
    .head(16)
    .to_string(index=False)
)

print("\n\n============================================================")
print("3D Prototype Pairwise Distances")
print("============================================================\n")
print(prototype_pairwise_df.to_string(index=False))

print("\n\n============================================================")
print("3D Closest Family Pairs")
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

feature_table_path = SAVE_DIR / "constraint_audit3d_feature_table_with_skqo_geometry.csv"

lofo_summary_path = SAVE_DIR / "constraint_audit3d_lofo_summary.csv"
lofo_pred_path = SAVE_DIR / "constraint_audit3d_lofo_predictions.csv"
lofo_family_path = SAVE_DIR / "constraint_audit3d_lofo_family_geometry.csv"

group_lr_path = SAVE_DIR / "constraint_audit3d_groupkfold_logistic_summary.csv"

family_geometry_path = SAVE_DIR / "constraint_audit3d_family_geometry.csv"
group_residual_path = SAVE_DIR / "constraint_audit3d_group_residual_reduction.csv"
residual_reduction_path = SAVE_DIR / "constraint_audit3d_residual_reduction_by_family.csv"
mixed_skqo_path = SAVE_DIR / "constraint_audit3d_mixed_families_skqo.csv"

prototype_coordinates_path = SAVE_DIR / "constraint_audit3d_prototype_coordinates.csv"
pca_path = SAVE_DIR / "constraint_audit3d_deltaR_pca_coordinates.csv"

family_pairwise_path = SAVE_DIR / "constraint_audit3d_family_pairwise_distances.csv"
prototype_pairwise_path = SAVE_DIR / "constraint_audit3d_prototype_pairwise_distances.csv"

proto3_info_path = SAVE_DIR / "constraint_audit3d_skq_anchor_info.csv"
proto4_info_path = SAVE_DIR / "constraint_audit3d_skqo_anchor_info.csv"

df_geo.to_csv(feature_table_path, index=False)

lofo_summary_df.to_csv(lofo_summary_path, index=False)
lofo_pred_df.to_csv(lofo_pred_path, index=False)
lofo_family_df.to_csv(lofo_family_path, index=False)

group_lr_df.to_csv(group_lr_path, index=False)

family_geometry_df.to_csv(family_geometry_path, index=False)
group_residual_df.to_csv(group_residual_path, index=False)
residual_reduction_df.to_csv(residual_reduction_path, index=False)
mixed_skqo_df.to_csv(mixed_skqo_path, index=False)

prototype_coordinates_df.to_csv(prototype_coordinates_path, index=False)
pca_df.to_csv(pca_path, index=False)

family_pairwise_df.to_csv(family_pairwise_path, index=False)
prototype_pairwise_df.to_csv(prototype_pairwise_path, index=False)

proto3_info.to_csv(proto3_info_path, index=False)
proto4_info.to_csv(proto4_info_path, index=False)

print("\nSaved outputs:")
print(" ", feature_table_path)
print(" ", lofo_summary_path)
print(" ", lofo_pred_path)
print(" ", lofo_family_path)
print(" ", group_lr_path)
print(" ", family_geometry_path)
print(" ", group_residual_path)
print(" ", residual_reduction_path)
print(" ", mixed_skqo_path)
print(" ", prototype_coordinates_path)
print(" ", pca_path)
print(" ", family_pairwise_path)
print(" ", prototype_pairwise_path)
print(" ", proto3_info_path)
print(" ", proto4_info_path)

# ============================================================
# INTERPRETATION GUIDE
# ============================================================

print("\n\n============================================================")
print("Constraint-Audit-3D Interpretation Guide")
print("============================================================\n")

print("Main success conditions:")
print("  1. SKQO improves LOFO axis4 classification for override/exception.")
print("  2. Adding O reduces affine residual for closure_override / closure_exception.")
print("  3. O_rule_binding receives high mixture weight for override/exception.")
print("  4. K_high families remain K-Q mixed rather than becoming O.")
print("  5. S_shift families remain S-K mixed.")
print()
print("Read key files:")
print("  constraint_audit3d_lofo_summary.csv")
print("  constraint_audit3d_family_geometry.csv")
print("  constraint_audit3d_group_residual_reduction.csv")
print("  constraint_audit3d_residual_reduction_by_family.csv")
print("  constraint_audit3d_mixed_families_skqo.csv")
print()
print("Positive interpretation:")
print("  If O reduces residual mainly for closure_override / closure_exception:")
print("    rule-binding is a real fourth structural axis.")
print()
print("  If O also absorbs source_claim / equal_evidence:")
print("    source-prior and rule-binding may share a broader evidence-binding axis.")
print()
print("  If O does not reduce residual or does not attract override families:")
print("    override is not an independent axis in Delta-R; need another observable.")
print()
print("Theory update if positive:")
print("  C_struct_20:25 = DeltaR_20:25 ≈ alpha*S + beta*K + gamma*Q + delta*O + epsilon")
print()
print("Done.")