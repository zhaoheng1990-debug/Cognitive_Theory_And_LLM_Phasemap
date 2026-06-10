# ============================================================
# CM-4C: Prompt Direction Spectrum Audit
#
# Goal:
#   Fix CM-4B's failed scalar-distance criterion.
#
# CM-4B showed:
#   D(structure-change, clean) < D(relation-preserve, clean)
#
# This means magnitude-to-clean is not the correct variable.
# CM-4C tests whether clean-relative direction vectors form
# mechanism-specific directions after quotienting out the
# relation-preserving / surface subspace.
#
# Input:
#   cm4b_fast_outputs/
#       qwen_best_clean_relative_features.csv
#       llama_best_clean_relative_features.csv
#       gemma_best_clean_relative_features.csv
#       *_best_geometry.npz
#
# Output:
#   cm4c_outputs/
#       cm4c_model_summary.csv
#       cm4c_cross_model_direction_isomorphism.csv
#       cm4c_overall_summary.json
#       <model>_direction_summary.json
#       <model>_direction_vectors.csv
#       <model>_mechanism_prototype_cosine.csv
#
# Run:
#   python cm4c_prompt_direction_spectrum_audit.py
#
# ============================================================

import json
import math
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score

IN_DIR = Path("cm4b_fast_outputs")
OUT_DIR = Path("cm4c_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_KEYS = ["qwen", "llama", "gemma"]

FEATURE_COLS = [
    "rel_center_dist",
    "rel_jaccard_dist",
    "rel_weighted_jaccard_dist",
    "rel_composite_dist",
    "spread",
    "entropy",
]

# Mechanism labels from CM-4B.
MECH_MAP3 = {
    "stable": 0,
    "stable_shift": 0,
    "competition": 1,
    "closure": 2,
}

MECH_NAME = {0: "stable_like", 1: "competition", 2: "closure"}

REL_PRESERVE_CONDS = ["rename", "permuted", "redundant", "irrelevant", "paraphrase"]
STRUCTURE_CONDS = [
    "weak_distractor",
    "competition_balanced",
    "direct_conflict",
    "closure_update",
    "closure_override",
    "exception_override",
]

def safe_auc(y, score):
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y, score))
    except Exception:
        return np.nan

def normalize_rows(X):
    X = np.asarray(X, dtype=np.float64)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

def cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))

def projection_residual(X, basis):
    """
    Remove projection onto row-orthonormal basis.
    basis shape: [r, d]
    """
    if basis is None or len(basis) == 0:
        return X.copy()
    return X - (X @ basis.T) @ basis

def build_surface_basis(X_preserve, variance_threshold=0.90, max_components=3):
    """
    Estimate surface/noise subspace from relation-preserving clean-relative vectors.
    """
    Xp = np.asarray(X_preserve, dtype=np.float64)
    if Xp.shape[0] < 3:
        return np.zeros((0, Xp.shape[1]), dtype=np.float64), 0, []
    Xp_centered = Xp - Xp.mean(axis=0, keepdims=True)
    ncomp = min(max_components, Xp_centered.shape[0], Xp_centered.shape[1])
    pca = PCA(n_components=ncomp)
    pca.fit(Xp_centered)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    r = int(np.searchsorted(cumsum, variance_threshold) + 1)
    r = max(1, min(r, ncomp))
    return pca.components_[:r], r, pca.explained_variance_ratio_.tolist()

def prototype_classifier_cv(X, y, groups):
    """
    Leave-graph-group-out prototype classifier.
    Uses cosine to training centroids.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    X = normalize_rows(X)

    preds = np.zeros_like(y)
    scores_closure = np.zeros(len(y), dtype=float)
    scores_comp = np.zeros(len(y), dtype=float)

    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2 or len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan, np.nan

    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        classes = np.unique(y[tr])
        centroids = {}
        for c in classes:
            centroids[c] = normalize_rows(X[tr][y[tr] == c].mean(axis=0, keepdims=True))[0]

        for idx in te:
            sims = {c: cosine(X[idx], centroids[c]) for c in classes}
            pred = max(sims.items(), key=lambda kv: kv[1])[0]
            preds[idx] = pred
            scores_closure[idx] = sims.get(2, -1.0)
            scores_comp[idx] = sims.get(1, -1.0)

    acc = float(accuracy_score(y, preds))
    f1 = float(f1_score(y, preds, average="macro", zero_division=0))
    auc_closure = safe_auc((y == 2).astype(int), scores_closure)
    auc_comp = safe_auc((y == 1).astype(int), scores_comp)
    return acc, f1, auc_closure, auc_comp

def logistic_cv(X, y, groups):
    y = np.asarray(y)
    groups = np.asarray(groups)
    X = np.asarray(X, dtype=np.float64)
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2 or len(np.unique(y)) < 2:
        return np.nan, np.nan
    preds = np.zeros_like(y)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        clf.fit(X[tr], y[tr])
        preds[te] = clf.predict(X[te])
    return float(accuracy_score(y, preds)), float(f1_score(y, preds, average="macro", zero_division=0))

def ridge_deltaU_cv(X, y, groups):
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    X = np.asarray(X, dtype=float)
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        return np.nan, np.nan
    pred = np.zeros_like(y)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        reg = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0)),
        ])
        reg.fit(X[tr], y[tr])
        pred[te] = reg.predict(X[te])
    r2 = float(r2_score(y, pred))
    corr = float(np.corrcoef(y, pred)[0, 1]) if np.std(pred) > 1e-8 else 0.0
    return r2, corr

def process_model(model_key):
    path = IN_DIR / f"{model_key}_best_clean_relative_features.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    for c in FEATURE_COLS:
        if c not in df.columns:
            raise RuntimeError(f"{model_key}: missing feature col {c}")

    # Exclude clean because clean-relative vector is zero.
    df2 = df[df["condition"] != "clean"].copy().reset_index(drop=True)

    X_raw = df2[FEATURE_COLS].values.astype(float)
    # Standardize before direction-space operations so dimensions are comparable.
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_raw)

    # Relation-preserving subspace = surface/noise quotient.
    preserve_mask = df2["condition"].isin(REL_PRESERVE_CONDS).values
    basis, rank, surface_var = build_surface_basis(X_std[preserve_mask], variance_threshold=0.90, max_components=3)
    X_res = projection_residual(X_std, basis)

    # Direction normalize.
    X_dir = normalize_rows(X_std)
    X_res_dir = normalize_rows(X_res)

    y3 = df2["mechanism"].map(MECH_MAP3).values.astype(int)
    groups = df2["graph_id"].values.astype(int)
    y_closure = (y3 == 2).astype(int)
    y_comp = (y3 == 1).astype(int)

    # Prototype classification before and after quotient.
    proto_acc_raw, proto_f1_raw, proto_auc_closure_raw, proto_auc_comp_raw = prototype_classifier_cv(X_dir, y3, groups)
    proto_acc_res, proto_f1_res, proto_auc_closure_res, proto_auc_comp_res = prototype_classifier_cv(X_res_dir, y3, groups)

    # Logistic check.
    log_acc_raw, log_f1_raw = logistic_cv(X_std, y3, groups)
    log_acc_res, log_f1_res = logistic_cv(X_res, y3, groups)

    # Downstream DeltaU prediction.
    du = df2["DeltaU"].values.astype(float)
    du_r2_raw, du_corr_raw = ridge_deltaU_cv(X_std, du, groups)
    du_r2_res, du_corr_res = ridge_deltaU_cv(X_res, du, groups)

    # Mechanism centroids and cosine geometry.
    proto_rows = []
    proto_vectors = {}
    for label in sorted(np.unique(y3)):
        vec = X_res_dir[y3 == label].mean(axis=0)
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        proto_vectors[int(label)] = vec
        proto_rows.append({
            "model_key": model_key,
            "mechanism_id": int(label),
            "mechanism_name": MECH_NAME[int(label)],
            **{f"v{i}": float(vec[i]) for i in range(len(vec))}
        })

    cos_rows = []
    for a, b in combinations(sorted(proto_vectors), 2):
        cos_rows.append({
            "model_key": model_key,
            "mech_a": MECH_NAME[a],
            "mech_b": MECH_NAME[b],
            "cosine_after_surface_quotient": cosine(proto_vectors[a], proto_vectors[b]),
        })
    cos_df = pd.DataFrame(cos_rows)
    cos_df.to_csv(OUT_DIR / f"{model_key}_mechanism_prototype_cosine.csv", index=False, encoding="utf-8-sig")

    # Per-condition mean direction signature.
    cond_rows = []
    for (mech, cond), sub in df2.groupby(["mechanism", "condition"]):
        idx = sub.index.values
        vec = X_res_dir[idx].mean(axis=0)
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        cond_rows.append({
            "model_key": model_key,
            "mechanism": mech,
            "condition": cond,
            "n": len(sub),
            "mean_norm_raw": float(np.mean(np.linalg.norm(X_std[idx], axis=1))),
            "mean_norm_residual": float(np.mean(np.linalg.norm(X_res[idx], axis=1))),
            "mean_DeltaU": float(np.mean(sub["DeltaU"].values)),
            **{f"dir_{c}": float(v) for c, v in zip(FEATURE_COLS, vec)}
        })
    cond_df = pd.DataFrame(cond_rows)
    cond_df.to_csv(OUT_DIR / f"{model_key}_direction_vectors.csv", index=False, encoding="utf-8-sig")

    summary = {
        "model_key": model_key,
        "n_samples_nonclean": int(len(df2)),
        "surface_basis_rank": int(rank),
        "surface_basis_explained_variance": surface_var,
        "prototype_acc_raw": proto_acc_raw,
        "prototype_macro_f1_raw": proto_f1_raw,
        "prototype_auc_closure_raw": proto_auc_closure_raw,
        "prototype_auc_competition_raw": proto_auc_comp_raw,
        "prototype_acc_after_surface_quotient": proto_acc_res,
        "prototype_macro_f1_after_surface_quotient": proto_f1_res,
        "prototype_auc_closure_after_surface_quotient": proto_auc_closure_res,
        "prototype_auc_competition_after_surface_quotient": proto_auc_comp_res,
        "logistic_acc_raw": log_acc_raw,
        "logistic_macro_f1_raw": log_f1_raw,
        "logistic_acc_after_surface_quotient": log_acc_res,
        "logistic_macro_f1_after_surface_quotient": log_f1_res,
        "downstream_deltaU_r2_raw": du_r2_raw,
        "downstream_deltaU_corr_raw": du_corr_raw,
        "downstream_deltaU_r2_after_surface_quotient": du_r2_res,
        "downstream_deltaU_corr_after_surface_quotient": du_corr_res,
    }

    with open(OUT_DIR / f"{model_key}_direction_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary, cond_df

def cross_model_direction_isomorphism(cond_dfs):
    """
    Compare condition-level direction geometry across models.
    Since direction vectors live in the same engineered feature coordinate names,
    compare pairwise condition cosine-distance matrices.
    """
    rows = []
    mats = {}

    for mk, df in cond_dfs.items():
        keys = (df["mechanism"].astype(str) + "::" + df["condition"].astype(str)).values
        dir_cols = [c for c in df.columns if c.startswith("dir_")]
        X = df[dir_cols].values.astype(float)
        X = normalize_rows(X)
        cosmat = X @ X.T
        dist = 1.0 - cosmat
        mats[mk] = (keys, dist)

    def upper(M):
        return M[np.triu_indices_from(M, k=1)]

    def rank_corr(x, y):
        xr = pd.Series(x).rank().values
        yr = pd.Series(y).rank().values
        if np.std(xr) < 1e-8 or np.std(yr) < 1e-8:
            return np.nan
        return float(np.corrcoef(xr, yr)[0, 1])

    for a, b in combinations(MODEL_KEYS, 2):
        keys_a, mat_a = mats[a]
        keys_b, mat_b = mats[b]
        # They should have same condition rows; align just in case.
        common = [k for k in keys_a if k in set(keys_b)]
        ia = [list(keys_a).index(k) for k in common]
        ib = [list(keys_b).index(k) for k in common]
        A = mat_a[np.ix_(ia, ia)]
        B = mat_b[np.ix_(ib, ib)]
        va, vb = upper(A), upper(B)
        rows.append({
            "model_a": a,
            "model_b": b,
            "n_conditions": len(common),
            "pearson_condition_direction_distance_corr": float(np.corrcoef(va, vb)[0, 1]) if np.std(va) > 1e-8 and np.std(vb) > 1e-8 else np.nan,
            "spearman_condition_direction_distance_corr": rank_corr(va, vb),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "cm4c_cross_model_direction_isomorphism.csv", index=False, encoding="utf-8-sig")
    return out

def main():
    model_summaries = []
    cond_dfs = {}

    for mk in MODEL_KEYS:
        summary, cond_df = process_model(mk)
        model_summaries.append(summary)
        cond_dfs[mk] = cond_df

    model_summary_df = pd.DataFrame(model_summaries)
    model_summary_df.to_csv(OUT_DIR / "cm4c_model_summary.csv", index=False, encoding="utf-8-sig")

    cross = cross_model_direction_isomorphism(cond_dfs)

    overall = {
        "audit": "CM-4C Prompt Direction Spectrum Audit",
        "input_dir": str(IN_DIR),
        "models": model_summaries,
        "cross_model_direction_isomorphism": cross.to_dict(orient="records"),
        "interpretation": (
            "Tests whether clean-relative prompt initialization vectors become mechanism-separable "
            "after quotienting the relation-preserving/surface subspace. "
            "This audits direction-spectrum separation rather than scalar distance-to-clean."
        )
    }
    with open(OUT_DIR / "cm4c_overall_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    print("\nCM-4C complete.")
    print(model_summary_df)
    print("\nCross-model direction isomorphism:")
    print(cross)

if __name__ == "__main__":
    main()
