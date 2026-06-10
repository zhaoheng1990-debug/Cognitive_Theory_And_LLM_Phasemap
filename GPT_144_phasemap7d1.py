# ============================================================
# PhaseMap-7D.1 Operator Local Manifold Audit
#
# Purpose:
#   7D showed PCA(op_pc1..op_pc10) is not linearly low-dimensional
#   and may be whitened / coordinate-artifact. 7D.1 therefore audits
#   local organization rather than linear spectrum.
#
# Core questions:
#   1) Does O_cont have meaningful local neighborhoods?
#   2) Are neighborhoods aligned with phase / condition / operator labels?
#   3) Can predicted operator coordinates O_hat = G(Z, momentum, layer, context)
#      preserve the same local manifold organization?
#
# Input:
#   phasemap6b1_correction_dataset.csv in current working directory.
#   If not found, tries common local output paths.
#
# Outputs:
#   phasemap7d1_outputs/
#       phasemap7d1_summary.json
#       phasemap7d1_neighbor_purity.csv
#       phasemap7d1_trustworthiness.csv
#       phasemap7d1_centroid_distances.csv
#       phasemap7d1_layer_path.csv
#       phasemap7d1_prediction_quality.csv
#       phasemap7d1_residual_by_group.csv
#       phasemap7d1_embedding_true_2d.csv
#       phasemap7d1_embedding_pred_2d.csv
#
# Run:
#   python phasemap7d1_operator_local_manifold_audit.py
# ============================================================

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.manifold import trustworthiness
from sklearn.metrics import r2_score, pairwise_distances
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# -----------------------------
# CONFIG: fixed paths
# -----------------------------
INPUT_CSV = r"C:\Windows\System32\phasemap6b1_outputs\phasemap6b1_correction_dataset.csv"
OUTPUT_DIR = r"phasemap7d1_outputs"
RANDOM_SEED = 42
N_SPLITS = 5
K_LIST = [5, 10, 20]
GEOMETRY_SAMPLE_MAX = 2000  # cap expensive local-geometry audits for speed

OP_COLS = [f"op_pc{i}" for i in range(1, 11)]

# G7 feature set from 7C: state + previous dynamics + layer + phase/condition
NUM_FEATURES = [
    "R_l", "boundary_dist_l", "spread_l", "rank_gap_l",
    "R_prev_delta", "rank_gap_prev_delta",
    "center_backtrack_init", "center_backtrack_prevprev",
    "R_velocity_reversal", "rank_gap_reversal",
    "layer",
]
CAT_FEATURES = ["phase", "condition"]
GROUP_COL = "graph_id"
LABEL_COLS = ["phase", "condition", "heuristic_operator", "operator_cluster", "layer"]

# -----------------------------
# Utilities
# -----------------------------

def resolve_input_path() -> Path:
    candidates = [
        Path(INPUT_CSV),
        Path.cwd() / INPUT_CSV,
        Path("./phasemap6b1_outputs") / INPUT_CSV,
        Path(r"C:\Windows\System32\phasemap6b1_outputs\phasemap6b1_correction_dataset.csv"),
        Path("/mnt/data/phasemap6b1_correction_dataset.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot find phasemap6b1_correction_dataset.csv. "
        "Place this script in the same folder as the CSV, or edit INPUT_CSV."
    )


def safe_onehot_encoder():
    # sklearn changed sparse -> sparse_output
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_model(num_cols, cat_cols):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", safe_onehot_encoder()),
    ])
    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    pre = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([
        ("pre", pre),
        ("ridge", Ridge(alpha=1.0, random_state=RANDOM_SEED)),
    ])


def standardize_matrix(X):
    X = np.asarray(X, dtype=float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def cv_predict_operator(df, features_num, features_cat, target_cols, group_col):
    X = df[features_num + features_cat].copy()
    Y = df[target_cols].astype(float).values
    groups = df[group_col].values if group_col in df.columns else None

    if groups is not None and len(np.unique(groups)) >= N_SPLITS:
        splitter = GroupKFold(n_splits=N_SPLITS)
        split_iter = splitter.split(X, Y, groups)
        cv_kind = "GroupKFold"
    else:
        splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        split_iter = splitter.split(X, Y)
        cv_kind = "KFold"

    pred = np.zeros_like(Y, dtype=float)
    fold_rows = []
    for fold, (tr, te) in enumerate(split_iter):
        pipe = build_model(features_num, features_cat)
        pipe.fit(X.iloc[tr], Y[tr])
        yhat = pipe.predict(X.iloc[te])
        pred[te] = yhat
        fold_rows.append({
            "fold": fold,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "r2_micro": float(r2_score(Y[te], yhat, multioutput="uniform_average")),
        })
    return pred, pd.DataFrame(fold_rows), cv_kind


def nearest_indices(X, k):
    Xs = standardize_matrix(X)
    n = Xs.shape[0]
    k_eff = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(Xs)
    dist, ind = nn.kneighbors(Xs)
    return ind[:, 1:], dist[:, 1:]


def neighbor_purity(X, labels, label_name, k_list):
    labels = np.asarray(labels)
    rows = []
    # baseline = chance same-label probability under empirical distribution
    vals, counts = np.unique(labels.astype(str), return_counts=True)
    probs = counts / counts.sum()
    baseline = float(np.sum(probs ** 2))
    for k in k_list:
        ind, _ = nearest_indices(X, k)
        neigh_labels = labels[ind]
        same = (neigh_labels == labels[:, None])
        purity = float(np.nanmean(same))
        rows.append({
            "space": None,
            "label": label_name,
            "k": k,
            "purity": purity,
            "chance_baseline": baseline,
            "lift": purity - baseline,
            "lift_ratio": purity / baseline if baseline > 0 else np.nan,
            "n_classes": int(len(vals)),
        })
    return rows


def pca_embedding(X, n=2):
    Xs = standardize_matrix(X)
    pca = PCA(n_components=n, random_state=RANDOM_SEED)
    return pca.fit_transform(Xs), pca.explained_variance_ratio_


def centroid_table(X, df, group_col):
    X = np.asarray(X, dtype=float)
    tmp = df[[group_col]].copy()
    for i in range(X.shape[1]):
        tmp[f"dim{i+1}"] = X[:, i]
    cent = tmp.groupby(group_col).mean(numeric_only=True).reset_index()
    cent["n"] = tmp.groupby(group_col).size().values
    return cent


def centroid_distance_summary(X_true, X_pred, df, group_cols):
    rows = []
    for gc in group_cols:
        if gc not in df.columns:
            continue
        ct = centroid_table(standardize_matrix(X_true), df, gc)
        cp = centroid_table(standardize_matrix(X_pred), df, gc)
        dim_cols = [c for c in ct.columns if c.startswith("dim")]
        merged = ct[[gc] + dim_cols].merge(cp[[gc] + dim_cols], on=gc, suffixes=("_true", "_pred"))
        if len(merged) < 2:
            continue
        A = merged[[f"{c}_true" for c in dim_cols]].values
        B = merged[[f"{c}_pred" for c in dim_cols]].values
        # centroid matching distance
        match_dist = np.linalg.norm(A - B, axis=1)
        # pairwise distance correlation among centroids
        DA = pairwise_distances(A)
        DB = pairwise_distances(B)
        iu = np.triu_indices_from(DA, k=1)
        corr = np.corrcoef(DA[iu], DB[iu])[0, 1] if len(iu[0]) > 1 else np.nan
        rows.append({
            "group_col": gc,
            "n_centroids": int(len(merged)),
            "mean_matching_distance": float(np.mean(match_dist)),
            "median_matching_distance": float(np.median(match_dist)),
            "pairwise_distance_corr": float(corr) if not np.isnan(corr) else np.nan,
        })
    return pd.DataFrame(rows)


def layer_path_summary(X, df, space_name):
    if "layer" not in df.columns:
        return pd.DataFrame()
    Xs = standardize_matrix(X)
    tmp = df[["layer"]].copy()
    for i in range(Xs.shape[1]):
        tmp[f"dim{i+1}"] = Xs[:, i]
    cent = tmp.groupby("layer").mean(numeric_only=True).reset_index().sort_values("layer")
    dim_cols = [c for c in cent.columns if c.startswith("dim")]
    C = cent[dim_cols].values
    rows = []
    for i, row in cent.iterrows():
        rows.append({
            "space": space_name,
            "layer": int(row["layer"]),
            "centroid_norm": float(np.linalg.norm(row[dim_cols].values.astype(float))),
        })
    out = pd.DataFrame(rows)
    # add step distance / curvature proxy
    step = [np.nan]
    for i in range(1, len(C)):
        step.append(float(np.linalg.norm(C[i] - C[i-1])))
    out["step_from_prev_layer"] = step
    return out


def trustworthiness_rows(X_orig, embed_dict, k_list):
    rows = []
    Xs = standardize_matrix(X_orig)
    for name, E in embed_dict.items():
        for k in k_list:
            k_eff = min(k, len(Xs) - 1)
            if k_eff < 2:
                val = np.nan
            else:
                val = trustworthiness(Xs, E, n_neighbors=k_eff)
            rows.append({"embedding": name, "k": k, "trustworthiness": float(val) if not np.isnan(val) else np.nan})
    return pd.DataFrame(rows)


def main():
    np.random.seed(RANDOM_SEED)
    in_path = resolve_input_path()
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    required = OP_COLS + NUM_FEATURES + CAT_FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # keep rows with operator target available
    before = len(df)
    df = df.dropna(subset=OP_COLS).reset_index(drop=True)
    after = len(df)

    O_true = df[OP_COLS].astype(float).values
    O_pred, fold_df, cv_kind = cv_predict_operator(df, NUM_FEATURES, CAT_FEATURES, OP_COLS, GROUP_COL)
    residual = O_true - O_pred

    # Expensive neighborhood/trustworthiness audits are run on a fixed sample.
    if len(df) > GEOMETRY_SAMPLE_MAX:
        geom_df = df.sample(n=GEOMETRY_SAMPLE_MAX, random_state=RANDOM_SEED).sort_index().reset_index(drop=True)
        geom_idx = geom_df.index.values  # placeholder overwritten below
        # Need original indices after sampling before reset.
        geom_idx = df.sample(n=GEOMETRY_SAMPLE_MAX, random_state=RANDOM_SEED).sort_index().index.values
    else:
        geom_idx = np.arange(len(df))
        geom_df = df.copy().reset_index(drop=True)

    O_true_g = O_true[geom_idx]
    O_pred_g = O_pred[geom_idx]
    residual_g = residual[geom_idx]
    df_g = df.iloc[geom_idx].reset_index(drop=True)

    # Prediction quality
    target_rows = []
    for j, col in enumerate(OP_COLS):
        r2 = r2_score(O_true[:, j], O_pred[:, j])
        corr = np.corrcoef(O_true[:, j], O_pred[:, j])[0, 1]
        target_rows.append({"target": col, "r2": float(r2), "corr": float(corr)})
    pred_quality = pd.DataFrame(target_rows)
    pred_quality.loc[len(pred_quality)] = {
        "target": "macro",
        "r2": float(r2_score(O_true, O_pred, multioutput="uniform_average")),
        "corr": float(np.mean([x["corr"] for x in target_rows])),
    }

    # Embeddings: PCA2/3 of true and predicted spaces
    true2, true2_var = pca_embedding(O_true_g, 2)
    pred2, pred2_var = pca_embedding(O_pred_g, 2)
    true3, true3_var = pca_embedding(O_true_g, 3)
    pred3, pred3_var = pca_embedding(O_pred_g, 3)

    emb_true_df = df_g[["sample_id", "graph_id", "condition", "phase", "layer", "heuristic_operator", "operator_cluster"]].copy()
    emb_true_df["x"] = true2[:, 0]
    emb_true_df["y"] = true2[:, 1]
    emb_true_df.to_csv(out_dir / "phasemap7d1_embedding_true_2d.csv", index=False)

    emb_pred_df = df_g[["sample_id", "graph_id", "condition", "phase", "layer", "heuristic_operator", "operator_cluster"]].copy()
    emb_pred_df["x"] = pred2[:, 0]
    emb_pred_df["y"] = pred2[:, 1]
    emb_pred_df.to_csv(out_dir / "phasemap7d1_embedding_pred_2d.csv", index=False)

    # Neighbor purity in true, predicted, residual spaces
    purity_rows = []
    for space_name, X in [("O_true", O_true_g), ("O_pred_G7", O_pred_g), ("O_residual", residual_g)]:
        for lc in LABEL_COLS:
            if lc in df.columns:
                rows = neighbor_purity(X, df_g[lc].astype(str).values, lc, K_LIST)
                for r in rows:
                    r["space"] = space_name
                purity_rows.extend(rows)
    purity_df = pd.DataFrame(purity_rows)
    purity_df.to_csv(out_dir / "phasemap7d1_neighbor_purity.csv", index=False)

    # Trustworthiness of 2D/3D PCA embeddings for true/pred
    trust_df = pd.concat([
        trustworthiness_rows(O_true_g, {"O_true_PCA2": true2, "O_true_PCA3": true3}, K_LIST),
        trustworthiness_rows(O_pred_g, {"O_pred_PCA2": pred2, "O_pred_PCA3": pred3}, K_LIST),
    ], ignore_index=True)
    trust_df.to_csv(out_dir / "phasemap7d1_trustworthiness.csv", index=False)

    # Centroid structure true vs predicted
    centroid_dist = centroid_distance_summary(O_true_g, O_pred_g, df_g, ["phase", "condition", "heuristic_operator", "operator_cluster", "layer"])
    centroid_dist.to_csv(out_dir / "phasemap7d1_centroid_distances.csv", index=False)

    # Layer trajectories
    layer_path = pd.concat([
        layer_path_summary(O_true_g, df_g, "O_true"),
        layer_path_summary(O_pred_g, df_g, "O_pred_G7"),
        layer_path_summary(residual_g, df_g, "O_residual"),
    ], ignore_index=True)
    layer_path.to_csv(out_dir / "phasemap7d1_layer_path.csv", index=False)

    # Residual by group
    res_df = df[["sample_id", "graph_id", "condition", "phase", "layer", "heuristic_operator", "operator_cluster"]].copy()
    res_norm = np.linalg.norm(standardize_matrix(residual), axis=1)
    res_df["residual_norm"] = res_norm
    res_df.to_csv(out_dir / "phasemap7d1_residuals.csv", index=False)
    group_rows = []
    for gc in ["phase", "condition", "heuristic_operator", "operator_cluster", "layer"]:
        if gc in res_df.columns:
            g = res_df.groupby(gc)["residual_norm"].agg(["count", "mean", "median", "std"]).reset_index()
            g.insert(0, "group_col", gc)
            g.rename(columns={gc: "group_value"}, inplace=True)
            group_rows.append(g)
    residual_by_group = pd.concat(group_rows, ignore_index=True)
    residual_by_group.to_csv(out_dir / "phasemap7d1_residual_by_group.csv", index=False)

    pred_quality.to_csv(out_dir / "phasemap7d1_prediction_quality.csv", index=False)
    fold_df.to_csv(out_dir / "phasemap7d1_fold_summary.csv", index=False)

    # Main readout extraction
    def get_purity(space, label, k):
        x = purity_df[(purity_df["space"] == space) & (purity_df["label"] == label) & (purity_df["k"] == k)]
        return None if x.empty else float(x.iloc[0]["purity"])
    def get_lift(space, label, k):
        x = purity_df[(purity_df["space"] == space) & (purity_df["label"] == label) & (purity_df["k"] == k)]
        return None if x.empty else float(x.iloc[0]["lift"])
    def get_trust(name, k):
        x = trust_df[(trust_df["embedding"] == name) & (trust_df["k"] == k)]
        return None if x.empty else float(x.iloc[0]["trustworthiness"])
    def get_cent_corr(gc):
        x = centroid_dist[centroid_dist["group_col"] == gc]
        return None if x.empty else float(x.iloc[0]["pairwise_distance_corr"])

    main_readout = {
        "cv_kind": cv_kind,
        "G7_operator_prediction_macro_r2": float(pred_quality[pred_quality["target"] == "macro"].iloc[0]["r2"]),
        "G7_operator_prediction_mean_corr": float(pred_quality[pred_quality["target"] == "macro"].iloc[0]["corr"]),
        "true_phase_purity_k10": get_purity("O_true", "phase", 10),
        "pred_phase_purity_k10": get_purity("O_pred_G7", "phase", 10),
        "true_operator_purity_k10": get_purity("O_true", "heuristic_operator", 10),
        "pred_operator_purity_k10": get_purity("O_pred_G7", "heuristic_operator", 10),
        "true_condition_purity_k10": get_purity("O_true", "condition", 10),
        "pred_condition_purity_k10": get_purity("O_pred_G7", "condition", 10),
        "true_phase_lift_k10": get_lift("O_true", "phase", 10),
        "pred_phase_lift_k10": get_lift("O_pred_G7", "phase", 10),
        "trust_true_pca2_k10": get_trust("O_true_PCA2", 10),
        "trust_true_pca3_k10": get_trust("O_true_PCA3", 10),
        "trust_pred_pca2_k10": get_trust("O_pred_PCA2", 10),
        "trust_pred_pca3_k10": get_trust("O_pred_PCA3", 10),
        "centroid_distance_corr_phase": get_cent_corr("phase"),
        "centroid_distance_corr_condition": get_cent_corr("condition"),
        "centroid_distance_corr_operator": get_cent_corr("heuristic_operator"),
        "n_rows_total": int(before),
        "n_rows_used": int(after),
        "n_rows_geometry_sample": int(len(df_g)),
    }

    # Conservative pass rule: local semantic organization + predictability, not PCA low dim
    manifold_local_pass = (
        main_readout["G7_operator_prediction_macro_r2"] >= 0.70 and
        (main_readout["true_phase_lift_k10"] is not None and main_readout["true_phase_lift_k10"] > 0.10) and
        (main_readout["pred_phase_lift_k10"] is not None and main_readout["pred_phase_lift_k10"] > 0.05)
    )
    main_readout["local_manifold_pass"] = bool(manifold_local_pass)

    summary = {
        "experiment": "PhaseMap-7D.1 Operator Local Manifold Organization Audit",
        "input": str(in_path),
        "outdir": str(out_dir),
        "operator_cols": OP_COLS,
        "predictor_model": "G7 = state + prev_dynamics + layer + phase/condition -> op_pc",
        "main_readout": main_readout,
        "interpretation_rules": {
            "why_7d1": "7D PCA spectrum can be inconclusive if op_pc coordinates are whitened. 7D.1 tests local neighborhood organization instead.",
            "local_purity": "High neighbor purity/lift means local neighborhoods align with phase/operator/condition labels, supporting local manifold organization.",
            "predicted_manifold": "If O_pred from pre-transition G7 preserves true local purity and centroid structure, operator coordinates are at least partly generated from state+momentum+context.",
            "centroid_distance_corr": "Correlation of pairwise centroid distances between O_true and O_pred tests whether global semantic geometry is reconstructed.",
            "caveat": "This still uses op_pc as target coordinates; a later audit should repeat on raw operator signatures before PCA/whitening if available.",
        },
        "outputs": {
            "prediction_quality": str(out_dir / "phasemap7d1_prediction_quality.csv"),
            "fold_summary": str(out_dir / "phasemap7d1_fold_summary.csv"),
            "neighbor_purity": str(out_dir / "phasemap7d1_neighbor_purity.csv"),
            "trustworthiness": str(out_dir / "phasemap7d1_trustworthiness.csv"),
            "centroid_distances": str(out_dir / "phasemap7d1_centroid_distances.csv"),
            "layer_path": str(out_dir / "phasemap7d1_layer_path.csv"),
            "residuals": str(out_dir / "phasemap7d1_residuals.csv"),
            "residual_by_group": str(out_dir / "phasemap7d1_residual_by_group.csv"),
            "embedding_true_2d": str(out_dir / "phasemap7d1_embedding_true_2d.csv"),
            "embedding_pred_2d": str(out_dir / "phasemap7d1_embedding_pred_2d.csv"),
        },
    }

    with open(out_dir / "phasemap7d1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== PhaseMap-7D.1 complete ===")
    print(json.dumps(main_readout, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
