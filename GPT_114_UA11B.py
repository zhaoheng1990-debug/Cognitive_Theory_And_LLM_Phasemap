# ============================================================
# UA-11B: Phase Diagram Audit
#
# Goal:
#   Test whether (DeltaU_global, U_K, D_B) forms a low-dimensional
#   phase diagram for positive / critical / negative dynamics.
#
# Input:
#   ua11a_outputs/ua11a_eval.csv
#
# Outputs:
#   ua11b_outputs/
#       ua11b_phase_eval.csv
#       ua11b_phase_centroids.csv
#       ua11b_pairwise_boundaries.csv
#       ua11b_grid_phase_map.csv
#       ua11b_summary.json
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path( "C:/Windows/System32/ua11a_outputs/ua11a_eval.csv")
SAVE_DIR = Path("./ua11b_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

STATE_COLS = ["DeltaU_global", "U_K", "D_B"]
STATE_NONFINAL_COLS = ["DeltaU_global", "U_K", "D_B_nonfinal"]

PAIR_SPACES = {
    "DeltaU_UK": ["DeltaU_global", "U_K"],
    "DeltaU_DB": ["DeltaU_global", "D_B"],
    "UK_DB": ["U_K", "D_B"],
    "STATE_3D": STATE_COLS,
    "STATE_nonfinal_3D": STATE_NONFINAL_COLS,
}

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_PATH)

required = ["phase_target", "condition", "Gen_E"] + STATE_COLS + ["D_B_nonfinal"]

for c in required:
    if c not in df.columns:
        raise RuntimeError(f"Missing column: {c}")

df = df.copy()
df["phase_target"] = df["phase_target"].astype(str)

# ============================================================
# CV CLASSIFICATION
# ============================================================

def multiclass_cv(df, features, target="phase_target"):
    X = df[features].values.astype(float)
    y = df[target].values.astype(str)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    pred = np.empty(len(df), dtype=object)
    prob_store = []

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=5000)),
        ])

        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])

        probs = clf.predict_proba(X[te])
        classes = clf.named_steps["lr"].classes_

        for i, row_id in enumerate(te):
            r = {
                "row_index": int(row_id),
                "fold": int(fold),
                "true_phase": y[row_id],
                "pred_phase": pred[row_id],
            }
            for j, cls in enumerate(classes):
                r[f"p_{cls}"] = float(probs[i, j])
            prob_store.append(r)

    acc = accuracy_score(y, pred)
    macro_f1 = f1_score(y, pred, average="macro", zero_division=0)

    return {
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "pred": pred,
        "prob_rows": prob_store,
    }

eval_rows = []
all_pred_rows = []

for space_name, cols in PAIR_SPACES.items():
    res = multiclass_cv(df, cols)

    eval_rows.append({
        "space": space_name,
        "features": ",".join(cols),
        "acc": res["acc"],
        "macro_f1": res["macro_f1"],
    })

    for r in res["prob_rows"]:
        r["space"] = space_name
        all_pred_rows.append(r)

phase_eval = pd.DataFrame(eval_rows).sort_values("macro_f1", ascending=False)
phase_eval.to_csv(SAVE_DIR / "ua11b_phase_eval.csv", index=False)

pred_df = pd.DataFrame(all_pred_rows)
pred_df.to_csv(SAVE_DIR / "ua11b_cv_predictions.csv", index=False)

# ============================================================
# CENTROIDS / GEOMETRY
# ============================================================

centroid_rows = []

for phase, sub in df.groupby("phase_target"):
    row = {
        "phase": phase,
        "n": int(len(sub)),
        "GenE_rate": float(sub["Gen_E"].mean()),
    }

    for c in STATE_COLS + ["D_B_nonfinal"]:
        row[f"{c}_mean"] = float(sub[c].mean())
        row[f"{c}_std"] = float(sub[c].std())

    centroid_rows.append(row)

centroids = pd.DataFrame(centroid_rows)
centroids.to_csv(SAVE_DIR / "ua11b_phase_centroids.csv", index=False)

# Pairwise centroid distances
dist_rows = []
phases = list(centroids["phase"])
MEAN_STATE_COLS = [f"{c}_mean" for c in STATE_COLS]

for i in range(len(phases)):
    for j in range(i + 1, len(phases)):
        p1 = phases[i]
        p2 = phases[j]

        v1 = centroids[centroids["phase"] == p1][MEAN_STATE_COLS].iloc[0].values.astype(float)
        v2 = centroids[centroids["phase"] == p2][MEAN_STATE_COLS].iloc[0].values.astype(float)

        dist_rows.append({
            "phase_a": p1,
            "phase_b": p2,
            "euclidean_state_distance": float(np.linalg.norm(v1 - v2)),
            "DeltaU_diff": float(v1[0] - v2[0]),
            "U_K_diff": float(v1[1] - v2[1]),
            "D_B_diff": float(v1[2] - v2[2]),
        })

dist_df = pd.DataFrame(dist_rows)
dist_df.to_csv(SAVE_DIR / "ua11b_centroid_distances.csv", index=False)

# ============================================================
# LDA PHASE AXES
# ============================================================

X_state = df[STATE_COLS].values.astype(float)
y = df["phase_target"].values.astype(str)

lda = LinearDiscriminantAnalysis(n_components=2)
Z_lda = lda.fit_transform(X_state, y)

df["LDA1"] = Z_lda[:, 0]
df["LDA2"] = Z_lda[:, 1]

lda_scalings = pd.DataFrame(
    lda.scalings_[:, :2],
    index=STATE_COLS,
    columns=["LDA1_loading", "LDA2_loading"],
).reset_index().rename(columns={"index": "variable"})

lda_scalings.to_csv(SAVE_DIR / "ua11b_lda_loadings.csv", index=False)

# PCA of state space
pca = PCA(n_components=3, random_state=SEED)
Z_pca = pca.fit_transform(StandardScaler().fit_transform(X_state))

df["STATE_PC1"] = Z_pca[:, 0]
df["STATE_PC2"] = Z_pca[:, 1]
df["STATE_PC3"] = Z_pca[:, 2]

pca_loadings = pd.DataFrame(
    pca.components_.T,
    index=STATE_COLS,
    columns=["PC1", "PC2", "PC3"],
).reset_index().rename(columns={"index": "variable"})

pca_loadings.to_csv(SAVE_DIR / "ua11b_state_pca_loadings.csv", index=False)

df.to_csv(SAVE_DIR / "ua11b_state_embedding.csv", index=False)

# ============================================================
# PAIRWISE PHASE BOUNDARIES
# ============================================================

boundary_rows = []

phase_pairs = [
    ("positive", "critical"),
    ("critical", "negative"),
    ("positive", "negative"),
]

for a, b in phase_pairs:
    pair = df[df["phase_target"].isin([a, b])].copy()

    if len(pair) == 0 or pair["phase_target"].nunique() < 2:
        continue

    X = pair[STATE_COLS].values.astype(float)
    y_pair = (pair["phase_target"] == b).astype(int).values

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=5000)),
    ])

    clf.fit(X, y_pair)

    lr = clf.named_steps["lr"]
    scaler = clf.named_steps["scaler"]

    # Convert standardized coefficients back approximately:
    coef_scaled = lr.coef_[0]
    coef_raw = coef_scaled / scaler.scale_
    intercept_raw = lr.intercept_[0] - np.sum(coef_scaled * scaler.mean_ / scaler.scale_)

    acc = accuracy_score(y_pair, clf.predict(X))
    f1 = f1_score(y_pair, clf.predict(X), zero_division=0)

    boundary_rows.append({
        "boundary": f"{a}_vs_{b}",
        "positive_class": b,
        "acc_full_data": float(acc),
        "f1_full_data": float(f1),
        "coef_DeltaU_global": float(coef_raw[0]),
        "coef_U_K": float(coef_raw[1]),
        "coef_D_B": float(coef_raw[2]),
        "intercept": float(intercept_raw),
        "equation": (
            f"{coef_raw[0]:+.4f}*DeltaU "
            f"{coef_raw[1]:+.4f}*U_K "
            f"{coef_raw[2]:+.4f}*D_B "
            f"{intercept_raw:+.4f}=0"
        ),
    })

boundaries = pd.DataFrame(boundary_rows)
boundaries.to_csv(SAVE_DIR / "ua11b_pairwise_boundaries.csv", index=False)

# ============================================================
# 2D GRID MAPS
# ============================================================

# Generate grid maps for each 2D pair with the third coordinate held at global mean.
grid_rows = []

means = df[STATE_COLS].mean().to_dict()

for pair_name, cols in {
    "DeltaU_UK": ["DeltaU_global", "U_K"],
    "DeltaU_DB": ["DeltaU_global", "D_B"],
    "UK_DB": ["U_K", "D_B"],
}.items():

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=5000)),
    ])

    clf.fit(df[STATE_COLS].values.astype(float), y)

    x_col, y_col = cols

    x_min, x_max = df[x_col].quantile(0.02), df[x_col].quantile(0.98)
    y_min, y_max = df[y_col].quantile(0.02), df[y_col].quantile(0.98)

    xs = np.linspace(x_min, x_max, 60)
    ys = np.linspace(y_min, y_max, 60)

    for xv in xs:
        for yv in ys:
            point = {
                "DeltaU_global": means["DeltaU_global"],
                "U_K": means["U_K"],
                "D_B": means["D_B"],
            }

            point[x_col] = xv
            point[y_col] = yv

            Xp = np.array([[point[c] for c in STATE_COLS]], dtype=float)
            pred = clf.predict(Xp)[0]
            probs = clf.predict_proba(Xp)[0]
            classes = clf.named_steps["lr"].classes_

            row = {
                "grid_space": pair_name,
                x_col: float(xv),
                y_col: float(yv),
                "held_DeltaU_global": float(point["DeltaU_global"]),
                "held_U_K": float(point["U_K"]),
                "held_D_B": float(point["D_B"]),
                "pred_phase": pred,
            }

            for j, cls in enumerate(classes):
                row[f"p_{cls}"] = float(probs[j])

            grid_rows.append(row)

grid_df = pd.DataFrame(grid_rows)
grid_df.to_csv(SAVE_DIR / "ua11b_grid_phase_map.csv", index=False)

# ============================================================
# CONDITION SUMMARY
# ============================================================

condition_summary = df.groupby("condition").agg(
    n=("Gen_E", "count"),
    GenE_rate=("Gen_E", "mean"),
    DeltaU_mean=("DeltaU_global", "mean"),
    U_K_mean=("U_K", "mean"),
    D_B_mean=("D_B", "mean"),
    D_B_nonfinal_mean=("D_B_nonfinal", "mean"),
    LDA1_mean=("LDA1", "mean"),
    LDA2_mean=("LDA2", "mean"),
    STATE_PC1_mean=("STATE_PC1", "mean"),
    STATE_PC2_mean=("STATE_PC2", "mean"),
).reset_index()

condition_summary.to_csv(SAVE_DIR / "ua11b_condition_summary.csv", index=False)

# ============================================================
# SUMMARY
# ============================================================

summary = {
    "experiment": "UA-11B Phase Diagram Audit",
    "input_path": str(INPUT_PATH),
    "n_rows": int(len(df)),
    "state_cols": STATE_COLS,

    "phase_eval": phase_eval.to_dict(orient="records"),
    "phase_centroids": centroids.to_dict(orient="records"),
    "centroid_distances": dist_df.to_dict(orient="records"),
    "pairwise_boundaries": boundaries.to_dict(orient="records"),

    "lda_loadings": lda_scalings.to_dict(orient="records"),
    "state_pca_explained_variance": pca.explained_variance_ratio_.tolist(),
    "state_pca_loadings": pca_loadings.to_dict(orient="records"),

    "condition_summary": condition_summary.to_dict(orient="records"),

    "interpretation_rules": {
        "PASS_phase_diagram": [
            "STATE_3D or a 2D projection separates phase_target with macro-F1 clearly above single-variable baselines.",
            "Centroids form interpretable regions: positive far from boundary, critical near boundary with high U_K/DeltaU, negative near boundary with opposite sign.",
            "Pairwise boundaries have meaningful coefficients rather than being dominated only by D_B."
        ],
        "PASS_partial": [
            "STATE_3D separates critical well but Gen_E remains mostly D_B / final-readout dominated.",
            "This supports a phase diagram for dynamics, not a sufficient output predictor."
        ],
        "FAIL": [
            "Raw trajectory or signed final readout dominates all phase separation.",
            "State centroids overlap heavily.",
            "Pairwise boundaries are unstable or uninterpretable."
        ]
    }
}

with open(SAVE_DIR / "ua11b_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nSaved to: {SAVE_DIR.resolve()}")