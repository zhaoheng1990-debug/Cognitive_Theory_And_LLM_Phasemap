# ============================================================
# PhaseMap-1: 2D Basin Phase Diagram
#
# Goal:
#   Construct interpretable 2D phase diagram:
#
#       X = (U_K, D_B)
#
#   where:
#       U_K = competition intensity
#       D_B = distance to basin boundary
#
# Input:
#   ua11a_outputs/ua11a_eval.csv
#
# Outputs:
#   phasemap1_outputs/
#       phasemap1_eval.csv
#       phasemap1_phase_centroids.csv
#       phasemap1_phase_boundaries.csv
#       phasemap1_grid_map.csv
#       phasemap1_summary.json
#       phasemap1_phase_map.png
#       phasemap1_phase_map_with_points.png
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path(r"C:\Windows\System32\ua11a_outputs\ua11a_eval.csv")
SAVE_DIR = Path("./phasemap1_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
GRID_N = 250

FEATURES = ["U_K", "D_B"]
TARGET = "phase_target"

# ============================================================
# LOAD
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Cannot find input file: {INPUT_PATH.resolve()}")

df = pd.read_csv(INPUT_PATH)

required = ["U_K", "D_B", "phase_target", "condition", "Gen_E"]
for col in required:
    if col not in df.columns:
        raise RuntimeError(f"Missing column: {col}")

df = df.copy()
df["phase_target"] = df["phase_target"].astype(str)
df["Gen_E"] = df["Gen_E"].astype(int)

# ============================================================
# CROSS-VALIDATED PHASE CLASSIFICATION
# ============================================================

X = df[FEATURES].values.astype(float)
y = df[TARGET].values.astype(str)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

pred = np.empty(len(df), dtype=object)
prob_rows = []

for fold, (tr, te) in enumerate(skf.split(X, y)):
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=5000)),
    ])

    clf.fit(X[tr], y[tr])
    pred[te] = clf.predict(X[te])

    probs = clf.predict_proba(X[te])
    classes = clf.named_steps["lr"].classes_

    for i, idx in enumerate(te):
        row = {
            "row_index": int(idx),
            "fold": int(fold),
            "true_phase": y[idx],
            "pred_phase": pred[idx],
        }
        for j, cls in enumerate(classes):
            row[f"p_{cls}"] = float(probs[i, j])
        prob_rows.append(row)

acc = accuracy_score(y, pred)
macro_f1 = f1_score(y, pred, average="macro", zero_division=0)

df["pred_phase_UK_DB"] = pred
df.to_csv(SAVE_DIR / "phasemap1_eval.csv", index=False)

pd.DataFrame(prob_rows).to_csv(SAVE_DIR / "phasemap1_cv_probs.csv", index=False)

# ============================================================
# TRAIN FINAL MODEL FOR PHASE MAP
# ============================================================

final_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=5000)),
])

final_clf.fit(X, y)
classes = final_clf.named_steps["lr"].classes_

# ============================================================
# CENTROIDS
# ============================================================

centroids = df.groupby("phase_target").agg(
    n=("Gen_E", "count"),
    GenE_rate=("Gen_E", "mean"),
    U_K_mean=("U_K", "mean"),
    U_K_std=("U_K", "std"),
    D_B_mean=("D_B", "mean"),
    D_B_std=("D_B", "std"),
).reset_index()

centroids.to_csv(SAVE_DIR / "phasemap1_phase_centroids.csv", index=False)

# ============================================================
# PAIRWISE BOUNDARIES IN RAW COORDINATES
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

    Xp = pair[FEATURES].values.astype(float)
    yp = (pair["phase_target"] == b).astype(int).values

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=5000)),
    ])

    clf.fit(Xp, yp)

    scaler = clf.named_steps["scaler"]
    lr = clf.named_steps["lr"]

    coef_scaled = lr.coef_[0]
    coef_raw = coef_scaled / scaler.scale_
    intercept_raw = lr.intercept_[0] - np.sum(coef_scaled * scaler.mean_ / scaler.scale_)

    pred_pair = clf.predict(Xp)

    boundary_rows.append({
        "boundary": f"{a}_vs_{b}",
        "positive_class": b,
        "acc_full_data": float(accuracy_score(yp, pred_pair)),
        "f1_full_data": float(f1_score(yp, pred_pair, zero_division=0)),
        "coef_U_K": float(coef_raw[0]),
        "coef_D_B": float(coef_raw[1]),
        "intercept": float(intercept_raw),
        "equation": f"{coef_raw[0]:+.6f}*U_K {coef_raw[1]:+.6f}*D_B {intercept_raw:+.6f}=0",
    })

boundaries = pd.DataFrame(boundary_rows)
boundaries.to_csv(SAVE_DIR / "phasemap1_phase_boundaries.csv", index=False)

# ============================================================
# GRID PHASE MAP
# ============================================================

u_min, u_max = df["U_K"].quantile(0.01), df["U_K"].quantile(0.99)
d_min, d_max = df["D_B"].quantile(0.01), df["D_B"].quantile(0.99)

u_pad = 0.10 * (u_max - u_min)
d_pad = 0.10 * (d_max - d_min)

u_vals = np.linspace(u_min - u_pad, u_max + u_pad, GRID_N)
d_vals = np.linspace(max(0, d_min - d_pad), d_max + d_pad, GRID_N)

grid_rows = []

for uk in u_vals:
    for db in d_vals:
        point = np.array([[uk, db]], dtype=float)
        pred_phase = final_clf.predict(point)[0]
        probs = final_clf.predict_proba(point)[0]

        row = {
            "U_K": float(uk),
            "D_B": float(db),
            "pred_phase": pred_phase,
        }

        for j, cls in enumerate(classes):
            row[f"p_{cls}"] = float(probs[j])

        grid_rows.append(row)

grid_df = pd.DataFrame(grid_rows)
grid_df.to_csv(SAVE_DIR / "phasemap1_grid_map.csv", index=False)

# ============================================================
# PLOTS
# ============================================================

phase_to_num = {phase: i for i, phase in enumerate(classes)}
Z = grid_df["pred_phase"].map(phase_to_num).values.reshape(GRID_N, GRID_N).T

U_mesh, D_mesh = np.meshgrid(u_vals, d_vals)

plt.figure(figsize=(9, 7))
plt.contourf(U_mesh, D_mesh, Z, levels=len(classes), alpha=0.35)
plt.xlabel("U_K: competition intensity")
plt.ylabel("D_B: distance to basin boundary")
plt.title("PhaseMap-1: 2D Basin Phase Diagram")
plt.tight_layout()
plt.savefig(SAVE_DIR / "phasemap1_phase_map.png", dpi=180)
plt.close()

plt.figure(figsize=(9, 7))
plt.contourf(U_mesh, D_mesh, Z, levels=len(classes), alpha=0.25)

for phase in sorted(df["phase_target"].unique()):
    sub = df[df["phase_target"] == phase]
    plt.scatter(sub["U_K"], sub["D_B"], s=18, label=phase, alpha=0.75)

for _, row in centroids.iterrows():
    plt.scatter(row["U_K_mean"], row["D_B_mean"], s=180, marker="x")
    plt.text(row["U_K_mean"], row["D_B_mean"], f" {row['phase_target']}")

plt.xlabel("U_K: competition intensity")
plt.ylabel("D_B: distance to basin boundary")
plt.title("PhaseMap-1: 2D Basin Phase Diagram with Samples")
plt.legend()
plt.tight_layout()
plt.savefig(SAVE_DIR / "phasemap1_phase_map_with_points.png", dpi=180)
plt.close()

# ============================================================
# SUMMARY
# ============================================================

cm = confusion_matrix(y, pred, labels=classes)

summary = {
    "experiment": "PhaseMap-1 2D Basin Phase Diagram",
    "input_path": str(INPUT_PATH),
    "n_rows": int(len(df)),
    "features": FEATURES,
    "target": TARGET,
    "classes": classes.tolist(),
    "cv_metrics": {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
    },
    "confusion_matrix_labels": classes.tolist(),
    "confusion_matrix": cm.tolist(),
    "phase_centroids": centroids.to_dict(orient="records"),
    "pairwise_boundaries": boundaries.to_dict(orient="records"),
    "interpretation": {
        "positive": "low competition / far from boundary / stable basin",
        "critical": "high competition / close to boundary / high sensitivity",
        "negative": "committed conflict basin or wrong-basin region",
    },
}

with open(SAVE_DIR / "phasemap1_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nSaved to: {SAVE_DIR.resolve()}")