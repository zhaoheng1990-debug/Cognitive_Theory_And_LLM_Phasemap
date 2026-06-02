import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.decomposition import PCA, FactorAnalysis, FastICA, NMF
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score

# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = r"C:\Windows\System32\ua2b_outputs\ua2b_dataset_with_Uhat.csv"
SAVE_DIR = Path("./ua2c_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DELTA_LAYERS = list(range(20, 26))
FEATURES = [f"dltR_{l}" for l in DELTA_LAYERS]

PHASE_ID = {"positive": 0, "critical": 1, "negative": 2}

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_CSV)

if "phase_id" not in df.columns:
    df["phase_id"] = df["phase"].map(PHASE_ID)

X_raw = df[FEATURES].values.astype(float)
y_phase = df["phase_id"].values
groups = df["graph_id"].values

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ============================================================
# DECOMPOSITIONS
# ============================================================

rows_var = []
rows_proj = []
rows_load = []
rows_cluster = []
rows_cls = []

methods = {
    "PCA": PCA(n_components=6, random_state=42),
    "FA": FactorAnalysis(n_components=6, random_state=42),
    "ICA": FastICA(n_components=6, random_state=42, max_iter=2000, tol=1e-4),
}

# NMF needs non-negative input
X_nmf = MinMaxScaler().fit_transform(X_raw)
methods_nmf = {
    "NMF": NMF(n_components=6, random_state=42, max_iter=2000, init="nndsvda")
}

# ============================================================
# helper
# ============================================================

def evaluate_projection(Z, method):
    for k in [1, 2, 3, 4, 5, 6]:
        Zk = Z[:, :k]

        km = KMeans(n_clusters=3, random_state=42, n_init=20)
        pred = km.fit_predict(Zk)

        ari = adjusted_rand_score(y_phase, pred)
        nmi = normalized_mutual_info_score(y_phase, pred)

        sil = np.nan
        try:
            sil = silhouette_score(Zk, pred)
        except Exception:
            pass

        rows_cluster.append({
            "method": method,
            "n_components": k,
            "ARI_phase": ari,
            "NMI_phase": nmi,
            "silhouette": sil,
        })

        # supervised sanity check: can basis coordinates predict phase?
        gkf = GroupKFold(n_splits=6)
        preds = np.zeros(len(df), dtype=int)

        for tr, te in gkf.split(Zk, y_phase, groups):
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(Zk[tr], y_phase[tr])
            preds[te] = clf.predict(Zk[te])

        rows_cls.append({
            "method": method,
            "n_components": k,
            "phase_acc": accuracy_score(y_phase, preds),
            "phase_macro_f1": f1_score(y_phase, preds, average="macro"),
        })

def save_projection(Z, method):
    for i in range(Z.shape[1]):
        df[f"{method}_U{i+1}"] = Z[:, i]

    cond_summary = (
        df.groupby(["condition", "phase"])
        .agg(
            n=("graph_id", "count"),
            **{
                f"{method}_U{i+1}_mean": (f"{method}_U{i+1}", "mean")
                for i in range(min(6, Z.shape[1]))
            },
            **{
                f"{method}_U{i+1}_std": (f"{method}_U{i+1}", "std")
                for i in range(min(6, Z.shape[1]))
            },
            crit_dltR=("target_crit_dltR_mean", "mean"),
            basin_dltR=("target_basin_dltR_mean", "mean"),
            final_R=("target_final_R", "mean"),
        )
        .reset_index()
    )

    cond_summary["method"] = method
    return cond_summary

# ============================================================
# RUN PCA / FA / ICA
# ============================================================

condition_summaries = []

for method, model in methods.items():
    Z = model.fit_transform(X)

    if method == "PCA":
        evr = model.explained_variance_ratio_
        cum = np.cumsum(evr)
        for i, v in enumerate(evr):
            rows_var.append({
                "method": method,
                "component": i + 1,
                "explained_variance_ratio": v,
                "cumulative_variance": cum[i],
            })

        loadings = model.components_
    else:
        # FA / ICA do not have PCA-style explained variance.
        loadings = getattr(model, "components_", np.full((6, len(FEATURES)), np.nan))
        for i in range(6):
            rows_var.append({
                "method": method,
                "component": i + 1,
                "explained_variance_ratio": np.nan,
                "cumulative_variance": np.nan,
            })

    for ci in range(loadings.shape[0]):
        for fj, feat in enumerate(FEATURES):
            rows_load.append({
                "method": method,
                "component": ci + 1,
                "feature": feat,
                "loading": loadings[ci, fj],
            })

    evaluate_projection(Z, method)
    condition_summaries.append(save_projection(Z, method))

# ============================================================
# RUN NMF
# ============================================================

for method, model in methods_nmf.items():
    Z = model.fit_transform(X_nmf)
    loadings = model.components_

    # reconstruction explained variance proxy
    X_rec = model.inverse_transform(Z)
    total = np.sum((X_nmf - X_nmf.mean(axis=0)) ** 2)
    resid = np.sum((X_nmf - X_rec) ** 2)
    explained = 1 - resid / total

    for i in range(6):
        rows_var.append({
            "method": method,
            "component": i + 1,
            "explained_variance_ratio": np.nan,
            "cumulative_variance": explained if i == 5 else np.nan,
        })

    for ci in range(loadings.shape[0]):
        for fj, feat in enumerate(FEATURES):
            rows_load.append({
                "method": method,
                "component": ci + 1,
                "feature": feat,
                "loading": loadings[ci, fj],
            })

    evaluate_projection(Z, method)
    condition_summaries.append(save_projection(Z, method))

# ============================================================
# SAVE
# ============================================================

df_var = pd.DataFrame(rows_var)
df_load = pd.DataFrame(rows_load)
df_cluster = pd.DataFrame(rows_cluster)
df_cls = pd.DataFrame(rows_cls)
df_cond = pd.concat(condition_summaries, ignore_index=True)

df_var.to_csv(SAVE_DIR / "ua2c_variance_explained.csv", index=False, encoding="utf-8-sig")
df_load.to_csv(SAVE_DIR / "ua2c_component_loadings.csv", index=False, encoding="utf-8-sig")
df_cluster.to_csv(SAVE_DIR / "ua2c_cluster_metrics.csv", index=False, encoding="utf-8-sig")
df_cls.to_csv(SAVE_DIR / "ua2c_phase_classification.csv", index=False, encoding="utf-8-sig")
df_cond.to_csv(SAVE_DIR / "ua2c_condition_projection.csv", index=False, encoding="utf-8-sig")
df.to_csv(SAVE_DIR / "ua2c_dataset_with_basis.csv", index=False, encoding="utf-8-sig")

with open(SAVE_DIR / "ua2c_config.json", "w", encoding="utf-8") as f:
    json.dump({
        "input_csv": INPUT_CSV,
        "features": FEATURES,
        "question": "Determine Dim(U) and whether potential basis separates positive / critical / negative phases.",
        "pass_lite": "PC1+PC2+PC3 > 0.60 or 3D basis improves phase separation.",
        "pass": "PC1+PC2+PC3 > 0.75 and phase clusters separate.",
        "pass_strong": "Stable / competition / closure-like basis directions emerge without supervision."
    }, f, ensure_ascii=False, indent=2)

print("\n========== Variance Explained ==========")
print(df_var.to_string(index=False))

print("\n========== Cluster Metrics ==========")
print(df_cluster.sort_values(["method", "n_components"]).to_string(index=False))

print("\n========== Phase Classification ==========")
print(df_cls.sort_values(["method", "n_components"]).to_string(index=False))

print("\n========== Top Loadings ==========")
print(
    df_load.assign(abs_loading=df_load["loading"].abs())
    .sort_values(["method", "component", "abs_loading"], ascending=[True, True, False])
    .groupby(["method", "component"])
    .head(3)
    .to_string(index=False)
)

print("\n[DONE] Saved to:", SAVE_DIR)