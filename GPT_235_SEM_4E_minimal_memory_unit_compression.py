# -*- coding: utf-8 -*-
"""
GPT_217_SEM_4E_minimal_memory_unit_compression.py

SEM-4E: Minimal Memory Unit Compression
---------------------------------------

Goal:
    Find the minimal internal memory unit M_F^min for SeedFamily memory.

Background:
    SEM-4D showed a predictable seed-state transition chain:

        Seed_init        -> Seed_traj-shape       -> Seed_commit
        L0-L6               L7-L19                   L23-L25

    It also showed:
        shape state is the strongest μF retrieval index,
        shape -> commit transition is strong,
        shape mediates init -> commit.

Now SEM-4E asks:
    What is the minimal internal memory format?

Candidate MemoryUnits:
    1. shape_center_only
    2. commit_center_only
    3. shape_commit_center
    4. shape_center + radius
    5. commit_center + radius
    6. shape_commit_center + radius
    7. residual_modes / PCA basis per family
    8. compressed PCA versions of the above

Core metrics:
    - SeedFamily retrieval accuracy
    - true family similarity margin vs random family
    - compression dimension
    - effective memory compression score

Inputs:
    sem3a_outputs/
      sem3a_dataset.csv
      sem3a_features.csv

Outputs:
    sem4e_outputs/
      sem4e_config.json
      sem4e_memory_units.csv
      sem4e_memory_retrieval.csv
      sem4e_compression_curve.csv
      sem4e_radius_residual_audit.csv
      sem4e_results_summary.json

Run:
    python GPT_217_SEM_4E_minimal_memory_unit_compression.py

No model forward is needed.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


# ============================================================
# CONFIG
# ============================================================

@dataclass
class CFG:
    SEM3A_DIR: str = "./sem3a_outputs"
    OUTPUT_DIR: str = "./sem4e_outputs"

    SEM3A_DATASET: str = "sem3a_dataset.csv"
    SEM3A_FEATURES: str = "sem3a_features.csv"

    RANDOM_SEED: int = 42

    SHAPE_WINDOW: str = "L7_19"
    COMMIT_WINDOW: str = "L23_25"
    INIT_WINDOW: str = "L0_6"

    MAIN_FEATURE_FAMILY: str = "center_pca"
    FEATURE_FAMILIES: Tuple[str, ...] = ("center_pca", "scalar", "transport", "all")

    # PCA compressed dimensions for memory unit.
    PCA_DIMS: Tuple[int, ...] = (2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128)

    # How many residual modes to keep per family.
    RESIDUAL_MODE_DIMS: Tuple[int, ...] = (1, 2, 4, 8, 16)

    # Thresholds for verdict.
    STRONG_ACC: float = 0.95
    LITE_ACC: float = 0.85
    MIN_MARGIN: float = 0.25


cfg = CFG()


# ============================================================
# UTILITIES
# ============================================================

META_COLS = {
    "row_id", "row_type", "seed_family", "concept_star", "operator_star",
    "surface_id", "prompt", "operator_id", "concept_family", "operator_family",
    "question_form", "specificity_level", "surface_form", "domain_frame",
    "constraint_polarity", "expected_prompt_function"
}


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def json_dump(obj: Any, path: Path):
    def convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, float) and np.isnan(o):
            return None
        return str(o)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=convert)


def cosine_np(a, b, eps=1e-9):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def parse_window(window: str):
    a, b = window.replace("L", "").split("_")
    return int(a), int(b)


def layer_in_col(c: str, lo: int, hi: int):
    return any(f"_L{i}_" in c for i in range(lo, hi + 1))


def numeric_cols(df: pd.DataFrame):
    return [c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]


def get_cols(df: pd.DataFrame, window: str, family: str):
    lo, hi = parse_window(window)
    cols = [c for c in numeric_cols(df) if layer_in_col(c, lo, hi)]

    if family == "center_pca":
        cols = [c for c in cols if "centerPC" in c]
    elif family == "scalar":
        cols = [c for c in cols if "centerPC" not in c]
    elif family == "transport":
        cols = [c for c in cols if any(k in c for k in ["shift", "jaccard"])]
    elif family == "all":
        pass
    else:
        raise ValueError(family)
    return cols


def make_X(df: pd.DataFrame, cols: List[str]):
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    return df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32).values


def load_inputs(cfg: CFG):
    sem3a = Path(cfg.SEM3A_DIR)
    dataset = pd.read_csv(sem3a / cfg.SEM3A_DATASET)
    features = pd.read_csv(sem3a / cfg.SEM3A_FEATURES)
    if "prompt" not in features.columns:
        features = dataset.merge(features, on="row_id", how="left")
    features = features[features["row_type"] == "original_prompt"].copy().reset_index(drop=True)
    return dataset, features


# ============================================================
# STATE MATRICES
# ============================================================

def get_state_matrix(features: pd.DataFrame, window: str, family: str):
    cols = get_cols(features, window, family)
    X = make_X(features, cols)
    if X.shape[1] == 0:
        return X, cols, None
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    return Xz, cols, scaler


def get_states(features: pd.DataFrame, family: str):
    X_shape, cols_shape, _ = get_state_matrix(features, cfg.SHAPE_WINDOW, family)
    X_commit, cols_commit, _ = get_state_matrix(features, cfg.COMMIT_WINDOW, family)
    X_init, cols_init, _ = get_state_matrix(features, cfg.INIT_WINDOW, family)
    return {
        "init": (X_init, cols_init),
        "shape": (X_shape, cols_shape),
        "commit": (X_commit, cols_commit),
        "shape_commit": (np.concatenate([X_shape, X_commit], axis=1), cols_shape + cols_commit),
        "init_shape_commit": (np.concatenate([X_init, X_shape, X_commit], axis=1), cols_init + cols_shape + cols_commit),
    }


# ============================================================
# MEMORY UNIT CONSTRUCTION
# ============================================================

def build_centers(X: np.ndarray, y: np.ndarray):
    centers = {}
    radii = {}
    residuals = {}
    for fam in sorted(np.unique(y)):
        idx = np.where(y == fam)[0]
        Xf = X[idx]
        mu = Xf.mean(axis=0)
        centers[fam] = mu
        dists = np.linalg.norm(Xf - mu.reshape(1, -1), axis=1)
        radii[fam] = {
            "radius_l2_mean": float(np.mean(dists)),
            "radius_l2_p95": float(np.quantile(dists, 0.95)),
            "radius_cos_mean": float(np.mean([1.0 - cosine_np(x, mu) for x in Xf])),
        }
        residuals[fam] = Xf - mu.reshape(1, -1)
    return centers, radii, residuals


def retrieve_with_centers(X: np.ndarray, y: np.ndarray, centers: Dict[str, np.ndarray], use_radius=False, radii=None):
    fams = list(centers.keys())
    C = np.vstack([centers[f] for f in fams])

    preds = []
    margins = []
    true_gt_mean = []
    true_sims = []
    for x, true in zip(X, y):
        sims = np.array([cosine_np(x, c) for c in C])

        if use_radius and radii is not None:
            # radius-normalized score: penalize centers with large family spread
            penalties = np.array([radii[f]["radius_l2_mean"] for f in fams])
            penalties = penalties / (np.mean(penalties) + 1e-9)
            sims = sims - 0.02 * penalties

        pred = fams[int(np.argmax(sims))]
        preds.append(pred)
        true_sim = sims[fams.index(true)]
        other = np.array([s for f, s in zip(fams, sims) if f != true])
        margins.append(float(true_sim - other.mean()))
        true_gt_mean.append(float(true_sim > other.mean()))
        true_sims.append(float(true_sim))

    return {
        "nearest_family_acc": float(accuracy_score(y, preds)),
        "frac_true_gt_random_mean": float(np.mean(true_gt_mean)),
        "mean_margin_vs_random_mean": float(np.mean(margins)),
        "mean_true_similarity": float(np.mean(true_sims)),
    }


def build_memory_units(features: pd.DataFrame, cfg: CFG):
    y = features["seed_family"].astype(str).values
    rows = []

    for family in cfg.FEATURE_FAMILIES:
        states = get_states(features, family)
        for state_name, (X, cols) in states.items():
            if X.shape[1] == 0:
                continue
            centers, radii, residuals = build_centers(X, y)
            for fam, mu in centers.items():
                row = {
                    "feature_family": family,
                    "memory_type": f"{state_name}_center",
                    "state": state_name,
                    "seed_family": fam,
                    "dim": int(len(mu)),
                    "center_norm": float(np.linalg.norm(mu)),
                    "radius_l2_mean": radii[fam]["radius_l2_mean"],
                    "radius_l2_p95": radii[fam]["radius_l2_p95"],
                    "radius_cos_mean": radii[fam]["radius_cos_mean"],
                    "center_vector_json": json.dumps(mu.tolist()),
                }
                rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# MEMORY RETRIEVAL AUDIT
# ============================================================

def run_memory_retrieval(features: pd.DataFrame, cfg: CFG):
    y = features["seed_family"].astype(str).values
    rows = []

    for family in cfg.FEATURE_FAMILIES:
        states = get_states(features, family)
        for state_name, (X, cols) in states.items():
            if X.shape[1] == 0:
                continue
            centers, radii, residuals = build_centers(X, y)

            for use_radius in [False, True]:
                out = retrieve_with_centers(X, y, centers, use_radius=use_radius, radii=radii)
                rows.append({
                    "feature_family": family,
                    "memory_type": f"{state_name}_center" + ("_radius" if use_radius else ""),
                    "state": state_name,
                    "representation_dim": int(X.shape[1]),
                    "stored_dim_per_family": int(X.shape[1] + (3 if use_radius else 0)),
                    "uses_radius": int(use_radius),
                    **out,
                    "n": int(len(y)),
                })

    return pd.DataFrame(rows)


# ============================================================
# PCA COMPRESSION CURVE
# ============================================================

def pca_compressed_retrieval(X: np.ndarray, y: np.ndarray, dim: int):
    dim = min(dim, X.shape[1], X.shape[0] - 1)
    if dim < 1:
        return None
    pca = PCA(n_components=dim, random_state=cfg.RANDOM_SEED)
    Z = pca.fit_transform(X)
    centers, radii, residuals = build_centers(Z, y)
    out = retrieve_with_centers(Z, y, centers, use_radius=False)
    out["explained_variance"] = float(np.sum(pca.explained_variance_ratio_))
    return out


def run_compression_curve(features: pd.DataFrame, cfg: CFG):
    y = features["seed_family"].astype(str).values
    rows = []

    for family in cfg.FEATURE_FAMILIES:
        states = get_states(features, family)
        for state_name, (X, cols) in states.items():
            if X.shape[1] == 0:
                continue

            # full baseline
            centers, radii, residuals = build_centers(X, y)
            base = retrieve_with_centers(X, y, centers)
            rows.append({
                "feature_family": family,
                "state": state_name,
                "compression_type": "full",
                "compressed_dim": int(X.shape[1]),
                "original_dim": int(X.shape[1]),
                "compression_ratio": 1.0,
                "explained_variance": 1.0,
                **base,
                "n": int(len(y)),
            })

            for dim in cfg.PCA_DIMS:
                if dim >= X.shape[1]:
                    continue
                out = pca_compressed_retrieval(X, y, dim)
                if out is None:
                    continue
                rows.append({
                    "feature_family": family,
                    "state": state_name,
                    "compression_type": "pca_center",
                    "compressed_dim": int(dim),
                    "original_dim": int(X.shape[1]),
                    "compression_ratio": float(X.shape[1] / dim),
                    **out,
                    "n": int(len(y)),
                })

    return pd.DataFrame(rows)


# ============================================================
# RADIUS / RESIDUAL MODES AUDIT
# ============================================================

def residual_mode_retrieval(X: np.ndarray, y: np.ndarray, mode_dim: int):
    """
    Family memory = center + top residual modes.
    Query scoring = center cosine + residual subspace reconstruction bonus.
    """
    centers, radii, residuals = build_centers(X, y)
    fams = list(centers.keys())

    modes = {}
    for fam in fams:
        R = residuals[fam]
        if R.shape[0] < 2:
            modes[fam] = np.zeros((0, X.shape[1]))
            continue
        k = min(mode_dim, R.shape[0] - 1, R.shape[1])
        if k < 1:
            modes[fam] = np.zeros((0, X.shape[1]))
            continue
        pca = PCA(n_components=k, random_state=cfg.RANDOM_SEED)
        pca.fit(R)
        modes[fam] = pca.components_

    preds = []
    margins = []
    for x, true in zip(X, y):
        scores = []
        for fam in fams:
            mu = centers[fam]
            base = cosine_np(x, mu)
            Rv = x - mu
            U = modes[fam]
            if U.shape[0] > 0:
                proj = U.T @ (U @ Rv)
                recon_quality = 1.0 - (np.linalg.norm(Rv - proj) / (np.linalg.norm(Rv) + 1e-9))
            else:
                recon_quality = 0.0
            score = base + 0.10 * recon_quality
            scores.append(score)
        scores = np.array(scores)
        pred = fams[int(np.argmax(scores))]
        preds.append(pred)
        true_score = scores[fams.index(true)]
        other = np.array([s for f, s in zip(fams, scores) if f != true])
        margins.append(float(true_score - other.mean()))

    return {
        "nearest_family_acc": float(accuracy_score(y, preds)),
        "mean_margin_vs_random_mean": float(np.mean(margins)),
    }


def run_radius_residual_audit(features: pd.DataFrame, cfg: CFG):
    y = features["seed_family"].astype(str).values
    rows = []

    for family in cfg.FEATURE_FAMILIES:
        states = get_states(features, family)
        for state_name, (X, cols) in states.items():
            if X.shape[1] == 0:
                continue

            centers, radii, residuals = build_centers(X, y)
            base = retrieve_with_centers(X, y, centers, use_radius=False)
            rad = retrieve_with_centers(X, y, centers, use_radius=True, radii=radii)

            rows.append({
                "feature_family": family,
                "state": state_name,
                "variant": "center_only",
                "extra_dim": 0,
                **base,
            })
            rows.append({
                "feature_family": family,
                "state": state_name,
                "variant": "center_plus_radius",
                "extra_dim": 3,
                **rad,
            })

            for k in cfg.RESIDUAL_MODE_DIMS:
                out = residual_mode_retrieval(X, y, k)
                rows.append({
                    "feature_family": family,
                    "state": state_name,
                    "variant": "center_plus_residual_modes",
                    "extra_dim": int(k),
                    **out,
                })

    return pd.DataFrame(rows)


# ============================================================
# SUMMARY
# ============================================================

def select_minimal_passing(curve: pd.DataFrame, threshold_acc: float, threshold_margin: float):
    sub = curve[
        (curve["nearest_family_acc"] >= threshold_acc) &
        (curve["mean_margin_vs_random_mean"] >= threshold_margin)
    ].copy()
    if sub.empty:
        return {}
    sub = sub.sort_values(["compressed_dim", "nearest_family_acc", "mean_margin_vs_random_mean"], ascending=[True, False, False])
    return sub.iloc[0].to_dict()


def build_results_summary(retrieval, curve, residual):
    summary = {
        "config": asdict(cfg),
        "verdict": "UNDETERMINED",
        "best": {},
        "minimal_units": {},
        "diagnosis": "",
        "interpretation": [],
    }

    main = cfg.MAIN_FEATURE_FAMILY

    # Best retrieval
    sub = retrieval[retrieval["feature_family"] == main].copy()
    if len(sub):
        summary["best"]["retrieval"] = sub.sort_values(
            ["nearest_family_acc", "mean_margin_vs_random_mean"],
            ascending=[False, False]
        ).iloc[0].to_dict()

    # Best full state
    for state in ["shape", "commit", "shape_commit", "init_shape_commit"]:
        s = sub[sub["state"] == state] if len(sub) else pd.DataFrame()
        if len(s):
            summary["best"][f"{state}_retrieval"] = s.sort_values(
                ["nearest_family_acc", "mean_margin_vs_random_mean"],
                ascending=[False, False]
            ).iloc[0].to_dict()

    # Minimal PCA units
    cmain = curve[(curve["feature_family"] == main) & (curve["compression_type"] == "pca_center")].copy()
    summary["minimal_units"]["strong"] = select_minimal_passing(cmain, cfg.STRONG_ACC, cfg.MIN_MARGIN)
    summary["minimal_units"]["lite"] = select_minimal_passing(cmain, cfg.LITE_ACC, cfg.MIN_MARGIN)

    # Residual/radius best
    rmain = residual[residual["feature_family"] == main].copy()
    if len(rmain):
        summary["best"]["radius_residual"] = rmain.sort_values(
            ["nearest_family_acc", "mean_margin_vs_random_mean"],
            ascending=[False, False]
        ).iloc[0].to_dict()

    # Diagnosis
    shape = summary["best"].get("shape_retrieval", {})
    commit = summary["best"].get("commit_retrieval", {})
    sc = summary["best"].get("shape_commit_retrieval", {})

    shape_acc = shape.get("nearest_family_acc", np.nan)
    commit_acc = commit.get("nearest_family_acc", np.nan)
    sc_acc = sc.get("nearest_family_acc", np.nan)

    minimal_strong = summary["minimal_units"].get("strong", {})
    minimal_lite = summary["minimal_units"].get("lite", {})

    if minimal_strong:
        summary["diagnosis"] = "STRONG_COMPRESSED_MEMORY_UNIT_FOUND"
        summary["verdict"] = "PASS_STRONG_MINIMAL_MEMORY_COMPRESSION"
    elif minimal_lite:
        summary["diagnosis"] = "LITE_COMPRESSED_MEMORY_UNIT_FOUND"
        summary["verdict"] = "PASS_LITE_MINIMAL_MEMORY_COMPRESSION"
    elif np.nan_to_num(shape_acc) >= cfg.STRONG_ACC or np.nan_to_num(sc_acc) >= cfg.STRONG_ACC:
        summary["diagnosis"] = "FULL_MEMORY_UNIT_STRONG_BUT_COMPRESSION_WEAK"
        summary["verdict"] = "PASS_LITE_FULL_MEMORY_UNIT"
    elif np.nan_to_num(shape_acc) >= cfg.LITE_ACC or np.nan_to_num(sc_acc) >= cfg.LITE_ACC:
        summary["diagnosis"] = "FULL_MEMORY_UNIT_LITE"
        summary["verdict"] = "PARTIAL_PASS_MEMORY_UNIT"
    else:
        summary["diagnosis"] = "NO_MEMORY_UNIT_YET"
        summary["verdict"] = "NO_PASS_YET"

    summary["interpretation"] = [
        "MemoryUnit candidates are internal trajectory geometry centers, not natural-language prompts.",
        "shape center corresponds to L7-L19 trajectory-shaping seed memory.",
        "commit center corresponds to L23-L25 trajectory-identity memory.",
        "shape+commit tests whether memory needs both retrieval and confirmation states.",
        "PCA compression curve estimates how many dimensions are needed per family.",
        "Radius/residual modes test whether variation structure beyond the center is necessary."
    ]
    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_dir(cfg.OUTPUT_DIR)
    outdir = Path(cfg.OUTPUT_DIR)
    json_dump(asdict(cfg), outdir / "sem4e_config.json")

    dataset, features = load_inputs(cfg)

    memory_units = build_memory_units(features, cfg)
    memory_units.to_csv(outdir / "sem4e_memory_units.csv", index=False, encoding="utf-8-sig")

    retrieval = run_memory_retrieval(features, cfg)
    retrieval.to_csv(outdir / "sem4e_memory_retrieval.csv", index=False, encoding="utf-8-sig")

    curve = run_compression_curve(features, cfg)
    curve.to_csv(outdir / "sem4e_compression_curve.csv", index=False, encoding="utf-8-sig")

    residual = run_radius_residual_audit(features, cfg)
    residual.to_csv(outdir / "sem4e_radius_residual_audit.csv", index=False, encoding="utf-8-sig")

    summary = build_results_summary(retrieval, curve, residual)
    json_dump(summary, outdir / "sem4e_results_summary.json")

    print("=" * 100)
    print("SEM-4E: Minimal Memory Unit Compression")
    print("=" * 100)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nOutput files:")
    for p in [
        "sem4e_config.json",
        "sem4e_memory_units.csv",
        "sem4e_memory_retrieval.csv",
        "sem4e_compression_curve.csv",
        "sem4e_radius_residual_audit.csv",
        "sem4e_results_summary.json",
    ]:
        print("  -", outdir / p)


if __name__ == "__main__":
    main()
