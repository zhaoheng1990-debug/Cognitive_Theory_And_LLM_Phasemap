# ============================================================
# OA-5B v2: Local Tangent Plane Audit
# Fixed-path version
#
# Goal:
#   Test whether shallow TopK identity neighborhoods from OA-5A
#   form a local low-dimensional tangent plane for the same
#   topic+task paraphrase family.
#
# Core tests:
#   1) Same-task local PCA explains most within-family variance.
#   2) A held-out paraphrase is reconstructed better by its own
#      task plane than by mismatched planes.
#   3) Own-plane projection energy is higher than:
#        - same entity / different task planes
#        - different entity / same task planes
#        - random mismatched planes
#
# Inputs expected from OA-5A:
#   oa5a_topk_records.csv
#   oa5a_prompt_dataset.csv
#
# Main outputs:
#   oa5b_v2_outputs/oa5b_v2_tangent_lowdim_summary.csv
#   oa5b_v2_outputs/oa5b_v2_projection_records.csv
#   oa5b_v2_outputs/oa5b_v2_projection_summary.csv
#   oa5b_v2_outputs/oa5b_v2_verdict_summary.csv
# ============================================================

from __future__ import annotations

import ast
import math
import random
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ============================================================
# PATH CONFIG - edit BASE_DIR only if needed
# ============================================================

BASE_DIR = Path(r"C:\Users\ZH\Desktop\AGI\python_script\tangent_audit_oa5a_outputs")

INPUT_DIR = BASE_DIR
OUTPUT_DIR = BASE_DIR / "oa5b_v2_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOPK_FILE = INPUT_DIR / "oa5a_topk_records.csv"
DATASET_FILE = INPUT_DIR / "oa5a_prompt_dataset.csv"

# ============================================================
# EXPERIMENT CONFIG
# ============================================================

RANDOM_SEED = 42

# Shallow initialization window. OA-5A usually includes emb/L0-L6.
# If your OA-5A records have only numeric layer_index, this will use 0..6.
SHALLOW_LAYERS = list(range(0, 7))

# Tangent plane dimensions to test.
TANGENT_DIMS = [1, 2, 3]

# Minimum paraphrases in one topic+task group.
MIN_GROUP_SIZE = 4

# Limit mismatched planes per held-out sample to keep runtime small.
# None means use all candidates.
MAX_MISMATCH_PLANES = None

# Normalize binary TopK vector by sqrt(k). This makes distances comparable
# across k values and turns dot product into overlap-like similarity.
NORMALIZE_BY_SQRT_K = True

# Verdict thresholds.
PASS_STRONG_OWN_GT_MISMATCH = 0.80
PASS_LITE_OWN_GT_MISMATCH = 0.65
PASS_STRONG_ENERGY_LIFT = 1.15
PASS_LITE_ENERGY_LIFT = 1.05
PASS_LOW_DIM_VAR90 = 0.85

# ============================================================
# UTILITIES
# ============================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def parse_topk_ids(x) -> List[int]:
    """Parse topk_ids stored as a list-like string."""
    if isinstance(x, list):
        return [int(v) for v in x]
    if isinstance(x, tuple):
        return [int(v) for v in x]
    if isinstance(x, np.ndarray):
        return [int(v) for v in x.tolist()]
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    try:
        obj = ast.literal_eval(s)
        return [int(v) for v in obj]
    except Exception:
        # Fallback for strings like "1 2 3" or "1,2,3"
        s = s.replace("[", " ").replace("]", " ").replace(",", " ")
        return [int(v) for v in s.split() if v.strip().lstrip("-").isdigit()]


def binary_matrix_from_records(records: pd.DataFrame, vocab: Sequence[int], k: int) -> np.ndarray:
    """Build dense binary incidence matrix over the provided local vocab."""
    index = {tok: j for j, tok in enumerate(vocab)}
    X = np.zeros((len(records), len(vocab)), dtype=np.float32)
    scale = math.sqrt(k) if NORMALIZE_BY_SQRT_K and k > 0 else 1.0

    for row_i, ids in enumerate(records["topk_ids_list"].values):
        for tok in ids:
            j = index.get(tok)
            if j is not None:
                X[row_i, j] = 1.0 / scale
    return X


def fit_pca_plane(X_train: np.ndarray, dim: int) -> Tuple[np.ndarray, PCA | None]:
    """Return mean and PCA plane. Handles tiny / degenerate groups."""
    mu = X_train.mean(axis=0, keepdims=True)
    Xc = X_train - mu
    max_dim = min(dim, X_train.shape[0] - 1, X_train.shape[1])
    if max_dim <= 0 or float(np.linalg.norm(Xc)) < 1e-12:
        return mu, None
    pca = PCA(n_components=max_dim, svd_solver="full", random_state=RANDOM_SEED)
    pca.fit(Xc)
    return mu, pca


def reconstruct_error_and_energy(x: np.ndarray, mu: np.ndarray, pca: PCA | None) -> Tuple[float, float, float]:
    """Return reconstruction error, baseline mean error, projection energy."""
    xc = x.reshape(1, -1) - mu
    baseline = float(np.linalg.norm(xc))
    if baseline < 1e-12 or pca is None:
        return baseline, baseline, 0.0

    z = pca.transform(xc)
    xr = pca.inverse_transform(z)
    err = float(np.linalg.norm(xc - xr))
    # Energy fraction captured relative to mean-only baseline.
    energy = 1.0 - (err * err) / max(baseline * baseline, 1e-12)
    return err, baseline, float(energy)


def local_vocab_for_two_groups(a: pd.DataFrame, b: pd.DataFrame) -> List[int]:
    toks = set()
    for ids in a["topk_ids_list"].values:
        toks.update(ids)
    for ids in b["topk_ids_list"].values:
        toks.update(ids)
    return sorted(toks)


def plane_eval_for_test(
    train_group: pd.DataFrame,
    test_row: pd.Series,
    dim: int,
    k: int,
) -> Tuple[float, float, float]:
    """Fit plane on train_group, evaluate test_row in union vocabulary."""
    test_df = pd.DataFrame([test_row])
    vocab = local_vocab_for_two_groups(train_group, test_df)
    X_train = binary_matrix_from_records(train_group, vocab, k)
    X_test = binary_matrix_from_records(test_df, vocab, k)
    mu, pca = fit_pca_plane(X_train, dim)
    err, base, energy = reconstruct_error_and_energy(X_test[0], mu, pca)
    return err, base, energy


def summarize_numeric(df: pd.DataFrame, cols: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for c in cols:
        if c in df.columns:
            out[f"{c}_mean"] = float(df[c].mean())
            out[f"{c}_median"] = float(df[c].median())
            out[f"{c}_std"] = float(df[c].std(ddof=0))
    return out

# ============================================================
# LOAD DATA
# ============================================================


def load_oa5a_records() -> pd.DataFrame:
    print("=" * 72)
    print("OA-5B v2 Local Tangent Plane Audit")
    print("=" * 72)
    print("Input TOPK_FILE :", TOPK_FILE)
    print("Input DATASET   :", DATASET_FILE)
    print("Output dir      :", OUTPUT_DIR)

    if not TOPK_FILE.exists():
        raise FileNotFoundError(f"Missing OA-5A topk records: {TOPK_FILE}")
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Missing OA-5A prompt dataset: {DATASET_FILE}")

    df = pd.read_csv(TOPK_FILE)
    required = {"prompt_id", "topic", "task", "paraphrase_id", "layer_index", "k", "topk_ids"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {TOPK_FILE}: {sorted(missing)}")

    df = df.copy()
    df["topk_ids_list"] = df["topk_ids"].apply(parse_topk_ids)
    df = df[df["topk_ids_list"].map(len) > 0].reset_index(drop=True)
    df["group_id"] = df["topic"].astype(str) + "::" + df["task"].astype(str)

    # Keep shallow layers only.
    df = df[df["layer_index"].isin(SHALLOW_LAYERS)].reset_index(drop=True)

    print("Loaded rows:", len(df))
    print("Topics:", sorted(df["topic"].unique().tolist()))
    print("Tasks :", sorted(df["task"].unique().tolist()))
    print("Layers:", sorted(df["layer_index"].unique().tolist()))
    print("K values:", sorted(df["k"].unique().tolist()))
    return df

# ============================================================
# LOW-DIMENSIONALITY AUDIT
# ============================================================


def run_lowdim_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (k, layer, topic, task), g in df.groupby(["k", "layer_index", "topic", "task"]):
        if len(g) < MIN_GROUP_SIZE:
            continue
        vocab = sorted({tok for ids in g["topk_ids_list"].values for tok in ids})
        X = binary_matrix_from_records(g, vocab, int(k))
        Xc = X - X.mean(axis=0, keepdims=True)
        total = float(np.sum(Xc * Xc))

        if total < 1e-12 or X.shape[0] < 2:
            evr = np.zeros(min(X.shape[0], X.shape[1]), dtype=np.float32)
        else:
            ncomp = min(X.shape[0] - 1, X.shape[1])
            pca = PCA(n_components=ncomp, svd_solver="full", random_state=RANDOM_SEED)
            pca.fit(Xc)
            evr = pca.explained_variance_ratio_

        cum = np.cumsum(evr) if len(evr) else np.array([0.0])
        dim90 = int(np.searchsorted(cum, 0.90) + 1) if len(evr) else 0
        dim95 = int(np.searchsorted(cum, 0.95) + 1) if len(evr) else 0

        rows.append({
            "k": int(k),
            "layer_index": int(layer),
            "topic": topic,
            "task": task,
            "group_id": f"{topic}::{task}",
            "n": int(len(g)),
            "local_vocab_size": int(len(vocab)),
            "pc1": float(evr[0]) if len(evr) > 0 else 0.0,
            "pc1_pc2": float(cum[min(1, len(cum) - 1)]) if len(cum) > 0 else 0.0,
            "pc1_pc2_pc3": float(cum[min(2, len(cum) - 1)]) if len(cum) > 0 else 0.0,
            "effective_dim_90": dim90,
            "effective_dim_95": dim95,
        })
    return pd.DataFrame(rows)

# ============================================================
# LOCAL TANGENT PLANE PROJECTION AUDIT
# ============================================================


def run_projection_audit(df: pd.DataFrame) -> pd.DataFrame:
    all_rows = []

    for (k, layer), sub in df.groupby(["k", "layer_index"]):
        print(f"\nProjection audit: k={k}, layer={layer}, rows={len(sub)}")

        group_map = {gid: g.reset_index(drop=True) for gid, g in sub.groupby("group_id") if len(g) >= MIN_GROUP_SIZE}
        group_meta = {}
        for gid, g in group_map.items():
            group_meta[gid] = {
                "topic": str(g["topic"].iloc[0]),
                "task": str(g["task"].iloc[0]),
            }

        gids = sorted(group_map.keys())
        if len(gids) < 3:
            continue

        for dim in TANGENT_DIMS:
            for gid in gids:
                g = group_map[gid]
                topic = group_meta[gid]["topic"]
                task = group_meta[gid]["task"]

                same_entity_diff_task = [
                    x for x in gids
                    if x != gid and group_meta[x]["topic"] == topic and group_meta[x]["task"] != task
                ]
                diff_entity_same_task = [
                    x for x in gids
                    if x != gid and group_meta[x]["topic"] != topic and group_meta[x]["task"] == task
                ]
                random_other = [x for x in gids if x != gid]

                if MAX_MISMATCH_PLANES is not None:
                    same_entity_diff_task = random.sample(
                        same_entity_diff_task,
                        min(MAX_MISMATCH_PLANES, len(same_entity_diff_task)),
                    )
                    diff_entity_same_task = random.sample(
                        diff_entity_same_task,
                        min(MAX_MISMATCH_PLANES, len(diff_entity_same_task)),
                    )
                    random_other = random.sample(
                        random_other,
                        min(MAX_MISMATCH_PLANES, len(random_other)),
                    )

                for test_idx in range(len(g)):
                    test_row = g.iloc[test_idx]
                    train_own = g.drop(index=test_idx).reset_index(drop=True)
                    if len(train_own) < MIN_GROUP_SIZE - 1:
                        continue

                    own_err, own_base, own_energy = plane_eval_for_test(train_own, test_row, dim, int(k))

                    def eval_candidates(cands: Sequence[str]) -> Tuple[float, float, float, float]:
                        errs, bases, energies = [], [], []
                        for cg in cands:
                            e, b, en = plane_eval_for_test(group_map[cg], test_row, dim, int(k))
                            errs.append(e)
                            bases.append(b)
                            energies.append(en)
                        if not errs:
                            return np.nan, np.nan, np.nan, np.nan
                        return (
                            float(np.mean(errs)),
                            float(np.min(errs)),
                            float(np.mean(energies)),
                            float(np.max(energies)),
                        )

                    se_err_mean, se_err_min, se_energy_mean, se_energy_max = eval_candidates(same_entity_diff_task)
                    ds_err_mean, ds_err_min, ds_energy_mean, ds_energy_max = eval_candidates(diff_entity_same_task)
                    ro_err_mean, ro_err_min, ro_energy_mean, ro_energy_max = eval_candidates(random_other)

                    all_rows.append({
                        "k": int(k),
                        "layer_index": int(layer),
                        "tangent_dim": int(dim),
                        "topic": topic,
                        "task": task,
                        "group_id": gid,
                        "prompt_id": test_row["prompt_id"],
                        "paraphrase_id": test_row["paraphrase_id"],
                        "own_error": own_err,
                        "own_baseline": own_base,
                        "own_energy": own_energy,
                        "same_entity_diff_task_error_mean": se_err_mean,
                        "same_entity_diff_task_error_min": se_err_min,
                        "same_entity_diff_task_energy_mean": se_energy_mean,
                        "same_entity_diff_task_energy_max": se_energy_max,
                        "diff_entity_same_task_error_mean": ds_err_mean,
                        "diff_entity_same_task_error_min": ds_err_min,
                        "diff_entity_same_task_energy_mean": ds_energy_mean,
                        "diff_entity_same_task_energy_max": ds_energy_max,
                        "random_other_error_mean": ro_err_mean,
                        "random_other_error_min": ro_err_min,
                        "random_other_energy_mean": ro_energy_mean,
                        "random_other_energy_max": ro_energy_max,
                    })

    proj = pd.DataFrame(all_rows)
    if proj.empty:
        return proj

    # Derived comparisons.
    for prefix in ["same_entity_diff_task", "diff_entity_same_task", "random_other"]:
        proj[f"own_better_than_{prefix}_mean"] = proj["own_error"] < proj[f"{prefix}_error_mean"]
        proj[f"own_better_than_{prefix}_min"] = proj["own_error"] < proj[f"{prefix}_error_min"]
        proj[f"energy_lift_vs_{prefix}_mean"] = (
            (proj["own_energy"] + 1e-9) / (proj[f"{prefix}_energy_mean"] + 1e-9)
        )
        proj[f"error_ratio_vs_{prefix}_mean"] = (
            (proj[f"{prefix}_error_mean"] + 1e-9) / (proj["own_error"] + 1e-9)
        )

    return proj


def summarize_projection(proj: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if proj.empty:
        return pd.DataFrame()

    group_cols = ["k", "layer_index", "tangent_dim"]
    for keys, g in proj.groupby(group_cols):
        k, layer, dim = keys
        row = {
            "k": int(k),
            "layer_index": int(layer),
            "tangent_dim": int(dim),
            "n_tests": int(len(g)),
            "own_energy_mean": float(g["own_energy"].mean()),
            "own_energy_median": float(g["own_energy"].median()),
            "own_error_mean": float(g["own_error"].mean()),
        }

        for prefix in ["same_entity_diff_task", "diff_entity_same_task", "random_other"]:
            row[f"{prefix}_energy_mean"] = float(g[f"{prefix}_energy_mean"].mean())
            row[f"{prefix}_error_mean"] = float(g[f"{prefix}_error_mean"].mean())
            row[f"own_better_{prefix}_mean_frac"] = float(g[f"own_better_than_{prefix}_mean"].mean())
            row[f"own_better_{prefix}_min_frac"] = float(g[f"own_better_than_{prefix}_min"].mean())
            row[f"energy_lift_vs_{prefix}_mean"] = float(g[f"energy_lift_vs_{prefix}_mean"].replace([np.inf, -np.inf], np.nan).mean())
            row[f"error_ratio_vs_{prefix}_mean"] = float(g[f"error_ratio_vs_{prefix}_mean"].replace([np.inf, -np.inf], np.nan).mean())

        rows.append(row)
    return pd.DataFrame(rows)

# ============================================================
# VERDICT
# ============================================================


def make_verdict(lowdim: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if summary.empty:
        return pd.DataFrame()

    low_agg = lowdim.groupby(["k", "layer_index"]).agg(
        mean_pc1_pc2=("pc1_pc2", "mean"),
        mean_pc1_pc2_pc3=("pc1_pc2_pc3", "mean"),
        mean_effective_dim_90=("effective_dim_90", "mean"),
        mean_effective_dim_95=("effective_dim_95", "mean"),
    ).reset_index()

    merged = summary.merge(low_agg, on=["k", "layer_index"], how="left")

    for _, r in merged.iterrows():
        same_task_win = r["own_better_diff_entity_same_task_mean_frac"]
        same_entity_win = r["own_better_same_entity_diff_task_mean_frac"]
        random_win = r["own_better_random_other_mean_frac"]

        energy_lift_task = r["energy_lift_vs_diff_entity_same_task_mean"]
        energy_lift_entity = r["energy_lift_vs_same_entity_diff_task_mean"]
        lowdim_ok = r["mean_pc1_pc2_pc3"] >= PASS_LOW_DIM_VAR90

        if (
            same_task_win >= PASS_STRONG_OWN_GT_MISMATCH
            and same_entity_win >= PASS_STRONG_OWN_GT_MISMATCH
            and random_win >= PASS_STRONG_OWN_GT_MISMATCH
            and energy_lift_task >= PASS_STRONG_ENERGY_LIFT
            and energy_lift_entity >= PASS_LITE_ENERGY_LIFT
            and lowdim_ok
        ):
            verdict = "PASS-Strong"
        elif (
            same_task_win >= PASS_LITE_OWN_GT_MISMATCH
            and same_entity_win >= PASS_LITE_OWN_GT_MISMATCH
            and random_win >= PASS_LITE_OWN_GT_MISMATCH
            and energy_lift_task >= PASS_LITE_ENERGY_LIFT
            and lowdim_ok
        ):
            verdict = "PASS-Lite"
        else:
            verdict = "MIXED/FAIL"

        rows.append({
            "k": int(r["k"]),
            "layer_index": int(r["layer_index"]),
            "tangent_dim": int(r["tangent_dim"]),
            "verdict": verdict,
            "own_energy_mean": float(r["own_energy_mean"]),
            "diff_entity_same_task_energy_mean": float(r["diff_entity_same_task_energy_mean"]),
            "same_entity_diff_task_energy_mean": float(r["same_entity_diff_task_energy_mean"]),
            "random_other_energy_mean": float(r["random_other_energy_mean"]),
            "own_better_diff_entity_same_task_frac": float(same_task_win),
            "own_better_same_entity_diff_task_frac": float(same_entity_win),
            "own_better_random_other_frac": float(random_win),
            "energy_lift_vs_diff_entity_same_task": float(energy_lift_task),
            "energy_lift_vs_same_entity_diff_task": float(energy_lift_entity),
            "energy_lift_vs_random_other": float(r["energy_lift_vs_random_other_mean"]),
            "mean_pc1_pc2": float(r["mean_pc1_pc2"]),
            "mean_pc1_pc2_pc3": float(r["mean_pc1_pc2_pc3"]),
            "mean_effective_dim_90": float(r["mean_effective_dim_90"]),
        })

    verdict_df = pd.DataFrame(rows)

    # Add aggregate shallow verdict by k and tangent dim across layers.
    agg_rows = []
    for (k, dim), g in verdict_df.groupby(["k", "tangent_dim"]):
        pass_strong = (g["verdict"] == "PASS-Strong").mean()
        pass_lite_or_better = g["verdict"].isin(["PASS-Strong", "PASS-Lite"]).mean()
        agg_rows.append({
            "k": int(k),
            "layer_index": "ALL_SHALLOW",
            "tangent_dim": int(dim),
            "verdict": (
                "PASS-Strong" if pass_strong >= 0.5 else
                "PASS-Lite" if pass_lite_or_better >= 0.5 else
                "MIXED/FAIL"
            ),
            "own_energy_mean": float(g["own_energy_mean"].mean()),
            "diff_entity_same_task_energy_mean": float(g["diff_entity_same_task_energy_mean"].mean()),
            "same_entity_diff_task_energy_mean": float(g["same_entity_diff_task_energy_mean"].mean()),
            "random_other_energy_mean": float(g["random_other_energy_mean"].mean()),
            "own_better_diff_entity_same_task_frac": float(g["own_better_diff_entity_same_task_frac"].mean()),
            "own_better_same_entity_diff_task_frac": float(g["own_better_same_entity_diff_task_frac"].mean()),
            "own_better_random_other_frac": float(g["own_better_random_other_frac"].mean()),
            "energy_lift_vs_diff_entity_same_task": float(g["energy_lift_vs_diff_entity_same_task"].mean()),
            "energy_lift_vs_same_entity_diff_task": float(g["energy_lift_vs_same_entity_diff_task"].mean()),
            "energy_lift_vs_random_other": float(g["energy_lift_vs_random_other"].mean()),
            "mean_pc1_pc2": float(g["mean_pc1_pc2"].mean()),
            "mean_pc1_pc2_pc3": float(g["mean_pc1_pc2_pc3"].mean()),
            "mean_effective_dim_90": float(g["mean_effective_dim_90"].mean()),
        })

    return pd.concat([verdict_df, pd.DataFrame(agg_rows)], ignore_index=True)

# ============================================================
# MAIN
# ============================================================


def main() -> None:
    set_seed(RANDOM_SEED)
    df = load_oa5a_records()

    print("\n[1/4] Running low-dimensionality audit...")
    lowdim = run_lowdim_audit(df)
    lowdim_file = OUTPUT_DIR / "oa5b_v2_tangent_lowdim_summary.csv"
    lowdim.to_csv(lowdim_file, index=False, encoding="utf-8-sig")
    print("Saved:", lowdim_file)

    print("\n[2/4] Running local tangent projection audit...")
    proj = run_projection_audit(df)
    proj_file = OUTPUT_DIR / "oa5b_v2_projection_records.csv"
    proj.to_csv(proj_file, index=False, encoding="utf-8-sig")
    print("Saved:", proj_file)

    print("\n[3/4] Summarizing projection results...")
    summary = summarize_projection(proj)
    summary_file = OUTPUT_DIR / "oa5b_v2_projection_summary.csv"
    summary.to_csv(summary_file, index=False, encoding="utf-8-sig")
    print("Saved:", summary_file)

    print("\n[4/4] Making verdict summary...")
    verdict = make_verdict(lowdim, summary)
    verdict_file = OUTPUT_DIR / "oa5b_v2_verdict_summary.csv"
    verdict.to_csv(verdict_file, index=False, encoding="utf-8-sig")
    print("Saved:", verdict_file)

    print("\n" + "=" * 72)
    print("OA-5B v2 complete.")
    print("Key files:")
    print("  ", lowdim_file)
    print("  ", summary_file)
    print("  ", verdict_file)
    print("=" * 72)

    if not verdict.empty:
        print("\nAggregate verdict rows:")
        print(verdict[verdict["layer_index"].astype(str) == "ALL_SHALLOW"].to_string(index=False))


if __name__ == "__main__":
    main()
