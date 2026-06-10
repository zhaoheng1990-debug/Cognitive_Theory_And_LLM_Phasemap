# -*- coding: utf-8 -*-
r"""
GPT_209_DSTA_2B_graph_paired_drift_attractor_audit.py

DSTA-2B: Graph-Paired Drift Attractor Audit

目标：
1. 直接读取 DSTA-2A.1 输出，不重新跑模型；
2. 使用 graph-paired clean-relative signed metrics，消除 graph/entity/surface 层面的偏置；
3. 构造 condition drift attractor geometry；
4. 验证：
   - exception 是否位于 clean 与 override/update 之间；
   - negation 是否更接近 update / replacement trajectory；
   - hallucination_like 是否朝 wrong-closure attractor 漂移；
   - weak_distractor 的高 TopK-center rotation 是否是 graph-paired 后仍存在；
5. 将 DSTA 从 signed rotation/reversal 推进到：
       Direction Spectrum Attractor Geometry

输入文件：
    默认读取：
    C:/Users/ZH/Desktop/AGI/python_script/dsta2a1_outputs/dsta2a1_graph_paired_signed_metrics.csv
    C:/Users/ZH/Desktop/AGI/python_script/dsta2a1_outputs/dsta2a1_signed_layer_metrics.csv
    C:/Users/ZH/Desktop/AGI/python_script/dsta2a1_outputs/dsta2a1_signed_window_summary.csv

如果你把本脚本放在同一目录运行，也会自动尝试：
    ./dsta2a1_outputs/...

输出目录：
    dsta2b_outputs/

关键输出：
    dsta2b_graph_paired_window_summary.csv
    dsta2b_condition_attractor_vectors.csv
    dsta2b_condition_attractor_distance_matrix.csv
    dsta2b_condition_attractor_similarity_matrix.csv
    dsta2b_condition_attractor_geometry_summary.csv
    dsta2b_attractor_verdict.csv
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


CONFIG = {
    # 优先使用你本地脚本目录下的 2A.1 输出
    "DSTA2A1_OUTPUT_DIR": r"C:\Users\ZH\Desktop\AGI\python_script\dsta2a1_outputs",

    # fallback: 当前目录
    "DSTA2A1_OUTPUT_DIR_FALLBACK": r"dsta2a1_outputs",

    "OUTPUT_DIR": r"dsta2b_outputs",

    "TOPK_MAIN": 500,

    "WINDOWS": ["init_0_6", "mid_7_19", "critical_20_22", "commit_23_26"],

    # 用于构造 attractor vector 的主要窗口
    "ATTRACTOR_WINDOWS": ["critical_20_22", "commit_23_26"],

    # 用于 pairwise geometry 的特征
    "ATTRACTOR_FEATURES": [
        "center_rotation_abs_deg_mean",
        "center_reversal_score_mean",
        "delta_center_norm_mean",
        "transport_rotation_abs_deg_mean",
        "transport_reversal_score_mean",
        "transport_delta_norm_mean",
        "center_signed_align_mean",
        "transport_signed_align_mean",
    ],

    # reference condition
    "REFERENCE_CONDITION": "stable_clean",

    # semantic groups for verdict
    "UPDATE_CONDITIONS": ["closure_update", "closure_temporal", "closure_authority"],
    "OVERRIDE_CONDITIONS": ["closure_override"],
    "EXCEPTION_CONDITIONS": ["closure_exception"],
    "NEGATION_CONDITIONS": ["closure_negation"],
    "HALLUCINATION_CONDITIONS": ["hallucination_like"],
    "COMPETITION_CONDITIONS": ["competition_direct", "competition_equal_evidence", "competition_source_claim"],
    "CONTROL_CONDITIONS": ["stable_clean", "stable_redundant", "weak_distractor"],
}


# =========================
# IO
# =========================

def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_input_dir() -> Path:
    p1 = Path(CONFIG["DSTA2A1_OUTPUT_DIR"])
    if p1.exists():
        return p1
    p2 = Path(CONFIG["DSTA2A1_OUTPUT_DIR_FALLBACK"])
    if p2.exists():
        return p2
    # One more fallback: current working directory if files are there
    p3 = Path(".")
    if (p3 / "dsta2a1_graph_paired_signed_metrics.csv").exists():
        return p3
    raise FileNotFoundError(
        "Cannot find DSTA-2A.1 output directory. Tried:\n"
        f"  {p1}\n  {p2}\n  {p3.resolve()}\n"
        "请把本脚本放到 python_script 目录，或修改 CONFIG['DSTA2A1_OUTPUT_DIR']。"
    )


def load_inputs(input_dir: Path):
    graph_path = input_dir / "dsta2a1_graph_paired_signed_metrics.csv"
    layer_path = input_dir / "dsta2a1_signed_layer_metrics.csv"
    window_path = input_dir / "dsta2a1_signed_window_summary.csv"

    if not graph_path.exists():
        raise FileNotFoundError(f"Missing required file: {graph_path}")

    graph_df = pd.read_csv(graph_path)
    layer_df = pd.read_csv(layer_path) if layer_path.exists() else None
    window_df = pd.read_csv(window_path) if window_path.exists() else None
    return graph_df, layer_df, window_df


# =========================
# Feature construction
# =========================

def assign_window(layer: int) -> Optional[str]:
    if 0 <= layer <= 6:
        return "init_0_6"
    if 7 <= layer <= 19:
        return "mid_7_19"
    if 20 <= layer <= 22:
        return "critical_20_22"
    if 23 <= layer <= 26:
        return "commit_23_26"
    return None


def summarize_graph_paired(graph_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    对 graph-paired signed metrics 按 condition / window 聚合。
    """
    df = graph_df.copy()
    if "window" not in df.columns:
        df["window"] = df["layer"].apply(assign_window)
    df = df[df["window"].notna()].copy()

    # ensure required cols
    required = [
        "center_signed_align_vs_clean",
        "center_reversal_score",
        "center_rotation_abs_deg",
        "delta_center_norm",
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column in graph paired metrics: {c}")

    rows = []
    group_cols = ["topk", "condition", "window"]
    for (topk, cond, win), sub in df.groupby(group_cols):
        rows.append({
            "topk": int(topk),
            "condition": cond,
            "window": win,
            "n_graph_layer_points": int(len(sub)),
            "n_graphs": int(sub["graph_id"].nunique()) if "graph_id" in sub.columns else np.nan,
            "center_signed_align_mean": float(sub["center_signed_align_vs_clean"].mean()),
            "center_signed_align_std": float(sub["center_signed_align_vs_clean"].std(ddof=0)),
            "center_reversal_score_mean": float(sub["center_reversal_score"].mean()),
            "center_rotation_abs_deg_mean": float(sub["center_rotation_abs_deg"].mean()),
            "center_signed_angle_deg_mean": float(sub["center_signed_angle_deg"].mean()) if "center_signed_angle_deg" in sub.columns else np.nan,
            "delta_center_norm_mean": float(sub["delta_center_norm"].mean()),
        })

    out = pd.DataFrame(rows)

    # If layer-level transport metrics are not available in graph paired file, merge later from window summary if possible.
    out.to_csv(out_dir / "dsta2b_graph_paired_window_summary.csv", index=False, encoding="utf-8-sig")
    return out


def merge_transport_features(gp_summary: pd.DataFrame, layer_window_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    graph paired file has center-based metrics only.
    If DSTA-2A.1 signed_window_summary is available, merge transport metrics by topk/condition/window.
    """
    out = gp_summary.copy()
    if layer_window_df is None or layer_window_df.empty:
        for c in ["transport_rotation_abs_deg_mean", "transport_reversal_score_mean", "transport_delta_norm_mean",
                  "transport_signed_align_mean"]:
            out[c] = np.nan
        return out

    cols = [
        "topk", "condition", "window",
        "transport_signed_align_mean",
        "transport_reversal_score_mean",
        "transport_rotation_abs_deg_mean",
        "transport_delta_norm_mean",
        "delta_align_to_update_mean",
        "delta_align_to_override_mean",
    ]
    existing = [c for c in cols if c in layer_window_df.columns]
    merge_df = layer_window_df[existing].copy()
    out = out.merge(merge_df, on=["topk", "condition", "window"], how="left")
    return out


def build_attractor_vectors(summary_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    每个 condition 一个 attractor vector：
    concat critical + commit 的 signed center/transport features.
    """
    df = summary_df[summary_df["topk"] == CONFIG["TOPK_MAIN"]].copy()
    if df.empty:
        # fallback to all topk or first topk
        df = summary_df.copy()

    rows = []
    conditions = sorted(df["condition"].unique())
    for cond in conditions:
        subc = df[df["condition"] == cond]
        row = {"condition": cond, "topk": int(subc["topk"].iloc[0]) if len(subc) else CONFIG["TOPK_MAIN"]}

        for win in CONFIG["ATTRACTOR_WINDOWS"]:
            sw = subc[subc["window"] == win]
            if len(sw) == 0:
                continue
            sw = sw.iloc[0]
            for feat in CONFIG["ATTRACTOR_FEATURES"]:
                if feat in sw.index:
                    row[f"{win}__{feat}"] = float(sw[feat])

        rows.append(row)

    vec_df = pd.DataFrame(rows)

    # Fill NaN with column mean; if all NaN then 0.
    feature_cols = [c for c in vec_df.columns if c not in ["condition", "topk"]]
    for c in feature_cols:
        if vec_df[c].isna().all():
            vec_df[c] = 0.0
        else:
            vec_df[c] = vec_df[c].fillna(vec_df[c].mean())

    vec_df.to_csv(out_dir / "dsta2b_condition_attractor_vectors.csv", index=False, encoding="utf-8-sig")
    return vec_df


def standardize_matrix(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd < 1e-9] = 1.0
    return (X - mu) / sd


def compute_pairwise_geometry(vec_df: pd.DataFrame, out_dir: Path):
    feature_cols = [c for c in vec_df.columns if c not in ["condition", "topk"]]
    conds = vec_df["condition"].tolist()
    X = vec_df[feature_cols].values.astype(float)
    Xz = standardize_matrix(X)

    n = len(conds)
    dist = np.zeros((n, n), dtype=float)
    sim = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(Xz[i] - Xz[j])
            denom = np.linalg.norm(Xz[i]) * np.linalg.norm(Xz[j]) + 1e-12
            sim[i, j] = float(np.dot(Xz[i], Xz[j]) / denom)

    dist_df = pd.DataFrame(dist, index=conds, columns=conds)
    sim_df = pd.DataFrame(sim, index=conds, columns=conds)

    dist_df.to_csv(out_dir / "dsta2b_condition_attractor_distance_matrix.csv", encoding="utf-8-sig")
    sim_df.to_csv(out_dir / "dsta2b_condition_attractor_similarity_matrix.csv", encoding="utf-8-sig")

    # long form
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            rows.append({
                "condition_i": conds[i],
                "condition_j": conds[j],
                "distance": dist[i, j],
                "cosine_similarity": sim[i, j],
            })
    long_df = pd.DataFrame(rows).sort_values("distance")
    long_df.to_csv(out_dir / "dsta2b_condition_attractor_pairwise_long.csv", index=False, encoding="utf-8-sig")
    return dist_df, sim_df, long_df


# =========================
# Geometry tests / verdicts
# =========================

def nearest_condition(dist_df: pd.DataFrame, cond: str, exclude_self=True) -> Tuple[str, float]:
    if cond not in dist_df.index:
        return "", np.nan
    s = dist_df.loc[cond].copy()
    if exclude_self and cond in s.index:
        s.loc[cond] = np.inf
    j = s.idxmin()
    return j, float(s.loc[j])


def between_score(dist_df: pd.DataFrame, x: str, a: str, b: str) -> Dict:
    """
    判断 x 是否位于 a 与 b 的中间附近：
    - dist(x,a), dist(x,b), dist(a,b)
    - midpoint residual in standardized vector space will be approximated elsewhere if vectors available
    """
    if any(c not in dist_df.index for c in [x, a, b]):
        return {"x": x, "a": a, "b": b, "available": False}
    d_xa = float(dist_df.loc[x, a])
    d_xb = float(dist_df.loc[x, b])
    d_ab = float(dist_df.loc[a, b])
    # If x is between, d_xa + d_xb ≈ d_ab; define excess
    excess = d_xa + d_xb - d_ab
    return {
        "x": x, "a": a, "b": b, "available": True,
        "d_x_a": d_xa,
        "d_x_b": d_xb,
        "d_a_b": d_ab,
        "between_excess": excess,
        "relative_position_from_a": d_xa / (d_ab + 1e-12),
    }


def project_position(vec_df: pd.DataFrame, x: str, a: str, b: str) -> Dict:
    """
    Project x onto line a->b in standardized attractor vector space.
    t=0 at a, t=1 at b.
    perpendicular residual reports off-axis component.
    """
    feature_cols = [c for c in vec_df.columns if c not in ["condition", "topk"]]
    df = vec_df.set_index("condition")
    if any(c not in df.index for c in [x, a, b]):
        return {"x": x, "a": a, "b": b, "available": False}

    X = standardize_matrix(vec_df[feature_cols].values.astype(float))
    zdf = pd.DataFrame(X, index=vec_df["condition"].values, columns=feature_cols)

    vx = zdf.loc[x].values
    va = zdf.loc[a].values
    vb = zdf.loc[b].values
    axis = vb - va
    denom = np.dot(axis, axis) + 1e-12
    t = float(np.dot(vx - va, axis) / denom)
    proj = va + t * axis
    perp = float(np.linalg.norm(vx - proj))
    axis_len = float(np.linalg.norm(axis))
    return {
        "x": x, "a": a, "b": b, "available": True,
        "projection_t": t,
        "axis_length": axis_len,
        "perpendicular_residual": perp,
        "normalized_perp": perp / (axis_len + 1e-12),
    }


def make_geometry_summary(summary_df: pd.DataFrame, vec_df: pd.DataFrame, dist_df: pd.DataFrame, sim_df: pd.DataFrame, long_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []

    # nearest neighbor for every condition
    for cond in dist_df.index:
        nn, d = nearest_condition(dist_df, cond)
        rows.append({
            "test": "nearest_neighbor",
            "condition": cond,
            "nearest": nn,
            "distance": d,
            "result": "",
            "note": "Nearest attractor in standardized critical+commit signed geometry."
        })

    # Specific hypotheses
    hypotheses = [
        ("closure_exception", "stable_clean", "closure_override", "exception_between_clean_override"),
        ("closure_exception", "closure_update", "closure_override", "exception_between_update_override"),
        ("closure_negation", "closure_update", "closure_override", "negation_between_update_override"),
        ("hallucination_like", "closure_update", "closure_override", "hallucination_between_update_override"),
        ("weak_distractor", "stable_clean", "competition_direct", "weak_between_clean_competition"),
    ]

    for x, a, b, name in hypotheses:
        bs = between_score(dist_df, x, a, b)
        ps = project_position(vec_df, x, a, b)
        row = {"test": name, "condition": x, "anchor_a": a, "anchor_b": b}
        row.update(bs)
        row.update({f"proj_{k}": v for k, v in ps.items() if k not in ["x", "a", "b", "available"]})
        if bs.get("available") and ps.get("available"):
            t = ps["projection_t"]
            perp = ps["normalized_perp"]
            if -0.15 <= t <= 1.15 and perp < 0.80:
                result = "ON_AXIS_OR_NEAR_AXIS"
            elif t > 1.15:
                result = "BEYOND_B"
            elif t < -0.15:
                result = "BEFORE_A"
            else:
                result = "OFF_AXIS"
            row["result"] = result
            row["note"] = f"projection_t={t:.3f}; normalized_perp={perp:.3f}"
        else:
            row["result"] = "UNAVAILABLE"
            row["note"] = ""
        rows.append(row)

    # Rank nearest to override/update anchors
    for cond in dist_df.index:
        row = {
            "test": "anchor_distance",
            "condition": cond,
            "d_to_stable": float(dist_df.loc[cond, "stable_clean"]) if "stable_clean" in dist_df.columns else np.nan,
            "d_to_update": float(dist_df.loc[cond, "closure_update"]) if "closure_update" in dist_df.columns else np.nan,
            "d_to_override": float(dist_df.loc[cond, "closure_override"]) if "closure_override" in dist_df.columns else np.nan,
            "d_to_exception": float(dist_df.loc[cond, "closure_exception"]) if "closure_exception" in dist_df.columns else np.nan,
            "result": "",
            "note": "Anchor distances."
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "dsta2b_condition_attractor_geometry_summary.csv", index=False, encoding="utf-8-sig")
    return out


def make_verdict(summary_df: pd.DataFrame, vec_df: pd.DataFrame, dist_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []

    def get_proj(x, a, b):
        return project_position(vec_df, x, a, b)

    def get_dist(a, b):
        if a in dist_df.index and b in dist_df.columns:
            return float(dist_df.loc[a, b])
        return np.nan

    # Hypothesis 1: exception is between update and override or clean and override
    for axis_a, axis_b, label in [
        ("stable_clean", "closure_override", "exception_clean_override_axis"),
        ("closure_update", "closure_override", "exception_update_override_axis"),
    ]:
        ps = get_proj("closure_exception", axis_a, axis_b)
        if ps.get("available"):
            t, perp = ps["projection_t"], ps["normalized_perp"]
            passed = (-0.10 <= t <= 1.15 and perp < 0.85)
            rows.append({
                "claim": label,
                "target": "closure_exception",
                "axis": f"{axis_a}->{axis_b}",
                "projection_t": t,
                "normalized_perp": perp,
                "verdict": "PASS_EXCEPTION_ON_AXIS" if passed else "FAIL_EXCEPTION_OFF_AXIS",
                "interpretation": "exception behaves like local conditioned rotation along override axis" if passed else "exception not explained by this axis",
            })

    # Hypothesis 2: negation closer to update/replacement than override
    for anchor in ["closure_update", "closure_override", "closure_exception"]:
        rows.append({
            "claim": "negation_anchor_distance",
            "target": "closure_negation",
            "anchor": anchor,
            "distance": get_dist("closure_negation", anchor),
            "verdict": "",
            "interpretation": "",
        })
    d_neg_update = get_dist("closure_negation", "closure_update")
    d_neg_override = get_dist("closure_negation", "closure_override")
    rows.append({
        "claim": "negation_update_vs_override",
        "target": "closure_negation",
        "d_to_update": d_neg_update,
        "d_to_override": d_neg_override,
        "verdict": "NEGATION_UPDATE_LIKE" if d_neg_update < d_neg_override else "NEGATION_OVERRIDE_LIKE",
        "interpretation": "negation behaves more like replacement/update trajectory" if d_neg_update < d_neg_override else "negation behaves more like override trajectory",
    })

    # Hypothesis 3: hallucination_like wrong-closure drift aligns with override/update side, not stable
    d_h_stable = get_dist("hallucination_like", "stable_clean")
    d_h_update = get_dist("hallucination_like", "closure_update")
    d_h_override = get_dist("hallucination_like", "closure_override")
    closest = min(
        [("stable_clean", d_h_stable), ("closure_update", d_h_update), ("closure_override", d_h_override)],
        key=lambda x: x[1] if np.isfinite(x[1]) else np.inf
    )
    rows.append({
        "claim": "hallucination_anchor",
        "target": "hallucination_like",
        "d_to_stable": d_h_stable,
        "d_to_update": d_h_update,
        "d_to_override": d_h_override,
        "closest_anchor": closest[0],
        "verdict": "WRONG_CLOSURE_SIDE" if closest[0] in ["closure_update", "closure_override"] else "STABLE_SIDE",
        "interpretation": "hallucination_like drifts toward wrong-closure attractor side" if closest[0] in ["closure_update", "closure_override"] else "hallucination_like not separated from stable in this geometry",
    })

    # Hypothesis 4: weak_distractor high rotation risk check
    d_w_stable = get_dist("weak_distractor", "stable_clean")
    d_w_comp = get_dist("weak_distractor", "competition_direct")
    d_w_update = get_dist("weak_distractor", "closure_update")
    rows.append({
        "claim": "weak_distractor_geometry",
        "target": "weak_distractor",
        "d_to_stable": d_w_stable,
        "d_to_competition": d_w_comp,
        "d_to_update": d_w_update,
        "verdict": "WEAK_STABLE_LIKE" if d_w_stable < min(d_w_comp, d_w_update) else "WEAK_NOT_STABLE_LIKE",
        "interpretation": "If weak remains near stable, prior high TopK rotation was likely unrelated-token geometry not risk." if d_w_stable < min(d_w_comp, d_w_update) else "weak_distractor still forms non-stable drift attractor",
    })

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "dsta2b_attractor_verdict.csv", index=False, encoding="utf-8-sig")
    return out


def main():
    out_dir = ensure_dir(CONFIG["OUTPUT_DIR"])
    with open(out_dir / "dsta2b_config.json", "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, ensure_ascii=False, indent=2)

    input_dir = resolve_input_dir()
    print("[LOAD] input_dir:", input_dir.resolve())

    graph_df, layer_df, layer_window_df = load_inputs(input_dir)

    gp_summary = summarize_graph_paired(graph_df, out_dir)
    merged_summary = merge_transport_features(gp_summary, layer_window_df)
    merged_summary.to_csv(out_dir / "dsta2b_graph_paired_window_summary_merged.csv", index=False, encoding="utf-8-sig")

    vec_df = build_attractor_vectors(merged_summary, out_dir)
    dist_df, sim_df, long_df = compute_pairwise_geometry(vec_df, out_dir)
    geom_summary = make_geometry_summary(merged_summary, vec_df, dist_df, sim_df, long_df, out_dir)
    verdict = make_verdict(merged_summary, vec_df, dist_df, out_dir)

    print("\n[DONE] Outputs saved to:", out_dir.resolve())
    print("\n[ATTRACTOR VECTORS]")
    print(vec_df.to_string(index=False))

    print("\n[NEAREST PAIRS]")
    print(long_df.head(20).to_string(index=False))

    print("\n[VERDICT]")
    print(verdict.to_string(index=False))

    print("\n[KEY FILES]")
    for name in [
        "dsta2b_graph_paired_window_summary.csv",
        "dsta2b_graph_paired_window_summary_merged.csv",
        "dsta2b_condition_attractor_vectors.csv",
        "dsta2b_condition_attractor_distance_matrix.csv",
        "dsta2b_condition_attractor_similarity_matrix.csv",
        "dsta2b_condition_attractor_pairwise_long.csv",
        "dsta2b_condition_attractor_geometry_summary.csv",
        "dsta2b_attractor_verdict.csv",
    ]:
        print(" -", out_dir / name)


if __name__ == "__main__":
    main()
