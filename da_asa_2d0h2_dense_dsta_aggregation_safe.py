#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DA-ASA-2D.0h2 Dense DSTA Aggregation (duplicate-column safe)

Purpose:
  Build a sample-level policy feature table from DA-ASA baseline/policy outputs
  plus DSTA condition-level / graph-condition-level features.

Key fix vs 2D.0g:
  DSTA tables are NOT merged by sample_id. They are aggregated by condition
  and, when possible, by (graph_id, condition), then broadcast to all samples.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

POLICY_NAME_TO_ID = {
    "NO_INTERVENTION": 0,
    "CORE_CLOSURE_UPDATE": 1,
    "OVERRIDE_THREE_STAGE": 2,
    "EQUAL_EVIDENCE_ORDER": 3,
    "RESERVED_HALLUCINATION_CONTROLLER": 4,
}

CONDITION_TO_POLICY = {
    "stable": "NO_INTERVENTION",
    "stable_clean": "NO_INTERVENTION",
    "stable_clean_basic": "NO_INTERVENTION",
    "stable_redundant": "NO_INTERVENTION",
    "stable_weak_note": "NO_INTERVENTION",
    "weak_distractor": "NO_INTERVENTION",
    "competition_direct": "NO_INTERVENTION",
    "competition_source_claim": "NO_INTERVENTION",
    "source_claim": "NO_INTERVENTION",
    "hallucination_like": "NO_INTERVENTION",
    "non_target": "NO_INTERVENTION",
    "nontarget": "NO_INTERVENTION",
    "control": "NO_INTERVENTION",
    # policy targets from DA-ASA-2C rule
    "closure_exception": "CORE_CLOSURE_UPDATE",
    "closure_negation": "CORE_CLOSURE_UPDATE",
    "closure_override": "OVERRIDE_THREE_STAGE",
    "competition_equal_evidence": "EQUAL_EVIDENCE_ORDER",
    "equal_evidence": "EQUAL_EVIDENCE_ORDER",
    # intentionally protected in 2D.0h unless explicitly changed later
    "closure_update": "NO_INTERVENTION",
    "closure_temporal": "NO_INTERVENTION",
    "closure_authority": "NO_INTERVENTION",
}

EXPECTED_GROUPS = {
    "baseline_trajectory": [
        "R_final", "R_20_22", "R_23_25", "baseline_margin",
        "clean_conflict_margin", "boundary_distance_proxy", "order_precursor_score",
        "deltaU", "dR20_25_l2", "dR20_25_pc1",
    ],
    "policy_outcome": [
        "policy_clean_rate", "policy_conflict_rate", "policy_delta_clean_rate",
        "policy_mean_shift", "policy_abs_shift", "target_hit_rate",
        "nontarget_hit_rate", "specificity", "safety_adjusted_gain",
    ],
    "dsta_signed_attractor": [
        "center_signed_align", "center_rotation_abs_deg", "delta_center_norm",
        "transport_signed_align", "transport_rotation_abs_deg",
        "delta_align_to_update", "delta_align_to_override",
        "nearest_attractor_distance", "attractor_distance_update",
        "attractor_distance_override", "attractor_distance_stable", "attractor_distance_wrong",
    ],
    "direction_spectrum": [
        "basis_rotation", "basis_similarity", "weight_shift", "weight_entropy",
        "dominance_gap", "critical_weight_transport", "commit_weight_transport",
    ],
    "topk_vim": [
        "topk_center_trajectory", "topk_spread", "topk_entropy",
        "topk_delta_norm", "topk_delta_structure_score", "vim_center_drift",
        "vim_spread_delta", "vim_entropy_delta", "topk_jaccard_to_clean",
        "topk_center_cos_to_clean",
    ],
}

ID_COL_CANDIDATES = ["sample_id", "id", "_id", "prompt_id", "row_index"]


def safe_name(x: object, max_len: int = 80) -> str:
    s = str(x)
    s = re.sub(r"[^0-9A-Za-z_\-]+", "_", s).strip("_")
    if not s:
        s = "na"
    if re.match(r"^[0-9]", s):
        s = "v_" + s
    return s[:max_len]


def normalize_condition(s: object) -> str:
    if pd.isna(s):
        return ""
    return str(s).strip()


def make_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate column names, keeping the first occurrence.
    Pandas returns a DataFrame for df["topk"] when duplicate topk columns exist,
    which breaks groupby/pivot with: Grouper for 'topk' not 1-dimensional.
    """
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def read_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="utf-8-sig")
    return make_unique_columns(df)


def ensure_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "condition" not in df.columns:
        for c in df.columns:
            if c.lower().endswith("condition") or c.lower() in ["_condition", "cond"]:
                df = df.rename(columns={c: "condition"})
                break
    if "condition" not in df.columns:
        raise RuntimeError("Base file must contain a condition column.")
    df["condition"] = df["condition"].map(normalize_condition)
    if "sample_id" not in df.columns:
        for c in ID_COL_CANDIDATES:
            if c in df.columns:
                df["sample_id"] = df[c].astype(str)
                break
    if "sample_id" not in df.columns:
        df["sample_id"] = np.arange(len(df)).astype(str)
    if "graph_id" in df.columns:
        df["graph_id"] = df["graph_id"].astype(str)
    else:
        df["graph_id"] = ""
    return df


def assign_policy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["policy_class"] = df["condition"].map(CONDITION_TO_POLICY).fillna("NO_INTERVENTION")
    df["policy_id"] = df["policy_class"].map(POLICY_NAME_TO_ID).astype(int)
    df["is_core_closure_target"] = df["condition"].isin(["closure_exception", "closure_negation"]).astype(int)
    df["is_override_target"] = (df["condition"] == "closure_override").astype(int)
    df["is_equal_evidence_target"] = df["condition"].isin(["competition_equal_evidence", "equal_evidence"]).astype(int)
    df["is_hallucination_like"] = (df["condition"] == "hallucination_like").astype(int)
    df["is_source_claim"] = df["condition"].isin(["competition_source_claim", "source_claim"]).astype(int)
    df["is_protected_no_intervention"] = (df["policy_class"] == "NO_INTERVENTION").astype(int)
    return df


def derive_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "baseline_R_final" in df.columns:
        df["R_final"] = pd.to_numeric(df["baseline_R_final"], errors="coerce")
    elif "R_final" in df.columns:
        df["R_final"] = pd.to_numeric(df["R_final"], errors="coerce")
    elif {"baseline_C_logit", "baseline_E_logit"}.issubset(df.columns):
        df["R_final"] = pd.to_numeric(df["baseline_C_logit"], errors="coerce") - pd.to_numeric(df["baseline_E_logit"], errors="coerce")
    elif {"C_logit", "E_logit"}.issubset(df.columns):
        df["R_final"] = pd.to_numeric(df["C_logit"], errors="coerce") - pd.to_numeric(df["E_logit"], errors="coerce")
    if "R_final" in df.columns:
        df["baseline_margin"] = df["R_final"]
        df["boundary_distance_proxy"] = df["R_final"].abs()
    return df


def infer_group(path: Path) -> str:
    p = str(path).lower()
    name = path.name.lower()
    if "dsta" in p and ("signed" in name or "attractor" in name or "distance" in name or "geometry" in name or "pairwise" in name or "similarity" in name or "vector" in name):
        return "dsta_signed_attractor"
    if "dsta" in p and ("topk" in name or "scalar" in name or "vim" in name):
        return "topk_vim"
    if "direction" in name or "spectrum" in name or "transport" in name:
        return "direction_spectrum"
    if "policy" in name or "condition_summary" in name or "verdict" in name:
        return "policy_outcome"
    return "baseline_trajectory"


def numeric_cols(df: pd.DataFrame, exclude: Iterable[str]) -> List[str]:
    excl = set(exclude)
    cols = []
    for c in df.columns:
        if c in excl:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
        else:
            # accept numeric-like object columns, but avoid long text/id columns
            if c.lower() in ["top_ids_head", "text", "source_file", "applied_actions"]:
                continue
            converted = pd.to_numeric(df[c], errors="coerce")
            if converted.notna().mean() > 0.8:
                df[c] = converted
                cols.append(c)
    return cols


def prefix_cols(df: pd.DataFrame, prefix: str, keys: List[str]) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        if c not in keys and not str(c).startswith(prefix + "_"):
            ren[c] = f"{prefix}_{c}"
    return df.rename(columns=ren)


def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    return make_unique_columns(df)


def safe_merge(base: pd.DataFrame, other: pd.DataFrame, keys: List[str], prefix: str, report: List[dict], path: Optional[Path] = None) -> pd.DataFrame:
    if not keys:
        return base
    other = other.copy()
    for k in keys:
        if k in base.columns and k in other.columns:
            base[k] = base[k].astype(str)
            other[k] = other[k].astype(str)
    other = prefix_cols(other, prefix, keys)
    # Drop duplicate non-key columns already in base.
    keep = []
    for c in other.columns:
        if c in keys or c not in base.columns:
            keep.append(c)
    other = other[keep]
    before = set(base.columns)
    merged = base.merge(other, on=keys, how="left")
    merged = dedupe_columns(merged)
    added = [c for c in merged.columns if c not in before]
    report.append({
        "group": prefix,
        "keys": keys,
        "rows": int(len(other)),
        "added_cols": int(len(added)),
        "added_preview": added[:20],
        "path": str(path) if path else None,
    })
    return merged


def aggregate_condition_level(df: pd.DataFrame, group: str) -> Optional[pd.DataFrame]:
    df = make_unique_columns(df.copy())
    if "condition" not in df.columns:
        for c in df.columns:
            if c.lower() in ["_condition", "cond"] or c.lower().endswith("condition"):
                df = df.rename(columns={c: "condition"})
                break
    if "condition" not in df.columns:
        return None
    df["condition"] = df["condition"].map(normalize_condition)
    exclude = ["condition", "graph_id", "sample_id", "id", "_id", "source_file", "text", "top_ids_head", "policy", "applied_actions", "mechanism", "risk_regime", "window"]
    nums = numeric_cols(df, exclude)
    if not nums:
        return None

    # Basic condition means.
    out = df.groupby("condition", dropna=False)[nums].mean(numeric_only=True).reset_index()

    # Add window-specific means when a window column exists and cardinality is small.
    if "window" in df.columns and not isinstance(df["window"], pd.DataFrame) and df["window"].nunique(dropna=True) <= 20:
        pivs = []
        for col in nums:
            piv = df.pivot_table(index="condition", columns="window", values=col, aggfunc="mean")
            piv.columns = [f"{col}_window_{safe_name(w)}" for w in piv.columns]
            pivs.append(piv.reset_index())
        if pivs:
            wide = pivs[0]
            for piv in pivs[1:]:
                wide = wide.merge(piv, on="condition", how="outer")
            out = out.merge(wide, on="condition", how="outer")

    # Add topk-specific means when topk cardinality is small.
    if "topk" in df.columns and not isinstance(df["topk"], pd.DataFrame) and df["topk"].nunique(dropna=True) <= 8:
        pivs = []
        for col in nums:
            piv = df.pivot_table(index="condition", columns="topk", values=col, aggfunc="mean")
            piv.columns = [f"{col}_topk_{safe_name(k)}" for k in piv.columns]
            pivs.append(piv.reset_index())
        if pivs:
            wide = pivs[0]
            for piv in pivs[1:]:
                wide = wide.merge(piv, on="condition", how="outer")
            out = out.merge(wide, on="condition", how="outer")

    return out


def aggregate_graph_condition_level(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    df = make_unique_columns(df.copy())
    if "condition" not in df.columns or "graph_id" not in df.columns:
        return None
    df["condition"] = df["condition"].map(normalize_condition)
    df["graph_id"] = df["graph_id"].astype(str)
    exclude = ["condition", "graph_id", "sample_id", "id", "_id", "source_file", "text", "top_ids_head", "policy", "applied_actions", "mechanism", "risk_regime", "window"]
    nums = numeric_cols(df, exclude)
    if not nums:
        return None
    return df.groupby(["graph_id", "condition"], dropna=False)[nums].mean(numeric_only=True).reset_index()


def prepare_sample_level_policy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "condition" not in df.columns:
        return df
    df["condition"] = df["condition"].map(normalize_condition)
    if "sample_id" not in df.columns:
        for c in ID_COL_CANDIDATES:
            if c in df.columns:
                df["sample_id"] = df[c].astype(str)
                break
    if "sample_id" not in df.columns:
        return df
    if "R_final" in df.columns:
        df["baseline_margin"] = pd.to_numeric(df["R_final"], errors="coerce")
        df["boundary_distance_proxy"] = df["baseline_margin"].abs()
    return df


def present_by_group(columns: List[str]) -> Dict[str, List[str]]:
    low_cols = {c.lower(): c for c in columns}
    out: Dict[str, List[str]] = {}
    for group, feats in EXPECTED_GROUPS.items():
        found = []
        for f in feats:
            fl = f.lower()
            for lc, orig in low_cols.items():
                if fl in lc:
                    found.append(orig)
                    break
        out[group] = sorted(set(found))
    return out


def missing_expected(present: Dict[str, List[str]]) -> Dict[str, List[str]]:
    miss = {}
    for group, feats in EXPECTED_GROUPS.items():
        found_low = "\n".join([x.lower() for x in present.get(group, [])])
        miss[group] = [f for f in feats if f.lower() not in found_low]
    return miss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-file", required=True)
    ap.add_argument("--include-file", action="append", default=[])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    base_path = Path(args.base_file)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = base_path.parent.parent / "da_asa_2d0h_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = ensure_base_columns(read_csv(base_path))
    base = derive_base_features(base)
    base = assign_policy(base)

    loaded = []
    merge_report = []
    merged = base.copy()

    for f in args.include_file:
        path = Path(f)
        if not path.exists() or path.resolve() == base_path.resolve():
            continue
        try:
            df = read_csv(path)
        except Exception as e:
            loaded.append({"path": str(path), "error": repr(e)})
            continue
        group = infer_group(path)
        loaded.append({"path": str(path), "group": group, "rows": int(len(df)), "columns": list(map(str, df.columns))})

        # DSTA files are dense condition-level or graph-condition-level signals.
        if group in ["dsta_signed_attractor", "direction_spectrum", "topk_vim"]:
            gc = aggregate_graph_condition_level(df)
            if gc is not None and "graph_id" in merged.columns:
                merged = safe_merge(merged, gc, ["graph_id", "condition"], group, merge_report, path)
            cond = aggregate_condition_level(df, group)
            if cond is not None:
                merged = safe_merge(merged, cond, ["condition"], group, merge_report, path)
            continue

        # DA-ASA sample-level and condition-level outputs.
        df2 = prepare_sample_level_policy(df)
        if "sample_id" in df2.columns and "condition" in df2.columns:
            merged = safe_merge(merged, df2, ["sample_id", "condition"], group, merge_report, path)
        elif "condition" in df2.columns:
            cond = aggregate_condition_level(df2, group)
            if cond is not None:
                merged = safe_merge(merged, cond, ["condition"], group, merge_report, path)

    # Remove direct leakage columns from numeric training table, but keep in full table.
    feature_table = dedupe_columns(merged)
    numeric_cols_all = [c for c in feature_table.columns if pd.api.types.is_numeric_dtype(feature_table[c])]

    full_path = out_dir / "da_asa2d_policy_feature_table.csv"
    num_path = out_dir / "da_asa2d_policy_feature_numeric.csv"
    schema_path = out_dir / "da_asa2d_feature_schema.json"
    miss_path = out_dir / "da_asa2d_missing_report.json"
    summary_path = out_dir / "da_asa2d_summary.json"

    feature_table.to_csv(full_path, index=False, encoding="utf-8-sig")
    keep_keys = [c for c in ["sample_id", "graph_id", "condition", "policy_class", "policy_id"] if c in feature_table.columns]
    numeric_table = feature_table[keep_keys + [c for c in numeric_cols_all if c not in keep_keys]].copy()
    numeric_table.to_csv(num_path, index=False, encoding="utf-8-sig")

    present = present_by_group(list(feature_table.columns))
    missing = missing_expected(present)
    dsta_cols = [c for c in numeric_cols_all if c.startswith("dsta_signed_attractor_") or c.startswith("direction_spectrum_") or c.startswith("topk_vim_")]
    dsta_nonnull = {c: float(feature_table[c].notna().mean()) for c in dsta_cols}
    dsta_coverage = {
        "n_dsta_like_numeric_cols": len(dsta_cols),
        "max_nonnull_frac": max(dsta_nonnull.values()) if dsta_nonnull else 0.0,
        "mean_nonnull_frac": float(np.mean(list(dsta_nonnull.values()))) if dsta_nonnull else 0.0,
    }

    schema = {
        "experiment": "DA-ASA-2D.0h2 Dense DSTA Aggregation (duplicate-column safe)",
        "base_file": str(base_path),
        "key_columns": keep_keys,
        "policy_name_to_id": POLICY_NAME_TO_ID,
        "condition_to_policy": CONDITION_TO_POLICY,
        "numeric_feature_columns": numeric_cols_all,
        "present_by_group": present,
        "missing_expected_features_by_group": missing,
        "dsta_coverage": dsta_coverage,
        "source_report": {"loaded_files": loaded, "merged_files": merge_report},
    }
    summary = {
        "experiment": "DA-ASA-2D.0h2 Dense DSTA Aggregation (duplicate-column safe)",
        "base_file": str(base_path),
        "rows": int(len(feature_table)),
        "columns": int(feature_table.shape[1]),
        "numeric_feature_columns": int(len(numeric_cols_all)),
        "policy_counts": feature_table["policy_class"].value_counts().to_dict(),
        "condition_counts": feature_table["condition"].value_counts().to_dict(),
        "loaded_files_count": len([x for x in loaded if "error" not in x]),
        "merged_files_count": len(merge_report),
        "present_counts_by_group": {k: len(v) for k, v in present.items()},
        "dsta_coverage": dsta_coverage,
        "outputs": {
            "feature_table": str(full_path),
            "numeric_table": str(num_path),
            "schema": str(schema_path),
            "missing_report": str(miss_path),
        },
    }
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    with open(miss_path, "w", encoding="utf-8") as f:
        json.dump({"missing_expected_features_by_group": missing, "present_by_group": present, "dsta_coverage": dsta_coverage, "source_report": schema["source_report"]}, f, ensure_ascii=False, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[done] DA-ASA-2D.0h dense table written")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
