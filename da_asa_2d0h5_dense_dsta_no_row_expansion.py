# -*- coding: utf-8 -*-
"""
DA-ASA-2D.0h5 Dense DSTA Aggregation - no row expansion

Purpose:
  Build a one-row-per-sample policy feature table for DA-ASA-2D.1.
  This version avoids one-to-many expansion caused by per-policy policy_results.

Key fixes vs 2D.0h4:
  - Base table remains 156 rows (or whatever base-file has).
  - Tables with multiple rows per sample/condition are aggregated before merge.
  - condition-level DSTA and policy summaries are broadcast by condition.
  - graph-condition DSTA tables are aggregated by graph_id+condition before merge.
  - no topk/window pivot is used.

Example:
python da_asa_2d0h5_dense_dsta_no_row_expansion.py ^
  --base-file C:\\Users\\ZH\\Desktop\\AGI\\python_script\\da_asa2c_outputs\\da_asa2c_baseline_scores.csv ^
  --include-file C:\\Users\\ZH\\Desktop\\AGI\\python_script\\da_asa2a_outputs\\da_asa2a_baseline_scores.csv ^
  ...
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
    "closure_update": "NO_INTERVENTION",
    "closure_temporal": "NO_INTERVENTION",
    "closure_authority": "NO_INTERVENTION",
    "closure_exception": "CORE_CLOSURE_UPDATE",
    "closure_negation": "CORE_CLOSURE_UPDATE",
    "closure_override": "OVERRIDE_THREE_STAGE",
    "competition_equal_evidence": "EQUAL_EVIDENCE_ORDER",
    "equal_evidence": "EQUAL_EVIDENCE_ORDER",
}

EXPECTED = {
    "baseline_trajectory": [
        "baseline_R_final", "baseline_margin", "boundary_distance_proxy", "R_final", "R_20_22", "R_23_25",
        "clean_conflict_margin", "order_precursor_score", "deltaU", "dR20_25_l2", "dR20_25_pc1"
    ],
    "policy_outcome": [
        "policy_clean_rate", "policy_conflict_rate", "policy_delta_clean_rate", "policy_mean_shift",
        "policy_abs_shift", "target_hit_rate", "nontarget_hit_rate", "specificity", "safety_adjusted_gain",
        "policy_outcome_clean_rate", "policy_outcome_clean_rate_gain", "policy_outcome_mean_delta_R"
    ],
    "dsta_signed_attractor": [
        "dsta_signed_attractor_center_signed_align_vs_clean",
        "dsta_signed_attractor_center_rotation_abs_deg",
        "dsta_signed_attractor_delta_center_norm",
        "dsta_signed_attractor_transport_signed_align_vs_clean",
        "dsta_signed_attractor_transport_rotation_abs_deg",
        "dsta_signed_attractor_delta_align_to_update",
        "dsta_signed_attractor_delta_align_to_override",
        "dsta_signed_attractor_d_to_stable",
        "dsta_signed_attractor_d_to_update",
        "dsta_signed_attractor_d_to_override",
        "dsta_signed_attractor_d_to_exception",
    ],
    "topk_vim": [
        "topk_vim_logit_top1", "topk_vim_logit_mean", "topk_vim_logit_std", "topk_vim_logit_gap_1_2",
        "topk_vim_entropy", "topk_vim_entropy_norm", "topk_vim_spread", "topk_vim_center_norm"
    ],
}


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # pandas may create duplicate column names; keep first occurrence
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]
    df.columns = [str(c).strip() for c in df.columns]
    return df


def normalize_condition(s):
    if pd.isna(s):
        return s
    return str(s).strip()


def read_csv_safe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = clean_columns(df)
    # normalize common condition aliases
    if "condition" not in df.columns:
        for c in ["_condition", "baseline_trajectory_condition", "policy_outcome_condition"]:
            if c in df.columns:
                df = df.rename(columns={c: "condition"})
                break
    if "condition" in df.columns:
        df["condition"] = df["condition"].map(normalize_condition)
    if "sample_id" not in df.columns:
        for c in ["id", "_id", "prompt_id"]:
            if c in df.columns:
                df["sample_id"] = df[c]
                break
    return clean_columns(df)


def group_for_path(path: str) -> str:
    p = path.lower()
    if "dsta" in p and ("signed" in p or "attractor" in p or "paired" in p or "distance" in p or "geometry" in p):
        return "dsta_signed_attractor"
    if "scalar_topk" in p or "topk" in p or "vim" in p:
        return "topk_vim"
    if "policy" in p or "condition_summary" in p:
        return "policy_outcome"
    return "baseline_trajectory"


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "baseline_R_final" in df.columns and "R_final" not in df.columns:
        df["R_final"] = df["baseline_R_final"]
    if "R_final" in df.columns:
        df["baseline_margin"] = pd.to_numeric(df["R_final"], errors="coerce")
        df["boundary_distance_proxy"] = df["baseline_margin"].abs()
    if "baseline_R_final" in df.columns:
        df["baseline_margin"] = pd.to_numeric(df["baseline_R_final"], errors="coerce")
        df["boundary_distance_proxy"] = df["baseline_margin"].abs()
    return df


def assign_policy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "condition" not in df.columns:
        raise RuntimeError("No condition column found in base file.")
    df["policy_class"] = df["condition"].map(CONDITION_TO_POLICY).fillna("NO_INTERVENTION")
    df["policy_id"] = df["policy_class"].map(POLICY_NAME_TO_ID).astype(int)
    df["is_core_closure_target"] = df["condition"].isin(["closure_exception", "closure_negation"]).astype(int)
    df["is_override_target"] = (df["condition"] == "closure_override").astype(int)
    df["is_equal_evidence_target"] = (df["condition"] == "competition_equal_evidence").astype(int)
    df["is_hallucination_like"] = (df["condition"] == "hallucination_like").astype(int)
    df["is_source_claim"] = (df["condition"] == "competition_source_claim").astype(int)
    df["is_protected_no_intervention"] = (df["policy_class"] == "NO_INTERVENTION").astype(int)
    return df


def numeric_cols(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
        else:
            # try coercion for numeric-looking strings
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().mean() > 0.8:
                df[c] = coerced
                cols.append(c)
    return cols


def prefix_columns(df: pd.DataFrame, prefix: str, key_cols: List[str]) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        if c not in key_cols and not str(c).startswith(prefix + "_"):
            rename[c] = f"{prefix}_{c}"
    return df.rename(columns=rename)


def aggregate_condition_level(df: pd.DataFrame, group: str) -> Optional[pd.DataFrame]:
    df = clean_columns(df)
    if "condition" not in df.columns:
        return None
    exclude = ["condition", "sample_id", "graph_id", "id", "prompt_id", "source_file", "text", "top_ids_head", "policy", "applied_actions", "mechanism", "risk_regime", "window", "nearest", "test", "result", "note", "anchor_a", "anchor_b", "x", "a", "b", "available", "condition_i", "condition_j"]
    nums = numeric_cols(df, exclude)
    if not nums:
        return None
    agg = df.groupby("condition", dropna=False)[nums].mean().reset_index()
    cnt = df.groupby("condition", dropna=False).size().reset_index(name="n_rows_agg")
    agg = agg.merge(cnt, on="condition", how="left")
    agg = prefix_columns(agg, group, ["condition"])
    return clean_columns(agg)


def aggregate_graph_condition_level(df: pd.DataFrame, group: str) -> Optional[pd.DataFrame]:
    df = clean_columns(df)
    if not {"graph_id", "condition"}.issubset(df.columns):
        return None
    exclude = ["condition", "sample_id", "graph_id", "id", "prompt_id", "source_file", "text", "top_ids_head", "policy", "applied_actions", "mechanism", "risk_regime", "window"]
    nums = numeric_cols(df, exclude)
    if not nums:
        return None
    agg = df.groupby(["graph_id", "condition"], dropna=False)[nums].mean().reset_index()
    cnt = df.groupby(["graph_id", "condition"], dropna=False).size().reset_index(name="n_rows_agg")
    agg = agg.merge(cnt, on=["graph_id", "condition"], how="left")
    agg = prefix_columns(agg, group, ["graph_id", "condition"])
    return clean_columns(agg)


def make_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    seen = {}
    for c in df.columns:
        if c not in seen:
            seen[c] = 0
            cols.append(c)
        else:
            seen[c] += 1
            cols.append(f"{c}__dup{seen[c]}")
    df = df.copy()
    df.columns = cols
    return df


def merge_no_expand(base: pd.DataFrame, other: pd.DataFrame, keys: List[str], group: str, path: str) -> Tuple[pd.DataFrame, Dict]:
    base = clean_columns(base)
    other = clean_columns(other)
    # only keep keys + columns not already in base
    keep = [k for k in keys if k in other.columns]
    added = []
    for c in other.columns:
        if c in keep:
            continue
        if c not in base.columns:
            keep.append(c)
            added.append(c)
    other2 = other[keep].copy()
    # guarantee uniqueness by aggregating any duplicate key rows before merge
    if other2.duplicated(keys).any():
        nums = [c for c in other2.columns if c not in keys and pd.api.types.is_numeric_dtype(other2[c])]
        nonnums = [c for c in other2.columns if c not in keys and c not in nums]
        parts = []
        if nums:
            parts.append(other2.groupby(keys, dropna=False)[nums].mean())
        if nonnums:
            parts.append(other2.groupby(keys, dropna=False)[nonnums].first())
        if parts:
            other2 = pd.concat(parts, axis=1).reset_index()
        else:
            other2 = other2.drop_duplicates(keys)
    before_rows = len(base)
    merged = base.merge(other2, on=keys, how="left")
    if len(merged) != before_rows:
        raise RuntimeError(f"Row expansion detected while merging {path}: {before_rows} -> {len(merged)}")
    report = {"group": group, "keys": keys, "rows": int(len(other2)), "added_cols": len([c for c in other2.columns if c not in keys]), "added_preview": [c for c in other2.columns if c not in keys][:20], "path": path}
    return clean_columns(merged), report


def dsta_coverage(df: pd.DataFrame) -> Dict:
    cols = [c for c in df.columns if c.startswith("dsta_") and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        return {"n_dsta_like_numeric_cols": 0, "max_nonnull_frac": 0.0, "mean_nonnull_frac": 0.0}
    fracs = [float(df[c].notna().mean()) for c in cols]
    return {"n_dsta_like_numeric_cols": len(cols), "max_nonnull_frac": max(fracs), "mean_nonnull_frac": float(np.mean(fracs))}


def present_by_group(cols: List[str]) -> Dict:
    out = {}
    for group, expected in EXPECTED.items():
        found = []
        for e in expected:
            for c in cols:
                if e.lower() in c.lower() or c.lower().endswith(e.lower()):
                    found.append(c)
                    break
        out[group] = sorted(set(found))
    return out


def missing_by_group(cols: List[str]) -> Dict:
    pres = present_by_group(cols)
    out = {}
    for group, expected in EXPECTED.items():
        found_expected = []
        for e in expected:
            if any(e.lower() in c.lower() or c.lower().endswith(e.lower()) for c in cols):
                found_expected.append(e)
        out[group] = [e for e in expected if e not in found_expected]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-file", required=True)
    ap.add_argument("--include-file", action="append", default=[])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    base_path = Path(args.base_file)
    out_dir = Path(args.out_dir) if args.out_dir else base_path.parent.parent / "da_asa_2d0h5_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = read_csv_safe(str(base_path))
    base = add_basic_features(base)
    base = assign_policy(base)
    base = clean_columns(base)
    original_rows = len(base)

    loaded = []
    merged_reports = []
    merged = base.copy()

    for path in args.include_file:
        if not path:
            continue
        if str(Path(path).resolve()) == str(base_path.resolve()):
            continue
        if not Path(path).exists():
            loaded.append({"path": path, "error": "not found"})
            continue
        try:
            df = read_csv_safe(path)
            group = group_for_path(path)
            df = add_basic_features(df)
            loaded.append({"path": path, "group": group, "rows": int(len(df)), "columns": list(df.columns)[:80]})

            candidates = []
            # graph-condition dense aggregation first if possible
            gc = aggregate_graph_condition_level(df, group)
            if gc is not None and {"graph_id", "condition"}.issubset(gc.columns) and {"graph_id", "condition"}.issubset(merged.columns):
                candidates.append((gc, ["graph_id", "condition"]))
            # condition-level aggregation for all tables
            cond = aggregate_condition_level(df, group)
            if cond is not None and "condition" in cond.columns:
                candidates.append((cond, ["condition"]))
            # sample-level aggregation only when sample_id+condition uniquely possible
            if {"sample_id", "condition"}.issubset(df.columns):
                exclude = ["condition", "sample_id", "graph_id", "source_file", "text", "top_ids_head", "policy", "applied_actions"]
                nums = numeric_cols(df, exclude)
                if nums:
                    tmp = df[["sample_id", "condition"] + nums].copy()
                    tmp = tmp.groupby(["sample_id", "condition"], dropna=False)[nums].mean().reset_index()
                    tmp = prefix_columns(tmp, group, ["sample_id", "condition"])
                    candidates.append((tmp, ["sample_id", "condition"]))

            for other, keys in candidates:
                # Skip if no new columns after prefixing
                new_cols = [c for c in other.columns if c not in keys and c not in merged.columns]
                if not new_cols:
                    continue
                merged, rep = merge_no_expand(merged, other, keys, group, path)
                merged_reports.append(rep)
        except Exception as e:
            loaded.append({"path": path, "error": repr(e)})

    if len(merged) != original_rows:
        raise RuntimeError(f"Final row count changed unexpectedly: {original_rows} -> {len(merged)}")

    # Remove direct label leakage for numeric output but keep labels/key columns
    table = merged.copy()
    numeric_cols = [c for c in table.columns if pd.api.types.is_numeric_dtype(table[c])]
    key_cols = [c for c in ["sample_id", "graph_id", "condition", "policy_class", "policy_id"] if c in table.columns]
    numeric_table = table[key_cols + [c for c in numeric_cols if c not in key_cols]].copy()

    feature_table_path = out_dir / "da_asa2d_policy_feature_table.csv"
    numeric_path = out_dir / "da_asa2d_policy_feature_numeric.csv"
    schema_path = out_dir / "da_asa2d_feature_schema.json"
    missing_path = out_dir / "da_asa2d_missing_report.json"
    summary_path = out_dir / "da_asa2d_summary.json"

    table.to_csv(feature_table_path, index=False, encoding="utf-8-sig")
    numeric_table.to_csv(numeric_path, index=False, encoding="utf-8-sig")

    cols = list(table.columns)
    schema = {
        "experiment": "DA-ASA-2D.0h5 Dense DSTA Aggregation (no row expansion)",
        "base_file": str(base_path),
        "key_columns": key_cols,
        "policy_name_to_id": POLICY_NAME_TO_ID,
        "condition_to_policy": CONDITION_TO_POLICY,
        "numeric_feature_columns": [c for c in numeric_table.columns if c not in key_cols],
        "present_by_group": present_by_group(cols),
        "missing_expected_features_by_group": missing_by_group(cols),
        "dsta_coverage": dsta_coverage(table),
        "source_report": {"loaded_files": loaded, "merged_files": merged_reports},
    }
    missing = {
        "missing_expected_features_by_group": schema["missing_expected_features_by_group"],
        "present_by_group": schema["present_by_group"],
        "dsta_coverage": schema["dsta_coverage"],
        "source_report": schema["source_report"],
    }
    summary = {
        "experiment": schema["experiment"],
        "base_file": str(base_path),
        "rows": int(len(table)),
        "columns": int(table.shape[1]),
        "numeric_feature_columns": int(len(schema["numeric_feature_columns"])),
        "policy_counts": table["policy_class"].value_counts().to_dict() if "policy_class" in table.columns else {},
        "condition_counts": table["condition"].value_counts().to_dict() if "condition" in table.columns else {},
        "loaded_files_count": len([x for x in loaded if "error" not in x]),
        "merged_files_count": len(merged_reports),
        "present_counts_by_group": {k: len(v) for k, v in schema["present_by_group"].items()},
        "dsta_coverage": schema["dsta_coverage"],
        "outputs": {
            "feature_table": str(feature_table_path),
            "numeric_table": str(numeric_path),
            "schema": str(schema_path),
            "missing_report": str(missing_path),
        },
    }

    schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    missing_path.write_text(json.dumps(missing, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
