# -*- coding: utf-8 -*-
"""
DA-ASA-2D.0g Explicit-File Feature Assembly

Purpose
-------
Build a DA-ASA-2D policy feature table using ONLY explicitly provided
DA-ASA 2A/2B/2C output files. This avoids scanning hundreds of unrelated CSVs.

Recommended Windows run:
    cd /d C:\\Users\\ZH\\Desktop\\AGI\\python_script
    python da_asa_2d0g_explicit_files.py ^
      --base-file C:\\Users\\ZH\\Desktop\\AGI\\python_script\\da_asa2c_outputs\\da_asa2c_baseline_scores.csv ^
      --include-file C:\\Users\\ZH\\Desktop\\AGI\\python_script\\da_asa2a_outputs\\da_asa2a_baseline_scores.csv ^
      --include-file C:\\Users\\ZH\\Desktop\\AGI\\python_script\\da_asa2a_outputs\\da_asa2a_policy_results.csv ^
      --include-file C:\\Users\\ZH\\Desktop\\AGI\\python_script\\da_asa2a_outputs\\da_asa2a_condition_summary.csv ^
      --include-file C:\\Users\\ZH\\Desktop\\AGI\\python_script\\da_asa2b_outputs\\da_asa2b_baseline_scores.csv ^
      --include-file C:\\Users\\ZH\\Desktop\\AGI\\python_script\\da_asa2b_outputs\\da_asa2b_policy_results.csv ^
      --include-file C:\\Users\\ZH\\Desktop\\AGI\\python_script\\da_asa2b_outputs\\da_asa2b_condition_summary.csv ^
      --include-file C:\\Users\\ZH\\Desktop\\AGI\\python_script\\da_asa2c_outputs\\da_asa2c_policy_results.csv ^
      --include-file C:\\Users\\ZH\\Desktop\\AGI\\python_script\\da_asa2c_outputs\\da_asa2c_policy_summary.csv ^
      --include-file C:\\Users\\ZH\\Desktop\\AGI\\python_script\\da_asa2c_outputs\\da_asa2c_condition_summary.csv

Outputs
-------
    da_asa_2d0g_outputs/da_asa2d_policy_feature_table.csv
    da_asa_2d0g_outputs/da_asa2d_policy_feature_numeric.csv
    da_asa_2d0g_outputs/da_asa2d_summary.json
    da_asa_2d0g_outputs/da_asa2d_missing_report.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    # core repair targets
    "closure_exception": "CORE_CLOSURE_UPDATE",
    "closure_negation": "CORE_CLOSURE_UPDATE",
    "closure_override": "OVERRIDE_THREE_STAGE",
    "competition_equal_evidence": "EQUAL_EVIDENCE_ORDER",
    "equal_evidence": "EQUAL_EVIDENCE_ORDER",
}

EXPECTED_CANONICAL = {
    "baseline_trajectory": [
        "R_final", "R_20_22", "R_23_25", "baseline_margin", "clean_conflict_margin",
        "boundary_distance_proxy", "order_precursor_score", "deltaU", "dR20_25_l2", "dR20_25_pc1",
    ],
    "policy_outcome": [
        "policy_clean_rate", "policy_conflict_rate", "policy_delta_clean_rate", "policy_mean_shift",
        "policy_abs_shift", "target_hit_rate", "nontarget_hit_rate", "specificity",
        "safety_adjusted_gain",
    ],
    "dsta_signed_attractor": [
        "center_signed_align", "center_rotation_abs_deg", "delta_center_norm",
        "transport_signed_align", "transport_rotation_abs_deg", "delta_align_to_update",
        "delta_align_to_override", "nearest_attractor_distance", "attractor_distance_update",
        "attractor_distance_override", "attractor_distance_stable", "attractor_distance_wrong",
    ],
    "direction_spectrum": [
        "basis_rotation", "basis_similarity", "weight_shift", "weight_entropy", "dominance_gap",
        "critical_weight_transport", "commit_weight_transport", "basis_rotation_0_6",
        "weight_shift_7_19", "commit_weight_shift_23_25",
    ],
    "topk_vim": [
        "topk_center_trajectory", "topk_spread", "topk_entropy", "topk_delta_norm",
        "topk_delta_structure_score", "vim_center_drift", "vim_spread_delta", "vim_entropy_delta",
        "topk_jaccard_to_clean", "topk_center_cos_to_clean",
    ],
}


def read_csv_safely(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        df["source_file"] = str(path)
        return df
    except Exception as exc:
        print(f"[warn] failed to read {path}: {exc}")
        return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # common aliases
    aliases = {
        "sample_index": "sample_id",
        "idx": "sample_id",
        "cond": "condition",
        "Condition": "condition",
        "Graph": "graph_id",
        "PromptID": "prompt_id",
    }
    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    if "condition" in df.columns:
        df["condition"] = df["condition"].astype(str).str.strip()
    return df


def add_sample_id_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "sample_id" not in df.columns:
        if "prompt_id" in df.columns:
            df["sample_id"] = df["prompt_id"]
        else:
            df["sample_id"] = np.arange(len(df))
    return df


def assign_policy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "condition" not in df.columns:
        raise RuntimeError("No condition column found in base table; cannot assign policy labels.")
    df["policy_class"] = df["condition"].map(CONDITION_TO_POLICY).fillna("NO_INTERVENTION")
    df["policy_id"] = df["policy_class"].map(POLICY_NAME_TO_ID).astype(int)
    # one-hot target flags
    df["is_core_closure_target"] = df["condition"].isin(["closure_exception", "closure_negation"]).astype(int)
    df["is_override_target"] = (df["condition"] == "closure_override").astype(int)
    df["is_equal_evidence_target"] = df["condition"].isin(["competition_equal_evidence", "equal_evidence"]).astype(int)
    df["is_hallucination_like"] = (df["condition"] == "hallucination_like").astype(int)
    df["is_source_claim"] = df["condition"].isin(["competition_source_claim", "source_claim"]).astype(int)
    df["is_protected_no_intervention"] = (df["policy_class"] == "NO_INTERVENTION").astype(int)
    return df


def derive_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # final margin aliases
    if "R_final" not in df.columns:
        for c in ["baseline_R_final", "R_final_margin", "baseline_baseline_R_final"]:
            if c in df.columns:
                df["R_final"] = pd.to_numeric(df[c], errors="coerce")
                break
    if "C_logit_final" not in df.columns:
        for c in ["baseline_C_logit", "final_clean_logit", "baseline_baseline_C_logit"]:
            if c in df.columns:
                df["C_logit_final"] = pd.to_numeric(df[c], errors="coerce")
                break
    if "E_logit_final" not in df.columns:
        for c in ["baseline_E_logit", "final_conflict_logit", "baseline_baseline_E_logit"]:
            if c in df.columns:
                df["E_logit_final"] = pd.to_numeric(df[c], errors="coerce")
                break
    if "R_final" not in df.columns and {"C_logit_final", "E_logit_final"}.issubset(df.columns):
        df["R_final"] = pd.to_numeric(df["C_logit_final"], errors="coerce") - pd.to_numeric(df["E_logit_final"], errors="coerce")
    if "baseline_margin" not in df.columns and "R_final" in df.columns:
        df["baseline_margin"] = df["R_final"]
    if "boundary_distance_proxy" not in df.columns and "R_final" in df.columns:
        df["boundary_distance_proxy"] = df["R_final"].abs()

    # layer-window features when available
    r_cols_20_22 = [c for c in ["R_L20", "R_L21", "R_L22"] if c in df.columns]
    r_cols_23_25 = [c for c in ["R_L23", "R_L24", "R_L25"] if c in df.columns]
    if r_cols_20_22 and "R_20_22" not in df.columns:
        df["R_20_22"] = df[r_cols_20_22].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    if r_cols_23_25 and "R_23_25" not in df.columns:
        df["R_23_25"] = df[r_cols_23_25].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    d_cols = [c for c in ["dR_L20", "dR_L21", "dR_L22", "dR_L23", "dR_L24", "dR_L25"] if c in df.columns]
    if d_cols and "dR20_25_l2" not in df.columns:
        X = df[d_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
        df["dR20_25_l2"] = np.linalg.norm(X, axis=1)
    if "DeltaU" in df.columns and "deltaU" not in df.columns:
        df["deltaU"] = pd.to_numeric(df["DeltaU"], errors="coerce")
    if "D_B_proxy" in df.columns and "boundary_distance_proxy" not in df.columns:
        df["boundary_distance_proxy"] = pd.to_numeric(df["D_B_proxy"], errors="coerce")
    return df


def infer_group(path: Path) -> str:
    s = path.name.lower()
    if "policy_results" in s or "policy_summary" in s or "condition_summary" in s or "verdict" in s:
        return "policy_outcome"
    if "baseline" in s or "scores" in s or "features" in s or "merged_shift" in s or "steering_results" in s:
        return "baseline_trajectory"
    if "attractor" in s or "specificity" in s or "signed" in s:
        return "dsta_signed_attractor"
    if "direction" in s or "spectrum" in s:
        return "direction_spectrum"
    if "topk" in s or "vim" in s:
        return "topk_vim"
    return "misc"


def prefix_nonkeys(df: pd.DataFrame, prefix: str, keys: List[str]) -> pd.DataFrame:
    df = df.copy()
    rename = {}
    for c in df.columns:
        if c in keys or c == "source_file":
            continue
        if c in ["condition", "prompt_id", "graph_id", "sample_id"]:
            continue
        if not c.startswith(prefix + "_"):
            rename[c] = f"{prefix}_{c}"
    return df.rename(columns=rename)


def aggregate_to_keys(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    """Ensure one row per merge key; aggregate numeric by mean, nonnumeric by first."""
    df = df.copy()
    if not keys or not all(k in df.columns for k in keys):
        return df
    if df.duplicated(keys).any():
        num_cols = [c for c in df.columns if c not in keys and pd.api.types.is_numeric_dtype(df[c])]
        other_cols = [c for c in df.columns if c not in keys and c not in num_cols]
        agg = {c: "mean" for c in num_cols}
        agg.update({c: "first" for c in other_cols})
        df = df.groupby(keys, as_index=False).agg(agg)
    return df


def merge_table(base: pd.DataFrame, other: pd.DataFrame, group: str) -> Tuple[pd.DataFrame, Optional[Dict]]:
    base = base.copy()
    other = normalize_columns(other)
    other = add_sample_id_if_missing(other)
    other = derive_baseline_features(other)

    # Choose safest keys, preferring sample-level.
    candidate_key_sets = [
        ["sample_id", "condition"],
        ["prompt_id", "condition"],
        ["graph_id", "condition"],
        ["condition"],
    ]
    keys = None
    for ks in candidate_key_sets:
        if all(k in base.columns for k in ks) and all(k in other.columns for k in ks):
            keys = ks
            break
    if keys is None:
        return base, None

    other = aggregate_to_keys(other, keys)
    prefix = group
    other = prefix_nonkeys(other, prefix, keys)

    # Drop duplicate non-key columns that already exist in base.
    drop_cols = [c for c in other.columns if c not in keys and c in base.columns]
    if drop_cols:
        other = other.drop(columns=drop_cols)
    if len([c for c in other.columns if c not in keys]) == 0:
        return base, None

    before_cols = set(base.columns)
    try:
        merged = base.merge(other, on=keys, how="left", suffixes=("", f"_{prefix}"))
    except Exception:
        # Emergency de-dup rename
        ren = {}
        for c in other.columns:
            if c not in keys and c in base.columns:
                ren[c] = c + "__dup"
        other = other.rename(columns=ren)
        merged = base.merge(other, on=keys, how="left", suffixes=("", f"_{prefix}"))
    # Remove duplicate columns generated accidentally
    merged = merged.loc[:, ~merged.columns.duplicated()]
    added = [c for c in merged.columns if c not in before_cols]
    report = {
        "group": group,
        "keys": keys,
        "rows": int(len(other)),
        "added_cols": len(added),
        "added_preview": added[:20],
    }
    return merged, report


def present_expected(df: pd.DataFrame) -> Dict[str, List[str]]:
    out = {}
    lower_map = {c.lower(): c for c in df.columns}
    for group, feats in EXPECTED_CANONICAL.items():
        present = []
        for f in feats:
            if f in df.columns:
                present.append(f)
            elif f.lower() in lower_map:
                present.append(lower_map[f.lower()])
            else:
                # also accept prefixed versions ending with canonical name
                matches = [c for c in df.columns if c.lower().endswith("_" + f.lower())]
                if matches:
                    present.extend(matches[:3])
        out[group] = sorted(set(present))
    return out


def numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {
        "policy_id", "sample_id", "graph_id", "prompt_id", "entity_id", "relation_id",
        "clean_token_id", "conflict_token_id",
    }
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-file", required=True)
    ap.add_argument("--include-file", action="append", default=[])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    base_path = Path(args.base_file)
    if not base_path.exists():
        raise FileNotFoundError(f"base file not found: {base_path}")
    out_dir = Path(args.out_dir) if args.out_dir else base_path.parent.parent / "da_asa_2d0g_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[base] {base_path}")
    base = read_csv_safely(base_path)
    if base is None:
        raise RuntimeError("failed to read base file")
    base = normalize_columns(base)
    base = add_sample_id_if_missing(base)
    base = derive_baseline_features(base)
    base = assign_policy(base)

    loaded_reports = []
    merge_reports = []
    merged = base.copy()

    # Include base as a regular source if useful, then user files.
    files = []
    seen = set()
    for p in [str(base_path)] + args.include_file:
        pp = Path(p)
        if pp.exists() and str(pp.resolve()) not in seen:
            files.append(pp)
            seen.add(str(pp.resolve()))
        else:
            print(f"[warn] include-file missing or duplicate: {p}")

    print(f"[files] explicit count={len(files)}")
    for i, path in enumerate(files, 1):
        group = infer_group(path)
        print(f"[read] {i}/{len(files)} {path.name} -> {group}")
        df = read_csv_safely(path)
        if df is None or df.empty:
            continue
        df = normalize_columns(df)
        df = add_sample_id_if_missing(df)
        loaded_reports.append({
            "path": str(path),
            "group": group,
            "rows": int(len(df)),
            "columns": list(df.columns),
        })
        if path.resolve() == base_path.resolve():
            continue
        merged2, rep = merge_table(merged, df, group)
        if rep and rep["added_cols"] > 0:
            rep["path"] = str(path)
            merge_reports.append(rep)
            merged = merged2

    merged = derive_baseline_features(merged)
    merged = assign_policy(merged)
    present = present_expected(merged)
    missing = {g: [f for f in feats if f not in present.get(g, [])] for g, feats in EXPECTED_CANONICAL.items()}
    num_cols = numeric_feature_columns(merged)

    feature_table = out_dir / "da_asa2d_policy_feature_table.csv"
    numeric_table = out_dir / "da_asa2d_policy_feature_numeric.csv"
    schema_path = out_dir / "da_asa2d_feature_schema.json"
    missing_path = out_dir / "da_asa2d_missing_report.json"
    summary_path = out_dir / "da_asa2d_summary.json"

    merged.to_csv(feature_table, index=False, encoding="utf-8-sig")
    keep_cols = [c for c in ["sample_id", "graph_id", "prompt_id", "condition", "policy_class", "policy_id"] if c in merged.columns] + num_cols
    merged[keep_cols].to_csv(numeric_table, index=False, encoding="utf-8-sig")

    schema = {
        "experiment": "DA-ASA-2D.0g Explicit-File Feature Assembly",
        "base_file": str(base_path),
        "key_columns": [c for c in ["sample_id", "graph_id", "prompt_id", "condition"] if c in merged.columns],
        "policy_name_to_id": POLICY_NAME_TO_ID,
        "condition_to_policy": CONDITION_TO_POLICY,
        "numeric_feature_columns": num_cols,
        "present_by_group": present,
        "missing_expected_features_by_group": missing,
        "source_report": {
            "loaded_files": loaded_reports,
            "merged_files": merge_reports,
        },
    }
    summary = {
        "experiment": "DA-ASA-2D.0g Explicit-File Feature Assembly",
        "base_file": str(base_path),
        "rows": int(len(merged)),
        "columns": int(len(merged.columns)),
        "numeric_feature_columns": int(len(num_cols)),
        "policy_counts": merged["policy_class"].value_counts().to_dict(),
        "condition_counts": merged["condition"].value_counts().to_dict() if "condition" in merged.columns else {},
        "loaded_files_count": len(loaded_reports),
        "merged_files_count": len(merge_reports),
        "present_counts_by_group": {k: len(v) for k, v in present.items()},
        "outputs": {
            "feature_table": str(feature_table),
            "numeric_table": str(numeric_table),
            "schema": str(schema_path),
            "missing_report": str(missing_path),
        },
    }
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    with open(missing_path, "w", encoding="utf-8") as f:
        json.dump({"missing_expected_features_by_group": missing, "present_by_group": present, "source_report": schema["source_report"]}, f, ensure_ascii=False, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[DONE]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
