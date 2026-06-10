# -*- coding: utf-8 -*-
r"""
DA-ASA-2D.0b Recursive Feature Assembly
---------------------------------------
Purpose:
  Assemble policy-class feature table for DA-ASA-2D from existing DSTA/DA-ASA outputs.

Main fix over 2D.0:
  - If executed inside an output folder such as da_asa1c_outputs, automatically search the parent project folder.
  - Recursively discover sibling output folders: dsta*, da_asa*, asa*, phasemap*, topk*, vim*.
  - Derive canonical expected features from prefixed columns when possible.
  - Produce a stronger missing/discovery report.

Usage:
  cd /d C:\Users\ZH\Desktop\AGI\python_script
  python da_asa_2d0b_feature_assembly_recursive.py

Optional:
  python da_asa_2d0b_feature_assembly_recursive.py --root C:\Users\ZH\Desktop\AGI\python_script
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
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
    # protected / no intervention
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

    # enabled controllers
    "closure_exception": "CORE_CLOSURE_UPDATE",
    "closure_negation": "CORE_CLOSURE_UPDATE",
    "closure_override": "OVERRIDE_THREE_STAGE",
    "competition_equal_evidence": "EQUAL_EVIDENCE_ORDER",
    "equal_evidence": "EQUAL_EVIDENCE_ORDER",

    # If these are present from older broader datasets, keep conservative unless 2C policy explicitly enables them.
    "closure_update": "NO_INTERVENTION",
    "closure_temporal": "NO_INTERVENTION",
    "closure_authority": "NO_INTERVENTION",
}

EXPECTED_GROUPS = {
    "dsta_signed_attractor": [
        "center_signed_align",
        "center_rotation_abs_deg",
        "delta_center_norm",
        "transport_signed_align",
        "transport_rotation_abs_deg",
        "delta_align_to_update",
        "delta_align_to_override",
        "nearest_attractor_distance",
        "attractor_distance_update",
        "attractor_distance_override",
        "attractor_distance_stable",
        "attractor_distance_wrong",
    ],
    "direction_spectrum": [
        "basis_rotation",
        "basis_similarity",
        "weight_shift",
        "weight_entropy",
        "dominance_gap",
        "critical_weight_transport",
        "commit_weight_transport",
        "basis_rotation_0_6",
        "weight_shift_7_19",
        "commit_weight_shift_23_25",
    ],
    "baseline_trajectory": [
        "R_final",
        "R_20_22",
        "R_23_25",
        "baseline_margin",
        "clean_conflict_margin",
        "boundary_distance_proxy",
        "order_precursor_score",
        "deltaU",
        "dR20_25_l2",
        "dR20_25_pc1",
    ],
    "topk_vim": [
        "topk_center_trajectory",
        "topk_spread",
        "topk_entropy",
        "topk_delta_norm",
        "topk_delta_structure_score",
        "vim_center_drift",
        "vim_spread_delta",
        "vim_entropy_delta",
        "topk_jaccard_to_clean",
        "topk_center_cos_to_clean",
    ],
}

# Canonical feature aliases. The script searches these substrings in normalized column names.
ALIASES = {
    "R_final": ["r_final", "baseline_r_final", "baseline_trajectory_baseline_r_final"],
    "baseline_margin": ["baseline_margin", "baseline_r_final", "baseline_trajectory_baseline_r_final"],
    "clean_conflict_margin": ["clean_conflict_margin", "c_minus_e", "r_final", "baseline_r_final"],
    "R_20_22": ["r_20_22", "boundary_r", "boundary_r_mean", "b_seq", "boundary_seq"],
    "R_23_25": ["r_23_25", "late_r", "late_seq", "basin_r"],
    "boundary_distance_proxy": ["boundary_distance", "d_b", "db", "dist_to_boundary", "abs_r_final"],
    "order_precursor_score": ["order_precursor", "position_ridge", "w_deltau", "delta_u_precursor"],
    "deltaU": ["deltau", "delta_u", "pc1_delta_r", "du"],
    "dR20_25_l2": ["dr20_25_l2", "delta_r_l2", "decision_dr_l2"],
    "dR20_25_pc1": ["dr20_25_pc1", "delta_r_pc1", "decision_dr_pc1"],

    "center_signed_align": ["center_signed_align", "signed_align_center", "center_align_signed"],
    "center_rotation_abs_deg": ["center_rotation_abs_deg", "rotation_abs_deg", "center_angle_deg"],
    "delta_center_norm": ["delta_center_norm", "center_delta_norm", "dcenter_norm"],
    "transport_signed_align": ["transport_signed_align", "signed_transport_align"],
    "transport_rotation_abs_deg": ["transport_rotation_abs_deg", "transport_angle_deg"],
    "delta_align_to_update": ["delta_align_to_update", "align_to_update", "to_update_align"],
    "delta_align_to_override": ["delta_align_to_override", "align_to_override", "to_override_align"],
    "nearest_attractor_distance": ["nearest_attractor_distance", "nearest_attr_dist", "attractor_dist_min"],
    "attractor_distance_update": ["attractor_distance_update", "dist_update", "update_dist"],
    "attractor_distance_override": ["attractor_distance_override", "dist_override", "override_dist"],
    "attractor_distance_stable": ["attractor_distance_stable", "dist_stable", "stable_dist"],
    "attractor_distance_wrong": ["attractor_distance_wrong", "dist_wrong", "wrong_dist", "hallucination_dist"],

    "basis_rotation": ["basis_rotation", "basis_angle", "basis_rot"],
    "basis_similarity": ["basis_similarity", "basis_cos", "basis_align"],
    "weight_shift": ["weight_shift", "w_shift", "direction_weight_shift"],
    "weight_entropy": ["weight_entropy", "w_entropy", "direction_entropy"],
    "dominance_gap": ["dominance_gap", "weight_gap", "top1_top2_gap"],
    "critical_weight_transport": ["critical_weight_transport", "critical_transport"],
    "commit_weight_transport": ["commit_weight_transport", "commit_transport"],
    "basis_rotation_0_6": ["basis_rotation_0_6", "basis_rot_0_6", "shallow_basis_rotation"],
    "weight_shift_7_19": ["weight_shift_7_19", "mid_weight_shift", "weight_shift_mid"],
    "commit_weight_shift_23_25": ["commit_weight_shift_23_25", "late_weight_shift", "commit_weight_shift"],

    "topk_center_trajectory": ["topk_center_trajectory", "center_trajectory", "topk_center_traj"],
    "topk_spread": ["topk_spread", "spread"],
    "topk_entropy": ["topk_entropy", "entropy"],
    "topk_delta_norm": ["topk_delta_norm", "topk_delta", "delta_topk_norm"],
    "topk_delta_structure_score": ["topk_delta_structure_score", "structure_score", "topk_structure"],
    "vim_center_drift": ["vim_center_drift", "center_drift"],
    "vim_spread_delta": ["vim_spread_delta", "spread_delta"],
    "vim_entropy_delta": ["vim_entropy_delta", "entropy_delta"],
    "topk_jaccard_to_clean": ["topk_jaccard_to_clean", "jaccard_to_clean", "topk_jaccard_clean"],
    "topk_center_cos_to_clean": ["topk_center_cos_to_clean", "center_cos_to_clean", "topk_center_cos_clean"],
}

GROUP_KEYWORDS = {
    "dsta_signed_attractor": ["dsta", "signed", "attractor", "align", "rotation", "asa2", "da_asa"],
    "direction_spectrum": ["direction", "spectrum", "basis", "weight", "transport", "dsta"],
    "baseline_trajectory": ["baseline", "score", "trajectory", "delta", "margin", "da_asa", "asa"],
    "topk_vim": ["topk", "vim", "center", "jaccard", "spread", "entropy"],
}

KEY_COLS = ["sample_id", "graph_id", "prompt_id", "condition"]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def infer_root(user_root: Optional[str]) -> Path:
    if user_root:
        return Path(user_root).expanduser().resolve()
    cwd = Path.cwd().resolve()
    # If running inside a known output folder, search the parent project folder.
    if re.search(r"(_outputs|outputs)$", cwd.name.lower()):
        return cwd.parent
    return cwd


def candidate_roots(root: Path) -> List[Path]:
    roots = [root]
    # If root itself is da_asa1c_outputs, include parent.
    if re.search(r"(_outputs|outputs)$", root.name.lower()):
        roots.append(root.parent)
    # Include script folder if different from cwd.
    try:
        script_dir = Path(__file__).resolve().parent
        roots.append(script_dir)
        if re.search(r"(_outputs|outputs)$", script_dir.name.lower()):
            roots.append(script_dir.parent)
    except Exception:
        pass
    # Deduplicate preserving order.
    out = []
    seen = set()
    for r in roots:
        try:
            rr = r.resolve()
        except Exception:
            rr = r
        if str(rr).lower() not in seen and rr.exists():
            out.append(rr)
            seen.add(str(rr).lower())
    return out


def discover_csvs(roots: List[Path], max_files: int = 5000) -> List[Path]:
    found = []
    seen = set()
    for root in roots:
        for p in root.rglob("*.csv"):
            s = str(p).lower()
            if "da_asa_2d0_outputs" in s:
                continue
            if "__pycache__" in s:
                continue
            if s not in seen:
                found.append(p)
                seen.add(s)
                if len(found) >= max_files:
                    return found
    return found


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
            if df.empty:
                return None
            return df
        except Exception:
            return None


def file_relevance(path: Path, df: pd.DataFrame) -> Dict[str, int]:
    text = " ".join([path.name.lower(), path.parent.name.lower()] + [norm(c) for c in df.columns])
    scores = {}
    for group, kws in GROUP_KEYWORDS.items():
        scores[group] = sum(1 for kw in kws if kw in text)
    return scores


def ensure_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Normalize common aliases to canonical keys.
    colmap = {norm(c): c for c in df.columns}
    for key in KEY_COLS:
        if key not in df.columns:
            aliases = [
                key,
                key.replace("_id", ""),
                f"baseline_trajectory_{key}",
                f"daasa_{key}",
                f"da_asa_{key}",
                f"dsta_{key}",
                f"source_{key}",
            ]
            # exact normalized aliases first
            for alias in aliases:
                if norm(alias) in colmap:
                    df[key] = df[colmap[norm(alias)]]
                    break
        if key not in df.columns:
            # fallback: any column whose normalized name ends with the canonical key
            # e.g. baseline_trajectory_condition, xxx_graph_id
            for ncol, orig in colmap.items():
                if ncol.endswith(norm(key)) or ncol.endswith(norm(key.replace("_id", ""))):
                    df[key] = df[orig]
                    break
    if "condition" in df.columns:
        df["condition"] = df["condition"].astype(str).str.strip()
    if "graph_id" in df.columns:
        df["graph_id"] = df["graph_id"].astype(str).str.strip()
    if "sample_id" not in df.columns:
        # Stable fallback based on row order; only useful for baseline table creation.
        df["sample_id"] = [f"row_{i:05d}" for i in range(len(df))]
    if "prompt_id" not in df.columns:
        df["prompt_id"] = np.nan
    return df


def prefix_nonkeys(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        if c in KEY_COLS:
            continue
        if c.startswith(prefix + "_"):
            continue
        rename[c] = f"{prefix}_{c}"
    return df.rename(columns=rename)


def select_base_table(tables: List[Tuple[Path, pd.DataFrame, Dict[str, int]]]) -> Tuple[Path, pd.DataFrame]:
    # Prefer tables with condition and graph_id and many rows, especially baseline/score files.
    # IMPORTANT: policy assignment requires condition, so never choose a base without condition
    # when any condition-bearing table exists.
    candidates = []
    diagnostics = []
    for path, df, scores in tables:
        df2 = ensure_keys(df)
        cols = set(df2.columns)
        diagnostics.append({"path": str(path), "rows": int(len(df2)), "columns": list(df2.columns)[:40], "has_condition": "condition" in cols})
        if "condition" not in cols:
            continue
        score = len(df2)
        if "graph_id" in cols: score += 5000
        if "sample_id" in cols: score += 2000
        if scores.get("baseline_trajectory", 0) > 0: score += 1000 * scores["baseline_trajectory"]
        lname = path.name.lower()
        if "baseline_scores" in lname or "baseline" in lname: score += 10000
        if "da_asa" in lname or "dsta" in lname: score += 1000
        candidates.append((score, path, df2))
    if not candidates:
        msg = ["No usable base table with a condition column found.", "Candidate CSV diagnostics:"]
        for d in diagnostics[:30]:
            msg.append(f"- {d['path']} | rows={d['rows']} | has_condition={d['has_condition']} | cols={d['columns']}")
        raise RuntimeError("\n".join(msg))
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, path, df = candidates[0]
    return path, df


def merge_table(base: pd.DataFrame, other: pd.DataFrame, prefix: str) -> pd.DataFrame:
    base = base.copy()
    other = other.copy()
    # Choose merge keys from most to least specific.
    key_sets = [
        ["sample_id", "graph_id", "condition"],
        ["graph_id", "condition"],
        ["sample_id"],
        ["condition"],
    ]
    for keys in key_sets:
        if all(k in base.columns and k in other.columns for k in keys):
            # Avoid many-to-many explosions. Aggregate numeric/object first if duplicates.
            if other.duplicated(keys).any():
                agg = {}
                for c in other.columns:
                    if c in keys:
                        continue
                    if pd.api.types.is_numeric_dtype(other[c]):
                        agg[c] = "mean"
                    else:
                        agg[c] = lambda x: x.dropna().iloc[0] if len(x.dropna()) else np.nan
                other = other.groupby(keys, as_index=False).agg(agg)
            before_cols = set(base.columns)
            merged = base.merge(other, on=keys, how="left", suffixes=("", f"_{prefix}"))
            # Accept merge only if it adds any non-all-null column.
            added = [c for c in merged.columns if c not in before_cols]
            if added and any(merged[c].notna().any() for c in added):
                return merged
    return base


def derive_canonical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    norm_to_col = {norm(c): c for c in df.columns}

    def find_col(canon: str) -> Optional[str]:
        aliases = ALIASES.get(canon, [canon])
        # exact normalized match first
        for a in aliases:
            na = norm(a)
            if na in norm_to_col:
                return norm_to_col[na]
        # substring match second
        cols = list(df.columns)
        ncols = [(c, norm(c)) for c in cols]
        for a in aliases:
            na = norm(a)
            for c, nc in ncols:
                if na and na in nc:
                    return c
        return None

    for group, feats in EXPECTED_GROUPS.items():
        for feat in feats:
            if feat in df.columns:
                continue
            c = find_col(feat)
            if c is not None:
                df[feat] = pd.to_numeric(df[c], errors="coerce")

    # Additional useful derivations from C/E logits.
    c_cols = [c for c in df.columns if norm(c).endswith("c_logit") or "baseline_c_logit" in norm(c)]
    e_cols = [c for c in df.columns if norm(c).endswith("e_logit") or "baseline_e_logit" in norm(c)]
    if "R_final" not in df.columns and c_cols and e_cols:
        df["R_final"] = pd.to_numeric(df[c_cols[0]], errors="coerce") - pd.to_numeric(df[e_cols[0]], errors="coerce")
    if "baseline_margin" not in df.columns and "R_final" in df.columns:
        df["baseline_margin"] = df["R_final"]
    if "clean_conflict_margin" not in df.columns and "R_final" in df.columns:
        df["clean_conflict_margin"] = df["R_final"]
    if "boundary_distance_proxy" not in df.columns and "R_final" in df.columns:
        df["boundary_distance_proxy"] = pd.to_numeric(df["R_final"], errors="coerce").abs()
    return df


def assign_policy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "condition" not in df.columns:
        raise RuntimeError("No condition column found; cannot assign policy labels.")
    cond_norm = df["condition"].astype(str).map(lambda x: norm(x))
    df["policy_class"] = cond_norm.map(CONDITION_TO_POLICY).fillna("NO_INTERVENTION")
    df["policy_class_id"] = df["policy_class"].map(POLICY_NAME_TO_ID).astype(int)
    df["is_source_claim_holdout"] = cond_norm.isin(["competition_source_claim", "source_claim"]).astype(int)
    df["is_hallucination_holdout"] = cond_norm.isin(["hallucination_like"]).astype(int)
    df["is_core_closure"] = cond_norm.isin(["closure_exception", "closure_negation"]).astype(int)
    df["is_equal_evidence"] = cond_norm.isin(["competition_equal_evidence", "equal_evidence"]).astype(int)
    df["is_override"] = cond_norm.isin(["closure_override"]).astype(int)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="Project root to search recursively. Default: current folder, or parent if in *_outputs.")
    ap.add_argument("--out", default=None, help="Output directory. Default: <root>/da_asa_2d0c_outputs")
    ap.add_argument("--base-file", default=None, help="Optional explicit baseline/sample CSV containing condition labels.")
    args = ap.parse_args()

    root = infer_root(args.root)
    roots = candidate_roots(root)
    csvs = discover_csvs(roots)

    tables = []
    failed = []
    for p in csvs:
        df = safe_read_csv(p)
        if df is None:
            failed.append(str(p))
            continue
        df = ensure_keys(df)
        scores = file_relevance(p, df)
        if max(scores.values()) == 0 and not {"condition", "graph_id"}.issubset(df.columns):
            continue
        tables.append((p, df, scores))

    if not tables:
        raise RuntimeError(f"No usable tables found under roots: {[str(r) for r in roots]}")

    if args.base_file:
        base_path = Path(args.base_file)
        base_df = safe_read_csv(base_path)
        if base_df is None:
            raise RuntimeError(f"Could not read --base-file: {base_path}")
        base_df = ensure_keys(base_df)
        # Add explicit base file to tables if it was outside discovered roots.
        if not any(Path(p) == base_path for p, _, _ in tables):
            tables.append((base_path, base_df, file_relevance(base_path, base_df)))
    else:
        base_path, base_df = select_base_table(tables)
    base_df = ensure_keys(base_df)
    base_df = assign_policy(base_df)
    base_df["source_file_base"] = str(base_path)

    loaded_files = []
    discovered_by_group = {g: [] for g in EXPECTED_GROUPS}

    # Prefix and merge relevant tables except base.
    merged = base_df.copy()
    for path, df, scores in tables:
        if path == base_path:
            group = max(scores, key=scores.get)
            discovered_by_group[group].append(str(path))
            loaded_files.append({"path": str(path), "rows": int(len(df)), "group": group, "columns": list(df.columns)[:80], "used_as_base": True})
            continue
        group = max(scores, key=scores.get)
        if scores[group] <= 0:
            continue
        discovered_by_group[group].append(str(path))
        pref = group
        other = prefix_nonkeys(df, pref)
        before_shape = merged.shape
        merged2 = merge_table(merged, other, pref)
        used = merged2.shape[1] > before_shape[1]
        merged = merged2
        loaded_files.append({"path": str(path), "rows": int(len(df)), "group": group, "columns": list(df.columns)[:80], "merged": bool(used)})

    merged = derive_canonical_features(merged)
    merged = assign_policy(merged)

    # Reorder key/policy columns first.
    first_cols = [c for c in [
        "sample_id", "graph_id", "prompt_id", "condition", "policy_class", "policy_class_id",
        "is_source_claim_holdout", "is_hallucination_holdout", "is_core_closure", "is_equal_evidence", "is_override"
    ] if c in merged.columns]
    rest = [c for c in merged.columns if c not in first_cols]
    merged = merged[first_cols + rest]

    expected_all = [f for feats in EXPECTED_GROUPS.values() for f in feats]
    present_by_group = {g: [f for f in feats if f in merged.columns and merged[f].notna().any()] for g, feats in EXPECTED_GROUPS.items()}
    missing_by_group = {g: [f for f in feats if f not in present_by_group[g]] for g, feats in EXPECTED_GROUPS.items()}

    numeric_cols = []
    for c in merged.columns:
        if c in ["policy_class_id"]:
            continue
        if c in first_cols and c != "policy_class_id":
            continue
        if pd.api.types.is_numeric_dtype(merged[c]):
            numeric_cols.append(c)
        else:
            # convert if possible and mostly numeric
            conv = pd.to_numeric(merged[c], errors="coerce")
            if conv.notna().sum() >= max(3, int(0.5 * len(conv))):
                merged[c] = conv
                numeric_cols.append(c)

    numeric_table = merged[[c for c in ["sample_id", "graph_id", "prompt_id", "condition", "policy_class", "policy_class_id"] if c in merged.columns] + numeric_cols]

    out_dir = Path(args.out).resolve() if args.out else root / "da_asa_2d0b_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    table_path = out_dir / "da_asa2d_policy_feature_table.csv"
    numeric_path = out_dir / "da_asa2d_policy_feature_numeric.csv"
    schema_path = out_dir / "da_asa2d_feature_schema.json"
    missing_path = out_dir / "da_asa2d_missing_report.json"
    summary_path = out_dir / "da_asa2d_summary.json"

    merged.to_csv(table_path, index=False, encoding="utf-8-sig")
    numeric_table.to_csv(numeric_path, index=False, encoding="utf-8-sig")

    schema = {
        "experiment": "DA-ASA-2D.0b Recursive Feature Assembly",
        "root_dir": str(root),
        "searched_roots": [str(r) for r in roots],
        "base_file": str(base_path),
        "key_columns": KEY_COLS,
        "policy_name_to_id": POLICY_NAME_TO_ID,
        "condition_to_policy": CONDITION_TO_POLICY,
        "feature_groups_expected": EXPECTED_GROUPS,
        "present_by_group": present_by_group,
        "feature_columns_selected": numeric_cols,
        "source_report": {
            "loaded_files": loaded_files,
            "failed_or_empty_count": len(failed),
            "failed_or_empty_sample": failed[:50],
            "discovered": discovered_by_group,
        },
    }
    missing_report = {
        "missing_expected_features_by_group": missing_by_group,
        "present_by_group": present_by_group,
        "searched_roots": [str(r) for r in roots],
        "base_file": str(base_path),
        "discovered": discovered_by_group,
        "loaded_files_count": len(loaded_files),
    }
    summary = {
        "rows": int(len(merged)),
        "columns": int(merged.shape[1]),
        "numeric_feature_columns": int(len(numeric_cols)),
        "policy_counts": merged["policy_class"].value_counts(dropna=False).to_dict() if "policy_class" in merged.columns else {},
        "condition_counts": merged["condition"].value_counts(dropna=False).to_dict() if "condition" in merged.columns else {},
        "present_counts_by_group": {g: len(v) for g, v in present_by_group.items()},
        "missing_counts_by_group": {g: len(v) for g, v in missing_by_group.items()},
        "outputs": {
            "feature_table": str(table_path),
            "numeric_table": str(numeric_path),
            "schema": str(schema_path),
            "missing_report": str(missing_path),
            "summary": str(summary_path),
        },
    }

    for p, obj in [(schema_path, schema), (missing_path, missing_report), (summary_path, summary)]:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
