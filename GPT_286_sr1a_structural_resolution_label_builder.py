# -*- coding: utf-8 -*-
"""
SR-1A: StructuralResolution Label Builder

Purpose
-------
Convert SEM-6L.7 / SEM-6L.8 replay outputs into a clean supervised dataset
for Structural Resolution estimation.

Core output:
    sr1a_outputs/sr1a_structural_resolution_labels.csv
    sr1a_outputs/sr1a_policy_targets.csv
    sr1a_outputs/sr1a_summary.json

Theory
------
StructuralResolution is treated as the ability to decide whether a reflexive
operator should be:
    direct_reuse / observe / reject

Default policy:
    boundary_strength >= 4 -> direct_reuse
    boundary_strength == 3 -> observe
    boundary_strength <= 2 -> reject
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(r"C:\Users\ZH\Desktop\AGI\python_script")

IN_6L7 = BASE_DIR / "sem6l7_outputs" / "sem6l7_pivot_results.csv"
IN_6L8 = BASE_DIR / "sem6l8_outputs" / "sem6l8_policy_results.csv"

OUT_DIR = BASE_DIR / "sr1a_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_LABELS = OUT_DIR / "sr1a_structural_resolution_labels.csv"
OUT_POLICY = OUT_DIR / "sr1a_policy_targets.csv"
OUT_SUMMARY = OUT_DIR / "sr1a_summary.json"


def read_csv_safe(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required input file not found: {path}")
        print(f"[WARN] Optional file not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    print(f"[LOAD] {path}")
    print(f"       shape={df.shape}")
    print(f"       columns={list(df.columns)}")
    return df


def norm_colname(c: str) -> str:
    return str(c).strip().lower().replace(" ", "_").replace("-", "_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [norm_colname(c) for c in out.columns]
    return out


def find_col(df: pd.DataFrame, aliases, required=False, default=None):
    aliases = [norm_colname(a) for a in aliases]
    for a in aliases:
        if a in df.columns:
            return a

    if required:
        raise KeyError(
            "Cannot find required column. "
            f"Tried aliases: {aliases}. Existing: {list(df.columns)}"
        )
    return default


def safe_num(s, default=np.nan):
    try:
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return pd.Series([default] * len(s))


def infer_boundary_strength_from_text(text: str) -> int:
    """
    Conservative heuristic for old 6L.7 reuse-gradient groups.

    5: strict / exact structural isomorphism
    4: local geometric near-isomorphism
    3: dynamic-context similarity / operator-context similarity
    2: surface similarity but structural mismatch
    1: unrelated / random / non-target
    """
    t = str(text).lower()

    if any(k in t for k in [
        "strict", "exact", "same_graph", "same_relation",
        "isomorphic", "iso", "identical", "target"
    ]):
        return 5

    if any(k in t for k in [
        "local_geometry", "local_geom", "near_isomorphic",
        "near_iso", "geometry", "structural", "structure"
    ]):
        return 4

    if any(k in t for k in [
        "dynamic", "context", "operator_context", "same_operator",
        "trajectory", "mechanism"
    ]):
        return 3

    if any(k in t for k in [
        "surface", "paraphrase", "lexical", "template",
        "same_words", "wording"
    ]):
        return 2

    if any(k in t for k in [
        "random", "unrelated", "non_target", "nontarget",
        "irrelevant", "mismatch", "wrong"
    ]):
        return 1

    return 3


def action_from_boundary_strength(bs: float) -> str:
    if pd.isna(bs):
        return "unknown"
    if bs >= 4:
        return "direct_reuse"
    if bs == 3:
        return "observe"
    return "reject"


def classify_region(action: str):
    return {
        "is_direct_region": int(action == "direct_reuse"),
        "is_observe_region": int(action == "observe"),
        "is_reject_region": int(action == "reject"),
    }


def choose_oracle_action(row, correct_col, wrong_col, random_col, no_col):
    """
    Outcome-based oracle action.

    If correct reuse gives clear positive gain and exceeds nulls -> direct_reuse.
    If correct reuse is not harmful but not clearly superior -> observe.
    If correct reuse is harmful or worse than wrong/random/no memory -> reject.
    """
    cg = row.get(correct_col, np.nan) if correct_col else np.nan
    wg = row.get(wrong_col, np.nan) if wrong_col else np.nan
    rg = row.get(random_col, np.nan) if random_col else np.nan
    ng = row.get(no_col, 0.0) if no_col else 0.0

    vals = [v for v in [wg, rg, ng] if not pd.isna(v)]
    best_null = max(vals) if vals else 0.0

    if pd.isna(cg):
        return "unknown"

    if cg > best_null and cg > 0:
        return "direct_reuse"

    if cg >= -0.02:
        return "observe"

    return "reject"


def build_sr1a():
    df7_raw = read_csv_safe(IN_6L7, required=True)
    df8_raw = read_csv_safe(IN_6L8, required=False)

    df7 = normalize_columns(df7_raw)
    df8 = normalize_columns(df8_raw) if len(df8_raw) else pd.DataFrame()

    task_col = find_col(df7, [
        "task_id", "row_id", "sample_id", "id", "case_id"
    ], required=False)

    group_col = find_col(df7, [
        "reuse_group", "boundary_group", "similarity_group",
        "group", "condition", "reuse_condition", "boundary_condition"
    ], required=False)

    family_col = find_col(df7, [
        "task_family", "family", "seed_family", "condition_family",
        "target_family"
    ], required=False)

    op_col = find_col(df7, [
        "target_operator", "operator", "operator_id",
        "canonical_operator", "memory_operator"
    ], required=False)

    request_col = find_col(df7, [
        "request", "prompt", "query", "text"
    ], required=False)

    concept_col = find_col(df7, [
        "concept", "concept_id", "concept_phrase", "topic"
    ], required=False)

    bs_col = find_col(df7, [
        "boundary_strength", "structural_resolution",
        "sr", "reuse_strength", "constraint_strength"
    ], required=False)

    correct_gain_col = find_col(df7, [
        "correct_memory_gain", "target_memory_gain", "constraint_insertion_gain",
        "operator_gain", "reuse_gain", "direct_reuse_gain",
        "correct_gain", "gain_correct"
    ], required=False)

    wrong_gain_col = find_col(df7, [
        "wrong_memory_gain", "wrong_operator_gain", "negative_transfer_gain",
        "wrong_gain", "gain_wrong"
    ], required=False)

    random_gain_col = find_col(df7, [
        "random_memory_gain", "random_gain", "shuffle_gain",
        "shuffled_gain", "gain_random"
    ], required=False)

    no_gain_col = find_col(df7, [
        "no_memory_gain", "no_reuse_gain", "baseline_gain",
        "none_gain", "gain_none"
    ], required=False)

    out = pd.DataFrame()
    out["src_row"] = np.arange(len(df7))

    if task_col:
        out["task_id"] = df7[task_col].astype(str)
    else:
        out["task_id"] = ["task_%05d" % i for i in range(len(df7))]

    out["reuse_group"] = df7[group_col].astype(str) if group_col else "unknown"
    out["task_family"] = df7[family_col].astype(str) if family_col else "unknown"
    out["target_operator"] = df7[op_col].astype(str) if op_col else "unknown"
    out["request"] = df7[request_col].astype(str) if request_col else ""
    out["concept"] = df7[concept_col].astype(str) if concept_col else ""

    if bs_col:
        out["boundary_strength"] = safe_num(df7[bs_col])
    else:
        infer_text = (
            out["reuse_group"].astype(str)
            + " "
            + out["task_family"].astype(str)
            + " "
            + out["target_operator"].astype(str)
        )
        out["boundary_strength"] = infer_text.apply(infer_boundary_strength_from_text)

    out["correct_memory_gain"] = safe_num(df7[correct_gain_col]) if correct_gain_col else np.nan
    out["wrong_memory_gain"] = safe_num(df7[wrong_gain_col]) if wrong_gain_col else np.nan
    out["random_memory_gain"] = safe_num(df7[random_gain_col]) if random_gain_col else np.nan
    out["no_memory_gain"] = safe_num(df7[no_gain_col]) if no_gain_col else 0.0

    out["sr_policy_v1_action"] = out["boundary_strength"].apply(action_from_boundary_strength)

    regions = out["sr_policy_v1_action"].apply(classify_region).apply(pd.Series)
    out = pd.concat([out, regions], axis=1)

    if correct_gain_col:
        oracle_actions = []
        for _, row in out.iterrows():
            oracle_actions.append(
                choose_oracle_action(
                    row,
                    "correct_memory_gain",
                    "wrong_memory_gain",
                    "random_memory_gain",
                    "no_memory_gain",
                )
            )
        out["oracle_reuse_action"] = oracle_actions
    else:
        out["oracle_reuse_action"] = out["sr_policy_v1_action"]

    out["correct_minus_wrong_gain"] = out["correct_memory_gain"] - out["wrong_memory_gain"]
    out["correct_minus_random_gain"] = out["correct_memory_gain"] - out["random_memory_gain"]
    out["correct_minus_no_memory_gain"] = out["correct_memory_gain"] - out["no_memory_gain"]

    out["has_positive_correct_gain"] = (out["correct_memory_gain"] > 0).astype(int)
    out["has_negative_transfer_risk"] = (
        (out["wrong_memory_gain"].fillna(0) > out["correct_memory_gain"].fillna(-1e9))
        | (out["random_memory_gain"].fillna(0) > out["correct_memory_gain"].fillna(-1e9))
    ).astype(int)

    policy = out[[
        "task_id",
        "reuse_group",
        "task_family",
        "target_operator",
        "boundary_strength",
        "sr_policy_v1_action",
        "oracle_reuse_action",
        "correct_memory_gain",
        "wrong_memory_gain",
        "random_memory_gain",
        "no_memory_gain",
        "correct_minus_wrong_gain",
        "correct_minus_random_gain",
        "correct_minus_no_memory_gain",
    ]].copy()

    if len(df8):
        df8_aux = df8.copy()
        df8_aux.to_csv(
            OUT_DIR / "sr1a_aux_sem6l8_policy_results_normalized.csv",
            index=False,
            encoding="utf-8-sig",
        )

        df8_task_col = find_col(df8_aux, ["task_id", "row_id", "sample_id", "case_id"], required=False)
        if df8_task_col:
            df8_aux["_merge_task_id"] = df8_aux[df8_task_col].astype(str)
            policy["_merge_task_id"] = policy["task_id"].astype(str)

            keep = ["_merge_task_id"]
            for c in df8_aux.columns:
                if any(k in c for k in ["policy", "action", "gain", "score", "verdict", "reuse"]):
                    keep.append(c)
            keep = list(dict.fromkeys(keep))

            policy = policy.merge(
                df8_aux[keep],
                on="_merge_task_id",
                how="left",
                suffixes=("", "_6l8"),
            ).drop(columns=["_merge_task_id"])

    summary = {}

    summary["input_files"] = {
        "sem6l7": str(IN_6L7),
        "sem6l8": str(IN_6L8) if IN_6L8.exists() else None,
    }

    summary["n_rows"] = int(len(out))
    summary["columns_detected"] = {
        "task_col": task_col,
        "group_col": group_col,
        "family_col": family_col,
        "op_col": op_col,
        "request_col": request_col,
        "concept_col": concept_col,
        "boundary_strength_col": bs_col,
        "correct_gain_col": correct_gain_col,
        "wrong_gain_col": wrong_gain_col,
        "random_gain_col": random_gain_col,
        "no_gain_col": no_gain_col,
    }

    summary["boundary_strength_counts"] = (
        out["boundary_strength"].value_counts(dropna=False).sort_index().to_dict()
    )

    summary["sr_policy_v1_action_counts"] = (
        out["sr_policy_v1_action"].value_counts(dropna=False).to_dict()
    )

    summary["oracle_reuse_action_counts"] = (
        out["oracle_reuse_action"].value_counts(dropna=False).to_dict()
    )

    summary["gain_summary"] = {}
    for c in [
        "correct_memory_gain",
        "wrong_memory_gain",
        "random_memory_gain",
        "no_memory_gain",
        "correct_minus_wrong_gain",
        "correct_minus_random_gain",
        "correct_minus_no_memory_gain",
    ]:
        if c in out.columns:
            s = pd.to_numeric(out[c], errors="coerce")
            summary["gain_summary"][c] = {
                "mean": None if s.dropna().empty else float(s.mean()),
                "median": None if s.dropna().empty else float(s.median()),
                "min": None if s.dropna().empty else float(s.min()),
                "max": None if s.dropna().empty else float(s.max()),
                "non_null": int(s.notna().sum()),
            }

    action_counts = summary["sr_policy_v1_action_counts"]
    has_direct = action_counts.get("direct_reuse", 0) > 0
    has_observe = action_counts.get("observe", 0) > 0
    has_reject = action_counts.get("reject", 0) > 0

    summary["pass_checks"] = {
        "has_direct_region": bool(has_direct),
        "has_observe_region": bool(has_observe),
        "has_reject_region": bool(has_reject),
        "label_dataset_nonempty": bool(len(out) > 0),
        "has_gain_columns": bool(correct_gain_col is not None),
    }

    if has_direct and has_observe and has_reject and len(out) > 0:
        verdict = "PASS_SR1A_LABELS_READY"
    elif has_direct and has_reject and len(out) > 0:
        verdict = "PASS_LITE_SR1A_LABELS_READY_NO_OBSERVE_REGION"
    else:
        verdict = "NEEDS_AUDIT_LABELS_INCOMPLETE"

    summary["verdict"] = verdict

    out.to_csv(OUT_LABELS, index=False, encoding="utf-8-sig")
    policy.to_csv(OUT_POLICY, index=False, encoding="utf-8-sig")

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[SAVED]")
    print(f"  labels : {OUT_LABELS}")
    print(f"  policy : {OUT_POLICY}")
    print(f"  summary: {OUT_SUMMARY}")

    print("\n[SUMMARY]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build_sr1a()
