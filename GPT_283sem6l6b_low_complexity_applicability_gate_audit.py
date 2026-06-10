# -*- coding: utf-8 -*-
r"""
SEM-6L.6B: Low-Complexity ApplicabilityGate Audit

Purpose
-------
SEM-6L.6A showed:

    ConstraintInsertionOperator works on target-like tasks,
    but unconditional OperatorMemory write causes severe non-target damage.

Therefore the true memory unit is not:

    Operator

but:

    ConditionalOperatorMemory = Operator + LowComplexityApplicabilityGate + BoundaryPolicy

This script audits whether a small/simple gate can preserve target gains and remove
non-target damage.

Inputs
------
Default:
    sem6l6a_outputs\sem6l6a_true_live_results.csv
    sem6l6a_outputs\sem6l6a_heldout_tasks.csv

Outputs
-------
sem6l6b_outputs\
    sem6l6b_gate_policy_results.csv
    sem6l6b_gate_summary.csv
    sem6l6b_gate_vs_controls.csv
    sem6l6b_best_gate_rows.csv
    sem6l6b_verdict.json

Run
---
conda activate dhrf_4080s
cd C:\Users\ZH\Desktop\AGI\python_script

python GPT_sem6l6b_low_complexity_applicability_gate_audit.py

Interpretation
--------------
PASS-Lite:
    At least one low-complexity gate has positive target gain and non-target damage near zero.

PASS-Strong:
    Best gate beats no_memory / wrong_memory / random_memory, target win_rate >= 0.60,
    non-target damage is removed, and gate complexity <= 3.

This is a pure evaluation script. It reuses 6L.6A model-forward rows and simulates
conditional memory retrieval:

    if gate(task):
        use correct_operator_memory
    else:
        use no_memory
"""

import argparse
import json
import math
import re
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd


DEFAULT_ROOT = r"C:\Users\ZH\Desktop\AGI\python_script"
DEFAULT_L6A_RESULTS = rf"{DEFAULT_ROOT}\sem6l6a_outputs\sem6l6a_true_live_results.csv"
DEFAULT_L6A_TASKS = rf"{DEFAULT_ROOT}\sem6l6a_outputs\sem6l6a_heldout_tasks.csv"
DEFAULT_OUT_DIR = rf"{DEFAULT_ROOT}\sem6l6b_outputs"


TARGET_FAMILIES = {
    "risk_audit",
    "boundary_check",
    "closure_check",
    "exception_override",
    "policy_selection_boundary",
}

NON_TARGET_FAMILIES = {
    "non_target_definition",
    "non_target_comparison",
    "non_target_property",
    "non_target_planning",
}


def json_safe(obj):
    import numpy as _np
    import pandas as _pd
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (_np.bool_,)):
        return bool(obj)
    if isinstance(obj, (_np.ndarray,)):
        return [json_safe(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(json_safe(k)): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(x) for x in obj]
    try:
        if _pd.isna(obj):
            return None
    except Exception:
        pass
    return str(obj)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def read_results(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing 6L.6A results: {p}")
    df = pd.read_csv(p)
    required = ["replay_root_id", "memory_condition", "live_margin_gain", "task_family", "target_operator", "request"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"6L.6A results missing columns: {missing}")
    df["live_margin_gain"] = pd.to_numeric(df["live_margin_gain"], errors="coerce")
    if "is_target_class" not in df.columns:
        df["is_target_class"] = df["task_family"].isin(TARGET_FAMILIES).astype(int)
    else:
        df["is_target_class"] = pd.to_numeric(df["is_target_class"], errors="coerce").fillna(0).astype(int)
    return df


def pivot_conditions(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "replay_root_id", "task_id", "concept", "task_family", "target_operator",
        "request", "is_target_class"
    ]
    base_cols = [c for c in base_cols if c in df.columns]

    piv = df.pivot_table(
        index="replay_root_id",
        columns="memory_condition",
        values="live_margin_gain",
        aggfunc="mean"
    ).reset_index()

    meta = df.sort_values("memory_condition").drop_duplicates("replay_root_id")[base_cols]
    out = meta.merge(piv, on="replay_root_id", how="left")

    for c in ["no_memory", "correct_operator_memory", "wrong_operator_memory", "random_operator_memory"]:
        if c not in out.columns:
            out[c] = np.nan
    return out


def gate_family_set(rows: pd.DataFrame, include_families: set) -> pd.Series:
    return rows["task_family"].astype(str).isin(include_families)


def gate_lexical(rows: pd.DataFrame, pos_terms, neg_terms=()) -> pd.Series:
    text = (
        rows.get("request", "").astype(str) + " " +
        rows.get("task_family", "").astype(str) + " " +
        rows.get("target_operator", "").astype(str)
    ).str.lower()

    pos = pd.Series(False, index=rows.index)
    for t in pos_terms:
        pos = pos | text.str.contains(re.escape(t.lower()), regex=True)

    neg = pd.Series(False, index=rows.index)
    for t in neg_terms:
        neg = neg | text.str.contains(re.escape(t.lower()), regex=True)

    return pos & (~neg)


def make_gate_specs(rows: pd.DataFrame, max_complexity: int = 3):
    specs = []

    # Hand-designed minimal gates from theory.
    specs.append({
        "gate_id": "G1_closure_exception_only",
        "complexity": 1,
        "description": "task_family in {closure_check, exception_override}",
        "mask": gate_family_set(rows, {"closure_check", "exception_override"}),
    })
    specs.append({
        "gate_id": "G2_closure_exception_risk",
        "complexity": 2,
        "description": "task_family in {closure_check, exception_override, risk_audit}",
        "mask": gate_family_set(rows, {"closure_check", "exception_override", "risk_audit"}),
    })
    specs.append({
        "gate_id": "G3_target_without_plain_boundary",
        "complexity": 3,
        "description": "target-like except boundary_check",
        "mask": gate_family_set(rows, TARGET_FAMILIES - {"boundary_check"}),
    })
    specs.append({
        "gate_id": "G4_all_target_like",
        "complexity": 3,
        "description": "all target-like classes",
        "mask": gate_family_set(rows, TARGET_FAMILIES),
    })
    specs.append({
        "gate_id": "G5_risk_or_closure_lexical_exclude_plain",
        "complexity": 3,
        "description": "lexical risk/closure/exception/override/uncertainty and not plain definition/property/planning/comparison",
        "mask": gate_lexical(
            rows,
            pos_terms=[
                "risk", "failure", "boundary", "hidden assumption", "uncertainty",
                "closure", "consistent", "exception", "override", "caveat", "safety"
            ],
            neg_terms=["define", "definition", "property", "properties", "compare", "comparison", "plan a staged"]
        ),
    })

    # Exhaustive low-complexity family gates. This is still interpretable: OR over <= max_complexity families.
    families = sorted(rows["task_family"].dropna().astype(str).unique())
    for k in range(1, min(max_complexity, len(families)) + 1):
        for fams in combinations(families, k):
            fams = set(fams)
            specs.append({
                "gate_id": "FAM_OR_" + "__".join(sorted(fams)),
                "complexity": k,
                "description": " OR ".join(sorted(fams)),
                "mask": gate_family_set(rows, fams),
            })

    # Deduplicate by gate_id
    seen = set()
    unique = []
    for s in specs:
        if s["gate_id"] not in seen:
            seen.add(s["gate_id"])
            unique.append(s)
    return unique


def evaluate_gate(rows: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, dict]:
    mask = spec["mask"].fillna(False).astype(bool).to_numpy()

    out = rows.copy()
    out["gate_id"] = spec["gate_id"]
    out["gate_complexity"] = spec["complexity"]
    out["gate_description"] = spec["description"]
    out["gate_call"] = mask.astype(int)

    # Conditional memory policy:
    # call -> correct_operator_memory
    # no call -> no_memory
    out["policy_margin_gain"] = np.where(mask, out["correct_operator_memory"], out["no_memory"])

    # Control policies with same gate:
    out["wrong_policy_margin_gain"] = np.where(mask, out["wrong_operator_memory"], out["no_memory"])
    out["random_policy_margin_gain"] = np.where(mask, out["random_operator_memory"], out["no_memory"])
    out["no_memory_margin_gain"] = out["no_memory"]

    # Direct effect vs baseline no memory
    out["policy_vs_no"] = out["policy_margin_gain"] - out["no_memory_margin_gain"]
    out["policy_vs_wrong"] = out["policy_margin_gain"] - out["wrong_policy_margin_gain"]
    out["policy_vs_random"] = out["policy_margin_gain"] - out["random_policy_margin_gain"]

    def stats(vals):
        vals = pd.to_numeric(vals, errors="coerce").dropna().to_numpy(float)
        return {
            "n": int(len(vals)),
            "mean": float(np.mean(vals)) if len(vals) else np.nan,
            "median": float(np.median(vals)) if len(vals) else np.nan,
            "positive_rate": float(np.mean(vals > 0)) if len(vals) else np.nan,
        }

    target = out[out["is_target_class"] == 1]
    nontarget = out[out["is_target_class"] == 0]

    s_all = stats(out["policy_margin_gain"])
    s_target = stats(target["policy_margin_gain"])
    s_nt = stats(nontarget["policy_margin_gain"])

    def comp(diff_col):
        vals = pd.to_numeric(out[diff_col], errors="coerce").dropna().to_numpy(float)
        if len(vals) > 1 and np.std(vals, ddof=1) > 1e-12:
            z = float(np.mean(vals) / (np.std(vals, ddof=1) / math.sqrt(len(vals))))
        else:
            z = np.nan
        return float(np.mean(vals)) if len(vals) else np.nan, float(np.mean(vals > 0)) if len(vals) else np.nan, z

    mean_no, win_no, z_no = comp("policy_vs_no")
    mean_wrong, win_wrong, z_wrong = comp("policy_vs_wrong")
    mean_random, win_random, z_random = comp("policy_vs_random")

    target_mean_no = float(target["policy_vs_no"].mean()) if len(target) else np.nan
    target_win_no = float((target["policy_vs_no"] > 0).mean()) if len(target) else np.nan
    nt_mean_no = float(nontarget["policy_vs_no"].mean()) if len(nontarget) else np.nan
    nt_damage = bool(pd.notna(nt_mean_no) and nt_mean_no < -0.25)

    gate_call_rate = float(np.mean(mask)) if len(mask) else np.nan
    target_call_rate = float(target["gate_call"].mean()) if len(target) else np.nan
    nontarget_call_rate = float(nontarget["gate_call"].mean()) if len(nontarget) else np.nan

    summary = {
        "gate_id": spec["gate_id"],
        "gate_complexity": int(spec["complexity"]),
        "gate_description": spec["description"],
        "n_tasks": int(len(out)),
        "gate_call_rate": gate_call_rate,
        "target_call_rate": target_call_rate,
        "nontarget_call_rate": nontarget_call_rate,
        "overall_mean_gain": s_all["mean"],
        "overall_positive_rate": s_all["positive_rate"],
        "target_mean_gain": s_target["mean"],
        "target_positive_rate": s_target["positive_rate"],
        "nontarget_mean_gain": s_nt["mean"],
        "nontarget_positive_rate": s_nt["positive_rate"],
        "mean_diff_vs_no": mean_no,
        "win_rate_vs_no": win_no,
        "z_vs_no": z_no,
        "mean_diff_vs_wrong": mean_wrong,
        "win_rate_vs_wrong": win_wrong,
        "z_vs_wrong": z_wrong,
        "mean_diff_vs_random": mean_random,
        "win_rate_vs_random": win_random,
        "z_vs_random": z_random,
        "target_mean_diff_vs_no": target_mean_no,
        "target_win_rate_vs_no": target_win_no,
        "nontarget_mean_diff_vs_no": nt_mean_no,
        "nontarget_damage": nt_damage,
    }

    # Simple policy score: reward target gain, control wins, penalize non-target damage + complexity.
    score = 0.0
    score += max(0.0, target_mean_no if pd.notna(target_mean_no) else 0.0)
    score += 0.5 * max(0.0, mean_wrong if pd.notna(mean_wrong) else 0.0)
    score += 0.5 * max(0.0, mean_random if pd.notna(mean_random) else 0.0)
    score -= max(0.0, -nt_mean_no if pd.notna(nt_mean_no) else 0.0) * 2.0
    score -= 0.05 * spec["complexity"]
    summary["gate_score"] = float(score)

    # Pass criteria for a low-complexity gate.
    lite = (
        spec["complexity"] <= 3
        and pd.notna(target_mean_no) and target_mean_no > 0
        and pd.notna(target_win_no) and target_win_no >= 0.55
        and pd.notna(nt_mean_no) and nt_mean_no >= -0.05
        and pd.notna(mean_wrong) and mean_wrong > 0
        and pd.notna(mean_random) and mean_random > 0
    )
    strong = (
        lite
        and pd.notna(win_no) and win_no >= 0.60
        and pd.notna(win_wrong) and win_wrong >= 0.60
        and pd.notna(win_random) and win_random >= 0.60
        and pd.notna(z_no) and z_no > 2
        and pd.notna(z_wrong) and z_wrong > 2
        and pd.notna(z_random) and z_random > 2
    )
    summary["pass_lite"] = bool(lite)
    summary["pass_strong"] = bool(strong)

    return out, summary


def make_control_comparisons(best_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in [
        ("policy_vs_no", "gated_correct>no_memory"),
        ("policy_vs_wrong", "gated_correct>gated_wrong"),
        ("policy_vs_random", "gated_correct>gated_random"),
    ]:
        vals = pd.to_numeric(best_rows[col], errors="coerce").dropna().to_numpy(float)
        if len(vals) > 1 and np.std(vals, ddof=1) > 1e-12:
            z = float(np.mean(vals) / (np.std(vals, ddof=1) / math.sqrt(len(vals))))
        else:
            z = np.nan
        rows.append({
            "comparison": label,
            "n": int(len(vals)),
            "mean_diff": float(np.mean(vals)) if len(vals) else np.nan,
            "median_diff": float(np.median(vals)) if len(vals) else np.nan,
            "win_rate": float(np.mean(vals > 0)) if len(vals) else np.nan,
            "z": z,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l6a_results", default=DEFAULT_L6A_RESULTS)
    ap.add_argument("--l6a_tasks", default=DEFAULT_L6A_TASKS)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--max_complexity", type=int, default=3)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_results(args.l6a_results)
    rows = pivot_conditions(df)

    gate_specs = make_gate_specs(rows, max_complexity=args.max_complexity)
    all_policy_rows = []
    summaries = []
    for spec in gate_specs:
        gres, summ = evaluate_gate(rows, spec)
        all_policy_rows.append(gres)
        summaries.append(summ)

    policy_results = pd.concat(all_policy_rows, ignore_index=True)
    summary = pd.DataFrame(summaries)

    # Sort: prefer pass, then score, then low complexity.
    summary = summary.sort_values(
        ["pass_strong", "pass_lite", "gate_score", "gate_complexity"],
        ascending=[False, False, False, True]
    )

    best = summary.iloc[0].to_dict()
    best_rows = policy_results[policy_results["gate_id"] == best["gate_id"]].copy()
    comp = make_control_comparisons(best_rows)

    policy_results.to_csv(out_dir / "sem6l6b_gate_policy_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "sem6l6b_gate_summary.csv", index=False, encoding="utf-8-sig")
    best_rows.to_csv(out_dir / "sem6l6b_best_gate_rows.csv", index=False, encoding="utf-8-sig")
    comp.to_csv(out_dir / "sem6l6b_gate_vs_controls.csv", index=False, encoding="utf-8-sig")

    if bool(best.get("pass_strong", False)):
        verdict_str = "PASS_STRONG_LOW_COMPLEXITY_GATE"
    elif bool(best.get("pass_lite", False)):
        verdict_str = "PASS_LITE_LOW_COMPLEXITY_GATE"
    else:
        verdict_str = "FAIL_OR_INCONCLUSIVE"

    verdict = {
        "stage": "SEM-6L.6B",
        "mode": "low_complexity_applicability_gate_audit",
        "verdict": verdict_str,
        "best_gate": best,
        "n_gates_tested": int(len(summary)),
        "notes": [
            "6L.6B tests whether a simple ApplicabilityGate can convert ConstraintInsertionOperator into ConditionalOperatorMemory.",
            "A pass means the memory write unit is not Operator alone, but Operator + low-complexity gate + boundary policy.",
            "This audit reuses 6L.6A live model-forward rows; no new model calls are required.",
        ],
        "outputs": {
            "gate_policy_results": str(out_dir / "sem6l6b_gate_policy_results.csv"),
            "gate_summary": str(out_dir / "sem6l6b_gate_summary.csv"),
            "gate_vs_controls": str(out_dir / "sem6l6b_gate_vs_controls.csv"),
            "best_gate_rows": str(out_dir / "sem6l6b_best_gate_rows.csv"),
            "verdict": str(out_dir / "sem6l6b_verdict.json"),
        },
    }

    with open(out_dir / "sem6l6b_verdict.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(verdict), f, ensure_ascii=False, indent=2)

    print("\n========== SEM-6L.6B ==========")
    print(json.dumps(json_safe(verdict), ensure_ascii=False, indent=2))
    print("\nTop gates:")
    print(summary.head(20).to_string(index=False))
    print("\nBest gate comparisons:")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
