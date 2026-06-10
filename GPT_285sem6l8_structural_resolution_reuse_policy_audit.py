# -*- coding: utf-8 -*-
r"""
SEM-6L.8: StructuralResolution-Guided ReusePolicy Audit

Purpose
-------
SEM-6L.7 showed PASS-Strong Reuse Boundary Gradient:

    reuse gain declines as boundary constraints weaken.

This motivates a new object:

    StructuralResolution = ability to identify boundary constraints precisely
    ReusePolicy = choose direct reuse / observe / reject based on structural resolution

SEM-6L.8 tests whether a StructuralResolution-guided policy beats:
    - always reuse
    - source-only strict reuse
    - surface-similarity reuse
    - random reuse
    - no reuse

Input
-----
Default:
    sem6l7_outputs\sem6l7_pivot_results.csv
    sem6l7_outputs\sem6l7_gradient_summary.csv

The script reuses true model-forward rows from 6L.7; it does not call the model.

Reuse actions
-------------
    direct_reuse:
        use correct_memory margin gain
    observe:
        default no_memory, but count as candidate/observation; no immediate operator call
    reject:
        use no_memory
    random_reuse:
        use random_memory

Policies
--------
1. SR_policy_v1:
    strength >= 4 -> direct_reuse
    strength == 3 -> observe
    strength <= 2 -> reject

2. SR_policy_v2:
    strength >= 3 -> direct_reuse
    strength == 2 -> observe
    strength <= 1 -> reject

3. strict_only:
    strength == 4 -> direct_reuse
    else reject

4. always_reuse:
    all direct_reuse

5. surface_reuse:
    G0/G1/G3 direct reuse, G2 observe, G4 reject
    (surface-similar but structurally different is intentionally included)

6. random_policy:
    random direct/observe/reject with fixed seed

Outputs
-------
sem6l8_outputs/
    sem6l8_policy_results.csv
    sem6l8_policy_summary.csv
    sem6l8_policy_vs_controls.csv
    sem6l8_best_policy_rows.csv
    sem6l8_verdict.json

Run
---
conda activate dhrf_4080s
cd C:\Users\ZH\Desktop\AGI\python_script

python GPT_sem6l8_structural_resolution_reuse_policy_audit.py

Interpretation
--------------
PASS-Lite:
    best SR policy beats always_reuse and no_reuse with non-target damage controlled.

PASS-Strong:
    SR policy is best or tied-best, beats always_reuse / surface_reuse / random,
    and preserves gains in high-strength groups while rejecting low-strength groups.

Theory
------
If PASS:
    StructuralResolution is supported as the state variable for reflexive
    generalization strategy selection.
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = r"C:\Users\ZH\Desktop\AGI\python_script"
DEFAULT_PIVOT = rf"{DEFAULT_ROOT}\sem6l7_outputs\sem6l7_pivot_results.csv"
DEFAULT_GRADIENT = rf"{DEFAULT_ROOT}\sem6l7_outputs\sem6l7_gradient_summary.csv"
DEFAULT_OUT_DIR = rf"{DEFAULT_ROOT}\sem6l8_outputs"

SEED = 20260606


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


def read_pivot(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing SEM-6L.7 pivot file: {p}")
    df = pd.read_csv(p)

    required = [
        "replay_root_id", "reuse_group", "boundary_strength",
        "correct_memory", "no_memory", "wrong_memory", "random_memory"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Pivot missing columns: {missing}")

    for c in ["boundary_strength", "correct_memory", "no_memory", "wrong_memory", "random_memory"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # In previous scripts no_memory gain is usually 0. Keep explicit.
    df["no_memory"] = df["no_memory"].fillna(0.0)
    return df


def action_for_policy(row: pd.Series, policy_id: str) -> str:
    s = float(row["boundary_strength"])
    g = str(row["reuse_group"])

    if policy_id == "no_reuse":
        return "reject"

    if policy_id == "always_reuse":
        return "direct_reuse"

    if policy_id == "strict_only":
        return "direct_reuse" if s >= 4 else "reject"

    if policy_id == "sr_policy_v1":
        if s >= 4:
            return "direct_reuse"
        if s == 3:
            return "observe"
        return "reject"

    if policy_id == "sr_policy_v2":
        if s >= 3:
            return "direct_reuse"
        if s == 2:
            return "observe"
        return "reject"

    if policy_id == "surface_reuse":
        # Deliberately wrong-ish baseline: treats surface-similar G3 as reusable.
        if g in {"G0_STRICT_SOURCE", "G1_LOCAL_GEOMETRY", "G3_SURFACE_SIM_STRUCT_DIFF"}:
            return "direct_reuse"
        if g == "G2_DYNAMIC_CONTEXT":
            return "observe"
        return "reject"

    if policy_id == "dynamic_context_reuse":
        if s >= 2:
            return "direct_reuse"
        return "reject"

    if policy_id == "random_policy":
        # deterministic per replay id for reproducibility
        rng = random.Random(SEED + int(abs(hash(str(row.get("replay_root_id", "")))) % 10_000_000))
        return rng.choice(["direct_reuse", "observe", "reject"])

    raise ValueError(f"Unknown policy_id: {policy_id}")


def policy_value(row: pd.Series, action: str) -> float:
    if action == "direct_reuse":
        return float(row["correct_memory"])
    if action == "observe":
        return float(row["no_memory"])  # no immediate memory call; observation only
    if action == "reject":
        return float(row["no_memory"])
    if action == "random_reuse":
        return float(row["random_memory"])
    return float(row["no_memory"])


def control_value(row: pd.Series, action: str, control: str) -> float:
    if action != "direct_reuse":
        return float(row["no_memory"])
    if control == "wrong":
        return float(row["wrong_memory"])
    if control == "random":
        return float(row["random_memory"])
    if control == "no":
        return float(row["no_memory"])
    return float(row["no_memory"])


def apply_policy(df: pd.DataFrame, policy_id: str) -> pd.DataFrame:
    out = df.copy()
    out["policy_id"] = policy_id
    out["reuse_action"] = out.apply(lambda r: action_for_policy(r, policy_id), axis=1)
    out["policy_gain"] = out.apply(lambda r: policy_value(r, r["reuse_action"]), axis=1)
    out["wrong_control_gain"] = out.apply(lambda r: control_value(r, r["reuse_action"], "wrong"), axis=1)
    out["random_control_gain"] = out.apply(lambda r: control_value(r, r["reuse_action"], "random"), axis=1)
    out["no_control_gain"] = out["no_memory"]

    out["policy_vs_no"] = out["policy_gain"] - out["no_control_gain"]
    out["policy_vs_wrong"] = out["policy_gain"] - out["wrong_control_gain"]
    out["policy_vs_random"] = out["policy_gain"] - out["random_control_gain"]

    out["called_direct"] = (out["reuse_action"] == "direct_reuse").astype(int)
    out["observed_only"] = (out["reuse_action"] == "observe").astype(int)
    out["rejected"] = (out["reuse_action"] == "reject").astype(int)
    return out


def summarize_policy(rows: pd.DataFrame) -> dict:
    def arr(col):
        return pd.to_numeric(rows[col], errors="coerce").dropna().to_numpy(float)

    v = arr("policy_gain")
    d_no = arr("policy_vs_no")
    d_wrong = arr("policy_vs_wrong")
    d_random = arr("policy_vs_random")

    def zscore(vals):
        if len(vals) > 1 and np.std(vals, ddof=1) > 1e-12:
            return float(np.mean(vals) / (np.std(vals, ddof=1) / math.sqrt(len(vals))))
        return np.nan

    # High/mid/low strength behavior
    high = rows[rows["boundary_strength"] >= 3]
    strict = rows[rows["boundary_strength"] >= 4]
    mid = rows[rows["boundary_strength"] == 2]
    low = rows[rows["boundary_strength"] <= 1]

    return {
        "policy_id": rows["policy_id"].iloc[0],
        "n": int(len(rows)),
        "mean_gain": float(np.mean(v)) if len(v) else np.nan,
        "median_gain": float(np.median(v)) if len(v) else np.nan,
        "positive_rate": float(np.mean(v > 0)) if len(v) else np.nan,
        "direct_call_rate": float(rows["called_direct"].mean()) if len(rows) else np.nan,
        "observe_rate": float(rows["observed_only"].mean()) if len(rows) else np.nan,
        "reject_rate": float(rows["rejected"].mean()) if len(rows) else np.nan,
        "mean_diff_vs_no": float(np.mean(d_no)) if len(d_no) else np.nan,
        "win_rate_vs_no": float(np.mean(d_no > 0)) if len(d_no) else np.nan,
        "z_vs_no": zscore(d_no),
        "mean_diff_vs_wrong": float(np.mean(d_wrong)) if len(d_wrong) else np.nan,
        "win_rate_vs_wrong": float(np.mean(d_wrong > 0)) if len(d_wrong) else np.nan,
        "z_vs_wrong": zscore(d_wrong),
        "mean_diff_vs_random": float(np.mean(d_random)) if len(d_random) else np.nan,
        "win_rate_vs_random": float(np.mean(d_random > 0)) if len(d_random) else np.nan,
        "z_vs_random": zscore(d_random),
        "strict_mean_gain": float(strict["policy_gain"].mean()) if len(strict) else np.nan,
        "high_mean_gain": float(high["policy_gain"].mean()) if len(high) else np.nan,
        "mid_mean_gain": float(mid["policy_gain"].mean()) if len(mid) else np.nan,
        "low_mean_gain": float(low["policy_gain"].mean()) if len(low) else np.nan,
        "low_direct_call_rate": float(low["called_direct"].mean()) if len(low) else np.nan,
        "low_damage": bool(len(low) and low["policy_gain"].mean() < -0.25),
    }


def compare_policies(summary: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    policy_ids = summary["policy_id"].tolist()
    sr_candidates = [p for p in policy_ids if p.startswith("sr_policy")]
    control_ids = ["always_reuse", "surface_reuse", "random_policy", "no_reuse", "strict_only", "dynamic_context_reuse"]
    for sr in sr_candidates:
        a = results[results["policy_id"] == sr][["replay_root_id", "policy_gain"]].rename(columns={"policy_gain": "sr_gain"})
        for ctrl in control_ids:
            if ctrl not in policy_ids:
                continue
            b = results[results["policy_id"] == ctrl][["replay_root_id", "policy_gain"]].rename(columns={"policy_gain": "control_gain"})
            m = a.merge(b, on="replay_root_id", how="inner")
            diff = (m["sr_gain"] - m["control_gain"]).dropna().to_numpy(float)
            if len(diff) > 1 and np.std(diff, ddof=1) > 1e-12:
                z = float(np.mean(diff) / (np.std(diff, ddof=1) / math.sqrt(len(diff))))
            else:
                z = np.nan
            rows.append({
                "sr_policy": sr,
                "control_policy": ctrl,
                "n": int(len(diff)),
                "mean_diff": float(np.mean(diff)) if len(diff) else np.nan,
                "median_diff": float(np.median(diff)) if len(diff) else np.nan,
                "win_rate": float(np.mean(diff > 0)) if len(diff) else np.nan,
                "z": z,
            })
    return pd.DataFrame(rows)


def select_best_sr(summary: pd.DataFrame) -> dict:
    sr = summary[summary["policy_id"].str.startswith("sr_policy")].copy()
    if sr.empty:
        raise ValueError("No SR policies found")
    # score rewards mean gain, high gain, rejects low area, control diff
    sr["policy_score"] = (
        sr["mean_gain"].fillna(0)
        + 0.4 * sr["high_mean_gain"].fillna(0)
        + 0.3 * sr["mean_diff_vs_wrong"].fillna(0)
        + 0.3 * sr["mean_diff_vs_random"].fillna(0)
        - 2.0 * sr["low_direct_call_rate"].fillna(0)
        - 1.0 * sr["low_damage"].astype(float)
    )
    sr = sr.sort_values("policy_score", ascending=False)
    return sr.iloc[0].to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pivot", default=DEFAULT_PIVOT)
    ap.add_argument("--gradient", default=DEFAULT_GRADIENT)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    piv = read_pivot(args.pivot)

    policy_ids = [
        "sr_policy_v1",
        "sr_policy_v2",
        "strict_only",
        "always_reuse",
        "surface_reuse",
        "dynamic_context_reuse",
        "random_policy",
        "no_reuse",
    ]

    all_results = []
    summaries = []
    for pid in policy_ids:
        rows = apply_policy(piv, pid)
        all_results.append(rows)
        summaries.append(summarize_policy(rows))

    results = pd.concat(all_results, ignore_index=True)
    summary = pd.DataFrame(summaries)

    # Add policy score / sorting
    best_sr = select_best_sr(summary)
    comparisons = compare_policies(summary, results)

    best_rows = results[results["policy_id"] == best_sr["policy_id"]].copy()

    # Determine verdict.
    comp_sr = comparisons[comparisons["sr_policy"] == best_sr["policy_id"]]
    def comp_ok(ctrl, min_diff=0.0):
        r = comp_sr[comp_sr["control_policy"] == ctrl]
        if r.empty:
            return False
        rr = r.iloc[0]
        return pd.notna(rr["mean_diff"]) and rr["mean_diff"] > min_diff

    best_sr_summary = summary[summary["policy_id"] == best_sr["policy_id"]].iloc[0].to_dict()
    beats_always = comp_ok("always_reuse", 0.0)
    beats_surface = comp_ok("surface_reuse", 0.0)
    beats_random = comp_ok("random_policy", 0.0)
    beats_no = comp_ok("no_reuse", 0.0)

    low_protected = (
        pd.notna(best_sr_summary.get("low_direct_call_rate"))
        and best_sr_summary["low_direct_call_rate"] <= 0.05
        and not bool(best_sr_summary.get("low_damage", True))
    )
    high_positive = (
        pd.notna(best_sr_summary.get("high_mean_gain"))
        and best_sr_summary["high_mean_gain"] > 0
    )

    pass_lite = bool(beats_always and beats_surface and beats_random and high_positive and low_protected)
    pass_strong = bool(pass_lite and beats_no and best_sr_summary["mean_diff_vs_wrong"] > 0 and best_sr_summary["mean_diff_vs_random"] > 0)

    if pass_strong:
        verdict_str = "PASS_STRONG_STRUCTURAL_RESOLUTION_REUSE_POLICY"
    elif pass_lite:
        verdict_str = "PASS_LITE_STRUCTURAL_RESOLUTION_REUSE_POLICY"
    else:
        verdict_str = "FAIL_OR_INCONCLUSIVE"

    # Save
    results.to_csv(out_dir / "sem6l8_policy_results.csv", index=False, encoding="utf-8-sig")
    summary.sort_values("mean_gain", ascending=False).to_csv(out_dir / "sem6l8_policy_summary.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(out_dir / "sem6l8_policy_vs_controls.csv", index=False, encoding="utf-8-sig")
    best_rows.to_csv(out_dir / "sem6l8_best_policy_rows.csv", index=False, encoding="utf-8-sig")

    verdict = {
        "stage": "SEM-6L.8",
        "mode": "structural_resolution_guided_reuse_policy_audit",
        "verdict": verdict_str,
        "best_sr_policy": best_sr["policy_id"],
        "best_sr_summary": best_sr_summary,
        "beats": {
            "always_reuse": beats_always,
            "surface_reuse": beats_surface,
            "random_policy": beats_random,
            "no_reuse": beats_no,
        },
        "low_strength_protected": low_protected,
        "high_strength_positive": high_positive,
        "notes": [
            "6L.8 tests whether structural-resolution-guided policy selection beats fixed reuse baselines.",
            "SR policy should call direct reuse only where boundary constraints are strong enough, observe or reject otherwise.",
            "A pass supports StructuralResolution as a core state variable for reflexive generalization."
        ],
        "outputs": {
            "policy_results": str(out_dir / "sem6l8_policy_results.csv"),
            "policy_summary": str(out_dir / "sem6l8_policy_summary.csv"),
            "policy_vs_controls": str(out_dir / "sem6l8_policy_vs_controls.csv"),
            "best_policy_rows": str(out_dir / "sem6l8_best_policy_rows.csv"),
            "verdict": str(out_dir / "sem6l8_verdict.json"),
        }
    }

    with open(out_dir / "sem6l8_verdict.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(verdict), f, ensure_ascii=False, indent=2)

    print("\n========== SEM-6L.8 ==========")
    print(json.dumps(json_safe(verdict), ensure_ascii=False, indent=2))
    print("\nPolicy summary:")
    print(summary.sort_values("mean_gain", ascending=False).to_string(index=False))
    print("\nPolicy comparisons:")
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
