r"""
DA-ASA-2K.4 Policy Results Table Generation

Purpose
-------
Generate a policy_results table for expanded DA-ASA-2K.1 samples using the
2K.2 baseline C/E scoring table and a conservative table-level policy/action
simulator. This stage does not require model forward intervention. It creates
condition/action-level and sample/action-level outcome tables that can be used
by 2K.5 effect-vector + LPF assignment.

Inputs expected under ROOT:
  da_asa2k2_baseline_outputs/da_asa2k2_baseline_scores.csv
  da_asa2k3_dsta_outputs/da_asa2k3_dsta_sample_features.csv

Outputs:
  da_asa2k4_policy_outputs/da_asa2k4_policy_results.csv
  da_asa2k4_policy_outputs/da_asa2k4_policy_condition_action_summary.csv
  da_asa2k4_policy_outputs/da_asa2k4_policy_family_action_summary.csv
  da_asa2k4_policy_outputs/da_asa2k4_diagnostics.json
  da_asa2k4_policy_outputs/da_asa2k4_next_stage_seed.md

Notes
-----
This is a table-level expanded policy result generator, not true activation
intervention. It is designed to unblock 2K.5 effect-vector / LPF assignment.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\ZH\Desktop\AGI\python_script")
BASELINE_PATH = ROOT / "da_asa2k2_baseline_outputs" / "da_asa2k2_baseline_scores.csv"
DSTA_PATH = ROOT / "da_asa2k3_dsta_outputs" / "da_asa2k3_dsta_sample_features.csv"
OUT_DIR = ROOT / "da_asa2k4_policy_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 24 action classes, kept intentionally broad to match the 2J topology scale.
ACTIONS = [
    "NO_INTERVENTION",
    "CORE_CLOSURE_UPDATE",
    "CORE_CLOSURE_UPDATE_STRONG",
    "CORE_CLOSURE_UPDATE_LIGHT",
    "OVERRIDE_THREE_STAGE",
    "OVERRIDE_THREE_STAGE_STRONG",
    "OVERRIDE_THREE_STAGE_LIGHT",
    "EQUAL_EVIDENCE_ORDER",
    "EQUAL_EVIDENCE_ORDER_STRONG",
    "EQUAL_EVIDENCE_ORDER_LIGHT",
    "ORDER_CORRECTION_ONLY",
    "COMMIT_CORRECTION_ONLY",
    "ATTRACTOR_ESCAPE_ONLY",
    "ATTRACTOR_ESCAPE_PLUS_COMMIT",
    "PROTECT_STABLE",
    "PROTECT_STABLE_STRONG",
    "SOURCE_CLAIM_GUARD",
    "TEMPORAL_UPDATE_GUARD",
    "AUTHORITY_UPDATE_GUARD",
    "EXCEPTION_BINDING_GUARD",
    "NEGATION_REWRITE_GUARD",
    "HALLUCINATION_DAMPING",
    "WEAK_DISTRACTOR_DAMPING",
    "BALANCED_MINIMAL_EDIT",
]

FAMILY_CANONICAL = {
    "stable_protect": "PROTECT_STABLE",
    "competition": "EQUAL_EVIDENCE_ORDER",
    "closure_update": "CORE_CLOSURE_UPDATE",
    "override_exception": "OVERRIDE_THREE_STAGE",
    "hallucination_like": "HALLUCINATION_DAMPING",
}

FAMILY_ALTERNATES = {
    "stable_protect": {"NO_INTERVENTION", "PROTECT_STABLE", "PROTECT_STABLE_STRONG", "WEAK_DISTRACTOR_DAMPING"},
    "competition": {"EQUAL_EVIDENCE_ORDER", "EQUAL_EVIDENCE_ORDER_STRONG", "EQUAL_EVIDENCE_ORDER_LIGHT", "ORDER_CORRECTION_ONLY", "BALANCED_MINIMAL_EDIT"},
    "closure_update": {"CORE_CLOSURE_UPDATE", "CORE_CLOSURE_UPDATE_STRONG", "CORE_CLOSURE_UPDATE_LIGHT", "COMMIT_CORRECTION_ONLY", "TEMPORAL_UPDATE_GUARD", "AUTHORITY_UPDATE_GUARD"},
    "override_exception": {"OVERRIDE_THREE_STAGE", "OVERRIDE_THREE_STAGE_STRONG", "OVERRIDE_THREE_STAGE_LIGHT", "ATTRACTOR_ESCAPE_ONLY", "ATTRACTOR_ESCAPE_PLUS_COMMIT", "EXCEPTION_BINDING_GUARD", "NEGATION_REWRITE_GUARD"},
    "hallucination_like": {"HALLUCINATION_DAMPING", "SOURCE_CLAIM_GUARD", "ATTRACTOR_ESCAPE_ONLY", "ATTRACTOR_ESCAPE_PLUS_COMMIT", "BALANCED_MINIMAL_EDIT"},
}


def read_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    return pd.read_csv(path)


def find_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def standardize_keys(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.copy()
    graph_col = find_col(df, ["graph_id", "graph", "gid"])
    cond_col = find_col(df, ["condition", "cond", "condition_name"])
    fam_col = find_col(df, ["condition_family", "family", "condition_family_inferred"])
    if graph_col is None or cond_col is None:
        raise KeyError(f"[{name}] cannot find graph_id/condition. Columns={list(df.columns)}")
    if graph_col != "graph_id":
        df = df.rename(columns={graph_col: "graph_id"})
    if cond_col != "condition":
        df = df.rename(columns={cond_col: "condition"})
    if fam_col is not None and fam_col != "condition_family":
        df = df.rename(columns={fam_col: "condition_family"})
    df["graph_id"] = df["graph_id"].astype(str).str.strip()
    df["condition"] = df["condition"].astype(str).str.strip()
    if "condition_family" not in df.columns:
        df["condition_family"] = df["condition"].map(infer_family)
    else:
        df["condition_family"] = df["condition_family"].astype(str).str.strip().replace({"nan": np.nan})
        df["condition_family"] = df["condition_family"].fillna(df["condition"].map(infer_family))
    return df


def infer_family(condition: str) -> str:
    s = str(condition).lower()
    if any(x in s for x in ["stable", "protect", "redundant", "clean", "weak_distractor"]):
        return "stable_protect"
    if any(x in s for x in ["competition", "equal", "source_claim", "direct"]):
        return "competition"
    if any(x in s for x in ["closure", "temporal", "authority", "update"]):
        return "closure_update"
    if any(x in s for x in ["override", "exception", "negation", "rewrite"]):
        return "override_exception"
    if any(x in s for x in ["hallucination", "fabrication", "unsupported"]):
        return "hallucination_like"
    return "other"


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def action_profile(action: str) -> Dict[str, float]:
    """Return generic action effects before family matching."""
    a = action.upper()
    profile = {
        "repair": 0.0,
        "escape": 0.0,
        "commit": 0.0,
        "protect": 0.0,
        "order": 0.0,
        "risk": 0.0,
        "strength": 1.0,
    }
    if "CORE" in a or "UPDATE" in a or "CLOSURE" in a:
        profile["repair"] += 0.55
        profile["commit"] += 0.25
    if "OVERRIDE" in a or "ESCAPE" in a:
        profile["escape"] += 0.60
        profile["repair"] += 0.25
        profile["risk"] += 0.10
    if "EQUAL" in a or "ORDER" in a:
        profile["order"] += 0.55
        profile["repair"] += 0.25
    if "COMMIT" in a:
        profile["commit"] += 0.55
    if "PROTECT" in a or "NO_INTERVENTION" in a:
        profile["protect"] += 0.60
    if "HALLUCINATION" in a or "DAMPING" in a or "SOURCE" in a:
        profile["escape"] += 0.25
        profile["protect"] += 0.25
        profile["repair"] += 0.20
    if "GUARD" in a:
        profile["protect"] += 0.15
        profile["repair"] += 0.15
    if "STRONG" in a:
        profile["strength"] = 1.35
        profile["risk"] += 0.12
    if "LIGHT" in a or "MINIMAL" in a:
        profile["strength"] = 0.72
        profile["risk"] -= 0.03
    return profile


def need_vector(row: pd.Series) -> Dict[str, float]:
    fam = str(row.get("condition_family", "other"))
    # Baseline margin > 0 means C over E if columns are conventional.
    margin = float(row.get("baseline_margin", row.get("ce_margin", 0.0)))
    prob_gold = float(row.get("prob_gold", 0.5)) if "prob_gold" in row.index else 0.5
    uncertainty = float(1.0 - abs(prob_gold - 0.5) * 2.0)
    wrong_pressure = float(max(0.0, 0.65 - prob_gold))

    needs = {"repair": 0.0, "escape": 0.0, "commit": 0.0, "protect": 0.0, "order": 0.0}
    if fam == "stable_protect":
        needs.update({"protect": 0.75, "repair": 0.05, "escape": 0.02, "commit": 0.02, "order": 0.05})
    elif fam == "competition":
        needs.update({"order": 0.75, "repair": 0.35, "escape": 0.12, "commit": 0.18, "protect": 0.05})
    elif fam == "closure_update":
        needs.update({"repair": 0.80, "commit": 0.55, "escape": 0.15, "order": 0.10, "protect": 0.02})
    elif fam == "override_exception":
        needs.update({"escape": 0.82, "repair": 0.55, "commit": 0.45, "order": 0.18, "protect": 0.05})
    elif fam == "hallucination_like":
        needs.update({"escape": 0.65, "protect": 0.35, "repair": 0.45, "commit": 0.20, "order": 0.12})
    else:
        needs.update({"repair": 0.25, "escape": 0.25, "commit": 0.20, "protect": 0.20, "order": 0.20})

    # Harder samples need slightly stronger repair/escape.
    needs["repair"] += 0.25 * wrong_pressure + 0.10 * uncertainty
    needs["escape"] += 0.15 * wrong_pressure
    needs["commit"] += 0.10 * wrong_pressure
    return needs


def profile_score(needs: Dict[str, float], prof: Dict[str, float], action: str, family: str) -> Tuple[float, float, float]:
    dimensions = ["repair", "escape", "commit", "protect", "order"]
    # Goodness is alignment between action profile and mechanism needs.
    align = sum(min(needs[d], prof[d] * prof["strength"]) for d in dimensions)
    overshoot = sum(max(0.0, prof[d] * prof["strength"] - needs[d]) for d in dimensions)
    canonical_bonus = 0.18 if action == FAMILY_CANONICAL.get(family, "") else 0.0
    alt_bonus = 0.08 if action in FAMILY_ALTERNATES.get(family, set()) else 0.0
    risk = prof["risk"] + 0.15 * overshoot
    score = align + canonical_bonus + alt_bonus - risk
    return score, align, risk


def generate_policy_results(base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in base.iterrows():
        needs = need_vector(r)
        family = str(r.get("condition_family", "other"))
        base_margin = float(r.get("baseline_margin", 0.0)) if "baseline_margin" in r.index else 0.0
        base_prob_gold = float(r.get("prob_gold", 0.5)) if "prob_gold" in r.index else 0.5
        base_r = float(r.get("baseline_R", r.get("R_baseline", base_margin)))
        for action in ACTIONS:
            prof = action_profile(action)
            score, align, risk = profile_score(needs, prof, action, family)
            if action == "NO_INTERVENTION":
                score = 0.0
                align = 0.0
                risk = 0.0
            # Translate functional score into synthetic table outcome anchored to baseline.
            delta_prob = 0.18 * np.tanh(score)
            delta_r = 1.15 * np.tanh(score)
            pred_clean = float(np.clip(base_prob_gold + delta_prob, 0.0, 1.0))
            r_final = float(base_r + delta_r)
            safety_penalty = float(max(0.0, risk))
            rows.append({
                "graph_id": r["graph_id"],
                "condition": r["condition"],
                "condition_family": family,
                "action_class": action,
                "baseline_margin": base_margin,
                "baseline_prob_gold": base_prob_gold,
                "pred_clean": pred_clean,
                "R_final": r_final,
                "functional_score": float(score),
                "alignment_score": float(align),
                "safety_penalty": safety_penalty,
                "is_no_intervention": int(action == "NO_INTERVENTION"),
                "is_family_canonical": int(action == FAMILY_CANONICAL.get(family, "")),
            })
    out = pd.DataFrame(rows)
    # Per-key action ranks.
    out["rank_within_sample"] = out.groupby(["graph_id", "condition"])["functional_score"].rank(method="first", ascending=False)
    return out


def summarize(policy: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    return (
        policy.groupby(group_cols, as_index=False)
        .agg(
            n_rows=("functional_score", "size"),
            mean_pred_clean=("pred_clean", "mean"),
            mean_R_final=("R_final", "mean"),
            mean_functional_score=("functional_score", "mean"),
            best_functional_score=("functional_score", "max"),
            mean_safety_penalty=("safety_penalty", "mean"),
        )
    )


def write_seed(diag: Dict) -> None:
    text = f"""# DA-ASA-2K.5 Next Stage Seed\n\n## Current status\n\nDA-ASA-2K.4 generated an expanded policy_results table.\n\nVerdict:\n\n```text\n{diag['verdict']}\n```\n\n## Paths\n\nRoot:\n\n```text\n{ROOT}\n```\n\nPolicy results table:\n\n```text\n{diag['outputs']['policy_results']}\n```\n\nCondition-action summary:\n\n```text\n{diag['outputs']['condition_action_summary']}\n```\n\nFamily-action summary:\n\n```text\n{diag['outputs']['family_action_summary']}\n```\n\nDenseDSTA feature table from 2K.3:\n\n```text\n{DSTA_PATH}\n```\n\n## Next required stage\n\nRun effect-vector construction and LPF assignment / clustering on expanded policy results.\n\nRecommended sequence:\n\n```text\n1. Build condition/action effect vectors from da_asa2k4_policy_results.csv\n2. Assign or recluster latent policy factors\n3. Join LPF labels with 2K.3 DenseDSTA features using graph_id + condition\n4. Run DenseDSTA -> TransferableLPF CV\n```\n\n## Main comparison baseline\n\nCompare against 2J.1:\n\n```text\nleave-one-condition MacroF1 baseline: 0.5397\nleave-one-family MacroF1 baseline:    0.3824\n```\n"""
    (OUT_DIR / "da_asa2k4_next_stage_seed.md").write_text(text, encoding="utf-8")


def main() -> None:
    base = standardize_keys(read_csv(BASELINE_PATH, "baseline scores"), "baseline scores")
    dsta = standardize_keys(read_csv(DSTA_PATH, "DSTA features"), "DSTA features")

    # Keep only baseline rows; DSTA path is validated here for continuity.
    key_overlap = len(base[["graph_id", "condition"]].drop_duplicates().merge(
        dsta[["graph_id", "condition"]].drop_duplicates(), on=["graph_id", "condition"], how="inner"
    ))

    policy = generate_policy_results(base)
    cond_action = summarize(policy, ["condition_family", "condition", "action_class"])
    fam_action = summarize(policy, ["condition_family", "action_class"])

    policy_path = OUT_DIR / "da_asa2k4_policy_results.csv"
    cond_path = OUT_DIR / "da_asa2k4_policy_condition_action_summary.csv"
    fam_path = OUT_DIR / "da_asa2k4_policy_family_action_summary.csv"
    diag_path = OUT_DIR / "da_asa2k4_diagnostics.json"

    policy.to_csv(policy_path, index=False, encoding="utf-8-sig")
    cond_action.to_csv(cond_path, index=False, encoding="utf-8-sig")
    fam_action.to_csv(fam_path, index=False, encoding="utf-8-sig")

    diag = {
        "stage": "DA-ASA-2K.4",
        "purpose": "Expanded policy_results table generation across DA-ASA action set",
        "verdict": "POLICY_RESULTS_TABLE_COMPLETE",
        "root": str(ROOT),
        "input_baseline_csv": str(BASELINE_PATH),
        "input_dsta_csv": str(DSTA_PATH),
        "output_dir": str(OUT_DIR),
        "n_samples": int(len(base)),
        "n_conditions": int(base["condition"].nunique()),
        "n_condition_families": int(base["condition_family"].nunique()),
        "n_actions": int(len(ACTIONS)),
        "n_policy_rows": int(len(policy)),
        "dsta_key_overlap": int(key_overlap),
        "mean_functional_score": float(policy["functional_score"].mean()),
        "mean_pred_clean": float(policy["pred_clean"].mean()),
        "mean_R_final": float(policy["R_final"].mean()),
        "outputs": {
            "policy_results": str(policy_path),
            "condition_action_summary": str(cond_path),
            "family_action_summary": str(fam_path),
            "next_stage_seed": str(OUT_DIR / "da_asa2k4_next_stage_seed.md"),
        },
        "next_stage": "DA-ASA-2K.5 effect vectors, LPF assignment, DenseDSTA->TransferableLPF CV",
        "caveat": "This is table-level policy outcome generation, not true activation intervention.",
    }
    diag_path.write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")
    write_seed(diag)

    print(json.dumps(diag, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
