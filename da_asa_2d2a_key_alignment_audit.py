# -*- coding: utf-8 -*-
"""
DA-ASA-2D.2a Replay Key Alignment Audit

Purpose
-------
Audit whether predictions, baseline_scores, and policy_results can be aligned
by graph_id + condition, without relying on sample_id / row_index.

Default paths assume your local project directory:
C:\Users\ZH\Desktop\AGI\python_script

Outputs
-------
da_asa_2d2a_key_audit_outputs/
  da_asa2d2a_key_audit_summary.json
  da_asa2d2a_pred_selected_preview.csv
  da_asa2d2a_base_preview.csv
  da_asa2d2a_policy_preview.csv
  da_asa2d2a_key_overlap.csv
"""

import argparse
import json
from pathlib import Path
import pandas as pd


POLICY_MAP = {
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


def norm_condition(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def norm_graph(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    # normalize common numeric-like graph ids: "1.0" -> "1"
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def norm_policy_label(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    aliases = {
        "no_intervention": "NO_INTERVENTION",
        "none": "NO_INTERVENTION",
        "control": "NO_INTERVENTION",
        "core_closure_update": "CORE_CLOSURE_UPDATE",
        "closure_update": "CORE_CLOSURE_UPDATE",
        "to_update": "CORE_CLOSURE_UPDATE",
        "override_three_stage": "OVERRIDE_THREE_STAGE",
        "closure_override": "OVERRIDE_THREE_STAGE",
        "equal_evidence_order": "EQUAL_EVIDENCE_ORDER",
        "competition_equal_evidence": "EQUAL_EVIDENCE_ORDER",
        "unknown": "UNKNOWN",
    }
    low = s.lower()
    return aliases.get(low, s.upper())


def add_norm_keys(df):
    df = df.copy()
    if "condition" not in df.columns:
        raise ValueError(f"No condition column found. Columns={list(df.columns)}")
    if "graph_id" not in df.columns:
        raise ValueError(f"No graph_id column found. Columns={list(df.columns)}")
    df["_condition_norm"] = df["condition"].map(norm_condition)
    df["_graph_norm"] = df["graph_id"].map(norm_graph)
    df["_key"] = df["_graph_norm"] + "||" + df["_condition_norm"]
    return df


def read_csv_safe(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    df = pd.read_csv(p)
    # remove duplicate column names if any
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    return df


def select_predictions(pred, feature_set, split, model):
    pred = pred.copy()

    for col in ["feature_set", "split", "model"]:
        if col not in pred.columns:
            raise ValueError(f"Predictions file missing required column: {col}. Columns={list(pred.columns)}")

    sub = pred[
        (pred["feature_set"].astype(str) == str(feature_set)) &
        (pred["split"].astype(str) == str(split)) &
        (pred["model"].astype(str) == str(model))
    ].copy()

    if sub.empty:
        available = pred[["feature_set", "split", "model"]].drop_duplicates().head(50)
        raise RuntimeError(
            "No prediction rows matched requested feature_set/split/model.\n"
            f"Requested: {feature_set} / {split} / {model}\n"
            f"Available first 50 combinations:\n{available.to_string(index=False)}"
        )

    # If duplicate prediction rows exist per graph+condition, keep first.
    sub = add_norm_keys(sub)
    before = len(sub)
    sub = sub.drop_duplicates("_key", keep="first").copy()
    after = len(sub)

    # Ensure true policy label exists
    if "policy_class" not in sub.columns:
        if "true" in sub.columns:
            sub["policy_class"] = sub["true"].map(norm_policy_label)
        elif "y_true" in sub.columns:
            sub["policy_class"] = sub["y_true"].map(norm_policy_label)
        else:
            sub["policy_class"] = sub["_condition_norm"].map(lambda c: POLICY_MAP.get(c, "NO_INTERVENTION"))

    # Ensure predicted policy label exists
    pred_col = None
    for c in ["pred", "y_pred", "pred_policy", "predicted_policy", "policy_pred", "pred_class"]:
        if c in sub.columns:
            pred_col = c
            break
    if pred_col is None:
        raise ValueError(f"No prediction column found. Columns={list(sub.columns)}")

    sub["_pred_policy_norm"] = sub[pred_col].map(norm_policy_label)
    sub["_true_policy_norm"] = sub["policy_class"].map(norm_policy_label)

    return sub, {"initial_rows_for_combo": before, "deduped_rows_for_combo": after, "pred_col": pred_col}


def policy_results_action_class(pol):
    pol = pol.copy()
    if "policy" not in pol.columns:
        raise ValueError(f"policy_results missing policy column. Columns={list(pol.columns)}")

    # The policy names in files may already correspond to the desired action class.
    # We infer from policy text conservatively.
    def infer_action(s):
        if pd.isna(s):
            return "UNKNOWN"
        t = str(s).lower()
        if "no" in t and "intervention" in t:
            return "NO_INTERVENTION"
        if "strong_equal" in t or "equal" in t and "order" in t:
            return "EQUAL_EVIDENCE_ORDER"
        if "equal_evidence" in t:
            return "EQUAL_EVIDENCE_ORDER"
        if "override" in t and ("three" in t or "stage" in t or "strong" in t):
            return "OVERRIDE_THREE_STAGE"
        if "override" in t:
            return "OVERRIDE_THREE_STAGE"
        if "core" in t and "closure" in t:
            return "CORE_CLOSURE_UPDATE"
        if "to_update" in t or "update" in t:
            return "CORE_CLOSURE_UPDATE"
        if "stable" in t and "fixed" in t:
            return "NO_INTERVENTION"
        return norm_policy_label(s)

    pol["_action_class"] = pol["policy"].map(infer_action)
    return pol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-file", default=r"C:\Users\ZH\Desktop\AGI\python_script\da_asa_2d1b_safe_outputs\da_asa2d1b_predictions.csv")
    ap.add_argument("--baseline-file", default=r"C:\Users\ZH\Desktop\AGI\python_script\da_asa2c_outputs\da_asa2c_baseline_scores.csv")
    ap.add_argument("--policy-results-file", action="append", default=[
        r"C:\Users\ZH\Desktop\AGI\python_script\da_asa2a_outputs\da_asa2a_policy_results.csv",
        r"C:\Users\ZH\Desktop\AGI\python_script\da_asa2b_outputs\da_asa2b_policy_results.csv",
        r"C:\Users\ZH\Desktop\AGI\python_script\da_asa2c_outputs\da_asa2c_policy_results.csv",
    ])
    ap.add_argument("--feature-set", default="strict_graph_dsta_only")
    ap.add_argument("--split", default="group_graph")
    ap.add_argument("--model", default="logreg_balanced")
    ap.add_argument("--out-dir", default=r"C:\Users\ZH\Desktop\AGI\python_script\da_asa_2d2a_key_audit_outputs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_raw = read_csv_safe(args.predictions_file)
    base_raw = read_csv_safe(args.baseline_file)
    pol_parts = []
    failed_policy_files = []
    for p in args.policy_results_file:
        try:
            dfp = read_csv_safe(p)
            dfp["source_policy_file"] = str(p)
            pol_parts.append(dfp)
        except Exception as e:
            failed_policy_files.append({"path": str(p), "error": repr(e)})

    if not pol_parts:
        raise RuntimeError("No policy results files loaded.")

    pol_raw = pd.concat(pol_parts, ignore_index=True)
    pred, pred_info = select_predictions(pred_raw, args.feature_set, args.split, args.model)
    base = add_norm_keys(base_raw)
    pol = add_norm_keys(policy_results_action_class(pol_raw))

    pred_keys = set(pred["_key"])
    base_keys = set(base["_key"])
    pol_keys = set(pol["_key"])

    overlap_rows = [
        {"pair": "pred_vs_base", "left_n": len(pred_keys), "right_n": len(base_keys), "overlap": len(pred_keys & base_keys), "left_missing": len(pred_keys - base_keys)},
        {"pair": "pred_vs_policy", "left_n": len(pred_keys), "right_n": len(pol_keys), "overlap": len(pred_keys & pol_keys), "left_missing": len(pred_keys - pol_keys)},
        {"pair": "base_vs_policy", "left_n": len(base_keys), "right_n": len(pol_keys), "overlap": len(base_keys & pol_keys), "left_missing": len(base_keys - pol_keys)},
    ]
    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(out_dir / "da_asa2d2a_key_overlap.csv", index=False, encoding="utf-8-sig")

    # Check action coverage for predicted and oracle policy classes.
    action_pairs = pol[["_key", "_action_class"]].drop_duplicates()
    pred_action = pred[["_key", "_pred_policy_norm", "_true_policy_norm", "_graph_norm", "_condition_norm"]].copy()
    pred_action = pred_action.merge(
        action_pairs.rename(columns={"_action_class": "_pred_policy_norm"}),
        on=["_key", "_pred_policy_norm"],
        how="left",
        indicator="pred_action_merge",
    )
    pred_action_found = (pred_action["pred_action_merge"] == "both").mean()

    oracle_action = pred[["_key", "_pred_policy_norm", "_true_policy_norm", "_graph_norm", "_condition_norm"]].copy()
    oracle_action = oracle_action.merge(
        action_pairs.rename(columns={"_action_class": "_true_policy_norm"}),
        on=["_key", "_true_policy_norm"],
        how="left",
        indicator="oracle_action_merge",
    )
    oracle_action_found = (oracle_action["oracle_action_merge"] == "both").mean()

    # Save previews
    pred.head(50).to_csv(out_dir / "da_asa2d2a_pred_selected_preview.csv", index=False, encoding="utf-8-sig")
    base.head(50).to_csv(out_dir / "da_asa2d2a_base_preview.csv", index=False, encoding="utf-8-sig")
    pol.head(100).to_csv(out_dir / "da_asa2d2a_policy_preview.csv", index=False, encoding="utf-8-sig")
    pred_action.to_csv(out_dir / "da_asa2d2a_pred_action_match.csv", index=False, encoding="utf-8-sig")
    oracle_action.to_csv(out_dir / "da_asa2d2a_oracle_action_match.csv", index=False, encoding="utf-8-sig")

    summary = {
        "experiment": "DA-ASA-2D.2a Replay Key Alignment Audit",
        "predictions_file": str(args.predictions_file),
        "baseline_file": str(args.baseline_file),
        "policy_results_files": [str(x) for x in args.policy_results_file],
        "failed_policy_files": failed_policy_files,
        "requested_selection": {
            "feature_set": args.feature_set,
            "split": args.split,
            "model": args.model,
        },
        "prediction_selection": pred_info,
        "shapes": {
            "pred_raw": list(pred_raw.shape),
            "pred_selected": list(pred.shape),
            "baseline": list(base.shape),
            "policy_results": list(pol.shape),
        },
        "key_overlap": overlap_rows,
        "policy_action_counts": pol["_action_class"].value_counts(dropna=False).to_dict(),
        "prediction_policy_counts": pred["_pred_policy_norm"].value_counts(dropna=False).to_dict(),
        "true_policy_counts": pred["_true_policy_norm"].value_counts(dropna=False).to_dict(),
        "pred_action_found_rate_by_key_plus_action": float(pred_action_found),
        "oracle_action_found_rate_by_key_plus_action": float(oracle_action_found),
        "missing_pred_action_examples": pred_action[pred_action["pred_action_merge"] != "both"][["_graph_norm","_condition_norm","_pred_policy_norm","_true_policy_norm"]].head(20).to_dict(orient="records"),
        "missing_oracle_action_examples": oracle_action[oracle_action["oracle_action_merge"] != "both"][["_graph_norm","_condition_norm","_pred_policy_norm","_true_policy_norm"]].head(20).to_dict(orient="records"),
        "outputs": {
            "summary": str(out_dir / "da_asa2d2a_key_audit_summary.json"),
            "key_overlap": str(out_dir / "da_asa2d2a_key_overlap.csv"),
            "pred_action_match": str(out_dir / "da_asa2d2a_pred_action_match.csv"),
            "oracle_action_match": str(out_dir / "da_asa2d2a_oracle_action_match.csv"),
        }
    }

    if len(pred_keys & base_keys) == len(pred_keys) and len(pred_keys & pol_keys) == len(pred_keys):
        if pred_action_found > 0.95 and oracle_action_found > 0.95:
            summary["verdict"] = "PASS_KEYS_AND_ACTIONS_ALIGNED"
        else:
            summary["verdict"] = "PASS_KEYS_FAIL_ACTION_CLASS_ALIGNMENT"
    else:
        summary["verdict"] = "FAIL_KEY_ALIGNMENT"

    with open(out_dir / "da_asa2d2a_key_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
