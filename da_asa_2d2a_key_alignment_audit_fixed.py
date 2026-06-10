# -*- coding: utf-8 -*-
# DA-ASA-2D.2a Replay Key Alignment Audit
# This script avoids Windows backslash examples in docstrings to prevent unicodeescape errors.

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
    low = s.lower()
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
    return aliases.get(low, s.upper())


def read_csv_safe(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    df = pd.read_csv(p)
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    return df


def add_norm_keys(df, name):
    df = df.copy()
    if "condition" not in df.columns:
        raise ValueError(f"{name}: no condition column. Columns={list(df.columns)}")
    if "graph_id" not in df.columns:
        raise ValueError(f"{name}: no graph_id column. Columns={list(df.columns)}")
    df["_condition_norm"] = df["condition"].map(norm_condition)
    df["_graph_norm"] = df["graph_id"].map(norm_graph)
    df["_key"] = df["_graph_norm"] + "||" + df["_condition_norm"]
    return df


def infer_action_from_policy(policy_name):
    if pd.isna(policy_name):
        return "UNKNOWN"
    t = str(policy_name).lower()

    if "no_intervention" in t or "no intervention" in t:
        return "NO_INTERVENTION"

    if "policy_selection_rule" in t:
        # A whole rule is not one action; keep as UNKNOWN so we do not falsely match.
        return "UNKNOWN"

    if "equal_evidence" in t or ("equal" in t and "order" in t):
        return "EQUAL_EVIDENCE_ORDER"

    if "override" in t and ("three" in t or "stage" in t or "strong" in t):
        return "OVERRIDE_THREE_STAGE"
    if "override" in t:
        return "OVERRIDE_THREE_STAGE"

    if "core_closure" in t or "to_update" in t or "update" in t:
        return "CORE_CLOSURE_UPDATE"

    return norm_policy_label(policy_name)


def select_predictions(pred_raw, feature_set, split, model):
    required = ["feature_set", "split", "model"]
    for col in required:
        if col not in pred_raw.columns:
            raise ValueError(f"predictions missing {col}; columns={list(pred_raw.columns)}")

    sub = pred_raw[
        (pred_raw["feature_set"].astype(str) == str(feature_set)) &
        (pred_raw["split"].astype(str) == str(split)) &
        (pred_raw["model"].astype(str) == str(model))
    ].copy()

    if sub.empty:
        combos = pred_raw[["feature_set", "split", "model"]].drop_duplicates().head(80)
        raise RuntimeError(
            "No predictions matched requested selection.\n"
            f"requested feature_set={feature_set}, split={split}, model={model}\n"
            f"available:\n{combos.to_string(index=False)}"
        )

    sub = add_norm_keys(sub, "predictions")
    before = len(sub)
    sub = sub.drop_duplicates("_key", keep="first").copy()
    after = len(sub)

    if "policy_class" not in sub.columns:
        if "true" in sub.columns:
            sub["policy_class"] = sub["true"]
        elif "y_true" in sub.columns:
            sub["policy_class"] = sub["y_true"]
        else:
            sub["policy_class"] = sub["_condition_norm"].map(lambda c: POLICY_MAP.get(c, "NO_INTERVENTION"))

    pred_col = None
    for c in ["pred", "y_pred", "pred_policy", "predicted_policy", "policy_pred", "pred_class"]:
        if c in sub.columns:
            pred_col = c
            break
    if pred_col is None:
        raise ValueError(f"no prediction column found; columns={list(sub.columns)}")

    sub["_true_policy_norm"] = sub["policy_class"].map(norm_policy_label)
    sub["_pred_policy_norm"] = sub[pred_col].map(norm_policy_label)

    return sub, {"initial_rows_for_combo": int(before), "deduped_rows_for_combo": int(after), "pred_col": pred_col}


def main():
    default_root = Path.cwd()

    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-file", default=str(default_root / "da_asa_2d1b_safe_outputs" / "da_asa2d1b_predictions.csv"))
    ap.add_argument("--baseline-file", default=str(default_root / "da_asa2c_outputs" / "da_asa2c_baseline_scores.csv"))
    ap.add_argument("--policy-results-file", action="append", default=None)
    ap.add_argument("--feature-set", default="strict_graph_dsta_only")
    ap.add_argument("--split", default="group_graph")
    ap.add_argument("--model", default="logreg_balanced")
    ap.add_argument("--out-dir", default=str(default_root / "da_asa_2d2a_key_audit_outputs"))
    args = ap.parse_args()

    if args.policy_results_file is None:
        args.policy_results_file = [
            str(default_root / "da_asa2a_outputs" / "da_asa2a_policy_results.csv"),
            str(default_root / "da_asa2b_outputs" / "da_asa2b_policy_results.csv"),
            str(default_root / "da_asa2c_outputs" / "da_asa2c_policy_results.csv"),
        ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_raw = read_csv_safe(args.predictions_file)
    base_raw = read_csv_safe(args.baseline_file)
    pred, pred_info = select_predictions(pred_raw, args.feature_set, args.split, args.model)
    base = add_norm_keys(base_raw, "baseline")

    pol_parts = []
    failed = []
    for p in args.policy_results_file:
        try:
            df = read_csv_safe(p)
            df["source_policy_file"] = str(p)
            pol_parts.append(df)
        except Exception as e:
            failed.append({"path": str(p), "error": repr(e)})
    if not pol_parts:
        raise RuntimeError(f"No policy result files loaded. Failed={failed}")

    pol_raw = pd.concat(pol_parts, ignore_index=True)
    pol = add_norm_keys(pol_raw, "policy_results")
    if "policy" not in pol.columns:
        raise ValueError(f"policy_results has no policy column; columns={list(pol.columns)}")
    pol["_action_class"] = pol["policy"].map(infer_action_from_policy)

    pred_keys = set(pred["_key"])
    base_keys = set(base["_key"])
    pol_keys = set(pol["_key"])

    overlap = [
        {"pair": "pred_vs_base", "left_n": len(pred_keys), "right_n": len(base_keys), "overlap": len(pred_keys & base_keys), "left_missing": len(pred_keys - base_keys)},
        {"pair": "pred_vs_policy", "left_n": len(pred_keys), "right_n": len(pol_keys), "overlap": len(pred_keys & pol_keys), "left_missing": len(pred_keys - pol_keys)},
        {"pair": "base_vs_policy", "left_n": len(base_keys), "right_n": len(pol_keys), "overlap": len(base_keys & pol_keys), "left_missing": len(base_keys - pol_keys)},
    ]

    action_pairs = pol[["_key", "_action_class"]].drop_duplicates()

    pred_action = pred[["_key", "_graph_norm", "_condition_norm", "_pred_policy_norm", "_true_policy_norm"]].copy()
    pred_action = pred_action.merge(
        action_pairs.rename(columns={"_action_class": "_pred_policy_norm"}),
        on=["_key", "_pred_policy_norm"],
        how="left",
        indicator="pred_action_merge",
    )

    oracle_action = pred[["_key", "_graph_norm", "_condition_norm", "_pred_policy_norm", "_true_policy_norm"]].copy()
    oracle_action = oracle_action.merge(
        action_pairs.rename(columns={"_action_class": "_true_policy_norm"}),
        on=["_key", "_true_policy_norm"],
        how="left",
        indicator="oracle_action_merge",
    )

    pred_action_found = float((pred_action["pred_action_merge"] == "both").mean()) if len(pred_action) else 0.0
    oracle_action_found = float((oracle_action["oracle_action_merge"] == "both").mean()) if len(oracle_action) else 0.0

    pd.DataFrame(overlap).to_csv(out_dir / "da_asa2d2a_key_overlap.csv", index=False, encoding="utf-8-sig")
    pred.head(100).to_csv(out_dir / "da_asa2d2a_pred_selected_preview.csv", index=False, encoding="utf-8-sig")
    base.head(100).to_csv(out_dir / "da_asa2d2a_base_preview.csv", index=False, encoding="utf-8-sig")
    pol.head(200).to_csv(out_dir / "da_asa2d2a_policy_preview.csv", index=False, encoding="utf-8-sig")
    pred_action.to_csv(out_dir / "da_asa2d2a_pred_action_match.csv", index=False, encoding="utf-8-sig")
    oracle_action.to_csv(out_dir / "da_asa2d2a_oracle_action_match.csv", index=False, encoding="utf-8-sig")

    if len(pred_keys & base_keys) == len(pred_keys) and len(pred_keys & pol_keys) == len(pred_keys):
        if pred_action_found > 0.95 and oracle_action_found > 0.95:
            verdict = "PASS_KEYS_AND_ACTIONS_ALIGNED"
        else:
            verdict = "PASS_KEYS_FAIL_ACTION_CLASS_ALIGNMENT"
    else:
        verdict = "FAIL_KEY_ALIGNMENT"

    summary = {
        "experiment": "DA-ASA-2D.2a Replay Key Alignment Audit",
        "requested_selection": {"feature_set": args.feature_set, "split": args.split, "model": args.model},
        "prediction_selection": pred_info,
        "shapes": {
            "pred_raw": list(pred_raw.shape),
            "pred_selected": list(pred.shape),
            "baseline": list(base.shape),
            "policy_results": list(pol.shape),
        },
        "key_overlap": overlap,
        "policy_action_counts": pol["_action_class"].value_counts(dropna=False).to_dict(),
        "policy_raw_counts_head": pol["policy"].value_counts(dropna=False).head(50).to_dict(),
        "prediction_policy_counts": pred["_pred_policy_norm"].value_counts(dropna=False).to_dict(),
        "true_policy_counts": pred["_true_policy_norm"].value_counts(dropna=False).to_dict(),
        "pred_action_found_rate_by_key_plus_action": pred_action_found,
        "oracle_action_found_rate_by_key_plus_action": oracle_action_found,
        "missing_pred_action_examples": pred_action[pred_action["pred_action_merge"] != "both"][["_graph_norm", "_condition_norm", "_pred_policy_norm", "_true_policy_norm"]].head(20).to_dict(orient="records"),
        "missing_oracle_action_examples": oracle_action[oracle_action["oracle_action_merge"] != "both"][["_graph_norm", "_condition_norm", "_pred_policy_norm", "_true_policy_norm"]].head(20).to_dict(orient="records"),
        "failed_policy_files": failed,
        "verdict": verdict,
        "outputs": {
            "summary": str(out_dir / "da_asa2d2a_key_audit_summary.json"),
            "key_overlap": str(out_dir / "da_asa2d2a_key_overlap.csv"),
            "pred_action_match": str(out_dir / "da_asa2d2a_pred_action_match.csv"),
            "oracle_action_match": str(out_dir / "da_asa2d2a_oracle_action_match.csv"),
        },
    }

    with open(out_dir / "da_asa2d2a_key_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
