# -*- coding: utf-8 -*-
# DA-ASA-2D.2b Predicted Policy Table Replay
# Uses graph_id + condition + canonical raw policy mapping.
# No Windows paths are embedded in docstrings.

import argparse
import json
from pathlib import Path

import numpy as np
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

CANONICAL_POLICY_PRIORITY = {
    "NO_INTERVENTION": [
        "no_intervention",
    ],
    "CORE_CLOSURE_UPDATE": [
        "fixed_to_update_a015",
        "fixed_to_update_a020",
        "source_to_update_a005",
    ],
    "OVERRIDE_THREE_STAGE": [
        "fixed_three_stage_all",
        "condition_aware_v2_override_3stage",
        "generalized_three_stage_v2",
        "generalized_three_stage_v1",
    ],
    "EQUAL_EVIDENCE_ORDER": [
        "equal_strong_order",
        "equal_light_order",
        "source_no_intervention_equal_light",
    ],
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


def infer_action_from_policy(policy_name):
    if pd.isna(policy_name):
        return "UNKNOWN"
    t = str(policy_name).lower()

    if t == "no_intervention" or "no_intervention" in t or "no intervention" in t:
        return "NO_INTERVENTION"

    if "policy_selection_rule" in t:
        return "UNKNOWN"

    if "equal_evidence" in t or "equal_strong" in t or "equal_light" in t or ("equal" in t and "order" in t):
        return "EQUAL_EVIDENCE_ORDER"

    if "fixed_three_stage" in t:
        return "OVERRIDE_THREE_STAGE"
    if "override" in t and ("three" in t or "stage" in t or "strong" in t):
        return "OVERRIDE_THREE_STAGE"
    if "override" in t:
        return "OVERRIDE_THREE_STAGE"

    if "fixed_to_update" in t or "core_closure" in t or "to_update" in t or t.endswith("_update"):
        return "CORE_CLOSURE_UPDATE"

    return norm_policy_label(policy_name)


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


def select_predictions(pred_raw, feature_set, split, model):
    for col in ["feature_set", "split", "model"]:
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
    sub = sub.drop_duplicates("_key", keep="first").copy()

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
    return sub, pred_col


def load_policy_results(policy_files):
    parts = []
    failed = []
    for p in policy_files:
        try:
            df = read_csv_safe(p)
            df["source_policy_file"] = str(p)
            parts.append(df)
        except Exception as e:
            failed.append({"path": str(p), "error": repr(e)})
    if not parts:
        raise RuntimeError(f"No policy result files loaded. Failed={failed}")

    pol = pd.concat(parts, ignore_index=True)
    pol = add_norm_keys(pol, "policy_results")
    if "policy" not in pol.columns:
        raise ValueError(f"policy_results has no policy column; columns={list(pol.columns)}")
    pol["_policy_raw_norm"] = pol["policy"].astype(str).str.strip().str.lower()
    pol["_action_class"] = pol["policy"].map(infer_action_from_policy)
    return pol, failed


def build_outcome_table(pol):
    rows = []
    missing = []
    for action, priorities in CANONICAL_POLICY_PRIORITY.items():
        candidates = pol[pol["_policy_raw_norm"].isin([p.lower() for p in priorities])].copy()
        if candidates.empty:
            missing.append({"action_class": action, "priorities": priorities})
            continue

        # choose the highest-priority policy available per key
        priority_rank = {p.lower(): i for i, p in enumerate(priorities)}
        candidates["_priority"] = candidates["_policy_raw_norm"].map(priority_rank).fillna(999).astype(int)
        candidates = candidates.sort_values(["_key", "_priority"])
        chosen = candidates.drop_duplicates("_key", keep="first").copy()
        chosen["_selected_action_class"] = action
        chosen["_selected_policy_raw"] = chosen["policy"]
        rows.append(chosen)

    if not rows:
        raise RuntimeError("No canonical policy outcomes found.")

    out = pd.concat(rows, ignore_index=True)
    return out, missing


def clean_bool_series(s):
    if s.dtype == bool:
        return s.astype(float)
    return pd.to_numeric(s, errors="coerce")


def prepare_baseline(base):
    base = base.copy()
    if "baseline_pred_clean" in base.columns:
        base["_baseline_pred_clean"] = clean_bool_series(base["baseline_pred_clean"])
    elif "pred_clean" in base.columns:
        base["_baseline_pred_clean"] = clean_bool_series(base["pred_clean"])
    else:
        base["_baseline_pred_clean"] = np.nan

    if "baseline_R_final" in base.columns:
        base["_baseline_R_final"] = pd.to_numeric(base["baseline_R_final"], errors="coerce")
    elif "R_final" in base.columns:
        base["_baseline_R_final"] = pd.to_numeric(base["R_final"], errors="coerce")
    else:
        base["_baseline_R_final"] = np.nan

    return base


def merge_mode(pred, base, outcomes, mode_name, action_col):
    tmp = pred[[
        "_key", "_graph_norm", "_condition_norm", "_true_policy_norm", "_pred_policy_norm"
    ]].copy()

    tmp["_requested_action"] = tmp[action_col]

    merged = tmp.merge(
        base[["_key", "_baseline_pred_clean", "_baseline_R_final"]],
        on="_key",
        how="left",
    )

    merged = merged.merge(
        outcomes[[
            "_key", "_selected_action_class", "_selected_policy_raw",
            "R_final", "pred_clean"
        ]].rename(columns={
            "_selected_action_class": "_requested_action",
            "R_final": "replay_R_final",
            "pred_clean": "replay_pred_clean",
        }),
        on=["_key", "_requested_action"],
        how="left",
    )

    merged["replay_mode"] = mode_name
    merged["missing_action_outcome"] = merged["replay_R_final"].isna() & merged["replay_pred_clean"].isna()
    merged["replay_pred_clean_num"] = clean_bool_series(merged["replay_pred_clean"])
    merged["replay_R_final_num"] = pd.to_numeric(merged["replay_R_final"], errors="coerce")
    merged["baseline_clean_num"] = pd.to_numeric(merged["_baseline_pred_clean"], errors="coerce")
    merged["baseline_R_num"] = pd.to_numeric(merged["_baseline_R_final"], errors="coerce")
    return merged


def summarize_mode(df, mode_name):
    protected = df["_true_policy_norm"].eq("NO_INTERVENTION")
    target = ~protected

    def mean_safe(s):
        return float(pd.to_numeric(s, errors="coerce").mean()) if len(s) else float("nan")

    clean_rate = mean_safe(df["replay_pred_clean_num"])
    base_clean = mean_safe(df["baseline_clean_num"])
    r_final = mean_safe(df["replay_R_final_num"])
    base_r = mean_safe(df["baseline_R_num"])

    target_clean = mean_safe(df.loc[target, "replay_pred_clean_num"])
    target_base_clean = mean_safe(df.loc[target, "baseline_clean_num"])
    prot_clean = mean_safe(df.loc[protected, "replay_pred_clean_num"])
    prot_base_clean = mean_safe(df.loc[protected, "baseline_clean_num"])

    protected_harm = 0.0
    if protected.any():
        harm = (df.loc[protected, "replay_pred_clean_num"] < df.loc[protected, "baseline_clean_num"]).mean()
        protected_harm = float(harm)

    return {
        "replay_mode": mode_name,
        "n": int(len(df)),
        "missing_action_rate": float(df["missing_action_outcome"].mean()),
        "clean_rate": clean_rate,
        "baseline_clean_rate": base_clean,
        "clean_rate_gain": clean_rate - base_clean if pd.notna(clean_rate) and pd.notna(base_clean) else float("nan"),
        "mean_R_final": r_final,
        "baseline_R_final": base_r,
        "mean_R_gain": r_final - base_r if pd.notna(r_final) and pd.notna(base_r) else float("nan"),
        "target_clean_rate": target_clean,
        "target_baseline_clean_rate": target_base_clean,
        "target_clean_gain": target_clean - target_base_clean if pd.notna(target_clean) and pd.notna(target_base_clean) else float("nan"),
        "protected_clean_rate": prot_clean,
        "protected_baseline_clean_rate": prot_base_clean,
        "protected_clean_gain": prot_clean - prot_base_clean if pd.notna(prot_clean) and pd.notna(prot_base_clean) else float("nan"),
        "protected_harm": protected_harm,
        "safety_adjusted_target_gain": (target_clean - target_base_clean - protected_harm)
            if pd.notna(target_clean) and pd.notna(target_base_clean) else float("nan"),
    }


def main():
    root = Path.cwd()
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-file", default=str(root / "da_asa_2d1b_safe_outputs" / "da_asa2d1b_predictions.csv"))
    ap.add_argument("--baseline-file", default=str(root / "da_asa2c_outputs" / "da_asa2c_baseline_scores.csv"))
    ap.add_argument("--policy-results-file", action="append", default=None)
    ap.add_argument("--feature-set", default="strict_graph_dsta_only")
    ap.add_argument("--split", default="group_graph")
    ap.add_argument("--model", default="logreg_balanced")
    ap.add_argument("--out-dir", default=str(root / "da_asa_2d2b_outputs"))
    args = ap.parse_args()

    if args.policy_results_file is None:
        args.policy_results_file = [
            str(root / "da_asa2a_outputs" / "da_asa2a_policy_results.csv"),
            str(root / "da_asa2b_outputs" / "da_asa2b_policy_results.csv"),
            str(root / "da_asa2c_outputs" / "da_asa2c_policy_results.csv"),
        ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_raw = read_csv_safe(args.predictions_file)
    pred, pred_col = select_predictions(pred_raw, args.feature_set, args.split, args.model)

    base = prepare_baseline(add_norm_keys(read_csv_safe(args.baseline_file), "baseline"))
    pol, failed_policy_files = load_policy_results(args.policy_results_file)
    outcomes, missing_canonical = build_outcome_table(pol)

    # Check key/action coverage
    action_pairs = set(zip(outcomes["_key"], outcomes["_selected_action_class"]))
    pred_action_found = np.mean([(k, a) in action_pairs for k, a in zip(pred["_key"], pred["_pred_policy_norm"])])
    oracle_action_found = np.mean([(k, a) in action_pairs for k, a in zip(pred["_key"], pred["_true_policy_norm"])])

    modes = []
    modes.append(merge_mode(pred, base, outcomes, "predicted_policy", "_pred_policy_norm"))
    modes.append(merge_mode(pred, base, outcomes, "oracle_policy", "_true_policy_norm"))

    fixed_actions = {
        "no_intervention": "NO_INTERVENTION",
        "fixed_core_update": "CORE_CLOSURE_UPDATE",
        "fixed_override_three_stage": "OVERRIDE_THREE_STAGE",
        "fixed_equal_order": "EQUAL_EVIDENCE_ORDER",
    }
    for name, action in fixed_actions.items():
        pred_tmp = pred.copy()
        pred_tmp["_fixed_action"] = action
        modes.append(merge_mode(pred_tmp, base, outcomes, name, "_fixed_action"))

    replay = pd.concat(modes, ignore_index=True)
    summary_rows = [summarize_mode(m, mode) for mode, m in replay.groupby("replay_mode", sort=False)]
    summary = pd.DataFrame(summary_rows)

    replay.to_csv(out_dir / "da_asa2d2b_sample_replay.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "da_asa2d2b_replay_summary.csv", index=False, encoding="utf-8-sig")

    # condition summary for predicted and oracle
    cond_summary = replay.groupby(["replay_mode", "_condition_norm"]).agg(
        n=("_key", "size"),
        missing_action_rate=("missing_action_outcome", "mean"),
        clean_rate=("replay_pred_clean_num", "mean"),
        baseline_clean_rate=("baseline_clean_num", "mean"),
        mean_R_final=("replay_R_final_num", "mean"),
        baseline_R_final=("baseline_R_num", "mean"),
    ).reset_index()
    cond_summary["clean_rate_gain"] = cond_summary["clean_rate"] - cond_summary["baseline_clean_rate"]
    cond_summary["mean_R_gain"] = cond_summary["mean_R_final"] - cond_summary["baseline_R_final"]
    cond_summary.to_csv(out_dir / "da_asa2d2b_condition_summary.csv", index=False, encoding="utf-8-sig")

    pred_acc = float((pred["_pred_policy_norm"] == pred["_true_policy_norm"]).mean())
    pred_row = summary[summary["replay_mode"] == "predicted_policy"].iloc[0].to_dict()
    oracle_row = summary[summary["replay_mode"] == "oracle_policy"].iloc[0].to_dict()
    no_row = summary[summary["replay_mode"] == "no_intervention"].iloc[0].to_dict()

    fixed_rows = summary[summary["replay_mode"].isin(["fixed_core_update", "fixed_override_three_stage", "fixed_equal_order"])]
    best_fixed = fixed_rows.sort_values("safety_adjusted_target_gain", ascending=False).iloc[0].to_dict()

    verdict = "FAIL_REPLAY"
    if pred_row["missing_action_rate"] < 0.05 and oracle_row["missing_action_rate"] < 0.05:
        if pred_row["safety_adjusted_target_gain"] >= best_fixed["safety_adjusted_target_gain"] - 1e-9:
            verdict = "PASS_PREDICTED_POLICY_TABLE_REPLAY"
        elif pred_row["safety_adjusted_target_gain"] > no_row["safety_adjusted_target_gain"]:
            verdict = "PASS_LITE_PREDICTED_BEATS_NO_INTERVENTION"
        else:
            verdict = "FAIL_PREDICTED_NOT_BETTER_THAN_BASELINES"

    verdict_obj = {
        "experiment": "DA-ASA-2D.2b Predicted Policy Table Replay",
        "method": "graph_id_condition_canonical_policy_replay",
        "selection": {
            "feature_set": args.feature_set,
            "split": args.split,
            "model": args.model,
            "pred_col": pred_col,
            "prediction_accuracy": pred_acc,
        },
        "coverage": {
            "pred_action_found_rate_by_key_plus_action": float(pred_action_found),
            "oracle_action_found_rate_by_key_plus_action": float(oracle_action_found),
            "missing_canonical_policy_classes": missing_canonical,
            "failed_policy_files": failed_policy_files,
        },
        "predicted_policy": pred_row,
        "oracle_policy": oracle_row,
        "no_intervention": no_row,
        "best_fixed_policy": best_fixed,
        "verdict": verdict,
        "caveats": [
            "This is table replay from existing policy_results, not a true model rerun.",
            "Canonical action mapping uses fixed_to_update_a015, fixed_three_stage_all, equal_strong_order, and no_intervention when available.",
            "If predicted policies include impossible unseen classes, table replay cannot validate those without model rerun.",
        ],
        "outputs": {
            "sample_replay": str(out_dir / "da_asa2d2b_sample_replay.csv"),
            "replay_summary": str(out_dir / "da_asa2d2b_replay_summary.csv"),
            "condition_summary": str(out_dir / "da_asa2d2b_condition_summary.csv"),
            "verdict": str(out_dir / "da_asa2d2b_verdict.json"),
        },
    }

    with open(out_dir / "da_asa2d2b_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict_obj, f, ensure_ascii=False, indent=2)

    print(json.dumps(verdict_obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
