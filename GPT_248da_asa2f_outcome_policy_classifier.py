# -*- coding: utf-8 -*-
"""
DA-ASA-2F: OutcomePolicyClass Classifier v4

Purpose
-------
Train and audit classifiers for the target discovered in DA-ASA-2E:

    DSTA features -> OutcomePolicyClass

This script intentionally does NOT treat raw action labels as the only target.
It compares:

1. raw_policy_class
   target = oracle_action

2. outcome_policy_class
   target = oracle_outcome_class
   This is condition-specific and table-defined by DA-ASA-2E.

3. outcome_representative
   target = oracle_outcome_class_representative
   Usually collapses back to the representative action label, useful as a sanity check.

It also evaluates whether raw action mistakes are rescued by outcome equivalence.

Expected inputs
---------------
Required:
  - da_asa2e_prediction_outcome_equivalence.csv
  - a DA-ASA-2D.0h5 feature table, e.g.
      da_asa2d0h5_feature_table.csv
      da_asa2d0h5_dense_feature_table.csv
      da_asa2d_feature_table_h5.csv
      da_asa2d_feature_table.csv

Optional:
  - da_asa2e_equivalence_classes_long.csv
  - da_asa2d2_v2_policy_outcomes_aggregated.csv

Outputs
-------
  da_asa2f_outputs/
    da_asa2f_diagnostics.json
    da_asa2f_cv_summary.csv
    da_asa2f_predictions_long.csv
    da_asa2f_feature_sets.json
    da_asa2f_outcome_rescue_summary.csv
    da_asa2f_replay_summary.csv
    da_asa2f_best_confusion_*.csv

Notes
-----
- No model forward pass is run.
- All splits are graph-heldout GroupKFold by graph_id.
- Outcome equivalence is table-defined using pred_clean and R_final from 2E.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# 0. Config
# =============================================================================

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "da_asa2f_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_FILE_CANDIDATES = [
    "da_asa2e_prediction_outcome_equivalence.csv",
]

FEATURE_FILE_CANDIDATES = [
    "da_asa2d0h5_feature_table.csv",
    "da_asa2d0h5_dense_feature_table.csv",
    "da_asa2d0h5_final_feature_table.csv",
    "da_asa2d_feature_table_h5.csv",
    "da_asa2d_feature_table_dense.csv",
    "da_asa2d_feature_table.csv",
    "da_asa2d1b_feature_table.csv",
    "da_asa2d1b_training_table.csv",
    "da_asa2d1b_input_table.csv",
]

EQUIV_FILE_CANDIDATES = [
    "da_asa2e_equivalence_classes_long.csv",
]

POLICY_OUTCOME_CANDIDATES = [
    "da_asa2d2_v2_policy_outcomes_aggregated.csv",
    "da_asa2d2_policy_outcomes_aggregated.csv",
]

KEY_COLS = ["graph_id", "condition"]
RANDOM_SEED = 20260606
N_SPLITS_MAX = 5

STRICT_GRAPH_DSTA_COLS = [
    "dsta_signed_attractor_transport_signed_align_vs_clean",
    "dsta_signed_attractor_transport_rotation_abs_deg",
    "dsta_signed_attractor_transport_delta_norm",
    "dsta_signed_attractor_delta_align_to_update",
    "dsta_signed_attractor_delta_align_to_override",
]

# Columns with these patterns are not allowed as predictive features because they
# are labels, direct policy outcomes, or replay results.
LEAKAGE_PATTERNS = [
    r"(^|_)pred($|_)",
    r"(^|_)oracle($|_)",
    r"(^|_)true($|_)",
    r"(^|_)label($|_)",
    r"(^|_)target($|_)",
    r"policy_class",
    r"outcome_class",
    r"action_class",
    r"selected_action",
    r"predicted_policy",
    r"raw_action",
    r"representative",
    r"condition_family",
    r"row_index",
    r"pred_clean",
    r"r_final",
    r"clean_rate",
    r"hit_rate",
    r"success",
    r"safety_adjusted_gain",
    r"gain",
    r"score$",
]

ID_LIKE_COLS = {
    "graph_id", "condition", "sample_id", "row_id", "row_index", "seed", "prompt",
    "text", "surface", "source_file", "_source_file", "condition_family"
}


# =============================================================================
# 1. IO helpers
# =============================================================================

def find_file(candidates: List[str], required: bool = True) -> Optional[Path]:
    for name in candidates:
        direct = ROOT / name
        if direct.exists():
            return direct
        hits = list(ROOT.rglob(name))
        if hits:
            return hits[0]
    if required:
        raise FileNotFoundError(f"None of these files were found: {candidates}")
    return None


def read_csv_first(candidates: List[str], required: bool = True) -> Optional[pd.DataFrame]:
    path = find_file(candidates, required=required)
    if path is None:
        return None
    print(f"[LOAD] {path.name} -> {path}")
    return pd.read_csv(path)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def standardize_keys(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = normalize_columns(df)
    graph_col = find_col(df, ["graph_id", "graph", "gid"])
    cond_col = find_col(df, ["condition", "cond", "condition_name"])
    if graph_col is None or cond_col is None:
        raise KeyError(f"[{name}] cannot find graph_id/condition. Columns={list(df.columns)}")
    if graph_col != "graph_id":
        df = df.rename(columns={graph_col: "graph_id"})
    if cond_col != "condition":
        df = df.rename(columns={cond_col: "condition"})
    df["graph_id"] = df["graph_id"].astype(str).str.strip()
    df["condition"] = df["condition"].astype(str).str.strip()
    return df


def canonical_action(x) -> str:
    if pd.isna(x):
        return "MISSING_ACTION"
    s = str(x).strip()
    u = s.upper()
    aliases = {
        "NONE": "NO_INTERVENTION",
        "NOOP": "NO_INTERVENTION",
        "NO_OP": "NO_INTERVENTION",
        "NO INTERVENTION": "NO_INTERVENTION",
        "NO_INTERVENTION": "NO_INTERVENTION",
        "CORE": "CORE_CLOSURE_UPDATE",
        "CORE_CLOSURE": "CORE_CLOSURE_UPDATE",
        "CORE_UPDATE": "CORE_CLOSURE_UPDATE",
        "CLOSURE_UPDATE": "CORE_CLOSURE_UPDATE",
        "CORE_CLOSURE_UPDATE": "CORE_CLOSURE_UPDATE",
        "OVERRIDE": "OVERRIDE_THREE_STAGE",
        "OVERRIDE_3_STAGE": "OVERRIDE_THREE_STAGE",
        "THREE_STAGE_OVERRIDE": "OVERRIDE_THREE_STAGE",
        "OVERRIDE_THREE_STAGE": "OVERRIDE_THREE_STAGE",
        "EQUAL": "EQUAL_EVIDENCE_ORDER",
        "EQUAL_EVIDENCE": "EQUAL_EVIDENCE_ORDER",
        "EQUAL_ORDER": "EQUAL_EVIDENCE_ORDER",
        "EQUAL_EVIDENCE_ORDER": "EQUAL_EVIDENCE_ORDER",
    }
    if u in aliases:
        return aliases[u]
    if "NO" in u and "INTERVENTION" in u:
        return "NO_INTERVENTION"
    if "CORE" in u and ("CLOSURE" in u or "UPDATE" in u):
        return "CORE_CLOSURE_UPDATE"
    if "OVERRIDE" in u and ("THREE" in u or "3" in u or "STAGE" in u):
        return "OVERRIDE_THREE_STAGE"
    if "EQUAL" in u and ("EVIDENCE" in u or "ORDER" in u):
        return "EQUAL_EVIDENCE_ORDER"
    return u


# =============================================================================
# 2. Feature set construction
# =============================================================================

def is_leakage_col(col: str) -> bool:
    c = str(col).lower()
    if c in ID_LIKE_COLS:
        return True
    for pat in LEAKAGE_PATTERNS:
        if re.search(pat, c):
            return True
    return False


def numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if c in KEY_COLS:
            continue
        if is_leakage_col(c):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            nunique = df[c].nunique(dropna=True)
            if nunique > 1:
                cols.append(c)
    return cols


def build_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    all_safe = numeric_feature_columns(df)
    cols_l = {c: c.lower() for c in all_safe}

    def has_any(c: str, patterns: Iterable[str]) -> bool:
        lc = cols_l[c]
        return any(p in lc for p in patterns)

    feature_sets: Dict[str, List[str]] = {}

    strict = [c for c in STRICT_GRAPH_DSTA_COLS if c in df.columns and c in all_safe]
    if strict:
        feature_sets["strict_graph_dsta_only"] = strict

    baseline = [c for c in all_safe if has_any(c, ["baseline", "base_"])]
    if baseline:
        feature_sets["baseline_only"] = baseline

    dsta_dense = [
        c for c in all_safe
        if has_any(c, ["dsta", "attractor", "transport", "signed", "align", "rotation", "delta_norm"])
    ]
    if dsta_dense:
        feature_sets["dsta_dense_no_policy_outcome"] = dsta_dense

    graph_dsta = [
        c for c in dsta_dense
        if not has_any(c, ["condition_summary", "condition_attractor", "broadcast", "summary_"])
    ]
    if graph_dsta:
        feature_sets["graph_dsta_no_condition_broadcast"] = graph_dsta

    condition_broadcast = [
        c for c in dsta_dense
        if has_any(c, ["condition", "summary", "distance_matrix", "pairwise"])
    ]
    if condition_broadcast:
        feature_sets["condition_broadcast_dsta_only"] = condition_broadcast

    policyish = [c for c in all_safe if has_any(c, ["policy", "intervention", "steer"])]
    if policyish:
        feature_sets["policy_diagnostic_safe_numeric"] = policyish

    if all_safe:
        feature_sets["all_numeric_safe_no_policy_outcome"] = all_safe

    # Stable deterministic order and remove duplicates while preserving order.
    clean_sets = {}
    for name, cols in feature_sets.items():
        seen = set()
        clean_cols = []
        for c in cols:
            if c not in seen:
                clean_cols.append(c)
                seen.add(c)
        if clean_cols:
            clean_sets[name] = clean_cols
    return clean_sets


# =============================================================================
# 3. Evaluation
# =============================================================================

def make_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs",
            C=1.0,
            random_state=RANDOM_SEED,
        )),
    ])


def make_dummy() -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent")


def group_cv_predict(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    model,
    max_splits: int = N_SPLITS_MAX,
) -> np.ndarray:
    unique_groups = groups.astype(str).nunique()
    n_splits = max(2, min(max_splits, unique_groups))
    preds = np.empty(len(y), dtype=object)
    gkf = GroupKFold(n_splits=n_splits)

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), start=1):
        y_tr = y.iloc[tr]
        # Some very high-granularity targets can have absent classes in a fold;
        # that is acceptable for prediction but should be recorded by metrics.
        m = clone(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X.iloc[tr], y_tr)
        preds[te] = m.predict(X.iloc[te])
    return preds


def metric_row(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def build_action_to_outcome(equiv_long: Optional[pd.DataFrame]) -> Dict[Tuple[str, str], str]:
    mapping = {}
    if equiv_long is None:
        return mapping
    for _, r in equiv_long.iterrows():
        cond = str(r["condition"]).strip()
        act = canonical_action(r["action_class"])
        oc = str(r["outcome_class"]).strip()
        mapping[(cond, act)] = oc
    return mapping


def build_outcome_to_rep(equiv_long: Optional[pd.DataFrame], labels: pd.DataFrame) -> Dict[Tuple[str, str], str]:
    """Build condition-level outcome_class -> representative action mapping.

    In 2E outputs, da_asa2e_equivalence_classes_long.csv is condition-level
    and usually contains outcome_class_representative. The graph-level labels
    table may or may not still contain oracle_outcome_class columns by the
    time this helper is called, so label fallback must be optional.
    """
    mapping = {}

    if equiv_long is not None:
        if "outcome_class_representative" in equiv_long.columns:
            for _, r in equiv_long.iterrows():
                cond = str(r["condition"]).strip()
                oc = str(r["outcome_class"]).strip()
                rep = canonical_action(r["outcome_class_representative"])
                mapping[(cond, oc)] = rep
        elif {"outcome_class", "action_class"}.issubset(set(equiv_long.columns)):
            # Conservative fallback: use the first action encountered in each
            # condition/outcome class as the representative.
            for _, r in equiv_long.iterrows():
                cond = str(r["condition"]).strip()
                oc = str(r["outcome_class"]).strip()
                rep = canonical_action(r["action_class"])
                mapping.setdefault((cond, oc), rep)

    # Optional fallback from graph-level labels, only when the columns exist.
    label_cols = set(labels.columns)
    if {"condition", "oracle_outcome_class", "oracle_outcome_class_representative"}.issubset(label_cols):
        for _, r in labels.iterrows():
            cond = str(r["condition"]).strip()
            oc = str(r["oracle_outcome_class"]).strip()
            rep = canonical_action(r["oracle_outcome_class_representative"])
            mapping.setdefault((cond, oc), rep)

    return mapping


def outcome_from_action(row, action_to_outcome: Dict[Tuple[str, str], str]) -> Optional[str]:
    return action_to_outcome.get((str(row["condition"]).strip(), canonical_action(row["pred_label"])))


def representative_for_outcome(row, outcome_to_rep: Dict[Tuple[str, str], str]) -> Optional[str]:
    return outcome_to_rep.get((str(row["condition"]).strip(), str(row["pred_label"]).strip()))



def discover_and_build_feature_table(label_keys: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Fallback builder for environments where the 2D.0h5 feature table was not saved
    under a canonical name.

    It scans nearby CSVs, keeps files that contain graph_id + condition, aggregates
    numeric columns to one row per graph_id + condition, prefixes columns by source
    filename, and merges them into a sample-level feature table.

    Design goal: recover DSTA / baseline diagnostic features without using 2E labels
    or replay outcomes as predictors.
    """
    label_key_set = set(map(tuple, label_keys[KEY_COLS].astype(str).values.tolist()))

    include_keywords = [
        "dsta", "da_asa2d", "da_asa2a", "da_asa2b", "da_asa2c", "baseline_scores",
    ]
    exclude_keywords = [
        "da_asa2e", "da_asa2f", "replay", "equivalence", "metric_summary",
        "diagnostics", "prediction_outcome", "policy_outcomes_aggregated",
        "policy_summary", "condition_compression", "action_outcome_signatures",
    ]
    # policy_results are one-to-many and contain action outcome columns. They are
    # intentionally excluded for classifier features to avoid replay leakage.
    exclude_keywords += ["policy_results"]

    frames = []
    used_files = []
    skipped_reasons = defaultdict(int)

    for path in sorted(ROOT.rglob("*.csv")):
        name = path.name.lower()
        if not any(k in name for k in include_keywords):
            skipped_reasons["no_include_keyword"] += 1
            continue
        if any(k in name for k in exclude_keywords):
            skipped_reasons["excluded_keyword"] += 1
            continue
        try:
            df0 = pd.read_csv(path)
        except Exception as exc:
            print(f"[WARN] cannot read candidate feature CSV {path}: {exc}")
            skipped_reasons["read_error"] += 1
            continue
        try:
            df = standardize_keys(df0, f"feature candidate {path.name}")
        except Exception:
            skipped_reasons["no_keys"] += 1
            continue

        overlap = set(map(tuple, df[KEY_COLS].astype(str).values.tolist())) & label_key_set
        if not overlap:
            skipped_reasons["zero_key_overlap"] += 1
            continue

        num_cols = []
        for c in df.columns:
            if c in KEY_COLS:
                continue
            if is_leakage_col(c):
                continue
            if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 1:
                num_cols.append(c)

        if not num_cols:
            skipped_reasons["no_safe_numeric"] += 1
            continue

        small = df[KEY_COLS + num_cols].copy()
        # Keep only label keys to avoid accidental unrelated rows.
        small = small[small[KEY_COLS].astype(str).apply(tuple, axis=1).isin(label_key_set)].copy()
        if small.empty:
            skipped_reasons["empty_after_key_filter"] += 1
            continue
        if small.duplicated(KEY_COLS).any():
            small = small.groupby(KEY_COLS, as_index=False).mean(numeric_only=True)

        # Use a path-aware unique prefix. Several DA-ASA runs can contain CSV files
        # with the same stem in different output folders; using path.stem alone can
        # create duplicate columns during repeated fallback merges.
        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        prefix_raw = "__".join(rel.with_suffix("").parts[-3:])
        stem = re.sub(r"[^A-Za-z0-9]+", "_", prefix_raw).strip("_")
        # Add a short deterministic suffix so even identical leaf names from
        # different directories cannot collide after sanitization.
        import hashlib
        short_hash = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:8]
        stem = f"{stem}_{short_hash}"

        # Drop accidental duplicate/non-unique columns before renaming. Some source
        # tables are themselves merge products and contain *_x / *_y columns. We keep
        # them if numeric, but the unique prefix prevents merge collisions.
        small = small.loc[:, ~small.columns.duplicated()].copy()
        rename = {c: f"{stem}__{c}" for c in num_cols if c in small.columns}
        small = small.rename(columns=rename)
        frames.append(small)
        used_files.append(str(path))

    if not frames:
        print("[FALLBACK] Could not build feature table from nearby CSV files.")
        print(f"[FALLBACK] skipped reasons: {dict(skipped_reasons)}")
        return None

    out = label_keys[KEY_COLS].drop_duplicates().copy()
    seen_cols = set(out.columns)
    for idx, fr in enumerate(frames):
        fr = fr.loc[:, ~fr.columns.duplicated()].copy()
        # Defensive duplicate-column filter. This should rarely trigger because
        # prefixes are path-aware, but it prevents pandas MergeError if source
        # tables already contain duplicated names.
        keep_cols = KEY_COLS + [c for c in fr.columns if c not in KEY_COLS and c not in seen_cols]
        fr = fr[keep_cols]
        seen_cols.update([c for c in fr.columns if c not in KEY_COLS])
        out = out.merge(fr, on=KEY_COLS, how="left", validate="one_to_one")

    meta = {
        "builder": "discover_and_build_feature_table",
        "used_files_count": len(used_files),
        "used_files": used_files,
        "skipped_reasons": dict(skipped_reasons),
        "rows": int(len(out)),
        "numeric_feature_columns": int(len(numeric_feature_columns(out))),
    }
    with open(OUT_DIR / "da_asa2f_feature_table_fallback_builder.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    out.to_csv(OUT_DIR / "da_asa2f_fallback_feature_table.csv", index=False, encoding="utf-8-sig")
    print(f"[FALLBACK] built feature table from {len(used_files)} CSV files")
    print(f"[FALLBACK] safe numeric feature columns = {meta['numeric_feature_columns']}")
    return out


# =============================================================================
# 4. Main
# =============================================================================

def main() -> None:
    labels = read_csv_first(LABEL_FILE_CANDIDATES, required=True)
    labels = standardize_keys(labels, "2E prediction outcome equivalence")

    required_label_cols = [
        "oracle_action",
        "oracle_outcome_class",
        "oracle_outcome_class_representative",
    ]
    missing = [c for c in required_label_cols if c not in labels.columns]
    if missing:
        raise KeyError(f"2E label file is missing required columns: {missing}")

    labels["raw_policy_class"] = labels["oracle_action"].map(canonical_action)
    labels["outcome_policy_class"] = labels["oracle_outcome_class"].astype(str)
    labels["outcome_representative"] = labels["oracle_outcome_class_representative"].map(canonical_action)

    label_cols = KEY_COLS + [
        "raw_policy_class",
        "outcome_policy_class",
        "outcome_representative",
    ]
    if "condition_family" in labels.columns:
        label_cols.append("condition_family")
    labels = labels[label_cols].drop_duplicates(KEY_COLS).copy()

    features_path = find_file(FEATURE_FILE_CANDIDATES, required=False)
    if features_path is None:
        print("[WARN] canonical 2D feature table not found; using fallback feature discovery.")
        features = discover_and_build_feature_table(labels[KEY_COLS].drop_duplicates())
        if features is None:
            raise FileNotFoundError(
                "Could not find or reconstruct a DA-ASA-2D feature table. "
                "Either put one canonical feature table next to this script, or keep the raw DSTA/DA-ASA CSVs "
                "with graph_id+condition keys in this folder tree. Canonical names: "
                + ", ".join(FEATURE_FILE_CANDIDATES)
            )
    else:
        print(f"[LOAD] feature table -> {features_path}")
        features = pd.read_csv(features_path)
        features = standardize_keys(features, "feature table")

    # Keep one feature row per graph_id + condition. If multiple rows exist,
    # aggregate numeric columns by mean and first nonnumeric columns by first.
    if features.duplicated(KEY_COLS).any():
        print("[WARN] feature table has duplicate keys; aggregating to sample-level mean.")
        numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        agg = {c: "mean" for c in numeric_cols if c not in KEY_COLS}
        for c in features.columns:
            if c not in agg and c not in KEY_COLS:
                agg[c] = "first"
        features = features.groupby(KEY_COLS, as_index=False).agg(agg)

    data = labels.merge(features, on=KEY_COLS, how="inner", validate="one_to_one")
    if len(data) == 0:
        raise ValueError("Labels and feature table have zero overlapping graph_id+condition keys.")

    equiv_long = read_csv_first(EQUIV_FILE_CANDIDATES, required=False)
    if equiv_long is not None:
        # 2E equivalence_classes_long is condition-level, not graph-level.
        # It intentionally has no graph_id; map by condition + action_class.
        equiv_long = normalize_columns(equiv_long)
        cond_col = find_col(equiv_long, ["condition", "cond", "condition_name"])
        if cond_col is None:
            raise KeyError(
                "[2E equivalence classes] cannot find condition column. "
                f"Columns={list(equiv_long.columns)}"
            )
        if cond_col != "condition":
            equiv_long = equiv_long.rename(columns={cond_col: "condition"})
        equiv_long["condition"] = equiv_long["condition"].astype(str).str.strip()
        if "action_class" in equiv_long.columns:
            equiv_long["action_class"] = equiv_long["action_class"].map(canonical_action)
        else:
            raise KeyError(
                "[2E equivalence classes] cannot find action_class column. "
                f"Columns={list(equiv_long.columns)}"
            )
        if "outcome_class" not in equiv_long.columns:
            raise KeyError(
                "[2E equivalence classes] cannot find outcome_class column. "
                f"Columns={list(equiv_long.columns)}"
            )

    policy_outcomes = read_csv_first(POLICY_OUTCOME_CANDIDATES, required=False)
    if policy_outcomes is not None:
        policy_outcomes = standardize_keys(policy_outcomes, "policy outcomes")
        if "action_class" in policy_outcomes.columns:
            policy_outcomes["action_class"] = policy_outcomes["action_class"].map(canonical_action)

    action_to_outcome = build_action_to_outcome(equiv_long)
    outcome_to_rep = build_outcome_to_rep(equiv_long, labels)

    feature_sets = build_feature_sets(data)
    if not feature_sets:
        raise ValueError("No safe numeric feature columns found after leakage filtering.")

    with open(OUT_DIR / "da_asa2f_feature_sets.json", "w", encoding="utf-8") as f:
        json.dump({k: {"n": len(v), "columns": v} for k, v in feature_sets.items()}, f, ensure_ascii=False, indent=2)

    targets = {
        "raw_policy_class": "raw_policy_class",
        "outcome_policy_class": "outcome_policy_class",
        "outcome_representative": "outcome_representative",
    }

    summary_rows = []
    pred_frames = []
    rescue_rows = []
    replay_rows = []

    groups = data["graph_id"].astype(str)
    model = make_model()
    dummy = make_dummy()

    for fs_name, cols in feature_sets.items():
        X = data[cols].copy()
        for target_name, target_col in targets.items():
            y = data[target_col].astype(str)
            if y.nunique() < 2:
                print(f"[SKIP] {fs_name}/{target_name}: only one target class")
                continue

            try:
                y_pred = group_cv_predict(X, y, groups, model)
                y_dummy = group_cv_predict(X, y, groups, dummy)
            except Exception as e:
                print(f"[FAIL] {fs_name}/{target_name}: {e}")
                continue

            row = {
                "feature_set": fs_name,
                "target": target_name,
                "n_rows": int(len(y)),
                "n_features": int(len(cols)),
                "n_classes": int(y.nunique()),
                **{f"model_{k}": v for k, v in metric_row(y, y_pred).items()},
                **{f"dummy_{k}": v for k, v in metric_row(y, y_dummy).items()},
            }
            row["macro_f1_lift_vs_dummy"] = row["model_macro_f1"] - row["dummy_macro_f1"]
            row["acc_lift_vs_dummy"] = row["model_accuracy"] - row["dummy_accuracy"]
            summary_rows.append(row)

            pf = data[KEY_COLS + ["raw_policy_class", "outcome_policy_class", "outcome_representative"]].copy()
            if "condition_family" in data.columns:
                pf["condition_family"] = data["condition_family"].values
            pf["feature_set"] = fs_name
            pf["target"] = target_name
            pf["true_label"] = y.values
            pf["pred_label"] = y_pred
            pf["correct"] = (pf["true_label"] == pf["pred_label"])

            # Convert any predicted label to a table-defined outcome class where possible.
            if target_name == "raw_policy_class" or target_name == "outcome_representative":
                pf["pred_action"] = pf["pred_label"].map(canonical_action)
                pf["pred_outcome_class"] = pf.apply(lambda r: outcome_from_action(r, action_to_outcome), axis=1)
            elif target_name == "outcome_policy_class":
                pf["pred_outcome_class"] = pf["pred_label"].astype(str)
                pf["pred_action"] = pf.apply(lambda r: representative_for_outcome(r, outcome_to_rep), axis=1)
            else:
                pf["pred_action"] = None
                pf["pred_outcome_class"] = None

            pf["outcome_class_correct"] = pf["pred_outcome_class"].astype(str) == pf["outcome_policy_class"].astype(str)
            pf["raw_action_correct"] = pf["pred_action"].map(canonical_action).astype(str) == pf["raw_policy_class"].astype(str)
            pred_frames.append(pf)

            raw_wrong = (~pf["raw_action_correct"]).sum()
            outcome_correct_among_raw_wrong = ((~pf["raw_action_correct"]) & pf["outcome_class_correct"]).sum()
            rescue_rows.append({
                "feature_set": fs_name,
                "target": target_name,
                "n": int(len(pf)),
                "raw_action_accuracy_after_mapping": float(pf["raw_action_correct"].mean()),
                "outcome_class_accuracy_after_mapping": float(pf["outcome_class_correct"].mean()),
                "raw_wrong": int(raw_wrong),
                "raw_wrong_but_outcome_correct": int(outcome_correct_among_raw_wrong),
                "rescued_fraction_of_raw_wrong": float(outcome_correct_among_raw_wrong / raw_wrong) if raw_wrong else float("nan"),
            })

            # Optional replay of predicted actions.
            if policy_outcomes is not None and "action_class" in policy_outcomes.columns and pf["pred_action"].notna().any():
                left = pf[KEY_COLS + ["feature_set", "target", "pred_action"]].copy()
                left = left.rename(columns={"pred_action": "action_class"})
                merged = left.merge(policy_outcomes, on=KEY_COLS + ["action_class"], how="left")
                metric_cols = [c for c in ["pred_clean", "R_final"] if c in merged.columns]
                rr = {
                    "feature_set": fs_name,
                    "target": target_name,
                    "n": int(len(merged)),
                    "matched_rate": float(merged[metric_cols].notna().any(axis=1).mean()) if metric_cols else float("nan"),
                }
                for mcol in metric_cols:
                    rr[f"mean_{mcol}"] = float(pd.to_numeric(merged[mcol], errors="coerce").mean())
                replay_rows.append(rr)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["target", "model_macro_f1", "model_accuracy"], ascending=[True, False, False]
    )
    predictions = pd.concat(pred_frames, ignore_index=True, sort=False) if pred_frames else pd.DataFrame()
    rescue = pd.DataFrame(rescue_rows).sort_values(
        ["target", "outcome_class_accuracy_after_mapping", "raw_action_accuracy_after_mapping"],
        ascending=[True, False, False]
    ) if rescue_rows else pd.DataFrame()
    replay = pd.DataFrame(replay_rows).sort_values(
        ["target", "mean_pred_clean"], ascending=[True, False]
    ) if replay_rows else pd.DataFrame()

    summary.to_csv(OUT_DIR / "da_asa2f_cv_summary.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT_DIR / "da_asa2f_predictions_long.csv", index=False, encoding="utf-8-sig")
    rescue.to_csv(OUT_DIR / "da_asa2f_outcome_rescue_summary.csv", index=False, encoding="utf-8-sig")
    replay.to_csv(OUT_DIR / "da_asa2f_replay_summary.csv", index=False, encoding="utf-8-sig")

    # Save confusion matrices for best model per target.
    best_by_target = []
    for target_name, sub in summary.groupby("target"):
        best = sub.sort_values(["model_macro_f1", "model_accuracy"], ascending=False).iloc[0]
        best_by_target.append(best.to_dict())
        pf = predictions[(predictions["feature_set"] == best["feature_set"]) & (predictions["target"] == target_name)]
        labels_sorted = sorted(set(pf["true_label"].astype(str)) | set(pf["pred_label"].astype(str)))
        cm = confusion_matrix(pf["true_label"].astype(str), pf["pred_label"].astype(str), labels=labels_sorted)
        cm_df = pd.DataFrame(cm, index=[f"true__{x}" for x in labels_sorted], columns=[f"pred__{x}" for x in labels_sorted])
        safe_name = re.sub(r"[^A-Za-z0-9_\-]+", "_", target_name)
        cm_df.to_csv(OUT_DIR / f"da_asa2f_best_confusion_{safe_name}.csv", encoding="utf-8-sig")

    # Main verdict logic.
    def best_metric(target: str, metric: str) -> float:
        sub = summary[summary["target"] == target]
        if sub.empty:
            return float("nan")
        return float(sub.sort_values(["model_macro_f1", "model_accuracy"], ascending=False).iloc[0][metric])

    outcome_acc = best_metric("outcome_policy_class", "model_accuracy")
    outcome_f1 = best_metric("outcome_policy_class", "model_macro_f1")
    raw_acc = best_metric("raw_policy_class", "model_accuracy")
    raw_f1 = best_metric("raw_policy_class", "model_macro_f1")

    if math.isfinite(outcome_acc) and outcome_acc >= 0.95 and outcome_f1 >= 0.90:
        verdict = "PASS_OUTCOME_POLICY_CLASSIFIER_STRONG"
    elif math.isfinite(outcome_acc) and outcome_acc >= 0.85 and outcome_f1 >= 0.75:
        verdict = "PASS_OUTCOME_POLICY_CLASSIFIER_LITE"
    elif math.isfinite(outcome_acc) and outcome_acc > 0.50:
        verdict = "PARTIAL_OUTCOME_POLICY_SIGNAL"
    else:
        verdict = "FAIL_OUTCOME_POLICY_CLASSIFIER"

    diagnostics = {
        "stage": "DA-ASA-2F",
        "purpose": "Train DSTA/safe features to predict table-defined OutcomePolicyClass",
        "verdict": verdict,
        "root": str(ROOT),
        "output_dir": str(OUT_DIR),
        "feature_file": str(features_path),
        "n_rows_after_merge": int(len(data)),
        "n_graphs": int(data["graph_id"].nunique()),
        "n_conditions": int(data["condition"].nunique()),
        "feature_sets": {k: len(v) for k, v in feature_sets.items()},
        "target_class_counts": {
            t: data[col].astype(str).value_counts().to_dict()
            for t, col in targets.items()
        },
        "best_by_target": best_by_target,
        "best_outcome_policy_accuracy": outcome_acc,
        "best_outcome_policy_macro_f1": outcome_f1,
        "best_raw_policy_accuracy": raw_acc,
        "best_raw_policy_macro_f1": raw_f1,
    }

    with open(OUT_DIR / "da_asa2f_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 88)
    print("DA-ASA-2F OUTCOME POLICY CLASSIFIER")
    print("=" * 88)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print("\n[CV SUMMARY]")
    if not summary.empty:
        cols = [
            "target", "feature_set", "n_features", "n_classes",
            "model_accuracy", "model_macro_f1", "model_balanced_accuracy",
            "dummy_accuracy", "dummy_macro_f1", "macro_f1_lift_vs_dummy",
        ]
        print(summary[cols].to_string(index=False))
    print(f"\n[OUT] {OUT_DIR}")


if __name__ == "__main__":
    main()
