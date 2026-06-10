# -*- coding: utf-8 -*-
"""
DA-ASA-2K: Condition Family Expansion Builder / Audit

Purpose
-------
After DA-ASA-2J.1, CleanDenseDSTA -> LatentPolicyFactor is strong in known-condition
graph-heldout setting, but leave-one-condition/family generalization remains weak.

2K does NOT train another classifier first. It audits condition-family coverage and builds
an expansion plan / candidate prompt table so later 2K.1 can test TransferableLPF.

Outputs
-------
da_asa2k_outputs/
  da_asa2k_diagnostics.json
  da_asa2k_existing_condition_coverage.csv
  da_asa2k_family_expansion_plan.csv
  da_asa2k_candidate_conditions.csv
  da_asa2k_candidate_prompts.csv
  da_asa2k_next_experiment_seed.md

Run
---
python da_asa2k_condition_family_expansion_builder.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "da_asa2k_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Target condition count per family for unseen-condition/family generalization.
TARGET_CONDITIONS_PER_FAMILY = 8
GRAPHS_PER_CONDITION = 12

# Known files from prior stages. The script uses whichever exist.
INPUT_CANDIDATES = {
    "2j1_training": [
        "da_asa2j1_outputs/da_asa2j1_training_table_used.csv",
        "da_asa2j1_training_table_used.csv",
    ],
    "2j_lpf_table": [
        "da_asa2j_outputs/da_asa2j_latent_policy_factor_table.csv",
        "da_asa2j_latent_policy_factor_table.csv",
    ],
    "2f_predictions": [
        "da_asa2f_outputs/da_asa2f_predictions_long.csv",
        "da_asa2f_predictions_long.csv",
    ],
    "2e_equiv": [
        "da_asa2e_outputs/da_asa2e_equivalence_classes_long.csv",
        "da_asa2e_equivalence_classes_long.csv",
    ],
}

# A richer family taxonomy for expansion. Existing family names are mapped into these.
FAMILY_RULES = [
    ("stable_protect", ["stable", "redundant", "clean", "weak_distractor"]),
    ("competition", ["competition", "equal_evidence", "source_claim", "direct"]),
    ("closure_update", ["closure_update", "closure_temporal", "temporal", "authority", "canonical_update", "closure_authority"]),
    ("override_exception", ["override", "exception", "negation", "rule_override", "exception_binding", "negation_rewrite"]),
    ("hallucination_like", ["hallucination", "fabricated", "false", "unsupported"]),
]

# Candidate condition templates. These are data-generation specs, not model prompts only.
# They are intentionally concise and parameterized so you can expand manually or by script.
CANDIDATE_SPECS = {
    "stable_protect": [
        ("stable_clean_fact", "Clean known fact; correct answer already dominant; intervention should protect/no-op."),
        ("stable_redundant_context", "Correct answer plus redundant but non-conflicting context; no intervention expected."),
        ("stable_minor_noise", "Correct answer with irrelevant distractor; no repair needed."),
        ("stable_paraphrase", "Same fact expressed through paraphrase; stable trajectory should remain clean."),
        ("stable_multi_sentence", "Longer clean context with multiple supporting clauses; protect clean attractor."),
        ("stable_numeric_fact", "Simple numeric/date fact with clear correct answer; no-op preferred."),
        ("stable_entity_attribute", "Entity-attribute fact with strong prior support; protect."),
        ("stable_definition", "Definition-style prompt with unambiguous answer; protect."),
    ],
    "competition": [
        ("competition_equal_evidence", "Two plausible alternatives receive balanced evidence; order/evidence policy may matter."),
        ("competition_direct_conflict", "Direct conflict between two candidate answers; needs resolving competition."),
        ("competition_source_claim", "Source-like statement conflicts with model prior; detect whether to trust or override."),
        ("competition_late_disambiguation", "Early text supports one answer, late clause disambiguates another."),
        ("competition_role_context", "Role/context changes answer among close semantic alternatives."),
        ("competition_temporal_context", "Temporal cue switches correct answer between old/new state."),
        ("competition_category_boundary", "Two categories overlap; correct answer depends on boundary condition."),
        ("competition_alias_conflict", "Alias or name collision produces two plausible entities."),
    ],
    "closure_update": [
        ("closure_canonical_update", "A known relation must be updated by explicit local context."),
        ("closure_temporal_update", "Temporal update changes previously true relation."),
        ("closure_authority_update", "Authority/source in prompt supplies an update to relation closure."),
        ("closure_definition_update", "A local definition changes downstream inference."),
        ("closure_rule_update", "A rule statement changes how examples should be classified."),
        ("closure_contextual_binding", "Local binding maps symbol/entity to new meaning."),
        ("closure_multi_hop_update", "Update must propagate across a short relation chain."),
        ("closure_counterfactual_update", "Counterfactual premise creates local closure distinct from world prior."),
    ],
    "override_exception": [
        ("override_rule_exception", "General rule has explicit exception; policy should override default closure."),
        ("override_three_stage", "Initial default, conflicting update, final exception/override sequence."),
        ("override_negation", "Negation rewrites expected relation and must escape default attractor."),
        ("override_scope_exception", "Exception applies only inside a bounded scope."),
        ("override_priority_rule", "Priority ordering among rules determines final answer."),
        ("override_nested_exception", "Exception to an exception; tests staged commitment."),
        ("override_local_dictionary", "Prompt defines local dictionary overriding common meaning."),
        ("override_adversarial_distractor", "Strong distractor suggests default but explicit exception wins."),
    ],
    "hallucination_like": [
        ("hallucination_unknown_entity", "Prompt asks about entity with insufficient support; intervention should avoid fabrication."),
        ("hallucination_false_premise", "Question contains false premise; model should resist closure into false answer."),
        ("hallucination_unsupported_citation", "Citation-like cue is unsupported; avoid confident invented details."),
        ("hallucination_impossible_relation", "Relation is semantically tempting but impossible/invalid."),
        ("hallucination_name_collision", "Similar entity names invite false association."),
        ("hallucination_sparse_context", "Sparse prompt with high ambiguity; should abstain or protect."),
        ("hallucination_over_specific", "Request demands specific detail not present in context."),
        ("hallucination_confabulated_chain", "Multi-hop chain can be closed incorrectly from weak cues."),
    ],
}

PROMPT_TEMPLATES = {
    "stable_protect": [
        "Question: {question}\nContext: {context}\nAnswer with the single best option: C or E.",
        "Given the clean context below, decide whether C or E is correct.\n{context}\nQuestion: {question}",
    ],
    "competition": [
        "Two candidate answers are plausible. Use only the final context to decide.\nContext: {context}\nQuestion: {question}\nOptions: C={c_ans}; E={e_ans}.",
        "Resolve the conflict carefully.\n{context}\nQuestion: {question}\nChoose C or E.",
    ],
    "closure_update": [
        "Local rule/update: {context}\nApply this update when answering.\nQuestion: {question}\nChoose C or E.",
        "In this mini-world, the following relation is true: {context}\nQuestion: {question}\nAnswer C or E.",
    ],
    "override_exception": [
        "General rule and exception: {context}\nQuestion: {question}\nChoose C or E according to the exception if applicable.",
        "Follow the rule hierarchy in the context.\n{context}\nQuestion: {question}\nAnswer C or E.",
    ],
    "hallucination_like": [
        "Use only supported information. If the premise is unsupported, choose the safer option.\nContext: {context}\nQuestion: {question}\nOptions: C={c_ans}; E={e_ans}.",
        "Avoid inventing facts.\n{context}\nQuestion: {question}\nChoose C or E.",
    ],
}


def find_existing(paths: List[str]) -> Path | None:
    for rel in paths:
        p = ROOT / rel
        if p.exists():
            return p
    # recursive fallback by basename
    for rel in paths:
        hits = list(ROOT.rglob(Path(rel).name))
        if hits:
            return hits[0]
    return None


def load_table(paths: List[str]) -> Tuple[pd.DataFrame | None, str | None]:
    p = find_existing(paths)
    if p is None:
        return None, None
    return pd.read_csv(p), str(p)


def infer_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def canonical_family(condition: str) -> str:
    s = str(condition).strip().lower()
    for fam, keys in FAMILY_RULES:
        if any(k in s for k in keys):
            return fam
    if "closure" in s:
        return "closure_update"
    if "stable" in s or "weak" in s:
        return "stable_protect"
    return "other"


def build_existing_coverage() -> Tuple[pd.DataFrame, Dict]:
    loaded = {}
    primary_df = None
    primary_path = None
    for name, candidates in INPUT_CANDIDATES.items():
        df, path = load_table(candidates)
        loaded[name] = {"found": df is not None, "path": path}
        if primary_df is None and df is not None:
            cond_col = infer_col(df, ["condition", "cond", "condition_name"])
            if cond_col is not None:
                primary_df = df
                primary_path = path

    if primary_df is None:
        raise FileNotFoundError("Could not find any prior table with a condition column. Run from python_script root or place 2E/2J outputs nearby.")

    cond_col = infer_col(primary_df, ["condition", "cond", "condition_name"])
    graph_col = infer_col(primary_df, ["graph_id", "graph", "gid"])
    action_col = infer_col(primary_df, ["action_class", "action", "policy", "outcome_representative"])
    lpf_col = infer_col(primary_df, ["latent_policy_factor", "lpf", "pred_lpf", "true_lpf"])

    base = primary_df.copy()
    base["condition"] = base[cond_col].astype(str).str.strip()
    base["condition_family"] = base["condition"].map(canonical_family)

    group_cols = ["condition_family", "condition"]
    rows = []
    for keys, sub in base.groupby(group_cols, dropna=False):
        fam, cond = keys
        rows.append({
            "condition_family": fam,
            "condition": cond,
            "rows": int(len(sub)),
            "n_graphs": int(sub[graph_col].nunique()) if graph_col else None,
            "n_actions": int(sub[action_col].nunique()) if action_col else None,
            "n_lpf": int(sub[lpf_col].nunique()) if lpf_col else None,
        })
    cov = pd.DataFrame(rows).sort_values(["condition_family", "condition"])

    meta = {
        "loaded_inputs": loaded,
        "primary_table_path": primary_path,
        "primary_condition_col": cond_col,
        "primary_graph_col": graph_col,
        "primary_action_col": action_col,
        "primary_lpf_col": lpf_col,
        "n_primary_rows": int(len(primary_df)),
        "n_existing_conditions": int(cov["condition"].nunique()),
        "n_existing_families": int(cov["condition_family"].nunique()),
    }
    return cov, meta


def build_expansion_plan(cov: pd.DataFrame) -> pd.DataFrame:
    existing = cov.groupby("condition_family")["condition"].nunique().to_dict()
    rows = []
    for fam in CANDIDATE_SPECS.keys():
        n_exist = int(existing.get(fam, 0))
        n_needed = max(0, TARGET_CONDITIONS_PER_FAMILY - n_exist)
        rows.append({
            "condition_family": fam,
            "existing_conditions": n_exist,
            "target_conditions": TARGET_CONDITIONS_PER_FAMILY,
            "additional_needed": n_needed,
            "target_graphs_per_condition": GRAPHS_PER_CONDITION,
            "new_rows_needed_if_full": n_needed * GRAPHS_PER_CONDITION,
            "priority": "HIGH" if n_needed > 0 else "OK",
        })
    return pd.DataFrame(rows)


def build_candidate_conditions(cov: pd.DataFrame) -> pd.DataFrame:
    existing_conditions = set(cov["condition"].astype(str).str.lower())
    rows = []
    for fam, specs in CANDIDATE_SPECS.items():
        for cond, desc in specs:
            rows.append({
                "condition_family": fam,
                "condition": cond,
                "description": desc,
                "already_present_name_match": cond.lower() in existing_conditions,
                "recommended_n_graphs": GRAPHS_PER_CONDITION,
                "expected_policy_object": "latent_policy_factor",
                "status": "candidate" if cond.lower() not in existing_conditions else "already_present",
            })
    return pd.DataFrame(rows)


def build_candidate_prompts(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in candidates.iterrows():
        fam = r["condition_family"]
        cond = r["condition"]
        desc = r["description"]
        templates = PROMPT_TEMPLATES.get(fam, PROMPT_TEMPLATES["stable_protect"])
        for i in range(GRAPHS_PER_CONDITION):
            template = templates[i % len(templates)]
            graph_id = f"2k_{cond}_g{i+1:02d}"
            # Placeholders are deliberately explicit, so user can replace them or feed to a generator.
            context = f"[{cond}] {desc} Example variant {i+1}. Replace with concrete C/E pair."
            question = f"For {cond} variant {i+1}, which answer follows from the context?"
            c_ans = "clean/correct candidate"
            e_ans = "error/distractor candidate"
            prompt = template.format(context=context, question=question, c_ans=c_ans, e_ans=e_ans)
            rows.append({
                "graph_id": graph_id,
                "condition_family": fam,
                "condition": cond,
                "variant_id": i + 1,
                "prompt": prompt,
                "C_answer_placeholder": c_ans,
                "E_answer_placeholder": e_ans,
                "gold_answer_placeholder": "C_or_E_TO_FILL",
                "notes": "Replace placeholders with concrete controlled sample before model run.",
            })
    return pd.DataFrame(rows)


def write_seed(meta: Dict, plan: pd.DataFrame) -> str:
    total_new = int(plan["new_rows_needed_if_full"].sum())
    seed = f"""# DA-ASA-2K 新窗口 Seed: Condition Family Expansion for Transferable LPF

## 目标

当前 2J/2J.1 已经证明：

```text
PolicyResults -> EffectVector -> LatentPolicyFactor -> LPFPolicyReplay   PASS-Strong
CleanDenseDSTA -> LatentPolicyFactor                                    PASS in known-condition graph-heldout
```

但 2J.1 同时表明：

```text
leave-one-condition / leave-one-family 仍未闭合
```

因此 2K 不继续调分类器，而是扩展 condition family，使每个机制族拥有多个 condition，以便真正测试：

```text
DenseDSTA -> TransferableLPF
```

---

## 当前路径配置

工作根目录：

```text
{ROOT}
```

2K 输出目录：

```text
{OUT_DIR}
```

本次识别到的主输入表：

```text
{meta.get('primary_table_path')}
```

关键输入存在性：

```json
{json.dumps(meta.get('loaded_inputs'), ensure_ascii=False, indent=2)}
```

---

## 已生成 2K 文件

```text
da_asa2k_outputs/da_asa2k_diagnostics.json
da_asa2k_outputs/da_asa2k_existing_condition_coverage.csv
da_asa2k_outputs/da_asa2k_family_expansion_plan.csv
da_asa2k_outputs/da_asa2k_candidate_conditions.csv
da_asa2k_outputs/da_asa2k_candidate_prompts.csv
da_asa2k_outputs/da_asa2k_next_experiment_seed.md
```

---

## 下一步实验建议：DA-ASA-2K.1

使用 `da_asa2k_candidate_prompts.csv` 作为模板，替换占位符后生成真实 controlled C/E 样本。

然后按原 DA-ASA 流程跑：

```text
1. baseline scores
2. DSTA signed/transport/attractor features
3. policy_results table
4. effect vectors
5. LPF clustering / LPF label assignment
6. DenseDSTA -> LPF CV
```

2K.1 的核心判定：

```text
如果 leave-one-condition macro-F1 明显高于 2J.1 的 0.5397，且 leave-one-family macro-F1 明显高于 0.3824，说明 family expansion 提升 TransferableLPF。
```

---

## 本次 expansion 预计新增样本

```text
new_rows_needed_if_full = {total_new}
```

优先补齐 HIGH family。
"""
    return seed


def main() -> None:
    cov, meta = build_existing_coverage()
    plan = build_expansion_plan(cov)
    candidates = build_candidate_conditions(cov)
    prompts = build_candidate_prompts(candidates[candidates["status"] == "candidate"].copy())

    diagnostics = {
        "stage": "DA-ASA-2K",
        "purpose": "Condition-family expansion after 2J.1 known-condition-only DenseDSTA->LPF pass",
        "verdict": "PLAN_READY_CONDITION_FAMILY_EXPANSION",
        "root": str(ROOT),
        "output_dir": str(OUT_DIR),
        "target_conditions_per_family": TARGET_CONDITIONS_PER_FAMILY,
        "graphs_per_condition": GRAPHS_PER_CONDITION,
        **meta,
        "existing_family_condition_counts": cov.groupby("condition_family")["condition"].nunique().to_dict(),
        "planned_new_conditions": int(plan["additional_needed"].sum()),
        "planned_new_rows_if_full": int(plan["new_rows_needed_if_full"].sum()),
        "candidate_prompt_rows": int(len(prompts)),
        "theory_update": {
            "2J_1_freeze": "DenseDSTA -> LPF is strong only in known-condition graph-heldout setting.",
            "2K_goal": "Expand condition families to test transferable LPF generalization.",
            "not_a_classifier_tuning_stage": True,
        },
    }

    cov.to_csv(OUT_DIR / "da_asa2k_existing_condition_coverage.csv", index=False, encoding="utf-8-sig")
    plan.to_csv(OUT_DIR / "da_asa2k_family_expansion_plan.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(OUT_DIR / "da_asa2k_candidate_conditions.csv", index=False, encoding="utf-8-sig")
    prompts.to_csv(OUT_DIR / "da_asa2k_candidate_prompts.csv", index=False, encoding="utf-8-sig")

    seed = write_seed(meta, plan)
    (OUT_DIR / "da_asa2k_next_experiment_seed.md").write_text(seed, encoding="utf-8")
    (OUT_DIR / "da_asa2k_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print("DA-ASA-2K CONDITION FAMILY EXPANSION BUILDER")
    print("=" * 80)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print("\n[EXPANSION PLAN]")
    print(plan.to_string(index=False))
    print(f"\n[OUT] {OUT_DIR}")


if __name__ == "__main__":
    main()
