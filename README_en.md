# HPM-RT1 v1.1 English Guide

HPM-RT1 v1.0 is positioned as a **black-box LLM external runtime warning and audit layer**.

It runs outside the model and uses visible input, visible output, user-supplied evidence, and page/API metadata to provide hallucination-risk warnings, constraint-frame identification, evidence-boundary audits, advisory actions, trace logging, and PDF reports.

It is not a model-internal probe. It does not read or claim access to:

```text
hidden states
logits / token probabilities
attention
true chain-of-thought
model-internal confidence
provider-side retrieval traces
sampling parameters or internal reasoning process
```

The v1.0 engineering boundary is:

```text
black-box external warning
advisory only
no automatic answer rewriting
no final decision takeover
no model weight modification
no proof of what the model internally knows
```

HPM-RT1 Beta1 is an external advisory guard. It monitors hallucination risk, constraint-field identification, evidence needs, advisory actions, cross-model / cross-parameter window alignment, calibration writeback, and runtime trace collection.

It does not do the following:

```text
no automatic answer rewriting
no final decision takeover
no model weight modification
```

Default output directory:

```text
C:\Users\ZH\Desktop\AGI\outputs\hpm_rt1_beta_output
```

## 0. Added in v0.5

The main upgrade in v0.5 is bilingual documentation. All major instruction files now have Chinese and English versions for internal research, external beta testing, cross-team review, and future open-source packaging.

```text
README_zh.md / README_en.md
MANIFEST_zh.md / MANIFEST_en.md
HPM_RT1_Beta_Module_Protocol_zh.md / HPM_RT1_Beta_Module_Protocol_en.md
HPM_RT1_Testing_and_Output_Interpretation_zh.md / HPM_RT1_Testing_and_Output_Interpretation_en.md
HPM_RT1_Testset_Construction_Guide_zh.md / HPM_RT1_Testset_Construction_Guide_en.md
HPM_RT1_W_Manifold_Mapping_Gap_Note_zh.md / HPM_RT1_W_Manifold_Mapping_Gap_Note_en.md
```

Legacy unsuffixed documentation files are still included. New testers should prefer the `_zh` or `_en` files.

## 1. Recommended reading order

Chinese testers:

```text
README_zh.md
HPM_RT1_Beta_Module_Protocol_zh.md
HPM_RT1_Testset_Construction_Guide_zh.md
HPM_RT1_Testing_and_Output_Interpretation_zh.md
HPM_RT1_W_Manifold_Mapping_Gap_Note_zh.md
```

English testers:

```text
README_en.md
HPM_RT1_Beta_Module_Protocol_en.md
HPM_RT1_Testset_Construction_Guide_en.md
HPM_RT1_Testing_and_Output_Interpretation_en.md
HPM_RT1_W_Manifold_Mapping_Gap_Note_en.md
```

## 2. Quick install and run

From the package root:

```bash
pip install -e .
```

Run the basic sample:

```bash
python -m hpm_rt1_beta.cli --jsonl examples/sample_cases.jsonl --output-dir "C:\Users\ZH\Desktop\AGI\outputs\hpm_rt1_beta_output"
```

Run the teaching case-study test set:

```bash
python -m hpm_rt1_beta.cli --jsonl examples/hpm_rt1_testset_case_study.jsonl --output-dir "C:\Users\ZH\Desktop\AGI\outputs\hpm_rt1_v05_case_study_output"
```

Run cross-model / cross-parameter window alignment diagnostics:

```bash
python tools/hpm_rt1_window_alignment_diagnostic.py --eval-jsonl examples/sample_alignment_eval.jsonl --output-dir "C:\Users\ZH\Desktop\AGI\outputs\hpm_rt1_alignment_output"
```

Write the diagnostic profile back into RT1 active config:

```bash
python tools/hpm_rt1_writeback_calibration.py --profile "C:\Users\ZH\Desktop\AGI\outputs\hpm_rt1_alignment_output\hpm_rt1_calibration_profile.json" --install-dir . --dry-run

python tools/hpm_rt1_writeback_calibration.py --profile "C:\Users\ZH\Desktop\AGI\outputs\hpm_rt1_alignment_output\hpm_rt1_calibration_profile.json" --install-dir .
```

## 3. Package structure

```text
hpm_rt1_beta/core.py          main guard
hpm_rt1_beta/sro.py           structural resolution / constraint-field identification
hpm_rt1_beta/risk.py          risk scoring
hpm_rt1_beta/advisory.py      advisory action policy
hpm_rt1_beta/adapters.py      cross-model adapter interface
hpm_rt1_beta/calibration.py   runtime calibration loader
hpm_rt1_beta/logger.py        JSONL trace logger
tools/hpm_rt1_window_alignment_diagnostic.py   cross-model window diagnostics
tools/hpm_rt1_writeback_calibration.py         calibration writeback script
examples/hpm_rt1_testset_case_study.jsonl      teaching test set
examples/hpm_rt1_testset_template.csv          external tester template
```

## 4. Beta1 boundary

RT1 v0.5 remains Beta1 only:

```text
advisory only
no answer rewrite
no model weight update
no active HPM-12 repair
```

HPM-12 candidate repair must remain a shadow / candidate arm until independent validation passes.

## 5. Cross-model / cross-parameter principle

The module can be mounted across models, but thresholds must not be assumed to transfer directly.

```text
The wrapper interface can be cross-model.
Calibration / thresholds / risk weights must be diagnosed by model_family × parameter_scale × task_family × window_id.
```

This is why v0.2 added window alignment diagnostics and calibration writeback.

## 6. W-manifold mapping gap

Since v0.4, the package formally records that RT1 still lacks full W-Manifold Mapping / ManifoldCoverageRisk implementation. v0.5 keeps this boundary.

Currently supported:

```text
MCR shadow annotation
test-set construction guidance
reserved interface for future HPM-WM1 experiments
```

Not implemented:

```text
hidden-state W mapping
full TopK/VIM probe
active MCR policy
```

## 7. Output files

RT1 CLI typically writes:

```text
hpm_rt1_beta_trace.jsonl
```

Window diagnostics typically write:

```text
hpm_rt1_group_window_metrics.csv
hpm_rt1_window_alignment_summary.csv
hpm_rt1_calibration_profile.json
hpm_rt1_alignment_report.md
```

The writeback script updates:

```text
hpm_rt1_beta/config/rt1_active_config.json
```

## 8. Minimal conclusion

v0.5 does not claim that HPM eliminates hallucinations. It provides a testable, calibratable, traceable, cross-model comparable advisory guard that can collect standardized data for HPM-12 independent validation and future W-Manifold Mapping experiments.


## v0.6 Sample Testsets

Added runnable sample testsets and templates: `HPM_RT1_Sample_Testsets_zh.md` / `HPM_RT1_Sample_Testsets_en.md`, plus `examples/hpm_rt1_external_testset_min32.jsonl`, `examples/hpm12_independent_testset_180.jsonl`, `examples/hpm_rt1_alignment_eval_min32.jsonl`, and `examples/hpm12_delayed_feedback_template.csv`.


---

## v0.7 Core Update: HPM = Hallucination Risk Governance

v0.7 formally reframes HPM: it is not a universal hallucination corrector, but a hallucination risk governance module. It adds `AnswerabilityGate`: when W coverage, evidence support, constraint-field clarity, temporal validity, CGA / constraint alignment, or source-prior clarity are insufficient, RT1 should recommend `retrieve / ask / refuse / defer / uncertainty mark` rather than trying to repair unsupported closure into a correct answer.

New documents:

```text
HPM_Core_Theorem_Update_zh.md
HPM_Core_Theorem_Update_en.md
HPM_Core_Theorem_Update.md
```

New runtime output field:

```text
answerability_gate
```

Boundary unchanged: Beta1 advisory only; no answer rewriting, no model-weight update, no active HPM-12 candidate repair.

---

## v0.8 Added: ContextFrameGate

v0.8 adds `ContextFrameGate`. The main chain is now:

```text
Prompt → ContextFrameGate → SRO → AnswerabilityGate → ClosureMonitor → ConstraintAlignmentAudit → PolicyAction
```

New output fields:

```text
context_frame_gate
risk_scores.context_frame_risk
```

Core principle:

```text
Resolve context first, then audit hallucination.
Hallucination is constraint violation under the active context frame.
```

New documentation:

```text
HPM_ContextFrameGate_Update_zh.md
HPM_ContextFrameGate_Update_en.md
```

New sample set:

```text
examples/hpm_rt1_context_frame_gate_cases.jsonl
examples/hpm_rt1_context_frame_gate_cases.csv
```

Run:

```bash
python -m hpm_rt1_beta.cli --jsonl examples/hpm_rt1_context_frame_gate_cases.jsonl
```

## New in v0.9

Added Codex engineering guide: `HPM_RT1_Codex_Engineering_Guide_en.md`. It covers API server, frontend dashboard, software architecture, core module responsibilities, debugging workflow, integration roadmap, and acceptance criteria.

---

## v1.0 Positioning Freeze: Black-box LLM External Warning and Audit Layer

v1.0 formally freezes the current engineering position:

```text
HPM-RT1 = black-box LLM external runtime warning and audit layer
```

Evidence coordinate:

```text
Internal Project Evidence + Controlled Runtime Wrapper Evidence
```

Accepted object:

```text
visible input / visible output / evidence boundary / page or API metadata
  -> ContextFrameGate
  -> AnswerabilityGate
  -> RiskScores
  -> AdvisoryAction
  -> runtime warning / completed-turn audit log
```

Preserved boundary:

```text
no model-internal state access
no logits, attention, hidden layers, or true reasoning-chain access
no proof of model-internal knowledge state
no hard blocking of web-page model generation
not a replacement for model-service-side governance
```

v1.0 can provide external black-box risk governance: pre-send prompt checks, near-real-time DOM-based output warnings, completed-turn gateway audits, and JSONL/PDF logging. Model context adaptation covers GPT, Gemini, Kimi, Doubao, Claude, DeepSeek, Qwen, Llama, Gemma, and local/other models.
If deployed later inside a model service or API middleware, it can become a stronger streaming governance and pre-commit gate; that is a deeper deployment layer, not a capability claimed by the current external black-box version.

---

## v1.1 Added: Prompt-type Weighted Audit Confidence

v1.1 changes `overall_audit_confidence`: it is no longer a plain average across confidence dimensions. HPM-RT1 now chooses weights from `task_family`, `constraint_field`, and `reality_mode`.

This is audit-judgment confidence: higher values mean HPM-RT1 is more confident that its audit conclusion is supported. It is not the answer-risk value. Answer risk should be read from `risk_scores.hallucination_risk`, `answerability_gate`, and `advisory_action`.

```text
temporal / expired-memory prompts: higher evidence and freshness weight
source conflict / low-evidence traps: higher evidence and constraint-alignment weight
counterfactual / roleplay / simulation prompts: higher context-frame weight
urgent action / high-stakes prompts: higher action-safety and advisory-action weight
creative open-space prompts: lower evidence weight, higher context/action weight
```

Each output now also includes:

```text
audit_confidence_profile
audit_confidence_weights
audit_confidence_threshold
audit_confidence_gate
```
