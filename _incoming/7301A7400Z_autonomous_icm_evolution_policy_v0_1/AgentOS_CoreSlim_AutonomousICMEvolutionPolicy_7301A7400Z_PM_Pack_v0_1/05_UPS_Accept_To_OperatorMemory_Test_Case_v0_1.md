# UPS ACCEPT → OperatorMemory Test Case v0.1

## Purpose

Test whether an ACCEPT-level research-line closure can autonomously enter project-scoped OperatorMemory.

## Input

```json
{
  "research_line": "UtilityPolicySelector",
  "status": "ACCEPT_READY_WITH_EVIDENCE",
  "evidence_refs": [
    "SR_series",
    "UPS_validation",
    "reuse_gain_records",
    "negative_transfer_audit"
  ],
  "scope": "AgentOS project runtime policy selection",
  "negative_transfer_risk": "bounded",
  "future_cbit_gain": "positive"
}
```

## Expected Kernel decision

```text
AUTONOMOUS_PROJECT_OPERATORMEMORY_WRITE
```

## Expected write payload

```json
{
  "operator_id": "UtilityPolicySelectorOperator",
  "operator_family": "PolicySelection",
  "purpose": "select lifecycle / next-action / retention policy when structural resolution alone is insufficient",
  "applicability_gate": {
    "requires": [
      "ranked_compatibility_fiber",
      "utility_signal",
      "risk_signal",
      "cbit_gain_signal"
    ]
  },
  "boundary_policy": {
    "do_not_use_when": [
      "evidence_missing",
      "high_negative_transfer",
      "external_action_required_without_authority"
    ]
  },
  "reuse_policy": {
    "mode": "project_scoped",
    "requires_replay": true,
    "drift_watch": true
  }
}
```

## Expected reports

```text
ups_operator_memory_write_report_7301A7400Z.md
evolution_ledger_report_7301A7400Z.md
rollback_replay_report_7301A7400Z.md
human_posthoc_report_7301A7400Z.md
```
