# Project-Scoped Durable Write Contract v0.1

## Durable target root

```text
project_scoped_icm_store/
```

## Allowed subpaths

```text
memory_units/
operator_memory/
policy_priors/
applicability_gates/
boundary_policies/
reuse_policies/
validity_map/
drift_watch/
evolution_ledger/
negative_transfer_quarantine/
rollback/
replay_manifest/
human_posthoc_reports/
```

## Write envelope

```json
{
  "write_id": "...",
  "decision_id": "...",
  "authorized_by": "AgentOSKernel.ICMEvolutionPolicy",
  "target_scope": "project_scoped_durable",
  "target_type": "MemoryUnit | OperatorMemory | PolicyPrior | ValidityMap | DriftWatch | Quarantine",
  "target_path": "...",
  "payload": {},
  "evidence_refs": [],
  "accept_decision_ref": "...",
  "applicability_gate": {},
  "boundary_policy": {},
  "reuse_policy": {},
  "drift_watch": {},
  "rollback_required": true,
  "production_activation": false
}
```

## Write receipt

```json
{
  "write_id": "...",
  "status": "PASS | FAIL | BLOCKED | ROLLED_BACK",
  "target_path": "...",
  "sha256_before": "...",
  "sha256_after": "...",
  "rollback_pointer": "...",
  "replay_manifest_ref": "...",
  "policy_violations": []
}
```

## Release wording

Allowed:

```text
project-scoped durable ICM evolution write
```

Forbidden:

```text
global production ICM write
unrestricted memory write
self-modifying production policy
```
