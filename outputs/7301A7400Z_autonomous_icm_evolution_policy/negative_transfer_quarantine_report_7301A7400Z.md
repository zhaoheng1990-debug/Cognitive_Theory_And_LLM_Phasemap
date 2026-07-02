# Negative Transfer Quarantine Report 7301A7400Z

Verdict: PASS_NEGATIVE_TRANSFER_AUTONOMOUS_QUARANTINE

Review decision: `AUTONOMOUS_QUARANTINE`

Quarantine receipt:

```json
{
  "write_id": "evo-negative_transfer_case-autonomous_quarantine",
  "status": "PASS",
  "target_path": "C:\\Users\\ZH\\Documents\\Logos-AgentOS\\outputs\\7301A7400Z_autonomous_icm_evolution_policy\\project_scoped_icm_store\\negative_transfer_quarantine\\negative_transfer_case.json",
  "sha256_before": "",
  "sha256_after": "18a100a913fba868ed77ea93d8eb9a10986a8274b24d24d500753c10f8b1272e",
  "rollback_pointer": "C:\\Users\\ZH\\Documents\\Logos-AgentOS\\outputs\\7301A7400Z_autonomous_icm_evolution_policy\\project_scoped_icm_store\\rollback\\ba9bec9321a046a2.rollback.json",
  "replay_manifest_ref": "C:\\Users\\ZH\\Documents\\Logos-AgentOS\\outputs\\7301A7400Z_autonomous_icm_evolution_policy\\project_scoped_icm_store\\replay_manifest\\ba9bec9321a046a2.replay.json",
  "policy_violations": [],
  "authorized_by": "AgentOSKernel.ICMEvolutionPolicy",
  "target_scope": "project_scoped_durable",
  "production_activation": false,
  "receipt_hash": "a88a9b5061b14b6dd39fad5ccdf0a0110c4863133218058569e55dfafc4fb344"
}
```

The negative-transfer candidate was written only to `negative_transfer_quarantine/`; it was not promoted to MemoryUnit, OperatorMemory, Policy, AcceptedEvidence, or baseline.
