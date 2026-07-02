# Rollback Replay Report 7301A7400Z

Verdict: PASS_ROLLBACK_REPLAY

UPS replay retained write:

```json
{
  "write_id": "evo-ups_accept_operator_memory-autonomous_project_operatormemory_write",
  "replay_status": "PASS",
  "current_sha256": "8e8cc3ec681c1eb845c9b3d4f32b12a0c029ae230727c71ad9c3417883a46242",
  "expected_sha256": "8e8cc3ec681c1eb845c9b3d4f32b12a0c029ae230727c71ad9c3417883a46242"
}
```

Rollback demo replay-before-rollback:

```json
{
  "write_id": "evo-ups_rollback_demo-autonomous_project_operatormemory_write",
  "replay_status": "PASS",
  "current_sha256": "8e8cc3ec681c1eb845c9b3d4f32b12a0c029ae230727c71ad9c3417883a46242",
  "expected_sha256": "8e8cc3ec681c1eb845c9b3d4f32b12a0c029ae230727c71ad9c3417883a46242"
}
```

Rollback result:

```json
{
  "write_id": "evo-ups_rollback_demo-autonomous_project_operatormemory_write",
  "status": "ROLLED_BACK",
  "target_path": "C:\\Users\\ZH\\Documents\\Logos-AgentOS\\outputs\\7301A7400Z_autonomous_icm_evolution_policy\\project_scoped_icm_store\\operator_memory\\ups_rollback_demo.json",
  "restored_sha256": "",
  "matches_before_hash": true
}
```
