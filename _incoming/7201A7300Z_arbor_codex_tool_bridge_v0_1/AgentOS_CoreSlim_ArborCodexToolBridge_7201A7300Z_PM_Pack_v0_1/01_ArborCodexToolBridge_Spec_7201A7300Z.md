# AgentOS CoreSlim 7201A–7300Z
## Arbor-Codex ToolBridge Spec v0.1

## Objective

Strengthen Harness Plane execution by allowing Arbor to use a bounded Codex-like tool bridge for local engineering actions.

This is a Harness capability patch, not a Kernel cognition patch.

## New objects

### CodexToolBridge

```json
{
  "bridge_id": "codex_tool_bridge_local_v0_1",
  "owner": "HarnessPlane",
  "kernel_policy_owner": "AgentOSKernel",
  "mode": "local_bounded_execution",
  "external_mutation_allowed": false
}
```

### ToolCallEnvelope

```json
{
  "tool_call_id": "...",
  "parent_harness_envelope_id": "...",
  "requested_by": "ArborHarness",
  "authorized_by": "AgentOSKernel.HarnessDispatchPolicy",
  "tool_capability": "...",
  "input_artifacts": [],
  "allowed_paths": [],
  "forbidden_paths": [],
  "expected_outputs": [],
  "budget": {
    "max_runtime_seconds": 300,
    "max_output_bytes": 20000000
  },
  "rollback_required": true
}
```

### ToolReceipt

```json
{
  "tool_call_id": "...",
  "status": "PASS | FAIL | BLOCKED",
  "executed_capability": "...",
  "outputs": [],
  "artifacts": [],
  "hashes": [],
  "stdout_summary": "...",
  "stderr_summary": "...",
  "mutation_summary": "...",
  "rollback_pointer": "...",
  "policy_violations": []
}
```

## Allowed tool capabilities

Minimum first set:

```text
READ_ARTIFACT
WRITE_LOCAL_PATCH
RUN_PYTEST
RUN_SCRIPT
PACKAGE_ZIP
GENERATE_HASH_INVENTORY
INSPECT_MANIFEST
RUN_REPLAY
ROLLBACK_LOCAL_WRITE
```

## Forbidden tool capabilities

```text
SEND_EMAIL
WIRE_TRANSFER
EXTERNAL_API_MUTATION
GIT_PUSH
PRODUCTION_DEPLOY
GLOBAL_MEMORY_WRITE
GLOBAL_ICM_WRITE
UNBOUNDED_WEB_ACTION
LEGAL_SIGNATURE
INVESTMENT_COMMITMENT
```

## Required execution chain

```text
AgentOSKernel.HarnessDispatchPolicy
→ HarnessDispatchEnvelope
→ ArborHarness
→ CodexToolBridge.ToolCallEnvelope
→ ToolReceipt
→ ArborHarnessReceipt
→ AgentOSKernel.ResultIngestion
→ CbitEvaluation / NextAction
```

## Acceptance target

A minimal release-preflight run must demonstrate:

1. Arbor receives a typed HarnessDispatchEnvelope.
2. Arbor delegates one or more bounded tool calls to CodexToolBridge.
3. CodexToolBridge executes local tool calls.
4. ToolReceipt is produced.
5. ArborHarnessReceipt incorporates ToolReceipt.
6. AgentOSKernel ingests result and produces final adjudication.
7. No Kernel decision is owned by CodexToolBridge.
