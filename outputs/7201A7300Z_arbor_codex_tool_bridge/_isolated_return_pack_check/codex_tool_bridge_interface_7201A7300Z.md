# CodexToolBridge Interface 7201A7300Z

Verdict: INTERFACE_DEFINED

## Object

`CodexToolBridge` is a Harness Plane executor used by Arbor under `AgentOSKernel.HarnessDispatchPolicy` authorization.

```json
{
  "bridge_id": "codex_tool_bridge_local_v0_1",
  "owner": "HarnessPlane",
  "kernel_policy_owner": "AgentOSKernel",
  "mode": "local_bounded_execution",
  "external_mutation_allowed": false,
  "codex_tool_bridge_final_decision_owner": false
}
```

## Call Shape

`execute(tool_call_envelope, harness_dispatch_envelope) -> ToolReceipt`

The bridge fails closed when the Kernel-authorized dispatch envelope is missing, mismatched, or lacks the requested capability.

## Minimal Capabilities

- `READ_ARTIFACT`
- `WRITE_LOCAL_PATCH`
- `RUN_PYTEST`
- `RUN_SCRIPT`
- `PACKAGE_ZIP`
- `GENERATE_HASH_INVENTORY`
- `INSPECT_MANIFEST`
- `RUN_REPLAY`
- `ROLLBACK_LOCAL_WRITE`

## Forbidden Capabilities

- `SEND_EMAIL`
- `WIRE_TRANSFER`
- `EXTERNAL_API_MUTATION`
- `GIT_PUSH`
- `PRODUCTION_DEPLOY`
- `GLOBAL_MEMORY_WRITE`
- `GLOBAL_ICM_WRITE`
- `UNBOUNDED_WEB_ACTION`
- `LEGAL_SIGNATURE`
- `INVESTMENT_COMMITMENT`

## Ownership Boundary

CodexToolBridge does not own SRO, NextAction, ICMMetabolismPolicyDecision, Productization verdicts, AGI precursor verdicts, AcceptedEvidence, MemoryUnit, OperatorMemory, Policy, or baseline writeback. It emits `ToolReceipt` only; AgentOSKernel owns result ingestion and final adjudication.

## Implementation Location

- `agentos_core_slim_v0/agentos_kernel/codex_tool_bridge.py`
- `agentos_core_slim_v0/tests/test_codex_tool_bridge.py`
