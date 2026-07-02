# Arbor ToolBridge Integration Report 7201A7300Z

Verdict: ARBOR_INTEGRATION_EXAMPLE_PRESENT

## Required Chain

AgentOSKernel.HarnessDispatchPolicy -> HarnessDispatchEnvelope -> ArborHarness -> CodexToolBridge.ToolCallEnvelope -> ToolReceipt -> ArborHarnessReceipt -> AgentOSKernel.ResultIngestion.

## Example HarnessDispatchEnvelope

```json
{
  "harness_dispatch_envelope_id": "hde-001",
  "authorized_by": "AgentOSKernel.HarnessDispatchPolicy",
  "requested_harness": "ArborHarness",
  "authorized_capabilities": ["READ_ARTIFACT", "RUN_PYTEST", "PACKAGE_ZIP"],
  "kernel_final_decision_owner": true
}
```

## Example Arbor Delegation

Arbor receives the Kernel dispatch envelope and emits a ToolCallEnvelope whose `parent_harness_envelope_id` matches `hde-001`. The bridge validates the parent envelope and returns a ToolReceipt. Arbor may aggregate multiple ToolReceipts into an ArborHarnessReceipt, but it cannot reinterpret them as Kernel decisions.

## ArborHarnessReceipt Sketch

```json
{
  "harness": "ArborHarness",
  "parent_harness_envelope_id": "hde-001",
  "tool_receipts": ["<ToolReceipt>"],
  "result_ingestion_target": "AgentOSKernel.ResultIngestion",
  "kernel_decision_owner_preserved": true
}
```
