# Result Ingestion ToolReceipt Report 7201A7300Z

Verdict: PASS_TOOL_RECEIPT_INGESTION_COMPATIBLE

AgentOS ResultIngestion consumes ToolReceipt as evidence, not as final policy. The receipt carries:

- `status`: PASS / FAIL / BLOCKED;
- `executed_capability`;
- outputs and artifacts;
- hashes;
- stdout/stderr summaries;
- mutation summary;
- rollback pointer;
- policy violations;
- bridge owner and Kernel policy owner;
- `codex_tool_bridge_final_decision_owner = false`.

This supports CbitEvaluation / NextAction downstream without moving ownership to CodexToolBridge.
