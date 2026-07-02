# Arbor-Codex ToolBridge Test Matrix v0.1

| Test | Expected |
|---|---|
| Arbor delegates READ_ARTIFACT | ToolReceipt PASS |
| Arbor delegates RUN_PYTEST | ToolReceipt PASS with pytest summary |
| Arbor delegates PACKAGE_ZIP | ToolReceipt PASS with hash |
| Arbor delegates WRITE_LOCAL_PATCH | ToolReceipt PASS with rollback |
| Rollback local patch | original hash restored |
| Unauthorized capability | BLOCKED |
| External mutation attempt | BLOCKED |
| Missing Kernel envelope | BLOCKED |
| ArborHarnessReceipt includes ToolReceipt | PASS |
| AgentOS ResultIngestion accepts ToolReceipt | PASS |
| CodexToolBridge final decision ownership | false |
