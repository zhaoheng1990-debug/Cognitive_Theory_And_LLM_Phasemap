# Kernel Authorized Dispatch Report 7201A7300Z

Verdict: PASS_KERNEL_AUTHORIZED_DISPATCH_REQUIRED

The bridge requires `HarnessDispatchEnvelope.authorized_by == AgentOSKernel.HarnessDispatchPolicy`. A missing dispatch envelope, a parent envelope mismatch, or a capability absent from `authorized_capabilities` returns `ToolReceipt.status = BLOCKED` before execution.

Test coverage:

- Missing Kernel envelope -> BLOCKED.
- Unauthorized capability -> BLOCKED.
- Authorized READ_ARTIFACT -> PASS.
- Authorized local write with rollback metadata -> PASS.

CodexToolBridge never self-authorizes and never converts a ToolReceipt into a final Kernel route.
