# PM Correction: Insert Arbor-Codex ToolBridge Before Release Readiness

## Verdict

The user's judgment is accepted.

Current Harness execution layer, especially Arbor, has insufficient practical tool-calling capability for staged beta readiness.

Therefore, before Productization Release Readiness Review, insert:

\[
oxed{
ArborCodexToolBridge
}
\]

## Critical boundary

This does not move Codex into AgentOS Kernel.

Correct architecture:

```text
AgentOS Kernel
→ HarnessDispatchEnvelope
→ Arbor Harness
→ CodexToolBridge
→ bounded tool execution
→ ToolReceipt
→ ArborHarnessReceipt
→ AgentOS ResultIngestion
```

## What this patch adds

A controlled tool-calling bridge in the Harness Plane.

It allows Arbor to request Codex-like execution capabilities under a typed envelope:

- file read/write inside allowed workspace;
- script execution;
- pytest execution;
- packaging;
- manifest/hash generation;
- local replay;
- artifact inspection;
- bounded patch application.

## What this patch does not add

- Codex does not own SRO.
- Codex does not own NextAction.
- Codex does not own ICM write decisions.
- Codex does not own theory judgment.
- Codex does not own Productization or AGI Precursor verdicts.
- Codex does not perform unbounded external mutation.
- Codex does not bypass AgentOS Kernel policy.

## Why this is release-relevant

A governance kernel without a capable execution harness becomes a review system rather than an operating system.

A capable harness without Kernel governance becomes uncontrolled automation.

The productized architecture needs both:

\[
oxed{
AgentOS\ governs;\ Arbor/CodexToolBridge\ executes.
}
\]
