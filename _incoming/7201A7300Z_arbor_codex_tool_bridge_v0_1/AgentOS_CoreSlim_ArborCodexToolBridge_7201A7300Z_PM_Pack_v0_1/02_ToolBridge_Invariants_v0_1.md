# Arbor-Codex ToolBridge Invariants v0.1

## Kernel policy invariant

CodexToolBridge cannot self-authorize.

Every tool call must reference an AgentOSKernel-authorized HarnessDispatchEnvelope.

## Harness plane invariant

CodexToolBridge belongs to Harness Plane.

It is not a Role Agent.  
It is not an SRO.  
It is not a Planner.  
It is not an ICMMetabolismPolicy owner.

## Local-only invariant

Default mode is local bounded execution.

External mutation is forbidden.

## Receipt invariant

No tool call is considered complete without a ToolReceipt.

## Rollback invariant

Any write-capable tool call must have rollback metadata.

## Replay invariant

ToolReceipt must be replay-indexable and hash-addressed.

## Anti-additive invariant

Do not add a large tool marketplace.

Implement the minimal bridge needed for Arbor to execute local dev-RC tasks:

```text
read
write local patch
run test
package
hash
replay
rollback
```
