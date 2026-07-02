# Autonomous ICM Evolution Invariants v0.1

## Kernel ownership

Only AgentOSKernel.ICMEvolutionPolicy may issue final autonomous durable write decisions.

LLM API may advise.  
Harness may execute.  
CodexToolBridge may run local tools.  
Arbor may execute envelopes.

None may own final evolution decisions.

## Project-scoped durability

Autonomous writes in this line are durable within the project-scoped ICM store.

They are not global production writes.

## Evidence-first invariant

No autonomous write from vibes, recency, user preference, or model confidence alone.

Required:

```text
ACCEPTDecisionRecord
replayable evidence
bounded scope
Cbit gain
negative transfer audit
rollback pointer
```

## User feedback invariant

User feedback is a signal, not authorization.

Positive user feedback can increase UserValueSignal.  
It cannot bypass evidence or policy.

## Research-line closure invariant

A research line closure to ACCEPT should be able to trigger autonomous ICM evolution if policy criteria are met.

This is essential for AgentOS self-evolution.

## UPS example invariant

If UPS is ACCEPT and evidence is replayable, the system should produce:

```text
UtilityPolicySelectorOperatorMemoryRecord
ApplicabilityGate
BoundaryPolicy
ReusePolicy
DriftWatch
EvolutionLedgerEntry
```

without requiring manual preauthorization.

## Rollback invariant

Every write must be rollbackable.

Rollback is a product requirement, not a postscript.

## Anti-additive invariant

Do not add new gate families.

Evolve the existing ICMMetabolismPolicy into autonomous project-scoped ICM evolution.
