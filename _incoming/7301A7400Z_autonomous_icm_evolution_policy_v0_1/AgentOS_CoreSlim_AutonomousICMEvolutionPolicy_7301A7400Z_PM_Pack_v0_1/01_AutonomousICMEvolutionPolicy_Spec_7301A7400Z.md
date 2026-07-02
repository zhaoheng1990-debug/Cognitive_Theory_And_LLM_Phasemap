# AgentOS CoreSlim 7301A–7400Z
## Autonomous ICM Evolution Policy Spec v0.1

## Objective

Enable AgentOS Kernel to autonomously write project-scoped durable ICM / MemoryUnit / OperatorMemory updates when a candidate line reaches ACCEPT-level closure under policy.

This is release-preflight work before staged beta.

## Target chain

```text
ResearchLineClosure
→ ACCEPTDecisionRecord
→ ICMEvolutionEligibilityReview
→ AutonomousICMEvolutionPolicyDecision
→ ProjectScopedDurableWriteEnvelope
→ MemoryUnit / OperatorMemory / PolicyPrior write
→ ValidityMap / DriftWatch / VersionLedger update
→ Replay / rollback
→ HumanPosthocReport
```

## Key example: UPS

If UtilityPolicySelector / UPS reaches ACCEPT:

```text
UPS_ACCEPTDecisionRecord
→ OperatorMemoryWriteEligibility
→ UtilityPolicySelectorOperatorMemoryRecord
→ ProjectScopedDurable OperatorMemory write
→ ReusePolicy update
→ ApplicabilityGate update
→ DriftWatch update
```

The system must not wait for manual preauthorization for ordinary project-scoped self-evolution.

## Scope levels

### E0: Candidate only

No durable write.  
Use when evidence is insufficient.

### E1: Controlled sandbox write

Local test-only write.  
Already validated in 6501.

### E2: Project-scoped durable ICM write

Allowed in this line under Kernel policy.

Writes to:

```text
project_scoped_icm/
project_scoped_memory_units/
project_scoped_operator_memory/
validity_map/
drift_watch/
evolution_ledger/
negative_transfer_quarantine/
```

### E3: Global / production ICM write

Still forbidden in this line.

Requires future product/human authority policy.

## Allowed autonomous write targets

```text
MemoryUnit
OperatorMemory
PolicyPrior
ApplicabilityGate
BoundaryPolicy
ReusePolicy
ValidityMap
DriftWatch
EvolutionLedger
NegativeTransferQuarantine
```

## Forbidden autonomous write targets

```text
global user memory
production ICM
theory baseline source files
external services
financial/legal/investment/medical actions
native ActionRuntime side effects
git push / email / deployment
```

## Autonomous write eligibility

A candidate may be autonomously written if all are true:

```text
1. ACCEPTDecisionRecord exists
2. evidence trail is replayable
3. scope is project-bounded
4. negative transfer risk is low or bounded by gate
5. future Cbit gain is positive
6. applicability gate is explicit
7. boundary policy is explicit
8. rollback is available
9. drift watch is attached
10. no contradictory unresolved blocker
```

## Policy decision actions

```text
NO_WRITE_KEEP_CANDIDATE
AUTONOMOUS_PROJECT_MEMORYUNIT_WRITE
AUTONOMOUS_PROJECT_OPERATORMEMORY_WRITE
AUTONOMOUS_POLICY_PRIOR_WRITE
AUTONOMOUS_VALIDITY_UPDATE
AUTONOMOUS_SCOPE_NARROW
AUTONOMOUS_QUARANTINE
REQUEST_HUMAN_SCOPE_ESCALATION
BLOCK_WRITE_INSUFFICIENT_EVIDENCE
ROLLBACK_PREVIOUS_WRITE
```

## Required receipts

Every autonomous write must produce:

```text
ProjectScopedDurableWriteReceipt
EvolutionLedgerEntry
ReplayManifestEntry
RollbackPointer
HumanPosthocReportEntry
```

## Human role

Human is not removed.

Human role changes from:

```text
preauthorize every ordinary self-evolution write
```

to:

```text
posthoc review, rollback, scope escalation approval, high-risk boundary authority
```
