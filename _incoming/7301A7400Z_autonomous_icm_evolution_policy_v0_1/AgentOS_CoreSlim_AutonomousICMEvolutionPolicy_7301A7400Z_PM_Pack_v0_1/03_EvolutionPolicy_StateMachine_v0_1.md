# Autonomous ICM Evolution Policy State Machine v0.1

```text
CandidateRecord
→ EvidenceAccumulation
→ ReuseAttempt
→ ResearchLineClosure
→ ACCEPTDecisionRecord
→ EvolutionEligibilityReview
→ ICMEvolutionPolicyDecision
→ ProjectScopedDurableWriteEnvelope
→ Harness / ToolBridge Execution
→ WriteReceipt
→ ReplayVerification
→ RollbackPointer
→ HumanPosthocReport
→ FutureReuse
```

## Decision states

```text
CANDIDATE_ONLY
VALIDATION_READY
ACCEPT_READY
ELIGIBLE_FOR_PROJECT_SCOPED_EVOLUTION
WRITTEN_PROJECT_SCOPED_DURABLE
QUARANTINED
ROLLED_BACK
EXPIRED
```

## Transition examples

### UPS accepted

```text
UPS_CANDIDATE
→ ACCEPT_READY
→ ELIGIBLE_FOR_PROJECT_SCOPED_EVOLUTION
→ AUTONOMOUS_PROJECT_OPERATORMEMORY_WRITE
→ WRITTEN_PROJECT_SCOPED_DURABLE
```

### Evidence weak but user positive

```text
USER_POSITIVE_CANDIDATE
→ EvidenceAccumulation
→ BLOCK_WRITE_INSUFFICIENT_EVIDENCE
→ CANDIDATE_ONLY
```

### Negative transfer

```text
Candidate
→ ReuseAttempt
→ NegativeTransferDetected
→ AUTONOMOUS_QUARANTINE
→ QUARANTINED
```

### Drift detected

```text
DurableRecord
→ ConstraintDriftEvent
→ EvolutionEligibilityReview
→ AUTONOMOUS_SCOPE_NARROW / ROLLBACK_PREVIOUS_WRITE
```
