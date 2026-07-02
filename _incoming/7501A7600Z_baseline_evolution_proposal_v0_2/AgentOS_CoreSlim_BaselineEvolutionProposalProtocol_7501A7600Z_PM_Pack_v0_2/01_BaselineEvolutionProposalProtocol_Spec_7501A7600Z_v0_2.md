# AgentOS CoreSlim 7501A–7600Z
## Baseline Evolution Proposal Protocol Spec v0.2

## Objective

After 7301 closed project-scoped autonomous ICM evolution, define how AgentOS turns empirically validated project-scoped durable learning into official/global baseline update proposals.

This prevents canonical baselines from freezing while preserving signed authority for actual S3/S4 writes.

## Core principle

\[
\boxed{
S2\ autonomous\ write;\ S3/S4\ autonomous\ proposal;\ signed\ promotion.
}
\]

## Full chain

```text
EmpiricalValidation
→ ResearchLineClosure
→ ACCEPTDecisionRecord
→ ProjectScopedDurableICMUpdate
→ BaselineUpdateProposalEligibility
→ BaselineUpdateProposal
→ ProposalQueue
→ HumanReviewPacket
→ SignedBaselinePromotionAuthorization
→ ControlledBaselinePromotion
```

## Write/proposal boundary

### Direct autonomous write allowed

```text
S1 project-scoped candidate memory
S2 project-scoped durable ICM / MemoryUnit / OperatorMemory
```

### Autonomous proposal allowed

```text
S3 official theory baseline
S4 global production ICM / product policy candidate
```

### Direct autonomous write forbidden

```text
S3 official theory baseline
S4 global production ICM / product policy
```

## Trigger modes

### Event-driven trigger

Triggered when:

```text
ResearchLineClosure == ACCEPT
AND project-scoped durable ICM write succeeds
AND reuse evidence exists
AND negative transfer audit passes
```

### Periodic / milestone trigger

AgentOS reviews project-scoped durable ICM at:

```text
release-readiness review
project milestone
N accepted research closures
evidence saturation threshold
manual user request
```

## Proposal eligibility

A candidate may enter BaselineUpdateProposalQueue if:

```text
1. ACCEPTDecisionRecord exists
2. empirical validation evidence is replayable
3. project-scoped durable ICM/Memory/Operator record exists
4. future cross-project reuse value is positive
5. negative transfer risk is bounded
6. contradiction/conflict scan is complete
7. proposed baseline scope is explicit
8. rollback/promotion route exists
9. human-readable diff summary exists
```

## Required output

The run must demonstrate:

```text
UPS ACCEPT + project OperatorMemory write
→ BaselineUpdateProposal for official theory baseline
→ ProposalQueue entry
→ HumanReviewPacket
→ no direct official baseline write
```
