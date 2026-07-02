# Proposal Eligibility Policy v0.2

## Inputs

```text
ACCEPTDecisionRecord
ProjectScopedDurableICMRecord
EmpiricalValidationReport
ReuseEvidence
NegativeTransferAudit
ConflictScan
CbitGainReport
DriftWatch
```

## Eligibility rule

A candidate should be proposed when:

```text
evidence_strength >= threshold
AND empirical_replayability == true
AND project_scoped_durable_write_exists == true
AND cross_project_reuse_value > 0
AND negative_transfer_risk != high_unbounded
AND conflict_scan != unresolved_blocker
```

## Non-eligible cases

Do not propose if:

```text
only user praise exists
only LLM/model judgment exists
no empirical validation exists
no replay trail exists
no project-scoped durable write exists
scope is not bounded
contradiction with ACCEPT baseline exists
negative transfer risk is unbounded
```

## Proposal queue actions

```text
ENQUEUE_BASELINE_UPDATE_PROPOSAL
MERGE_WITH_EXISTING_PROPOSAL
DEFER_INSUFFICIENT_EVIDENCE
REJECT_INSUFFICIENT_GENERALITY
ROUTE_TO_EXPERIMENT
REQUEST_CONFLICT_REVIEW
```

## Key distinction

A proposal is allowed to be autonomous.

A S3/S4 write is not.
