# Periodic Baseline Review State Machine v0.2

```text
ProjectScopedDurableICMStore
→ PeriodicReviewTrigger
→ MaturedCandidateScan
→ ProposalEligibilityReview
→ BaselineUpdateProposal
→ ProposalQueue
→ HumanReviewPacket
→ SignedBaselinePromotionAuthorization
→ ControlledBaselinePromotion
```

## Trigger types

```text
ON_RELEASE_READINESS_REVIEW
ON_PROJECT_MILESTONE
ON_N_ACCEPTED_CLOSURES
ON_EVIDENCE_SATURATION
ON_CONFLICT_DRIFT_DETECTED
MANUAL_USER_REQUEST
```

## States

```text
NOT_REVIEWED
MATURED_PROJECT_SCOPED_RECORD
ELIGIBLE_FOR_PROPOSAL
PROPOSAL_ENQUEUED
HUMAN_REVIEW_READY
APPROVED_FOR_PROMOTION
REJECTED
DEFERRED
PROMOTED
```

## MVP note

This does not require background automation in the current ChatGPT window.

It is implemented as event/milestone review logic inside AgentOS.
