# Periodic Baseline Review State Machine 7501A7600Z

Verdict: PASS_PERIODIC_REVIEW_STATE_MACHINE_PRESENT

```text
ProjectScopedDurableICMStore
-> PeriodicReviewTrigger
-> MaturedCandidateScan
-> ProposalEligibilityReview
-> BaselineUpdateProposal
-> ProposalQueue
-> HumanReviewPacket
-> SignedBaselinePromotionAuthorization
-> ControlledBaselinePromotion
```

Trigger types:

- ON_RELEASE_READINESS_REVIEW
- ON_PROJECT_MILESTONE
- ON_N_ACCEPTED_CLOSURES
- ON_EVIDENCE_SATURATION
- ON_CONFLICT_DRIFT_DETECTED
- MANUAL_USER_REQUEST

States:

- NOT_REVIEWED
- MATURED_PROJECT_SCOPED_RECORD
- ELIGIBLE_FOR_PROPOSAL
- PROPOSAL_ENQUEUED
- HUMAN_REVIEW_READY
- APPROVED_FOR_PROMOTION
- REJECTED
- DEFERRED
- PROMOTED

This line stops at HUMAN_REVIEW_READY. It does not enter PROMOTED.
