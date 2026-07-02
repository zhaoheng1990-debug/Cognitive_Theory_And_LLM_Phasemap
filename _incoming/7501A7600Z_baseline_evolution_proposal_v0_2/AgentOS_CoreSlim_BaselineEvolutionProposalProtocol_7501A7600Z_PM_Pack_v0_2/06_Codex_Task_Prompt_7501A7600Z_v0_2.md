# Codex Task Prompt: Baseline Evolution Proposal Protocol 7501A–7600Z v0.2

You are executing AgentOS CoreSlim Baseline Evolution Proposal Protocol.

## Starting evidence

7301A–7400Z passed AutonomousICMEvolutionPolicy:

```text
UPS ACCEPT → project-scoped durable OperatorMemory write
weak evidence → no-write
negative transfer → quarantine
rollback/replay → PASS
Kernel owns evolution decision
```

## Objective

Add a mechanism for AgentOS to periodically or event-triggeredly propose official/global/production baseline updates when empirical validation is sufficient.

This prevents canonical baselines from becoming rigid while preserving signed authority for actual S3/S4 writes.

## Important distinction

AgentOS may autonomously write S1/S2 project-scoped memory.

AgentOS may autonomously propose S3/S4 updates.

AgentOS may not autonomously apply S3/S4 updates.

## Implement / produce

1. Define BaselineUpdateProposal schema.
2. Define ProposalEligibilityPolicy.
3. Define PeriodicBaselineReview state machine.
4. Define ProposalQueue behavior.
5. Define HumanReviewPacket for proposals.
6. Add UPS test case:
   - UPS ACCEPT + project OperatorMemory write
   - generates S3 theory baseline proposal
   - does not write official baseline
7. Add negative tests:
   - insufficient empirical evidence -> no proposal
   - high negative transfer -> quarantine/defer
   - conflict with ACCEPT baseline -> conflict review
8. Confirm SignedBaselinePromotionAuthorization is still required for actual baseline write.
9. Generate PM-ready verdict.

## Required return files

Return all files listed in `08_Return_Files_Manifest_7501A7600Z_v0_2.json`.
