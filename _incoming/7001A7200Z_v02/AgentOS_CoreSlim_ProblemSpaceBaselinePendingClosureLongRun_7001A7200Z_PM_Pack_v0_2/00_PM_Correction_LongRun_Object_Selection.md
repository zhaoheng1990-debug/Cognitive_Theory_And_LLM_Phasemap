# PM Correction: Long-Run Object Selection

## Verdict

Use the ProblemSpaceBaseline pending ledger as the long-run soak object.

This replaces the generic Dev-RC stability-only soak with a higher-Cbit long-run:

\[
\boxed{
ProductizationLongRunDevRCSoak
\rightarrow
ProblemSpaceBaselinePendingClosureLongRun
}
\]

## Why this is the right object

A pure replay loop tests stability, but low Cbit.  
Closing the ProblemSpaceBaseline pending ledger tests:

- long-horizon task vision retention;
- SRO and problem-space continuity;
- candidate lifecycle;
- ICM metabolism policy;
- controlled MemoryUnit / OperatorMemory write;
- negative route preservation;
- replay consistency;
- report readiness.

## Critical boundary

This run must not claim that all pending items become ACCEPT.

The goal is:

\[
\boxed{
PendingItem
\rightarrow
ClosureActionDecision
}
\]

Allowed decisions:

- ACCEPT_READY_WITH_EVIDENCE
- PROMOTE_TO_VALIDATION_SEED
- KEEP_PENDING_WITH_NEXT_TEST
- EXPIRE_OR_DEPRECATE
- QUARANTINE_NEGATIVE_TRANSFER
- MERGE_WITH_EXISTING
- ROUTE_TO_SEPARATE_EXPERIMENT
- HUMAN_REPORT_ONLY

## Anti-additive constraint

Do not add new theory branches.  
Do not add new modules.  
Do not create more boundary-proof cases.

The long-run must reduce the active pending surface.
