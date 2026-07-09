# Provider-Backed Runtime Cognition Audit

This note records the P0-P3 audit baseline for AgentOS CoreSlim provider-backed runtime cognition.

## Principle

AgentOS runtime remains the cognitive subject. It owns goals, boundaries, meta-rules, evidence organization, conflict handling, replay, rollback, and final candidate state.

Providers support those runtime abilities by returning structured semantic judgment receipts. They do not replace runtime authority, authorize promotion, write accepted state, or make final Kernel decisions.

## P0-P3 Closure Target

The provider-backed layer is considered structurally closed for P0-P3 when:

- each runtime cognitive responsibility maps to at least one provider-backed semantic operation;
- provider receipts carry source/input hashes, rationale, confidence, unsupported parts, risk flags, Cbit estimate, freshness, scope, and runtime responsibility mapping;
- provider receipts cannot set `accepted_state_written`;
- provider receipts cannot declare final accepted/promoted candidate state;
- provider receipts cannot set `scope.promotion_allowed = true`;
- Kernel/runtime deterministic boundaries still own replay, rollback, state transitions, safety hard stops, capability envelopes, provider receipt existence checks, and accepted writes.

## Responsibility Coverage

| Runtime responsibility | Provider-backed support operations |
| --- | --- |
| Goal and constraint management | `TemporalSRO`, `UtilityPolicySelection`, `NextIterationActionSelection` |
| Boundary governance | `TemporalSRO`, `StructuralRouting`, `UtilityPolicySelection`, `ValidityDriftWatch` |
| Meta-rule application | `TemporalSRO`, `StructuralRouting`, `OperatorFiberRanking`, `UtilityPolicySelection` |
| Evidence organization | `EvidenceRelevance`, `EvidenceClaimSupport`, `EntityEventExtraction`, `DomainScopeSynthesis` |
| Conflict handling | `EvidenceClaimSupport`, `ValidityDriftWatch`, `DomainScopeSynthesis`, `UtilityPolicySelection` |
| Replay and rollback reasoning | `OperatorMemoryEquivalenceReview`, `ValidityDriftWatch`, `RetentionCandidateEvaluation` |
| Final candidate-state judgment | `UtilityPolicySelection`, `CbitGainEstimation`, `RetentionCandidateEvaluation`, `NextIterationActionSelection` |

## Current Audit Mechanism

`ProviderRequirementPolicy.audit_runtime_cognition_support()` checks whether every runtime cognitive responsibility is backed by registered provider semantic operations and whether deterministic runtime authority remains present.

`SemanticJudgmentReceipt.validate()` checks that provider receipts support runtime cognition without taking over final authority.

## P4 Deferred Validation

P4 should validate live multi-provider and multi-Harness cycles:

- cross-provider receipt consistency;
- provider failure and degraded-mode behavior;
- conflict resolution across contradictory receipts;
- replay of a full runtime cycle from stored receipts;
- human/PM review of candidate-state promotion packets.
