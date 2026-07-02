# Codex Task Prompt: AgentOS OM-1A ObjectModeler Mainline Patch

You are working on AgentOS mainline. Implement a minimal, governed ObjectModeler role patch.

## Objective

Add a cross-domain `DomainObjectModeler` role and candidate-only object model evolution flow to AgentOS mainline.

This patch must not rewrite CoreSlim and must not directly mutate accepted registries. It must introduce the role, candidate objects, lifecycle policy, boundary checks, and synthetic validation fixtures.

## Why

Domain object models created at PRD time are incomplete. Real workflows, documents, business materials, decision traces, permission denials, and user corrections reveal missing objects, wrong relations, missing lifecycle states, overloaded object types, stale objects, and permission-surface gaps. AgentOS needs a governed runtime role to detect these mismatches and propose object model candidates.

## Required Implementation

Implement minimal files under the existing AgentOS mainline structure, using current project conventions. Prefer a small patch rather than broad refactor.

### Required concepts

- `DomainObjectModeler`
- `ObjectModelCandidate`
- `ObjectRelationCandidate`
- `ObjectStateCandidate`
- `ObjectLifecycleDeltaCandidate`
- `ObjectPermissionDeltaCandidate`
- `ObjectModelConflictRecord`
- `ObjectModelEvidenceRecord`
- `ObjectModelPatchCandidate`
- `ObjectModelRetentionDecision`

### Required lifecycle

```text
ObservedSignal
→ ObjectModelCandidate
→ EvidenceAttachment
→ ConflictCheck
→ PermissionImpactCheck
→ SchemaImpactCheck
→ RetentionReview
→ ACCEPT / REVISE / REJECT / ARCHIVE candidate
```

All generated object model outputs must remain `PENDING` / candidate-only. No accepted registry mutation.

## Required Boundary Tests

Add tests proving:

1. ObjectModeler role is registered.
2. Candidate object model outputs are PENDING only.
3. Accepted object registry cannot be mutated by ObjectModeler.
4. Harness worker cannot promote object candidates.
5. Every object model candidate requires evidence lineage.
6. Permission impact is required for new permission-relevant objects.
7. Schema impact is required for object model patch candidates.
8. Private/confidential input cannot be promoted to shared/global model without explicit review marker.
9. VC fixture can propose a missing VC object candidate.
10. Non-VC fixture can propose a missing generic domain object candidate.
11. Existing object conflict is detected.
12. Stale / overloaded object can produce revise/archive candidate, not direct deletion.

## Synthetic Fixtures

Include at least two fixtures:

### Fixture A: VC domain

Existing baseline objects:

```text
DealTarget
IndustryDomain
BPClaim
DiligenceTask
CognitiveAsset
```

Workflow signal contains repeated references to:

```text
FounderCredibilitySignal
CustomerBudgetSignal
CoInvestorInformationBoundary
```

Expected output: pending object candidates with evidence and permission impact notes.

### Fixture B: non-VC domain

Use education or manufacturing.

Example missing objects:

```text
LearningBottleneck
StudentMisconceptionCluster
```

or

```text
ProcessWindow
YieldLossPattern
SupplierConstraint
```

Expected output: pending object candidates with evidence and conflict check.

## Hard Boundaries

Do not:

```text
Rewrite CoreSlim
Implement full ontology engine
Mutate accepted registries
Promote candidates automatically
Use live web
Use real private data
Use real target/company data
Make investment recommendations
Weaken permission boundaries
Allow Harness to accept object model changes
```

## Required Return Files

Package and return the following files:

```text
object_modeler_role_spec.md
object_model_candidate_schema.json
object_model_lifecycle_policy.json
object_model_patch_candidate_contract.json
object_modeler_boundary_tests.py
object_modeler_fixture_vc.json
object_modeler_fixture_cross_domain.json
object_modeler_validation_report.md
object_modeler_verdict.json
pytest_output.txt
return_files_manifest_AgentOS_OM1A_v0_1.json
HASH_INVENTORY.csv
```

If project layout requires different paths, include the equivalent files and list actual paths in the manifest.

## Acceptance

Return PASS only if all boundary tests pass and all generated object model changes remain candidate-only.
