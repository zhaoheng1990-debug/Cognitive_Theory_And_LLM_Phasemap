# AgentOS Mainline Patch Seed v0.1

## Patch Title

**Object Modeler Role / Domain Ontology Evolution Layer**

## PM Verdict

ACCEPT AS MAINLINE PATCH CANDIDATE.

This patch should enter AgentOS mainline, not only VC-AgentOS. The VC-AgentOS work exposed a general architecture gap: current domain object modeling is performed largely at PRD / baseline time, but real business operation will inevitably reveal object mismatches, missing lifecycle states, wrong relations, hidden workflow entities, access-control edge cases, and obsolete objects.

AgentOS therefore needs a cross-domain role and runtime mechanism for continuous business object modeling.

```text
Business workflow traces
+ user documents
+ project materials
+ operational events
+ decision records
+ Harness outputs
+ human corrections
→ ObjectModelCandidate / ObjectRelationCandidate / ObjectLifecycleDelta
→ Evidence + conflict review
→ Retention / Schema Review Gate
→ accepted domain object model update candidate
```

The role is inspired by enterprise ontology thinking: a production domain system should not only store documents or tables, but continuously model real operational objects, object relations, object states, actions, permissions, and lifecycle transitions.

## Core Correction

Previous domain-adapter flow:

```text
PRD stage object schema
→ static domain registry
→ workflow implementation
```

Corrected AgentOS mainline flow:

```text
Initial PRD object schema
→ runtime object observation
→ object mismatch detection
→ candidate object model evolution
→ evidence / governance review
→ schema / registry update candidate
→ domain runtime improvement
```

The object model is not a one-time design artifact. It is a living cognitive asset.

## New AgentOS Role

### Role Name

`DomainObjectModeler`

Chinese name: `对象建模师`

### Role Placement

AgentOS Governance / Cognitive Runtime Layer.

It is not a Harness worker and not a simple schema generator. It is a domain-cognition role responsible for discovering, revising, and governing domain object models from business operation evidence.

### Scope

The role applies to all DomainAgentOS lines:

```text
VC-AgentOS
Education-AgentOS
Enterprise-AgentOS
Research-AgentOS
Manufacturing-AgentOS
Healthcare-AgentOS
Legal / compliance domains
AgentOS internal domain modeling
```

## Role Responsibility

`DomainObjectModeler` is responsible for:

1. Discovering missing domain objects from real workflows and documents.
2. Detecting mismatch between baseline schema and actual business operation.
3. Proposing new object types, relations, states, actions, permissions, and lifecycle transitions.
4. Mapping unstructured business materials into candidate object structures.
5. Comparing new object candidates with existing domain registries.
6. Producing object model delta candidates, never direct registry mutation.
7. Working with EvidenceLedger, RetentionGate, SchemaEngineer, PermissionGovernor, and HumanReview when required.
8. Detecting stale or overloaded objects that should be split, merged, deprecated, or archived.
9. Maintaining domain object model evolution as a first-class cognitive asset.

## Non-Responsibilities

`DomainObjectModeler` must not:

```text
Directly mutate accepted object registry
Directly change database schema
Directly bypass Schema Review Gate
Directly promote a candidate object to ACCEPT
Directly decide business actions
Directly weaken permission boundaries
Directly absorb confidential documents into global domain model without access review
```

## Differentiation from Existing Roles

| Role | Responsibility | Difference from DomainObjectModeler |
|---|---|---|
| SchemaEngineer | Implements accepted schema / registry structure | Encodes accepted object model; does not discover business objects from operation |
| EvidenceLedgerGovernor | Tracks evidence and source quality | Judges support strength; does not own object ontology evolution |
| RetentionGate | Decides retain / revise / archive | Gates retention; does not generate object candidates |
| Harness Worker | Parses documents / runs extraction tasks | Executes extraction; does not govern object model lifecycle |
| Domain PM / Product Owner | Defines initial PRD objects | Baseline design; not continuous runtime object learning |
| PermissionGovernor | Controls access | Reviews permission consequences of proposed object model changes |

## New Object Types

Add the following cross-domain objects to AgentOS mainline registry as candidate-capable types.

```text
DomainObjectModel
DomainObjectType
DomainObjectInstancePattern
DomainObjectRelation
DomainObjectState
DomainObjectAction
DomainObjectLifecycle
DomainObjectPermissionSurface
ObjectModelCandidate
ObjectRelationCandidate
ObjectStateCandidate
ObjectLifecycleDeltaCandidate
ObjectPermissionDeltaCandidate
ObjectModelConflictRecord
ObjectModelEvidenceRecord
ObjectModelDecayRecord
ObjectModelRetentionDecision
ObjectModelPatchCandidate
```

## Object Model Candidate Lifecycle

```text
ObservedSignal
→ ObjectModelCandidate
→ EvidenceAttachment
→ ConflictCheck
→ PermissionImpactCheck
→ SchemaImpactCheck
→ RetentionReview
→ HumanReview if required
→ ACCEPT / REVISE / REJECT / ARCHIVE candidate
```

The default status for all runtime-discovered objects is `PENDING`, never `ACCEPT`.

## Evidence Inputs

The Object Modeler may use:

```text
Business workflows
User-uploaded documents
Project files
BP / deck / diligence materials
Operational logs
Audit events
Human corrections
External expert notes
Portfolio feedback
Harness extraction outputs
Permission denial logs
Object access failure logs
Repeated user-created ad hoc fields
Repeated manual workarounds
```

All evidence must retain source, timestamp, access level, confidence, and validity scope.

## Object Mismatch Patterns

The Object Modeler should detect:

```text
Missing object type
Overloaded object type
Wrong object boundary
Missing relation
Wrong relation direction
Missing lifecycle state
Missing action type
Missing permission surface
Domain-specific object not captured by generic schema
Private / confidential object leaking into shared model
Stale object no longer used
Object split needed
Object merge needed
Object relation conflict
Workflow event not represented in object model
```

## VC-AgentOS Example

Baseline VC objects may include:

```text
DealTarget
IndustryDomain
Subsector
CognitiveAsset
BPClaim
DiligenceTask
PortfolioCompany
ExternalPartner
```

But real VC operation may reveal missing objects such as:

```text
SyndicatePosition
CompetitiveDealConflict
FounderCredibilitySignal
CustomerBudgetSignal
TechnicalTransferRisk
GovernmentGuidanceSignal
StrategicLPConstraint
CoInvestorInformationBoundary
FollowOnReservePressure
```

These should not be added manually each time by PM. AgentOS should observe repeated workflow evidence, propose candidates, attach evidence, and route them through governance.

## Cross-Domain Example

Education domain may reveal missing objects:

```text
LearningBottleneck
ConceptMasteryTrajectory
TeacherIntervention
StudentMisconceptionCluster
CurriculumConstraint
AssessmentSignal
```

Manufacturing domain may reveal:

```text
ProcessWindow
YieldLossPattern
SupplierConstraint
EquipmentDowntimeEvent
QualityEscapeRisk
```

Therefore, the role belongs in AgentOS mainline.

## Mainline Architecture Delta

Add a Domain Ontology Evolution Layer:

```text
Domain Runtime Traces
+ Domain Documents
+ Harness Extraction Outputs
+ Human Corrections
        ↓
DomainObjectModeler
        ↓
ObjectModelCandidate Ledger
        ↓
Evidence / Conflict / Permission / Schema Impact Review
        ↓
RetentionGate / HumanGate when required
        ↓
Domain Registry Patch Candidate
```

## Required Boundaries

```text
No direct schema mutation
No direct accepted registry write
No permission weakening without PermissionGovernor
No confidential-to-shared leakage
No auto-promotion from BP / private materials
No replacement of PRD baseline; only candidate evolution
No Harness autonomous ontology acceptance
```

## Suggested Mainline Build

### AgentOS OM-1A: ObjectModeler Role Registry + Candidate Object Lifecycle

Goal:

Implement the role registry entry, object candidate lifecycle, and non-mutating object model patch candidate flow.

Deliverables:

```text
object_modeler_role_spec.md
object_model_candidate_schema.json
object_model_lifecycle_policy.json
object_model_patch_candidate_contract.json
object_modeler_boundary_tests.py
```

### AgentOS OM-1B: Runtime Object Mismatch Detection Fixtures

Goal:

Use synthetic domain workflow traces to detect missing objects, overloaded objects, missing lifecycle states, and permission surface gaps.

### AgentOS OM-1C: Domain Adapter Integration

Goal:

Integrate ObjectModeler with VC-AgentOS as first domain fixture, while preserving cross-domain generality.

## Acceptance Criteria

PASS only if:

```text
DomainObjectModeler exists as governed AgentOS role
Object candidates are generated as PENDING only
Accepted domain registry is not mutated directly
Evidence lineage is required for every candidate
Permission impact is explicitly assessed
Schema impact is explicitly assessed
Confidential/private source constraints are preserved
Harness cannot promote object candidates
Human/Retention gate is required for accepted model change
Synthetic VC and non-VC fixtures both pass
```

FAIL if:

```text
ObjectModeler becomes a free-form schema generator
Object candidates are accepted automatically
Private BP-derived objects enter global model without review
Harness can mutate accepted object registry
Permission surface is ignored
No conflict with existing objects is checked
Role is implemented as VC-only rather than AgentOS-mainline
```

## PM Priority

High.

This patch directly supports AgentOS self-evolution and domain adaptability. It is not product bloat. It is a core precursor to domain runtime learning: AgentOS must not only execute workflows inside predefined object models; it must learn when the object model itself is wrong or incomplete.

## Progress Impact Estimate

```text
Productization Closure: +4% after OM-1A/1B closure
AGI Precursor Closure: +6% after cross-domain object-model evolution validation
```

## PM Instruction

Proceed with OM-1A as a mainline patch seed. Keep the first implementation minimal and candidate-only. Do not attempt full ontology auto-evolution. Close the role, schema, lifecycle, boundaries, and synthetic mismatch fixtures first.
