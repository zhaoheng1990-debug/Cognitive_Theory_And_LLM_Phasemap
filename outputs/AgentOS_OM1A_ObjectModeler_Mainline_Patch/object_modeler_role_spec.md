# DomainObjectModeler Role Spec

## Role

`DomainObjectModeler` is a cross-domain AgentOS governance / cognitive-runtime role.

It observes workflow traces, project materials, synthetic fixtures, permission denials, operational events, harness extraction outputs, and human corrections to propose candidate-only changes to a domain object model.

## Responsibility

- Discover missing domain objects from runtime evidence.
- Detect object boundary, relation, state, lifecycle, permission, and schema mismatches.
- Produce `PENDING` object model candidates with evidence lineage.
- Route candidates through conflict, permission, schema, retention, and human review gates.
- Preserve accepted registry boundaries.

## Non-Responsibility

- No accepted object registry mutation.
- No direct database schema change.
- No production/global activation.
- No candidate promotion.
- No private/confidential-to-shared promotion without explicit PermissionGovernor review marker.
- No Harness worker promotion authority.

## Runtime Placement

AgentOS Governance / Cognitive Runtime Layer.

The role is not a Harness worker. Harness may extract signals, but AgentOSKernel owns object-model lifecycle decisions.

## Implementation

- Kernel module: `agentos_core_slim_v0/agentos_kernel/object_modeler.py`
- Export: `agentos_kernel.DomainObjectModeler`
- Boundary tests: `agentos_core_slim_v0/tests/test_object_modeler.py`
