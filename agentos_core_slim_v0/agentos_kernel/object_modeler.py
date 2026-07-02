"""Candidate-only domain object model evolution primitives.

DomainObjectModeler observes workflow evidence and proposes object-model
candidates. It never mutates accepted object registries and never promotes
candidates to accepted state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any


ROLE_NAME = "DomainObjectModeler"
ROLE_OWNER = "AgentOSKernel.DomainObjectModelerPolicy"
PENDING_STATUS = "PENDING"

ROLE_REGISTRY = {
    ROLE_NAME: {
        "role_name": ROLE_NAME,
        "layer": "AgentOS Governance / Cognitive Runtime Layer",
        "scope": "cross_domain",
        "candidate_only": True,
        "final_decision_owner": "AgentOSKernel",
        "may_mutate_accepted_registry": False,
        "may_promote_candidates": False,
    }
}

REQUIRED_LIFECYCLE = [
    "ObservedSignal",
    "ObjectModelCandidate",
    "EvidenceAttachment",
    "ConflictCheck",
    "PermissionImpactCheck",
    "SchemaImpactCheck",
    "RetentionReview",
]

CANDIDATE_KINDS = {
    "ObjectModelCandidate",
    "ObjectRelationCandidate",
    "ObjectStateCandidate",
    "ObjectLifecycleDeltaCandidate",
    "ObjectPermissionDeltaCandidate",
    "ObjectModelPatchCandidate",
}

REVIEW_DECISIONS = {"ACCEPT", "REVISE", "REJECT", "ARCHIVE"}
PRIVATE_ACCESS_LEVELS = {"private", "confidential", "restricted"}


class ObjectModelPolicyBlocked(RuntimeError):
    """Raised when object-model governance must fail closed."""


@dataclass(frozen=True)
class ObjectModelReview:
    decision: str
    reason: str
    eligible_for_review: bool


def _hash_payload(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(data).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DomainObjectModeler:
    """Governed role for proposing object model candidates from evidence."""

    role_name = ROLE_NAME
    owner = ROLE_OWNER
    candidate_only = True
    harness_role = "evidence_extraction_only"
    final_decision_owner = "AgentOSKernel"

    def role_registry_entry(self) -> dict[str, Any]:
        return dict(ROLE_REGISTRY[ROLE_NAME])

    def propose_patch_candidate(
        self,
        observed_signal: dict[str, Any],
        accepted_registry: dict[str, Any],
    ) -> dict[str, Any]:
        self._validate_observed_signal(observed_signal)
        baseline_objects = set(accepted_registry.get("objects", []))
        observed_objects = observed_signal.get("observed_objects", [])
        candidates = []
        conflicts = []
        retention_decisions = []

        for item in observed_objects:
            name = item["name"]
            evidence_refs = item.get("evidence_refs") or observed_signal.get("evidence_refs") or []
            if not evidence_refs:
                raise ObjectModelPolicyBlocked(f"missing_evidence_lineage:{name}")

            mismatch = item.get("mismatch", "missing_object")
            if name in baseline_objects and mismatch == "missing_object":
                conflicts.append(self._conflict_record(name, "existing_object_conflict", evidence_refs))
                continue

            if mismatch in {"stale_object", "overloaded_object"}:
                retention_decisions.append(self._retention_decision(item, evidence_refs))
                continue

            candidate = self._candidate_from_item(item, observed_signal, accepted_registry, evidence_refs)
            candidates.append(candidate)

        patch = {
            "patch_candidate_id": f"omp-{observed_signal['signal_id']}",
            "kind": "ObjectModelPatchCandidate",
            "role": ROLE_NAME,
            "status": PENDING_STATUS,
            "target_scope": observed_signal.get("target_scope", "project_domain_model"),
            "source_access_level": observed_signal.get("access_level", "internal"),
            "object_candidates": candidates,
            "relation_candidates": self._relation_candidates(observed_signal, baseline_objects),
            "state_candidates": self._state_candidates(observed_signal),
            "lifecycle_delta_candidates": self._lifecycle_delta_candidates(observed_signal),
            "permission_delta_candidates": self._permission_delta_candidates(candidates),
            "conflict_records": conflicts,
            "retention_decisions": retention_decisions,
            "schema_impact": observed_signal.get("schema_impact") or self._default_schema_impact(candidates),
            "permission_impact": observed_signal.get("permission_impact") or self._default_permission_impact(candidates, observed_signal),
            "lifecycle": list(REQUIRED_LIFECYCLE),
            "review_required": "RetentionReview",
            "accepted_registry_mutation": False,
            "production_activation": False,
            "created_at": _utc_now(),
        }
        self.validate_patch_candidate(patch)
        patch["patch_hash"] = _hash_payload(patch)
        return patch

    def validate_patch_candidate(self, patch: dict[str, Any]) -> None:
        if patch.get("status") != PENDING_STATUS:
            raise ObjectModelPolicyBlocked("object_model_patch_must_remain_pending")
        if patch.get("accepted_registry_mutation"):
            raise ObjectModelPolicyBlocked("accepted_registry_mutation_forbidden")
        if patch.get("production_activation"):
            raise ObjectModelPolicyBlocked("production_activation_forbidden")
        if not patch.get("schema_impact"):
            raise ObjectModelPolicyBlocked("schema_impact_required")
        if patch.get("source_access_level") in PRIVATE_ACCESS_LEVELS and patch.get("target_scope") in {"shared", "global"}:
            marker = patch.get("explicit_review_marker")
            if marker != "PermissionGovernorReviewedForSharedModel":
                raise ObjectModelPolicyBlocked("private_to_shared_requires_explicit_review_marker")
        for candidate in patch.get("object_candidates", []):
            self._validate_candidate(candidate)

    def review_candidate(self, candidate: dict[str, Any], decision: str) -> dict[str, Any]:
        if decision not in REVIEW_DECISIONS:
            raise ObjectModelPolicyBlocked(f"unknown_retention_decision:{decision}")
        self._validate_candidate(candidate)
        return {
            "retention_decision_id": f"omr-{candidate['candidate_id']}-{decision.lower()}",
            "candidate_id": candidate["candidate_id"],
            "decision": decision,
            "status_after_decision": PENDING_STATUS,
            "accepted_registry_mutation": False,
            "requires_human_or_schema_gate_for_accept": decision == "ACCEPT",
        }

    def mutate_accepted_registry(self, *_args: Any, **_kwargs: Any) -> None:
        raise ObjectModelPolicyBlocked("DomainObjectModeler_cannot_mutate_accepted_registry")

    def promote_candidate(self, candidate: dict[str, Any], actor: str) -> None:
        if actor in {"HarnessWorker", "DomainObjectModeler", "CodexToolBridge"}:
            raise ObjectModelPolicyBlocked(f"{actor}_cannot_promote_object_model_candidate")
        if not candidate.get("signed_human_schema_review"):
            raise ObjectModelPolicyBlocked("promotion_requires_signed_human_schema_review")
        raise ObjectModelPolicyBlocked("promotion_not_implemented_in_candidate_only_om1a")

    def _validate_observed_signal(self, signal: dict[str, Any]) -> None:
        if not signal.get("signal_id"):
            raise ObjectModelPolicyBlocked("signal_id_required")
        if not signal.get("evidence_refs") and not any(item.get("evidence_refs") for item in signal.get("observed_objects", [])):
            raise ObjectModelPolicyBlocked("evidence_lineage_required")

    def _candidate_from_item(
        self,
        item: dict[str, Any],
        signal: dict[str, Any],
        registry: dict[str, Any],
        evidence_refs: list[str],
    ) -> dict[str, Any]:
        permission_relevant = bool(item.get("permission_relevant")) or bool(item.get("permission_surface"))
        permission_impact = item.get("permission_impact") or signal.get("permission_impact")
        if permission_relevant and not permission_impact:
            raise ObjectModelPolicyBlocked(f"permission_impact_required:{item['name']}")
        candidate = {
            "candidate_id": f"omc-{signal['signal_id']}-{item['name']}",
            "kind": item.get("kind", "ObjectModelCandidate"),
            "object_name": item["name"],
            "status": PENDING_STATUS,
            "source_signal_id": signal["signal_id"],
            "source_domain": signal.get("domain", "generic"),
            "evidence": [
                {
                    "evidence_ref": ref,
                    "source": signal.get("source", "synthetic_fixture"),
                    "access_level": signal.get("access_level", "internal"),
                    "confidence": item.get("confidence", signal.get("confidence", 0.7)),
                    "validity_scope": signal.get("validity_scope", signal.get("domain", "project_domain")),
                    "timestamp": signal.get("timestamp", _utc_now()),
                }
                for ref in evidence_refs
            ],
            "conflict_check": self._conflict_check(item["name"], registry),
            "permission_impact": permission_impact or {"permission_relevant": False, "notes": "no new permission surface detected"},
            "schema_impact": item.get("schema_impact") or {"requires_schema_review": True, "impact": "candidate_object_type"},
            "retention_review_required": True,
            "accepted_registry_mutation": False,
        }
        self._validate_candidate(candidate)
        return candidate

    def _validate_candidate(self, candidate: dict[str, Any]) -> None:
        if candidate.get("kind") not in CANDIDATE_KINDS:
            raise ObjectModelPolicyBlocked(f"unknown_candidate_kind:{candidate.get('kind')}")
        if candidate.get("status") != PENDING_STATUS:
            raise ObjectModelPolicyBlocked("object_model_candidate_must_remain_pending")
        if candidate.get("accepted_registry_mutation"):
            raise ObjectModelPolicyBlocked("accepted_registry_mutation_forbidden")
        if not candidate.get("evidence"):
            raise ObjectModelPolicyBlocked("candidate_evidence_lineage_required")
        if not candidate.get("schema_impact"):
            raise ObjectModelPolicyBlocked("candidate_schema_impact_required")
        impact = candidate.get("permission_impact") or {}
        if impact.get("permission_relevant") and not impact.get("notes"):
            raise ObjectModelPolicyBlocked("candidate_permission_impact_notes_required")

    @staticmethod
    def _conflict_check(name: str, registry: dict[str, Any]) -> dict[str, Any]:
        baseline = set(registry.get("objects", []))
        if name in baseline:
            return {"status": "CONFLICT", "reason": "object_already_exists"}
        return {"status": "CLEAR", "reason": "not_in_accepted_registry"}

    @staticmethod
    def _conflict_record(name: str, reason: str, evidence_refs: list[str]) -> dict[str, Any]:
        return {
            "kind": "ObjectModelConflictRecord",
            "object_name": name,
            "reason": reason,
            "evidence_refs": evidence_refs,
            "status": PENDING_STATUS,
        }

    @staticmethod
    def _retention_decision(item: dict[str, Any], evidence_refs: list[str]) -> dict[str, Any]:
        mismatch = item.get("mismatch", "overloaded_object")
        decision = "ARCHIVE" if mismatch == "stale_object" else "REVISE"
        return {
            "kind": "ObjectModelRetentionDecision",
            "object_name": item["name"],
            "recommended_decision": decision,
            "status": PENDING_STATUS,
            "direct_deletion": False,
            "evidence_refs": evidence_refs,
        }

    @staticmethod
    def _relation_candidates(signal: dict[str, Any], baseline_objects: set[str]) -> list[dict[str, Any]]:
        relations = []
        for relation in signal.get("observed_relations", []):
            relations.append({
                "kind": "ObjectRelationCandidate",
                "status": PENDING_STATUS,
                "from_object": relation["from"],
                "to_object": relation["to"],
                "relation_name": relation["name"],
                "conflict_check": {
                    "status": "PENDING_REVIEW" if relation["from"] not in baseline_objects or relation["to"] not in baseline_objects else "CLEAR"
                },
            })
        return relations

    @staticmethod
    def _state_candidates(signal: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "kind": "ObjectStateCandidate",
                "status": PENDING_STATUS,
                "object_name": state["object"],
                "state_name": state["state"],
                "evidence_refs": state.get("evidence_refs", signal.get("evidence_refs", [])),
            }
            for state in signal.get("observed_states", [])
        ]

    @staticmethod
    def _lifecycle_delta_candidates(signal: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "kind": "ObjectLifecycleDeltaCandidate",
                "status": PENDING_STATUS,
                "object_name": delta["object"],
                "delta": delta["delta"],
                "evidence_refs": delta.get("evidence_refs", signal.get("evidence_refs", [])),
            }
            for delta in signal.get("lifecycle_deltas", [])
        ]

    @staticmethod
    def _permission_delta_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deltas = []
        for candidate in candidates:
            impact = candidate.get("permission_impact") or {}
            if impact.get("permission_relevant"):
                deltas.append({
                    "kind": "ObjectPermissionDeltaCandidate",
                    "status": PENDING_STATUS,
                    "object_name": candidate["object_name"],
                    "permission_impact": impact,
                })
        return deltas

    @staticmethod
    def _default_schema_impact(candidates: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "requires_schema_review": True,
            "candidate_count": len(candidates),
            "registry_write_allowed": False,
        }

    @staticmethod
    def _default_permission_impact(candidates: list[dict[str, Any]], signal: dict[str, Any]) -> dict[str, Any]:
        relevant = any((candidate.get("permission_impact") or {}).get("permission_relevant") for candidate in candidates)
        return {
            "permission_relevant": relevant,
            "access_level": signal.get("access_level", "internal"),
            "notes": "permission surfaces remain candidate-only and require PermissionGovernor review",
        }
