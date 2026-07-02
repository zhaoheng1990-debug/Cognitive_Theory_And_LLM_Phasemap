import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentos_kernel import DomainObjectModeler, ObjectModelPolicyBlocked, ROLE_REGISTRY


VC_REGISTRY = {
    "objects": [
        "DealTarget",
        "IndustryDomain",
        "BPClaim",
        "DiligenceTask",
        "CognitiveAsset",
    ]
}


def vc_signal():
    return {
        "signal_id": "vc-fixture-a",
        "domain": "VC-AgentOS",
        "source": "synthetic_vc_fixture",
        "access_level": "confidential",
        "target_scope": "project_domain_model",
        "evidence_refs": ["workflow_ref_001", "permission_denial_ref_001"],
        "observed_objects": [
            {
                "name": "FounderCredibilitySignal",
                "permission_relevant": True,
                "permission_impact": {
                    "permission_relevant": True,
                    "notes": "founder-signal access must remain project-scoped until review",
                },
            },
            {
                "name": "CustomerBudgetSignal",
                "permission_relevant": True,
                "permission_impact": {
                    "permission_relevant": True,
                    "notes": "customer budget signal may reveal confidential sales context",
                },
            },
            {
                "name": "CoInvestorInformationBoundary",
                "permission_relevant": True,
                "permission_impact": {
                    "permission_relevant": True,
                    "notes": "co-investor boundaries require explicit access policy",
                },
            },
        ],
        "schema_impact": {"requires_schema_review": True, "impact": "new_vc_object_candidates"},
    }


def education_signal():
    return {
        "signal_id": "edu-fixture-b",
        "domain": "Education-AgentOS",
        "source": "synthetic_education_fixture",
        "access_level": "internal",
        "target_scope": "project_domain_model",
        "evidence_refs": ["lesson_trace_001"],
        "observed_objects": [
            {"name": "LearningBottleneck"},
            {"name": "StudentMisconceptionCluster"},
        ],
        "schema_impact": {"requires_schema_review": True, "impact": "new_education_object_candidates"},
    }


def test_object_modeler_role_is_registered():
    assert "DomainObjectModeler" in ROLE_REGISTRY
    entry = DomainObjectModeler().role_registry_entry()
    assert entry["candidate_only"] is True
    assert entry["may_mutate_accepted_registry"] is False


def test_candidate_object_model_outputs_are_pending_only():
    patch = DomainObjectModeler().propose_patch_candidate(vc_signal(), VC_REGISTRY)
    assert patch["status"] == "PENDING"
    assert all(candidate["status"] == "PENDING" for candidate in patch["object_candidates"])
    assert patch["accepted_registry_mutation"] is False


def test_accepted_object_registry_cannot_be_mutated_by_object_modeler():
    modeler = DomainObjectModeler()
    try:
        modeler.mutate_accepted_registry(VC_REGISTRY, {"object_name": "FounderCredibilitySignal"})
    except ObjectModelPolicyBlocked as exc:
        assert "cannot_mutate_accepted_registry" in str(exc)
    else:
        raise AssertionError("DomainObjectModeler must not mutate accepted registries")


def test_harness_worker_cannot_promote_object_candidates():
    patch = DomainObjectModeler().propose_patch_candidate(vc_signal(), VC_REGISTRY)
    try:
        DomainObjectModeler().promote_candidate(patch["object_candidates"][0], actor="HarnessWorker")
    except ObjectModelPolicyBlocked as exc:
        assert "HarnessWorker_cannot_promote" in str(exc)
    else:
        raise AssertionError("Harness worker must not promote object model candidates")


def test_every_candidate_requires_evidence_lineage():
    signal = vc_signal()
    signal["evidence_refs"] = []
    for item in signal["observed_objects"]:
        item.pop("evidence_refs", None)
    try:
        DomainObjectModeler().propose_patch_candidate(signal, VC_REGISTRY)
    except ObjectModelPolicyBlocked as exc:
        assert "evidence_lineage_required" in str(exc)
    else:
        raise AssertionError("object candidates without evidence lineage must be blocked")


def test_permission_impact_required_for_permission_relevant_objects():
    signal = vc_signal()
    signal["observed_objects"][0].pop("permission_impact")
    signal.pop("permission_impact", None)
    try:
        DomainObjectModeler().propose_patch_candidate(signal, VC_REGISTRY)
    except ObjectModelPolicyBlocked as exc:
        assert "permission_impact_required" in str(exc)
    else:
        raise AssertionError("permission-relevant objects require permission impact")


def test_schema_impact_required_for_patch_candidates():
    patch = DomainObjectModeler().propose_patch_candidate(education_signal(), VC_REGISTRY)
    patch["schema_impact"] = {}
    try:
        DomainObjectModeler().validate_patch_candidate(patch)
    except ObjectModelPolicyBlocked as exc:
        assert "schema_impact_required" in str(exc)
    else:
        raise AssertionError("patch candidates without schema impact must be blocked")


def test_private_input_cannot_enter_shared_model_without_review_marker():
    signal = vc_signal()
    signal["target_scope"] = "global"
    try:
        DomainObjectModeler().propose_patch_candidate(signal, VC_REGISTRY)
    except ObjectModelPolicyBlocked as exc:
        assert "private_to_shared_requires_explicit_review_marker" in str(exc)
    else:
        raise AssertionError("confidential source cannot promote to global model without review marker")


def test_vc_fixture_proposes_missing_vc_object_candidates():
    patch = DomainObjectModeler().propose_patch_candidate(vc_signal(), VC_REGISTRY)
    names = {candidate["object_name"] for candidate in patch["object_candidates"]}
    assert {"FounderCredibilitySignal", "CustomerBudgetSignal", "CoInvestorInformationBoundary"} <= names
    assert patch["permission_delta_candidates"]


def test_non_vc_fixture_proposes_missing_generic_domain_objects():
    patch = DomainObjectModeler().propose_patch_candidate(education_signal(), {"objects": ["Course", "Student"]})
    names = {candidate["object_name"] for candidate in patch["object_candidates"]}
    assert {"LearningBottleneck", "StudentMisconceptionCluster"} <= names
    assert all(candidate["source_domain"] == "Education-AgentOS" for candidate in patch["object_candidates"])


def test_existing_object_conflict_is_detected():
    signal = education_signal()
    signal["observed_objects"].append({"name": "DealTarget"})
    patch = DomainObjectModeler().propose_patch_candidate(signal, VC_REGISTRY)
    assert any(record["object_name"] == "DealTarget" for record in patch["conflict_records"])


def test_stale_or_overloaded_object_produces_revise_or_archive_not_deletion():
    signal = education_signal()
    signal["observed_objects"] = [
        {"name": "LegacyCourseBucket", "mismatch": "stale_object"},
        {"name": "StudentProfile", "mismatch": "overloaded_object"},
    ]
    patch = DomainObjectModeler().propose_patch_candidate(signal, {"objects": ["LegacyCourseBucket", "StudentProfile"]})
    decisions = {item["object_name"]: item for item in patch["retention_decisions"]}
    assert decisions["LegacyCourseBucket"]["recommended_decision"] == "ARCHIVE"
    assert decisions["StudentProfile"]["recommended_decision"] == "REVISE"
    assert all(item["direct_deletion"] is False for item in patch["retention_decisions"])
