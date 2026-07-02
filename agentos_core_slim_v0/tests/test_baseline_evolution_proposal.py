import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentos_kernel import BaselineEvolutionProposalProtocol, ProposalQueue


def ups_proposal_candidate():
    return {
        "candidate_id": "ups_theory_baseline_update",
        "source_project": "AgentOS_CoreSlim",
        "source_decision_ref": "7301A7400Z",
        "accept_decision_ref": "UPS_ACCEPTDecisionRecord",
        "project_scoped_write_ref": "project_scoped_icm_store/operator_memory/ups_accept_operator_memory.json",
        "project_scoped_icm_refs": ["UtilityPolicySelectorOperatorMemoryRecord"],
        "evidence_refs": ["SR_series", "UPS_validation", "reuse_gain_records", "negative_transfer_audit"],
        "empirical_validation_refs": ["replay_PASS", "rollback_pointer_present"],
        "replayable_evidence": True,
        "cross_project_reuse_value": 0.82,
        "negative_transfer_risk": "bounded",
        "conflict_scan": "complete_no_unresolved_blocker",
        "target_scope": "S3_OFFICIAL_THEORY_BASELINE",
        "target_path_or_registry": r"C:\Users\ZH\Desktop\AGI\理论基线",
        "proposed_action": "PROPOSE_APPEND_BASELINE_UPDATE",
        "proposed_diff_summary": "Append UtilityPolicySelector as accepted project-scoped operator memory evidence for theory baseline review.",
        "conflict_scan_summary": "complete_no_unresolved_blocker",
        "negative_transfer_audit": "bounded",
        "cross_project_reuse_argument": "UPS supports repeated policy selection when structural resolution is insufficient.",
        "risk_if_not_promoted": "Canonical baseline may lag accepted self-evolution evidence.",
        "risk_if_promoted_too_early": "Overgeneralization without signed review.",
    }


def test_ups_project_operator_memory_generates_theory_baseline_proposal(tmp_path):
    protocol = BaselineEvolutionProposalProtocol()
    queue = ProposalQueue(tmp_path / "proposal_queue")

    review = protocol.review_eligibility(ups_proposal_candidate())
    proposal = protocol.build_proposal(ups_proposal_candidate(), review)
    queue_entry = queue.enqueue(proposal)
    packet = protocol.human_review_packet(proposal)

    assert review.action == "ENQUEUE_BASELINE_UPDATE_PROPOSAL"
    assert proposal["proposal_type"] == "THEORY_BASELINE"
    assert proposal["target_scope"] == "S3_OFFICIAL_THEORY_BASELINE"
    assert proposal["requires_signed_authorization"] is True
    assert proposal["production_activation"] is False
    assert proposal["official_baseline_written"] is False
    assert queue_entry["official_baseline_written"] is False
    assert packet["review_required"] == "SignedBaselinePromotionAuthorization"


def test_insufficient_empirical_evidence_produces_no_proposal():
    protocol = BaselineEvolutionProposalProtocol()
    candidate = ups_proposal_candidate()
    candidate["empirical_validation_refs"] = []

    review = protocol.review_eligibility(candidate)

    assert review.action == "DEFER_INSUFFICIENT_EVIDENCE"
    assert review.eligible is False


def test_high_negative_transfer_routes_to_experiment_or_quarantine():
    protocol = BaselineEvolutionProposalProtocol()
    candidate = ups_proposal_candidate()
    candidate["negative_transfer_risk"] = "high_unbounded"

    review = protocol.review_eligibility(candidate)

    assert review.action == "ROUTE_TO_EXPERIMENT"
    assert review.eligible is False


def test_conflict_with_accept_baseline_routes_to_conflict_review():
    protocol = BaselineEvolutionProposalProtocol()
    candidate = ups_proposal_candidate()
    candidate["conflict_scan"] = "unresolved_blocker"

    review = protocol.review_eligibility(candidate)

    assert review.action == "REQUEST_CONFLICT_REVIEW"
    assert review.eligible is False


def test_user_praise_only_cannot_generate_proposal():
    protocol = BaselineEvolutionProposalProtocol()
    candidate = {
        "candidate_id": "praise_only",
        "accept_decision_ref": "",
        "project_scoped_write_ref": "",
        "empirical_validation_refs": [],
        "user_praise": True,
        "cross_project_reuse_value": 1,
    }

    review = protocol.review_eligibility(candidate)

    assert review.eligible is False
    assert review.action == "DEFER_INSUFFICIENT_EVIDENCE"
