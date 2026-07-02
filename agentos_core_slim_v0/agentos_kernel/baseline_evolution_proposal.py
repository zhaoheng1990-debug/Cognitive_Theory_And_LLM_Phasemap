"""S3/S4 baseline evolution proposal protocol.

AgentOS may autonomously propose official/global baseline updates from mature
project-scoped durable learning. It may not autonomously apply those updates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any


PROPOSAL_ACTIONS = {
    "ENQUEUE_BASELINE_UPDATE_PROPOSAL",
    "MERGE_WITH_EXISTING_PROPOSAL",
    "DEFER_INSUFFICIENT_EVIDENCE",
    "REJECT_INSUFFICIENT_GENERALITY",
    "ROUTE_TO_EXPERIMENT",
    "REQUEST_CONFLICT_REVIEW",
}


@dataclass(frozen=True)
class ProposalEligibilityReview:
    action: str
    eligible: bool
    reason: str


def _hash_payload(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(data).hexdigest()


class BaselineEvolutionProposalProtocol:
    """Proposal-only bridge from S2 learning to S3/S4 human-authorized promotion."""

    owner = "AgentOSKernel.BaselineEvolutionProposalPolicy"
    signed_authorization_required_for_write = True
    direct_s3_s4_write_allowed = False

    def review_eligibility(self, candidate: dict[str, Any]) -> ProposalEligibilityReview:
        if not candidate.get("accept_decision_ref"):
            return ProposalEligibilityReview("DEFER_INSUFFICIENT_EVIDENCE", False, "missing_accept_decision_ref")
        if not candidate.get("project_scoped_write_ref"):
            return ProposalEligibilityReview("DEFER_INSUFFICIENT_EVIDENCE", False, "missing_project_scoped_write_ref")
        if not candidate.get("empirical_validation_refs"):
            return ProposalEligibilityReview("DEFER_INSUFFICIENT_EVIDENCE", False, "missing_empirical_validation")
        if not candidate.get("replayable_evidence", False):
            return ProposalEligibilityReview("DEFER_INSUFFICIENT_EVIDENCE", False, "missing_replayable_evidence")
        if candidate.get("negative_transfer_risk") in {"high_unbounded", "unbounded", "high"}:
            return ProposalEligibilityReview("ROUTE_TO_EXPERIMENT", False, "high_negative_transfer")
        if candidate.get("conflict_scan") == "unresolved_blocker":
            return ProposalEligibilityReview("REQUEST_CONFLICT_REVIEW", False, "conflict_with_accept_baseline")
        if candidate.get("cross_project_reuse_value", 0) <= 0:
            return ProposalEligibilityReview("REJECT_INSUFFICIENT_GENERALITY", False, "no_cross_project_reuse_value")
        return ProposalEligibilityReview("ENQUEUE_BASELINE_UPDATE_PROPOSAL", True, "eligible_empirical_project_scoped_learning")

    def build_proposal(self, candidate: dict[str, Any], review: ProposalEligibilityReview) -> dict[str, Any]:
        if not review.eligible:
            raise ValueError(f"candidate_not_eligible_for_baseline_update_proposal:{review.reason}")
        proposal = {
            "proposal_id": f"bup-{candidate['candidate_id']}",
            "proposal_type": candidate.get("proposal_type", "THEORY_BASELINE"),
            "source_project": candidate.get("source_project", "AgentOS_CoreSlim"),
            "source_decision_ref": candidate.get("source_decision_ref", ""),
            "accept_decision_ref": candidate["accept_decision_ref"],
            "project_scoped_write_ref": candidate["project_scoped_write_ref"],
            "evidence_refs": candidate.get("evidence_refs", []),
            "empirical_validation_refs": candidate.get("empirical_validation_refs", []),
            "project_scoped_icm_refs": candidate.get("project_scoped_icm_refs", [candidate["project_scoped_write_ref"]]),
            "target_scope": candidate.get("target_scope", "S3_OFFICIAL_THEORY_BASELINE"),
            "target_path_or_registry": candidate.get("target_path_or_registry", ""),
            "proposed_action": candidate.get("proposed_action", "PROPOSE_APPEND_BASELINE_UPDATE"),
            "proposed_diff_summary": candidate.get("proposed_diff_summary", ""),
            "conflict_scan_summary": candidate.get("conflict_scan_summary", "complete_no_unresolved_blocker"),
            "negative_transfer_audit": candidate.get("negative_transfer_audit", "bounded"),
            "cross_project_reuse_argument": candidate.get("cross_project_reuse_argument", ""),
            "risk_if_not_promoted": candidate.get("risk_if_not_promoted", ""),
            "risk_if_promoted_too_early": candidate.get("risk_if_promoted_too_early", ""),
            "recommended_human_decision": candidate.get("recommended_human_decision", "APPROVE"),
            "requires_signed_authorization": True,
            "production_activation": False,
            "official_baseline_written": False,
        }
        proposal["proposal_hash"] = _hash_payload(proposal)
        return proposal

    def human_review_packet(self, proposal: dict[str, Any]) -> dict[str, Any]:
        return {
            "packet_id": f"hrp-{proposal['proposal_id']}",
            "proposal_id": proposal["proposal_id"],
            "proposal_hash": proposal["proposal_hash"],
            "review_required": "SignedBaselinePromotionAuthorization",
            "decision_options": ["APPROVE", "REJECT", "REQUEST_MORE_EVIDENCE", "NARROW_SCOPE"],
            "baseline_write_precondition": "signed_authorization_required",
            "official_baseline_written": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


class ProposalQueue:
    """Append-only local queue for baseline update proposals."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.queue_path = self.root / "baseline_update_proposal_queue.jsonl"

    def enqueue(self, proposal: dict[str, Any]) -> dict[str, Any]:
        entry = {
            "queued_at": datetime.now(timezone.utc).isoformat(),
            "proposal_id": proposal["proposal_id"],
            "proposal_hash": proposal["proposal_hash"],
            "target_scope": proposal["target_scope"],
            "requires_signed_authorization": proposal["requires_signed_authorization"],
            "official_baseline_written": False,
        }
        with self.queue_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def entries(self) -> list[dict[str, Any]]:
        if not self.queue_path.exists():
            return []
        return [json.loads(line) for line in self.queue_path.read_text(encoding="utf-8").splitlines() if line.strip()]
