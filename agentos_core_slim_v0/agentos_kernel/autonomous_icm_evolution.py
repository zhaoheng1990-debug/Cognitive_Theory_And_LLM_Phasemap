"""Project-scoped autonomous ICM evolution policy.

This module lets AgentOSKernel promote ACCEPT-level, replayable research-line
closures into project-scoped durable ICM artifacts. It deliberately does not
write global memory, production ICM, policy promotion, AcceptedEvidence, or
theory-baseline files.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any


POLICY_OWNER = "AgentOSKernel.ICMEvolutionPolicy"

POLICY_DECISIONS = {
    "NO_WRITE_KEEP_CANDIDATE",
    "AUTONOMOUS_PROJECT_MEMORYUNIT_WRITE",
    "AUTONOMOUS_PROJECT_OPERATORMEMORY_WRITE",
    "AUTONOMOUS_POLICY_PRIOR_WRITE",
    "AUTONOMOUS_VALIDITY_UPDATE",
    "AUTONOMOUS_SCOPE_NARROW",
    "AUTONOMOUS_QUARANTINE",
    "REQUEST_HUMAN_SCOPE_ESCALATION",
    "BLOCK_WRITE_INSUFFICIENT_EVIDENCE",
    "ROLLBACK_PREVIOUS_WRITE",
}

TARGET_DIR_BY_TYPE = {
    "MemoryUnit": "memory_units",
    "OperatorMemory": "operator_memory",
    "PolicyPrior": "policy_priors",
    "ApplicabilityGate": "applicability_gates",
    "BoundaryPolicy": "boundary_policies",
    "ReusePolicy": "reuse_policies",
    "ValidityMap": "validity_map",
    "DriftWatch": "drift_watch",
    "Quarantine": "negative_transfer_quarantine",
}

ALLOWED_STORE_DIRS = set(TARGET_DIR_BY_TYPE.values()) | {
    "evolution_ledger",
    "rollback",
    "replay_manifest",
    "human_posthoc_reports",
}


class EvolutionPolicyBlocked(RuntimeError):
    """Raised when an autonomous durable write must be blocked."""


def _hash_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def _hash_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_hash(payload: Any) -> str:
    return _hash_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


@dataclass(frozen=True)
class EvolutionReview:
    decision: str
    target_type: str
    reason: str
    eligible: bool


class AutonomousICMEvolutionPolicy:
    """Kernel-owned decision policy for project-scoped durable evolution."""

    owner = POLICY_OWNER
    llm_api_role = "advisory_only"
    harness_role = "execute_write_envelope_only"
    final_decision_owner = "AgentOSKernel"

    def review(self, candidate: dict[str, Any]) -> EvolutionReview:
        if candidate.get("scope") not in {"project_scoped", "AgentOS project runtime policy selection"}:
            return EvolutionReview("REQUEST_HUMAN_SCOPE_ESCALATION", "None", "scope_not_project_bounded", False)
        if candidate.get("negative_transfer_risk") in {"high", "unbounded"} or candidate.get("negative_transfer_detected"):
            return EvolutionReview("AUTONOMOUS_QUARANTINE", "Quarantine", "negative_transfer_detected", True)
        if not self._has_accept_evidence(candidate):
            return EvolutionReview("BLOCK_WRITE_INSUFFICIENT_EVIDENCE", "None", "missing_accept_or_replayable_evidence", False)
        if candidate.get("future_cbit_gain") not in {"positive", "high", True}:
            return EvolutionReview("NO_WRITE_KEEP_CANDIDATE", "None", "future_cbit_gain_not_positive", False)
        kind = candidate.get("target_type") or ("OperatorMemory" if "UPS" in candidate.get("research_line", "").upper() or "UtilityPolicySelector" in candidate.get("research_line", "") else "MemoryUnit")
        if kind == "OperatorMemory":
            return EvolutionReview("AUTONOMOUS_PROJECT_OPERATORMEMORY_WRITE", kind, "accept_replayable_project_scoped_operator_memory", True)
        if kind == "PolicyPrior":
            return EvolutionReview("AUTONOMOUS_POLICY_PRIOR_WRITE", kind, "accept_replayable_project_scoped_policy_prior", True)
        return EvolutionReview("AUTONOMOUS_PROJECT_MEMORYUNIT_WRITE", "MemoryUnit", "accept_replayable_project_scoped_memory_unit", True)

    @staticmethod
    def _has_accept_evidence(candidate: dict[str, Any]) -> bool:
        status = str(candidate.get("status", ""))
        refs = candidate.get("evidence_refs") or []
        return "ACCEPT" in status and bool(refs) and bool(candidate.get("replayable_evidence", True))

    def build_envelope(self, candidate: dict[str, Any], review: EvolutionReview, store: "ProjectScopedDurableStore") -> dict[str, Any]:
        if not review.eligible:
            raise EvolutionPolicyBlocked(review.reason)
        payload = self._payload_for(candidate, review)
        write_id = f"evo-{candidate['candidate_id']}-{review.decision.lower()}"
        target_path = store.target_path(review.target_type, f"{candidate['candidate_id']}.json")
        return {
            "write_id": write_id,
            "decision_id": f"decision-{candidate['candidate_id']}",
            "authorized_by": POLICY_OWNER,
            "target_scope": "project_scoped_durable",
            "target_type": review.target_type,
            "target_path": str(target_path),
            "payload": payload,
            "evidence_refs": candidate.get("evidence_refs", []),
            "accept_decision_ref": candidate.get("accept_decision_ref", ""),
            "applicability_gate": payload.get("applicability_gate", {}),
            "boundary_policy": payload.get("boundary_policy", {}),
            "reuse_policy": payload.get("reuse_policy", {}),
            "drift_watch": payload.get("drift_watch", {}),
            "rollback_required": True,
            "production_activation": False,
        }

    @staticmethod
    def _payload_for(candidate: dict[str, Any], review: EvolutionReview) -> dict[str, Any]:
        if review.target_type == "OperatorMemory" and "UtilityPolicySelector" in candidate.get("research_line", ""):
            return {
                "operator_id": "UtilityPolicySelectorOperator",
                "operator_family": "PolicySelection",
                "purpose": "select lifecycle / next-action / retention policy when structural resolution alone is insufficient",
                "applicability_gate": {
                    "requires": [
                        "ranked_compatibility_fiber",
                        "utility_signal",
                        "risk_signal",
                        "cbit_gain_signal",
                    ]
                },
                "boundary_policy": {
                    "do_not_use_when": [
                        "evidence_missing",
                        "high_negative_transfer",
                        "external_action_required_without_authority",
                    ]
                },
                "reuse_policy": {
                    "mode": "project_scoped",
                    "requires_replay": True,
                    "drift_watch": True,
                },
                "drift_watch": {
                    "watch_family": "utility_policy_selector",
                    "revalidate_on_scope_change": True,
                },
            }
        if review.target_type == "Quarantine":
            return {
                "candidate_id": candidate["candidate_id"],
                "quarantine_reason": "negative_transfer_detected",
                "reuse_allowed": False,
                "requires_human_or_kernel_repair": True,
            }
        return {
            "candidate_id": candidate["candidate_id"],
            "research_line": candidate.get("research_line"),
            "summary": candidate.get("summary", ""),
            "applicability_gate": candidate.get("applicability_gate", {"requires": ["replayable_evidence"]}),
            "boundary_policy": candidate.get("boundary_policy", {"scope": "project_scoped"}),
            "reuse_policy": candidate.get("reuse_policy", {"mode": "project_scoped", "requires_replay": True}),
            "drift_watch": candidate.get("drift_watch", {"enabled": True}),
        }


class ProjectScopedDurableStore:
    """Durable project-scoped store with rollback and replay metadata."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ALLOWED_STORE_DIRS:
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def target_path(self, target_type: str, name: str) -> Path:
        if target_type not in TARGET_DIR_BY_TYPE:
            raise EvolutionPolicyBlocked(f"target_type_not_allowed:{target_type}")
        return self.root / TARGET_DIR_BY_TYPE[target_type] / name

    def write(self, envelope: dict[str, Any]) -> dict[str, Any]:
        self._validate_envelope(envelope)
        target = Path(envelope["target_path"]).resolve()
        before_exists = target.exists()
        before_bytes = target.read_bytes() if before_exists else b""
        sha_before = _hash_bytes(before_bytes) if before_exists else ""
        target.parent.mkdir(parents=True, exist_ok=True)
        payload_bytes = json.dumps(envelope["payload"], indent=2, sort_keys=True).encode("utf-8")
        target.write_bytes(payload_bytes)
        sha_after = _hash_file(target)
        rollback_pointer = self._write_rollback(envelope, before_exists, before_bytes, sha_before, sha_after)
        replay_ref = self._write_replay_manifest(envelope, sha_after)
        receipt = {
            "write_id": envelope["write_id"],
            "status": "PASS",
            "target_path": str(target),
            "sha256_before": sha_before,
            "sha256_after": sha_after,
            "rollback_pointer": str(rollback_pointer),
            "replay_manifest_ref": str(replay_ref),
            "policy_violations": [],
            "authorized_by": envelope["authorized_by"],
            "target_scope": envelope["target_scope"],
            "production_activation": envelope["production_activation"],
            "receipt_hash": "",
        }
        receipt["receipt_hash"] = _json_hash(receipt)
        self._append_ledger(envelope, receipt)
        self._write_posthoc_report(envelope, receipt)
        return receipt

    def quarantine(self, candidate: dict[str, Any], policy: AutonomousICMEvolutionPolicy) -> dict[str, Any]:
        review = EvolutionReview("AUTONOMOUS_QUARANTINE", "Quarantine", "negative_transfer_detected", True)
        envelope = policy.build_envelope(candidate, review, self)
        return self.write(envelope)

    def rollback(self, rollback_pointer: str | Path) -> dict[str, Any]:
        pointer = Path(rollback_pointer).resolve()
        self._ensure_under_root(pointer)
        rollback_payload = json.loads(pointer.read_text(encoding="utf-8"))
        target = Path(rollback_payload["target_path"]).resolve()
        self._ensure_under_root(target)
        if rollback_payload["before_exists"]:
            target.write_bytes(bytes.fromhex(rollback_payload["before_content_hex"]))
            status = "ROLLED_BACK"
        elif target.exists():
            target.unlink()
            status = "ROLLED_BACK"
        else:
            status = "ROLLED_BACK"
        restored_hash = _hash_file(target)
        return {
            "write_id": rollback_payload["write_id"],
            "status": status,
            "target_path": str(target),
            "restored_sha256": restored_hash,
            "matches_before_hash": restored_hash == rollback_payload["sha256_before"],
        }

    def replay(self, receipt: dict[str, Any]) -> dict[str, Any]:
        target = Path(receipt["target_path"]).resolve()
        self._ensure_under_root(target)
        current_hash = _hash_file(target)
        return {
            "write_id": receipt["write_id"],
            "replay_status": "PASS" if current_hash == receipt["sha256_after"] else "FAIL",
            "current_sha256": current_hash,
            "expected_sha256": receipt["sha256_after"],
        }

    def inventory(self) -> list[dict[str, Any]]:
        rows = []
        for path in sorted(self.root.rglob("*")):
            if path.is_file():
                rows.append({
                    "relative_path": path.relative_to(self.root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _hash_file(path),
                })
        return rows

    def _validate_envelope(self, envelope: dict[str, Any]) -> None:
        if envelope.get("authorized_by") != POLICY_OWNER:
            raise EvolutionPolicyBlocked("write_not_authorized_by_kernel_icm_evolution_policy")
        if envelope.get("target_scope") != "project_scoped_durable":
            raise EvolutionPolicyBlocked("target_scope_not_project_scoped")
        if envelope.get("production_activation") is not False:
            raise EvolutionPolicyBlocked("production_activation_forbidden")
        if not envelope.get("rollback_required"):
            raise EvolutionPolicyBlocked("rollback_required")
        target = Path(envelope.get("target_path", "")).resolve()
        self._ensure_under_root(target)
        relative_parts = target.relative_to(self.root).parts
        if not relative_parts or relative_parts[0] not in ALLOWED_STORE_DIRS:
            raise EvolutionPolicyBlocked("target_subpath_not_allowed")

    def _ensure_under_root(self, path: Path) -> None:
        if path != self.root and self.root not in path.parents:
            raise EvolutionPolicyBlocked(f"path_outside_project_scoped_store:{path}")

    def _write_rollback(
        self,
        envelope: dict[str, Any],
        before_exists: bool,
        before_bytes: bytes,
        sha_before: str,
        sha_after: str,
    ) -> Path:
        pointer = self.root / "rollback" / f"{_hash_bytes(envelope['write_id'].encode('utf-8'))[:16]}.rollback.json"
        payload = {
            "write_id": envelope["write_id"],
            "target_path": envelope["target_path"],
            "before_exists": before_exists,
            "before_content_hex": before_bytes.hex(),
            "sha256_before": sha_before,
            "sha256_after": sha_after,
        }
        pointer.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return pointer

    def _write_replay_manifest(self, envelope: dict[str, Any], sha_after: str) -> Path:
        ref = self.root / "replay_manifest" / f"{_hash_bytes(envelope['write_id'].encode('utf-8'))[:16]}.replay.json"
        payload = {
            "write_id": envelope["write_id"],
            "target_path": envelope["target_path"],
            "sha256_after": sha_after,
            "evidence_refs": envelope.get("evidence_refs", []),
            "accept_decision_ref": envelope.get("accept_decision_ref", ""),
        }
        ref.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return ref

    def _append_ledger(self, envelope: dict[str, Any], receipt: dict[str, Any]) -> None:
        ledger = self.root / "evolution_ledger" / "evolution_ledger.jsonl"
        entry = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "write_id": envelope["write_id"],
            "decision_id": envelope["decision_id"],
            "target_type": envelope["target_type"],
            "target_path": envelope["target_path"],
            "receipt_hash": receipt["receipt_hash"],
            "kernel_policy_owner": POLICY_OWNER,
        }
        with ledger.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    def _write_posthoc_report(self, envelope: dict[str, Any], receipt: dict[str, Any]) -> None:
        report = self.root / "human_posthoc_reports" / f"{_hash_bytes(envelope['write_id'].encode('utf-8'))[:16]}.posthoc.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "write_id": envelope["write_id"],
            "target_type": envelope["target_type"],
            "target_path": envelope["target_path"],
            "status": receipt["status"],
            "rollback_pointer": receipt["rollback_pointer"],
            "human_role": "posthoc_review_rollback_scope_escalation",
        }
        report.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def reset_store(root: str | Path) -> ProjectScopedDurableStore:
    path = Path(root)
    if path.exists():
        shutil.rmtree(path)
    return ProjectScopedDurableStore(path)
