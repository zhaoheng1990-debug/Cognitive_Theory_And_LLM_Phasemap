from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class RuntimeState(str, Enum):
    INTAKE = "INTAKE"
    PROBLEM_REGISTERED = "PROBLEM_REGISTERED"
    ADVISORY_RECEIVED = "ADVISORY_RECEIVED"
    HARNESS_RECEIPT_RECEIVED = "HARNESS_RECEIPT_RECEIVED"
    KERNEL_DECIDED = "KERNEL_DECIDED"
    CONTROLLED_WRITE_READY = "CONTROLLED_WRITE_READY"
    ROLLBACK_VERIFIED = "ROLLBACK_VERIFIED"
    CLOSED = "CLOSED"


PolicyAction = Literal[
    "OBSERVE_ONLY",
    "REQUEST_MORE_VALIDATION",
    "QUARANTINE_NEGATIVE_TRANSFER",
    "PROMOTE_TO_DURABLE_ICM_CONTROLLED",
]


@dataclass(frozen=True)
class TaskEnvelope:
    task_id: str
    objective: str
    scope: str = "standalone_portable_dev_rc_local"
    production_release: bool = False


@dataclass(frozen=True)
class ProviderAdvisoryReceipt:
    advisory_id: str
    recommended_action: PolicyAction
    confidence: float
    rationale: str
    advisory_only: bool = True
    owner: str = "ProviderAugmentationGateway"


@dataclass(frozen=True)
class HarnessEnvelope:
    envelope_id: str
    task_id: str
    intent: str
    payload: dict[str, Any]
    no_external_effect: bool = True


@dataclass(frozen=True)
class HarnessReceipt:
    receipt_id: str
    envelope_id: str
    success: bool
    no_external_effect: bool
    note: str


@dataclass(frozen=True)
class ICMMetabolismPolicyDecision:
    decision_id: str
    kernel_owner: str
    final_action: PolicyAction
    advisory_ref: str
    advisory_followed: bool
    write_permission_scope: str
    production_release: bool = False
    override_reason: str | None = None


@dataclass(frozen=True)
class ControlledWriteReceipt:
    receipt_id: str
    decision_id: str
    relative_path: str
    sha256: str
    sandbox_root: str


@dataclass(frozen=True)
class RollbackReceipt:
    rollback_id: str
    original_receipt_id: str
    relative_path: str
    post_rollback_absent: bool


@dataclass(frozen=True)
class ReplayVerification:
    replay_id: str
    deterministic_match: bool
    production_release: bool = False
    evidence_refs: list[str] = field(default_factory=list)
