from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .objects import (
    HarnessEnvelope,
    HarnessReceipt,
    ICMMetabolismPolicyDecision,
    ProviderAdvisoryReceipt,
    RuntimeState,
    TaskEnvelope,
)


class ProviderAugmentationGateway:
    """Local advisory stub. It never owns final policy decisions."""

    def advise(self, task: TaskEnvelope, candidate_strength: str = "strong") -> ProviderAdvisoryReceipt:
        action = "PROMOTE_TO_DURABLE_ICM_CONTROLLED" if candidate_strength == "strong" else "REQUEST_MORE_VALIDATION"
        confidence = 0.82 if candidate_strength == "strong" else 0.45
        return ProviderAdvisoryReceipt(
            advisory_id=f"advisory:{task.task_id}",
            recommended_action=action,
            confidence=confidence,
            rationale="Local deterministic advisory stub; Kernel must adjudicate.",
            advisory_only=True,
        )


class TypedHarness:
    """Typed no-effect harness. Harness success is not a Kernel decision."""

    def execute(self, envelope: HarnessEnvelope) -> HarnessReceipt:
        if not envelope.no_external_effect:
            return HarnessReceipt(
                receipt_id=f"receipt:{envelope.envelope_id}",
                envelope_id=envelope.envelope_id,
                success=False,
                no_external_effect=False,
                note="Rejected: external effects are outside the portable dev-RC subset.",
            )
        return HarnessReceipt(
            receipt_id=f"receipt:{envelope.envelope_id}",
            envelope_id=envelope.envelope_id,
            success=True,
            no_external_effect=True,
            note="Typed envelope accepted for local no-effect portable replay.",
        )


@dataclass
class AgentOSKernel:
    owner: str = "AgentOSKernel.ICMMetabolismPolicy"
    state: RuntimeState = RuntimeState.INTAKE
    ledger: list[dict[str, Any]] = field(default_factory=list)

    def intake(self, task: TaskEnvelope) -> RuntimeState:
        if task.production_release:
            raise ValueError("production_release must remain false in portable dev-RC")
        self.state = RuntimeState.PROBLEM_REGISTERED
        self.ledger.append({"event": "intake", "task_id": task.task_id, "scope": task.scope})
        return self.state

    def record_advisory(self, advisory: ProviderAdvisoryReceipt) -> RuntimeState:
        if not advisory.advisory_only:
            raise ValueError("provider advisory must be advisory-only")
        self.state = RuntimeState.ADVISORY_RECEIVED
        self.ledger.append({"event": "advisory", "advisory_id": advisory.advisory_id})
        return self.state

    def record_harness_receipt(self, receipt: HarnessReceipt) -> RuntimeState:
        if not receipt.no_external_effect:
            raise ValueError("harness receipt indicates external effect")
        self.state = RuntimeState.HARNESS_RECEIPT_RECEIVED
        self.ledger.append({"event": "harness_receipt", "receipt_id": receipt.receipt_id})
        return self.state

    def decide_icm_metabolism(
        self,
        task: TaskEnvelope,
        advisory: ProviderAdvisoryReceipt,
        evidence_strength: str = "strong",
        negative_transfer: bool = False,
    ) -> ICMMetabolismPolicyDecision:
        if negative_transfer:
            final_action = "QUARANTINE_NEGATIVE_TRANSFER"
            followed = advisory.recommended_action == final_action
            override = None if followed else "Kernel quarantine overrides non-quarantine advisory."
        elif evidence_strength == "strong":
            final_action = "PROMOTE_TO_DURABLE_ICM_CONTROLLED"
            followed = advisory.recommended_action == final_action
            override = None if followed else "Kernel controlled-write threshold selected promotion."
        else:
            final_action = "REQUEST_MORE_VALIDATION"
            followed = advisory.recommended_action == final_action
            override = None if followed else "Weak evidence cannot authorize controlled durable write."
        decision = ICMMetabolismPolicyDecision(
            decision_id=f"kernel-decision:{task.task_id}",
            kernel_owner=self.owner,
            final_action=final_action,
            advisory_ref=advisory.advisory_id,
            advisory_followed=followed,
            override_reason=override,
            write_permission_scope="portable_dev_rc_controlled_icm_sandbox",
            production_release=False,
        )
        self.state = RuntimeState.KERNEL_DECIDED
        self.ledger.append({"event": "kernel_decision", "decision_id": decision.decision_id, "action": final_action})
        return decision
