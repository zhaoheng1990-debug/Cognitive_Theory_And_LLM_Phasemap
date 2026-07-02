from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .objects import ControlledWriteReceipt, ICMMetabolismPolicyDecision, ReplayVerification, RollbackReceipt


class ControlledICMStore:
    """Local sandbox store. It rejects paths outside its root."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ["memory_units", "operator_memory", "write_receipts", "rollback_receipts"]:
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def _target(self, relative_path: str) -> Path:
        target = (self.root / relative_path).resolve()
        if self.root != target and self.root not in target.parents:
            raise ValueError(f"out-of-scope write blocked: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def _receipt_path(self, folder: str, receipt_id: str) -> Path:
        digest = hashlib.sha256(receipt_id.encode("utf-8")).hexdigest()[:16]
        return self._target(f"{folder}/{digest}.json")

    def write_memory_unit(self, decision: ICMMetabolismPolicyDecision, payload: dict[str, Any]) -> ControlledWriteReceipt:
        if decision.kernel_owner != "AgentOSKernel.ICMMetabolismPolicy":
            raise ValueError("controlled write requires Kernel-owned decision")
        if decision.final_action != "PROMOTE_TO_DURABLE_ICM_CONTROLLED":
            raise ValueError("decision does not authorize controlled durable write")
        if decision.production_release:
            raise ValueError("production release writes are forbidden")
        rel = "memory_units/portable-memory-unit.json"
        target = self._target(rel)
        data = {"decision_id": decision.decision_id, "production_release": False, **payload}
        target.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        receipt = ControlledWriteReceipt(
            receipt_id=f"write-receipt:{decision.decision_id}",
            decision_id=decision.decision_id,
            relative_path=rel,
            sha256=digest,
            sandbox_root=str(self.root),
        )
        receipt_path = self._receipt_path("write_receipts", receipt.receipt_id)
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt.__dict__, indent=2, sort_keys=True), encoding="utf-8")
        return receipt

    def rollback(self, receipt: ControlledWriteReceipt) -> RollbackReceipt:
        target = self._target(receipt.relative_path)
        if target.exists():
            target.unlink()
        rollback = RollbackReceipt(
            rollback_id=f"rollback:{receipt.receipt_id}",
            original_receipt_id=receipt.receipt_id,
            relative_path=receipt.relative_path,
            post_rollback_absent=not target.exists(),
        )
        rollback_path = self._receipt_path("rollback_receipts", rollback.rollback_id)
        rollback_path.parent.mkdir(parents=True, exist_ok=True)
        rollback_path.write_text(json.dumps(rollback.__dict__, indent=2, sort_keys=True), encoding="utf-8")
        return rollback

    def replay_verify(self, decision: ICMMetabolismPolicyDecision, receipt: ControlledWriteReceipt, rollback: RollbackReceipt) -> ReplayVerification:
        return ReplayVerification(
            replay_id=f"replay:{decision.decision_id}",
            deterministic_match=(
                decision.kernel_owner == "AgentOSKernel.ICMMetabolismPolicy"
                and decision.final_action == "PROMOTE_TO_DURABLE_ICM_CONTROLLED"
                and rollback.post_rollback_absent
                and bool(receipt.sha256)
            ),
            production_release=False,
            evidence_refs=[decision.decision_id, receipt.receipt_id, rollback.rollback_id],
        )
