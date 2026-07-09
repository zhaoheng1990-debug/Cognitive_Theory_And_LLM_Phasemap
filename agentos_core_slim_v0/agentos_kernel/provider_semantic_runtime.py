"""Provider-backed semantic runtime contracts for AgentOS CoreSlim.

Providers support the runtime's semantic judgment path; they do not replace
the runtime. Provider-backed receipts give AgentOS auditable semantic evidence,
while Kernel policy owns final promotion, state transitions, safety boundaries,
replay, rollback, and accepted-registry writes.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any


PROVIDER_RUNTIME_OWNER = "AgentOSKernel.ProviderRequirementPolicy"

PROVIDER_REQUIRED_OPERATIONS = {
    "TemporalSRO",
    "StructuralRouting",
    "OperatorFiberRanking",
    "UtilityPolicySelection",
    "CbitGainEstimation",
    "RetentionCandidateEvaluation",
    "OperatorMemoryEquivalenceReview",
    "ValidityDriftWatch",
    "EvidenceClaimSupport",
    "EvidenceRelevance",
    "EntityEventExtraction",
    "DomainScopeSynthesis",
    "NextIterationActionSelection",
}

RUNTIME_COGNITIVE_RESPONSIBILITIES = {
    "GoalAndConstraintManagement": (
        "TemporalSRO",
        "UtilityPolicySelection",
        "NextIterationActionSelection",
    ),
    "BoundaryGovernance": (
        "TemporalSRO",
        "StructuralRouting",
        "UtilityPolicySelection",
        "ValidityDriftWatch",
    ),
    "MetaRuleApplication": (
        "TemporalSRO",
        "StructuralRouting",
        "OperatorFiberRanking",
        "UtilityPolicySelection",
    ),
    "EvidenceOrganization": (
        "EvidenceRelevance",
        "EvidenceClaimSupport",
        "EntityEventExtraction",
        "DomainScopeSynthesis",
    ),
    "ConflictHandling": (
        "EvidenceClaimSupport",
        "ValidityDriftWatch",
        "DomainScopeSynthesis",
        "UtilityPolicySelection",
    ),
    "ReplayAndRollbackReasoning": (
        "OperatorMemoryEquivalenceReview",
        "ValidityDriftWatch",
        "RetentionCandidateEvaluation",
    ),
    "FinalCandidateStateJudgment": (
        "UtilityPolicySelection",
        "CbitGainEstimation",
        "RetentionCandidateEvaluation",
        "NextIterationActionSelection",
    ),
}

RUNTIME_NATIVE_DETERMINISTIC_OPERATIONS = {
    "JsonSchemaValidation",
    "RequiredFieldValidation",
    "EnumValidation",
    "HashInventory",
    "Replay",
    "Rollback",
    "AbsenceCheck",
    "CapabilityEnvelopeValidation",
    "AllowedPathValidation",
    "ForbiddenActionHardStop",
    "SafetyKernelHardStop",
    "ArtifactPackaging",
    "ManifestGeneration",
    "ReturnPackPackaging",
    "StateMachineTransitionValidation",
    "ProviderReceiptExistenceCheck",
    "SourceLinkPersistenceCheck",
}

MECHANICAL_LOCAL_OPERATIONS = RUNTIME_NATIVE_DETERMINISTIC_OPERATIONS

FORBIDDEN_PROVIDER_FINAL_DECISIONS = {
    "accept",
    "accepted",
    "approve",
    "approved",
    "final",
    "promote",
    "promoted",
}

REQUIRED_RECEIPT_FIELDS = {
    "receipt_id",
    "judgment_type",
    "provider",
    "model",
    "schema_version",
    "input_refs",
    "input_hashes",
    "source_refs",
    "source_hashes",
    "claim_or_task",
    "runtime_responsibilities",
    "judgment",
    "confidence",
    "rationale",
    "unsupported_parts",
    "risk_flags",
    "cbit_gain_estimate",
    "scope",
    "freshness",
    "negative_transfer_risk",
    "created_at",
    "final_decision_owner",
}


class ProviderSemanticRuntimeBlocked(RuntimeError):
    """Raised when provider-backed semantic runtime policy fails closed."""


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _hash_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def sha256_path(path: str | os.PathLike[str]) -> str:
    target = Path(path)
    h = sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _responsibilities_for_operation(operation: str) -> list[str]:
    return [
        responsibility
        for responsibility, operations in RUNTIME_COGNITIVE_RESPONSIBILITIES.items()
        if operation in operations
    ]


@dataclass(frozen=True)
class ProviderRequirementReview:
    status: str
    operation: str
    reason: str
    provider_receipt_required: bool
    runtime_authority_retained: bool = True


@dataclass(frozen=True)
class ProviderBackedRuntimeAudit:
    status: str
    reason: str
    responsibility_coverage: dict[str, tuple[str, ...]]
    missing_provider_operations: dict[str, tuple[str, ...]]
    final_authority_retained: bool


class SemanticJudgmentReceipt:
    """Factory and validator for provider-backed semantic judgment receipts."""

    schema_version = "agentos.semantic_judgment_receipt.v0_1"

    @classmethod
    def build(
        cls,
        *,
        judgment_type: str,
        provider: str,
        model: str,
        claim_or_task: str,
        judgment: dict[str, Any],
        confidence: float | int | str,
        rationale: str,
        input_refs: list[str] | None = None,
        source_refs: list[str] | None = None,
        input_hashes: list[dict[str, str]] | None = None,
        source_hashes: list[dict[str, str]] | None = None,
        unsupported_parts: list[str] | None = None,
        risk_flags: list[str] | None = None,
        cbit_gain_estimate: dict[str, Any] | None = None,
        scope: dict[str, Any] | None = None,
        freshness: dict[str, Any] | None = None,
        negative_transfer_risk: str = "unknown",
        raw_provider_response_hash: str = "",
        prompt_hash: str = "",
        runtime_responsibilities: list[str] | None = None,
    ) -> dict[str, Any]:
        receipt = {
            "receipt_id": "",
            "judgment_type": judgment_type,
            "provider": provider,
            "model": model,
            "schema_version": cls.schema_version,
            "input_refs": input_refs or [],
            "input_hashes": input_hashes or [],
            "source_refs": source_refs or [],
            "source_hashes": source_hashes or [],
            "claim_or_task": claim_or_task,
            "runtime_responsibilities": runtime_responsibilities or _responsibilities_for_operation(judgment_type),
            "judgment": judgment,
            "confidence": confidence,
            "rationale": rationale,
            "unsupported_parts": unsupported_parts or [],
            "risk_flags": risk_flags or [],
            "cbit_gain_estimate": cbit_gain_estimate or {"direction": "unknown", "estimate": "provider_not_quantified"},
            "scope": scope or {"mode": "candidate", "promotion_allowed": False},
            "freshness": freshness or {"status": "unknown"},
            "negative_transfer_risk": negative_transfer_risk,
            "created_at": _utc_now(),
            "raw_provider_response_hash": raw_provider_response_hash,
            "prompt_hash": prompt_hash,
            "final_decision_owner": "AgentOSKernel",
            "provider_final_decision_owner": False,
            "accepted_state_written": False,
            "receipt_hash": "",
        }
        receipt["receipt_id"] = f"sjr-{_json_hash(receipt)[:16]}"
        receipt["receipt_hash"] = _json_hash(receipt)
        cls.validate(receipt)
        return receipt

    @classmethod
    def validate(cls, receipt: dict[str, Any]) -> None:
        missing = sorted(REQUIRED_RECEIPT_FIELDS - set(receipt))
        if missing:
            raise ProviderSemanticRuntimeBlocked(f"semantic_receipt_missing_fields:{','.join(missing)}")
        if receipt.get("schema_version") != cls.schema_version:
            raise ProviderSemanticRuntimeBlocked("semantic_receipt_schema_version_mismatch")
        if receipt.get("final_decision_owner") != "AgentOSKernel":
            raise ProviderSemanticRuntimeBlocked("semantic_receipt_final_decision_owner_must_be_kernel")
        if receipt.get("provider_final_decision_owner") is not False:
            raise ProviderSemanticRuntimeBlocked("provider_must_not_be_final_decision_owner")
        if receipt.get("accepted_state_written") is not False:
            raise ProviderSemanticRuntimeBlocked("provider_receipt_must_not_write_accepted_state")
        if not isinstance(receipt.get("judgment"), dict):
            raise ProviderSemanticRuntimeBlocked("semantic_receipt_judgment_must_be_object")
        responsibilities = receipt.get("runtime_responsibilities")
        if receipt.get("judgment_type") in PROVIDER_REQUIRED_OPERATIONS:
            if not isinstance(responsibilities, list) or not responsibilities:
                raise ProviderSemanticRuntimeBlocked("provider_receipt_runtime_responsibility_required")
            unknown_responsibilities = set(responsibilities) - set(RUNTIME_COGNITIVE_RESPONSIBILITIES)
            if unknown_responsibilities:
                raise ProviderSemanticRuntimeBlocked("provider_receipt_unknown_runtime_responsibility")
        decision = str(receipt["judgment"].get("decision", "")).lower()
        if decision in FORBIDDEN_PROVIDER_FINAL_DECISIONS:
            raise ProviderSemanticRuntimeBlocked("provider_receipt_must_not_make_final_candidate_state")
        scope = receipt.get("scope")
        if isinstance(scope, dict) and scope.get("promotion_allowed") is True:
            raise ProviderSemanticRuntimeBlocked("provider_receipt_must_not_authorize_promotion")
        if not receipt.get("provider") or not receipt.get("model"):
            raise ProviderSemanticRuntimeBlocked("semantic_receipt_provider_and_model_required")
        if not receipt.get("claim_or_task"):
            raise ProviderSemanticRuntimeBlocked("semantic_receipt_claim_or_task_required")


class ProviderRequirementPolicy:
    """Kernel gate for provider-supported runtime semantic judgments."""

    owner = PROVIDER_RUNTIME_OWNER
    final_decision_owner = "AgentOSKernel"

    def audit_runtime_cognition_support(self) -> ProviderBackedRuntimeAudit:
        missing = {
            responsibility: tuple(op for op in operations if op not in PROVIDER_REQUIRED_OPERATIONS)
            for responsibility, operations in RUNTIME_COGNITIVE_RESPONSIBILITIES.items()
        }
        missing = {responsibility: operations for responsibility, operations in missing.items() if operations}
        deterministic_boundary_present = {
            "Replay",
            "Rollback",
            "StateMachineTransitionValidation",
            "CapabilityEnvelopeValidation",
            "SafetyKernelHardStop",
            "ProviderReceiptExistenceCheck",
        }.issubset(RUNTIME_NATIVE_DETERMINISTIC_OPERATIONS)
        final_authority_retained = self.final_decision_owner == "AgentOSKernel" and deterministic_boundary_present
        status = "PASS" if not missing and final_authority_retained else "BLOCKED"
        reason = "provider_backed_runtime_cognition_supported" if status == "PASS" else "provider_backed_runtime_cognition_gap"
        return ProviderBackedRuntimeAudit(
            status=status,
            reason=reason,
            responsibility_coverage=dict(RUNTIME_COGNITIVE_RESPONSIBILITIES),
            missing_provider_operations=missing,
            final_authority_retained=final_authority_retained,
        )

    def review(self, operation: str, provider_receipt: dict[str, Any] | None = None) -> ProviderRequirementReview:
        if operation in PROVIDER_REQUIRED_OPERATIONS:
            if provider_receipt is None:
                return ProviderRequirementReview(
                    "BLOCKED",
                    operation,
                    "provider_backing_required_for_runtime_semantic_judgment",
                    True,
                )
            SemanticJudgmentReceipt.validate(provider_receipt)
            if provider_receipt.get("judgment_type") != operation:
                return ProviderRequirementReview("BLOCKED", operation, "provider_receipt_judgment_type_mismatch", True)
            return ProviderRequirementReview("PASS", operation, "provider_backed_runtime_judgment_present", True)
        if operation in RUNTIME_NATIVE_DETERMINISTIC_OPERATIONS:
            if provider_receipt is not None:
                return ProviderRequirementReview(
                    "BLOCKED",
                    operation,
                    "runtime_native_boundary_must_not_be_provider_overridden",
                    False,
                )
            return ProviderRequirementReview("PASS", operation, "runtime_native_deterministic_operation", False)
        return ProviderRequirementReview("BLOCKED", operation, "unknown_runtime_operation", False)

    def require(self, operation: str, provider_receipt: dict[str, Any] | None = None) -> ProviderRequirementReview:
        review = self.review(operation, provider_receipt)
        if review.status != "PASS":
            raise ProviderSemanticRuntimeBlocked(review.reason)
        return review


class ProviderSemanticBridge:
    """Base bridge for semantic providers.

    Subclasses may call network APIs. This base class supplies deterministic
    mock receipts for local tests and provider-unavailable flows.
    """

    provider_name = "mock"
    model = "mock-semantic-provider-v0"

    def build_mock_receipt(
        self,
        judgment_type: str,
        claim_or_task: str,
        *,
        judgment: dict[str, Any] | None = None,
        source_refs: list[str] | None = None,
        input_refs: list[str] | None = None,
    ) -> dict[str, Any]:
        return SemanticJudgmentReceipt.build(
            judgment_type=judgment_type,
            provider=self.provider_name,
            model=self.model,
            claim_or_task=claim_or_task,
            judgment=judgment or {"decision": "candidate", "support": "mock_provider_backed"},
            confidence=0.5,
            rationale="Mock provider receipt for runtime contract validation.",
            input_refs=input_refs,
            source_refs=source_refs,
            input_hashes=[{"ref": ref, "sha256": _hash_text(ref)} for ref in (input_refs or [])],
            source_hashes=[{"ref": ref, "sha256": _hash_text(ref)} for ref in (source_refs or [])],
            cbit_gain_estimate={"direction": "unknown", "estimate": "mock"},
            scope={"mode": "candidate", "promotion_allowed": False},
            freshness={"status": "not_evaluated"},
            negative_transfer_risk="unknown",
            raw_provider_response_hash=_hash_text("mock"),
            prompt_hash=_hash_text(claim_or_task),
        )


class DeepSeekSemanticBridge(ProviderSemanticBridge):
    """DeepSeek-backed semantic provider bridge using an OpenAI-compatible API."""

    provider_name = "deepseek"
    default_base_url = "https://api.deepseek.com"
    default_model = "deepseek-v4-flash"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: int = 60,
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL") or self.default_base_url).rstrip("/")
        self.model = model or os.getenv("DEEPSEEK_MODEL") or self.default_model
        self.timeout_seconds = timeout_seconds

    def judge(
        self,
        judgment_type: str,
        claim_or_task: str,
        *,
        input_refs: list[str] | None = None,
        source_refs: list[str] | None = None,
        evidence_text: str = "",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise ProviderSemanticRuntimeBlocked("deepseek_api_key_missing")
        if judgment_type not in PROVIDER_REQUIRED_OPERATIONS:
            raise ProviderSemanticRuntimeBlocked(f"unsupported_provider_judgment_type:{judgment_type}")

        prompt = self._build_prompt(judgment_type, claim_or_task, evidence_text, context or {})
        raw = self._post_chat_completion(prompt)
        parsed = self._parse_provider_json(raw)
        return SemanticJudgmentReceipt.build(
            judgment_type=judgment_type,
            provider=self.provider_name,
            model=self.model,
            claim_or_task=claim_or_task,
            judgment=parsed.get("judgment", {}),
            confidence=parsed.get("confidence", "unknown"),
            rationale=str(parsed.get("rationale", "")),
            input_refs=input_refs,
            source_refs=source_refs,
            input_hashes=[{"ref": ref, "sha256": _hash_text(ref)} for ref in (input_refs or [])],
            source_hashes=[{"ref": ref, "sha256": _hash_text(ref)} for ref in (source_refs or [])],
            unsupported_parts=parsed.get("unsupported_parts", []),
            risk_flags=parsed.get("risk_flags", []),
            cbit_gain_estimate=parsed.get("cbit_gain_estimate", {"direction": "unknown"}),
            scope=parsed.get("scope", {"mode": "candidate", "promotion_allowed": False}),
            freshness=parsed.get("freshness", {"status": "provider_not_explicit"}),
            negative_transfer_risk=str(parsed.get("negative_transfer_risk", "unknown")),
            raw_provider_response_hash=_hash_text(raw),
            prompt_hash=_hash_text(prompt),
        )

    @staticmethod
    def _build_prompt(judgment_type: str, claim_or_task: str, evidence_text: str, context: dict[str, Any]) -> str:
        schema = {
            "judgment": {"decision": "string", "support": "supported|partial|unsupported|candidate"},
            "confidence": "number_or_string",
            "rationale": "short explanation",
            "unsupported_parts": ["strings"],
            "risk_flags": ["strings"],
            "cbit_gain_estimate": {"direction": "positive|neutral|negative|unknown", "reason": "string"},
            "scope": {"mode": "candidate", "promotion_allowed": False},
            "freshness": {"status": "fresh|stale|unknown", "reason": "string"},
            "negative_transfer_risk": "none|low|medium|high|unknown",
        }
        return json.dumps(
            {
                "instruction": (
                    "Return strict JSON only. You are a semantic support provider for the AgentOS "
                    "runtime. You provide evidence-grounded semantic judgment receipts that support "
                    "runtime cognition. You do not replace AgentOS runtime authority, make final "
                    "Kernel decisions, write accepted state, or authorize actions."
                ),
                "judgment_type": judgment_type,
                "claim_or_task": claim_or_task,
                "evidence_text": evidence_text,
                "context": context,
                "required_output_schema": schema,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def _post_chat_completion(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return strict JSON only for AgentOS provider-backed runtime judgment receipts.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        start = time.time()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise ProviderSemanticRuntimeBlocked(f"deepseek_provider_call_failed:{exc}") from exc
        if time.time() - start > self.timeout_seconds:
            raise ProviderSemanticRuntimeBlocked("deepseek_provider_call_timeout")
        data = json.loads(body)
        return data["choices"][0]["message"]["content"]

    @staticmethod
    def _parse_provider_json(raw: str) -> dict[str, Any]:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderSemanticRuntimeBlocked("provider_returned_non_json") from exc
        if not isinstance(parsed, dict):
            raise ProviderSemanticRuntimeBlocked("provider_returned_non_object")
        return parsed
