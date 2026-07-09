import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentos_kernel import (
    DeepSeekSemanticBridge,
    ProviderRequirementPolicy,
    ProviderSemanticBridge,
    ProviderSemanticRuntimeBlocked,
    RUNTIME_COGNITIVE_RESPONSIBILITIES,
    SemanticJudgmentReceipt,
)


def test_provider_required_operation_blocks_without_semantic_receipt():
    policy = ProviderRequirementPolicy()

    review = policy.review("TemporalSRO")

    assert review.status == "BLOCKED"
    assert review.provider_receipt_required is True
    assert review.runtime_authority_retained is True
    assert review.reason == "provider_backing_required_for_runtime_semantic_judgment"


def test_provider_required_operation_accepts_matching_receipt():
    policy = ProviderRequirementPolicy()
    receipt = ProviderSemanticBridge().build_mock_receipt(
        "TemporalSRO",
        "Identify constraint field for a VCOS runtime task.",
        source_refs=["source://seed-pack"],
    )

    review = policy.require("TemporalSRO", receipt)

    assert review.status == "PASS"
    assert review.reason == "provider_backed_runtime_judgment_present"
    assert review.runtime_authority_retained is True
    assert receipt["final_decision_owner"] == "AgentOSKernel"
    assert receipt["provider_final_decision_owner"] is False
    assert receipt["accepted_state_written"] is False
    assert set(receipt["runtime_responsibilities"]) == {
        "GoalAndConstraintManagement",
        "BoundaryGovernance",
        "MetaRuleApplication",
    }
    assert receipt["receipt_hash"]


def test_provider_backed_layer_supports_runtime_cognitive_responsibilities():
    audit = ProviderRequirementPolicy().audit_runtime_cognition_support()

    assert audit.status == "PASS"
    assert audit.reason == "provider_backed_runtime_cognition_supported"
    assert audit.final_authority_retained is True
    assert audit.missing_provider_operations == {}
    assert set(audit.responsibility_coverage) == set(RUNTIME_COGNITIVE_RESPONSIBILITIES)
    assert "GoalAndConstraintManagement" in audit.responsibility_coverage
    assert "FinalCandidateStateJudgment" in audit.responsibility_coverage


def test_provider_required_operation_rejects_mismatched_receipt_type():
    policy = ProviderRequirementPolicy()
    receipt = ProviderSemanticBridge().build_mock_receipt(
        "EvidenceClaimSupport",
        "Does the source support the claim?",
    )

    review = policy.review("TemporalSRO", receipt)

    assert review.status == "BLOCKED"
    assert review.reason == "provider_receipt_judgment_type_mismatch"


def test_runtime_native_operation_must_not_be_provider_overridden():
    policy = ProviderRequirementPolicy()
    receipt = ProviderSemanticBridge().build_mock_receipt(
        "EvidenceClaimSupport",
        "Does the source support the claim?",
    )

    review = policy.review("HashInventory", receipt)

    assert review.status == "BLOCKED"
    assert review.provider_receipt_required is False
    assert review.reason == "runtime_native_boundary_must_not_be_provider_overridden"


def test_runtime_native_deterministic_operation_passes_without_provider_receipt():
    review = ProviderRequirementPolicy().require("HashInventory")

    assert review.status == "PASS"
    assert review.reason == "runtime_native_deterministic_operation"


def test_semantic_receipt_validation_rejects_provider_final_decision_owner():
    receipt = ProviderSemanticBridge().build_mock_receipt(
        "UtilityPolicySelection",
        "Choose next action.",
    )
    receipt["provider_final_decision_owner"] = True

    with pytest.raises(ProviderSemanticRuntimeBlocked) as exc:
        SemanticJudgmentReceipt.validate(receipt)

    assert "provider_must_not_be_final_decision_owner" in str(exc.value)


def test_semantic_receipt_validation_rejects_provider_final_candidate_state():
    receipt = ProviderSemanticBridge().build_mock_receipt(
        "UtilityPolicySelection",
        "Choose next candidate state.",
    )
    receipt["judgment"]["decision"] = "accepted"

    with pytest.raises(ProviderSemanticRuntimeBlocked) as exc:
        SemanticJudgmentReceipt.validate(receipt)

    assert "provider_receipt_must_not_make_final_candidate_state" in str(exc.value)


def test_semantic_receipt_validation_requires_runtime_responsibility_mapping():
    receipt = ProviderSemanticBridge().build_mock_receipt(
        "EvidenceClaimSupport",
        "Does the source support the claim?",
    )
    receipt["runtime_responsibilities"] = []

    with pytest.raises(ProviderSemanticRuntimeBlocked) as exc:
        SemanticJudgmentReceipt.validate(receipt)

    assert "provider_receipt_runtime_responsibility_required" in str(exc.value)


def test_semantic_receipt_validation_rejects_provider_promotion_authorization():
    receipt = ProviderSemanticBridge().build_mock_receipt(
        "RetentionCandidateEvaluation",
        "Evaluate whether a memory candidate is reusable.",
    )
    receipt["scope"]["promotion_allowed"] = True

    with pytest.raises(ProviderSemanticRuntimeBlocked) as exc:
        SemanticJudgmentReceipt.validate(receipt)

    assert "provider_receipt_must_not_authorize_promotion" in str(exc.value)


def test_deepseek_bridge_uses_deepseek_env_key_without_exposing_secret(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-value")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    bridge = DeepSeekSemanticBridge()

    assert bridge.api_key == "secret-value"
    assert bridge.provider_name == "deepseek"
    assert bridge.base_url == "https://api.deepseek.com"


def test_deepseek_bridge_wraps_provider_json_as_semantic_receipt(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-value")

    class FixtureDeepSeekBridge(DeepSeekSemanticBridge):
        def _post_chat_completion(self, prompt):
            return json.dumps(
                {
                    "judgment": {"decision": "candidate", "support": "partial"},
                    "confidence": 0.74,
                    "rationale": "Evidence partially supports the claim.",
                    "unsupported_parts": ["funding date"],
                    "risk_flags": ["freshness_unknown"],
                    "cbit_gain_estimate": {"direction": "positive", "reason": "narrows scope"},
                    "scope": {"mode": "candidate", "promotion_allowed": False},
                    "freshness": {"status": "unknown"},
                    "negative_transfer_risk": "low",
                }
            )

    receipt = FixtureDeepSeekBridge().judge(
        "EvidenceClaimSupport",
        "Claim: ExampleCo raised a Series A.",
        source_refs=["https://example.test/news"],
        evidence_text="ExampleCo announced new funding without disclosing the round.",
    )

    assert receipt["provider"] == "deepseek"
    assert receipt["judgment_type"] == "EvidenceClaimSupport"
    assert receipt["judgment"]["support"] == "partial"
    assert receipt["unsupported_parts"] == ["funding date"]
    assert receipt["cbit_gain_estimate"]["direction"] == "positive"
    assert receipt["final_decision_owner"] == "AgentOSKernel"


def test_deepseek_bridge_blocks_without_any_api_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ProviderSemanticRuntimeBlocked) as exc:
        DeepSeekSemanticBridge().judge("TemporalSRO", "Identify task constraints.")

    assert "deepseek_api_key_missing" in str(exc.value)
