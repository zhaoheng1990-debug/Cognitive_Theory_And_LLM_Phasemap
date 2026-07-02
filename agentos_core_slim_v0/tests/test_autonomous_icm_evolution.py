import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentos_kernel import AutonomousICMEvolutionPolicy, ProjectScopedDurableStore


def ups_candidate():
    return {
        "candidate_id": "ups_accept_operator_memory",
        "research_line": "UtilityPolicySelector",
        "status": "ACCEPT_READY_WITH_EVIDENCE",
        "evidence_refs": [
            "SR_series",
            "UPS_validation",
            "reuse_gain_records",
            "negative_transfer_audit",
        ],
        "accept_decision_ref": "UPS_ACCEPTDecisionRecord",
        "scope": "AgentOS project runtime policy selection",
        "negative_transfer_risk": "bounded",
        "future_cbit_gain": "positive",
        "replayable_evidence": True,
    }


def test_ups_accept_triggers_project_scoped_operator_memory_write(tmp_path):
    policy = AutonomousICMEvolutionPolicy()
    store = ProjectScopedDurableStore(tmp_path / "project_scoped_icm_store")
    review = policy.review(ups_candidate())

    envelope = policy.build_envelope(ups_candidate(), review, store)
    receipt = store.write(envelope)

    assert review.decision == "AUTONOMOUS_PROJECT_OPERATORMEMORY_WRITE"
    assert receipt["status"] == "PASS"
    assert "operator_memory" in receipt["target_path"]
    payload = json.loads(Path(receipt["target_path"]).read_text(encoding="utf-8"))
    assert payload["operator_id"] == "UtilityPolicySelectorOperator"
    assert receipt["authorized_by"] == "AgentOSKernel.ICMEvolutionPolicy"
    assert receipt["production_activation"] is False


def test_evidence_weak_user_positive_remains_no_write(tmp_path):
    policy = AutonomousICMEvolutionPolicy()
    store = ProjectScopedDurableStore(tmp_path / "project_scoped_icm_store")
    candidate = {
        "candidate_id": "user_positive_weak",
        "research_line": "UserPraisedHeuristic",
        "status": "USER_POSITIVE_CANDIDATE",
        "user_value_signal": "positive",
        "evidence_refs": [],
        "scope": "project_scoped",
        "negative_transfer_risk": "low",
        "future_cbit_gain": "positive",
    }

    review = policy.review(candidate)

    assert review.decision == "BLOCK_WRITE_INSUFFICIENT_EVIDENCE"
    assert not any((store.root / "memory_units").iterdir())
    assert not any((store.root / "operator_memory").iterdir())


def test_negative_transfer_candidate_is_quarantined(tmp_path):
    policy = AutonomousICMEvolutionPolicy()
    store = ProjectScopedDurableStore(tmp_path / "project_scoped_icm_store")
    candidate = {
        "candidate_id": "negative_transfer_case",
        "research_line": "RiskyReusePattern",
        "status": "ACCEPT_READY_WITH_EVIDENCE",
        "evidence_refs": ["replay_case"],
        "scope": "project_scoped",
        "negative_transfer_detected": True,
        "future_cbit_gain": "positive",
    }
    review = policy.review(candidate)
    receipt = store.quarantine(candidate, policy)

    assert review.decision == "AUTONOMOUS_QUARANTINE"
    assert receipt["status"] == "PASS"
    assert "negative_transfer_quarantine" in receipt["target_path"]


def test_replay_and_rollback_for_project_scoped_write(tmp_path):
    policy = AutonomousICMEvolutionPolicy()
    store = ProjectScopedDurableStore(tmp_path / "project_scoped_icm_store")
    candidate = ups_candidate()
    review = policy.review(candidate)
    receipt = store.write(policy.build_envelope(candidate, review, store))

    replay = store.replay(receipt)
    rollback = store.rollback(receipt["rollback_pointer"])

    assert replay["replay_status"] == "PASS"
    assert rollback["status"] == "ROLLED_BACK"
    assert rollback["matches_before_hash"] is True
    assert not Path(receipt["target_path"]).exists()


def test_global_or_production_write_is_blocked(tmp_path):
    store = ProjectScopedDurableStore(tmp_path / "project_scoped_icm_store")
    target = tmp_path / "global_icm" / "bad.json"
    envelope = {
        "write_id": "bad",
        "decision_id": "bad",
        "authorized_by": "AgentOSKernel.ICMEvolutionPolicy",
        "target_scope": "global_production",
        "target_type": "OperatorMemory",
        "target_path": str(target),
        "payload": {},
        "rollback_required": True,
        "production_activation": True,
    }

    try:
        store.write(envelope)
    except Exception as exc:
        assert "target_scope_not_project_scoped" in str(exc)
    else:
        raise AssertionError("global production write should be blocked")
