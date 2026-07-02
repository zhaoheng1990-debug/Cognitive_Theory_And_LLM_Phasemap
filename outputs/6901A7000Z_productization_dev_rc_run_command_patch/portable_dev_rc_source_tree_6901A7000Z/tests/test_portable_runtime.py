from pathlib import Path
import shutil

from agentos_core_slim_portable import (
    AgentOSKernel,
    ControlledICMStore,
    HarnessEnvelope,
    ProviderAugmentationGateway,
    RuntimeState,
    TaskEnvelope,
    TypedHarness,
    replay_truthfulness_matrix,
)


def local_tmp_path(name: str) -> Path:
    path = Path(".pytest_tmp") / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def test_source_package_imports() -> None:
    import agentos_core_slim_portable

    assert agentos_core_slim_portable.AgentOSKernel


def test_state_machine_smoke() -> None:
    kernel = AgentOSKernel()
    task = TaskEnvelope(task_id="state", objective="state machine smoke")
    assert kernel.intake(task) == RuntimeState.PROBLEM_REGISTERED
    advisory = ProviderAugmentationGateway().advise(task)
    assert kernel.record_advisory(advisory) == RuntimeState.ADVISORY_RECEIVED
    receipt = TypedHarness().execute(HarnessEnvelope("env:state", task.task_id, "local_no_effect", {}))
    assert kernel.record_harness_receipt(receipt) == RuntimeState.HARNESS_RECEIPT_RECEIVED


def test_kernel_policy_ownership() -> None:
    task = TaskEnvelope(task_id="policy", objective="policy owner")
    advisory = ProviderAugmentationGateway().advise(task)
    decision = AgentOSKernel().decide_icm_metabolism(task, advisory, evidence_strength="strong")
    assert decision.kernel_owner == "AgentOSKernel.ICMMetabolismPolicy"
    assert decision.final_action == "PROMOTE_TO_DURABLE_ICM_CONTROLLED"
    assert decision.production_release is False


def test_llm_advisory_only_cannot_directly_decide() -> None:
    task = TaskEnvelope(task_id="weak", objective="weak evidence")
    advisory = ProviderAugmentationGateway().advise(task, candidate_strength="strong")
    decision = AgentOSKernel().decide_icm_metabolism(task, advisory, evidence_strength="weak")
    assert advisory.advisory_only is True
    assert advisory.recommended_action == "PROMOTE_TO_DURABLE_ICM_CONTROLLED"
    assert decision.final_action == "REQUEST_MORE_VALIDATION"
    assert decision.advisory_followed is False


def test_harness_envelope_receipt_no_external_effect() -> None:
    receipt = TypedHarness().execute(HarnessEnvelope("env:harness", "task", "local_no_effect", {"x": 1}))
    assert receipt.success is True
    assert receipt.no_external_effect is True


def test_controlled_write_rollback_replay() -> None:
    tmp_path = local_tmp_path("test_controlled_write_rollback")
    task = TaskEnvelope(task_id="write", objective="controlled write")
    advisory = ProviderAugmentationGateway().advise(task)
    decision = AgentOSKernel().decide_icm_metabolism(task, advisory, evidence_strength="strong")
    store = ControlledICMStore(tmp_path / "controlled_icm_store")
    write_receipt = store.write_memory_unit(decision, {"content": "portable memory"})
    assert (tmp_path / "controlled_icm_store" / write_receipt.relative_path).exists()
    rollback = store.rollback(write_receipt)
    assert rollback.post_rollback_absent is True
    replay = store.replay_verify(decision, write_receipt, rollback)
    assert replay.deterministic_match is True
    assert replay.production_release is False


def test_no_external_mutation_and_replay_truthfulness() -> None:
    tmp_path = local_tmp_path("test_no_external_mutation")
    store = ControlledICMStore(tmp_path / "controlled_icm_store")
    try:
        store._target("../outside.json")
        raise AssertionError("out-of-scope write was not blocked")
    except ValueError:
        pass
    rows = replay_truthfulness_matrix()
    assert any(row["standalone_portable_dev_rc_source_tree_replay"] is True for row in rows)
    assert all(row["production_release"] is False for row in rows)
