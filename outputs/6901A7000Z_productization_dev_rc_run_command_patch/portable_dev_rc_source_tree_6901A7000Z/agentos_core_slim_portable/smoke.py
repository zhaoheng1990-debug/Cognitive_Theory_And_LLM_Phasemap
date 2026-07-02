from pathlib import Path
import tempfile

from agentos_core_slim_portable import (
    AgentOSKernel,
    ControlledICMStore,
    HarnessEnvelope,
    ProviderAugmentationGateway,
    TaskEnvelope,
    TypedHarness,
    replay_truthfulness_matrix,
)


def main() -> None:
    task = TaskEnvelope(task_id="portable-smoke", objective="verify portable dev-RC subset")
    kernel = AgentOSKernel()
    kernel.intake(task)
    advisory = ProviderAugmentationGateway().advise(task)
    kernel.record_advisory(advisory)
    receipt = TypedHarness().execute(HarnessEnvelope("envelope:portable-smoke", task.task_id, "local_no_effect", {}))
    kernel.record_harness_receipt(receipt)
    decision = kernel.decide_icm_metabolism(task, advisory, evidence_strength="strong")
    with tempfile.TemporaryDirectory() as tmp:
        store = ControlledICMStore(Path(tmp) / "controlled_icm_store")
        write_receipt = store.write_memory_unit(decision, {"content": "portable smoke memory unit"})
        rollback = store.rollback(write_receipt)
        replay = store.replay_verify(decision, write_receipt, rollback)

    print(f"kernel_owner = {decision.kernel_owner}")
    print(f"advisory_only = {advisory.advisory_only}")
    print(f"harness_no_external_effect = {receipt.no_external_effect}")
    print(f"replay_verified = {replay.deterministic_match}")
    print(f"production_release = {replay.production_release}")
    print(f"truthfulness_rows = {len(replay_truthfulness_matrix())}")


if __name__ == "__main__":
    main()
