import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentos_kernel import CodexToolBridge


def dispatch(*capabilities):
    return {
        "harness_dispatch_envelope_id": "hde-001",
        "authorized_by": "AgentOSKernel.HarnessDispatchPolicy",
        "requested_harness": "ArborHarness",
        "authorized_capabilities": list(capabilities),
        "kernel_final_decision_owner": True,
    }


def envelope(tmp_path: Path, capability: str, **extra):
    data = {
        "tool_call_id": f"tc-{capability.lower()}",
        "parent_harness_envelope_id": "hde-001",
        "requested_by": "ArborHarness",
        "authorized_by": "AgentOSKernel.HarnessDispatchPolicy",
        "tool_capability": capability,
        "input_artifacts": [],
        "allowed_paths": [str(tmp_path)],
        "forbidden_paths": [],
        "expected_outputs": [],
        "budget": {"max_runtime_seconds": 20, "max_output_bytes": 200000},
        "rollback_required": capability in {"WRITE_LOCAL_PATCH", "PACKAGE_ZIP", "ROLLBACK_LOCAL_WRITE"},
    }
    data.update(extra)
    return data


def test_read_artifact_passes_with_kernel_authorization(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("bounded read", encoding="utf-8")
    bridge = CodexToolBridge(tmp_path)

    receipt = bridge.execute(
        envelope(tmp_path, "READ_ARTIFACT", input_artifacts=[str(artifact)]),
        dispatch("READ_ARTIFACT"),
    )

    assert receipt["status"] == "PASS"
    assert receipt["codex_tool_bridge_final_decision_owner"] is False
    assert receipt["hashes"][0]["sha256"]


def test_unauthorized_capability_blocks_closed(tmp_path):
    bridge = CodexToolBridge(tmp_path)

    receipt = bridge.execute(
        envelope(tmp_path, "RUN_SCRIPT", script_path=str(tmp_path / "script.py")),
        dispatch("READ_ARTIFACT"),
    )

    assert receipt["status"] == "BLOCKED"
    assert "capability_not_in_kernel_authorized_envelope" in receipt["policy_violations"][0]


def test_forbidden_external_mutation_blocks(tmp_path):
    bridge = CodexToolBridge(tmp_path)

    receipt = bridge.execute(
        envelope(tmp_path, "EXTERNAL_API_MUTATION"),
        dispatch("EXTERNAL_API_MUTATION"),
    )

    assert receipt["status"] == "BLOCKED"
    assert "forbidden_capability" in receipt["policy_violations"][0]


def test_missing_kernel_dispatch_blocks(tmp_path):
    bridge = CodexToolBridge(tmp_path)

    receipt = bridge.execute(envelope(tmp_path, "READ_ARTIFACT"), None)

    assert receipt["status"] == "BLOCKED"
    assert "missing_kernel_authorized_harness_dispatch_envelope" in receipt["policy_violations"][0]


def test_write_rollback_and_replay_receipt(tmp_path):
    bridge = CodexToolBridge(tmp_path)
    target = tmp_path / "mutable.txt"
    target.write_text("before", encoding="utf-8")

    write_receipt = bridge.execute(
        envelope(
            tmp_path,
            "WRITE_LOCAL_PATCH",
            patch={"path": str(target), "content": "after"},
        ),
        dispatch("WRITE_LOCAL_PATCH"),
    )
    assert write_receipt["status"] == "PASS"
    assert target.read_text(encoding="utf-8") == "after"
    assert write_receipt["rollback_pointer"]

    receipt_file = tmp_path / "write_receipt.json"
    receipt_file.write_text(json.dumps(write_receipt), encoding="utf-8")
    replay_receipt = bridge.execute(
        envelope(tmp_path, "RUN_REPLAY", input_artifacts=[str(receipt_file)]),
        dispatch("RUN_REPLAY"),
    )
    assert replay_receipt["status"] == "PASS"
    assert replay_receipt["outputs"][0]["receipt_count"] == 1

    rollback_receipt = bridge.execute(
        envelope(
            tmp_path,
            "ROLLBACK_LOCAL_WRITE",
            rollback_pointer=write_receipt["rollback_pointer"],
        ),
        dispatch("ROLLBACK_LOCAL_WRITE"),
    )
    assert rollback_receipt["status"] == "PASS"
    assert target.read_text(encoding="utf-8") == "before"


def test_package_zip_and_manifest_inspection(tmp_path):
    bridge = CodexToolBridge(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"required_return_files": ["manifest.json"]}), encoding="utf-8")
    output_zip = tmp_path / "pack.zip"

    package_receipt = bridge.execute(
        envelope(
            tmp_path,
            "PACKAGE_ZIP",
            input_artifacts=[str(manifest)],
            zip_output=str(output_zip),
        ),
        dispatch("PACKAGE_ZIP"),
    )
    assert package_receipt["status"] == "PASS"
    assert output_zip.exists()

    inspect_receipt = bridge.execute(
        envelope(tmp_path, "INSPECT_MANIFEST", manifest_path=str(manifest)),
        dispatch("INSPECT_MANIFEST"),
    )
    assert inspect_receipt["status"] == "PASS"
    assert "required_return_files" in inspect_receipt["outputs"][0]["top_level_keys"]
