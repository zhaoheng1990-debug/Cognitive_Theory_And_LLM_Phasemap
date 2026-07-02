"""Bounded Arbor-to-Codex tool bridge for CoreSlim.

The bridge is intentionally a Harness Plane executor. It cannot authorize its
own calls and it never emits Kernel-owned final decisions such as SRO,
NextAction, or ICM metabolism policy decisions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any


ALLOWED_CAPABILITIES = {
    "READ_ARTIFACT",
    "WRITE_LOCAL_PATCH",
    "RUN_PYTEST",
    "RUN_SCRIPT",
    "PACKAGE_ZIP",
    "GENERATE_HASH_INVENTORY",
    "INSPECT_MANIFEST",
    "RUN_REPLAY",
    "ROLLBACK_LOCAL_WRITE",
}

FORBIDDEN_CAPABILITIES = {
    "SEND_EMAIL",
    "WIRE_TRANSFER",
    "EXTERNAL_API_MUTATION",
    "GIT_PUSH",
    "PRODUCTION_DEPLOY",
    "GLOBAL_MEMORY_WRITE",
    "GLOBAL_ICM_WRITE",
    "UNBOUNDED_WEB_ACTION",
    "LEGAL_SIGNATURE",
    "INVESTMENT_COMMITMENT",
}

WRITE_CAPABILITIES = {"WRITE_LOCAL_PATCH", "PACKAGE_ZIP", "ROLLBACK_LOCAL_WRITE"}
EXECUTION_CAPABILITIES = {"RUN_PYTEST", "RUN_SCRIPT", "RUN_REPLAY"}


class ToolBridgeBlocked(RuntimeError):
    """Raised internally when a tool call must fail closed."""


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class CodexToolBridge:
    """Local bounded executor callable from Arbor under Kernel authorization."""

    bridge_id = "codex_tool_bridge_local_v0_1"
    owner = "HarnessPlane"
    kernel_policy_owner = "AgentOSKernel"
    mode = "local_bounded_execution"
    external_mutation_allowed = False
    final_decision_owner = False

    def __init__(self, workspace_root: str | os.PathLike[str]):
        self.workspace_root = Path(workspace_root).resolve()

    def execute(self, tool_call_envelope: dict[str, Any], harness_dispatch_envelope: dict[str, Any] | None) -> dict[str, Any]:
        start = time.time()
        try:
            self._validate_authorization(tool_call_envelope, harness_dispatch_envelope)
            capability = tool_call_envelope["tool_capability"]
            result = self._execute_capability(capability, tool_call_envelope)
            status = "PASS"
            violations: list[str] = []
        except ToolBridgeBlocked as exc:
            capability = str(tool_call_envelope.get("tool_capability", "UNKNOWN"))
            result = {
                "outputs": [],
                "artifacts": [],
                "hashes": [],
                "stdout_summary": "",
                "stderr_summary": str(exc),
                "mutation_summary": "blocked_before_execution",
                "rollback_pointer": "",
            }
            status = "BLOCKED"
            violations = [str(exc)]
        except Exception as exc:  # deterministic failure receipt, not Kernel adjudication
            capability = str(tool_call_envelope.get("tool_capability", "UNKNOWN"))
            result = {
                "outputs": [],
                "artifacts": [],
                "hashes": [],
                "stdout_summary": "",
                "stderr_summary": f"{type(exc).__name__}: {exc}",
                "mutation_summary": "failed_during_local_execution",
                "rollback_pointer": "",
            }
            status = "FAIL"
            violations = []

        receipt = {
            "tool_call_id": tool_call_envelope.get("tool_call_id", ""),
            "parent_harness_envelope_id": tool_call_envelope.get("parent_harness_envelope_id", ""),
            "status": status,
            "executed_capability": capability,
            "outputs": result["outputs"],
            "artifacts": result["artifacts"],
            "hashes": result["hashes"],
            "stdout_summary": result["stdout_summary"][:4000],
            "stderr_summary": result["stderr_summary"][:4000],
            "mutation_summary": result["mutation_summary"],
            "rollback_pointer": result["rollback_pointer"],
            "policy_violations": violations,
            "bridge_owner": self.owner,
            "kernel_policy_owner": self.kernel_policy_owner,
            "codex_tool_bridge_final_decision_owner": False,
            "elapsed_seconds": round(time.time() - start, 6),
        }
        receipt["receipt_hash"] = _json_hash(receipt)
        return receipt

    def _validate_authorization(self, envelope: dict[str, Any], dispatch: dict[str, Any] | None) -> None:
        if not dispatch:
            raise ToolBridgeBlocked("missing_kernel_authorized_harness_dispatch_envelope")
        if dispatch.get("authorized_by") != "AgentOSKernel.HarnessDispatchPolicy":
            raise ToolBridgeBlocked("dispatch_not_authorized_by_agentos_kernel")
        capability = envelope.get("tool_capability")
        if capability in FORBIDDEN_CAPABILITIES:
            raise ToolBridgeBlocked(f"forbidden_capability:{capability}")
        if capability not in ALLOWED_CAPABILITIES:
            raise ToolBridgeBlocked(f"unknown_or_unregistered_capability:{capability}")
        if capability not in set(dispatch.get("authorized_capabilities", [])):
            raise ToolBridgeBlocked(f"capability_not_in_kernel_authorized_envelope:{capability}")
        if envelope.get("parent_harness_envelope_id") != dispatch.get("harness_dispatch_envelope_id"):
            raise ToolBridgeBlocked("parent_harness_envelope_mismatch")
        if capability in WRITE_CAPABILITIES and not envelope.get("rollback_required", False):
            raise ToolBridgeBlocked("write_capability_requires_rollback_metadata")
        for raw in envelope.get("allowed_paths", []):
            self._ensure_workspace_path(raw)
        for raw in envelope.get("forbidden_paths", []):
            path = Path(raw).resolve()
            if path == self.workspace_root or self.workspace_root in path.parents:
                continue
            raise ToolBridgeBlocked("forbidden_path_outside_workspace_scope")

    def _ensure_workspace_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.workspace_root / path
        resolved = path.resolve()
        if resolved != self.workspace_root and self.workspace_root not in resolved.parents:
            raise ToolBridgeBlocked(f"path_outside_workspace:{raw_path}")
        return resolved

    def _path_allowed_by_envelope(self, path: Path, envelope: dict[str, Any]) -> None:
        allowed_roots = [self._ensure_workspace_path(p) for p in envelope.get("allowed_paths", [])]
        if not allowed_roots:
            raise ToolBridgeBlocked("missing_allowed_paths")
        resolved = path.resolve()
        if not any(resolved == root or root in resolved.parents for root in allowed_roots):
            raise ToolBridgeBlocked(f"path_not_under_allowed_paths:{path}")
        for raw in envelope.get("forbidden_paths", []):
            forbidden = self._ensure_workspace_path(raw)
            if resolved == forbidden or forbidden in resolved.parents:
                raise ToolBridgeBlocked(f"path_under_forbidden_path:{path}")

    def _execute_capability(self, capability: str, envelope: dict[str, Any]) -> dict[str, Any]:
        if capability == "READ_ARTIFACT":
            return self._read_artifact(envelope)
        if capability == "WRITE_LOCAL_PATCH":
            return self._write_local_patch(envelope)
        if capability == "RUN_PYTEST":
            return self._run_pytest(envelope)
        if capability == "RUN_SCRIPT":
            return self._run_script(envelope)
        if capability == "PACKAGE_ZIP":
            return self._package_zip(envelope)
        if capability == "GENERATE_HASH_INVENTORY":
            return self._generate_hash_inventory(envelope)
        if capability == "INSPECT_MANIFEST":
            return self._inspect_manifest(envelope)
        if capability == "RUN_REPLAY":
            return self._run_replay(envelope)
        if capability == "ROLLBACK_LOCAL_WRITE":
            return self._rollback_local_write(envelope)
        raise ToolBridgeBlocked(f"unimplemented_capability:{capability}")

    def _read_artifact(self, envelope: dict[str, Any]) -> dict[str, Any]:
        hashes = []
        outputs = []
        for raw in envelope.get("input_artifacts", []):
            path = self._ensure_workspace_path(raw)
            self._path_allowed_by_envelope(path, envelope)
            if not path.is_file():
                raise ToolBridgeBlocked(f"artifact_not_found:{raw}")
            hashes.append({"path": str(path), "sha256": sha256_path(path), "bytes": path.stat().st_size})
            outputs.append({"path": str(path), "preview": path.read_text(encoding="utf-8", errors="replace")[:500]})
        return self._result(outputs=outputs, artifacts=[], hashes=hashes, stdout="read_artifact_completed")

    def _write_local_patch(self, envelope: dict[str, Any]) -> dict[str, Any]:
        patch = envelope.get("patch") or {}
        path = self._ensure_workspace_path(patch.get("path", ""))
        self._path_allowed_by_envelope(path, envelope)
        before_exists = path.exists()
        before_content = path.read_text(encoding="utf-8", errors="replace") if before_exists else None
        before_hash = sha256_path(path) if before_exists else ""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(patch.get("content", "")), encoding="utf-8")
        rollback_path = path.with_suffix(path.suffix + ".rollback.json")
        rollback_payload = {
            "path": str(path),
            "before_exists": before_exists,
            "before_content": before_content,
            "before_hash": before_hash,
            "after_hash": sha256_path(path),
        }
        rollback_path.write_text(json.dumps(rollback_payload, indent=2, sort_keys=True), encoding="utf-8")
        return self._result(
            outputs=[{"path": str(path)}],
            artifacts=[str(path), str(rollback_path)],
            hashes=[{"path": str(path), "sha256": sha256_path(path)}],
            stdout="write_local_patch_completed",
            mutation=f"wrote_local_patch:{path}",
            rollback=str(rollback_path),
        )

    def _run_pytest(self, envelope: dict[str, Any]) -> dict[str, Any]:
        target = self._ensure_workspace_path(envelope.get("pytest_target", "."))
        self._path_allowed_by_envelope(target, envelope)
        timeout = int(envelope.get("budget", {}).get("max_runtime_seconds", 30))
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(target), "-q"],
            cwd=str(self.workspace_root),
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        status = "pytest_passed" if proc.returncode == 0 else f"pytest_failed_return_code_{proc.returncode}"
        return self._result(outputs=[{"return_code": proc.returncode}], artifacts=[], hashes=[], stdout=f"{status}\n{proc.stdout}", stderr=proc.stderr)

    def _run_script(self, envelope: dict[str, Any]) -> dict[str, Any]:
        script = self._ensure_workspace_path(envelope.get("script_path", ""))
        self._path_allowed_by_envelope(script, envelope)
        if script.suffix != ".py":
            raise ToolBridgeBlocked("run_script_only_allows_python_fixture_scripts")
        timeout = int(envelope.get("budget", {}).get("max_runtime_seconds", 30))
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(self.workspace_root),
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return self._result(outputs=[{"return_code": proc.returncode}], artifacts=[], hashes=[], stdout=proc.stdout, stderr=proc.stderr)

    def _package_zip(self, envelope: dict[str, Any]) -> dict[str, Any]:
        output = self._ensure_workspace_path(envelope.get("zip_output", "toolbridge_package.zip"))
        self._path_allowed_by_envelope(output, envelope)
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
            for raw in envelope.get("input_artifacts", []):
                path = self._ensure_workspace_path(raw)
                self._path_allowed_by_envelope(path, envelope)
                if path.is_file():
                    zf.write(path, path.relative_to(self.workspace_root).as_posix())
        return self._result(
            outputs=[{"path": str(output)}],
            artifacts=[str(output)],
            hashes=[{"path": str(output), "sha256": sha256_path(output)}],
            stdout="package_zip_completed",
            mutation=f"wrote_zip:{output}",
            rollback=str(output),
        )

    def _generate_hash_inventory(self, envelope: dict[str, Any]) -> dict[str, Any]:
        root = self._ensure_workspace_path(envelope.get("inventory_root", "."))
        self._path_allowed_by_envelope(root, envelope)
        output = self._ensure_workspace_path(envelope.get("inventory_output", "hash_inventory.csv"))
        self._path_allowed_by_envelope(output, envelope)
        rows = []
        for path in sorted(root.rglob("*")):
            if path.is_file():
                rows.append({"relative_path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_path(path)})
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["relative_path", "bytes", "sha256"])
            writer.writeheader()
            writer.writerows(rows)
        return self._result(outputs=[{"row_count": len(rows)}], artifacts=[str(output)], hashes=[{"path": str(output), "sha256": sha256_path(output)}], stdout="hash_inventory_completed")

    def _inspect_manifest(self, envelope: dict[str, Any]) -> dict[str, Any]:
        manifest = self._ensure_workspace_path(envelope.get("manifest_path", ""))
        self._path_allowed_by_envelope(manifest, envelope)
        data = json.loads(manifest.read_text(encoding="utf-8"))
        return self._result(outputs=[{"manifest_path": str(manifest), "top_level_keys": sorted(data.keys())}], artifacts=[], hashes=[{"path": str(manifest), "sha256": sha256_path(manifest)}], stdout="inspect_manifest_completed")

    def _run_replay(self, envelope: dict[str, Any]) -> dict[str, Any]:
        receipts = []
        for raw in envelope.get("input_artifacts", []):
            path = self._ensure_workspace_path(raw)
            self._path_allowed_by_envelope(path, envelope)
            receipt = json.loads(path.read_text(encoding="utf-8"))
            receipts.append({"path": str(path), "receipt_hash": receipt.get("receipt_hash", _json_hash(receipt))})
        replay_hash = _json_hash(receipts)
        return self._result(outputs=[{"receipt_count": len(receipts), "replay_hash": replay_hash}], artifacts=[], hashes=[{"replay_hash": replay_hash}], stdout="run_replay_completed")

    def _rollback_local_write(self, envelope: dict[str, Any]) -> dict[str, Any]:
        rollback_path = self._ensure_workspace_path(envelope.get("rollback_pointer", ""))
        self._path_allowed_by_envelope(rollback_path, envelope)
        payload = json.loads(rollback_path.read_text(encoding="utf-8"))
        target = self._ensure_workspace_path(payload["path"])
        self._path_allowed_by_envelope(target, envelope)
        if payload["before_exists"]:
            target.write_text(payload["before_content"], encoding="utf-8")
            restored_hash = sha256_path(target)
            if restored_hash != payload["before_hash"]:
                raise RuntimeError("rollback_hash_mismatch")
        elif target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            restored_hash = ""
        else:
            restored_hash = ""
        return self._result(outputs=[{"path": str(target), "restored_hash": restored_hash}], artifacts=[str(target)], hashes=[{"path": str(target), "sha256": restored_hash}], stdout="rollback_local_write_completed", mutation=f"rolled_back:{target}")

    @staticmethod
    def _result(
        outputs: list[Any],
        artifacts: list[str],
        hashes: list[Any],
        stdout: str,
        stderr: str = "",
        mutation: str = "none",
        rollback: str = "",
    ) -> dict[str, Any]:
        return {
            "outputs": outputs,
            "artifacts": artifacts,
            "hashes": hashes,
            "stdout_summary": stdout,
            "stderr_summary": stderr,
            "mutation_summary": mutation,
            "rollback_pointer": rollback,
        }
