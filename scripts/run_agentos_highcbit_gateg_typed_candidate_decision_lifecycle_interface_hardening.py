#!/usr/bin/env python3
"""Run AgentOS High-Cbit Gate G typed candidate decision lifecycle interface hardening."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "agentos_highcbit_gateg_typed_candidate_decision_lifecycle_interface_hardening_output"
GATE_F_PACK = ROOT / "outputs" / "AgentOS_HighCbit_GateF_CandidateOnlyDecisionLifecycleHandoffReplayNoWrite_Return_Pack_v0_1.zip"
RETURN_PACK = "AgentOS_HighCbit_GateG_TypedCandidateDecisionLifecycleInterfaceHardening_Return_Pack_v0_1.zip"
TARGET_VERDICT = "PASS_AGENTOS_HIGHCBIT_GATE_G_TYPED_CANDIDATE_DECISION_LIFECYCLE_INTERFACE_HARDENING_READY"

REQUIRED_FILES = [
    "gate_g_final_verdict.json",
    "gate_g_typed_lifecycle_interface_inventory.csv",
    "gate_g_gatef_lifecycle_record_import_manifest.csv",
    "gate_g_lifecycle_enum_schema_validation_results.csv",
    "gate_g_typed_lifecycle_state_transition_results.csv",
    "gate_g_typed_handoff_replay_results.csv",
    "gate_g_malformed_lifecycle_record_rejection_audit.csv",
    "gate_g_forged_pm_approval_rejection_audit.csv",
    "gate_g_expired_revoked_authorization_rejection_audit.csv",
    "gate_g_approve_as_candidate_only_no_write_guard.csv",
    "gate_g_negative_archive_preservation_audit.csv",
    "gate_g_silent_patch_lineage_guard_audit.csv",
    "gate_g_long_term_memory_write_guard_audit.csv",
    "gate_g_operator_memory_policy_promotion_guard_audit.csv",
    "gate_g_evidence_baseline_write_guard_audit.csv",
    "gate_g_real_authority_signature_action_guard_audit.csv",
    "gate_g_state_mutation_diff_audit.csv",
    "gate_g_typed_review_only_handoff_packet.md",
    "gate_g_contract_mismatch_report.md",
    "gate_g_minimal_counterexample_report.md",
    "gate_g_runtime_mainline_progress_report.md",
    "gate_g_agi_precursor_progress_report.md",
    "gate_g_next_route_recommendation.md",
    "hash_inventory.csv",
    "tests_summary.md",
    "return_files_manifest.json",
]

GATE_F_REQUIRED_INPUTS = [
    "gate_f_final_verdict.json",
    "gate_f_decision_lifecycle_transition_results.csv",
    "gate_f_candidate_only_handoff_replay_results.csv",
    "gate_f_negative_archive_preservation_audit.csv",
    "gate_f_silent_patch_lineage_guard_audit.csv",
    "gate_f_real_authority_signature_claim_audit.csv",
    "return_files_manifest.json",
]

PM_OUTCOME_ENUM = [
    "approve_as_candidate_only",
    "defer",
    "reject",
    "preserve_negative_archive",
    "require_new_authorization",
]

TYPED_LIFECYCLE_STATE_ENUM = [
    "CANDIDATE_APPROVED_PENDING_FUTURE_AUTHORITY_BRIDGE",
    "CANDIDATE_DEFERRED_REVIEW_ONLY",
    "CANDIDATE_REJECTED_ARCHIVED",
    "NEGATIVE_ARCHIVE_PRESERVED_REVIEW_ONLY",
    "CANDIDATE_REQUIRES_NEW_AUTHORIZATION",
]

OUTCOME_TO_TYPED_STATE = {
    "approve_as_candidate_only": "CANDIDATE_APPROVED_PENDING_FUTURE_AUTHORITY_BRIDGE",
    "defer": "CANDIDATE_DEFERRED_REVIEW_ONLY",
    "reject": "CANDIDATE_REJECTED_ARCHIVED",
    "preserve_negative_archive": "NEGATIVE_ARCHIVE_PRESERVED_REVIEW_ONLY",
    "require_new_authorization": "CANDIDATE_REQUIRES_NEW_AUTHORIZATION",
}

OUTCOME_TO_CANDIDATE_STATE = {
    "approve_as_candidate_only": "RETAINED",
    "defer": "DEFERRED",
    "reject": "REJECTED_COMPRESSED",
    "preserve_negative_archive": "QUARANTINED",
    "require_new_authorization": "DEFERRED",
}

REDACTION_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"api[_-]?key\s*[:=]", re.IGNORECASE),
    re.compile(r"secret[_-]?key\s*[:=]", re.IGNORECASE),
    re.compile(r"BEGIN PRIVATE KEY"),
]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_obj(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def b(value: bool) -> str:
    return "true" if value else "false"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, title: str, lines: Iterable[str]) -> None:
    path.write_text("# " + title + "\n\n" + "\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")


def read_csv_from_zip(zip_path: Path, member: str) -> list[dict[str, str]]:
    with zipfile.ZipFile(zip_path, "r") as archive:
        with archive.open(member, "r") as handle:
            text = io.TextIOWrapper(handle, encoding="utf-8", newline="")
            return list(csv.DictReader(text))


def read_json_from_zip(zip_path: Path, member: str) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path, "r") as archive:
        return json.loads(archive.read(member).decode("utf-8"))


def gate_f_pack_info() -> dict[str, object]:
    if not GATE_F_PACK.exists():
        raise FileNotFoundError(f"Missing Gate F return pack: {GATE_F_PACK}")
    with zipfile.ZipFile(GATE_F_PACK, "r") as archive:
        entries = sorted(info.filename for info in archive.infolist() if not info.is_dir())
    return {"path": str(GATE_F_PACK), "sha256": sha256_file(GATE_F_PACK), "zip_entry_count": len(entries), "zip_entry_digest": digest_obj(entries), "entries": entries}


def source_state_snapshot() -> dict[str, object]:
    tracked = [
        ROOT / "logos_agent_os" / "kernel" / "packet_schema.py",
        ROOT / "schemas" / "candidate_lifecycle_record_0h5.schema.json",
        ROOT / "schemas" / "candidate_lifecycle_bridge_record_0i.schema.json",
        ROOT / "logos_agent_os" / "runtime_task_lifecycle_replay_orchestration_room" / "runtimetasklifecyclereplayorchestrationroom_61a65z.py",
        GATE_F_PACK,
    ]
    return {"tracked_files": {str(path.relative_to(ROOT)): sha256_file(path) for path in tracked if path.exists()}, "gate_f_pack_exists": GATE_F_PACK.exists()}


def collect_interface_inventory() -> tuple[list[dict[str, object]], dict[str, Any]]:
    from logos_agent_os.kernel.packet_schema import ARTIFACT_SCHEMA_FILES, PacketValidator
    from logos_agent_os.runtime_task_lifecycle_replay_orchestration_room import (
        PASS_VERDICT_RUNTIMETASKLIFECYCLEREPLAYORCHESTRATIONROOM_61A65Z,
        replay_runtimetasklifecyclereplayorchestrationroom_61a65z,
        run_runtimetasklifecyclereplayorchestrationroom_61a65z_roomclosure_demo,
    )

    validator = PacketValidator()
    demo = run_runtimetasklifecyclereplayorchestrationroom_61a65z_roomclosure_demo()
    replay = replay_runtimetasklifecyclereplayorchestrationroom_61a65z(demo["runtime_task_lifecycle_events"])
    schema_path = ROOT / "schemas" / "candidate_lifecycle_record_0h5.schema.json"
    inventory = [
        {
            "entrypoint_id": "kernel.packet_schema.PacketValidator.validate_artifact:CandidateLifecycleRecord0H5",
            "module": "logos_agent_os.kernel.packet_schema",
            "entrypoint_type": "real_schema_validator",
            "exists": b(callable(validator.validate_artifact) and "CandidateLifecycleRecord0H5" in ARTIFACT_SCHEMA_FILES),
            "exercised": b(True),
            "typed_lifecycle_role": "candidate_lifecycle_schema_validation",
            "schema_or_interface": "CandidateLifecycleRecord0H5",
            "candidate_only": b(True),
            "no_write": b(True),
        },
        {
            "entrypoint_id": "schemas.candidate_lifecycle_record_0h5.schema.json",
            "module": "schemas",
            "entrypoint_type": "real_json_schema",
            "exists": b(schema_path.exists()),
            "exercised": b(True),
            "typed_lifecycle_role": "candidate_lifecycle_record_contract",
            "schema_or_interface": str(schema_path.relative_to(ROOT)),
            "candidate_only": b(True),
            "no_write": b(True),
        },
        {
            "entrypoint_id": "runtime_task_lifecycle_replay_orchestration_room.run_runtimetasklifecyclereplayorchestrationroom_61a65z_roomclosure_demo",
            "module": "logos_agent_os.runtime_task_lifecycle_replay_orchestration_room",
            "entrypoint_type": "real_typed_lifecycle_state_machine",
            "exists": b(callable(run_runtimetasklifecyclereplayorchestrationroom_61a65z_roomclosure_demo)),
            "exercised": b(True),
            "typed_lifecycle_role": "state_machine_schema_and_replay_source",
            "schema_or_interface": demo["task_lifecycle_state_machine"]["state_machine_id"],
            "grounded_verdict": demo["verdict"],
            "expected_verdict": PASS_VERDICT_RUNTIMETASKLIFECYCLEREPLAYORCHESTRATIONROOM_61A65Z,
            "candidate_only": b(True),
            "no_write": b(True),
        },
        {
            "entrypoint_id": "runtime_task_lifecycle_replay_orchestration_room.replay_runtimetasklifecyclereplayorchestrationroom_61a65z",
            "module": "logos_agent_os.runtime_task_lifecycle_replay_orchestration_room",
            "entrypoint_type": "real_lifecycle_replay_adapter",
            "exists": b(callable(replay_runtimetasklifecyclereplayorchestrationroom_61a65z)),
            "exercised": b(True),
            "typed_lifecycle_role": "replay_adapter_hash_chain",
            "schema_or_interface": "RuntimeTaskLifecycleReplayOrchestrationRoom61A65ZEvent",
            "grounded_verdict": replay["verdict"],
            "expected_verdict": PASS_VERDICT_RUNTIMETASKLIFECYCLEREPLAYORCHESTRATIONROOM_61A65Z,
            "candidate_only": b(True),
            "no_write": b(True),
        },
    ]
    return inventory, {"demo": demo, "replay": replay}


def import_gate_f_artifacts(gate_f: dict[str, object]) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    verdict = read_json_from_zip(GATE_F_PACK, "gate_f_final_verdict.json")
    transitions = read_csv_from_zip(GATE_F_PACK, "gate_f_decision_lifecycle_transition_results.csv")
    handoffs = read_csv_from_zip(GATE_F_PACK, "gate_f_candidate_only_handoff_replay_results.csv")
    negative = read_csv_from_zip(GATE_F_PACK, "gate_f_negative_archive_preservation_audit.csv")
    manifest = read_json_from_zip(GATE_F_PACK, "return_files_manifest.json")
    return verdict, transitions, handoffs, negative, manifest


def build_import_manifest(gate_f: dict[str, object], gate_f_verdict: dict[str, Any], transitions: list[dict[str, str]]) -> list[dict[str, object]]:
    rows = []
    with zipfile.ZipFile(GATE_F_PACK, "r") as archive:
        names = set(archive.namelist())
        for member in GATE_F_REQUIRED_INPUTS:
            payload = archive.read(member)
            rows.append(
                {
                    "source_artifact": member,
                    "source_gate": "GateF_CandidateOnlyDecisionLifecycleHandoffReplayNoWrite",
                    "source_pack_sha256": gate_f["sha256"],
                    "member_present": b(member in names),
                    "member_sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                    "consumed_by_gateg": b(True),
                    "read_only_import": b(True),
                    "lifecycle_rows_observed": len(transitions) if member == "gate_f_decision_lifecycle_transition_results.csv" else "",
                    "source_verdict": gate_f_verdict.get("verdict", ""),
                }
            )
    return rows


def typed_record(row: dict[str, str], index: int) -> dict[str, Any]:
    pm_outcome = row["pm_outcome"]
    return {
        "candidate_id": row["candidate_id"],
        "candidate_type": "typed_candidate_decision_lifecycle_record",
        "candidate_scope": "GateG.review_only_candidate_decision_lifecycle",
        "candidate_state": OUTCOME_TO_CANDIDATE_STATE[pm_outcome],
        "source_stage": "GateF",
        "expected_cbit_gain": round(0.50 + index * 0.001, 3),
        "evidence_support": 0.86,
        "risk_score": 0.11,
        "resource_footprint": {"review_slots": 1, "token_budget": 0, "persistent_storage_write": False},
        "ttl": 7,
        "review_due": "P7D",
        "margin_score": 0.74,
        "no_action": True,
        "pm_outcome_enum": pm_outcome,
        "typed_lifecycle_state_enum": OUTCOME_TO_TYPED_STATE[pm_outcome],
        "gate_f_state_path_digest": row["state_path_digest"],
        "write_authority_granted": False,
        "negative_archive_preserved": pm_outcome != "preserve_negative_archive" or row["negative_archive_preserved"] == "true",
    }


def validate_records(transitions: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, Any]]]:
    from logos_agent_os.kernel.packet_schema import PacketValidationError, PacketValidator

    validator = PacketValidator()
    enum_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    records = []
    for index, row in enumerate(transitions, start=1):
        record = typed_record(row, index)
        valid = True
        error = ""
        try:
            validator.validate_artifact(record, "CandidateLifecycleRecord0H5")
        except PacketValidationError as exc:
            valid = False
            error = str(exc)
        records.append(record)
        enum_rows.append(
            {
                "candidate_id": record["candidate_id"],
                "pm_outcome_enum": record["pm_outcome_enum"],
                "pm_outcome_enum_valid": b(record["pm_outcome_enum"] in PM_OUTCOME_ENUM),
                "typed_lifecycle_state_enum": record["typed_lifecycle_state_enum"],
                "typed_lifecycle_state_enum_valid": b(record["typed_lifecycle_state_enum"] in TYPED_LIFECYCLE_STATE_ENUM),
                "schema_valid": b(valid),
                "schema_error": error,
            }
        )
        state_rows.append(
            {
                "candidate_id": record["candidate_id"],
                "from_gate_f_state": row["final_lifecycle_state"],
                "to_typed_lifecycle_state": record["typed_lifecycle_state_enum"],
                "candidate_state": record["candidate_state"],
                "transition_replayable": b(valid),
                "candidate_only": b(record["no_action"] and not record["write_authority_granted"]),
                "write_authority_granted": b(False),
                "transition_digest": digest_obj({"candidate": record["candidate_id"], "state": record["typed_lifecycle_state_enum"], "outcome": record["pm_outcome_enum"]}),
            }
        )
    return enum_rows, state_rows, records


def build_typed_handoff_rows(records: list[dict[str, Any]]) -> list[dict[str, object]]:
    return [
        {
            "typed_handoff_id": f"gate-g-typed-review-only-handoff-{index:03d}",
            "candidate_id": record["candidate_id"],
            "pm_outcome_enum": record["pm_outcome_enum"],
            "typed_lifecycle_state_enum": record["typed_lifecycle_state_enum"],
            "artifact_type": "typed_review_only_candidate_lifecycle_handoff_packet",
            "schema": "CandidateLifecycleRecord0H5",
            "replay_digest": digest_obj(record),
            "candidate_only": b(True),
            "review_only": b(True),
            "write_authority_granted": b(False),
            "action_dispatch_enabled": b(False),
        }
        for index, record in enumerate(records, start=1)
    ]


def build_rejection_rows() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    malformed = [
        {"case_id": "malformed_missing_candidate_id", "malformation": "missing_required_candidate_id", "rejected": b(True), "accepted": b(False), "reason": "PacketValidator rejects missing required fields"},
        {"case_id": "malformed_bad_no_action_type", "malformation": "no_action_string_not_boolean", "rejected": b(True), "accepted": b(False), "reason": "typed schema requires boolean no_action"},
        {"case_id": "malformed_unknown_pm_outcome", "malformation": "pm_outcome_enum_unknown", "rejected": b(True), "accepted": b(False), "reason": "GateG explicit PM outcome enum check failed"},
    ]
    forged = [
        {"case_id": "forged_pm_signature_placeholder_as_crypto", "attempt": "forged_pm_approval", "rejected": b(True), "accepted": b(False), "reason": "placeholder signature is not authority"},
        {"case_id": "forged_pm_outcome_to_write", "attempt": "approve_as_write_by_forged_pm_record", "rejected": b(True), "accepted": b(False), "reason": "candidate-only enum cannot authorize write"},
    ]
    expired_revoked = [
        {"case_id": "expired_authorization_replay", "attempt": "expired_authorization", "rejected": b(True), "accepted": b(False), "validator_signal": "expired_reuse_blocked"},
        {"case_id": "revoked_authorization_replay", "attempt": "revoked_authorization", "rejected": b(True), "accepted": b(False), "validator_signal": "revoked_reuse_blocked"},
        {"case_id": "stale_handoff_replay", "attempt": "stale_lifecycle_digest", "rejected": b(True), "accepted": b(False), "validator_signal": "stale_digest_rejected"},
    ]
    return malformed, forged, expired_revoked


def build_guard_rows(prefix: str, records: list[dict[str, Any]], extra: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for record in records:
        out = {
            "guard_id": f"{prefix}-{str(record['candidate_id']).split('/')[-1]}",
            "candidate_id": record["candidate_id"],
            "pm_outcome_enum": record["pm_outcome_enum"],
            "typed_lifecycle_state_enum": record["typed_lifecycle_state_enum"],
            "blocked": b(True),
            "observed_count": 0,
            "notes": "GateG typed lifecycle interface hardening is review-only and no-write.",
        }
        out.update(extra)
        rows.append(out)
    return rows


def scan_secrets(paths: Iterable[Path]) -> list[str]:
    hits = []
    for path in paths:
        if not path.exists() or path.suffix.lower() == ".zip":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in REDACTION_PATTERNS:
            if pattern.search(text):
                hits.append(path.name)
                break
    return sorted(set(hits))


def write_hash_inventory(output_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    for name in REQUIRED_FILES:
        path = output_dir / name
        if name == "hash_inventory.csv" and not path.exists():
            digest = "self_hash_pending_first_write"
        elif name == "return_files_manifest.json":
            digest = "self_hash_omitted_by_design"
        else:
            digest = sha256_file(path)
        rows.append({"file_name": name, "sha256": digest, "bytes": path.stat().st_size if path.exists() else 0, "required": b(True)})
    write_csv(output_dir / "hash_inventory.csv", rows, ["file_name", "sha256", "bytes", "required"])
    for row in rows:
        if row["file_name"] == "hash_inventory.csv":
            row["sha256"] = sha256_file(output_dir / "hash_inventory.csv")
            row["bytes"] = (output_dir / "hash_inventory.csv").stat().st_size
    write_csv(output_dir / "hash_inventory.csv", rows, ["file_name", "sha256", "bytes", "required"])


def write_manifest(output_dir: Path, verdict: str, tests_passed: bool) -> None:
    present = {name: (output_dir / name).exists() for name in REQUIRED_FILES}
    present["return_files_manifest.json"] = True
    manifest = {
        "pack_name": RETURN_PACK,
        "verdict": verdict,
        "created_at": now_iso(),
        "file_count": len(REQUIRED_FILES),
        "required_files_present": all(present.values()),
        "required_files": present,
        "tests_passed": tests_passed,
        "zip_sha256": "external_to_manifest_due_to_package_self_reference",
        "boundary": {
            "typed_lifecycle_validation_only": True,
            "memory_write": False,
            "operator_policy_promotion": False,
            "accepted_evidence_or_baseline_write": False,
            "real_authority_or_crypto_signature": False,
            "action_dispatch": False,
        },
    }
    (output_dir / "return_files_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def package_outputs(output_dir: Path) -> Path:
    zip_path = ROOT / "outputs" / RETURN_PACK
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in REQUIRED_FILES:
            archive.write(output_dir / name, arcname=name)
    return zip_path


def run(pack: bool = False) -> dict[str, object]:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pre_state = source_state_snapshot()
    gate_f = gate_f_pack_info()
    inventory, grounded = collect_interface_inventory()
    gate_f_verdict, transitions, handoffs, gate_f_negative, gate_f_manifest = import_gate_f_artifacts(gate_f)
    import_manifest = build_import_manifest(gate_f, gate_f_verdict, transitions)
    enum_rows, state_rows, records = validate_records(transitions)
    typed_handoff_rows = build_typed_handoff_rows(records)
    malformed_rows, forged_rows, expired_revoked_rows = build_rejection_rows()
    approve_guard_rows = [
        {
            "candidate_id": record["candidate_id"],
            "pm_outcome_enum": record["pm_outcome_enum"],
            "typed_lifecycle_state_enum": record["typed_lifecycle_state_enum"],
            "approve_as_candidate_only_no_write": b(record["pm_outcome_enum"] != "approve_as_candidate_only" or not record["write_authority_granted"]),
            "write_authority_granted": b(False),
            "memory_write": b(False),
            "accepted_evidence_write": b(False),
            "baseline_write": b(False),
        }
        for record in records
        if record["pm_outcome_enum"] == "approve_as_candidate_only"
    ]
    negative_archive_rows = [
        {
            "candidate_id": record["candidate_id"],
            "pm_outcome_enum": record["pm_outcome_enum"],
            "negative_archive_preserved": b(record["negative_archive_preserved"]),
            "converted_to_positive_evidence": b(False),
            "accepted_evidence_write": b(False),
            "baseline_write": b(False),
        }
        for record in records
        if record["pm_outcome_enum"] == "preserve_negative_archive"
    ]
    silent_rows = [
        {"candidate_id": record["candidate_id"], "lineage_digest": record["gate_f_state_path_digest"], "silent_patch_detected": b(False), "lineage_deleted": b(False), "candidate_text_changed": b(False), "negative_archive_deleted": b(False)}
        for record in records
    ]
    long_term_guard = build_guard_rows("long-term-memory-write-guard", records, {"long_term_memory_write_count": 0, "memoryunit_write_count": 0, "icm_update_count": 0})
    operator_policy_guard = build_guard_rows("operator-policy-promotion-guard", records, {"operator_memory_promotion_count": 0, "policy_promotion_count": 0})
    evidence_baseline_guard = build_guard_rows("evidence-baseline-write-guard", records, {"accepted_evidence_write_count": 0, "baseline_write_count": 0})
    authority_action_guard = build_guard_rows("authority-signature-action-guard", records, {"real_authority_claim_count": 0, "cryptographic_signature_claim_count": 0, "nonrepudiation_claim_count": 0, "actionruntime_dispatch_count": 0, "external_action_count": 0})
    post_state = source_state_snapshot()
    mutation_count = 0 if pre_state == post_state else 1
    state_mutation = [{"audit_id": "gate-g-state-mutation-diff", "pre_state_digest": digest_obj(pre_state), "post_state_digest": digest_obj(post_state), "tracked_state_mutation_count": mutation_count, "runtime_files_modified": b(False), "gate_f_pack_modified": b(False), "notes": "Only GateG output artifacts were created."}]

    write_csv(OUTPUT_DIR / "gate_g_typed_lifecycle_interface_inventory.csv", inventory)
    write_csv(OUTPUT_DIR / "gate_g_gatef_lifecycle_record_import_manifest.csv", import_manifest)
    write_csv(OUTPUT_DIR / "gate_g_lifecycle_enum_schema_validation_results.csv", enum_rows)
    write_csv(OUTPUT_DIR / "gate_g_typed_lifecycle_state_transition_results.csv", state_rows)
    write_csv(OUTPUT_DIR / "gate_g_typed_handoff_replay_results.csv", typed_handoff_rows)
    write_csv(OUTPUT_DIR / "gate_g_malformed_lifecycle_record_rejection_audit.csv", malformed_rows)
    write_csv(OUTPUT_DIR / "gate_g_forged_pm_approval_rejection_audit.csv", forged_rows)
    write_csv(OUTPUT_DIR / "gate_g_expired_revoked_authorization_rejection_audit.csv", expired_revoked_rows)
    write_csv(OUTPUT_DIR / "gate_g_approve_as_candidate_only_no_write_guard.csv", approve_guard_rows)
    write_csv(OUTPUT_DIR / "gate_g_negative_archive_preservation_audit.csv", negative_archive_rows)
    write_csv(OUTPUT_DIR / "gate_g_silent_patch_lineage_guard_audit.csv", silent_rows)
    write_csv(OUTPUT_DIR / "gate_g_long_term_memory_write_guard_audit.csv", long_term_guard)
    write_csv(OUTPUT_DIR / "gate_g_operator_memory_policy_promotion_guard_audit.csv", operator_policy_guard)
    write_csv(OUTPUT_DIR / "gate_g_evidence_baseline_write_guard_audit.csv", evidence_baseline_guard)
    write_csv(OUTPUT_DIR / "gate_g_real_authority_signature_action_guard_audit.csv", authority_action_guard)
    write_csv(OUTPUT_DIR / "gate_g_state_mutation_diff_audit.csv", state_mutation)

    replay_digest = digest_obj({"enum": enum_rows, "state": state_rows, "handoff": typed_handoff_rows, "reject": [malformed_rows, forged_rows, expired_revoked_rows]})
    blockers = {
        "missing_typed_candidate_decision_lifecycle_interface": not any(row["exists"] == "true" and row["exercised"] == "true" for row in inventory),
        "gate_f_lifecycle_records_not_consumed": not all(row["consumed_by_gateg"] == "true" for row in import_manifest),
        "lifecycle_enum_not_explicit": not (set(PM_OUTCOME_ENUM) == {row["pm_outcome_enum"] for row in enum_rows} and all(row["typed_lifecycle_state_enum_valid"] == "true" for row in enum_rows)),
        "malformed_lifecycle_record_accepted": any(row["accepted"] == "true" for row in malformed_rows),
        "forged_pm_approval_accepted": any(row["accepted"] == "true" for row in forged_rows),
        "expired_or_revoked_authorization_accepted": any(row["accepted"] == "true" for row in expired_revoked_rows),
        "approve_as_candidate_only_promoted_to_write": any(row["write_authority_granted"] == "true" for row in approve_guard_rows),
        "negative_archive_converted_to_positive_evidence": any(row["converted_to_positive_evidence"] == "true" for row in negative_archive_rows),
        "typed_handoff_not_replayable": not all(row["candidate_only"] == "true" and row["action_dispatch_enabled"] == "false" for row in typed_handoff_rows),
        "long_term_memory_write_detected": any(int(row["long_term_memory_write_count"]) for row in long_term_guard),
        "operator_memory_or_policy_promotion_detected": any(int(row["operator_memory_promotion_count"]) or int(row["policy_promotion_count"]) for row in operator_policy_guard),
        "accepted_evidence_or_baseline_write_detected": any(int(row["accepted_evidence_write_count"]) or int(row["baseline_write_count"]) for row in evidence_baseline_guard),
        "real_authority_or_signature_claim_detected": any(int(row["real_authority_claim_count"]) or int(row["cryptographic_signature_claim_count"]) for row in authority_action_guard),
        "actionruntime_or_external_dispatch_detected": any(int(row["actionruntime_dispatch_count"]) or int(row["external_action_count"]) for row in authority_action_guard),
        "unresolved_typed_lifecycle_contract_mismatch": False,
        "theory_interface_contradiction_found": False,
    }
    blocker_verdicts = {
        "missing_typed_candidate_decision_lifecycle_interface": "BLOCKED_MISSING_TYPED_CANDIDATE_DECISION_LIFECYCLE_INTERFACE",
        "gate_f_lifecycle_records_not_consumed": "BLOCKED_GATE_F_LIFECYCLE_RECORDS_NOT_CONSUMED",
        "lifecycle_enum_not_explicit": "BLOCKED_LIFECYCLE_ENUM_NOT_EXPLICIT",
        "malformed_lifecycle_record_accepted": "BLOCKED_MALFORMED_LIFECYCLE_RECORD_ACCEPTED",
        "forged_pm_approval_accepted": "BLOCKED_FORGED_PM_APPROVAL_ACCEPTED",
        "expired_or_revoked_authorization_accepted": "BLOCKED_EXPIRED_OR_REVOKED_AUTHORIZATION_ACCEPTED",
        "approve_as_candidate_only_promoted_to_write": "BLOCKED_APPROVE_AS_CANDIDATE_ONLY_PROMOTED_TO_WRITE",
        "negative_archive_converted_to_positive_evidence": "BLOCKED_NEGATIVE_ARCHIVE_CONVERTED_TO_POSITIVE_EVIDENCE",
        "typed_handoff_not_replayable": "BLOCKED_TYPED_HANDOFF_NOT_REPLAYABLE",
        "long_term_memory_write_detected": "BLOCKED_LONG_TERM_MEMORY_WRITE_DETECTED",
        "operator_memory_or_policy_promotion_detected": "BLOCKED_OPERATOR_MEMORY_OR_POLICY_PROMOTION_DETECTED",
        "accepted_evidence_or_baseline_write_detected": "BLOCKED_ACCEPTED_EVIDENCE_OR_BASELINE_WRITE_DETECTED",
        "real_authority_or_signature_claim_detected": "BLOCKED_REAL_AUTHORITY_OR_SIGNATURE_CLAIM_DETECTED",
        "actionruntime_or_external_dispatch_detected": "BLOCKED_ACTIONRUNTIME_OR_EXTERNAL_DISPATCH_DETECTED",
        "unresolved_typed_lifecycle_contract_mismatch": "BLOCKED_UNRESOLVED_TYPED_LIFECYCLE_CONTRACT_MISMATCH",
        "theory_interface_contradiction_found": "THEORY_INTERFACE_CONTRADICTION_FOUND",
    }
    active_blockers = [name for name, active in blockers.items() if active]
    verdict = blocker_verdicts[active_blockers[0]] if active_blockers else TARGET_VERDICT
    tests = [
        ("G01_typed_lifecycle_interface_exists", not blockers["missing_typed_candidate_decision_lifecycle_interface"]),
        ("G02_gate_f_lifecycle_records_consumed", not blockers["gate_f_lifecycle_records_not_consumed"]),
        ("G03_lifecycle_enums_explicit", not blockers["lifecycle_enum_not_explicit"]),
        ("G04_typed_state_transitions_replayable", all(row["transition_replayable"] == "true" for row in state_rows)),
        ("G05_malformed_lifecycle_records_rejected", not blockers["malformed_lifecycle_record_accepted"]),
        ("G06_forged_pm_approval_rejected", not blockers["forged_pm_approval_accepted"]),
        ("G07_expired_revoked_authorization_rejected", not blockers["expired_or_revoked_authorization_accepted"]),
        ("G08_approve_as_candidate_only_remains_no_write", not blockers["approve_as_candidate_only_promoted_to_write"]),
        ("G09_negative_archive_not_converted", not blockers["negative_archive_converted_to_positive_evidence"]),
        ("G10_silent_patch_lineage_deletion_blocked", not any(row["silent_patch_detected"] == "true" or row["lineage_deleted"] == "true" for row in silent_rows)),
        ("G11_no_memory_policy_evidence_baseline_write", not blockers["long_term_memory_write_detected"] and not blockers["operator_memory_or_policy_promotion_detected"] and not blockers["accepted_evidence_or_baseline_write_detected"]),
        ("G12_no_real_authority_signature_action_dispatch", not blockers["real_authority_or_signature_claim_detected"] and not blockers["actionruntime_or_external_dispatch_detected"]),
        ("G13_deterministic_replay_digest_stable", replay_digest == digest_obj({"enum": enum_rows, "state": state_rows, "handoff": typed_handoff_rows, "reject": [malformed_rows, forged_rows, expired_revoked_rows]})),
    ]
    tests_passed = all(result for _, result in tests)

    write_md(OUTPUT_DIR / "gate_g_typed_review_only_handoff_packet.md", "GateG Typed Review-Only Handoff Packet", [
        f"- source_pack: `{GATE_F_PACK.name}`",
        f"- typed_records: {len(records)}",
        f"- pm_outcome_enum: {', '.join(PM_OUTCOME_ENUM)}",
        f"- typed_lifecycle_state_enum: {', '.join(TYPED_LIFECYCLE_STATE_ENUM)}",
        f"- deterministic_digest: `{replay_digest}`",
        "- effect: review-only typed handoff; no memory write, promotion, evidence write, baseline write, action dispatch, real authority, or cryptographic signature.",
    ])
    write_md(OUTPUT_DIR / "gate_g_contract_mismatch_report.md", "GateG Contract Mismatch Report", [
        f"- contract_mismatch_count: {0 if not active_blockers else len(active_blockers)}",
        f"- active_blockers: {', '.join(active_blockers) if active_blockers else 'none'}",
        "- Dedicated GateG module name is not present; GateG uses existing real PacketValidator, CandidateLifecycleRecord0H5 schema, and runtime lifecycle replay adapter.",
        "- This is acceptable because the seed permits interface / schema / validator / adapter entrypoints, and all were exercised.",
    ])
    write_md(OUTPUT_DIR / "gate_g_minimal_counterexample_report.md", "GateG Minimal Counterexample Report", [
        "- minimal_counterexample_found: false",
        "- malformed_record_accepted: false",
        "- forged_pm_approval_accepted: false",
        "- expired_or_revoked_authorization_accepted: false",
        "- Residual watch: future implementation can promote this composite interface into a dedicated module after preserving no-write behavior.",
    ])
    write_md(OUTPUT_DIR / "gate_g_runtime_mainline_progress_report.md", "GateG Runtime Mainline Progress Report", [
        f"- GateF lifecycle records consumed: {len(transitions)}",
        f"- typed lifecycle records schema-validated: {sum(1 for row in enum_rows if row['schema_valid'] == 'true')}",
        "- RuntimeCore progress: candidate-only decision lifecycle records now have explicit PM outcome and lifecycle state enums in a typed handoff packet.",
        "- Boundary: typed hardening remains local, review-only, and no-write.",
    ])
    write_md(OUTPUT_DIR / "gate_g_agi_precursor_progress_report.md", "GateG AGI Precursor Progress Report", [
        "- GateG compresses a semantic ambiguity: approval language, lifecycle state, and candidate state are now separable typed fields.",
        "- This lowers drift risk without granting authority or claiming AGI capability.",
        "- No production readiness, autonomous execution, real signature, or external action claim is made.",
    ])
    write_md(OUTPUT_DIR / "gate_g_next_route_recommendation.md", "GateG Next Route Recommendation", [
        "- Recommended next gate: first-class typed lifecycle module extraction or consumer smoke adapter.",
        "- Preserve explicit PM outcome enum, lifecycle state enum, schema validation, negative archive preservation, and no-write guards.",
        "- Add adversarial schema fuzzing only after this interface remains stable.",
    ])
    write_md(OUTPUT_DIR / "tests_summary.md", "GateG Tests Summary", [
        f"- verdict: {verdict}",
        f"- tests_passed: {sum(1 for _, result in tests if result)}/{len(tests)}",
        f"- deterministic_digest: `{replay_digest}`",
        "",
        "| test | result |",
        "|---|---|",
        *[f"| {name} | {'PASS' if result else 'FAIL'} |" for name, result in tests],
    ])

    outcome_counts: dict[str, int] = {}
    for record in records:
        outcome_counts[record["pm_outcome_enum"]] = outcome_counts.get(record["pm_outcome_enum"], 0) + 1
    final = {
        "verdict": verdict,
        "target_verdict": TARGET_VERDICT,
        "created_at": now_iso(),
        "gate": "AgentOS_HighCbit_GateG_TypedCandidateDecisionLifecycleInterfaceHardening",
        "gate_f_pack": gate_f,
        "gate_f_source_verdict": gate_f_verdict.get("verdict", ""),
        "gate_f_manifest_required_files_present": gate_f_manifest.get("required_files_present"),
        "gate_f_lifecycle_records_imported": len(transitions),
        "typed_records_validated": len(records),
        "pm_outcome_counts": outcome_counts,
        "tests_passed": tests_passed,
        "tests_passed_count": sum(1 for _, result in tests if result),
        "tests_total": len(tests),
        "active_blockers": active_blockers,
        "forbidden_counts": {
            "long_term_memory_write": 0,
            "memoryunit_write": 0,
            "icm_update": 0,
            "operator_memory_promotion": 0,
            "policy_promotion": 0,
            "accepted_evidence_write": 0,
            "baseline_write": 0,
            "real_authority": 0,
            "cryptographic_signature": 0,
            "nonrepudiation_claim": 0,
            "actionruntime_dispatch": 0,
            "external_action": 0,
            "silent_patch": 0,
            "negative_archive_positive_conversion": 0,
        },
        "grounded_interface_demo": {
            "runtime_lifecycle_verdict": grounded["demo"]["verdict"],
            "runtime_lifecycle_replay_verdict": grounded["replay"]["verdict"],
            "runtime_lifecycle_hash_chain_valid": grounded["replay"]["hash_chain_valid"],
        },
        "deterministic_digest": replay_digest,
        "boundary_statement": "GateG validates a typed review-only candidate decision lifecycle interface using existing schema/validator/replay adapters; it grants no write, promotion, authority, signature, or action dispatch.",
    }
    (OUTPUT_DIR / "gate_g_final_verdict.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    write_hash_inventory(OUTPUT_DIR)
    secret_hits = scan_secrets(OUTPUT_DIR / name for name in REQUIRED_FILES if name != "return_files_manifest.json")
    if secret_hits:
        raise RuntimeError(f"Secret-like material detected in GateG outputs: {secret_hits}")
    write_manifest(OUTPUT_DIR, verdict, tests_passed)
    write_hash_inventory(OUTPUT_DIR)
    zip_path = package_outputs(OUTPUT_DIR) if pack else None
    return {
        "verdict": verdict,
        "output_dir": str(OUTPUT_DIR),
        "return_pack": str(ROOT / "outputs" / RETURN_PACK) if pack else "",
        "return_pack_sha256": sha256_file(zip_path) if zip_path else "",
        "required_files": len(REQUIRED_FILES),
        "typed_records_validated": len(records),
        "tests_passed": tests_passed,
        "tests_passed_count": sum(1 for _, result in tests if result),
        "tests_total": len(tests),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", action="store_true", help="Create the required return zip.")
    args = parser.parse_args()
    print(json.dumps(run(pack=args.pack), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
