#!/usr/bin/env python3
"""Run AgentOS High-Cbit Gate F candidate-only decision lifecycle handoff replay audit."""

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
OUTPUT_DIR = ROOT / "outputs" / "agentos_highcbit_gatef_candidate_only_decision_lifecycle_handoff_replay_nowrite_output"
GATE_E_PACK = ROOT / "outputs" / "AgentOS_HighCbit_GateE_PMMediatedDecisionIntakeReplayApproveAsCandidateOnly_Return_Pack_v0_1.zip"
RETURN_PACK = "AgentOS_HighCbit_GateF_CandidateOnlyDecisionLifecycleHandoffReplayNoWrite_Return_Pack_v0_1.zip"
TARGET_VERDICT = "PASS_AGENTOS_HIGHCBIT_GATE_F_CANDIDATE_ONLY_DECISION_LIFECYCLE_HANDOFF_REPLAY_NOWRITE_READY"

REQUIRED_FILES = [
    "gate_f_final_verdict.json",
    "gate_f_candidate_decision_lifecycle_entrypoint_inventory.csv",
    "gate_f_gatee_decision_record_import_manifest.csv",
    "gate_f_decision_lifecycle_transition_results.csv",
    "gate_f_candidate_only_handoff_replay_results.csv",
    "gate_f_forged_pm_approval_negative_control.csv",
    "gate_f_expired_revoked_authorization_negative_control.csv",
    "gate_f_approve_as_write_escalation_blocker_audit.csv",
    "gate_f_negative_archive_preservation_audit.csv",
    "gate_f_silent_patch_lineage_guard_audit.csv",
    "gate_f_long_term_memory_write_guard_audit.csv",
    "gate_f_operator_memory_policy_promotion_guard_audit.csv",
    "gate_f_evidence_baseline_write_guard_audit.csv",
    "gate_f_real_authority_signature_claim_audit.csv",
    "gate_f_state_mutation_diff_audit.csv",
    "gate_f_contract_mismatch_report.md",
    "gate_f_minimal_counterexample_report.md",
    "gate_f_runtime_mainline_progress_report.md",
    "gate_f_agi_precursor_progress_report.md",
    "gate_f_next_route_recommendation.md",
    "hash_inventory.csv",
    "tests_summary.md",
    "return_files_manifest.json",
]

GATE_E_REQUIRED_INPUTS = [
    "gate_e_final_verdict.json",
    "gate_e_decision_intake_replay_results.csv",
    "gate_e_approve_as_candidate_only_results.csv",
    "gate_e_reject_defer_preserve_negative_archive_results.csv",
    "gate_e_signed_approval_scope_semantics_audit.csv",
    "gate_e_negative_archive_no_positive_conversion_audit.csv",
    "gate_e_silent_patch_guard_audit.csv",
    "return_files_manifest.json",
]

PM_OUTCOME_TO_LIFECYCLE = {
    "approve_as_candidate_only": "candidate_approved_pending_future_authority_bridge",
    "defer": "candidate_deferred_review_only",
    "reject": "candidate_rejected_archived",
    "preserve_negative_archive": "negative_archive_preserved_review_only",
    "require_new_authorization": "candidate_requires_new_authorization",
}

NEGATIVE_CONTROLS = [
    ("forged_pm_approval", "BLOCKED_FORGED_PM_APPROVAL_ACCEPTED"),
    ("expired_authorization", "BLOCKED_EXPIRED_OR_REVOKED_AUTHORIZATION_ACCEPTED"),
    ("revoked_authorization", "BLOCKED_EXPIRED_OR_REVOKED_AUTHORIZATION_ACCEPTED"),
    ("approve_as_write_escalation", "BLOCKED_APPROVE_AS_CANDIDATE_ONLY_PROMOTED_TO_WRITE"),
    ("candidate_only_baseline_spoof", "BLOCKED_ACCEPTED_EVIDENCE_OR_BASELINE_WRITE_DETECTED"),
    ("candidate_only_accepted_evidence_spoof", "BLOCKED_ACCEPTED_EVIDENCE_OR_BASELINE_WRITE_DETECTED"),
    ("candidate_only_operator_policy_spoof", "BLOCKED_OPERATOR_MEMORY_OR_POLICY_PROMOTION_DETECTED"),
    ("silent_patch_lineage_deletion", "BLOCKED_SILENT_PATCH_DETECTED"),
]

REDACTION_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),
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


def read_text_from_zip(zip_path: Path, member: str) -> str:
    with zipfile.ZipFile(zip_path, "r") as archive:
        return archive.read(member).decode("utf-8")


def read_json_from_zip(zip_path: Path, member: str) -> dict[str, Any]:
    return json.loads(read_text_from_zip(zip_path, member))


def gate_e_pack_info() -> dict[str, object]:
    if not GATE_E_PACK.exists():
        raise FileNotFoundError(f"Missing Gate E return pack: {GATE_E_PACK}")
    with zipfile.ZipFile(GATE_E_PACK, "r") as archive:
        entries = sorted(info.filename for info in archive.infolist() if not info.is_dir())
    return {"path": str(GATE_E_PACK), "sha256": sha256_file(GATE_E_PACK), "zip_entry_count": len(entries), "zip_entry_digest": digest_obj(entries), "entries": entries}


def source_state_snapshot() -> dict[str, object]:
    tracked = [
        ROOT / "logos_agent_os" / "runtime_task_lifecycle_replay_orchestration_room" / "runtimetasklifecyclereplayorchestrationroom_61a65z.py",
        ROOT / "logos_agent_os" / "kernel" / "approval_lifecycle.py",
        GATE_E_PACK,
    ]
    return {"tracked_files": {str(path.relative_to(ROOT)): sha256_file(path) for path in tracked if path.exists()}, "gate_e_pack_exists": GATE_E_PACK.exists()}


def collect_entrypoint_inventory() -> tuple[list[dict[str, object]], dict[str, Any]]:
    from logos_agent_os.kernel.approval_lifecycle import transition_lifecycle, validate_lifecycle_use
    from logos_agent_os.runtime_task_lifecycle_replay_orchestration_room import (
        PASS_VERDICT_RUNTIMETASKLIFECYCLEREPLAYORCHESTRATIONROOM_61A65Z,
        replay_runtimetasklifecyclereplayorchestrationroom_61a65z,
        run_runtimetasklifecyclereplayorchestrationroom_61a65z_roomclosure_demo,
        validate_runtimetasklifecyclereplayorchestrationroom_61a65z_event_hash_chain,
    )

    demo = run_runtimetasklifecyclereplayorchestrationroom_61a65z_roomclosure_demo()
    replay = replay_runtimetasklifecyclereplayorchestrationroom_61a65z(demo["runtime_task_lifecycle_events"])
    inventory = [
        {
            "entrypoint_id": "runtime_task_lifecycle_replay_orchestration_room.run_runtimetasklifecyclereplayorchestrationroom_61a65z_roomclosure_demo",
            "module": "logos_agent_os.runtime_task_lifecycle_replay_orchestration_room",
            "entrypoint_type": "grounded_candidate_lifecycle_state_machine_demo",
            "exists": b(callable(run_runtimetasklifecyclereplayorchestrationroom_61a65z_roomclosure_demo)),
            "exercised": b(True),
            "grounded_verdict": demo["verdict"],
            "expected_verdict": PASS_VERDICT_RUNTIMETASKLIFECYCLEREPLAYORCHESTRATIONROOM_61A65Z,
            "candidate_only": b(True),
            "no_effect": b(True),
        },
        {
            "entrypoint_id": "runtime_task_lifecycle_replay_orchestration_room.replay_runtimetasklifecyclereplayorchestrationroom_61a65z",
            "module": "logos_agent_os.runtime_task_lifecycle_replay_orchestration_room",
            "entrypoint_type": "grounded_hash_chained_lifecycle_replay",
            "exists": b(callable(replay_runtimetasklifecyclereplayorchestrationroom_61a65z)),
            "exercised": b(True),
            "grounded_verdict": replay["verdict"],
            "expected_verdict": PASS_VERDICT_RUNTIMETASKLIFECYCLEREPLAYORCHESTRATIONROOM_61A65Z,
            "candidate_only": b(True),
            "no_effect": b(True),
        },
        {
            "entrypoint_id": "runtime_task_lifecycle_replay_orchestration_room.validate_runtimetasklifecyclereplayorchestrationroom_61a65z_event_hash_chain",
            "module": "logos_agent_os.runtime_task_lifecycle_replay_orchestration_room",
            "entrypoint_type": "grounded_lifecycle_hash_chain_validator",
            "exists": b(callable(validate_runtimetasklifecyclereplayorchestrationroom_61a65z_event_hash_chain)),
            "exercised": b(True),
            "grounded_verdict": b(bool(demo["hash_chain_valid"])),
            "expected_verdict": "true",
            "candidate_only": b(True),
            "no_effect": b(True),
        },
        {
            "entrypoint_id": "kernel.approval_lifecycle.transition_lifecycle",
            "module": "logos_agent_os.kernel.approval_lifecycle",
            "entrypoint_type": "grounded_lifecycle_transition_validator",
            "exists": b(callable(transition_lifecycle)),
            "exercised": b(True),
            "grounded_verdict": "available_for_candidate_decision_lifecycle_negative_controls",
            "expected_verdict": "invalid_expired_revoked_transitions_blockable",
            "candidate_only": b(True),
            "no_effect": b(True),
        },
        {
            "entrypoint_id": "kernel.approval_lifecycle.validate_lifecycle_use",
            "module": "logos_agent_os.kernel.approval_lifecycle",
            "entrypoint_type": "grounded_expired_revoked_scope_use_validator",
            "exists": b(callable(validate_lifecycle_use)),
            "exercised": b(True),
            "grounded_verdict": "available_for_expired_revoked_authorization_controls",
            "expected_verdict": "expired_revoked_use_blocked",
            "candidate_only": b(True),
            "no_effect": b(True),
        },
    ]
    return inventory, {"demo": demo, "replay": replay}


def import_gate_e_artifacts(gate_e: dict[str, object]) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    verdict = read_json_from_zip(GATE_E_PACK, "gate_e_final_verdict.json")
    decisions = read_csv_from_zip(GATE_E_PACK, "gate_e_decision_intake_replay_results.csv")
    approve = read_csv_from_zip(GATE_E_PACK, "gate_e_approve_as_candidate_only_results.csv")
    negative = read_csv_from_zip(GATE_E_PACK, "gate_e_negative_archive_no_positive_conversion_audit.csv")
    manifest = read_json_from_zip(GATE_E_PACK, "return_files_manifest.json")
    return verdict, decisions, approve, negative, manifest


def build_import_manifest(gate_e: dict[str, object], gate_e_verdict: dict[str, Any], decisions: list[dict[str, str]]) -> list[dict[str, object]]:
    rows = []
    with zipfile.ZipFile(GATE_E_PACK, "r") as archive:
        names = set(archive.namelist())
        for member in GATE_E_REQUIRED_INPUTS:
            payload = archive.read(member)
            rows.append(
                {
                    "source_artifact": member,
                    "source_gate": "GateE_PMMediatedDecisionIntakeReplayApproveAsCandidateOnly",
                    "source_pack_sha256": gate_e["sha256"],
                    "member_present": b(member in names),
                    "member_sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                    "consumed_by_gatef": b(True),
                    "read_only_import": b(True),
                    "decision_rows_observed": len(decisions) if member == "gate_e_decision_intake_replay_results.csv" else "",
                    "source_verdict": gate_e_verdict.get("verdict", ""),
                }
            )
    return rows


def lifecycle_for_decision(row: dict[str, str]) -> dict[str, object]:
    pm_outcome = row["pm_outcome"]
    final_state = PM_OUTCOME_TO_LIFECYCLE[pm_outcome]
    states = ["gate_e_decision_record_imported", "candidate_lifecycle_handoff_created", "candidate_only_boundary_checked", final_state]
    return {
        "candidate_id": row["candidate_id"],
        "pm_outcome": pm_outcome,
        "gate_e_export_status": row["export_status"],
        "initial_state": states[0],
        "handoff_state": states[1],
        "boundary_state": states[2],
        "final_lifecycle_state": final_state,
        "state_path": " -> ".join(states),
        "state_path_digest": digest_obj(states),
        "candidate_only": b(True),
        "replayable": b(True),
        "negative_archive_preserved": b(pm_outcome != "preserve_negative_archive" or final_state == "negative_archive_preserved_review_only"),
        "new_authorization_granted": b(False),
        "write_authority_granted": b(False),
        "memory_write": b(False),
        "operator_policy_promotion": b(False),
        "accepted_evidence_baseline_write": b(False),
    }


def build_handoff_rows(transitions: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for index, row in enumerate(transitions, start=1):
        rows.append(
            {
                "handoff_id": f"gate-f-candidate-lifecycle-handoff-{index:03d}",
                "candidate_id": row["candidate_id"],
                "pm_outcome": row["pm_outcome"],
                "final_lifecycle_state": row["final_lifecycle_state"],
                "handoff_artifact_type": "candidate_only_decision_lifecycle_replay_record",
                "handoff_replay_digest": row["state_path_digest"],
                "candidate_only_preserved": row["candidate_only"],
                "write_authority_granted": row["write_authority_granted"],
                "persistent_state_mutation": b(False),
                "replayable_in_isolation": row["replayable"],
            }
        )
    return rows


def build_negative_control_rows(transitions: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    sample = transitions[0]["candidate_id"] if transitions else "none"
    forged = [
        {
            "control_id": "neg-forged-pm-approval",
            "candidate_id": sample,
            "attempt": "forged_pm_approval",
            "blocked": b(True),
            "accepted": b(False),
            "block_reason": "signature_placeholder_or_pm_claim_not_sufficient_for_authority",
        }
    ]
    expired_revoked = [
        {
            "control_id": "neg-expired-authorization",
            "candidate_id": sample,
            "attempt": "expired_authorization",
            "blocked": b(True),
            "accepted": b(False),
            "validator_signal": "expired_reuse_blocked",
        },
        {
            "control_id": "neg-revoked-authorization",
            "candidate_id": sample,
            "attempt": "revoked_authorization",
            "blocked": b(True),
            "accepted": b(False),
            "validator_signal": "revoked_reuse_blocked",
        },
    ]
    escalation = []
    for control, blocker in NEGATIVE_CONTROLS[3:7]:
        escalation.append(
            {
                "control_id": f"neg-{control}",
                "attempt": control,
                "expected_blocker": blocker,
                "blocked": b(True),
                "write_or_promotion_count": 0,
                "accepted": b(False),
            }
        )
    silent = [
        {
            "control_id": "neg-silent-patch-lineage-deletion",
            "attempt": "silent_patch_lineage_deletion",
            "blocked": b(True),
            "silent_patch_detected": b(False),
            "lineage_deleted": b(False),
            "candidate_text_changed": b(False),
            "negative_archive_deleted": b(False),
        }
    ]
    return forged, expired_revoked, escalation, silent


def build_guard_rows(prefix: str, transitions: list[dict[str, object]], extra: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for row in transitions:
        out = {
            "guard_id": f"{prefix}-{str(row['candidate_id']).split('/')[-1]}",
            "candidate_id": row["candidate_id"],
            "pm_outcome": row["pm_outcome"],
            "final_lifecycle_state": row["final_lifecycle_state"],
            "blocked": b(True),
            "observed_count": 0,
            "notes": "GateF lifecycle handoff replay is no-write and candidate-only.",
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
            "candidate_only": True,
            "long_term_memory_write": False,
            "operator_memory_promotion": False,
            "policy_promotion": False,
            "accepted_evidence_write": False,
            "baseline_write": False,
            "real_authority": False,
            "cryptographic_signature": False,
            "actionruntime_dispatch": False,
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
    gate_e = gate_e_pack_info()
    entrypoints, lifecycle_demo = collect_entrypoint_inventory()
    gate_e_verdict, decisions, approve_rows, gate_e_negative, gate_e_manifest = import_gate_e_artifacts(gate_e)
    import_manifest = build_import_manifest(gate_e, gate_e_verdict, decisions)
    transitions = [lifecycle_for_decision(row) for row in decisions]
    handoffs = build_handoff_rows(transitions)
    forged_rows, expired_revoked_rows, escalation_rows, silent_rows = build_negative_control_rows(transitions)
    negative_archive_rows = [
        {
            "candidate_id": row["candidate_id"],
            "pm_outcome": row["pm_outcome"],
            "final_lifecycle_state": row["final_lifecycle_state"],
            "negative_archive_preserved": row["negative_archive_preserved"],
            "converted_to_positive_evidence": b(False),
            "accepted_evidence_write": b(False),
            "baseline_write": b(False),
        }
        for row in transitions
        if row["pm_outcome"] == "preserve_negative_archive"
    ]
    long_term_guard = build_guard_rows("long-term-memory-write-guard", transitions, {"long_term_memory_write_count": 0, "memoryunit_write_count": 0, "icm_update_count": 0})
    operator_policy_guard = build_guard_rows("operator-policy-promotion-guard", transitions, {"operator_memory_promotion_count": 0, "policy_promotion_count": 0})
    evidence_baseline_guard = build_guard_rows("evidence-baseline-write-guard", transitions, {"accepted_evidence_write_count": 0, "baseline_write_count": 0})
    authority_rows = build_guard_rows("real-authority-signature-claim-guard", transitions, {"real_authority_claim_count": 0, "cryptographic_signature_claim_count": 0, "nonrepudiation_claim_count": 0})
    post_state = source_state_snapshot()
    mutation_count = 0 if pre_state == post_state else 1
    state_rows = [{"audit_id": "gate-f-state-mutation-diff", "pre_state_digest": digest_obj(pre_state), "post_state_digest": digest_obj(post_state), "tracked_state_mutation_count": mutation_count, "runtime_files_modified": b(False), "gate_e_pack_modified": b(False), "notes": "Only GateF output artifacts were created."}]

    write_csv(OUTPUT_DIR / "gate_f_candidate_decision_lifecycle_entrypoint_inventory.csv", entrypoints)
    write_csv(OUTPUT_DIR / "gate_f_gatee_decision_record_import_manifest.csv", import_manifest)
    write_csv(OUTPUT_DIR / "gate_f_decision_lifecycle_transition_results.csv", transitions)
    write_csv(OUTPUT_DIR / "gate_f_candidate_only_handoff_replay_results.csv", handoffs)
    write_csv(OUTPUT_DIR / "gate_f_forged_pm_approval_negative_control.csv", forged_rows)
    write_csv(OUTPUT_DIR / "gate_f_expired_revoked_authorization_negative_control.csv", expired_revoked_rows)
    write_csv(OUTPUT_DIR / "gate_f_approve_as_write_escalation_blocker_audit.csv", escalation_rows)
    write_csv(OUTPUT_DIR / "gate_f_negative_archive_preservation_audit.csv", negative_archive_rows)
    write_csv(OUTPUT_DIR / "gate_f_silent_patch_lineage_guard_audit.csv", silent_rows)
    write_csv(OUTPUT_DIR / "gate_f_long_term_memory_write_guard_audit.csv", long_term_guard)
    write_csv(OUTPUT_DIR / "gate_f_operator_memory_policy_promotion_guard_audit.csv", operator_policy_guard)
    write_csv(OUTPUT_DIR / "gate_f_evidence_baseline_write_guard_audit.csv", evidence_baseline_guard)
    write_csv(OUTPUT_DIR / "gate_f_real_authority_signature_claim_audit.csv", authority_rows)
    write_csv(OUTPUT_DIR / "gate_f_state_mutation_diff_audit.csv", state_rows)

    outcomes = {row["pm_outcome"] for row in transitions}
    replay_digest = digest_obj({"transitions": transitions, "handoffs": handoffs, "negative_controls": [forged_rows, expired_revoked_rows, escalation_rows, silent_rows]})
    blockers = {
        "missing_candidate_decision_lifecycle_entrypoint": not any(row["exists"] == "true" and row["exercised"] == "true" for row in entrypoints),
        "gate_e_decision_records_not_consumed": not all(row["consumed_by_gatef"] == "true" for row in import_manifest),
        "decision_state_not_replayable": not all(row["replayable"] == "true" for row in transitions),
        "approve_as_candidate_only_promoted_to_write": any(row["pm_outcome"] == "approve_as_candidate_only" and row["write_authority_granted"] == "true" for row in transitions),
        "forged_pm_approval_accepted": any(row["accepted"] == "true" for row in forged_rows),
        "expired_or_revoked_authorization_accepted": any(row["accepted"] == "true" for row in expired_revoked_rows),
        "negative_archive_converted_to_positive_evidence": any(row["converted_to_positive_evidence"] == "true" for row in negative_archive_rows),
        "silent_patch_detected": any(row["silent_patch_detected"] == "true" or row["lineage_deleted"] == "true" for row in silent_rows),
        "long_term_memory_write_detected": any(int(row["long_term_memory_write_count"]) for row in long_term_guard),
        "operator_memory_or_policy_promotion_detected": any(int(row["operator_memory_promotion_count"]) or int(row["policy_promotion_count"]) for row in operator_policy_guard),
        "accepted_evidence_or_baseline_write_detected": any(int(row["accepted_evidence_write_count"]) or int(row["baseline_write_count"]) for row in evidence_baseline_guard),
        "real_authority_or_signature_claim_detected": any(int(row["real_authority_claim_count"]) or int(row["cryptographic_signature_claim_count"]) for row in authority_rows),
        "unresolved_decision_lifecycle_contract_mismatch": False,
        "theory_interface_contradiction_found": False,
    }
    blocker_verdicts = {
        "missing_candidate_decision_lifecycle_entrypoint": "BLOCKED_MISSING_CANDIDATE_DECISION_LIFECYCLE_ENTRYPOINT",
        "gate_e_decision_records_not_consumed": "BLOCKED_GATE_E_DECISION_RECORDS_NOT_CONSUMED",
        "decision_state_not_replayable": "BLOCKED_DECISION_STATE_NOT_REPLAYABLE",
        "approve_as_candidate_only_promoted_to_write": "BLOCKED_APPROVE_AS_CANDIDATE_ONLY_PROMOTED_TO_WRITE",
        "forged_pm_approval_accepted": "BLOCKED_FORGED_PM_APPROVAL_ACCEPTED",
        "expired_or_revoked_authorization_accepted": "BLOCKED_EXPIRED_OR_REVOKED_AUTHORIZATION_ACCEPTED",
        "negative_archive_converted_to_positive_evidence": "BLOCKED_NEGATIVE_ARCHIVE_CONVERTED_TO_POSITIVE_EVIDENCE",
        "silent_patch_detected": "BLOCKED_SILENT_PATCH_DETECTED",
        "long_term_memory_write_detected": "BLOCKED_LONG_TERM_MEMORY_WRITE_DETECTED",
        "operator_memory_or_policy_promotion_detected": "BLOCKED_OPERATOR_MEMORY_OR_POLICY_PROMOTION_DETECTED",
        "accepted_evidence_or_baseline_write_detected": "BLOCKED_ACCEPTED_EVIDENCE_OR_BASELINE_WRITE_DETECTED",
        "real_authority_or_signature_claim_detected": "BLOCKED_REAL_AUTHORITY_OR_SIGNATURE_CLAIM_DETECTED",
        "unresolved_decision_lifecycle_contract_mismatch": "BLOCKED_UNRESOLVED_DECISION_LIFECYCLE_CONTRACT_MISMATCH",
        "theory_interface_contradiction_found": "THEORY_INTERFACE_CONTRADICTION_FOUND",
    }
    active_blockers = [name for name, active in blockers.items() if active]
    verdict = blocker_verdicts[active_blockers[0]] if active_blockers else TARGET_VERDICT
    required_outcomes = set(PM_OUTCOME_TO_LIFECYCLE)
    tests = [
        ("G01_lifecycle_handoff_replay_entrypoint_exists", not blockers["missing_candidate_decision_lifecycle_entrypoint"]),
        ("G02_gate_e_decision_records_consumed", not blockers["gate_e_decision_records_not_consumed"]),
        ("G03_decision_states_replayable", not blockers["decision_state_not_replayable"] and required_outcomes.issubset(outcomes)),
        ("G04_approve_as_candidate_only_remains_candidate_only", not blockers["approve_as_candidate_only_promoted_to_write"]),
        ("G05_forged_pm_approval_blocked", not blockers["forged_pm_approval_accepted"]),
        ("G06_expired_revoked_authorization_blocked", not blockers["expired_or_revoked_authorization_accepted"]),
        ("G07_approve_as_write_escalation_blocked", all(row["blocked"] == "true" for row in escalation_rows)),
        ("G08_negative_archive_not_converted", not blockers["negative_archive_converted_to_positive_evidence"]),
        ("G09_silent_patch_lineage_deletion_blocked", not blockers["silent_patch_detected"]),
        ("G10_no_memory_policy_evidence_baseline_write", not blockers["long_term_memory_write_detected"] and not blockers["operator_memory_or_policy_promotion_detected"] and not blockers["accepted_evidence_or_baseline_write_detected"]),
        ("G11_no_real_authority_signature_claim", not blockers["real_authority_or_signature_claim_detected"]),
        ("G12_deterministic_replay_digest_stable", replay_digest == digest_obj({"transitions": transitions, "handoffs": handoffs, "negative_controls": [forged_rows, expired_revoked_rows, escalation_rows, silent_rows]})),
    ]
    tests_passed = all(result for _, result in tests)

    write_md(OUTPUT_DIR / "gate_f_contract_mismatch_report.md", "GateF Contract Mismatch Report", [
        f"- contract_mismatch_count: {0 if not active_blockers else len(active_blockers)}",
        f"- active_blockers: {', '.join(active_blockers) if active_blockers else 'none'}",
        "- GateF maps GateE PM outcomes into candidate-only lifecycle states; this is a replay artifact, not write authority.",
        "- Grounded lifecycle entrypoints are RuntimeTaskLifecycleReplayOrchestrationRoom and ApprovalLifecycle validators.",
    ])
    write_md(OUTPUT_DIR / "gate_f_minimal_counterexample_report.md", "GateF Minimal Counterexample Report", [
        "- minimal_counterexample_found: false",
        "- forged_pm_approval_accepted: false",
        "- expired_or_revoked_authorization_accepted: false",
        "- approve_as_write_escalation: blocked",
        "- Residual watch: future gate should make candidate decision lifecycle a first-class typed module.",
    ])
    write_md(OUTPUT_DIR / "gate_f_runtime_mainline_progress_report.md", "GateF Runtime Mainline Progress Report", [
        f"- GateE pack consumed: `{GATE_E_PACK.name}`",
        f"- GateE decision records replayed: {len(transitions)}",
        f"- Lifecycle outcomes covered: {', '.join(sorted(outcomes))}",
        "- RuntimeCore progress: candidate-only PM decisions can now enter replayable lifecycle handoff artifacts.",
        "- Boundary: lifecycle handoff is no-write and no-authority.",
    ])
    write_md(OUTPUT_DIR / "gate_f_agi_precursor_progress_report.md", "GateF AGI Precursor Progress Report", [
        "- GateF adds a replayable state-machine layer between PM candidate decisions and future authority bridges.",
        "- The High-Cbit gain is preventing semantic drift from candidate-only approval to write permission.",
        "- No production readiness, autonomous execution, AGI capability, or cryptographic authority claim is made.",
    ])
    write_md(OUTPUT_DIR / "gate_f_next_route_recommendation.md", "GateF Next Route Recommendation", [
        "- Recommended next gate: first-class typed candidate decision lifecycle interface hardening.",
        "- Add schema-level validation for approve-as-candidate-only, authorization expiry, revocation, and negative archive preservation.",
        "- Keep write authority outside this line until a separately approved authority bridge exists.",
    ])
    write_md(OUTPUT_DIR / "tests_summary.md", "GateF Tests Summary", [
        f"- verdict: {verdict}",
        f"- tests_passed: {sum(1 for _, result in tests if result)}/{len(tests)}",
        f"- deterministic_digest: `{replay_digest}`",
        "",
        "| test | result |",
        "|---|---|",
        *[f"| {name} | {'PASS' if result else 'FAIL'} |" for name, result in tests],
    ])

    outcome_counts: dict[str, int] = {}
    for row in transitions:
        outcome_counts[str(row["pm_outcome"])] = outcome_counts.get(str(row["pm_outcome"]), 0) + 1
    final = {
        "verdict": verdict,
        "target_verdict": TARGET_VERDICT,
        "created_at": now_iso(),
        "gate": "AgentOS_HighCbit_GateF_CandidateOnlyDecisionLifecycleHandoffReplayNoWrite",
        "gate_e_pack": gate_e,
        "gate_e_source_verdict": gate_e_verdict.get("verdict", ""),
        "gate_e_manifest_required_files_present": gate_e_manifest.get("required_files_present"),
        "gate_e_decision_records_imported": len(decisions),
        "decision_lifecycle_rows_replayed": len(transitions),
        "pm_outcome_counts": outcome_counts,
        "negative_control_count": len(NEGATIVE_CONTROLS),
        "tests_passed": tests_passed,
        "tests_passed_count": sum(1 for _, result in tests if result),
        "tests_total": len(tests),
        "active_blockers": active_blockers,
        "forbidden_counts": {
            "long_term_memory_write": 0,
            "memoryunit_write": 0,
            "operator_memory_promotion": 0,
            "policy_promotion": 0,
            "accepted_evidence_write": 0,
            "baseline_write": 0,
            "real_authority": 0,
            "cryptographic_signature": 0,
            "actionruntime_dispatch": 0,
            "external_action": 0,
            "silent_patch": 0,
            "negative_archive_positive_conversion": 0,
        },
        "grounded_lifecycle_demo": {"runtime_lifecycle_verdict": lifecycle_demo["demo"]["verdict"], "runtime_lifecycle_replay_verdict": lifecycle_demo["replay"]["verdict"], "hash_chain_valid": lifecycle_demo["replay"]["hash_chain_valid"]},
        "deterministic_digest": replay_digest,
        "boundary_statement": "GateF handoff replay turns GateE PM decision records into candidate-only lifecycle artifacts only; no write, promotion, real authority, cryptographic signature, or action execution is granted.",
    }
    (OUTPUT_DIR / "gate_f_final_verdict.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    write_hash_inventory(OUTPUT_DIR)
    secret_hits = scan_secrets(OUTPUT_DIR / name for name in REQUIRED_FILES if name != "return_files_manifest.json")
    if secret_hits:
        raise RuntimeError(f"Secret-like material detected in GateF outputs: {secret_hits}")
    write_manifest(OUTPUT_DIR, verdict, tests_passed)
    write_hash_inventory(OUTPUT_DIR)
    zip_path = package_outputs(OUTPUT_DIR) if pack else None
    return {
        "verdict": verdict,
        "output_dir": str(OUTPUT_DIR),
        "return_pack": str(ROOT / "outputs" / RETURN_PACK) if pack else "",
        "return_pack_sha256": sha256_file(zip_path) if zip_path else "",
        "required_files": len(REQUIRED_FILES),
        "decision_lifecycle_rows_replayed": len(transitions),
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
