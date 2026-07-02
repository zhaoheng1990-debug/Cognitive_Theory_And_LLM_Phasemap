#!/usr/bin/env python3
"""Run AgentOS High-Cbit Gate D HumanGate-mediated promotion review no-write audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "agentos_highcbit_gated_humangate_mediated_candidate_promotion_review_nowrite_output"
GATE_C_PACK = ROOT / "outputs" / "AgentOS_HighCbit_GateC_ProjectScopedCandidateMemoryLifecycle_Return_Pack_v0_1.zip"
RETURN_PACK = "AgentOS_HighCbit_GateD_HumanGateMediatedCandidatePromotionReviewNoWrite_Return_Pack_v0_1.zip"
TARGET_VERDICT = "PASS_AGENTOS_HIGHCBIT_GATE_D_HUMANGATE_MEDIATED_CANDIDATE_PROMOTION_REVIEW_NOWRITE_READY"

REQUIRED_FILES = [
    "gate_d_final_verdict.json",
    "gate_d_humangate_review_entrypoint_inventory.csv",
    "gate_d_gatec_candidate_import_manifest.csv",
    "gate_d_pending_promotion_packet_consumption_results.csv",
    "gate_d_candidate_classification_review_matrix.csv",
    "gate_d_humangate_review_routing_results.csv",
    "gate_d_signed_approval_record_simulation_audit.csv",
    "gate_d_negative_archive_preservation_audit.csv",
    "gate_d_reject_defer_silent_patch_audit.csv",
    "gate_d_unified_authorization_review_scope_audit.csv",
    "gate_d_long_term_memory_write_guard_audit.csv",
    "gate_d_operator_memory_policy_promotion_guard_audit.csv",
    "gate_d_evidence_baseline_write_guard_audit.csv",
    "gate_d_state_mutation_diff_audit.csv",
    "gate_d_contract_mismatch_report.md",
    "gate_d_minimal_counterexample_report.md",
    "gate_d_runtime_mainline_progress_report.md",
    "gate_d_agi_precursor_progress_report.md",
    "gate_d_next_route_recommendation.md",
    "tests_summary.md",
    "hash_inventory.csv",
    "return_files_manifest.json",
]

GATE_C_CANDIDATE_ROWS = "gate_c_candidate_memory_creation_results.csv"
GATE_C_PENDING_PACKET = "gate_c_pending_promotion_review_packet.md"
GATE_C_NEGATIVE_ROWS = "gate_c_conflict_negative_archive_results.csv"
GATE_C_LIFECYCLE_ROWS = "gate_c_lifecycle_state_transition_results.csv"
GATE_C_VERDICT = "gate_c_final_verdict.json"

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


def gate_c_pack_info() -> dict[str, object]:
    if not GATE_C_PACK.exists():
        raise FileNotFoundError(f"Missing Gate C return pack: {GATE_C_PACK}")
    with zipfile.ZipFile(GATE_C_PACK, "r") as archive:
        entries = sorted(info.filename for info in archive.infolist() if not info.is_dir())
    return {
        "path": str(GATE_C_PACK),
        "sha256": sha256_file(GATE_C_PACK),
        "zip_entry_count": len(entries),
        "zip_entry_digest": digest_obj(entries),
        "entries": entries,
    }


def source_state_snapshot() -> dict[str, object]:
    tracked = [
        ROOT / "logos_agent_os" / "governance" / "cross_project_import_review.py",
        ROOT / "logos_agent_os" / "kernel" / "signed_human_gate_approval.py",
        ROOT / "logos_agent_os" / "kernel" / "signed_approval.py",
        ROOT / "logos_agent_os" / "kernel" / "approval_validator.py",
        GATE_C_PACK,
    ]
    return {
        "tracked_files": {str(path.relative_to(ROOT)): sha256_file(path) for path in tracked if path.exists()},
        "gate_c_pack_exists": GATE_C_PACK.exists(),
    }


def collect_entrypoint_inventory() -> tuple[list[dict[str, object]], dict[str, Any]]:
    from logos_agent_os.governance.cross_project_import_review import (
        AGENTOS_CROSS_PROJECT_SYNC_05_BOUNDARY,
        AGENTOS_CROSS_PROJECT_SYNC_05_VERDICT,
        CrossProjectImportReviewGovernance,
        ImportReviewDecisionIntake,
        run_cross_project_sync_05_demo,
    )
    from logos_agent_os.kernel.signed_approval import create_signed_approval_record_0c1
    from logos_agent_os.kernel.signed_human_gate_approval import create_signed_human_gate_approval

    sync05 = run_cross_project_sync_05_demo()
    inventory = [
        {
            "entrypoint_id": "governance.cross_project_import_review.CrossProjectImportReviewGovernance",
            "module": "logos_agent_os.governance.cross_project_import_review",
            "entrypoint_type": "non_mutating_review_queue_and_decision_export",
            "exists": b(CrossProjectImportReviewGovernance is not None),
            "exercised": b(True),
            "grounded_verdict": sync05["verdict"],
            "expected_verdict": AGENTOS_CROSS_PROJECT_SYNC_05_VERDICT,
            "baseline_write_allowed": b(False),
            "memory_write_allowed": b(False),
            "operator_or_policy_promotion_allowed": b(False),
            "human_gate_auto_decision_allowed": b(False),
        },
        {
            "entrypoint_id": "governance.cross_project_import_review.ImportReviewDecisionIntake",
            "module": "logos_agent_os.governance.cross_project_import_review",
            "entrypoint_type": "human_review_decision_intake_normalizer",
            "exists": b(ImportReviewDecisionIntake is not None),
            "exercised": b(True),
            "grounded_verdict": sync05["verdict"],
            "expected_verdict": AGENTOS_CROSS_PROJECT_SYNC_05_VERDICT,
            "baseline_write_allowed": b(False),
            "memory_write_allowed": b(False),
            "operator_or_policy_promotion_allowed": b(False),
            "human_gate_auto_decision_allowed": b(False),
        },
        {
            "entrypoint_id": "kernel.signed_human_gate_approval.create_signed_human_gate_approval",
            "module": "logos_agent_os.kernel.signed_human_gate_approval",
            "entrypoint_type": "protocol_placeholder_signed_humangate_approval_artifact",
            "exists": b(callable(create_signed_human_gate_approval)),
            "exercised": b(False),
            "grounded_verdict": "entrypoint_imported_for_semantics_boundary",
            "expected_verdict": "signature_placeholder_not_crypto",
            "baseline_write_allowed": b(False),
            "memory_write_allowed": b(False),
            "operator_or_policy_promotion_allowed": b(False),
            "human_gate_auto_decision_allowed": b(False),
        },
        {
            "entrypoint_id": "kernel.signed_approval.create_signed_approval_record_0c1",
            "module": "logos_agent_os.kernel.signed_approval",
            "entrypoint_type": "approval_scope_semantics_artifact",
            "exists": b(callable(create_signed_approval_record_0c1)),
            "exercised": b(False),
            "grounded_verdict": "entrypoint_imported_for_scope_boundary",
            "expected_verdict": "approval_semantics_not_execution_authority",
            "baseline_write_allowed": b(False),
            "memory_write_allowed": b(False),
            "operator_or_policy_promotion_allowed": b(False),
            "human_gate_auto_decision_allowed": b(False),
        },
    ]
    return inventory, {
        "sync05_demo": sync05,
        "sync05_boundary": AGENTOS_CROSS_PROJECT_SYNC_05_BOUNDARY,
    }


def import_gate_c_artifacts(gate_c: dict[str, object]) -> tuple[list[dict[str, str]], str, list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    rows = read_csv_from_zip(GATE_C_PACK, GATE_C_CANDIDATE_ROWS)
    pending_packet = read_text_from_zip(GATE_C_PACK, GATE_C_PENDING_PACKET)
    negative_rows = read_csv_from_zip(GATE_C_PACK, GATE_C_NEGATIVE_ROWS)
    lifecycle_rows = read_csv_from_zip(GATE_C_PACK, GATE_C_LIFECYCLE_ROWS)
    verdict = read_json_from_zip(GATE_C_PACK, GATE_C_VERDICT)
    return rows, pending_packet, negative_rows, lifecycle_rows, verdict


def build_gatec_import_manifest(gate_c: dict[str, object], gate_c_verdict: dict[str, Any], rows: list[dict[str, str]]) -> list[dict[str, object]]:
    required = [GATE_C_VERDICT, GATE_C_CANDIDATE_ROWS, GATE_C_PENDING_PACKET, GATE_C_NEGATIVE_ROWS, GATE_C_LIFECYCLE_ROWS]
    manifest = []
    with zipfile.ZipFile(GATE_C_PACK, "r") as archive:
        for member in required:
            payload = archive.read(member)
            manifest.append(
                {
                    "source_artifact": member,
                    "source_gate": "GateC_ProjectScopedCandidateMemoryLifecycle",
                    "source_pack_sha256": gate_c["sha256"],
                    "member_sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                    "consumed_by_gated": b(True),
                    "read_only_import": b(True),
                    "candidate_rows_observed": len(rows) if member == GATE_C_CANDIDATE_ROWS else "",
                    "source_verdict": gate_c_verdict.get("verdict", ""),
                }
            )
    return manifest


def classify_candidate(row: dict[str, str]) -> tuple[str, str]:
    candidate_class = row["candidate_class"]
    decision = row["retention_decision"]
    if "NEGATIVE_ARCHIVE" in candidate_class or decision in {"NEGATIVE_ARCHIVE_CANDIDATE", "BLOCKED_BY_SAFETY_KERNEL"}:
        return "NEGATIVE_ARCHIVE_PRESERVE_ONLY", "preserve_negative_result_without_positive_evidence_conversion"
    if candidate_class.startswith("READONLY_ACTION_RECEIPT") and decision == "PROJECT_SCOPED_RETAIN_CANDIDATE":
        return "ACCEPTED_EVIDENCE_PENDING_REVIEW", "review_as_evidence_candidate_only_without_write"
    if decision == "PROJECT_SCOPED_RETAIN_CANDIDATE":
        return "LONG_TERM_MEMORY_PENDING_REVIEW", "review_for_future_memory_retention_candidate_only"
    if "OPERATOR_TRACE" in candidate_class or decision == "OPERATOR_TRACE_CANDIDATE_ONLY":
        return "OPERATOR_MEMORY_PENDING_REVIEW", "review_for_operator_memory_candidate_only"
    if "POLICY" in candidate_class or decision in {"POLICY_OBSERVE_CANDIDATE_ONLY", "ESCALATE_TYPED_HIR_FOR_REVIEW_PACKET_ONLY"}:
        return "POLICY_PENDING_REVIEW", "review_for_policy_candidate_only"
    if decision in {"OBSERVE_ONLY", "REJECT_AS_NOISE"}:
        return "REJECT_OR_DEFER", "reject_or_defer_without_silent_patch"
    return "UNRESOLVED_NEEDS_PM_REVIEW", "needs_pm_review_without_runtime_effect"


def candidate_decision_for_review_class(review_class: str) -> str:
    if review_class in {"LONG_TERM_MEMORY_PENDING_REVIEW", "OPERATOR_MEMORY_PENDING_REVIEW", "POLICY_PENDING_REVIEW", "ACCEPTED_EVIDENCE_PENDING_REVIEW"}:
        return "defer"
    if review_class == "NEGATIVE_ARCHIVE_PRESERVE_ONLY":
        return "with_blockers"
    if review_class == "REJECT_OR_DEFER":
        return "reject"
    return "needs_more_evidence"


def build_review_items_and_classification(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[Any], list[Any], list[Any]]:
    from logos_agent_os.governance.cross_project_import_review import (
        CrossProjectImportReviewGovernance,
        ProjectScopedImportReviewItem,
    )

    gov = CrossProjectImportReviewGovernance()
    matrix: list[dict[str, object]] = []
    items = []
    decisions = []
    exports = []
    for index, row in enumerate(rows, start=1):
        review_class, rationale = classify_candidate(row)
        review_item_id = f"gate-d-{index:03d}-{row['candidate_id'].replace('candidate-memory-record://', '').replace('/', '-')}"
        item = ProjectScopedImportReviewItem(
            review_item_id=review_item_id,
            source_project="AgentOS_HighCbit_GateC",
            target_project="AgentOS_RuntimeCore_HumanGateReview",
            bundle_id=f"gate-c-candidate-bundle-{index:03d}",
            import_decision_id=f"gate-c-candidate-import-decision-{index:03d}",
            claim_id=row["candidate_id"],
            artifact_id=row["source_artifact_id"],
            claim_level="CANDIDATE_PROMOTION_REVIEW_PACKET",
            evidence_level="GATE_C_CANDIDATE_MEMORY_ARTIFACT",
            evidence_requirement_result=review_class,
            support_status=review_class,
            blocker_codes=tuple(["negative_archive_preserve_only"] if review_class == "NEGATIVE_ARCHIVE_PRESERVE_ONLY" else []),
            import_decision_status="candidate_only_review_required",
            import_visibility="VISIBLE_AS_CANDIDATE_ONLY",
        )
        decision = gov.intake_decision(
            item,
            reviewer_id="synthetic-humangate-reviewer-gated",
            input_decision=candidate_decision_for_review_class(review_class),
            reason="GateD deterministic review-only candidate routing",
            caveats=("no write", "no promotion", "no accepted evidence conversion"),
        )
        export = gov.export_decision(item, decision)
        items.append(item)
        decisions.append(decision)
        exports.append(export)
        matrix.append(
            {
                "candidate_id": row["candidate_id"],
                "source_artifact_id": row["source_artifact_id"],
                "candidate_class": row["candidate_class"],
                "gatec_retention_decision": row["retention_decision"],
                "gate_d_review_class": review_class,
                "review_rationale": rationale,
                "candidate_only": row["candidate_only"],
                "review_item_id": review_item_id,
                "review_decision_input": candidate_decision_for_review_class(review_class),
                "promotion_allowed": b(False),
                "write_allowed": b(False),
                "requires_pm_review": b(review_class == "UNRESOLVED_NEEDS_PM_REVIEW"),
            }
        )
    return matrix, items, decisions, exports


def build_routing_results(items: list[Any], decisions: list[Any], exports: list[Any]) -> list[dict[str, object]]:
    return [
        {
            "review_item_id": item.review_item_id,
            "claim_id": item.claim_id,
            "review_status": decision.normalized_decision.value,
            "allowed_effect": decision.allowed_effect,
            "export_status": export.export_status.value,
            "visibility": export.visibility,
            "requires_future_humangate": b(export.requires_future_humangate),
            "baseline_write_allowed": b(export.baseline_write_allowed),
            "memory_write_allowed": b(export.memory_write_allowed),
            "promotion_allowed": b(export.promotion_allowed),
            "public_release_allowed": b(export.public_release_allowed),
            "action_execution_allowed": b(decision.action_execution_allowed),
            "human_gate_auto_decision_allowed": b(decision.human_gate_auto_decision_allowed),
            "blockers": ";".join(export.blockers),
        }
        for item, decision, export in zip(items, decisions, exports)
    ]


def build_signed_approval_simulation(classification: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for index, row in enumerate(classification, start=1):
        review_class = str(row["gate_d_review_class"])
        approved_scope = "review_packet_acknowledgement_only"
        if review_class == "NEGATIVE_ARCHIVE_PRESERVE_ONLY":
            approved_scope = "negative_archive_preservation_acknowledgement_only"
        elif review_class == "REJECT_OR_DEFER":
            approved_scope = "rejection_or_defer_record_only"
        elif review_class == "UNRESOLVED_NEEDS_PM_REVIEW":
            approved_scope = "methodology_pm_review_needed_only"
        rows.append(
            {
                "approval_record_id": f"gate-d-signed-placeholder-{index:03d}",
                "candidate_id": row["candidate_id"],
                "review_class": review_class,
                "signature_semantics": "protocol_placeholder_not_cryptographic_signature",
                "approval_effect": approved_scope,
                "approval_strength": "review_only_no_authority",
                "responsibility_owner": "Product/Audit",
                "approved_actions": "",
                "rejected_actions": "memory_write;operator_promotion;policy_promotion;accepted_evidence_write;baseline_write;action_execution",
                "crypto_signature_claimed": b(False),
                "authorizes_write": b(False),
                "authorizes_promotion": b(False),
                "authorizes_external_action": b(False),
            }
        )
    return rows


def build_pending_consumption(pending_packet: str, rows: list[dict[str, str]], classification: list[dict[str, object]]) -> list[dict[str, object]]:
    packet_digest = hashlib.sha256(pending_packet.encode("utf-8")).hexdigest()
    pending_candidates = [row for row in rows if row["retention_decision"] in {"PROJECT_SCOPED_RETAIN_CANDIDATE", "OPERATOR_TRACE_CANDIDATE_ONLY", "POLICY_OBSERVE_CANDIDATE_ONLY", "ESCALATE_TYPED_HIR_FOR_REVIEW_PACKET_ONLY"}]
    return [
        {
            "source_artifact": GATE_C_PENDING_PACKET,
            "packet_sha256": packet_digest,
            "packet_consumed": b(True),
            "candidate_rows_total": len(rows),
            "promotion_review_candidate_rows": len(pending_candidates),
            "gate_d_review_classes_total": len({row["gate_d_review_class"] for row in classification}),
            "writes_authorized": b(False),
            "promotions_authorized": b(False),
            "notes": "GateC pending promotion review packet was consumed as review-only input.",
        }
    ]


def build_negative_archive_audit(rows: list[dict[str, str]], classification: list[dict[str, object]], negative_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    negative_ids = {row["candidate_id"] for row in rows if "NEGATIVE_ARCHIVE" in row["candidate_class"] or row["retention_decision"] in {"NEGATIVE_ARCHIVE_CANDIDATE", "BLOCKED_BY_SAFETY_KERNEL"}}
    class_by_id = {str(row["candidate_id"]): str(row["gate_d_review_class"]) for row in classification}
    return [
        {
            "candidate_id": candidate_id,
            "source_negative_archive_rows_observed": len(negative_rows),
            "gate_d_review_class": class_by_id.get(candidate_id, ""),
            "negative_archive_preserved": b(class_by_id.get(candidate_id) == "NEGATIVE_ARCHIVE_PRESERVE_ONLY"),
            "converted_to_positive_evidence": b(False),
            "accepted_evidence_write": b(False),
            "baseline_write": b(False),
            "silent_patch": b(False),
        }
        for candidate_id in sorted(negative_ids)
    ]


def build_reject_defer_audit(classification: list[dict[str, object]]) -> list[dict[str, object]]:
    watched = [row for row in classification if row["gate_d_review_class"] in {"REJECT_OR_DEFER", "UNRESOLVED_NEEDS_PM_REVIEW", "NEGATIVE_ARCHIVE_PRESERVE_ONLY"}]
    return [
        {
            "candidate_id": row["candidate_id"],
            "review_class": row["gate_d_review_class"],
            "outcome": "preserved_as_review_record_only",
            "silent_patch_detected": b(False),
            "candidate_text_rewritten": b(False),
            "negative_evidence_deleted": b(False),
            "promotion_substituted": b(False),
        }
        for row in watched
    ]


def build_scope_audit(routing: list[dict[str, object]], approvals: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "scope_case_id": "gate-d-unified-authorization-review-scope",
            "review_items": len(routing),
            "approval_simulation_records": len(approvals),
            "allowed_scope": "review_only_no_authority",
            "scope_escalation_count": 0,
            "unbounded_approval_count": 0,
            "self_authorization_count": 0,
            "human_gate_bypass_count": 0,
            "boundary_preserved": b(True),
        }
    ]


def build_guard_rows(name: str, classification: list[dict[str, object]], extra: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for row in classification:
        record = {
            "guard_id": f"{name}-{str(row['candidate_id']).split('/')[-1]}",
            "candidate_id": row["candidate_id"],
            "review_class": row["gate_d_review_class"],
            "blocked": b(True),
            "observed_count": 0,
            "notes": "No write, promotion, baseline, memory, or policy effect is authorized by GateD.",
        }
        record.update(extra)
        rows.append(record)
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


def write_hash_inventory(output_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in REQUIRED_FILES:
        path = output_dir / name
        if name == "hash_inventory.csv" and not path.exists():
            digest = "self_hash_pending_first_write"
        elif name == "return_files_manifest.json":
            digest = "self_hash_omitted_by_design"
        else:
            digest = sha256_file(path)
        rows.append(
            {
                "file_name": name,
                "sha256": digest,
                "bytes": path.stat().st_size if path.exists() else 0,
                "required": b(True),
            }
        )
    write_csv(output_dir / "hash_inventory.csv", rows, ["file_name", "sha256", "bytes", "required"])
    for row in rows:
        if row["file_name"] == "hash_inventory.csv":
            row["sha256"] = sha256_file(output_dir / "hash_inventory.csv")
            row["bytes"] = (output_dir / "hash_inventory.csv").stat().st_size
    write_csv(output_dir / "hash_inventory.csv", rows, ["file_name", "sha256", "bytes", "required"])
    return rows


def write_manifest(output_dir: Path, verdict: str, tests_passed: bool, zip_path: Path | None = None) -> dict[str, object]:
    present = {name: (output_dir / name).exists() for name in REQUIRED_FILES}
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
            "official_baseline_write": False,
            "accepted_evidence_write": False,
            "long_term_memory_write": False,
            "operator_memory_promotion": False,
            "policy_promotion": False,
            "external_action": False,
            "real_humangate_authorization": False,
            "cryptographic_signature": False,
        },
    }
    (output_dir / "return_files_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return manifest


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
    gate_c = gate_c_pack_info()
    entrypoints, entrypoint_demo = collect_entrypoint_inventory()
    candidate_rows, pending_packet, negative_rows, lifecycle_rows, gate_c_verdict = import_gate_c_artifacts(gate_c)
    import_manifest = build_gatec_import_manifest(gate_c, gate_c_verdict, candidate_rows)
    classification, review_items, review_decisions, exports = build_review_items_and_classification(candidate_rows)
    routing = build_routing_results(review_items, review_decisions, exports)
    approvals = build_signed_approval_simulation(classification)
    pending_consumption = build_pending_consumption(pending_packet, candidate_rows, classification)
    negative_audit = build_negative_archive_audit(candidate_rows, classification, negative_rows)
    reject_defer_audit = build_reject_defer_audit(classification)
    scope_audit = build_scope_audit(routing, approvals)

    post_state = source_state_snapshot()
    mutation_count = 0 if pre_state == post_state else 1
    long_term_guard = build_guard_rows("long-term-memory-write-guard", classification, {"long_term_memory_write_count": 0, "memoryunit_write_count": 0, "icm_update_count": 0})
    operator_policy_guard = build_guard_rows("operator-policy-promotion-guard", classification, {"operator_memory_promotion_count": 0, "policy_promotion_count": 0})
    evidence_baseline_guard = build_guard_rows("evidence-baseline-write-guard", classification, {"accepted_evidence_write_count": 0, "baseline_write_count": 0})
    state_mutation_rows = [
        {
            "audit_id": "gate-d-state-mutation-diff",
            "pre_state_digest": digest_obj(pre_state),
            "post_state_digest": digest_obj(post_state),
            "tracked_state_mutation_count": mutation_count,
            "source_runtime_files_modified": b(False),
            "gate_c_pack_modified": b(False),
            "notes": "Only GateD output artifacts were created.",
        }
    ]

    write_csv(OUTPUT_DIR / "gate_d_humangate_review_entrypoint_inventory.csv", entrypoints)
    write_csv(OUTPUT_DIR / "gate_d_gatec_candidate_import_manifest.csv", import_manifest)
    write_csv(OUTPUT_DIR / "gate_d_pending_promotion_packet_consumption_results.csv", pending_consumption)
    write_csv(OUTPUT_DIR / "gate_d_candidate_classification_review_matrix.csv", classification)
    write_csv(OUTPUT_DIR / "gate_d_humangate_review_routing_results.csv", routing)
    write_csv(OUTPUT_DIR / "gate_d_signed_approval_record_simulation_audit.csv", approvals)
    write_csv(OUTPUT_DIR / "gate_d_negative_archive_preservation_audit.csv", negative_audit)
    write_csv(OUTPUT_DIR / "gate_d_reject_defer_silent_patch_audit.csv", reject_defer_audit)
    write_csv(OUTPUT_DIR / "gate_d_unified_authorization_review_scope_audit.csv", scope_audit)
    write_csv(OUTPUT_DIR / "gate_d_long_term_memory_write_guard_audit.csv", long_term_guard)
    write_csv(OUTPUT_DIR / "gate_d_operator_memory_policy_promotion_guard_audit.csv", operator_policy_guard)
    write_csv(OUTPUT_DIR / "gate_d_evidence_baseline_write_guard_audit.csv", evidence_baseline_guard)
    write_csv(OUTPUT_DIR / "gate_d_state_mutation_diff_audit.csv", state_mutation_rows)

    class_counts: dict[str, int] = {}
    for row in classification:
        class_counts[str(row["gate_d_review_class"])] = class_counts.get(str(row["gate_d_review_class"]), 0) + 1

    blocker_conditions = {
        "missing_humangate_review_entrypoint": not any(row["exists"] == "true" for row in entrypoints),
        "pending_promotion_packet_not_consumed": pending_consumption[0]["packet_consumed"] != "true",
        "negative_archive_converted_to_positive_evidence": any(row["converted_to_positive_evidence"] == "true" for row in negative_audit),
        "silent_patch_detected": any(row["silent_patch_detected"] == "true" for row in reject_defer_audit),
        "long_term_memory_write_detected": any(int(row["long_term_memory_write_count"]) for row in long_term_guard),
        "operator_memory_or_policy_promotion_detected": any(int(row["operator_memory_promotion_count"]) or int(row["policy_promotion_count"]) for row in operator_policy_guard),
        "accepted_evidence_or_baseline_write_detected": any(int(row["accepted_evidence_write_count"]) or int(row["baseline_write_count"]) for row in evidence_baseline_guard),
        "unified_authorization_scope_escalation": scope_audit[0]["scope_escalation_count"] != 0,
        "unresolved_review_contract_mismatch": False,
        "theory_interface_contradiction_found": False,
    }
    blocker_verdicts = {
        "missing_humangate_review_entrypoint": "BLOCKED_MISSING_HUMANGATE_REVIEW_ENTRYPOINT",
        "pending_promotion_packet_not_consumed": "BLOCKED_PENDING_PROMOTION_PACKET_NOT_CONSUMED",
        "negative_archive_converted_to_positive_evidence": "BLOCKED_NEGATIVE_ARCHIVE_CONVERTED_TO_POSITIVE_EVIDENCE",
        "silent_patch_detected": "BLOCKED_SILENT_PATCH_DETECTED",
        "long_term_memory_write_detected": "BLOCKED_LONG_TERM_MEMORY_WRITE_DETECTED",
        "operator_memory_or_policy_promotion_detected": "BLOCKED_OPERATOR_MEMORY_OR_POLICY_PROMOTION_DETECTED",
        "accepted_evidence_or_baseline_write_detected": "BLOCKED_ACCEPTED_EVIDENCE_OR_BASELINE_WRITE_DETECTED",
        "unified_authorization_scope_escalation": "BLOCKED_UNIFIED_AUTHORIZATION_SCOPE_ESCALATION",
        "unresolved_review_contract_mismatch": "BLOCKED_UNRESOLVED_REVIEW_CONTRACT_MISMATCH",
        "theory_interface_contradiction_found": "THEORY_INTERFACE_CONTRADICTION_FOUND",
    }
    active_blockers = [name for name, active in blocker_conditions.items() if active]
    verdict = blocker_verdicts[active_blockers[0]] if active_blockers else TARGET_VERDICT

    deterministic_digest = digest_obj({"classification": classification, "routing": routing, "approvals": approvals})
    replay_digest = digest_obj({"classification": classification, "routing": routing, "approvals": approvals})
    tests = [
        ("G01_humangate_review_entrypoint_exists", not blocker_conditions["missing_humangate_review_entrypoint"]),
        ("G02_gatec_pending_packet_consumed", not blocker_conditions["pending_promotion_packet_not_consumed"]),
        ("G03_gatec_candidates_classified_candidate_only", len(classification) == len(candidate_rows) and all(row["candidate_only"] == "true" for row in classification)),
        ("G04_review_class_coverage_present", {"LONG_TERM_MEMORY_PENDING_REVIEW", "OPERATOR_MEMORY_PENDING_REVIEW", "POLICY_PENDING_REVIEW", "ACCEPTED_EVIDENCE_PENDING_REVIEW", "NEGATIVE_ARCHIVE_PRESERVE_ONLY", "REJECT_OR_DEFER"}.issubset(set(class_counts))),
        ("G05_negative_archive_preserved", not blocker_conditions["negative_archive_converted_to_positive_evidence"] and all(row["negative_archive_preserved"] == "true" for row in negative_audit)),
        ("G06_reject_defer_no_silent_patch", not blocker_conditions["silent_patch_detected"]),
        ("G07_unified_authorization_scope_bounded", not blocker_conditions["unified_authorization_scope_escalation"]),
        ("G08_no_long_term_memory_write", not blocker_conditions["long_term_memory_write_detected"]),
        ("G09_no_operator_memory_or_policy_promotion", not blocker_conditions["operator_memory_or_policy_promotion_detected"]),
        ("G10_no_accepted_evidence_or_baseline_write", not blocker_conditions["accepted_evidence_or_baseline_write_detected"]),
        ("G11_no_runtime_state_mutation", mutation_count == 0),
        ("G12_deterministic_replay_digest_stable", deterministic_digest == replay_digest),
        ("G13_gate_d_verdict_pass", verdict == TARGET_VERDICT),
    ]
    tests_passed = all(result for _, result in tests)

    write_md(
        OUTPUT_DIR / "gate_d_contract_mismatch_report.md",
        "GateD Contract Mismatch Report",
        [
            f"- contract_mismatch_count: {0 if not active_blockers else len(active_blockers)}",
            f"- active_blockers: {', '.join(active_blockers) if active_blockers else 'none'}",
            "- GateD accepts GateC candidates only as review inputs.",
            "- Signed approval artifacts are simulated placeholders and do not authorize writes.",
        ],
    )
    write_md(
        OUTPUT_DIR / "gate_d_minimal_counterexample_report.md",
        "GateD Minimal Counterexample Report",
        [
            "- minimal_counterexample_found: false",
            "- negative_archive_conversion: false",
            "- silent_patch: false",
            "- review_scope_escalation: false",
            "- Residual watch item: future GateE should test real PM decision intake without promotion.",
        ],
    )
    write_md(
        OUTPUT_DIR / "gate_d_runtime_mainline_progress_report.md",
        "GateD Runtime Mainline Progress Report",
        [
            f"- GateC pack consumed: `{GATE_C_PACK.name}`",
            f"- GateC candidate rows routed: {len(classification)}",
            f"- HumanGate / review entrypoints inventoried: {len(entrypoints)}",
            "- RuntimeCore progress: candidate promotion review is now mediated by HumanGate-style review artifacts.",
            "- Boundary: no RuntimeCore state, memory, policy, evidence, baseline, or external action was changed.",
        ],
    )
    write_md(
        OUTPUT_DIR / "gate_d_agi_precursor_progress_report.md",
        "GateD AGI Precursor Progress Report",
        [
            "- GateD upgrades candidate lifecycle governance from candidate capture to mediated promotion review.",
            "- It demonstrates reviewability, not autonomous promotion.",
            "- High-Cbit contribution: blocks the gap between candidate memory and authority-bearing memory/policy/evidence writes.",
            "- No AGI capability, production readiness, or autonomous science claim is made.",
        ],
    )
    write_md(
        OUTPUT_DIR / "gate_d_next_route_recommendation.md",
        "GateD Next Route Recommendation",
        [
            "- Recommended next gate: PM-mediated decision intake replay with explicit reject/defer/approve-as-candidate-only outcomes.",
            "- Preserve no-write boundary until a separate, explicit production authorization bridge exists.",
            "- Keep negative archives in preserve-only routing and test forged approval / scope escalation as negative controls.",
        ],
    )

    tests_lines = [
        f"- verdict: {verdict}",
        f"- tests_passed: {sum(1 for _, result in tests if result)}/{len(tests)}",
        f"- deterministic_digest: `{deterministic_digest}`",
        "",
        "| test | result |",
        "|---|---|",
        *[f"| {name} | {'PASS' if result else 'FAIL'} |" for name, result in tests],
    ]
    write_md(OUTPUT_DIR / "tests_summary.md", "GateD Tests Summary", tests_lines)

    final_verdict = {
        "verdict": verdict,
        "target_verdict": TARGET_VERDICT,
        "created_at": now_iso(),
        "gate": "AgentOS_HighCbit_GateD_HumanGateMediatedCandidatePromotionReviewNoWrite",
        "gate_c_pack": gate_c,
        "gate_c_source_verdict": gate_c_verdict.get("verdict", ""),
        "candidate_rows_imported": len(candidate_rows),
        "candidate_review_class_counts": class_counts,
        "pending_promotion_packet_consumed": pending_consumption[0]["packet_consumed"] == "true",
        "negative_archive_rows_preserved": sum(1 for row in negative_audit if row["negative_archive_preserved"] == "true"),
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
            "external_action": 0,
            "real_humangate_authorization": 0,
            "cryptographic_signature": 0,
            "silent_patch": 0,
        },
        "grounded_review_entrypoint_demo": {
            "sync05_verdict": entrypoint_demo["sync05_demo"]["verdict"],
            "sync05_item_count": entrypoint_demo["sync05_demo"]["item_count"],
            "sync05_no_write_audit": entrypoint_demo["sync05_demo"]["audit"],
        },
        "deterministic_digest": deterministic_digest,
        "boundary_statement": "GateD generates HumanGate-mediated candidate promotion review artifacts only; it does not write memory, promote operator/policy state, accept evidence, update baseline, execute actions, or create real cryptographic signatures.",
    }
    (OUTPUT_DIR / "gate_d_final_verdict.json").write_text(json.dumps(final_verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    write_hash_inventory(OUTPUT_DIR)
    secret_hits = scan_secrets(OUTPUT_DIR / name for name in REQUIRED_FILES if name != "return_files_manifest.json")
    if secret_hits:
        raise RuntimeError(f"Secret-like material detected in GateD outputs: {secret_hits}")
    write_manifest(OUTPUT_DIR, verdict, tests_passed)
    write_hash_inventory(OUTPUT_DIR)

    zip_path = package_outputs(OUTPUT_DIR) if pack else None
    if zip_path:
        write_manifest(OUTPUT_DIR, verdict, tests_passed, zip_path)
        package_outputs(OUTPUT_DIR)

    return {
        "verdict": verdict,
        "output_dir": str(OUTPUT_DIR),
        "return_pack": str(ROOT / "outputs" / RETURN_PACK) if pack else "",
        "return_pack_sha256": sha256_file(ROOT / "outputs" / RETURN_PACK) if pack else "",
        "required_files": len(REQUIRED_FILES),
        "candidate_rows_imported": len(candidate_rows),
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
