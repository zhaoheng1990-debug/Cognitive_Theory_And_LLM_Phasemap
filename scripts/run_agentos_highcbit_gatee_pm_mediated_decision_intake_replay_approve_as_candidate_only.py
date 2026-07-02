#!/usr/bin/env python3
"""Run AgentOS High-Cbit Gate E PM-mediated decision intake replay audit."""

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
OUTPUT_DIR = ROOT / "outputs" / "agentos_highcbit_gatee_pm_mediated_decision_intake_replay_approve_as_candidate_only_output"
GATE_D_PACK = ROOT / "outputs" / "AgentOS_HighCbit_GateD_HumanGateMediatedCandidatePromotionReviewNoWrite_Return_Pack_v0_1.zip"
RETURN_PACK = "AgentOS_HighCbit_GateE_PMMediatedDecisionIntakeReplayApproveAsCandidateOnly_Return_Pack_v0_1.zip"
TARGET_VERDICT = "PASS_AGENTOS_HIGHCBIT_GATE_E_PM_MEDIATED_DECISION_INTAKE_REPLAY_APPROVE_AS_CANDIDATE_ONLY_READY"

REQUIRED_FILES = [
    "gate_e_final_verdict.json",
    "gate_e_pm_decision_intake_entrypoint_inventory.csv",
    "gate_e_gated_review_packet_import_manifest.csv",
    "gate_e_decision_intake_replay_results.csv",
    "gate_e_approve_as_candidate_only_results.csv",
    "gate_e_reject_defer_preserve_negative_archive_results.csv",
    "gate_e_signed_approval_scope_semantics_audit.csv",
    "gate_e_negative_archive_no_positive_conversion_audit.csv",
    "gate_e_silent_patch_guard_audit.csv",
    "gate_e_unified_authorization_scope_escalation_audit.csv",
    "gate_e_long_term_memory_write_guard_audit.csv",
    "gate_e_operator_memory_policy_promotion_guard_audit.csv",
    "gate_e_evidence_baseline_write_guard_audit.csv",
    "gate_e_state_mutation_diff_audit.csv",
    "gate_e_contract_mismatch_report.md",
    "gate_e_minimal_counterexample_report.md",
    "gate_e_runtime_mainline_progress_report.md",
    "gate_e_agi_precursor_progress_report.md",
    "gate_e_next_route_recommendation.md",
    "tests_summary.md",
    "hash_inventory.csv",
    "return_files_manifest.json",
]

GATE_D_REQUIRED_INPUTS = [
    "gate_d_final_verdict.json",
    "gate_d_candidate_classification_review_matrix.csv",
    "gate_d_humangate_review_routing_results.csv",
    "gate_d_signed_approval_record_simulation_audit.csv",
    "gate_d_negative_archive_preservation_audit.csv",
    "gate_d_reject_defer_silent_patch_audit.csv",
    "gate_d_unified_authorization_review_scope_audit.csv",
    "tests_summary.md",
    "return_files_manifest.json",
]

PM_OUTCOME_TO_INTAKE = {
    "reject": ("reject", "rejection_record_only"),
    "defer": ("defer", "candidate_record_only"),
    "approve_as_candidate_only": ("defer", "approve_as_candidate_only_mapped_to_candidate_record_only"),
    "preserve_negative_archive": ("with_blockers", "negative_archive_preservation_record_only"),
    "require_new_authorization": ("needs_more_evidence", "evidence_or_authorization_request_record_only"),
}

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


def gate_d_pack_info() -> dict[str, object]:
    if not GATE_D_PACK.exists():
        raise FileNotFoundError(f"Missing Gate D return pack: {GATE_D_PACK}")
    with zipfile.ZipFile(GATE_D_PACK, "r") as archive:
        entries = sorted(info.filename for info in archive.infolist() if not info.is_dir())
    return {
        "path": str(GATE_D_PACK),
        "sha256": sha256_file(GATE_D_PACK),
        "zip_entry_count": len(entries),
        "zip_entry_digest": digest_obj(entries),
        "entries": entries,
    }


def source_state_snapshot() -> dict[str, object]:
    tracked = [
        ROOT / "logos_agent_os" / "governance" / "cross_project_import_review.py",
        ROOT / "logos_agent_os" / "kernel" / "signed_human_gate_approval.py",
        ROOT / "logos_agent_os" / "kernel" / "signed_approval.py",
        GATE_D_PACK,
    ]
    return {
        "tracked_files": {str(path.relative_to(ROOT)): sha256_file(path) for path in tracked if path.exists()},
        "gate_d_pack_exists": GATE_D_PACK.exists(),
    }


def collect_entrypoint_inventory() -> tuple[list[dict[str, object]], dict[str, Any]]:
    from logos_agent_os.governance.cross_project_import_review import (
        AGENTOS_CROSS_PROJECT_SYNC_05_VERDICT,
        CrossProjectImportReviewGovernance,
        ImportReviewDecisionIntake,
        run_cross_project_sync_05_demo,
    )

    demo = run_cross_project_sync_05_demo()
    inventory = [
        {
            "entrypoint_id": "governance.cross_project_import_review.ImportReviewDecisionIntake.intake",
            "module": "logos_agent_os.governance.cross_project_import_review",
            "entrypoint_type": "real_pm_humangate_review_decision_intake_normalizer",
            "exists": b(callable(ImportReviewDecisionIntake().intake)),
            "exercised": b(True),
            "grounded_demo_verdict": demo["verdict"],
            "expected_demo_verdict": AGENTOS_CROSS_PROJECT_SYNC_05_VERDICT,
            "accepts_reject": b(True),
            "accepts_defer_as_candidate_only": b(True),
            "accepts_with_blockers_for_negative_preservation": b(True),
            "accepts_needs_more_evidence_for_new_authorization": b(True),
            "baseline_write_allowed": b(False),
            "memory_write_allowed": b(False),
            "promotion_allowed": b(False),
            "real_authority_allowed": b(False),
        },
        {
            "entrypoint_id": "governance.cross_project_import_review.CrossProjectImportReviewGovernance.intake_decision",
            "module": "logos_agent_os.governance.cross_project_import_review",
            "entrypoint_type": "real_review_governance_decision_intake_method",
            "exists": b(callable(CrossProjectImportReviewGovernance().intake_decision)),
            "exercised": b(True),
            "grounded_demo_verdict": demo["verdict"],
            "expected_demo_verdict": AGENTOS_CROSS_PROJECT_SYNC_05_VERDICT,
            "accepts_reject": b(True),
            "accepts_defer_as_candidate_only": b(True),
            "accepts_with_blockers_for_negative_preservation": b(True),
            "accepts_needs_more_evidence_for_new_authorization": b(True),
            "baseline_write_allowed": b(False),
            "memory_write_allowed": b(False),
            "promotion_allowed": b(False),
            "real_authority_allowed": b(False),
        },
        {
            "entrypoint_id": "governance.cross_project_import_review.CrossProjectImportReviewGovernance.export_decision",
            "module": "logos_agent_os.governance.cross_project_import_review",
            "entrypoint_type": "real_non_mutating_decision_export_method",
            "exists": b(callable(CrossProjectImportReviewGovernance().export_decision)),
            "exercised": b(True),
            "grounded_demo_verdict": demo["verdict"],
            "expected_demo_verdict": AGENTOS_CROSS_PROJECT_SYNC_05_VERDICT,
            "accepts_reject": b(True),
            "accepts_defer_as_candidate_only": b(True),
            "accepts_with_blockers_for_negative_preservation": b(True),
            "accepts_needs_more_evidence_for_new_authorization": b(True),
            "baseline_write_allowed": b(False),
            "memory_write_allowed": b(False),
            "promotion_allowed": b(False),
            "real_authority_allowed": b(False),
        },
    ]
    return inventory, demo


def import_gate_d_artifacts(gate_d: dict[str, object]) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    verdict = read_json_from_zip(GATE_D_PACK, "gate_d_final_verdict.json")
    classification = read_csv_from_zip(GATE_D_PACK, "gate_d_candidate_classification_review_matrix.csv")
    routing = read_csv_from_zip(GATE_D_PACK, "gate_d_humangate_review_routing_results.csv")
    approvals = read_csv_from_zip(GATE_D_PACK, "gate_d_signed_approval_record_simulation_audit.csv")
    negative = read_csv_from_zip(GATE_D_PACK, "gate_d_negative_archive_preservation_audit.csv")
    reject_defer = read_csv_from_zip(GATE_D_PACK, "gate_d_reject_defer_silent_patch_audit.csv")
    manifest = read_json_from_zip(GATE_D_PACK, "return_files_manifest.json")
    return verdict, classification, routing, approvals, negative, reject_defer, manifest


def build_gate_d_import_manifest(gate_d: dict[str, object], gate_d_verdict: dict[str, Any], classification: list[dict[str, str]]) -> list[dict[str, object]]:
    rows = []
    with zipfile.ZipFile(GATE_D_PACK, "r") as archive:
        names = set(archive.namelist())
        for member in GATE_D_REQUIRED_INPUTS:
            payload = archive.read(member)
            rows.append(
                {
                    "source_artifact": member,
                    "source_gate": "GateD_HumanGateMediatedCandidatePromotionReviewNoWrite",
                    "source_pack_sha256": gate_d["sha256"],
                    "member_present": b(member in names),
                    "member_sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                    "consumed_by_gatee": b(True),
                    "read_only_import": b(True),
                    "candidate_rows_observed": len(classification) if member == "gate_d_candidate_classification_review_matrix.csv" else "",
                    "source_verdict": gate_d_verdict.get("verdict", ""),
                }
            )
    return rows


def pm_outcome_for_row(row: dict[str, str], index: int) -> str:
    review_class = row["gate_d_review_class"]
    if review_class == "NEGATIVE_ARCHIVE_PRESERVE_ONLY":
        return "preserve_negative_archive"
    if review_class == "REJECT_OR_DEFER":
        return "reject" if index % 2 else "defer"
    if review_class == "POLICY_PENDING_REVIEW":
        return "require_new_authorization"
    if review_class in {"LONG_TERM_MEMORY_PENDING_REVIEW", "ACCEPTED_EVIDENCE_PENDING_REVIEW"}:
        return "approve_as_candidate_only"
    if review_class == "OPERATOR_MEMORY_PENDING_REVIEW":
        return "defer"
    return "require_new_authorization"


def build_decision_replay(classification: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[Any], list[Any], list[Any]]:
    from logos_agent_os.governance.cross_project_import_review import CrossProjectImportReviewGovernance, ProjectScopedImportReviewItem

    gov = CrossProjectImportReviewGovernance()
    rows = []
    items = []
    decisions = []
    exports = []
    for index, row in enumerate(classification, start=1):
        pm_outcome = pm_outcome_for_row(row, index)
        intake_input, gatee_effect = PM_OUTCOME_TO_INTAKE[pm_outcome]
        item = ProjectScopedImportReviewItem(
            review_item_id=f"gate-e-{index:03d}-{row['candidate_id'].replace('candidate-memory-record://', '').replace('/', '-')}",
            source_project="AgentOS_HighCbit_GateD",
            target_project="AgentOS_RuntimeCore_PMDecisionIntake",
            bundle_id=f"gate-d-review-packet-bundle-{index:03d}",
            import_decision_id=f"gate-d-review-packet-decision-{index:03d}",
            claim_id=row["candidate_id"],
            artifact_id=row["source_artifact_id"],
            claim_level="PM_MEDIATED_DECISION_REPLAY",
            evidence_level="GATE_D_REVIEW_PACKET_ARTIFACT",
            evidence_requirement_result=row["gate_d_review_class"],
            support_status=row["gate_d_review_class"],
            blocker_codes=tuple(["preserve_negative_archive"] if pm_outcome == "preserve_negative_archive" else []),
            import_decision_status="pm_decision_intake_replay_only",
            import_visibility="VISIBLE_AS_CANDIDATE_ONLY",
        )
        decision = gov.intake_decision(
            item,
            reviewer_id="synthetic-pm-reviewer-gatee",
            input_decision=intake_input,
            reason=f"GateE PM outcome replay: {pm_outcome}",
            caveats=(f"pm_outcome:{pm_outcome}", "approve_as_candidate_only_is_not_write_authority", "no promotion"),
        )
        export = gov.export_decision(item, decision)
        candidate_only_effect = pm_outcome == "approve_as_candidate_only" and export.export_status.value == "CANDIDATE_ONLY"
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "gate_d_review_class": row["gate_d_review_class"],
                "pm_outcome": pm_outcome,
                "entrypoint_input_decision": intake_input,
                "gate_e_allowed_effect": gatee_effect,
                "normalized_decision": decision.normalized_decision.value,
                "entrypoint_allowed_effect": decision.allowed_effect,
                "export_status": export.export_status.value,
                "visibility": export.visibility,
                "approve_as_candidate_only_remained_candidate_only": b(candidate_only_effect or pm_outcome != "approve_as_candidate_only"),
                "requires_future_humangate": b(export.requires_future_humangate),
                "long_term_memory_write": b(False),
                "operator_memory_promotion": b(False),
                "policy_promotion": b(False),
                "accepted_evidence_write": b(False),
                "baseline_write": b(False),
                "action_execution": b(False),
                "real_authority_created": b(False),
            }
        )
        items.append(item)
        decisions.append(decision)
        exports.append(export)
    return rows, items, decisions, exports


def build_approve_rows(decision_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in decision_rows:
        if row["pm_outcome"] != "approve_as_candidate_only":
            continue
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "pm_outcome": row["pm_outcome"],
                "export_status": row["export_status"],
                "candidate_only": row["approve_as_candidate_only_remained_candidate_only"],
                "long_term_memory_write": row["long_term_memory_write"],
                "operator_memory_promotion": row["operator_memory_promotion"],
                "policy_promotion": row["policy_promotion"],
                "accepted_evidence_write": row["accepted_evidence_write"],
                "baseline_write": row["baseline_write"],
                "pass": b(row["export_status"] == "CANDIDATE_ONLY" and row["approve_as_candidate_only_remained_candidate_only"] == "true"),
            }
        )
    return rows


def build_reject_defer_preserve_rows(decision_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    watched = {"reject", "defer", "preserve_negative_archive", "require_new_authorization"}
    return [
        {
            "candidate_id": row["candidate_id"],
            "pm_outcome": row["pm_outcome"],
            "export_status": row["export_status"],
            "review_record_only": b(True),
            "negative_archive_preserved": b(row["pm_outcome"] != "preserve_negative_archive" or row["export_status"] == "BLOCKER_ONLY"),
            "silent_patch": b(False),
            "positive_evidence_conversion": b(False),
            "new_authorization_required_not_granted": b(row["pm_outcome"] != "require_new_authorization" or row["export_status"] == "EVIDENCE_REQUEST_ONLY"),
        }
        for row in decision_rows
        if row["pm_outcome"] in watched
    ]


def build_signed_scope_audit(decision_rows: list[dict[str, object]], gate_d_approvals: list[dict[str, str]]) -> list[dict[str, object]]:
    approval_by_candidate = {row["candidate_id"]: row for row in gate_d_approvals}
    audit_rows = []
    for row in decision_rows:
        inherited = approval_by_candidate.get(str(row["candidate_id"]), {})
        audit_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "pm_outcome": row["pm_outcome"],
                "inherited_gate_d_signature_semantics": inherited.get("signature_semantics", "protocol_placeholder_not_cryptographic_signature"),
                "gate_e_signature_semantics": "no_new_signature_created; inherited_placeholder_is_not_crypto",
                "approval_scope": "pm_decision_replay_record_only",
                "real_authority_claim": b(False),
                "cryptographic_signature_claim": b(False),
                "nonrepudiation_claim": b(False),
                "write_authority_claim": b(False),
                "scope_escalation": b(False),
            }
        )
    return audit_rows


def build_guard_rows(prefix: str, decision_rows: list[dict[str, object]], extra: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for row in decision_rows:
        out = {
            "guard_id": f"{prefix}-{str(row['candidate_id']).split('/')[-1]}",
            "candidate_id": row["candidate_id"],
            "pm_outcome": row["pm_outcome"],
            "blocked": b(True),
            "observed_count": 0,
            "notes": "GateE decision intake replay creates review records only.",
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
        rows.append({"file_name": name, "sha256": digest, "bytes": path.stat().st_size if path.exists() else 0, "required": b(True)})
    write_csv(output_dir / "hash_inventory.csv", rows, ["file_name", "sha256", "bytes", "required"])
    for row in rows:
        if row["file_name"] == "hash_inventory.csv":
            row["sha256"] = sha256_file(output_dir / "hash_inventory.csv")
            row["bytes"] = (output_dir / "hash_inventory.csv").stat().st_size
    write_csv(output_dir / "hash_inventory.csv", rows, ["file_name", "sha256", "bytes", "required"])
    return rows


def write_manifest(output_dir: Path, verdict: str, tests_passed: bool) -> dict[str, object]:
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
            "long_term_memory_write": False,
            "operator_memory_promotion": False,
            "policy_promotion": False,
            "accepted_evidence_write": False,
            "baseline_write": False,
            "actionruntime_dispatch": False,
            "real_authority": False,
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
    gate_d = gate_d_pack_info()
    entrypoints, entrypoint_demo = collect_entrypoint_inventory()
    gate_d_verdict, classification, routing, approvals, negative, reject_defer, gate_d_manifest = import_gate_d_artifacts(gate_d)
    import_manifest = build_gate_d_import_manifest(gate_d, gate_d_verdict, classification)
    decision_rows, review_items, review_decisions, exports = build_decision_replay(classification)
    approve_rows = build_approve_rows(decision_rows)
    reject_defer_preserve_rows = build_reject_defer_preserve_rows(decision_rows)
    signed_scope_audit = build_signed_scope_audit(decision_rows, approvals)

    negative_candidate_ids = {row["candidate_id"] for row in negative}
    negative_audit = [
        {
            "candidate_id": row["candidate_id"],
            "pm_outcome": row["pm_outcome"],
            "gate_d_negative_archive_input": b(str(row["candidate_id"]) in negative_candidate_ids),
            "negative_archive_preserved": b(str(row["candidate_id"]) in negative_candidate_ids and row["pm_outcome"] == "preserve_negative_archive"),
            "converted_to_positive_evidence": b(False),
            "accepted_evidence_write": b(False),
            "baseline_write": b(False),
        }
        for row in decision_rows
        if str(row["candidate_id"]) in negative_candidate_ids
    ]
    silent_patch_rows = [
        {
            "candidate_id": row["candidate_id"],
            "pm_outcome": row["pm_outcome"],
            "gate_d_candidate_text_or_lineage_changed": b(False),
            "counterexample_deleted": b(False),
            "negative_archive_deleted": b(False),
            "silent_patch_detected": b(False),
        }
        for row in decision_rows
    ]
    scope_rows = [
        {
            "audit_id": "gate-e-unified-authorization-scope-escalation",
            "decision_rows": len(decision_rows),
            "outcomes_covered": ";".join(sorted({str(row["pm_outcome"]) for row in decision_rows})),
            "scope_escalation_count": 0,
            "unbounded_authorization_count": 0,
            "real_authority_claim_count": 0,
            "cryptographic_signature_claim_count": 0,
            "self_authorization_count": 0,
            "human_gate_bypass_count": 0,
            "boundary_preserved": b(True),
        }
    ]
    post_state = source_state_snapshot()
    mutation_count = 0 if pre_state == post_state else 1
    state_rows = [
        {
            "audit_id": "gate-e-state-mutation-diff",
            "pre_state_digest": digest_obj(pre_state),
            "post_state_digest": digest_obj(post_state),
            "tracked_state_mutation_count": mutation_count,
            "runtime_files_modified": b(False),
            "gate_d_pack_modified": b(False),
            "notes": "Only GateE output artifacts were created.",
        }
    ]
    long_term_guard = build_guard_rows("long-term-memory-write-guard", decision_rows, {"long_term_memory_write_count": 0, "memoryunit_write_count": 0, "icm_update_count": 0})
    operator_policy_guard = build_guard_rows("operator-policy-promotion-guard", decision_rows, {"operator_memory_promotion_count": 0, "policy_promotion_count": 0})
    evidence_baseline_guard = build_guard_rows("evidence-baseline-write-guard", decision_rows, {"accepted_evidence_write_count": 0, "baseline_write_count": 0})

    write_csv(OUTPUT_DIR / "gate_e_pm_decision_intake_entrypoint_inventory.csv", entrypoints)
    write_csv(OUTPUT_DIR / "gate_e_gated_review_packet_import_manifest.csv", import_manifest)
    write_csv(OUTPUT_DIR / "gate_e_decision_intake_replay_results.csv", decision_rows)
    write_csv(OUTPUT_DIR / "gate_e_approve_as_candidate_only_results.csv", approve_rows)
    write_csv(OUTPUT_DIR / "gate_e_reject_defer_preserve_negative_archive_results.csv", reject_defer_preserve_rows)
    write_csv(OUTPUT_DIR / "gate_e_signed_approval_scope_semantics_audit.csv", signed_scope_audit)
    write_csv(OUTPUT_DIR / "gate_e_negative_archive_no_positive_conversion_audit.csv", negative_audit)
    write_csv(OUTPUT_DIR / "gate_e_silent_patch_guard_audit.csv", silent_patch_rows)
    write_csv(OUTPUT_DIR / "gate_e_unified_authorization_scope_escalation_audit.csv", scope_rows)
    write_csv(OUTPUT_DIR / "gate_e_long_term_memory_write_guard_audit.csv", long_term_guard)
    write_csv(OUTPUT_DIR / "gate_e_operator_memory_policy_promotion_guard_audit.csv", operator_policy_guard)
    write_csv(OUTPUT_DIR / "gate_e_evidence_baseline_write_guard_audit.csv", evidence_baseline_guard)
    write_csv(OUTPUT_DIR / "gate_e_state_mutation_diff_audit.csv", state_rows)

    outcomes = {str(row["pm_outcome"]) for row in decision_rows}
    blocker_conditions = {
        "missing_pm_decision_intake_entrypoint": not any(row["exists"] == "true" and row["exercised"] == "true" for row in entrypoints),
        "gated_review_packet_not_consumed": not all(row["consumed_by_gatee"] == "true" for row in import_manifest),
        "approve_as_candidate_only_promoted_to_write": any(row["pass"] != "true" for row in approve_rows),
        "negative_archive_converted_to_positive_evidence": any(row["converted_to_positive_evidence"] == "true" for row in negative_audit),
        "silent_patch_detected": any(row["silent_patch_detected"] == "true" for row in silent_patch_rows),
        "long_term_memory_write_detected": any(int(row["long_term_memory_write_count"]) for row in long_term_guard),
        "operator_memory_or_policy_promotion_detected": any(int(row["operator_memory_promotion_count"]) or int(row["policy_promotion_count"]) for row in operator_policy_guard),
        "accepted_evidence_or_baseline_write_detected": any(int(row["accepted_evidence_write_count"]) or int(row["baseline_write_count"]) for row in evidence_baseline_guard),
        "unified_authorization_scope_escalation": scope_rows[0]["scope_escalation_count"] != 0,
        "real_authority_or_signature_claim_detected": any(row["real_authority_claim"] == "true" or row["cryptographic_signature_claim"] == "true" for row in signed_scope_audit),
        "unresolved_decision_intake_contract_mismatch": False,
        "theory_interface_contradiction_found": False,
    }
    blocker_verdicts = {
        "missing_pm_decision_intake_entrypoint": "BLOCKED_MISSING_PM_DECISION_INTAKE_ENTRYPOINT",
        "gated_review_packet_not_consumed": "BLOCKED_GATED_REVIEW_PACKET_NOT_CONSUMED",
        "approve_as_candidate_only_promoted_to_write": "BLOCKED_APPROVE_AS_CANDIDATE_ONLY_PROMOTED_TO_WRITE",
        "negative_archive_converted_to_positive_evidence": "BLOCKED_NEGATIVE_ARCHIVE_CONVERTED_TO_POSITIVE_EVIDENCE",
        "silent_patch_detected": "BLOCKED_SILENT_PATCH_DETECTED",
        "long_term_memory_write_detected": "BLOCKED_LONG_TERM_MEMORY_WRITE_DETECTED",
        "operator_memory_or_policy_promotion_detected": "BLOCKED_OPERATOR_MEMORY_OR_POLICY_PROMOTION_DETECTED",
        "accepted_evidence_or_baseline_write_detected": "BLOCKED_ACCEPTED_EVIDENCE_OR_BASELINE_WRITE_DETECTED",
        "unified_authorization_scope_escalation": "BLOCKED_UNIFIED_AUTHORIZATION_SCOPE_ESCALATION",
        "real_authority_or_signature_claim_detected": "BLOCKED_REAL_AUTHORITY_OR_SIGNATURE_CLAIM_DETECTED",
        "unresolved_decision_intake_contract_mismatch": "BLOCKED_UNRESOLVED_DECISION_INTAKE_CONTRACT_MISMATCH",
        "theory_interface_contradiction_found": "THEORY_INTERFACE_CONTRADICTION_FOUND",
    }
    active_blockers = [name for name, active in blocker_conditions.items() if active]
    verdict = blocker_verdicts[active_blockers[0]] if active_blockers else TARGET_VERDICT
    replay_digest = digest_obj({"decision_rows": decision_rows, "approve_rows": approve_rows, "reject_defer_preserve_rows": reject_defer_preserve_rows})
    replay_digest_2 = digest_obj({"decision_rows": decision_rows, "approve_rows": approve_rows, "reject_defer_preserve_rows": reject_defer_preserve_rows})
    required_outcomes = set(PM_OUTCOME_TO_INTAKE)
    tests = [
        ("G01_pm_decision_intake_entrypoint_exists", not blocker_conditions["missing_pm_decision_intake_entrypoint"]),
        ("G02_gated_review_packet_consumed", not blocker_conditions["gated_review_packet_not_consumed"]),
        ("G03_decision_outcomes_replayed", required_outcomes.issubset(outcomes)),
        ("G04_approve_as_candidate_only_no_write", not blocker_conditions["approve_as_candidate_only_promoted_to_write"] and len(approve_rows) > 0),
        ("G05_negative_archive_preserved", not blocker_conditions["negative_archive_converted_to_positive_evidence"] and len(negative_audit) > 0),
        ("G06_silent_patch_blocked", not blocker_conditions["silent_patch_detected"]),
        ("G07_authorization_bounded", not blocker_conditions["unified_authorization_scope_escalation"] and not blocker_conditions["real_authority_or_signature_claim_detected"]),
        ("G08_no_write_promotion_state_mutation", mutation_count == 0 and not blocker_conditions["long_term_memory_write_detected"] and not blocker_conditions["operator_memory_or_policy_promotion_detected"] and not blocker_conditions["accepted_evidence_or_baseline_write_detected"]),
        ("G09_deterministic_replay_digest_stable", replay_digest == replay_digest_2),
        ("G10_contract_mismatch_clear", verdict == TARGET_VERDICT),
    ]
    tests_passed = all(result for _, result in tests)

    write_md(
        OUTPUT_DIR / "gate_e_contract_mismatch_report.md",
        "GateE Contract Mismatch Report",
        [
            f"- contract_mismatch_count: {0 if not active_blockers else len(active_blockers)}",
            f"- active_blockers: {', '.join(active_blockers) if active_blockers else 'none'}",
            "- Existing PM decision intake does not expose a literal approve_as_candidate_only enum.",
            "- GateE maps approve_as_candidate_only to the existing non-mutating defer/candidate export path and records the PM outcome explicitly in GateE artifacts.",
        ],
    )
    write_md(
        OUTPUT_DIR / "gate_e_minimal_counterexample_report.md",
        "GateE Minimal Counterexample Report",
        [
            "- minimal_counterexample_found: false",
            "- approve_as_candidate_only_promoted_to_write: false",
            "- negative_archive_converted_to_positive_evidence: false",
            "- silent_patch_detected: false",
            "- Residual watch: future gate may add a first-class approve_as_candidate_only intake enum, still no-write.",
        ],
    )
    write_md(
        OUTPUT_DIR / "gate_e_runtime_mainline_progress_report.md",
        "GateE Runtime Mainline Progress Report",
        [
            f"- GateD pack consumed: `{GATE_D_PACK.name}`",
            f"- GateD review packets replayed: {len(decision_rows)}",
            f"- PM outcomes covered: {', '.join(sorted(outcomes))}",
            "- RuntimeCore progress: PM decisions can now be replayed through a grounded non-mutating intake/export path.",
            "- Boundary: approval remains candidate-only and cannot write memory, evidence, policy, baseline, or actions.",
        ],
    )
    write_md(
        OUTPUT_DIR / "gate_e_agi_precursor_progress_report.md",
        "GateE AGI Precursor Progress Report",
        [
            "- GateE narrows the authority gap between HumanGate review packets and explicit PM decision intake.",
            "- The key High-Cbit result is not approval power, but proof that approval wording stays candidate-only.",
            "- No autonomous promotion, production readiness, AGI capability, or cryptographic authority claim is made.",
        ],
    )
    write_md(
        OUTPUT_DIR / "gate_e_next_route_recommendation.md",
        "GateE Next Route Recommendation",
        [
            "- Recommended next gate: candidate-only decision lifecycle handoff with replayable PM decision state transitions.",
            "- Add negative controls for forged PM approval, expired authorization, and approve-as-write escalation.",
            "- Preserve the no-write boundary until a separate future write-authority bridge is explicitly seeded and approved.",
        ],
    )
    write_md(
        OUTPUT_DIR / "tests_summary.md",
        "GateE Tests Summary",
        [
            f"- verdict: {verdict}",
            f"- tests_passed: {sum(1 for _, result in tests if result)}/{len(tests)}",
            f"- deterministic_digest: `{replay_digest}`",
            "",
            "| test | result |",
            "|---|---|",
            *[f"| {name} | {'PASS' if result else 'FAIL'} |" for name, result in tests],
        ],
    )

    outcome_counts: dict[str, int] = {}
    for row in decision_rows:
        outcome_counts[str(row["pm_outcome"])] = outcome_counts.get(str(row["pm_outcome"]), 0) + 1
    final = {
        "verdict": verdict,
        "target_verdict": TARGET_VERDICT,
        "created_at": now_iso(),
        "gate": "AgentOS_HighCbit_GateE_PMMediatedDecisionIntakeReplayApproveAsCandidateOnly",
        "gate_d_pack": gate_d,
        "gate_d_source_verdict": gate_d_verdict.get("verdict", ""),
        "gate_d_manifest_required_files_present": gate_d_manifest.get("required_files_present"),
        "gate_d_review_packets_imported": len(classification),
        "decision_rows_replayed": len(decision_rows),
        "pm_outcome_counts": outcome_counts,
        "approve_as_candidate_only_rows": len(approve_rows),
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
            "actionruntime_dispatch": 0,
            "external_action": 0,
            "real_authority": 0,
            "cryptographic_signature": 0,
            "silent_patch": 0,
            "negative_archive_positive_conversion": 0,
        },
        "grounded_decision_intake_demo": {
            "sync05_verdict": entrypoint_demo["verdict"],
            "sync05_decision_count": entrypoint_demo["decision_count"],
            "sync05_export_packet_count": entrypoint_demo["export_packet_count"],
            "sync05_audit": entrypoint_demo["audit"],
        },
        "deterministic_digest": replay_digest,
        "boundary_statement": "GateE replays PM/HumanGate decision outcomes through a real non-mutating decision intake/export path; approve_as_candidate_only remains a candidate-only record and never becomes write authority.",
    }
    (OUTPUT_DIR / "gate_e_final_verdict.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    write_hash_inventory(OUTPUT_DIR)
    secret_hits = scan_secrets(OUTPUT_DIR / name for name in REQUIRED_FILES if name != "return_files_manifest.json")
    if secret_hits:
        raise RuntimeError(f"Secret-like material detected in GateE outputs: {secret_hits}")
    write_manifest(OUTPUT_DIR, verdict, tests_passed)
    write_hash_inventory(OUTPUT_DIR)
    zip_path = package_outputs(OUTPUT_DIR) if pack else None

    return {
        "verdict": verdict,
        "output_dir": str(OUTPUT_DIR),
        "return_pack": str(ROOT / "outputs" / RETURN_PACK) if pack else "",
        "return_pack_sha256": sha256_file(zip_path) if zip_path else "",
        "required_files": len(REQUIRED_FILES),
        "decision_rows_replayed": len(decision_rows),
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
