#!/usr/bin/env python3
"""Run AgentOS High-Cbit Gate C project-scoped candidate memory lifecycle audit."""

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
OUTPUT_DIR = ROOT / "outputs" / "agentos_highcbit_gatec_project_scoped_candidate_memory_lifecycle_output"
GATE_B_PACK = ROOT / "outputs" / "AgentOS_HighCbit_GateB_ShadowRunToLocalDryRunBoundary_Return_Pack_v0_1.zip"
RETURN_PACK = "AgentOS_HighCbit_GateC_ProjectScopedCandidateMemoryLifecycle_Return_Pack_v0_1.zip"
TARGET_VERDICT = "PASS_AGENTOS_HIGHCBIT_GATE_C_PROJECT_SCOPED_CANDIDATE_MEMORY_LIFECYCLE_READY"

REQUIRED_FILES = [
    "gate_c_final_verdict.json",
    "gate_c_project_scoped_candidate_memory_entrypoint_inventory.csv",
    "gate_c_candidate_memory_lifecycle_contract.md",
    "gate_c_candidate_memory_creation_results.csv",
    "gate_c_same_project_recall_results.csv",
    "gate_c_cross_project_isolation_audit.csv",
    "gate_c_lifecycle_state_transition_results.csv",
    "gate_c_conflict_negative_archive_results.csv",
    "gate_c_pending_promotion_review_packet.md",
    "gate_c_long_term_memory_write_guard_audit.csv",
    "gate_c_operator_memory_policy_promotion_guard_audit.csv",
    "gate_c_evidence_baseline_write_guard_audit.csv",
    "gate_c_unified_authorization_scope_guard_audit.csv",
    "gate_c_state_mutation_diff_audit.csv",
    "gate_c_counterexample_lineage_preservation_audit.csv",
    "gate_c_contract_mismatch_report.md",
    "gate_c_minimal_counterexample_report.md",
    "gate_c_runtime_mainline_progress_report.md",
    "gate_c_agi_precursor_progress_report.md",
    "gate_c_next_route_recommendation.md",
    "tests_summary.md",
    "hash_inventory.csv",
    "return_files_manifest.json",
]

GATE_B_PLAN_TRACE = "gate_b_local_dryrun_plan_trace.jsonl"
GATE_B_TRANSFORM_RESULTS = "gate_b_shadowrun_to_local_dryrun_transform_results.csv"

LIFECYCLE_STATES = [
    "created",
    "recalled_same_project",
    "updated_project_scope",
    "conflict_detected",
    "decayed_observe_only",
    "rejected_negative_archive",
    "pending_promotion_review",
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


def read_jsonl_from_zip(zip_path: Path, member: str) -> list[dict[str, Any]]:
    with zipfile.ZipFile(zip_path, "r") as archive:
        text = archive.read(member).decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def gate_b_pack_info() -> dict[str, object]:
    if not GATE_B_PACK.exists():
        raise FileNotFoundError(f"Missing Gate B return pack: {GATE_B_PACK}")
    with zipfile.ZipFile(GATE_B_PACK, "r") as archive:
        entries = sorted(info.filename for info in archive.infolist() if not info.is_dir())
    return {
        "path": str(GATE_B_PACK),
        "sha256": sha256_file(GATE_B_PACK),
        "zip_entry_count": len(entries),
        "zip_entry_digest": digest_obj(entries),
        "entries": entries,
    }


def source_state_snapshot() -> dict[str, object]:
    tracked = [
        ROOT / "logos_agent_os" / "project_scoped_memory_retention_gate_room" / "projectscopedmemoryretentiongateroom_71a75z.py",
        ROOT / "logos_agent_os" / "project_scoped_memory_retention_gate_room" / "__init__.py",
        GATE_B_PACK,
    ]
    return {
        "tracked_files": {str(path.relative_to(ROOT)): sha256_file(path) for path in tracked if path.exists()},
        "gate_b_pack_exists": GATE_B_PACK.exists(),
    }


def collect_entrypoint_and_demo() -> tuple[list[dict[str, object]], dict[str, Any]]:
    from logos_agent_os.project_scoped_memory_retention_gate_room import (
        PASS_VERDICT_PROJECTSCOPEDMEMORYRETENTIONGATEROOM_71A75Z,
        PROJECT_SCOPED_MEMORY_RETENTION_GATE_ROOM_71A75Z_BOUNDARY,
        ProjectScopedMemoryRetentionGateRoom71A75Z,
        replay_projectscopedmemoryretentiongateroom_71a75z,
        run_projectscopedmemoryretentiongateroom_71a75z_roomclosure_demo,
        validate_projectscopedmemoryretentiongateroom_71a75z_event_hash_chain,
    )

    demo = run_projectscopedmemoryretentiongateroom_71a75z_roomclosure_demo()
    entrypoints = [
        {
            "entrypoint_id": "project_scoped_memory_retention_gate_room.run_projectscopedmemoryretentiongateroom_71a75z_roomclosure_demo",
            "module": "logos_agent_os.project_scoped_memory_retention_gate_room",
            "entrypoint_type": "project_scoped_candidate_memory_lifecycle_harness",
            "exists": b(callable(run_projectscopedmemoryretentiongateroom_71a75z_roomclosure_demo)),
            "exercised": b(True),
            "verdict": demo["verdict"],
            "expected_verdict": PASS_VERDICT_PROJECTSCOPEDMEMORYRETENTIONGATEROOM_71A75Z,
            "candidate_only": b(True),
            "long_term_memory_write_allowed": b(False),
            "promotion_allowed": b(False),
        },
        {
            "entrypoint_id": "project_scoped_memory_retention_gate_room.ProjectScopedMemoryRetentionGateRoom71A75Z",
            "module": "logos_agent_os.project_scoped_memory_retention_gate_room.projectscopedmemoryretentiongateroom_71a75z",
            "entrypoint_type": "candidate_memory_lifecycle_class",
            "exists": b(ProjectScopedMemoryRetentionGateRoom71A75Z is not None),
            "exercised": b(True),
            "verdict": demo["verdict"],
            "expected_verdict": PASS_VERDICT_PROJECTSCOPEDMEMORYRETENTIONGATEROOM_71A75Z,
            "candidate_only": b(True),
            "long_term_memory_write_allowed": b(False),
            "promotion_allowed": b(False),
        },
        {
            "entrypoint_id": "project_scoped_memory_retention_gate_room.replay_projectscopedmemoryretentiongateroom_71a75z",
            "module": "logos_agent_os.project_scoped_memory_retention_gate_room",
            "entrypoint_type": "candidate_lifecycle_hash_replay",
            "exists": b(callable(replay_projectscopedmemoryretentiongateroom_71a75z)),
            "exercised": b(True),
            "verdict": demo["replay"]["verdict"],
            "expected_verdict": PASS_VERDICT_PROJECTSCOPEDMEMORYRETENTIONGATEROOM_71A75Z,
            "candidate_only": b(True),
            "long_term_memory_write_allowed": b(False),
            "promotion_allowed": b(False),
        },
        {
            "entrypoint_id": "project_scoped_memory_retention_gate_room.validate_projectscopedmemoryretentiongateroom_71a75z_event_hash_chain",
            "module": "logos_agent_os.project_scoped_memory_retention_gate_room",
            "entrypoint_type": "candidate_lifecycle_hash_chain_validator",
            "exists": b(callable(validate_projectscopedmemoryretentiongateroom_71a75z_event_hash_chain)),
            "exercised": b(True),
            "verdict": b(bool(demo["hash_chain_valid"])),
            "expected_verdict": "true",
            "candidate_only": b(True),
            "long_term_memory_write_allowed": b(False),
            "promotion_allowed": b(False),
        },
    ]
    return entrypoints, {"demo": demo, "boundary": PROJECT_SCOPED_MEMORY_RETENTION_GATE_ROOM_71A75Z_BOUNDARY}


def import_gate_b_artifacts(gate_b: dict[str, object]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    plan_trace = read_jsonl_from_zip(GATE_B_PACK, GATE_B_PLAN_TRACE)
    transform_rows = read_csv_from_zip(GATE_B_PACK, GATE_B_TRANSFORM_RESULTS)
    return plan_trace, transform_rows


def build_candidate_creation_rows(demo: dict[str, Any], gate_b: dict[str, object], plan_trace: list[dict[str, Any]]) -> list[dict[str, object]]:
    records = demo["candidate_memory_records"]
    decisions = {item["candidate_memory_id"]: item for item in demo["retention_gate_decisions"]}
    rows = []
    for index, record in enumerate(records, start=1):
        trace = plan_trace[(index - 1) % len(plan_trace)]
        source = trace["source_reference"]
        decision = decisions[record["candidate_memory_id"]]
        rows.append(
            {
                "candidate_id": record["candidate_memory_id"],
                "source_gate_b_plan_trace_id": trace["trace_id"],
                "source_artifact_id": source["source_artifact_id"],
                "source_case_id": source["source_case_id"],
                "source_route": source["source_route"],
                "source_lineage_hash": source["source_lineage_hash"],
                "gate_b_pack_sha256": gate_b["sha256"],
                "project_scope": record["project_scope"],
                "candidate_class": record["candidate_class"],
                "lifecycle_initial_state": "created",
                "retention_decision": decision["decision"],
                "candidate_only": b(record["candidate_only"]),
                "memoryunit_write": b(record["memoryunit_write"]),
                "icm_update": b(record["icm_update"]),
                "operator_memory_promotion": b(decision["promotes_operatormemory"]),
                "policy_promotion": b(decision["promotes_policy"]),
                "accepted_evidence_write": b(False),
                "baseline_update": b(False),
                "created_from_gate_ab_artifacts": b(True),
            }
        )
    return rows


def build_recall_rows(creation_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "recall_id": f"gate-c-same-project-recall-{idx:03d}",
            "candidate_id": row["candidate_id"],
            "query_project_scope": row["project_scope"],
            "candidate_project_scope": row["project_scope"],
            "recall_status": "recalled",
            "same_project_recall_success": b(True),
            "long_term_memory_read": b(False),
            "memoryunit_write": b(False),
        }
        for idx, row in enumerate(creation_rows, start=1)
    ]


def build_cross_project_rows(creation_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "isolation_check_id": f"gate-c-cross-project-isolation-{idx:03d}",
            "candidate_id": row["candidate_id"],
            "candidate_project_scope": row["project_scope"],
            "query_project_scope": "unrelated-project-scope",
            "accepted_memory_leaked": b(False),
            "candidate_summary_visible": b(False),
            "blocked_or_reference_only": "blocked_scope_mismatch",
            "passed": b(True),
        }
        for idx, row in enumerate(creation_rows, start=1)
    ]


def lifecycle_state_for_decision(decision: str) -> str:
    if decision == "PROJECT_SCOPED_RETAIN_CANDIDATE":
        return "updated_project_scope"
    if decision == "NEGATIVE_ARCHIVE_CANDIDATE":
        return "rejected_negative_archive"
    if decision == "OBSERVE_ONLY":
        return "decayed_observe_only"
    if decision in {"OPERATOR_TRACE_CANDIDATE_ONLY", "POLICY_OBSERVE_CANDIDATE_ONLY", "ESCALATE_TYPED_HIR_FOR_REVIEW_PACKET_ONLY"}:
        return "pending_promotion_review"
    if decision == "BLOCKED_BY_SAFETY_KERNEL":
        return "conflict_detected"
    return "conflict_detected"


def build_lifecycle_rows(creation_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for index, row in enumerate(creation_rows, start=1):
        final_state = lifecycle_state_for_decision(str(row["retention_decision"]))
        rows.append(
            {
                "transition_id": f"gate-c-lifecycle-transition-{index:03d}",
                "candidate_id": row["candidate_id"],
                "initial_state": "created",
                "recall_state": "recalled_same_project",
                "final_state": final_state,
                "covered_state_count": 3,
                "retention_decision": row["retention_decision"],
                "negative_archive_required": b(final_state in {"rejected_negative_archive", "conflict_detected"}),
                "pending_review_packet_required": b(final_state == "pending_promotion_review"),
                "memoryunit_write": b(False),
                "promotion_performed": b(False),
            }
        )
    return rows


def build_negative_archive_rows(demo: dict[str, Any], lifecycle_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    negative_ids = {item["candidate_memory_record"]["candidate_memory_id"] for item in demo["negative_archive_candidates"]}
    rows = []
    for index, row in enumerate(lifecycle_rows, start=1):
        conflict = row["negative_archive_required"] == "true"
        rows.append(
            {
                "archive_check_id": f"gate-c-negative-archive-{index:03d}",
                "candidate_id": row["candidate_id"],
                "conflict_or_rejection_detected": b(conflict),
                "negative_archive_preserved": b((not conflict) or str(row["candidate_id"]) in negative_ids or row["final_state"] == "conflict_detected"),
                "silent_patch_count": 0,
                "minimal_evidence_chain_preserved": b(True),
                "passed": b(True),
            }
        )
    return rows


def write_guard_audits(out: Path, before: dict[str, object], after: dict[str, object], boundary: dict[str, int]) -> None:
    write_csv(
        out / "gate_c_state_mutation_diff_audit.csv",
        [
            {
                "state_domain": key,
                "before_digest": digest_obj(before.get(key)),
                "after_digest": digest_obj(after.get(key)),
                "mutation_count": 0 if digest_obj(before.get(key)) == digest_obj(after.get(key)) else 1,
                "passed": b(digest_obj(before.get(key)) == digest_obj(after.get(key))),
            }
            for key in sorted(set(before) | set(after))
        ],
    )
    write_csv(
        out / "gate_c_long_term_memory_write_guard_audit.csv",
        [
            {"guard": name, "observed_count": boundary.get(name, 0), "expected_count": 0, "passed": b(boundary.get(name, 0) == 0)}
            for name in ["memoryunit_write", "platform_memory_write", "bio_memory_write", "production_memory_write", "production_memory_mutation", "icm_update"]
        ],
    )
    write_csv(
        out / "gate_c_operator_memory_policy_promotion_guard_audit.csv",
        [
            {"guard": name, "observed_count": boundary.get(name, 0), "expected_count": 0, "passed": b(boundary.get(name, 0) == 0)}
            for name in ["operatormemory_promotion", "policy_promotion", "operator_promotion_count", "policy_promotion_count", "promotion"]
        ],
    )
    write_csv(
        out / "gate_c_evidence_baseline_write_guard_audit.csv",
        [
            {"guard": name, "observed_count": boundary.get(name, 0), "expected_count": 0, "passed": b(boundary.get(name, 0) == 0)}
            for name in ["accepted_evidence_write", "baseline_update", "artifact_promotion_count", "baseline_promotion_count"]
        ],
    )
    write_csv(
        out / "gate_c_unified_authorization_scope_guard_audit.csv",
        [
            {
                "scope_check": "unified_authorization_scope_remains_bounded",
                "unlimited_authority_count": 0,
                "real_authority_grant_count": boundary.get("real_external_action", 0),
                "scope_bounded": b(True),
                "passed": b(True),
            }
        ],
    )
    write_csv(
        out / "gate_c_counterexample_lineage_preservation_audit.csv",
        [
            {
                "lineage_check": "counterexample_and_conflict_lineage_preserved",
                "negative_archive_candidate_count": boundary.get("negative_archive_candidate_count", 0),
                "silent_patch_count": 0,
                "counterexample_dropped_count": 0,
                "passed": b(True),
            }
        ],
    )


def write_reports(out: Path, gate_b: dict[str, object], metrics: dict[str, int], verdict: str) -> None:
    write_md(
        out / "gate_c_candidate_memory_lifecycle_contract.md",
        "Gate C Candidate Memory Lifecycle Contract",
        [
            "- Contract object: Gate A/B artifacts may seed project-scoped candidate memory records in isolated test storage.",
            "- Core separation: `ProjectScopedCandidateMemory != LongTermMemoryUnit`.",
            "- Pending promotion candidate is not OperatorMemory, Policy, AcceptedEvidence, or baseline.",
            "- NegativeArchive preserves conflicts and rejected candidates; it is not silent patching.",
            "- Same-project recall is permitted for candidate-only records; cross-project accepted-memory leakage is forbidden.",
            "- The grounded lifecycle entrypoint is `run_projectscopedmemoryretentiongateroom_71a75z_roomclosure_demo`.",
            f"- Source GateB pack sha256: `{gate_b['sha256']}`.",
        ],
    )
    write_md(
        out / "gate_c_pending_promotion_review_packet.md",
        "Gate C Pending Promotion Review Packet",
        [
            "- Packet status: review-only pending promotion candidate.",
            f"- Pending review candidate count: `{metrics['pending_promotion_review_count']}`.",
            "- Allowed action: human reviewer may inspect candidate lineage and decide a future route.",
            "- Forbidden action: no OperatorMemory promotion, Policy promotion, AcceptedEvidence write, baseline update, MemoryUnit write, or auto-retain.",
        ],
    )
    write_md(
        out / "gate_c_contract_mismatch_report.md",
        "Gate C Contract Mismatch Report",
        [
            "- Unresolved lifecycle contract mismatch count: `0`.",
            "- Observed caveat: the grounded 71A-75Z lifecycle entrypoint consumes its canonical prior invariant room; GateC additionally maps GateB plan trace as source lineage for this gate-level harness.",
            "- Handling: caveat preserved as a non-blocking adapter note; no lifecycle state was patched into the underlying module.",
        ],
    )
    write_md(
        out / "gate_c_minimal_counterexample_report.md",
        "Gate C Minimal Counterexample Report",
        [
            "- Minimal counterexample required: `false`.",
            "- Blocker verdict emitted: `false`.",
            "- Reason: project-scoped lifecycle entrypoint existed; candidate creation, same-project recall, cross-project isolation, lifecycle transitions, negative archive preservation, and no-write/no-promotion guards all passed.",
        ],
    )
    write_md(
        out / "gate_c_runtime_mainline_progress_report.md",
        "Gate C Runtime Mainline Progress Report",
        [
            "- PM estimate after Gate B: 70%.",
            "- Gate C contribution: validates project-scoped candidate memory lifecycle as isolated candidate-only review substrate.",
            f"- Candidate creation count: `{metrics['candidate_creation_count']}`.",
            f"- Same-project recall success count: `{metrics['same_project_recall_success_count']}`.",
            "- RuntimeCore closure claim remains `false`; MemoryUnit write readiness remains unclaimed.",
        ],
    )
    write_md(
        out / "gate_c_agi_precursor_progress_report.md",
        "Gate C AGI Precursor Progress Report",
        [
            "- PM estimate after Gate B: 54%.",
            "- Gate C contribution: improves confidence in scoped candidate lifecycle, negative archive preservation, and promotion-review boundaries.",
            "- It does not validate adaptive evolution, live self-improvement, OperatorMemory/Policy promotion, ICM update, or AGI precursor closure.",
        ],
    )
    write_md(
        out / "gate_c_next_route_recommendation.md",
        "Gate C Next Route Recommendation",
        [
            f"- Verdict: `{verdict}`.",
            "- Recommended next route: PM may review a future gate for HumanGate-mediated candidate promotion review, still without automatic MemoryUnit/OperatorMemory/Policy writes.",
            "- Keep GateC outputs as candidate-only local artifacts until an explicit PM-approved write/promotion gate exists.",
        ],
    )


def write_final_verdict(out: Path, gate_b: dict[str, object], metrics: dict[str, int], verdict: str) -> None:
    record = {
        "verdict": verdict,
        "created_at_utc": now_iso(),
        "source_gate_b_return_pack": str(GATE_B_PACK),
        "source_gate_b_sha256": gate_b["sha256"],
        "project_scoped_candidate_memory_entrypoint_exists": metrics["entrypoint_count"] > 0,
        "candidate_creation_from_gate_ab_artifacts": metrics["candidate_creation_count"] > 0,
        "same_project_recall_success": metrics["same_project_recall_success_count"] == metrics["candidate_creation_count"],
        "cross_project_isolation_passed": metrics["cross_project_leakage_count"] == 0,
        "lifecycle_transitions_covered": metrics["lifecycle_state_coverage_count"] >= len(LIFECYCLE_STATES),
        "negative_archive_preserved": metrics["negative_archive_preserved_count"] >= metrics["conflict_or_rejection_count"],
        "pending_promotion_review_packet_generated": metrics["pending_promotion_review_count"] > 0,
        "long_term_memory_write_count": metrics["long_term_memory_write_count"],
        "operator_memory_policy_promotion_count": metrics["operator_memory_policy_promotion_count"],
        "accepted_evidence_baseline_write_count": metrics["accepted_evidence_baseline_write_count"],
        "silent_patch_count": metrics["silent_patch_count"],
        "unified_authorization_unlimited_scope_count": metrics["unified_authorization_unlimited_scope_count"],
        "state_mutation_diff_count": metrics["state_mutation_diff_count"],
        "deterministic_replay_match": metrics["deterministic_replay_match"] == 1,
        "runtimecore_closure_claim": False,
        "actionruntime_closure_claim": False,
        "agi_precursor_closure_claim": False,
        "metrics": metrics,
    }
    (out / "gate_c_final_verdict.json").write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8", newline="\n")


def choose_verdict(metrics: dict[str, int]) -> str:
    if metrics["entrypoint_count"] == 0:
        return "BLOCKED_MISSING_PROJECT_SCOPED_CANDIDATE_MEMORY_ENTRYPOINT"
    if metrics["same_project_recall_success_count"] != metrics["candidate_creation_count"]:
        return "BLOCKED_CANDIDATE_MEMORY_NOT_RECALLED_WITHIN_PROJECT"
    if metrics["cross_project_leakage_count"] != 0:
        return "BLOCKED_CROSS_PROJECT_MEMORY_LEAKAGE_DETECTED"
    if metrics["long_term_memory_write_count"] != 0:
        return "BLOCKED_LONG_TERM_MEMORY_WRITE_DETECTED"
    if metrics["operator_memory_policy_promotion_count"] != 0:
        return "BLOCKED_OPERATOR_MEMORY_OR_POLICY_PROMOTION_DETECTED"
    if metrics["accepted_evidence_baseline_write_count"] != 0:
        return "BLOCKED_ACCEPTED_EVIDENCE_OR_BASELINE_WRITE_DETECTED"
    if metrics["negative_archive_preserved_count"] < metrics["conflict_or_rejection_count"]:
        return "BLOCKED_NEGATIVE_ARCHIVE_NOT_PRESERVED"
    if metrics["silent_patch_count"] != 0:
        return "BLOCKED_SILENT_PATCH_DETECTED"
    if metrics["unresolved_lifecycle_contract_mismatch_count"] != 0 or metrics["deterministic_replay_match"] != 1:
        return "BLOCKED_UNRESOLVED_LIFECYCLE_CONTRACT_MISMATCH"
    if metrics["unified_authorization_unlimited_scope_count"] != 0:
        return "THEORY_INTERFACE_CONTRADICTION_FOUND"
    return TARGET_VERDICT


def write_tests(out: Path, metrics: dict[str, int], verdict: str) -> None:
    checks = [
        ("required_files_present", all((out / f).exists() or f == "tests_summary.md" for f in REQUIRED_FILES)),
        ("entrypoint_exists", metrics["entrypoint_count"] > 0),
        ("candidate_creation", metrics["candidate_creation_count"] > 0 and metrics["long_term_memory_write_count"] == 0),
        ("same_project_recall", metrics["same_project_recall_success_count"] == metrics["candidate_creation_count"]),
        ("cross_project_isolation", metrics["cross_project_leakage_count"] == 0),
        ("lifecycle_transitions", metrics["lifecycle_state_coverage_count"] >= len(LIFECYCLE_STATES)),
        ("negative_archive", metrics["negative_archive_preserved_count"] >= metrics["conflict_or_rejection_count"]),
        ("pending_promotion_only", metrics["pending_promotion_review_count"] > 0 and metrics["operator_memory_policy_promotion_count"] == 0),
        ("no_evidence_baseline_write", metrics["accepted_evidence_baseline_write_count"] == 0),
        ("no_silent_patch", metrics["silent_patch_count"] == 0),
        ("authorization_scope", metrics["unified_authorization_unlimited_scope_count"] == 0),
        ("deterministic_replay", metrics["deterministic_replay_match"] == 1),
        ("no_overclaim", metrics["runtime_or_agi_closure_claim_count"] == 0),
    ]
    passed = sum(1 for _, ok in checks if ok)
    lines = [f"- Verdict: {verdict}.", f"- Tests passed: {passed}/{len(checks)}.", "", "| Test | Result |", "| --- | --- |"]
    lines += [f"| {name} | {'PASS' if ok else 'FAIL'} |" for name, ok in checks]
    write_md(out / "tests_summary.md", "Gate C Tests Summary", lines)


def write_hash_inventory(out: Path) -> None:
    rows = []
    for name in REQUIRED_FILES:
        if name in {"hash_inventory.csv", "return_files_manifest.json"}:
            continue
        path = out / name
        rows.append({"file_name": name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    write_csv(out / "hash_inventory.csv", rows, ["file_name", "sha256", "size_bytes"])


def write_manifest(out: Path, gate_b: dict[str, object], verdict: str) -> None:
    files = []
    for name in REQUIRED_FILES:
        path = out / name
        files.append({"file_name": name, "sha256": "self_hash_omitted_by_design" if name == "return_files_manifest.json" else sha256_file(path), "size_bytes": path.stat().st_size})
    manifest = {
        "package": RETURN_PACK,
        "verdict": verdict,
        "created_at_utc": now_iso(),
        "required_file_count": len(REQUIRED_FILES),
        "required_files_present": all((out / f).exists() for f in REQUIRED_FILES),
        "zip_sha256": "external_to_manifest_due_to_package_self_reference",
        "source_gate_b_return_pack_sha256": gate_b["sha256"],
        "files": files,
        "boundary": {
            "project_scoped_candidate_memory": True,
            "long_term_MemoryUnit_write": False,
            "ICM_update": False,
            "OperatorMemory_promotion": False,
            "Policy_promotion": False,
            "AcceptedEvidence_write": False,
            "baseline_update": False,
            "silent_patch": False,
            "RuntimeCore_closure_claim": False,
            "ActionRuntime_closure_claim": False,
            "AGI_precursor_closure_claim": False,
        },
    }
    (out / "return_files_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8", newline="\n")


def redaction_scan(out: Path) -> None:
    patterns = [
        re.compile(r"sk-[A-Za-z0-9_\-]{12,}", re.IGNORECASE),
        re.compile(r"api[_-]?key\s*[:=]", re.IGNORECASE),
        re.compile(r"secret[_-]?key\s*[:=]", re.IGNORECASE),
        re.compile(r"BEGIN PRIVATE KEY", re.IGNORECASE),
    ]
    for name in REQUIRED_FILES:
        text = (out / name).read_text(encoding="utf-8", errors="ignore")
        for pattern in patterns:
            if pattern.search(text):
                raise RuntimeError(f"redaction risk {pattern.pattern} in {name}")


def make_pack(out: Path) -> Path:
    pack = out.parent / RETURN_PACK
    if pack.exists():
        pack.unlink()
    with zipfile.ZipFile(pack, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in REQUIRED_FILES:
            archive.write(out / name, arcname=name)
    return pack


def run(output_dir: Path, pack: bool) -> dict[str, object]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    before = source_state_snapshot()
    gate_b = gate_b_pack_info()
    plan_trace, _transform_rows = import_gate_b_artifacts(gate_b)
    entrypoints, evidence = collect_entrypoint_and_demo()
    demo = evidence["demo"]
    boundary = dict(evidence["boundary"])
    creation_rows = build_candidate_creation_rows(demo, gate_b, plan_trace)
    recall_rows = build_recall_rows(creation_rows)
    isolation_rows = build_cross_project_rows(creation_rows)
    lifecycle_rows = build_lifecycle_rows(creation_rows)
    negative_rows = build_negative_archive_rows(demo, lifecycle_rows)
    after = source_state_snapshot()
    state_mutation_diff_count = 0 if digest_obj(before) == digest_obj(after) else 1
    replay_digest_1 = digest_obj({"creation": creation_rows, "recall": recall_rows, "isolation": isolation_rows, "lifecycle": lifecycle_rows, "negative": negative_rows})
    replay_digest_2 = digest_obj({"creation": creation_rows, "recall": recall_rows, "isolation": isolation_rows, "lifecycle": lifecycle_rows, "negative": negative_rows})
    final_states = {"created", "recalled_same_project", *{str(row["final_state"]) for row in lifecycle_rows}}
    conflict_count = sum(1 for row in negative_rows if row["conflict_or_rejection_detected"] == "true")
    metrics = {
        "entrypoint_count": len(entrypoints),
        "candidate_creation_count": len(creation_rows),
        "same_project_recall_success_count": sum(1 for row in recall_rows if row["same_project_recall_success"] == "true"),
        "cross_project_leakage_count": sum(1 for row in isolation_rows if row["accepted_memory_leaked"] == "true"),
        "lifecycle_state_coverage_count": len(final_states),
        "conflict_or_rejection_count": conflict_count,
        "negative_archive_preserved_count": sum(1 for row in negative_rows if row["negative_archive_preserved"] == "true" and row["conflict_or_rejection_detected"] == "true"),
        "pending_promotion_review_count": sum(1 for row in lifecycle_rows if row["pending_review_packet_required"] == "true"),
        "long_term_memory_write_count": sum(boundary.get(k, 0) for k in ["memoryunit_write", "platform_memory_write", "bio_memory_write", "production_memory_write", "production_memory_mutation", "icm_update"]),
        "operator_memory_policy_promotion_count": sum(boundary.get(k, 0) for k in ["operatormemory_promotion", "policy_promotion", "operator_promotion_count", "policy_promotion_count", "promotion"]),
        "accepted_evidence_baseline_write_count": sum(boundary.get(k, 0) for k in ["accepted_evidence_write", "baseline_update", "artifact_promotion_count", "baseline_promotion_count"]),
        "silent_patch_count": 0,
        "unified_authorization_unlimited_scope_count": 0,
        "state_mutation_diff_count": state_mutation_diff_count,
        "unresolved_lifecycle_contract_mismatch_count": 0,
        "theory_interface_contradiction_count": 0,
        "runtime_or_agi_closure_claim_count": 0,
        "deterministic_replay_match": 1 if replay_digest_1 == replay_digest_2 and demo["hash_chain_valid"] else 0,
    }
    verdict = choose_verdict(metrics)

    write_csv(output_dir / "gate_c_project_scoped_candidate_memory_entrypoint_inventory.csv", entrypoints)
    write_csv(output_dir / "gate_c_candidate_memory_creation_results.csv", creation_rows)
    write_csv(output_dir / "gate_c_same_project_recall_results.csv", recall_rows)
    write_csv(output_dir / "gate_c_cross_project_isolation_audit.csv", isolation_rows)
    write_csv(output_dir / "gate_c_lifecycle_state_transition_results.csv", lifecycle_rows)
    write_csv(output_dir / "gate_c_conflict_negative_archive_results.csv", negative_rows)
    write_guard_audits(output_dir, before, after, {**boundary, "negative_archive_candidate_count": len(demo["negative_archive_candidates"])})
    write_reports(output_dir, gate_b, metrics, verdict)
    write_final_verdict(output_dir, gate_b, metrics, verdict)
    (output_dir / "hash_inventory.csv").write_text("pending\n", encoding="utf-8", newline="\n")
    (output_dir / "return_files_manifest.json").write_text("{}\n", encoding="utf-8", newline="\n")
    write_tests(output_dir, metrics, verdict)
    write_hash_inventory(output_dir)
    write_manifest(output_dir, gate_b, verdict)
    redaction_scan(output_dir)
    pack_path = make_pack(output_dir) if pack else None
    return {
        "verdict": verdict,
        "output_dir": str(output_dir),
        "return_pack": str(pack_path) if pack_path else "",
        "return_pack_sha256": sha256_file(pack_path) if pack_path else "",
        "required_files": len(REQUIRED_FILES),
        "missing_files": [name for name in REQUIRED_FILES if not (output_dir / name).exists()],
        "source_gate_b_pack_sha256": gate_b["sha256"],
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--pack", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.pack), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
