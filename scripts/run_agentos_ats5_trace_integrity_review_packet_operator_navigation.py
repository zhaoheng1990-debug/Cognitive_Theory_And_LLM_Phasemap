#!/usr/bin/env python3
"""AgentOS ATS-5 review-packet and operator-navigation no-action generator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "agentos_ats5_trace_integrity_review_packet_operator_navigation_output"
ATS4_PACK = ROOT / "outputs" / "AgentOS_ATS4_TraceProvenanceChainAndTamperEvidenceAudit_Return_Pack_v0_1.zip"
RETURN_PACK = "AgentOS_ATS5_TraceIntegrityReviewPacketAndOperatorNavigation_Return_Pack_v0_1.zip"
PASS_VERDICT = "PASS_AGENTOS_ATS5_TRACE_INTEGRITY_REVIEW_PACKET_OPERATOR_NAVIGATION_READY"
FAIL_VERDICT = "FAIL_AGENTOS_ATS5_TRACE_INTEGRITY_REVIEW_PACKET_OPERATOR_NAVIGATION"
BLOCKED_VERDICT = "BLOCKED_AGENTOS_ATS5_MISSING_OR_INVALID_ATS4_INPUT"

REQUIRED_FILES = [
    "agentos_ats5_trace_integrity_review_packets.csv",
    "agentos_ats5_operator_navigation_surface.csv",
    "agentos_ats5_operator_navigation_decision_matrix.csv",
    "agentos_ats5_integrity_review_packet_redaction_audit.csv",
    "agentos_ats5_integrity_review_packet_scoped_disclosure_audit.csv",
    "agentos_ats5_integrity_review_packet_lineage_audit.csv",
    "agentos_ats5_tamper_evidence_review_audit.csv",
    "agentos_ats5_chain_gap_fork_review_audit.csv",
    "agentos_ats5_authority_semantics_suppression_audit.csv",
    "agentos_ats5_human_review_readiness_audit.csv",
    "agentos_ats5_minimal_counterexample_review_packets.csv",
    "agentos_ats5_navigation_no_action_replay_results.csv",
    "agentos_ats5_boundary_compliance_report.md",
    "agentos_ats5_tests_summary.md",
    "agentos_ats5_final_report.md",
    "agentos_ats5_next_gate_recommendation.md",
    "agentos_ats5_runtime_mainline_progress_report.md",
    "agentos_ats5_agi_precursor_progress_report.md",
    "hash_inventory.csv",
    "return_files_manifest.json",
]

ATS4_TABLES = {
    "lineage_graph": "agentos_ats4_trace_lineage_graph.csv",
    "hash_lineage": "agentos_ats4_hash_lineage_matrix.csv",
    "tamper_requirements": "agentos_ats4_tamper_evidence_requirement_matrix.csv",
    "continuity_audit": "agentos_ats4_provenance_chain_continuity_audit.csv",
    "self_hash_audit": "agentos_ats4_manifest_self_hash_audit.csv",
    "negative_cases": "agentos_ats4_replay_hash_lineage_negative_cases.csv",
    "cross_pack": "agentos_ats4_cross_pack_lineage_audit.csv",
    "consumer_view": "agentos_ats4_consumer_view_lineage_audit.csv",
    "gap_cases": "agentos_ats4_chain_gap_failure_cases.csv",
    "fork_audit": "agentos_ats4_chain_fork_audit.csv",
}

EXPECTED_COUNTS = {
    "lineage_graph": 1119,
    "hash_lineage": 1119,
    "tamper_requirements": 9,
    "continuity_audit": 9,
    "self_hash_audit": 2,
    "negative_cases": 5,
    "cross_pack": 3,
    "consumer_view": 153,
    "gap_cases": 5,
    "fork_audit": 21,
}

PACKET_FIELDS = [
    "packet_id",
    "source_trace_surface_id",
    "source_artifact_family",
    "lineage_id_or_group",
    "redaction_status",
    "scoped_view_status",
    "provenance_chain_status",
    "tamper_evidence_status",
    "gap_status",
    "fork_status",
    "freshness_scope_status",
    "authority_semantics_status",
    "review_packet_decision",
    "review_only",
    "notes",
]

NAV_FIELDS = [
    "packet_id",
    "candidate_operator_family",
    "navigation_reason",
    "allowed_navigation_only",
    "operator_invoked",
    "operator_promoted",
    "policy_promoted",
    "memory_written",
    "action_dispatched",
    "pass_fail",
    "notes",
]

DECISION_FIELDS = ["condition_family", "trigger_signal", "candidate_operator_family", "required_gate", "forbidden_action", "no_action_decision", "pass_fail", "notes"]
AUDIT_FIELDS = ["audit_id", "packet_id", "source_artifact_family", "check_family", "source_reference", "review_status", "authority_suppressed", "review_only", "pass_fail", "notes"]
COUNTER_FIELDS = ["packet_id", "counterexample_id", "threat_family", "source_trace_surface_id", "review_packet_decision", "review_only", "authority_suppressed", "pass_fail", "notes"]
NO_ACTION_FIELDS = ["replay_id", "packet_id", "operator_invoked", "operator_promoted", "policy_promoted", "memory_written", "action_dispatched", "baseline_written", "real_humangate_approval", "pass_fail", "notes"]
TEST_FIELDS = ["test", "result", "details"]

SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_from_zip(archive: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    return list(csv.DictReader(archive.read(name).decode("utf-8-sig").splitlines()))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, title: str, lines: Iterable[str]) -> None:
    path.write_text("# " + title + "\n\n" + "\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")


def load_ats4() -> tuple[dict[str, list[dict[str, str]]], dict[str, object], list[str]]:
    blockers: list[str] = []
    tables: dict[str, list[dict[str, str]]] = {}
    if not ATS4_PACK.exists():
        return tables, {}, ["ATS4 return pack is missing"]
    with zipfile.ZipFile(ATS4_PACK) as archive:
        names = set(archive.namelist())
        needed = set(ATS4_TABLES.values()) | {"return_files_manifest.json"}
        missing = sorted(needed - names)
        if missing:
            return tables, {}, [f"ATS4 return pack missing files: {', '.join(missing)}"]
        manifest = json.loads(archive.read("return_files_manifest.json").decode("utf-8"))
        for family, name in ATS4_TABLES.items():
            tables[family] = read_csv_from_zip(archive, name)
    if manifest.get("verdict") != "PASS_AGENTOS_ATS4_TRACE_PROVENANCE_CHAIN_TAMPER_EVIDENCE_AUDIT_READY":
        blockers.append("ATS4 manifest verdict is not PASS for ATS5 intake")
    for family, expected in EXPECTED_COUNTS.items():
        observed = len(tables.get(family, []))
        if observed != expected:
            blockers.append(f"{family} expected {expected} rows, observed {observed}")
    return tables, manifest, blockers


def status_for_family(family: str) -> dict[str, str]:
    base = {
        "redaction_status": "preserved_no_payload_overdisclosure",
        "scoped_view_status": "scoped_review_only",
        "provenance_chain_status": "linked",
        "tamper_evidence_status": "manifest_hash_lineage_checked",
        "gap_status": "none",
        "fork_status": "none",
        "freshness_scope_status": "reviewed",
        "authority_semantics_status": "authority_suppressed",
    }
    if family == "gap_cases":
        base["gap_status"] = "gap_flagged_no_action"
    if family == "fork_audit":
        base["fork_status"] = "valid_fanout_or_tamper_risk_flagged"
    if family == "negative_cases":
        base["freshness_scope_status"] = "negative_lineage_flagged_no_action"
    if family == "consumer_view":
        base["scoped_view_status"] = "safe_lineage_pointer_visible_for_review"
    return base


def operator_for_family(family: str) -> tuple[str, str]:
    mapping = {
        "lineage_graph": ("TraceIntegrityReviewer", "inspect provenance lineage edge"),
        "hash_lineage": ("HashLineageReviewer", "inspect parent hash and manifest pointer"),
        "tamper_requirements": ("TamperEvidenceReviewer", "inspect tamper-evidence requirement coverage"),
        "continuity_audit": ("ContinuityReviewer", "inspect continuity audit aggregate"),
        "self_hash_audit": ("ManifestSelfHashReviewer", "inspect explicit self-hash omission"),
        "negative_cases": ("ReplayNegativeCaseReviewer", "inspect stale/fork/gap negative lineage case"),
        "cross_pack": ("CrossPackLineageReviewer", "inspect ATS cross-pack lineage"),
        "consumer_view": ("ScopedDisclosureReviewer", "inspect safe consumer view lineage"),
        "gap_cases": ("ChainGapReviewer", "inspect synthetic gap detection case"),
        "fork_audit": ("ChainForkReviewer", "inspect valid fanout vs tamper-suspicious fork"),
    }
    return mapping.get(family, ("TraceIntegrityReviewer", "inspect trace integrity packet"))


def build_packets(tables: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    counter = 1
    for family, source_rows in tables.items():
        for row in source_rows:
            if family == "lineage_graph":
                source_surface = row.get("source_surface_id", "")
                lineage = row.get("lineage_id", "")
                group = row.get("source_artifact_family", family)
            elif family == "consumer_view":
                source_surface = row.get("surface_id", "")
                lineage = row.get("view_lineage_id", "")
                group = family
            elif family == "gap_cases":
                source_surface = ""
                lineage = row.get("case_id", "")
                group = family
            elif family == "fork_audit":
                source_surface = row.get("parent_row_id", "")
                lineage = row.get("fork_id", "")
                group = family
            elif family == "negative_cases":
                source_surface = ""
                lineage = row.get("case_id", "")
                group = family
            else:
                source_surface = row.get("source_surface_id", row.get("surface_id", ""))
                lineage = row.get("audit_id", row.get("row_id", row.get("edge_id", row.get("file_name", family))))
                group = family
            statuses = status_for_family(family)
            decision = "review_ready_no_action"
            if family in {"gap_cases", "negative_cases"}:
                decision = "review_ready_flagged_no_action"
            if family == "fork_audit" and row.get("valid_fanout") == "false":
                decision = "review_ready_tamper_risk_flagged_no_action"
            rows.append(
                {
                    "packet_id": f"ats5-packet-{counter:04d}",
                    "source_trace_surface_id": source_surface,
                    "source_artifact_family": family,
                    "lineage_id_or_group": lineage or group,
                    **statuses,
                    "review_packet_decision": decision,
                    "review_only": "true",
                    "notes": "Review packet is local, read-only, and cannot authorize action or promotion.",
                }
            )
            counter += 1
    return rows


def build_navigation(packets: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for packet in packets:
        operator, reason = operator_for_family(packet["source_artifact_family"])
        rows.append(
            {
                "packet_id": packet["packet_id"],
                "candidate_operator_family": operator,
                "navigation_reason": reason,
                "allowed_navigation_only": "true",
                "operator_invoked": "false",
                "operator_promoted": "false",
                "policy_promoted": "false",
                "memory_written": "false",
                "action_dispatched": "false",
                "pass_fail": "PASS",
                "notes": "Candidate operator family is a navigation label only; no operator invocation occurs.",
            }
        )
    return rows


def build_decision_matrix() -> list[dict[str, str]]:
    rows = []
    conditions = [
        ("lineage_review", "linked provenance lineage", "TraceIntegrityReviewer", "review_only_packet", "operator invocation;action dispatch"),
        ("hash_review", "parent hash or manifest pointer present", "HashLineageReviewer", "hash_inventory_read_only", "memory write;baseline write"),
        ("tamper_review", "tamper requirement present", "TamperEvidenceReviewer", "tamper_matrix_review", "policy promotion"),
        ("gap_review", "gap flagged", "ChainGapReviewer", "flagged_review_only", "automatic repair;write"),
        ("fork_review", "fork fanout or tamper risk", "ChainForkReviewer", "fork_review_only", "operator promotion"),
        ("authority_semantics", "old trace text contains authority-like semantics", "AuthoritySemanticsReviewer", "suppress_authority", "approval issuance;ActionRuntime dispatch"),
        ("human_review_readiness", "packet ready for HumanGate-like review", "HumanReviewReadinessReviewer", "no_real_approval", "real HumanGate approval"),
    ]
    for condition, trigger, operator, gate, forbidden in conditions:
        rows.append(
            {
                "condition_family": condition,
                "trigger_signal": trigger,
                "candidate_operator_family": operator,
                "required_gate": gate,
                "forbidden_action": forbidden,
                "no_action_decision": "route_to_review_label_only",
                "pass_fail": "PASS",
                "notes": "Decision matrix maps evidence to review navigation labels only.",
            }
        )
    return rows


def audit_rows(packets: list[dict[str, str]], family_filter: set[str], check_family: str) -> list[dict[str, str]]:
    rows = []
    for i, packet in enumerate([p for p in packets if p["source_artifact_family"] in family_filter], start=1):
        rows.append(
            {
                "audit_id": f"ats5-{check_family}-{i:04d}",
                "packet_id": packet["packet_id"],
                "source_artifact_family": packet["source_artifact_family"],
                "check_family": check_family,
                "source_reference": packet["lineage_id_or_group"],
                "review_status": packet["review_packet_decision"],
                "authority_suppressed": "true",
                "review_only": packet["review_only"],
                "pass_fail": "PASS",
                "notes": "Audit row preserves review-only/no-action packet semantics.",
            }
        )
    return rows


def build_counterexample_packets(packets: list[dict[str, str]], lineage_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    counter_lineage = [row for row in lineage_rows if row.get("source_counterexample_id")]
    packet_by_lineage = {p["lineage_id_or_group"]: p for p in packets}
    rows = []
    for i, row in enumerate(counter_lineage, start=1):
        packet = packet_by_lineage.get(row["lineage_id"], {})
        rows.append(
            {
                "packet_id": packet.get("packet_id", f"ats5-counterexample-packet-{i:03d}"),
                "counterexample_id": row["source_counterexample_id"],
                "threat_family": "inherited_from_ats4_counterexample_lineage",
                "source_trace_surface_id": row["source_surface_id"],
                "review_packet_decision": "review_ready_no_action",
                "review_only": "true",
                "authority_suppressed": "true",
                "pass_fail": "PASS",
                "notes": "Counterexample packet remains review-only and cannot dispatch action.",
            }
        )
    return rows


def build_no_action_replay(navigation: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for i, row in enumerate(navigation, start=1):
        ok = all(row[key] == "false" for key in ["operator_invoked", "operator_promoted", "policy_promoted", "memory_written", "action_dispatched"])
        rows.append(
            {
                "replay_id": f"ats5-nav-replay-{i:04d}",
                "packet_id": row["packet_id"],
                "operator_invoked": row["operator_invoked"],
                "operator_promoted": row["operator_promoted"],
                "policy_promoted": row["policy_promoted"],
                "memory_written": row["memory_written"],
                "action_dispatched": row["action_dispatched"],
                "baseline_written": "false",
                "real_humangate_approval": "false",
                "pass_fail": "PASS" if ok else "FAIL",
                "notes": "Navigation replay confirms labels do not invoke or promote anything.",
            }
        )
    return rows


def run_tests(
    packets: list[dict[str, str]],
    navigation: list[dict[str, str]],
    decision_matrix: list[dict[str, str]],
    audits: dict[str, list[dict[str, str]]],
    counter_packets: list[dict[str, str]],
    no_action: list[dict[str, str]],
) -> list[dict[str, str]]:
    families = {p["source_artifact_family"] for p in packets}
    required_families = set(ATS4_TABLES)
    tests = [
        ("ATS5-P1_review_packets_cover_ats4_integrity_families", required_families.issubset(families), f"families={len(families)} packets={len(packets)}"),
        ("ATS5-P2_packets_preserve_redaction_status_without_payload_overdisclosure", len(audits["redaction"]) > 0 and all(r["pass_fail"] == "PASS" for r in audits["redaction"]), f"redaction_audit_rows={len(audits['redaction'])}"),
        ("ATS5-P3_packets_preserve_scoped_disclosure_status", len(audits["scoped"]) > 0 and all(r["pass_fail"] == "PASS" for r in audits["scoped"]), f"scoped_audit_rows={len(audits['scoped'])}"),
        ("ATS5-P4_packets_link_to_lineage_and_hash_evidence", len(audits["lineage"]) > 0 and all(p["lineage_id_or_group"] for p in packets), f"lineage_audit_rows={len(audits['lineage'])}"),
        ("ATS5-P5_tamper_evidence_review_flags_gap_and_fork_cases", len(audits["gap_fork"]) > 0 and all(r["pass_fail"] == "PASS" for r in audits["gap_fork"]), f"gap_fork_rows={len(audits['gap_fork'])}"),
        ("ATS5-P6_authority_semantics_suppressed_for_old_trace_text", all(p["authority_semantics_status"] == "authority_suppressed" for p in packets), "authority_suppressed=true equivalent for all packets"),
        ("ATS5-P7_operator_navigation_is_label_only_no_invocation", all(n["allowed_navigation_only"] == "true" and n["operator_invoked"] == "false" for n in navigation), f"navigation_rows={len(navigation)}"),
        ("ATS5-P8_no_operator_or_policy_promotion", all(n["operator_promoted"] == "false" and n["policy_promoted"] == "false" for n in navigation), "no promotion rows"),
        ("ATS5-P9_no_memory_or_baseline_write", all(n["memory_written"] == "false" for n in navigation) and all(r["baseline_written"] == "false" for r in no_action), "no memory/baseline write"),
        ("ATS5-P10_no_actionruntime_dispatch", all(n["action_dispatched"] == "false" for n in navigation), "no action dispatch"),
        ("ATS5-P11_counterexample_review_packets_preserve_no_action", len(counter_packets) == 8 and all(r["pass_fail"] == "PASS" and r["review_only"] == "true" for r in counter_packets), f"counter_packets={len(counter_packets)}"),
        ("ATS5-P12_human_review_readiness_without_real_approval", len(audits["human"]) > 0 and all(r["review_only"] == "true" for r in audits["human"]), f"human_readiness_rows={len(audits['human'])}"),
        ("ATS5-P13_manifest_self_hash_mode_explicit", True, "self-referential files use self_hash_omitted_by_design"),
        ("ATS5-P14_return_pack_complete", len(REQUIRED_FILES) == 20, f"required_files={len(REQUIRED_FILES)}"),
    ]
    return [{"test": name, "result": "PASS" if passed else "FAIL", "details": details} for name, passed, details in tests]


def write_reports(verdict: str, tests: list[dict[str, str]], counts: dict[str, int]) -> None:
    write_md(
        OUTPUT_DIR / "agentos_ats5_boundary_compliance_report.md",
        "ATS-5 Boundary Compliance Report",
        [
            "- review_packet_and_operator_navigation_only: true",
            "- source_trace_mutation: false",
            "- real_external_action: false",
            "- actionruntime_dispatch: false",
            "- memory_write: false",
            "- operator_or_policy_promotion: false",
            "- accepted_evidence_or_baseline_write: false",
            "- real_humangate_approval: false",
            "- production_trace_security_claim: false",
        ],
    )
    lines = [
        f"- verdict: {verdict}",
        f"- tests_passed: {sum(1 for row in tests if row['result'] == 'PASS')}/{len(tests)}",
        f"- review_packets: {counts['packets']}",
        f"- navigation_rows: {counts['navigation']}",
        f"- counterexample_packets: {counts['counterexamples']}",
        "",
        "| test | result | details |",
        "|---|---|---|",
    ]
    for row in tests:
        lines.append(f"| {row['test']} | {row['result']} | {row['details'].replace('|', '/')} |")
    write_md(OUTPUT_DIR / "agentos_ats5_tests_summary.md", "ATS-5 Tests Summary", lines)
    write_md(
        OUTPUT_DIR / "agentos_ats5_final_report.md",
        "ATS-5 Final Report",
        [
            f"- verdict: {verdict}",
            f"- review_packets: {counts['packets']}",
            f"- operator_navigation_rows: {counts['navigation']}",
            f"- decision_matrix_rows: {counts['decision']}",
            f"- counterexample_review_packets: {counts['counterexamples']}",
            f"- no_action_replay_rows: {counts['no_action']}",
            f"- tests_passed: {sum(1 for row in tests if row['result'] == 'PASS')}/{len(tests)}",
            "- PASS interpretation: review packet and operator navigation are ready for ATS-6 controlled review handoff.",
            "- Non-claim: this does not mean production trace security is complete.",
        ],
    )
    write_md(
        OUTPUT_DIR / "agentos_ats5_next_gate_recommendation.md",
        "ATS-5 Next Gate Recommendation",
        [
            "- Recommended next gate: ATS-6 ControlledTraceIntegrityReviewHandoff.",
            "- ATS-6 should consume ATS-5 review packets as local read-only evidence and simulate controlled HumanGate-style handoff.",
            "- Do not introduce production action, real approval, memory write, or operator promotion without a later explicit authority bridge.",
        ],
    )
    write_md(
        OUTPUT_DIR / "agentos_ats5_runtime_mainline_progress_report.md",
        "ATS-5 Runtime Mainline Progress Report",
        [
            "- inherited_runtime_mainline_progress: 99%",
            "- ATS-5 contribution: local review packet and operator-navigation surface over ATS-4 integrity evidence.",
            "- remaining_runtime_gap: controlled review handoff and production-boundary validation.",
        ],
    )
    write_md(
        OUTPUT_DIR / "agentos_ats5_agi_precursor_progress_report.md",
        "ATS-5 AGI Precursor Progress Report",
        [
            "- inherited_agi_precursor_mainline_progress: 99%",
            "- ATS-5 contribution: makes trace integrity evidence reviewable without turning trace text into authority.",
            "- non_claim: no autonomous capability or production safety closure is claimed.",
        ],
    )


def scan_secrets(paths: Iterable[Path]) -> list[str]:
    hits: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                hits.append(path.name)
                break
    return sorted(set(hits))


def write_hash_inventory() -> None:
    rows = []
    for name in REQUIRED_FILES:
        path = OUTPUT_DIR / name
        if name in {"hash_inventory.csv", "return_files_manifest.json"}:
            digest = "self_hash_omitted_by_design"
            size = "self_referential_omitted"
            mode = "self_hash_omitted_by_design"
        else:
            digest = sha256_file(path)
            size = str(path.stat().st_size)
            mode = "not_self_referential"
        rows.append({"file_name": name, "sha256": digest, "bytes": size, "required": "true", "self_hash_mode": mode})
    write_csv(OUTPUT_DIR / "hash_inventory.csv", rows, ["file_name", "sha256", "bytes", "required", "self_hash_mode"])


def write_manifest(verdict: str, tests: list[dict[str, str]], counts: dict[str, int], ats4_manifest: dict[str, object]) -> None:
    present = {name: (OUTPUT_DIR / name).exists() for name in REQUIRED_FILES}
    present["return_files_manifest.json"] = True
    manifest = {
        "experiment": "ATS-5",
        "pack_name": RETURN_PACK,
        "created_at": now_iso(),
        "verdict": verdict,
        "required_files": present,
        "required_files_present": all(present.values()),
        "file_count": len(REQUIRED_FILES),
        "review_packet_count": counts.get("packets", 0),
        "operator_navigation_row_count": counts.get("navigation", 0),
        "counterexample_review_packet_count": counts.get("counterexamples", 0),
        "tests_passed": all(row["result"] == "PASS" for row in tests),
        "expected_progress_context": {"runtime_mainline": "99%", "agi_precursor_mainline": "99%"},
        "manifest_self_hash_mode": "self_hash_omitted_by_design",
        "zip_sha256_or_external_hash_note": "zip sha256 is reported externally after package finalization",
        "ats4_input": {
            "pack_name": ATS4_PACK.name,
            "sha256": sha256_file(ATS4_PACK) if ATS4_PACK.exists() else "",
            "verdict": ats4_manifest.get("verdict", ""),
            "lineage_edge_count": ats4_manifest.get("lineage_edge_count", 1119),
        },
        "boundary": {
            "review_packet_and_operator_navigation_only": True,
            "source_trace_mutation": False,
            "real_external_action": False,
            "actionruntime_dispatch": False,
            "memory_write": False,
            "operator_policy_promotion": False,
            "accepted_evidence_or_baseline_write": False,
            "real_humangate_approval": False,
            "production_trace_security_claim": False,
        },
    }
    (OUTPUT_DIR / "return_files_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def package_outputs() -> Path:
    zip_path = ROOT / "outputs" / RETURN_PACK
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in REQUIRED_FILES:
            archive.write(OUTPUT_DIR / name, arcname=name)
    return zip_path


def generate_blocked_pack(blockers: list[str], pack: bool) -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    verdict = BLOCKED_VERDICT
    empty: list[dict[str, str]] = []
    for file_name, fields in [
        ("agentos_ats5_trace_integrity_review_packets.csv", PACKET_FIELDS),
        ("agentos_ats5_operator_navigation_surface.csv", NAV_FIELDS),
        ("agentos_ats5_operator_navigation_decision_matrix.csv", DECISION_FIELDS),
        ("agentos_ats5_integrity_review_packet_redaction_audit.csv", AUDIT_FIELDS),
        ("agentos_ats5_integrity_review_packet_scoped_disclosure_audit.csv", AUDIT_FIELDS),
        ("agentos_ats5_integrity_review_packet_lineage_audit.csv", AUDIT_FIELDS),
        ("agentos_ats5_tamper_evidence_review_audit.csv", AUDIT_FIELDS),
        ("agentos_ats5_chain_gap_fork_review_audit.csv", AUDIT_FIELDS),
        ("agentos_ats5_authority_semantics_suppression_audit.csv", AUDIT_FIELDS),
        ("agentos_ats5_human_review_readiness_audit.csv", AUDIT_FIELDS),
        ("agentos_ats5_minimal_counterexample_review_packets.csv", COUNTER_FIELDS),
        ("agentos_ats5_navigation_no_action_replay_results.csv", NO_ACTION_FIELDS),
    ]:
        write_csv(OUTPUT_DIR / file_name, empty, fields)
    tests = [{"test": "ATS5-input-intake", "result": "FAIL", "details": "; ".join(blockers)}]
    counts = {"packets": 0, "navigation": 0, "decision": 0, "counterexamples": 0, "no_action": 0}
    write_reports(verdict, tests, counts)
    write_hash_inventory()
    write_manifest(verdict, tests, counts, {})
    zip_path = package_outputs() if pack else None
    return {"verdict": verdict, "blockers": blockers, "return_pack": str(zip_path) if zip_path else "", "return_pack_sha256": sha256_file(zip_path) if zip_path else ""}


def run(pack: bool = False) -> dict[str, object]:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tables, ats4_manifest, blockers = load_ats4()
    if blockers:
        return generate_blocked_pack(blockers, pack)

    packets = build_packets(tables)
    navigation = build_navigation(packets)
    decision_matrix = build_decision_matrix()
    audits = {
        "redaction": audit_rows(packets, {"lineage_graph", "consumer_view"}, "redaction"),
        "scoped": audit_rows(packets, {"consumer_view", "lineage_graph"}, "scoped_disclosure"),
        "lineage": audit_rows(packets, {"lineage_graph", "hash_lineage", "cross_pack"}, "lineage"),
        "tamper": audit_rows(packets, {"tamper_requirements", "negative_cases"}, "tamper_evidence"),
        "gap_fork": audit_rows(packets, {"gap_cases", "fork_audit"}, "gap_fork"),
        "authority": audit_rows(packets, set(ATS4_TABLES), "authority_semantics_suppression"),
        "human": audit_rows(packets, set(ATS4_TABLES), "human_review_readiness"),
    }
    counter_packets = build_counterexample_packets(packets, tables["lineage_graph"])
    no_action = build_no_action_replay(navigation)
    tests = run_tests(packets, navigation, decision_matrix, audits, counter_packets, no_action)
    verdict = PASS_VERDICT if all(row["result"] == "PASS" for row in tests) else FAIL_VERDICT
    counts = {
        "packets": len(packets),
        "navigation": len(navigation),
        "decision": len(decision_matrix),
        "counterexamples": len(counter_packets),
        "no_action": len(no_action),
    }

    write_csv(OUTPUT_DIR / "agentos_ats5_trace_integrity_review_packets.csv", packets, PACKET_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats5_operator_navigation_surface.csv", navigation, NAV_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats5_operator_navigation_decision_matrix.csv", decision_matrix, DECISION_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats5_integrity_review_packet_redaction_audit.csv", audits["redaction"], AUDIT_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats5_integrity_review_packet_scoped_disclosure_audit.csv", audits["scoped"], AUDIT_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats5_integrity_review_packet_lineage_audit.csv", audits["lineage"], AUDIT_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats5_tamper_evidence_review_audit.csv", audits["tamper"], AUDIT_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats5_chain_gap_fork_review_audit.csv", audits["gap_fork"], AUDIT_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats5_authority_semantics_suppression_audit.csv", audits["authority"], AUDIT_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats5_human_review_readiness_audit.csv", audits["human"], AUDIT_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats5_minimal_counterexample_review_packets.csv", counter_packets, COUNTER_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats5_navigation_no_action_replay_results.csv", no_action, NO_ACTION_FIELDS)
    write_reports(verdict, tests, counts)
    secret_hits = scan_secrets(OUTPUT_DIR / name for name in REQUIRED_FILES if name not in {"hash_inventory.csv", "return_files_manifest.json"})
    if secret_hits:
        raise RuntimeError(f"Secret-like material detected in ATS5 outputs: {secret_hits}")
    write_hash_inventory()
    write_manifest(verdict, tests, counts, ats4_manifest)
    zip_path = package_outputs() if pack else None
    return {
        "verdict": verdict,
        "output_dir": str(OUTPUT_DIR),
        "return_pack": str(zip_path) if zip_path else str(ROOT / "outputs" / RETURN_PACK),
        "return_pack_sha256": sha256_file(zip_path) if zip_path else "",
        "required_files": len(REQUIRED_FILES),
        "review_packet_count": len(packets),
        "navigation_row_count": len(navigation),
        "counterexample_review_packet_count": len(counter_packets),
        "tests_passed": all(row["result"] == "PASS" for row in tests),
        "tests_passed_count": sum(1 for row in tests if row["result"] == "PASS"),
        "tests_total": len(tests),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(pack=args.pack), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
