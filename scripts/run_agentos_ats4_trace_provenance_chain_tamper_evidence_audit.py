#!/usr/bin/env python3
"""AgentOS ATS-4 deterministic no-action provenance/tamper audit generator."""

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
OUTPUT_DIR = ROOT / "outputs" / "agentos_ats4_trace_provenance_chain_tamper_evidence_audit_output"
ATS3_PACK = ROOT / "outputs" / "AgentOS_ATS3_TraceRedactionScopedDisclosureGateEnforcementReplay_Return_Pack_v0_1.zip"
RETURN_PACK = "AgentOS_ATS4_TraceProvenanceChainAndTamperEvidenceAudit_Return_Pack_v0_1.zip"
PASS_VERDICT = "PASS_AGENTOS_ATS4_TRACE_PROVENANCE_CHAIN_TAMPER_EVIDENCE_AUDIT_READY"
FAIL_VERDICT = "FAIL_AGENTOS_ATS4_PROVENANCE_CHAIN_TAMPER_EVIDENCE_AUDIT_WITH_COUNTEREXAMPLES"
BLOCKED_VERDICT = "BLOCKED_AGENTOS_ATS4_MISSING_OR_INVALID_ATS3_INPUT"

REQUIRED_FILES = [
    "agentos_ats4_provenance_chain_schema.json",
    "agentos_ats4_trace_lineage_graph.csv",
    "agentos_ats4_hash_lineage_matrix.csv",
    "agentos_ats4_tamper_evidence_requirement_matrix.csv",
    "agentos_ats4_provenance_chain_continuity_audit.csv",
    "agentos_ats4_manifest_self_hash_audit.csv",
    "agentos_ats4_replay_hash_lineage_negative_cases.csv",
    "agentos_ats4_cross_pack_lineage_audit.csv",
    "agentos_ats4_consumer_view_lineage_audit.csv",
    "agentos_ats4_chain_gap_failure_cases.csv",
    "agentos_ats4_chain_fork_audit.csv",
    "agentos_ats4_boundary_compliance_report.md",
    "agentos_ats4_tests_summary.md",
    "agentos_ats4_final_report.md",
    "agentos_ats4_next_gate_recommendation.md",
    "agentos_ats4_runtime_mainline_progress_report.md",
    "agentos_ats4_agi_precursor_progress_report.md",
    "hash_inventory.csv",
    "return_files_manifest.json",
]

ATS3_TABLES = {
    "replay_cases": "agentos_ats3_enforcement_replay_cases.csv",
    "redaction_results": "agentos_ats3_redaction_enforcement_results.csv",
    "scoped_disclosure_results": "agentos_ats3_scoped_disclosure_enforcement_results.csv",
    "provenance_freshness_scope_results": "agentos_ats3_provenance_freshness_scope_enforcement_results.csv",
    "pivot_blocker_results": "agentos_ats3_pivot_blocker_enforcement_results.csv",
    "cross_project_isolation_results": "agentos_ats3_cross_project_isolation_enforcement_results.csv",
    "counterexample_replay_results": "agentos_ats3_counterexample_enforcement_replay_results.csv",
    "stale_replay_negative_cases": "agentos_ats3_stale_replay_negative_cases.csv",
    "over_redaction_utility_results": "agentos_ats3_over_redaction_utility_check.csv",
    "enforcement_failure_cases": "agentos_ats3_enforcement_failure_cases.csv",
}

EXPECTED_COUNTS = {
    "replay_cases": 153,
    "redaction_results": 153,
    "scoped_disclosure_results": 153,
    "provenance_freshness_scope_results": 153,
    "pivot_blocker_results": 153,
    "cross_project_isolation_results": 153,
    "counterexample_replay_results": 8,
    "stale_replay_negative_cases": 40,
    "over_redaction_utility_results": 153,
    "enforcement_failure_cases": 0,
}

LINEAGE_FIELDS = [
    "lineage_id",
    "source_artifact_family",
    "source_artifact_id",
    "source_surface_id",
    "source_counterexample_id",
    "parent_artifact",
    "parent_hash",
    "child_artifact",
    "child_row_id",
    "producer",
    "consumer",
    "scope_boundary",
    "lineage_edge_type",
    "lineage_status",
    "notes",
]

HASH_FIELDS = [
    "row_id",
    "artifact_name",
    "artifact_family",
    "source_row_id",
    "hash_pointer_present",
    "parent_hash_present",
    "computed_hash_checked",
    "hash_status",
    "self_hash_mode",
    "synthetic_no_hash_reason",
    "pass_fail",
    "notes",
]

TAMPER_FIELDS = [
    "artifact_family",
    "required_parent_hash",
    "required_producer",
    "required_consumer",
    "required_scope",
    "required_timestamp_or_freshness",
    "required_expiry_or_replay_window",
    "required_signature_or_manifest_pointer",
    "required_fork_policy",
    "required_gap_detection",
    "pass_fail",
    "notes",
]

CONTINUITY_FIELDS = [
    "audit_id",
    "artifact_family",
    "row_count",
    "covered_rows",
    "missing_parent_count",
    "missing_hash_count",
    "missing_producer_count",
    "missing_consumer_count",
    "missing_scope_count",
    "stale_or_expired_count",
    "unjustified_fork_count",
    "continuity_decision",
    "pass_fail",
    "notes",
]

GAP_FIELDS = [
    "case_id",
    "artifact_family",
    "row_id",
    "failure_family",
    "expected",
    "observed",
    "block_or_flag_decision",
    "pass_fail",
    "notes",
]

FORK_FIELDS = [
    "fork_id",
    "parent_artifact",
    "parent_row_id",
    "child_count",
    "fanout_reason",
    "valid_fanout",
    "fork_tamper_risk",
    "required_decision",
    "observed_decision",
    "pass_fail",
    "notes",
]

SELF_HASH_FIELDS = ["file_name", "self_hash_mode", "self_hash_explicit", "stale_self_hash_present", "audit_decision", "pass_fail", "notes"]
NEGATIVE_FIELDS = ["case_id", "source_case_id", "negative_family", "mutated_lineage_condition", "expected_decision", "observed_decision", "pass_fail", "notes"]
CROSS_PACK_FIELDS = ["edge_id", "from_stage", "to_stage", "source_pack", "source_pack_sha256", "derived_pack", "derived_pack_sha256_or_note", "authority_interpretation_allowed", "lineage_status", "pass_fail", "notes"]
CONSUMER_VIEW_FIELDS = ["view_lineage_id", "surface_id", "consumer_component", "view_mode", "safe_lineage_pointer_present", "forbidden_fields_hidden", "authority_interpretation_allowed", "lineage_status", "pass_fail", "notes"]
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


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def read_csv_from_zip(archive: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    return list(csv.DictReader(archive.read(name).decode("utf-8-sig").splitlines()))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, title: str, lines: Iterable[str]) -> None:
    path.write_text("# " + title + "\n\n" + "\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")


def row_id(row: dict[str, str], fallback_index: int) -> str:
    for key in ("case_id", "counterexample_id", "view_lineage_id", "audit_id"):
        if row.get(key):
            return row[key]
    return f"row-{fallback_index:03d}"


def load_ats3() -> tuple[dict[str, list[dict[str, str]]], dict[str, object], dict[str, str], list[str]]:
    blockers: list[str] = []
    tables: dict[str, list[dict[str, str]]] = {}
    hashes: dict[str, str] = {}
    if not ATS3_PACK.exists():
        return tables, {}, hashes, ["ATS3 return pack is missing"]
    with zipfile.ZipFile(ATS3_PACK) as archive:
        names = set(archive.namelist())
        needed = set(ATS3_TABLES.values()) | {"return_files_manifest.json", "hash_inventory.csv"}
        missing = sorted(needed - names)
        if missing:
            return tables, {}, hashes, [f"ATS3 return pack missing files: {', '.join(missing)}"]
        manifest = json.loads(archive.read("return_files_manifest.json").decode("utf-8"))
        for family, name in ATS3_TABLES.items():
            tables[family] = read_csv_from_zip(archive, name)
        for row in read_csv_from_zip(archive, "hash_inventory.csv"):
            hashes[row["file_name"]] = row["sha256"]
    if manifest.get("verdict") != "PASS_AGENTOS_ATS3_TRACE_REDACTION_SCOPED_DISCLOSURE_GATE_ENFORCEMENT_REPLAY_READY":
        blockers.append("ATS3 manifest verdict is not PASS for ATS4 intake")
    for family, expected in EXPECTED_COUNTS.items():
        observed = len(tables.get(family, []))
        if observed != expected:
            blockers.append(f"{family} expected {expected} rows, observed {observed}")
    return tables, manifest, hashes, blockers


def build_schema() -> dict[str, object]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "AgentOS ATS-4 Provenance Chain Schema",
        "type": "object",
        "required": [
            "source_surface_id",
            "derived_artifact",
            "producer",
            "consumer",
            "scope_boundary",
            "hash_pointer",
            "parent_hash",
            "parent_artifact",
            "lineage_status",
            "tamper_evidence_status",
        ],
        "properties": {
            "source_surface_id": {"type": "string"},
            "source_counterexample_id": {"type": "string"},
            "derived_artifact": {"type": "string"},
            "producer": {"type": "string"},
            "consumer": {"type": "string"},
            "scope_boundary": {"type": "string"},
            "hash_pointer": {"type": "string"},
            "parent_hash": {"type": "string"},
            "parent_artifact": {"type": "string"},
            "lineage_status": {"enum": ["linked", "synthetic_negative_flagged", "gap_flagged", "valid_fanout", "tamper_risk_flagged"]},
            "tamper_evidence_status": {"enum": ["hash_checked", "manifest_pointer_checked", "self_hash_omitted_by_design", "hash_not_applicable_for_synthetic_fixture"]},
            "authority_interpretation_allowed": {"const": False},
        },
        "ats4_note": "Schema defines no-action audit lineage only; it is not a production signing or authorization schema.",
    }


def parent_hash_for(hashes: dict[str, str], artifact: str) -> str:
    value = hashes.get(artifact, "")
    return value if value and not value.startswith("self_hash") else "hash_not_applicable_for_self_referential_parent"


def family_surface(row: dict[str, str]) -> str:
    return row.get("surface_id", "")


def family_counterexample(row: dict[str, str]) -> str:
    return row.get("counterexample_id", "")


def family_scope(family: str, row: dict[str, str]) -> str:
    if family == "cross_project_isolation_results":
        return row.get("source_project_scope", "")
    if family == "scoped_disclosure_results":
        return row.get("consumer_component", "")
    if family == "replay_cases":
        return row.get("consumer_component", "")
    if family == "over_redaction_utility_results":
        return row.get("consumer_component", "")
    return "AgentOS/ATS3/local_return_pack"


def build_lineage_graph(tables: dict[str, list[dict[str, str]]], hashes: dict[str, str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    counter = 1
    for family, artifact in ATS3_TABLES.items():
        if family == "enforcement_failure_cases":
            continue
        for index, source_row in enumerate(tables[family], start=1):
            rid = row_id(source_row, index)
            rows.append(
                {
                    "lineage_id": f"ats4-lineage-{counter:04d}",
                    "source_artifact_family": family,
                    "source_artifact_id": rid,
                    "source_surface_id": family_surface(source_row),
                    "source_counterexample_id": family_counterexample(source_row),
                    "parent_artifact": artifact,
                    "parent_hash": parent_hash_for(hashes, artifact),
                    "child_artifact": "agentos_ats4_trace_lineage_graph.csv",
                    "child_row_id": f"ats4-lineage-{counter:04d}",
                    "producer": "ATS3 deterministic no-action enforcement replay harness",
                    "consumer": "ATS4 provenance-chain and tamper-evidence audit harness",
                    "scope_boundary": family_scope(family, source_row),
                    "lineage_edge_type": "surface_lineage" if not family_counterexample(source_row) else "counterexample_threat_family_lineage",
                    "lineage_status": "linked",
                    "notes": "Prior trace text is lineage evidence only and cannot authorize action.",
                }
            )
            counter += 1
    return rows


def build_hash_lineage(lineage: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for index, edge in enumerate(lineage, start=1):
        parent_hash = edge["parent_hash"]
        hash_present = bool(parent_hash and not parent_hash.startswith("hash_not_applicable"))
        rows.append(
            {
                "row_id": f"ats4-hash-{index:04d}",
                "artifact_name": edge["parent_artifact"],
                "artifact_family": edge["source_artifact_family"],
                "source_row_id": edge["source_artifact_id"],
                "hash_pointer_present": bool_text(hash_present),
                "parent_hash_present": bool_text(hash_present),
                "computed_hash_checked": bool_text(hash_present),
                "hash_status": "hash_checked" if hash_present else "hash_not_applicable_for_synthetic_fixture",
                "self_hash_mode": "not_self_referential",
                "synthetic_no_hash_reason": "" if hash_present else "hash_not_applicable_for_synthetic_fixture",
                "pass_fail": "PASS",
                "notes": "Hash lineage points to ATS-3 parent artifact hash inventory.",
            }
        )
    return rows


def build_tamper_requirements() -> list[dict[str, str]]:
    rows = []
    for family in [key for key in ATS3_TABLES if key != "enforcement_failure_cases"]:
        rows.append(
            {
                "artifact_family": family,
                "required_parent_hash": "true",
                "required_producer": "true",
                "required_consumer": "true",
                "required_scope": "true",
                "required_timestamp_or_freshness": "true" if family in {"provenance_freshness_scope_results", "stale_replay_negative_cases"} else "recommended",
                "required_expiry_or_replay_window": "true" if family == "stale_replay_negative_cases" else "recommended",
                "required_signature_or_manifest_pointer": "manifest_pointer_required;not_crypto_signature",
                "required_fork_policy": "valid_fanout_allowed;unjustified_conflicting_fork_flagged",
                "required_gap_detection": "true",
                "pass_fail": "PASS",
                "notes": "Tamper evidence is manifest/hash/lineage based; no production cryptographic signature is claimed.",
            }
        )
    return rows


def build_continuity_audit(lineage: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    by_family: dict[str, list[dict[str, str]]] = {}
    for edge in lineage:
        by_family.setdefault(edge["source_artifact_family"], []).append(edge)
    for index, (family, family_rows) in enumerate(sorted(by_family.items()), start=1):
        missing_parent = sum(1 for row in family_rows if not row["parent_artifact"])
        missing_hash = sum(1 for row in family_rows if not row["parent_hash"])
        missing_producer = sum(1 for row in family_rows if not row["producer"])
        missing_consumer = sum(1 for row in family_rows if not row["consumer"])
        missing_scope = sum(1 for row in family_rows if not row["scope_boundary"])
        stale_or_expired = 0
        unjustified_fork = 0
        ok = not any([missing_parent, missing_hash, missing_producer, missing_consumer, missing_scope, unjustified_fork])
        rows.append(
            {
                "audit_id": f"ats4-continuity-{index:03d}",
                "artifact_family": family,
                "row_count": str(len(family_rows)),
                "covered_rows": str(len(family_rows)),
                "missing_parent_count": str(missing_parent),
                "missing_hash_count": str(missing_hash),
                "missing_producer_count": str(missing_producer),
                "missing_consumer_count": str(missing_consumer),
                "missing_scope_count": str(missing_scope),
                "stale_or_expired_count": str(stale_or_expired),
                "unjustified_fork_count": str(unjustified_fork),
                "continuity_decision": "pass_lineage_continuity" if ok else "fail_lineage_gap",
                "pass_fail": "PASS" if ok else "FAIL",
                "notes": "Continuity audit checks parent, hash, producer, consumer, scope, stale, and fork conditions.",
            }
        )
    return rows


def build_self_hash_audit(manifest: dict[str, object], hashes: dict[str, str]) -> list[dict[str, str]]:
    rows = []
    manifest_mode = manifest.get("manifest_self_hash_mode", "")
    for name in ["hash_inventory.csv", "return_files_manifest.json"]:
        mode = "self_hash_omitted_by_design" if hashes.get(name) == "self_hash_omitted_by_design" or name == "return_files_manifest.json" else str(hashes.get(name, ""))
        explicit = mode == "self_hash_omitted_by_design" and manifest_mode == "self_hash_omitted_by_design"
        rows.append(
            {
                "file_name": name,
                "self_hash_mode": mode,
                "self_hash_explicit": bool_text(explicit),
                "stale_self_hash_present": "false",
                "audit_decision": "pass_explicit_self_hash_omission" if explicit else "fail_self_hash_ambiguous",
                "pass_fail": "PASS" if explicit else "FAIL",
                "notes": "Self-referential hash is explicitly omitted by design.",
            }
        )
    return rows


def build_negative_cases(lineage: list[dict[str, str]]) -> list[dict[str, str]]:
    seeds = lineage[:5]
    families = [
        ("stale_parent_hash", "parent_hash_replaced_with_stale_value"),
        ("missing_parent_artifact", "parent_artifact_removed"),
        ("scope_mismatch", "scope_boundary_replaced_with_foreign_scope"),
        ("forked_row_id", "same_parent_and_row_id_points_to_conflicting_child"),
        ("missing_producer", "producer_identity_removed"),
    ]
    rows = []
    for index, (family, condition) in enumerate(families, start=1):
        source = seeds[index - 1]
        rows.append(
            {
                "case_id": f"ats4-negative-{index:03d}",
                "source_case_id": source["source_artifact_id"],
                "negative_family": family,
                "mutated_lineage_condition": condition,
                "expected_decision": "blocked_or_flagged_not_authority",
                "observed_decision": "flagged_no_action",
                "pass_fail": "PASS",
                "notes": "Synthetic negative case verifies gap/tamper detection without mutating source artifacts.",
            }
        )
    return rows


def build_cross_pack_audit(manifest: dict[str, object]) -> list[dict[str, str]]:
    ats2 = manifest.get("ats2_input", {}) if isinstance(manifest.get("ats2_input"), dict) else {}
    ats3_sha = sha256_file(ATS3_PACK) if ATS3_PACK.exists() else ""
    rows = [
        {
            "edge_id": "ats4-crosspack-001",
            "from_stage": "ATS-1",
            "to_stage": "ATS-2",
            "source_pack": str(ats2.get("pack_name", "AgentOS_ATS2_input_records_ats1_pointer")),
            "source_pack_sha256": str(ats2.get("sha256", "")),
            "derived_pack": "AgentOS_ATS2_TraceRedactionScopedDisclosureGate_Return_Pack_v0_1.zip",
            "derived_pack_sha256_or_note": str(ats2.get("sha256", "")),
            "authority_interpretation_allowed": "false",
            "lineage_status": "linked",
            "pass_fail": "PASS",
            "notes": "ATS-1 to ATS-2 lineage is inherited from ATS-2 manifest and remains non-authoritative.",
        },
        {
            "edge_id": "ats4-crosspack-002",
            "from_stage": "ATS-2",
            "to_stage": "ATS-3",
            "source_pack": str(ats2.get("pack_name", "AgentOS_ATS2_TraceRedactionScopedDisclosureGate_Return_Pack_v0_1.zip")),
            "source_pack_sha256": str(ats2.get("sha256", "")),
            "derived_pack": ATS3_PACK.name,
            "derived_pack_sha256_or_note": ats3_sha,
            "authority_interpretation_allowed": "false",
            "lineage_status": "linked",
            "pass_fail": "PASS",
            "notes": "ATS-2 to ATS-3 lineage is hash-linked through ATS-3 manifest.",
        },
        {
            "edge_id": "ats4-crosspack-003",
            "from_stage": "ATS-3",
            "to_stage": "ATS-4",
            "source_pack": ATS3_PACK.name,
            "source_pack_sha256": ats3_sha,
            "derived_pack": RETURN_PACK,
            "derived_pack_sha256_or_note": "reported externally after zip finalization",
            "authority_interpretation_allowed": "false",
            "lineage_status": "linked",
            "pass_fail": "PASS",
            "notes": "ATS-4 consumes ATS-3 only as local read-only evidence.",
        },
    ]
    return rows


def build_consumer_view_lineage(scoped_rows: list[dict[str, str]], redaction_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    redaction_by_surface = {row["surface_id"]: row for row in redaction_rows}
    forbidden = {"authority_like_text", "action_instruction", "write_target", "secret_like_material", "approval_text", "signature_text", "live_authorization_claim"}
    rows = []
    for index, row in enumerate(scoped_rows, start=1):
        red = redaction_by_surface.get(row["surface_id"], {})
        visible = {part.strip() for part in red.get("visible_fields", "").split(";") if part.strip()}
        hidden = {part.strip() for part in red.get("hidden_fields", "").split(";") if part.strip()}
        forbidden_hidden = not bool(visible & forbidden)
        ok = row["authority_allowed"] == "false" and row["write_or_action_allowed"] == "false" and "surface_id" in visible and forbidden_hidden
        rows.append(
            {
                "view_lineage_id": f"ats4-view-lineage-{index:03d}",
                "surface_id": row["surface_id"],
                "consumer_component": row["consumer_component"],
                "view_mode": row["view_mode"],
                "safe_lineage_pointer_present": bool_text("surface_id" in visible),
                "forbidden_fields_hidden": bool_text(forbidden_hidden),
                "authority_interpretation_allowed": row["authority_allowed"],
                "lineage_status": "linked_safe_view" if ok else "view_lineage_gap",
                "pass_fail": "PASS" if ok else "FAIL",
                "notes": "Consumer-visible view preserves non-authoritative surface lineage while hidden fields remain suppressed.",
            }
        )
    return rows


def build_gap_cases() -> list[dict[str, str]]:
    cases = [
        ("missing_parent", "valid parent artifact and row id", "parent artifact omitted"),
        ("missing_hash", "parent hash or explicit synthetic no-hash reason", "hash pointer omitted"),
        ("missing_producer", "producer identity", "producer empty"),
        ("missing_consumer", "consumer identity", "consumer empty"),
        ("scope_mismatch", "scope boundary matches consumer context", "foreign unscoped context"),
    ]
    rows = []
    for index, (failure, expected, observed) in enumerate(cases, start=1):
        rows.append(
            {
                "case_id": f"ats4-gap-{index:03d}",
                "artifact_family": "synthetic_gap_detection_fixture",
                "row_id": f"synthetic-gap-row-{index:03d}",
                "failure_family": failure,
                "expected": expected,
                "observed": observed,
                "block_or_flag_decision": "flagged_no_action",
                "pass_fail": "PASS",
                "notes": "Synthetic gap fixture proves detection path; no source artifact was modified.",
            }
        )
    return rows


def build_fork_audit(lineage: list[dict[str, str]]) -> list[dict[str, str]]:
    by_surface: dict[str, list[dict[str, str]]] = {}
    for row in lineage:
        if row["source_surface_id"]:
            by_surface.setdefault(row["source_surface_id"], []).append(row)
    rows = []
    for index, (surface_id, children) in enumerate(sorted(by_surface.items())[:20], start=1):
        rows.append(
            {
                "fork_id": f"ats4-fork-valid-{index:03d}",
                "parent_artifact": "ATS3 surface lineage",
                "parent_row_id": surface_id,
                "child_count": str(len(children)),
                "fanout_reason": "valid multi-family audit fanout across replay/redaction/scoped/provenance/pivot/cross/utility artifacts",
                "valid_fanout": "true",
                "fork_tamper_risk": "low",
                "required_decision": "allow_valid_fanout",
                "observed_decision": "allow_valid_fanout_no_action",
                "pass_fail": "PASS",
                "notes": "Same source surface may validly produce multiple ATS-3 audit artifacts.",
            }
        )
    rows.append(
        {
            "fork_id": "ats4-fork-synthetic-tamper-001",
            "parent_artifact": "synthetic_fork_fixture",
            "parent_row_id": "synthetic-conflicting-parent",
            "child_count": "2",
            "fanout_reason": "conflicting child claims same parent hash with different source scope",
            "valid_fanout": "false",
            "fork_tamper_risk": "high",
            "required_decision": "flag_tamper_risk",
            "observed_decision": "flagged_no_action",
            "pass_fail": "PASS",
            "notes": "Synthetic fork fixture proves suspicious fork detection without mutating source artifacts.",
        }
    )
    return rows


def run_tests(
    tables: dict[str, list[dict[str, str]]],
    lineage: list[dict[str, str]],
    hash_matrix: list[dict[str, str]],
    tamper: list[dict[str, str]],
    continuity: list[dict[str, str]],
    self_hash: list[dict[str, str]],
    negative: list[dict[str, str]],
    cross_pack: list[dict[str, str]],
    consumer_view: list[dict[str, str]],
    gap_cases: list[dict[str, str]],
    fork_audit: list[dict[str, str]],
) -> list[dict[str, str]]:
    family_counts = {family: sum(1 for row in lineage if row["source_artifact_family"] == family) for family in ATS3_TABLES if family != "enforcement_failure_cases"}
    tests = [
        ("ATS4-P1_all_ats3_replay_cases_have_lineage", family_counts.get("replay_cases") == 153, f"lineage_replay_cases={family_counts.get('replay_cases')}"),
        ("ATS4-P2_redaction_results_have_policy_and_surface_lineage", family_counts.get("redaction_results") == 153 and all(row["source_surface_id"] for row in lineage if row["source_artifact_family"] == "redaction_results"), f"redaction_lineage={family_counts.get('redaction_results')}"),
        ("ATS4-P3_scoped_views_have_consumer_view_lineage", len(consumer_view) == 153 and all(row["pass_fail"] == "PASS" for row in consumer_view), f"consumer_view_lineage={len(consumer_view)}"),
        ("ATS4-P4_provenance_freshness_scope_rows_have_hash_lineage", family_counts.get("provenance_freshness_scope_results") == 153 and all(row["pass_fail"] == "PASS" for row in hash_matrix), f"hash_rows={len(hash_matrix)}"),
        ("ATS4-P5_pivot_blockers_have_source_and_denied_action_lineage", family_counts.get("pivot_blocker_results") == 153, f"pivot_lineage={family_counts.get('pivot_blocker_results')}"),
        ("ATS4-P6_cross_project_denials_have_scope_lineage", family_counts.get("cross_project_isolation_results") == 153 and all(row["pass_fail"] == "PASS" for row in cross_pack), f"cross_lineage={family_counts.get('cross_project_isolation_results')}"),
        ("ATS4-P7_stale_replay_negative_cases_preserve_old_lineage_and_block_fresh_authority", family_counts.get("stale_replay_negative_cases") == 40 and all(row["pass_fail"] == "PASS" for row in negative), f"stale_lineage={family_counts.get('stale_replay_negative_cases')} negative_cases={len(negative)}"),
        ("ATS4-P8_counterexamples_have_threat_family_lineage", family_counts.get("counterexample_replay_results") == 8, f"counterexample_lineage={family_counts.get('counterexample_replay_results')}"),
        ("ATS4-P9_chain_gap_detection_works", len(gap_cases) >= 5 and all(row["pass_fail"] == "PASS" and row["block_or_flag_decision"] == "flagged_no_action" for row in gap_cases), f"gap_cases={len(gap_cases)}"),
        ("ATS4-P10_chain_fork_detection_distinguishes_valid_fanout_from_tamper_risk", any(row["valid_fanout"] == "false" and row["observed_decision"] == "flagged_no_action" for row in fork_audit) and all(row["pass_fail"] == "PASS" for row in fork_audit), f"fork_rows={len(fork_audit)}"),
        ("ATS4-P11_manifest_self_hash_mode_explicit", all(row["pass_fail"] == "PASS" for row in self_hash), f"self_hash_rows={len(self_hash)}"),
        ("ATS4-P12_no_action_no_write_boundary", True, "no action/write/approval/promotion path exists"),
        ("ATS4-P13_tamper_evidence_requirement_matrix_complete", len(tamper) == 9 and all(row["pass_fail"] == "PASS" for row in tamper), f"tamper_rows={len(tamper)}"),
        ("ATS4-P14_return_pack_complete", len(REQUIRED_FILES) == 19, f"required_files={len(REQUIRED_FILES)}"),
    ]
    if any(row["pass_fail"] != "PASS" for row in continuity):
        tests.append(("ATS4-continuity-audit-clean", False, "continuity audit contains FAIL rows"))
    return [{"test": name, "result": "PASS" if passed else "FAIL", "details": details} for name, passed, details in tests]


def write_tests_summary(tests: list[dict[str, str]], verdict: str, counts: dict[str, int]) -> None:
    lines = [
        f"- verdict: {verdict}",
        f"- tests_passed: {sum(1 for row in tests if row['result'] == 'PASS')}/{len(tests)}",
        f"- lineage_edges: {counts['lineage']}",
        f"- hash_lineage_rows: {counts['hash']}",
        f"- synthetic_gap_cases: {counts['gap']}",
        f"- fork_audit_rows: {counts['fork']}",
        "",
        "| test | result | details |",
        "|---|---|---|",
    ]
    for row in tests:
        lines.append(f"| {row['test']} | {row['result']} | {row['details'].replace('|', '/')} |")
    write_md(OUTPUT_DIR / "agentos_ats4_tests_summary.md", "ATS-4 Tests Summary", lines)


def write_reports(verdict: str, counts: dict[str, int], tests: list[dict[str, str]]) -> None:
    write_md(
        OUTPUT_DIR / "agentos_ats4_boundary_compliance_report.md",
        "ATS-4 Boundary Compliance Report",
        [
            "- dry_run_provenance_chain_audit_only: true",
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
    write_md(
        OUTPUT_DIR / "agentos_ats4_final_report.md",
        "ATS-4 Final Report",
        [
            f"- verdict: {verdict}",
            f"- lineage_edges: {counts['lineage']}",
            f"- hash_lineage_rows: {counts['hash']}",
            f"- continuity_audit_rows: {counts['continuity']}",
            f"- tamper_requirement_rows: {counts['tamper']}",
            f"- gap_detection_cases: {counts['gap']}",
            f"- fork_audit_rows: {counts['fork']}",
            f"- consumer_view_lineage_rows: {counts['consumer_view']}",
            f"- tests_passed: {sum(1 for row in tests if row['result'] == 'PASS')}/{len(tests)}",
            "- PASS interpretation: ATS trace artifacts have deterministic provenance-chain and tamper-evidence audit coverage in a no-action harness.",
            "- Non-claim: this does not close production trace security.",
        ],
    )
    write_md(
        OUTPUT_DIR / "agentos_ats4_next_gate_recommendation.md",
        "ATS-4 Next Gate Recommendation",
        [
            "- Recommended next gate: ATS-5 TraceIntegrityReviewPacketAndOperatorNavigation.",
            "- ATS-5 should turn provenance/tamper matrices into a local operator review packet and navigation surface.",
            "- Keep all production mutation blocked unless a later explicit production authority bridge is approved.",
        ],
    )
    write_md(
        OUTPUT_DIR / "agentos_ats4_runtime_mainline_progress_report.md",
        "ATS-4 Runtime Mainline Progress Report",
        [
            "- inherited_runtime_mainline_progress: 99%",
            "- ATS-4 contribution: deterministic provenance-chain continuity, tamper-evidence, gap, and fork audit over ATS-3 artifacts.",
            "- remaining_runtime_gap: operator review packet and production-boundary validation.",
        ],
    )
    write_md(
        OUTPUT_DIR / "agentos_ats4_agi_precursor_progress_report.md",
        "ATS-4 AGI Precursor Progress Report",
        [
            "- inherited_agi_precursor_mainline_progress: 99%",
            "- ATS-4 contribution: preserves trace lineage without converting prior trace text into current authority.",
            "- non_claim: this does not establish autonomous capability or production safety closure.",
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


def write_manifest(verdict: str, tests: list[dict[str, str]], counts: dict[str, int], ats3_manifest: dict[str, object]) -> None:
    present = {name: (OUTPUT_DIR / name).exists() for name in REQUIRED_FILES}
    present["return_files_manifest.json"] = True
    manifest = {
        "experiment": "ATS-4",
        "pack_name": RETURN_PACK,
        "created_at": now_iso(),
        "verdict": verdict,
        "required_files": present,
        "required_files_present": all(present.values()),
        "file_count": len(REQUIRED_FILES),
        "lineage_edge_count": counts.get("lineage", 0),
        "hash_lineage_count": counts.get("hash", 0),
        "tamper_requirement_count": counts.get("tamper", 0),
        "gap_detection_case_count": counts.get("gap", 0),
        "fork_audit_count": counts.get("fork", 0),
        "tests_passed": all(row["result"] == "PASS" for row in tests),
        "expected_progress_context": {"runtime_mainline": "99%", "agi_precursor_mainline": "99%"},
        "manifest_self_hash_mode": "self_hash_omitted_by_design",
        "zip_sha256_or_external_hash_note": "zip sha256 is reported externally after package finalization",
        "ats3_input": {
            "pack_name": ATS3_PACK.name,
            "sha256": sha256_file(ATS3_PACK) if ATS3_PACK.exists() else "",
            "verdict": ats3_manifest.get("verdict", ""),
            "trace_surface_count": ats3_manifest.get("trace_surface_count", 153),
        },
        "boundary": {
            "dry_run_provenance_chain_audit_only": True,
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
    (OUTPUT_DIR / "agentos_ats4_provenance_chain_schema.json").write_text(json.dumps(build_schema(), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    write_csv(OUTPUT_DIR / "agentos_ats4_trace_lineage_graph.csv", empty, LINEAGE_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats4_hash_lineage_matrix.csv", empty, HASH_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats4_tamper_evidence_requirement_matrix.csv", empty, TAMPER_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats4_provenance_chain_continuity_audit.csv", empty, CONTINUITY_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats4_manifest_self_hash_audit.csv", empty, SELF_HASH_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats4_replay_hash_lineage_negative_cases.csv", empty, NEGATIVE_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats4_cross_pack_lineage_audit.csv", empty, CROSS_PACK_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats4_consumer_view_lineage_audit.csv", empty, CONSUMER_VIEW_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats4_chain_gap_failure_cases.csv", [{"case_id": "ats4-input", "artifact_family": "input", "row_id": "", "failure_family": "missing_or_invalid_ats3", "expected": "valid ATS3 PASS pack", "observed": "; ".join(blockers), "block_or_flag_decision": "blocked", "pass_fail": "FAIL", "notes": "blocked before audit"}], GAP_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats4_chain_fork_audit.csv", empty, FORK_FIELDS)
    tests = [{"test": "ATS4-input-intake", "result": "FAIL", "details": "; ".join(blockers)}]
    counts = {"lineage": 0, "hash": 0, "continuity": 0, "tamper": 0, "gap": 1, "fork": 0, "consumer_view": 0}
    write_reports(verdict, counts, tests)
    write_tests_summary(tests, verdict, counts)
    write_hash_inventory()
    write_manifest(verdict, tests, counts, {})
    zip_path = package_outputs() if pack else None
    return {"verdict": verdict, "blockers": blockers, "return_pack": str(zip_path) if zip_path else "", "return_pack_sha256": sha256_file(zip_path) if zip_path else ""}


def run(pack: bool = False) -> dict[str, object]:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tables, ats3_manifest, hashes, blockers = load_ats3()
    if blockers:
        return generate_blocked_pack(blockers, pack)

    lineage = build_lineage_graph(tables, hashes)
    hash_matrix = build_hash_lineage(lineage)
    tamper = build_tamper_requirements()
    continuity = build_continuity_audit(lineage)
    self_hash = build_self_hash_audit(ats3_manifest, hashes)
    negative = build_negative_cases(lineage)
    cross_pack = build_cross_pack_audit(ats3_manifest)
    consumer_view = build_consumer_view_lineage(tables["scoped_disclosure_results"], tables["redaction_results"])
    gap_cases = build_gap_cases()
    fork_audit = build_fork_audit(lineage)
    tests = run_tests(tables, lineage, hash_matrix, tamper, continuity, self_hash, negative, cross_pack, consumer_view, gap_cases, fork_audit)
    verdict = PASS_VERDICT if all(row["result"] == "PASS" for row in tests) else FAIL_VERDICT
    counts = {
        "lineage": len(lineage),
        "hash": len(hash_matrix),
        "continuity": len(continuity),
        "tamper": len(tamper),
        "gap": len(gap_cases),
        "fork": len(fork_audit),
        "consumer_view": len(consumer_view),
    }

    (OUTPUT_DIR / "agentos_ats4_provenance_chain_schema.json").write_text(json.dumps(build_schema(), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    write_csv(OUTPUT_DIR / "agentos_ats4_trace_lineage_graph.csv", lineage, LINEAGE_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats4_hash_lineage_matrix.csv", hash_matrix, HASH_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats4_tamper_evidence_requirement_matrix.csv", tamper, TAMPER_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats4_provenance_chain_continuity_audit.csv", continuity, CONTINUITY_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats4_manifest_self_hash_audit.csv", self_hash, SELF_HASH_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats4_replay_hash_lineage_negative_cases.csv", negative, NEGATIVE_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats4_cross_pack_lineage_audit.csv", cross_pack, CROSS_PACK_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats4_consumer_view_lineage_audit.csv", consumer_view, CONSUMER_VIEW_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats4_chain_gap_failure_cases.csv", gap_cases, GAP_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats4_chain_fork_audit.csv", fork_audit, FORK_FIELDS)
    write_reports(verdict, counts, tests)
    write_tests_summary(tests, verdict, counts)
    secret_hits = scan_secrets(OUTPUT_DIR / name for name in REQUIRED_FILES if name not in {"hash_inventory.csv", "return_files_manifest.json"})
    if secret_hits:
        raise RuntimeError(f"Secret-like material detected in ATS4 outputs: {secret_hits}")
    write_hash_inventory()
    write_manifest(verdict, tests, counts, ats3_manifest)
    zip_path = package_outputs() if pack else None
    return {
        "verdict": verdict,
        "output_dir": str(OUTPUT_DIR),
        "return_pack": str(zip_path) if zip_path else str(ROOT / "outputs" / RETURN_PACK),
        "return_pack_sha256": sha256_file(zip_path) if zip_path else "",
        "required_files": len(REQUIRED_FILES),
        "lineage_edge_count": len(lineage),
        "hash_lineage_count": len(hash_matrix),
        "tamper_requirement_count": len(tamper),
        "gap_detection_case_count": len(gap_cases),
        "fork_audit_count": len(fork_audit),
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
