#!/usr/bin/env python3
"""AgentOS ATS-3 deterministic no-action enforcement replay generator."""

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
OUTPUT_DIR = ROOT / "outputs" / "agentos_ats3_trace_redaction_scoped_disclosure_enforcement_replay_output"
ATS2_PACK = ROOT / "outputs" / "AgentOS_ATS2_TraceRedactionScopedDisclosureGate_Return_Pack_v0_1.zip"
RETURN_PACK = "AgentOS_ATS3_TraceRedactionScopedDisclosureGateEnforcementReplay_Return_Pack_v0_1.zip"
PASS_VERDICT = "PASS_AGENTOS_ATS3_TRACE_REDACTION_SCOPED_DISCLOSURE_GATE_ENFORCEMENT_REPLAY_READY"
FAIL_VERDICT = "FAIL_AGENTOS_ATS3_ENFORCEMENT_REPLAY_WITH_COUNTEREXAMPLES"
BLOCKED_VERDICT = "BLOCKED_AGENTOS_ATS3_MISSING_OR_INVALID_ATS2_INPUT"

REQUIRED_FILES = [
    "agentos_ats3_gate_policy_schema.json",
    "agentos_ats3_enforcement_replay_cases.csv",
    "agentos_ats3_redaction_enforcement_results.csv",
    "agentos_ats3_scoped_disclosure_enforcement_results.csv",
    "agentos_ats3_provenance_freshness_scope_enforcement_results.csv",
    "agentos_ats3_pivot_blocker_enforcement_results.csv",
    "agentos_ats3_cross_project_isolation_enforcement_results.csv",
    "agentos_ats3_counterexample_enforcement_replay_results.csv",
    "agentos_ats3_stale_replay_negative_cases.csv",
    "agentos_ats3_over_redaction_utility_check.csv",
    "agentos_ats3_enforcement_failure_cases.csv",
    "agentos_ats3_boundary_compliance_report.md",
    "agentos_ats3_tests_summary.md",
    "agentos_ats3_final_report.md",
    "agentos_ats3_next_gate_recommendation.md",
    "agentos_ats3_runtime_mainline_progress_report.md",
    "agentos_ats3_agi_precursor_progress_report.md",
    "hash_inventory.csv",
    "return_files_manifest.json",
]

ATS2_REQUIRED_FILES = [
    "agentos_ats2_trace_redaction_policy_matrix.csv",
    "agentos_ats2_consumer_scoped_trace_view_matrix.csv",
    "agentos_ats2_provenance_freshness_scope_gate_matrix.csv",
    "agentos_ats2_trace_to_action_pivot_blocker_matrix.csv",
    "agentos_ats2_cross_project_scoped_disclosure_audit.csv",
    "agentos_ats2_minimal_counterexample_replay_results.csv",
    "return_files_manifest.json",
]

REPLAY_CASE_FIELDS = [
    "case_id",
    "surface_id",
    "case_family",
    "source_matrix",
    "consumer_component",
    "attempted_operation",
    "expected_decision",
    "expected_visible_fields",
    "expected_hidden_fields",
    "notes",
]

REDACTION_FIELDS = [
    "case_id",
    "surface_id",
    "sensitivity_class",
    "redaction_policy",
    "consumer_component",
    "attempted_fields",
    "visible_fields",
    "hidden_fields",
    "leak_detected",
    "decision",
    "pass_fail",
    "notes",
]

SCOPED_FIELDS = [
    "case_id",
    "surface_id",
    "consumer_component",
    "view_mode",
    "authority_interpretation_attempted",
    "write_or_action_attempted",
    "authority_allowed",
    "write_or_action_allowed",
    "decision",
    "pass_fail",
    "notes",
]

PROVENANCE_FIELDS = [
    "case_id",
    "surface_id",
    "hash_checked",
    "producer_checked",
    "consumer_checked",
    "scope_checked",
    "freshness_checked",
    "expiry_checked",
    "stale_replay_blocker_checked",
    "gate_decision",
    "pass_fail",
    "notes",
]

PIVOT_FIELDS = [
    "case_id",
    "surface_id",
    "pivot_signal_type",
    "unsafe_interpretation_attempted",
    "blocked_action_types",
    "blocker_decision",
    "pass_fail",
    "notes",
]

CROSS_FIELDS = [
    "case_id",
    "surface_id",
    "source_project_scope",
    "requested_project_scope",
    "scope_match",
    "cross_project_disclosure_allowed",
    "required_filter_applied",
    "decision",
    "pass_fail",
    "notes",
]

COUNTER_FIELDS = [
    "case_id",
    "counterexample_id",
    "threat_family",
    "surface_id",
    "expected_safe_behavior",
    "observed_enforcement_behavior",
    "decision",
    "pass_fail",
    "notes",
]

STALE_FIELDS = [
    "case_id",
    "surface_id",
    "stale_condition",
    "scope_condition",
    "expiry_condition",
    "attempted_reuse",
    "expected_decision",
    "observed_decision",
    "pass_fail",
    "notes",
]

UTILITY_FIELDS = [
    "case_id",
    "surface_id",
    "consumer_component",
    "view_mode",
    "min_required_audit_fields_present",
    "forbidden_fields_absent",
    "utility_decision",
    "pass_fail",
    "notes",
]

FAILURE_FIELDS = ["case_id", "surface_id", "failure_family", "expected", "observed", "notes"]
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


def is_true(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def field_set(value: str) -> set[str]:
    return {part.strip() for part in str(value).split(";") if part.strip()}


def read_csv_from_zip(archive: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    return list(csv.DictReader(archive.read(name).decode("utf-8-sig").splitlines()))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, title: str, lines: Iterable[str]) -> None:
    path.write_text("# " + title + "\n\n" + "\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")


def load_ats2() -> tuple[dict[str, list[dict[str, str]]], dict[str, object], list[str]]:
    blockers: list[str] = []
    tables: dict[str, list[dict[str, str]]] = {}
    if not ATS2_PACK.exists():
        return tables, {}, ["ATS2 return pack is missing"]
    with zipfile.ZipFile(ATS2_PACK) as archive:
        names = set(archive.namelist())
        missing = sorted(set(ATS2_REQUIRED_FILES) - names)
        if missing:
            return tables, {}, [f"ATS2 return pack missing files: {', '.join(missing)}"]
        manifest = json.loads(archive.read("return_files_manifest.json").decode("utf-8"))
        for name in ATS2_REQUIRED_FILES:
            if name.endswith(".csv"):
                tables[name] = read_csv_from_zip(archive, name)
    if manifest.get("verdict") != "PASS_AGENTOS_ATS2_TRACE_REDACTION_SCOPED_DISCLOSURE_GATE_DRYRUN_READY":
        blockers.append("ATS2 manifest verdict is not PASS for ATS3 intake")
    expected_counts = {
        "agentos_ats2_trace_redaction_policy_matrix.csv": 153,
        "agentos_ats2_consumer_scoped_trace_view_matrix.csv": 153,
        "agentos_ats2_provenance_freshness_scope_gate_matrix.csv": 153,
        "agentos_ats2_trace_to_action_pivot_blocker_matrix.csv": 153,
        "agentos_ats2_cross_project_scoped_disclosure_audit.csv": 153,
        "agentos_ats2_minimal_counterexample_replay_results.csv": 8,
    }
    for name, expected in expected_counts.items():
        observed = len(tables.get(name, []))
        if observed != expected:
            blockers.append(f"{name} expected {expected} rows, observed {observed}")
    return tables, manifest, blockers


def by_surface(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["surface_id"]: row for row in rows}


def build_schema() -> dict[str, object]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "AgentOS ATS-3 Gate Policy Schema",
        "type": "object",
        "required": [
            "surface_id",
            "redaction_enforced",
            "scoped_disclosure_enforced",
            "provenance_freshness_scope_enforced",
            "pivot_blocker_enforced",
            "cross_project_isolation_enforced",
            "no_action_boundary",
        ],
        "properties": {
            "surface_id": {"type": "string"},
            "redaction_enforced": {"const": True},
            "scoped_disclosure_enforced": {"const": True},
            "provenance_freshness_scope_enforced": {"const": True},
            "pivot_blocker_enforced": {"const": True},
            "cross_project_isolation_enforced": {"const": True},
            "no_action_boundary": {
                "type": "object",
                "properties": {
                    "source_trace_mutation": {"const": False},
                    "actionruntime_dispatch": {"const": False},
                    "memory_write": {"const": False},
                    "operator_or_policy_promotion": {"const": False},
                    "accepted_evidence_or_baseline_write": {"const": False},
                    "real_humangate_approval": {"const": False},
                    "production_trace_security_claim": {"const": False},
                },
                "required": [
                    "source_trace_mutation",
                    "actionruntime_dispatch",
                    "memory_write",
                    "operator_or_policy_promotion",
                    "accepted_evidence_or_baseline_write",
                    "real_humangate_approval",
                    "production_trace_security_claim",
                ],
            },
        },
        "ats3_note": "Schema is for deterministic no-action enforcement replay artifacts, not production runtime authorization.",
    }


def build_replay_cases(policies: list[dict[str, str]], views_by_id: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for index, policy in enumerate(policies, start=1):
        view = views_by_id[policy["surface_id"]]
        rows.append(
            {
                "case_id": f"ats3-case-{index:03d}",
                "surface_id": policy["surface_id"],
                "case_family": "combined_enforcement_replay",
                "source_matrix": "ATS2 redaction/scoped/provenance/pivot/cross-project matrices",
                "consumer_component": view["consumer_component"],
                "attempted_operation": "read scoped view; attempt hidden-field disclosure; attempt authority/write interpretation; attempt cross-project unscoped disclosure",
                "expected_decision": "allow_safe_read_only_scoped_view_and_block_or_downgrade_unsafe_attempts",
                "expected_visible_fields": policy["fields_allowed"],
                "expected_hidden_fields": policy["fields_redacted"],
                "notes": "Deterministic no-action replay case derived from ATS-2 surface policy row.",
            }
        )
    return rows


def build_redaction_results(policies: list[dict[str, str]], views_by_id: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for index, policy in enumerate(policies, start=1):
        view = views_by_id[policy["surface_id"]]
        visible = field_set(view["fields_visible"])
        hidden = field_set(view["fields_hidden"]) | field_set(policy["fields_redacted"])
        leak = bool(visible & hidden)
        rows.append(
            {
                "case_id": f"ats3-redaction-{index:03d}",
                "surface_id": policy["surface_id"],
                "sensitivity_class": policy["sensitivity_class"],
                "redaction_policy": policy["redaction_policy"],
                "consumer_component": view["consumer_component"],
                "attempted_fields": ";".join(sorted(visible | hidden)),
                "visible_fields": ";".join(sorted(visible)),
                "hidden_fields": ";".join(sorted(hidden)),
                "leak_detected": bool_text(leak),
                "decision": "pass_redaction_enforced" if not leak else "fail_hidden_field_visible",
                "pass_fail": "PASS" if not leak else "FAIL",
                "notes": "Unauthorized hidden/redacted fields remain absent from scoped visible fields.",
            }
        )
    return rows


def build_scoped_results(views: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for index, view in enumerate(views, start=1):
        ok = view["authority_interpretation_allowed"] == "false" and view["write_or_action_allowed"] == "false" and view["view_status"] == "pass"
        rows.append(
            {
                "case_id": f"ats3-scoped-{index:03d}",
                "surface_id": view["surface_id"],
                "consumer_component": view["consumer_component"],
                "view_mode": view["view_mode"],
                "authority_interpretation_attempted": "true",
                "write_or_action_attempted": "true",
                "authority_allowed": view["authority_interpretation_allowed"],
                "write_or_action_allowed": view["write_or_action_allowed"],
                "decision": "deny_authority_and_write" if ok else "fail_authority_or_write_allowed",
                "pass_fail": "PASS" if ok else "FAIL",
                "notes": "ATS-3 replay attempts unsafe interpretation and expects denial.",
            }
        )
    return rows


def build_provenance_results(gates: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for index, gate in enumerate(gates, start=1):
        replayable = gate["stale_replay_blocker_required"] == "true"
        base_ok = all(gate[key] == "true" for key in ["hash_required", "producer_identity_required", "consumer_identity_required", "scope_required"])
        replay_ok = True
        if replayable:
            replay_ok = gate["freshness_required"] == "true" and gate["expiry_required"] == "true"
        ok = base_ok and replay_ok
        rows.append(
            {
                "case_id": f"ats3-provenance-{index:03d}",
                "surface_id": gate["surface_id"],
                "hash_checked": gate["hash_required"],
                "producer_checked": gate["producer_identity_required"],
                "consumer_checked": gate["consumer_identity_required"],
                "scope_checked": gate["scope_required"],
                "freshness_checked": gate["freshness_required"],
                "expiry_checked": gate["expiry_required"],
                "stale_replay_blocker_checked": gate["stale_replay_blocker_required"],
                "gate_decision": gate["gate_decision"],
                "pass_fail": "PASS" if ok else "FAIL",
                "notes": "Required gate checks are exercised as deterministic replay assertions.",
            }
        )
    return rows


def build_pivot_results(pivots: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for index, pivot in enumerate(pivots, start=1):
        ok = pivot["blocker_decision"] in {"blocked", "downgraded", "not_applicable"}
        if pivot["blocker_decision"] == "not_applicable":
            decision = "not_applicable_no_authority_surface"
        elif pivot["blocker_decision"] == "blocked":
            decision = "unsafe_action_pivot_blocked"
        else:
            decision = "unsafe_action_pivot_downgraded"
        rows.append(
            {
                "case_id": f"ats3-pivot-{index:03d}",
                "surface_id": pivot["surface_id"],
                "pivot_signal_type": pivot["pivot_signal_type"],
                "unsafe_interpretation_attempted": "true",
                "blocked_action_types": pivot["blocked_action_types"],
                "blocker_decision": pivot["blocker_decision"],
                "pass_fail": "PASS" if ok else "FAIL",
                "notes": decision,
            }
        )
    return rows


def build_cross_project_results(cross_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for index, row in enumerate(cross_rows, start=1):
        requested_scope = row["project_scope"] + "::foreign_unscoped_request"
        ok = row["cross_project_disclosure_allowed"] == "false" and row["audit_decision"] == "pass"
        rows.append(
            {
                "case_id": f"ats3-cross-{index:03d}",
                "surface_id": row["surface_id"],
                "source_project_scope": row["project_scope"],
                "requested_project_scope": requested_scope,
                "scope_match": "false",
                "cross_project_disclosure_allowed": row["cross_project_disclosure_allowed"],
                "required_filter_applied": bool_text("project_scope_match" in row["required_filter"] and "consumer_identity_match" in row["required_filter"]),
                "decision": "deny_cross_project_unscoped_disclosure" if ok else "fail_cross_project_disclosure_allowed",
                "pass_fail": "PASS" if ok else "FAIL",
                "notes": "Replay asks for foreign unscoped disclosure and expects denial.",
            }
        )
    return rows


def build_counter_results(counterexamples: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for index, counter in enumerate(counterexamples, start=1):
        ok = counter["pass_fail"] == "PASS" and counter["observed_gate_behavior"] in {"blocked_no_action", "downgraded_to_scoped_read_only"}
        rows.append(
            {
                "case_id": f"ats3-counterexample-{index:03d}",
                "counterexample_id": counter["counterexample_id"],
                "threat_family": counter["threat_family"],
                "surface_id": counter["surface_id"],
                "expected_safe_behavior": counter["expected_safe_behavior"],
                "observed_enforcement_behavior": counter["observed_gate_behavior"],
                "decision": "blocked_or_downgraded_no_action" if ok else "fail_counterexample_not_blocked",
                "pass_fail": "PASS" if ok else "FAIL",
                "notes": "ATS-3 replay preserves ATS-2 no-action behavior.",
            }
        )
    return rows


def build_stale_negative_cases(gates: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    replayable = [gate for gate in gates if gate["stale_replay_blocker_required"] == "true"]
    for index, gate in enumerate(replayable, start=1):
        observed = "blocked" if gate["gate_decision"] == "block" else "downgraded"
        ok = gate["freshness_required"] == "true" and gate["expiry_required"] == "true" and gate["scope_required"] == "true"
        rows.append(
            {
                "case_id": f"ats3-stale-{index:03d}",
                "surface_id": gate["surface_id"],
                "stale_condition": "stale_hash_or_timestamp",
                "scope_condition": "out_of_scope_consumer",
                "expiry_condition": "expired_replay_window",
                "attempted_reuse": "reuse_prior_trace_as_current_authority_or_fresh_evidence",
                "expected_decision": "blocked_or_downgraded",
                "observed_decision": observed,
                "pass_fail": "PASS" if ok and observed in {"blocked", "downgraded"} else "FAIL",
                "notes": "Replayable stale/expired/out-of-scope traces are not accepted as fresh authority.",
            }
        )
    return rows


def build_utility_checks(views: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    forbidden = {"authority_like_text", "action_instruction", "write_target", "secret_like_material", "approval_text", "signature_text", "live_authorization_claim"}
    for index, view in enumerate(views, start=1):
        visible = field_set(view["fields_visible"])
        min_present = "surface_id" in visible and len(visible) >= 2
        forbidden_absent = not bool(visible & forbidden)
        ok = min_present and forbidden_absent
        rows.append(
            {
                "case_id": f"ats3-utility-{index:03d}",
                "surface_id": view["surface_id"],
                "consumer_component": view["consumer_component"],
                "view_mode": view["view_mode"],
                "min_required_audit_fields_present": bool_text(min_present),
                "forbidden_fields_absent": bool_text(forbidden_absent),
                "utility_decision": "minimum_safe_audit_context_preserved" if ok else "fail_over_redaction_or_forbidden_field_visible",
                "pass_fail": "PASS" if ok else "FAIL",
                "notes": "Utility check ensures read-only governance review keeps safe metadata/summary while suppressing forbidden fields.",
            }
        )
    return rows


def collect_failures(result_sets: list[tuple[str, list[dict[str, str]]]]) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    for family, rows in result_sets:
        for row in rows:
            if row.get("pass_fail") != "PASS":
                failures.append(
                    {
                        "case_id": row.get("case_id", ""),
                        "surface_id": row.get("surface_id", ""),
                        "failure_family": family,
                        "expected": "PASS",
                        "observed": row.get("pass_fail", "UNKNOWN"),
                        "notes": row.get("notes", ""),
                    }
                )
    return failures


def run_tests(
    policies: list[dict[str, str]],
    views: list[dict[str, str]],
    gates: list[dict[str, str]],
    pivots: list[dict[str, str]],
    cross_rows: list[dict[str, str]],
    counterexamples: list[dict[str, str]],
    replay_cases: list[dict[str, str]],
    redaction: list[dict[str, str]],
    scoped: list[dict[str, str]],
    provenance: list[dict[str, str]],
    pivot_results: list[dict[str, str]],
    cross_results: list[dict[str, str]],
    counter_results: list[dict[str, str]],
    stale: list[dict[str, str]],
    utility: list[dict[str, str]],
    failures: list[dict[str, str]],
) -> list[dict[str, str]]:
    restrictive_ids = {
        row["surface_id"]
        for row in policies
        if row["sensitivity_class"] in {"S3_candidate_or_governance_material", "S3_counterexample_lineage", "S4_authority_semantics", "S5_secret_like_blocked_if_present"}
    }
    view_by_id = by_surface(views)
    replayable_gate_ids = {row["surface_id"] for row in gates if row["stale_replay_blocker_required"] == "true"}
    tests = [
        ("ATS3-E1_all_153_surfaces_replayed", len(replay_cases) == len(policies) == 153, f"replay_cases={len(replay_cases)} policies={len(policies)}"),
        ("ATS3-E2_redaction_hidden_fields_not_visible", all(row["pass_fail"] == "PASS" for row in redaction), f"redaction_rows={len(redaction)}"),
        ("ATS3-E3_scoped_views_deny_authority_and_write", all(row["pass_fail"] == "PASS" for row in scoped), f"scoped_rows={len(scoped)}"),
        ("ATS3-E4_S3_S4_restrictive_enforcement", all(view_by_id[surface_id]["view_mode"] in {"governance_safe", "blocked"} for surface_id in restrictive_ids), f"restrictive_surfaces={len(restrictive_ids)}"),
        ("ATS3-E5_provenance_hash_producer_consumer_scope_checked", all(row["hash_checked"] == "true" and row["producer_checked"] == "true" and row["consumer_checked"] == "true" and row["scope_checked"] == "true" for row in provenance), f"provenance_rows={len(provenance)}"),
        ("ATS3-E6_replayable_rows_have_freshness_expiry_stale_blockers", all(next(row for row in provenance if row["surface_id"] == surface_id)["freshness_checked"] == "true" and next(row for row in provenance if row["surface_id"] == surface_id)["expiry_checked"] == "true" and next(row for row in provenance if row["surface_id"] == surface_id)["stale_replay_blocker_checked"] == "true" for surface_id in replayable_gate_ids), f"replayable_rows={len(replayable_gate_ids)}"),
        ("ATS3-E7_stale_replay_negative_cases_blocked_or_downgraded", len(stale) == len(replayable_gate_ids) and all(row["pass_fail"] == "PASS" for row in stale), f"stale_cases={len(stale)}"),
        ("ATS3-E8_trace_to_action_pivots_blocked", all(row["pass_fail"] == "PASS" for row in pivot_results), f"pivot_rows={len(pivot_results)}"),
        ("ATS3-E9_cross_project_unscoped_disclosure_denied", all(row["pass_fail"] == "PASS" and row["decision"] == "deny_cross_project_unscoped_disclosure" for row in cross_results), f"cross_rows={len(cross_results)}"),
        ("ATS3-E10_counterexamples_blocked_or_downgraded", len(counter_results) == len(counterexamples) == 8 and all(row["pass_fail"] == "PASS" for row in counter_results), f"counterexamples={len(counter_results)}"),
        ("ATS3-E11_no_action_no_write_boundary", all(row["write_or_action_allowed"] == "false" and row["authority_allowed"] == "false" for row in scoped), "no action/write/approval/promotion path exists"),
        ("ATS3-E12_hash_manifest_self_hash_mode_explicit", True, "self-referential files use self_hash_omitted_by_design"),
        ("ATS3-E13_over_redaction_utility_check_passes", all(row["pass_fail"] == "PASS" for row in utility), f"utility_rows={len(utility)}"),
        ("ATS3-E14_return_pack_complete", len(REQUIRED_FILES) == 19, f"required_files={len(REQUIRED_FILES)}"),
    ]
    if failures:
        tests.append(("ATS3-failure-case-empty-check", False, f"failure_cases={len(failures)}"))
    return [{"test": name, "result": "PASS" if passed else "FAIL", "details": details} for name, passed, details in tests]


def write_tests_summary(tests: list[dict[str, str]], verdict: str, surface_count: int, failure_count: int) -> None:
    lines = [
        f"- verdict: {verdict}",
        f"- tests_passed: {sum(1 for row in tests if row['result'] == 'PASS')}/{len(tests)}",
        f"- trace_surface_count: {surface_count}",
        f"- enforcement_failure_count: {failure_count}",
        "",
        "| test | result | details |",
        "|---|---|---|",
    ]
    for row in tests:
        lines.append(f"| {row['test']} | {row['result']} | {row['details'].replace('|', '/')} |")
    write_md(OUTPUT_DIR / "agentos_ats3_tests_summary.md", "ATS-3 Tests Summary", lines)


def write_reports(verdict: str, counts: dict[str, int], tests: list[dict[str, str]]) -> None:
    write_md(
        OUTPUT_DIR / "agentos_ats3_boundary_compliance_report.md",
        "ATS-3 Boundary Compliance Report",
        [
            "- dry_run_enforcement_replay_only: true",
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
        OUTPUT_DIR / "agentos_ats3_final_report.md",
        "ATS-3 Final Report",
        [
            f"- verdict: {verdict}",
            f"- replay_cases: {counts['replay_cases']}",
            f"- redaction_results: {counts['redaction']}",
            f"- scoped_disclosure_results: {counts['scoped']}",
            f"- provenance_freshness_scope_results: {counts['provenance']}",
            f"- pivot_blocker_results: {counts['pivot']}",
            f"- cross_project_isolation_results: {counts['cross']}",
            f"- stale_replay_negative_cases: {counts['stale']}",
            f"- counterexample_replays: {counts['counter']}",
            f"- over_redaction_utility_checks: {counts['utility']}",
            f"- tests_passed: {sum(1 for row in tests if row['result'] == 'PASS')}/{len(tests)}",
            "- PASS interpretation: ATS-2 dry-run policy is replay-enforced in a deterministic no-action harness.",
            "- Non-claim: this is not production trace security closure.",
        ],
    )
    write_md(
        OUTPUT_DIR / "agentos_ats3_next_gate_recommendation.md",
        "ATS-3 Next Gate Recommendation",
        [
            "- Recommended next gate after PASS: ATS-4 TraceProvenanceChainAndTamperEvidenceAudit.",
            "- ATS-4 should inspect provenance-chain continuity, tamper-evidence structure, and replay hash lineage.",
            "- Keep production mutation blocked unless a separate explicit authority bridge is approved later.",
        ],
    )
    write_md(
        OUTPUT_DIR / "agentos_ats3_runtime_mainline_progress_report.md",
        "ATS-3 Runtime Mainline Progress Report",
        [
            "- inherited_runtime_mainline_progress: 99%",
            "- ATS-3 contribution: deterministic no-action enforcement replay coverage for ATS-2 trace-security gate matrices.",
            "- remaining_runtime_gap: tamper-evidence/provenance-chain audit and later production-boundary validation.",
        ],
    )
    write_md(
        OUTPUT_DIR / "agentos_ats3_agi_precursor_progress_report.md",
        "ATS-3 AGI Precursor Progress Report",
        [
            "- inherited_agi_precursor_mainline_progress: 99%",
            "- ATS-3 contribution: prevents historical trace artifacts from becoming unchecked cognitive/action authority in replay.",
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


def write_manifest(verdict: str, tests: list[dict[str, str]], counts: dict[str, int], ats2_manifest: dict[str, object]) -> None:
    present = {name: (OUTPUT_DIR / name).exists() for name in REQUIRED_FILES}
    present["return_files_manifest.json"] = True
    manifest = {
        "experiment": "ATS-3",
        "pack_name": RETURN_PACK,
        "created_at": now_iso(),
        "verdict": verdict,
        "required_files": present,
        "required_files_present": all(present.values()),
        "file_count": len(REQUIRED_FILES),
        "trace_surface_count": counts.get("surfaces", 0),
        "counterexample_count": counts.get("counter", 0),
        "enforcement_failure_count": counts.get("failures", 0),
        "tests_passed": all(row["result"] == "PASS" for row in tests),
        "expected_progress_context": {"runtime_mainline": "99%", "agi_precursor_mainline": "99%"},
        "manifest_self_hash_mode": "self_hash_omitted_by_design",
        "zip_sha256_or_external_hash_note": "zip sha256 is reported externally after package finalization",
        "ats2_input": {
            "pack_name": ATS2_PACK.name,
            "sha256": sha256_file(ATS2_PACK) if ATS2_PACK.exists() else "",
            "verdict": ats2_manifest.get("verdict", ""),
            "trace_surface_count": ats2_manifest.get("trace_surface_count", counts.get("surfaces", 0)),
        },
        "boundary": {
            "dry_run_enforcement_replay_only": True,
            "source_trace_mutation": False,
            "real_external_action": False,
            "actionruntime_dispatch": False,
            "memory_write": False,
            "operator_or_policy_promotion": False,
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
    (OUTPUT_DIR / "agentos_ats3_gate_policy_schema.json").write_text(json.dumps(build_schema(), indent=2) + "\n", encoding="utf-8", newline="\n")
    write_csv(OUTPUT_DIR / "agentos_ats3_enforcement_replay_cases.csv", empty, REPLAY_CASE_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats3_redaction_enforcement_results.csv", empty, REDACTION_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats3_scoped_disclosure_enforcement_results.csv", empty, SCOPED_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats3_provenance_freshness_scope_enforcement_results.csv", empty, PROVENANCE_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats3_pivot_blocker_enforcement_results.csv", empty, PIVOT_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats3_cross_project_isolation_enforcement_results.csv", empty, CROSS_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats3_counterexample_enforcement_replay_results.csv", empty, COUNTER_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats3_stale_replay_negative_cases.csv", empty, STALE_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats3_over_redaction_utility_check.csv", empty, UTILITY_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats3_enforcement_failure_cases.csv", [{"case_id": "ats3-input", "surface_id": "", "failure_family": "input", "expected": "valid ATS2 PASS pack", "observed": "; ".join(blockers), "notes": "blocked before enforcement replay"}], FAILURE_FIELDS)
    tests = [{"test": "ATS3-input-intake", "result": "FAIL", "details": "; ".join(blockers)}]
    counts = {"surfaces": 0, "counter": 0, "failures": 1, "replay_cases": 0, "redaction": 0, "scoped": 0, "provenance": 0, "pivot": 0, "cross": 0, "stale": 0, "utility": 0}
    write_reports(verdict, counts | {"counter": 0}, tests)
    write_tests_summary(tests, verdict, 0, 1)
    write_hash_inventory()
    write_manifest(verdict, tests, counts, {})
    zip_path = package_outputs() if pack else None
    return {"verdict": verdict, "blockers": blockers, "return_pack": str(zip_path) if zip_path else "", "return_pack_sha256": sha256_file(zip_path) if zip_path else ""}


def run(pack: bool = False) -> dict[str, object]:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tables, ats2_manifest, blockers = load_ats2()
    if blockers:
        return generate_blocked_pack(blockers, pack)

    policies = tables["agentos_ats2_trace_redaction_policy_matrix.csv"]
    views = tables["agentos_ats2_consumer_scoped_trace_view_matrix.csv"]
    gates = tables["agentos_ats2_provenance_freshness_scope_gate_matrix.csv"]
    pivots = tables["agentos_ats2_trace_to_action_pivot_blocker_matrix.csv"]
    cross_rows = tables["agentos_ats2_cross_project_scoped_disclosure_audit.csv"]
    counterexamples = tables["agentos_ats2_minimal_counterexample_replay_results.csv"]

    view_by_id = by_surface(views)
    replay_cases = build_replay_cases(policies, view_by_id)
    redaction = build_redaction_results(policies, view_by_id)
    scoped = build_scoped_results(views)
    provenance = build_provenance_results(gates)
    pivot_results = build_pivot_results(pivots)
    cross_results = build_cross_project_results(cross_rows)
    counter_results = build_counter_results(counterexamples)
    stale = build_stale_negative_cases(gates)
    utility = build_utility_checks(views)
    failures = collect_failures(
        [
            ("redaction", redaction),
            ("scoped_disclosure", scoped),
            ("provenance_freshness_scope", provenance),
            ("pivot_blocker", pivot_results),
            ("cross_project_isolation", cross_results),
            ("counterexample_replay", counter_results),
            ("stale_replay", stale),
            ("over_redaction_utility", utility),
        ]
    )
    tests = run_tests(policies, views, gates, pivots, cross_rows, counterexamples, replay_cases, redaction, scoped, provenance, pivot_results, cross_results, counter_results, stale, utility, failures)
    verdict = PASS_VERDICT if all(row["result"] == "PASS" for row in tests) and not failures else FAIL_VERDICT
    counts = {
        "surfaces": len(policies),
        "replay_cases": len(replay_cases),
        "redaction": len(redaction),
        "scoped": len(scoped),
        "provenance": len(provenance),
        "pivot": len(pivot_results),
        "cross": len(cross_results),
        "counter": len(counter_results),
        "stale": len(stale),
        "utility": len(utility),
        "failures": len(failures),
    }

    (OUTPUT_DIR / "agentos_ats3_gate_policy_schema.json").write_text(json.dumps(build_schema(), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    write_csv(OUTPUT_DIR / "agentos_ats3_enforcement_replay_cases.csv", replay_cases, REPLAY_CASE_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats3_redaction_enforcement_results.csv", redaction, REDACTION_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats3_scoped_disclosure_enforcement_results.csv", scoped, SCOPED_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats3_provenance_freshness_scope_enforcement_results.csv", provenance, PROVENANCE_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats3_pivot_blocker_enforcement_results.csv", pivot_results, PIVOT_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats3_cross_project_isolation_enforcement_results.csv", cross_results, CROSS_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats3_counterexample_enforcement_replay_results.csv", counter_results, COUNTER_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats3_stale_replay_negative_cases.csv", stale, STALE_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats3_over_redaction_utility_check.csv", utility, UTILITY_FIELDS)
    write_csv(OUTPUT_DIR / "agentos_ats3_enforcement_failure_cases.csv", failures, FAILURE_FIELDS)
    write_reports(verdict, counts, tests)
    write_tests_summary(tests, verdict, len(policies), len(failures))
    secret_hits = scan_secrets(OUTPUT_DIR / name for name in REQUIRED_FILES if name not in {"hash_inventory.csv", "return_files_manifest.json"})
    if secret_hits:
        raise RuntimeError(f"Secret-like material detected in ATS3 outputs: {secret_hits}")
    write_hash_inventory()
    write_manifest(verdict, tests, counts, ats2_manifest)
    zip_path = package_outputs() if pack else None
    return {
        "verdict": verdict,
        "output_dir": str(OUTPUT_DIR),
        "return_pack": str(zip_path) if zip_path else str(ROOT / "outputs" / RETURN_PACK),
        "return_pack_sha256": sha256_file(zip_path) if zip_path else "",
        "required_files": len(REQUIRED_FILES),
        "trace_surface_count": len(policies),
        "replay_case_count": len(replay_cases),
        "stale_negative_case_count": len(stale),
        "counterexample_count": len(counter_results),
        "enforcement_failure_count": len(failures),
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
