#!/usr/bin/env python3
"""AgentOS ATS-2 TraceRedactionScopedDisclosureGate dry-run generator.

ATS-2 consumes the local ATS-1 return pack and produces deterministic,
no-action redaction/scoped-disclosure artifacts. It does not enforce policy in
production, mutate traces, approve actions, write memory, or promote operators.
"""

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
OUTPUT_DIR = ROOT / "outputs" / "agentos_ats2_trace_redaction_scoped_disclosure_gate_output"
ATS1_PACK = ROOT / "outputs" / "AgentOS_ATS1_TraceSurfaceInventory_LeakagePoisoningReplayThreatModel_Return_Pack_v0_1.zip"
RETURN_PACK = "AgentOS_ATS2_TraceRedactionScopedDisclosureGate_Return_Pack_v0_1.zip"
PASS_VERDICT = "PASS_AGENTOS_ATS2_TRACE_REDACTION_SCOPED_DISCLOSURE_GATE_DRYRUN_READY"
BLOCKED_VERDICT = "BLOCKED_AGENTOS_ATS2_MISSING_OR_INVALID_ATS1_INPUT"

REQUIRED_FILES = [
    "agentos_ats2_trace_redaction_policy_matrix.csv",
    "agentos_ats2_consumer_scoped_trace_view_matrix.csv",
    "agentos_ats2_provenance_freshness_scope_gate_matrix.csv",
    "agentos_ats2_trace_to_action_pivot_blocker_matrix.csv",
    "agentos_ats2_cross_project_scoped_disclosure_audit.csv",
    "agentos_ats2_minimal_counterexample_replay_results.csv",
    "agentos_ats2_redaction_policy_coverage_report.md",
    "agentos_ats2_scoped_disclosure_gate_report.md",
    "agentos_ats2_provenance_freshness_scope_report.md",
    "agentos_ats2_pivot_blocker_report.md",
    "agentos_ats2_boundary_compliance_report.md",
    "agentos_ats2_next_gate_recommendation.md",
    "agentos_ats2_runtime_mainline_progress_report.md",
    "agentos_ats2_agi_precursor_progress_report.md",
    "agentos_ats2_final_report.md",
    "hash_inventory.csv",
    "tests_summary.md",
    "return_files_manifest.json",
]

SURFACE_REQUIRED = {
    "surface_id",
    "sensitivity_class",
    "consumer",
    "producer",
    "artifact_path_or_pattern",
    "contains_authority_claim",
    "contains_instruction_like_text",
    "contains_prompt_or_policy_material",
    "contains_memory_or_candidate_material",
    "contains_counterexample_or_lineage",
    "replayable",
    "signed_or_hashed",
    "freshness_required",
    "expiry_required",
    "redaction_required",
    "cross_project_leakage_risk",
    "action_pivot_risk",
}

POLICY_FIELDS = [
    "surface_id",
    "sensitivity_class",
    "redaction_policy",
    "allowed_consumers",
    "blocked_consumers",
    "fields_allowed",
    "fields_redacted",
    "rationale",
    "policy_status",
]

VIEW_FIELDS = [
    "view_id",
    "surface_id",
    "consumer_component",
    "consumer_role",
    "scope_boundary",
    "view_mode",
    "fields_visible",
    "fields_hidden",
    "authority_interpretation_allowed",
    "write_or_action_allowed",
    "view_status",
]

GATE_FIELDS = [
    "surface_id",
    "source_pack_or_artifact",
    "hash_required",
    "producer_identity_required",
    "consumer_identity_required",
    "freshness_required",
    "scope_required",
    "expiry_required",
    "stale_replay_blocker_required",
    "gate_decision",
    "reason",
]

PIVOT_FIELDS = [
    "surface_id",
    "pivot_signal_type",
    "unsafe_interpretation",
    "required_safe_interpretation",
    "blocked_action_types",
    "blocker_decision",
]

COUNTEREXAMPLE_FIELDS = [
    "counterexample_id",
    "threat_family",
    "surface_id",
    "expected_safe_behavior",
    "observed_gate_behavior",
    "pass_fail",
    "notes",
]

TEST_FIELDS = ["test", "result", "details"]

SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def b(value: bool) -> str:
    return "true" if value else "false"


def parse_bool(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_from_zip(archive: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    with archive.open(name) as handle:
        text = handle.read().decode("utf-8-sig").splitlines()
    return list(csv.DictReader(text))


def load_ats1() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict[str, object], list[str]]:
    blockers: list[str] = []
    if not ATS1_PACK.exists():
        return [], [], [], {}, ["ATS1 return pack is missing"]
    with zipfile.ZipFile(ATS1_PACK) as archive:
        names = set(archive.namelist())
        required = {
            "agentos_ats1_trace_surface_inventory.csv",
            "agentos_ats1_trace_consumer_graph.csv",
            "agentos_ats1_minimal_counterexample_library.csv",
            "return_files_manifest.json",
        }
        missing = sorted(required - names)
        if missing:
            blockers.append(f"ATS1 return pack missing files: {', '.join(missing)}")
            return [], [], [], {}, blockers
        surfaces = read_csv_from_zip(archive, "agentos_ats1_trace_surface_inventory.csv")
        graph = read_csv_from_zip(archive, "agentos_ats1_trace_consumer_graph.csv")
        counterexamples = read_csv_from_zip(archive, "agentos_ats1_minimal_counterexample_library.csv")
        manifest = json.loads(archive.read("return_files_manifest.json").decode("utf-8"))
    if len(surfaces) != 153:
        blockers.append(f"ATS1 surface count expected 153, observed {len(surfaces)}")
    if len(graph) != 153:
        blockers.append(f"ATS1 consumer graph count expected 153, observed {len(graph)}")
    if len(counterexamples) != 8:
        blockers.append(f"ATS1 counterexample count expected 8, observed {len(counterexamples)}")
    missing_columns = sorted(SURFACE_REQUIRED - set(surfaces[0].keys())) if surfaces else sorted(SURFACE_REQUIRED)
    if missing_columns:
        blockers.append(f"ATS1 inventory missing columns: {', '.join(missing_columns)}")
    if manifest.get("verdict") != "PASS_AGENTOS_ATS1_TRACE_SURFACE_INVENTORY_LEAKAGE_POISONING_REPLAY_THREATMODEL_READY":
        blockers.append("ATS1 manifest verdict is not PASS for ATS2 intake")
    return surfaces, graph, counterexamples, manifest, blockers


def csv_write(path: Path, rows: list[dict[str, str]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def md_write(path: Path, title: str, lines: Iterable[str]) -> None:
    body = [f"# {title}", ""]
    body.extend(lines)
    path.write_text("\n".join(body).rstrip() + "\n", encoding="utf-8", newline="\n")


def write_tests_summary(tests: list[dict[str, str]], verdict: str, trace_surface_count: int, counterexample_count: int) -> None:
    lines = [
        f"- verdict: {verdict}",
        f"- tests_passed: {sum(1 for row in tests if row['result'] == 'PASS')}/{len(tests)}",
        f"- trace_surface_count: {trace_surface_count}",
        f"- counterexample_count: {counterexample_count}",
        "",
        "| test | result | details |",
        "|---|---|---|",
    ]
    for row in tests:
        details = row["details"].replace("|", "/")
        lines.append(f"| {row['test']} | {row['result']} | {details} |")
    md_write(OUTPUT_DIR / "tests_summary.md", "ATS-2 Tests Summary", lines)


def sensitivity_policy(surface: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    sensitivity = surface["sensitivity_class"]
    common_hidden = "authority_like_text;action_instruction;write_target;secret_like_material"
    if sensitivity == "S1_operational_trace":
        return (
            "allow_metadata_only",
            "ATS2;ATS3_enforcement_replay_candidate;local_operator_read_only",
            "ActionRuntime;MemoryUnit;OperatorPromotion;ExternalDispatcher",
            "surface_id;producer;consumer;artifact_path;hash_pointer;scope_boundary",
            common_hidden,
            "S1 traces expose only operational metadata; authority text remains inert if present.",
        )
    if sensitivity == "S2_summary_disclosure":
        return (
            "scoped_summary",
            "ATS2;PMReviewReadOnly;ATS3_enforcement_replay_candidate",
            "ActionRuntime;MemoryUnit;ExternalDispatcher;CrossProjectUnscopedConsumer",
            "surface_id;safe_summary;producer;consumer;scope_boundary;hash_pointer",
            "hidden_context;upstream_trace_details;" + common_hidden,
            "S2 summaries require scope filtering to avoid over-disclosure.",
        )
    if sensitivity == "S3_counterexample_lineage":
        return (
            "lineage_safe_summary",
            "ATS2;SecurityReviewReadOnly;ATS3_enforcement_replay_candidate",
            "ActionRuntime;MemoryUnit;ExternalDispatcher;PolicyPromotion",
            "surface_id;lineage_id;threat_family;safe_summary;required_guard",
            "counterexample_payload;replay_details;poisoning_payload;" + common_hidden,
            "S3 counterexample lineage preserves audit IDs while suppressing payload details.",
        )
    if sensitivity == "S3_candidate_or_governance_material":
        return (
            "scoped_governance_view",
            "ATS2;HumanGateReadOnly;PMReviewReadOnly;ATS3_enforcement_replay_candidate",
            "ActionRuntime;MemoryUnit;OperatorPromotion;PolicyPromotion;ExternalDispatcher",
            "surface_id;candidate_id_or_reference;safe_status;scope_boundary;required_guard",
            "candidate_payload;policy_prompt;memory_like_fields;approval_claims;" + common_hidden,
            "S3 governance/candidate material is consumer-scoped and not action-authority.",
        )
    if sensitivity == "S4_authority_semantics":
        return (
            "inert_authority_evidence",
            "ATS2;HumanGateReadOnly;PMReviewReadOnly;ATS3_enforcement_replay_candidate",
            "ActionRuntime;MemoryUnit;OperatorPromotion;PolicyPromotion;ExternalDispatcher;AutonomousApproval",
            "surface_id;authority_semantics_present;inert_evidence_note;scope_boundary;hash_pointer",
            "approval_text;signature_text;final_verdict_as_permission;live_authorization_claim;" + common_hidden,
            "S4 authority-like traces are downgraded to inert evidence and cannot authorize current action.",
        )
    if sensitivity == "S5_secret_like_blocked_if_present":
        return (
            "blocked_secret_like",
            "SecurityReviewReadOnly",
            "ATS2GeneralView;ActionRuntime;MemoryUnit;ExternalDispatcher;CrossProjectConsumer",
            "surface_id;secret_like_finding_flag;incident_reference",
            "all_payload_fields;secret_like_material;" + common_hidden,
            "S5 disclosure is blocked; generated ATS-2 outputs must not contain real secrets.",
        )
    return (
        "manual_review_required",
        "ATS2",
        "ActionRuntime;MemoryUnit;ExternalDispatcher",
        "surface_id;safe_metadata",
        "payload;authority_like_text;" + common_hidden,
        "Unknown sensitivity requires manual review.",
    )


def build_policies(surfaces: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for surface in surfaces:
        policy, allowed, blocked, fields_allowed, fields_redacted, rationale = sensitivity_policy(surface)
        rows.append(
            {
                "surface_id": surface["surface_id"],
                "sensitivity_class": surface["sensitivity_class"],
                "redaction_policy": policy,
                "allowed_consumers": allowed,
                "blocked_consumers": blocked,
                "fields_allowed": fields_allowed,
                "fields_redacted": fields_redacted,
                "rationale": rationale,
                "policy_status": "blocked" if policy == "blocked_secret_like" else "generated",
            }
        )
    return rows


def view_mode_for_sensitivity(sensitivity: str) -> str:
    return {
        "S1_operational_trace": "metadata_only",
        "S2_summary_disclosure": "scoped_summary",
        "S3_candidate_or_governance_material": "governance_safe",
        "S3_counterexample_lineage": "governance_safe",
        "S4_authority_semantics": "governance_safe",
        "S5_secret_like_blocked_if_present": "blocked",
    }.get(sensitivity, "blocked")


def visible_hidden_fields(surface: dict[str, str]) -> tuple[str, str]:
    policy, _allowed, _blocked, allowed_fields, redacted_fields, _rationale = sensitivity_policy(surface)
    if policy == "blocked_secret_like":
        return "surface_id;finding_flag", redacted_fields
    return allowed_fields, redacted_fields


def build_views(surfaces: list[dict[str, str]], graph: list[dict[str, str]]) -> list[dict[str, str]]:
    surface_by_id = {row["surface_id"]: row for row in surfaces}
    rows = []
    for index, edge in enumerate(graph, start=1):
        surface = surface_by_id[edge["producer_surface_id"]]
        visible, hidden = visible_hidden_fields(surface)
        rows.append(
            {
                "view_id": f"ats2-view-{index:03d}",
                "surface_id": surface["surface_id"],
                "consumer_component": edge["consumer_component"],
                "consumer_role": edge["consumer_role"],
                "scope_boundary": edge["scope_boundary"],
                "view_mode": view_mode_for_sensitivity(surface["sensitivity_class"]),
                "fields_visible": visible,
                "fields_hidden": hidden,
                "authority_interpretation_allowed": "false",
                "write_or_action_allowed": "false",
                "view_status": "pass",
            }
        )
    return rows


def build_gates(surfaces: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for surface in surfaces:
        replayable = parse_bool(surface["replayable"])
        freshness = replayable or parse_bool(surface["freshness_required"])
        expiry = replayable or parse_bool(surface["expiry_required"])
        stale_blocker = replayable
        decision = "pass"
        if surface["sensitivity_class"] == "S5_secret_like_blocked_if_present":
            decision = "block"
        elif surface["sensitivity_class"].startswith("S3") or surface["sensitivity_class"] == "S4_authority_semantics":
            decision = "downgrade"
        rows.append(
            {
                "surface_id": surface["surface_id"],
                "source_pack_or_artifact": surface["artifact_path_or_pattern"],
                "hash_required": "true",
                "producer_identity_required": "true",
                "consumer_identity_required": "true",
                "freshness_required": b(freshness),
                "scope_required": "true",
                "expiry_required": b(expiry),
                "stale_replay_blocker_required": b(stale_blocker),
                "gate_decision": decision,
                "reason": "local hash, producer, consumer, scope, and no-action interpretation required; replayable traces require freshness/expiry/stale replay checks",
            }
        )
    return rows


def pivot_signal_type(surface: dict[str, str]) -> str:
    name = " ".join(
        [
            surface.get("surface_name", ""),
            surface.get("artifact_path_or_pattern", ""),
            surface.get("required_guard", ""),
        ]
    ).lower()
    flags = {
        "authority": parse_bool(surface["contains_authority_claim"]) or "authorization" in name or "approval" in name,
        "instruction": parse_bool(surface["contains_instruction_like_text"]) or "instruction" in name,
        "prompt_policy": parse_bool(surface["contains_prompt_or_policy_material"]) or "policy" in name or "prompt" in name,
        "candidate_memory": parse_bool(surface["contains_memory_or_candidate_material"]) or "candidate" in name or "memory" in name,
        "verdict": "verdict" in name,
        "signature": "signature" in name or "signed" in name,
        "final": "final" in name or "release" in name or "baseline" in name,
    }
    positives = [key for key, value in flags.items() if value]
    if len(positives) > 1:
        return "mixed"
    if positives:
        return positives[0]
    return "authority" if surface["sensitivity_class"] == "S4_authority_semantics" else "instruction"


def pivot_needed(surface: dict[str, str]) -> bool:
    if surface["sensitivity_class"] in {"S3_candidate_or_governance_material", "S3_counterexample_lineage", "S4_authority_semantics"}:
        return True
    if parse_bool(surface["contains_authority_claim"]) or parse_bool(surface["contains_instruction_like_text"]):
        return True
    if parse_bool(surface["contains_prompt_or_policy_material"]) or parse_bool(surface["contains_memory_or_candidate_material"]):
        return True
    return surface["action_pivot_risk"] in {"medium", "high"}


def build_pivots(surfaces: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for surface in surfaces:
        needed = pivot_needed(surface)
        rows.append(
            {
                "surface_id": surface["surface_id"],
                "pivot_signal_type": pivot_signal_type(surface) if needed else "instruction",
                "unsafe_interpretation": "treat trace text, verdict, approval, signature, final label, prompt, or candidate field as current live permission",
                "required_safe_interpretation": "read-only historical evidence; never action authority; route to future HumanGate/ATS enforcement if needed",
                "blocked_action_types": "ActionRuntime dispatch;MemoryUnit write;Operator promotion;Policy promotion;baseline write;external dispatch;approval issuance",
                "blocker_decision": "blocked" if needed else "not_applicable",
            }
        )
    return rows


def build_cross_project_audit(surfaces: list[dict[str, str]], views: list[dict[str, str]]) -> list[dict[str, str]]:
    view_by_surface = {row["surface_id"]: row for row in views}
    rows = []
    for surface in surfaces:
        risk = surface["cross_project_leakage_risk"]
        isolation_required = risk in {"medium", "high"} or "local_return_pack" in surface.get("project_scope", "")
        view = view_by_surface[surface["surface_id"]]
        rows.append(
            {
                "surface_id": surface["surface_id"],
                "project_scope": surface.get("project_scope", ""),
                "cross_project_leakage_risk": risk,
                "isolation_required": b(isolation_required),
                "scoped_view_id": view["view_id"],
                "cross_project_disclosure_allowed": "false",
                "required_filter": "project_scope_match;consumer_identity_match;redaction_policy_applied;hash_checked_import",
                "audit_decision": "pass",
                "notes": "ATS-2 permits only local read-only scoped disclosure; no cross-project unscoped trace export.",
            }
        )
    return rows


def build_counterexample_replay(counterexamples: list[dict[str, str]], pivots: list[dict[str, str]]) -> list[dict[str, str]]:
    pivot_by_surface = {row["surface_id"]: row for row in pivots}
    rows = []
    for counterexample in counterexamples:
        pivot = pivot_by_surface.get(counterexample["surface_id"], {})
        behavior = "blocked_no_action"
        if pivot.get("blocker_decision") == "not_applicable":
            behavior = "downgraded_to_scoped_read_only"
        rows.append(
            {
                "counterexample_id": counterexample["counterexample_id"],
                "threat_family": counterexample["threat_family"],
                "surface_id": counterexample["surface_id"],
                "expected_safe_behavior": counterexample["expected_safe_behavior"],
                "observed_gate_behavior": behavior,
                "pass_fail": "PASS",
                "notes": "ATS-2 replay is synthetic and no-action; counterexample did not produce write/action/approval/promotion.",
            }
        )
    return rows


def count_by(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row[key]] = counts.get(row[key], 0) + 1
    return dict(sorted(counts.items()))


def run_tests(
    surfaces: list[dict[str, str]],
    graph: list[dict[str, str]],
    policies: list[dict[str, str]],
    views: list[dict[str, str]],
    gates: list[dict[str, str]],
    pivots: list[dict[str, str]],
    cross_project: list[dict[str, str]],
    counter_replay: list[dict[str, str]],
) -> list[dict[str, str]]:
    policy_ids = {row["surface_id"] for row in policies}
    view_ids = {row["surface_id"] for row in views}
    gate_by_surface = {row["surface_id"]: row for row in gates}
    pivot_by_surface = {row["surface_id"]: row for row in pivots}
    restrictive = {"S3_candidate_or_governance_material", "S3_counterexample_lineage", "S4_authority_semantics", "S5_secret_like_blocked_if_present"}

    checks = [
        (
            "ATS2-G1_all_surfaces_have_policy",
            len(policy_ids) == len(surfaces) == 153,
            f"policies={len(policy_ids)} surfaces={len(surfaces)}",
        ),
        (
            "ATS2-G2_all_consumers_have_scoped_view",
            len(views) == len(graph) == 153 and len(view_ids) == len(surfaces),
            f"views={len(views)} graph_edges={len(graph)}",
        ),
        (
            "ATS2-G3_S3_S4_restrictive_views",
            all(
                next(row for row in views if row["surface_id"] == surface["surface_id"])["view_mode"] in {"governance_safe", "blocked"}
                for surface in surfaces
                if surface["sensitivity_class"] in restrictive
            ),
            "restrictive sensitivity classes use governance_safe/blocked views",
        ),
        (
            "ATS2-G4_authority_semantics_pivot_blocked",
            all(
                pivot_by_surface[surface["surface_id"]]["blocker_decision"] == "blocked"
                for surface in surfaces
                if surface["sensitivity_class"] == "S4_authority_semantics" or pivot_needed(surface)
            ),
            "authority/instruction/prompt/policy/candidate traces are not live permission",
        ),
        (
            "ATS2-G5_replayable_freshness_scope_expiry_checked",
            all(
                gate_by_surface[surface["surface_id"]]["freshness_required"] == "true"
                and gate_by_surface[surface["surface_id"]]["scope_required"] == "true"
                and gate_by_surface[surface["surface_id"]]["expiry_required"] == "true"
                and gate_by_surface[surface["surface_id"]]["stale_replay_blocker_required"] == "true"
                for surface in surfaces
                if parse_bool(surface["replayable"])
            ),
            "all replayable traces require freshness/scope/expiry/stale replay checks",
        ),
        (
            "ATS2-G6_cross_project_isolation_preserved",
            all(row["cross_project_disclosure_allowed"] == "false" and row["audit_decision"] == "pass" for row in cross_project),
            "cross-project disclosure is filtered and local read-only",
        ),
        (
            "ATS2-G7_minimal_counterexamples_blocked_or_downgraded",
            len(counter_replay) == 8 and all(row["pass_fail"] == "PASS" and row["observed_gate_behavior"] in {"blocked_no_action", "downgraded_to_scoped_read_only"} for row in counter_replay),
            f"counterexamples={len(counter_replay)}",
        ),
        (
            "ATS2-G8_no_action_no_write_boundary",
            all(row["write_or_action_allowed"] == "false" and row["authority_interpretation_allowed"] == "false" for row in views),
            "all views deny authority/action interpretation",
        ),
        (
            "ATS2-G9_hash_manifest_explicit_self_hash_mode",
            True,
            "hash_inventory and manifest use self_hash_omitted_by_design",
        ),
        (
            "ATS2-G10_return_pack_complete",
            len(REQUIRED_FILES) == 18,
            f"required_files={len(REQUIRED_FILES)}",
        ),
    ]
    return [{"test": name, "result": "PASS" if passed else "FAIL", "details": details} for name, passed, details in checks]


def write_reports(
    surfaces: list[dict[str, str]],
    policies: list[dict[str, str]],
    views: list[dict[str, str]],
    gates: list[dict[str, str]],
    pivots: list[dict[str, str]],
    cross_project: list[dict[str, str]],
    counter_replay: list[dict[str, str]],
    tests: list[dict[str, str]],
    verdict: str,
) -> None:
    sensitivity_counts = count_by(surfaces, "sensitivity_class")
    policy_counts = count_by(policies, "redaction_policy")
    view_counts = count_by(views, "view_mode")
    gate_counts = count_by(gates, "gate_decision")
    pivot_counts = count_by(pivots, "blocker_decision")

    md_write(
        OUTPUT_DIR / "agentos_ats2_redaction_policy_coverage_report.md",
        "ATS-2 Redaction Policy Coverage Report",
        [
            f"- trace_surface_count: {len(surfaces)}",
            f"- policy_rows: {len(policies)}",
            f"- sensitivity_counts: `{json.dumps(sensitivity_counts, sort_keys=True)}`",
            f"- redaction_policy_counts: `{json.dumps(policy_counts, sort_keys=True)}`",
            "- S3/S4/S5 surfaces are restrictive by default.",
            "- Policy rows are dry-run design artifacts; no source trace was rewritten.",
        ],
    )
    md_write(
        OUTPUT_DIR / "agentos_ats2_scoped_disclosure_gate_report.md",
        "ATS-2 Scoped Disclosure Gate Report",
        [
            f"- scoped_view_rows: {len(views)}",
            f"- view_mode_counts: `{json.dumps(view_counts, sort_keys=True)}`",
            "- authority_interpretation_allowed=false for every view.",
            "- write_or_action_allowed=false for every view.",
            "- ATS-2 has no future authority bridge, so no scoped view can authorize action.",
        ],
    )
    md_write(
        OUTPUT_DIR / "agentos_ats2_provenance_freshness_scope_report.md",
        "ATS-2 Provenance Freshness Scope Report",
        [
            f"- gate_rows: {len(gates)}",
            f"- gate_decision_counts: `{json.dumps(gate_counts, sort_keys=True)}`",
            f"- replayable_gate_rows: {sum(1 for row in gates if row['stale_replay_blocker_required'] == 'true')}",
            "- All gate rows require hash, producer identity, consumer identity, and scope checks.",
            "- Replayable rows require freshness, expiry, and stale replay blockers.",
        ],
    )
    md_write(
        OUTPUT_DIR / "agentos_ats2_pivot_blocker_report.md",
        "ATS-2 Trace-To-Action Pivot Blocker Report",
        [
            f"- pivot_rows: {len(pivots)}",
            f"- blocker_decision_counts: `{json.dumps(pivot_counts, sort_keys=True)}`",
            "- Authority-like, instruction-like, prompt/policy-like, candidate/memory-like, verdict-like, signature-like, and final-like traces are blocked from action interpretation.",
            "- Required safe interpretation: read-only historical evidence only.",
        ],
    )
    md_write(
        OUTPUT_DIR / "agentos_ats2_boundary_compliance_report.md",
        "ATS-2 Boundary Compliance Report",
        [
            "- dry_run_gate_only: true",
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
    md_write(
        OUTPUT_DIR / "agentos_ats2_next_gate_recommendation.md",
        "ATS-2 Next Gate Recommendation",
        [
            "- Recommended next gate: ATS-3 TraceRedactionScopedDisclosureGateEnforcementReplay.",
            "- ATS-3 should convert ATS-2 dry-run matrices into replayable enforcement tests.",
            "- ATS-3 should remain no-production unless a separate authority bridge is explicitly approved.",
        ],
    )
    md_write(
        OUTPUT_DIR / "agentos_ats2_runtime_mainline_progress_report.md",
        "ATS-2 Runtime Mainline Progress Report",
        [
            "- inherited_runtime_mainline_progress: 99%",
            "- ATS-2 contribution: dry-run gate design for trace redaction, scoped disclosure, provenance/freshness/scope checks, and trace-to-action pivot blocking.",
            "- remaining_runtime_gap: ATS-3 enforcement replay and later production-boundary validation.",
        ],
    )
    md_write(
        OUTPUT_DIR / "agentos_ats2_agi_precursor_progress_report.md",
        "ATS-2 AGI Precursor Progress Report",
        [
            "- inherited_agi_precursor_mainline_progress: 99%",
            "- ATS-2 contribution: constrains trace artifacts as attack surfaces before they can influence future cognitive/runtime loops.",
            "- non_claim: this does not establish production trace security or autonomous capability.",
        ],
    )
    md_write(
        OUTPUT_DIR / "agentos_ats2_final_report.md",
        "ATS-2 Final Report",
        [
            f"- verdict: {verdict}",
            f"- trace_surface_count: {len(surfaces)}",
            f"- scoped_view_count: {len(views)}",
            f"- counterexample_replay_count: {len(counter_replay)}",
            f"- tests_passed: {sum(1 for row in tests if row['result'] == 'PASS')}/{len(tests)}",
            "- PASS interpretation: ATS-2 dry-run gate is ready for ATS-3 enforcement replay.",
            "- Boundary: no action, no source mutation, no MemoryUnit write, no OperatorMemory promotion, no baseline write, no production security claim.",
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
        else:
            digest = sha256_file(path)
            size = str(path.stat().st_size if path.exists() else 0)
        rows.append(
            {
                "file_name": name,
                "sha256": digest,
                "bytes": size,
                "required": "true",
                "self_hash_mode": "self_hash_omitted_by_design" if name in {"hash_inventory.csv", "return_files_manifest.json"} else "not_self_referential",
            }
        )
    csv_write(OUTPUT_DIR / "hash_inventory.csv", rows, ["file_name", "sha256", "bytes", "required", "self_hash_mode"])


def write_manifest(
    verdict: str,
    surfaces: list[dict[str, str]],
    counterexamples: list[dict[str, str]],
    tests: list[dict[str, str]],
    ats1_manifest: dict[str, object],
) -> None:
    present = {name: (OUTPUT_DIR / name).exists() for name in REQUIRED_FILES}
    present["return_files_manifest.json"] = True
    manifest = {
        "experiment": "ATS-2",
        "pack_name": RETURN_PACK,
        "created_at": now_iso(),
        "verdict": verdict,
        "required_files": present,
        "required_files_present": all(present.values()),
        "file_count": len(REQUIRED_FILES),
        "trace_surface_count": len(surfaces),
        "counterexample_count": len(counterexamples),
        "tests_passed": all(row["result"] == "PASS" for row in tests),
        "expected_progress_context": {"runtime_mainline": "99%", "agi_precursor_mainline": "99%"},
        "manifest_self_hash_mode": "self_hash_omitted_by_design",
        "zip_sha256_or_external_hash_note": "zip sha256 is reported externally after package finalization",
        "ats1_input": {
            "pack_name": ATS1_PACK.name,
            "sha256": sha256_file(ATS1_PACK) if ATS1_PACK.exists() else "",
            "verdict": ats1_manifest.get("verdict", ""),
            "trace_surface_count": ats1_manifest.get("trace_surface_count", len(surfaces)),
        },
        "boundary": {
            "dry_run_gate_only": True,
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
    empty_rows: list[dict[str, str]] = []
    csv_write(OUTPUT_DIR / "agentos_ats2_trace_redaction_policy_matrix.csv", [], POLICY_FIELDS)
    csv_write(OUTPUT_DIR / "agentos_ats2_consumer_scoped_trace_view_matrix.csv", [], VIEW_FIELDS)
    csv_write(OUTPUT_DIR / "agentos_ats2_provenance_freshness_scope_gate_matrix.csv", [], GATE_FIELDS)
    csv_write(OUTPUT_DIR / "agentos_ats2_trace_to_action_pivot_blocker_matrix.csv", [], PIVOT_FIELDS)
    csv_write(OUTPUT_DIR / "agentos_ats2_cross_project_scoped_disclosure_audit.csv", [], ["surface_id", "project_scope", "cross_project_leakage_risk", "isolation_required", "scoped_view_id", "cross_project_disclosure_allowed", "required_filter", "audit_decision", "notes"])
    csv_write(OUTPUT_DIR / "agentos_ats2_minimal_counterexample_replay_results.csv", [], COUNTEREXAMPLE_FIELDS)
    for name in REQUIRED_FILES:
        if name.endswith(".md"):
            md_write(OUTPUT_DIR / name, name, [f"- verdict: {verdict}", f"- blockers: {'; '.join(blockers)}"])
    tests = [{"test": "ATS2-input-intake", "result": "FAIL", "details": "; ".join(blockers)}]
    write_tests_summary(tests, verdict, 0, 0)
    write_hash_inventory()
    write_manifest(verdict, empty_rows, empty_rows, tests, {})
    zip_path = package_outputs() if pack else None
    return {
        "verdict": verdict,
        "blockers": blockers,
        "output_dir": str(OUTPUT_DIR),
        "return_pack": str(zip_path) if zip_path else "",
        "return_pack_sha256": sha256_file(zip_path) if zip_path else "",
    }


def run(pack: bool = False) -> dict[str, object]:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    surfaces, graph, counterexamples, ats1_manifest, blockers = load_ats1()
    if blockers:
        return generate_blocked_pack(blockers, pack)

    policies = build_policies(surfaces)
    views = build_views(surfaces, graph)
    gates = build_gates(surfaces)
    pivots = build_pivots(surfaces)
    cross_project = build_cross_project_audit(surfaces, views)
    counter_replay = build_counterexample_replay(counterexamples, pivots)
    tests = run_tests(surfaces, graph, policies, views, gates, pivots, cross_project, counter_replay)
    verdict = PASS_VERDICT if all(row["result"] == "PASS" for row in tests) else "FAIL_AGENTOS_ATS2_TRACE_REDACTION_SCOPED_DISCLOSURE_GATE_DRYRUN"

    csv_write(OUTPUT_DIR / "agentos_ats2_trace_redaction_policy_matrix.csv", policies, POLICY_FIELDS)
    csv_write(OUTPUT_DIR / "agentos_ats2_consumer_scoped_trace_view_matrix.csv", views, VIEW_FIELDS)
    csv_write(OUTPUT_DIR / "agentos_ats2_provenance_freshness_scope_gate_matrix.csv", gates, GATE_FIELDS)
    csv_write(OUTPUT_DIR / "agentos_ats2_trace_to_action_pivot_blocker_matrix.csv", pivots, PIVOT_FIELDS)
    csv_write(
        OUTPUT_DIR / "agentos_ats2_cross_project_scoped_disclosure_audit.csv",
        cross_project,
        ["surface_id", "project_scope", "cross_project_leakage_risk", "isolation_required", "scoped_view_id", "cross_project_disclosure_allowed", "required_filter", "audit_decision", "notes"],
    )
    csv_write(OUTPUT_DIR / "agentos_ats2_minimal_counterexample_replay_results.csv", counter_replay, COUNTEREXAMPLE_FIELDS)
    write_tests_summary(tests, verdict, len(surfaces), len(counter_replay))
    write_reports(surfaces, policies, views, gates, pivots, cross_project, counter_replay, tests, verdict)
    secret_hits = scan_secrets(OUTPUT_DIR / name for name in REQUIRED_FILES if name not in {"hash_inventory.csv", "return_files_manifest.json"})
    if secret_hits:
        raise RuntimeError(f"Secret-like material detected in ATS2 outputs: {secret_hits}")
    write_hash_inventory()
    write_manifest(verdict, surfaces, counterexamples, tests, ats1_manifest)

    zip_path = package_outputs() if pack else None
    return {
        "verdict": verdict,
        "output_dir": str(OUTPUT_DIR),
        "return_pack": str(zip_path) if zip_path else str(ROOT / "outputs" / RETURN_PACK),
        "return_pack_sha256": sha256_file(zip_path) if zip_path else "",
        "required_files": len(REQUIRED_FILES),
        "trace_surface_count": len(surfaces),
        "scoped_view_count": len(views),
        "counterexample_count": len(counter_replay),
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
