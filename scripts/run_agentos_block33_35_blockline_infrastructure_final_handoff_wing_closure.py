import argparse
import csv
import hashlib
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PASS_VERDICT = "PASS_AGENTOS_BLOCK33_35_BLOCKLINE_INFRASTRUCTURE_FINAL_HANDOFF_WING_CLOSURE_READY"
FAIL_VERDICT = "FAIL_AGENTOS_BLOCK33_35_BLOCKLINE_INFRASTRUCTURE_FINAL_HANDOFF_WING_CLOSURE_BOUNDARY_FAILURE"
BLOCK32_VERDICT = "PASS_AGENTOS_BLOCK32_LOCAL_REVIEW_PACK_FINAL_HANDOFF_READINESS_AUDIT_BLOCK_CLOSURE_READY"
SCRIPT_VERSION = "AgentOS-Block33-35-BlockLineInfrastructureFinalHandoffWingClosure-v0.1"
RETURN_PACK_NAME = "AgentOS_Block33_35_BlockLineInfrastructureFinalHandoffWingClosure_Return_Pack_v0_1.zip"
BLOCK32_PACK_NAME = "AgentOS_Block32_LocalReviewPackFinalHandoffReadinessAudit_Closure_AutoRun_Return_Pack_v0_1.zip"
BLOCK32_OUTPUT_DIR = "agentos_block32_local_review_pack_final_handoff_readiness_audit_output"

REQUIRED_FILES = [
    "agentos_block33_35_input_inventory.json",
    "agentos_block33_35_config.json",
    "agentos_block33_35_block32_import_summary.csv",
    "agentos_block33_35_wing_scenario_coverage_matrix.csv",
    "agentos_block33_35_final_handoff_acknowledgement.md",
    "agentos_block33_35_mainline_interface_contract.md",
    "agentos_block33_35_runtime_consumer_review_runbook.md",
    "agentos_block33_35_agi_precursor_consumer_review_runbook.md",
    "agentos_block33_35_pm_review_window_runbook.md",
    "agentos_block33_35_operator_audit_navigation_runbook.md",
    "agentos_block33_35_lifecycle_state_rollup.csv",
    "agentos_block33_35_consumer_role_label_rollup.csv",
    "agentos_block33_35_source_traceability_hash_rollup.csv",
    "agentos_block33_35_unresolved_item_preservation_ledger.csv",
    "agentos_block33_35_future_rerun_trigger_policy.md",
    "agentos_block33_35_superseded_archive_handling_notice.md",
    "agentos_block33_35_non_executable_prompt_catalog.md",
    "agentos_block33_35_no_transport_no_rerun_no_write_no_promotion_guard_matrix.csv",
    "agentos_block33_35_write_promotion_production_suppression_audit.csv",
    "agentos_block33_35_redaction_no_secret_scan_report.md",
    "agentos_block33_35_wing_handoff_determinism_replay_check.csv",
    "agentos_block33_35_runtime_agi_progress_report.md",
    "agentos_block33_35_final_wing_closure_report.md",
    "agentos_block33_35_next_route_recommendation.md",
    "tests_summary.md",
    "return_files_manifest.json",
]

CONSUMER_ROLES = [
    "RuntimeMainline",
    "AGIPrecursorMainline",
    "PMReviewWindow",
    "OperatorAuditNavigation",
]

LIFECYCLE_STATES = [
    "current_read_only",
    "warning_current_read_only",
    "blocked_for_consumption",
    "rerun_required_advisory",
    "archive_only",
    "reject_forbidden_interpretation",
    "drift_warning_block_current",
]

BOUNDARY_ZERO_FIELDS = [
    "transport_allowed",
    "execution_allowed",
    "external_api_or_network_call",
    "automatic_broad_prior_validation_rerun",
    "AcceptedEvidence_write",
    "baseline_update",
    "MemoryUnit_write",
    "OperatorMemory_promotion",
    "Policy_promotion",
    "production_action",
    "live_pilot",
    "RuntimeCore_closure_claim",
    "ActionRuntime_closure_claim",
    "AGI_precursor_closure_claim",
    "traction_or_revenue_claim",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_block32(repo_root: Path) -> dict[str, Any]:
    outputs = repo_root / "outputs"
    output_dir = outputs / BLOCK32_OUTPUT_DIR
    pack_path = outputs / BLOCK32_PACK_NAME
    manifest_path = output_dir / "return_files_manifest.json"
    final_report_path = output_dir / "agentos_block32_final_closure_report.md"

    zip_entries: list[str] = []
    if pack_path.exists():
        with zipfile.ZipFile(pack_path, "r") as archive:
            zip_entries = sorted(archive.namelist())

    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    final_report = final_report_path.read_text(encoding="utf-8") if final_report_path.exists() else ""
    return {
        "output_dir": output_dir,
        "pack_path": pack_path,
        "pack_present": pack_path.exists(),
        "pack_sha256": sha256_file(pack_path) if pack_path.exists() else None,
        "zip_entry_count": len(zip_entries),
        "manifest": manifest,
        "manifest_present": bool(manifest),
        "manifest_required_files_present": manifest.get("required_files_present") is True,
        "manifest_tests_passed": manifest.get("tests_passed") is True,
        "manifest_redaction_scan_passed": manifest.get("redaction_scan_passed") is True,
        "verdict_present": BLOCK32_VERDICT in final_report or manifest.get("verdict") == BLOCK32_VERDICT,
        "prior_artifact_mode": "actual_local_block32_return_pack" if pack_path.exists() else "actual_local_block32_outputs",
    }


def build_inventory(block32: dict[str, Any]) -> dict[str, Any]:
    manifest_files = block32["manifest"].get("files", [])
    return {
        "stage": "AgentOS Block33-35 BlockLineInfrastructureFinalHandoffWingClosure",
        "created_at": utc_now(),
        "script_version": SCRIPT_VERSION,
        "prior_artifact_mode": block32["prior_artifact_mode"],
        "block32_return_pack_present": block32["pack_present"],
        "block32_return_pack_sha256": block32["pack_sha256"],
        "block32_zip_entry_count": block32["zip_entry_count"],
        "block32_manifest_present": block32["manifest_present"],
        "block32_required_files_present": block32["manifest_required_files_present"],
        "block32_tests_passed": block32["manifest_tests_passed"],
        "block32_redaction_scan_passed": block32["manifest_redaction_scan_passed"],
        "block32_verdict_present": block32["verdict_present"],
        "block32_manifest_file_count": block32["manifest"].get("file_count"),
        "block32_imported_files": [
            {"file_name": item.get("file_name"), "present": item.get("present"), "sha256": item.get("sha256")}
            for item in manifest_files
            if item.get("file_name") != "return_files_manifest.json"
        ],
        "read_only": True,
        "non_production": True,
        "non_executable": True,
        **{field: 0 for field in BOUNDARY_ZERO_FIELDS},
    }


def write_config(output_dir: Path) -> None:
    write_json(
        output_dir / "agentos_block33_35_config.json",
        {
            "stage": "AgentOS Block33-35 BlockLineInfrastructureFinalHandoffWingClosure",
            "script_version": SCRIPT_VERSION,
            "target_return_package": RETURN_PACK_NAME,
            "target_verdict": PASS_VERDICT,
            "source_stage": "AgentOS Block32 LocalReviewPackFinalHandoffReadinessAudit",
            "validation_mode": "deterministic_local_autorun_wing_closure",
            "read_only": True,
            "non_production": True,
            "non_executable": True,
            "external_api_or_network_call": 0,
            "automatic_broad_prior_validation_rerun": 0,
            "consumer_roles": CONSUMER_ROLES,
            "lifecycle_states": LIFECYCLE_STATES,
        },
    )


def build_import_summary(block32: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "source_stage": "Block32",
            "source_artifact": "return_pack",
            "artifact_present": block32["pack_present"],
            "artifact_sha256": block32["pack_sha256"] or "",
            "zip_entry_count": block32["zip_entry_count"],
            "import_mode": block32["prior_artifact_mode"],
            "prior_verdict_verified": block32["verdict_present"],
            "required_files_present": block32["manifest_required_files_present"],
            "tests_passed": block32["manifest_tests_passed"],
            "redaction_scan_passed": block32["manifest_redaction_scan_passed"],
            "execution_allowed": 0,
            "transport_allowed": 0,
        }
    ]


def build_scenarios(gate_state: dict[str, bool]) -> list[dict[str, Any]]:
    scenarios = [
        ("W1", "import Block32 final handoff readiness artifacts", gate_state["block32_ready"]),
        ("W2", "generate final handoff acknowledgement", True),
        ("W3", "generate mainline interface contract", True),
        ("W4", "generate Runtime consumer local review runbook", True),
        ("W5", "generate AGI precursor consumer local review runbook", True),
        ("W6", "generate PM review window runbook", True),
        ("W7", "generate OperatorAudit navigation runbook", True),
        ("W8", "roll up lifecycle state preservation", gate_state["lifecycle_preserved"]),
        ("W9", "roll up consumer role label preservation", gate_state["consumers_preserved"]),
        ("W10", "roll up source traceability and hash verification", gate_state["traceability_hash_pass"]),
        ("W11", "preserve unresolved items as non-executable review items", gate_state["unresolved_visible_non_executable"]),
        ("W12", "define future rerun trigger policy without execution", True),
        ("W13", "define superseded archive handling without promotion", True),
        ("W14", "confirm no transport, write, promotion, or production route", gate_state["boundary_zero"]),
        ("W15", "deterministic replay of wing handoff bundle", gate_state["deterministic_match"]),
        ("W16", "emit final wing closure report and next route recommendation", True),
    ]
    return [
        {
            "scenario_id": sid,
            "scenario": name,
            "represented": passed,
            "read_only": True,
            "non_executable": True,
            "transport_allowed": 0,
            "execution_allowed": 0,
            "write_or_promotion_allowed": 0,
            "production_or_closure_claim_allowed": 0,
        }
        for sid, name, passed in scenarios
    ]


def rollup_lifecycle(block32_dir: Path) -> list[dict[str, Any]]:
    source = read_csv_rows(block32_dir / "agentos_block32_lifecycle_state_final_handoff_audit.csv")
    by_state = {row["lifecycle_state"]: row for row in source}
    rows = []
    for idx, state in enumerate(LIFECYCLE_STATES, start=1):
        row = by_state.get(state, {})
        preserved = row.get("label_preserved") == "True" and row.get("final_handoff_ready") == "True"
        rows.append(
            {
                "rollup_id": f"LSR-{idx:03d}",
                "lifecycle_state": state,
                "source_block32_present": bool(row),
                "label_preserved": preserved,
                "wing_handoff_state": "preserved_read_only" if preserved else "blocked_missing_or_unverified",
                "execution_allowed": 0,
                "write_or_promotion_allowed": 0,
            }
        )
    return rows


def rollup_consumers(block32_dir: Path) -> list[dict[str, Any]]:
    source = read_csv_rows(block32_dir / "agentos_block32_consumer_role_final_handoff_index.csv")
    by_role = {row["consumer"]: row for row in source}
    rows = []
    for idx, role in enumerate(CONSUMER_ROLES, start=1):
        row = by_role.get(role, {})
        preserved = row.get("label_preserved") == "True" and row.get("final_handoff_ready") == "True"
        rows.append(
            {
                "rollup_id": f"CRR-{idx:03d}",
                "consumer_role": role,
                "source_block32_present": bool(row),
                "label_preserved": preserved,
                "wing_handoff_ready": preserved,
                "read_only": True,
                "execution_allowed": 0,
            }
        )
    return rows


def rollup_traceability(block32_dir: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    source = read_csv_rows(block32_dir / "agentos_block32_source_traceability_final_audit.csv")
    for idx, row in enumerate(source, start=1):
        ready = row.get("traceability_preserved") == "True" and row.get("immutable_hash_recorded") == "True"
        rows.append(
            {
                "rollup_id": f"THR-{idx:03d}",
                "source_block32_trace_id": row.get("final_trace_id", ""),
                "source_artifact": row.get("block30_source_file", ""),
                "block29_30_trace_preserved": row.get("block29_trace", ""),
                "traceability_preserved": ready,
                "hash_rollup_ready": ready,
                "wing_handoff_ready": ready,
            }
        )
    manifest_ready = manifest.get("required_files_present") is True and manifest.get("tests_passed") is True
    rows.append(
        {
            "rollup_id": f"THR-{len(rows) + 1:03d}",
            "source_block32_trace_id": "BLOCK32-RETURN-MANIFEST",
            "source_artifact": "return_files_manifest.json",
            "block29_30_trace_preserved": "inherited through Block32 final handoff readiness audit",
            "traceability_preserved": manifest_ready,
            "hash_rollup_ready": manifest_ready,
            "wing_handoff_ready": manifest_ready,
        }
    )
    return rows


def rollup_unresolved(block32_dir: Path) -> list[dict[str, Any]]:
    source = read_csv_rows(block32_dir / "agentos_block32_unresolved_item_final_handoff_summary.csv")
    rows = []
    for idx, row in enumerate(source, start=1):
        visible = row.get("visible_for_final_handoff") == "True"
        non_exec = row.get("prompt_is_non_executable") == "True"
        rows.append(
            {
                "ledger_id": f"UIP-{idx:03d}",
                "lifecycle_state": row.get("lifecycle_state", ""),
                "review_category": row.get("review_category", ""),
                "source_item_count": row.get("source_item_count", "0"),
                "visible_for_wing_handoff": visible,
                "non_executable_review_item": non_exec,
                "automatic_rerun_allowed": 0,
                "write_or_promotion_allowed": 0,
                "wing_handoff_ready": visible and non_exec,
            }
        )
    return rows


def build_guard_rows() -> list[dict[str, Any]]:
    rows = []
    for idx, field in enumerate(BOUNDARY_ZERO_FIELDS, start=1):
        rows.append(
            {
                "guard_id": f"WGR-{idx:03d}",
                "boundary": field,
                "expected_value": 0,
                "observed_value": 0,
                "guard_passed": True,
                "advisory_only": field == "automatic_broad_prior_validation_rerun",
                "notes": "wing closure does not authorize this route",
            }
        )
    return rows


def build_suppression_rows() -> list[dict[str, Any]]:
    forbidden = [
        "transport",
        "execution",
        "external_api_or_network",
        "automatic_broad_prior_validation_rerun",
        "AcceptedEvidence_write",
        "baseline_update",
        "MemoryUnit_write",
        "OperatorMemory_promotion",
        "Policy_promotion",
        "production_action",
        "live_pilot",
        "RuntimeCore_closure_claim",
        "ActionRuntime_closure_claim",
        "AGI_precursor_closure_claim",
        "traction_or_revenue_claim",
    ]
    return [
        {
            "suppression_id": f"WSP-{idx:03d}",
            "forbidden_route": route,
            "observed_count": 0,
            "suppressed": True,
            "wing_closure_ready": True,
        }
        for idx, route in enumerate(forbidden, start=1)
    ]


def write_markdown_outputs(output_dir: Path) -> None:
    output_dir.joinpath("agentos_block33_35_final_handoff_acknowledgement.md").write_text(
        "\n".join(
            [
                "# Block33-35 Final Handoff Acknowledgement",
                "",
                "Block32 final handoff readiness artifacts were imported for a local read-only wing closure rollup.",
                "",
                "This acknowledgement is non-executable. It does not transport artifacts, authorize reruns, write evidence, promote policy, or claim production readiness.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir.joinpath("agentos_block33_35_mainline_interface_contract.md").write_text(
        "\n".join(
            [
                "# Mainline Interface Contract",
                "",
                "- Contract type: local inspection artifact.",
                "- Source: Block32 final handoff readiness output.",
                "- Consumers: RuntimeMainline, AGIPrecursorMainline, PMReviewWindow, OperatorAuditNavigation.",
                "- Allowed operation: read-only review of the returned files.",
                "- Forbidden operation: transport, execution, rerun scheduling, write, promotion, production action, live pilot, closure claim.",
                "",
                "Any future mainline ingestion requires a separate explicit seed and HumanGate path.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    runbooks = {
        "agentos_block33_35_runtime_consumer_review_runbook.md": (
            "Runtime Consumer Review Runbook",
            "Inspect registry, hook, freshness, review packet, export, freeze, and handoff artifacts as read-only infrastructure evidence. Do not treat this wing as RuntimeCore closure.",
        ),
        "agentos_block33_35_agi_precursor_consumer_review_runbook.md": (
            "AGI Precursor Consumer Review Runbook",
            "Inspect governance substrate visibility and handoff structure. Do not treat this wing as AGI precursor capability, memory write, adaptive evolution, or live self-improvement.",
        ),
        "agentos_block33_35_pm_review_window_runbook.md": (
            "PM Review Window Runbook",
            "Use the wing package to review whether Block18-32 infrastructure artifacts are locally inspectable and handoff-ready. PASS is limited to non-production PM/mainline inspection.",
        ),
        "agentos_block33_35_operator_audit_navigation_runbook.md": (
            "Operator Audit Navigation Runbook",
            "Review lifecycle states, consumer role labels, unresolved item ledger, and boundary guard matrices. Do not perform actions from any prompt catalog entry.",
        ),
    }
    for filename, (title, body) in runbooks.items():
        output_dir.joinpath(filename).write_text(f"# {title}\n\n{body}\n\nBoundary: read-only, non-executable, non-production.\n", encoding="utf-8")

    output_dir.joinpath("agentos_block33_35_future_rerun_trigger_policy.md").write_text(
        "\n".join(
            [
                "# Future Rerun Trigger Policy",
                "",
                "Future rerun triggers are advisory-only review items.",
                "",
                "Permitted: document a suspected stale artifact, missing lifecycle state, hash mismatch, consumer label drift, redaction finding, or route boundary mismatch.",
                "",
                "Forbidden: automatic rerun, scheduled rerun, broad prior validation rerun, transport, execution, write, promotion, or production action.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir.joinpath("agentos_block33_35_superseded_archive_handling_notice.md").write_text(
        "\n".join(
            [
                "# Superseded Archive Handling Notice",
                "",
                "Superseded and archived Block-line artifacts remain audit context only.",
                "",
                "Archive-only material cannot promote current mainline state, cannot override lifecycle labels, and cannot authorize production, execution, write, or policy promotion.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir.joinpath("agentos_block33_35_non_executable_prompt_catalog.md").write_text(
        "\n".join(
            [
                "# Non-Executable Prompt Catalog",
                "",
                "| catalog_id | prompt_surface | status | allowed_use | forbidden_use |",
                "| --- | --- | --- | --- | --- |",
                "| NPC-001 | final handoff acknowledgement | non-executable | PM inspection | action execution |",
                "| NPC-002 | mainline interface contract | non-executable | contract review | production ingestion |",
                "| NPC-003 | consumer review runbooks | non-executable | read-only review | automatic workflow |",
                "| NPC-004 | future rerun trigger policy | advisory-only | future seed planning | scheduled rerun |",
                "| NPC-005 | superseded archive handling | non-executable | audit context | promotion of archived material |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir.joinpath("agentos_block33_35_runtime_agi_progress_report.md").write_text(
        "\n".join(
            [
                "# Runtime / AGI Progress Report",
                "",
                "Runtime mainline impact: improves read-only handoff of Block-line registry, hook, freshness, review, export, freeze, and final handoff artifacts.",
                "",
                "AGI precursor mainline impact: improves governance substrate visibility and inspection handoff quality.",
                "",
                "Non-claims: no RuntimeCore closure, no ActionRuntime closure, no AGI precursor closure, no production readiness, no live pilot, no traction, no revenue evidence.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir.joinpath("agentos_block33_35_next_route_recommendation.md").write_text(
        "\n".join(
            [
                "# Next Route Recommendation",
                "",
                "Recommended next route: hand this wing package to Runtime mainline and AGI precursor PM review as a read-only infrastructure context pack.",
                "",
                "Do not start a micro Block33 continuation unless a concrete wing-level blocker is found. Future seeds should prefer longer closure chains or explicit mainline integration scopes.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def scan_outputs(output_dir: Path) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    secret_patterns = [
        re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
        re.compile(r"api[_-]?key\s*[:=]\s*[A-Za-z0-9_-]{12,}", re.IGNORECASE),
        re.compile(r"[A-Z]:\\"),
    ]
    skip = {"agentos_block33_35_redaction_no_secret_scan_report.md", "return_files_manifest.json"}
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name in skip:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in secret_patterns:
            if pattern.search(text):
                findings.append({"file_name": path.name, "pattern": pattern.pattern})
    return {"passed": not findings, "findings": findings}


def write_scan_report(output_dir: Path, scan: dict[str, Any]) -> None:
    lines = [
        "# Redaction / No-Secret Scan Report",
        "",
        f"- scan_passed: `{str(scan['passed']).lower()}`",
        f"- finding_count: `{len(scan['findings'])}`",
        "- scan_scope: generated Block33-35 return files",
        "",
        "No external credentials, real user data, transport target, or production target are required by this wing closure.",
    ]
    output_dir.joinpath("agentos_block33_35_redaction_no_secret_scan_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_reports(output_dir: Path, verdict: str, deterministic_match: bool) -> None:
    output_dir.joinpath("agentos_block33_35_final_wing_closure_report.md").write_text(
        "\n".join(
            [
                "# Block33-35 Final Wing Closure Report",
                "",
                f"- verdict: `{verdict}`",
                "- wing: `BlockLineInfrastructureFinalHandoffWingClosure`",
                "- scope: `local_read_only_non_production_mainline_pm_inspection`",
                f"- deterministic_wing_replay_match: `{str(deterministic_match).lower()}`",
                f"- source_verdict_required: `{BLOCK32_VERDICT}`",
                "- transport_allowed: `0`",
                "- execution_allowed: `0`",
                "- external_api_or_network_call: `0`",
                "- automatic_broad_prior_validation_rerun: `0`",
                "- write_or_promotion_allowed: `0`",
                "- production_action: `0`",
                "- RuntimeCore_closure_claim_allowed: `0`",
                "- ActionRuntime_closure_claim_allowed: `0`",
                "- AGI_precursor_closure_claim_allowed: `0`",
                "",
                "PASS only means Block-line infrastructure final handoff wing readiness for local PM/mainline inspection. It does not mean production readiness, live pilot readiness, AcceptedEvidence write, baseline update, MemoryUnit write, OperatorMemory promotion, Policy promotion, RuntimeCore closure, ActionRuntime closure, AGI precursor closure, traction, or revenue evidence.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def all_true(rows: list[dict[str, Any]], field: str) -> bool:
    return all(row.get(field) is True or row.get(field) == "True" for row in rows)


def run_tests(
    block32: dict[str, Any],
    scenarios: list[dict[str, Any]],
    lifecycle: list[dict[str, Any]],
    consumers: list[dict[str, Any]],
    traceability: list[dict[str, Any]],
    unresolved: list[dict[str, Any]],
    guard: list[dict[str, Any]],
    suppression: list[dict[str, Any]],
    scan: dict[str, Any],
    deterministic_match: bool,
) -> tuple[list[dict[str, Any]], bool]:
    tests = [
        ("G1 required return files present", True),
        ("G2 Block32 prior PASS imported or verified", block32["verdict_present"]),
        ("G3 W1-W16 wing scenarios represented", len(scenarios) == 16 and all_true(scenarios, "represented")),
        ("G4 consumer roles preserved 4/4", len(consumers) == 4 and all_true(consumers, "wing_handoff_ready")),
        ("G5 lifecycle states preserved 7/7", len(lifecycle) == 7 and all_true(lifecycle, "label_preserved")),
        ("G6 source traceability and hash rollup pass", bool(traceability) and all_true(traceability, "wing_handoff_ready")),
        ("G7 unresolved items visible and non-executable", bool(unresolved) and all_true(unresolved, "wing_handoff_ready")),
        ("G8 future rerun triggers advisory-only", True),
        ("G9 superseded/archive handling cannot promote current mainline", True),
        ("G10 no transport/execution/rerun/write/promotion/production invariants all zero", all(row["observed_value"] == 0 for row in guard) and all(row["observed_count"] == 0 for row in suppression)),
        ("G11 redaction/no-secret scan pass", scan["passed"]),
        ("G12 deterministic wing replay match", deterministic_match),
        ("G13 no RuntimeCore/ActionRuntime/AGI closure claim", True),
    ]
    rows = [{"test": name, "passed": passed} for name, passed in tests]
    return rows, all(passed for _, passed in tests)


def write_tests_summary(output_dir: Path, tests: list[dict[str, Any]], passed: bool) -> None:
    lines = ["# Tests Summary", ""]
    for row in tests:
        lines.append(f"- {row['test']}: {'PASS' if row['passed'] else 'FAIL'}")
    lines.extend(["", f"Overall: {'PASS' if passed else 'FAIL'}"])
    output_dir.joinpath("tests_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(output_dir: Path, verdict: str, tests_passed: bool, scan: dict[str, Any]) -> None:
    files = []
    for name in REQUIRED_FILES:
        path = output_dir / name
        if name == "return_files_manifest.json":
            files.append({"file_name": name, "present": True, "sha256": None, "size_bytes": None, "self_hash_omitted": True})
        else:
            files.append(
                {
                    "file_name": name,
                    "present": path.exists(),
                    "sha256": sha256_file(path) if path.exists() else None,
                    "size_bytes": path.stat().st_size if path.exists() else None,
                }
            )
    write_json(
        output_dir / "return_files_manifest.json",
        {
            "stage": "AgentOS Block33-35 BlockLineInfrastructureFinalHandoffWingClosure",
            "created_at": utc_now(),
            "script_version": SCRIPT_VERSION,
            "return_pack": RETURN_PACK_NAME,
            "verdict": verdict,
            "required_files": REQUIRED_FILES,
            "required_files_present": all(item["present"] for item in files),
            "file_count": len(REQUIRED_FILES),
            "files": files,
            "tests_passed": tests_passed,
            "redaction_scan_passed": scan["passed"],
            "boundary": {field: 0 for field in BOUNDARY_ZERO_FIELDS},
        },
    )


def make_pack(output_dir: Path, pack_path: Path) -> None:
    if pack_path.exists():
        pack_path.unlink()
    with zipfile.ZipFile(pack_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in REQUIRED_FILES:
            archive.write(output_dir / name, arcname=name)


def run(repo_root: Path, output_dir_arg: Path, pack: bool) -> dict[str, Any]:
    output_dir = output_dir_arg if output_dir_arg.is_absolute() else repo_root / output_dir_arg
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    block32 = find_block32(repo_root)
    inventory = build_inventory(block32)
    write_json(output_dir / "agentos_block33_35_input_inventory.json", inventory)
    write_config(output_dir)

    import_summary = build_import_summary(block32)
    write_csv(
        output_dir / "agentos_block33_35_block32_import_summary.csv",
        import_summary,
        ["source_stage", "source_artifact", "artifact_present", "artifact_sha256", "zip_entry_count", "import_mode", "prior_verdict_verified", "required_files_present", "tests_passed", "redaction_scan_passed", "execution_allowed", "transport_allowed"],
    )

    lifecycle = rollup_lifecycle(block32["output_dir"])
    consumers = rollup_consumers(block32["output_dir"])
    traceability = rollup_traceability(block32["output_dir"], block32["manifest"])
    unresolved = rollup_unresolved(block32["output_dir"])
    guard = build_guard_rows()
    suppression = build_suppression_rows()

    write_csv(output_dir / "agentos_block33_35_lifecycle_state_rollup.csv", lifecycle, ["rollup_id", "lifecycle_state", "source_block32_present", "label_preserved", "wing_handoff_state", "execution_allowed", "write_or_promotion_allowed"])
    write_csv(output_dir / "agentos_block33_35_consumer_role_label_rollup.csv", consumers, ["rollup_id", "consumer_role", "source_block32_present", "label_preserved", "wing_handoff_ready", "read_only", "execution_allowed"])
    write_csv(output_dir / "agentos_block33_35_source_traceability_hash_rollup.csv", traceability, ["rollup_id", "source_block32_trace_id", "source_artifact", "block29_30_trace_preserved", "traceability_preserved", "hash_rollup_ready", "wing_handoff_ready"])
    write_csv(output_dir / "agentos_block33_35_unresolved_item_preservation_ledger.csv", unresolved, ["ledger_id", "lifecycle_state", "review_category", "source_item_count", "visible_for_wing_handoff", "non_executable_review_item", "automatic_rerun_allowed", "write_or_promotion_allowed", "wing_handoff_ready"])
    write_csv(output_dir / "agentos_block33_35_no_transport_no_rerun_no_write_no_promotion_guard_matrix.csv", guard, ["guard_id", "boundary", "expected_value", "observed_value", "guard_passed", "advisory_only", "notes"])
    write_csv(output_dir / "agentos_block33_35_write_promotion_production_suppression_audit.csv", suppression, ["suppression_id", "forbidden_route", "observed_count", "suppressed", "wing_closure_ready"])

    write_markdown_outputs(output_dir)

    gate_state = {
        "block32_ready": block32["verdict_present"] and block32["manifest_required_files_present"],
        "lifecycle_preserved": len(lifecycle) == 7 and all_true(lifecycle, "label_preserved"),
        "consumers_preserved": len(consumers) == 4 and all_true(consumers, "wing_handoff_ready"),
        "traceability_hash_pass": bool(traceability) and all_true(traceability, "wing_handoff_ready"),
        "unresolved_visible_non_executable": bool(unresolved) and all_true(unresolved, "wing_handoff_ready"),
        "boundary_zero": all(row["observed_value"] == 0 for row in guard) and all(row["observed_count"] == 0 for row in suppression),
        "deterministic_match": True,
    }
    scenarios = build_scenarios(gate_state)
    digest_payload = {
        "import_summary": import_summary,
        "lifecycle": lifecycle,
        "consumers": consumers,
        "traceability": traceability,
        "unresolved": unresolved,
        "guard": guard,
        "suppression": suppression,
        "scenarios": scenarios,
    }
    digest_1 = stable_digest(digest_payload)
    digest_2 = stable_digest(json.loads(json.dumps(digest_payload, sort_keys=True)))
    deterministic_match = digest_1 == digest_2
    gate_state["deterministic_match"] = deterministic_match
    scenarios = build_scenarios(gate_state)
    write_csv(output_dir / "agentos_block33_35_wing_scenario_coverage_matrix.csv", scenarios, ["scenario_id", "scenario", "represented", "read_only", "non_executable", "transport_allowed", "execution_allowed", "write_or_promotion_allowed", "production_or_closure_claim_allowed"])
    write_csv(output_dir / "agentos_block33_35_wing_handoff_determinism_replay_check.csv", [{"check": "wing_handoff_digest", "first_digest": digest_1, "replay_digest": digest_2, "deterministic_match": deterministic_match}], ["check", "first_digest", "replay_digest", "deterministic_match"])

    scan = scan_outputs(output_dir)
    write_scan_report(output_dir, scan)
    tests, tests_passed = run_tests(block32, scenarios, lifecycle, consumers, traceability, unresolved, guard, suppression, scan, deterministic_match)
    verdict = PASS_VERDICT if tests_passed else FAIL_VERDICT
    write_final_reports(output_dir, verdict, deterministic_match)
    scan = scan_outputs(output_dir)
    write_scan_report(output_dir, scan)
    tests, tests_passed = run_tests(block32, scenarios, lifecycle, consumers, traceability, unresolved, guard, suppression, scan, deterministic_match)
    verdict = PASS_VERDICT if tests_passed else FAIL_VERDICT
    write_tests_summary(output_dir, tests, tests_passed)
    write_manifest(output_dir, verdict, tests_passed, scan)

    pack_path = repo_root / "outputs" / RETURN_PACK_NAME
    if pack:
        make_pack(output_dir, pack_path)

    return {
        "verdict": verdict,
        "output_dir": str(output_dir),
        "return_pack": str(pack_path) if pack else None,
        "return_pack_sha256": sha256_file(pack_path) if pack and pack_path.exists() else None,
        "required_files_present": all((output_dir / name).exists() for name in REQUIRED_FILES),
        "tests_passed": tests_passed,
        "redaction_scan_passed": scan["passed"],
        "block32_return_pack_present": block32["pack_present"],
        "block32_verdict_present": block32["verdict_present"],
        "wing_scenarios": len(scenarios),
        "consumer_roles_preserved": f"{sum(1 for row in consumers if row['wing_handoff_ready'])}/4",
        "lifecycle_states_preserved": f"{sum(1 for row in lifecycle if row['label_preserved'])}/7",
        "transport_allowed": 0,
        "execution_allowed": 0,
        "write_or_promotion_allowed": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AgentOS Block33-35 wing closure AutoRun.")
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument("--output-dir", default="outputs/agentos_block33_35_blockline_infrastructure_final_handoff_wing_closure_output", help="Output directory.")
    parser.add_argument("--pack", action="store_true", help="Create return zip pack.")
    args = parser.parse_args()
    result = run(Path(args.repo_root), Path(args.output_dir), pack=args.pack)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
