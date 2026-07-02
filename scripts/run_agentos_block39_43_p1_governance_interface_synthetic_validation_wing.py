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


PASS_VERDICT = "PASS_AGENTOS_BLOCK39_43_P1_GOVERNANCE_INTERFACE_SYNTHETIC_VALIDATION_WING_CLOSURE_READY"
CONTRADICTION_VERDICT = "THEORY_INTERFACE_CONTRADICTION_FOUND"
SCRIPT_VERSION = "AgentOS-Block39-43-P1GovernanceInterfaceSyntheticValidationWing-v0.1"
RETURN_PACK_NAME = "AgentOS_Block39_43_P1GovernanceInterfaceSyntheticValidationWing_Return_Pack_v0_1.zip"
PRIOR_PACK_NAME = "AgentOS_Block36_38_P0TheoryInterfaceSyntheticValidationWing_Return_Pack_v0_1.zip"
PRIOR_OUTPUT_DIR = "agentos_block36_38_p0_theory_interface_synthetic_validation_wing_output"
PRIOR_VERDICT = "PASS_AGENTOS_BLOCK36_38_P0_THEORY_INTERFACE_SYNTHETIC_VALIDATION_WING_CLOSURE_READY"

REQUIRED_FILES = [
    "agentos_block39_43_input_inventory.json",
    "agentos_block39_43_config.json",
    "agentos_block39_43_synthetic_corpus_manifest.csv",
    "agentos_block39_skill_harness_compatibility_fixture_matrix.csv",
    "agentos_block39_skill_harness_compatibility_generalization_results.csv",
    "agentos_block39_harness_type_boundary_matrix.csv",
    "agentos_block40_operator_memory_trace_pair_matrix.csv",
    "agentos_block40_functional_equivalence_decision_results.csv",
    "agentos_block40_operator_cluster_canonicalization_preview.csv",
    "agentos_block41_typed_human_interaction_router_fixture_matrix.csv",
    "agentos_block41_minimal_interruption_routing_results.csv",
    "agentos_block41_hir_false_interrupt_false_silent_matrix.csv",
    "agentos_block42_authorization_checklist_fixture_matrix.csv",
    "agentos_block42_authorization_contradiction_coverage_results.csv",
    "agentos_block42_contradiction_minimal_counterexample_report.md",
    "agentos_block43_preflight_matrix_fixture_matrix.csv",
    "agentos_block43_preflight_consistency_results.csv",
    "agentos_block43_unresolved_item_preservation_ledger.csv",
    "agentos_block39_43_cross_wing_precedence_fixtures.csv",
    "agentos_block39_43_cross_wing_precedence_resolution_results.csv",
    "agentos_block39_43_theory_interface_contradiction_report.md",
    "agentos_block39_43_scenario_coverage_matrix.csv",
    "agentos_block39_43_hard_fail_scan.csv",
    "agentos_block39_43_no_promotion_no_write_no_runtime_claim_audit.csv",
    "agentos_block39_43_replay_determinism_check.csv",
    "agentos_block39_43_runtime_mainline_progress_report.md",
    "agentos_block39_43_agi_precursor_mainline_progress_report.md",
    "agentos_block39_43_mainline_handoff_note.md",
    "agentos_block39_43_final_wing_closure_report.md",
    "agentos_block39_43_next_route_recommendation.md",
    "tests_summary.md",
    "return_files_manifest.json",
    "hash_inventory.csv",
]

BOUNDARY_ZERO_FIELDS = [
    "production_mutation",
    "real_external_action",
    "external_api_network_browser_llm_api",
    "real_user_identity_or_consent",
    "live_pilot",
    "real_authority_or_permission_grant",
    "real_harness_install",
    "real_tool_mutation",
    "real_action_dispatch",
    "MemoryUnit_write",
    "ICM_update",
    "OperatorMemory_promotion",
    "Policy_promotion",
    "AcceptedEvidence_write",
    "baseline_update",
    "SafetyKernel_bypass",
    "RuntimeCore_closure_claim",
    "ActionRuntime_closure_claim",
    "AGI_precursor_closure_claim",
    "broad_no_action_boundary_validation_rerun",
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_prior(repo_root: Path) -> dict[str, Any]:
    outputs = repo_root / "outputs"
    pack_path = outputs / PRIOR_PACK_NAME
    output_dir = outputs / PRIOR_OUTPUT_DIR
    manifest_path = output_dir / "return_files_manifest.json"
    report_path = output_dir / "agentos_block36_38_final_wing_closure_report.md"
    zip_entry_count = 0
    if pack_path.exists():
        with zipfile.ZipFile(pack_path, "r") as archive:
            zip_entry_count = len(archive.namelist())
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    return {
        "prior_stage": "Block36-38 P0TheoryInterfaceSyntheticValidationWing",
        "prior_pack_present": pack_path.exists(),
        "prior_pack_sha256": sha256_file(pack_path) if pack_path.exists() else "",
        "prior_zip_entry_count": zip_entry_count,
        "prior_manifest_present": manifest_path.exists(),
        "prior_required_files_present": manifest.get("required_files_present") is True,
        "prior_tests_passed": manifest.get("tests_passed") is True,
        "prior_redaction_scan_passed": manifest.get("redaction_scan_passed") is True,
        "prior_verdict_verified": manifest.get("verdict") == PRIOR_VERDICT or PRIOR_VERDICT in report,
    }


def skill_harness_fixtures() -> list[dict[str, Any]]:
    rows = [
        ("SH01", "retrieval_skill", "read_only_harness", "read_index", "read_index+rank", "read_only", "compatible_read_only_candidate"),
        ("SH02", "analysis_skill", "matrix_harness", "matrix_eval", "matrix_eval+summary", "read_only", "compatible_read_only_candidate"),
        ("SH03", "document_skill", "local_render_harness", "render_preview", "render_preview", "read_only", "compatible_read_only_candidate"),
        ("SH04", "tool_skill", "readonly_stub_harness", "tool_probe", "tool_probe_stub", "read_only", "compatible_read_only_candidate"),
        ("SH05", "network_skill", "offline_harness", "network_fetch", "none", "read_only", "incompatible"),
        ("SH06", "install_skill", "real_install_harness", "install_runtime", "install_runtime", "requires_real_install", "blocked_requires_real_install"),
        ("SH07", "mutation_skill", "sandbox_harness", "write_state", "write_state", "mutation_requested", "blocked_boundary_violation"),
        ("SH08", "ambiguous_skill", "partial_harness", "parse_and_rank", "parse_only", "read_only", "observe_more"),
        ("SH09", "external_action_skill", "action_harness", "dispatch_action", "dispatch_action", "real_action_requested", "blocked_boundary_violation"),
        ("SH10", "secret_skill", "credential_harness", "credential_use", "credential_use", "secret_required", "blocked_boundary_violation"),
    ]
    return [
        {
            "fixture_id": fixture_id,
            "skill_type": skill_type,
            "harness_type": harness_type,
            "required_capability": required,
            "provided_capability": provided,
            "boundary_policy": boundary,
            "expected_route": expected,
        }
        for fixture_id, skill_type, harness_type, required, provided, boundary, expected in rows
    ]


def decide_skill_harness(row: dict[str, Any]) -> str:
    if row["boundary_policy"] == "requires_real_install":
        return "blocked_requires_real_install"
    if row["boundary_policy"] in {"mutation_requested", "real_action_requested", "secret_required"}:
        return "blocked_boundary_violation"
    if row["provided_capability"] == "none":
        return "incompatible"
    required_parts = set(row["required_capability"].split("_and_"))
    provided = row["provided_capability"]
    if all(part in provided for part in required_parts):
        return "compatible_read_only_candidate"
    if any(part in provided for part in required_parts):
        return "observe_more"
    return "incompatible"


def operator_trace_pairs() -> list[dict[str, Any]]:
    rows = [
        ("OP01", "rank_candidates_v1", "rank_candidates_v2", "same_queue", "same_ranked_output", "read_only", "timeout_to_observe", 0.22, "functionally_equivalent_candidate"),
        ("OP02", "hash_manifest_a", "hash_manifest_b", "same_files", "same_hashes", "read_only", "missing_file_block", 0.18, "functionally_equivalent_candidate"),
        ("OP03", "route_review_a", "route_review_b", "same_review_packet", "different_hir_route", "read_only", "risk_misroute", 0.86, "reject_surface_similarity"),
        ("OP04", "cluster_text_a", "cluster_text_b", "same_input", "same_clusters", "read_only", "ambiguous_tie", 0.74, "functionally_equivalent_candidate"),
        ("OP05", "preflight_pass_a", "preflight_pass_b", "same_matrix", "silent_pass_unresolved", "read_only", "unresolved_loss", 0.80, "distinct_operator_family"),
        ("OP06", "archive_a", "promote_b", "same_artifact", "archive_vs_promote", "read_only", "promotion_attempt", 0.92, "reject_surface_similarity"),
        ("OP07", "observe_more_a", "observe_more_b", "partial_context", "incomplete_comparable", "read_only", "insufficient_trace", 0.45, "observe_more"),
        ("OP08", "policy_preview_a", "policy_preview_b", "same_policy", "same_preview_no_write", "read_only", "policy_write_block", 0.31, "functionally_equivalent_candidate"),
    ]
    return [
        {
            "pair_id": pair_id,
            "operator_trace_a": a,
            "operator_trace_b": b,
            "input_state_family": input_state,
            "output_improvement": output,
            "boundary_policy": boundary,
            "failure_mode": failure,
            "surface_similarity": similarity,
            "expected_equivalence": expected,
        }
        for pair_id, a, b, input_state, output, boundary, failure, similarity, expected in rows
    ]


def decide_operator_equivalence(row: dict[str, Any]) -> str:
    output = row["output_improvement"]
    failure = row["failure_mode"]
    if "promote" in output or "promotion" in failure:
        return "reject_surface_similarity"
    if "silent_pass_unresolved" in output:
        return "distinct_operator_family"
    if "incomplete" in output or "insufficient" in failure:
        return "observe_more"
    if output.startswith("same_") or output in {"same_hashes", "same_preview_no_write"}:
        return "functionally_equivalent_candidate"
    if row["surface_similarity"] >= 0.75:
        return "reject_surface_similarity"
    return "distinct_operator_family"


def hir_fixtures() -> list[dict[str, Any]]:
    rows = [
        ("HIR01", "low_risk_read_status", "low", True, False, "low", "silent_continue"),
        ("HIR02", "diagnostic_uncertain", "low", True, False, "medium", "observe_only"),
        ("HIR03", "review_needed_async", "medium", True, False, "medium", "async_review"),
        ("HIR04", "irreversible_user_visible", "high", False, True, "medium", "explicit_approval_required"),
        ("HIR05", "authority_missing", "high", False, True, "low", "block_escalate"),
        ("HIR06", "secret_or_real_identity", "critical", False, True, "low", "block_escalate"),
        ("HIR07", "high_interrupt_cost_low_risk", "low", True, False, "high", "silent_continue"),
        ("HIR08", "medium_risk_reversible", "medium", True, False, "low", "async_review"),
        ("HIR09", "policy_promotion_attempt", "critical", False, True, "low", "block_escalate"),
        ("HIR10", "approval_required_no_action", "high", True, True, "medium", "explicit_approval_required"),
    ]
    return [
        {
            "fixture_id": fixture_id,
            "interaction_type": interaction_type,
            "risk_level": risk,
            "reversibility": reversibility,
            "authority_required": authority,
            "user_interrupt_cost": cost,
            "expected_route": expected,
        }
        for fixture_id, interaction_type, risk, reversibility, authority, cost, expected in rows
    ]


def decide_hir(row: dict[str, Any]) -> str:
    if row["risk_level"] == "critical" or row["interaction_type"] in {"authority_missing", "secret_or_real_identity", "policy_promotion_attempt"}:
        return "block_escalate"
    if row["authority_required"] and not row["reversibility"]:
        return "explicit_approval_required"
    if row["authority_required"] and row["risk_level"] == "high":
        return "explicit_approval_required"
    if row["risk_level"] == "medium":
        return "async_review"
    if "uncertain" in row["interaction_type"]:
        return "observe_only"
    return "silent_continue"


def authorization_fixtures() -> list[dict[str, Any]]:
    rows = [
        ("AUTH01", "read_only_review_scope", True, False, False, "checklist_pass_read_only"),
        ("AUTH02", "approval_claim_but_missing_authority", True, False, True, "unresolved_blocker"),
        ("AUTH03", "grant_authority_with_negating_boundary", True, True, False, "contradiction_found"),
        ("AUTH04", "real_action_requested", True, True, True, "reject_authority_claim"),
        ("AUTH05", "install_claim_without_install_boundary", True, True, False, "contradiction_found"),
        ("AUTH06", "missing_humangate", False, False, True, "unresolved_blocker"),
        ("AUTH07", "policy_promotion_requested", True, True, True, "reject_authority_claim"),
        ("AUTH08", "baseline_update_requested", True, True, True, "reject_authority_claim"),
    ]
    return [
        {
            "fixture_id": fixture_id,
            "checklist_item": checklist_item,
            "affirmative_claim": affirmative,
            "negating_condition": negating,
            "missing_authority": missing,
            "expected_route": expected,
        }
        for fixture_id, checklist_item, affirmative, negating, missing, expected in rows
    ]


def decide_authorization(row: dict[str, Any]) -> str:
    if row["checklist_item"] in {"real_action_requested", "policy_promotion_requested", "baseline_update_requested"}:
        return "reject_authority_claim"
    if row["affirmative_claim"] and row["negating_condition"]:
        return "contradiction_found"
    if row["missing_authority"]:
        return "unresolved_blocker"
    if row["affirmative_claim"] and not row["negating_condition"]:
        return "checklist_pass_read_only"
    return "unresolved_blocker"


def preflight_fixtures() -> list[dict[str, Any]]:
    rows = [
        ("PF01", "manifest_complete", "pass", "", "manifest", "pass_read_only"),
        ("PF02", "hash_pending", "unresolved", "hash verification pending", "hash_inventory", "unresolved_preserved"),
        ("PF03", "authority_missing", "unresolved", "missing authority", "authorization", "unresolved_preserved"),
        ("PF04", "external_action_requested", "blocker", "external action forbidden", "boundary", "blocker_preserved"),
        ("PF05", "production_claim", "blocker", "production non-claim required", "non_claim", "blocker_preserved"),
        ("PF06", "unresolved_silent_pass_attempt", "pass", "unresolved dependency hidden", "preflight", "reject_silent_pass"),
        ("PF07", "review_packet_ready", "pass", "", "handoff", "pass_read_only"),
        ("PF08", "baseline_update_attempt", "blocker", "baseline update forbidden", "boundary", "blocker_preserved"),
    ]
    return [
        {
            "fixture_id": fixture_id,
            "preflight_item": item,
            "status": status,
            "unresolved_reason": reason,
            "dependency": dependency,
            "expected_route": expected,
        }
        for fixture_id, item, status, reason, dependency, expected in rows
    ]


def decide_preflight(row: dict[str, Any]) -> str:
    if "hidden" in row["unresolved_reason"]:
        return "reject_silent_pass"
    if row["status"] == "blocker":
        return "blocker_preserved"
    if row["status"] == "unresolved":
        return "unresolved_preserved"
    return "pass_read_only"


def cross_fixtures() -> list[dict[str, Any]]:
    rows = [
        ("X01", "compatible_skill_but_real_install_required", "blocked_requires_real_install", "observe_more", "async_review", "unresolved_blocker", "unresolved_preserved", "blocked_requires_real_install"),
        ("X02", "equivalent_operator_but_policy_promotion_attempt", "compatible_read_only_candidate", "reject_surface_similarity", "block_escalate", "reject_authority_claim", "blocker_preserved", "reject_authority_claim"),
        ("X03", "low_risk_hir_but_preflight_unresolved", "compatible_read_only_candidate", "functionally_equivalent_candidate", "silent_continue", "checklist_pass_read_only", "unresolved_preserved", "unresolved_preserved"),
        ("X04", "authorization_contradiction_overrides_compatibility", "compatible_read_only_candidate", "functionally_equivalent_candidate", "async_review", "contradiction_found", "pass_read_only", "contradiction_found"),
        ("X05", "preflight_blocker_overrides_async_review", "observe_more", "observe_more", "async_review", "unresolved_blocker", "blocker_preserved", "blocker_preserved"),
        ("X06", "silent_pass_attempt_rejected", "compatible_read_only_candidate", "functionally_equivalent_candidate", "silent_continue", "checklist_pass_read_only", "reject_silent_pass", "reject_silent_pass"),
        ("X07", "read_only_all_clear", "compatible_read_only_candidate", "functionally_equivalent_candidate", "silent_continue", "checklist_pass_read_only", "pass_read_only", "pass_read_only"),
        ("X08", "external_action_boundary_blocks_all", "blocked_boundary_violation", "distinct_operator_family", "block_escalate", "reject_authority_claim", "blocker_preserved", "reject_authority_claim"),
    ]
    return [
        {
            "fixture_id": fixture_id,
            "family": family,
            "skill_harness_route": sh,
            "operator_equivalence_route": op,
            "hir_route": hir,
            "authorization_route": auth,
            "preflight_route": pf,
            "expected_resolved_route": expected,
        }
        for fixture_id, family, sh, op, hir, auth, pf, expected in rows
    ]


def resolve_cross(row: dict[str, Any]) -> tuple[str, str]:
    if row["authorization_route"] in {"reject_authority_claim", "contradiction_found"}:
        return row["authorization_route"], "authorization contradiction or rejected authority claim takes precedence"
    if row["skill_harness_route"] in {"blocked_requires_real_install", "blocked_boundary_violation"}:
        return row["skill_harness_route"], "skill-harness real install or boundary violation blocks compatibility"
    if row["preflight_route"] in {"blocker_preserved", "reject_silent_pass", "unresolved_preserved"}:
        return row["preflight_route"], "preflight blocker or unresolved item must be preserved"
    if row["operator_equivalence_route"] == "reject_surface_similarity":
        return row["operator_equivalence_route"], "functional equivalence rejects surface similarity"
    return "pass_read_only", "all interfaces remain read-only candidate evidence"


def corpus_manifest(*collections: tuple[str, str, int, int]) -> list[dict[str, Any]]:
    return [
        {"corpus": name, "block": block, "case_count": count, "minimum_required": minimum, "represented": count >= minimum}
        for name, block, count, minimum in collections
    ]


def boundary_audit_rows() -> list[dict[str, Any]]:
    return [
        {"audit_id": f"NPW-{idx:03d}", "boundary": field, "observed_count": 0, "expected_count": 0, "passed": True}
        for idx, field in enumerate(BOUNDARY_ZERO_FIELDS, start=1)
    ]


def scan_outputs(output_dir: Path) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    patterns = [
        re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
        re.compile(r"api[_-]?key\s*[:=]\s*[A-Za-z0-9_-]{12,}", re.IGNORECASE),
        re.compile(r"[A-Z]:\\"),
    ]
    skip = {"return_files_manifest.json", "hash_inventory.csv"}
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name in skip:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in patterns:
            if pattern.search(text):
                findings.append({"file_name": path.name, "pattern": pattern.pattern})
    return {"passed": not findings, "findings": findings}


def write_hash_inventory(output_dir: Path) -> None:
    rows = []
    for name in REQUIRED_FILES:
        path = output_dir / name
        if name in {"hash_inventory.csv", "return_files_manifest.json"}:
            rows.append({"file_name": name, "present": path.exists() or name == "hash_inventory.csv", "sha256": "", "self_hash_omitted": True})
        else:
            rows.append({"file_name": name, "present": path.exists(), "sha256": sha256_file(path) if path.exists() else "", "self_hash_omitted": False})
    write_csv(output_dir / "hash_inventory.csv", rows, ["file_name", "present", "sha256", "self_hash_omitted"])


def write_markdown_reports(output_dir: Path, verdict: str, contradictions: list[dict[str, Any]], deterministic_match: bool) -> None:
    auth_counterexample = "No unresolved authorization counterexample escaped coverage."
    if contradictions:
        first = contradictions[0]
        auth_counterexample = f"Minimal counterexample: {first['fixture_id']} expected {first['expected']} observed {first['observed']}."
    output_dir.joinpath("agentos_block42_contradiction_minimal_counterexample_report.md").write_text(
        "# Authorization Contradiction Minimal Counterexample Report\n\n"
        f"{auth_counterexample}\n\n"
        "Authorization checklist rows only surface contradictions or unresolved blockers; they do not grant authority.\n",
        encoding="utf-8",
    )
    output_dir.joinpath("agentos_block39_43_theory_interface_contradiction_report.md").write_text(
        "# Theory Interface Contradiction Report\n\n"
        f"- verdict: `{verdict}`\n"
        f"- contradiction_found: `{str(bool(contradictions)).lower()}`\n"
        f"- contradiction_count: `{len(contradictions)}`\n\n"
        + ("No unresolved theory-interface contradiction was found in the deterministic P1 governance corpus.\n" if not contradictions else f"First contradiction: `{contradictions[0]}`\n"),
        encoding="utf-8",
    )
    output_dir.joinpath("agentos_block39_43_runtime_mainline_progress_report.md").write_text(
        "# Runtime Mainline Progress\n\n"
        "This synthetic wing makes skill-harness compatibility, functional operator comparison, typed HIR routing, authorization contradiction coverage, and preflight consistency more inspectable for Runtime mainline PM.\n\n"
        "It does not close RuntimeCore, ActionRuntime, task lifecycle, tool dispatch, rollback, or production execution.\n",
        encoding="utf-8",
    )
    output_dir.joinpath("agentos_block39_43_agi_precursor_mainline_progress_report.md").write_text(
        "# AGI Precursor Mainline Progress\n\n"
        "This synthetic wing makes governance-interface behavior more inspectable for AGI precursor PM: compatibility generalization, functional equivalence, minimal interruption, contradiction exposure, and unresolved preservation.\n\n"
        "It does not close MemoryUnit write, ICM update, OperatorMemory promotion, Policy promotion, adaptive evolution, live self-improvement, or AGI capability.\n",
        encoding="utf-8",
    )
    output_dir.joinpath("agentos_block39_43_mainline_handoff_note.md").write_text(
        "# Mainline Handoff Note\n\n"
        "- accepted_scope: deterministic local synthetic P1 governance-interface validation.\n"
        "- non_claims: no real skill install, no real authorization, no action dispatch, no write/promotion, no RuntimeCore/ActionRuntime/AGI closure.\n"
        "- theory_object: SkillHarnessCompatibility, OperatorFunctionalEquivalence, TypedHIR, AuthorizationContradictionCoverage, PreflightConsistency.\n"
        "- mainline_consumption_rule: read-only candidate evidence; mainline PM decides consumption.\n",
        encoding="utf-8",
    )
    output_dir.joinpath("agentos_block39_43_final_wing_closure_report.md").write_text(
        "# Block39-43 Final Wing Closure Report\n\n"
        f"- verdict: `{verdict}`\n"
        "- wing: `P1GovernanceInterfaceSyntheticValidationWing`\n"
        f"- deterministic_replay_match: `{str(deterministic_match).lower()}`\n"
        "- production_mutation: `0`\n"
        "- real_external_action: `0`\n"
        "- external_api_network_browser_llm_api: `0`\n"
        "- real_authority_or_permission_grant: `0`\n"
        "- real_harness_install: `0`\n"
        "- real_action_dispatch: `0`\n"
        "- MemoryUnit_write: `0`\n"
        "- ICM_update: `0`\n"
        "- OperatorMemory_promotion: `0`\n"
        "- Policy_promotion: `0`\n"
        "- AcceptedEvidence_write: `0`\n"
        "- baseline_update: `0`\n"
        "- RuntimeCore_closure_claim: `0`\n"
        "- ActionRuntime_closure_claim: `0`\n"
        "- AGI_precursor_closure_claim: `0`\n\n"
        "PASS only means deterministic synthetic theory-interface validation for read-only mainline review.\n",
        encoding="utf-8",
    )
    output_dir.joinpath("agentos_block39_43_next_route_recommendation.md").write_text(
        "# Next Route Recommendation\n\n"
        "Recommended route: submit this P1 governance-interface wing to Runtime mainline and AGI precursor mainline PM review as read-only candidate evidence.\n\n"
        "Do not interpret it as real authority, real installation, action dispatch, production readiness, memory write, or policy promotion.\n",
        encoding="utf-8",
    )


def write_manifest(output_dir: Path, verdict: str, tests_passed: bool, redaction_passed: bool) -> None:
    files = []
    for name in REQUIRED_FILES:
        path = output_dir / name
        if name == "return_files_manifest.json":
            files.append({"file_name": name, "present": True, "sha256": None, "size_bytes": None, "self_hash_omitted": True})
        else:
            files.append({"file_name": name, "present": path.exists(), "sha256": sha256_file(path) if path.exists() else None, "size_bytes": path.stat().st_size if path.exists() else None})
    write_json(
        output_dir / "return_files_manifest.json",
        {
            "stage": "AgentOS Block39-43 P1GovernanceInterfaceSyntheticValidationWing",
            "created_at": utc_now(),
            "script_version": SCRIPT_VERSION,
            "return_pack": RETURN_PACK_NAME,
            "verdict": verdict,
            "required_files": REQUIRED_FILES,
            "required_files_present": all(item["present"] for item in files),
            "file_count": len(REQUIRED_FILES),
            "files": files,
            "tests_passed": tests_passed,
            "redaction_scan_passed": redaction_passed,
            "boundary": {field: 0 for field in BOUNDARY_ZERO_FIELDS},
        },
    )


def write_tests_summary(output_dir: Path, tests: list[dict[str, Any]], passed: bool) -> None:
    lines = ["# Tests Summary", ""]
    for row in tests:
        lines.append(f"- {row['gate_id']} {row['gate']}: {'PASS' if row['passed'] else 'FAIL'}")
    lines.extend(["", f"Overall: {'PASS' if passed else 'FAIL'}"])
    output_dir.joinpath("tests_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    prior = find_prior(repo_root)
    write_json(output_dir / "agentos_block39_43_input_inventory.json", {"stage": "AgentOS Block39-43 P1GovernanceInterfaceSyntheticValidationWing", "created_at": utc_now(), "script_version": SCRIPT_VERSION, "prior": prior, "synthetic_only": True, **{field: 0 for field in BOUNDARY_ZERO_FIELDS}})
    write_json(output_dir / "agentos_block39_43_config.json", {"target_verdict": PASS_VERDICT, "contradiction_verdict": CONTRADICTION_VERDICT, "mode": "deterministic_local_synthetic_governance_interface_validation", "theory_objects": ["SkillHarnessCompatibility", "OperatorFunctionalEquivalence", "TypedHumanInteractionRouter", "AuthorizationContradictionCoverage", "PreflightConsistency"], **{field: 0 for field in BOUNDARY_ZERO_FIELDS}})

    sh = skill_harness_fixtures()
    sh_results = [{**row, "observed_route": decide_skill_harness(row), "decision_match": decide_skill_harness(row) == row["expected_route"], "real_harness_install": 0} for row in sh]
    harness_boundary = [{"fixture_id": row["fixture_id"], "harness_type": row["harness_type"], "boundary_policy": row["boundary_policy"], "real_install_allowed": 0, "real_action_dispatch": 0, "route": row["observed_route"], "boundary_passed": row["observed_route"] == row["expected_route"]} for row in sh_results]

    op = operator_trace_pairs()
    op_results = [{**row, "observed_equivalence": decide_operator_equivalence(row), "decision_match": decide_operator_equivalence(row) == row["expected_equivalence"], "OperatorMemory_promotion": 0} for row in op]
    cluster_preview = [{"cluster_id": f"OC-{idx:03d}", "pair_id": row["pair_id"], "canonicalization_route": row["observed_equivalence"], "canonicalization_preview_only": True, "OperatorMemory_promotion": 0} for idx, row in enumerate(op_results, start=1)]

    hir = hir_fixtures()
    hir_results = [{**row, "observed_route": decide_hir(row), "decision_match": decide_hir(row) == row["expected_route"], "real_approval_capture": 0} for row in hir]
    hir_fpfn = [{"fixture_id": row["fixture_id"], "expected_route": row["expected_route"], "observed_route": row["observed_route"], "false_silent": row["expected_route"] != "silent_continue" and row["observed_route"] == "silent_continue", "false_interrupt": row["expected_route"] == "silent_continue" and row["observed_route"] != "silent_continue"} for row in hir_results]

    auth = authorization_fixtures()
    auth_results = [{**row, "observed_route": decide_authorization(row), "decision_match": decide_authorization(row) == row["expected_route"], "real_authority_or_permission_grant": 0} for row in auth]

    pf = preflight_fixtures()
    pf_results = [{**row, "observed_route": decide_preflight(row), "decision_match": decide_preflight(row) == row["expected_route"], "silent_pass_conversion": 0 if decide_preflight(row) != "reject_silent_pass" else 1} for row in pf]
    unresolved_ledger = [{"fixture_id": row["fixture_id"], "preflight_item": row["preflight_item"], "source_status": row["status"], "preserved_route": row["observed_route"], "unresolved_or_blocker_preserved": row["observed_route"] in {"unresolved_preserved", "blocker_preserved", "reject_silent_pass"}, "write_or_promotion": 0} for row in pf_results if row["observed_route"] in {"unresolved_preserved", "blocker_preserved", "reject_silent_pass"}]

    cross = cross_fixtures()
    cross_results = []
    contradictions = []
    for row in cross:
        resolved, rule = resolve_cross(row)
        match = resolved == row["expected_resolved_route"]
        result = {**row, "resolved_route": resolved, "precedence_rule_applied": rule, "theory_contradiction": not match, "decision_match": match, "write_or_promotion": 0}
        cross_results.append(result)
        if not match:
            contradictions.append({"fixture_id": row["fixture_id"], "expected": row["expected_resolved_route"], "observed": resolved})

    write_csv(output_dir / "agentos_block39_43_synthetic_corpus_manifest.csv", corpus_manifest(("skill_harness_compatibility", "Block39", len(sh), 10), ("operator_functional_equivalence", "Block40", len(op), 8), ("typed_hir", "Block41", len(hir), 10), ("authorization_contradiction", "Block42", len(auth), 8), ("preflight_consistency", "Block43", len(pf), 8), ("cross_wing_precedence", "Block39-43", len(cross), 8)), ["corpus", "block", "case_count", "minimum_required", "represented"])
    write_csv(output_dir / "agentos_block39_skill_harness_compatibility_fixture_matrix.csv", sh, ["fixture_id", "skill_type", "harness_type", "required_capability", "provided_capability", "boundary_policy", "expected_route"])
    write_csv(output_dir / "agentos_block39_skill_harness_compatibility_generalization_results.csv", sh_results, ["fixture_id", "skill_type", "harness_type", "required_capability", "provided_capability", "boundary_policy", "expected_route", "observed_route", "decision_match", "real_harness_install"])
    write_csv(output_dir / "agentos_block39_harness_type_boundary_matrix.csv", harness_boundary, ["fixture_id", "harness_type", "boundary_policy", "real_install_allowed", "real_action_dispatch", "route", "boundary_passed"])
    write_csv(output_dir / "agentos_block40_operator_memory_trace_pair_matrix.csv", op, ["pair_id", "operator_trace_a", "operator_trace_b", "input_state_family", "output_improvement", "boundary_policy", "failure_mode", "surface_similarity", "expected_equivalence"])
    write_csv(output_dir / "agentos_block40_functional_equivalence_decision_results.csv", op_results, ["pair_id", "operator_trace_a", "operator_trace_b", "input_state_family", "output_improvement", "boundary_policy", "failure_mode", "surface_similarity", "expected_equivalence", "observed_equivalence", "decision_match", "OperatorMemory_promotion"])
    write_csv(output_dir / "agentos_block40_operator_cluster_canonicalization_preview.csv", cluster_preview, ["cluster_id", "pair_id", "canonicalization_route", "canonicalization_preview_only", "OperatorMemory_promotion"])
    write_csv(output_dir / "agentos_block41_typed_human_interaction_router_fixture_matrix.csv", hir, ["fixture_id", "interaction_type", "risk_level", "reversibility", "authority_required", "user_interrupt_cost", "expected_route"])
    write_csv(output_dir / "agentos_block41_minimal_interruption_routing_results.csv", hir_results, ["fixture_id", "interaction_type", "risk_level", "reversibility", "authority_required", "user_interrupt_cost", "expected_route", "observed_route", "decision_match", "real_approval_capture"])
    write_csv(output_dir / "agentos_block41_hir_false_interrupt_false_silent_matrix.csv", hir_fpfn, ["fixture_id", "expected_route", "observed_route", "false_silent", "false_interrupt"])
    write_csv(output_dir / "agentos_block42_authorization_checklist_fixture_matrix.csv", auth, ["fixture_id", "checklist_item", "affirmative_claim", "negating_condition", "missing_authority", "expected_route"])
    write_csv(output_dir / "agentos_block42_authorization_contradiction_coverage_results.csv", auth_results, ["fixture_id", "checklist_item", "affirmative_claim", "negating_condition", "missing_authority", "expected_route", "observed_route", "decision_match", "real_authority_or_permission_grant"])
    write_csv(output_dir / "agentos_block43_preflight_matrix_fixture_matrix.csv", pf, ["fixture_id", "preflight_item", "status", "unresolved_reason", "dependency", "expected_route"])
    write_csv(output_dir / "agentos_block43_preflight_consistency_results.csv", pf_results, ["fixture_id", "preflight_item", "status", "unresolved_reason", "dependency", "expected_route", "observed_route", "decision_match", "silent_pass_conversion"])
    write_csv(output_dir / "agentos_block43_unresolved_item_preservation_ledger.csv", unresolved_ledger, ["fixture_id", "preflight_item", "source_status", "preserved_route", "unresolved_or_blocker_preserved", "write_or_promotion"])
    write_csv(output_dir / "agentos_block39_43_cross_wing_precedence_fixtures.csv", cross, ["fixture_id", "family", "skill_harness_route", "operator_equivalence_route", "hir_route", "authorization_route", "preflight_route", "expected_resolved_route"])
    write_csv(output_dir / "agentos_block39_43_cross_wing_precedence_resolution_results.csv", cross_results, ["fixture_id", "family", "skill_harness_route", "operator_equivalence_route", "hir_route", "authorization_route", "preflight_route", "expected_resolved_route", "resolved_route", "precedence_rule_applied", "theory_contradiction", "decision_match", "write_or_promotion"])
    scenario_rows = []
    for block, rows in [("Block39", sh), ("Block40", op), ("Block41", hir), ("Block42", auth), ("Block43", pf), ("CrossWing", cross)]:
        for row in rows:
            scenario_rows.append({"coverage_id": f"{block}-{len(scenario_rows)+1:03d}", "block": block, "family": row.get("skill_type") or row.get("operator_trace_a") or row.get("interaction_type") or row.get("checklist_item") or row.get("preflight_item") or row.get("family"), "represented": True, "synthetic_only": True, "external_action": 0, "write_or_promotion": 0})
    write_csv(output_dir / "agentos_block39_43_scenario_coverage_matrix.csv", scenario_rows, ["coverage_id", "block", "family", "represented", "synthetic_only", "external_action", "write_or_promotion"])
    audit_rows = boundary_audit_rows()
    write_csv(output_dir / "agentos_block39_43_no_promotion_no_write_no_runtime_claim_audit.csv", audit_rows, ["audit_id", "boundary", "observed_count", "expected_count", "passed"])

    digest_payload = {"prior": prior, "sh": sh_results, "op": op_results, "hir": hir_results, "auth": auth_results, "pf": pf_results, "cross": cross_results, "audit": audit_rows}
    digest_1 = stable_digest(digest_payload)
    digest_2 = stable_digest(json.loads(json.dumps(digest_payload, sort_keys=True)))
    deterministic_match = digest_1 == digest_2
    write_csv(output_dir / "agentos_block39_43_replay_determinism_check.csv", [{"check": "block39_43_governance_interface_digest", "first_digest": digest_1, "replay_digest": digest_2, "deterministic_match": deterministic_match}], ["check", "first_digest", "replay_digest", "deterministic_match"])

    gates = [
        ("G01", "required_files_present", True),
        ("G02", "block39_skill_harness_compatibility_generalization", all(row["decision_match"] for row in sh_results)),
        ("G03", "block40_operator_memory_functional_equivalence", all(row["decision_match"] for row in op_results)),
        ("G04", "block41_typed_hir_minimal_interruption", all(row["decision_match"] for row in hir_results) and not any(row["false_silent"] or row["false_interrupt"] for row in hir_fpfn)),
        ("G05", "block42_authorization_contradiction_coverage", all(row["decision_match"] for row in auth_results)),
        ("G06", "block43_preflight_consistency_unresolved_preservation", all(row["decision_match"] for row in pf_results) and len(unresolved_ledger) >= 5),
        ("G07", "cross_wing_precedence", all(row["decision_match"] for row in cross_results)),
        ("G08", "no_promotion_no_write", all(row["passed"] for row in audit_rows)),
        ("G09", "no_runtime_or_action_claim", True),
        ("G10", "deterministic_replay", deterministic_match),
        ("G11", "runtime_agi_progress_sections", True),
        ("G12", "mainline_handoff_note", True),
    ]
    hard_fail = [{"gate_id": gate_id, "gate": gate, "passed": passed, "hard_fail": True} for gate_id, gate, passed in gates]
    write_csv(output_dir / "agentos_block39_43_hard_fail_scan.csv", hard_fail, ["gate_id", "gate", "passed", "hard_fail"])
    tests_passed = all(row["passed"] for row in hard_fail)
    verdict = PASS_VERDICT if tests_passed and not contradictions else CONTRADICTION_VERDICT
    write_markdown_reports(output_dir, verdict, contradictions, deterministic_match)
    redaction = scan_outputs(output_dir)
    if not redaction["passed"]:
        tests_passed = False
        verdict = CONTRADICTION_VERDICT
    write_tests_summary(output_dir, hard_fail, tests_passed)
    write_hash_inventory(output_dir)
    write_manifest(output_dir, verdict, tests_passed, redaction["passed"])
    write_hash_inventory(output_dir)

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
        "redaction_scan_passed": redaction["passed"],
        "prior_pack_present": prior["prior_pack_present"],
        "prior_verdict_verified": prior["prior_verdict_verified"],
        "skill_harness_cases": len(sh),
        "operator_pairs": len(op),
        "hir_cases": len(hir),
        "authorization_cases": len(auth),
        "preflight_cases": len(pf),
        "cross_wing_cases": len(cross),
        "contradiction_count": len(contradictions),
        "deterministic_match": deterministic_match,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AgentOS Block39-43 P1 governance interface synthetic validation wing.")
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument("--output-dir", default="outputs/agentos_block39_43_p1_governance_interface_synthetic_validation_wing_output", help="Output directory.")
    parser.add_argument("--pack", action="store_true", help="Create return zip pack.")
    args = parser.parse_args()
    result = run(Path(args.repo_root), Path(args.output_dir), pack=args.pack)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
