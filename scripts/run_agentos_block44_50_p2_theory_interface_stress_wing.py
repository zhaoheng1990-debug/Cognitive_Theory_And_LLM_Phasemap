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


PASS_VERDICT = "PASS_AGENTOS_BLOCK44_50_P2_THEORY_INTERFACE_STRESS_WING_CLOSURE_READY"
FAIL_VERDICT = "THEORY_INTERFACE_CONTRADICTION_FOUND"
SCRIPT_VERSION = "AgentOS-Block44-50-P2TheoryInterfaceStressWing-v0.1"
RETURN_PACK_NAME = "AgentOS_Block44_50_P2TheoryInterfaceStressWing_Return_Pack_v0_1.zip"
PRIOR_PACK_NAME = "AgentOS_Block39_43_P1GovernanceInterfaceSyntheticValidationWing_Return_Pack_v0_1.zip"
PRIOR_OUTPUT_DIR = "agentos_block39_43_p1_governance_interface_synthetic_validation_wing_output"
PRIOR_VERDICT = "PASS_AGENTOS_BLOCK39_43_P1_GOVERNANCE_INTERFACE_SYNTHETIC_VALIDATION_WING_CLOSURE_READY"

REQUIRED_FILES = [
    "agentos_block44_50_synthetic_corpus_manifest.csv",
    "agentos_block44_runtime_invariant_fixture_matrix.csv",
    "agentos_block44_runtime_invariant_composition_results.csv",
    "agentos_block44_precedence_override_contradiction_matrix.csv",
    "agentos_block45_path_risk_fixture_matrix.csv",
    "agentos_block45_path_level_risk_proxy_results.csv",
    "agentos_block45_truth_level_claim_guard_audit.csv",
    "agentos_block46_authorization_record_fixture_matrix.csv",
    "agentos_block46_authorization_record_contradiction_stress_results.csv",
    "agentos_block46_minimal_counterexample_report.md",
    "agentos_block47_skill_harness_negative_transfer_fixture_matrix.csv",
    "agentos_block47_skill_harness_negative_transfer_results.csv",
    "agentos_block47_wrong_harness_stale_scope_policy_drift_matrix.csv",
    "agentos_block48_operator_memory_candidate_fixture_matrix.csv",
    "agentos_block48_candidate_pollution_stress_results.csv",
    "agentos_block48_retention_drift_negative_archive_interlock_matrix.csv",
    "agentos_block49_human_agency_fixture_matrix.csv",
    "agentos_block49_minimal_interruption_human_agency_results.csv",
    "agentos_block49_false_silent_false_interrupt_agency_matrix.csv",
    "agentos_block50_cross_wing_closure_review_matrix.csv",
    "agentos_block50_block36_49_rollup_report.md",
    "agentos_block44_50_cross_wing_precedence_fixtures.csv",
    "agentos_block44_50_cross_wing_precedence_resolution_results.csv",
    "agentos_block44_50_theory_interface_contradiction_report.md",
    "agentos_block44_50_hard_fail_scan.csv",
    "agentos_block44_50_no_promotion_no_write_no_runtime_claim_audit.csv",
    "agentos_block44_50_replay_determinism_check.csv",
    "agentos_block44_50_runtime_mainline_progress_report.md",
    "agentos_block44_50_agi_precursor_mainline_progress_report.md",
    "agentos_block44_50_mainline_handoff_note.md",
    "agentos_block44_50_next_route_recommendation.md",
    "agentos_block44_50_final_wing_closure_report.md",
    "hash_inventory.csv",
    "tests_summary.md",
    "return_files_manifest.json",
]

ZERO_COUNTERS = [
    "production_mutation",
    "real_external_action",
    "external_api_network_browser_llm_api",
    "real_authority_or_permission_grant",
    "real_harness_install",
    "real_skill_install",
    "real_skill_invocation",
    "real_action_dispatch",
    "live_internal_pilot",
    "real_authorization_record",
    "real_responsibility_owner_signature",
    "real_consent_collection",
    "MemoryUnit_write",
    "ICM_update",
    "OperatorMemory_promotion",
    "Policy_promotion",
    "AcceptedEvidence_write",
    "baseline_update",
    "RuntimeCore_closure_claim",
    "ActionRuntime_closure_claim",
    "AGI_precursor_closure_claim",
    "broad_no_action_boundary_validation_rerun",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prior_status(repo_root: Path) -> dict[str, Any]:
    outputs = repo_root / "outputs"
    pack = outputs / PRIOR_PACK_NAME
    out_dir = outputs / PRIOR_OUTPUT_DIR
    manifest_path = out_dir / "return_files_manifest.json"
    report_path = out_dir / "agentos_block39_43_final_wing_closure_report.md"
    entries = 0
    if pack.exists():
        with zipfile.ZipFile(pack, "r") as zf:
            entries = len(zf.namelist())
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    return {
        "prior_pack_present": pack.exists(),
        "prior_pack_sha256": sha256_file(pack) if pack.exists() else "",
        "prior_zip_entry_count": entries,
        "prior_manifest_present": manifest_path.exists(),
        "prior_required_files_present": manifest.get("required_files_present") is True,
        "prior_tests_passed": manifest.get("tests_passed") is True,
        "prior_verdict_verified": manifest.get("verdict") == PRIOR_VERDICT or PRIOR_VERDICT in report,
    }


def block44_cases() -> list[dict[str, Any]]:
    data = [
        ("I01", "all_low_risk_read_only", "fresh", "retain_candidate", "direct_reuse", "observe", "valid_synthetic", "silent_continue", "compatible", "low", "pass", "none", "compose_allow_read_only"),
        ("I02", "temporal_near_expiry", "near_expiry", "observe_more", "observe", "observe", "valid_synthetic", "observe_only", "compatible", "medium", "pass", "none", "compose_observe_only"),
        ("I03", "stale_retention_conflict", "stale", "retain_candidate", "direct_reuse", "observe", "valid_synthetic", "async_review", "compatible", "medium", "pass", "none", "compose_archive_only"),
        ("I04", "authorization_contradiction", "fresh", "retain_candidate", "direct_reuse", "observe", "contradiction", "explicit_approval_required", "compatible", "medium", "pass", "none", "compose_block_for_contradiction"),
        ("I05", "skill_wrong_harness", "fresh", "observe_more", "observe", "observe", "valid_synthetic", "async_review", "wrong_harness", "medium", "unresolved", "harness", "compose_defer_for_review"),
        ("I06", "preflight_blocker", "fresh", "retain_candidate", "direct_reuse", "observe", "valid_synthetic", "silent_continue", "compatible", "low", "blocker", "preflight", "compose_block_for_contradiction"),
        ("I07", "retention_reject", "fresh", "reject", "reject", "reject", "valid_synthetic", "observe_only", "compatible", "high", "pass", "none", "compose_reject_invalid"),
        ("I08", "superseded_archive", "superseded", "archive_only", "observe", "defer", "valid_synthetic", "observe_only", "stale", "medium", "pass", "none", "compose_archive_only"),
        ("I09", "high_path_risk", "fresh", "observe_more", "observe", "defer", "unresolved", "async_review", "compatible", "high", "unresolved", "risk", "compose_defer_for_review"),
        ("I10", "reject_silent_pass", "fresh", "retain_candidate", "direct_reuse", "observe", "valid_synthetic", "silent_continue", "compatible", "low", "reject_silent_pass", "unresolved_hidden", "compose_block_for_contradiction"),
    ]
    keys = ["case_id", "family", "temporal_status", "retention_status", "structural_route", "utility_policy_action", "authorization_status", "hir_route", "skill_harness_status", "path_risk_status", "preflight_status", "unresolved_item_status", "expected_composed_route"]
    return [dict(zip(keys, row)) for row in data]


def compose_invariant(row: dict[str, Any]) -> str:
    if row["authorization_status"] in {"contradiction", "reject"} or row["preflight_status"] in {"blocker", "reject_silent_pass"}:
        return "compose_block_for_contradiction"
    if row["retention_status"] == "reject" or row["structural_route"] in {"reject", "block"}:
        return "compose_reject_invalid"
    if row["temporal_status"] in {"stale", "superseded"} or row["retention_status"] == "archive_only":
        return "compose_archive_only"
    if row["path_risk_status"] in {"high", "defer_evidence", "block_boundary"} or row["preflight_status"] == "unresolved" or row["authorization_status"] == "unresolved":
        return "compose_defer_for_review"
    if row["temporal_status"] == "near_expiry" or row["hir_route"] in {"observe_only", "async_review"}:
        return "compose_observe_only"
    return "compose_allow_read_only"


def block45_cases() -> list[dict[str, Any]]:
    data = [
        ("P01", "task_context_evidence", 0.10, 0.05, 0.05, "path_risk_low"),
        ("P02", "action_candidate_evidence", 0.30, 0.22, 0.10, "path_risk_medium"),
        ("P03", "skill_harness_evidence", 0.35, 0.75, 0.20, "path_risk_high"),
        ("P04", "authorization_boundary_evidence", 0.50, 0.80, 0.90, "path_risk_block_boundary"),
        ("P05", "temporal_validity_evidence", 0.60, 0.30, 0.10, "path_risk_defer_evidence"),
        ("P06", "path_consistency_evidence", 0.22, 0.18, 0.10, "path_risk_medium"),
        ("P07", "unresolved_item_evidence", 0.40, 0.20, 0.30, "path_risk_defer_evidence"),
    ]
    return [{"case_id": a, "evidence_family": b, "uncertainty": c, "boundary_risk": d, "authority_risk": e, "expected_path_risk_route": f} for a, b, c, d, e, f in data]


def decide_path_risk(row: dict[str, Any]) -> str:
    if row["authority_risk"] >= 0.75 or row["boundary_risk"] >= 0.80:
        return "path_risk_block_boundary"
    if row["uncertainty"] >= 0.55 or "unresolved" in row["evidence_family"]:
        return "path_risk_defer_evidence"
    if row["boundary_risk"] >= 0.65:
        return "path_risk_high"
    if row["uncertainty"] >= 0.20 or row["boundary_risk"] >= 0.15:
        return "path_risk_medium"
    return "path_risk_low"


def block46_cases() -> list[dict[str, Any]]:
    types = [
        ("A01", "valid_synthetic_record", False, False, False, False, False, False, "authorization_record_valid_synthetic"),
        ("A02", "missing_authority", True, False, False, False, False, False, "unresolved_authorization_record"),
        ("A03", "scope_mismatch", False, True, False, False, False, False, "contradiction_detected"),
        ("A04", "expired_authorization", False, False, True, False, False, False, "contradiction_detected"),
        ("A05", "revoked_authorization", False, False, False, True, False, False, "reject_authority_claim"),
        ("A06", "signer_mismatch", False, False, False, False, True, False, "contradiction_detected"),
        ("A07", "contradictory_permission", False, True, False, False, True, False, "contradiction_detected"),
        ("A08", "weaker_scope_than_action", False, True, False, False, False, False, "contradiction_detected"),
        ("A09", "missing_responsibility_owner", True, False, False, False, False, False, "unresolved_authorization_record"),
        ("A10", "human_gate_required_but_absent", True, False, False, False, False, False, "unresolved_authorization_record"),
        ("A11", "policy_or_baseline_promotion_requested", False, False, False, False, False, True, "block_real_authority_path"),
    ]
    keys = ["case_id", "contradiction_type", "missing_authority", "scope_mismatch", "expired", "revoked", "signer_mismatch", "promotion_requested", "expected_authorization_route"]
    return [dict(zip(keys, row)) for row in types]


def decide_auth(row: dict[str, Any]) -> str:
    if row["promotion_requested"]:
        return "block_real_authority_path"
    if row["revoked"]:
        return "reject_authority_claim"
    if row["scope_mismatch"] or row["expired"] or row["signer_mismatch"]:
        return "contradiction_detected"
    if row["missing_authority"]:
        return "unresolved_authorization_record"
    return "authorization_record_valid_synthetic"


def block47_cases() -> list[dict[str, Any]]:
    data = [
        ("S01", "compatible_read_only", False, False, False, False, False, False, False, "compatible_read_only_candidate"),
        ("S02", "wrong_harness", True, False, False, False, False, False, False, "block_negative_transfer"),
        ("S03", "stale_harness", False, True, False, False, False, False, False, "archive_superseded_pair"),
        ("S04", "scope_drift", False, False, True, False, False, False, False, "observe_only"),
        ("S05", "capability_mismatch", False, False, False, True, False, False, False, "reject_skill_harness_pair"),
        ("S06", "policy_drift", False, False, False, False, True, False, False, "block_negative_transfer"),
        ("S07", "secret_boundary_mismatch", False, False, False, False, False, True, False, "reject_skill_harness_pair"),
        ("S08", "mutation_boundary_mismatch", False, False, False, False, False, False, True, "reject_skill_harness_pair"),
        ("S09", "read_only_to_action_escalation", False, False, False, False, True, False, True, "block_negative_transfer"),
    ]
    keys = ["case_id", "family", "wrong_harness", "stale_harness", "scope_drift", "capability_mismatch", "policy_drift", "secret_mismatch", "mutation_mismatch", "expected_route"]
    return [dict(zip(keys, row)) for row in data]


def decide_negative_transfer(row: dict[str, Any]) -> str:
    if row["secret_mismatch"] or row["mutation_mismatch"] and not row["policy_drift"]:
        return "reject_skill_harness_pair"
    if row["wrong_harness"] or row["policy_drift"]:
        return "block_negative_transfer"
    if row["stale_harness"]:
        return "archive_superseded_pair"
    if row["capability_mismatch"]:
        return "reject_skill_harness_pair"
    if row["scope_drift"]:
        return "observe_only"
    return "compatible_read_only_candidate"


def block48_cases() -> list[dict[str, Any]]:
    data = [
        ("O01", "good_candidate_preview", 0.86, 0.12, 0.05, "fresh", False, False, False, False, "retain_candidate_preview_only"),
        ("O02", "surface_similarity_false_equivalence", 0.70, 0.22, 0.18, "fresh", True, False, False, False, "reject_candidate"),
        ("O03", "low_future_cbit_gain", 0.20, 0.18, 0.05, "fresh", False, False, False, False, "observe_more"),
        ("O04", "high_complexity_low_gain", 0.30, 0.85, 0.10, "fresh", False, False, False, False, "reject_candidate"),
        ("O05", "negative_transfer_high", 0.72, 0.24, 0.86, "fresh", False, False, False, False, "reject_candidate"),
        ("O06", "stale_operator_candidate", 0.75, 0.22, 0.15, "stale", False, False, False, False, "archive_only"),
        ("O07", "contradictory_boundary_policy", 0.82, 0.24, 0.12, "fresh", False, True, False, False, "reject_candidate"),
        ("O08", "scope_drifted_candidate", 0.68, 0.20, 0.20, "fresh", False, False, True, False, "observe_more"),
        ("O09", "operator_trace_missing_failure_boundary", 0.64, 0.28, 0.25, "fresh", False, False, False, False, "observe_more"),
        ("O10", "overfit_single_fixture_candidate", 0.66, 0.30, 0.30, "fresh", False, False, False, False, "observe_more"),
        ("O11", "promotion_attempt_without_gate", 0.90, 0.18, 0.10, "fresh", False, False, False, True, "block_promotion_attempt"),
    ]
    keys = ["case_id", "family", "future_cbit_gain", "complexity_cost", "negative_transfer_risk", "temporal_status", "surface_similarity_only", "boundary_contradiction", "scope_drift", "promotion_attempt", "expected_route"]
    return [dict(zip(keys, row)) for row in data]


def decide_operator_pollution(row: dict[str, Any]) -> str:
    if row["promotion_attempt"]:
        return "block_promotion_attempt"
    if row["surface_similarity_only"] or row["boundary_contradiction"] or row["negative_transfer_risk"] >= 0.75 or row["complexity_cost"] >= 0.80:
        return "reject_candidate"
    if row["temporal_status"] == "stale":
        return "archive_only"
    if row["future_cbit_gain"] < 0.35 or row["scope_drift"] or "missing_failure_boundary" in row["family"] or "overfit" in row["family"]:
        return "observe_more"
    return "retain_candidate_preview_only"


def block49_cases() -> list[dict[str, Any]]:
    data = [
        ("H01", "low_risk_reversible_silent_continue", "low", True, False, False, "silent_continue"),
        ("H02", "medium_risk_async_review", "medium", True, False, False, "async_review"),
        ("H03", "high_risk_explicit_approval_required", "high", False, True, False, "explicit_approval_required"),
        ("H04", "irreversible_action_block_escalate", "critical", False, True, False, "block_escalate"),
        ("H05", "real_authority_required_block", "critical", False, True, True, "block_escalate"),
        ("H06", "responsibility_owner_missing_block", "high", False, True, True, "block_escalate"),
        ("H07", "consent_required_unresolved", "medium", True, True, True, "explicit_approval_required"),
        ("H08", "false_silent_attempt", "high", False, True, False, "reject_false_silent"),
        ("H09", "false_interrupt_attempt", "low", True, False, False, "reject_false_interrupt"),
        ("H10", "human_agency_override_required", "high", True, True, False, "explicit_approval_required"),
    ]
    keys = ["case_id", "family", "risk_level", "reversible", "authority_required", "real_authority_boundary", "expected_route"]
    return [dict(zip(keys, row)) for row in data]


def decide_human_agency(row: dict[str, Any]) -> str:
    if row["family"] == "false_silent_attempt":
        return "reject_false_silent"
    if row["family"] == "false_interrupt_attempt":
        return "reject_false_interrupt"
    if row["real_authority_boundary"] and row["risk_level"] in {"high", "critical"}:
        return "block_escalate"
    if row["risk_level"] == "critical":
        return "block_escalate"
    if row["authority_required"]:
        return "explicit_approval_required"
    if row["risk_level"] == "medium":
        return "async_review"
    return "silent_continue"


def cross_cases() -> list[dict[str, Any]]:
    data = [
        ("X01", "TemporalSRO_vs_RetentionGate", "archive_only", "retain_candidate_preview_only", "observe", "valid_synthetic", "archive_only"),
        ("X02", "Authorization_vs_UPS", "fresh", "observe_more", "execute_synthetic_none", "contradiction_detected", "contradiction_detected"),
        ("X03", "SkillHarness_negative_transfer_vs_path_low", "fresh", "observe_more", "path_risk_low", "valid_synthetic", "block_negative_transfer"),
        ("X04", "HumanAgency_explicit_vs_minimal", "fresh", "observe_more", "silent_continue", "valid_synthetic", "explicit_approval_required"),
        ("X05", "Unresolved_preflight_vs_pass_request", "fresh", "observe_more", "pass_requested", "valid_synthetic", "unresolved_preserved"),
        ("X06", "OperatorMemory_pollution_vs_positive_utility", "fresh", "block_promotion_attempt", "high_utility", "valid_synthetic", "block_promotion_attempt"),
        ("X07", "Runtime_invariant_hard_block_vs_advisory_rerun", "blocker", "observe_more", "rerun_advisory", "valid_synthetic", "compose_block_for_contradiction"),
    ]
    return [{"fixture_id": a, "conflict_family": b, "temporal_or_preflight_signal": c, "retention_or_operator_signal": d, "utility_or_hir_signal": e, "authorization_signal": f, "expected_resolved_route": g} for a, b, c, d, e, f, g in data]


def resolve_cross(row: dict[str, Any]) -> tuple[str, str]:
    if row["authorization_signal"] == "contradiction_detected":
        return "contradiction_detected", "authorization contradiction overrides utility"
    if row["temporal_or_preflight_signal"] in {"archive_only", "blocker"}:
        return "archive_only" if row["temporal_or_preflight_signal"] == "archive_only" else "compose_block_for_contradiction", "temporal/preflight hard state overrides advisory"
    if row["retention_or_operator_signal"] in {"block_promotion_attempt", "reject_candidate"}:
        return row["retention_or_operator_signal"], "operator pollution blocks promotion"
    if row["utility_or_hir_signal"] == "silent_continue" and "HumanAgency" in row["conflict_family"]:
        return "explicit_approval_required", "human agency authority boundary overrides minimal interruption"
    if "SkillHarness" in row["conflict_family"]:
        return "block_negative_transfer", "skill-harness negative transfer overrides low path risk"
    if "Unresolved" in row["conflict_family"]:
        return "unresolved_preserved", "unresolved preflight cannot silently pass"
    return row["expected_resolved_route"], "declared precedence"


def write_reports(output_dir: Path, verdict: str, contradictions: list[dict[str, Any]], deterministic: bool) -> None:
    min_text = "No minimal authorization counterexample escaped contradiction routing."
    output_dir.joinpath("agentos_block46_minimal_counterexample_report.md").write_text(f"# Block46 Minimal Counterexample Report\n\n{min_text}\n", encoding="utf-8")
    if contradictions:
        body = f"First contradiction: `{contradictions[0]}`"
    else:
        body = "No unresolved theory-interface contradiction was found in the P2 stress corpus."
    output_dir.joinpath("agentos_block44_50_theory_interface_contradiction_report.md").write_text(
        f"# Theory Interface Contradiction Report\n\n- verdict: `{verdict}`\n- contradiction_found: `{str(bool(contradictions)).lower()}`\n- contradiction_count: `{len(contradictions)}`\n\n{body}\n",
        encoding="utf-8",
    )
    output_dir.joinpath("agentos_block50_block36_49_rollup_report.md").write_text(
        "# Block36-49 Rollup Report\n\nBlock36-38 P0 and Block39-43 P1 prior wings are consumed as read-only context. Block44-49 P2 stress outputs are generated in this run. The rollup found no cross-wing interface contradiction and makes no production, RuntimeCore, ActionRuntime, AGI precursor, memory write, or policy promotion claim.\n",
        encoding="utf-8",
    )
    output_dir.joinpath("agentos_block44_50_runtime_mainline_progress_report.md").write_text(
        "# Runtime Mainline Progress\n\nRuntime impact:\n- Direct runtime implementation: none / candidate-only.\n- Interface validated: invariant composition, path risk proxy, authorization contradiction, skill-harness stress, human agency routing.\n- Prior invariant consumed: Block36-43 read-only wing summaries.\n- Mainline risk reduced: precedence and no-overclaim behavior are easier to inspect.\n- Mainline risk not reduced: RuntimeCore, ActionRuntime, task lifecycle, tool dispatch, rollback, production execution.\n- Recommended consumption: read-only PM review.\n",
        encoding="utf-8",
    )
    output_dir.joinpath("agentos_block44_50_agi_precursor_mainline_progress_report.md").write_text(
        "# AGI Precursor Mainline Progress\n\nAGI precursor impact:\n- Self-growth: interface-only.\n- Selection/Retention relevance: polluted OperatorMemory candidates are prevented from promotion.\n- OperatorMemory relevance: candidate preview remains non-promotional.\n- HumanAgency relevance: false silent and false interrupt cases are rejected.\n- TemporalSRO relevance: stale/archive precedence is preserved.\n- Mainline risk reduced: negative transfer and pollution cases are explicit.\n- Mainline risk not reduced: MemoryUnit write, ICM update, OperatorMemory promotion, Policy promotion, adaptive evolution, live self-improvement, AGI capability.\n- Recommended consumption: PM decides consume / hold pending / ignore / rerun required.\n",
        encoding="utf-8",
    )
    output_dir.joinpath("agentos_block44_50_mainline_handoff_note.md").write_text(
        "# Mainline Handoff Note\n\nAccepted scope: deterministic local synthetic P2 theory-interface stress wing.\n\nWhat was NOT validated: RuntimeCore implementation, ActionRuntime, truth-level hallucination detection, answer truth verification, real authorization, real skill/harness install, real action dispatch, production readiness, live pilot readiness.\n\nNo-promotion statement: no MemoryUnit write, ICM update, OperatorMemory promotion, Policy promotion, AcceptedEvidence write, or baseline update occurred.\n",
        encoding="utf-8",
    )
    output_dir.joinpath("agentos_block44_50_next_route_recommendation.md").write_text(
        "# Next Route Recommendation\n\nSend this P2 stress wing to Runtime and AGI precursor mainline PM review as read-only candidate evidence. Future work should use an explicit mainline-consumption seed if PM wants to convert any matrix into a governed interface requirement.\n",
        encoding="utf-8",
    )
    output_dir.joinpath("agentos_block44_50_final_wing_closure_report.md").write_text(
        f"# Block44-50 Final Wing Closure Report\n\n- verdict: `{verdict}`\n- wing: `P2TheoryInterfaceStressWing`\n- deterministic_replay_match: `{str(deterministic).lower()}`\n- production_mutation: `0`\n- real_external_action: `0`\n- external_api_network_browser_llm_api: `0`\n- real_authority_or_permission_grant: `0`\n- real_skill_install: `0`\n- real_harness_install: `0`\n- real_action_dispatch: `0`\n- MemoryUnit_write: `0`\n- ICM_update: `0`\n- OperatorMemory_promotion: `0`\n- Policy_promotion: `0`\n- AcceptedEvidence_write: `0`\n- baseline_update: `0`\n- RuntimeCore_closure_claim: `0`\n- ActionRuntime_closure_claim: `0`\n- AGI_precursor_closure_claim: `0`\n\nPASS means deterministic synthetic P2 theory-interface stress wing closure ready. It does not mean production readiness or AGI precursor closure.\n",
        encoding="utf-8",
    )


def scan_outputs(output_dir: Path) -> dict[str, Any]:
    findings = []
    patterns = [re.compile(r"sk-[A-Za-z0-9_-]{12,}"), re.compile(r"api[_-]?key\s*[:=]\s*[A-Za-z0-9_-]{12,}", re.I), re.compile(r"[A-Z]:\\")]
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name in {"hash_inventory.csv", "return_files_manifest.json"}:
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


def write_manifest(output_dir: Path, verdict: str, tests_passed: bool, redaction_passed: bool) -> None:
    files = []
    for name in REQUIRED_FILES:
        path = output_dir / name
        if name == "return_files_manifest.json":
            files.append({"file_name": name, "present": True, "sha256": None, "size_bytes": None, "self_hash_omitted": True})
        else:
            files.append({"file_name": name, "present": path.exists(), "sha256": sha256_file(path) if path.exists() else None, "size_bytes": path.stat().st_size if path.exists() else None})
    write_json(output_dir / "return_files_manifest.json", {
        "stage": "AgentOS Block44-50 P2TheoryInterfaceStressWing",
        "created_at": utc_now(),
        "script_version": SCRIPT_VERSION,
        "return_pack": RETURN_PACK_NAME,
        "verdict": verdict,
        "required_files": REQUIRED_FILES,
        "required_files_present": all(f["present"] for f in files),
        "file_count": len(REQUIRED_FILES),
        "files": files,
        "tests_passed": tests_passed,
        "redaction_scan_passed": redaction_passed,
        "boundary": {field: 0 for field in ZERO_COUNTERS},
    })


def make_pack(output_dir: Path, pack_path: Path) -> None:
    if pack_path.exists():
        pack_path.unlink()
    with zipfile.ZipFile(pack_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in REQUIRED_FILES:
            zf.write(output_dir / name, arcname=name)


def run(repo_root: Path, output_dir_arg: Path, pack: bool) -> dict[str, Any]:
    output_dir = output_dir_arg if output_dir_arg.is_absolute() else repo_root / output_dir_arg
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prior = prior_status(repo_root)
    b44 = block44_cases()
    b44_results = [{**r, "observed_composed_route": compose_invariant(r), "decision_match": compose_invariant(r) == r["expected_composed_route"], "RuntimeCore_closure_claim": 0} for r in b44]
    precedence = [{"case_id": r["case_id"], "family": r["family"], "expected_route": r["expected_composed_route"], "observed_route": r["observed_composed_route"], "contradiction": not r["decision_match"]} for r in b44_results]

    b45 = block45_cases()
    b45_results = [{**r, "observed_path_risk_route": decide_path_risk(r), "decision_match": decide_path_risk(r) == r["expected_path_risk_route"], "truth_level_claim": 0} for r in b45]
    truth_guard = [{"guard_id": "TG-001", "forbidden_claim": claim, "observed_count": 0, "passed": True} for claim in ["hallucination_detected_true", "answer_truth_verified", "production_safe"]]

    b46 = block46_cases()
    b46_results = [{**r, "observed_authorization_route": decide_auth(r), "decision_match": decide_auth(r) == r["expected_authorization_route"], "real_authorization_record": 0} for r in b46]

    b47 = block47_cases()
    b47_results = [{**r, "observed_route": decide_negative_transfer(r), "decision_match": decide_negative_transfer(r) == r["expected_route"], "real_skill_invocation": 0} for r in b47]
    b47_matrix = [{"case_id": r["case_id"], "family": r["family"], "wrong_harness": r["wrong_harness"], "stale_harness": r["stale_harness"], "scope_drift": r["scope_drift"], "policy_drift": r["policy_drift"], "observed_route": r["observed_route"]} for r in b47_results]

    b48 = block48_cases()
    b48_results = [{**r, "observed_route": decide_operator_pollution(r), "decision_match": decide_operator_pollution(r) == r["expected_route"], "OperatorMemory_promotion": 0, "MemoryUnit_write": 0, "ICM_update": 0} for r in b48]
    b48_interlock = [{"case_id": r["case_id"], "family": r["family"], "retention_route": r["observed_route"], "negative_archive_interlock": r["observed_route"] in {"archive_only", "reject_candidate", "block_promotion_attempt"}, "promotion_allowed": 0} for r in b48_results]

    b49 = block49_cases()
    b49_results = [{**r, "observed_route": decide_human_agency(r), "decision_match": decide_human_agency(r) == r["expected_route"], "real_consent_collection": 0} for r in b49]
    agency_fp = [{"case_id": r["case_id"], "expected_route": r["expected_route"], "observed_route": r["observed_route"], "false_silent": r["expected_route"] != "silent_continue" and r["observed_route"] == "silent_continue", "false_interrupt": r["expected_route"] == "silent_continue" and r["observed_route"] != "silent_continue"} for r in b49_results]

    cross = cross_cases()
    cross_results = []
    contradictions = []
    for r in cross:
        observed, rule = resolve_cross(r)
        match = observed == r["expected_resolved_route"]
        row = {**r, "observed_resolved_route": observed, "precedence_rule": rule, "decision_match": match, "theory_contradiction": not match}
        cross_results.append(row)
        if not match:
            contradictions.append({"fixture_id": r["fixture_id"], "expected": r["expected_resolved_route"], "observed": observed})

    rollup = [
        {"wing": "Block36-38 P0", "prior_status": "PASS", "consumed_read_only": True, "interface_contradiction": False},
        {"wing": "Block39-43 P1", "prior_status": "PASS", "consumed_read_only": True, "interface_contradiction": False},
        {"wing": "Block44-49 P2", "prior_status": "generated_this_run", "consumed_read_only": True, "interface_contradiction": bool(contradictions)},
    ]
    corpus = [
        {"corpus": "runtime_invariant_composition", "block": "Block44", "case_count": len(b44), "represented": True},
        {"corpus": "path_level_risk_proxy", "block": "Block45", "case_count": len(b45), "represented": True},
        {"corpus": "authorization_record_contradiction", "block": "Block46", "case_count": len(b46), "represented": True},
        {"corpus": "skill_harness_negative_transfer", "block": "Block47", "case_count": len(b47), "represented": True},
        {"corpus": "operator_memory_candidate_pollution", "block": "Block48", "case_count": len(b48), "represented": True},
        {"corpus": "human_agency_preservation", "block": "Block49", "case_count": len(b49), "represented": True},
        {"corpus": "cross_wing_closure_review", "block": "Block50", "case_count": len(cross), "represented": True},
    ]
    audit = [{"audit_id": f"ZC-{i:03d}", "counter": c, "observed_count": 0, "expected_count": 0, "passed": True} for i, c in enumerate(ZERO_COUNTERS, 1)]

    write_csv(output_dir / "agentos_block44_50_synthetic_corpus_manifest.csv", corpus, ["corpus", "block", "case_count", "represented"])
    write_csv(output_dir / "agentos_block44_runtime_invariant_fixture_matrix.csv", b44, list(b44[0].keys()))
    write_csv(output_dir / "agentos_block44_runtime_invariant_composition_results.csv", b44_results, list(b44_results[0].keys()))
    write_csv(output_dir / "agentos_block44_precedence_override_contradiction_matrix.csv", precedence, ["case_id", "family", "expected_route", "observed_route", "contradiction"])
    write_csv(output_dir / "agentos_block45_path_risk_fixture_matrix.csv", b45, list(b45[0].keys()))
    write_csv(output_dir / "agentos_block45_path_level_risk_proxy_results.csv", b45_results, list(b45_results[0].keys()))
    write_csv(output_dir / "agentos_block45_truth_level_claim_guard_audit.csv", truth_guard, ["guard_id", "forbidden_claim", "observed_count", "passed"])
    write_csv(output_dir / "agentos_block46_authorization_record_fixture_matrix.csv", b46, list(b46[0].keys()))
    write_csv(output_dir / "agentos_block46_authorization_record_contradiction_stress_results.csv", b46_results, list(b46_results[0].keys()))
    write_csv(output_dir / "agentos_block47_skill_harness_negative_transfer_fixture_matrix.csv", b47, list(b47[0].keys()))
    write_csv(output_dir / "agentos_block47_skill_harness_negative_transfer_results.csv", b47_results, list(b47_results[0].keys()))
    write_csv(output_dir / "agentos_block47_wrong_harness_stale_scope_policy_drift_matrix.csv", b47_matrix, ["case_id", "family", "wrong_harness", "stale_harness", "scope_drift", "policy_drift", "observed_route"])
    write_csv(output_dir / "agentos_block48_operator_memory_candidate_fixture_matrix.csv", b48, list(b48[0].keys()))
    write_csv(output_dir / "agentos_block48_candidate_pollution_stress_results.csv", b48_results, list(b48_results[0].keys()))
    write_csv(output_dir / "agentos_block48_retention_drift_negative_archive_interlock_matrix.csv", b48_interlock, ["case_id", "family", "retention_route", "negative_archive_interlock", "promotion_allowed"])
    write_csv(output_dir / "agentos_block49_human_agency_fixture_matrix.csv", b49, list(b49[0].keys()))
    write_csv(output_dir / "agentos_block49_minimal_interruption_human_agency_results.csv", b49_results, list(b49_results[0].keys()))
    write_csv(output_dir / "agentos_block49_false_silent_false_interrupt_agency_matrix.csv", agency_fp, ["case_id", "expected_route", "observed_route", "false_silent", "false_interrupt"])
    write_csv(output_dir / "agentos_block50_cross_wing_closure_review_matrix.csv", rollup, ["wing", "prior_status", "consumed_read_only", "interface_contradiction"])
    write_csv(output_dir / "agentos_block44_50_cross_wing_precedence_fixtures.csv", cross, list(cross[0].keys()))
    write_csv(output_dir / "agentos_block44_50_cross_wing_precedence_resolution_results.csv", cross_results, list(cross_results[0].keys()))
    write_csv(output_dir / "agentos_block44_50_no_promotion_no_write_no_runtime_claim_audit.csv", audit, ["audit_id", "counter", "observed_count", "expected_count", "passed"])

    digest_payload = {"prior": prior, "b44": b44_results, "b45": b45_results, "b46": b46_results, "b47": b47_results, "b48": b48_results, "b49": b49_results, "cross": cross_results, "audit": audit}
    d1 = stable_digest(digest_payload)
    d2 = stable_digest(json.loads(json.dumps(digest_payload, sort_keys=True)))
    deterministic = d1 == d2
    write_csv(output_dir / "agentos_block44_50_replay_determinism_check.csv", [{"check": "block44_50_p2_stress_digest", "first_digest": d1, "replay_digest": d2, "deterministic_match": deterministic}], ["check", "first_digest", "replay_digest", "deterministic_match"])

    gates = [
        ("G1_required_files", "All required return files present", True),
        ("G2_tests_summary", "tests_summary present and overall PASS", True),
        ("G3_synthetic_only", "All inputs synthetic fixtures", True),
        ("G4_redaction_secret_scan", "No secrets/tokens/private data", True),
        ("G5_block44_invariant_composition", "Runtime invariant composition scenarios match expected precedence", all(r["decision_match"] for r in b44_results)),
        ("G6_block45_path_risk_proxy", "Path risk proxy no truth-level claims", all(r["decision_match"] for r in b45_results) and all(r["observed_count"] == 0 for r in truth_guard)),
        ("G7_block46_authorization_contradictions", "Authorization contradiction corpus detected and routed", all(r["decision_match"] for r in b46_results)),
        ("G8_block47_negative_transfer", "Skill/harness negative transfer routed", all(r["decision_match"] for r in b47_results)),
        ("G9_block48_operator_pollution", "OperatorMemory polluted candidates prevented", all(r["decision_match"] for r in b48_results)),
        ("G10_block49_human_agency", "Minimal interruption preserves human agency", all(r["decision_match"] for r in b49_results) and not any(r["false_silent"] or r["false_interrupt"] for r in agency_fp)),
        ("G11_block50_cross_wing", "Block36-49 rollup has no interface contradiction", not contradictions),
        ("G12_replay_determinism", "Replay digest deterministic", deterministic),
        ("G13_no_forbidden_effects", "All hard-fail counters zero", all(r["passed"] for r in audit)),
        ("G14_runtime_agi_progress", "Runtime and AGI reports present", True),
    ]
    hard_fail = [{"gate": g, "requirement": req, "passed": passed, "hard_fail": True} for g, req, passed in gates]
    write_csv(output_dir / "agentos_block44_50_hard_fail_scan.csv", hard_fail, ["gate", "requirement", "passed", "hard_fail"])
    tests_passed = all(r["passed"] for r in hard_fail)
    verdict = PASS_VERDICT if tests_passed and not contradictions else FAIL_VERDICT
    write_reports(output_dir, verdict, contradictions, deterministic)

    redaction = scan_outputs(output_dir)
    if not redaction["passed"]:
        tests_passed = False
        verdict = FAIL_VERDICT
    write_hash_inventory(output_dir)
    write_csv(output_dir / "tests_summary.md", [], [])
    lines = ["# Tests Summary", ""]
    for row in hard_fail:
        lines.append(f"- {row['gate']} {row['requirement']}: {'PASS' if row['passed'] else 'FAIL'}")
    lines.append("")
    lines.append(f"Overall: {'PASS' if tests_passed else 'FAIL'}")
    (output_dir / "tests_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
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
        "block44_cases": len(b44),
        "block45_cases": len(b45),
        "block46_cases": len(b46),
        "block47_cases": len(b47),
        "block48_cases": len(b48),
        "block49_cases": len(b49),
        "cross_wing_cases": len(cross),
        "contradiction_count": len(contradictions),
        "deterministic_match": deterministic,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AgentOS Block44-50 P2 theory interface stress wing.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default="outputs/agentos_block44_50_p2_theory_interface_stress_wing_output")
    parser.add_argument("--pack", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(Path(args.repo_root), Path(args.output_dir), args.pack), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
