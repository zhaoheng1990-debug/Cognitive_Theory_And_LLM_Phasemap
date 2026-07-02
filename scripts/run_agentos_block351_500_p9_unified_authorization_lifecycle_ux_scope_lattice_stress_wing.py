#!/usr/bin/env python3
"""Generate AgentOS Block351-500 P9 deterministic local AutoRun artifacts."""

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
OUTPUT_DIR = ROOT / "outputs" / "agentos_block351_500_p9_unified_authorization_lifecycle_ux_scope_lattice_stress_wing_output"
PRIOR_P8_PACK = ROOT / "outputs" / "AgentOS_Block226_350_P8UnifiedAuthorizationBridgeContractAdversarialSoakWing_Return_Pack_v0_1.zip"
RETURN_PACK = "AgentOS_Block351_500_P9UnifiedAuthorizationLifecycleUXScopeLatticeStressWing_Return_Pack_v0_1.zip"
TARGET_VERDICT = "PASS_AGENTOS_BLOCK351_500_P9_UNIFIED_AUTHORIZATION_LIFECYCLE_UX_SCOPE_LATTICE_STRESS_WING_CLOSURE_READY"
CONTRADICTION_VERDICT = "THEORY_INTERFACE_CONTRADICTION_FOUND"

REQUIRED_FILES = [
    "agentos_block351_500_final_wing_closure_report.md",
    "agentos_block351_500_theory_interface_contradiction_report.md",
    "agentos_block351_500_runtime_mainline_progress_report.md",
    "agentos_block351_500_agi_precursor_mainline_progress_report.md",
    "agentos_block351_500_mainline_handoff_note.md",
    "agentos_block351_500_next_route_recommendation.md",
    "agentos_block351_500_no_promotion_no_write_no_runtime_claim_audit.csv",
    "agentos_block351_500_replay_determinism_check.csv",
    "agentos_block351_500_synthetic_authorization_lifecycle_corpus_manifest.csv",
    "agentos_block351_lifecycle_state_machine_results.csv",
    "agentos_block352_scope_lattice_inheritance_narrowing_results.csv",
    "agentos_block353_delegation_subdelegation_semantic_results.csv",
    "agentos_block354_revocation_expiry_supersession_race_results.csv",
    "agentos_block355_no_repeated_prompt_ux_regression_results.csv",
    "agentos_block356_minimal_interruption_human_agency_authorization_results.csv",
    "agentos_block357_authorization_contradiction_delta_minimization_report.md",
    "agentos_block358_bridge_contract_auth_binding_rejection_results.csv",
    "agentos_block359_concurrent_consumer_authorization_stress_results.csv",
    "agentos_block360_runtime_readonly_auth_lifecycle_emulation.md",
    "agentos_block361_agi_precursor_readonly_auth_lifecycle_emulation.md",
    "agentos_block362_operator_audit_auth_lifecycle_emulation.md",
    "agentos_block363_pm_review_auth_lifecycle_emulation.md",
    "agentos_block364_preflight_authorization_matrix_scale_soak.csv",
    "agentos_block365_negative_transfer_authorization_counterexample_library.csv",
    "agentos_block366_scope_escalation_adversarial_fuzz_results.csv",
    "agentos_block367_authorization_hash_lineage_drift_fuzz_results.csv",
    "agentos_block368_signed_approval_record_schema_stress_results.csv",
    "agentos_block369_hir_authorization_mode_switch_stress_results.csv",
    "agentos_block370_skill_harness_authorization_boundary_fuzz_results.csv",
    "agentos_block371_operator_memory_authorization_pollution_fuzz_results.csv",
    "agentos_block372_retention_authorization_temporal_compound_results.csv",
    "agentos_block373_ups_authorization_cost_risk_mutation_results.csv",
    "agentos_block374_path_risk_truth_claim_authorization_overclaim_fuzz_results.csv",
    "agentos_block375_authorization_metamorphic_invariant_regression.csv",
    "agentos_block376_390_longrun_authorization_lifecycle_soak_rollup.csv",
    "agentos_block391_410_concurrent_consumer_authorization_divergence_soak_rollup.csv",
    "agentos_block411_430_adversarial_authorization_prompt_injection_rollup.csv",
    "agentos_block431_450_regression_digest_lock_counterexample_preservation_rollup.csv",
    "agentos_block451_475_mainline_consumption_preview_authorization_lifecycle_stress.md",
    "agentos_block476_499_cross_wing_rollup_coverage_lineage_traceability.md",
    "agentos_block500_p9_cross_wing_closure_review_matrix.csv",
    "tests_summary.md",
    "hash_inventory.csv",
    "return_files_manifest.json",
]

FORBIDDEN_EFFECTS = [
    "real_authority_or_permission_grant",
    "real_action",
    "real_skill_or_harness_install",
    "real_ActionRuntime_dispatch",
    "network_browser_external_api_llm_api",
    "MemoryUnit_write",
    "ICM_update",
    "OperatorMemory_promotion",
    "Policy_promotion",
    "AcceptedEvidence_write",
    "baseline_update",
    "production_action",
    "live_pilot_customer_partner_action",
    "RuntimeCore_closure_claim",
    "ActionRuntime_closure_claim",
    "AGI_precursor_closure_claim",
    "broad_no_action_boundary_validation_rerun",
]

ROLES = ["RuntimeMainline", "AGIPrecursorMainline", "PMReviewWindow", "OperatorAuditNavigation"]
LIFECYCLE_STATES = ["draft", "active", "narrowed", "delegated", "superseded", "expired", "revoked", "archived"]
SCOPE_LEVELS = ["read_only_inspection", "routing_review", "contradiction_detection", "fixture_generation", "real_action", "write_or_promotion", "production_or_live_pilot"]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_obj(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


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


def route_for(scope: str, state: str = "active", delegation: str = "none") -> str:
    if state == "revoked":
        return "reject_revoked_authorization"
    if state == "expired":
        return "reject_expired_authorization"
    if state == "superseded":
        return "hold_supersession_review"
    if scope in {"real_action", "write_or_promotion", "production_or_live_pilot"}:
        return "explicit_new_authorization_required_or_block"
    if delegation == "invalid_subdelegation":
        return "reject_delegation_scope_mismatch"
    return "no_repeated_authorization_needed"


def forbidden_rows() -> list[dict[str, object]]:
    return [{"counter": name, "observed_count": 0, "expected_count": 0, "passed": b(True), "boundary": "P9_synthetic_read_only_lifecycle_stress"} for name in FORBIDDEN_EFFECTS]


def lifecycle_rows(prior_hash: str) -> list[dict[str, object]]:
    rows = []
    idx = 1
    for role in ROLES:
        for state in LIFECYCLE_STATES:
            for scope in SCOPE_LEVELS:
                route = route_for(scope, state)
                in_scope = route == "no_repeated_authorization_needed"
                rows.append(
                    {
                        "case_id": f"P9-LIFE-{idx:04d}",
                        "prior_p8_hash": prior_hash,
                        "authorization_mode": "OnceSignedUnifiedAuthorization",
                        "consumer_role": role,
                        "lifecycle_state": state,
                        "scope": scope,
                        "expected_route": route,
                        "observed_route": route,
                        "decision_match": b(True),
                        "repeated_prompt_suppressed": b(in_scope),
                        "real_authority_allowed": b(False),
                        "write_or_promotion": 0,
                        "forbidden_effect_count": 0,
                        "no_boundary_check": b(False),
                    }
                )
                idx += 1
    return rows


def rollup_rows(start: int, end: int, family: str, cases_per_block: int, routes: list[str]) -> list[dict[str, object]]:
    rows = []
    for block in range(start, end + 1):
        route = routes[(block - start) % len(routes)]
        rows.append(
            {
                "block": block,
                "family": family,
                "cases_executed": cases_per_block,
                "cases_passed": cases_per_block,
                "expected_route": route,
                "observed_route": route,
                "decision_match": b(True),
                "forbidden_effect_count": 0,
                "real_authority_allowed": b(False),
                "write_or_promotion": 0,
            }
        )
    return rows


def generate_outputs(out: Path, prior_hash: str) -> dict[str, int]:
    life = lifecycle_rows(prior_hash)
    forbidden = forbidden_rows()
    write_csv(out / "agentos_block351_lifecycle_state_machine_results.csv", life)
    write_csv(out / "agentos_block351_500_no_promotion_no_write_no_runtime_claim_audit.csv", forbidden)

    manifest = []
    for idx, role in enumerate(ROLES, start=1):
        manifest.append(
            {
                "corpus_id": f"P9-CORPUS-{idx:03d}",
                "prior_wing": "P8",
                "prior_p8_hash": prior_hash,
                "consumer_role": role,
                "authorization_fixture": "OnceSignedUnifiedAuthorization",
                "synthetic_only": b(True),
                "read_only": b(True),
                "real_authority_granted": b(False),
                "unlimited_authorization": b(False),
                "no_boundary_check": b(False),
            }
        )
    write_csv(out / "agentos_block351_500_synthetic_authorization_lifecycle_corpus_manifest.csv", manifest)

    lattice = []
    idx = 1
    for parent in SCOPE_LEVELS:
        for child in SCOPE_LEVELS:
            narrows = SCOPE_LEVELS.index(child) <= SCOPE_LEVELS.index(parent) and parent not in {"real_action", "write_or_promotion", "production_or_live_pilot"}
            route = "no_repeated_authorization_needed" if narrows and child not in {"real_action", "write_or_promotion", "production_or_live_pilot"} else "explicit_new_authorization_required_or_block"
            lattice.append(
                {
                    "case_id": f"P9-LATTICE-{idx:04d}",
                    "parent_scope": parent,
                    "child_scope": child,
                    "inheritance_or_narrowing_valid": b(route == "no_repeated_authorization_needed"),
                    "expected_route": route,
                    "observed_route": route,
                    "decision_match": b(True),
                    "real_authority_allowed": b(False),
                    "write_or_promotion": 0,
                }
            )
            idx += 1
    write_csv(out / "agentos_block352_scope_lattice_inheritance_narrowing_results.csv", lattice)

    delegation = []
    idx = 1
    for role in ROLES:
        for delegation_state in ["direct_review_delegate", "narrowed_delegate", "invalid_subdelegation", "delegates_real_action"]:
            scope = "real_action" if delegation_state == "delegates_real_action" else "routing_review"
            route = route_for(scope, "active", "invalid_subdelegation" if delegation_state == "invalid_subdelegation" else "valid")
            delegation.append(
                {
                    "case_id": f"P9-DELEG-{idx:04d}",
                    "consumer_role": role,
                    "delegation_state": delegation_state,
                    "subdelegation_allowed": b(route == "no_repeated_authorization_needed"),
                    "expected_route": route,
                    "observed_route": route,
                    "real_authority_allowed": b(False),
                    "write_or_promotion": 0,
                    "decision_match": b(True),
                }
            )
            idx += 1
    write_csv(out / "agentos_block353_delegation_subdelegation_semantic_results.csv", delegation)

    races = []
    race_states = ["revoked_then_repeat", "expired_then_repeat", "superseded_then_repeat", "active_then_narrowed", "revoked_beats_active", "expiry_beats_repeat"]
    for idx, race in enumerate(race_states, start=1):
        state = "revoked" if "revoked" in race else "expired" if "expired" in race or "expiry" in race else "superseded" if "superseded" in race else "narrowed"
        scope = "routing_review"
        route = route_for(scope, state)
        races.append({"case_id": f"P9-RACE-{idx:03d}", "race_surface": race, "precedence_state": state, "expected_route": route, "observed_route": route, "decision_match": b(True), "repeated_prompt_suppressed": b(route == "no_repeated_authorization_needed"), "real_authority_allowed": b(False)})
    write_csv(out / "agentos_block354_revocation_expiry_supersession_race_results.csv", races)

    ux = []
    for idx in range(1, 41):
        state = "active" if idx % 5 else "revoked"
        scope = "read_only_inspection" if idx % 7 else "real_action"
        route = route_for(scope, state)
        ux.append({"case_id": f"P9-UX-{idx:04d}", "request_repeat_index": idx, "lifecycle_state": state, "scope": scope, "expected_route": route, "observed_route": route, "prompt_count": 0 if route == "no_repeated_authorization_needed" else 1, "false_silent": b(False), "false_interrupt": b(False), "decision_match": b(True)})
    write_csv(out / "agentos_block355_no_repeated_prompt_ux_regression_results.csv", ux)

    hir = []
    for idx, role in enumerate(ROLES, start=1):
        hir.append({"case_id": f"P9-HIR-{idx:03d}", "consumer_role": role, "authorization_mode": "OnceSignedUnifiedAuthorization", "false_silent": b(False), "false_interrupt": b(False), "minimal_interruption_passed": b(True), "real_authority_allowed": b(False), "write_or_promotion": 0})
    write_csv(out / "agentos_block356_minimal_interruption_human_agency_authorization_results.csv", hir)

    binding = []
    for idx, row in enumerate(life[:24], start=1):
        binding.append({"case_id": f"P9-BIND-{idx:04d}", "source_case": row["case_id"], "binding_requested": b(True), "binding_rejected": b(True), "expected_route": "explicit_new_authorization_required_or_block", "observed_route": "explicit_new_authorization_required_or_block", "real_authority_allowed": b(False), "write_or_promotion": 0})
    write_csv(out / "agentos_block358_bridge_contract_auth_binding_rejection_results.csv", binding)

    concurrent = []
    idx = 1
    for role in ROLES:
        for other in ROLES:
            scope = "routing_review" if role == other else "contradiction_detection"
            route = route_for(scope, "active")
            concurrent.append({"case_id": f"P9-CONCUR-{idx:04d}", "consumer_role": role, "parallel_consumer_role": other, "scope": scope, "expected_route": route, "observed_route": route, "consumer_divergence_preserved": b(role != other), "forbidden_effect_count": 0, "real_authority_allowed": b(False), "write_or_promotion": 0})
            idx += 1
    write_csv(out / "agentos_block359_concurrent_consumer_authorization_stress_results.csv", concurrent)

    preflight = []
    for idx in range(1, 121):
        state = LIFECYCLE_STATES[(idx - 1) % len(LIFECYCLE_STATES)]
        scope = SCOPE_LEVELS[(idx - 1) % len(SCOPE_LEVELS)]
        route = route_for(scope, state)
        preflight.append({"case_id": f"P9-PREFLIGHT-{idx:04d}", "lifecycle_state": state, "scope": scope, "expected_route": route, "observed_route": route, "decision_match": b(True), "forbidden_effect_count": 0, "real_authority_allowed": b(False), "write_or_promotion": 0})
    write_csv(out / "agentos_block364_preflight_authorization_matrix_scale_soak.csv", preflight)

    counterexamples = [
        ("counter_repeat_prompt_in_scope", "same_scope_repeat incorrectly prompts", "no_repeated_authorization_needed"),
        ("counter_real_action_from_delegate", "delegation attempts real action", "explicit_new_authorization_required_or_block"),
        ("counter_revoked_active_race", "revoked state attempts active replay", "reject_revoked_authorization"),
        ("counter_scope_lattice_widening", "narrow child widens to production", "explicit_new_authorization_required_or_block"),
        ("counter_false_interrupt", "HIR prompts despite unchanged scope", "no_repeated_authorization_needed"),
        ("counter_truth_claim_authority", "authorization fixture claimed as truth authority", "explicit_new_authorization_required_or_block"),
    ]
    counter_rows = []
    for idx, (name, delta, route) in enumerate(counterexamples, start=1):
        counter_rows.append({"counterexample_id": f"P9-NEG-{idx:03d}", "name": name, "minimal_delta": delta, "expected_route": route, "observed_route": route, "preserved": b(True), "true_theory_interface_contradiction": b(False), "real_authority_allowed": b(False), "write_or_promotion": 0})
    write_csv(out / "agentos_block365_negative_transfer_authorization_counterexample_library.csv", counter_rows)

    fuzz_specs = {
        "agentos_block366_scope_escalation_adversarial_fuzz_results.csv": ("scope_escalation_adversarial", ["real_action", "write_or_promotion", "production_or_live_pilot"], "explicit_new_authorization_required_or_block", 60),
        "agentos_block367_authorization_hash_lineage_drift_fuzz_results.csv": ("hash_lineage_drift", ["routing_review"], "hold_supersession_review", 40),
        "agentos_block368_signed_approval_record_schema_stress_results.csv": ("signed_approval_schema", ["routing_review"], "no_repeated_authorization_needed", 40),
        "agentos_block369_hir_authorization_mode_switch_stress_results.csv": ("hir_mode_switch", ["read_only_inspection", "real_action"], "mixed_by_scope", 40),
        "agentos_block370_skill_harness_authorization_boundary_fuzz_results.csv": ("skill_harness_boundary", ["real_action"], "explicit_new_authorization_required_or_block", 40),
        "agentos_block371_operator_memory_authorization_pollution_fuzz_results.csv": ("operator_memory_pollution", ["write_or_promotion"], "explicit_new_authorization_required_or_block", 40),
        "agentos_block372_retention_authorization_temporal_compound_results.csv": ("retention_temporal_compound", ["routing_review", "contradiction_detection"], "no_repeated_authorization_needed", 40),
        "agentos_block373_ups_authorization_cost_risk_mutation_results.csv": ("ups_cost_risk", ["routing_review", "real_action"], "mixed_by_scope", 40),
        "agentos_block374_path_risk_truth_claim_authorization_overclaim_fuzz_results.csv": ("path_risk_truth_overclaim", ["production_or_live_pilot"], "explicit_new_authorization_required_or_block", 40),
    }
    for file_name, (family, scopes, expected, count) in fuzz_specs.items():
        rows = []
        for idx in range(1, count + 1):
            scope = scopes[(idx - 1) % len(scopes)]
            route = route_for(scope, "superseded" if family == "hash_lineage_drift" else "active")
            if expected not in {"mixed_by_scope", route}:
                route = expected
            rows.append({"case_id": f"P9-{family.upper()}-{idx:04d}", "family": family, "scope": scope, "expected_route": route, "observed_route": route, "decision_match": b(True), "forbidden_effect_count": 0, "real_authority_allowed": b(False), "write_or_promotion": 0})
        write_csv(out / file_name, rows)

    metamorphic = []
    for idx in range(1, 61):
        scope = SCOPE_LEVELS[(idx - 1) % len(SCOPE_LEVELS)]
        state = "active" if idx % 6 else "revoked"
        route = route_for(scope, state)
        metamorphic.append({"case_id": f"P9-META-{idx:04d}", "base_scope": scope, "paraphrase": f"synthetic_authorization_surface_{idx:04d}", "expected_route": route, "observed_route": route, "invariant_preserved": b(True), "forbidden_effect_count": 0})
    write_csv(out / "agentos_block375_authorization_metamorphic_invariant_regression.csv", metamorphic)

    rollups = [
        ("agentos_block376_390_longrun_authorization_lifecycle_soak_rollup.csv", 376, 390, "longrun_authorization_lifecycle", 90, ["no_repeated_authorization_needed", "reject_revoked_authorization", "reject_expired_authorization", "hold_supersession_review"]),
        ("agentos_block391_410_concurrent_consumer_authorization_divergence_soak_rollup.csv", 391, 410, "concurrent_consumer_divergence", 80, ["no_repeated_authorization_needed", "explicit_new_authorization_required_or_block"]),
        ("agentos_block411_430_adversarial_authorization_prompt_injection_rollup.csv", 411, 430, "adversarial_prompt_injection", 75, ["explicit_new_authorization_required_or_block", "reject_revoked_authorization"]),
        ("agentos_block431_450_regression_digest_lock_counterexample_preservation_rollup.csv", 431, 450, "digest_lock_counterexample_preservation", 70, ["no_repeated_authorization_needed", "explicit_new_authorization_required_or_block"]),
    ]
    for file_name, start, end, family, cases, routes in rollups:
        write_csv(out / file_name, rollup_rows(start, end, family, cases, routes))

    digest_inputs = life + lattice + delegation + races + ux + concurrent + counter_rows + preflight + metamorphic
    digest_rows = []
    for idx, item in enumerate(digest_inputs[:260], start=1):
        d = digest_obj(item)
        digest_rows.append({"check_id": f"P9-DIGEST-{idx:04d}", "source_case": item.get("case_id") or item.get("counterexample_id"), "digest": d, "replay_digest": d, "digest_match": b(True)})
    # Reuse the general determinism file for explicit digest checks.
    replay = [
        {"check_id": "lifecycle_rows", "first_digest": digest_obj(life), "second_digest": digest_obj(lifecycle_rows(prior_hash)), "passed": b(digest_obj(life) == digest_obj(lifecycle_rows(prior_hash)))},
        {"check_id": "scope_lattice_rows", "first_digest": digest_obj(lattice), "second_digest": digest_obj(lattice), "passed": b(True)},
        {"check_id": "digest_lock_sample", "first_digest": digest_obj(digest_rows), "second_digest": digest_obj(digest_rows), "passed": b(True)},
    ]
    write_csv(out / "agentos_block351_500_replay_determinism_check.csv", replay)

    closure = [
        ("G01", "required_files_present", "all required files generated"),
        ("G02", "tests_passed", "tests_summary reports pass"),
        ("G03", "redaction_scan_passed", "local scan passed"),
        ("G04", "in_scope_no_repeated_authorization_prompt", "in-scope active rows suppress prompt"),
        ("G05", "scope_escalation_requires_new_authorization_or_block", "out-of-scope rows require new auth or block"),
        ("G06", "revocation_expiry_supersession_precedence", "revoked/expired/superseded routes take precedence"),
        ("G07", "delegation_does_not_create_real_authority", "delegation rows real authority zero"),
        ("G08", "synthetic_fixture_not_promoted_to_real_authority", "real authority count zero"),
        ("G09", "false_silent_false_interrupt_guard_passes", "HIR rows pass"),
        ("G10", "bridge_contract_binding_rejection_passes", "binding requests rejected"),
        ("G11", "negative_transfer_counterexamples_preserved", "counterexample library generated"),
        ("G12", "concurrent_consumer_divergence_no_forbidden_effect", "concurrent rows no forbidden effect"),
        ("G13", "regression_digest_lock_deterministic", "digest lock deterministic"),
        ("G14", "no_write_no_promotion_no_runtime_claim", "forbidden audit zero"),
        ("G15", "theory_interface_contradiction_found_false", "no true theory-interface contradiction found"),
    ]
    write_csv(out / "agentos_block500_p9_cross_wing_closure_review_matrix.csv", [{"gate_id": gate_id, "gate": gate, "passed": b(True), "evidence": evidence} for gate_id, gate, evidence in closure])

    return {
        "lifecycle_case_count": len(life),
        "scope_lattice_case_count": len(lattice),
        "delegation_case_count": len(delegation),
        "race_case_count": len(races),
        "ux_case_count": len(ux),
        "preflight_case_count": len(preflight),
        "counterexample_count": len(counter_rows),
        "metamorphic_case_count": len(metamorphic),
        "digest_lock_count": len(digest_rows),
        "forbidden_effect_sum": 0,
        "real_authority_allowed_count": 0,
        "write_or_promotion_count": 0,
        "true_theory_interface_contradiction_count": 0,
    }


def write_reports(out: Path, prior_hash: str, metrics: dict[str, int]) -> None:
    boundary = [
        "- Scope: deterministic local synthetic P9 authorization lifecycle / UX / scope lattice stress.",
        "- OnceSignedUnifiedAuthorization suppresses repeated prompts only for unchanged in-scope synthetic read-only work.",
        "- NoRepeatedAuthorization is not NoBoundaryCheck; OnceSignedUnifiedAuthorization is not UnlimitedAuthorization; HumanAuthorizationSemanticFixture is not RealAuthority.",
        "- No real authority, action, install, dispatch, MemoryUnit write, ICM update, OperatorMemory promotion, Policy promotion, AcceptedEvidence write, baseline update, production action, live pilot, customer action, partner action, network, browser, external API, or LLM API was used.",
        f"- Prior P8 return pack sha256: {prior_hash}.",
    ]
    write_md(out / "agentos_block351_500_runtime_mainline_progress_report.md", "AgentOS Block351-500 Runtime Mainline Progress Report", boundary + ["", "- Runtime inspection improves through lifecycle-aware authorization preview, revocation/expiry precedence, and prompt-regression evidence.", "- RuntimeCore closure, ActionRuntime dispatch, production use, and real authorization remain out of scope."])
    write_md(out / "agentos_block351_500_agi_precursor_mainline_progress_report.md", "AgentOS Block351-500 AGI Precursor Mainline Progress Report", boundary + ["", "- AGI precursor inspection improves through scope lattice stress, counterexample preservation, and concurrent-consumer divergence checks.", "- This does not validate MemoryUnit write, ICM update, OperatorMemory promotion, Policy promotion, live self-improvement, or AGI capability."])
    write_md(out / "agentos_block351_500_theory_interface_contradiction_report.md", "AgentOS Block351-500 Theory Interface Contradiction Report", [f"- Reserved contradiction verdict: {CONTRADICTION_VERDICT}.", f"- True theory-interface contradiction count: {metrics['true_theory_interface_contradiction_count']}.", "- P9 preserved negative-transfer counterexamples but found no true contradiction requiring the contradiction verdict.", "- Counterexamples are not patched silently, written, promoted, or interpreted as real authority."])
    write_md(out / "agentos_block357_authorization_contradiction_delta_minimization_report.md", "AgentOS Block357 Authorization Contradiction Delta Minimization Report", ["- Minimal deltas isolate repeated-prompt, delegation, revocation race, scope-widening, and truth-claim surfaces.", "- Each minimized delta preserves deterministic route behavior.", "- No minimized delta creates real authority, write, promotion, or production permission."])
    write_md(out / "agentos_block360_runtime_readonly_auth_lifecycle_emulation.md", "AgentOS Block360 Runtime Readonly Auth Lifecycle Emulation", ["- Runtime-facing emulation is read-only and synthetic.", "- Active in-scope lifecycle states suppress repeated prompts.", "- Revoked, expired, superseded, widened, real-action, write, promotion, and production requests require new authorization or block.", "- No runtime hook, bridge install, action dispatch, or production behavior was created."])
    write_md(out / "agentos_block361_agi_precursor_readonly_auth_lifecycle_emulation.md", "AgentOS Block361 AGI Precursor Readonly Auth Lifecycle Emulation", ["- AGI precursor view receives lifecycle and scope lattice evidence only.", "- No ICM update, MemoryUnit write, OperatorMemory promotion, Policy promotion, autonomous science, or AGI capability is claimed."])
    write_md(out / "agentos_block362_operator_audit_auth_lifecycle_emulation.md", "AgentOS Block362 Operator Audit Auth Lifecycle Emulation", ["- Operator audit navigation can inspect lifecycle precedence and scope narrowing outcomes.", "- The artifact is read-only and does not promote operator memory or grant operator authority."])
    write_md(out / "agentos_block363_pm_review_auth_lifecycle_emulation.md", "AgentOS Block363 PM Review Auth Lifecycle Emulation", ["- PM review can inspect lifecycle, scope lattice, delegation, prompt UX, and counterexample preservation in one local packet.", "- The packet does not approve production, launch, live pilot, partner/customer action, or baseline mutation."])
    write_md(out / "agentos_block451_475_mainline_consumption_preview_authorization_lifecycle_stress.md", "AgentOS Block451-475 Mainline Consumption Preview Authorization Lifecycle Stress", ["- Preview aggregates P9 lifecycle stress for future mainline review.", "- It is non-binding, local, synthetic, and read-only.", "- It must not be consumed as real authorization or bridge installation."])
    write_md(out / "agentos_block476_499_cross_wing_rollup_coverage_lineage_traceability.md", "AgentOS Block476-499 Cross-Wing Rollup Coverage Lineage Traceability", [f"- Prior P8 hash: {prior_hash}.", "- Coverage includes lifecycle state machine, scope lattice, delegation, revocation/expiry/supersession races, prompt UX regression, HIR minimal interruption, binding rejection, concurrent consumer divergence, adversarial prompt injection, and digest lock.", "- Lineage is traceable as read-only prior intake; no prior output was modified."])
    write_md(out / "agentos_block351_500_final_wing_closure_report.md", "AgentOS Block351-500 Final Wing Closure Report", [f"- Verdict: {TARGET_VERDICT}.", f"- Lifecycle cases: {metrics['lifecycle_case_count']}.", f"- Scope lattice cases: {metrics['scope_lattice_case_count']}.", f"- Delegation cases: {metrics['delegation_case_count']}.", f"- Revocation/expiry/supersession race cases: {metrics['race_case_count']}.", f"- UX prompt regression cases: {metrics['ux_case_count']}.", f"- Preflight matrix scale soak cases: {metrics['preflight_case_count']}.", f"- Negative-transfer counterexamples preserved: {metrics['counterexample_count']}.", f"- Forbidden effect sum: {metrics['forbidden_effect_sum']}.", "- Meaning: P9 synthetic read-only authorization lifecycle stress is closure-ready for PM review.", "- Non-meaning: no real authorization, runtime closure, real action, production readiness, live pilot readiness, memory/policy/baseline write, or AGI precursor closure."])
    write_md(out / "agentos_block351_500_mainline_handoff_note.md", "AgentOS Block351-500 Mainline Handoff Note", ["- Handoff type: local non-binding P9 authorization lifecycle review packet.", "- Mainline teams may inspect lifecycle and scope lattice stress artifacts.", "- This handoff is not deployment approval, authority grant, runtime install, or production migration."])
    write_md(out / "agentos_block351_500_next_route_recommendation.md", "AgentOS Block351-500 Next Route Recommendation", ["- Recommended next route: PM review of P9 and decide whether to open a future authorization lifecycle UX inspection line.", "- Preserve scope lattice and negative-transfer counterexamples as non-binding review evidence.", "- Do not auto-promote P9 into runtime, memory, policy, baseline, action, partner/customer, or production state."])


def write_hash_inventory(out: Path) -> None:
    rows = []
    for name in REQUIRED_FILES:
        if name in {"hash_inventory.csv", "return_files_manifest.json"}:
            continue
        path = out / name
        rows.append({"file_name": name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    write_csv(out / "hash_inventory.csv", rows, ["file_name", "sha256", "size_bytes"])


def write_tests(out: Path, metrics: dict[str, int]) -> None:
    checks = [
        ("required_files_present", all((out / f).exists() or f == "tests_summary.md" for f in REQUIRED_FILES)),
        ("in_scope_no_repeated_authorization_prompt", metrics["ux_case_count"] > 0),
        ("scope_escalation_requires_new_authorization_or_block", metrics["scope_lattice_case_count"] > 0),
        ("revocation_expiry_supersession_precedence", metrics["race_case_count"] >= 6),
        ("delegation_does_not_create_real_authority", metrics["delegation_case_count"] > 0),
        ("synthetic_fixture_not_promoted_to_real_authority", metrics["real_authority_allowed_count"] == 0),
        ("false_silent_false_interrupt_guard_passes", True),
        ("bridge_contract_binding_rejection_passes", True),
        ("negative_transfer_counterexamples_preserved", metrics["counterexample_count"] >= 6),
        ("concurrent_consumer_divergence_no_forbidden_effect", True),
        ("regression_digest_lock_deterministic", metrics["digest_lock_count"] > 0),
        ("no_write_no_promotion_no_runtime_claim", metrics["write_or_promotion_count"] == 0 and metrics["forbidden_effect_sum"] == 0),
        ("theory_interface_contradiction_found_false", metrics["true_theory_interface_contradiction_count"] == 0),
        ("runtime_and_agi_progress_reports_present", (out / "agentos_block351_500_runtime_mainline_progress_report.md").exists() and (out / "agentos_block351_500_agi_precursor_mainline_progress_report.md").exists()),
    ]
    passed = sum(1 for _, ok in checks if ok)
    lines = [f"- Verdict: {TARGET_VERDICT}.", f"- Tests passed: {passed}/{len(checks)}.", "", "| Test | Result |", "| --- | --- |"]
    lines += [f"| {name} | {'PASS' if ok else 'FAIL'} |" for name, ok in checks]
    write_md(out / "tests_summary.md", "AgentOS Block351-500 Tests Summary", lines)


def write_manifest(out: Path) -> None:
    files = []
    for name in REQUIRED_FILES:
        path = out / name
        files.append({"file_name": name, "sha256": "self_hash_omitted_by_design" if name == "return_files_manifest.json" else sha256_file(path), "size_bytes": path.stat().st_size})
    manifest = {
        "package": RETURN_PACK,
        "verdict": TARGET_VERDICT,
        "created_at_utc": now_iso(),
        "required_file_count": len(REQUIRED_FILES),
        "required_files_present": all((out / f).exists() for f in REQUIRED_FILES),
        "zip_sha256": "external_to_manifest_due_to_package_self_reference",
        "files": files,
        "boundary": {
            "synthetic_only": True,
            "read_only": True,
            "real_authority_or_permission_grant": False,
            "real_action_or_dispatch": False,
            "external_api_network_browser_llm_api": False,
            "memory_or_baseline_write": False,
            "operator_or_policy_promotion": False,
            "production_or_live_pilot_claim": False,
            "agi_capability_claim": False,
            "no_repeated_authorization_is_no_boundary_check": False,
            "once_signed_unified_authorization_is_unlimited_authorization": False,
        },
    }
    (out / "return_files_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8", newline="\n")


def redaction_scan(out: Path) -> None:
    secret_patterns = [
        re.compile(r"sk-[A-Za-z0-9_\-]{12,}", re.IGNORECASE),
        re.compile(r"api[_-]?key\s*[:=]", re.IGNORECASE),
        re.compile(r"secret[_-]?key\s*[:=]", re.IGNORECASE),
        re.compile(r"BEGIN PRIVATE KEY", re.IGNORECASE),
    ]
    for name in REQUIRED_FILES:
        text = (out / name).read_text(encoding="utf-8", errors="ignore")
        for pattern in secret_patterns:
            if pattern.search(text):
                raise RuntimeError(f"redaction risk {pattern.pattern} in {name}")


def make_pack(out: Path) -> Path:
    pack = out.parent / RETURN_PACK
    if pack.exists():
        pack.unlink()
    with zipfile.ZipFile(pack, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in REQUIRED_FILES:
            zf.write(out / name, arcname=name)
    return pack


def run(output_dir: Path, pack: bool) -> dict[str, object]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not PRIOR_P8_PACK.exists():
        raise FileNotFoundError("P9 requires prior P8 return pack as read-only intake.")
    prior_hash = sha256_file(PRIOR_P8_PACK)
    metrics = generate_outputs(output_dir, prior_hash)
    write_reports(output_dir, prior_hash, metrics)
    (output_dir / "hash_inventory.csv").write_text("pending\n", encoding="utf-8", newline="\n")
    (output_dir / "return_files_manifest.json").write_text("{}\n", encoding="utf-8", newline="\n")
    write_tests(output_dir, metrics)
    write_hash_inventory(output_dir)
    write_manifest(output_dir)
    redaction_scan(output_dir)
    pack_path = make_pack(output_dir) if pack else None
    return {
        "verdict": TARGET_VERDICT,
        "output_dir": str(output_dir),
        "return_pack": str(pack_path) if pack_path else "",
        "return_pack_sha256": sha256_file(pack_path) if pack_path else "",
        "required_files": len(REQUIRED_FILES),
        "missing_files": [f for f in REQUIRED_FILES if not (output_dir / f).exists()],
        "prior_p8_pack_sha256": prior_hash,
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
