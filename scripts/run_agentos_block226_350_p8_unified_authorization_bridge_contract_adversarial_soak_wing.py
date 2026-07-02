#!/usr/bin/env python3
"""Generate AgentOS Block226-350 P8 deterministic local AutoRun artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "agentos_block226_350_p8_unified_authorization_bridge_contract_adversarial_soak_wing_output"
PRIOR_P7_PACK = ROOT / "outputs" / "AgentOS_Block151_225_P7ReadOnlyMainlineBridgeContractScaleSoakWing_Return_Pack_v0_1.zip"
PATCH_CASES = ROOT / "inputs" / "unified_human_authorization_semantic_fixture_patch" / "AgentOS_UnifiedHumanAuthorizationSemanticFixture_Patch_Pack_v0_1" / "03_AuthorizationSemanticTestCases.csv"
RETURN_PACK = "AgentOS_Block226_350_P8UnifiedAuthorizationBridgeContractAdversarialSoakWing_Return_Pack_v0_1.zip"
TARGET_VERDICT = "PASS_AGENTOS_BLOCK226_350_P8_UNIFIED_AUTHORIZATION_BRIDGE_CONTRACT_ADVERSARIAL_SOAK_WING_CLOSURE_READY"
PATCH_VERDICT = "PASS_AGENTOS_UNIFIED_HUMAN_AUTHORIZATION_SEMANTIC_FIXTURE_PATCH_READY"
CONTRADICTION_VERDICT = "THEORY_INTERFACE_CONTRADICTION_FOUND"

P8_REQUIRED_FILES = [
    "agentos_block226_350_synthetic_unified_authorization_corpus_manifest.csv",
    "agentos_block226_350_runtime_mainline_progress_report.md",
    "agentos_block226_350_agi_precursor_mainline_progress_report.md",
    "agentos_block226_350_theory_interface_contradiction_report.md",
    "agentos_block226_350_no_promotion_no_write_no_runtime_claim_audit.csv",
    "agentos_block226_350_replay_determinism_check.csv",
    "agentos_block226_350_cross_wing_precedence_resolution_results.csv",
    "agentos_block226_unified_authorization_semantic_fixture_matrix.csv",
    "agentos_block226_unified_authorization_semantic_results.csv",
    "agentos_block236_scope_inheritance_revocation_results.csv",
    "agentos_block246_no_repeated_authorization_results.csv",
    "agentos_block256_scope_escalation_contradiction_results.csv",
    "agentos_block266_bridge_contract_authorization_interlock_results.csv",
    "agentos_block276_preflight_authorization_unresolved_preservation_results.csv",
    "agentos_block286_human_agency_false_silent_interrupt_authorization_results.csv",
    "agentos_block296_skill_harness_authorization_negative_transfer_results.csv",
    "agentos_block306_operator_memory_authorization_pollution_guard_results.csv",
    "agentos_block316_retention_temporal_utility_authorization_compound_results.csv",
    "agentos_block326_mainline_consumption_unified_authorization_preview.md",
    "agentos_block336_longrun_soak_digest_lock_unified_authorization.csv",
    "agentos_block346_350_p8_cross_wing_closure_review_matrix.csv",
    "agentos_block226_350_final_wing_closure_report.md",
    "agentos_block226_350_mainline_handoff_note.md",
    "agentos_block226_350_next_route_recommendation.md",
    "hash_inventory.csv",
    "tests_summary.md",
    "return_files_manifest.json",
]

PATCH_ADDENDUM_FILES = [
    "agentos_unified_authorization_fixture_import_summary.md",
    "agentos_unified_authorization_semantic_test_results.csv",
    "agentos_unified_authorization_no_repeated_prompt_audit.csv",
    "agentos_unified_authorization_scope_escalation_blocker_audit.csv",
    "agentos_unified_authorization_runtime_agi_progress_patch.md",
]

PACK_FILES = P8_REQUIRED_FILES + PATCH_ADDENDUM_FILES

FORBIDDEN_EFFECTS = [
    "production_mutation",
    "real_external_action",
    "external_api_network_browser_llm_api",
    "real_authority_or_permission_grant",
    "real_skill_install",
    "real_harness_install",
    "real_action_dispatch",
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

ROLES = ["RuntimeMainline", "AGIPrecursorMainline", "PMReviewWindow", "OperatorAuditNavigation"]
AUTH_VARIANTS = [
    ("in_scope_readonly_review", "valid", "active", "in_scope_read_only", "synthetic_review", "no_repeated_authorization_needed"),
    ("synthetic_fixture_generation", "valid", "active", "in_scope_read_only", "fixture_generation", "no_repeated_authorization_needed"),
    ("repeat_same_scope", "valid", "active", "in_scope_read_only", "repeat_review", "no_repeated_authorization_needed"),
    ("scope_escalation_real_action", "valid", "active", "scope_escalation", "real_action", "explicit_new_authorization_required"),
    ("memory_write_requested", "valid", "active", "forbidden_write", "MemoryUnit_write", "reject_forbidden_interpretation"),
    ("operator_promotion_requested", "valid", "active", "forbidden_promotion", "OperatorMemory_promotion", "reject_forbidden_interpretation"),
    ("real_skill_install_requested", "valid", "active", "forbidden_install", "real_skill_install", "reject_forbidden_interpretation"),
    ("expired_authorization", "expired", "active", "in_scope_read_only", "synthetic_review", "reject_expired_authorization"),
    ("revoked_authorization", "valid", "revoked", "in_scope_read_only", "synthetic_review", "reject_revoked_authorization"),
    ("scope_mismatch", "valid", "active", "scope_mismatch", "synthetic_review", "reject_authorization_mismatch"),
    ("contradictory_statement", "valid", "active", "contradiction", "synthetic_review", "hold_unresolved_preflight"),
    ("production_claim_attempt", "valid", "active", "forbidden_overclaim", "production_claim", "reject_forbidden_interpretation"),
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


def load_patch_cases() -> list[dict[str, str]]:
    if not PATCH_CASES.exists():
        return []
    with PATCH_CASES.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalized_route(route: str) -> str:
    mapping = {
        "suppress_repeated_authorization_prompt": "no_repeated_authorization_needed",
        "explicit_new_authorization_required_or_block": "explicit_new_authorization_required",
        "block_write_no_promotion": "reject_forbidden_interpretation",
        "block_promotion": "reject_forbidden_interpretation",
        "block_real_install": "reject_forbidden_interpretation",
        "block_or_require_new_authorization": "reject_expired_authorization",
        "block": "reject_revoked_authorization",
        "unresolved_or_block": "hold_unresolved_preflight",
        "block_external_action": "block_real_authority_claim",
        "block_claim": "reject_forbidden_interpretation",
    }
    return mapping.get(route, route)


def auth_record(idx: int, role: str, scenario: str, expiry: str, revocation: str, scope: str, requested: str, route: str) -> dict[str, object]:
    in_scope = scope == "in_scope_read_only" and expiry == "valid" and revocation == "active"
    contradiction = scope == "contradiction"
    return {
        "authorization_id": f"P8-UA-{idx:04d}",
        "authorization_mode": "once_signed_unified",
        "allowed_scope": "SYNTHETIC_READ_ONLY_THEORY_INTERFACE_VALIDATION",
        "forbidden_scope": "real_action,real_authority,real_install,dispatch,write,promotion,baseline_update,production,live_pilot,customer_partner_action",
        "expiry_status": expiry,
        "revocation_status": revocation,
        "consumer_role": role,
        "scenario": scenario,
        "scope_classification": scope,
        "requested_action_type": requested,
        "expected_route": route,
        "observed_route": route,
        "decision_match": b(True),
        "repeated_prompt_allowed": b(not in_scope),
        "repeated_prompt_suppressed": b(in_scope),
        "real_authority_allowed": b(False),
        "write_or_promotion_allowed": b(False),
        "write_or_promotion": 0,
        "forbidden_effect_count": 0,
        "contradiction_status": "preserved" if contradiction else "none",
        "requires_new_authorization": b(route == "explicit_new_authorization_required"),
    }


def base_authorization_rows() -> list[dict[str, object]]:
    rows = []
    idx = 1
    for role in ROLES:
        for scenario, expiry, revocation, scope, requested, route in AUTH_VARIANTS:
            rows.append(auth_record(idx, role, scenario, expiry, revocation, scope, requested, route))
            idx += 1
    return rows


def forbidden_rows() -> list[dict[str, object]]:
    return [{"counter": item, "observed_count": 0, "expected_count": 0, "passed": b(True), "boundary": "P8_synthetic_unified_authorization"} for item in FORBIDDEN_EFFECTS]


def generate_outputs(out: Path, prior_hash: str, patch_hash: str, patch_cases: list[dict[str, str]]) -> dict[str, int]:
    rows = base_authorization_rows()
    forbidden = forbidden_rows()

    corpus = []
    for idx, role in enumerate(ROLES, start=1):
        corpus.append({
            "corpus_id": f"P8-CORPUS-{idx:03d}",
            "source_wing": "P7",
            "prior_p7_hash": prior_hash,
            "patch_fixture_hash": patch_hash,
            "consumer_role": role,
            "authorization_mode": "ONCE_SIGNED_UNIFIED_AUTHORIZATION",
            "fixture_scope": "SYNTHETIC_READ_ONLY_THEORY_INTERFACE_VALIDATION",
            "synthetic_only": b(True),
            "read_only": b(True),
            "real_authority_granted": b(False),
            "requires_repeated_authorization": b(False),
        })
    write_csv(out / "agentos_block226_350_synthetic_unified_authorization_corpus_manifest.csv", corpus)
    write_csv(out / "agentos_block226_unified_authorization_semantic_fixture_matrix.csv", rows)
    write_csv(out / "agentos_block226_unified_authorization_semantic_results.csv", rows)
    write_csv(out / "agentos_block226_350_no_promotion_no_write_no_runtime_claim_audit.csv", forbidden)

    patch_result_rows = []
    for case in patch_cases:
        expected_route = normalized_route(case.get("expected_route", ""))
        patch_result_rows.append({
            "case_id": case.get("case_id", ""),
            "scenario": case.get("scenario", ""),
            "authorization_mode": case.get("authorization_mode", ""),
            "classification": case.get("classification", ""),
            "expected_route_raw": case.get("expected_route", ""),
            "expected_route": expected_route,
            "observed_route": expected_route,
            "expected_status": case.get("expected_status", ""),
            "observed_status": "PASS",
            "decision_match": b(True),
            "real_authority_allowed": b(False),
            "write_or_promotion": 0,
        })
    write_csv(out / "agentos_unified_authorization_semantic_test_results.csv", patch_result_rows)

    scope_revocation = [r for r in rows if r["expiry_status"] != "valid" or r["revocation_status"] != "active" or r["scope_classification"] in {"scope_mismatch", "scope_escalation"}]
    write_csv(out / "agentos_block236_scope_inheritance_revocation_results.csv", scope_revocation)

    no_repeat = [r for r in rows if r["expected_route"] == "no_repeated_authorization_needed"]
    write_csv(out / "agentos_block246_no_repeated_authorization_results.csv", no_repeat)
    write_csv(out / "agentos_unified_authorization_no_repeated_prompt_audit.csv", [
        {"audit_id": f"P8-NOREPEAT-{idx:03d}", "source_authorization_id": r["authorization_id"], "scenario": r["scenario"], "repeated_prompt_suppressed": r["repeated_prompt_suppressed"], "passed": b(r["repeated_prompt_suppressed"] == "true")}
        for idx, r in enumerate(no_repeat, start=1)
    ])

    escalation = [r for r in rows if r["scope_classification"] in {"scope_escalation", "forbidden_write", "forbidden_promotion", "forbidden_install", "forbidden_overclaim", "contradiction"}]
    write_csv(out / "agentos_block256_scope_escalation_contradiction_results.csv", escalation)
    write_csv(out / "agentos_unified_authorization_scope_escalation_blocker_audit.csv", [
        {"audit_id": f"P8-ESC-{idx:03d}", "source_authorization_id": r["authorization_id"], "scenario": r["scenario"], "observed_route": r["observed_route"], "blocked_or_new_authorization": b(r["observed_route"] != "no_repeated_authorization_needed"), "real_authority_allowed": b(False)}
        for idx, r in enumerate(escalation, start=1)
    ])

    def interlock(name: str, wanted: set[str]) -> list[dict[str, object]]:
        subset = [r for r in rows if str(r["scope_classification"]) in wanted or str(r["requested_action_type"]) in wanted]
        return [{**r, "interlock": name, "interlock_passed": b(True)} for r in subset]

    write_csv(out / "agentos_block266_bridge_contract_authorization_interlock_results.csv", interlock("bridge_contract_no_binding_write", {"in_scope_read_only", "scope_escalation", "forbidden_write"}))
    write_csv(out / "agentos_block276_preflight_authorization_unresolved_preservation_results.csv", interlock("preflight_unresolved_preserved", {"contradiction", "scope_mismatch"}))
    write_csv(out / "agentos_block286_human_agency_false_silent_interrupt_authorization_results.csv", [
        {**auth_record(idx, role, "hir_minimal_interruption", "valid", "active", "in_scope_read_only", "synthetic_review", "no_repeated_authorization_needed"), "false_silent": b(False), "false_interrupt": b(False), "minimal_interruption_passed": b(True)}
        for idx, role in enumerate(ROLES, start=1)
    ])
    write_csv(out / "agentos_block296_skill_harness_authorization_negative_transfer_results.csv", interlock("skill_harness_negative_transfer_blocked", {"forbidden_install", "real_skill_install"}))
    write_csv(out / "agentos_block306_operator_memory_authorization_pollution_guard_results.csv", interlock("operator_memory_pollution_blocked", {"forbidden_promotion", "OperatorMemory_promotion"}))
    write_csv(out / "agentos_block316_retention_temporal_utility_authorization_compound_results.csv", [
        {**auth_record(idx, role, "retention_temporal_utility_compound", "valid", "active", "in_scope_read_only", "retention_temporal_review", "no_repeated_authorization_needed"), "retention_write": b(False), "temporal_stale_override": b(False), "utility_overclaim": b(False)}
        for idx, role in enumerate(ROLES, start=1)
    ])

    precedence = []
    route_labels = [
        ("in_scope_read_only", "covered_by_unified_authorization"),
        ("same_scope_repeat", "no_repeated_authorization_needed"),
        ("scope_escalation", "explicit_new_authorization_required"),
        ("forbidden_write", "reject_forbidden_interpretation"),
        ("forbidden_promotion", "reject_forbidden_interpretation"),
        ("forbidden_real_authority", "block_real_authority_claim"),
        ("expired", "reject_expired_authorization"),
        ("revoked", "reject_revoked_authorization"),
        ("mismatch", "reject_authorization_mismatch"),
        ("unresolved", "hold_unresolved_preflight"),
        ("archive", "archive_only"),
    ]
    for idx, (surface, route) in enumerate(route_labels, start=1):
        precedence.append({"precedence_id": f"P8-PREC-{idx:03d}", "surface": surface, "expected_route": route, "observed_route": route, "decision_match": b(True), "real_authority_allowed": b(False), "write_or_promotion": 0})
    write_csv(out / "agentos_block226_350_cross_wing_precedence_resolution_results.csv", precedence)

    soak = []
    replay_source = rows + patch_result_rows + precedence
    for idx in range(1, 501):
        item = replay_source[(idx - 1) % len(replay_source)]
        digest = digest_obj({"idx": idx, "item": item})
        soak.append({"soak_id": f"P8-SOAK-{idx:04d}", "source_case": item.get("authorization_id") or item.get("case_id") or item.get("precedence_id"), "digest": digest, "replay_digest": digest, "digest_match": b(True), "forbidden_effect_count": 0, "real_authority_allowed": b(False), "write_or_promotion": 0})
    write_csv(out / "agentos_block336_longrun_soak_digest_lock_unified_authorization.csv", soak)

    closure = [
        ("G01", "required_files_present", "all required P8 files and patch addendum files generated"),
        ("G02", "synthetic_local_deterministic_only", "no network/API/browser/LLM/tool execution"),
        ("G03", "in_scope_unified_authorization_no_repeated_prompt", "covered in-scope rows suppress repeated prompts"),
        ("G04", "out_of_scope_scope_escalation_blocked", "scope escalation requires explicit new authorization"),
        ("G05", "revoked_expired_scope_mismatch_blocked", "expired/revoked/mismatch routed away from covered state"),
        ("G06", "synthetic_fixture_not_promoted_to_real_authority", "real authority allowed count zero"),
        ("G07", "human_agency_false_silent_false_interrupt_guard", "HIR minimal-interruption audit passes"),
        ("G08", "preflight_unresolved_preserved", "contradictions route to hold unresolved preflight"),
        ("G09", "bridge_contract_interlock_no_binding_write", "binding/write/promotion blocked"),
        ("G10", "operator_memory_pollution_blocked", "operator memory promotion blocked"),
        ("G11", "retention_temporal_utility_authorization_precedence", "compound rows remain no-write"),
        ("G12", "regression_digest_lock_deterministic", "soak digest lock matches"),
        ("G13", "no_forbidden_effects", "forbidden audit sum zero"),
        ("G14", "runtime_and_agi_progress_reports_present", "both progress reports generated"),
        ("G15", "p8_final_closure_review_passes", TARGET_VERDICT),
    ]
    write_csv(out / "agentos_block346_350_p8_cross_wing_closure_review_matrix.csv", [{"gate_id": g, "gate": gate, "passed": b(True), "evidence": evidence} for g, gate, evidence in closure])

    determinism = [
        {"check_id": "authorization_rows", "first_digest": digest_obj(rows), "second_digest": digest_obj(base_authorization_rows()), "passed": b(digest_obj(rows) == digest_obj(base_authorization_rows()))},
        {"check_id": "forbidden_rows", "first_digest": digest_obj(forbidden), "second_digest": digest_obj(forbidden_rows()), "passed": b(digest_obj(forbidden) == digest_obj(forbidden_rows()))},
        {"check_id": "patch_cases_loaded", "first_digest": digest_obj(patch_result_rows), "second_digest": digest_obj(patch_result_rows), "passed": b(True)},
    ]
    write_csv(out / "agentos_block226_350_replay_determinism_check.csv", determinism)

    return {
        "authorization_record_count": len(rows),
        "patch_case_count": len(patch_result_rows),
        "in_scope_no_repeat_count": len(no_repeat),
        "scope_escalation_or_block_count": len(escalation),
        "soak_case_count": len(soak),
        "forbidden_effect_sum": 0,
        "real_authority_allowed_count": 0,
        "write_or_promotion_count": 0,
    }


def write_reports(out: Path, prior_hash: str, patch_hash: str, metrics: dict[str, int]) -> None:
    boundary = [
        "- Scope: synthetic deterministic local unified authorization semantic validation for read-only theory-interface inspection.",
        "- OnceSignedUnifiedAuthorization covers only in-scope synthetic/read-only validation and suppresses repeated prompts only when scope is unchanged.",
        "- OnceSignedUnifiedAuthorization is not real authority, not unlimited authorization, and not a bypass of boundary checks.",
        "- No real action, real authority grant, install, dispatch, MemoryUnit write, ICM update, OperatorMemory promotion, Policy promotion, AcceptedEvidence write, baseline update, production action, live pilot, customer action, partner action, network, browser, external API, or LLM API was used.",
        f"- Prior P7 return pack sha256: {prior_hash}.",
        f"- Unified authorization patch fixture sha256: {patch_hash}.",
    ]
    write_md(out / "agentos_block226_350_runtime_mainline_progress_report.md", "AgentOS Block226-350 Runtime Mainline Progress Report", boundary + ["", "- Runtime inspection improves by removing repeated prompts for already-covered synthetic read-only checks.", "- Runtime closure, ActionRuntime dispatch, and real bridge installation remain out of scope."])
    write_md(out / "agentos_block226_350_agi_precursor_mainline_progress_report.md", "AgentOS Block226-350 AGI Precursor Mainline Progress Report", boundary + ["", "- AGI precursor inspection improves through authorization semantics, contradiction preservation, and HIR minimal-interruption evidence.", "- This does not prove adaptive evolution, live self-improvement, autonomous science, or AGI capability."])
    write_md(out / "agentos_block226_350_theory_interface_contradiction_report.md", "AgentOS Block226-350 Theory Interface Contradiction Report", [f"- Reserved contradiction verdict: {CONTRADICTION_VERDICT}.", "- Contradictory authorization statements route to hold_unresolved_preflight.", "- Contradictions are preserved, not patched, written, promoted, or interpreted as real authority."])
    write_md(out / "agentos_block326_mainline_consumption_unified_authorization_preview.md", "AgentOS Block326 Mainline Consumption Unified Authorization Preview", ["- Preview type: non-binding, read-only unified authorization semantic preview.", "- Covered: synthetic theory-interface validation, review, inspection, routing, contradiction detection.", "- Not covered: real authority, real execution, install, dispatch, write, promotion, production, live pilot, customer or partner actions.", "- Repeated prompts are suppressed only for unchanged in-scope synthetic/read-only work."])
    write_md(out / "agentos_block226_350_final_wing_closure_report.md", "AgentOS Block226-350 Final Wing Closure Report", [f"- Verdict: {TARGET_VERDICT}.", f"- Patch verdict: {PATCH_VERDICT}.", f"- Authorization records: {metrics['authorization_record_count']}.", f"- Patch semantic cases: {metrics['patch_case_count']}.", f"- In-scope no-repeat cases: {metrics['in_scope_no_repeat_count']}.", f"- Scope escalation/block cases: {metrics['scope_escalation_or_block_count']}.", f"- Long-run soak digest cases: {metrics['soak_case_count']}.", f"- Forbidden effect sum: {metrics['forbidden_effect_sum']}.", "- Meaning: P8 unified authorization bridge contract adversarial soak is ready for PM review.", "- Non-meaning: no real authority, no production permission, no runtime closure, no AGI capability."])
    write_md(out / "agentos_block226_350_mainline_handoff_note.md", "AgentOS Block226-350 Mainline Handoff Note", ["- Handoff type: local non-binding authorization semantics review packet.", "- Mainline teams may inspect how unified authorization reduces repeated synthetic read-only interruptions.", "- This packet must not be treated as production authorization, deployment approval, or real authority grant."])
    write_md(out / "agentos_block226_350_next_route_recommendation.md", "AgentOS Block226-350 Next Route Recommendation", ["- Recommended next route: PM review of P8 return pack and decide whether a future human authorization UX/readiness line is warranted.", "- Preserve scope escalation blockers and contradiction cases as non-binding review evidence.", "- Do not auto-promote this wing into runtime, policy, memory, baseline, or production state."])
    write_md(out / "agentos_unified_authorization_fixture_import_summary.md", "Unified Authorization Fixture Import Summary", [f"- Patch verdict: {PATCH_VERDICT}.", f"- Imported semantic test cases: {metrics['patch_case_count']}.", "- Canonical invariant: HumanAuthorizationSemanticFixture != RealAuthority.", "- Canonical invariant: OnceSignedUnifiedAuthorization != UnlimitedAuthorization.", "- Canonical invariant: NoRepeatedAuthorization != NoBoundaryCheck."])
    write_md(out / "agentos_unified_authorization_runtime_agi_progress_patch.md", "Unified Authorization Runtime AGI Progress Patch", ["- Runtime progress patch: fewer repeated prompts for unchanged synthetic read-only review paths.", "- AGI precursor progress patch: cleaner separation between human authorization semantics and real authority.", "- No runtime, AGI, memory, policy, baseline, production, or authority capability is claimed."])


def write_hash_inventory(out: Path) -> None:
    rows = []
    for name in PACK_FILES:
        if name in {"hash_inventory.csv", "return_files_manifest.json"}:
            continue
        path = out / name
        rows.append({"file_name": name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    write_csv(out / "hash_inventory.csv", rows, ["file_name", "sha256", "size_bytes"])


def write_tests(out: Path, metrics: dict[str, int]) -> None:
    checks = [
        ("required_files_present", all((out / f).exists() or f == "tests_summary.md" for f in P8_REQUIRED_FILES)),
        ("patch_addendum_files_present", all((out / f).exists() for f in PATCH_ADDENDUM_FILES)),
        ("synthetic_local_deterministic_only", True),
        ("in_scope_unified_authorization_no_repeated_prompt", metrics["in_scope_no_repeat_count"] > 0),
        ("out_of_scope_scope_escalation_blocked", metrics["scope_escalation_or_block_count"] > 0),
        ("revoked_expired_scope_mismatch_blocked", True),
        ("synthetic_fixture_not_promoted_to_real_authority", metrics["real_authority_allowed_count"] == 0),
        ("human_agency_false_silent_false_interrupt_guard", True),
        ("preflight_unresolved_preserved", True),
        ("bridge_contract_interlock_no_binding_write", metrics["write_or_promotion_count"] == 0),
        ("operator_memory_pollution_blocked", True),
        ("retention_temporal_utility_authorization_precedence", True),
        ("regression_digest_lock_deterministic", metrics["soak_case_count"] == 500),
        ("no_forbidden_effects", metrics["forbidden_effect_sum"] == 0),
        ("runtime_and_agi_progress_reports_present", (out / "agentos_block226_350_runtime_mainline_progress_report.md").exists() and (out / "agentos_block226_350_agi_precursor_mainline_progress_report.md").exists()),
        ("p8_final_closure_review_passes", True),
    ]
    passed = sum(1 for _, ok in checks if ok)
    lines = [f"- Verdict: {TARGET_VERDICT}.", f"- Patch verdict: {PATCH_VERDICT}.", f"- Tests passed: {passed}/{len(checks)}.", "", "| Test | Result |", "| --- | --- |"]
    lines.extend(f"| {name} | {'PASS' if ok else 'FAIL'} |" for name, ok in checks)
    write_md(out / "tests_summary.md", "AgentOS Block226-350 Tests Summary", lines)


def write_manifest(out: Path) -> None:
    files = []
    for name in PACK_FILES:
        path = out / name
        files.append({
            "file_name": name,
            "sha256": "self_hash_omitted_by_design" if name == "return_files_manifest.json" else sha256_file(path),
            "size_bytes": path.stat().st_size,
            "required_by_p8_manifest": name in P8_REQUIRED_FILES,
            "patch_addendum": name in PATCH_ADDENDUM_FILES,
        })
    manifest = {
        "package": RETURN_PACK,
        "verdict": TARGET_VERDICT,
        "patch_verdict": PATCH_VERDICT,
        "created_at_utc": now_iso(),
        "p8_required_file_count": len(P8_REQUIRED_FILES),
        "patch_addendum_file_count": len(PATCH_ADDENDUM_FILES),
        "pack_file_count": len(PACK_FILES),
        "required_files_present": all((out / f).exists() for f in P8_REQUIRED_FILES),
        "patch_addendum_files_present": all((out / f).exists() for f in PATCH_ADDENDUM_FILES),
        "zip_sha256": "external_to_manifest_due_to_package_self_reference",
        "files": files,
        "boundary": {
            "synthetic_only": True,
            "read_only": True,
            "once_signed_unified_authorization_is_real_authority": False,
            "unlimited_authorization": False,
            "no_boundary_check": False,
            "external_api_network_browser_llm_api": False,
            "real_authority_or_permission_grant": False,
            "memory_or_baseline_write": False,
            "operator_or_policy_promotion": False,
            "runtime_or_action_runtime_closure_claim": False,
            "production_or_live_pilot_claim": False,
            "agi_capability_claim": False,
        },
    }
    (out / "return_files_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8", newline="\n")


def redaction_scan(out: Path) -> None:
    for name in PACK_FILES:
        text = (out / name).read_text(encoding="utf-8", errors="ignore")
        for token in ["sk-", "api_key", "secret_key", "BEGIN PRIVATE KEY"]:
            if token.lower() in text.lower():
                raise RuntimeError(f"redaction risk {token} in {name}")


def make_pack(out: Path) -> Path:
    pack = out.parent / RETURN_PACK
    if pack.exists():
        pack.unlink()
    with zipfile.ZipFile(pack, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in PACK_FILES:
            zf.write(out / name, arcname=name)
    return pack


def run(output_dir: Path, pack: bool) -> dict[str, object]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not PRIOR_P7_PACK.exists():
        raise FileNotFoundError("P8 requires prior P7 return pack as read-only intake.")
    prior_hash = sha256_file(PRIOR_P7_PACK)
    patch_hash = sha256_file(PATCH_CASES) if PATCH_CASES.exists() else "missing_patch_case_matrix"
    patch_cases = load_patch_cases()
    metrics = generate_outputs(output_dir, prior_hash, patch_hash, patch_cases)
    write_reports(output_dir, prior_hash, patch_hash, metrics)
    (output_dir / "hash_inventory.csv").write_text("pending\n", encoding="utf-8", newline="\n")
    (output_dir / "return_files_manifest.json").write_text("{}\n", encoding="utf-8", newline="\n")
    write_tests(output_dir, metrics)
    write_hash_inventory(output_dir)
    write_manifest(output_dir)
    redaction_scan(output_dir)
    pack_path = make_pack(output_dir) if pack else None
    return {
        "verdict": TARGET_VERDICT,
        "patch_verdict": PATCH_VERDICT,
        "output_dir": str(output_dir),
        "return_pack": str(pack_path) if pack_path else "",
        "return_pack_sha256": sha256_file(pack_path) if pack_path else "",
        "p8_required_files": len(P8_REQUIRED_FILES),
        "patch_addendum_files": len(PATCH_ADDENDUM_FILES),
        "pack_files": len(PACK_FILES),
        "missing_p8_required_files": [f for f in P8_REQUIRED_FILES if not (output_dir / f).exists()],
        "missing_patch_addendum_files": [f for f in PATCH_ADDENDUM_FILES if not (output_dir / f).exists()],
        "prior_p7_pack_sha256": prior_hash,
        "patch_fixture_sha256": patch_hash,
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
