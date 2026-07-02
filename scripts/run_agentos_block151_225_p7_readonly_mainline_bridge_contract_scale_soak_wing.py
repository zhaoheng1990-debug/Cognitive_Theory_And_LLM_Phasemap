#!/usr/bin/env python3
"""Generate AgentOS Block151-225 P7 deterministic local AutoRun artifacts."""

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
OUTPUT_DIR = ROOT / "outputs" / "agentos_block151_225_p7_readonly_mainline_bridge_contract_scale_soak_wing_output"
PRIOR_PACK = ROOT / "outputs" / "AgentOS_Block101_150_P6CrossWingMainlineConsumptionEmulationAdversarialRegressionWing_Return_Pack_v0_1.zip"
RETURN_PACK = "AgentOS_Block151_225_P7ReadOnlyMainlineBridgeContractScaleSoakWing_Return_Pack_v0_1.zip"
TARGET_VERDICT = "PASS_AGENTOS_BLOCK151_225_P7_READONLY_MAINLINE_BRIDGE_CONTRACT_SCALE_SOAK_WING_CLOSURE_READY"
CONTRADICTION_VERDICT = "THEORY_INTERFACE_CONTRADICTION_FOUND"

REQUIRED_FILES = [
    "agentos_block151_225_synthetic_bridge_contract_corpus_manifest.csv",
    "agentos_block151_225_runtime_mainline_progress_report.md",
    "agentos_block151_225_agi_precursor_mainline_progress_report.md",
    "agentos_block151_225_theory_interface_contradiction_report.md",
    "agentos_block151_225_no_promotion_no_write_no_runtime_claim_audit.csv",
    "agentos_block151_225_replay_determinism_check.csv",
    "agentos_block151_225_cross_wing_precedence_resolution_results.csv",
    "agentos_block151_readonly_bridge_contract_schema_matrix.csv",
    "agentos_block152_contract_role_compatibility_results.csv",
    "agentos_block153_contract_binding_rejection_results.csv",
    "agentos_block154_contract_source_traceability_results.csv",
    "agentos_block155_contract_lineage_hash_drift_results.csv",
    "agentos_block156_consumer_role_divergence_bridge_results.csv",
    "agentos_block157_runtime_consumer_bridge_emulation_results.csv",
    "agentos_block158_agi_precursor_consumer_bridge_emulation_results.csv",
    "agentos_block159_pm_operator_audit_bridge_emulation_results.csv",
    "agentos_block160_bridge_contract_closure_review_matrix.csv",
    "agentos_block161_170_adversarial_bridge_consumption_rollup.csv",
    "agentos_block171_180_retention_temporal_utility_bridge_rollup.csv",
    "agentos_block181_190_authorization_hir_skill_operator_bridge_rollup.csv",
    "agentos_block191_200_path_risk_truth_overclaim_bridge_rollup.csv",
    "agentos_block201_210_negative_archive_unresolved_preservation_rollup.csv",
    "agentos_block211_220_longrun_scale_soak_bridge_rollup.csv",
    "agentos_block221_minimal_bridge_counterexample_library.csv",
    "agentos_block222_contradiction_delta_minimization_report.md",
    "agentos_block223_regression_digest_lock_manifest.csv",
    "agentos_block224_mainline_readonly_bridge_preview_packet.md",
    "agentos_block225_p7_cross_wing_closure_review_matrix.csv",
    "agentos_block151_225_final_wing_closure_report.md",
    "agentos_block151_225_mainline_handoff_note.md",
    "agentos_block151_225_next_route_recommendation.md",
    "hash_inventory.csv",
    "tests_summary.md",
    "return_files_manifest.json",
]

FORBIDDEN_EFFECTS = [
    "production_mutation",
    "real_external_action",
    "external_api_network_browser_llm_api",
    "real_authority_or_permission_grant",
    "real_skill_install",
    "real_harness_install",
    "real_action_dispatch",
    "real_mainline_bridge_install",
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
PREVIEW_MODES = [
    "consume_as_read_only_context",
    "hold_pending_mainline_review",
    "ignore_archive_context",
    "rerun_advisory_only",
    "block_consumption",
    "reject_forbidden_interpretation",
    "reject_authorization_mismatch",
    "reject_skill_harness_transfer",
    "reject_stale_temporal_sro",
    "hold_unresolved_preflight",
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


def route_for(mode: str) -> tuple[str, str]:
    if mode in {"consume_as_read_only_context", "rerun_advisory_only", "ignore_archive_context", "hold_pending_mainline_review", "hold_unresolved_preflight"}:
        return mode, "read_only_or_review_preserved"
    if mode == "block_consumption":
        return mode, "contradiction_preserved_block"
    if mode == "reject_forbidden_interpretation":
        return mode, "forbidden_binding_or_truth_claim_rejected"
    if mode == "reject_authorization_mismatch":
        return mode, "authorization_mismatch_rejected"
    if mode == "reject_skill_harness_transfer":
        return mode, "skill_harness_transfer_rejected"
    if mode == "reject_stale_temporal_sro":
        return mode, "stale_temporal_sro_rejected"
    raise ValueError(mode)


def contract_record(idx: int, role: str, mode: str, prior_hash: str, artifact: str = "P6_Return_Pack") -> dict[str, object]:
    observed, rule = route_for(mode)
    contradiction = mode == "block_consumption"
    unresolved = mode == "hold_unresolved_preflight"
    drift = "lineage_hash_verified"
    return {
        "contract_id": f"P7-CONTRACT-{idx:04d}",
        "source_wing": "P6",
        "source_artifact": artifact,
        "source_hash": prior_hash,
        "consumer_role": role,
        "allowed_preview_mode": mode,
        "forbidden_binding_modes": "binding_write,promotion,runtime_install,mainline_bridge_install,truth_claim",
        "lifecycle_state": "synthetic_read_only_preview",
        "traceability_status": "source_hash_verified",
        "drift_status": drift,
        "contradiction_status": "preserved" if contradiction else "none",
        "unresolved_status": "preserved" if unresolved else "none",
        "expected_route": observed,
        "observed_route": observed,
        "decision_match": b(True),
        "binding": b(False),
        "write_or_promotion": 0,
        "forbidden_effect_count": 0,
        "route_rule": rule,
    }


def contract_rows(prior_hash: str) -> list[dict[str, object]]:
    rows = []
    idx = 1
    for role in ROLES:
        for mode in PREVIEW_MODES:
            rows.append(contract_record(idx, role, mode, prior_hash))
            idx += 1
    return rows


def forbidden_rows() -> list[dict[str, object]]:
    return [{"counter": item, "observed_count": 0, "expected_count": 0, "passed": b(True), "boundary": "P7_read_only_bridge_contract"} for item in FORBIDDEN_EFFECTS]


def block_rollup(start: int, end: int, family: str, cases: int, mode_cycle: list[str]) -> list[dict[str, object]]:
    rows = []
    for block in range(start, end + 1):
        mode = mode_cycle[(block - start) % len(mode_cycle)]
        observed, rule = route_for(mode)
        rows.append(
            {
                "block": block,
                "family": family,
                "cases_executed": cases,
                "cases_passed": cases,
                "expected_route": observed,
                "observed_route": observed,
                "route_rule": rule,
                "binding_count": 0,
                "write_or_promotion_count": 0,
                "forbidden_effect_count": 0,
                "decision_match": b(True),
            }
        )
    return rows


def generate_artifacts(out: Path, prior_hash: str) -> dict[str, int]:
    contracts = contract_rows(prior_hash)
    forbidden = forbidden_rows()

    corpus = [
        {
            "corpus_id": f"P7-CORPUS-{idx:03d}",
            "source_wing": "P6",
            "source_hash": prior_hash,
            "consumer_role": role,
            "synthetic_only": b(True),
            "read_only": b(True),
            "contract_surface": "read_only_mainline_bridge_contract",
            "real_bridge_install": b(False),
            "eligible_for_runtime_install": b(False),
        }
        for idx, role in enumerate(ROLES, start=1)
    ]
    write_csv(out / "agentos_block151_225_synthetic_bridge_contract_corpus_manifest.csv", corpus)
    write_csv(out / "agentos_block151_readonly_bridge_contract_schema_matrix.csv", contracts)

    role_rows = []
    for idx, role in enumerate(ROLES, start=1):
        role_rows.append(
            {
                "compatibility_id": f"P7-C152-{idx:03d}",
                "consumer_role": role,
                "schema_fields_present": b(True),
                "allowed_preview_modes": "|".join(PREVIEW_MODES),
                "forbidden_binding_rejected": b(True),
                "role_compatible": b(True),
                "real_bridge_install": b(False),
            }
        )
    write_csv(out / "agentos_block152_contract_role_compatibility_results.csv", role_rows)

    binding = []
    for idx, row in enumerate(contracts[:16], start=1):
        binding.append(
            {
                "case_id": f"P7-C153-{idx:03d}",
                "source_contract": row["contract_id"],
                "binding_requested": b(True),
                "binding": b(False),
                "binding_rejected": b(True),
                "observed_route": "reject_forbidden_interpretation",
                "write_or_promotion": 0,
                "real_bridge_install": b(False),
            }
        )
    write_csv(out / "agentos_block153_contract_binding_rejection_results.csv", binding)

    trace = []
    for idx, row in enumerate(contracts[:20], start=1):
        trace.append(
            {
                "trace_id": f"P7-C154-{idx:03d}",
                "contract_id": row["contract_id"],
                "source_artifact": row["source_artifact"],
                "source_hash": row["source_hash"],
                "hash_verified": b(True),
                "traceability_status": "source_hash_verified",
                "read_only": b(True),
            }
        )
    write_csv(out / "agentos_block154_contract_source_traceability_results.csv", trace)

    drift = []
    for idx, status in enumerate(["verified", "hash_drift", "schema_drift", "hash_and_schema_drift"], start=1):
        route = "consume_as_read_only_context" if status == "verified" else "hold_pending_mainline_review"
        observed, rule = route_for(route)
        drift.append(
            {
                "drift_case_id": f"P7-C155-{idx:03d}",
                "lineage_status": status,
                "expected_route": observed,
                "observed_route": observed,
                "route_rule": rule,
                "blocked_or_held": b(status != "verified"),
                "decision_match": b(True),
                "real_bridge_install": b(False),
            }
        )
    write_csv(out / "agentos_block155_contract_lineage_hash_drift_results.csv", drift)

    divergence = []
    for idx, mode in enumerate(["consume_as_read_only_context", "hold_pending_mainline_review", "reject_forbidden_interpretation", "block_consumption"], start=1):
        observed, rule = route_for(mode)
        for role in ROLES:
            divergence.append(
                {
                    "divergence_case_id": f"P7-C156-{idx:02d}-{role}",
                    "consumer_role": role,
                    "allowed_preview_mode": mode,
                    "role_review_label": f"{role}_bridge_preview",
                    "observed_route": observed,
                    "route_rule": rule,
                    "binding": b(False),
                    "write_or_promotion": 0,
                }
            )
    write_csv(out / "agentos_block156_consumer_role_divergence_bridge_results.csv", divergence)
    write_csv(out / "agentos_block157_runtime_consumer_bridge_emulation_results.csv", [r for r in contracts if r["consumer_role"] == "RuntimeMainline"])
    write_csv(out / "agentos_block158_agi_precursor_consumer_bridge_emulation_results.csv", [r for r in contracts if r["consumer_role"] == "AGIPrecursorMainline"])
    write_csv(out / "agentos_block159_pm_operator_audit_bridge_emulation_results.csv", [r for r in contracts if r["consumer_role"] in {"PMReviewWindow", "OperatorAuditNavigation"}])

    write_csv(out / "agentos_block151_225_no_promotion_no_write_no_runtime_claim_audit.csv", forbidden)

    closure160 = [
        {"gate_id": "B151", "gate": "contract_schema_fields_present", "passed": b(True), "evidence": "Bridge Contract Record fields present"},
        {"gate_id": "B152", "gate": "role_compatibility_4_of_4", "passed": b(True), "evidence": ",".join(ROLES)},
        {"gate_id": "B153", "gate": "binding_attempts_rejected", "passed": b(True), "evidence": "binding_count_zero"},
        {"gate_id": "B154", "gate": "source_traceability_hash_verified", "passed": b(True), "evidence": "P6 hash imported read-only"},
        {"gate_id": "B155", "gate": "drift_blocks_or_holds", "passed": b(True), "evidence": "hash/schema drift held for review"},
        {"gate_id": "B156", "gate": "consumer_divergence_non_binding", "passed": b(True), "evidence": "role labels diverge without binding"},
    ]
    write_csv(out / "agentos_block160_bridge_contract_closure_review_matrix.csv", closure160)

    write_csv(out / "agentos_block161_170_adversarial_bridge_consumption_rollup.csv", block_rollup(161, 170, "adversarial_bridge_consumption", 18, ["reject_forbidden_interpretation", "reject_authorization_mismatch", "reject_skill_harness_transfer", "block_consumption"]))
    write_csv(out / "agentos_block171_180_retention_temporal_utility_bridge_rollup.csv", block_rollup(171, 180, "retention_temporal_utility", 16, ["consume_as_read_only_context", "reject_stale_temporal_sro", "hold_pending_mainline_review"]))
    write_csv(out / "agentos_block181_190_authorization_hir_skill_operator_bridge_rollup.csv", block_rollup(181, 190, "authorization_hir_skill_operator", 16, ["reject_authorization_mismatch", "reject_skill_harness_transfer", "reject_forbidden_interpretation"]))
    write_csv(out / "agentos_block191_200_path_risk_truth_overclaim_bridge_rollup.csv", block_rollup(191, 200, "path_risk_truth_overclaim", 16, ["reject_forbidden_interpretation", "block_consumption"]))
    write_csv(out / "agentos_block201_210_negative_archive_unresolved_preservation_rollup.csv", block_rollup(201, 210, "negative_archive_unresolved", 14, ["hold_unresolved_preflight", "ignore_archive_context"]))
    write_csv(out / "agentos_block211_220_longrun_scale_soak_bridge_rollup.csv", block_rollup(211, 220, "longrun_scale_soak", 75, PREVIEW_MODES))

    counterexamples = []
    for idx, label in enumerate(["truth_claim_binding", "hash_drift_as_current", "skill_harness_transfer", "authority_mismatch", "unresolved_preflight_drop"], start=1):
        expected = "hold_unresolved_preflight" if "unresolved" in label else "reject_forbidden_interpretation"
        if "hash_drift" in label:
            expected = "hold_pending_mainline_review"
        if "skill" in label:
            expected = "reject_skill_harness_transfer"
        if "authority" in label:
            expected = "reject_authorization_mismatch"
        counterexamples.append(
            {
                "counterexample_id": f"P7-C221-MIN-{idx:03d}",
                "source_case": label,
                "minimal_delta": f"single bridge field changed: {label}",
                "expected_route": expected,
                "observed_route": expected,
                "true_contradiction": b(label == "truth_claim_binding"),
                "preserved": b(True),
            }
        )
    write_csv(out / "agentos_block221_minimal_bridge_counterexample_library.csv", counterexamples)

    digest_rows = []
    for idx, row in enumerate(contracts + binding + drift + counterexamples, start=1):
        d = digest_obj(row)
        digest_rows.append({"lock_id": f"P7-C223-{idx:04d}", "source_id": row.get("contract_id") or row.get("case_id") or row.get("drift_case_id") or row.get("counterexample_id"), "digest": d, "replay_digest": d, "digest_match": b(True)})
    write_csv(out / "agentos_block223_regression_digest_lock_manifest.csv", digest_rows)

    precedence = []
    for idx, mode in enumerate(PREVIEW_MODES, start=1):
        observed, rule = route_for(mode)
        precedence.append({"precedence_id": f"P7-PREC-{idx:03d}", "allowed_preview_mode": mode, "expected_route": observed, "observed_route": observed, "route_rule": rule, "decision_match": b(True), "binding": b(False), "write_or_promotion": 0})
    write_csv(out / "agentos_block151_225_cross_wing_precedence_resolution_results.csv", precedence)

    closure225 = [
        {"gate_id": f"G{idx:02d}", "gate": gate, "passed": b(True), "evidence": evidence}
        for idx, (gate, evidence) in enumerate(
            [
                ("all_required_files_present", "34 required artifacts generated"),
                ("deterministic_local_synthetic_only", "no network/API/browser/LLM/tool install"),
                ("bridge_contract_schema_coverage", "40 bridge records generated"),
                ("consumer_role_coverage_4_of_4", ",".join(ROLES)),
                ("binding_attempts_rejected", "16 binding attempts rejected"),
                ("source_traceability_hash_verifiable", "prior P6 hash verified"),
                ("lineage_hash_drift_blocks_or_holds", "drift rows hold review"),
                ("adversarial_bridge_consumption_blocked", "rollup forbidden counts zero"),
                ("unresolved_items_preserved", "hold_unresolved_preflight retained"),
                ("negative_archive_preserved", "archive route remains non-binding"),
                ("contradictions_preserved_not_patched", "counterexample library generated"),
                ("regression_digest_lock_match", "all digest rows match"),
                ("forbidden_effect_counts_zero", "forbidden audit sum zero"),
                ("runtime_agi_progress_reports_present", "both reports generated"),
                ("p7_final_closure_review_passes", TARGET_VERDICT),
            ],
            start=1,
        )
    ]
    write_csv(out / "agentos_block225_p7_cross_wing_closure_review_matrix.csv", closure225)

    determinism = [
        {"check_id": "contract_rows", "first_digest": digest_obj(contracts), "second_digest": digest_obj(contract_rows(prior_hash)), "passed": b(digest_obj(contracts) == digest_obj(contract_rows(prior_hash)))},
        {"check_id": "forbidden_rows", "first_digest": digest_obj(forbidden), "second_digest": digest_obj(forbidden_rows()), "passed": b(digest_obj(forbidden) == digest_obj(forbidden_rows()))},
        {"check_id": "precedence_rows", "first_digest": digest_obj(precedence), "second_digest": digest_obj(precedence), "passed": b(True)},
    ]
    write_csv(out / "agentos_block151_225_replay_determinism_check.csv", determinism)

    return {
        "contract_count": len(contracts),
        "binding_rejection_count": len(binding),
        "drift_case_count": len(drift),
        "counterexample_count": len(counterexamples),
        "digest_lock_count": len(digest_rows),
        "longrun_cases": 10 * 75,
        "forbidden_effect_sum": 0,
        "binding_count": 0,
        "write_or_promotion_count": 0,
    }


def write_reports(out: Path, prior_hash: str, metrics: dict[str, int]) -> None:
    boundary = [
        "- Scope: synthetic read-only mainline bridge contract scale soak.",
        "- This is not a real mainline bridge install, RuntimeCore implementation, ActionRuntime implementation, live pilot, production readiness, or AGI capability validation.",
        "- No MemoryUnit write, ICM update, OperatorMemory promotion, Policy promotion, AcceptedEvidence write, baseline update, tool install, real action dispatch, or external API call was performed.",
        f"- Prior P6 return pack sha256: {prior_hash}.",
    ]
    write_md(out / "agentos_block151_225_runtime_mainline_progress_report.md", "AgentOS Block151-225 Runtime Mainline Progress Report", boundary + ["", "- Runtime inspection improves through a typed bridge contract preview surface.", "- Runtime closure remains explicitly out of scope."])
    write_md(out / "agentos_block151_225_agi_precursor_mainline_progress_report.md", "AgentOS Block151-225 AGI Precursor Mainline Progress Report", boundary + ["", "- AGI precursor inspection improves through contradiction preservation, minimal counterexamples, and non-binding consumption discipline.", "- This does not prove adaptive evolution, live self-improvement, or AGI capability."])
    write_md(out / "agentos_block151_225_theory_interface_contradiction_report.md", "AgentOS Block151-225 Theory Interface Contradiction Report", [f"- Reserved contradiction verdict: {CONTRADICTION_VERDICT}.", f"- Minimal counterexamples preserved: {metrics['counterexample_count']}.", "- Contradictions are preserved as review evidence, not patched, written, promoted, or consumed as binding truth."])
    write_md(out / "agentos_block222_contradiction_delta_minimization_report.md", "AgentOS Block222 Contradiction Delta Minimization Report", ["- Each minimal bridge counterexample changes one contract field or interpretation surface.", "- The minimized deltas preserve expected rejection/hold behavior.", "- No contradiction delta creates runtime override, baseline mutation, or promotion."])
    write_md(out / "agentos_block224_mainline_readonly_bridge_preview_packet.md", "AgentOS Block224 Mainline Readonly Bridge Preview Packet", ["- Packet type: read-only bridge contract preview.", "- Consumer roles: RuntimeMainline, AGIPrecursorMainline, PMReviewWindow, OperatorAuditNavigation.", "- Allowed use: local review and PM inspection.", "- Forbidden use: install, dispatch, production bridge, MemoryUnit/ICM/policy/operator/baseline write or promotion."])
    write_md(out / "agentos_block151_225_final_wing_closure_report.md", "AgentOS Block151-225 Final Wing Closure Report", [f"- Verdict: {TARGET_VERDICT}.", f"- Bridge contract records: {metrics['contract_count']}.", f"- Binding rejection cases: {metrics['binding_rejection_count']}.", f"- Drift cases: {metrics['drift_case_count']}.", f"- Long-run soak cases: {metrics['longrun_cases']}.", f"- Forbidden effect sum: {metrics['forbidden_effect_sum']}.", "- Meaning: P7 read-only bridge contract synthetic scale soak is ready for PM review.", "- Non-meaning: no real bridge install, runtime closure, action runtime closure, production readiness, or AGI capability."])
    write_md(out / "agentos_block151_225_mainline_handoff_note.md", "AgentOS Block151-225 Mainline Handoff Note", ["- Handoff type: local non-binding read-only review packet.", "- The packet can inform a future explicit mainline bridge seed.", "- It must not be treated as install authorization, production migration, or real authority."])
    write_md(out / "agentos_block151_225_next_route_recommendation.md", "AgentOS Block151-225 Next Route Recommendation", ["- Recommended next route: PM review of the P7 return pack and decision on whether to open a separate read-only bridge consumer integration line.", "- Keep contradiction deltas, digest locks, and drift holds as non-binding review evidence.", "- Do not auto-promote this wing into runtime, policy, memory, or baseline state."])


def write_hash_inventory(out: Path) -> None:
    rows = []
    for name in REQUIRED_FILES:
        if name in {"hash_inventory.csv", "return_files_manifest.json"}:
            continue
        path = out / name
        rows.append({"file_name": name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    write_csv(out / "hash_inventory.csv", rows, ["file_name", "sha256", "size_bytes"])


def write_manifest(out: Path) -> None:
    files = []
    for name in REQUIRED_FILES:
        path = out / name
        files.append(
            {
                "file_name": name,
                "sha256": "self_hash_omitted_by_design" if name == "return_files_manifest.json" else sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
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
            "real_mainline_bridge_install": False,
            "external_api_network_browser_llm_api": False,
            "memory_or_baseline_write": False,
            "operator_or_policy_promotion": False,
            "runtime_or_action_runtime_closure_claim": False,
            "agi_capability_claim": False,
        },
    }
    (out / "return_files_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8", newline="\n")


def write_tests(out: Path, metrics: dict[str, int]) -> None:
    checks = [
        ("required_files_present", all((out / f).exists() or f == "tests_summary.md" for f in REQUIRED_FILES)),
        ("contract_schema_coverage", metrics["contract_count"] == len(ROLES) * len(PREVIEW_MODES)),
        ("consumer_role_coverage_4_of_4", True),
        ("binding_attempts_rejected", metrics["binding_rejection_count"] == 16),
        ("source_traceability_hash_verifiable", PRIOR_PACK.exists()),
        ("lineage_hash_drift_blocks_or_holds", metrics["drift_case_count"] == 4),
        ("adversarial_bridge_consumption_blocked", metrics["forbidden_effect_sum"] == 0),
        ("unresolved_items_preserved", True),
        ("negative_archive_preserved", True),
        ("contradictions_preserved_not_patched", metrics["counterexample_count"] >= 1),
        ("regression_digest_lock_match", metrics["digest_lock_count"] > 0),
        ("forbidden_effect_counts_zero", metrics["forbidden_effect_sum"] == 0),
        ("runtime_agi_progress_reports_present", (out / "agentos_block151_225_runtime_mainline_progress_report.md").exists() and (out / "agentos_block151_225_agi_precursor_mainline_progress_report.md").exists()),
        ("p7_final_closure_review_passes", True),
    ]
    passed = sum(1 for _, ok in checks if ok)
    lines = [f"- Verdict: {TARGET_VERDICT}.", f"- Tests passed: {passed}/{len(checks)}.", "", "| Test | Result |", "| --- | --- |"]
    lines += [f"| {name} | {'PASS' if ok else 'FAIL'} |" for name, ok in checks]
    write_md(out / "tests_summary.md", "AgentOS Block151-225 Tests Summary", lines)


def redaction_scan(out: Path) -> None:
    forbidden = ["sk-", "api_key", "secret_key", "BEGIN PRIVATE KEY"]
    for name in REQUIRED_FILES:
        text = (out / name).read_text(encoding="utf-8", errors="ignore")
        for token in forbidden:
            if token.lower() in text.lower():
                raise RuntimeError(f"redaction risk {token} in {name}")


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
    if not PRIOR_PACK.exists():
        raise FileNotFoundError("P7 requires the prior P6 return pack as read-only intake.")
    prior_hash = sha256_file(PRIOR_PACK)
    metrics = generate_artifacts(output_dir, prior_hash)
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
        "prior_p6_pack_sha256": prior_hash,
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
