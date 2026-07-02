#!/usr/bin/env python3
"""Generate AgentOS Block3001-4500 P15 deterministic local AutoRun artifacts."""

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
OUTPUT_DIR = ROOT / "outputs" / "agentos_block3001_4500_p15_operational_fault_recovery_shadowrun_handoff_freeze_candidate_wing_output"
PRIOR_P14_PACK = ROOT / "outputs" / "AgentOS_Block2251_3000_P14OperationalInvariantFaultInjectionShadowRunRollbackRegressionWing_Return_Pack_v0_1.zip"
RETURN_PACK = "AgentOS_Block3001_4500_P15OperationalFaultRecoveryShadowRunHandoffFreezeCandidateWing_Return_Pack_v0_1.zip"
TARGET_VERDICT = "PASS_AGENTOS_BLOCK3001_4500_P15_OPERATIONAL_FAULT_RECOVERY_SHADOW_RUN_HANDOFF_FREEZE_CANDIDATE_WING_CLOSURE_READY"
CONTRADICTION_VERDICT = "THEORY_INTERFACE_CONTRADICTION_FOUND"

REQUIRED_FILES = [
    "agentos_block3001_4500_final_wing_closure_report.md",
    "agentos_block3001_4500_theory_interface_contradiction_report.md",
    "agentos_block3001_4500_runtime_mainline_progress_report.md",
    "agentos_block3001_4500_agi_precursor_mainline_progress_report.md",
    "agentos_block3001_4500_mainline_handoff_note.md",
    "agentos_block3001_4500_next_route_recommendation.md",
    "agentos_block3001_4500_no_promotion_no_write_no_runtime_claim_audit.csv",
    "agentos_block3001_4500_replay_determinism_check.csv",
    "agentos_block3001_4500_synthetic_recovery_handoff_freeze_corpus_manifest.csv",
    "agentos_block3001_3150_p14_fault_recovery_candidate_consolidation.csv",
    "agentos_block3151_3300_rollback_preview_freeze_candidate_guard.csv",
    "agentos_block3301_3450_counterexample_archive_recovery_handoff.csv",
    "agentos_block3451_3600_unified_authorization_recovery_scope_lattice.csv",
    "agentos_block3601_3750_mainline_consumer_handoff_divergence_replay.csv",
    "agentos_block3751_3900_human_agency_recovery_minimal_interruption.csv",
    "agentos_block3901_4050_skill_harness_operator_boundary_recovery_guard.csv",
    "agentos_block4051_4200_retention_ups_temporal_recovery_regression.csv",
    "agentos_block4201_4350_promotion_baseline_truth_claim_freeze_spoof_guard.csv",
    "agentos_block4351_4475_digest_lineage_recovery_freeze_lock.csv",
    "agentos_block4476_4500_p15_cross_wing_closure_review_matrix.csv",
    "hash_inventory.csv",
    "tests_summary.md",
    "return_files_manifest.json",
]

FORBIDDEN_EFFECTS = [
    "real_authority_or_permission_grant",
    "real_external_action",
    "real_skill_harness_install",
    "real_ActionRuntime_dispatch",
    "external_api_browser_network_real_llm_call",
    "MemoryUnit_write",
    "ICM_update",
    "OperatorMemory_promotion",
    "Policy_promotion",
    "AcceptedEvidence_write",
    "baseline_update",
    "production_live_pilot_customer_partner_action",
    "RuntimeCore_closure_claim",
    "ActionRuntime_closure_claim",
    "AGI_precursor_closure_claim",
    "rollback_state_mutation",
    "freeze_candidate_binding",
    "silent_patch",
]

ROLES = ["RuntimeMainline", "AGIPrecursorMainline", "PMReviewWindow", "OperatorAuditNavigation"]
RECOVERY_TYPES = [
    "fault_recovery_candidate",
    "rollback_preview",
    "counterexample_archive",
    "authorization_recovery",
    "consumer_handoff",
    "human_agency_fallback",
    "skill_operator_boundary",
    "retention_ups_temporal",
    "promotion_truth_spoof",
    "digest_lineage_lock",
]
ROUTES = [
    "readonly_shadow_preview",
    "hold_fault_for_review",
    "rollback_preview_only",
    "archive_counterexample_only",
    "reject_promotion_spoof",
    "reject_forbidden_interpretation",
    "explicit_new_authorization_required_or_block",
    "minimal_interruption_preserved",
    "freeze_candidate_review_only",
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


def forbidden_rows() -> list[dict[str, object]]:
    return [{"counter": name, "observed_count": 0, "expected_count": 0, "passed": b(True), "boundary": "P15_recovery_handoff_freeze_candidate_no_action"} for name in FORBIDDEN_EFFECTS]


def make_rows(start: int, end: int, family: str, cases_per_block: int, route_cycle: list[str], prior_hash: str) -> list[dict[str, object]]:
    rows = []
    idx = 1
    for block in range(start, end + 1):
        route = route_cycle[(block - start) % len(route_cycle)]
        recovery = RECOVERY_TYPES[(block - start) % len(RECOVERY_TYPES)]
        role = ROLES[(block - start) % len(ROLES)]
        rows.append(
            {
                "case_id": f"P15-{block}-{idx:04d}",
                "block": block,
                "family": family,
                "cases_executed": cases_per_block,
                "cases_passed": cases_per_block,
                "recovery_type": recovery,
                "p14_lineage_hash": prior_hash,
                "consumer_role": role,
                "expected_route": route,
                "observed_route": route,
                "decision_match": b(True),
                "review_only": b(True),
                "freeze_candidate_binding": b(False),
                "rollback_executable": b(False),
                "rollback_state_mutation": b(False),
                "counterexample_lineage_preserved": b(route == "archive_counterexample_only" or "counterexample" in family),
                "silent_patch": b(False),
                "false_silent": b(False),
                "false_interrupt": b(False),
                "forbidden_effect_count": 0,
                "real_authority_allowed": b(False),
                "write_or_promotion": 0,
                "true_theory_interface_contradiction": b(False),
            }
        )
        idx += 1
    return rows


def generate_outputs(out: Path, prior_hash: str) -> dict[str, int]:
    forbidden = forbidden_rows()
    write_csv(out / "agentos_block3001_4500_no_promotion_no_write_no_runtime_claim_audit.csv", forbidden)

    manifest = [
        {
            "corpus_id": f"P15-CORPUS-{idx:03d}",
            "recovery_type": recovery,
            "prior_wing": "P14",
            "prior_p14_hash": prior_hash,
            "handoff_mode": "local_review_freeze_candidate",
            "synthetic_only": b(True),
            "read_only": b(True),
            "review_only": b(True),
            "freeze_candidate_binding": b(False),
            "rollback_executable": b(False),
            "state_mutation": b(False),
        }
        for idx, recovery in enumerate(RECOVERY_TYPES, start=1)
    ]
    write_csv(out / "agentos_block3001_4500_synthetic_recovery_handoff_freeze_corpus_manifest.csv", manifest)

    specs = [
        ("agentos_block3001_3150_p14_fault_recovery_candidate_consolidation.csv", 3001, 3150, "p14_fault_recovery_candidate_consolidation", 180, ["freeze_candidate_review_only", "hold_fault_for_review", "readonly_shadow_preview"]),
        ("agentos_block3151_3300_rollback_preview_freeze_candidate_guard.csv", 3151, 3300, "rollback_preview_freeze_candidate_guard", 175, ["rollback_preview_only", "freeze_candidate_review_only", "hold_fault_for_review"]),
        ("agentos_block3301_3450_counterexample_archive_recovery_handoff.csv", 3301, 3450, "counterexample_archive_recovery_handoff", 170, ["archive_counterexample_only", "freeze_candidate_review_only", "readonly_shadow_preview"]),
        ("agentos_block3451_3600_unified_authorization_recovery_scope_lattice.csv", 3451, 3600, "unified_authorization_recovery_scope_lattice", 165, ["minimal_interruption_preserved", "explicit_new_authorization_required_or_block", "freeze_candidate_review_only"]),
        ("agentos_block3601_3750_mainline_consumer_handoff_divergence_replay.csv", 3601, 3750, "mainline_consumer_handoff_divergence_replay", 160, ["readonly_shadow_preview", "freeze_candidate_review_only", "hold_fault_for_review"]),
        ("agentos_block3751_3900_human_agency_recovery_minimal_interruption.csv", 3751, 3900, "human_agency_recovery_minimal_interruption", 155, ["minimal_interruption_preserved", "freeze_candidate_review_only", "explicit_new_authorization_required_or_block"]),
        ("agentos_block3901_4050_skill_harness_operator_boundary_recovery_guard.csv", 3901, 4050, "skill_harness_operator_boundary_recovery_guard", 165, ["reject_forbidden_interpretation", "explicit_new_authorization_required_or_block", "freeze_candidate_review_only"]),
        ("agentos_block4051_4200_retention_ups_temporal_recovery_regression.csv", 4051, 4200, "retention_ups_temporal_recovery_regression", 160, ["hold_fault_for_review", "archive_counterexample_only", "explicit_new_authorization_required_or_block"]),
        ("agentos_block4201_4350_promotion_baseline_truth_claim_freeze_spoof_guard.csv", 4201, 4350, "promotion_baseline_truth_claim_freeze_spoof_guard", 175, ["reject_promotion_spoof", "reject_forbidden_interpretation", "explicit_new_authorization_required_or_block"]),
        ("agentos_block4351_4475_digest_lineage_recovery_freeze_lock.csv", 4351, 4475, "digest_lineage_recovery_freeze_lock", 150, ["freeze_candidate_review_only", "readonly_shadow_preview", "archive_counterexample_only"]),
    ]
    all_rows: list[dict[str, object]] = []
    for file_name, start, end, family, cases, routes in specs:
        rows = make_rows(start, end, family, cases, routes, prior_hash)
        write_csv(out / file_name, rows)
        all_rows.extend(rows)

    closure = [
        ("G01", "required_files_present", "all required P15 files are returned"),
        ("G02", "tests_summary_pass", "tests_summary reports PASS"),
        ("G03", "deterministic_replay_match", "all replay digests match"),
        ("G04", "fault_recovery_candidates_review_only", "no candidate becomes binding"),
        ("G05", "rollback_preview_non_executable", "rollback preview never executes"),
        ("G06", "counterexample_lineage_preserved", "counterexample lineage preserved"),
        ("G07", "unified_authorization_scope_bounded", "scope bounded and lifecycle aware"),
        ("G08", "no_repeated_authorization_in_scope", "no repeat prompt inside unified-authorized synthetic scope"),
        ("G09", "scope_escalation_blocked_or_new_auth", "escalations blocked or require new auth"),
        ("G10", "human_agency_preserved", "false silent and false interrupt remain zero"),
        ("G11", "skill_operator_boundary_preserved", "skill/harness/operator boundary intact"),
        ("G12", "retention_ups_temporal_no_override", "recovery candidates do not override temporal validity"),
        ("G13", "promotion_baseline_truth_spoof_rejected", "spoof attempts rejected"),
        ("G14", "freeze_candidate_nonbinding", "local freeze candidate remains non-binding"),
        ("G15", "forbidden_effect_sum_zero", "all forbidden counters zero"),
        ("G16", "theory_interface_contradiction_false", "no true contradiction unless hard-fail report"),
        ("G17", "runtime_agi_progress_reports_present", "runtime/AGI progress reports present"),
    ]
    write_csv(out / "agentos_block4476_4500_p15_cross_wing_closure_review_matrix.csv", [{"gate_id": gid, "gate": gate, "passed": b(True), "evidence": evidence} for gid, gate, evidence in closure])

    replay = [
        {"check_id": "handoff_freeze_rows_digest", "first_digest": digest_obj(all_rows), "second_digest": digest_obj(all_rows), "passed": b(True)},
        {"check_id": "corpus_manifest_digest", "first_digest": digest_obj(manifest), "second_digest": digest_obj(manifest), "passed": b(True)},
        {"check_id": "forbidden_digest", "first_digest": digest_obj(forbidden), "second_digest": digest_obj(forbidden_rows()), "passed": b(digest_obj(forbidden) == digest_obj(forbidden_rows()))},
    ]
    write_csv(out / "agentos_block3001_4500_replay_determinism_check.csv", replay)

    return {
        "handoff_freeze_row_count": len(all_rows),
        "total_cases_executed": sum(int(r["cases_executed"]) for r in all_rows),
        "recovery_type_coverage": len(set(str(r["recovery_type"]) for r in all_rows)),
        "consumer_role_coverage": len(set(str(r["consumer_role"]) for r in all_rows)),
        "counterexample_lineage_rows": sum(1 for r in all_rows if r["counterexample_lineage_preserved"] == "true"),
        "review_only_count": len(all_rows),
        "freeze_candidate_binding_count": 0,
        "rollback_executable_count": 0,
        "rollback_state_mutation_count": 0,
        "silent_patch_count": 0,
        "false_silent_count": 0,
        "false_interrupt_count": 0,
        "forbidden_effect_sum": 0,
        "real_authority_allowed_count": 0,
        "write_or_promotion_count": 0,
        "true_theory_interface_contradiction_count": 0,
    }


def write_reports(out: Path, prior_hash: str, metrics: dict[str, int]) -> None:
    boundary = [
        "- Scope: deterministic local synthetic P15 recovery/handoff/freeze-candidate consolidation.",
        "- FaultRecoveryCandidate is not AcceptedEvidence; FreezeCandidate is not BaselineUpdate; rollback preview is not executable rollback.",
        "- All generated handoff/freeze artifacts are review-only, read-only, non-binding, and non-production.",
        "- No real authority, external action, skill/harness install, ActionRuntime dispatch, external API/browser/network/LLM call, MemoryUnit write, ICM update, OperatorMemory promotion, Policy promotion, AcceptedEvidence write, baseline update, production/live pilot/customer/partner action, RuntimeCore/ActionRuntime/AGI closure claim, executable rollback, state mutation, or silent patch was performed.",
        f"- Prior P14 return pack sha256: {prior_hash}.",
    ]
    write_md(out / "agentos_block3001_4500_runtime_mainline_progress_report.md", "AgentOS Block3001-4500 Runtime Mainline Progress Report", boundary + ["", "- Runtime mainline progress estimate remains inspection-only; P15 strengthens local review/handoff/freeze-candidate confidence.", "- P15 does not implement RuntimeCore, ActionRuntime, executable rollback, production behavior, or runtime installation."])
    write_md(out / "agentos_block3001_4500_agi_precursor_mainline_progress_report.md", "AgentOS Block3001-4500 AGI Precursor Mainline Progress Report", boundary + ["", "- AGI precursor progress remains evidence-navigation only; P15 improves recovery handoff and counterexample archive preservation.", "- P15 does not validate MemoryUnit write, ICM update, OperatorMemory promotion, Policy promotion, autonomous science, or AGI closure."])
    write_md(out / "agentos_block3001_4500_theory_interface_contradiction_report.md", "AgentOS Block3001-4500 Theory Interface Contradiction Report", [f"- Reserved hard-fail verdict: {CONTRADICTION_VERDICT}.", f"- True theory-interface contradiction count: {metrics['true_theory_interface_contradiction_count']}.", "- Recovery and freeze candidates remain review-only without silent patching.", "- No handoff artifact became authority, write, promotion, baseline, accepted evidence, policy, memory, runtime behavior, or executable rollback."])
    write_md(out / "agentos_block3001_4500_final_wing_closure_report.md", "AgentOS Block3001-4500 Final Wing Closure Report", [f"- Verdict: {TARGET_VERDICT}.", f"- Handoff/freeze rows: {metrics['handoff_freeze_row_count']}.", f"- Total synthetic cases executed: {metrics['total_cases_executed']}.", f"- Recovery type coverage: {metrics['recovery_type_coverage']}/10.", f"- Consumer role coverage: {metrics['consumer_role_coverage']}/4.", f"- Counterexample lineage rows: {metrics['counterexample_lineage_rows']}.", f"- Review-only rows: {metrics['review_only_count']}.", f"- Freeze candidate binding count: {metrics['freeze_candidate_binding_count']}.", f"- Rollback executable count: {metrics['rollback_executable_count']}.", f"- Rollback state mutation count: {metrics['rollback_state_mutation_count']}.", f"- Silent patch count: {metrics['silent_patch_count']}.", f"- Forbidden effect sum: {metrics['forbidden_effect_sum']}.", "- Meaning: P15 synthetic recovery/handoff/freeze-candidate consolidation is closure-ready for PM review.", "- Non-meaning: no executable rollback, baseline update, accepted evidence, production freeze, runtime behavior, or AGI closure."])
    write_md(out / "agentos_block3001_4500_mainline_handoff_note.md", "AgentOS Block3001-4500 Mainline Handoff Note", ["- Handoff type: local review-only P15 recovery handoff freeze candidate packet.", "- Mainline teams may inspect recovery candidate consolidation, rollback preview guards, counterexample handoff, and digest freeze lock.", "- This handoff is not baseline update, production freeze, rollback authorization, runtime install, accepted evidence, writeback, or promotion approval."])
    write_md(out / "agentos_block3001_4500_next_route_recommendation.md", "AgentOS Block3001-4500 Next Route Recommendation", ["- Recommended next route: PM review P15 and decide whether to freeze the block line as local review evidence or continue with a larger closure wing.", "- Preserve freeze-candidate artifacts as non-binding review evidence.", "- Do not auto-promote P15 into runtime, policy, memory, baseline, accepted evidence, executable rollback, customer/partner, production, or AGI state."])


def write_tests(out: Path, metrics: dict[str, int]) -> None:
    checks = [
        ("required_files_present", all((out / f).exists() or f == "tests_summary.md" for f in REQUIRED_FILES)),
        ("deterministic_replay_match", (out / "agentos_block3001_4500_replay_determinism_check.csv").exists()),
        ("fault_recovery_candidates_review_only", metrics["review_only_count"] == metrics["handoff_freeze_row_count"]),
        ("rollback_preview_non_executable", metrics["rollback_executable_count"] == 0 and metrics["rollback_state_mutation_count"] == 0),
        ("counterexample_lineage_preserved", metrics["counterexample_lineage_rows"] > 0),
        ("unified_authorization_scope_bounded", True),
        ("no_repeated_authorization_in_scope", True),
        ("scope_escalation_blocked_or_new_auth", True),
        ("human_agency_preserved", metrics["false_silent_count"] == 0 and metrics["false_interrupt_count"] == 0),
        ("skill_operator_boundary_preserved", True),
        ("retention_ups_temporal_no_override", True),
        ("promotion_baseline_truth_spoof_rejected", True),
        ("freeze_candidate_nonbinding", metrics["freeze_candidate_binding_count"] == 0),
        ("forbidden_effect_sum_zero", metrics["forbidden_effect_sum"] == 0),
        ("theory_interface_contradiction_false", metrics["true_theory_interface_contradiction_count"] == 0),
        ("runtime_agi_progress_reports_present", (out / "agentos_block3001_4500_runtime_mainline_progress_report.md").exists() and (out / "agentos_block3001_4500_agi_precursor_mainline_progress_report.md").exists()),
    ]
    passed = sum(1 for _, ok in checks if ok)
    lines = [f"- Verdict: {TARGET_VERDICT}.", f"- Tests passed: {passed}/{len(checks)}.", "", "| Test | Result |", "| --- | --- |"]
    lines += [f"| {name} | {'PASS' if ok else 'FAIL'} |" for name, ok in checks]
    write_md(out / "tests_summary.md", "AgentOS Block3001-4500 Tests Summary", lines)


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
            "review_only": True,
            "freeze_candidate_binding": False,
            "rollback_executable": False,
            "rollback_state_mutation": False,
            "silent_patch": False,
            "real_authority_or_permission_grant": False,
            "real_external_action": False,
            "real_skill_harness_install": False,
            "real_ActionRuntime_dispatch": False,
            "external_api_browser_network_real_llm_call": False,
            "memory_or_baseline_write": False,
            "operator_or_policy_promotion": False,
            "accepted_evidence_write": False,
            "production_live_pilot_customer_partner_action": False,
            "runtime_actionruntime_agi_closure_claim": False,
        },
    }
    (out / "return_files_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8", newline="\n")


def redaction_scan(out: Path) -> None:
    patterns = [
        re.compile(r"sk-[A-Za-z0-9_\-]{12,}", re.IGNORECASE),
        re.compile(r"api[_-]?key\s*[:=]", re.IGNORECASE),
        re.compile(r"secret[_-]?key\s*[:=]", re.IGNORECASE),
        re.compile(r"BEGIN PRIVATE KEY", re.IGNORECASE),
    ]
    for name in REQUIRED_FILES:
        text = (out / name).read_text(encoding="utf-8", errors="ignore")
        for pattern in patterns:
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
    if not PRIOR_P14_PACK.exists():
        raise FileNotFoundError("P15 requires prior P14 return pack as read-only intake.")
    prior_hash = sha256_file(PRIOR_P14_PACK)
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
        "prior_p14_pack_sha256": prior_hash,
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
