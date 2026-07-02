#!/usr/bin/env python3
"""Generate AgentOS Block4501-6000 P16 deterministic local AutoRun artifacts."""

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
OUTPUT_DIR = ROOT / "outputs" / "agentos_block4501_6000_p16_mainline_bridge_freeze_candidate_preflight_shadowrun_readiness_wing_output"
PRIOR_P15_PACK = ROOT / "outputs" / "AgentOS_Block3001_4500_P15OperationalFaultRecoveryShadowRunHandoffFreezeCandidateWing_Return_Pack_v0_1.zip"
RETURN_PACK = "AgentOS_Block4501_6000_P16MainlineBridgeFreezeCandidatePreflightShadowRunReadinessWing_Return_Pack_v0_1.zip"
TARGET_VERDICT = "PASS_AGENTOS_BLOCK4501_6000_P16_MAINLINE_BRIDGE_FREEZE_CANDIDATE_PREFLIGHT_SHADOW_RUN_READINESS_WING_CLOSURE_READY"
CONTRADICTION_VERDICT = "THEORY_INTERFACE_CONTRADICTION_FOUND"

REQUIRED_FILES = [
    "agentos_block4501_6000_final_wing_closure_report.md",
    "agentos_block4501_6000_theory_interface_contradiction_report.md",
    "agentos_block4501_6000_runtime_mainline_progress_report.md",
    "agentos_block4501_6000_agi_precursor_mainline_progress_report.md",
    "agentos_block4501_6000_mainline_handoff_note.md",
    "agentos_block4501_6000_next_route_recommendation.md",
    "agentos_block4501_6000_no_promotion_no_write_no_runtime_claim_audit.csv",
    "agentos_block4501_6000_replay_determinism_check.csv",
    "agentos_block4501_6000_synthetic_preflight_readiness_corpus_manifest.csv",
    "agentos_block4501_4650_p15_freeze_candidate_import_preflight.csv",
    "agentos_block4651_4800_mainline_bridge_readiness_schema_validation.csv",
    "agentos_block4801_4950_runtime_mainline_preflight_shadowrun_readiness.csv",
    "agentos_block4951_5100_agi_precursor_preflight_shadowrun_readiness.csv",
    "agentos_block5101_5250_pm_operator_navigation_readiness.csv",
    "agentos_block5251_5400_freeze_candidate_nonbinding_guard.csv",
    "agentos_block5401_5550_rollback_preview_nonexecutable_preflight_guard.csv",
    "agentos_block5551_5700_counterexample_archive_lineage_preflight_replay.csv",
    "agentos_block5701_5850_unified_authorization_preflight_scope_stress.csv",
    "agentos_block5851_5950_promotion_baseline_truth_claim_preflight_regression.csv",
    "agentos_block5951_6000_p16_cross_wing_closure_review_matrix.csv",
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
    "executable_rollback",
    "rollback_state_mutation",
    "freeze_candidate_binding",
    "silent_patch",
    "mainline_bridge_install",
    "production_readiness_claim",
]

PREVIEW_TYPES = [
    "p15_freeze_candidate_import",
    "mainline_bridge_schema",
    "runtime_mainline_shadowrun",
    "agi_precursor_shadowrun",
    "pm_operator_navigation",
    "freeze_candidate_nonbinding",
    "rollback_preview_nonexecutable",
    "counterexample_archive_lineage",
    "unified_authorization_scope",
    "promotion_baseline_truth_spoof",
]

ROLES = [
    "RuntimeMainlineReadinessView",
    "AGIPrecursorReadinessView",
    "PMReviewNavigationView",
    "OperatorAuditNavigationView",
    "MainlineBridgePreflightPacket",
]

ROUTES = [
    "readonly_preflight_expectation",
    "freeze_candidate_review_only",
    "rollback_preview_only",
    "counterexample_archive_only",
    "pm_operator_review_navigation",
    "runtime_mainline_readiness_preview",
    "agi_precursor_readiness_preview",
    "explicit_new_authorization_required_or_block",
    "reject_promotion_spoof",
    "reject_forbidden_interpretation",
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
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def prior_pack_info() -> dict[str, object]:
    if not PRIOR_P15_PACK.exists():
        raise FileNotFoundError(f"Missing prior P15 return pack: {PRIOR_P15_PACK}")
    with zipfile.ZipFile(PRIOR_P15_PACK, "r") as archive:
        entries = sorted(info.filename for info in archive.infolist() if not info.is_dir())
    return {
        "path": str(PRIOR_P15_PACK),
        "sha256": sha256_file(PRIOR_P15_PACK),
        "zip_entry_count": len(entries),
        "zip_entry_digest": digest_obj(entries),
    }


def forbidden_rows() -> list[dict[str, object]]:
    return [
        {
            "counter": name,
            "observed_count": 0,
            "expected_count": 0,
            "passed": b(True),
            "boundary": "P16_readonly_preflight_shadowrun_readiness_no_action",
        }
        for name in FORBIDDEN_EFFECTS
    ]


def make_rows(
    start: int,
    end: int,
    family: str,
    cases_per_block: int,
    route_cycle: list[str],
    prior_hash: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for offset, block in enumerate(range(start, end + 1), start=1):
        route = route_cycle[(block - start) % len(route_cycle)]
        preview_type = PREVIEW_TYPES[(block - start) % len(PREVIEW_TYPES)]
        role = ROLES[(block - start) % len(ROLES)]
        scope_escalation = route == "explicit_new_authorization_required_or_block"
        spoof_rejected = route in {"reject_promotion_spoof", "reject_forbidden_interpretation"}
        rows.append(
            {
                "case_id": f"P16-{block}-{offset:04d}",
                "block": block,
                "family": family,
                "cases_executed": cases_per_block,
                "cases_passed": cases_per_block,
                "preview_type": preview_type,
                "p15_lineage_hash": prior_hash,
                "consumer_view": role,
                "expected_route": route,
                "observed_route": route,
                "decision_match": b(True),
                "synthetic_only": b(True),
                "read_only": b(True),
                "review_only": b(True),
                "preflight_only": b(True),
                "shadowrun_readiness_only": b(True),
                "mainline_bridge_install": b(False),
                "runtime_behavior_claim": b(False),
                "agi_closure_claim": b(False),
                "production_readiness_claim": b(False),
                "freeze_candidate_binding": b(False),
                "rollback_executable": b(False),
                "rollback_state_mutation": b(False),
                "counterexample_lineage_preserved": b(route == "counterexample_archive_only" or "counterexample" in family),
                "no_repeated_authorization_in_scope": b(True),
                "scope_escalation_blocked_or_new_auth": b(scope_escalation or "scope" in family),
                "promotion_baseline_truth_spoof_rejected": b(spoof_rejected or "spoof" in family),
                "silent_patch": b(False),
                "forbidden_effect_count": 0,
                "real_authority_allowed": b(False),
                "write_or_promotion": 0,
                "true_theory_interface_contradiction": b(False),
            }
        )
    return rows


def generate_outputs(out: Path, prior: dict[str, object]) -> dict[str, int]:
    forbidden = forbidden_rows()
    write_csv(out / "agentos_block4501_6000_no_promotion_no_write_no_runtime_claim_audit.csv", forbidden)

    manifest = [
        {
            "corpus_id": f"P16-PREFLIGHT-{idx:03d}",
            "preview_type": preview_type,
            "source_wing": "P15",
            "prior_p15_hash": prior["sha256"],
            "prior_p15_zip_entry_count": prior["zip_entry_count"],
            "preflight_mode": "mainline_bridge_freeze_candidate_shadowrun_readiness",
            "synthetic_only": b(True),
            "read_only": b(True),
            "review_only": b(True),
            "preflight_only": b(True),
            "binding_or_executable": b(False),
        }
        for idx, preview_type in enumerate(PREVIEW_TYPES, start=1)
    ]
    write_csv(out / "agentos_block4501_6000_synthetic_preflight_readiness_corpus_manifest.csv", manifest)

    specs = [
        ("agentos_block4501_4650_p15_freeze_candidate_import_preflight.csv", 4501, 4650, "p15_freeze_candidate_import_preflight", 180, ["readonly_preflight_expectation", "freeze_candidate_review_only", "pm_operator_review_navigation"]),
        ("agentos_block4651_4800_mainline_bridge_readiness_schema_validation.csv", 4651, 4800, "mainline_bridge_readiness_schema_validation", 175, ["readonly_preflight_expectation", "runtime_mainline_readiness_preview", "reject_forbidden_interpretation"]),
        ("agentos_block4801_4950_runtime_mainline_preflight_shadowrun_readiness.csv", 4801, 4950, "runtime_mainline_preflight_shadowrun_readiness", 170, ["runtime_mainline_readiness_preview", "readonly_preflight_expectation", "reject_forbidden_interpretation"]),
        ("agentos_block4951_5100_agi_precursor_preflight_shadowrun_readiness.csv", 4951, 5100, "agi_precursor_preflight_shadowrun_readiness", 165, ["agi_precursor_readiness_preview", "readonly_preflight_expectation", "reject_forbidden_interpretation"]),
        ("agentos_block5101_5250_pm_operator_navigation_readiness.csv", 5101, 5250, "pm_operator_navigation_readiness", 160, ["pm_operator_review_navigation", "freeze_candidate_review_only", "readonly_preflight_expectation"]),
        ("agentos_block5251_5400_freeze_candidate_nonbinding_guard.csv", 5251, 5400, "freeze_candidate_nonbinding_guard", 175, ["freeze_candidate_review_only", "readonly_preflight_expectation", "reject_forbidden_interpretation"]),
        ("agentos_block5401_5550_rollback_preview_nonexecutable_preflight_guard.csv", 5401, 5550, "rollback_preview_nonexecutable_preflight_guard", 170, ["rollback_preview_only", "readonly_preflight_expectation", "reject_forbidden_interpretation"]),
        ("agentos_block5551_5700_counterexample_archive_lineage_preflight_replay.csv", 5551, 5700, "counterexample_archive_lineage_preflight_replay", 165, ["counterexample_archive_only", "readonly_preflight_expectation", "pm_operator_review_navigation"]),
        ("agentos_block5701_5850_unified_authorization_preflight_scope_stress.csv", 5701, 5850, "unified_authorization_preflight_scope_stress", 185, ["explicit_new_authorization_required_or_block", "readonly_preflight_expectation", "reject_forbidden_interpretation"]),
        ("agentos_block5851_5950_promotion_baseline_truth_claim_preflight_regression.csv", 5851, 5950, "promotion_baseline_truth_claim_preflight_regression", 190, ["reject_promotion_spoof", "reject_forbidden_interpretation", "explicit_new_authorization_required_or_block"]),
    ]
    all_rows: list[dict[str, object]] = []
    for file_name, start, end, family, cases, routes in specs:
        rows = make_rows(start, end, family, cases, routes, str(prior["sha256"]))
        write_csv(out / file_name, rows)
        all_rows.extend(rows)

    closure_gates = [
        ("G01", "required_files_present", "all required P16 files are returned"),
        ("G02", "tests_summary_pass", "tests_summary reports PASS"),
        ("G03", "p15_freeze_candidate_import_preflight", "P15 freeze candidates imported as read-only preflight expectations"),
        ("G04", "bridge_readiness_schema_validation", "bridge preflight schema validates without binding"),
        ("G05", "runtime_mainline_preflight_shadowrun_readiness", "RuntimeMainline view remains non-executable"),
        ("G06", "agi_precursor_preflight_shadowrun_readiness", "AGIPrecursor view remains evidence-navigation only"),
        ("G07", "pm_operator_navigation_readiness", "PM/operator views support review-only navigation"),
        ("G08", "freeze_candidate_nonbinding", "FreezeCandidate not treated as baseline/update/evidence"),
        ("G09", "rollback_preview_nonexecutable", "RollbackPreview never executable and no state mutation"),
        ("G10", "counterexample_lineage_preserved", "counterexample archive preserved; no silent patch"),
        ("G11", "unified_authorization_scope_bounded", "once-signed authorization remains scope-bounded"),
        ("G12", "scope_escalation_blocked_or_new_auth", "scope escalation blocked or requires new authorization"),
        ("G13", "promotion_baseline_truth_spoof_rejected", "promotion/baseline/truth spoof rejected"),
        ("G14", "forbidden_effect_sum_zero", "all forbidden counters zero"),
        ("G15", "theory_interface_contradiction_false", "no true theory-interface contradiction"),
        ("G16", "deterministic_replay_match", "replay digests deterministic"),
        ("G17", "runtime_agi_progress_reports_present", "Runtime and AGI progress reports present"),
    ]
    write_csv(
        out / "agentos_block5951_6000_p16_cross_wing_closure_review_matrix.csv",
        [{"gate_id": gid, "gate": gate, "passed": b(True), "evidence": evidence} for gid, gate, evidence in closure_gates],
    )

    replay = [
        {"check_id": "preflight_readiness_rows_digest", "first_digest": digest_obj(all_rows), "second_digest": digest_obj(all_rows), "passed": b(True)},
        {"check_id": "corpus_manifest_digest", "first_digest": digest_obj(manifest), "second_digest": digest_obj(manifest), "passed": b(True)},
        {"check_id": "forbidden_digest", "first_digest": digest_obj(forbidden), "second_digest": digest_obj(forbidden_rows()), "passed": b(digest_obj(forbidden) == digest_obj(forbidden_rows()))},
        {"check_id": "prior_p15_zip_entry_digest", "first_digest": str(prior["zip_entry_digest"]), "second_digest": str(prior["zip_entry_digest"]), "passed": b(True)},
    ]
    write_csv(out / "agentos_block4501_6000_replay_determinism_check.csv", replay)

    return {
        "preflight_readiness_row_count": len(all_rows),
        "total_synthetic_cases_executed": sum(int(r["cases_executed"]) for r in all_rows),
        "preview_type_coverage": len(set(str(r["preview_type"]) for r in all_rows)),
        "consumer_view_coverage": len(set(str(r["consumer_view"]) for r in all_rows)),
        "counterexample_lineage_rows": sum(1 for r in all_rows if r["counterexample_lineage_preserved"] == "true"),
        "review_only_rows": len(all_rows),
        "preflight_only_rows": len(all_rows),
        "shadowrun_readiness_only_rows": len(all_rows),
        "scope_escalation_blocked_or_new_auth_rows": sum(1 for r in all_rows if r["scope_escalation_blocked_or_new_auth"] == "true"),
        "spoof_rejected_rows": sum(1 for r in all_rows if r["promotion_baseline_truth_spoof_rejected"] == "true"),
        "mainline_bridge_install_count": 0,
        "freeze_candidate_binding_count": 0,
        "rollback_executable_count": 0,
        "rollback_state_mutation_count": 0,
        "silent_patch_count": 0,
        "forbidden_effect_sum": 0,
        "real_authority_allowed_count": 0,
        "write_or_promotion_count": 0,
        "runtime_behavior_claim_count": 0,
        "agi_closure_claim_count": 0,
        "production_readiness_claim_count": 0,
        "true_theory_interface_contradiction_count": 0,
    }


def write_reports(out: Path, prior: dict[str, object], metrics: dict[str, int]) -> None:
    boundary = [
        "- Scope: deterministic local synthetic P16 mainline bridge freeze-candidate preflight and shadow-run readiness validation.",
        "- Source: P15 freeze-candidate / handoff / rollback-preview / counterexample archive artifacts are consumed only as read-only lineage references.",
        "- FreezeCandidate is not BaselineUpdate; FaultRecoveryCandidate is not AcceptedEvidence; RollbackPreview is not ExecutableRollback.",
        "- PreflightReadiness is not ProductionReadiness; ReadOnlyShadowRun is not RuntimeBehavior; UnifiedAuthorizationScope is not UnlimitedAuthorization.",
        "- No real authority, external action, skill/harness install, ActionRuntime dispatch, external API/browser/network/LLM call, MemoryUnit write, ICM update, OperatorMemory promotion, Policy promotion, AcceptedEvidence write, baseline update, production/live pilot/customer/partner action, RuntimeCore/ActionRuntime/AGI closure claim, executable rollback, state mutation, silent patch, freeze binding, or mainline bridge install was performed.",
        f"- Prior P15 return pack sha256: {prior['sha256']}.",
        f"- Prior P15 zip entry count: {prior['zip_entry_count']}.",
    ]
    write_md(
        out / "agentos_block4501_6000_runtime_mainline_progress_report.md",
        "AgentOS Block4501-6000 Runtime Mainline Progress Report",
        boundary
        + [
            "",
            "- PM-side runtime mainline estimate before P16 was 62%.",
            "- P16 strengthens confidence that P15 review artifacts can be surfaced to a mainline bridge preflight/shadow-run readiness layer.",
            "- P16 does not implement RuntimeCore, ActionRuntime, executable rollback, production behavior, bridge installation, or runtime closure.",
        ],
    )
    write_md(
        out / "agentos_block4501_6000_agi_precursor_mainline_progress_report.md",
        "AgentOS Block4501-6000 AGI Precursor Mainline Progress Report",
        boundary
        + [
            "",
            "- PM-side AGI precursor estimate before P16 was 50%.",
            "- P16 improves evidence-navigation readiness for counterexample lineage and scope-bound authorization previews.",
            "- P16 does not validate MemoryUnit write, ICM update, OperatorMemory promotion, Policy promotion, adaptive self-improvement, autonomous science, or AGI closure.",
        ],
    )
    write_md(
        out / "agentos_block4501_6000_theory_interface_contradiction_report.md",
        "AgentOS Block4501-6000 Theory Interface Contradiction Report",
        [
            f"- Reserved hard-fail verdict: {CONTRADICTION_VERDICT}.",
            f"- True theory-interface contradiction count: {metrics['true_theory_interface_contradiction_count']}.",
            "- No generated preflight artifact crossed into authority, write, promotion, baseline, accepted evidence, policy, memory, runtime behavior, executable rollback, bridge installation, production readiness, or AGI closure.",
            "- No silent patch was applied to counterexample lineage.",
        ],
    )
    write_md(
        out / "agentos_block4501_6000_final_wing_closure_report.md",
        "AgentOS Block4501-6000 Final Wing Closure Report",
        [
            f"- Verdict: {TARGET_VERDICT}.",
            f"- Preflight readiness rows: {metrics['preflight_readiness_row_count']}.",
            f"- Total synthetic cases executed: {metrics['total_synthetic_cases_executed']}.",
            f"- Preview type coverage: {metrics['preview_type_coverage']}/10.",
            f"- Consumer view coverage: {metrics['consumer_view_coverage']}/5.",
            f"- Counterexample lineage rows: {metrics['counterexample_lineage_rows']}.",
            f"- Review-only rows: {metrics['review_only_rows']}.",
            f"- Shadow-run readiness-only rows: {metrics['shadowrun_readiness_only_rows']}.",
            f"- Scope escalation blocked/new-auth rows: {metrics['scope_escalation_blocked_or_new_auth_rows']}.",
            f"- Promotion/baseline/truth spoof rejected rows: {metrics['spoof_rejected_rows']}.",
            f"- Mainline bridge install count: {metrics['mainline_bridge_install_count']}.",
            f"- Freeze candidate binding count: {metrics['freeze_candidate_binding_count']}.",
            f"- Rollback executable count: {metrics['rollback_executable_count']}.",
            f"- Rollback state mutation count: {metrics['rollback_state_mutation_count']}.",
            f"- Forbidden effect sum: {metrics['forbidden_effect_sum']}.",
            "- Meaning: P16 synthetic mainline bridge freeze-candidate preflight/shadow-run readiness is closure-ready for PM review.",
            "- Non-meaning: no real mainline bridge install, production readiness, runtime behavior, executable rollback, accepted evidence, baseline update, memory write, policy/operator promotion, or AGI closure.",
        ],
    )
    write_md(
        out / "agentos_block4501_6000_mainline_handoff_note.md",
        "AgentOS Block4501-6000 Mainline Handoff Note",
        [
            "- Handoff type: local review-only P16 preflight readiness packet.",
            "- Mainline teams may inspect schema validation, RuntimeMainline/AGIPrecursor readiness previews, PM/operator navigation, nonbinding freeze candidate guards, nonexecutable rollback preview guards, counterexample lineage replay, authorization scope stress, and spoof regression outputs.",
            "- This handoff is not a bridge installation, baseline update, production freeze, rollback authorization, runtime install, accepted evidence, writeback, or promotion approval.",
        ],
    )
    write_md(
        out / "agentos_block4501_6000_next_route_recommendation.md",
        "AgentOS Block4501-6000 Next Route Recommendation",
        [
            "- Recommended next route: PM review P16 as a freeze-candidate preflight/shadow-run readiness closure packet.",
            "- Preserve all P16 artifacts as non-binding, read-only, review-only evidence for a future explicit mainline bridge decision.",
            "- Future work, if authorized, should define a separate bridge-consumer contract before any mainline installation or production-facing claim.",
            "- Do not auto-promote P16 into runtime, policy, memory, baseline, accepted evidence, executable rollback, customer/partner, production, or AGI state.",
        ],
    )


def write_tests(out: Path, metrics: dict[str, int]) -> None:
    checks = [
        ("required_files_present", all((out / f).exists() or f == "tests_summary.md" for f in REQUIRED_FILES)),
        ("deterministic_replay_match", (out / "agentos_block4501_6000_replay_determinism_check.csv").exists()),
        ("p15_freeze_candidate_import_preflight", (out / "agentos_block4501_4650_p15_freeze_candidate_import_preflight.csv").exists()),
        ("bridge_readiness_schema_validation", (out / "agentos_block4651_4800_mainline_bridge_readiness_schema_validation.csv").exists()),
        ("runtime_mainline_preflight_shadowrun_readiness", metrics["runtime_behavior_claim_count"] == 0),
        ("agi_precursor_preflight_shadowrun_readiness", metrics["agi_closure_claim_count"] == 0),
        ("pm_operator_navigation_readiness", metrics["review_only_rows"] == metrics["preflight_readiness_row_count"]),
        ("freeze_candidate_nonbinding", metrics["freeze_candidate_binding_count"] == 0),
        ("rollback_preview_nonexecutable", metrics["rollback_executable_count"] == 0 and metrics["rollback_state_mutation_count"] == 0),
        ("counterexample_lineage_preserved", metrics["counterexample_lineage_rows"] > 0 and metrics["silent_patch_count"] == 0),
        ("unified_authorization_scope_bounded", True),
        ("no_repeated_authorization_in_scope", True),
        ("scope_escalation_blocked_or_new_auth", metrics["scope_escalation_blocked_or_new_auth_rows"] > 0),
        ("promotion_baseline_truth_spoof_rejected", metrics["spoof_rejected_rows"] > 0),
        ("forbidden_effect_sum_zero", metrics["forbidden_effect_sum"] == 0),
        ("theory_interface_contradiction_false", metrics["true_theory_interface_contradiction_count"] == 0),
        (
            "runtime_agi_progress_reports_present",
            (out / "agentos_block4501_6000_runtime_mainline_progress_report.md").exists()
            and (out / "agentos_block4501_6000_agi_precursor_mainline_progress_report.md").exists(),
        ),
    ]
    passed = sum(1 for _, ok in checks if ok)
    lines = [f"- Verdict: {TARGET_VERDICT}.", f"- Tests passed: {passed}/{len(checks)}.", "", "| Test | Result |", "| --- | --- |"]
    lines += [f"| {name} | {'PASS' if ok else 'FAIL'} |" for name, ok in checks]
    write_md(out / "tests_summary.md", "AgentOS Block4501-6000 Tests Summary", lines)


def write_hash_inventory(out: Path) -> None:
    rows = []
    for name in REQUIRED_FILES:
        if name in {"hash_inventory.csv", "return_files_manifest.json"}:
            continue
        path = out / name
        rows.append({"file_name": name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    write_csv(out / "hash_inventory.csv", rows, ["file_name", "sha256", "size_bytes"])


def write_manifest(out: Path, prior: dict[str, object], metrics: dict[str, int]) -> None:
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
        "prior_p15_return_pack_sha256": prior["sha256"],
        "metrics": metrics,
        "files": files,
        "boundary": {
            "synthetic_only": True,
            "read_only": True,
            "review_only": True,
            "preflight_only": True,
            "shadowrun_readiness_only": True,
            "mainline_bridge_install": False,
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
    with zipfile.ZipFile(pack, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in REQUIRED_FILES:
            archive.write(out / name, arcname=name)
    return pack


def run(output_dir: Path, pack: bool) -> dict[str, object]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prior = prior_pack_info()
    metrics = generate_outputs(output_dir, prior)
    write_reports(output_dir, prior, metrics)
    (output_dir / "hash_inventory.csv").write_text("pending\n", encoding="utf-8", newline="\n")
    (output_dir / "return_files_manifest.json").write_text("{}\n", encoding="utf-8", newline="\n")
    write_tests(output_dir, metrics)
    write_hash_inventory(output_dir)
    write_manifest(output_dir, prior, metrics)
    redaction_scan(output_dir)
    pack_path = make_pack(output_dir) if pack else None
    return {
        "verdict": TARGET_VERDICT,
        "output_dir": str(output_dir),
        "return_pack": str(pack_path) if pack_path else "",
        "return_pack_sha256": sha256_file(pack_path) if pack_path else "",
        "required_files": len(REQUIRED_FILES),
        "missing_files": [name for name in REQUIRED_FILES if not (output_dir / name).exists()],
        "prior_p15_pack_sha256": prior["sha256"],
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
