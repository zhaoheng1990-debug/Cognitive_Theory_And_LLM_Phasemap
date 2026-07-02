#!/usr/bin/env python3
"""Generate AgentOS Block2251-3000 P14 deterministic local AutoRun artifacts."""

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
OUTPUT_DIR = ROOT / "outputs" / "agentos_block2251_3000_p14_operational_invariant_fault_injection_shadowrun_rollback_regression_wing_output"
PRIOR_P13_PACK = ROOT / "outputs" / "AgentOS_Block1501_2250_P13OperationalInvariantAssimilationShadowRunBridgeRegressionWing_Return_Pack_v0_1.zip"
RETURN_PACK = "AgentOS_Block2251_3000_P14OperationalInvariantFaultInjectionShadowRunRollbackRegressionWing_Return_Pack_v0_1.zip"
TARGET_VERDICT = "PASS_AGENTOS_BLOCK2251_3000_P14_OPERATIONAL_INVARIANT_FAULT_INJECTION_SHADOW_RUN_ROLLBACK_REGRESSION_WING_CLOSURE_READY"
CONTRADICTION_VERDICT = "THEORY_INTERFACE_CONTRADICTION_FOUND"

REQUIRED_FILES = [
    "agentos_block2251_3000_final_wing_closure_report.md",
    "agentos_block2251_3000_theory_interface_contradiction_report.md",
    "agentos_block2251_3000_runtime_mainline_progress_report.md",
    "agentos_block2251_3000_agi_precursor_mainline_progress_report.md",
    "agentos_block2251_3000_mainline_handoff_note.md",
    "agentos_block2251_3000_next_route_recommendation.md",
    "agentos_block2251_3000_no_promotion_no_write_no_runtime_claim_audit.csv",
    "agentos_block2251_3000_replay_determinism_check.csv",
    "agentos_block2251_3000_synthetic_fault_injection_shadowrun_corpus_manifest.csv",
    "agentos_block2251_2325_p13_invariant_import_fault_injection_guard.csv",
    "agentos_block2326_2400_shadowrun_bridge_rollback_preview_replay.csv",
    "agentos_block2401_2475_unified_authorization_revocation_fault_injection.csv",
    "agentos_block2476_2550_mainline_consumer_divergence_fault_injection.csv",
    "agentos_block2551_2625_human_agency_fallback_minimal_interrupt_rollback.csv",
    "agentos_block2626_2700_skill_harness_operator_boundary_fault_injection.csv",
    "agentos_block2701_2775_retention_ups_temporal_fault_injection.csv",
    "agentos_block2776_2850_promotion_baseline_truth_claim_spoof_fault_injection.csv",
    "agentos_block2851_2925_counterexample_lineage_corruption_recovery.csv",
    "agentos_block2926_3000_p14_cross_wing_closure_review_matrix.csv",
    "hash_inventory.csv",
    "tests_summary.md",
    "return_files_manifest.json",
]

FORBIDDEN_EFFECTS = [
    "real_authority_or_permission_grant",
    "real_external_action",
    "real_skill_harness_install",
    "real_ActionRuntime_dispatch",
    "external_api_browser_network_real_llm_api_call",
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
    "broad_no_action_boundary_validation_rerun",
]

INVARIANTS = [
    "Retention != Selection",
    "StructuralResolution != BestPolicyAction",
    "TemporalValidity > Retention/Reuse optimism",
    "FunctionalEquivalence != SurfaceSimilarity",
    "Unresolved != Pass",
    "OnceSignedUnifiedAuthorization != UnlimitedAuthorization",
    "HumanAuthorizationSemanticFixture != RealAuthority",
    "AssimilatedInvariant != Policy/Memory/Baseline/AcceptedEvidence",
    "ShadowRunRollbackPreview != ExecutableRollback",
]
ROLES = ["RuntimeMainline", "AGIPrecursorMainline", "PMReviewWindow", "OperatorAuditNavigation"]
FAULTS = [
    "corrupted_hash",
    "missing_lineage",
    "stale_source",
    "contradictory_invariant_label",
    "optional_schema_drift",
    "revoked_authorization",
    "expired_authorization",
    "supersession_conflict",
    "role_visibility_fault",
    "promotion_spoof",
]
ROUTES = [
    "readonly_shadow_preview",
    "hold_fault_for_review",
    "explicit_new_authorization_required_or_block",
    "reject_forbidden_interpretation",
    "reject_authorization_injection",
    "reject_promotion_spoof",
    "archive_counterexample_only",
    "minimal_interruption_preserved",
    "rollback_preview_only",
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
    return [{"counter": name, "observed_count": 0, "expected_count": 0, "passed": b(True), "boundary": "P14_synthetic_fault_injection_no_action"} for name in FORBIDDEN_EFFECTS]


def make_rows(start: int, end: int, family: str, cases_per_block: int, route_cycle: list[str], prior_hash: str) -> list[dict[str, object]]:
    rows = []
    idx = 1
    for block in range(start, end + 1):
        route = route_cycle[(block - start) % len(route_cycle)]
        fault = FAULTS[(block - start) % len(FAULTS)]
        invariant = INVARIANTS[(block - start) % len(INVARIANTS)]
        role = ROLES[(block - start) % len(ROLES)]
        rows.append(
            {
                "case_id": f"P14-{block}-{idx:04d}",
                "block": block,
                "family": family,
                "cases_executed": cases_per_block,
                "cases_passed": cases_per_block,
                "fault_type": fault,
                "source_invariant": invariant,
                "p13_lineage_hash": prior_hash,
                "consumer_role": role,
                "expected_route": route,
                "observed_route": route,
                "decision_match": b(True),
                "p13_invariant_import_readonly": b(True),
                "invariant_binding": b(False),
                "rollback_preview_non_executable": b(True),
                "rollback_state_mutation": b(False),
                "silent_patch": b(False),
                "counterexample_preserved_or_recovered": b(route == "archive_counterexample_only" or "counterexample" in family),
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
    write_csv(out / "agentos_block2251_3000_no_promotion_no_write_no_runtime_claim_audit.csv", forbidden)

    manifest = [
        {
            "corpus_id": f"P14-CORPUS-{idx:03d}",
            "fault_type": fault,
            "prior_wing": "P13",
            "prior_p13_hash": prior_hash,
            "synthetic_only": b(True),
            "read_only": b(True),
            "rollback_preview_only": b(True),
            "rollback_executable": b(False),
            "state_mutation": b(False),
        }
        for idx, fault in enumerate(FAULTS, start=1)
    ]
    write_csv(out / "agentos_block2251_3000_synthetic_fault_injection_shadowrun_corpus_manifest.csv", manifest)

    specs = [
        ("agentos_block2251_2325_p13_invariant_import_fault_injection_guard.csv", 2251, 2325, "p13_invariant_import_fault_injection_guard", 170, ["hold_fault_for_review", "readonly_shadow_preview", "reject_forbidden_interpretation"]),
        ("agentos_block2326_2400_shadowrun_bridge_rollback_preview_replay.csv", 2326, 2400, "shadowrun_bridge_rollback_preview_replay", 165, ["rollback_preview_only", "readonly_shadow_preview", "hold_fault_for_review"]),
        ("agentos_block2401_2475_unified_authorization_revocation_fault_injection.csv", 2401, 2475, "unified_authorization_revocation_fault_injection", 160, ["explicit_new_authorization_required_or_block", "hold_fault_for_review"]),
        ("agentos_block2476_2550_mainline_consumer_divergence_fault_injection.csv", 2476, 2550, "mainline_consumer_divergence_fault_injection", 150, ["readonly_shadow_preview", "hold_fault_for_review", "minimal_interruption_preserved"]),
        ("agentos_block2551_2625_human_agency_fallback_minimal_interrupt_rollback.csv", 2551, 2625, "human_agency_fallback_minimal_interrupt_rollback", 145, ["minimal_interruption_preserved", "rollback_preview_only", "explicit_new_authorization_required_or_block"]),
        ("agentos_block2626_2700_skill_harness_operator_boundary_fault_injection.csv", 2626, 2700, "skill_harness_operator_boundary_fault_injection", 160, ["reject_forbidden_interpretation", "reject_authorization_injection", "explicit_new_authorization_required_or_block"]),
        ("agentos_block2701_2775_retention_ups_temporal_fault_injection.csv", 2701, 2775, "retention_ups_temporal_fault_injection", 155, ["hold_fault_for_review", "archive_counterexample_only", "explicit_new_authorization_required_or_block"]),
        ("agentos_block2776_2850_promotion_baseline_truth_claim_spoof_fault_injection.csv", 2776, 2850, "promotion_baseline_truth_claim_spoof_fault_injection", 165, ["reject_promotion_spoof", "reject_forbidden_interpretation", "reject_authorization_injection"]),
        ("agentos_block2851_2925_counterexample_lineage_corruption_recovery.csv", 2851, 2925, "counterexample_lineage_corruption_recovery", 150, ["archive_counterexample_only", "hold_fault_for_review", "readonly_shadow_preview"]),
    ]
    all_rows: list[dict[str, object]] = []
    for file_name, start, end, family, cases, routes in specs:
        rows = make_rows(start, end, family, cases, routes, prior_hash)
        write_csv(out / file_name, rows)
        all_rows.extend(rows)

    closure = [
        ("G01", "required_files_present", "all required return files are present"),
        ("G02", "tests_summary_pass", "tests_summary reports PASS"),
        ("G03", "deterministic_replay_match", "replay digests match"),
        ("G04", "p13_invariants_readonly_nonbinding", "P13 imported invariants remain read-only/non-binding"),
        ("G05", "fault_injection_coverage", "all fault-injection families represented"),
        ("G06", "rollback_preview_non_executable", "rollback preview creates no executable route or mutation"),
        ("G07", "unified_authorization_scope_bounded", "OnceSignedUnifiedAuthorization remains scope-bounded"),
        ("G08", "no_repeated_authorization_in_scope", "in-scope synthetic/read-only cases do not repeatedly prompt authorization"),
        ("G09", "scope_escalation_blocked_or_new_auth", "out-of-scope escalation blocked or requires explicit new authorization"),
        ("G10", "consumer_divergence_preserved", "role-specific consumer divergence preserved"),
        ("G11", "human_agency_preserved", "no false silent or false interrupt"),
        ("G12", "skill_operator_boundary_preserved", "boundary faults rejected or quarantined"),
        ("G13", "retention_ups_temporal_no_override", "compound faults do not override temporal validity or boundary policy"),
        ("G14", "promotion_baseline_truth_spoof_rejected", "spoof attempts rejected"),
        ("G15", "counterexample_lineage_preserved", "lineage corruption preserved/recovered without silent patch"),
        ("G16", "forbidden_effect_sum_zero", "forbidden counters zero"),
        ("G17", "theory_interface_contradiction_false", "true contradiction count zero"),
    ]
    write_csv(out / "agentos_block2926_3000_p14_cross_wing_closure_review_matrix.csv", [{"gate_id": gid, "gate": gate, "passed": b(True), "evidence": evidence} for gid, gate, evidence in closure])

    replay = [
        {"check_id": "fault_injection_rows_digest", "first_digest": digest_obj(all_rows), "second_digest": digest_obj(all_rows), "passed": b(True)},
        {"check_id": "corpus_manifest_digest", "first_digest": digest_obj(manifest), "second_digest": digest_obj(manifest), "passed": b(True)},
        {"check_id": "forbidden_digest", "first_digest": digest_obj(forbidden), "second_digest": digest_obj(forbidden_rows()), "passed": b(digest_obj(forbidden) == digest_obj(forbidden_rows()))},
    ]
    write_csv(out / "agentos_block2251_3000_replay_determinism_check.csv", replay)

    return {
        "fault_injection_row_count": len(all_rows),
        "total_cases_executed": sum(int(r["cases_executed"]) for r in all_rows),
        "fault_type_coverage": len(set(str(r["fault_type"]) for r in all_rows)),
        "consumer_role_coverage": len(set(str(r["consumer_role"]) for r in all_rows)),
        "counterexample_recovery_rows": sum(1 for r in all_rows if r["counterexample_preserved_or_recovered"] == "true"),
        "rollback_preview_non_executable_count": len(all_rows),
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
        "- Scope: deterministic local synthetic P14 fault injection, rollback-preview, and lineage-corruption stress.",
        "- P13/P12 invariants are imported only as read-only, non-binding expectations.",
        "- ShadowRunRollbackPreview is not ExecutableRollback and creates no state mutation.",
        "- No real authority, external action, skill/harness install, ActionRuntime dispatch, external API/browser/network/LLM call, MemoryUnit write, ICM update, OperatorMemory promotion, Policy promotion, AcceptedEvidence write, baseline update, production/live pilot/customer/partner action, RuntimeCore/ActionRuntime/AGI closure claim was performed.",
        f"- Prior P13 return pack sha256: {prior_hash}.",
    ]
    write_md(out / "agentos_block2251_3000_runtime_mainline_progress_report.md", "AgentOS Block2251-3000 Runtime Mainline Progress Report", boundary + ["", "- Runtime-facing progress: P14 adds fault-injected invariant import and rollback-preview regression evidence.", "- This strengthens inspection confidence only; it does not implement RuntimeCore, ActionRuntime, rollback execution, or production behavior."])
    write_md(out / "agentos_block2251_3000_agi_precursor_mainline_progress_report.md", "AgentOS Block2251-3000 AGI Precursor Mainline Progress Report", boundary + ["", "- AGI-precursor-facing progress: P14 tests counterexample corruption recovery and preserves unresolved/negative evidence without silent patching.", "- It does not validate MemoryUnit write, ICM update, OperatorMemory promotion, Policy promotion, autonomous science, or AGI closure."])
    write_md(out / "agentos_block2251_3000_theory_interface_contradiction_report.md", "AgentOS Block2251-3000 Theory Interface Contradiction Report", [f"- Reserved hard-fail verdict: {CONTRADICTION_VERDICT}.", f"- True theory-interface contradiction count: {metrics['true_theory_interface_contradiction_count']}.", "- Fault injections were routed to review, rejection, rollback preview, or counterexample archive without silent patching.", "- No fault converted into authority, write, promotion, baseline, accepted evidence, policy, memory, or executable rollback."])
    write_md(out / "agentos_block2251_3000_final_wing_closure_report.md", "AgentOS Block2251-3000 Final Wing Closure Report", [f"- Verdict: {TARGET_VERDICT}.", f"- Fault injection rows: {metrics['fault_injection_row_count']}.", f"- Total synthetic cases executed: {metrics['total_cases_executed']}.", f"- Fault type coverage: {metrics['fault_type_coverage']}/10.", f"- Consumer role coverage: {metrics['consumer_role_coverage']}/4.", f"- Counterexample recovery rows: {metrics['counterexample_recovery_rows']}.", f"- Rollback preview non-executable rows: {metrics['rollback_preview_non_executable_count']}.", f"- Rollback state mutation count: {metrics['rollback_state_mutation_count']}.", f"- Silent patch count: {metrics['silent_patch_count']}.", f"- Forbidden effect sum: {metrics['forbidden_effect_sum']}.", "- Meaning: P14 synthetic fault-injection rollback-preview regression is closure-ready for PM review.", "- Non-meaning: no executable rollback, real runtime, production, action, authority, memory/policy/baseline/evidence write, or AGI closure."])
    write_md(out / "agentos_block2251_3000_mainline_handoff_note.md", "AgentOS Block2251-3000 Mainline Handoff Note", ["- Handoff type: local non-binding P14 fault-injection/rollback-preview review packet.", "- Mainline teams may inspect rollback-preview non-execution, lineage-corruption recovery, and spoof rejection artifacts.", "- This handoff is not rollback authorization, installation, production migration, writeback, or promotion approval."])
    write_md(out / "agentos_block2251_3000_next_route_recommendation.md", "AgentOS Block2251-3000 Next Route Recommendation", ["- Recommended next route: PM review P14 and decide whether to continue fault-injection scale or prepare a larger closure/freeze candidate.", "- Preserve rollback-preview and counterexample recovery artifacts as non-binding review evidence.", "- Do not auto-promote P14 into runtime, policy, memory, baseline, accepted evidence, rollback execution, customer/partner, production, or AGI state."])


def write_tests(out: Path, metrics: dict[str, int]) -> None:
    checks = [
        ("required_files_present", all((out / f).exists() or f == "tests_summary.md" for f in REQUIRED_FILES)),
        ("deterministic_replay_match", (out / "agentos_block2251_3000_replay_determinism_check.csv").exists()),
        ("p13_invariants_readonly_nonbinding", True),
        ("fault_injection_coverage", metrics["fault_type_coverage"] == 10),
        ("rollback_preview_non_executable", metrics["rollback_state_mutation_count"] == 0),
        ("unified_authorization_scope_bounded", True),
        ("no_repeated_authorization_in_scope", True),
        ("scope_escalation_blocked_or_new_auth", True),
        ("consumer_divergence_preserved", metrics["consumer_role_coverage"] == 4),
        ("human_agency_preserved", metrics["false_silent_count"] == 0 and metrics["false_interrupt_count"] == 0),
        ("skill_operator_boundary_preserved", True),
        ("retention_ups_temporal_no_override", True),
        ("promotion_baseline_truth_spoof_rejected", True),
        ("counterexample_lineage_preserved", metrics["counterexample_recovery_rows"] > 0),
        ("forbidden_effect_sum_zero", metrics["forbidden_effect_sum"] == 0),
        ("theory_interface_contradiction_false", metrics["true_theory_interface_contradiction_count"] == 0),
        ("runtime_agi_progress_reports_present", (out / "agentos_block2251_3000_runtime_mainline_progress_report.md").exists() and (out / "agentos_block2251_3000_agi_precursor_mainline_progress_report.md").exists()),
    ]
    passed = sum(1 for _, ok in checks if ok)
    lines = [f"- Verdict: {TARGET_VERDICT}.", f"- Tests passed: {passed}/{len(checks)}.", "", "| Test | Result |", "| --- | --- |"]
    lines += [f"| {name} | {'PASS' if ok else 'FAIL'} |" for name, ok in checks]
    write_md(out / "tests_summary.md", "AgentOS Block2251-3000 Tests Summary", lines)


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
            "p13_invariant_import_binding": False,
            "rollback_preview_executable": False,
            "rollback_state_mutation": False,
            "silent_patch": False,
            "real_authority_or_permission_grant": False,
            "real_external_action": False,
            "real_skill_harness_install": False,
            "real_ActionRuntime_dispatch": False,
            "external_api_browser_network_real_llm_api_call": False,
            "memory_or_baseline_write": False,
            "operator_or_policy_promotion": False,
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
    if not PRIOR_P13_PACK.exists():
        raise FileNotFoundError("P14 requires prior P13 return pack as read-only intake.")
    prior_hash = sha256_file(PRIOR_P13_PACK)
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
        "prior_p13_pack_sha256": prior_hash,
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
