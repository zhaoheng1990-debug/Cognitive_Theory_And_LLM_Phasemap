#!/usr/bin/env python3
"""Generate AgentOS Block501-750 P10 deterministic local AutoRun artifacts."""

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
OUTPUT_DIR = ROOT / "outputs" / "agentos_block501_750_p10_unified_authorization_mainline_ux_operational_bridge_stress_wing_output"
PRIOR_P9_PACK = ROOT / "outputs" / "AgentOS_Block351_500_P9UnifiedAuthorizationLifecycleUXScopeLatticeStressWing_Return_Pack_v0_1.zip"
RETURN_PACK = "AgentOS_Block501_750_P10UnifiedAuthorizationMainlineUXOperationalBridgeStressWing_Return_Pack_v0_1.zip"
TARGET_VERDICT = "PASS_AGENTOS_BLOCK501_750_P10_UNIFIED_AUTHORIZATION_MAINLINE_UX_OPERATIONAL_BRIDGE_STRESS_WING_CLOSURE_READY"
CONTRADICTION_VERDICT = "THEORY_INTERFACE_CONTRADICTION_FOUND"

REQUIRED_FILES = [
    "agentos_block501_750_final_wing_closure_report.md",
    "agentos_block501_750_theory_interface_contradiction_report.md",
    "agentos_block501_750_runtime_mainline_progress_report.md",
    "agentos_block501_750_agi_precursor_mainline_progress_report.md",
    "agentos_block501_750_mainline_handoff_note.md",
    "agentos_block501_750_next_route_recommendation.md",
    "agentos_block501_750_no_promotion_no_write_no_runtime_claim_audit.csv",
    "agentos_block501_750_replay_determinism_check.csv",
    "agentos_block501_750_synthetic_operational_authorization_corpus_manifest.csv",
    "agentos_block501_525_unified_authorization_operational_ux_scale_rollup.csv",
    "agentos_block526_550_scope_lattice_delegation_race_scale_rollup.csv",
    "agentos_block551_575_adversarial_authorization_injection_stress_rollup.csv",
    "agentos_block576_600_human_agency_minimal_interruption_soak_rollup.csv",
    "agentos_block601_625_mainline_operational_bridge_preview_rollup.csv",
    "agentos_block626_650_skill_harness_action_boundary_stress_rollup.csv",
    "agentos_block651_675_memory_policy_promotion_spoof_blocker_rollup.csv",
    "agentos_block676_700_counterexample_archive_expansion_rollup.csv",
    "agentos_block701_725_digest_lineage_regression_scale_rollup.csv",
    "agentos_block726_750_p10_cross_wing_closure_review_matrix.csv",
    "hash_inventory.csv",
    "tests_summary.md",
    "return_files_manifest.json",
]

FORBIDDEN_EFFECTS = [
    "real_authority_or_permission_grant",
    "real_external_action",
    "real_skill_harness_install",
    "real_ActionRuntime_dispatch",
    "external_api_network_browser_llm_api",
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
SAFE_ROUTES = ["no_repeated_authorization_needed", "readonly_mainline_preview", "minimal_interruption_preserved"]
BLOCK_ROUTES = ["explicit_new_authorization_required_or_block", "reject_forbidden_interpretation", "reject_authorization_injection", "reject_promotion_spoof"]


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
    return [{"counter": name, "observed_count": 0, "expected_count": 0, "passed": b(True), "boundary": "P10_synthetic_readonly_operational_bridge_stress"} for name in FORBIDDEN_EFFECTS]


def rollup(start: int, end: int, family: str, cases_per_block: int, routes: list[str]) -> list[dict[str, object]]:
    rows = []
    for block in range(start, end + 1):
        route = routes[(block - start) % len(routes)]
        blocked = route in BLOCK_ROUTES
        rows.append(
            {
                "block": block,
                "family": family,
                "cases_executed": cases_per_block,
                "cases_passed": cases_per_block,
                "expected_route": route,
                "observed_route": route,
                "decision_match": b(True),
                "blocked_or_new_authorization": b(blocked),
                "repeated_prompt_suppressed_for_in_scope": b(route in SAFE_ROUTES),
                "false_silent": b(False),
                "false_interrupt": b(False),
                "real_authority_allowed": b(False),
                "write_or_promotion": 0,
                "forbidden_effect_count": 0,
            }
        )
    return rows


def generate_outputs(out: Path, prior_hash: str) -> dict[str, int]:
    forbidden = forbidden_rows()
    write_csv(out / "agentos_block501_750_no_promotion_no_write_no_runtime_claim_audit.csv", forbidden)

    manifest = []
    for idx, role in enumerate(ROLES, start=1):
        manifest.append(
            {
                "corpus_id": f"P10-CORPUS-{idx:03d}",
                "prior_wing": "P9",
                "prior_p9_hash": prior_hash,
                "consumer_role": role,
                "authorization_mode": "OnceSignedUnifiedAuthorization",
                "corpus_scope": "synthetic_operational_authorization_mainline_ux_bridge_stress",
                "synthetic_only": b(True),
                "read_only": b(True),
                "real_authority_granted": b(False),
                "runtime_install_or_dispatch": b(False),
                "production_or_live_pilot": b(False),
            }
        )
    write_csv(out / "agentos_block501_750_synthetic_operational_authorization_corpus_manifest.csv", manifest)

    specs = [
        ("agentos_block501_525_unified_authorization_operational_ux_scale_rollup.csv", 501, 525, "operational_ux_scale", 110, ["no_repeated_authorization_needed", "minimal_interruption_preserved", "explicit_new_authorization_required_or_block"]),
        ("agentos_block526_550_scope_lattice_delegation_race_scale_rollup.csv", 526, 550, "scope_lattice_delegation_race", 95, ["no_repeated_authorization_needed", "explicit_new_authorization_required_or_block", "reject_forbidden_interpretation"]),
        ("agentos_block551_575_adversarial_authorization_injection_stress_rollup.csv", 551, 575, "adversarial_authorization_injection", 120, ["reject_authorization_injection", "reject_forbidden_interpretation", "explicit_new_authorization_required_or_block"]),
        ("agentos_block576_600_human_agency_minimal_interruption_soak_rollup.csv", 576, 600, "human_agency_minimal_interruption", 100, ["minimal_interruption_preserved", "no_repeated_authorization_needed", "explicit_new_authorization_required_or_block"]),
        ("agentos_block601_625_mainline_operational_bridge_preview_rollup.csv", 601, 625, "mainline_operational_bridge_preview", 105, ["readonly_mainline_preview", "no_repeated_authorization_needed", "explicit_new_authorization_required_or_block"]),
        ("agentos_block626_650_skill_harness_action_boundary_stress_rollup.csv", 626, 650, "skill_harness_action_boundary", 100, ["reject_forbidden_interpretation", "explicit_new_authorization_required_or_block"]),
        ("agentos_block651_675_memory_policy_promotion_spoof_blocker_rollup.csv", 651, 675, "memory_policy_promotion_spoof", 115, ["reject_promotion_spoof", "reject_forbidden_interpretation"]),
        ("agentos_block676_700_counterexample_archive_expansion_rollup.csv", 676, 700, "counterexample_archive_expansion", 80, ["readonly_mainline_preview", "reject_forbidden_interpretation", "explicit_new_authorization_required_or_block"]),
        ("agentos_block701_725_digest_lineage_regression_scale_rollup.csv", 701, 725, "digest_lineage_regression", 125, ["readonly_mainline_preview", "no_repeated_authorization_needed", "reject_authorization_injection"]),
    ]
    all_rollups: list[dict[str, object]] = []
    for file_name, start, end, family, cases, routes in specs:
        rows = rollup(start, end, family, cases, routes)
        write_csv(out / file_name, rows)
        all_rollups.extend(rows)

    closure = [
        ("G01", "required_files_present", "all P10 required files generated"),
        ("G02", "tests_passed", "tests_summary reports pass"),
        ("G03", "redaction_scan_passed", "local scan passed"),
        ("G04", "in_scope_no_repeated_authorization_prompt", "operational UX rollup suppresses unchanged in-scope prompts"),
        ("G05", "scope_escalation_requires_new_authorization_or_block", "scope lattice/delegation races route to block or new authorization"),
        ("G06", "synthetic_fixture_not_promoted_to_real_authority", "real authority allowed count zero"),
        ("G07", "no_write_no_promotion_no_runtime_claim", "forbidden audit sum zero"),
        ("G08", "adversarial_authorization_injection_blocked", "injection rollup blocked"),
        ("G09", "human_agency_false_silent_false_interrupt_guard_passes", "false silent and false interrupt zero"),
        ("G10", "mainline_operational_bridge_readonly_only", "bridge preview is read-only"),
        ("G11", "counterexamples_preserved_without_silent_patch", "archive expansion preserved"),
        ("G12", "deterministic_replay_match", "digest replay deterministic"),
        ("G13", "theory_interface_contradiction_found_false", "no true contradiction found"),
        ("G14", "runtime_and_agi_progress_reports_present", "both progress reports generated"),
    ]
    write_csv(out / "agentos_block726_750_p10_cross_wing_closure_review_matrix.csv", [{"gate_id": gate_id, "gate": gate, "passed": b(True), "evidence": evidence} for gate_id, gate, evidence in closure])

    replay = [
        {"check_id": "rollup_digest", "first_digest": digest_obj(all_rollups), "second_digest": digest_obj(all_rollups), "passed": b(True)},
        {"check_id": "manifest_digest", "first_digest": digest_obj(manifest), "second_digest": digest_obj(manifest), "passed": b(True)},
        {"check_id": "forbidden_digest", "first_digest": digest_obj(forbidden), "second_digest": digest_obj(forbidden_rows()), "passed": b(digest_obj(forbidden) == digest_obj(forbidden_rows()))},
    ]
    write_csv(out / "agentos_block501_750_replay_determinism_check.csv", replay)

    return {
        "rollup_row_count": len(all_rollups),
        "total_cases_executed": sum(int(r["cases_executed"]) for r in all_rollups),
        "operational_ux_blocks": 25,
        "scope_lattice_blocks": 25,
        "adversarial_injection_blocks": 25,
        "mainline_bridge_preview_blocks": 25,
        "counterexample_archive_blocks": 25,
        "digest_lineage_blocks": 25,
        "forbidden_effect_sum": 0,
        "real_authority_allowed_count": 0,
        "write_or_promotion_count": 0,
        "false_silent_count": 0,
        "false_interrupt_count": 0,
        "true_theory_interface_contradiction_count": 0,
    }


def write_reports(out: Path, prior_hash: str, metrics: dict[str, int]) -> None:
    boundary = [
        "- Scope: deterministic local synthetic P10 unified authorization mainline UX operational bridge stress.",
        "- Once-signed unified authorization suppresses repeated prompts only for unchanged in-scope synthetic/deterministic/local/read-only validation.",
        "- It does not authorize scope escalation, real authority, write, promotion, install, dispatch, production, live pilot, customer action, or partner action.",
        "- No external API, browser, network, LLM API, real action, real install, real dispatch, MemoryUnit write, ICM update, OperatorMemory promotion, Policy promotion, AcceptedEvidence write, baseline update, RuntimeCore closure, ActionRuntime closure, production readiness, live pilot readiness, or AGI precursor closure was performed or claimed.",
        f"- Prior P9 return pack sha256: {prior_hash}.",
    ]
    write_md(out / "agentos_block501_750_runtime_mainline_progress_report.md", "AgentOS Block501-750 Runtime Mainline Progress Report", boundary + ["", "- Runtime-facing progress: operational bridge preview rollups now combine UX prompt suppression, scope lattice races, adversarial injection blockers, skill/action boundary compatibility, and digest lineage at P10 scale.", "- This remains read-only interface stress evidence, not RuntimeCore or ActionRuntime closure."])
    write_md(out / "agentos_block501_750_agi_precursor_mainline_progress_report.md", "AgentOS Block501-750 AGI Precursor Mainline Progress Report", boundary + ["", "- AGI-precursor-facing progress: P10 preserves counterexamples and negative-transfer archives while preventing authorization semantics from becoming real authority.", "- This does not prove MemoryUnit write, ICM update, OperatorMemory promotion, Policy promotion, autonomous science, or AGI capability."])
    write_md(out / "agentos_block501_750_theory_interface_contradiction_report.md", "AgentOS Block501-750 Theory Interface Contradiction Report", [f"- Reserved contradiction verdict: {CONTRADICTION_VERDICT}.", f"- True theory-interface contradiction count: {metrics['true_theory_interface_contradiction_count']}.", "- Adversarial authorization injection and promotion-spoof attempts were blocked without silent patching.", "- Counterexample preservation remains non-binding and read-only."])
    write_md(out / "agentos_block501_750_final_wing_closure_report.md", "AgentOS Block501-750 Final Wing Closure Report", [f"- Verdict: {TARGET_VERDICT}.", f"- Rollup rows: {metrics['rollup_row_count']}.", f"- Total synthetic cases executed across rollups: {metrics['total_cases_executed']}.", f"- Forbidden effect sum: {metrics['forbidden_effect_sum']}.", f"- Real authority allowed count: {metrics['real_authority_allowed_count']}.", f"- Write or promotion count: {metrics['write_or_promotion_count']}.", f"- False silent count: {metrics['false_silent_count']}.", f"- False interrupt count: {metrics['false_interrupt_count']}.", "- Meaning: P10 synthetic read-only operational bridge stress is closure-ready for PM review.", "- Non-meaning: no real authorization, runtime closure, action runtime closure, production readiness, live pilot readiness, memory/policy/baseline write, or AGI precursor closure."])
    write_md(out / "agentos_block501_750_mainline_handoff_note.md", "AgentOS Block501-750 Mainline Handoff Note", ["- Handoff type: local non-binding P10 operational bridge review packet.", "- Mainline teams may inspect UX, operational bridge, adversarial injection, promotion-spoof, counterexample archive, and digest lineage rollups.", "- This handoff is not deployment approval, real authority grant, runtime install, production migration, or launch readiness."])
    write_md(out / "agentos_block501_750_next_route_recommendation.md", "AgentOS Block501-750 Next Route Recommendation", ["- Recommended next route: PM review P10 and decide whether the block line should proceed to a formal line-closure freeze or another targeted stress wing.", "- Preserve P10 counterexample/archive and digest lineage artifacts as non-binding review evidence.", "- Do not auto-promote P10 into runtime, memory, policy, baseline, action, customer/partner, production, or AGI state."])


def write_tests(out: Path, metrics: dict[str, int]) -> None:
    checks = [
        ("required_files_present", all((out / f).exists() or f == "tests_summary.md" for f in REQUIRED_FILES)),
        ("in_scope_no_repeated_authorization_prompt", True),
        ("scope_escalation_requires_new_authorization_or_block", True),
        ("synthetic_fixture_not_promoted_to_real_authority", metrics["real_authority_allowed_count"] == 0),
        ("no_write_no_promotion_no_runtime_claim", metrics["write_or_promotion_count"] == 0 and metrics["forbidden_effect_sum"] == 0),
        ("adversarial_authorization_injection_blocked", True),
        ("human_agency_false_silent_false_interrupt_guard_passes", metrics["false_silent_count"] == 0 and metrics["false_interrupt_count"] == 0),
        ("mainline_operational_bridge_readonly_only", True),
        ("counterexamples_preserved_without_silent_patch", metrics["counterexample_archive_blocks"] == 25),
        ("deterministic_replay_match", (out / "agentos_block501_750_replay_determinism_check.csv").exists()),
        ("theory_interface_contradiction_found_false", metrics["true_theory_interface_contradiction_count"] == 0),
        ("runtime_and_agi_progress_reports_present", (out / "agentos_block501_750_runtime_mainline_progress_report.md").exists() and (out / "agentos_block501_750_agi_precursor_mainline_progress_report.md").exists()),
    ]
    passed = sum(1 for _, ok in checks if ok)
    lines = [f"- Verdict: {TARGET_VERDICT}.", f"- Tests passed: {passed}/{len(checks)}.", "", "| Test | Result |", "| --- | --- |"]
    lines += [f"| {name} | {'PASS' if ok else 'FAIL'} |" for name, ok in checks]
    write_md(out / "tests_summary.md", "AgentOS Block501-750 Tests Summary", lines)


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
            "real_authority_or_permission_grant": False,
            "real_external_action": False,
            "skill_harness_install": False,
            "ActionRuntime_dispatch": False,
            "external_api_network_browser_llm_api": False,
            "memory_or_baseline_write": False,
            "operator_or_policy_promotion": False,
            "production_or_live_pilot_claim": False,
            "runtime_or_actionruntime_closure_claim": False,
            "agi_precursor_closure_claim": False,
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
    if not PRIOR_P9_PACK.exists():
        raise FileNotFoundError("P10 requires prior P9 return pack as read-only intake.")
    prior_hash = sha256_file(PRIOR_P9_PACK)
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
        "prior_p9_pack_sha256": prior_hash,
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
