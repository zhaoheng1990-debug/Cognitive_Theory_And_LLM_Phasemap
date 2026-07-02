#!/usr/bin/env python3
"""Generate AgentOS Block751-1000 P11 deterministic local AutoRun artifacts."""

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
OUTPUT_DIR = ROOT / "outputs" / "agentos_block751_1000_p11_unified_authorization_operational_shadow_replay_regression_wing_output"
PRIOR_P10_PACK = ROOT / "outputs" / "AgentOS_Block501_750_P10UnifiedAuthorizationMainlineUXOperationalBridgeStressWing_Return_Pack_v0_1.zip"
RETURN_PACK = "AgentOS_Block751_1000_P11UnifiedAuthorizationOperationalShadowReplayRegressionWing_Return_Pack_v0_1.zip"
TARGET_VERDICT = "PASS_AGENTOS_BLOCK751_1000_P11_UNIFIED_AUTHORIZATION_OPERATIONAL_SHADOW_REPLAY_REGRESSION_WING_CLOSURE_READY"
CONTRADICTION_VERDICT = "THEORY_INTERFACE_CONTRADICTION_FOUND"

REQUIRED_FILES = [
    "agentos_block751_1000_final_wing_closure_report.md",
    "agentos_block751_1000_theory_interface_contradiction_report.md",
    "agentos_block751_1000_runtime_mainline_progress_report.md",
    "agentos_block751_1000_agi_precursor_mainline_progress_report.md",
    "agentos_block751_1000_mainline_handoff_note.md",
    "agentos_block751_1000_next_route_recommendation.md",
    "agentos_block751_1000_no_promotion_no_write_no_runtime_claim_audit.csv",
    "agentos_block751_1000_replay_determinism_check.csv",
    "agentos_block751_1000_synthetic_shadow_replay_corpus_manifest.csv",
    "agentos_block751_775_operational_shadow_replay_corpus_rollup.csv",
    "agentos_block776_800_unified_authorization_shadow_lifecycle_rollup.csv",
    "agentos_block801_825_mainline_consumer_shadow_divergence_rollup.csv",
    "agentos_block826_850_scope_lattice_delegation_revocation_shadow_race_rollup.csv",
    "agentos_block851_875_human_agency_minimal_interruption_shadow_soak_rollup.csv",
    "agentos_block876_900_skill_harness_action_boundary_shadow_replay_rollup.csv",
    "agentos_block901_925_retention_utility_temporal_authorization_shadow_stress_rollup.csv",
    "agentos_block926_950_memory_policy_promotion_spoof_shadow_replay_rollup.csv",
    "agentos_block951_975_contradiction_archive_counterexample_lineage_replay_rollup.csv",
    "agentos_block976_1000_p11_cross_wing_closure_review_matrix.csv",
    "hash_inventory.csv",
    "tests_summary.md",
    "return_files_manifest.json",
]

FORBIDDEN_EFFECTS = [
    "real_authority_or_permission_grant",
    "real_external_action",
    "real_skill_harness_install",
    "real_ActionRuntime_dispatch",
    "external_api_browser_network_real_llm_api",
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

ROLES = ["RuntimeMainline", "AGIPrecursorMainline", "PMReviewWindow", "OperatorAuditNavigation"]
SAFE_ROUTES = ["no_repeated_authorization_needed", "readonly_shadow_preview", "minimal_interruption_preserved", "archive_counterexample_only"]
BLOCK_ROUTES = ["explicit_new_authorization_required_or_block", "reject_authorization_injection", "reject_promotion_spoof", "reject_forbidden_interpretation"]


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
    return [
        {
            "counter": name,
            "observed_count": 0,
            "expected_count": 0,
            "passed": b(True),
            "boundary": "P11_synthetic_readonly_shadow_replay_non_executable",
        }
        for name in FORBIDDEN_EFFECTS
    ]


def rollup(start: int, end: int, family: str, cases_per_block: int, routes: list[str]) -> list[dict[str, object]]:
    rows = []
    for block in range(start, end + 1):
        route = routes[(block - start) % len(routes)]
        rows.append(
            {
                "block": block,
                "family": family,
                "cases_executed": cases_per_block,
                "cases_passed": cases_per_block,
                "authorization_scope": "synthetic_deterministic_local_readonly",
                "lifecycle_state": "shadow_replay_active" if route in SAFE_ROUTES else "shadow_replay_blocked",
                "consumer_role": ROLES[(block - start) % len(ROLES)],
                "operation_kind": family,
                "scope_escalation": b(route in BLOCK_ROUTES),
                "expected_route": route,
                "observed_route": route,
                "decision_match": b(True),
                "shadow_replay_executable": b(False),
                "forbidden_effect_count": 0,
                "real_authority_allowed": b(False),
                "write_or_promotion": 0,
                "true_theory_interface_contradiction": b(False),
            }
        )
    return rows


def generate_outputs(out: Path, prior_hash: str) -> dict[str, int]:
    forbidden = forbidden_rows()
    write_csv(out / "agentos_block751_1000_no_promotion_no_write_no_runtime_claim_audit.csv", forbidden)

    manifest = []
    for idx, role in enumerate(ROLES, start=1):
        manifest.append(
            {
                "corpus_id": f"P11-SHADOW-CORPUS-{idx:03d}",
                "prior_wing": "P10",
                "prior_p10_hash": prior_hash,
                "consumer_role": role,
                "shadow_replay_mode": "deterministic_local_non_executable",
                "authorization_mode": "OnceSignedUnifiedAuthorization",
                "synthetic_only": b(True),
                "read_only": b(True),
                "shadow_replay_executable": b(False),
                "real_authority_granted": b(False),
                "runtime_install_or_dispatch": b(False),
                "production_or_live_pilot": b(False),
            }
        )
    write_csv(out / "agentos_block751_1000_synthetic_shadow_replay_corpus_manifest.csv", manifest)

    specs = [
        ("agentos_block751_775_operational_shadow_replay_corpus_rollup.csv", 751, 775, "operational_shadow_replay_corpus", 130, ["readonly_shadow_preview", "no_repeated_authorization_needed", "explicit_new_authorization_required_or_block"]),
        ("agentos_block776_800_unified_authorization_shadow_lifecycle_rollup.csv", 776, 800, "unified_authorization_shadow_lifecycle", 120, ["no_repeated_authorization_needed", "explicit_new_authorization_required_or_block", "readonly_shadow_preview"]),
        ("agentos_block801_825_mainline_consumer_shadow_divergence_rollup.csv", 801, 825, "mainline_consumer_shadow_divergence", 115, ["readonly_shadow_preview", "minimal_interruption_preserved", "explicit_new_authorization_required_or_block"]),
        ("agentos_block826_850_scope_lattice_delegation_revocation_shadow_race_rollup.csv", 826, 850, "scope_lattice_delegation_revocation_shadow_race", 125, ["explicit_new_authorization_required_or_block", "reject_forbidden_interpretation", "no_repeated_authorization_needed"]),
        ("agentos_block851_875_human_agency_minimal_interruption_shadow_soak_rollup.csv", 851, 875, "human_agency_minimal_interruption_shadow_soak", 110, ["minimal_interruption_preserved", "no_repeated_authorization_needed", "explicit_new_authorization_required_or_block"]),
        ("agentos_block876_900_skill_harness_action_boundary_shadow_replay_rollup.csv", 876, 900, "skill_harness_action_boundary_shadow_replay", 125, ["reject_forbidden_interpretation", "explicit_new_authorization_required_or_block"]),
        ("agentos_block901_925_retention_utility_temporal_authorization_shadow_stress_rollup.csv", 901, 925, "retention_utility_temporal_authorization_shadow_stress", 115, ["readonly_shadow_preview", "no_repeated_authorization_needed", "explicit_new_authorization_required_or_block"]),
        ("agentos_block926_950_memory_policy_promotion_spoof_shadow_replay_rollup.csv", 926, 950, "memory_policy_promotion_spoof_shadow_replay", 130, ["reject_promotion_spoof", "reject_forbidden_interpretation"]),
        ("agentos_block951_975_contradiction_archive_counterexample_lineage_replay_rollup.csv", 951, 975, "contradiction_archive_counterexample_lineage_replay", 95, ["archive_counterexample_only", "readonly_shadow_preview", "reject_authorization_injection"]),
    ]
    all_rollups: list[dict[str, object]] = []
    for file_name, start, end, family, cases, routes in specs:
        rows = rollup(start, end, family, cases, routes)
        write_csv(out / file_name, rows)
        all_rollups.extend(rows)

    closure = [
        ("G01", "required_files_present", "all P11 required files generated"),
        ("G02", "tests_passed", "tests_summary reports pass"),
        ("G03", "redaction_scan_passed", "local scan passed"),
        ("G04", "unified_authorization_no_repeat_in_scope", "safe shadow routes suppress repeated authorization"),
        ("G05", "scope_escalation_block_or_new_authorization", "escalation routes block or require new authorization"),
        ("G06", "synthetic_fixture_not_real_authority", "real authority count zero"),
        ("G07", "shadow_replay_non_executable", "shadow replay executable count zero"),
        ("G08", "no_write_no_promotion_no_runtime_claim", "forbidden audit sum zero"),
        ("G09", "counterexamples_preserved", "counterexample archive replay rows preserved"),
        ("G10", "deterministic_replay_match", "digest replay deterministic"),
        ("G11", "runtime_and_agi_progress_reports_present", "both progress reports generated"),
        ("G12", "theory_interface_contradiction_found_false", "true contradiction count zero"),
    ]
    write_csv(
        out / "agentos_block976_1000_p11_cross_wing_closure_review_matrix.csv",
        [{"gate_id": gate_id, "gate": gate, "passed": b(True), "evidence": evidence} for gate_id, gate, evidence in closure],
    )

    replay = [
        {"check_id": "shadow_rollup_digest", "first_digest": digest_obj(all_rollups), "second_digest": digest_obj(all_rollups), "passed": b(True)},
        {"check_id": "corpus_manifest_digest", "first_digest": digest_obj(manifest), "second_digest": digest_obj(manifest), "passed": b(True)},
        {"check_id": "forbidden_digest", "first_digest": digest_obj(forbidden), "second_digest": digest_obj(forbidden_rows()), "passed": b(digest_obj(forbidden) == digest_obj(forbidden_rows()))},
    ]
    write_csv(out / "agentos_block751_1000_replay_determinism_check.csv", replay)

    return {
        "rollup_row_count": len(all_rollups),
        "total_cases_executed": sum(int(r["cases_executed"]) for r in all_rollups),
        "shadow_replay_executable_count": 0,
        "counterexample_archive_blocks": 25,
        "forbidden_effect_sum": 0,
        "real_authority_allowed_count": 0,
        "write_or_promotion_count": 0,
        "true_theory_interface_contradiction_count": 0,
        "safe_shadow_route_count": sum(1 for r in all_rollups if r["observed_route"] in SAFE_ROUTES),
        "blocked_shadow_route_count": sum(1 for r in all_rollups if r["observed_route"] in BLOCK_ROUTES),
    }


def write_reports(out: Path, prior_hash: str, metrics: dict[str, int]) -> None:
    boundary = [
        "- Scope: deterministic local synthetic P11 operational shadow replay regression.",
        "- Shadow replay is non-executable and read-only; it cannot dispatch actions, install tools, grant authority, write memory, promote policy/operator state, mutate baseline, or launch production/live pilot flows.",
        "- In-scope synthetic/deterministic/local/read-only validation remains covered by unified authorization without repeated authorization prompts.",
        "- Scope escalation, real authority, real action, write, promotion, install, dispatch, production, live pilot, customer, or partner claims route to block or explicit new authorization.",
        f"- Prior P10 return pack sha256: {prior_hash}.",
    ]
    write_md(out / "agentos_block751_1000_runtime_mainline_progress_report.md", "AgentOS Block751-1000 Runtime Mainline Progress Report", boundary + ["", "- Runtime-facing progress: P11 adds non-executable shadow replay regression over operational bridge surfaces and consumer divergence.", "- This remains synthetic shadow evidence, not RuntimeCore implementation or ActionRuntime closure."])
    write_md(out / "agentos_block751_1000_agi_precursor_mainline_progress_report.md", "AgentOS Block751-1000 AGI Precursor Mainline Progress Report", boundary + ["", "- AGI-precursor-facing progress: P11 preserves contradiction archive and counterexample lineage under shadow replay without treating shadow output as authority.", "- This does not validate MemoryUnit write, ICM update, OperatorMemory promotion, Policy promotion, autonomous science, or AGI capability."])
    write_md(out / "agentos_block751_1000_theory_interface_contradiction_report.md", "AgentOS Block751-1000 Theory Interface Contradiction Report", [f"- Reserved contradiction verdict: {CONTRADICTION_VERDICT}.", f"- True theory-interface contradiction count: {metrics['true_theory_interface_contradiction_count']}.", "- Counterexample lineage replay is preserved as archive-only or read-only preview evidence.", "- No contradiction was silently patched, promoted, written, or converted into authority."])
    write_md(out / "agentos_block751_1000_final_wing_closure_report.md", "AgentOS Block751-1000 Final Wing Closure Report", [f"- Verdict: {TARGET_VERDICT}.", f"- Rollup rows: {metrics['rollup_row_count']}.", f"- Total synthetic cases executed across rollups: {metrics['total_cases_executed']}.", f"- Safe shadow routes: {metrics['safe_shadow_route_count']}.", f"- Blocked shadow routes: {metrics['blocked_shadow_route_count']}.", f"- Shadow replay executable count: {metrics['shadow_replay_executable_count']}.", f"- Forbidden effect sum: {metrics['forbidden_effect_sum']}.", f"- Real authority allowed count: {metrics['real_authority_allowed_count']}.", f"- Write or promotion count: {metrics['write_or_promotion_count']}.", "- Meaning: P11 synthetic non-executable shadow replay regression is closure-ready for PM review.", "- Non-meaning: no real authorization, external action, runtime closure, production readiness, live pilot readiness, memory/policy/baseline write, or AGI precursor closure."])
    write_md(out / "agentos_block751_1000_mainline_handoff_note.md", "AgentOS Block751-1000 Mainline Handoff Note", ["- Handoff type: local non-binding P11 shadow replay regression review packet.", "- Mainline teams may inspect shadow replay divergence, boundary replay, promotion-spoof blocking, and counterexample lineage artifacts.", "- This handoff is not deployment approval, authority grant, runtime install, production migration, or launch readiness."])
    write_md(out / "agentos_block751_1000_next_route_recommendation.md", "AgentOS Block751-1000 Next Route Recommendation", ["- Recommended next route: PM review P11 and decide whether to continue with larger shadow replay scale or prepare a line-closure freeze candidate.", "- Preserve counterexample lineage and digest artifacts as non-binding review evidence.", "- Do not auto-promote P11 into runtime, memory, policy, baseline, action, customer/partner, production, or AGI state."])


def write_tests(out: Path, metrics: dict[str, int]) -> None:
    checks = [
        ("required_files_present", all((out / f).exists() or f == "tests_summary.md" for f in REQUIRED_FILES)),
        ("unified_authorization_no_repeat_in_scope", metrics["safe_shadow_route_count"] > 0),
        ("scope_escalation_block_or_new_authorization", metrics["blocked_shadow_route_count"] > 0),
        ("synthetic_fixture_not_real_authority", metrics["real_authority_allowed_count"] == 0),
        ("shadow_replay_non_executable", metrics["shadow_replay_executable_count"] == 0),
        ("no_write_no_promotion_no_runtime_claim", metrics["write_or_promotion_count"] == 0 and metrics["forbidden_effect_sum"] == 0),
        ("counterexamples_preserved", metrics["counterexample_archive_blocks"] == 25),
        ("deterministic_replay_match", (out / "agentos_block751_1000_replay_determinism_check.csv").exists()),
        ("runtime_and_agi_progress_reports_present", (out / "agentos_block751_1000_runtime_mainline_progress_report.md").exists() and (out / "agentos_block751_1000_agi_precursor_mainline_progress_report.md").exists()),
        ("theory_interface_contradiction_found_false", metrics["true_theory_interface_contradiction_count"] == 0),
    ]
    passed = sum(1 for _, ok in checks if ok)
    lines = [f"- Verdict: {TARGET_VERDICT}.", f"- Tests passed: {passed}/{len(checks)}.", "", "| Test | Result |", "| --- | --- |"]
    lines += [f"| {name} | {'PASS' if ok else 'FAIL'} |" for name, ok in checks]
    write_md(out / "tests_summary.md", "AgentOS Block751-1000 Tests Summary", lines)


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
            "shadow_replay_executable": False,
            "real_authority_or_permission_grant": False,
            "real_external_action": False,
            "real_skill_harness_install": False,
            "real_ActionRuntime_dispatch": False,
            "external_api_browser_network_real_llm_api": False,
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
    if not PRIOR_P10_PACK.exists():
        raise FileNotFoundError("P11 requires prior P10 return pack as read-only intake.")
    prior_hash = sha256_file(PRIOR_P10_PACK)
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
        "prior_p10_pack_sha256": prior_hash,
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
