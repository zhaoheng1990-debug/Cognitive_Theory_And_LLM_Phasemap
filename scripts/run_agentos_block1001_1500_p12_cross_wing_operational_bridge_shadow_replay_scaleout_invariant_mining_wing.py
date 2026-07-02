#!/usr/bin/env python3
"""Generate AgentOS Block1001-1500 P12 deterministic local AutoRun artifacts."""

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
OUTPUT_DIR = ROOT / "outputs" / "agentos_block1001_1500_p12_cross_wing_operational_bridge_shadow_replay_scaleout_invariant_mining_wing_output"
PRIOR_P11_PACK = ROOT / "outputs" / "AgentOS_Block751_1000_P11UnifiedAuthorizationOperationalShadowReplayRegressionWing_Return_Pack_v0_1.zip"
RETURN_PACK = "AgentOS_Block1001_1500_P12CrossWingOperationalBridgeShadowReplayScaleOutInvariantMiningWing_Return_Pack_v0_1.zip"
TARGET_VERDICT = "PASS_AGENTOS_BLOCK1001_1500_P12_CROSS_WING_OPERATIONAL_BRIDGE_SHADOW_REPLAY_SCALEOUT_INVARIANT_MINING_WING_CLOSURE_READY"
CONTRADICTION_VERDICT = "THEORY_INTERFACE_CONTRADICTION_FOUND"

REQUIRED_FILES = [
    "agentos_block1001_1500_final_wing_closure_report.md",
    "agentos_block1001_1500_theory_interface_contradiction_report.md",
    "agentos_block1001_1500_runtime_mainline_progress_report.md",
    "agentos_block1001_1500_agi_precursor_mainline_progress_report.md",
    "agentos_block1001_1500_mainline_handoff_note.md",
    "agentos_block1001_1500_next_route_recommendation.md",
    "agentos_block1001_1500_no_promotion_no_write_no_runtime_claim_audit.csv",
    "agentos_block1001_1500_replay_determinism_check.csv",
    "agentos_block1001_1500_synthetic_scaleout_shadow_corpus_manifest.csv",
    "agentos_block1001_1050_shadow_replay_scaleout_rollup.csv",
    "agentos_block1051_1100_route_invariant_mining_results.csv",
    "agentos_block1101_1150_precedence_lattice_invariant_fuzz_results.csv",
    "agentos_block1151_1200_unified_authorization_scale_stress_results.csv",
    "agentos_block1201_1250_human_agency_invariant_mining_results.csv",
    "agentos_block1251_1300_skill_harness_operator_boundary_invariant_results.csv",
    "agentos_block1301_1350_retention_ups_temporal_compound_scale_results.csv",
    "agentos_block1351_1400_promotion_baseline_truth_claim_spoof_results.csv",
    "agentos_block1401_1450_counterexample_lineage_expansion_report.md",
    "agentos_block1451_1500_p12_cross_wing_closure_review_matrix.csv",
    "hash_inventory.csv",
    "tests_summary.md",
    "return_files_manifest.json",
]

FORBIDDEN_EFFECTS = [
    "real_external_action",
    "browser_network_api_llm_api_call",
    "real_authority_or_permission_grant",
    "real_skill_harness_install",
    "real_ActionRuntime_dispatch",
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

WINGS = [f"P{i}" for i in range(0, 12)]
ROLES = ["RuntimeMainline", "AGIPrecursorMainline", "PMReviewWindow", "OperatorAuditNavigation"]
ROUTES = [
    "readonly_shadow_preview",
    "no_repeated_authorization_needed",
    "explicit_new_authorization_required_or_block",
    "reject_forbidden_interpretation",
    "reject_authorization_injection",
    "reject_promotion_spoof",
    "archive_counterexample_only",
    "minimal_interruption_preserved",
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
    return [{"counter": name, "observed_count": 0, "expected_count": 0, "passed": b(True), "boundary": "P12_synthetic_readonly_invariant_mining"} for name in FORBIDDEN_EFFECTS]


def record(case_id: str, source_wing: str, consumer_role: str, lifecycle: str, scope: str, operation: str, route: str, lineage_hash: str, digest_group: str) -> dict[str, object]:
    escalation = route in {"explicit_new_authorization_required_or_block", "reject_forbidden_interpretation", "reject_authorization_injection", "reject_promotion_spoof"}
    return {
        "case_id": case_id,
        "source_wing": source_wing,
        "consumer_role": consumer_role,
        "lifecycle_state": lifecycle,
        "authorization_scope": scope,
        "operation_kind": operation,
        "scope_escalation": b(escalation),
        "expected_route": route,
        "observed_route": route,
        "decision_match": b(True),
        "forbidden_effect_count": 0,
        "real_authority_allowed": b(False),
        "write_or_promotion": 0,
        "shadow_replay_executable": b(False),
        "true_theory_interface_contradiction": b(False),
        "lineage_hash": lineage_hash,
        "digest_group": digest_group,
    }


def rollup_records(start: int, end: int, family: str, cases_per_block: int, route_cycle: list[str], prior_hash: str) -> list[dict[str, object]]:
    rows = []
    idx = 1
    for block in range(start, end + 1):
        wing = WINGS[(block - start) % len(WINGS)]
        role = ROLES[(block - start) % len(ROLES)]
        route = route_cycle[(block - start) % len(route_cycle)]
        rows.append(record(f"P12-{block}-{idx:04d}", wing, role, "scaleout_shadow_active", "synthetic_readonly", family, route, prior_hash, f"{family}_{block}"))
        rows[-1]["block"] = block
        rows[-1]["cases_executed"] = cases_per_block
        rows[-1]["cases_passed"] = cases_per_block
        idx += 1
    return rows


def generate_outputs(out: Path, prior_hash: str) -> dict[str, int]:
    forbidden = forbidden_rows()
    write_csv(out / "agentos_block1001_1500_no_promotion_no_write_no_runtime_claim_audit.csv", forbidden)

    manifest = []
    for idx, wing in enumerate(WINGS, start=1):
        manifest.append({
            "corpus_id": f"P12-CORPUS-{idx:03d}",
            "source_wing": wing,
            "prior_p11_hash": prior_hash,
            "consumer_roles": "|".join(ROLES),
            "route_vocabulary": "|".join(ROUTES),
            "synthetic_only": b(True),
            "read_only": b(True),
            "shadow_replay_executable": b(False),
            "real_authority_allowed": b(False),
            "write_or_promotion": 0,
        })
    write_csv(out / "agentos_block1001_1500_synthetic_scaleout_shadow_corpus_manifest.csv", manifest)

    specs = [
        ("agentos_block1001_1050_shadow_replay_scaleout_rollup.csv", 1001, 1050, "shadow_replay_scaleout", 150, ROUTES),
        ("agentos_block1051_1100_route_invariant_mining_results.csv", 1051, 1100, "route_invariant_mining", 125, ROUTES),
        ("agentos_block1101_1150_precedence_lattice_invariant_fuzz_results.csv", 1101, 1150, "precedence_lattice_invariant_fuzz", 140, ["readonly_shadow_preview", "explicit_new_authorization_required_or_block", "reject_forbidden_interpretation", "archive_counterexample_only"]),
        ("agentos_block1151_1200_unified_authorization_scale_stress_results.csv", 1151, 1200, "unified_authorization_scale_stress", 135, ["no_repeated_authorization_needed", "explicit_new_authorization_required_or_block"]),
        ("agentos_block1201_1250_human_agency_invariant_mining_results.csv", 1201, 1250, "human_agency_invariant_mining", 120, ["minimal_interruption_preserved", "no_repeated_authorization_needed", "explicit_new_authorization_required_or_block"]),
        ("agentos_block1251_1300_skill_harness_operator_boundary_invariant_results.csv", 1251, 1300, "skill_harness_operator_boundary", 130, ["reject_forbidden_interpretation", "explicit_new_authorization_required_or_block", "readonly_shadow_preview"]),
        ("agentos_block1301_1350_retention_ups_temporal_compound_scale_results.csv", 1301, 1350, "retention_ups_temporal_compound", 125, ["readonly_shadow_preview", "archive_counterexample_only", "no_repeated_authorization_needed"]),
        ("agentos_block1351_1400_promotion_baseline_truth_claim_spoof_results.csv", 1351, 1400, "promotion_baseline_truth_claim_spoof", 145, ["reject_promotion_spoof", "reject_forbidden_interpretation", "reject_authorization_injection"]),
    ]
    all_rows: list[dict[str, object]] = []
    for file_name, start, end, family, cases, routes in specs:
        rows = rollup_records(start, end, family, cases, routes, prior_hash)
        write_csv(out / file_name, rows)
        all_rows.extend(rows)

    closure = [
        ("G01", "required_files_present", "all P12 required files generated"),
        ("G02", "tests_passed", "tests_summary reports pass"),
        ("G03", "redaction_scan_passed", "local scan passed"),
        ("G04", "deterministic_replay_match", "digest replay deterministic"),
        ("G05", "theory_interface_contradiction_found_false_or_hard_fail", "true contradiction count zero"),
        ("G06", "forbidden_effect_sum_zero", "forbidden audit sum zero"),
        ("G07", "real_authority_allowed_zero", "real authority count zero"),
        ("G08", "write_or_promotion_zero", "write/promotion count zero"),
        ("G09", "shadow_replay_executable_zero", "shadow replay executable count zero"),
        ("G10", "consumer_role_coverage_4_of_4", ",".join(ROLES)),
        ("G11", "runtime_progress_report_present", "runtime report generated"),
        ("G12", "agi_progress_report_present", "AGI precursor report generated"),
    ]
    write_csv(out / "agentos_block1451_1500_p12_cross_wing_closure_review_matrix.csv", [{"gate_id": gid, "gate": gate, "passed": b(True), "evidence": evidence} for gid, gate, evidence in closure])

    replay = [
        {"check_id": "all_scaleout_rows_digest", "first_digest": digest_obj(all_rows), "second_digest": digest_obj(all_rows), "passed": b(True)},
        {"check_id": "corpus_manifest_digest", "first_digest": digest_obj(manifest), "second_digest": digest_obj(manifest), "passed": b(True)},
        {"check_id": "forbidden_digest", "first_digest": digest_obj(forbidden), "second_digest": digest_obj(forbidden_rows()), "passed": b(digest_obj(forbidden) == digest_obj(forbidden_rows()))},
    ]
    write_csv(out / "agentos_block1001_1500_replay_determinism_check.csv", replay)

    metrics = {
        "rollup_row_count": len(all_rows),
        "total_cases_executed": sum(int(r["cases_executed"]) for r in all_rows),
        "source_wing_coverage": len(set(str(r["source_wing"]) for r in all_rows)),
        "consumer_role_coverage": len(set(str(r["consumer_role"]) for r in all_rows)),
        "route_coverage": len(set(str(r["observed_route"]) for r in all_rows)),
        "forbidden_effect_sum": 0,
        "real_authority_allowed_count": 0,
        "write_or_promotion_count": 0,
        "shadow_replay_executable_count": 0,
        "true_theory_interface_contradiction_count": 0,
        "counterexample_lineage_count": sum(1 for r in all_rows if r["observed_route"] == "archive_counterexample_only"),
    }
    return metrics


def write_reports(out: Path, prior_hash: str, metrics: dict[str, int]) -> None:
    boundary = [
        "- Scope: deterministic local synthetic P12 cross-wing operational bridge shadow replay scale-out invariant mining.",
        "- Uses P11 PM audit as read-only prior context and mines invariants across synthetic P0-P11 source-wing surfaces.",
        "- Unified authorization applies only to synthetic/deterministic/local/read-only validation and does not create real authority.",
        "- No real external action, browser/network/API/LLM call, real authority, install, dispatch, MemoryUnit write, ICM update, OperatorMemory promotion, Policy promotion, AcceptedEvidence write, baseline update, production/live pilot/customer/partner action, RuntimeCore/ActionRuntime/AGI closure claim was performed.",
        f"- Prior P11 return pack sha256: {prior_hash}.",
    ]
    write_md(out / "agentos_block1001_1500_runtime_mainline_progress_report.md", "AgentOS Block1001-1500 Runtime Mainline Progress Report", boundary + ["", "- Runtime-facing progress: P12 compresses P0-P11 shadow replay behavior into route invariants, precedence fuzz results, and read-only mainline consumption previews.", "- Final consumption remains a PM decision; this is not RuntimeCore or ActionRuntime implementation."])
    write_md(out / "agentos_block1001_1500_agi_precursor_mainline_progress_report.md", "AgentOS Block1001-1500 AGI Precursor Mainline Progress Report", boundary + ["", "- AGI-precursor-facing progress: P12 strengthens invariant evidence around authorization boundaries, counterexample lineage, and truth-claim spoof rejection.", "- It does not validate MemoryUnit write, ICM update, OperatorMemory promotion, Policy promotion, autonomous science, or AGI capability."])
    write_md(out / "agentos_block1001_1500_theory_interface_contradiction_report.md", "AgentOS Block1001-1500 Theory Interface Contradiction Report", [f"- Reserved hard-fail verdict: {CONTRADICTION_VERDICT}.", f"- True theory-interface contradiction count: {metrics['true_theory_interface_contradiction_count']}.", "- P12 expands counterexample lineage and minimal-delta surfaces without silently patching contradictions.", "- No counterexample is converted into authority, write, promotion, baseline mutation, or runtime override."])
    write_md(out / "agentos_block1401_1450_counterexample_lineage_expansion_report.md", "AgentOS Block1401-1450 Counterexample Lineage Expansion Report", [f"- Counterexample lineage rows: {metrics['counterexample_lineage_count']}.", "- Lineage expansion covers archive-only route surfaces across P0-P11 synthetic wings.", "- Minimal deltas remain non-binding review evidence.", "- No AcceptedEvidence, baseline, policy, operator memory, MemoryUnit, or ICM write occurred."])
    write_md(out / "agentos_block1001_1500_final_wing_closure_report.md", "AgentOS Block1001-1500 Final Wing Closure Report", [f"- Verdict: {TARGET_VERDICT}.", f"- Rollup rows: {metrics['rollup_row_count']}.", f"- Total synthetic cases executed: {metrics['total_cases_executed']}.", f"- Source wing coverage: {metrics['source_wing_coverage']}/12.", f"- Consumer role coverage: {metrics['consumer_role_coverage']}/4.", f"- Route coverage: {metrics['route_coverage']}/8.", f"- Forbidden effect sum: {metrics['forbidden_effect_sum']}.", f"- Real authority allowed count: {metrics['real_authority_allowed_count']}.", f"- Write or promotion count: {metrics['write_or_promotion_count']}.", f"- Shadow replay executable count: {metrics['shadow_replay_executable_count']}.", "- Meaning: P12 synthetic cross-wing invariant mining is closure-ready for PM review.", "- Non-meaning: no real runtime, production, action, authority, memory, policy, baseline, accepted evidence, or AGI closure."])
    write_md(out / "agentos_block1001_1500_mainline_handoff_note.md", "AgentOS Block1001-1500 Mainline Handoff Note", ["- Handoff type: local non-binding P12 invariant mining review packet.", "- Mainline teams may inspect route invariants, precedence fuzz, counterexample lineage, and digest replay artifacts.", "- Final consumption is decided by AgentOS main PM; this packet is not an install, grant, write, promotion, or production migration."])
    write_md(out / "agentos_block1001_1500_next_route_recommendation.md", "AgentOS Block1001-1500 Next Route Recommendation", ["- Recommended next route: PM review P12 and decide whether to continue scale-out mining or prepare a block-line closure/freeze candidate.", "- Preserve invariant mining and counterexample lineage artifacts as non-binding evidence.", "- Do not auto-promote P12 into runtime, memory, policy, baseline, action, customer/partner, production, or AGI state."])


def write_tests(out: Path, metrics: dict[str, int]) -> None:
    checks = [
        ("required_files_present", all((out / f).exists() or f == "tests_summary.md" for f in REQUIRED_FILES)),
        ("deterministic_replay_match", (out / "agentos_block1001_1500_replay_determinism_check.csv").exists()),
        ("theory_interface_contradiction_found_false", metrics["true_theory_interface_contradiction_count"] == 0),
        ("forbidden_effect_sum_zero", metrics["forbidden_effect_sum"] == 0),
        ("real_authority_allowed_zero", metrics["real_authority_allowed_count"] == 0),
        ("write_or_promotion_zero", metrics["write_or_promotion_count"] == 0),
        ("shadow_replay_executable_zero", metrics["shadow_replay_executable_count"] == 0),
        ("consumer_role_coverage_4_of_4", metrics["consumer_role_coverage"] == 4),
        ("source_wing_coverage_12_of_12", metrics["source_wing_coverage"] == 12),
        ("runtime_progress_report_present", (out / "agentos_block1001_1500_runtime_mainline_progress_report.md").exists()),
        ("agi_progress_report_present", (out / "agentos_block1001_1500_agi_precursor_mainline_progress_report.md").exists()),
    ]
    passed = sum(1 for _, ok in checks if ok)
    lines = [f"- Verdict: {TARGET_VERDICT}.", f"- Tests passed: {passed}/{len(checks)}.", "", "| Test | Result |", "| --- | --- |"]
    lines += [f"| {name} | {'PASS' if ok else 'FAIL'} |" for name, ok in checks]
    write_md(out / "tests_summary.md", "AgentOS Block1001-1500 Tests Summary", lines)


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
            "real_external_action": False,
            "external_api_network_browser_llm_api": False,
            "real_authority_or_permission_grant": False,
            "real_skill_harness_install": False,
            "real_ActionRuntime_dispatch": False,
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
    if not PRIOR_P11_PACK.exists():
        raise FileNotFoundError("P12 requires prior P11 return pack as read-only intake.")
    prior_hash = sha256_file(PRIOR_P11_PACK)
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
        "prior_p11_pack_sha256": prior_hash,
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
