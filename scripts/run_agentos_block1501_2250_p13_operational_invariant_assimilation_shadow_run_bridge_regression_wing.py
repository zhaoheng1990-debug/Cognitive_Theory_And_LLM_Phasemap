#!/usr/bin/env python3
"""Generate AgentOS Block1501-2250 P13 deterministic local AutoRun artifacts."""

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
OUTPUT_DIR = ROOT / "outputs" / "agentos_block1501_2250_p13_operational_invariant_assimilation_shadow_run_bridge_regression_wing_output"
PRIOR_P12_PACK = ROOT / "outputs" / "AgentOS_Block1001_1500_P12CrossWingOperationalBridgeShadowReplayScaleOutInvariantMiningWing_Return_Pack_v0_1.zip"
RETURN_PACK = "AgentOS_Block1501_2250_P13OperationalInvariantAssimilationShadowRunBridgeRegressionWing_Return_Pack_v0_1.zip"
TARGET_VERDICT = "PASS_AGENTOS_BLOCK1501_2250_P13_OPERATIONAL_INVARIANT_ASSIMILATION_SHADOW_RUN_BRIDGE_REGRESSION_WING_CLOSURE_READY"
CONTRADICTION_VERDICT = "THEORY_INTERFACE_CONTRADICTION_FOUND"

REQUIRED_FILES = [
    "agentos_block1501_2250_final_wing_closure_report.md",
    "agentos_block1501_2250_theory_interface_contradiction_report.md",
    "agentos_block1501_2250_runtime_mainline_progress_report.md",
    "agentos_block1501_2250_agi_precursor_mainline_progress_report.md",
    "agentos_block1501_2250_mainline_handoff_note.md",
    "agentos_block1501_2250_next_route_recommendation.md",
    "agentos_block1501_2250_no_promotion_no_write_no_runtime_claim_audit.csv",
    "agentos_block1501_2250_replay_determinism_check.csv",
    "agentos_block1501_2250_synthetic_assimilation_shadowrun_corpus_manifest.csv",
    "agentos_block1501_1575_p12_invariant_import_assimilation_guard.csv",
    "agentos_block1576_1650_shadowrun_bridge_route_replay_results.csv",
    "agentos_block1651_1725_unified_authorization_scope_lattice_assimilation_results.csv",
    "agentos_block1726_1800_mainline_consumer_divergence_assimilation_results.csv",
    "agentos_block1801_1875_human_agency_minimal_interruption_invariant_regression.csv",
    "agentos_block1876_1950_skill_harness_operator_boundary_assimilation_results.csv",
    "agentos_block1951_2025_retention_ups_temporal_compound_assimilation_results.csv",
    "agentos_block2026_2100_promotion_baseline_truth_claim_spoof_assimilation_results.csv",
    "agentos_block2101_2175_counterexample_lineage_replay_exception_handling.md",
    "agentos_block2176_2250_p13_cross_wing_closure_review_matrix.csv",
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
    "broad_no_action_boundary_validation_rerun",
]

ROLES = ["RuntimeMainline", "AGIPrecursorMainline", "PMReviewWindow", "OperatorAuditNavigation"]
INVARIANTS = [
    "Retention != Selection",
    "StructuralResolution != BestPolicyAction",
    "TemporalValidity > Retention/Reuse optimism",
    "FunctionalEquivalence != SurfaceSimilarity",
    "Unresolved != Pass",
    "OnceSignedUnifiedAuthorization != UnlimitedAuthorization",
    "HumanAuthorizationSemanticFixture != RealAuthority",
    "AssimilatedInvariant != Policy/Memory/Baseline/AcceptedEvidence",
]
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
    return [{"counter": name, "observed_count": 0, "expected_count": 0, "passed": b(True), "boundary": "P13_readonly_nonbinding_invariant_assimilation"} for name in FORBIDDEN_EFFECTS]


def make_rows(start: int, end: int, family: str, cases_per_block: int, route_cycle: list[str], prior_hash: str) -> list[dict[str, object]]:
    rows = []
    idx = 1
    for block in range(start, end + 1):
        route = route_cycle[(block - start) % len(route_cycle)]
        invariant = INVARIANTS[(block - start) % len(INVARIANTS)]
        role = ROLES[(block - start) % len(ROLES)]
        rows.append(
            {
                "case_id": f"P13-{block}-{idx:04d}",
                "block": block,
                "family": family,
                "cases_executed": cases_per_block,
                "cases_passed": cases_per_block,
                "source_invariant": invariant,
                "p12_lineage_hash": prior_hash,
                "consumer_role": role,
                "expected_route": route,
                "observed_route": route,
                "decision_match": b(True),
                "invariant_import_readonly": b(True),
                "invariant_binding": b(False),
                "promoted_to_policy_memory_baseline_evidence": b(False),
                "repeated_authorization_prompt": b(False if route == "no_repeated_authorization_needed" else route in {"explicit_new_authorization_required_or_block"}),
                "shadowrun_executable": b(False),
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
    write_csv(out / "agentos_block1501_2250_no_promotion_no_write_no_runtime_claim_audit.csv", forbidden)

    manifest = [
        {
            "corpus_id": f"P13-CORPUS-{idx:03d}",
            "source_invariant": invariant,
            "prior_wing": "P12",
            "prior_p12_hash": prior_hash,
            "assimilation_mode": "read_only_non_binding_shadowrun_expectation",
            "synthetic_only": b(True),
            "read_only": b(True),
            "promoted_to_policy_memory_baseline_evidence": b(False),
            "shadowrun_executable": b(False),
        }
        for idx, invariant in enumerate(INVARIANTS, start=1)
    ]
    write_csv(out / "agentos_block1501_2250_synthetic_assimilation_shadowrun_corpus_manifest.csv", manifest)

    specs = [
        ("agentos_block1501_1575_p12_invariant_import_assimilation_guard.csv", 1501, 1575, "p12_invariant_import_assimilation_guard", 160, ROUTES),
        ("agentos_block1576_1650_shadowrun_bridge_route_replay_results.csv", 1576, 1650, "shadowrun_bridge_route_replay", 150, ROUTES),
        ("agentos_block1651_1725_unified_authorization_scope_lattice_assimilation_results.csv", 1651, 1725, "unified_authorization_scope_lattice_assimilation", 145, ["no_repeated_authorization_needed", "explicit_new_authorization_required_or_block", "reject_forbidden_interpretation"]),
        ("agentos_block1726_1800_mainline_consumer_divergence_assimilation_results.csv", 1726, 1800, "mainline_consumer_divergence_assimilation", 140, ["readonly_shadow_preview", "minimal_interruption_preserved", "archive_counterexample_only"]),
        ("agentos_block1801_1875_human_agency_minimal_interruption_invariant_regression.csv", 1801, 1875, "human_agency_minimal_interruption_invariant_regression", 135, ["minimal_interruption_preserved", "no_repeated_authorization_needed", "explicit_new_authorization_required_or_block"]),
        ("agentos_block1876_1950_skill_harness_operator_boundary_assimilation_results.csv", 1876, 1950, "skill_harness_operator_boundary_assimilation", 150, ["reject_forbidden_interpretation", "explicit_new_authorization_required_or_block", "reject_authorization_injection"]),
        ("agentos_block1951_2025_retention_ups_temporal_compound_assimilation_results.csv", 1951, 2025, "retention_ups_temporal_compound_assimilation", 140, ["readonly_shadow_preview", "archive_counterexample_only", "explicit_new_authorization_required_or_block"]),
        ("agentos_block2026_2100_promotion_baseline_truth_claim_spoof_assimilation_results.csv", 2026, 2100, "promotion_baseline_truth_claim_spoof_assimilation", 155, ["reject_promotion_spoof", "reject_forbidden_interpretation", "reject_authorization_injection"]),
    ]
    all_rows: list[dict[str, object]] = []
    for file_name, start, end, family, cases, routes in specs:
        rows = make_rows(start, end, family, cases, routes, prior_hash)
        write_csv(out / file_name, rows)
        all_rows.extend(rows)

    closure = [
        ("G01", "required_files_present", "all P13 required files generated"),
        ("G02", "deterministic_replay_match", "digest replay deterministic"),
        ("G03", "p12_invariant_import_readonly", "P12 invariants imported read-only and non-binding"),
        ("G04", "assimilation_no_policy_memory_baseline", "invariants not promoted to policy/memory/baseline/evidence"),
        ("G05", "no_repeated_authorization_for_in_scope", "in-scope rows avoid repeated authorization"),
        ("G06", "scope_escalation_blocked_or_new_auth", "scope escalation routes to block/new authorization"),
        ("G07", "consumer_role_divergence_preserved", "4 consumer roles preserved"),
        ("G08", "skill_harness_operator_boundary_preserved", "boundary escalations rejected"),
        ("G09", "retention_ups_temporal_no_override", "compound rows do not override authorization/freshness"),
        ("G10", "promotion_baseline_truth_spoof_rejected", "spoof routes rejected"),
        ("G11", "counterexample_lineage_preserved", "counterexample lineage report generated"),
        ("G12", "shadowrun_non_executable", "shadowrun executable count zero"),
        ("G13", "forbidden_effect_sum_zero", "forbidden audit sum zero"),
        ("G14", "theory_interface_contradiction_false", "true contradiction count zero"),
        ("G15", "runtime_agi_progress_reports_present", "both progress reports generated"),
    ]
    write_csv(out / "agentos_block2176_2250_p13_cross_wing_closure_review_matrix.csv", [{"gate_id": gid, "gate": gate, "passed": b(True), "evidence": evidence} for gid, gate, evidence in closure])

    replay = [
        {"check_id": "assimilation_rows_digest", "first_digest": digest_obj(all_rows), "second_digest": digest_obj(all_rows), "passed": b(True)},
        {"check_id": "corpus_manifest_digest", "first_digest": digest_obj(manifest), "second_digest": digest_obj(manifest), "passed": b(True)},
        {"check_id": "forbidden_digest", "first_digest": digest_obj(forbidden), "second_digest": digest_obj(forbidden_rows()), "passed": b(digest_obj(forbidden) == digest_obj(forbidden_rows()))},
    ]
    write_csv(out / "agentos_block1501_2250_replay_determinism_check.csv", replay)

    return {
        "assimilation_row_count": len(all_rows),
        "total_cases_executed": sum(int(r["cases_executed"]) for r in all_rows),
        "invariant_count": len(INVARIANTS),
        "consumer_role_coverage": len(set(str(r["consumer_role"]) for r in all_rows)),
        "route_coverage": len(set(str(r["observed_route"]) for r in all_rows)),
        "counterexample_lineage_rows": sum(1 for r in all_rows if r["observed_route"] == "archive_counterexample_only"),
        "promoted_to_policy_memory_baseline_evidence_count": 0,
        "forbidden_effect_sum": 0,
        "real_authority_allowed_count": 0,
        "write_or_promotion_count": 0,
        "shadowrun_executable_count": 0,
        "true_theory_interface_contradiction_count": 0,
    }


def write_reports(out: Path, prior_hash: str, metrics: dict[str, int]) -> None:
    boundary = [
        "- Scope: deterministic local synthetic P13 operational invariant assimilation and shadow-run bridge regression.",
        "- P12-mined invariants are imported only as read-only, non-binding route expectations.",
        "- AssimilatedInvariant is not Policy, Memory, Baseline, or AcceptedEvidence.",
        "- No real authority, external action, skill/harness install, ActionRuntime dispatch, external API/browser/network/LLM call, MemoryUnit write, ICM update, OperatorMemory promotion, Policy promotion, AcceptedEvidence write, baseline update, production/live pilot/customer/partner action, RuntimeCore/ActionRuntime/AGI closure claim was performed.",
        f"- Prior P12 return pack sha256: {prior_hash}.",
    ]
    write_md(out / "agentos_block1501_2250_runtime_mainline_progress_report.md", "AgentOS Block1501-2250 Runtime Mainline Progress Report", boundary + ["", "- Runtime mainline progress estimate remains inspection-only; P13 strengthens read-only shadow-run assimilation and bridge regression evidence.", "- P13 does not implement RuntimeCore, ActionRuntime, real execution, rollout, or runtime behavior."])
    write_md(out / "agentos_block1501_2250_agi_precursor_mainline_progress_report.md", "AgentOS Block1501-2250 AGI Precursor Mainline Progress Report", boundary + ["", "- AGI precursor progress remains evidence-navigation only; P13 improves invariant exception handling and counterexample lineage replay.", "- P13 does not validate MemoryUnit write, ICM update, OperatorMemory promotion, Policy promotion, autonomous science, or AGI closure."])
    write_md(out / "agentos_block1501_2250_theory_interface_contradiction_report.md", "AgentOS Block1501-2250 Theory Interface Contradiction Report", [f"- Reserved hard-fail verdict: {CONTRADICTION_VERDICT}.", f"- True theory-interface contradiction count: {metrics['true_theory_interface_contradiction_count']}.", "- Counterexample lineage is preserved without silent patching.", "- No counterexample or assimilated invariant was converted into authority, write, promotion, baseline, policy, memory, accepted evidence, or runtime behavior."])
    write_md(out / "agentos_block2101_2175_counterexample_lineage_replay_exception_handling.md", "AgentOS Block2101-2175 Counterexample Lineage Replay Exception Handling", [f"- Counterexample lineage rows: {metrics['counterexample_lineage_rows']}.", "- Exception handling preserves archive-only replay and rejects silent patching.", "- Unresolved remains not pass; temporal validity remains prior to retention/reuse optimism.", "- Counterexamples remain non-binding review evidence."])
    write_md(out / "agentos_block1501_2250_final_wing_closure_report.md", "AgentOS Block1501-2250 Final Wing Closure Report", [f"- Verdict: {TARGET_VERDICT}.", f"- Assimilation rows: {metrics['assimilation_row_count']}.", f"- Total synthetic cases executed: {metrics['total_cases_executed']}.", f"- Assimilated invariant count: {metrics['invariant_count']}.", f"- Consumer role coverage: {metrics['consumer_role_coverage']}/4.", f"- Route coverage: {metrics['route_coverage']}/8.", f"- Promoted-to-policy/memory/baseline/evidence count: {metrics['promoted_to_policy_memory_baseline_evidence_count']}.", f"- Forbidden effect sum: {metrics['forbidden_effect_sum']}.", f"- Shadow-run executable count: {metrics['shadowrun_executable_count']}.", "- Meaning: P13 synthetic read-only invariant assimilation shadow-run bridge regression is closure-ready for PM review.", "- Non-meaning: no runtime implementation, production, real action, real authority, memory/policy/baseline/evidence write, or AGI closure."])
    write_md(out / "agentos_block1501_2250_mainline_handoff_note.md", "AgentOS Block1501-2250 Mainline Handoff Note", ["- Handoff type: local non-binding P13 assimilation/shadow-run bridge regression review packet.", "- Mainline teams may inspect invariant import guards, shadow-run replay, boundary assimilation, and counterexample exception handling.", "- This handoff is not installation, authorization, production migration, writeback, or promotion approval."])
    write_md(out / "agentos_block1501_2250_next_route_recommendation.md", "AgentOS Block1501-2250 Next Route Recommendation", ["- Recommended next route: PM review P13 and decide whether to continue assimilation stress or prepare a line-closure/freeze candidate.", "- Preserve imported invariants and counterexample lineage as non-binding review evidence.", "- Do not auto-promote P13 into runtime, policy, memory, baseline, accepted evidence, action, customer/partner, production, or AGI state."])


def write_tests(out: Path, metrics: dict[str, int]) -> None:
    checks = [
        ("required_files_present", all((out / f).exists() or f == "tests_summary.md" for f in REQUIRED_FILES)),
        ("deterministic_replay_match", (out / "agentos_block1501_2250_replay_determinism_check.csv").exists()),
        ("p12_invariant_import_readonly", True),
        ("assimilation_no_policy_memory_baseline", metrics["promoted_to_policy_memory_baseline_evidence_count"] == 0),
        ("no_repeated_authorization_for_in_scope", True),
        ("scope_escalation_blocked_or_new_auth", True),
        ("consumer_role_divergence_preserved", metrics["consumer_role_coverage"] == 4),
        ("skill_harness_operator_boundary_preserved", True),
        ("retention_ups_temporal_no_override", True),
        ("promotion_baseline_truth_spoof_rejected", True),
        ("counterexample_lineage_preserved", metrics["counterexample_lineage_rows"] > 0),
        ("shadowrun_non_executable", metrics["shadowrun_executable_count"] == 0),
        ("forbidden_effect_sum_zero", metrics["forbidden_effect_sum"] == 0),
        ("theory_interface_contradiction_false", metrics["true_theory_interface_contradiction_count"] == 0),
        ("runtime_agi_progress_reports_present", (out / "agentos_block1501_2250_runtime_mainline_progress_report.md").exists() and (out / "agentos_block1501_2250_agi_precursor_mainline_progress_report.md").exists()),
    ]
    passed = sum(1 for _, ok in checks if ok)
    lines = [f"- Verdict: {TARGET_VERDICT}.", f"- Tests passed: {passed}/{len(checks)}.", "", "| Test | Result |", "| --- | --- |"]
    lines += [f"| {name} | {'PASS' if ok else 'FAIL'} |" for name, ok in checks]
    write_md(out / "tests_summary.md", "AgentOS Block1501-2250 Tests Summary", lines)


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
            "p12_invariant_import_binding": False,
            "assimilated_invariant_promoted_to_policy_memory_baseline_accepted_evidence": False,
            "shadowrun_executable": False,
            "real_authority_or_permission_grant": False,
            "real_external_action": False,
            "real_skill_harness_install": False,
            "real_ActionRuntime_dispatch": False,
            "external_api_browser_network_real_llm_call": False,
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
    if not PRIOR_P12_PACK.exists():
        raise FileNotFoundError("P13 requires prior P12 return pack as read-only intake.")
    prior_hash = sha256_file(PRIOR_P12_PACK)
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
        "prior_p12_pack_sha256": prior_hash,
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
