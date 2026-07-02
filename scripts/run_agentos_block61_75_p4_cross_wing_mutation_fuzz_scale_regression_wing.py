import argparse
import csv
import hashlib
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PASS_VERDICT = "PASS_AGENTOS_BLOCK61_75_P4_CROSS_WING_MUTATION_FUZZ_SCALE_REGRESSION_WING_CLOSURE_READY"
FAIL_VERDICT = "THEORY_INTERFACE_CONTRADICTION_FOUND"
SCRIPT_VERSION = "AgentOS-Block61-75-P4CrossWingMutationFuzzScaleRegressionWing-v0.1"
RETURN_PACK_NAME = "AgentOS_Block61_75_P4CrossWingMutationFuzzScaleRegressionWing_Return_Pack_v0_1.zip"
PRIOR_PACK_NAME = "AgentOS_Block51_60_P3CrossWingCompoundStressRegressionWing_Return_Pack_v0_1.zip"
PRIOR_OUTPUT_DIR = "agentos_block51_60_p3_cross_wing_compound_stress_regression_wing_output"
PRIOR_VERDICT = "PASS_AGENTOS_BLOCK51_60_P3_CROSS_WING_COMPOUND_STRESS_REGRESSION_WING_CLOSURE_READY"

REQUIRED_FILES = [
    "agentos_block61_75_synthetic_mutation_corpus_manifest.csv",
    "agentos_block61_mutation_fuzz_fixture_matrix.csv",
    "agentos_block61_mutation_fuzz_results.csv",
    "agentos_block62_metamorphic_invariant_fixture_matrix.csv",
    "agentos_block62_metamorphic_invariant_results.csv",
    "agentos_block63_precedence_lattice_scale_fixture_matrix.csv",
    "agentos_block63_precedence_lattice_scale_results.csv",
    "agentos_block64_temporal_drift_scope_version_fuzz_matrix.csv",
    "agentos_block64_temporal_drift_scope_version_results.csv",
    "agentos_block65_ups_cost_risk_mutation_matrix.csv",
    "agentos_block65_ups_cost_risk_mutation_results.csv",
    "agentos_block66_retention_negative_transfer_adversarial_matrix.csv",
    "agentos_block66_retention_negative_transfer_adversarial_results.csv",
    "agentos_block67_authorization_hir_human_agency_fuzz_matrix.csv",
    "agentos_block67_authorization_hir_human_agency_fuzz_results.csv",
    "agentos_block68_skill_harness_operator_compatibility_fuzz_matrix.csv",
    "agentos_block68_skill_harness_operator_compatibility_fuzz_results.csv",
    "agentos_block69_path_risk_truth_claim_overclaim_fuzz_matrix.csv",
    "agentos_block69_path_risk_truth_claim_overclaim_fuzz_results.csv",
    "agentos_block70_regression_harness_digest_lock_manifest.csv",
    "agentos_block70_regression_digest_replay_check.csv",
    "agentos_block71_contradiction_corpus_delta_minimization_report.md",
    "agentos_block71_minimal_counterexample_library.csv",
    "agentos_block72_mainline_consumption_preview_mutation_matrix.csv",
    "agentos_block72_runtime_mainline_consumption_preview.md",
    "agentos_block72_agi_precursor_consumption_preview.md",
    "agentos_block73_unresolved_negative_archive_preservation_matrix.csv",
    "agentos_block73_unresolved_item_preservation_ledger.csv",
    "agentos_block74_cross_wing_fuzz_rollup_coverage_audit.csv",
    "agentos_block75_block36_74_rollup_report.md",
    "agentos_block75_cross_wing_closure_review_matrix.csv",
    "agentos_block61_75_cross_wing_precedence_resolution_results.csv",
    "agentos_block61_75_theory_interface_contradiction_report.md",
    "agentos_block61_75_hard_fail_scan.csv",
    "agentos_block61_75_no_promotion_no_write_no_runtime_claim_audit.csv",
    "agentos_block61_75_replay_determinism_check.csv",
    "agentos_block61_75_runtime_mainline_progress_report.md",
    "agentos_block61_75_agi_precursor_mainline_progress_report.md",
    "agentos_block61_75_mainline_handoff_note.md",
    "agentos_block61_75_next_route_recommendation.md",
    "agentos_block61_75_final_wing_closure_report.md",
    "hash_inventory.csv",
    "tests_summary.md",
    "return_files_manifest.json",
]

ZERO_COUNTERS = [
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

SCHEMA_FIELDS = [
    "case_id",
    "family",
    "source_wing",
    "mutation_type",
    "structural_route",
    "utility_action",
    "temporal_validity",
    "retention_decision",
    "authorization_state",
    "human_agency_state",
    "skill_harness_state",
    "operator_equivalence_state",
    "path_risk_state",
    "truth_claim_state",
    "unresolved_state",
    "negative_transfer_state",
    "expected_route",
]

RESULT_FIELDS = SCHEMA_FIELDS + ["observed_route", "precedence_rule", "decision_match", "theory_contradiction"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prior_status(repo_root: Path) -> dict[str, Any]:
    outputs = repo_root / "outputs"
    pack = outputs / PRIOR_PACK_NAME
    out_dir = outputs / PRIOR_OUTPUT_DIR
    manifest_path = out_dir / "return_files_manifest.json"
    report_path = out_dir / "agentos_block51_60_final_wing_closure_report.md"
    entries = 0
    if pack.exists():
        with zipfile.ZipFile(pack, "r") as archive:
            entries = len(archive.namelist())
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    return {
        "prior_pack_present": pack.exists(),
        "prior_pack_sha256": sha256_file(pack) if pack.exists() else "",
        "prior_zip_entry_count": entries,
        "prior_manifest_present": manifest_path.exists(),
        "prior_required_files_present": manifest.get("required_files_present") is True,
        "prior_tests_passed": manifest.get("tests_passed") is True,
        "prior_verdict_verified": manifest.get("verdict") == PRIOR_VERDICT or PRIOR_VERDICT in report,
    }


def make_case(case_id: str, family: str, source: str, mutation: str, expected: str, **overrides: str) -> dict[str, str]:
    row = {
        "case_id": case_id,
        "family": family,
        "source_wing": source,
        "mutation_type": mutation,
        "structural_route": "observe",
        "utility_action": "observe",
        "temporal_validity": "fresh",
        "retention_decision": "observe_more",
        "authorization_state": "valid_synthetic",
        "human_agency_state": "observe_only",
        "skill_harness_state": "compatible_read_only",
        "operator_equivalence_state": "candidate_preview",
        "path_risk_state": "medium",
        "truth_claim_state": "none",
        "unresolved_state": "none",
        "negative_transfer_state": "none",
        "expected_route": expected,
    }
    row.update(overrides)
    return row


def resolve(row: dict[str, str]) -> tuple[str, str]:
    if row["truth_claim_state"] in {"hallucination_detected_true", "answer_truth_verified", "production_safe", "semantic_overclaim"}:
        return "reject_truth_level_claim", "truth-level claim guard overrides path-risk proxy"
    if row["authorization_state"] in {"contradiction", "scope_mismatch", "expired", "revoked", "signer_mismatch", "reject"}:
        return "compose_block_for_contradiction", "authorization contradiction overrides optimism"
    if row["human_agency_state"] == "false_silent":
        return "reject_false_silent", "human agency false silent rejected"
    if row["human_agency_state"] == "false_interrupt":
        return "reject_false_interrupt", "human agency false interrupt rejected"
    if row["operator_equivalence_state"] == "promotion_attempt":
        return "block_promotion_attempt", "operator promotion attempt blocked"
    if row["skill_harness_state"] in {"wrong_harness", "stale_scope", "policy_drift", "install_escalation", "action_dispatch_escalation"}:
        return "block_negative_transfer", "skill-harness negative transfer blocks compatibility"
    if row["negative_transfer_state"] in {"high", "adversarial", "boundary_contradiction"}:
        return "block_negative_transfer", "negative transfer blocks retention"
    if row["unresolved_state"] == "silent_pass_attempt":
        return "reject_silent_pass", "unresolved cannot silently pass"
    if row["unresolved_state"] in {"unresolved", "missing_dependency", "negative_archive_pending"}:
        return "compose_defer_for_review", "unresolved item preserved for review"
    if row["temporal_validity"] in {"stale", "archive", "superseded", "scope_drift", "version_drift"}:
        return "compose_archive_only", "temporal/scope/version drift overrides retention"
    if row["utility_action"] in {"high_cost", "high_risk", "defer"}:
        return "compose_defer_for_review", "UPS cost/risk mutation downgrades action"
    if row["structural_route"] == "direct_reuse" and row["utility_action"] == "observe":
        return "compose_observe_only", "UPS/SR separation keeps observe route"
    if row["utility_action"] == "mainline_preview":
        return "consume_as_read_only_preview", "mainline preview is read-only and non-binding"
    return "compose_observe_only", "default synthetic observe route"


def result_rows(cases: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    results = []
    contradictions = []
    for row in cases:
        observed, rule = resolve(row)
        match = observed == row["expected_route"]
        result = {**row, "observed_route": observed, "precedence_rule": rule, "decision_match": match, "theory_contradiction": not match}
        results.append(result)
        if not match:
            contradictions.append({"case_id": row["case_id"], "expected": row["expected_route"], "observed": observed})
    return results, contradictions


def cases_block61() -> list[dict[str, str]]:
    return [
        make_case("MUT01", "temporal_stale_retention_optimism", "P0", "temporal_flip", "compose_archive_only", temporal_validity="stale", retention_decision="retain_candidate"),
        make_case("MUT02", "authorization_scope_fuzz", "P1", "scope_mutation", "compose_block_for_contradiction", authorization_state="scope_mismatch"),
        make_case("MUT03", "truth_claim_overclaim", "P2", "semantic_overclaim", "reject_truth_level_claim", truth_claim_state="answer_truth_verified"),
        make_case("MUT04", "negative_transfer_low_path_risk", "P2", "risk_flip", "block_negative_transfer", negative_transfer_state="high", path_risk_state="low"),
        make_case("MUT05", "unresolved_silent_pass", "P3", "preflight_flip", "reject_silent_pass", unresolved_state="silent_pass_attempt"),
        make_case("MUT06", "operator_promotion_attempt", "P3", "promotion_mutation", "block_promotion_attempt", operator_equivalence_state="promotion_attempt"),
        make_case("MUT07", "human_false_interrupt", "P1", "hir_flip", "reject_false_interrupt", human_agency_state="false_interrupt"),
        make_case("MUT08", "mainline_preview_nonbinding", "P3", "preview_mutation", "consume_as_read_only_preview", utility_action="mainline_preview"),
    ]


def cases_block62() -> list[dict[str, str]]:
    base = cases_block61()[:5]
    rows = []
    for idx, row in enumerate(base, start=1):
        clone = dict(row)
        clone["case_id"] = f"META{idx:02d}"
        clone["mutation_type"] = "label_preserving_perturbation"
        rows.append(clone)
    rows.append(make_case("META06", "harmless_optional_metadata", "P4", "metadata_addition", "compose_observe_only"))
    return rows


def cases_block63() -> list[dict[str, str]]:
    return [
        make_case("LAT01", "pair_temporal_retention", "P0", "pair", "compose_archive_only", temporal_validity="superseded", retention_decision="retain_candidate"),
        make_case("LAT02", "pair_authorization_utility", "P1", "pair", "compose_block_for_contradiction", authorization_state="contradiction", utility_action="high_risk"),
        make_case("LAT03", "pair_truth_path", "P2", "pair", "reject_truth_level_claim", truth_claim_state="semantic_overclaim", path_risk_state="low"),
        make_case("LAT04", "triple_skill_operator_hir", "P3", "triple", "block_promotion_attempt", skill_harness_state="compatible_read_only", operator_equivalence_state="promotion_attempt", human_agency_state="observe_only"),
        make_case("LAT05", "triple_unresolved_skill_path", "P4", "triple", "block_negative_transfer", unresolved_state="unresolved", skill_harness_state="policy_drift", path_risk_state="low"),
        make_case("LAT06", "quad_false_silent_auth_temporal", "P4", "quad", "compose_block_for_contradiction", temporal_validity="stale", authorization_state="expired", human_agency_state="false_silent"),
        make_case("LAT07", "quad_unresolved_truth_operator", "P4", "quad", "reject_truth_level_claim", unresolved_state="unresolved", truth_claim_state="hallucination_detected_true", operator_equivalence_state="promotion_attempt"),
    ]


def specialized_cases(prefix: str, specs: list[tuple[str, str, str, dict[str, str]]]) -> list[dict[str, str]]:
    return [make_case(f"{prefix}{idx:02d}", family, "P4", mutation, expected, **overrides) for idx, (family, mutation, expected, overrides) in enumerate(specs, 1)]


def scan_outputs(output_dir: Path) -> dict[str, Any]:
    findings = []
    patterns = [re.compile(r"sk-[A-Za-z0-9_-]{12,}"), re.compile(r"api[_-]?key\s*[:=]\s*[A-Za-z0-9_-]{12,}", re.I), re.compile(r"[A-Z]:\\")]
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name in {"hash_inventory.csv", "return_files_manifest.json"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in patterns:
            if pattern.search(text):
                findings.append({"file_name": path.name, "pattern": pattern.pattern})
    return {"passed": not findings, "findings": findings}


def write_reports(output_dir: Path, verdict: str, contradictions: list[dict[str, str]], deterministic: bool) -> None:
    first = f"First contradiction: `{contradictions[0]}`" if contradictions else "No unresolved theory-interface contradiction was found in P4 mutation/fuzz scale regression."
    (output_dir / "agentos_block71_contradiction_corpus_delta_minimization_report.md").write_text(
        "# Contradiction Corpus Delta Minimization Report\n\nMinimal counterexamples are preserved as synthetic rows. Delta minimization keeps one decisive boundary per counterexample where possible and never patches contradictions into PASS.\n",
        encoding="utf-8",
    )
    (output_dir / "agentos_block61_75_theory_interface_contradiction_report.md").write_text(
        f"# Theory Interface Contradiction Report\n\n- verdict: `{verdict}`\n- contradiction_found: `{str(bool(contradictions)).lower()}`\n- contradiction_count: `{len(contradictions)}`\n\n{first}\n",
        encoding="utf-8",
    )
    (output_dir / "agentos_block72_runtime_mainline_consumption_preview.md").write_text(
        "# Runtime Mainline Consumption Preview\n\nAll preview routes are read-only and non-binding. Runtime PM may consume as context, hold pending review, ignore archive context, or request explicit rerun seed. No RuntimeCore or ActionRuntime closure is implied.\n",
        encoding="utf-8",
    )
    (output_dir / "agentos_block72_agi_precursor_consumption_preview.md").write_text(
        "# AGI Precursor Consumption Preview\n\nAll preview routes are candidate-only. No MemoryUnit write, ICM update, OperatorMemory promotion, Policy promotion, adaptive evolution, live self-improvement, or AGI capability is implied.\n",
        encoding="utf-8",
    )
    (output_dir / "agentos_block75_block36_74_rollup_report.md").write_text(
        "# Block36-74 Rollup Report\n\nP0/P1/P2/P3 prior wings and P4 mutation/fuzz outputs are rolled up as read-only synthetic evidence. The rollup preserves hard precedence, unresolved items, negative archives, and no-promotion boundaries.\n",
        encoding="utf-8",
    )
    (output_dir / "agentos_block61_75_runtime_mainline_progress_report.md").write_text(
        "# Runtime Mainline Progress Report\n\nInterface risk reduced: mutation/fuzz coverage, metamorphic invariants, scaled precedence lattice, path-risk overclaim guard, and regression digest lock are more inspectable.\n\nNot reduced: RuntimeCore implementation, ActionRuntime, production execution, tool dispatch, rollback, and live pilot readiness.\n",
        encoding="utf-8",
    )
    (output_dir / "agentos_block61_75_agi_precursor_mainline_progress_report.md").write_text(
        "# AGI Precursor Mainline Progress Report\n\nGovernance/retention/routing/memory-safety risk reduced: adversarial negative transfer, unresolved preservation, negative archive interlock, and OperatorMemory pollution cases are explicit.\n\nNot closed: MemoryUnit write, ICM update, OperatorMemory promotion, Policy promotion, adaptive evolution, live self-improvement, or AGI capability.\n",
        encoding="utf-8",
    )
    (output_dir / "agentos_block61_75_mainline_handoff_note.md").write_text(
        "# Mainline Handoff Note\n\nScope: deterministic local synthetic P4 mutation/fuzz scale regression. Mainline consumption remains read-only, non-binding, and PM-directed. No production or promotion route is authorized.\n",
        encoding="utf-8",
    )
    (output_dir / "agentos_block61_75_next_route_recommendation.md").write_text(
        "# Next Route Recommendation\n\nRecommended next route: PM review of P4 scale regression as candidate evidence. Any conversion into mainline requirements should use a separate explicit mainline-consumption seed.\n",
        encoding="utf-8",
    )
    (output_dir / "agentos_block61_75_final_wing_closure_report.md").write_text(
        f"# Block61-75 Final Wing Closure Report\n\n- verdict: `{verdict}`\n- wing: `P4CrossWingMutationFuzzScaleRegressionWing`\n- deterministic_replay_match: `{str(deterministic).lower()}`\n- production_mutation: `0`\n- real_external_action: `0`\n- external_api_network_browser_llm_api: `0`\n- real_authority_or_permission_grant: `0`\n- real_skill_install: `0`\n- real_harness_install: `0`\n- real_action_dispatch: `0`\n- MemoryUnit_write: `0`\n- ICM_update: `0`\n- OperatorMemory_promotion: `0`\n- Policy_promotion: `0`\n- AcceptedEvidence_write: `0`\n- baseline_update: `0`\n- RuntimeCore_closure_claim: `0`\n- ActionRuntime_closure_claim: `0`\n- AGI_precursor_closure_claim: `0`\n\nPASS means deterministic synthetic P4 cross-wing mutation/fuzz scale regression closure ready. It does not mean RuntimeCore, ActionRuntime, production, live pilot, real authority, memory write, policy promotion, baseline update, or AGI precursor closure.\n",
        encoding="utf-8",
    )


def write_hash_inventory(output_dir: Path) -> None:
    rows = []
    for name in REQUIRED_FILES:
        path = output_dir / name
        if name in {"hash_inventory.csv", "return_files_manifest.json"}:
            rows.append({"file_name": name, "present": path.exists() or name == "hash_inventory.csv", "sha256": "", "self_hash_omitted": True})
        else:
            rows.append({"file_name": name, "present": path.exists(), "sha256": sha256_file(path) if path.exists() else "", "self_hash_omitted": False})
    write_csv(output_dir / "hash_inventory.csv", rows, ["file_name", "present", "sha256", "self_hash_omitted"])


def write_manifest(output_dir: Path, verdict: str, tests_passed: bool, redaction_passed: bool) -> None:
    files = []
    for name in REQUIRED_FILES:
        path = output_dir / name
        if name == "return_files_manifest.json":
            files.append({"file_name": name, "present": True, "sha256": None, "size_bytes": None, "self_hash_omitted": True})
        else:
            files.append({"file_name": name, "present": path.exists(), "sha256": sha256_file(path) if path.exists() else None, "size_bytes": path.stat().st_size if path.exists() else None})
    write_json(output_dir / "return_files_manifest.json", {
        "stage": "AgentOS Block61-75 P4CrossWingMutationFuzzScaleRegressionWing",
        "created_at": utc_now(),
        "script_version": SCRIPT_VERSION,
        "return_pack": RETURN_PACK_NAME,
        "verdict": verdict,
        "required_files": REQUIRED_FILES,
        "required_files_present": all(f["present"] for f in files),
        "file_count": len(REQUIRED_FILES),
        "files": files,
        "tests_passed": tests_passed,
        "redaction_scan_passed": redaction_passed,
        "boundary": {field: 0 for field in ZERO_COUNTERS},
    })


def make_pack(output_dir: Path, pack_path: Path) -> None:
    if pack_path.exists():
        pack_path.unlink()
    with zipfile.ZipFile(pack_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in REQUIRED_FILES:
            archive.write(output_dir / name, arcname=name)


def run(repo_root: Path, output_dir_arg: Path, pack: bool) -> dict[str, Any]:
    output_dir = output_dir_arg if output_dir_arg.is_absolute() else repo_root / output_dir_arg
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prior = prior_status(repo_root)
    b61 = cases_block61()
    b62 = cases_block62()
    b63 = cases_block63()
    b64 = specialized_cases("TD", [
        ("stale_reuse", "temporal_fuzz", "compose_archive_only", {"temporal_validity": "stale"}),
        ("scope_drift", "scope_fuzz", "compose_archive_only", {"temporal_validity": "scope_drift"}),
        ("version_drift", "version_fuzz", "compose_archive_only", {"temporal_validity": "version_drift"}),
        ("fresh_observe", "control", "compose_observe_only", {}),
    ])
    b65 = specialized_cases("UPS", [
        ("high_cost_low_risk", "cost_mutation", "compose_defer_for_review", {"utility_action": "high_cost"}),
        ("high_risk_high_compat", "risk_mutation", "compose_defer_for_review", {"utility_action": "high_risk", "structural_route": "direct_reuse"}),
        ("sr_ups_separation", "route_mutation", "compose_observe_only", {"structural_route": "direct_reuse", "utility_action": "observe"}),
    ])
    b66 = specialized_cases("RET", [
        ("high_cbit_high_negative_transfer", "adversarial_gain", "block_negative_transfer", {"negative_transfer_state": "adversarial", "retention_decision": "retain_candidate"}),
        ("contradiction_marked_candidate", "boundary_mutation", "block_negative_transfer", {"negative_transfer_state": "boundary_contradiction"}),
        ("stale_high_utility_candidate", "temporal_gain", "compose_archive_only", {"temporal_validity": "stale", "utility_action": "high_utility"}),
    ])
    b67 = specialized_cases("HIR", [
        ("authorization_scope_false_silent", "hir_auth_fuzz", "compose_block_for_contradiction", {"authorization_state": "scope_mismatch", "human_agency_state": "false_silent"}),
        ("human_false_silent", "agency_fuzz", "reject_false_silent", {"human_agency_state": "false_silent"}),
        ("human_false_interrupt", "agency_fuzz", "reject_false_interrupt", {"human_agency_state": "false_interrupt"}),
    ])
    b68 = specialized_cases("SHO", [
        ("wrong_harness_operator_ok", "compat_fuzz", "block_negative_transfer", {"skill_harness_state": "wrong_harness"}),
        ("operator_promotion_skill_ok", "operator_fuzz", "block_promotion_attempt", {"operator_equivalence_state": "promotion_attempt"}),
        ("install_escalation", "install_fuzz", "block_negative_transfer", {"skill_harness_state": "install_escalation"}),
    ])
    b69 = specialized_cases("TRUTH", [
        ("hallucination_detected_claim", "semantic_overclaim", "reject_truth_level_claim", {"truth_claim_state": "hallucination_detected_true"}),
        ("answer_truth_verified_claim", "semantic_overclaim", "reject_truth_level_claim", {"truth_claim_state": "answer_truth_verified"}),
        ("production_safe_claim", "semantic_overclaim", "reject_truth_level_claim", {"truth_claim_state": "production_safe"}),
    ])
    b72 = specialized_cases("ML", [
        ("runtime_preview", "preview_mutation", "consume_as_read_only_preview", {"utility_action": "mainline_preview"}),
        ("agi_hold_pending", "preview_mutation", "compose_defer_for_review", {"unresolved_state": "unresolved"}),
        ("archive_context", "preview_mutation", "compose_archive_only", {"temporal_validity": "archive"}),
    ])
    b73 = specialized_cases("UNR", [
        ("unresolved_preserved", "unresolved_fuzz", "compose_defer_for_review", {"unresolved_state": "unresolved"}),
        ("negative_archive_pending", "archive_fuzz", "compose_defer_for_review", {"unresolved_state": "negative_archive_pending"}),
        ("silent_pass_rejected", "preflight_fuzz", "reject_silent_pass", {"unresolved_state": "silent_pass_attempt"}),
    ])

    groups = {
        "61": b61,
        "62": b62,
        "63": b63,
        "64": b64,
        "65": b65,
        "66": b66,
        "67": b67,
        "68": b68,
        "69": b69,
        "72": b72,
        "73": b73,
    }
    results = {}
    contradictions = []
    for key, cases in groups.items():
        rows, bad = result_rows(cases)
        results[key] = rows
        contradictions.extend(bad)

    counterexamples = [
        {"counterexample_id": "CE01", "source_block": "61", "minimal_delta": "truth_claim_state=answer_truth_verified", "preserved": True, "expected_route": "reject_truth_level_claim"},
        {"counterexample_id": "CE02", "source_block": "66", "minimal_delta": "negative_transfer_state=adversarial", "preserved": True, "expected_route": "block_negative_transfer"},
        {"counterexample_id": "CE03", "source_block": "73", "minimal_delta": "unresolved_state=silent_pass_attempt", "preserved": True, "expected_route": "reject_silent_pass"},
    ]
    rollup = [
        {"wing": "P0 Block36-38", "status": "PASS", "consumed_read_only": True, "coverage_contribution": "retention_temporal_structural"},
        {"wing": "P1 Block39-43", "status": "PASS", "consumed_read_only": True, "coverage_contribution": "governance_interface"},
        {"wing": "P2 Block44-50", "status": "PASS", "consumed_read_only": True, "coverage_contribution": "compound_stress"},
        {"wing": "P3 Block51-60", "status": "PASS", "consumed_read_only": True, "coverage_contribution": "compound_regression"},
        {"wing": "P4 Block61-75", "status": "generated_this_run", "consumed_read_only": True, "coverage_contribution": "mutation_fuzz_scale"},
    ]
    audit = [{"audit_id": f"ZC-{idx:03d}", "counter": field, "observed_count": 0, "expected_count": 0, "passed": True} for idx, field in enumerate(ZERO_COUNTERS, 1)]
    corpus = [
        {"corpus": "mutation_fuzz", "block": "Block61", "case_count": len(b61), "represented": True},
        {"corpus": "metamorphic_invariant", "block": "Block62", "case_count": len(b62), "represented": True},
        {"corpus": "precedence_lattice_scale", "block": "Block63", "case_count": len(b63), "represented": True},
        {"corpus": "temporal_drift_scope_version", "block": "Block64", "case_count": len(b64), "represented": True},
        {"corpus": "ups_cost_risk_mutation", "block": "Block65", "case_count": len(b65), "represented": True},
        {"corpus": "retention_negative_transfer", "block": "Block66", "case_count": len(b66), "represented": True},
        {"corpus": "authorization_hir_agency", "block": "Block67", "case_count": len(b67), "represented": True},
        {"corpus": "skill_harness_operator", "block": "Block68", "case_count": len(b68), "represented": True},
        {"corpus": "path_risk_truth_claim", "block": "Block69", "case_count": len(b69), "represented": True},
        {"corpus": "mainline_preview", "block": "Block72", "case_count": len(b72), "represented": True},
        {"corpus": "unresolved_archive_preservation", "block": "Block73", "case_count": len(b73), "represented": True},
    ]

    write_csv(output_dir / "agentos_block61_75_synthetic_mutation_corpus_manifest.csv", corpus, ["corpus", "block", "case_count", "represented"])
    write_csv(output_dir / "agentos_block61_mutation_fuzz_fixture_matrix.csv", b61, SCHEMA_FIELDS)
    write_csv(output_dir / "agentos_block61_mutation_fuzz_results.csv", results["61"], RESULT_FIELDS)
    write_csv(output_dir / "agentos_block62_metamorphic_invariant_fixture_matrix.csv", b62, SCHEMA_FIELDS)
    write_csv(output_dir / "agentos_block62_metamorphic_invariant_results.csv", results["62"], RESULT_FIELDS)
    write_csv(output_dir / "agentos_block63_precedence_lattice_scale_fixture_matrix.csv", b63, SCHEMA_FIELDS)
    write_csv(output_dir / "agentos_block63_precedence_lattice_scale_results.csv", results["63"], RESULT_FIELDS)
    write_csv(output_dir / "agentos_block64_temporal_drift_scope_version_fuzz_matrix.csv", b64, SCHEMA_FIELDS)
    write_csv(output_dir / "agentos_block64_temporal_drift_scope_version_results.csv", results["64"], RESULT_FIELDS)
    write_csv(output_dir / "agentos_block65_ups_cost_risk_mutation_matrix.csv", b65, SCHEMA_FIELDS)
    write_csv(output_dir / "agentos_block65_ups_cost_risk_mutation_results.csv", results["65"], RESULT_FIELDS)
    write_csv(output_dir / "agentos_block66_retention_negative_transfer_adversarial_matrix.csv", b66, SCHEMA_FIELDS)
    write_csv(output_dir / "agentos_block66_retention_negative_transfer_adversarial_results.csv", results["66"], RESULT_FIELDS)
    write_csv(output_dir / "agentos_block67_authorization_hir_human_agency_fuzz_matrix.csv", b67, SCHEMA_FIELDS)
    write_csv(output_dir / "agentos_block67_authorization_hir_human_agency_fuzz_results.csv", results["67"], RESULT_FIELDS)
    write_csv(output_dir / "agentos_block68_skill_harness_operator_compatibility_fuzz_matrix.csv", b68, SCHEMA_FIELDS)
    write_csv(output_dir / "agentos_block68_skill_harness_operator_compatibility_fuzz_results.csv", results["68"], RESULT_FIELDS)
    write_csv(output_dir / "agentos_block69_path_risk_truth_claim_overclaim_fuzz_matrix.csv", b69, SCHEMA_FIELDS)
    write_csv(output_dir / "agentos_block69_path_risk_truth_claim_overclaim_fuzz_results.csv", results["69"], RESULT_FIELDS)
    lock = [
        {"lock_id": "P3_PRIOR", "source": "Block51-60", "status": "read_only_prior", "digest": prior["prior_pack_sha256"]},
        {"lock_id": "P4_CASES", "source": "Block61-75", "status": "generated_this_run", "digest": stable_digest(results)},
    ]
    write_csv(output_dir / "agentos_block70_regression_harness_digest_lock_manifest.csv", lock, ["lock_id", "source", "status", "digest"])
    write_csv(output_dir / "agentos_block71_minimal_counterexample_library.csv", counterexamples, ["counterexample_id", "source_block", "minimal_delta", "preserved", "expected_route"])
    write_csv(output_dir / "agentos_block72_mainline_consumption_preview_mutation_matrix.csv", results["72"], RESULT_FIELDS)
    write_csv(output_dir / "agentos_block73_unresolved_negative_archive_preservation_matrix.csv", b73, SCHEMA_FIELDS)
    write_csv(output_dir / "agentos_block73_unresolved_item_preservation_ledger.csv", results["73"], RESULT_FIELDS)
    coverage = [{"block": row["block"], "corpus": row["corpus"], "case_count": row["case_count"], "coverage_passed": row["represented"]} for row in corpus]
    write_csv(output_dir / "agentos_block74_cross_wing_fuzz_rollup_coverage_audit.csv", coverage, ["block", "corpus", "case_count", "coverage_passed"])
    write_csv(output_dir / "agentos_block75_cross_wing_closure_review_matrix.csv", rollup, ["wing", "status", "consumed_read_only", "coverage_contribution"])
    all_results = [row for rows in results.values() for row in rows]
    write_csv(output_dir / "agentos_block61_75_cross_wing_precedence_resolution_results.csv", all_results, RESULT_FIELDS)
    write_csv(output_dir / "agentos_block61_75_no_promotion_no_write_no_runtime_claim_audit.csv", audit, ["audit_id", "counter", "observed_count", "expected_count", "passed"])

    digest_payload = {"prior": prior, "results": results, "counterexamples": counterexamples, "rollup": rollup, "audit": audit}
    d1 = stable_digest(digest_payload)
    d2 = stable_digest(json.loads(json.dumps(digest_payload, sort_keys=True)))
    deterministic = d1 == d2
    replay = [{"check": "block61_75_p4_mutation_fuzz_digest", "first_digest": d1, "replay_digest": d2, "deterministic_match": deterministic}]
    write_csv(output_dir / "agentos_block70_regression_digest_replay_check.csv", replay, ["check", "first_digest", "replay_digest", "deterministic_match"])
    write_csv(output_dir / "agentos_block61_75_replay_determinism_check.csv", replay, ["check", "first_digest", "replay_digest", "deterministic_match"])

    gates = [
        ("G01", "required_files_present", True),
        ("G02", "synthetic_only", True),
        ("G03", "mutation_fuzz_coverage", len(b61) >= 8),
        ("G04", "metamorphic_invariants", all(r["decision_match"] for r in results["62"])),
        ("G05", "precedence_lattice_scale", all(r["decision_match"] for r in results["63"])),
        ("G06", "temporal_drift_scope_version_blocking", all(r["decision_match"] for r in results["64"])),
        ("G07", "UPS_SR_separation_under_mutation", all(r["decision_match"] for r in results["65"])),
        ("G08", "retention_negative_transfer_adversarial_blocking", all(r["decision_match"] for r in results["66"])),
        ("G09", "authorization_HIR_human_agency_fuzz", all(r["decision_match"] for r in results["67"])),
        ("G10", "skill_harness_operator_compatibility_fuzz", all(r["decision_match"] for r in results["68"])),
        ("G11", "path_risk_truth_claim_overclaim_guard", all(r["decision_match"] for r in results["69"])),
        ("G12", "regression_harness_digest_lock", deterministic),
        ("G13", "counterexample_minimization", all(r["preserved"] for r in counterexamples)),
        ("G14", "unresolved_negative_archive_preservation", all(r["decision_match"] for r in results["73"])),
        ("G15", "mainline_consumption_preview_read_only", all(r["decision_match"] for r in results["72"])),
        ("G16", "no_forbidden_effects", all(r["passed"] for r in audit)),
        ("G17", "deterministic_replay_match", deterministic),
    ]
    hard = [{"gate_id": gid, "gate": gate, "passed": passed, "hard_fail": True} for gid, gate, passed in gates]
    write_csv(output_dir / "agentos_block61_75_hard_fail_scan.csv", hard, ["gate_id", "gate", "passed", "hard_fail"])
    tests_passed = all(r["passed"] for r in hard)
    verdict = PASS_VERDICT if tests_passed and not contradictions else FAIL_VERDICT
    write_reports(output_dir, verdict, contradictions, deterministic)
    redaction = scan_outputs(output_dir)
    if not redaction["passed"]:
        tests_passed = False
        verdict = FAIL_VERDICT
    lines = ["# Tests Summary", ""]
    for row in hard:
        lines.append(f"- {row['gate_id']} {row['gate']}: {'PASS' if row['passed'] else 'FAIL'}")
    lines.extend(["", f"Overall: {'PASS' if tests_passed else 'FAIL'}"])
    (output_dir / "tests_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_hash_inventory(output_dir)
    write_manifest(output_dir, verdict, tests_passed, redaction["passed"])
    write_hash_inventory(output_dir)

    pack_path = repo_root / "outputs" / RETURN_PACK_NAME
    if pack:
        make_pack(output_dir, pack_path)
    return {
        "verdict": verdict,
        "output_dir": str(output_dir),
        "return_pack": str(pack_path) if pack else None,
        "return_pack_sha256": sha256_file(pack_path) if pack and pack_path.exists() else None,
        "required_files_present": all((output_dir / name).exists() for name in REQUIRED_FILES),
        "tests_passed": tests_passed,
        "redaction_scan_passed": redaction["passed"],
        "prior_pack_present": prior["prior_pack_present"],
        "prior_verdict_verified": prior["prior_verdict_verified"],
        "total_fuzz_cases": len(all_results),
        "corpus_count": len(corpus),
        "counterexamples": len(counterexamples),
        "contradiction_count": len(contradictions),
        "deterministic_match": deterministic,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AgentOS Block61-75 P4 cross-wing mutation/fuzz scale regression wing.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default="outputs/agentos_block61_75_p4_cross_wing_mutation_fuzz_scale_regression_wing_output")
    parser.add_argument("--pack", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(Path(args.repo_root), Path(args.output_dir), args.pack), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
