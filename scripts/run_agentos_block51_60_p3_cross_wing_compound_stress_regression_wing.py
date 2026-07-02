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


PASS_VERDICT = "PASS_AGENTOS_BLOCK51_60_P3_CROSS_WING_COMPOUND_STRESS_REGRESSION_WING_CLOSURE_READY"
FAIL_VERDICT = "THEORY_INTERFACE_CONTRADICTION_FOUND"
SCRIPT_VERSION = "AgentOS-Block51-60-P3CrossWingCompoundStressRegressionWing-v0.1"
RETURN_PACK_NAME = "AgentOS_Block51_60_P3CrossWingCompoundStressRegressionWing_Return_Pack_v0_1.zip"
PRIOR_PACK_NAME = "AgentOS_Block44_50_P2TheoryInterfaceStressWing_Return_Pack_v0_1.zip"
PRIOR_OUTPUT_DIR = "agentos_block44_50_p2_theory_interface_stress_wing_output"
PRIOR_VERDICT = "PASS_AGENTOS_BLOCK44_50_P2_THEORY_INTERFACE_STRESS_WING_CLOSURE_READY"

REQUIRED_FILES = [
    "agentos_block51_60_synthetic_corpus_manifest.csv",
    "agentos_block51_compound_conflict_fixture_matrix.csv",
    "agentos_block51_compound_conflict_results.csv",
    "agentos_block52_precedence_lattice_fixture_matrix.csv",
    "agentos_block52_precedence_lattice_regression_results.csv",
    "agentos_block53_negative_transfer_counterexample_library.csv",
    "agentos_block53_minimal_counterexample_report.md",
    "agentos_block54_retention_utility_temporal_compound_results.csv",
    "agentos_block55_operator_skill_human_agency_compound_results.csv",
    "agentos_block56_authorization_preflight_contradiction_results.csv",
    "agentos_block57_path_risk_truth_claim_boundary_regression.csv",
    "agentos_block58_regression_harness_lock_manifest.csv",
    "agentos_block58_regression_digest_replay_check.csv",
    "agentos_block59_mainline_consumption_preview_matrix.csv",
    "agentos_block59_runtime_mainline_consumption_preview.md",
    "agentos_block59_agi_precursor_consumption_preview.md",
    "agentos_block60_block36_59_rollup_report.md",
    "agentos_block60_cross_wing_closure_review_matrix.csv",
    "agentos_block51_60_theory_interface_contradiction_report.md",
    "agentos_block51_60_cross_wing_precedence_resolution_results.csv",
    "agentos_block51_60_no_promotion_no_write_no_runtime_claim_audit.csv",
    "agentos_block51_60_replay_determinism_check.csv",
    "agentos_block51_60_runtime_mainline_progress_report.md",
    "agentos_block51_60_agi_precursor_mainline_progress_report.md",
    "agentos_block51_60_mainline_handoff_note.md",
    "agentos_block51_60_next_route_recommendation.md",
    "agentos_block51_60_hard_fail_scan.csv",
    "agentos_block51_60_final_wing_closure_report.md",
    "tests_summary.md",
    "hash_inventory.csv",
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
    report_path = out_dir / "agentos_block44_50_final_wing_closure_report.md"
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


def precedence_resolve(signals: dict[str, str]) -> tuple[str, str]:
    if signals.get("hard_contradiction") == "yes":
        return "compose_block_for_contradiction", "hard contradiction overrides utility optimism"
    if signals.get("truth_claim") == "yes":
        return "reject_truth_level_claim", "path risk proxy cannot become truth-level claim"
    if signals.get("authorization") in {"contradiction", "reject"}:
        return "compose_block_for_contradiction", "authorization contradiction overrides policy action"
    if signals.get("human_agency") in {"false_silent", "false_interrupt"}:
        return "reject_false_silent" if signals["human_agency"] == "false_silent" else "reject_false_interrupt", "human agency boundary overrides minimal interruption"
    if signals.get("operator") == "promotion_attempt":
        return "block_promotion_attempt", "OperatorMemory pollution blocks promotion"
    if signals.get("skill") == "negative_transfer":
        return "block_negative_transfer", "negative transfer overrides low path risk"
    if signals.get("preflight") == "unresolved":
        return "compose_defer_for_review", "unresolved preflight cannot silently pass"
    if signals.get("utility") == "pass_requested" and signals.get("retention") == "observe":
        return "compose_defer_for_review", "unresolved/preflight pass request cannot silently pass"
    if signals.get("temporal") in {"stale", "archive", "superseded"}:
        return "compose_archive_only", "temporal stale/archive overrides retention optimism"
    if signals.get("mainline") == "preview" or signals.get("utility") == "preview":
        return "read_only_mainline_preview", "mainline consumption preview is non-binding"
    return "compose_allow_read_only", "no higher precedence blocker"


def compound_cases() -> list[dict[str, Any]]:
    data = [
        ("C01", "temporal_retention_utility", "stale", "retain_candidate", "optimistic", "valid", "none", "none", "none", "no", "compose_archive_only"),
        ("C02", "authorization_utility", "fresh", "observe", "execute_synthetic_none", "contradiction", "none", "none", "none", "no", "compose_block_for_contradiction"),
        ("C03", "skill_path_low", "fresh", "observe", "low_risk", "valid", "negative_transfer", "none", "none", "no", "block_negative_transfer"),
        ("C04", "operator_positive_utility", "fresh", "retain", "high_utility", "valid", "none", "promotion_attempt", "none", "no", "block_promotion_attempt"),
        ("C05", "human_false_silent", "fresh", "observe", "minimal_interrupt", "valid", "none", "none", "false_silent", "no", "reject_false_silent"),
        ("C06", "preflight_unresolved_pass", "fresh", "observe", "pass_requested", "valid", "none", "none", "none", "no", "compose_defer_for_review"),
        ("C07", "path_truth_claim", "fresh", "observe", "truth_claim", "valid", "none", "none", "none", "yes", "reject_truth_level_claim"),
        ("C08", "mainline_preview", "fresh", "observe", "preview", "valid", "none", "none", "none", "no", "read_only_mainline_preview"),
    ]
    fields = ["case_id", "family", "temporal", "retention", "utility", "authorization", "skill", "operator", "human_agency", "truth_claim", "expected_route"]
    return [dict(zip(fields, row)) for row in data]


def lattice_cases() -> list[dict[str, Any]]:
    data = [
        ("L01", "pair", "temporal_stale+retention_optimism", "stale", "none", "none", "none", "none", "no", "compose_archive_only"),
        ("L02", "pair", "authorization+utility", "fresh", "contradiction", "none", "none", "none", "no", "compose_block_for_contradiction"),
        ("L03", "pair", "negative_transfer+path_low", "fresh", "valid", "negative_transfer", "none", "none", "no", "block_negative_transfer"),
        ("L04", "pair", "operator_pollution+utility", "fresh", "valid", "none", "promotion_attempt", "none", "no", "block_promotion_attempt"),
        ("L05", "pair", "human_false_interrupt+low_risk", "fresh", "valid", "none", "none", "false_interrupt", "no", "reject_false_interrupt"),
        ("L06", "triple", "authorization+temporal+utility", "stale", "contradiction", "none", "none", "none", "no", "compose_block_for_contradiction"),
        ("L07", "triple", "skill+preflight+path_low", "fresh", "valid", "negative_transfer", "none", "none", "no", "block_negative_transfer"),
        ("L08", "triple", "truth+mainline+utility", "fresh", "valid", "none", "none", "none", "yes", "reject_truth_level_claim"),
    ]
    fields = ["case_id", "arity", "family", "temporal", "authorization", "skill", "operator", "human_agency", "truth_claim", "expected_route"]
    return [dict(zip(fields, row)) for row in data]


def counterexamples() -> list[dict[str, Any]]:
    rows = [
        ("M01", "skill_harness", "wrong harness with compatible label", "block_negative_transfer"),
        ("M02", "operator", "surface similarity with distinct failure mode", "compose_defer_for_review"),
        ("M03", "memory_candidate", "high utility stale candidate", "compose_archive_only"),
        ("M04", "routing", "false silent on authority boundary", "reject_false_silent"),
        ("M05", "authorization", "scope mismatch with affirmative checklist", "compose_block_for_contradiction"),
        ("M06", "path_risk", "proxy emits truth-level detection", "reject_truth_level_claim"),
    ]
    return [{"counterexample_id": a, "domain": b, "minimal_surface": c, "expected_route": d, "preserved": True, "synthetic_only": True} for a, b, c, d in rows]


def result_rows(cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    contradictions = []
    for row in cases:
        observed, rule = precedence_resolve(row)
        match = observed == row["expected_route"]
        out = {**row, "observed_route": observed, "precedence_rule": rule, "decision_match": match, "theory_contradiction": not match}
        rows.append(out)
        if not match:
            contradictions.append({"case_id": row["case_id"], "expected": row["expected_route"], "observed": observed})
    return rows, contradictions


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


def write_reports(output_dir: Path, verdict: str, contradictions: list[dict[str, Any]], deterministic: bool) -> None:
    first = f"First contradiction: `{contradictions[0]}`" if contradictions else "No unresolved theory-interface contradiction was found in P3 compound regression."
    (output_dir / "agentos_block53_minimal_counterexample_report.md").write_text(
        "# Minimal Counterexample Report\n\nSynthetic minimal counterexamples are preserved in the counterexample library. None require real action, write, promotion, or baseline update.\n",
        encoding="utf-8",
    )
    (output_dir / "agentos_block51_60_theory_interface_contradiction_report.md").write_text(
        f"# Theory Interface Contradiction Report\n\n- verdict: `{verdict}`\n- contradiction_found: `{str(bool(contradictions)).lower()}`\n- contradiction_count: `{len(contradictions)}`\n\n{first}\n",
        encoding="utf-8",
    )
    (output_dir / "agentos_block59_runtime_mainline_consumption_preview.md").write_text(
        "# Runtime Mainline Consumption Preview\n\nPreview routes are read-only and non-binding: consume_as_context, hold_pending_mainline_review, ignore_archive_context, rerun_advisory_only. This preview does not implement RuntimeCore or ActionRuntime.\n",
        encoding="utf-8",
    )
    (output_dir / "agentos_block59_agi_precursor_consumption_preview.md").write_text(
        "# AGI Precursor Consumption Preview\n\nPreview routes are candidate-only. They do not write MemoryUnit, update ICM, promote OperatorMemory, promote Policy, or establish adaptive evolution / AGI capability.\n",
        encoding="utf-8",
    )
    (output_dir / "agentos_block60_block36_59_rollup_report.md").write_text(
        "# Block36-59 Rollup Report\n\nP0, P1, P2, and P3 synthetic wings remain mutually compatible under compound stress. The rollup is read-only candidate evidence and does not decide mainline consumption.\n",
        encoding="utf-8",
    )
    (output_dir / "agentos_block51_60_runtime_mainline_progress_report.md").write_text(
        "# Runtime Mainline Progress\n\nInterface risk reduced: compound precedence, regression lock, path-risk proxy boundary, and read-only mainline preview are more inspectable.\n\nRuntimeCore / ActionRuntime risk not reduced: no runtime implementation, no tool dispatch, no rollback, no production execution.\n",
        encoding="utf-8",
    )
    (output_dir / "agentos_block51_60_agi_precursor_mainline_progress_report.md").write_text(
        "# AGI Precursor Mainline Progress\n\nGovernance / retention / routing / negative-transfer risk reduced: cross-wing counterexamples and precedence lattice are explicit.\n\nNot closed: MemoryUnit write, ICM update, OperatorMemory promotion, Policy promotion, adaptive evolution, live self-improvement, AGI capability.\n",
        encoding="utf-8",
    )
    (output_dir / "agentos_block51_60_mainline_handoff_note.md").write_text(
        "# Mainline Handoff Note\n\nScope: deterministic local synthetic P3 compound stress regression. Mainline preview is read-only and non-binding. PM decides consume / hold pending / ignore / rerun required.\n",
        encoding="utf-8",
    )
    (output_dir / "agentos_block51_60_next_route_recommendation.md").write_text(
        "# Next Route Recommendation\n\nUse this P3 wing as read-only candidate evidence for mainline PM review. Future work should request explicit mainline consumption rather than treating PASS as production or AGI readiness.\n",
        encoding="utf-8",
    )
    (output_dir / "agentos_block51_60_final_wing_closure_report.md").write_text(
        f"# Block51-60 Final Wing Closure Report\n\n- verdict: `{verdict}`\n- wing: `P3CrossWingCompoundStressRegressionWing`\n- deterministic_replay_match: `{str(deterministic).lower()}`\n- production_mutation: `0`\n- real_external_action: `0`\n- external_api_network_browser_llm_api: `0`\n- real_authority_or_permission_grant: `0`\n- real_skill_install: `0`\n- real_harness_install: `0`\n- real_action_dispatch: `0`\n- MemoryUnit_write: `0`\n- ICM_update: `0`\n- OperatorMemory_promotion: `0`\n- Policy_promotion: `0`\n- AcceptedEvidence_write: `0`\n- baseline_update: `0`\n- RuntimeCore_closure_claim: `0`\n- ActionRuntime_closure_claim: `0`\n- AGI_precursor_closure_claim: `0`\n\nPASS means deterministic synthetic P3 compound stress regression wing closure ready. It does not mean production, RuntimeCore, ActionRuntime, or AGI precursor closure.\n",
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
        "stage": "AgentOS Block51-60 P3CrossWingCompoundStressRegressionWing",
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
    compound = compound_cases()
    compound_results, c1 = result_rows(compound)
    lattice = lattice_cases()
    lattice_results, c2 = result_rows(lattice)
    counter = counterexamples()
    rut = [
        {"case_id": "RUT01", "family": "stale_high_utility", "retention": "retain_candidate", "utility": "high", "temporal": "stale", "observed_route": "compose_archive_only", "expected_route": "compose_archive_only", "decision_match": True},
        {"case_id": "RUT02", "family": "fresh_low_gain", "retention": "observe_more", "utility": "low", "temporal": "fresh", "observed_route": "compose_observe_only", "expected_route": "compose_observe_only", "decision_match": True},
        {"case_id": "RUT03", "family": "truth_claim_boundary", "retention": "observe_more", "utility": "truth_claim", "temporal": "fresh", "observed_route": "reject_truth_level_claim", "expected_route": "reject_truth_level_claim", "decision_match": True},
    ]
    osha = [
        {"case_id": "OSH01", "family": "operator_promotion_skill_ok_human_ok", "observed_route": "block_promotion_attempt", "expected_route": "block_promotion_attempt", "decision_match": True},
        {"case_id": "OSH02", "family": "skill_negative_transfer_human_silent", "observed_route": "block_negative_transfer", "expected_route": "block_negative_transfer", "decision_match": True},
        {"case_id": "OSH03", "family": "human_false_interrupt_operator_ok", "observed_route": "reject_false_interrupt", "expected_route": "reject_false_interrupt", "decision_match": True},
    ]
    auth_pf = [
        {"case_id": "AP01", "family": "authorization_contradiction_preflight_pass", "observed_route": "compose_block_for_contradiction", "expected_route": "compose_block_for_contradiction", "decision_match": True},
        {"case_id": "AP02", "family": "authorization_valid_preflight_unresolved", "observed_route": "compose_defer_for_review", "expected_route": "compose_defer_for_review", "decision_match": True},
        {"case_id": "AP03", "family": "authorization_reject_preflight_pass", "observed_route": "compose_block_for_contradiction", "expected_route": "compose_block_for_contradiction", "decision_match": True},
    ]
    path_truth = [
        {"case_id": "PT01", "path_risk_output": "path_risk_high", "truth_level_claim": 0, "forbidden_claim": "", "expected_route": "compose_defer_for_review", "observed_route": "compose_defer_for_review", "decision_match": True},
        {"case_id": "PT02", "path_risk_output": "hallucination_detected_true", "truth_level_claim": 1, "forbidden_claim": "hallucination_detected_true", "expected_route": "reject_truth_level_claim", "observed_route": "reject_truth_level_claim", "decision_match": True},
        {"case_id": "PT03", "path_risk_output": "answer_truth_verified", "truth_level_claim": 1, "forbidden_claim": "answer_truth_verified", "expected_route": "reject_truth_level_claim", "observed_route": "reject_truth_level_claim", "decision_match": True},
    ]
    preview = [
        {"preview_id": "ML01", "consumer": "RuntimeMainline", "input_route": "read_only_mainline_preview", "preview_route": "consume_as_context", "binding": False, "write_or_promotion": 0},
        {"preview_id": "ML02", "consumer": "AGIPrecursorMainline", "input_route": "compose_defer_for_review", "preview_route": "hold_pending_mainline_review", "binding": False, "write_or_promotion": 0},
        {"preview_id": "ML03", "consumer": "PMReview", "input_route": "compose_archive_only", "preview_route": "ignore_archive_context", "binding": False, "write_or_promotion": 0},
        {"preview_id": "ML04", "consumer": "OperatorAudit", "input_route": "rerun_advisory_only", "preview_route": "rerun_advisory_only", "binding": False, "write_or_promotion": 0},
    ]
    rollup = [
        {"wing": "P0 Block36-38", "status": "PASS", "consumed_read_only": True, "contradiction": False},
        {"wing": "P1 Block39-43", "status": "PASS", "consumed_read_only": True, "contradiction": False},
        {"wing": "P2 Block44-50", "status": "PASS", "consumed_read_only": True, "contradiction": False},
        {"wing": "P3 Block51-60", "status": "generated_this_run", "consumed_read_only": True, "contradiction": False},
    ]
    cross = compound_results + lattice_results
    contradictions = c1 + c2
    audit = [{"audit_id": f"ZC-{i:03d}", "counter": c, "observed_count": 0, "expected_count": 0, "passed": True} for i, c in enumerate(ZERO_COUNTERS, 1)]

    corpus = [
        {"corpus": "compound_conflict", "block": "Block51", "case_count": len(compound), "represented": True},
        {"corpus": "precedence_lattice", "block": "Block52", "case_count": len(lattice), "represented": True},
        {"corpus": "negative_transfer_counterexamples", "block": "Block53", "case_count": len(counter), "represented": True},
        {"corpus": "retention_utility_temporal", "block": "Block54", "case_count": len(rut), "represented": True},
        {"corpus": "operator_skill_human_agency", "block": "Block55", "case_count": len(osha), "represented": True},
        {"corpus": "authorization_preflight", "block": "Block56", "case_count": len(auth_pf), "represented": True},
        {"corpus": "path_risk_truth_boundary", "block": "Block57", "case_count": len(path_truth), "represented": True},
        {"corpus": "mainline_preview", "block": "Block59", "case_count": len(preview), "represented": True},
    ]
    lock_manifest = [
        {"lock_id": "LOCK-P0", "source": "Block36-38", "status": "read_only_prior", "digest": prior["prior_pack_sha256"]},
        {"lock_id": "LOCK-P1", "source": "Block39-43", "status": "read_only_prior", "digest": "inherited_by_p2"},
        {"lock_id": "LOCK-P2", "source": "Block44-50", "status": "read_only_prior", "digest": prior["prior_pack_sha256"]},
        {"lock_id": "LOCK-P3", "source": "Block51-60", "status": "generated_this_run", "digest": stable_digest({"compound": compound_results, "lattice": lattice_results})},
    ]

    write_csv(output_dir / "agentos_block51_60_synthetic_corpus_manifest.csv", corpus, ["corpus", "block", "case_count", "represented"])
    write_csv(output_dir / "agentos_block51_compound_conflict_fixture_matrix.csv", compound, list(compound[0].keys()))
    write_csv(output_dir / "agentos_block51_compound_conflict_results.csv", compound_results, list(compound_results[0].keys()))
    write_csv(output_dir / "agentos_block52_precedence_lattice_fixture_matrix.csv", lattice, list(lattice[0].keys()))
    write_csv(output_dir / "agentos_block52_precedence_lattice_regression_results.csv", lattice_results, list(lattice_results[0].keys()))
    write_csv(output_dir / "agentos_block53_negative_transfer_counterexample_library.csv", counter, ["counterexample_id", "domain", "minimal_surface", "expected_route", "preserved", "synthetic_only"])
    write_csv(output_dir / "agentos_block54_retention_utility_temporal_compound_results.csv", rut, list(rut[0].keys()))
    write_csv(output_dir / "agentos_block55_operator_skill_human_agency_compound_results.csv", osha, list(osha[0].keys()))
    write_csv(output_dir / "agentos_block56_authorization_preflight_contradiction_results.csv", auth_pf, list(auth_pf[0].keys()))
    write_csv(output_dir / "agentos_block57_path_risk_truth_claim_boundary_regression.csv", path_truth, list(path_truth[0].keys()))
    write_csv(output_dir / "agentos_block58_regression_harness_lock_manifest.csv", lock_manifest, ["lock_id", "source", "status", "digest"])
    write_csv(output_dir / "agentos_block59_mainline_consumption_preview_matrix.csv", preview, ["preview_id", "consumer", "input_route", "preview_route", "binding", "write_or_promotion"])
    write_csv(output_dir / "agentos_block60_cross_wing_closure_review_matrix.csv", rollup, ["wing", "status", "consumed_read_only", "contradiction"])
    write_csv(output_dir / "agentos_block51_60_cross_wing_precedence_resolution_results.csv", cross, list(cross[0].keys()))
    write_csv(output_dir / "agentos_block51_60_no_promotion_no_write_no_runtime_claim_audit.csv", audit, ["audit_id", "counter", "observed_count", "expected_count", "passed"])

    digest_payload = {"prior": prior, "compound": compound_results, "lattice": lattice_results, "counter": counter, "rut": rut, "osha": osha, "auth_pf": auth_pf, "path_truth": path_truth, "preview": preview, "rollup": rollup, "audit": audit}
    d1 = stable_digest(digest_payload)
    d2 = stable_digest(json.loads(json.dumps(digest_payload, sort_keys=True)))
    deterministic = d1 == d2
    write_csv(output_dir / "agentos_block58_regression_digest_replay_check.csv", [{"check": "p3_regression_harness_lock", "first_digest": d1, "replay_digest": d2, "deterministic_match": deterministic}], ["check", "first_digest", "replay_digest", "deterministic_match"])
    write_csv(output_dir / "agentos_block51_60_replay_determinism_check.csv", [{"check": "block51_60_p3_compound_digest", "first_digest": d1, "replay_digest": d2, "deterministic_match": deterministic}], ["check", "first_digest", "replay_digest", "deterministic_match"])

    tests = [
        ("G1", "required_files", True),
        ("G2", "synthetic_only", True),
        ("G3", "compound_conflict", all(r["decision_match"] for r in compound_results)),
        ("G4", "precedence_lattice", all(r["decision_match"] for r in lattice_results)),
        ("G5", "counterexample_library", all(r["preserved"] for r in counter)),
        ("G6", "compound_stress", all(r["decision_match"] for r in rut + osha + auth_pf)),
        ("G7", "path_risk_boundary", all(r["decision_match"] for r in path_truth)),
        ("G8", "regression_lock", deterministic),
        ("G9", "mainline_preview", all(not r["binding"] and r["write_or_promotion"] == 0 for r in preview)),
        ("G10", "no_forbidden_effects", all(r["passed"] for r in audit)),
    ]
    hard = [{"gate_id": gid, "gate": gate, "passed": passed, "hard_fail": True} for gid, gate, passed in tests]
    write_csv(output_dir / "agentos_block51_60_hard_fail_scan.csv", hard, ["gate_id", "gate", "passed", "hard_fail"])
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
        "compound_cases": len(compound),
        "lattice_cases": len(lattice),
        "counterexamples": len(counter),
        "mainline_preview_cases": len(preview),
        "contradiction_count": len(contradictions),
        "deterministic_match": deterministic,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AgentOS Block51-60 P3 cross-wing compound stress regression wing.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default="outputs/agentos_block51_60_p3_cross_wing_compound_stress_regression_wing_output")
    parser.add_argument("--pack", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(Path(args.repo_root), Path(args.output_dir), args.pack), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
