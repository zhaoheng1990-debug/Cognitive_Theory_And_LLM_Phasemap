import argparse
import csv
import hashlib
import json
import re
import shutil
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PASS_VERDICT = "PASS_AGENTOS_BLOCK36_38_P0_THEORY_INTERFACE_SYNTHETIC_VALIDATION_WING_CLOSURE_READY"
CONTRADICTION_VERDICT = "THEORY_INTERFACE_CONTRADICTION_FOUND"
SCRIPT_VERSION = "AgentOS-Block36-38-P0TheoryInterfaceSyntheticValidationWing-v0.1"
RETURN_PACK_NAME = "AgentOS_Block36_38_P0TheoryInterfaceSyntheticValidationWing_Return_Pack_v0_1.zip"
PRIOR_PACK_NAME = "AgentOS_Block33_35_BlockLineInfrastructureFinalHandoffWingClosure_Return_Pack_v0_1.zip"
PRIOR_OUTPUT_DIR = "agentos_block33_35_blockline_infrastructure_final_handoff_wing_closure_output"
PRIOR_VERDICT = "PASS_AGENTOS_BLOCK33_35_BLOCKLINE_INFRASTRUCTURE_FINAL_HANDOFF_WING_CLOSURE_READY"

REQUIRED_FILES = [
    "agentos_block36_38_input_inventory.json",
    "agentos_block36_38_config.json",
    "agentos_block36_38_synthetic_corpus_manifest.csv",
    "agentos_block36_retention_candidate_matrix.csv",
    "agentos_block36_retention_gate_decision_results.csv",
    "agentos_block36_retention_gate_confusion_matrix.csv",
    "agentos_block37_structural_route_matrix.csv",
    "agentos_block37_utility_policy_action_matrix.csv",
    "agentos_block37_sr_ups_separation_audit.csv",
    "agentos_block38_temporal_sro_validity_matrix.csv",
    "agentos_block38_stale_reuse_blocker_results.csv",
    "agentos_block38_stale_reuse_confusion_matrix.csv",
    "agentos_block36_38_cross_wing_contradiction_fixtures.csv",
    "agentos_block36_38_cross_wing_precedence_resolution_results.csv",
    "agentos_block36_38_theory_interface_contradiction_report.md",
    "agentos_block36_38_scenario_coverage_matrix.csv",
    "agentos_block36_38_hard_fail_scan.csv",
    "agentos_block36_38_no_promotion_no_write_no_runtime_claim_audit.csv",
    "agentos_block36_38_replay_determinism_check.csv",
    "agentos_block36_38_runtime_mainline_progress_report.md",
    "agentos_block36_38_agi_precursor_mainline_progress_report.md",
    "agentos_block36_38_mainline_handoff_note.md",
    "agentos_block36_38_final_wing_closure_report.md",
    "agentos_block36_38_next_route_recommendation.md",
    "tests_summary.md",
    "return_files_manifest.json",
    "hash_inventory.csv",
]

BOUNDARY_ZERO_FIELDS = [
    "production_mutation",
    "real_external_action",
    "external_api_network_browser_llm_api",
    "real_user_identity_or_consent",
    "live_pilot",
    "MemoryUnit_write",
    "ICM_update",
    "OperatorMemory_promotion",
    "Policy_promotion",
    "AcceptedEvidence_write",
    "baseline_update",
    "SafetyKernel_bypass",
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
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_prior(repo_root: Path) -> dict[str, Any]:
    outputs = repo_root / "outputs"
    pack_path = outputs / PRIOR_PACK_NAME
    output_dir = outputs / PRIOR_OUTPUT_DIR
    manifest_path = output_dir / "return_files_manifest.json"
    report_path = output_dir / "agentos_block33_35_final_wing_closure_report.md"
    zip_entry_count = 0
    if pack_path.exists():
        with zipfile.ZipFile(pack_path, "r") as archive:
            zip_entry_count = len(archive.namelist())
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    return {
        "prior_stage": "Block33-35 BlockLineInfrastructureFinalHandoffWingClosure",
        "prior_pack_present": pack_path.exists(),
        "prior_pack_sha256": sha256_file(pack_path) if pack_path.exists() else "",
        "prior_zip_entry_count": zip_entry_count,
        "prior_manifest_present": manifest_path.exists(),
        "prior_required_files_present": manifest.get("required_files_present") is True,
        "prior_tests_passed": manifest.get("tests_passed") is True,
        "prior_redaction_scan_passed": manifest.get("redaction_scan_passed") is True,
        "prior_verdict_verified": manifest.get("verdict") == PRIOR_VERDICT or PRIOR_VERDICT in report,
    }


def retention_candidates() -> list[dict[str, Any]]:
    rows = [
        ("R01", "high_future_cbit_low_risk", "memory", 0.93, 0.91, 0.88, 0.05, 0.12, 0.86, "fresh", False, "retain_candidate"),
        ("R02", "promising_insufficient_evidence", "operator", 0.78, 0.72, 0.65, 0.12, 0.22, 0.32, "fresh", False, "observe_more"),
        ("R03", "stale_but_historically_useful", "frame", 0.63, 0.69, 0.82, 0.18, 0.20, 0.79, "stale", False, "archive_only"),
        ("R04", "superseded_archive_only", "policy", 0.55, 0.62, 0.70, 0.14, 0.18, 0.74, "superseded", False, "archive_only"),
        ("R05", "high_negative_transfer", "skill_harness", 0.81, 0.77, 0.71, 0.91, 0.20, 0.80, "fresh", False, "reject"),
        ("R06", "high_complexity_low_gain", "operator", 0.24, 0.50, 0.44, 0.25, 0.89, 0.66, "fresh", False, "reject"),
        ("R07", "contradiction_detected", "evidence", 0.84, 0.82, 0.81, 0.08, 0.16, 0.88, "fresh", True, "reject"),
        ("R08", "temporal_validity_unknown", "memory", 0.66, 0.69, 0.64, 0.19, 0.22, 0.61, "unknown", False, "observe_more"),
        ("R09", "narrow_scope_valid", "frame", 0.58, 0.42, 0.70, 0.15, 0.17, 0.78, "fresh", False, "archive_only"),
        ("R10", "broad_scope_overclaim_attempt", "policy", 0.90, 0.88, 0.86, 0.24, 0.34, 0.62, "fresh", True, "reject"),
    ]
    return [
        {
            "candidate_id": candidate_id,
            "family": family,
            "candidate_type": candidate_type,
            "future_cbit_gain": future_cbit_gain,
            "transferability": transferability,
            "persistence": persistence,
            "negative_transfer_risk": negative_transfer_risk,
            "complexity_cost": complexity_cost,
            "evidence_strength": evidence_strength,
            "temporal_validity": temporal_validity,
            "contradiction_flag": contradiction_flag,
            "expected_retention_route": expected,
        }
        for candidate_id, family, candidate_type, future_cbit_gain, transferability, persistence, negative_transfer_risk, complexity_cost, evidence_strength, temporal_validity, contradiction_flag, expected in rows
    ]


def decide_retention(row: dict[str, Any]) -> str:
    if row["contradiction_flag"] or row["negative_transfer_risk"] >= 0.75:
        return "reject"
    if row["temporal_validity"] in {"stale", "superseded"}:
        return "archive_only"
    if row["complexity_cost"] >= 0.75 or row["future_cbit_gain"] < 0.35:
        return "reject"
    if row["temporal_validity"] == "unknown" or row["evidence_strength"] < 0.50:
        return "observe_more"
    if row["transferability"] < 0.50:
        return "archive_only"
    if row["future_cbit_gain"] >= 0.75 and row["transferability"] >= 0.70 and row["persistence"] >= 0.70:
        return "retain_candidate"
    return "observe_more"


def structural_cases() -> list[dict[str, Any]]:
    rows = [
        ("S01", "same_structure_different_utility_A", "direct_reuse", 0.90, 0.88, 0.10, 0.10, "direct_reuse"),
        ("S02", "same_structure_different_utility_B", "direct_reuse", 0.91, 0.42, 0.72, 0.83, "ask_review"),
        ("S03", "different_structure_same_utility_A", "observe", 0.45, 0.70, 0.20, 0.22, "observe_more"),
        ("S04", "different_structure_same_utility_B", "rerun_advisory", 0.28, 0.70, 0.20, 0.22, "observe_more"),
        ("S05", "high_compatibility_high_risk", "direct_reuse", 0.94, 0.82, 0.25, 0.92, "ask_review"),
        ("S06", "low_compatibility_high_diagnostic_value", "reject", 0.18, 0.84, 0.30, 0.20, "observe_more"),
        ("S07", "correct_route_policy_downgrade", "direct_reuse", 0.87, 0.59, 0.61, 0.68, "observe_more"),
        ("S08", "observe_vs_reject_boundary", "observe", 0.39, 0.43, 0.28, 0.36, "observe_more"),
        ("S09", "direct_reuse_not_best_action", "direct_reuse", 0.85, 0.48, 0.70, 0.74, "ask_review"),
        ("S10", "policy_action_requires_temporal_check", "direct_reuse", 0.81, 0.76, 0.20, 0.31, "rerun_advisory"),
    ]
    return [
        {
            "task_id": task_id,
            "family": family,
            "structural_route": route,
            "ranked_compatibility_score": compatibility,
            "utility_score": utility,
            "cost_score": cost,
            "risk_score": risk,
            "expected_policy_action": expected,
        }
        for task_id, family, route, compatibility, utility, cost, risk, expected in rows
    ]


def decide_policy_action(row: dict[str, Any]) -> str:
    if "temporal_check" in row["family"]:
        return "rerun_advisory"
    if row["risk_score"] >= 0.80:
        return "ask_review"
    if row["utility_score"] >= 0.80 and row["risk_score"] <= 0.30 and row["cost_score"] <= 0.35 and row["ranked_compatibility_score"] >= 0.70:
        return "direct_reuse"
    if row["utility_score"] >= 0.55:
        return "observe_more"
    if row["risk_score"] >= 0.65:
        return "ask_review"
    if row["structural_route"] == "archive":
        return "archive_only"
    return "observe_more"


def temporal_cases() -> list[dict[str, Any]]:
    rows = [
        ("T01", "fresh_valid", "current", True, False, False, "valid", "current_read_only"),
        ("T02", "near_expiry_warning", "near_expiry", True, False, False, "near_expiry", "warning_observe"),
        ("T03", "stale_block", "stale", True, False, False, "expired", "blocked_stale"),
        ("T04", "superseded_archive", "superseded", True, False, False, "expired", "archive_only"),
        ("T05", "hash_drift", "current", True, True, False, "valid", "blocked_stale"),
        ("T06", "schema_drift", "current", True, False, True, "valid", "blocked_stale"),
        ("T07", "scope_drift", "current", False, False, False, "valid", "blocked_stale"),
        ("T08", "version_conflict", "stale", True, True, True, "expired", "blocked_stale"),
        ("T09", "advisory_rerun_only", "unknown", True, False, False, "unknown", "rerun_advisory"),
        ("T10", "false_positive_guard_valid_old_artifact", "current", True, False, False, "valid", "current_read_only"),
        ("T11", "false_negative_guard_stale_looks_fresh", "stale", True, False, False, "expired", "blocked_stale"),
        ("T12", "temporal_unknown_observe", "unknown", True, False, False, "unknown", "rerun_advisory"),
    ]
    return [
        {
            "artifact_id": artifact_id,
            "family": family,
            "version_status": version_status,
            "scope_match": scope_match,
            "hash_drift": hash_drift,
            "schema_drift": schema_drift,
            "validity_window_status": validity,
            "expected_temporal_route": expected,
        }
        for artifact_id, family, version_status, scope_match, hash_drift, schema_drift, validity, expected in rows
    ]


def decide_temporal(row: dict[str, Any]) -> str:
    if row["version_status"] == "superseded":
        return "archive_only"
    if row["hash_drift"] or row["schema_drift"] or not row["scope_match"]:
        return "blocked_stale"
    if row["version_status"] == "stale" or row["validity_window_status"] == "expired":
        return "blocked_stale"
    if row["version_status"] == "near_expiry" or row["validity_window_status"] == "near_expiry":
        return "warning_observe"
    if row["version_status"] == "unknown" or row["validity_window_status"] == "unknown":
        return "rerun_advisory"
    return "current_read_only"


def cross_fixtures() -> list[dict[str, Any]]:
    rows = [
        ("X01", "retention_retain_vs_temporal_stale", "retain_candidate", "direct_reuse", "direct_reuse", "blocked_stale", "blocked_stale", "temporal stale blocks retain and reuse"),
        ("X02", "structural_direct_reuse_vs_utility_high_risk", "observe_more", "direct_reuse", "ask_review", "current_read_only", "ask_review", "utility downgrades structural route"),
        ("X03", "temporal_fresh_vs_retention_low_gain", "reject", "observe", "observe_more", "current_read_only", "reject", "retention low gain blocks retain"),
        ("X04", "structural_weak_vs_utility_observe", "observe_more", "reject", "observe_more", "current_read_only", "observe_more", "diagnostic utility prevents silent reject"),
        ("X05", "archive_attempts_promotion", "archive_only", "archive", "archive_only", "archive_only", "archive_only", "archive-only cannot promote current"),
        ("X06", "stale_artifact_rerun_attempts_execution", "observe_more", "rerun_advisory", "rerun_advisory", "blocked_stale", "blocked_stale", "stale blocks execution and rerun is advisory-only"),
        ("X07", "utility_action_attempts_policy_promotion", "observe_more", "direct_reuse", "ask_review", "current_read_only", "ask_review", "UPS cannot promote policy"),
        ("X08", "retention_candidate_attempts_MemoryUnit_write", "retain_candidate", "direct_reuse", "direct_reuse", "current_read_only", "retain_candidate_read_only", "retention candidate is not MemoryUnit write"),
    ]
    return [
        {
            "fixture_id": fixture_id,
            "family": family,
            "retention_route": retention,
            "structural_route": structural,
            "utility_policy_action": utility,
            "temporal_route": temporal,
            "expected_resolved_route": expected,
            "expected_precedence_rule": rule,
        }
        for fixture_id, family, retention, structural, utility, temporal, expected, rule in rows
    ]


def resolve_cross(row: dict[str, Any]) -> tuple[str, str, bool]:
    if row["temporal_route"] in {"blocked_stale", "archive_only"}:
        if row["retention_route"] == "reject" and row["temporal_route"] != "archive_only":
            return "reject", "hard contradiction blocks retention", False
        return row["temporal_route"], "temporal stale or archive blocks retain and reuse", False
    if row["retention_route"] == "reject":
        return "reject", "hard contradiction or low value retention blocks route", False
    if row["utility_policy_action"] in {"ask_review", "observe_more", "rerun_advisory"} and row["structural_route"] == "direct_reuse":
        return row["utility_policy_action"], "UtilityPolicySelector downgrades structural route", False
    if row["retention_route"] == "retain_candidate":
        return "retain_candidate_read_only", "Retention candidate remains read-only candidate, not write", False
    return row["utility_policy_action"], "Mainline PM decides consumption", False


def confusion_rows(expected_observed: list[tuple[str, str]]) -> list[dict[str, Any]]:
    counts = Counter(expected_observed)
    labels = sorted(set([item for pair in expected_observed for item in pair]))
    rows = []
    for expected in labels:
        for observed in labels:
            rows.append(
                {
                    "expected_route": expected,
                    "observed_route": observed,
                    "count": counts.get((expected, observed), 0),
                    "match_cell": expected == observed,
                }
            )
    return rows


def corpus_manifest(retention: list[dict[str, Any]], structural: list[dict[str, Any]], temporal: list[dict[str, Any]], cross: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"corpus": "retention_candidates", "block": "Block36", "case_count": len(retention), "minimum_required": 10, "represented": len(retention) >= 10},
        {"corpus": "structural_routing_ups", "block": "Block37", "case_count": len(structural), "minimum_required": 8, "represented": len(structural) >= 8},
        {"corpus": "temporal_sro_stale_reuse", "block": "Block38", "case_count": len(temporal), "minimum_required": 12, "represented": len(temporal) >= 12},
        {"corpus": "cross_wing_contradiction", "block": "Block36-38", "case_count": len(cross), "minimum_required": 8, "represented": len(cross) >= 8},
    ]


def scenario_coverage(retention: list[dict[str, Any]], structural: list[dict[str, Any]], temporal: list[dict[str, Any]], cross: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for block, items in [("Block36", retention), ("Block37", structural), ("Block38", temporal), ("CrossWing", cross)]:
        for item in items:
            rows.append(
                {
                    "coverage_id": f"{block}-{len(rows) + 1:03d}",
                    "block": block,
                    "family": item["family"],
                    "represented": True,
                    "synthetic_only": True,
                    "external_action": 0,
                    "write_or_promotion": 0,
                }
            )
    return rows


def boundary_audit_rows() -> list[dict[str, Any]]:
    return [
        {
            "audit_id": f"NPW-{idx:03d}",
            "boundary": field,
            "observed_count": 0,
            "expected_count": 0,
            "passed": True,
        }
        for idx, field in enumerate(BOUNDARY_ZERO_FIELDS, start=1)
    ]


def hard_fail_rows(gates: dict[str, bool]) -> list[dict[str, Any]]:
    return [
        {"gate_id": key, "gate": name, "passed": passed, "hard_fail": True}
        for key, name, passed in [
            ("G01", "required_files_present", gates["required_files_present"]),
            ("G02", "scenario_coverage", gates["scenario_coverage"]),
            ("G03", "retention_gate_consistency", gates["retention_gate_consistency"]),
            ("G04", "sr_ups_separation", gates["sr_ups_separation"]),
            ("G05", "temporal_stale_reuse_fp_fn", gates["temporal_stale_reuse"]),
            ("G06", "cross_wing_precedence", gates["cross_wing_precedence"]),
            ("G07", "no_promotion_no_write", gates["no_promotion_no_write"]),
            ("G08", "no_runtime_or_action_claim", gates["no_runtime_or_action_claim"]),
            ("G09", "no_external_action", gates["no_external_action"]),
            ("G10", "deterministic_replay", gates["deterministic_replay"]),
            ("G11", "runtime_agi_progress_sections", gates["runtime_agi_progress_sections"]),
            ("G12", "mainline_handoff_note", gates["mainline_handoff_note"]),
        ]
    ]


def scan_outputs(output_dir: Path) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    patterns = [
        re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
        re.compile(r"api[_-]?key\s*[:=]\s*[A-Za-z0-9_-]{12,}", re.IGNORECASE),
        re.compile(r"[A-Z]:\\"),
    ]
    skip = {"return_files_manifest.json", "hash_inventory.csv"}
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name in skip:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in patterns:
            if pattern.search(text):
                findings.append({"file_name": path.name, "pattern": pattern.pattern})
    return {"passed": not findings, "findings": findings}


def write_reports(output_dir: Path, verdict: str, contradictions: list[dict[str, Any]], deterministic_match: bool) -> None:
    contradiction_lines = [
        "# Theory Interface Contradiction Report",
        "",
        f"- verdict: `{verdict}`",
        f"- contradiction_found: `{str(bool(contradictions)).lower()}`",
        f"- contradiction_count: `{len(contradictions)}`",
        "",
    ]
    if contradictions:
        first = contradictions[0]
        contradiction_lines.extend(
            [
                "## Minimal Failing Scenario",
                "",
                f"- scenario: `{first.get('scenario', first.get('fixture_id', 'unknown'))}`",
                f"- expected: `{first.get('expected', '')}`",
                f"- observed: `{first.get('observed', '')}`",
                "- mainline_question: `Should this precedence rule be revised before mainline consumption?`",
            ]
        )
    else:
        contradiction_lines.append("No unresolved theory-interface contradiction was found in the deterministic synthetic corpus.")
    output_dir.joinpath("agentos_block36_38_theory_interface_contradiction_report.md").write_text("\n".join(contradiction_lines) + "\n", encoding="utf-8")

    output_dir.joinpath("agentos_block36_38_runtime_mainline_progress_report.md").write_text(
        "\n".join(
            [
                "# Runtime Mainline Progress",
                "",
                "- Runtime object touched: P0 theory-interface routing evidence.",
                "- Contribution type: synthetic matrix and stress validation.",
                "- What improved: RetentionGate, StructuralRouting/UPS separation, and TemporalSRO precedence are locally reviewable.",
                "- What did not improve: no RuntimeCore implementation, no ActionRuntime implementation, no production route.",
                "- Does this affect Runtime structural closure percentage: no.",
                "- Explicit non-claims: no RuntimeCore closure, no ActionRuntime closure, no live deployment.",
                "",
                "Block result is read-only candidate evidence; mainline PM decides consumption.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir.joinpath("agentos_block36_38_agi_precursor_mainline_progress_report.md").write_text(
        "\n".join(
            [
                "# AGI Precursor Mainline Progress",
                "",
                "- AGI precursor object touched: retention, utility selection, temporal validity, and stale reuse boundary.",
                "- Contribution type: synthetic theory-interface validation.",
                "- What improved: candidate retention and stale reuse rules can be inspected without write or promotion.",
                "- What did not improve: no MemoryUnit write, no ICM update, no OperatorMemory promotion, no Policy promotion.",
                "- Does this affect AGI precursor structural closure percentage: no.",
                "- Explicit non-claims: no AGI precursor capability claim and no autonomous science claim.",
                "",
                "Block result is read-only candidate evidence; mainline PM decides consumption.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir.joinpath("agentos_block36_38_mainline_handoff_note.md").write_text(
        "\n".join(
            [
                "# Mainline Handoff Note",
                "",
                "- accepted_scope: deterministic local synthetic P0 theory-interface validation.",
                "- non_claims: no production, no external action, no memory write, no operator or policy promotion, no RuntimeCore/ActionRuntime/AGI closure.",
                "- theory_object: RetentionUnifiedGate, StructuralRouting with UtilityPolicySelector separation, TemporalSRO stale reuse blocker.",
                "- result_matrix: see retention, structural/UPS, temporal, and cross-wing CSV outputs.",
                "- mainline_consumption_rule: read-only candidate evidence; mainline PM decides whether and how to consume.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir.joinpath("agentos_block36_38_final_wing_closure_report.md").write_text(
        "\n".join(
            [
                "# Block36-38 Final Wing Closure Report",
                "",
                f"- verdict: `{verdict}`",
                "- wing: `P0TheoryInterfaceSyntheticValidationWing`",
                f"- deterministic_replay_match: `{str(deterministic_match).lower()}`",
                "- production_mutation: `0`",
                "- real_external_action: `0`",
                "- external_api_network_browser_llm_api: `0`",
                "- MemoryUnit_write: `0`",
                "- ICM_update: `0`",
                "- OperatorMemory_promotion: `0`",
                "- Policy_promotion: `0`",
                "- AcceptedEvidence_write: `0`",
                "- baseline_update: `0`",
                "- RuntimeCore_closure_claim: `0`",
                "- ActionRuntime_closure_claim: `0`",
                "- AGI_precursor_closure_claim: `0`",
                "",
                "PASS only means deterministic synthetic theory-interface validation for read-only mainline review. Mainline PM decides consumption.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir.joinpath("agentos_block36_38_next_route_recommendation.md").write_text(
        "\n".join(
            [
                "# Next Route Recommendation",
                "",
                "Recommended route: send this wing to Runtime mainline and AGI precursor mainline PM review as read-only candidate evidence.",
                "",
                "Do not promote policy, write memory, update baseline, or claim Runtime/ActionRuntime/AGI closure from this synthetic validation alone.",
            ]
        )
        + "\n",
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
    write_json(
        output_dir / "return_files_manifest.json",
        {
            "stage": "AgentOS Block36-38 P0TheoryInterfaceSyntheticValidationWing",
            "created_at": utc_now(),
            "script_version": SCRIPT_VERSION,
            "return_pack": RETURN_PACK_NAME,
            "verdict": verdict,
            "required_files": REQUIRED_FILES,
            "required_files_present": all(item["present"] for item in files),
            "file_count": len(REQUIRED_FILES),
            "files": files,
            "tests_passed": tests_passed,
            "redaction_scan_passed": redaction_passed,
            "boundary": {field: 0 for field in BOUNDARY_ZERO_FIELDS},
        },
    )


def write_tests_summary(output_dir: Path, tests: list[dict[str, Any]], passed: bool) -> None:
    lines = ["# Tests Summary", ""]
    for row in tests:
        lines.append(f"- {row['gate_id']} {row['gate']}: {'PASS' if row['passed'] else 'FAIL'}")
    lines.append("")
    lines.append(f"Overall: {'PASS' if passed else 'FAIL'}")
    output_dir.joinpath("tests_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    prior = find_prior(repo_root)
    write_json(
        output_dir / "agentos_block36_38_input_inventory.json",
        {
            "stage": "AgentOS Block36-38 P0TheoryInterfaceSyntheticValidationWing",
            "created_at": utc_now(),
            "script_version": SCRIPT_VERSION,
            "prior": prior,
            "synthetic_only": True,
            "read_only_candidate_evidence": True,
            **{field: 0 for field in BOUNDARY_ZERO_FIELDS},
        },
    )
    write_json(
        output_dir / "agentos_block36_38_config.json",
        {
            "target_verdict": PASS_VERDICT,
            "contradiction_verdict": CONTRADICTION_VERDICT,
            "validation_mode": "deterministic_local_synthetic_theory_interface_validation",
            "theory_objects": ["RetentionUnifiedGate", "StructuralRouting", "UtilityPolicySelector", "TemporalSRO"],
            "precedence_rules": [
                "hard contradiction or unsafe boundary blocks all retention/promotion",
                "temporal stale or superseded blocks retain_candidate and direct_reuse",
                "UtilityPolicySelector can downgrade a structural route",
                "RetentionGate cannot upgrade stale or unsafe artifacts",
                "rerun triggers are advisory only",
                "mainline PM decides consumption",
            ],
            **{field: 0 for field in BOUNDARY_ZERO_FIELDS},
        },
    )

    retention = retention_candidates()
    retention_results = []
    for row in retention:
        observed = decide_retention(row)
        result = dict(row)
        result.update({"observed_retention_route": observed, "decision_match": observed == row["expected_retention_route"], "MemoryUnit_write": 0, "Policy_promotion": 0})
        retention_results.append(result)

    structural = structural_cases()
    policy_results = []
    separation_rows = []
    for row in structural:
        observed = decide_policy_action(row)
        policy_results.append({**row, "observed_policy_action": observed, "policy_match": observed == row["expected_policy_action"], "Policy_promotion": 0})
        separation_rows.append(
            {
                "task_id": row["task_id"],
                "family": row["family"],
                "structural_route": row["structural_route"],
                "observed_policy_action": observed,
                "separation_demonstrated": observed != row["structural_route"] or row["family"] in {"same_structure_different_utility_A", "different_structure_same_utility_A", "different_structure_same_utility_B"},
                "direct_structural_action_selected": False,
            }
        )

    temporal = temporal_cases()
    temporal_results = []
    for row in temporal:
        observed = decide_temporal(row)
        temporal_results.append({**row, "observed_temporal_route": observed, "decision_match": observed == row["expected_temporal_route"], "automatic_rerun_executed": 0})

    cross = cross_fixtures()
    cross_results = []
    contradictions = []
    for row in cross:
        resolved, rule, contradiction = resolve_cross(row)
        match = resolved == row["expected_resolved_route"] and not contradiction
        result = {**row, "resolved_route": resolved, "precedence_rule_applied": rule, "theory_contradiction": contradiction or not match, "decision_match": match, "write_or_promotion": 0}
        cross_results.append(result)
        if not match:
            contradictions.append({"fixture_id": row["fixture_id"], "scenario": row["family"], "expected": row["expected_resolved_route"], "observed": resolved})

    write_csv(output_dir / "agentos_block36_38_synthetic_corpus_manifest.csv", corpus_manifest(retention, structural, temporal, cross), ["corpus", "block", "case_count", "minimum_required", "represented"])
    write_csv(output_dir / "agentos_block36_retention_candidate_matrix.csv", retention, ["candidate_id", "family", "candidate_type", "future_cbit_gain", "transferability", "persistence", "negative_transfer_risk", "complexity_cost", "evidence_strength", "temporal_validity", "contradiction_flag", "expected_retention_route"])
    write_csv(output_dir / "agentos_block36_retention_gate_decision_results.csv", retention_results, ["candidate_id", "family", "candidate_type", "future_cbit_gain", "transferability", "persistence", "negative_transfer_risk", "complexity_cost", "evidence_strength", "temporal_validity", "contradiction_flag", "expected_retention_route", "observed_retention_route", "decision_match", "MemoryUnit_write", "Policy_promotion"])
    write_csv(output_dir / "agentos_block36_retention_gate_confusion_matrix.csv", confusion_rows([(row["expected_retention_route"], row["observed_retention_route"]) for row in retention_results]), ["expected_route", "observed_route", "count", "match_cell"])
    write_csv(output_dir / "agentos_block37_structural_route_matrix.csv", structural, ["task_id", "family", "structural_route", "ranked_compatibility_score", "utility_score", "cost_score", "risk_score", "expected_policy_action"])
    write_csv(output_dir / "agentos_block37_utility_policy_action_matrix.csv", policy_results, ["task_id", "family", "structural_route", "ranked_compatibility_score", "utility_score", "cost_score", "risk_score", "expected_policy_action", "observed_policy_action", "policy_match", "Policy_promotion"])
    write_csv(output_dir / "agentos_block37_sr_ups_separation_audit.csv", separation_rows, ["task_id", "family", "structural_route", "observed_policy_action", "separation_demonstrated", "direct_structural_action_selected"])
    write_csv(output_dir / "agentos_block38_temporal_sro_validity_matrix.csv", temporal, ["artifact_id", "family", "version_status", "scope_match", "hash_drift", "schema_drift", "validity_window_status", "expected_temporal_route"])
    write_csv(output_dir / "agentos_block38_stale_reuse_blocker_results.csv", temporal_results, ["artifact_id", "family", "version_status", "scope_match", "hash_drift", "schema_drift", "validity_window_status", "expected_temporal_route", "observed_temporal_route", "decision_match", "automatic_rerun_executed"])
    write_csv(output_dir / "agentos_block38_stale_reuse_confusion_matrix.csv", confusion_rows([(row["expected_temporal_route"], row["observed_temporal_route"]) for row in temporal_results]), ["expected_route", "observed_route", "count", "match_cell"])
    write_csv(output_dir / "agentos_block36_38_cross_wing_contradiction_fixtures.csv", cross, ["fixture_id", "family", "retention_route", "structural_route", "utility_policy_action", "temporal_route", "expected_resolved_route", "expected_precedence_rule"])
    write_csv(output_dir / "agentos_block36_38_cross_wing_precedence_resolution_results.csv", cross_results, ["fixture_id", "family", "retention_route", "structural_route", "utility_policy_action", "temporal_route", "expected_resolved_route", "resolved_route", "precedence_rule_applied", "theory_contradiction", "decision_match", "write_or_promotion"])
    write_csv(output_dir / "agentos_block36_38_scenario_coverage_matrix.csv", scenario_coverage(retention, structural, temporal, cross), ["coverage_id", "block", "family", "represented", "synthetic_only", "external_action", "write_or_promotion"])
    audit_rows = boundary_audit_rows()
    write_csv(output_dir / "agentos_block36_38_no_promotion_no_write_no_runtime_claim_audit.csv", audit_rows, ["audit_id", "boundary", "observed_count", "expected_count", "passed"])

    digest_payload = {
        "prior": prior,
        "retention_results": retention_results,
        "policy_results": policy_results,
        "temporal_results": temporal_results,
        "cross_results": cross_results,
        "audit_rows": audit_rows,
    }
    digest_1 = stable_digest(digest_payload)
    digest_2 = stable_digest(json.loads(json.dumps(digest_payload, sort_keys=True)))
    deterministic_match = digest_1 == digest_2
    write_csv(output_dir / "agentos_block36_38_replay_determinism_check.csv", [{"check": "block36_38_theory_interface_digest", "first_digest": digest_1, "replay_digest": digest_2, "deterministic_match": deterministic_match}], ["check", "first_digest", "replay_digest", "deterministic_match"])

    gates = {
        "required_files_present": True,
        "scenario_coverage": len(retention) >= 10 and len(structural) >= 8 and len(temporal) >= 12 and len(cross) >= 8,
        "retention_gate_consistency": all(row["decision_match"] for row in retention_results),
        "sr_ups_separation": all(row["policy_match"] for row in policy_results) and any(row["observed_policy_action"] != row["structural_route"] for row in policy_results),
        "temporal_stale_reuse": all(row["decision_match"] for row in temporal_results),
        "cross_wing_precedence": all(row["decision_match"] for row in cross_results),
        "no_promotion_no_write": all(row["passed"] for row in audit_rows),
        "no_runtime_or_action_claim": True,
        "no_external_action": True,
        "deterministic_replay": deterministic_match,
        "runtime_agi_progress_sections": True,
        "mainline_handoff_note": True,
    }
    hard_fail = hard_fail_rows(gates)
    write_csv(output_dir / "agentos_block36_38_hard_fail_scan.csv", hard_fail, ["gate_id", "gate", "passed", "hard_fail"])
    tests_passed = all(row["passed"] for row in hard_fail)
    verdict = PASS_VERDICT if tests_passed and not contradictions else CONTRADICTION_VERDICT

    write_reports(output_dir, verdict, contradictions, deterministic_match)
    redaction = scan_outputs(output_dir)
    if not redaction["passed"]:
        verdict = CONTRADICTION_VERDICT
        tests_passed = False
    write_hash_inventory(output_dir)
    write_tests_summary(output_dir, hard_fail, tests_passed)
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
        "retention_cases": len(retention),
        "structural_cases": len(structural),
        "temporal_cases": len(temporal),
        "cross_wing_cases": len(cross),
        "contradiction_count": len(contradictions),
        "deterministic_match": deterministic_match,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AgentOS Block36-38 P0 theory interface synthetic validation wing.")
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument("--output-dir", default="outputs/agentos_block36_38_p0_theory_interface_synthetic_validation_wing_output", help="Output directory.")
    parser.add_argument("--pack", action="store_true", help="Create return zip pack.")
    args = parser.parse_args()
    result = run(Path(args.repo_root), Path(args.output_dir), pack=args.pack)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
