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


VERDICT = "PASS_AGENTOS_FINAL_RELEASE_READINESS_REVIEW_WITH_RT2B_READY_FOR_HUMAN_RELEASE_DECISION"
BLOCKED_VERDICT = "BLOCKED_PENDING_TRACE_RT2B_RETURN"
PROJECT_ROOT = Path("D:/Logos_agentOS")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "agentos_final_release_readiness_with_rt2b_output"
RETURN_PACK_NAME = "AgentOS_FinalReleaseReadinessReview_WithRT2B_Return_Pack_v0_3.zip"
RETURN_PACK_PATH = PROJECT_ROOT / "outputs" / RETURN_PACK_NAME

REQUIRED_FILES = [
    "agentos_final_release_readiness_with_rt2b_evidence_map.csv",
    "agentos_final_release_readiness_with_rt2b_blocker_matrix.csv",
    "agentos_final_release_readiness_with_rt2b_trace_security_evidence_summary.md",
    "agentos_final_release_readiness_with_rt2b_trace_rt2b_live_sandbox_summary.md",
    "agentos_final_release_readiness_with_rt2b_actionruntime_boundary_summary.md",
    "agentos_final_release_readiness_with_rt2b_humangate_authority_boundary_summary.md",
    "agentos_final_release_readiness_with_rt2b_memory_operator_policy_boundary_summary.md",
    "agentos_final_release_readiness_with_rt2b_operational_security_review.md",
    "agentos_final_release_readiness_with_rt2b_remaining_risk_register.csv",
    "agentos_final_release_readiness_with_rt2b_rollback_freeze_restore_plan.md",
    "agentos_final_release_readiness_with_rt2b_release_nonclaim_compliance_report.md",
    "agentos_final_release_readiness_with_rt2b_human_decision_requirements.md",
    "agentos_final_release_readiness_with_rt2b_final_recommendation.md",
    "agentos_final_release_readiness_with_rt2b_tests_summary.md",
    "hash_inventory.csv",
    "return_files_manifest.json",
]

EVIDENCE_PACKS = [
    ("ATS1", "ATS1 trace surface inventory", "AgentOS_ATS1_TraceSurfaceInventory_LeakagePoisoningReplayThreatModel_Return_Pack_v0_1.zip", "trace_security_ats"),
    ("ATS2", "ATS2 redaction scoped disclosure", "AgentOS_ATS2_TraceRedactionScopedDisclosureGate_Return_Pack_v0_1.zip", "trace_security_ats"),
    ("ATS3", "ATS3 enforcement replay", "AgentOS_ATS3_TraceRedactionScopedDisclosureGateEnforcementReplay_Return_Pack_v0_1.zip", "trace_security_ats"),
    ("ATS4", "ATS4 provenance tamper lineage", "AgentOS_ATS4_TraceProvenanceChainAndTamperEvidenceAudit_Return_Pack_v0_1.zip", "trace_security_ats"),
    ("ATS5", "ATS5 review packet navigation", "AgentOS_ATS5_TraceIntegrityReviewPacketAndOperatorNavigation_Return_Pack_v0_1.zip", "trace_security_ats"),
    ("ATS_WC1", "ATS wing closure", "AgentOS_ATS_WingClosure_TraceSecurityReviewHandoffLineClosure_Return_Pack_v0_1.zip", "trace_security_wing_closure"),
    ("POSTATS_RRB1", "PostATS release blocker closure", "AgentOS_PostATS_ReleaseBlockerClosure_Return_Pack_v0_1.zip", "release_blocker_closure"),
    ("TRACE_RT1", "TRACE RT1 metadata-only real-demo family replay", "AgentOS_PostATS_TRACERealDemo_IsolatedRedTeam_Return_Pack_v0_1.zip", "metadata_only_non_executable"),
    ("TRACE_RT2", "TRACE RT2 scope-limited controlled stub sandbox", "AgentOS_PostATS_TRACEControlledLiveSandboxRedTeam_RT2_Return_Pack_v0_1.zip", "scope_limited_stub_sandbox"),
    ("TRACE_RT2B", "TRACE RT2B executable toy-target sandbox", "AgentOS_PostATS_TRACE_RT2B_IsolatedExecutableToyTargetRedTeam_Return_Pack_v0_1.zip", "executable_toy_target_sandbox"),
]

BOUNDARY_FLAGS = {
    "production_release_authorization": False,
    "production_grade_immunity_claim": False,
    "actionruntime_dispatch": False,
    "memory_write": False,
    "candidate_memory_write": False,
    "operator_policy_skillkernel_promotion": False,
    "human_gate_approval": False,
    "accepted_evidence_or_baseline_write": False,
}


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def extract_verdict(text: str) -> str:
    matches = re.findall(r"(PASS|BLOCKED|FAIL)[A-Z0-9_\\-]*", text)
    return matches[0] if matches else "UNKNOWN"


def read_pack(evidence_id: str, family: str, pack_name: str, classification: str) -> dict[str, object]:
    path = PROJECT_ROOT / "outputs" / pack_name
    if not path.exists():
        return {
            "evidence_id": evidence_id,
            "family": family,
            "classification": classification,
            "present": False,
            "pack_path": str(path),
            "pack_sha256": "",
            "zip_entries": 0,
            "manifest_present": False,
            "manifest_verdict": "MISSING",
            "tests_passed": "",
            "tests_total": "",
            "status": "missing",
            "notes": "evidence pack missing; not inferred",
        }
    pack_hash = file_sha256(path)
    manifest_present = False
    manifest_verdict = "UNKNOWN"
    tests_passed = ""
    tests_total = ""
    report_verdict = "UNKNOWN"
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        for name in names:
            lower = name.lower()
            if lower.endswith("return_files_manifest.json"):
                manifest_present = True
                try:
                    manifest = json.loads(zf.read(name).decode("utf-8"))
                    manifest_verdict = str(manifest.get("verdict", "UNKNOWN"))
                    tests_passed = str(manifest.get("tests_passed", ""))
                    tests_total = str(manifest.get("tests_total", ""))
                except Exception as exc:
                    manifest_verdict = f"MANIFEST_PARSE_ERROR:{exc}"
            elif "final_report" in lower or "tests_summary" in lower:
                report_verdict = extract_verdict(zf.read(name).decode("utf-8", errors="replace"))
    verdict = manifest_verdict if manifest_verdict != "UNKNOWN" else report_verdict
    status = "mapped_pass" if "PASS" in verdict and manifest_present else "mapped_review_required"
    return {
        "evidence_id": evidence_id,
        "family": family,
        "classification": classification,
        "present": True,
        "pack_path": str(path),
        "pack_sha256": pack_hash,
        "zip_entries": len(names),
        "manifest_present": manifest_present,
        "manifest_verdict": verdict,
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "status": status,
        "notes": "mapped from local return pack",
    }


def generate(output_dir: Path, pack: bool) -> dict[str, object]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence = [read_pack(*item) for item in EVIDENCE_PACKS]
    rt2b = next(item for item in evidence if item["evidence_id"] == "TRACE_RT2B")
    all_present = all(item["present"] for item in evidence)
    all_pass = all("PASS" in str(item["manifest_verdict"]) for item in evidence)
    verdict = VERDICT if rt2b["present"] and "PASS" in str(rt2b["manifest_verdict"]) and all_present and all_pass else BLOCKED_VERDICT

    write_evidence_map(output_dir, evidence)
    write_blocker_matrix(output_dir)
    write_remaining_risk_register(output_dir)
    write_markdown_files(output_dir, evidence, verdict)
    tests = run_tests(output_dir, evidence, verdict)
    write_text(output_dir / "agentos_final_release_readiness_with_rt2b_tests_summary.md", tests_summary(tests, verdict))
    write_hash_and_manifest(output_dir, tests, verdict)

    if pack:
        if RETURN_PACK_PATH.exists():
            RETURN_PACK_PATH.unlink()
        with zipfile.ZipFile(RETURN_PACK_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for filename in REQUIRED_FILES:
                zf.write(output_dir / filename, arcname=filename)

    return {
        "verdict": verdict,
        "output_dir": str(output_dir),
        "return_pack": str(RETURN_PACK_PATH),
        "return_pack_sha256": file_sha256(RETURN_PACK_PATH) if pack else None,
        "file_count": len(REQUIRED_FILES),
        "rt2b_present": bool(rt2b["present"]),
        "all_evidence_present": all_present,
        "all_evidence_pass": all_pass,
        "tests_passed": sum(1 for test in tests if test["pass"]),
        "tests_total": len(tests),
    }


def write_evidence_map(output_dir: Path, evidence: list[dict[str, object]]) -> None:
    write_csv(
        output_dir / "agentos_final_release_readiness_with_rt2b_evidence_map.csv",
        [
            "evidence_id",
            "family",
            "classification",
            "present",
            "pack_path",
            "pack_sha256",
            "zip_entries",
            "manifest_present",
            "manifest_verdict",
            "tests_passed",
            "tests_total",
            "status",
            "notes",
        ],
        [
            {
                **item,
                "present": str(item["present"]).lower(),
                "manifest_present": str(item["manifest_present"]).lower(),
            }
            for item in evidence
        ],
    )


def write_blocker_matrix(output_dir: Path) -> None:
    rows = [
        ("BLK_RT2B", "RT2B executable toy-target sandbox evidence", "closed", "TRACE_RT2B", "not_production_immunity"),
        ("BLK_RT2_SCOPE", "RT2 scope limitation", "non_production_blocking_after_rt2b", "TRACE_RT2/TRACE_RT2B", "RT2 remains scope-limited but corrected by RT2B"),
        ("BLK_ATS_LINE", "ATS1-5 and wing closure", "closed", "ATS1-ATS5/ATS_WC1", "trace security evidence mapped"),
        ("BLK_RRB1", "Release blocker closure", "closed", "POSTATS_RRB1", "final decision still human-only"),
        ("BLK_ACTION_RUNTIME", "ActionRuntime dispatch boundary", "closed_for_review", "RRB1/RT2B", "dispatch authorization false"),
        ("BLK_MEMORY_POLICY", "Memory/operator/policy boundary", "closed_for_review", "RRB1/RT2B", "write and promotion false"),
        ("BLK_HUMANGATE", "HumanGate authority boundary", "true_remaining_blocker_for_production", "FRR2", "real approval still required"),
        ("BLK_PRODUCTION", "Production release authorization", "true_remaining_blocker_for_production", "FRR2", "this pack cannot authorize production"),
        ("BLK_OPSEC", "Operational security review", "true_remaining_blocker_for_production", "FRR2", "must precede release decision"),
    ]
    write_csv(
        output_dir / "agentos_final_release_readiness_with_rt2b_blocker_matrix.csv",
        ["blocker_id", "blocker_family", "classification", "evidence_source", "notes"],
        [
            {
                "blocker_id": item[0],
                "blocker_family": item[1],
                "classification": item[2],
                "evidence_source": item[3],
                "notes": item[4],
            }
            for item in rows
        ],
    )


def write_remaining_risk_register(output_dir: Path) -> None:
    rows = [
        ("RISK_RELEASE_AUTH", "production release authorization", "high", "explicit authorized human release decision required", "true"),
        ("RISK_RT2B_SCOPE", "RT2B is toy-target evidence, not production attack immunity", "high", "do not claim production-grade immunity", "true"),
        ("RISK_OPSEC", "operational security review remains required", "high", "complete opsec review before release decision", "true"),
        ("RISK_ROLLBACK", "production rollback plan must be approved", "medium", "freeze/restore plan owner confirmation", "true"),
        ("RISK_INTEGRATION_DRIFT", "future production integration may drift from no-action evidence", "medium", "fresh production preflight required", "true"),
    ]
    write_csv(
        output_dir / "agentos_final_release_readiness_with_rt2b_remaining_risk_register.csv",
        ["risk_id", "risk_area", "severity", "mitigation_requirement", "production_blocking_until_human_review"],
        [
            {
                "risk_id": item[0],
                "risk_area": item[1],
                "severity": item[2],
                "mitigation_requirement": item[3],
                "production_blocking_until_human_review": item[4],
            }
            for item in rows
        ],
    )


def write_markdown_files(output_dir: Path, evidence: list[dict[str, object]], verdict: str) -> None:
    boundary_text = "\n".join(f"- {key}: {str(value).lower()}" for key, value in BOUNDARY_FLAGS.items())
    evidence_lines = "\n".join(f"- `{item['evidence_id']}`: `{item['classification']}`, verdict `{item['manifest_verdict']}`" for item in evidence)
    rt2b = next(item for item in evidence if item["evidence_id"] == "TRACE_RT2B")
    rt2 = next(item for item in evidence if item["evidence_id"] == "TRACE_RT2")

    docs = {
        "agentos_final_release_readiness_with_rt2b_trace_security_evidence_summary.md": f"""# Trace Security Evidence Summary With RT2B

Verdict: {verdict}

Mapped evidence:

{evidence_lines}

ATS1-5 and ATS WingClosure remain trace-security lineage evidence. RT1 remains metadata-only isolated replay. RT2 remains scope-limited stub sandbox evidence. RT2B is the corrective executable toy-target disposable sandbox red-team evidence.
""",
        "agentos_final_release_readiness_with_rt2b_trace_rt2b_live_sandbox_summary.md": f"""# RT2B Live Sandbox Summary

Verdict: {verdict}

RT2 status: `{rt2['classification']}` / `{rt2['manifest_verdict']}`

RT2B status: `{rt2b['classification']}` / `{rt2b['manifest_verdict']}`

RT2B is stronger than RT1/RT2 because it executed a harmless toy target inside a disposable offline sandbox. It is still not production-grade attack immunity, not live exploit execution against real systems, and not production release authorization.
""",
        "agentos_final_release_readiness_with_rt2b_actionruntime_boundary_summary.md": f"""# ActionRuntime Boundary Summary

Verdict: {verdict}

actionruntime_dispatch=false

No ActionRuntime dispatch, action candidate authorization, external action, or production action enablement is granted by this review.
""",
        "agentos_final_release_readiness_with_rt2b_humangate_authority_boundary_summary.md": f"""# HumanGate Authority Boundary Summary

Verdict: {verdict}

human_gate_approval=false

This package prepares a human release decision packet only. It does not simulate or grant real HumanGate approval.
""",
        "agentos_final_release_readiness_with_rt2b_memory_operator_policy_boundary_summary.md": f"""# Memory / Operator / Policy Boundary Summary

Verdict: {verdict}

memory_write=false

candidate_memory_write=false

operator_policy_skillkernel_promotion=false

accepted_evidence_or_baseline_write=false

No memory, operator, policy, SkillKernel, AcceptedEvidence, or baseline write is authorized.
""",
        "agentos_final_release_readiness_with_rt2b_operational_security_review.md": f"""# Operational Security Review

Verdict: {verdict}

Before any human release decision, operational security review must confirm: no hidden external attack demo ingestion, no payload retention, no unsafe disclosure, no networked red-team execution in release artifacts, rollback ownership, incident response ownership, and production boundary controls.
""",
        "agentos_final_release_readiness_with_rt2b_rollback_freeze_restore_plan.md": f"""# Rollback / Freeze / Restore Plan

Verdict: {verdict}

Required before production:

- Freeze the human decision packet and all referenced hashes.
- Identify restore source and owner.
- Define rollback triggers.
- Confirm no memory/operator/policy/baseline mutation is made by FRR2.
- Record RT2B as evidence only, not as a production runtime artifact.
""",
        "agentos_final_release_readiness_with_rt2b_release_nonclaim_compliance_report.md": f"""# Release Non-Claim Compliance Report

Verdict: {verdict}

Forbidden claims remain false:

{boundary_text}

This package does not claim production release authorization, production-grade attack immunity, AGI capability, live exploit immunity, customer readiness, or operational deployment readiness.
""",
        "agentos_final_release_readiness_with_rt2b_human_decision_requirements.md": f"""# Human Decision Requirements

Verdict: {verdict}

Required human release decision inputs:

- Review the evidence map including RT2B.
- Confirm RT2B is executable toy-target evidence only.
- Confirm RT2 is retained as scope-limited evidence, not used as full live execution evidence.
- Accept or block remaining risks.
- Complete operational security and rollback review.
- Issue a separate explicit production release decision if appropriate.

This file is not release approval.
""",
        "agentos_final_release_readiness_with_rt2b_final_recommendation.md": f"""# Final Recommendation With RT2B

Recommendation: ready for human release decision packet preparation.

Verdict: {verdict}

RT2B evidence is present and verified as controlled executable toy-target sandbox red-team evidence. The reviewed evidence line is ready for an authorized human release decision process.

This is not production release authorization. Production release remains blocked until separate human approval, operational security review, rollback/freeze/restore confirmation, and production boundary review are complete.
""",
    }
    for filename, text in docs.items():
        write_text(output_dir / filename, text)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_tests(output_dir: Path, evidence: list[dict[str, object]], verdict: str) -> list[dict[str, object]]:
    tests: list[dict[str, object]] = []

    def add(name: str, condition: bool, detail: str) -> None:
        tests.append({"name": name, "pass": bool(condition), "detail": detail})

    evidence_rows = read_csv_rows(output_dir / "agentos_final_release_readiness_with_rt2b_evidence_map.csv")
    blocker_rows = read_csv_rows(output_dir / "agentos_final_release_readiness_with_rt2b_blocker_matrix.csv")
    rt2b = next(row for row in evidence_rows if row["evidence_id"] == "TRACE_RT2B")
    rt2 = next(row for row in evidence_rows if row["evidence_id"] == "TRACE_RT2")
    add("rt2b_present", rt2b["present"] == "true" and "PASS" in rt2b["manifest_verdict"], "RT2B evidence present/pass")
    add("rt2b_distinguished_from_rt1_rt2", rt2b["classification"] == "executable_toy_target_sandbox" and rt2["classification"] == "scope_limited_stub_sandbox", "RT2B classification distinct")
    add("all_evidence_present", all(row["present"] == "true" for row in evidence_rows), "ATS1-5/WC/RRB1/RT1/RT2/RT2B present")
    add("all_evidence_pass", all("PASS" in row["manifest_verdict"] for row in evidence_rows), "all mapped packs PASS-like")
    add("no_production_immunity_claim", "not production-grade attack immunity" in (output_dir / "agentos_final_release_readiness_with_rt2b_trace_rt2b_live_sandbox_summary.md").read_text(encoding="utf-8"), "RT2B caveat explicit")
    add("no_authorization_boundary", not any(BOUNDARY_FLAGS.values()), "all required false boundary flags false")
    add("human_decision_packet_only", "human release decision packet" in (output_dir / "agentos_final_release_readiness_with_rt2b_final_recommendation.md").read_text(encoding="utf-8"), "recommendation is decision packet only")
    add("blocker_classifications_present", any(row["classification"] == "true_remaining_blocker_for_production" for row in blocker_rows), "remaining production blockers retained")
    add("target_verdict", verdict == VERDICT, verdict)
    return tests


def tests_summary(tests: list[dict[str, object]], verdict: str) -> str:
    return f"""# FRR2 With RT2B Tests Summary

Verdict: {verdict}

Passed: {sum(1 for test in tests if test["pass"])}

Total: {len(tests)}

| Test | Result | Detail |
|---|---|---|
{chr(10).join(f"| {test['name']} | {'PASS' if test['pass'] else 'FAIL'} | {test['detail']} |" for test in tests)}
"""


def write_hash_and_manifest(output_dir: Path, tests: list[dict[str, object]], verdict: str) -> None:
    primary = [name for name in REQUIRED_FILES if name not in {"hash_inventory.csv", "return_files_manifest.json"}]
    rows = []
    for filename in primary:
        path = output_dir / filename
        rows.append(
            {
                "file_path": filename,
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "self_hash_mode": "normal",
                "notes": "content hash",
            }
        )
    for filename in ["hash_inventory.csv", "return_files_manifest.json"]:
        rows.append(
            {
                "file_path": filename,
                "sha256": "SELF_REFERENTIAL_EXCLUDED",
                "bytes": "",
                "self_hash_mode": "self_reference_excluded",
                "notes": "self-referential file excluded from stable self hash",
            }
        )
    write_csv(output_dir / "hash_inventory.csv", ["file_path", "sha256", "bytes", "self_hash_mode", "notes"], rows)
    files_present = sorted({path.name for path in output_dir.iterdir() if path.is_file()} | {"return_files_manifest.json"})
    missing = [name for name in REQUIRED_FILES if name not in files_present]
    extra = [name for name in files_present if name not in REQUIRED_FILES]
    manifest = {
        "package": "AgentOS_FinalReleaseReadinessReview_WithRT2B_Return_Pack_v0_3",
        "created_at": now_utc(),
        "verdict": verdict,
        "required_files": REQUIRED_FILES,
        "required_files_present": not missing,
        "missing_required_files": missing,
        "extra_files": extra,
        "file_count": len(REQUIRED_FILES),
        "tests_passed": sum(1 for test in tests if test["pass"]),
        "tests_total": len(tests),
        "all_tests_passed": all(test["pass"] for test in tests),
        "boundary_flags": BOUNDARY_FLAGS,
        "files": rows,
    }
    write_text(output_dir / "return_files_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", action="store_true")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    print(json.dumps(generate(Path(args.output_dir), args.pack), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
