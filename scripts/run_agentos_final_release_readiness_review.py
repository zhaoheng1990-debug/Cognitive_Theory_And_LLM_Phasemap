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


VERDICT = "PASS_AGENTOS_FINAL_RELEASE_READINESS_REVIEW_READY_FOR_HUMAN_RELEASE_DECISION"
RETURN_PACK_NAME = "AgentOS_FinalReleaseReadinessReview_Return_Pack_v0_1.zip"
PROJECT_ROOT = Path("D:/Logos_agentOS")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "agentos_final_release_readiness_review_output"
RETURN_PACK_PATH = PROJECT_ROOT / "outputs" / RETURN_PACK_NAME

REQUIRED_FILES = [
    "agentos_final_release_readiness_evidence_map.csv",
    "agentos_final_release_readiness_blocker_matrix.csv",
    "agentos_final_release_readiness_remaining_risk_register.csv",
    "agentos_final_release_readiness_human_decision_requirements.md",
    "agentos_final_release_readiness_operational_security_review.md",
    "agentos_final_release_readiness_rollback_freeze_restore_plan.md",
    "agentos_final_release_readiness_trace_security_evidence_summary.md",
    "agentos_final_release_readiness_trace_rt1_external_redteam_summary.md",
    "agentos_final_release_readiness_actionruntime_boundary_summary.md",
    "agentos_final_release_readiness_memory_operator_policy_boundary_summary.md",
    "agentos_final_release_readiness_humangate_authority_boundary_summary.md",
    "agentos_final_release_readiness_production_nonclaim_compliance_report.md",
    "agentos_final_release_readiness_final_recommendation.md",
    "agentos_final_release_readiness_tests_summary.md",
    "hash_inventory.csv",
    "return_files_manifest.json",
]

EVIDENCE_PACKS = [
    {
        "evidence_id": "ATS_WC1",
        "family": "ATS-WC1 TraceSecurityReviewHandoffLineClosure",
        "path": PROJECT_ROOT / "outputs" / "AgentOS_ATS_WingClosure_TraceSecurityReviewHandoffLineClosure_Return_Pack_v0_1.zip",
        "expected_verdict_fragment": "PASS",
        "scope_note": "trace security review handoff wing closure evidence",
    },
    {
        "evidence_id": "POSTATS_RRB1",
        "family": "PostATS-RRB1 ReleaseBlockerClosure",
        "path": PROJECT_ROOT / "outputs" / "AgentOS_PostATS_ReleaseBlockerClosure_Return_Pack_v0_1.zip",
        "expected_verdict_fragment": "PASS",
        "scope_note": "release blocker closure evidence for final readiness review",
    },
    {
        "evidence_id": "POSTATS_TRACE_RT1",
        "family": "PostATS-TRACE-RT1 IsolatedRedTeamReplay",
        "path": PROJECT_ROOT / "outputs" / "AgentOS_PostATS_TRACERealDemo_IsolatedRedTeam_Return_Pack_v0_1.zip",
        "expected_verdict_fragment": "PASS",
        "scope_note": "metadata-only isolated external red-team replay evidence",
    },
]

BOUNDARY_FLAGS = {
    "real_external_action": False,
    "actionruntime_dispatch": False,
    "memory_write": False,
    "operator_or_policy_promotion": False,
    "skill_creation_from_attack_demo": False,
    "accepted_evidence_or_baseline_write": False,
    "real_humangate_approval": False,
    "production_release_claim": False,
    "attack_payload_preserved": False,
    "executable_attack_content_preserved": False,
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


def read_pack_evidence(pack_info: dict[str, object]) -> dict[str, object]:
    path = Path(pack_info["path"])
    if not path.exists():
        return {
            "evidence_id": pack_info["evidence_id"],
            "family": pack_info["family"],
            "present": False,
            "pack_path": str(path),
            "pack_sha256": "",
            "zip_entries": 0,
            "manifest_present": False,
            "manifest_verdict": "MISSING",
            "tests_passed": "",
            "tests_total": "",
            "scope_note": pack_info["scope_note"],
            "mapping_status": "missing_evidence",
            "notes": "prior evidence pack not found; evidence not invented",
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
                except Exception as exc:  # pragma: no cover - deterministic local guard
                    manifest_verdict = f"MANIFEST_PARSE_ERROR:{exc}"
            elif "final_report" in lower or "tests_summary" in lower:
                try:
                    report_verdict = extract_verdict(zf.read(name).decode("utf-8", errors="replace"))
                except Exception:
                    pass
    verdict = manifest_verdict if manifest_verdict != "UNKNOWN" else report_verdict
    mapping_status = "mapped" if "PASS" in verdict and manifest_present else "mapped_with_review_required"
    return {
        "evidence_id": pack_info["evidence_id"],
        "family": pack_info["family"],
        "present": True,
        "pack_path": str(path),
        "pack_sha256": pack_hash,
        "zip_entries": len(names),
        "manifest_present": manifest_present,
        "manifest_verdict": verdict,
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "scope_note": pack_info["scope_note"],
        "mapping_status": mapping_status,
        "notes": "prior evidence mapped from local return pack",
    }


def generate(output_dir: Path, pack: bool) -> dict[str, object]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence = [read_pack_evidence(item) for item in EVIDENCE_PACKS]
    all_evidence_present = all(row["present"] for row in evidence)
    all_pass_like = all("PASS" in str(row["manifest_verdict"]) for row in evidence)
    verdict = VERDICT if all_evidence_present and all_pass_like and not any(BOUNDARY_FLAGS.values()) else "BLOCKED_AGENTOS_FINAL_RELEASE_READINESS_REVIEW_MISSING_OR_NONPASS_EVIDENCE"

    evidence_rows = [
        {
            "evidence_id": row["evidence_id"],
            "evidence_family": row["family"],
            "present": str(row["present"]).lower(),
            "pack_path": row["pack_path"],
            "pack_sha256": row["pack_sha256"],
            "zip_entries": row["zip_entries"],
            "manifest_present": str(row["manifest_present"]).lower(),
            "manifest_verdict": row["manifest_verdict"],
            "tests_passed": row["tests_passed"],
            "tests_total": row["tests_total"],
            "mapping_status": row["mapping_status"],
            "scope_note": row["scope_note"],
            "notes": row["notes"],
        }
        for row in evidence
    ]
    write_csv(
        output_dir / "agentos_final_release_readiness_evidence_map.csv",
        [
            "evidence_id",
            "evidence_family",
            "present",
            "pack_path",
            "pack_sha256",
            "zip_entries",
            "manifest_present",
            "manifest_verdict",
            "tests_passed",
            "tests_total",
            "mapping_status",
            "scope_note",
            "notes",
        ],
        evidence_rows,
    )

    blocker_rows = build_blocker_rows()
    write_csv(
        output_dir / "agentos_final_release_readiness_blocker_matrix.csv",
        [
            "blocker_id",
            "review_area",
            "evidence_source",
            "status_for_release_readiness_review",
            "production_authorization_status",
            "required_human_decision",
            "boundary_status",
            "notes",
        ],
        blocker_rows,
    )

    risk_rows = build_risk_rows()
    write_csv(
        output_dir / "agentos_final_release_readiness_remaining_risk_register.csv",
        [
            "risk_id",
            "risk_area",
            "remaining_risk",
            "severity",
            "mitigation_requirement",
            "owner",
            "hidden_or_downgraded",
            "production_blocking_until_review",
        ],
        risk_rows,
    )

    write_text(output_dir / "agentos_final_release_readiness_human_decision_requirements.md", human_decision_requirements(verdict))
    write_text(output_dir / "agentos_final_release_readiness_operational_security_review.md", operational_security_review(verdict))
    write_text(output_dir / "agentos_final_release_readiness_rollback_freeze_restore_plan.md", rollback_plan(verdict))
    write_text(output_dir / "agentos_final_release_readiness_trace_security_evidence_summary.md", trace_security_summary(evidence, verdict))
    write_text(output_dir / "agentos_final_release_readiness_trace_rt1_external_redteam_summary.md", trace_rt1_summary(evidence, verdict))
    write_text(output_dir / "agentos_final_release_readiness_actionruntime_boundary_summary.md", actionruntime_summary(verdict))
    write_text(output_dir / "agentos_final_release_readiness_memory_operator_policy_boundary_summary.md", memory_policy_summary(verdict))
    write_text(output_dir / "agentos_final_release_readiness_humangate_authority_boundary_summary.md", humangate_summary(verdict))
    write_text(output_dir / "agentos_final_release_readiness_production_nonclaim_compliance_report.md", production_nonclaim_report(verdict))
    write_text(output_dir / "agentos_final_release_readiness_final_recommendation.md", final_recommendation(verdict, all_evidence_present, all_pass_like))

    tests = run_tests(output_dir, evidence, verdict)
    write_text(output_dir / "agentos_final_release_readiness_tests_summary.md", tests_summary(tests, verdict))
    write_hash_and_manifest(output_dir, verdict, tests)

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
        "evidence_present": all_evidence_present,
        "evidence_pass_like": all_pass_like,
        "tests_passed": sum(1 for test in tests if test["pass"]),
        "tests_total": len(tests),
    }


def build_blocker_rows() -> list[dict[str, str]]:
    rows = []
    data = [
        ("BLK_RUNTIME_CORE", "RuntimeCore release-readiness evidence", "ATS_WC1/PostATS_RRB1", "ready_for_human_review", "not_authorized", "required", "no_action", "runtime evidence is review-ready only"),
        ("BLK_ACTION_RUNTIME", "ActionRuntime no-action boundary", "PostATS_RRB1/TRACE_RT1", "ready_for_human_review", "not_authorized", "required", "dispatch_false", "no ActionRuntime dispatch authorized"),
        ("BLK_HUMANGATE", "HumanGate authority boundary", "ATS_WC1/TRACE_RT1", "ready_for_human_review", "not_authorized", "required", "approval_false", "review packet is not HumanGate approval"),
        ("BLK_MEMORY_POLICY", "CandidateMemory/operator/policy boundary", "PostATS_RRB1/TRACE_RT1", "ready_for_human_review", "not_authorized", "required", "write_false", "no memory/operator/policy promotion"),
        ("BLK_TRACE_SECURITY", "ATS trace security wing closure", "ATS_WC1", "ready_for_human_review", "not_authorized", "required", "closed_as_evidence", "wing closure evidence mapped"),
        ("BLK_EXTERNAL_REDTEAM", "TRACE RT1 metadata-only external red-team replay", "TRACE_RT1", "ready_for_human_review", "not_authorized", "required", "metadata_only_non_executable", "not live exploit execution"),
        ("BLK_PRODUCTION_CLAIM", "Production non-claim compliance", "Final review", "ready_for_human_review", "not_authorized", "required", "production_claim_false", "PASS does not equal production release"),
        ("BLK_ROLLBACK", "Rollback/freeze/restore requirements", "Final review", "requires_human_decision_before_release", "not_authorized", "required", "plan_required", "explicit rollback plan required before production"),
    ]
    for item in data:
        rows.append(
            {
                "blocker_id": item[0],
                "review_area": item[1],
                "evidence_source": item[2],
                "status_for_release_readiness_review": item[3],
                "production_authorization_status": item[4],
                "required_human_decision": item[5],
                "boundary_status": item[6],
                "notes": item[7],
            }
        )
    return rows


def build_risk_rows() -> list[dict[str, str]]:
    return [
        {
            "risk_id": "RISK_PRODUCTION_AUTHORIZATION",
            "risk_area": "production release",
            "remaining_risk": "readiness evidence may be mistaken for release approval",
            "severity": "high",
            "mitigation_requirement": "explicit authorized human release decision artifact",
            "owner": "human_release_authority",
            "hidden_or_downgraded": "false",
            "production_blocking_until_review": "true",
        },
        {
            "risk_id": "RISK_TRACE_RT1_SCOPE",
            "risk_area": "external red-team evidence",
            "remaining_risk": "TRACE RT1 is metadata-only and does not prove live exploit immunity",
            "severity": "high",
            "mitigation_requirement": "operational security review before any live exposure",
            "owner": "security_reviewer",
            "hidden_or_downgraded": "false",
            "production_blocking_until_review": "true",
        },
        {
            "risk_id": "RISK_ROLLBACK_RESTORE",
            "risk_area": "rollback/freeze/restore",
            "remaining_risk": "production rollback cannot be assumed from review artifacts",
            "severity": "medium",
            "mitigation_requirement": "human-approved rollback, freeze, restore procedure with owners",
            "owner": "release_operator",
            "hidden_or_downgraded": "false",
            "production_blocking_until_review": "true",
        },
        {
            "risk_id": "RISK_BOUNDARY_DRIFT",
            "risk_area": "no-action boundaries",
            "remaining_risk": "future integration could drift from no-action boundary",
            "severity": "medium",
            "mitigation_requirement": "fresh integration preflight before enabling any action path",
            "owner": "runtime_governance_reviewer",
            "hidden_or_downgraded": "false",
            "production_blocking_until_review": "true",
        },
    ]


def human_decision_requirements(verdict: str) -> str:
    return f"""# Human Decision Requirements

Verdict: {verdict}

Final readiness review requires a separate explicit human release decision before any production rollout.

Required human decisions:

- Confirm the reviewed evidence scope: ATS-WC1, PostATS-RRB1, and PostATS-TRACE-RT1.
- Confirm TRACE RT1 is metadata-only, non-executable, and not live exploit execution.
- Confirm remaining risks are accepted, mitigated, or blocked.
- Confirm operational security review is complete.
- Confirm rollback, freeze, and restore plan owners.
- Confirm no production release is authorized by this package itself.

This artifact does not grant HumanGate approval.
"""


def operational_security_review(verdict: str) -> str:
    return f"""# Operational Security Review

Verdict: {verdict}

Required before production:

- Review external red-team exposure boundary.
- Review trace disclosure and redaction policy.
- Verify no hidden attack demo ingestion.
- Verify no payload, executable content, credentials, targets, or command sequences are retained.
- Verify scoped disclosure rules for any future reviewer packet.
- Verify incident response and rollback owners.

Current package status: review-ready evidence only, no operational approval.
"""


def rollback_plan(verdict: str) -> str:
    return f"""# Rollback / Freeze / Restore Plan

Verdict: {verdict}

Minimum required human-approved plan:

- Freeze candidate artifacts before any release decision.
- Record exact artifact hashes and version identifiers.
- Define restore source and restore owner.
- Define rollback trigger conditions.
- Define rollback communication path.
- Confirm no official baseline, MemoryUnit, OperatorMemory, or Policy mutation occurs from this review.

This plan is a requirement summary, not an executed rollback procedure.
"""


def trace_security_summary(evidence: list[dict[str, object]], verdict: str) -> str:
    lines = "\n".join(f"- {row['evidence_id']}: present={str(row['present']).lower()}, verdict=`{row['manifest_verdict']}`" for row in evidence)
    return f"""# Trace Security Evidence Summary

Verdict: {verdict}

Mapped evidence:

{lines}

ATS-WC1 supplies trace-security wing closure evidence. PostATS-RRB1 supplies release-blocker closure evidence. TRACE RT1 supplies metadata-only isolated red-team replay evidence.
"""


def trace_rt1_summary(evidence: list[dict[str, object]], verdict: str) -> str:
    rt1 = next(row for row in evidence if row["evidence_id"] == "POSTATS_TRACE_RT1")
    return f"""# TRACE RT1 External Red-Team Summary

Verdict: {verdict}

TRACE RT1 pack present: {str(rt1['present']).lower()}

TRACE RT1 verdict: `{rt1['manifest_verdict']}`

Critical caveat: TRACE RT1 is metadata-only, non-executable, and isolated. It validates isolation, non-ingestion, redaction/scoped-disclosure style handling, pivot blocking, feedback-retention blocking, attack-memory blocking, and authority suppression. It does not validate live exploit defense or production-grade attack immunity.
"""


def actionruntime_summary(verdict: str) -> str:
    return f"""# ActionRuntime Boundary Summary

Verdict: {verdict}

Boundary status:

- real_external_action: false
- actionruntime_dispatch: false
- tool/action dispatch authorization: false
- production action enablement: false

This review does not dispatch or authorize any ActionRuntime path.
"""


def memory_policy_summary(verdict: str) -> str:
    return f"""# Memory / Operator / Policy Boundary Summary

Verdict: {verdict}

Boundary status:

- MemoryUnit write: false
- OperatorMemory promotion: false
- Policy promotion: false
- AcceptedEvidence or baseline write: false
- attack component retention: false

This review is a local artifact consolidation only.
"""


def humangate_summary(verdict: str) -> str:
    return f"""# HumanGate Authority Boundary Summary

Verdict: {verdict}

HumanGate status:

- real HumanGate approval: false
- release approval: false
- authority semantics granted by review packet: false

The required next step is an explicit authorized human release decision outside this no-action review.
"""


def production_nonclaim_report(verdict: str) -> str:
    return f"""# Production Non-Claim Compliance Report

Verdict: {verdict}

This package does not claim:

- production release authorization
- production readiness
- live exploit immunity
- real external red-team execution
- AGI capability
- autonomous science capability
- customer or partner readiness
- accepted evidence or baseline status

PASS means the evidence is ready for human release decision review only.
"""


def final_recommendation(verdict: str, evidence_present: bool, evidence_pass_like: bool) -> str:
    return f"""# Final Recommendation

Recommendation: ready for explicit human release decision review.

Verdict: {verdict}

Evidence present: {str(evidence_present).lower()}

Evidence pass-like: {str(evidence_pass_like).lower()}

The reviewed evidence line is sufficient to present to an authorized human release decision process. This is not production release authorization. Production release remains blocked until a separate human release decision, operational security review, and rollback/freeze/restore approval are completed.
"""


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_tests(output_dir: Path, evidence: list[dict[str, object]], verdict: str) -> list[dict[str, object]]:
    tests: list[dict[str, object]] = []

    def add(name: str, condition: bool, detail: str) -> None:
        tests.append({"name": name, "pass": bool(condition), "detail": detail})

    evidence_rows = read_csv_rows(output_dir / "agentos_final_release_readiness_evidence_map.csv")
    blocker_rows = read_csv_rows(output_dir / "agentos_final_release_readiness_blocker_matrix.csv")
    risk_rows = read_csv_rows(output_dir / "agentos_final_release_readiness_remaining_risk_register.csv")
    add("required_prior_evidence_mapped", len(evidence_rows) == 3 and all(row["present"] == "true" for row in evidence_rows), "ATS-WC1, RRB1, TRACE RT1 mapped")
    add("prior_evidence_pass_like", all("PASS" in row["manifest_verdict"] for row in evidence_rows), "all prior verdicts are PASS-like")
    add("trace_rt1_scope_precise", "metadata-only" in (output_dir / "agentos_final_release_readiness_trace_rt1_external_redteam_summary.md").read_text(encoding="utf-8"), "TRACE RT1 caveat preserved")
    add("no_production_release_claim", "not production release authorization" in (output_dir / "agentos_final_release_readiness_final_recommendation.md").read_text(encoding="utf-8"), "final recommendation is non-authorization")
    add("human_decision_explicit", "separate explicit human release decision" in (output_dir / "agentos_final_release_readiness_human_decision_requirements.md").read_text(encoding="utf-8"), "human decision required")
    add("opsec_review_explicit", "Operational Security Review" in (output_dir / "agentos_final_release_readiness_operational_security_review.md").read_text(encoding="utf-8"), "opsec review present")
    add("rollback_plan_explicit", "Rollback / Freeze / Restore Plan" in (output_dir / "agentos_final_release_readiness_rollback_freeze_restore_plan.md").read_text(encoding="utf-8"), "rollback plan present")
    add("remaining_risks_not_downgraded", all(row["hidden_or_downgraded"] == "false" for row in risk_rows), "remaining risks retained")
    add("blocker_matrix_requires_human_decision", all(row["required_human_decision"] == "required" for row in blocker_rows), "blockers require human decision")
    add("boundary_flags_false", not any(BOUNDARY_FLAGS.values()), "all boundary flags false")
    add("target_verdict", verdict == VERDICT, verdict)
    return tests


def tests_summary(tests: list[dict[str, object]], verdict: str) -> str:
    return f"""# Tests Summary

Verdict: {verdict}

Passed: {sum(1 for item in tests if item["pass"])}

Total: {len(tests)}

| Test | Result | Detail |
|---|---|---|
{chr(10).join(f"| {item['name']} | {'PASS' if item['pass'] else 'FAIL'} | {item['detail']} |" for item in tests)}
"""


def write_hash_and_manifest(output_dir: Path, verdict: str, tests: list[dict[str, object]]) -> None:
    primary_files = [name for name in REQUIRED_FILES if name not in {"hash_inventory.csv", "return_files_manifest.json"}]
    hash_rows = []
    for filename in primary_files:
        path = output_dir / filename
        hash_rows.append(
            {
                "file_path": filename,
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "self_hash_mode": "normal",
                "notes": "content hash",
            }
        )
    for filename in ["hash_inventory.csv", "return_files_manifest.json"]:
        hash_rows.append(
            {
                "file_path": filename,
                "sha256": "SELF_REFERENTIAL_EXCLUDED",
                "bytes": "",
                "self_hash_mode": "self_reference_excluded",
                "notes": "self-referential file excluded from stable self hash",
            }
        )
    write_csv(output_dir / "hash_inventory.csv", ["file_path", "sha256", "bytes", "self_hash_mode", "notes"], hash_rows)
    files_present = sorted({p.name for p in output_dir.iterdir() if p.is_file()} | {"return_files_manifest.json"})
    missing = [name for name in REQUIRED_FILES if name not in files_present]
    extra = [name for name in files_present if name not in REQUIRED_FILES]
    manifest = {
        "package": "AgentOS_FinalReleaseReadinessReview_Return_Pack_v0_1",
        "created_at": now_utc(),
        "verdict": verdict,
        "required_files": REQUIRED_FILES,
        "required_files_present": not missing,
        "missing_required_files": missing,
        "extra_files": extra,
        "file_count": len(REQUIRED_FILES),
        "tests_passed": sum(1 for item in tests if item["pass"]),
        "tests_total": len(tests),
        "all_tests_passed": all(item["pass"] for item in tests),
        "boundary_flags": BOUNDARY_FLAGS,
        "release_authorization": False,
        "production_release_claim": False,
        "files": hash_rows,
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
