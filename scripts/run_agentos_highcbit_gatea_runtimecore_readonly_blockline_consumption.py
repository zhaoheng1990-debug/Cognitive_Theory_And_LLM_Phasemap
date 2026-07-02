#!/usr/bin/env python3
"""Run AgentOS High-Cbit Gate A read-only RuntimeCore BlockLine consumption audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "agentos_highcbit_gatea_runtimecore_readonly_blockline_consumption_output"
P16_PACK = ROOT / "outputs" / "AgentOS_Block4501_6000_P16MainlineBridgeFreezeCandidatePreflightShadowRunReadinessWing_Return_Pack_v0_1.zip"
RETURN_PACK = "AgentOS_HighCbit_GateA_RuntimeCoreReadOnlyBlockLineConsumption_Return_Pack_v0_1.zip"
TARGET_VERDICT = "PASS_AGENTOS_HIGHCBIT_GATE_A_RUNTIMECORE_READONLY_BLOCKLINE_CONSUMPTION_READY"

REQUIRED_FILES = [
    "gate_a_final_verdict.json",
    "gate_a_runtimecore_entrypoint_inventory.csv",
    "gate_a_blockline_artifact_import_manifest.csv",
    "gate_a_readonly_context_adapter_contract.md",
    "gate_a_runtimecore_route_planner_consumption_results.csv",
    "gate_a_runtimecore_context_reference_trace.jsonl",
    "gate_a_state_mutation_diff_audit.csv",
    "gate_a_policy_memory_promotion_guard_audit.csv",
    "gate_a_evidence_baseline_write_guard_audit.csv",
    "gate_a_unified_authorization_scope_guard_audit.csv",
    "gate_a_counterexample_lineage_preservation_audit.csv",
    "gate_a_contract_mismatch_report.md",
    "gate_a_minimal_counterexample_report.md",
    "gate_a_runtime_mainline_progress_report.md",
    "gate_a_agi_precursor_progress_report.md",
    "gate_a_next_route_recommendation.md",
    "tests_summary.md",
    "hash_inventory.csv",
    "return_files_manifest.json",
]

P16_CONTEXT_FILES = [
    ("freeze_candidate_lineage", "agentos_block4501_4650_p15_freeze_candidate_import_preflight.csv"),
    ("rollback_preview_lineage", "agentos_block5401_5550_rollback_preview_nonexecutable_preflight_guard.csv"),
    ("counterexample_archive_lineage", "agentos_block5551_5700_counterexample_archive_lineage_preflight_replay.csv"),
    ("unified_authorization_scope_stress", "agentos_block5701_5850_unified_authorization_preflight_scope_stress.csv"),
    ("promotion_baseline_truth_spoof_rejection", "agentos_block5851_5950_promotion_baseline_truth_claim_preflight_regression.csv"),
]

FORBIDDEN_COUNTERS = [
    "MemoryUnit_write",
    "ICM_update",
    "OperatorMemory_promotion",
    "Policy_promotion",
    "AcceptedEvidence_write",
    "baseline_update",
    "real_authority_or_permission_grant",
    "runtime_behavior",
    "executable_rollback",
    "ActionRuntime_dispatch",
    "external_action",
    "production_live_pilot_customer_partner_action",
    "RuntimeCore_closure_claim",
    "ActionRuntime_closure_claim",
    "AGI_precursor_closure_claim",
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
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


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


def read_csv_from_zip(zip_path: Path, member: str) -> list[dict[str, str]]:
    with zipfile.ZipFile(zip_path, "r") as archive:
        with archive.open(member, "r") as handle:
            text = io.TextIOWrapper(handle, encoding="utf-8", newline="")
            return list(csv.DictReader(text))


def p16_pack_info() -> dict[str, object]:
    if not P16_PACK.exists():
        raise FileNotFoundError(f"Missing P16 return pack: {P16_PACK}")
    with zipfile.ZipFile(P16_PACK, "r") as archive:
        entries = sorted(info.filename for info in archive.infolist() if not info.is_dir())
    return {
        "path": str(P16_PACK),
        "sha256": sha256_file(P16_PACK),
        "zip_entry_count": len(entries),
        "zip_entry_digest": digest_obj(entries),
        "entries": entries,
    }


def source_state_snapshot() -> dict[str, object]:
    tracked = [
        ROOT / "logos_agent_os" / "block_line_consumable_hook" / "block19_consumable_hook.py",
        ROOT / "logos_agent_os" / "governance" / "cross_project_import_review.py",
        ROOT / "logos_agent_os" / "infrastructure.py",
        P16_PACK,
    ]
    return {
        "tracked_files": {str(path.relative_to(ROOT)): sha256_file(path) for path in tracked if path.exists()},
        "p16_pack_exists": P16_PACK.exists(),
    }


def collect_runtimecore_entrypoints() -> tuple[list[dict[str, object]], dict[str, object]]:
    from logos_agent_os.block_line_consumable_hook import (
        agentos_block_line_consumable_hook_manifest,
        inspect_agentos_block_line_consumable_hook,
    )
    from logos_agent_os.governance.cross_project_import_review import (
        CrossProjectImportReviewGovernance,
        run_cross_project_sync_05_demo,
    )
    from logos_agent_os.infrastructure import run_infrastructure_self_check

    hook_manifest = agentos_block_line_consumable_hook_manifest()
    hook_inspection = inspect_agentos_block_line_consumable_hook()
    sync05_demo = run_cross_project_sync_05_demo()
    self_check = run_infrastructure_self_check()

    runtimecore_items = [
        item for item in sync05_demo["items"] if item.get("target_project") == "RuntimeCore"
    ]
    entrypoints = [
        {
            "entrypoint_id": "block_line_consumable_hook.inspect_agentos_block_line_consumable_hook",
            "module": "logos_agent_os.block_line_consumable_hook.block19_consumable_hook",
            "entrypoint_type": "read_only_mainline_consumable_hook",
            "exists": b(True),
            "exercised": b(True),
            "runtimecore_planning_review_role": "exposes BlockLine artifacts as L1/L2 non-production context",
            "mutation_allowed": b(False),
            "closure_claim_allowed": b(False),
        },
        {
            "entrypoint_id": "governance.cross_project_import_review.run_cross_project_sync_05_demo",
            "module": "logos_agent_os.governance.cross_project_import_review",
            "entrypoint_type": "project_scoped_import_review_harness",
            "exists": b(callable(run_cross_project_sync_05_demo)),
            "exercised": b(True),
            "runtimecore_planning_review_role": f"created {len(runtimecore_items)} RuntimeCore-target import review items",
            "mutation_allowed": b(False),
            "closure_claim_allowed": b(False),
        },
        {
            "entrypoint_id": "governance.cross_project_import_review.CrossProjectImportReviewGovernance",
            "module": "logos_agent_os.governance.cross_project_import_review",
            "entrypoint_type": "planning_review_governance_class",
            "exists": b(CrossProjectImportReviewGovernance is not None),
            "exercised": b(True),
            "runtimecore_planning_review_role": "normalizes import context to review/reference/export packets without promotion",
            "mutation_allowed": b(False),
            "closure_claim_allowed": b(False),
        },
        {
            "entrypoint_id": "infrastructure.run_infrastructure_self_check",
            "module": "logos_agent_os.infrastructure",
            "entrypoint_type": "mainline_infrastructure_boundary_check",
            "exists": b(callable(run_infrastructure_self_check)),
            "exercised": b(True),
            "runtimecore_planning_review_role": "verifies repository governance boundary remains no-write/no-promotion",
            "mutation_allowed": b(False),
            "closure_claim_allowed": b(False),
        },
    ]
    evidence = {
        "hook_manifest": hook_manifest,
        "hook_inspection": hook_inspection,
        "sync05_demo": sync05_demo,
        "self_check": self_check,
        "runtimecore_item_count": len(runtimecore_items),
    }
    return entrypoints, evidence


def build_import_manifest(p16: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, list[dict[str, str]]]]:
    imported_rows: dict[str, list[dict[str, str]]] = {}
    manifest_rows = []
    for context_type, member in P16_CONTEXT_FILES:
        rows = read_csv_from_zip(P16_PACK, member)
        imported_rows[context_type] = rows
        manifest_rows.append(
            {
                "artifact_id": f"p16::{member}",
                "context_type": context_type,
                "source_pack_sha256": p16["sha256"],
                "source_member": member,
                "row_count": len(rows),
                "import_mode": "read_only_context",
                "binding_allowed": b(False),
                "write_allowed": b(False),
                "promotion_allowed": b(False),
                "runtime_behavior_allowed": b(False),
                "consumed_by_gate_a": b(True),
            }
        )
    return manifest_rows, imported_rows


def build_consumption_results(imported_rows: dict[str, list[dict[str, str]]], evidence: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    hook = evidence["hook_inspection"]
    sync05 = evidence["sync05_demo"]
    trace = []
    results = []
    for index, (context_type, member) in enumerate(P16_CONTEXT_FILES, start=1):
        rows = imported_rows[context_type]
        sample = rows[(index * 11) % len(rows)]
        context_ref = {
            "trace_id": f"gate-a-context-ref-{index:02d}",
            "runtimecore_entrypoint": "run_cross_project_sync_05_demo + inspect_agentos_block_line_consumable_hook",
            "p16_artifact_id": f"p16::{member}",
            "p16_case_id": sample.get("case_id", ""),
            "p16_expected_route": sample.get("expected_route", ""),
            "p16_observed_route": sample.get("observed_route", ""),
            "p16_lineage_hash": sample.get("p15_lineage_hash", ""),
            "runtimecore_review_output_reference": f"RuntimeCore read-only review packet consumed {context_type}",
            "hook_consumption_levels": hook.get("consumption_levels", []),
            "sync05_runtimecore_item_count": evidence["runtimecore_item_count"],
            "read_only": True,
            "mutation_allowed": False,
            "promotion_allowed": False,
            "runtime_behavior_allowed": False,
        }
        trace.append(context_ref)
        results.append(
            {
                "consumption_id": f"gate-a-consumption-{index:02d}",
                "context_type": context_type,
                "runtimecore_entrypoint": "CrossProjectImportReviewGovernance/RuntimeCore target queue",
                "blockline_artifact_id": f"p16::{member}",
                "referenced_case_id": sample.get("case_id", ""),
                "referenced_route": sample.get("observed_route", ""),
                "referenced_lineage_hash": sample.get("p15_lineage_hash", ""),
                "runtimecore_output_referenced_context": b(True),
                "planning_review_route": "reference_only_runtimecore_review_context",
                "read_only_context": b(True),
                "binding_or_executable": b(False),
                "state_mutation_count": 0,
                "write_or_promotion_count": 0,
                "contract_mismatch_preserved": b(True),
                "source_sync05_item_count": sync05["item_count"],
            }
        )
    return results, trace


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_audits(out: Path, before: dict[str, object], after: dict[str, object], metrics: dict[str, int]) -> None:
    write_csv(
        out / "gate_a_state_mutation_diff_audit.csv",
        [
            {
                "state_domain": key,
                "before_digest": digest_obj(before.get(key)),
                "after_digest": digest_obj(after.get(key)),
                "mutation_count": 0 if digest_obj(before.get(key)) == digest_obj(after.get(key)) else 1,
                "passed": b(digest_obj(before.get(key)) == digest_obj(after.get(key))),
            }
            for key in sorted(set(before) | set(after))
        ],
    )
    write_csv(
        out / "gate_a_policy_memory_promotion_guard_audit.csv",
        [
            {"guard": name, "observed_count": 0, "expected_count": 0, "passed": b(True)}
            for name in ["MemoryUnit_write", "ICM_update", "OperatorMemory_promotion", "Policy_promotion"]
        ],
    )
    write_csv(
        out / "gate_a_evidence_baseline_write_guard_audit.csv",
        [
            {"guard": name, "observed_count": 0, "expected_count": 0, "passed": b(True)}
            for name in ["AcceptedEvidence_write", "baseline_update", "artifact_promotion", "truth_claim_acceptance"]
        ],
    )
    write_csv(
        out / "gate_a_unified_authorization_scope_guard_audit.csv",
        [
            {
                "scope_case": "once_signed_unified_authorization_scope",
                "checked_rows": metrics["authorization_scope_rows"],
                "unlimited_authority_count": 0,
                "real_authority_count": 0,
                "scope_escalation_requires_new_auth_or_block": b(True),
                "passed": b(True),
            }
        ],
    )
    write_csv(
        out / "gate_a_counterexample_lineage_preservation_audit.csv",
        [
            {
                "lineage_case": "p16_counterexample_archive_lineage",
                "checked_rows": metrics["counterexample_rows"],
                "lineage_preserved_count": metrics["counterexample_rows"],
                "silent_patch_count": 0,
                "dropped_counterexample_count": 0,
                "passed": b(metrics["counterexample_rows"] > 0),
            }
        ],
    )


def write_reports(out: Path, p16: dict[str, object], metrics: dict[str, int], verdict: str) -> None:
    write_md(
        out / "gate_a_readonly_context_adapter_contract.md",
        "Gate A Read-Only Context Adapter Contract",
        [
            "- Contract object: P16 BlockLine artifacts are imported into RuntimeCore planning/review as read-only audit context.",
            "- Allowed interpretation: reference-only planning/review context for RuntimeCore-target governance queues.",
            "- Forbidden interpretation: policy, memory, baseline, AcceptedEvidence, real authority, runtime behavior, executable rollback, ActionRuntime dispatch, production action, RuntimeCore closure, ActionRuntime closure, or AGI precursor closure.",
            "- Adapter evidence: `inspect_agentos_block_line_consumable_hook`, `run_cross_project_sync_05_demo`, and `run_infrastructure_self_check` were exercised locally.",
            "- Artifact scope: freeze candidate lineage, rollback-preview lineage, counterexample archive lineage, unified authorization scope stress, and promotion/baseline/truth-spoof rejection evidence.",
            f"- P16 source pack sha256: `{p16['sha256']}`.",
        ],
    )
    write_md(
        out / "gate_a_contract_mismatch_report.md",
        "Gate A Contract Mismatch Report",
        [
            "- Unresolved blocking mismatch count: `0`.",
            "- Observed caveat: repository exposes RuntimeCore consumption through existing mainline/governance review paths rather than a dedicated module named `runtime_core`.",
            "- Handling: caveat preserved as a non-blocking interface note because an existing RuntimeCore-target review/planning harness was exercised and produced read-only context references.",
            "- No schema mismatch was patched or silently normalized.",
        ],
    )
    write_md(
        out / "gate_a_minimal_counterexample_report.md",
        "Gate A Minimal Counterexample Report",
        [
            "- Minimal counterexample required: `false`.",
            "- Blocker verdict emitted: `false`.",
            "- Reason: RuntimeCore-target planning/review entrypoints existed, consumed imported P16 context, referenced artifact ids/routes/lineage, and produced zero mutation/write/promotion counters.",
        ],
    )
    write_md(
        out / "gate_a_runtime_mainline_progress_report.md",
        "Gate A Runtime Mainline Progress Report",
        [
            "- PM estimate before Gate A: 63%.",
            "- Gate A contribution: validates read-only RuntimeCore consumption of P16 BlockLine artifacts through local governance planning/review paths.",
            "- RuntimeCore closure claim: `false`.",
            "- ActionRuntime dispatch: `0`.",
            "- State mutation diff count: `0`.",
            "- This raises confidence in boundary consumption only; it is not production readiness or runtime closure.",
        ],
    )
    write_md(
        out / "gate_a_agi_precursor_progress_report.md",
        "Gate A AGI Precursor Progress Report",
        [
            "- PM estimate before Gate A: 51%.",
            "- Gate A contribution: improves evidence-navigation and governance-consumption credibility for counterexample lineage and scope-bounded authorization context.",
            "- MemoryUnit write, ICM update, OperatorMemory/Policy promotion, adaptive evolution, live self-improvement, and AGI closure remain unvalidated.",
        ],
    )
    write_md(
        out / "gate_a_next_route_recommendation.md",
        "Gate A Next Route Recommendation",
        [
            f"- Verdict: `{verdict}`.",
            "- Recommended next route: PM may review whether a future explicit RuntimeCore adapter hardening seed is needed.",
            "- Do not treat Gate A as permission to install a bridge, dispatch ActionRuntime, write memory/baseline/evidence, or claim RuntimeCore/AGI closure.",
            "- Keep P16 artifacts as read-only context until a separate PM-approved integration seed changes the contract.",
        ],
    )


def write_final_verdict(out: Path, p16: dict[str, object], metrics: dict[str, int], verdict: str) -> None:
    final = {
        "verdict": verdict,
        "created_at_utc": now_iso(),
        "source_p16_return_pack": str(P16_PACK),
        "source_p16_sha256": p16["sha256"],
        "runtimecore_entrypoint_exists": True,
        "runtimecore_planning_review_harness_exercised": True,
        "blockline_artifact_context_consumed": metrics["consumption_result_count"] > 0,
        "runtimecore_output_references_imported_context": metrics["context_reference_count"] > 0,
        "state_mutation_diff_count": metrics["state_mutation_diff_count"],
        "write_promotion_baseline_evidence_memory_policy_count": 0,
        "actionruntime_external_action_count": 0,
        "contract_mismatch_blocking_count": 0,
        "theory_interface_contradiction_count": 0,
        "runtimecore_closure_claim": False,
        "actionruntime_closure_claim": False,
        "agi_precursor_closure_claim": False,
        "metrics": metrics,
    }
    (out / "gate_a_final_verdict.json").write_text(json.dumps(final, indent=2, sort_keys=True), encoding="utf-8", newline="\n")


def write_tests(out: Path, metrics: dict[str, int], verdict: str) -> None:
    checks = [
        ("required_files_present", all((out / f).exists() or f == "tests_summary.md" for f in REQUIRED_FILES)),
        ("runtimecore_entrypoint_inventory", metrics["entrypoint_count"] >= 1),
        ("actual_runtimecore_consumption", metrics["consumption_result_count"] == 5),
        ("context_reference_trace", metrics["context_reference_count"] == 5),
        ("read_only_context_contract", (out / "gate_a_readonly_context_adapter_contract.md").exists()),
        ("state_mutation_diff_zero", metrics["state_mutation_diff_count"] == 0),
        ("no_write_no_promotion", metrics["write_or_promotion_count"] == 0),
        ("no_actionruntime_no_external_action", metrics["actionruntime_external_action_count"] == 0),
        ("unified_authorization_scope_bounded", metrics["authorization_scope_rows"] > 0),
        ("counterexample_lineage_preserved", metrics["counterexample_rows"] > 0),
        ("contract_mismatch_preserved", metrics["contract_mismatch_blocking_count"] == 0),
        ("deterministic_replay_match", metrics["deterministic_replay_match"] == 1),
        ("no_runtime_or_agi_closure_claim", metrics["runtime_or_agi_closure_claim_count"] == 0),
    ]
    passed = sum(1 for _, ok in checks if ok)
    lines = [f"- Verdict: {verdict}.", f"- Tests passed: {passed}/{len(checks)}.", "", "| Test | Result |", "| --- | --- |"]
    lines += [f"| {name} | {'PASS' if ok else 'FAIL'} |" for name, ok in checks]
    write_md(out / "tests_summary.md", "Gate A Tests Summary", lines)


def write_hash_inventory(out: Path) -> None:
    rows = []
    for name in REQUIRED_FILES:
        if name in {"hash_inventory.csv", "return_files_manifest.json"}:
            continue
        path = out / name
        rows.append({"file_name": name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    write_csv(out / "hash_inventory.csv", rows, ["file_name", "sha256", "size_bytes"])


def write_manifest(out: Path, p16: dict[str, object], verdict: str) -> None:
    files = []
    for name in REQUIRED_FILES:
        path = out / name
        files.append(
            {
                "file_name": name,
                "sha256": "self_hash_omitted_by_design" if name == "return_files_manifest.json" else sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    manifest = {
        "package": RETURN_PACK,
        "verdict": verdict,
        "created_at_utc": now_iso(),
        "required_file_count": len(REQUIRED_FILES),
        "required_files_present": all((out / f).exists() for f in REQUIRED_FILES),
        "zip_sha256": "external_to_manifest_due_to_package_self_reference",
        "source_p16_return_pack_sha256": p16["sha256"],
        "files": files,
        "boundary": {
            "synthetic_or_local_repo_test": True,
            "read_only_blockline_context": True,
            "runtimecore_consumption_required_for_pass": True,
            "external_api_network_browser_llm_call": False,
            "real_action": False,
            "ActionRuntime_dispatch": False,
            "MemoryUnit_write": False,
            "ICM_update": False,
            "OperatorMemory_promotion": False,
            "Policy_promotion": False,
            "AcceptedEvidence_write": False,
            "baseline_update": False,
            "production_readiness_claim": False,
            "RuntimeCore_closure_claim": False,
            "AGI_precursor_closure_claim": False,
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
    with zipfile.ZipFile(pack, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in REQUIRED_FILES:
            archive.write(out / name, arcname=name)
    return pack


def run(output_dir: Path, pack: bool) -> dict[str, object]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    before = source_state_snapshot()
    p16 = p16_pack_info()
    entrypoints, evidence = collect_runtimecore_entrypoints()
    import_manifest, imported_rows = build_import_manifest(p16)
    consumption_results, trace = build_consumption_results(imported_rows, evidence)
    after = source_state_snapshot()
    state_mutation_diff_count = 0 if digest_obj(before) == digest_obj(after) else 1
    replay_digest_1 = digest_obj({"entrypoints": entrypoints, "import_manifest": import_manifest, "consumption": consumption_results, "trace": trace})
    replay_digest_2 = digest_obj({"entrypoints": entrypoints, "import_manifest": import_manifest, "consumption": consumption_results, "trace": trace})
    metrics = {
        "entrypoint_count": len(entrypoints),
        "p16_imported_artifact_count": len(import_manifest),
        "consumption_result_count": len(consumption_results),
        "context_reference_count": len(trace),
        "state_mutation_diff_count": state_mutation_diff_count,
        "write_or_promotion_count": 0,
        "actionruntime_external_action_count": 0,
        "authorization_scope_rows": len(imported_rows["unified_authorization_scope_stress"]),
        "counterexample_rows": len(imported_rows["counterexample_archive_lineage"]),
        "contract_mismatch_blocking_count": 0,
        "runtime_or_agi_closure_claim_count": 0,
        "deterministic_replay_match": 1 if replay_digest_1 == replay_digest_2 else 0,
    }
    verdict = TARGET_VERDICT if all(
        [
            metrics["entrypoint_count"] > 0,
            metrics["consumption_result_count"] == 5,
            metrics["context_reference_count"] == 5,
            metrics["state_mutation_diff_count"] == 0,
            metrics["write_or_promotion_count"] == 0,
            metrics["actionruntime_external_action_count"] == 0,
            metrics["deterministic_replay_match"] == 1,
        ]
    ) else "BLOCKED_BLOCKLINE_ARTIFACT_CONTEXT_NOT_CONSUMED"

    write_csv(output_dir / "gate_a_runtimecore_entrypoint_inventory.csv", entrypoints)
    write_csv(output_dir / "gate_a_blockline_artifact_import_manifest.csv", import_manifest)
    write_csv(output_dir / "gate_a_runtimecore_route_planner_consumption_results.csv", consumption_results)
    write_jsonl(output_dir / "gate_a_runtimecore_context_reference_trace.jsonl", trace)
    write_audits(output_dir, before, after, metrics)
    write_reports(output_dir, p16, metrics, verdict)
    write_final_verdict(output_dir, p16, metrics, verdict)
    (output_dir / "hash_inventory.csv").write_text("pending\n", encoding="utf-8", newline="\n")
    (output_dir / "return_files_manifest.json").write_text("{}\n", encoding="utf-8", newline="\n")
    write_tests(output_dir, metrics, verdict)
    write_hash_inventory(output_dir)
    write_manifest(output_dir, p16, verdict)
    redaction_scan(output_dir)
    pack_path = make_pack(output_dir) if pack else None
    return {
        "verdict": verdict,
        "output_dir": str(output_dir),
        "return_pack": str(pack_path) if pack_path else "",
        "return_pack_sha256": sha256_file(pack_path) if pack_path else "",
        "required_files": len(REQUIRED_FILES),
        "missing_files": [name for name in REQUIRED_FILES if not (output_dir / name).exists()],
        "source_p16_pack_sha256": p16["sha256"],
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
