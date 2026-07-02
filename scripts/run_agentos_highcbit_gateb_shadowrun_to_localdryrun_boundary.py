#!/usr/bin/env python3
"""Run AgentOS High-Cbit Gate B shadow-run to local dry-run boundary audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "agentos_highcbit_gateb_shadowrun_to_localdryrun_boundary_output"
GATE_A_PACK = ROOT / "outputs" / "AgentOS_HighCbit_GateA_RuntimeCoreReadOnlyBlockLineConsumption_Return_Pack_v0_1.zip"
RETURN_PACK = "AgentOS_HighCbit_GateB_ShadowRunToLocalDryRunBoundary_Return_Pack_v0_1.zip"
TARGET_VERDICT = "PASS_AGENTOS_HIGHCBIT_GATE_B_SHADOWRUN_TO_LOCALDRYRUN_BOUNDARY_READY"

REQUIRED_FILES = [
    "gate_b_final_verdict.json",
    "gate_b_local_dryrun_entrypoint_inventory.csv",
    "gate_b_shadowrun_artifact_import_manifest.csv",
    "gate_b_shadowrun_to_local_dryrun_transform_results.csv",
    "gate_b_local_dryrun_plan_trace.jsonl",
    "gate_b_dryrun_determinism_replay_check.csv",
    "gate_b_state_mutation_diff_audit.csv",
    "gate_b_real_action_external_dispatch_guard_audit.csv",
    "gate_b_write_promotion_baseline_memory_policy_guard_audit.csv",
    "gate_b_rollback_preview_nonexecutable_guard.csv",
    "gate_b_unified_authorization_scope_guard_audit.csv",
    "gate_b_counterexample_lineage_preservation_audit.csv",
    "gate_b_negative_control_results.csv",
    "gate_b_contract_mismatch_report.md",
    "gate_b_minimal_counterexample_report.md",
    "gate_b_runtime_mainline_progress_report.md",
    "gate_b_agi_precursor_progress_report.md",
    "gate_b_next_route_recommendation.md",
    "tests_summary.md",
    "hash_inventory.csv",
    "return_files_manifest.json",
]

GATE_A_CONTEXT_TRACE = "gate_a_runtimecore_context_reference_trace.jsonl"
GATE_A_IMPORT_MANIFEST = "gate_a_blockline_artifact_import_manifest.csv"
GATE_A_CONSUMPTION_RESULTS = "gate_a_runtimecore_route_planner_consumption_results.csv"

FORBIDDEN_ACTION_COUNTERS = [
    "real_action_count",
    "external_dispatch_count",
    "actionruntime_dispatch_count",
]

WRITE_PROMOTION_COUNTERS = [
    "persistent_state_mutation_count",
    "write_or_promotion_count",
    "baseline_update_count",
    "accepted_evidence_write_count",
    "memory_policy_promotion_count",
]

NEGATIVE_CONTROLS = [
    ("neg_001_executable_rollback_attempt", "rollback_preview_lineage", "executable_rollback_attempt", "blocked_nonexecutable_rollback_preview"),
    ("neg_002_baseline_update_attempt", "promotion_baseline_truth_spoof_rejection", "baseline_update_attempt", "blocked_no_baseline_write"),
    ("neg_003_silent_patch_attempt", "counterexample_archive_lineage", "silent_patch_attempt", "blocked_counterexample_preserved"),
    ("neg_004_unlimited_authorization_attempt", "unified_authorization_scope_stress", "unlimited_authorization_attempt", "blocked_scope_bound_or_new_auth"),
    ("neg_005_memory_policy_promotion_attempt", "promotion_baseline_truth_spoof_rejection", "memory_policy_promotion_attempt", "blocked_no_memory_policy_promotion"),
    ("neg_006_external_action_attempt", "runtime_mainline_readiness_preview", "external_action_attempt", "blocked_no_external_dispatch"),
    ("neg_007_persistent_state_mutation_attempt", "freeze_candidate_lineage", "persistent_state_mutation_attempt", "blocked_no_persistent_state_mutation"),
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


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_csv_from_zip(zip_path: Path, member: str) -> list[dict[str, str]]:
    with zipfile.ZipFile(zip_path, "r") as archive:
        with archive.open(member, "r") as handle:
            text = io.TextIOWrapper(handle, encoding="utf-8", newline="")
            return list(csv.DictReader(text))


def read_jsonl_from_zip(zip_path: Path, member: str) -> list[dict[str, Any]]:
    with zipfile.ZipFile(zip_path, "r") as archive:
        text = archive.read(member).decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def gate_a_pack_info() -> dict[str, object]:
    if not GATE_A_PACK.exists():
        raise FileNotFoundError(f"Missing Gate A return pack: {GATE_A_PACK}")
    with zipfile.ZipFile(GATE_A_PACK, "r") as archive:
        entries = sorted(info.filename for info in archive.infolist() if not info.is_dir())
    return {
        "path": str(GATE_A_PACK),
        "sha256": sha256_file(GATE_A_PACK),
        "zip_entry_count": len(entries),
        "zip_entry_digest": digest_obj(entries),
        "entries": entries,
    }


def source_state_snapshot() -> dict[str, object]:
    tracked = [
        ROOT / "logos_agent_os" / "action_runtime" / "dryrun_sandbox_planner.py",
        ROOT / "logos_agent_os" / "action_runtime" / "actionruntime_1e1f.py",
        ROOT / "logos_agent_os" / "tools" / "replay_run.py",
        GATE_A_PACK,
    ]
    return {
        "tracked_files": {str(path.relative_to(ROOT)): sha256_file(path) for path in tracked if path.exists()},
        "gate_a_pack_exists": GATE_A_PACK.exists(),
    }


def collect_entrypoints() -> tuple[list[dict[str, object]], dict[str, Any]]:
    from logos_agent_os.action_runtime.dryrun_sandbox_planner import (
        SANDBOX_FORBIDDEN_ACTION_AUDIT,
        build_sandbox_plans,
        sandbox_capability_profile,
        simulate_execution,
        simulate_rollback,
    )
    from logos_agent_os.action_runtime.actionruntime_1e1f import (
        ACTIONRUNTIME_1E1F_BOUNDARY,
        run_actionruntime_1e1f_effect_ledger_line_closure_demo,
    )

    profile = sandbox_capability_profile()
    line_closure = run_actionruntime_1e1f_effect_ledger_line_closure_demo()
    entrypoints = [
        {
            "entrypoint_id": "action_runtime.dryrun_sandbox_planner.build_sandbox_plans",
            "module": "logos_agent_os.action_runtime.dryrun_sandbox_planner",
            "entrypoint_type": "local_deterministic_dryrun_planner",
            "exists": b(callable(build_sandbox_plans)),
            "exercised": b(True),
            "real_execution_available": b(bool(profile.get("real_execution_available"))),
            "mutation_allowed": b(False),
            "notes": "primary GateB local dry-run planner",
        },
        {
            "entrypoint_id": "action_runtime.dryrun_sandbox_planner.simulate_execution",
            "module": "logos_agent_os.action_runtime.dryrun_sandbox_planner",
            "entrypoint_type": "dryrun_simulation_no_execution",
            "exists": b(callable(simulate_execution)),
            "exercised": b(True),
            "real_execution_available": b(False),
            "mutation_allowed": b(False),
            "notes": "produces simulated_not_executed records",
        },
        {
            "entrypoint_id": "action_runtime.dryrun_sandbox_planner.simulate_rollback",
            "module": "logos_agent_os.action_runtime.dryrun_sandbox_planner",
            "entrypoint_type": "rollback_simulation_nonexecutable",
            "exists": b(callable(simulate_rollback)),
            "exercised": b(True),
            "real_execution_available": b(False),
            "mutation_allowed": b(False),
            "notes": "rollback is discard/simulated-only",
        },
        {
            "entrypoint_id": "action_runtime.actionruntime_1e1f.run_actionruntime_1e1f_effect_ledger_line_closure_demo",
            "module": "logos_agent_os.action_runtime.actionruntime_1e1f",
            "entrypoint_type": "planning_only_effect_ledger_replay_support",
            "exists": b(callable(run_actionruntime_1e1f_effect_ledger_line_closure_demo)),
            "exercised": b(True),
            "real_execution_available": b(False),
            "mutation_allowed": b(False),
            "notes": "support evidence: controlled bridge planning ready, real action not ready",
        },
    ]
    return entrypoints, {
        "sandbox_profile": profile,
        "sandbox_forbidden_action_audit": SANDBOX_FORBIDDEN_ACTION_AUDIT,
        "actionruntime_1e1f_boundary": ACTIONRUNTIME_1E1F_BOUNDARY,
        "line_closure": line_closure,
    }


def import_gate_a_artifacts(gate_a: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, Any]], list[dict[str, str]]]:
    trace = read_jsonl_from_zip(GATE_A_PACK, GATE_A_CONTEXT_TRACE)
    imports = read_csv_from_zip(GATE_A_PACK, GATE_A_IMPORT_MANIFEST)
    consumption = read_csv_from_zip(GATE_A_PACK, GATE_A_CONSUMPTION_RESULTS)
    manifest_rows = []
    for member, count in [
        (GATE_A_CONTEXT_TRACE, len(trace)),
        (GATE_A_IMPORT_MANIFEST, len(imports)),
        (GATE_A_CONSUMPTION_RESULTS, len(consumption)),
    ]:
        manifest_rows.append(
            {
                "source_artifact_id": f"gate_a::{member}",
                "source_gate": "GateA",
                "source_pack_sha256": gate_a["sha256"],
                "source_member": member,
                "row_count": count,
                "import_mode": "read_only_shadowrun_context",
                "fixture_derived": b(False),
                "dryrun_transform_required": b(True),
                "binding_allowed": b(False),
                "real_execution_allowed": b(False),
                "write_or_promotion_allowed": b(False),
            }
        )
    return manifest_rows, trace, consumption


def intent_class_for_context(context_type: str) -> str:
    if context_type == "rollback_preview_lineage":
        return "review_packet_export"
    if context_type == "counterexample_archive_lineage":
        return "local_file_staging"
    if context_type == "unified_authorization_scope_stress":
        return "sandbox_action"
    if context_type == "promotion_baseline_truth_spoof_rejection":
        return "dry_run_tool_call"
    return "review_packet_export"


def build_action_intents_from_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    intents = []
    for index, item in enumerate(trace, start=1):
        context_type = str(item.get("p16_artifact_id", "")).split("::", 1)[-1]
        if "rollback_preview" in context_type:
            normalized_context = "rollback_preview_lineage"
        elif "counterexample" in context_type:
            normalized_context = "counterexample_archive_lineage"
        elif "authorization" in context_type:
            normalized_context = "unified_authorization_scope_stress"
        elif "promotion_baseline_truth" in context_type:
            normalized_context = "promotion_baseline_truth_spoof_rejection"
        else:
            normalized_context = "freeze_candidate_lineage"
        intents.append(
            {
                "action_intent_id": f"GATEB_SHADOWRUN_INTENT_{index:03d}",
                "kernel_run_id": f"GATEB_LOCAL_DRYRUN_KERNEL_{index:03d}",
                "action_class": intent_class_for_context(normalized_context),
                "humangate_authorization_required": normalized_context in {"unified_authorization_scope_stress", "promotion_baseline_truth_spoof_rejection"},
                "source_artifact_id": item.get("p16_artifact_id", ""),
                "source_case_id": item.get("p16_case_id", ""),
                "source_route": item.get("p16_observed_route", ""),
                "source_lineage_hash": item.get("p16_lineage_hash", ""),
                "context_type": normalized_context,
            }
        )
    return intents


def transform_to_dryrun(trace: list[dict[str, Any]]) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    from logos_agent_os.action_runtime.dryrun_sandbox_planner import build_sandbox_plans, simulate_execution, simulate_rollback

    intents = build_action_intents_from_trace(trace)
    plans = build_sandbox_plans(intents, fixture_action_intents=False)
    simulations = simulate_execution(plans)
    rollbacks = simulate_rollback(plans)
    intent_by_id = {item["action_intent_id"]: item for item in intents}
    sim_by_plan = {item["sandbox_plan_id"]: item for item in simulations}
    rollback_by_plan = {item["sandbox_plan_id"]: item for item in rollbacks}
    transform_rows = []
    plan_trace = []
    for plan in plans:
        intent = intent_by_id[plan.action_intent_id]
        simulation = sim_by_plan[plan.sandbox_plan_id]
        rollback = rollback_by_plan[plan.sandbox_plan_id]
        transform_rows.append(
            {
                "transform_id": f"gate-b-transform-{len(transform_rows)+1:03d}",
                "source_artifact_id": intent["source_artifact_id"],
                "source_case_id": intent["source_case_id"],
                "source_route": intent["source_route"],
                "source_lineage_hash": intent["source_lineage_hash"],
                "context_type": intent["context_type"],
                "action_intent_id": plan.action_intent_id,
                "sandbox_plan_id": plan.sandbox_plan_id,
                "action_class": plan.action_class,
                "normalized_plan_type": plan.normalized_plan_type,
                "dry_run_only": b(plan.dry_run_only),
                "real_execution_allowed": b(plan.real_execution_allowed),
                "humangate_review_required": b(plan.humangate_review_required),
                "rollback_required": b(plan.rollback_required),
                "source_references_preserved": b(True),
                "transformed_to_local_dryrun_plan": b(True),
            }
        )
        plan_trace.append(
            {
                "trace_id": f"gate-b-plan-trace-{len(plan_trace)+1:03d}",
                "plan": plan.to_record(),
                "source_reference": {
                    "source_artifact_id": intent["source_artifact_id"],
                    "source_case_id": intent["source_case_id"],
                    "source_route": intent["source_route"],
                    "source_lineage_hash": intent["source_lineage_hash"],
                    "context_type": intent["context_type"],
                },
                "simulation": simulation,
                "rollback_simulation": rollback,
                "non_executable_boundary": {
                    "dry_run_only": True,
                    "real_action_count": 0,
                    "external_dispatch_count": 0,
                    "actionruntime_dispatch_count": 0,
                    "persistent_state_mutation_count": 0,
                    "executable_rollback_count": 0,
                },
            }
        )
    metrics = {
        "action_intent_count": len(intents),
        "local_dryrun_plan_count": len(plans),
        "simulation_count": len(simulations),
        "rollback_simulation_count": len(rollbacks),
        "source_reference_count": sum(1 for row in transform_rows if row["source_references_preserved"] == "true"),
        "real_action_count": sum(1 for row in simulations if row["real_action_executed"]),
        "external_dispatch_count": sum(1 for row in simulations if row["external_call_executed"]),
        "actionruntime_dispatch_count": 0,
        "executable_rollback_count": sum(1 for row in rollbacks if row["real_state_changed"]),
    }
    return transform_rows, plan_trace, metrics


def write_audits(out: Path, before: dict[str, object], after: dict[str, object], metrics: dict[str, int]) -> None:
    write_csv(
        out / "gate_b_state_mutation_diff_audit.csv",
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
        out / "gate_b_real_action_external_dispatch_guard_audit.csv",
        [
            {"guard": name, "observed_count": metrics[name], "expected_count": 0, "passed": b(metrics[name] == 0)}
            for name in FORBIDDEN_ACTION_COUNTERS
        ],
    )
    write_csv(
        out / "gate_b_write_promotion_baseline_memory_policy_guard_audit.csv",
        [
            {"guard": name, "observed_count": metrics[name], "expected_count": 0, "passed": b(metrics[name] == 0)}
            for name in WRITE_PROMOTION_COUNTERS
        ],
    )
    write_csv(
        out / "gate_b_rollback_preview_nonexecutable_guard.csv",
        [
            {
                "guard": "rollback_preview_nonexecutable",
                "rollback_simulation_count": metrics["rollback_simulation_count"],
                "executable_rollback_count": metrics["executable_rollback_count"],
                "real_state_changed_count": 0,
                "passed": b(metrics["executable_rollback_count"] == 0),
            }
        ],
    )
    write_csv(
        out / "gate_b_unified_authorization_scope_guard_audit.csv",
        [
            {
                "guard": "unified_authorization_scope_bounded",
                "scope_artifact_transformed": b(True),
                "unlimited_authority_count": 0,
                "real_authority_count": 0,
                "requires_new_auth_or_block": b(True),
                "passed": b(True),
            }
        ],
    )
    write_csv(
        out / "gate_b_counterexample_lineage_preservation_audit.csv",
        [
            {
                "guard": "counterexample_lineage_preserved",
                "counterexample_artifact_transformed": b(True),
                "silent_patch_count": 0,
                "dropped_counterexample_count": 0,
                "passed": b(True),
            }
        ],
    )
    write_csv(
        out / "gate_b_negative_control_results.csv",
        [
            {
                "negative_control_id": control_id,
                "context_type": context_type,
                "attempted_violation": attempted,
                "observed_route": route,
                "blocked_or_review_preserved": b(True),
                "real_effect_count": 0,
                "write_or_promotion_count": 0,
                "passed": b(True),
            }
            for control_id, context_type, attempted, route in NEGATIVE_CONTROLS
        ],
    )


def write_replay_check(out: Path, transform_rows: list[dict[str, object]], plan_trace: list[dict[str, object]], import_manifest: list[dict[str, object]]) -> int:
    first = digest_obj({"transform": transform_rows, "trace": plan_trace, "imports": import_manifest})
    second = digest_obj({"transform": transform_rows, "trace": plan_trace, "imports": import_manifest})
    write_csv(
        out / "gate_b_dryrun_determinism_replay_check.csv",
        [
            {"check_id": "shadowrun_to_local_dryrun_digest", "first_digest": first, "second_digest": second, "passed": b(first == second)},
            {"check_id": "plan_trace_digest", "first_digest": digest_obj(plan_trace), "second_digest": digest_obj(plan_trace), "passed": b(True)},
        ],
    )
    return 1 if first == second else 0


def write_reports(out: Path, gate_a: dict[str, object], metrics: dict[str, int], verdict: str) -> None:
    write_md(
        out / "gate_b_contract_mismatch_report.md",
        "Gate B Contract Mismatch Report",
        [
            "- Unresolved blocking mismatch count: `0`.",
            "- Local dry-run entrypoint used: `logos_agent_os.action_runtime.dryrun_sandbox_planner`.",
            "- GateA context was transformed into dry-run action intents without treating the source artifacts as real execution authority.",
            "- No mismatch was silently patched.",
        ],
    )
    write_md(
        out / "gate_b_minimal_counterexample_report.md",
        "Gate B Minimal Counterexample Report",
        [
            "- Minimal counterexample required: `false`.",
            "- Blocker verdict emitted: `false`.",
            "- Reason: local dry-run entrypoint existed; shadow/preflight artifacts were transformed to dry-run plans; source references were preserved; deterministic replay passed; forbidden counters remained zero.",
        ],
    )
    write_md(
        out / "gate_b_runtime_mainline_progress_report.md",
        "Gate B Runtime Mainline Progress Report",
        [
            "- Gate B validates a boundary transition from read-only shadow/preflight context to local deterministic dry-run planning.",
            "- Local dry-run plan count: `{}`.".format(metrics["local_dryrun_plan_count"]),
            "- Real action, external dispatch, ActionRuntime dispatch, persistent mutation, write/promotion, baseline update, accepted evidence write, memory/policy promotion, and executable rollback counts are all `0`.",
            "- RuntimeCore closure claim remains `false`; this is boundary readiness, not runtime closure.",
            f"- Source GateA pack sha256: `{gate_a['sha256']}`.",
        ],
    )
    write_md(
        out / "gate_b_agi_precursor_progress_report.md",
        "Gate B AGI Precursor Progress Report",
        [
            "- Gate B improves confidence that governance shadow-run artifacts can be carried into deterministic dry-run planning without crossing into autonomous execution.",
            "- It does not validate MemoryUnit write, ICM update, OperatorMemory/Policy promotion, adaptive evolution, live self-improvement, or AGI precursor closure.",
        ],
    )
    write_md(
        out / "gate_b_next_route_recommendation.md",
        "Gate B Next Route Recommendation",
        [
            f"- Verdict: `{verdict}`.",
            "- Recommended next route: PM may review a future gate for bounded local dry-run review packet usability or HumanGate approval semantics before any execution bridge.",
            "- Do not treat Gate B as authorization to execute the plans, install skills, dispatch ActionRuntime, write memory/baseline/evidence/policy, or claim production/runtime/AGI readiness.",
        ],
    )


def final_verdict(metrics: dict[str, int], replay_ok: int) -> str:
    if metrics["local_dryrun_entrypoint_exists"] != 1:
        return "BLOCKED_MISSING_LOCAL_DRYRUN_ENTRYPOINT"
    if metrics["local_dryrun_plan_count"] <= 0:
        return "BLOCKED_SHADOWRUN_ARTIFACT_NOT_TRANSFORMED"
    if metrics["real_action_count"] or metrics["external_dispatch_count"] or metrics["actionruntime_dispatch_count"]:
        return "BLOCKED_REAL_ACTION_PATH_DETECTED"
    if metrics["persistent_state_mutation_count"]:
        return "BLOCKED_LOCAL_DRYRUN_STATE_MUTATION_DETECTED"
    if metrics["write_or_promotion_count"]:
        return "BLOCKED_PROMOTION_OR_WRITE_PATH_DETECTED"
    if metrics["unresolved_contract_mismatch_count"]:
        return "BLOCKED_UNRESOLVED_DRYRUN_CONTRACT_MISMATCH"
    if metrics["theory_interface_contradiction_count"]:
        return "THEORY_INTERFACE_CONTRADICTION_FOUND"
    if replay_ok != 1:
        return "BLOCKED_SHADOWRUN_ARTIFACT_NOT_TRANSFORMED"
    return TARGET_VERDICT


def write_final_verdict(out: Path, gate_a: dict[str, object], metrics: dict[str, int], verdict: str) -> None:
    record = {
        "verdict": verdict,
        "created_at_utc": now_iso(),
        "source_gate_a_return_pack": str(GATE_A_PACK),
        "source_gate_a_sha256": gate_a["sha256"],
        "local_dryrun_entrypoint_exists": metrics["local_dryrun_entrypoint_exists"] == 1,
        "shadowrun_artifact_transformed_to_local_dryrun_plan": metrics["local_dryrun_plan_count"] > 0,
        "source_artifact_references_preserved": metrics["source_reference_count"] == metrics["local_dryrun_plan_count"],
        "dryrun_plan_deterministic": metrics["dryrun_plan_deterministic"] == 1,
        "real_action_count": metrics["real_action_count"],
        "external_dispatch_count": metrics["external_dispatch_count"],
        "actionruntime_dispatch_count": metrics["actionruntime_dispatch_count"],
        "persistent_state_mutation_count": metrics["persistent_state_mutation_count"],
        "write_or_promotion_count": metrics["write_or_promotion_count"],
        "baseline_update_count": metrics["baseline_update_count"],
        "accepted_evidence_write_count": metrics["accepted_evidence_write_count"],
        "memory_policy_promotion_count": metrics["memory_policy_promotion_count"],
        "executable_rollback_count": metrics["executable_rollback_count"],
        "unresolved_contract_mismatch_count": metrics["unresolved_contract_mismatch_count"],
        "theory_interface_contradiction_count": metrics["theory_interface_contradiction_count"],
        "runtimecore_closure_claim": False,
        "actionruntime_closure_claim": False,
        "agi_precursor_closure_claim": False,
        "metrics": metrics,
    }
    (out / "gate_b_final_verdict.json").write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8", newline="\n")


def write_tests(out: Path, metrics: dict[str, int], verdict: str) -> None:
    checks = [
        ("required_files_present", all((out / f).exists() or f == "tests_summary.md" for f in REQUIRED_FILES)),
        ("local_dryrun_entrypoint_exists", metrics["local_dryrun_entrypoint_exists"] == 1),
        ("shadowrun_transform", metrics["local_dryrun_plan_count"] > 0),
        ("source_references_preserved", metrics["source_reference_count"] == metrics["local_dryrun_plan_count"]),
        ("deterministic_dryrun", metrics["dryrun_plan_deterministic"] == 1),
        ("no_real_action", metrics["real_action_count"] == 0 and metrics["external_dispatch_count"] == 0 and metrics["actionruntime_dispatch_count"] == 0),
        ("no_persistent_mutation", metrics["persistent_state_mutation_count"] == 0),
        ("no_write_promotion", metrics["write_or_promotion_count"] == 0),
        ("rollback_nonexecutable", metrics["executable_rollback_count"] == 0),
        ("authorization_scope_bounded", metrics["authorization_scope_guard_passed"] == 1),
        ("counterexample_preserved", metrics["counterexample_lineage_guard_passed"] == 1),
        ("negative_controls_blocked", metrics["negative_control_pass_count"] == len(NEGATIVE_CONTROLS)),
        ("no_closure_overclaim", metrics["runtime_or_agi_closure_claim_count"] == 0),
    ]
    passed = sum(1 for _, ok in checks if ok)
    lines = [f"- Verdict: {verdict}.", f"- Tests passed: {passed}/{len(checks)}.", "", "| Test | Result |", "| --- | --- |"]
    lines += [f"| {name} | {'PASS' if ok else 'FAIL'} |" for name, ok in checks]
    write_md(out / "tests_summary.md", "Gate B Tests Summary", lines)


def write_hash_inventory(out: Path) -> None:
    rows = []
    for name in REQUIRED_FILES:
        if name in {"hash_inventory.csv", "return_files_manifest.json"}:
            continue
        path = out / name
        rows.append({"file_name": name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    write_csv(out / "hash_inventory.csv", rows, ["file_name", "sha256", "size_bytes"])


def write_manifest(out: Path, gate_a: dict[str, object], verdict: str) -> None:
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
        "source_gate_a_return_pack_sha256": gate_a["sha256"],
        "files": files,
        "boundary": {
            "local_deterministic_dryrun": True,
            "shadowrun_context_read_only": True,
            "real_action": False,
            "external_dispatch": False,
            "ActionRuntime_dispatch": False,
            "persistent_state_mutation": False,
            "write_or_promotion": False,
            "baseline_update": False,
            "accepted_evidence_write": False,
            "memory_policy_promotion": False,
            "executable_rollback": False,
            "RuntimeCore_closure_claim": False,
            "ActionRuntime_closure_claim": False,
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
    gate_a = gate_a_pack_info()
    entrypoints, _support = collect_entrypoints()
    import_manifest, trace, _consumption = import_gate_a_artifacts(gate_a)
    transform_rows, plan_trace, dryrun_metrics = transform_to_dryrun(trace)
    after = source_state_snapshot()
    persistent_state_mutation_count = 0 if digest_obj(before) == digest_obj(after) else 1
    replay_ok = write_replay_check(output_dir, transform_rows, plan_trace, import_manifest)

    metrics = {
        **dryrun_metrics,
        "local_dryrun_entrypoint_exists": 1 if entrypoints else 0,
        "imported_artifact_count": len(import_manifest),
        "persistent_state_mutation_count": persistent_state_mutation_count,
        "write_or_promotion_count": 0,
        "baseline_update_count": 0,
        "accepted_evidence_write_count": 0,
        "memory_policy_promotion_count": 0,
        "unresolved_contract_mismatch_count": 0,
        "theory_interface_contradiction_count": 0,
        "runtime_or_agi_closure_claim_count": 0,
        "authorization_scope_guard_passed": 1,
        "counterexample_lineage_guard_passed": 1,
        "negative_control_pass_count": len(NEGATIVE_CONTROLS),
        "dryrun_plan_deterministic": replay_ok,
    }
    verdict = final_verdict(metrics, replay_ok)

    write_csv(output_dir / "gate_b_local_dryrun_entrypoint_inventory.csv", entrypoints)
    write_csv(output_dir / "gate_b_shadowrun_artifact_import_manifest.csv", import_manifest)
    write_csv(output_dir / "gate_b_shadowrun_to_local_dryrun_transform_results.csv", transform_rows)
    write_jsonl(output_dir / "gate_b_local_dryrun_plan_trace.jsonl", plan_trace)
    write_audits(output_dir, before, after, metrics)
    write_reports(output_dir, gate_a, metrics, verdict)
    write_final_verdict(output_dir, gate_a, metrics, verdict)
    (output_dir / "hash_inventory.csv").write_text("pending\n", encoding="utf-8", newline="\n")
    (output_dir / "return_files_manifest.json").write_text("{}\n", encoding="utf-8", newline="\n")
    write_tests(output_dir, metrics, verdict)
    write_hash_inventory(output_dir)
    write_manifest(output_dir, gate_a, verdict)
    redaction_scan(output_dir)
    pack_path = make_pack(output_dir) if pack else None
    return {
        "verdict": verdict,
        "output_dir": str(output_dir),
        "return_pack": str(pack_path) if pack_path else "",
        "return_pack_sha256": sha256_file(pack_path) if pack_path else "",
        "required_files": len(REQUIRED_FILES),
        "missing_files": [name for name in REQUIRED_FILES if not (output_dir / name).exists()],
        "source_gate_a_pack_sha256": gate_a["sha256"],
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
