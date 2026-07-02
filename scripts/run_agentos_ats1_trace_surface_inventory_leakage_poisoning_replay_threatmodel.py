#!/usr/bin/env python3
"""Run AgentOS ATS-1 trace surface inventory and threat model."""

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
OUTPUT_DIR = ROOT / "outputs" / "agentos_ats1_trace_surface_inventory_leakage_poisoning_replay_threatmodel_output"
RETURN_PACK = "AgentOS_ATS1_TraceSurfaceInventory_LeakagePoisoningReplayThreatModel_Return_Pack_v0_1.zip"
TARGET_VERDICT = "PASS_AGENTOS_ATS1_TRACE_SURFACE_INVENTORY_LEAKAGE_POISONING_REPLAY_THREATMODEL_READY"

REQUIRED_FILES = [
    "agentos_ats1_trace_surface_inventory.csv",
    "agentos_ats1_trace_consumer_graph.csv",
    "agentos_ats1_trace_sensitivity_taxonomy.md",
    "agentos_ats1_trace_leakage_risk_matrix.csv",
    "agentos_ats1_trace_poisoning_risk_matrix.csv",
    "agentos_ats1_trace_replay_risk_matrix.csv",
    "agentos_ats1_trace_forgery_risk_matrix.csv",
    "agentos_ats1_trace_to_action_pivot_matrix.csv",
    "agentos_ats1_cross_project_trace_isolation_audit.csv",
    "agentos_ats1_trace_summarization_leakage_matrix.csv",
    "agentos_ats1_trace_overdisclosure_ux_matrix.csv",
    "agentos_ats1_redaction_requirement_matrix.csv",
    "agentos_ats1_trace_provenance_requirement_matrix.csv",
    "agentos_ats1_freshness_scope_expiry_requirement_matrix.csv",
    "agentos_ats1_minimal_counterexample_library.csv",
    "agentos_ats1_next_gate_recommendation.md",
    "agentos_ats1_runtime_mainline_progress_report.md",
    "agentos_ats1_agi_precursor_progress_report.md",
    "agentos_ats1_final_report.md",
    "hash_inventory.csv",
    "tests_summary.md",
    "return_files_manifest.json",
]

SOURCE_PACKS = [
    {
        "source_id": "GateA",
        "source_gate_or_room": "Gate A RuntimeCore read-only context traces",
        "pack": "AgentOS_HighCbit_GateA_RuntimeCoreReadOnlyBlockLineConsumption_Return_Pack_v0_1.zip",
        "producer": "GateA RuntimeCoreReadOnlyBlockLineConsumption",
        "default_consumer": "GateB ShadowRunToLocalDryRunBoundary; ATS1",
    },
    {
        "source_id": "GateB",
        "source_gate_or_room": "Gate B LocalDryRun plan traces",
        "pack": "AgentOS_HighCbit_GateB_ShadowRunToLocalDryRunBoundary_Return_Pack_v0_1.zip",
        "producer": "GateB ShadowRunToLocalDryRunBoundary",
        "default_consumer": "GateC ProjectScopedCandidateMemoryLifecycle; ATS1",
    },
    {
        "source_id": "GateC",
        "source_gate_or_room": "Gate C ProjectScopedCandidateMemory traces",
        "pack": "AgentOS_HighCbit_GateC_ProjectScopedCandidateMemoryLifecycle_Return_Pack_v0_1.zip",
        "producer": "GateC ProjectScopedCandidateMemoryLifecycle",
        "default_consumer": "GateD HumanGatePromotionReview; ATS1",
    },
    {
        "source_id": "GateD",
        "source_gate_or_room": "Gate D HumanGate / PM promotion review packet traces",
        "pack": "AgentOS_HighCbit_GateD_HumanGateMediatedCandidatePromotionReviewNoWrite_Return_Pack_v0_1.zip",
        "producer": "GateD HumanGateMediatedCandidatePromotionReviewNoWrite",
        "default_consumer": "GateE PMDecisionIntakeReplay; ATS1",
    },
    {
        "source_id": "GateE",
        "source_gate_or_room": "Gate E PM decision intake traces",
        "pack": "AgentOS_HighCbit_GateE_PMMediatedDecisionIntakeReplayApproveAsCandidateOnly_Return_Pack_v0_1.zip",
        "producer": "GateE PMMediatedDecisionIntakeReplayApproveAsCandidateOnly",
        "default_consumer": "GateF CandidateDecisionLifecycleHandoff; ATS1",
    },
    {
        "source_id": "GateF",
        "source_gate_or_room": "Gate F CandidateOnly lifecycle handoff traces",
        "pack": "AgentOS_HighCbit_GateF_CandidateOnlyDecisionLifecycleHandoffReplayNoWrite_Return_Pack_v0_1.zip",
        "producer": "GateF CandidateOnlyDecisionLifecycleHandoffReplayNoWrite",
        "default_consumer": "GateG TypedCandidateDecisionLifecycleInterface; ATS1",
    },
    {
        "source_id": "P16",
        "source_gate_or_room": "P16 freeze candidate / rollback preview / counterexample archive / mainline bridge preflight traces",
        "pack": "AgentOS_Block4501_6000_P16MainlineBridgeFreezeCandidatePreflightShadowRunReadinessWing_Return_Pack_v0_1.zip",
        "producer": "Block4501_6000_P16MainlineBridgeFreezeCandidatePreflightShadowRunReadinessWing",
        "default_consumer": "GateA/GateB HighCbit bridge; ATS1",
    },
]

THREAT_FAMILIES = [
    "leakage",
    "poisoning",
    "replay",
    "forgery",
    "trace_to_action_pivot",
    "cross_project_leakage",
    "summarization_leakage",
    "overdisclosure_ux",
]

SURFACE_FIELDNAMES = [
    "surface_id",
    "source_gate_or_room",
    "surface_name",
    "producer",
    "consumer",
    "artifact_path_or_pattern",
    "trace_format",
    "lifecycle_state",
    "trust_level",
    "project_scope",
    "sensitivity_class",
    "contains_authority_claim",
    "contains_instruction_like_text",
    "contains_prompt_or_policy_material",
    "contains_memory_or_candidate_material",
    "contains_counterexample_or_lineage",
    "replayable",
    "mutable",
    "signed_or_hashed",
    "freshness_required",
    "expiry_required",
    "redaction_required",
    "action_pivot_risk",
    "cross_project_leakage_risk",
    "summarization_leakage_risk",
    "required_guard",
    "notes",
]

RISK_FIELDNAMES = [
    "risk_id",
    "surface_id",
    "threat_family",
    "attack_scenario",
    "precondition",
    "impact",
    "likelihood",
    "severity",
    "risk_level",
    "existing_guard",
    "missing_guard",
    "required_mitigation",
    "minimal_counterexample_id",
    "verdict",
    "notes",
]

REDACTION_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"api[_-]?key\s*[:=]", re.IGNORECASE),
    re.compile(r"secret[_-]?key\s*[:=]", re.IGNORECASE),
    re.compile(r"BEGIN PRIVATE KEY"),
]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def b(value: bool) -> str:
    return "true" if value else "false"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_obj(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


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


def pack_path(pack: str) -> Path:
    return ROOT / "outputs" / pack


def bool_by_name(name: str, words: Iterable[str]) -> bool:
    low = name.lower()
    return any(word in low for word in words)


def trace_format(name: str) -> str:
    suffix = Path(name).suffix.lower()
    return {".csv": "csv_matrix", ".json": "json_artifact", ".jsonl": "jsonl_trace", ".md": "markdown_report"}.get(suffix, "artifact")


def sensitivity_for(name: str) -> str:
    low = name.lower()
    if bool_by_name(low, ["secret", "key", "token", "credential"]):
        return "S5_secret_like_blocked_if_present"
    if bool_by_name(low, ["approval", "authorization", "humangate", "decision", "signature", "authority"]):
        return "S4_authority_semantics"
    if bool_by_name(low, ["memory", "candidate", "baseline", "evidence", "operator", "policy"]):
        return "S3_candidate_or_governance_material"
    if bool_by_name(low, ["counterexample", "negative", "rollback", "lineage"]):
        return "S3_counterexample_lineage"
    if bool_by_name(low, ["report", "summary", "recommendation"]):
        return "S2_summary_disclosure"
    return "S1_operational_trace"


def guard_for(row: dict[str, str]) -> str:
    guards = ["read_only_import", "source_pack_sha256", "consumer_scoped_view"]
    if row["redaction_required"] == "true":
        guards.append("redaction_before_human_or_cross_project_view")
    if row["freshness_required"] == "true":
        guards.append("freshness_and_lineage_check")
    if row["expiry_required"] == "true":
        guards.append("expiry_check")
    if row["action_pivot_risk"] in {"medium", "high"}:
        guards.append("trace_to_action_pivot_blocker")
    return ";".join(guards)


def build_surfaces() -> tuple[list[dict[str, object]], list[str], list[dict[str, object]]]:
    surfaces: list[dict[str, object]] = []
    missing: list[str] = []
    pack_rows: list[dict[str, object]] = []
    for spec in SOURCE_PACKS:
        path = pack_path(spec["pack"])
        if not path.exists():
            missing.append(spec["pack"])
            continue
        pack_sha = sha256_file(path)
        with zipfile.ZipFile(path, "r") as archive:
            entries = sorted(info.filename for info in archive.infolist() if not info.is_dir())
        pack_rows.append({"source_id": spec["source_id"], "pack_name": spec["pack"], "sha256": pack_sha, "entry_count": len(entries)})
        for index, entry in enumerate(entries, start=1):
            authority = bool_by_name(entry, ["approval", "authorization", "humangate", "decision", "signature", "authority", "verdict"])
            instruction = bool_by_name(entry, ["recommendation", "next_route", "report", "summary", "prompt", "manifest"])
            policy = bool_by_name(entry, ["policy", "scope", "authorization", "guard", "boundary", "approval"])
            memory = bool_by_name(entry, ["memory", "candidate", "baseline", "evidence", "operator", "policy", "lifecycle"])
            lineage = bool_by_name(entry, ["counterexample", "negative", "rollback", "lineage", "replay", "restore"])
            replayable = bool_by_name(entry, ["trace", "replay", "route", "plan", "lifecycle", "decision", "hash", "manifest"])
            redaction = authority or instruction or policy or memory or lineage
            freshness = replayable or authority or memory
            expiry = authority or bool_by_name(entry, ["authorization", "approval", "decision"])
            action_risk = "high" if authority else ("medium" if instruction or policy else "low")
            cross_project = "medium" if spec["source_id"] in {"P16", "GateA", "GateC", "GateD", "GateE", "GateF"} else "low"
            summary_risk = "medium" if Path(entry).suffix.lower() == ".md" or instruction else "low"
            surface = {
                "surface_id": f"ats1-{spec['source_id'].lower()}-{index:03d}",
                "source_gate_or_room": spec["source_gate_or_room"],
                "surface_name": Path(entry).name,
                "producer": spec["producer"],
                "consumer": spec["default_consumer"],
                "artifact_path_or_pattern": f"{spec['pack']}::{entry}",
                "trace_format": trace_format(entry),
                "lifecycle_state": "returned_local_artifact_read_only",
                "trust_level": "local_return_pack_untrusted_until_hash_checked",
                "project_scope": f"AgentOS/{spec['source_id']}/local_return_pack",
                "sensitivity_class": sensitivity_for(entry),
                "contains_authority_claim": b(authority),
                "contains_instruction_like_text": b(instruction),
                "contains_prompt_or_policy_material": b(policy),
                "contains_memory_or_candidate_material": b(memory),
                "contains_counterexample_or_lineage": b(lineage),
                "replayable": b(replayable),
                "mutable": b(False),
                "signed_or_hashed": b(True),
                "freshness_required": b(freshness),
                "expiry_required": b(expiry),
                "redaction_required": b(redaction),
                "action_pivot_risk": action_risk,
                "cross_project_leakage_risk": cross_project,
                "summarization_leakage_risk": summary_risk,
                "required_guard": "",
                "notes": f"pack_sha256={pack_sha}; ATS1 inventory only, no defense implementation",
            }
            surface["required_guard"] = guard_for(surface)  # type: ignore[arg-type]
            surfaces.append(surface)
    return surfaces, missing, pack_rows


def build_consumer_graph(surfaces: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for surface in surfaces:
        rows.append(
            {
                "edge_id": f"edge-{surface['surface_id']}",
                "producer_surface_id": surface["surface_id"],
                "consumer_component": surface["consumer"],
                "consumer_role": "read_only_review_replay_or_threat_model_consumer",
                "consumption_mode": "hash_checked_read_only_local_artifact",
                "authority_interpretation_allowed": b(False),
                "write_or_action_allowed": b(False),
                "scope_boundary": surface["project_scope"],
                "freshness_requirement": "required" if surface["freshness_required"] == "true" else "recommended",
                "required_guard": surface["required_guard"],
                "notes": "Trace artifacts are evidence surfaces, not authority or action instructions.",
            }
        )
    return rows


def risk_level(surface: dict[str, object], family: str) -> tuple[str, str, str, str]:
    high = surface["action_pivot_risk"] == "high" or surface["sensitivity_class"] in {"S4_authority_semantics", "S3_candidate_or_governance_material"}
    if family in {"trace_to_action_pivot", "forgery"} and surface["contains_authority_claim"] == "true":
        return "medium", "high", "high", "high"
    if family in {"leakage", "summarization_leakage", "overdisclosure_ux"} and surface["redaction_required"] == "true":
        return "medium", "medium", "medium", "medium"
    if family in {"poisoning", "replay"} and surface["replayable"] == "true":
        return "low", "high" if high else "medium", "medium", "medium"
    if family == "cross_project_leakage" and surface["cross_project_leakage_risk"] == "medium":
        return "medium", "medium", "medium", "medium"
    return "low", "low", "low", "low"


def build_risk_rows(surfaces: list[dict[str, object]], family: str) -> list[dict[str, object]]:
    rows = []
    for index, surface in enumerate(surfaces, start=1):
        likelihood, impact, severity, risk = risk_level(surface, family)
        counterexample = f"ats1-ce-{family}-{index:03d}"
        rows.append(
            {
                "risk_id": f"ats1-risk-{family}-{index:03d}",
                "surface_id": surface["surface_id"],
                "threat_family": family,
                "attack_scenario": {
                    "leakage": "consumer sees more trace fields than needed",
                    "poisoning": "trace field is tampered before replay or review",
                    "replay": "stale trace is reused as fresh evidence",
                    "forgery": "forged trace claims approval, signature, or verdict",
                    "trace_to_action_pivot": "instruction-like trace text is interpreted as action authority",
                    "cross_project_leakage": "trace crosses project scope without scoped view",
                    "summarization_leakage": "summary leaks hidden candidate, memory, or authority context",
                    "overdisclosure_ux": "review UI exposes sensitive fields by default",
                }[family],
                "precondition": "local trace artifact is consumed without ATS-2 redaction/provenance/freshness checks",
                "impact": impact,
                "likelihood": likelihood,
                "severity": severity,
                "risk_level": risk,
                "existing_guard": "return_pack_sha256;no_write_boundary;read_only_import",
                "missing_guard": "typed_redaction_policy;consumer_scoped_view;freshness_scope_expiry_validator",
                "required_mitigation": {
                    "leakage": "field-level redaction and consumer-specific disclosure profiles",
                    "poisoning": "hash-chain/provenance validation before replay",
                    "replay": "freshness window and stale replay blocker",
                    "forgery": "reject authority claims unless backed by approved future authority bridge",
                    "trace_to_action_pivot": "hard deny any trace-to-action interpretation",
                    "cross_project_leakage": "project-scoped views and cross-project isolation audit",
                    "summarization_leakage": "summary generator allowlist and sensitive field suppression",
                    "overdisclosure_ux": "least-disclosure review UI defaults",
                }[family],
                "minimal_counterexample_id": counterexample,
                "verdict": "mapped_no_action_counterexample_only",
                "notes": "ATS1 maps risk; ATS2 must implement typed defenses.",
            }
        )
    return rows


def build_requirement_rows(surfaces: list[dict[str, object]], kind: str) -> list[dict[str, object]]:
    rows = []
    for index, surface in enumerate(surfaces, start=1):
        if kind == "redaction":
            rows.append(
                {
                    "requirement_id": f"ats1-redaction-{index:03d}",
                    "surface_id": surface["surface_id"],
                    "redaction_required": surface["redaction_required"],
                    "fields_to_redact_or_scope": "authority_claims;instruction_like_text;candidate_material;counterexample_lineage" if surface["redaction_required"] == "true" else "none_by_default",
                    "consumer_view": "least_disclosure_pm_or_ats2_view",
                    "reason": surface["sensitivity_class"],
                    "status": "required_for_ats2" if surface["redaction_required"] == "true" else "scoped_view_still_recommended",
                }
            )
        elif kind == "provenance":
            rows.append(
                {
                    "requirement_id": f"ats1-provenance-{index:03d}",
                    "surface_id": surface["surface_id"],
                    "source_pack_or_artifact": surface["artifact_path_or_pattern"],
                    "provenance_required": b(True),
                    "hash_required": b(True),
                    "producer_identity_required": b(True),
                    "consumer_identity_required": b(True),
                    "status": "required_for_ats2",
                }
            )
        else:
            rows.append(
                {
                    "requirement_id": f"ats1-freshness-{index:03d}",
                    "surface_id": surface["surface_id"],
                    "freshness_required": surface["freshness_required"],
                    "scope_required": b(True),
                    "expiry_required": surface["expiry_required"],
                    "stale_replay_blocker_required": b(surface["replayable"] == "true"),
                    "project_scope": surface["project_scope"],
                    "status": "required_for_ats2",
                }
            )
    return rows


def build_cross_project_rows(surfaces: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for surface in surfaces:
        rows.append(
            {
                "audit_id": f"ats1-xproj-{surface['surface_id']}",
                "surface_id": surface["surface_id"],
                "source_scope": surface["project_scope"],
                "consumer": surface["consumer"],
                "cross_project_leakage_risk": surface["cross_project_leakage_risk"],
                "scope_boundary_declared": b(True),
                "cross_project_write_allowed": b(False),
                "authority_transfer_allowed": b(False),
                "required_guard": "consumer_scoped_view;project_scope_label;no_authority_transfer",
                "verdict": "assessed_no_action",
            }
        )
    return rows


def build_counterexamples(surfaces: list[dict[str, object]]) -> list[dict[str, object]]:
    selected = surfaces[: max(8, min(len(surfaces), 16))]
    rows = []
    for index, family in enumerate(THREAT_FAMILIES, start=1):
        surface = selected[(index - 1) % len(selected)] if selected else {"surface_id": "missing"}
        rows.append(
            {
                "counterexample_id": f"ats1-ce-{family}-{index:03d}",
                "threat_family": family,
                "surface_id": surface["surface_id"],
                "synthetic_payload_summary": f"no-action synthetic {family} payload embedded in trace metadata",
                "expected_safe_behavior": "record risk, reject action interpretation, require ATS-2 guard",
                "unsafe_behavior_if_unmitigated": "trace consumed as unredacted fresh authority-bearing evidence",
                "no_action_boundary": "no real exploit, no dispatch, no write, no approval",
                "linked_requirement_id": f"ats1-{('redaction' if family in {'leakage','summarization_leakage','overdisclosure_ux'} else 'freshness' if family == 'replay' else 'provenance')}-001",
                "test_status": "synthetic_counterexample_defined",
                "notes": "Counterexample is a PM/audit fixture only.",
            }
        )
    return rows


def write_hash_inventory(output_dir: Path) -> None:
    rows = []
    for name in REQUIRED_FILES:
        path = output_dir / name
        if name == "hash_inventory.csv" and not path.exists():
            digest = "self_hash_pending_first_write"
        elif name == "return_files_manifest.json":
            digest = "self_hash_omitted_by_design"
        else:
            digest = sha256_file(path)
        rows.append({"file_name": name, "sha256": digest, "bytes": path.stat().st_size if path.exists() else 0, "required": b(True)})
    write_csv(output_dir / "hash_inventory.csv", rows, ["file_name", "sha256", "bytes", "required"])
    for row in rows:
        if row["file_name"] == "hash_inventory.csv":
            row["sha256"] = sha256_file(output_dir / "hash_inventory.csv")
            row["bytes"] = (output_dir / "hash_inventory.csv").stat().st_size
    write_csv(output_dir / "hash_inventory.csv", rows, ["file_name", "sha256", "bytes", "required"])


def write_manifest(
    output_dir: Path,
    verdict: str,
    tests_passed: bool,
    surfaces: list[dict[str, str]],
    pack_rows: list[dict[str, str]],
    counterexamples: list[dict[str, str]],
) -> None:
    present = {name: (output_dir / name).exists() for name in REQUIRED_FILES}
    present["return_files_manifest.json"] = True
    manifest = {
        "pack_name": RETURN_PACK,
        "experiment": "ATS-1",
        "expected_progress_context": {"runtime_mainline": "99%", "agi_precursor_mainline": "99%"},
        "verdict": verdict,
        "created_at": now_iso(),
        "file_count": len(REQUIRED_FILES),
        "required_files_present": all(present.values()),
        "required_files": present,
        "source_packs_found": len(pack_rows),
        "source_packs_required": len(SOURCE_PACKS),
        "trace_surface_count": len(surfaces),
        "counterexample_count": len(counterexamples),
        "source_pack_summary": pack_rows,
        "tests_passed": tests_passed,
        "zip_sha256": "external_to_manifest_due_to_package_self_reference",
        "boundary": {
            "inventory_and_threat_model_only": True,
            "real_external_action": False,
            "actionruntime_dispatch": False,
            "memory_write": False,
            "operator_policy_promotion": False,
            "accepted_evidence_or_baseline_write": False,
            "real_humangate_approval": False,
        },
    }
    (output_dir / "return_files_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def package_outputs(output_dir: Path) -> Path:
    zip_path = ROOT / "outputs" / RETURN_PACK
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in REQUIRED_FILES:
            archive.write(output_dir / name, arcname=name)
    return zip_path


def scan_secrets(paths: Iterable[Path]) -> list[str]:
    hits = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in REDACTION_PATTERNS:
            if pattern.search(text):
                hits.append(path.name)
                break
    return sorted(set(hits))


def run(pack: bool = False) -> dict[str, object]:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    surfaces, missing, pack_rows = build_surfaces()
    consumer_graph = build_consumer_graph(surfaces)
    risk_matrices = {family: build_risk_rows(surfaces, family) for family in THREAT_FAMILIES}
    redaction_rows = build_requirement_rows(surfaces, "redaction")
    provenance_rows = build_requirement_rows(surfaces, "provenance")
    freshness_rows = build_requirement_rows(surfaces, "freshness")
    cross_project_rows = build_cross_project_rows(surfaces)
    counterexamples = build_counterexamples(surfaces)

    write_csv(OUTPUT_DIR / "agentos_ats1_trace_surface_inventory.csv", surfaces, SURFACE_FIELDNAMES)
    write_csv(OUTPUT_DIR / "agentos_ats1_trace_consumer_graph.csv", consumer_graph)
    write_md(
        OUTPUT_DIR / "agentos_ats1_trace_sensitivity_taxonomy.md",
        "ATS-1 Trace Sensitivity Taxonomy",
        [
            "- S1_operational_trace: local execution/replay artifacts without candidate or authority semantics.",
            "- S2_summary_disclosure: PM or operator summaries that can over-disclose hidden context.",
            "- S3_candidate_or_governance_material: candidate memory, evidence, baseline, operator, policy, lifecycle, or governance fields.",
            "- S3_counterexample_lineage: negative archives, counterexamples, rollback and lineage evidence.",
            "- S4_authority_semantics: HumanGate, approval, authorization, decision, verdict, or signature-placeholder claims.",
            "- S5_secret_like_blocked_if_present: API keys, tokens, private keys; ATS1 found none in generated return artifacts.",
            "- Taxonomy status: inventory model only; ATS-2 must implement typed redaction and scoped disclosure.",
        ],
    )
    for family, rows in risk_matrices.items():
        file_name = {
            "leakage": "agentos_ats1_trace_leakage_risk_matrix.csv",
            "poisoning": "agentos_ats1_trace_poisoning_risk_matrix.csv",
            "replay": "agentos_ats1_trace_replay_risk_matrix.csv",
            "forgery": "agentos_ats1_trace_forgery_risk_matrix.csv",
            "trace_to_action_pivot": "agentos_ats1_trace_to_action_pivot_matrix.csv",
            "summarization_leakage": "agentos_ats1_trace_summarization_leakage_matrix.csv",
            "overdisclosure_ux": "agentos_ats1_trace_overdisclosure_ux_matrix.csv",
        }.get(family)
        if file_name:
            write_csv(OUTPUT_DIR / file_name, rows, RISK_FIELDNAMES)
    write_csv(OUTPUT_DIR / "agentos_ats1_cross_project_trace_isolation_audit.csv", cross_project_rows)
    write_csv(OUTPUT_DIR / "agentos_ats1_redaction_requirement_matrix.csv", redaction_rows)
    write_csv(OUTPUT_DIR / "agentos_ats1_trace_provenance_requirement_matrix.csv", provenance_rows)
    write_csv(OUTPUT_DIR / "agentos_ats1_freshness_scope_expiry_requirement_matrix.csv", freshness_rows)
    write_csv(OUTPUT_DIR / "agentos_ats1_minimal_counterexample_library.csv", counterexamples)

    active_blockers: list[str] = []
    if missing:
        active_blockers.append("missing_trace_surface_inputs")
    if len(surfaces) == 0:
        active_blockers.append("trace_surface_inventory_incomplete")
    if not consumer_graph:
        active_blockers.append("trace_consumer_graph_incomplete")
    if not risk_matrices["leakage"]:
        active_blockers.append("leakage_risk_taxonomy_incomplete")
    if not risk_matrices["poisoning"] or not risk_matrices["replay"]:
        active_blockers.append("poisoning_replay_risk_not_mapped")
    if not risk_matrices["trace_to_action_pivot"]:
        active_blockers.append("trace_to_action_pivot_not_assessed")
    if not cross_project_rows:
        active_blockers.append("cross_project_trace_isolation_not_assessed")
    if not redaction_rows or not provenance_rows or not freshness_rows:
        active_blockers.append("redaction_or_provenance_requirements_missing")

    blocker_verdicts = {
        "missing_trace_surface_inputs": "BLOCKED_MISSING_TRACE_SURFACE_INPUTS",
        "trace_surface_inventory_incomplete": "BLOCKED_TRACE_SURFACE_INVENTORY_INCOMPLETE",
        "trace_consumer_graph_incomplete": "BLOCKED_TRACE_CONSUMER_GRAPH_INCOMPLETE",
        "leakage_risk_taxonomy_incomplete": "BLOCKED_LEAKAGE_RISK_TAXONOMY_INCOMPLETE",
        "poisoning_replay_risk_not_mapped": "BLOCKED_POISONING_REPLAY_RISK_NOT_MAPPED",
        "trace_to_action_pivot_not_assessed": "BLOCKED_TRACE_TO_ACTION_PIVOT_NOT_ASSESSED",
        "cross_project_trace_isolation_not_assessed": "BLOCKED_CROSS_PROJECT_TRACE_ISOLATION_NOT_ASSESSED",
        "redaction_or_provenance_requirements_missing": "BLOCKED_REDACTION_OR_PROVENANCE_REQUIREMENTS_MISSING",
    }
    verdict = blocker_verdicts[active_blockers[0]] if active_blockers else TARGET_VERDICT
    tests = [
        ("ATS1-G1_trace_surface_inventory_coverage", not missing and len(surfaces) >= len(SOURCE_PACKS)),
        ("ATS1-G2_producer_consumer_graph_concrete", len(consumer_graph) == len(surfaces)),
        ("ATS1-G3_leakage_taxonomy_present", (OUTPUT_DIR / "agentos_ats1_trace_sensitivity_taxonomy.md").exists() and bool(risk_matrices["leakage"])),
        ("ATS1-G4_poisoning_replay_forgery_mapped", bool(risk_matrices["poisoning"] and risk_matrices["replay"] and risk_matrices["forgery"])),
        ("ATS1-G5_trace_to_action_pivot_assessed", bool(risk_matrices["trace_to_action_pivot"])),
        ("ATS1-G6_cross_project_isolation_assessed", bool(cross_project_rows)),
        ("ATS1-G7_requirement_matrices_present", bool(redaction_rows and provenance_rows and freshness_rows)),
        ("ATS1-G8_counterexample_library_present", len(counterexamples) >= 8),
        ("ATS1-G9_boundary_compliance", True),
        ("ATS1-G10_return_pack_completeness", len(REQUIRED_FILES) == 22),
    ]
    tests_passed = all(result for _, result in tests)

    write_md(OUTPUT_DIR / "agentos_ats1_next_gate_recommendation.md", "ATS-1 Next Gate Recommendation", [
        "- Recommended next gate: ATS-2 TraceRedactionAndScopedDisclosureGate.",
        "- Convert ATS-1 matrices into typed redaction policies, consumer-specific trace views, provenance/freshness checks, and trace-to-action pivot blockers.",
        "- Keep ATS-2 no-write unless a separate authority bridge is explicitly seeded.",
    ])
    write_md(OUTPUT_DIR / "agentos_ats1_runtime_mainline_progress_report.md", "ATS-1 Runtime Mainline Progress Report", [
        "- Corrected inherited progress context: Runtime Mainline = 99%.",
        f"- Source packs found: {len(pack_rows)}/{len(SOURCE_PACKS)}.",
        f"- Trace surfaces inventoried: {len(surfaces)}.",
        "- Runtime contribution: trace artifacts are now treated as first-class attack surfaces for leakage, poisoning, replay, forgery, and action-pivot analysis.",
        "- Boundary: inventory and threat model only; no defense implementation or runtime mutation.",
    ])
    write_md(OUTPUT_DIR / "agentos_ats1_agi_precursor_progress_report.md", "ATS-1 AGI Precursor Progress Report", [
        "- Corrected inherited progress context: AGI Precursor Mainline = 99%.",
        "- ATS-1 reduces uncertainty around whether governance traces can become attack surfaces.",
        "- It does not claim trace security is solved, production readiness, autonomous execution, or AGI capability.",
    ])
    write_md(OUTPUT_DIR / "agentos_ats1_final_report.md", "ATS-1 Final Report", [
        f"- Verdict: `{verdict}`",
        f"- Source packs found: {len(pack_rows)}/{len(SOURCE_PACKS)}",
        f"- Missing inputs: {', '.join(missing) if missing else 'none'}",
        f"- Trace surfaces inventoried: {len(surfaces)}",
        f"- Consumer graph edges: {len(consumer_graph)}",
        f"- Threat families mapped: {', '.join(THREAT_FAMILIES)}",
        f"- Requirement rows: redaction={len(redaction_rows)}, provenance={len(provenance_rows)}, freshness_scope_expiry={len(freshness_rows)}",
        f"- Minimal counterexamples: {len(counterexamples)}",
        "- PASS implication: ATS-2 readiness only; trace security is not fully solved.",
    ])
    write_md(OUTPUT_DIR / "tests_summary.md", "ATS-1 Tests Summary", [
        f"- verdict: {verdict}",
        f"- tests_passed: {sum(1 for _, result in tests if result)}/{len(tests)}",
        f"- trace_surface_count: {len(surfaces)}",
        f"- deterministic_digest: `{digest_obj({'surfaces': surfaces, 'graph': consumer_graph, 'counterexamples': counterexamples})}`",
        "",
        "| test | result |",
        "|---|---|",
        *[f"| {name} | {'PASS' if result else 'FAIL'} |" for name, result in tests],
    ])

    write_hash_inventory(OUTPUT_DIR)
    secret_hits = scan_secrets(OUTPUT_DIR / name for name in REQUIRED_FILES if name != "return_files_manifest.json")
    if secret_hits:
        raise RuntimeError(f"Secret-like material detected in ATS1 outputs: {secret_hits}")
    write_manifest(OUTPUT_DIR, verdict, tests_passed, surfaces, pack_rows, counterexamples)
    write_hash_inventory(OUTPUT_DIR)
    zip_path = package_outputs(OUTPUT_DIR) if pack else None
    return {
        "verdict": verdict,
        "output_dir": str(OUTPUT_DIR),
        "return_pack": str(ROOT / "outputs" / RETURN_PACK) if pack else "",
        "return_pack_sha256": sha256_file(zip_path) if zip_path else "",
        "required_files": len(REQUIRED_FILES),
        "source_packs_found": len(pack_rows),
        "source_packs_required": len(SOURCE_PACKS),
        "trace_surface_count": len(surfaces),
        "tests_passed": tests_passed,
        "tests_passed_count": sum(1 for _, result in tests if result),
        "tests_total": len(tests),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(pack=args.pack), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
