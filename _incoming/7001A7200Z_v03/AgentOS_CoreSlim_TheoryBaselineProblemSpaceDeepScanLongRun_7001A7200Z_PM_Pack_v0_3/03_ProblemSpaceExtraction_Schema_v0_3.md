# ProblemSpace Extraction Schema v0.3

Each extracted item must be represented as:

```json
{
  "problem_id": "...",
  "source_file": "...",
  "source_section": "...",
  "source_line_or_anchor": "...",
  "title": "...",
  "family": "...",
  "status": "PENDING | CANDIDATE | DEFERRED | VALIDATION_READY | GAP | UNKNOWN",
  "claim_or_question": "...",
  "why_unclosed": "...",
  "evidence_refs": [],
  "missing_evidence": [],
  "suggested_next_test": "...",
  "risk_if_ignored": "...",
  "risk_if_promoted_too_early": "...",
  "related_items": [],
  "dedupe_key": "..."
}
```

## Required clustering dimensions

Cluster by:

```text
Retention / Memory / ICM
Operator / UPS / Policy
SRO / Routing / StructuralResolution
HPM / Hallucination / ConstraintAlignment
PhaseMap / LLM Dynamics
SEM / MemoryUnit / Retrieval
ASA / Control
DSTA / Direction Spectrum
SOB / Meaning / Subject-Object
Cbit / Cognitive Ontology
FMS / Margin / Selection
AgentOS / Runtime / Harness
Education / DomainAgentOS / Applied Lines
Other
```

## Deduplication

If multiple files refer to the same unresolved object, merge into one cluster and preserve all source refs.

Do not double-count repeated summaries.

## Required output files

```text
problemspace_candidate_inventory_7001A7200Z.jsonl
pending_cluster_map_7001A7200Z.csv
problemspace_deduplication_report_7001A7200Z.md
problemspace_family_coverage_report_7001A7200Z.md
```
