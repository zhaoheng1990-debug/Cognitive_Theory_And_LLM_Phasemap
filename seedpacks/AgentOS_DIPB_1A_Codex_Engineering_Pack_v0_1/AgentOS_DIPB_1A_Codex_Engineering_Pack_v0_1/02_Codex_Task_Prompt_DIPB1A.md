You are working in the Logos AgentOS repository.

Implement:

```text
AgentOS_DIPB_1A_RawSourceInventory_ProvenanceCapture_NoExtractionNoEligibility
```

## Context

- Arbor-NB Phase2 is closed/frozen.
- Do not extend Phase2.
- Do not add a tenth family.
- Phase2 proved governance of already-structured domain packs.
- DIPB-1A starts a new engineering line: governed construction of domain input packs from raw domain materials.
- DIPB-1A only inventories raw sources and captures provenance.
- DIPB-1A must not extract final domain cases or mark replay eligibility.

## Core task

Given a local folder:

```text
raw_domain_materials/
```

generate source inventory and provenance artifacts under:

```text
outputs/dipb1a_source_inventory_output/
```

## Required output files

Create all of the following under `outputs/dipb1a_source_inventory_output/`:

1. `source_inventory.csv`
2. `provenance_ledger.jsonl`
3. `source_hash_manifest.json`
4. `source_type_summary.md`
5. `missing_source_metadata_report.md`
6. `no_authorization_no_replay_audit.md`
7. `forbidden_action_audit.md`
8. `isolated_replay_report.md`
9. `dipb1a_summary.md`
10. `tests_summary.md`
11. `return_files_manifest.json`
12. `command_replay_log.md`
13. `README.md`

## Source inventory required columns

```text
source_id
source_path
source_type
file_extension
sha256
size_bytes
last_modified_utc
provided_by
provider_role
authorization_status
provenance_note
parse_status
inventory_only
eligible_for_case_extraction
notes
```

## Supported file types for metadata inventory

```text
.md
.txt
.csv
.json
.jsonl
.yaml
.yml
.py
.ipynb
.pdf
.docx
.xlsx
.log
```

Unsupported files are still inventoried, hashed, and represented. Mark them as `unsupported_inventory_only`; do not parse content.

## Stable source_id rule

Use deterministic stable IDs based on sorted normalized relative paths, for example:

```text
src_000001
src_000002
...
```

The same folder contents and path ordering must reproduce the same IDs.

## Authorization rule

For DIPB-1A, default every source to:

```text
authorization_status=authorization_unknown
inventory_only=true
eligible_for_case_extraction=false
```

Do not infer authorization from filename or content.

## Allowed

- Local file scan under `raw_domain_materials/`
- Local hashing
- Local metadata extraction
- Deterministic source ID generation
- Local tests
- Local fixture generation only if clearly marked as test fixture

## Forbidden

- Web browsing
- External API
- Online LLM API
- Online LLM as fact source
- Source search
- Benchmark / leaderboard
- Final case extraction
- Candidate case extraction as final artifact
- Eligible case count
- Replay-ready decision
- AcceptedEvidence write
- AcceptedBaseline write
- MemoryUnit write
- Operator promotion
- Policy promotion
- Domain capability claim
- HumanGate bypass

## Tests required

At minimum, implement and report tests for:

1. All raw files represented in `source_inventory.csv`.
2. Every source has SHA256.
3. Stable deterministic `source_id` under repeated runs.
4. One provenance JSONL record per source.
5. `eligible_for_case_extraction=false` for all sources in DIPB-1A.
6. No accepted evidence/baseline/memory/operator/policy write artifacts exist.
7. Unsupported files are inventoried as `unsupported_inventory_only`.
8. Return manifest contains every required file.
9. Return pack can be replayed in isolation.

## Required verdict

If all criteria pass, write this exact verdict in `dipb1a_summary.md` and `return_files_manifest.json`:

```text
PASS_DIPB1A_SOURCE_INVENTORY_PROVENANCE_CAPTURE_NO_EXTRACTION_NO_ELIGIBILITY
```

If not, write a `BLOCKED_DIPB1A_*` or `FAIL_DIPB1A_*` verdict and list blockers.

## Required return pack

Create and return:

```text
AgentOS_DIPB_1A_RawSourceInventory_ProvenanceCapture_NoExtractionNoEligibility_Return_Pack_v0_1.zip
```

The zip must include exactly the required return files, plus any local implementation scripts/tests needed for isolated replay if they are not already in the repository.

## Complete return file list

- `README.md`
- `dipb1a_summary.md`
- `source_inventory.csv`
- `provenance_ledger.jsonl`
- `source_hash_manifest.json`
- `source_type_summary.md`
- `missing_source_metadata_report.md`
- `no_authorization_no_replay_audit.md`
- `forbidden_action_audit.md`
- `isolated_replay_report.md`
- `tests_summary.md`
- `return_files_manifest.json`
- `command_replay_log.md`

## Final reminder

DIPB-1A proves only source inventory and provenance capture. It does not prove domain capability, case eligibility, replay readiness, or production readiness.
