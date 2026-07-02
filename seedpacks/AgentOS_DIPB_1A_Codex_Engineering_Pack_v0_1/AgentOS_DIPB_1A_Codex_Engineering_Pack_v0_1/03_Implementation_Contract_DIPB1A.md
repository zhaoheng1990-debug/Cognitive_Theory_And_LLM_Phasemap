# Implementation Contract: DIPB-1A

## Recommended repository additions

Codex may choose exact file names, but the implementation should be easy to locate and replay. Recommended structure:

```text
tools/dipb/build_source_inventory.py
tests/test_dipb1a_source_inventory.py
outputs/dipb1a_source_inventory_output/
raw_domain_materials/
```

## Script behavior

The implementation script should:

1. Resolve repository root.
2. Locate `raw_domain_materials/` relative to repository root.
3. Refuse to use web or external APIs.
4. Recursively enumerate files only, excluding directories.
5. Sort files by normalized POSIX-style relative path.
6. Assign deterministic IDs as `src_000001`, `src_000002`, etc.
7. Compute SHA256 using local bytes.
8. Infer only file extension and coarse source type from extension.
9. Write inventory and ledger outputs.
10. Write audits and summaries.
11. Write command replay log.
12. Create return manifest.

## Source type mapping

Recommended coarse mapping:

```text
.md,.txt,.pdf,.docx,.log → document
.csv,.xlsx → table
.json,.jsonl,.yaml,.yml → structured_data
.py,.ipynb → code_or_notebook
otherwise → unsupported
```

## Parse status mapping

```text
supported inventory metadata only → parsed_metadata_only
unsupported extension → unsupported_inventory_only
metadata extraction failure → parse_error_inventory_only
not parsed by design → not_parsed_inventory_only
```

For DIPB-1A, parsing must be metadata-only. Do not extract final candidate cases.

## Provenance ledger JSONL fields

Each line must be valid JSON and include:

```json
{
  "source_id": "src_000001",
  "source_path": "raw_domain_materials/example.md",
  "sha256": "...",
  "provenance_status": "captured_inventory_only",
  "provider": "unknown",
  "provider_role": "unknown",
  "authorization_status": "authorization_unknown",
  "allowed_use": "inventory_only",
  "forbidden_use": [
    "accepted_evidence",
    "accepted_baseline",
    "memory_write",
    "operator_promotion",
    "policy_promotion",
    "capability_claim",
    "replay_ready_decision",
    "case_extraction"
  ],
  "notes": "metadata only; missing provenance must be reviewed by PM/HumanGate"
}
```

## Missing metadata policy

Do not fabricate:

```text
provided_by
provider_role
authorization_status beyond safe default
provenance_note
source origin
human review decision
```

If unavailable, write `unknown` or `authorization_unknown`, then list it in `missing_source_metadata_report.md`.

## Empty input behavior

If `raw_domain_materials/` is missing or empty, do not pass. Return a BLOCKED verdict and still create a minimal output folder explaining the blocker.

Suggested verdict:

```text
BLOCKED_DIPB1A_NO_RAW_DOMAIN_MATERIALS
```

## Replayability

The return pack must allow PM to inspect:

```text
which files were scanned
which hashes were generated
which metadata was missing
which forbidden actions were checked
which command was run
which tests passed
```

No return artifact may require internet access to interpret.
