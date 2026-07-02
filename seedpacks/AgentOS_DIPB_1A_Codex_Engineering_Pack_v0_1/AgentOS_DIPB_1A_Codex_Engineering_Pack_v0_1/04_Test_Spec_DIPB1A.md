# Test Spec: DIPB-1A

## Required tests

Codex should implement local tests and summarize results in `tests_summary.md`.

### T1: all raw files represented

Enumerate raw files under `raw_domain_materials/` and compare with `source_inventory.csv` paths.

PASS if every raw file appears exactly once.

### T2: SHA256 existence and validity

Every inventory row must include a 64-character SHA256 hex digest.

Optional stronger check: recompute and compare.

### T3: deterministic source IDs

Run inventory twice on unchanged inputs. Source IDs and paths must be identical.

### T4: provenance ledger completeness

`provenance_ledger.jsonl` must contain one valid JSON object per source and each object must include `source_id`, `source_path`, `sha256`, `authorization_status`, `allowed_use`, and `forbidden_use`.

### T5: eligibility forbidden by default

All inventory rows must have:

```text
inventory_only=true
eligible_for_case_extraction=false
```

### T6: forbidden writes absent

No files or records may indicate:

```text
AcceptedEvidence
AcceptedBaseline
MemoryUnit write
Operator promotion
Policy promotion
domain capability claim
replay ready
final case extraction
```

### T7: unsupported file safety

Unsupported files must still be inventoried and hashed, with `parse_status=unsupported_inventory_only`.

### T8: manifest complete

`return_files_manifest.json` must list every required return file and indicate existence.

### T9: isolated replay

A fresh local run with the same `raw_domain_materials/` should reproduce inventory path ordering, IDs, and hash manifest.

## Summary format

`tests_summary.md` should contain:

```text
verdict: PASS or FAIL/BLOCKED
number_of_tests
passed
failed
blocked
per_test_results
blockers
```
