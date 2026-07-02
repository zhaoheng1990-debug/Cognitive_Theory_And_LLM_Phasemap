# PM Next Instruction: DIPB-1A

## Objective

Implement the governed first stage of the Domain Input Pack Builder:

```text
RawDomainMaterial → SourceInventory + ProvenanceLedger
```

Given a local repository folder:

```text
raw_domain_materials/
```

Codex/Harness must generate deterministic, replayable source inventory outputs under:

```text
outputs/dipb1a_source_inventory_output/
```

## Stage boundary

This stage is strictly inventory/provenance capture. It does not authorize extraction, eligibility, replay readiness, accepted evidence, baseline writes, memory writes, operator promotion, policy promotion, benchmark, leaderboard, or any domain capability claim.

## Required implementation behavior

1. Recursively scan `raw_domain_materials/`.
2. Include every file under that folder in `source_inventory.csv`.
3. Generate stable deterministic `source_id` values.
4. Compute SHA256 for every source.
5. Capture local file metadata: path, extension, size, last modified UTC.
6. Record missing provenance metadata explicitly instead of inferring it.
7. Write one provenance ledger JSONL record per source.
8. Set `inventory_only=true` for every row.
9. Set `eligible_for_case_extraction=false` for every row unless a local explicit PM authorization file exists and is parsed by a future stage. For DIPB-1A default must be false.
10. Run local tests and produce a return pack that is replayable in isolation.

## Required PM verdict condition

Return `PASS_DIPB1A_SOURCE_INVENTORY_PROVENANCE_CAPTURE_NO_EXTRACTION_NO_ELIGIBILITY` only if every acceptance criterion passes.

Return `BLOCKED_DIPB1A_*` or `FAIL_DIPB1A_*` with explicit blockers if any hard fail is encountered.
