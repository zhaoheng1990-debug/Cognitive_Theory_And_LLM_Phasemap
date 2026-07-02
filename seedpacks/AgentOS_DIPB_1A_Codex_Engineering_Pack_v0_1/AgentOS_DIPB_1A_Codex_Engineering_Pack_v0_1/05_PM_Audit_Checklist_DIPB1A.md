# PM Audit Checklist: DIPB-1A Return Pack

Use this checklist when the Codex return pack is received.

## Manifest audit

- [ ] Return pack name matches required name.
- [ ] All required files are present.
- [ ] `return_files_manifest.json` is internally consistent.
- [ ] Return pack is inspectable without internet access.

## Source coverage audit

- [ ] Every file under `raw_domain_materials/` appears exactly once in `source_inventory.csv`.
- [ ] Every source has stable `source_id`.
- [ ] Every source has SHA256.
- [ ] `source_hash_manifest.json` agrees with inventory.

## Provenance audit

- [ ] `provenance_ledger.jsonl` has one valid record per source.
- [ ] Missing metadata is explicitly reported.
- [ ] Unknown fields are not fabricated.
- [ ] Authorization defaults are safe.

## Boundary audit

- [ ] No case extraction.
- [ ] No eligible case count.
- [ ] No replay-ready decision.
- [ ] No AcceptedEvidence write.
- [ ] No AcceptedBaseline write.
- [ ] No MemoryUnit write.
- [ ] No Operator promotion.
- [ ] No Policy promotion.
- [ ] No domain capability claim.
- [ ] No external API / web / online LLM fact source.

## Verdict audit

PASS only if the return pack states:

```text
PASS_DIPB1A_SOURCE_INVENTORY_PROVENANCE_CAPTURE_NO_EXTRACTION_NO_ELIGIBILITY
```

and no hard fail condition is present.

If any hard fail is found, PM verdict must be FAIL or BLOCKED and the next seed should repair DIPB-1A, not advance to DIPB-1B.
