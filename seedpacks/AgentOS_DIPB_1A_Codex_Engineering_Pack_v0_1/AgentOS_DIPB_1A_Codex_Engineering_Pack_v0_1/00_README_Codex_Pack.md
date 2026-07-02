# AgentOS DIPB-1A Codex Engineering Pack v0.1

## Line

AgentOS-DIPB-1: RawDomainMaterial → StructuredDomainInputPack

## Stage

DIPB-1A: Source Inventory & Provenance Capture

## Purpose

This pack is for Codex/Harness implementation. It starts a new engineering line after Arbor-NB Phase2 closure. Do not continue Phase2 and do not expand a tenth family.

DIPB-1A verifies the first practical bridge from unstructured raw domain materials to a governed future structured input pack:

```text
raw_domain_materials/
→ source_inventory.csv
→ provenance_ledger.jsonl
→ source_hash_manifest.json
→ PM-auditable return pack
```

## Non-negotiable boundary

AgentOS may build a candidate domain input pack, but AgentOS may not self-authorize that pack as valid.

DIPB-1A is inventory/provenance only. It must not extract final cases, mark eligibility, authorize replay, write accepted evidence, write baselines, write memory, promote operators, promote policies, use external APIs, or claim domain capability.

## Files in this pack

```text
00_README_Codex_Pack.md
01_PM_Next_Instruction_DIPB1A.md
02_Codex_Task_Prompt_DIPB1A.md
03_Implementation_Contract_DIPB1A.md
04_Test_Spec_DIPB1A.md
05_PM_Audit_Checklist_DIPB1A.md
06_Domain_Boundary_DIPB1A.md
schemas/domain_input_pack_schema_v0_1.json
schemas/source_inventory_schema_v0_1.json
schemas/provenance_ledger_schema_v0_1.json
templates/raw_source_inventory_template.csv
templates/human_review_gate_template.csv
templates/return_files_manifest_DIPB1A.json
return_contract/expected_return_file_list.md
fixtures/raw_domain_materials_sample/example_policy.md
fixtures/raw_domain_materials_sample/example_metrics.csv
fixtures/raw_domain_materials_sample/example_config.json
MANIFEST_Codex_Engineering_Pack_v0_1.json
```

## Expected Codex return pack name

```text
AgentOS_DIPB_1A_RawSourceInventory_ProvenanceCapture_NoExtractionNoEligibility_Return_Pack_v0_1.zip
```

## Required verdict

```text
PASS_DIPB1A_SOURCE_INVENTORY_PROVENANCE_CAPTURE_NO_EXTRACTION_NO_ELIGIBILITY
```
