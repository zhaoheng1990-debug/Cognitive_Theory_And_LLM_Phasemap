# Expected Return File List: DIPB-1A

Codex must return a single zip named:

```text
AgentOS_DIPB_1A_RawSourceInventory_ProvenanceCapture_NoExtractionNoEligibility_Return_Pack_v0_1.zip
```

The zip must include:

```text
README.md
dipb1a_summary.md
source_inventory.csv
provenance_ledger.jsonl
source_hash_manifest.json
source_type_summary.md
missing_source_metadata_report.md
no_authorization_no_replay_audit.md
forbidden_action_audit.md
isolated_replay_report.md
tests_summary.md
return_files_manifest.json
command_replay_log.md
```

If Codex adds implementation scripts or test scripts for isolated replay, list them under an `extra_replay_files` field in `return_files_manifest.json` and include them in the pack.
