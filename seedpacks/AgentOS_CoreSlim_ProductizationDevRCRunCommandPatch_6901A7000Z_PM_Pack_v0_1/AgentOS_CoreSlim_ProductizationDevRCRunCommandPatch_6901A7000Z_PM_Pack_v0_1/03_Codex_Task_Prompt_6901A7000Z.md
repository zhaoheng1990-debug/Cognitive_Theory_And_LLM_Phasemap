# Codex Task Prompt — 6901A-7000Z

You are Codex acting as external engineering runner / packager, not AgentOS Kernel.

## Task

Apply a minimal productization run-command patch to the AgentOS CoreSlim standalone portable dev-RC source tree and freeze the dev-RC package.

Use the prior 6801 standalone source tree and the external project window preflight result as input context.

## Must do

1. Patch README / install-run guide so the documented smoke command runs from a clean extracted source bundle.
2. Prefer either:
   - `PYTHONPATH=. python examples/run_portable_smoke.py`; or
   - a robust module entrypoint such as `python -m agentos_core_slim_portable.smoke`.
3. Rebuild the standalone source bundle zip.
4. Extract the rebuilt zip into a clean temp directory.
5. Run:
   - `python -m pytest tests -q`
   - the documented smoke command exactly as written.
6. Confirm output includes:
   - `kernel_owner = AgentOSKernel.ICMMetabolismPolicy`
   - `advisory_only = True`
   - `harness_no_external_effect = True`
   - `replay_verified = True`
   - `production_release = False`
7. Produce productization dev-RC freeze reports.
8. Package all return files into one zip.

## Must not do

- Do not add new AgentOS cognitive capability.
- Do not add new Harness type.
- Do not add new AgentProfile.
- Do not perform production ICM write.
- Do not perform external API / network calls.
- Do not claim production release.
- Do not claim AGI or AGI Precursor 100%.

## Required return files

Return these files exactly:

1. `run_command_patch_report_6901A7000Z.md`
2. `standalone_clean_extract_pytest_report_6901A7000Z.md`
3. `standalone_clean_extract_smoke_report_6901A7000Z.md`
4. `productization_dev_rc_freeze_report_6901A7000Z.md`
5. `external_preflight_integration_report_6901A7000Z.md`
6. `replay_truthfulness_dev_rc_matrix_6901A7000Z.md`
7. `kernel_harness_boundary_freeze_report_6901A7000Z.md`
8. `known_caveats_forbidden_claims_6901A7000Z.md`
9. `portable_dev_rc_source_bundle_6901A7000Z.zip`
10. `runtime_source_tree_inventory_6901A7000Z.csv`
11. `hash_inventory_6901A7000Z.csv`
12. `pack_hygiene_report_6901A7000Z.md`
13. `raw_budget_report_6901A7000Z.md`
14. `bug_report_if_any_6901A7000Z.md`
15. `pm_ready_final_verdict_6901A7000Z.md`
16. `return_files_manifest_6901A7000Z.json`
17. `AgentOS_CoreSlim_ProductizationDevRCRunCommandPatch_6901A7000Z_Return_Pack_v0_1.zip`
