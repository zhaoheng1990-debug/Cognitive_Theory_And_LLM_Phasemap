# Run Command Patch Report 6901A-7000Z

Verdict: `PASS_PRODUCTIZATION_DEV_RC_RUN_COMMAND_PATCH_AND_FREEZE`

## Patch Applied

- Updated standalone README default smoke command to `python -m agentos_core_slim_portable.smoke`.
- Updated `INSTALL_RUN.md` to use the same clean-extract module entrypoint.
- Added minimal module entrypoint `agentos_core_slim_portable.smoke`.
- Kept existing example script available; default documented smoke path no longer requires caller-managed `PYTHONPATH`.
- Hardened local receipt file naming against deep Windows extraction paths by using stable short SHA-256 receipt filenames while preserving full receipt IDs inside receipt payloads.
- Reworked portable tests to use package-local `.pytest_tmp` scratch space so `python -m pytest tests -q` does not depend on user-level temp directory permissions.

## Documented Commands Verified

```powershell
python -m pytest tests -q
python -m agentos_core_slim_portable.smoke
```

## Scope Boundary

This was a productization run-command / packaging patch only. No new cognitive capability, Harness type, AgentProfile, production write, external API, network call, or AGI/production claim was added.
