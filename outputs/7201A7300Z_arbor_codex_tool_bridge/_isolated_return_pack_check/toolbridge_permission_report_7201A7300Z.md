# ToolBridge Permission Report 7201A7300Z

Verdict: PASS_PERMISSION_MODEL_BOUNDARY

## Tier 0 Read-only

- READ_ARTIFACT
- INSPECT_MANIFEST
- GENERATE_HASH_INVENTORY

## Tier 1 Local Reversible Write

- WRITE_LOCAL_PATCH
- PACKAGE_ZIP
- ROLLBACK_LOCAL_WRITE

Requires rollback metadata and allowed workspace paths.

## Tier 2 Local Execution

- RUN_PYTEST
- RUN_SCRIPT
- RUN_REPLAY

Requires bounded timeout and captured stdout/stderr.

## Tier 3 Forbidden

- SEND_EMAIL
- WIRE_TRANSFER
- EXTERNAL_API_MUTATION
- GIT_PUSH
- PRODUCTION_DEPLOY
- GLOBAL_MEMORY_WRITE
- GLOBAL_ICM_WRITE
- UNBOUNDED_WEB_ACTION
- LEGAL_SIGNATURE
- INVESTMENT_COMMITMENT

All path-bearing calls must remain under `allowed_paths`; source theory baseline, global memory, production ICM, external service state, and unscoped user directories remain forbidden.
