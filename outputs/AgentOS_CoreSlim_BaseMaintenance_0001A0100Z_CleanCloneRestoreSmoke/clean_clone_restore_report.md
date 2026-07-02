# Clean Clone Restore Smoke Report

Line: $line

## Verdict

PASS_CLEAN_CLONE_RESTORE_SMOKE

## Source Inputs

- Seed: $seed
- Handoff pointer: $handoffPointer
- Infrastructure pointer read first: $infraPointer
- Clone-base pointer read first: $clonePointer
- Clean clone base ZIP: $backupZip

## Backup Verification

- Expected SHA256 from seed/pointer: $expectedBackupHash
- Actual SHA256: $actualBackupHash
- Match: True

Note: the restored backup's internal `metadata/backup_manifest.json` contains older/finalization-sidecar ZIP hash fields that do not match the current ZIP. The authoritative seed/pointer hash for this restore is the matched value above.

## Restore Shape

- New maintenance workspace: $root
- Mutable workspace materialized at workspace root: yes
- Original extracted backup preserved under: $(Join-Path D:\Logos_AgentOS_base_maintenance '_restore_source')
- Theory baseline snapshot restored into runtime: no

## Workspace Hash Compare

- Expected workspace files: 395
- Verified workspace files: 395
- Missing: 0
- Mismatched: 0

## Smoke Test

Command:

`powershell
pytest -q agentos_core_slim_v0\tests --basetemp=D:\Logos_AgentOS_base_maintenance\_tmp_pytest_smoke
`

Result: `16 passed in 0.08s`

The first pytest attempt used the default temp root and failed during fixture setup with `PermissionError` on `C:\Users\ZH\AppData\Local\Temp\pytest-of-ZH`. The workspace-local `--basetemp` rerun passed.

## Boundaries

- Continued research-AgentOS self-evolution line: no
- Imported 8001A-8200Z research outputs: no
- Official theory baseline write: no
- Production/global registry write: no
- External API/network/browser/connector calls: no
