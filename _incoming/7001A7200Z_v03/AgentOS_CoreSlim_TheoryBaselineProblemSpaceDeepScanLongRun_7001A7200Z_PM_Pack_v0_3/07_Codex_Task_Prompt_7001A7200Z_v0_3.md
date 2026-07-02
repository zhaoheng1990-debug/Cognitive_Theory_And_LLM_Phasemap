# Codex Task Prompt 7001A–7200Z v0.3

You are executing AgentOS CoreSlim TheoryBaselineProblemSpaceDeepScanLongRun.

## Critical correction

The prior 7001 v0.2 run processed only 3 known CoreSlim pending lines. That is insufficient.

You must first rescan the user's local theory baseline directory:

```text
C:\Users\ZH\Desktop\AGI\理论基线
```

Do not silently fall back to the old 3-item mini-ledger.

## Task

1. Recursively scan the local theory baseline directory.
2. Build source inventory and hash inventory.
3. Extract all pending / unresolved / future-test / remaining-gap objects.
4. Deduplicate into pending clusters.
5. Run SRO per cluster.
6. Produce Cbit/risk/drift/reuse audit per cluster.
7. Produce ICMMetabolismPolicyDecision per cluster.
8. Produce ClosureActionDecision per cluster.
9. Generate controlled write/no-write receipts in local sandbox only.
10. Run at least 3 deterministic replay passes.
11. Run rollback replay if controlled writes are created.
12. Produce final human report.

## Allowed closure actions

```text
ACCEPT_READY_WITH_EVIDENCE
PROMOTE_TO_VALIDATION_SEED
KEEP_PENDING_WITH_NEXT_TEST
EXPIRE_OR_DEPRECATE
QUARANTINE_NEGATIVE_TRANSFER
MERGE_WITH_EXISTING
ROUTE_TO_SEPARATE_EXPERIMENT
HUMAN_REPORT_ONLY
```

## Hard boundaries

- AgentOSKernel owns final closure decisions.
- LLM API advisory only.
- Harness executes only typed envelopes.
- No direct theory baseline write.
- No production ICM write.
- No global MemoryUnit / OperatorMemory write.
- No AGI 100% claim.
- No “all pending accepted” claim.

## Return

Package all files listed in `06_Return_Files_Manifest_7001A7200Z_v0_3.json`.
