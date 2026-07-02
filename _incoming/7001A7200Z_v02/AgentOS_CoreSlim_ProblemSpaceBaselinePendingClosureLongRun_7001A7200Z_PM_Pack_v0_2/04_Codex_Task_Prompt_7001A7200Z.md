# Codex Task Prompt: 7001A–7200Z

You are executing AgentOS CoreSlim ProductizationLongRun Dev-RC Soak v0.2.

## Task

Run a true long-horizon soak using the ProblemSpaceBaseline pending ledger as the task object.

Do not run a low-Cbit generic replay loop only.  
The core object is the baseline pending problem-space ledger.

## Required workflow

1. Locate/import the latest ProblemSpaceBaseline if available.
2. Build a PendingItemInventory.
3. For every pending item:
   - run SRO / constraint-field interpretation;
   - identify evidence references or missing evidence;
   - produce Cbit/risk/drift/reuse assessment;
   - produce ICMMetabolismPolicyDecision;
   - produce one ClosureActionDecision.
4. Generate controlled write/no-write receipts in local sandbox only.
5. Run replay passes over the ledger.
6. Run at least one rollback replay for controlled write artifacts.
7. Generate final human report.

## Allowed closure actions

- ACCEPT_READY_WITH_EVIDENCE
- PROMOTE_TO_VALIDATION_SEED
- KEEP_PENDING_WITH_NEXT_TEST
- EXPIRE_OR_DEPRECATE
- QUARANTINE_NEGATIVE_TRANSFER
- MERGE_WITH_EXISTING
- ROUTE_TO_SEPARATE_EXPERIMENT
- HUMAN_REPORT_ONLY

## Return files

Package all files listed in `06_Return_Files_Manifest_7001A7200Z_v0_2.json`.

Do not omit reports.  
Do not replace reports with a single summary.  
Return a zip pack.
