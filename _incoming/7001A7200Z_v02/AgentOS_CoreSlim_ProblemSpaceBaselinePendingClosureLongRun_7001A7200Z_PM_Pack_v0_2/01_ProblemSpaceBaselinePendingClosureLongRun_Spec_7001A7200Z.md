# AgentOS CoreSlim 7001A–7200Z
## ProblemSpaceBaseline Pending Closure Long-Run Soak Spec v0.2

## Objective

Run a true long-horizon Dev-RC soak using the ProblemSpaceBaseline pending ledger as the task object.

The run must process every pending item in the baseline problem-space ledger at least once, producing a bounded closure action decision for each.

## Core workflow

\[
ProblemSpaceBaseline
\rightarrow
PendingItemInventory
\rightarrow
SRO
\rightarrow
EvidenceIndexLookup
\rightarrow
CandidateTrajectory
\rightarrow
Cbit/Risk/Drift/Reuse Audit
\rightarrow
ICMMetabolismPolicyDecision
\rightarrow
ClosureActionDecision
\rightarrow
HumanReportPacket
\]

## Required closure action per pending item

Each pending item must receive exactly one primary action:

1. ACCEPT_READY_WITH_EVIDENCE  
2. PROMOTE_TO_VALIDATION_SEED  
3. KEEP_PENDING_WITH_NEXT_TEST  
4. EXPIRE_OR_DEPRECATE  
5. QUARANTINE_NEGATIVE_TRANSFER  
6. MERGE_WITH_EXISTING  
7. ROUTE_TO_SEPARATE_EXPERIMENT  
8. HUMAN_REPORT_ONLY  

## Long-run dimensions

The run must test:

- full pending inventory coverage;
- long-horizon objective retention;
- no topic drift;
- repeated Kernel-owned NextAction decisions;
- ICMMetabolismPolicy actions;
- controlled write / no-write receipts;
- replay consistency;
- resource growth and raw budget control;
- negative route carryover;
- final human-readable closure report.

## Controlled write scope

Controlled writes are allowed only to:

- controlled local ICM sandbox store;
- MemoryUnit controlled test namespace;
- OperatorMemory controlled test namespace;
- validity_map;
- drift_watch;
- metabolism_ledger;
- negative_transfer_quarantine;
- replay_manifest;
- human report artifacts.

Forbidden:

- production ICM;
- global user memory;
- theory baseline direct writes;
- production policy writes;
- external services;
- uncontrolled filesystem mutation.

## Minimum run thresholds

The run must include:

- 100% pending inventory scan;
- at least one closure action decision per pending item;
- at least 3 full replay passes over the closure ledger;
- at least 1 rollback replay for controlled write objects;
- final Cbit accounting by pending family;
- final “active pending surface reduced / not reduced” verdict.

## Output verdicts

Allowed final verdicts:

- PASS_PENDING_CLOSURE_LONGRUN
- PASS_WITH_DEFERRED_ITEMS
- PARTIAL_PASS_BLOCKED_BY_BASELINE_INVENTORY
- FAIL_TOPIC_DRIFT
- FAIL_UNCONTROLLED_WRITE
- FAIL_REPLAY_INCONSISTENCY
