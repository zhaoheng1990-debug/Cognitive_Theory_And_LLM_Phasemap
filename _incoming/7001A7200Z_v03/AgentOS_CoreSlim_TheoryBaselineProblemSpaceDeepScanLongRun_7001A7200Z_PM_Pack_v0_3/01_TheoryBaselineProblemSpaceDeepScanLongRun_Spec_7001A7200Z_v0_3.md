# AgentOS CoreSlim 7001A–7200Z v0.3
## TheoryBaseline ProblemSpace DeepScan Long-Run Spec

## Objective

Run the release-preflight long-run against the real theory baseline problem space.

Primary source path:

```text
C:\Users\ZH\Desktop\AGI\理论基线
```

The run must not rely only on the prior 3-item CoreSlim mini-ledger.

## Core pipeline

```text
LocalTheoryBaselineDeepScan
→ SourceInventory
→ ProblemSpaceExtraction
→ PendingClusterMap
→ SRO per cluster
→ Evidence / Cbit / Drift / Risk audit
→ ICMMetabolismPolicyDecision
→ ClosureActionDecision
→ Controlled write/no-write receipt
→ Replay / rollback
→ HumanProblemSpaceClosureReport
```

## Required scan behavior

Recursively scan the theory baseline directory for:

```text
*.md
*.txt
*.json
*.jsonl
*.csv
*.yaml
*.yml
```

Optionally include index files and README files from adjacent baseline folders if they are referenced by the primary baseline documents.

If the path is unavailable, the run must return:

```text
BLOCKED_LOCAL_THEORY_BASELINE_PATH_MISSING
```

and must not silently fall back to the 3-item mini-ledger.

## Extraction targets

Extract unresolved or pending objects including:

```text
PENDING
待验证
未闭合
缺口
future experiment
next test
candidate
hypothesis
TODO
needs validation
open question
remaining gap
problem space
not accepted
deferred
requires replay
requires live validation
```

Also extract implicit unresolved objects from sections named:

```text
Problem Space
Remaining Gaps
Pending Baseline
Future Work
Open Questions
未闭合问题
待验证假说
未来实验
```

## Minimum expected output

The run must produce:

1. SourceInventory with file count and hash list.
2. ProblemSpaceCandidateInventory.
3. PendingClusterMap.
4. Deduplication / merge report.
5. Coverage report explaining why each included/excluded file was treated that way.
6. One ClosureActionDecision per discovered pending cluster.
7. ICMMetabolismPolicyDecision per cluster.
8. Replay and rollback report.
9. Human summary.

## Closure actions

Allowed actions:

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

## Critical rule

A cluster may be accepted only if evidence exists and scope is bounded.

The default should be:

```text
PROMOTE_TO_VALIDATION_SEED
KEEP_PENDING_WITH_NEXT_TEST
MERGE_WITH_EXISTING
EXPIRE_OR_DEPRECATE
```

not automatic ACCEPT.
