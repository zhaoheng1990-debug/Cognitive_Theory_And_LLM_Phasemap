# LongRun Test Matrix v0.2

| Test | Purpose | Minimum |
|---|---|---:|
| Pending inventory scan | Ensure all pending items are listed | 100% |
| SRO per pending item | Ensure each item gets constraint-field interpretation | 100% |
| ClosureActionDecision | Ensure one action per item | 100% |
| Cbit accounting | Estimate value / cost / risk | 100% |
| ICMMetabolismPolicyDecision | Decide retain/expire/quarantine/merge/promote-candidate | 100% |
| Controlled write/no-write receipt | Verify policy action materializes | 100% |
| Negative route carryover | Ensure bad routes preserved | all negative items |
| Replay pass | Verify deterministic replay | >= 3 |
| Rollback replay | Verify controlled write rollback | >= 1 |
| HumanReportPacket | Summarize for PM/user | 1 |
| Raw budget | Prevent artifact bloat | PASS |
| Forbidden claims | Prevent overclaim | PASS |
