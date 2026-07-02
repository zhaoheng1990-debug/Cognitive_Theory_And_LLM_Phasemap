# Codex Task Prompt: 7301A–7400Z Autonomous ICM Evolution Policy

You are executing AgentOS CoreSlim Autonomous ICM Evolution Policy.

## Objective

Close the self-evolution gap before release-readiness review.

AgentOS must not depend on human preauthorization for every ordinary project-scoped ICM / MemoryUnit / OperatorMemory evolution write.

If a research line closes to ACCEPT with replayable evidence, AgentOSKernel must be able to autonomously write project-scoped durable ICM records under policy.

## Implement / demonstrate

1. Define AutonomousICMEvolutionPolicy.
2. Define ProjectScopedDurableWriteEnvelope.
3. Define ProjectScopedDurableWriteReceipt.
4. Add policy decisions:
   - NO_WRITE_KEEP_CANDIDATE
   - AUTONOMOUS_PROJECT_MEMORYUNIT_WRITE
   - AUTONOMOUS_PROJECT_OPERATORMEMORY_WRITE
   - AUTONOMOUS_POLICY_PRIOR_WRITE
   - AUTONOMOUS_VALIDITY_UPDATE
   - AUTONOMOUS_SCOPE_NARROW
   - AUTONOMOUS_QUARANTINE
   - REQUEST_HUMAN_SCOPE_ESCALATION
   - BLOCK_WRITE_INSUFFICIENT_EVIDENCE
   - ROLLBACK_PREVIOUS_WRITE
5. Demonstrate UPS ACCEPT → OperatorMemory autonomous project-scoped durable write.
6. Demonstrate evidence-weak user-positive candidate remains no-write.
7. Demonstrate negative-transfer candidate is quarantined.
8. Demonstrate rollback and replay.
9. Confirm Kernel owns final decision.
10. Confirm LLM API advisory only.
11. Confirm Harness / CodexToolBridge only execute envelopes.

## Hard boundaries

- No global production ICM write.
- No global user memory write.
- No theory baseline direct write.
- No external mutation.
- No high-risk action.
- No human preauthorization required for ordinary project-scoped durable ICM evolution.
- Human review is posthoc, rollback, and scope escalation.

## Required return files

Package all files listed in `08_Return_Files_Manifest_7301A7400Z.json`.
