# Autonomous ICM Evolution Policy Report 7301A7400Z

Verdict: POLICY_DEFINED

`AutonomousICMEvolutionPolicy` is Kernel-owned and converts ACCEPT-level, replayable, project-bounded candidates into project-scoped durable ICM evolution writes.

Policy decisions supported:

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

Allowed durable root: `project_scoped_icm_store/`.

Forbidden: global production ICM write, global user memory write, theory baseline direct write, external mutation, high-risk action, LLM-owned decision, Harness-owned decision.

Implementation: `agentos_core_slim_v0/agentos_kernel/autonomous_icm_evolution.py`.
