# Kernel Ownership Report 7301A7400Z

Verdict: PASS_KERNEL_OWNS_FINAL_EVOLUTION_DECISION

Only `AgentOSKernel.ICMEvolutionPolicy` can issue final autonomous durable write decisions. The write envelope requires `authorized_by = AgentOSKernel.ICMEvolutionPolicy` and `target_scope = project_scoped_durable`.

LLM API may advise. Harness and CodexToolBridge may execute an envelope. Neither may decide retention, OperatorMemory, ICM, rollback, quarantine, or scope escalation.
