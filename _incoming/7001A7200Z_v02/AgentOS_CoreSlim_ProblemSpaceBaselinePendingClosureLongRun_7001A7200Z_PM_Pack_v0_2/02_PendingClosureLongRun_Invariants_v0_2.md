# Pending Closure Long-Run Invariants v0.2

## Kernel ownership

AgentOSKernel owns:

- SRO;
- ProblemSpaceLedger interpretation;
- ICMMetabolismPolicyDecision;
- ClosureActionDecision;
- NextActionPolicy;
- HumanReportPacket.

LLM API may advise.  
Harness may execute.  
Codex may run/package.  
Arbor may execute delegated tasks.

But:

\[
\boxed{
LLM/API/Harness/Codex/Arbor\ must\ not\ own\ final\ closure\ decisions.
}
\]

## Closure is not automatic ACCEPT

A pending item is not ACCEPT just because it appears useful.

A pending item can only become ACCEPT_READY_WITH_EVIDENCE if:

- evidence references exist;
- scope is bounded;
- negative transfer risk is low;
- future Cbit gain is positive;
- replay trail exists;
- no conflicting unresolved caveat blocks it.

## Anti-additive invariant

The active pending surface should shrink or be clarified.

The run fails anti-additive if it creates more pending objects than it resolves without a clear Cbit gain.

## Replay truthfulness

Must distinguish:

- configured_project_replay;
- portable_artifact_replay;
- standalone_dev_rc_source_tree;
- production_release=false.

## Controlled write invariant

Controlled write does not imply production write.

Allowed wording:

\[
\boxed{
controlled local ICM metabolism write evidence
}
\]

Forbidden wording:

\[
\boxed{
global production ICM write complete
}
\]
