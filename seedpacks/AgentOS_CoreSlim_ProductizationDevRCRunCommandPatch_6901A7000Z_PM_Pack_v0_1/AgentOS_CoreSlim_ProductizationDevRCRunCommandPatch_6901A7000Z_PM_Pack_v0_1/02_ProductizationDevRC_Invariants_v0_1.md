# Productization Dev-RC Invariants v0.1

## Kernel / Harness / Runner boundary

- Codex is external runner / packager.
- AgentOSKernel owns policy and adjudication.
- Arbor remains one executable Harness plane component.
- LLM API is advisory only.
- Harness executes typed envelopes; it does not own ICM policy.

## Replay truthfulness

Allowed claims:

```text
standalone portable dev-RC source tree assembled
external project window preflight passed with caveats
configured project replay supported
portable artifact inspection supported
```

Forbidden claims:

```text
production release complete
unrestricted ICM write
global MemoryUnit write
global OperatorMemory write
AGI achieved
AGI Precursor 100%
full historical configured replay reproduced standalone
```

## Anti-additive constraint

This task fixes the run surface and freezes dev-RC packaging.
It must not add new capabilities or revive boundary-proof sprawl.

## Required output discipline

Return pack must include a complete file manifest and a single zip bundle containing all return files.
