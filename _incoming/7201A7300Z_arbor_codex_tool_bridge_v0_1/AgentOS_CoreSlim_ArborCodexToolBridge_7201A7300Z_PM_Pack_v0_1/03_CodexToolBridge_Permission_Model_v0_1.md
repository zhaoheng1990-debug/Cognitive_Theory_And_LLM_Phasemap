# CodexToolBridge Permission Model v0.1

## Permission tiers

### Tier 0: read-only

```text
READ_ARTIFACT
INSPECT_MANIFEST
GENERATE_HASH_INVENTORY
```

### Tier 1: local reversible write

```text
WRITE_LOCAL_PATCH
PACKAGE_ZIP
ROLLBACK_LOCAL_WRITE
```

Requires rollback pointer.

### Tier 2: local execution

```text
RUN_PYTEST
RUN_SCRIPT
RUN_REPLAY
```

Requires timeout and stdout/stderr capture.

### Tier 3: forbidden in this line

```text
external mutation
production deploy
network side effect
global memory write
native action runtime
```

## Path policy

Allowed:

```text
workspace sandbox
temporary run directory
controlled output directory
return pack directory
```

Forbidden:

```text
production ICM
global user memory
theory baseline direct write
external service state
unscoped local user directories
```

## Authorization

ToolBridge execution requires:

```text
HarnessDispatchEnvelope.authorized_capabilities
```

and must fail closed if missing.
