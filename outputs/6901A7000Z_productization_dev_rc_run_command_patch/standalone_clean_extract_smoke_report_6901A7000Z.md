# Standalone Clean Extract Smoke Report 6901A-7000Z

Command run exactly as documented from clean extracted bundle:

```powershell
python -m agentos_core_slim_portable.smoke
```

Exit code: `0`

Output:

```text
kernel_owner = AgentOSKernel.ICMMetabolismPolicy
advisory_only = True
harness_no_external_effect = True
replay_verified = True
production_release = False
truthfulness_rows = 2
```

Required output checks:

| Check | Passed |
|---|---:|
| `kernel_owner = AgentOSKernel.ICMMetabolismPolicy` | True |
| `advisory_only = True` | True |
| `harness_no_external_effect = True` | True |
| `replay_verified = True` | True |
| `production_release = False` | True |
