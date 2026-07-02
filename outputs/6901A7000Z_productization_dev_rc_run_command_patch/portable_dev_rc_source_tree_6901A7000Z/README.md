# AgentOS CoreSlim Portable Dev-RC Source Tree 6901A-7000Z

Verdict target: `PASS_PRODUCTIZATION_DEV_RC_RUN_COMMAND_PATCH_AND_FREEZE`

This package is a minimal standalone developer-RC source tree for the declared CoreSlim subset. It can be unpacked and tested without prior configured AgentOS output roots.

It proves:

- importable portable package;
- runtime object definitions and state-machine smoke path;
- ProviderAugmentationGateway advisory-only path;
- Harness envelope / receipt interfaces;
- AgentOSKernel-owned ICMMetabolismPolicy decision;
- controlled local ICM sandbox write and rollback;
- replay truthfulness distinctions;
- no external mutation inside the declared tests.

It does not prove:

- production release;
- standalone replay of full historical configured project evidence;
- unrestricted ICM write;
- global MemoryUnit or OperatorMemory write;
- AGI achieved or AGI Precursor 100%;
- native ActionRuntime, UI, Human Authority Surface, SelfBoundaryState, or FrameRevision closure.

Run from the extracted source tree root:

```powershell
python -m pytest tests -q --basetemp .pytest_tmp
python -m agentos_core_slim_portable.smoke
```
