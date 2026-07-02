# Codex Task Prompt: Arbor-Codex ToolBridge 7201A–7300Z

You are executing AgentOS CoreSlim Arbor-Codex ToolBridge patch.

## Goal

Strengthen the Harness execution layer by adding a bounded CodexToolBridge that Arbor can call under AgentOS Kernel-authorized typed envelopes.

Do not move Codex into Kernel cognition.

## Implement / produce

1. Define CodexToolBridge interface.
2. Define ToolCallEnvelope schema.
3. Define ToolReceipt schema.
4. Add minimal local capabilities:
   - READ_ARTIFACT
   - WRITE_LOCAL_PATCH
   - RUN_PYTEST
   - RUN_SCRIPT
   - PACKAGE_ZIP
   - GENERATE_HASH_INVENTORY
   - INSPECT_MANIFEST
   - RUN_REPLAY
   - ROLLBACK_LOCAL_WRITE
5. Add ArborHarness integration example.
6. Add Kernel-authorized HarnessDispatchEnvelope example.
7. Add tests for pass/block/rollback/replay.
8. Generate reports.

## Hard boundaries

- CodexToolBridge cannot own SRO.
- CodexToolBridge cannot own NextAction.
- CodexToolBridge cannot own ICMMetabolismPolicyDecision.
- CodexToolBridge cannot write global MemoryUnit / OperatorMemory / ICM.
- CodexToolBridge cannot perform external mutation.
- AgentOSKernel remains final policy owner.

## Required return files

Return all files listed in `07_Return_Files_Manifest_7201A7300Z.json`.

Package them into a zip.
