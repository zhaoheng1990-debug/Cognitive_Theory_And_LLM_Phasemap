# ToolBridge Pass/Block Test Report 7201A7300Z

Verdict: PASS_AUTHORIZED_AND_BLOCK_TESTS

Covered cases from `agentos_core_slim_v0/tests/test_codex_tool_bridge.py`:

- Authorized READ_ARTIFACT returns PASS with hash.
- Unauthorized RUN_SCRIPT returns BLOCKED before execution.
- Forbidden EXTERNAL_API_MUTATION returns BLOCKED.
- Missing Kernel dispatch returns BLOCKED.
- WRITE_LOCAL_PATCH returns PASS only with rollback metadata.
- PACKAGE_ZIP and INSPECT_MANIFEST return PASS inside allowed workspace.

Pytest command:

```text
D:\anaconda\python.exe -m pytest agentos_core_slim_v0/tests/test_codex_tool_bridge.py -q --basetemp outputs/7201A7300Z_arbor_codex_tool_bridge/pytest_tmp_report
```

Pytest output:

```text
......                                                                   [100%]
6 passed in 0.05s

```
