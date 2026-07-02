# GitHub Upload Manifest

Target repository: `zhaoheng1990-debug/Cognitive_Theory_And_LLM_Phasemap`

Target branch: `AgentOS`

Included core paths:

- `README.md`
- `.gitignore`
- `agentos_core_slim_v0/`
- `configs/`
- `project_baselines/`
- `scripts/`
- `seedpacks/`
- `_incoming/`
- `outputs/`

Excluded local-only paths:

- `_restore_source/`
- `_tmp*/`
- `.pytest_cache/`
- `__pycache__/`
- `outputs/release_size_check/stage_*/`

Validation:

```text
pytest -q agentos_core_slim_v0/tests
28 passed

python -m compileall -q agentos_core_slim_v0
passed
```
