# Bug Report If Any 6901A-7000Z

Bugs found and fixed during productization validation:

1. Documented direct smoke command required source-root path setup in clean extraction contexts.
   - Fix: document and verify `python -m agentos_core_slim_portable.smoke`.

2. Deep Windows paths could exceed traditional path limits when receipt filenames embedded full receipt IDs.
   - Fix: store receipt files with stable short SHA-256 filenames while preserving full receipt IDs in payload content.

3. Bare `python -m pytest tests -q` could touch a user-level pytest temp directory with restricted permissions.
   - Fix: portable tests now create `.pytest_tmp` inside the extracted source tree instead of using pytest `tmp_path` fixture.

No boundary violation was observed.
