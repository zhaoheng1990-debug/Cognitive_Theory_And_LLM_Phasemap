# Rollback Replay Report 7201A7300Z

Verdict: PASS_ROLLBACK_AND_REPLAY

The rollback test writes `mutable.txt`, records pre-write content/hash in a local rollback pointer, runs receipt replay, then restores the original content through `ROLLBACK_LOCAL_WRITE`.

Required assertions:

- Write receipt status: PASS.
- Replay receipt status: PASS.
- Replay receipt count: 1.
- Rollback receipt status: PASS.
- Final file content equals original content.
- ToolReceipt remains hash-addressed through `receipt_hash`.
