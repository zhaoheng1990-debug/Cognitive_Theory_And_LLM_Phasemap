# Negative Transfer Defer/Quarantine Report 7501A7600Z

Verdict: PASS_HIGH_NEGATIVE_TRANSFER_DEFER_OR_QUARANTINE

Review:

```json
{
  "action": "ROUTE_TO_EXPERIMENT",
  "eligible": false,
  "reason": "high_negative_transfer"
}
```

High/unbounded negative transfer does not enter baseline proposal queue. It routes to experiment/quarantine instead.
