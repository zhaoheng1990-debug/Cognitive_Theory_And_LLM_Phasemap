# UPS BaselineUpdateProposal Test Case v0.2

## Input

From 7301:

```text
UPS ACCEPT
→ AUTONOMOUS_PROJECT_OPERATORMEMORY_WRITE
→ UtilityPolicySelectorOperatorMemoryRecord
→ replay PASS
→ rollback pointer present
```

## Expected 7501 behavior

AgentOS should generate:

```text
BaselineUpdateProposal
```

with:

```text
proposal_type = THEORY_BASELINE
target_scope = S3_OFFICIAL_THEORY_BASELINE
target_path_or_registry = C:\Users\ZH\Desktop\AGI\理论基线
proposed_action = PROPOSE_APPEND_BASELINE_UPDATE
requires_signed_authorization = true
production_activation = false
```

## Expected boundary

The official theory baseline must not be written in this line.

The output is a HumanReviewPacket and proposal queue entry only.

Actual promotion requires SignedBaselinePromotionAuthorization.
