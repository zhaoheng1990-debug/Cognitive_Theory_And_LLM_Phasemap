# Engineering Seed: HumanReview UI-1B

## Name

\[
oxed{
HumanReview	ext{-}UI	ext{-}1B:
DecisionExportBridgeForPB1S\_NoMutationNoAccept
}
\]

## Objective

Build the bridge from HumanReview UI decision buttons to PB1S-compatible decision records.

The primary workflow must remain:

```text
review card -> button decision -> export -> validation -> PB1S intake-ready record
```

The reviewer must not manually edit machine CSV/schema as the primary workflow.

## Required capabilities

1. Load a local `HumanReviewObject` fixture.
2. Render the PB retention review card.
3. Let the reviewer choose one allowed decision.
4. Generate a PB1S-compatible JSON decision record.
5. Generate a PB1S-compatible CSV decision record.
6. Validate both JSON and CSV against the PB1S decision schema.
7. Show a readable post-export confirmation.
8. Preserve raw schema under Advanced.
9. Produce a no-mutation audit.
10. Produce a no-accept/no-promotion audit.

## Allowed decisions

```text
keep_pending_retention_review
request_more_boundary_cases
request_scope_revision
reject_retention
authorize_retention_accept_preflight_next
```

## Forbidden decisions / actions

```text
retention_accept
signed_accept
accepted_baseline
accepted_evidence
memory_write
operator_promotion
policy_promotion
```

## Required output state

```text
PB1S_DECISION_RECORD_READY_FOR_INTAKE
```

or safe blocked state:

```text
PB1S_DECISION_EXPORT_BLOCKED_WITH_REASONS
```

## Required verdict

```text
PASS_HUMAN_REVIEW_UI_1B_DECISION_EXPORT_BRIDGE_FOR_PB1S_NO_MUTATION_NO_ACCEPT
```
