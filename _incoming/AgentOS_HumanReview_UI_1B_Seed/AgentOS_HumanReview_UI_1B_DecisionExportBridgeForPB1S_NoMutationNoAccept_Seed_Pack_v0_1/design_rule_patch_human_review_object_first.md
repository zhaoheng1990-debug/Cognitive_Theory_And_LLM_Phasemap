# AgentOS Design Rule Patch: HumanReviewObject First

## Rule

\[
oxed{
HumanReviewObject ightarrow MachineSchema
}
\]

not:

\[
oxed{
MachineSchema ightarrow HumanReview}
\]

## Practical requirements

1. Human reviewers see readable cards first.
2. Machine schema is hidden under Advanced.
3. Decision options are buttons with consequences, not raw enum guessing.
4. CSV/JSON is an export artifact, not the primary human interface.
5. Forbidden actions must be visible before action buttons.
6. UI surfaces must be no-mutation unless a separate governed execution gate is active.

## Applies to

```text
PB retention review
SV pending evidence review
DIPB domain pack review
future HumanGate queues
```
