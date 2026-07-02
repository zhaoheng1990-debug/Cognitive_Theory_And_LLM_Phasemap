# Hard Fail Conditions: HumanReview UI-1B

Immediate FAIL if:

1. PM/human reviewer must manually edit raw CSV/schema as primary workflow.
2. UI export cannot produce both JSON and CSV decision records.
3. Exported records do not validate against PB1S schema.
4. Forbidden decisions are exportable.
5. UI executes RetentionAccept or SignedAccept.
6. UI writes AcceptedBaseline, AcceptedEvidence, Memory, Operator, or Policy.
7. UI mutates backend state.
8. UI uses external API or web browsing.
9. Raw schema is shown as the primary review surface.
10. Evidence chain or forbidden actions are not shown before decision export.
