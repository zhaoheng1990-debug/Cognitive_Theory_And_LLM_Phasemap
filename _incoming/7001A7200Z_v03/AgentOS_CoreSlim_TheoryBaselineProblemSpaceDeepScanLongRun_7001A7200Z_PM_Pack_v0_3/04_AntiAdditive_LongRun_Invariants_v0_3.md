# Anti-Additive LongRun Invariants v0.3

## Do not expand modules

This run must not add:

```text
new harnesses
new profile families
new promotion gates
new no-write blockers
new provider variants
new UI
new native action runtime
```

## Reduce uncertainty

The run must reduce or clarify active problem-space uncertainty.

Allowed reductions:

```text
pending -> validation seed
pending -> expired/deprecated
pending -> merged
pending -> quarantined
pending -> next concrete test
pending -> separate experiment route
```

## Fail conditions

Fail if:

1. Only the prior 3-item mini-ledger is processed while local theory baseline is available.
2. New pending items are generated without family clustering or Cbit justification.
3. More than 20% of extracted items have no closure action.
4. Kernel ownership of final closure decisions is violated.
5. External production writes occur.
6. The run claims all pending items are accepted.
