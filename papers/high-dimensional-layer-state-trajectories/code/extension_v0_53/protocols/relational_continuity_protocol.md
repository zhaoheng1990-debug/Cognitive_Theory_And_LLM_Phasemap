# Random-initialization cross-task replication

## Objective

Falsify the interpretation that the random-initialization long-range relational
geometry is specific to the controlled relation-graph prompt family.

## Design

Two new initialization seeds (`1801`, `1802`) are used for each of Qwen, Llama
and Gemma. A model is initialized once per architecture and seed, then applied
unchanged to 96 lexical-category, 96 arithmetic-addition and 96 ARC-Challenge
prompts. This holds the random map fixed across tasks within a seed.

Every condition is tested against:

```text
condition/answer-stratified lag-1-matched Markov donor;
condition/answer-stratified contiguous four-step donor;
prompt-fingerprint-preserving donor.
```

The primary metrics are cross-half and first-to-last-quarter unbiased linear
CKA. A task-model-seed condition passes only when both metrics exceed the 95th
percentile of all three nulls with one-sided empirical `P <= 0.05`.

## Cross-task gate

Architecture-level replication requires both seeds to pass in at least two of
the three new tasks for every architecture. Task-level replication requires
both seeds to pass in at least two of the three architectures for every new
task. Failure of either requirement rejects a cross-task architecture claim.

Passing this gate supports only architecture-compatible preservation of prompt
relational geometry across depth. It does not support semantic organization,
endpoint direction, native-order privilege, task function or a geodesic.
