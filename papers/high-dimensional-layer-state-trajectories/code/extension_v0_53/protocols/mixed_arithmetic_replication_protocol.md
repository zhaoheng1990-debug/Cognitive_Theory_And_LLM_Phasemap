# Mixed-operation path-memory replication

The addition-only confirmatory window produced six of nine joint passes across
Qwen, Llama and Gemma, whereas lexical-category produced zero of nine. This
follow-up uses the pre-existing independent mixed-operation cache containing
addition, subtraction and multiplication prompts. Its prompts differ from the
addition-only set and only three expressions overlap.

All feature models, grouped splits, horizons, order nulls, classifiers and
multiple-testing procedures are inherited unchanged from
`trajectory_incremental_prediction_protocol_v0_1.md`.

Replication passes when at least four of nine model-horizon conditions pass the
same four-way incremental ordered-path gate and the passes span at least two
checkpoints. This supports a bounded computation-task path-history effect, not
a universal trajectory-memory law.
