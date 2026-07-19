# v0.55 mixed-arithmetic transfer code

This incremental extension contains the frozen mixed-arithmetic actuator-transfer runner, an independent read-only verifier and the deterministic R script for Extended Data Fig. 9. Earlier closure runners remain in extensions v0.51-v0.54.

## Licence

This code is released under the Apache License 2.0. The same licence applies to the paper-specific GitHub branch and the Zenodo Code record.

## Fast verification

```bash
python verify_extension_v0_55.py
```

The verifier recomputes the three checkpoint summaries, all 150 address nulls, action-dose multiset preservation, replay gates, pooled specificity and the predeclared decision from retained prompt-level rows.

## Model-level reruns

`run_mixed_arithmetic_actuator_transfer.py` reuses the public v0.52 `local_transition_executor.py` and `boundary_controller.py`. Full forward reruns require the cited checkpoints and local GPU memory. Model paths are read through the same environment variables documented in v0.52. `mixed_arithmetic_task.py` fixes the prompt construction. `draw_extended_data9_mixed_arithmetic_public.R` rebuilds the panel from included Source Data.

## Boundaries

The pooled address test is positive, but only Qwen passes the powered positive-direction gate. This is partial protocol transfer, not confirmed cross-task output control, correctness, open-ended generation or deployment control. Internal development identifiers and workstation paths are intentionally absent from this public extension.
