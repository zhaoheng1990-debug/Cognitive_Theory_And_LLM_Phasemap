# v0.54 methodology-closure code

This extension adds public, neutral-named entry points for the four closure audits reported in the submission-ready manuscript.

## Fast verification

```bash
python verify_extension_v0_54.py
```

The verifier checks the archived target-excluded transition prediction, prospective functional-window boundary, actuator component attribution and label crosswalk.

## Model-level reruns

`run_predictive_transition_boundary.py` reruns the grouped target-excluded two-stage prediction from the included transition table. Set `TRANSITION_DATASET` and `TRANSITION_OUTPUT_DIR` only when overriding repository defaults.

`run_prospective_functional_window_confirmation.py` consumes the included discovery tables and `functional_window_shared.py`. Model paths can be supplied with `QWEN_MODEL_PATH`, `LLAMA_MODEL_PATH` and `GEMMA_MODEL_PATH`.

`run_actuator_component_attribution.py` reuses the public v0.52 `local_transition_executor.py`, `boundary_controller.py` and archived held-out boundary inputs. Full forward reruns require the cited checkpoints and local GPU memory.

## Boundaries

The prediction branch is exploratory and target-excluded, not a universal state equation. The prospective window recurrence gate failed. The actuator result attributes leverage within one bounded controller and declared candidate boundary; it is not correctness, open-ended generation or deployment control. Internal development identifiers and workstation paths are intentionally absent from this public extension.
