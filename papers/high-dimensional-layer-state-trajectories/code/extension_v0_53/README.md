# v0.53 falsification-first reproducibility extension

This extension separates architecture-compatible relational continuity, native ordered execution, metric sensitivity and training-organized task coordinates. It also retains the bounded actuator-to-coordinate-to-margin audit. All public workflow names are descriptive; model weights and large hidden-state arrays are not redistributed.

## Workflows

1. `run_cross_task_relational_continuity.py`: random fixed-map continuity across Qwen-, Llama- and Gemma-like schedules, three task families and two seeds.
2. `run_reordered_execution_audit.py`: native, reversed and fixed-permutation execution geometry versus output fidelity.
3. `run_incremental_path_history.py`: group-held-out incremental prediction beyond current state, including the mixed-arithmetic replication.
4. `extract_llama_residual_states.py` plus `analyze_llama_radial_*.py`: direction, norm and radial-exponent metric controls.
5. `run_pythia_component_swap.py`: complete initialized/trained E/B/N/W factorial intervention.
6. `analyze_pythia_task_coordinates.py`: model-native and fixed-final coordinate emergence across seven Pythia checkpoints.
7. `run_actuator_coordinate_margin_audit.py`: bounded internal-displacement to candidate-margin audit; not formal mediation.

Each entry point exposes input and output arguments. Set local model paths with `QWEN_MODEL_PATH`, `LLAMA_MODEL_PATH`, `LLM_DYNAMICS_SOURCE_ROOT` and `HF_TRAINING_CACHE` where required. Protocols record seeds, nulls and decision gates. Processed audit outputs are in `source_data/source_data/extension_v0_53/`.

## Processed-data verifier replay

From the paper-package root, the following verifiers run without model weights or hidden-state arrays:

```bash
python code/extension_v0_53/verify_reordered_execution.py source_data/source_data/extension_v0_53/execution_order
python code/extension_v0_53/verify_incremental_path_history.py source_data/source_data/extension_v0_53/incremental_path_history --expected-conditions 18
python code/extension_v0_53/verify_incremental_path_history.py source_data/source_data/extension_v0_53/mixed_arithmetic_replication --expected-conditions 9
python code/extension_v0_53/verify_actuator_coordinate_margin.py source_data/source_data/extension_v0_53/actuator_coordinate_margin --matched-verdict source_data/source_data/extension_v0_53/actuator_coordinate_margin/safety_matched_verdict.json
python code/extension_v0_53/verify_pythia_component_swap.py --result-dir source_data/source_data/extension_v0_53/pythia_component_compatibility --prior-outcomes source_data/source_data/extension_v0_53/pythia_component_compatibility/training_checkpoint_prompt_outcomes.csv
```

The continuity, Llama metric and Pythia coordinate verifiers additionally require the derivative state arrays described in their protocols. Their frozen outputs and source hashes remain in Source Data.

## Boundaries

Random relational continuity does not establish learned semantics, native-order privilege or endpoint direction. Positive chord statistics are not sufficient for native function and are radially sensitive under the declared metric. The tested ordered-prefix summaries do not establish a general path-memory law. The Pythia result is one small checkpoint series and one lexical diagnostic. Candidate-boundary control is not answer-correctness, open-ended generation, safety or deployment control.
