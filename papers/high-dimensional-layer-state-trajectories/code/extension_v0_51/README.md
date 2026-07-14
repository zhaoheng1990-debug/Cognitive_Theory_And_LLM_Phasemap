# v0.51 reproducibility extension

This directory contains neutral, portable entry points for four additions to the preprint:

1. independent lexical-category validation;
2. trained-versus-random-initialization trajectory geometry;
3. the held-out displacement-to-candidate-boundary bridge; and
4. selective Qwen output-boundary control with a safety-matched action-reassignment null.

The scripts preserve the frozen analysis logic while replacing workstation paths with public checkpoint identifiers. Model access is governed by each upstream model license; Llama and Gemma may require authenticated access.

## Environment

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## Entry points

```bash
python run_lexical_category_validation.py --output-dir lexical_category_validation
python run_random_initialization_geometry.py --metadata inputs/relation_metadata.csv --trained-hidden inputs/qwen_trained_hidden.npy --subset-manifest inputs/controlled_subset_manifest.csv
python analyze_displacement_boundary_bridge.py --paired-input inputs/paired_intervention_behavior_rows.csv --controller-root inputs/internal_control_outputs
python run_qwen_selective_output_control.py --output output
python analyze_qwen_selective_output_control.py
python verify_qwen_selective_output_control.py
python run_safety_matched_action_nulls.py
```

The large extracted hidden-state arrays are intentionally not redistributed. They are deterministic intermediate products generated from the cited checkpoints and synthetic prompts. The accompanying source-data archive includes row-level outcomes, split records, null distributions, audit tables and figure inputs.

## Claim boundary

`DeltaU` is a task-constructed diagnostic. Random-initialization results concern one Qwen architecture and a controlled subset. The output-control result is candidate-boundary control on a synthetic relation-graph task: the chain-consistent candidate is not generally synonymous with answer correctness under closure, and the result is not open-ended generation or deployment control. Safety-matched reassignment does not support fine-grained per-trajectory dose specificity.
