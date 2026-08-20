# Source Data for Architecture, training and execution organize Transformer computation

This directory stages processed, panel-level Source Data for **Architecture,
training and execution organize Transformer computation**. The synchronized
repository release is **v2.0.0**.

## Citation and access

- Source Data v2.0.0: https://doi.org/10.5281/zenodo.21813455
- Reproduction package v2.0.0: https://doi.org/10.5281/zenodo.22021200

This GitHub directory is a browsable mirror and working index. The versioned
Zenodo records are the citable archives.

## Contents and scope

- `source_data/extension_v0_51/` to `source_data/extension_v0_56/`: incremental prompt-level evidence, nulls, verification records and figure-source inputs.
- `metadata/`: global figure mappings, metric definitions, prompt and checkpoint metadata, grouped splits, null specifications and classifier details.

Tabular data use CSV; structured summaries use JSON; manifests record file-level integrity information. The archive contains processed values needed to inspect figures and reported statistics. It does not redistribute model weights or large hidden-state arrays, which are derivative intermediate outputs of third-party checkpoints. Checkpoint revisions, prompts and extraction metadata define the regeneration route.

## Licence

Processed Source Data and documentation in the published Zenodo version are released under CC-BY-4.0. Upstream checkpoint access and licensing remain governed by their providers.
