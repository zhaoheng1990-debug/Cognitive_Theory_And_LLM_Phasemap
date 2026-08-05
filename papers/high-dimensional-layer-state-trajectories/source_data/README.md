# Source Data for high-dimensional layer-state trajectories

This directory stages processed, panel-level Source Data for **Ordered layer-state trajectories enable process-level analysis of transformer inference**. The synchronized repository release is **v0.56**.

## Citation and access

- Source Data concept DOI: https://doi.org/10.5281/zenodo.21317355
- Public code concept DOI: https://doi.org/10.5281/zenodo.21317440

The v0.56 Source Data upload set is prepared as a new version of the existing Source Data concept record. Until that version is published, the concept DOI resolves to an earlier archive and does not yet contain all v0.52-v0.56 extensions. Use the versioned Zenodo record for formal citation once it is published; this GitHub directory is the public mirror and working index.

## Contents and scope

- `source_data/extension_v0_51/` to `source_data/extension_v0_56/`: incremental prompt-level evidence, nulls, verification records and figure-source inputs.
- `metadata/`: global figure mappings, metric definitions, prompt and checkpoint metadata, grouped splits, null specifications and classifier details.

Tabular data use CSV; structured summaries use JSON; manifests record file-level integrity information. The archive contains processed values needed to inspect figures and reported statistics. It does not redistribute model weights or large hidden-state arrays, which are derivative intermediate outputs of third-party checkpoints. Checkpoint revisions, prompts and extraction metadata define the regeneration route.

## Licence

Processed Source Data and documentation in the published v0.56 Zenodo version are released under CC-BY-4.0. Upstream checkpoint access and licensing remain governed by their providers.
