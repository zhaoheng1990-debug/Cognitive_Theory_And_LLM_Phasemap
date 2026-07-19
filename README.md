# High-dimensional state trajectories reveal transformer inference dynamics

Public paper, Source Data and reproducibility code for the study of ordered high-dimensional layer-state trajectories in transformer inference.

The current repository release is **v0.55**. Start with the [paper-specific README](papers/high-dimensional-layer-state-trajectories/README.md).

## Repository scope

This repository supports four bounded empirical questions:

1. whether ordered hidden states provide a useful process-level object for comparing inference across depth and interventions;
2. whether direct-state coordinates preserve information that vocabulary readbacks can obscure;
3. whether realized layer order differs from endpoint-preserving order nulls; and
4. whether declared internal and emitted-token boundaries can be selectively perturbed under explicit controls.

The release includes submission-ready manuscript files, processed figure data, grouped predictive tests, null controls, intervention attribution and independent verification scripts.

## Scientific boundaries

The evidence does not establish an exact geodesic, a least-action principle, a universal dynamical equation, answer-correctness control, open-ended generation control or deployment safety. Vocabulary readbacks remain diagnostics. Model-specific functional windows remain exploratory after a prospective recurrence gate failed.

## Archives

- Source Data: https://doi.org/10.5281/zenodo.21317355
- Code: https://doi.org/10.5281/zenodo.21317440

This branch contains the current v0.55 extension. Until the v0.55 Zenodo versions are published, the concept records resolve to the earlier archived release; the v0.52-v0.55 deposit set is prepared for publication as a new version of each existing concept record.

Model weights and large derivative hidden-state arrays are not redistributed. Full forward reruns require the cited upstream checkpoints and compatible local compute.

Contact: Heng Zhao <zhaoheng1990@gmail.com>; ORCID <https://orcid.org/0009-0004-2395-9393>.
