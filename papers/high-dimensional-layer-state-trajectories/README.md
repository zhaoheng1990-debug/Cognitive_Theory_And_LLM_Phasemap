# High-dimensional state trajectories reveal transformer inference dynamics

Public paper and reproducibility package. The current synchronized release is **v0.54**.

The public arXiv preprint remains v0.53 until its next revision. The v0.54 repository release adds the submission-ready manuscript, complete figure source data and the final methodology-closure evidence without relabelling the earlier preprint.

## Start here

1. Compact manuscript view: `paper/nature_compact_8page_layout_v0_29.pdf`
2. Full submission draft: `paper/nature_initial_submission_v0_29.pdf`
3. Supplementary Methods: `paper/supplementary_methods_v0_15.pdf`
4. v0.54 verification: `code/extension_v0_54/verify_extension_v0_54.py`
5. v0.54 Source Data: `source_data/source_data/extension_v0_54/README.md`

Run the public closure checks from the v0.54 code directory:

```bash
python verify_extension_v0_54.py
```

The verifier reads archived evidence and checks nine fixed results. Full model-level reruns require the checkpoints listed in the manuscript and compatible local GPU memory.

## What v0.54 closes

- **Target-excluded prediction:** a grouped out-of-fold model reached macro R2 = 0.862, improving over the state-only baseline by 0.0394 (group-bootstrap 95% interval 0.0368-0.0426). It retains 25.1% of the gain associated with the realized descriptive transition coordinate and remains exploratory.
- **Label crosswalk:** coordinate-audit and intervention prompt labels are separately executable. They are analysis classes, not trajectory-derived dynamical phases or correctness labels.
- **Actuator attribution:** the confidence-gated local transition operator carries most declared output-boundary leverage. Operator-only control reproduced 39 pooled strict crossings; the precursor alone produced 1. Uniform maximum action produced 47 crossings with 3 non-target changes, so fine prompt-specific dose assignment is not supported.
- **Prospective window test:** the frozen three-checkpoint recurrence gate failed. Joint role recurrence passed 0/3 checkpoints and location recurrence passed 1/3. Functional windows therefore remain exploratory model-specific indexing choices.

## Repository map

- `paper/`: public preprint, submission-ready PDFs and manuscript sources.
- `code/extension_v0_54/`: neutral-named closure runners and public verifier.
- `source_data/source_data/extension_v0_54/`: predictive, prospective, intervention, label-crosswalk and figure source data with SHA-256 manifests.
- `code/extension_v0_51/` to `code/extension_v0_53/`: earlier public experiment closures.
- `code/analysis_scripts_raw/`: provenance-preserving earlier analysis scripts.
- `code/environment/`: Python and R environment notes.

## Boundaries and availability

The evidence supports an observable ordered process and bounded control of a declared candidate boundary. It does not establish an exact geodesic, least action, universal state equation, answer-correctness improvement, open-ended generation control or deployment safety. Vocabulary readbacks remain diagnostics.

- Source Data concept DOI: https://doi.org/10.5281/zenodo.21317355
- Code concept DOI: https://doi.org/10.5281/zenodo.21317440

The repository branch is current at v0.54. As of 18 July 2026, the Zenodo concept records still resolve to the earlier archived release; the complete v0.52-v0.54 deposit set is awaiting publication as the next version.

Model weights and large hidden-state arrays are not redistributed. All work was conducted as independent research without employer data, compute, funding, facilities or internal systems.

Contact: Heng Zhao <zhaoheng1990@gmail.com>; ORCID <https://orcid.org/0009-0004-2395-9393>.
