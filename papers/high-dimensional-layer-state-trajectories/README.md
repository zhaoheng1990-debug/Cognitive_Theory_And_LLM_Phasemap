# High-dimensional state trajectories reveal transformer inference dynamics

Public paper and reproducibility package. The current synchronized repository release is **v0.56**; the submission route is a full Nature Article, with manuscript source v0.33 and Supplementary Methods v0.18. The former eight-page layout is retained only as a non-primary editor preview.

## Start here

1. Full Nature Article draft: `paper/nature_initial_submission_v0_33.pdf`
2. Manuscript source: `paper/manuscript_v0_33_closure_audits.md`
3. Supplementary Methods: `paper/supplementary_methods_v0_18.pdf`
4. v0.56 claim-family verification: `code/extension_v0_56/claim_family_verification_matrix_v0_56.md`
5. v0.56 Source Data: `source_data/source_data/extension_v0_56/README.md`

Run the three read-only v0.56 verifiers from `code/extension_v0_56/`:

```bash
python verify_predecision_coordinate_audit.py ../../source_data/source_data/extension_v0_56/coordinate_audit
python verify_empirical_direction_null.py ../../source_data/source_data/extension_v0_56/direction_null
python verify_newgraph_geometry_confirmation.py --output-root ../../source_data/source_data/extension_v0_56/newgraph_geometry
```

These scripts recompute the frozen predecision coordinate audit, the matched empirical-subspace direction null and the paired new-graph geometry result from retained processed tables. The earlier mixed-arithmetic transfer audit remains available in `code/extension_v0_55/`.

## What v0.56 adds

The v0.56 closure package addresses three explicit alternatives without reopening window selection or intervention-policy search.

- `DeltaU_pre` did not improve group-held-out mechanism-label classification over matched recent logit margins in Qwen, Llama or Gemma. Complete predecision histories and cutoff hidden states did improve, but no unordered-history comparator was tested.
- A norm-, layer-, gate- and action-matched empirical-subspace random-direction null did not support direction specificity after correction across the three checkpoints.
- On 24 fixed new graph groups (240 prompts), Qwen-1.5B and Gemma-3-12B-it-QAT-Q4_0 both exceeded all 50 shared endpoint-preserving shuffles in chord alignment. This is checkpoint-specific geometry confirmation, not a scale law.

The package makes the positive and limiting outcomes equally reproducible. The supported control claim remains bounded to declared relation-graph candidate boundaries; it does not establish answer correctness, deployment safety, checkpoint-general direction specificity or confirmed cross-task output control.

## Repository map

- `paper/`: current Nature Article sources, submission PDFs, compact preview and prior public preprints.
- `code/extension_v0_56/`: frozen protocol, three runners, three independent verifiers, multiplicity register, verification matrix and native Gemma capture source.
- `source_data/source_data/extension_v0_56/`: processed coordinate, direction-null and new-graph geometry evidence, raw fixed prompts and integrity receipts.
- `code/extension_v0_51/` to `code/extension_v0_54/`: earlier public experiment closures.
- `code/environment/`: Python and R environment notes.

## Boundaries and availability

The evidence supports an observable ordered process and bounded control of a declared candidate boundary. It does not establish an exact geodesic, least action, universal state equation, answer-correctness improvement, open-ended generation control or deployment safety. Vocabulary readbacks remain diagnostics.

- Source Data concept DOI: https://doi.org/10.5281/zenodo.21317355
- Code concept DOI: https://doi.org/10.5281/zenodo.21317440

The v0.56 Zenodo upload set is prepared as a new version of these existing concept records. Until that version is published, the concept records may resolve to an earlier archive version.

Model weights and large hidden-state arrays are not redistributed. All work was conducted as independent research without employer data, compute, funding, facilities or internal systems.

Contact: Heng Zhao <zhaoheng1990@gmail.com>; ORCID <https://orcid.org/0009-0004-2395-9393>.
