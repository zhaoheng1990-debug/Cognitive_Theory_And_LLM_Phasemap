# Reproducibility release V1R9

This release accompanies the V1R9 Article **Architecture, training and
execution organize Transformer computation**. It adds a paired Gemma-3-12B
geometry confirmation to the V1R4 evidence record and provides a documented
route from the submitted Article to processed Source Data and portable
endpoint verification.

## Start here

* Article PDF: `paper/nature_initial_submission_v1r9.pdf`
* Article source: `paper/manuscript_v1r9_nature_initial.md`
* Updated Figure 2: `paper/figures_v1r9/Figure_2_architecture_scaffold.pdf`
* V1R9 release notes: `RELEASE_NOTES_v1r9.md`
* Source Data: `source_data/source_data/v1r9_final/`
* Portable verifier: `code/release_v1r9/run_v1r9_reproduction.ps1`

## One-command verification

From this directory, run:

```powershell
./code/release_v1r9/run_v1r9_reproduction.ps1
```

The command recalculates inherited V1R4 endpoints and the V1R9 paired 12B
geometry confirmation from processed Source Data. It requires Python 3.9 or
later and the standard library only. It does not run inference.
On Windows systems without long-path support, clone the repository close to a
drive root.

## Full-model reruns

The earlier model-execution runners remain in `code/extension_v0_51` through
`code/extension_v0_56`. A full rerun requires the named third-party
checkpoints, upstream prompt resources, a compatible GPU environment and
sufficient storage for hidden-state derivatives. Raw hidden states and model
weights are intentionally not redistributed.

## Scope and archive links

The verifier provides read-only confirmation of the reported numerical
endpoints. It does not broaden the Article's task-specific claims. Source Data
and Code are prepared as the V1R9 update under the stable concept records
https://doi.org/10.5281/zenodo.21317355 and
https://doi.org/10.5281/zenodo.21317440. The release-specific DOI will be
assigned when the new record versions are published.
