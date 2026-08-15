# Reproducibility release v0.60

This release accompanies the V1R4 Article **Architecture, training and
execution organize Transformer computation**. It provides a documented route
from the final Article to its processed Source Data and portable endpoint
verification.

## Start here

* Article PDF: `paper/nature_initial_submission_v0_60.pdf`
* Article source: `paper/manuscript_v0_60_nature_initial.md`
* Full language audit: `paper/humanizer_audit_v0_60.md`
* V1R4 release notes: `RELEASE_NOTES_v0_60.md`
* Source Data: `source_data/source_data/v1r4_final/`
* Portable verifier: `code/release_v0_60/verify_v1r4_source_data.py`

## One-command verification

From this directory, run:

```powershell
./code/release_v0_60/run_v1r4_reproduction.ps1
```

The command recalculates 20 reported endpoints from the processed V1R4 Source
Data and writes `verification/v1r4_source_data_verification.json`. It requires
Python 3.9 or later and the standard library only. It does not run inference.
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
and Code are prepared for Zenodo v1.2.0 under the stable concept records
https://doi.org/10.5281/zenodo.21317355 and
https://doi.org/10.5281/zenodo.21317440. The release-specific DOI will be
assigned when the v1.2.0 records are published.
