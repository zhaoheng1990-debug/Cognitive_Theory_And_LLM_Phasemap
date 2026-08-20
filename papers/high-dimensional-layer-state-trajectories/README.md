# Reproducibility release v2.0.0

This release accompanies **Architecture, training and execution organize
Transformer computation**. It provides processed Source Data, portable
verifiers and documented data provenance for the reported architecture,
training and execution analyses, including the paired Gemma-3-12B geometry
confirmation.

## Start here

* v2.0.0 release notes: `RELEASE_NOTES_v2_0.md`
* Release metadata: `RELEASE_METADATA_v2_0.yml`
* Source Data: `source_data/source_data/v1r9_final/`
* Full processed-data verifier: `code/release_v2_0/run_v2_0_reproduction.py`

## One-command verification

From this directory, run:

```powershell
python code/release_v2_0/run_v2_0_reproduction.py
```

The command recomputes eight documented endpoint families from processed Source
Data. It requires Python 3.9 or later plus the dependencies listed in
`requirements.txt`; it does not run model inference.
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
endpoints. It does not broaden the Article's task-specific claims.

* Source Data, version 2.0.0: https://doi.org/10.5281/zenodo.21813455
* Reproduction package, version 2.0.0: https://doi.org/10.5281/zenodo.22021200
