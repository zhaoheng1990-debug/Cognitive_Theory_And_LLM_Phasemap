# V1R9 Source Data

This directory contains the processed panel- and row-level tables supporting
the Article **Architecture, training and execution organize Transformer
computation**, including the V1R9 paired 12B geometry confirmation.

## Contents

* `authoritative/` is the canonical input to the portable endpoint verifier.
* `derived/` contains figure-oriented tables derived from the authoritative
  records.
* `metadata/` records prompt subsets, checkpoint revisions, split and null
  specifications, source-data mappings and figure contracts.
* `extension_v0_56/newgraph_geometry/` contains the V1R9 paired 12B prompt
  records, 50-null samples, frozen prompt manifest and validation records.
* `MANIFEST_SHA256.csv` is generated for this public release and covers the
  contents of this directory except the manifest itself.

## Immediate verification

From the parent release directory, run:

```powershell
./code/release_v1r9/run_v1r9_reproduction.ps1
```

This recomputes inherited endpoints from the `authoritative/` tables and the
paired 12B confirmation. It is a read-only Source Data audit; it does not
perform a model forward pass.

## Data boundary

The archive includes the processed inputs required to verify reported numerical
claims. It deliberately excludes raw hidden-state arrays, model weights and
any third-party content that cannot be redistributed. The accompanying code
documents how eligible users can regenerate omitted derivatives from the
named checkpoints and prompts.
