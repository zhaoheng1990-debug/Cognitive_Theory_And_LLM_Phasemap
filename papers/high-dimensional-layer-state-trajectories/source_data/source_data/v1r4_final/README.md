# V1R4 Source Data

This directory contains the processed panel- and row-level tables supporting
the Article **Architecture, training and execution organize Transformer
computation**.

## Contents

* `authoritative/` is the canonical input to the portable endpoint verifier.
* `derived/` contains figure-oriented tables derived from the authoritative
  records.
* `metadata/` records prompt subsets, checkpoint revisions, split and null
  specifications, source-data mappings and figure contracts.
* `MANIFEST_SHA256.csv` is generated for this public release and covers the
  contents of this directory except the manifest itself.

## Immediate verification

From the parent release directory, run:

```powershell
./code/run_v1r4_reproduction.ps1
```

This recomputes 20 reported endpoints from the `authoritative/` tables. It is a
read-only Source Data audit; it does not perform a model forward pass.

## Data boundary

The archive includes the processed inputs required to verify reported numerical
claims. It deliberately excludes raw hidden-state arrays, model weights and
any third-party content that cannot be redistributed. The accompanying code
documents how eligible users can regenerate omitted derivatives from the
named checkpoints and prompts.
