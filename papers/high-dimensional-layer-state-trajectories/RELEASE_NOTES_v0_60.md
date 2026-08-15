# Release notes v0.60

Version 0.60 synchronizes the public repository with the V1R4 Article state.
It updates the Article title, author-affiliation record, processed Source Data
tree, verification entry point and Zenodo v1.2.0 metadata drafts.

## New public materials

* `source_data/source_data/v1r4_final/` contains 70 frozen V1R4 source files
  plus the release README, licence and checksum manifest.
* `code/release_v0_60/` provides a dependency-free Python verifier and a
  PowerShell wrapper.
* `verification/v1r4_source_data_verification.json` records the maintained
  20/20 endpoint verification result.
* `paper/` holds the Article, Markdown source and full prose audit.

## Claim boundary

The update preserves the retained negative results and the Article's stated
scope. It does not provide model weights, raw hidden-state arrays or a claim of
general output control. Full model execution continues to require separately
licensed inputs and a suitable runtime.
