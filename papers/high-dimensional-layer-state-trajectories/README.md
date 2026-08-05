# Reproducibility release v0.59

This release is a publication-material successor to the frozen v0.56 source-data and executable-verifier package. It adds the deep-humanized full Nature Article v0.40, synchronized editorial documents and retains Supplementary Methods v0.20 without changing any code, processed Source Data, verifier, result, null distribution or numerical claim. It does not redistribute model weights, commercial/runtime binaries or large raw hidden-state arrays.

## Start here

- Full Article: `paper/nature_initial_submission_v0_40.pdf`
- Article source: `paper/manuscript_v0_40_deep_humanized.md`
- Language audit: `paper/humanizer_audit_v0_40.md`
- Editorial documents: `paper/presubmission_enquiry_nature_v0_7_deep_humanized.md` and `paper/cover_letter_nature_v0_5_deep_humanized.md`
- Editorial language audit: `paper/editorial_humanizer_audit_v0_1.md`
- Supplementary Methods: `paper/supplementary_methods_v0_20.pdf`
- Claim-family verification map: `code/extension_v0_56/claim_family_verification_matrix_v0_56.md`

## Verification route

Run the three commands in `code/extension_v0_56/README.md` from the repository root. The coordinate and direction verifiers recompute released summaries from processed rows. The geometry verifier operates in processed-source verification mode unless archive-only raw arrays are supplied.

## Archive-only raw intermediates

`source_data/source_data/extension_v0_56/newgraph_geometry/raw_intermediates_manifest_v0_56.csv` records SHA-256 hashes for the large raw geometry arrays. They are excluded from this public package to avoid redistributing derivative captures tied to separately licensed checkpoints. They are retained for controlled archival verification.

## Human-review boundary

`paper/author_numeric_review_checklist_v0_56.md` remains unsigned. No author-level item-by-item numerical-review completion is claimed by this release.
