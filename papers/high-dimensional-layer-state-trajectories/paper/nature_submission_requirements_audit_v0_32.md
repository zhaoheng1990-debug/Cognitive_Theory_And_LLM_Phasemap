# Nature submission requirements audit v0.32

Audit date: 20 July 2026

## Scope and basis

This audit concerns submission readiness, availability and public-surface consistency. It does not alter the scientific claims, figures, data or methods. Current policy pages were checked on 20 July 2026:

- Nature initial submission: https://www.nature.com/nature/for-authors/initial-submission
- Nature formatting guide: https://www.nature.com/nature/for-authors/formatting-guide
- Nature reporting standards: https://www.nature.com/nature/editorial-policies/reporting-standards
- Nature Portfolio authorship policy: https://www.nature.com/nature-portfolio/editorial-policies/authorship
- Springer Nature research-data policy: https://www.springernature.com/gp/journal-policies/15369670

## Submission-file audit

| Requirement | Local evidence | Status |
|---|---|---|
| Article title should fit two print lines; 75 characters is the Nature guide. | Title is 73 characters: `High-dimensional state trajectories reveal transformer inference dynamics`. | Pass |
| Summary paragraph should be non-technical and ideally no more than 200 words. | Abstract is 157 words. | Pass |
| Typical 8-page Article target is about 4,300 words with 5-6 modest display items. | Main text is 3,737 words; five main figures; compact editorial layout is eight pages. | Pass |
| Figures, legends and line numbering should be readable in one initial-submission PDF. | `output/pdf/nature_initial_submission_v0_32/nature_initial_submission_v0_32.pdf` is 1.02 MB and includes line numbering; legends were visually audited after rendering. | Pass |
| Up to ten Extended Data display items. | Nine Extended Data figures and one Extended Data table. | Pass |
| Main references should normally remain at or below 50. | Twenty-six references, with titles supplied. | Pass |
| Methods must include separate Data Availability and Code Availability sections. | Present in manuscript lines 350-356. | Conditional: archive release must be published before upload. |
| LLM use must be documented in Methods and LLMs cannot be authors. | Human-oversight disclosure is in Methods lines 346-348; no AI system is listed as an author. | Pass |
| Author contribution and competing-interest statements are required. | Present in manuscript lines 368-374 and in submission metadata. | Conditional: all authors must confirm. |

## Data and code availability audit

| Requirement | Local evidence | Status |
|---|---|---|
| Supporting minimum dataset has a public, persistent access route. | Source Data concept DOI: `10.5281/zenodo.21317355`; package, manifest and checksums are prepared locally. | Blocked externally: latest published record is still v1.1.0 and lacks v0.52-v0.55 extension archives. |
| Central custom code can be inspected and repeated. | Code concept DOI: `10.5281/zenodo.21317440`; public verifier, environment files and source mapping are prepared. | Blocked externally: latest published record is still v1.1.0 and lacks v0.52-v0.55 extension archives. |
| A public code mirror points to the paper's actual release, not unrelated project material. | Data and Code Availability statements now point to the dedicated `high-dimensional-layer-state-trajectories` branch. | Pass after GitHub sync. |
| Data and code licences are explicit and internally consistent. | Source Data is CC-BY-4.0. The paper branch uses Apache-2.0; Zenodo Code metadata, package README and bundled LICENSE now use Apache-2.0. | Pass after the Code Zenodo version is published with Apache-2.0. |
| Large third-party derivatives and model weights have an explicit access boundary. | Manuscript identifies hidden-state arrays and weights as non-redistributed derivatives, while prompts, revisions and extraction scripts are supplied. | Pass |

## Author actions before finalization

1. Confirm author order, affiliations, CRediT roles, funding and competing-interest declarations using `author_confirmation_release_form_v0_32.md`.
2. Confirm the public preprint identifier or confirm no public preprint, and confirm no concurrent submission.
3. Create and publish v0.55 under both existing Zenodo concept records, retain the prior files, add the four v0.52-v0.55 extension archives, and use the titles and licences in `release_v0_55/zenodo_upload_set_v0_55/README.md`.
4. Run `verify_zenodo_release_v0_55.py`; publication is complete only when every check passes.
5. Verify that the dedicated GitHub branch resolves to the current reproducibility release, then use `finalize_nature_submission_v0_32.py` only after all release gates pass.

## Non-blocking editorial recommendation

Nature asks that acknowledgements be brief. The current acknowledgements are substantive personal context, not a compliance failure. Before actual submission, H.Z. should decide whether to retain them in full or replace them with a shorter acknowledgement while preserving the funding and resource-provenance disclosures in Methods.

## Verdict

The paper is submission-format ready. The only hard blockers are author confirmation and publication of the two v0.55 Zenodo records. The GitHub default branch should not be cited; the manuscript and release materials now use the paper-specific branch to keep the public evidence surface aligned with the paper's bounded claims.
