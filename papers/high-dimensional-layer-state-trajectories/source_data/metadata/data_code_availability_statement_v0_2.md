# Data and Code Availability draft V0.2

## Data availability

Source data for all main and Extended Data figure panels are organized as processed panel-level tables in the manuscript Source Data package. The pre-submission staging package contains processed Source Data, a Source Data manifest, figure-source checks, detail-audit records, export-QA records, metric definitions, prompt metadata, hidden-state extraction metadata, random seeds, null-control specifications, classifier specifications, model-checkpoint metadata and Supplementary Methods reproducibility notes. These materials will be deposited in a public archival repository before submission, and the repository DOI or accession will be added to the final manuscript.

Raw hidden-state arrays are large intermediate outputs generated from third-party model checkpoints. They will either be deposited separately, regenerated from the archived prompts, checkpoint identifiers and path-parameterized extraction scripts, or excluded from the public archive with an explicit storage-size and model-licensing rationale. Model weights are not redistributed by this manuscript and remain subject to the access terms and licences of their original providers.

## Code availability

Analysis scripts, figure-generation scripts, environment records, portable smoke-check wrappers and a portable path-configuration template are included in the pre-submission code-provenance package. The raw-script workstation-path cleanup pass is complete for the detected Python analysis scripts, which now support repository-relative paths and public preflight checks. Full raw-output-to-final-figure recomputation still requires upstream datasets, model checkpoints, GPU runtime, raw hidden-state array decisions and upstream figure panel-source dependencies. Before submission, a public code release will be deposited with the processed Source Data archive or in a linked archival code repository, and the DOI or accession will be added to the final manuscript.

## Repository and citation actions

- Complete the public repository record and DOI/accession.
- Choose processed-data, analysis-code and documentation/metadata licences.
- Fill `repository_staging_v0_1/CITATION.cff.template`, then rename it to `CITATION.cff` only after real DOI/accession, version, date, authorship and licence fields are final.
- Decide whether raw hidden-state arrays are deposited, regenerated or explicitly excluded with justification.
- Decide whether upstream figure panel-source tables are deposited or regenerated.
- Fill `repository_release_metadata_template_v0_1.yml`.

## Chinese author check

- 需要作者确认最终数据仓库类型和 DOI/accession。
- 需要作者确认 processed Source Data、analysis code 和 metadata/documentation 的 licence。
- 需要作者决定 raw hidden-state arrays 是单独存放、通过脚本再生成，还是因体积和模型许可限制不公开存放但给出可复现路径。
- 需要作者决定上游 figure panel-source tables 是直接存放还是通过脚本再生成。
- 需要在最终投稿前把 `CITATION.cff.template` 填完并改成正式 `CITATION.cff`。
