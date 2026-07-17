# High-dimensional state trajectories reveal transformer inference dynamics

Public reproducibility package for the paper and preprint. The current synchronized release is **v0.53**.

## Headline evidence

- Ordered hidden states are treated as the empirical process object; vocabulary readbacks remain diagnostics.
- Endpoint-preserving order effects recur across representative Qwen, Llama and Gemma checkpoints, but stronger controls show that coarse geometry is neither learned reasoning nor sufficient for native function.
- Random fixed mappings preserve prompt relations in 18/18 architecture-task-seed conditions, supporting an architecture-compatible propagation scaffold.
- Pythia component swaps and checkpoint audits show training-associated task coordinates and compatibility among input, block and output components.
- Confidence-gated local-transition actuators cross a declared full-vocabulary candidate boundary in held-out prompts, within explicit correctness and safety boundaries.

## Start here

1. Paper: `paper/arxiv_preprint_v0_53.pdf`
2. New code: `code/extension_v0_53/README.md`
3. New Source Data: `source_data/source_data/extension_v0_53/README.md`
4. File hashes: each extension has a `manifest.csv`

## Repository map

- `paper/`: versioned preprint PDFs and arXiv TeX source.
- `code/analysis_scripts_raw/`: provenance-preserving earlier analysis scripts.
- `code/extension_v0_51/` and `code/extension_v0_52/`: earlier public closures.
- `code/extension_v0_53/`: falsification-first scaffold, execution, metric and training audits.
- `source_data/source_data/`: figure and extension-level processed data.
- `code/environment/`: Python and R environment notes.

## Availability and boundaries

Source Data concept DOI: https://doi.org/10.5281/zenodo.21317355
Code concept DOI: https://doi.org/10.5281/zenodo.21317440

Model weights and large hidden-state arrays are not redistributed. Full reruns require the listed upstream checkpoints and compatible local compute. The evidence does not establish a universal dynamical equation, an exact geodesic, answer-correctness control, open-ended generation control or deployment safety.

## 中文说明

本仓库是论文 **High-dimensional state trajectories reveal transformer inference dynamics** 的公开复现包。v0.53 新增架构连续性、层顺序功能、轨迹历史增益、Llama 径向敏感性、Pythia 组件兼容性与训练期任务坐标审计。建议先阅读 `source_data/source_data/extension_v0_53/README.md`，再按 `manifest.csv` 核验文件。仓库不重分发模型权重或大型隐藏状态数组。

Contact: Heng Zhao <zhaoheng1990@gmail.com>; ORCID <https://orcid.org/0009-0004-2395-9393>.
