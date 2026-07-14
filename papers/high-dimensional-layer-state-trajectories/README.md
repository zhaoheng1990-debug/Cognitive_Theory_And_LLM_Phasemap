# High-dimensional state trajectories reveal transformer inference dynamics

[English](#english) | [中文](#中文)

## English

This directory contains the public reproducibility materials associated with the preprint by Heng Zhao, Yufei Wang, Ruidian Song and Weisheng Chen.

Latest audited preprint in this branch: `arxiv_preprint_v0_51.pdf`.

Persistent archives: [Source Data](https://doi.org/10.5281/zenodo.21317355) and [Code](https://doi.org/10.5281/zenodo.21317440).

## Contents

- `paper/`: the current preprint PDF and its compilable arXiv LaTeX source.
- `code/analysis_scripts_raw/`: analysis and experiment scripts retained with provenance-preserving filenames.
- `code/wrappers/`: portable entry points and path wrappers.
- `code/figure_scripts/`: R scripts used to generate the manuscript figures.
- `code/config/`: public configuration files.
- `code/environment/`: Python and R environment notes.
- `code/extension_v0_51/`: portable code for the new task, random-initialization, boundary-bridge and output-control analyses.
- `source_data/source_data/`: processed panel-level tables for the main and supplementary figures.
- `source_data/metadata/`: checkpoint, prompt, split, metric, null-control and provenance metadata.
- `source_data/source_data/extension_v0_51/`: row-level evidence and figure inputs for the v0.51 extension.

## Reproduction levels

### Figure reproduction

The processed tables in `source_data/source_data/` and the scripts in `code/figure_scripts/` support regeneration of the reported figures without redistributing model weights or raw hidden-state arrays.

### Analysis reproduction

The scripts in `code/analysis_scripts_raw/` and `code/wrappers/` reproduce the reported analyses from the documented intermediate inputs. See the metadata tables for the model-specific windows, random seeds, classifier definitions and null controls.

### Full experiment reruns

Full reruns require local access to the listed Qwen, Llama and Gemma checkpoints and sufficient compute to extract hidden states. Model weights and large derived hidden-state arrays are not included. Checkpoint identifiers and revisions are recorded in `source_data/metadata/model_checkpoints_v0_2.csv`.

## Recommended starting points

1. Read `source_data/README.md` and the source-data manifest.
2. Inspect `code/config/example_paths.yaml` and use repository-relative paths.
3. Install the dependencies documented under `code/environment/`.
4. Run the relevant wrapper or figure script from the repository root.

## Evidence and scope

The v0.51 extension adds an independent lexical-category task, a random-initialization control, a held-out boundary bridge and selective full-vocabulary top-1 boundary control in Qwen. Vocabulary readbacks are treated as diagnostics. The trajectory coordinates reported in the manuscript are bounded empirical observables rather than universal order parameters. Geodesic-biased flow does not imply an exact geodesic, and the internal modulation experiments do not establish answer correctness or deployment-level control.

## Archival record

Persistent archives: [Source Data](https://doi.org/10.5281/zenodo.21317355) and [Code](https://doi.org/10.5281/zenodo.21317440).

## Contact

Heng Zhao: zhaoheng1990@gmail.com  
ORCID: https://orcid.org/0009-0004-2395-9393

---

## 中文

本目录收录 Heng Zhao、Yufei Wang、Ruidian Song 和 Weisheng Chen 所著预印本《High-dimensional layer-state trajectories reveal the internal dynamics of transformer inference》的公开复现材料。

本分支当前最新审计版本为 `arxiv_preprint_v0_51.pdf`。

持久归档：[Source Data](https://doi.org/10.5281/zenodo.21317355)；[Code](https://doi.org/10.5281/zenodo.21317440)。

### 目录内容

- `paper/`：当前预印本 PDF 及可编译的 arXiv LaTeX 源文件。
- `code/analysis_scripts_raw/`：保留原始溯源文件名的分析与实验脚本。
- `code/wrappers/`：可移植运行入口及路径封装脚本。
- `code/figure_scripts/`：生成论文图件的 R 脚本。
- `code/config/`：公开配置文件。
- `code/environment/`：Python 与 R 环境说明。
- `source_data/source_data/`：主图和补充图对应的面板级处理后数据表。
- `source_data/metadata/`：模型检查点、提示词、数据划分、指标、空值对照和溯源元数据。

### 复现层级

#### 图件复现

使用 `source_data/source_data/` 中的处理后数据表和 `code/figure_scripts/` 中的脚本，可以在不重新分发模型权重或原始隐藏状态数组的情况下重绘论文图件。

#### 分析复现

`code/analysis_scripts_raw/` 与 `code/wrappers/` 中的脚本可从文档规定的中间输入复现论文所报告的分析。模型特定功能窗口、随机种子、分类器定义和空值对照见元数据表。

#### 完整实验重跑

完整重跑需要在本地获取所列 Qwen、Llama 和 Gemma 检查点，并具备提取隐藏状态所需的计算资源。本仓库不包含模型权重和大型衍生隐藏状态数组。检查点标识符和版本记录见 `source_data/metadata/model_checkpoints_v0_2.csv`。

### 建议起点

1. 阅读 `source_data/README.md` 和 Source Data manifest。
2. 查看 `code/config/example_paths.yaml`，并使用仓库相对路径。
3. 按照 `code/environment/` 中的说明安装依赖。
4. 从仓库根目录运行相应的 wrapper 或绘图脚本。

### 证据与范围边界

v0.51 新增独立词汇类别任务、随机初始化对照、held-out 边界桥接分析，以及 Qwen 中选择性的全词表 top-1 候选边界控制。词表读出在本文中作为诊断量使用。论文报告的轨迹坐标是有边界的经验观测量，并非普适序参量。“测地线偏置流”不表示精确测地线；内部调制实验也不等同于答案正确性控制或部署级行为控制。

### 长期归档

持久归档：[Source Data](https://doi.org/10.5281/zenodo.21317355)；[Code](https://doi.org/10.5281/zenodo.21317440)。

### 联系方式

Heng Zhao：zhaoheng1990@gmail.com  
ORCID：https://orcid.org/0009-0004-2395-9393
