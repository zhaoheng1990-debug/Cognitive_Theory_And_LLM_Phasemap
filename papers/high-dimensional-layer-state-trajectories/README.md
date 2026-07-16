# High-dimensional state trajectories reveal transformer inference dynamics

[English](#english) | [中文](#中文)

## English

This repository is the audited reproducibility package for the paper and preprint **High-dimensional state trajectories reveal transformer inference dynamics** by Heng Zhao, Yufei Wang, Ruidian Song and Weisheng Chen.

Latest audited preprint: `paper/arxiv_preprint_v0_52.pdf`.

- Source Data concept DOI: <https://doi.org/10.5281/zenodo.21317355>
- Code concept DOI: <https://doi.org/10.5281/zenodo.21317440>

### Start here

1. Read `source_data/source_data/extension_v0_52/README.md` for the new evidence map.
2. Inspect `source_data/source_data/extension_v0_52/manifest.csv` for file-level hashes.
3. Use `code/extension_v0_52/README.md` for portable experiment entry points.
4. Compile `paper/arxiv_tex_v0_52/main.tex` to reproduce the preprint.

### Repository layout

```text
paper/                         audited PDF and compilable arXiv source
source_data/metadata/          checkpoint, split, metric and null-control metadata
source_data/source_data/       processed evidence for the original analyses
source_data/source_data/extension_v0_51/
                               first independent-task and Qwen boundary extension
source_data/source_data/extension_v0_52/
                               three-closure row-level evidence and figure sources
code/analysis_scripts_raw/     provenance-preserving historical scripts
code/figure_scripts/           deterministic R figure scripts
code/extension_v0_51/          earlier portable extension
code/extension_v0_52/          current portable closure workflows
```

### What v0.52 closes

| Question | Test | Result | Boundary |
|---|---|---|---|
| Does order geometry recur beyond binary relation graphs? | Lexical, addition and unreduced four-choice ARC across Qwen, Llama and Gemma | Real order exceeded all 50 endpoint shuffles in all 9 task-checkpoint tests | Task-specific scalar coordinates transferred in 5/6 controlled tests; relation-graph `DeltaU` was not reused |
| Is there predictive evidence? | Graph-held-out ridge forecasts from ordered partial trajectories | Held-out final-margin R2 = 0.393-0.714 and AUROC = 0.837-0.984 at the earliest order-informative cutoff | Candidate-defined reduced-order forecast, not an autonomous state equation |
| Is the geometry trained, and is the declared output boundary reachable across checkpoints? | Three random initializations per architecture; confidence-gated full-vocabulary top-1 intervention | Random order gains exceeded trained gains in all 9 initializations; guarded flips were 17/87, 13/87 and 9/87 | Architecture-compatible scaffold; candidate crossing is not correctness; matched reassignments do not establish fine-grained dose specificity |

### Reproduction levels

- **Level 1, numerical audit:** inspect CSV/JSON tables and manifests; no model weights required.
- **Level 2, figure regeneration:** run the deterministic R scripts from the supplied source tables.
- **Level 3, processed-input rerun:** run the portable Python analyses with regenerated trajectory tables.
- **Level 4, full extraction:** obtain the cited checkpoints and regenerate hidden-state arrays locally.

The repository does not redistribute Qwen, Llama or Gemma weights or large hidden-state derivatives. Upstream model and dataset licences continue to apply.

### Interpretation boundaries

The controlled tasks are mechanism experiments, not capability leaderboards. Vocabulary readbacks are diagnostics. `DeltaU` is a task-constructed candidate-path separation observable, not an energy or universal order parameter. Chord-directed geometry is measured relative to endpoint-preserving layer-order nulls and is not an exact geodesic theorem. The intervention establishes bounded trajectory and candidate-boundary control, not general correctness, open-ended generation control or deployment safety.

## 中文

本仓库是论文 **High-dimensional state trajectories reveal transformer inference dynamics** 的审计版复现包。当前版本为 `v0.52`，新增三项实验闭合：跨任务与四选项泛化、held-out 局部轨迹预测、三架构随机初始化与输出候选边界控制。

建议先阅读 `source_data/source_data/extension_v0_52/README.md`，再根据 `manifest.csv` 核验文件哈希；复跑入口见 `code/extension_v0_52/README.md`。仓库不重分发模型权重或大型隐藏状态数组。当前证据支持层顺序几何的跨任务复现、受边界约束的候选 margin 预测、架构兼容的随机态顺序脚手架，以及三检查点中声明候选边界的可达性；不支持普适标量坐标、自治状态方程、通用正确性控制、部署安全或精细逐轨迹剂量特异性。

Contact: Heng Zhao <zhaoheng1990@gmail.com>; ORCID <https://orcid.org/0009-0004-2395-9393>.
