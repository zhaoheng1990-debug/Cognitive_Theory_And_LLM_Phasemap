# R figure environment manifest V0.1

## Scope

This manifest records the R environment used to render the Nature-style manuscript figures.

## Environment

| Field | Value |
|---|---|
| Conda environment | `nature-r-fig` |
| R | `4.3.3 (2024-02-29 ucrt)` |
| Validation script | `verify_r_figure_env.R` |
| Validation status | passed |

## Core packages

| Package | Version |
|---|---|
| `ggplot2` | `3.5.2` |
| `ggforce` | `0.5.0` |
| `ggnewscale` | `0.5.2` |
| `patchwork` | `1.3.2` |
| `dplyr` | `1.1.4` |
| `tidyr` | `1.3.1` |
| `scales` | `1.4.0` |
| `svglite` | `2.2.1` |
| `ragg` | `1.5.0` |
| `readr` | `2.1.5` |
| `ggrepel` | `0.9.6` |
| `cowplot` | `1.2.0` |

## Execution note

Use the R figure environment for figure regeneration:

```powershell
conda run -n nature-r-fig Rscript draw_manuscript_figures_v0_6_merged_r.R
```

The environment was validated by running:

```powershell
conda run -n nature-r-fig Rscript verify_r_figure_env.R
```
