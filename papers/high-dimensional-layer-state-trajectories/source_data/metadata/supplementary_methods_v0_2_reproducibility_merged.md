# Supplementary Methods V0.2 reproducibility draft

## Purpose and scope

This Supplementary Methods draft consolidates the reproducibility updates recovered after `supplementary_methods_v0_1.md`. It supports the manuscript *High-dimensional layer-state trajectories reveal the dynamics of transformer inference* and should be used as the current Methods metadata basis before final journal formatting.

The direct empirical scope is inference-stage trajectory dynamics across Qwen, Llama and Gemma. Training-shaped semantic structure, prompt-conditioned initial conditions and local transformer unfolding are discussed as the broader landscape, but the present experiments directly test the inference-stage layer-state trajectory process.

## Model checkpoints

Analyses used three instruction-tuned model families. Model weights are not redistributed and remain subject to provider access conditions.

| Model family | Checkpoint | Revision/SHA | Licence tag | Context/config note |
|---|---|---|---|---|
| Qwen | `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | `apache-2.0` | Public config reports `max_position_embeddings=32768`; tokenizer config reports `model_max_length=131072`. |
| Llama | `meta-llama/Llama-3.2-1B-Instruct` | `9213176726f574b556790deb65791e0c5aa438b6` | `llama3.2` | Hugging Face metadata is public but files are manually gated; local config reports `max_position_embeddings=131072`. |
| Gemma | `google/gemma-2-2b-it` | `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8` | `gemma` | Hugging Face metadata is public but files are manually gated; local config reports `max_position_embeddings=8192` and `sliding_window=4096`. |

Checkpoint metadata was checked on `2026-06-21`. The machine-readable table is `model_checkpoints_v0_2.csv`.

## Prompt inventory

The prompt-level inventory contains 96 prompts across 12 prompt conditions and three model families. Prompt metadata are stored in `prompt_table_v0_1.csv` and summarized in `prompt_condition_summary_v0_1.csv`. Prompt conditions are used as experimental ontology labels and should not be treated as natural-language task categories alone.

## Hidden-state trajectory extraction

For the OPA-1A-lite artifacts reused by OPA-1B, hidden-state tensors were extracted using Hugging Face model outputs with `output_hidden_states=True` and `use_cache=False`. For each batch item and transformer layer index `l`, the stored vector was:

```text
h_{m,p,l} = outputs.hidden_states[l + 1][batch_index, last_nonpadding_token_position, :]
```

The `l + 1` indexing excludes the embedding hidden state. The token position was the last non-padding token from the attention mask, with tokenizer padding side set to left. Models were loaded as float16 on CUDA or float32 on CPU; saved hidden-state arrays were cast to float32. Extraction used `MAX_LEN = 320`, `BATCH_SIZE = 4`, `SEED = 42` and `USE_CHAT_TEMPLATE = False`.

| Model family | Hidden-state artifact | Shape | Observed OPA-1B layers |
|---|---|---:|---|
| Qwen | `qwen_hidden_last_token_layers.npy` | `[960,28,1536]` | 7-25 |
| Llama | `llama_hidden_last_token_layers.npy` | `[960,16,2048]` | 4-14 |
| Gemma | `gemma_hidden_last_token_layers.npy` | `[960,26,2304]` | 6-23 |

OPA-1B used the first 96 prompts from these 960-row artifacts. The machine-readable extraction table is `hidden_state_extraction_table_v0_2.csv`.

## Projection-selection geometry controls

Projection-selection geometry controls compared real output weights with row-permuted weights and Gaussian projections. Row-permutation controls permuted vocabulary rows. Gaussian controls matched the analysed real projection matrix by per-dimension mean and standard deviation, then rescaled row norms to the real row-norm distribution.

OPA-1B used five repeats for rowpermW and gaussianR controls and `RANDOM_SEED = 42`. Projection preservation was computed as Pearson/Spearman correlation between pairwise cosine-distance vectors in hidden-state space and readback-center space.

Readbacks are diagnostic observables. These controls do not imply that vocabulary-facing coordinates exhaust the mechanism coordinate system.

## Delta U and CM advantage-flow analyses

CM-5A and CM-5B used a clean/conflict label readback margin:

```text
R_l = logit_l(clean_label) - logit_l(conflict_label).
```

For each graph, the clean-anchor trajectory was subtracted:

```text
dR_l = R_l(condition) - R_l(clean_anchor).
```

Decision layers were selected from `DECISION_FRAC = (0.70, 0.92)`:

```text
decision_layers = floor(0.70 * L) ... floor(0.92 * L).
```

Delta U was PC1 of the clean-relative decision-layer matrix:

```text
DeltaU = PC1(dR over decision_layers).
```

The sign was flipped when the stable mean was lower than the closure mean, so the displayed orientation separated stable and closure trajectories consistently.

CM-5A/5B used an observable layerwise advantage-flow proxy:

```text
O_adv_l = dR_{l+1} - dR_l.
```

Window-level `O_adv` features were evaluated with graph-grouped cross-validation. Delta U prediction used standardized Ridge regression with `alpha=10.0`. Mechanism classification used standardized logistic regression with `max_iter=1000` and `class_weight="balanced"`. The grouped split unit was `graph_id`.

CM-5A evaluated broader O-advantage windows, including all transitions and model-specific initialization-to-decision windows. CM-5B restricted predictors to predecision transitions. CM-5C used the same decision-window Delta U target but predicted it from predecision TopK/VIM neighborhood transition features rather than `dR` differences.

`O_adv` is an observable proxy for path-advantage growth. It should not be conflated with PhaseMap `O_l^cont`.

## UA-4B band control

UA-4B defined a global Delta U band from negative, critical and positive reference means:

```text
U_minus = (u_neg + u_crit) / 2
U_plus  = (u_crit + u_pos) / 2.
```

Rows were assigned to below-band, in-band or above-band phases. Figure 3d reports phase accuracy and macro-F1 from this global band midpoint classifier. This is an auxiliary band-control check and not the primary ontology.

## PhaseMap state-equation closure

PhaseMap-7E audited one-step next-layer state-equation closure. The target vector was:

```text
y_{l+1} = (R_l1, boundary_dist_l1, spread_l1, rank_gap_l1).
```

The regression model was `Ridge(alpha=10.0, random_state=42)` inside a preprocessing pipeline. Numeric predictors were median-imputed and standardized. Categorical predictors were most-frequent-imputed and one-hot encoded. Cross-validation used `GroupKFold(n_splits=5)` by `graph_id` when enough graph groups were available; otherwise the script fell back to shuffled `KFold(n_splits=5, random_state=42)`.

The main feature groups were:

| Group | Features |
|---|---|
| R | `R_l`, `rank_gap_l` |
| B | `boundary_dist_l` |
| T | `spread_l` |
| H-shape | `R_prev_delta`, `rank_gap_prev_delta`, `R_velocity_reversal`, `rank_gap_reversal`, `center_backtrack_prevprev`, `center_backtrack_init` |
| `O_l^cont` | `op_pc1` to `op_pc10` |

The reported score is macro `R^2`:

```text
R2_macro = r2_score(y_true, y_pred, multioutput = "uniform_average").
```

`op_pc1-op_pc10` are latent transition/operator coordinates derived from transition signatures. They support a local `F_O` state-equation description but should not be described as pure pre-transition observables.

## GV geodesic-bias and residual analyses

GV-1B is implemented in `GPT_171_GV_1B.py`. It tests whether real TopK/VIM trajectories are biased toward a dominant endpoint or geodesic direction, while allowing structured residuals. It does not claim strict equality between the real trajectory and a shortest path.

Figure 4d display metrics are:

```text
Alignment gain = bias_real_align_mean - bias_shuffle_align_mean
Detour reduction = log2(bias_shuffle_detour_mean / bias_real_detour_mean)
Curvature reduction = log2(bias_shuffle_curvature_mean / bias_real_curvature_mean).
```

GV-2 is implemented in `GPT_172_GV_2.py`. It compares residual-only, projection-only and mixed projection-residual feature groups. Figure 4e reports:

```text
Projection = best_projection_deltaU_corr
Residual   = best_resid_deltaU_corr
Mixed      = best_mixed_deltaU_corr.
```

GV-3 is implemented in `GPT_173_GV_3.py`. It defines proxy feature families for geodesic-parallel and structured-perpendicular components. Figure 4f reports:

```text
Parallel      = best_parallel_deltaU_corr
Perpendicular = best_perp_deltaU_corr
Both          = best_both_deltaU_corr.
```

Parallel and perpendicular labels are proxy component families. They should not be described as literal orthogonal decomposition in the full hidden-state manifold.

## ASA-9A.1 closed-loop steering replication

ASA-9A.1 used `seed = 42`, `n_graphs = 96`, `batch_size = 4`, `test_size = 0.30` and `GroupShuffleSplit` by `graph_id`. Layer windows were model-specific:

| Model family | Operator layers | Precursor layers | Decision layers |
|---|---|---|---|
| Qwen | 15-19 | 17-19 | 20-25 |
| Llama | 8-11 | 10-12 | 12-15 |
| Gemma | 13-17 | 15-17 | 18-23 |

Mechanism classifiers used `du_hat`, `abs_du_hat`, `db_hat`, `vel_norm`, `vel_z`, `layer_norm`, `is_precursor_layer`, `baseline_DeltaU` and `baseline_DB`. The classifier was `RandomForestClassifier(n_estimators=250, max_depth=7, min_samples_leaf=6, class_weight=balanced_subsample, random_state=42+shuffle_id)`.

Policy classifiers used the mechanism features plus `p_mech_stable`, `p_mech_competition`, `p_mech_closure`, `pred_mech_stable`, `pred_mech_competition` and `pred_mech_closure`. The classifier was `RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=8, class_weight=balanced_subsample, random_state=42+100+shuffle_id)`.

The policy action grid was:

```text
(0.0,0.0), (0.1,0.1), (0.2,0.2), (0.3,0.3),
(0.3,0.0), (0.0,0.3), (0.3,0.2), (0.2,0.3).
```

The fixed same-combo baseline used `alpha = 0.30` and `beta = 0.30`.

Specificity was computed as:

```text
shift = DeltaU - DeltaU_base
specificity = mean(abs(shift) | mechanism = closure)
              - mean(abs(shift) | mechanism != closure).
```

The 50-shuffle replication used three matched null families: `shuffle_mech_policy`, `shuffle_policy_label` and `shuffle_both`. Mechanism-label shuffles used seed `42 + 1000 + shuffle_id`; policy-label shuffles used seed `42 + 2000 + shuffle_id`. For `shuffle_both`, the implementation used `shuffle_id = 1000 + s` before applying the same offsets.

Closed-loop steering is interpreted as internal trajectory-dominance modulation under classifier and shuffle-null controls, not task-level correctness or deployment control.

## Software environments

Python GPU/model-side analyses used the `dhrf_4080s` Conda environment: Python 3.11.15, PyTorch 2.5.1+cu121, CUDA 12.1, Transformers 5.9.0, NumPy 2.4.4, pandas 3.0.3, scikit-learn 1.8.0 and SciPy 1.17.1. CUDA was available on one NVIDIA GeForce RTX 4080 SUPER GPU.

R figure rendering used the `nature-r-fig` Conda environment: R 4.3.3 with ggplot2 3.5.2, ggforce 0.5.0, ggnewscale 0.5.2, patchwork 1.3.2, dplyr 1.1.4, tidyr 1.3.1, scales 1.4.0, svglite 2.2.1, ragg 1.5.0, readr 2.1.5, ggrepel 0.9.6 and cowplot 1.2.0. R figure export validation passed with `verify_r_figure_env.R`.

## Repository and code status

The first repository staging folder is `repository_staging_v0_1/`. It contains processed Source Data, metadata, raw analysis scripts, figure scripts and environment manifests. Raw analysis scripts are retained for provenance and have been converted, where workstation paths were detected, to repository-relative, argument-driven entry points or preflightable wrappers. The public release still requires final repository identifiers, licences, upstream data decisions and figure-regeneration dependency decisions.

The current staging inventory is `repository_staging_v0_1/metadata/repository_staging_inventory_v0_1.csv`.

## Claim boundaries

Delta U is a bounded dominance observable, not the full mechanism. Readbacks are diagnostic observables, not mechanism coordinates. `O_adv` and `O_l^cont` are distinct objects. Low-dimensional visualizations are orientation aids. GV supports geodesic-biased flow with structured residuals, not exact geodesic motion. ASA steering supports internal trajectory-dominance modulation, not guaranteed task correctness.
