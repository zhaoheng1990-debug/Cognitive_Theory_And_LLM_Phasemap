options(stringsAsFactors = FALSE)

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
  library(scales)
  library(grid)
})

project_dir <- normalizePath(".", winslash = "/", mustWork = TRUE)
root_dir <- file.path(project_dir, "output")
source_dir <- file.path(project_dir, "inputs", "summary_tables")
output_dir <- file.path(root_dir, "three_closure_figures")
panel_source_dir <- file.path(output_dir, "source_data")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(panel_source_dir, recursive = TRUE, showWarnings = FALSE)

model_levels <- c("qwen", "llama", "gemma")
model_labels <- c(qwen = "Qwen", llama = "Llama", gemma = "Gemma")
model_colours <- c(qwen = "#3182BD", llama = "#33B5A5", gemma = "#7A6BB7")
neutral_dark <- "#272727"
neutral_mid <- "#767676"
neutral_light <- "#D8D8D8"
signal_orange <- "#E28E2C"
signal_green <- "#2E9E44"

theme_nature <- function(base_size = 6.4) {
  theme_classic(base_size = base_size, base_family = "Arial") +
    theme(
      axis.line = element_line(linewidth = 0.32, colour = "black"),
      axis.ticks = element_line(linewidth = 0.28, colour = "black"),
      axis.title = element_text(size = base_size),
      axis.text = element_text(size = base_size - 0.5, colour = "black"),
      strip.background = element_blank(),
      strip.text = element_text(size = base_size - 0.2, face = "bold"),
      legend.title = element_text(size = base_size - 0.4),
      legend.text = element_text(size = base_size - 0.7),
      legend.key.height = unit(3.2, "mm"),
      legend.key.width = unit(4.2, "mm"),
      plot.title = element_text(size = base_size + 0.6, face = "bold", margin = margin(b = 2)),
      plot.subtitle = element_text(size = base_size - 0.7, colour = "#555555", margin = margin(b = 3)),
      panel.grid = element_blank(),
      plot.margin = margin(4, 5, 4, 5)
    )
}

write_source <- function(data, filename) {
  write.csv(data, file.path(panel_source_dir, filename), row.names = FALSE)
}

geometry <- read.csv(
  file.path(source_dir, "task_geometry_generalization.csv"),
  check.names = FALSE
) %>%
  mutate(
    model = factor(model, levels = model_levels),
    model_label = factor(model_labels[as.character(model)], levels = model_labels[model_levels]),
    task_short = case_when(
      grepl("lexical", task) ~ "Lexical",
      grepl("addition", task) ~ "Addition",
      TRUE ~ "ARC-Challenge"
    )
  )

forecast <- read.csv(
  file.path(source_dir, "heldout_predictive_forecasting.csv"),
  check.names = FALSE
) %>%
  mutate(
    model = factor(model, levels = model_levels),
    model_label = factor(model_labels[as.character(model)], levels = model_labels[model_levels]),
    is_earliest_order_informative_cutoff = as.logical(is_earliest_order_informative_cutoff)
  )

initialization <- read.csv(
  file.path(source_dir, "crossmodel_initialization_geometry.csv"),
  check.names = FALSE
) %>%
  mutate(
    model = factor(model, levels = model_levels),
    model_label = factor(model_labels[as.character(model)], levels = rev(model_labels[model_levels]))
  )

boundary <- read.csv(
  file.path(source_dir, "crossmodel_output_boundary.csv"),
  check.names = FALSE
) %>%
  mutate(
    model = factor(model, levels = model_levels),
    model_label = factor(model_labels[as.character(model)], levels = model_labels[model_levels])
  )

assignment <- read.csv(
  file.path(source_dir, "crossmodel_assignment_matched_nulls.csv"),
  check.names = FALSE
) %>%
  mutate(
    model = factor(model, levels = model_levels),
    model_label = factor(model_labels[as.character(model)], levels = model_labels[model_levels]),
    p_label = paste0("P=", sprintf("%.3f", empirical_one_sided_p))
  )

controlled_geometry <- geometry %>% filter(task_short != "ARC-Challenge")
arc_geometry <- geometry %>% filter(task_short == "ARC-Challenge")

p_a <- ggplot(
  controlled_geometry,
  aes(alignment_z, task_short, colour = model, group = model)
) +
  geom_vline(xintercept = 0, linewidth = 0.35, colour = neutral_mid) +
  geom_point(
    position = position_dodge(width = 0.48),
    size = 2.25
  ) +
  scale_colour_manual(values = model_colours, labels = model_labels, name = NULL) +
  coord_cartesian(xlim = c(0, 8)) +
  labs(
    title = "Order geometry recurs across task families",
    subtitle = "real versus 50 endpoint-preserving shuffles",
    x = "Null-standardized alignment",
    y = NULL
  ) +
  theme_nature() +
  theme(
    axis.line.y = element_blank(),
    axis.ticks.y = element_blank(),
    legend.position = "bottom"
  )

p_b <- ggplot(arc_geometry, aes(y = model_label)) +
  geom_segment(
    aes(x = shuffle_alignment_mean, xend = real_alignment_mean, yend = model_label),
    linewidth = 1.0,
    colour = neutral_light,
    lineend = "round"
  ) +
  geom_point(aes(x = shuffle_alignment_mean), shape = 21, size = 2.2, fill = "white", colour = neutral_mid) +
  geom_point(aes(x = real_alignment_mean, fill = model), shape = 21, size = 2.5, colour = "white") +
  geom_text(
    aes(
      x = real_alignment_mean + 0.006,
      label = paste0("accuracy ", sprintf("%.3f", behavioural_accuracy))
    ),
    hjust = 0,
    size = 1.75,
    colour = neutral_dark
  ) +
  scale_fill_manual(values = model_colours, guide = "none") +
  coord_cartesian(xlim = c(0.035, 0.158), clip = "off") +
  labs(
    title = "Four-choice structure preserves the order effect",
    subtitle = "96 ARC-Challenge validation items; no binary wrapper",
    x = "Mean chord alignment",
    y = NULL
  ) +
  theme_nature() +
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank())

p_c <- ggplot(forecast, aes(n_observed_layers, r2, colour = model)) +
  geom_line(linewidth = 0.65) +
  geom_point(size = 1.4) +
  geom_line(
    data = forecast %>% filter(!is.na(order_null_r2_q95)),
    aes(y = order_null_r2_q95),
    colour = neutral_mid,
    linewidth = 0.45,
    linetype = "22"
  ) +
  geom_point(
    data = forecast %>% filter(is_earliest_order_informative_cutoff),
    aes(fill = model),
    shape = 23,
    size = 2.6,
    stroke = 0.35,
    colour = "white"
  ) +
  facet_wrap(~ model_label, nrow = 1) +
  scale_colour_manual(values = model_colours, guide = "none") +
  scale_fill_manual(values = model_colours, guide = "none") +
  scale_x_continuous(breaks = 1:6) +
  coord_cartesian(ylim = c(-0.08, 1.02)) +
  labs(
    title = "Predictability emerges at checkpoint-specific depths",
    subtitle = "solid, held-out R2; dashed, order-null q95; diamond, earliest exceedance",
    x = "Observed decision-window layers",
    y = expression("Final-margin " * R^2)
  ) +
  theme_nature() +
  theme(panel.spacing = unit(2.6, "mm"))

p_d <- ggplot(initialization, aes(y = model_label)) +
  geom_segment(
    aes(
      x = trained_alignment_gain,
      xend = random_alignment_gain_mean,
      yend = model_label
    ),
    linewidth = 0.9,
    colour = neutral_light,
    lineend = "round"
  ) +
  geom_errorbarh(
    aes(xmin = random_alignment_gain_min, xmax = random_alignment_gain_max),
    height = 0.12,
    linewidth = 0.55,
    colour = neutral_mid
  ) +
  geom_point(
    aes(x = trained_alignment_gain),
    shape = 21,
    size = 2.2,
    fill = "white",
    colour = neutral_dark
  ) +
  geom_point(
    aes(x = random_alignment_gain_mean, fill = model),
    shape = 22,
    size = 2.4,
    colour = "white"
  ) +
  scale_fill_manual(values = model_colours, guide = "none") +
  coord_cartesian(xlim = c(0.025, 0.11)) +
  labs(
    title = "Random initializations preserve\nthe order scaffold",
    subtitle = "circle, trained; square and range, random mean and three-seed range",
    x = "Real-order alignment gain",
    y = NULL
  ) +
  theme_nature(base_size = 5.8) +
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank())

boundary_controls <- bind_rows(
  boundary %>% transmute(
    model, model_label, control = "Ungated policy",
    target_rate = ungated_strict_conflict_to_clean_top1 / heldout_closure_n,
    collateral_rate = ungated_nonclosure_top1_change_rate
  ),
  boundary %>% transmute(
    model, model_label, control = "Confidence-gated",
    target_rate = guarded_strict_conflict_to_clean_top1 / heldout_closure_n,
    collateral_rate = guarded_nonclosure_top1_change_rate
  ),
  boundary %>% transmute(
    model, model_label, control = "Shuffled policy",
    target_rate = shuffled_policy_strict_conflict_to_clean_top1 / heldout_closure_n,
    collateral_rate = shuffled_policy_nonclosure_top1_change_rate
  )
) %>%
  mutate(
    control = factor(control, levels = c("Ungated policy", "Confidence-gated", "Shuffled policy"))
  )

p_e <- ggplot(
  boundary_controls,
  aes(collateral_rate, target_rate, colour = model, shape = control)
) +
  geom_path(aes(group = model), linewidth = 0.45, alpha = 0.45) +
  geom_point(size = 2.3, stroke = 0.5) +
  scale_colour_manual(values = model_colours, labels = model_labels, name = NULL) +
  scale_shape_manual(
    values = c("Ungated policy" = 21, "Confidence-gated" = 23, "Shuffled policy" = 4),
    name = NULL
  ) +
  scale_x_continuous(labels = percent_format(accuracy = 1), breaks = c(0, 0.1, 0.2, 0.3)) +
  scale_y_continuous(labels = percent_format(accuracy = 1), breaks = c(0, 0.1, 0.2)) +
  coord_cartesian(xlim = c(-0.01, 0.38), ylim = c(-0.01, 0.24)) +
  labs(
    title = "A fixed gate limits\ncollateral output changes",
    subtitle = "full-vocabulary top-1; lines connect controls within checkpoint",
    x = "Non-closure change rate",
    y = "Closure conflict-to-clean rate"
  ) +
  theme_nature(base_size = 5.8) +
  theme(legend.position = "bottom")

qwen_null <- read.csv(
  file.path(
    project_dir,
    "inputs",
    "matched_nulls",
    "qwen",
    "matched_null_summary.csv"
  ),
  check.names = FALSE
) %>%
  transmute(model = "qwen", null_id, strict_crossings = crossings)

llama_null <- read.csv(
  file.path(project_dir, "inputs", "matched_nulls", "llama", "matched_null_summary.csv"),
  check.names = FALSE
) %>%
  transmute(model = "llama", null_id, strict_crossings)

gemma_null <- read.csv(
  file.path(project_dir, "inputs", "matched_nulls", "gemma", "matched_null_summary.csv"),
  check.names = FALSE
) %>%
  transmute(model = "gemma", null_id, strict_crossings)

matched_nulls <- bind_rows(qwen_null, llama_null, gemma_null) %>%
  mutate(
    model = factor(model, levels = model_levels),
    model_label = factor(model_labels[as.character(model)], levels = model_labels[model_levels])
  )

p_f <- ggplot(matched_nulls, aes(model_label, strict_crossings)) +
  geom_boxplot(
    width = 0.48,
    outlier.shape = NA,
    fill = neutral_light,
    colour = neutral_mid,
    linewidth = 0.4
  ) +
  geom_point(
    position = position_jitter(width = 0.13, height = 0, seed = 42),
    size = 0.75,
    colour = alpha(neutral_mid, 0.48)
  ) +
  geom_point(
    data = assignment,
    aes(model_label, real_guarded_strict_crossings, fill = model),
    shape = 23,
    size = 2.8,
    stroke = 0.4,
    colour = "white"
  ) +
  geom_text(
    data = assignment,
    aes(model_label, null_max_strict_crossings + 1.6, label = p_label),
    size = 1.75,
    colour = neutral_dark
  ) +
  scale_fill_manual(values = model_colours, guide = "none") +
  coord_cartesian(ylim = c(4.5, 22.5), clip = "off") +
  labs(
    title = "Matched reassignment does not resolve\naction specificity",
    subtitle = "50 safety-matched reassignments; diamond, observed",
    x = NULL,
    y = "Strict top-1 crossings"
  ) +
  theme_nature(base_size = 5.8)

write_source(controlled_geometry, "extended_data12_panel_a_task_geometry.csv")
write_source(arc_geometry, "extended_data12_panel_b_arc_geometry.csv")
write_source(forecast, "extended_data12_panel_c_predictive_forecast.csv")
write_source(initialization, "extended_data12_panel_d_initialization.csv")
write_source(boundary_controls, "extended_data12_panel_e_boundary_controls.csv")
write_source(matched_nulls, "extended_data12_panel_f_matched_nulls.csv")
write_source(assignment, "extended_data12_panel_f_real_assignments.csv")

top_row <- p_a | p_b
bottom_row <- (p_d | p_e | p_f) + plot_layout(widths = c(1.02, 1.08, 0.90))

figure <- top_row / p_c / bottom_row +
  plot_layout(heights = c(0.92, 1.08, 1.04), guides = "keep") +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 8, face = "bold", family = "Arial"))

width_in <- 183 / 25.4
height_in <- 180 / 25.4
base <- file.path(output_dir, "extended_data12_three_closures")

svglite::svglite(paste0(base, ".svg"), width = width_in, height = height_in)
print(figure)
dev.off()

grDevices::cairo_pdf(paste0(base, ".pdf"), width = width_in, height = height_in, family = "Arial")
print(figure)
dev.off()

ragg::agg_tiff(
  paste0(base, ".tiff"),
  width = width_in,
  height = height_in,
  units = "in",
  res = 600,
  compression = "lzw"
)
print(figure)
dev.off()

ragg::agg_png(
  paste0(base, ".png"),
  width = width_in,
  height = height_in,
  units = "in",
  res = 300
)
print(figure)
dev.off()

writeLines(
  c(
    "Extended Data Fig. 12: three experiment closures",
    "Backend: R only (ggplot2 + patchwork).",
    "Core conclusion: geometry transfers beyond binary prompts, ordered partial histories have bounded predictive value, the order effect is architecture-compatible, and the declared output boundary is reachable across checkpoints.",
    "Panel a: controlled lexical and addition tasks, three checkpoints, 50 order shuffles.",
    "Panel b: 96 four-choice ARC-Challenge validation items per checkpoint, no binary reduction.",
    "Panel c: 67 training and 29 held-out graph groups; 50 independent within-prompt order nulls.",
    "Panel d: trained checkpoints and three random initializations per architecture.",
    "Panel e: ungated, confidence-gated and shuffled-policy full-vocabulary endpoints.",
    "Panel f: 50 gate-, condition- and action-multiset-matched reassignments per checkpoint.",
    "Boundaries: task-specific coordinates are not a transferred DeltaU axis; prediction is candidate-defined; crossing is not answer correctness; action-assignment specificity is unsupported."
  ),
  file.path(output_dir, "README_extended_data12.md")
)
