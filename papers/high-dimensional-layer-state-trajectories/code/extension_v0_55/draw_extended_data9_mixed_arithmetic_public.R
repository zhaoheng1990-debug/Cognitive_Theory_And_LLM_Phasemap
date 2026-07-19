library(ggplot2)
library(patchwork)
library(dplyr)
library(scales)
library(grid)

args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) >= 1) args[[1]] else file.path("source_data", "source_data", "extension_v0_55", "figure_source_data")
out_dir <- if (length(args) >= 2) args[[2]] else "."
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

palette <- c(
  neutral_dark = "#272727", neutral_mid = "#767676", neutral_light = "#D8D8D8",
  neutral_pale = "#F2F2F2", qwen = "#3182BD", llama = "#33A695",
  gemma = "#7A6BB7", accent = "#D9822B", caution = "#C84B40"
)
model_levels <- c("Qwen", "Llama", "Gemma")
model_cols <- c(Qwen = palette[["qwen"]], Llama = palette[["llama"]], Gemma = palette[["gemma"]])

theme_nature <- function(base_size = 6.5, base_family = "Arial") {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      axis.line = element_line(linewidth = 0.32, colour = "black"),
      axis.ticks = element_line(linewidth = 0.28, colour = "black"),
      axis.title = element_text(size = base_size),
      axis.text = element_text(size = base_size - 0.35, colour = "black"),
      legend.title = element_text(size = base_size - 0.25),
      legend.text = element_text(size = base_size - 0.55),
      legend.key.height = unit(3.0, "mm"),
      legend.key.width = unit(4.0, "mm"),
      legend.background = element_blank(),
      strip.background = element_blank(),
      strip.text = element_text(size = base_size - 0.05, face = "bold"),
      plot.title = element_text(size = base_size + 0.45, face = "bold", hjust = 0),
      plot.subtitle = element_text(size = base_size - 0.45, colour = palette[["neutral_mid"]], hjust = 0),
      plot.margin = margin(3, 4, 3, 4), panel.grid = element_blank()
    )
}
theme_set(theme_nature())

primary <- read.csv(file.path(data_dir, "extended_data9_primary_summary.csv"), check.names = FALSE) |>
  mutate(
    checkpoint = factor(checkpoint, levels = model_levels),
    crossing_label = sprintf("%d/%d", strict_crossings, eligible_target_prompts)
  )
null_long <- read.csv(file.path(data_dir, "extended_data9_address_reassignment_nulls.csv"), check.names = FALSE) |>
  mutate(group = factor(group, levels = c(model_levels, "Pooled")))
null_stats <- read.csv(file.path(data_dir, "extended_data9_address_null_summary.csv"), check.names = FALSE) |>
  mutate(group = factor(group, levels = c(model_levels, "Pooled")))
real_specificity <- null_stats |>
  transmute(group, specificity)
controls <- read.csv(file.path(data_dir, "extended_data9_component_controls.csv"), check.names = FALSE) |>
  mutate(
    checkpoint = factor(checkpoint, levels = model_levels),
    action = factor(action, levels = c("Primary", "Operator", "Precursor", "Uniform max")),
    metric = factor(metric, levels = c("Target crossing", "Non-target change"))
  )

p_a <- ggplot(primary, aes(y = checkpoint)) +
  geom_col(aes(x = eligible_target_prompts), width = 0.58, fill = palette[["neutral_light"]]) +
  geom_col(aes(x = strict_crossings, fill = checkpoint), width = 0.58) +
  geom_text(aes(x = eligible_target_prompts + 1.0, label = crossing_label), hjust = 0, size = 2.15, family = "Arial") +
  geom_text(
    data = filter(primary, model_level_gate != "Pass"),
    aes(x = 35.0, label = ifelse(model_level_gate == "Underpowered", "n < 5", "median < 0")),
    hjust = 1, size = 1.85, colour = palette[["caution"]], family = "Arial"
  ) +
  scale_fill_manual(values = model_cols, guide = "none") +
  scale_x_continuous(limits = c(0, 36), breaks = c(0, 10, 20, 30), expand = expansion(mult = c(0, 0))) +
  labs(title = "Sparse crossings with uneven power", subtitle = "strict crossings / baseline-eligible target prompts", x = "Number of held-out prompts", y = NULL) +
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank())

p_b <- ggplot(primary, aes(y = checkpoint, colour = checkpoint)) +
  geom_vline(xintercept = 0, linewidth = 0.35, linetype = "22", colour = palette[["neutral_mid"]]) +
  geom_segment(aes(x = 0, xend = median_candidate_margin_shift, yend = checkpoint), linewidth = 0.75) +
  geom_point(aes(x = median_candidate_margin_shift), size = 2.15) +
  geom_text(
    aes(x = median_candidate_margin_shift, label = sprintf("%+.3f", median_candidate_margin_shift),
        hjust = ifelse(median_candidate_margin_shift >= 0, -0.25, 1.25)),
    size = 2.05, colour = palette[["neutral_dark"]], family = "Arial"
  ) +
  annotate("text", x = 0.60, y = 3, label = "underpowered", size = 1.85, colour = palette[["caution"]], family = "Arial") +
  scale_colour_manual(values = model_cols, guide = "none") +
  scale_x_continuous(limits = c(-0.40, 0.75), breaks = c(-0.4, 0, 0.4), expand = expansion(mult = 0)) +
  labs(title = "Only Qwen passes the directional gate", subtitle = "median consistent-minus-distractor margin shift in eligible prompts", x = "Candidate-margin shift", y = NULL) +
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank())

p_c <- ggplot(null_long, aes(x = specificity, y = group)) +
  geom_violin(scale = "width", width = 0.72, fill = palette[["neutral_pale"]], colour = palette[["neutral_mid"]], linewidth = 0.28, trim = TRUE) +
  geom_point(position = position_jitter(height = 0.10, width = 0, seed = 20260720), size = 0.60, alpha = 0.34, colour = palette[["neutral_mid"]]) +
  geom_point(data = real_specificity, aes(x = specificity, y = group, colour = "Real"), shape = 18, size = 2.45) +
  geom_point(data = null_stats, aes(x = null_q95, y = group, colour = "Null 95%"), shape = 124, size = 3.4, stroke = 0.8) +
  annotate("text", x = 0.071, y = 4.28, label = "pooled P = 1/51", hjust = 0, size = 1.95, family = "Arial") +
  scale_colour_manual(values = c(Real = palette[["qwen"]], `Null 95%` = palette[["accent"]]), name = NULL) +
  scale_x_continuous(limits = c(-0.15, 0.25), breaks = c(-0.1, 0, 0.1, 0.2), expand = expansion(mult = c(0, 0.02))) +
  labs(title = "Pooled selectivity exceeds address null", subtitle = "specificity = target crossing rate - non-target change rate", x = "Specificity", y = NULL) +
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(), legend.position = "top", legend.justification = "left", legend.margin = margin(0, 0, 0, 0))

p_d <- ggplot(controls, aes(x = action, y = checkpoint, fill = value)) +
  geom_tile(colour = "white", linewidth = 0.7) +
  geom_text(aes(label = count), size = 1.90, family = "Arial") +
  facet_wrap(~metric, nrow = 1) +
  scale_fill_gradient(low = palette[["neutral_pale"]], high = palette[["qwen"]], limits = c(0, 0.25), oob = squish) +
  labs(title = "Operator carries the two primary crossings", subtitle = "cells show crossing/eligible or changed/non-target counts", x = NULL, y = NULL, fill = "Rate") +
  theme(axis.line = element_blank(), axis.ticks = element_blank(), axis.text.x = element_text(angle = 28, hjust = 1), legend.position = "right", panel.spacing = unit(2.5, "mm"))

figure <- ((p_a | p_b) / (p_c | p_d)) +
  plot_layout(heights = c(0.88, 1.12), widths = c(0.93, 1.07), guides = "keep") +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 8, face = "bold", family = "Arial"))

stem <- file.path(out_dir, "extended_data9_mixed_arithmetic_actuator_transfer")
w <- 183 / 25.4
h <- 118 / 25.4
svglite::svglite(paste0(stem, ".svg"), width = w, height = h); print(figure); dev.off()
grDevices::cairo_pdf(paste0(stem, ".pdf"), width = w, height = h, family = "Arial"); print(figure); dev.off()
ragg::agg_tiff(paste0(stem, ".tiff"), width = w, height = h, units = "in", res = 600); print(figure); dev.off()
ragg::agg_png(paste0(stem, ".png"), width = w, height = h, units = "in", res = 300); print(figure); dev.off()
