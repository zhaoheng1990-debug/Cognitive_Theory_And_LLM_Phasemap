library(ggplot2)
library(ggforce)
library(ggnewscale)
library(patchwork)
library(dplyr)
library(svglite)
library(ragg)
library(grid)

args <- commandArgs(trailingOnly = TRUE)
root_dir <- if (length(args) >= 1) args[[1]] else file.path(getwd(), "outputs")
out_dir <- file.path(root_dir, "nature_r_mechanism_figure_env_validation")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

palette <- c(
  neutral_dark = "#272727",
  neutral_mid = "#767676",
  neutral_light = "#D8D8D8",
  neutral_pale = "#F4F4F4",
  signal_teal = "#33B5A5",
  signal_blue = "#3182BD",
  signal_violet = "#7A6BB7",
  accent_red = "#D24B40",
  accent_orange = "#E28E2C"
)

save_pub <- function(plot, stem, width_mm = 183, height_mm = 92, dpi = 600) {
  w <- width_mm / 25.4
  h <- height_mm / 25.4
  svglite::svglite(paste0(stem, ".svg"), width = w, height = h)
  print(plot)
  dev.off()
  grDevices::cairo_pdf(paste0(stem, ".pdf"), width = w, height = h, family = "Arial")
  print(plot)
  dev.off()
  ragg::agg_tiff(paste0(stem, ".tiff"), width = w, height = h, units = "in", res = dpi)
  print(plot)
  dev.off()
  ragg::agg_png(paste0(stem, ".png"), width = w, height = h, units = "in", res = 300)
  print(plot)
  dev.off()
}

theme_mechanism <- theme_void(base_family = "Arial") +
  theme(
    plot.title = element_text(size = 8.2, face = "bold", colour = palette["neutral_dark"]),
    plot.subtitle = element_text(size = 6.1, colour = palette["neutral_mid"]),
    plot.margin = margin(4, 5, 4, 5),
    legend.position = "none"
  )

field <- data.frame(
  x0 = c(0.18, 0.52, 0.76),
  y0 = c(0.54, 0.55, 0.50),
  a = c(0.13, 0.20, 0.16),
  b = c(0.09, 0.14, 0.10),
  angle = c(-0.25, 0.12, -0.18),
  field = c("local neighbourhood", "dominance basin", "commitment region")
) |>
  mutate(field = factor(field, levels = c("local neighbourhood", "dominance basin", "commitment region")))

trajectory <- data.frame(
  group = rep(c("candidate A", "candidate B", "dominant path"), each = 4),
  x = c(
    0.12, 0.25, 0.40, 0.58,
    0.12, 0.28, 0.45, 0.66,
    0.12, 0.31, 0.55, 0.83
  ),
  y = c(
    0.55, 0.66, 0.74, 0.70,
    0.55, 0.49, 0.37, 0.34,
    0.55, 0.58, 0.54, 0.52
  ),
  width = rep(c(0.45, 0.45, 1.25), each = 4)
) |>
  mutate(group = factor(group, levels = c("candidate A", "candidate B", "dominant path")))

readback <- data.frame(
  x = c(0.40, 0.55, 0.70),
  y = c(0.70, 0.54, 0.52),
  xend = c(0.40, 0.55, 0.70),
  yend = c(0.24, 0.22, 0.26)
)

p <- ggplot() +
  ggforce::geom_ellipse(
    data = field,
    aes(x0 = x0, y0 = y0, a = a, b = b, angle = angle, fill = field),
    alpha = 0.22,
    colour = NA
  ) +
  scale_fill_manual(values = c(
    "local neighbourhood" = unname(palette["signal_blue"]),
    "dominance basin" = unname(palette["signal_teal"]),
    "commitment region" = unname(palette["signal_violet"])
  )) +
  ggnewscale::new_scale_colour() +
  ggforce::geom_bezier(
    data = trajectory,
    aes(x = x, y = y, group = group, colour = group, linewidth = width),
    lineend = "round"
  ) +
  scale_colour_manual(values = c(
    "candidate A" = unname(palette["neutral_mid"]),
    "candidate B" = unname(palette["neutral_mid"]),
    "dominant path" = unname(palette["signal_teal"])
  )) +
  scale_linewidth_identity() +
  geom_segment(
    data = readback,
    aes(x = x, y = y, xend = xend, yend = yend),
    linetype = "dashed",
    linewidth = 0.35,
    colour = palette["neutral_mid"],
    arrow = arrow(length = unit(1.4, "mm"), type = "closed")
  ) +
  annotate(
    "curve",
    x = 0.86, y = 0.31, xend = 0.68, yend = 0.48,
    curvature = 0.35,
    linewidth = 0.46,
    colour = palette["accent_orange"],
    arrow = arrow(length = unit(1.7, "mm"), type = "closed")
  ) +
  annotate("segment", x = 0.45, xend = 0.45, y = 0.74, yend = 0.37,
           linewidth = 0.35, colour = palette["accent_red"],
           arrow = arrow(length = unit(1.4, "mm"), ends = "both", type = "closed")) +
  annotate("text", x = 0.47, y = 0.56, label = "Delta U", hjust = 0,
           size = 2.2, colour = palette["accent_red"]) +
  annotate("label", x = 0.12, y = 0.55, label = "Prompt",
           size = 2.2, label.size = 0.18, label.r = unit(1.6, "mm"),
           fill = palette["neutral_pale"], colour = palette["neutral_dark"]) +
  annotate("text", x = 0.55, y = 0.16, label = "dashed probes: readback observables",
           size = 2.0, colour = palette["neutral_mid"]) +
  annotate("text", x = 0.86, y = 0.26, label = "steering feedback",
           size = 2.0, colour = palette["accent_orange"]) +
  coord_cartesian(xlim = c(0.02, 0.98), ylim = c(0.08, 0.88), expand = FALSE) +
  labs(
    title = "Mechanism visual grammar validation",
    subtitle = "R stack supports curved trajectories, local fields, readback probes, Delta U gaps and feedback arcs."
  ) +
  theme_mechanism

stem <- file.path(out_dir, "mechanism_visual_grammar_validation")
save_pub(p, stem)

pkgs <- c(
  "ggplot2", "ggforce", "ggnewscale", "patchwork",
  "dplyr", "svglite", "ragg", "grid", "colorspace", "systemfonts"
)
status <- data.frame(
  package = pkgs,
  available = vapply(pkgs, requireNamespace, logical(1), quietly = TRUE),
  stringsAsFactors = FALSE
)
write.csv(status, file.path(out_dir, "mechanism_r_package_status.csv"), row.names = FALSE)

outputs <- list.files(out_dir, pattern = "\\.(svg|pdf|tiff|png)$", full.names = TRUE)
qa <- data.frame(
  file = basename(outputs),
  format = tools::file_ext(outputs),
  bytes = file.info(outputs)$size,
  export_pass = file.info(outputs)$size > 1000,
  stringsAsFactors = FALSE
)
write.csv(qa, file.path(out_dir, "mechanism_export_qa.csv"), row.names = FALSE)

message("Mechanism R figure environment validated in: ", out_dir)
