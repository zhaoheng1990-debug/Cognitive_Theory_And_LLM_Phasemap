packages <- c(
  "ggplot2", "patchwork", "dplyr", "tidyr", "readr",
  "ggrepel", "svglite", "ragg", "scales", "cowplot"
)

missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  stop("Missing R packages: ", paste(missing, collapse = ", "))
}

library(ggplot2)
library(patchwork)

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[[1]] else file.path(getwd(), "outputs", "nature_r_figure_env_validation")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

palette_contract <- c(
  neutral_dark = "#272727",
  neutral_mid = "#767676",
  neutral_light = "#D8D8D8",
  signal_blue = "#3182BD",
  signal_teal = "#33B5A5",
  accent_red = "#D24B40",
  accent_orange = "#E28E2C"
)

theme_nature_contract <- function(base_size = 6.5, base_family = "Arial") {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = "black"),
      axis.ticks = element_line(linewidth = 0.35, colour = "black"),
      axis.title = element_text(size = base_size),
      axis.text = element_text(size = base_size - 0.5),
      legend.title = element_text(size = base_size - 0.3),
      legend.text = element_text(size = base_size - 0.7),
      strip.text = element_text(size = base_size - 0.3, face = "bold"),
      plot.title = element_text(size = base_size + 0.5, face = "bold"),
      panel.grid = element_blank()
    )
}

theme_set(theme_nature_contract())

save_pub_r <- function(plot, filename, width_mm = 89, height_mm = 65, dpi = 600) {
  w <- width_mm / 25.4
  h <- height_mm / 25.4
  svglite::svglite(paste0(filename, ".svg"), width = w, height = h)
  print(plot)
  dev.off()
  grDevices::cairo_pdf(paste0(filename, ".pdf"), width = w, height = h, family = "Arial")
  print(plot)
  dev.off()
  ragg::agg_tiff(paste0(filename, ".tiff"), width = w, height = h, units = "in", res = dpi)
  print(plot)
  dev.off()
}

df <- data.frame(
  group = factor(c("real", "shuffle", "fixed"), levels = c("real", "fixed", "shuffle")),
  value = c(1.00, 0.61, 0.74)
)

p <- ggplot(df, aes(group, value, fill = group)) +
  geom_col(width = 0.62, colour = "white", linewidth = 0.25) +
  geom_point(aes(y = value + 0.035), size = 0.75, colour = palette_contract["neutral_dark"]) +
  scale_fill_manual(values = c(
    real = unname(palette_contract["signal_teal"]),
    fixed = unname(palette_contract["signal_blue"]),
    shuffle = unname(palette_contract["neutral_light"])
  ), guide = "none") +
  scale_y_continuous(limits = c(0, 1.12), expand = c(0, 0)) +
  labs(x = NULL, y = "Validation score", title = "R export validation") +
  theme(plot.title = element_text(hjust = 0))

save_pub_r(p, file.path(out_dir, "r_export_validation"))

writeLines(c(
  paste0("R.version=", R.version.string),
  paste0("packages=", paste(packages, collapse = ",")),
  paste0("outputs=", paste(list.files(out_dir, pattern = "r_export_validation", full.names = TRUE), collapse = ";"))
), file.path(out_dir, "r_env_validation_summary.txt"))

message("R figure environment validation complete: ", out_dir)
