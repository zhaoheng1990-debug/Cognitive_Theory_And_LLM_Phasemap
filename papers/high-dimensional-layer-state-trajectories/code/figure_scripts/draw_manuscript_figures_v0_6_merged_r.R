library(ggplot2)
library(ggforce)
library(ggnewscale)
library(patchwork)
library(dplyr)
library(tidyr)
library(scales)
library(grid)

parse_cli_args <- function(args) {
  out <- list()
  i <- 1
  while (i <= length(args)) {
    item <- args[[i]]
    if (grepl("^--[^=]+=", item)) {
      kv <- strsplit(sub("^--", "", item), "=", fixed = TRUE)[[1]]
      out[[kv[[1]]]] <- paste(kv[-1], collapse = "=")
    } else if (grepl("^--", item)) {
      key <- sub("^--", "", item)
      if (i < length(args) && !grepl("^--", args[[i + 1]])) {
        out[[key]] <- args[[i + 1]]
        i <- i + 1
      } else {
        out[[key]] <- "true"
      }
    }
    i <- i + 1
  }
  out
}

cli_args <- parse_cli_args(commandArgs(trailingOnly = TRUE))

get_setting <- function(cli_name, env_name, default) {
  if (!is.null(cli_args[[cli_name]]) && nzchar(cli_args[[cli_name]])) {
    return(cli_args[[cli_name]])
  }
  env_value <- Sys.getenv(env_name, unset = "")
  if (nzchar(env_value)) env_value else default
}

repo_root <- normalizePath(get_setting("repo-root", "NATURE_LLM_REPO_ROOT", "."), mustWork = FALSE)
root_dir <- normalizePath(get_setting("root-dir", "NATURE_LLM_ROOT_DIR", repo_root), mustWork = FALSE)
table_dir <- normalizePath(
  get_setting("table-dir", "NATURE_LLM_TABLE_DIR", file.path(root_dir, "nature_figure_source_column_audit_output", "panel_source_tables")),
  mustWork = FALSE
)
out_dir <- normalizePath(
  get_setting("out-dir", "NATURE_LLM_FIGURE_OUT_DIR", file.path(root_dir, "outputs", "figures")),
  mustWork = FALSE
)
source_dir <- normalizePath(
  get_setting("source-dir", "NATURE_LLM_SOURCE_DATA_DIR", file.path(out_dir, "source_data")),
  mustWork = FALSE
)
highdim_dir <- normalizePath(
  get_setting("highdim-dir", "NATURE_LLM_HIGHDIM_DIR", file.path(root_dir, "nature_mechanism_highdim_experiment_v0_4_frozen")),
  mustWork = FALSE
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(source_dir, recursive = TRUE, showWarnings = FALSE)

palette <- c(
  neutral_dark = "#272727",
  neutral_mid = "#767676",
  neutral_light = "#D8D8D8",
  neutral_pale = "#F2F2F2",
  signal_blue = "#3182BD",
  signal_teal = "#33B5A5",
  signal_violet = "#7A6BB7",
  accent_red = "#D24B40",
  accent_orange = "#E28E2C",
  accent_green = "#2E9E44"
)

col <- function(name) unname(palette[name])

model_cols <- c(qwen = col("signal_blue"), llama = col("signal_teal"), gemma = col("signal_violet"))

model_labs <- c(qwen = "Qwen", llama = "Llama", gemma = "Gemma")
model_cols_display <- c(
  Qwen = unname(model_cols["qwen"]),
  Llama = unname(model_cols["llama"]),
  Gemma = unname(model_cols["gemma"])
)

theme_nature <- function(base_size = 6.4, base_family = "Arial") {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      axis.line = element_line(linewidth = 0.32, colour = "black"),
      axis.ticks = element_line(linewidth = 0.28, colour = "black"),
      axis.title = element_text(size = base_size),
      axis.text = element_text(size = base_size - 0.45, colour = "black"),
      legend.title = element_text(size = base_size - 0.35),
      legend.text = element_text(size = base_size - 0.65),
      legend.key.height = unit(3.5, "mm"),
      legend.key.width = unit(4.5, "mm"),
      legend.background = element_blank(),
      legend.box.background = element_blank(),
      strip.background = element_blank(),
      strip.text = element_text(size = base_size - 0.15, face = "bold"),
      plot.title = element_text(size = base_size + 0.45, face = "bold", hjust = 0),
      plot.subtitle = element_text(size = base_size - 0.45, colour = palette["neutral_mid"], hjust = 0),
      plot.margin = margin(2.5, 3.5, 2.5, 3.5),
      panel.grid = element_blank()
    )
}

theme_set(theme_nature())

save_pub <- function(plot, stem, width_mm = 183, height_mm = 118, dpi = 600) {
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

read_panel <- function(name) {
  read.csv(file.path(table_dir, name), check.names = FALSE, stringsAsFactors = FALSE)
}

as_model_factor <- function(x) {
  factor(x, levels = c("qwen", "llama", "gemma"), labels = c("Qwen", "Llama", "Gemma"))
}

short_metric <- function(x) {
  dplyr::recode(
    x,
    hidden = "Hidden",
    hshape = "H-shape",
    topk_real = "TopK realW",
    topk_rowperm = "TopK rowperm",
    topk_gaussian = "TopK Gaussian",
    real = "Real",
    fixed = "Fixed",
    shuffle = "Shuffle",
    .default = x
  )
}

parse_layers_start <- function(x) {
  nums <- regmatches(x, gregexpr("[0-9]+", x))
  vapply(nums, function(z) if (length(z) == 0) NA_real_ else as.numeric(z[1]), numeric(1))
}

parse_layers_end <- function(x) {
  nums <- regmatches(x, gregexpr("[0-9]+", x))
  vapply(nums, function(z) if (length(z) == 0) NA_real_ else as.numeric(tail(z, 1)), numeric(1))
}

format_window_label <- function(x) {
  dplyr::recode(
    x,
    abs_L0_6 = "absolute L0-6",
    rel_0_25 = "relative 0-25%",
    rel_15_35 = "relative 15-35%",
    .default = x
  )
}

percent_label <- function(x) sprintf("%.0f%%", 100 * x)
num_label <- function(x) sprintf("%.2f", x)

write_source <- function(df, name) {
  write.csv(df, file.path(source_dir, name), row.names = FALSE, fileEncoding = "UTF-8")
}

panel_tag_theme <- theme(
  plot.tag = element_text(size = 8.2, face = "bold", family = "Arial"),
  plot.tag.position = c(0.01, 0.99)
)

wrap_label <- function(x, width = 18) {
  vapply(strwrap(x, width = width, simplify = FALSE), paste, collapse = "\n", FUN.VALUE = character(1))
}

theme_schematic <- function() {
  theme_void(base_family = "Arial") +
    theme(
      plot.title = element_text(size = 7.1, face = "bold", hjust = 0, margin = margin(b = 2)),
      plot.subtitle = element_text(size = 5.8, colour = col("neutral_mid"), hjust = 0, margin = margin(b = 2)),
      plot.margin = margin(2, 3, 2, 3)
    )
}

flow_schematic <- function(title, subtitle, nodes, note = NULL, highlight = integer(0), width = 18,
                           y = 0.58, note_y = 0.16) {
  n <- length(nodes)
  xs <- seq(0.08, 0.92, length.out = n)
  df <- data.frame(
    x = xs,
    y = y,
    label = wrap_label(nodes, width),
    fill = ifelse(seq_len(n) %in% highlight, col("signal_teal"), col("neutral_pale")),
    text = ifelse(seq_len(n) %in% highlight, "white", col("neutral_dark")),
    stringsAsFactors = FALSE
  )
  p <- ggplot() +
    xlim(0, 1) +
    ylim(0, 1) +
    theme_schematic() +
    labs(title = title, subtitle = subtitle)
  if (n > 1) {
    for (i in seq_len(n - 1)) {
      p <- p + annotate(
        "segment",
        x = xs[i] + 0.055, xend = xs[i + 1] - 0.055, y = y, yend = y,
        linewidth = 0.32, colour = col("neutral_mid"),
        arrow = arrow(length = unit(1.6, "mm"), type = "closed")
      )
    }
  }
  p <- p +
    geom_label(
      data = df,
      aes(x, y, label = label),
      fill = df$fill,
      colour = df$text,
      label.size = 0.18,
      label.r = unit(2, "mm"),
      size = 2.05,
      lineheight = 0.88,
      label.padding = unit(2.1, "mm")
    )
  if (!is.null(note)) {
    p <- p + annotate(
      "text", x = 0.5, y = note_y, label = note,
      size = 1.95, colour = col("neutral_mid"), hjust = 0.5, lineheight = 0.9
    )
  }
  p
}

branch_schematic <- function(title, subtitle, left, branches, right, note = NULL) {
  p <- ggplot() +
    xlim(0, 1) +
    ylim(0, 1) +
    theme_schematic() +
    labs(title = title, subtitle = subtitle) +
    annotate("label", x = 0.13, y = 0.55, label = wrap_label(left, 16), fill = col("neutral_pale"),
             colour = col("neutral_dark"), label.size = 0.18, label.r = unit(2, "mm"),
             size = 2.05, lineheight = 0.88) +
    annotate("label", x = 0.86, y = 0.55, label = wrap_label(right, 16), fill = col("signal_teal"),
             colour = "white", label.size = 0.18, label.r = unit(2, "mm"),
             size = 2.05, lineheight = 0.88)
  ys <- seq(0.74, 0.36, length.out = length(branches))
  for (i in seq_along(branches)) {
    p <- p +
      annotate("segment", x = 0.23, xend = 0.39, y = 0.55, yend = ys[i],
               linewidth = 0.28, colour = col("neutral_mid"),
               arrow = arrow(length = unit(1.4, "mm"), type = "closed")) +
      annotate("label", x = 0.51, y = ys[i], label = wrap_label(branches[i], 15),
               fill = col("neutral_pale"), colour = col("neutral_dark"),
               label.size = 0.16, label.r = unit(2, "mm"), size = 1.85, lineheight = 0.86) +
      annotate("segment", x = 0.63, xend = 0.76, y = ys[i], yend = 0.55,
               linewidth = 0.28, colour = col("neutral_mid"),
               arrow = arrow(length = unit(1.4, "mm"), type = "closed"))
  }
  if (!is.null(note)) {
    p <- p + annotate("text", x = 0.5, y = 0.13, label = note, size = 1.9, colour = col("neutral_mid"))
  }
  p
}

mechanism_theme <- function() {
  theme_void(base_family = "Arial") +
    theme(
      plot.title = element_text(size = 7.4, face = "bold", hjust = 0, margin = margin(b = 1)),
      plot.subtitle = element_text(size = 5.8, colour = col("neutral_mid"), hjust = 0, margin = margin(b = 1)),
      plot.margin = margin(2, 4, 2, 4),
      legend.position = "none"
    )
}

mechanism_label <- function(label, x, y, size = 1.95, colour = col("neutral_dark"), hjust = 0.5) {
  annotate("text", x = x, y = y, label = label, size = size, colour = colour, hjust = hjust, lineheight = 0.9)
}

bezier_path <- function(group, xs, ys, colour, linewidth = 0.55, linetype = "solid", alpha = 1) {
  data.frame(
    group = group,
    x = xs,
    y = ys,
    colour = colour,
    linewidth = linewidth,
    linetype = linetype,
    alpha = alpha,
    stringsAsFactors = FALSE
  )
}

mechanism_canvas <- function(title, subtitle) {
  ggplot() +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE, clip = "off") +
    labs(title = title, subtitle = subtitle) +
    mechanism_theme()
}

draw_fields <- function(p, fields) {
  if (nrow(fields) == 0) return(p)
  p +
    ggforce::geom_ellipse(
      data = fields,
      aes(x0 = x0, y0 = y0, a = a, b = b, angle = angle, fill = fill),
      alpha = 0.18,
      colour = NA
    ) +
    scale_fill_identity()
}

draw_paths <- function(p, paths) {
  if (nrow(paths) == 0) return(p)
  p +
    ggforce::geom_bezier(
      data = paths,
      aes(x = x, y = y, group = group, colour = colour, linewidth = linewidth,
          linetype = linetype, alpha = alpha),
      lineend = "round"
    ) +
    scale_colour_identity() +
    scale_linewidth_identity() +
    scale_linetype_identity() +
    scale_alpha_identity()
}

draw_probes <- function(p, probes) {
  if (nrow(probes) == 0) return(p)
  p +
    geom_segment(
      data = probes,
      aes(x = x, y = y, xend = xend, yend = yend),
      linewidth = 0.26,
      linetype = "dashed",
      colour = col("neutral_mid"),
      arrow = arrow(length = unit(1.25, "mm"), type = "closed")
    )
}

mechanism_trajectory_competition <- function(title, subtitle, note = NULL,
                                             show_readback = TRUE,
                                             show_delta = TRUE,
                                             show_feedback = FALSE,
                                             show_null = FALSE) {
  fields <- data.frame(
    x0 = c(0.17, 0.54, 0.80),
    y0 = c(0.56, 0.56, 0.50),
    a = c(0.13, 0.22, 0.15),
    b = c(0.11, 0.18, 0.12),
    angle = c(-0.25, 0.10, -0.15),
    fill = c(col("signal_blue"), col("signal_teal"), col("signal_violet")),
    alpha = c(0.18, 0.18, 0.16)
  )
  paths <- bind_rows(
    bezier_path("candidate_hi", c(0.11, 0.28, 0.44, 0.62), c(0.56, 0.69, 0.75, 0.70), col("neutral_mid"), 0.34, alpha = 0.80),
    bezier_path("candidate_lo", c(0.11, 0.30, 0.46, 0.66), c(0.56, 0.48, 0.36, 0.34), col("neutral_mid"), 0.34, alpha = 0.80),
    bezier_path("dominant", c(0.11, 0.32, 0.55, 0.86), c(0.56, 0.57, 0.53, 0.50), col("signal_teal"), 1.05)
  )
  if (show_null) {
    paths <- bind_rows(paths, bezier_path("shuffle_null", c(0.11, 0.29, 0.49, 0.80), c(0.42, 0.39, 0.41, 0.38), col("neutral_light"), 0.62, "dashed", 0.90))
  }
  probes <- if (show_readback) {
    data.frame(x = c(0.38, 0.55, 0.72), y = c(0.69, 0.53, 0.51), xend = c(0.38, 0.55, 0.72), yend = c(0.23, 0.20, 0.24))
  } else data.frame()
  p <- mechanism_canvas(title, subtitle) %>%
    draw_fields(fields) %>%
    draw_paths(paths) %>%
    draw_probes(probes) +
    annotate("label", x = 0.11, y = 0.56, label = "Prompt", size = 1.9,
             label.size = 0.15, label.r = unit(1.5, "mm"),
             fill = col("neutral_pale"), colour = col("neutral_dark")) +
    mechanism_label("local field", 0.18, 0.34, colour = col("neutral_mid")) +
    mechanism_label("dominance basin", 0.53, 0.81, colour = col("neutral_mid")) +
    mechanism_label("commitment", 0.83, 0.32, colour = col("neutral_mid"))
  if (show_delta) {
    p <- p +
      annotate("segment", x = 0.44, xend = 0.44, y = 0.73, yend = 0.37,
               linewidth = 0.34, colour = col("accent_red"),
               arrow = arrow(length = unit(1.35, "mm"), ends = "both", type = "closed")) +
      mechanism_label("Delta U", 0.455, 0.56, size = 2.0, colour = col("accent_red"), hjust = 0)
  }
  if (show_feedback) {
    p <- p +
      annotate("curve", x = 0.88, y = 0.30, xend = 0.66, yend = 0.49, curvature = 0.32,
               linewidth = 0.42, colour = col("accent_orange"),
               arrow = arrow(length = unit(1.55, "mm"), type = "closed")) +
      mechanism_label("feedback", 0.88, 0.22, colour = col("accent_orange"))
  }
  if (!is.null(note)) {
    p <- p + mechanism_label(note, 0.50, 0.09, size = 1.8, colour = col("neutral_mid"))
  }
  p
}

mechanism_readback_projection <- function() {
  fields <- data.frame(
    x0 = c(0.30, 0.68),
    y0 = c(0.60, 0.30),
    a = c(0.23, 0.19),
    b = c(0.14, 0.08),
    angle = c(0.05, 0.0),
    fill = c(col("signal_teal"), col("signal_blue")),
    alpha = c(0.18, 0.13)
  )
  paths <- bind_rows(
    bezier_path("hidden", c(0.10, 0.28, 0.50, 0.72), c(0.57, 0.68, 0.63, 0.55), col("signal_teal"), 0.95),
    bezier_path("rowperm", c(0.10, 0.29, 0.50, 0.72), c(0.45, 0.40, 0.38, 0.42), col("neutral_mid"), 0.36, "dashed", 0.9)
  )
  probes <- data.frame(x = c(0.34, 0.53, 0.68), y = c(0.66, 0.61, 0.56), xend = c(0.50, 0.63, 0.75), yend = c(0.31, 0.29, 0.31))
  mechanism_canvas("Mechanism map", "hidden trajectory with readback probes") %>%
    draw_fields(fields) %>%
    draw_paths(paths) %>%
    draw_probes(probes) +
    mechanism_label("hidden-state trajectory", 0.35, 0.78, colour = col("signal_teal")) +
    mechanism_label("TopK/VIM readback plane", 0.68, 0.1×~6öÚ$z{-®éÜj×÷'"æÒ6öÂ‚'6–væÅö&ÇVR"’’ÂwV–FRÒ&æöæR"’°¢Æ'2‡F—FÆRÒ%&VF–7FVBvVöÖWG'’&VÖ–ç2G&6V&ÆR"Â7V'F—FÆRÒ%†6TÖÓtBã"Â‚ÒåTÄÂÂ’Ò%fÇVR"’°¢F†VÖR††—2çFW‡Bç‚ÒVÆVÖVçE÷FW‡B†ævÆRÒ#RÂ†§W7BÒ’ ¦f–sEööÆEö6Æ÷7W&RÂÒ‡F"ò‡FÂF2’’°¢Æ÷EöÆ–÷WB††V–v‡G2Ò2ƒãRÂ’ÂwV–FW2Ò&6öÆÆV7B"’°¢Æ÷Eöææ÷FF–öâ‡FuöÆWfVÇ2Ò&"’bæVÅ÷Fu÷F†VÖP ¢2f–wW&RRÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÐ¦wcÂÒ&VE÷æVÂ‚&f–wW&SUóTöwc%öÖöFVÅ÷7VÖÖ'’æ77b"’SâP¢G&ç6×WFR€¢ÖöFVÂÒ5öÖöFVÅöf7F÷"†ÖöFVÅö¶W’’À¢Æ–væÖVçBv–æÒ2æçVÖW&–2†&–5÷&VÅöÆ–våöÖVâ’Ò2æçVÖW&–2†&–5÷6‡VffÆUöÆ–våöÖVâ’À¢FWF÷W"&VGV7F–öæÒÆös"†2æçVÖW&–2†&–5÷6‡VffÆUöFWF÷W%öÖVâ’ò2æçVÖW&–2†&–5÷&VÅöFWF÷W%öÖVâ’’À¢7W'fGW&R&VGV7F–öæÒÆös"†2æçVÖW&–2†&–5÷6‡VffÆUö7W'fGW&UöÖVâ’ò2æçVÖW&–2†&–5÷&VÅö7W'fGW&UöÖVâ’¢’SâP¢—f÷EöÆöævW"‚ÖÖöFVÂÂæÖW5÷FòÒ&ÖWG&–2"ÂfÇVW5÷FòÒ'fÇVR" ¦wc"ÂÒ&VE÷æVÂ‚&f–wW&SUóT%öwc%öÖöFVÅ÷7VÖÖ'’æ77b"’SâP¢G&ç6×WFR€¢ÖöFVÂÒ5öÖöFVÅöf7F÷"†ÖöFVÅö¶W’’À¢&ö¦V7F–öâÒ2æçVÖW&–2†&W7E÷&ö¦V7F–öåöFVÇFUö6÷'"’À¢&W6–GVÂÒ2æçVÖW&–2†&W7E÷&W6–EöFVÇFUö6÷'"’À¢Ö—†VBÒ2æçVÖW&–2†&W7EöÖ—†VEöFVÇFUö6÷'"¢’SâP¢—f÷EöÆöævW"‚ÖÖöFVÂÂæÖW5÷FòÒ&fVGW&R"ÂfÇVW5÷FòÒ&6÷'"" ¦wc2ÂÒ&VE÷æVÂ‚&f–wW&SUóT5öwc5öÖöFVÅ÷7VÖÖ'’æ77b"’SâP¢G&ç6×WFR€¢ÖöFVÂÒ5öÖöFVÅöf7F÷"†ÖöFVÅö¶W’’À¢&ÆÆVÂÒ2æçVÖW&–2†&W7E÷&ÆÆVÅöFVÇFUö6÷'"’À¢W'VæF–7VÆ"Ò2æçVÖW&–2†&W7E÷W'öFVÇFUö6÷'"’À¢&÷F‚Ò2æçVÖW&–2†&W7Eö&÷F…öFVÇFUö6÷'"¢’SâP¢—f÷EöÆöævW"‚ÖÖöFVÂÂæÖW5÷FòÒ&6ö×öæVçB"ÂfÇVW5÷FòÒ&6÷'"" ¦wcBÂÒ&VE÷æVÂ‚&f–wW&SUóTEöwcEöÖöFVÅ÷7VÖÖ'’æ77b"’SâP¢G&ç6×WFR€¢ÖöFVÂÒ5öÖöFVÅöf7F÷"†ÖöFVÅö¶W’’À¢fÖ–Ç’FVÇFR6÷'"æÒ2æçVÖW&–2†&W7EöfÖ–Ç•öFVÇFUö6÷'"’À¢fÖ–Ç’ÖV6†æ—6ÒcÒ2æçVÖW&–2†&W7EöfÖ–Ç•öÖV6†æ—6Õöc’À¢6ö×WF—F–öâT6Ò2æçVÖW&–2†&W7Eö&æEö6ö×WF—F–öåöV2’À¢&÷VæF'’T6Ò2æçVÖW&–2†&W7Eö&æEö&÷VæF'•öV2¢’SâP¢—f÷EöÆöævW"‚ÖÖöFVÂÂæÖW5÷FòÒ&ÖWG&–2"ÂfÇVW5÷FòÒ'fÇVR" §w&—FU÷6÷W&6R†wcÂ&f–wW&SE÷æVÅöEövVöFW6–5ö&–5ö–æF–6W2æ77b"§w&—FU÷6÷W&6R†wc"Â&f–wW&SE÷æVÅöU÷&W6–GVÅ÷&VF–7F–öâæ77b"§w&—FU÷6÷W&6R†wc2Â&f–wW&SE÷æVÅöe÷&ÆÆVÅ÷W'VæF–7VÆ"æ77b"§w&—FU÷6÷W&6R†wcBÂ&W‡FVæFVEöFF%ö÷F–öæÅ÷&W6–GVÅöfÖ–Æ–W2æ77b" §VÂÒvwÆ÷B†wcÂW2†ÖWG&–2ÂÖöFVÂÂf–ÆÂÒfÇVR’’°¢vVöÕ÷F–ÆR†6öÆ÷W"Ò'v†—FR"ÂÆ–æWv–GF‚ÒãCR’°¢vVöÕ÷FW‡B†W2†Æ&VÂÒçVÕöÆ&VÂ‡fÇVR’’Â6—¦RÒ"ã’°¢66ÆUöf–ÆÅöw&F–VçC"†Æ÷rÒÆWGFU²&66VçE÷&VB%ÒÂÖ–BÒ'v†—FR"Â†–v‚ÒÆWGFU²&66VçEöw&VVâ%ÒÂÖ–Gö–çBÒÂæÖRÒ$v–â"’°¢Æ'2‡F—FÆRÒ%&VÂG&¦V7F÷&–W26†÷rvVöFW6–2Ö&–6VBfÆ÷r"Â7V'F—FÆRÒ$ubÓ#²÷6—F—fRfÇVW2ff÷W"&VÂG&¦V7F÷&–W2"Â‚ÒåTÄÂÂ’ÒåTÄÂ’°¢F†VÖR††—2çFW‡Bç‚ÒVÆVÖVçE÷FW‡B†ævÆRÒ#RÂ†§W7BÒ’ §V"ÂÒvwÆ÷B†wc"ÂW2†fVGW&RÂ6÷'"Âf–ÆÂÒfVGW&R’’°¢vVöÕö6öÂ‡v–GF‚ÒãbÂ6öÆ÷W"Ò'v†—FR"ÂÆ–æWv–GF‚Òã‚’°¢f6WE÷w&‡âÖöFVÂÂç&÷rÒ’°¢66ÆUöf–ÆÅöÖçVÂ‡fÇVW2Ò2…&ö¦V7F–öâÒ6öÂ‚&æWWG&ÅöÖ–B"’Â&W6–GVÂÒ6öÂ‚'6–væÅ÷FVÂ"’ÂÖ—†VBÒ6öÂ‚'6–væÅö&ÇVR"’’ÂwV–FRÒ&æöæR"’°¢6ö÷&Eö6'FW6–â‡–Æ–ÒÒ2ƒÂÖ‚†wc"F6÷'"’²ãb’’°¢Æ'2‡F—FÆRÒ%&W6–GVÂÖöæÇ’fVGW&W2&WF–â&VF–7F–öâ6–væÂ"Â7V'F—FÆRÒ$ubÓ""Â‚ÒåTÄÂÂ’Ò$FVÇFR6÷'"â"’°¢F†VÖR††—2çFW‡Bç‚ÒVÆVÖVçE÷FW‡B†ævÆRÒ#RÂ†§W7BÒ’ §V2ÂÒvwÆ÷B†wc2ÂW2†6ö×öæVçBÂ6÷'"Âf–ÆÂÒ6ö×öæVçB’’°¢vVöÕö6öÂ‡v–GF‚ÒãS‚Â6öÆ÷W"Ò'v†—FR"ÂÆ–æWv–GF‚Òã‚’°¢f6WE÷w&‡âÖöFVÂÂç&÷rÒ’°¢66ÆUöf–ÆÅöÖçVÂ‡fÇVW2Ò2…&ÆÆVÂÒ6öÂ‚'6–væÅö&ÇVR"’ÂW'VæF–7VÆ"Ò6öÂ‚'6–væÅ÷FVÂ"’Â&÷F‚Ò6öÂ‚'6–væÅ÷f–öÆWB"’’ÂwV–FRÒ&æöæR"’°¢6ö÷&Eö6'FW6–â‡–Æ–ÒÒ2ƒÂÖ‚†wc2F6÷'"’²ãb’’°¢Æ'2‡F—FÆRÒ%&ÆÆVÂæBW'VæF–7VÆ"&W6–GVÇ2&R6ö×ÆVÖVçF'’"Â7V'F—FÆRÒ$ubÓ2"Â‚ÒåTÄÂÂ’Ò$FVÇFR6÷'"â"’°¢F†VÖR††—2çFW‡Bç‚ÒVÆVÖVçE÷FW‡B†ævÆRÒ#RÂ†§W7BÒ’ §VBÂÒvwÆ÷B†wcBÂW2†ÖWG&–2ÂÖöFVÂÂf–ÆÂÒfÇVR’’°¢vVöÕ÷F–ÆR†6öÆ÷W"Ò'v†—FR"ÂÆ–æWv–GF‚ÒãCR’°¢vVöÕ÷FW‡B†W2†Æ&VÂÒçVÕöÆ&VÂ‡fÇVR’’Â6—¦RÒ"ã’°¢66ÆUöf–ÆÅöw&F–VçB†Æ÷rÒÆWGFU²&æWWG&ÅöÆ–v‡B%ÒÂ†–v‚ÒÆWGFU²'6–væÅ÷f–öÆWB%ÒÂæÖRÒ%66÷&R"’°¢Æ'2‡F—FÆRÒ%&W6–GVÂfÖ–Æ–W26''’7G'V7GW&VB6–væÂ"Â7V'F—FÆRÒ$ubÓBôubÓD""Â‚ÒåTÄÂÂ’ÒåTÄÂ’°¢F†VÖR††—2çFW‡Bç‚ÒVÆVÖVçE÷FW‡B†ævÆRÒ#RÂ†§W7BÒ’ ¦f–sBÂÒ‡F"ò‡FÂF2’òVò‡V"ÂV2’’°¢Æ÷EöÆ–÷WB††V–v‡G2Ò2ƒã“RÂã“RÂã“"Âã’ÂwV–FW2Ò&6öÆÆV7B"’°¢Æ÷Eöææ÷FF–öâ‡FuöÆWfVÇ2Ò&"’bæVÅ÷Fu÷F†VÖP¦ÖW76vR‚%6f–ærf–wW&RB"§6fU÷V"†f–sBÂf–ÆRçF‚†÷WEöF—"Â&f–wW&SE÷7FFUöWVF–öå÷G&¦V7F÷'•övVöÖWG'’"’Âƒ2Â##" ¢2f–wW&RRÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÐ¦6ÂÒ&VE÷æVÂ‚&f–wW&Seódö6–ö7&÷75öÖöFVÅ÷7VÖÖ'’æ77b"’SâP¢×WFFR€¢ÖöFVÅöbÒ5öÖöFVÅöf7F÷"†ÖöFVÂ’À¢&VÅ÷&VFÖV6…÷7V6–f–6—G’Ò2æçVÖW&–2‡&VÅ÷&VFÖV6…÷7V6–f–6—G’’À¢f—†VEö6öÖ&õ÷7V6–f–6—G’Ò2æçVÖW&–2†f—†VEö6öÖ&õ÷7V6–f–6—G’’À¢v–åö÷fW%öf—†VBÒ2æçVÖW&–2†v–åö÷fW%öf—†VB’À¢¥÷g5÷6‡VffÆUö&÷F‚Ò2æçVÖW&–2‡¥÷g5÷6‡VffÆUö&÷F‚’À¢ÖV6†æ—6Õ÷G&–åö67W&7’Ò2æçVÖW&–2‡fW&F–7Eö77e÷&VÅöÖV6…÷G&–åö67W&7’’À¢ÖV6†æ—6ÕöÖ7&õöcÒ2æçVÖW&–2‡fW&F–7Eö77e÷&VÅöÖV6…÷G&–åöÖ7&õöc’À¢öÆ–7•÷G&–åö67W&7’Ò2æçVÖW&–2‡fW&F–7Eö77e÷&VÅ÷öÆ–7•÷G&–åö67W&7’’À¢öÆ–7•öÖ7&õöcÒ2æçVÖW&–2‡fW&F–7Eö77e÷&VÅ÷öÆ–7•÷G&–åöÖ7&õöc¢ ¦6÷7V2ÂÒ6SâP¢G&ç6×WFR†ÖöFVÂÒÖöFVÅöbÂ&VÂÒ&VÅ÷&VFÖV6…÷7V6–f–6—G’Âf—†VBÒf—†VEö6öÖ&õ÷7V6–f–6—G’’SâP¢—f÷EöÆöævW"‚ÖÖöFVÂÂæÖW5÷FòÒ&6öæF—F–öâ"ÂfÇVW5÷FòÒ'7V6–f–6—G’" ¦6ö6Ç2ÂÒ6SâP¢G&ç6×WFR€¢ÖöFVÂÒÖöFVÅöbÀ¢ÖV6†æ—6Ò62æÒÖV6†æ—6Õ÷G&–åö67W&7’À¢ÖV6†æ—6ÒcÒÖV6†æ—6ÕöÖ7&õöcÀ¢öÆ–7’62æÒöÆ–7•÷G&–åö67W&7’À¢öÆ–7’cÒöÆ–7•öÖ7&õöc¢’SâP¢—f÷EöÆöævW"‚ÖÖöFVÂÂæÖW5÷FòÒ&ÖWG&–2"ÂfÇVW5÷FòÒ'fÇVR" ¦6ö7F–öâÂÒ&VE÷æVÂ‚&f–wW&Seód5ö6–÷öÆ–7•ö7F–öåöF—7G&–'WF–öâæ77b"’SâP¢6÷VçB†ÖöFVÂÂ7F–öåöÆ&VÂÂæÖRÒ&â"’SâP¢w&÷Wö'’†ÖöFVÂ’SâP¢×WFFR‡6†&RÒâò7VÒ†â’ÂÖöFVÂÒ5öÖöFVÅöf7F÷"†ÖöFVÂ’’SâP¢Væw&÷W‚’SâP¢w&÷Wö'’†ÖöFVÂ’SâP¢6Æ–6UöÖ‚‡6†&RÂâÒRÂv—F…÷F–W2ÒdÅ4R’SâP¢Væw&÷W‚ §w&—FU÷6÷W&6R†6÷7V2Â&f–wW&SU÷æVÅö÷&VÅöf—†VE÷7V6–f–6—G’æ77b"§w&—FU÷6÷W&6R†6SâR6VÆV7B†ÖöFVÅöbÂv–åö÷fW%öf—†VBÂ¥÷g5÷6‡VffÆUö&÷F‚ÂW†6VVG5÷6‡VffÆUö&÷F…÷“R’Â&f–wW&SU÷æVÅö%÷&WÆ–6F–öå÷7FF—7F–72æ77b"§w&—FU÷6÷W&6R†6ö6Ç2Â&f–wW&SU÷æVÅö5ö6Æ76–f–W%÷VÆ—G’æ77b"§w&—FU÷6÷W&6R†6ö7F–öâÂ&f–wW&SU÷æVÅöE÷öÆ–7•ö7F–öåöF—7G&–'WF–öâæ77b" §fÂÒvwÆ÷B†6÷7V2ÂW2‡7V6–f–6—G’ÂÖöFVÂÂ6öÆ÷W"Ò6öæF—F–öâ’’°¢vVöÕöÆ–æR†W2†w&÷WÒÖöFVÂ’Â6öÆ÷W"ÒÆWGFU²&æWWG&ÅöÆ–v‡B%ÒÂÆ–æWv–GF‚ÒãcR’°¢vVöÕ÷ö–çB‡6—¦RÒãcR’°¢66ÆUö6öÆ÷W%öÖçVÂ‡fÇVW2Ò2…&VÂÒ6öÂ‚'6–væÅ÷FVÂ"’Âf—†VBÒ6öÂ‚&æWWG&ÅöÖ–B"’’ÂæÖRÒ$6öæF—F–öâ"’°¢66ÆU÷…ö6öçF–çV÷W2‡G&ç2Ò6WVFõöÆöu÷G&ç2‡6–vÖÒã‚’Â'&V·2Ò2ƒÂ2Â’ÂÆ&VÇ2Ò2‚#"Â#2"Â#"’’°¢Æ'2‡F—FÆRÒ$6Æ÷6VBÖÆö÷7FVW&–ær–æ7&V6W27V6–f–6—G’"Â7V'F—FÆRÒ$4Ó”ãÂS6‡VffÆW2"Â‚Ò%7V6–f–6—G’‡6WVFòÖÆör’"Â’ÒåTÄÂ §f"ÂÒvwÆ÷B†6ÂW2†ÖöFVÅöbÂ¥÷g5÷6‡VffÆUö&÷F‚Âf–ÆÂÒÖöFVÂ’’°¢vVöÕö6öÂ‡v–GF‚ÒãS‚Â6öÆ÷W"Ò'v†—FR"ÂÆ–æWv–GF‚Òã‚’°¢vVöÕö†Æ–æR‡––çFW&6WBÒ"ÂÆ–æWv–GF‚Òã3RÂÆ–æWG—RÒ&F6†VB"Â6öÆ÷W"ÒÆWGFU²&æWWG&ÅöÖ–B%Ò’°¢66ÆUöf–ÆÅöÖçVÂ‡fÇVW2ÒÖöFVÅö6öÇ2ÂwV–FRÒ&æöæR"’°¢6ö÷&Eö6'FW6–â‡–Æ–ÒÒ2ƒÂÖ‚†6G¥÷g5÷6‡VffÆUö&÷F‚’²ãCR’’°¢Æ'2‡F—FÆRÒ$ÆÂÖöFVÇ2W†6VVB6‡VffÆRÖ&÷F‚çVÆÇ2"Â7V'F—FÆRÒ&F6†VBÆ–æS¢¢Ò""Â‚ÒåTÄÂÂ’Ò'¢g26‡VffÆRÖ&÷F‚" §f2ÂÒvwÆ÷B†6ö6Ç2ÂW2†ÖWG&–2ÂÖöFVÂÂf–ÆÂÒfÇVR’’°¢vVöÕ÷F–ÆR†6öÆ÷W"Ò'v†—FR"ÂÆ–æWv–GF‚ÒãCR’°¢vVöÕ÷FW‡B†W2†Æ&VÂÒçVÕöÆ&VÂ‡fÇVR’’Â6—¦RÒ"ã’°¢66ÆUöf–ÆÅöw&F–VçB†Æ÷rÒÆWGFU²&æWWG&ÅöÆ–v‡B%ÒÂ†–v‚ÒÆWGFU²'6–væÅ÷FVÂ%ÒÂæÖRÒ$6Æ76–f–W"66÷&R"’°¢Æ'2‡F—FÆRÒ$6Æ76–f–W"VÆ—G’&÷VæG2–çFW'&WFF–öâ"Â7V'F—FÆRÒ&ÖV6†æ—6ÒæBöÆ–7’ÖöFVÇ2"Â‚ÒåTÄÂÂ’ÒåTÄÂ’°¢F†VÖR††—2çFW‡Bç‚ÒVÆVÖVçE÷FW‡B†ævÆRÒ#RÂ†§W7BÒ’ §fBÂÒvwÆ÷B†6ö7F–öâÂW2‡&V÷&FW"†7F–öåöÆ&VÂÂ6†&R’Â6†&RÂf–ÆÂÒÖöFVÂ’’°¢vVöÕö6öÂ‡v–GF‚ÒãS‚Â6öÆ÷W"Ò'v†—FR"ÂÆ–æWv–GF‚Òã‚’°¢f6WE÷w&‡âÖöFVÂÂ66ÆW2Ò&g&VU÷‚"Âç&÷rÒ’°¢66ÆUöf–ÆÅöÖçVÂ‡fÇVW2ÒÖöFVÅö6öÇ5öF—7Æ’ÂwV–FRÒ&æöæR"’°¢66ÆU÷•ö6öçF–çV÷W2†Æ&VÇ2ÒW&6VçEöf÷&ÖB†67W&7’Ò’’°¢Æ'2‡F—FÆRÒ%öÆ–7’7F–öç26öæ6VçG&FR–çFò–çFW'&WF&ÆRÖöFW2"Â7V'F—FÆRÒ'F÷f—fR7F–öç2W"ÖöFVÂ"Â‚ÒåTÄÂÂ’Ò$7F–öâ6†&R"’°¢F†VÖR††—2çFW‡Bç‚ÒVÆVÖVçE÷FW‡B†ævÆRÒCRÂ†§W7BÒ’ ¦f–sRÂÒ‚‡fÂf"’ò‡f2ÂfB’’°¢Æ÷EöÆ–÷WB††V–v‡G2Ò2ƒÂ’ÂwV–FW2Ò&6öÆÆV7B"’°¢Æ÷Eöææ÷FF–öâ‡FuöÆWfVÇ2Ò&"’bæVÅ÷Fu÷F†VÖP¦ÖW76vR‚%6f–ærf–wW&RR"§6fU÷V"†f–sRÂf–ÆRçF‚†÷WEöF—"Â&f–wW&SUö6Æ÷6VEöÆö÷÷7FVW&–æu÷&WÆ–6F–öâ"’Âƒ2Â3R ¢26öçG&7G2æBÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÐ¦6÷•öW‡÷'EöfÖ–Ç’ÂÒgVæ7F–öâ†g&öÕöF—"Âg&öÕ÷7FVÒÂFõöF—"ÂFõ÷7FVÒ’°¢f÷"†W‡B–â2‚'7fr"Â'Fb"Â'F–fb"Â'ær"’’°¢7&2ÂÒf–ÆRçF‚†g&öÕöF—"Â7FS†g&öÕ÷7FVÒÂ"â"ÂW‡B’¢G7BÂÒf–ÆRçF‚‡FõöF—"Â7FS‡Fõ÷7FVÒÂ"â"ÂW‡B’¢–b†f–ÆRæW†—7G2‡7&2’’f–ÆRæ6÷’‡7&2ÂG7BÂ÷fW'w&—FRÒE%TR¢Ð§Ð ¦6÷•öW‡÷'EöfÖ–Ç’€¢†–v†F–ÕöF—"À¢&†–v†F–ÕöÆ–W%÷fV7F÷%öÖV6†æ—6Õ÷f–Wu÷cóB"À¢÷WEöF—"À¢&f–wW&Sö†–v†F–ÕöÆ–W%÷fV7F÷%öÖV6†æ—6Õ÷f–Wr ¢¦6÷•öW‡÷'EöfÖ–Ç’€¢†–v†F–ÕöF—"À¢&†–v†F–ÕöæV–v†&÷W&†ööEöF–væ÷7F–75÷cóB"À¢÷WEöF—"À¢&W‡FVæFVEöFF%ö†–v†F–ÕöæV–v†&÷W&†ööEöF–væ÷7F–72 ¢¦6÷•öW‡÷'EöfÖ–Ç’€¢†–v†F–ÕöF—"À¢'&Wf–WuööæÇ•ö†–v†F–Õö6ö×7Eö6ö×÷6—FU÷cóB"À¢÷WEöF—"À¢'&Wf–WuööæÇ•ö†–v†F–Õö6ö×7Eö6ö×÷6—FR ¢ ¦†–v†F–Õ÷6÷W&6RÂÒf–ÆRçF‚††–v†F–ÕöF—"Â'6÷W&6UöFF"¦–b†F—"æW†—7G2††–v†F–Õ÷6÷W&6R’’°¢f–ÆRæ6÷’€¢Æ—7Bæf–ÆW2††–v†F–Õ÷6÷W&6RÂGFW&âÒ%ÅÂæ77bB"ÂgVÆÂææÖW2ÒE%TR’À¢6÷W&6UöF—"À¢÷fW'w&—FRÒE%TP¢§Ð ¦6öçG&7G2ÂÒF–&&ÆS£§G&–&&ÆR€¢æf–wW&RÂæ6÷&Uö6öæ6ÇW6–öâÂæ&6†WG—RÂæ†W&õöWf–FVæ6RÂæ6öçG&öÇ5ö÷%÷&—6·2À¢$f–wW&R"Â$ÄÄÒG–æÖ–72&RG&¦V7F÷'’6–væGW&W2–â†–v‚ÖF–ÖVç6–öæÂÆ–W"×7FFR6ö÷&F–æFR7—7FVÒâ"Â&†–v‚ÖF–ÖVç6–öæÂÖV6†æ—6Ò6ö×÷6—FR"Â&Æ–W"×fV7F÷"&&6öFRæB&ÆÆVÂÖ6ö÷&F–æFRG&¦V7F÷&–W2"Â&fö–G2G&VF–ærF†RÖV6†æ—6Ò2Æ—FW&Â4B7W&f6R"À¢$f–wW&R""Â$Æ–W"×7FFRG&¦V7F÷&–W2æBÖöFVÂ×7V6–f–2v–æF÷w2FVf–æRF†RV×—&–6Â6ö÷&F–æFRö&¦V7B&WGFW"F†âfö6'VÆ'’&VF&6·2÷"æ—fRÆ–væÖVçBâ"Â&7–ÖÖWG&–2VçF—FF—fR6ö×÷6—FR"Â$õ&÷‡’†–W&&6‡’ÇW24ÒÖöFVÂ×7V6–f–2v–æF÷w2"Â%4r6öçG&öÇ2Â&VF&6²Öv–â6öçG&öÇ2æB—'v—6RF—&V7F–öâ—6öÖ÷'†—6Ò"À¢$f–wW&R2"Â$FVÇFR—2&÷VæFVBFöÖ–ææ6Rö'6W'f&ÆR7W÷'FVB'’æöâÖÆV²æBæöâÕ"6öçG&öÇ2â"Â'VçF—FF—fR6ö×÷6—FR"Â$4ÒÓTFVÇFRf&–æ6Rö6÷'&VÆF–öâ"Â$4ÒÓT"Â4ÒÓT2æBTW†–Æ–'’6öçG&öÇ2"À¢$f–wW&RB"Â$6öçF–çV÷W2Ö÷'†—6Ò6ö÷&F–æFW2æB&W6–GVÂvVöÖWG'’6öç7G&–âF†R7FFRÖWVF–öâf–WröbG&ç6f÷&ÖW"G&¦V7F÷&–W2â"Â&7–ÖÖWG&–2VçF—FF—fR6ö×÷6—FR"Â%†6TÖÓtR6Æ÷7W&R&ÆF–öâ"Â%†6TÖÓt2&6VÆ–æRÂ†6TÖÓtBãvVöÖWG'’F–væ÷7F–72ÂubÓ"FòubÓ2&W6–GVÂFW7G2"À¢$f–wW&RR"Â$ÖV6†æ—6ÒÖv&R6Æ÷6VBÖÆö÷7FVW&–ærÖöGVÆFW2G&¦V7F÷'’FöÖ–ææ6R7&÷72ÖöFVÇ2VæFW"6‡VffÆRÖçVÆÂ6öç7G&–çG2â"Â'VçF—FF—fR6ö×÷6—FR"Â$4Ó”ã&VÂ×fW'7W2Öf—†VBæB¢×6‡VffÆR&WÆ–6F–öâ"Â&6Æ76–f–W"VÆ—G’æBöÆ–7’Ö7F–öâF—7G&–'WF–öâ"À¢$W‡FVæFVBFF"Â$Æ÷rÖF–ÖVç6–öæÂ&W6–GVÂÕ4&ö¦V7F–öâ6â÷&–VçB&VFW'2FòG&¦V7F÷'’6ö×WF—F–öââ"Â&÷&–VçFF–öâ–B"Â'6–ævÆR&WF–æVB&ö¦V7F–öâæVÂ"Â&×W7B&R6F–öæVB2f—7VÂ'7G&7F–öâÂæ÷BÆ—FW&ÂÖæ–föÆB"À¢$W‡FVæFVBFF""Â$gVÆÂ×fV7F÷"æV–v†&÷W&†ööG2ÂÆ–W"ÖF–ÖVç6–öâ6öçG&–'WF–öç2æB÷F–öæÂ&W6–GVÂÖfÖ–Ç’F–væ÷7F–727W÷'BF†R†–v‚ÖF–ÖVç6–öæÂÖV6†æ—6Òf–Wrâ"Â&F–væ÷7F–27W÷'B"Â&6÷6–æRæV–v†&÷W&†ööB†VFÖæBf&–æ6RÖ6öçG&–'WF–öâ&öf–ÆR"Â'7W÷'F–ærf–wW&R&F†W"F†âÖ–âÖV6†æ—6Ò÷fW'f–Wr ¢§w&—FU÷6÷W&6R†6öçG&7G2Â&f–wW&Uö6öçG&7G2æ77b" ¦Æ&VÅöVF—BÂÒF–&&ÆS£§G&–&&ÆR€¢æ—FVÒÂçcóeö7F–öâÂç&VFW%÷&—6µ÷&VGV6VBÀ¢%&–Ö'’ÖV6†æ—6Ò"Â$¶WBF†R†–v‚ÖF–ÖVç6–öæÂÆ–W"×fV7F÷"ÖV6†æ—6Ò2f–wW&Râ"Â%&W6W'fW2&VF&ÆRVçG'’ö–çBf÷"F†RF†V÷'’ö&¦V7Bâ"À¢$6ö÷&F–æFRWf–FVæ6R"Â$ÖW&vVB&VF&6²6öçG&öÇ2v—F‚ÖöFVÂ×7V6–f–2†öÖöÆöw’–çFòf–wW&R"â"Â%6†÷w2v‡’F†RV×—&–6Â6ö÷&F–æFRö&¦V7B&WV—&W2&÷F‚G&¦V7F÷'’&VF÷WG2æBÆ–væVBv–æF÷w2â"À¢$FöÖ–ææ6Rö'6W'f&ÆR"Â%&VçVÖ&W&VB&÷VæFVBFVÇFRFöÖ–ææ6R6öçG&öÇ22f–wW&R2â"Â$¶VW2F†R†–v‚×&—6²FöÖ–ææ6R6Æ–Ò—&VBv—F‚—G2æöâÖÆV²æBæöâÕ"6öçG&öÇ2â"À¢$G–æÖ–6Â7G'V7GW&R"Â$ÖW&vVBÖ÷'†—6Ò6Æ÷7W&Rv—F‚vVöFW6–2÷&W6–GVÂvVöÖWG'’–çFòf–wW&RBâ"Â$Æ–æ·27FFRÖWVF–öâ6Æ÷7W&RFò7G'V7GW&VBG&¦V7F÷'’&W6–GVÇ2v—F†÷WBFF–ær6W&FRF—7Æ’—FVÒâ"À¢$–çFW'fVçF–öâfÆ–FF–öâ"Â%&VçVÖ&W&VB6Æ÷6VBÖÆö÷7FVW&–ær2f–wW&RRâ"Â%&WF–ç2F†R–çFW'fVçF–öâ×7G–ÆRfÆ–FF–öâ–âF†RÖ–âW"â"À¢%&W6–GVÂÖfÖ–Ç’÷fW&fÆ÷r"Â$Ö÷fVB&W6–GVÂÖfÖ–Ç’6÷W&6RFFFòâ÷F–öæÂW‡FVæFVBFF7W÷'B&öÆRâ"Â%&WfVçG2F†RÖW&vVBf–wW&RBg&öÒ&V6öÖ–ærâ÷fW&gVÆÂF6†&ö&Bâ"À¢$Æ÷rÖF–ÖVç6–öæÂ&ö¦V7F–öâ"Â%&WF–æVBW†7FÇ’öæ6R2W‡FVæFVBFFf–rââ"Â$¶VW2&VFW"÷&–VçFF–öâv†–ÆR&÷VæF–ærF†R&W&W6VçFF–öæÂ6Æ–Òâ"À¢$F—7Æ’Ö—FVÒ6÷VçB"Â%&VGV6VBF†RÖ–âf–wW&R6WBg&öÒ6WfVâFòf—fRf–wW&W2â"Â$&WGFW"ÖF6†W2æGW&R'F–6ÆRF—7Æ’Ö—FVÒF—66—Æ–æRâ ¢§w&—FU÷6÷W&6R†Æ&VÅöVF—BÂ&f–wW&UöFWF–ÅöVF—E÷cóbæ77b" ¦÷WGWG2ÂÒÆ—7Bæf–ÆW2†÷WEöF—"ÂGFW&âÒ%ÅÂâ‡7fwÇFgÇF–fgÇær’B"ÂgVÆÂææÖW2ÒE%TR§ÂÒF–&&ÆS£§F–&&ÆR€¢f–ÆRÒ&6VæÖR†÷WGWG2’À¢'—FW2Òf–ÆRæ–æfò†÷WGWG2’G6—¦RÀ¢f÷&ÖBÒFööÇ3£¦f–ÆUöW‡B†÷WGWG2’À¢&6¶VæBÒ%""À¢W‡÷'E÷72Òf–ÆRæ–æfò†÷WGWG2’G6—¦Râ ¢§w&—FU÷6÷W&6R‡Â&W‡÷'E÷÷7VÖÖ'’æ77b" §w&—FTÆ–æW2†2€¢"2æGW&RÖçW67&—Bf–wW&W2ÂcãbÖW&vVB"À¢""À¢7FS‚$vVæW&FVBC¢"Â7—2çF–ÖR‚’’À¢""À¢$&6¶VæC¢"öæÇ’â"À¢$f÷&ÖG3¢5dr&–Ö'’ÂDb6V6öæF'’fV7F÷"ÂD”dbcG’&7FW"â"À¢$6ö×ÆW„†VFÖv2æ÷B&WV—&VC²†VFÖÖÆ–¶RæVÇ2W6RvwÆ÷C"vVöÕ÷F–ÆRâ"À¢%cãb–×ÆVÖVçG2F†R&÷fVBF—7Æ’Ö—FVÒÖW&vRÆââ"À¢$f–wW&RW6W2F†R†–v‚ÖF–ÖVç6–öæÂÆ–W"×fV7F÷"ÖV6†æ—6Ò&W&W6VçFF–öââ"À¢$f–wW&R"ÖW&vW2&VF&6²Ö6ö÷&F–æFR6öçG&öÇ2v—F‚ÖöFVÂ×7V6–f–2†öÖöÆöw’â"À¢$f–wW&R2&WF–ç2&÷VæFVBFVÇFRFöÖ–ææ6R6öçG&öÇ2â"À¢$f–wW&RBÖW&vW2Ö÷'†—6Ò6Æ÷7W&Rv—F‚vVöFW6–2Ö&–2æB&W6–GVÂÖvVöÖWG'’Wf–FVæ6Râ"À¢$f–wW&RR&WF–ç26Æ÷6VBÖÆö÷7FVW&–æræB6‡VffÆRÖçVÆÂfÆ–FF–öââ"À¢$W‡FVæFVBFF&WF–ç2öæRÆ÷rÖF–ÖVç6–öæÂ&W6–GVÂÕ4&ö¦V7F–öâöæÇ’2â÷&–VçFF–öâ–Bâ"À¢$W‡FVæFVBFF"6öçF–ç2†–v‚ÖF–ÖVç6–öæÂæV–v†&÷W&†ööBæBF–ÖVç6–öâÖ6öçG&–'WF–öâF–væ÷7F–72â"À¢%F†R6ö×7B†–v‚ÖF–ÖVç6–öæÂ6ö×÷6—FR—2&Wf–WrÖöæÇ’æBæ÷B&V6öÖÖVæFVBf÷"ÖçW67&—BW6Râ"À¢%6VR6÷W&6UöFFöf–wW&UöFWF–ÅöVF—E÷cóbæ77bæB6÷W&6UöFFöf–wW&Uö6öçG&7G2æ77bf÷"f–wW&RöÆ–7’â"À¢""À¢$f–ÆW3¢"À¢7FS‚"Ò"Â&6VæÖR†÷WGWG2’¢’Âf–ÆRçF‚†÷WEöF—"Â%$TDÔRæÖB"’ ¦ÖW76vR‚$vVæW&FVBæGW&RÖçW67&—Bf–wW&W2cãbÖW&vVB–ã¢"Â÷WEöF—"