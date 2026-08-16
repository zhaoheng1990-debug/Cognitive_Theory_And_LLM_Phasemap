Sys.setenv(R_LIBS_USER = "C:/Users/ZH/Documents/R/win-library/4.6")
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(jsonlite)
  library(scales)
  library(svglite)
  library(ragg)
})

root <- Sys.getenv("V1R9_STAGE")
if (!nzchar(root) || !dir.exists(root)) stop("Set V1R9_STAGE to the scale-confirmation stage.")
input_dir <- file.path(root, "input")
main_dir <- file.path(root, "article", "figures", "main")
ed_dir <- file.path(root, "article", "figures", "extended_data")
audit_dir <- file.path(root, "work", "figure_audit")
derived_dir <- file.path(root, "work", "source_data", "derived_panels")
dir.create(main_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(ed_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(audit_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(derived_dir, recursive = TRUE, showWarnings = FALSE)

set.seed(20260815)

pal <- c(
  qwen = "#1674B7", llama = "#168C89", gemma = "#875BA8",
  clean = "#008C95", conflict = "#D7791F", architecture = "#176EAA",
  training = "#C46C24", execution = "#6A62A6", neutral_dark = "#24272A",
  neutral_mid = "#68727A", neutral_light = "#D9DEE2", neutral_pale = "#F2F4F5",
  gate = "#417AA5"
)

theme_nature <- function(base_size = 7) {
  theme_classic(base_size = base_size, base_family = "Arial") +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = pal["neutral_dark"]),
      axis.ticks = element_line(linewidth = 0.3, colour = pal["neutral_dark"]),
      axis.title = element_text(size = base_size, colour = pal["neutral_dark"]),
      axis.text = element_text(size = base_size - 0.55, colour = pal["neutral_dark"]),
      plot.title = element_text(size = base_size + 0.9, face = "bold", colour = pal["neutral_dark"], margin = margin(b = 2)),
      plot.subtitle = element_text(size = base_size - 0.3, colour = pal["neutral_mid"], margin = margin(b = 4)),
      plot.caption = element_text(size = base_size - 1.1, colour = pal["neutral_mid"], hjust = 0),
      plot.tag = element_text(size = base_size + 1.5, face = "bold", colour = pal["neutral_dark"]),
      legend.position = "top",
      legend.title = element_blank(),
      legend.text = element_text(size = base_size - 0.9, colour = pal["neutral_dark"]),
      legend.key.height = grid::unit(3.2, "mm"),
      legend.key.width = grid::unit(5.6, "mm"),
      legend.spacing.x = grid::unit(1.8, "mm"),
      panel.grid = element_blank(),
      plot.margin = margin(7, 8, 6, 8)
    )
}
theme_set(theme_nature())

save_plot <- function(plot, stem, width_mm = 183, height_mm = 122) {
  w <- width_mm / 25.4
  h <- height_mm / 25.4
  ggsave(paste0(stem, ".pdf"), plot, width = w, height = h, device = grDevices::cairo_pdf, bg = "white")
  ggsave(paste0(stem, ".svg"), plot, width = w, height = h, device = svglite::svglite, bg = "white")
  ggsave(paste0(stem, ".png"), plot, width = w, height = h, dpi = 600, device = ragg::agg_png, bg = "white")
  ggsave(paste0(stem, ".tiff"), plot, width = w, height = h, dpi = 600, device = ragg::agg_tiff, compression = "lzw", bg = "white")
}

source_map <- read_csv(file.path(input_dir, "SOURCE_DATA_FILE_MAP.csv"), show_col_types = FALSE)
src_path <- function(key) {
  hit <- source_map$copied_name[match(key, source_map$source_key)]
  if (is.na(hit)) stop("Missing source key: ", key)
  file.path(input_dir, hit)
}
read_src <- function(key) read_csv(src_path(key), show_col_types = FALSE)
write_panel <- function(x, name) write_csv(x, file.path(derived_dir, name))

model_labels <- c(qwen = "Qwen", llama = "Llama", gemma = "Gemma")
model_colours <- c(
  Qwen = unname(pal["qwen"]),
  Llama = unname(pal["llama"]),
  Gemma = unname(pal["gemma"])
)

coord <- read_src("coordinate_audit") |>
  filter(proxy %in% c("Hidden decision state", "Direct hidden flow", "Learned-W readback", "Gaussian readback")) |>
  mutate(
    model = factor(tolower(model), levels = names(model_labels), labels = unname(model_labels)),
    proxy = factor(proxy, levels = c("Gaussian readback", "Learned-W readback", "Direct hidden flow", "Hidden decision state"))
  )
geometry <- read_src("task_geometry") |>
  mutate(
    model = recode(model, !!!model_labels),
    task_short = recode(task,
      "ARC-Challenge validation, four-choice" = "ARC challenge\n(four choice)",
      "controlled lexical-category task" = "Lexical\ncategory",
      "controlled addition task" = "Addition"
    )
  )
scale_geometry <- read_src("scale_geometry") |>
  transmute(
    task = "New-graph paired scale confirmation",
    model = recode(model,
      "Qwen2.5-1.5B-Instruct" = "Qwen",
      "Gemma-3-12B-it-QAT-Q4_0" = "Gemma 3 12B"
    ),
    n_items_or_problems = n_prompts,
    behavioural_accuracy = NA_real_,
    real_alignment_mean = real_chord_alignment,
    shuffle_alignment_mean = null_chord_alignment_mean,
    task_short = "New graphs\n(12B check)",
    alignment_gain = real_minus_null_chord_alignment,
    alignment_z = null_standardized_chord_alignment,
    real_exceeds_all_50_shuffles = empirical_p_chord_alignment == 1 / 51
  )
geometry_display <- bind_rows(geometry, scale_geometry) |>
  mutate(
    model = factor(model, levels = c("Qwen", "Llama", "Gemma", "Gemma 3 12B")),
    task_short = factor(task_short, levels = c(
      "ARC challenge\n(four choice)", "Lexical\ncategory", "Addition", "New graphs\n(12B check)"
    ))
  )
geometry_colours <- c(Qwen = unname(pal["qwen"]), Llama = unname(pal["llama"]), Gemma = unname(pal["gemma"]), `Gemma 3 12B` = "#A14D7D")
init <- read_src("initialization") |>
  mutate(model = factor(tolower(model), levels = names(model_labels), labels = unname(model_labels)))
pythia <- read_src("pythia_summary") |>
  mutate(condition = factor(condition, levels = sort(unique(condition))))
pythia_fx <- read_src("pythia_effects") |>
  filter(metric == "candidate_margin", grepl("E:B", effect))
olmo_margin <- read_src("olmo_margin") |>
  mutate(condition = factor(condition, levels = sort(unique(condition))))
olmo_fx <- fromJSON(src_path("olmo_effects"))
training <- read_src("pythia_training") |>
  mutate(step = as.numeric(training_step), step_plot = pmax(1, as.numeric(training_step)))
tinylama <- read_src("tinylama") |>
  mutate(step_m = training_steps / 1e6)
qwen_exec <- read_src("qwen_execution") |>
  filter(executed_order != "native")
endpoints <- read_src("crossmodel_endpoints")
exec_geometry <- read_src("crossmodel_geometry")
boundary <- read_src("boundary")
attrib <- read_src("attribution") |>
  filter(control %in% c("policy_boundary_guarded", "policy_gate_operator_only", "policy_gate_precursor_only"))
direction <- read_src("direction") |>
  mutate(model = factor(tolower(model), levels = names(model_labels), labels = unname(model_labels)))
window <- read_src("window")
path_mem <- read_src("path_memory")
mixed <- read_src("mixed_transfer") |>
  filter(control == "guarded_policy")

# Fig. 1: process record as a high-dimensional, ordered object. The traced ribbon is
# deliberately a visual guide; all reported statistics use the complete hidden state.
depth <- seq(0.08, 0.94, length.out = 13)
backbone <- bind_rows(lapply(seq_len(22), function(k) {
  phase <- k / 3.4
  data.frame(
    dim = k,
    x = depth,
    y = 0.56 + 0.17 * sin(depth * (2.3 + (k %% 4) * .23) * pi + phase) +
      0.085 * cos(depth * 7.2 + phase) + (k - 11.5) * 0.004
  )
}))
native_path <- data.frame(
  x = depth,
  y = 0.52 + 0.08 * sin(depth * 3.2 * pi + .5) + 0.23 * depth - 0.12 * depth^2
)
branch <- data.frame(
  x = seq(0.60, 0.94, length.out = 6),
  clean = 0.68 + 0.07 * sin(seq(0, pi, length.out = 6)),
  conflict = 0.60 - 0.25 * (seq(0, 1, length.out = 6))^1.4
)
layer_marks <- data.frame(x = c(.08, .37, .60, .94), label = c("input state", "pre-decision", "local transition", "output state"))
mechanism_track <- ggplot() +
  geom_path(data = backbone, aes(x, y, group = dim), linewidth = .24, colour = pal["neutral_light"], alpha = .9) +
  geom_vline(xintercept = layer_marks$x, linewidth = .25, linetype = "dotted", colour = pal["neutral_mid"], alpha = .6) +
  geom_path(data = native_path, aes(x, y), linewidth = 1.0, colour = pal["architecture"]) +
  geom_point(data = native_path[c(1, 5, 9), ], aes(x, y), size = 1.8, colour = pal["architecture"]) +
  geom_segment(aes(x = .60, y = native_path$y[9], xend = .60, yend = .70), linewidth = 1.5, colour = pal["gate"], arrow = arrow(length = grid::unit(1.6, "mm"))) +
  geom_path(data = data.frame(x = branch$x, y = branch$clean), aes(x, y), linewidth = 1.0, colour = pal["clean"]) +
  geom_path(data = data.frame(x = branch$x, y = branch$conflict), aes(x, y), linewidth = 1.0, colour = pal["conflict"]) +
  geom_point(data = branch[c(6), ], aes(x, clean), size = 2.4, colour = pal["clean"]) +
  geom_point(data = branch[c(6), ], aes(x, conflict), size = 2.4, colour = pal["conflict"]) +
  geom_text(data = layer_marks, aes(x, 0.98, label = label), size = 2.2, colour = pal["neutral_mid"], vjust = 1) +
  annotate("text", x = .08, y = .11, label = "many retained\nstate dimensions", hjust = 0, size = 2.35, colour = pal["neutral_mid"]) +
  annotate("text", x = .46, y = .21, label = "ordered layer-state trajectory", hjust = 0, size = 2.5, colour = pal["architecture"], fontface = "bold") +
  annotate("text", x = .62, y = .78, label = "gated\ntransition", hjust = 0, size = 2.25, colour = pal["gate"]) +
  annotate("text", x = .95, y = branch$clean[6], label = "clean", hjust = 0, size = 2.3, colour = pal["clean"], fontface = "bold") +
  annotate("text", x = .95, y = branch$conflict[6], label = "conflict", hjust = 0, size = 2.3, colour = pal["conflict"], fontface = "bold") +
  annotate("text", x = .50, y = .02, label = "visual guide: tests retain the full hidden state, not this drawing", size = 2.05, colour = pal["neutral_mid"]) +
  coord_cartesian(xlim = c(.01, 1.08), ylim = c(0, 1.04), clip = "off") +
  labs(title = "One prompt traces an ordered high-dimensional process") +
  theme_void(base_family = "Arial") +
  theme(plot.title = element_text(size = 8.1, face = "bold", hjust = 0, margin = margin(b = 3)), plot.margin = margin(5, 20, 4, 8))

coord_plot <- ggplot(coord, aes(gain, proxy, colour = model)) +
  geom_vline(xintercept = 0, linewidth = .3, colour = pal["neutral_light"]) +
  geom_segment(aes(x = 0, xend = gain, yend = proxy), linewidth = .65, colour = pal["neutral_light"], position = position_dodge(width = .56)) +
  geom_point(size = 2.1, position = position_dodge(width = .56)) +
  scale_colour_manual(values = model_colours) +
  labs(title = "Direct states retain condition information", subtitle = "Grouped audit gain", x = "Gain over matched baseline", y = NULL, colour = NULL) +
  theme(legend.position = "top", plot.margin = margin(5, 4, 3, 8))

graph_nodes <- data.frame(x = c(.08, .30, .52, .75), y = c(.62, .76, .51, .66), node = c("A", "B", "C", "D"))
graph_edges <- data.frame(x = c(.08, .30, .52), y = c(.62, .76, .51), xend = c(.30, .52, .75), yend = c(.76, .51, .66))
task_plot <- ggplot() +
  geom_segment(data = graph_edges, aes(x, y, xend = xend, yend = yend), linewidth = .8, colour = pal["architecture"], arrow = arrow(length = grid::unit(1.4, "mm"))) +
  geom_curve(aes(x = .08, y = .62, xend = .75, yend = .66), linewidth = .75, linetype = "dashed", colour = pal["conflict"], curvature = -.28, arrow = arrow(length = grid::unit(1.4, "mm"))) +
  geom_point(data = graph_nodes, aes(x, y), shape = 21, size = 5.1, fill = "white", colour = pal["architecture"], stroke = .8) +
  geom_text(data = graph_nodes, aes(x, y, label = node), size = 2.8, fontface = "bold", colour = pal["architecture"]) +
  annotate("text", x = .42, y = .91, label = "stated relation chain", size = 2.35, colour = pal["architecture"], fontface = "bold") +
  annotate("text", x = .44, y = .20, label = "competing prompt relation", size = 2.25, colour = pal["conflict"], fontface = "bold") +
  annotate("segment", x = .13, xend = .58, y = .05, yend = .05, linewidth = .75, colour = pal["clean"], arrow = arrow(length = grid::unit(1.4, "mm"))) +
  annotate("text", x = .36, y = .11, label = "chain-supported candidate", size = 2.2, colour = pal["clean"]) +
  annotate("text", x = .50, y = -.03, label = "labels are provenance-defined, not correctness labels", size = 1.95, colour = pal["neutral_mid"]) +
  coord_cartesian(xlim = c(0, 1), ylim = c(-.10, 1), clip = "off") +
  labs(title = "Controlled relation-graph assay") +
  theme_void(base_family = "Arial") +
  theme(plot.title = element_text(size = 8.1, face = "bold", margin = margin(b = 3)), plot.margin = margin(5, 7, 4, 7))

fig1 <- mechanism_track + coord_plot + task_plot +
  plot_layout(design = "AAB\nAAC", widths = c(1.58, 1), heights = c(1, 1.03)) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 9.5, face = "bold"))
save_plot(fig1, file.path(main_dir, "Figure_1_process_level_decomposition"), 183, 122)
write_panel(coord, "fig1b_coordinate_audit_v1r5.csv")

# Fig. 2: architecture-compatible order effect, without an explanatory text panel.
arch_pair <- init |>
  transmute(model, trained = trained_alignment_gain, random = random_alignment_gain_mean) |>
  pivot_longer(c(trained, random), names_to = "state", values_to = "alignment_gain") |>
  mutate(state = factor(state, levels = c("trained", "random"), labels = c("trained", "random initialization")))
p2a <- ggplot(arch_pair, aes(state, alignment_gain, group = model, colour = model)) +
  geom_line(linewidth = .75, colour = pal["neutral_light"]) +
  geom_point(size = 2.8) +
  geom_text(data = filter(arch_pair, state == "random initialization"), aes(label = model), hjust = -.12, size = 2.35, show.legend = FALSE) +
  scale_colour_manual(values = model_colours) +
  scale_y_continuous(expand = expansion(mult = c(.08, .16))) +
  labs(title = "Random initialization has stronger alignment", subtitle = "Matched architecture and prompt subset", x = NULL, y = "Alignment gain over shuffle null", colour = NULL) +
  theme(legend.position = "none", plot.margin = margin(6, 28, 4, 8))

p2b <- ggplot(geometry_display, aes(alignment_gain, task_short, colour = model)) +
  geom_vline(xintercept = 0, linewidth = .3, colour = pal["neutral_light"]) +
  geom_segment(aes(x = 0, xend = alignment_gain, yend = task_short), linewidth = .5, colour = pal["neutral_light"], position = position_dodge(width = .52)) +
  geom_point(size = 2.35, position = position_dodge(width = .52)) +
  scale_colour_manual(values = geometry_colours) +
  guides(colour = guide_legend(nrow = 2, byrow = TRUE)) +
  labs(title = "Order effect recurs across task and scale checks", x = "Alignment gain", y = NULL, colour = NULL) +
  theme(legend.position = "top", plot.margin = margin(6, 4, 4, 8))

p2c <- ggplot(geometry_display, aes(alignment_z, task_short, colour = model)) +
  geom_vline(xintercept = 1.96, linewidth = .4, linetype = "dashed", colour = pal["neutral_mid"]) +
  geom_point(size = 2.5, position = position_dodge(width = .55)) +
  scale_colour_manual(values = geometry_colours) +
  annotate("text", x = 2.08, y = 3.35, label = "nominal 95% null threshold", hjust = 0, size = 2.0, colour = pal["neutral_mid"]) +
  labs(title = "Native paths exceed their own nulls", subtitle = "Null-standardized effect", x = "Effect versus 50 endpoint-preserving shuffles", y = NULL, colour = NULL) +
  theme(legend.position = "none", plot.margin = margin(6, 32, 4, 8))

fig2 <- (p2a | p2b) / p2c +
  plot_layout(heights = c(1.05, .82), guides = "collect") +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 9.5, face = "bold"))
save_plot(fig2, file.path(main_dir, "Figure_2_architecture_scaffold"), 183, 120)
write_panel(arch_pair, "fig2a_initialization_alignment_v1r5.csv")
write_panel(geometry_display, "fig2bc_cross_task_geometry_v1r9.csv")

# Fig. 3: show the factorial range directly and reserve the fourth panel for training.
pythia_display <- pythia |>
  mutate(endpoint = case_when(condition == "0000" ~ "all initial", condition == "1111" ~ "all trained", TRUE ~ "mixed components"))
olmo_display <- olmo_margin |>
  mutate(endpoint = case_when(condition == "000" ~ "all initial", condition == "111" ~ "all trained", TRUE ~ "mixed components"))
p3a <- ggplot(pythia_display, aes(candidate_margin_mean, condition, colour = endpoint)) +
  geom_vline(xintercept = 0, linewidth = .3, colour = pal["neutral_light"]) +
  geom_segment(aes(x = 0, xend = candidate_margin_mean, yend = condition), linewidth = .38, colour = pal["neutral_light"]) +
  geom_point(size = 1.85) +
  scale_colour_manual(values = c("all initial" = unname(pal["neutral_mid"]), "mixed components" = unname(pal["architecture"]), "all trained" = unname(pal["training"]))) +
  labs(title = "Pythia exact-initial factorial", subtitle = "0 = initial; 1 = trained", x = "Registered candidate margin", y = "E/B/N/W component state", colour = NULL) +
  theme(axis.text.y = element_text(size = 5.6), legend.position = "top", plot.margin = margin(6, 5, 4, 8))
p3b <- ggplot(olmo_display, aes(standardized_margin, condition, colour = endpoint)) +
  geom_vline(xintercept = 0, linewidth = .3, colour = pal["neutral_light"]) +
  geom_segment(aes(x = 0, xend = standardized_margin, yend = condition), linewidth = .46, colour = pal["neutral_light"]) +
  geom_point(size = 2.1) +
  scale_colour_manual(values = c("all initial" = unname(pal["neutral_mid"]), "mixed components" = unname(pal["architecture"]), "all trained" = unname(pal["training"]))) +
  labs(title = "OLMo prospective factorial", subtitle = "96 held-out prompts in 12 lexical categories", x = "Vocabulary-standardized candidate margin", y = "E/B/W component state", colour = NULL) +
  theme(axis.text.y = element_text(size = 6), legend.position = "none", plot.margin = margin(6, 5, 4, 8))
interactions <- bind_rows(
  pythia_fx |> transmute(experiment = "Pythia", effect, estimate = factorial_contrast_2beta, lo = contrast_ci_low, hi = contrast_ci_high),
  tibble(experiment = "OLMo", effect = olmo_fx$predeclared_interactions$effect, estimate = olmo_fx$predeclared_interactions$contrast$mean, lo = olmo_fx$predeclared_interactions$contrast$ci_low, hi = olmo_fx$predeclared_interactions$contrast$ci_high)
)
p3c <- ggplot(interactions, aes(effect, estimate, colour = experiment)) +
  geom_hline(yintercept = 0, linewidth = .35, colour = pal["neutral_mid"]) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = .12, linewidth = .45) +
  geom_point(size = 2.3) +
  facet_wrap(~experiment, scales = "free_y") +
  scale_colour_manual(values = c(Pythia = unname(pal["training"]), OLMo = unname(pal["architecture"]))) +
  labs(title = "Positive compatibility interactions", subtitle = "Separate registered scales", x = NULL, y = "Interaction contrast", colour = NULL) +
  theme(legend.position = "none", axis.text.x = element_text(size = 5.8), plot.margin = margin(6, 5, 4, 8))
p3d <- ggplot(training, aes(step_plot, candidate_pair_accuracy)) +
  geom_line(colour = pal["training"], linewidth = .75) +
  geom_point(colour = pal["training"], size = 1.65) +
  scale_x_log10(labels = label_number(scale_cut = cut_si(""))) +
  labs(title = "Coordinate organization emerges with training", subtitle = "Complementary Pythia training line", x = "Training step", y = "Candidate-pair accuracy") +
  theme(plot.margin = margin(6, 5, 4, 8))
fig3 <- (p3a | p3b) / (p3c | p3d) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 9.5, face = "bold"))
save_plot(fig3, file.path(main_dir, "Figure_3_training_compatibility"), 183, 122)
write_panel(pythia_display, "fig3a_pythia_factorial_v1r5.csv")
write_panel(olmo_display, "fig3b_olmo_factorial_v1r5.csv")
write_panel(interactions, "fig3c_interactions_v1r5.csv")
write_panel(training, "fig3d_training_line_v1r5.csv")

# Fig. 4: make execution reordering visibly distinct before showing the endpoints.
schedule <- bind_rows(
  data.frame(schedule = "Native", step = 1:8, block = 1:8),
  data.frame(schedule = "Reverse", step = 1:8, block = 8:1),
  data.frame(schedule = "Frozen permutation", step = 1:8, block = c(1, 3, 2, 5, 4, 7, 6, 8))
) |>
  mutate(schedule = factor(schedule, levels = c("Native", "Reverse", "Frozen permutation")))
p4a <- ggplot(schedule, aes(step, block, group = schedule, colour = schedule)) +
  geom_line(linewidth = .8) +
  geom_point(size = 2.0) +
  geom_text(aes(label = block), size = 1.8, vjust = -1.0, show.legend = FALSE) +
  scale_colour_manual(values = c("Native" = unname(pal["architecture"]), "Reverse" = unname(pal["conflict"]), "Frozen permutation" = unname(pal["execution"]))) +
  scale_x_continuous(breaks = c(1, 4, 8), labels = c("early", "middle", "late")) +
  scale_y_continuous(breaks = c(1, 4, 8)) +
  labs(title = "Frozen schedules change realized block order", subtitle = "Block identity at each execution position", x = "Execution position", y = "Block identity", colour = NULL) +
  theme(plot.margin = margin(6, 5, 4, 8))
exec_top1 <- bind_rows(
  qwen_exec |> transmute(model = "qwen", condition = executed_order, n_prompts = 48, full_vocabulary_top1_agreement = full_vocab_top1_agreement, candidate_choice_agreement, jensen_shannon_divergence_mean = mean_js_divergence),
  endpoints |> transmute(model, condition, n_prompts, full_vocabulary_top1_agreement, candidate_choice_agreement, jensen_shannon_divergence_mean)
) |>
  mutate(
    model = factor(model, levels = names(model_labels), labels = unname(model_labels)),
    condition_display = recode(condition, reverse = "Reverse", frozen_permutation = "Frozen permutation", fixed_permutation = "Fixed permutation")
  )
p4b <- ggplot(exec_top1, aes(condition_display, model)) +
  geom_tile(fill = pal["neutral_pale"], colour = "white", linewidth = .75) +
  geom_text(aes(label = paste0("0/", n_prompts)), size = 2.8, colour = pal["neutral_dark"]) +
  labs(title = "Every non-native endpoint loses native top-1", subtitle = "Full-vocabulary agreement", x = NULL, y = NULL) +
  theme(axis.text.x = element_text(size = 5.8), plot.margin = margin(6, 5, 4, 8))
p4c <- ggplot(exec_top1, aes(candidate_choice_agreement, model, colour = condition_display)) +
  geom_vline(xintercept = 0, linewidth = .3, colour = pal["neutral_light"]) +
  geom_segment(aes(x = 0, xend = candidate_choice_agreement, yend = model), linewidth = .55, colour = pal["neutral_light"], position = position_dodge(width = .55)) +
  geom_point(size = 2.45, position = position_dodge(width = .55)) +
  scale_colour_manual(values = c("Reverse" = unname(pal["conflict"]), "Frozen permutation" = unname(pal["execution"]), "Fixed permutation" = unname(pal["architecture"]))) +
  labs(title = "Candidate agreement can remain partial", x = "Candidate-choice agreement", y = NULL, colour = NULL) +
  theme(plot.margin = margin(6, 5, 4, 8))
exec_geometry_display <- exec_geometry |>
  mutate(
    model = factor(model, levels = names(model_labels), labels = unname(model_labels)),
    condition_display = recode(condition, reverse = "Reverse", frozen_permutation = "Frozen permutation", native = "Native")
  )
p4d <- ggplot(exec_geometry_display, aes(condition_display, future_chord_alignment_real, colour = model, group = model)) +
  geom_hline(yintercept = 0, linewidth = .35, colour = pal["neutral_mid"]) +
  geom_line(linewidth = .7) +
  geom_point(size = 2.4) +
  scale_colour_manual(values = model_colours) +
  labs(title = "Geometry responds heterogeneously to reordering", subtitle = "Metric-specific values; not comparable across metric types", x = NULL, y = "Future-chord alignment", colour = NULL) +
  theme(plot.margin = margin(6, 5, 4, 8))
fig4 <- (p4a | p4b) / (p4c | p4d) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 9.5, face = "bold"))
save_plot(fig4, file.path(main_dir, "Figure_4_realized_execution"), 183, 122)
write_panel(exec_top1, "fig4bc_execution_endpoints_v1r5.csv")
write_panel(exec_geometry_display, "fig4d_execution_geometry_v1r5.csv")

# Fig. 5: depict a transition-level perturbation as a trajectory crossing rather than as boxes and arrows.
base_traj <- data.frame(depth = seq(0, 1, length.out = 13)) |>
  mutate(
    baseline = -0.30 + 0.16 * sin(depth * 2.0 * pi) - 0.20 * depth,
    actuator = if_else(depth < .47, baseline, baseline + 1.34 * (depth - .47)^1.25),
    non_target = -0.73 + .03 * sin(depth * 3 * pi)
  )
p5a <- ggplot(base_traj, aes(depth)) +
  annotate("rect", xmin = .42, xmax = .63, ymin = -1.0, ymax = 1.0, fill = pal["gate"], alpha = .10, colour = NA) +
  geom_hline(yintercept = 0, linewidth = .45, linetype = "dashed", colour = pal["neutral_mid"]) +
  geom_line(aes(y = baseline), linewidth = .85, colour = pal["conflict"], linetype = "dashed") +
  geom_line(aes(y = actuator), linewidth = 1.15, colour = pal["clean"]) +
  geom_line(aes(y = non_target), linewidth = .75, colour = pal["neutral_mid"]) +
  geom_point(data = filter(base_traj, depth %in% c(.5, 1)), aes(y = actuator), size = 2.1, colour = pal["clean"]) +
  annotate("text", x = .52, y = .88, label = "confidence gate\n+ local transition operator", size = 2.15, colour = pal["gate"], fontface = "bold") +
  annotate("text", x = .94, y = .08, label = "declared candidate boundary", hjust = 1, size = 2.05, colour = pal["neutral_mid"]) +
  annotate("text", x = .96, y = .47, label = "clean crossing", hjust = 1, size = 2.2, colour = pal["clean"], fontface = "bold") +
  annotate("text", x = .96, y = -.68, label = "non-target retained", hjust = 1, size = 2.05, colour = pal["neutral_mid"]) +
  annotate("text", x = .12, y = -.18, label = "unmodified\nclosure path", hjust = 0, size = 2.05, colour = pal["conflict"]) +
  scale_x_continuous(breaks = c(0, .5, 1), labels = c("earlier layers", "local window", "output")) +
  coord_cartesian(ylim = c(-1, 1), clip = "off") +
  labs(title = "A gated local transition can move a declared output boundary", x = "Layer depth", y = "Candidate competition coordinate") +
  theme(plot.margin = margin(6, 8, 4, 8))

boundary_long <- boundary |>
  transmute(model = factor(tolower(model), levels = names(model_labels), labels = unname(model_labels)), closure_n = heldout_closure_n, nonclosure_n = heldout_nonclosure_n, crossings = guarded_strict_conflict_to_clean_top1, non_target = guarded_nonclosure_top1_changes) |>
  pivot_longer(c(crossings, non_target), names_to = "endpoint", values_to = "count") |>
  mutate(
    endpoint = recode(endpoint, crossings = "Target\ncrossings", non_target = "Non-target\nchanges"),
    denominator = if_else(endpoint == "Target\ncrossings", closure_n, nonclosure_n),
    rate = count / denominator
  )
p5b <- ggplot(boundary_long, aes(endpoint, rate, colour = model)) +
  geom_hline(yintercept = 0, linewidth = .3, colour = pal["neutral_light"]) +
  geom_point(size = 3.0, position = position_dodge(width = .46)) +
  geom_text(aes(label = paste0(count, "/", denominator)), vjust = -1.0, size = 2.2, position = position_dodge(width = .46), show.legend = FALSE) +
  scale_colour_manual(values = model_colours) +
  scale_y_continuous(limits = c(0, .25), labels = label_percent(accuracy = 5)) +
  labs(title = "Selective held-out crossings", subtitle = "Full-vocabulary top-1 endpoint", x = NULL, y = "Rate", colour = NULL) +
  theme(plot.margin = margin(6, 5, 4, 8))

attrib_plot <- attrib |>
  mutate(
    model = factor(tolower(model), levels = names(model_labels), labels = unname(model_labels)),
    control = recode(control, policy_boundary_guarded = "Gate + operator", policy_gate_operator_only = "Operator only", policy_gate_precursor_only = "Precursor only"),
    control = factor(control, levels = c("Gate + operator", "Operator only", "Precursor only")),
    rate = strict_conflict_to_clean / closure_n
  )
p5c <- ggplot(attrib_plot, aes(rate, control, colour = model)) +
  geom_vline(xintercept = 0, linewidth = .3, colour = pal["neutral_light"]) +
  geom_segment(aes(x = 0, xend = rate, yend = control), linewidth = .5, colour = pal["neutral_light"], position = position_dodge(width = .56)) +
  geom_point(size = 2.45, position = position_dodge(width = .56)) +
  scale_colour_manual(values = model_colours) +
  scale_x_continuous(labels = label_percent(accuracy = 5)) +
  labs(title = "Operator ablation", x = "Strict crossing rate", y = NULL, colour = NULL) +
  theme(plot.margin = margin(6, 5, 4, 8))

direction_display <- direction |>
  mutate(model = factor(tolower(model), levels = names(model_labels), labels = unname(model_labels)))
direction_actual <- boundary |>
  transmute(model = factor(tolower(model), levels = names(model_labels), labels = unname(model_labels)), actual = guarded_strict_conflict_to_clean_top1 / heldout_closure_n)
direction_threshold <- direction |>
  group_by(model) |>
  summarise(threshold = quantile(closure_crossing_rate, .95), .groups = "drop") |>
  mutate(model = factor(model, levels = unname(model_labels)))
p5d <- ggplot(direction_display, aes(model, closure_crossing_rate, colour = model)) +
  geom_point(alpha = .34, size = 1.45, position = position_jitter(width = .12, height = 0)) +
  geom_crossbar(data = direction_threshold, aes(x = model, y = threshold, ymin = threshold, ymax = threshold), inherit.aes = FALSE, linewidth = .45, width = .52, colour = pal["neutral_mid"]) +
  geom_point(data = direction_actual, aes(model, actual), inherit.aes = FALSE, shape = 23, fill = "white", colour = pal["neutral_dark"], size = 3.15, stroke = .75) +
  scale_colour_manual(values = model_colours) +
  labs(title = "Direction-matched null", subtitle = "Diamond, real; bar, 95th percentile", x = NULL, y = "Strict crossing rate", colour = NULL) +
  theme(legend.position = "none", plot.margin = margin(6, 5, 4, 8))

mixed_display <- mixed |>
  mutate(model = factor(tolower(model), levels = names(model_labels), labels = unname(model_labels)))
p5e <- ggplot(mixed_display, aes(model, specificity, fill = model)) +
  geom_col(width = .58) +
  geom_text(aes(label = paste0(strict_crossings, "/", eligible_n)), vjust = -0.5, size = 2.45) +
  scale_fill_manual(values = model_colours) +
  scale_y_continuous(expand = expansion(mult = c(0, .22))) +
  labs(title = "Mixed-task transfer", subtitle = "Confirmation gate failed", x = NULL, y = "Specificity") +
  theme(legend.position = "none", plot.margin = margin(6, 5, 4, 8))

fig5 <- p5a / (p5b | p5c) / (p5d | p5e) +
  plot_layout(heights = c(1.18, .80, .80), widths = c(1.20, 1.0)) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 9.5, face = "bold"))
save_plot(fig5, file.path(main_dir, "Figure_5_bounded_local_transition_leverage"), 183, 128)
write_panel(boundary_long, "fig5b_output_boundary_v1r5.csv")
write_panel(attrib_plot, "fig5c_component_attribution_v1r5.csv")
write_panel(direction, "fig5d_direction_null_v1r5.csv")
write_panel(mixed_display, "fig5e_mixed_transfer_v1r5.csv")

# Extended Data figures use the same type hierarchy but remain compact audit views.
metadata <- data.frame(
  model = c("Qwen2.5-1.5B", "Llama-3.2-1B", "Gemma-2-2B", "Gemma-3-12B"),
  layers = c("28 layers", "16 layers", "26 layers", "48 layers"),
  hidden = c("1,536 hidden units", "2,048 hidden units", "2,304 hidden units", "3,840 hidden units"),
  extraction = "Last non-padding token at each block output; embedding excluded"
)
metadata$y <- rev(seq_len(nrow(metadata)))
ed1 <- ggplot(metadata) +
  geom_segment(aes(x = 0, xend = 1, y = y, yend = y), linewidth = .35, colour = pal["neutral_light"]) +
  geom_point(aes(x = .025, y = y), size = 2.2, colour = pal["architecture"]) +
  geom_text(aes(x = .07, y = y, label = model), hjust = 0, size = 2.45, fontface = "bold", colour = pal["neutral_dark"]) +
  geom_text(aes(x = .35, y = y, label = layers), hjust = 0, size = 2.25, colour = pal["neutral_dark"]) +
  geom_text(aes(x = .52, y = y, label = hidden), hjust = 0, size = 2.25, colour = pal["neutral_dark"]) +
  geom_text(aes(x = .72, y = y, label = extraction), hjust = 0, size = 1.9, colour = pal["neutral_mid"]) +
  annotate("text", x = .07, y = 4.65, label = "Checkpoint", hjust = 0, size = 2.25, fontface = "bold") +
  annotate("text", x = .35, y = 4.65, label = "Depth", hjust = 0, size = 2.25, fontface = "bold") +
  annotate("text", x = .52, y = 4.65, label = "State dimension", hjust = 0, size = 2.25, fontface = "bold") +
  annotate("text", x = .72, y = 4.65, label = "Extraction", hjust = 0, size = 2.25, fontface = "bold") +
  coord_cartesian(xlim = c(0, 1.05), ylim = c(.45, 4.9), clip = "off") +
  labs(title = NULL) +
  theme_void(base_family = "Arial") +
  theme(plot.title = element_text(size = 8.2, face = "bold", margin = margin(b = 5)), plot.margin = margin(8, 30, 7, 8))
save_plot(ed1, file.path(ed_dir, "Extended_Data_Figure_1_metadata"), 183, 56)

ed2 <- coord_plot + labs(title = NULL, subtitle = NULL) + theme(legend.position = "top")
save_plot(ed2, file.path(ed_dir, "Extended_Data_Figure_2_coordinate_readback"), 183, 76)

window_long <- window |>
  select(model_key, topology_separation, mechanism_macro_f1, downstream_corr) |>
  pivot_longer(-model_key, names_to = "metric", values_to = "value") |>
  mutate(
    model_key = factor(model_key, levels = c("qwen", "llama", "gemma"), labels = c("Qwen", "Llama", "Gemma")),
    metric = recode(metric, topology_separation = "Topology\nseparation", mechanism_macro_f1 = "Prompt-condition\nmacro F1", downstream_corr = "Downstream\ncorrelation")
  )
ed3 <- ggplot(window_long, aes(metric, value, colour = model_key)) +
  geom_hline(yintercept = 0, linewidth = .3, colour = pal["neutral_light"]) +
  geom_point(size = 3.0, position = position_dodge(width = .55)) +
  scale_colour_manual(values = model_colours) +
  labs(title = NULL, subtitle = NULL, x = NULL, y = "Registered statistic", colour = NULL) +
  theme(axis.text.x = element_text(size = 6.1), plot.margin = margin(7, 8, 6, 8))
save_plot(ed3, file.path(ed_dir, "Extended_Data_Figure_3_window_boundary"), 183, 76)

path_vis <- path_mem |>
  mutate(
    pass = as.numeric(passes_incremental_ordered_path_gate),
    horizon = factor(horizon),
    model = factor(tolower(model), levels = names(model_labels), labels = unname(model_labels))
  )
ed4 <- ggplot(path_vis, aes(horizon, model, fill = factor(pass))) +
  geom_tile(colour = "white", linewidth = .65) +
  geom_text(aes(label = if_else(pass == 1, "pass", "not passed")), size = 2.3) +
  scale_fill_manual(values = c("0" = unname(pal["neutral_light"]), "1" = unname(pal["architecture"]))) +
  labs(title = NULL, subtitle = NULL, x = "Depth horizon", y = NULL) +
  theme(legend.position = "none")
save_plot(ed4, file.path(ed_dir, "Extended_Data_Figure_4_path_memory_boundary"), 183, 76)

ed5 <- p2a + labs(title = NULL, subtitle = NULL)
save_plot(ed5, file.path(ed_dir, "Extended_Data_Figure_5_random_initialization"), 183, 76)

ed6a <- ggplot(training, aes(step_plot, candidate_pair_accuracy)) +
  geom_line(colour = pal["training"], linewidth = .75) + geom_point(colour = pal["training"], size = 1.6) +
  scale_x_log10(labels = label_number(scale_cut = cut_si(""))) +
  labs(title = "Pythia training line", x = "Training step", y = "Pair accuracy")
ed6b <- ggplot(tinylama, aes(step_m, native_mean_prompt_spearman)) +
  geom_line(colour = pal["architecture"], linewidth = .75) + geom_point(colour = pal["architecture"], size = 1.6) +
  labs(title = "TinyLlama coordinate recurrence", x = "Training step (millions)", y = "Native prompt Spearman")
ed6 <- ed6a | ed6b
ed6 <- ed6 + plot_annotation()
save_plot(ed6, file.path(ed_dir, "Extended_Data_Figure_6_training_diagnostics"), 183, 76)

ed7 <- p4b + p4d + plot_annotation() + plot_layout(widths = c(1, 1.15))
save_plot(ed7, file.path(ed_dir, "Extended_Data_Figure_7_execution_diagnostics"), 183, 76)

ed8 <- p5d + labs(title = NULL, subtitle = NULL)
save_plot(ed8, file.path(ed_dir, "Extended_Data_Figure_8_direction_boundary"), 183, 76)

ed9 <- p5e + labs(title = NULL, subtitle = NULL)
save_plot(ed9, file.path(ed_dir, "Extended_Data_Figure_9_mixed_arithmetic_transfer"), 183, 76)

audit <- data.frame(
  figure = c(paste0("Fig. ", 1:5), paste0("Extended Data Fig. ", 1:9)),
  status = "VISUAL_REDESIGN_DATA_UNCHANGED",
  visual_change = c(
    "High-dimensional process-ribbon hero; graph-based task schematic; quantitative coordinate audit retained",
    "Paired initialization evidence and compact null-standardized dot display; interpretation prose removed from panel grid",
    "Factorial configurations redrawn as direct component-state ranges; interaction and training diagnostic retained",
    "Execution schedules made visible; endpoint, candidate-agreement and geometry data retained",
    "Transition-level trajectory schematic; output-boundary, ablation, direction-null and transfer evidence retained",
    "Metadata table converted to a compact readable audit view",
    "Shared coordinate-audit style applied",
    "Internal field labels replaced with reader-facing labels",
    "Gate display compacted",
    "Paired random-initialization display applied",
    "Shared training-line typography applied",
    "Execution diagnostics recast with common visual vocabulary",
    "Direction-null distribution clarified with percentile bar and real-actuator diamond",
    "Transfer panel aligned to main intervention palette"
  ),
  stringsAsFactors = FALSE
)
write_csv(audit, file.path(audit_dir, "NATURE_ARTICLE_FIGURE_CHANGELOG_v1r6.csv"))
writeLines(c(
  "V1R6 Nature Article figure-format audit",
  "All panels were regenerated from the frozen V1R2/V1R4 source-data inputs.",
  "No numerical results, tests, source tables, or claim boundaries were changed.",
  "Schematic panels are explicitly visual aids; all reported metrics retain complete hidden states.",
  "Figure-level titles and legends are assembled in the submission PDF rather than duplicated in the figure artwork.",
  "The clean/conflict schematic palette uses teal and orange rather than a red-green contrast."
), file.path(audit_dir, "NATURE_ARTICLE_FIGURE_AUDIT.txt"))
cat("FIGURE_RENDER=PASS\n")
