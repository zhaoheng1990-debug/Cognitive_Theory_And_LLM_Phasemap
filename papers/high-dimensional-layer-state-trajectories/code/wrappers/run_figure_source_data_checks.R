#!/usr/bin/env Rscript

# Smoke-check staged figure Source Data from a repository-root-relative config.
# This validates the public Source Data package; it does not run the full
# historical figure-generation script.

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

parse_simple_yaml <- function(path) {
  lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
  result <- list()
  section <- NULL
  for (raw in lines) {
    line <- sub("#.*$", "", raw)
    if (!nzchar(trimws(line))) next
    indent <- nchar(line) - nchar(sub("^ *", "", line))
    stripped <- trimws(line)
    if (!grepl(":", stripped, fixed = TRUE)) next
    key <- trimws(sub(":.*$", "", stripped))
    value <- trimws(sub("^[^:]*:", "", stripped))
    value <- gsub("^[\"']|[\"']$", "", value)
    if (indent == 0 && !nzchar(value)) {
      section <- key
      result[[section]] <- list()
    } else if (indent == 0) {
      result[[key]] <- value
      section <- NULL
    } else if (!is.null(section)) {
      result[[section]][[key]] <- value
    }
  }
  result
}

json_escape <- function(x) {
  x <- gsub("\\\\", "\\\\\\\\", x)
  x <- gsub('"', '\\"', x)
  x <- gsub("\n", "\\n", x, fixed = TRUE)
  x
}

to_json_value <- function(x, indent = 0) {
  pad <- paste(rep(" ", indent), collapse = "")
  if (is.list(x) && !is.data.frame(x)) {
    if (length(x) == 0) return("[]")
    if (is.null(names(x)) || all(names(x) == "")) {
      inner <- vapply(x, to_json_value, character(1), indent = indent + 2)
      paste0("[\n", paste0(pad, "  ", inner, collapse = ",\n"), "\n", pad, "]")
    } else {
      inner <- vapply(names(x), function(nm) {
        paste0(pad, "  \"", json_escape(nm), "\": ", to_json_value(x[[nm]], indent + 2))
      }, character(1))
      paste0("{\n", paste(inner, collapse = ",\n"), "\n", pad, "}")
    }
  } else if (is.logical(x)) {
    ifelse(x, "true", "false")
  } else if (is.numeric(x)) {
    ifelse(is.finite(x), as.character(x), "null")
  } else if (is.null(x)) {
    "null"
  } else {
    paste0("\"", json_escape(as.character(x)), "\"")
  }
}

write_json <- function(x, path) {
  writeLines(to_json_value(x), path, useBytes = TRUE)
}

display_path <- function(path, repo_root) {
  rel <- tryCatch(normalizePath(path, mustWork = FALSE), error = function(e) path)
  root <- normalizePath(repo_root, mustWork = FALSE)
  if (startsWith(rel, root)) {
    rel <- substring(rel, nchar(root) + 2)
  }
  gsub("\\\\", "/", rel)
}

args <- parse_cli_args(commandArgs(trailingOnly = TRUE))
config_path <- if (!is.null(args$config)) args$config else "code/config/example_paths.yaml"
config_path <- normalizePath(config_path, mustWork = TRUE)
config <- parse_simple_yaml(config_path)
repo_root <- if (!is.null(args[["repo-root"]])) args[["repo-root"]] else config$project_root
if (is.null(repo_root) || repo_root == ".") {
  repo_root <- normalizePath(file.path(dirname(config_path), "..", ".."), mustWork = TRUE)
} else {
  repo_root <- normalizePath(repo_root, mustWork = TRUE)
}

source_data_dir <- file.path(repo_root, config$data$source_data_dir)
metadata_dir <- file.path(repo_root, config$data$metadata_dir)
output_dir <- file.path(repo_root, config$data$output_dir)
figure_output_dir <- file.path(repo_root, config$figures$export_dir)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_output_dir, recursive = TRUE, showWarnings = FALSE)

required_packages <- c("ggplot2", "patchwork", "dplyr", "tidyr", "scales", "svglite", "ragg")
package_status <- lapply(required_packages, function(pkg) {
  list(package = pkg, available = requireNamespace(pkg, quietly = TRUE))
})
missing_packages <- vapply(package_status, function(x) !isTRUE(x$available), logical(1))

source_files <- list.files(source_data_dir, pattern = "\\.csv$", full.names = TRUE)
manifest_path <- file.path(metadata_dir, "source_data_manifest_v0_1.csv")
manifest <- read.csv(manifest_path, check.names = FALSE, stringsAsFactors = FALSE)
manifest_files <- basename(manifest$file)
available_files <- basename(source_files)
missing_manifest_files <- setdiff(manifest_files, available_files)

row_counts <- lapply(source_files, function(path) {
  df <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE, nrows = 100000)
  list(file = basename(path), rows = nrow(df), columns = ncol(df))
})

preview_path <- file.path(figure_output_dir, "figure_source_data_smoke_preview.png")
if (!any(missing_packages) && file.exists(file.path(source_data_dir, "figure_freeze_manifest_v0_4.csv"))) {
  freeze <- read.csv(file.path(source_data_dir, "figure_freeze_manifest_v0_4.csv"), check.names = FALSE, stringsAsFactors = FALSE)
  label_col <- if ("figure_or_table" %in% names(freeze)) "figure_or_table" else "file_stem"
  freeze$label_name <- as.character(freeze[[label_col]])
  freeze$label_value <- nchar(as.character(freeze[[label_col]]))
  png(preview_path, width = 1400, height = 800, res = 160)
  print(
    ggplot2::ggplot(freeze, ggplot2::aes(x = reorder(label_name, label_value), y = label_value)) +
      ggplot2::geom_col(fill = "#3182BD", width = 0.65) +
      ggplot2::coord_flip() +
      ggplot2::labs(x = NULL, y = "Label length", title = "Figure Source Data smoke preview") +
      ggplot2::theme_classic(base_size = 8) +
      ggplot2::theme(plot.margin = ggplot2::margin(8, 8, 8, 8))
  )
  dev.off()
}

report <- list(
  repo_root = ".",
  source_data_dir = display_path(source_data_dir, repo_root),
  metadata_dir = display_path(metadata_dir, repo_root),
  figure_output_dir = display_path(figure_output_dir, repo_root),
  source_csv_count = length(source_files),
  manifest_rows = nrow(manifest),
  missing_manifest_files = as.list(missing_manifest_files),
  package_status = package_status,
  row_counts = row_counts,
  preview = if (file.exists(preview_path)) display_path(preview_path, repo_root) else NULL
)

report_path <- file.path(output_dir, "figure_source_data_check_report.json")
write_json(report, report_path)
message("Wrote ", report_path)
message("Source CSV files: ", length(source_files))
message("Missing manifest files: ", length(missing_manifest_files))
message("Missing packages: ", sum(missing_packages))

quit(status = ifelse(length(missing_manifest_files) == 0, 0, 1))
