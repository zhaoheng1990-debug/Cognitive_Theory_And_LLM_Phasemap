Sys.setenv(R_LIBS_USER = "C:/Users/ZH/Documents/R/win-library/4.6")
root <- Sys.getenv("V1R9_STAGE")
if (!nzchar(root) || !dir.exists(root)) stop("Set V1R9_STAGE to the scale-confirmation stage.")

source_file <- file.path(root, "article", "14_nature_article_submission_manuscript.md")
tex_file <- file.path(root, "article", "nature_article_initial_submission_v1r9.tex")
lines <- readLines(source_file, warn = FALSE, encoding = "UTF-8")

protect_urls <- function(x) {
  saved <- character()
  loc <- gregexpr("https?://[^[:space:]]+", x, perl = TRUE)[[1]]
  if (loc[1] == -1) return(list(text = x, saved = saved))
  len <- attr(loc, "match.length")
  for (i in rev(seq_along(loc))) {
    token <- paste0("@@URL", length(saved) + 1, "@@")
    value <- substr(x, loc[i], loc[i] + len[i] - 1)
    saved <- c(saved, value)
    x <- paste0(substr(x, 1, loc[i] - 1), token, substr(x, loc[i] + len[i], nchar(x)))
  }
  list(text = x, saved = saved)
}

restore_urls <- function(x, saved) {
  if (length(saved) == 0) return(x)
  for (i in seq_along(saved)) x <- gsub(paste0("@@URL", i, "@@"), paste0("\\url{", saved[i], "}"), x, fixed = TRUE)
  x
}

protect_math <- function(x) {
  saved <- character()
  repeat {
    m <- regexpr("\\\\\\([^\\n]*?\\\\\\)", x, perl = TRUE)
    if (m[1] == -1) break
    token <- paste0("@@MATH", length(saved) + 1, "@@")
    saved <- c(saved, regmatches(x, m))
    x <- paste0(substr(x, 1, m[1] - 1), token, substr(x, m[1] + attr(m, "match.length"), nchar(x)))
  }
  list(text = x, saved = saved)
}

restore_math <- function(x, saved) {
  if (length(saved) == 0) return(x)
  for (i in seq_along(saved)) x <- gsub(paste0("@@MATH", i, "@@"), saved[i], x, fixed = TRUE)
  x
}

escape_plain <- function(x) {
  literal_replace <- function(value, from, to) paste(strsplit(value, from, fixed = TRUE)[[1]], collapse = to)
  x <- literal_replace(x, "&", "\\&")
  x <- literal_replace(x, "%", "\\%")
  x <- literal_replace(x, "#", "\\#")
  x <- literal_replace(x, "_", "\\_")
  x <- literal_replace(x, "{", "\\{")
  x <- literal_replace(x, "}", "\\}")
  x
}

inline_tex <- function(x) {
  u <- protect_urls(x)
  p <- protect_math(u$text)
  y <- escape_plain(p$text)
  y <- gsub("<sup>([^<]+)</sup>", "\\\\textsuperscript{\\1}", y, perl = TRUE)
  y <- gsub("`([^`]+)`", "\\\\codeid{\\1}", y, perl = TRUE)
  y <- gsub("\\*\\*([^*]+)\\*\\*", "\\\\textbf{\\1}", y, perl = TRUE)
  y <- gsub("\\*([^*]+)\\*", "\\\\emph{\\1}", y, perl = TRUE)
  y <- gsub("([A-Fa-f0-9]{24,})", "\\\\seqsplit{\\1}", y, perl = TRUE)
  y <- restore_math(y, p$saved)
  restore_urls(y, u$saved)
}

table_cells <- function(line) {
  cells <- strsplit(sub("^\\|", "", sub("\\|$", "", line)), "\\|", fixed = FALSE)[[1]]
  trimws(cells)
}

table_spec <- function(n) {
  widths <- switch(as.character(n),
    "2" = c(.28, .64),
    "3" = c(.23, .34, .35),
    "4" = c(.15, .34, .16, .27),
    rep(.92 / n, n)
  )
  paste0("@{}", paste0("L{", sprintf("%.3f", widths), "\\linewidth}", collapse = ""), "@{}")
}

title <- sub("^# ", "", lines[grep("^# ", lines)[1]])
out <- c(
  "\\documentclass[12pt]{article}",
  "\\usepackage{fontspec}",
  "\\IfFontExistsTF{Times New Roman}{\\setmainfont{Times New Roman}}{\\setmainfont{TeX Gyre Termes}}",
  "\\usepackage[margin=0.80in]{geometry}",
  "\\usepackage{amsmath,amssymb,graphicx,booktabs,longtable,array,caption,hyperref,xurl,seqsplit,ragged2e,setspace}",
  "\\usepackage[switch]{lineno}",
  "\\hypersetup{colorlinks=true,linkcolor=black,urlcolor=blue,citecolor=black,breaklinks=true}",
  "\\urlstyle{same}",
  "\\Urlmuskip=0mu plus 1mu",
  "\\setlength{\\emergencystretch}{3em}",
  "\\sloppy",
  "\\setlength{\\parindent}{0pt}",
  "\\setlength{\\parskip}{4.3pt}",
  "\\renewcommand{\\arraystretch}{1.18}",
  "\\newcolumntype{L}[1]{>{\\RaggedRight\\arraybackslash}p{#1}}",
  "\\newcommand{\\codeid}[1]{{\\ttfamily\\footnotesize #1}}",
  paste0("\\title{", inline_tex(title), "}"),
  "\\author{Heng Zhao\\textsuperscript{1,*}, Yufei Wang\\textsuperscript{2}, Ruidian Song\\textsuperscript{3}, Weisheng Chen\\textsuperscript{4}}",
  "\\date{}",
  "\\begin{document}",
  "\\maketitle",
  "\\begin{center}\\small",
  "\\textsuperscript{1}Independent Researcher, Beijing 100071, China.\\\\",
  "\\textsuperscript{2}China Mobile Communications Group Co., Ltd., Beijing 100054, China.\\\\",
  "\\textsuperscript{3}Sinotrans Logistics Co., Ltd., Beijing 100054, China.\\\\",
  "\\textsuperscript{4}Independent Researcher.\\\\",
  "\\textsuperscript{*}Correspondence: Heng Zhao (zhaoheng1990@gmail.com). ORCID: 0009-0004-2395-9393.",
  "\\end{center}",
  "\\doublespacing",
  "\\modulolinenumbers[5]",
  "\\linenumbers"
)

start <- grep("^## Summary$", lines)[1]
body <- lines[start:length(lines)]
in_display <- FALSE
in_table <- FALSE
table_header <- TRUE
summary_open <- FALSE
legend_mode <- NULL
main_legends <- character()
extended_data_legends <- character()
extended_table_lines <- character()
in_extended_tables <- FALSE
for (line in body) {
  trimmed <- trimws(line)
  if (identical(trimmed, "## Figure legends")) {
    legend_mode <- "main"
    next
  }
  if (identical(trimmed, "## Extended Data figure legends")) {
    legend_mode <- "extended"
    next
  }
  if (identical(trimmed, "## Extended Data tables")) {
    legend_mode <- NULL
    in_extended_tables <- TRUE
    next
  }
  if (in_extended_tables) {
    extended_table_lines <- c(extended_table_lines, line)
    next
  }
  if (!is.null(legend_mode)) {
    if (nzchar(trimmed)) {
      if (identical(legend_mode, "main")) main_legends <- c(main_legends, trimmed)
      if (identical(legend_mode, "extended")) extended_data_legends <- c(extended_data_legends, trimmed)
    }
    next
  }
  if (in_display) {
    out <- c(out, line)
    if (trimws(line) == "\\]") in_display <- FALSE
    next
  }
  if (trimws(line) == "\\[") {
    out <- c(out, "\\[")
    in_display <- TRUE
    next
  }
  if (grepl("^\\|", line)) {
    cells <- table_cells(line)
    if (all(grepl("^-+", cells))) next
    if (!in_table) {
      out <- c(out, "\\begingroup\\footnotesize\\setlength{\\tabcolsep}{2.5pt}", paste0("\\begin{longtable}{", table_spec(length(cells)), "}"), "\\toprule")
      in_table <- TRUE
      table_header <- TRUE
    }
    row <- paste(vapply(cells, inline_tex, character(1)), collapse = " & ")
    out <- c(out, paste0(row, " ", strrep("\\\\", 1)))
    if (table_header) {
      out <- c(out, "\\midrule", "\\endfirsthead", "\\toprule", row, " \\\\ ", "\\midrule", "\\endhead")
      table_header <- FALSE
    }
    next
  }
  if (in_table) {
    out <- c(out, "\\bottomrule", "\\end{longtable}", "\\endgroup")
    in_table <- FALSE
  }
  if (trimws(line) == "") {
    out <- c(out, "")
    next
  }
  if (grepl("^## ", line)) {
    heading <- sub("^## ", "", line)
    if (heading == "Summary") {
      out <- c(out, "\\begin{center}\\begin{minipage}{0.91\\linewidth}\\small\\noindent\\bfseries")
      summary_open <- TRUE
    } else if (heading == "Main text") {
      if (summary_open) out <- c(out, "\\normalfont", "\\end{minipage}\\end{center}")
      summary_open <- FALSE
    } else {
      out <- c(out, paste0("\\section*{", inline_tex(heading), "}"))
    }
    next
  }
  if (grepl("^### ", line)) {
    out <- c(out, paste0("\\subsection*{", inline_tex(sub("^### ", "", line)), "}"))
    next
  }
  if (grepl("^#### ", line)) {
    out <- c(out, paste0("\\subsubsection*{", inline_tex(sub("^#### ", "", line)), "}"))
    next
  }
  out <- c(out, inline_tex(line))
}
if (in_table) out <- c(out, "\\bottomrule", "\\end{longtable}", "\\endgroup")
if (summary_open) out <- c(out, "\\normalfont", "\\end{minipage}\\end{center}")

if (length(main_legends) != 5L) stop("Expected five main-figure legends, found ", length(main_legends))
if (length(extended_data_legends) != 9L) stop("Expected nine Extended Data legends, found ", length(extended_data_legends))

figure_block <- function(label, path, legend, height) {
  c(
    "\\clearpage",
    "\\nolinenumbers",
    "\\begin{spacing}{1.0}",
    paste0("\\noindent\\textbf{", label, "}\\par\\medskip"),
    paste0("\\begin{center}\\includegraphics[width=\\linewidth,height=", height, ",keepaspectratio]{", path, "}\\end{center}"),
    "\\vspace{1mm}",
    "\\begin{minipage}{\\linewidth}\\footnotesize\\setstretch{1.12}",
    inline_tex(legend),
    "\\end{minipage}",
    "\\end{spacing}",
    "\\linenumbers"
  )
}

extended_table_block <- function(lines) {
  out <- c(
    "\\clearpage",
    "\\nolinenumbers",
    "\\section*{Extended Data tables}"
  )
  in_table <- FALSE
  table_header <- TRUE
  for (line in lines) {
    trimmed <- trimws(line)
    if (!nzchar(trimmed)) next
    if (grepl("^\\*\\*Extended Data Table", trimmed)) {
      if (in_table) {
        out <- c(out, "\\bottomrule", "\\end{longtable}", "\\endgroup")
        in_table <- FALSE
      }
      out <- c(out, paste0("\\noindent\\textbf{", inline_tex(gsub("^\\*\\*|\\*\\*$", "", trimmed)), "}\\par\\medskip"))
      next
    }
    if (!grepl("^\\|", line)) next
    cells <- table_cells(line)
    if (all(grepl("^-+", cells))) next
    if (!in_table) {
      out <- c(out, "\\begingroup\\footnotesize\\setlength{\\tabcolsep}{2.5pt}", paste0("\\begin{longtable}{", table_spec(length(cells)), "}"), "\\toprule")
      in_table <- TRUE
      table_header <- TRUE
    }
    row <- paste(vapply(cells, inline_tex, character(1)), collapse = " & ")
    out <- c(out, paste0(row, " ", strrep("\\\\", 1)))
    if (table_header) {
      out <- c(out, "\\midrule", "\\endfirsthead", "\\toprule", row, " \\\\ ", "\\midrule", "\\endhead")
      table_header <- FALSE
    }
  }
  if (in_table) out <- c(out, "\\bottomrule", "\\end{longtable}", "\\endgroup")
  c(out, "\\linenumbers")
}

out <- c(out)
main_paths <- c(
  "figures/main/Figure_1_process_level_decomposition.pdf",
  "figures/main/Figure_2_architecture_scaffold.pdf",
  "figures/main/Figure_3_training_compatibility.pdf",
  "figures/main/Figure_4_realized_execution.pdf",
  "figures/main/Figure_5_bounded_local_transition_leverage.pdf"
)
for (i in seq_along(main_paths)) {
  out <- c(out, figure_block(paste0("Figure ", i), main_paths[i], main_legends[i], "0.58\\textheight"))
}

out <- c(out)
extended_paths <- c(
  "figures/extended_data/Extended_Data_Figure_1_metadata.pdf",
  "figures/extended_data/Extended_Data_Figure_2_coordinate_readback.pdf",
  "figures/extended_data/Extended_Data_Figure_3_window_boundary.pdf",
  "figures/extended_data/Extended_Data_Figure_4_path_memory_boundary.pdf",
  "figures/extended_data/Extended_Data_Figure_5_random_initialization.pdf",
  "figures/extended_data/Extended_Data_Figure_6_training_diagnostics.pdf",
  "figures/extended_data/Extended_Data_Figure_7_execution_diagnostics.pdf",
  "figures/extended_data/Extended_Data_Figure_8_direction_boundary.pdf",
  "figures/extended_data/Extended_Data_Figure_9_mixed_arithmetic_transfer.pdf"
)
for (i in seq_along(extended_paths)) {
  out <- c(out, figure_block(paste0("Extended Data Figure ", i), extended_paths[i], extended_data_legends[i], "0.48\\textheight"))
}
out <- c(out, extended_table_block(extended_table_lines))
out <- c(out, "\\end{document}")

writeLines(out, tex_file, useBytes = TRUE)
cat("NATURE_ARTICLE_SUBMISSION_TEX=PASS\n")
