#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(stringr)
  library(tidyr)
  library(RColorBrewer)
  library(scales)
})

set.seed(100)

args <- commandArgs(trailingOnly = TRUE)
script_path <- tryCatch(normalizePath(sys.frame(1)$ofile), error = function(e) NA)
script_dir <- if (!is.na(script_path)) dirname(script_path) else getwd()
base_dir <- if (length(args) >= 1) args[1] else normalizePath(file.path(script_dir, ".."))

csv_dir <- file.path(base_dir, "output")
output_dir <- csv_dir
if (!dir.exists(output_dir)) dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

shot_levels <- c("Zero-shot", "Three-shot", "Six-shot")
proc_levels <- c("Regex", "RAG", "No Preprocessing")
model_levels <- c("Gemma", "LLaMA-2", "LLaMA-3")

proc_colors <- brewer.pal(3, "Set2")
names(proc_colors) <- proc_levels

csv_file <- file.path(csv_dir, "consolidated_summary_subject_system_sensitivity.csv")
if (!file.exists(csv_file)) stop(sprintf("CSV not found: %s", csv_file))

df_raw <- read_csv(csv_file, show_col_types = FALSE)

names(df_raw) <- gsub("\\.", "_", tolower(names(df_raw)))
need <- c("model", "setting", "p_c")
if (!all(need %in% names(df_raw))) {
  stop(sprintf("Required columns not found. Present: %s", paste(names(df_raw), collapse = ", ")))
}

map_model <- c(gemma = "Gemma", llama2 = "LLaMA-2", llama3 = "LLaMA-3")

df <- df_raw %>%
  mutate(
    model_label = recode(model, !!!map_model, .default = model),
    shot_setting = case_when(
      str_detect(setting, regex("^zero-shot", ignore_case = TRUE)) ~ "Zero-shot",
      str_detect(setting, regex("^three-shot", ignore_case = TRUE)) ~ "Three-shot",
      str_detect(setting, regex("^six-shot", ignore_case = TRUE)) ~ "Six-shot",
      TRUE ~ NA_character_
    ),
    preprocessing_type = case_when(
      str_detect(setting, regex("regex", ignore_case = TRUE)) ~ "Regex",
      str_detect(setting, regex("rag", ignore_case = TRUE)) ~ "RAG",
      str_detect(setting, regex("nonpro", ignore_case = TRUE)) ~ "No Preprocessing",
      TRUE ~ NA_character_
    ),
    model_label = factor(model_label, levels = model_levels),
    shot_setting = factor(shot_setting, levels = shot_levels),
    preprocessing_type = factor(preprocessing_type, levels = proc_levels),
    p_c = as.numeric(p_c)
  ) %>%
  group_by(model_label, shot_setting, preprocessing_type) %>%
  summarise(p_c = mean(p_c, na.rm = TRUE), .groups = "drop")


df_complete <- expand_grid(
  model_label = factor(model_levels, levels = model_levels),
  shot_setting = factor(shot_levels, levels = shot_levels),
  preprocessing_type = factor(proc_levels, levels = proc_levels)
) %>%
  left_join(df, by = c("model_label", "shot_setting", "preprocessing_type")) %>%
  mutate(p_c = ifelse(is.na(p_c), 0, p_c))

p <- ggplot(df_complete, aes(x = preprocessing_type, y = p_c, fill = preprocessing_type)) +
  geom_col(color = "black", alpha = 0.8, width = 0.75) +
  geom_text(aes(label = sprintf("%.2f", p_c)), vjust = -0.35, size = 3.8, fontface = "bold") +
  facet_grid(model_label ~ shot_setting) +
  scale_fill_manual(values = proc_colors, name = "Preprocessing Method") +
  scale_y_continuous(limits = c(0, 1.05), labels = percent_format(accuracy = 1)) +
  labs(
    x = "Preprocessing Method",
    y = "Sensitivity"
  ) +
  theme_bw(base_size = 14) +
  theme(
    plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5, margin = margin(b = 10)),
    legend.position = "bottom",
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(face = "bold", size = 12),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_blank(),
    strip.background = element_blank(),
    panel.border = element_blank(),
    axis.line = element_line(color = "black")
  )

output_jpg <- file.path(output_dir, "sys_sensitive_3model_private_subject.jpg")
output_pdf <- file.path(output_dir, "sys_sensitive_3model_private_subject.pdf")

ggsave(output_jpg, p, width = 12, height = 9, dpi = 300)
ggsave(output_pdf, p, width = 12, height = 9)
cat("Saved:", output_jpg, "\n")
cat("Saved:", output_pdf, "\n")
