#!/bin/bash

set -euo pipefail

INPUT_DIR=${1:-"/home/shilin/temp/LLM_edit/HNC_SUBSET_all/Results/P1_LLM"}
OUTPUT_DIR=${2:-"/home/shilin/temp/LLM_edit/HNC_SUBSET_all/Results/P2_sum_subject"}
PYTHON_BIN=${3:-"/home/shilin/miniconda3/envs/try1/bin/python"}
SCRIPT_PATH=${4:-"/home/shilin/temp/LLM_edit/HNC_SUBSET_all/CODE/P001_create_sum_bytask_subject_merged.py"}
LOG_DIR=${5:-"/home/shilin/temp/LLM_edit/HNC_SUBSET_all/log"}

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/P001_subject_summary_processing_$TIMESTAMP.log"

log_message() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_message "Starting SUBJECT LEVEL summary processing"

shopt -s nullglob
files=("$INPUT_DIR"/*.csv)
if [[ ${#files[@]} -eq 0 ]]; then
  log_message "No CSV files found in $INPUT_DIR"
  exit 1
fi

count=0
for file_path in "${files[@]}"; do
  file_name=$(basename "$file_path")
  count=$((count + 1))
  log_message "Processing file $count/${#files[@]}: $file_name"
  "$PYTHON_BIN" "$SCRIPT_PATH" \
    --input_data_path "$INPUT_DIR/" \
    --input_data_file "$file_name" \
    --output_prefix "$OUTPUT_DIR/" 2>&1 | tee -a "$LOG_FILE"
  log_message "Completed: $file_name"
done

log_message "Completed SUBJECT LEVEL summary processing"
