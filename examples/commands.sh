#!/usr/bin/env bash
set -euo pipefail

python -m precllm catalog

python -m precllm prompt \
  --phenotype metastasis \
  --shot 3 \
  --note-text "Patient has metastatic lesions in liver and lung."

python -m precllm run \
  --input-csv examples/sample_input.csv \
  --output-dir outputs \
  --model llama2-7b \
  --phenotype metastasis \
  --preprocess rag \
  --shot 3 \
  --id-columns subject_id,hadm_id \
  --text-column text \
  --inference-backend rule
