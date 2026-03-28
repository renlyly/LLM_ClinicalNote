\# PrecLLM

**GitHub Repository:** [https://github.com/renlyly/LLM_ClinicalNote](https://github.com/renlyly/LLM_ClinicalNote)   

**Preprint (arXiv):** [https://arxiv.org/abs/2412.02868](https://arxiv.org/abs/2412.02868)

PrecLLM is a dedicated package built to support paper-oriented clinical note phenotyping workflows. It is engineered to seamlessly preserve the original project's core logic, encompassing data preprocessing strategies (`regex`, `rag`, or `nonprocess`), prompt-controlled inference (`zero`, `three`, or `six-shot`), and model-specific execution

## Architecture & Workflow

<p align="center">
  <img src="assets/precllm-workflow-diagram.png" alt="PrecLLM workflow diagram" width="960" />
</p>
The pipeline ingests note-level CSV data and applies your chosen preprocessing strategy (`regex`, `rag`, or `nonprocess`). It then constructs phenotype-specific prompts using zero, three, or six-shot configurations. These prompts are routed to either a deterministic `rule` baseline or a `transformers` model for inference. The final output consists of a preprocessed CSV file.

## Prompt Generation System

Located within `src/precllm/prompting.py`, this module governs deterministic prompt generation using the `build_prompt` function. It includes phenotype-specific task templates for metastasis, insulin, and hypertension. To maintain alignment with legacy scripts, the module relies on a shot-aware selection logic that scales example counts based on the requested parameter: 0 shots load 3 examples, 3 shots load 9, and 6 shots load 18.

Inspect a prompt directly:

```bash
python -m precllm prompt \
  --phenotype metastasis \
  --shot 6 \
  --note-text "Patient with known metastatic disease in liver."
```

## Model Backend & Inference

PrecLLM orchestrates predictions through `src/precllm/predict.py`, pulling from backend logic defined in `src/precllm/llm_backends.py`. The system supports two primary inference modes:

- **`rule`**: A deterministic baseline that requires no external LLM dependencies.
- **`transformers`**: Hugging Face model inference utilizing `AutoTokenizer` and `AutoModelForCausalLM`.

When using the `transformers` backend, ensure the LLM runtime dependencies are installed:

```bash
python -m pip install -e '.[llm]'
```

Default model registry:
- `gemma-7b` -> `google/gemma-7b-it`
- `llama2-7b` -> `lianggq/llama-2-7b-chat-med`
- `llama3-8b` -> `ContactDoctor/Bio-Medical-Llama-3-8B`
- `llama3-70b` -> `meta-llama/Meta-Llama-3-70B-Instruct`

Override model path when needed:
- `--model-path <hf_repo_or_local_path>`

## Data Contract and Schema Flexibility

The package requires a minimal data schema: at least one text column designated for clinical notes (`--text-column`) and one or more identifier columns (`--id-column` or `--id-columns`). All provided ID columns are validated prior to execution and preserved in the final outputs. Furthermore, the system automatically generates a stable composite key, `record_id`, from these provided IDs.

Single-ID example:

```bash
python -m precllm run \
  --input-csv your_data.csv \
  --output-dir outputs \
  --model llama2-7b \
  --phenotype metastasis \
  --preprocess rag \
  --shot 3 \
  --id-column note_id \
  --text-column text \
  --inference-backend rule
```

Multi-ID example (`subject_id` + `hadm_id`):

```bash
python -m precllm run \
  --input-csv your_data.csv \
  --output-dir outputs \
  --model llama2-7b \
  --phenotype metastasis \
  --preprocess regex \
  --shot 3 \
  --id-columns subject_id,hadm_id \
  --text-column clinical_note \
  --inference-backend transformers
```



## CLI Reference

You can list all supported options and system defaults by running:

```bash
python -m precllm catalog
```

To verify your configuration without triggering a full inference run, use the `--dry-run` flag to resolve the config and prompt metadata:

```bash
python -m precllm run \
  --input-csv examples/sample_input.csv \
  --output-dir outputs \
  --model gemma-7b \
  --phenotype insulin \
  --preprocess rag \
  --shot 0 \
  --id-columns subject_id,hadm_id \
  --text-column text \
  --inference-backend rule \
  --dry-run
```

## Output Specifications

Every execution generates three primary files formatted with a descriptive run stem (`{model}_{phenotype}_{preprocess}_{shot}shot_seed{seed}`):

- **`<run_stem>.preprocessed.csv`**: The processed input data.
- **`<run_stem>.predictions.csv`**: Contains the original ID columns, the generated `record_id`, and key prediction fields including `label`, `evidence`, and `raw_response`.
- **`<run_stem>.manifest.json`**: Metadata tracking the run details.

## Package Layout

```text
LLM_Note/
- assets/
- examples/
- src/precllm/
- tests/
- pyproject.toml
- README.md
```


## License and Compliance

This project is licensed under the MIT License (`LICENSE`).

**Please note:** PrecLLM is an engineering tool designed strictly for research workflows, not a medical device. Outputs must not be used as standalone clinical decisions. Users are fully responsible for maintaining HIPAA and privacy compliance, data governance, and secure model access controls within their own environments.
