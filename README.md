# LLM-EHR: Clinical Phenotyping with Large Language Models

## Overview

This repository contains code for evaluating Large Language Models (LLMs) on clinical phenotype classification tasks using Electronic Health Records (EHR) data from MIMIC-IV. We compare **fine-tuning** and **in-context learning** approaches across multiple LLMs and preprocessing strategies.

## Repository Structure

```
LLM_EHR-main/
├── fine_tune/                    # Fine-tuning pipeline
│   ├── 01_load_data.py           # Data loading and preprocessing
│   ├── 02_model_finetuning.py    # QLoRA fine-tuning with Gemma
│   └── 03_classification.py      # Model inference
│
└── in_context_learning/          # In-context learning experiments
    ├── Process/                  # Text preprocessing
    │   ├── Subset/
    │   │   ├── P00_generate_subset.py    # Subset sampling
    │   │   ├── P01_extract_notes.py      # Note extraction
    │   │   ├── P02_RAGsentence.py        # RAG-based extraction
    │   │   └── P02_regex.py              # Regex-based extraction
    │   └── Metastasis_all/
    │
    └── Predict/                  # Classification experiments
        ├── Gemma_7b/             # Gemma-7B experiments
        ├── Llema2_7b/            # Llama2-7B experiments
        ├── Llema3_8b/            # Llama3-8B experiments
        └── Llema3_70b/           # Llama3-70B experiments
```

## Methods

### Classification Tasks
- **Metastasis**: Presence of metastatic cancer
- **Hypertension**: Diagnosis of hypertension
- **Insulin**: Insulin usage status

### Models
| Model | Parameters |
|-------|------------|
| Gemma | 7B |
| Llama-2 | 7B |
| Llama-3 | 8B |
| Llama-3 | 70B |

### Text Preprocessing Strategies
1. **No Preprocessing** (`nonprocess`): Full clinical notes
2. **Regex-based** (`regex`): Keyword pattern matching extraction
3. **RAG-based** (`rag`): Semantic similarity retrieval using sentence embeddings (FAISS + all-MiniLM-L6-v2)

### Experimental Conditions
- **Few-shot settings**: 0-shot, 3-shot, 6-shot
- **Fine-tuning**: QLoRA with 4-bit quantization

## Requirements

```
torch
transformers
peft
bitsandbytes
sentence-transformers
faiss-cpu
pandas
numpy
```

## Usage

### Data Preprocessing
```bash
# Generate subset
python in_context_learning/Process/Subset/P00_generate_subset.py

# RAG-based extraction
python in_context_learning/Process/Subset/P02_RAGsentence.py metastasis

# Regex-based extraction
python in_context_learning/Process/Subset/P02_regex.py
```

### In-Context Learning
```bash
# Example: Gemma-7B, Metastasis, RAG preprocessing, 3-shot
python in_context_learning/Predict/Gemma_7b/Metastasis/A_metastasis_rag_3shot.py [SEED]
```

### Fine-Tuning
```bash
# Load and preprocess data
python fine_tune/01_load_data.py

# Fine-tune model with QLoRA
python fine_tune/02_model_finetuning.py

# Run classification
python fine_tune/03_classification.py
```

## Data

This project uses discharge summaries from the MIMIC-IV database. Access requires PhysioNet credentialing.

## License

This project is for research purposes only.
