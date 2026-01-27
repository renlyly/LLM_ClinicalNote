# LLM-EHR: Clinical Phenotyping with Large Language Models

## overview

This framework evaluates large language models for clinical phenotyping using zero-shot and few-shot learning paradigms on the MIMIC-IV dataset, with comparisons against fine-tuned LLMs. Our preprocessing strategies significantly enhance the performance of smaller LLMs, making them suitable for privacy-sensitive and resource-constrained healthcare applications. This study provides practical guidance for deploying LLMs in local, secure, and efficient healthcare settings while addressing challenges in privacy, computational feasibility, and clinical applicability.



## Structure

```
LLM_Note/
├── fine_tune/                    # Fine-tuning pipeline
│   ├── 01_load_data.py           # Data loading and preprocessing
│   ├── 02_model_finetuning.py    # QLoRA fine-tuning implementation
│   └── 03_classification.py      # Model inference and classification
│
└── Context_learning/             # In-context learning experiments
    ├── Process/                  # Text preprocessing pipelines
    │   ├── Subset/
    │   │   ├── P00_generate_subset.py    # Dataset subset generation
    │   │   ├── P01_extract_notes.py      # Clinical note extraction
    │   │   ├── P02_RAGsentence.py        # RAG-based text extraction
    │   │   └── P02_regex.py              # Regex-based text extraction
    │   └── Metastasis_all/               # Full metastasis dataset processing
    │
    └── Predict/                  # Classification experiments by model
        ├── Gemma_7b/             # Gemma-7B experiments
        ├── Llema2_7b/            # Llama2-7B experiments
        ├── Llema3_8b/            # Llama3-8B experiments
        └── Llema3_70b/           # Llama3-70B experiments
```

## Methods

### Classification Tasks

Three clinical phenotyping tasks are evaluated:

- **Metastasis**: Detection of metastatic cancer presence from clinical notes
- **Hypertension**: Identification of hypertension diagnosis
- **Insulin**: Classification of long-term insulin usage status

Each task uses a three-class classification scheme:
- **(1) Yes**: Condition explicitly confirmed in clinical notes
- **(2) No**: Condition explicitly absent or discontinued
- **(3) Unknown**: Insufficient information to determine status

### Models

Four open-source large language models are evaluated:

| Model Name | Parameters | Description |
|------------|------------|-------------|
| **Gemma-7B** | 7B | Google's instruction-tuned model for general tasks |
| **LLaMA-2-7B-Chat-Med** | 7B | Meta's LLaMA-2 fine-tuned for medical dialogue |
| **Bio-Medical-LLaMA-3-8B** | 8B | LLaMA-3 specialized for biomedical text |
| **Meta-Llama-3-70B-Instruct** | 70B | Meta's largest instruction-following model |

### Experimental Design

**Text Preprocessing Strategies:**
1. **No Preprocessing** (`nonprocess`): Full discharge summaries without modification
2. **Regex-based** (`regex`): Sentence extraction using keyword pattern matching for target phenotypes
3. **RAG-based** (`rag`): Semantic similarity retrieval using sentence embeddings (FAISS + all-MiniLM-L6-v2 encoder)

**Few-shot Learning Settings:**
- **0-shot**: Task description with 3 generic examples
- **3-shot**: Task description with 9 examples (3 per class)
- **6-shot**: Task description with 18 examples (6 per class)

This yields **27 experimental conditions** per model (3 preprocessing × 3 few-shot × 3 tasks), totaling **108 experiments** across all models.


## Usage

### Complete Example: Gemma-7B for Metastasis Detection

This example demonstrates the full pipeline for classifying metastatic cancer using Gemma-7B with RAG-based preprocessing and 3-shot learning.

#### Step 1: Preprocess Clinical Notes (RAG-based Extraction)

Extract relevant sentences from discharge summaries using semantic similarity:

```bash
# Use RAG (Retrieval-Augmented Generation) to extract sentences related to metastasis
# This creates: /subset_data/filtered_2sen_discharge_notes_metastasis.csv
python Context_learning/Process/Subset/P02_RAGsentence.py metastasis
```

**What this does:**
- Loads discharge summaries from MIMIC-IV
- Uses FAISS and all-MiniLM-L6-v2 embeddings to find semantically similar sentences
- Extracts the top 2 most relevant sentences per note
- Saves preprocessed data with column `EXTRACTED_TEXT`

#### Step 2: Run Classification with In-Context Learning

Execute the Gemma-7B model with 3-shot learning:

```bash
# Run classification with random seed for reproducibility
# Output: /results/P1/A02_metastasis_11_T2_p1_[SEED].csv
python Context_learning/Predict/Gemma_7b/Metastasis/A_metastasis_rag_3shot.py 42
```

**What this does:**
- Downloads Gemma-7B-IT weights from KaggleHub
- Loads the RAG-preprocessed data (`EXTRACTED_TEXT` column)
- Applies 3-shot prompt with 9 clinical examples (3 per class)
- Classifies each note as: (1) Yes, (2) No, or (3) Unknown
- Saves results with `LLM_class` predictions
- Logs runtime to `/results/P2/`

**Script parameters:**
- `42`: Random seed for reproducible results
- Temperature: 0.1 (low temperature for consistent outputs)
- Max output length: 4 tokens (sufficient for classification codes)

#### Alternative Configurations

**Different preprocessing strategies:**
```bash
# No preprocessing (full discharge summary)
python Context_learning/Predict/Gemma_7b/Metastasis/A_metastasis_nonprocess_3shot.py 42

# Regex-based extraction (keyword matching)
python Context_learning/Predict/Gemma_7b/Metastasis/A_metastasis_regex_3shot.py 42
```

**Different few-shot settings:**
```bash
# 0-shot learning (minimal examples)
python Context_learning/Predict/Gemma_7b/Metastasis/A_metastasis_rag_0shot.py 42

# 6-shot learning (maximum examples)
python Context_learning/Predict/Gemma_7b/Metastasis/A_metastasis_rag_6shot.py 42
```

**Different models:**
```bash
# LLaMA-2-7B-Chat-Med
python Context_learning/Predict/Llema2_7b/Metastasis/A_metastasis_rag_3shot.py 42

# Bio-Medical-LLaMA-3-8B
python Context_learning/Predict/Llema3_8b/Metastasis/A_metastasis_rag_3shot.py 42

# Meta-Llama-3-70B-Instruct
python Context_learning/Predict/Llema3_70b/Metastasis/A_metastasis_rag_3shot.py 42
```

### Fine-Tuning Pipeline (QLoRA)

For fine-tuning experiments with parameter-efficient adaptation:

#### Step 1: Load and Prepare Training Data
```bash
# Load MIMIC-IV discharge summaries and create train/validation splits
# Balances classes and formats data for instruction tuning
python fine_tune/01_load_data.py
```


#### Step 2: Fine-Tune Model with QLoRA
```bash
# Fine-tune Gemma-7B using QLoRA (4-bit quantization + LoRA adapters)
# Training time: ~2-4 hours on a single A100 GPU
python fine_tune/02_model_finetuning.py
```


#### Step 3: Run Inference with Fine-Tuned Model
```bash
# Classify test set using fine-tuned model
# Output: classification results with predictions
python fine_tune/03_classification.py
```


### Generate Research Subset (Optional)

For creating balanced subsets for pilot studies:

```bash
# Generate stratified subset from full MIMIC-IV dataset
# Creates subset with balanced class distributions
python Context_learning/Process/Subset/P00_generate_subset.py
```

## Data

This project uses discharge summaries from the MIMIC-IV database.


