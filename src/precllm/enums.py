from __future__ import annotations

from enum import Enum


class ModelName(str, Enum):
    """Supported model names exposed by the package API."""

    GEMMA_7B = "gemma-7b"
    LLAMA2_7B = "llama2-7b"
    LLAMA3_8B = "llama3-8b"
    LLAMA3_70B = "llama3-70b"


class Phenotype(str, Enum):
    """Supported clinical phenotypes."""

    METASTASIS = "metastasis"
    INSULIN = "insulin"
    HYPERTENSION = "hypertension"


class PreprocessMethod(str, Enum):
    """Supported preprocessing strategies."""

    REGEX = "regex"
    RAG = "rag"
    NONPROCESS = "nonprocess"


class PredictionLabel(str, Enum):
    """Standardized output labels for classification."""

    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


class InferenceBackend(str, Enum):
    """Inference backend options for prediction."""

    RULE = "rule"
    TRANSFORMERS = "transformers"
