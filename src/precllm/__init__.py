"""PrecLLM package.

This package provides a production-grade, configurable workflow for paper-level
clinical note phenotyping experiments.
"""

from .enums import InferenceBackend, ModelName, Phenotype, PreprocessMethod
from .models import RunConfig, RunSummary
from .pipeline import run_pipeline
from .prompting import build_prompt, prompt_metadata

__all__ = [
    "InferenceBackend",
    "ModelName",
    "Phenotype",
    "PreprocessMethod",
    "RunConfig",
    "RunSummary",
    "build_prompt",
    "prompt_metadata",
    "run_pipeline",
]
