from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .constants import SHOT_OPTIONS
from .enums import InferenceBackend, ModelName, Phenotype, PredictionLabel, PreprocessMethod
from .exceptions import ConfigValidationError


@dataclass(frozen=True)
class RunConfig:
    """Validated runtime configuration for one experiment run."""

    model: ModelName
    phenotype: Phenotype
    preprocess: PreprocessMethod
    shot: int
    seed: int
    rag_top_k: int
    text_column: str
    id_columns: tuple[str, ...]
    inference_backend: InferenceBackend
    model_path: str | None
    max_new_tokens: int
    temperature: float
    do_sample: bool
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.shot not in SHOT_OPTIONS:
            raise ConfigValidationError(
                f"Invalid shot value '{self.shot}'. Allowed values are: {SHOT_OPTIONS}."
            )
        if self.seed < 0:
            raise ConfigValidationError("Seed must be a non-negative integer.")
        if self.rag_top_k <= 0:
            raise ConfigValidationError("RAG top-k must be greater than zero.")
        if self.max_new_tokens <= 0:
            raise ConfigValidationError("max_new_tokens must be greater than zero.")
        if self.temperature < 0:
            raise ConfigValidationError("temperature must be non-negative.")
        if not self.text_column.strip():
            raise ConfigValidationError("The text column name cannot be empty.")
        if not self.id_columns:
            raise ConfigValidationError("At least one id column must be provided.")
        if any(not col.strip() for col in self.id_columns):
            raise ConfigValidationError("Id column names cannot be empty.")


@dataclass(frozen=True)
class NoteRecord:
    """Input note record with source identifier metadata."""

    record_id: str
    text: str
    source_ids: dict[str, str]


@dataclass(frozen=True)
class PreprocessedRecord:
    """Output of preprocessing stage."""

    record_id: str
    source_ids: dict[str, str]
    original_text: str
    extracted_text: str


@dataclass(frozen=True)
class PredictionRecord:
    """Output of prediction stage."""

    record_id: str
    source_ids: dict[str, str]
    label: PredictionLabel
    evidence: str
    raw_response: str


@dataclass(frozen=True)
class RunSummary:
    """Structured execution summary for reporting and audit."""

    input_path: Path
    output_dir: Path
    total_notes: int
    yes_count: int
    no_count: int
    unknown_count: int
