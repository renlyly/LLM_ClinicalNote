from __future__ import annotations

import logging
from pathlib import Path

from .io_utils import read_note_records, write_manifest, write_predictions_csv, write_preprocessed_csv
from .models import PredictionRecord, RunConfig, RunSummary
from .predict import predict_records
from .preprocess import preprocess_records

LOGGER = logging.getLogger(__name__)


def _count_labels(predictions: list[PredictionRecord]) -> tuple[int, int, int]:
    yes_count = sum(1 for row in predictions if row.label.value == "yes")
    no_count = sum(1 for row in predictions if row.label.value == "no")
    unknown_count = sum(1 for row in predictions if row.label.value == "unknown")
    return yes_count, no_count, unknown_count


def run_pipeline(input_path: Path, output_dir: Path, config: RunConfig) -> RunSummary:
    """Execute preprocess and prediction stages and persist outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Reading input notes from %s", input_path)
    notes = read_note_records(
        input_path,
        id_columns=config.id_columns,
        text_column=config.text_column,
    )

    LOGGER.info("Running preprocessing using '%s'", config.preprocess.value)
    preprocessed = preprocess_records(
        records=notes,
        method=config.preprocess,
        phenotype=config.phenotype,
        rag_top_k=config.rag_top_k,
    )

    LOGGER.info(
        "Running prediction stage for phenotype '%s' with backend '%s'",
        config.phenotype.value,
        config.inference_backend.value,
    )
    predictions = predict_records(preprocessed, config=config)

    run_stem = (
        f"{config.model.value}_{config.phenotype.value}_{config.preprocess.value}_"
        f"{config.shot}shot_seed{config.seed}"
    )

    preprocessed_path = output_dir / f"{run_stem}.preprocessed.csv"
    predictions_path = output_dir / f"{run_stem}.predictions.csv"
    manifest_path = output_dir / f"{run_stem}.manifest.json"

    LOGGER.info("Writing outputs to %s", output_dir)
    write_preprocessed_csv(preprocessed_path, preprocessed, id_columns=config.id_columns)
    write_predictions_csv(predictions_path, predictions, id_columns=config.id_columns)
    write_manifest(manifest_path, config, input_path=input_path, output_dir=output_dir)

    yes_count, no_count, unknown_count = _count_labels(predictions)
    return RunSummary(
        input_path=input_path,
        output_dir=output_dir,
        total_notes=len(notes),
        yes_count=yes_count,
        no_count=no_count,
        unknown_count=unknown_count,
    )
