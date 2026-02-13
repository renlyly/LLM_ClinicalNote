from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .exceptions import DataValidationError
from .models import NoteRecord, PredictionRecord, PreprocessedRecord, RunConfig


def _build_record_id(source_ids: dict[str, str], row_index: int) -> str:
    values = [value for value in source_ids.values() if value]
    if values:
        return "|".join(values)
    return f"row-{row_index}"


def read_note_records(input_path: Path, id_columns: tuple[str, ...], text_column: str) -> list[NoteRecord]:
    """Read note records from CSV and validate mandatory columns."""

    if not input_path.exists():
        raise DataValidationError(f"Input file does not exist: {input_path}")

    with input_path.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        header = reader.fieldnames or []

        for id_column in id_columns:
            if id_column not in header:
                raise DataValidationError(
                    f"Missing required id column '{id_column}'. Available columns: {header}"
                )

        if text_column not in header:
            raise DataValidationError(
                f"Missing required text column '{text_column}'. Available columns: {header}"
            )

        records: list[NoteRecord] = []
        for index, row in enumerate(reader, start=1):
            source_ids = {col: (row.get(col) or "").strip() for col in id_columns}
            record_id = _build_record_id(source_ids, index)
            text = (row.get(text_column) or "").strip()
            records.append(NoteRecord(record_id=record_id, text=text, source_ids=source_ids))

    if not records:
        raise DataValidationError(f"Input file has no rows: {input_path}")

    return records


def write_preprocessed_csv(
    path: Path,
    rows: Iterable[PreprocessedRecord],
    id_columns: tuple[str, ...],
) -> None:
    """Write preprocessing output to CSV."""

    fieldnames = [*id_columns, "record_id", "original_text", "extracted_text"]
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {column: row.source_ids.get(column, "") for column in id_columns}
            payload.update(
                {
                    "record_id": row.record_id,
                    "original_text": row.original_text,
                    "extracted_text": row.extracted_text,
                }
            )
            writer.writerow(payload)


def write_predictions_csv(
    path: Path,
    rows: Iterable[PredictionRecord],
    id_columns: tuple[str, ...],
) -> None:
    """Write prediction output to CSV."""

    fieldnames = [*id_columns, "record_id", "label", "evidence", "raw_response"]
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {column: row.source_ids.get(column, "") for column in id_columns}
            payload.update(
                {
                    "record_id": row.record_id,
                    "label": row.label.value,
                    "evidence": row.evidence,
                    "raw_response": row.raw_response,
                }
            )
            writer.writerow(payload)


def write_manifest(path: Path, config: RunConfig, input_path: Path, output_dir: Path) -> None:
    """Persist a run manifest for traceability and audit."""

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "config": {
            key: value.value if hasattr(value, "value") else value
            for key, value in asdict(config).items()
        },
    }
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)
