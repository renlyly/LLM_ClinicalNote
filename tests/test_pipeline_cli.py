from __future__ import annotations

import csv
from pathlib import Path

from precllm.cli import main
from precllm.enums import InferenceBackend, ModelName, Phenotype, PreprocessMethod
from precllm.models import RunConfig
from precllm.pipeline import run_pipeline


def _write_input(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject_id", "hadm_id", "text"])
        writer.writeheader()
        writer.writerow(
            {
                "subject_id": "s1",
                "hadm_id": "h1",
                "text": "There is metastatic disease in liver.",
            }
        )
        writer.writerow(
            {
                "subject_id": "s2",
                "hadm_id": "h2",
                "text": "No evidence of metastasis in this visit.",
            }
        )


def test_pipeline_generates_outputs_with_multi_ids(tmp_path: Path) -> None:
    input_csv = tmp_path / "input.csv"
    output_dir = tmp_path / "output"
    _write_input(input_csv)

    config = RunConfig(
        model=ModelName.LLAMA2_7B,
        phenotype=Phenotype.METASTASIS,
        preprocess=PreprocessMethod.REGEX,
        shot=3,
        seed=123,
        rag_top_k=3,
        text_column="text",
        id_columns=("subject_id", "hadm_id"),
        inference_backend=InferenceBackend.RULE,
        model_path=None,
        max_new_tokens=8,
        temperature=0.1,
        do_sample=False,
    )

    summary = run_pipeline(input_csv, output_dir, config)
    assert summary.total_notes == 2
    predictions = output_dir / "llama2-7b_metastasis_regex_3shot_seed123.predictions.csv"
    assert predictions.exists()
    content = predictions.read_text(encoding="utf-8")
    assert "subject_id" in content
    assert "hadm_id" in content


def test_cli_prompt_and_dry_run_commands() -> None:
    prompt_code = main(["prompt", "--phenotype", "metastasis", "--shot", "3"])
    assert prompt_code == 0


def test_cli_dry_run_with_multi_ids(tmp_path: Path) -> None:
    input_csv = tmp_path / "input.csv"
    _write_input(input_csv)

    exit_code = main(
        [
            "run",
            "--input-csv",
            str(input_csv),
            "--output-dir",
            str(tmp_path / "out"),
            "--model",
            "gemma-7b",
            "--phenotype",
            "metastasis",
            "--preprocess",
            "rag",
            "--shot",
            "3",
            "--id-columns",
            "subject_id,hadm_id",
            "--inference-backend",
            "rule",
            "--dry-run",
        ]
    )
    assert exit_code == 0
