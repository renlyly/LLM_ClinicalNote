from __future__ import annotations

from pathlib import Path

import pytest

from precllm.enums import InferenceBackend, ModelName, Phenotype, PreprocessMethod
from precllm.exceptions import ConfigValidationError
from precllm.models import RunConfig
from precllm.prompting import prompt_metadata


def test_prompt_metadata_matches_legacy_example_counts() -> None:
    assert prompt_metadata(Phenotype.METASTASIS, 0)["example_count"] == 3
    assert prompt_metadata(Phenotype.METASTASIS, 3)["example_count"] == 9
    assert prompt_metadata(Phenotype.METASTASIS, 6)["example_count"] == 18


def test_run_config_accepts_multi_id_columns() -> None:
    config = RunConfig(
        model=ModelName.GEMMA_7B,
        phenotype=Phenotype.INSULIN,
        preprocess=PreprocessMethod.RAG,
        shot=3,
        seed=100,
        rag_top_k=3,
        text_column="text",
        id_columns=("subject_id", "hadm_id"),
        inference_backend=InferenceBackend.RULE,
        model_path=None,
        max_new_tokens=8,
        temperature=0.1,
        do_sample=False,
    )
    assert config.id_columns == ("subject_id", "hadm_id")


def test_run_config_rejects_empty_id_columns() -> None:
    with pytest.raises(ConfigValidationError):
        RunConfig(
            model=ModelName.GEMMA_7B,
            phenotype=Phenotype.METASTASIS,
            preprocess=PreprocessMethod.REGEX,
            shot=3,
            seed=100,
            rag_top_k=3,
            text_column="text",
            id_columns=(),
            inference_backend=InferenceBackend.RULE,
            model_path=None,
            max_new_tokens=8,
            temperature=0.1,
            do_sample=False,
        )
