from __future__ import annotations

from .enums import ModelName, Phenotype

SHOT_OPTIONS: tuple[int, ...] = (0, 3, 6)

NEGATION_TERMS: tuple[str, ...] = (
    "no",
    "not",
    "without",
    "denies",
    "negative for",
    "rule out",
    "ruled out",
)

PHENOTYPE_QUERY_TERMS: dict[Phenotype, tuple[str, ...]] = {
    Phenotype.METASTASIS: (
        "metastasis",
        "metastatic",
        "metastasized",
        "secondary tumor",
    ),
    Phenotype.INSULIN: (
        "insulin",
        "insulin-dependent",
        "iddm",
    ),
    Phenotype.HYPERTENSION: (
        "hypertension",
        "hypertensive",
        "htn",
        "high blood pressure",
    ),
}

DEFAULT_MODEL_PATHS: dict[ModelName, str] = {
    ModelName.GEMMA_7B: "google/gemma-7b-it",
    ModelName.LLAMA2_7B: "lianggq/llama-2-7b-chat-med",
    ModelName.LLAMA3_8B: "ContactDoctor/Bio-Medical-Llama-3-8B",
    ModelName.LLAMA3_70B: "meta-llama/Meta-Llama-3-70B-Instruct",
}

# Legacy alignment: original scripts used 3, 9, 18 total examples for 0/3/6-shot.
SHOT_TO_EXAMPLE_COUNT: dict[int, int] = {
    0: 3,
    3: 9,
    6: 18,
}
