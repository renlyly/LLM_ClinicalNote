from __future__ import annotations

import re
from typing import Iterable

from .constants import NEGATION_TERMS, PHENOTYPE_QUERY_TERMS, SHOT_TO_EXAMPLE_COUNT
from .enums import InferenceBackend, Phenotype, PredictionLabel
from .llm_backends import TransformersBackend, TransformersGenerationConfig
from .models import PredictionRecord, PreprocessedRecord, RunConfig
from .prompting import build_prompt


def _count_mentions(text: str, terms: tuple[str, ...]) -> tuple[int, int]:
    lowered = text.lower()
    positive_mentions = 0
    negated_mentions = 0

    for term in terms:
        pattern = re.compile(re.escape(term.lower()))
        for match in pattern.finditer(lowered):
            start = match.start()
            context_start = max(0, start - 40)
            context = lowered[context_start:start]
            if any(negation in context for negation in NEGATION_TERMS):
                negated_mentions += 1
            else:
                positive_mentions += 1

    return positive_mentions, negated_mentions


def classify_text_rule(text: str, phenotype: Phenotype) -> tuple[PredictionLabel, str]:
    """Rule baseline used when model inference is disabled or unavailable."""

    cleaned = text.strip()
    if not cleaned:
        return PredictionLabel.UNKNOWN, "No extracted evidence."

    terms = PHENOTYPE_QUERY_TERMS[phenotype]
    positive_mentions, negated_mentions = _count_mentions(cleaned, terms)

    if positive_mentions > 0 and negated_mentions == 0:
        return PredictionLabel.YES, f"Positive mentions: {positive_mentions}."
    if positive_mentions == 0 and negated_mentions > 0:
        return PredictionLabel.NO, f"Negated mentions: {negated_mentions}."
    if positive_mentions > 0 and negated_mentions > 0:
        return PredictionLabel.UNKNOWN, "Conflicting positive and negated mentions."

    return PredictionLabel.UNKNOWN, "No phenotype-specific keywords found."


def _parse_model_response(raw_response: str) -> PredictionLabel:
    """Parse free-form LLM output into standardized labels."""

    lowered = raw_response.lower()

    numeric_match = re.search(r"\((1|2|3)\)", lowered)
    if not numeric_match:
        numeric_match = re.search(r"\b(1|2|3)\b", lowered)

    if numeric_match:
        mapping = {
            "1": PredictionLabel.YES,
            "2": PredictionLabel.NO,
            "3": PredictionLabel.UNKNOWN,
        }
        return mapping[numeric_match.group(1)]

    if "yes" in lowered:
        return PredictionLabel.YES
    if "unknown" in lowered or "uncertain" in lowered:
        return PredictionLabel.UNKNOWN
    if re.search(r"\bno\b", lowered):
        return PredictionLabel.NO

    return PredictionLabel.UNKNOWN


def _predict_with_rule(
    records: Iterable[PreprocessedRecord],
    config: RunConfig,
) -> list[PredictionRecord]:
    outputs: list[PredictionRecord] = []
    for row in records:
        label, evidence = classify_text_rule(row.extracted_text, config.phenotype)
        outputs.append(
            PredictionRecord(
                record_id=row.record_id,
                source_ids=row.source_ids,
                label=label,
                evidence=evidence,
                raw_response=label.value,
            )
        )
    return outputs


def _predict_with_transformers(
    records: Iterable[PreprocessedRecord],
    config: RunConfig,
) -> list[PredictionRecord]:
    generation = TransformersGenerationConfig(
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        do_sample=config.do_sample,
    )
    backend = TransformersBackend(
        model=config.model,
        model_path=config.model_path,
        generation=generation,
    )

    outputs: list[PredictionRecord] = []
    for row in records:
        if not row.extracted_text.strip():
            outputs.append(
                PredictionRecord(
                    record_id=row.record_id,
                    source_ids=row.source_ids,
                    label=PredictionLabel.UNKNOWN,
                    evidence="No extracted evidence.",
                    raw_response="",
                )
            )
            continue

        prompt = build_prompt(config.phenotype, config.shot, row.extracted_text)
        raw_response = backend.generate(prompt)
        label = _parse_model_response(raw_response)
        evidence = (
            f"Model response parsed with {SHOT_TO_EXAMPLE_COUNT[config.shot]} prompt examples "
            f"for shot={config.shot}."
        )
        outputs.append(
            PredictionRecord(
                record_id=row.record_id,
                source_ids=row.source_ids,
                label=label,
                evidence=evidence,
                raw_response=raw_response,
            )
        )

    return outputs


def predict_records(
    records: Iterable[PreprocessedRecord],
    config: RunConfig,
) -> list[PredictionRecord]:
    """Run classification for each preprocessed note."""

    if config.inference_backend == InferenceBackend.RULE:
        return _predict_with_rule(records, config)

    return _predict_with_transformers(records, config)
