from __future__ import annotations

import re
from typing import Iterable

from .constants import PHENOTYPE_QUERY_TERMS
from .enums import Phenotype, PreprocessMethod
from .models import NoteRecord, PreprocessedRecord

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?:\n{2,}|(?<=[.!?])\s+)")
_WORD_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def _split_sentences(text: str) -> list[str]:
    sentences = [part.strip() for part in _SENTENCE_SPLIT_PATTERN.split(text) if part.strip()]
    return sentences


def _window_join(sentences: list[str], index: int, window: int = 1) -> str:
    start = max(index - window, 0)
    end = min(index + window + 1, len(sentences))
    return " ".join(sentences[start:end]).strip()


def _keyword_regex(terms: tuple[str, ...]) -> re.Pattern[str]:
    escaped = [re.escape(term) for term in terms]
    return re.compile(r"(" + "|".join(escaped) + r")", re.IGNORECASE)


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _WORD_PATTERN.findall(text)}


def _regex_extract(text: str, terms: tuple[str, ...]) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return ""

    pattern = _keyword_regex(terms)
    extracted: list[str] = []
    for idx, sentence in enumerate(sentences):
        if pattern.search(sentence):
            extracted.append(_window_join(sentences, idx, window=1))
    return " ".join(extracted).strip()


def _rag_extract(text: str, terms: tuple[str, ...], top_k: int) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return ""

    query_tokens = _tokenize(" ".join(terms))
    scored: list[tuple[int, str]] = []
    for sentence in sentences:
        sentence_tokens = _tokenize(sentence)
        score = len(query_tokens.intersection(sentence_tokens))
        if score > 0:
            scored.append((score, sentence))

    if not scored:
        return ""

    scored.sort(key=lambda item: item[0], reverse=True)
    top_sentences = [sentence for _, sentence in scored[:top_k]]
    return " ".join(top_sentences).strip()


def preprocess_records(
    records: Iterable[NoteRecord],
    method: PreprocessMethod,
    phenotype: Phenotype,
    rag_top_k: int,
) -> list[PreprocessedRecord]:
    """Run preprocessing over note records using the selected strategy."""

    terms = PHENOTYPE_QUERY_TERMS[phenotype]
    outputs: list[PreprocessedRecord] = []

    for row in records:
        if method == PreprocessMethod.NONPROCESS:
            extracted_text = row.text.strip()
        elif method == PreprocessMethod.REGEX:
            extracted_text = _regex_extract(row.text, terms)
        else:
            extracted_text = _rag_extract(row.text, terms, top_k=rag_top_k)

        outputs.append(
            PreprocessedRecord(
                record_id=row.record_id,
                source_ids=row.source_ids,
                original_text=row.text,
                extracted_text=extracted_text,
            )
        )

    return outputs
