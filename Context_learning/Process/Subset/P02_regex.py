from tqdm import tqdm
import pandas as pd
import spacy
import sys
import re


# -----------------------------
# Load input data
# -----------------------------
df = pd.read_csv("subset_data/discharge_notes_p1.csv")


# -----------------------------
# Read keyword from CLI argument
# -----------------------------
# Usage: python script.py insulin
if len(sys.argv) > 1:
    keyword = sys.argv[1]
else:
    keyword = "insulin"


# -----------------------------
# Load spaCy model for sentence segmentation
# -----------------------------
nlp = spacy.load("en_core_web_sm")


# -----------------------------
# Define regex patterns for conditions
# -----------------------------
pattern_htn = (
    r"\b(?<!pre)hypertension\b"
    r"|\bhypertensive\b"
    r"|\bhtn\b"
    r"|\bhigh\s+blood\s+pressure\b"
    r"|\bbp\s*(?:elevated|high)\b"
)

pattern_insulin = (
    r"\binsulin\b"
    r"|\binsulin[- ]dependent\b"
    r"|\bon\s+insulin\b"
    r"|(?<!N)\bIDDM\b"
)


def extract_sentences_regex(text: str, pattern: str):
    """
    Extract a window of sentences (previous, current, next) whenever the current
    sentence matches the given regex pattern (case-insensitive).
    """
    doc = nlp(text)
    sentences = list(doc.sents)

    extracted = []
    for i, sentence in enumerate(sentences):
        if re.search(pattern, sentence.text, re.IGNORECASE):
            prev_sent = sentences[i - 1].text if i > 0 else ""
            next_sent = sentences[i + 1].text if i < len(sentences) - 1 else ""
            extracted.append(f"{prev_sent} {sentence.text} {next_sent}".strip())

    return extracted


def extract_sentences(text: str, keywords):
    """
    Extract a window of sentences (previous, current, next) whenever the current
    sentence contains any of the provided keywords (case-insensitive substring match).
    """
    doc = nlp(text)
    sentences = list(doc.sents)

    extracted = []
    for i, sentence in enumerate(sentences):
        if any(k.lower() in sentence.text.lower() for k in keywords):
            prev_sent = sentences[i - 1].text if i > 0 else ""
            next_sent = sentences[i + 1].text if i < len(sentences) - 1 else ""
            extracted.append(f"{prev_sent} {sentence.text} {next_sent}".strip())

    return extracted


# -----------------------------
# Condition lookup tables
# -----------------------------
# If a disease has a regex pattern here, we will use regex-based filtering/extraction.
disease_patterns = {
    "hypertension": pattern_htn,
    "insulin": pattern_insulin,
}

# Otherwise, we fall back to keyword-based filtering/extraction.
disease_keywords = {
    "hypertension": ["hypertension", "htn", "bp elevated"],
    "insulin": ["insulin", "insulin-dependent"],
    "metastasis": [
        "metastasis",
        "metastatic",
        "metastasize",
        "metastasized",
        "metastasizing",
        "secondary tumor",
    ],
}


# -----------------------------
# Filter notes and extract sentence windows
# -----------------------------
if keyword in disease_patterns:
    pattern = disease_patterns[keyword]

    # Filter notes using the regex pattern (case-insensitive).
    filtered_notes = df[df["text"].str.contains(pattern, na=False, case=False, regex=True)]

    # Extract sentence windows using the same regex pattern.
    filtered_notes["EXTRACTED_TEXT"] = [
        extract_sentences_regex(text, pattern) for text in tqdm(filtered_notes["text"])
    ]
else:
    keywords = disease_keywords.get(keyword, [keyword])

    # Filter notes using keyword OR matching.
    filtered_notes = df[df["text"].str.contains("|".join(keywords), na=False, case=False)]

    # Extract sentence windows using keyword matching.
    filtered_notes["EXTRACTED_TEXT"] = [
        extract_sentences(text, keywords) for text in tqdm(filtered_notes["text"])
    ]


# -----------------------------
# Expand rows (one row per extracted window)
# -----------------------------
extracted_notes = filtered_notes.explode("EXTRACTED_TEXT")


# -----------------------------
# Save results
# -----------------------------
output_path = f"subset_data/extracted_discharge_notes_{keyword}_p1.csv"
extracted_notes.to_csv(output_path, index=False)
