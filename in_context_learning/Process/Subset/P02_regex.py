# %%
from tqdm import tqdm
import pandas as pd
import numpy as np
import spacy
import sys
import re
# %%
# Load the CSV file

df = pd.read_csv('subset_data/discharge_notes_p1.csv')

if len(sys.argv) > 1:
    keyword = sys.argv[1]
else:
    keyword = "insulin"


# %%

nlp = spacy.load("en_core_web_sm")

# Define regex patterns for medical conditions
pattern_htn = r"\b(?<!pre)hypertension\b|\bhypertensive\b|\bhtn\b|\bhigh\s+blood\s+pressure\b|\bbp\s*(?:elevated|high)\b"
pattern_insulin = r"\binsulin\b|\binsulin[- ]dependent\b|\bon\s+insulin\b|(?<!N)\bIDDM\b"
pattern_metastasis = r"\bmetastasis\b|\bmetastatic\b|\bmetastasize\b|\bmetastasized\b|\bmetastasizing\b|\bsecondary tumor\b"

def extract_sentences_regex(text, pattern):
    doc = nlp(text)
    sentences = list(doc.sents)
    extracted = []
    for i, sentence in enumerate(sentences):
        if re.search(pattern, sentence.text, re.IGNORECASE):
            prev_sent = sentences[i-1].text if i > 0 else ""
            next_sent = sentences[i+1].text if i < len(sentences) - 1 else ""
            extracted.append(f"{prev_sent} {sentence.text} {next_sent}".strip())
    return extracted

def extract_sentences(text, keywords):
    doc = nlp(text)
    sentences = list(doc.sents)
    extracted = []
    for i, sentence in enumerate(sentences):
        if any(keyword.lower() in sentence.text.lower() for keyword in keywords):
            prev_sent = sentences[i-1].text if i > 0 else ""
            next_sent = sentences[i+1].text if i < len(sentences) - 1 else ""
            extracted.append(f"{prev_sent} {sentence.text} {next_sent}".strip())
    return extracted

disease_patterns = {
    "hypertension": pattern_htn,
    "insulin": pattern_insulin,
    
}

disease_keywords = {
    "hypertension": ['hypertension', 'htn', "bp elevated"],
    "insulin": ['insulin', 'insulin-dependent'],
    "metastasis": [ 'metastasis','metastatic', 'metastasize','metastasized','metastasizing', 'secondary tumor']
}


# Use regex patterns if available, otherwise fall back to keywords
if keyword in disease_patterns:
    pattern = disease_patterns[keyword]
    # Filter notes using regex pattern
    filtered_notes = df[df['text'].str.contains(pattern, na=False, case=False, regex=True)]
    # Extract relevant sentences using regex
    filtered_notes['EXTRACTED_TEXT'] = [extract_sentences_regex(text, pattern) for text in tqdm(filtered_notes['text'])]
else:
    keywords = disease_keywords.get(keyword, [keyword])
    # Assuming df is your DataFrame
    filtered_notes = df[df['text'].str.contains('|'.join(keywords), na=False, case=False)]
    # Extract relevant sentences
    filtered_notes['EXTRACTED_TEXT'] = [extract_sentences(text, keywords) for text in tqdm(filtered_notes['text'])]
# Expand rows for notes with multiple extractions
extracted_notes = filtered_notes.explode('EXTRACTED_TEXT')
# %%

extracted_notes.to_csv(f'/subset_data/extracted_discharge_notes_{keyword}_p1.csv', index=False)
# %%
