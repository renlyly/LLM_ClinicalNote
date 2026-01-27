# %%
import pandas as pd
import spacy
from tqdm import tqdm

# Paths
input_file = "/home/shilin/temp/LLM_edit/results/discharge_notes_p1.csv"
output_file = "/home/shilin/temp/LLM_edit/results/extracted_discharge_notes_p1.csv"

print(f"Loading data from {input_file}...")
df = pd.read_csv(input_file)

# Load spaCy model (run: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model not found. Run: python -m spacy download en_core_web_sm")
    raise


def extract_sentences(text: str, keywords):
    """Return sentence windows (prev/current/next) where current contains any keyword."""
    if not isinstance(text, str):
        return []

    doc = nlp(text)
    sents = list(doc.sents)

    extracted = []
    for i, sent in enumerate(sents):
        if any(k.lower() in sent.text.lower() for k in keywords):
            prev_sent = sents[i - 1].text if i > 0 else ""
            next_sent = sents[i + 1].text if i < len(sents) - 1 else ""
            extracted.append(f"{prev_sent} {sent.text} {next_sent}".strip())

    return extracted


# Metastasis keywords
keywords = [
    "metastasis",
    "metastatic",
    "metastasize",
    "metastasized",
    "metastasizing",
    "secondary tumor",
]

print("Filtering notes containing keywords...")
filtered_notes = df[df["text"].str.contains("|".join(keywords), na=False, case=False)].copy()
print(f"Found {len(filtered_notes)} notes with keywords. Starting extraction...")

filtered_notes["EXTRACTED_TEXT"] = [
    extract_sentences(text, keywords)
    for text in tqdm(filtered_notes["text"], desc="Extracting")
]

extracted_notes = filtered_notes.explode("EXTRACTED_TEXT")
extracted_notes = extracted_notes.dropna(subset=["EXTRACTED_TEXT"])

print(f"Saving extracted notes to {output_file}...")
extracted_notes.to_csv(output_file, index=False)
print("Done.")
# %%
