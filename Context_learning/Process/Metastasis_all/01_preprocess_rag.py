# %%
import pandas as pd
import spacy
from tqdm import tqdm

# %%
# Load the CSV file
df = pd.read_csv('discharge.csv')

# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")

# Function to extract relevant sentences based on keywords
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

# Define keywords to search for
keywords = [
    'metastasis',
    'metastatic',
    'metastasize',
    'metastases',
    'metastasized',
    'dissemination',
    'distant spread',
    'metachronous',
    'hematogenous spread',
    'lymphatic spread',
    'micrometastases',
    'infiltration',
    'tumor spread',
    'extra-nodal extension'
]


# Filter notes that contain any of the keywords
filtered_notes = df[df['text'].str.contains('|'.join(keywords), na=False, case=False)]

# Extract relevant sentences from the filtered notes
filtered_notes['EXTRACTED_TEXT'] = [extract_sentences(text, keywords) for text in tqdm(filtered_notes['text'])]

# Expand rows for notes with multiple extracted sentences
extracted_notes = filtered_notes.explode('EXTRACTED_TEXT')

# Save the extracted notes to a new CSV file
extracted_notes.to_csv('extracted_discharge_notes_all.csv', index=False)

# %%
