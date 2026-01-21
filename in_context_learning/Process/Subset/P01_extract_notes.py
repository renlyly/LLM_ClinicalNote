# %%
import pandas as pd
import spacy
from tqdm import tqdm

# --- Configuration ---
input_file = '/home/shilin/temp/LLM_edit/results/discharge_notes_p1.csv'
output_file = '/home/shilin/temp/LLM_edit/results/extracted_discharge_notes_p1.csv'

# Load the CSV file
print(f"Loading data from {input_file}...")
df = pd.read_csv(input_file)

# Load Spacy model
# Ensure you have run: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model not found. Please run: python -m spacy download en_core_web_sm")
    exit()

def extract_sentences(text, keywords):
    """
    Extracts sentences containing keywords, including the preceding and succeeding sentences for context.
    """
    # Spacy increases memory usage; handling NaN/empty text checks is good practice
    if not isinstance(text, str):
        return []
        
    doc = nlp(text)
    sentences = list(doc.sents)
    extracted = []
    
    for i, sentence in enumerate(sentences):
        # Check if any keyword exists in the current sentence
        if any(keyword.lower() in sentence.text.lower() for keyword in keywords):
            prev_sent = sentences[i-1].text if i > 0 else ""
            next_sent = sentences[i+1].text if i < len(sentences) - 1 else ""
            # Combine previous, current, and next sentences
            extracted.append(f"{prev_sent} {sentence.text} {next_sent}".strip())
            
    return extracted

# Define keywords related to metastasis
keywords = [
    'metastasis',
    'metastatic',
    'metastasize',
    'metastasized',
    'metastasizing',
    'secondary tumor'
]

print("Filtering notes containing keywords...")
# Pre-filter: Only process rows that definitely contain the keywords (speeds up Spacy processing)
filtered_notes = df[df['text'].str.contains('|'.join(keywords), na=False, case=False)].copy()

print(f"Found {len(filtered_notes)} notes with keywords. Starting extraction...")

# Extract relevant sentences with context
# Using tqdm to show progress bar
tqdm.pandas()
filtered_notes['EXTRACTED_TEXT'] = [extract_sentences(text, keywords) for text in tqdm(filtered_notes['text'])]

# Expand rows for notes with multiple extracted segments
extracted_notes = filtered_notes.explode('EXTRACTED_TEXT')

# Drop rows where extraction might have failed or returned empty (optional cleanup)
extracted_notes = extracted_notes.dropna(subset=['EXTRACTED_TEXT'])

# Save to CSV
print(f"Saving extracted notes to {output_file}...")
extracted_notes.to_csv(output_file, index=False)
print("Done.")
# %%