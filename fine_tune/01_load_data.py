# %%
import pandas as pd
import re
from datasets import Dataset

# %%
# Load data
# df: DataFrame from MIMIC-IV
# meta_note_id: list of note IDs with metastasis
# no_meta_note_id: list of note IDs without metastasis
def load_data(df, meta_note_id, no_meta_note_id):
    df['meta_status'] = df['note_id'].apply(lambda x: True if x in meta_note_id else (False if x in no_meta_note_id else None))
    df_filtered = df[df['note_id'].isin(meta_note_id + no_meta_note_id)]
    return df_filtered, meta_note_id, no_meta_note_id

# %%
# Create a dataset with text and label
def create_dataset(df_filtered, meta_note_id, no_meta_note_id):
    metastasis_terms = r"\b(metastasis|metastatic|metastasize|metastases|metastasized|dissemination|distant spread|metachronous|hematogenous spread|lymphatic spread|micrometastases|infiltration|tumor spread|extra-nodal extension)\b"
    data = []

    for _, row in df_filtered.iterrows():
        paragraphs = re.split(r"\n\s*\n", row["text"].strip())
        matching_paragraphs = [p for p in paragraphs if re.search(metastasis_terms, p, re.IGNORECASE)]
        
        # Extract matching paragraphs
        if matching_paragraphs:
            tailored_text = "\n\n".join(matching_paragraphs).strip()
        else:
            tailored_text = "No relevant paragraph mentioning metastasis was found."

        # Set label
        if row["note_id"] in meta_note_id:
            label = 1  # Metastasis present
        elif row["note_id"] in no_meta_note_id:
            label = 0  # No metastasis

        # Add to data list
        data.append({"text": tailored_text, "label": label})

    # Create a Dataset object
    dataset = Dataset.from_pandas(pd.DataFrame(data))
    return dataset

# %%
# Load and process data
df_filtered, meta_note_id, no_meta_note_id = load_data(df, meta_note_id, no_meta_note_id)

# Create the dataset
dataset = create_dataset(df_filtered, meta_note_id, no_meta_note_id)

# %%
# Split the dataset into train and test sets
dataset_final = dataset.train_test_split(test_size=0.2)

# Print dataset information
print(dataset_final)

# %%
# Save the dataset
dataset_final.save_to_disk("dataset_final")
# %%
