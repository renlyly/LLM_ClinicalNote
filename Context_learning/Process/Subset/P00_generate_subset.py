import pandas as pd
import numpy as np

# Reproducible sampling
np.random.seed(100)

# Paths
input_note_file = "discharge.csv"
output_subject_ids = "/subset_data/sample_subject_ids_n.csv"  # 1% sampled subjects
output_notes_p1 = "/subset_data/discharge_notes_p1.csv"

# Step 1: Sample 1% of unique subject_ids
print("Reading subject IDs from source...")
df_ids = pd.read_csv(input_note_file, usecols=["subject_id"])

unique_ids = df_ids["subject_id"].unique()
sample_size = int(len(unique_ids) * 0.01)

sampled_ids = np.random.choice(unique_ids, size=sample_size, replace=False)

pd.DataFrame({"subject_id": sampled_ids}).to_csv(output_subject_ids, index=False)
print(f"Sampled 1% Subject IDs ({sample_size} subjects) saved to {output_subject_ids}")

# Step 2: Filter full notes by sampled subject_ids
print("Reading full discharge notes...")
all_notes = pd.read_csv(input_note_file)
print(f"Total notes loaded: {len(all_notes)}")

filtered_notes = all_notes[all_notes["subject_id"].isin(sampled_ids)]
filtered_notes.to_csv(output_notes_p1, index=False)

print(f"Filtered discharge notes (p1) saved to {output_notes_p1}")
print(f"Number of notes in output: {len(filtered_notes)}")
