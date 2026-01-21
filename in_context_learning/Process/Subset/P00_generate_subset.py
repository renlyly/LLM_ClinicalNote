import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(1)

# --- Configuration ---
# Input file containing raw discharge notes
input_note_file = 'discharge.csv'# Path to the discharge notes CSV file

# Intermediate file to save the sampled Subject IDs
output_subject_ids = '/subset_data/sample_subject_ids_n.csv' # sampled 1%

# Final output file for the filtered notes
output_notes_p1 = '/subset_data/discharge_notes_p1.csv'

# --- Step 1: Generate Random Sample of Subject IDs ---

# Read only the 'subject_id' column to save memory
print("Reading subject IDs from source...")
df = pd.read_csv(input_note_file, usecols=['subject_id'])

# Get unique subject_ids
unique_ids = df['subject_id'].unique()

# Calculate sample size (1% of unique subjects)
sample_size = int(len(unique_ids) * 0.01)

# Randomly select 1% of IDs
sampled_ids = np.random.choice(unique_ids, size=sample_size, replace=False)

# Save the sampled IDs to CSV
sampled_df = pd.DataFrame(sampled_ids, columns=['subject_id'])
sampled_df.to_csv(output_subject_ids, index=False)
print(f"Sampled 1% Subject IDs ({sample_size} subjects) saved to {output_subject_ids}")

# --- Step 2: Filter Notes Based on Sampled IDs ---

# Note: The original script re-reads the full file here. 
# If the file is very large, consider reading in chunks, but here we follow the original logic.
print("Reading full discharge notes...")
# Correcting the path variable to match the one used above (ensure this path is correct on your system)
# The original script had two different input paths. I am using the one defined at the top.
all_notes = pd.read_csv(input_note_file)

print(f"Total notes loaded: {len(all_notes)}")

# Filter notes where subject_id exists in our sampled list
filtered_notes = all_notes[all_notes['subject_id'].isin(sampled_ids)]

# Save the filtered notes to CSV
filtered_notes.to_csv(output_notes_p1, index=False)

print(f"Filtered discharge notes (p1) saved to {output_notes_p1}")
print(f"Number of notes in output: {len(filtered_notes)}")