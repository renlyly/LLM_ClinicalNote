# %%
import pandas as pd
import numpy as np
from tqdm import tqdm  
import time 

VARIANT = '7b-it'  # version of model
if '1.1-' in VARIANT:
    VARIANT2 = VARIANT.replace('1.1-', '')  
else:
    VARIANT2 = VARIANT  
MACHINE_TYPE = 'cuda:1'  
import os
import kagglehub # hub to download model

weights_dir = kagglehub.model_download(f'google/gemma/pyTorch/{VARIANT}')

# Ensure that the tokenizer is present
tokenizer_path = os.path.join(weights_dir, 'tokenizer.model')
assert os.path.isfile(tokenizer_path), 'Tokenizer not found!'

# Ensure that the checkpoint is present
ckpt_path = os.path.join(weights_dir, f'gemma-{VARIANT2}.ckpt')
assert os.path.isfile(ckpt_path), 'PyTorch checkpoint not found!'

import sys
# sys.path.insert(0, '/home/shilin/temp/LLM_edit/code/')   # gemma_pytorch

# %%
from gemma_pytorch.gemma.config import get_config_for_7b, get_config_for_2b
from gemma_pytorch.gemma.model import GemmaForCausalLM
import torch

# Set up model config.
model_config = get_config_for_2b() if "2b" in VARIANT else get_config_for_7b()
model_config.tokenizer = tokenizer_path
model_config.quant = 'quant' in VARIANT2
# print(model_config)
# Instantiate the model and load the weights.
torch.set_default_dtype(model_config.get_dtype())
device = torch.device(MACHINE_TYPE)
model = GemmaForCausalLM(model_config)
model.load_weights(ckpt_path)
model = model.to(device).eval()


RANDOM_SEED=sys.argv[1]
# Set model randomness settings.
def set_random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")
# Example: Set the seed for reproducibility.
set_random_seed(RANDOM_SEED)

# %%
TASK_DESCRIPTION = """<start_of_turn>user
Task: Classify the use of insulin from patient clinical notes. Respond only with the following numerical codes based on the notes provided:
- (1) Yes: The notes explicitly confirm the patient uses insulin.
- (2) No: The notes explicitly confirm the patient does not use insulin.
- (3) Unknown: The notes do not contain sufficient information to determine the patient's insulin use status.


Instructions:
1. Do not provide explanations or reasons for your classification.
2. Use only the information in the notes for your classification.
3. If the notes are ambiguous or lack details regarding insulin use, choose "(3) Unknown".
4. Only provide a single numerical code as the response.

Examples:
- Example 1: "___ was required to be on a sliding scale of insulin administration." --> (1)
- Example 2: "Glargine 6 Units Bedtime Insulin SC Sliding Scale using HUM Insulin" --> (1)
- Example 3: "Patient requires insulin therapy to control blood glucose." --> (1)
- Example 4: "His fingersticks and sliding scale insulin were subsequently stopped." --> (2)
- Example 5: "His insulin gtt was then discontinued and he was started on subq insulin." --> (2)
- Example 6: "Patient does not require insulin at this time." --> (2)
- Example 7: "We also adjusted your insulin dosing because your sugar levels were very low in the beginning of your hospitalization." --> (3)
- Example 8: "They adjusted your insulin and treated your diabetic ketoacidosis." --> (3)
- Example 9: "She may need insulin if her renal function does not return to baseline." --> (3)

"""

USER_CHAT_TEMPLATE = 'Analyze the following clinical notes to determine if the patient Long-term (current) use of insulin:\n{notes}.<end_of_turn>\n'

    # Template for model response based on the analysis of the clinical notes
MODEL_CHAT_TEMPLATE = '<start_of_turn> Response:\n'


import re

# Regular expression to match only valid stage responses
stage_pattern = re.compile(r'\((1|2|3)\)')

tasks = [
 
    (
        '/subset_data/extracted_discharge_notes_insulin_p1.csv',
        '/results/P1/A01_insulin_21_preprocessed_Regex_shot_p1'
    )
]


TEXT_COLUMN_NAME = 'EXTRACTED_TEXT' # Or 'text', depending on your dataset

script_start_time = time.time()


for input_csv_path, output_base_path in tasks:
    final_output_csv_path = f"{output_base_path}_{RANDOM_SEED}.csv"
    tmp_progress_csv_path = f"{output_base_path}_progress_temp3_{RANDOM_SEED}.csv"

    current_df = None # To store the DataFrame currently being processed


    # Resume logic: Check if a temporary progress file exists; if so, load data from it to continue processing
    if os.path.exists(tmp_progress_csv_path):
        print(f"Resuming from temporary progress file: {tmp_progress_csv_path}")
        current_df = pd.read_csv(tmp_progress_csv_path)
        if 'LLM_class' not in current_df.columns: # Ensure 'LLM_class' column exists
            current_df['LLM_class'] = 'Pending'
        current_df['LLM_class'] = current_df['LLM_class'].fillna('Pending').astype(str) # Handle NaNs and ensure string type
    else:
        # If no temporary file, start fresh processing from the original input CSV
        print(f"Starting fresh processing for: {input_csv_path}")
        if not os.path.exists(input_csv_path): # Check if input file exists
            print(f"Error: Input file {input_csv_path} not found. Skipping task.")
            continue
        current_df = pd.read_csv(input_csv_path)
        current_df.index = pd.RangeIndex(len(current_df.index)) # Reset index (optional, but good for consistency)
        current_df['LLM_class'] = 'Pending' # Initialize the LLM classification result column

    if TEXT_COLUMN_NAME not in current_df.columns:
        print(f"Error: Text column '{TEXT_COLUMN_NAME}' not found in {input_csv_path}. Skipping task.")
        print(f"Available columns: {current_df.columns.tolist()}")
        continue # Move to the next task
    
    print(f"Processing DataFrame for {os.path.basename(input_csv_path)} with {len(current_df)} rows.")

    # Core processing logic (LLM calls, etc.) -- this part is similar to single-file processing,
    # but now operates on 'current_df'
    for index, row in tqdm(current_df.iterrows(), total=current_df.shape[0], desc=f"Processing {os.path.basename(input_csv_path)}"):
        llm_class_val = str(row['LLM_class'])
        # Only process rows with status 'Pending' or 'nan' (which might come from a previous incomplete load)
        if llm_class_val == 'Pending' or llm_class_val == 'nan': 
            
            if pd.isna(row[TEXT_COLUMN_NAME]): # Handle cases where text content is missing
                current_df.at[index, 'LLM_class'] = '3' 
                continue
            
            clinical_notes = str(row[TEXT_COLUMN_NAME]).strip()
            
            prompt = (
                TASK_DESCRIPTION
                + USER_CHAT_TEMPLATE.format(notes=clinical_notes)
                + MODEL_CHAT_TEMPLATE
            )

            response_text = '' 
            generation_attempt_count = 0 
            error_occurred = False
            matched_classification = None 

            while not matched_classification and generation_attempt_count < 10 and not error_occurred:
                torch.cuda.empty_cache() # Clear CUDA cache before generation
                try:
                    response_text = model.generate(
                        prompt,
                        device=device, temperature=0.1,
                        output_len=4 # For (1), (2), (3)
                    )
                    #print(response_text)

                    match = stage_pattern.search(response_text)
                    if match:
                        matched_classification = match.group(1)
                    else:
                        matched_classification = "3"    
                except Exception as e:
                    torch.cuda.empty_cache() # Clear cache on error
                    print(f"Error during model generation for index {index}: {e}")
                    error_occurred = True # This will break the inner while loop
                generation_attempt_count += 1

            if error_occurred or not matched_classification:
                final_classification = '3' # Default to Unknown if error or no match
            else:
                final_classification = matched_classification

            current_df.at[index, 'LLM_class'] = final_classification # Update DataFrame

            if (index + 1) % 1000 == 0: # Save every 200 records
                print(f"Saving temporary progress for {os.path.basename(input_csv_path)} at index {index}")
                current_df.to_csv(tmp_progress_csv_path, index=False)

    print(f"Saving final results for {os.path.basename(input_csv_path)} to {final_output_csv_path}")
    current_df.to_csv(final_output_csv_path, index=False)
    print(f"Finished processing {os.path.basename(input_csv_path)}. Results saved to {final_output_csv_path}")

    # Clean up the temporary progress file for this task
    if os.path.exists(tmp_progress_csv_path):
        try:
            os.remove(tmp_progress_csv_path)
            print(f"Removed temporary progress file: {tmp_progress_csv_path}")
        except OSError as e:
            print(f"Error removing temporary file {tmp_progress_csv_path}: {e}")


print("All tasks completed.")

script_end_time = time.time()
total_runtime_seconds = script_end_time - script_start_time

# Convert to a human-readable format (e.g., HH:MM:SS)
hours, rem = divmod(total_runtime_seconds, 3600)
minutes, seconds = divmod(rem, 60)
runtime_formatted = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

# Define the output directory and filename for runtime log
runtime_log_dir = "/results/P2"
os.makedirs(runtime_log_dir, exist_ok=True) # Ensure directory exists

# Get the script name dynamically
script_name = os.path.basename(__file__)
runtime_log_filename = f"{script_name}_runtime_log_{RANDOM_SEED}.txt"
runtime_log_path = os.path.join(runtime_log_dir, runtime_log_filename)

with open(runtime_log_path, 'w') as f:
    f.write(f"Script Name: {script_name}\n")
    f.write(f"Random Seed (Args[1]): {RANDOM_SEED}\n")
    f.write(f"Start Time: {time.ctime(script_start_time)}\n")
    f.write(f"End Time: {time.ctime(script_end_time)}\n")
    f.write(f"Total Runtime: {runtime_formatted}\n")
    f.write(f"Total Runtime (seconds): {total_runtime_seconds:.2f}\n")
    f.write(f"Tasks Processed: {len(tasks)}\n") # Number of tasks processed

print(f"\nTotal script runtime recorded to: {runtime_log_path}")
print(f"Total script runtime: {runtime_formatted}")
