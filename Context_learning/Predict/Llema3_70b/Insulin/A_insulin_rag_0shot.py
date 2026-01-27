
#%%
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import torch
import re

# Load the dataset
extracted_notes = pd.read_csv('/subset_data/filtered_2sen_discharge_notes_insulin.csv')


hf_model_name = "Meta-Llama-3-70B-Instruct" # modifty to your model path
# Load Hugging Face tokenizer and model
auto_tokenizer = AutoTokenizer.from_pretrained(hf_model_name, use_fast=True)

quant_config = BitsAndBytesConfig(
    load_in_8bit=True,                
    llm_int8_threshold=6.0,           
    llm_int8_has_fp16_weight=False,   
    llm_int8_skip_modules=None        
)


auto_model = AutoModelForCausalLM.from_pretrained(
    hf_model_name,
    device_map="auto",
    quantization_config=quant_config)


import sys

RANDOM_SEED=sys.argv[1]
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

print(f"Model is loaded on: {auto_model.device}")

extracted_notes['LLM_class'] = 'Pending'

# Task description for model input
TASK_DESCRIPTION = """<start_of_turn> user
    Task: Classify the use of insulin from patient clinical notes. Respond only with the following numerical codes based on the notes provided:
    - (1) Yes: The notes explicitly confirm the patient uses insulin.
    - (2) No: The notes explicitly confirm the patient does not use insulin.
    - (3) Unknown: The notes do not contain sufficient information to determine the patient's insulin use status.


    Instructions:
    1. Do not provide explanations or reasons for your classification.
    2. Use only the information in the notes for your classification.
    3. If the notes are ambiguous or lack details regarding insulin, choose "(3) Unknown".
    4. Only provide a single numerical code as the response.

    Examples:
    - Example 1: "Patient requires insulin therapy to control blood glucose." --> (1)
    - Example 2: "After delivery her insulin was discontinued and her blood sugars were measured for 48 hours and were within goal range." --> (2)
    - Example 3: "She may need insulin if her renal function does not return to baseline." --> (3)
<end_of_turn>
"""
USER_CHAT_TEMPLATE = '<start_of_turn> Classify the following clinical notes to determine if the patient uses insulin:\n{notes}.<end_of_turn>\n'
MODEL_CHAT_TEMPLATE = '<start_of_turn> Answer: \n'


stage_pattern = re.compile(r'\((1|2|3)\)')

# Prepare output paths
csv_out_path = '/temp/tmpi.csv'
tmp_csv_out_path = csv_out_path

# Processing notes and generating classifications
for index, row in tqdm(extracted_notes.iterrows(), total=extracted_notes.shape[0], desc="Processing notes"):
    if row['LLM_class'] == 'Pending':
        clinical_notes = row['EXTRACTED_TEXT'].strip()
        prompt = (
            TASK_DESCRIPTION
            + USER_CHAT_TEMPLATE.format(notes=clinical_notes)
            + MODEL_CHAT_TEMPLATE
        )

        response = ''
        count = 0
        error_occurred = False
        matched_response = None

        while not matched_response and count < 10 and not error_occurred:
            torch.cuda.empty_cache()
            try:
                # Hugging Face Model Prediction
                inputs = auto_tokenizer(prompt, return_tensors="pt").to(auto_model.device)
                input_length = inputs["input_ids"].shape[1]
                hf_outputs = auto_model.generate(**inputs, max_new_tokens=4, pad_token_id=auto_tokenizer.eos_token_id, do_sample=True, 
                                                temperature=0.1)
                hf_response = auto_tokenizer.decode(hf_outputs[0, input_length:], skip_special_tokens=True)

                hf_match = stage_pattern.search(hf_response)
                if hf_match:
                    matched_response = hf_match.group(1)
                else:
                    matched_response = "3"

            except Exception as e:
                torch.cuda.empty_cache()
                print(f"Error during model generation: {e}")
                error_occurred = True
                break
            count += 1

        extracted_notes.at[index, 'LLM_class'] = matched_response if not error_occurred else '3'

        # Save progress periodically
        if (index + 1) % 3000 == 0:
            print(f"Saving progress at index {index}")
            if os.path.exists(tmp_csv_out_path):
                os.remove(tmp_csv_out_path)
            tmp_csv_out_path = csv_out_path.replace('.csv', f'_{index}_progress.csv')
            extracted_notes.iloc[:index].to_csv(tmp_csv_out_path, index=False)

extracted_notes.to_csv(f"/results/P1/A11_insulin_llama70_11_rag2_{RANDOM_SEED}.csv", index=False)

# Clean up all temporary files
import glob
# Get the base path without the extension
base_path = csv_out_path.replace('.csv', '')
# Find all temporary progress files
temp_files = glob.glob(f"{base_path}*_progress.csv")
# Add the main temporary file if it exists
if os.path.exists(tmp_csv_out_path):
    temp_files.append(tmp_csv_out_path)
# Remove all temporary files
for temp_file in temp_files:
    try:
        os.remove(temp_file)
        print(f"Removed temporary file: {temp_file}")
    except OSError as e:
        print(f"Error removing temporary file {temp_file}: {e}")
