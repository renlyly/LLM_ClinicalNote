
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


hf_model_name = "lianggq/llama-2-7b-chat-med" # modifty to your model path


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
TASK_DESCRIPTION = """<start_of_turn>user
Task: Classify the use of insulin from patient clinical notes. Respond only with the following numerical codes based on the notes provided:
    - (1) Yes: The notes explicitly confirm the patient uses insulin.
    - (2) No: The notes explicitly confirm the patient does not use insulin.
    - (3) Unknown: The notes do not contain sufficient information to determine the patient's insulin use status.


Instructions:
1. Do not provide explanations or reasons for your classification.
2. Use only the information in the notes for your classification.
3. If the notes are ambiguous or lack details regarding insulin, choose "(3) Unknown".
4. Only provide a single numerical code as the response you must follow this rule.

Examples:
- Example 1: "___ was required to be on a sliding scale of insulin administration." --> (1)
- Example 2: "NovoFine 30 30 X ___ Needle Sig: One (1) needle Miscellaneous three times a day: for insulin injections." --> (1)
- Example 3: "Levemir FlexTouch U-100 Insuln (insulin detemir U-100) 100 unit/mL (3 mL) subcutaneous DAILY" --> (1)
- Example 4: "Glargine 6 Units Bedtime Insulin SC Sliding Scale using HUM Insulin" --> (1)
- Example 5: "_ at the ___ for IDDM on insulin pump, c/b diabetic retinopathy, neprhopathy and neuropathy" --> (1)
- Example 6: "Patient requires insulin therapy to control blood glucose." --> (1)
- Example 7: "His fingersticks and sliding scale insulin were subsequently stopped." --> (2)
- Example 8: "His insulin gtt was then discontinued and he was started on subq insulin." --> (2)
- Example 9: "After delivery her insulin was discontinued and her blood sugars were measured for 48 hours and were within goal range." --> (2)
- Example 10: "As per PMD, patient has poorly controlled diabetes and was switched off insulin prior to leaving for ___." --> (2)
- Example 11: "After long discussions with the patient, he was initially reluctant to start insulin, but it was determined to start him on lantus 24 units qHS given that it would be difficult to obtain a goal A1c without insulin." --> (2)
- Example 12: "Patient does not require insulin at this time." --> (2)
- Example 13: "Blood sugars were elevated during admission requiring intermittent insulin administration." --> (3)
- Example 14: "We also adjusted your insulin dosing because your sugar levels were very low in the beginning of your hospitalization." --> (3)
- Example 15: "They adjusted your insulin and treated your diabetic ketoacidosis." --> (3)
- Example 16: "We obtained a diabetic consult with ___ who adjusted your insulin prescription." --> (3)
- Example 17: "It is very important that you followup with the ___ ___ within the next few days to adjust your dose of insulin." --> (3)
- Example 18: "She may need insulin if her renal function does not return to baseline." --> (3)
<end_of_turn>
"""
USER_CHAT_TEMPLATE = '<start_of_turn> Classify the following clinical notes to determine if the patient uses insulin:\n{notes}.<end_of_turn>\n'
MODEL_CHAT_TEMPLATE = '<start_of_turn> Answer: \n'

# Regular expression to match valid responses
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

        # Update the 'LLM_class' field for the current row
        extracted_notes.at[index, 'LLM_class'] = matched_response if not error_occurred else '3'

        # Save progress periodically
        if (index + 1) % 3000 == 0:
            print(f"Saving progress at index {index}")
            if os.path.exists(tmp_csv_out_path):
                os.remove(tmp_csv_out_path)
            tmp_csv_out_path = csv_out_path.replace('.csv', f'_{index}_progress.csv')
            extracted_notes.iloc[:index].to_csv(tmp_csv_out_path, index=False)

# Save final results
extracted_notes.to_csv(f"/results/P1/A11_insulin_llama70_31_rag2_{RANDOM_SEED}.csv", index=False)

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
