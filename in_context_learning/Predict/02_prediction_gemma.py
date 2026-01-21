# %%
# Model download and verification (This repository updates often, so the model may be different from the one used in the notebook, we include the model folder in our repository so that you don't have to download it again)
# !git clone https://github.com/google/gemma_pytorch.git

# %%
import pandas as pd
import torch
from tqdm import tqdm
import re
import os
import kagglehub

import sys
sys.path.append('gemma_pytorch')
from gemma_pytorch.gemma.config import get_config_for_7b, get_config_for_2b
from gemma_pytorch.gemma.model import GemmaForCausalLM
# %%
# Load extracted clinical notes
extracted_notes = pd.read_csv('extracted_discharge_notes_all.csv')
# Add a 'Pending' classification status for metastasis
extracted_notes['LLM_class'] = 'Pending'

# %%
VARIANT = '7b-it'
MACHINE_TYPE = 'cuda'

weights_dir = kagglehub.model_download(f'google/gemma/pyTorch/{VARIANT}')

# Verify that the tokenizer and checkpoint exist
tokenizer_path = os.path.join(weights_dir, 'tokenizer.model')
ckpt_path = os.path.join(weights_dir, f'gemma-{VARIANT}.ckpt')

assert os.path.isfile(tokenizer_path), 'Tokenizer not found!'
assert os.path.isfile(ckpt_path), 'PyTorch checkpoint not found!'

# Set up model configuration and load weights
model_config = get_config_for_2b() if "2b" in VARIANT else get_config_for_7b()
model_config.tokenizer = tokenizer_path
model_config.quant = 'quant' in VARIANT

torch.set_default_dtype(model_config.get_dtype())
device = torch.device(MACHINE_TYPE)

# Instantiate the model
model = GemmaForCausalLM(model_config)
model.load_weights(ckpt_path)
model = model.to(device).eval()

# %%
# Task description and templates
# For few-shot learning, please replace the simple examples with the real examples
TASK_DESCRIPTION = """<start_of_turn>user
Task: Classify the presence of metastasis from patient clinical notes. Respond only with the following numerical codes based on the notes provided:
- (1) Yes: The notes explicitly confirm the patient has metastasis.
- (2) No: The notes explicitly confirm the patient does not have metastasis.
- (3) Unknown: The notes do not contain sufficient information to determine the patient's metastasis status.

Instructions:
1. Do not provide explanations or reasons for your classification.
2. Use only the information in the notes for your classification.
3. If the notes are ambiguous or lack details regarding metastasis, choose "(3) Unknown".
4. Only provide a single numerical code as the response.

Examples:
- Example 1: "The CT scan shows multiple nodules in the liver consistent with metastasis." --> (1)
- Example 2: "Patient has a history of cancer but no evidence of metastatic disease on recent imaging." --> (2)
- Example 3: "Patient treated for localized breast absent of metastasis." --> (3)

"""

USER_CHAT_TEMPLATE = 'Analyze the following clinical notes to determine if the patient has metastasis:\n{notes}\n'
MODEL_CHAT_TEMPLATE = '<start_of_turn>model\n'

# Regular expression to match valid response (1, 2, or 3)
stage_pattern = re.compile(r'\((1|2|3)\)')

# Path to save intermediate results
csv_out_path = 'tmp_results.csv'

# %%
# Process the clinical notes and classify metastasis status
for index, row in tqdm(extracted_notes.iterrows(), total=extracted_notes.shape[0], desc="Processing notes"):
    if row['LLM_class'] == 'Pending':
        clinical_notes = row['EXTRACTED_TEXT'].strip()
        prompt = TASK_DESCRIPTION + USER_CHAT_TEMPLATE.format(notes=clinical_notes) + MODEL_CHAT_TEMPLATE

        matched_response = None
        error_occurred = False
        attempts = 0

        # Try generating the model's response (up to 10 attempts)
        while not matched_response and attempts < 10 and not error_occurred:
            try:
                torch.cuda.empty_cache()
                response = model.generate(prompt, device=device, output_len=3)
                match = stage_pattern.search(response)
                if match:
                    matched_response = match.group(1)
            except Exception as e:
                print(f"Error during model generation: {e}")
                error_occurred = True
                break
            attempts += 1

        response_new = matched_response if matched_response else '3'  # Default to 'Unknown' if no valid response

        # Update the 'LLM_class' field for the current row
        extracted_notes.at[index, 'LLM_class'] = response_new

        # Save intermediate results every 3000 iterations
        if (index + 1) % 3000 == 0:
            print(f"Saving progress at index {index}")
            extracted_notes.iloc[:index].to_csv(csv_out_path.replace('.csv', f'_{index}.csv'), index=False)

# %%
# Save final results
extracted_notes.to_csv('all_results_classification.csv', index=False)
