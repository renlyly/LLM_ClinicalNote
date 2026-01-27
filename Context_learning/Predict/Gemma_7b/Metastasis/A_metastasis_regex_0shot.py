# %%
import pandas as pd
import numpy as np
# import spacy
from tqdm import tqdm  # 用于显示进度条的库
# %%head -n 5 /results/extracted_discharge_notes_p1.csv

extracted_notes = pd.read_csv('/results/extracted_discharge_notes_p1.csv')


# extracted_notes = pd.read_csv('/media/wu_lab/Data/mimic-iv/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/extracted_discharge_notes_all.csv')
# big data,  Sample fix 1% data and standard the output form we need to
# understand the structure for input
# note_id,subject_id,hadm_id,note_type,note_seq,charttime,storetime,text,EXTRACTED_TEXT
# sample subject id
# output_path = '/results/'
# %%
#
# %%
# !git clone https://github.com/google/gemma_pytorch.git


# %%

VARIANT = '7b-it'  # version of model
if '1.1-' in VARIANT:
    VARIANT2 = VARIANT.replace('1.1-', '')  
else:
    VARIANT2 = VARIANT  
# VARIANT = '7b-it'  # version of model
# it seems the version is updated
MACHINE_TYPE = 'cuda:0'  # 0 1 两个gpu
import os
import kagglehub # hub to download model

# %%
# Load model weights
weights_dir = kagglehub.model_download(f'google/gemma/pyTorch/{VARIANT}')


# Ensure that the tokenizer is present
tokenizer_path = os.path.join(weights_dir, 'tokenizer.model')
assert os.path.isfile(tokenizer_path), 'Tokenizer not found!'

# Ensure that the checkpoint is present
ckpt_path = os.path.join(weights_dir, f'gemma-{VARIANT2}.ckpt')
assert os.path.isfile(ckpt_path), 'PyTorch checkpoint not found!'

import sys
sys.path.append('/home/shilin/temp/LLM_edit/code/gemma_pytorch/')
sys.path.append('/temp/LLM_edit/code/gemma_pytorch/')


# sys.path.append("/home/shilin/kaggle/working/gemma_pytorch/") 
# from gemma.config import GemmaConfig, get_config_for_7b, get_config_for_2b
# %%
from gemma_pytorch.gemma.config import get_config_for_7b, get_config_for_2b
from gemma_pytorch.gemma.model import GemmaForCausalLM

# %% [markdown]
# ## Setup the model
# %%
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
extracted_notes['LLM_class'] = 'Pending'



# %%
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

# Template for user input, including clinical notes to be analyzed
USER_CHAT_TEMPLATE = 'Analyze the following clinical notes to determine if the patient has metastasis status:\n{notes}<end_of_turn>\n'

# Template for model response based on the analysis of the clinical notes
MODEL_CHAT_TEMPLATE = '<start_of_turn>model\n'


import re

# Regular expression to match only valid stage responses
stage_pattern = re.compile(r'\((1|2|3)\)')

csv_out_path = '/results/tmp.csv'
tmp_csv_out_path = csv_out_path
# %%
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
                torch.cuda.empty_cache()
                response = model.generate(
                    prompt,
                    device=device,   temperature=0.1 ,# top_k=40, 
                    output_len=3
                )
                match = stage_pattern.search(response)
                if match:
                    matched_response = match.group(1)  # Get only the number without parentheses
            except Exception as e:
                torch.cuda.empty_cache()
                print(f"Error during model generation: {e}")
                error_occurred = True
                break  # Break out of the loop if an error occurs
            count += 1

        if error_occurred or not matched_response:
            response_new = '3'
        else:
            response_new = matched_response

        # Update the 'LLM_class' field for the current row
        extracted_notes.at[index, 'LLM_class'] = response_new

        # Every 300 iterations, save the DataFrame
        if (index + 1) % 3000 == 0:
            print(f"Saving progress at index {index}")
            if os.path.exists(tmp_csv_out_path):
                os.remove(tmp_csv_out_path)
            tmp_csv_out_path = csv_out_path.replace('.csv', f'_{index}_011.csv')
            extracted_notes.iloc[:index].to_csv(tmp_csv_out_path, index=False)
            
# %%
extracted_notes.to_csv(f"/results/p2/p1_results_021_{RANDOM_SEED}.csv", index=False)


# %%
