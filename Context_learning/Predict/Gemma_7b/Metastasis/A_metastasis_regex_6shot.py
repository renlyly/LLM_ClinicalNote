# %%
import pandas as pd
import numpy as np
# import spacy
from tqdm import tqdm
# %%
extracted_notes = pd.read_csv('/results/extracted_discharge_notes_p1.csv')

# %%
# !git clone https://github.com/google/gemma_pytorch.git


# %%
VARIANT = '7b-it'  # version of model
if '1.1-' in VARIANT:
    VARIANT2 = VARIANT.replace('1.1-', '')  
else:
    VARIANT2 = VARIANT  
# VARIANT = '7b-it'  # version of model
MACHINE_TYPE = 'cuda:0'
import os
import kagglehub
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
- Example 1: "Innumerable hepatic and pulmonary metastases.  No obvious primary malignancy is identified on this study." --> (1)
- Example 2: "New diagnosis of metastatic cancer." --> (1)
- Example 3: "Ms. ___ is a ___ with metastatic cancer of unknown primary (known lesions in lungs and liver) presenting with shortness of breath." --> (1)
- Example 4: "No evidence of metastatic disease in the brain." --> (2)
- Example 5: "No definitive metastatic disease in the chest." --> (2)
- Example 6: "No metastatic disease." --> (2)
- Example 7: "___ year old woman who underwent colonoscopy with polypectomy two days prior to presentation." --> (3)
- Example 8: "She was asymptomatic in the ED and no gross rectal bleeding was noted. Guaiac positive with brown/black stools." --> (3)
- Example 9: "He has no history of head injury or loss of consciousness." --> (3)
- Example 10: "___ is a ___ year old woman with metastatic pancreatic
cancer on palliative gemcitabine who is admitted with one week 
of progressive epigastric abdominal pain." --> (1)
- Example 11: "Metastatic pancreatic cancer: Initially thought locally advanced and went to OR for Whipple after neoadjuvant chemotherapy, unfortunately found to have metastatic liver disease intraoperatively." --> (1)
- Example 12: "There is a metastatic lesion involving the T5 vertebral body on the right." --> (1)
- Example 13: "Staging CT of the chest was negative for metastases." --> (2)
- Example 14: "There was no evidence of distant metastases." --> (2)
- Example 15: "A 10-point review was otherwise not suggestive of metastatic disease." --> (2)
- Example 16: "The left atrium is mildly dilated. No atrial septal 
defect is seen by 2D or color Doppler." --> (3)
- Example 17: " He also reports crampy lower abdominal pain for the past several days which has been worsening." --> (3)
- Example 18: "No evidence of deep venous thrombosis in the bilaterallower 
extremity veins." --> (3)
"""

# Template for user input, including clinical notes to be analyzed
USER_CHAT_TEMPLATE = 'Analyze the following clinical notes to determine if the patient has metastasis status:\n{notes}<end_of_turn>\n'

# Template for model response based on the analysis of the clinical notes
MODEL_CHAT_TEMPLATE = '<start_of_turn> Model:\n'


import re

# Regular expression to match only valid stage responses
stage_pattern = re.compile(r'\((1|2|3)\)')

csv_out_path = '/results/tmp.csv'
tmp_csv_out_path = csv_out_path
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
                    device=device,temperature=0.1 ,# top_k=40, 
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
extracted_notes.to_csv(f"/results/p2/p1_results_0236_few_6shot_{RANDOM_SEED}.csv", index=False)

# %%
