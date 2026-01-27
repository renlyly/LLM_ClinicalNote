# %%
import pandas as pd
import numpy as np
# import spacy
from tqdm import tqdm
import re
# %%
print('Reading data...')
df = pd.read_csv('/subset_data/discharge_notes_p1.csv')

# %%

VARIANT = '7b-it' 
if '1.1-' in VARIANT:
    VARIANT2 = VARIANT.replace('1.1-', '') 
else:
    VARIANT2 = VARIANT  # 

MACHINE_TYPE = 'cuda:1'
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
import os

# sys.path.insert(0, '/home/shilin/temp/LLM_edit/code/')  
# %%
from gemma_pytorch.gemma.config import get_config_for_7b, get_config_for_2b
from gemma_pytorch.gemma.model import GemmaForCausalLM


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

disease_keywords = {
    "hypertension": ['hypertension', 'htn'],
    "insulin": ['insulin', 'insulin-dependent'],
}

target_disease = "insulin" 

keywords = disease_keywords[target_disease]
pattern = re.compile("|".join(keywords), re.IGNORECASE)

tqdm.pandas(desc=f"Filtering rows for {target_disease}")

extracted_notes = df[df['text'].progress_apply(lambda x: bool(pattern.search(x)) if pd.notna(x) else False)]

# %%
extracted_notes.index = pd.RangeIndex(len(extracted_notes.index))
extracted_notes['LLM_class'] = 'Pending'


# %%
# %%
def generate_response(note_text):

    TASK_DESCRIPTION = """<start_of_turn> user
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
    - Example 1: "The patient using insulin" --> (1)
    - Example 2: "The patient has no confirmed diabetes" --> (2)
    - Example 3: "She may need insulin if her renal function does not return to baseline." --> (3)

    """
    USER_CHAT_TEMPLATE = 'Analyze the following clinical notes to determine if the patient Long-term (current) use of insulin:\n{notes}.<end_of_turn>\n'

    # Template for model response based on the analysis of the clinical notes
    MODEL_CHAT_TEMPLATE = '<start_of_turn> Response:\n'


    prompt = (
        TASK_DESCRIPTION
        + USER_CHAT_TEMPLATE.format(notes=note_text)
        + MODEL_CHAT_TEMPLATE
    )
    stage_pattern = re.compile(r'\((1|2|3)\)')
    for attempt in range(5):  # Retry logic
        try:
            torch.cuda.empty_cache()
            response = model.generate(
                prompt,
                device=device,temperature=0.1 , #top_k=40, 
                output_len=3
            )
            
      
            match = stage_pattern.search(response)
            if match:
                return match.group(1)    
                
                
        except RuntimeError as e:
            if 'out of memory' in str(e):
                torch.cuda.empty_cache()
            else:
                raise e
            break
    if attempt == 4:
        return '3'
    else:
        return None  # None indicates no valid response generated

def try_splitting_notes(notes, split_num):
    # Check if the notes are too short to be meaningfully split
    if len(notes) < 50:
        return '3'  # Default code for insufficient information


    part_length = len(notes) // split_num
    responses = []
    
    # Split the notes and generate responses for each part
    for i in range(split_num):
        start_idx = i * part_length
        end_idx = start_idx + part_length if i < split_num - 1 else len(notes)  # Last segment takes the remainder
        part_notes = notes[start_idx:end_idx]
        # print("1")
        response = generate_response(part_notes)
        #print(f"response inside {response}")
        if response is None:
            return None  # Return None if any part fails to generate a response
        else:
            responses.append(response)
            
    # print(responses)
    responses_no_3 = [response for response in responses if response != '3']
    if responses_no_3:
        return '3' if responses_no_3.count("1") == responses_no_3.count("2") else ('1' if responses_no_3.count("1") > responses_no_3.count("2") else '2')
    else:
        return '3'
csv_out_path = '/results/tmp.csv'
tmp_csv_out_path = csv_out_path

for index, row in tqdm(extracted_notes.iterrows(), total=extracted_notes.shape[0], desc="Processing notes"):
    if row['LLM_class'] == 'Pending':
        clinical_notes = row['text'].strip()
        
        index_num = 0 if len(clinical_notes) < 16000 else len(clinical_notes)//16000 + 1
        matched_response = None
        while matched_response is None:
            split_num = 2 ** index_num
            matched_response = try_splitting_notes(clinical_notes, split_num)
            #print(f"matched_response at index {matched_response}")
            index_num += 1
        # print(split_num)
        # Update DataFrame
        extracted_notes.at[index, 'LLM_class'] = matched_response if matched_response else '3'

        # Periodically save progress
        if (index + 1) % 1000 == 0:
            print(f"Saving progress at index {index}")
            if os.path.exists(tmp_csv_out_path):
                os.remove(tmp_csv_out_path)
            tmp_csv_out_path = csv_out_path.replace('.csv', f'_{index}.csv')
            extracted_notes.iloc[:index].to_csv(tmp_csv_out_path, index=False)
extracted_notes.to_csv(f"/results/P1/A01_insulin_12_no_preprocess_p1_{RANDOM_SEED}.csv", index=False)
