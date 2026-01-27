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
#!git clone https://github.com/google/gemma_pytorch.git
# %%
# next((df for df in df['text'] if 'no metastatic' in df.lower()), None)

# %%
VARIANT = '7b-it'  # version of model
if '1.1-' in VARIANT:
    VARIANT2 = VARIANT.replace('1.1-', '')  
else:
    VARIANT2 = VARIANT  
# VARIANT = '7b-it'  # version of model
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
keywords = [
    'metastasis',
    'metastatic',
    'metastasize',
    'metastasized',
    'metastasizing',
    'secondary tumor'
]

pattern = re.compile("|".join(keywords), re.IGNORECASE)  # Compiling regex, added re.IGNORECASE for case insensitivity

# Prepare tqdm to work with pandas
tqdm.pandas(desc="Filtering rows")

# Applying the regex to filter rows, ensure the column 'notes' is specified
extracted_notes = df[df['text'].progress_apply(lambda x: bool(pattern.search(x)) if pd.notna(x) else False)]

# %%
extracted_notes.index = pd.RangeIndex(len(extracted_notes.index))
extracted_notes['LLM_class'] = 'Pending'


# %%
def generate_response(note_text):

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
    MODEL_CHAT_TEMPLATE = '<start_of_turn> Model: \n'


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
                prompt,#temperature=0.05,
                device=device, temperature=0.1 , #top_k=60, 
                output_len=3
            )
            match = stage_pattern.search(response)
            if match:
                return match.group(1)  # Extract the matched number
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

csv_out_path = '/results/tmptt.csv'
tmp_csv_out_path = csv_out_path

for index, row in tqdm(extracted_notes.iterrows(), total=extracted_notes.shape[0], desc="Processing notes"):
    if row['LLM_class'] == 'Pending':
        clinical_notes = row['text'].strip()
        
        index_num = 0 if len(clinical_notes) < 12000 else len(clinical_notes)//12000 + 1
        matched_response = None
        while matched_response is None:
            split_num = 2 ** index_num
            matched_response = try_splitting_notes(clinical_notes, split_num)
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
# %%
extracted_notes.to_csv(f"/results/p2/p1_results_0246_nopreprocess_few_6shot_{RANDOM_SEED}.csv", index=False)
# %%
