import pandas as pd
import numpy as np
from tqdm import tqdm
import re

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
print('Reading data...')
df = pd.read_csv('/subset_data/discharge_notes_p1.csv')
disease_keywords = {
    "hypertension": ['hypertension', 'htn'],
    "insulin": ['insulin', 'insulin-dependent']
}

target_disease = "insulin"  
keywords = disease_keywords[target_disease]
pattern = re.compile("|".join(keywords), re.IGNORECASE)
tqdm.pandas(desc=f"Filtering rows for {target_disease}")
extracted_notes = df[df['text'].progress_apply(lambda x: bool(pattern.search(x)) if pd.notna(x) else False)]
extracted_notes.index = pd.RangeIndex(len(extracted_notes.index))
extracted_notes['LLM_class'] = 'Pending'



hf_model_name = "lianggq/llama-2-7b-chat-med" # modifty to your model path


# Load Hugging Face tokenizer and model
auto_tokenizer = AutoTokenizer.from_pretrained(hf_model_name, use_fast=True)
#quant_config = BitsAndBytesConfig(load_in_4bit=True)
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
device = auto_model.device
print(f"Model is loaded on: {auto_model.device}")


# Task description for model input
# %%

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

def generate_response(note_text):

    prompt = (
        TASK_DESCRIPTION
        + USER_CHAT_TEMPLATE.format(notes=note_text)
        + MODEL_CHAT_TEMPLATE
    )
    stage_pattern = re.compile(r'\((1|2|3)\)')
    
    for attempt in range(5):
        try:
            torch.cuda.empty_cache()
            inputs = auto_tokenizer(prompt, return_tensors="pt").to(auto_model.device)
            input_length = inputs["input_ids"].shape[1]
            outputs = auto_model.generate(**inputs, max_new_tokens=4,pad_token_id=auto_tokenizer.eos_token_id,do_sample=True,    
                                                    top_p=0.9  , 
                                                    temperature=0.1)
            response = auto_tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True)
            #print(response)
 
            match = stage_pattern.search(response)
            if match:
                return match.group(1)
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                torch.cuda.empty_cache()
            else:
                raise e
            break
    return '3'

# %%


def try_splitting_notes(notes, split_num):
    if len(notes) < 50:
        return '3'

    part_length = len(notes) // split_num
    responses = []

    for i in range(split_num):
        start_idx = i * part_length
        end_idx = start_idx + part_length if i < split_num - 1 else len(notes)
        part_notes = notes[start_idx:end_idx]
        response = generate_response(part_notes)
        if response is None:
            return None
        else:
            responses.append(response)

    responses_no_3 = [response for response in responses if response != '3']
    if responses_no_3:
        return '3' if responses_no_3.count("1") == responses_no_3.count("2") else ('1' if responses_no_3.count("1") > responses_no_3.count("2") else '2')
    else:
        return '3'

# %%

csv_out_path = '/temp/tmpi.csv'
tmp_csv_out_path = csv_out_path

for index, row in tqdm(extracted_notes.iterrows(), total=extracted_notes.shape[0], desc="Processing notes"):
    if row['LLM_class'] == 'Pending':
        clinical_notes = row['text'].strip()
        index_num = 0 if len(clinical_notes) < 12000 else len(clinical_notes) // 12000 + 1
        matched_response = None
        while matched_response is None:
            split_num = 2 ** index_num
            matched_response = try_splitting_notes(clinical_notes, split_num)
            index_num += 1
        extracted_notes.at[index, 'LLM_class'] = matched_response if matched_response else '3'

        if (index + 1) % 1000 == 0:
            print(f"Saving progress at index {index}")
            if os.path.exists(tmp_csv_out_path):
                os.remove(tmp_csv_out_path)
            tmp_csv_out_path = csv_out_path.replace('.csv', f'_{index}.csv')
            extracted_notes.iloc[:index].to_csv(tmp_csv_out_path, index=False)
            
extracted_notes.to_csv(f"/results/P1/A10_insulin_llama70_12_non_{RANDOM_SEED}.csv", index=False)


 


