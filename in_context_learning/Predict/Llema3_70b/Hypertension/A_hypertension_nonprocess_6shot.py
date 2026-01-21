
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import os
import sys
import torch
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# %%
print('Reading data...')
df = pd.read_csv('/subset_data/discharge_notes_p1.csv')


disease_keywords = {
    "hypertension": ['hypertension', 'htn'],
    "insulin": ['insulin', 'insulin-dependent'],
    "anemia": [
        'anemia', 'anaemia', 'low hemoglobin', 'hemoglobin', 'haemoglobin',
        'low hgb', 'low hct', 'low hematocrit'    ]
}

target_disease = "hypertension"  
keywords = disease_keywords[target_disease]
pattern = re.compile("|".join(keywords), re.IGNORECASE)
tqdm.pandas(desc=f"Filtering rows for {target_disease}")
extracted_notes = df[df['text'].progress_apply(lambda x: bool(pattern.search(x)) if pd.notna(x) else False)]

# %%
extracted_notes.index = pd.RangeIndex(len(extracted_notes.index))
extracted_notes['LLM_class'] = 'Pending'

hf_model_name = "Meta-Llama-3-70B-Instruct" # modifty to your model path

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
def generate_response(note_text):
    TASK_DESCRIPTION = """<start_of_turn>user
    Task: Classify the presence of hypertension status from patient clinical notes. Respond only with the following numerical codes based on the notes provided:
- (1) Yes: The notes explicitly confirm the patient has hypertension.
- (2) No: The notes explicitly confirm the patient does not have hypertension.
- (3) Unknown: The notes do not contain sufficient information to determine the patient's hypertension status.
    
    Instructions:
    1. Do not provide explanations or reasons for your classification.
    2. Use only the information in the notes for your classification.
    3. If the notes are ambiguous regarding hypertension, choose "(3) Unknown".
    4. Only provide a single numerical code as the response.

    Examples:
    - Example 1: "Past Medical History: - GERD - bronchitis - hypertension - ovarian cyst" --> (1)
    - Example 2: "CARDIAC RISK FACTORS: Dyslipidemia, Hypertension" --> (1)
    - Example 3: "HTN: Home antihypertensive medications were held as above." --> (1)
    - Example 4: "Moderate pulmonary hypertension." --> (1)
    - Example 5: "Compared with the prior study, the degree of pulmonary hypertension has decreased." --> (1)
    - Example 6: "SECONDARY DIAGNOSES: Hypertension" --> (1)
    - Example 7: "He has no history of hypertension, and this is likely secondary to cyclosporine." --> (2)
    - Example 8: "She was not hypertensive during her stay." --> (2)
    - Example 9: "No pulmonary hypertension or clinically-significant valvular disease seen." --> (2)
    - Example 10: "Family History: No coronary artery disease, no diabetes, no hypertension." --> (2)
    - Example 11: "Pt not on any dedicated hypertensives other than metoprolol on admission." --> (2)
    - Example 12: "No evidence of hypertension." --> (2)
    - Example 13: "There is borderline pulmonary artery systolic hypertension." --> (3)
    - Example 14: "Given her high blood pressures when she was admitted, she was started on 5 mg amlodipine per day."--> (3)
    - Example 15: "Borderline systolic hypertension." --> (3)
    - Example 16: "Chronic kidney disease, likely due to hypertension." --> (3)
    - Example 17: "Family History: Significant for breast cancer, high blood pressure."  --> (3)
    - Example 18: "Elevated BP noted, could be stress-induced."--> (3)
    """

    USER_CHAT_TEMPLATE = 'Analyze the following clinical notes to determine if the patient has hypertension status:\n{notes}.<end_of_turn>\n'
    MODEL_CHAT_TEMPLATE = '<start_of_turn> Response: \n'    
    prompt = (
        TASK_DESCRIPTION
        + USER_CHAT_TEMPLATE.format(notes=note_text)
        + MODEL_CHAT_TEMPLATE
    )
    stage_pattern = re.compile(r'\((1|2|3)\)')
    
    for attempt in range(3):
        try:
            torch.cuda.empty_cache()
            inputs = auto_tokenizer(prompt, return_tensors="pt").to(auto_model.device)
            input_length = inputs["input_ids"].shape[1]
            outputs = auto_model.generate(**inputs, max_new_tokens=4,pad_token_id=auto_tokenizer.eos_token_id,do_sample=True,    
                                                    # top_p=0.95 ,
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
csv_out_path = '/temp/tmph.csv'
tmp_csv_out_path = csv_out_path

for index, row in tqdm(extracted_notes.iterrows(), total=extracted_notes.shape[0], desc="Processing notes"):
    if row['LLM_class'] == 'Pending':
        clinical_notes = row['text'].strip()
        index_num = 0 if len(clinical_notes) < 11000 else len(clinical_notes) // 11000 + 1
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
extracted_notes.to_csv(f"/results/P1/A10_hypertension_llama70_32_non_{RANDOM_SEED}.csv", index=False)

# Clean up the final temporary file
if os.path.exists(tmp_csv_out_path):
    try:
        os.remove(tmp_csv_out_path)
        print(f"Removed temporary file: {tmp_csv_out_path}")
    except OSError as e:
        print(f"Error removing temporary file {tmp_csv_out_path}: {e}")
