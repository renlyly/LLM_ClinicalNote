# %%
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

# %%
save_directory = "model_classification"
tokenizer = AutoTokenizer.from_pretrained(save_directory)

# %%
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  
    bnb_4bit_use_double_quant=True,  
    bnb_4bit_quant_type="nf4",  
    bnb_4bit_compute_dtype=torch.bfloat16  
)

# Load model with quantization config
model = AutoModelForSequenceClassification.from_pretrained(
    save_directory,  
    num_labels=2,  
    quantization_config=bnb_config,  
    device_map={"": 0}  
)

# %%
def predict(input_text):
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda:1")  
    with torch.no_grad():
        outputs = model(**inputs).logits  
    y_prob = torch.sigmoid(outputs).tolist()[0]  
    return np.round(y_prob, 5)  

# %%
# Load dataset
extracted_notes = pd.read_csv('clinical_notes.csv')

# Update a column to track classification status
extracted_notes['LLM_class'] = 'Pending'

# Define output file paths
csv_out_path = 'tmp.csv'
tmp_csv_out_path = csv_out_path

# %%
# Processing notes and making predictions
for index, row in tqdm(extracted_notes.iterrows(), total=extracted_notes.shape[0], desc="Processing notes"):
    if row['LLM_class'] == 'Pending':
        clinical_notes = row['EXTRACTED_TEXT'].strip()
        prompt = clinical_notes

        predictions = predict(prompt)
        response_new = np.argmax(predictions)

        # Update the 'LLM_class' field with the prediction
        extracted_notes.at[index, 'LLM_class'] = response_new

        # Save progress every 3000 iterations
        if (index + 1) % 3000 == 0:
            print(f"Saving progress at index {index}")
            if os.path.exists(tmp_csv_out_path):
                os.remove(tmp_csv_out_path)
            tmp_csv_out_path = csv_out_path.replace('.csv', f'_{index}.csv')
            extracted_notes.iloc[:index].to_csv(tmp_csv_out_path, index=False)

# Final save
extracted_notes.to_csv("results.csv", index=False)

# %%
