# %%
import torch
import pandas as pd
from datasets import Dataset, load_from_disk
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    AutoModelForSequenceClassification, 
    Trainer,
    DataCollatorWithPadding
)

import bitsandbytes as bnb
import evaluate
import numpy as np

# %%
model_id  = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
print(f'Vocab size of the model {model_id}: {len(tokenizer.get_vocab())}')

# %%
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

# %%
# Load preprocessed dataset
dataset_final = load_from_disk('dataset_final')
tokenized_imdb = dataset_final.map(preprocess_function, batched=True)

# %%
id2label = {0: "No Metastasis", 1: "Metastasis"}
label2id = {"No Metastasis": 0, "Metastasis": 1}

# %%
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)  # Convert logits to predicted labels
    return metric.compute(predictions=predictions, references=labels)

# %%
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit quantization
    bnb_4bit_use_double_quant=True,  # Use double quantization
    bnb_4bit_quant_type="nf4",  # Quantization type
    bnb_4bit_compute_dtype=torch.bfloat16  # Compute dtype for efficiency
)

# %%
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=2,  # Binary classification
    id2label=id2label,
    label2id=label2id,
    quantization_config=bnb_config,
    device_map={"": 0}  # Single GPU device mapping
)

# %%
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# %%
def find_linear_names(model):
    """
    Identify linear layer names with 4-bit quantization.
    """
    cls = bnb.nn.Linear4bit  
    lora_module_names = set()

    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        # Special case: remove 'lm_head' if present
        if 'lm_head' in lora_module_names: 
            lora_module_names.remove('lm_head')
    
    return list(lora_module_names)

modules = find_linear_names(model)
print(modules)

# %%
lora_config = LoraConfig(
    r=64,  # Reduction factor
    lora_alpha=32,  # Adapter projection dimension
    target_modules=modules,  # Apply to these modules
    lora_dropout=0.05,  # Dropout rate
    bias="none",  # No bias
    task_type="SEQ_CLS"  # Sequence classification task
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# %%
training_args = TrainingArguments(
    output_dir="epoch_weights",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to="none",
    metric_for_best_model='eval_loss'
)

# %%
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

# %%
def predict(input_text):
    """
    Predict the metastasis label for a given text input.
    """
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")  # Move input to GPU
    with torch.no_grad():
        outputs = model(**inputs).logits  # Get model output
    y_prob = torch.sigmoid(outputs).tolist()[0]  # Apply sigmoid activation
    return np.round(y_prob, 5)  # Return rounded prediction

# %%
# Save the model and tokenizer
model.save_pretrained("model_classification")
tokenizer.save_pretrained("model_classification")

# %%
# Example prediction
predict("The patient has metastasis. He is in a critical condition.")
