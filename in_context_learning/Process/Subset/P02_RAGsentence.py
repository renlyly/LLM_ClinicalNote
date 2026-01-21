#! /home/shilin/miniconda3/envs/tryrag/bin/python

# %%
import faiss
import sys
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import re
from tqdm import tqdm

# 1. Define keywords
keyword_map = {
    "insulin used": "insulin",
    "coronary artery disease": "CAD",
    "chronic kidney disease": "CKD",
    "hypertension": "hypertension",
    "anemia": "anemia"
}

if len(sys.argv) > 1:
    keywords = sys.argv[1:]
else:
    keywords = ['metastasis']
print(f"Keywords for extraction: {keywords}")
main_keyword = keywords[0]
keywords_str = keyword_map.get(main_keyword, main_keyword.replace(" ", "_"))

# 2. Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def split_sentences(text):
    """
    Splits input text into meaningful sentences using regular expressions.
    Handles medical notes formatting with structured data.

    Parameters:
    - text (str): Input text to split.

    Returns:
    - List of sentences.
    """
    # Use regular expressions to split by newlines or punctuation markers
    sentences = re.split(r'(?:\n{2,}|(?<=[.!?])\s+)', text)
    sentences = [s.strip() for s in sentences if s.strip()]  # Remove empty and whitespace-only strings
    return sentences

# Define the function for extracting relevant text
def extract_relevant_text(input_text, model, keywords, distance_threshold=1.2, top_k=5):
    """
    Extract relevant sentences from the input text based on similarity to keywords.

    Parameters:
    - input_text (str): Input text as a single string.
    - model (SentenceTransformer): Pre-trained sentence embedding model.
    - keywords (list): List of keywords to search for relevance.
    - distance_threshold (float): Maximum allowed distance for relevance.
    - top_k (int): Number of top matches to consider for each keyword.

    Returns:
    - extracted_text (str): Combined relevant sentences.
    """
    # Step 1: Split input text into sentences
    sentences = split_sentences(input_text)
    if not sentences:
        return []    
    
    # Step 2: Generate sentence embeddings
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True).cpu().numpy()

    # Step 3: Generate keyword embeddings
    keyword_embeddings = model.encode(keywords, convert_to_tensor=True).cpu().numpy()

    # Step 4: Construct the FAISS index
    dimension = sentence_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(sentence_embeddings)

    # Step 5: Perform similarity search for each keyword
    retrieved_sentences = set()  # Use a set to store unique sentences
    for i, keyword in enumerate(keywords):
        # Perform similarity search
        distances, indices = index.search(np.array([keyword_embeddings[i]]), top_k)  # Top k matches
        
        # Collect results
        for dist, idx in zip(distances[0], indices[0]):
            if dist < distance_threshold:  # Only include results within the distance threshold
                sentence = sentences[idx]
                retrieved_sentences.add(sentence)

    # Step 6: Combine all unique sentences into one paragraph
    extracted_text = " ".join(retrieved_sentences)
    return extracted_text




# %%
# Example usage
if __name__ == "__main__":
    # 3. Load the dataset
    file_path = 'subset_data/discharge_notes_p1.csv'
    df = pd.read_csv(file_path)

    # 4. Create a copy of the dataset
    filtered_notes2 = df.copy()
    filtered_notes3 = df.copy()
    filtered_notes5 = df.copy()

    # 5. Apply the function to extract relevant text with progress bar
    print("Processing filtered_notes2...")
    filtered_notes2['EXTRACTED_TEXT'] = [
        extract_relevant_text(x, model, keywords, top_k=2) if isinstance(x, str) else ""
        for x in tqdm(filtered_notes2['text'], desc="Processing filtered_notes2")
    ]
    output_path2 = f'/subset_data/filtered_2sen_discharge_notes_{keywords_str}.csv'
    filtered_notes2.to_csv(output_path2, index=False)
    print("Processing filtered_notes3...")
    filtered_notes3['EXTRACTED_TEXT'] = [
        extract_relevant_text(x, model, keywords, top_k=3) if isinstance(x, str) else ""
        for x in tqdm(filtered_notes3['text'], desc="Processing filtered_notes3")
    ]
    output_path3 = f'/subset_data/filtered_3sen_discharge_notes_{keywords_str}.csv'
    filtered_notes3.to_csv(output_path3, index=False)
    print("Processing filtered_notes5...")
    filtered_notes5['EXTRACTED_TEXT'] = [
        extract_relevant_text(x, model, keywords, top_k=6) if isinstance(x, str) else ""
        for x in tqdm(filtered_notes5['text'], desc="Processing filtered_notes5")
    ]

    output_path6 = f'/subset_data/filtered_6sen_discharge_notes_{keywords_str}.csv'
    filtered_notes5.to_csv(output_path6, index=False)
    # 6. Save the updated dataset



    print(f"\nProcessed data saved to: {output_path2}")
    print(f"\nProcessed data saved to: {output_path3}")
    print(f"\nProcessed data saved to: {output_path6}")
# %%
