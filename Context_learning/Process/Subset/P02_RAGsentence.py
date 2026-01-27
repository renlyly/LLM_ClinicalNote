#!/home/shilin/miniconda3/envs/tryrag/bin/python

# %%
import faiss
import sys
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# Keyword aliases for output naming
keyword_map = {
    "insulin used": "insulin",
    "coronary artery disease": "CAD",
    "chronic kidney disease": "CKD",
    "hypertension": "hypertension",
    "anemia": "anemia",
}

# Read keywords from CLI; default to metastasis
if len(sys.argv) > 1:
    keywords = sys.argv[1:]
else:
    keywords = ["metastasis"]

print(f"Keywords for extraction: {keywords}")
main_keyword = keywords[0]
keywords_str = keyword_map.get(main_keyword, main_keyword.replace(" ", "_"))

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def split_sentences(text: str):
    """Split text into sentences using simple regex rules (notes-friendly)."""
    sentences = re.split(r"(?:\n{2,}|(?<=[.!?])\s+)", text)
    return [s.strip() for s in sentences if s.strip()]


def extract_relevant_text(input_text, model, keywords, distance_threshold=1.2, top_k=5):
    """Retrieve semantically similar sentences to the given keywords via FAISS (L2)."""
    sentences = split_sentences(input_text)
    if not sentences:
        return []

    # Encode sentences and keywords
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True).cpu().numpy()
    keyword_embeddings = model.encode(keywords, convert_to_tensor=True).cpu().numpy()

    # Build FAISS index
    dimension = sentence_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(sentence_embeddings)

    # Search top-k per keyword and keep matches within threshold
    retrieved_sentences = set()
    for i, _ in enumerate(keywords):
        distances, indices = index.search(np.array([keyword_embeddings[i]]), top_k)
        for dist, idx in zip(distances[0], indices[0]):
            if dist < distance_threshold:
                retrieved_sentences.add(sentences[idx])

    return " ".join(retrieved_sentences)


# %%
if __name__ == "__main__":
    file_path = "subset_data/discharge_notes_p1.csv"
    df = pd.read_csv(file_path)

    # Copy dataframe for different top_k settings
    filtered_notes2 = df.copy()
    filtered_notes3 = df.copy()
    filtered_notes5 = df.copy()

    print("Processing filtered_notes2...")
    filtered_notes2["EXTRACTED_TEXT"] = [
        extract_relevant_text(x, model, keywords, top_k=2) if isinstance(x, str) else ""
        for x in tqdm(filtered_notes2["text"], desc="Processing filtered_notes2")
    ]
    output_path2 = f"/subset_data/filtered_2sen_discharge_notes_{keywords_str}.csv"
    filtered_notes2.to_csv(output_path2, index=False)

    print("Processing filtered_notes3...")
    filtered_notes3["EXTRACTED_TEXT"] = [
        extract_relevant_text(x, model, keywords, top_k=3) if isinstance(x, str) else ""
        for x in tqdm(filtered_notes3["text"], desc="Processing filtered_notes3")
    ]
    output_path3 = f"/subset_data/filtered_3sen_discharge_notes_{keywords_str}.csv"
    filtered_notes3.to_csv(output_path3, index=False)

    print("Processing filtered_notes5...")
    filtered_notes5["EXTRACTED_TEXT"] = [
        extract_relevant_text(x, model, keywords, top_k=6) if isinstance(x, str) else ""
        for x in tqdm(filtered_notes5["text"], desc="Processing filtered_notes5")
    ]
    output_path6 = f"/subset_data/filtered_6sen_discharge_notes_{keywords_str}.csv"
    filtered_notes5.to_csv(output_path6, index=False)

    print(f"\nProcessed data saved to: {output_path2}")
    print(f"Processed data saved to: {output_path3}")
    print(f"Processed data saved to: {output_path6}")

# %%
