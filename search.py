import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load data
df = pd.read_csv("data/chunks/chunks.csv")

# Load model (CPU)
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Load FAISS index
index = faiss.read_index("faiss_index.bin")

def search(query, top_k=5):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        row = df.iloc[idx]
        results.append({
            "text": row["text"],
            "doc_type": row["doc_type"],
            "doc_id": row["doc_id"]
        })

    return results


# Run
query = input("Enter your legal query: ")

results = search(query)

print("\nTop Results:\n")
for i, res in enumerate(results):
    print(f"{i+1}. [{res['doc_type']}] {res['doc_id']}")
    print(res["text"][:300], "\n")