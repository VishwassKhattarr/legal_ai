import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# Load data
df = pd.read_csv("data/chunks/chunks.csv")

# Models
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device="cpu")

# Load index
index = faiss.read_index("faiss_index.bin")


# 🔥 Query expansion
def expand_query(query):
    return query + " criminal law IPC bail anticipatory bail legal conditions court decision judgment"


# 🔥 Keyword filtering
def is_relevant(text, query):
    keywords = query.lower().split()
    text = text.lower()
    return any(word in text for word in keywords)


def search_and_rerank(query, top_k=5, initial_k=30):
    # Encode query
    query_embedding = bi_encoder.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # FAISS search
    distances, indices = index.search(query_embedding, initial_k)

    candidates = []
    meta = []

    for idx in indices[0]:
        row = df.iloc[idx]
        text = row["text"]

        # Apply filter
        if is_relevant(text, query):
            candidates.append(text)
            meta.append((row["doc_id"], row["doc_type"]))

    # Fallback if everything filtered out
    if len(candidates) == 0:
        for idx in indices[0]:
            row = df.iloc[idx]
            candidates.append(row["text"])
            meta.append((row["doc_id"], row["doc_type"]))

    # Prepare pairs
    pairs = [[query, doc] for doc in candidates]

    # Cross-encoder scoring
    scores = cross_encoder.predict(pairs)

    # Rank
    ranked = sorted(zip(candidates, scores, meta), key=lambda x: x[1], reverse=True)

    return ranked[:top_k]
def explain(text, query):
    explanation = []

    if "bail" in text.lower():
        explanation.append("Mentions bail")

    if "anticipatory bail" in text.lower():
        explanation.append("Discusses anticipatory bail")

    if "ipc" in text.lower():
        explanation.append("Refers to IPC section")

    if "condition" in text.lower() or "criteria" in text.lower():
        explanation.append("Explains legal conditions")

    return ", ".join(explanation)

# Run
query = input("Enter your legal query: ")
query = expand_query(query)

results = search_and_rerank(query)

print("\n🔥 Top Reranked Results:\n")
for i, (text, score, (doc_id, doc_type)) in enumerate(results):
    print(f"{i+1}. [{doc_type}] {doc_id} | Score: {score:.4f}")
    print(text[:300])
    
    reason = explain(text, query)
    print(f"Reason: {reason}")
    
    print("-" * 80)