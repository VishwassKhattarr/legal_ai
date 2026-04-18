import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("Loading chunks...")

df = pd.read_csv("data/chunks/chunks.csv")

texts = df["text"].tolist()

print(f"Total chunks: {len(texts)}")

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2",device="cpu")

print("Generating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)

embeddings = np.array(embeddings).astype("float32")

print("Building FAISS index...")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("Saving index...")

faiss.write_index(index, "faiss_index.bin")
np.save("embeddings.npy", embeddings)

print("DONE ✅")