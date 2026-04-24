import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────
TAGGED_PATH  = "data/chunks/chunks_tagged.csv"
INDEX_DIR    = "faiss_indexes"
BATCH_SIZE   = 64
MODEL_NAME   = "all-MiniLM-L6-v2"

# Roles we want separate indexes for
# (skip NONE and very small roles for now)
TARGET_ROLES = ["RPC", "RATIO", "FAC", "STA", "ISSUE",
                "PRE_RELIED", "RLC", "PREAMBLE", "ARG_PETITIONER"]

# ── Main ─────────────────────────────────────────────────────────────────────

def encode_in_batches(model, texts, batch_size=64):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i+batch_size]
        embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings).astype("float32")


def main():
    print("Loading tagged chunks...")
    df = pd.read_csv(TAGGED_PATH)
    print(f"Total chunks: {len(df)}")

    print("\nLoading model...")
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    print("Model loaded on GPU ✅")

    os.makedirs(INDEX_DIR, exist_ok=True)

    # ── Build one index per role ──────────────────────────────────────────────
    role_stats = []

    for role in TARGET_ROLES:
        role_df = df[df["role"] == role].reset_index(drop=True)
        count = len(role_df)

        if count == 0:
            print(f"\n⚠️  Skipping {role} — no chunks found")
            continue

        print(f"\n── Building index for role: {role} ({count} chunks) ──")

        texts = role_df["text"].tolist()
        embeddings = encode_in_batches(model, texts, BATCH_SIZE)

        # Build FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        # Save index
        index_path = os.path.join(INDEX_DIR, f"index_{role}.bin")
        faiss.write_index(index, index_path)

        # Save the corresponding chunk metadata
        meta_path = os.path.join(INDEX_DIR, f"meta_{role}.csv")
        role_df.to_csv(meta_path, index=False)

        print(f"✅ Saved: {index_path} ({count} vectors)")
        role_stats.append({"role": role, "chunks": count, "index": index_path})

    # ── Also build a FULL index (all chunks) as fallback ─────────────────────
    print(f"\n── Building FULL index ({len(df)} chunks) ──")
    all_texts = df["text"].tolist()
    all_embeddings = encode_in_batches(model, all_texts, BATCH_SIZE)

    dim = all_embeddings.shape[1]
    full_index = faiss.IndexFlatL2(dim)
    full_index.add(all_embeddings)
    faiss.write_index(full_index, os.path.join(INDEX_DIR, "index_ALL.bin"))
    df.to_csv(os.path.join(INDEX_DIR, "meta_ALL.csv"), index=False)
    print(f"✅ Saved full index")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n══════════════════════════════")
    print("        INDEX SUMMARY         ")
    print("══════════════════════════════")
    for s in role_stats:
        print(f"  {s['role']:<20} {s['chunks']:>5} chunks")
    print(f"  {'ALL (fallback)':<20} {len(df):>5} chunks")
    print("══════════════════════════════")
    print(f"\nAll indexes saved to: {INDEX_DIR}/")


if __name__ == "__main__":
    main()